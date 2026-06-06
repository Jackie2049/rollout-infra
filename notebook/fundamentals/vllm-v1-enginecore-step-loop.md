# vLLM V1 EngineCore Step Loop 深度阅读

> 2026-06-07 | vllm/v1/engine/core.py (~2214 行)

## 核心架构

### 双进程架构: EngineCoreProc

vLLM V1 使用**双进程架构** (AsyncLLM + EngineCoreProc):

```
┌──────────────────────────────────────────────────┐
│  API Server Process (AsyncLLM)                    │
│  ├── asyncio event loop (request handling)        │
│  ├── EngineCoreClient (ZMQ sender)               │
│  └── output_handler (background Task, ZMQ recv)  │
└──────────────────┬───────────────────────────────┘
                   │ ZMQ IPC (input/output sockets)
┌──────────────────▼───────────────────────────────┐
│  EngineCore Process (EngineCoreProc)              │
│  ├── Main Thread: run_busy_loop()                │
│  │   ├── _process_input_queue()                  │
│  │   └── _process_engine_step() → step_fn()      │
│  ├── Input Thread: process_input_sockets()        │
│  │   └── ZMQ → input_queue → preprocess_add_req │
│  └── Output Thread: process_output_sockets()      │
│      └── output_queue → ZMQ → API server         │
└──────────────────────────────────────────────────┘
```

**关键设计**:
- Input/Output 线程释放 GIL (ZMQ 是 C 库) → 与 GPU 计算重叠
- `input_queue` (Python Queue) 是主线程与 IO 线程的桥梁
- `output_queue` 同理, Output 线程异步序列化并发送
- `preprocess_add_request()` 在 input 线程执行 → grammar 初始化与 model forward 并行

### 三线程分工

| Thread | 职责 | ZMQ 角色 |
|--------|------|----------|
| Main | busy loop + step + schedule | 不直接用 ZMQ |
| Input | ZMQ DEALER recv → input_queue | 接收 ADD/ABORT/UTILITY |
| Output | output_queue → ZMQ PUSH send | 发送 EngineCoreOutputs |

## EngineCore.__init__ 初始化链

```python
def __init__(self, vllm_config, executor_class, ...):
    # 1. 创建 model executor
    self.model_executor = executor_class(vllm_config)

    # 2. 初始化 KV cache (determine_available_memory → auto-fit)
    scheduler_kv_cache_config = self._initialize_kv_caches(vllm_config)

    # 3. 创建 scheduler
    self.scheduler = Scheduler(
        scheduler_config, scheduler_kv_cache_config, ...

    # 4. 选择 step 函数
    self.step_fn = self.step if self.batch_queue is None \
        else self.step_with_batch_queue

    # 5. gc.freeze() — 启动 heap 标记为静态, 减少 GC pause
    freeze_gc_heap()
```

**关键**: `_initialize_kv_caches()` 包含 auto-fit binary search — 如果 max_model_len 因显存不足被降低, 会 `collective_rpc("update_max_model_len")` 同步到所有 worker.

## step() — 核心迭代

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 1. 调度: 决定哪些请求进入这一步
    scheduler_output = self.scheduler.schedule()

    # 2. 模型前向 (非阻塞)
    model_output = self.model_executor.execute_model(
        scheduler_output, non_block=True)

    # 3. 结构化输出 bitmask (grammar 约束)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 采样
    model_output = self.model_executor.sample_tokens(grammar_output)

    # 5. 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)

    return engine_core_outputs, model_executed
```

**5 步流水线**: `schedule → execute → grammar → sample → update`

### non_block=True 的含义

`execute_model(non_block=True)` 返回 Future, 允许:
- grammar bitmask 计算与 GPU 计算重叠
- CPU 可以在 GPU 执行期间处理 abort

但 `sample_tokens()` 必须等 execute 完成 (需要 logits).

## step_with_batch_queue() — Pipeline Parallelism 变体

```python
def step_with_batch_queue(self):
    # 1. 从 batch_queue 取上一步的 deferred 输出
    if batch_queue:
        future, sched_out, exec_future = batch_queue.popleft()
        model_output = future.result()  # 等待上一步 sample 完成

    # 2. 当前步的调度
    scheduler_output = self.scheduler.schedule()

    # 3. 当前步的模型执行 (defer grammar bitmask)
    exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 4. 处理上一步的 grammar + sample
    grammar_output = self.scheduler.get_grammar_bitmask(prev_sched_out)
    sample_future = self.model_executor.sample_tokens(grammar_output, non_block=True)

    # 5. 入队 (deferred) — 当前步的 grammar 会在下一步处理
    batch_queue.appendleft((sample_future, scheduler_output, exec_future))

    # 6. 更新上一步的输出
    engine_core_outputs = self.scheduler.update_from_output(prev_sched_out, model_output)
```

**核心思想**: Grammar bitmask 计算被推迟到下一步, 形成流水线:
- Step N: execute model N
- Step N+1: grammar N → sample N → execute model N+1
- Step N+2: grammar N+1 → sample N+1 → execute model N+2

这样 grammar+sample (CPU) 与 execute (GPU) 可以重叠!

### Speculative Decoding 与 Batch Queue

```python
if self.use_spec_decode:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    self.scheduler.update_draft_token_ids_in_output(
        draft_token_ids, deferred_scheduler_output)
```

Spec decode 需要 draft token ids 来计算 grammar bitmask → 必须从 executor 拿上一步的 draft tokens.

## run_busy_loop() — 进程级主循环

```python
def run_busy_loop(self):
    while self._handle_shutdown():  # 检查 shutdown 状态
        self._process_input_queue()  # 1. 处理输入直到有工作
        self._process_engine_step()  # 2. 执行一步并输出

    raise SystemExit
```

### _process_input_queue(): 等待工作

```python
def _process_input_queue(self):
    while not self.has_work() and self.is_running():
        # idle callbacks (sleep/wake 协调)
        self._notify_idle_state_callbacks()
        # 阻塞等待 input_queue
        req = self.input_queue.get(block=True)
        self._handle_client_request(*req)

    # 非阻塞处理剩余请求
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)
```

**has_work() 判断**: `engines_running OR scheduler.has_requests() OR batch_queue非空`

### _process_engine_step(): 执行+输出

```python
def _process_engine_step(self):
    outputs, model_executed = self.step_fn()
    # 发送输出到 output_queue → output_thread → ZMQ → API server
    for output in outputs.items():
        self.output_queue.put_nowait(output)
    self.post_step(model_executed)

    # 0-token 步 (无 model 执行) → sleep 1ms 让 KV transfer 线程推进
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)
```

## EngineCoreProc 请求处理

### _handle_client_request(): 请求分发

| RequestType | 处理 |
|-------------|------|
| ADD | `preprocess_add_request()` → `add_request()` |
| ABORT | `abort_requests()` + `aborts_queue` (双队列, 保证顺序) |
| UTILITY | getattr + invoke, 结果入 output_queue |
| WAKEUP | 仅唤醒 busy loop (sleep/wake) |
| EXECUTOR_FAILED | raise RuntimeError |

**双队列设计**: ABORT 同时入 `input_queue` 和 `aborts_queue`:
- `aborts_queue`: 在 `_process_engine_step()` 的 `_process_aborts_queue()` 中急切处理 (模型执行期间)
- `input_queue`: 正常顺序处理 (避免泄漏请求)

## Shutdown 协议

```
RUNNING → REQUESTED → SHUTTING_DOWN → SystemExit
```

1. **SIGTERM/SIGINT**: 触发 `shutdown_state = REQUESTED`
2. **REQUESTED 状态**:
   - `shutdown_timeout=0`: 立即 abort 所有请求
   - `timeout>0`: drain 排空在途请求
3. **SHUTTING_DOWN**: 继续 stepping 直到 `has_work()=False`
4. **完成**: `_handle_shutdown()=False` → `raise SystemExit`

## DPEngineCoreProc — MoE Data Parallel

```python
class DPEngineCoreProc(EngineCoreProc):
    """Only for MoE models with DP>1"""
    self.step_counter = 0       # 步数计数
    self.current_wave = 0       # DP wave 编号
    self.pending_pause = False   # 两阶段暂停
```

**两阶段暂停协议**:
1. Phase 1: 设置 `pending_pause=True`, kick-start engines (设 `engines_running=True`)
2. Phase 2: `_has_global_unfinished_reqs()` 中 AllReduce 确认所有 rank 都 pending → 集体停止

**MoE DP 不同于 Dense DP**:
- Dense: 每个 DP rank 独立实例 (DP=1 per rank)
- MoE: 需要 All-to-All 通信 → DPEngineCoreProc 协调

## Sleep/Wake 机制

```python
def sleep(self, level=1, mode="abort"):
    # Level 0: 仅暂停调度, 无 GPU 内存变化
    # Level 1: offload weights to CPU, discard KV cache
    # Level 2: discard all GPU memory
    self.pause_scheduler(mode, clear_cache=(level>=1))
    self.model_executor.sleep(level)

def wake_up(self, tags=None):
    # tags=["scheduling"]: Level 0 wake (只恢复调度)
    # tags=None or other: model_executor.wake_up() + resume_scheduler
```

**Sleep 的意义**: GPU 可以被释放给其他任务 (如训练), 需要时快速 wake up.

## 关键设计决策总结

| 设计 | 原因 |
|------|------|
| 双进程 (AsyncLLM + EngineCoreProc) | asyncio 与 busy loop 不兼容; ZMQ IPC 低延迟 |
| 三线程 (main + input + output) | ZMQ 释放 GIL → IO 与计算重叠 |
| gc.freeze() | 启动 heap 静态化 → 减少 GC pause (避免 decode jitter) |
| non_block=True execute | grammar bitmask 与 GPU 计算重叠 |
| batch_queue (PP) | grammar+sample 推迟 → 流水线重叠 |
| 双队列 abort | 急切处理 + 顺序保证 |
| auto-fit binary search | 显存不够时自动降 max_model_len |
| engines_running flag | sleep/wake 时需要手动标记 "有工作" |
| 0-token 步 sleep(1ms) | 让 KV transfer 后台线程推进 |

## 与 SGLang 对比

| | vLLM V1 | SGLang |
|---|---------|--------|
| 架构 | 双进程 ZMQ | 单进程 asyncio + threading |
| 主循环 | busy loop (阻塞) | 事件循环 (asyncio) |
| IO 重叠 | 三线程 ZMQ | asyncio + ZMQ |
| PP 流水线 | batch_queue | 无内置 PP |
| Sleep/Wake | 3 级 (0/1/2) | 无 |
| DP | DPEngineCoreProc (MoE) | 无内置 DP |
| GC 控制 | freeze/unfreeze | 无特殊处理 |

**vLLM 更重** (适合多 GPU 生产环境), **SGLang 更轻** (单机快速迭代).