# vLLM V1 Engine Core Source Reading — Dual-Process Architecture + ZMQ IPC + Busy Loop + Sleep/Wake + Batch Queue + Data Parallel

> 2026-06-12 | vLLM V1 Engine Core全链路源码分析: EngineCore(inner loop)+EngineCoreProc(ZMQ wrapper)+MPClient(frontend)+InprocClient(single-proc)+DPEngineCoreProc(MoE DP)+run_busy_loop+Sleep/Wake 3级+Batch Queue+Handshake
> 源码: vllm/v1/engine/core.py (2239行), vllm/v1/engine/core_client.py (1743行)
> 关联: vllm-v1-gpu-model-runner-source-reading.md, vllm-v1-scheduler-architecture-source-reading.md

## 0. 核心定律: 双进程架构 + ZMQ IPC + Busy Loop + 三线程

```
vLLM V1 Engine架构:

  Frontend Process (API Server / MPClient / InprocClient)
    → 接收HTTP请求 → add_request → ZMQ ROUTER发送
    → ZMQ PULL接收 → EngineCoreOutput → 返回用户

  Backend Process (EngineCoreProc)
    → ZMQ DEALER接收 → input_queue → _handle_client_request
    → run_busy_loop → _process_input_queue → _process_engine_step
    → output_queue → ZMQ PUSH发送 → Frontend接收

  三线程(EngineCoreProc内):
    Thread 1: input_thread → ZMQ DEALER→input_queue (Socket→Queue)
    Thread 2: output_thread → output_queue→ZMQ PUSH (Queue→Socket)
    Thread 3: Main → busy_loop (Queue→step→Queue)

关键设计:
  → 双进程: Frontend处理HTTP+序列化 / Backend处理GPU计算 → 互不阻塞!
  → ZMQ IPC: ROUTER→DEALER / PUSH→PULL → 零拷贝(msgspec+msgpack) → 异步通信
  → 三线程: IO线程释放GIL → Main线程专注于GPU调度 → overlap IO和compute
  → Busy Loop: while has_work → process_input+step → sleep(0.001) if idle → 无轮询浪费
  → Batch Queue: PP用 → 多batch异步重叠 → 消除pipeline气泡!
  → Sleep/Wake 3级: RUNNING→REQUESTED→SHUTTING_DOWN → drain或abort → 安全关闭
```

## 1. EngineCore — Inner Loop核心

```
文件: vllm/v1/engine/core.py

class EngineCore (内循环, 非IPC):

__init__核心组件:
  → model_executor: Executor → GPU执行器(Uni/MP/TP/PP)
  → scheduler: SchedulerInterface → 调度器(FCFS/Priority)
  → structured_output_manager: StructuredOutputManager → FSM grammar
  → mm_receiver_cache → 多模态预处理器缓存
  → batch_queue: deque → PP batch队列(max_concurrent_batches>1时)
  → use_spec_decode → 投机解码标志
  → async_scheduling → 异步调度标志
  → step_fn → step() or step_with_batch_queue() → 根据batch_queue选择

KV Cache初始化:
  → _initialize_kv_caches() → model_executor.get_kv_cache_specs()
  → → profiling → 确定可用GPU内存 → 计算block数量
  → → warmup → 适配max_model_len → 创建KVCacheConfig
  → → register_all_kvcache_specs → 注册所有spec类型

step()核心流程(~30行):
  → if not scheduler.has_requests() → return {}, False (无请求→跳过)
  → scheduler_output = scheduler.schedule() → 调度
  → future = model_executor.execute_model(scheduler_output, non_block=True) → 非阻塞!
  → grammar_output = scheduler.get_grammar_bitmask(scheduler_output) → FSM bitmask
  → model_output = future.result() → 等待GPU执行完成
  → if model_output is None → model_executor.sample_tokens(grammar_output) → 两阶段!
  → scheduler.update_from_output(scheduler_output, model_output) → 更新状态
  → return engine_core_outputs, model_executed

post_step():
  → async_scheduling=False + spec_decode → take_draft_token_ids() → 更新scheduler
  → → async_scheduling=True → 不更新(worker进程直接处理)
```

## 2. EngineCoreProc — ZMQ IPC Wrapper

```
class EngineCoreProc(EngineCore):

双进程架构:
  → Frontend(MPClient) → ZMQ ROUTER(input_socket) + ZMQ PULL(output_socket)
  → Backend(EngineCoreProc) → ZMQ DEALER(input) + ZMQ PUSH(output)

__init__:
  → input_queue: queue.Queue → Python线程安全队列(Socket→Queue)
  → output_queue: queue.Queue → Python线程安全队列(Queue→Socket)
  → shutdown_state: EngineShutdownState → RUNNING/REQUESTED/SHUTTING_DOWN
  → tensor_ipc_receiver: TensorIpcReceiver → torch_shm IPC for多模态tensor

  → _perform_handshakes → ZMQ DEALER→握手→获取addresses→DEALER→ROUTER
  → → 获取: input/output ZMQ addresses → DP coordinator addresses
  → → 两种握手: local(DP=1) / remote(DP>1 with coordinator)

  → 启动三线程:
    Thread 1: input_thread → process_input_sockets → ZMQ→input_queue
    Thread 2: output_thread → process_output_sockets → output_queue→ZMQ
    Thread 3: Main → run_busy_loop → 核心!

关键buffer:
  → input_queue: 每个请求(EngineCoreRequestType, payload)
  → output_queue: 每个输出(dp_rank, EngineCoreOutputs)
  → aborts_queue: 中止请求队列 → 与input_queue并行消费
  → pending_messages: ZMQ MessageTracker → 防止过早释放序列化buffer
```

## 3. run_busy_loop — 核心循环

```
run_busy_loop()流程:

while _handle_shutdown():  → 检查shutdown状态→决定继续或退出

  1. _process_input_queue() → 处理所有输入请求
     → while not has_work() and is_running():
       → _notify_idle_state_callbacks() → 通知idle状态(用于sleep/wake)
       → input_queue.get(block=True) → 阻塞等待新请求
       → _handle_client_request(*req) → 处理请求
     → while not input_queue.empty(): → 批量处理剩余请求

  2. _process_engine_step() → 执行一步
     → outputs, model_executed = step_fn() → schedule+execute+output
     → for output in outputs.items():
       → output_queue.put_nowait(output) → 非阻塞放入output queue
     → post_step(model_executed) → 处理spec decode draft tokens
     → if not model_executed and scheduler.has_requests():
       → time.sleep(0.001) → yield GIL→让KV connector线程运行

  → raise SystemExit → shutdown完成

关键设计:
  → Busy Loop不poll GPU → 只通过input_queue等待 → 空闲时CPU不浪费
  → → 但GPU空闲时 → EngineCore进程占用一个CPU核心 → 可能影响其他进程
  → → sleep(0.001) → 1ms → 让background KV transfer线程运行→P/D分离需要
  → → idle callbacks → _notify_idle_state_callbacks → Sleep/Wake机制用!
```

## 4. _handle_client_request — 请求处理分发

```
EngineCoreRequestType枚举:
  → ADD → 添加新请求 → scheduler.add_request()
  → ABORT → 中止请求 → scheduler.finish_requests(FINISHED_ABORTED)
  → WAKEUP → 唤醒引擎 → 不处理→只激活busy_loop
  → PAUSE → 暂停引擎 → PauseMode(New/All) → scheduler.pause()
  → RESUME → 恢复引擎 → scheduler.resume() → engines_running=True
  → EXECUTOR_FAILED → Executor失败 → shutdown
  → RECONFIGURE_DISTRIBUTED → Elastic EP重配置 → 动态扩缩
  → UTILITY → utility操作(profile/health/etc)

PAUSE/RESUME 3级:
  → PAUSE_NEW → 不调度新请求(但继续running请求)
  → PAUSE_ALL → 不调度任何请求(完全暂停) → Sleep模式
  → RESUME → 恢复调度 → Wake模式

Sleep/Wake用于vLLM+verl colocation:
  → verl训练时 → vLLM sleep(PAUSE_ALL) → 释放GPU内存给训练
  → verl推理时 → vLLM wake(RESUME) → 重新占用GPU → 推理
  → → 交替: 训练→sleep→wake→推理→sleep→训练... → 单GPU交替使用!
```

## 5. MPClient — Frontend多进程客户端

```
class MPClient(EngineCoreClient):

__init__:
  → ZMQ setup:
    → sync_ctx = zmq.Context(io_threads=2) → 2个IO线程
    → asyncio_mode → zmq.asyncio.Context → AsyncLLM用
    → input_socket: zmq.ROUTER → 接收engine requests → bind=True
    → output_socket: zmq.PULL → 接收engine outputs → bind=True
    → router_handover → Elastic EP用 → 允许新engine替换dead connection

  → launch_core_engines() → 启动后台EngineCore进程
    → EngineManager → MultiprocessExecutor → 每个GPU一个进程
    → coordinator → DPCoordinator(DP>1时) → DP协调器
    → tensor_queue → torch_shm IPC → 多模态tensor共享

  → Serialization:
    → encoder: MsgpackEncoder → msgpack序列化 → oob_tensor_consumer
    → decoder: MsgpackDecoder → msgpack反序列化 → EngineCoreOutputs
    → → msgspec struct → 高效序列化 → 比 pickle快10x+!

  → DP Setup:
    → core_engines: list[EngineIdentity] → 每个engine的ZMQ identity
    → → identity = rank.to_bytes(2, "little") → 2字节身份标识
    → engine_ranks_managed → 此client管理的DP ranks

  → Monitor Thread:
    → monitor_engine_cores() → 监控engine进程liveness
    → → 如果engine进程意外退出 → engine_dead=True → shutdown client

  → Ready Wait:
    → 同步等待所有engine发送READY消息 → timeout=VLLM_ENGINE_READY_TIMEOUT_S
    → → 大模型权重加载慢 → 需较长timeout → 默认300s
```

## 6. InprocClient — 单进程客户端

```
class InprocClient(EngineCoreClient):

单进程模式 → EngineCore直接在API server进程内运行!
  → 不需要ZMQ IPC → 不需要子进程 → 最简架构
  → engine_core = EngineCore(...) → 直接创建
  → step() → 直接调用 → 无IPC延迟
  → 但! GPU计算阻塞API server → 不能overlap HTTP处理和GPU

适用场景:
  → DP=1 → 单GPU推理 → 最简配置
  → → RTX 4090单GPU → InprocClient最合适! → 无IPC overhead
  → → 但Inproc实际上用MPClient(默认) → ZMQ IPC overhead <1ms → 可接受

关键方法:
  → add_request() → scheduler.add_request() → 直接调用
  → abort() → scheduler.finish_requests() → 直接调用
  → step() → engine_core.step() → 直接调用
  → get_output() → scheduler.update_from_output() → 直接调用
```

## 7. DPEngineCoreProc — MoE数据并行引擎

```
class DPEngineCoreProc(EngineCoreProc):

MoE专用 → DP>1 → EP(Expert Parallelism)

关键属性:
  → step_counter → 步计数器 → 每N步同步finished状态
  → current_wave → DP wave编号 → 用于协调暂停/恢复
  → pending_pause → 是否在等待所有DP rank暂停
  → ignore_start_dp_wave → 忽略过期的START_DP_WAVE消息
  → eep_scaling_state → Elastic EP动态扩缩状态

两阶段暂停协议:
  → Phase 1: pending_pause=True → 继续stepping(空batch) → 等AllReduce确认
  → Phase 2: 所有rank都pending → 暂停 → ignore_start_dp_wave防止过期唤醒
  → → 防止: 只有部分rank暂停 → 不一致 → 死锁!

DP协调:
  → AllReduce确认finished → 所有rank知道哪些请求已完成
  → → 7B MoE: EP不需要(单GPU足够) → 671B DeepSeek-V3: EP必需!

_init_data_parallel:
  → 配置stateless process group → DP通信
  → → NCCL → 128-byte store → 同步屏障 → 启动/停止协调
```

## 8. Batch Queue — Pipeline Parallelism异步重叠

```
step_with_batch_queue()流程(~100行):

目的: PP需要多batch同时执行 → 消除pipeline气泡 → overlap通信

流程:
  1. 如果batch_queue不满 → schedule新batch → execute_model(non_block=True)
     → → 不等待GPU完成! → 继续调度下一个batch
  2. 如果pending_structured_output_tokens → defer sampling
     → → grammar bitmask需要等待前一步完成 → 延迟采样
  3. 添加(future, scheduler_output, exec_future)到batch_queue
  4. 如果batch_queue满或无新请求 → 等待第一个batch完成
     → future.result() → 阻塞等待GPU → 获取model_output
  5. sample_tokens(grammar_output) → 如果不是deferred
  6. scheduler.update_from_output() → 更新状态
  7. 返回outputs → 清理batch_queue

关键设计:
  → batch_queue_size = max_concurrent_batches → PP并行度
  → → PP=4 → max_concurrent_batches=2 → 两个batch交错执行
  → → → 消除1F1B气泡 → 利用pipeline空闲时间调度
  → → RTX 4090: PP不适用(单GPU) → batch_queue=None → 用step()
```

## 9. RTX 4090 Engine Core Implications

```
1. InprocClient vs MPClient:
  → RTX 4090单GPU → InprocClient最简 → 无IPC overhead
  → → 但vLLM默认用MPClient → ZMQ IPC→额外进程→1个CPU核心
  → → ZMQ IPC延迟<1ms → 对RTX 4090推理 negligible
  → → 建议: DP=1 + PP=1 → InprocClient → 省一个进程+CPU核心

2. Busy Loop vs Event-driven:
  → Busy Loop → 占用CPU核心 → RTX 4090单GPU → 可能影响verl训练
  → → sleep/wake → verl colocation → 交替训练和推理 → 不冲突!
  → → → 训练时vLLM sleep(PAUSE_ALL) → 释放GPU → GPU全给训练
  → → → 推理时vLLM wake(RESUME) → 重新占用GPU → 推理

3. Batch Queue:
  → RTX 4090单GPU → batch_queue=None → 用step() → 无batch overlap
  → → PP不可行 → 无pipeline → 不需要batch queue

4. Sleep/Wake对verl:
  → verl GRPO → vLLM推理 + FSDP训练 → 交替使用GPU
  → → Sleep: 释放GPU内存 → vLLM模型权重卸载到CPU → GPU空出给训练
  → → Wake: 重新加载模型权重 → GPU给推理 → 训练暂停
  → → → 实测: sleep/wake开销几秒 → 需要足够推理步数才值得

5. ZMQ IPC overhead:
  → msgpack序列化 → SchedulerOutput → ~KB级 → 序列化<1ms
  → → EngineCoreOutput → 包含sampled_token_ids → 也KB级
  → → → ZMQ IPC总延迟≈1ms → vs GPU step≈10ms → <10% → 可接受
  → → → 但! 多模态 → mm_features大 → torch_shm IPC避免序列化

6. Grammar Bitmask两阶段:
  → execute_model → forward → 保存execute_model_state → 返回None/IntermediateTensors
  → → sample_tokens → grammar bitmask → sampling → 返回ModelRunnerOutput
  → → → RTX 4090: grammar bitmask overhead<1% → 几乎free → 两阶段必要!

7. 生产最优:
  → RTX 4090单GPU → InprocClient → step() → 无ZMQ IPC
  → → verl colocation → Sleep/Wake交替 → 释放GPU给训练
  → → 7B INT4+INT8KV+EAGLE → BF16推理 → grammar bitmask → <1% overhead
  → → DP=1 → 无DPEngineCoreProc → 无coordinator → 最简架构
```

## 参考文献

```
1. vLLM V1 Engine Core源码:
   - vllm/v1/engine/core.py (2239行) — EngineCore + EngineCoreProc + DPEngineCoreProc
   - vllm/v1/engine/core_client.py (1743行) — MPClient + InprocClient
   - vllm/v1/engine/__init__.py — EngineCoreRequestType + outputs

2. ZMQ: ZeroMQ — ROUTER/DEALER/PUSH/PULL pattern
3. msgspec: 高性能Python序列化 → struct+msgpack
4. Sleep/Wake: vLLM + verl colocation design

我们的笔记:
- vllm-v1-gpu-model-runner-source-reading.md — GPU Model Runner
- vllm-v1-scheduler-architecture-source-reading.md — Scheduler