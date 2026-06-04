# vLLM V1 Engine Core 源码阅读

> EngineCore: vLLM 的心跳 — 调度、执行、输出的核心循环

## 1. 架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│                    EngineCore 继承体系                             │
│                                                                  │
│  EngineCore (base)                                               │
│    ├── step()                    ← 核心循环: 调度→执行→输出       │
│    ├── _initialize_kv_caches()   ← GPU 显存分析+KV Cache 分配    │
│    ├── add_request()             ← 请求注册到 Scheduler           │
│    └── abort_requests()          ← 请求中止                      │
│                                                                  │
│  EngineCoreProc (EngineCore)                                     │
│    ├── ZMQ 通信                  ← input_thread + output_thread  │
│    ├── run_busy_loop()           ← 后台进程主循环                │
│    ├── process_input_sockets()   ← ZMQ DEALER 接收请求           │
│    └── process_output_sockets()  ← ZMQ PUSH 发送输出            │
│                                                                  │
│  DPEngineCoreProc (EngineCoreProc)                               │
│    ├── AllReduce DP 同步         ← 每 N 步检查全局完成状态       │
│    ├── Wave 协调                 ← 请求波次管理                  │
│    └── 两阶段暂停协议            ← DP 感知的 pause 机制          │
│                                                                  │
│  EngineCoreActor (EngineCoreActorMixin + EngineCoreProc)          │
│    └── Ray Actor 模式            ← 分布式部署                    │
└──────────────────────────────────────────────────────────────────┘
```

## 2. 核心文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `vllm/v1/engine/core.py` | ~2214 | EngineCore 主类 + EngineCoreProc + DPEngineCoreProc |
| `vllm/v1/engine/core_client.py` | ~1500 | 客户端: InprocClient / SyncMPClient / AsyncMPClient / DPAsyncMPClient |
| `vllm/v1/engine/__init__.py` | 数据类型 | EngineCoreRequest/Output/Outputs 定义 |
| `vllm/v1/engine/async_llm.py` | 前端 | AsyncLLM 使用 AsyncMPClient |

## 3. EngineCore 初始化

```python
class EngineCore:
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        # 1. 加载插件
        load_general_plugins()

        # 2. 创建模型执行器 (Executor)
        self.model_executor = executor_class(vllm_config)

        # 3. KV Cache 初始化 (显存 profiling → 分配)
        kv_cache_config = self._initialize_kv_caches(vllm_config)

        # 4. 创建结构化输出管理器
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # 5. 创建调度器
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        self.scheduler = Scheduler(
            vllm_config, kv_cache_config,
            structured_output_manager, ...
        )

        # 6. KV Connector 握手 (P/D 分离)
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector:
            metadata = self.model_executor.get_kv_connector_handshake_metadata()
            kv_connector.set_xfer_handshake_metadata(metadata)

        # 7. Batch Queue (Pipeline Parallelism)
        if max_concurrent_batches > 1:
            self.batch_queue = deque(maxlen=max_concurrent_batches)
            self.step_fn = self.step_with_batch_queue
        else:
            self.step_fn = self.step

        # 8. Prefix Caching 哈希函数
        if enable_prefix_caching:
            self.request_block_hasher = get_request_block_hasher(...)

        # 9. GC 优化: 冻结启动堆 + 环境变量缓存
        freeze_gc_heap()
        enable_envs_cache()
```

### 3.1 KV Cache 初始化流程

```python
def _initialize_kv_caches(self, vllm_config):
    # 1. 注册所有 KV Cache spec
    register_all_kvcache_specs(vllm_config)

    # 2. 获取模型需要的 KV Cache 规格
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # 3. Profile GPU 显存 → 确定可用内存
    available_gpu_memory = self.model_executor.determine_available_memory()

    # 4. 计算 KV Cache 配置 (block 数量, block 大小)
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )

    # 5. Auto-fit: 如果显存不足, 自动降低 max_model_len
    if max_model_len_after != max_model_len_before:
        self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

    # 6. 生成调度器 KV Cache 配置
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)

    # 7. 初始化 KV Cache 并 warmup
    self.model_executor.initialize_from_config(kv_cache_configs)

    return scheduler_kv_cache_config
```

**关键**: 如果 GPU 显存不足以支持 `max_model_len`，会自动降低 `max_model_len` 并同步到所有 worker。

## 4. step() — 核心循环

### 4.1 基础模式 (无 Batch Queue)

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 0. 无请求则直接返回
    if not self.scheduler.has_requests():
        return {}, False

    # 1. 调度: Scheduler 决定本次执行哪些请求/多少 tokens
    scheduler_output = self.scheduler.schedule()

    # 2. 执行: 提交模型前向传播 (non-blocking Future)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 结构化输出: 计算 grammar bitmask
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待执行完成 + 采样
    with self.log_error_detail(scheduler_output), \
         self.log_iteration_details(scheduler_output):
        model_output = future.result()
        if model_output is None:
            # None → 需要单独调用 sample_tokens
            model_output = self.model_executor.sample_tokens(grammar_output)

    # 5. 处理执行期间到达的 abort
    self._process_aborts_queue()

    # 6. 更新调度器状态 + 生成输出
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### 4.2 Batch Queue 模式 (Pipeline Parallelism)

```python
def step_with_batch_queue(self):
    """流水线并行: 调度和执行解耦, 消除气泡"""

    # 1. 尝试调度新 batch (如果 queue 未满)
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)

        if not pending_structured_output_tokens:
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        else:
            # 延迟采样 (spec decode + structured output 交互)
            deferred_scheduler_output = scheduler_output

        # 加入 batch queue
        batch_queue.appendleft((future, scheduler_output, exec_future))

        # 如果 queue 未满且还有请求 → 继续调度, 不等待
        if len(batch_queue) < batch_queue_size and (
            model_executed or self.scheduler.has_requests()
        ):
            return None, model_executed

    # 2. Queue 已满 → 阻塞等待最早的 batch 完成
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    model_output = future.result()

    # 3. 更新调度器
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    # 4. 处理延迟的结构化输出 (spec decode)
    if deferred_scheduler_output:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        self.scheduler.update_draft_token_ids_in_output(
            draft_token_ids, deferred_scheduler_output
        )
        grammar_output = self.scheduler.get_grammar_bitmask(deferred_scheduler_output)
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed
```

**关键设计**:
- `max_concurrent_batches` 控制 batch queue 深度
- 优先填满 queue, 再阻塞等待输出
- Spec Decode + Structured Output 需要"延迟采样"模式

## 5. post_step() — 步后处理

```python
def post_step(self, model_executed: bool):
    # Speculative Decoding: 获取 draft token IDs 更新调度器
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

## 6. EngineCoreProc — 后台进程模式

### 6.1 启动流程

```python
@staticmethod
def run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    """在后台进程中启动 EngineCore"""

    # 1. 根据是否 DP / 是否 MoE 选择类
    if data_parallel and is_moe:
        engine_core = DPEngineCoreProc(...)
    else:
        # 非 MoE DP rank 完全独立 → 视为 DP=1
        parallel_config.data_parallel_size = 1
        engine_core = EngineCoreProc(...)

    # 2. 注册信号处理
    signal.signal(SIGTERM, signal_handler)
    signal.signal(SIGINT, signal_handler)

    # 3. 进入 busy loop
    engine_core.run_busy_loop()
```

### 6.2 Busy Loop

```python
def run_busy_loop(self):
    """EngineCore 核心主循环"""
    while self._handle_shutdown():
        # 1. 处理输入队列 (等待有工作可做)
        self._process_input_queue()
        # 2. 执行一步 (调度 + 执行 + 输出)
        self._process_engine_step()

    raise SystemExit
```

### 6.3 输入队列处理

```python
def _process_input_queue(self):
    """等待直到有工作可做"""
    while not self.has_work() and self.is_running():
        # 空闲回调 (pause 等待等)
        self._notify_idle_state_callbacks()
        # 阻塞等待新请求
        req = self.input_queue.get(block=True)
        self._handle_client_request(*req)

    # 非阻塞处理剩余请求
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)
```

### 6.4 引擎步进

```python
def _process_engine_step(self):
    outputs, model_executed = self.step_fn()
    # 输出放入 output_queue → output_thread 发送到前端
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)
    # 步后处理
    self.post_step(model_executed)
    # 无模型执行但有调度器工作 → yield GIL 1ms
    if not model_executed and self.scheduler.has_requests():
        time.sleep(0.001)
```

### 6.5 请求类型处理

```python
def _handle_client_request(self, request_type, request):
    if request_type == ADD:
        self.add_request(req, request_wave)
    elif request_type == ABORT:
        self.abort_requests(request)
    elif request_type == UTILITY:
        # RPC 调用 (profile, reset_cache 等)
        self._invoke_utility_method(...)
    elif request_type == EXECUTOR_FAILED:
        raise RuntimeError("Executor failed.")
```

## 7. ZMQ 通信架构

```
┌───────────────────────────────┐       ZMQ        ┌──────────────────────────┐
│       EngineCoreClient        │                   │     EngineCoreProc       │
│       (前端进程)               │                   │     (后台进程)            │
│                               │                   │                          │
│  input_socket (ROUTER) ─────────── DEALER ─── input_thread                  │
│  output_socket (PULL) ←────────── PUSH ───── output_thread                 │
│                               │                   │                          │
│  add_request() → serialize ──────→ input_queue → _handle_client_request    │
│  get_output()  ← deserialize ←──── output_queue ← _process_engine_step     │
└───────────────────────────────┘                   └──────────────────────────┘
```

**关键设计**:
- **ROUTER/DEALER 模式**: 前端 ROUTER 可连接多个 DEALER (多 DP rank)
- **PULL/PUSH 模式**: 输出单向推送, 零等待
- **双线程解耦**: input_thread 和 output_thread 独立处理 ZMQ IO
- **Buffer 复用**: output_thread 使用 `reuse_buffers` 避免频繁分配
- **Zero-copy**: 大 buffer 使用 `copy=False, track=True` 零拷贝发送

### 7.1 输入线程 (process_input_sockets)

```python
def process_input_sockets(self, input_addresses, coord_input_address, identity, ...):
    # DEALER socket 连接前端 ROUTER
    input_sockets = [make_zmq_socket(ctx, addr, zmq.DEALER, ...) for addr in input_addresses]

    # 发送 ready_payload → 前端 ROUTER 才能回复
    for socket in input_sockets:
        socket.send(ready_payload)
        poller.register(socket, zmq.POLLIN)

    while True:
        for socket, _ in poller.poll():
            type_frame, *data_frames = socket.recv_multipart()
            request_type = EngineCoreRequestType(bytes(type_frame.buffer))

            if request_type == ADD:
                req = add_request_decoder.decode(data_frames)
                req = self.preprocess_add_request(req)
            elif request_type == ABORT:
                # 同时放入 aborts_queue (提前中止)
                self.aborts_queue.put_nowait(request)

            self.input_queue.put_nowait((request_type, request))
```

### 7.2 输出线程 (process_output_sockets)

```python
def process_output_sockets(self, output_paths, coord_output_path, engine_index):
    sockets = [make_zmq_socket(ctx, path, zmq.PUSH, linger=4000) for path in output_paths]

    while True:
        output = self.output_queue.get()
        if output == ENGINE_CORE_DEAD:
            for socket in sockets: socket.send(output)
            break
        client_index, outputs = output
        # 复用 buffer
        buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
        buffers = encoder.encode_into(outputs, buffer)
        sockets[client_index].send_multipart(buffers, copy=False, track=True)
```

## 8. 握手协议 (Startup Handshake)

```
EngineCoreProc                          Frontend (API Server)
     │                                        │
     │  ──── HELLO {local, headless} ────────→│  DEALER → ROUTER
     │                                        │
     │  ←── InitMessage {addresses, config} ──│  ROUTER → DEALER
     │                                        │
     │  (初始化模型 + KV Cache + Scheduler)    │
     │                                        │
     │  ──── READY {status, config_hash} ────→│  DEALER → ROUTER
     │                                        │
     │  ←── EngineCoreReadyResponse ──────────│  ROUTER → DEALER
     │                                        │
     │  ──── ready_payload ──────────────────→│  正式数据传输开始
```

## 9. EngineCoreClient 体系

```
EngineCoreClient (ABC)
  │
  ├── InprocClient                    ← V0 风格, 进程内直接调用
  │   └── 直接调用 EngineCore.step()
  │
  └── MPClient                        ← 多进程 ZMQ 通信
       │
       ├── SyncMPClient               ← LLM (同步) 使用
       │   └── get_output() 阻塞等待
       │
       └── AsyncMPClient              ← AsyncLLM 使用
            │   ├── get_output_async() asyncio
            │   └── add_request_async() asyncio
            │
            ├── DPAsyncMPClient       ← 外部 LB, 每个 DP rank 一个 client
            │
            └── DPLBAsyncMPClient     ← 内部 LB, client 自动分发到 DP ranks
```

### 9.1 Client 选择逻辑

```python
# make_client():
if multiprocess_mode and asyncio_mode:
    if dp_size > 1:
        if external_lb:   → DPAsyncMPClient
        else:             → DPLBAsyncMPClient  # 内部负载均衡
    else:                 → AsyncMPClient

elif multiprocess_mode:
                         → SyncMPClient

else:                     → InprocClient
```

### 9.2 MPClient 初始化

```python
class MPClient(EngineCoreClient):
    def __init__(self, asyncio_mode, vllm_config, executor_class, ...):
        # ZMQ 上下文
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # 输入 socket: ROUTER (可连接多个 DEALER)
        self.input_socket = make_zmq_socket(ctx, address, zmq.ROUTER, bind=True)

        # 输出 socket: PULL
        self.output_socket = make_zmq_socket(ctx, address, zmq.PULL)

        # 启动后台 EngineCore 进程
        self.process = spawn(run_engine_core, args=(...))
```

## 10. DPEngineCoreProc — 数据并行扩展

### 10.1 初始化

```python
class DPEngineCoreProc(EngineCoreProc):
    """仅用于 MoE 模型的 DP"""

    def __init__(self, ...):
        # Wave 管理
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # 两阶段暂停
        self.pending_pause = False
        self.ignore_start_dp_wave = False

        # 初始化 DP process group
        self.dp_group, self.dp_store = parallel_config.stateless_init_dp_group()
```

### 10.2 请求波次管理

```python
def add_request(self, request, request_wave=0):
    super().add_request(request, request_wave)
    if self.has_coordinator and request_wave != self.current_wave:
        if request_wave > self.current_wave:
            self.current_wave = request_wave
        elif not self.engines_running:
            # 收到已完成 wave 的请求 → 通知前端启动下一个
            self.engines_running = True
```

### 10.3 两阶段暂停

```python
def _pause_complete(self):
    """DP 感知的两阶段暂停"""
    # Phase 1: 设置本地暂停标志, 启动引擎让它到达 all-reduce 检查点
    self.pending_pause = True
    self.engines_running = True
    return False  # 不立即完成, 等待 all-reduce 共识

    # Phase 2: 在 _has_global_unfinished_reqs 中
    # 通过 all-reduce 确认所有 rank 都 pending_pause
    # 集体停止 stepping
```

## 11. 关闭流程

```python
def shutdown(self):
    # 1. 清理结构化输出
    self.structured_output_manager.clear_backend()
    # 2. 关闭模型执行器
    self.model_executor.shutdown()
    # 3. 关闭调度器
    self.scheduler.shutdown()
    # 4. 解冻 GC (允许回收模型权重等)
    gc.unfreeze()
    # 5. 清理分布式状态
    cleanup_dist_env_and_memory()
```

**优雅关闭**:
- `shutdown_timeout > 0`: 等待在途请求完成
- `shutdown_timeout == 0`: 立即中止所有请求
- 信号处理: SIGTERM/SIGINT 触发关闭流程

## 12. 关键洞察

1. **step() 是心跳**: 调度→执行→采样→输出, 一次完整迭代
2. **non-blocking 执行**: `execute_model(non_block=True)` 返回 Future, 与 grammar bitmask 计算并行
3. **Batch Queue**: PP 场景下调度和执行解耦, 填满 queue 优先于等待输出
4. **Busy Loop**: EngineCoreProc 运行在独立进程, 无 Python GIL 限制 (ZMQ 释放 GIL)
5. **双线程 ZMQ IO**: input_thread + output_thread 与 busy loop 解耦, 重叠序列化与计算
6. **Buffer 复用**: output_thread 维护 `reuse_buffers` 减少 GC 压力
7. **Auto-fit**: 显存不足时自动降低 `max_model_len`, 保证启动不失败
8. **GC 冻结**: `freeze_gc_heap()` 标记启动堆为静态, 减少老年代 GC 暂停
9. **DP 隔离**: 非 MoE DP rank 视为 DP=1, 完全独立运行
10. **两阶段暂停**: DP 场景需要 all-reduce 共识才能安全暂停

## 13. 数据流总结

```
Request 流入:
  API Server → AsyncLLM.add_request()
    → InputProcessor.process_inputs()     # tokenize + MM + LoRA
    → EngineCoreClient.add_request_async()
      → ZMQ DEALER → ROUTER              # input_thread
        → input_queue
          → _handle_client_request(ADD)
            → scheduler.add_request()

Request 执行 (每次 step):
  run_busy_loop()
    → _process_input_queue()              # 处理新请求
    → _process_engine_step()
      → step()
        → scheduler.schedule()            # 调度
        → model_executor.execute_model()  # GPU 前向
        → sample_tokens()                 # 采样
        → scheduler.update_from_output()  # 更新状态
      → output_queue.put()               # 放入输出队列
        → output_thread
          → ZMQ PUSH → PULL              # output_thread
            → EngineCoreClient.get_output_async()

Request 完成:
  update_from_output() → finish_reason
    → EngineCoreOutput(finished=True)
      → output_queue → output_thread → ZMQ → client
        → AsyncLLM.output_handler
          → RequestOutputCollector.put()
            → generate() yield RequestOutput
```

## 参考资料

- `vllm/v1/engine/core.py` — EngineCore + EngineCoreProc + DPEngineCoreProc
- `vllm/v1/engine/core_client.py` — InprocClient / MPClient / AsyncMPClient
- `vllm/v1/engine/__init__.py` — EngineCoreRequest/Output 类型定义
- `vllm/v1/engine/async_llm.py` — AsyncLLM 前端
- `vllm/v1/engine/utils.py` — EngineHandshakeMetadata, EngineZmqAddresses
- 相关: [AsyncLLM Frontend](vllm-async-engine-frontend-reading.md), [Architecture Map](vllm-v1-architecture-map.md)
