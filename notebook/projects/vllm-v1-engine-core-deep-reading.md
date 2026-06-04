# vLLM V1 Engine Core 深度源码分析

> 源码路径: `vllm/v1/engine/core.py` (1600+ 行)
> 核心类: `EngineCore` (单进程) + `EngineCoreProc` (多进程 ZMQ wrapper)

## 1. 双进程架构

```
┌─────────────────────────┐        ZMQ          ┌──────────────────────────┐
│   Frontend Process       │                      │   EngineCore Process      │
│   (AsyncLLM)             │                      │                           │
│                          │   DEALER/ROUTER      │                           │
│   output_handler ────────│◄─────────────────────│── output_thread           │
│   (asyncio Task)         │                      │   (Python Thread)         │
│                          │                      │                           │
│   generate() ────────────│────────────────────►│── input_thread            │
│   abort()                │   DEALER/ROUTER      │   (Python Thread)         │
│                          │                      │                           │
│                          │                      │   core_busy_loop (main)   │
│                          │                      │   ├─ process_input_queue  │
│                          │                      │   └─ process_engine_step  │
└─────────────────────────┘                      └──────────────────────────┘
```

**为什么用双进程?**
1. **GIL 隔离**: EngineCore 在独立进程中运行, GPU 计算不受前端 asyncio 影响
2. **ZMQ IPC**: 两个进程通过 ZMQ DEALER/ROUTER 通信, 零拷贝 msgpack 序列化
3. **多前端共享**: 多个 API server 进程可以连接同一个 EngineCore (DP serving)

## 2. 核心循环: run_busy_loop()

```python
# core.py:1216-1224
def run_busy_loop(self):
    while self._handle_shutdown():
        # 1) 轮询 input_queue 直到有工作
        self._process_input_queue()
        # 2) 执行引擎步进并返回输出
        self._process_engine_step()
    raise SystemExit
```

**3 种状态**:
- **IDLE**: `has_work() == False` → 阻塞等待 input_queue
- **ACTIVE**: `has_work() == True` → 循环执行 step
- **SHUTDOWN**: `SIGTERM/SIGINT` → 优雅关闭 (drain 或 abort)

## 3. 三线程设计

### 3.1 Input Thread (process_input_sockets)

```python
# core.py:1423-1518
def process_input_sockets(self, input_addresses, ...):
    # 1. 为每个 input address 创建 DEALER socket
    # 2. 发送 EngineCoreReadyResponse (max_model_len, num_gpu_blocks, etc.)
    # 3. 轮询所有 socket:
    while True:
        for input_socket, _ in poller.poll():
            type_frame, *data_frames = input_socket.recv_multipart(copy=False)
            request_type = EngineCoreRequestType(bytes(type_frame.buffer))
            # 反序列化: msgpack decode
            request = add_request_decoder.decode(data_frames)  # ADD
            # 或 generic_decoder.decode(data_frames)           # ABORT/UTILITY
            # 推入 input_queue
            self.input_queue.put_nowait((request_type, request))
```

**零拷贝**: `recv_multipart(copy=False)` 避免数据复制, 直接引用 ZMQ buffer。

### 3.2 Output Thread (process_output_sockets)

```python
# core.py:1520-1585
def process_output_sockets(self, output_paths, ...):
    # 1. 为每个 output path 创建 PUSH socket
    # 2. 循环读取 output_queue:
    while True:
        output = self.output_queue.get()
        client_index, outputs = output
        # 序列化 + 发送
        buffers = encoder.encode_into(outputs, buffer)  # msgpack
        tracker = sockets[client_index].send_multipart(buffers, copy=False, track=True)
        # Buffer 重用
        if not tracker.done:
            pending.appendleft((tracker, ref, buffer))
```

**Buffer 池**: 重用 bytearray 避免频繁分配, `max_reuse_bufs = num_sockets + 1`。

### 3.3 Main Thread (core_busy_loop)

主线程运行 busy loop, 从 input_queue 读取请求, 通过 step() 执行 GPU 推理, 结果放入 output_queue。

## 4. EngineCore.step() — 核心推理循环

```python
# core.py:443-472
def step(self):
    if not self.scheduler.has_requests():
        return {}, False

    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**4 步管线**:
1. **Schedule**: `scheduler.schedule()` → 决定哪些请求参与 prefill/decode
2. **Execute**: `model_executor.execute_model()` → GPU 前向传播 (non-blocking future)
3. **Sample**: `sample_tokens()` → 从 logits 采样下一个 token
4. **Update**: `scheduler.update_from_output()` → 更新请求状态, 检查完成

**non_block=True**: execute_model 返回 Future, 允许 CPU 在 GPU 执行期间做其他工作 (如 grammar bitmask 计算)。

## 5. Batch Queue 模式 (Pipeline Parallelism)

```python
# core.py:484-598
def step_with_batch_queue(self):
    # 1. 尝试调度新 batch (非阻塞)
    scheduler_output = self.scheduler.schedule()
    exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 2. 如果 batch queue 未满, 直接返回 (优先填充队列)
    if len(batch_queue) < self.batch_queue_size:
        batch_queue.appendleft((future, scheduler_output, exec_future))
        return None, model_executed

    # 3. 队列已满, 阻塞等待最早的结果
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    model_output = future.result()
    engine_core_outputs = self.scheduler.update_from_output(...)
```

**设计**: PP 场景下多个 micro-batch 同时在 pipeline 中执行, batch_queue 缓冲多个 in-flight batch。

## 6. 请求类型处理

```python
# core.py:1317-1346
def _handle_client_request(self, request_type, request):
    if request_type == EngineCoreRequestType.ADD:
        self.add_request(req, request_wave)
    elif request_type == EngineCoreRequestType.ABORT:
        self.abort_requests(request)
    elif request_type == EngineCoreRequestType.UTILITY:
        # 动态方法调用: getattr(self, method_name)(*args)
        self._invoke_utility_method(...)
    elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
        raise RuntimeError("Executor failed.")
```

**UTILITY**: 运行时调用 EngineCore 的任意方法 (如 pause_scheduler, profile 等), 返回 UtilityOutput。

## 7. Abort 处理

**双队列设计**:
1. `aborts_queue`: 紧急 abort 队列, 在 step() 之间处理
2. `input_queue`: 常规 abort, 按顺序处理

```python
# Abort 同时推入两个队列 (core.py:1510-1515)
if request_type == EngineCoreRequestType.ABORT:
    self.aborts_queue.put_nowait(request)  # 紧急通道
# 也通过 input_queue 正常处理
```

**为什么双队列?**
- `aborts_queue`: 在 step() 的 `execute_model` 期间就可以检查, 快速取消
- `input_queue`: 保证顺序正确, 防止 abort 后又处理了该请求的后续操作

## 8. 优雅关闭

```python
# core.py:1281-1315
def _handle_shutdown(self):
    if shutdown_state == REQUESTED:
        if shutdown_timeout == 0:
            # 立即 abort 所有请求
            aborted = self.scheduler.finish_requests(None, FINISHED_ABORTED)
        else:
            # 继续运行, drain 所有 in-flight 请求
            logger.info("Draining %d in-flight requests (timeout=%ds)", ...)
    # 真正退出: 所有请求完成或 abort 完成
    if not self.has_work():
        return False
```

**gc.unfreeze()**: shutdown 时恢复垃圾回收, 释放模型权重等大对象 (init 时 gc.freeze() 防止 GC 扫描)。

## 9. 性能优化总结

| 优化 | 方法 | 效果 |
|------|------|------|
| 双进程 | ZMQ IPC | GIL 隔离, GPU 不受前端影响 |
| 三线程 | input/output/main | IO 与计算重叠 |
| non_block execute_model | Future 异步 | CPU 在 GPU 执行期间做 grammar 等 |
| 零拷贝 | `copy=False` + buffer 重用 | 减少序列化开销 |
| msgpack | 二进制序列化 | 比 JSON 快 5-10x |
| input_queue 批量处理 | 一次处理多个请求 | 减少锁竞争 |
| batch_queue | PP micro-batch 缓冲 | 多 batch 流水线执行 |

## 10. 数据流总览

```
Client Request (HTTP)
    │
    ▼
AsyncLLM.generate()
    │ ZMQ DEALER (msgpack)
    ▼
Input Thread ──→ input_queue ──→ Main Thread
    │                                │
    │                          ┌─────┴─────┐
    │                          │ has_work? │
    │                          └─────┬─────┘
    │                                │ Yes
    │                          ┌─────▼──────┐
    │                          │ schedule() │
    │                          │ execute()  │── GPU
    │                          │ sample()   │
    │                          │ update()   │
    │                          └─────┬──────┘
    │                                │
    │                          output_queue
    │                                │
    ▼                                ▼
Output Thread ──→ ZMQ PUSH ──→ Frontend output_handler
                                    │
                                    ▼
                              RequestOutput (streaming)
```
