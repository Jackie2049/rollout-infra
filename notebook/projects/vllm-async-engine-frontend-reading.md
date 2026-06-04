# vLLM V1 AsyncLLM Engine Frontend 源码阅读

> AsyncLLM: API Server 与 EngineCore 之间的异步桥梁

## 1. 架构概览

```
┌──────────────────────────────────────────────────────────┐
│                      API Server (asyncio)                  │
│   POST /v1/completions → async for chunk in generate()    │
└──────────────────────┬───────────────────────────────────┘
                        │
    ┌───────────────────┴───────────────────┐
    │           AsyncLLM (Frontend)          │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │ InputProcessor                  │  │ ← EngineInput → EngineCoreRequest
    │  │ (input_processor.py)            │  │    Tokenize + MM + LoRA
    │  └─────────────────────────────────┘  │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │ OutputProcessor                 │  │ ← EngineCoreOutputs → RequestOutput
    │  │ (output_processor.py)           │  │    Detokenize + Stats
    │  └─────────────────────────────────┘  │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │ EngineCoreClient (async)        │  │ ← ZMQ 通信
    │  │ (core_client.py)                │  │    add_request / get_output
    │  └─────────────────────────────────┘  │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │ output_handler (background)     │  │ ← 持续拉取 output
    │  │ asyncio.Task                    │  │    分发到 per-request queue
    │  └─────────────────────────────────┘  │
    │                                       │
    │  ┌─────────────────────────────────┐  │
    │  │ StatLoggerManager               │  │ ← Stats 记录
    │  │ (metrics/loggers.py)            │  │    Prometheus + Logging
    │  └─────────────────────────────────┘  │
    └───────────────────────┬───────────────┘
                            │ ZMQ
    ┌───────────────────────┴───────────────┐
    │        EngineCore (Background Proc)     │
    │   Scheduler → Executor → Worker → GPU   │
    └───────────────────────────────────────┘
```

## 2. 核心请求流程

### 2.1 generate() — 主入口

```python
async def generate(self, prompt, sampling_params, request_id, ...) \
        -> AsyncGenerator[RequestOutput, None]:
    # 1. 添加请求 (InputProcessor + OutputProcessor + EngineCore)
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    # 2. 从 per-request queue 拉取输出
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out

    # 3. 断开连接时自动 abort
    except (asyncio.CancelledError, GeneratorExit):
        await self.abort(q.request_id, internal=True)
```

### 2.2 add_request() — 请求注册

```python
async def add_request(self, request_id, prompt, params, ...):
    # 1. InputProcessor: 转换输入
    request = self.input_processor.process_inputs(
        request_id, prompt, params, ...
    )

    # 2. OutputProcessor: 注册 per-request queue
    queue = RequestOutputCollector(params.output_kind, request.request_id)

    # 3. n>1: fan-out 为多个子请求 (ParentRequest)
    if params.n > 1:
        parent_request = ParentRequest(request)
        for idx in range(params.n):
            await self._add_request(child_request, ...)
    else:
        await self._add_request(request, prompt, None, 0, queue)

    return queue
```

### 2.3 _add_request() — 实际注册

```python
async def _add_request(self, request, prompt, parent_req, index, queue):
    # 1. 注册到 OutputProcessor (前端进程)
    self.output_processor.add_request(request, prompt, parent_req, index, queue)

    # 2. 发送到 EngineCore (后端进程, via ZMQ)
    await self.engine_core.add_request_async(request)
```

## 3. output_handler — 后台输出循环

```python
async def output_handler():
    while True:
        # 1. 从 EngineCore 拉取输出 (async ZMQ)
        outputs = await engine_core.get_output_async()

        # 2. 分块处理 (避免阻塞事件循环)
        iteration_stats = IterationStats() if log_stats else None
        for start in range(0, num_outputs, chunk_size):
            # 3. OutputProcessor 处理 (detokenize + stats)
            processed = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, iteration_stats
            )

            # 4. 中断因 stop string 完成的请求
            if processed.reqs_to_abort:
                await engine_core.abort_requests_async(processed.reqs_to_abort)

        # 5. 更新 scheduler stats + 日志
        output_processor.update_scheduler_stats(outputs.scheduler_stats)
        logger_manager.record(scheduler_stats, iteration_stats, ...)
```

**关键设计**:
- `chunk_size = VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`: 分块避免事件循环饥饿
- `await asyncio.sleep(0)`: chunk 之间让出控制权
- `RequestOutputCollector`: per-request queue, 支持 `get_nowait()` 零开销路径

## 4. 核心组件

### 4.1 InputProcessor (`vllm/v1/engine/input_processor.py`)

```python
class InputProcessor:
    def process_inputs(self, request_id, prompt, params, ...):
        """转换各种输入格式 → EngineCoreRequest"""
        # 1. Tokenize (如果未 tokenized)
        # 2. 处理 MM 输入
        # 3. 处理 LoRA
        # 4. 验证 sampling params
        # 5. 创建 EngineCoreRequest
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=...,
            mm_features=...,
            sampling_params=...,
            arrival_time=time.time(),
            lora_request=...,
        )
```

### 4.2 OutputProcessor (`vllm/v1/engine/output_processor.py`)

```python
class OutputProcessor:
    def process_outputs(self, outputs, timestamp, iteration_stats):
        """转换 EngineCoreOutputs → RequestOutput"""
        # 1. Detokenize (token IDs → text)
        # 2. 收集 logprobs
        # 3. 更新 iteration stats
        # 4. 推送到 per-request RequestOutputCollector
        # 5. 处理 finish reason

    def add_request(self, request, prompt, parent_req, index, queue):
        """注册新请求的 detokenizer 状态"""
```

### 4.3 EngineCoreClient (`vllm/v1/engine/core_client.py`)

```python
class EngineCoreClient:
    """与 EngineCore 进程通信的异步客户端"""

    async def add_request_async(self, request): ...
    async def get_output_async(self) -> EngineCoreOutputs: ...
    async def abort_requests_async(self, request_ids): ...
```

通信方式: ZMQ (异步 IPC)

### 4.4 RequestOutputCollector

```python
class RequestOutputCollector:
    """per-request 输出队列"""

    def put(self, output: RequestOutput): ...
    def get_nowait(self) -> RequestOutput | None: ...
    async def get(self) -> RequestOutput: ...
```

## 5. Streaming Input 支持

```python
async def _add_streaming_input_request(self, request_id, input_stream, ...):
    """流式输入: 边接收输入边处理"""

    async def handle_inputs():
        async for input_chunk in input_stream:
            req = self.input_processor.process_inputs(
                request_id, input_chunk.prompt, params, resumable=True, ...
            )
            await self.engine_core.add_request_async(req)
```

支持场景:
- 逐 chunk 接收 prompt
- 多模态输入流式处理
- 边收边发，降低 TTFT

## 6. n>1 请求 (并行采样)

```python
# ParentRequest 管理 n>1 的子请求
parent_request = ParentRequest(request)
for idx in range(params.n):
    child_request = copy(request)
    child_request.request_id = parent_request.get_child_info(idx)
    await self._add_request(child_request, ..., parent_request, idx, queue)
```

所有子请求的输出合并到同一个 `RequestOutputCollector`。

## 7. 错误处理

| 场景 | 行为 |
|------|------|
| 客户端断开 | `CancelledError` → abort request |
| Engine 死亡 | `EngineDeadError` → 不 abort (已关闭) |
| 请求验证错误 | `ValueError` → 直接抛出 |
| 输入流错误 | `InputStreamError` → abort + 传播 |
| 未知错误 | abort + 记录日志 |

## 8. 关键洞察

1. **双进程架构**: AsyncLLM (前端, asyncio) ← ZMQ → EngineCore (后端, busy loop)
2. **output_handler**: 后台 asyncio Task 持续拉取输出，分块处理避免阻塞
3. **per-request queue**: `RequestOutputCollector` 实现零开销路径 (`get_nowait`)
4. **InputProcessor**: 统一处理 tokenize + MM + LoRA + 验证
5. **OutputProcessor**: detokenize + stats + 推送到 per-request queue
6. **n>1 fan-out**: `ParentRequest` 管理并行采样的多个子请求
7. **streaming input**: 支持边接收 prompt 边处理
8. **分块处理**: `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 避免事件循环饥饿
9. **自动 abort**: 客户端断开连接时自动清理请求
10. **Log Stats**: 每次 output_handler 迭代记录 iteration_stats

## 参考资料

- `vllm/v1/engine/async_llm.py` — AsyncLLM 主类
- `vllm/v1/engine/input_processor.py` — InputProcessor
- `vllm/v1/engine/output_processor.py` — OutputProcessor
- `vllm/v1/engine/core_client.py` — EngineCoreClient (ZMQ)
- `vllm/v1/engine/core.py` — EngineCore / EngineCoreProc
- `vllm/v1/engine/parallel_sampling.py` — ParentRequest
- 相关: [V1 Architecture Map](vllm-v1-architecture-map.md), [Observability](vllm-observability-reading.md)
