# vLLM V1 UBatch (Micro-Batching) 与 DBO (Dual Batch Overlap) 源码深度分析

> 阅读日期: 2026-06-05
> 源码版本: vllm-latest (main 分支)
> 核心文件: 8 个文件, ~2900 行

---

## 1. 总览: DBO 解决什么问题?

DBO (Dual Batch Overlap) 是 vLLM V1 中专门为 **DP+EP (Data Parallel + Expert Parallel)** 部署设计的通信-计算重叠机制。核心动机:

MoE 模型在 EP 模式下, 每个 Transformer 层需要一次 All-to-All (A2A) dispatch + 一次 A2A combine。在 decode 阶段 (memory-bound), 计算 GEMM 很快但通信 A2A 占比大, 形成瓶颈。DBO 通过将 batch 拆成两个 micro-batch, 让一个 micro-batch 的计算与另一个 micro-batch 的通信在 GPU 上同时执行, 从而隐藏通信延迟。

**关键设计约束**:
- 仅支持 DP+EP 部署 (`--data-parallel-size > 1` + `--enable-expert-parallel`)
- 依赖 DeepEP 通信库 (high-throughput 或 low-latency 模式)
- 默认 micro-batch 数 = 2, 硬编码 (`_NUM_UBATCHES: int = 2`)
- 仅支持 Full CUDA Graph 模式

**启用命令示例**:
```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --all2all-backend deepep_low_latency
```

---

## 2. 架构全景

```
┌──────────────────────────────────────────────────────────┐
│                   GPUModelRunner                          │
│  _determine_batch_execution_and_padding()                │
│    └── coordinate_batch_across_dp() [dp_utils.py]       │
│        └── AllReduce 同步所有 DP rank                    │
│  maybe_create_ubatch_slices() [ubatch_utils.py]         │
│    └── 拆分成 UBatchSlice[0] + UBatchSlice[1]          │
│  set_forward_context(ubatch_slices=...)                  │
│    └── self.model(...) → UBatchWrapper.__call__()       │
└────────────────┬─────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────┐
│                   UBatchWrapper                           │
│  [gpu_ubatch_wrapper.py]                                 │
│  管理: 线程生命周期 / CUDA Graph / SM 分区              │
│                                                          │
│  _make_ubatch_metadata()                                │
│    ├── 为每个 ubatch 创建独立 ForwardContext             │
│    └── make_ubatch_contexts() → UBatchContext[0,1]      │
│                                                          │
│  _run_ubatches() / _capture_ubatches()                  │
│    ├── Thread(ubatch_0) ──→ model(...) on compute_stream │
│    ├── Thread(ubatch_1) ──→ model(...) on compute_stream │
│    └── 两个线程通过 UBatchContext 乒乓切换              │
└────────────────┬─────────────────────────────────────────┘
                 │ (每个线程独立跑 model forward)
┌────────────────▼─────────────────────────────────────────┐
│              FusedMoEKernelModularImpl                    │
│  [modular_kernel.py]                                     │
│                                                          │
│  _prepare():                                             │
│    ├── dbo_maybe_run_recv_hook() ← 执行上一轮 recv      │
│    ├── prepare_async() → hook + receiver                 │
│    ├── dbo_register_recv_hook(hook)                      │
│    └── dbo_yield() ← 让出 CPU, 唤醒另一个线程           │
│                                                          │
│  _fused_experts(): expert GEMM 计算                      │
│                                                          │
│  _finalize():                                            │
│    ├── finalize_async() → hook + receiver                │
│    ├── SharedExperts.apply() ← 与 A2A combine 重叠      │
│    ├── dbo_register_recv_hook(hook)                      │
│    └── dbo_yield() ← 再次让出                           │
└──────────────────────────────────────────────────────────┘
```

---

## 3. UBatch 切分: Token Batch 拆成 Micro-Batch

### 3.1 决策流程

**文件**: `vllm/v1/worker/gpu_model_runner.py:3764-3876` (`_determine_batch_execution_and_padding`)

在每次 forward 之前, GPUModelRunner 决定是否启用 micro-batching:

1. 检查 `parallel_config.use_ubatching` (由 `--enable-dbo` 或 `--ubatch-size > 1` 触发)
2. 检查 token 阈值 (`check_ubatch_thresholds`):
   - decode-only batch: token 数 >= `dbo_decode_token_threshold` (默认 32)
   - 包含 prefill: token 数 >= `dbo_prefill_token_threshold` (默认 512)
3. 通过 `coordinate_batch_across_dp()` (dp_utils.py) 在所有 DP rank 间 AllReduce 同步:
   - 每个 rank 报告自己的 token 数和是否愿意 ubatch
   - **所有 rank 必须同意 ubatch, 否则全部不 ubatch** (取 `torch.all(tensor[2] == 1)`)
   - 所有 rank pad 到最大 token 数, 确保均匀切分
   - 如果任何一个 rank 切分后第二个 ubatch 会是空的, 则放弃 ubatch

**文件**: `vllm/v1/worker/dp_utils.py:101-161` (`_synchronize_dp_ranks`)

```python
# 每个 rank 发送 4x dp_size 的 tensor:
# [0]: orig_num_tokens (未 pad 的 token 数)
# [1]: padded_num_tokens (pad 后的 token 数)
# [2]: 1 表示愿意 ubatch, 0 表示不愿意
# [3]: cudagraph_mode
tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
tensor_cpu[3][dp_rank] = cudagraph_mode
dist.all_reduce(tensor, group=group)
```

### 3.2 切分算法

**文件**: `vllm/v1/worker/ubatch_utils.py:63-114` (`maybe_create_ubatch_slices`)

切分核心思想: 将 token 维度均匀切成 2 份, 然后确定每个 ubatch 包含哪些 request:

```python
split_point = num_tokens_padded // num_ubatches  # 简单对半切

# 对每个切分点, 用 cumulative sum + searchsorted 找到对应的 request 范围
cu_num_tokens = np.cumsum(num_scheduled_tokens)
req_start = np.searchsorted(cu_num_tokens, start_token, side="right") - 1
req_stop = np.searchsorted(cu_num_tokens, end_token, side="left")
```

关键数据结构 `UBatchSlice`:
```python
@dataclass
class UBatchSlice:
    request_slice: slice  # 这个 ubatch 包含哪些 request (索引范围)
    token_slice: slice    # 这个 ubatch 包含哪些 token (索引范围)
```

**注意**: 一个 request 可能被跨 ubatch 切分! 如果 request A 有 100 个 token, 切分点在第 50 个 token, 那么:
- ubatch 0: request_slice = [0, 1], token_slice = [0, 50)
- ubatch 1: request_slice = [1, N), token_slice = [50, total)

这种切分需要特殊处理 attention metadata (`_make_metadata_with_slice`):

```python
splits_first_request = first_tok > start_locs[first_req]  # 首个 request 是否被切分
splits_last_request = last_tok < start_locs[last_req + 1] - 1  # 最后一个 request 是否被切分

if splits_first_request:
    tokens_skipped = first_tok - start_locs[first_req]
    query_start_loc[1:] -= tokens_skipped  # 调整 start loc
```

---

## 4. 双 CUDA Stream 设计

**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:113-141` (UBatchWrapper.__init__)
**文件**: `vllm/v1/worker/ubatching.py:20-37` (UBatchContext 构造)

两个核心 CUDA stream:
- **compute_stream**: 执行模型计算 (attention, FFN, expert GEMM)
- **comm_stream**: 执行 All-to-All 通信 (DeepEP dispatch/combine)

```python
# UBatchWrapper.__init__
self.comm_stream = torch.cuda.Stream(device=device)
# compute_stream 是调用时从外部传入的, 通常是 torch.cuda.current_stream()
```

**为什么需要两个 stream?** 因为 GPU 可以在不同 stream 上并行执行 kernel。当一个 ubatch 在 compute_stream 上做 expert GEMM 时, 另一个 ubatch 可以在 comm_stream 上做 A2A 通信, 两者物理上并行。

### Stream 切换 API

**文件**: `vllm/v1/worker/ubatching.py:107-148`

UBatchContext 提供完整的 stream 管理接口:

| 方法 | 作用 |
|------|------|
| `switch_to_comm()` | 切换到 comm_stream (不等待) |
| `switch_to_compute()` | 切换到 compute_stream (不等待) |
| `switch_to_comm_sync()` | 先 record compute_done event, 切到 comm_stream, 然后 comm_stream wait compute_done |
| `switch_to_compute_sync()` | 先 record comm_done event, 切到 compute_stream, 然后 compute_stream wait comm_done |
| `yield_and_switch_from_compute_to_comm()` | record compute_done + CPU yield + 切到 comm + wait compute_done |
| `yield_and_switch_from_comm_to_compute()` | record comm_done + CPU yield + 切到 compute + wait comm_done |

---

## 5. GPU Event 同步: 无 CPU 参与

**文件**: `vllm/v1/worker/ubatching.py:82-93`

两个线程之间通过 `torch.cuda.Event` 同步 GPU 操作, **完全不需要 CPU 参与**:

```python
# 在 compute_stream 上记录 "计算完成" 事件
def _signal_compute_done(self):
    self.gpu_compute_done_event.record(self.compute_stream)

# 在 comm_stream 上等待 "计算完成" 事件
def _wait_compute_done(self):
    self.comm_stream.wait_event(self.gpu_compute_done_event)

# 反向同理
def _signal_comm_done(self):
    self.gpu_comm_done_event.record(self.comm_stream)

def _wait_comm_done(self):
    self.compute_stream.wait_event(self.gpu_comm_done_event)
```

**关键**: 每个 UBatchContext 拥有独立的 GPU Event 对, 所以两个 ubatch 的 event 互不干扰:
```python
gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]
```

`torch.cuda.Event` 是 GPU 端轻量级同步原语, `record()` 在 stream 上插入一个标记, `wait_event()` 让目标 stream 等待该标记完成。整个过程在 GPU 上执行, 不涉及 CPU 中断或同步。

---

## 6. 线程模型: Barrier + Event 环形唤醒

**文件**: `vllm/v1/worker/ubatching.py:202-241` (`make_ubatch_contexts`)
**文件**: `vllm/v1/worker/ubatching.py:51-72` (UBatchContext.__enter__/__exit__)

### 6.1 初始化

`make_ubatch_contexts` 创建环形 CPU event 链:

```python
cpu_events = [threading.Event() for _ in range(num_micro_batches)]  # num_micro_batches = 2

# ubatch 0: wait = cpu_events[0], signal = cpu_events[1]
# ubatch 1: wait = cpu_events[1], signal = cpu_events[0]
# 形成环形: 0 → 1 → 0 → 1 → ...
ctx = UBatchContext(
    cpu_wait_event=cpu_events[i],
    cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],  # 环形!
)
```

加上一个 `threading.Barrier(num_ubatches + 1)` (UBatchWrapper.__init__), 用于所有线程 + 主线程的同步点。

### 6.2 启动流程

**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:295-331` (`_run_ubatches`)

```python
# 主线程启动两个 worker 线程
with override_forward_context(None):  # 清空 forward context
    for metadata in ubatch_metadata:
        thread = Thread(target=_ubatch_thread, args=(results, model, metadata))
        thread.start()
    self.ready_barrier.wait()  # 等 2 个线程 + 主线程都到达 barrier
    ubatch_metadata[0].context.cpu_wait_event.set()  # 唤醒线程 0
    for thread in ubatch_threads:
        thread.join()
```

每个 worker 线程的入口 (`_ubatch_thread`):
```python
with ubatch_metadata.context:  # UBatchContext.__enter__
    # 1. ready_barrier.wait() — 等所有线程就绪
    # 2. cpu_wait_event.wait() — 等被唤醒
    # 3. _restore_context() — 恢复 forward_context
    # 4. update_stream(compute_stream) — 设置初始 stream
    model_output = model(input_ids=..., positions=..., ...)
```

### 6.3 Yield (乒乓切换) 机制

**文件**: `vllm/v1/worker/ubatching.py:94-106` (`_cpu_yield`)

```python
def _cpu_yield(self):
    assert forward_context._forward_context == self.forward_context  # 安全检查
    assert current_stream() == self.current_stream
    self.cpu_signal_event.set()       # 唤醒另一个线程
    self.cpu_wait_event.wait()        # 自己休眠
    self.cpu_wait_event.clear()
    self._restore_context()           # 恢复 forward context (因为另一个线程可能修改了全局变量)
```

这是纯 CPU 端的线程调度。关键保证: **任意时刻只有一个线程在执行 Python 代码**。通过 assert 确保:
```python
assert forward_context._forward_context == self.forward_context
assert current_stream() == self.current_stream
```

### 6.4 乒乓执行时序

以 DeepEP High-Throughput 的 `_do_dispatch` 和 `_finalize` 为例:

```
时间线 →
Thread 0 (ubatch 0):   [A0₀][A1₀] ───yield──→              [MLP₀][S₀] ───yield──→ [A0₁][A1₁]
Thread 1 (ubatch 1):              ───yield──→ [MLP₁][S₁] ───yield──→ [A0₂][A1₂]
Comm Stream:           ──────── [D₁ send] ──→ [D₀ send] ──→ [C₁ send] ──→ [C₀ send] ──→

其中:
  A0 = QKV projection (compute)
  A1 = Attention + output proj + MoE gate (compute)
  S  = Shared experts (compute)
  MLP = Routed expert GEMM (compute)
  D  = DeepEP dispatch (A2A send)
  C  = DeepEP combine (A2A receive)
```

---

## 7. SM 分区: SMControlContextManager

**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:68-111`

### 7.1 设计动机

当 DBO 让 compute_stream 和 comm_stream 同时运行时, 两者会争抢 GPU 的 SM (Streaming Multiprocessor)。如果不加控制:
- DeepEP A2A 通信 kernel 可能占用大量 SM, 挤压计算 kernel
- 计算 kernel 也可能占满 SM, 导致通信 kernel 无法调度

SMControlContextManager 通过分区解决: 进入时, 给通信分配固定数量的 SM, 计算拿剩余的 SM; 退出时恢复全部 SM。

### 7.2 实现

```python
class SMControlContextManager:
    def __init__(self, comm_sms, set_comm_sms, set_compute_sms):
        total_sms = num_compute_units(device)  # e.g., A100 有 108 SMs
        self.comm_sms = comm_sms               # 通信 SM 数量 (默认 VLLM_DBO_COMM_SMS=20)
        self.compute_sms = total_sms - comm_sms # 计算 SM 数量

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)       # 限制 DeepEP kernel 最多用 20 SMs
        self.set_compute_sms(self.compute_sms)  # 限制 DeepGEMM kernel 最多用 88 SMs

    def __exit__(self, ...):
        self.set_comm_sms(self.total_sms)       # 恢复: DeepEP 可以用全部 108 SMs
        self.set_compute_sms(self.total_sms)    # 恢复: DeepGEMM 可以用全部 108 SMs
```

### 7.3 配置来源

**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:154-185` (`_create_sm_control_context`)

```python
comm_sms = envs.VLLM_DBO_COMM_SMS  # 默认 20

# 通信端: DeepEP all2all_manager
if enable_expert_parallel:
    all2all_manager = get_ep_group().device_communicator.all2all_manager
    max_sms_used = all2all_manager.max_sms_used()
    if max_sms_used is not None:
        comm_sms = min(comm_sms, max_sms_used)  # 不超过 DeepEP 实际需要
    set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

# 计算端: DeepGEMM (如果可用)
if has_deep_gemm() and comm_sms > 0:
    set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)
```

SM 分区通过 `VLLM_DBO_COMM_SMS` 环境变量控制, 默认 20 个 SM 给通信。在 `_run_ubatches` 和 `_capture_ubatches` 中使用:

```python
with self.sm_control:  # 进入 SM 分区模式
    return self._run_ubatches(ubatch_metadata, self.runnable)
```

---

## 8. DBO 在 Modular MoE Kernel 中的集成

### 8.1 Modular Kernel 架构

**文件**: `vllm/model_executor/layers/fused_moe/modular_kernel.py`

MoE kernel 被拆成三个组件:
1. **Prepare** (`prepare_async`): 量化 + DeepEP A2A dispatch (发送 tokens 到对应 EP rank)
2. **Experts** (`apply`): Expert GEMM 计算 (w1 matmul + activation + w2 matmul)
3. **Finalize** (`finalize_async`): DeepEP A2A combine (收回计算结果) + 权重应用 + 归约

### 8.2 Prepare 阶段的 DBO 集成

**文件**: `modular_kernel.py:1106-1192` (`_prepare`)

```python
def _prepare(self, ...):
    if not self.prepare_finalize.supports_async():
        # 非异步后端, 不能做 DBO
        assert not dbo_enabled()
        return self.prepare_finalize.prepare(...)
    else:
        # Step 1: 执行上一个 ubatch 留下的 recv hook
        dbo_maybe_run_recv_hook()

        # Step 2: 启动异步 prepare (DeepEP dispatch)
        prepare_ret = self.prepare_finalize.prepare_async(...)
        hook, receiver = prepare_ret

        # Step 3: 注册 recv hook 并让出 CPU
        if dbo_enabled():
            dbo_register_recv_hook(hook)  # 把 hook 注册给下一个 ubatch
            dbo_yield()                    # 让出, 另一个线程开始执行
        else:
            hook()  # 非 DBO 模式直接调用

        # Step 4: 等待 dispatch 完成, 获取结果
        return receiver()
```

### 8.3 DeepEP High-Throughput 的 Dispatch 实现

**文件**: `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:97-181` (`_do_dispatch`)

```python
def _do_dispatch(self, tokens, ...):
    # 1. 在 compute_stream 上捕获 event (标记当前 ubatch 的计算进度)
    previous_event = dbo_get_previous_event(self.buffer.capture)

    # 2. 让出 CPU + 切到 comm_stream
    dbo_yield_and_switch_from_compute_to_comm()

    # 3. 在 comm_stream 上执行 dispatch kernel
    (num_tokens_per_rank, ..., event) = self.buffer.get_dispatch_layout(...)
    (token_data, ..., handle, event) = self.buffer.dispatch(...)

    # 4. 记录 handle (每个 ubatch 独立!)
    a2a_idx = dbo_current_ubatch_id()
    self.handles[a2a_idx] = handle

    # 5. 切回 compute_stream 并同步
    dbo_switch_to_compute_sync()

    # 6. 返回 receiver (等待 A2A 完成的回调)
    return lambda: self._receiver(event, ...)
```

**关键**: `dbo_get_previous_event` 在 `compute_stream` 上记录一个 event, 确保 comm_stream 上的 dispatch kernel 等待当前 ubatch 的计算完成后才开始。这避免了数据竞争。

```python
# ubatching.py:193-199
def dbo_get_previous_event(func, *args, **kwargs):
    ctx = _CURRENT_CONTEXTS[_THREAD_ID_TO_CONTEXT[threading.get_ident()]]
    with torch.cuda.stream(ctx.compute_stream):  # 在 compute_stream 上!
        return func(*args, **kwargs)  # buffer.capture() 记录 event
```

### 8.4 Finalize 阶段的 DBO 集成

**文件**: `modular_kernel.py:1275-1341` (`_finalize`)

```python
def _finalize(self, output, fused_out, ...):
    # Step 1: 启动异步 finalize (DeepEP combine)
    finalize_ret = self.prepare_finalize.finalize_async(...)

    # Step 2: Shared Expert 计算与 A2A combine 重叠!
    self._maybe_apply_shared_experts(shared_experts, shared_experts_input)

    # Step 3: 注册 recv hook 并让出
    if dbo_enabled():
        dbo_register_recv_hook(hook)
        dbo_yield()
    else:
        hook()

    # Step 4: 等待 combine 完成
    receiver()
```

**Shared Expert 重叠**: 这是 DBO 的一个精妙之处。`_maybe_apply_shared_experts` 在 `finalize_async` 之后立即调用, 此时 DeepEP combine 正在 comm_stream 上执行。Shared Expert 的计算在 compute_stream 上执行, 两者自然重叠。

### 8.5 DeepEP HT 的 Finalize 实现

**文件**: `deepep_ht.py:336-397` (`_finalize`)

```python
def _finalize(self, output, fused_expert_output, ...):
    a2a_idx = dbo_current_ubatch_id()
    handle = self.handles[a2a_idx]  # 取出 dispatch 时保存的 handle

    # 1. 应用权重和归约
    fused_expert_output = weight_and_reduce_impl.apply(...)

    # 2. 捕获 event + yield + 切到 comm_stream
    previous_event = dbo_get_previous_event(self.buffer.capture)
    dbo_yield_and_switch_from_compute_to_comm()

    # 3. 在 comm_stream 上执行 combine
    combined_x, _, event = self.buffer.combine(x=fused_expert_output, handle=handle, ...)

    # 4. 切回 compute_stream (不同步)
    dbo_switch_to_compute()

    # 5. 返回 receiver: 等待 combine 完成后 copy 结果
    def _receiver():
        if event.event is not None:
            event.current_stream_wait()  # GPU 端等待
        dbo_switch_to_comm()
        output.copy_(combined_x, non_blocking=True)
        dbo_yield_and_switch_from_comm_to_compute()
    return _receiver
```

### 8.6 DeepEP Low-Latency 模式

**文件**: `deepep_ll.py:52-449`

Low-Latency 模式与 HT 模式的主要区别:
- HT (`High-Throughput`): 适合 prefill, 更大的 batch, 支持 SM 分区
- LL (`Low-Latency`): 适合 decode, 更小的 batch, 使用 batched expert 格式 (`FusedMoEActivationFormat.BatchedExperts`)
- LL 模式的 dispatch 使用 `buffer.low_latency_dispatch()`, combine 使用 `buffer.low_latency_combine()`
- LL 模式通过 `return_recv_hook=True` 获取 recv hook, 直接在 `_finalize` 中调用 `dbo_maybe_run_recv_hook()`

---

## 9. CUDA Graph 兼容性与多线程捕获

**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:202-293` (`_capture_ubatches`)

DBO 的 CUDA Graph 捕获是多线程的, 这是整个系统最复杂的部分:

### 9.1 捕获流程

```python
def _capture_ubatches(self, ubatch_metadata, model):
    # Step 1: 启动 worker 线程, 每个线程初始化 CUDA context
    for metadata in ubatch_metadata:
        thread = Thread(target=_capture_ubatch_thread, args=(results, metadata))
        thread.start()
    self.ready_barrier.wait()  # 等所有线程就绪

    # Step 2: 在 compute_stream 上开始 CUDA Graph 捕获
    cudagraph_metadata = CUDAGraphMetaData(cudagraph=torch.cuda.CUDAGraph(), ...)
    with torch.cuda.graph(cudagraph_metadata.cudagraph, stream=compute_stream, pool=...):
        ubatch_metadata[0].context.cpu_wait_event.set()  # 唤醒线程 0
        for thread in ubatch_threads:
            thread.join()  # 等所有线程完成
        # 合并结果
        sorted_results = [value for position, value in sorted(results)]
        result = _cat_ubatch_outputs(sorted_results)

    # Step 3: 保存 graph
    self.cudagraphs[num_tokens] = cudagraph_metadata
```

### 9.2 多线程 CUDA Context 初始化

**关键**: 每个 worker 线程在进入 CUDA Graph 捕获前, 必须先初始化 CUDA context:

```python
def _capture_ubatch_thread(results, ubatch_metadata):
    torch.accelerator.set_device_index(self.device)
    # 在两个 stream 上都初始化 BLAS handle
    with torch.cuda.stream(ubatch_context.compute_stream):
        _ = torch.cuda.current_blas_handle()
    with torch.cuda.stream(ubatch_context.comm_stream):
        _ = torch.cuda.current_blas_handle()
    # 进入 UBatchContext, 执行模型
    with ubatch_context:
        model_output = model(input_ids=..., ...)
```

### 9.3 Graph Replay

一旦 CUDA Graph 被捕获, replay 时完全不需要多线程或 CPU 同步:

```python
# gpu_ubatch_wrapper.py:502-511
elif num_tokens in self.cudagraphs and cudagraph_runtime_mode is CUDAGraphMode.FULL:
    cudagraph_metadata = self.cudagraphs[num_tokens]
    get_offloader().sync_prev_onload()
    cudagraph_metadata.cudagraph.replay()  # 直接 replay!
    return cudagraph_metadata.outputs
```

**为什么 replay 不需要多线程?** 因为 CUDA Graph 记录的是 GPU kernel 序列。两个 ubatch 的 kernel 已经交错记录在同一个 graph 中 (通过多线程捕获时的 ping-pong), replay 时 GPU 按记录的顺序执行, 自然实现了通信-计算重叠。

### 9.4 DBO 只支持 Full CUDA Graph

这是一个设计限制: UBatchWrapper 检查 `cudagraph_runtime_mode is CUDAGraphMode.FULL`。如果无法使用 Full CUDA Graph, 则要么退回 eager 执行 (但仍然使用 DBO 多线程), 要么完全禁用 DBO。

---

## 10. ForwardContext 管理

**文件**: `vllm/forward_context.py:128-186` (ForwardContext dataclass)
**文件**: `vllm/v1/worker/gpu_ubatch_wrapper.py:333-398` (`_make_ubatch_metadata`)

每个 ubatch 需要独立的 ForwardContext:

```python
# ForwardContext 中的 ubatch 相关字段
@dataclass
class ForwardContext:
    attn_metadata: dict | list[dict]  # list[dict] 用于 DBO, 每个 ubatch 一个
    slot_mapping: dict | list[dict]    # list[dict] 用于 DBO, 每个 ubatch 一个
    dp_metadata: DPMetadata | None     # 每个 ubatch 有自己的 DPMetadata
    ubatch_slices: UBatchSlices | None # 切分信息
```

在 `_make_ubatch_metadata` 中:
```python
for i, ubatch_slice in enumerate(ubatch_slices):
    forward_contexts.append(create_forward_context(
        attn_metadata[i],  # 切分后的 attention metadata
        self.vllm_config,
        dp_metadata=dp_metadata[i],  # 每个 ubatch 独立的 DP metadata
        ...
    ))
```

UBatchContext 在线程切换时负责恢复 forward context:
```python
def _restore_context(self):
    forward_context._forward_context = self.forward_context  # 全局变量!
```

---

## 11. DBO 全局函数注册表

**文件**: `vllm/v1/worker/ubatching.py:150-183`

DBO 通过线程局部状态 (thread ID → context ID 映射) 实现自动感知:

```python
_THREAD_ID_TO_CONTEXT: dict = {}  # {thread_id: ubatch_id}
_CURRENT_CONTEXTS: list = []      # [UBatchContext_0, UBatchContext_1]

def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)
    return wrapper
```

这样, 模型代码中的 `dbo_yield()` 等函数会自动路由到当前线程对应的 UBatchContext:

```python
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
dbo_register_recv_hook  # 直接操作 _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
```

`dbo_enabled()` 检查 `_THREAD_ID_TO_CONTEXT` 是否非空, 如果不在 DBO 上下文中, 所有 `dbo_*` 函数都是 no-op。这允许同一份模型代码在 DBO 和非 DBO 模式下运行。

---

## 12. 配置参数汇总

| 参数 | 来源 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-dbo` | ParallelConfig | False | 启用 DBO |
| `--ubatch-size` | ParallelConfig | 0 | micro-batch 数量 (0=不启用) |
| `--dbo-decode-token-threshold` | ParallelConfig | 32 | decode batch 的 DBO token 阈值 |
| `--dbo-prefill-token-threshold` | ParallelConfig | 512 | prefill batch 的 DBO token 阈值 |
| `VLLM_DBO_COMM_SMS` | envs.py | 20 | 分配给通信的 SM 数量 |
| `--all2all-backend` | - | - | 必须设为 `deepep_low_latency` 或 `deepep_high_throughput` |
| `--data-parallel-size` | ParallelConfig | 1 | 必须 > 1 |
| `--enable-expert-parallel` | ParallelConfig | False | 必须启用 |

**派生属性**:
```python
# parallel.py:505-510
@property
def use_ubatching(self) -> bool:
    return self.enable_dbo or self.ubatch_size > 1

@property
def num_ubatches(self) -> int:
    return 2 if self.enable_dbo else self.ubatch_size
```

---

## 13. 完整执行时序图

以 DeepEP HT + DBO 为例, 展示单层 MoE 的完整执行过程:

```
UBatch 0 Thread          UBatch 1 Thread          compute_stream           comm_stream
================          ================          ===============          ============
    |                         |                         |                      |
    | [A0₀ compute]           |                         | ← A0₀ kernel         |
    | [A1₀ compute]           |                         | ← A1₀ kernel         |
    | dbo_maybe_run_recv_hook |                         |                      |
    |   (无 hook, skip)       |                         |                      |
    | prepare_async:          |                         |                      |
    |   buffer.capture()      |                         | ← record event       |
    |   dbo_yield_            |                         |                      |
    |   switch_compute→comm   |                         |                      |
    ├────────────────────→    |                         |                      |
    |   (Thread 0 sleeps)     | [A0₁ compute]           | ← A0₁ kernel         |
    |                         | [A1₁ compute]           | ← A1₁ kernel         |
    |                         | dbo_maybe_run_recv_hook |                      |
    |                         |   (有 hook! 执行)       |                      |
    |                         |   → DeepEP recv_wait    | ← wait dispatch done |
    |                         | prepare_async:          |                      |
    |                         |   buffer.capture()      | ← record event       |
    |                         |   dbo_yield_            |                      |
    |                         |   switch_compute→comm   |                      |
    |                         ├────────────────────→    |                      |
    | [receiver(): wait]      | (Thread 1 sleeps)       |                      |
    | [MLP₀ GEMM]            |                         | ← MLP₀ kernels       |
    | finalize_async:         |                         |                      |
    |   SharedExperts(S₀)    |                         | ← S₀ kernels         |
    |   buffer.capture()      |                         | ← record event       |
    |   dbo_yield_            |                         |                      |
    |   switch_compute→comm   |                         |                      |
    ├────────────────────→    |                         |                      |
    |   (Thread 0 sleeps)     | [receiver(): wait]      |                      |
    |                         | [MLP₁ GEMM]            | ← MLP₁ kernels       |
    |                         | finalize_async:         |                      |
    |                         |   SharedExperts(S₁)    | ← S₁ kernels         |
    |                         |   buffer.capture()      | ← record event       |
    |                         |   dbo_yield_            |                      |
    |                         |   switch_compute→comm   |                      |
    |                         ├────────────────────→    |                      |
    |                         |                         |                      | ← D₁ dispatch
    |                         |                         |                      | ← D₀ dispatch
    |                         |                         |                      | ← C₁ combine
    |                         |                         |                      | ← C₀ combine
```

---

## 14. 文件索引

| 文件路径 | 行数 | 核心内容 |
|----------|------|----------|
| `vllm/v1/worker/ubatch_utils.py` | 266 | UBatchSlice 数据结构, 切分算法, attention metadata 切分 |
| `vllm/v1/worker/ubatching.py` | 242 | UBatchContext, 线程同步, dbo_* 全局函数 |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 528 | UBatchWrapper, SMControlContextManager, 多线程 CUDA Graph |
| `vllm/v1/worker/dp_utils.py` | 226 | DP rank 间 AllReduce 同步, micro-batching 决策 |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | 1630 | Modular MoE kernel 接口, DBO yield 点集成 |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` | 438 | DeepEP HT dispatch/combine + DBO stream 切换 |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` | 449 | DeepEP LL dispatch/combine + DBO recv hook |
| `vllm/v1/worker/gpu_model_runner.py` | ~6800 | batch 决策, ubatch 切分, UBatchWrapper 安装 |
| `vllm/forward_context.py` | 356 | ForwardContext, DPMetadata, ubatch_slices 字段 |
| `vllm/config/parallel.py` | ~530 | enable_dbo, threshold 配置, use_ubatching 属性 |
| `docs/design/dbo.md` | 89 | DBO 设计文档 |

---

## 15. 关键设计洞察

1. **不是 Pipeline Parallelism**: DBO 的 micro-batching 与 Pipeline Parallel 的 micro-batch 有本质区别。DBO 的两个 micro-batch 属于同一个 batch 的不同 token 子集, 目标是 **通信-计算重叠** (overlap), 不是流水线并行。两个 ubatch 各自独立跑完整个模型 forward。

2. **CPU 线程 + GPU Stream 的双层并行**: CPU 端通过 `threading.Event` 乒乓切换 (保证同一时刻只有一个线程执行 Python 代码), GPU 端通过 `torch.cuda.Event` + 双 stream 实现物理并行 (compute_stream 和 comm_stream 同时执行 kernel)。

3. **recv_hook 延迟执行**: `dbo_register_recv_hook` 把 A2A 的 recv callback 注册给 **下一个** ubatch (`(ctx_idx + 1) % _NUM_UBATCHES`), 由下一个 ubatch 在 `dbo_maybe_run_recv_hook()` 中执行。这确保了时序正确性: A2A send 在线程 A, A2A recv 等待在线程 B。

4. **SM 分区是可选优化**: 如果 DeepEP 的 `all2all_manager` 不支持 SM 控制, 或 DeepGEMM 不可用, `set_comm_sms` / `set_compute_sms` 是空操作 (lambda sms: None)。

5. **模型代码无感知**: 所有 DBO 逻辑通过 `dbo_*` 全局函数 + `forward_context` 全局变量注入。模型代码 (如 `FusedMoEKernelModularImpl._prepare`) 只需要调用 `dbo_yield()`, 不需要知道自己在哪个线程或哪个 ubatch 中。

6. **CUDA Graph 的多线程捕获是关键创新**: PyTorch 的 CUDA Graph 通常在单线程中捕获, 但 DBO 需要两个线程交替提交 kernel。通过 `threading.Barrier` 同步 + `with torch.cuda.graph()` 在 compute_stream 上捕获, 两个线程提交的 kernel 都被记录到同一个 graph 中。
