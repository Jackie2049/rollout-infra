# vLLM V1 UBatch (Micro-Batching) 机制源码分析

> 文件: `vllm/v1/worker/gpu_ubatch_wrapper.py` (~528行), `vllm/v1/worker/ubatching.py` (~242行)
> 分析日期: 2026-06-04

## 核心设计目标

**通信-计算重叠 (Communication-Computation Overlap)**

在 Data Parallel (DP) 或 Expert Parallel (EP) 场景下，一次 forward 包含:
1. **计算**: Attention + FFN (GEMM)
2. **通信**: AllReduce / All-to-All (梯度同步 / MoE routing)

UBatch 将单个 batch 切成 2 个 microbatch，让通信和计算在时间上重叠:
```
Ubatch 0: [Compute] ----> [Comm] ---->
Ubatch 1:             [Compute] ----> [Comm]
                        ↑ 重叠区域
```

---

## 架构概览

```
UBatchWrapper (包装 model runnable)
    ├── UBatchContext × 2 (默认) — 每个 ubatch 的同步上下文
    │   ├── compute_stream — 计算 CUDA stream
    │   ├── comm_stream — 通信 CUDA stream
    │   ├── cpu_wait_event / cpu_signal_event — CPU 线程同步
    │   └── gpu_comm_done_event / gpu_compute_done_event — GPU stream 同步
    ├── threading.Barrier — 主线程 + ubatch 线程就绪同步
    ├── SMControlContextManager — SM 分配 (DeepEP only)
    └── CUDAGraphWrapper — CUDA Graph 捕获支持
```

---

## UBatchContext: 细粒度同步 (`ubatching.py`)

### 创建流程 (`make_ubatch_contexts`)

```python
_NUM_UBATCHES = 2  # 硬编码默认 2 个 microbatch

for i in range(num_micro_batches):
    ctx = UBatchContext(
        id=i,
        compute_stream=compute_stream,      # 共享 compute stream
        comm_stream=comm_stream,            # 共享 comm stream
        cpu_wait_event=cpu_events[i],
        cpu_signal_event=cpu_events[(i + 1) % N],  # 环形信号: i 唤醒 i+1
        gpu_comm_done_event=...,            # GPU 事件
        gpu_compute_done_event=...,
    )
```

**关键设计**: CPU 事件环形链接 — ubatch i 完成后信号唤醒 ubatch i+1，实现流水线。

### 上下文管理器协议

```python
def __enter__(self):
    self.ready_barrier.wait()       # 1. 所有线程就绪
    self.cpu_wait_event.wait()      # 2. 等待被唤醒
    self._restore_context()         # 3. 恢复 forward_context
    self.update_stream(compute_stream)

def __exit__(self, ...):
    self.cpu_signal_event.set()     # 唤醒下一个 ubatch
```

### Stream 切换原语

| 方法 | 行为 |
|------|------|
| `switch_to_comm()` | 切到 comm_stream |
| `switch_to_compute()` | 切到 compute_stream |
| `switch_to_comm_sync()` | signal compute done + wait compute done + 切 stream |
| `switch_to_compute_sync()` | signal comm done + wait comm done + 切 stream |
| `yield_()` | 让出 CPU 给下一个 ubatch |
| `yield_and_switch_from_compute_to_comm()` | 完整切换: signal → yield → wait → 切 stream |

**GPU 同步机制** (非阻塞):
```python
def _signal_compute_done(self):
    self.gpu_compute_done_event.record(self.compute_stream)

def _wait_compute_done(self):
    self.comm_stream.wait_event(self.gpu_compute_done_event)
```
使用 `torch.cuda.Event` 实现跨 stream 依赖，无需 CPU 同步。

### DBO 全局函数

```python
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
...
```

这些函数被模型代码调用（如 `custom_all_reduce`），自动检测是否在 ubatch 上下文中，如果在则执行对应的 yield/switch 操作。

**注册机制**: `_register_ubatch_function` 检查 `_THREAD_ID_TO_CONTEXT`，非 ubatch 线程调用为空操作。

---

## UBatchWrapper: 调度与执行 (`gpu_ubatch_wrapper.py`)

### 输入切分

```python
def _slice_model_inputs(self, tokens_slice, input_ids, positions, ...):
    sliced_input_ids = input_ids[tokens_slice]
    sliced_positions = positions[tokens_slice]
    # ... 同理处理 embeds 和 intermediate_tensors
```

按 `token_slice` 沿 batch dim 切分输入。每个 ubatch 处理一部分 token。

### 执行模式

```python
def __call__(self, *args, **kwargs):
    if ubatch_slices is None:
        # 无 ubatching — 直接运行
        return self.runnable(*args, **kwargs)

    if num_tokens not in self.cudagraphs and cudagraph_mode == FULL:
        # 首次 — 捕获 CUDA Graph
        return self._capture_ubatches(ubatch_metadata, self.runnable)
    elif num_tokens in self.cudagraphs and cudagraph_mode == FULL:
        # 已捕获 — 直接 replay
        return self.cudagraphs[num_tokens].cudagraph.replay()
    else:
        # 非 Graph 模式 — 普通运行
        return self._run_ubatches(ubatch_metadata, self.runnable)
```

### CUDA Graph 捕获 (`_capture_ubatches`)

**复杂度来源**: 多线程 CUDA Graph 捕获需要每个线程先初始化 CUDA context。

```python
# 1. 启动 ubatch 线程，初始化 cuBLAS handle
def _capture_ubatch_thread(results, metadata):
    torch.cuda.current_blas_handle()  # 初始化 context
    with ubatch_context:
        model_output = model(...)
    results.append((context_id, model_output))

# 2. 主线程: barrier 等待 → 开始 graph capture → 唤醒第一个 ubatch
self.ready_barrier.wait()
with torch.cuda.graph(cudagraph, stream=compute_stream):
    ubatch_metadata[0].context.cpu_wait_event.set()  # 唤醒 thread 0
    for thread in ubatch_threads:
        thread.join()  # 等待所有完成
    result = _cat_ubatch_outputs(sorted_results)
```

### 输出拼接

```python
def _cat_ubatch_outputs(sorted_results):
    if isinstance(sorted_results[0], tuple):
        # EAGLE3 speculative decoding: 多个输出张量
        return tuple(torch.cat(parts, dim=0) for parts in zip(*sorted_results))
    return torch.cat(sorted_results, dim=0)
```

按 ubatch id 排序后沿 batch dim (dim=0) 拼接。

---

## SM 控制 (`SMControlContextManager`)

```python
with self.sm_control:
    return self._run_ubatches(...)
```

进入上下文时:
- `comm_sms = VLLM_DBO_COMM_SMS` (环境变量，默认 24)
- `compute_sms = total_sms - comm_sms`
- 设置 DeepEP all2all_manager 的 SM 数量
- 设置 DeepGEMM 的 SM 数量

**目的**: 为通信 kernel 预留一部分 SM，避免通信和计算抢占 SM 导致效率下降。

仅影响:
- DeepEP high-throughput all2all
- DeepGEMM (如果启用)

---

## 数据流

```
Input Batch (N tokens)
    ↓ _slice_model_inputs
Ubatch 0 (N/2 tokens) ──┐
Ubatch 1 (N/2 tokens) ──┤ 并行执行 (threading)
    ↓ model.forward     │   内部通过 yield/switch 重叠 comm/compute
Output 0 + Output 1     │
    ↓ _cat_ubatch_outputs
Final Output (N tokens)
```

---

## 关键设计决策

1. **默认 2 个 microbatch**: `_NUM_UBATCHES = 2`，平衡重叠收益和调度开销
2. **线程数 = ubatch 数 + 1**: 主线程 + N 个 ubatch 线程
3. **环形 CPU 事件**: ubatch i 完成后唤醒 i+1，形成流水线
4. **GPU Event 跨 stream 同步**: 避免 CPU 介入，纯 GPU 硬件调度
5. **CUDA Graph 支持**: 首次捕获，后续 replay，包括多线程 graph
6. **DBO 全局函数**: 模型代码无需感知 ubatch，通过全局钩子自动切换

---

## 与 Pipeline Parallelism 的区别

| 维度 | UBatch (Micro-Batching) | Pipeline Parallelism |
|------|------------------------|----------------------|
| 切分维度 | Batch (token) | Layer / Stage |
| 目的 | 通信-计算重叠 | 模型并行 + 吞吐扩展 |
| 同步粒度 | Stream Event | P2P send/recv |
| 空泡 | 无 (完全重叠设计) | 有 (bubble) |
| 适用 | DP/EP 通信场景 | 大模型跨设备 |

---

## 代码位置参考

| 文件 | 行号 | 内容 |
|------|------|------|
| `gpu_ubatch_wrapper.py` | L113-140 | `UBatchWrapper.__init__` |
| `gpu_ubatch_wrapper.py` | L202-293 | `_capture_ubatches` |
| `gpu_ubatch_wrapper.py` | L295-331 | `_run_ubatches` |
| `gpu_ubatch_wrapper.py` | L431-527 | `__call__` 调度逻辑 |
| `ubatching.py` | L20-148 | `UBatchContext` 同步原语 |
| `ubatching.py` | L202-241 | `make_ubatch_contexts` |
