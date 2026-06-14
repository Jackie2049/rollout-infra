# DeepSpeed 通信-计算重叠 源码深度阅读

> 源码: deepspeed/runtime/zero/stage_1_and_2.py + stage3.py + partitioned_param_coordinator.py + zenflow/
> 核心: 4层重叠机制 → ZeRO-2 backward RS / ZeRO-3 forward AllGather / ZeRO-3 backward RS / ZenFlow optimizer step

## 1. ZeRO-2: ReduceScatter与Backward重叠

### 三个互联机制

**A. Per-Parameter Gradient Accumulation Hooks**

```python
# stage_1_and_2.py line 1050-1077
create_gradient_handling_hooks():
    # 每个param的gradient被计算时 → grad_handling_hook立即触发
    # → process_gradients(param, i) → reduce_independent_p_g_buckets_and_remove_grads()
    # → 在backward还在执行后续层时, 就开始ReduceScatter!
```

**B. IPG (Independent Partition Gradient) Bucketing**

```python
# line 110-123
@dataclass
class IPGBucket:
    buffer: List[torch.Tensor]   # 连续梯度buffer (双buffer!)
    params: List[torch.Tensor]   # 参数引用
    grads: List[torch.Tensor]    # 梯度引用
    elements: int                # 当前填充量
    index: int                   # 双buffer索引 (0 or 1)
```

`reduce_independent_p_g_buckets_and_remove_grads` (line 1091-1138):
1. 获取当前参数梯度
2. 检查是否超过`reduce_bucket_size`(默认5e8 elements)
3. **溢出时**: 立即触发`reduce_ipg_grads()` → ReduceScatter → **swap双buffer索引** (`1-index`)
4. 复制梯度到连续buffer: `bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())`

**C. Double-Buffering for Overlap (line 2352-2373)**

```python
# overlap_comm=True时: 2个连续buffer per dtype
buf_0 = torch.empty(reduce_bucket_size, dtype=dtype, ...)  # buffer 0
buf_1 = torch.empty(reduce_bucket_size, dtype=dtype, ...)  # buffer 1 (overlap时!)
# bucket 0在做ReduceScatter时 → bucket 1可以填充新梯度 → 避免race condition!
```

**D. Dedicated CUDA Stream (line 516)**

```python
self.reduction_stream = get_accelerator().Stream()  # 专用CUDA stream!

def average_tensor(self, tensor, communication_data_type):
    if self.overlap_comm:
        stream = self.reduction_stream
        stream.wait_stream(current_stream())   # 确保梯度写完成
        current_stream().wait_stream(stream)    # 确保reduce结果可用
    with get_accelerator().stream(stream):
        # ReduceScatter or AllReduce on dedicated stream
```

**E. Deferred Gradient Cleanup (line 1579-1615)**

```python
# overlap_comm=True时: 其他partition的梯度不能立即清除 (reduce还在进行中!)
# → append to self.previous_reduced_grads[comm_dtype]
# → 在independent_gradient_partition_epilogue()中清除:
if self.overlap_comm:
    get_accelerator().synchronize()  # 确保所有reduce完成
    self._clear_previous_reduced_grads()  # 批量清除
```

### ReduceScatter路径 (ZeRO-2, reduce_scatter=True)

```python
# line 1248-1324: average_tensor
# 1. 预除world_size (line 1302)
# 2. 计算rank_and_offsets → 每个梯度有目标rank和offset
# 3. 合并同rank连续切片 → 减少通信消息数
# 4. use_multi_rank_bucket_allreduce=True → coalesce到1个AllReduce+scatter
```

---

## 2. ZeRO-3 Forward: AllGather与Forward重叠

### PartitionedParameterCoordinator (trace-based prefetch)

**3阶段trace**: RECORD → COMPLETE → INVALID (动态自适应)

**fetch_sub_module (line 297-469)**:

```
1. Fetch: 异步AllGather当前submodule参数 → __all_gather_params (line 365)
         → all_gather_coalesced → 多参数合并1次AllGather → O(1)通信

2. Prefetch: 如果trace已完成 → 预取下一个submodule参数 (line 431-464)
           → bounded by prefetch_bucket_sz 和 max_n_available_params
           → 在module N计算时 → module N+1的AllGather已经在飞行!

3. Wait: 阻塞等待当前submodule的AllGather完成 (line 385)
        → current_stream().wait_stream(allgather_stream) (line 394)
```

**release_sub_module (line 472-493)**:
```
计算完成后 → partition(释放)参数:
  max_reuse_distance_in_numel → 不释放即将reuse的param
  max_n_available_params → 不超过内存预算
  ds_persist → 小param(<threshold)永不partition
```

**Event-based Backpressure (line 131-133)**:
```python
__ongoing_fetch_events: Deque[Event] = collections.deque()
__max_ongoing_fetch_events: int = 2  # 最多2个AllGather同时飞行!
# → 控制内存压力 → 防止过多未完成的AllGather
```

**Dedicated AllGather Stream (line 120)**:
```python
self.__allgather_stream: Stream = allgather_stream  # 专用CUDA stream!
# → 在default stream上计算forward → 在allgather stream上AllGather → 真正overlap!
```

---

## 3. ZeRO-3 Backward: ReduceScatter与Backward重叠

### Separate Reduction Stream (line 322-323)

```python
self.reduce_and_partition_stream = get_accelerator().Stream()  # 专用stream!
# → 和ZeRO-2的reduction_stream类似, 但名称不同
```

### IPG Bucket + Contiguous Gradient Copy (line 1381-1403)

```python
def __add_grad_to_ipg_bucket(self, param):
    # 在reduction stream上复制梯度到连续buffer:
    with get_accelerator().stream(self.reduce_and_partition_stream):
        new_grad_tensor = bucket.buffer.narrow(0, bucket.elements, param.numel()).view_as(param.grad)
        new_grad_tensor.copy_(param.grad, non_blocking=True)  # 异步复制!
        param.grad.data = new_grad_tensor  # 替换原始梯度引用
```

### ReduceScatter + Partition Pipeline (line 1407-1451)

```python
with get_accelerator().stream(self.reduce_and_partition_stream):
    # 1. ReduceScatter梯度
    grad_partitions = self.__avg_scatter_contiguous_grads(...)
    # 2. Partition梯度 (复制到本地fp32 buffer)
    self.partition_grads(params_in_bucket, grad_partitions)
    # 3. 记录完成event → backpressure
    event.record()
    self.param_reduce_events.append(event)
```

**Backpressure (line 1421-1424)**:
```python
while self.param_reduce_events and self.param_reduce_events[0].query():
    self.param_reduce_events.popleft()  # 清除已完成的
if len(self.param_reduce_events) > self.max_param_reduce_events:  # max=2!
    self.param_reduce_events.popleft().synchronize()  # 阻塞等待最早的一个
```

---

## 4. ZenFlow: Optimizer Step与Forward/Backward重叠 (第4层!)

### 概念

```
传统: fwd → bwd → optimizer_step → fwd → bwd → optimizer_step ...
ZenFlow: fwd_N+1/bwd_N+1 ‖ optimizer_step_N (CPU process) → pipeline!
```

### Double-Buffered Gradient State (line 606-615)

```python
# 2个fp32梯度buffer per optimizer subgroup
self.single_partition_of_fp32_groups[i].overlap_grad = [buffer, buffer.clone()]
# 交替使用: get_overlap_step_state()
# warmup时: micro_step & 1
# 稳定时: (micro_step // update_interval) & 1 或 zenflow_state
```

### Pipeline Pattern

```
1. Warmup (< full_warm_up_rounds): optimizer同步运行
2. 稳定: optimizer在**独立CPU进程**中异步运行:
   - zenflow_cpu_optimizer_step → pipe发送梯度到CPU optimizer进程
   - wait_last_update_and_copy → 阻塞等待**上一个**optimizer step完成
   - Pipeline: step N optimizer(CPU) ‖ step N+1 fwd/bwd(GPU)
```

### Process Management (zenflow_utils.py)

```python
# CPU optimizer在独立subprocess中运行:
def zenflow_optimizer_process(pipe, param_groups, shared_overlap_grad_map, ...):
    optimizer = ZenFlowCPUAdam(param_groups, overlap_step=True)
    # 共享内存tensor: zero-copy GPU→CPU传输
    param.overlap_grad[0].data.share_memory_()
    param.overlap_grad[1].data.share_memory_()
```

---

## 5. 与PyTorch DDP Bucketed AllReduce对比

| 维度 | PyTorch DDP | DeepSpeed ZeRO-2 |
|------|-------------|-------------------|
| Hook类型 | AccumulateGrad post-hook | Custom grad_handling_hook |
| Bucketing顺序 | 反向参数注册顺序 | 自然backward顺序(后层先入bucket) |
| 通信op | AllReduce(所有rank完整梯度) | ReduceScatter(只保留1/N) |
| Overlap机制 | 单stream, DDP在末尾同步 | 专用reduction_stream + 双buffer IPG |
| 每rank内存 | 完整梯度(所有参数) | 只有自己partition(1/N) |
| Buffer管理 | 预分配flat buffer, 无双buffer | 双buffer(overlap_comm=True) |
| 梯度清除 | AllReduce后立即清除 | Deferred到epilogue(overlap_comm=True) |

---

## 6. 架构总结: 4层重叠体系

| 层 | 机制 | Stream | Overlap目标 |
|----|------|--------|-------------|
| ZeRO-2 backward | IPG bucket + ReduceScatter | `reduction_stream` | RS与backward计算 |
| ZeRO-3 forward | PartitionedParameterCoordinator + AllGather | `__allgather_stream` | AllGather与forward计算 |
| ZeRO-3 backward | IPG bucket + ReduceScatter + partition | `reduce_and_partition_stream` | RS与backward计算 |
| ZenFlow step | 独立CPU进程 + 双buffer overlap_grad | CPU process | CPU optimizer与下一step fwd/bwd |

**共同pattern**: 双buffer(避免读写冲突) + 专用stream(GPU并发) + Event backpressure(限制2个异步操作) + trace-based prefetch(首轮学习顺序→后续预测预取)

---

## 7. 与7框架的通信重叠对比

| 框架 | 通信重叠机制 | Stream数 | Backpressure |
|------|-------------|---------|-------------|
| DeepSpeed ZeRO-2 | IPG双buffer+RS+reduction_stream | 2(default+reduction) | bucket_size阈值 |
| DeepSpeed ZeRO-3 | Trace prefetch+AllGather+RS+2专用stream | 3(default+allgather+reduce_partition) | 2 events max |
| DeepSpeed ZenFlow | CPU optimizer进程+shared_memory+双buffer | GPU+CPU(process) | warmup_rounds |
| Megatron SP | AllGather+ReduceScatter=AllReduce→无额外stream | 1(但有async_op) | N/A(等价AllReduce) |
| Megatron PP | P2PCommunicator+batched_p2p+two-group trick | 2(PP+WORLD) | Pipeline fill discipline |
| Megatron MoE overlap | ScheduleNode+每node CUDA stream+event | 多stream | stream_acquire_context |
| vLLM | 无重叠(v1 single AllReduce) | 1 | N/A |
| PyTorch DDP | 单bucket+AllReduce | 1 | bucket阈值 |
| PyTorch FSDP2 | per-param AllGather+RS | 2(default+comm) | 无(小参数快速) |
| verl | FSDP/Megatron worker → 同ZeRO/FSDP | 同backend | 同backend |
| rLLM Tinker | in-process → 无跨GPU通信 | 1 | N/A |
| DeepEP | ElasticBuffer+async_with_compute+EventOverlap | 2(default+comm) | buffer_count限制 |

---

## 8. RTX 4090影响

```
RTX 4090 + DeepSpeed overlap:
  - ZeRO-2 overlap_comm=True:
    reduction_stream → 单GPU上reduction_stream无意义(无跨GPU通信)
    → RTX 4090 单GPU → overlap_comm无用!
  - ZeRO-3:
    AllGather/RS → 需要多GPU → RTX 4090 PCIe瓶颈
    → ZeRO-3不适合RTX 4090(3Ψ通信)
  - ZenFlow:
    CPU Adam optimizer → 单GPU最有价值!
    → RTX 4090: ZeRO-2+CPU Adam+ZenFlow → 训练方案

  推荐: ZeRO-2 + CPU Adam + ZenFlow(如果可用) + LoRA
  → 内存: 4Ψ+(12+K)Ψ/N → 单GPU24GB → 7B LoRA可行
```
