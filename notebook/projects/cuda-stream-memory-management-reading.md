# CUDA Stream Optimization & GPU Memory Management 源码级深度阅读

> 2026-06-15 | 7框架AI infra研究中的关键gap填补
> 源码: PyTorch c10/cuda/CUDACachingAllocator(5391行) + CUDAStream + torch.Stream + torch.cuda.Stream/Event; DeepSpeed ZeRO 4层overlap; Megatron PP/DDP stream; vLLM async_output_copy_stream; verl worker lifecycle; CUDA Programming Guide
> 核心: CUDA stream=GPU并发执行单位 → 专用stream实现计算-通信overlap → 双buffer避免race → Event backpressure → CachingAllocator Block/Segment/ExpandableSegment → recordStream跨stream安全 → vLLM PagedAttention block pool → DeepSpeed defragment
> ★ ★ ★ RTX 4090单GPU=stream overlap有限(无跨GPU通信) → 内存管理更重要 → PyTorch caching allocator expandable_segments=关键

---

## 目录

1. [CUDA Stream基础](#1-stream-basics)
2. [7框架Multi-Stream Pattern](#2-multi-stream-patterns)
3. [GPU Memory Management Pattern](#3-memory-management)
4. [Stream-Memory交互](#4-stream-memory-interaction)
5. [RTX 4090实战分析](#5-rtx4090)

---

## 1. CUDA Stream基础 <a id="1-stream-basics"></a>

### 1.1 什么是CUDA Stream?

```
CUDA Stream = GPU上的线性执行序列 → 一个queue of GPU operations
  → 同一stream内: 严格顺序执行 (FIFO)
  → 不同stream间: 可并发执行 (如果GPU有足够资源)

★ 关键属性:
  1. Stream是per-device的 → stream 0 on device 0 ≠ stream 0 on device 1
  2. 每个CUDA context至少有1个stream (default stream / stream 0)
  3. 操作入队是非阻塞的 → CPU立即返回 → GPU异步执行
  4. 多stream并发 → GPU SM可以同时执行来自不同stream的kernel
     → 但同一kernel内部的thread block不能跨stream!
```

### 1.2 Default Stream vs Non-Default Streams

```
Default Stream (Legacy / Stream 0):
  → 特殊行为: 与所有non-default stream有隐式同步!
  → cudaStreamLegacy = 阻塞式default stream
  → 任何non-default stream上的操作 → 等default stream完成才开始
  → 反之亦然: default stream → 等所有non-default stream完成

Non-Default Streams:
  → cudaStreamNonBlocking = 不与default stream隐式同步
  → PyTorch创建的所有non-default stream都是NonBlocking!
  → 关键: PyTorch用cudaStreamNonBlocking → 无隐式同步 → 必须显式同步!

★ ★ ★ PyTorch Stream Pool设计 (c10/cuda/CUDAStream.cpp):
  → 3个pool per device:
     Pool 0: Default stream (只有1个)
     Pool 1: Low priority (32 streams, round-robin分配)
     Pool 2: High priority (32 streams, round-robin分配)
  → kStreamsPerPool = 32 (1 << 5)
  → kDefaultFlags = cudaStreamNonBlocking → ★ 所有PyTorch non-default stream都是非阻塞!

StreamId编码 (64-bit):
  -- 54 bits --  -- 5 bits --  -- 4 bits --  -- 1 bit --
  zeros          stream index   priority type   ext/native
  → native stream: last bit = 1
  → external stream: last bit = 0 (cudaStream_t指针直接存)
  → default stream: stream_id = 0

★ round-robin分配: 第1个请求→index 0, 第2个→index 1, ...第33个→index 0(复用!)
  → 如果33个low priority streams被请求 → 第1和第33个实际上同一stream!
  → 这些streams不能并发 → 性能陷阱!
```

### 1.3 Stream同步机制

```
★ 3种同步方式:

1. cudaStreamSynchronize(stream) → 阻塞CPU等待stream完成
   → PyTorch: stream.synchronize()
   → 用途: 确保GPU工作完成后才继续CPU操作
   → 缺点: CPU stall → 破坏overlap!

2. cudaEventRecord + cudaStreamWaitEvent → GPU-side同步
   → PyTorch:
     event = torch.cuda.Event()
     event.record(stream_A)           # stream_A完成到此点时记录event
     stream_B.wait_event(event)       # stream_B未来操作等event完成
   → ★ 关键: wait_event是非阻塞CPU → 只影响GPU-side执行顺序!
   → 用途: 跨stream依赖 → 计算-通信overlap核心!

3. cudaStreamWaitStream → 等另一个stream所有已入队操作
   → PyTorch: stream_B.wait_stream(stream_A)
   → 内部: record_event(stream_A) + wait_event(stream_B)
   → 等价: stream_B等stream_A当前所有操作完成

★ ★ ★ Event选项:
  enable_timing=False → 默认 → 最快 (不记录时间戳)
  blocking=False → 默认 → wait不阻塞CPU线程
  interprocess=False → 默认 → 不能跨进程
  external=False → 默认 → CUDA graph内部用internal cross-stream dependency

  → AI框架通常: enable_timing=False + blocking=False → 最小开销
  → Profiling时: enable_timing=True → 可用event.elapsed_time(begin_event)

★ CUDA Graph中的Event:
  → external=True → 创建event_record/event_wait节点 → 跨stream依赖显式在graph中
  → external=False → internal cross-stream dependency → GPU硬件级依赖 → 更快但不可移植
  → PyTorch默认external=False → 但vLLM CUDA graph capture时需要处理
```

### 1.4 PyTorch torch.Stream vs torch.cuda.Stream

```
★ ★ PyTorch 2.12引入torch.Stream → 跨accelerator通用stream抽象!

torch.Stream (c10::Stream C++绑定):
  → 统一抽象 → CUDA/XPU/MTIA都可用torch.Stream
  → torch.accelerator.current_stream() → 返回torch.Stream
  → torch.accelerator.set_stream(stream) → 设置当前stream

torch.cuda.Stream (CUDA专用):
  → 继承torch._C._CudaStreamBase → CUDA特有功能
  → wait_event(event) → cudaStreamWaitEvent
  → wait_stream(stream) → 内部record_event+wait_event
  → record_event() → cudaEventRecord
  → synchronize() → cudaStreamSynchronize (阻塞CPU!)
  → cuda_stream属性 → 返回底层cudaStream_t指针

★ ★ 关键区别:
  torch.Stream → 跨设备通用 → 但无device-specific方法
  torch.cuda.Stream → CUDA专用 → 有cuda_stream/synchronize/wait_event

  → AI框架用torch.cuda.Stream → 需要底层cudaStream_t → NCCL/FlashInfer等
  → torch.Stream用于未来multi-accelerator场景 → XPU/MTIA/CUDA统一
```

---

## 2. 7框架Multi-Stream Pattern <a id="2-multi-stream-patterns"></a>

### 2.1 DeepSpeed ZeRO: 4层重叠体系 (最复杂!)

```
★ ★ ★ DeepSpeed是AI框架中最完整的stream overlap体系!

Layer 1: ZeRO-2 backward → ReduceScatter overlap
  专用stream: reduction_stream
  Pattern:
    default_stream: backward计算 → 每层梯度hook触发
    reduction_stream: ReduceScatter梯度 → 与backward并发!

    双buffer(IPG):
      buffer_0做ReduceScatter → buffer_1填充新梯度 → 避免race!
      bucket溢出 → swap buffer index → 1-index

    同步:
      reduction_stream.wait_stream(current_stream()) → 确保梯度写完成
      current_stream().wait_stream(reduction_stream) → 确保reduce结果可用

Layer 2: ZeRO-3 forward → AllGather overlap
  专用stream: __allgather_stream
  Pattern:
    default_stream: forward计算 → module N的forward
    __allgather_stream: AllGather module N+1参数 → prefetch!

    trace-based prefetch:
      第1轮(RECORD): 记录forward顺序 → 建立trace
      第2轮(COMPLETE): 用trace预测 → 预取下一个module
      第3轮+: INVALID → trace失效 → 重新学习

    ★ ★ 2-Event Backpressure:
      __ongoing_fetch_events: Deque[Event] = collections.deque()
      __max_ongoing_fetch_events: int = 2 → 最多2个AllGather飞行!

      while param_reduce_events and param_reduce_events[0].query():
        param_reduce_events.popleft()  # 清除已完成

      if len(param_reduce_events) > max_param_reduce_events:
        param_reduce_events.popleft().synchronize()  # ★ 阻塞等最早一个!

      → 控制内存压力 → 防止过多未完成AllGather → 每次最多2Ψ参数在GPU!

Layer 3: ZeRO-3 backward → ReduceScatter overlap
  专用stream: reduce_and_partition_stream
  Pattern: 类似ZeRO-2 → 但多一步partition(fp32 gradient copy)

  流程:
    with stream(reduce_and_partition_stream):
      1. new_grad_tensor.copy_(param.grad, non_blocking=True)  # 异步复制!
      2. ReduceScatter梯度
      3. partition_grads → copy to fp32 buffer
      4. event.record() → backpressure

Layer 4: ZenFlow → Optimizer overlap
  ★ 不用GPU stream → 用CPU subprocess!
  Pattern:
    GPU: step N+1 forward/backward
    CPU subprocess: step N optimizer → shared memory → zero-copy GPU→CPU

    双buffer overlap_grad:
      single_partition_of_fp32_groups[i].overlap_grad = [buffer, buffer.clone()]
      → micro_step & 1 → 选择buffer 0或1 → 交替使用
```

### 2.2 Megatron: PP + DDP Stream Overlap

```
★ Megatron PP overlap: pre/post hooks on separate stream

P2PCommunicator (pipeline parallelism):
  → batched_p2p_ops → 9种P2P方法 → send/recv on NCCL stream
  → two-group trick: PP group + WORLD group → 不同通信用不同process group
  → pre-hook: 上一micro-batch的recv → 在forward前开始接收
  → post-hook: 当前micro-batch的send → forward后立即发送

  ★ 1F1B Pipeline Schedule:
    warmup阶段: 只forward → 填充pipeline
    steady阶段: 1F1B → forward + backward交替 → 每步有1个forward+1个backward
    cooldown阶段: 只backward → 清空pipeline

    → ★ 气泡 = (PP_size - 1) / micro_batch_num → 更多micro_batch→更少气泡

Megatron DDP stream (param_and_grad_buffer.py):
  ★ communication_stream → 多DistOpt实例时需要

  同步Pattern:
    Compute Stream: -------------Gradient compute-------------------
    Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
    NCCL Stream:         -------RS------     -------AR------

    → communication_stream.wait_stream(current_stream()) → 等梯度计算完成
    → 然后ReduceScatter/AllReduce on communication_stream

  ★ ★ async_op关键:
    overlap_grad_reduce=True → async_op=True → NCCL返回handle → 不等完成
    overlap_grad_reduce=False → async_op=False → NCCL同步完成

    CoalescingManager: _coalescing_manager → 合并多个bucket的通信 → 1次NCCL call!

  完整代码 (param_and_grad_buffer.py line 614-634):
    if overlap_grad_reduce:
      stream_context = torch.cuda.stream(self.communication_stream)
      self.communication_stream.wait_stream(torch.cuda.current_stream())
    else:
      stream_context = nullcontext()

    with stream_context, _coalescing_manager(group, async_ops=async_op):
      for bucket in self.buckets:
        # ReduceScatter or AllReduce per bucket
```

### 2.3 vLLM V1: Async Output Copy Stream

```
★ ★ ★ vLLM V1的stream使用 = 异步D2H copy → 不是计算-通信overlap!

GPUModelRunner.__init__ (line 691-699):
  self.async_output_copy_stream: torch.cuda.Stream | None = None
  self.prepare_inputs_event: torch.Event | None = None
  if self.use_async_scheduling:
    self.async_output_copy_stream = torch.cuda.Stream()
    self.prepare_inputs_event = torch.Event()

★ ★ AsyncGPUModelRunnerOutput (line 238-289):
  → 3步异步D2H:
  1. default_stream: forward+sample → GPU-side完成
  2. async_output_copy_stream:
     async_output_copy_stream.wait_stream(default_stream)  # 等GPU完成
     sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)  # D2H异步!
     logprobs_tensors_cpu = logprobs_tensors.to_cpu_nonblocking()
     async_copy_ready_event.record()  # ★ 记录copy完成event!
  3. get_output()调用时:
     async_copy_ready_event.synchronize()  # ★ 阻塞等D2H完成
     → CPU处理output → 下一轮scheduler准备

★ ★ N-gram GPU path stream (line 838-842):
  self._num_valid_draft_tokens_event: torch.cuda.Event | None = None
  self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None
  → 同样pattern: GPU计算 → async D2H copy → event记录 → 后续同步

★ ★ ★ Pipeline Overlap (MRv2):
  V1: execute_model() = forward+sample一体 → 无overlap可能
  MRv2: execute_model() → 只forward → sample_tokens()分离
  → ★ 最后PP rank在sample step T → 第一PP rank可forward step T+1!
  → 这是stream overlap的新方向 → 不是计算-通信 → 而是forward-sample overlap!

★ 关键洞察: vLLM的stream不是overlap不同GPU操作 → 而是overlap GPU→CPU transfer!
  → inference框架的"通信"= D2H → 不是跨GPU → 异步copy是关键优化!
```

### 2.4 verl: Async Training Overlap

```
★ verl的stream使用 = 委托给backend → FSDP/Megatron/vLLM/rLLM

verl异步架构 (verl-fully-async-policy-reading.md):
  3 Ray actors: Rollouter + Trainer + MessageQueue
  → gen_loop(异步) + train_loop(异步) → asyncio

  ★ stream使用取决于backend:
    FSDP worker → PyTorch FSDP stream (default+comm)
    Megatron worker → Megatron communication_stream
    vLLM rollout → vLLM async_output_copy_stream
    rLLM Tinker → in-process → 1个stream → 无overlap

  ★ Sleep/Wake (vLLM level 1/2):
    sleep: 释放KV cache → 空出GPU内存 → trainer可用
    wake: 重新分配KV cache → rollout恢复推理
    → ★ sleep/wake不需要额外stream → 是内存管理操作!
    → vLLM level 1: 释放KV cache但保留model weights → wake快
    → vLLM level 2: 释放全部 → wake需要重新allocate → 慢

  ★ Weight Sync 4种方式:
    naive: generator→trainer零拷贝 → 无stream → 最快
    CUDA IPC: ZMQ + shared memory → 需要stream同步
    NCCL: 跨GPU → 需要NCCL stream
    NIXL: RDMA → 需要RDMA stream
```

### 2.5 PyTorch FSDP2: Per-Param AllGather + ReduceScatter

```
★ ★ PyTorch FSDP2 stream使用 → 简洁但关键!

FSDP2 (torch/distributed/fsdp/):
  → 2个stream: default_stream + comm_stream

  Forward:
    1. AllGather参数 → on comm_stream → async_op=True
    2. 等AllGather完成 → current_stream.wait_stream(comm_stream)
    3. Forward计算 → on default_stream
    4. Free参数 → 释放unsharded参数

  Backward:
    1. AllGather参数 → on comm_stream
    2. Backward计算 → on default_stream
    3. ReduceScatter梯度 → on comm_stream

  ★ vs DeepSpeed ZeRO-3:
    FSDP2: per-param AllGather → 小参数快速 → 无需trace/prefetch
    ZeRO-3: trace-based prefetch → 大参数bucket → 双buffer+backpressure

    → FSDP2更简单 → 但overlap不如DeepSpeed精细
    → DeepSpeed更复杂 → 但多GPUoverlap更好!

★ ★ ★ PyTorch CUDA Graph Stream:
  Megatron full_cuda_graph.py:
    _shared_capture_stream = torch.cuda.Stream() → 全局共享capture stream!
    → 所有full-iter和optimizer graph capture用同一stream
    → ★ per-stream alloc segments会膨胀memory_reserved → 共享stream减少pool膨胀!

  Capture流程:
    1. torch.cuda.synchronize() → 确保所有GPU工作完成
    2. capture_stream.wait_stream(torch.cuda.current_stream()) → 等default stream
    3. with torch.cuda.stream(capture_stream):
       graph.capture_begin(pool=shared_pool, capture_error_mode="thread_local")
       # ... 捕获操作 ...
       graph.capture_end()
    4. torch.cuda.current_stream().wait_stream(capture_stream) → 等capture完成
    5. graph.replay() → 在capture_stream上replay
```

### 2.6 rLLM Tinker: In-Process = 1 Stream

```
★ ★ ★ rLLM Tinker = 最简单的stream使用!

TinkerBackend:
  → in-process → 无跨GPU通信 → 无需额外stream
  → 1个default stream → 所有操作顺序执行
  → LoRA zero-copy weight sync → generator→trainer同一GPU → 无stream transfer!

  → ★ RTX 4090最优: 单GPU → stream overlap无意义(无跨GPU通信)
  → Tinker的"overlap" = CPU-side asyncio → 不是GPU stream overlap!
```

### 2.7 7框架Stream使用对比

| 框架 | 专用Stream数 | Overlap目标 | 同步方式 | Backpressure |
|------|-------------|------------|---------|-------------|
| DeepSpeed ZeRO-2 | 2(default+reduction) | RS vs backward | wait_stream双同步 | bucket_size阈值 |
| DeepSpeed ZeRO-3 fwd | 3(default+allgather+reduce_partition) | AllGather vs forward | wait_stream+2-event | max=2 events |
| DeepSpeed ZeRO-3 bwd | 3 | RS vs backward | wait_stream+event | max=2 events |
| DeepSpeed ZenFlow | GPU+CPU(process) | CPU optim vs GPU fwd/bwd | shared memory semaphore | warmup_rounds |
| Megatron DDP | 2(default+communication) | RS/AR vs gradient compute | wait_stream+coalescing | bucket_num |
| Megatron PP | 2(PP+WORLD NCCL) | P2P vs pipeline compute | async_op+handle | pipeline discipline |
| vLLM V1 | 2(default+async_copy) | D2H vs GPU forward/sample | Event record+synchronize | none |
| vLLM MRv2 | 2 | forward vs sample pipeline | two-phase execution | pipeline natural |
| verl | 委托backend | 委托backend | 委托backend | 委托backend |
| PyTorch FSDP2 | 2(default+comm) | AllGather/RS vs compute | wait_stream | none |
| rLLM Tinker | 1 | none(单GPU) | none | none |
| DeepEP | 2(default+comm) | AllToAll vs compute | EventOverlap | buffer_count |

---

## 3. GPU Memory Management Pattern <a id="3-memory-management"></a>

### 3.1 PyTorch Caching Allocator: 核心引擎

```
★ ★ ★ PyTorch CUDA内存管理 = CUDACachingAllocator (C++, 5391行)

核心数据结构:
  Block:
    device: DeviceIndex
    ptr: void* → GPU内存地址
    size: size_t → block大小
    requested_size: size_t → 用户请求大小(可能<size, 因为rounding)
    stream: cudaStream_t → ★ 分配时所在的stream!
    stream_uses: stream_set → ★ 在哪些stream上被使用过!
    pool: BlockPool* → 所属pool (large/small/private)
    prev, next: Block* → split后的前后block
    event_count: int → ★ outstanding CUDA events数量!
    expandable_segment_: ExpandableSegment* → ★ expandable segment指针

  BlockPool:
    blocks: std::set<Block*> → 按size排序 (best-fit查找)
    blocks_by_addr: std::set<Block*> → 按地址排序 (地址查找)
    unmapped: std::set<Block*> → 未映射的expandable block
    owner_PrivatePool: PrivatePool* → CUDA graph用

  Segment:
    → cudaMalloc分配的连续内存 → 包含多个Block
    → large pool: >=2MB → 直接cudaMalloc
    → small pool: <2MB → 从2MB buffer切分

★ ★ 分配策略 (malloc函数, line 1700+):

  1. prepare_for_malloc:
     → process_events() → 检查跨stream使用是否完成 → 回收内存!
     → ★ CUDA graph capture时: skip process_events → cudaEventQuery illegal!

  2. round_size:
     → <2MB: 不round → small pool
     → 2MB~10MB: 分配20MB → split → 减少fragmentation!
     → >10MB: round到最近2MB → large pool
     → max_split_size以上: 不允许split → 防止碎片化!

  3. get_free_block:
     → best-fit查找 → 找最小够大的free block
     → ★ 同stream优先! → freed block可在同stream立即reuse → 无需同步
     → 跨stream: 必须等recordStream events完成 → 才能reuse

  4. alloc_block (cudaMalloc):
     → 如果free block不够 → cudaMalloc新segment
     → ★ ★ OOM处理链:
       a. garbage_collect_cached_blocks → GC old blocks (threshold控制)
       b. try_mempool_fallback → 尝试overflow pool (use_on_oom=True)
       c. release_available_cached_blocks → 释放free cached blocks → retry
       d. release_cached_blocks → 释放所有non-split cached blocks → retry
       → 全部失败 → CUDA OOM!

★ ★ 释放策略 (free函数):

  1. 释放block → 如果prev/next是free → merge → 减少碎片
  2. 如果block有stream_uses → 不能立即reuse!
     → insert_events → 为每个stream_use创建CUDA event
     → block.event_count = stream_uses数量
     → 等所有event完成 → event_count降到0 → free_block

  3. free_block:
     → 如果size >= max_split_size → 直接cudaFree → 返还系统
     → 否则 → 放回BlockPool → 等待reuse
```

### 3.2 PyTorch Caching Allocator: Expandable Segments

```
★ ★ ★ Expandable Segments (v2.0+默认开启) → 解决fragmentation!

问题:
  传统allocator: 每次分配cudaMalloc → 释放后可能形成slivers
  → batch size N → 分配(N*A)和(N*A*B) → 切换到N+1 → 需要(N+1)*A和(N+1)*A*B
  → 已有N*A segment可以容纳部分(N+1)*A → 但不完美 → 留slivers
  → 50+层 → 50+次slivers → ★ 碎片化灾难!

解决:
  ExpandableSegment → 初始分配少量 → 需要时扩展 → 不创建slivers!
  → 1个segment per stream → 所有同stream分配在同一segment → tile nicely

  ★ CUDA Low-Level Memory API (类似mmap):
    cuMemAddressReserve → 分配虚拟地址空间(256TiB!)
    cuMemCreate → 分配物理内存(GPU page = 2MiB small, 20MiB large)
    cuMemMap → 映射物理内存到虚拟地址
    cuMemSetAccess → 设置访问权限
    cuMemUnmap → 取消映射 → 返还物理内存

  实现:
    → 初始: 映射少量pages → 满足当前需求
    → 扩展: 需要更多 → cuMemMap更多pages → append到segment尾部
    → OOM: cuMemUnmap empty pages → 返还给CUDA → 减少压力
    → ★ 填gap: OOM后unmap的pages → gap → 新分配时优先填gap → 防止碎片!

  ★ ★ RTX 4090影响:
    expandable_segments=True → 默认开启 → PyTorch 2.12+
    → 对inference特别重要 → batch size变化 → 碎片化严重
    → → ★ vLLM decode batch size动态 → expandable_segments减少碎片!
    → 小成本: 初次分配稍慢 → IPC不支持 → 但inference不需要IPC

  限制:
    IPC不支持 → cross-process tensor sharing不可用
    → multiprocessing DataLoader不兼容 → 需暂时disable
    CUDA runtime peer access不工作 → 需allocator手动enablePeerAccess
```

### 3.3 PyTorch MemPool: CUDA Graph内存隔离

```
★ ★ torch.cuda.MemPool → 内存池隔离 → CUDA Graph必需!

class MemPool(_MemPool):
  → allocator: _cuda_CUDAAllocator → 自定义分配器
  → use_on_oom: bool → OOM时是否fallback到这个pool → 默认False
  → no_split: bool → 不split segment → 默认False

使用:
  pool = torch.cuda.MemPool()
  with torch.cuda.use_mem_pool(pool):
    # 所有分配进入pool → 与默认pool隔离!

  → ★ CUDA Graph capture必须用private pool → 保证地址不变!
  → graph_pool_handle() → 返回opaque token → CUDAGraph.capture_begin(pool=token)
  → ★ replay时地址必须与capture一致 → private pool保证这一点!

★ ★ no_split选项:
  → no_split=True → segment不切分 → 分配地址连续 → 避免碎片
  → 代价: 内存利用率低 → 但CUDA graph需要固定地址 → 必须no_split!

PrivatePool (C++):
  → id: MempoolId_t → pool标识
  → use_count: int → 引用计数 → graph销毁时减1
  → cudaMalloc_count: int → cudaMalloc次数 → 0时可删除pool
  → large_blocks, small_blocks → 私有BlockPool → 与默认pool隔离
  → ★ ★ graph销毁后 → pool内存才能返还系统!
```

### 3.4 vLLM KV Cache Block Pool: PagedAttention

```
★ ★ ★ vLLM KV cache内存管理 = BlockPool → PagedAttention核心!

BlockPool (vllm/v1/core/block_pool.py):
  blocks: list[KVCacheBlock] → 所有block → block_id 0~N-1
  free_block_queue: FreeKVCacheBlockQueue → ★ 自定义双向链表!
  cached_block_hash_to_block: BlockHashToBlockMap → prefix caching hash

★ ★ FreeKVCacheBlockQueue (自定义双向链表,非deque):
  → fake head/tail sentinel → 减少分支 → O(1)中间删除
  → 排序: LRU → 最久未用在前 → 新释放的在后
  → popleft_n(num_blocks) → 分配N个block → 从头部取 → ★ LRU优先淘汰!
  → append_n(blocks) → 释放 → 逆序添加

★ KVCacheBlock (@dataclass(slots=True)):
  block_id: int → 0 ~ num_gpu_blocks-1
  ref_cnt: int = 0 → 引用计数 → prefix共享时>1!
  _block_hash: BlockHashWithGroupId | None → prefix缓存hash
  prev_free_block, next_free_block → 双向链表指针
  is_null: bool → null_block占位符(block_id=0)

★ ★ Block分配/释放:
  get_new_blocks(n):
    → 从free_queue头部取n个 → LRU优先淘汰!
    → _maybe_evict_cached_block → 如果在prefix hash → 移除hash entry
    → ref_cnt += 1

  free_blocks(blocks):
    → ref_cnt -= 1 → ref_cnt==0 → append到free_queue尾部

  touch(blocks):
    → ref_cnt += 1 → ref_cnt从0到1 → 从free_queue移除 → ★ prefix命中保护!

★ ★ Block Size计算:
  默认block_size=16 tokens (CacheConfig.block_size)
  page_size_bytes per block per layer = 2 * block_size * num_kv_heads * head_size * dtype_size
  → INT8KV → dtype_size=1 → ★ 省一半内存!
  → GQA-8 → num_kv_heads=8 → ★ 省更多!

  num_blocks = available_memory // page_size_bytes // num_layers
  → 所有worker取最小值 → 统一block数量!

★ ★ vs PyTorch Caching Allocator:
  vLLM: 固定大小block → 无split → 无碎片 → 但浪费固定block_size空间
  PyTorch: 动态大小block → 有split → 有碎片 → 但利用率高

  → ★ inference用固定block → 简洁可控 → 无碎片化问题!
  → ★ training用动态分配 → 灵活 → 但需要GC/defrag!
```

### 3.5 ZeRO Offloading: CPU/GPU Memory Ping-Pong

```
★ ★ ★ ZeRO offloading = CPU/GPU内存ping-pong → 专用stream传输!

ZeRO-Offload (stage_1_and_2.py + offload_config.py):
  → optimizer state offload到CPU → CPU Adam → 省GPU内存!
  → gradient offload: GPU→CPU → non_blocking=True → async D2H!

  ★ offload stream:
    → GPU→CPU传输: pinned memory → non_blocking=True → 异步DMA!
    → CPU→GPU传输: optimizer结果回传 → 也non_blocking=True
    → → ★ 不需要专用stream! → pinned memory + non_blocking → DMA引擎处理!

  ★ ★ Pinned Memory (cudaMallocHost/cudaHostRegister):
    → 页锁定内存 → DMA直接传输 → 不经过CPU copy!
    → 比普通malloc快2-3x → 但消耗系统RAM → 不能oversubscribe!
    → DeepSpeed: gradient buffer用pinned memory → 异步offload!

  ★ NVMe Offload (ZenFlow NVMe fix):
    → CPU RAM不够 → optimizer state offload到NVMe SSD!
    → 异步I/O → POSIX AIO → overlap GPU计算和NVMe I/O
    → → ★ 专用I/O线程 → 不是CUDA stream → 是CPU线程!

ZeRO-Infinity:
  → 3层内存: GPU → CPU RAM → NVMe SSD
  → 每层有独立的allocation和释放策略
  → 参数: GPU→forward → 释放 → CPU→backward → 释放 → NVMe→optimizer
  → ★ ping-pong: 参数在不同层之间来回传递!
```

### 3.6 Memory Fragmentation: DeepSpeed Defragment + PyTorch GC

```
★ ★ ★ 内存碎片是AI training的隐形杀手!

PyTorch Caching Allocator Fragmentation:
  问题来源:
    → 动态batch size → 分配不同大小 → 释放后留下gap
    → 50+层 → 50+种大小 → 交叉分配释放 → 碎片化累积
    → max_split_size以上block → 不split → 大块浪费

  ★ ★ 解决方案1: Expandable Segments (v2.0+):
    → 1 segment per stream → 所有分配tile在同一segment → 减少slivers
    → → ★ vLLM inference → expandable_segments=True → 默认开启

  ★ ★ 解决方案2: garbage_collect_cached_blocks:
    → GC阈值: garbage_collection_threshold × total_memory
    → 超过阈值 → free blocks exceeding avg age → 释放旧block
    → → age = gc_count() = pool.get_free_blocks_call_count - gc_count_base
    → → 按age排序 → 释放age > avg的block → 减少碎片

  ★ ★ 解决方案3: empty_cache:
    → release_cached_blocks → cudaFree所有free block → 返还给CUDA
    → → ★ extreme fragmentation时 → empty_cache重新分配 → 消除所有碎片
    → → 代价: 下次分配需要重新cudaMalloc → 慢!

DeepSpeed Defragment (v0.19.0+, utils.py line 207-235):
  ★ ★ 33行精简但完整的defragment!

  def defragment(tensors):
    → 步骤:
    1. 创建CPU flat buffer → sum of all tensor sizes
    2. 逐个copy tensor到CPU buffer → narrow + copy_
    3. tensor.data = torch.empty(0) → ★ 释放GPU内存!
    4. gc.collect() → Python垃圾回收 → 释放引用
    5. get_accelerator().empty_cache() → ★ PyTorch empty_cache → 返还GPU内存!
    6. device_buffer = cpu_buffer.to(orig_device) → ★ 重新分配 → 连续!
    7. 恢复tensor.data = device_buffer.narrow(offset, size) → 指向新位置

  ★ ★ 关键洞察:
    → GPU→CPU→empty_cache→CPU→GPU → 完全消除碎片!
    → 代价: 2次全量GPU↔CPU传输 → 适合step间执行 → 不适合forward/backward内
    → ds_tensor碎片 → ZeRO-3 partitioned参数 → gather/release循环 → 碎片累积

  适用场景:
    → 长时间训练 → 碎片化累积 → 定期defragment
    → 大模型 → ds_tensor大 → 碎片化影响严重
    → CPU/NVMe offload → GPU内存更紧张 → 碎片更致命
    → DeepCompile → 编译图改变参数生命周期 → 需要灵活内存管理

★ ★ vLLM Block Pool Fragmentation:
  → ★ vLLM没有碎片化问题! → 固定大小block → 无split → 无碎片!
  → 只有prefix caching eviction → block回到free_queue → 可reuse
  → ★ vLLM V1的问题不是碎片 → 而是preemption(full reset) → 丢失所有KV!
```

### 3.7 CUDA Memory Pools: cudaMemPoolCreate

```
★ CUDA 11.2+引入cudaMemPool_t → stream-ordered memory allocation!

cudaMemPoolCreate → 创建memory pool → 与stream关联
cudaMallocAsync → 从pool异步分配 → stream-ordered → 不需要同步!
cudaFreeAsync → 异步释放 → stream-ordered → 安全!

★ ★ 与PyTorch Caching Allocator的区别:
  CUDA MemPool:
    → 硬件级 → GPU driver管理 → 线程安全 → 低开销
    → stream-ordered → free在同一stream上 → 无需event同步!
    → 但不能跨stream reuse → 需要cudaMemPoolExportShareableHandle

  PyTorch Caching Allocator:
    → 软件级 → C++实现 → 递归mutex → 跨线程安全
    → 不stream-ordered → recordStream()显式管理跨stream reuse
    → 更灵活 → 但开销更大 → 需要process_events

  ★ ★ 当前: PyTorch Caching Allocator仍是主流 → cudaMemPoolAsync作为可选backend
    → CUDAMallocAsyncAllocator.cpp → PyTorch的cudaMemPool实现
    → 但expandable_segments只在CachingAllocator中实现!

★ ★ RTX 4090 (SM 89):
    cudaMemPool → ✓ → CUDA 11.2+ → Ada支持
    但: PyTorch默认用CachingAllocator → 不是cudaMemPool
    → expandable_segments = CachingAllocator feature → ★ RTX 4090可用!
```

---

## 4. Stream-Memory交互 <a id="4-stream-memory-interaction"></a>

### 4.1 Stream影响内存可见性

```
★ ★ ★ CUDA内存模型 → stream是最重要的同步单位!

CUDA Memory Visibility规则:
  1. 同一stream内: 所有操作顺序执行 → 内存自动可见 → 无需同步
  2. 不同stream间: ★ 无隐式同步! → 内存修改可能不可见!
     → 需要显式同步: event或stream_wait
  3. Default stream: ★ 与所有non-blocking stream无隐式同步!
     → PyTorch non-default streams都是NonBlocking → 无隐式同步!

★ ★ ★ PyTorch Caching Allocator的recordStream → 跨stream安全的核心!

recordStream(Block* block, cuda::CUDAStream stream) (line 2530):
  if stream.stream() == block.stream:
    return  # 同stream → 不需要同步 → 忽略!
  block.stream_uses.insert(stream)  # ★ 记录stream use!

  → ★ ★ 意义: block在stream_A分配 → 在stream_B使用 → 释放后不能立即reuse!
  → → 需要等stream_B完成使用 → recordStream记录 → insert_events → CUDA event同步!

insert_events(Block* block):
  → 为每个stream_use创建CUDA event → event.record(stream_use)
  → block.event_count = stream_uses数量
  → cuda_events[stream].append(event, block)
  → ★ 等所有event完成 → event_count降到0 → free_block → 可reuse!

★ ★ ★ ★ 内存安全时序:

  分配block on stream_A → block.stream = stream_A
  使用block on stream_B → recordStream(stream_B) → block.stream_uses = {stream_B}
  释放block → insert_events → event_B on stream_B
  下一分配 → get_free_block → block.event_count > 0 → 不能reuse!
  → process_events → cudaEventQuery(event_B) → 如果not ready → skip
  → → 等stream_B完成 → event_B query → success → event_count -= 1 → 0 → free_block → reuse!

★ ★ ★ 跨stream内存reuse的3种安全方式:
  1. stream_B.wait_stream(stream_A) → 确保stream_A写完成 → stream_B可读
  2. recordStream(stream_B) → allocator自动管理 → 等stream_B完成 → block可reuse
  3. 同stream → 无需任何同步 → 最快!
```

### 4.2 CUDA Graph中的Stream-Memory交互

```
★ ★ ★ CUDA Graph capture → 内存管理必须特殊处理!

Capture阶段:
  → notifyCaptureBegin → 创建PrivatePool → 分配进入private pool
  → 所有分配地址固定 → replay时必须一致!
  → ★ ★ process_events被skip! → cudaEventQuery illegal during capture!
  → → 跨stream block不能在capture时回收 → deferred到capture结束

  ★ graph_capture_record_stream_reuse=True (新feature):
    → free_safe_blocks_in_capture → 在capture中检查是否有安全reuse的block
    → → 只有当block的所有stream_use在capture开始前已完成 → 才安全!

  ★ ★ ★ Private Pool关键:
    → capture时的所有分配 → 进入private pool → 与默认pool隔离
    → high-water mark的内存 → 被private pool保留 → 直到graph销毁!
    → → ★ vLLM CUDA graph pool → 全局共享 → 多个graph用同一pool → 减少膨胀!

Replay阶段:
  → graph.replay() → 在capture_stream上执行 → 所有操作一次性
  → → ★ 不需要stream同步 → 所有操作在同一个graph内!
  → 但: 如果graph操作需要等外部stream → ★ 需要external event!

  → capture_stream.wait_stream(default_stream) → 等输入准备
  → default_stream.wait_stream(capture_stream) → 等graph完成
```

### 4.3 实战: 何时可以安全reuse内存?

```
★ ★ ★ ★ 4种场景的安全reuse判断:

1. 同stream顺序释放-分配:
   stream_A: use(block) → free(block) → allocate(新block)
   → ★ 安全! → 同stream内顺序执行 → free完成后allocate → 无race!

2. 不同stream写-读:
   stream_A: write(block) → event_A.record()
   stream_B: wait_event(event_A) → read(block)
   → ★ 安全! → event确保stream_A写完成 → stream_B才读

3. 不同stream释放-分配:
   stream_A: free(block) → allocator insert_event(stream_A)
   stream_B: allocate → allocator process_events → check event_A
   → ★ 安全! → allocator等stream_A完成 → 才给stream_B reuse!
   → → 但: 如果process_events没调用 → ★ 不安全! → 旧数据可能被覆写!

4. ★ ★ ★ 不安全的场景:
   stream_A: write(block)
   stream_B: write(block) → ★ 无同步!
   → data race → 结果不可预测 → ★★ 严禁!

★ ★ ★ AI框架中的典型安全Pattern:

  DeepSpeed ZeRO-3:
    → AllGather on __allgather_stream → 参数gather完成
    → current_stream().wait_stream(allgather_stream) → 确保参数可用
    → forward on default_stream → 使用gathered参数 → 安全!
    → release_sub_module → partition参数 → 内存释放
    → ★ 同一参数: allgather_stream写 → default_stream读 → 有同步 → 安全!

  vLLM async output:
    → default_stream: forward+sample → GPU result
    → async_copy_stream: wait_stream(default_stream) → 确保GPU完成
    → D2H copy → CPU result → ★ GPU result此时只被copy_stream读 → 安全!
    → copy_stream完成后 → GPU result释放 → CPU result使用 → 安全!

  Megatron DDP:
    → default_stream: gradient compute
    → communication_stream: wait_stream(default_stream) → 确保梯度写完成
    → ReduceScatter → ★ gradient此时只被comm_stream读 → 安全!
```

---

## 5. RTX 4090实战分析 <a id="5-rtx4090"></a>

### 5.1 SM89 Stream能力

```
★ RTX 4090 (Ada Lovelace, SM 8.9) Stream能力:

CUDA Stream: ✓ → 完全支持 → 所有stream feature可用
  → Non-blocking stream: ✓ → PyTorch cudaStreamNonBlocking
  → Stream priority: ✓ → 2级priority (high/low)
  → Stream callback: ✓ → cudaStreamAddCallback
  → Multi-stream并发: ✓ → 128 CUDA cores/SM × 128 SM → 多stream可并发

★ ★ ★ 但: RTX 4090单GPU → stream overlap的价值有限!

原因:
  → 单GPU → 所有stream在同一GPU执行 → 共享同一SM资源
  → compute-bound kernel → 占满所有SM → 其他stream无法获得SM → 无法并发!
  → memory-bound kernel → 只用少量SM → 其他stream可获得SM → 可以并发!

  → ★ ★ RTX 4090 decode = memory-bound → 95.1% weight reads → 4.9% compute
  → → ★ decode可以和其他memory-bound stream并发!
  → → ★ 但: 单GPU inference → 只有1个stream → 无并发需求!

  → ★ ★ RTX 4090 training:
  → ZeRO-3 → 需要多GPU → PCIe scaling灾难 → 不可行!
  → ZeRO-2 → 单GPU → 无跨GPU通信 → reduction_stream无意义!
  → → ★ RTX 4090最优 = LoRA + CPU Adam → 无stream overlap → 1个stream!
```

### 5.2 Memory Bandwidth和Stream并发

```
RTX 4090 Memory:
  HBM: 24 GB → 890.8 GB/s → GDDR6X
  PCIe 4.0 x16: ~32 GB/s (bidirectional ~64 GB/s)

  ★ ★ 关键比例:
  HBM带宽 / PCIe带宽 = 890.8 / 32 = ~28x
  → GPU内部操作28x快于跨GPU传输!
  → → ★ PCIe传输是瓶颈 → 多GPU通信 = 灾难!

★ ★ Stream并发在实际中的作用:

  1. 单GPU inference:
     → 所有compute在default_stream → 无并发需求
     → async_output_copy_stream → D2H copy → 与GPU decode并发?
     → → decode=memory-bound → copy也memory-bound → 共享HBM带宽!
     → → ★ 实际并发收益小 → 但减少CPU stall!

  2. 单GPU training (ZeRO-2 + LoRA + CPU Adam):
     → 无跨GPU通信 → reduction_stream无用
     → CPU offload stream → D2H/H2D → 与GPU compute并发?
     → → ★ 有收益! → GPU backward → CPU optimizer async → overlap!
     → → → ZenFlow pattern → CPU optimizer subprocess → 省GPU时间!

  3. ★ ★ 最佳RTX 4090 stream pattern:
     → default_stream: GPU forward/backward
     → 无额外stream → 无跨GPU通信
     → CPU-side: asyncio → verl/rLLM → gen_loop+train_loop overlap
     → → ★ CPU-side overlap = RTX 4090真正的overlap路径!
```

### 5.3 PCIe Transfer Stream考量

```
★ ★ ★ RTX 4090多GPU = PCIe → stream传输考量:

1. PCIe P2P Access:
   → RTX 4090无NVLink → P2P只能通过PCIe
   → cudaMemcpyAsync Peer → PCIe带宽 ~32 GB/s → 远慢于NVLink!
   → → ★ NCCL on PCIe → ring/allreduce → 带宽共享 → 更慢!

2. PCIe Stream并发:
   → PCIe是共享总线 → 所有GPU pair共享带宽!
   → 8 GPU → 7对PCIe传输 → 每对~32/7 ≈ 4.6 GB/s → ★ 灾难!
   → → 不能overlap → 带宽共享 → 总吞吐不增加!

3. ★ ★ 结论: RTX 4090 PCIe stream → 几乎无用!
   → 7B模型 ZeRO-3 8GPU → 0.46x scaling → 反比单GPU慢!
   → → RTX 4090最优 = 单GPU → LoRA + CPU Adam → 无PCIe通信!
```

### 5.4 单GPU Stream优化策略

```
★ ★ ★ ★ RTX 4090单GPU最优stream策略:

1. Inference:
   → 1个default_stream → 全部decode/prefill
   → INT4 + INT8KV + GQA-8 → 4,791 tok/s → memory-bound
   → EAGLE spec decode → 9,088 tok/s → 多步forward → 同stream顺序执行
   → async_output_copy_stream → D2H异步 → ★ 减少CPU stall → +5-10%吞吐
   → → ★ vLLM use_async_scheduling=True → 必须开!

2. Training (GRPO + LoRA):
   → 1个default_stream → GPU forward/backward
   → CPU optimizer → 不需要GPU stream → CPU-side overlap!
   → → ★ rLLM Tinker → in-process → fused fwd-bwd-optim → 1个stream!
   → → ★ verl HYBRID → 同进程 → 1个stream → CPU asyncio overlap!

   → stream优化关键:
     a. 确保所有GPU操作在同一stream → 避免不必要的跨stream同步
     b. pinned memory → D2H/H2D非阻塞 → DMA引擎处理 → 不需要额外stream!
     c. CPU Adam → CPU-side计算 → 与GPU并发 → ★ 不需要GPU stream!

3. ★ ★ 内存管理优化 (更重要!):
   → expandable_segments=True → ★ 减少碎片 → inference特别重要
   → block_size=16 → vLLM KV cache → 固定block → 无碎片
   → INT4 + INT8KV → 7B model → ~11GB → 13GB headroom → OOM安全
   → empty_cache → 碎片严重时 → 但代价是重新分配 → 步间使用!
   → garbage_collection_threshold → 定期GC → PyTorch 2.12默认行为

4. ★ ★ ★ 禁忌:
   → ZeRO-3 → 多GPU AllGather → PCIe灾难 → ❌
   → NCCL通信stream → 单GPU无通信 → ❌
   → 多GPU PP stream → PCIe瓶颈 → ❌
   → DeepEP → SM90 only → ❌
   → NVLS → Hopper only → ❌
   → TMA → SM90 only → ❌
```

### 5.5 RTX 4090决策树: Stream vs Memory

```
★ ★ ★ ★ 3秒决策树:

问: 需要GPU stream overlap吗?
  → 单GPU? → ✗ → 无跨GPU通信 → stream overlap无收益
    → → 改为CPU-side overlap → asyncio/多线程!
  → 多GPU PCIe? → ✗ → 带宽瓶颈 → overlap增加延迟
    → → 改为单GPU + LoRA + CPU Adam!
  → 多GPU NVLink? → ✓ → stream overlap有意义 → 但RTX 4090没有NVLink!

问: 内存管理优先级?
  → inference? → expandable_segments=True → INT4 → 固定block → 无碎片
  → training? → ZeRO-2 + LoRA → CPU optimizer → 17GB → 可行
    → → 碎片? → 定期empty_cache或gc → 步间执行
  → OOM? → expandable_segments=True → garbage_collection_threshold → use_on_oom pool

★ ★ ★ ★ RTX 4090最优配置总结:

Inference:
  vLLM V1 + INT4 + INT8KV + GQA-8 + expandable_segments=True
  + use_async_scheduling=True (async_output_copy_stream)
  → 4,791 tok/s → EAGLE → 9,088 tok/s

Training:
  rLLM Tinker + GRPO + LoRA(rank=32) + CPU Adam + BF16
  + expandable_segments=True → in-process → 1个GPU stream
  → ~17GB → 24GB GPU → 7GB headroom → 安全
```

---

## 6. 跨框架Pattern总结

### 6.1 6大通用Stream-Memory Pattern

```
★ ★ ★ Pattern 1: 专用Stream + 双Buffer + Event Backpressure
  → DeepSpeed ZeRO系列 → 最完整实现
  → 关键: reduction_stream/allgather_stream + IPG双buffer + 2-event max
  → 适用: 多GPU AllReduce/AllGather overlap
  → RTX 4090: ✗ → 单GPU无通信

★ ★ Pattern 2: Async D2H Copy Stream + Event Synchronize
  → vLLM V1 → async_output_copy_stream
  → 关键: GPU→CPU non_blocking copy → event record → 后续同步
  → 适用: inference输出异步传输 → 减少CPU stall
  → RTX 4090: ✓ → 单GPU也受益 → +5-10%吞吐

★ ★ Pattern 3: Communication Stream + Coalescing
  → Megatron DDP → communication_stream + _coalescing_manager
  → 关键: 多bucket合并1次NCCL → 减少launch开销
  → 适用: 多GPU梯度通信overlap
  → RTX 4090: ✗ → 单GPU无通信

★ ★ ★ Pattern 4: CPU-side Async + GPU Stream分离
  → verl/rLLM → asyncio gen_loop + train_loop → GPU 1个stream
  → 关键: GPU只做forward/backward → CPU异步调度 → overlap在CPU
  → 适用: 单GPU RL training → GRPO rollout+train overlap
  → RTX 4090: ✓ → ★ 最优路径!

★ Pattern 5: CUDA Graph Private Pool + Shared Stream
  → Megatron/vLLM → graph_pool_handle() → private pool隔离
  → 关键: capture时分配进private pool → replay地址一致 → shared pool减少膨胀
  → 适用: decode CUDA graph → 固定batch → 消除launch开销
  → RTX 4090: ✓ → INT4+CUDA graph → 3.4x+10%加速

★ ★ Pattern 6: Expandable Segment + Fragmentation Mitigation
  → PyTorch Caching Allocator → expandable_segments=True
  → 关键: 1 segment per stream → 动态扩展 → 无slivers
  → 适用: dynamic batch inference → 碎片化严重场景
  → RTX 4090: ✓ → ★ inference必须开启!
```

### 6.2 何时需要多Stream? 何时单Stream够用?

```
★ ★ ★ ★ 关键判断:

需要多Stream:
  1. 多GPU + NVLink → 通信overlap → 2+ stream
  2. 计算-通信overlap → AllGather/ReduceScatter → 专用stream
  3. 异步D2H → inference输出 → async_copy_stream
  4. Pipeline parallel → PP P2P → send/recv stream
  5. MoE AllToAll → DeepEP → 2 stream (default+comm)

单Stream够用:
  1. ★ ★ ★ 单GPU → 无跨GPU通信 → 1 stream
  2. LoRA + CPU Adam → 无GPU-side overlap → CPU-side async
  3. CUDA Graph → 所有操作在graph → 同一stream replay
  4. rLLM Tinker → in-process → 1 stream最优

★ ★ ★ 核心洞察:
  → Stream的价值 = overlap不同类型的GPU操作
  → 单GPU: compute-bound时 → 其他stream无法获得SM → overlap无效!
  → 单GPU: memory-bound时 → 其他stream可获得SM → 但HBM带宽共享!
  → → ★ 单GPU → stream overlap收益极小 → 内存管理更重要!
  → → ★ 多GPU NVLink → stream overlap是性能关键 → DeepSpeed/Megatron核心!
  → → ★ RTX 4090 → 单GPU最优 → 内存管理 > stream优化!
```

---

## 7. 源码索引

```
★ 关键源码文件:

PyTorch:
  c10/cuda/CUDACachingAllocator.cpp (5391行) → Block/Segment/ExpandableSegment/malloc/free/OOM/gc
  c10/cuda/CUDACachingAllocator.h → BlockPool/PrivatePool/CUDAAllocator interface
  c10/cuda/CUDAStream.cpp → Stream pool (3 pool per device, 32 streams per pool, round-robin)
  c10/cuda/CUDAStream.h → Stream pool design note + StreamId encoding
  c10/cuda/CUDAAllocatorConfig.cpp → expandable_segments/garbage_collection/max_split_size config
  torch/cuda/streams.py → Stream/Event/ExternalStream Python wrapper (271行)
  torch/cuda/memory.py → MemPool/caching_allocator/memory_stats (1500行)
  torch/cuda/graphs.py → CUDAGraph/graph_pool_handle/capture_begin/end

DeepSpeed:
  deepspeed/runtime/zero/stage_1_and_2.py → reduction_stream + IPG双buffer
  deepspeed/runtime/zero/stage3.py → allgather_stream + reduce_partition_stream + backpressure
  deepspeed/runtime/zero/partitioned_param_coordinator.py → trace-based prefetch + 2-event backpressure
  deepspeed/runtime/zero/utils.py → defragment() (33行!)
  deepspeed/runtime/zenflow/ → ZenFlow CPU optimizer process + shared memory

Megatron:
  megatron/core/distributed/param_and_grad_buffer.py → communication_stream + overlap_grad_reduce
  megatron/core/full_cuda_graph.py → shared_capture_stream + shared graph pool

vLLM:
  vllm/v1/worker/gpu_model_runner.py → async_output_copy_stream + ngram_copy_stream + prepare_inputs_event
  vllm/v1/core/block_pool.py → BlockPool + FreeKVCacheBlockQueue + prefix caching
  vllm/v1/core/kv_cache_utils.py → KVCacheBlock + BlockHash + hash functions

verl:
  verl/workers/rollout/ → backend-specific → vLLM/Megatron/FSDP stream usage
```

---

## 8. 与其他阅读的连接

```
本note填补的gap → 之前7框架研究中的stream/memory盲区:

→ DeepSpeed ZeRO overlap (已有: deepspeed-comm-overlap-reading.md) → ★ 补充了stream-memory交互细节
→ vLLM CUDA Graph (已有: vllm-cuda-graph-reading.md) → ★ 补充了private pool + recordStream机制
→ Megatron PP (已有: megatron-pp-reading.md) → ★ 补充了communication_stream + async_op
→ vLLM KV Cache (已有: vllm-v1-kv-cache-management-reading.md) → ★ 补充了BlockPool vs Caching Allocator对比
→ PyTorch compile (已有: pytorch-compile-stack-knowledge-synthesis.md) → ★ 补充了torch.Stream + CUDA graph stream
→ NCCL (缺失: memory文件已不在) → ★ 补充了NCCL stream与PyTorch allocator交互

★ ★ ★ 新洞察:
  1. PyTorch Caching Allocator的recordStream → 跨stream内存reuse的底层保障 → 所有框架依赖!
  2. Expandable Segments → inference碎片化的终极解决 → RTX 4090必须开启
  3. vLLM的stream不是计算-通信overlap → 而是GPU→CPU异步copy → ★ inference特有pattern!
  4. DeepSpeed defragment → 33行 → GPU→CPU→empty_cache→CPU→GPU → 完全消除碎片 → 简洁!
  5. RTX 4090 → stream优化几乎无用 → 内存管理是关键 → ★ expandable_segments > 多stream!
  6. ★ ★ ★ rLLM Tinker → 1 stream + CPU asyncio → RTX 4090最优 → 简单就是美!
```
