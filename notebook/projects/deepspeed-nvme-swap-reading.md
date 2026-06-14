# DeepSpeed ZeRO-Infinity NVMe Swap 数据流 (源码级)

> 2026-06-15 | 源码: deepspeed/runtime/swap_tensor/partitioned_param_swapper.py (399行)
> 核心: 异步I/O(aio_handle/GDS) + buffer pool + 3级pipeline(GPU→CPU→NVMe) + prefetch协调
> 这是DeepSpeed独有功能 — 无其他框架实现NVMe offload

## 1. AsyncPartitionedParameterSwapper 概述

```python
class AsyncPartitionedParameterSwapper:
    # 核心职责: 参数分片(ds_tensor)的NVMe存储和恢复
    # 3个状态: AVAILABLE(CPU/GPU内存中) / NOT_AVAILABLE(在NVMe) / INFLIGHT(正在从NVMe读取)

    def __init__(self, ds_config, model_dtype):
        self._configure_aio(ds_config)  # 配置异步I/O引擎
        self.id_to_path = {}            # param.ds_id → NVMe文件路径
        self.param_id_to_buffer_id = {}  # param → swap buffer ID
        self.available_buffer_ids = []   # 可用buffer池
        self.inflight_params = []        # 正在swap-in的参数
        self.pending_writes = 0          # 待完成的NVMe写入数
        self.pending_reads = 0           # 待完成的NVMe读取数
```

## 2. 异步I/O引擎配置 (_configure_aio)

```python
def _configure_aio(self, ds_config):
    # NVMe存储路径: /nvme_path/zero_stage_3/bfloat16params/rank{rank}/
    self.swap_folder = os.path.join(
        self.swap_config.nvme_path, 'zero_stage_3',
        f'{torch_dtype_string}params', f'rank{dist.get_rank()}'
    )

    # GDS (GPUDirect Storage) 或 AIO (Async I/O)
    self.use_gds = self.aio_config[AIO_USE_GDS]
    self.aio_handle = (
        GDSBuilder().load().gds_handle     # GPUDirect: NVMe→GPU 直读!
        if self.use_gds else
        AsyncIOBuilder().load().aio_handle  # 标准AIO: NVMe→CPU→GPU
    )

    # I/O对齐: block_size + intra_op_parallelism
    self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE])
    self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_INTRA_OP_PARALLELISM]
    self.numel_alignment = self.aligned_bytes // self.swap_element_size

    # Buffer池: buffer_count个buffer, 每个buffer_size元素
    self.elements_per_buffer = self.swap_config.buffer_size
    self.param_buffer_count = self.swap_config.buffer_count

    # 两个独立的AIO handle: 读写可并行!
    self.aio_read_handle = self.aio_handle(...)
    self.aio_write_handle = self.aio_handle(...)

    # Buffer分配: CPU pinned memory(非GDS) 或 GPU内存(GDS)
    buffer_device = get_accelerator().device_name() if self.use_gds else "cpu"
    self.buffers = torch.empty(
        self.aligned_elements_per_buffer * self.param_buffer_count,
        dtype=self.dtype, device=buffer_device
    )
    if not self.use_gds:
        self.buffers = get_accelerator().pin_memory(self.buffers)  # CPU pinned!
```

**关键参数**:

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| nvme_path | NVMe SSD路径 | /local_nvme/ |
| buffer_size | 每个swap buffer的参数元素数 | 1e5-1e6 |
| buffer_count | swap buffer数量 | 5-10 |
| AIO_BLOCK_SIZE | 每次I/O的块大小 | 1MB (1048576) |
| AIO_QUEUE_DEPTH | I/O队列深度 | 8-32 |
| AIO_INTRA_OP_PARALLELISM | 单次请求的线程并行数 | 1-4 |
| AIO_OVERLAP_EVENTS | 是否overlap读写 | True |
| AIO_USE_GDS | 是否用GPUDirect Storage | False(大多数硬件不支持) |

## 3. Swap-Out: GPU/CPU → NVMe (参数分片写入磁盘)

```python
def swap_out_and_release(self, params, async_op=False, force_buffer_release=False):
    self._swap_out(params, async_op=async_op)

def _swap_out(self, params, async_op=True):
    swap_out_paths = self._get_swap_paths(params)   # 获取NVMe文件路径
    swap_out_params = self._get_swap_buffers(params) # 获取swap buffer

    swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)
    # → aio_write_handle: 异步写入 → 不阻塞训练!

    self.pending_writes += len(swap_out_params)
    self.swap_out_params += params

    if not async_op:
        self.synchronize_writes()  # 阻塞等待写入完成
```

**Swap-Out 数据流**:

```
参数分片(param.ds_tensor) → swap buffer(CPU pinned) → NVMe SSD文件

详细:
  1. 参数partition后: param.ds_tensor.data = 1/N分片数据 (在CPU/GPU内存)
  2. _swap_out: 将分片数据写入NVMe文件
     → 文件路径: {nvme_path}/zero_stage_3/{dtype}params/rank{rank}/{param_id}_param.tensor.swp
  3. 写入完成后(异步): param.ds_tensor.data = invalid_buffer → 释放内存!
  4. param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE → 标记在NVMe
```

**I/O对齐**: NVMe SSD要求对齐读写 → `numel_alignment` → padding确保块大小对齐

## 4. Swap-In: NVMe → CPU/GPU (参数分片从磁盘恢复)

```python
def swap_in(self, params, async_op=True, swap_in_buffers=None):
    assert all([param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE for param in params])

    swap_in_paths = self._get_swap_paths(params)

    if swap_in_buffers is None:
        # 从buffer池分配
        assert len(swap_in_paths) <= len(self.available_buffer_ids)  # buffer够吗?
        compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)

    swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
    # → aio_read_handle: 异步读取 → 不阻塞训练!

    self._update_inflight_swap_in(params, swap_in_buffers, inflight_numel)
    # → inflight_params记录 → param.ds_tensor.status = INFLIGHT

    if not async_op:
        self.synchronize_reads()  # 阻塞等待读取完成
```

**Swap-In 数据流**:

```
NVMe SSD文件 → swap buffer(CPU pinned/GPU) → param.ds_tensor.data恢复

详细:
  1. 检查param状态: 必须是NOT_AVAILABLE(在NVMe)
  2. 分配swap buffer: 从available_buffer_ids池中获取
     → buffer_id = self.available_buffer_ids.pop()
     → swap_buffer = self.buffers.narrow(buffer_id * aligned_elements, aligned_numel)
  3. 异步读取: aio_read_handle → NVMe→buffer → 不阻塞!
  4. 读取完成: synchronize_reads() → param.ds_tensor.data = compute_buffer.data
  5. param.ds_tensor.status = PartitionedParamStatus.AVAILABLE → 可用于AllGather
```

**Buffer池管理**:

```
buffer_count个固定大小buffer:
  available_buffer_ids: [0, 1, 2, 3, 4, ...]  ← 可用
  → swap_in时pop → 用完append回available
  → buffer_count限制 → 同时swap-in的参数数量受限!

不足时: "Not enough swap in buffers" → 必须等待前一批swap-in完成
→ 这是为什么prefetch需要协调buffer使用!
```

## 5. 3级Pipeline: GPU ↔ CPU ↔ NVMe

```
完整数据流(一个参数的生命周期):

  Init: 完整参数 → broadcast → partition → ds_tensor(1/N分片) → 可在CPU/GPU内存
  ↓
  Forward需要: AllGather ds_tensor → 恢复完整参数 → 计算
  ↓
  Forward完成: partition → 释放完整参数 → ds_tensor回到1/N分片
  ↓
  不立即需要: swap_out → ds_tensor写入NVMe → 释放CPU/GPU内存!
  ↓                                        → 只剩invalid_buffer(1个half元素!)
  Prefetch协调: 判断即将需要 → swap_in → 从NVMe恢复ds_tensor到CPU pinned buffer
  ↓
  Forward需要: AllGather ds_tensor(已从NVMe恢复) → 恢复完整参数 → 计算
  ↓
  Backward同理...
  ↓
  Optimizer: FP32 optimizer states → 也可swap到NVMe(另一个swapper!)

  → → → GPU内存只持有: 当前需要的完整参数 + 1/N分片(prefetched) + optimizer 1/N
  → → → → NVMe持有: 所有不需要的1/N分片 + FP32 optimizer states
  → → → → → → GPU内存 ≈ 0 (只持当前层!) → 支撑任意大模型!
```

## 6. GPUDirect Storage (GDS) — NVMe→GPU直读

```python
if self.use_gds:
    # GDS: NVMe SSD → GPU内存 → 绕过CPU!
    buffer_device = get_accelerator().device_name()  # GPU内存
    self.aio_read_handle.pin_device_tensor(self.buffers)  # Pin GPU buffer
else:
    # 标准AIO: NVMe → CPU pinned → GPU (需要额外拷贝)
    buffer_device = "cpu"
    self.buffers = get_accelerator().pin_memory(self.buffers)
```

**GDS vs 标准AIO对比**:

| 方面 | 标准AIO | GPUDirect Storage |
|------|---------|-------------------|
| 读取路径 | NVMe→CPU pinned→GPU | NVMe→GPU(直读!) |
| 拷贝次数 | 2次 | 1次 |
| 延迟 | 较高 | 较低 |
| 硬件要求 | 任何NVMe | NVIDIA GPU+GDS-capable NVMe |
| Buffer位置 | CPU pinned内存 | GPU内存 |
| 页面缓存 | Linux page cache | 无(bypass) |
| 适用性 | ✅ 广泛 | ⚠️ 特定硬件 |

## 7. 与PrefetchCoordinator的协作

```
3级pipeline时间线:

时间:  0ms     5ms     10ms    15ms    20ms    25ms
       │       │       │       │       │       │
  GPU: [Layer1 fwd] [Layer2 fwd] [Layer3 fwd] ...
       │       │       │       │
  CPU: [AllGather L1] [AllGather L2] [AllGather L3] ...
       │       │       │       │       │
  NVMe: [swap_in L2] [swap_in L3] [swap_in L4] [swap_in L5] ...
       ↑       ↑       ↑       ↑
       PrefetchCoordinator触发:
       L1计算时 → prefetch L2(AllGather) + swap_in L3(NVMe→CPU)
       L2计算时 → prefetch L3(AllGather) + swap_in L4(NVMe→CPU)
       → → → 3级overlap → GPU几乎0空闲!
```

**PrefetchCoordinator的NVMe prefetch**:

```python
# partitioned_param_coordinator.py: __prefetch_nvme_param_partitions()
def __prefetch_nvme_param_partitions(self):
    numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)
    swap_in_params = []
    for param_in_trace in self.__param_queue:
        param = param_in_trace.param
        # 预取距离: 2×当前inflight → 平衡提前量和buffer占用
        if numel_considered > 2 * numel_in_flight:
            break
        # buffer够吗?
        if len(swap_in_params) >= param.nvme_swapper.available_swap_in_buffers():
            break  # buffer不够 → 停止预取
        if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
            swap_in_params.append(param)

    if swap_in_params:
        swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)
        # → 异步! → 不阻塞当前计算!
```

## 8. NVMe Swap性能分析

```
NVMe SSD性能基准:
  - 读取带宽: 3-7 GB/s (PCIe Gen4 NVMe)
  - 写入带宽: 2-5 GB/s
  - 随机I/O延迟: 50-100μs

7B模型参数1/N分片(N=8):
  - 1/N分片大小: 14GB/8 = 1.75GB
  - Swap-out时间: 1.75GB / 5GB/s = 350ms → 异步, 不阻塞!
  - Swap-in时间: 1.75GB / 7GB/s = 250ms → 异步, 但prefetch需提前250ms开始

100B模型(N=64):
  - 1/N分片大小: 200GB/64 = 3.125GB
  - Swap-out时间: 625ms → 可接受(异步)
  - Swap-in时间: 450ms → prefetch需提前450ms

→ → → NVMe带宽足够 → 只要prefetch提前量够 → 不影响训练吞吐!
→ → → 关键瓶颈: buffer_count限制 → 同时swap-in的参数数量 → 需精心配置
```

## 9. ZeRO-Infinity完整内存公式

```
ZeRO-Infinity per-rank GPU内存:

  当前计算层的完整参数: Ψ_layer * 2 bytes (BF16)
  1/N分片(prefetched, 在GPU): Ψ/N * 2 bytes
  当前层的激活值: activation_layer
  1/N梯度: Ψ/N * 2 bytes
  1/N optimizer states: Ψ/N * 12 bytes (在NVMe或CPU)

  → GPU内存 ≈ 2Ψ_layer + 2Ψ/N + activations + gradient_shard + opt_shard
  → → → Ψ_layer << Ψ → GPU内存 ≈ activations + small fixed overhead!
  → → → → 支撑任意大模型! 只需NVMe容量足够!

vs ZeRO-3(无NVMe):
  → Peak = 16Ψ/N + 4Ψ → 受限于GPU内存
  → → → 7B N=8: 84GB → 24GB RTX 4090不行!

vs ZeRO-Infinity(有NVMe):
  → Peak ≈ 2Ψ_layer + activations → 只需当前层!
  → → → 7B单层 ≈ 200MB → 24GB RTX 4090完全可以!
  → → → → 但: NVMe offload速度慢 → wall time增加 → 不适合实时训练

RTX 4090实际:
  → ZeRO-Infinity理论上可行 → 但NVMe SSD未必有 → 消费级PC通常无
  → verl GRPO + LoRA + CPU Adam = 更实用方案 (20.04GB fits 24GB)
```

## 10. 与其他框架对比

| 方面 | DeepSpeed ZeRO-Infinity | FSDP2 | Megatron-LM |
|------|-------------------------|-------|-------------|
| NVMe offload | ✅ 独有 | ❌ | ❌ |
| CPU offload | ✅ optimizer+param | 🔄(开发中) | ✅(DeepSpeed集成) |
| GDS支持 | ✅ NVMe→GPU直读 | ❌ | ❌ |
| Async I/O | ✅ aio_handle(GDS/标准) | ❌ | ❌ |
| Buffer池管理 | ✅ 固定buffer_count | ❌ | ❌ |
| 3级pipeline | ✅ GPU↔CPU↔NVMe | ❌(2级GPU↔CPU) | ❌ |
| 超大模型支持 | ✅(200B+) | ❌(GPU+CPU限制) | ✅(多GPU+TP+PP) |

**关键**: ZeRO-Infinity是唯一能在单GPU上训练任意大模型的技术 → 但wall time增加

## 11. 下一步

- [ ] 研究 AllGatherCoalescedHandle内部实现(coalesced buffer合并策略)
- [ ] 研究 ZeRO-Infinity optimizer swapper(FP32 optimizer states的NVMe offload)
- [ ] 在GPU+NVMe可用时实测ZeRO-Infinity性能 vs ZeRO-Offload vs verl GRPO
- [ ] 研究 DeepSpeed async_swapper.py(更通用的swap框架)
- [ ] 对比 ZeRO-Infinity vs Gemstone(Google) vs Gradient-Checkpointing vs PagedAttention的内存策略
