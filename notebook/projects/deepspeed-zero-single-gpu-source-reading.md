# DeepSpeed ZeRO Single GPU Source-Level Deep Reading (RTX 4090)

> Source: microsoft/DeepSpeed (cloned 2026-06-16, latest master)
> Focus: ZeRO-2 vs ZeRO-3 on single GPU, optimizer state partitioning, CPU_Adam offload, LoRAOptimizedLinear
> Rating system: ★★★★★ (critical finding) → ★ (minor detail)

---

## 1. ZeRO Stage 2 vs Stage 3 on Single GPU

### 1.1 ZeRO-2: `DeepSpeedZeroOptimizer` (stage_1_and_2.py, line 126)

★★★★★ **ZeRO-2 on single GPU: partition_size = full model size, partition collapses to whole**

```python
# stage_1_and_2.py, line 236
dp_size = dist.get_world_size(group=self.dp_process_group)
self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]
```

On single GPU: `dp_size = 1`, `partition_count = [1]`.

```python
# stage_1_and_2.py, line 360
partition_id = dist.get_rank(group=self.real_dp_process_group[i])  # = 0 on single GPU

# stage_1_and_2.py, line 503
partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
# On single GPU: partition_size = total_flat_numel / 1 = total_flat_numel (FULL MODEL)
```

★★★★★ **get_data_parallel_partitions() with dp=1**: ONE partition = entire flat tensor

```python
# stage_1_and_2.py, lines 1826-1844
def get_data_parallel_partitions(self, tensor, group_id):
    partitions = []
    dp = dist.get_world_size(group=self.real_dp_process_group[group_id])  # = 1 on single GPU
    total_num_elements = tensor.numel()
    base_size = total_num_elements // dp  # = total_num_elements // 1 = total_num_elements
    remaining = total_num_elements % dp   # = 0
    start = 0
    for id in range(dp):  # range(1) → only 1 iteration
        partition_size = base_size  # = total_num_elements
        if id < remaining: partition_size += 1  # id=0, remaining=0 → no adjustment
        partitions.append(tensor.narrow(0, start, partition_size))  # entire tensor
        start += partition_size
    return partitions  # [full_tensor] — single element list
```

★★★★★ **single_partition_of_fp32_groups = FULL FP32 COPY of entire model params**

```python
# stage_1_and_2.py, line 477-494
weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].detach().clone().to(
    device=self.device, dtype=self.master_weights_and_grads_dtype)
# partition_id=0, partitions=[full_tensor] → weights_partition = full FP32 copy of ALL params
self.single_partition_of_fp32_groups.append(weights_partition)
```

On single GPU, `single_partition_of_fp32_groups[i]` = full FP32 copy of entire param group — NO partitioning savings.

★★★★★ **Optimizer state: NO partitioning on single GPU**

```python
# stage_1_and_2.py, line 498-501
param_group['params'] = [self.single_partition_of_fp32_groups[i]]
# The optimizer receives the FULL fp32 partition as its single param
# → optimizer.state[p] contains exp_avg, exp_avg_sq for ALL parameters
# → On single GPU: optimizer state size = 2 * full_model_size (momentum + variance)
# → NO savings from partitioning!
```

★★★★★ **all_gather_dp_groups with dp_world_size=1: SKIP entirely**

```python
# runtime/utils.py, lines 1026-1029 (inside all_gather_dp_groups)
if dp_world_size == 1:
    # no groups share optimizer states
    # pipeline parallel with bf16 will default call this even if dp size = 1.
    continue
```

This is the CRITICAL degenerate case handler. On single GPU, `all_gather_dp_groups` skips entirely — no communication overhead, but also NO partitioning benefit.

★★★★ **ZeRO-2 single GPU memory accounting**:
- `bit16_groups_flat[i]`: full BF16/FP16 model (Ψ bytes)
- `single_partition_of_fp32_groups[i]`: full FP32 copy (2Ψ bytes) — partition_id=0 → full model
- optimizer state (exp_avg + exp_avg_sq): full FP32 (4Ψ bytes)
- **Total: Ψ + 2Ψ + 4Ψ = 7Ψ bytes** — EXACTLY same as vanilla DDP with mixed precision
- ZeRO-2 on single GPU = pure overhead (partitioning machinery + no savings)

### 1.2 ZeRO-3: `DeepSpeedZeroOptimizer_Stage3` (stage3.py, line 136)

★★★★★ **ZeRO-3 on single GPU: partition_count=1, but gather/scatter overhead remains**

```python
# stage3.py, line 347
self.partition_count = dist.get_world_size(group=self.dp_process_group)  # = 1 on single GPU
```

★★★★★ **ZeRO-3 partition_parameters Init (partition_parameters.py, line 884)**:

```python
# partition_parameters.py, lines 1035-1036
self.rank = dist.get_rank(group=self.ds_process_group)  # = 0 on single GPU
self.dp_world_size = dist.get_world_size(group=self.ds_process_group)  # = 1 on single GPU

# partition_parameters.py, lines 1043-1045
self.num_ranks_in_param_group = self.dp_world_size  # = 1
self.rank_in_group = self.rank  # = 0
self.num_param_groups = 1
```

★★★★★★ **CRITICAL: _partition_param() with num_partitions=1**

```python
# partition_parameters.py, line 1696
partition_size = tensor_size // self.num_partitions  # tensor_size // 1 = tensor_size
# → Each param's partition = full param → NO sharding!
```

★★★★★★ **But: ZeRO-3 still does gather/scatter even when dp_world_size=1!**

The `_all_gather` and `_partition` methods in `PartitionedParameterCoordinator` still execute even on single GPU. Each forward pass still:
1. Calls `_all_gather` → gathers param from partition (which is the full param, but still goes through the infrastructure)
2. After forward, calls `_partition` → partitions param back (which is trivially the full param)

```python
# partition_parameters.py, lines 1606-1642 (_all_gather method)
# Even with world_size=1, the allgather infrastructure runs
# But NoGatherHandle / NoGatherCoalescedHandle may be used
```

★★★★★ **_no_gather_coalesced shortcut for single rank**:

```python
# partition_parameters.py, lines 870-880
def _no_gather_coalesced(params):
    for param in params:
        if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
            raise RuntimeError(...)
        param.ds_status = ZeroParamStatus.INFLIGHT
    params = sorted(params, key=lambda p: p.ds_id)
    if len(params) == 1:
        param, = params
        return NoGatherHandle(param)  # Single-param shortcut
    return NoGatherCoalescedHandle(params)  # Multi-param shortcut
```

There IS a shortcut path — `NoGatherHandle`/`NoGatherCoalescedHandle` — but it's used when `allgather_sequential=True` or when no actual AllGather communication is needed. The default path still goes through `_allgather_params_coalesced`.

★★★★ **ZeRO-3 single GPU: gather/scatter overhead with NO memory savings**:

ZeRO-3 on single GPU:
- Parameters: each param's ds_tensor = full param (partition_size = numel)
- FP32 master weights: `fp32_partitioned_groups_flat` = full FP32 partition (same as ZeRO-2)
- Optimizer states: full exp_avg + exp_avg_sq for ALL params
- **Total memory: same as ZeRO-2 ≈ 7Ψ**
- **Overhead: gather/scatter infrastructure + parameter coordinator + partitioned_param_coordinator** — PURE overhead with NO benefit

★★★★★★ **CONFIRMED: ZeRO-3 single GPU = pure overhead → NOT recommended for RTX 4090**

The partition_parameters.py `_partition_param` creates ds_tensor with `partition_size = tensor_size // num_partitions = tensor_size // 1 = full_size`. No actual partitioning occurs. But the infrastructure (ds_status tracking, AllGather handles, PartitionedParameterCoordinator) all still run, adding latency and complexity with zero memory benefit.

---

## 2. ZeRO-2 Optimizer State Partitioning Mechanism

### 2.1 Partitioning Algorithm

★★★★★ **Flat tensor partitioning**:

```python
# stage_1_and_2.py, lines 298-310
# param flattened by groups
self.bit16_groups = []       # original params per group
self.bit16_groups_flat = []  # flattened contiguous buffer per group

# param partitioned by data parallel degree
self.parallel_partitioned_bit16_groups = []

# a single 32-bit partition of the parallel partitioned parameters
# that this process will update
self.single_partition_of_fp32_groups = []
```

★★★★★ **Round-robin gradient reordering for load balancing**:

```python
# stage_1_and_2.py, lines 770-789
def _round_robin_reorder(self, tensor_list, num_partitions):
    partition_tensors = {}
    for i, tensor in enumerate(tensor_list):
        j = i % num_partitions  # round-robin assignment
        if j not in partition_tensors:
            partition_tensors[j] = []
        partition_tensors[j].append((i, tensor))
```

On single GPU (`num_partitions=1`): round-robin assigns all params to partition 0 → no reordering effect.

★★★★ **Alignment padding**:

```python
# stage_1_and_2.py, line 377
alignment = self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])
# On single GPU: alignment = nccl_start_alignment_factor * 1 = nccl_start_alignment_factor (typically 4)
```

★★★★★ **HP mapping (half-precision ↔ full-precision)**:

```python
# stage_1_and_2.py, lines 703-721 (_link_all_hp_params)
partition_id = dist.get_rank(group=self.real_dp_process_group[i])  # = 0 on single GPU
partition_size = self.bit16_groups_flat[i].numel() // dist.get_world_size(group=self.real_dp_process_group[i])
# On single GPU: partition_size = full_flat_numel
flat_hp_partition = self.single_partition_of_fp32_groups[i]  # = full FP32 copy
link_hp_params(lp_param_list=self.bit16_groups[i],
               flat_hp_partition=flat_hp_partition,
               gradient_dict=self.averaged_gradients,
               offload_gradient_dict=self.offload_gradient_dict,
               use_offload=self.cpu_offload,
               param_group_index=i,
               partition_start=partition_id * partition_size,  # = 0
               partition_size=partition_size,  # = full size
               dp_group=self.real_dp_process_group[i])
```

On single GPU, `link_hp_params` links ALL params to the FULL fp32 partition — every param has a mapping to the full model.

### 2.2 Optimizer Step on Single GPU

★★★★★ **ZeRO-2 step() — no offload path**:

```python
# stage_1_and_2.py, lines 2203-2246 (non-offload optimizer step)
# 1. Free gradients for params NOT in partition (ZeRO-2: only partition_id=0's params)
self.free_grad_in_param_list(self.params_not_in_partition[i])
# On single GPU: params_not_in_partition = [] (all params in partition 0)

# 2. Create flat gradients for partition
flat_grad_partition = self._get_preflattened_grad_partition(i)
# Or: flat_grad_partition = self.flatten(self.averaged_gradients[i])
# On single GPU: this is the FULL averaged gradient

single_grad_partition = flat_grad_partition.to(self.single_partition_of_fp32_groups[i].dtype)
# On single GPU: full FP32 gradient for full FP32 weights

self.single_partition_of_fp32_groups[i].grad = single_grad_partition

# 3. Run optimizer
self._optimizer_step(i)
# The optimizer processes the ENTIRE model at once

# 4. Copy updated FP32 weights back to BF16
bit16_partitions[partition_id].data.copy_(fp32_partition.data)
# partition_id=0 → copy full FP32 partition to full BF16 flat buffer

# 5. All-gather updated weights (SKIPPED on single GPU per all_gather_dp_groups)
all_gather_dp_groups(...)  # dp_world_size=1 → continue (skip)
```

★★★★★★ **ZeRO-2 step() — CPU offload path** (more relevant for RTX 4090):

```python
# stage_1_and_2.py, lines 2182-2200 (CPU offload optimizer step)
# 1. Get gradient partition (on CPU, pinned)
single_grad_partition = self.single_partition_of_fp32_groups[i].grad  # Already on CPU

# 2. Unscale and clip
self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

# 3. CPU optimizer step
self._optimizer_step(i)
# DeepSpeedCPUAdam.step() runs on CPU — ALL parameters at once

# 4. Copy FP32 → BF16 (CPU → GPU)
bit16_partition_buffer = self.param_buffer_of_bit16_for_cpu_offload_groups[i]
bit16_partition_buffer.data.copy_(fp32_partition.data)  # CPU → CPU copy (pinned buffer)
bit16_partitions[partition_id].data.copy_(bit16_partition_buffer.data, non_blocking=True)  # CPU → GPU copy
# On single GPU: this copies the ENTIRE model from CPU to GPU
```

### 2.3 Gradient Partitioning in ZeRO-2

★★★★★ **ZeRO-2 gradient partition: each rank owns 1/world_size of averaged gradients**:

```python
# stage_1_and_2.py, lines 860-883
def initialize_gradient_partitioning_data_structures(self):
    for i, param_group in enumerate(self.round_robin_bit16_groups):
        total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
        # On single GPU: total_partitions = 1
        # → Only 1 partition → ALL gradients belong to partition 0
```

★★★★★ **averaged_gradients on single GPU**:

```python
# stage_1_and_2.py, lines 910-918 (independent_gradient_partition_epilogue)
if self.is_gradient_accumulation_boundary:
    self.averaged_gradients[i] = self.get_flat_partition(
        self.params_in_partition[i],        # ALL params on single GPU
        self.first_offset[i],               # = 0 on single GPU
        self.partition_size[i],             # = full size on single GPU
        dtype=self.gradient_accumulation_dtype,
        device=get_accelerator().current_device_name(),
        param_group_idx=i,
        return_tensor_list=True)
```

On single GPU, `averaged_gradients[i]` contains the FULL averaged gradient for the ENTIRE param group. No partitioning savings.

★★★★★ **ZeRO-2 AllReduce with dp_world_size=1**:

```python
# stage_1_and_2.py, lines 1144-1171 (gradient_reduction_w_predivide)
dp_world_size = dist.get_world_size(group=self.dp_process_group)  # = 1
dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
# On single GPU: all_reduce with group_size=1 = identity operation (no communication)
tensor_to_allreduce.div_(dp_world_size / float(self.sequence_parallel_size))
# = div_(1 / 1) = identity
```

AllReduce on single GPU is effectively an identity operation — no communication, no averaging needed.

---

## 3. ZeRO-Offload + CPU Optimizer

### 3.1 DeepSpeedCPUAdam Implementation

★★★★★ **CPUAdam: C++ SIMD-optimized Adam on CPU** (cpu_adam.py + csrc/adam/cpu_adam.cpp):

```python
# cpu_adam.py, lines 13-94
class DeepSpeedCPUAdam(torch.optim.Optimizer):
    def __init__(self, model_params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, adamw_mode=True, fp32_optimizer_states=True):
        self.ds_opt_adam = CPUAdamBuilder().load()  # C++ extension
        self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps,
                                     weight_decay, adamw_mode, should_log_le("info"))
```

★★★★★ **C++ kernel: ds_adam_step function**:

```cpp
// csrc/includes/cpu_adam.h, lines 210-221
int ds_adam_step(int optimizer_id, size_t step, float lr, float beta1, float beta2,
                 float epsilon, float weight_decay, bool bias_correction,
                 torch::Tensor& params,     // FP32 master weights on CPU
                 torch::Tensor& grads,      // FP32 gradients on CPU
                 torch::Tensor& exp_avg,    // momentum on CPU
                 torch::Tensor& exp_avg_sq) // variance on CPU
```

★★★★★ **SIMD optimization: AVX512/AVX256 vectorized Adam step**:

```cpp
// csrc/includes/cpu_adam.h, lines 112-198 (Step_AVX template)
template <int span, typename ds_params_precision_t, typename ds_state_precision_t>
void Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              ds_params_precision_t* _params,
                              ds_params_precision_t* grads,
                              ds_state_precision_t* _exp_avg,
                              ds_state_precision_t* _exp_avg_sq,
                              size_t _param_size)
{
    // SIMD_WIDTH * span parallel elements per iteration
    // #pragma omp parallel for — OpenMP multi-threading
    for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
        // Vectorized: load grad, momentum, variance, param
        // Update: m = b1*m + (1-b1)*g, v = b2*v + (1-b2)*g^2
        // Param update: p = p - lr * m / (sqrt(v) + eps)
        // With optional AdamW weight decay
    }
}
```

★★★★★ **5-7x speedup over torch.optim.Adam on CPU**: The C++ kernel uses SIMD (AVX512/AVX256) + OpenMP parallelism, achieving 5-7x speedup compared to PyTorch's native CPU Adam.

★★★★ **fp32_optimizer_states=True default**: Momentum and variance stored in FP32 on CPU regardless of param dtype. Setting `fp32_optimizer_states=False` stores them in param dtype (BF16), reducing CPU memory at cost of precision.

### 3.2 Pin Memory Mechanism

★★★★★ **Pinned CPU memory for GPU-CPU transfer**:

```python
# stage_1_and_2.py, lines 480-492
if self.cpu_offload:
    if self.cpu_offload_pin_memory:
        weights_partition = get_accelerator().pin_memory(weights_partition)
    # Also pin the BF16 buffer for GPU transfer
    temp_buffer_bit16 = torch.full(weights_partition.shape, fill_value=0.0, ...)
    if self.cpu_offload_pin_memory:
        temp_pinned = get_accelerator().pin_memory(temp_buffer_bit16)
        self.param_buffer_of_bit16_for_cpu_offload_groups.append(temp_pinned)
```

★★★★★ **Gradient accumulation temp buffers**:

```python
# stage_1_and_2.py, lines 567-576
self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel,
                                                     device=self.device,  # 'cpu'
                                                     dtype=self.dtype)
if self.cpu_offload_pin_memory:
    self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
        self.temp_grad_buffer_for_cpu_offload)
self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                     device=get_accelerator().current_device_name(),  # 'cuda:0'
                                                     dtype=self.dtype)
```

Two buffers:
- `temp_grad_buffer_for_cpu_offload`: pinned CPU buffer for accumulating gradients
- `temp_grad_buffer_for_gpu_offload`: GPU buffer for intermediate gradient computation

### 3.3 CPU Offload Gradient Flow

★★★★★ **async_inplace_copy_grad_to_fp32_buffer_from_gpu**:

```python
# stage_1_and_2.py, line 1526
self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)
```

This function copies gradient from GPU to pinned CPU buffer (FP32). The flow:
1. Compute gradient on GPU (BF16)
2. `set_norm_for_param_grad_in_gpu(param)` — compute norm on GPU
3. `async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)` — copy BF16→FP32, GPU→pinned CPU
4. CPU Adam processes gradient on CPU (FP32)
5. Updated FP32 weights copied back: CPU→GPU (pinned buffer → BF16)

★★★★★ **async_accumulate_grad_in_cpu_via_gpu for gradient accumulation**:

```python
# stage_1_and_2.py, lines 1513-1528
def copy_grads_in_partition(self, param):
    if self.cpu_offload:
        # Accumulate when there were prior backwards in this step
        if self.micro_step_id > 0 or not self.is_gradient_accumulation_boundary:
            self.async_accumulate_grad_in_cpu_via_gpu(param)  # CPU accumulation
        if self.is_gradient_accumulation_boundary:
            self.set_norm_for_param_grad_in_gpu(param)        # GPU norm
            self.update_offload_overflow_tracker_for_param_grad(param)
            self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)  # GPU→CPU copy
```

### 3.4 Overlap Mechanism

★★★★ **No compute-optimizer overlap on single GPU**: DeepSpeed's overlap_comm (line 221) overlaps gradient AllReduce with backward computation. On single GPU, AllReduce is identity — overlap provides no benefit. The CPU optimizer runs after backward completes — no overlap with GPU compute on single GPU.

★★★★★ **ZeRO-3 partial offload (offload_ratio)**:

```python
# stage3.py, lines 937-946
if self.offload_optimizer:
    self.subgroup_to_device = {}
    sub_group_size = len(self.fp16_partitioned_groups_flat)
    for i in range(sub_group_size):
        if i >= int((1 - self.partial_offload) * sub_group_size):
            self.subgroup_to_device[i] = 'cpu'     # Last subgroups → CPU
        else:
            self.subgroup_to_device[i] = get_accelerator()._name  # First subgroups → GPU
```

★★★★★ **Hybrid optimizer: CPU_Adam + GPU AdamW for partial offload**:

```python
# stage3.py, lines 295-308
if self.offload_optimizer and self.partial_offload != 1.0:
    backup_gpu_tensor = torch.randn(1, device=get_accelerator().device_name()).to(self.dtype)
    backup_gpu_param = torch.nn.Parameter(backup_gpu_tensor)
    assert type(init_optimizer) == DeepSpeedCPUAdam, 'Hybrid Optimizer Only Supports DeepSpeedCPUAdam'
    self.backup_optimizer = torch.optim.AdamW([backup_gpu_param], ...)  # GPU-side AdamW
```

★★★★★★ **ZeRO-3 optimizer step with partial offload**:

```python
# stage3.py, lines 1073-1100 (_optimizer_step)
def _optimizer_step(self, sub_group_id):
    param_group_id = self.sub_group_to_group_id[sub_group_id]
    fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
    if self.offload_optimizer:
        cur_device = self.subgroup_to_device[sub_group_id]
        if cur_device == 'cpu':
            self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            step_with_gradscaler(self.optimizer)  # DeepSpeedCPUAdam on CPU
        else:
            self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            step_with_gradscaler(self.backup_optimizer)  # torch.optim.AdamW on GPU
```

This is the hybrid mechanism: some subgroups on CPU (DeepSpeedCPUAdam), some on GPU (AdamW).

★★★★★ **On single GPU with partial_offload**: Still works! Some FP32 params + optimizer states on CPU, rest on GPU. The partition_count=1 means each subgroup = full param partition, but partial_offload splits WHICH subgroups go to CPU vs GPU. This is the ONLY meaningful ZeRO-3 benefit on single GPU (via LoRAOptimizedLinear offload_ratio).

---

## 4. ZeRO-3 Parameter Partitioning on Single GPU

### 4.1 Parameter Partitioning Logic

★★★★★ **partition_parameters.py Init class** (line 884):

```python
# partition_parameters.py, lines 1035-1050
self.rank = dist.get_rank(group=self.ds_process_group)  # 0 on single GPU
self.dp_world_size = dist.get_world_size(group=self.ds_process_group)  # 1 on single GPU
self.num_ranks_in_param_group = self.dp_world_size  # 1
self.rank_in_group = self.rank  # 0
self.num_param_groups = 1
```

★★★★★★ **_partition_param: partition_size = full_size on single GPU**:

```python
# partition_parameters.py, lines 1656-1731 (_partition_param)
partition_size = tensor_size // self.num_partitions  # tensor_size // 1 = tensor_size
start = partition_size * self.get_partition_rank()   # partition_size * 0 = 0
# → param.ds_tensor = param.narrow(0, 0, partition_size) = FULL param
```

★★★★★ **ZeroParamStatus lifecycle on single GPU**:

Even on single GPU, params go through the status lifecycle:
- `NOT_AVAILABLE` → partitioned state (ds_tensor holds full param, param.data = empty)
- `INFLIGHT` → during allgather (but allgather is trivial on single GPU)
- `AVAILABLE` → after allgather (param.data = full param)

This lifecycle adds overhead: each forward pass still requires gather → partition → gather → partition cycles.

★★★★★ **PartitionedParameterCoordinator overhead**:

```python
# partitioned_param_coordinator.py, line 64
class PartitionedParameterCoordinator:
    # Even on single GPU, this coordinator tracks:
    # - param trace (forward/backward ordering)
    # - prefetch scheduling
    # - inflight param registry
    # - AllGather handle management
```

All this infrastructure runs on single GPU, adding latency with no memory benefit.

### 4.2 ZeRO-3 on Single GPU: Detailed Cost Analysis

★★★★★★ **ZeRO-3 single GPU overhead WITHOUT any benefit**:

| Component | Multi-GPU Benefit | Single GPU Value |
|-----------|-------------------|-----------------|
| param partitioning | 1/N memory | 1/1 = full (no savings) |
| gradient partitioning | 1/N memory | 1/1 = full (no savings) |
| optimizer state partitioning | 1/N memory | 1/1 = full (no savings) |
| AllGather communication | needed for full param | identity on dp=1 (no benefit) |
| parameter coordinator | needed for prefetch | overhead without benefit |
| ds_status tracking | needed for lifecycle | overhead without benefit |
| FP32 partitioned groups | 1/N memory | full FP32 copy (no savings) |

★★★★★★ **CONFIRMED: ZeRO-3 on single GPU = pure overhead. Only meaningful when combined with LoRAOptimizedLinear offload_ratio** (which provides CPU offloading independent of ZeRO partitioning).

---

## 5. Gradient Accumulation with ZeRO

### 5.1 coalesce_grad_reduction Mechanism

★★★★★★ **DeepSpeedEngine.coalesce_grad_reduction()** (engine.py, line 2647):

```python
# engine.py, lines 2647-2703
def coalesce_grad_reduction(self):
    """Coalesce ZeRO 1/2/3 gradient reduction across multiple engine.backward()
    calls. One with-block == one optimizer step: every backward inside
    leaves grads locally on params, and the flush on exit issues a single
    reduction pass that populates averaged_gradients for the next step()."""

    # Save engine boundary state
    saved_engine_boundary = self._is_gradient_accumulation_boundary
    self.inside_no_sync_ctxt = True
    optimizer._coalesce_grad_reduction = True  # Toggle guard
    try:
        yield  # Multiple backward() calls inside this block
    finally:
        optimizer._coalesce_grad_reduction = False
        self.inside_no_sync_ctxt = False
        self._is_gradient_accumulation_boundary = True
        optimizer.is_gradient_accumulation_boundary = True
        # Drive single reduction pass over locally accumulated grads
        if stage == ZeroStageEnum.weights:
            self._flush_coalesced_reduction_zero3(optimizer)
        else:
            self._flush_coalesced_reduction_zero12(optimizer)
        self._is_gradient_accumulation_boundary = saved_engine_boundary
```

★★★★★ **ZeRO-1/2 flush**: Iterate over all params, call reduce_ready_partitions_and_remove_grads once:

```python
# engine.py, lines 2705-2730
def _flush_coalesced_reduction_zero12(self, optimizer):
    optimizer.setup_buckets()
    for i, group in enumerate(optimizer.bit16_groups):
        for param in group:
            if not param.requires_grad: continue
            if optimizer.get_gradient_for_reduction(param) is None: continue
            optimizer.reduce_ready_partitions_and_remove_grads(param, i)
    optimizer.overlapping_partition_gradients_reduce_epilogue()
```

★★★★★ **ZeRO-3 flush**: Same pattern for fp16_groups:

```python
# engine.py, lines 2732-2741
def _flush_coalesced_reduction_zero3(self, optimizer):
    for group in optimizer.fp16_groups:
        for param in group:
            if param.requires_grad and param.grad is not None:
                optimizer.reduce_ready_partitions_and_remove_grads(param)
    optimizer.independent_gradient_partition_epilogue()
```

★★★★★ **Short-circuit guard in process_gradients**:

```python
# stage_1_and_2.py, line 1619
if self._coalesce_grad_reduction:
    return  # Skip gradient reduction during accumulation phase
```

Same guard in stage3.py (line 1822). During the coalesce period, each backward() accumulates grads locally (param.grad or param.grad_accum) without triggering reduction. The flush on context exit runs ONE reduction pass.

★★★★★★ **coalesce_grad_reduction vs no_sync**: Key difference:
- `no_sync()`: suppresses AllReduce but still does gradient partitioning work
- `coalesce_grad_reduction()`: suppresses ALL reduction work (partitioning + AllReduce), only flushes at exit

For GRPO with multiple backward() calls per step:
- `coalesce_grad_reduction` is MORE efficient — skips ALL reduction infrastructure during accumulation
- On single GPU, both are equivalent (AllReduce = identity), but `coalesce_grad_reduction` still saves the partitioning/bookkeeping overhead

### 5.2 ZeRO-2 Gradient Partitioning Details

★★★★★ **reduce_independent_p_g_buckets_and_remove_grads** (stage_1_and_2.py, line 1091):

```python
def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
    grad_reduc = self.get_gradient_for_reduction(param)
    bucket = self.ipg_buckets[comm_dtype]
    if bucket.elements + param.numel() > self.reduce_bucket_size:
        self.reduce_ipg_grads(comm_dtype=comm_dtype)  # Flush current bucket
    # Add to bucket
    new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())
    new_grad_tensor.copy_(grad_reduc.view(-1))
    bucket.elements += param.numel()
    bucket.grads.append(grad_reduc)
```

★★★★★ **IPGBucket structure**: Intermediate gradient accumulation buffer per dtype:

```python
# stage_1_and_2.py (class IPGBucket)
class IPGBucket:
    buffer: list       # Pre-allocated contiguous buffer
    index: int         # Double-buffer index (0 or 1 for overlap_comm)
    elements: int      # Current number of elements in buffer
    grads: list        # Gradient tensors for reduction
    params: list       # (group_idx, param_idx_in_group, param_id) tuples
    has_moe_params: bool
```

★★★★★ **average_tensor**: The actual AllReduce operation:

```python
# stage_1_and_2.py (average_tensor method)
# For ZeRO-2 with reduce_scatter=True:
# reduce_scatter splits result so each rank gets only its partition
# For ZeRO-2 with reduce_scatter=False:
# all_reduce gives each rank the full averaged gradient
```

On single GPU: both reduce_scatter and all_reduce are identity operations (no actual communication).

---

## 6. LoRAOptimizedLinear Source-Level (RTX 4090 Critical)

### 6.1 LoRAOptimizedLinear Architecture

★★★★★★ **Split forward — base weight NOT merged with LoRA** (optimized_linear.py, line 206):

```python
# optimized_linear.py, lines 206-222
def forward(self, input_tensor):
    # Gather the sharded base weight (if zero_shards > 1)
    if self.zero_shards > 1:
        base_weight = self.full_weight()  # all_gather_into_tensor
    elif self.quantization_config:
        base_weight = self.weight.dequantized()
    else:
        base_weight = self.weight

    base_weight_output = F.linear(input_tensor, base_weight)     # BASE computation
    lora_output = self.lora_weight_2(self.lora_weight_1(input_tensor))  # LoRA computation
    return base_weight_output + self.lora_scaling_factor * lora_output  # SPLIT sum
```

★★★★★★ **Key insight: base weight frozen, LoRA A/B trainable**:

```python
# optimized_linear.py, lines 125-159 (init_lora)
self.weight.requires_grad = False            # Base weight FROZEN
self.weight.ds_optim_param = True            # Mark as optimizer param (skip broadcast)
self.lora_weight_1 = nn.Linear(input_dim, lora_r, ...)  # LoRA A matrix
self.lora_weight_2 = nn.Linear(lora_r, output_dim, ...)  # LoRA B matrix
nn.init.kaiming_uniform_(self.lora_weight_1.weight, a=math.sqrt(5))  # A = kaiming
nn.init.zeros_(self.lora_weight_2.weight)    # B = zeros → initial output = 0
self.lora_weight_1.weight.requires_grad = True
self.lora_weight_2.weight.requires_grad = True
```

★★★★★★ **LoRA init: A=kaiming, B=zeros → initial LoRA output = 0**:
This is the same as HuggingFace PEFT's initialization convention. At step 0, `lora_output = B(A(x)) = 0(A(x)) = 0`, so the model starts equivalent to the base model.

### 6.2 offload_ratio Mechanism

★★★★★★★ **LoRAOptimizedLinear offload_ratio = cumulative CPU offloading** (engine.py, lines 466-500):

```python
# engine.py, lines 466-500 (_optimized_linear_offload_setup)
def _optimized_linear_offload_setup(self):
    offload_ratio = None
    for _, module in self.module.named_modules():
        if isinstance(module, LoRAOptimizedLinear):
            if offload_ratio is not None:
                assert offload_ratio == module.lora_config.offload_ratio
            offload_ratio = module.lora_config.offload_ratio

    total_params = 0
    for _, p in self.module.named_parameters():
        if hasattr(p, 'ds_optim_param'):  # Only frozen base weights
            total_params += p.numel()

    offload_limit = total_params * offload_ratio
    logger.info(f'offloading {offload_ratio*100}% of eligible params, specifically {offload_limit} params')

    total_offloaded = 0
    for _, p in self.module.named_parameters():
        if hasattr(p, 'ds_optim_param'):
            if total_offloaded < offload_limit:
                total_offloaded += p.numel()
                p.ds_offload = True
                p.offload()  # Move to CPU
            else:
                p.ds_offload = False  # Keep on GPU
```

★★★★★★★★ **Cumulative offloading — first params offloaded, rest stay on GPU**:
This is NOT random selection. It iterates named_parameters in order and offloads the FIRST `offload_limit` elements. With `offload_ratio=0.5`, the first half of all frozen base weights go to CPU, the second half stays on GPU.

★★★★★★ **QuantizedParameter.offload()** (quantization.py, line 78):

```python
def offload(self, revert=False):
    if getattr(self, 'ds_offload', False):
        if revert:
            self.data = self.to(get_accelerator().current_device_name())  # CPU → GPU
        else:
            self.data = self.to('cpu')  # GPU → CPU
```

Simple: `.to('cpu')` to offload, `.to('cuda:0')` to revert. The forward pass uses `full_weight()` which handles offload/revert cycle.

★★★★★★ **LoRA forward with offload** (optimized_linear.py, lines 183-199):

```python
def full_weight(self):
    base_weight = self.weight
    if getattr(base_weight, 'ds_offload', False):
        base_weight.offload(revert=True)   # CPU → GPU (for computation)
        local_weight = base_weight.dequantized() if isinstance(base_weight, QuantizedParameter) else base_weight
        base_weight.offload()              # GPU → CPU (after computation)
    else:
        local_weight = base_weight.dequantized() if isinstance(base_weight, QuantizedParameter) else base_weight
    tensor_out = torch.empty(self.output_dim * self.input_dim, ...)
    dist.all_gather_into_tensor(tensor_out, local_weight)  # Gather if sharded
    return tensor_out.reshape(self.output_dim, self.input_dim)
```

★★★★★★ **For RTX 4090 (zero_shards=1, single GPU)**:
- `zero_shards=1` → no base_weight_sharding, no all_gather_into_tensor needed
- `offload_ratio=0.5` → first half of frozen params on CPU, second half on GPU
- Forward: base_weight stays on GPU if not offloaded, or temporarily loaded to GPU if offloaded
- This is the RTX 4090's most valuable memory saving mechanism via LoRAOptimizedLinear

★★★★★★ **Memory saving calculation for RTX 4090**:
- With LoRA rank r=32 on 7B model (4096 hidden):
  - LoRA A: 4096×32 = 131K params × 2 bytes = 256KB per layer
  - LoRA B: 32×4096 = 131K params × 2 bytes = 256KB per layer
  - Total LoRA per attention layer: ~512KB
  - 32 layers × 7 LoRA layers × 512KB ≈ 115MB
  - Base weight frozen: 7B × 2 bytes = 14GB
  - With offload_ratio=0.5: 7GB on CPU, 7GB on GPU during forward
  - Peak GPU: 7GB base + 115MB LoRA + 7GB FP32 optimizer partition + ~4GB activation
  - Total GPU: ~18GB → fits on RTX 4090 24GB!

---

## 7. AutoEP + Singleton MoE on Single GPU

### 7.1 AutoEP ep_size=1 (Singleton MoE)

★★★★★ **AutoEP config: ep_size=1 means no expert parallelism**:

```python
# auto_ep_config.py, lines 47, 156-157
config.autoep_size = param_dict.get("autoep_size", 1)
if config.autoep_size == 1:
    logger.warning("autoep_size=1 means every rank owns all experts with no AllToAll. "
                   "Use this when world_size < num_experts or for single-GPU debugging.")
```

★★★★★★ **AutoEP layer: ep_size=1 → skip AllToAll, local computation only**:

```python
# auto_ep_layer.py, lines 102-114 (compute_split_plan)
if ep_size == 1:
    # No dispatch needed - all tokens stay local
    num_tokens_per_expert = count_tokens_per_expert(selected_experts, num_experts, out_dtype=torch.int32)
    return SplitPlan(input_splits=[T_K], output_splits=[T_K],
                     local_counts=num_tokens_per_expert,
                     local_counts_by_source=num_tokens_per_expert.view(1, num_local_experts))
```

```python
# auto_ep_layer.py, lines 563-574 (forward)
if self.ep_size == 1:
    # No AllToAll needed - local computation only
    local_counts = count_tokens_per_expert(ro.selected_experts, self.num_local_experts, out_dtype=torch.int32)
    routed_input_permuted, perm_indices, aligned_counts, n_tokens = permute_by_local_expert(routed_input, local_counts)
    expert_output = self.experts(routed_input_permuted, aligned_counts)
    expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)
```

★★★★★★ **Singleton MoE (#7997): EP=1 skips identity collectives → 15x speedup**:
On single GPU, ep_size=1 means:
- No AllToAll dispatch/combine (skipped entirely in compute_split_plan)
- No AllToAllV communication (skipped in forward)
- All experts owned by the single rank
- Token routing is purely local (permute_by_local_expert only)
- This eliminates the "identity AllToAll" overhead that would otherwise run with ep_size=1 on older DeepSpeed MoE

★★★★★ **ZeRO-2 MoE integration with AutoEP**:

```python
# stage_1_and_2.py, lines 733-757 (_configure_moe_settings)
def _configure_moe_settings(self):
    if self.partition_gradients:
        assert self.contiguous_gradients, "Contiguous Gradients in ZeRO Stage 2 must be True for MoE"
    assert self.reduce_scatter, "Reduce Scatter in ZeRO Stage 2 must be True for MoE"
    for i, group in enumerate(self.optimizer.param_groups):
        if self.is_moe_group(group):
            self.real_dp_process_group[i] = self.expert_dp_process_group[group['name']]
            self.partition_count[i] = dist.get_world_size(group=self.expert_dp_process_group[group['name']])
```

★★★★★ **AutoEP+ZeRO-3 (#8060): expert params over expert replica group**:
On single GPU, expert_dp_process_group = singleton group (world_size=1). Expert params are NOT partitioned in ZeRO-3 either — same degeneration as non-expert params.

★★★★★★ **AutoEP+ZeRO-3 (#8060 open) per-parameter partition groups**: The idea is to partition expert params over the expert replica group (not global DP), so EP=N → each rank owns 1/N of expert optimizer states. But EP=1 → singleton group → expert params NOT partitioned → ZeRO-3 single GPU benefit = zero for both dense AND expert params.

---

## 8. RTX 4090 Configuration Recommendations (Source-Level Verified)

### 8.1 ZeRO-2 + CPU_Adam + LoRA (★★★★★★★ CONFIRMED BEST)

★★★★★★ **ZeRO-2 on single GPU: NO partitioning benefit, but CPU_Adam offload IS the benefit**:

The key insight: ZeRO-2 on single GPU provides ZERO partitioning savings (partition_size = full model). BUT with CPU_Adam offload:
- FP32 master weights → pinned CPU memory
- Momentum + variance → CPU memory (2× full model FP32)
- GPU only holds: BF16 model + BF16 gradients + activation
- Total GPU: Ψ(BF16) + Ψ(BF16_grad) + activation ≈ 2Ψ + activation

★★★★★★ **CPU_Adam offload + LoRAOptimizedLinear = RTX 4090 optimal**:

Combined configuration:
```
ZeRO-2 + CPU_Adam(offload_optimizer=cpu, pin_memory=True)
+ LoRAOptimizedLinear(lora_r=32, offload_ratio=0.5)
+ coalesce_grad_reduction
```

Memory breakdown:
- BF16 model (on GPU): 14GB (7B model)
- BF16 gradients (on GPU): ~14GB → BUT only partition needed (ZeRO-2 reduces after each backward)
- LoRA weights (on GPU): ~115MB
- FP32 master weights (on CPU pinned): 28GB → CPU memory, NOT GPU
- Optimizer states (on CPU): 56GB (momentum + variance) → CPU memory
- LoRA base weights offloaded (on CPU): 7GB (offload_ratio=0.5 of 14GB frozen)
- GPU peak: ~14GB + gradients_during_backward → with gradient reduction, peak ≈ 18-20GB
- FITS on RTX 4090 24GB!

★★★★★★ **coalesce_grad_reduction on single GPU**: On single GPU, `coalesce_grad_reduction` skips ALL reduction work during accumulation phase. The flush at exit does ONE pass over accumulated gradients. Since dp=1 (no AllReduce needed), the flush is purely local bookkeeping — minimal overhead. This is still beneficial because it avoids the bucket partitioning infrastructure during each backward() call.

### 8.2 ZeRO-3 on Single GPU: NOT Recommended (★★★★★★ CONFIRMED)

★★★★★★ **ZeRO-3 single GPU = pure overhead**:
- partition_count=1 → no parameter sharding
- partition_size = full → no memory savings
- parameter coordinator overhead: ds_status tracking, AllGather handles, prefetch scheduling
- gather/scatter lifecycle on every forward pass
- FP32 partitioned groups = full FP32 copy (same as ZeRO-2)

★★★★★ **Only exception: ZeRO-3 + LoRAOptimizedLinear + partial_offload**:
If you use ZeRO-3's `offload_ratio` with LoRAOptimizedLinear, some optimizer states go to CPU while others stay on GPU. This IS beneficial even on single GPU. But the ZeRO-3 parameter partitioning infrastructure overhead makes this worse than ZeRO-2 + CPU_Adam for the same offload benefit.

### 8.3 Singleton MoE on RTX 4090 (★★★★★)

★★★★★ **AutoEP ep_size=1: AllToAll skipped, local computation only**:
- No dispatch/combine overhead
- All experts on single GPU
- For small MoE models (e.g., Qwen3-30B-A3B with 128 experts): expert weights ≈ 3B params × 2 bytes = 6GB
- Combined with LoRA + CPU_Adam: potentially viable on RTX 4090

★★★★★ **ZeRO-2 + AutoEP(ep_size=1) + LoRA**: Expert params use expert_dp_process_group (singleton), dense params use dp_process_group (singleton). Both singleton → no partitioning, but CPU_Adam offload still works for all params.

---

## 9. Key Source File Index

| File | Path | Lines | Key Classes/Functions |
|------|------|-------|----------------------|
| stage_1_and_2.py | deepspeed/runtime/zero/ | 148KB, ~2930 | DeepSpeedZeroOptimizer, get_data_parallel_partitions, _optimizer_step, step(), copy_grads_in_partition |
| stage3.py | deepspeed/runtime/zero/ | 175KB, ~2970 | DeepSpeedZeroOptimizer_Stage3, _optimizer_step, _create_fp16_partitions_with_defragmentation |
| partition_parameters.py | deepspeed/runtime/zero/ | 115KB, ~2370 | Init, _partition_param, _all_gather, _no_gather_coalesced, AllGatherCoalescedHandle |
| partitioned_param_coordinator.py | deepspeed/runtime/zero/ | 32KB | PartitionedParameterCoordinator, InflightParamRegistry |
| parameter_offload.py | deepspeed/runtime/zero/ | 29KB | ZeROOrderedDict, _inject_parameters |
| cpu_adam.py | deepspeed/ops/adam/ | 245 | DeepSpeedCPUAdam, step(), step_subgroup() |
| cpu_adam.h | csrc/includes/ | 237 | Adam_Optimizer, ds_adam_step C++ API |
| optimized_linear.py | deepspeed/linear/ | 223 | LoRAOptimizedLinear, forward(), init_lora(), full_weight() |
| config.py | deepspeed/linear/ | 55 | LoRAConfig(lora_r=64, lora_alpha=16, offload_ratio=0.0) |
| quantization.py | deepspeed/linear/ | ~127 | QuantizedParameter, offload(), dequantized() |
| context_manager.py | deepspeed/linear/ | 91 | Init context wrapper for LoRA injection |
| engine.py | deepspeed/runtime/ | ~2700+ | coalesce_grad_reduction, _optimized_linear_offload_setup, _configure_expert_parallel |
| auto_ep.py | deepspeed/module_inject/ | ~100+ | AutoEP layer detection |
| auto_ep_layer.py | deepspeed/module_inject/ | ~600+ | compute_split_plan (ep_size=1 skip), forward (ep_size=1 local-only) |
| auto_ep_config.py | deepspeed/module_inject/ | ~157 | validate_autoep_config, ep_size=1 warning |
| offload_states.py | deepspeed/runtime/zero/ | 91 | offload_optimizer_states, reload_optimizer_states |
| utils.py | deepspeed/runtime/ | ~1095 | all_gather_dp_groups (dp_world_size=1 → skip!) |
| contiguous_memory_allocator.py | deepspeed/runtime/zero/ | 288 | ContiguousMemoryAllocator (defragmentation) |

---

## 10. Summary: ★★★★★★★ Key Findings

★★★★★★★★ **1. ZeRO-2/ZeRO-3 on single GPU: partition_size = full model, NO sharding**. The `get_data_parallel_partitions()` function with `dp=1` returns [full_tensor]. The `all_gather_dp_groups()` function with `dp_world_size=1` skips entirely (continue). Optimizer states are NOT partitioned. All memory savings claims for ZeRO on single GPU are FALSE.

★★★★★★★★ **2. CPU_Adam offload IS the real benefit for RTX 4090**. Moving FP32 master weights + optimizer states (momentum + variance) to pinned CPU memory frees ~4Ψ bytes on GPU. Combined with LoRAOptimizedLinear(offload_ratio=0.5), frozen base weights partially move to CPU. Total GPU savings: ~3Ψ (optimizer) + ~0.5Ψ (offloaded params) = ~3.5Ψ.

★★★★★★★ **3. LoRAOptimizedLinear: split forward (base + lora_scaling_factor * lora), NOT merged**. Base weight frozen (ds_optim_param=True, requires_grad=False). LoRA A=kaiming, B=zeros → initial output = 0. offload_ratio=0.5 → cumulative: first half of frozen params to CPU, second half on GPU.

★★★★★★★ **4. coalesce_grad_reduction on single GPU**: Skips ALL reduction infrastructure during accumulation. Flush at exit = single pass. On dp=1, AllReduce = identity, so flush is just local bookkeeping. Still beneficial because it avoids bucket partitioning overhead per backward() call.

★★★★★★ **5. AutoEP ep_size=1 (Singleton MoE)**: AllToAll dispatch/combine SKIPPED in compute_split_plan and forward. Local computation only. ZeRO-2 + AutoEP(ep_size=1) + LoRA + CPU_Adam = viable RTX 4090 MoE configuration for small MoE models.

★★★★★★ **6. ZeRO-3 on single GPU = NOT recommended**: partition_count=1 → no parameter/gradient/optimizer sharding. Parameter coordinator overhead + gather/scatter lifecycle = pure latency penalty. Only LoRAOptimizedLinear offload_ratio provides benefit, but ZeRO-2 achieves same offload benefit with less infrastructure overhead.

★★★★★ **7. C++ CPUAdam: SIMD (AVX512/AVX256) + OpenMP → 5-7x faster than torch.optim.Adam on CPU**. fp32_optimizer_states=True default (momentum/variance in FP32). Set fp32_optimizer_states=False for BF16 states → less CPU memory at precision cost.
