# DeepSpeed ZeRO Stage 1/2 Gradient Flow Deep Reading

**Date**: 2026-07-15
**Source**: `/tmp/deepspeed-fork/deepspeed/runtime/zero/stage_1_and_2.py` (122KB, 2104 lines)
**Key class**: `DeepSpeedZeroOptimizer` (line 97)
**Purpose**: Understand the complete gradient reduction pipeline, IPG bucket system, CPU offload mechanics, and how #8061 overlap_comm+torch.compile NaN bug arises from this architecture.

---

## 1. Architecture Overview

### Data Structures (Key Memory Layout)

| Structure | Purpose | Location |
|---|---|---|
| `bit16_groups[i]` | Per-group list of trainable fp16/bf16 parameters | line 248 |
| `bit16_groups_flat[i]` | Flattened contiguous buffer for each param group | line 249 |
| `parallel_partitioned_bit16_groups[i]` | List of dp_size partitions of bit16_groups_flat | line 254 |
| `single_partition_of_fp32_groups[i]` | fp32 master copy of THIS rank's partition | line 258 |
| `params_in_partition[i]` | Parameters owned by this rank (for gradient accumulation) | line 266 |
| `params_not_in_partition[i]` | Parameters NOT owned by this rank | line 263 |

**Partitioning logic**: Each `bit16_groups_flat[i]` is split into `dp_world_size` equal partitions via `get_data_parallel_partitions()` (line 1595). Rank `partition_id` owns `parallel_partitioned_bit16_groups[i][partition_id]`. The fp32 master weights (`single_partition_of_fp32_groups`) are a clone+detach of this partition, cast to float32.

**ZeRO-1 vs ZeRO-2 distinction**: `self.partition_gradients` (line 174) controls which stage:
- **ZeRO-1**: `partition_gradients=False` -- optimizer states partitioned, gradients NOT partitioned. Each rank holds full gradients.
- **ZeRO-2**: `partition_gradients=True` -- BOTH optimizer states AND gradients partitioned. Each rank only keeps its own partition's gradients, discards the rest.

---

## 2. IPG (In-flight Parameter Gradient) Bucket System

### 2.1 Bucket Data Structures

```python
# line 435-445
self.grads_in_ipg_bucket = []          # list of gradient tensor references
self.params_in_ipg_bucket = []         # list of (group_i, param, param_id) tuples
self.elements_in_ipg_bucket = 0        # current element count in bucket
self.ipg_bucket_has_moe_params = False # flag for MoE mixed buckets
self.ipg_bucket_producer_streams = []  # CUDA streams that wrote to this bucket (for #8061 fix)
self.params_already_reduced = []       # per-param boolean, prevents double reduction
self.ipg_index = 0                     # double-buffer index (0 or 1)
```

### 2.2 IPG Buffer Allocation (Double-Buffered for overlap_comm)

The IPG buffers are allocated in `backward()` (line 2045) just before `loss.backward()`:

```python
# line 2055-2068
self.ipg_buffer = []
buf_0 = torch.empty(int(self.reduce_bucket_size), dtype=self.dtype, device=current_device)
self.ipg_buffer.append(buf_0)

if self.overlap_comm:
    buf_1 = torch.empty(int(self.reduce_bucket_size), dtype=self.dtype, device=current_device)
    self.ipg_buffer.append(buf_1)
self.ipg_index = 0
```

**Key insight**: The IPG buffer size = `reduce_bucket_size` elements (controlled by `reduce_bucket_size` parameter, default 500M). When `overlap_comm=True`, TWO buffers are allocated for double-buffering. The `ipg_index` toggles between 0 and 1.

**For Pipeline Parallelism** (non-hook mode), buffers are created in `reduce_gradients()` instead (line 710-716), with only a single buffer regardless of overlap_comm.

### 2.3 bucket_cap_mb Parameter Impact

The `reduce_bucket_size` is set from config parameter `bucket_cap_mb` (default 5e8 elements = ~500M elements). At fp16/bf16 (2 bytes per element), that is ~1 GB per buffer. For RTX 4090 dp=1:
- **overlap_comm=False**: 1 buffer x ~1 GB = ~1 GB IPG overhead
- **overlap_comm=True**: 2 buffers x ~1 GB = ~2 GB IPG overhead
- For RTX 4090 24 GB with a ~7B model (~14 GB weights), this leaves only ~10 GB, of which ~2 GB is consumed by IPG double-buffer. Recommend `bucket_cap_mb=32` (0.5 MB x 2 = 1 MB overhead for overlap_comm).

**When bucket is full**: The trigger condition is `self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size` (line 941). When this is exceeded, `reduce_ipg_grads()` is called immediately to flush the current bucket before adding the new param.

---

## 3. Gradient Reduction Pipeline: Complete Call Chain

### 3.1 Backward Hook Registration (overlap_comm or ZeRO-2)

When `self.partition_gradients` (ZeRO-2) or `self.overlap_comm` is True, backward hooks are registered (line 526-527):

```python
# line 903-919
def create_reduce_and_remove_grad_hooks(self):
    for i, param_group in enumerate(self.bit16_groups):
        for param in param_group:
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                def reduce_partition_and_remove_grads(*notneeded):
                    self.reduce_ready_partitions_and_remove_grads(param, i)
                grad_acc.register_hook(reduce_partition_and_remove_grads)
```

**Mechanism**: Uses PyTorch's `grad_fn.next_functions[0][0].register_hook()` to attach a callback that fires when the gradient for each parameter is computed during backward. This enables **gradient reduction interleaved with backward computation** -- the key to overlap_comm.

### 3.2 Hook Path (ZeRO-2 or overlap_comm)

```
backward hook fires
  -> reduce_ready_partitions_and_remove_grads(param, i)        [line 1432]
     -> checks: self.partition_gradients OR self.is_gradient_accumulation_boundary
     -> reduce_independent_p_g_buckets_and_remove_grads(param, i)  [line 938]
```

### 3.3 Non-Hook Path (ZeRO-1, no overlap_comm, no PP)

```
reduce_gradients(pipeline_parallel=False)                       [line 705]
  -> for each param: reduce_ready_partitions_and_remove_grads(param, i)  [line 723]
     -> same as above
  -> overlapping_partition_gradients_reduce_epilogue()            [line 725]
     -> independent_gradient_partition_epilogue()                  [line 873]
```

### 3.4 reduce_independent_p_g_buckets_and_remove_grads() -- Full Flow

This is the **core function** that accumulates gradients into the IPG bucket (line 938-980):

```
Step 1: Get gradient for reduction
  grad_reduc = get_gradient_for_reduction(param)  [line 940]
  -> param.grad_accum.to(dtype) OR param.grad, depending on use_grad_accum_attribute

Step 2: Check bucket overflow
  if elements_in_ipg_bucket + param.numel() > reduce_bucket_size:
      reduce_ipg_grads()                           [line 943]
      if contiguous_gradients AND overlap_comm:
          ipg_index = 1 - ipg_index                 [line 944-946, DOUBLE-BUFFER SWAP]

Step 3: Assert no double reduction
  assert params_already_reduced[param_id] == False  [line 950]

Step 4: Copy gradient into IPG buffer (contiguous path)
  if contiguous_gradients:
      if param.numel() > reduce_bucket_size:        # EXTRA-LARGE PARAM
          extra_large_param_to_reduce = param       [line 957]
      else:
          new_grad_tensor = ipg_buffer[ipg_index].narrow(0, elements_in_ipg_bucket, param.numel())
          new_grad_tensor.copy_(grad_reduc.view(-1))  [line 960-961]
          grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)  [line 962]

Step 5: Update bucket bookkeeping
  elements_in_ipg_bucket += param.numel()           [line 964]
  grads_in_ipg_bucket.append(grad_reduc)            [line 968]
  params_in_ipg_bucket.append((i, param, param_id)) [line 969]
  ipg_bucket_producer_streams.append(current_stream())  [line 974, #8061 FIX]
  if is_moe_param(param): ipg_bucket_has_moe_params = True  [line 978]
```

**Critical observation for #8061**: At line 960-961, `new_grad_tensor.copy_(grad_reduc.view(-1))` copies the gradient into the IPG buffer. This `copy_()` runs on `current_stream()`. When `torch.compile` schedules backward hooks on **different streams** per parameter, multiple `copy_()` operations are running on different CUDA streams simultaneously, all writing to the same IPG buffer. The stream is recorded at line 974.

### 3.5 reduce_ipg_grads() -- Full Flow

When the bucket is full or at epilogue, this function performs the actual reduction (line 1373-1429):

```
Step 1: Choose reduction path based on contiguous_gradients
  if contiguous_gradients:
      if extra_large_param_to_reduce is not None:
          # Single param exceeds bucket size -- reduce directly
          average_tensor(extra_large_grad_reduc.view(-1))    [line 1381]
      else:
          # Normal bucket -- reduce the IPG buffer slice
          average_tensor(ipg_buffer[ipg_index].narrow(0, 0, elements_in_ipg_bucket))  [line 1384]
  else:
      # Fallback: non-contiguous path
      buffered_reduce_fallback(None, grads_in_ipg_bucket, ...)  [line 1386-1388]

Step 2: Post-reduction cleanup (stream-dependent)
  stream = reduction_stream (if overlap_comm)
           OR current_stream() (if cpu_offload OR normal)

  with stream:
      for _, param, param_id in params_in_ipg_bucket:
          params_already_reduced[param_id] = True          [line 1408]

          if partition_gradients:  # ZeRO-2
              if NOT in_current_partition:
                  if overlap_comm AND NOT contiguous:
                      # Deferred clear -- store in previous_reduced_grads
                      previous_reduced_grads.append(param)  [line 1416]
                  else:
                      clear_grad_attribute(param)            [line 1418]
              elif contiguous_gradients:
                  copy_grads_in_partition(param)              [line 1420]
          else:  # ZeRO-1
              if contiguous AND in_current_partition:
                  copy_grads_in_partition(param)              [line 1423]

Step 3: Reset bucket state
  grads_in_ipg_bucket = []
  params_in_ipg_bucket = []
  ipg_bucket_has_moe_params = False
  elements_in_ipg_bucket = 0
  ipg_bucket_producer_streams = []                        [line 1425-1429]
```

---

## 4. average_tensor() -- The Reduction Engine

### 4.1 Stream Synchronization (#8061 Fix)

```python
# line 1055-1067
def average_tensor(self, tensor):
    if self.overlap_comm:
        stream = self.reduction_stream
        if not get_accelerator().resolves_data_dependency():
            # Wait for ALL producer streams before reading IPG buffer
            for producer_stream in self.ipg_bucket_producer_streams:
                stream.wait_stream(producer_stream)        [line 1064]
    else:
        stream = current_stream()
```

**The #8061 bug root cause**: Before this fix, `average_tensor()` only ran on `reduction_stream` and did NOT wait for producer streams. When `torch.compile` scheduled backward hooks on multiple streams, the `copy_()` operations on those streams might not have completed before `average_tensor()` read the IPG buffer. This is a **read-before-write race**: the reduction stream starts reading the IPG buffer while other streams are still writing gradient data into it. Result: NaN/Inf in gradients.

**The fix**: Record each producer stream in `ipg_bucket_producer_streams` (line 974), then in `average_tensor()` make `reduction_stream` wait for ALL of them before reading (line 1064-1065).

### 4.2 Two Reduction Paths

**Path A: reduce_scatter=False** (simple allreduce, line 1070-1072):
```python
self.gradient_reduction_w_predivide(tensor)
  -> div by predivide_factor OR just all_reduce
  -> postscale: allreduce then rescale
  -> prescale: scale down, allreduce, scale up
```

**Path B: reduce_scatter=True** (efficient, line 1074-1149):

This is the **default and preferred** path. It performs reduce-scatter instead of allreduce, so each rank only receives its own partition of the averaged gradients:

```
Step 1: Build rank_and_offsets mapping (line 1078-1128)
  For each param in params_in_ipg_bucket:
    - Determine which DP rank(s) own each slice of this param's gradient
    - Calculate (partition_id, bucket_offset, numel) for each slice
    - Merge consecutive slices belonging to the same rank

Step 2: Divide by dp_world_size (line 1130)
  tensor.div_(dp_world_size / sequence_parallel_size)

Step 3: Create per-process-group buckets (line 1132-1143)
  For each (dst_rank, grad_slice):
    - Bucket by process_group (for MoE: expert_dp_process_group vs dp_process_group)
    - use_multi_rank_bucket_allreduce: group by process_group alone
    - Otherwise: group by (dst_rank, process_group)

Step 4: reduce-scatter each bucket (line 1144-1149)
  for bucket_key in buckets:
      allreduce_and_scatter(buckets[bucket_key], ...)
```

### 4.3 allreduce_and_scatter() (line 1027-1053)

Splits the bucket into sub-buckets by `numel_per_bucket`, then calls `allreduce_and_copy_with_multiple_ranks()` for each sub-bucket. This function:
1. Flattens all tensors in the sub-bucket via `allreduce_bucket()`
2. Performs all_reduce on the flattened tensor
3. For each tensor in the sub-bucket, only copies back the result if `dist.get_rank() == bucket_rank` (i.e., this rank owns that slice)

This is effectively a **reduce-scatter** operation, implemented as allreduce + selective copy.

---

## 5. allreduce_and_copy() -- Non-Contiguous Fallback

Used when `contiguous_gradients=False` (line 1536-1557):

```python
def allreduce_and_copy(self, small_bucket, rank=None, divide=True, process_group=None):
    if self.overlap_comm:
        if not resolves_data_dependency():
            synchronize()                                [line 1540]
        _clear_previous_reduced_grads()                  [line 1542]
        stream = reduction_stream                        [line 1543]
    else:
        stream = current_stream()

    with stream:
        allreduced = allreduce_bucket(small_bucket, ...)  [line 1548-1554]
        for buf, synced in zip(small_bucket, unflatten(allreduced, ...)):
            buf.copy_(synced)                             [line 1556-1557]
```

**Note**: `allreduce_and_copy()` uses `get_accelerator().synchronize()` (full device sync) at line 1540 when overlap_comm is True. This is a **heavier synchronization** than the targeted `stream.wait_stream()` used in `average_tensor()`. This is because the non-contiguous path does not use an IPG buffer -- gradients are in their original param.grad tensors, and a full sync is needed to ensure backward computation has completed.

---

## 6. Continuous Bucket Reduction vs Traditional Full-Bucket

### 6.1 Traditional: reduce_independent_p_g_buckets_and_remove_grads (line 938)

This is the **hook-based continuous reduction** path. Each time a backward hook fires:
1. The param's gradient is added to the IPG bucket
2. If the bucket overflows, it is immediately reduced (`reduce_ipg_grads()`)
3. After reduction, the IPG index is swapped (double-buffer) if overlap_comm
4. The next param starts filling the NEXT bucket

This provides **pipeline-style** reduction: backward computation continues filling the next bucket while the current bucket is being reduced on `reduction_stream`.

### 6.2 Continuous: reduce_ready_partitions_and_remove_grads (line 1432)

```python
def reduce_ready_partitions_and_remove_grads(self, param, i):
    if self.partition_gradients or self.is_gradient_accumulation_boundary:
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
```

This is a **thin wrapper** that conditionally delegates. For ZeRO-2 (`partition_gradients=True`), it always calls the bucket reduction. For ZeRO-1 (`partition_gradients=False`), it only calls during gradient accumulation boundaries (when `is_gradient_accumulation_boundary=True`), because ZeRO-1 does not need to discard gradients during micro-steps.

### 6.3 Epilogue: independent_gradient_partition_epilogue (line 763-807)

After all backward hooks have fired (or after explicit `reduce_gradients()` for non-hook mode):

```
Step 1: Flush remaining IPG bucket
  reduce_ipg_grads()                                    [line 765]

Step 2: Reset params_already_reduced flags
  for i: params_already_reduced[i] = False              [line 770-771]

Step 3: overlap_comm cleanup
  if overlap_comm:
      synchronize()                                     [line 774-775]
      _clear_previous_reduced_grads()                    [line 777]

Step 4: Build averaged_gradients (non-offload path)
  for each param_group i:
      averaged_gradients[i] = get_flat_partition(
          params_in_partition[i], first_offset[i], partition_size[i],
          dtype=gradient_accumulation_dtype, device=current_device)

Step 5: Release IPG buffers and zero_grad
  _release_ipg_buffers()                                [line 801]
  zero_grad(set_to_none=True)                            [line 806]
```

**Key**: `averaged_gradients` holds the reduced, averaged gradients for this rank's partition only. These are fp32 (or gradient_accumulation_dtype) tensors that will be used in the optimizer step.

---

## 7. ZeRO-2 Optimizer Step Flow (Single GPU, dp=1)

### 7.1 Step() Overview (line 1833-1951)

```
Step 0: Reset micro_step_id
  micro_step_id = INITIAL_MICRO_STEP_ID = -1             [line 1837]

Step 1: Check overflow (fp16 only)
  if dtype == torch.float16:
      check_overflow()                                    [line 1842-1843]

Step 2: Handle overflow
  if overflow:
      zero_grad(set_to_none=True)
      reset_cpu_buffers() (if cpu_offload) OR averaged_gradients = {} (if not)
      return (skip optimizer step)                        [line 1847-1860]

Step 3: Compute scaled global gradient norm
  scaled_global_grad_norm = scaled_global_norm()           [line 1864]
  _global_grad_norm = scaled_global_grad_norm / prev_scale [line 1865]
```

### 7.2 CPU Offload Step (dp=1, ZeRO-2 + DeepSpeedCPUAdam)

```
For each param_group i:
  Step 3a: Get fp32 gradient partition (already in CPU from copy_grads_in_partition)
    single_grad_partition = single_partition_of_fp32_groups[i].grad    [line 1873]

  Step 3b: Unscale and clip gradients
    unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)  [line 1874]
    -> combined_scale = loss_scale * clip_factor
    -> grad.data.mul_(1.0 / combined_scale)

  Step 3c: Optimizer step on CPU
    _optimizer_step(i)                                    [line 1878]
    -> optimizer.param_groups = [original_param_groups[i]]
    -> optimizer.step()  (DeepSpeedCPUAdam runs on CPU)
    -> optimizer.param_groups = original_param_groups

  Step 3d: Copy updated fp32 weights back to GPU bit16 partition
    bit16_partitions[partition_id].data.copy_(
        fp32_partition.to(current_device).data)           [line 1888-1889]
```

**Note**: For dp=1, `partition_id=0`, so `bit16_partitions[0]` is the entire model (no actual partitioning). The CPU optimizer runs on `single_partition_of_fp32_groups[0]` which is the full fp32 master copy.

### 7.3 Non-Offload Step (dp=1, ZeRO-2, GPU optimizer)

```
For each param_group i:
  Step 3a: Free gradients of params NOT in this partition (ZeRO-2)
    free_grad_in_param_list(params_not_in_partition[i])  [line 1894]

  Step 3b: Flatten averaged_gradients into single_grad_partition
    if partition_id == dp_world_size - 1:
        # Last partition may need alignment padding
        single_grad_partition = flatten_dense_tensors_aligned(
            averaged_gradients[i], int(partition_size[i])) [line 1899-1901]
    else:
        single_grad_partition = flatten(averaged_gradients[i]) [line 1903-1904]

  Step 3c: Assign to fp32 partition
    single_partition_of_fp32_groups[i].grad = single_grad_partition [line 1909]

  Step 3d: Free averaged_gradients
    free_grad_in_param_list(params_in_partition[i])
    averaged_gradients[i] = None                          [line 1911-1913]

  Step 3e: Unscale and clip
    unscale_and_clip_grads([single_grad_partition], ...)  [line 1915]

  Step 3f: Optimizer step
    _optimizer_step(i)                                    [line 1921]

  Step 3g: Copy fp32 -> bit16, free fp32 grad
    single_partition_of_fp32_groups[i].grad = None        [line 1923]
    bit16_partitions[partition_id].data.copy_(fp32_partition.data) [line 1927]
```

### 7.4 All-Gather Updated Weights (All Ranks)

```
Step 4: All-gather the updated weight partitions from all ranks
  all_gather_dp_groups(
      groups_flat=bit16_groups_flat,
      partitioned_param_groups=parallel_partitioned_bit16_groups,
      dp_process_group=real_dp_process_group,
      allgather_bucket_size=allgather_bucket_size)        [line 1937-1941]

  -> For dp=1: dp_world_size==1, SKIPS all-gather entirely  [line 977-980]

Step 5: Update model bit16 weights from flat buffer
  _update_model_bit16_weights(i)                          [line 1946]
  -> unflatten bit16_groups_flat -> update round_robin_bit16_groups
  -> map back to original bit16_groups ordering
```

**dp=1 critical insight**: At dp=1, `all_gather_dp_groups` is COMPLETELY SKIPPED (line 977-980: `if dp_world_size == 1: continue`). The optimizer updates `single_partition_of_fp32_groups[0]` on CPU, then copies back to `bit16_partitions[0]` which IS the entire flat weight buffer. No communication needed.

---

## 8. CPU_Adam Offload Mechanics (ZeRO-2, dp=1)

### 8.1 Gradient Path: GPU -> CPU

The gradient flow for CPU offload happens during backward, inside `copy_grads_in_partition()` (line 1337-1350):

```python
def copy_grads_in_partition(self, param):
    if self.cpu_offload:
        # Step 1: Accumulate gradient in CPU (for gradient_accumulation_steps > 1)
        if self.gradient_accumulation_steps > 1:
            async_accumulate_grad_in_cpu_via_gpu(param)  [line 1341]

        # Step 2: At boundary, compute norm and check overflow on GPU
        if self.is_gradient_accumulation_boundary:
            set_norm_for_param_grad_in_gpu(param)         [line 1344]
            update_overflow_tracker_for_param_grad(param)  [line 1346]
            # Step 3: Copy gradient to fp32 CPU buffer
            async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)  [line 1348]
        return
```

### 8.2 async_accumulate_grad_in_cpu_via_gpu (line 1201-1246)

For gradient accumulation (multiple micro-steps before optimizer step):

```
For micro_step_id == 0 (first micro-step):
  - Copy param.grad_accum to accumulated_grads_in_cpu[param_id] on CPU
  - (Non-blocking copy via pin_memory)

For micro_step_id > 0 (subsequent micro-steps):
  - Copy accumulated_grads_in_cpu[param_id] to GPU temp buffer (dest_buffer)
  - Add to param.grad_accum on GPU
  - Then copy back to CPU (accumulated_grads_in_cpu)
```

**Key**: The accumulation happens on GPU for the add_ operation, but the persistent storage is on CPU. This avoids keeping all accumulated gradients on GPU.

### 8.3 set_norm_for_param_grad_in_gpu (line 1261-1274)

Computes L2 norm of the gradient slice that belongs to this partition, stored in `norm_for_param_grads[param_id]`. This is needed for global gradient norm computation during the optimizer step.

### 8.4 async_inplace_copy_grad_to_fp32_buffer_from_gpu (line 1276-1292)

At the gradient accumulation boundary (last micro-step):

```
dest_tensor = single_partition_of_fp32_groups[i].grad.narrow(dest_offset, num_elements)
src_tensor = grad_accum.narrow(source_offset, num_elements)
if not fp16_master_weights_and_gradients:
    src_tensor = src_tensor.float()                     # Cast to fp32
dest_tensor.copy_(src_tensor, non_blocking=True)        # GPU -> CPU copy
param.grad = None                                       # Free GPU gradient
```

**The fp32 gradient partition** is stored in `single_partition_of_fp32_groups[i].grad`, which is a CPU tensor (pinned memory if `cpu_offload_pin_memory=True`). The non-blocking copy enables overlap with subsequent backward operations.

### 8.5 CPU Optimizer State Layout

For DeepSpeedCPUAdam with ZeRO-2:
- **Weights**: `single_partition_of_fp32_groups[i]` on CPU (fp32, pinned memory)
- **Gradients**: `single_partition_of_fp32_groups[i].grad` on CPU (fp32, pinned memory)
- **Optimizer states** (m, v): Inside DeepSpeedCPUAdam, also on CPU

At dp=1, all of these are the FULL model (no partitioning), each consuming:
- fp32 weights: ~2x model size (e.g., 7B model = ~28 GB CPU memory)
- fp32 gradients: ~2x model size
- Adam m+v: ~4x model size
- **Total CPU RAM needed**: ~8x model size = ~56 GB for 7B model

---

## 9. #8061 Bug: overlap_comm + torch.compile NaN -- Root Cause in This Architecture

### 9.1 The Race Condition

The bug occurs in this sequence:

```
1. backward() starts, allocates ipg_buffer[0] and ipg_buffer[1]  [line 2056-2067]
2. torch.compile schedules backward hooks on MULTIPLE CUDA streams
3. Each hook fires reduce_independent_p_g_buckets_and_remove_grads()
4. Inside this function, copy_(grad_reduc.view(-1)) into ipg_buffer  [line 961]
   - This copy_() runs on the stream where the hook was scheduled
   - With torch.compile, hooks may be on DIFFERENT streams
5. When bucket is full, reduce_ipg_grads() is called
6. average_tensor() reads ipg_buffer[ipg_index] on reduction_stream  [line 1384]
7. **RACE**: reduction_stream starts reading BEFORE all copy_() operations complete
   - Some gradient slices in ipg_buffer contain UNINITIALIZED/PARTIAL data
   - Result: NaN or Inf in the averaged gradient
```

### 9.2 The Fix (already applied in source)

The fix adds two mechanisms:

**A. Producer stream tracking** (line 974):
```python
self.ipg_bucket_producer_streams.append(get_accelerator().current_stream())
```

**B. Stream wait in average_tensor()** (line 1059-1065):
```python
if not get_accelerator().resolves_data_dependency():
    for producer_stream in self.ipg_bucket_producer_streams:
        stream.wait_stream(producer_stream)
```

This makes `reduction_stream` wait for ALL streams that wrote data into the IPG bucket before reading from it.

### 9.3 Why dp=1 RTX 4090 is Safe from #8061

At dp=1, `reduce_scatter` cannot be used (reduce_scatter requires dp>=2). The code uses `allreduce` or skips reduction entirely. But the **real reason dp=1 is safe**: at dp=1, `overlap_comm` provides NO benefit (there is nothing to overlap with -- allreduce/reduce-scatter with 1 rank is identity). Setting `overlap_comm=True` on dp=1 is wasteful and DANGEROUS.

**RECOMMENDATION**: ALWAYS set `overlap_comm=False` on dp=1 RTX 4090. The #8061 bug only manifests with `overlap_comm=True`, and overlap_comm provides zero benefit at dp=1.

---

## 10. Complete Gradient Flow Timeline (ZeRO-2, dp=1, CPU Offload)

```
T0: backward(loss) called
    -> Allocate ipg_buffer (1 buffer, overlap_comm=False at dp=1)
    -> micro_step_id += 1
    -> loss.backward() starts PyTorch autograd engine

T1..Tn: Backward hooks fire for each param (reverse order)
    For each param:
    a. reduce_ready_partitions_and_remove_grads(param, i)
    b. reduce_independent_p_g_buckets_and_remove_grads(param, i)
       - grad_reduc = param.grad
       - If bucket would overflow: reduce_ipg_grads() (flush current bucket)
       - copy_(grad_reduc) into ipg_buffer at current offset
       - elements_in_ipg_bucket += param.numel()
       - Record in params_in_ipg_bucket

    When bucket is flushed (reduce_ipg_grads):
    c. average_tensor(ipg_buffer.narrow(0, 0, elements_in_ipg_bucket))
       - dp=1: reduce_scatter is identity, tensor.div_(1) = no-op
       - Result: averaged gradient slices for this rank's partition
    d. For each param in bucket:
       - params_already_reduced[param_id] = True
       - if is_param_in_current_partition[param_id]:
           copy_grads_in_partition(param)
           -> set_norm_for_param_grad_in_gpu(param)  [compute local norm]
           -> update_overflow_tracker_for_param_grad(param)  [check inf/nan]
           -> async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)
              [copy gradient slice to CPU fp32 buffer, non_blocking=True]
           -> param.grad = None  [free GPU gradient]
       - Reset bucket state

Tn+1: Epilogue (after all hooks done OR explicit reduce_gradients)
    independent_gradient_partition_epilogue():
    e. reduce_ipg_grads()  [flush final partial bucket]
    f. Reset params_already_reduced
    g. (overlap_comm: sync + clear previous reduced grads)
    h. Build averaged_gradients from partition params
       -> For dp=1 cpu_offload: gradients are already in CPU fp32 buffer
    i. _release_ipg_buffers()
    j. zero_grad(set_to_none=True)

Tn+2: optimizer.step() called
    Step 1: check_overflow() (fp16 only)
    Step 2: scaled_global_norm()
       -> complete_grad_norm_calculation_for_cpu_offload(params_in_partition[i])
       -> Sum of norm_for_param_grads[param_id]^2 across all params
       -> all_reduce across dp_process_group (dp=1: identity)
    Step 3: For each param_group i:
       - single_grad_partition = single_partition_of_fp32_groups[i].grad  [CPU fp32]
       - unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
         -> combined_scale = loss_scale * max(1, (total_norm/loss_scale + 1e-6) / clip_grad)
         -> grad.data.mul_(1.0 / combined_scale)
       - _optimizer_step(i)
         -> DeepSpeedCPUAdam.step() on CPU (updates fp32 weights + m/v states)
       - Copy updated fp32 weights to GPU:
         bit16_partitions[0].data.copy_(fp32_partition.to(device).data)

    Step 4: all_gather_dp_groups()
       -> dp_world_size==1: SKIP entirely

    Step 5: _update_model_bit16_weights(i)
       -> Unflatten bit16_groups_flat and update model param.data references
```

---

## 11. Key Data Sizes (dp=1, RTX 4090, 7B Model)

| Item | Size | Location |
|---|---|---|
| fp16/bf16 weights | ~14 GB | GPU (bit16_groups_flat) |
| fp32 master weights (partition) | ~28 GB | CPU (single_partition_of_fp32_groups) |
| fp32 gradients (partition) | ~28 GB | CPU (single_partition_of_fp32_groups.grad) |
| Adam m + v states | ~56 GB | CPU (DeepSpeedCPUAdam internal) |
| IPG buffer (1x, overlap_comm=False) | ~1 GB | GPU (reduce_bucket_size default) |
| Norm tracking | ~56 KB | CPU (norm_for_param_grads dict) |
| **Total GPU** | ~15 GB | Leaves ~9 GB for activations + KV cache |
| **Total CPU RAM** | ~112 GB | Needs ample system RAM |

**With bucket_cap_mb=32 (recommended for RTX 4090)**: IPG buffer = ~0.5 MB (32M elements x 2 bytes), negligible.

---

## 12. Critical Observations for GRPO on RTX 4090

1. **overlap_comm MUST be False on dp=1**: No benefit, only risk (#8061). Saves ~1 GB GPU memory by avoiding double-buffer.

2. **reduce_scatter is identity at dp=1**: The reduce-scatter path in `average_tensor()` still runs the code but `tensor.div_(1)` and `allreduce_and_scatter` with 1 rank is effectively a no-op. The gradient is just copied back to the same place.

3. **ZeRO-1 vs ZeRO-2 at dp=1**: Both have identical behavior at dp=1 (one partition = entire model). ZeRO-2's gradient partitioning is moot. ZeRO-2 is still preferred because it frees non-partition gradients sooner (less peak GPU memory during backward), but at dp=1 there is only one partition so all gradients are "in partition."

4. **gradient_accumulation_steps micro-step handling**: `micro_step_id` starts at -1, incremented in `backward()`. At `micro_step_id > 0`, `async_accumulate_grad_in_cpu_via_gpu()` adds to CPU accumulated buffer. At the boundary (last micro-step), gradients are copied to the fp32 CPU buffer for optimizer step.

5. **clip_grad default**: The `clip_grad` parameter defaults to 0.0 (no clipping) in the constructor. After #8068 (merged June 2026), the default changed to 1.0. **Always set explicitly** for GRPO.

6. **loss_scale**: For bf16, `loss_scale` is always 1.0 (no dynamic scaling). The `unscale_and_clip_grads()` effectively divides by `combined_scale = clip_factor * loss_scale = clip_factor` when loss_scale=1.0.

7. **The gradient norm computation** for CPU offload is a two-phase process:
   - Phase 1 (during backward): `set_norm_for_param_grad_in_gpu()` computes per-param norm on GPU
   - Phase 2 (during step): `complete_grad_norm_calculation_for_cpu_offload()` sums squared norms across all params, then all_reduce across dp ranks

8. **fp16_master_weights_and_gradients mode**: Special mode for DeepSpeedCPUAdam where both master weights AND gradients are kept in fp16 (not fp32). Saves CPU memory but reduces precision. Not recommended for GRPO where gradient accuracy matters.

---

## 13. Function Call Tree Summary

```
backward(loss)
  |> allocate ipg_buffer(s)
  |> loss.backward()  [PyTorch autograd]
  |   |> backward hooks fire per param:
  |      |> reduce_ready_partitions_and_remove_grads(param, i)
  |         |> reduce_independent_p_g_buckets_and_remove_grads(param, i)
  |            |> get_gradient_for_reduction(param)
  |            |> [if bucket overflow] reduce_ipg_grads()
  |            |   |> average_tensor(ipg_buffer slice OR extra_large_param)
  |            |   |   |> [overlap_comm] reduction_stream.wait_stream(producer_streams)
  |            |   |   |> [reduce_scatter=True] build rank_and_offsets, allreduce_and_scatter
  |            |   |   |> [reduce_scatter=False] gradient_reduction_w_predivide
  |            |   |> post-reduction cleanup:
  |            |      |> [ZeRO-2, in_partition] copy_grads_in_partition(param)
  |            |      |   |> [cpu_offload] async_accumulate + norm + overflow + copy_to_cpu
  |            |      |   |> [no offload] copy to grads_in_partition buffer
  |            |      |> [ZeRO-2, NOT in_partition] clear_grad_attribute OR deferred clear
  |            |> copy_() gradient into ipg_buffer
  |> [if use_grad_accum_attribute] fill_grad_accum_attribute()

step()
  |> check_overflow() [fp16 only]
  |> scaled_global_norm()
  |   |> [cpu_offload] complete_grad_norm_calculation_for_cpu_offload()
  |   |> [no offload] get_grad_norm_direct(averaged_gradients)
  |> for each param_group:
  |   |> [cpu_offload] unscale_and_clip + _optimizer_step + copy fp32->bit16
  |   |> [no offload] flatten averaged_gradients + unscale_and_clip + _optimizer_step + copy
  |> all_gather_dp_groups() [dp=1: SKIP]
  |> _update_model_bit16_weights()
```

---

## 14. Comparison: overlap_comm=True vs False Pipeline

### overlap_comm=False (RECOMMENDED for dp=1)

```
Timeline (single stream):
  backward_hook_1 -> copy to bucket -> backward_hook_2 -> copy to bucket
  -> ... -> bucket_full -> reduce_ipg_grads() on default stream -> ... -> epilogue
  All operations sequential on default stream. No races. Simple.
```

### overlap_comm=True (DANGEROUS for dp=1, MANDATORY stream sync fix)

```
Timeline (two streams):
  default_stream: backward_hook -> copy_ to ipg_buffer[0] -> backward_hook -> copy_ to ipg_buffer[0]
                   -> bucket_full -> swap ipg_index=1 -> backward_hook -> copy_ to ipg_buffer[1]
  reduction_stream: reduce_ipg_grads() reads ipg_buffer[0] -> average_tensor -> allreduce/scatter
                     -> copy_grads_in_partition -> ...

  RACE: reduction_stream may read ipg_buffer[0] before all copy_() on default_stream complete.
  FIX: reduction_stream.wait_stream(ALL producer_streams) before reading.

  At dp=1: No actual benefit (allreduce is identity), only risk. NEVER enable.
```

---

*End of deep reading. Source: /tmp/deepspeed-fork/deepspeed/runtime/zero/stage_1_and_2.py*
