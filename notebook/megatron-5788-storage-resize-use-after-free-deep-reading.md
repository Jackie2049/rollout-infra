# Megatron #5788: StorageResizeBasedBucketAllocator Use-After-Free

## Bug Overview

**Title**: `StorageResizeBasedBucketAllocator.free` resizes gather storage to zero without `record_stream` (use-after-free race)
**Author**: yuhezhang-ai
**Created**: July 13, 2026
**Status**: OPEN, 0 comments
**File**: `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`

## Root Cause

In `param_and_grad_buffer.py`, the `StorageResizeBasedBucketAllocator.free()` method frees temporary all-gather bucket storage without recording the consuming CUDA stream:

```python
class StorageResizeBasedBucketAllocator(TemporaryBucketAllocator):
    def free(self, bucket_id: int):
        if bucket_id in self.buckets:
            _free_storage(self.buckets[bucket_id].data)  # <-- BUG: no record_stream
```

**Mechanism**:
1. Megatron-FSDP allocates temporary all-gather buckets on a **dedicated parameter-gather CUDA stream**.
2. TP/FSDP compute kernels consume these parameters on the **default compute stream**.
3. When `free()` calls `_free_storage()`, the storage is returned to the CUDA caching allocator.
4. Without `record_stream()`, the caching allocator may **recycle the memory immediately** for a new allocation.
5. If an in-flight compute kernel is still reading from the recycled memory → **use-after-free** → intermittent numerical corruption or illegal memory access.

## Fix

Add `record_stream` on the bucket tensor before freeing, ensuring the caching allocator waits until all kernels on the consuming stream have completed:

```python
def free(self, bucket_id: int):
    if bucket_id in self.buckets:
        self.buckets[bucket_id].data.record_stream(torch.cuda.current_stream())
        _free_storage(self.buckets[bucket_id].data)
```

## Intermittent Nature

The bug is difficult to trigger deterministically because:
- It depends on exact CUDA kernel timing and stream scheduling
- It requires the caching allocator to recycle the freed block before the consumer finishes
- `compute-sanitizer` can surface the cross-stream violation

## Cross-Framework Pattern Family: `cuda_stream_use_after_free`

This bug is the **same pattern** as DeepSpeed #8061 (overlap_comm NaN), now resolved via #8080 on July 14.

| Bug | Framework | Symptom | Root Cause | Fix |
|-----|-----------|---------|------------|-----|
| #5788 | Megatron | Intermittent numerical corruption under overlapped param gather | No record_stream on resized storage before free | Add record_stream(current_stream) |
| #8061 | DeepSpeed | overlap_comm+torch.compile=NaN | IPG bucket copy_ on multiple streams, average_tensor() only waits current stream | Wait ALL producer streams per IPG bucket |
| #45552 | vLLM | CuMemAllocator sleep/wake CUDART illegal-memory crash | In-flight kernels race cuMemUnmap + cudaMemcpy | Add cuda.synchronize() before unmap |
| #46125 | vLLM | Stale KV cache after weight update (reverted #45093) | Stale cache kernels race with new weight state | Reverted |

**Common mechanism**: In all cases, overlapped/asynchronous parameter or gradient communication splits work across multiple CUDA streams. Buffer release happens on one stream while kernels on another stream are still in-flight.

## RTX 4090 GRPO Impact

This bug affects any training setup using:
1. Megatron-FSDP with `overlap_param_gather=True`
2. Multi-stream parameter gather (which is the default with FSDP overlap)

For RTX 4090 GRPO training:
- RTX 4090 has compute capability 8.9 (Ada), not Hopper
- Megatron-FSDP is not the primary training path for RTX 4090 (verl+FSDP is)
- But the **pattern** is universal: any overlapped weight sync between rollout and training is vulnerable

## Contribution Opportunity

**Tier 1 UNIQUE**: Comment linking #5788 ↔ #8061 ↔ #45552 as same pattern family. The DeepSpeed #8061 resolution (#8080) provides a production-confirmed template.

Unlike #8061 (which was resolved), #5788 has **0 comments** and is straightforward to fix. The issue author already provides the correct fix in the issue description.

The cross-framework connection is our unique insight — no existing comment makes this link.
