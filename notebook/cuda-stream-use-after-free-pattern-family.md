# CUDA Stream Use-After-Free Pattern Family: Megatron #5788 × DeepSpeed #8061 × vLLM #45552

## Pattern Family: `cuda_stream_use_after_free_in_overlapped_comm`

All three bugs share the same root cause: **in overlapped/async parameter gather, storage is resized/released on one CUDA stream while another stream may still be reading from it**.

| Bug | Framework | Symptom | Root Cause | Resolution |
|-----|-----------|---------|------------|------------|
| #8061 | DeepSpeed | overlap_comm+torch.compile=NaN | IPG bucket copy_ on multiple streams, average_tensor() only waits current_stream → reads bucket before all producer streams complete | MERGED #8080: wait ALL producer streams per IPG bucket |
| #5788 | Megatron | intermittent numerical corruption under overlapped param gather | StorageResizeBasedBucketAllocator.free resizes gather storage to zero without record_stream → use-after-free race | OPEN, no fix yet |
| #45552 | vLLM | CuMemAllocator sleep/wake CUDART illegal-memory crash | In-flight kernels race cuMemUnmap + cudaMemcpy → no cuda.synchronize() between unmap and access | OPEN, no fix yet |

## Common Pattern

1. **Parameter/gradient gather is split across multiple CUDA streams** for overlap with compute
2. **Storage/bucket release happens before all producer streams finish**
3. **Consumer reads stale/freed memory → NaN, corruption, or crash**
4. **Bug is intermittent and hard to reproduce** — depends on exact kernel timing

## Why This Pattern Matters for GRPO

GRPO training on RTX 4090 requires weight sync between rollout and training engines. Any overlapped communication during this sync is vulnerable to this pattern family. Specifically:
- verl's weight sync uses multiple CUDA streams
- SGLang/vLLM sleep/wake involves CuMem unmap/remap
- DeepSpeed overlap_comm is EXPLICITLY unsafe on single GPU (overlap_comm=False MANDATORY for RTX 4090)

## Cross-Framework Contribution Opportunity

**Tier 1 UNIQUE**: Comment on Megatron #5788 linking to DeepSpeed #8061 and vLLM #45552 as same pattern family. No existing comment makes this cross-framework connection. This establishes a systematic bug taxonomy for CUDA stream races in overlapped parameter gather.

The pattern family has 5 known members:
1. DeepSpeed #8061 — overlap_comm NaN (RESOLVED)
2. Megatron #5788 — StorageResize use-after-free (OPEN)
3. vLLM #45552 — CuMem sleep/wake crash (OPEN)
4. verl weight sync stream race (potential, not yet reported)
5. vLLM #46125 — weight reload stale cache (MERGED revert of #45093)

## Proposed Comment Draft (for user authorization)

Could post on Megatron #5788:
```
This is the same bug pattern family as DeepSpeed #8061 (overlap_comm NaN, now
resolved via #8080) and vLLM #45552 (CuMem sleep/wake crash).

All three share the same root cause: overlapped/async parameter gather where
storage is resized or released on one CUDA stream while another stream may
still be reading from it. The DeepSpeed fix (#8080) resolved this by waiting
on ALL producer streams per IPG bucket before consuming — analogous to
adding record_stream on the resized storage here.

For Megatron, the fix would be: add torch.cuda.current_stream().record_stream(
storage) before resizing, ensuring all in-flight kernels on the gather stream
complete before the storage is freed/resized.

This pattern family affects GRPO training on RTX 4090 and other single-GPU
setups where overlap_comm is used — see our analysis at
notebook/cuda-stream-race-pattern-family.md.
```
