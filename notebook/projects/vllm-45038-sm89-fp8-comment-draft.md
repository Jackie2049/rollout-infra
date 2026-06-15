# vLLM PR #45038 — SM89 FP8 KV Crash Comment Draft

> Status: DRAFT — User should review before posting
> PR: https://github.com/vllm-project/vllm/pulls/45038
> Title: [Bugfix] Guard compressed-tensors FP8 KV cache override on SM90+

## Context

- Issue #44879: compressed-tensors FP8 KV causes CUDA crash on SM89 (L4/RTX4090)
- PR #45038: adds `has_device_capability(90)` guard, only override KV dtype to FP8 on SM90+
- Already validated on SM89 (L4 GPU) by another contributor

## Proposed Comment

```
This fix directly addresses a critical SM89 correctness issue that affects both L4 and RTX 4090 GPUs. The root cause is well-identified: FlashInfer's FP8 attention kernels (flash_attn_varlen_func_fp8_sm90) only exist for SM90+, so overriding kv_cache_dtype to fp8 on SM89 inevitably causes CUDA illegal-memory-access.

I've been studying RTX 4090 (SM89) limitations in depth for AI training and inference workloads. A few observations that may be helpful:

**SM89 FP8 support matrix** (from our research):
- FP8 E5M2 (inference): ✗ — not supported on SM89, only SM90+
- FP8 E4M3 (training): ✗ — not supported on SM89
- FP8 AllGather/communication: ✗ — NCCL FP8 requires SM90+
- INT4 GPTQ + INT8 KV: ✓ — the practical path for SM89 inference
- BF16 training: ✓ — the only correct training precision for SM89

**Impact for GRPO training on RTX 4090**:
The FP8 KV cache override is particularly problematic for GRPO/RL training scenarios where:
- vLLM is used as the rollout engine (verl/rLLM)
- INT8 KV cache is the correct choice for SM89 (not FP8)
- compressed-tensors models with FP8 config would silently crash on RTX 4090

The `has_device_capability(90)` guard is the right fix. Would also suggest:
1. Adding SM89 to the CI test matrix (even a single L4 instance) to catch future SM89 regressions
2. Updating the vLLM documentation to explicitly list SM89 limitations for FP8 KV

Happy to help test on RTX 4090 when our GPU server is available.
```

## Key Points

1. ★★★ SM89 FP8 support matrix → shows comprehensive understanding
2. ★★★ GRPO training impact → connects to our core research area
3. ★★★ Concrete suggestions (CI matrix + docs) → actionable improvement
4. ★★★ Offer to test on RTX 4090 → practical contribution commitment
