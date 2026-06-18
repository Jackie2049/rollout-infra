# DeepSpeed #8073 Tier 1 Comment Draft — ZeRO-3+PEFT LoRA Regression Fix

> 2026-06-18 | Tier 1 comment opportunity | 0 reviews on critical regression fix
> ★★★★★★★★ 1-line dtype fix addresses ZeRO-3+LoRA regression in v0.19.2
> ★★★★★★★★ Issue #8072 confirmed → but fix PR #8073 has ZERO comments → stalled!

---

## Comment Draft

This is a critical regression fix that affects anyone using ZeRO-3 with PEFT/LoRA in DeepSpeed v0.19.2. A few observations:

### Root cause clarity

The bug is precisely identified: `_allgather_params_coalesced` allocates all output buffers using `param_list[0].ds_tensor.dtype` instead of the per-parameter dtype. This causes a TypeError when LoRA adapters (which may be fp32) are mixed with base weights (bf16) in the same parameter group. The 2-line fix (enumerate + per-param dtype) directly addresses this.

### Impact assessment for LoRA training

ZeRO-3 + PEFT/LoRA is a common configuration for memory-constrained training (single GPU or small clusters). This regression makes it completely unusable in v0.19.2 — any LoRA adapter with different dtype from base weights triggers the crash. This is particularly impactful for:
- RTX 4090 users trying to fit large models with ZeRO-3 + LoRA (though ZeRO-2 + CPU_Adam remains the safer path)
- Anyone using mixed-precision LoRA (fp32 adapters on bf16 base model)

### ZeRO-2 unaffected but ZeRO-3 adoption blocked

ZeRO-2 + CPU_Adam works correctly for single GPU training, so this regression doesn't block the most common RTX 4090 path. However, ZeRO-3 is needed for larger models (30B+) and for multi-GPU scenarios where ZeRO-3's partitioning provides better memory efficiency. This regression blocks ZeRO-3 adoption for LoRA users entirely.

### Suggestion: also add a regression test

The fix is minimal and correct. Adding a regression test that creates a ZeRO-3 model with PEFT LoRA and verifies `_allgather_params_coalesced` produces correctly-typed output buffers would prevent similar regressions in the future. A simple test with mixed-dtype parameters (bf16 base + fp32 LoRA) would catch this.

---

## References

- DeepSpeed #8072: https://github.com/microsoft/DeepSpeed/issues/8072 (regression report)
- DeepSpeed #8073: https://github.com/microsoft/DeepSpeed/pull/8073 (1-line fix)
- DeepSpeed #8075: https://github.com/microsoft/DeepSpeed/pull/8075 (alternative ZeRO-3+PEFT fix, 2-line)
- RTX 4090 ZeRO analysis: tools/deepspeed_zero_safety_checker.py
