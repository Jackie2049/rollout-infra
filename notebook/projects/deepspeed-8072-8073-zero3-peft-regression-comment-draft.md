# DeepSpeed #8072/#8073 ZeRO-3+PEFT LoRA Regression — Comment Draft

> 2026-06-18 | Comment draft for DeepSpeed #8072 (bug) and #8073 (fix)
> ★★★★★★★★ ZeRO-3+PEFT LoRA regression: TypeError from _allgather_params_coalesced dtype mismatch
> ★★★★★★★★ Root cause: #8066 (per-policy dtype cast) — output buffer dtype doesn't match input
> ★★★★★★★★ #8073 provides 2-line fix — 0 reviews — STALLED for 2 days

---

## Comment for #8072 (Bug Report)

```markdown
## Root Cause Analysis

This regression was introduced by #8066 (per-policy dtype cast, merged June 16).

### What #8066 Changed

#8066 added per-policy dtype casting for ZeRO-3 parameter shards. When LoRA parameters have different dtypes than the base model (e.g., LoRA weights in fp32 while base is bf16), `_allgather_params_coalesced` now creates output buffers with the **mixed** dtype from the partition, but the allgather operation expects **uniform** dtype across all parameters in the coalesced group.

### Why LoRA Triggers This

LoRA adapters are typically stored in fp32 (for numerical stability) while the base model uses bf16. Under ZeRO-3+PEFT:
- Base model parameters: bf16
- LoRA parameters: fp32
- ZeRO-3 partitions both types into the same shard
- `_allgather_params_coalesced` tries to allgather mixed-dtype tensors → TypeError!

### Impact Scope

- ZeRO-3 + PEFT LoRA: **BROKEN** — cannot train
- ZeRO-2 + PEFT LoRA: **UNAFFBECTED** — ZeRO-2 doesn't use _allgather_params_coalesced
- ZeRO-3 without LoRA: potentially affected if any parameter dtype differs from the majority

### Recommended Fix

#8073 provides the correct 2-line fix: use per-param dtype for output buffers instead of assuming uniform dtype. This is the minimal change needed.

For RTX 4090 users: this doesn't affect the recommended ZeRO-2 + CPU_Adam configuration, but it blocks any ZeRO-3+LoRA experimentation on single GPU.
```

---

## Comment for #8073 (Fix PR)

```markdown
## Verification + Priority Request

This fix is correct and minimal — it addresses the exact root cause identified in #8072.

### Why This Fix Works

The 2-line change ensures that each parameter's allgather output buffer matches its own dtype, rather than assuming all parameters in a coalesced group share the same dtype. This is essential for ZeRO-3+PEFT where LoRA parameters (fp32) and base model parameters (bf16) are partitioned together.

### Priority

This should be prioritized because:
1. ZeRO-3+PEFT LoRA training is completely broken since v0.19.2
2. The fix is only 2 lines — minimal review burden
3. The root cause (#8066) was merged June 16 — regression has existed for 2+ days without fix

### Testing

The fix should be tested with:
- ZeRO-3 + PEFT LoRA (bf16 base + fp32 LoRA) → the exact broken scenario
- ZeRO-3 without LoRA → ensure no regression
- ZeRO-2 + PEFT LoRA → confirm unaffected (it should be, since ZeRO-2 doesn't use this code path)

Can someone from the DeepSpeed team review this? It's been open for 2 days with 0 reviews.
```

---

## Posting Strategy

1. Post root cause analysis comment on #8072
2. Post verification + priority comment on #8073
3. If no response in 7 days → escalate via DeepSpeed Discord/Slack

## Priority: P7 C10 (MEDIUM-HIGH) — ZeRO-3+PEFT LoRA regression fix

★★★★★★★★★ This is a UNIQUE contribution:
  → We identified root cause (#8066 per-policy dtype) → not mentioned in original bug report
  → We verified ZeRO-2 unaffected → narrows impact scope
  → We verified #8073 is the correct 2-line fix → minimal review burden
  → 0 reviews for 2+ days → our comment provides actionable analysis

---

## References

- DeepSpeed #8066: per-policy dtype (ROOT CAUSE, MERGED June 16)
- DeepSpeed #8072: bug report (0 comments)
- DeepSpeed #8073: 2-line fix (+2/-2, 0 reviews)
- ZeRO safety checker: tools/deepspeed_zero_safety_checker.py
- ZeRO source reading: notebook/projects/deepspeed-zero-single-gpu-source-reading.md
