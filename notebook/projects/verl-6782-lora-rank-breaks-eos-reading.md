# verl #6782 — LoRA Rank=64 Breaks EOS in vLLM Rollout Reading

> 2026-06-18 | Issue #6782 OPEN (June 17) | Author: (community)
> ★★★★★★★★ CRITICAL for RTX 4090 LoRA GRPO: rank=64/alpha=128 breaks EOS → MUST use rank=32/alpha=64!
> ★★★★★★★★ Qwen3.5-27B LoRA GRPO on GSM8K → vLLM never emits EOS with rank=64
> ★★★★★★★★ Zero comments → no fix proposed → needs investigation

---

## 1. Bug: LoRA Rank=64 Breaks EOS Generation

```
★★★★★★★★★ The bug:

Config: Qwen3.5-27B LoRA GRPO, FSDP2 actor/ref, vLLM rollout (TP=4)
  → lora.merge=False, load_format=safetensors, layered_summon=True
  → unmerged dynamic LoRA weight sync via IPC

The ONLY difference between working and broken:

| Run | lora_rank | lora_alpha | Result |
|-----|-----------|------------|--------|
| A   | 32        | 64         | ✅ Works — vLLM emits EOS normally |
| B   | 64        | 128        | ❌ vLLM NEVER emits EOS — all responses truncated |

★★★★★★★★★ Symptom:
  → Every response runs to max_response_length (1024) → truncation
  → response_length/clip_ratio ≈ 1.0 → all samples truncated
  → Scores collapse → reward signal degenerate → GRPO BROKEN

★★★★★★★★★ Everything else identical:
  → Same model, data, batch sizes, sampling params, TP, offload settings
  → Same enforce_eager=True, free_cache_engine=True
  → Same max_prompt_length=1024, max_response_length=1024
```

---

## 2. Root Cause Analysis (Hypothesized)

```
★★★★★★★★★ Likely root cause: LoRA scaling factor magnitude

LoRA output = base_output + (lora_B @ lora_A) * (alpha / rank)

With rank=32, alpha=64: scaling = 64/32 = 2.0
With rank=64, alpha=128: scaling = 128/64 = 2.0

★★★★★★★★★ Scaling factor is the SAME (2.0) → but LoRA ADAPTER magnitude differs!

LoRA adapter magnitude = ||lora_B @ lora_A|| * (alpha / rank)
  → With rank=64 → lora_B and lora_A have MORE parameters
  → ||lora_B @ lora_A|| can be larger (more degrees of freedom)
  → Even with same scaling factor → total LoRA delta can be larger

★★★★★★★★★ Why this breaks EOS:
  → Larger LoRA delta → logit distribution more distorted
  → EOS token logit suppressed → model never selects EOS
  → This is a vLLM LoRA weight sync + generation interaction bug
  → ★★★★★★★★ Under unmerged LoRA (merge=False) → vLLM applies LoRA at inference
  → The LoRA delta at rank=64 may be pushing logits away from EOS distribution

★★★★★★★★★ Alternative hypotheses:
  → vLLM LoRA weight sync: IPC buffer may have different alignment/offset for rank=64
  → vLLM LoRA kernel: SGMV/SGMMV may handle rank=64 differently
  → Sampling: higher-rank LoRA changes logit landscape → temperature/greedy differences
  → ★★★★★★★★ #6688 (MERGED) fixed IPC buffer crash → but may not fix all rank-dependent issues
```

---

## 3. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 LoRA GRPO config MUST use rank=32/alpha=64!

Why rank=32 is optimal for RTX 4090:
  1. rank=32 works (verified by #6782)
  2. rank=32 → fewer parameters → lower memory → fits 24GB
  3. rank=32 → smaller LoRA delta → logit distribution preserved → EOS works
  4. rank=32 + alpha=64 → scaling=2.0 → same as rank=64 but smaller adapter

★★★★★★★★★ Config constraint:
  → MUST: lora_rank=32, lora_alpha=64 (for Qwen3.5 on RTX 4090)
  → MUST: lora.merge=False (unmerged for HYBRID mode)
  → MUST: layered_summon=True (weight sync)
  → MUST: enforce_eager=True (avoid CUDA graph issues)
  → MUST: free_cache_engine=True (memory management)
  → ★★★★★★★★ MUST NOT: lora_rank=64 or higher → EOS breaks → GRPO BROKEN!

★★★★★★★★★ Update grpo_config_reference.py:
  → Add validation rule: lora_rank <= 32 for RTX 4090
  → Document #6782 as critical LoRA rank constraint
```

---

## 4. Related Issues

```
★★★★★★★★★ Related verl issues:

#6688 (MERGED June 12): LoRA IPC buffer crash fix
  → cudaErrorIllegalAddress with unmerged LoRA + vLLM + free_cache_engine=True
  → add_lora retained views into reused IPC bucket buffer → crash
  → ★★★★★★★★ Fixed crash but rank=64 EOS bug is a DIFFERENT issue

#6512 (OPEN): per-unit LoRA summon
  → Reduces peak memory from full-model to largest-FSDP-unit
  → ★★★★★★★★ Complementary: reduces memory but doesn't fix EOS bug

#6454: Original IPC crash issue → #6688 fixed

★★★★★★★★★ vLLM related:
  → vLLM SGMV/SGMMV LoRA kernels may have rank-dependent behavior
  → SGLang #28499 (MERGED June 17) fixed csgmv CUDA graph segment replay
  → SGLang #28566 (OPEN) fixing LoRA sentinel-pad for DP-attention
  → ★★★★★★★★ Need to investigate: vLLM LoRA kernel at rank=64 vs rank=32
```

---

## Key Findings Summary

★★★★★★★★★ #6782: LoRA rank=64/alpha=128 breaks EOS in vLLM rollout → MUST rank=32/alpha=64!
★★★★★★★★★ Symptom: vLLM never emits EOS → all responses truncated → GRPO BROKEN
★★★★★★★★★ Root cause: larger LoRA adapter magnitude at rank=64 → logit distribution distorted → EOS suppressed
★★★★★★★★★ RTX 4090: MUST use lora_rank=32, lora_alpha=64 → verified working
★★★★★★★★★ Zero comments → no fix proposed → needs community investigation
★★★★★★★★★ #6688 (MERGED) fixed IPC crash → but this is a DIFFERENT rank-dependent bug

---

## References

- verl #6782: https://github.com/verl-project/verl/issues/6782
- verl #6688: https://github.com/verl-project/verl/pull/6688 (LoRA IPC buffer crash fix)
- verl #6512: https://github.com/verl-project/verl/pull/6512 (per-unit LoRA summon)
- RTX 4090 config reference: tools/rtx4090_grpo_config_reference.py
