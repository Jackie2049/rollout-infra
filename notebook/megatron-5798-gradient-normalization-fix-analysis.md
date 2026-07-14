# Megatron #5798: Cross-Framework Gradient Normalization Fix Analysis

## Overview

Megatron #5798 was submitted as a **PR** (not just an issue) on July 14, 2026 by **yuchenwang3** (+432/-3). It fixes the `seq_aux_loss` MBS-dependent scaling bug under `--calculate-per-token-loss`.

**Author pattern**: yuchenwang3 is the same author as #5395 (skip_grad_norm_clip), showing sustained engagement with Megatron optimizer/loss correctness.

## The Bug

In `_apply_seq_aux_loss()`, the routing_map reshaping to `[seq_length, bsz*num_experts]` means:
```python
local_num_tokens = routing_map.shape[0]  # = seq_length, ignores bsz!
```
This produces a spurious `1/MBS` scaling factor on the `seq_load_balancing_loss` gradient when `calculate_per_token_loss=True`.

## The Fix

Pass `valid_token_count = local_num_tokens * bsz` instead of just `local_num_tokens`. This restores the MBS-independent gradient scaling.

The PR adds a test: `test_seq_aux_loss_mbs_invariant_per_token_loss` — asserts cumulative router gradients are equal for MBS=1 vs MBS=N.

## Relationship to #4590

Our #4590 analysis (158% gradient bias with variable-length completions) documented the SAME architectural flaw:

| Flag | Bug | Affected Loss | Root Cause |
|------|-----|---------------|------------|
| `calculate_per_token_loss=False` | #4590 | Main policy loss | local_mean over microbatch → DP/CP average |
| `calculate_per_token_loss=True` | #5798 | seq_aux_loss | local_num_tokens ignores bsz dimension |

**Key insight**: The `calculate_per_token_loss` flag is architecturally dual:
1. **False**: main loss normalization uses local scope → gradient bias
2. **True**: main loss fixed, but auxiliary losses still use local scope → MBS-dependent scaling

This means no single flag toggle fixes all loss terms — the fix must be applied PER LOSS TERM.

## Cross-Framework Extension: verl #6836

verl #6836 (MERGED July 14) is the SAME pattern: MoE aux/z-loss grad blowup at CP>1 when `calculate_per_token_loss=True`. CP creates per-CP-chunk normalization, which is another form of local-scope normalization.

**Triple framework pattern**:
| Framework | Bug | Normalization Scope | Fix Status |
|-----------|-----|-------------------|------------|
| Megatron | #4590 | Local microbatch → DP average | OPEN, analysis documented |
| Megatron | #5798 | Local seq_length (ignores bsz) | PR OPEN by yuchenwang3 |
| verl | #6836 | Local CP chunk | MERGED |

## Contribution Strategy

**Comment on #5798** linking to #4590 and verl #6836:
- Shows both bugs as SAME architectural pattern
- Extends from main policy loss to ALL auxiliary losses
- Cross-framework connection (verl) adds weight

**Draft** (ready for user authorization):
```
This bug shares the same root cause as #4590 (calculate_per_token_loss gradient bias).

Both arise from normalization over a local scope (microbatch/chunk) instead of the
global training-step scope. #4590 documents the 158% gradient bias on the main policy
loss with variable-length completions when calculate_per_token_loss=False. Here, #5798
shows that even with calculate_per_token_loss=True, auxiliary losses (seq_aux_loss)
STILL normalize over a local scope, producing MBS-dependent gradient scaling.

This pattern also manifests in verl (#6836 — MoE aux/z-loss grad blowup at CP>1 with
calculate_per_token_loss), making it a cross-framework concern.

The unified fix should ensure ALL loss terms (main + auxiliary) use
global-sum/global-sum normalization regardless of MBS or CP configuration.
```

## RTX 4090 Relevance

For RTX 4090 (single GPU, no DP/CP):
- `calculate_per_token_loss=False` → no DP average → #4590 bug does NOT manifest
- `calculate_per_token_loss=True` → auxiliary losses still correct on single GPU
- But the **pattern matters**: any normalization that depends on batch/group size causes gradient bias with variable-length completions

## Monitoring

- #5798 has 0 human reviews, needs `/ok to test` from NVIDIA CI
- yuchenwang3 is likely busy with #5395 (JanEbert's 7-point refactoring request)
- Expected timeline: 2-4 weeks for review given author's current bandwidth
