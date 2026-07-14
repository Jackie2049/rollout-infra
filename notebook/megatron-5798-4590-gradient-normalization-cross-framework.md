# Megatron #5798 × #4590: calculate_per_token_loss Gradient Scaling Bug Cross-Framework Analysis

## New Bug: Megatron #5798 (July 14, 2026)

**Title**: Fix for sequence-level aux MoE loss being dependent on batch size
**Author**: OlegSudakov
**Root cause**: Under `--calculate-per-token-loss`, `_apply_seq_aux_loss` reshapes `routing_map` to `[seq_length, bsz*num_experts]`, so `local_num_tokens = routing_map.shape[0] = seq_length` — independent of bsz/MBS. This produces a spurious `1/MBS` scaling factor on the `seq_load_balancing_loss` gradient.

## Our Analysis: Megatron #4590

We documented the SAME root cause in #4590:
- `calculate_per_token_loss=False` (default) divides by local token count per microbatch, then averages across DP/CP — producing **158% gradient bias** with imbalanced token counts
- `calculate_per_token_loss=True` (correct) uses global-sum numerator / global-sum denominator, divides once at finalize time

## Cross-Framework Pattern

Both bugs share the same pattern: **per-token loss normalization that depends on MBS/batch-size instead of using a global denominator**.

| Bug | Framework | Symptom | Root Cause |
|-----|-----------|---------|------------|
| #4590 | Megatron | 158% gradient bias with variable-length completions | local_mean over microbatch tokens instead of global_sum/global_sum |
| #5798 | Megatron | 1/MBS spurious scaling on seq_aux_loss gradient | `local_num_tokens = seq_length` (ignores bsz dimension) |
| #6836 | verl | MoE aux/z-loss grad blowup at CP>1 | calculate_per_token_loss with CP creates per-CP-chunk normalization |
| P7-2 | TRL | top_n_sigma clipping needs group-level normalization | GRPO advantage per-group, not per-token |

**Pattern family**: `incorrect_gradient_normalization_scope` — normalization over a local/per-microbatch scope instead of global/training-step scope.

## Key Insight

The `calculate_per_token_loss` flag is a DUAL bug:
1. **When False**: local token-count normalization → gradient bias (our #4590 finding)
2. **When True**: correct for main loss, but auxiliary losses (MoE z-loss, seq_aux_loss) have separate normalization paths that STILL use local scope → MBS-dependent gradients (#5798 finding)

This means #4590 and #5798 are NOT independent bugs — they're the same architectural flaw manifesting in different loss terms.

## Contribution Opportunity

**Tier 1 UNIQUE**: Cross-framework comment linking #4590 ↔ #5798, showing both share `incorrect_gradient_normalization_scope` pattern family. No existing comment makes this connection. This extends the analysis from just the main policy loss to ALL auxiliary losses under `calculate_per_token_loss`.

Additionally, verl #6836 (MoE aux/z-loss grad blowup at CP>1) is the SAME pattern in a different framework, making this a **3-framework cross-reference**.

## Proposed Comment Draft (for user authorization)

Could post on Megatron #5798:
```
This bug shares the same root cause as #4590 (calculate_per_token_loss gradient bias).

Both arise from the same architectural pattern: normalization over a local scope
(microbatch/chunk) instead of the global training-step scope. #4590 documents the
158% gradient bias on the main policy loss with variable-length completions when
calculate_per_token_loss=False. Here, #5798 shows that even with
calculate_per_token_loss=True, auxiliary losses (seq_aux_loss) STILL normalize
over a local scope, producing MBS-dependent gradient scaling.

This pattern also manifests in verl (#6836 — MoE aux/z-loss grad blowup at CP>1
with calculate_per_token_loss), making it a cross-framework concern.

The unified fix should ensure ALL loss terms (main + auxiliary) use
global-sum/global-sum normalization regardless of MBS or CP configuration.
```
