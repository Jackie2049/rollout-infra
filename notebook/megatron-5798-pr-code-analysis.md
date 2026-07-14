# Megatron #5798 PR: Deep Code Analysis

## PR Summary
- **Author**: yuchenwang3 (same as #5395 skip_grad_norm_clip)
- **Files changed**: 2 files: +76/-1 total (+3/-1 actual code, +73 tests)
- **State**: OPEN draft (auto-converted), waiting for `/ok to test` from NVIDIA CI
- **0 human reviews** as of July 14

## The Fix (router.py: +3/-1)

```python
# BUGGY (line ~before fix):
# local_num_tokens = routing_map.shape[0]  # = seq_length, ignores bsz!
# The reshape to [seq_len, bsz*num_experts] collapses batch into expert dim

# FIXED:
valid_token_count = local_num_tokens * bsz
```

The key insight: `routing_map` is reshaped to `[seq_length, bsz * num_experts]`, so `shape[0]` = `seq_length` regardless of MBS. Passing `local_num_tokens * bsz` recovers the total micro-batch tokens.

## Why This Is Correct

The `valid_token_count` parameter pre-multiplies the aux gradient to cancel the global `1/total_tokens` division in `finalize_model_grads()`:
- `total_tokens = GBS * seq_length` (constant across MBS)
- Without fix: `valid_token_count = seq_length` → gradient scaled by `seq_length / (GBS*seq_length) = 1/GBS` = **missing MBS factor**
- With fix: `valid_token_count = seq_length * bsz` → gradient scaled by `(seq_length*bsz) / (GBS*seq_length) = bsz/GBS = 1/DP_size` = **correct**

With padding (`calculate_per_token_loss=True`), `local_num_tokens` is the mean valid tokens per sequence, so `* bsz` recovers the total valid tokens in the micro-batch — still correct.

## Test Validation

The test `test_seq_aux_loss_mbs_invariant_per_token_loss`:
1. Runs with MBS=1, accumulates router gradients
2. Runs with MBS=N, accumulates router gradients
3. **Asserts they are equal** (fails pre-fix, passes post-fix)
4. Tests both padded and unpadded cases

## Pattern Validation

Our cross-framework analysis was correct:
- #5798 confirms the `incorrect_gradient_normalization_scope` pattern
- The fix is localized (+3/-1) because it's a single normalization factor error
- Same pattern as #4590 (main loss 158% bias) and verl #6836 (CP-chunk normalization)

## Monitoring Path

- PR needs NVIDIA CI trigger (`/ok to test`)
- Needs human reviewer (auto-assigned by CODEOWNERS)
- After review, tests must pass (likely, since tests were written for this fix)
- Timeline: 2-4 weeks given author's bandwidth on #5395 refactoring
