# UP-GRPO Implementation in TRL — Technical Details

**Date**: 2026-07-14
**PR**: https://github.com/Jackie2049/trl/pull/6
**Algorithm**: UP-GRPO (Unbounded Positive asymmetric policy optimization, arXiv:2607.06987)

## Algorithm Overview

UP-GRPO addresses the **exploration-stability dilemma** in GRPO training:
- Standard GRPO clips both positive and negative advantage updates, limiting exploration
- UP-GRPO removes the upper clip for positive advantages while maintaining stability for negatives

### Core Formula

For advantage Â and importance ratio r = π_θ/π_old:

**Â > 0 (positive advantages)**:
```
r̃ = exp(logπ_θ - sg(logπ_θ))   # self-anchored ratio
loss = -r̃ * Â                   # no clip → pure REINFORCE gradient
```

**Â ≤ 0 (non-positive advantages)**:
```
loss = -min(r * Â, clamp(r, 1-ε_low, 1+ε_high) * Â)   # standard GRPO clip
```

### Self-Anchored Ratio Properties

The self-anchored ratio `r̃ = exp(logπ - sg(logπ))` has unique gradient properties:
- Forward pass: r̃ ≈ 1.0 (since logπ ≈ sg(logπ) when not diverged)
- Backward pass: gradient flows only through numerator (logπ), denominator is detached
- This gives pure REINFORCE gradient: ∂loss/∂π = -Â / sg(π)
- No importance sampling bias, no clipping distortion for positive advantages

## Implementation Details

### Files Modified

1. **`grpo_config.py`** (lines ~791-825):
   - Added `'up'` to loss_type help string
   - Added UP-GRPO description with paper reference
   - No new config fields needed — uses existing `epsilon`/`epsilon_high`

2. **`grpo_trainer.py`** (4 locations):
   - **Loss computation** (~line 2966): `elif self.loss_type == "up"` branch
   - **Normalization** (~line 3020): Grouped with `["cispo", "dapo", "vespo", "up"]`
   - **Entropy normalization** (~line 3045): Same grouping
   - **Metrics** (~line 3140): `clip_ratio/neg_low_mean` + `up/self_anchored_ratio_mean`

### Key Implementation Code

```python
elif self.loss_type == "up":
    self_anchored_ratio = torch.exp(per_token_logps - per_token_logps.detach())
    clipped_ratio = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    per_token_loss = -torch.where(
        advantages > 0,
        self_anchored_ratio * advantages,      # Â>0: no clip, exploration-friendly
        torch.min(coef_1 * advantages, clipped_ratio * advantages),  # Â≤0: standard clip
    )
```

### Normalization Choice

UP-GRPO uses DAPO-style normalization (global token count):
```python
elif self.loss_type in ["cispo", "dapo", "vespo", "up"]:
    normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
    loss = (per_token_loss * mask).sum() / normalizer
    policy_loss = loss.detach()
```

This is appropriate because:
- UP-GRPO encourages exploration (like DAPO), so length bias elimination is important
- DAPO normalization counts active tokens globally across the accumulated batch
- This prevents the model from preferring shorter completions with positive advantages

### Metrics

- `clip_ratio/neg_low_mean`: Fraction of Â≤0 tokens where coef_1 < 1-ε_low (lower clip activated)
- `up/self_anchored_ratio_mean`: Mean self-anchored ratio for Â>0 tokens (should be ~1.0 at start, drifts as policy diverges)

## Relationship to Other Patches

| Patch | What it does | Relationship to UP-GRPO |
|-------|-------------|------------------------|
| P9-1 `bypass_mode` | Replace pi_old with pi_theta.detach() | **Structurally half** of UP-GRPO's self-anchor |
| P7-2 `top_n_sigma` | Clip top-N rewards by sigma | Compatible; reduces outlier advantage magnitude |
| verl `bypass_mode` | Same as P9-1 in verl | Same relationship; verl PR already exists |

**Key insight**: Our P9-1 `bypass_mode` replaces the old policy ratio with the self-anchored ratio, but keeps standard clipping. UP-GRPO goes further by also **removing the positive clip**. The combination `bypass_mode=True + loss_type="up"` would be the strongest version.

## Test Results

5/5 standalone tests passed:
1. `test_self_anchored_ratio_gradient`: Gradient flows through numerator only
2. `test_positive_advantages_unbounded`: No clip on Â>0 tokens
3. `test_negative_advantages_clipped`: Standard GRPO clip for Â≤0
4. `test_asymmetric_behavior`: UP-GRPO loss > GRPO loss for Â>0 tokens
5. `test_zero_advantage`: A=0 tokens produce zero loss

## Usage Example

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    loss_type="up",          # UP-GRPO
    epsilon=0.2,             # lower clip for Â≤0
    epsilon_high=0.28,       # upper clip for Â≤0 (DAPO recommended)
    beta=0.001,              # KL penalty (optional)
    num_iterations=1,        # single iteration per batch
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)
```
