# UP-GRPO: Unbounded Positive Asymmetric Policy Optimization

**Paper**: arXiv:2607.06987 (Chongyu Fan, ByteDance Seed & Michigan State University)
**verl PR**: #7022 (`compute_policy_loss_up` in `verl/trainer/ppo/core_algos.py`)

## Core Insight: Probability Capacity (Cap)

The improvement a token can contribute is bounded by how much probability mass it can still gain.
Low-probability **correct** tokens have the most capacity, yet standard GRPO clipping caps their update hardest.

This is the **exploration-stability dilemma**:
- Pure importance sampling (IS) → gradient explosion (unstable)
- Clipping to prevent instability → structurally stifles exploration (suboptimal)

## Asymmetric Design (Eq. 15)

### For A > 0 (positive advantages): Unbounded REINFORCE

```
J+ = A * log pi_theta(o | q, o_<t})
```

Implemented via **stop-gradient self-anchored ratio**:
```python
r_tilde = pi_theta / sg(pi_theta)   # forward: r_tilde = 1
                                    # backward: grad = A * grad(log pi_theta)
```

- `r_tilde = torch.exp(log_prob - log_prob.detach())`
- Forward pass: ratio = 1 trivially
- Backward pass: `sg(pi_theta)` is constant → gradient is pure REINFORCE
- No pi_old anchor, no upper clip, no IS explosion

### For A ≤ 0 (negative advantages): Standard GRPO clip

```
min(r * A, clip(r, 1-eps, 1+eps) * A)
```

With dual-clip lower bound `clip_ratio_c > 1.0` for stability.
**Identical** to standard GRPO negative branch.

## Why Asymmetric?

Applying unbounded updates to BOTH advantage signs collapses training within ~25 steps.
Negative advantages with unbounded updates → aggressive gradient ascent → destroys original representation.

## Comparison Table

| Concept | Standard GRPO | UP-GRPO |
|---------|--------------|---------|
| Positive A | `min(r*A, (1+eps)*A)` — gradient vanishes outside trust region | `A * log pi_theta` — no clip, pure REINFORCE |
| Negative A | `min(r*A, clip(r, 1-eps, 1+eps)*A)` | Same as GRPO |
| IS ratio denominator | `pi_old` — can explode for rare tokens | `sg(pi_theta)` — forward=1, backward=constant |
| Probability Cap for A>0 | `Cap = min(1, (1+eps)*pi_old) - pi_theta` — bounded by pi_old | `Cap = 1 - pi_theta` — unbounded by history |

## Relationship to Our Patches

### P9-1 bypass_mode — Structural Equivalent (half of UP-GRPO)

When `bypass_mode=True` with `num_iterations=1`:
- `old_logprob = per_token_logprob.detach()` → ratio = `exp(log_prob - log_prob.detach())`
- This is **exactly** the self-anchored ratio `r_tilde`
- But current GRPO *still clips* both sides symmetrically

**UP-GRPO = bypass_mode + remove-positive-clip**

P9-1 provides the self-anchor mechanism; UP-GRPO adds the crucial second half: asymmetric removal of positive clip.

### P7-2 top_n_sigma — Orthogonal and Complementary

- top_n_sigma clips **advantage values** (input to loss function)
- UP-GRPO changes **loss function structure** (how advantages drive policy gradient)
- They address different failure modes:
  - UP-GRPO: "symmetric clip suppresses exploration for rare correct tokens"
  - top_n_sigma: "noisy reward outliers produce destabilizing advantage values"

**Recommended combo**: UP-GRPO + top_n_sigma=3.0 + bypass_mode=True

top_n_sigma=3.0 only clips genuine outliers (>3 std), leaving the vast majority of positive advantages unmodified — these are the ones driving exploration. The rare extreme outliers that get clipped are likely noise.

### Potential Enhancement: Asymmetric top_n_sigma

For purest UP-GRPO behavior, could apply top_n_sigma only to negative advantages, or use larger sigma for positive (e.g., `top_n_sigma_pos=5.0, top_n_sigma_neg=3.0`). This would preserve maximum exploration freedom for positive advantages while still protecting against extreme outliers.

## Implementation in verl (#7022)

```python
# Positive branch: self-anchored REINFORCE
r_tilde = torch.exp(log_prob - log_prob.detach())
pg_losses_pos = -advantages * r_tilde

# Negative branch: standard GRPO symmetric clip + dual-clip
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses_neg = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

pg_losses = torch.where(advantages > 0, pg_losses_pos, pg_losses_neg)
```

## Key Takeaways for Our Work

1. **bypass_mode is not just a memory hack** — it's structurally equivalent to UP-GRPO's self-anchor. Our P9-1 has deeper significance than we initially realized.
2. **Asymmetric clipping is the future** — symmetric clip for both advantage signs is suboptimal. UP-GRPO validates this with theory + experiments.
3. **Our patch stack (P7-2 + P9-1) is compatible with UP-GRPO** — and the combo would be even stronger.
4. **Consider adding UP-GRPO to TRL** — as a new loss_type option, not replacing standard GRPO.
