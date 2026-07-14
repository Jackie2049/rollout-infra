# verl GRPO NaN Propagation Bug Analysis

## Bug: No NaN Protection in GRPO Advantage Computation

### Discovery
While analyzing verl's `core_algos.py` for contribution opportunities, I found that ALL 8+ advantage estimators have **zero NaN protection**. This contrasts with TRL's `GRPOTrainer`, which uses `torch.nan_to_num(advantages, nan=0.0)` after advantage computation.

### Impact
**NaN rewards → NaN advantages → NaN loss → NaN model parameters → training crash**

Common scenarios that produce NaN rewards:
1. Reward model produces NaN output (common with FP8 quantization)
2. Inf/NaN in token-level rewards from malformed log probs (vLLM #48585)
3. Division by zero in reward normalization when group std=0
4. Corrupted data from Ray serialization errors

### Evidence
- verl `core_algos.py`: No `nan_to_num`, `isnan`, or NaN check in ANY advantage estimator
- verl `ray_trainer.py`: No NaN check downstream in `compute_advantage()`
- verl `v1/trainer_base.py`: No NaN check in advantage or loss computation
- TRL `GRPOTrainer`: Has `torch.nan_to_num(advantages, nan=0.0)` as a safety guard

### Comparison: TRL vs verl NaN Handling

| Step | TRL | verl |
|------|-----|------|
| Reward computation | NaN can occur | NaN can occur |
| Advantage normalization | `nan_to_num(advantages, nan=0.0)` | **No protection** |
| Policy loss | Clipped ratio + dual-clip | Clipped ratio + dual-clip |
| NaN result | Advantages→0, training continues | **NaN propagates → crash** |

### Specific Code Locations

**`compute_grpo_outcome_advantage`** (core_algos.py:304):
```python
scores = token_level_rewards.sum(dim=-1)  # NaN reward → NaN score
# ... group mean/std computation → NaN mean/std
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # NaN/NaN → NaN
```

**`compute_grpo_vectorized_outcome_advantage`** (core_algos.py:350):
```python
scores = token_level_rewards.sum(dim=-1)  # NaN reward → NaN score
mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0)  # NaN propagates through scatter
scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)  # NaN/NaN → NaN
```

**All other estimators** (grpo_passk, reinforce++, reinforce++_baseline, etc.):
Same pattern — `scores = token_level_rewards.sum(dim=-1)` → no NaN guard.

### Proposed Fix

Add `nan_to_num` at two levels:

**Level 1: Per-estimator NaN guard** (in each advantage estimator):
```python
# After advantage computation:
advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
```

**Level 2: Centralized NaN guard** (in `compute_advantage()` at ray_trainer.py:282):
```python
# After ALL advantage estimators:
if torch.any(torch.isnan(data.batch["advantages"])):
    logger.warning("NaN detected in advantages, replacing with 0.0")
    data.batch["advantages"] = torch.nan_to_num(data.batch["advantages"], nan=0.0, posinf=0.0, neginf=0.0)
    data.batch["returns"] = torch.nan_to_num(data.batch["returns"], nan=0.0, posinf=0.0, neginf=0.0)
```

Level 2 is preferred because:
1. It's centralized — one change covers all estimators
2. It's visible — logs a warning so users know NaN occurred
3. It doesn't modify individual estimator code (less invasive)
4. It catches NaN from ANY estimator, not just GRPO

### Risk Assessment
- LOW risk: `nan_to_num` replaces NaN with 0.0, which means NaN samples get zero advantage (no learning signal for that sample)
- This is exactly what TRL does, and it's been validated in production
- NaN samples shouldn't drive learning anyway — their reward signal is corrupted
- The warning log helps users identify the root cause (reward model issue, data issue, etc.)

### Implementation Plan
1. Fork verl to Jackie2049/verl (already exists)
2. Create branch `fix/grpo-nan-to-num-advantage-guard`
3. Add centralized NaN guard in `compute_advantage()` at ray_trainer.py:282
4. Add test case: feed NaN rewards to GRPO advantage, verify advantages are 0.0 not NaN
5. Create PR on Jackie2049/verl fork (NOT upstream — per user instructions)

### Related Issues
- vLLM #48585: FP8 NaN/Inf logprob values (same root cause — NaN from quantization)
- verl #2911: loss≈0 early training (possibly related — if NaN advantages are silently becoming 0 without warning, user wouldn't know)
- TRL P7-2 top_n_sigma: Clips advantage outliers but doesn't handle NaN (torch.clamp(NaN) = NaN)
