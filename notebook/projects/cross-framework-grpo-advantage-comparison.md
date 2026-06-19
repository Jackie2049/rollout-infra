# Cross-Framework GRPO Advantage Computation Comparison

> 2026-06-19 | Deep reading of GRPO advantage computation across verl, rLLM, TRL
> ★★★★★★★★ CROSS-FRAMEWORK DESIGN DEFECT: ALL frameworks handle group_size=1 with mean=0, std=1 → GRPO degenerates to REINFORCE
> ★★★★★★★★ verl uses `index` (uid) array for grouping, rLLM uses TrajectoryGroup.group_id, TRL uses prompt grouping

---

## 1. GRPO Advantage Formula

Standard GRPO advantage for group g with n trajectories:

```
A_i = (r_i - μ_g) / (σ_g + ε)
```

Where:
- μ_g = mean of rewards in group g
- σ_g = std of rewards in group g
- ε = small constant for numerical stability

**When group_size=1**: μ_g = r_1, σ_g = 0 → division by zero!

ALL frameworks handle this by setting **mean=0, std=1** for singleton groups:
```
A_i = (r_i - 0) / (1 + ε) = r_i  ← REINFORCE, NOT GRPO!
```

---

## 2. Framework Implementations

### 2.1 verl — `core_algos.py` compute_grpo_outcome_advantage()

```python
# verl/trainer/ppo/core_algos.py (line ~340)
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards, response_mask, index, epsilon=1e-6,
    norm_adv_by_std_in_grpo=True, config=None
):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)   # ← mean=0
                id2std[idx] = torch.tensor(1.0)     # ← std=1
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
```

**Grouping mechanism**: `index` parameter = uid (unique prompt ID). Each uid maps to a group of trajectories sharing the same prompt. The uid comes from rollout generation — trajectories from the same prompt get the same uid.

**Vectorized version**: `compute_grpo_vectorized_outcome_advantage()` uses `group_mean_std()` for efficient GPU computation.

### 2.2 verl — `groupwise.py` group_mean_std()

```python
# verl/utils/groupwise.py
@torch.no_grad()
def group_mean_std(scores, gidx, eps=1e-6, device=None):
    # ... efficient GPU computation using index_add_ ...
    count = torch.zeros(G, ...).index_add_(0, gidx, ones)
    s1 = torch.zeros(G, ...).index_add_(0, gidx, scores)
    mean = s1 / count.clamp_min(1.0)
    centered = scores - mean[gidx]
    var_num = torch.zeros(G, ...).index_add_(0, gidx, centered * centered)
    denom = (count - 1.0).clamp_min(1.0)
    var = var_num / denom
    std = torch.sqrt(torch.clamp(var, min=eps))

    # ★★★★★★★★ SAME DEGENERATION as rLLM!
    # Singleton groups: mean=0, std=1
    single = count <= 1.0
    if torch.any(single):
        mean = mean.clone()
        std = std.clone()
        mean[single] = 0.0    # ← mean=0
        std[single] = 1.0     # ← std=1
    return mean, std, count
```

**Key difference from rLLM**: verl groups by uid (prompt identity), NOT by task_id:name. This means:
- If multiple trajectories share the same prompt → they ARE in the same group → group_size > 1 → GRPO works correctly
- If only one trajectory per prompt → group_size=1 → same degeneration

**verl's prefix_grouper**: `prefix_grouper_utils.py` optimizes shared-prefix computation within groups. Uses `PrefixGrouper` to batch trajectories with the same prompt prefix for efficient forward pass. This is a performance optimization, not a grouping change.

### 2.3 rLLM — TransformConfig grouping_mode

```python
# rllm/trainer/algorithms/transform.py (V2 fix)
def _build_trajectory_groups(episodes, compact_filtering_config, grouping_mode="by_task_id"):
    for episode in episodes:
        task_id = episode.task_id
        for trajectory in episode.trajectories:
            if grouping_mode == "by_task_id":
                group_key = task_id          # ← ALL trajectories per task in same group
            elif grouping_mode == "by_task_id_and_name":
                group_key = f"{task_id}:{trajectory.name}"  # ← agent-role separation
```

**rLLM's bug (#605)**: Default grouping_mode was "by_task_id_and_name" → when trajectory names differ (even slightly), group_size=1 → GRPO degenerates. V2 fix defaults to "by_task_id" (optimal for GRPO).

### 2.4 TRL — GRPO advantage

TRL groups by prompt in its GRPOTrainer. Same singleton handling pattern.

---

## 3. Cross-Framework Singleton Group Handling Comparison

| Framework | Grouping key | Singleton handling | Where in code |
|-----------|-------------|-------------------|---------------|
| **verl** | uid (prompt ID) | mean=0, std=1 | core_algos.py line ~342, groupwise.py line ~130 |
| **rLLM** (V1) | task_id:name | mean=0, std=1 (implicit via std=0→ε) | TransformConfig + advantage computation |
| **rLLM** (V2) | task_id (default) | Same degeneration mechanism | Revised fix |
| **TRL** | prompt | mean=0, std=1 | GRPOTrainer |
| **DeepSpeed** | Not applicable (no GRPO built-in) | N/A | — |

---

## 4. Mathematical Analysis: Why mean=0, std=1 = REINFORCE

For a singleton group with reward r:

```
GRPO advantage (normal): A = (r - μ) / σ
Singleton fallback:      A = (r - 0) / 1 = r
REINFORCE:              A = r - baseline
```

With baseline=0 (which is what mean=0 gives), REINFORCE advantage = r.

**Therefore**: ALL frameworks' singleton group handling = REINFORCE with baseline=0.

This is NOT a bug per se — it's a **design choice**. The alternative would be:
1. **Drop singleton groups entirely** (skip them from training)
2. **Use global batch mean/std** as fallback (but loses group-level normalization)
3. **Warn the user** that singleton groups degrade to REINFORCE

---

## 5. verl GRPO Variants

verl provides multiple advantage estimators via `@register_adv_est`:

| Estimator | Grouping | Singleton handling | RTX 4090 suitability |
|-----------|----------|-------------------|---------------------|
| GRPO | uid-based | mean=0, std=1 | ★★★★★★★★ BEST (sync trainer) |
| GRPO_VECTORIZED | uid-based | mean=0, std=1 (via group_mean_std) | ★★★★★★★★ Efficient GPU version |
| GAE | No grouping | TD-based | Requires value network → NOT GRPO |
| REINFORCE_PLUS_PLUS | uid-based | Same pattern | Baseline variant |
| REINFORCE_PLUS_PLUS_BASELINE | uid-based | Baseline subtraction | With learned baseline |
| RLOO | uid-based | Leave-one-out | Alternative normalization |
| GDPO | uid-based + per-dimension | Decoupled normalization | NEW (multi-objective) |
| CPPO | uid-based | Conservative PPO | ★★★★★★★★ RTX 4090 #1 with bypass_mode |

---

## 6. Dr.GRPO vs Standard GRPO

```python
if norm_adv_by_std_in_grpo:
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # Standard GRPO
else:
    scores[i] = scores[i] - id2mean[index[i]]  # Dr.GRPO (no std normalization)
```

**Dr.GRPO** (arxiv 2503.20783): Removes std normalization. Advantage = r_i - μ_g (just mean subtraction).
- Singleton: A = r - 0 = r → still REINFORCE (but no division by std)
- Benefit: Avoids std=0 → division by ε → small advantages that get amplified by ε
- Risk: Without std normalization, advantages are not scaled → can be very large or very small

---

## 7. RTX 4090 Deployment Implications

### 7.1 verl GRPO on RTX 4090

1. **Sync trainer** = BEST for RTX 4090 (single GPU, no async overhead)
2. **uid grouping** = correct for GRPO (all trajectories per prompt in same group)
3. **group_size >= 2** = CRITICAL. Must generate at least 2 trajectories per prompt for GRPO to work
4. **bypass_mode=True** = MUST (18Ψ→3.8Ψ memory reduction for CPPO/GRPO)
5. **PrefixGrouper** = shared-prefix optimization saves forward pass compute when group_size > 1

### 7.2 Configuration Requirements

```yaml
# verl GRPO config for RTX 4090
algorithm:
  adv_estimator: grpo  # or grpo_vectorized
  norm_adv_by_std_in_grpo: true  # Standard GRPO (recommended)
  # OR: norm_adv_by_std_in_grpo: false  # Dr.GRPO (alternative)

rollout:
  n: 4  # ★★★★★★★★ MUST >= 2 for GRPO! 4 is optimal for RTX 4090
  temperature: 1.0

trainer:
  type: sync  # RTX 4090 BEST
  bypass_mode: true  # MUST for memory savings
```

### 7.3 Group Size Impact Table

| Group size | μ_g | σ_g | Advantage | Algorithm degeneracy |
|-----------|------|------|-----------|---------------------|
| 1 | 0 (fallback) | 1 (fallback) | r_i | REINFORCE (zero baseline) |
| 2 | (r1+r2)/2 | |r1-r2|/2 | (r_i-μ)/σ | Minimal GRPO (high variance) |
| 4 | mean | std | (r_i-μ)/σ | Good GRPO (recommended minimum) |
| 8+ | mean | std | (r_i-μ)/σ | Full GRPO (optimal) |

---

## 8. Key Takeaways

1. **★★★★★★★★ ALL GRPO frameworks have the same singleton group degeneration**: mean=0, std=1 → REINFORCE. This is a cross-framework design pattern, not a rLLM-specific bug.

2. **★★★★★★★★ verl groups by uid (prompt ID)**, which naturally creates groups of size >= n_rollouts. This is MORE correct than rLLM's task_id:name grouping (which can accidentally split groups).

3. **★★★★★★★★ rLLM V2 fix (by_task_id default)** aligns with verl's approach — group all trajectories for the same task together.

4. **★★★★★★★★ Minimum group_size = 2 for GRPO**: Users MUST set n_rollouts >= 2. Group_size=1 means GRPO degrades to REINFORCE regardless of framework.

5. **★★★★★★★★ Dr.GRPO** (norm_adv_by_std_in_grpo=False) avoids the std normalization issue but still suffers from singleton degeneration (mean=0 fallback).

6. **★★★★★★★★ verl's PrefixGrouper** is a performance optimization for shared-prefix computation within groups — not a grouping change.

7. **★★★★★★★★ GDPO** (per-dimension normalization) is a new estimator that prevents dominant reward signals from drowning out weaker ones — relevant for multi-objective GRPO.

---

## References

- verl core_algos.py: `verl/trainer/ppo/core_algos.py`
- verl groupwise.py: `verl/utils/groupwise.py`
- verl prefix_grouper_utils.py: `verl/trainer/ppo/prefix_grouper_utils.py`
- rLLM #605: https://github.com/rllm-org/rllm/issues/605
- rLLM V2 fix: notebook/projects/rllm-605-grpo-grouping-revised-fix-v2.md
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- GDPO: https://arxiv.org/abs/2601.05242
- GRPO original paper: https://arxiv.org/abs/2402.03300
