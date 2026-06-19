# rLLM GRPO Advantage Computation — Deep Reading

> 2026-06-19 | Post-PR #667 deep analysis of GRPO advantage computation pipeline
> ★★★★★★★★ CONFIRMED: group_size=1 → group_mean=0, group_std=1 → advantages=raw rewards → REINFORCE degeneracy
> ★★★★★★★★ rLLM uses registry pattern for advantage estimators (GRPO, RLOO)
> ★★★★★★★★ PR #667 fix changes grouping key to task_id only → restores proper group-level normalization

---

## 1. Architecture Overview

rLLM's advantage computation pipeline:

```
transform.py → transform_episodes_to_trajectory_groups()
  → _build_trajectory_groups() [GROUPING — our bug fix]
  → _validate_and_propagate_rewards() [VALIDATION]

advantage.py → collect_reward_and_advantage_from_trajectory_groups()
  → get_rllm_adv_estimator(config) [REGISTRY LOOKUP]
  → calculate_grpo_advantages() [GRPO computation]

rl_algo.py → calculate_grpo_advantages_per_group() [CORE ALGORITHM]
  → group_mean, group_std computation
  → (rewards - mean) / (std + epsilon) normalization
```

---

## 2. GRPO Advantage Algorithm (rl_algo.py)

```python
def calculate_grpo_advantages_per_group(
    rewards: np.ndarray,
    norm_adv_by_std_in_grpo=True,
    episilion=1e-6
) -> tuple[np.ndarray, np.ndarray]:
    if len(rewards) <= 1:
        group_mean, group_std = 0.0, 1.0  # ★★★★★★★★ BUG TRIGGER!
    else:
        group_mean = np.mean(rewards)
        group_std = np.std(rewards)

    if norm_adv_by_std_in_grpo:
        advantages = (rewards - group_mean) / (group_std + episilion)
    else:
        advantages = rewards - group_mean

    return advantages, advantages
```

### ★★★★★★★★ Bug Analysis: group_size=1 → REINFORCE Degeneracy

When grouping by `task_id:trajectory_name` (the broken grouping):
- Each group has exactly 1 trajectory
- `len(rewards) == 1` → enters the `<= 1` branch
- `group_mean = 0.0, group_std = 1.0` (fallback values)
- `advantages = (rewards - 0.0) / (1.0 + 1e-6) ≈ rewards`
- This is **REINFORCE** (raw rewards), NOT GRPO (group-normalized advantages)

When grouping by `task_id` only (the fixed grouping):
- Each group has N trajectories (N = number of agents/samples per task)
- `len(rewards) >= 2` → enters the proper computation branch
- `group_mean = np.mean(rewards)`, `group_std = np.std(rewards)`
- `advantages = (rewards - mean) / (std + epsilon)` — proper GRPO normalization
- Variance reduction via group baseline (the whole point of GRPO!)

### Mathematical Proof of Degeneracy

GRPO advantage: $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$

With group_size=1: $\hat{A}_i = \frac{r_i - 0}{1 + \epsilon} = r_i$

This is REINFORCE: $\hat{A}_i = r_i$ (no baseline, no variance reduction)

GRPO provides $\approx 1/K$ variance reduction where $K$ = group size. With $K=1$, variance reduction = 0. This means:
- Gradient variance = $O(1)$ (same as REINFORCE)
- Sample efficiency = same as REINFORCE
- All GRPO benefits are LOST

---

## 3. Advantage Estimator Registry Pattern

rLLM uses a decorator-based registry for advantage estimators:

```python
RLLM_ADV_ESTIMATOR_REGISTRY: dict[str, Callable] = {}

@register_rllm_adv_estimator(rLLMAdvantageEstimator.GRPO)
def calculate_grpo_advantages(...): ...

@register_rllm_adv_estimator(rLLMAdvantageEstimator.RLOO)
def calculate_rloo_advantages(...): ...
```

This is a clean extensibility pattern — users can add custom advantage estimators via:
```python
@register_rllm_adv_estimator("my_custom_estimator")
def my_estimator(rewards, config, **kwargs): ...
```

### RLOO Algorithm

```python
def calculate_rloo_advantages_per_group(rewards: np.ndarray):
    num_trajs = len(rewards)
    if num_trajs <= 1:
        return rewards, rewards  # ★★★★★★★★ Same degeneracy risk!
    advantages = num_trajs / (num_trajs - 1) * (rewards - rewards.mean())
    return advantages, advantages
```

RLOO (Reverse Leave-One-Out) uses a leave-one-out baseline:
- $\hat{A}_i = \frac{K}{K-1}(r_i - \bar{r}_{-i})$
- Provides even better variance reduction than GRPO
- BUT has the SAME group_size=1 degeneracy issue!

This confirms that our grouping fix (#667) affects BOTH GRPO and RLOO advantage estimators.

---

## 4. Transform Pipeline — Full Call Chain

```
transform_episodes_to_trajectory_groups(episodes, config)
  → _impute_trajectory_names(episodes, config)  # Step 1: Name imputation
  → trajectory_grouping_hook(episodes, config, compact_filtering)  # Step 2: Grouping
    → _default_trajectory_grouping_hook():
      → _build_trajectory_groups(episodes, compact_filtering)  # ★★★★★★★★ BUG HERE!
        → trajectories_by_name[f"{task_id}:{trajectory.name}"]  # BROKEN: includes name
        → trajectories_by_name[task_id]  # FIXED: only task_id
      → _validate_and_propagate_rewards(groups, config)  # Reward validation
  → _get_transform_metrics(episodes, groups)  # Step 3: Metrics
  → return groups, metrics
```

### Compact Filtering

`CompactFilteringConfig.should_mask(termination_reason)` can filter out episodes:
- Only include episodes with specific termination reasons
- Useful for removing failed/crashed episodes from advantage computation
- Applied BEFORE grouping → doesn't affect the grouping bug

---

## 5. PR #667 Impact Verification

### Before Fix (broken grouping):
- Group key = `task_id:trajectory_name` (e.g., "math_problem_1:agent_0")
- Group size = 1 (each trajectory is unique)
- GRPO advantages = raw rewards (REINFORCE)
- RLOO advantages = raw rewards (REINFORCE)
- No variance reduction → poor sample efficiency

### After Fix (correct grouping):
- Group key = `task_id` (e.g., "math_problem_1")
- Group size = K (number of trajectories per task)
- GRPO advantages = (rewards - mean) / (std + epsilon) (proper GRPO)
- RLOO advantages = K/(K-1) * (rewards - mean) (proper RLOO)
- Variance reduction ≈ 1/K → good sample efficiency

### Cross-Framework Comparison

| Framework | Grouping Key | Group Size | Advantage Type |
|-----------|-------------|-----------|---------------|
| rLLM (after fix) | task_id | K | GRPO (proper) |
| rLLM (before fix) | task_id:name | 1 | REINFORCE (broken) |
| verl | prompt/task_id | K | GRPO (proper) |
| DeepSpeed | prompt/task_id | K | GRPO (proper) |
| Megatron | prompt/task_id | K | GRPO (proper) |

All other frameworks group by prompt (task_id), not by agent name. Our fix aligns rLLM with the established pattern.

---

## 6. Additional rLLM Architecture Findings

### AlgorithmConfig

`rllm/trainer/algorithms/config.py` defines `AlgorithmConfig` with:
- `rLLMAdvantageEstimator`: enum for GRPO/RLOO/custom
- `norm_adv_by_std_in_grpo`: bool for GRPO normalization
- Other algorithm parameters

### Metrics Module

`rllm/trainer/algorithms/metrics.py` provides advantage computation metrics.

### Visualization Module

`rllm/trainer/algorithms/visualization.py` provides trajectory group visualization.

### Rejection Sampling

`rllm/trainer/algorithms/rejection_sampling.py` provides rejection sampling filtering.

All these modules work on TrajectoryGroups — so the grouping bug affects ALL downstream computation, not just advantage estimation.

---

## 7. Cross-Framework Singleton Degeneration (NEW — 2026-06-19 Session 2)

★★★★★★★★ ALL GRPO frameworks have the SAME singleton group handling:

| Framework | Grouping key | Singleton handling | Source |
|-----------|-------------|-------------------|--------|
| **verl** | uid (prompt ID) | mean=0, std=1 | core_algos.py ~342, groupwise.py ~130 |
| **rLLM** (V2) | task_id (default) | implicit via std=0→ε | TransformConfig grouping_mode |
| **TRL** | prompt | mean=0, std=1 | GRPOTrainer |

The mathematical consequence: group_size=1 → advantage = raw reward → REINFORCE degeneracy.

This is NOT a rLLM-specific bug — it's a **cross-framework design pattern** that all GRPO implementations inherit from the original paper's formulation.

**verl's groupwise.py implementation** (most detailed):

```python
# verl/utils/groupwise.py — group_mean_std()
single = count <= 1.0
if torch.any(single):
    mean = mean.clone()
    std = std.clone()
    mean[single] = 0.0    # ← same as rLLM
    std[single] = 1.0     # ← same as rLLM
```

**verl's uid-based grouping is MORE correct**: All trajectories sharing the same prompt are naturally grouped together, unlike rLLM's task_id:name grouping which can accidentally split groups.

See: `notebook/projects/cross-framework-grpo-advantage-comparison.md` for full analysis.

---

## References

- PR #667: https://github.com/rllm-org/rllm/pull/667
- rLLM repo: rllm-org/rllm
- Transform pipeline: rllm/trainer/algorithms/transform.py
- Advantage computation: rllm/trainer/algorithms/advantage.py + rl_algo.py
- GRPO algorithm: notebook/fundamentals/transformer-architecture-mathematical-derivation.md
- Cross-framework comparison: notebook/projects/cross-framework-grpo-advantage-comparison.md
- verl V1 trainer: notebook/projects/verl-v1-trainer-architecture-deep-reading.md
