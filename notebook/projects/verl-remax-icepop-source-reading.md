# verl ReMax (#6340) and IcePop (#5722) Source-Level Reading

> 2026-06-16 | verl-project/verl | PR #6340 (merged) + PR #5722 (merged) | Source-level deep analysis
> ★★★★★ Two critical merged algorithms that directly impact RTX 4090 GRPO training

## Part I: ReMax Algorithm (#6340) — Source-Level Analysis

### 1. ★★★★★ Core Advantage Function: compute_remax_outcome_advantage

**File**: `verl/trainer/ppo/core_algos.py` (line 732-765)

```python
@register_adv_est(AdvantageEstimator.REMAX)  # Registered as "remax"
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,   # KEY: greedy baseline reward (scalar per sequence)
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
    return advantages, returns
```

★★★★★★★ Mathematical formulation:
- `returns[t] = sum(token_level_rewards[t:end])` — cumulative reward from token t to end
- `advantages[t] = returns[t] - greedy_baseline` — per-token advantage subtracting greedy reward
- The greedy baseline is a **single scalar** per sequence, broadcast to all tokens via `unsqueeze(-1)`
- This is fundamentally different from GRPO where baseline = group_mean (statistical), vs ReMax where baseline = greedy_response_reward (deterministic)

★★★★★★★ Key difference from GRPO:
- GRPO: `advantage = (r_i - mu_group) / sigma_group` — relative to OTHER sampled responses
- ReMax: `advantage = r_i - greedy_reward` — relative to the BEST POSSIBLE response under current policy
- ReMax baseline has **zero variance** (deterministic greedy sampling) — no group statistics needed
- ReMax advantage captures: "How much better is this sampled response than what the model would choose greedily?"

★★★★★★★ Critical: `reward_baselines` shape is `(batch_size,)`, NOT `(batch_size, seq_length)`
- Greedy baseline is a single **total sequence reward** (outcome reward)
- The `.unsqueeze(-1)` broadcasts to all tokens, creating a flat per-token baseline
- No token-level granularity in baseline — entire response gets same baseline subtraction

### 2. ★★★★★ Greedy Baseline Generation: apply_greedy_sampling_params

**File**: `verl/trainer/main_ppo_sync.py` (line 109-113)

```python
def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0
```

★★★★★★★ This is how the greedy baseline is computed:
- `temperature=0` forces deterministic (greedy) token selection
- `top_p=1.0` and `top_k=-1` remove all sampling constraints
- The greedy response is the model's **argmax** sequence — the single best path
- Its total reward serves as the per-prompt baseline

★★★★★★★ Why this works: Under temperature=0, the model picks the single highest-probability token at each step. This gives the "model's best guess" for each prompt. Any sampled response that scores higher than this greedy baseline has positive advantage (good), any that scores lower has negative advantage (bad).

### 3. ★★★★★ ReMax + TransferQueue Sync Trainer Flow

**File**: `verl/trainer/main_ppo_sync.py` (lines 1688-1720)

★★★★★★★ Step-by-step ReMax flow in the sync trainer (PPOTrainer.step):

```python
# Step 1: Create TWO batches — sampled + baseline
if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
    rollout_n = self.config.actor_rollout_ref.rollout.n
    sampled_batch_dict = batch_dict.copy()
    sampled_batch_dict["__do_sample__"] = np.ones(len(batch_dict), dtype=bool)     # do sampling
    sampled_batch_dict["__rollout_n__"] = np.full(len(batch_dict), rollout_n)       # n samples per prompt

    baseline_batch_dict = batch_dict.copy()
    baseline_batch_dict["uid"] = np.array([f"remax_baseline_{uid}" for uid in batch_dict["uid"]])
    baseline_batch_dict["__do_sample__"] = np.zeros(len(batch_dict), dtype=bool)   # greedy mode
    baseline_batch_dict["__rollout_n__"] = np.ones(len(batch_dict), dtype=np.int64) # exactly 1 greedy per prompt

    batch = torch.cat([tu.get_tensordict(sampled_batch_dict), tu.get_tensordict(baseline_batch_dict)])
```

★★★★★★★ Key design decisions:
- Both sampled and baseline go into a **single batch** sent to rollout engine
- Baseline uids are prefixed with `"remax_baseline_"` — distinguish from sampled trajectories
- `__do_sample__=False` triggers `apply_greedy_sampling_params()` in AgentLoopWorkerTQ
- `__rollout_n__=1` means exactly one greedy trajectory per prompt
- The rollout engine processes both in one call — no separate inference needed

★★★★★★★ Why single-batch is important (ray_trainer.py line 1451-1454 comment):
```
NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
Keep them in a single agent-loop/vLLM request to avoid sending a second
rollout after resumes have been put to sleep, which can leave async vLLM
engines in an invalid state for multi-turn agent workloads.
```

### 4. ★★★★★ ReMax Baseline Extraction: _add_remax_reward_baselines

**File**: `verl/trainer/main_ppo_sync.py` (lines 1212-1255)

★★★★★★★ The function:
1. Scans KV batch keys for `"remax_baseline_"` prefix to identify baseline trajectories
2. For multi-output agent loops: keeps only the **FINAL output** per baseline (highest index)
3. Extracts `rm_scores.sum(dim=-1)` — total outcome reward for each baseline
4. Matches each sampled trajectory's uid to its corresponding baseline uid
5. Writes `reward_baselines` tensor back to TransferQueue for sampled keys
6. Cleans up all baseline data from TransferQueue (no lingering KV cache)

★★★★★★★ Critical observations:
- Baseline data is **extracted, matched to sampled, then cleaned up** — no lingering baseline trajectory in KV
- The greedy baseline reward = `rm_scores.sum(dim=-1)` — total outcome reward (not token-level)
- For multi-output agent loops: only the **final output** per baseline is used (highest index)
- The `removeprefix("remax_baseline_")` maps `"remax_baseline_<uid>"` back to `<uid>` for matching

### 5. ★★★★★ ReMax + use_kl_in_reward Interaction

★★★★★★★ When `use_kl_in_reward=True` (standard PPO/GRPO KL penalty mode):
- KL penalty is applied BEFORE advantage computation
- `token_level_rewards = token_level_scores - beta * kl_divergence`
- ReMax then computes: `advantages = returns(token_level_rewards) - greedy_baseline`
- **PROBLEM**: The greedy baseline was computed with raw reward (no KL penalty), but sampled rewards have KL subtracted
- This creates an **inconsistency**: baseline is from raw reward, advantages use KL-penalized reward
- ★★★★★★★ ReMax canonical requires ref model for KL → NOT compatible with bypass_mode!

★★★★★★★ When `use_kl_in_reward=False` (bypass_mode semantics):
- No KL penalty in reward
- `token_level_rewards = token_level_scores` (raw reward, no modification)
- ReMax computes: `advantages = returns(raw_reward) - greedy_baseline`
- Both sampled and baseline use raw reward → **consistent baseline**
- ★★★★★★★ This is the **correct** ReMax configuration for RTX 4090

★★★★★★★ ReMax + bypass_mode = RTX 4090 optimal ReMax config:
```yaml
algorithm:
  adv_estimator: remax
  use_kl_in_reward: false    # MUST — keep baseline consistent
actor_rollout_ref:
  rollout:
    n: 4-8                    # sampled responses per prompt
  actor:
    use_kl_loss: true          # KL in loss, not reward
    kl_loss_coef: 0.05
```

★★★★★★★ ReMax requires ref model for `use_kl_in_reward=True`:
- The `apply_kl_penalty()` function uses `ref_log_prob` — needs reference model
- ★★★★★★ In bypass_mode: KL penalty is zero — no ref model needed — consistent baseline
- ★★★★★ Without bypass: ref model needed → adds 14GB → RTX 4090 NOT feasible

### 6. ★★★★★ AdvantageEstimator enum and registry path

**File**: `verl/trainer/ppo/core_algos.py` (line 101)

```python
class AdvantageEstimator(str, Enum):
    ...
    REMAX = "remax"   # Line 101
```

Config path: `algorithm.adv_estimator: "remax"`

The registry lookup in `compute_advantage()` (ray_trainer.py line 246-277):
```python
adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
adv_kwargs = {
    "token_level_rewards": data.batch["token_level_rewards"],
    "response_mask": data.batch["response_mask"],
    "config": config
}
if "reward_baselines" in data.batch:  # KEY: ReMax needs reward_baselines
    adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
```

★★★★★★★ The `reward_baselines` key must be present in `data.batch` for ReMax. It is injected by:
- ray_trainer.py: `batch["reward_baselines"] = reward_baseline_tensor` (line ~1492)
- main_ppo_sync.py: `_add_remax_reward_baselines()` writes to TransferQueue (line 1246)

### 7. ★★★★★ ReMax vs GRPO vs PPO — Source-Level Comparison

| Aspect | GRPO | ReMax | PPO (GAE) |
|--------|------|-------|-----------|
| Baseline | `mean(group_rewards)` (statistical) | `greedy_response_reward` (deterministic) | `V(s)` (learned) |
| Variance | Medium (group std) | ★★★★★ Lowest (zero-variance baseline) | Low (learned) |
| Needs Critic | No | No | Yes (V network) |
| Needs Ref Model | Optional (bypass) | Optional (bypass) | Yes (KL) |
| RTX 4090 Memory | ~18GB (actor+ref bypass) | ★★★★★ ~18GB (actor only, bypass) | ~28GB (actor+critic+ref) |
| Extra Rollout Cost | None | +1 greedy per prompt (n+1 total) | None |
| Advantage Type | Outcome-only (scalar per response) | Outcome-only (scalar per response) | Token-level (GAE residual) |
| Group Dependency | Needs n>=4 for normalization | ★★★★★ Works with n=1 (baseline independent) | No group needed |

★★★★★★★ ReMax uniqueness: The ONLY advantage estimator that works perfectly with n=1 (single sample per prompt), because the baseline is external (greedy) rather than group-dependent. GRPO with n=1 degenerates to advantage=raw_reward (no normalization).

### 8. ★★★★★ Source File Summary for ReMax

| File | Lines | Purpose |
|------|-------|---------|
| `verl/trainer/ppo/core_algos.py` | 732-765 | `compute_remax_outcome_advantage()` — advantage computation |
| `verl/trainer/ppo/core_algos.py` | 96-101 | `AdvantageEstimator.REMAX="remax"` — enum registration |
| `verl/trainer/main_ppo_sync.py` | 109-113 | `apply_greedy_sampling_params()` — greedy rollout config |
| `verl/trainer/main_ppo_sync.py` | 1691-1720 | ReMax batch creation (sampled + baseline) |
| `verl/trainer/main_ppo_sync.py` | 1212-1255 | `_add_remax_reward_baselines()` — baseline extraction from TransferQueue |
| `verl/trainer/ppo/ray_trainer.py` | 1450-1494 | ReMax in async trainer — combined batch generation |
| `verl/trainer/ppo/ray_trainer.py` | 185-280 | `compute_advantage()` — dispatches to ReMax estimator |
| `verl/trainer/config/algorithm.py` | 101 | `REMAX="remax"` — enum in AlgoConfig |

---

## Part II: IcePop IS Correction (#5722) — Source-Level Analysis

### 1. ★★★★★ What is IcePop? — Importance Sampling with Lower-Upper Bounds

★★★★★★★ IcePop = **I**mportance sampling **C**orrection with **E**xact **Pop**ulation bounds (lower_upper threshold)

The name comes from the `"lower_upper"` threshold specification format, e.g., `"0.5_5.0"`.

★★★★★★★ IcePop vs TIS (Truncated Importance Sampling) — the critical difference:

| Mechanism | TIS (standard) | IcePop |
|-----------|----------------|--------|
| Threshold | Single float (e.g., `2.0`) | `"lower_upper"` string (e.g., `"0.5_5.0"`) |
| Action on high weights | `.clamp(max=threshold)` → clip | Zero entirely (remove from population) |
| Action on low weights | No lower bound | Zero entirely (remove from population) |
| Effect on response_mask | **No change** | **No change** (IcePop only changes IS coefficients) |
| Mathematical effect | Truncated (biased, low variance) | ★★★★★ Exact population (unbiased within bounds) |
| Out-of-range handling | Clip to threshold value | ★★★★★ Set to exactly zero (remove from effective population) |

★★★★★★★ The key insight: IcePop zeros out-of-range IS weights to EXACTLY zero, while TIS clips them to the threshold value. IcePop produces a cleaner "effective population" — only samples within the trust region contribute gradient. TIS distorts the contribution of extreme samples by clipping rather than removing.

### 2. ★★★★★ IcePop Implementation in rollout_corr_helper.py

**File**: `verl/trainer/ppo/rollout_corr_helper.py`

The IcePop code path is in `compute_rollout_correction_weights()` (lines 520-655):

```python
# Parse threshold — key distinction
rollout_is_threshold_upper, rollout_is_threshold_lower = _parse_rollout_is_threshold(rollout_is_threshold)
use_icepop = rollout_is_threshold_lower is not None  # True when "lower_upper" format

# Compute IS weights from log ratio
if rollout_is == "token":
    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    raw_rollout_is_weights = torch.exp(log_ratio_safe)
elif rollout_is == "sequence":
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    raw_rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)

# Zero out padding tokens
raw_rollout_is_weights = raw_rollout_is_weights * response_mask

# ★★★★★ THE CRITICAL BRANCH: IcePop vs TIS
if not use_icepop:  # TIS mode
    rollout_is_weights = raw_rollout_is_weights.clamp(max=rollout_is_threshold_upper)  # clip
else:               # IcePop mode
    assert rollout_is_threshold_lower is not None
    token_kept_mask = (raw_rollout_is_weights >= rollout_is_threshold_lower) & (
        raw_rollout_is_weights <= rollout_is_threshold_upper
    )
    # ★★★★★★★★ ZERO out-of-range weights (not clip!)
    rollout_is_weights = torch.where(
        token_kept_mask, raw_rollout_is_weights, torch.zeros_like(raw_rollout_is_weights)
    )
```

★★★★★★★★★ The `torch.where()` is the heart of IcePop:
- If weight is within `[lower, upper]`: keep the **exact** weight (no distortion, no bias from clipping)
- If weight is outside `[lower, upper]`: set to **exactly zero** (remove from effective population)
- This creates a **hard boundary** in the IS weight space, like rejection sampling on weights
- Unlike rejection sampling, IcePop does NOT modify `response_mask` — only the IS coefficients change

★★★★★★★ IcePop out-of-bound (OOB) metric:
```python
if use_icepop:
    oob_mask = (raw_rollout_is_weights < rollout_is_threshold_lower) | (
        raw_rollout_is_weights > rollout_is_threshold_upper
    )
    metrics["rollout_is_oob_ratio"] = verl_F.masked_mean(oob_mask.float(), response_mask).item()
```

### 3. ★★★★★ The `_parse_rollout_is_threshold` Function

**File**: `verl/trainer/ppo/rollout_corr_helper.py` (lines 93-129)

```python
def _parse_rollout_is_threshold(threshold_spec: str | float) -> tuple[float, Optional[float]]:
    if isinstance(threshold_spec, int | float):
        upper = float(threshold_spec)     # e.g., 2.0
        lower = None                      # No lower bound = TIS mode
    elif isinstance(threshold_spec, str):
        if "_" in spec:
            lower_str, upper_str = spec.split("_", 1)  # e.g., "0.5_5.0"
            lower = float(lower_str)    # IcePop lower bound
            upper = float(upper_str)    # IcePop upper bound
        else:
            upper = float(spec)         # Single number = TIS upper only
            lower = None
    return upper, lower
```

★★★★★★★ The parsing is the decision point:
- `"2.0"` or float `2.0` → TIS mode (upper only, no lower)
- `"0.5_5.0"` → IcePop mode (both bounds, exact population)

### 4. ★★★★★ Token-Level vs Sequence-Level IcePop

★★★★★★★ Token-level IcePop (`rollout_is="token"` + `"0.5_5.0"`):
- Each token's IS weight ρ_t is independently checked against [0.5, 5.0]
- Out-of-range tokens get weight=0, in-range tokens keep exact weight
- Per-token granularity: only the problematic tokens are zeroed, not the entire sequence
- ★★★★★★★★ A sequence with 95% in-range tokens and 5% out-of-range tokens still contributes gradient for the 95% — only the 5% outlier tokens are silenced

★★★★★★★ Sequence-level IcePop (`rollout_is="sequence"` + `"0.5_5.0"`):
- The product ρ_seq = ∏_t ρ_t is checked against [0.5, 5.0]
- If the entire sequence ratio is out of range, ALL tokens in that sequence get weight=0
- More aggressive: entire sequences are removed from the effective population
- ★★★★★★★★ This is equivalent to rejection sampling on IS weights (but doesn't modify response_mask)

### 5. ★★★★★ IcePop Preset Configurations

**File**: `verl/trainer/config/algorithm.py` (lines 212-230, 370-389)

```python
@classmethod
def decoupled_token_icepop(cls, threshold=5.0, threshold_lower=0.5):
    """Decoupled Mode with exact token-level IcePop."""
    return cls(rollout_is="token", rollout_is_threshold=f"{threshold_lower}_{threshold}", rollout_rs=None)

@classmethod
def bypass_pg_token_icepop(cls, threshold=5.0, threshold_lower=0.5):
    """Bypass mode with REINFORCE loss and exact token-level IcePop."""
    return cls(
        rollout_is="token",
        rollout_is_threshold=f"{threshold_lower}_{threshold}",
        rollout_rs=None,
        bypass_mode=True,
        loss_type="reinforce",
    )
```

★★★★★★★ Default IcePop bounds: `[0.5, 5.0]`:
- ρ_t < 0.5 → policy diverged significantly in negative direction → weight=0
- ρ_t > 5.0 → policy diverged significantly in positive direction → weight=0
- Within [0.5, 5.0]: exact IS weight, no distortion

### 6. ★★★★★ IcePop + bypass_mode Interaction

★★★★★★★ In bypass mode (`bypass_mode=True`), IcePop IS weights are computed **inside the loss function**:

```python
# In bypass mode: old_log_prob IS rollout_log_prob
rollout_log_prob = old_log_prob

# Compute IS weights DURING loss computation (not in trainer)
with torch.no_grad():
    rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
        compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob,         # π_current (evolving!)
            rollout_log_prob=rollout_log_prob,  # π_rollout (frozen)
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,  # IcePop bounds
            ...
        )
    )
```

★★★★★★★★★ Critical observation: In bypass mode, IS weights (including IcePop) are computed **inside the loss function**, not in the trainer. This means:
1. The IS ratio uses the CURRENT policy π_θ (not a frozen reference)
2. IcePop bounds are checked against π_θ/π_rollout at every gradient step
3. As π_θ evolves during training, the OOB fraction changes
4. The `.detach()` on IS weights (line 623) prevents gradient flow through the weights

### 7. ★★★★★ IcePop vs Rejection Sampling — The Separation

★★★★★★★★★ IcePop and RS are **orthogonal mechanisms** in verl:

| Mechanism | What it modifies | IcePop | RS |
|-----------|-----------------|--------|-----|
| IS weights | `rollout_is_weights` tensor | Yes (zeros out-of-range) | No |
| Response mask | `response_mask` tensor | **No** (does NOT modify!) | Yes (sets rejected to 0) |
| Effect on loss | Weight multiplies gradient | Zero weight = zero gradient contribution | Mask excludes from aggregation |
| Theoretical basis | Change of measure (IS) | Exact population bounds | Hard trust region filter |

★★★★★★★ They CAN be combined:
```yaml
rollout_is: token                  # IcePop IS weights
rollout_is_threshold: "0.5_5.0"    # IcePop bounds
rollout_rs: seq_mean_k1            # Geometric RS mask
rollout_rs_threshold: "0.999_1.001" # RS bounds
```

### 8. ★★★★★ IcePop + CPPO Relationship

★★★★★★★★★ IcePop and CPPO (#6731, open) address the same problem from different angles:

| Aspect | IcePop (#5722) | CPPO (#6731) |
|--------|----------------|---------------|
| Problem | Off-policy distribution shift | Off-policy trust region |
| Approach | IS weight correction (hard bounds) | Position-weighted cumulative prefix divergence |
| Bounds | [lower, upper] on IS ratio (e.g., [0.5, 5.0]) | Prefix divergence threshold |
| With bypass_mode | ✓ (REINFORCE + explicit IS) | ✓ (bypass_mode REQUIRED) |
| Gradient distortion | None within bounds (exact) | Near-zero overhead (prefix-weighted) |
| Out-of-range handling | Weight=0 (remove from population) | Advantages modulated by divergence |
| RTX 4090 | ★★★★★ bypass_pg_token_icepop (simple, correct) | ★★★★★ bypass_mode + CPPO (best trust region) |

★★★★★★★★★ IcePop + CPPO cannot combine directly:
- IcePop zeros IS weights for out-of-range tokens → some tokens contribute zero gradient
- CPPO modulates advantages based on prefix divergence → all tokens contribute with adjusted magnitude
- They serve different purposes: IcePop = population control, CPPO = trust region shape
- ★★★★★★★★ For RTX 4090: choose ONE — IcePop for simplicity, CPPO for best trust region

### 9. ★★★★★ Mathematical Formulation — IcePop vs TIS

★★★★★★★ TIS (Truncated Importance Sampling):
```
w_t = min(ρ_t, C_IS)    where ρ_t = π_train/π_rollout
```
- Clips high weights to C_IS (biased, reduces variance)
- No lower bound (preserves unbiasedness for small weights)
- The clipped weight still contributes gradient (just capped)

★★★★★★★ IcePop (Exact Population IS):
```
w_t = ρ_t  if lower ≤ ρ_t ≤ upper
w_t = 0    otherwise
```
- Within bounds: EXACT weight (no distortion, no bias from clipping)
- Outside bounds: EXACTLY zero (remove from effective population)
- ★★★★★★★★ This is closer to the theoretical ideal: use only the "good" samples, ignore the "bad" ones entirely

★★★★★★★ The bias-variance tradeoff:
- TIS: Biased (clipped weights distort the measure), but all samples contribute some gradient
- IcePop: Unbiased within bounds (exact weights), but zeroed samples contribute nothing (reduced effective sample size)
- ★★★★★★★★ IcePop preferred when out-of-range samples are likely "toxic" (noise, not signal)
- ★★★★★★ TIS preferred when all samples carry signal but some are overweighted

### 10. ★★★★★ IcePop Metric Tracking

★★★★★★★ IcePop-specific metrics:

```python
# OOB (Out-of-Bounds) ratio — fraction of tokens/sequences zeroed by IcePop
metrics["rollout_is_oob_ratio"] = verl_F.masked_mean(oob_mask.float(), response_mask).item()
```

★★★★★★★ Standard IS metrics also apply:
- `rollout_corr/rollout_is_mean`: Mean weight (should be near 1.0)
- `rollout_corr/rollout_is_eff_sample_size`: ESS = 1/E[w²] (lower with IcePop due to zeroed weights)
- `rollout_corr/rollout_is_ratio_fraction_high`: Fraction above upper bound
- `rollout_corr/rollout_is_ratio_fraction_low`: Fraction below lower bound

★★★★★★★ Monitoring recommendation for RTX 4090:
- Watch `rollout_is_oob_ratio` — if >20%, bounds may be too tight
- Watch `rollout_is_eff_sample_size` — if <0.5, effective population is too small
- Start with `"0.5_5.0"` and widen if too many zeroed tokens

### 11. ★★★★★ Source File Summary for IcePop

| File | Lines | Purpose |
|------|-------|---------|
| `verl/trainer/ppo/rollout_corr_helper.py` | 520-655 | `compute_rollout_correction_weights()` — IcePop branch |
| `verl/trainer/ppo/rollout_corr_helper.py` | 93-129 | `_parse_rollout_is_threshold()` — IcePop vs TIS detection |
| `verl/trainer/ppo/rollout_corr_helper.py` | 593-602 | IcePop zeroing logic: `torch.where(token_kept_mask, weight, 0)` |
| `verl/trainer/ppo/rollout_corr_helper.py` | 614-619 | IcePop OOB metric computation |
| `verl/trainer/ppo/rollout_corr_helper.py` | 779-894 | `compute_rollout_correction_and_rejection_mask()` — unified pipeline |
| `verl/trainer/config/algorithm.py` | 212-230 | `decoupled_token_icepop()` preset |
| `verl/trainer/config/algorithm.py` | 370-389 | `bypass_pg_token_icepop()` preset |
| `verl/trainer/ppo/core_algos.py` | 2351-2487 | `compute_policy_loss_bypass_mode()` — bypass + IcePop integration |
| `verl/trainer/config/algorithm/rollout_correction.yaml` | 1-28 | YAML default config |
| `verl/docs/algo/rollout_corr.md` | Full doc | Usage guide |
| `verl/docs/algo/rollout_corr_math.md` | Full doc | Mathematical formulations |

---

## Part III: ★★★★★ RTX 4090 GRPO — Combined Impact Analysis

### 1. ★★★★★★★ ReMax + IcePop Combination for RTX 4090

★★★★★★★★★ These two algorithms address DIFFERENT aspects of GRPO training:

- ReMax: Better **advantage estimation** (greedy baseline vs group baseline)
- IcePop: Better **off-policy correction** (IS weight bounds vs no correction)

★★★★★★★ Can they be combined? YES:

```yaml
algorithm:
  adv_estimator: remax               # ReMax greedy baseline
  use_kl_in_reward: false            # MUST for ReMax consistency
  rollout_correction:
    rollout_is: token                 # IcePop IS weights
    rollout_is_threshold: "0.5_5.0"  # IcePop bounds
    rollout_rs: null
    bypass_mode: true                 # 2-policy mode
    loss_type: reinforce              # REINFORCE + explicit IS weights
```

★★★★★★★ But there's a subtlety: In bypass_mode + ppo_clip, IcePop IS weights are computed for **metrics only** — they are NOT applied to the loss (PPO ratio already handles IS). For IcePop to actually affect gradients, you need `loss_type: "reinforce"`.

★★★★★★★★★ RTX 4090 recommended configurations:

**Option A: ReMax + bypass_ppo_clip (simplest, recommended for beginners)**
- ReMax provides low-variance advantage (greedy baseline)
- bypass_mode skips ref model (saves 14GB)
- PPO-clip provides trust region via ratio clipping
- ★★★★★★★★ SIMPLEST RTX 4090 config — zero extra memory, zero extra computation

**Option B: ReMax + bypass_pg_icepop (most precise gradient)**
- ReMax advantage + IcePop IS correction + REINFORCE loss
- IcePop zeros out-of-range IS weights for exact population
- More precise gradient estimates than PPO-clip
- ★★★★★★★★ PRECISE but requires IS weight computation overhead (~1-3%)

**Option C: GRPO + bypass_ppo_clip (standard baseline)**
- Standard GRPO with group baseline
- bypass_mode for memory savings
- ★★★★★ WORKS but higher variance than ReMax

### 2. ★★★★★★★★ RTX 4090 Memory Budget Analysis

| Config | Models | Memory | Feasible |
|--------|--------|--------|----------|
| ReMax + bypass_ppo_clip | 1 (actor only) | ~18GB | ★★★★★★★★ YES |
| ReMax + bypass_pg_icepop | 1 (actor only) | ~18GB + ~1% IS overhead | ★★★★★★★★ YES |
| ReMax + use_kl_in_reward=True | 2 (actor + ref) | ~32GB | ★★★★✗ NO (24GB limit) |
| GRPO + bypass_ppo_clip | 1 (actor only) | ~18GB | ★★★★★★★★ YES |
| CPPO + bypass_mode | 1 (actor only) | ~18GB | ★★★★★★★★ YES |

★★★★★★★★★ ReMax + bypass_mode = 1 model (actor only) = ~18GB = RTX 4090 feasible
★★★★★★★★★ ReMax without bypass = 2 models (actor + ref) = ~32GB = RTX 4090 NOT feasible

### 3. ★★★★★★★★ RTX 4090 GRPO Algorithm Ranking (Updated 2026-06-16)

| Rank | Algorithm | Advantage | Trust Region | Memory | Variance | Simplicity |
|------|-----------|-----------|-------------|--------|----------|------------|
| ★★★★★ #1 | rLLM Tinker | auto-safe | bypass default | ~18GB | ★★★★★ | ★★★★★★★★ |
| ★★★★★ #2 | verl ReMax+bypass | greedy baseline | PPO-clip | ~18GB | ★★★★★★★★ | ★★★★★★ |
| ★★★★★ #2.5 | verl ReMax+IcePop | greedy baseline | IcePop bounds | ~18GB | ★★★★★★★★ | ★★★★★ |
| ★★★★★ #2.5 | verl CPPO+bypass | GRPO group | prefix-weighted | ~18GB | ★★★★ | ★★★★ |
| ★★★★ #3 | verl GRPO+bypass | group mean | PPO-clip | ~18GB | ★★★ | ★★★★★★★★ |

★★★★★★★★★ Updated insight: ReMax+IcePop provides BOTH best advantage (greedy baseline) AND precise IS correction (IcePop bounds). This combination is theoretically stronger than ReMax alone, but requires `loss_type: "reinforce"` which is less stable than PPO-clip for beginners.

★★★★★★★★★ RTX 4090 practical recommendation:
- **Beginners**: rLLM Tinker or verl GRPO+bypass_ppo_clip → simplest → get started fast
- **Math/reasoning**: verl ReMax+bypass_ppo_clip → greedy baseline → lowest variance → GSM8k 97 vs 89
- **Maximum precision**: verl ReMax+IcePop+bypass_pg → greedy + IS correction → theoretically strongest
- **Long CoT (4k+)**: verl CPPO+bypass → prefix-weighted trust region → prevents drift

## References
- PR #6340: ReMax algorithm (merged) — https://github.com/verl-project/verl/pull/6340
- PR #5722: IcePop IS correction (merged) — https://github.com/verl-project/verl/pull/5722
- ReMax paper: https://arxiv.org/abs/2310.10505
- IcePop/Rollout Correction paper: https://arxiv.org/abs/2512.23075 (Trust Region Masking)
- Blog series: https://richardli.xyz/rl-collapse
- Source files analyzed:
  - `verl/trainer/ppo/core_algos.py` (lines 732-765, 2351-2487)
  - `verl/trainer/ppo/rollout_corr_helper.py` (1100+ lines)
  - `verl/trainer/main_ppo_sync.py` (lines 109-113, 1212-1255, 1688-1720)
  - `verl/trainer/ppo/ray_trainer.py` (lines 76-115, 185-280, 1450-1494)
  - `verl/trainer/config/algorithm.py` (lines 60-212, 370-389)
  - `verl/docs/algo/rollout_corr.md`
  - `verl/docs/algo/rollout_corr_math.md`
- Related notes: verl-cppo-algorithm-reading.md, verl-v080-latest-developments-2026-06-reading.md, rtx4090-grpo-trust-region-comparison.md, rtx4090-verl-cppo-grpo-training-guide.md
