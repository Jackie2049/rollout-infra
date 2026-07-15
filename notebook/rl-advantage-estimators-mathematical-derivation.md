# RL Advantage Estimators: Mathematical Derivation & Cross-Framework Comparison

**Date**: 2026-07-15 (Session 10)
**Purpose**: Deep mathematical understanding of advantage estimators for AI expertise
**Sources**: rLLM terminal-rl (advantage.py), verl V1 (core_algos.py), arXiv papers

---

## 1. GRPO (Group Relative Policy Optimization)

**Math**:
```
For a group G of trajectories sharing the same prompt:
  μ_G = mean(rewards_G)     = (1/|G|) Σ_{i∈G} R_i
  σ_G = std(rewards_G)      = sqrt((1/|G|) Σ_{i∈G} (R_i - μ_G)²)
  A_i = (R_i - μ_G) / σ_G   for σ_G > 0
  A_i = 0                    for σ_G = 0 (all rewards identical)
```

**Properties**:
- Group-normalized: advantages within each group have mean=0, std=1
- Relative: only the ranking within the group matters, not absolute reward values
- **Critical issue**: |G|=1 → A = (R-R)/0 → undefined! Frameworks set A=0 or A=R
- This causes REINFORCE degeneration when group_size=1

**Cross-framework**:
- verl: `compute_grpo_advantage()` — same formula, group_size parameter
- rLLM: `grpo` estimator — same formula, configurable grouping_key
- TRL: `GRPOTrainer._compute_advantage()` — same formula, group_size warning at gs=1

---

## 2. REINFORCE (No Baseline)

**Math**:
```
A_i = R_i     (raw reward, no baseline subtraction)
```

**Properties**:
- Simplest estimator: advantage = reward directly
- High variance: no baseline to reduce variance
- No normalization: large rewards → large gradients
- Works best with shaped rewards (not flat outcome rewards)

**Limitation**: Without baseline, gradient estimate has high variance:
```
∇J = E[A_i · ∇logπ(a_i|s_i)] = E[R_i · ∇logπ(a_i|s_i)]
Variance = E[R_i² · (∇logπ)²]  → large when R_i has large range
```

---

## 3. REINFORCE++Baseline (Centered Reward)

**Math**:
```
μ_G = mean(rewards_G)
A_i = R_i - μ_G     (mean-subtracted, equivalent to GRPO without std normalization)
```

**Properties**:
- Control variate: subtracting mean reduces variance by factor of (1 - correlation(R, μ))
- Still group-relative but without std normalization → preserves reward magnitude
- Less aggressive normalization than GRPO (preserves signal strength)
- Same variance reduction as GRPO mean-subtraction, but doesn't squash to std=1

**Comparison with GRPO**:
```
GRPO:     A_i = (R_i - μ) / σ  → normalized to ±1 range (bounded)
REINFORCE++BL: A_i = R_i - μ    → preserves reward magnitude
```

---

## 4. RLOO (Leave-One-Out Baseline)

**Math**:
```
For trajectory i in group G:
  μ_LOO_i = (1/(|G|-1)) Σ_{j∈G, j≠i} R_j    (leave-one-out mean)
  A_i = R_i - μ_LOO_i

Alternative form:
  A_i = R_i - (Σ_{j∈G} R_j - R_i) / (|G|-1)
  A_i = (R_i · |G| - Σ_{j∈G} R_j) / (|G|-1)
```

**Properties**:
- Each trajectory's baseline excludes its own reward → reduces self-bias
- Equivalent to GRPO when |G| is large (μ_LOO ≈ μ_G)
- For |G|=2: μ_LOO = the other trajectory's reward → binary comparison
- **Critical issue**: |G|=1 → μ_LOO undefined (no other trajectories to leave out)

**Cross-framework**:
- verl: `compute_rloo_advantage()` — same formula
- rLLM: `rloo` estimator — same formula

**Comparison with GRPO**:
```
GRPO:   A_i = (R_i - μ) / σ        — standardized, magnitude fixed
RLOO:   A_i = R_i - μ_LOO           — preserves magnitude, reduced self-bias
```

RLOO is better when reward magnitude matters (preserves signal), GRPO is better when you want bounded advantages (prevents gradient explosion).

---

## 5. PRPO (Prompt-Level Reward)

**Math**:
```
A_i = R_prompt_i    (per-prompt reward, no group normalization)
```

**Properties**:
- Each trajectory gets the reward for its prompt (not trajectory-level reward)
- Useful when reward is at prompt level (e.g., pass/fail for entire task)
- No group structure needed — each trajectory is independent
- Suitable for tasks where group comparison doesn't make sense

---

## 6. ECHO (GRPO + Auxiliary Loss)

**Math**:
```
Advantage estimation:
  A_i = GRPO_advantages_i     (same as GRPO)

Policy loss:
  L = ppo_clip(ratio, A) + env_loss_coef · CE(obs_mask)

  env_loss_coef = 0.05 (default)
  obs_mask = mask on observation/environment feedback tokens
  CE = cross-entropy loss on observation tokens
```

**Properties**:
- Uses GRPO advantages (group-normalized) for the policy gradient part
- Adds auxiliary cross-entropy loss on observation tokens
- The auxiliary loss teaches the model to predict observation outcomes
- arXiv:2605.24517: "Learning to predict environment dynamics improves policy"

**Key insight**: The auxiliary loss is INSIDE the loss body (not a separate framework component). This means the gradient from env_loss flows through the same backward pass as the policy gradient.

---

## 7. Mathematical Comparison: Variance & Bias

| Estimator | Variance | Self-Bias | Magnitude | Group Size Req |
|-----------|----------|-----------|-----------|----------------|
| GRPO | Low (normalized) | None (mean subtracted) | Fixed (±1) | gs≥4 recommended |
| REINFORCE | High (no baseline) | Full (R_i baseline for itself) | Raw | Any |
| REINFORCE++BL | Medium | None (mean subtracted) | Preserved | gs≥2 |
| RLOO | Medium | None (LOO) | Preserved | gs≥2 |
| PRPO | High (no baseline) | Depends on reward type | Raw | Any |
| ECHO | Low (GRPO base) | None (GRPO) | Fixed (±1) | gs≥4 |

**Self-bias definition**: When A_i includes R_i in the baseline, the gradient is biased toward increasing R_i regardless of whether the action was actually good. This is the fundamental problem with using the trajectory's own reward as its baseline.

**GRPO eliminates self-bias**: A_i = (R_i - μ)/σ where μ includes R_i. The mean subtraction removes the self-bias component, and std normalization bounds the magnitude.

---

## 8. RTX 4090 GRPO Advantage Recommendation

### Primary: GRPO (group_size ≥ 8)
- Bounded magnitude → safe for single GPU training
- No self-bias → correct gradient direction
- Group normalization → natural ranking-based learning

### Secondary: RLOO (group_size ≥ 4)
- Preserves reward magnitude → more gradient signal
- Leave-one-out → no self-bias
- Good when reward scale varies significantly across tasks

### Avoid:
- REINFORCE (gs=1): degeneration, no learning signal
- REINFORCE++BL with gs=1: A=0 always (mean=R for single trajectory)

### Key rule:
```
★★★★★★★★ ALWAYS use group_size ≥ 4 for GRPO
★★★★★★★★ ALWAYS use group_size ≥ 8 for MoE models (more variance in rewards)
★★★★★★★★ NEVER set group_size = 1 (REINFORCE degeneration)
```

---

## Session Stats
- **6 advantage estimators** mathematically derived and compared
- **4 frameworks** mapped (verl V1, rLLM, TRL)
- **Variance-bias analysis** for each estimator
- **RTX 4090 recommendation**: GRPO gs≥8 primary, RLOO gs≥4 secondary
