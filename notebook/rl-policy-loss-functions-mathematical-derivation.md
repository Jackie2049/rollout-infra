# RL Policy Loss Functions: Mathematical Derivation & Cross-Framework Comparison

**Date**: 2026-07-15 (Session 10)
**Purpose**: Deep understanding of 9 policy loss functions for AI expertise
**Sources**: rLLM terminal-rl (loss.py), verl V1 (core_algos.py), arXiv papers

---

## 1. PPO-Clip (Standard)

**Formula**:
```
L = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
```
where `ratio = exp(logp - logp_old)`, `A = advantages`, `ε = clip_ratio`

**Properties**:
- Upper clip prevents excessive policy change for positive advantages
- Lower clip prevents excessive policy change for negative advantages
- Both clips create trust region around old policy
- Default: ε=0.2, ratio ∈ [0.8, 1.2] for clipping to activate

**Limitation**: Upper clip can suppress large positive advantages → conservative for GRPO

---

## 2. UP-GRPO (arXiv:2607.06987)

**Formula**:
```
A ≥ 0: L = ratio·A               (NO upper clip)
A < 0: L = max(ratio·A, clip(ratio, 1-ε, 1+ε)·A, clip_ratio_c·A)
```
where `clip_ratio_c` is dual-clip upper bound (default 3.0)

**Properties**:
- Positive advantages: ratio unbounded → policy can increase freely
- Negative advantages: standard PPO-clip + extra clip at ratio_c
- clip_ratio_c prevents ratio from going too high even for negative A
- **Key insight**: upper clip in PPO-clip is ALWAYS harmful for positive advantages (it limits learning signal)

**Why it's better for GRPO**: Group-normalized advantages often have strong positive signals — upper clip in vanilla PPO-clip suppresses these

---

## 3. DPPO-TV (arXiv:2602.04879)

**Formula**:
```
L = ratio·A · mask(d_TV(π, π_old) ≤ δ)
```
where `d_TV` is binary total variation divergence, `δ` is trust region threshold

**Properties**:
- Uses divergence mask instead of clipping
- C=∞ ratio (no truncation on ratio)
- Binary mask: inside trust region → gradient flows, outside → zero gradient
- More principled than PPO-clip (explicit trust region, not heuristic clipping)

**Limitation**: Binary mask is harsh — either full gradient or zero → can oscillate at boundary

---

## 4. DPPO-KL (arXiv:2602.04879)

**Formula**:
```
L = ratio·A · mask(KL(π_old, π) ≤ δ)
```
Same structure as DPPO-TV but with KL divergence instead of total variation

**Properties**:
- KL divergence is smoother than TV → less harsh boundary effects
- Still uses binary mask (not soft KL penalty)
- C=∞ ratio within trust region

---

## 5. CISPO (arXiv:2506.13585)

**Formula**:
```
L = clamp(ratio).detach() · A · logp_curr
```

**Properties**:
- `clamp(ratio)` = [1-ε, 1+ε] → the ratio is clamped but **detached** (stop-gradient)
- Gradient flows through `logp_curr` (the current policy's log-probability)
- ALL tokens keep gradient (no zero-gradient from clipping)
- Key difference from PPO-clip: PPO-clip zeros gradient for out-of-range ratios; CISPO always has gradient flowing through logp_curr
- The detached clamped ratio acts as a **weight** on the gradient, not a gate

**Why it's better**: Better gradient utilization — no tokens are completely silenced by clipping

---

## 6. GSPO (arXiv:2507.18071)

**Formula**:
```
s_i = (π(y_i) / π_old(y_i))^(1/|y_i|)    (sequence-level ratio)
L = min(s_i·A_i, clip(s_i, 1-ε, 1+ε)·A_i)
aggregation: seq-mean-token-mean (FORCED)
```

**Properties**:
- Sequence-level ratio: normalized by sequence length
- `|y_i|` = number of tokens in sequence
- Addresses PPO's bias toward longer sequences (longer sequences accumulate larger ratios)
- FORCES seq-mean-token-mean aggregation (every sequence equal weight)
- Token-level ratio → sequence-level ratio: `exp(sum(logp - logp_old) / |y|)`

**Why it's better**: Length-normalized ratio prevents long sequences from dominating

---

## 7. IcePop (arXiv:2510.18855)

**Formula**:
```
L = ratio·A · mask(ratio ∈ [alpha, beta])
```
where `alpha` and `beta` define the importance sampling band

**Properties**:
- Double-sided importance sampling band
- `alpha` = lower bound (typically < 1-ε)
- `beta` = upper bound (typically > 1+ε)
- Requires `logp_rollout` (true inference/sampling log-probs, not training log-probs)
- Masks tokens with out-of-range ratios → zero gradient outside band

**Limitation**: Requires rollout log-probs (not always available in training frameworks)

---

## 8. REINFORCE (No Ratio)

**Formula**:
```
L = -A · logp_curr
```

**Properties**:
- No ratio, no clipping
- Simplest policy gradient: directly weight log-probability by advantage
- Equivalent to PPO-clip with ε=∞ (no trust region)
- High variance without baseline

---

## 9. REINFORCE-KL (IS-weighted)

**Formula**:
```
L = -IS_weight · A · logp_curr + fwd_KL + bwd_KL
```
where `IS_weight` = importance sampling ratio from rollout policy

**Properties**:
- Requires `logp_rollout` (true sampling log-probs)
- IS-weighted: corrects for distribution shift between rollout and training
- fwd_KL: KL(π_curr, π_old) → encourages coverage
- bwd_KL: KL(π_old, π_curr) → prevents divergence
- Perfect pair with bypass_mode (no proximal forward needed)

---

## 10. ECHO (arXiv:2605.24517)

**Formula**:
```
L = ppo_clip_loss + env_loss_coef · CE(obs_mask)
```
where `env_loss_coef` = 0.05 (default), `obs_mask` = observation token mask

**Properties**:
- Uses GRPO advantages (from @register_adv_estimator("echo"))
- Auxiliary loss: cross-entropy on observation (environment feedback) tokens
- Env loss is INSIDE the loss body (no separate framework needed)
- Default pairing: estimator="echo" → loss="echo"

---

## Cross-Framework Comparison: Loss Functions

| Loss | Ratio Clip | Gradient | Aggregation | Requires logp_rollout | Framework Support |
|------|-----------|----------|-------------|----------------------|-------------------|
| PPO-clip | dual [1-ε, 1+ε] | some tokens zeroed | any | No | ALL (verl, rLLM, TRL) |
| UP-GRPO | positive unbounded, negative dual-clip | positive always flows | any | No | verl (our PR), rLLM |
| DPPO-TV | C=∞ with mask | binary (0 or full) | any | No | rLLM |
| DPPO-KL | C=∞ with mask | binary (0 or full) | any | No | rLLM |
| CISPO | detached clamp | ALL tokens (always) | any | No | rLLM, verl |
| GSPO | sequence-level dual | some tokens zeroed | seq-mean-token-mean ONLY | No | rLLM |
| IcePop | band [alpha, beta] | outside band zero | any | Yes | rLLM |
| REINFORCE | None | all | any | No | rLLM |
| REINFORCE-KL | None + KL | all + KL terms | any | Yes | rLLM |
| ECHO | PPO-clip + env CE | some + env | any | No | rLLM |

---

## RTX 4090 GRPO Recommendations

1. **Best**: UP-GRPO (unbounded positive, dual-clip negative) — our PR #9 on Jackie2049/verl
2. **Alternative**: CISPO (all tokens keep gradient) — better gradient utilization
3. **Experimental**: GSPO (sequence-level ratio) — addresses length bias
4. **Avoid**: REINFORCE (no trust region), IcePop (needs rollout log-probs)

---

## Session Stats
- **10 loss functions** mathematically derived and compared
- **6 frameworks** mapped (verl V1, rLLM, TRL, with DPPO/CISPO/GSPO as rLLM-specific)
- **RTX 4090 recommendation**: UP-GRPO #1, CISPO #2
