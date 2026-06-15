# verl v0.8 Registered Policy Loss Catalog — Source-Level

> 2026-06-16 | verl/verl/trainer/ppo/core_algos.py | 11 registered policy loss types
> Focus: Each loss type's semantics, RTX 4090 viability, bypass compatibility
> ★★★★★★★★ bypass_mode dispatches to ppo_clip or reinforce — RTX 4090 critical!

---

## 1. Policy Loss Registry Overview

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```python
@register_policy_loss("vanilla")       → line 1278  — Standard PPO clipped objective
@register_policy_loss("dppo_tv")       → line 1372  — DPPO Binary-TV divergence threshold
@register_policy_loss("dppo_kl")       → line 1453  — DPPO KL divergence threshold
@register_policy_loss("gspo")          → line 1538  — GSPO geometric policy optimization
@register_policy_loss("sapo")          → line 1614  — SAPO softmax policy optimization
@register_policy_loss("gpg")           → line 1699  — Generalized policy gradient
@register_policy_loss("clip_cov")      → line 1735  — Clipped covariance policy loss
@register_policy_loss("kl_cov")        → line 1840  — KL covariance policy loss
@register_policy_loss("geo_mean")      → line 1920  — Geometric mean policy loss
@register_policy_loss("cispo")         → line 2006  — CISPO (IcePop) IS correction
@register_policy_loss("bypass_mode")   → line 2351  — Bypass mode (RTX 4090 critical!)
```

★★★★★★★★★ CPPO #6731 (OPEN, not merged) → would be 12th registered loss type

---

## 2. ★★★★★★★★ Bypass Mode Deep Analysis (lines 2351-2487)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```python
@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob,  # = rollout_log_prob (NOT pi_old from separate forward!)
    log_prob,      # current policy π_θ
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
):
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ KEY: bypass_mode is NOT a separate loss function — it's a DISPATCHER!

It computes IS weights and rejection mask, then dispatches to:
- loss_type="ppo_clip" → compute_policy_loss_vanilla (standard PPO-clip, NO IS weights)
- loss_type="reinforce" → compute_policy_loss_reinforce (REINFORCE with explicit IS weights)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Config options (in config.policy_loss.rollout_correction):

```
loss_type:            "ppo_clip" (default) or "reinforce"
rollout_is:           "token", "sequence", or None
rollout_is_threshold: Upper IS threshold (default: 2.0)
rollout_rs:           Rejection sampling mode
rollout_rs_threshold: RS threshold specification
rollout_is_batch_normalize: Normalize IS weights to mean=1.0
```

### ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ CRITICAL CORRECTNESS: PPO-clip bypass DOES NOT apply IS weights!

In bypass_mode with loss_type="ppo_clip":
  → ratio r = π_current / π_rollout (via old_log_prob=rollout_log_prob)
  → PPO-clip already constrains this ratio → clipping = implicit IS bound
  → Applying additional IS weights = DOUBLE-COUNTING → INCORRECT!

In bypass_mode with loss_type="reinforce":
  → IS weights w = π_current / π_rollout (computed explicitly)
  → REINFORCE: L = -E[w * log π(a|s) * A] → IS weights applied correctly
  → This is the correct way to use IS with REINFORCE

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. RTX 4090 Loss Type Recommendations

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Loss Type | Bypass Compat? | RTX 4090 Viable? | Memory Impact | Recommendation |
|-----------|---------------|-----------------|---------------|---------------|
| vanilla (PPO-clip) | ✓ (as bypass dispatch) | ✓✓✓ | +0 (bypass) | ★★★★★★★ Standard safe |
| dppo_tv | ✓ (needs config) | ✓✓✓ | +0 (bypass) | ★★★★★ Principled divergence |
| dppo_kl | ✓ (needs config) | ✓✓✓ | +0 (bypass) | ★★★★★ KL-based divergence |
| gspo | ✓? | ✓✓✓ | +0 | ★★★ Experimental |
| sapo | ✓? | ✓✓✓ | +0 | ★★★ Experimental |
| gpg | ✓? | ✓✓✓ | +0 | ★★★ Basic |
| clip_cov | ✓? | ✓✓✓ | +0 | ★★★★ Covariance-aware |
| kl_cov | ✓? | ✓✓✓ | +0 | ★★★★ KL covariance |
| geo_mean | ✓? | ✓✓✓ | +0 | ★★★ Experimental |
| cispo (IcePop) | ✓ (bypass_pg_token_icepop preset) | ✓✓✓ | +0 | ★★★★★★ Most precise IS |
| bypass_mode | ✓ (it IS bypass) | ✓✓✓✓✓✓✓ | -14GB (ref model!) | ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 MUST! |
| cppo (OPEN #6731) | ✓✓✓ MUST use bypass | ✓✓✓✓✓✓✓ | +0 (bypass) | ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 #2 algorithm! |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ RTX 4090 BEST combinations:
1. bypass_mode + loss_type="ppo_clip" → standard safe PPO → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ simplest
2. bypass_mode + loss_type="reinforce" + cispo → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ most precise
3. bypass_mode + CPPO (when merged) → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ best bound
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. CISPO (IcePop) Loss Analysis (lines 2006-2270)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

CISPO = Clipped IS Policy Optimization = IcePop (#5722)

```python
@register_policy_loss("cispo")
def compute_policy_loss_cispo(...):
    # Token-level IS correction with exact population bounds
    # threshold: "0.5_5.0" = IcePop (lower, upper bounds)
    #            "2.0" = TIS (upper bound only)

    # IcePop: zeros out-of-range IS weights (not clips!)
    # torch.where(token_kept_mask, weight, 0) → hard boundary

    # Exact population: keeps only weights in [0.5, 5.0]
    # → precise gradient estimation → no variance from outlier weights
```

★★★★★★★★★ IcePop vs TIS:
  → TIS: clips weights > threshold → still includes low weights → biased
  → IcePop: zeros both low AND high outliers → exact population → unbiased!
  → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ IcePop + bypass_pg_token_icepop preset = most precise gradient on RTX 4090!

---

## 5. DPPO Variants (dppo_tv, dppo_kl)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```python
@register_policy_loss("dppo_tv")  → Binary-TV divergence threshold
@register_policy_loss("dppo_kl")  → KL divergence threshold
```

Both replace heuristic PPO clipping with principled divergence constraints:
- dppo_tv: D_t = |π(y_t|s_t) - μ(y_t|s_t)| → Binary-TV per-token divergence → threshold δ
- dppo_kl: KL divergence between π_θ and reference policy → threshold δ

★★★★★★★★★ DPPO = principled trust region → but still pointwise/uniform → CPPO extends to position-weighted cumulative

---

## 6. Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ bypass_mode is NOT a separate loss — it's a DISPATCHER → ppo_clip or reinforce → config determines which!

2. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ PPO-clip bypass = NO IS weights (clipping handles ratio) → REINFORCE bypass = explicit IS weights → DIFFERENT semantics → choose carefully!

3. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ 11 registered loss types → rich catalog → RTX 4090 choice = bypass_mode dispatcher + appropriate loss_type!

4. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ CISPO (IcePop) = exact population [0.5, 5.0] → zeros outliers → most precise IS → bypass_pg_token_icepop preset!

5. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ CPPO (OPEN #6731) = 12th type when merged → position-weighted cumulative prefix divergence → MUST use bypass → RTX 4090 #2 algorithm!
