# verl CPPO Async Trainer Incompatibility — Source-Level Analysis

> 2026-06-18 | Complete source-level proof of CPPO + async trainer incompatibility
> ★★★★★★★★ `apply_bypass_mode()` at rollout_corr_helper.py:1137 UNCONDITIONALLY overrides `loss_mode` → destroys CPPO mask!
> ★★★★★★★★ V1 TransferQueue trainers use `_compute_old_log_prob()` → data-only substitution → CPPO-compatible!
> ★★★★★★★★ Legacy Ray trainers use `apply_bypass_mode()` → data substitution + override → CPPO-INCOMPATIBLE!

---

## 1. The Exact Incompatibility: One Line Destroys CPPO

```
★★★★★★★★★ rollout_corr_helper.py:1102-1137 — apply_bypass_mode():

Line 1137 is THE problem:
  with open_dict(policy_loss_config):
      policy_loss_config["rollout_correction"] = rollout_corr_config
      policy_loss_config["loss_mode"] = "bypass_mode"   # ← DESTROYS CPPO!

★★★★★★★★★ What this line does:
  → Sets loss_mode = "bypass_mode" → UNCONDITIONALLY replaces "cppo"
  → Dispatcher routes to compute_policy_loss_bypass_mode() → PPO-clip/REINFORCE
  → CPPO's position-weighted cumulative-prefix-divergence mask NEVER runs!
  → CPPO divergence measurement D_t = |π - μ| → NEVER computed!
  → Dynamic threshold c_t = min(δ, δ + δ_b*W_{t-1} - S_{t-1}) → NEVER applied!

★★★★★★★★★ Why bypass_mode data substitution IS required for CPPO:
  → CPPO measures divergence against rollout policy μ → old_log_prob MUST = rollout_log_prob
  → If bypass not active → old_log_prob = pi_old (separate forward) → NOT μ → wrong divergence
  → Three failure modes without bypass:
    1. Budget calibration wrong: quantile(D_t, 0.9) uses |pi-pi_old| instead of |pi-μ|
    2. toward_mu clause wrong: A*(ρ_t-1)<=0 checks pi/pi_old instead of pi/μ
    3. Position weighting wrong: designed for divergence from μ, not pi_old
  → ★★★★★★★★ CPPO NEEDS bypass_mode data → but NOT bypass_mode loss_mode!
```

---

## 2. Trainer Compatibility Matrix

```
★★★★★★★★★ 5 trainers, 2 paths, different compatibility:

| Trainer | Path | bypass handling | loss_mode handling | CPPO Compatible? |
|---------|------|-----------------|--------------------|--------------------|
| Legacy ray_trainer.py | apply_bypass_mode() | Data+override | Overrides to "bypass_mode" | NO |
| Experimental separation/ray_trainer.py | apply_bypass_mode() | Data+override | Overrides to "bypass_mode" | NO |
| V1 trainer_sync.py | _compute_old_log_prob() | Data-only | Leaves untouched | YES |
| V1 trainer_colocate_async.py | _compute_old_log_prob() | Data-only | Leaves untouched | YES |
| V1 trainer_separate_async.py | _compute_old_log_prob() | Data-only+forced bypass=True | Leaves untouched | YES |

★★★★★★★★★ Key distinction:
  → apply_bypass_mode() = data substitution + loss_mode override → TWO operations
  → _compute_old_log_prob() = data substitution ONLY → ONE operation → loss_mode untouched
  → CPPO needs data substitution but NOT loss_mode override
  → → V1 trainers CORRECTLY separate the two operations → CPPO-compatible!
  → → Legacy trainers COMBINE the two operations → CPPO-incompatible!

★★★★★★★★★ V1 _compute_old_log_prob() (trainer_base.py:1165-1179):
  def _compute_old_log_prob(self, batch, metrics):
      bypass_recomputing_logprobs = rollout_corr_config.get("bypass_mode", False)
      if bypass_recomputing_logprobs:
          data = tq.kv_batch_get(keys=batch.keys, ...)
          data["old_log_probs"] = data.pop("rollout_log_probs")  # Data swap ONLY!
          tq.kv_batch_put(keys=batch.keys, fields=data)
          return batch
  # → NO loss_mode override → CPPO can set loss_mode="cppo" → works!

★★★★★★★★★ separate_async trainer forces bypass_mode=True (line 60-61):
  self.config.algorithm.rollout_correction.bypass_mode = True
  # → BUT uses _compute_old_log_prob() → only data substitution → CPPO still works!
```

---

## 3. Policy Loss Dispatcher Chain

```
★★★★★★★★★ losses.py:101-103 — dispatcher:
  loss_mode = config.policy_loss.get("loss_mode", "vanilla")
  policy_loss_fn = get_policy_loss_fn(loss_mode)
  pg_loss, pg_metrics = policy_loss_fn(...)

★★★★★★★★★ core_algos.py:50-85 — registry:
  POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
  → "vanilla" → standard PPO/GRPO
  → "bypass_mode" → PPO-clip/REINFORCE with IS weights (line 2351)
  → "cppo" → position-weighted cumulative prefix divergence (#6731, OPEN)

★★★★★★★★★ When loss_mode="bypass_mode" overrides "cppo":
  → Dispatcher routes to compute_policy_loss_bypass_mode()
  → This computes IS weights + rejection mask → PPO/REINFORCE logic
  → CPPO mask construction NEVER happens → divergence budget NOT enforced
  → → TRAINING LOSES THE CPPO ADVANTAGE → same as vanilla GRPO!
```

---

## 4. RTX 4090 Recommendations

```
★★★★★★★★★ MUST use sync TransferQueue trainer with CPPO:
  → verl.trainer.main_ppo_sync → V1 sync trainer → CPPO-compatible
  → MUST NOT use legacy ray_trainer.py → CPPO-incompatible
  → MUST NOT use experimental separation/ray_trainer.py → CPPO-incompatible

★★★★★★★★★ Config:
  algorithm:
    rollout_correction:
      bypass_mode: true    # MUST: data substitution for CPPO divergence
    type: cppo             # MUST: CPPO loss mode
  actor:
    policy_loss:
      loss_mode: cppo      # MUST: will be preserved by V1 trainer
      cppo_w_min: 0.8       # position weight minimum
      cppo_delta_b: 0.02    # cumulative prefix budget

★★★★★★★★★ Why this works:
  → V1 sync trainer calls _compute_old_log_prob()
  → _compute_old_log_prob() swaps old_log_probs = rollout_log_probs (data)
  → V1 trainer DOES NOT touch loss_mode → "cppo" preserved
  → Dispatcher routes to compute_policy_loss_cppo()
  → CPPO mask correctly constructed → divergence budget enforced
  → → RTX 4090 GRPO training gets BEST trust region!
```

---

## Key Findings Summary

★★★★★★★★★ apply_bypass_mode() line 1137 overrides loss_mode → destroys CPPO mask
★★★★★★★★★ V1 TransferQueue trainers use _compute_old_log_prob() → data-only → CPPO-compatible
★★★★★★★★★ Legacy Ray trainers use apply_bypass_mode() → data+override → CPPO-incompatible
★★★★★★★★★ CPPO NEEDS bypass_mode data substitution but NOT loss_mode override
★★★★★★★★★ RTX 4090: MUST use sync TransferQueue trainer with CPPO+bypass_mode
★★★★★★★★★ #6731 explicitly documents sync trainer only → PR author knows this limitation

---

## References

- rollout_corr_helper.py:1102-1137 — apply_bypass_mode() with loss_mode override
- ray_trainer.py:1528-1537 — legacy trainer calling apply_bypass_mode()
- trainer_base.py:1165-1179 — V1 _compute_old_log_prob() data-only substitution
- trainer_separate_async.py:60-61 — forced bypass_mode=True
- losses.py:101-103 — loss_mode dispatcher
- core_algos.py:50-85 — POLICY_LOSS_REGISTRY
- CPPO algorithm: notebook/projects/verl-cppo-algorithm-reading.md
