# verl CPPO bypass_mode Source-Level Reading

> 2026-06-16 | PR #6731 (OPEN) | Paper: arXiv:2606.10968 | Authors: Tencent Hunyuan (Mao, Zhou, Tao, Ding, Shi, Lin, Wu, Zhu, Qiu, Zhu)
> Focus: CPPO divergence mechanism, bypass_mode dispatcher interaction, code path trace
> ★★★★★★★★ CPPO + bypass_mode = mathematically REQUIRED pairing -- not optional!
> ★★★★★★★★ CPPO divergence measured against rollout policy mu, NOT pi_old

---

## 1. CPPO Mathematical Formulation

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 1.1 Per-Token Divergence (Binary-TV)

```
D_t = |pi(y_t|s_t) - mu(y_t|s_t)|

where:
  pi = current policy pi_theta (being optimized)
  mu = ROLLOUT policy (data collection, vLLM/SGLang)
  y_t = sampled token at position t
  s_t = context state (prefix tokens)
```

★★★★★★★★★ KEY: D_t measures divergence against mu (rollout policy), NOT pi_old!

This is the fundamental difference from standard PPO/GRPO/DPPO:
- Standard PPO: ratio r_t = pi_theta / pi_old → divergence implicit in ratio
- DPPO: D_t = |pi_theta - pi_old| → divergence against pi_old
- CPPO: D_t = |pi_theta - mu| → divergence against rollout policy mu

★★★★★★★★★ When bypass_mode=True: old_log_prob = rollout_log_prob → pi_old = mu
  → CPPO reads mu directly from old_log_prob → divergence measurement CORRECT!

★★★★★★★★★ When bypass_mode=False: old_log_prob from separate forward → pi_old != mu
  → CPPO divergence = |pi_theta - pi_old| ≠ |pi_theta - mu| → INCORRECT!

### 1.2 Position Weight (Eq. 9)

```
w_t = w_min + (1 - w_min) * (T - t) / (T - 1)

Properties:
  - w_t ∈ [w_min, 1] → decreasing along response
  - Early tokens (small t): w_t ≈ 1 → tightly constrained
  - Late tokens (large t): w_t ≈ w_min → relaxed constraint
  - Default w_min = 0.8 → position weight ranges [0.8, 1.0]
  - T = padded response length (not valid length!)
```

★★★★★★★★★ Why early tokens need tighter constraint (Theorem 1):

Starting from finite-horizon performance-difference identity:
  J(pi') - J(pi) = sum_t E[ A_t * (pi'(y_t|s_t) - pi(y_t|s_t)) ]

An early-token shift at position t carries a remaining-horizon penalty for
ALL subsequent positions t+1, ..., T. Position-weighting is theoretically
optimal because it matches the structure of the performance-difference lemma.

### 1.3 Weighted Divergence and Prefix Sums

```
Z_t = w_t * D_t                              # weighted divergence

Prefix sums:
  S_t = Σ_{j<=t} Z_j                          # cumulative weighted divergence
  W_t = Σ_{j<=t} w_j                          # cumulative position weight

Initialization:
  S_0 = W_0 = 0 (one-token right-shift in implementation)

Interpretation:
  S_t = total "budget spent" by prefix up to position t
  W_t = total "budget allocated" by prefix up to position t
  S_t / W_t = average weighted divergence over prefix → should stay <= delta_b
```

### 1.4 Effective Threshold (Eq. 8)

```
c_t = min(delta, delta + delta_b * W_{t-1} - S_{t-1})

Properties:
  - c_t <= delta (always bounded by token-level threshold)
  - c_t shrinks when S_{t-1} > delta_b * W_{t-1} (prefix over-spent budget)
  - c_t = delta when prefix is "on budget" (S_{t-1} <= delta_b * W_{t-1})

★★★★★★★★★ Dynamic tightening mechanism:
  → As prefix diverges from mu → S_{t-1} grows → c_t shrinks
  → Later tokens face tighter effective threshold → prevents cascading drift!
  → GRPO/DPPO: ALL tokens face fixed threshold delta → misses budget overruns!
  → CPPO: catches "budget overruns" that all uniform methods miss!
```

### 1.5 Keep Decision (Eq. 10)

```
keep token t  iff  A_t * (rho_t - 1) <= 0  OR  Z_t <= c_t

Two clauses:
  1. A_t * (rho_t - 1) <= 0 → always keep "safe" updates
     → These move pi BACK toward mu → reducing divergence → always beneficial!
     → rho_t = pi_theta(y_t|s_t) / mu(y_t|s_t) → ratio vs rollout policy

  2. Z_t <= c_t → keep if within effective budget
     → Token-level weighted divergence within dynamic threshold
     → Budget only restricts terms that move pi FURTHER from mu

★★★★★★★★★ The first clause is critical:
  → When advantage A_t > 0 AND rho_t < 1 (current policy assigns less probability
    than rollout → update INCREASES probability toward rollout) → A_t*(rho_t-1) < 0 → KEEP
  → When advantage A_t < 0 AND rho_t > 1 (current policy assigns more probability
    than rollout → update DECREASES probability toward rollout) → A_t*(rho_t-1) < 0 → KEEP
  → These are the "toward-mu" updates → ALWAYS allowed → even if budget over-spent!
```

### 1.6 Per-Sequence Dynamic Budget Calibration (Eq. 22)

```
delta_b^seq = clamp(delta_b_k * quantile(D_t, delta_b_q), delta_b_min, 2 * delta_b_min)

Defaults: delta_b_q = 0.9, delta_b_k = 1.0 → P90 calibration

Properties:
  - Each sequence calibrates from its OWN divergence statistics → adaptive!
  - Higher-divergence sequences → proportionally larger budget (up to 2x floor)
  - delta_b_min = cppo_delta_b (default 0.02) → floor for budget
  - Empty sequences → NaN quantile → torch.nan_to_num → fallback to delta_b_min
```

★★★★★★★★★ Adaptive budget ensures:
  1. Low-divergence sequences → tight budget → more aggressive masking
  2. High-divergence sequences → generous budget → less aggressive masking
  3. Never below delta_b_min or above 2*delta_b_min → bounded range

---

## 2. ★★★★★★★★ Source-Level Code Trace

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 2.1 CPPO Registration and Entry Point

Source: PR #6731 diff, verl/trainer/ppo/core_algos.py (new function inserted BEFORE @register_policy_loss("gspo"))

```python
@register_policy_loss("cppo")
def compute_policy_loss_cppo(
    old_log_prob: torch.Tensor,       # Line: rollout mu log-probs (bypass_mode)
    log_prob: torch.Tensor,            # Line: current pi_theta log-probs
    advantages: torch.Tensor,          # Line: GRPO/GAE advantage estimates
    response_mask: torch.Tensor,       # Line: valid token mask
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
```

★★★★★★★★★ Parameter semantics:
  old_log_prob → In bypass_mode, this IS rollout_log_prob = mu log-probs
  log_prob → Current policy pi_theta (computed during forward pass)
  advantages → From GRPO outcome advantage or GAE
  config → Contains cppo_w_min, cppo_delta_b, cppo_delta_b_q, cppo_delta_b_k

### 2.2 Config Extraction

```python
pl = config.policy_loss
delta = float(config.clip_ratio)                        # Token-level threshold (reuses DPPO field)
w_min = float(pl.get("cppo_w_min", 0.8))               # Position weight floor
delta_b = float(pl.get("cppo_delta_b", 0.02))           # Prefix budget floor
delta_b_q = float(pl.get("cppo_delta_b_q", 0.9))       # Quantile for budget calibration
delta_b_k = float(pl.get("cppo_delta_b_k", 1.0))       # Scale for budget calibration
```

★★★★★★★★★ PolicyLossConfig new fields (verl/workers/config/actor.py +12 lines):
  cppo_w_min: float = 0.8      → position weight floor
  cppo_delta_b: float = 0.02   → prefix budget floor
  cppo_delta_b_q: float = 0.9  → quantile (P90)
  cppo_delta_b_k: float = 1.0  → scale factor

★★★★★★★★★ delta reuses clip_ratio → same convention as DPPO → no new field needed!
  Default: 0.20 for MoE models, 0.15 for dense models

### 2.3 Ratio and KL Computation

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)                    # rho_t = pi_theta / mu (bypass)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

clip_ratio_c = config.get("clip_ratio_c", 20.0)
truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)
truncated_ratio = truncated_ratio.detach()               # stop-gradient on TIS ratio
```

★★★★★★★★★ Same pattern as DPPO: truncated importance sampling instead of dual-clip PPO
  → truncated_ratio = clamp(ratio, max=clip_ratio_c) → max ratio = 20.0
  → .detach() → stop-gradient → truncated_ratio treated as constant coefficient

★★★★★★★★★ In bypass_mode: ratio = pi_theta / mu (not pi_theta / pi_old)
  → This IS the importance sampling ratio from rollout policy → correct for CPPO!

### 2.4 ★★★★★★★★ Mask Construction (Core Algorithm)

```python
with torch.no_grad():                                    # Mask is trust-region gate, NOT loss!
    response_mask_f = response_mask.float()

    # Binary-TV per-token divergence
    prob = torch.exp(log_prob)                            # pi_theta(y_t|s_t)
    old_prob = torch.exp(old_log_prob)                    # mu(y_t|s_t) (rollout!)
    D_t = (prob - old_prob).abs() * response_mask_f      # D_t = |pi - mu|

    # Position weight
    bs, resp_len = response_mask_f.shape
    T_fixed = float(resp_len)                             # Padded length (not valid!)
    pos = torch.arange(1, resp_len + 1, device=log_prob.device).unsqueeze(0).expand(bs, -1).float()
    frac = ((T_fixed - pos) / max(T_fixed - 1.0, 1.0)).clamp(0.0, 1.0)
    w_t = (w_min + (1.0 - w_min) * frac) * response_mask_f

    Z_t = w_t * D_t                                       # Weighted divergence
```

★★★★★★★★★ Implementation details:
  1. torch.no_grad() → mask is NOT part of loss → trust-region gate only
  2. T_fixed = padded response length → not valid length → matches reference CPPO
  3. Position schedule keyed on fixed padded length → right-padding zeroed by mask
  4. D_t = |exp(log_prob) - exp(old_log_prob)| → Binary-TV divergence
  5. w_t decreasing: early tokens w_t ≈ 1, late tokens w_t ≈ w_min

```python
    # Prefix sums with one-token right-shift
    S_cum = torch.cumsum(Z_t, dim=-1)
    W_cum = torch.cumsum(w_t, dim=-1)
    S_prev = torch.cat([torch.zeros_like(S_cum[:, :1]), S_cum[:, :-1]], dim=-1)
    W_prev = torch.cat([torch.zeros_like(W_cum[:, :1]), W_cum[:, :-1]], dim=-1)
```

★★★★★★★★★ Prefix sums:
  S_cum = cumsum(Z_t) → cumulative weighted divergence
  W_cum = cumsum(w_t) → cumulative position weight
  S_prev = right-shift of S_cum → S_prev[t] = S_cum[t-1] → S_prev[0] = 0
  W_prev = right-shift of W_cum → W_prev[t] = W_cum[t-1] → W_prev[0] = 0
  → Decision at position t uses ONLY preceding prefix → S_{t-1}, W_{t-1}

```python
    # Per-sequence dynamic budget calibration (Eq. 22)
    D_for_q = torch.where(response_mask_f.bool(), D_t, torch.full_like(D_t, float("nan")))
    q_seq = torch.nanquantile(D_for_q, q=delta_b_q, dim=-1)  # (B,)
    q_seq = torch.nan_to_num(q_seq, nan=delta_b)              # Empty seq fallback
    delta_b_seq = (delta_b_k * q_seq).clamp(min=delta_b, max=2.0 * delta_b).unsqueeze(-1)  # (B, 1)
```

★★★★★★★★★ Per-sequence budget calibration:
  1. D_for_q → NaN for padding positions → nanquantile ignores padding
  2. q_seq → P90 of D_t per sequence → adaptive budget per sequence
  3. torch.nan_to_num → empty sequences (all padding) → NaN → fallback to delta_b
  4. delta_b_seq → clamp(k * P90(D), delta_b, 2*delta_b) → bounded range

```python
    # Effective threshold (Eq. 8)
    c_t = (delta + delta_b_seq * W_prev - S_prev).clamp(max=delta)

    # Keep decision (Eq. 10)
    feasible = Z_t <= c_t                                 # Budget feasibility
    toward_mu = (advantages * (ratio - 1.0)) <= 0.0       # Safe updates toward mu
    valid_mask = (toward_mu | feasible).detach().float() * response_mask_f
```

★★★★★★★★★ Two-clause mask:
  1. toward_mu = A_t * (rho_t - 1) <= 0 → updates that REDUCE divergence → always keep
  2. feasible = Z_t <= c_t → within dynamic budget → keep
  3. OR logic → either clause suffices → toward_mu overrides budget constraint
  4. .detach().float() → mask is stop-gradient → not part of loss computation

★★★★★★★★★ toward_mu clause semantics (with bypass_mode ratio rho_t = pi/mu):
  - A_t > 0, rho_t < 1 → pi assigns less prob than mu → update increases toward mu → SAFE
  - A_t < 0, rho_t > 1 → pi assigns more prob than mu → update decreases toward mu → SAFE
  - A_t > 0, rho_t > 1 → pi assigns more prob than mu → update increases divergence → RISKY → need budget check
  - A_t < 0, rho_t < 1 → pi assigns less prob than mu → update increases divergence → RISKY → need budget check

### 2.5 Loss Computation

```python
pg_losses = -advantages * truncated_ratio * log_prob * valid_mask

if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights

pg_loss = agg_loss(
    loss_mat=pg_losses, loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode, **config.global_batch_info
)
```

★★★★★★★★★ Loss formula identical to DPPO:
  pg_losses = -A * truncated_ratio * log_prob * valid_mask

  - truncated_ratio = clamp(pi/mu, max=20.0) → TIS upper bound
  - log_prob → gradients flow through here (not through ratio or mask)
  - valid_mask → CPPO trust-region gate → stop-gradient
  - Same surrogate objective as DPPO → only masking decision differs!

★★★★★★★★★ Important: valid_mask multiplies the ENTIRE loss term →
  Masked tokens get pg_losses = 0 → excluded from gradient →
  Unmasked tokens get full gradient → toward-mu tokens always unmasked

### 2.6 Metrics

```python
rejected = (1.0 - valid_mask) * response_mask_f
pg_clipfrac = verl_F.masked_mean(rejected, response_mask_f)
pg_clipfrac_lower = verl_F.masked_mean((ratio > clip_ratio_c).float() * valid_mask, response_mask_f)

pg_metrics = {
    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    "actor/ppo_kl": ppo_kl.detach().item(),
    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
}
```

★★★★★★★★★ Metrics match DPPO convention:
  pg_clipfrac → fraction of response tokens rejected by CPPO mask (excludes toward-mu)
  ppo_kl → KL(pi_theta || mu) measured by masked_mean(log_prob - old_log_prob)
  pg_clipfrac_lower → fraction of TIS-truncated ratios among valid tokens

---

## 3. ★★★★★★★★ bypass_mode Dispatcher Interaction

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.1 bypass_mode Entry Point (rollout_corr_helper.py:1102-1138)

```python
def apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
    """Setup bypass mode: Use rollout_log_probs as old_log_probs."""
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]  # ZERO-COST substitution!

    with open_dict(policy_loss_config):
        policy_loss_config["rollout_correction"] = rollout_corr_config
        policy_loss_config["loss_mode"] = "bypass_mode"
```

★★★★★★★★★ apply_bypass_mode does TWO things:
  1. old_log_probs = rollout_log_probs → substitution at batch level → no forward pass!
  2. policy_loss_config.loss_mode = "bypass_mode" → dispatches to bypass_mode loss function

★★★★★★★★★ BUT: CPPO does NOT use loss_mode="bypass_mode"!
  → CPPO uses loss_mode="cppo" directly → separate registered policy loss
  → CPPO + bypass_mode = two separate configs that work together!

### 3.2 ★★★★★★★★ CPPO + bypass_mode: TWO Configs Working Together

The key insight: CPPO and bypass_mode are INDEPENDENT mechanisms that MUST be combined:

```
bypass_mode (RolloutCorrectionConfig):
  → Sets old_log_probs = rollout_log_probs at BATCH level
  → This is done BEFORE the loss function is called
  → Effect: the "old_log_prob" parameter passed to ANY loss function IS rollout_log_prob

CPPO (policy_loss.loss_mode):
  → Registered as separate @register_policy_loss("cppo")
  → Receives old_log_prob parameter → which IS rollout_log_prob thanks to bypass_mode
  → CPPO reads mu directly from old_log_prob → divergence D_t = |pi - mu| = CORRECT!
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ CRITICAL: CPPO divergence MUST be measured against mu (rollout policy)!
  Without bypass_mode → old_log_prob from separate forward → pi_old ≠ mu
  → CPPO divergence = |pi - pi_old| ≠ |pi - mu| → INCORRECT measurement!

  With bypass_mode → old_log_prob = rollout_log_prob = mu
  → CPPO divergence = |pi - mu| → CORRECT measurement!

  Therefore: CPPO MUST use bypass_mode → not optional → REQUIRED for correctness!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.3 main_ppo_sync Trainer Interaction

From PR #6731 example script (examples/cppo_trainer/run_qwen3_30b_a3b_megatron.sh):

```bash
# In the synchronous TransferQueue trainer:
# bypass_mode=True sets old_log_probs = rollout_log_probs
# and leaves loss_mode untouched → CPPO reads mu directly
algorithm.rollout_correction.bypass_mode=True
actor_rollout_ref.actor.policy_loss.loss_mode=cppo
```

★★★★★★★★★ main_ppo_sync trainer flow:
  1. Rollout phase → collect trajectories with vLLM → compute rollout_log_probs
  2. bypass_mode=True → apply_bypass_mode() → old_log_probs = rollout_log_probs
  3. Training phase → loss_mode="cppo" → compute_policy_loss_cppo() called
  4. CPPO receives old_log_prob = rollout_log_prob = mu → divergence measured against mu

★★★★★★★★★ This is different from ray_trainer.py bypass_mode flow:
  ray_trainer.py → bypass_mode → loss_mode = "bypass_mode" → dispatches to ppo_clip or reinforce
  main_ppo_sync → bypass_mode → leaves loss_mode unchanged → CPPO reads mu directly

★★★★★★★★★ TransferQueue requirement:
  → pip install TransferQueue → needed for main_ppo_sync trainer
  → +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=1

### 3.4 ★★★★★★★★ Why CPPO Cannot Work Without bypass_mode

```
★★★★★★★★★ Mathematical proof that bypass_mode is REQUIRED:

CPPO divergence definition:
  D_t = |pi(y_t|s_t) - mu(y_t|s_t)|

Implementation reads:
  prob = torch.exp(log_prob)        → pi_theta(y_t|s_t)
  old_prob = torch.exp(old_log_prob) → THIS MUST BE mu(y_t|s_t)!

Without bypass_mode:
  old_log_prob = pi_old(y_t|s_t)    → from separate actor forward pass
  → old_prob = pi_old ≠ mu          → divergence = |pi - pi_old| ≠ |pi - mu|
  → INCORRECT! CPPO's position-weighted budget is calibrated against WRONG divergence!

With bypass_mode:
  old_log_prob = rollout_log_prob    → directly from rollout collection
  → old_prob = mu                   → divergence = |pi - mu| = CORRECT!
  → Budget calibrated against true divergence → position weighting meaningful!

★★★★★★★★★ Specific failure modes without bypass_mode:

  1. Budget calibration wrong:
     → quantile(D_t, 0.9) where D_t = |pi - pi_old| → budget based on pi_old gap
     → BUT CPPO's position weighting is designed for mu gap → mismatch!
     → Budget may over-allocate (if pi_old ≈ mu) or under-allocate (if pi_old far from mu)

  2. toward_mu clause wrong:
     → toward_mu = A_t * (rho_t - 1) <= 0 where rho_t = pi/pi_old
     → BUT rho_t should be pi/mu → toward_mu should check against mu, not pi_old!
     → "Safe" updates may be misclassified → moving toward pi_old ≠ moving toward mu

  3. Position weighting wrong:
     → w_t designed to constrain divergence from mu → early tokens vs mu
     → Without bypass → constraining divergence from pi_old → mismatched semantics!
```

---

## 4. ★★★★★★★★ Exact Code Path: CPPO + bypass_mode + GRPO Advantage on RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 4.1 Complete Training Loop Code Path

```
Step 1: ROLLOUT (vLLM inference engine)
  → vLLM generates trajectories with pi_rollout = mu
  → rollout.calculate_log_probs = True → compute mu(y_t|s_t) for each token
  → Store: rollout_log_probs (shape: [batch, seq_len])

Step 2: ADVANTAGE COMPUTATION (GRPO)
  → algorithm.adv_estimator = grpo
  → compute_grpo_outcome_advantage()
  → scores = token_level_rewards.sum(dim=-1) → per-sequence total reward
  → For each group (same prompt):
    → id2mean[idx] = mean(scores in group)
    → id2std[idx] = std(scores in group)
    → advantages[i] = (scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
  → Returns: advantages, returns (both shape: [batch, seq_len])

★★★★★★★★★ GRPO advantage = group-normalized outcome reward → no critic needed!

Step 3: BYPASS MODE SETUP
  → algorithm.rollout_correction.bypass_mode = True
  → apply_bypass_mode() called in main_ppo_sync
  → batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
  → ZERO-COST substitution! No separate forward pass needed!

★★★★★★★★★ This saves the ref model forward pass → ~14GB VRAM on RTX 4090!

Step 4: CPPO LOSS COMPUTATION
  → actor.policy_loss.loss_mode = cppo
  → compute_policy_loss_cppo() called with:
    old_log_prob = rollout_log_probs (= mu!) ← from bypass_mode
    log_prob = current pi_theta ← from actor forward pass
    advantages = GRPO advantages ← from Step 2

  Inside compute_policy_loss_cppo():
    a) Extract config: delta, w_min, delta_b, delta_b_q, delta_b_k
    b) Compute ratio = pi_theta / mu (via log_prob - old_log_prob)
    c) Compute truncated_ratio = clamp(ratio, max=20.0) + detach()
    d) Under torch.no_grad(): construct CPPO mask
       → D_t = |pi - mu| (Binary-TV divergence)
       → w_t = position weight decreasing
       → Z_t = w_t * D_t
       → S_prev, W_prev = prefix sums (right-shifted)
       → delta_b_seq = P90 calibration per sequence
       → c_t = dynamic threshold
       → toward_mu = A*(ratio-1) <= 0
       → feasible = Z_t <= c_t
       → valid_mask = toward_mu OR feasible
    e) Compute loss: -A * truncated_ratio * log_prob * valid_mask
    f) Aggregate loss via agg_loss()
    g) Return: pg_loss, metrics (pg_clipfrac, ppo_kl, pg_clipfrac_lower)

★★★★★★★★★ CPPO mask overhead: near-zero! (cumsum + quantile + comparison ops)
  → All under torch.no_grad() → no backward overhead
  → Trivial compared to forward/backward through model
```

### 4.2 Memory Budget on RTX 4090

```
★★★★★★★★★ RTX 4090 (24GB) VRAM budget for CPPO + GRPO + bypass + LoRA-32:

| Component        | Qwen3-4B BF16 | Qwen2.5-7B BF16 | Qwen2.5-1.5B BF16 |
|-----------------|---------------|-----------------|-------------------|
| Base model       | ~8GB          | ~14GB           | ~3GB              |
| LoRA-32 adapters | ~0.3GB        | ~0.5GB          | ~0.1GB            |
| Adam (LoRA FP32) | ~1GB          | ~2GB            | ~0.5GB            |
| KV cache (rollout)| ~3GB         | ~4GB            | ~1GB              |
| Training buffers | ~1.5GB        | ~2GB            | ~0.5GB            |
| CPPO mask tensors| ~0.01GB       | ~0.01GB         | ~0.01GB           |
| **Total**        | **~13.8GB**   | **~22.5GB**     | **~5.1GB**        |
| **Headroom**     | **~10.2GB**   | **~1.5GB**      | **~18.9GB**       |

★★★★★★★★★ Key savings from bypass_mode:
  → No ref model → save ~14GB (7B model)
  → No pi_old forward pass → save ~3-5GB activation memory
  → CPPO mask tensors → negligible (<0.01GB) → D_t, w_t, Z_t, c_t, valid_mask
```

---

## 5. ★★★★★★★★ CPPO vs DPPO vs PPO-clip vs Vanilla PPO Comparison

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 5.1 Trust Region Mechanism Comparison

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Method    | Trust Region           | Divergence Anchor | Position-Aware | Prefix-Aware | Adaptive Budget |
|-----------|------------------------|-------------------|----------------|-------------|----------------|
| PPO-clip  | |rho-1| <= epsilon     | pi_old            | NO (uniform)  | NO          | NO (fixed)     |
| DPPO-TV   | D_t <= delta           | pi_old            | NO (uniform)  | NO          | NO (fixed)     |
| DPPO-KL   | KL_t <= delta          | pi_old            | NO (uniform)  | NO          | NO (fixed)     |
| CISPO     | clip(ratio) * log_prob | pi_old            | NO (uniform)  | NO          | NO (TIS only)  |
| CPPO      | w_t*D_t <= c_t         | mu (rollout!)     | YES (w_t)     | YES (c_t)   | YES (P90 seq)  |

★★★★★★★★★ CPPO is the ONLY method with ALL three improvements:
  1. Position-aware → w_t ∈ [w_min, 1] → early tokens tighter
  2. Prefix-aware → c_t = min(delta, delta + delta_b*W_prev - S_prev) → dynamic budget
  3. Adaptive budget → delta_b_seq = P90 calibration → per-sequence

★★★★★★★★★ CPPO divergence anchor = mu (rollout) vs pi_old for all others:
  → This is a SEMANTIC difference, not just a parameter difference
  → CPPO measures "how far from rollout" → others measure "how far from pi_old"
  → With bypass_mode → pi_old = mu → all methods agree
  → Without bypass → CPPO is WRONG (divergence vs pi_old ≠ vs mu)
```

### 5.2 Loss Term Comparison

```
★★★★★★★★★ All four methods use SAME surrogate objective:
  pg_losses = -A * truncated_ratio * log_prob * mask

Difference is ONLY in mask construction:

PPO-clip mask:
  → mask = 1 (always) → clipping handled by min(r*A, clip(r)*A) → heuristic
  → OR: dual-clip PPO → where(A<0, min(r*A, clip(r)*A), min(r*A, c*A))

DPPO-TV mask:
  → valid_positive_mask = (prob - old_prob) <= delta_high
  → valid_negative_mask = (prob - old_prob) >= -delta_low
  → valid_mask = where(A>0, valid_positive_mask, valid_negative_mask)

DPPO-KL mask:
  → binary_kl = old_prob*(old_log_prob - log_prob) + (1-old_prob)*log((1-old_prob)/(1-prob))
  → valid_positive_mask = (binary_kl <= delta_high) OR (prob <= old_prob)
  → valid_negative_mask = (binary_kl <= delta_low) OR (prob >= old_prob)
  → valid_mask = where(A>0, valid_positive_mask, valid_negative_mask)

CPPO mask:
  → D_t = |prob - old_prob| (Binary-TV, same as DPPO-TV)
  → w_t = position weight → Z_t = w_t * D_t
  → c_t = min(delta, delta + delta_b_seq * W_prev - S_prev)
  → toward_mu = A*(ratio-1) <= 0 → always keep safe updates
  → feasible = Z_t <= c_t → budget check
  → valid_mask = toward_mu OR feasible
```

★★★★★★★★★ CPPO vs DPPO-TV mask differences:
  1. CPPO adds position weight w_t → DPPO-TV: uniform (all w_t = 1 implicit)
  2. CPPO adds prefix budget c_t → DPPO-TV: fixed delta
  3. CPPO adds toward_mu clause → DPPO-TV: A-sign-based mask only
  4. CPPO: toward_mu OR feasible → DPPO-TV: A>0 check positive, A<0 check negative

★★★★★★★★★ The toward_mu clause is MORE GENERAL than DPPO's A-sign mask:
  DPPO-TV: A>0 → check prob <= old_prob + delta → A<0 → check prob >= old_prob - delta
  CPPO: A*(ratio-1)<=0 → keep ANY update that reduces divergence from mu → regardless of magnitude

### 5.3 Theoretical Bound Comparison

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

PPO-clip (Schulman 2017):
  → Heuristic ratio bound → no divergence guarantee
  → |rho_t - 1| <= epsilon → does NOT bound actual policy divergence
  → Low-probability tokens: |rho-1| volatile → over-penalized
  → High-probability tokens: |rho-1| stable → under-penalized

DPPO (Qi 2026, arXiv:2602.04879):
  → Principled divergence bound → D_t <= delta → guarantees TV/KL constraint
  → But: UNIFORM per-token → ignores position and prefix structure
  → Pointwise → doesn't account for cumulative divergence

CPPO (Mao 2026, arXiv:2606.10968):
  → Position-weighted divergence bound → w_t * D_t <= c_t
  → Prefix-aware → c_t shrinks as divergence accumulates
  → ★★★★★★★★ Theorem 1: provably tighter policy-improvement bound than uniform methods
  → Starting from finite-horizon performance-difference identity:
    → Early-token shifts carry remaining-horizon penalty
    → Position-weighting matches this structure → theoretically optimal
  → ★★★★★★★★ CPPO catches "budget overruns" that PPO/DPPO ALL miss!
```

### 5.4 RTX 4090 Practical Comparison

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Factor              | PPO-clip+bypass | DPPO-TV+bypass | CPPO+bypass     |
|---------------------|-----------------|----------------|-----------------|
| VRAM overhead       | Baseline        | +0             | +0 (<0.01GB)    |
| Compute overhead    | Baseline        | +0             | +near-zero      |
| Ref model needed    | NO (bypass)     | NO (bypass)    | NO (bypass)     |
| Trust region quality| Heuristic ★★    | Principled ★★★★| Optimal ★★★★★  |
| Position-aware      | NO              | NO             | YES             |
| Prefix-aware        | NO              | NO             | YES             |
| Long-horizon stable | Can collapse    | Better         | Best            |
| Short-horizon stable| OK              | OK             | OK (same + early-token)|
| Theoretical bound   | None            | Uniform D_t    | Position+prefix |
| Config complexity   | Simple          | Same+delta     | Same+4 fields   |

★★★★★★★★★ RTX 4090 recommendation:
  → CPPO+bypass+GRPO = optimal (near-zero overhead + provably better bound)
  → DPPO-TV+bypass+GRPO = good alternative (principled but uniform)
  → PPO-clip+bypass+GRPO = standard (works but heuristic)
  → All three: same VRAM! same compute! → CPPO wins by theory alone!
```

---

## 6. ★★★★★★★★ bypass_mode Dispatcher vs CPPO Direct Registration

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 6.1 Two Parallel Paths in verl

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Path A: bypass_mode dispatcher (existing, for PPO-clip and REINFORCE):
  → apply_bypass_mode() → sets loss_mode = "bypass_mode"
  → compute_policy_loss_bypass_mode() → dispatcher function
  → Reads rollout_corr_config → dispatches to:
    → loss_type="ppo_clip" → compute_policy_loss_vanilla()
    → loss_type="reinforce" → compute_policy_loss_reinforce()
  → IS weights computed inside dispatcher → passed to underlying loss function

Path B: CPPO direct registration (NEW, PR #6731):
  → apply_bypass_mode() → sets old_log_probs = rollout_log_probs
  → BUT does NOT set loss_mode = "bypass_mode" for CPPO!
  → main_ppo_sync → bypass_mode=True → leaves loss_mode untouched
  → loss_mode="cppo" → compute_policy_loss_cppo() called directly
  → CPPO does its own mask construction → no need for bypass_mode dispatcher

★★★★★★★★★ Why CPPO doesn't go through the bypass_mode dispatcher:
  1. CPPO needs to compute D_t = |pi - mu| → uses exp(log_prob) and exp(old_log_prob)
     → The bypass_mode dispatcher computes IS weights → not what CPPO needs
  2. CPPO mask is position-weighted cumulative → fundamentally different from PPO clip
     → Cannot be expressed as PPO-clip or REINFORCE dispatch
  3. CPPO's toward_mu clause uses ratio = pi/mu → semantically different from PPO clip
  4. CPPO is a standalone loss function → @register_policy_loss("cppo")

★★★★★★★★★ BUT CPPO still NEEDS bypass_mode for correctness:
  → bypass_mode provides old_log_prob = rollout_log_prob = mu
  → This is a DATA-LEVEL operation → happens BEFORE any loss function
  → CPPO reads old_log_prob → which IS mu thanks to bypass_mode
  → The bypass_mode dispatcher is NOT used → but bypass_mode DATA substitution IS used!
```

### 6.2 Implication for ray_trainer.py vs main_ppo_sync.py

```
★★★★★★★★★ ray_trainer.py (Ray-based, async):
  → apply_bypass_mode() → ALWAYS sets loss_mode = "bypass_mode"
  → → CPPO cannot be used with ray_trainer.py currently!
  → → Would need modification: allow loss_mode = "cppo" with bypass_mode data setup

★★★★★★★★★ main_ppo_sync.py (TransferQueue, sync):
  → apply_bypass_mode() → sets old_log_probs = rollout_log_probs
  → → BUT leaves loss_mode untouched → CPPO can use loss_mode="cppo" directly!
  → → This is the trainer used in PR #6731 example → CPPO designed for this trainer

★★★★★★★★★ For RTX 4090: main_ppo_sync is the RECOMMENDED trainer anyway:
  → TransferQueue → zero-copy → FusedWorker HYBRID → in-process → faster than Ray
  → Same trainer as verl v0.8 default for bypass mode
```

---

## 7. ★★★★★★★★ Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
★★★★★★★★★ 10 key findings from this source-level reading:

1. ★★★★★★★★ CPPO divergence MUST be measured against rollout policy mu
   → old_log_prob must be rollout_log_prob → bypass_mode REQUIRED
   → Without bypass → divergence = |pi - pi_old| ≠ |pi - mu| → INCORRECT

2. ★★★★★★★★ CPPO + bypass_mode = mathematically REQUIRED pairing
   → Not optional → correctness requirement → not just a convenience
   → toward_mu clause: A*(pi/mu - 1) <= 0 → needs mu, not pi_old
   → Budget calibration: quantile(|pi-mu|) → needs mu divergence statistics

3. ★★★★★★★★ CPPO mask = 3 improvements over DPPO
   → Position weight w_t ∈ [w_min, 1] → early tokens tighter (Theorem 1)
   → Prefix budget c_t → dynamic threshold → prevents cascading drift
   → Per-sequence adaptive budget → P90 calibration → delta_b_seq

4. ★★★★★★★★ CPPO mask computed under torch.no_grad()
   → Trust-region gate → NOT part of loss → no backward overhead
   → Same convention as DPPO valid_mask → .detach().float()

5. ★★★★★★★★ CPPO loss term IDENTICAL to DPPO
   → pg_losses = -A * truncated_ratio * log_prob * valid_mask
   → Only masking decision changes → no new loss terms
   → Same truncated_ratio with clip_ratio_c = 20.0 → TIS upper bound

6. ★★★★★★★★ CPPO is a standalone @register_policy_loss("cppo")
   → Does NOT go through bypass_mode dispatcher
   → Direct registration → main_ppo_sync sets loss_mode="cppo"
   → bypass_mode only provides DATA: old_log_prob = rollout_log_prob

7. ★★★★★★★★ Position weight uses padded length (not valid length)
   → T_fixed = float(resp_len) → padded response length
   → Right-padding zeroed by response_mask → padded positions excluded
   → This matches reference CPPO implementation (paper Appendix D)

8. ★★★★★★★★ Per-sequence budget calibration via nanquantile
   → D_for_q → NaN for padding → nanquantile ignores padding
   → Empty sequences → NaN → torch.nan_to_num → fallback to delta_b
   → clamp(k * P90(D), delta_b, 2*delta_b) → bounded adaptive budget

9. ★★★★★★★★ RTX 4090: CPPO + GRPO + bypass + LoRA-32 = optimal
   → Near-zero overhead (cumsum/quantile <0.01GB)
   → Provably better trust region (Theorem 1)
   → No ref model (bypass) → ~14GB savings on 7B
   → No critic (GRPO advantage) → saves value model memory
   → Works with main_ppo_sync (TransferQueue) → zero-copy

10. ★★★★★★★★ CPPO is NOT yet merged → PR #6731 OPEN
    → Needs: checkout PR branch or wait for merge
    → Requires: pip install TransferQueue (main_ppo_sync)
    → 12th registered policy loss (when merged → after current 11)
```

---

## 8. Source File References

```
★★★★★★★★★ Key source files traced in this reading:

1. verl/trainer/ppo/core_algos.py (PR #6731 diff, +166 lines)
   → @register_policy_loss("cppo") → compute_policy_loss_cppo()
   → Lines: mask construction, position weight, prefix sums, budget calibration

2. verl/trainer/ppo/core_algos.py (existing, lines 2351-2487)
   → @register_policy_loss("bypass_mode") → compute_policy_loss_bypass_mode()
   → Dispatcher → loss_type="ppo_clip" or "reinforce"

3. verl/trainer/ppo/core_algos.py (existing, lines 1372-1450)
   → @register_policy_loss("dppo_tv") → compute_policy_loss_dppo_tv()
   → DPPO Binary-TV → CPPO comparison baseline

4. verl/trainer/ppo/rollout_corr_helper.py (existing, lines 1102-1138)
   → apply_bypass_mode() → old_log_probs = rollout_log_probs
   → Data-level substitution → prerequisite for CPPO correctness

5. verl/workers/config/actor.py (PR #6731 diff, +12 lines)
   → PolicyLossConfig: cppo_w_min, cppo_delta_b, cppo_delta_b_q, cppo_delta_b_k

6. verl/trainer/config/algorithm.py (existing, lines 60-614)
   → RolloutCorrectionConfig → bypass_mode, loss_type presets

7. examples/cppo_trainer/run_qwen3_30b_a3b_megatron.sh (PR #6731 diff, +165 lines)
   → Full example script → bypass_mode=True + loss_mode=cppo + GRPO

8. docs/algo/rollout_corr_math.md (existing)
   → Decoupled PPO theory → bypass mode formulation → IS/RS theory

9. docs/algo/dppo.md (existing)
   → DPPO Binary-TV/KL documentation → CPPO comparison reference
```

---

## 参考

- PR #6731: https://github.com/verl-project/verl/pull/6731
- Paper: arXiv:2606.10968 "Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning"
- Project page: https://hunyuan-cppo.github.io
- DPPO paper: arXiv:2602.04879 "Rethinking the Trust Region in LLM RL"
- PPO paper: arXiv:1707.06347 "Proximal Policy Optimization Algorithms"
- Rollout Correction theory: https://richardli.xyz/rl-collapse
- Rollout Correction paper: arXiv:2512.23075 "Trust Region Masking for Long-Horizon LLM RL"
- verl/trainer/ppo/core_algos.py (PR diff)
- verl/workers/config/actor.py (PR diff)
- Related notes: verl-cppo-algorithm-reading.md, verl-policy-loss-catalog-source-reading.md, rtx4090-grpo-trust-region-comparison.md, rtx4090-verl-cppo-grpo-training-guide.md
