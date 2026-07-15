# GRPO Training Algorithm: Unified Synthesis

**Date**: 2026-07-15 (Session 10)
**Purpose**: Connect all mathematical components (advantage, loss, KL, masks, aggregation) into a single coherent training algorithm
**Sources**: rl-advantage-estimators-mathematical-derivation.md, rl-policy-loss-functions-mathematical-derivation.md, verl V1 trainer, rLLM terminal-rl, arXiv papers

---

## 1. GRPO End-to-End Algorithm

A single GRPO training step, from data input to gradient output:

```
INPUT: batch of (prompt, response) pairs with rewards

Step 1: GROUP BY PROMPT
  Group trajectories by shared prompt:
    G_k = {trajectory_i : prompt_i = prompt_k}
  |G_k| = group_size (typically 8-16)

Step 2: COMPUTE ADVANTAGES (per group)
  For group G_k:
    μ_k = mean(R_i for i ∈ G_k)
    σ_k = std(R_i for i ∈ G_k)

    For each trajectory i ∈ G_k:
      A_i = (R_i - μ_k) / σ_k    [GRPO]
      A_i = R_i - μ_k              [REINFORCE++BL]
      A_i = R_i - μ_LOO_i          [RLOO]
      A_i = R_i                    [REINFORCE]

Step 3: COMPUTE TOKEN-LEVEL ADVANTAGES
  Expand trajectory-level A_i to token-level:
    A_{i,t} = A_i for t ∈ response tokens
    A_{i,t} = 0 for t ∈ prompt tokens

  With response_mask:
    A_{i,t} = A_i · mask_{i,t}
    mask_{i,t} = 1 if t ∈ response, 0 if t ∈ prompt

Step 4: COMPUTE LOG-PROBABILITIES
  Current policy:    logp_curr_{i,t} = log π(a_{i,t} | s_{i,t})
  Old policy:        logp_old_{i,t} = log π_old(a_{i,t} | s_{i,t})
  Reference policy:  logp_ref_{i,t} = log π_ref(a_{i,t} | s_{i,t})

  Ratio: r_{i,t} = exp(logp_curr_{i,t} - logp_old_{i,t})

  bypass_mode: skip logp_old forward → use stored logprobs from rollout

Step 5: COMPUTE POLICY LOSS (per token)
  PPO-clip:
    L_{i,t} = min(r_{i,t}·A_{i,t}, clip(r_{i,t}, 1-ε, 1+ε)·A_{i,t})

  UP-GRPO:
    A ≥ 0: L_{i,t} = r_{i,t}·A_{i,t}                         (NO upper clip)
    A < 0: L_{i,t} = max(r·A, clip(r,1-ε,1+ε)·A, c·A)       (dual-clip)

  CISPO:
    L_{i,t} = clamp(r_{i,t}).detach() · A_{i,t} · logp_curr_{i,t}

Step 6: COMPUTE KL PENALTY (per token)
  Forward KL:  KL_fwd_{i,t} = logp_curr_{i,t} - logp_ref_{i,t}
  Backward KL: KL_bwd_{i,t} = logp_ref_{i,t} - logp_curr_{i,t}
  Symmetric KL: KL_sym = (KL_fwd + KL_bwd) / 2

  Applied as:
    L_total_{i,t} = L_policy_{i,t} + kl_coef · KL_{i,t}

  kl_coef types:
    - Fixed: constant coefficient
    - Adaptive: KL controller (target KL → adjust coef)
    - None: no KL penalty (trust region from clipping only)

Step 7: APPLY MASK AND AGGREGATE
  Mask:
    masked_loss_{i,t} = L_total_{i,t} · response_mask_{i,t}

  Aggregation modes:
    token-mean:  L_i = (1/T_i) Σ_t masked_loss_{i,t}     [per-trajectory, then mean across trajectories]
    seq-mean:    L = (1/N) Σ_i L_i                         [equal weight per trajectory]
    token-mean-token-mean: L = mean over all (i,t) pairs  [each token equal weight]
    seq-mean-token-mean: L = (1/N) Σ_i (1/T_i Σ_t loss)  [each trajectory equal weight, then token-normalize]

Step 8: BACKWARD PASS + OPTIMIZER STEP
  ∇L → gradient accumulation → clip_grad_norm → optimizer.step()
  clip_grad_norm: 1.0 (MUST, not default 0.0!)
  Optimizer: cpu_adam (ZeRO-2 offload) for RTX 4090

Step 9: WEIGHT SYNC (for HYBRID rollout)
  Training engine → weight update → sync to rollout engine
  LoRA mode: delta sync (only LoRA adapter weights)
  Full mode: full param sync (AVOID on RTX 4090!)
```

---

## 2. Mathematical Proof: Why Each Component Matters

### 2.1 Group Normalization (GRPO)

**Claim**: GRPO advantages have zero mean and unit variance within each group.

**Proof**:
```
Σ_i A_i = Σ_i (R_i - μ)/σ = (Σ_i R_i - |G|μ)/σ = (|G|μ - |G|μ)/σ = 0 ✓

Σ_i A_i² = Σ_i (R_i - μ)²/σ² = (σ²|G|)/(σ²) = |G|
→ mean(A²) = |G|/|G| = 1 ✓
→ std(A) = 1 ✓
```

**Implication**: Zero mean → half the group has positive advantage, half negative → balanced learning signal. Unit variance → bounded magnitude → no gradient explosion.

### 2.2 PPO-Clip Trust Region

**Claim**: PPO-clip creates an implicit trust region around π_old.

**Proof sketch**:
```
When r > 1+ε and A > 0: gradient = 0 (upper clip activates)
When r < 1-ε and A < 0: gradient = 0 (lower clip activates)

This means:
- If policy moves too far from π_old in ANY direction → gradient shuts off
- Policy is "trapped" in [1-ε, 1+ε] ratio region for each token
- Result: implicit trust region of radius ε around old policy

But: UP-GRPO removes upper clip for A>0 → no trust region ceiling for good actions!
```

### 2.3 KL Penalty vs Clipping

**Claim**: KL penalty and clipping serve different purposes and can complement each other.

```
Clipping:
  - Local per-token constraint → prevents individual token ratio explosion
  - Heuristic (not principled) → ε=0.2 works empirically but no theory guarantees
  - Zero-gradient outside trust region → no learning signal

KL Penalty:
  - Global sequence-level constraint → penalizes overall distribution shift
  - Principled → comes from trust region theory (TRPO)
  - Continuous gradient → always provides learning signal (soft constraint)

Best: BOTH together → clipping handles per-token, KL handles per-sequence
  This is what PPO originally proposed but implementations often drop KL
```

### 2.4 Response Mask

**Claim**: Masking out prompt tokens is mathematically necessary.

```
Without mask:
  ∇J = Σ_{all tokens} A_{i,t} · ∇logπ(a_{i,t}|s_{i,t})
  → prompt tokens contribute to gradient → policy changes its response to prompts
  → but prompts are FIXED input → we cannot optimize prompt distribution!
  → gradient from prompt tokens is noise (increases variance without reducing bias)

With mask:
  ∇J = Σ_{response tokens} A_{i,t} · ∇logπ(a_{i,t}|s_{i,t})
  → only response tokens contribute → policy only changes generation behavior
  → correct: we want to optimize response quality, not prompt sensitivity
```

### 2.5 Aggregation Matters

**Claim**: Different aggregation modes create different gradient weighting.

```
token-mean-token-mean:
  L = mean(all response token losses)
  → each token has equal weight → long sequences dominate (more tokens = more gradient)
  → bias toward optimizing long sequences

seq-mean-token-mean:
  L = (1/N) Σ_i (1/T_i Σ_t loss_i,t)
  → each trajectory has equal weight, regardless of length
  → fair: short and long sequences contribute equally
  → RTX 4090 RECOMMENDED (variable-length GRPO)

GSPO forces: seq-mean-token-mean with sequence-level ratio
  → length-normalized ratio + trajectory-equal weighting
  → addresses BOTH length bias and aggregation bias
```

---

## 3. GRPO vs PPO Architecture Comparison

| Aspect | PPO (Original) | GRPO (2024+) |
|--------|---------------|---------------|
| Critic (Value Network) | Required (V(s) baseline) | NOT required (group baseline) |
| Baseline | Value function V(s) | Group mean μ_G |
| Advantage | A = R + γV(s') - V(s) | A = (R - μ_G) / σ_G |
| Memory overhead | 2× model (actor + critic) | 1× model (actor only) |
| Training complexity | TD error + GAE + critic loss | Simple group normalization |
| RTX 4090 fit | OOM (2× model > 24GB) | VIABLE (1× model < 14.6GB) |
| Variance | Low (learned baseline) | Low (group normalization) |
| Bias | None (optimal baseline) | None (mean subtraction) |
| Group size req | Any (critic provides baseline) | gs≥4 (statistical baseline) |

**Key insight**: GRPO eliminates the critic → saves 50% memory → enables single-GPU training. The group baseline is a "free" baseline that doesn't require training a separate network.

---

## 4. Complete GRPO Step: Data Flow Diagram

```
                         GRPO Training Step
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   Rollout Data          Reward Data           Model Weights
   (prompts,             (per-trajectory        (π_old, π_ref,
    responses,            scores)                π_curr)
    logprobs)                 │                     │
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌─────────────┐
   │ Group   │         │Advantage│         │ Log-Probs   │
   │ By      │────────▶│ Compute │         │ Forward     │
   │ Prompt  │         │ (GRPO)  │         │ Pass        │
   └─────────┘         └────┬────┘         └──────┬──────┘
                           │                     │
                           ▼                     ▼
                    ┌─────────────┐        ┌─────────────┐
                    │ Token-Level │        │   Ratio     │
                    │ Expand +    │        │ r = exp(Δ)  │
                    │ Mask        │        └──────┬──────┘
                    └──────┬──────┘               │
                           │                      │
                           ▼                      ▼
                    ┌─────────────────────────────────┐
                    │     Policy Loss (UP-GRPO)       │
                    │  A≥0: L=r·A  A<0: L=max(r·A,   │
                    │              clip(r,1-ε,1+ε)·A, │
                    │              c·A)               │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┼──────────────────┐
                    │              │                  │
                    ▼              ▼                  ▼
             ┌──────────┐  ┌──────────┐      ┌──────────┐
             │   KL     │  │  Mask +  │      │ Entropy  │
             │ Penalty  │  │ Aggregate│      │ Bonus    │
             └──────────┘  └──────────┘      └──────────┘
                    │              │                  │
                    ▼              ▼                  ▼
             ┌──────────────────────────────────────────┐
             │           Total Loss                     │
             │  L = L_policy + kl_coef·KL + ent_coef·H │
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │          Backward Pass                    │
             │  ∇L → clip_grad_norm(1.0) → optimizer    │
             └──────────────────┬───────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────────────┐
             │        Weight Update & Sync               │
             │  π_old ← π_curr (for next step ratio)    │
             │  Rollout engine sync (LoRA delta)         │
             └──────────────────────────────────────────┘
```

---

## 5. Cross-Framework Implementation Matrix

### Advantage Estimators × Loss Functions × Aggregation

The "recipe" for each framework's GRPO training:

| Framework | Advantage | Loss | Aggregation | KL | Critic |
|-----------|-----------|------|-------------|-----|--------|
| **verl V1** | GRPO/RLOO/REINFORCE++BL | PPO-clip/UP-GRPO | token-mean/seq-mean | adaptive/fixed | No (GRPO mode) |
| **rLLM terminal-rl** | GRPO/RLOO/REINFORCE/ECHO | 9 losses via plugin | 3 modes (mean/sum/seq-mean-token-mean) | KL configurable | No (GRPO mode) |
| **TRL** | GRPO/REINFORCE++BL | PPO-clip | token-mean | None/fixed | No (GRPOTrainer) |
| **DeepSpeed** | REINFORCE (no group) | PPO-clip | token-mean | adaptive | Optional |
| **Megatron** | Not built-in | Not built-in | Custom | Custom | Optional |

**verl V1 is the most complete** with 14 advantage estimators + 11 policy losses via registry pattern.

---

## 6. RTX 4090 GRPO Training Algorithm (Optimal Configuration)

```
★ RTX 4090 24GB Optimal GRPO Configuration ★

Algorithm: GRPO + UP-GRPO loss
Framework: verl V1 sync trainer
Model: 7B dense (Qwen2.5-7B) or 16B MoE (Qwen3-30B-A3B)

Training config:
  group_size = 8 (ALWAYS ≥4, ≥8 for MoE)
  advantage_estimator = "grpo"
  loss_type = "up_grpo" (our PR #9)
  clip_ratio = 0.2
  clip_ratio_c = 3.0 (dual-clip upper bound)
  kl_coef = 0.01 (fixed, small)
  aggregation = "seq-mean-token-mean" (fair weighting)
  response_mask = True (ALWAYS mask prompt tokens)
  bypass_mode = True (skip logp_old forward → 18Ψ→3.8Ψ)

Infrastructure:
  zero_stage = 2 (ZeRO-2, NOT ZeRO-3!)
  optimizer = cpu_adam (offload to CPU)
  gradient_clipping = 1.0 (MUST set explicitly)
  overlap_comm = False (CUDA stream race #8061)
  fsdp_mode = fsdp1 (NOT fsdp2! leak + MoE bug)
  enforce_eager = True (no CUDA graph for training)
  checkpoint_mode = naive (zero GPU overhead)

Memory budget (24 GiB):
  Model weights (BF16):     ~14.0 GiB (7B dense)
  Activations (bypass):     ~3.8 GiB (vs 18Ψ without bypass)
  Optimizer states (CPU):    0 GiB (offloaded)
  Gradients:                 ~1.4 GiB
  KV cache (rollout):       ~3.0 GiB (during generation phase)
  Total training:            ~19.2 GiB ← fits 24 GiB with 5 GiB headroom
  Total rollout+train:       ~22.2 GiB ← tight but viable with COLOCATE mode

Weight sync:
  sleep_level = 1 (LoRA mode: only KV cache released)
  LoRA rank = 32, alpha = 64 (NEVER rank=64! breaks EOS)
  sync_mode = delta (only LoRA adapter weights, ~50 MiB vs ~14 GiB)
  rollout_engine = sglang (sleep/wake support)

★ REINFORCE degeneration rule:
  group_size = 1 → A = 0 → NO learning signal → NEVER!
  group_size = 2 → σ ≈ 0 for similar rewards → fragile
  group_size = 4 → minimum viable for GRPO
  group_size = 8 → RTX 4090 recommended (balanced variance + throughput)
  group_size = 16 → more stable but less throughput (2× generation time)
```

---

## 7. Gradient Flow Analysis

### 7.1 Where does the gradient come from?

```
L_total = Σ_{i,t} (response_mask_{i,t} · [L_policy_{i,t} + kl_coef · KL_{i,t}])

Gradient for parameter θ:
  ∇_θ L = Σ_{i,t} mask_{i,t} · [∇_θ L_policy_{i,t} + kl_coef · ∇_θ KL_{i,t}]

PPO-clip gradient:
  If r ∈ [1-ε, 1+ε]:  ∇_θ L_policy = A · ∇_θ logp_curr   (full gradient)
  If r > 1+ε, A > 0:  ∇_θ L_policy = 0                    (upper clip)
  If r < 1-ε, A < 0:  ∇_θ L_policy = 0                    (lower clip)

UP-GRPO gradient:
  A ≥ 0:  ∇_θ L_policy = A · ∇_θ logp_curr   (ALWAYS flows for positive A!)
  A < 0:  ∇_θ L_policy = same as PPO-clip     (dual-clip protects)

CISPO gradient:
  ALL tokens: ∇_θ L_policy = clamp(r).detach() · A · ∇_θ logp_curr
  → clamp(r) is detached → acts as WEIGHT not GATE
  → gradient ALWAYS flows through logp_curr (never zeroed!)

KL gradient:
  ∇_θ KL_fwd = ∇_θ logp_curr - ∇_θ logp_ref   (but logp_ref is from frozen ref model)
  → only ∇_θ logp_curr contributes from KL_fwd term

Total gradient = Σ tokens: [policy_gradient + kl_gradient]
```

### 7.2 Gradient Magnitude

```
Expected gradient magnitude:
  E[|∇L|] ≈ E[|A|] · E[|∇logπ|] + kl_coef · E[|∇KL|]

  GRPO: E[|A|] ≈ 1 (normalized to std=1) → stable magnitude
  REINFORCE: E[|A|] = E[|R|] → depends on reward scale → unstable
  UP-GRPO: positive A unbounded → larger gradient for good actions → faster learning
  CISPO: clamp(r) weight ≈ 1 in trust region → similar magnitude to PPO-clip

  clip_grad_norm = 1.0:
    If total grad norm > 1.0 → scale all gradients by 1/norm
    → prevents gradient explosion regardless of advantage estimator
    → MANDATORY (default 0.0 means NO clipping → explosion risk!)
```

---

## 8. Common Failure Modes

| Failure | Root Cause | Symptom | Fix |
|---------|-----------|---------|-----|
| REINFORCE degeneration | gs=1 → A=0 | Loss=0, no learning | gs≥4 (MUST) |
| Gradient explosion | grad_clip=0.0 | Loss spikes, NaN | grad_clip=1.0 (MUST) |
| Length bias | token-mean aggregation | Long responses dominate | seq-mean-token-mean |
| NaN in advantage | zero std (all same reward) | Division by zero | σ=0 → A=0 fallback |
| EOS token lost | LoRA rank=64 | Responses never end | rank=32 + alpha=64 |
| Policy collapse | KL too strong | All responses identical | Reduce kl_coef |
| Memory leak | FSDP2 all_gather | OOM after many steps | FSDP1 ONLY |
| Stream race | overlap_comm=True | Intermittent NaN | overlap_comm=False |
| Weight staleness | No sync after update | Rollout uses old weights | delta sync per step |

---

## Session Stats
- **Unified synthesis** of 6 advantage estimators + 10 policy losses + KL + masks + aggregation
- **Mathematical proofs** for group normalization, trust region, mask necessity, aggregation bias
- **Complete data flow diagram** from rollout data to weight update
- **Cross-framework matrix** mapping each framework's GRPO recipe
- **RTX 4090 optimal config** with memory budget and gradient analysis
- **9 common failure modes** with root causes and fixes
