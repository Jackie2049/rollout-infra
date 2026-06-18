# GRPO Algorithm — Complete Mathematical Derivation

> 2026-06-19 | Algorithm Theory | From REINFORCE to GRPO to CPPO
> ★★★★★★★★ RTX 4090 Context: Understanding the math behind each algorithm is CRITICAL for choosing the right training approach
> ★★★★★★★★ GRPO = Group Relative Policy Optimization = REINFORCE with group-normalized advantage = no critic needed!

---

## 1. The RL Alignment Problem

### 1.1 Objective

Given a policy $\pi_\theta$ (the LLM being trained), we want to maximize expected reward:

$$J(\theta) = \mathbb{E}_{\mathbf{y} \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y})]$$

Where:
- $\mathbf{x}$ = prompt (question/task)
- $\mathbf{y} = (y_1, y_2, ..., y_T)$ = response (token sequence)
- $r(\mathbf{x}, \mathbf{y})$ = reward function (score of the response)
- $\pi_\theta(y_t | \mathbf{x}, y_1, ..., y_{t-1})$ = policy probability of token $y_t$

### 1.2 The Gradient Problem

We need $\nabla_\theta J(\theta)$ to optimize. But $J(\theta)$ involves sampling from $\pi_\theta$ (discrete, non-differentiable) → can't directly compute gradient through the sampling process.

**Solution**: Use the **score function trick** (REINFORCE trick, aka log-ratio trick):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathbf{y} \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y}) \cdot \nabla_\theta \log \pi_\theta(\mathbf{y} | \mathbf{x})]$$

**Derivation**:
$$J(\theta) = \int \pi_\theta(\mathbf{y}|\mathbf{x}) \cdot r(\mathbf{x},\mathbf{y}) \, d\mathbf{y}$$
$$\nabla_\theta J(\theta) = \int \nabla_\theta \pi_\theta(\mathbf{y}|\mathbf{x}) \cdot r(\mathbf{x},\mathbf{y}) \, d\mathbf{y}$$

Using the identity $\nabla_\theta \pi_\theta = \pi_\theta \cdot \nabla_\theta \log \pi_\theta$:

$$\nabla_\theta J(\theta) = \int \pi_\theta(\mathbf{y}|\mathbf{x}) \cdot r(\mathbf{x},\mathbf{y}) \cdot \nabla_\theta \log \pi_\theta(\mathbf{y}|\mathbf{x}) \, d\mathbf{y}$$
$$= \mathbb{E}_{\mathbf{y} \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y}) \cdot \nabla_\theta \log \pi_\theta(\mathbf{y} | \mathbf{x})]$$

---

## 2. REINFORCE: The Foundation

### 2.1 Basic REINFORCE

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} r(\mathbf{x}, \mathbf{y}^i) \cdot \nabla_\theta \log \pi_\theta(\mathbf{y}^i | \mathbf{x})$$

Where $\mathbf{y}^i$ are N samples from $\pi_\theta$.

**Expanding per token**:
$$\log \pi_\theta(\mathbf{y}^i | \mathbf{x}) = \sum_{t=1}^{T} \log \pi_\theta(y_t^i | \mathbf{x}, y_1^i, ..., y_{t-1}^i)$$

So the gradient per sample:
$$\nabla_\theta \log \pi_\theta(\mathbf{y}^i | \mathbf{x}) = \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t^i | \mathbf{x}, y_1^i, ..., y_{t-1}^i)$$

**Problem**: REINFORCE has **HIGH variance**. The reward $r(\mathbf{x}, \mathbf{y})$ scales the entire gradient, so one very good/bad response dominates → unstable training.

### 2.2 REINFORCE with Baseline

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \left(r(\mathbf{x}, \mathbf{y}^i) - b(\mathbf{x})\right) \cdot \nabla_\theta \log \pi_\theta(\mathbf{y}^i | \mathbf{x})$$

The baseline $b(\mathbf{x})$ is reward-independent → doesn't change the gradient's expectation (because $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$), but reduces variance!

**Common baselines**:
- Mean reward: $b(\mathbf{x}) = \frac{1}{N}\sum_i r(\mathbf{x}, \mathbf{y}^i)$ — simplest
- Learned value function: $b(\mathbf{x}) = V_\phi(\mathbf{x})$ — PPO's approach (requires critic network!)
- Group mean (GRPO): $b(\mathbf{x}) = \frac{1}{G}\sum_{g=1}^{G} r(\mathbf{x}, \mathbf{y}^g)$ — GRPO's approach!

---

## 3. PPO: Proximal Policy Optimization

### 3.1 The Ratio Trick

Instead of using $\log \pi_\theta$ directly, PPO uses the **probability ratio**:

$$\rho_t(\theta) = \frac{\pi_\theta(y_t | \mathbf{x}, y_1, ..., y_{t-1})}{\pi_{\theta_{old}}(y_t | \mathbf{x}, y_1, ..., y_{t-1})}$$

**Advantage**: This ratio measures how much the policy has changed since the last update. If $\rho_t = 1$, no change. If $\rho_t > 1$, policy increased probability of this token.

### 3.2 PPO Objective (Clipped)

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t(\theta) \cdot A_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

Where $A_t$ is the **advantage** at step $t$:
$$A_t = r(\mathbf{x}, \mathbf{y}) - V_\phi(\mathbf{x}, y_1, ..., y_t)$$

**The clip prevents destructive updates**:
- If $A_t > 0$ (good action) and $\rho_t > 1+\epsilon$: objective = $(1+\epsilon) \cdot A_t$ → clipped, no incentive to increase further
- If $A_t < 0$ (bad action) and $\rho_t < 1-\epsilon$: objective = $(1-\epsilon) \cdot A_t$ → clipped, no incentive to decrease further

### 3.3 PPO's Problems for RTX 4090

1. **Requires critic network** $V_\phi$ → extra ~2 GiB GPU memory for value model
2. **Requires reference model** $\pi_{ref}$ → extra ~11 GiB for KL penalty
3. **Total: policy + critic + ref = ~24 GiB** → EXCEEDS RTX 4090 capacity!
4. **PPO needs 3 models on GPU simultaneously** → NOT viable on single RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**PPO is NOT viable on RTX 4090 dp=1**: policy (~11 GiB) + critic (~2 GiB) + reference (~11 GiB) = ~24 GiB → need bypass_mode to drop ref/critic → GRPO is the answer!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. GRPO: Group Relative Policy Optimization

### 4.1 Key Idea: Replace Critic with Group Mean

GRPO eliminates the critic $V_\phi$ by using **group-relative advantages**:

For a prompt $\mathbf{x}$, sample $G$ responses $\{\mathbf{y}^1, ..., \mathbf{y}^G\}$ from $\pi_\theta$. Compute rewards $\{r^1, ..., r^G\}$. Then:

$$A^i = \frac{r^i - \mu_G}{\sigma_G}$$

Where:
- $\mu_G = \frac{1}{G}\sum_{g=1}^{G} r^g$ — group mean (baseline!)
- $\sigma_G = \sqrt{\frac{1}{G}\sum_{g=1}^{G}(r^g - \mu_G)^2}$ — group std (normalization!)

**This is EXACTLY REINFORCE with baseline = group mean + normalization!**

The normalization by $\sigma_G$ is important — it makes advantages scale-invariant. Whether rewards are [0,1] or [0,100], advantages are always in ~[-2, +2] range.

### 4.2 GRPO Objective

$$L^{GRPO}(\theta) = \mathbb{E}_{\mathbf{x}, \{\mathbf{y}^g\}}\left[\frac{1}{G}\sum_{g=1}^{G}\frac{1}{T}\sum_{t=1}^{T}\min\left(\rho_t^g(\theta) \cdot A^g, \text{clip}(\rho_t^g(\theta), 1-\epsilon, 1+\epsilon) \cdot A^g\right) - \beta \cdot D_{KL}(\pi_\theta, \pi_{ref})\right]$$

**Components**:
1. **Ratio term**: Same as PPO, but advantage $A^g$ is group-relative (no critic needed!)
2. **KL penalty**: Optional, with weight $\beta$. In bypass_mode, this is REMOVED entirely!
3. **Per-token average**: Normalized by $T$ (response length) — treats each token equally

### 4.3 bypass_mode: The RTX 4090 Game-Changer

When `bypass_mode=True` (verl CPPO/GRPO):
- **No reference model** $\pi_{ref}$ → save ~11 GiB!
- **No KL penalty** → simpler objective, fewer models needed
- **No critic** $V_\phi$ → save ~2 GiB!
- **Only policy model** on GPU → ~11 GiB → FITS in RTX 4090!

$$L^{bypass\_GRPO}(\theta) = \mathbb{E}_{\mathbf{x}, \{\mathbf{y}^g\}}\left[\frac{1}{G}\sum_{g=1}^{G}\frac{1}{T}\sum_{t=1}^{T}\min\left(\rho_t^g(\theta) \cdot A^g, \text{clip}(\rho_t^g(\theta), 1-\epsilon, 1+\epsilon) \cdot A^g\right)\right]$$

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**bypass_mode GRPO = RTX 4090 #1 BEST approach**: No reference model, no critic, just policy + group-relative advantages. Memory: policy ~11 GiB + optimizer CPU offload + KV cache ~4-6 GiB → total ~15-17 GiB → fits in 24 GiB!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 4.4 GRPO Group Size

The group size $G$ controls:
- **Variance reduction**: Larger $G$ → better baseline estimate → lower variance → more stable
- **Compute cost**: $G$ responses per prompt → $G$ times more rollout compute
- **Typical values**: $G = 8$ to $64$ (verl default: $G = 8$ for RTX 4090)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**rLLM #605 CRITICAL BUG**: `AgentWorkflowPPOTrainer` groups by `trajectory.uid` instead of prompt/agent key → each group has size 1 → $G = 1$ → NO baseline normalization → $A^i = (r^i - r^i)/0 = \text{UNDEFINED}$ → GRPO is BROKEN! This is why #605 has ZERO rewards — group size = 1 → advantages are all zero → gradient is zero → no learning!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. CPPO: Constrained Policy Optimization (verl #6731)

### 5.1 The Prefix Drift Problem

Standard PPO/GRPO uses a **uniform clip** $\epsilon$ for ALL tokens. But in autoregressive generation:
- Token $y_1$ affects all subsequent tokens → early tokens have outsized impact
- If $y_1$ diverges from reference → ALL tokens after will diverge → **cascading drift**
- Uniform clip doesn't prevent this → early tokens can drift freely → compound error

### 5.2 CPPO Solution: Position-Weighted Trust Region

CPPO uses a **position-dependent divergence threshold**:

$$D_t \leq \delta_t = w_{min} + (1 - w_{min}) \cdot \frac{T - t}{T - 1}$$

Where:
- $t$ = token position (1-indexed)
- $T$ = total response length
- $w_{min}$ = minimum weight for last token (e.g., 0.1)
- At $t=1$ (first token): $\delta_1 = w_{min} + (1 - w_{min}) \cdot 1 = 1$ — NO, wait...

Actually, the formula means:
- At $t=1$ (first token): $\delta_1 = w_{min} + (1-w_{min}) \cdot \frac{T-1}{T-1} = 1$ → **maximum divergence allowed**
- Wait, that's wrong. Let me re-derive...

The weight function $w_t$:
$$w_t = w_{min} + (1 - w_{min}) \cdot \frac{t-1}{T-1}$$

This gives:
- $w_1 = w_{min}$ → **first token: TIGHT constraint**
- $w_T = w_{min} + (1-w_{min}) \cdot 1 = 1$ → **last token: LOOSE constraint**

**Intuition**: Early tokens are constrained TIGHTER because their drift cascades. Later tokens are constrained LOOSER because they only affect a few remaining tokens.

### 5.3 CPPO Objective

$$L^{CPPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{g=1}^{G}\frac{1}{T}\sum_{t=1}^{T} A^g_t \cdot \min\left(\rho_t^g, 1 + w_t \cdot \epsilon\right) - \beta \cdot \sum_t w_t \cdot D_t\right]$$

Where the clip and KL penalty are BOTH position-weighted!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**CPPO = RTX 4090 #1 BEST algorithm**: Combines GRPO's no-critic advantage with principled position-weighted trust region → prevents prefix drift → better than both PPO and GRPO. verl #6731 MANDATORY for CPPO+bypass_mode.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. DAPO: Dynamic Advantage Policy Optimization

DAPO adds dynamic advantage normalization:

$$A^i_{DAPO} = \frac{r^i - \mu_G}{\max(\sigma_G, \sigma_{min})}$$

Where $\sigma_{min}$ is a floor to prevent division by near-zero std. This is essentially GRPO with a safety guard for small variance groups.

**verl DAPO config**: Same as GRPO but with `dynamic_advantage_normalization=True`.

---

## 7. Algorithm Comparison: Complete Mathematical Table

| Algorithm | Objective | Advantage | Trust Region | Critic Needed | Ref Model Needed | RTX 4090 Memory |
|-----------|-----------|-----------|--------------|---------------|-----------------|-----------------|
| **REINFORCE** | $r \cdot \nabla \log \pi$ | $A = r$ | None | No | No | policy only (~11 GiB) |
| **PPO** | $\min(\rho A, \text{clip}(\rho) A) - \beta D_{KL}$ | $A = r - V$ | Uniform clip $\epsilon$ | Yes (~2 GiB) | Yes (~11 GiB) | **NOT VIABLE** (24 GiB) |
| **GRPO** | $\min(\rho A^g, \text{clip}(\rho) A^g) - \beta D_{KL}$ | $A^g = \frac{r-\mu}{\sigma}$ | Uniform clip + KL | No | Yes (unless bypass) | ~22 GiB (with ref) |
| **GRPO+bypass** | $\min(\rho A^g, \text{clip}(\rho) A^g)$ | $A^g = \frac{r-\mu}{\sigma}$ | Uniform clip only | No | No! | **policy only (~11 GiB)** ✓ |
| **CPPO+bypass** | Position-weighted $\min(\rho A^g, \text{clip}_w(\rho) A^g)$ | $A^g = \frac{r-\mu}{\sigma}$ | Position-weighted ✓ | No | No! | **policy only (~11 GiB)** ✓ |
| **DAPO** | Same as GRPO + floor $\sigma_{min}$ | $A^g = \frac{r-\mu}{\max(\sigma, \sigma_{min})}$ | Uniform clip + KL | No | Yes (unless bypass) | same as GRPO |

---

## 8. Gradient Computation: Per-Layer Breakdown

### 8.1 Forward Pass (per training step)

For each response $\mathbf{y}^g$ sampled from $\pi_\theta$:

1. **Token-by-token generation**: $y_t^g \sim \pi_\theta(y_t | \mathbf{x}, y_1, ..., y_{t-1})$ — autoregressive
2. **Reward computation**: $r^g = r(\mathbf{x}, \mathbf{y}^g)$ — from reward model or rule-based
3. **Group advantage**: $A^g = \frac{r^g - \mu_G}{\sigma_G}$ — normalize across group
4. **Log-prob computation**: $\log \pi_\theta(\mathbf{y}^g | \mathbf{x}) = \sum_t \log \pi_\theta(y_t^g | ...)$

### 8.2 Backward Pass (gradient computation)

$$\nabla_\theta L = -\frac{1}{G} \sum_{g=1}^{G} \sum_{t=1}^{T} \nabla_\theta \hat{\rho}_t^g \cdot A^g$$

Where $\hat{\rho}_t^g$ is the clipped ratio:
$$\hat{\rho}_t^g = \begin{cases} \rho_t^g & \text{if } A^g \geq 0 \text{ and } \rho_t^g \leq 1+\epsilon \\ 1+\epsilon & \text{if } A^g \geq 0 \text{ and } \rho_t^g > 1+\epsilon \\ \rho_t^g & \text{if } A^g < 0 \text{ and } \rho_t^g \geq 1-\epsilon \\ 1-\epsilon & \text{if } A^g < 0 \text{ and } \rho_t^g < 1-\epsilon \end{cases}$$

**Memory during backward**: For each token in each response, we need:
- Activation of the Transformer block that produced that token
- Log-prob ratio $\rho_t^g$ and advantage $A^g$
- Gradients flow through ALL layers that contributed to that token's generation

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**verl #6699 CRITICAL**: Without `detach model_output`, activations from the rollout forward pass are ALSO kept → 4x memory waste! The fix: detach the model output after generation, so only the training forward pass's activations are retained for gradient computation.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 8.3 Gradient Accumulation across Group

$$\nabla_\theta L^{total} = -\frac{1}{G}\sum_{g=1}^{G}\nabla_\theta L^g$$

The gradients from ALL $G$ responses are averaged → group-level update → this is why group size matters!

---

## 9. Reward Function Design

### 9.1 Rule-Based Rewards (GRPO/CPPO with bypass)

Since bypass_mode has NO reference model → reward must be self-contained:

| Reward Type | Formula | Use Case | RTX 4090 Suitability |
|------------|---------|----------|---------------------|
| **Accuracy** | $r = \text{exact\_match}$ | Math reasoning | ★★★★★ Perfect |
| **Format** | $r = \text{has\_answer\_tag}$ | Structured output | ★★★★★ Perfect |
| **Length penalty** | $r = \text{accuracy} - \lambda \cdot \text{len}$ | Prevent verbosity | ★★★★★ Perfect |
| **Soft accuracy** | $r = \text{digit\_match\_ratio}$ | Numerical answers | ★★★★★ Good |
| **Reward model** | $r = R_\phi(\mathbf{x}, \mathbf{y})$ | General alignment | ★★★★★ Needs extra GPU for R_model! |

### 9.2 KL Penalty (when NOT using bypass)

$$D_{KL}(\pi_\theta, \pi_{ref}) = \mathbb{E}_t\left[\frac{\pi_\theta(y_t)}{\pi_{ref}(y_t)} \log \frac{\pi_\theta(y_t)}{\pi_{ref}(y_t)}\right]$$

This requires the reference model $\pi_{ref}$ to be loaded → **not viable on RTX 4090 dp=1**!

**verl bypass_mode**: Removes KL penalty entirely → relies on the clip mechanism alone for trust region → works well in practice when $\epsilon$ is well-chosen.

---

## 10. Training Step Breakdown: Memory and Compute

### 10.1 GRPO+bypass Training Step (Qwen3-8B on RTX 4090)

| Phase | GPU Memory | Duration | Notes |
|-------|-----------|----------|-------|
| 1. Rollout (generate G responses) | ~16 GiB (model) + ~4 GiB (KV) | ~5-10 min | sleep/wake cycle: release KV → generate |
| 2. Reward computation | ~2 GiB | ~1 min | Rule-based (no model needed) |
| 3. Advantage normalization | ~0.5 GiB | ~10 sec | CPU computation |
| 4. Training forward (re-compute log-probs) | ~16 GiB (model) + ~4 GiB (activations) | ~2-3 min | FSDP summon per-unit (#6512) |
| 5. Training backward | ~16 GiB + ~6 GiB (gradients) | ~3-5 min | Offloaded to CPU |
| 6. Optimizer step | ~3.8 GiB (CPU_Adam) | ~30 sec | CPU computation |
| 7. Weight sync (LoRA deltas) | ~0.3 GiB | ~10 sec | sleep_level=1, ~200 MiB transfer |

**Total per step**: ~15-20 min on RTX 4090 for Qwen3-8B with G=8, T=2048.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**sleep/wake cycle timing**: Rollout phase uses ~20 GiB, training phase uses ~22 GiB (with activations). The sleep/wake cycle (release KV → weight sync → training → resume KV) takes ~30 seconds overhead per step. sleep_level=1 (LoRA) reduces weight sync to ~200 MiB → ~10 sec. sleep_level=2 (merge) would need ~16 GiB transfer → ~2-3 min → MUCH slower!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 11. Connections to Framework Issues

| Math Concept | Framework Issue | Connection |
|-------------|----------------|------------|
| Advantage normalization | rLLM #605 GRPO grouping | Group size=1 → normalization undefined → BROKEN |
| Position-weighted trust region | verl #6731 CPPO | Position-dependent clip → prevents prefix drift |
| KL penalty removal | verl bypass_mode | Eliminates reference model → saves ~11 GiB |
| Critic elimination | GRPO group advantage | No critic needed → saves ~2 GiB |
| Gradient flow | verl #6699 detach | Rollout forward activations waste 4x memory |
| Per-token gradient | verl #6512 per-unit summon | 10x memory reduction for LoRA backward |
| Gradient clipping | DeepSpeed #8068 | Default 0→1.0 → MUST set explicitly |
| MoE gradient | Megatron #5394 | Global clipping stalls both AdamW and Muon |
| Loss scaling | DeepSpeed #8061 overlap_comm | NaN with overlap_comm=True on single GPU |

---

## References

1. Schulman et al. "Proximal Policy Optimization Algorithms" (2017) — PPO original paper
2. Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024) — GRPO original
3. verl #6731 — CPPO (Constrained Policy Optimization) implementation
4. rLLM documentation — Tinker framework, GRPO implementation
5. Williams "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992) — REINFORCE original
