# Advanced Optimizer Theory: Adam, LAMB, Muon — Mathematical Derivation

> 2026-06-19 | Algorithm Theory | From SGD to Adam to CPU_Adam to Muon: The Math Behind LLM Training Optimizers
> ★★★★★★★★ RTX 4090 Context: CPU_Adam (18Ψ→3.8Ψ) is the ONLY viable optimizer for single GPU GRPO → Muon BLOCKED by 6 blockers
> ★★★★★★★★ Gradient clipping MUST be 1.0 (#8068) — default 0→1.0 → ALWAYS set explicitly!

---

## 1. SGD: The Foundation

### 1.1 Vanilla SGD

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

Where $g_t = \nabla_\theta L(\theta_t)$ — gradient of loss at step $t$.

**Problems**:
- High variance in gradient estimates (especially with small batches)
- Same learning rate for ALL parameters → suboptimal for different gradient scales
- No momentum → oscillates in narrow valleys of loss landscape

### 1.2 SGD with Momentum

$$v_{t+1} = \mu \cdot v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

Where $\mu$ is the momentum coefficient (typically 0.9).

**Nesterov momentum**: Look ahead before computing gradient:
$$v_{t+1} = \mu \cdot v_t + \nabla_\theta L(\theta_t - \eta \cdot \mu \cdot v_t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_{t+1}$$

---

## 2. Adam: Adaptive Moment Estimation

### 2.1 Full Algorithm

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$  — first moment (mean)
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$  — second moment (variance)
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$  — bias-corrected first moment
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$  — bias-corrected second moment
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$  — parameter update

**Where**:
- $\beta_1 = 0.9$ — exponential decay for first moment
- $\beta_2 = 0.999$ — exponential decay for second moment
- $\epsilon = 10^{-8}$ — numerical stability (prevents division by zero)
- $\eta$ — learning rate

### 2.2 Why Adam Works for LLM Training

1. **Per-parameter adaptive learning rate**: Each parameter gets its own effective LR = $\eta / (\sqrt{\hat{v}_t} + \epsilon)$. Parameters with large historical gradients get smaller LR (they've moved enough). Parameters with small gradients get larger LR (they need more movement).

2. **Bias correction**: Early steps have biased moment estimates (too small). Correction $\hat{m}_t = m_t / (1-\beta_1^t)$ ensures unbiased estimates even in step 1.

3. **Invariant to gradient scale**: The normalization by $\sqrt{\hat{v}_t}$ makes the update size approximately $\eta$ regardless of gradient magnitude → can use same LR for all layers!

### 2.3 AdamW: Decoupled Weight Decay

**Standard Adam + L2 regularization**:
$$L_{regularized} = L + \lambda \|\theta\|^2$$
$$g_t = \nabla_\theta L + \lambda \theta$$  — gradient includes regularization

**Problem**: Regularization gradient is added to $m_t$ → momentum accumulates regularization → effective weight decay depends on LR and momentum schedule → NOT what we want!

**AdamW fix**: Apply weight decay DIRECTLY to parameters, not through gradient:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda \cdot \theta_t$$

This decouples weight decay from the adaptive LR mechanism → true weight decay regardless of gradient scale.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**ALL modern LLM training uses AdamW** (not Adam). The decoupled weight decay is critical for large-scale training. verl, rLLM, DeepSpeed, Megatron all use AdamW. The default weight decay is 0.01 (LLaMA recommendation).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. CPU_Adam: The RTX 4090 Optimizer

### 3.1 Why CPU_Adam

Standard Adam optimizer state per parameter:
- $m_t$ (first moment): same size as parameter → $N$ values
- $v_t$ (second moment): same size as parameter → $N$ values
- $\hat{m}_t, \hat{v}_t$: computed on the fly, not stored

**Memory per parameter**: 1 param (BF16, 2 bytes) + 1 $m_t$ (FP32, 4 bytes) + 1 $v_t$ (FP32, 4 bytes) = 10 bytes

**For Qwen3-8B** ($N = 8 \times 10^9$):
- Parameters: $8 \times 10^9 \times 2 = 16$ GiB (BF16)
- Adam states: $8 \times 10^9 \times 8 = 64$ GiB (FP32) → **OVER 2x MODEL SIZE!**
- Total: 80 GiB → EXCEEDS RTX 4090 by 3.3x!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Adam optimizer states = THE memory bottleneck**:
- 8B model: 16 GiB params + 64 GiB Adam states = 80 GiB → impossible on RTX 4090
- Solution: CPU_Adam — move optimizer states to CPU memory → 64 GiB on CPU RAM (typically 64+ GiB) → GPU only needs parameters + gradients temporarily → 18Ψ→3.8Ψ!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.2 CPU_Adam Memory Flow

```
Training Step (with CPU_Adam):

1. Forward pass: params on GPU → compute loss → activations on GPU
2. Backward pass: compute gradients on GPU → offload gradients to CPU
3. Optimizer step (ON CPU!):
   a. CPU: params_fp32 + m + v → Adam update → new params_fp32
   b. CPU→GPU: copy updated params (BF16) back to GPU
4. Weight sync: LoRA deltas → rollout server (if HYBRID mode)
```

**Memory at each stage**:
| Stage | GPU Memory | CPU Memory |
|-------|-----------|-----------|
| Forward | ~22 GiB (params + activations) | ~64 GiB (optimizer states) |
| Backward | ~22 GiB (params + grad buffers) | ~64 GiB (optimizer states) |
| Optimizer | ~16 GiB (params only, grad released) | ~72 GiB (states + new params) |
| After sync | ~16 GiB (params only) | ~72 GiB |

**ZeRO-2 offload**: Reduces GPU memory from 80 GiB to ~22 GiB → fits in RTX 4090!

### 3.3 Why NOT CPU_SGD or CPU_LAMB?

**LAMB**: Large Batch Adam variant — scales trust region for large batches. Designed for distributed training with batch sizes >8K. For RTX 4090 (batch size ~2-4 per step), LAMB has no advantage → same as AdamW.

**SGD**: Requires manual LR tuning per layer → impractical for 8B+ models with hundreds of layers. AdamW's adaptive LR is essential for LLM training quality.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSpeed #8058 ZenFlow**: NEW CPU optimizer that reduces memory from 2944→256 MiB! Uses compressed optimizer states (8-bit quantization for m and v). This could further reduce CPU_Adam's 64 GiB requirement to ~16 GiB → even more headroom on RTX 4090. Currently under review (delock reviewing).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. Muon Optimizer: Momentum + Orthogonal Normalization

### 4.1 What is Muon?

**Muon** = Momentum + Newton-schur Orthogonalized gradient update:

$$\text{Muon update} = \text{momentum} \cdot \text{whiten}(g_t) \cdot \text{PS}(g_t)$$

Where:
- $\text{whiten}(g_t) = g_t / \|g_t\|$ — normalize gradient to unit norm
- $\text{PS}(g_t)$ — preconditioning via matrix square root (Newton-Schulz iteration)
- Momentum: standard exponential moving average

**Key claim**: Muon converges faster than Adam for LLM training, with fewer optimizer state parameters.

### 4.2 Muon Memory

Muon stores only:
- Momentum: $N$ FP32 values → $4N$ bytes
- No second moment ($v_t$)!

**For Qwen3-8B**: $8 \times 10^9 \times 4 = 32$ GiB → 2x reduction vs Adam (64 GiB)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Muon is BLOCKED on RTX 4090 by 6 blockers**:
1. #7939: Muon+CPU_offload closed without merge → Muon+CPU_Adam NOT viable
2. #7776/#8068: gradient clipping default → MUST skip global clipping for Muon
3. #7878: reduce_scatter gradient aggregation incompatible
4. Megatron #5394/#5395/#5400/#5179: ChainedOptimizer clipping stalls Muon too
5. Megatron #5179: Muon package is placeholder stub on PyPI → can't even install!
6. AdamW ALSO stalls under global clipping (#5394 breakthrough) → optimizer-agnostic bug

**Conclusion**: CPU_Adam remains ONLY viable optimizer for RTX 4090 GRPO. Muon is theoretically interesting but practically blocked.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. Gradient Clipping: The Critical Safety Mechanism

### 5.1 Global Gradient Norm Clipping

$$\hat{g}_t = \frac{g_t}{\max(1, \frac{\|g_t\|}{c})}$$

Where $c$ is the clipping threshold (typically 1.0 for GRPO).

**Effect**: If total gradient norm exceeds $c$, all gradients are scaled down proportionally. This prevents catastrophic updates from anomalous gradient spikes.

### 5.2 Why Clipping Matters for GRPO

GRPO advantages can be very large (group with 1 very good response vs rest → advantage >> 1). Without clipping, the gradient from one very good response could dominate → destabilize training → potential NaN.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSpeed #8068**: gradient_clipping default changed from 0→1.0 → previously, clipping was disabled (threshold=0 → no clipping). This is DANGEROUS for GRPO → MUST always set `gradient_clipping=1.0` explicitly! Do NOT rely on the new default — be explicit!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 5.3 Per-Parameter vs Global Clipping

**Global clipping**: Scale ALL gradients by same ratio → preserves gradient direction.

**Per-parameter clipping**: Clip each parameter's gradient independently → changes gradient direction → can hurt convergence.

**Megatron #5395**: `skip_grad_norm_clip` attribute (+15/-1 lines) → allows Muon/AdamW to skip global clipping → currently 0 reviews → stalled!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Megatron #5394 BREAKTHROUGH**: AdamW ALSO stalls under global grad-norm clipping (not just Muon!). This is an optimizer-agnostic bug — the clipping stalls ANY optimizer that has momentum-based updates. The controlled experiments confirm this → volunteer stepping up to fix → #5395 is the fix PR (+15/-1, 0 reviews!).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. Learning Rate Schedules

### 6.1 Common Schedules

| Schedule | Formula | Characteristics | Typical for LLM |
|----------|---------|----------------|----------------|
| **Constant** | $\eta_t = \eta_0$ | Simple, no decay | Rarely used |
| **Step decay** | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$ | Periodic reduction | Rarely used |
| **Cosine decay** | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_0 - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$ | Smooth, gradual | ★★★★★★★★ MOST COMMON |
| **Linear warmup + cosine decay** | warmup: $\eta_t = \eta_0 \cdot \frac{t}{W}$ for $t \leq W$, then cosine | Prevent early instability | ★★★★★★★★ DEFAULT in verl/LLaMA |
| **Warmup stable decay** | warmup→constant→decay | Three phases | DeepSpeed/Megatron default |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Cosine decay with linear warmup is THE standard for LLM training**:
- Warmup phase (first 5-10% of steps): Gradually increase LR from 0 to $\eta_0$ → prevents early gradient explosion
- Cosine decay phase: Smoothly decrease from $\eta_0$ to $\eta_{min}$ → ensures convergence
- $\eta_{min}$ typically = $\eta_0 / 10$ or $\eta_0 / 20$
- verl default: warmup_steps=10, max_lr=1e-6 for LoRA GRPO

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 6.2 LoRA Learning Rate Considerations

For LoRA GRPO on RTX 4090:
- **Base model LR**: 0 (base model frozen during LoRA training!)
- **LoRA LR**: Higher than full-model LR → typically 1e-4 to 1e-5
- **Warmup**: Same structure (linear warmup + cosine decay)
- **Scaling**: LoRA alpha/rank ratio (alpha=64, rank=32 → effective LR = LR × alpha/rank = LR × 2)

---

## 7. Optimizer Memory Comparison: RTX 4090

| Optimizer | States per param | Total states (8B) | GPU (with offload) | CPU needed | RTX 4090 viable? |
|-----------|-----------------|-------------------|-------------------|-----------|------------------|
| **SGD** | 0 (no states) | 0 | ✓ | 0 | ★★★★★★★★ Yes (but poor convergence) |
| **AdamW** | 2 (m, v FP32) | 64 GiB | ✓ (CPU_Adam) | 64 GiB | ★★★★★★★★ Yes (CPU_Adam) |
| **AdamW (ZenFlow)** | 2 (quantized 8-bit) | ~16 GiB | ✓ | ~16 GiB | ★★★★★★★★ Yes (even better!) |
| **Muon** | 1 (momentum FP32) | 32 GiB | ❌ (BLOCKED) | 32 GiB | ❌ BLOCKED! |
| **LAMB** | 2 (m, v FP32) | 64 GiB | ✓ | 64 GiB | ★★★ Yes (but no advantage on small batch) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**RTX 4090 GRPO optimizer hierarchy**:
- ★★★★★★★★ CPU_Adam (ZeRO-2 + param_offload + grad_offload) — ONLY proven viable optimizer
- ★★★★★★★★ ZenFlow (future, #8058 under review) — 2944→256 MiB, same quality
- ❌ Muon — 6 blockers, NOT viable for single GPU
- ❌ SGD — poor convergence for 8B+ models
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. verl Optimizer Integration

### 8.1 verl Optimizer Config (ppo_trainer.yaml)

```yaml
actor:
  optimizer:
    type: cpu_adam  # CPU_Adam → 18Ψ→3.8Ψ
    lr: 1e-6       # LoRA learning rate
    weight_decay: 0.01  # AdamW decoupled weight decay
    grad_clip: 1.0      # ★★★★★ MUST set explicitly! (#8068 default 0→1.0)
    warmup_steps: 10    # Linear warmup
    lr_scheduler: cosine  # Cosine decay
```

### 8.2 verl #6765: Per-Step Optimizer Overrides (MERGED June 18)

Allows changing optimizer parameters PER STEP:
- Step-specific LR → enables Muon LR scheduling (even though Muon itself is blocked)
- Step-specific weight_decay → enables curriculum-based regularization
- Step-specific grad_clip → enables adaptive clipping

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**verl #6765 enables future Muon integration**: Even though Muon is blocked now, the per-step override infrastructure is ready. When Muon blockers are resolved (#5395 skip_grad_norm_clip, #7939 CPU_offload), verl can immediately support Muon with LR scheduling.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 9. Connections to Framework Issues

| Optimizer Concept | Framework Issue | Connection |
|------------------|----------------|------------|
| CPU_Adam | DeepSpeed ZeRO-2 | 18Ψ→3.8Ψ, ONLY viable single GPU optimizer |
| Gradient clipping | DeepSpeed #8068 | Default 0→1.0, MUST set 1.0 explicitly |
| Muon BLOCKED | #7939, #7776, #7878, #5394, #5395, #5179 | 6 blockers → NOT viable |
| AdamW + clipping | Megatron #5394 | BREAKTHROUGH: AdamW ALSO stalls → optimizer-agnostic |
| skip_grad_norm | Megatron #5395 | +15/-1, 0 reviews, stalled! |
| ZenFlow | DeepSpeed #8058 | 2944→256 MiB, delock reviewing |
| Per-step overrides | verl #6765 | MERGED → enables future Muon LR scheduling |
| Weight decay | AdamW vs Adam | Decoupled decay → essential for LLM training |
| Warmup schedule | verl ppo_trainer.yaml | Linear warmup + cosine decay → standard |

---

## References

1. Kingma & Ba "Adam: A Method for Stochastic Optimization" (2015) — Adam original
2. Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2019) — AdamW
3. Jordan et al. "Muon: An optimizer for hidden layers in neural networks" (2024) — Muon optimizer
4. You et al. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (2019) — LAMB
5. DeepSpeed #8058 — ZenFlow CPU optimizer (2944→256 MiB)
6. verl #6765 — Per-step optimizer overrides
