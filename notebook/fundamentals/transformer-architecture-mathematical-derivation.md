# Transformer Architecture — Complete Mathematical Derivation

> 2026-06-19 | Algorithm Theory | From Attention to Full Transformer Block
> ★★★★★★★★ RTX 4090 Context: Understanding each layer's memory/compute ratio is CRITICAL for GRPO training optimization
> ★★★★★★★★ This note covers the MATH behind every Transformer component — not just "what it does" but "why it works"

---

## 1. Transformer Block: The Atomic Unit

Every modern LLM (GPT, LLaMA, Qwen, DeepSeek) is a stack of identical Transformer blocks. Understanding ONE block = understanding the entire model.

```
┌─────────────────────────────────────────────────────────┐
│               Transformer Block (Layer l)                │
│                                                         │
│  x_l ──→ LayerNorm ──→ Multi-Head Attention ──→ + ──→  │  ← residual #1
│          │                                            │  │
│          ↓                                            │  │
│         attn_output                                   │  │
│                                                         │
│  ──→ LayerNorm ──→ MLP (SwiGLU) ──→ + ──→ x_{l+1}    │  ← residual #2
│          │                                            │  │
│          ↓                                            │  │
│         mlp_output                                     │  │
└─────────────────────────────────────────────────────────┘

  x_{l+1} = x_l + Attention(LayerNorm(x_l))  + MLP(LayerNorm(x_l + Attention(LayerNorm(x_l))))
```

**Pre-norm vs Post-norm**: Modern models (LLaMA, Qwen, DeepSeek) use **Pre-norm** (LayerNorm BEFORE attention/MLP). Original Transformer used Post-norm (LayerNorm AFTER). Pre-norm = more stable gradients = better training = what we use today.

---

## 2. Multi-Head Self-Attention: Mathematical Derivation

### 2.1 Single-Head Attention

Given input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$ (n tokens, d model dimension):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

Where:
- $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{n \times d_k}$ — query projections
- $\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{n \times d_k}$ — key projections
- $\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{n \times d_v}$ — value projections
- $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$ — learned weight matrices
- $\sqrt{d_k}$ — scaling factor (prevents softmax saturation)

**Why $\sqrt{d_k}$ scaling?**: Without scaling, when $d_k$ is large, the dot products $\mathbf{Q}\mathbf{K}^\top$ have large variance ($\approx d_k$), pushing softmax into regions with very small gradients (near 0 or 1). Scaling by $\sqrt{d_k}$ normalizes variance to ~1, keeping gradients healthy.

### 2.2 Multi-Head Attention (MHA)

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}_O$$

Where each head:
$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^i, \mathbf{X}\mathbf{W}_K^i, \mathbf{X}\mathbf{W}_V^i)$$

**Dimension budget**: Typically $d_k = d_v = d/h$ (each head gets $d/h$ dimensions). This gives:
- Total parameters: $4 \times d \times d$ (W_Q, W_K, W_V, W_O) — same as single-head with full dimension!
- But each head operates in a smaller subspace → can focus on different semantic patterns

**RTX 4090 memory**: For Qwen3-8B with $d=4096, h=32$:
- Per-layer attention weights: $4 \times 4096 \times 4096 = 67$ MiB (BF16)
- KV cache per token: $2 \times 32 \times 128 \times 2$ bytes = 16 KiB (BF16)
- At 4096 context length: KV cache = 64 MiB per layer

### 2.3 Attention Variants (Modern LLMs)

| Variant | Model | Key Idea | KV Cache Savings | RTX 4090 Impact |
|---------|-------|----------|------------------|-----------------|
| **MHA** | GPT-2/3 | Full multi-head | None (baseline) | Baseline |
| **GQA** | LLaMA-2/3 | Grouped queries, K/V heads < Q heads | ~2-4x | ★★★★ Moderate savings |
| **MQA** | PaLM, Mistral | Single K/V head, multiple Q heads | ~h times | ★★★★★ Max savings |
| **MLA** | DeepSeek V2/V3/V4 | Latent compression of K/V | ★★★★★★★★ ~90% reduction! | ★★★★★★★★ GAME-CHANGER! |

**GQA (Grouped-Query Attention)**: LLaMA-2/3 uses $h_Q=32$ query heads but $h_K=h_V=8$ key/value groups. Each K/V group is shared by 4 Q heads. KV cache per token: $2 \times 8 \times 128 \times 2 = 4$ KiB (vs 16 KiB MHA → **4x savings**).

**MLA (Multi-head Latent Attention)**: DeepSeek compresses K/V into a latent vector $\mathbf{c}_{KV} \in \mathbb{R}^{d_c}$ where $d_c \ll d \times h$:

$$\mathbf{c}_{KV} = \mathbf{X}\mathbf{W}_{DKV} \in \mathbb{R}^{n \times d_c}$$
$$\mathbf{K} = \mathbf{c}_{KV}\mathbf{W}_{UK} \in \mathbb{R}^{n \times d_k \times h}$$
$$\mathbf{V} = \mathbf{c}_{KV}\mathbf{W}_{UV} \in \mathbb{R}^{n \times d_v \times h}$$

KV cache stores only $\mathbf{c}_{KV}$ (latent) instead of full K/V → **~90% reduction**. For DSV4-Flash: $d_c=512$ vs $d \times h = 4096 \times 128 = 524288$ → **1000x theoretical compression** (but actual implementation is more nuanced with RoPE and up-projection).

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**MLA is THE game-changer for RTX 4090**:
- Qwen3-30B-A3B uses MLA → KV cache ~90% smaller → enables longer context on 24 GiB
- DeepSeek V4 uses MLA → 8xL20 (SM89) can serve it with TP=8 (#28618)
- FlashMLA kernel is SM90-only → RTX 4090 uses Triton fallback → slightly slower but still viable
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. Position Encoding: From Sinusoidal to RoPE

### 3.1 Sinusoidal Positional Encoding (Original Transformer)

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Properties**: Fixed (not learned), relative position can be recovered via $\sin(\alpha-\beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta$, but this is approximate.

### 3.2 Learned Positional Encoding (GPT-2)

Simply learn $\mathbf{W}_{pos} \in \mathbb{R}^{max\_len \times d}$ as a regular parameter. Limited to max_len positions → can't extrapolate beyond training length.

### 3.3 RoPE (Rotary Position Embedding) — Modern Standard

**Used by**: LLaMA, Qwen, DeepSeek, Mistral — ALL modern LLMs.

$$\mathbf{q}_m = \mathbf{W}_Q \mathbf{x}_m \odot \mathbf{R}_m$$
$$\mathbf{k}_n = \mathbf{W}_K \mathbf{x}_n \odot \mathbf{R}_n$$

Where $\mathbf{R}_m$ is a rotation matrix that encodes position $m$:

$$\mathbf{R}_{\theta,m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

**Key property**: $\mathbf{q}_m^\top \mathbf{k}_n = f(m-n)$ — the dot product depends ONLY on relative position $m-n$, not absolute positions. This gives:

$$\text{Attention}(m, n) = \text{softmax}\left(\frac{\mathbf{q}_m^\top \mathbf{k}_n}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{f(m-n)}{\sqrt{d_k}}\right)$$

**RTX 4090 implications**:
- RoPE is applied to Q and K BEFORE attention computation → increases compute per attention step
- RoPE doesn't change KV cache size (it's applied at runtime, not stored)
- BUT: RoPE interpolation for longer contexts is critical (NTK-aware, YaRN, etc.)

---

## 4. MLP Layer: From ReLU to SwiGLU

### 4.1 Original MLP (ReLU)

$$\text{MLP}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

Parameters: $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$ → total $8d^2$ parameters per layer.

### 4.2 SwiGLU MLP (Modern Standard)

**Used by**: LLaMA, Qwen, DeepSeek, Mistral — ALL modern LLMs.

$$\text{SwiGLU}(\mathbf{x}) = (\text{SiLU}(\mathbf{W}_G \mathbf{x}) \odot \mathbf{W}_U \mathbf{x}) \mathbf{W}_D$$

Where:
- $\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1+e^{-x}}$ — sigmoid linear unit (aka Swish)
- $\mathbf{W}_G \in \mathbb{R}^{d \times d_{hidden}}$ — gate projection
- $\mathbf{W}_U \in \mathbb{R}^{d \times d_{hidden}}$ — up projection
- $\mathbf{W}_D \in \mathbb{R}^{d_{hidden} \times d}$ — down projection

**Parameter budget**: With $d_{hidden} = \frac{8}{3}d$ (rounded to nearest multiple of 256 for hardware alignment):
- $\mathbf{W}_G$: $d \times \frac{8}{3}d$
- $\mathbf{W}_U$: $d \times \frac{8}{3}d$
- $\mathbf{W}_D$: $\frac{8}{3}d \times d$
- Total: $\frac{24}{3}d^2 = 8d^2$ — same budget as ReLU MLP, but better performance!

**Why SwiGLU > ReLU**:
1. **SiLU is smooth** (no hard boundary at 0) → better gradient flow
2. **Gate mechanism** (multiplicative interaction) → more expressive than ReLU's simple threshold
3. **Empirical**: LLaMA paper shows SwiGLU consistently outperforms GeLU and ReLU variants

**RTX 4090 memory for Qwen3-8B** ($d=4096, d_{hidden}=11008 \approx \frac{8}{3} \times 4096$):
- Per-layer MLP weights: $3 \times 4096 \times 11008 \times 2$ bytes = 265 MiB (BF16)
- Total MLP for 32 layers: ~8.5 GiB — this is where MOST of the model weights live!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**MLP layer = the memory hog of Transformer models**:
- Attention weights: ~67 MiB per layer (4 × d²)
- MLP weights: ~265 MiB per layer (3 × d × d_hidden)
- MLP is ~4x larger than attention per layer!
- For LoRA: MLP layers have the biggest LoRA targets (gate/up/down projections)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 4.3 MoE MLP: Expert Routing

**Mixture of Experts** replaces the single MLP with multiple expert MLPs + a router:

$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{k} g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

Where:
- $g_i(\mathbf{x}) = \text{TopK}(\text{softmax}(\mathbf{W}_R \mathbf{x}))$ — router selects top-k experts
- $E_i(\mathbf{x}) = \text{SwiGLU}_i(\mathbf{x})$ — each expert is a separate SwiGLU MLP
- $k$ typically = 6 or 8 (DeepSeek V3/V4 uses top-8 out of 256+ experts)

**Memory implication**: Each expert has its own $\mathbf{W}_G, \mathbf{W}_U, \mathbf{W}_D$. For Qwen3-30B-A3B (128 experts, top-8):
- Per expert MLP: 3 × 768 × 3072 × 2 bytes ≈ 18 MiB (small! because active_dim=768)
- All 128 experts: ~2.3 GiB total
- BUT only top-8 active per token → compute is ~8 × 18 MiB = 144 MiB worth of compute
- This is why MoE models have many parameters but low FLOPs → "parameter-efficient inference"

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**MoE is PERFECT for RTX 4090**:
- Qwen3-30B-A3B: 30B total params but only 3B active per token → 6 GiB active params (fits in 24 GiB!)
- Expert params can be offloaded to CPU → CPU_Aadam + ZeRO-2 → training viable
- LoRA on active params only: rank=32 → ~60 MiB LoRA deltas → sleep_level=1 100x payload reduction
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. Layer Normalization: RMSNorm vs LayerNorm

### 5.1 LayerNorm (Original Transformer)

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \gamma + \beta$$

Where $\mu = \frac{1}{d}\sum x_i$, $\sigma = \sqrt{\frac{1}{d}\sum(x_i - \mu)^2}$.

Parameters: $\gamma, \beta \in \mathbb{R}^d$ (2d parameters per layer norm). Two per block → 4d parameters.

### 5.2 RMSNorm (Modern Standard — LLaMA/Qwen/DeepSeek)

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma$$

Where $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum x_i^2}$.

**Key difference**: No centering ($\mathbf{x} - \mu$), no bias ($\beta$). Just scale by RMS and multiply by learned gain.

**Why RMSNorm > LayerNorm**:
1. **Fewer parameters**: $d$ vs $2d$ per layer norm → 2d vs 4d per block
2. **Faster compute**: No mean subtraction, no bias addition → ~10-15% faster
3. **Same quality**: Empirically equivalent or slightly better (LLaMA paper)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**SGLang #24459**: `rms_norm` + `mm.dtype` constexpr override → KERNEL-level determinism! SGLang overrides `torch.nn.functional.rms_norm` with a custom Triton kernel that uses constexpr dtype to ensure consistent floating-point behavior across batch sizes. This is EXACTLY the SM89 batch invariance pattern → connects to P9 Fusion Guard thesis!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. Residual Connection and Gradient Flow

### 6.1 Residual Connection

$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\mathbf{x}_l)$$

Where $F_l$ is the sub-layer (attention or MLP with layer norm).

**Gradient flow**: In backward pass, gradient flows through BOTH paths:
$$\frac{\partial L}{\partial \mathbf{x}_l} = \frac{\partial L}{\partial \mathbf{x}_{l+1}} + \frac{\partial L}{\partial \mathbf{x}_{l+1}} \cdot \frac{\partial F_l}{\partial \mathbf{x}_l}$$

The first term $\frac{\partial L}{\partial \mathbf{x}_{l+1}}$ passes through unchanged (identity path) → **gradient highway**. This prevents gradient vanishing even with deep networks (100+ layers).

### 6.2 Pre-norm Advantage

With Pre-norm:
$$\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\text{Norm}(\mathbf{x}_l))$$

The residual path is $\mathbf{x}_l$ (unnormalized) → gradient flows through the RAW activations → even cleaner gradient highway.

**verl #6699 connection**: The `detach model_output` fix prevents the gradient from flowing through the WRONG path (through rollout engine's forward pass instead of the training forward pass). Without detach, gradient computes through TWO forward paths → 4x memory waste + incorrect gradient direction!

---

## 7. Full Transformer Block: Parameter Count

### 7.1 Per-Block Parameter Budget (LLaMA-style, SwiGLU + GQA + RMSNorm)

| Component | Parameters | Memory (BF16) | Notes |
|-----------|-----------|---------------|-------|
| Attention W_Q | $d \times d$ | $2d^2$ bytes | With GQA: actually $d \times (d/h \times h_Q)$ |
| Attention W_K | $d \times (d/h \times h_K)$ | varies | GQA: $h_K < h_Q$ |
| Attention W_V | $d \times (d/h \times h_K)$ | varies | Same as W_K |
| Attention W_O | $(d/h \times h_Q) \times d$ | $2d^2$ bytes | Output projection |
| MLP W_G | $d \times d_{hidden}$ | varies | Gate projection |
| MLP W_U | $d \times d_{hidden}$ | varies | Up projection |
| MLP W_D | $d_{hidden} \times d$ | varies | Down projection |
| RMSNorm (2x) | $2 \times d$ | $4d$ bytes | Gain only, no bias |

**Total per block**: $\approx 4d^2 + 3d \times d_{hidden} + 2d$

For Qwen3-8B ($d=4096, d_{hidden}=11008, h_Q=32, h_K=4$):
- Attention: $4 \times 4096^2 = 67$ MiB
- MLP: $3 \times 4096 \times 11008 = 265$ MiB
- RMSNorm: negligible (8 KiB)
- **Per block: ~332 MiB**
- **32 blocks total: ~10.6 GiB** (plus embedding ~128 MiB, lm_head ~128 MiB)
- **Total: ~11 GiB** (BF16) → fits in RTX 4090 24 GiB with headroom for KV cache + optimizer

### 7.2 LoRA Target Parameters

LoRA adds low-rank matrices $\mathbf{A} \in \mathbb{R}^{r \times d}$ and $\mathbf{B} \in \mathbb{R}^{d_{out} \times r}$:

$$\mathbf{W}' = \mathbf{W} + \mathbf{B}\mathbf{A}$$

For rank=32:
- Per LoRA target: $2 \times 32 \times (d + d_{out})$ parameters
- Attention targets (Q, K, V, O): $4 \times 2 \times 32 \times 4096 \times 2$ bytes ≈ 2 MiB per layer
- MLP targets (G, U, D): $3 \times 2 \times 32 \times (4096 + 11008) \times 2$ bytes ≈ 7 MiB per layer
- **Per block LoRA: ~9 MiB**
- **32 blocks: ~288 MiB total LoRA**

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**LoRA delta sync payload for sleep_level=1**: ~288 MiB vs ~10.6 GiB full model → **37x reduction** for Qwen3-8B. For MoE models (LoRA on active params only): even more dramatic savings.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. FLOPs Analysis: Training vs Inference

### 8.1 Training FLOPs (per token, per layer)

$$\text{FLOPs}_{train} \approx 2 \times (4d^2 + 3d \times d_{hidden}) \times n_{tokens} \times n_{layers} \times 3$$

The factor of 3 = forward (1x) + backward (2x, because gradient computation requires both $\frac{\partial L}{\partial \mathbf{W}}$ and $\frac{\partial L}{\partial \mathbf{x}}$).

### 8.2 Inference FLOPs (per token, per layer)

For autoregressive generation (1 token at a time):
$$\text{FLOPs}_{decode} \approx 2 \times (4d^2 + 3d \times d_{hidden})$$

This is per NEW token — KV cache avoids recomputing past tokens.

### 8.3 Prefill FLOPs (batch of n tokens)

$$\text{FLOPs}_{prefill} \approx 2 \times (4d^2 + 3d \times d_{hidden}) \times n$$

Plus attention: $2 \times n^2 \times d$ (quadratic in sequence length!).

---

## 9. Modern Architecture Comparison

| Feature | GPT-3/4 | LLaMA-2/3 | Qwen3-8B | Qwen3-30B-A3B | DeepSeek V4 |
|---------|---------|-----------|----------|---------------|-------------|
| Norm | LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Attention | MHA | GQA | GQA | GQA + MLA | MLA |
| MLP | GeLU | SwiGLU | SwiGLU | SwiGLU + MoE | SwiGLU + MoE |
| Position | Learned | RoPE | RoPE | RoPE | RoPE |
| Bias | Yes | No | No | No | No |
| d_model | 12288 | 4096 | 4096 | 2048 | ? |
| d_hidden | 4d | 8/3d | 8/3d | 8/3d (per expert) | 8/3d (per expert) |
| Num experts | 1 | 1 | 1 | 128 (top-8) | 256+ (top-8) |
| Active params | 175B | 8B | 8B | 3B | ~?B |
| KV per token | 16 KiB | 4 KiB | 4 KiB | ~0.4 KiB (MLA) | ~0.4 KiB (MLA) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**RTX 4090 GRPO ranking by model**:
1. Qwen3-30B-A3B (MoE + MLA): 3B active, ~6 GiB resident, LoRA ~60 MiB → BEST candidate
2. Qwen3-8B (dense): 8B, ~11 GiB resident, LoRA ~288 MiB → viable with CPU offload
3. DeepSeek V4 (MLA + MoE): requires SM90 FlashMLA or SM89 Triton fallback → needs #28618
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 10. Connections to Framework Issues

| Concept | Framework Issue | Connection |
|---------|----------------|------------|
| MLA KV compression | SGLang #28618 SM89 DSV4 | FlashMLa requires SM90 → Triton fallback on RTX 4090 |
| RMSNorm determinism | SGLang #24459 constexpr | KERNEL-level override for batch invariance |
| SwiGLU MoE routing | vLLM #45683 MoE combine | Deterministic routing critical for GRPO |
| MoE expert padding | vLLM #45309 eager_break | Dynamic routing = non-capturable → enforce_eager |
| Residual gradient | verl #6699 detach | Gradient flows through wrong path → 4x memory waste |
| Position encoding | Megatron #5384 Indexer Replay | RoPE state needs train/rollout consistency |
| LayerNorm params | DeepSpeed #8072 ZeRO-3+PEFT | Norm parameters mis-handled in ZeRO-3+LoRA |

---

## References

1. Vaswani et al. "Attention Is All You Need" (2017) — Original Transformer
2. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023) — SwiGLU, RMSNorm, RoPE
3. DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient MoE" (2024) — MLA architecture
4. Shazeer "GLU Variants Improve Transformer" (2020) — SwiGLU theory
5. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) — RoPE theory
