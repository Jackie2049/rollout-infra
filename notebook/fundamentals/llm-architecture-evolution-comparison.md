# LLM Architecture Evolution: GPT → LLaMA → Qwen → DeepSeek — Design Comparison

> 2026-06-19 | Algorithm Theory | Understanding WHY each architecture made specific design choices
> ★★★★★★★★ RTX 4090 Context: Architecture choices directly determine memory/compute requirements → GRPO viability depends on architecture!

---

## 1. Evolution Timeline

```
GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) → GPT-4 (2023)
    ↓
LLaMA-1 (2023) → LLaMA-2 (2023) → LLaMA-3 (2024)
    ↓
Qwen-1 (2023) → Qwen-2 (2024) → Qwen-3 (2025)
    ↓
DeepSeek-V1 (2024) → DeepSeek-V2 (2024) → DeepSeek-V3 (2024) → DeepSeek-V4 (2025)
```

Each generation introduced KEY architectural innovations that changed the RTX 4090 viability landscape.

---

## 2. Architecture Feature Comparison Matrix

| Feature | GPT-3/4 | LLaMA-2/3 | Qwen-3 | DeepSeek-V4 |
|---------|---------|-----------|---------|-------------|
| **Normalization** | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| **Position Encoding** | Learned | RoPE | RoPE | RoPE |
| **Attention Type** | MHA | GQA | GQA | MLA |
| **MLP Type** | GeLU | SwiGLU | SwiGLU | SwiGLU |
| **Bias Terms** | Yes | No | No | No |
| **MoE** | No (dense) | No (dense) | Optional (30B-A3B) | Yes (256+ experts) |
| **KV Compression** | No | No | GQA savings | MLA ~90% savings |
| **Weight Quantization** | BF16 | BF16 | BF16 | FP8 W8A8 |
| **d_model** | 12288 | 4096 | 4096 | ? |
| **n_heads (Q)** | 96 | 32 | 32 | ? (MLA) |
| **n_heads (K/V)** | 96 | 8 (GQA) | 4 (GQA) | latent (MLA) |
| **n_layers** | 96 | 32 | 32 | ? |
| **Context Length** | 4K→128K | 4K→128K | 32K | 128K+ |
| **Active Parameters** | 175B | 8B | 3B (MoE) | ~?B |
| **KV per token (BF16)** | 768 KiB | 16 KiB | 4 KiB (GQA) | ~0.4 KiB (MLA) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**KV per token is THE critical metric for RTX 4090 GRPO**:
- GPT-3: 768 KiB → 128K context = 96 GiB KV → IMPOSSIBLE on RTX 4090
- LLaMA-2: 16 KiB → 128K context = 2 GiB KV → viable with FP8
- Qwen-3 GQA: 4 KiB → 128K context = 0.5 GiB KV → excellent
- DeepSeek MLA: 0.4 KiB → 128K context = 50 MiB KV → INCREDIBLE! RTX 4090 GRPO champion
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. GPT Architecture: The Foundation

### 3.1 Key Design Choices

| Choice | Why | Impact |
|--------|-----|--------|
| **LayerNorm** | Original Transformer standard | Stable but slower than RMSNorm |
| **Learned PE** | Simple, effective for fixed-length | Can't extrapolate beyond training length |
| **MHA** | Full multi-head attention | Maximum attention flexibility |
| **GeLU MLP** | Smooth activation function | Good but SwiGLU outperforms |
| **Bias = Yes** | Traditional design | Extra parameters (negligible) |
| **d_model = 12288** | Scale-dependent | Very large → requires multi-GPU |

**GPT's weaknesses for RTX 4090**:
- MHA KV cache: 768 KiB per token → even 4K context = 3 GiB KV per layer → total ~300 GiB → impossible
- No weight sharing: 175B dense model → 350 GiB BF16 → impossible
- Learned position encoding: Fixed max_len → can't extend context dynamically

---

## 4. LLaMA Architecture: The Efficiency Revolution

### 4.1 Key Innovations (vs GPT)

| Innovation | Change | Why | Memory Impact | RTX 4090 Impact |
|-----------|--------|-----|-------------|----------------|
| **RMSNorm** | LayerNorm→RMSNorm | Faster, fewer params, same quality | 2x→1x norm params | Negligible (<8 KiB per layer) |
| **RoPE** | Learned→RoPE | Relative position, extrapolation | No PE table needed | ★★★★★★★★ Critical for long context |
| **GQA** | MHA→GQA (8 KV groups) | KV cache reduction, minimal quality loss | 4x KV savings | ★★★★★★★★ Enables longer context |
| **SwiGLU** | GeLU→SwiGLU | Better expressiveness, same param budget | Same (8d²) | ★★★★ Better quality per param |
| **No bias** | bias→no bias | LLaMA paper: removing bias doesn't hurt | Small savings | Negligible |

### 4.2 LLaMA-2 vs LLaMA-3

| Feature | LLaMA-2 7B | LLaMA-3 8B |
|---------|-----------|-----------|
| d_model | 4096 | 4096 |
| n_heads Q | 32 | 32 |
| n_heads KV | 32 (MHA) | 8 (GQA) ★★★★★★★★ |
| d_head | 128 | 128 |
| Context | 4096 | 8192 |
| Vocabulary | 32000 | 128256 ★★★★★★★★ |
| Training data | 2T tokens | 15T tokens |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**LLaMA-3's two key changes**: GQA (4x KV savings) + 4x larger vocabulary (128K vs 32K). The larger vocabulary means the embedding layer is now ~128K × 4096 × 2 bytes = 1 GiB → much larger! But this improves multilingual understanding significantly.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. Qwen Architecture: The MoE Pioneer

### 5.1 Qwen-3 Key Features

| Feature | Qwen-3 8B (dense) | Qwen-3 30B-A3B (MoE) |
|---------|-------------------|---------------------|
| d_model | 4096 | 2048 |
| n_heads Q | 32 | 16 |
| n_heads KV | 4 (GQA) | 4 (GQA) |
| n_experts | 1 | 128 |
| top-k experts | 1 | 8 |
| active params | 8B | 3B |
| total params | 8B | 30B |
| LoRA targets | All linear | Active experts only |
| LoRA rank=32 payload | ~288 MiB | ~60 MiB ★★★★★★★★ |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Qwen-3 30B-A3B is the BEST RTX 4090 GRPO candidate**:
- Only 3B active params per token → ~6 GiB BF16 → fits in 24 GiB with headroom
- LoRA on active params only: rank=32 → ~60 MiB deltas → sleep_level=1 100x payload reduction
- Expert params (27B) can be offloaded to CPU → CPU_Adam + ZeRO-2
- GQA KV: ~4 KiB per token → 4K context = ~16 MiB → very manageable
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 5.2 Qwen MoE Architecture Detail

```
┌─────────────────────────────────────────────────────────┐
│               Qwen-3 30B-A3B MoE Block                  │
│                                                         │
│  x_l ──→ RMSNorm ──→ Attention (GQA) ──→ + ──→        │  ← residual #1
│                                                         │
│  ──→ RMSNorm ──→ Router ──→ Top-8 experts ──→ + ──→   │  ← residual #2
│                    │                                    │
│                    ├─→ E_1 (SwiGLU MLP)                  │
│                    ├─→ E_2 (SwiGLU MLP)                  │
│                    ├─→ ...                               │
│                    ├─→ E_8 (SwiGLU MLP)                  │
│                    │                                    │
│                    └─→ E_9...E_128 (inactive, on CPU)    │
│                                                         │
│  Router output: g_i(x) = softmax(W_R x) top-8         │
│  MoE output: Σ g_i * E_i(x) for i in top-8            │
└─────────────────────────────────────────────────────────┘
```

---

## 6. DeepSeek Architecture: MLA + MoE + FP8

### 6.1 DeepSeek-V2: MLA Introduction

**MLA (Multi-head Latent Attention)** — the KEY innovation:

Instead of storing full K/V per token, compress into a latent vector:

$$\mathbf{c}_{KV} = \mathbf{W}_{DKV} \mathbf{x} \in \mathbb{R}^{d_c}$$

Then up-project at runtime:
$$\mathbf{K} = \mathbf{c}_{KV} \mathbf{W}_{UK}$$
$$\mathbf{V} = \mathbf{c}_{KV} \mathbf{W}_{UV}$$

**KV cache savings**: Store $\mathbf{c}_{KV}$ instead of $\mathbf{K}, \mathbf{V}$ → from $2 \times h \times d_{head}$ to $d_c$ → **~90% reduction**!

For DSV4: $d_c = 512$ → KV per token = $2 \times 512 \times 2$ bytes = 2 KiB (vs 16 KiB MHA → **8x savings**, vs 4 KiB GQA → **2x savings**).

### 6.2 DeepSeek-V3/V4: MLA + MoE + FP8

| Feature | DSV3 | DSV4 |
|---------|------|------|
| MLA | Yes | Yes (improved) |
| MoE | 256 experts, top-8 | 256+ experts, top-8 |
| FP8 quantization | W8A8 E4M3/E5M2 | W8A8 E4M3/E5M2 |
| Online compression | C128 compression | C128 + MTP |
| Total params | 671B | 685B |
| Active params | ~37B | ~?B |
| Context | 128K | 128K+ |
| MTP (Multi-Token Prediction) | No | Yes ★★★★★★★★ |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DSV4 is NOT viable on RTX 4090** (685B total = 343 GiB FP8 > 24 GiB). But the MLA/MoE/FP8 architectural principles ARE applicable to smaller models. DSV2-Lite (16B) with MLA can run on RTX 4090 with FP8 → deployment pathway opened by #28620!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 6.3 MTP (Multi-Token Prediction)

DSV4 introduces **MTP** — predicting multiple tokens per step:

$$\mathbf{y}_{t+1}, \mathbf{y}_{t+2}, ..., \mathbf{y}_{t+k} = \text{MTP\_heads}(\mathbf{h}_t)$$

This is related to **speculative decoding** — the MTP heads generate candidate future tokens, which are then verified in a single forward pass. This can improve throughput by 2-3x for autoregressive generation.

**DSV4 instability connection**: MTP introduces per-step dynamic state (which MTP heads are active, what candidates are generated) → same pattern as MoE routing → CANNOT be cached in CUDA graph → enforce_eager=True MANDATORY.

---

## 7. Architecture→RTX 4090 Viability Matrix

| Model | Architecture | BF16 Memory | FP8 Memory | Active Params | KV (4K ctx) | RTX 4090 GRPO? |
|-------|-------------|------------|-----------|--------------|-------------|----------------|
| **LLaMA-2 7B** | GQA+SwiGLU+RoPE | 14 GiB | 7 GiB | 7B | 16 MiB | ★★★★★★★★ Viable (BF16) |
| **LLaMA-3 8B** | GQA+SwiGLU+RoPE | 16 GiB | 8 GiB | 8B | 8 MiB | ★★★★★★★★ Viable (BF16) |
| **Qwen-3 8B** | GQA+SwiGLU+RoPE | 16 GiB | 8 GiB | 8B | 8 MiB | ★★★★★★★★ Viable (BF16) |
| **Qwen-3 30B-A3B** | GQA+MoE+SwiGLU | ~60 GiB | ~30 GiB | 3B | 4 MiB | ★★★★★★★★ BEST (MoE+LoRA+offload) |
| **DSV2-Lite 16B** | MLA+MoE+SwiGLU | ~32 GiB | ~16 GiB | ~2.4B | 2 MiB | ★★★★★ Viable (FP8+#28620) |
| **DSV4 685B** | MLA+MoE+FP8+MTP | ~1300 GiB | ~343 GiB | ~37B | 50 MiB | ❌ NOT viable (memory) |
| **GPT-3 175B** | MHA+GeLU | ~350 GiB | ~175 GiB | 175B | 3 GiB | ❌ NOT viable |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**RTX 4090 GRPO training ranking (by architecture)**:
1. ★★★★★★★★ **Qwen-3 30B-A3B**: MoE + GQA + LoRA → 3B active → ~60 MiB LoRA → sleep_level=1 → BEST
2. ★★★★★★★★ **Qwen-3 8B**: Dense + GQA + LoRA → ~288 MiB LoRA → sleep_level=1 → good baseline
3. ★★★★★ **DSV2-Lite**: MLA + MoE → requires FP8 + Triton fallback → experimental
4. ❌ **DSV4/DSV3**: Too large for RTX 4090 → needs multi-GPU or disaggregated inference
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. Key Design Principles: Why Each Choice Was Made

### 8.1 RMSNorm vs LayerNorm

**Why RMSNorm**: LLaMA paper found that removing the centering ($\mathbf{x} - \mu$) and bias ($\beta$) from LayerNorm doesn't hurt quality but:
- Eliminates 2d parameters per norm (saves ~32 KiB per 4096-dim layer)
- ~10-15% faster computation (no mean subtraction, no bias addition)
- Equivalent quality in practice

### 8.2 RoPE vs Learned Position

**Why RoPE**: Three advantages:
1. **Relative position**: $\mathbf{q}_m^\top \mathbf{k}_n = f(m-n)$ → natural length extrapolation
2. **No max_len limit**: Learned PE has fixed $\mathbf{W}_{pos} \in \mathbb{R}^{max\_len \times d}$ → can't extend beyond
3. **Interpolation**: NTK-aware and YaRN methods enable longer context without retraining

### 8.3 GQA vs MHA

**Why GQA**: KV cache is the bottleneck for long-context serving. GQA reduces KV heads from 32 to 8 → 4x KV savings. Quality impact: LLaMA-3 paper shows GQA is nearly equivalent to MHA for most tasks.

### 8.4 SwiGLU vs GeLU

**Why SwiGLU**: GLU paper shows that multiplicative gating (SwiGLU = SiLU × linear) outperforms simple activations (GeLU, ReLU). Same parameter budget ($8d^2$ vs $8d^2$) → better quality per parameter.

### 8.5 MLA vs GQA

**Why MLA**: DeepSeek's MLA goes further than GQA — instead of reducing KV head count, compress the ENTIRE KV representation into a latent vector. This gives:
- ~90% KV cache reduction (vs GQA's ~75% reduction)
- Latent compression preserves more information than simply reducing heads
- Can reconstruct full KV when needed (no quality loss)

### 8.6 MoE vs Dense

**Why MoE**: For large-scale models, MoE provides:
- **Parameter efficiency**: 30B total but 3B active → same compute as 3B dense model
- **Specialization**: Each expert can specialize in different domains (math, code, language)
- **Scalability**: Adding experts scales capacity without scaling compute

**RTX 4090 benefit**: MoE models have many parameters but few active → can offload inactive experts to CPU → only active params need GPU → enables training much "larger" models on constrained hardware.

---

## 9. Connections to Framework Issues

| Architecture Feature | Framework Issue | Connection |
|---------------------|----------------|------------|
| MLA KV compression | SGLang #28618 SM89 DSV4 | FlashMLa requires SM90 → Triton fallback on RTX 4090 |
| GQA KV savings | vLLM BudgetRefiner SLO | KV budget determines prefill chunk size |
| MoE expert routing | vLLM #45309/#45863 DSV4 reverts | Dynamic routing = non-capturable → enforce_eager=True |
| MTP multi-token | SGLang #28591/#28612 DSV4 | MTP state management lifecycle bugs |
| FP8 quantization | PyTorch #184119 P9 guard | FP8→BF16 fusion creates batch-dependent kernels on SM89 |
| MoE quantization | vLLM #45656 | is_sym guard regression for quantized MoE |
| SwiGLU MLP | Triton dequant_swiglu_quant | P6 contribution: fused MoE SwiGLU kernel |
| RMSNorm determinism | SGLang #24459 | constexpr override → kernel-level batch invariance |

---

## References

1. Brown et al. "Language Models are Few-Shot Learners" (GPT-3, 2020)
2. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023)
3. Touvron et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (2023)
4. Bai et al. "Qwen Technical Report" (Qwen series, 2023-2025)
5. DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024)
6. DeepSeek-AI "DeepSeek-V3 Technical Report" (2024)
7. Shazeer "GLU Variants Improve Transformer" (2020)
8. Su et al. "RoFormer" (RoPE, 2021)
