# Quantization Theory — Mathematical Derivation for LLM Inference

> 2026-06-19 | Algorithm Theory | From Float32 to FP8/INT4: The Math Behind Neural Network Quantization
> ★★★★★★★★ RTX 4090 Context: SM89 supports FP8 storage but NOT FP8 compute (no wgmma) → FP8→BF16 prologue fusion → P9 guard!
> ★★★★★★★★ Quantization is the #1 technique for deploying >8B models on RTX 4090 (24 GiB)

---

## 1. Why Quantization Matters

A model with $N$ parameters in BF16 (2 bytes each) occupies $2N$ bytes. For Qwen3-8B:
- $N = 8 \times 10^9$ → $2N = 16$ GiB → fits in RTX 4090
- For Qwen3-30B-A3B: $N = 30 \times 10^9$ → $2N = 60$ GiB → DOES NOT fit!

**Quantization reduces memory**: FP8 (1 byte) → $1N = 8$ GiB (fits!). INT4 (0.5 byte) → $0.5N = 4$ GiB (fits easily!).

But quantization comes with accuracy loss. The key question: **how much accuracy loss is acceptable, and how to minimize it?**

---

## 2. Uniform Quantization: The Foundation

### 2.1 Linear (Affine) Quantization

Map floating-point values to integers:

$$x_q = \text{round}\left(\frac{x}{s}\right) + z$$

Where:
- $s = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}$ — scale factor
- $z = \text{round}\left(\frac{q_{max} \cdot x_{min}}{x_{max} - x_{min}}\right)$ — zero point
- $q_{max}, q_{min}$ = integer range limits (e.g., INT8: $[-128, 127]$, FP8 E4M3: $[-448, 448]$)

**Dequantization** (recover approximate float):

$$x_{approx} = s \cdot (x_q - z)$$

**Quantization error**:

$$\epsilon = x - x_{approx} = x - s \cdot \left(\text{round}\left(\frac{x}{s}\right) + z\right)$$

The error $\epsilon$ satisfies $|\epsilon| \leq \frac{s}{2}$ (bounded by half-scale).

### 2.2 Symmetric Quantization (No Zero Point)

When $z = 0$:

$$x_q = \text{round}\left(\frac{x}{s}\right)$$
$$x_{approx} = s \cdot x_q$$

**Scale for symmetric**: $s = \frac{x_{max}}{q_{max}}$ (assuming symmetric range $[-q_{max}, q_{max}]$)

**Advantages**: Simpler, no zero-point offset, compatible with hardware that doesn't support asymmetric arithmetic.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**FP8 E4M3 and E5M2 are symmetric quantization formats** — no zero point needed. E4M3 has 4 exponent bits, 3 mantissa bits → range $[-448, 448]$. E5M2 has 5 exponent bits, 2 mantissa bits → range $[-57344, 57344]$ (wider range but less precision). vLLM/SGLang use E4M3 for weights (forward pass) and E5M2 for gradients (backward pass).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. Block-wise Quantization: The Modern Standard

### 3.1 Problem with Per-Tensor Quantization

If a tensor has values in $[0.001, 100]$, the scale $s = \frac{100}{448}$ ≈ 0.22 → small values (0.001) get rounded to 0 → COMPLETE loss of dynamic range!

**Solution**: Quantize in blocks (sub-groups) instead of the entire tensor.

### 3.2 Block-wise FP8 Quantization

Divide a weight tensor into blocks of size $B$ (typically $B = 128$ or $B = 256$). Each block gets its own scale:

$$x_q^{(b)} = \text{round}\left(\frac{x^{(b)}}{s_b}\right) \text{ for } b \in [1, ..., \frac{N}{B}]$$

Where $s_b = \frac{\max(|x^{(b)}|)}{448}$ — per-block scale for E4M3.

**Memory**: Weights in FP8 (1 byte) + scales in higher precision (e.g., BF16, 2 bytes per block):
- Total = $N \times 1 + \frac{N}{B} \times 2$ bytes
- For $B = 128$: $N + \frac{N}{64} = N \times 1.016$ bytes → only 1.6% overhead!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSeek V3/V4 uses block-wise FP8 quantization** with $B = 128$. This gives ~50% memory reduction (BF16 → FP8) with only ~1.6% scale overhead. The FP8 weights + per-block scales are stored in the model checkpoint → requires FP8-aware inference engine.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.3 MX (Microscaling) Format — OCP Standard

**MX** (Microscaling) is the OCP (Open Compute Project) standard for block-wise quantization:

| Format | Element Type | Block Size | Scale Type | Range | Mantissa Bits |
|--------|-------------|------------|-----------|-------|--------------|
| **MXFP8 (E4M3)** | FP8 E4M3 | 32 | FP8 E8M0 | $[-448 \times 2^{s}, 448 \times 2^{s}]$ | 3 |
| **MXFP8 (E5M2)** | FP8 E5M2 | 32 | FP8 E8M0 | wider range | 2 |
| **MXFP6** | FP6 E3M2 | 32 | FP8 E8M0 | custom | 2 |
| **MXINT8** | INT8 | 32 | FP8 E8M0 | $[-128 \times 2^{s}, 127 \times 2^{s}]$ | 0 |
| **MXFP4** | FP4 E2M1 | 32 | FP8 E8M0 | $[-6 \times 2^{s}, 6 \times 2^{s}]$ | 1 |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**MindIE/vLLM-Ascend #10730 NEW**: MX quant fusion for DSV4 → this is the Ascend pathway for MX-based inference. NVIDIA's Blackwell (SM100) will also support MX natively. RTX 5090 (SM120) = MX-capable → P3 FP4/MXFP4 kernel target!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)

### 4.1 PTQ (Post-Training Quantization)

Quantize a PRE-TRAINED model WITHOUT retraining:

$$\text{PTQ}: \text{BF16 model} \xrightarrow{\text{calibration}} \text{FP8 model}$$

**Calibration**: Run a small dataset through the model, collect activation statistics (min/max per block), compute scales, quantize weights.

**Accuracy**: Typically ~1-3% degradation for FP8, ~3-10% for INT8, ~10-30% for INT4 (without fine-tuning).

**RTX 4090 advantage**: FP8 PTQ is nearly lossless for most models → can deploy FP8-quantized models with minimal accuracy impact.

### 4.2 QAT (Quantization-Aware Training)

Simulate quantization effects DURING training:

$$\text{QAT}: x_{fake\_quant} = \text{dequant}(\text{quant}(x))$$

**Fake quantization**: Insert quantize→dequantize operations in the training graph. The model "sees" quantized values during training → learns to be robust to quantization error.

**STE (Straight-Through Estimator)**: The round() operation has zero gradient almost everywhere. Solution: pretend the gradient passes through unchanged:

$$\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial x_{fake\_quant}}$$

This is a biased estimator but works well in practice.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSeek V3/V4 uses FP8 QAT** (quantization-aware training with FP8 fake quantization during training). This produces models that are naturally FP8-friendly → PTQ calibration is trivial → almost zero accuracy loss.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. Mixed-Precision Quantization

### 5.1 Per-Layer Precision Selection

Not all layers benefit equally from quantization:

| Layer Type | Quantization Sensitivity | Recommended Precision | RTX 4090 Reason |
|-----------|------------------------|---------------------|----------------|
| **Attention (Q/K/V)** | HIGH (first/last layers) | BF16 (or FP8 with calibration) | Attention is critical for reasoning |
| **Attention (O projection)** | MEDIUM | FP8 E4M3 | Less sensitive than Q/K/V |
| **MLP (gate/up/down)** | LOW-MEDIUM | FP8 E4M3 | Most weights → biggest memory savings |
| **Embedding layer** | VERY HIGH | BF16 | Token embeddings must be exact |
| **Final lm_head** | HIGH | BF16 | Output logits must be accurate |
| **Layer norms** | HIGH | BF16 | Norm parameters are small (<8 KiB per layer) |
| **MoE router** | VERY HIGH | BF16 | Routing decisions are discrete → quantization changes expert selection! |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSpeed #8066 (MERGED June 16) per-policy dtype**: Allows DIFFERENT quantization policies per parameter type → this CAUSED the #8072 ZeRO-3+PEFT regression! Mixed-precision + ZeRO-3 + LoRA = broken dtype mismatch → ZeRO-2 unaffected → MUST use ZeRO-2!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 5.2 MoE Quantization Challenges

MoE models have special quantization challenges:

1. **Router sensitivity**: The gating network produces discrete expert assignments. Even small quantization errors can change which expert is selected → dramatically different model behavior.

2. **Expert weight sharing**: Some MoE architectures share parameters across experts → quantizing shared weights affects ALL experts simultaneously.

3. **Expert activation sparsity**: Only top-k experts are active per token → most expert weights are idle → quantization error on idle experts doesn't matter during inference, but DOES matter during training (backward pass touches ALL experts for gradient accumulation).

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**vLLM #45656 NEW**: GPTQ/CT MoE `is_sym` guard regression → quantization parameter mis-handling for MoE models. Symmetric vs asymmetric quantization config MUST be checked per-expert → current code has a regression that can produce garbage output on quantized MoE models.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. SM89 (RTX 4090) FP8 Limitations

### 6.1 SM89 vs SM90 Hardware Comparison

| Feature | SM89 (Ada Lovelace / RTX 4090) | SM90 (Hopper / H100) | Impact |
|---------|-------------------------------|---------------------|--------|
| **FP8 storage** | ✅ Supported | ✅ Supported | Both can store FP8 tensors |
| **FP8 compute (wgmma)** | ❌ NOT supported | ✅ Supported (wgmma FP8 tensor ops) | SM89 must convert FP8→BF16 for compute |
| **TMA (Tensor Memory Accelerator)** | ❌ NOT supported | ✅ Supported | SM89 uses traditional memory access |
| **Thread Block Clusters** | ❌ NOT supported | ✅ Supported | SM89 can't use cooperative_groups::this_cluster() |
| **CUDA graph FP8** | ⚠️ Risky | ✅ Generally safe | SM89 Inductor can fuse FP8→BF16 prologue → batch-dependent |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**The P9 Inductor SM<90 Fusion Guard (PyTorch #184119) addresses EXACTLY this**: On SM89, PyTorch's Inductor can fuse FP8→BF16 prologue conversions into downstream kernels. This creates batch-dependent fused kernels → when batch size changes, the fused kernel produces different numeric results → silent accuracy regression. The P9 guard blocks this fusion on SM<90 hardware → forces separate FP8→BF16 conversion step → ensures batch-independent behavior.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**SGLang #28618 confirms SM89 gaps**: The RFC identifies 3 hardware gaps — TMA, wgmma, Thread Block Clusters — that block DSV4's FlashMLA and DeepGEMM on SM89. The PR uses Triton fallback kernels to bypass all three. This validates the P9 thesis from a DIFFERENT angle (runtime dispatch vs compilation guard).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 6.2 FP8→BF16 Prologue Fusion: The P9 Problem

On SM89, when weights are stored in FP8 but compute requires BF16:

```
FP8 weight → [FP8→BF16 conversion] → [BF16 matmul] → output
```

**Inductor's behavior on SM89**: If max_autotune is enabled, Inductor may FUSE the FP8→BF16 conversion into the matmul kernel:

```
FP8 weight → [FUSED: convert+matmul] → output  ← ★★★★★★★★ PROBLEMATIC!
```

This fused kernel is now **batch-dependent** — the matmul shape depends on batch size, and the fused kernel's layout choices vary with shape → different numeric behavior for different batch sizes.

**P9 guard solution**: Block the fusion on SM<90 → force separate conversion → batch-independent behavior:

```
FP8 weight → [SEPARATE: convert to BF16] → [SEPARATE: BF16 matmul] → output  ← ★★★★★★★★ SAFE!
```

---

## 7. Weight-Only vs Activation Quantization

### 7.1 Weight-Only Quantization (W8A16, W4A16)

Only quantize weights → activations stay in BF16:

| Format | Weight Precision | Activation Precision | Memory Savings | Accuracy Impact |
|--------|-----------------|---------------------|---------------|----------------|
| **W8A16** | FP8 E4M3 or INT8 | BF16 | ~50% weights | ~1-3% |
| **W4A16** | INT4 or FP4 | BF16 | ~75% weights | ~5-15% |
| **W8A8** | FP8 E4M3 | FP8 E5M2 | ~50% total | ~1-5% |

**RTX 4090 benefit**: W8A16 saves ~50% of weight memory. For Qwen3-8B: 16→8 GiB → MUCH more room for KV cache!

**How it works**: The engine stores weights in FP8, converts to BF16 just before the matmul, then computes in BF16. This is EXACTLY the SM89 prologue fusion pattern → P9 guard needed!

### 7.2 Activation Quantization (W8A8)

Quantize BOTH weights and activations → more savings but more accuracy risk:

**FP8 E4M3 for weights** (forward pass): range $[-448, 448]$, 3 mantissa bits → good precision for weights.

**FP8 E5M2 for activations** (backward pass): range $[-57344, 57344]$, 2 mantissa bits → wider range needed for activation distributions (high variance, outliers).

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSeek V3/V4 FP8 quantization**: W8A8 with E4M3 for weights (forward) and E5M2 for activations (backward). Per-block scale with block_size=128. This is the production-standard FP8 quantization approach. Requires SM90 wgmma for efficient FP8 compute → on SM89 (RTX 4090), must dequantize to BF16 first → P9 guard critical!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. Quantization and GRPO Training

### 8.1 Training in BF16, Serving in FP8

The standard approach for RTX 4090 GRPO:

1. **Training phase**: All computations in BF16 → no quantization needed → FSDP + CPU_Adam
2. **Rollout phase**: SGLang/vLLM serves the model with FP8 KV cache → saves ~50% KV memory → longer context possible
3. **Weight sync**: LoRA deltas in BF16 → sleep_level=1 → base weights stay resident → 80x payload reduction

**FP8 KV cache**: `--kv-cache-dtype fp8` in vLLM → stores KV cache in FP8 E5M2 → ~50% KV memory reduction → critical for long-context GRPO on RTX 4090.

### 8.2 Quantization During Training (QAT for GRPO)

If we want to train with FP8 awareness:

1. **LoRA in BF16**: LoRA deltas must be full precision → quantization error would accumulate
2. **Base model in FP8**: Store base weights in FP8 → dequantize for forward/backward → ~50% base weight memory saved
3. **Combined**: FP8 base + BF16 LoRA → ~8 GiB base + ~200 MiB LoRA → total ~8.2 GiB → RTX 4090 can train larger models!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**This is an UNEXPLORED research direction**: FP8 base model + BF16 LoRA for GRPO training on RTX 4090. No existing framework supports this combination well. Could be a unique contribution!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 9. Connections to Framework Issues

| Quantization Concept | Framework Issue | Connection |
|---------------------|----------------|------------|
| FP8→BF16 prologue fusion | PyTorch #184119 (P9 guard) | Blocks batch-dependent fusion on SM89 |
| Per-policy dtype | DeepSpeed #8066 (MERGED → CAUSED #8072) | Mixed-precision + ZeRO-3 + LoRA = broken |
| MoE is_sym guard | vLLM #45656 NEW | Quantized MoE config regression |
| FP8 KV cache | vLLM `--kv-cache-dtype fp8` | 50% KV memory reduction → long context |
| FP8 KV scales reset | vLLM CuMemAllocator wake_up | Scales → 1.0 after wake → accuracy concern |
| MX quant fusion | vLLM-Ascend #10730 | Ascend MX pathway for DSV4 |
| Triton FP8 MQA | SGLang #28620 (SM89) | Triton kernel replaces DeepGEMM on SM89 |
| DeepGEMM disabled SM89 | SGLang issue | SM89 lacks wgmma → DeepGEMM can't run |
| vLLM w8a8 Inductor break | PyTorch #187484 | 150+ DeepGEMM failures on torch 2.13 → blocks #45731 |
| FP4/MXFP4 kernel | P3 contribution target | RTX 5090 (SM120) native MX → future-phase |

---

## 10. Quantization Decision Tree: RTX 4090

```
                        Model Size
                            │
           ┌────────────────┼────────────────┐
           │                │                │
       <8B (fits BF16)  8-16B (needs FP8)   >16B (needs MoE+FP8)
           │                │                │
           │           ┌────┴────┐      ┌────┴────┐
           │           │         │      │         │
           │       W8A16    W8A8   MoE + FP8  MoE + LoRA
           │       (weight    (full    (active   (LoRA on
           │        only)    FP8)     3B FP8)   3B BF16)
           │           │         │      │         │
           │       P9 guard  P9 guard  +offload  sleep_level=1
           │       needed    needed    experts   80x reduction
           │           │         │      │         │
           │       ★★★★     ★★★★★  ★★★★★★★  ★★★★★★★★
           │       Good      Best    BEST for   OPTIMAL for
           │                         30B-A3B    RTX 4090 GRPO
```

---

## References

1. Micikevicius et al. "FP8 Formats for Deep Learning" (2022) — FP8 format specification
2. Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022) — INT8 quantization
3. Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022) — GPTQ
4. Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (2023) — activation quantization
5. DeepSeek-AI "DeepSeek-V3 Technical Report" (2024) — FP8 W8A8 quantization, block_size=128
6. OCP Microscaling Specifications — MX format standard
