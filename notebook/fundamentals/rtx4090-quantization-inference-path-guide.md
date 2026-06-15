# RTX 4090 LLM Inference Quantization Path Guide

> 2026-06-16 | RTX 4090 consulting reference — SM89 quantization viability
> Focus: Complete guide to which quantization paths work/fail on RTX 4090 (SM89)
> Critical: 3 FP8 KV paths must be distinguished; INT4 Triton/Marlin = only viable inference quantization

---

## 1. SM89 Quantization Viability Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Quantization Type | Method | SM89 Status | Framework Support | Use Case |
|-------------------|--------|-------------|-------------------|----------|
| BF16 | None (full precision) | ✓ WORKS | All | Training (LoRA), small model inference |
| INT4 W4A16 | GPTQ-Int4 + Marlin | ✓ WORKS | vLLM, SGLang | ★★★★★ Primary inference quantization for 7-8B |
| INT4 W4A16 | GPTQ-Int4 + Triton fallback | ✓ WORKS | vLLM (#43731) | ★★★★★ Fallback when Marlin unavailable |
| FP8 W8A8 | compressed-tensors | ✗ CRASH on KV | vLLM (#44879/#45038) | AVOID for KV on SM89 |
| FP8 W8A8 | Triton FP8 KV | ✓ ALLOWED (#43914) | vLLM | ★★★★ Only viable FP8 KV path on SM89 |
| FP8 W8A8 | FlashInfer FP8 KV | ✗ NOT supported | vLLM | FlashInfer FP8 gate blocks SM89 |
| INT8 KV | FlashInfer INT8 | ✓ WORKS | vLLM | ★★★★★ Only production KV quantization |
| FP16 KV | Default | ✓ WORKS | All | Standard, no quantization |
| AWQ W4A16 | AWQ + GPTQ-Marlin | ✓ WORKS | vLLM, SGLang | Alternative INT4 method |
| MXFP4 | float4_e2m1fn_x2 | ✗ SM120 only | MindIE, PyTorch v2.12 | RTX 5090 future direction |
| DeepSeek-V4 Flash | SM90 exclusive | ✗ SM90 only | Megatron (#5266) | Not available on SM89 |

---

## 2. The 3 FP8 KV Path Distinction (CRITICAL!)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Many people confuse the 3 FP8 KV cache paths on SM89. They are fundamentally different:

### Path 1: Triton FP8 KV Cache (vLLM #43914 — ALLOWED on SM89)

```python
# vllm/v1/attention/backends/triton_attn.py
# Uses Triton kernels for FP8 KV cache operations
# SM89 gate: ALLOWED (not blocked by hardware check)
# Works because: Triton FP8 kernels can run on SM89
# Limitation: Triton backend less optimized than FlashInfer for large batches
```

★★★★★ Status: WORKS on SM89. Triton FP8 KV is allowed by the SM version check. However, Triton backend may be slower than FlashInfer for some workload patterns.

### Path 2: FlashInfer FP8 KV Cache (NOT supported on SM89)

```python
# vllm/v1/attention/backends/flashinfer.py
# FlashInfer FP8 KV cache requires SM90+ hardware support
# SM89 gate: BLOCKED by FlashInfer's internal SM version check
# Reason: FlashInfer's FP8 kernels use SM90-specific instructions (TMA, etc.)
```

★★★★★ Status: NOT available on SM89. FlashInfer's FP8 kernels require SM90 hardware features. Cannot be bypassed.

### Path 3: compressed-tensors FP8 KV Override (CRASH on SM89 — #44879/#45038)

```python
# vllm/v1/attention/backends/flashinfer.py + compressed-tensors quant config
# compressed-tensors overrides kv_cache_dtype to FP8 regardless of backend
# This BYPASSES the FlashInfer SM90 gate → FlashInfer tries to run FP8 on SM89 → CRASH!
# Bug: compressed-tensors doesn't check SM version before setting kv_cache_dtype
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Status: CRASHES on SM89. This is the bug reported in #44879/#45038. The compressed-tensors quantization config overrides kv_cache_dtype to FP8, bypassing FlashInfer's SM90 gate, causing a crash.

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

**Key takeaway**: FP8 KV cache is NOT production-viable on SM89. The only working path (Triton FP8) is slower than INT8 KV via FlashInfer. INT8 KV = recommended path for RTX 4090.

---

## 3. INT4 Inference: The RTX 4090 Production Path

### 3.1 GPTQ-Int4 + Marlin (Best Performance)

★★★★★★★★★ Marlin is the optimized INT4 inference kernel for vLLM:

```python
# vllm/model_executor/layers/quantization/marlin.py
# Marlin kernel: optimized W4A16 matmul
# SM89: WORKS (Marlin supports SM89)
# Performance: ~2-3x throughput vs FP16 on RTX 4090
# Model format: GPTQ-Int4 quantized weights → Marlin format conversion on load
```

★★★★★★★ Marlin is the fastest INT4 inference path on SM89. It uses:
- 4-bit weight decomposition (group quantization)
- Optimized CUDA kernel for W4A16 matmul
- Pre-packed weight format (load-time conversion from GPTQ to Marlin format)

### 3.2 GPTQ-Int4 + Triton Fallback (vLLM #43731)

★★★★★★★★★ Triton INT4 fallback for models without Marlin support:

```python
# vllm/v0.23.0 new feature (#43731)
# When Marlin kernel unavailable for a model → falls back to Triton INT4
# SM89: WORKS (Triton INT4 kernels support SM89)
# Performance: slower than Marlin but still viable
# Use case: models where Marlin format conversion fails
```

★★★★★★★ vLLM v0.23.0 added this fallback, making INT4 inference more robust on SM89.

### 3.3 AWQ + GPTQ-Marlin (Alternative INT4)

★★★★★ AWQ is another 4-bit quantization method:

```python
# AWQ: Activation-aware Weight Quantization
# Different quantization strategy than GPTQ (preserves salient weights)
# vLLM: AWQ models loaded through GPTQ-Marlin kernel
# SM89: WORKS (same Marlin kernel backend)
# Performance: similar to GPTQ-Int4+Marlin
```

---

## 4. Recommended Quantization Strategy per Model Size

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B (3.4GB BF16)

| Component | Quantization | Memory | Reason |
|-----------|-------------|--------|--------|
| Model weights | BF16 (full) | 3.4GB | Small enough, no quantization needed |
| KV cache | INT8 (FlashInfer) | ~1-2GB | 50% savings vs FP16 KV |
| LoRA weights | BF16 | ~0.6GB | LoRA must stay in full precision for training |
| Total inference | BF16 + INT8 KV | ~5-6GB | Fits easily in 24GB |

★★★★★ For 1.7B: BF16 model + INT8 KV cache = simplest and sufficient. No weight quantization needed.

### Qwen3-8B (16GB BF16)

| Component | Quantization | Memory | Reason |
|-----------|-------------|--------|--------|
| Model weights | GPTQ-Int4 + Marlin | ~4GB | MUST quantize to fit 24GB |
| KV cache | INT8 (FlashInfer) | ~2-3GB | 50% savings vs FP16 KV |
| LoRA weights | BF16 | ~0.6GB | LoRA stays full precision |
| Total inference | INT4 + INT8 KV + BF16 LoRA | ~7-8GB | Fits 24GB with room for activations |

★★★★★★★ For 8B: INT4 weight quantization + INT8 KV cache = mandatory for 24GB. BF16 LoRA on top of INT4 base.

★★★★★★★★★ Hybrid approach: LoRA on BF16 base weights + INT4 for inference only:
- During training: BF16 base weights + LoRA → ~16.6GB (tight but fits with optimizations)
- During rollout/inference: INT4 quantized model → ~4GB → room for KV + activations
- The switch between training and inference is handled by verl/rLLM framework

### Llama-3.1-8B (16GB BF16)

Same strategy as Qwen3-8B: GPTQ-Int4 + Marlin + INT8 KV + BF16 LoRA.

---

## 5. Quantization + Framework Integration Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Framework | Inference Quant | KV Quant | Training Quant | Batch Invariance Fix |
|-----------|----------------|---------|----------------|---------------------|
| vLLM | INT4 Marlin/Triton ✓ | INT8 FlashInfer ✓ | BF16 LoRA only | Inductor Guard / enforce_eager |
| SGLang | INT4 Marlin ✓ | INT8 Triton ✓ | BF16 LoRA only | Deterministic inference built-in |
| rLLM Tinker | vLLM backend | Same as vLLM | BF16 LoRA only | Same as vLLM |
| verl | vLLM or SGLang | Same | BF16 LoRA only | Same |
| DeepSpeed | N/A (training) | N/A | BF16 + CPU offload | N/A (training only) |
| Megatron | INT8 recommended SM89 | FP16 KV | BF16 only | batch_invariant_mode (slow) |

---

## 6. Quantization Pitfalls on SM89

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Pitfall 1: FP8 KV crash via compressed-tensors (#44879/#45038)

```python
# WRONG:
quantization_config = {
    "quant_method": "compressed-tensors",
    "kv_cache_dtype": "float8_e4m3fn"  # CRASHES on SM89!
}

# CORRECT:
kv_cache_dtype = None  # Default FP16, or:
# Use INT8 KV via FlashInfer backend:
kv_cache_config = {"kv_cache_dtype": "int8"}
```

★★★★★★★ Fix: Never use FP8 KV cache on SM89. Always use INT8 KV via FlashInfer or FP16 KV.

### Pitfall 2: FlashInfer FP8 not supported on SM89

```python
# FlashInfer FP8 kernel gate:
if current_platform.has_device_capability(90):
    # FP8 KV allowed (SM90+)
else:
    # FP8 KV BLOCKED → fallback to FP16
```

★★★★★ Don't try to override this gate. Use INT8 KV instead.

### Pitfall 3: Triton FP8 KV is slower than INT8 FlashInfer KV

```python
# Triton FP8 KV: WORKS but slower than INT8 KV via FlashInfer
# On SM89, INT8 KV (FlashInfer) is the recommended path:
# - Better performance (FlashInfer optimized)
# - Same memory savings (~50%)
# - Production-tested on SM89
```

★★★★★ Use INT8 KV via FlashInfer for best performance on SM89.

### Pitfall 4: INT4 weight quantization + FP8 KV = incompatible mix

```python
# Some models ship with both INT4 weights AND FP8 KV config
# On SM89: INT4 weights ✓, FP8 KV ✗ → CRASH!
# Fix: load INT4 model with INT8 KV override:
llm = LLM(model="Qwen3-8B-GPTQ-Int4", kv_cache_dtype="int8")
```

★★★★★ Always override KV cache to INT8 when using INT4 models on SM89.

---

## 7. Future Direction: FP4/MXFP4 on SM120

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

- RTX 5090 = SM120 + FP4 native → MXFP4 quantization path
- FP4 will replace INT4: floating-point + MX scaling = better accuracy + HW acceleration
- MindIE already has MXFP4 (float4_e2m1fn_x2) on Ascend 910C+
- PyTorch v2.12: MXFP4 AOTI shim support → kernel infrastructure
- vLLM: SM120 FP4/MXFP4 kernel gap = NEXT-PHASE contribution window
- RTX 4090 (SM89) won't benefit from FP4 → INT4 remains the best quantization path for SM89

---

## 8. Quick Reference: RTX 4090 Quantization Commands

### vLLM INT4 + INT8 KV inference

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B-GPTQ-Int4 \
  --kv-cache-dtype int8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048
```

### vLLM BF16 + INT8 KV (small model)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --kv-cache-dtype int8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

### SGLang deterministic + INT4 inference

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-GPTQ-Int4 \
  --enable-deterministic-inference \
  --mem-fraction-static 0.85
```

### rLLM Tinker GRPO training (BF16 LoRA)

```bash
bash tools/train_tinker_rtx4090.sh --model Qwen3-1.7B --task math
```
