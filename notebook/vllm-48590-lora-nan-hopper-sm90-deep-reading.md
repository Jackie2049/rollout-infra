# vLLM #48590: LoRA NaN on Hopper sm_90 (lora_expand block_n=128)

## Overview
- **Issue**: vllm-project/vllm #48590 by n0whereRuoxi, July 14, 2026
- **Status**: OPEN, 0 comments, 0 maintainer response
- **Severity**: ★★★★★★★★ CRITICAL for GRPO+LoRA on Hopper (H100/H20)
- **Affected kernel**: `lora_expand` Triton kernel in `vllm/lora/ops/triton_ops/lora_expand_op.py`
- **Root cause**: `block_n=128` tile size on sm_90 triggers NaN
- **Workaround**: Set `block_n=32` in `vllm/lora/ops/triton_ops/utils.py`

## Bug Details

### Symptom
Serving a Qwen3-VL-8B-Instruct LoRA adapter on Hopper (H20/H100) produces garbled output (repeated `!!!!!!` or mixed-language/code tokens). Base model without LoRA works perfectly. NaN counts:
- `torch.bfloat16`: **4,480 NaNs** per expand call
- `torch.float16`: **36,352 NaNs** per expand call

### Scope
Confirmed across vLLM **0.23.0, 0.24.0, 0.25.0** (kernel code byte-identical).

### What Was Ruled Out (exhaustive)
- vLLM version
- `--enforce-eager` toggle
- Attention backends (FlashAttn v2/v3, TORCH_SDPA)
- Sampler backend (flashinfer on/off)
- Quantization (bf16, fp8, bitsandbytes-4bit)
- Virtualization (bare-metal H20 same)
- Triton cache (wiping `~/.triton/cache`)
- Works on Blackwell (RTX 5070, sm_120) — **architecture-specific**
- Works via HF Transformers+PEFT on same GPU — **vLLM Triton kernel specific**

### Root Cause
The default single-slice expand kernel config in `utils.py`:
```python
"block_n": 64 if num_slices > 1 else 128
```
On Hopper (sm_90), `block_n=128` causes NaN — suspected Triton codegen issue at that tile size (OOB read on boundary N-tile, or MMA/shared-memory layout issue).

### Workaround
Set `block_n=32` for sm_90. Negligible performance impact, CUDA-graph safe. Restored accuracy: 0.63 → ~0.90.

## GRPO+LoRA Impact

Directly relevant to GRPO+LoRA training on H100/H20:
- **verl GRPO on H20-3e**: Partner server uses H20-3e (Hopper, sm_90). If using LoRA adapters, this bug triggers NaN in LoRA expanded weights
- **All LoRA GRPO on Hopper**: Any vLLM-based rollout engine with LoRA on sm_90 produces corrupted outputs
- **Not Blackwell-safe**: The bug is SM-agnostic at the utils.py level — it affects all sm_90 GPUs

## Cross-Framework Connections
- Same pattern as **vLLM #47303**: "Triton MoE kernels producing NaN due to OOB reads" — similar tile-size boundary issue
- Pattern: **P6 (Silent Corruption)** — NaN propagates silently through the generation pipeline, producing plausible-looking but wrong text
- If using vLLM for GRPO rollout generation with LoRA on H20-3e: **MUST apply workaround**

## Monitoring
- 0 comments, 0 reviews — completely untouched
- Opportunity: Fork fix with workaround (already verified)
- Opportunity: Cross-reference to DeepSpeed and SGLang LoRA paths for same pattern
