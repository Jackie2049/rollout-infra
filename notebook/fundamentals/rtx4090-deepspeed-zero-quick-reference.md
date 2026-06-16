# RTX 4090 DeepSpeed ZeRO Quick Reference Card

> 2026-06-16 | Print-ready reference | 7 critical pitfalls + 4 safe configs
> Based on source-level analysis of stage_1_and_2.py, engine.py, config.py

---

## CRITICAL: 5 Pitfalls That Will BREAK Your Training

| # | Pitfall | Symptom | Fix | Issue |
|---|---------|---------|-----|-------|
| 1 | overlap_comm=True + torch.compile | NaN from step 1 | overlap_comm=False | #8061 |
| 2 | gradient_clipping=0.0 (default) | Gradient explosion, training collapse | gradient_clipping=1.0 | #8068 |
| 3 | ZeRO-3 on single GPU | Wasted 2-4GB, no benefit | ZeRO-2 + CPU_Adam | source |
| 4 | FP8 compressed-tensors KV on SM89 | Crash/OOM | INT8 FlashInfer or Triton FP8 | #44879 |
| 5 | bf16+fp16 both enabled | Config error | bf16 only | SM89 |

---

## RTX 4090 Optimal DeepSpeed Config Template

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-4, "betas": [0.9, 0.999]}
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "all_contiguous_gradients": true,
        "overlap_comm": false,
        "reduce_bucket_size": 5e6
    },
    "bf16": {"enabled": true},
    "fp16": {"enabled": false},
    "data_types": {"grad_accum_dtype": "fp32"}
}
```

---

## Why overlap_comm=False on Single GPU

- dp_world_size=1 → reduce_scatter is identity (no sharding)
- all_reduce with dp=1 is a no-op
- "Overlap" = between grad compute and reduction of SAME GPU's grads → meaningless
- overlap_comm=True + torch.compile = NaN (#8061): average_tensor() only waits current_stream()
- Therefore: **overlap_comm=False = zero penalty + safer** on single GPU

---

## 4 Safe Config Variants

| Scenario | Config | Optimizer | LoRA | Key Notes |
|----------|--------|-----------|------|-----------|
| LoRA GRPO | lora-grpo_rtx4090.json | AdamW | rank=32 | Standard training |
| LoRA GRPO + Muon | muon_lora_zero2_rtx4090.json | Muon Gram | rank=32/64 | Experimental! Compare with AdamW baseline |
| MoE AutoEP | moe-autoep_rtx4090.json | AdamW | N/A | Qwen3-MoE EP=1, gradient_clipping=1.0 |
| OPD Distill | opd-distill_rtx4090.json | AdamW | rank=32 | Student-only GPU, teacher CPU-offloaded |

All configs PASS all 7 safety checks. Verify with:
```bash
python3 tools/deepspeed_zero_safety_checker.py --mode check --config <path>
```

---

## ZeRO Stage Comparison on Single GPU

| Aspect | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--------|--------|--------|--------|--------|
| Param partitioning | No | No (dp=1) | No (dp=1) | No (dp=1) |
| Grad partitioning | No | No (dp=1) | No (dp=1) | No (dp=1) |
| Optim offload to CPU | No | Optional | Yes (CPU_Adam) | Yes |
| CPU_Adam benefit | N/A | Yes | Yes (5-7x faster) | Yes |
| Overhead | Minimal | Minimal | Low | HIGH (2-4GB waste) |
| RTX 4090 recommendation | Small models | LoRA small | **RECOMMENDED** | **AVOID** |

---

## Memory Budget Estimates (RTX 4090 24GB)

| Model | ZeRO-2+LoRA | ZeRO-2+Full | ZeRO-3+LoRA | ZeRO-3+Full |
|-------|-------------|-------------|--------------|-------------|
| Qwen2.5-0.5B | ~2.6GB | ~6GB | ~4.6GB (waste) | ~8GB (waste) |
| Qwen3-1.7B | ~5GB | ~14GB | ~7GB (waste) | ~16GB (waste) |
| Qwen3-4B | ~8GB | OOM | ~10GB (waste) | OOM |
| Qwen3-MoE (A0.6B+B4B) | ~20GB | OOM | N/A | N/A |

---

## Quick Diagnostic Commands

```bash
# Check config for all 7 pitfalls
python3 tools/deepspeed_zero_safety_checker.py --mode check --config configs/your_config.json

# Generate safe config
python3 tools/deepspeed_zero_safety_checker.py --mode generate --scenario lora-grpo --model qwen3-1.7b

# Explain specific pitfall
python3 tools/deepspeed_zero_safety_checker.py --mode explain --check-name overlap_comm_compile_nan

# Cross-framework safety matrix
python3 tools/rtx4090_cross_framework_safety_matrix.py --mode summary

# GRPO troubleshooter (verl/rLLM specific)
python3 tools/grpo_troubleshooter_4090.py --mode check
```

---

## #8061 Root Cause Quick Summary

average_tensor() in stage_1_and_2.py:1230-1237:
```python
if self.overlap_comm:
    stream = self.reduction_stream
    stream.wait_stream(get_accelerator().current_stream())  # ← BUG! Only waits current_stream
```

torch.compile → compiled autograd → multi-stream copy_() → average_tensor reads incomplete data → NaN.

Fix: overlap_comm=False eliminates risk. Single GPU: zero penalty anyway (dp=1, no cross-GPU comm).

---

## #8068 Gradient Clipping Quick Summary

DeepSpeed GRADIENT_CLIPPING_DEFAULT = 0.0 (in constants.py).
Most RL/LLM training clips at 1.0. Omitting gradient_clipping = silently unclipped = gradient explosion risk.
GRPO especially vulnerable (advantage computation can explode).

Fix: Always set `"gradient_clipping": 1.0` explicitly in config.
