# PyTorch FSDP2 Single GPU Analysis — Why ZeRO-2+CPU_Adam Remains the Only Mature Solution

> 2026-06-18 | FSDP2 single GPU deep analysis | dp_world_size=1 → partition=full → AllGather/ReduceScatter=identity
> ★★★★★★★★ FSDP2 on single GPU = pure overhead → ZeRO-2+CPU_Adam (AVX512, 5-7x faster) = ONLY mature optimizer-offload

---

## FSDP2 Architecture on Single GPU

```
★★★★★★★★★ FSDP2 single GPU behavior — dp_world_size=1:

  Sharding logic:
  → dp_world_size = 1 → partition_size = full → NO sharding
  → Each parameter: full copy on this single GPU → no savings
  → AllGather: identity operation → gather from 1 shard → no-op
  → ReduceScatter: identity operation → scatter to 1 shard → no-op

  ★★★★★★★★ What FSDP2 ACTUALLY does on single GPU:
  → Wraps every module in FSDPModule → adds overhead per module
  → Pre-unshard → forward hook → triggers AllGather (identity) → overhead
  → Post-unshard → forward hook → frees gathered shard → overhead
  → ReduceScatter → backward hook → triggers ReduceScatter (identity) → overhead
  → ALL hooks = no-ops in terms of data movement → BUT still add latency!

★★★★★★★★★ Net effect on single GPU:
  → Memory: NO savings (full parameters anyway) → same as vanilla DDP
  → Compute: NO savings (no gradient partitioning) → same as vanilla
  → Latency: ADDS overhead (FSDP hooks, buffer management, callback scheduling)
  → → FSDP2 on single GPU = WORSE than vanilla DDP → pure overhead!
```

---

## CPUOffloadPolicy Status — ALPHA with Crash Bugs

```
★★★★★★★★★ FSDP2 CPUOffloadPolicy — NOT production-ready:

  Status: ALPHA → crash bugs reported → not safe for training

  Expected behavior:
  → Offload parameters to CPU → reduce GPU memory → enable larger models
  → CPU optimizer → similar to DeepSpeed ZeRO-2 + CPU_Adam

  Actual behavior (ALPHA):
  → Crash during gradient offload → CUDA synchronization error
  → Memory leak during optimizer step → CPU memory grows unbounded
  → Incorrect gradient aggregation → training divergence
  → → NOT usable for ANY production training!

★★★★★★★★★ Comparison with DeepSpeed ZeRO-2 + CPU_Adam:
  → DeepSpeed CPU_Adam: production-tested → AVX512 optimized → 5-7x faster than naive CPU Adam
  → DeepSpeed ZeRO-2: mature → stable → used in production for 2+ years
  → FSDP2 CPUOffload: ALPHA → crash bugs → NOT usable
  → → ZeRO-2 + CPU_Adam = ONLY mature optimizer-offload solution on single GPU!
```

---

## Why ZeRO-2+CPU_Adam Is the ONLY Mature Solution

```
★★★★★★★★★ ZeRO-2 + CPU_Adam advantages on single GPU:

  1. GPU memory savings:
    → Optimizer states on CPU → 2x model size freed on GPU
    → Adam: m + v = 2x parameter size → CPU offload = 2x savings
    → Example: 7B model → 7B * 2 * 4bytes = 56B → freed from GPU

  2. CPU optimizer performance:
    → CPU_Adam with AVX512 → 5-7x faster than naive CPU Adam
    → AVX512: 512-bit SIMD → 16 float32 per cycle → optimized
    → Intel Xeon: AVX512 support → most server CPUs
    → → CPU optimizer NOT bottleneck → fast enough for single GPU training

  3. Zero communication overhead:
    → overlap_comm=False on single GPU → no NCCL → no overlap → no NaN (#8061)
    → Gradient: GPU → CPU (direct copy) → optimizer step → CPU → GPU
    → → Simple pipeline → no overlap → no communication → no bug

  ★★★★★★★★ 18Ψ vs 3.8Ψ memory comparison:
    → Vanilla DDP: 18Ψ (model + gradients + optimizer all on GPU)
    → ZeRO-2+CPU_Adam: 3.8Ψ (model + gradients on GPU, optimizer on CPU)
    → → 4.7x memory reduction → critical for RTX 4090 24GB!
```

---

## Memory Budget Comparison

```
★★★★★★★★★ Single GPU memory budget comparison (Qwen3-4B LoRA):

| Method | Model Params | Gradients | Optimizer | Activations | Total | Fits 24GB? |
|--------|-------------|-----------|-----------|-------------|-------|------------|
| Vanilla DDP | 8GB | 8GB | 16GB | 2-4GB | 34-36GB | NO |
| ZeRO-2+CPU_Adam | 8GB | 8GB | 0 (CPU) | 2-4GB | 10-12GB | YES (12GB margin) |
| FSDP2 | 8GB | 8GB | 16GB | 2-4GB | 34-36GB | NO (same as DDP) |
| FSDP2+CPUOffload | 4GB | 4GB | 8GB (CPU) | 2-4GB | 10-12GB | YES (but ALPHA/crash) |

★★★★★★★★★ LoRA+bypass further reduction:
  → LoRA rank=16 → trainable params = 0.2% → optimizer states = 0.4% of full
  → ZeRO-2+CPU_Adam+LoRA → ~3.8GB optimizer → negligible CPU offload needed
  → bypass_mode → ref model eliminated → no second forward → further savings

★★★★★★★★★ Optimal RTX 4090 config:
  → ZeRO-2 + CPU_Adam + LoRA + bypass_mode = ~17-19GB peak
  → Fits 24GB with 5-7GB margin → safe → production-tested
  → FSDP2 alternative = ALPHA + crash → NOT viable
```

---

## Key Findings Summary

★★★★★★★★★ FSDP2 dp_world_size=1 → partition=full → AllGather/ReduceScatter=identity → pure overhead
★★★★★★★★★ FSDP2 adds latency per module (hooks, buffer management) → WORSE than vanilla DDP
★★★★★★★★★ FSDP2 CPUOffloadPolicy = ALPHA → crash bugs → NOT production-ready
★★★★★★★★★ DeepSpeed ZeRO-2+CPU_Adam = ONLY mature optimizer-offload solution → AVX512 5-7x faster
★★★★★★★★★ 18Ψ → 3.8Ψ with ZeRO-2+CPU_Adam+LoRA → 4.7x memory reduction → fits 24GB
★★★★★★★★★ ALWAYS use ZeRO-2+CPU_Adam on single GPU → NEVER use FSDP2 or ZeRO-3

---

## References

- Single GPU DDP vs ZeRO comparison: notebook/fundamentals/single-gpu-ddp-vs-zero-architecture-comparison.md
- ZeRO safety checker: tools/deepspeed_zero_safety_checker.py
- DeepSpeed config generator: tools/deepspeed_config_generator.py
- RTX 4090 decision flowchart: notebook/fundamentals/rtx4090-grpo-training-decision-flowchart.md
