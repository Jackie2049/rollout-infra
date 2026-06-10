# LoRA Fine-tuning Benchmark — OPT-125M on RTX 4090

## Overview

Validates "RTX 4090最优=LoRA单GPU训练" recommendation with actual training benchmark.
6 experiments: Full FT baseline, rank sweep, target modules, merge overhead, batch scaling, memory detail.

## Key Results

### Exp1: Full Fine-tuning vs LoRA

| Method | Step Time | Throughput | Peak Memory | Trainable % |
|---------|-----------|------------|-------------|-------------|
| Full FT  | 0.0605s  | 97,332 tok/s | 6,076 MB   | 100%        |
| LoRA r=8 | 0.0818s  | 71,959 tok/s | 7,761 MB   | 1.05%       |

**Surprising**: LoRA is **37% slower** and uses **28% more memory** at B=256!

**Why?** PEFT adds Python dispatch overhead (adapter wrapping). At large batch (B=256), optimizer states dominate memory → LoRA optimizer states (Adam = 2×params per trainable param) + base model params → total > full FT.

### Exp2: LoRA Rank Sweep

| Rank | Alpha | Step Time | Throughput | Memory | Trainable % | Loss |
|------|-------|-----------|------------|--------|-------------|------|
| r=2  | 4     | 0.0831s   | 70,889     | 7,740  | 0.26%       | 3.75 |
| r=4  | 8     | 0.0829s   | 71,060     | 7,747  | 0.53%       | 3.38 |
| r=8  | 16    | 0.0818s   | 71,959     | 7,761  | 1.05%       | 2.65 |
| r=16 | 32    | 0.0826s   | 71,325     | 7,789  | 2.08%       | 1.85 |
| r=32 | 64    | 0.0835s   | 70,488     | 7,845  | 4.07%       | 1.17 |

**Step time nearly constant** (~0.082s) regardless of rank! LoRA adapter compute is negligible compared to base model forward/backward.

**Loss decreases with rank** — more parameters → better fitting (but watch for overfitting in real training).

**Memory increases slowly** — 7,740→7,845 MB (+105 MB) for r=2→32. LoRA adapter memory is tiny.

### Exp3: Target Module Comparison

| Targets | Step Time | Loss | Memory | Trainable |
|---------|-----------|------|--------|-----------|
| attn_only (q,k,v,out) | 0.057s | 3.08 | 6,448 MB | 589,824 (0.47%) |
| ffn_only (fc1,fc2)     | 0.067s | 3.55 | 6,663 MB | 737,280 (0.59%) |
| all_linear             | 0.082s | 2.67 | 7,761 MB | 1,327,104 (1.05%) |

**Attn-only LoRA**: fastest + least memory + decent loss
**All-linear**: best loss but slowest + most memory

**Practical recommendation**: attn-only (q_proj, v_proj) for production, all-linear for quality.

### Exp4: LoRA Merge Overhead

| Mode | Time per inference | Speedup |
|------|-------------------|---------|
| Base model | 5.71ms | 1.00x |
| LoRA unmerged | 14.36ms | 0.40x (**2.51x slower!**) |
| LoRA merged | 5.62ms | 1.01x (**same as base!**) |

**LoRA merge eliminates ALL inference overhead!** After merging, LoRA model = base model speed.
- Unmerged: 2.51x slower (adapter dispatch overhead)
- Merged: virtually identical to base (weights absorbed into base)

**Production recommendation**: Always merge LoRA after training for inference.

### Exp5: LoRA Batch Size Scaling

| Batch | Step Time | Throughput | Memory |
|-------|-----------|------------|--------|
| B=4   | 0.041s    | 2,223 tok/s | 623 MB |
| B=8   | 0.042s    | 4,423 tok/s | 730 MB |
| B=16  | 0.042s    | 8,847 tok/s | 946 MB |
| B=32  | 0.043s    | 17,279 tok/s | 1,384 MB |
| B=64  | 0.041s    | 35,594 tok/s | 2,259 MB |

**Step time nearly constant** (~0.04s) — memory-bound at small batches, compute throughput scales linearly.

**Memory scales linearly** with batch — B=4→623MB, B=64→2,259MB (3.6x).

**RTX 4090 24GB**: can handle B=~256 with LoRA r=8 all-linear → ~6GB → plenty of room for larger models.

### Exp6: LoRA Memory Detail (B=8, after 1 step)

| Method | Memory | Saving |
|--------|--------|--------|
| Full FT | 892 MB | — |
| LoRA r=8 | 720 MB | **19.3%** |

At small batch, LoRA saves 19% memory — optimizer states for 1.05% params vs 100% params.

## 7 Core Laws — LoRA Training on RTX 4090

1. **LoRA-Not-Faster-For-Training**: LoRA is 37% slower than full FT (Python adapter dispatch overhead), not faster. Speed benefit is for inference only (after merge).

2. **LoRA-Memory-Saving-Small-Batch**: At small batch (B=8), LoRA saves 19% memory. At large batch (B=256), LoRA uses MORE memory (optimizer states for adapters + base model params).

3. **LoRA-Rank-Near-Constant-Time**: Step time ~0.082s regardless of rank (r=2→32). Adapter compute negligible vs base model compute.

4. **LoRA-Merge-Eliminates-Overhead**: After merging, LoRA model = base model speed (5.62ms vs 5.71ms). Unmerged = 2.51x slower. Always merge for production inference.

5. **LoRA-Attn-Only-Best-Efficiency**: attn-only targets: fastest (0.057s), least memory (6,448MB), decent quality. Production recommendation for efficiency.

6. **LoRA-Enables-Large-Model**: Key benefit is NOT speed — it's enabling training of models that would OOM with full FT. 7B LoRA r=8 = 0.5MB trainable params → fits on single RTX 4090.

7. **LoRA-Batch-Scaling-Linear**: LoRA training throughput scales linearly with batch (step time constant). B=4→2,223 tok/s, B=64→35,594 tok/s.

## Practical Recommendations for RTX 4090

### Small Model (<500M) — OPT-125M
- **Full FT preferred** if model fits in memory (faster training)
- **LoRA useful** only for memory savings at specific batch sizes
- **LoRA merge** critical for inference speedup

### Large Model (7B)
- **LoRA is REQUIRED** — full FT OOM on 24GB GPU
- LoRA r=8, alpha=16, all-linear targets recommended
- After training → merge → inference at base model speed
- LoRA enables single-GPU training → avoids PCIe multi-GPU disaster

### Production Serving
1. Train with LoRA (enables 7B on single GPU)
2. Merge LoRA weights into base model
3. Apply INT4 quantization (AWQ/Marlin) on merged model
4. Serve with FlashInfer → optimal RTX 4090 serving

## Tool

`tools/lora_125m_benchmark_4090.py` — 6 experiments, argparse-based

## Related

- LoRA/PEFT Deep Dive: `notebook/fundamentals/lora-peft-deep-dive.md`
- FSDP Scaling 125M: `notebook/fundamentals/fsdp-scaling-125m-benchmark-rtx4090.md`
- RTX 4090 PCIe Decision Guide: `notebook/fundamentals/rtx4090-pcie-decision-guide.md`