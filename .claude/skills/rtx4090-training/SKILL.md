# RTX 4090 Training Pipeline Skill

Provides RTX 4090-specific training, inference, and optimization recommendations backed by 30+ actual GPU benchmarks. Every recommendation is validated with measurement data, not theory.

## When to Use

- User asks about RTX 4090 training/inference optimization
- User wants to know what techniques work/don't work on consumer GPUs with PCIe
- User is planning a training or serving pipeline for LLMs on RTX 4090
- User asks about multi-GPU scaling on PCIe systems
- User needs quick reference for RTX 4090 benchmark results

## Key Data Points

### Training Pipeline (validated on RTX 4090 with OPT-125M)

**Optimal configuration: BF16 + LoRA r=8 α=16 all-linear + Lion/AdamW + B=4 accum=4**

| Technique | Result | Why |
|-----------|--------|-----|
| BF16 training | 1.23x speed + 9% accuracy + 37% memory saving vs FP32 | BF16 dynamic range = FP32, no gradient underflow |
| FP16 AMP | **Zero speedup** + 11% accuracy DROP | Legacy for V100 era, RTX 4090 BF16 native |
| LoRA r=8 | Enables 7B single-GPU training | Only 1.05% trainable params → optimizer states tiny |
| LoRA training speed | **37% slower** than Full FT | PEFT adapter dispatch overhead, NOT speed gain |
| LoRA merge for inference | **Zero overhead** after merge | merged = base model speed (5.62ms vs 5.71ms) |
| Lion optimizer | **95.5% convergence** (best!) | Sign-based updates, 2.6x faster initial convergence |
| Lion memory | **50% optimizer saving** (1× vs 2× params) | Only exp_avg, no exp_avg_sq |
| SGD+Momentum | 91% convergence, viable for LoRA | Simplest optimizer, cheapest |
| Gradient accumulation | **15x throughput loss** at accum=16 | Linear throughput decay, constant memory |
| Accum memory | **352MB constant** regardless of accum | Only micro-batch activations stored |
| Checkpointing | **31% throughput cost**, saves 11-45% memory | Forward recomputation cost is constant |
| Checkpointing (Full FT) | **Only 1% memory saving** | Optimizer states dominate → useless |
| Checkpointing (LoRA) | **35% memory saving** at B=32 | Activations fraction larger → effective |

### Multi-GPU on RTX 4090 PCIe (DO NOT USE for >2B models!)

| Config | Per-GPU Speedup | Total Speedup |
|--------|-----------------|---------------|
| FSDP 7B 8GPU | 0.46x | **DISASTER** (0.46x total!) |
| DDP 7B 8GPU | 0.23x | **DISASTER** |
| FSDP 125M 8GPU | 0.79x per-GPU | 1.79x total (acceptable) |
| DDP 125M 8GPU | 0.94x per-GPU | 1.90x total (acceptable) |
| P2P access | 0/56 disabled | Consumer GPU → PCIe only → no NVLink |

**NCCL benchmark data:**
- AllReduce: 2GPU=7.5→4GPU=5.0→8GPU=3.0 GB/s (linear decay)
- AllGather: 2GPU=12.8→4GPU=6.0→8GPU=5.6 GB/s (2.3x decay)
- 7B FSDP comm: 8GPU=1536ms → 75% communication → disaster

### Inference Pipeline (validated on RTX 4090)

**Optimal serving: AWQ INT4 + INT8 KV + FlashInfer GQA-8**

| Technique | Result | Key Data |
|-----------|--------|----------|
| AWQ INT4 | 75% memory saving, cos_sim=0.993 | Needs fused Marlin kernel (Python dequant 20x slow!) |
| INT8 KV | 50% KV saving, 3-12% overhead | Near-free with Python dequant |
| FP8 E4M3 KV | **Best KV quantization** | cos_sim=0.999996 (more accurate than INT8!) |
| FlashInfer GQA-8 | **83.41x** vs SDPA | Production attention answer |
| FlashInfer overall | 1.06-3.20x real speedup | Not just attention (15x), whole-model speedup |
| N-gram spec decode | 2.14x speedup | Simplest spec decode, zero-cost draft |
| Eagle spec decode | 4.2x speedup | Best spec decode for RTX 4090 |
| StreamingLLM | Infinite context | Fixed KV = 168MB → unlimited conversation |

**7B INT4 AWQ + INT8 KV + FlashInfer → B=118 → 4,791 tok/s**

### Key Hardware Data

- HBM bandwidth: 890.8 GB/s (实测)
- GEMM peak: 169.6 TFLOPS (FP16)
- Ridge point: AI≈185 → decode severely memory-bound
- P2P: All disabled → PCIe only → no NVLink
- SM89 (Ada): HMMA, cp.async, 100KB smem, 72MB L2

## What NOT to Use on RTX 4090

| Technique | Why Not | Alternative |
|-----------|---------|-------------|
| FSDP/DDP 8GPU (7B) | PCIe comm disaster (0.23-0.46x) | Single GPU + LoRA |
| FP16 AMP | Zero speedup, accuracy drop | BF16 (no AMP) |
| Ring Attention | 7-67x slower (PCIe P2P disabled) | Single GPU |
| Multi-stream | Negative (0.69-0.90x) | Single stream |
| torch.compile (B≥16) | 1.01x only | FlashInfer (inference) |
| Python INT4 dequant | 0.05x (20x slower!) | AWQ/Marlin fused kernel |
| Full FT checkpointing | Only 1% memory saving | Use LoRA instead |
| DeepSpeed ZeRO-3 (single GPU) | Pure overhead, no benefit (dp=1) | ZeRO-2 + CPU_Adam |
| overlap_comm=True (single GPU) | NaN with torch.compile (#8061) + no benefit (dp=1) | overlap_comm=False |
| gradient_clipping=0.0 | Silently disabled → gradient explosion (#8068) | gradient_clipping=1.0 |

## RTX 4090 DeepSpeed ZeRO Safety Checklist

CRITICAL: Run `python3 tools/deepspeed_zero_safety_checker.py --mode check --config <path>` before ANY ZeRO training.

5 most common pitfalls:
1. **overlap_comm=True + torch.compile = NaN** (#8061) → Fix: overlap_comm=False (zero penalty on single GPU)
2. **gradient_clipping=0.0 (default)** (#8068) → Fix: set gradient_clipping=1.0 explicitly
3. **ZeRO-3 on single GPU** → Fix: Use ZeRO-2 + CPU_Adam (partition_size=full anyway with dp=1)
4. **FP8 on SM89** → Fix: Use INT8 KV (FlashInfer) or Triton FP8 (#43914), NOT compressed-tensors
5. **bf16+fp16 both enabled** → Fix: Choose bf16 only (SM89 native)

Generated safe configs: configs/lora-grpo_rtx4090.json, configs/moe-autoep_rtx4090.json, configs/opd-distill_rtx4090.json, configs/muon_lora_zero2_rtx4090.json

## Files Reference

Key benchmark result files in `results/`:
- `fsdp_scaling_125m_benchmark.json` — FSDP/DDP scaling
- `lora_125m_benchmark_4090.json` — LoRA training
- `mixed_precision_training_2.28m_4090.json` — BF16 vs FP16 vs FP32
- `gradient_accumulation_125m_4090.json` — Grad accum cost
- `optimizer_comparison_125m_4090.json` — Lion/AdamW/SGD
- `activation_checkpointing_125m_4090.json` — Checkpointing cost

Key notes in `notebook/fundamentals/`:
- `rtx4090-training-pipeline-guide.md` — Complete reference
- `rtx4090-pcie-decision-guide.md` — Decision tree
- All individual benchmark notes (6 files)