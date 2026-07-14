# Megatron-LM Architecture: GRPO Training Foundation

## Overview
Megatron-LM (`megatron/core/`) is NVIDIA's distributed training framework supporting 6 parallelism dimensions: TP/PP/DP/SP/EP/CP. It's the primary training engine for large-scale GRPO pipelines.

## Parallelism Dimensions

| Type | Split Dim | Communication | GRPO Relevance |
|------|-----------|--------------|----------------|
| TP | Weight matrix | AllReduce | Required for models > single GPU memory |
| PP | Layers | P2P | Not typical for GRPO (added latency) |
| DP | Data batch | AllReduce grads | **Standard GRPO**: different prompts per DP rank |
| SP | Sequence | AllGather/ReduceScatter | Long rollout sequences |
| EP | MoE experts | All-to-All | MoE models (Qwen3-MoE, DeepSeek) |
| CP | Ultra-long seq | Ring Attention | Very long rollouts |

Rank mapping: `global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size`

## TP Layer Pattern
```
ColumnParallel (split output dim) → activation → RowParallel (split input dim + AllReduce)
```
Each Transformer layer: 2x AllReduce (Attention output + MLP FC2).

**Critical for GRPO**: TP communication overhead scales with model size. On single GPU (RTX 4090): TP=1, no communication.

## Sequence Parallelism (SP)
Replaces TP AllReduce with AllGather + ReduceScatter along sequence dim. Same total communication volume but activation memory saved by TP factor. Important for long-context GRPO rollouts.

## Distributed Optimizer (ZeRO-1)
Shards Adam (m, v) across DP ranks. Each rank stores `12/DP` bytes/param.

**Memory per param (bytes)**:
| Config | DP=1 | DP=2 | DP=8 |
|--------|------|------|------|
| No ZeRO | 16 | 16 | 16 |
| ZeRO-1 | 16 | 8 | 5.5 |
| ZeRO-2 | 16 | 7.5 | 3.75 |
| ZeRO-3 | 16 | 4 | 2 |

**For RTX 4090 (24GB)**: ZeRO-2 + CPU_Adam is optimal (verified by our testing).

## Step Flow
```
AllReduce grads → unscale → clip → local optimizer.step() → AllGather params
```

## MoE Pipeline (GRPO for MoE models)
```
Router (Top-K) → Dispatch (permute + A2A) → Experts (grouped GEMM) → Combine (A2A + unpermute)
```
2x All-to-All per MoE layer. EP config: EP=1 (all experts local, no communication), EP=E (each rank 1 expert, max communication).

**For GRPO+MoE on RTX 4090**: EP=1 recommended (all experts local). TP+EP hybrid adds complexity without benefit on single GPU.

## PP Schedule (1F1B)
```
Warmup forward → 1F1B steady (forward+backward alternating) → Cooldown backward
```
Bubble formula: `(stages - 1) / (microbatches + stages - 1)`. Not recommended for GRPO (added latency > benefit).

## Key Files
- `megatron/core/tensor_parallel/layers.py` — ColumnParallelLinear, RowParallelLinear
- `megatron/core/tensor_parallel/mappings.py` — Communication primitives (copy/reduce/gather/scatter)
- `megatron/core/pipeline_parallel/schedules.py` — 1F1B + Interleaved schedules
- `megatron/core/parallel_state.py` — Process group management
- `megatron/core/optimizer/distrib_optimizer.py` — ZeRO-1 sharded optimizer (~3000 lines)
- `megatron/core/transformer/moe/` — MoE router + dispatcher + experts (13 files)

## GRPO Integration Points
- **Loss computation**: `calculate_per_token_loss` (#4590 gradient normalization bug — MUST set `True`)
- **Gradient clipping**: `clip_grads.py` — Muon clipping interaction (#5394, fix in #5395)
- **FP8 training**: `fp8_autocast` — DSV4 failures (#5317 apply_rope_fusion NaN)
