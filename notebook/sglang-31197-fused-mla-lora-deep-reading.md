# SGLang #31197: Fused MLA kv_b_proj LoRA Correction (4→2 Kernels)

## Overview
- **PR**: sgl-project/sglang #31197 by yoavkantor, July 14, 2026 (Draft)
- **Scope**: Fuses step-A + step-B SGMM kernels for each side (q, v) of MLA kv_b_proj LoRA correction
- **Speedup**: ~1.6× on A100, ~1.6-1.7× on H100 for decode shapes
- **Status**: Draft, 0 human reviews, CI failing (likely infra), needs Kimi-K2.5 nightly confirmation

## Motivation

The absorbed-MLA `kv_b_proj` LoRA correction currently launches **4 Triton kernels**:
- **q side**: step-A SGMM writes small `(S, H, rank)` intermediate → global memory → step-B SGMM reads it back
- **v side**: same pattern

For decode shapes where rank is tiny (16-32), the global memory round trip + second kernel launch **dominate cost**.

## The Fix: Fused Kernels

PR adds `q_side_fused_fwd` / `v_side_fused_fwd` in `kv_b_lora_absorbed.py`:
- Keeps per-`(token-tile, head)` rank intermediate **in registers**
- Performs second contraction in the same Triton program
- Reduces launches from 4 → 2

### Correctness Guarantees
Preserves all existing functionality:
- Per-slot `weight_indices` routing
- `permutation` (SORTED_BY_ADAPTER)
- `use_cuda_graph` segment grid
- Mixed/zero per-slot ranks
- Per-slot `scalings`
- Accumulate-in-place

The rank intermediate is "cast back to the input dtype between the two dots, exactly as the split kernels store/reload it" — preserving numerical equivalence.

### Fallback
Split kernels kept as fallback, triggered when:
1. Padded-rank exceeds `_fuse_max_rank()` (default 64, override via `SGLANG_MLA_LORA_FUSE_MAX_RANK`)
2. Fused launch fails

Crossover thresholds: ~96 (A100), ~128 (H100). Default 64 is conservative.

## Performance

| Metric | Value |
|--------|-------|
| A100 decode shapes (rank 16-32) | **~1.6×** median speedup |
| H100 same shapes | **~1.6-1.7×** speedup |
| Correctness validation | 14/14 configs match upstream + fp32 reference |

## GRPO+LoRA Relevance

- **Directly relevant**: MLA-based models (DeepSeek, Kimi-K2.5) are commonly used for RL training
- **LoRA correction**: The kv_b_proj path is the bottleneck for GDN/MLA LoRA adapters
- **SGLang as rollout engine**: If GRPO training uses SGLang for rollout generation with LoRA, this optimization reduces per-token overhead
- **RTX 4090**: Not tested, but SM89 fallback works (Triton general)
- **Caveat**: Optimization targets A100/H100 decode shapes — RTX 4090 benefit may differ

## Monitoring
- Needs Kimi-K2.5 nightly confirmation (author lacks TP=8 setup)
- 5 reviewers requested: Ying1123, Fridge003, lifuhuang, yushengsu-thu, jybsuper
- CI failure likely infrastructure (gemini-code-assist bot daily quota exceeded)
- Wait for maintainer review after CI passes
