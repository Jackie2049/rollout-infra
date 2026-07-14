# SGLang #31190: SM89 FP8 Paged Indexer Operator Deep Reading

## Overview
**PR**: SGLang #31190 by mochgolf (+603/-0, 2 files, 25 tests)
**Status**: OPEN July 14, 2026. CI running (quota warning).
**Purpose**: Triton FP8 paged indexer for SM89 (Ada Lovelace) GPUs. Enables GLM DSA inference on RTX 4090.

## Architecture

The kernel (`sm89_paged_fp8_index_logits`) performs:

1. **Page lookup**: Given `logical_page` and per-batch `page_table`, resolve to `physical_page`
2. **Sparse FP8 KV read**: Load FP8 keys from physical page with FP32 scale factors
3. **Per-head ReLU + weighted reduction**: `max(per_head_dot, 0) * head_weights` per head, then sum across heads
4. **Scale application**: Multiply by `key_scales` (per-token FP32 scale)
5. **Masking**: Invalid pages/tokens get 0.0 in loaded values → `-inf` in output logits

## Key Parameters (compile-time constants)
```python
_PAGE_SIZE = 64
_NUM_HEADS = 32
_HEAD_DIM = 128
_FP8_DTYPE = torch.float8_e4m3fn  # E4M3
```

## Performance
- **Eager**: 0.068ms at seqlen 32768, 0.068ms at seqlen 262144
- **CUDA Graph replay**: 0.009ms at seqlen 32768, 0.021ms at seqlen 262144
- **Validated**: 2x RTX 4090 48GB mod, GLM-5.2-504B, DSA, FP8 KV, TP=2, 262K-token KV budget

## Testing (25 tests)
- CUDA Graph replay (capture + 2 replay phases with different inputs)
- Numerical: matches reference FP32 implementation
- Masking: invalid positions → -inf, valid positions → finite
- Non-contiguous page table support
- Column device lengths support
- ReLU per-head behavior verification
- Source has no host-device reads (security)

## RTX 4090 GRPO Relevance

This PR is significant for RTX 4090 GRPO training because:

1. **FP8 KV cache on SM89**: First SGLang FP8 kernel validated on Ada architecture (RTX 4090 = SM89). Previously FP8 was only tested on Hopper (SM90) or Blackwell (SM120).

2. **DSA model support**: Enables GLM DSA models on RTX 4090. DSA is the sparse attention architecture used by GLM-5.2-504B and similar models commonly used for RL training.

3. **Memory efficiency**: FP8 KV cache at 1 byte per element vs 2 bytes for FP16/BF16 → 2x KV cache capacity on the same 24GB VRAM. This is critical for GRPO where long rollout sequences consume large KV caches.

4. **CUDA Graph latency**: 0.009ms per kernel call with CUDA Graphs → small enough to not be a bottleneck in the generation pipeline.

## Limitations
- GLM DSA format-specific (not general-purpose)
- Requires follow-up integration with DSA indexer selector (pending)
- FP8 training (as opposed to inference) not addressed — this is inference-only KV cache

## Monitoring
- CI: hourglass (running, quota-limited)
- No human reviews yet
- Wait for CI to pass → maintainer review
