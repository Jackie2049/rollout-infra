# SGLang Architecture: Core Components for GRPO

## Overview
SGLang uses a 3-layer architecture: `TokenizerManager (API) → Scheduler (orchestration) → TpModelWorker (GPU)`. The Scheduler is the heart, with 6 mixins and dual event loops.

## Scheduler: Dual Event Loop Architecture
The Scheduler (~4034 lines) runs one of 6 event loops:
- **Normal**: Single-threaded orchestration
- **Overlap**: GPU-CPU parallel (20-40% throughput improvement) — critical for GRPO rollout
- **PDMux**: Time-slice GPU between prefill/decode (no separate machines needed)
- **PP**: Pipeline parallel support
- **DisaggPrefill/DisaggDecode**: Separate prefill/decode workers

**Overlap mode** is the most relevant for GRPO:
- Uses separate CUDA streams: schedule_stream, forward_stream, copy_stream
- FutureMap (pool-indexed relay, zero-copy) for GPU→CPU communication
- WAR barrier for write-after-read safety
- Consecutive prefill overlap can be disabled via `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` (for TTFT)

## RadixAttention vs vLLM BlockHash

| Feature | SGLang RadixAttention | vLLM V1 |
|---------|----------------------|---------|
| Structure | RadixTree (variable edge labels) | BlockHashToBlockMap (1:N) |
| Prefix match | Longest Prefix Match | Block-aligned hash |
| Memory pool | req_to_token + token_to_kv | BlockPool + FreeQueue |
| Eviction | 7 policies (LRU, LFU, etc.) | LRU via FreeKVCacheBlockQueue |
| HiCache | 3-layer: GPU+CPU+SSD | No equivalent |

For GRPO: RadixAttention's prefix caching is more flexible for variable-length sequences common in RL rollouts.

## Speculative Decode (6 algorithms)
Relevant for GRPO throughput:
- **EAGLE/EAGLE3**: Draft model (requires separate model)
- **DFlash**: Non-causal bidirectional draft (~17% acceptance)
- **FrozenKV**: Reuse KV cache as draft (zero overhead)
- **NGRAM**: Zero-overhead n-gram draft
- **Standalone**: Independent draft model

**For RTX 4090 GRPO**: NGRAM or FrozenKV are most practical (no extra model memory).

## Key GRPO-Relevant Paths

1. **Rollout generation**: TokenizerManager → Scheduler → TpModelWorker.forward_batch_generation
2. **KV cache management**: RadixCache.match_prefix → allocate → evict(LRU/LFU)
3. **Triton kernel execution**: Model runner dispatches to FlashInfer/Triton backends
4. **FP8 KV cache**: Paged FP8 read with FP32 scale factors (SGLang #31190 pathway)

## Key Files
- `sglang/srt/managers/scheduler.py` (~4034 lines, 6 mixins) — orchestration core
- `sglang/srt/managers/schedule_policy.py` (~1070 lines) — admission control
- `sglang/srt/managers/schedule_batch.py` (~2799 lines) — batch management
- `sglang/srt/layers/radix_attention.py` — RadixAttention backend
- `sglang/srt/mem_cache/` — KV cache (RadixTree, HiCache, ChunkCache)
- `sglang/srt/managers/model_runner.py` — TpModelWorker GPU execution
