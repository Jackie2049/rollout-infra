---
name: sglang-nav
description: Navigate SGLang source code. Quick reference for subsystem locations, key classes, and data flow.
triggers:
  - sglang source
  - sglang architecture
  - sglang code location
  - sglang scheduler
  - sglang radix
  - sglang radix attention
  - sglang spec decode
  - sglang overlap
  - sglang model runner
  - sglang tokenizer manager
  - sglang kv cache
  - sglang prefill
  - sglang decode
---

# SGLang Source Code Navigator

## Architecture: 3-Layer

```
TokenizerManager (API) → Scheduler (orchestration) → TpModelWorker (GPU execution)
```

## Subsystem Index

### 1. Scheduler (~4034 lines)
- **File**: `sglang/python/sglang/srt/managers/scheduler.py`
- **Notes**: `notebook/projects/sglang-scheduler-reading.md`
- **Key**: Mixin architecture (6 mixins), 18 sub-components, Normal/Overlap dual event loops
- **Mixins**: DisaggregationDecode, DisaggregationPrefill, Multiplex, PP, Dllm, MlxOverlap

### 2. Schedule Policy (~1070 lines)
- **File**: `sglang/python/sglang/srt/managers/schedule_policy.py`
- **Key**: PrefillAdder (admission control), LPM policy (Longest Prefix Match), batch_is_full optimization

### 3. Schedule Batch (~2799 lines)
- **File**: `sglang/python/sglang/srt/managers/schedule_batch.py`
- **Key**: ReqToTokenPool, TokenToKVPoolAllocator, batch merge/rebuild logic

### 4. RadixAttention
- **Files**: `sglang/python/sglang/srt/layers/radix_attention.py` + `sglang/python/sglang/srt/mem_cache/`
- **Notes**: `notebook/fundamentals/sglang-radix-attention.md`
- **Key**: RadixTree (variable-length edge labels, no block alignment), node splitting, 7 eviction policies
- **Data**: `req_to_token_pool[max_running, max_seq_len]` + `token_to_kv_pool_allocator`
- **Flow**: match_prefix → insert(evicted req) → evict(LRU/LFU) → allocate

### 5. Speculative Decoding (~7500 lines)
- **Files**: `sglang/python/sglang/srt/speculative/` (12 core files)
- **Notes**: `notebook/projects/sglang-spec-decode-reading.md`
- **Key**: 6 algorithms(EAGLE/EAGLE3/FrozenKV/DFlash/NGRAM/Standalone) + Plugin registry(SpeculativeAlgorithm.register)
- **Workers**: EAGLEWorkerV2 / DFlashWorkerV2 / FrozenKVMTPWorkerV2 / NGRAMWorker / StandaloneWorkerV2
- **Flow**: Draft → Verify → DraftExtend (3-phase pipeline)
- **SpecInput**: abstract class with SpecInputType(9 subtypes) → is_draft_input()/is_verify_input() dispatch
- **Overlap**: Spec decode + FutureMap overlap → publish draft info mid-forward → schedule prep overlaps

### 6. Overlap Event Loop + FutureMap
- **File**: `sglang/python/sglang/srt/managers/scheduler.py` (event_loop_overlap) + `overlap_utils.py`
- **Key**: FutureMap(pool-indexed relay, zero-copy) + WAR barrier + separate CUDA streams(schedule/forward/copy)
- **Flow**: recv_requests → schedule on schedule_stream → forward on forward_stream → WAR barrier → process_batch_result → overlap GPU/CPU
- **Result**: 20-40% throughput improvement via GPU-CPU parallelism
- **Dynamic**: Consecutive prefill overlap disabled(SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP) for TTFT

### 7. Model Runner
- **File**: `sglang/python/sglang/srt/managers/model_runner.py` (TpModelWorker)
- **Key**: Forward batch generation, weight loading, CUDA Graph integration

### 8. Tokenizer Manager (API Layer)
- **File**: `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- **Key**: HTTP/gRPC API, request routing, tokenization

### 9. Memory Management + HiCache
- **Files**: `sglang/python/sglang/srt/mem_cache/` (radix_cache.py, unified_radix_cache.py, chunk_cache.py)
- **Key**: RadixTree for KV cache indexing + HiCache(3层:GPU+CPU+SSD) + async offload + HiSparse coordinator
- **UnifiedRadixCache**: RadixTree + HiCache → decode async KV offload to CPU/SSD → GPU memory freed

### 10. Communication Layer
- **Files**: `sglang/python/sglang/srt/layers/` (communication via detensorizer)
- **Key**: TP/PP/EP support, NCCL integration

### 11. PD Disaggregation
- **Files**: `sglang/python/sglang/srt/disaggregation/` (5 transfer backends)
- **Key**: DisaggregationMode(NULL/PREFILL/DECODE) → PrefillWorker+DecodeWorker+KVSender+KVReceiver
- **Backends**: NIXL(RDMA零拷贝) + Mooncake(RDMA+全局KV) + MORI + fake(测试)
- **PDMux**: 单GPU时间片调度 → prefill→decode切换 → 无需2台机器

### 12. EnvField API
- **File**: `sglang/python/sglang/srt/environ.py` (1061 lines!)
- **Key**: EnvField(.get/.set/.override context manager/.is_set/.clear) + EnvStr/EnvBool/EnvInt/EnvTuple
- **Naming**: SGLANG_* prefix mandatory; ENABLE/DISABLE/USE/FORCE + MAX/MIN/NUM/SIZE + DIR/PATH/PORT
- **temp_set_env**: rejects SGLANG_* keys by default → use allow_sglang=True only for special cases
- **Cache dirs**: SGLANG_CACHE_DIR=~/.cache/sglang; Issue #19612 to unify fragmented caches

## Key Data Flow

```
User Request
  → TokenizerManager (API + tokenize)
  → Scheduler (Normal or Overlap event loop)
    → PrefillAdder (admission control + LPM policy)
    → RadixCache.match_prefix() → prefix hit
    → Batch merge (prefill into running_batch)
    → TpModelWorker.forward_batch_generation()
      → Model forward → Sampling → Output
  → TokenizerManager (detokenize)
  → User Response
```

## Critical Paths

- **Prefill**: tokenize → match_prefix → allocate KV slots → forward → write KV
- **Decode**: running_batch → forward → sample → check stop → update output
- **Spec Decode**: draft(tree) → verify(target forward) → draft_extend(catch up KV)
- **Overlap**: prefill_batch + running_batch → merge → forward → split results
- **RadixAttention**: match_prefix(inserted reqs) → allocate → evict(if needed) → compute
- **Prefix Caching**: insert(completed req) → match(new req) → reuse KV slots

## Comparison with vLLM

| Feature | SGLang | vLLM V1 |
|---------|--------|---------|
| KV Cache | RadixTree (variable edges) | BlockHash (fixed blocks) |
| Prefix Match | Longest Prefix Match | Block-aligned hash |
| Scheduling | Merge prefill into running | Replace prefill batch |
| Spec Decode | 6 algorithms + plugin registry + overlap | Proposer-Scorer dual layer |
| Event Loop | 6 loops: Normal/Overlap/PDMux/PP/DisaggPrefill/DisaggDecode | Single step() loop |
| Memory Pool | req_to_token + token_to_kv + HiCache(3层) | BlockPool + FreeQueue |
| PD Separation | Full(Prefill+Decode+KV Transfer+Bootstrap+PDMux) | Experimental |
| Env Vars | EnvField API(1061 lines) + strict naming | os.getenv |
