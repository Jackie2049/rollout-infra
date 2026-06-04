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
- **Key**: Dual-path dispatch (builtin enum + plugin registry), EAGLE v1/v2, N-gram, DFlash
- **Workers**: EAGLEWorker(v1) / EAGLEWorkerV2(overlap) / DFlashWorker / NGRAMWorker
- **Flow**: Draft → Verify → DraftExtend (3-phase pipeline)
- **EAGLE**: Tree-structured speculation, draft model reuses target's hidden states

### 6. Overlap Event Loop
- **File**: `sglang/python/sglang/srt/managers/scheduler.py` (OverlapSchedulerMixin)
- **Key**: Separate prefill/decode paths, batch merging (not replacing), adaptive new_token_ratio
- **Flow**: new_token_ratio tracking → batch weight adjustment → merge prefill into running_batch

### 7. Model Runner
- **File**: `sglang/python/sglang/srt/managers/model_runner.py` (TpModelWorker)
- **Key**: Forward batch generation, weight loading, CUDA Graph integration

### 8. Tokenizer Manager (API Layer)
- **File**: `sglang/python/sglang/srt/managers/tokenizer_manager.py`
- **Key**: HTTP/gRPC API, request routing, tokenization

### 9. Memory Management
- **Files**: `sglang/python/sglang/srt/mem_cache/` (radix_cache.py, chunk_cache.py)
- **Key**: RadixTree for KV cache indexing, token pool allocation, LRU eviction

### 10. Communication Layer
- **Files**: `sglang/python/sglang/srt/layers/` (communication via detensorizer)
- **Key**: TP/PP/EP support, NCCL integration

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
| Spec Decode | EAGLE v1/v2 + overlap | Proposer-Scorer dual layer |
| Event Loop | Normal/Overlap dual mode | Single step() loop |
| Memory Pool | req_to_token + token_to_kv | BlockPool + FreeQueue |
