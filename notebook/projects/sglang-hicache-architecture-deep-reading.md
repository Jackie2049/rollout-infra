# SGLang HiCache Architecture: 3-Layer KV Cache Offloading Deep Reading

**Date**: 2026-07-15 (Session 10)
**Purpose**: Understand SGLang's 3-layer GPU→CPU→SSD KV cache hierarchy for production serving
**Sources**: SGLang sglang/srt/mem_cache/ source, SGLang nav skill, overlap event loop reading

---

## 1. HiCache Architecture Overview

```
HiCache (Hierarchical Inference Cache) — 3-layer KV cache storage:

Layer 1: GPU Memory (VRAM)
  → Fastest access: ~0.01ms per token lookup
  → Limited capacity: ~3 GiB for KV cache on RTX 4090 (24 GiB total)
  → RadixTree indexing → prefix sharing → reduced storage
  → Alloc: req_to_token_pool + token_to_kv_pool_allocator

Layer 2: CPU Memory (DRAM)
  → Medium speed: ~1-5ms per token transfer (PCIe bandwidth)
  → Large capacity: host RAM (32-128 GiB typical)
  → Async offload: decode kernel → copy_stream → pinned memory → CPU tensor
  → UnifiedRadixCache manages GPU+CPU tree entries

Layer 3: SSD (NVMe)
  → Slowest: ~10-50ms per page (4KB random read)
  → Massive capacity: 1-4 TiB
  → Sequential read optimization (not random access)
  → Long-term storage for rarely-used prefixes

Data flow:
  Hot prefixes → GPU (RadixTree leaves)
  Warm prefixes → CPU (pinned memory, async offload during decode)
  Cold prefixes → SSD (evicted from CPU, reloaded on demand)

Key innovation: HiSparse coordinator manages async offload/reload
  without blocking the decode loop → overlap copy_stream with forward_stream
```

---

## 2. UnifiedRadixCache: GPU + CPU Combined Tree

```
UnifiedRadixCache extends RadixTree with CPU memory pool:

GPU Pool:
  req_to_token_pool[max_running, max_seq_len]
    → Maps each running request to its token positions
    → Indexed by request RID

  token_to_kv_pool_allocator
    → Allocates KV cache slots from FreeQueue
    → O(1) doubly-linked list for allocation/deallocation

CPU Pool:
  cpu_req_to_token_pool[max_cpu_requests, max_seq_len]
    → Mirrors GPU structure but in CPU pinned memory
    → Uses cudaMemcpyAsync for GPU→CPU transfer

  cpu_token_to_kv_pool_allocator
    → CPU-side allocation for offloaded KV blocks

Shared Tree:
  RadixTree nodes can reference either GPU or CPU pools
  → node.pool_type = "gpu" or "cpu"
  → match_prefix traverses both pools transparently
  → Prefix hit on CPU → async reload to GPU before compute
```

---

## 3. Async Offload Pipeline

```
During decode, KV cache blocks are asynchronously offloaded to CPU:

decode_step():
  1. Forward pass on GPU (forward_stream)
  2. New KV slots allocated on GPU
  3. HiSparse.check_offload() → determine which blocks to move
  4. If offload needed:
     → copy_stream: cudaMemcpyAsync(GPU→CPU, pinned buffer)
     → WAR barrier ensures forward_stream completes before copy starts
     → CPU allocation: cpu_token_to_kv_pool_allocator.alloc()
     → GPU release: token_to_kv_pool_allocator.free()
  5. Continue decode on GPU (remaining hot blocks)

Timing overlap:
  - Forward: ~0.5-2ms per decode step (GPU)
  - Offload: ~0.5-1ms per block (copy_stream, overlapped with next forward)
  - Net impact: ~0-5% throughput reduction (overlap hides most cost)

★★★★★★★★ For RTX 4090 GRPO: HiCache offload during rollout generation
  - During decode: KV cache grows → eventually exceeds GPU budget
  - HiCache moves cold KV blocks to CPU → frees GPU for more decode steps
  - This enables longer rollout sequences (up to 2048+ tokens) on 24 GiB GPU
  - WITHOUT HiCache: RTX 4090 can only fit ~512 tokens of KV cache for 7B model
```

---

## 4. HiSparse Coordinator: Load-Balancing GPU/CPU

```
HiSparse decides which KV blocks to keep on GPU vs offload to CPU:

Decision criteria:
  1. Block access frequency (LRU/LFU tracking)
     → Frequently accessed prefixes stay on GPU
     → Rarely accessed prefixes move to CPU

  2. Block age (time since last access)
     → Recent blocks: GPU
     → Old blocks: candidate for offload

  3. Request priority (active vs completed)
     → Active requests: GPU (need KV for decode)
     → Completed requests: candidate for CPU/SSD eviction

  4. Memory pressure
     → GPU > 90% capacity: aggressive offload
     → GPU < 50% capacity: no offload needed

Strategy:
  - Decode phase: offload completed requests' KV to CPU
  - Between decode phases: HiSparse evaluates all blocks
  - Prefill phase: reload matching CPU blocks to GPU (prefix hit)

★★★★★★★★ This is EXACTLY the pattern needed for GRPO sleep/wake:
  During training: model weights need GPU → KV cache evicted
  During rollout: KV cache needs GPU → model weights can share GPU
  HiCache manages this trade-off automatically for serving
  But for GRPO training, verl HYBRID uses sleep/wake instead
```

---

## 5. PD Disaggregation + HiCache

```
In PDMux (single-GPU time slicing) + HiCache:

Phase 1: Prefill (compute-heavy)
  → Load full model weights on GPU
  → Process prompts → generate initial KV cache
  → HiCache: all new KV on GPU (hot)

Phase 2: Switch to Decode (memory-heavy)
  → Keep model weights on GPU (smaller decode batches)
  → Decode step → new KV tokens added
  → HiCache: offload completed requests' KV to CPU → free GPU slots
  → More decode capacity without increasing GPU memory

Phase 3: Back to Prefill (next batch)
  → HiCache: keep warm prefixes on CPU for prefix reuse
  → Only reload matching prefixes → partial GPU→CPU→GPU cycle

★★★★★★★★ PDMux + HiCache = single-GPU GRPO serving enabler
  - Prefill phase: generate responses (group_size per prompt)
  - Decode phase: complete all responses
  - HiCache: manage KV cache across phases → maximum utilization
  - Sleep/wake not needed → HiCache handles memory pressure
```

---

## 6. Comparison: HiCache vs vLLM CuMem vs DeepSpeed ZeRO Offload

| Feature | SGLang HiCache | vLLM CuMem | DeepSpeed ZeRO Offload |
|---------|---------------|-----------|------------------------|
| **Layers** | 3 (GPU→CPU→SSD) | 2 (GPU→CPU, cuMem unmap) | 2 (GPU→CPU, optimizer states) |
| **KV cache** | Yes (RadixTree nodes) | Yes (block-based) | No (model weights only) |
| **Async** | Yes (copy_stream + WAR) | Yes (cuMemAsync) | Yes (async prefetch) |
| **Prefix sharing** | Yes (RadixTree) | Yes (BlockHash) | No |
| **Training support** | No (inference only) | Sleep/wake for training | Training-native |
| **RTX 4090** | Rollout KV offload | Sleep/wake weight offload | ZeRO-2 optimizer offload |
| **Memory freed** | KV blocks (~50-200 MiB) | Weights (~14 GiB) | Optimizer (~28 GiB) |

**Key insight**: HiCache and CuMem solve DIFFERENT problems:
- HiCache: KV cache memory during serving/rollout
- CuMem: model weight memory during training/sleep
- DeepSpeed: optimizer state memory during training

For GRPO on RTX 4090, we need ALL THREE:
  1. HiCache (or equivalent) for KV cache during rollout generation
  2. CuMem sleep/wake for weight memory during training
  3. ZeRO-2 CPU offload for optimizer states during training

```
Memory budget (RTX 4090, 24 GiB):

Rollout phase (SGLang serving):
  Model weights: 14 GiB (GPU)
  KV cache (active): 2-3 GiB (GPU, HiCache manages)
  KV cache (offloaded): unlimited (CPU+SSD, HiCache manages)

Training phase (verl HYBRID):
  sleep(level=1): release KV → model stays on GPU
  ZeRO-2: optimizer on CPU → 0 GiB GPU for optimizer
  Activations (bypass): 3.8 GiB GPU
  Gradients: 1.4 GiB GPU
  Total training: 19.2 GiB → fits!
```

---

## 7. HiCache Limitations for GRPO

```
1. Inference-only: HiCache does NOT support training weight updates
   → Cannot use HiCache for model weight offload during training
   → Need CuMem or ZeRO for training

2. SSD latency: 10-50ms per page → too slow for real-time serving
   → Only useful for cold prefixes that won't be accessed soon
   → Not useful for active GRPO rollout

3. CPU reload latency: 1-5ms per block → acceptable for serving
   → But adds latency to prefix reuse during rollout
   → For GRPO: prefix reuse between rollout steps → some latency cost

4. Memory fragmentation: CPU pinned memory is limited
   → cudaMallocHost for pinned memory → limited by host RAM
   → Fragmentation from variable-length RadixTree edges
   → UnifiedRadixCache handles this with pool allocation

★★★★★★★★ HiCache is BEST for serving, NOT for training
  For GRPO training on RTX 4090: use HiCache for rollout KV offload
  For GRPO training: use ZeRO-2 + sleep/wake for weight/optimizer management
```

---

## 8. RTX 4090 HiCache Configuration

```
For GRPO rollout on RTX 4090 with SGLang:

SGlang config:
  --enable-hicache              # Enable HiCache for KV offload
  --hicache-ratio 0.5           # 50% of freed GPU slots go to HiCache
  --hicache-watermark 0.9       # Offload when GPU > 90% capacity
  --gpu-memory-utilization 0.85 # Leave 15% headroom for training

Result:
  - 7B model weights: ~14 GiB (GPU)
  - Active KV cache: ~2 GiB (GPU, for running requests)
  - Offloaded KV: CPU DRAM (unlimited)
  - Total GPU: ~16 GiB → 8 GiB headroom for training activations

★★★★★★★★ HiCache + sleep/wake = complete RTX 4090 GRPO pipeline
  1. Rollout: SGLang + HiCache → generate responses, KV managed
  2. Sleep(level=1): release KV cache → keep model weights
  3. Training: ZeRO-2 + CPU optimizer → update model
  4. Wake: reload model (same weights, just updated) → next rollout
  5. Delta sync: only LoRA adapter weights (~50 MiB) → fast
```

---

## Session Stats
- **HiCache architecture**: 3-layer GPU→CPU→SSD KV cache hierarchy analyzed
- **UnifiedRadixCache**: combined GPU+CPU tree structure
- **HiSparse coordinator**: load-balancing decision criteria
- **PD Disaggregation + HiCache**: PDMux time-slicing + KV offload
- **Cross-framework comparison**: HiCache vs CuMem vs ZeRO offload
- **RTX 4090 GRPO pipeline**: HiCache + sleep/wake + ZeRO-2 complete
