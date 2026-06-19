# SGLang Scheduler/Router Architecture — Deep Reading

> 2026-06-19 | SGLang scheduler.py (4034 lines), tokenizer_manager.py (2998 lines), schedule_batch.py (2799 lines)
> ★★★★★★★★ Key: RadixCache prefix sharing, LoRA extra_key isolation, complete KV flush on weight update, tag-based sleep/wake

---

## 1. Request Lifecycle

```
HTTP Request → TokenizerManager.generate_request() → tokenize → ZMQ PUSH to Scheduler
→ Scheduler.process_input_requests() → handle_generate_request() → Req object → waiting_queue
→ Scheduler.event_loop_normal() → get_next_batch_to_run() → PrefillAdder budget admission
→ run_batch() → model_worker.forward_batch_generation() → process_batch_result()
→ Output via ZMQ → TokenizerManager._wait_one_response() → HTTP SSE to client
```

IPC: TokenizerManager ↔ Scheduler via ZMQ PUSH/PULL. model_update_lock (RWLock) blocks inference during weight updates.

---

## 2. Batch Formation Algorithm

get_next_batch_to_run() (scheduler.py line 2405):
1. Merge prefill results into running_batch (decode batch)
2. Try to schedule new prefill via PrefillAdder (token budget system)
   - Budget = available + evictable - offset - new_token_ratio * running_tokens
   - LoRA scheduling: _can_schedule_lora_req() checks drainer and max_loras_per_batch
3. If no prefill, run decode with ForwardMode.DECODE
4. Mixed chunked prefill: mix_with_running() if enable_mixed_chunk=True

---

## 3. RadixAttention KV Cache Management

### RadixCache Architecture
- Radix tree: TreeNode with key (RadixKey), value (KV slot indices), lock_ref, hit_count
- RadixKey = token_ids + extra_key (LoRA adapter ID namespace)
- ★★★★★★★★ extra_key enables LoRA-aware prefix isolation: different lora_id → separate KV cache trees

### Prefix Matching: match_prefix()
- Walk tree, split nodes at match boundaries
- Returns device_indices (KV slot tensor), last_device_node

### Eviction: priority heap by LRU (last_access_time)
- Evictable leaves (lock_ref==0) popped from heap, KV indices freed

### Lock Management
- inc_lock_ref: protects matched prefix (0→1 transition)
- dec_lock_ref: releases protection (1→0 transition, becomes evictable)

---

## 4. Sleep/Wake for RLHF

### Tag-based release/resume pattern
- tags=["kv_cache"]: KV cache offload only → sleep_level=1 → RTX 4090 OPTIMAL
- tags=["weights"]: base weights offload → sleep_level=2 → AVOID on RTX 4090
- tags=["cuda_graph"]: CUDA graph offload

### ★★★★★★★★ CRITICAL: Complete KV cache flush on every weight update
- flush_cache_after_weight_update() resets ENTIRE radix tree + req_to_token_pool + token_to_kv_pool_allocator
- NO prefix sharing survives across GRPO training steps
- RolloutKV proposal (#28608) aims to pin system prompt KV across weight updates (+768/-5)

### Weight Update Pathways
1. From disk: update_weights_from_disk() — loads checkpoint, reloads model
2. From distributed: update_weights_from_distributed() — torch.distributed.broadcast()
3. From tensor: update_weights_from_tensor() — direct named tensors (verl uses this)
4. From IPC: update_weights_from_ipc() — checkpoint engine ZMQ handles

---

## 5. LoRA Lifecycle for GRPO

- Load: LoRAManager.load_lora_adapter() → validate → load GPU tensors → register
- Unload: LoRAManager.unload_lora_adapter()
- Scheduling: _can_schedule_lora_req() checks drainer + max_loras_per_batch
- LoRAOverlapLoader: async GPU LoRA loading on separate CUDA stream
- LoRADrainer: evicts starving adapters when max_loras_per_batch full
- ★★★★★★★★ extra_key namespace isolation prevents cross-LoRA KV contamination

---

## 6. RTX 4090 GRPO Key Config

| Parameter | RTX 4090 Value | Notes |
|-----------|---------------|-------|
| chunked_prefill_size | 8192 | Limits prefill tokens per batch |
| enable_mixed_chunk | True | Mix decode + prefill |
| max_running_requests | 4-8 | Limited by 24 GiB KV cache |
| enable_lora | True | Required for GRPO |
| max_loras_per_batch | 1 | Single adapter for GRPO |
| enable_memory_saver | True | Required for HYBRID sleep/wake |
| speculative_algorithm | NONE | Spec decode adds memory overhead |
| enforce_eager | True | MANDATORY for DSV4 on SM89 |
| kv_cache_dtype | auto or fp8_e5m2 | FP8 KV reduces memory 2x |

---

## 7. Key RTX 4090 GRPO Architecture Findings

1. ★★★★★★★★ Complete KV cache flush on every weight update → no prefix sharing across steps
2. ★★★★★★★★ LoRA extra_key isolation → separate KV cache trees per adapter
3. ★★★★★★★★ sleep_level=1 = tags=["kv_cache"] → 80x payload reduction → RTX 4090 OPTIMAL
4. MXFP8 MoE cache clobber (#28676) → dict.clear() + weight-load funnel
5. DSA Hadamard loss (#10684) → class variable lost during state transfer
6. GDN degeneracy (#28679) → worsens over uptime, silent corruption

---

## References

- SGLang scheduler: python/sglang/srt/managers/scheduler.py (4034 lines)
- SGLang tokenizer_manager: python/sglang/srt/managers/tokenizer_manager.py (2998 lines)
- RadixCache: python/sglang/srt/mem_cache/radix_cache.py (799 lines)
- LoRAManager: python/sglang/srt/lora/lora_manager.py (824 lines)
- WeightUpdater: python/sglang/srt/managers/scheduler_components/weight_updater.py (272 lines)
- verl sleep/wake: notebook/projects/cross-framework-partial-wake-safety-analysis.md
- RolloutKV: notebook/projects/sglang-28608-rolloutkv-prefix-pin-reading.md
