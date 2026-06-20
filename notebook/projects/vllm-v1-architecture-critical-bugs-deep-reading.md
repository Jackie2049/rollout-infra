# vLLM V1 Architecture & Critical Bugs Deep Reading Report

> 2026-06-20 | Comprehensive synthesis of V1 architecture source code + 7 critical bugs + RTX 4090 impact + cross-framework connections
> Scope: Architecture deep read (Engine Core, Scheduler, GPUModelRunner, KV Cache, Memory, LoRA) + Issues #46125, #45552, #46204, #46203, #46195, #45979, #46118

---

## Part I: V1 Architecture Deep Reading

### 1. Engine Core — One-Process-Per-GPU, ZMQ IPC

**Source**: `vllm/v1/engine/core.py` (~1600 lines)

The V1 engine fundamentally differs from V0 by eliminating the separate worker process model. The core architecture:

```
Frontend Process (AsyncLLM)
    |  ZMQ DEALER/ROUTER (msgpack serialization)
    v
EngineCore Process (isolated for GIL)
    |  Three threads:
    |  - input_thread: ZMQ socket polling, deserialization
    |  - output_thread: msgpack encoding, ZMQ PUSH
    |  - main_thread: core_busy_loop (schedule -> execute -> sample -> update)
```

**Key V0 -> V1 changes**:
- V0: separate Worker process per GPU, communicated via pickle RPC
- V1: EngineCore in same process as model execution, ZMQ IPC only for frontend
- Rationale: GIL isolation without pickle overhead; zero-copy msgpack; multiple API servers share one EngineCore (DP serving)

**The `step()` method** (core.py:443-472):
```python
def step(self):
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**Critical design**: `execute_model()` returns a Future with `non_block=True`. CPU computes grammar bitmask while GPU executes forward pass. This is the V1 pipeline overlap mechanism -- forward on GPU, grammar on CPU, then sample on GPU.

**Abort handling**: Dual-queue design (aborts_queue for urgent aborts checked between steps, input_queue for normal flow). Ensures abort is processed quickly even during model execution.

---

### 2. Scheduler — Unified Token Budget, No Phase Distinction

**Source**: `vllm/v1/core/sched/scheduler.py` (~2375 lines)

Woosuk's design philosophy comment (scheduler.py:342-350):
> "No concept of 'decode phase' or 'prefill phase'. Each request only has `num_computed_tokens` and `num_tokens_with_spec`. Each step tries to allocate tokens so `num_computed_tokens` catches up to `num_tokens_with_spec`. This is general enough to cover chunked prefill, prefix caching, and speculative decoding."

**Three core queues** (scheduler.py:157-168):
- `waiting`: new requests (WAITING status)
- `skipped_waiting`: requests blocked by async dependencies (KV loading, grammar, streaming)
- `running`: currently executing requests (plain list, NOT priority queue)

**schedule() flow** (scheduler.py:339-952):
1. RUNNING requests: compute num_new_tokens, allocate_slots, preempt on OOM
2. WAITING requests: prefix cache lookup, KV connector check, allocate_slots (NO preemption!)
3. Post-process: assert constraints, compute common prefix blocks, build SchedulerOutput

**Token budget management**:
- Initial: `token_budget = max_num_scheduled_tokens` (default 2048)
- RUNNING phase: each request consumes budget
- WAITING phase: each request consumes budget; if allocate_slots fails -> break (no preempt!)
- Preempted: token_budget += recovered tokens

**Preemption strategy**: V1 uses RECOMPUTE (no swap). Preempted request: free all KV blocks, reset num_computed_tokens=0, prepend to waiting queue. FCFS: pop last-added running request. PRIORITY: max(priority, arrival_time).

**BudgetRefiner integration point** (scheduler.py:407):
```python
token_budget = self.max_num_scheduled_tokens  # CURRENT: static
# Future: BudgetRefiner.adjust_prefill_budget() dynamic per-step
```

**Watermark mechanism**: watermark_blocks only applied to WAITING/PREEMPTED requests (scheduler.py:200). RUNNING requests skip watermark -- prevents admission starvation.

---

### 3. GPUModelRunner — Two-Phase Execution, Persistent Buffers

**Source**: `vllm/v1/worker/gpu_model_runner.py` (~7400 lines)

**Class architecture**: GPUModelRunner extends 3 Mixins (LoRA + KVConnector + ECConnector).

**Persistent CUDA Graph Buffers** (gpu_model_runner.py:714-760):
All critical tensors pre-allocated at fixed GPU addresses:
- `input_ids`, `positions`, `query_start_loc`, `seq_lens`, `num_computed_tokens`
- `req_indices`, `prev_positions`, `num_scheduled_tokens`
- `discard_request_mask`, `num_accepted_tokens`

Design: fixed addresses -> CUDA graph replay reads/writes directly. Dynamic values: CPU computes -> copy_to_gpu() -> written to fixed addresses.

**ExecuteModelState** (gpu_model_runner.py:401-414):
Two-phase execution: `execute_model()` returns None (forward only), `sample_tokens()` returns ModelRunnerOutput. This separation allows forward on GPU while CPU prepares next step.

**execute_model() 3-phase pipeline** (gpu_model_runner.py:4002-4363):
1. Preprocess: _update_states, _prepare_inputs, _determine_batch_execution_and_padding, _get_slot_mappings, _build_attention_metadata, _preprocess
2. Forward: self.model() inside set_forward_context()
3. Post-forward: extract hidden_states, compute logits, pack ExecuteModelState, deferred spec decode corrections

**_prepare_inputs()** (gpu_model_runner.py:1868-2187):
- Position IDs: `positions = num_computed_tokens[req_indices] + query_offset`
- Input IDs: 2D token_ids_cpu_tensor flattened, then torch.index_select by position
- Slot Mapping: Triton kernel `_compute_slot_mapping_kernel` (all-GPU, no CPU roundtrip)
- logits_indices: only compute logits at needed positions (decode=last token per request, spec=bonus+target verification)

**InputBatch** (gpu_input_batch.py):
CPU/GPU dual buffering. CPU prepares token_ids, sampling_params, block_tables; GPU copies via copy_to_gpu() to fixed addresses. This enables pipeline overlap: CPU preparing next step while GPU executing current step.

**Sampler 9-step pipeline** (sampler.py:67-144):
1. Logprobs save, 2. Float32 conversion, 3. apply_logits_processors (allowed_token_ids whitelist, bad_words, min_tokens, penalties, thinking_budget), 4. sample (greedy/random/mixed paths), 5-6. Int64->Int32 conversion, 7-9. output formatting

**sample_tokens() 4-phase** (gpu_model_runner.py:4381-4640):
1. Sample logits, 2. Speculative decoding (GPU proposers vs CPU proposers), 3. Bookkeeping (_bookkeeping_sync), 4. Output (ModelRunnerOutput or AsyncGPUModelRunnerOutput)

**CUDA Graph dispatch** (cudagraph_dispatcher.py):
CudagraphDispatcher manages FULL/PIECEWISE/NONE modes based on batch_size + LoRA + uniform_decode. FULL: entire forward captured. PIECEWISE: torch.compile + breakable graph. NONE: eager fallback.

---

### 4. KV Cache Management — BlockPool, Prefix Caching, Hybrid Groups

**Source**: `vllm/v1/core/block_pool.py` (527 lines) + `kv_cache_manager.py` (614 lines) + `kv_cache_coordinator.py`

**BlockPool** -- the physical memory manager:
- FreeKVCacheBlockQueue: custom doubly-linked list (not Python deque -- O(1) mid-removal for prefix hit)
- LRU eviction: least recently used at front, newly freed at tail
- BlockHashToBlockMap: dict[hash -> KVCacheBlock | dict[block_id -> KVCacheBlock]] -- avoids inner dict for 99%+ single-hit blocks, reduces GC pressure
- KVCacheBlock: metadata only (block_id, ref_cnt, block_hash, free list pointers); actual data in GPU tensor indexed by block_id

**Prefix caching**:
- hash = hash_function((parent_block_hash, curr_block_token_ids, extra_keys))
- Chained hashing: incremental, O(1) per block, not O(prefix_length)
- extra_keys: mm_hash, lora_name, cache_salt, prompt_embeds
- **NOTE**: hash does NOT include LoRA adapter ID -> cross-adapter sharing possible -> incorrect for multi-tenant serving

**allocate_slots() 5-stage layout**:
```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```
- comp: existing prefix blocks -> touch (ref_cnt+1)
- new_comp: new cache hits -> allocate, then cache
- ext_comp: external (PD transfer) blocks -> buffer for async transfer
- new: tokens to compute -> allocate fresh blocks
- lookahead: speculative decoding extra blocks

**HybridKVCacheCoordinator**: fixed-point algorithm for multi-attention-type models (Full + SWA + MLA + Mamba). Each group shares same BlockPool. Iterative convergence: each group shrinks candidate hit length until stable.

**V0 -> V1 key differences**:
1. No CPU/GPU swap (V1: preemption=recompute; V0: BlockSpaceManager+SwapMap)
2. Single BlockPool (flat doubly-linked list; V0: separate GPU/CPU pools)
3. Prefix caching integrated into BlockPool (V0: separate prefix manager)
4. KVCacheCoordinator for multi-group support (V0: single attention type only)
5. Preemption = full reset: num_computed_tokens=0 (SGLang: radix tree preserves prefix)

---

### 5. Memory Management — CuMemAllocator, Sleep/Wake, Tag System

**Source**: `vllm/device_allocator/cumem.py` + `vllm/v1/worker/gpu_worker.py` + `docs/features/sleep_mode.md`

**CuMemAllocator**: CUDA virtual memory pool using cuMemMap/cuMemUnmap.
- Physical backing released during sleep, virtual address preserved
- wake_up: re-map same virtual address -> PyTorch tensors still valid
- Incompatible with PyTorch expandable_segments (#147851)

**Sleep levels**:
| Level | What happens | CPU RAM needed | Wake-up speed |
|-------|-------------|---------------|--------------|
| S1 (level=1) | Offload weights to CPU, discard KV | Full model weights | Fast (reload + reallocate KV) |
| S2 (level=2) | Discard weights AND KV, save buffers only | Only buffers (RoPE, norms) | Slow (reallocate + reload/update weights) |

**S2 buffer preservation** (gpu_worker.py:165-200):
```python
if level == 2:
    self._sleep_saved_buffers = {
        name: buffer.cpu().clone() for name, buffer in model.named_buffers()
    }
```
CRITICAL: Without buffer restoration, RoPE cos/sin tables = zeros -> garbage output. FP8 KV scales reset to 1.0 after wake (potential accuracy issue for calibrated models like DSV4).

**Wake-up tag system**:
```python
llm.sleep(level=2)
llm.wake_up(tags=["weights"])       # Reallocate weight memory ONLY
llm.collective_rpc("reload_weights") # Load new weights in-place
llm.wake_up(tags=["kv_cache"])      # Now allocate KV cache
```
Two-phase wake avoids OOM: peak = max(weights, weights+kv), not weights+kv simultaneously.

**verl integration flow**:
```
1. wake_up(tags=["weights"]) -> reallocate weight memory
2. update_weights_from_ipc -> load updated weights (ZMQ or CUDA IPC)
3. clear_kv_cache -> invalidate stale prefix cache (RLHF critical!)
4. wake_up(tags=["kv_cache"]) -> reallocate KV cache
5. Generate rollouts
6. sleep() -> free GPU memory for training
7. Training step -> update weights in shared memory
8. Repeat
```

**sleep_level=1 for LoRA adapter path** (verl engine_workers.py:719-720):
```python
if not self.peft_merge and peft_config is not None:
    self.rollout.sleep_level = 1  # Only release KV, keep base weights
```
sleep_level=1: base weights stay resident, only LoRA deltas transferred (~200 MiB vs ~16 GiB -> 80x reduction). Also AVOIDS CuMemAllocator bug (#45552) since LoRA offload doesn't use cuMem unmap.

---

### 6. LoRA Serving — Punica SGMV, Fixed-Address Buffers, CUDA Graph Compatible

**Source**: `vllm/lora/` + `vllm/v1/worker/lora_model_runner_mixin.py`

**Punica SGMV (Segmented Grouped Matrix Vector)**:
- per-token LoRA mapping: LoRAMapping.index_mapping (per-token adapter slot index)
- shrink kernel: tokens sorted by LoRA ID -> batched GEMM per adapter -> [tokens, rank]
- expand kernel: batched -> add to base output in-place -> [tokens, output_dim]
- V1 does NOT merge LoRA into base weights -> fully dynamic multi-tenant

**Fixed-address GPU buffers** (base_linear.py:128-149):
```python
lora_a_stacked = torch.zeros(max_loras, 1, rank, input_size)  # fixed address
lora_b_stacked = torch.zeros(max_loras, 1, output_size, rank)  # fixed address
```
set_lora()/reset_lora() use .copy_() -> modify content, not address -> CUDA graph replay safe.

**LoRA + prefix caching incompatibility**:
- Prefix cache hash does NOT include LoRA adapter ID
- Request A (LoRA-1) and Request B (LoRA-2) could share same prefix block -> wrong KV values
- GRPO safe: rollout_n=8 copies all use same adapter -> prefix shareable
- Multi-tenant serving: different adapters -> prefix caching incompatible

**LoRA + CUDA Graph** (cudagraph_dispatcher.py:115-134):
- specialize_active_lora=False (default): capture 1 case with max_loras+1 slots
- specialize_active_lora=True: capture powers-of-2 adapter counts -> more graphs, better perf
- Warmup: maybe_setup_dummy_loras() -> zero-filled adapters at lora_warmup_rank

---

## Part II: Critical Issues Deep Read

### 7. #46125 — Stale Encoder Cache Revert (DANGEROUS for RLHF/GRPO)

**Status**: OPEN, NOT merged. Revert of #45093 (which added cache reset after weight update).

**Why #45093 was correct**: Encoder cache entries keyed only by mm_hash (NOT weight version). After weight update, old encoder outputs are stale. Prefix cache KV blocks computed with old weights. Serving stale state = silent corruption.

**Why #46125 revert is DANGEROUS**:
- RL training updates weights EVERY step
- Without cache reset, stale KV/encoder outputs persist across weight updates
- Silent corruption: no NaN, just subtly wrong logits -> GRPO advantage on corrupted outputs
- Same pattern as SGLang #28676 (MoE cache clobbered on weight reload -> 64x accuracy blowup)

**RTX 4090 impact**: CRITICAL. GRPO training with vLLM + weight updates MUST reset prefix cache + encoder cache after every weight update. If #46125 merges, vLLM's RLHF weight-update path silently serves stale state.

**Correct approaches**:
- Option A: Keep #45093 but make it configurable (`reset_cache=True` default)
- Option B: Add weight version to cache keys (architectural fix, requires refactoring)
- Option C: verl-style lifecycle (free + re-allocate KV cache, inherently invalidates stale state)

**Source references**:
- `vllm/v1/engine/async_llm.py`: finish_weight_update() with reset_prefix_cache() + reset_encoder_cache()
- `vllm/v1/core/sched/scheduler.py:1923-1971`: reset_prefix_cache() implementation
- `vllm/v1/core/encoder_cache_manager.py`: encoder cache keyed by mm_hash only

---

### 8. #45552 — CuMem Stream Sync Bug ("200 Lie" Pattern)

**Status**: OPEN (+256/-0). 2-line fix: add `torch.cuda.synchronize()` before cuMemUnmap and after H2D restore.

**Root cause**: CuMemAllocator.sleep() and wake_up() missing CUDA stream synchronization barriers. In-flight decode kernels race with cuMemUnmap/cudaMemcpy -> cudaErrorIllegalAddress.

**The "200 Lie" pattern**: HTTP /sleep endpoint returns 200 OK while engine has already crashed. Client believes operation succeeded, but subsequent operations fail. In RLHF: trainer calls /sleep, gets 200, engine is crashed, next /wake_up fails, training pipeline stalls.

**Sleep crash path**:
```
1. HTTP /sleep request
2. pause_scheduler(mode="abort") -> Python flag only
3. No CUDA stream drain -> in-flight kernels keep running
4. CuMemAllocator.sleep() -> cuMemUnmap on memory kernel is writing -> READ-BEFORE-WRITE-COMPLETE race
5. cudaErrorIllegalAddress -> engine crash
6. HTTP 200 OK returned -> THE "200 LIE"
```

**RTX 4090 impact**: CRITICAL BLOCKER. verl HYBRID mode does sleep/wake EVERY training step. Without sync: crashes within first 1-3 steps.

**Workaround hierarchy**:
1. sleep_level=1 (LoRA adapter path): completely avoids CuMemAllocator bug (LoRA offload uses regular tensor.to(cpu), not cuMem unmap)
2. Patch #45552 locally: add torch.cuda.synchronize() in cumem.py (~5ms overhead per cycle, negligible)
3. verl integration wrapper: add synchronize() in verl's sleep/wake hooks
4. Disable cumem: use regular memory management (higher memory usage)

**Pattern family connection**:
- Same root cause as DeepSpeed #8061 (overlap_comm multi-stream race), verl #6794 CRITICAL-1 (missing record_stream)
- #46203: ROCm cumem same bug (confirms platform-universal pattern)
- #44395: wake_up(weights) + forward -> illegal memory (same missing sync)

**Source references**:
- `vllm/device_allocator/cumem.py:202`: libcudart.cudaMemcpy before unmap without synchronize
- `vllm/v1/worker/gpu_worker.py:165-200`: sleep/wake implementation
- SGLang `release_memory_occupation()`: HAS synchronize() in sleep path but MISSING in wake path

---

### 9. #46204 — MiniMax MSA P/D Disaggregation Bug

**Status**: OPEN. MiniMax-M3 (MSA) + NixlConnector layout mismatch.

**Root cause**: NixlConnector forces HND (Head-Num-block-Dim) KV layout for all non-MLA models. MiniMax-M3 is GQA (4 KV heads, native NHD layout). HND permutation (stride order 0,1,3,2,4) swaps block_size and num_kv_heads axes. The MSA SM100 fmha path expects NHD-native memory ordering -> crash.

**Triggering conditions**: heads_per_rank > 1 (TP=1 or TP=2). At TP>=4, head axis is size-1 and HND=NHD no-op.

**RTX 4090 impact**: LOW. P/D disaggregation requires multiple GPUs; RTX 4090 single GPU not viable for this scenario. However, the pattern (KV layout mismatch in disaggregation) connects to the State Lifecycle Mismatch family.

**Source references**:
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:141-157`: get_required_kvcache_layout() returns "HND" for all non-MLA
- `vllm/models/minimax_m3/common/sparse_attention.py:97-113`: native NHD shape

---

### 10. #46203 — ROCm Cumem Sleep Fix

**Status**: OPEN. Same root cause as #45552, platform-universal confirmation.

**Two fixes in one PR**:
1. ROCm sleep mode: `hipMemUnmap`/`hipMemRelease` did not release physical VRAM while virtual address remained reserved -> sleep didn't actually free GPU memory
2. Interpreter teardown crash: kept-alive cumem MemPool objects destroyed after pluggable allocator wrappers gone -> crash during shutdown

**Significance**: Confirms #45552 bug is NOT NVIDIA-specific. Same pattern exists across GPU vendors. This is a systematic design gap, not a platform-specific bug.

**RTX 4090 impact**: None directly (NVIDIA platform), but validates the pattern classification.

---

### 11. #46195 — PP Broadcast Hang on Device Error

**Status**: OPEN. Pipeline parallel broadcast hangs permanently when peer GPU experiences device-level error (e.g., UR_RESULT_ERROR_DEVICE_LOST).

**Root cause**: `torch.distributed.broadcast()` is blocking collective. If peer GPU crashes, stuck worker process stays alive (existing worker-monitor only detects process death, not GPU hang). Other workers block on shared-memory broadcast ring buffer waiting for RPC response. Error surfaces only after 300-second RPC timeout.

**Fix approach**:
- `_broadcast_with_timeout()`: wrapper running broadcast in daemon thread with configurable timeout (default 180s via VLLM_PP_BROADCAST_TIMEOUT_SECONDS)
- `multiproc_executor.py`: catch TimeoutError, check is_failed before re-raising
- Diagnostic hint in shm_broadcast.py warning logs

**RTX 4090 impact**: LOW for single GPU (no PP). HIGH for multi-GPU PP setups. Pattern connection: this is the 8th member of State Lifecycle Mismatch family -- collective ordering deadlock at weight-sync boundary.

---

### 12. #45979 — DSV4 Sparse Cache Revert (CLOSED Without Merge, False Alarm)

**Status**: CLOSED without merging. The sparse cache (#45863) was VINDICATED.

**Key finding**: #45863 sparse cache is an INTRA-STEP optimization (per-request, per-forward-pass). It is NOT an inter-step cache. Metadata is created fresh each forward pass, so cache is never stale. The GSM8K 6.75% regression was caused by #45309 (cudagraph), NOT by #45863 (sparse cache).

**Critical distinction**:
| Cache type | Scope | Staleness risk | Safety |
|------------|-------|---------------|--------|
| CUDA graph replay | Inter-step (captured once, replayed many) | HIGH | DANGEROUS |
| Persistent Python dict | Inter-step (survives across forward passes) | HIGH | DANGEROUS |
| Sparse index cache (#45863) | Intra-step (per-request, per-pass) | NONE | SAFE |

**Universal rule refined**:
- Rule 1: Per-STEP dynamic data MUST NOT be cached across steps (CUDA graph, persistent dict)
- Rule 2: Per-REQUEST data CAN be cached across layers within same forward pass (sparse cache)
- Rule 3: @eager_break_during_capture is a CORRECTNESS boundary, not a performance limitation

**RTX 4090 impact**: The sparse cache (2-4% TTFT improvement) REMAINS on main. For DSV2-Lite 16B: enforce_eager=True + sparse_cache=True = optimal config.

---

### 13. #46118 — MTP Grammar FSM Conflict (58% Request Failure)

**Status**: OPEN. Fix PR #44297 OPEN (+455/-15, 0/50 failure after fix vs 29/50 baseline).

**4 interlocking defects**:

1. **Mid-window reasoning-end loses bitmask switch**: FSM uses start-of-step `should_fill_bitmask` for ENTIRE window. Post-marker positions get unconstrained mask -> model emits grammar-invalid tokens -> FSM rejects -> request terminated.

2. **Bonus row inherits stale apply_bitmask from -1 padding**: `-1` padding in async spec-decode path flips `apply_bitmask=False`, contaminating the bonus-token slot.

3. **Mid-window grammar advance against invalid drafts raises AssertionError**: Strict-start grammars (response_format=json_object) reject post-marker drafts -> assertion fires.

4. **Post-#42452 scheduler feeds reasoning content into grammar**: scheduler advances grammar with tokens including reasoning-end marker (248069 = `</think>`). Grammar excludes this marker -> accept_tokens rejects -> request dies.

**Fix pattern**: Per-position reasoning-end detection. For each draft token position, call `reasoner.is_reasoning_end_streaming()` on running prefix. Once detected, flip `apply_bitmask=True` for all remaining positions. Bonus row: `should_fill_bitmask(request) or apply_bitmask` (can't be flipped by -1). Tolerate accept_tokens rejection for post-marker drafts.

**RTX 4090 impact**: HIGH. Reproduces on RTX 4090 hardware. MTP is RTX 4090's primary throughput mechanism (2-3x speedup). 58% failure rate = production blocker for GRPO configs using MTP + structured output.

**Current RTX 4090 decision matrix**:
| Config | MTP | Structured Output | kv-cache-dtype | Status |
|--------|-----|-------------------|-----------------|--------|
| A (safe baseline) | OFF | OFF | fp8 | WORKS |
| B (MTP speedup) | ON | OFF | fp8 | WORKS (~2-3x) |
| C (structured output) | OFF | ON | fp8 | WORKS |
| D (full stack) | ON | ON | fp8 | BROKEN (#46118) |

**Recommended until #44297 merges**: Config B (MTP ON, structured output OFF, fp8 KV).

**Source references**:
- `vllm/v1/structured_output/__init__.py`: StructuredOutputManager.grammar_bitmask() (4 defects)
- `vllm/v1/core/sched/scheduler.py`: update_from_output() (Defect 4)
- `vllm/v1/structured_output/backend_xgrammar.py:158/162`: "Failed to advance FSM" error surface
- `vllm/v1/structured_output/request.py`: new reasoning_end_token_index field

---

## Part III: RTX 4090 GRPO Impact Assessment

### 14. Memory Implications

**RTX 4090 (24 GiB) memory partitioning for GRPO HYBRID mode**:

| Component | sleep_level=1 (LoRA) | sleep_level=2 (merge) |
|-----------|---------------------|----------------------|
| Base model weights (7B bf16) | ~14 GiB (kept resident!) | DISCARDED during sleep |
| LoRA adapter (rank=32) | ~4 MiB (offloaded during training) | N/A |
| KV cache (rollout) | ~4-6 GiB | ~4-6 GiB |
| Optimizer (CPU_Adam) | CPU resident (~3.8 GiB) | CPU resident |
| Training activations | ~2-3 GiB | ~2-3 GiB |
| Gradient buffers | ~2 GiB | ~2 GiB |
| **Total GPU (rollout)** | **~18-20 GiB** | **~18-20 GiB** |
| **Total GPU (training)** | **~16-18 GiB** (base weights + grads + act) | **~8 GiB** (no weights) |

**Critical insight**: sleep_level=1 avoids CuMemAllocator entirely (LoRA offload = tensor.to(cpu), not cuMem unmap). This means #45552 crash CANNOT occur with sleep_level=1 on RTX 4090.

**sleep_level=2 risks on RTX 4090**:
- CuMemAllocator crash (#45552): without patch, crashes within 1-3 steps
- Weight re-transfer overhead: ~16 GiB per step vs ~200 MiB (LoRA deltas) -> 80x more network/disk I/O
- Memory fragmentation: repeated allocate/free of 14 GiB weight blocks -> PyTorch allocator fragmentation
- Buffer restoration: RoPE tables, norm stats must be correctly restored after S2 wake

### 15. Weight Sync Implications

**verl weight sync mechanisms**:
- HYBRID mode: same-process, ZMQ handles for weight tensors
- BucketedWeightSender: splits weights into chunks (bucket_size_mb configurable)
- CUDA IPC (inter-process): GPU-to-GPU direct copy when available
- Shared memory fallback: when IPC not supported

**vLLM weight update path**:
```python
llm.sleep(level=2)
llm.wake_up(tags=["weights"])       # Reallocate
llm.collective_rpc("update_weights_from_ipc")  # ZMQ weight transfer
llm.wake_up(tags=["kv_cache"])      # Reallocate KV
```

**LoRA adapter path (sleep_level=1)**:
- Base weights STAY resident on GPU -> no transfer needed
- Only LoRA deltas: ~200 MiB per step vs ~16 GiB -> 80x reduction
- Adapter lifecycle: unload old LoRA, load new LoRA by tensor (serialized + MultiprocessingSerializer)
- LoRA name = constant (one adapter at a time) -> old must be unloaded before new loaded

**Delta weight sync (#6794)**: bytewise diff encoding. At typical RL learning rates, >99% of bf16 bytes unchanged step-over-step. Delta payload ~100x smaller than full transfer. BUT:
- Currently SGLang-only (requires SGLang #26519 receiver side)
- Two CRITICAL review issues: missing record_stream + OOM from big_values
- Deferred for LoRA adapter path currently (LoRA deltas already small)

**RTX 4090 optimal path**: sleep_level=1 + LoRA adapter + verl HYBRID. No CuMemAllocator involvement, no weight re-transfer, minimal overhead per step.

### 16. Throughput Implications

**Speculative decoding on RTX 4090**:
- MTP (Multi-Token Prediction): ~2-3x throughput increase
- EAGLE: similar speedup but requires separate draft model head
- DFlash (#46105): diffusion-based drafting, future paradigm
- Current blocker: MTP + structured output = 58% failure (#46118). Until #44297 merges, MTP can only be used WITHOUT structured output.

**Prefix caching on RTX 4090**:
- GRPO rollout_n=8: same adapter -> prefix shareable -> 7x prefill savings (per SGLang measurement)
- vLLM prefix hash does NOT include LoRA ID -> safe for GRPO (all copies same adapter)
- Preemption = full reset (no radix tree preservation like SGLang) -> more wasteful
- BudgetRefiner SLO: NO RTX 4090 profile data -> contribution opportunity

**CUDA graph on RTX 4090**:
- SM89 (Ada Lovelace): supports FA2 UNIFORM_BATCH -> FULL cudagraph possible for uniform decode
- DSV4-family models: enforce_eager=True MANDATORY -> ~30% throughput loss
- BudgetRefiner can compensate: throttles prefill when decode busy -> protects decode latency

**Attention backend selection on RTX 4090**:
- FA2 (SM80+): available, UNIFORM_BATCH for cudagraph
- FlashInfer: general serving backend, SINGLE_TOKEN mode for decode cudagraph
- Triton: fallback, works but slower
- FlashMLA: NOT available on SM89 (requires SM90)
- CUTLASS MLA: NOT available on SM89 (requires SM100)

---

## Part IV: Cross-Framework Connections

### 17. vLLM vs SGLang Sleep/Wake Comparison

| Feature | SGLang | vLLM |
|---------|--------|------|
| Sleep mechanism | `tokenizer_manager.release_memory_occupation()` | `engine.sleep(level=sleep_level)` |
| Wake mechanism | `tokenizer_manager.resume_memory_occupation()` | `engine.wake_up()` |
| Sleep path safety | HAS `torch.cuda.synchronize()` | MISSING (bug #45552) |
| Wake path safety | MISSING `torch.cuda.synchronize()` | MISSING (same bug) |
| Overall safety | Half-safe (sleep OK, wake unsafe) | Fully unsafe (both paths unsafe) |
| Tags granularity | `["kv_cache"]`, `["weights"]`, combined | `level=1` or `level=2` (integer) |
| LoRA as adapter | `lora_as_adapter` property -> `tags=["kv_cache"]` | sleep_level=1 conditional |
| Weight update | HTTP-based + LoRA load_lora_adapter_from_tensor | ZMQ IPC update_weights |
| Prefix caching | Radix tree (preserves prefix on eviction) | BlockPool hash (full reset on eviction) |
| Memory elasticity | Radix tree -> prefix reuse after eviction | Preemption = full recomputation |

**Key difference**: SGLang's radix tree preserves prefix blocks on eviction -> preempted request can reuse prefix. vLLM's preemption = full reset -> must recompute everything. This is a throughput disadvantage for vLLM on memory-constrained GPUs like RTX 4090.

**Both frameworks need `torch.cuda.synchronize()` in wake path**: SGLang's `resume_memory_occupation()` submits H2D copies asynchronously and returns immediately without synchronize -> same race condition as vLLM #45552 wake path.

---

### 18. vLLM vs verl Weight Sync Integration

**verl's 3 RolloutModes**:
| Mode | Process Layout | Weight Sync | GPU Sharing |
|------|---------------|-------------|-------------|
| HYBRID | Same process | sleep->wake->update | Same GPU, same process |
| COLOCATED | Separate process, same PG | Not required (weights stay) | Same GPU, diff process |
| STANDALONE | Separate GPU | Full transfer | Different GPU |

**HYBRID mode flow** (the RTX 4090 optimal path):
```
1. wake_up(tags=["weights"]) -> reallocate weight space
2. get_per_tensor_param() -> summon from FSDP
3. if LoRA adapter (merge=False):
   a. sleep_level = 1 (only release KV later)
   b. first time: base_sync (full weights)
   c. subsequent: adapter sync (LoRA deltas only, ~200 MiB)
4. aggressive_empty_cache() -> reclaim GPU memory
5. wake_up(tags=["kv_cache"]) -> allocate KV space
6. set_expandable_segments(True) -> dynamic batch sizing
```

**Buffer vs parameter updates** (verl weight_update_utils.py):
- Parameters: linear weights, biases -> standard update
- Buffers: RoPE cos/sin, norm running stats -> must be COPIED in-place
- These buffers are the SAME ones preserved during S2 sleep in vLLM
- If buffer restoration fails -> garbage output -> same pattern as DSV4 instability

---

### 19. CUDA Stream Safety Pattern Family

**Unified root cause**: GPU operations are asynchronous, but framework code assumes they are synchronous.

**Pattern instances** (11+ across 7 frameworks):

| Instance | Framework | Root Cause | Fix Pattern |
|----------|-----------|------------|-------------|
| vLLM #45552 | vLLM | Missing cuda.synchronize in sleep/wake | Add synchronize() |
| vLLM #46203 | vLLM (ROCm) | Same bug, AMD platform | Same 2-line fix |
| vLLM #44395 | vLLM | wake + forward race | synchronize after wake |
| vLLM #46195 | vLLM | PP broadcast hang on device error | broadcast_with_timeout |
| vLLM #46118 | vLLM | MTP + grammar FSM phase boundary | Per-position state evaluation |
| vLLM #46125 | vLLM | Stale encoder cache after weight revert | Invalidate cache at boundary |
| SGLang #28676 | SGLang | MoE cache clobbered on weight reload | dict.clear() at boundary |
| SGLang #28679 | SGLang | GDN intermittent degeneracy | Ring cursor reset at boundary |
| vLLM-Ascend #10684 | vLLM-Ascend | DSA Hadamard ALL-ZERO after sleep/wake | Preserve class variables |
| DeepSpeed #8061 | DeepSpeed | overlap_comm multi-stream race | record_stream + wait_stream |
| DeepSpeed #8080 | DeepSpeed | Fix for #8061 (maintainer-authored) | Multi-stream synchronization |

**Three sub-categories**:
1. **Stream synchronization**: Missing cuda.synchronize() or record_stream -> data race (#45552, #46203, #44395, #8061)
2. **Cache invalidation**: Stale cache after weight change -> silent corruption (#46125, #28676, #28679, #10684)
3. **Phase boundary**: State mismatch when execution crosses phase boundary (#46118, #45309, #46195)

**Universal fix principle**: ALWAYS add explicit synchronization barriers at state transition points. Never assume async GPU operations have completed without explicit verification. Per-position state evaluation (not per-step) when multi-position scheduling is used.

**4-Layer Defense Stack**:
| Layer | Defense | What to Check |
|-------|---------|---------------|
| 1. Framework Safety | record_stream + cuda.synchronize | ALL multi-stream code paths |
| 2. Cache Invalidation | Reset prefix/encoder cache | ALL caches after weight-reload |
| 3. Phase Boundary | Per-position state evaluation | ALL multi-position scheduling |
| 4. Dynamic Data | Per-step data not cached across steps | DSV4/MoE routing, sparse metadata |

---

## Part V: Source Code File Reference Index

### Architecture Source Files

| File | Lines | Purpose | Key Line References |
|------|-------|---------|-------------------|
| `vllm/v1/engine/core.py` | ~1600 | EngineCore + EngineCoreProc | step():443-472, run_busy_loop():1216-1224, process_input_sockets():1423-1518, process_output_sockets():1520-1585, _handle_client_request():1317-1346 |
| `vllm/v1/core/sched/scheduler.py` | ~2375 | Core scheduler | __init__:66-288, schedule():339-952, Running phase:376-550, Waiting phase:562-853, Preempt:458-508, _preempt_request():959-979, update_from_output():1310-1630, reset_prefix_cache():1923-1971, BudgetRefiner point:407 |
| `vllm/v1/core/sched/async_scheduler.py` | ~68 | Async scheduler extension | _update_after_schedule():19-41, _update_request_with_output():43-67 |
| `vllm/v1/core/sched/interface.py` | ~245 | SchedulerInterface ABC | PauseState:22-33 |
| `vllm/v1/core/sched/request_queue.py` | ~209 | FCFS/Priority queues | FCFSRequestQueue:75-129, PriorityRequestQueue:131-198 |
| `vllm/v1/worker/gpu_model_runner.py` | ~7400 | GPUModelRunner + 3 Mixins | __init__:420-893, execute_model():4002-4363, _prepare_inputs():1868-2187, _build_attention_metadata():2188-2491, sample_tokens():4381-4640, Persistent buffers:714-760, ExecuteModelState:401-414, CudagraphDispatcher init:813, _determine_batch_execution_and_padding():3768-3880 |
| `vllm/v1/worker/gpu_input_batch.py` | - | InputBatch + CachedRequestState | CachedRequestState:34-89, InputBatch:91-, add_request():335-481 |
| `vllm/v1/worker/gpu_worker.py` | ~600 | GPU worker sleep/wake | sleep():165-200, wake_up():165-200, _sleep_saved_buffers preservation |
| `vllm/v1/core/kv_cache_manager.py` | 614 | KVCacheManager | allocate_slots() 5-stage layout |
| `vllm/v1/core/block_pool.py` | 527 | BlockPool | BlockHashToBlockMap:34-128, cache_full_blocks():211-331, get_new_blocks/free_blocks/touch operations |
| `vllm/v1/core/kv_cache_coordinator.py` | ~250 | HybridKVCacheCoordinator | Fixed-point convergence algorithm |
| `vllm/v1/core/single_type_kv_cache_manager.py` | ~400 | Per-attention-type managers | FullAttention, SlidingWindow, MLA, Mamba |
| `vllm/v1/core/kv_cache_utils.py` | ~400 | KVCacheBlock + FreeKVCacheBlockQueue | KVCacheBlock:165-400, hash_block_tokens:541-568 |
| `vllm/device_allocator/cumem.py` | ~300 | CuMemAllocator | sleep():202 (missing sync), wake_up() (missing sync) |
| `vllm/v1/sample/sampler.py` | ~144 | Sampler 9-step pipeline | forward():67-144 |
| `vllm/v1/structured_output/__init__.py` | - | StructuredOutputManager | grammar_bitmask() (4 defects for #46118) |
| `vllm/v1/cudagraph_dispatcher.py` | - | CudagraphDispatcher | LoRA specialization:115-134 |

### LoRA Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/lora/request.py` | 73 | LoRARequest (msgspec.Struct) |
| `vllm/lora/layers/base_linear.py` | 149 | GPU buffer allocation + set_lora/reset_lora |
| `vllm/lora/ops/triton_ops/` | - | lora_shrink/lora_expand/fused_moe_lora kernels |
| `vllm/lora/lora_kernel_metadata.py` | - | LoRAKernelMeta (per-token metadata) |
| `vllm/lora/punica_wrapper/` | - | PunicaWrapper + convert_mapping |
| `vllm/v1/worker/lora_model_runner_mixin.py` | 131 | LoRAModelRunnerMixin, warmup, adapter management |

### Bug-Specific Source References

| Bug | Key Files | Lines |
|-----|-----------|-------|
| #46125 (encoder cache) | `vllm/v1/engine/async_llm.py`, `vllm/v1/core/sched/scheduler.py` | finish_weight_update(), reset_prefix_cache():1923-1971 |
| #45552 (cumem sync) | `vllm/device_allocator/cumem.py`, `vllm/v1/worker/gpu_worker.py` | sleep() cumem.py:202, wake_up() gpu_worker:165-200 |
| #46118 (MTP+grammar) | `vllm/v1/structured_output/__init__.py`, `vllm/v1/core/sched/scheduler.py` | grammar_bitmask(), update_from_output() |
| #46204 (MSA P/D) | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` | get_required_kvcache_layout():141-157 |
| #45979 (sparse cache) | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | flashinfer_sparse_index_cache logic |

---

## Part VI: RTX 4090 GRPO Deployment Rules (Updated)

### MUST DO

| # | Rule | Setting | Reason | Bug |
|---|------|---------|--------|-----|
| 1 | sleep_level=1 | LoRA adapter path | Best memory + avoids cumem | #45552 |
| 2 | overlap_comm=False | DeepSpeed config | NaN on single GPU | #8061 |
| 3 | zero_single_gpu_optim=True | DeepSpeed config | Avoid unnecessary ops | - |
| 4 | weight_mode="full" | verl config | Delta sync overhead in HYBRID | #6794 |
| 5 | torch.compile cautiously | PyTorch | Stream safety concerns | #8061 |
| 6 | CPU_Adam optimizer | DeepSpeed/verl | Reduce GPU memory | - |
| 7 | ZeRO2 only (not ZeRO3) | DeepSpeed | ZeRO3 PEFT regression | #8072/#8073 |
| 8 | Patch sleep/wake sync | vLLM/SGLang | CRITICAL: crashes without this | #45552 |
| 9 | LoRA rank=16-32 | verl config | Balance quality/memory | #6782 |
| 10 | gradient_clipping=1.0 | DeepSpeed | Muon clipping gap workaround | #8068 |
| 11 | reset_prefix/encoder cache | vLLM | After EVERY weight update | #46125 |
| 12 | MTP OFF + structured_output OFF | vLLM | Until #44297 merges | #46118 |
| 13 | kv-cache-dtype=fp8 (not auto) | vLLM | Avoids #46088 cross-sequence garbage | #46088 |
| 14 | enforce_eager=True | vLLM | For DSV4-family models | #45309 |

### MUST NOT

| # | Rule | Reason | Bug |
|---|------|--------|-----|
| 1 | Use sleep_level=2 without #45552 patch | Crashes within 1-3 steps | #45552 |
| 2 | Use overlap_comm=True on single GPU | NaN guaranteed | #8061 |
| 3 | Use ZeRO3 with PEFT | Regression confirmed on 2 platforms | #8072/#8073 |
| 4 | Use LoRA rank=64 | Breaks EOS token | #6782 |
| 5 | Use MTP + structured_output simultaneously | 58% request failure | #46118 |
| 6 | Use --kv-cache-dtype auto with MTP | Cross-sequence garbage | #46088 |
| 7 | Use CUDA graph with DSV4-family models | Stale dynamic routing data | #45309 |
| 8 | Assume prefix cache valid after weight update | Silent corruption | #46125 |

---

## References

### GitHub Issues/PRs

- vLLM #46125: https://github.com/vllm-project/vllm/pull/46125 (OPEN -- REVERT of stale cache fix)
- vLLM #45093: https://github.com/vllm-project/vllm/pull/45093 (MERGED -- cache reset after weight update)
- vLLM #45552: https://github.com/vllm-project/vllm/pull/45552 (OPEN -- cumem stream sync fix)
- vLLM #46204: https://github.com/vllm-project/vllm/issues/46204 (OPEN -- MiniMax MSA P/D bug)
- vLLM #46203: https://github.com/vllm-project/vllm/pull/46203 (OPEN -- ROCm cumem fix)
- vLLM #46195: https://github.com/vllm-project/vllm/issues/46195 (OPEN -- PP broadcast hang)
- vLLM #45979: https://github.com/vllm-project/vllm/pull/45979 (CLOSED -- sparse cache false alarm)
- vLLM #45863: https://github.com/vllm-project/vllm/pull/45863 (MERGED -- sparse cache, VINDICATED)
- vLLM #45972: https://github.com/vllm-project/vllm/pull/45972 (MERGED -- DSV4 cudagraph revert)
- vLLM #46118: https://github.com/vllm-project/vllm/issues/46118 (OPEN -- MTP+grammar conflict)
- vLLM #44297: https://github.com/vllm-project/vllm/pull/44297 (OPEN -- MTP grammar fix, 0/50 failure)
- vLLM #44395: https://github.com/vllm-project/vllm/issues/44395 (OPEN -- wake+forward illegal memory)
- vLLM-Ascend #10684: https://github.com/vllm-project/vllm-ascend/issues/10684 (DSA Hadamard sleep/wake)
- SGLang #28676: https://github.com/sgl-project/sglang/issues/28676 (MoE cache clobber)
- SGLang #28771: https://github.com/sgl-project/sglang/issues/28771 (EAGLE accept_length degradation)
- DeepSpeed #8061: https://github.com/deepspeedai/DeepSpeed/issues/8061 (overlap_comm NaN)
- DeepSpeed #8080: https://github.com/deepspeedai/DeepSpeed/pull/8080 (stream race fix)
- verl #6794: https://github.com/verl-project/verl/pull/6794 (delta weight sync)

### Local Reading Notes

- `vllm-v1-architecture-map.md` -- complete V1 architecture map
- `vllm-v1-engine-core-deep-reading.md` -- EngineCore step() loop, ZMQ IPC
- `vllm-v1-scheduler-deep-reading.md` -- Scheduler 613-line schedule() method
- `vllm-v1-gpu-model-runner-reading.md` -- GPUModelRunner two-phase execution
- `vllm-v1-kv-cache-management-reading.md` -- BlockPool, prefix caching
- `vllm-v1-memory-management-architecture-reading.md` -- CuMemAllocator, sleep/wake
- `vllm-lora-serving-reading.md` -- Punica SGMV, fixed-address buffers
- `vllm-45552-cumem-stream-sync-deep-reading.md` -- cumem bug with "200 lie" pattern
- `vllm-46125-stale-encoder-cache-revert-reading.md` -- #46125 danger analysis
- `vllm-45979-dsv4-sparse-cache-revert-reading.md` -- #45979 false alarm vindication
- `vllm-46118-mtp-grammar-fsm-conflict-reading.md` -- 4 interlocking defects
- `verl-hybrid-sleep-wake-architecture-reading.md` -- verl sleep/wake comparison
- `verl-6794-delta-weight-sync-deep-reading.md` -- delta weight sync architecture
- `cuda-stream-memory-management-reading.md` -- CUDA stream safety patterns

*Report created 2026-06-20. Comprehensive deep reading of vLLM V1 architecture, 7 critical bugs, RTX 4090 GRPO impact, and cross-framework connections.*
