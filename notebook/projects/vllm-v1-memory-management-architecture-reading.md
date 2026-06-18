# vLLM V1 Memory Management Architecture — Deep Reading

> 2026-06-18 | vllm-project/vllm (main branch, v0.23.0 era)
> ★★★★★★★★ V1 KV cache = BlockPool + KVCacheCoordinator + SingleTypeKVCacheManager → radical simplification over V0
> ★★★★★★★★ Sleep/Wake = CuMemAllocator (CUDA virtual memory) + buffer preservation → CRITICAL for verl HYBRID
> ★★★★★★★★ DSV4 breaks because dynamic routing data CANNOT survive sleep/wake cycles or CUDA graph replay

---

## 1. V1 KV Cache Architecture — BlockPool Centric

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ V1 eliminates BlockSpaceManager entirely → single BlockPool + KVCacheCoordinator
★★★★★★★★★ BlockPool = doubly-linked free list + hash-based prefix cache → all operations O(1)
★★★★★★★★★ KVCacheCoordinator orchestrates multiple KV cache groups (full-attn, SWA, MLA, Mamba)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 1.1 Core Class Hierarchy

```
KVCacheManager (kv_cache_manager.py, 614 lines)
  ├── coordinator: KVCacheCoordinator (kv_cache_coordinator.py)
  │     ├── block_pool: BlockPool (block_pool.py, 527 lines)
  │     │     ├── free_block_queue: FreeKVCacheBlockQueue (doubly-linked list)
  │     │     ├── cached_block_hash_to_block: BlockHashToBlockMap
  │     │     └── blocks: list[KVCacheBlock] (all physical blocks)
  │     │
  │     └── single_type_managers: list[SingleTypeKVCacheManager]
  │           ├── FullAttentionManager
  │           ├── SlidingWindowManager
  │           ├── MLAAttentionManager
  │           ├── MambaManager
  │           ├── ChunkedLocalAttentionManager
  │           ├── CrossAttentionManager
  │           └── SinkFullAttentionManager
  │
  ├── watermark_blocks: int (preemption headroom)
  ├── empty_kv_cache_blocks: KVCacheBlocks (pre-allocated empty result)
  └── prefix_cache_stats: PrefixCacheStats
```

### 1.2 BlockPool — The Physical Memory Manager

**Source**: `vllm/v1/core/block_pool.py` (527 lines)

```python
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks."""

    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.null_block = self.free_block_queue.popleft()  # placeholder for unused slots
```

**Key operations**:
| Method | Purpose | Complexity |
|--------|---------|-----------|
| `get_new_blocks(n)` | Allocate n blocks from free list front | O(n) linked-list pop |
| `free_block(block)` | Return block to free list tail (or evict from hash cache) | O(1) |
| `cache_full_blocks()` | Hash full blocks → insert into prefix cache map | O(block_count) |
| `get_cached_block(hash, group_ids)` | Lookup prefix cache hit by block hash | O(num_groups) |
| `evict_blocks(block_ids)` | Force-evict specific blocks from hash cache | O(block_ids) |
| `reset_prefix_cache()` | Clear entire prefix cache (used in RLHF weight update!) | O(cache_size) |
| `get_num_free_blocks()` | Return free_block_queue.num_free_blocks | O(1) |
| `get_usage()` | Return fraction of blocks in use | O(1) |

### 1.3 FreeKVCacheBlockQueue — Doubly-Linked Free List

**Source**: `vllm/v1/core/kv_cache_utils.py` (lines 165-400)

```
★★★★★★★★★ LRU eviction order: least recently used block at the FRONT of the queue
★★★★★★★★★ When blocks are freed, they go to the TAIL → natural LRU ordering
★★★★★★★★★ Doubly-linked list with fake head/tail → no branching in pop/append → fast!
★★★★★★★★★ Uses block.prev_free_block / block.next_free_block → no extra Python objects!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

**Eviction order rules**:
1. Least recently used block at the front (LRU)
2. If same allocation time, tail of block chain first (reverse order freed)
3. This ordering is maintained by reversing blocks during `free()` operations

**Methods**:
- `popleft()` → pop first free block (allocation)
- `popleft_n(n)` → pop n blocks at once (batch allocation)
- `append(block)` → add block to tail (freeing)
- `append_n(blocks)` → add n blocks to tail (batch freeing)

### 1.4 KVCacheBlock — Physical Block Metadata

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int               # 0 to num_gpu_blocks-1 → index into GPU tensor
    ref_cnt: int = 0            # reference count for prefix sharing
    _block_hash: BlockHashWithGroupId | None = None  # hash for prefix caching (only when full)
    prev_free_block: KVCacheBlock | None = None      # doubly-linked list pointer
    next_free_block: KVCacheBlock | None = None      # doubly-linked list pointer
    is_null: bool = False       # placeholder block (never cached)
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ KVCacheBlock is METADATA only — no data stored in the Python object!
★★★★★★★★★ Actual KV data lives in GPU tensor indexed by block_id
★★★★★★★★★ block_hash uses sha256_cbor → BlockHashWithGroupId = hash_bytes + group_id (4 bytes)
★★★★★★★★★ ref_cnt tracks prefix sharing — block freed only when ref_cnt drops to 0
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 1.5 BlockHashToBlockMap — Prefix Cache Lookup

**Novel design**: Instead of `dict[hash → single_block]`, uses `dict[hash → KVCacheBlock | dict[block_id → KVCacheBlock]]`.

```
★★★★★★★★★ WHY dict variant? → Multiple blocks can have the same hash (duplicate tokens)!
★★★★★★★★★ Single block: self._cache[key] = KVCacheBlock (most common case → no inner dict)
★★★★★★★★★ Multiple blocks: self._cache[key] = {block_id: KVCacheBlock} (rare → avoids GC overhead)
★★★★★★★★★ This optimization avoids creating inner dicts for 99%+ of blocks → reduces GC pressure!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 2. KVCacheManager — Allocation Flow

### 2.1 Scheduler → KVCacheManager Interaction

```
Scheduler.schedule() (scheduler.py, 2618 lines)
  ├── Phase 1: RUNNING requests → allocate_slots for decode tokens
  ├── Phase 2: WAITING → RUNNING transition → allocate for prefill
  ├── Phase 3: Preemption when can_allocate_slot returns None
  │
  ├── Token budget management:
  │     token_budget = self.max_num_scheduled_tokens
  │     Each RUNNING request consumes: min(num_new_tokens, token_budget)
  │     token_budget decrements per request → global budget constraint!
  │
  └── KV cache budget (BlockPool level):
        available_blocks = free_blocks - watermark_blocks - reserved_blocks
        If required_blocks > available_blocks → return None → preempt!
```

### 2.2 allocate_slots — The Core Allocation Method

**Source**: `kv_cache_manager.py` lines ~200-400

```
★★★★★★★★★ allocate_slots has THREE stages:
★★★★★★★★★ Stage 1: Free unnecessary blocks (outside sliding window) → check free block count
★★★★★★★★★ Stage 2: Handle prefix tokens (computed + new cache hits + external)
★★★★★★★★★ Stage 3: Allocate new blocks for tokens to be computed (new + lookahead)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

**Block layout** (from source code comments):
```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                                  |   < to be computed >     |
----------------------------------------------------------------------
                                  |            < to be allocated >           |
----------------------------------------------------------------------
                                  | < to be cached (roughly)>               |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM or connector. Can be safely
  removed if outside sliding window.
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by vLLM, but cached by connector |
| ref_cnt increased       | ref_cnt not increased yet                   |
----------------------------------------------------------------------
```

**Watermark mechanism**:
```python
watermark_blocks = 0
if has_scheduled_reqs and request.status in (WAITING, PREEMPTED):
    watermark_blocks = self.watermark_blocks  # prevents over-admission
```

```
★★★★★★★★★ Watermark ONLY applies to WAITING/PREEMPTED requests → protects RUNNING requests!
★★★★★★★★★ RUNNING requests don't get watermark → they already have blocks allocated
★★★★★★★★★ This prevents "admission starvation" where new requests steal blocks from running ones
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 2.3 full_sequence_must_fit — Admission Gate

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > block_pool.get_num_free_blocks():
        return None  # Cannot admit → reject
```

```
★★★★★★★★★ full_sequence_must_fit prevents over-admission in chunked prefill!
★★★★★★★★★ Without this gate, chunked prefill would only check first chunk's blocks
★★★★★★★★★ → request admitted but can't finish → preempted mid-prefill → thrashing!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 2.4 Preemption Strategy — Recompute First

```
★★★★★★★★★ V1 prefers RECOMPUTE over swap (opposite of V0!)
★★★★★★★★★ Why? For short prefixes, recomputation is faster than CPU→GPU transfer
★★★★★★★★★ Also avoids maintaining CPU block tables → simpler implementation
★★★★★★★★★ Preemption selection: max(num_computed_tokens) → preempt the request with most tokens
★★★★★★★★★ (Because it has the most blocks → freeing it releases the most memory)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 3. Sleep/Wake Architecture — GPU Memory Lifecycle

### 3.1 Sleep Levels

**Source**: `docs/features/sleep_mode.md` + `gpu_worker.py`

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Level 1 (S1): Offload weights to CPU RAM, discard KV cache
★★★★★★★★★   → Good for: sleep and wake to run the SAME model again
★★★★★★★★★   → CPU RAM must hold full model weights!
★★★★★★★★★   → Wake-up faster: only need to reload weights + reallocate KV cache
★★★★★★★★★
★★★★★★★★★ Level 2 (S2): Discard BOTH weights AND KV cache (keep buffers in CPU)
★★★★★★★★★   → Good for: RLHF weight update (old weights not needed)
★★★★★★★★★   → CPU RAM only needs buffers (RoPE tables, etc) → MUCH less RAM needed
★★★★★★★★★   → Wake-up slower: need to reallocate weights memory + reload or update weights
★★★★★★★★★
★★★★★★★★★ verl uses S2 by default (VLLM_SLEEP_LEVEL=2 since vLLM >= 0.8.5)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 3.2 GPU Worker Sleep/Wake Implementation

**Source**: `vllm/v1/worker/gpu_worker.py` lines 165-200

```python
def sleep(self, level: int = 1) -> None:
    free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

    # S2: Save constant buffers to CPU before discarding everything
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone() for name, buffer in model.named_buffers()
        }

    # CuMemAllocator handles the actual memory release
    allocator = get_mem_allocator_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    # S1: offload_tags=("weights",) → only weights offloaded, others discarded
    # S2: offload_tags=() → ALL memory discarded (nothing offloaded to CPU)

def wake_up(self, tags: list[str] | None = None) -> None:
    allocator = get_mem_allocator_instance()
    allocator.wake_up(tags)

    # S2: Restore constant buffers from CPU backup
    if len(self._sleep_saved_buffers):
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    # Re-initialize KV cache after wake-up
    if tags is None or "kv_cache" in tags:
        self.model_runner.post_kv_cache_wake_up()
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ S2 constant buffer preservation is CRITICAL for correctness!
★★★★★★★★★ Without it: RoPE cos/sin tables, attention biases → all zeros → garbage output!
★★★★★★★★★ post_kv_cache_wake_up() zeros KV cache tensors + resets FP8 scales to 1.0
★★★★★★★★★ FP8 scales left at 0.0 → ALL KV cache values effectively zero → gibberish!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 3.3 CuMemAllocator — CUDA Virtual Memory Pool

**Source**: `vllm/device_allocator/cumem.py` (singleton class)

```python
class CuMemAllocator:
    """Singleton that manages a memory pool for CUDA tensors.
    Inside use_memory_pool(tag), all tensors created have that tag.
    sleep() offloads tagged tensors to CPU, discards the rest.
    wake_up() loads offloaded tensors back, discards the rest."""

    def sleep(self, offload_tags=("default",)):
        for ptr, data in self.pointer_to_data.items():
            if data.tag in offload_tags:
                # Copy to pinned CPU memory → preserved for wake_up
                cpu_backup = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup
            # ALL allocations: unmap_and_release(handle) → free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags=None):
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                create_and_map(handle)  # Re-map virtual address to GPU memory
                if data.cpu_backup_tensor is not None:
                    # Copy CPU backup back to GPU → restore data
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                    data.cpu_backup_tensor = None
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ CuMemAllocator uses CUDA virtual memory (cuMemMap/cuMemUnmap) — NOT regular alloc!
★★★★★★★★★ This enables: unmap releases physical backing → virtual address stays registered
★★★★★★★★★ wake_up: re-map same virtual address → tensors still reference valid pointers!
★★★★★★★★★ Key advantage: no pointer invalidation → PyTorch tensors still "work" after wake_up
★★★★★★★★★ Expandable segments are INCOMPATIBLE with this pool (PyTorch #147851)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 3.4 Wake-Up Tags — Fine-Grained RLHF Control

```python
# RLHF weight update flow (from docs/features/sleep_mode.md):
llm.sleep(level=2)                       # Discard everything
llm.wake_up(tags=["weights"])            # Reallocate weights memory ONLY (no KV cache)
llm.collective_rpc("reload_weights")     # Load new weights in-place
llm.wake_up(tags=["kv_cache"])           # Now allocate KV cache (after weights updated)
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ tags=["weights"] first → avoids OOM during weight update!
★★★★★★★★★ If KV cache allocated simultaneously → peak memory = weights + KV cache = OOM risk
★★★★★★★★★ Sequential: weights only → update → then KV cache → peak = max(weights, weights+kv)
★★★★★★★★★ But: weights > kv_cache for most models → peak ≈ weights only → safe!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 3.5 Memory Occupation Summary by Sleep Level

```
| Phase             | S1 Sleep               | S2 Sleep                    | Awake             |
|-------------------|------------------------|-----------------------------|-------------------|
| Model weights     | CPU RAM (pinned)       | DISCARDED                   | GPU VRAM          |
| KV cache          | DISCARDED              | DISCARDED                   | GPU VRAM          |
| Constant buffers  | GPU VRAM (resident)    | CPU RAM (saved_buffers)     | GPU VRAM          |
| FP8 KV scales     | DISCARDED (with KV)    | DISCARDED (with KV)         | Reset to 1.0      |
| CUDA graphs       | DISCARDED              | DISCARDED                   | Re-captured        |
| CuMem allocations  | Tagged → CPU, rest → discard | ALL discarded | Re-mapped to GPU  |
| GPU free memory   | ~90%+ freed            | ~95%+ freed                 | Fully occupied    |
```

---

## 4. verl HYBRID Mode Integration — Weight Sync Architecture

### 4.1 verl Sleep/Wake Integration

**Source**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

```python
# verl ServerAdapter wraps vLLM server for HYBRID rollout

class ServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh, ...):
        # Sleep level selection: S2 by default, S1 for layered_summon / EP
        if config.layered_summon or (config.expert_parallel_size > 1 and
                                      not _check_vllm_version_for_sleep_level()):
            self.sleep_level = 1  # S1 forced for EP > 1 or layered summon
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL  # S2 (default since vLLM 0.8.5)

    async def resume(self, tags: list[str]):
        """Wake up vLLM server (weights or kv_cache)."""
        if self.config.free_cache_engine:
            await self.server_handle.wake_up.remote(tags=tags)

    async def release(self):
        """Sleep vLLM server (free GPU memory for training)."""
        if self.config.free_cache_engine:
            await self.server_handle.sleep.remote()

    async def update_weights(self, weights, global_steps=None):
        """Update model weights via CUDA IPC (or shared memory fallback)."""
        # 1. Send update command to vLLM workers
        future = await self._execute_method("update_weights_from_ipc", non_block=True,
                                             kwargs={"use_shm": self.use_shm})
        # 2. BucketedWeightSender sends weight tensors via ZMQ
        sender = BucketedWeightSender(zmq_handle=self.zmq_handle,
                                       bucket_size_mb=bucket_size_mb,
                                       use_shm=self.use_shm)
        await sender.async_send_weights(weights)
        # 3. Wait for vLLM workers to finish processing
        await future
        # 4. Clear KV cache after weight update (CRITICAL for RLHF!)
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ verl HYBRID mode flow:
★★★★★★★★★ 1. wake_up(tags=["weights"]) → reallocate weight memory
★★★★★★★★★ 2. update_weights_from_ipc → load updated weights (ZMQ or CUDA IPC)
★★★★★★★★★ 3. clear_kv_cache → invalidate stale prefix cache (RLHF critical!)
★★★★★★★★★ 4. wake_up(tags=["kv_cache"]) → reallocate KV cache
★★★★★★★★★ 5. Generate rollouts → produce trajectories
★★★★★★★★★ 6. sleep() → free GPU memory for training
★★★★★★★★★ 7. Training step → update weights in shared memory
★★★★★★★★★ 8. Repeat from step 1
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 4.2 Weight Transfer Mechanisms

**Source**: `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`

```
★★★★★★★★★ verl uses BucketedWeightSender for weight updates:
★★★★★★★★★ → ZMQ-based transfer between actor process and vLLM server process
★★★★★★★★★ → CUDA IPC (inter-process) when available → GPU-to-GPU direct copy
★★★★★★★★★ → Shared memory (shm) fallback when IPC not supported
★★★★★★★★★ → Bucketed transfer: split weights into chunks (bucket_size_mb configurable)
★★★★★★★★★ → Avoids allocating entire model weight tensor at once → reduces peak memory
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 4.3 Buffer vs Parameter Updates

**Source**: `verl/workers/rollout/vllm_rollout/weight_update_utils.py`

```python
def split_buffer_updates(model, weights):
    """Split weight updates into parameter and buffer updates."""
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))  # RoPE tables, norms, etc
        else:
            param_updates.append((name, tensor))    # Linear weights, biases, etc
    return param_updates, buffer_updates, named_buffers
```

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Buffer updates (RoPE cos/sin, norm running stats) must be COPIED in-place
★★★★★★★★★ These are the SAME constant buffers preserved during S2 sleep!
★★★★★★★★★ If buffer restoration fails → garbage output → same pattern as DSV4!
★★★★★★★★★ DSV4 risk: MLA has MORE buffers than standard models → more restoration points
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 4.4 Sleep Level Selection Rules

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ EP > 1 → MUST use S1 (sleep_level=1) → cannot discard weights safely
★★★★★★★★★ Reason: EP splits experts across GPUs → S2 discard → expert shards lost!
★★★★★★★★★ S1 preserves weights in CPU → wake_up reloads → expert shards intact
★★★★★★★★★ layered_summon (per-unit LoRA) → MUST use S1 → summon needs full model structure
★★★★★★★★★ Default (DP=1, no EP, no layered): S2 → maximum memory savings
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 5. DSV4 Systematic Instability — Memory Management Root Cause

### 5.1 The 5 Dynamic Routing Layers in DSV4

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ DSV4 has MORE layers of dynamic routing than any previous model:
★★★★★★★★★
★★★★★★★★★ DeepSeek-V2/V3:  MoE (expert selection per token)
★★★★★★★★★ DeepSeek-V4:     MoE + DSA + MTP + Online Compress + MLA
★★★★★★★★★                   = 5 layers of per-step discrete decisions!
★★★★★★★★★
★★★★★★★★★ Each dynamic routing layer creates state that:
★★★★★★★★★ 1. Changes per step (cannot be captured in CUDA graph)
★★★★★★★★★ 2. Cannot survive sleep/wake cycles (per-step dynamic data)
★★★★★★★★★ 3. Cannot be cached in prefix cache (changes with each request)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 5.2 vLLM #45309 → #45972: eager_break Garbage Output

```
★★★★★★★★★ Root cause: Removed @eager_break_during_capture from DSV4 attention
★★★★★★★★★ → indexer + compressor captured inside CUDA graph
★★★★★★★★★ → During REPLAY: uses STATIC capture-time buffers, not live per-request data
★★★★★★★★★ → DSA indexer selects WRONG positions → garbage attention → "the the the..."
★★★★★★★★★ → Correct fix: @eager_break_during_capture separates static GEMMs from dynamic routing
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 5.3 vLLM #45979: FlashInfer Sparse Cache Accuracy Regression

```
★★★★★★★★★ Root cause: #45863 cached DSV4 flashinfer sparse index metadata
★★★★★★★★★ → Sparse index describes which KV positions are active per query
★★★★★★★★★ → This metadata changes PER STEP (dynamic routing)
★★★★★★★★★ → Caching it = using stale metadata from previous step
★★★★★★★★★ → GSM8K accuracy: 6.75% vs 87% threshold → catastrophic!
★★★★★★★★★ → Same pattern as #45309: per-step dynamic data MUST NOT be cached!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 5.4 Sleep/Wake Impact on DSV4

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ DSV4 + S2 sleep/wake creates THREE instability vectors:
★★★★★★★★★
★★★★★★★★★ Vector 1: Constant buffer restoration
★★★★★★★★★ → DSV4 MLA has MORE buffers than standard MHA (latent projection, compression)
★★★★★★★★★ → _sleep_saved_buffers must capture ALL named_buffers
★★★★★★★★★ → If ANY buffer restoration fails → MLA state corrupt → garbage output
★★★★★★★★★ → FP8 KV scales reset to 1.0 → may not match calibrated DSV4 scales!
★★★★★★★★★
★★★★★★★★★ Vector 2: Prefix cache invalidation timing
★★★★★★★★★ → After weight update, verl calls clear_kv_cache → reset_prefix_cache()
★★★★★★★★★ → But DSV4 MLA KV cache groups (MLAAttentionSpec, SlidingWindowMLASpec)
★★★★★★★★★ → Each group has separate hash cache → must ALL be reset!
★★★★★★★★★ → If ANY group missed → stale MLA attention → wrong latent projection
★★★★★★★★★
★★★★★★★★★ Vector 3: CUDA graph recapture + dynamic routing
★★★★★★★★★ → S2 sleep discards CUDA graphs → must re-capture after wake_up
★★★★★★★★★ → DSV4 requires @eager_break_during_capture for dynamic routing
★★★★★★★★★ → If re-capture misses the eager break → same garbage output as #45309!
★★★★★★★★★ → enforce_eager=True is MANDATORY for DSV4 on RTX 4090 (SM89)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 5.5 Universal Rule: Per-Step Dynamic Data Must NOT Be Cached

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ This rule applies beyond CUDA graphs — also to prefix cache and sleep/wake:
★★★★★★★★★
★★★★★★★★★ What CAN be cached (static, invariant across steps):
★★★★★★★★★ → Model weights (change only with weight updates → clear_kv_cache handles this)
★★★★★★★★★ → Prefix cache of prompt tokens (same prompt → same KV → safe to reuse)
★★★★★★★★★ → Constant buffers (RoPE tables, norm stats → preserved in _sleep_saved_buffers)
★★★★★★★★★ → Static GEMMs (weight matmuls, layer norms → same across steps)
★★★★★★★★★
★★★★★★★★★ What MUST NOT be cached (dynamic, changes per step/request):
★★★★★★★★★ → MoE expert selection (top-k varies per token batch)
★★★★★★★★★ → DSA sparse indexer positions (top-k KV varies per query)
★★★★★★★★★ → MTP draft token decisions (which tokens to draft varies)
★★★★★★★★★ → Online Compress decisions (which KV to compress varies)
★★★★★★★★★ → MLA latent projection state (changes with attention computation)
★★★★★★★★★ → FlashInfer sparse metadata (changes with each attention step)
★★★★★★★★★ → FP8 KV scales after training update (may differ from 1.0 default)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 6. BudgetRefiner SLO — Memory Budget Governance

### 6.1 How BudgetRefiner Interacts with V1 Memory Management

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ BudgetRefiner operates at the SCHEDULER level, not the BlockPool level
★★★★★★★★★ It adjusts the TOKEN BUDGET per scheduling step → controls prefill chunk size
★★★★★★★★★ The BlockPool independently manages block availability → hard constraint
★★★★★★★★★ BudgetRefiner is a SOFT constraint: "don't over-schedule prefill when decode is busy"
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 6.2 Two-Level Budget System

```
Level 1: Token Budget (Scheduler)
  ├── max_num_scheduled_tokens → global token limit per step
  ├── BudgetRefiner.refine_budget() → adjusts prefill budget based on SLO
  └── token_budget decrements per scheduled request → running total

Level 2: Block Budget (KVCacheManager)
  ├── max_num_running_reqs → max concurrent requests
  ├── BlockPool.get_num_free_blocks() → available physical blocks
  ├── watermark_blocks → headroom for preemption avoidance
  └── allocate_slots returns None if blocks insufficient → preempt
```

### 6.3 BudgetRefiner Profile Table — RTX 4090 Gap

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Current profile_table.csv = Ascend A2/B3 data ONLY!
★★★★★★★★★ RTX 4090 has NO profile data → BudgetRefiner disabled on RTX 4090!
★★★★★★★★★ This is the P10 contribution opportunity:
★★★★★★★★★ → Collect RTX 4090 profile data (unique — no other contributor has it!)
★★★★★★★★★ → Create profile_table.csv for RTX 4090 with SLO benchmarks
★★★★★★★★★ → Submit as BudgetRefiner SLO upstream contribution
★★★★★★★★★ → 58 lines of core logic → 100% GPU-generic → only CSV is HW-specific!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 7. KV Cache Group Architecture — Multi-Type Support

### 7.1 KVCacheCoordinator — Orchestrating Multiple Attention Types

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ V1 supports multiple KV cache groups per model:
★★★★★★★★★ → Standard MHA: 1 group (FullAttentionSpec)
★★★★★★★★★ → Mamba/hybrid: 2 groups (FullAttention + Mamba hidden state)
★★★★★★★★★ → DSV4 MLA: 3+ groups (MLAAttention + SlidingWindowMLA + Mamba)
★★★★★★★★★ → Encoder-decoder: 2 groups (self-attn + cross-attn)
★★★★★★★★★
★★★★★★★★★ Each group has its own SingleTypeKVCacheManager
★★★★★★★★★ → Different block_size, eviction policy, hash logic per type
★★★★★★★★★ → But ALL share the same BlockPool (unified physical memory)!
★★★★★★★★★ → KVCacheBlocks.groups = tuple of block lists per group
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 7.2 KVCacheSpec Types

| Spec | Block Management | Hash Caching | Sliding Window | DSV4 Uses? |
|------|-----------------|-------------|---------------|-----------|
| `FullAttentionSpec` | Hold all blocks until request ends | Yes (dense) | No | Partial |
| `SlidingWindowSpec` | Recycle blocks outside window | Sparse (block_mask!) | Yes | No |
| `MLAAttentionSpec` | Hold all (latent representation) | Yes (dense) | No | YES |
| `SlidingWindowMLASpec` | Recycle MLA blocks outside SWA | Sparse (block_mask!) | Yes | YES |
| `MambaSpec` | No blocks (hidden state) | No | No | YES |
| `ChunkedLocalAttentionSpec` | Recycle per chunk | Sparse | Local window | No |
| `CrossAttentionSpec` | Encoder KV, separate lifecycle | Yes (dense) | No | No |

```
★★★★★★★★★ DSV4 uses 3+ KV cache groups → BlockPool shared across all!
★★★★★★★★★ SlidingWindowMLASpec has block_mask → blocks outside SWA NOT hashed!
★★★★★★★★★ This means SWA MLA blocks are eligible for eviction → reduces memory
★★★★★★★★★ But also means: stale MLA state can survive if eviction is delayed!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 7.3 KVCacheBlocks — The Interface Object

```python
@dataclass
class KVCacheBlocks:
    """Allocation result of KVCacheManager, interface between Scheduler and Manager."""
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = i-th kv_cache_group, j-th block of tokens

    def get_block_ids(self) -> tuple[list[int], ...]:
        # Returns (group_0_block_ids, group_1_block_ids, ...)

    def get_unhashed_block_ids(self) -> list[int]:
        # Blocks WITHOUT hash → newly allocated, not in prefix cache yet

    def new_empty(self) -> "KVCacheBlocks":
        # Pre-allocated empty result → avoids GC overhead
```

---

## 8. RTX 4090 Impact Analysis

### 8.1 Memory Budget on RTX 4090 (24 GiB)

```
★★★★★★★★★ RTX 4090 memory partitioning for GRPO training (HYBRID mode):
★★★★★★★★★
★★★★★★★★★ Qwen3-30B-A3B (MoE, 3B active params):
★★★★★★★★★   Model weights:    ~6 GiB (BF16, 3B active)
★★★★★★★★★   Optimizer states:  ~12 GiB (ZeRO-2 + CPU_Adam, GPU resident: ~0 GiB)
★★★★★★★★★   KV cache:          ~4-6 GiB (depends on max_model_len + block_size)
★★★★★★★★★   Gradients:         ~6 GiB (during training step)
★★★★★★★★★   CUDA overhead:     ~2 GiB (fragmentation, PyTorch allocator)
★★★★★★★★★
★★★★★★★★★ Sleep/Wake flow:
★★★★★★★★★   Sleep (S2):       free ~16 GiB (weights + KV + graphs)
★★★★★★★★★   Training:         use ~12 GiB (weights + gradients)
★★★★★★★★★   Wake weights:     ~6 GiB (reallocate)
★★★★★★★★★   Update weights:   ~6 GiB (in-place)
★★★★★★★★★   Wake KV cache:    ~6 GiB (reallocate)
★★★★★★★★★   Peak awake:       ~12 GiB (weights + KV) → fits in 24 GiB!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 8.2 DSV4 on RTX 4090 — BLOCKED

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ DSV4 CANNOT run on RTX 4090 in production:
★★★★★★★★★ → 671B params × BF16 = ~1342 GiB → needs multi-GPU TP
★★★★★★★★★ → RTX 4090 single GPU → TP=1 → model doesn't fit!
★★★★★★★★★ → Even with extreme quantization: 671B × FP8 = ~671 GiB → still impossible
★★★★★★★★★
★★★★★★★★★ DSV4-Flash (smaller variant) might fit with TP=4 + EP:
★★★★★★★★★ → But DSV4 instability means: enforce_eager=True MANDATORY
★★★★★★★★★ → enforce_eager = no CUDA graph = ~30% throughput loss
★★★★★★★★★ → Plus: S2 sleep level forced to S1 for EP > 1 → less memory savings
★★★★★★★★★ → RTX 4090 + DSV4 = impractical for production RLHF
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 8.3 BudgetRefiner SLO on RTX 4090 — P10 Opportunity

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ BudgetRefiner SLO = #1 OSS contribution (unique RTX 4090 profile data)
★★★★★★★★★
★★★★★★★★★ Why unique:
★★★★★★★★★ → No other vLLM contributor has RTX 4090 profile_table.csv
★★★★★★★★★ → BudgetRefiner core logic = 58 lines, 100% GPU-generic
★★★★★★★★★ → Only profile_table.csv is HW-specific → RTX 4090 data fills the gap
★★★★★★★★★ → 95%+ GPU-generic: A2/B3 data → same logic, same code, different CSV
★★★★★★★★★
★★★★★★★★★ What needs profiling:
★★★★★★★★★ → (ctx_len, d_num) → iteration_cost for each chunk_size
★★★★★★★★★ → For Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B on RTX 4090
★★★★★★★★★ → SLO thresholds: 50ms, 100ms, 200ms → different budget curves
★★★★★★★★★
★★★★★★★★★ Connection to V1 memory architecture:
★★★★★★★★★ → BudgetRefiner controls token_budget → indirectly controls block demand
★★★★★★★★★ → Lower token budget → fewer blocks needed per step → less preemption
★★★★★★★★★ → On RTX 4090: limited blocks → BudgetRefiner prevents thrashing!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 9. Cross-Framework Connections

### 9.1 vLLM → SGLang Sleep/Wake Comparison

```
★★★★★★★★★ SGLang also has sleep/wake mechanism for RLHF:
★★★★★★★★★ → Similar to vLLM: model offload, KV cache discard
★★★★★★★★★ → SGLang #28612: DSV4 C128 state mapping lifecycle fix
★★★★★★★★★ → Same pattern: constant buffer restoration critical for correctness
★★★★★★★★★ → SGLang uses Triton kernels → different memory management
★★★★★★★★★ → But same fundamental issue: per-step dynamic data MUST NOT survive sleep/wake
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 9.2 vLLM → vLLM-Ascend Sleep/Wake

```
★★★★★★★★★ vLLM-Ascend has sleep/wake but with NPU-specific allocator:
★★★★★★★★★ → xpumem.py instead of cumem.py
★★★★★★★★★ → #10724: DSV4 crash on 2*A2 PD-Mix multi-node (Ascend)
★★★★★★★★★ → #10684: DSA Hadamard ALL-ZERO after sleep/wake → verl RLHF blocker!
★★★★★★★★★ → Same pattern: constant buffer lost during state transfer
★★★★★★★★★ → DSA Hadamard values = per-step dynamic data → destroyed during S2 sleep!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 9.3 vLLM → DeepSpeed ZeRO-2 Sleep/Wake Analogy

```
★★★★★★★★★ DeepSpeed ZeRO-2 + CPU_Adam = analogous to vLLM S2 sleep:
★★★★★★★★★ → ZeRO-2 shards optimizer states → GPU memory freed for model
★★★★★★★★★ → CPU_Adam keeps optimizer on CPU → same as S2 offloading weights
★★★★★★★★★ → overlap_comm=False MANDATORY on single GPU → same as enforce_eager for DSV4!
★★★★★★★★★ → Both: multi-stream execution breaks on single GPU → NaN/garbage output
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 9.4 vLLM → verl FSDP2 Weight Sync

```
★★★★★★★★★ verl #6512 MERGED: per-unit LoRA summon → 10x memory reduction
★★★★★★★★★ → Before: FSDP1 whole-model summon = 60 GiB peak → OOM on RTX 4090!
★★★★★★★★★ → After: FSDP2 per-unit summon = 6-8 GiB peak → fits RTX 4090!
★★★★★★★★★ → Dynamic FSDP unit discovery replaces 8 hard-coded LoRA prefixes
★★★★★★★★★
★★★★★★★★★ Connection to V1 memory management:
★★★★★★★★★ → FSDP2 sharding = analogous to BlockPool block allocation
★★★★★★★★★ → Per-unit summon = analogous to allocate_slots for individual requests
★★★★★★★★★ → Watermark in V1 = analogous to FSDP2 memory reservation
★★★★★★★★★ → Both: memory budget must be managed to avoid OOM on 24 GiB!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 10. Source Code File Map — Complete Reference

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/v1/core/kv_cache_manager.py` | 614 | Main KVCacheManager — allocation, freeing, hashing interface |
| `vllm/v1/core/block_pool.py` | 527 | BlockPool — physical block pool, prefix cache, eviction |
| `vllm/v1/core/kv_cache_utils.py` | ~400 | KVCacheBlock, FreeKVCacheBlockQueue, hash utilities |
| `vllm/v1/core/kv_cache_coordinator.py` | ~250 | KVCacheCoordinator — orchestrates multi-group managers |
| `vllm/v1/core/single_type_kv_cache_manager.py` | ~400 | Per-attention-type managers (Full, SWA, MLA, Mamba) |
| `vllm/v1/core/sched/scheduler.py` | 2618 | Scheduler — token budget, preemption, request lifecycle |
| `vllm/v1/worker/gpu_worker.py` | ~600 | GPU worker — sleep/wake, buffer preservation |
| `vllm/v1/worker/gpu_model_runner.py` | ~5400 | Model runner — init_fp8_kv_scales, post_kv_cache_wake_up |
| `vllm/device_allocator/cumem.py` | ~300 | CuMemAllocator — CUDA virtual memory pool, sleep/wake |
| `vllm/v1/kv_cache_interface.py` | ~900 | KVCacheConfig, KVCacheSpec, KVCacheGroupSpec definitions |
| `vllm/v1/kv_cache_spec_registry.py` | ~200 | Registry mapping model → KVCacheSpec per layer |
| `vllm/v1/kv_offload/base.py` | ~200 | KV offload abstractions (OffloadPolicy, LoadStoreSpec) |
| `vllm/v1/engine/core.py` | ~2200 | EngineCore — KV cache initialization, block count determination |
| `vllm/model_executor/offloader/base.py` | ~150 | Weight offloader hierarchy (Noop, UVA, Prefetch) |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | ~200 | verl HYBRID adapter — sleep/wake, weight sync |
| `verl/workers/rollout/vllm_rollout/weight_update_utils.py` | ~100 | Buffer vs parameter split, in-place update |
| `verl/third_party/vllm/__init__.py` | ~50 | VLLM_SLEEP_LEVEL=2 default, version gating |

---

## 11. Key Findings Summary

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. V1 KV cache = radical simplification over V0:
   → Single BlockPool replaces multiple BlockSpaceManager variants
   → Doubly-linked free list → O(1) allocation/free with LRU eviction
   → Hash-based prefix caching → automatic, built-in, not optional
   → FractionalBlockTable removed → scheduler_block_size replaces it

2. CuMemAllocator = CUDA virtual memory pool for sleep/wake:
   → cuMemMap/cuMemUnmap → physical backing released, virtual address preserved
   → Tags for selective offload: S1 offloads weights, S2 discards everything
   → Constant buffer preservation via _sleep_saved_buffers → CRITICAL for correctness

3. Sleep/Wake is THE memory lifecycle mechanism for RLHF:
   → S2 default in verl → maximum memory savings for training
   → Two-phase wake: weights first → update → then KV cache → avoids OOM
   → FP8 KV scales reset to 1.0 after wake → potential accuracy issue for calibrated models

4. DSV4 breaks because per-step dynamic data cannot survive ANY lifecycle transition:
   → CUDA graph replay: uses capture-time static data → WRONG routing decisions
   → Prefix cache: dynamic routing data changes per step → stale cache = garbage
   → Sleep/wake: constant buffers preserved but routing state destroyed → same pattern
   → enforce_eager=True MANDATORY for DSV4 → no CUDA graph → -30% throughput

5. BudgetRefiner SLO = token budget governance, NOT block budget:
   → Soft constraint on prefill chunk size when decode requests are active
   → RTX 4090 has NO profile data → contribution opportunity!
   → 58 lines core logic, 100% GPU-generic → only CSV needs RTX 4090 data

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## References

- vLLM V1 KV Cache Manager: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- vLLM V1 BlockPool: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py
- vLLM V1 Scheduler: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- vLLM Sleep Mode Docs: https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md
- vLLM CuMemAllocator: https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py
- vLLM Sleep Mode Blog: https://blog.vllm.ai/2025/10/26/sleep-mode.html
- vLLM V1 KV Cache Blog: https://blog.vllm.ai/2024/v1-kv-cache.html
- vLLM #45972 (2nd DSV4 revert): https://github.com/vllm-project/vllm/pull/45972
- vLLM #45979 (3rd DSV4 revert): https://github.com/vllm-project/vllm/pull/45979
- verl HYBRID rollout: https://github.com/volcengine/verl/blob/main/verl/workers/rollout/vllm_rollout/vllm_rollout.py
- verl Weight Update Utils: https://github.com/volcengine/verl/blob/main/verl/workers/rollout/vllm_rollout/weight_update_utils.py
- vLLM-Ascend #10684 (DSA Hadamard sleep/wake ALL-ZERO): https://github.com/vllm-project/vllm-ascend/issues/10684
- vLLM-Ascend #10724 (DSV4 8th failure): https://github.com/vllm-project/vllm-ascend/issues/10724
- SGLang #28612 (DSV4 C128 state mapping): https://github.com/sgl-project/sglang/pull/28612
- Local: budgetrefiner-slo-source-reading.md (58 lines core logic, profile table schema)
- Local: dsv4-systematic-instability-pattern-synthesis.md (9 failures across 3 frameworks)
- Local: vllm-engine-core-reading.md (EngineCore architecture, ZMQ, step loop)
- Local: vllm-architecture.md (V1 multi-process architecture overview)
