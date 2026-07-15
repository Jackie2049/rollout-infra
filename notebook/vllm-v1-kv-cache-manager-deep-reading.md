# vLLM V1 KV Cache Manager 深度阅读

> Date: 2026-07-15
> Source: /tmp/vllm-fork/vllm/v1/core/
> Focus: KV cache allocation, prefix caching, scheduler interaction, GRPO relevance

---

## 1. Architecture Overview

```
KVCacheManager (kv_cache_manager.py)         ← Top-level facade
    │
    ├── KVCacheCoordinator (kv_cache_coordinator.py)   ← Multi-group orchestrator
    │     ├── UnitaryKVCacheCoordinator         ← Single KV group (most models)
    │     ├── HybridKVCacheCoordinator          ← Multiple KV types (DSv4, Jamba)
    │     └── KVCacheCoordinatorNoPrefixCache   ← Caching disabled
    │
    ├── BlockPool (block_pool.py)                       ← Shared physical block pool
    │     ├── BlockHashToBlockMap                       ← 1:N hash→blocks mapping
    │     ├── FreeKVCacheBlockQueue                     ← O(1) doubly-linked free list
    │     └── KVCacheMetricsCollector                   ← Sampling-based metrics
    │
    └── SingleTypeKVCacheManager (single_type_kv_cache_manager.py)  ← Per-group logic
          ├── FullAttentionManager            ← Full attn + MLA + HiddenState
          ├── SlidingWindowManager            ← SWA (right-to-left scan)
          ├── ChunkedLocalAttentionManager    ← Chunked local (DSv4-like)
          ├── MambaManager                   ← Mamba (align/default modes)
          ├── CrossAttentionManager          ← Encoder-decoder cross-attn
          └── SinkFullAttentionManager        ← StreamingLLM sink attention
```

Key design principle: **Physical blocks are shared across all KV cache groups** via a single
`BlockPool`, while logical management (prefix cache hit, eviction, block tracking) is
per-group via `SingleTypeKVCacheManager` subclasses. The `KVCacheCoordinator` reconciles
conflicting constraints across groups.

---

## 2. BlockHashToBlockMap: 1:N Mapping

**File:** `/tmp/vllm-fork/vllm/v1/core/block_pool.py` lines 34-128

### Design

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

The key insight: `BlockHash` is a `bytes` NewType, and `BlockHashWithGroupId` packs a 4-byte
group ID suffix onto the hash bytes. This allows the **same physical block to be found** for
multiple KV cache groups sharing the same prefix tokens.

The value is a **union type**:
- When 1 block matches a hash → stored directly as `KVCacheBlock` (avoids dict overhead)
- When N blocks match the same hash → stored as `dict[int, KVCacheBlock]` keyed by block_id

### Why 1:N?

The comment says (line 48-52): "We currently don't de-duplicate the blocks in the cache,
meaning that if a block becomes full and is cached, we don't check if there is already
an identical block in the cache. This is because we want to make sure the allocated block
IDs won't change so that block tables are append-only."

This means two requests with the same prefix can have **different physical block IDs** mapping
to the same logical content. The 1:N mapping enables finding any matching block for prefix
cache hits while preserving block table stability.

### Operations

- `get_one_block(key)` → Returns any block with matching hash (first from dict)
- `insert(key, block)` → Escalates single→dict when duplicate hash appears
- `pop(key, block_id)` → Removes specific block_id; shrinks dict→single when size drops to 1

### GRPO Implication

In GRPO, the same prompt prefix is shared across multiple rollout completions. With 1:N mapping,
each completion can reuse a cached block for the prompt prefix **without sharing the same physical
block ID**. This prevents cross-request interference when one request frees its blocks.

---

## 3. FreeKVCacheBlockQueue: O(1) Doubly-Linked List

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_utils.py` lines 165-374

### Design

Custom doubly-linked list implementation using `prev_free_block` / `next_free_block` attributes
directly on `KVCacheBlock` objects. This avoids allocating separate node objects and supports
O(1) removal from the middle of the list (critical for `touch()` which removes blocks from the
free list when they're reused by prefix cache hits).

### Sentinel Nodes

```python
fake_free_list_head: KVCacheBlock(block_id=-1)  # Never popped
fake_free_list_tail: KVCacheBlock(block_id=-1)  # Never popped
```

Using fake head/tail eliminates branching in traversal and manipulation code.

### Eviction Order

The queue is ordered by **LRU + tail-preference**:
1. Least recently used blocks at the front (evicted first)
2. Blocks from the same allocation batch are ordered so tail blocks are evicted first

This ordering is maintained by **reversing** the block list when freeing a request's blocks
(see `free()` method: `ordered_blocks = reversed(req_blocks)`).

### Key Operations

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `popleft()` | O(1) | Allocate from front (LRU eviction candidate) |
| `popleft_n(n)` | O(n) | Batch allocation |
| `remove(block)` | O(1) | Touch: remove from free list on prefix cache hit |
| `append(block)` | O(1) | Free: add back at tail |
| `append_n(blocks)` | O(n) | Batch free |

### GRPO Implication

In GRPO with variable-length completions, shorter completions free their blocks sooner. The
LRU ordering ensures these freed blocks are reused for new requests before evicting blocks
from still-running longer completions. The O(1) `remove()` operation is critical for the
`touch()` flow during prefix cache reuse.

---

## 4. KVCacheBlock Metadata

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_utils.py` lines 116-162

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                # Physical block index [0, num_gpu_blocks-1]
    ref_cnt: int = 0             # Reference count (0 = in free list)
    _block_hash: BlockHashWithGroupId | None = None  # Cached hash (only for full cached blocks)
    prev_free_block: KVCacheBlock | None = None       # Doubly-linked list
    next_free_block: KVCacheBlock | None = None       # Doubly-linked list
    is_null: bool = False        # Null block placeholder (never cached/freed)
```

Key behaviors:
- `ref_cnt=0` → block is in free list (eviction candidate)
- `ref_cnt>0` → block is actively used by at least one request
- `block_hash=None` → block not yet cached in prefix cache
- `block_hash set` → block is in `cached_block_hash_to_block` hash map
- `is_null=True` → placeholder for skipped blocks (SWA, Mamba, chunked local)

The `ref_cnt` mechanism enables **prefix sharing**: when request B hits a prefix block already
allocated by request A, `touch()` increments `ref_cnt` on that block. The block is only freed
when `ref_cnt` drops to 0 (all sharing requests have finished or been preempted).

---

## 5. Block Hash Computation

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_utils.py` lines 542-569

### Chain Hashing

```python
def hash_block_tokens(
    hash_function, parent_block_hash, curr_block_token_ids, extra_keys
) -> BlockHash:
```

Block hashes are **chained**: each block's hash depends on the parent block's hash plus the
current block's token IDs plus optional extra keys. This creates a Merkle-tree-like structure
where the hash of block N encodes the entire prefix from block 0 to block N.

- `parent_block_hash=None` → uses `NONE_HASH` (random seed or PYTHONHASHSEED-derived)
- `hash_function` → configurable (sha256, xxhash, or custom via CBOR encoding)

### Extra Keys

Extra keys differentiate blocks that have the same token IDs but different contexts:
1. **LoRA name** (`request.lora_request.lora_name`) → LoRA requests get different hashes
2. **MM features** (`mm_feature.identifier, offset`) → multimodal inputs affect hash
3. **Cache salt** (`request.cache_salt`) → explicit user-provided hash salt (first block only)
4. **Prompt embeds** (SHA256 of prompt embeddings per block range)

### GRPO LoRA Implication

**Critical**: LoRA requests include `lora_name` in the block hash. This means:
- Same prompt tokens with different LoRA adapters → **different block hashes** → no prefix sharing
- Same LoRA adapter + same prompt → **shared prefix cache hit** → major memory savings

For GRPO: all rollout completions for a prompt use the **same LoRA adapter** (or no LoRA),
so they will share the prompt prefix blocks. The `lora_name` extra key prevents accidental
cross-adapter prefix sharing (which would produce incorrect KV values since LoRA changes the
KV computation).

### BlockHashList and BlockHashListWithBlockSize

For hybrid models (multiple KV cache groups with different block sizes), the `BlockHashListWithBlockSize`
class lazily converts block hashes from `hash_block_size` granularity to `target_block_size`
granularity by concatenating consecutive hashes. Example: block_size 16 hashes [A, B] become
a single block_size 32 hash `AB`.

---

## 6. KVCacheManager: Top-Level Facade

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_manager.py` lines 110-573

### Key Methods

| Method | Purpose | GRPO Relevance |
|--------|---------|---------------|
| `get_computed_blocks(request)` | Find prefix cache hit for a new request | HIGH: prompt reuse |
| `allocate_slots(request, ...)` | Allocate blocks for new + existing tokens | HIGH: every scheduling step |
| `free(request)` | Free all blocks for a finished request | HIGH: after completion |
| `reset_prefix_cache()` | Invalidate all prefix cache entries | CRITICAL: weight update |
| `cache_blocks(request, num_tokens)` | Cache full blocks into prefix cache | MEDIUM: after compute |
| `evict_blocks(block_ids)` | Evict specific blocks from prefix cache | MEDIUM: KV connector |
| `remove_skipped_blocks(req_id, tokens)` | Free blocks outside attention window | LOW: SWA only |

### KVCacheBlocks Data Structure

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # blocks[group_idx][block_idx]
```

This is the interface between Scheduler and KVCacheManager. The outer tuple indexes KV cache
groups, inner sequences index blocks within each group. For single-group models (most cases),
this is effectively a flat list.

Key design: `empty_kv_cache_blocks` is pre-constructed to avoid GC overhead for requests with
no prefix cache hits.

### allocate_slots() Flow (THE Critical Path)

This is the most complex method (lines 238-429). The block layout is:

```
| <comp> | <new_comp> | <ext_comp> | <new> | <lookahead> |
```

Three-stage allocation:
1. **Free unnecessary blocks** in `comp` range (e.g., outside sliding window)
2. **Handle prefix tokens**: Touch computed blocks (increment ref_cnt), allocate for external tokens
3. **Allocate new blocks** for `new + lookahead` tokens from BlockPool

Returns `None` if insufficient free blocks (scheduler must preempt).

---

## 7. BlockPool: Shared Physical Block Pool

**File:** `/tmp/vllm-fork/vllm/v1/core/block_pool.py` lines 130-521

### Key Properties

- `num_gpu_blocks`: Total number of physical blocks (computed at startup)
- `null_block`: Sentinel block (block_id=0) for placeholder positions, never cached/freed
- `blocks`: List of all `KVCacheBlock` objects
- `free_block_queue`: Doubly-linked free list
- `cached_block_hash_to_block`: The 1:N hash map

### Block Allocation Flow

```python
def get_new_blocks(num_blocks):
    # 1. Pop from free_block_queue (LRU eviction order)
    # 2. If caching enabled: evict any cached block's hash metadata
    # 3. Set ref_cnt = 1
    # 4. Track metrics
```

When allocating a block that currently has a cached hash (`_maybe_evict_cached_block`),
the block's hash is popped from `cached_block_hash_to_block` and reset. This means
**allocating a new block can evict a prefix cache entry** if that entry's physical block
is reused.

### touch() Flow (Prefix Cache Hit)

```python
def touch(blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            # Remove from free list (was eviction candidate)
            free_block_queue.remove(block)
        block.ref_cnt += 1  # Share the block
```

When a prefix cache hit occurs, the cached block's `ref_cnt` is incremented and it's
removed from the free list. This is the **prefix sharing** mechanism.

### reset_prefix_cache() (Weight Update Critical Path)

```python
def reset_prefix_cache():
    # 1. Check all blocks are freed (except null_block)
    # 2. Replace cached_block_hash_to_block with empty BlockHashToBlockMap
    # 3. Reset all block hashes (block.reset_hash())
    # 4. Log "Successfully reset prefix cache"
```

**CRITICAL GRPO ISSUE**: This method requires ALL blocks to be freed (only null_block remains).
If any request is still running, it returns `False`. In RLHF flows, the scheduler must preempt
all running requests before calling this. This is the **weight update → stale cache** problem.

---

## 8. KVCacheCoordinator: Multi-Group Orchestration

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_coordinator.py` lines 28-692

### Three Variants

| Coordinator | When Used | Prefix Cache |
|-------------|-----------|-------------|
| `KVCacheCoordinatorNoPrefixCache` | Caching disabled / no KV groups | Never |
| `UnitaryKVCacheCoordinator` | Single KV group (most LLMs) | Left-to-right scan |
| `HybridKVCacheCoordinator` | Multiple KV types (Jamba, DSv4) | Fixed-point convergence |

### UnitaryKVCacheCoordinator

For single-group models, prefix cache lookup is straightforward:
- `find_longest_cache_hit()` delegates to the single `SingleTypeKVCacheManager.find_longest_cache_hit()`
- Returns `hit_blocks, len(hit_blocks[0]) * block_size`

### HybridKVCacheCoordinator (CRITICAL for DSv4-Hybrid)

Uses an **iterative fixed-point algorithm**:

```python
while True:
    curr_hit_length = hit_length
    for spec_group in attention_groups:
        hit_blocks = manager_cls.find_longest_cache_hit(...)
        curr_hit_length = len(hit_blocks[0]) * spec.block_size
        if curr_hit_length < hit_length:
            eagle_verified.clear()  # Reset EAGLE verification
    if curr_hit_length >= hit_length:
        break  # Converged
```

**Why fixed-point?** In hybrid models, full attention and SWA groups have conflicting constraints:
- Full attention: all prefix blocks are valid → downward-closed
- SWA: only the last N tokens matter → can reduce the valid prefix length

Each group can reduce the hit length, requiring re-checking all groups. Convergence is guaranteed
because hit_length monotonically decreases and is bounded by 0.

For "simple hybrid" (1 full attn + 1 other group), one iteration suffices.

---

## 9. SingleTypeKVCacheManager Per-Group Logic

**File:** `/tmp/vllm-fork/vllm/v1/core/single_type_kv_cache_manager.py`

### FullAttentionManager (lines 510-569)

**Most common case**. Left-to-right scan through block hashes:

```python
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        computed.append(cached)
    else:
        break  # Chain breaks: no further blocks can be cached
```

This exploits the **chained hash** property: if block N misses, all blocks >N must also miss
(because their hashes depend on block N's hash).

If EAGLE/MTP is enabled, the last matched block is dropped to force recomputation for the
draft head's hidden state.

Alignment: if `alignment_tokens > block_size` (hybrid model), blocks are trimmed until
`len(hit_blocks) * block_size % alignment_tokens == 0`.

### SlidingWindowManager (lines 571-740)

**Fundamentally different approach**: right-to-left scan with contiguous block requirement.

SWA can only reuse the **last N tokens** (window_size). So it needs `cdiv(window_size-1, block_size)`
contiguous cached blocks at the end of the prefix. It searches from right to left, finding
the first match and checking if enough contiguous blocks precede it.

Hit result format: `[null_block, null_block, ..., hit_block1, hit_block2]` where null blocks
represent prefix positions outside the window.

### MambaManager (lines 893-1171)

**Most complex manager**. Two modes:
- `"default"`: similar to FullAttentionManager but with speculative block allocation
- `"align"`: prefix caching with Mamba state alignment. Only needs 1 running state block
  plus speculative blocks. Uses `last_state_block_idx` to track which block holds the
  previous step's Mamba state for copy-on-write.

**GRPO concern**: Mamba blocks use `cached_blocks_this_step` to prevent a request from
reusing blocks cached by **another request in the same scheduling step**. This is because
Mamba state is order-dependent (sequential computation). Returns `num_gpu_blocks + 1` to
block scheduling in the current step and defer to next step.

---

## 10. Scheduler-KVCacheManager Interaction

**File:** `/tmp/vllm-fork/vllm/v1/core/sched/scheduler.py`

### schedule() Flow (lines 339-960)

```
1. kv_cache_manager.new_step_starts()           ← Clear per-step state (Mamba cached_blocks_this_step)

2. RUNNING requests (decode phase):
   for request in running:
       num_new_tokens = num_tokens_with_spec - num_computed_tokens
       while True:
           new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
           if new_blocks is None:
               preempt lowest-priority request → kv_cache_manager.free(preempted_req)
               retry
           else:
               break

3. WAITING requests (prefill phase):
   for request in waiting:
       if request.num_computed_tokens == 0:
           new_computed_blocks, num_new_tokens = kv_cache_manager.get_computed_blocks(request)
           ← Prefix cache lookup
       new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
       ← Full allocation including prefix cache hits

4. After scheduling:
   num_common_prefix_blocks = kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
   ← For cascade attention optimization
```

### Prefill vs Decode KV Cache Handling

| Aspect | Prefill (new request) | Decode (running request) |
|--------|----------------------|-------------------------|
| Prefix cache lookup | `get_computed_blocks()` | Skip (running reqs don't have new hits) |
| Block allocation | `allocate_slots()` with `new_computed_blocks` | `allocate_slots()` without |
| num_computed_tokens | Set from cache hit + external tokens | Increment by num_scheduled_tokens |
| Block hash caching | After compute via `cache_blocks()` | After compute via `cache_blocks()` |

### Preemption (lines 959-979)

```python
def _preempt_request(request, timestamp):
    self.kv_cache_manager.free(request)           # Free ALL blocks
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0                # Must recompute everything
    request.num_preemptions += 1
    self.waiting.prepend_request(request)
```

Preemption completely frees all blocks and resets computed tokens to 0. On resumption,
the request starts from scratch but may hit prefix cache for the prompt tokens.

---

## 11. Weight Update and KV Cache (CRITICAL for GRPO)

### #46125 Revert Context

PR #45093 attempted to preserve KV cache across weight updates by only resetting prefix
cache hashes without freeing blocks. This was reverted in #46125 because **stale KV cache
produces incorrect attention scores** after weight update.

### Current Flow: reset_prefix_cache()

```python
# Engine core:
def sleep(level=1):
    if level >= 1:
        clear_prefix_cache = True
        pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        # Preempt all running requests
        # Reset prefix cache: invalidate all block hashes
        model_executor.sleep(level)  # CuMemAllocator offloads weights

def wake_up(tags=None):
    model_executor.wake_up(tags)  # CuMemAllocator restores weights
    if tags is None or "kv_cache" in tags:
        model_runner.post_kv_cache_wake_up()  # Zero out KV cache tensors, reset FP8 scales
```

### Sleep Levels and KV Cache Impact

| Level | Scheduler | Prefix Cache | GPU Memory | KV Cache Tensors |
|-------|-----------|-------------|------------|-----------------|
| 0 | Paused (new only) | Not cleared | No change | Preserved |
| 1 | Paused (all) + cleared | Reset (hashes cleared) | Weights offloaded to CPU | Preserved (but stale) |
| 2 | Paused (all) + cleared | Reset (hashes cleared) | All GPU freed | Discarded |

**Level 1 concern**: KV cache tensors are preserved in GPU memory after wake_up, but their
**content is stale** (computed with old weights). The `post_kv_cache_wake_up()` method zeros
out KV cache tensors and resets FP8 scales to 1.0, but this only affects the physical storage.
The scheduler's `reset_prefix_cache()` ensures no new request will hit stale cached blocks.

### verl sleep/wake Integration

In verl's HYBRID mode:
- `sleep_level=1`: tags=["kv_cache"] → only LoRA adapter is offloaded, base weights stay
- `sleep_level=2`: tags=["kv_cache", "weights"] → full re-transfer

After wake_up, vLLM must:
1. Reset prefix cache (invalidate stale hashes)
2. Zero out KV cache tensors (physical storage reset)
3. Reset FP8 scales to 1.0

---

## 12. LoRA and KV Cache

### LoRA Block Hash Isolation

LoRA requests include `lora_name` as an extra key in block hash computation:
```python
def _gen_lora_extra_hash_keys(request):
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]
```

This means:
- Same tokens + same LoRA → shared prefix cache (correct: same KV computation)
- Same tokens + different LoRA → different hashes (correct: different KV computation)
- Same tokens + no LoRA vs LoRA → different hashes (correct: base model KV differs from LoRA)

### LoRA Weight Update After sleep/wake

After LoRA adapter offload/reload:
1. All running requests are preempted
2. Prefix cache is reset (LoRA-specific hashes invalidated)
3. KV cache tensors zeroed out
4. On next scheduling step, requests re-enter as WAITING and recompute from scratch

### GRPO LoRA Scenario

In GRPO with LoRA:
- All completions for the same prompt use the same LoRA adapter → prefix sharing works
- After weight update, prefix cache is properly invalidated via `reset_prefix_cache()`
- But: **all completions must be finished** before weight update, because `reset_prefix_cache()`
  requires all blocks freed (ref_cnt check)

---

## 13. num_gpu_blocks Calculation

**File:** `/tmp/vllm-fork/vllm/v1/core/kv_cache_utils.py` line 936-953

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    num_blocks = int(available_memory // page_size // num_layers)
    return may_override_num_blocks(vllm_config, num_blocks)
```

Where:
- `available_memory` = GPU memory remaining after model weights + activations
- `page_size` = bytes per block per layer (block_size * num_kv_heads * head_dim * dtype_size)
- `num_layers` = number of attention layers sharing the same KV cache spec

The `num_gpu_blocks_override` config option can override the computed value (for debugging
or memory budgeting).

### RTX 4090 Budget

For a 7B model with block_size=16, 32 KV heads, head_dim=128, bf16:
- page_size per layer = 16 * 32 * 128 * 2 = 131,072 bytes (~128 KiB)
- 32 layers → 4 MiB per block across all layers
- 24 GiB GPU → ~1 GiB model weights → ~23 GiB available → ~5,750 blocks

Each block stores 16 tokens. Max concurrent tokens ≈ 5,750 * 16 = 92,000 tokens.

---

## 14. Comparison: vLLM Block Hash vs SGLang RadixAttention

### Data Structure Comparison

| Aspect | vLLM V1 | SGLang |
|--------|---------|--------|
| Structure | Flat hash map (BlockHash → Block) | Radix tree (token sequence → TreeNode) |
| Lookup | Hash-based O(1) per block | Tree traversal O(tokens) |
| Matching granularity | block_size tokens (default 16) | Variable-length (any token count) |
| Sharing mechanism | ref_cnt on physical blocks | lock_ref on tree nodes |
| Eviction | LRU via doubly-linked free list | LRU/LFU/FIFO configurable |
| Memory model | Page-based (paged attention) | Token-based (req_to_token_pool) |
| LoRA isolation | lora_name in hash extra keys | extra_key in RadixKey |
| Null blocks | Placeholder for skipped positions | Not needed (tree doesn't store skips) |

### Prefix Matching Comparison

**vLLM**: Block-hash chain matching. Each block hash encodes the full prefix from block 0.
Lookup is left-to-right (for full attention): if block N misses, all blocks >N must miss.
Hit granularity is always `block_size` (16 tokens by default).

**SGLang**: Radix tree longest prefix match. The tree stores variable-length token sequences
on edges, allowing matches at arbitrary token granularity. Node splitting enables partial
reuse (e.g., sharing a 100-token prefix when only 80 tokens match).

### GRPO Variable-Length Sequences

**vLLM advantage**: Hash-based lookup is O(1) per block. Fast for large batches of GRPO
completions sharing the same prompt prefix.

**vLLM disadvantage**: Hit granularity is `block_size` aligned. If the prompt is 103 tokens
with block_size=16, only 96 tokens (6 full blocks) can be cached. The remaining 7 tokens
must be recomputed every time.

**SGLang advantage**: Variable-length matching means any-length prefix can be reused. No
alignment waste. The radix tree naturally handles partial prefix reuse.

**SGLang disadvantage**: Tree traversal is O(tokens) for lookup, slower for very long prompts.
However, radix tree edges compress sequences, reducing traversal depth.

### Hybrid Model Handling

**vLLM**: The `HybridKVCacheCoordinator` uses a fixed-point algorithm to reconcile conflicting
constraints across KV cache groups. This is complex but necessary for DSv4-Hybrid models.

**SGLang**: All attention types share the same radix tree. SWA groups don't need special
handling in the tree structure; eviction handles the window constraint separately.

---

## 15. Critical Findings for GRPO

### 15.1 Prefix Cache Hit Workflow (GRPO Prompt Reuse)

```
1. GRPO step: generate N completions for prompt P
2. All N requests share same prompt tokens → same block hashes
3. First request allocates new blocks for prompt → caches them
4. Subsequent requests hit prefix cache via get_computed_blocks()
5. Only completion-specific blocks need new allocation
6. Memory savings: prompt length * (N-1) blocks saved
```

### 15.2 Weight Update Stale Cache Problem

```
1. Training step: update policy weights (LoRA or full)
2. Must call reset_prefix_cache() → requires all blocks freed
3. All running rollout requests must finish or be preempted
4. After reset, next rollout step recomputes prompt KV from scratch
5. Cost: prompt tokens recomputed every training step
```

**Optimization opportunity**: vLLM currently zeros all KV cache tensors on wake_up. An
alternative would be to selectively preserve prompt KV (since the prompt is the same across
GRPO steps), but this requires careful validation that the KV values are still correct after
weight update. The #46125 revert shows this is dangerous.

### 15.3 Variable-Length Completion Handling

GRPO completions have variable lengths. In vLLM:
- Each completion gets its own block allocation via `allocate_slots()`
- Shorter completions free blocks sooner → LRU order ensures reuse
- No cross-request interference because each has its own `req_to_blocks` tracking
- The `KVCacheBlocks` structure allows per-group block tracking (useful for hybrid models)

### 15.4 block_size Alignment Waste

With block_size=16, a 103-token prompt wastes 7 tokens (6.8%) on every prefix cache hit.
For GRPO with short completions (50-200 tokens), this waste can be significant if the
prompt + completion doesn't align well to block boundaries.

**Mitigation**: vLLM allows `hash_block_size` to differ from `block_size` for hybrid models,
but for single-group models they must be equal. No current option to reduce alignment waste.

### 15.5 Sleep/Wake KV Cache Handling

After sleep/wake cycle in GRPO (verl HYBRID mode):
1. `sleep_level=1`: LoRA adapter offloaded, KV cache preserved in GPU
2. `wake_up(tags=["kv_cache"])`: LoRA restored, KV cache tensors zeroed, FP8 scales reset
3. `reset_prefix_cache()`: All hash metadata cleared → no stale hits
4. All rollout requests preempted → blocks freed → KV cache storage reclaimed

**Key insight**: `post_kv_cache_wake_up()` zeros KV cache tensors AFTER `wake_up()`. This
means during the weight transfer phase, stale KV values are physically in GPU memory but
no request can access them (scheduler is paused, all requests preempted).

---

## 16. Manager Registry and Custom KV Cache Specs

**File:** `/tmp/vllm-fork/vllm/v1/core/single_type_kv_cache_manager.py` lines 1251-1348

```python
KVCacheSpecRegistry.register(FullAttentionSpec, FullAttentionManager, ...)
KVCacheSpecRegistry.register(SlidingWindowSpec, SlidingWindowManager, ...)
KVCacheSpecRegistry.register(MambaSpec, MambaManager, ...)
KVCacheSpecRegistry.register(MLAAttentionSpec, FullAttentionManager, ...)
KVCacheSpecRegistry.register(CrossAttentionSpec, CrossAttentionManager, ...)
KVCacheSpecRegistry.register(SinkFullAttentionSpec, SinkFullAttentionManager, ...)
```

**MLA uses FullAttentionManager** — MLA attention is treated as full attention for prefix
caching purposes. This is correct because MLA compresses KV into a latent representation
but the caching behavior is identical to full attention (all prefix tokens are needed).

**Platform-specific specs**: `current_platform.register_custom_kv_cache_specs(vllm_config)`
allows hardware vendors (Ascend, etc.) to register custom KV cache specs and managers.

---

## 17. File Map

| File | Lines | Purpose |
|------|-------|---------|
| `kv_cache_manager.py` | 573 | Top-level facade |
| `block_pool.py` | 521 | Shared physical block pool + hash map |
| `kv_cache_coordinator.py` | 692 | Multi-group orchestration |
| `single_type_kv_cache_manager.py` | 1348 | Per-group logic (7 managers) |
| `kv_cache_utils.py` | 2154 | KVCacheBlock, FreeQueue, hash computation, block size resolution |
| `kv_cache_metrics.py` | 97 | Sampling-based block lifecycle metrics |
| `sched/scheduler.py` | ~2100 | Schedule() → KVCacheManager interaction |

---

## 18. Key Code Paths for GRPO

### Path 1: Prompt Prefix Reuse (N completions, same prompt)

```
Scheduler.schedule()
  → kv_cache_manager.get_computed_blocks(request_1)  [MISS: no cache yet]
  → kv_cache_manager.allocate_slots(request_1)       [ALLOC: new blocks for prompt]
  → kv_cache_manager.cache_blocks(request_1)         [CACHE: prompt blocks hashed]

  → kv_cache_manager.get_computed_blocks(request_2)  [HIT: prompt prefix cached]
  → allocate_slots(request_2, new_computed_blocks)   [TOUCH: ref_cnt++ on prompt blocks]
  → allocate_new_blocks(completion_2)                 [ALLOC: only completion blocks]
  → cache_blocks(request_2)                          [CACHE: completion blocks hashed]

  ... repeat for request_3..N
```

### Path 2: Weight Update + Cache Reset

```
EngineCore.sleep(level=1)
  → pause_scheduler(mode="abort", clear_cache=True)
    → scheduler.reset_prefix_cache(reset_running_requests=True)
      → preempt all running requests
      → kv_cache_manager.free(request) for each
      → block_pool.reset_prefix_cache()  [hash map cleared, all block hashes reset]
  → model_executor.sleep(1)
    → CuMemAllocator.sleep(offload_tags=("weights",))
    → weights offloaded to CPU

EngineCore.wake_up(tags=["kv_cache"])
  → model_executor.wake_up(["kv_cache"])
    → CuMemAllocator.wake_up(["kv_cache"])
    → model_runner.post_kv_cache_wake_up()
      → zero out KV cache tensors
      → reset FP8 scales to 1.0
  → resume_scheduler()
```

### Path 3: Preemption Under Memory Pressure

```
Scheduler.schedule() [decode phase]
  → kv_cache_manager.allocate_slots(request, num_new_tokens=1)
  → returns None [insufficient free blocks]
  → _preempt_request(lowest_priority_request)
    → kv_cache_manager.free(preempted_request)  [all blocks freed]
    → request.status = PREEMPTED
    → request.num_computed_tokens = 0
  → retry allocate_slots(request) [now succeeds]
```
