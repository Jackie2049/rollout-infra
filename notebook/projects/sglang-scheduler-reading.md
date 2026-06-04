# SGLang Scheduler Deep Source Code Analysis

> Source: `sglang/python/sglang/srt/managers/` (scheduler.py ~4034 lines, schedule_policy.py ~1070 lines, schedule_batch.py ~2799 lines)
> Date: 2026-06-04

## 1. Architecture Overview

### 1.1 Scheduler Class Hierarchy (Mixin Architecture)

SGLang uses a Mixin-based class hierarchy rather than a monolithic class:

```
Scheduler
  |-- SchedulerDisaggregationDecodeMixin   # P/D disaggregation (decode side)
  |-- SchedulerDisaggregationPrefillMixin   # P/D disaggregation (prefill side)
  |-- SchedulerMultiplexMixin               # PD-multiplexing mode
  |-- SchedulerPPMixin                      # Pipeline parallelism
  |-- SchedulerDllmMixin                    # Diffusion LLM support
  |-- SchedulerMlxOverlapMixin              # Apple MLX overlap (Mac)
```

Each Mixin provides specialized `init_*` and processing methods. The main `Scheduler.__init__` calls ~30+ `init_*` methods in strict order to configure memory, scheduling, overlap, disaggregation, profiling, etc.

### 1.2 Component Decomposition

SGLang has decomposed scheduler functionality into ~18 sub-components under `scheduler_components/`:

```
scheduler_components/
  |-- batch_result_processor.py   # Post-forward result handling (prefill/decode/prebuilt)
  |-- dp_attn.py                  # DP attention AllGather coordination
  |-- flush_wrapper.py            # Cache flush with pending-request check
  |-- idle_sleeper.py             # Sleep when idle to reduce CPU usage
  |-- invariant_checker.py        # Memory leak detection and pool consistency
  |-- ipc_channels.py             # ZMQ channels to TokenizerManager
  |-- kv_events_publisher.py      # KV cache event publishing
  |-- load_inquirer.py            # Load estimation for load balancing
  |-- logprob_result_processor.py # Logprob computation and formatting
  |-- metrics_reporter.py         # Prometheus metrics collection
  |-- new_token_ratio_tracker.py  # Adaptive new_token_ratio for retraction
  |-- output_sender.py            # Send outputs via IPC
  |-- output_streamer.py          # Streaming output delivery
  |-- pool_stats_observer.py      # KV pool statistics
  |-- profiler_manager.py         # Profiling control
  |-- request_receiver.py         # Receive requests from ZMQ
  |-- weight_updater.py           # Weight loading/update management
```

### 1.3 Core Scheduling Loop

SGLang has **two event loop modes**:

**Normal Mode** (`event_loop_normal`):
```
while True:
    recv_reqs = request_receiver.recv_requests()
    process_input_requests(recv_reqs)     # Route to handler via TypeBasedDispatcher
    batch = get_next_batch_to_run()       # Core scheduling decision
    if batch:
        result = run_batch(batch)         # Forward pass on GPU
        process_batch_result(batch, result)  # Post-processing
    else:
        on_idle()                         # Housekeeping
```

**Overlap Mode** (`event_loop_overlap`):
```
while True:
    recv_reqs = request_receiver.recv_requests()
    process_input_requests(recv_reqs)
    schedule_stream.wait_stream(forward_stream)  # WAR barrier
    batch = get_next_batch_to_run()
    if batch:
        batch_result = run_batch(batch)
        result_queue.append((batch.copy(), batch_result))
    if last_batch:
        process_batch_result(last_batch, last_result)  # Overlap!
    if batch is None and last_batch is None:
        on_idle()
```

The overlap mode processes the **previous** batch's results while the GPU executes the **current** batch, achieving CPU/GPU overlap.

```
+--- Iteration N ---+  +--- Iteration N+1 ---+
| recv + schedule   |  | recv + schedule      |
| GPU: forward(N)   |  | GPU: forward(N+1)    |
| CPU: process(N-1) |  | CPU: process(N)      |  <-- Overlap!
+-------------------+  +----------------------+
```

Key overlap mechanism: `FutureMap` relays decode `input_ids` (next token) across iterations without CPU staging. The `schedule_stream` writes `input_ids` which the `forward_stream` reads -- a WAR barrier (`schedule_stream.wait_stream(forward_stream)`) prevents races.

## 2. Core Data Structures

### 2.1 Req (Request State)

```
class Req(ReqDllmMixin):
    # Identity
    rid: str
    origin_input_ids: array[int]     # Original prompt token IDs
    output_ids: array[int]           # Generated output token IDs
    fill_ids: array[int]             # origin_input_ids + output_ids (updated if chunked)

    # Prefix caching (RadixAttention)
    prefix_indices: torch.Tensor     # KV cache indices for matched prefix
    last_node: Any                   # Last matched RadixCache tree node
    last_host_node: Any              # Last matched host-side node (HiCache)
    best_match_node: Any             # Best match for init_load_back
    host_hit_length: int             # Tokens hit from host cache
    num_matched_prefix_tokens: int   # Total matched prefix length
    cache_protected_len: int         # Prefix length inserted into tree

    # Memory management
    req_pool_idx: int                # Slot in req_to_token_pool
    kv_committed_len: int            # KV length committed to tree
    kv_allocated_len: int            # KV length allocated (>= committed)

    # Scheduling state
    extend_input_len: int            # Tokens to prefill in current round
    inflight_middle_chunks: int      # Remaining chunked prefill count
    is_retracted: bool               # Currently retracted
    retracted_stain: bool            # Ever been retracted

    # Finish state
    finished_reason: Optional[BaseFinishReason]
    to_finish: Optional[BaseFinishReason]  # Deferred abort
```

### 2.2 ScheduleBatch (Batch State)

```
@dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    # Core
    reqs: List[Req]

    # Shared resources (engine-lifetime)
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache

    # GPU tensors -> ForwardBatch
    input_ids: torch.Tensor          # [b], int64
    req_pool_indices: torch.Tensor   # [b], int64
    seq_lens: torch.Tensor           # [b], int64
    out_cache_loc: torch.Tensor      # [total_tokens], int64

    # Forward mode
    forward_mode: ForwardMode        # EXTEND / DECODE / MIXED / IDLE / ...

    # Batch scheduling state
    batch_is_full: bool              # Skip prefill check when True
    chunked_req: Optional[Req]       # Currently chunked request
    decoding_reqs: List[Req]         # Decode reqs in mixed chunked prefill

    # Prefill metadata
    prefix_lens: List[int]
    extend_lens: List[int]
    extend_num_tokens: int

    # Sampling
    sampling_info: SamplingBatchInfo
    spec_info: Optional[SpecInput]   # Speculative decoding info
```

### 2.3 Data Flow

```
Request arrives
  -> handle_generate_request() -> Req object
  -> _add_request_to_queue()   -> waiting_queue

Scheduler loop:
  waiting_queue  --[PrefillAdder]--> can_run_list -> ScheduleBatch.init_new()
                                                         |
                                                    prepare_for_extend()
                                                         |
                                                    run_batch() -> ForwardBatch
                                                         |
                                              process_batch_result()
                                                         |
                                              finished? -> release_kv_cache()
                                              not finished? -> running_batch (decode)
                                                         |
                                              merge into running_batch
```

## 3. Core Scheduling Logic

### 3.1 get_next_batch_to_run() -- The Master Orchestrator

This is the central scheduling function (~130 lines):

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # 1. Handle chunked requests from last iteration
    if self.chunked_req is not None:
        if self._chunked_req_scheduled_last_iter:
            self.stash_chunked_request(self.chunked_req)  # Cache unfinished KV

    # 2. Merge last prefill batch into running batch
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.last_batch.filter_batch(chunked_req_to_exclude=...)
        if not self.last_batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge_batch(self.last_batch)

    # 3. Try to schedule new prefill
    new_batch = self.get_new_batch_prefill()

    # 4. DP attention synchronization
    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)

    # 5. Return prefill batch if available, else run decode
    if new_batch is not None:
        ret = new_batch
    elif not self.running_batch.is_empty():
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch
    else:
        ret = None

    return ret
```

Key insight: SGLang uses a **merge-based** approach where prefill batches merge into the running decode batch on the next iteration, rather than maintaining separate prefill and decode phases.

### 3.2 get_new_batch_prefill() -> _get_new_batch_prefill_raw()

The actual prefill scheduling logic:

```python
def _get_new_batch_prefill_raw(self, ...):
    # 1. Early exit checks
    if self.running_batch.batch_is_full and self.chunked_req is None:
        return None

    # 2. Priority scheduling
    self.policy.calc_priority(self.waiting_queue, self.running_batch)

    # 3. Create PrefillAdder with token budget
    adder = PrefillAdder(
        page_size, tree_cache, token_to_kv_pool_allocator,
        running_batch, new_token_ratio, max_prefill_tokens,
        chunked_prefill_size, ...
    )

    # 4. Handle ongoing chunked request first
    if self.chunked_req is not None:
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    # 5. Iterate waiting queue, try to add each request
    for req in self.waiting_queue:
        req.init_next_round_input(self.tree_cache)  # Prefix match
        res = adder.add_one_req(req, ...)
        if res != AddReqResult.CONTINUE:
            break

    # 6. Create batch from can_run_list
    new_batch = ScheduleBatch.init_new(can_run_list, ...)
    new_batch.prepare_for_extend()

    # 7. Mixed chunked prefill: merge decode + prefill
    if self.is_mixed_chunk and not self.running_batch.is_empty():
        self.running_batch.prepare_for_decode()
        new_batch.mix_with_running(self.running_batch)

    return new_batch
```

### 3.3 init_next_round_input() -- Prefix Matching per Request

Every time a request is considered for scheduling, it calls `init_next_round_input`:

```python
def init_next_round_input(self, tree_cache=None):
    self.fill_ids = self.origin_input_ids + self.output_ids
    token_ids_to_match = self.fill_ids[:self._compute_max_prefix_len(input_len)]

    # RadixCache prefix match
    match_result = tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=token_ids_to_match, extra_key=self.extra_key))
    )
    self.prefix_indices = match_result.device_indices
    self.last_node = match_result.last_device_node
    self.host_hit_length = match_result.host_hit_length

    # extend_input_len = total_tokens - cached_tokens
    self.set_extend_input_len(len(self.fill_ids) - len(self.prefix_indices))
```

This means every scheduling decision starts with a **fresh prefix match** against the RadixCache.

## 4. PrefillAdder -- Token Budget Management

### 4.1 Budget Tracking

PrefillAdder maintains a multi-layered budget:

```python
class PrefillAdder:
    # Token budgets
    rem_input_tokens: int         # Remaining from max_prefill_tokens
    rem_chunk_tokens: Optional[int]  # Remaining from chunked_prefill_size
    rem_total_tokens: property    # available_size + evictable_size - offset
    cur_rem_tokens: property      # Current-iteration available (no new_token offset)
    rem_swa_tokens: property      # Hybrid SWA pool budget

    # Running lists
    can_run_list: List[Req]       # Requests that can be scheduled
    preempt_list: List[Req]       # Requests to preempt (priority scheduling)
    new_chunked_req: Optional[Req]  # Request that got chunked
```

### 4.2 Budget Check Logic (add_one_req)

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size):
    # 1. PrefillDelayer negotiation (adaptive prefill delay)
    # 2. Max prefill requests limit
    # 3. Total token budget check
    total_tokens = req.extend_input_len + max_new + page_size
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # 4. SWA budget check (hybrid models)
    # 5. max_prefill_tokens check (non-chunked mode)

    # 6. Lock the prefix node (prevents eviction during scheduling)
    with self._lock_node(req.last_node):
        # 7. Host cache load-back (HiCache)
        if req.host_hit_length > 0:
            new_indices = self.tree_cache.init_load_back(...)

        # 8. Chunked prefill truncation
        if input_tokens > self.rem_chunk_tokens:
            trunc_len = self.rem_chunk_tokens // page_size * page_size
            req.set_extend_input_len(trunc_len)
            self.new_chunked_req = req

        # 9. Add to can_run_list and update budget
        self.can_run_list.append(req)
        self._update_prefill_budget(...)
```

### 4.3 Return Values

```python
class AddReqResult(Enum):
    CONTINUE = auto()   # Continue adding requests
    NO_TOKEN = auto()   # No KV cache tokens left
    OTHER = auto()      # Other reasons (prefill token limit, chunk limit)
```

## 5. Cache-Aware Scheduling Policy

### 5.1 SchedulePolicy

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"             # Longest Prefix Match (default)
    DFS_WEIGHT = "dfs-weight"

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"           # First Come First Serve
    LOF = "lof"             # Longest Output First
    RANDOM = "random"
    ROUTING_KEY = "routing-key"
```

**LPM auto-fallback**: When `len(waiting_queue) > 128`, LPM falls back to FCFS to avoid expensive prefix matching overhead.

### 5.2 LPM (Longest Prefix Match) -- The Default Policy

```python
def _compute_prefix_matches(self, waiting_queue, policy):
    temporary_deprioritized: Set[int] = set()
    self.waiting_queue_radix_tree.reset()  # Simulated radix tree

    for r in waiting_queue:
        prefix_ids = r.origin_input_ids + r.output_ids
        match_prefix_for_req(self.tree_cache, r, prefix_ids)

        # In-batch prefix caching optimization:
        if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:  # 32
            match_result = self.waiting_queue_radix_tree.match_prefix(...)
            in_batch_matching_prefixes = match_result.device_indices

            if len(in_batch_matching_prefixes) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:  # 32
                temporary_deprioritized.add(r.rid)
            else:
                # Insert into simulated tree for future matches
                self.waiting_queue_radix_tree.insert(...)
```

**In-batch prefix caching** is a key optimization:
1. For each request with a short cache hit (< 32 tokens), check against a simulated radix tree of the waiting queue
2. If multiple waiting requests share the same prefix, deprioritize all but one
3. This ensures one request populates the cache, then subsequent requests get cache hits

### 5.3 DFS-Weight Policy

```python
def _sort_by_dfs_weight(waiting_queue, tree_cache):
    # Build last_node -> [reqs] mapping
    # Compute weight per node = number of descendant requests
    # DFS traversal: visit children in weight-descending order
    # Result: requests sharing cache paths are grouped together
```

This maximizes cache locality by scheduling requests that share the same tree branches together.

## 6. Chunked Prefill Handling

### 6.1 Mechanism

When a request's prefill exceeds `chunked_prefill_size`:
1. PrefillAdder truncates `extend_input_len` to `rem_chunk_tokens`
2. Sets `self.new_chunked_req = req`
3. Only processes that chunk in the current iteration
4. On the next iteration, `self.chunked_req` continues from where it left off

```
Request with 8192 tokens, chunked_prefill_size=4096:

Iteration 1: prefill tokens [0, 4096)   -> chunked_req set
Iteration 2: prefill tokens [4096, 8192) -> chunked_req cleared, req enters decode
```

### 6.2 Mixed Chunked Prefill

When `enable_mixed_chunk=True`:
```python
# In _get_new_batch_prefill_raw:
if self.is_mixed_chunk and not self.running_batch.is_empty():
    self.running_batch.prepare_for_decode()
    new_batch.mix_with_running(self.running_batch)  # Merge decode into prefill batch
    new_batch.forward_mode = ForwardMode.MIXED
```

This batches decode tokens with the prefill batch, improving GPU utilization. Decode tokens only need 1 token each, while prefill tokens fill the remaining budget.

### 6.3 Stash/Unstash

After a chunked prefill iteration:
```python
def stash_chunked_request(self, req):
    maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
```
The KV cache for the partially-prefilled request is inserted into the RadixCache so other requests can potentially share it.

## 7. Overlap Optimization

### 7.1 Dual CUDA Stream Architecture

```
schedule_stream:  recv -> process_input -> get_next_batch -> prepare tensors
forward_stream:   resolve_forward_inputs -> model forward -> sampling -> stash

Timeline:
  schedule_stream: |--- recv+schedule(N+1) ---|--- wait ---|--- schedule(N+2) ---|
  forward_stream:  |--- forward(N) ---|--- resolve+forward(N+1) ---|
```

### 7.2 FutureMap

FutureMap is the core relay mechanism for decode tokens:
1. After decode forward, `future_map.stash(req_pool_indices, next_token_ids)` stores the next token per request
2. On the next iteration, `resolve_forward_inputs(batch, future_map)` gathers stashed tokens into `batch.input_ids`
3. This avoids CPU staging of decode tokens between iterations

### 7.3 Overlap Isolation

```python
def _overlap_forward_isolation(self, batch):
    # Snapshot all batch fields to prevent GPU tensor GC during forward
    attr_snapshot = [getattr(batch, f.name) for f in dataclasses.fields(batch)]
    self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]
```

The `batch_record_buf` ring buffer (size 2) keeps references to GPU tensors from the previous iteration, preventing premature garbage collection by PyTorch.

## 8. Load Balancing Across DP Workers

### 8.1 SchedulerDPAttnAdapter

When DP attention is enabled (`enable_dp_attention`), all DP ranks must synchronize their batch sizes for the MLP AllGather:

```python
def prepare_mlp_sync_batch(self, local_batch):
    mlp_sync_info = MLPSyncBatchInfo(
        num_tokens=local_batch.batch_size(),  # or extend_num_tokens
        can_cuda_graph=...,
        is_extend_in_batch=...,
    )
    mlp_sync_info.all_gather(device, group)  # AllGather across DP ranks
    # global_num_tokens tells each rank how many tokens others have
    # Ensures padding matches for AllGather
```

Key fields synchronized:
- `global_num_tokens`: Token count per DP rank (for padding)
- `can_cuda_graph`: Whether all ranks can use CUDA graph
- `is_extend_in_batch`: Whether any rank has prefill (affects overlap decision)
- `tbo_split_seq_index`: Two-batch-overlap sequence split point

### 8.2 PrefillDelayer

The PrefillDelayer is an adaptive mechanism that delays prefill to accumulate more requests:
```python
class PrefillDelayer:
    # Delays prefill when:
    # - Token usage is below low watermark
    # - Few requests in waiting queue
    # - Haven't exceeded max_delay_passes
    def negotiate_should_allow_prefill(self, local_prefillable, ...):
        ...
```

This trades off TTFT for throughput by batching more prefills together.

## 9. Memory Management Integration

### 9.1 KV Cache Allocation Flow

```
Prefill (prepare_for_extend):
  alloc_for_extend(batch)
    -> req_to_token_pool.alloc(num_reqs)           # Request slots
    -> token_to_kv_pool_allocator.alloc_extend(...) # KV slots
    -> Write out_cache_loc mapping

Decode (prepare_for_decode):
  alloc_for_decode(batch)
    -> token_to_kv_pool_allocator.alloc_decode(...) # 1 slot per request
    -> Update out_cache_loc
```

### 9.2 Retraction (Preemption)

When decode runs out of memory:
```python
def retract_decode(self, server_args):
    # Sort by (output_ids length DESC, origin_input_ids length ASC)
    # Retract longest-output requests first (they use the most KV cache)
    while not self.check_decode_mem():
        idx = sorted_indices.pop()
        req = self.reqs[idx]
        retracted_reqs.append(req)
        self.release_req(idx, ...)
    # Adjust new_token_ratio to be more conservative
    new_estimate_ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(...)
```

### 9.3 RadixCache Integration

```python
# After prefill, insert KV cache into tree:
maybe_cache_unfinished_req(req, tree_cache)

# After request finishes, release:
release_kv_cache(req, tree_cache)
  -> tree_cache.dec_lock_ref(req.last_node)
  -> tree_cache.evict(...)  # May trigger eviction

# During scheduling, prefix match:
tree_cache.match_prefix(key=RadixKey(token_ids, extra_key))
  -> Returns matched prefix_indices (device KV slots)
```

## 10. Key Differences from vLLM V1 Scheduler

| Aspect | SGLang | vLLM V1 |
|--------|--------|---------|
| **Architecture** | Single-file scheduler + Mixin composition (~4034 lines) | scheduler.py (~2400 lines), separate KV cache manager |
| **Cache structure** | RadixTree (node splitting, no block alignment) | BlockPool (fixed blocks, FreeKVCacheBlockQueue linked list) |
| **Prefix matching** | Every scheduling iteration, per-request `match_prefix` | BlockHashToBlockMap, 1:N mapping, LRU eviction |
| **Scheduling policy** | LPM / DFS-Weight / FCFS / LOF / Random / Routing-Key | FCFS / PRIORITY (simpler) |
| **In-batch prefix detection** | Yes -- simulated radix tree over waiting queue, deprioritize duplicates | No equivalent |
| **Chunked prefill** | `chunked_prefill_size` + `is_mixed_chunk` (merge decode into prefill batch) | Similar but no mixed-mode equivalent |
| **Overlap** | Dual CUDA stream + FutureMap relay (schedule_stream / forward_stream) | No equivalent in scheduler (scheduler is CPU-only) |
| **Result processing** | BatchResultProcessor (frozen dataclass, decomposed) | Inline in scheduler |
| **DP coordination** | SchedulerDPAttnAdapter with AllGather for DP attention | DPEngineCoreProc (Wave coordination, 32-step AllReduce) |
| **Priority preemption** | `preempt_to_schedule()` -- preempt lower-priority running requests | Preemption via recompute/swap in scheduler |
| **Retraction policy** | Sort by output length (retract longest first) | Recompute or swap (configurable) |
| **Memory pools** | req_to_token_pool + token_to_kv_pool_allocator (separate) | BlockPool (unified) + BlockSpaceManager |
| **New token estimation** | NewTokenRatioTracker (adaptive, decays over time) | Fixed ratio estimation |
| **Component decomposition** | 18 sub-components under scheduler_components/ | Mostly monolithic scheduler.py |
| **Disaggregation** | Built-in P/D Mixins + NIXL/FlexKV/Mooncake connectors | NIXL connector, separate P/D code path |
| **Diffusion LLM** | Built-in SchedulerDllmMixin | Not supported |

### Key SGLang Innovations over vLLM:

1. **RadixAttention**: Node-splitting eliminates block alignment waste. vLLM requires block-aligned prefixes (wasting up to `block_size - 1` tokens per request), while SGLang's RadixTree can split at any token boundary.

2. **In-batch prefix caching**: Detects shared prefixes within the waiting queue and deprioritizes duplicates, ensuring the first request populates the cache. vLLM has no equivalent.

3. **CPU/GPU overlap scheduling**: The dual-stream architecture with FutureMap relay is unique to SGLang. vLLM's scheduler is purely CPU-side and doesn't overlap with GPU computation.

4. **DFS-Weight policy**: Tree-aware scheduling that groups requests by shared cache paths, maximizing prefix reuse. vLLM's scheduling is cache-agnostic (FCFS only).

5. **Mixed chunked prefill**: Batches decode tokens with prefill chunks, improving GPU utilization during chunked prefill. vLLM separates prefill and decode strictly.

6. **PrefillDelayer**: Adaptive prefill delay mechanism that trades TTFT for throughput by accumulating more prefill requests before scheduling them.

## 11. Request Lifecycle State Machine

```
   [Arrival]
       |
       v
  handle_generate_request()
       |
       v
  _add_request_to_queue()
       |
       v
  WAITING_QUEUE  <-------+
       |                  |
       v                  | (retraction)
  init_next_round_input() |
  PrefillAdder.add_one_req|  (prefix match)
       |                  |
       v                  |
  can_run_list            |
       |                  |
       v                  |
  ScheduleBatch.init_new()|
  prepare_for_extend()    |
       |                  |
       v                  |
  run_batch() [EXTEND]    |
       |                  |
       v                  |
  process_batch_result()  |
       |                  |
       v                  |
  merge_batch() to running_batch
       |
       v
  RUNNING_BATCH (decode)
       |        ^
       v        | (retraction)
  run_batch() [DECODE]  --+
       |
       v
  process_batch_result_decode()
       |
       v
  finished? --Yes--> release_kv_cache() + stream_output()
       |
      No
       |
       v
  (stay in running_batch for next decode)
```

## 12. Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schedule_policy` | `"lpm"` | Scheduling policy |
| `chunked_prefill_size` | `None` | Max tokens per prefill chunk |
| `enable_mixed_chunk` | `False` | Mix decode + prefill in same batch |
| `disable_overlap_schedule` | `False` | Disable CPU/GPU overlap |
| `enable_dp_attention` | `False` | Data parallel attention |
| `enable_priority_scheduling` | `False` | Priority-based scheduling |
| `disable_priority_preemption` | `False` | Disable preemption for priority |
| `max_running_requests` | model-dependent | Max concurrent requests |
| `max_prefill_tokens` | model-dependent | Max prefill tokens per batch |
| `page_size` | `1` | KV cache page size |
| `enable_hierarchical_cache` | `False` | HiCache (host + storage tiers) |
| `enable_prefill_delayer` | `False` | Adaptive prefill delay |

## 13. Key Insights

1. **Scheduling is prefix-match-heavy**: Every `get_new_batch_prefill` call triggers `init_next_round_input` which does a full RadixCache match for each request. This is the dominant CPU cost. The LPM fallback to FCFS at queue_size > 128 is a pragmatic optimization.

2. **Lock-based cache protection**: `_lock_node(req.last_node)` during add_one_req prevents the RadixCache from evicting a request's matched prefix while the scheduler is building the batch. Without this, a concurrent eviction could invalidate prefix_indices.

3. **Merge, don't replace**: SGLang never replaces the running_batch. Prefill batches merge into it. This is different from vLLM where the scheduler produces a fresh batch each step.

4. **Adaptive token ratio**: `NewTokenRatioTracker` estimates how many decode tokens each request will generate. After retraction, it becomes more conservative. This ratio directly affects prefill admission via `_get_running_request_total_token_offset`.

5. **batch_is_full optimization**: Once the running batch is full, `batch_is_full=True` skips the entire prefill path on subsequent iterations. It's only reset when (a) requests finish, (b) retraction happens, or (c) priority preemption.

6. **Two-phase finish**: A request finishes in two steps: `update_finish_state()` sets `finished_reason`, then the next iteration's `filter_batch()` removes it. This means `running_batch.reqs` can contain finished requests temporarily.

7. **War barrier necessity**: In overlap mode, `schedule_stream.wait_stream(forward_stream)` at the top of each iteration ensures the scheduler's writes to GPU buffers don't race with the previous iteration's GPU reads.

8. **Component extraction pattern**: SGLang uses frozen dataclasses with `slots=True` for sub-components (e.g., `SchedulerBatchResultProcessor`, `SchedulerDPAttnAdapter`), capturing only the needed references from the scheduler. This is cleaner than vLLM's approach of passing the entire scheduler around.
