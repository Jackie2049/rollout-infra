# SGLang Overlap Event Loop Architecture: Deep Reading

**Date**: 2026-07-15
**Focus**: Overlap event_loop_overlap(), FutureMap zero-copy relay, 6 event loop variants, WAR barrier, vLLM UBatch/DBO comparison, Multi-LoRA RL (#31253), RTX 4090 implications
**Source files**: `scheduler.py` (4456 LOC), `overlap_utils.py` (534 LOC), `schedule_batch.py` (3106 LOC), `multiplexing_mixin.py`, `scheduler_pp_mixin.py`, `disaggregation/prefill.py`, `disaggregation/decode.py`, `lora_overlap_loader.py`

---

## 1. Architecture Overview: The Overlap Event Loop

### 1.1 Core Principle

SGLang's overlap event loop is a **single-process, three-stream** GPU-CPU parallelism mechanism that achieves 20-40% throughput improvement by overlapping:

- **CPU scheduling** (batch preparation, request processing) on `schedule_stream`
- **GPU forward computation** on `forward_stream`
- **GPU D2H result copy** on `copy_stream`

The key insight: while the GPU is running forward on batch N, the CPU can simultaneously schedule batch N+1 and process the results of batch N-1. This eliminates the CPU-GPU serialization bottleneck in the normal event loop.

### 1.2 The Normal Event Loop (Baseline)

```
event_loop_normal():
  while True:
    recv_reqs -> process_input_requests
    plan = get_next_batch_to_run
    batch = plan.batch_to_run
    if batch:
      result = run_batch(batch)          # GPU forward (blocks CPU)
      process_batch_result(batch, result) # CPU post-processing
    else:
      on_idle()
    last_batch = batch
```

In the normal loop, `run_batch` blocks until GPU forward completes, then `process_batch_result` runs on CPU. Total iteration time = GPU_time + CPU_time (serialized).

### 1.3 The Overlap Event Loop

```
event_loop_overlap():
  result_queue = deque()  # holds (batch_copy, batch_result)

  while True:
    recv_reqs -> process_input_requests
    _apply_war_barrier()                   # WAR: wait for prev forward read
    plan = get_next_batch_to_run
    batch = plan.batch_to_run
    disable_overlap = is_disable_overlap_for_batch(batch, last_batch)

    if disable_overlap:
      pop_and_process()                    # process prev result NOW (sync boundary)

    if batch:
      batch_result = run_batch(batch)      # GPU forward on forward_stream
      result_queue.append((batch.copy(), batch_result))  # defer processing
    else:
      batch_result = None

    if last_batch and not disable_overlap:
      pop_and_process()                    # overlap: CPU processes prev while GPU runs cur

    if batch_result is generation:
      launch_batch_sample_if_needed(batch_result, batch)  # delayed sample

    last_batch = batch
```

**The critical overlap pattern**: When overlap is enabled, `pop_and_process()` (processing batch N-1's results on CPU) runs **concurrently** with `run_batch()` (forwarding batch N on GPU). The result_queue provides the one-iteration lag needed for this pipelining.

### 1.4 Data Flow Diagram

```
Iteration N:
┌─────────────────────────────────────────────────────────────────┐
│ schedule_stream (CPU + GPU schedule ops)                        │
│   recv_requests → process_input_requests                        │
│   _apply_war_barrier (wait forward_stream read-done)            │
│   get_next_batch_to_run → build ScheduleBatch                  │
│   resolve_forward_inputs (gather from FutureMap)                │
│   → forward_stream.wait_stream(schedule_stream)                │
│   [schedule ops complete, schedule_stream goes idle]            │
├─────────────────────────────────────────────────────────────────┤
│ forward_stream (GPU computation)                                │
│   forward_stream.wait_stream(schedule_stream) [dependency]      │
│   resolve_forward_inputs (prefill H2D, decode gather)           │
│   forward_batch_generation (the actual GPU forward pass)        │
│   future_map.publish(future_indices, seq_lens+1)               │
│   _relay_forward_payload (stash for next iter)                  │
│   copy_stream.wait_stream(forward_stream) [D2H dependency]      │
│   forward_done.record() [event for unified memory]              │
├─────────────────────────────────────────────────────────────────┤
│ copy_stream (GPU D2H copy)                                      │
│   copy_stream.wait_stream(forward_stream)                       │
│   batch_result.copy_to_cpu() (logprobs, hidden states D2H)      │
│   copy_done.record()                                             │
├─────────────────────────────────────────────────────────────────┤
│ schedule_stream (CPU post-processing of PREVIOUS batch N-1)     │
│   pop_and_process() → process_batch_result(batch_N-1, result)  │
│   [this overlaps with forward_stream running batch N]           │
└─────────────────────────────────────────────────────────────────┤
│ Delayed sample (on forward_stream, AFTER process_batch_result) │
│   launch_batch_sample_if_needed (speculative decode sampling)   │
│   _relay_forward_payload (relay sampled tokens to FutureMap)    │
└─────────────────────────────────────────────────────────────────┤
```

---

## 2. Three CUDA Streams

### 2.1 Stream Architecture

| Stream | Purpose | Priority | Created Where |
|--------|---------|----------|---------------|
| `schedule_stream` | CPU scheduling, WAR barrier, batch preparation, FutureMap resolve | 0 (default) | `run_event_loop()` |
| `forward_stream` | GPU forward computation, sampling, FutureMap publish | default | `model_runner.__init__` line 321 |
| `copy_stream` | D2H result copy (logprobs, hidden states) | default | `init_overlap()` line 1287 |

**Key property**: `schedule_stream` has priority=0. All three streams run on the same GPU, exploiting CUDA's stream-level concurrency for SM partitioning and async execution.

### 2.2 Stream Ordering Protocol

The overlap loop enforces a strict ordering protocol:

1. **schedule_stream → forward_stream**: `forward_stream.wait_stream(schedule_stream)` before forward entry. This ensures the schedule has finished preparing input_ids, seq_lens, and other tensors before the GPU reads them.

2. **forward_stream → copy_stream**: `copy_stream.wait_stream(forward_stream)` before D2H copy. This ensures forward computation is complete before copying results to CPU.

3. **forward_stream → schedule_stream** (WAR): `_apply_war_barrier()` at the start of each iteration. This prevents the scheduler from overwriting shared buffers (FutureMap output_tokens_buf, new_seq_lens_buf) while the previous forward is still reading them.

4. **schedule_stream → forward_stream** (for delayed sample): `forward_stream.wait_stream(schedule_stream)` before sampling. This ensures the process_batch_result updates (grammar, etc.) are visible before sampling the current batch.

### 2.3 StreamContext Pattern

```python
self.forward_stream_ctx = self.device_module.stream(self.forward_stream)
self.copy_stream_ctx = self.device_module.stream(self.copy_stream)
```

These are used as `with self.forward_stream_ctx:` blocks to temporarily switch the current CUDA stream. The pattern is used throughout `run_batch()` to ensure operations execute on the correct stream.

---

## 3. WAR (Write-After-Read) Barrier

### 3.1 The Problem

In the overlap loop, the schedule_stream writes shared buffers (FutureMap's `output_tokens_buf`, `new_seq_lens_buf`) at the start of iteration N+1, while the forward_stream is still reading those buffers during the forward pass of iteration N. Without a barrier, the schedule_stream could overwrite data that the GPU hasn't finished reading.

### 3.2 The Solution: Two-Phase WAR Barrier

```python
def _apply_war_barrier(self):
    if not self._war_barrier_enabled:
        return
    runner = self.model_worker.war_fastpath_runner
    ev = runner.war_fastpath_read_done_event
    if ev is not None:
        self.schedule_stream.wait_event(ev)    # FAST: wait on specific read-done event
        runner.war_fastpath_read_done_event = None
    else:
        self.schedule_stream.wait_stream(self.forward_stream)  # FALLBACK: wait entire forward
```

**Fast path**: The forward runner records a `war_fastpath_read_done_event` after the snapshot point (when it has finished reading all shared buffers). The scheduler waits only on this specific event, minimizing the delay.

**Fallback path**: If no specific event is available (e.g., decode without spec), the scheduler waits for the entire forward_stream to complete, which is more conservative but still correct.

**Enablement**: On CUDA, the WAR barrier is always enabled (`_war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()`). On other platforms (ROCm, NPU), it can be force-enabled via `SGLANG_ENABLE_WAR_BARRIER`.

### 3.3 WAR Barrier and DSV4 Draft-Extend (#31270)

PR #31270 reveals a WAR ordering bug specific to DSV4 CUDA graph replay:

- **Problem**: The runner published `war_fastpath_read_done_event` **before** CUDA graph replay, but DSV4's captured graph reads shared buffer state (out_cache_loc mapping) **during** replay. So the scheduler could overwrite the mapping while replay was still reading it.
- **Fix**: Defer `war_fastpath_read_done_event` until **after** replay for DSV4; other backends retain the pre-replay fast path. This is the 13th DSV4 failure, same State Lifecycle Mismatch pattern family.

---

## 4. FutureMap: Zero-Copy Cross-Iteration Relay

### 4.1 Design Philosophy

FutureMap is an "always-on pool-indexed relay for cross-iter values." It serves as the bridge between consecutive iterations, allowing iteration N's forward outputs to be consumed by iteration N+1's schedule without CPU round-tripping.

**Key property**: FutureMap is active in **both** normal and overlap modes. In normal mode, it relays decode input_ids; in overlap mode, it additionally relays seq_lens, spec extras, and confidence data.

### 4.2 Pool-Indexed Buffers

FutureMap maintains GPU-resident buffers indexed by `req_pool_indices` (the position of each request in the `req_to_token_pool`):

```python
class FutureMap:
    output_tokens_buf: torch.Tensor    # shape (req_pool_size,), dtype int64
    new_seq_lens_buf: torch.Tensor     # shape (req_pool_size,), dtype int64
    topk_p_buf: torch.Tensor           # spec_v2 top-k probabilities
    topk_index_buf: torch.Tensor       # spec_v2 top-k indices
    hidden_states_buf: torch.Tensor    # spec_v2 hidden states
    draft_probs_buf: torch.Tensor      # spec_v2 draft probabilities
    dsa_topk_indices_buf: torch.Tensor # DSA top-k indices
```

**Slot 0 is padding**: The KV pool's row 0 is reserved for padding (CUDA graph padded batches where `req_pool_idx == 0`). Writes to slot 0 are harmless reads; the zero-copy relay naturally handles CUDA graph padding.

### 4.3 The Publish/Stash/Resolve Cycle

**Publish** (end of forward, on forward_stream):
```python
def publish(self, future_indices, new_seq_lens, confidence=None):
    self.new_seq_lens_buf[indices] = new_seq_lens
    # For spec_v2: record event for D2H gating
    self.publish_ready.record()  # gates fwd_prepare_d2h_stream
    # For confidence relay: scatter + ring copy
```

**Stash** (after forward, relay sampled tokens):
```python
def stash(self, future_indices, payload: RelayPayload):
    self.output_tokens_buf[indices] = payload.bonus_tokens
    if self.need_topk:
        self.topk_p_buf[indices] = payload.topk_p
        self.topk_index_buf[indices] = payload.topk_index
```

**Resolve** (start of next iteration, on schedule_stream):
```python
# resolve_forward_inputs():
#   Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu)
#   Decode: gather from output_tokens_buf[req_pool_indices]
#   Mixed: torch.cat([prefill_gpu, decode_gpu])

# resolve_seq_lens_cpu():
#   spec_v2: gather from new_seq_lens_buf[future_indices]
#   Then D2H copy via fwd_prepare_d2h_stream (overlaps with forward)
```

### 4.4 Zero-Copy Mechanism

The relay is **zero-copy on GPU**:
- `output_tokens_buf[indices]` uses scatter/gather indexing into a persistent GPU tensor
- No allocation, no memcpy between iterations
- The same buffer is written by forward_stream (stash/publish) and read by schedule_stream (resolve)
- The WAR barrier ensures write-after-read ordering between streams

**D2H is optimized**: `new_seq_lens_cpu_pinned` is a pinned CPU buffer. A private `fwd_prepare_d2h_stream` copies from GPU to pinned CPU, gated by `publish_ready` event. This allows the D2H copy to overlap with the next forward pass instead of blocking it.

### 4.5 RelayPayload

```python
@dataclass
class RelayPayload:
    bonus_tokens: torch.Tensor           # sampled token IDs (non-spec) or bonus tokens (spec)
    topk_p: Optional[torch.Tensor]       # spec_v2 top-k probabilities
    topk_index: Optional[torch.Tensor]   # spec_v2 top-k indices
    hidden_states: Optional[torch.Tensor] # spec_v2 hidden states
    draft_probs: Optional[torch.Tensor]  # spec_v2 draft probabilities
    dsa_topk_indices: Optional[torch.Tensor] # DSA seed metadata
```

Non-spec batches only fill `bonus_tokens`; spec extras are gated by `spec_algo`, not payload shape. This means a non-spec stash allocates no extra FutureMap bufs.

### 4.6 ConfidenceRelay (DSpark / DSA)

For speculative decoding with confidence-based budget (DSpark algorithm), FutureMap includes a `ConfidenceRelay` ring buffer:

```python
class ConfidenceRelay:
    conf_ring: torch.Tensor  # (depth, req_pool_size, gamma), pin_memory=True
    gen_ring: torch.Tensor   # (depth, req_pool_size), dtype int64
    copy_done: list[Event]   # per-ring-slot completion events
    ring_pos: int            # circular buffer position
```

- **Ring lag = 2**: The resolve reads from `ring_pos - CONFIDENCE_RELAY_RING_LAG`, ensuring the data was published 2 iterations ago (allowing overlap).
- **Ring depth = 3**: `CONFIDENCE_RELAY_RING_LAG + 1 = 3` slots in the ring buffer.
- **Async D2H**: The confidence ring uses `pin_memory=True` and a separate copy on `fwd_prepare_d2h_stream`, gated by `publish_ready`.

---

## 5. Forward Isolation (Batch Transaction Semantics)

### 5.1 The Problem

In the overlap loop, the ScheduleBatch object is shared between the schedule_stream (which builds it) and the forward_stream (which runs it). Spec V2 makes mid-forward mutations (rebinding `seq_lens`, `spec_info`, `input_ids`). Without isolation, these mutations would corrupt the batch state needed by the next schedule iteration.

### 5.2 _forward_isolation Context Manager

```python
@contextmanager
def _forward_isolation(self, batch, *, overlap: bool):
    # 1. Snapshot SB fields (full for spec_v2, sampling_info only for non-spec)
    sched_snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    sched_sampling_info = batch.sampling_info

    # 2. Substitute sampling_info with forward-only copy
    batch.sampling_info = sched_sampling_info.copy_for_forward()

    # 3. Pin for 2-iter tensor lifetime (overlap path only)
    if overlap:
        self.record_batch_in_overlap(batch)

    try:
        yield
    finally:
        # Restore: either full snapshot (spec_v2) or just sampling_info (non-spec)
        if snapshot_v2_full:
            for name, value in sched_snapshot.items():
                setattr(batch, name, value)
        else:
            batch.sampling_info = sched_sampling_info
```

**Transaction semantics**: The forward pass runs as a "transaction" on the ScheduleBatch. Any mutations during forward are automatically undone when the context exits. The original state is preserved for the next schedule iteration.

### 5.3 record_batch_in_overlap: 2-Iter Lifetime Pinning

```python
def record_batch_in_overlap(self, batch):
    attr_snapshot = [getattr(batch, f.name, None) for f in dataclasses.fields(batch)]
    self.batch_record_ct = (self.batch_record_ct + 1) % 2
    self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]
```

The overlap loop keeps **two** batch snapshots alive (alternating via `batch_record_ct % 2`). This ensures:
- GPU tensors in the snapshot survive the PyTorch caching allocator past the forward_stream
- Cross-stream tensor references are valid for 2 iterations (the maximum lag in the overlap pipeline)
- Workers can register additional refs via `GenerationBatchResult.extra_keep_alive_refs`

---

## 6. Disable Overlap Conditions

### 6.1 Consecutive Prefill Overlap

```python
def is_disable_overlap_for_batch(self, batch, last_batch):
    # Consecutive prefill batches: disable overlap for TTFT
    disable_overlap = (
        envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
        and batch_is_extend
        and last_batch_is_extend
    )

    # Spec + grammar: overlap unsupported, must sync
    need_grammar_sync = (
        batch and not batch.spec_algorithm.is_none()
        and batch.has_grammar
        and batch.forward_mode.is_decode()
        and len(self.result_queue) > 0
    )

    return disable_overlap or need_grammar_sync
```

**Why consecutive prefill overlap is disabled (by default)**:
- When two prefill batches overlap, the second prefill's TTFT includes the time waiting for the first prefill's forward to complete (since processing must happen before sampling)
- Disabling overlap for consecutive prefills improves TTFT of the first batch at the cost of slightly lower throughput
- `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` defaults to `False`, meaning overlap IS enabled by default for consecutive prefills (the env var name is "disable the overlap" = setting it True disables overlap)
- In DP attention mode, `is_extend_in_batch` is globally synchronized across DP ranks to ensure all ranks make the same overlap decision (avoiding deadlock)

**Opportunistic flush at sync boundary**: When overlap is disabled for a batch, the schedule_stream gets a "free" moment where forward_stream is idle (prev forward drained, next not launched). This is used for `_flush_opportunistic()` on the unified memory pool allocator -- compacting KV cache without a full synchronization.

### 6.2 Spec + Grammar Incompatibility

Spec V2 with grammar-constrained decoding requires the previous batch's grammar updates to be processed before the current batch can be scheduled. The result_queue lag would desynchronize the grammar state. Currently marked as `TODO(lsyin): support overlap + spec + grammar`.

---

## 7. The 6 Event Loop Variants

### 7.1 Dispatch Logic

```python
def dispatch_event_loop(scheduler):
    if disaggregation_mode == NULL:
        if enable_pdmux:       event_loop_pdmux()
        elif pp_size > 1:      event_loop_pp()
        elif enable_overlap_mlx: event_loop_overlap_mlx()
        elif enable_overlap:   event_loop_overlap()
        else:                  event_loop_normal()
    elif disaggregation_mode == PREFILL:
        if pp_size > 1:        event_loop_pp_disagg_prefill()
        elif enable_overlap:   event_loop_overlap_disagg_prefill()
        else:                  event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DECODE:
        if pp_size > 1:        event_loop_pp_disagg_decode()
        elif enable_overlap:   event_loop_overlap_disagg_decode()
        else:                  event_loop_normal_disagg_decode()
```

### 7.2 Variant Summary

| # | Variant | Use Case | Key Difference |
|---|---------|----------|----------------|
| 1 | `event_loop_normal` | Single GPU, no overlap | Serialized CPU-GPU, simplest |
| 2 | `event_loop_overlap` | Single GPU, overlap enabled | 3-stream pipelined, FutureMap relay |
| 3 | `event_loop_pdmux` | PD-multiplexing (prefill+decode concurrent) | SM partitioning, split-prefill across layers |
| 4 | `event_loop_pp` | Pipeline parallelism (pp_size > 1) | Multi-stage, async send/sync recv, micro-batching |
| 5 | `event_loop_overlap_disagg_prefill` | Disaggregated prefill server | KV transfer to decode server, staging buffer |
| 6 | `event_loop_overlap_disagg_decode` | Disaggregated decode server | KV receipt from prefill server, prebuilt batches |

**Common pattern across all overlap variants**: result_queue + 1-iter lag + WAR barrier. The disagg variants add KV transfer queues; PDMux adds SM partitioning and split-prefill layer-wise forwarding; PP adds inter-stage proxy tensor communication.

### 7.3 PDMux Deep Dive

PDMux (PD-Multiplexing) is the most complex variant:
- **SM partitioning**: The GPU's SMs are divided into prefill SMs and decode SMs. Different `stream_groups` allocate different SM ratios based on decode batch size.
- **Split-prefill**: Prefill is split across model layers (not tokens), forwarded in chunks (`split_forward_count` = token_budget / extend_num_tokens). Each chunk advances `split_index` forward by `forward_count` layers.
- **Concurrent execution**: Decode runs on `decode_stream`, prefill on `prefill_stream`, both using their respective SM allocation. They synchronize only at stream-group adjustment boundaries.
- **All-reduce coordination**: When all TP ranks finish their prefill chunks, an `allreduce` confirms completion before merging the split-prefill batch into the running decode batch.

### 7.4 MLX Overlap

On Apple MLX (M-series chips), overlap uses `mx.async_eval` instead of CUDA streams. The `SchedulerMlxOverlapMixin` provides `event_loop_overlap_mlx()` which mirrors the overlap pattern using MLX's async evaluation mechanism. FutureMap is still used for input_ids relay, but no CUDA-specific WAR barrier or stream contexts.

---

## 8. Batch Copy and Result Queue

### 8.1 Why batch.copy() is Needed

In the overlap loop, `result_queue.append((batch.copy(), batch_result))` creates a shallow snapshot of the ScheduleBatch. This is critical because:

1. The original `batch` object will be **mutated** by the next schedule iteration (rebuilding for the next forward)
2. `process_batch_result` needs the original `reqs`, `forward_mode`, `extend_lens`, `prefix_lens`, etc.
3. Without the copy, the overlap loop's concurrent schedule+process would corrupt the result processing

The copy is **shallow** -- it copies `reqs[:]`, `extend_lens[:]`, `prefix_lens[:]` (to protect against in-place mutations like `filter_batch` and `merge_batch`), but shares GPU tensor references (which are protected by `record_batch_in_overlap`'s 2-iter pinning).

### 8.2 result_queue Depth

The result_queue always holds at most **1** entry (deque with one push and one pop per iteration). This ensures:
- The overlap pipeline has exactly 1 iteration of lag
- No unbounded memory growth from queued results
- The WAR barrier only needs to protect 1-iter-old data

---

## 9. LoRA Overlap Loading

### 9.1 LoRAOverlapLoader Architecture

The `LoRAOverlapLoader` enables asynchronous LoRA adapter loading overlapped with forward computation:

```python
class LoRAOverlapLoader:
    load_stream: CudaStream                    # dedicated stream for LoRA weight loading
    load_stream_context: CudaStreamContext
    lora_to_overlap_load_event: Dict[str, CudaEvent]  # completion events per adapter
```

**Three states per adapter**:
- `NOT_LOADED`: Adapter not in memory pool, start async load if capacity
- `LOADING`: Adapter loading on `load_stream`, event pending
- `LOADED`: Adapter in memory pool, ready to use

**The overlap pattern**: `try_overlap_load_lora()` is called during schedule (on `schedule_stream`). It:
1. Drains completed loads (`_drain_completed_overlap_loads()`) by checking `event.query()` and waiting on `current_stream`
2. If the adapter is NOT_LOADED, validates memory capacity and starts loading on `load_stream`
3. Returns `True` only when LOADED (loading adapters return `False`, request waits)

**Stream ordering**: `current_stream.wait_event(event)` after drain ensures the load is visible on the schedule_stream before the adapter is used in forward.

### 9.2 GRPO Multi-LoRA Relevance

For GRPO training with multiple LoRA adapters (one per policy), the overlap loader enables:
- Adapter loading overlaps with forward computation
- New adapters can be loaded while the previous adapter's forward is still running on `forward_stream`
- The `load_stream` is independent of `forward_stream`, allowing true GPU concurrency

---

## 10. Comparison with vLLM UBatch/DBO

### 10.1 vLLM UBatch Architecture

vLLM's Dual Batch Overlap (DBO) / UBatch mechanism is a **multi-threaded** approach:

```
UBatchContext:
  compute_stream: torch.cuda.Stream   # model forward
  comm_stream: torch.cuda.Stream      # NCCL all2all communication
  gpu_compute_done_event: torch.Event # signals compute completion
  gpu_comm_done_event: torch.Event    # signals comm completion
  cpu_wait_event: threading.Event     # CPU synchronization between threads
  cpu_signal_event: threading.Event   # CPU signal to next thread
```

**Key differences from SGLang**:

| Aspect | SGLang Overlap | vLLM UBatch/DBO |
|--------|---------------|-----------------|
| **Concurrency model** | Single process, 3 CUDA streams | Multi-threaded (N+1 threads for N microbatches) |
| **Overlap target** | CPU schedule overlaps GPU forward | GPU compute overlaps GPU communication (NCCL all2all) |
| **Batch split** | No split; whole batch pipelined 1 iter | Split into microbatches, each runs in its own thread |
| **Primary benefit** | 20-40% throughput (CPU-GPU overlap) | EP/MoE comm-compute overlap, DP comm overlap |
| **SM control** | None (all streams use all SMs) | SM partitioning via DeepGEMM/DeepEP (comm_sms vs compute_sms) |
| **Synchronization** | CUDA events + stream waits + WAR barrier | threading.Event (CPU) + torch.Event (GPU) + Barrier |
| **Result relay** | FutureMap (pool-indexed GPU-resident) | Thread-local forward_context, no cross-iter relay |
| **Speculative decode** | Integrated via FutureMap publish/stash | Separate per-ubatch context, no cross-ubatch relay |
| **Tensor lifetime** | 2-iter pin via batch_record_buf | Thread join + CUDAGraph pool |

### 10.2 UBatchWrapper and CUDAGraph Integration

vLLM's `UBatchWrapper` wraps the model runnable with:
- `_capture_ubatches()`: Captures a CUDA graph across all microbatch threads simultaneously. Each thread runs its forward portion, results are concatenated.
- `_run_ubatches()`: Eager mode, threads run concurrently.
- SM partitioning via `SMControlContextManager`: During DBO execution, `comm_sms` SMs are reserved for NCCL all2all, `compute_sms = total_sms - comm_sms` for model forward.

### 10.3 Design Philosophy Differences

**SGLang**: "Overlap CPU scheduling with GPU forward" -- the bottleneck is CPU-GPU serialization. One batch, one-iter lag, zero-copy FutureMap relay. Simpler, more portable, works on any GPU.

**vLLM**: "Overlap GPU compute with GPU communication" -- the bottleneck is NCCL all2all in EP/MoE models. Micro-batching splits the batch into compute and comm phases that run on different streams with SM partitioning. More complex, requires DeepEP/DeepGEMM support, targets multi-GPU EP deployments.

**For RTX 4090 (single GPU, dp=1)**: SGLang's overlap is directly beneficial (CPU-GPU overlap on one GPU). vLLM's UBatch/DBO is less relevant (no NCCL all2all on single GPU, no EP).

---

## 11. Multi-LoRA RL Integration (#31253)

### 11.1 PR Overview

PR #31253 by yushengsu-thu (Miles team) combines two features for multi-LoRA async RL:

1. **Abort by rid prefix** (#30912): Retiring an adapter aborts its entire request namespace, including requests held in the tokenizer window during paused weight updates.
2. **LoRA upsert** (#30913): In-place adapter weight refresh on the `from_distributed` weight-sync path, with id reuse, staged rollback, and pinned accounting.

### 11.2 Abort-by-Rid-Prefix

In multi-LoRA RL, each policy has its own LoRA adapter with requests tagged by a rid (request ID) prefix (e.g., `policy_A_req_1`). When adapter A is retired:
- All requests with prefix `policy_A_*` are aborted, including those still in the tokenizer queue
- This prevents stale requests from an retired adapter from consuming GPU resources
- The abort propagates through the tokenizer's held-request window (a subtle edge case where requests are paused during weight updates)

### 11.3 LoRA Upsert

LoRA upsert enables **in-place weight refresh** without unloading/reloading:
- An adapter with the same `lora_id` can be updated with new weights
- The old weights are staged for rollback if the new load fails
- Pinned accounting ensures memory pool entries are correctly managed during the transition
- Scoped to `from_distributed` route (weight sync from trainer to rollout server)

### 11.4 GRPO Relevance

This PR is **critical for multi-policy GRPO** on RTX 4090:
- **verl's multi-LoRA rollout** uses similar abort-by-prefix semantics (via SGLang integration)
- **In-place weight refresh** eliminates the unload/reload cycle that causes KV cache invalidation
- **Staged rollback** prevents partial weight updates from corrupting inference
- The combination of abort + upsert enables a clean "swap adapter" workflow: abort old requests, upsert new weights, resume with new adapter

### 11.5 Files Changed

| File | Change | Relevance |
|------|--------|-----------|
| `lora_manager.py` | +121/-27: upsert logic, rollback, pinned accounting | Core weight refresh mechanism |
| `lora_registry.py` | +44/-2: upsert registry updates | Adapter lifecycle tracking |
| `tokenizer_control_mixin.py` | +42/-5: abort-by-prefix in tokenizer | Request namespace cleanup |
| `model_runner.py` | +17/-3: upsert integration in weight update path | Rollout engine integration |
| `test_lora_upsert.py` | +535: comprehensive upsert tests | Validation coverage |
| `test_abort_request_prefix.py` | +261: abort-by-prefix tests | Validation coverage |

---

## 12. DSV4 Draft-Extend WAR Ordering (#31270)

### 12.1 The Bug

DSV4 CUDA graph replay reads shared buffer state (out_cache_loc through the full-to-SWA mapping) **during** replay. But the WAR barrier's `war_fastpath_read_done_event` was recorded **before** replay, allowing the scheduler to overwrite the mapping while replay was still reading.

### 12.2 The Fix

- Add `AttentionBackend` capability flag for replay-time shared-buffer reads
- Defer `war_fastpath_read_done_event` until **after** replay for DSV4
- Other backends retain the pre-replay fast path
- Clamp DSV4 uniform-width causal lengths to 1 (padding metadata fix)

### 12.3 Pattern Family Connection

This is the **13th DSV4 failure**, belonging to the State Lifecycle Mismatch pattern family:
- Same root cause as #45552 (cumem sleep/wake sync), #8061 (overlap_comm data race), #5788 (FSDP use-after-free)
- All share: shared mutable state accessed across asynchronous boundaries without proper ordering
- The WAR barrier is the general solution; #31270 adds backend-specific replay ordering

---

## 13. No-Padding CUDA Graph Admission Fix (#31273)

### 13.1 The Bug

After the `ShapeKey` migration (#27857), the no-padding CUDA graph path was using the old raw batch-size key for graph lookup. Every lookup missed, falling back to eager mode, causing significant decode slowdown with `--disable-cuda-graph-padding`.

### 13.2 The Fix

8-line fix (+8/-6 across 4 files) that updates the graph lookup key to use `ShapeKey` instead of raw batch size. Standard and EAGLE decode both fixed.

### 13.3 GRPO Relevance

For GRPO with dynamic batch sizes, `--disable-cuda-graph-padding` is sometimes used to avoid padding overhead. This fix ensures the no-padding path actually uses CUDA graphs instead of silently falling back to eager mode.

---

## 14. RTX 4090 Implications

### 14.1 Overlap Event Loop Benefits

On RTX 4090 (single GPU, dp=1):
- **CPU-GPU overlap is the primary bottleneck**: With a single GPU, there's no NCCL communication to overlap. The main serialization is CPU scheduling + GPU forward.
- **SGLang's overlap loop directly addresses this**: 20-40% throughput improvement from overlapping CPU processing of batch N-1 with GPU forward of batch N.
- **FutureMap zero-copy relay avoids D2H round-trips**: Decode input_ids stay on GPU, eliminating ~0.1-0.5ms per iteration of D2H/H2D overhead.

### 14.2 Overlap + LoRA for GRPO

For multi-LoRA GRPO on RTX 4090:
- **LoRAOverlapLoader** enables adapter loading during forward (no serialization)
- **LoRA upsert** (#31253) enables in-place weight refresh without KV cache invalidation
- **The overlap loop + LoRA overlap loader** together provide: forward(batch_N, adapter_A) overlaps with load(adapter_B) and process_result(batch_N-1, adapter_C)
- This is the optimal pattern for GRPO's multi-policy rollout

### 14.3 WAR Barrier on RTX 4090

- CUDA is always enabled, so `_war_barrier_enabled = True` on RTX 4090
- The fast-path event is used when available (decode with CUDA graph)
- The fallback (wait_stream) is used for prefill batches
- No additional configuration needed

### 14.4 Concerns

- **Consecutive prefill overlap**: Default is ENABLED (SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=False). For GRPO with multiple prefill batches per rollout step, this could hurt TTFT. Consider setting True for GRPO.
- **Spec + grammar overlap disabled**: If GRPO uses structured output (JSON reward functions), overlap is disabled for spec+grammar batches, reverting to serialized mode.
- **Copy_stream D2H**: On RTX 4090, the D2H copy of logprobs on `copy_stream` overlaps with the next forward. For GRPO logprob collection, this is beneficial (no serialization).
- **Memory overhead**: batch_record_buf holds 2 snapshots of ScheduleBatch. For RTX 4090's 24 GiB, this is ~0.5-1 GiB per snapshot (depending on batch size), which is manageable.

---

## 15. Key Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` | False | Set True to disable overlap for consecutive prefill batches (improves TTFT) |
| `SGLANG_ENABLE_WAR_BARRIER` | False | Force-enable WAR barrier on non-CUDA platforms; always True on CUDA |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | False | Experimental: separate stream for overlap planning |
| `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` | False | Disable DSV4 draft-extend CUDA graph (workaround for #31270) |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | True | Enable multi-stream overlap for MoE experts |
| `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH` | False | Use NCCL all-gather in overlap scheduler sync batch |

---

## 16. Summary: Architectural Insights

1. **SGLang's overlap is fundamentally a CPU-GPU pipelining technique**, unlike vLLM's UBatch/DBO which is GPU-GPU (compute-comm) pipelining. This makes SGLang's overlap more broadly applicable, especially on single-GPU setups.

2. **FutureMap is the core innovation**: A pool-indexed, GPU-resident, zero-copy relay that bridges consecutive iterations without CPU round-trips. It handles decode tokens, seq_lens, spec extras, and confidence data with a unified scatter/gather interface.

3. **The WAR barrier is the correctness cornerstone**: Without it, the overlap loop would have write-after-read races on shared FutureMap buffers. The fast-path event minimizes the barrier's performance impact.

4. **Forward isolation provides transaction semantics**: The `_forward_isolation` context manager makes ScheduleBatch mutations transactional, ensuring the overlap loop's concurrent schedule+process doesn't corrupt batch state.

5. **The 6 event loop variants share the overlap pattern**: Normal, overlap, PDMux, PP, disagg-prefill, disagg-decode all use result_queue + 1-iter lag, with domain-specific extensions (KV transfer, SM partitioning, inter-stage communication).

6. **Multi-LoRA RL (#31253) is a GRPO enabler**: Abort-by-rid-prefix + LoRA upsert provides the clean adapter-swap workflow needed for multi-policy GRPO rollout, directly relevant to verl's SGLang integration path.

7. **DSV4 WAR ordering (#31270) validates the pattern family**: The 13th DSV4 failure is another instance of State Lifecycle Mismatch -- shared mutable state accessed across asynchronous boundaries without proper ordering. The WAR barrier is the general solution; backend-specific replay ordering is the targeted fix.
