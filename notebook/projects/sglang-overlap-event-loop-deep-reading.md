# SGLang Overlap Event Loop + FutureMap Deep Reading

**Date**: 2026-07-15 (Session 10 continued)
**Purpose**: Understand SGLang's overlap event loop and FutureMap for GPU-CPU parallelism in serving
**Sources**: SGLang scheduler.py source, sglang-nav skill, overlap_utils.py, PDMux reading

---

## 1. Overlap vs Normal Event Loop Architecture

```
SGLang has 2 event loop modes:

Normal event_loop():
  → Sequential: recv → schedule → forward → process → recv → schedule → ...
  → Simple, predictable, easy to debug
  → Throughput: limited by forward pass latency (~0.5-2ms per decode step)
  → GPU idle during CPU scheduling → ~20-40% throughput loss

Overlap event_loop_overlap():
  → Parallel: schedule on schedule_stream overlaps with forward on forward_stream
  → FutureMap: relay mechanism for zero-copy data sharing between streams
  → WAR barrier: ensures forward_stream completes before copy starts
  → Throughput: ~20-40% improvement over normal loop
  → More complex: requires careful stream management and synchronization

★★★★★★★★★ When overlap is used:
  - Default: overlap enabled for throughput
  - Disable: SGLANG_DISABLE_OVERLAP_LOOP or --disable-overlap
  - Disable consecutive prefill overlap: SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
    → When enabled: prefill batches processed back-to-back → lower TTFT
    → When disabled: each prefill gets full GPU → higher TTFT but better throughput
```

---

## 2. FutureMap: Zero-Copy Data Relay Between Streams

```
FutureMap is the core mechanism enabling overlap:

FutureMap(pool-indexed relay):
  → Publisher writes data to pool slot at index I on stream A
  → Consumer reads data from same pool slot at index I on stream B
  → Zero-copy: both streams share the same GPU memory location
  → No data duplication, no CPU-GPU round trip

Implementation (from overlap_utils.py):
  → FutureMap stores pending results indexed by pool position
  → When schedule_stream finishes scheduling → results stored in FutureMap
  → When forward_stream finishes forward → reads FutureMap for scheduled results
  → WAR barrier: forward_stream must complete before reading FutureMap for next step

★★★★★★★★ Key insight: FutureMap enables true GPU-CPU overlap
  - CPU scheduling (on schedule_stream) can prepare next batch WHILE GPU forwards current batch
  - No sequential bottleneck → GPU never idle while CPU works
  - FutureMap size = max_running_requests → bounded memory
```

---

## 3. WAR Barrier: Write-After-Read Synchronization

```
WAR (Write-After-Read) barrier ensures correct ordering:

Problem:
  Stream A writes to memory → Stream B reads from memory
  If B reads before A writes → stale data → wrong computation
  If A writes after B reads → race condition → corrupted results

WAR barrier solution:
  1. Stream A records CUDA event after writing
  2. Stream B waits on that CUDA event before reading
  3. Guarantees: A's write completes → event signals → B starts reading

In SGLang overlap loop:
  forward_stream: forward pass → record event → WAR barrier
  schedule_stream: wait on WAR barrier → schedule next batch → FutureMap write
  copy_stream: wait on WAR barrier → async copy → CPU offload

★★★★★★★★ This is EXACTLY the pattern needed for GRPO weight sync:
  Training stream: forward + backward → update model → record event → WAR barrier
  Rollout stream: wait on WAR barrier → generate responses with updated model
  Sleep/wake: WAR barrier between training phase and rollout phase
```

---

## 4. CUDA Stream Management in Overlap Loop

```
3 CUDA streams in overlap mode:

1. schedule_stream (CPU-bound):
   → recv_requests → process_input_requests → get_new_batch_prefill → schedule
   → Runs on CPU (mostly) with some GPU memory operations
   → Can overlap with forward_stream because it's mostly CPU work

2. forward_stream (GPU-bound):
   → forward_batch_generation → model forward → sampling → output
   → Runs entirely on GPU → 0.5-2ms per decode step
   → This is the main compute workload

3. copy_stream (GPU-CPU transfer):
   → async KV cache offload (HiCache) → CPU DRAM → SSD
   → Runs on separate CUDA stream → overlaps with forward
   → WAR barrier ensures forward completes before copy starts

Stream synchronization:
  schedule_stream → record_event → forward_stream wait_event → forward
  forward_stream → record_event → schedule_stream wait_event → schedule next
  forward_stream → record_event → copy_stream wait_event → offload

★★★★★★★★ RTX 4090 implications:
  3 streams = more GPU memory for stream contexts → minimal (<1 MiB)
  But: overlap only beneficial when CPU scheduling takes significant time
  For single-GPU RTX 4090: scheduling is fast → overlap benefit may be small
  For multi-GPU: scheduling includes NCCL operations → overlap very beneficial
```

---

## 5. Overlap Event Loop Step-by-Step Trace

```
One iteration of event_loop_overlap():

Step 1: recv_requests() on schedule_stream
  → Receive new GRPO rollout requests from TokenizerManager
  → Process input requests (tokenize, register)

Step 2: schedule on schedule_stream
  → get_new_batch_prefill() → create prefill batch
  → update_running_batch() → maintain decode batch
  → FutureMap: store scheduled results for forward_stream

Step 3: WAR barrier (forward_stream → schedule_stream)
  → forward_stream.record_event() → schedule_stream.wait_event()
  → Ensures forward_stream finished processing before scheduling reads

Step 4: forward on forward_stream
  → run_batch(prefill_batch) or run_batch(running_batch)
  → Model forward → attention → KV cache write → sampling → output

Step 5: process_batch_result on schedule_stream
  → After WAR barrier: schedule_stream can safely read forward results
  → FutureMap: retrieve results from pool slot → process output tokens
  → check_stop() → update output → merge batches

Step 6: copy_stream (HiCache offload)
  → After WAR barrier: copy_stream can safely read KV cache
  → async offload completed requests' KV to CPU/SSD
  → Free GPU KV slots → allocate for new requests

Step 7: loop → next iteration

★★★★★★★★ Timing analysis:
  Without overlap:
    recv(0.01ms) + schedule(0.5ms) + forward(1ms) + process(0.5ms) = 2.01ms total

  With overlap:
    recv + schedule on schedule_stream (0.5ms) overlaps with forward (1ms)
    → forward_stream never waits for schedule → GPU utilization ~100%
    → Net: forward(1ms) + process(0.5ms) = 1.5ms effective → 25% faster
```

---

## 6. Prefill-Decode Overlap Details

```
In overlap mode, prefill and decode can overlap:

Prefill batch:
  → New requests → process prompt tokens → compute initial KV cache
  → Compute-heavy: all model layers + prompt tokens
  → Duration: ~5-50ms (depends on prompt length)

Decode batch:
  → Running requests → decode one token per request
  → Memory-heavy: read/write KV cache per step
  → Duration: ~0.5-2ms per step

Overlap pattern:
  1. Prefill batch forward on forward_stream (5-50ms)
  2. Meanwhile: schedule next decode batch on schedule_stream
  3. After prefill done: merge prefill into running_batch
  4. Continue decode on forward_stream (0.5-2ms per step)

★★★★★★★★★ For GRPO rollout generation:
  Step 1: Prefill prompt (shared prefix across group) → 5-50ms
  Step 2: Decode group_size responses → each 0.5-2ms per token
  → Overlap: schedule next token while GPU forwards current token
  → Throughput improvement: ~20-40% for decode-heavy GRPO workloads

Consecutive prefill overlap:
  → Disabled by default (SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP)
  → When enabled: multiple prefills processed back-to-back → better TTFT
  → When disabled: prefill + decode interleaved → better overall throughput
  → For GRPO: consecutive prefill beneficial (many prompts to process)
```

---

## 7. Overlap vs vLLM UBatch Comparison

```
| Feature | SGLang Overlap | vLLM UBatch (DBO) |
|---------|---------------|-------------------|
| **Approach** | CPU-GPU overlap (schedule vs forward) | GPU-GPU overlap (2 microbatches) |
| **Streams** | schedule + forward + copy | compute + comm |
| **Sync mechanism** | FutureMap + WAR barrier | CUDA Event sync |
| **SM partitioning** | None (overlap is temporal) | SMControlContextManager (spatial) |
| **Benefit** | ~20-40% throughput | ~10-30% throughput |
| **Hardware requirement** | Any GPU | A100/H100 only (108+ SM + NVLink) |
| **RTX 4090 compatible** | Yes | No (too few SMs) |
| **Prefill overlap** | Yes (schedule_stream) | No (only compute-comm overlap) |

★★★★★★★★★ SGLang overlap is more universal:
  → Works on ANY GPU (no SM count requirement)
  → Overlaps CPU scheduling with GPU compute
  → FutureMap = lightweight relay, not heavy SM partitioning
  → RTX 4090: overlap works, UBatch does NOT (108 SM minimum)

vLLM UBatch is more hardware-specific:
  → Requires A100/H100 with 108+ SMs and NVLink
  → SM partitioning: reserve SMs for communication
  → NOT suitable for RTX 4090 (only 128 SM, no NVLink)

For RTX 4090 GRPO: SGLang overlap is the correct choice
```

---

## 8. OverlapSchedulerMixin Architecture

```
OverlapSchedulerMixin adds these methods to Scheduler:

init_overlap():
  → Initialize overlap-specific state
  → Create FutureMap for result relay
  → Set up schedule_stream, forward_stream, copy_stream

event_loop_overlap():
  → Main overlap loop (described in Section 5)
  → Replaces normal event_loop when overlap enabled

update_overlap_batch():
  → Manage batch scheduling during overlap
  → Use FutureMap to relay scheduled results to forward_stream

process_overlap_result():
  → Process forward results after WAR barrier
  → Merge prefill into running batch
  → Update output tokens for running requests

★★★★★★★★ The mixin architecture allows:
  → Normal Scheduler + OverlapSchedulerMixin = overlap mode
  → Normal Scheduler + PDMuxMixin = PDMux mode
  → Normal Scheduler + DisaggregationMixin = disaggregation mode
  → Clean separation: each mode adds its own scheduling logic
```

---

## 9. RTX 4090 GRPO Overlap Configuration

```
Recommended SGLang overlap config for RTX 4090 GRPO:

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B \
  --enable-overlap-loop           # Enable overlap (default for throughput)
  --enable-hicache                # HiCache for KV offload
  --hicache-ratio 0.5             # 50% freed slots go to HiCache
  --gpu-memory-utilization 0.85   # 85% VRAM for serving (15% for training)
  --enforce-eager                 # Disable CUDA graph for GRPO training safety
  --lora-rank 32                  # LoRA rank (NEVER 64!)
  --lora-alpha 64                 # LoRA alpha

★ Key settings:
  1. enable-overlap-loop → ~20-40% throughput improvement
  2. enable-hicache → KV offload during decode → longer sequences possible
  3. enforce-eager → NO CUDA graph (GRPO training safety)
  4. gpu-memory-utilization=0.85 → leave 15% headroom for training phase

Memory budget:
  Model weights: 14 GiB (GPU)
  KV cache (active): 2-3 GiB (GPU, HiCache manages)
  Overhead + overlap buffers: ~0.5 GiB (GPU)
  Training headroom: 3.5 GiB (reserved)
  Total: ~20 GiB → 4 GiB headroom for training activations
```

---

## 10. Limitations of Overlap Mode

```
1. Consecutive prefill overlap disabled by default:
   → SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = True
   → Prefills processed one-at-a-time → lower TTFT
   → Enable for GRPO: multiple prompts per step → better prompt processing

2. Overlap benefit depends on scheduling time:
   → If scheduling is fast (<0.1ms): overlap benefit negligible
   → If scheduling includes NCCL sync: overlap very beneficial
   → For single-GPU RTX 4090: scheduling is fast → ~5-10% benefit

3. FutureMap memory overhead:
   → FutureMap stores pending results → bounded by max_running_requests
   → Typically <1 MiB → negligible for RTX 4090

4. WAR barrier latency:
   → CUDA event wait adds ~0.01ms → negligible
   → But: multiple WAR barriers per step → cumulative ~0.05ms
   → Still much less than sequential bottleneck

★★★★★★★★ For GRPO on RTX 4090:
  Overlap is worth enabling (5-10% benefit for single GPU)
  But: PDMux provides MORE benefit (20-40% for SM partitioning)
  Best: PDMux + overlap + HiCache combined → maximum throughput
```

---

## Session Stats
- **Overlap vs Normal event loop**: timing comparison, throughput difference
- **FutureMap**: zero-copy data relay, pool-indexed, WAR barrier synchronization
- **3 CUDA streams**: schedule + forward + copy, synchronization via CUDA events
- **Step-by-step trace**: 7 steps per overlap iteration, timing analysis
- **Prefill-Decode overlap**: consecutive prefill, merge pattern, TTFT impact
- **Cross-framework comparison**: SGLang overlap vs vLLM UBatch (SGLang more universal)
- **OverlapSchedulerMixin**: methods added, mixin architecture
- **RTX 4090 config**: overlap + HiCache + enforce_eager + LoRA rank=32
- **Limitations**: consecutive prefill, scheduling speed, FutureMap overhead
