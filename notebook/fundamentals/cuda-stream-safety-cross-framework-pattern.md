# CUDA Stream Safety — Cross-Framework Pattern Analysis

**Created: 2026-06-19 | Synthesis of DeepSpeed #8061 + verl #6794 + Megatron ChainedOptimizer**

---

## 1. The Pattern: Multi-Stream CUDA Data Race

### 1.1 Mathematical Model

CUDA stream safety violations follow a consistent pattern:

```
Producer Stream P: tensor.copy_(src) or tensor = computed_value
Consumer Stream C: reads tensor before P completes

Race condition probability:
  P(race) = P(C reads before P writes) × P(P hasn't completed)
           = f(t_c - t_p) × f(overlap_comm)
           where f is a timing-dependent function
```

When `overlap_comm=True` or async side streams are used:
- `t_c < t_p` is common (consumer starts before producer finishes)
- Without synchronization: `P(race) → 1` on multi-stream paths
- With proper sync: `P(race) → 0`

### 1.2 Three Manifestations Across Frameworks

| Framework | Issue | Stream Pattern | Symptom | Root Cause |
|-----------|-------|---------------|---------|------------|
| DeepSpeed | #8061 | IPG bucket `copy_` on multiple autograd streams, `average_tensor()` on reduction stream | NaN in gradient reduction | `reduction_stream.wait_stream(current_stream)` only waits ONE stream |
| verl | #6794 CRITICAL-1 | Delta snapshot `copy_` on d2h_stream, reads on default stream | Silent data corruption in snapshot | Missing `tensor.record_stream(d2h_stream)` |
| Megatron | ChainedOptimizer | Optimizer operations on side streams, reads on default stream | Gradient stalls (all optimizers) | Global clipping on ALL sub-optimizers |

★★★★★★★★★ **Common element**: All three involve **non-default CUDA streams** without proper synchronization or lifetime tracking.

---

## 2. PyTorch CUDA Stream Safety Rules

### 2.1 The Default Stream Assumption

PyTorch's caching allocator assumes tensors are used on the **default stream** for lifetime tracking. When tensors are used on **non-default streams**:

- The allocator may reclaim tensor memory before the non-default stream completes
- This leads to **silent data corruption** (no error, no warning, wrong data)
- The corruption propagates through subsequent computations

### 2.2 Two Required Safety Mechanisms

1. **`tensor.record_stream(stream)`**: Tells PyTorch's allocator that `tensor` will be used on `stream`. The allocator then delays reclaiming `tensor`'s memory until `stream` completes all operations involving `tensor`.

2. **`stream.wait_stream(other_stream)`**: Makes `stream` wait until `other_stream` completes all previously enqueued operations. This ensures `stream` reads data that `other_stream` has fully written.

★★★★★★★★★ **Rule**: For ANY multi-stream code path, BOTH mechanisms may be needed:
- `record_stream` → prevents allocator from reclaiming memory
- `wait_stream` → prevents reading incomplete data

### 2.3 When Each Mechanism Is Needed

| Scenario | `record_stream` | `wait_stream` | Example |
|----------|-----------------|---------------|---------|
| Async copy to side stream, then read on default stream | YES (source tensor) | YES (default_stream.wait_stream(side_stream)) | verl #6794 D2H |
| Async copy from side stream to default stream buffer | YES (source tensor) | YES (consumer_stream.wait_stream(producer_stream)) | DeepSpeed #8061 IPG |
| Multi-stream writes to shared buffer | N/A (shared buffer) | YES (consumer_stream.wait_stream(ALL producer streams)) | DeepSpeed #8061 |
| Async compute on side stream, then use result | YES (input tensors) | YES (default_stream.wait_stream(side_stream)) | Megatron optimizer |

---

## 3. Framework-Specific Analysis

### 3.1 DeepSpeed #8061: IPG Bucket Multi-Stream Race

```
Stream A: copy_ grad_slice_A → bucket.buffer  (autograd backward)
Stream B: copy_ grad_slice_B → bucket.buffer  (compiled backward)
Stream C: average_tensor() reads bucket.buffer (reduction stream)

Current: reduction_stream.wait_stream(current_stream) → ONLY waits stream C
Fix:    reduction_stream.wait_stream(ALL streams that wrote to bucket)
        OR: torch.cuda.synchronize() (heavy, defeats overlap_comm purpose)
```

★★★★★★★★★ **RTX 4090 implication**: On dp=1, overlap_comm=False eliminates this race entirely (all operations on default stream). overlap_comm=True is pure overhead on dp=1 anyway (reduction is identity).

### 3.2 verl #6794: Delta Snapshot D2H Race

```
Default stream: tensor.detach() computed
d2h_stream:     snapshot[name].copy_(tensor.detach(), non_blocking=True)

Current: NO record_stream → allocator may reclaim tensor before d2h_stream completes
Fix:    tensor.record_stream(d2h_stream) → allocator delays reclaiming

Also applies to h2d_stream prefetch path.
```

★★★★★★★★★ **RTX 4090 implication**: Same risk applies. On dp=1, delta sync provides limited benefit (sleep_level=1 LoRA path already optimal). But the `record_stream` bug must be fixed regardless.

### 3.3 Megatron: ChainedOptimizer Stream Safety

```
Default stream: forward + backward computation
Optimizer streams: side streams for optimizer operations

Current: Global clipping applied to ALL sub-optimizers → stalls
Fix:    skip_grad_norm_clip for orthogonalizing optimizers (#5395)
```

★★★★★★★★★ **RTX 4090 implication**: Megatron requires multi-GPU → not directly relevant. But the pattern of optimizer-side-stream issues is related.

---

## 4. 4-Layer Defense Stack Update

### Layer 1: Framework Safety (CUDA Stream Safety Addition)

```
Layer 1: Framework Safety
  → enforce_eager=True (eliminates cudagraph crashes)
  → overlap_comm=False (eliminates NaN from multi-stream race)
  → ZeRO-2 (eliminates dtype mismatch from ZeRO-3)
  → FSDP backend only (eliminates detach leak from 3 unfixed backends)
  → record_stream on ALL async side-stream copies (NEW! prevents silent corruption)
  → wait_stream for ALL multi-producer reads (NEW! prevents stale reads)
```

### Layer 2: Cache Management (unchanged)

### Layer 3: Monitoring (CUDA Stream Monitoring Addition)

```
Layer 3: Monitoring
  → Output quality monitoring (detect Level 5 accumulation before critical)
  → Periodic engine restart every 20-50 steps (reset accumulated state errors)
  → ulimit -n 65536 (fd leak safety, #8075)
  → NanDetectMode (#187653) for forward-pass NaN detection
  → Host RAM monitoring (#6468 FSDP2 leak detection)
  → CUDA stream usage profiling (NEW! detect unexpected multi-stream paths)
```

### Layer 4: Algorithmic (unchanged)

---

## 5. RTX 4090 MUST DO / MUST NOT Rules Update

### MUST DO (adding CUDA stream safety rules):

11. **overlap_comm=False** — eliminates DeepSpeed #8061 multi-stream race (4-point evidence matrix)
12. **record_stream on ALL async copies** — prevents silent data corruption (verl #6794 pattern)
13. **FSDP1 backend** — avoids FSDP2 CPU leak (#6468, 0.6-6.3 GiB/step)
14. **Host RAM monitoring** — detect FSDP2 leak before OOM

### MUST NOT (adding CUDA stream safety rules):

11. **overlap_comm=True on single GPU** — #8061 NaN confirmed (4-point evidence matrix)
12. **Async side-stream copies without record_stream** — silent corruption (verl #6794 pattern)
13. **FSDP2 backend for long-running GRPO** — #6468 CPU leak → host OOM
14. **Multi-producer stream reads without wait_stream** — stale data (DeepSpeed #8061 pattern)

★★★★★★★★★ Total rules: 14 MUST DO + 14 MUST NOT (was 10+10)

---

## 6. Cross-Framework Stream Safety Audit Checklist

For ANY multi-stream code path in ANY framework, check:

1. **Are ALL producer streams waited on before reading shared data?**
   - DeepSpeed: IPG bucket → needs ALL stream waits (#8061)
   - verl: Delta snapshot → needs ALL stream waits on prefetch path

2. **Is `record_stream()` called on ALL source tensors for async copies?**
   - verl: Delta snapshot → needs `record_stream(d2h_stream)` (#6794)

3. **Is the default stream assumption correct?**
   - If tensors are used on non-default streams, the allocator must be informed
   - Missing `record_stream` → allocator reclaims → silent corruption

4. **Are there multi-producer single-consumer patterns?**
   - Multiple streams writing to the same buffer → consumer must wait for ALL
   - DeepSpeed: Multiple autograd streams → IPG bucket → single reduction

5. **Is `torch.cuda.synchronize()` used as a safety net?**
   - Heavy but guaranteed safe → acceptable for debugging, not for production
   - Production: specific `wait_stream` and `record_stream` per-producer

---

## 7. Detection and Debugging

### 7.1 How to Detect Stream Safety Violations

1. **Intermittent NaN or wrong values**: Pattern may not reproduce consistently (timing-dependent)
2. **Configuration-dependent**: NaN only with overlap_comm=True, compile=True, or async mode
3. **Reproducer difficulty**: Small scripts may not trigger the same timing as production

### 7.2 Debugging Protocol

1. **Configuration matrix**: Test all combinations of overlap_comm, torch.compile, async modes
2. **`torch.cuda.synchronize()` test**: Add full sync before the suspect read → if NaN disappears → stream race
3. **Stream profiling**: Use `torch.profiler` to identify which streams are active at the suspect point
4. **`record_stream` test**: Add `record_stream` on suspect async copy source → if corruption disappears → allocator race

### 7.3 NanDetectMode Integration

★★★★★★★★★ NanDetectMode (#187653) can detect stream race NaN in forward pass:
- If `average_tensor()` produces NaN due to stale reads → NanDetectMode catches the `div_` or `narrow` operation that reads corrupt data
- But: stream race corruption may not always produce NaN — it can produce **wrong but valid** values → silent corruption harder to detect than NaN

---

## 8. Implications for Future Code

### 8.1 When Writing Multi-Stream CUDA Code

★★★★★★★★★ **Golden Rule**: ANY time you use `torch.cuda.Stream()` or `non_blocking=True` for cross-stream operations:

1. Call `tensor.record_stream(side_stream)` on the SOURCE tensor
2. Call `consumer_stream.wait_stream(producer_stream)` before reading shared data
3. Document which streams are producer and which are consumer
4. Test with `torch.cuda.synchronize()` to verify correctness

### 8.2 When Reviewing Framework Code

★★★★★★★★★ **Review Checklist**: For ANY framework PR that touches:
- Gradient reduction / overlap_comm
- Weight sync / update_weights
- Async D2H/H2D copies
- Side streams for pipelining
- NCCL broadcast / all-reduce

Check: `record_stream` and `wait_stream` on ALL cross-stream operations.

---

*Created 2026-06-19. CUDA stream safety cross-framework pattern analysis — synthesis of #8061 + #6794 + Megatron.*
