# SGLang #28363 Gate Overlap WAR Barrier — Analysis

> 2026-06-16 | PR #28363 (OPEN) | RTX 4090: philosophical connection to BudgetRefiner
> Key: fine-grained read-done event gates WAR barrier → recovers Blackwell throughput regression

---

## 1. WAR Barrier Problem

SGLang overlap scheduling: `schedule_stream` (CPU prep + GPU buffer writes) and `forward_stream` (model compute) run concurrently. Schedule stream writes shared buffers (`req_to_token_pool`, SWA mappings) while previous forward reads them → **WAR (Write-After-Read) hazard**.

PR #26380 fixed by adding `schedule_stream.wait_stream(forward_stream)` → gates on **entire forward end** → correct but over-conservative → Blackwell regression: -34.6% (gpt-oss-120b), -11.9% (DeepSeek-R1), etc.

---

## 2. Fine-Grained Gating: Read-Done Event

Key insight: forward's reads of shared buffers finish **early** — during `replay_prepare`, before heavy compute (`backend.replay`). Compute/sampling phases never touch these buffers.

### Implementation

```python
# ModelRunner: shared_buf_read_done_event (CUDA Event)
# decode_cuda_graph_runner.replay():
self.replay_prepare(forward_batch, pp_proxy_tensors)
self.model_runner.shared_buf_read_done_event.record()  # ← read-done HERE
self.model_runner.shared_buf_read_done_fresh = True
output = self.backend.replay(...)                       # ← compute AFTER

# scheduler._apply_war_barrier():
if spec_algorithm.is_none() and mr.shared_buf_read_done_fresh:
    self.schedule_stream.wait_event(mr.shared_buf_read_done_event)  # ← fast path
    mr.shared_buf_read_done_fresh = False
else:
    self.schedule_stream.wait_stream(self.forward_stream)  # ← conservative fallback
```

Fast path: wait on read-done event → schedule prep overlaps with forward compute/sampling.
Fallback: spec decode + non-cuda-graph paths → conservative whole-forward wait.

---

## 3. min_new_tokens Penalizer: Non-Synchronizing

Old: boolean-mask indexing → forces CUDA→host sync → bubble in overlap pipeline
```python
mask = (self.len_output_tokens < self.min_new_tokens).expand_as(logits)
logits[mask] += self.stop_token_penalties[mask]  # host sync!
```

New: `torch.where` → pure elementwise GPU op → no host sync
```python
mask = self.len_output_tokens < self.min_new_tokens
logits.add_(torch.where(mask, self.stop_token_penalties, 0.0))  # no sync!
```

Also avoids `-inf * 0 = nan` edge case from boolean-mask indexing.

---

## 4. Philosophical Connection to BudgetRefiner

★★★★★★★★★ Both PRs share the same scheduling principle: **barriers should be scoped to the minimum necessary dependency, not to the entire operation**.

| Aspect | #28363 WAR Barrier | BudgetRefiner |
|--------|-------------------|---------------|
| Barrier scope | Entire forward → read-done event | FCFS → decode-first priority |
| Proactive vs reactive | Reactive (wait for end) → Proactive (gate at earliest safe point) | Reactive (wait for OOM) → Proactive (reduce budget under pressure) |
| Serialization | Over-conservative → fine-grained event | Over-conservative → dynamic budget reduction |
| Coordination signal | CUDA Event (GPU stream) | d_num (decode count) |
| Philosophy | **Minimum necessary dependency** | **Minimum necessary admission** |

BudgetRefiner applies the same principle: scheduling decisions should account for compute time, not just memory. Prefill admissions are gated on decode pressure (d_num) rather than waiting for KV cache to fill and preempting.

---

## 5. RTX 4090 Direct Impact

★★★ RTX 4090 does NOT directly benefit from #28363 — the regression was Blackwell-specific, and RTX 4090 doesn't use SGLang overlap scheduling in production.

★★★★★ But the **philosophical connection** strengthens the BudgetRefiner design rationale: fine-grained, compute-time-aware scheduling is a validated principle across inference frameworks.

---

## References

- SGLang PR #28363: Gate Overlap WAR Barrier on Forward Reads
- SGLang #26380: Original WAR barrier fix (caused regression)
- BudgetRefiner SLO: dynamic token budget + decode-first (same scheduling philosophy)
