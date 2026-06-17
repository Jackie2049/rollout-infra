# Megatron #5219 — Single-GPU Muon Crash Fix Source Reading

> 2026-06-18 | PR #5219 (Final Review, OPEN) | +14/-7 | Author: factnn
> ★★★★★★★★ None guard for dp_cp_params_list → single-GPU LayerWise optimizer initialization crash
> ★★★★★★★★ Irony: expt_dp_params_list already had the correct None guard → dp_cp path was missing
> ★★★★★★★★ CRITICAL for RTX 4090 — without this, Muon optimizer simply crashes on single GPU

---

## 1. Crash Mechanism

```
★★★★★★★★★ The crash path:

In LayerWiseDistributedOptimizer.shard_params() (layer_wise_optimizer.py lines 450-455):
  → When dp_cp_size == 1 (single GPU / no data parallelism)
  → Early return sets BOTH dp_cp_params_list and expt_dp_params_list to None
  → This is CORRECT → on single GPU, no sharding needed → no per-rank param lists

But in set_bucket_layerwise_params_list() (lines 598-609):
  → Iterates over self.dp_cp_params_list in zip() call WITHOUT None guard
  → When dp_cp_params_list is None → zip(bucket_params_list, None)
  → Python TypeError → CRASH at optimizer initialization!

★★★★★★★★★ The irony:
  → Expert parallel code path (lines 611-629) ALREADY had the correct None guard
  → expt_dp_params_list path: if/else with else branch setting
    bucket_params_list = [list(bucket.params_list)]
  → dp_cp_params_list path: NO such guard → just iterates unconditionally → crash
  → ★★★★★★★★ Same pattern should have been applied to dp_cp path → missed

★★★★★★★★★ Why the else branch is essential:
  → Even when no all-gather needed → bucket internal data structures MUST be initialized
  → bucket_params_list = [list(bucket.params_list)] → local params list for single-rank case
  → Optimizer can still function → just without cross-rank communication
```

---

## 2. The Fix (+14/-7)

```
★★★★★★★★★ The fix mirrors expt_dp_params_list pattern for dp_cp_params_list:

Before (lines 598-609):
  → Unconditionally iterates zip(bucket_params_list, self.dp_cp_params_list)
  → Crashes when dp_cp_params_list is None

After:
  → Adds if self.dp_cp_params_list is not None: guard
  → Full distribution logic inside the if branch
  → else branch: bucket_params_list = [list(bucket.params_list)]
  → ★★★★★★★★ IDENTICAL pattern to expert parallel else branch at line 628

★★★★★★★★★ Key files:
  → megatron/core/optimizer/layer_wise_optimizer.py — single changed file
  → Lines 598-629 — both unfixed dp_cp path and already-fixed expt_dp path
```

---

## 3. Review Status

```
★★★★★★★★★ Current status:
  → Open, Draft PR, "Final Review" label
  → Author: factnn (community contributor)
  → Reviewer: guihong-nv (NVIDIA maintainer) → approved /ok to test twice
  → Lint issues flagged → fixed
  → mergeable_state: blocked → NVIDIA internal CI vetting needed

★★★★★★★★★ When it might merge:
  → Draft PR + blocked mergeable state + Final Review label
  → Code change reviewed → lint fixed → no technical objections
  → Only 1 file changed → +14/-7 → straightforward None guard
  → ★★★★★★★★ Likely within days to weeks → NVIDIA internal CI vetting gate
```

---

## 4. RTX 4090 Muon Dependency Chain

```
★★★★★★★★★ RTX 4090 single-GPU Muon dependency chain:

PR #5219 (this fix): MUST merge FIRST → removes hard crash
  → Without this → Muon optimizer simply CRASHES at initialization on RTX 4090

PR #5394 + fix PR #5395: MUST merge for correct training → removes stalling
  → Without this → Muon runs but produces DEGRADED results (clipping destroys NS)

PR #5391 (compact DDP): Optional but highly beneficial → memory efficiency
  → Without this → Muon works but uses more memory → less room for LoRA/activations
  → With this → dp=1 padding removed → more RTX 4090 headroom

★★★★★★★★★ Full viable path requires #5219 + #5394/#5395:
  → #5219: crash fix → enables initialization
  → #5395: skip_grad_norm_clip → enables correct training
  → #5391: compact DDP → enables memory-efficient training
  → ★★★★★★★★ All 3 needed for production RTX 4090 Muon
```

---

## Key Findings Summary

★★★★★★★★★ #5219: None guard for dp_cp_params_list → single-GPU Muon crash fix (+14/-7)
★★★★★★★★★ Irony: expt_dp_params_list already had correct guard → dp_cp path was missing
★★★★★★★★★ Final Review status → no technical objections → likely days-weeks to merge
★★★★★★★★★ RTX 4090 Muon dependency: #5219 (crash) + #5395 (clipping) + #5391 (memory) = full viable path
★★★★★★★★★ Without #5219 → Muon CRASHES → without #5395 → Muon DEGRADED → both must merge

---

## References

- Megatron #5219: https://github.com/NVIDIA/Megatron-LM/pull/5219
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (clipping bug)
- Megatron #5395: https://github.com/NVIDIA/Megatron-LM/pull/5395 (clipping fix)
- Megatron #5391: https://github.com/NVIDIA/Megatron-LM/pull/5391 (compact DDP)
- Cross-framework Muon clipping: notebook/fundamentals/cross-framework-muon-gradient-clipping-bug.md
