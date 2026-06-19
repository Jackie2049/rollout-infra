# vLLM #45552 — CuMemAllocator Sleep/Wake Stream Synchronization Bugfix

> 2026-06-19 | PR #45552 OPEN (+256/-0) | Author: terafin
> ★★★★★★★★ 6th Weight Reload State Lifecycle Mismatch pattern member
> ★★★★★★★★ CRITICAL for RTX 4090 GRPO: sleep/wake missing cuda.synchronize() → CUDART crash

---

## 1. Bug Description

`CuMemAllocator.sleep()` and `wake_up()` are missing `torch.cuda.synchronize()` barriers around their `cuMemUnmap` / `cudaMemcpy` regions.

**Root cause**: The allocator assumes callers have drained all in-flight CUDA work, but on live V1 paths they haven't:

### sleep() crash path:
1. `/sleep` calls `pause_scheduler(mode="abort")` → sets Python-side scheduler state
2. **Immediately** enters allocator offload loop (no CUDA stream drain)
3. Kernels already submitted by `model_runner.execute_model` (decode steps, P2P sends, KV writes) keep running **asynchronously**
4. `libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)` at cumem.py:202 reads from a region a still-running kernel is writing into → **read-before-write-complete race**
5. `unmap_and_release(handle)` invalidates pages a kernel still holds → **invalidated-page-in-use race**
6. Both surface as `cudaErrorIllegalAddress` / `CUDART error: an illegal memory access was encountered`
7. `/sleep` returns HTTP 200 in ~300ms while engine is already shutting down → **the "200 lie" pattern**

### wake_up() crash path:
1. `/wake_up` issues per-allocation H2D `cudaMemcpy`s
2. No `torch.cuda.synchronize()` at end → control returns to HTTP `/wake_up` 200 response
3. Tail kernels may still be active on the device
4. A rapid subsequent `/sleep` (typical in RLHF rotation, swap-group, multi-tenant patterns) races those tail kernels → crash

---

## 2. Fix (2 targeted synchronize() calls)

### In sleep():
```python
# BEFORE any cuMemUnmap or D2H copy:
if libcudart is not None:
    torch.cuda.synchronize()
```
Guarantees: All in-flight CUDA work finishes BEFORE any `cuMemUnmap` or D2H `cudaMemcpy`

### In wake_up():
```python
# AFTER all H2D restore copies:
if libcudart is not None:
    torch.cuda.synchronize()
```
Guarantees: `wake_up` (and HTTP `/wake_up`) cannot return until all H2D restore copies have completed on device

---

## 3. Pattern Family Classification

★★★★★★★★ This is the **6th member** of the Weight Reload State Lifecycle Mismatch pattern family:

| # | Framework | Issue | Root Cause | Severity |
|---|-----------|-------|------------|----------|
| 1 | vLLM | #46125 | Stale encoder cache after weight update revert | HIGH |
| 2 | SGLang | #28676 | MXFP8 MoE cache clobbered on weight reload | CRITICAL |
| 3 | vLLM-Ascend | #10684 | DSA Hadamard ALL-ZERO after sleep/wake | CRITICAL |
| 4 | vLLM | #44395 | wake_up(weights) + forward → illegal memory | HIGH |
| 5 | SGLang | #28679 | GDN intermittent degeneracy | HIGH |
| 6 | vLLM | **#45552** | CuMem sleep/wake missing cuda.synchronize | CRITICAL |

**Universal rule**: Any GPU-resident cache or state MUST be invalidated/synchronized at weight-reload boundary.

**Related pattern**: CUDA Stream Safety (#8061, #6794) — same underlying issue of async GPU operations racing synchronization boundaries.

---

## 4. RTX 4090 GRPO Impact

★★★★★★★★ CRITICAL for RTX 4090 GRPO training:

- verl HYBRID sleep/wake pattern: sleep → train → wake → rollout → sleep → ...
- `/sleep` called after each rollout generation phase → in-flight kernels from decode steps
- `/wake_up` called before each rollout → H2D restore copies + immediately start generation
- Both paths MUST have stream synchronization → otherwise RTX 4090 GRPO CRASHES

**Without this fix**: RTX 4090 GRPO training will crash with `cudaErrorIllegalAddress` during sleep/wake transitions.

**MUST DO**: Either (1) wait for #45552 merge, or (2) add `torch.cuda.synchronize()` in custom sleep/wake hooks.

---

## 5. Test Coverage

PR includes 213-line test file `test_cumem_sync_before_unmap.py`:
- Tests do NOT require GPU — patch cumem C-extension entry points
- Invariant 1: sleep() syncs BEFORE any unmap or D2H copy
- Invariant 2: wake_up() syncs BEFORE returning
- Both ordering invariants are asserted in test functions

---

## References

- vLLM #45552: https://github.com/vllm-project/vllm/pull/45552
- vLLM #45520: https://github.com/vllm-project/vllm/issues/45520 (sleep crash)
- vLLM #36753: https://github.com/vllm-project/vllm/issues/36753 (wake_up crash)
- Weight reload pattern: notebook/fundamentals/cross-framework-partial-wake-safety-analysis.md
- CUDA stream safety: notebook/fundamentals/cuda-stream-safety-cross-framework-pattern.md
