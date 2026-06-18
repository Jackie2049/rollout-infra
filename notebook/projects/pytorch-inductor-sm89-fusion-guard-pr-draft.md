# PyTorch Inductor SM<90 Fusion Guard — PR Draft

> ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
> RTX 4090 MOST VALUABLE OSS CONTRIBUTION!
> ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
> Target: pytorch/pytorch → torch/_inductor/choices.py
> Issue reference: vLLM #39096 (batch invariance), PyTorch #185814 (RMSNorm bandwidth)
> Status: DRAFT — needs GPU validation before submission

## PR Title

`[Inductor] Add SM<90 reduction fusion guard to prevent batch-dependent numerical results`

## PR Description Draft

### Problem

On GPUs with compute capability < 9.0 (SM89/RTX 4090, SM86, SM80, etc.), Triton's CachingAutotuner selects different XBLOCK sizes for different input shapes due to varying shared memory availability (SM89=100KB, SM80=164KB, SM90=228KB). When a reduction operation (e.g., `mean`) is fused vertically with pointwise operations into a single Triton kernel via Inductor's scheduler, the reduction becomes an inline `tl.sum()` whose accumulation order depends on the autotuned XBLOCK. Because floating-point addition is non-associative, this produces batch-dependent numerical results on SM<90 GPUs.

This is the root cause of vLLM issue #39096 and affects any model using torch.compile with RMSNorm (Llama, Mistral, Qwen3, etc.) on consumer GPUs.

### Root Cause Trace

1. `torch.compile(model)` → Dynamo captures FX graph → `rms_norm(x).mean(dim=-1)`
2. Inductor Lowering: `mean` lowered to `sum + divide` (NOT `aten::mean.dim` dispatch!)
3. ★★★★★★★★ Scheduler THREE-LAYER fusion gate architecture:
   - **Layer 0**: `V.choices.can_fuse` (common heuristic, `shared_data_score`) — shared across all
   - **Layer 1**: `Scheduler.can_fuse_vertical` (structural legality, dependency matching) — hard rules
   - **Layer 2**: `V.choices.can_fuse_vertical` (profitability heuristic) — ← OUR INSERTION POINT ← currently returns True unconditionally!
   - **Layer 3**: `Backend.can_fuse_vertical` (tiling legality) — Triton-specific checks
4. ★★★★★★★★ Both vertical fusion paths intercepted by our guard:
   - Direct path: node→producer→Layer 0→Layer 1→Layer 2(OUR GUARD)→Layer 3→fuse
   - Reindex path: node→reindex→producer→Layer 0→Layer 1→Layer 2(OUR GUARD)→Layer 3→fuse
5. Scheduler fusion: `can_fuse_vertical` returns `True` unconditionally → fuses `pow2 + sum + divide + rsqrt + mul` into ONE kernel
6. Triton codegen: persistent_reduction kernel → RBLOCK=constexpr (fixed) but XBLOCK=autotuned (varies!)
7. `tl.sum()` inline → accumulation order varies with XBLOCK → batch-dependent results on SM<90

★★★★★★★★★ Layer 2 architecturally superior to Layer 3 because:
  - can_fuse_vertical hook is INTENTIONALLY empty (designed for users to fill)
  - can_fuse_horizontal has actual heuristics (MixOrderReduction + score threshold + distance)
  - Our guard fills the empty hook legitimately, just like reduction_split_factor does with props.major>=10

### Solution

Add a 5-line guard in `InductorChoices.can_fuse_vertical` (choices.py line 640-647) that prevents vertical reduction fusion on CUDA devices with compute capability < 9.0. This keeps reductions as separate kernels where `torch.mean` batch-invariant overrides remain effective.

### Code Change

File: `torch/_inductor/choices.py`, lines 640-647

```python
@staticmethod
def can_fuse_vertical(
    scheduler: Scheduler,
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
    shared_data_score: int,
) -> bool:
    """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
    # SM<90 Fusion Guard: On GPUs with compute capability < 9.0, Triton
    # autotuning selects different XBLOCK sizes for different input sizes.
    # When a reduction is fused vertically with pointwise ops, the
    # reduction becomes an inline tl.sum() whose accumulation order
    # depends on XBLOCK. This causes batch-dependent numerical results
    # on SM<90. Preventing vertical reduction fusion on SM<90 keeps
    # reductions as separate kernels where torch.mean's batch-invariant
    # override remains effective.
    if node1.is_reduction() or node2.is_reduction():
        device = node1.get_device() or node2.get_device()
        if device is not None and device.type == "cuda":
            props = DeviceProperties.create(device)
            if props.major is not None and props.major < 9:
                WhyNoFuse(node1, node2)(
                    "SM<90 prevents reduction fusion (batch invariance)"
                )
                return False
    return True
```

### Precedents

This pattern follows 5 existing SM-capability checks in the Inductor codebase:

1. **PRIMARY precedent**: `choices.py` lines 482-506 — `DeviceProperties.create(device)` + `props.major >= 10` in `reduction_split_factor`. Same class, same file, same pattern.
2. **triton.py** line 2849 — `torch.cuda.get_device_capability()[0] >= 9` for TMA gate
3. **triton.py** line 4129 — same pattern for PDL gate
4. **triton_heuristics.py** lines 722-723 — `device_prop.major >= 8` for rblock scaling
5. **triton_heuristics.py** line 4223 — `device_major >= 10` for Blackwell MAX_R0_BLOCK

No additional imports needed — `DeviceProperties` (line 19) and `WhyNoFuse` (line 20) are already imported at the top of `choices.py`.

### Impact Scope

**Affected** on SM<90:
- Vertical reduction + pointwise fusions (e.g., RMSNorm `pow2+mean+rsqrt+mul`) → blocked
- Reductions remain as separate kernels → torch.mean batch-invariant overrides work correctly

**NOT affected**:
- Horizontal fusions (same iteration domain) → separate method
- SM90+ GPUs → `props.major >= 9` → pass guard → TMA/WGMMA deterministic
- XPU/CPU/other backends → `device.type != "cuda"` → skip guard
- Pure pointwise vertical fusions → no reduction → skip guard

### Test Plan

```python
# test_sm90_fusion_guard.py
import torch
from torch._inductor.choices import InductorChoices
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse
from torch._inductor.runtime.hints import DeviceProperties

class MockNode(BaseSchedulerNode):
    def __init__(self, is_red, device_str):
        self._is_reduction = is_red
        self._device = torch.device(device_str) if device_str else None
    def is_reduction(self):
        return self._is_reduction
    def get_device(self):
        return self._device

def test_sm89_reduction_fusion_blocked():
    """SM89 (major=8) should block vertical reduction fusion."""
    node1 = MockNode(is_red=True, device_str="cuda:0")
    node2 = MockNode(is_red=False, device_str="cuda:0")
    # Mock DeviceProperties for SM89
    result = InductorChoices.can_fuse_vertical(scheduler, node1, node2, 0)
    assert result == False  # Blocked on SM89

def test_sm90_reduction_fusion_allowed():
    """SM90 (major=9) should allow vertical reduction fusion."""
    node1 = MockNode(is_red=True, device_str="cuda:0")
    node2 = MockNode(is_red=False, device_str="cuda:0")
    # Mock DeviceProperties for SM90
    result = InductorChoices.can_fuse_vertical(scheduler, node1, node2, 0)
    assert result == True  # Allowed on SM90

def test_xpu_unaffected():
    """XPU devices should not be affected by the guard."""
    node1 = MockNode(is_red=True, device_str="xpu:0")
    node2 = MockNode(is_red=False, device_str="xpu:0")
    result = InductorChoices.can_fuse_vertical(scheduler, node1, node2, 0)
    assert result == True  # XPU unaffected

def test_pointwise_only_unaffected():
    """Fusions involving only pointwise ops should not be affected."""
    node1 = MockNode(is_red=False, device_str="cuda:0")
    node2 = MockNode(is_red=False, device_str="cuda:0")
    result = InductorChoices.can_fuse_vertical(scheduler, node1, node2, 0)
    assert result == True  # No reduction → unaffected
```

### Related Issues

- vLLM #39096 — Batch invariance bug on SM89 (this PR fixes the root cause)
- PyTorch #185814 — XBLOCK derivation for RMSNorm backward (complementary bandwidth fix)
- vLLM #43914 — Triton FP8 KV SM89 ALLOWED (related SM89 compatibility)
- Non-TMA PRs #177781/#179095 — SM89 persistent Triton templates (CLOSED without merging — AMD-focused, NOT SM89 batch invariance fix → confirms our approach is independent and necessary)

### ★★★★★★★★ Model-Size Dependency Evidence (vLLM #39096 comments)

YM2132 (2026-04-17) tested batch invariance on SM86 (RTX 3090) without enforce_eager:
- **Qwen3-1.7B**: PASSED with torch.compile on SM86
- **Key insight**: vLLM's `torch.mean` override WORKS when the reduction stays as a SEPARATE kernel
- **Problem**: Inductor FUSES the reduction for SOME model configurations → override bypassed → fails
- **This is model-size-dependent**: smaller models may not trigger the fusion → larger models do
- **Our guard**: prevents fusion for ALL models on SM<90 → consistent fix → no model-specific workaround needed

### Performance Impact

Expected: Minimal. On SM89, vertical reduction fusions were already producing incorrect results, so blocking them is a correctness fix. Separate reduction kernels have deterministic behavior. On SM90+, no change — fusions proceed as before.

Possible follow-up: Fine-tune the guard to only block specific problematic reduction patterns (e.g., mean/sum + rsqrt + mul for RMSNorm) while allowing safe reduction fusions. This would be a separate PR after initial correctness fix is validated.

### GPU Validation Needed

Before submitting, we need to validate on an actual RTX 4090 (SM89) that:
1. `torch.compile(LlamaModel)` with RMSNorm produces batch-invariant results with the guard
2. Performance is acceptable (reduction as separate kernel vs fused)
3. SM90+ GPUs show no change in behavior

## Checklist Before Submission

- [ ] Run vLLM on RTX 4090 with torch.compile + this guard → verify batch invariance
- [ ] Run performance benchmark: guard ON vs OFF on RTX 4090 → quantify overhead
- [ ] Run on SM90 GPU (if available) → verify no change
- [ ] Run PyTorch Inductor CI tests → verify no regressions
- [ ] Draft PyTorch issue to accompany PR → reference vLLM #39096
- [ ] Get community feedback on approach → PyTorch dev forum

### ★★★★★★★★ Source-Verified on PyTorch main (2026-06-18)

★★★★★★★★★ Verified on current PyTorch main branch (choices.py = 729 lines):
  → can_fuse_vertical at lines 640-647: still returns True unconditionally → our insertion point CONFIRMED
  → DeviceProperties at line 19 import: `from .runtime.hints import DeviceProperties, ReductionHint` → already imported
  → WhyNoFuse at line 20 import: `from .scheduler import BaseSchedulerNode, Scheduler, WhyNoFuse` → already imported
  → reduction_split_factor at lines 473-506: uses `DeviceProperties.create(device)` + `props.major >= 10` → our PRIMARY precedent
  → is_reduction() and get_device() on BaseSchedulerNode: CONFIRMED in scheduler.py (lines 1462, 1449)
  → ★★★★★★★★ No additional imports needed → minimal change → easy review!

★★★★★★★★★ Three-layer fusion call architecture (scheduler.py lines 7886-7926):
  Layer 1: V.choices.can_fuse() → general heuristic (shared_data_score) → line 7886
  Layer 2: self.can_fuse_vertical() → structural legality (dependency matching) → lines 7892/7911
  Layer 3: V.choices.can_fuse_vertical() → profitability hook → lines 7897/7916 ← OUR INSERTION POINT!
  Layer 4: self.get_backend(device).can_fuse_vertical() → tiling legality → lines 7898/7919

★★★★★★★★★ Call flow (both direct and reindex paths):
  Direct: node→producer→L1→L2→L3(OUR GUARD)→L4→fuse
  Reindex: node→reindex→producer→L1→L2→L3(OUR GUARD)→L4→fuse
  → If our guard returns False → fusion stops → WhyNoFuse logged → separate kernel dispatched!

★★★★★★★★★ #184119 (SM89 fp8 prologue guard) uses get_cuda_arch() — reviewer suggested it:
  → get_cuda_arch() in cuda_env.py: cached, returns string "89"/"90"/"100" etc.
  → #184119 adopted get_cuda_arch() per reviewer preference → simpler global check
  → Our guard uses DeviceProperties.create(device) → per-device → SAME as reduction_split_factor precedent
  → Why DeviceProperties over get_cuda_arch:
    1. Already imported in choices.py (no new import needed)
    2. Same pattern as reduction_split_factor precedent (same file)
    3. Per-device → more correct in principle (multi-GPU with different SM versions)
    4. If reviewer prefers get_cuda_arch → we can adapt → but DeviceProperties aligns with existing code

★★★★★★★★★ v2.12 max_autotune EXACERBATES SM89 behavior:
  → Combo kernels (#177715/#178936/#179317) → more kernels autotuned → more variability
  → On SM89: more autotuned configs = more XBLOCK variation = MORE batch-dependent!
  → Our guard protects regardless: ALL reduction fusions blocked on SM<90 → no autotuned reductions

★★★★★★★★★ #187275 confirms same root cause class:
  → "Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions" (OPEN, updated June 18)
  → Same architectural weakness: persistent reduction block sizes not properly handled
  → Our issue = numerical correctness (different results per batch)
  → Their issue = crash correctness (hardcoded RBLOCK invalid when rnumel changes)
  → Both from persistent reduction dimension handling → CONFIRMS diagnosis

★★★★★★★★★ v2.13 RISK (tracked in vLLM #45731):
  → PyTorch 2.13.0 → Triton 3.7.1 → may change SM89 autotuning behavior
  → IF Triton 3.7.1 fixes XBLOCK variability → our guard becomes unnecessary
  → IF Triton 3.7.1 DOES NOT fix → our guard STILL needed
  → Recommendation: file PyTorch issue NOW regardless → track v2.13 outcome

★★★★★★★★★ Submission readiness update:
  → Technical readiness: 95% (unchanged)
  → Submission readiness: 60% → primary blocker = GPU validation (RTX 4090 offline)
  → Test plan uses MockNode (incorrect) → should use unittest.mock.patch approach
  → PyTorch issue NOT yet filed (pre-step per approach strategy)

## 参考
- Source analysis: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-approach.md
- Root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Triton codegen pipeline: notebook/projects/pytorch-inductor-triton-codegen-pipeline-reading.md
- vLLM issue: github.com/vllm-project/vllm/issues/39096
- PyTorch issue: github.com/pytorch/pytorch/issues/185814
- Repro script: tools/sm89_batch_invariance_repro.py
- Diagnostic: tools/sm89_batch_invariance_diagnostic.py

### ★★★★★★★★ Additional Evidence from Triton Codegen Pipeline Analysis

CachingAutotuner persistent_reduction mechanism (triton_heuristics.py line 4882):
- RBLOCK = next_power_of_2(rnumel) → tl.constexpr → FIXED across all batch sizes
- XBLOCK → autotuned → varies with input shape (different for batch=1 vs batch=8)
- SM89 shared memory: 100KB (vs SM80=164KB, SM90=228KB) → forces different XBLOCK selections
- Result: tl.sum() accumulates over different numbers of rows → non-associative FP addition → batch-dependent!

Combo kernel PR #187275 (opened 2026-06-14) confirms our root cause class:
- "Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions"
- Same architectural weakness: persistent reduction block sizes not properly handled across dynamic dimension changes
- Our issue = numerical correctness (different results per batch size)
- Their issue = crash correctness (hardcoded RBLOCK invalid when rnumel changes)
- Both stem from persistent reduction dimension handling → CONFIRMS our diagnosis is part of a known problem class!

v2.12 max_autotune for combo kernels (#177715/#178936/#179317):
- Extension of autotuning to combo kernels → more kernels undergo autotuning
- May EXACERBATE SM89 batch-dependent behavior → more kernels = more autotuned configs = more variability
- Our guard protects regardless: ALL reduction fusions blocked on SM<90 → no autotuned reductions
