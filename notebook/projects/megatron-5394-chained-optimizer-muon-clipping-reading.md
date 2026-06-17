# Megatron #5394 — ChainedOptimizer Muon Gradient Clipping Stall Source Reading

> 2026-06-18 | Issue #5394 (OPEN, June 17) | Author: yuchenwang3 (Yuchen Wang) | Cross-framework bug pattern
> ★★★★★★★★ SAME pattern as DeepSpeed #8068 (gradient_clipping default) + #7776 (ordering)
> ★★★★★★★★ ChainedOptimizer computes single global grad_norm across ALL chained sub-optimizers
> ★★★★★★★★ Muon+Adam chained → global norm dominated by large layers → clip coefficient tiny → Newton-Schulz degenerates

---

## 1. ChainedOptimizer Architecture

```
★★★★★★★★★ Megatron ChainedOptimizer design:

Purpose:
  → Chain multiple optimizers for mixed training (e.g., Muon for attention, Adam for MLP)
  → Each sub-optimizer operates on its own parameter group
  → BUT: gradient clipping uses a SINGLE global norm across ALL groups

★★★★★★★★★ The clipping flow:
  1. Compute global_grad_norm = norm(gradients_from_ALL_groups)
  2. Compute clip_coefficient = clip_grad / global_grad_norm
  3. Apply clip_coefficient to ALL gradients (including Muon groups)
  4. Step each sub-optimizer with clipped gradients

★★★★★★★★★ Why this is WRONG for Muon:
  → Muon discards gradient MAGNITUDE → only uses direction
  → Newton-Schulz spectral normalization preserves orthogonality
  → BUT: clip_coefficient scales magnitude BEFORE Muon sees it
  → When clip_coefficient is tiny → near-zero vectors → degenerate update
  → ★★★★★★★★ Scale-invariant optimizer + scale-change (clipping) = fundamental contradiction!
```

---

## 2. Positive-Feedback Stall Mechanism

```
★★★★★★★★★ The stall loop (same as DeepSpeed #7776):

Step 1: Large global grad_norm (e.g., 5e7 from Adam layers)
  → Adam layers: gradients are large but OK (Adam adapts lr per-param)
  → Muon layers: gradients contribute to global norm → norm grows

Step 2: Tiny clip coefficient c = clip_grad / grad_norm
  → c = 1.0 / (5e7 + 1e-6) ≈ 2e-8
  → All gradients scaled by c → near-zero

Step 3: Newton-Schulz on near-zero vectors → degenerate
  → G_clipped = G * c → magnitude ≈ 0
  → F.normalize(eps=1e-7) on near-zero vectors → unstable
  → Orthogonalization DEGENERATES → non-orthogonal update

Step 4: Layers stop updating → gradients don't improve
  → Degenerate update → model doesn't learn → loss plateaus
  → Stalled layers → their gradients stay large (no improvement)

Step 5: grad_norm grows further → c shrinks further
  → More layers collapse → more large gradients → positive feedback

★★★★★★★★★ Result: PERMANENT STALL → training stuck → loss flat → model useless

★★★★★★★★★ Minimal repro (from verified GitHub issue):
  → clip_grad=1.0, grad_norm=5e7 → c≈2e-8 → stall
  → Even moderate grad_norm (1e4) → c≈1e-4 → significant distortion

★★★★★★★★★ Training-level repro (from issue, fixed 32-example overfit batch):
  → clip_grad=1.0: loss 0.596→0.671→0.583 → stalls ~0.5, never overfits
    → grad_norm 7.5e7 → 2.2e11 (positive feedback growth)
  → clip_grad=0:   loss 0.596→0.530→0.019 → clean overfit ✅
  → ★★★★★★★★ Lowering Newton-Schulz eps 1e-7→1e-30 also unblocks → confirms clip→NS-eps interaction

★★★★★★★★★ Related: NVIDIA-NeMo/Emerging-Optimizers #229/#230:
  → Silent Newton-Schulz degeneration → looks completely normal in forward/loss
  → The two bugs interact → clip pushes grads below NS eps floor → silent collapse
```

---

## 3. Suggested Fixes from Issue

```
★★★★★★★★★ Fix option 1: Skip global norm-clipping for Muon groups (BEST):
  → Compute separate grad_norms for each sub-optimizer
  → Muon group: no clipping (or clip_grad=0) → Muon sees raw gradients
  → Adam group: clip at 1.0 → normal clipping behavior
  → ★★★★★★★★ This is the UNIVERSAL fix → per-optimizer-group clipping

★★★★★★★★★ Fix option 2: Default clip_grad=0 for dist_muon:
  → When ChainedOptimizer includes a dist_muon sub-optimizer
  → Automatically disable clipping for that sub-optimizer
  → Simple config-level fix → but doesn't handle Muon+Adam mixed

★★★★★★★★★ Fix option 3: Clamp clip coefficient with a floor:
  → clip_coeff = max(clip_grad / grad_norm, floor)
  → Floor prevents near-zero coefficients → but arbitrary
  → ★★★★★★★★ Suboptimal: still scales Muon inputs → just less extreme

★★★★★★★★★ Universal fix pattern (same as DeepSpeed):
  → Per-optimizer-group gradient clipping: Muon=0, Adam=1.0
  → Clip AFTER Muon step (if needed at all) → never BEFORE
  → Or: separate global norms per optimizer → no cross-contamination
```

---

## 4. Cross-Framework Comparison

```
★★★★★★★★★ SAME bug found in 3 independent discoveries:

DeepSpeed #8068 (OPEN):
  → gradient_clipping default 0→1.0 proposed → BUT enables the bug for Muon!
  → Fix: MUST set gradient_clipping explicitly → Muon groups: 0, Adam groups: 1.0

DeepSpeed #7776 (OPEN since Jan 12):
  → orthogonalization-before-clipping is WRONG
  → ZeRO clipping: applied BEFORE optimizer step → clips BEFORE Muon → WRONG
  → Fix: clip AFTER Muon step → or exclude Muon groups from global norm

Megatron #5394 (OPEN, June 17):
  → ChainedOptimizer single global norm → Muon degenerates
  → Same root cause: global clipping applied to scale-invariant optimizer
  → Fix: per-optimizer-group clipping → Muon=0, Adam=1.0

★★★★★★★★★ Cross-framework pattern:
  → All 3 bugs = SAME root cause → global gradient clipping applied to Muon groups
  → DeepSpeed: 2 bugs (#8068 default + #7776 ordering)
  → Megatron: 1 bug (#5394 ChainedOptimizer)
  → ★★★★★★★★ Scale-invariant optimizers MUST NOT be globally clipped!

★★★★★★★★★ Why this is UNIQUE to Muon (not Adam):
  → Adam: gradient magnitude preserved → clipping scales all uniformly → OK
  → Adam: adapts learning rate per-parameter → can handle scaled gradients
  → Muon: DISCARDS magnitude → only direction matters → clipping changes direction
  → Muon: Newton-Schulz spectral normalization → needs sufficient magnitude for stability
  → ★★★★★★★★ Fundamental insight: scale-invariant + scale-change = contradiction!
```

---

## 5. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 specific:

Current safe config (DeepSpeed ZeRO-2+CPU_Adam):
  → No Muon → gradient_clipping=1.0 → safe → no clipping bug
  → CPU_Adam handles optimizer states on CPU → 24GB GPU fits
  → ★★★★★★★★ RECOMMENDED → simplest and proven

If using Megatron with Muon+Adam:
  → MUST disable global clipping for Muon groups
  → MUST use per-optimizer-group clipping: Muon=0, Adam=1.0
  → ★★★★★★★★ ChainedOptimizer MUST NOT apply single global norm to Muon

★★★★★★★★★ Muon CPU offload:
  → DeepSpeed: BLOCKED (#7939 closed without merge)
  → Megatron: #5219 close to merge → single-GPU Muon crash fix → essential
  → ★★★★★★★★ Even with Muon CPU offload → MUST disable clipping for Muon groups
```

---

## Key Findings Summary

★★★★★★★★★ #5394: ChainedOptimizer global clipping stalls Muon → same pattern as DeepSpeed #8068/#7776
★★★★★★★★★ Root cause: single global grad_norm across ALL sub-optimizers → clip coefficient tiny → Newton-Schulz degenerates
★★★★★★★★★ Positive-feedback stall: global norm → tiny c → near-zero → degenerate → more large grads → smaller c → collapse
★★★★★★★★★ Universal fix: per-optimizer-group clipping → Muon=0, Adam=1.0 → clip AFTER step for Muon
★★★★★★★★★ Scale-invariant optimizers MUST NOT be globally clipped → fundamental insight
★★★★★★★★★ Cross-framework: 3 independent discoveries → same root cause → per-group clipping solves all
★★★★★★★★★ RTX 4090: CPU_Adam safe (no Muon) → Megatron+Muon MUST disable clipping per-group

---

## References

- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394
- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/pull/8068
- DeepSpeed #7776: https://github.com/microsoft/DeepSpeed/pull/7776
- Cross-framework Muon clipping bug: notebook/fundamentals/cross-framework-muon-gradient-clipping-bug.md
- Muon optimizer source reading: notebook/projects/deepspeed-muon-optimizer-source-reading.md
- ZeRO safety checker: tools/deepspeed_zero_safety_checker.py
