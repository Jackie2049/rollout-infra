# Cross-Framework Muon Gradient Clipping Bug Analysis — DeepSpeed + Megatron

> 2026-06-18 | Cross-framework bug pattern | Same bug found in 3 independent discoveries
> ★★★★★★★★ DeepSpeed #8068: gradient_clipping default 0→1.0 → MUST set explicitly for GRPO
> ★★★★★★★★ DeepSpeed #7776: orthogonalization-before-clipping is WRONG → Muon+GRPO blocker
> ★★★★★★★★ Megatron #5394: ChainedOptimizer global clipping stalls Muon → positive-feedback collapse
> ★★★★★★★★ All 3: global gradient clipping applied to Muon groups → orthogonalization degenerates

---

## 1. The Bug Pattern: Global Clipping + Muon Orthogonalization

```
★★★★★★★★★ Fundamental conflict:

Muon optimizer design:
  → Newton-Schulz spectral normalization → scale-invariant by construction
  → Discards gradient magnitude → only uses gradient direction
  → Update = muon_step(G) where muon_step preserves orthogonality

Global gradient clipping:
  → Compute global grad_norm across ALL parameters (including Muon groups)
  → Apply clip coefficient c = clip_grad / grad_norm to ALL gradients
  → When grad_norm is large → c is tiny → gradients scaled to near-zero

★★★★★★★★★ The conflict:
  → Muon DISCARDS magnitude → but global clipping SCALES magnitude before Muon sees it
  → When c is tiny → Newton-Schulz F.normalize(eps=1e-7) → near-zero vectors → non-orthogonal update
  → Result: orthogonalization DEGENERATES → layers stop updating → stall

★★★★★★★★★ Positive-feedback stall loop:
  1. Global grad_norm is large (e.g., 5e7)
  2. clip coefficient c = 1.0 / (5e7 + 1e-6) ≈ 2e-8 (near-zero)
  3. Muon sees G*c instead of G → magnitude near-zero
  4. Newton-Schulz on near-zero vectors → degenerate (non-orthogonal) update
  5. Layers stop updating → their gradients don't improve
  6. grad_norm grows further → c shrinks further → more layers collapse
  7. POSITIVE FEEDBACK → training stalls permanently
```

---

## 2. Three Independent Discoveries

```
★★★★★★★★★ DeepSpeed #8068 (OPEN, gradient_clipping default):
  → gradient_clipping default = 0.0 (disabled) → silently no clipping
  → Proposed change: default → 1.0 → but this would ENABLE the bug!
  → ★★★★★★★★ MUST set gradient_clipping explicitly for GRPO:
    → Muon groups: gradient_clipping = 0 (disable) or exclude from global norm
    → Adam groups: gradient_clipping = 1.0 (normal)
    → Mixed Muon+Adam: separate clipping per optimizer group

★★★★★★★★★ DeepSpeed #7776 (OPEN since Jan 12, stale):
  → orthogonalization-before-clipping is WRONG
  → Order matters: if you clip AFTER orthogonalization → OK (Muon already processed)
  → If you clip BEFORE orthogonalization → degenerate (near-zero input to Newton-Schulz)
  → DeepSpeed ZeRO clipping: applied BEFORE optimizer step → clips BEFORE Muon → WRONG
  → ★★★★★★★★ Fix: clip AFTER Muon → or exclude Muon groups from global norm

★★★★★★★★★ Megatron #5394 (OPEN, June 17):
  → ChainedOptimizer computes single global grad_norm across ALL chained sub-optimizers
  → Muon+Adam chained → global norm dominated by large layers → clip coefficient tiny
  → Newton-Schulz orthogonalization degenerates on near-zero vectors
  → Minimal repro: clip_grad=1.0, grad_norm=5e7 → c≈2e-8 → stall
  → Suggested fixes:
    → Skip global norm-clipping for orthogonalizing/Muon groups
    → Default clip_grad=0 for dist_muon
    → Clamp clip coefficient with a floor

★★★★★★★★★ Cross-framework pattern:
  → All 3 bugs = same root cause → global gradient clipping applied to Muon groups
  → DeepSpeed: 2 bugs (#8068 default + #7776 ordering)
  → Megatron: 1 bug (#5394 ChainedOptimizer)
  → All affect GRPO training with Muon on RTX 4090
```

---

## 3. RTX 4090 GRPO Implications

```
★★★★★★★★★ RTX 4090 GRPO + Muon implications:

Current recommended config (DeepSpeed ZeRO-2):
  → gradient_clipping: MUST set explicitly to 1.0 for Adam
  → If using Muon: MUST disable clipping for Muon groups (gradient_clipping: 0)
  → If using Muon+Adam mixed: MUST separate clipping per optimizer

★★★★★★★★★ Safe configurations for RTX 4090:

Config 1: ZeRO-2 + CPU_Adam (★★★★★★★★★ RECOMMENDED):
  → No Muon → gradient_clipping: 1.0 → safe → no clipping bug
  → CPU_Adam handles optimizer states on CPU → 24GB GPU fits

Config 2: ZeRO-2 + Muon (★★★★ BLOCKED):
  → Muon CPU offload BLOCKED (#7939) → NOT available on ZeRO-2
  → If available: MUST gradient_clipping: 0 for Muon groups → or clipping bug stalls
  → ★★★★★★★★ #7939 closed without merge → Muon+CPU_Adam NOT viable → CPU_Adam ONLY

Config 3: ZeRO-2 + Muon+Adam mixed (★★★★ BLOCKED):
  → Same CPU offload blocker
  → If available: MUST clip Adam groups at 1.0, Muon groups at 0 → separate per-group clipping
  → ★★★★★★★★ gradient_clipping config must be per-optimizer-group, not global

★★★★★★★★★ Key takeaway: for RTX 4090 GRPO today:
  → Use CPU_Adam (NOT Muon) → gradient_clipping: 1.0 → safe and proven
  → Muon is BLOCKED by 3 independent bugs + CPU offload blocker → NOT viable on RTX 4090
  → When Muon bugs are fixed and CPU offload is available → revisit with per-group clipping
```

---

## 4. Fix Options Across Frameworks

```
★★★★★★★★★ Fix options per framework:

DeepSpeed:
  → #8068: change default 0→1.0 → BUT this would ENABLE the bug for Muon!
  → #7776: clip AFTER Muon step → or exclude Muon groups from global norm
  → Better: per-optimizer-group gradient_clipping → Muon=0, Adam=1.0
  → Current workaround: gradient_clipping: 1.0 + CPU_Adam (no Muon) → safe

Megatron:
  → #5394: skip global norm-clipping for orthogonalizing groups
  → Or: default clip_grad=0 for dist_muon
  → Or: clamp clip coefficient with a floor
  → ★★★★★★★★ Same root cause → same fix pattern → per-group clipping

verl:
  → No known Muon clipping bug → verl uses separate optimizer configs
  → bypass_mode eliminates ref model → no separate clipping needed
  → ★★★★★★★★ verl's architecture avoids this bug by design

★★★★★★★★★ Universal fix pattern:
  → Per-optimizer-group gradient clipping: Muon groups = 0 (disabled), Adam groups = 1.0
  → Clip AFTER optimizer step for Muon (if needed) → never BEFORE
  → Or: separate global norms per optimizer → no cross-contamination
```

---

## 5. Comparison with Other Optimizer Clipping Patterns

```
★★★★★★★★★ Why this is UNIQUE to Muon:

Adam/AdamW:
  → Gradient magnitude preserved → clipping scales all gradients uniformly
  → Clipping before Adam → OK → Adam adapts learning rate per-parameter
  → Global clipping → safe → all parameters scaled proportionally

Muon:
  → Gradient magnitude DISCARDED → only direction matters
  → Clipping BEFORE Muon → scales direction vectors → near-zero → degenerate
  → Global clipping → WRONG → near-zero vectors → non-orthogonal update

★★★★★★★★★ The fundamental insight:
  → Adam is scale-adaptive → clipping is safe → Adam adjusts lr per-param
  → Muon is scale-invariant → clipping is DANGEROUS → Muon discards scale
  → Applying scale change (clipping) to scale-invariant optimizer → contradiction!
  → ★★★★★★★★ Scale-invariant optimizers MUST NOT be globally clipped!

★★★★★★★★★ MuonClip (from Kimi K2) as alternative:
  → Kimi K2 paper proposes Muon-specific clipping → preserves direction
  → Clips per-matrix norm BEFORE orthogonalization → but with floor
  → Or clips AFTER orthogonalization → preserves update magnitude
  → ★★★★★★★★ MuonClip is designed FOR Muon → not global clipping
```

---

## Key Findings Summary

★★★★★★★★★ 3 independent discoveries of SAME bug: global clipping + Muon orthogonalization conflict
★★★★★★★★★ DeepSpeed #8068 (default) + #7776 (ordering) + Megatron #5394 (ChainedOptimizer) → same pattern
★★★★★★★★★ Positive-feedback stall: global norm → tiny clip coefficient → near-zero vectors → degenerate
★★★★★★★★★ RTX 4090: CPU_Adam (NOT Muon) → gradient_clipping: 1.0 → safe → Muon BLOCKED by 3 bugs + #7939
★★★★★★★★★ Universal fix: per-optimizer-group clipping → Muon=0, Adam=1.0 → clip AFTER step for Muon
★★★★★★★★★ Scale-invariant optimizers MUST NOT be globally clipped → fundamental insight

---

## References

- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/pull/8068
- DeepSpeed #7776: https://github.com/microsoft/DeepSpeed/pull/7776
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394
- DeepSpeed Muon blockers: notebook/projects/deepspeed-v0.19.2-blockers-muon-cpu-offload-gap.md
- Muon source reading: notebook/projects/deepspeed-muon-optimizer-source-reading.md
- ZeRO safety checker: tools/deepspeed_zero_safety_checker.py
