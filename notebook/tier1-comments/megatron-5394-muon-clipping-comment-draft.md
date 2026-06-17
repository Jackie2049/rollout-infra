# Megatron #5394 Tier 1 Comment Draft — ChainedOptimizer Muon Clipping Bug

> 2026-06-18 | Tier 1 comment opportunity | Cross-framework bug pattern
> ★★★★★★★★ Source-level analysis + cross-framework comparison + suggested fix alignment

---

## Comment Draft

Great bug report — I've confirmed this pattern across two other frameworks (DeepSpeed #8068 and #7776) and can add some context.

### Cross-framework pattern: 3 independent discoveries of the SAME bug

| Framework | Issue | Mechanism | Key difference |
|-----------|-------|-----------|----------------|
| Megatron | #5394 | Clip BEFORE NS → degenerate | ChainedOptimizer forces shared OptimizerConfig → config-level fix impossible |
| DeepSpeed | #7776 | Clip BEFORE NS → degenerate | ZeRO clipping applied before optimizer step → ordering bug |
| DeepSpeed | #8068 | Default clipping disabled (0→1.0 proposed) | Would ENABLE the bug for Muon groups if changed |

All three share the same root cause: **magnitude-based global gradient clipping and Muon's scale-invariant Newton-Schulz orthogonalization are fundamentally incompatible**. Muon discards gradient magnitude by design — clipping changes the scale before Muon can process it, which is either meaningless (if vectors are large enough for NS to normalize) or harmful (if the clip coefficient pushes per-matrix gradients below NS's `F.normalize(eps=1e-7)` floor).

### Why the bug is optimizer-agnostic

The author's clarification is important: AdamW also stalls under the same clip when its `eps` floor is hit. The fix targets orthogonalizing optimizers because magnitude clipping is *semantically meaningless* for scale-invariant updates — a clean, side-effect-free skip.

### PR #5395 alignment

The `skip_grad_norm_clip` attribute approach is the right design given the ChainedOptimizer config constraint (line 914 forces `result.config = config`). A few observations:

1. The attribute is set in `__init__.py` after optimizer construction — this is the correct location since the config cannot be per-sub-optimizer
2. `getattr(optimizer, "skip_grad_norm_clip", False)` defaults to False — zero blast radius for existing optimizers
3. `grad_norm` is still computed globally for logging — good, users still see the overall norm

### Complementary fix: Emerging-Optimizers #229/#230

Lowering Newton-Schulz `eps` from `1e-7` to `1e-30` makes NS robust on small inputs. This doesn't fix the root cause (meaningless clipping applied to scale-invariant updates) but removes the silent failure mode. Both fixes should be pursued:
- #5395: skip clipping for orthogonalizing groups (root cause fix)
- Emerging-Optimizers #230: lower NS eps (robustness fix, complementary)

### Universal insight

**Scale-invariant optimizers must not be subject to magnitude-based global gradient clipping.** This applies to any optimizer that discards or normalizes gradient magnitude (Muon, spectral norm methods, etc.). Per-optimizer-group clipping (Muon=0, Adam=1.0) is the universal solution across all frameworks.

---

## References

- Cross-framework Muon clipping analysis: cross-framework-muon-gradient-clipping-bug.md
- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/pull/8068
- DeepSpeed #7776: https://github.com/microsoft/DeepSpeed/pull/7776
- Emerging-Optimizers #229: https://github.com/NVIDIA-NeMo/Emerging-Optimizers/issues/229
