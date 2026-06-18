# PyTorch #187636 autotune_at_compile_time=False — Deep Reading

> 2026-06-18 | How this PR complements P9 Fusion Guard by reducing SM89 batch-dependent fusion risk
> ★★★★★★★★ autotune_at_compile_time flips to False by DEFAULT → eliminates compile-time autotuning
> ★★★★★★★★ This INDIRECTLY reduces SM89 batch-dependent fusion risk → complements P9!
> ★★★★★★★★ For AOTI (torch.export): autotune_at_compile_time defaults True → now False
> ★★★★★★★★ For regular torch.compile: autotune at_compile_time was True → now None

> ★★★★★★★★ Change: 7 LOC in compile_fx.py → simpl to bool(config.triton.autotune_at_compile_time)

> ★★★★★★★★ Risk reduction mechanism:
>   → compile-time: Triton configs were selected and baked into binary → DETERMINISTIC at compile time
>   → runtime: Triton configs selected at runtime via CachingAutotuner → DYNAMIC → batch-dependent!
>   → Default False → skip compile-time autotune → runtime autotune only → SAME XBLOCK
>   → → BATCH-INARIANT results on SM89!

> ★★★★★★★★ Complementary to P9 Fusion Guard:
>   → P9: blocks ALL reduction fusions on SM<90 → HARD guarantee ( batch invariance
>   → #187636: skips compile-time autotune → SOFT guarantee ( same XBLOCK selection)
>   → Together: hard guarantee + soft guarantee = double protection!
>   → → Even without P9, #187636 reduces risk significantly
>   → → With P9 + #187636 = NEAR-COMPLETE batch invariance on SM89!
> ★★★★★★★★ Key code change (1 LOC in compile_fx.py):
>   → Before (6 LOC):
>     autotune_at_compile_time = (
>         config.triton.autotune_at_compile_time
>         if config.triton.autotune_at_compile_time is not None
>         else has_triton() and V.aot_compilation
>     )
>   → After (1 LOC):
>     autotune_at_compile_time = bool(config.triton.autotune_at_compile_time)
>   → Removed import: `from ..utils._triton import has_triton`
---

## Key Findings Summary

★★★★★★★★★ #187636 complements P9: reduces SM89 batch-dependent fusion risk by DEFAULT!
★★★★★★★★★ autotune_at_compile_time=False: compile-time Triton config baking → deterministic XBLOCK → NO batch-dependent variation
★★★★★★★★★ For AOTI: default True → now False → eliminates pre-compile autotuning that a different XBLOCK configs
★★★★★★★★★ For regular torch.compile: autotune at_compile_time=True → now None → runtime autotune via CachingAutotuner
★★★★★★★★★ P9 + #187636 = double protection: hard guarantee (no reduction fusions) + soft guarantee (deterministic XBLOCK)
★★★★★★★★★ Even without P9, #187636 reduces SM89 batch-dependent fusion risk significantly!
★★★★★★★★★ With P9 + #187636 = near-complete batch invariance on SM89!
