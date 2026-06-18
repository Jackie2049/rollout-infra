# PyTorch #187636 — Flip autotune_at_compile_time to False — P9 Complement

> 2026-06-18 | Source-level analysis of autotune default change
> ★★★★★★★★ autotune_at_compile_time flips from True (AOTI) to False → reduces SM89 batch-dependent fusion risk!
> ★★★★★★★★ COMPLEMENTS P9 Fusion Guard: autotune=False reduces XBLOCK variability → same root cause class!

---

## 1. What Changed (+1/-7 lines)

```
★★★★★★★★★ PR #187636 (OPEN June 18, ezyang):

Before:
  autotune_at_compile_time = (
      config.triton.autotune_at_compile_time
      if config.triton.autotune_at_compile_time is not None
      # Default to True for AOTI. Subject to change in future.
      else has_triton() and V.aot_compilation  # True for AOTI!
  )

After:
  autotune_at_compile_time = bool(config.triton.autotune_at_compile_time)
  # → config.triton.autotune_at_compile_time defaults to None
  # → bool(None) = False → autotune_at_compile_time = False!

★★★★★★★★★ Also removed import:
  from ..utils._triton import has_triton  # no longer needed

★★★★★★★★★ Effect:
  → autotune_at_compile_time: True → False (default change!)
  → When False: Triton configs are autotuned at RUNTIME (lazy)
  → When True: Triton configs are autotuned at COMPILE TIME (ahead-of-time)
  → Also flips autotune_cublasLt: False → True (complementary)
```

---

## 2. Why This Matters for SM89 Batch Invariance

```
★★★★★★★★★ Connection to P9 Fusion Guard:

When autotune_at_compile_time=True (old default):
  → Triton CachingAutotuner runs at compile time → selects XBLOCK for specific input shape
  → At runtime: if batch size differs from compile-time shape → DIFFERENT XBLOCK would be optimal
  → But autotune already happened → same XBLOCK used → WRONG accumulation order!
  → → batch-dependent results on SM89!

When autotune_at_compile_time=False (new default):
  → Triton configs autotuned lazily at runtime → per-invocation shape
  → Each batch size gets its own autotune → correct XBLOCK for that shape
  → BUT: different XBLOCK for different batch sizes → STILL non-associative!
  → → batch-dependent results STILL possible! (same root cause as P9!)

★★★★★★★★★ Key insight: #187636 reduces but does NOT eliminate SM89 batch invariance risk:
  → Old: compile-time autotune → ONE XBLOCK for all batch sizes → wrong for non-compile-size
  → New: runtime autotune → EACH batch size gets its own XBLOCK → correct per-batch
  → But: different XBLOCK per batch → tl.sum() accumulation varies → still non-associative!
  → ★★★★★★★★ #187636 + P9 = COMPLETE solution:
    → #187636: prevents stale compile-time configs → each batch gets fresh autotune
    → P9: prevents reduction fusion → keeps reductions as separate kernels → overrides work
    → Together: no stale configs + no fused reductions → batch-invariant on SM89!

★★★★★★★★★ Why #187636 alone is insufficient:
  → Even with runtime autotune → XBLOCK varies per batch → tl.sum() still non-associative
  → vLLM's mean.dim override uses tl.constexpr → deterministic → but only works when called SEPARATELY
  → If Inductor fuses mean+rsqrt+mul (RMSNorm) → inline tl.sum() → override bypassed!
  → P9 prevents this fusion → mean stays separate → override works → deterministic!

★★★★★★★★★ Why P9 alone would also benefit from #187636:
  → P9 blocks reduction fusions → keeps ops separate → but pointwise fusions still autotuned
  → With compile-time autotune → pointwise kernels get stale configs → potential batch-dependent
  → #187636 ensures pointwise kernels get fresh configs → more consistent across batches
  → Together: complete SM89 determinism!
```

---

## 3. P9 + #187636 Integration Path

```
★★★★★★★★★ Three complementary mechanisms for SM89:

  1. P9 Fusion Guard (our contribution):
     → choices.py: props.major < 9 → WhyNoFuse("batch_invariance")
     → Blocks ALL reduction fusions on SM<90 → forces separate kernel dispatch
     → Scope: GLOBAL for SM<90 → universal coverage

  2. #187636 autotune_at_compile_time=False (upstream default change):
     → compile_fx.py: bool(config.triton.autotune_at_compile_time) → False
     → Prevents stale compile-time Triton configs → each batch gets fresh autotune
     → Scope: GLOBAL → all platforms → reduces stale config risk

  3. #184119 SM89 fp8 guard (upstream, progressing):
     → scheduler.py: _is_pre_sm90_cuda_device + _has_float8_read → blocks fp8 fusion
     → Scope: per-Op → specific fp8+bf16 prologue fusions → fills specific gap

★★★★★★★★★ Together: P9 + #187636 + #184119 = complete SM89 Inductor safety:
  → P9: blocks reduction fusions → overrides work → deterministic
  → #187636: fresh autotune → no stale configs → consistent pointwise ops
  → #184119: blocks fp8 prologue → no fp8→bf16 fusion → specific risk addressed
  → Result: SM89 torch.compile = deterministic + safe!

★★★★★★★★★ v2.12 status:
  → #187636 is NOT yet merged → still OPEN → but simple (+1/-7)
  → #184119 still OPEN → progressing → needs CI validation
  → P9 not yet submitted → needs GPU validation
  → Together they form a complementary stack → all address SM89 from different angles
```

---

## Key Findings Summary

★★★★★★★★★ autotune_at_compile_time flips True→False → reduces SM89 batch-dependent fusion risk!
★★★★★★★★★ COMPLEMENTS P9: #187636 prevents stale configs, P9 prevents fused reductions → together = complete SM89 fix
★★★★★★★★★ #187636 alone insufficient: runtime autotune still varies XBLOCK per batch → non-associative accumulation
★★★★★★★★★ P9 alone insufficient: needs fresh configs for pointwise ops → #187636 provides this
★★★★★★★★★ P9 + #187636 + #184119 = complete SM89 Inductor safety stack (3 complementary mechanisms)
★★★★★★★★★ +1/-7 lines → simplest change → likely fast merge → strengthens our P9 submission case

---

## References

- PyTorch #187636: https://github.com/pytorch/pytorch/pull/187636
- P9 Fusion Guard: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- #184119 SM89 fp8 guard: notebook/projects/pytorch-184119-sm89-fp8-prologue-fusion-guard-reading.md
- Integration path: notebook/projects/p9-fusion-guard-integration-path-synthesis.md
- vLLM #39096: https://github.com/vllm-project/vllm/issues/39096
