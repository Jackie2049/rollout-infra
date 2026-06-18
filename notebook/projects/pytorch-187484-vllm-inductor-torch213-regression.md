# PyTorch #187484 — vLLM w8a8 Block-FP8 Inductor Regression on torch 2.13

> 2026-06-18 | NEW CRITICAL regression | OPEN | 150+ DeepGEMM test failures
> ★★★★★★★★ Building vLLM against torch 2.13 + triton 3.7.1 → Inductor AssertionError
> ★★★★★★★★ Same vLLM commit PASSES on torch 2.12 → regression in torch 2.13
> ★★★★★★★★ Blocks vLLM #45731 (torch 2.13 upgrade) → directly impacts P9 Inductor SM89 work

---

## 1. Bug Details

```
★★★★★★★★★ PyTorch #187484 (opened June 16):

Title: [vllm] [2.13 regression][Inductor] backend='inductor' raised bare AssertionError
  compiling w8a8 block-fp8 matmul

★★★★★★★★★ Impact:
  → 150 failed tests in DeepGEMM test suite
  → 17 failed tests in FusedMoE test suite
  → Same vLLM commit PASSES on torch 2.12
  → Regression in torch 2.13 Inductor compilation path

★★★★★★★★★ Error:
  → Inductor backend='inductor' raised bare AssertionError
  → Traceback bottoms out in codecache.py load_with_key
  → Building vLLM with torch.compile + w8a8 block-FP8 matmul
  → ★★★★★★★★ AssertionError with NO message → hard to diagnose!

★★★★★★★★★ Related to:
  → vLLM #45731 (torch 2.13 upgrade DRAFT) → BLOCKED until this is fixed
  → Umbrella issue #187473 → broader torch 2.13 Inductor regressions
  → P9 Inductor SM89 Fusion Guard → #184119 → torch 2.13 changes Inductor behavior
  → ★★★★★★★★ torch 2.13 Inductor changes may affect SM89 fusion patterns!
```

---

## 2. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 (SM89) implications:

1. vLLM on RTX 4090 uses torch.compile for w8a8 block-FP8:
   → Inductor compilation → may hit same assertion error
   → ★★★★★★★★ MUST: use enforce_eager=True on RTX 4090 → skip torch.compile → avoid regression
   → P9 Inductor Guard (#184119) would prevent problematic fusions on SM89

2. torch 2.13 upgrade timing:
   → vLLM #45731 (torch 2.13 DRAFT) → BLOCKED by #187484
   → vLLM currently on torch 2.12 → safe
   → ★★★★★★★★ DON'T upgrade to torch 2.13 until #187484 resolved!

3. P9 Inductor Guard relevance:
   → #184119 (SM89 fusion guard) → prevents fp8→bf16 prologue fusion on pre-sm90
   → torch 2.13 Inductor changes may ADD new problematic fusion patterns on SM89
   → ★★★★★★★★ Need: validate P9 guard works on both torch 2.12 AND 2.13

★★★★★★★★★ Current RTX 4090 recommendation:
  → Stay on torch 2.12 → NOT upgrade to 2.13
  → Use enforce_eager=True → skip torch.compile
  → When P9 guard merges → can selectively enable torch.compile for safe ops
```

---

## Key Findings Summary

★★★★★★★★★ #187484: vLLM w8a8 block-FP8 Inductor breaks on torch 2.13 → 150+ failures
★★★★★★★★★ Same vLLM commit PASSES on torch 2.12 → regression in 2.13
★★★★★★★★★ Bare AssertionError in Inductor codecache.py → no message → hard to debug
★★★★★★★★★ Blocks vLLM #45731 (torch 2.13 upgrade) → can't upgrade yet
★★★★★★★★★ RTX 4090: stay on torch 2.12 → use enforce_eager=True → skip torch.compile
★★★★★★★★★ P9 Inductor Guard: must validate on both torch 2.12 AND 2.13 when it merges

---

## References

- PyTorch #187484: https://github.com/pytorch/pytorch/issues/187484
- PyTorch #187473: umbrella torch 2.13 Inductor regressions
- PyTorch #184119: SM89 fp8→bf16 prologue fusion guard (P9)
- vLLM #45731: torch 2.13 upgrade DRAFT
- RTX 4090 Inductor guard: notebook/fundamentals/p9-fusion-guard-integration-path-synthesis.md
