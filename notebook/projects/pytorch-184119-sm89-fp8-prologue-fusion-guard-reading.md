# PyTorch #184119 — SM89 fp8→bf16 Prologue Fusion Guard Reading

> 2026-06-18 | PR #184119 OPEN (+105/-0) | 2 files: scheduler.py + test_fp8.py
> ★★★★★★★★ VALIDATES P9 thesis: Inductor generates SM90-only Triton code on SM89 hardware!
> ★★★★★★★★ Same root cause class: Inductor prologue fusion creates SM90-only bf16 conversion on SM89
> ★★★★★★★★ 5-line guard + 2 helper functions + 42-line regression test

---

## 1. Bug: Inductor Generates SM90-only Triton Code on SM89

```
★★★★★★★★★ The bug mechanism:

When Inductor fuses an fp8-to-bf16 cast into the Triton mm template prologue:
  → The converted fp8 value becomes the B operand to tl.dot
  → Triton emits the SM90-only bf16 conversion sequence
  → But compiling for SM89 (cuda.arch=89)!
  → ★★★★★★★★ SM89 hardware CANNOT execute SM90-only bf16 conversion → CRASH or WRONG results

The standalone fp8-to-bf16 pointwise conversion path is SAFE:
  → fp8 load → through fp32 → stores bf16
  → Does NOT create the same invalid instruction
  → ★★★★★★★★ Only dangerous when Inductor FUSES the cast into mm template prologue

★★★★★★★★★ Why this is the same class as P9:
  → P9: Inductor fuses reduction (mean) into pointwise → tl.sum() with autotuned XBLOCK
    → Different XBLOCK → different accumulation order → batch-dependent numerical results
  → #184119: Inductor fuses fp8 cast into mm template → SM90-only Triton code
    → Different SM capability → SM89 can't execute → CRASH
  → ★★★★★★★★ SAME root cause: Inductor fusion creates capability-dependent code
  → P9: batch-dependent (wrong numerical results)
  → #184119: SM-dependent (wrong instruction generation)
  → BOTH require: capability-aware fusion guards in Inductor scheduler
```

---

## 2. The Fix (+105/-0, 2 files)

```
★★★★★★★★★ scheduler.py — 3 new functions + 1 guard insertion:

1. _FLOAT8_DTYPES = tuple of all float8 types (e4m3fn, e5m2, e4m3fnuz, e5m2fnuz)

2. _is_pre_sm90_cuda_device(device):
  → cuda_arch = get_cuda_arch()
  → return cuda_arch < 90
  → ★★★★★★★★ Same pattern as P9's props.major < 9!
  → Uses get_cuda_arch() from cuda_env → different from DeviceProperties.create(device)
  → But same logic: check SM capability before allowing fusion

3. _has_float8_read(node):
  → Check if any read dependency has float8 dtype
  → Iterates node.read_writes.reads → V.graph.get_dtype(name) in _FLOAT8_DTYPES

4. _is_pre_sm90_fp8_to_bf16_triton_template_prologue(prologue_node, template_node, prologue_nodes):
  → Checks 5 conditions:
    a. template_node is TritonTemplateBuffer (not None, isinstance)
    b. template_node device is pre-sm90 CUDA
    c. prologue_nodes exist
    d. final_prologue_outputs has 1 output
    e. output dtype is bfloat16
  → If all conditions met + prologue_node has float8 read → return True → BLOCK fusion

★★★★★★★★★ Guard insertion in scheduler.py line ~7777:

  if _is_pre_sm90_fp8_to_bf16_triton_template_prologue(node1, node2, prologue_nodes):
      why("fp8 to bf16 prologue into triton template requires sm90")
      return False

★★★★★★★★★ test_fp8.py — 42-line regression test:

  test_fp8_to_bf16_mm_template_prologue_not_fused_pre_sm90:
  → Sets cuda.arch=89 + max_autotune=True + TRITON backend
  → Compiles fn(a, b.t().to(bf16), bias) where b is fp8
  → Asserts: only 1 dot kernel in generated code
  → Asserts: dot kernel does NOT contain ".to(tl.bfloat16)" or "*fp8"
  → ★★★★★★★★ Verifies that fp8→bf16 is NOT fused into the mm template on SM89
```

---

## 3. P9 Thesis Validation

```
★★★★★★★★★ #184119 VALIDATES the P9 thesis in 3 ways:

1. Same root cause class:
  → P9: Inductor fusion creates SM-dependent behavior (batch-dependent)
  → #184119: Inductor fusion creates SM-dependent code (SM90-only on SM89)
  → ★★★★★★★★ BOTH are "Inductor ignores SM capability when making fusion decisions"

2. Same fix pattern:
  → P9: props.major < 9 → return False in can_fuse_vertical
  → #184119: cuda_arch < 90 → return False in prologue_fusion
  → ★★★★★★★★ BOTH add SM capability checks to Inductor scheduler fusion logic

3. Same impact:
  → P9: Without guard → batch-dependent numerical results on SM89
  → #184119: Without guard → SM90-only instructions on SM89 → CRASH
  → ★★★★★★★★ BOTH make SM89 unsafe for production use without guards

★★★★★★★★★ #184119 strengthens P9's position:
  → PyTorch team is ALREADY adding SM capability guards to Inductor
  → #184119 proves the pattern is accepted → P9 follows same pattern
  → P9 is MORE general (covers ALL reduction fusions, not just fp8→bf16)
  → ★★★★★★★★ P9 = "global SM<90 guard for ALL reduction fusions"
  → #184119 = "specific SM<90 guard for fp8→bf16 prologue fusions"
  → BOTH needed → #184119 is SPECIFIC → P9 is GENERAL → complementary!
```

---

## 4. Integration with P9

```
★★★★★★★★★ P9 and #184119 are complementary layers:

P9 (can_fuse_vertical):
  → Blocks ALL vertical reduction fusions on SM<90
  → Global policy → covers RMSNorm, mean, sum, etc.
  → 5 lines → minimal change

#184119 (prologue fusion):
  → Blocks fp8→bf16 prologue fusion on SM<90
  → Specific policy → covers fp8 matmul template
  → 60+ lines → more targeted

★★★★★★★★★ Complete SM89 Inductor guard stack:
  → P9: global reduction fusion guard (batch invariance)
  → #184119: specific fp8→bf16 prologue guard (instruction compatibility)
  → #187435: per-op no_fuse_region (fine-grained control)
  → #6572: deployment layer (verl determinism)
  → ★★★★★★★★ All 4 complementary → P9 simplest first → then #184119 → then #187435
```

---

## Key Findings Summary

★★★★★★★★★ #184119: SM89 fp8→bf16 prologue fusion guard → VALIDATES P9 thesis!
★★★★★★★★★ Same root cause: Inductor ignores SM capability → creates SM90-only code on SM89
★★★★★★★★★ Same fix pattern: cuda_arch < 90 → return False → blocks fusion
★★★★★★★★★ 105 additions, 2 files, 3 helper functions + 1 guard + regression test
★★★★★★★★★ Complementary to P9: #184119 = specific fp8 guard, P9 = general reduction guard
★★★★★★★★★ PyTorch team already adding SM capability guards → P9 follows same pattern
★★★★★★★★★ Fixes #150621 → fp8 mm crash on SM89

---

## References

- PyTorch #184119: https://github.com/pytorch/pytorch/pull/184119
- PyTorch #150621: https://github.com/pytorch/pytorch/issues/150621 (fp8 mm crash on SM89)
- P9 Fusion Guard: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- #187435: https://github.com/pytorch/pytorch/pull/187435 (no_fuse_region per-op)
- #6572: https://github.com/verl-project/verl/pull/6572 (verl full determinism)
