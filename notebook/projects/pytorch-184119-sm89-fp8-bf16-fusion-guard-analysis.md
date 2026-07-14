# PyTorch PR #184119: SM89 fp8→bf16 Prologue Fusion Guard — P9 Validation

Based on PR #184119 by jansel (PyTorch Inductor team lead).  
Link: https://github.com/pytorch/pytorch/pull/184119

## Summary

PR #184119 blocks standard Triton matmul prologue fusion when a pre-sm90 CUDA
target would fuse a float8 read into a bf16 dot operand. This is a targeted
implementation of the P9 Fusion Guard concept.

## The Problem

Triton's `tl.dot` on sm90 (Hopper) supports native bf16 tensor cores that can
directly consume fp8 inputs via conversion instructions. On sm89 (Ada Lovelace),
bf16 tensor cores cannot consume fp8 inputs directly — the fp8 must be
materialized as bf16 first. When Inductor fuses the fp8→bf16 conversion into
the Triton MM template prologue, Triton emits sm90-only conversion instructions
that produce incorrect results on sm89.

## The Fix

In `torch/_inductor/scheduler.py`, ~4 lines added to prologue-fusion selection:

```python
def _is_pre_sm90_fp8_to_bf16_mm_template_prologue():
    # 1. Is this a standard Triton MM template?
    # 2. Is device pre-sm90?
    # 3. Is final prologue output bf16?
    # 4. Does prologue read an fp8 input?
    return all(conditions)
```

When all 4 conditions are met, the fp8→bf16 conversion is materialized as a
separate pointwise kernel before the bf16 matmul on pre-sm90 CUDA. The matmul
template remains available — only this specific fusion decision is blocked.

## P9 Thesis Validation

| Aspect | P9 Proposal | PR #184119 Implementation |
|--------|-------------|--------------------------|
| **Scope** | General SM<90 fusion guard | fp8→bf16 prologue fusion only |
| **Location** | `choices.py` (WhyNoFuse) | `scheduler.py` (prologue-fusion selection) |
| **Mechanism** | Reject fusion in inductor | Block prologue fusion in scheduler |
| **Granularity** | All fusion types | Specific fusion pattern only |
| **Status** | Draft concept | Merged via ghstack |

**Validation**: The PyTorch Inductor team independently identified and fixed the
same SM<90 fusion safety issue that P9 proposed. The fix is more targeted —
blocking only the known-bad pattern rather than all fusion — which is the
correct engineering approach.

## P9 Remaining Opportunity

The general SM<90 Fusion Guard (as a WhyNoFuse diagnostic) still has value:

1. **User-facing diagnostics**: "Why was my fp8 kernel slow on RTX 4090?" → "Prologue fusion was blocked because SM<90 doesn't support fp8→bf16 in Triton MM templates"
2. **fp8→fp16 path**: Review asked if fp16 (not just bf16) could also trigger sm90-only instructions
3. **Cross-framework relevance**: vLLM/SGLang users on RTX 4090 (SM89) hit the same issue — a PyTorch-level diagnostic helps debug deployment performance

## Test

A regression test in `test/inductor/test_fp8.py` simulates an sm89 target and
verifies the generated dot kernel does NOT contain `.to(tl.bfloat16)` or `*fp8`
patterns, confirming the prologue is materialized separately.

## Connections

- Validates P9 thesis: https://github.com/jackie2049/pytorch PR #1
- Same SM89 constraints affect: vLLM #46085 (aot_eager piecewise), SGLang #28618/#28620 (SM89 DSV4-Flash-FP8)
- Related: PyTorch #187636 (autotune_at_compile_time flips to False by default → reduces SM89 batch-dependent fusion risk)

