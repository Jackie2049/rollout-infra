# Megatron #5799: MLA YARN RoPE cos/sin Dim Mismatch Guard

## Overview
- **PR**: NVIDIA/Megatron-LM #5799 by CodersAcademy006 (Srijan Upadhyay), July 14, 2026
- **Status**: Draft, 0 human reviews, `community-request` label
- **Scope**: +4 lines (2 assertions × 2 functions) in `fused_mla_yarn_rope_apply.py`
- **Severity**: ★★★★★★★★ — directly addresses the 11th DSV4 failure pattern (#5317)

## The Bug: Silent OOB Read in Triton Kernels

### What Happens
The fused MLA YARN RoPE Triton kernels (`ApplyMLARotaryEmbQ.forward` and `ApplyMLARotaryEmbKV.forward`) index into `cos`/`sin` tensors using:
```python
tl.load(COS + token_idx * emb_dim + ...)
```
They stride through the buffer assuming each token's entry occupies exactly `emb_dim` elements. If a caller supplies a `cos`/`sin` tensor whose last dimension is **smaller than `emb_dim`**, the pointer arithmetic reads past the allocated buffer boundary.

### Why It's Silent
This is not a shape mismatch at the Python level — PyTorch won't catch it because the tensors are passed as opaque pointers to the Triton JIT kernel. The OOB read manifests as "arbitrary garbage" inside the kernel, which surfaces as **nondeterministic NaN far downstream** — making it very expensive to debug.

### Real-World Impact
- **Hit during DeepSeek-V4 SFT** with `apply_rope_fusion=True`
- Nondeterministic NaN at **iteration 2** — the NaN appears long after the OOB read
- Debugging requires tracing NaNs backward through the entire model
- Same pattern family as the 11th DSV4 failure documented in Megatron #5317 (apply_rope_fusion NaN)

## The Fix: Assert at Call Site

```python
assert cos.shape[-1] == emb_dim
assert sin.shape[-1] == emb_dim
```
Added next to existing `cos.is_contiguous()` / `sin.is_contiguous()` checks in both Q and KV forward passes. Raises `AssertionError` immediately when the rotary cache is misconfigured, rather than producing silent OOB reads.

Author notes the guard was "suggested in the DeepSeek-V4 tracking issue" (#4468) by @Meirtz.

## DSV4 Failure Family Connection

This is the **11th DSV4 failure pattern** (#5317). The connection:
- **#5317 (original)**: Triton in-place `rotary_fwd_q_kernel` bypasses autograd version counter → NaN at iter 2
- **#5799 (this PR)**: Misconfigured rotary cache width → OOB read → NaN at iter 2
- **Both** manifest as nondeterministic NaN at iteration 2 with `apply_rope_fusion=True`
- **Both** require `apply_rope_fusion=False` as workaround until fix is merged

The earlier analysis in `megatron-5317-apply-rope-fusion-nan-analysis.md` identified the autograd version counter issue. This PR shows there's a **second independent failure mode** in the same code path.

## Why Still Draft
- copy-pr-bot: "requires additional validation before NVIDIA runners"
- No CI triggered yet
- Author needs to respond to bot requirements

## Monitoring Path
- Needs `/ok to test` from NVIDIA CI
- Needs human reviewer
- Small fix (+4 lines) → fast review IF prioritised
- Connection to #4468 tracking issue may accelerate

## Cross-Framework
- Same Triton kernel pattern (OOB memory access due to shape assumption) appears in:
  - vLLM #48590 (lora_expand block_n=128 NaN — OOB on boundary N-tile)
  - SGLang DSV4 kernels (recurrent state OOB)
- Pattern: **P1 (CUDA Stream Use-After-Free) variant** — memory safety at kernel boundary
