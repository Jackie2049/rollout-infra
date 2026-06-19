# Megatron #5400 + #5401 — GDN Muon Routing (6th Blocker) + MoE z-loss CUDA Graph Fix

> 2026-06-19 | Deep Reading Note | Both PRs OPEN, community-request label
> #5400: +14/-1 (2 files), 0 reviews, DRAFT | #5401: +6/-4 (1 file), 0 reviews, OPEN
> Authors: yuchenwang3 (#5400), Baibaifan (#5401)
> ★★★★★★★★ TWO distinct failure classes in ONE day: structural Muon incompatibility + CUDA graph state mismatch
> ★★★★★★★★ Updated Muon blockers: now 6 (structural #5400, clipping #5394/#5395/#8068, infrastructure #5179, CPU offload #7939)
> ★★★★★★★★ #5401 reveals CUDA graph capture + z-loss interaction — same State Lifecycle Mismatch pattern family as DSV4 failures

---

## Table of Contents

1. PR #5400: GDN in_proj Adam Routing (6th Muon Blocker)
2. PR #5401: MoE Router z-loss + TE CUDA Graph Fix
3. Cross-Reference Analysis (#5394, #5395, #5396, #2885)
4. Updated Muon Blockers Count (6 total)
5. State Lifecycle Mismatch Pattern Family Connections
6. RTX 4090 GRPO Training Relevance
7. Key Takeaways and Action Items

---

## 1. PR #5400: GDN in_proj Adam Routing (6th Muon Blocker)

### 1.1 PR Overview

**Title**: fix(optimizer): route GatedDeltaNet in_proj to Adam instead of orthogonalizing it (Muon)

**URL**: https://github.com/NVIDIA/Megatron-LM/pull/5400

**Author**: yuchenwang3 (external contributor, same author as #5394/#5395/#5396)

**Status**: DRAFT (auto-converted by Megatron policy), OPEN, blocked merge state (needs NVIDIA vetting). 0 human reviews, 0 review comments, 2 bot comments (copy-pr-bot validation, github-actions draft conversion).

**Created**: 2026-06-18T07:33:24Z

**Stats**: +14/-1, 2 files changed, 1 commit

**Refs**: Issue #2885

### 1.2 Problem Statement

The GatedDeltaNet (GDN) `in_proj` weight packs six semantically distinct projections into a single 2D matrix:

```
in_proj_dim = qk_dim * 2 + v_dim * 2 + num_value_heads * 2

Fused components:
  q (query)         -> qk_dim           (attention query)
  k (key)           -> qk_dim           (attention key)
  v (value)         -> v_dim            (attention value)
  z/gate            -> v_dim            (output gating)
  beta              -> num_value_heads  (delta rule intensity)
  alpha             -> num_value_heads  (decay/forgetting rate)
```

Previously, Megatron had hard guard assertions preventing GDN + Muon:
```python
assert args.linear_attention_type is None, "Muon optimizer does not support linear attention type for now."
assert not args.attention_output_gate, "Muon optimizer does not support attention output gate for now."
```

These guards were **removed from main** during Muon optimizer-validation refactoring, but the routing they discussed was **never added**. This created a silent correctness regression: GDN + Muon is no longer explicitly blocked, but is silently incorrect.

The Muon routing predicate `_is_nonlinear_or_embedding` only excludes:
- Embedding/output params (`is_embedding_or_output_parameter`)
- Non-2D params (shape check)

`in_proj.weight` is 2D, untagged, not an embedding -> passes the predicate -> sent to Muon -> orthogonalized as one heterogeneous matrix -> **semantically meaningless updates**.

### 1.3 Why Orthogonalizing Heterogeneous Fusion Is Meaningless

Newton-Schulz iteration treats the entire `in_proj.weight` matrix as a single 2D object and orthogonalizes its momentum update as one matrix. This computes the orthogonal polar factor of a matrix whose row correlations mix six unrelated gradient flows:

- Query gradients have no meaningful correlation with gate gradients
- Key update directions should not be forced orthogonal to beta directions
- The result is mathematically valid (an orthogonal matrix) but **semantically incoherent** (mixing unrelated transformations)

**Contrast with standard QKV**: Standard attention also fuses q/k/v, but Megatron has `--muon-split-qkv` which:
1. Splits the weight into Q, K, V sub-blocks
2. Orthogonalizes each sub-block independently
3. Reassembles the orthogonalized sub-blocks

This works because Q/K/V are semantically related (all attention projections) and each sub-block is independently orthogonalized. `in_proj` cannot use this path because:
1. It fuses more than q/k/v -- conv, gate, beta are structurally different
2. The sub-block shapes are heterogeneous
3. The conv/gate/beta sub-blocks should NOT be orthogonalized at all
4. No `--muon-split-in-proj` infrastructure exists

### 1.4 The skip_orthogonalization Mechanism (Code Changes)

**File 1: `megatron/core/ssm/gated_delta_net.py`** (+9/-0)

```python
# After in_proj creation:
# in_proj fuses the per-head query/key/value projections together with the conv,
# gate and beta projections into a single matrix (in_proj_dim above). Orthogonalizing
# this heterogeneous fused weight as one 2D matrix (e.g. with Muon) is not meaningful,
# so flag it to fall back to the standard (Adam) optimizer, the same way fused/embedding
# params are kept out of the orthogonalizing optimizer.
# See https://github.com/NVIDIA/Megatron-LM/issues/2885
if hasattr(self.in_proj, "weight") and self.in_proj.weight is not None:
    self.in_proj.weight.skip_orthogonalization = True
```

Key design decisions:
- Tag is set **after** the module is created, not during initialization -- follows the existing pattern for `is_embedding_or_output_parameter`
- Uses `hasattr` + `is not None` guard -- handles TP-sharded models where weight may not exist on this rank
- Explicit comment references #2885 -- traces back to the original issue that identified this gap
- `out_proj` is **intentionally left on Muon** -- it is a regular 2D projection, semantically homogeneous

**File 2: `megatron/core/optimizer/emerging_optimizers.py`** (+5/-1)

```python
# Before (single-line predicate):
def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2

# After (expanded predicate):
def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return (
        getattr(param, 'is_embedding_or_output_parameter', False)
        or getattr(param, 'skip_orthogonalization', False)
        or len(param.shape) != 2
    )
```

### 1.5 Design Rationale: skip_orthogonalization vs Alternatives

The author explicitly chose `skip_orthogonalization` over alternatives:

| Approach | Pros | Cons | Chosen? |
|---|---|---|---|
| `skip_orthogonalization` attribute | Generic, semantically clear, reusable for other fused modules, no checkpoint coordination needed | Requires model code to set the attribute | YES |
| Overload `is_embedding_or_output_parameter` | Reuses existing predicate | Semantically wrong (in_proj is not an embedding), confusing for future readers | NO |
| Split-and-orthogonalize (like QKV split) | Correctly handles q/k/v sub-blocks, keeps them on Muon | More invasive, requires new split infrastructure, conv/gate/beta still need routing | NO (open to alternatives) |
| Name-based ParamKey override | No model code changes needed | Less modular, harder to maintain across model variants | NO |
| CLI/config flag | User-visible, configurable | ChainedOptimizer requires all sub-optimizers to share one OptimizerConfig, can't set per-group | NO |

The PR description notes: "Introduced a generic `skip_orthogonalization` attribute rather than overloading `is_embedding_or_output_parameter` (semantically clearer; other fused modules could reuse it). Happy to rename or switch to a name-based `ParamKey` override if you prefer."

### 1.6 Issue #2885 Resolution Context

Issue #2885 (opened January 2026) asked whether GDN + Muon was fundamentally incompatible.

**CodersAcademy006's January 2026 analysis**: "No hard theoretical incompatibility -- just an implementation limitation. Routing gated/linear-attention params to Adam would require explicit parameter tagging, split optimizer param groups, and coordinated stepping and checkpointing."

**yuchenwang3's June 2026 follow-up** (in #2885 comments, cross-referencing #5400): "The hard guards are already gone from main, but the routing was not added, so GDN + Muon is now **silently incorrect** rather than blocked. Good news: the concern about needing 'split optimizer param groups + coordinated stepping/checkpointing' is already solved -- the `default_param_overrides` / `ParamKey` mechanism that routes embeddings and non-2D params to Adam is exactly that infrastructure, and it is checkpoint-supported."

**Resolution path**: The infrastructure CodersAcademy006 thought was missing (split groups + checkpoint coordination) already exists. The fix is a small localized change, not new infrastructure.

### 1.7 Existing Muon Routing Predicate Flags (Updated Taxonomy)

| Attribute | Purpose | Origin | Added |
|---|---|---|---|
| `is_embedding_or_output_parameter` | Route embeddings/output to Adam | Model code (set on embed/output params) | Original |
| `is_qkv` | Enable per-sub-block orthogonalization | Muon kwargs config | Original |
| `len(param.shape) != 2` | Exclude non-matrix params (1D biases, norms) | Implicit shape check | Original |
| `skip_orthogonalization` | Route heterogeneous fused params to Adam | **NEW: #5400** | 2026-06-18 |

---

## 2. PR #5401: MoE Router z-loss + TE CUDA Graph Fix

### 2.1 PR Overview

**Title**: [Fix] Fix MoE router z-loss compatibility with TE CUDA Graph capture.

**URL**: https://github.com/NVIDIA/Megatron-LM/pull/5401

**Author**: Baibaifan (external contributor)

**Status**: OPEN, mergeable but blocked. 0 human review comments. 2 comments (copy-pr-bot validation, Victarry "/ok to test" CI trigger with +1 reaction). **Based against `dev` branch** (not `main` -- different from #5400 which targets `main`).

**Created**: 2026-06-18T08:47:23Z

**Stats**: +6/-4, 1 file changed, 1 commit

**File changed**: `megatron/core/transformer/moe/router.py`

### 2.2 Problem Statement

When MoE router z-loss is enabled together with TE (TransformerEngine) CUDA Graph capture for the MoE router, training fails during graph creation if `padding_mask` is None.

The root cause: the previous implementation created a CUDA tensor from a Python integer:
```python
# BEFORE (broken with CUDA graph capture):
num_local_tokens = torch.tensor(logits.shape[0], device=logits.device)
```

During CUDA graph capture, `torch.tensor(int_value, device=cuda_device)` records a **CPU-to-CUDA copy operation** in the graph. PyTorch's CUDA graph capture mechanism rejects this unless the CPU tensor is pinned (which it isn't -- the value comes from `logits.shape[0]`, a Python int).

**Why this is a CUDA graph problem**: CUDA graph capture records all GPU operations into a reusable graph. When a CPU-to-GPU transfer happens during capture, the graph would need to replay that transfer every invocation, but the CPU value may change between invocations (e.g., different batch sizes). PyTorch rejects this because:
1. The transfer is a host-to-device operation that breaks graph replayability
2. The value being transferred is data-dependent (batch size varies)
3. CUDA graphs require all inputs to be pre-allocated GPU tensors

### 2.3 The Fix (Code Changes)

**File: `megatron/core/transformer/moe/router.py`** (+6/-4)

```python
# BEFORE:
if padding_mask is not None:
    valid_mask = ~padding_mask
    z_loss_values = z_loss_values * valid_mask
    num_local_tokens = valid_mask.sum()
else:
    num_local_tokens = torch.tensor(logits.shape[0], device=logits.device)

z_loss_sum = z_loss_values.sum()
z_loss_mean = z_loss_sum / torch.clamp(num_local_tokens, min=1)

# AFTER:
if padding_mask is not None:
    valid_mask = ~padding_mask
    z_loss_values = z_loss_values * valid_mask
    num_local_tokens = valid_mask.sum()
    z_loss_sum = z_loss_values.sum()
    z_loss_mean = z_loss_sum / torch.clamp(num_local_tokens, min=1)
else:
    z_loss_sum = z_loss_values.sum()
    # Keep the token count as a Python scalar so CUDA graph capture does not
    # record a CPU-to-CUDA tensor creation.
    z_loss_mean = z_loss_sum / max(logits.shape[0], 1)
```

### 2.4 Analysis of the Fix

**Key changes**:
1. **No-padding path**: Replaced `torch.tensor(logits.shape[0], device=logits.device)` with Python scalar `max(logits.shape[0], 1)`. This eliminates the CPU-to-CUDA copy that breaks CUDA graph capture.
2. **Padding path**: Unchanged. `valid_mask.sum()` produces a GPU tensor directly from GPU operations, which CUDA graph capture handles correctly.
3. **z_loss_sum computation**: Moved from shared location to per-branch. In the no-padding branch, `z_loss_sum` is computed entirely on GPU (sum of GPU tensor), then divided by a Python scalar. This produces a valid GPU tensor without any host-to-device transfers.
4. **Clamp replacement**: `torch.clamp(num_local_tokens, min=1)` (GPU operation) replaced with `max(logits.shape[0], 1)` (Python operation) for the no-padding path. Both prevent division by zero.

**Why this works with CUDA graph capture**:
- `z_loss_values.sum()` is a pure GPU operation (sum of GPU tensor) -> graph-capturable
- `max(logits.shape[0], 1)` is a Python scalar computed before graph capture -> baked into the graph as a constant at capture time
- Division of GPU tensor by Python scalar is graph-capturable (PyTorch broadcasts the scalar to GPU)
- The entire no-padding branch is now graph-capturable without any host-to-device transfers

**Preserved behavior**:
- Per-token z-loss scaling: both paths compute `z_loss_mean` (sum / count), maintaining per-token loss normalization
- MTP scaling: `mtp_loss_scale` logic is unchanged (visible in the diff continuation)
- z-loss metric logging: `z_loss_mean` value is unchanged for both paths
- Padding-mask path: completely unchanged, continues to use `torch.clamp(valid_mask.sum(), min=1)`

### 2.5 MoE Router z-loss: Background

**What is z-loss?**: Z-loss is a training auxiliary loss for MoE models that penalizes large router logits. It prevents the router logit magnitudes from growing unboundedly, which would cause:
- Expert selection collapse (one expert gets overwhelmingly high probability)
- Gradient instability through the router
- Numerical overflow in softmax computation

The z-loss formula: `z_loss = weight * log(sum(exp(logits)))^2`, penalizing the log-sum-exp of router logits.

**Why it matters for CUDA graph**: MoE router computation is often captured in CUDA graphs for efficiency (replaying the expert selection + dispatch computation without re-tracing). z-loss is computed alongside router logits, so it must be graph-capturable too.

### 2.6 Connection to DSV4 CUDA Graph Concerns

This fix is directly relevant to our tracked DSV4 systematic instability findings:

**DSV4 CUDA graph failures** (from MEMORY.md):
- Megatron #5317: Triton in-place `rotary_fwd_q_kernel` bypasses autograd version counter -> NaN at iter 2. `apply_rope_fusion=False` MANDATORY for MLA/DSV4-Hybrid.
- SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED on RL weight reload -> 64x accuracy blowup. Fix: dict.clear() on cache + weight-load funnel call.
- SGLang #28679: GDN intermittent decode degeneracy -- worsens over uptime, clears on restart.
- vLLM #46085: aot_eager = SM89 batch-invariant BY DESIGN (Dynamo+AOTAutograd without Inductor -> no fusion -> no prologue).

**The pattern**: #5401's CPU-to-CUDA tensor creation breaking CUDA graph capture is a **State Lifecycle Mismatch** pattern family member. The Python int `logits.shape[0]` is a state that changes between invocations (variable batch size), but CUDA graph capture treats it as a constant. This mismatch between dynamic state (batch size) and static graph (captured operation) causes failure.

**Pattern family taxonomy**:
```
State Lifecycle Mismatch (10th theory):
  Level 1 (Crash): #5401 z-loss CUDA graph crash, #5317 Triton in-place NaN
  Level 2 (Degeneracy): #28679 GDN intermittent decode, #5394 Muon clipping stall
  Level 3 (Silent Corruption): #28676 MXFP8 MoE cache clobber, DSV4 hybrid aliasing
```

#5401 is Level 1 (explicit crash during graph capture -- user sees an error), which is actually the **least dangerous** level because it is immediately visible. The more dangerous variants (Levels 2-3) silently corrupt training without any error signal.

### 2.7 TE CUDA Graph + MoE: Architecture Implications

TransformerEngine (TE) provides optimized CUDA graph capture for MoE operations. The combination of:
- TE CUDA graph capture for MoE router
- MoE z-loss regularization
- Variable batch sizes (common in GRPO training)

Creates a specific vulnerability: any Python-to-GPU data transfer in the z-loss computation breaks the graph. The fix ensures all z-loss computation in the no-padding path stays pure-GPU + Python-constant.

**Implications for RTX 4090**:
- SM89 (RTX 4090's compute capability) has its own CUDA graph constraints (no TMA, limited Inductor fusion)
- MoE models on RTX 4090 are already constrained (memory limits expert count)
- If z-loss + CUDA graph is used for MoE GRPO on RTX 4090, this fix is MANDATORY
- However: RTX 4090 MoE GRPO is currently memory-constrained, making this a lower-priority concern than the memory/addressable-experts bottleneck

### 2.8 Branch Targeting: dev vs main

Notable difference: #5401 targets the **dev** branch while #5400 targets **main**. This suggests:
- `dev` is the development/staging branch for Megatron
- `main` is the production branch
- #5401 may need to go through dev -> main merge pipeline
- This is the standard Megatron contribution flow for smaller fixes

---

## 3. Cross-Reference Analysis

### 3.1 Connection to #5394 (ChainedOptimizer Muon Clipping Bug)

**#5394** documents that `ChainedOptimizer.step()` computes a single global `grad_norm` and applies clipping to every sub-optimizer, including Muon. The clip coefficient `c = clip_grad / grad_norm` pushes per-matrix gradients below Newton-Schulz's `F.normalize(eps=1e-7)` floor, causing silent training stall.

**How #5400 relates**: Both #5394 and #5400 were discovered by the SAME author (yuchenwang3) while training Qwen3.5-35B-A3B (GatedDeltaNet hybrid) on 16xB200. The GDN architecture creates **both** problems simultaneously:
1. Structural: `in_proj` heterogeneous fusion -> meaningless orthogonalization (#5400)
2. Clipping: large GDN `grad_norm` -> clip coefficient collapses -> NS degenerates (#5394)

**How #5401 relates**: Different failure class (CUDA graph, not optimizer). But the MoE z-loss is an **auxiliary training loss** that adds gradient pressure to the router, potentially contributing to the large `grad_norm` that triggers #5394's clipping degeneration. If z-loss + MoE are combined with GDN in a hybrid model, the gradient landscape becomes even more complex.

### 3.2 Connection to #5395 (skip_grad_norm_clip Fix PR)

**#5395** adds a `skip_grad_norm_clip` attribute on the Muon sub-optimizer to bypass global gradient clipping for orthogonalizing optimizers.

**ShauryaaSharma's review** (4 detailed comments, June 18):

1. **Critical -- flag set on container, not inner sub-optimizers**: `LayerWiseDistributedOptimizer` is a `ChainedOptimizer` subclass. Setting `skip_grad_norm_clip` on the container object doesn't propagate to inner per-layer sub-optimizers. When `results` is empty, the container IS the top-level optimizer, and `ChainedOptimizer.step()` iterates over `self.chained_optimizers` (the per-layer sub-optimizers) which don't carry the flag. **Fix is entirely bypassed for this path.** Suggested fix: `setattr(optimizer, 'skip_grad_norm_clip', True)` on each raw sub-optimizer before appending.

2. **High -- unconditional skip for ALL emerging optimizers**: `_get_megatron_emerging_optimizer` fires `setattr(..., 'skip_grad_norm_clip', True)` for ALL optimizers in `_EMERGING_OPTIMIZERS` (includes SOAP and Lion, not just Muon). The exemption is only semantically valid for orthogonalizing optimizers. SOAP and Lion are magnitude-sensitive and should remain subject to clipping. **Suggested fix**: allowlist `_ORTHOGONALIZING_OPTIMIZERS = frozenset({'muon', 'dist_muon'})` and gate on it.

3. **Medium -- Muon grads inflate shared grad_norm for threshold**: Even with skip_clip, Muon's large gradients still inflate the global `grad_norm` used for `grad_norm_skip_threshold`. If a user sets a finite threshold to guard against NaN cascades, AdamW's update will be suppressed based on a norm that Muon's gradients are dominating. **Suggested fix**: compute a separate `grad_norm` excluding skip_clip params for the threshold check.

4. **Low -- wasted collective**: `should_clip` predicate checks `clip_grad > 0` but doesn't consult `skip_grad_norm_clip`. When all clipping candidates carry `skip_grad_norm_clip=True`, `_compute_grad_norms_by_group()` still fires AllReduces to compute norms that no sub-optimizer will use. **Suggested fix**: add `skip_grad_norm_clip` to the predicate.

These 4 review issues are **high quality** and show genuine reviewer engagement. Two are CRITICAL (#1, #2) and must be fixed before merge. This is encouraging -- unlike most community PRs that get 0 reviews, #5395 has an NVIDIA reviewer (ShauryaaSharma) actively providing detailed feedback.

### 3.3 Connection to #5396 (GDN L2-norm Fold)

**#5396** folds the q/k L2-normalization into the FLA `gated_delta_rule` kernel instead of materializing it as a separate activation. This saves backward activation memory (keeps only small `rstd` vectors instead of full normalized q/k).

**Same author, same model, same training setup**: All three PRs (#5400, #5395, #5396) came from yuchenwang3 training Qwen3.5-35B-A3B (GatedDeltaNet hybrid) on 16xB200. The GDN architecture triggered multiple independent bugs:
- #5400: structural Muon routing incompatibility
- #5394/#5395: global clipping degeneration
- #5396: backward activation memory waste from explicit L2norm

**#5396 is DRAFT** (explicitly marked), questioning whether the hardcoded `use_qk_l2norm_in_kernel=False` was intentional for specific paths (cu_seqlens/packed-sequence or CP correctness). The author asks maintainers for guidance on gating conditions.

### 3.4 #5400 vs #5395: Complementary Fixes

Both #5400 and #5395 are needed for GDN + Muon to work correctly:

```
Without #5400: in_proj gets meaningless orthogonalization (semantically incoherent updates)
Without #5395: even correctly-routed Muon groups stall from clipping degeneration
Without BOTH:  GDN + Muon is silently broken in two independent ways simultaneously
```

They address different layers of the problem:
- #5400: **architecture-optimizer compatibility** (which params should Muon process?)
- #5395: **optimizer pipeline correctness** (how should Muon params be treated in the clip pipeline?)

### 3.5 Connection to DeepSpeed #8068 (gradient_clipping default change)

**#8068**: DeepSpeed changed `gradient_clipping` default from 0 to 1.0, silently clipping Muon updates. Same clipping degeneration pattern as #5394, but in DeepSpeed's ZeRO pipeline.

**Cross-framework pattern**: The clipping-Muon interaction bug exists in BOTH Megatron and DeepSpeed:
```
Megatron: ChainedOptimizer global grad_norm clip -> Muon group clipped -> NS degenerates
DeepSpeed: gradient_clipping default 1.0 -> Muon group clipped -> NS degenerates
```

Both frameworks apply magnitude-based clipping to a scale-invariant optimizer, producing the same silent stall.

### 3.6 Connection to vLLM MoE Issues

**vLLM #45683**: Deterministic MoE combine -- CRITICAL for GRPO. MoE expert combination must be deterministic for consistent reward computation across rollout steps.

**How #5401 connects**: #5401 fixes MoE router z-loss CUDA graph compatibility. If z-loss is used during MoE GRPO training (to stabilize expert selection), the CUDA graph incompatibility would prevent graph-captured training. But this is a Megatron-specific training concern, not a vLLM inference concern.

---

## 4. Updated Muon Blockers Count (Now 6)

### 4.1 Complete Muon Blockers List

| # | Issue/PR | Framework | Type | Blocker Description | Status |
|---|---|---|---|---|---|
| 1 | **#5179** | Megatron | Infrastructure | `emerging_optimizers` PyPI stub (999.9.9), can't install -> Muon completely unusable from pip | OPEN |
| 2 | **#7939** | DeepSpeed | Structural | ZeRO-2 CPU offload incompatible with Muon NS iteration (different memory space, sync requirements) | CLOSED without merge |
| 3 | **#5394** | Megatron | Clipping | ChainedOptimizer applies global grad_norm clip to Muon group -> NS degenerates -> silent stall | OPEN |
| 4 | **#5395** | Megatron | Clipping | Fix PR for #5394: `skip_grad_norm_clip` attribute. 4 review issues, 2 CRITICAL (flag propagation, allowlist) | OPEN, reviewer engagement |
| 5 | **#8068** | DeepSpeed | Clipping | `gradient_clipping` default 0->1.0 silently clips Muon updates | OPEN, 0 reviews, stalled |
| 6 | **#5400** | Megatron | Structural | GDN `in_proj` heterogeneous fusion orthogonalized as one matrix -> meaningless updates | DRAFT, 0 reviews |

### 4.2 Blocker Taxonomy: Two Failure Modes

```
Muon Failure Mode          | Symptoms                    | Root Cause Layer    | Fix Type
────────────────────────── | ────────────────────────── | ─────────────────── | ──────────
Structural Incompatibility | Meaningless updates         | Architecture design | Route to Adam (#5400)
Clipping Degeneration      | Silent training stall       | Optimizer pipeline  | Skip clip (#5395) / fix NS eps
Infrastructure Blockers    | Can't install / can't offload | Packaging / ZeRO   | PyPI package (#5179) / CPU offload redesign
```

### 4.3 Blocker Severity Assessment

| Blocker | Severity | RTX 4090 Impact | Production Impact |
|---|---|---|---|
| #5179 (PyPI stub) | CRITICAL | Blocks ALL Muon usage | Blocks ALL Muon usage everywhere |
| #7939 (CPU offload) | HIGH | Blocks single-GPU Muon (ZeRO-2+CPU_Adam was our best path) | Blocks CPU-offload training |
| #5394/#5395 (clipping) | HIGH | Any model with large grad_norm stalls (including non-GDN models) | Silent stall = worst bug pattern |
| #8068 (DeepSpeed clip) | HIGH | Same as #5394 but in DeepSpeed ecosystem | Same silent stall pattern |
| #5400 (GDN routing) | MEDIUM | Only affects GDN models (Qwen3.5-35B-A3B, etc.) | GDN is growing architecture class |

---

## 5. State Lifecycle Mismatch Pattern Family Connections

### 5.1 Pattern Family Members (Updated)

#5401 adds a new member to the State Lifecycle Mismatch pattern family:

```
State Lifecycle Mismatch Pattern Family (10th algorithm theory):
  Level 1 (Crash - immediately visible):
    - #5401: MoE z-loss CPU-to-CUDA tensor creation breaks CUDA graph capture
    - #5317: Triton in-place rotary_fwd_q_kernel bypasses autograd -> NaN at iter 2
  Level 2 (Degeneracy - worsens over time, clears on restart):
    - #28679: GDN intermittent decode degeneracy (SGLang)
    - #5394: Muon clipping stall (positive feedback: clip -> smaller update -> larger grad_norm -> smaller clip -> ...)
  Level 3 (Silent Corruption - no error signal, worst pattern):
    - #28676: MXFP8 MoE shuffle cache CLOBBERED (SGLang)
    - DSV4-Hybrid MQA aliasing compounds corruption (Megatron #5317)
    - #8072/#8073: ZeRO-3+PEFT dtype mismatch (DeepSpeed)
```

### 5.2 #5401's Pattern Analysis

The #5401 bug is a Level 1 (Crash) State Lifecycle Mismatch:

```
State: Python integer logits.shape[0] (batch size)
Lifecycle requirement: Must be constant at CUDA graph capture time, but actually varies between invocations

Mismatch: CUDA graph capture records torch.tensor(logits.shape[0], device=logits.device)
           as a CPU-to-CUDA transfer, but graph replay expects static operations.
           The batch size changes between training steps (variable-length sequences, padding, etc.)

Result: PyTorch rejects the graph capture (explicit crash -- Level 1 severity)

Fix: Keep the state as a Python scalar (max(logits.shape[0], 1)) which is
     baked into the graph at capture time as a constant, eliminating the transfer.
```

**Why this is Level 1 and not worse**: The crash happens during CUDA graph creation, which is early in training setup. The user sees an explicit error and can diagnose. This is the **least dangerous** level of the pattern family.

**What would make it Level 3**: If the CPU-to-CUDA transfer were allowed during graph capture but produced wrong values during replay (e.g., stale batch size from capture time used for replay with different batch size), the z-loss would silently compute incorrect normalization, corrupting MoE router training without any error signal. This hypothetical scenario is exactly what happens in SGLang #28676 (stale cache state) and vLLM #46118 (stale FSM state).

### 5.3 CUDA Graph + Dynamic State: General Principle

From #5401, a general principle for CUDA graph compatibility:

> **Any Python-derived value that changes between training steps (batch size, sequence length, padding count) MUST NOT be converted to a CUDA tensor during graph capture. Instead, it should remain as a Python scalar that is baked into the graph as a constant, OR be pre-computed as a GPU tensor before graph capture begins.**

This principle applies to:
- MoE z-loss token count (#5401)
- Batch size-dependent buffer sizes
- Dynamic padding masks
- Per-step metadata (sequence lengths, attention masks)

For RTX 4090 (SM89), CUDA graph usage is already constrained:
- `enforce_eager=True` is MANDATORY for DSV4 models (from our DSV4 systematic instability findings)
- `aot_eager` mode is SM89 batch-invariant by design (vLLM #46085)
- Triton kernels may have in-place operations that break autograd (#5317)

The CUDA graph + dynamic state problem compounds these constraints: even when CUDA graphs work on SM89, dynamic state handling must be careful.

---

## 6. RTX 4090 GRPO Training Relevance

### 6.1 Direct Relevance Assessment

| PR | Direct RTX 4090 Relevance | Reason |
|---|---|---|
| #5400 (GDN Muon routing) | **LOW** | Muon is NOT viable on RTX 4090 (6 blockers). GDN models require multi-GPU training (35B model > 24 GiB). The routing fix matters for future GPU setups but not current RTX 4090. |
| #5401 (MoE z-loss CUDA graph) | **LOW-MEDIUM** | MoE on RTX 4090 is memory-constrained (limited experts). z-loss + CUDA graph is a training concern, not inference. But the CUDA graph + dynamic state lesson applies broadly to RTX 4090 GRPO. |

### 6.2 Indirect Relevance: Lessons That Apply

**From #5400**:
- Optimizer-architecture compatibility is NOT automatic -- every heterogeneous fusion needs explicit routing
- Removing safety guards without adding routing creates silent correctness regressions
- `skip_orthogonalization` is a good reusable pattern for future models
- Even if we never use Muon on RTX 4090, the lesson about semantic homogeneity applies to any optimizer with implicit assumptions about weight structure

**From #5401**:
- CUDA graph capture + dynamic Python values is a systematic risk
- RTX 4090's SM89 already requires `enforce_eager=True` for DSV4, limiting CUDA graph usage
- The z-loss + CUDA graph pattern is relevant IF we ever train MoE models with CUDA graph on RTX 4090
- The general principle (Python-derived dynamic state -> Python scalar, not GPU tensor during graph capture) applies to ALL CUDA graph usage on RTX 4090

### 6.3 GRPO Training Stack Impact

**Current RTX 4090 GRPO ranking** (unchanged by these PRs):
```
verl CPPO+bypass #1 > verl Tinker primitives #1.5 > verl GRPO+bypass #2
> DeepSpeed ZeRO-2 #2.5 > rLLM Tinker #3 BLOCKED (#605) > Megatron core #4
```

**Why Megatron stays #4**:
- Muon has 6 blockers, making it unusable on any single-GPU setup
- GDN models require multi-GPU training (not viable on RTX 4090)
- MoE z-loss CUDA graph fix is Megatron-specific, not applicable to verl/DeepSpeed/rLLM
- Megatron's training infrastructure is the most complex, requiring the most configuration

**The RTX 4090 MUST DO rules (updated)**:
1. ALWAYS use ZeRO-2 + CPU_Adam for DeepSpeed (NOT ZeRO-3, NOT Muon)
2. ALWAYS set `overlap_comm=False` on single GPU (#8061)
3. ALWAYS set `gradient_clipping` explicitly (1.0 for GRPO, NOT default 0 -> 1.0 silent change)
4. ALWAYS use `enforce_eager=True` for DSV4 models on SM89
5. ALWAYS use `bypass_mode=True` for verl GRPO on RTX 4090
6. NEVER use Muon on RTX 4090 (6 blockers confirm it's research-only)
7. NEVER use CUDA graph capture with dynamic Python-to-GPU transfers (#5401 lesson)
8. NEVER orthogonalize heterogeneous fused projections as one matrix (#5400 lesson)

### 6.4 Future RTX 4090 Relevance Scenarios

**Scenario 1: Qwen3.5-14B-A2B GDN hybrid on RTX 4090**
- If a smaller GDN model fits on RTX 4090 (~14B sparse), #5400's routing fix becomes directly relevant
- Even without Muon, the `skip_orthogonalization` pattern could route specific params to different optimizers in a chained setup
- #5395's clipping fix would matter for ANY optimizer with large GDN grad_norm

**Scenario 2: MoE GRPO with z-loss on RTX 4090**
- If MoE models (e.g., Qwen3-30B-A3B with EP=1, LoRA) are trained with z-loss regularization
- CUDA graph capture for the MoE router would need #5401's fix
- But current verl/DeepSpeed setups don't use Megatron's MoE router, so this is framework-specific

---

## 7. yuchenwang3's PR Cluster: Systematic GDN Bug Discovery

### 7.1 The PR Cluster

All from the same author, same model (Qwen3.5-35B-A3B GDN hybrid), same hardware (16xB200):

| PR | Type | Bug Class | +lines/-lines | Status |
|---|---|---|---|---|
| #5394 (issue) | Bug report | Clipping degeneration | N/A | OPEN, 4 comments |
| #5395 | Fix PR | Clipping skip | +420/-3 | OPEN, 4 CRITICAL review issues |
| #5396 | Perf PR | L2norm fold | +7/-4 | DRAFT |
| #5400 | Fix PR | Structural routing | +14/-1 | DRAFT, 0 reviews |

This is a **systematic bug discovery** pattern: a single training setup on a novel architecture (GDN hybrid) exposed multiple independent bugs across the optimizer pipeline. The bugs span three different failure classes:
1. Pipeline correctness (clipping)
2. Architecture-optimizer compatibility (routing)
3. Activation memory efficiency (L2norm materialization)

### 7.2 The GDN Architecture: Bug Magnet

GatedDeltaNet's design creates an unusual combination of challenges:
- **Heterogeneous fused projection**: 6-way fusion (q/k/v/gate/beta/alpha) breaks optimizer assumptions
- **Large gradient norms**: GDN layers have inherently large and imbalanced gradients
- **Linear attention mechanics**: Different gradient flow patterns than standard softmax attention
- **Gated delta rules**: Additional gating parameters increase gradient complexity

These properties make GDN a "bug magnet" for optimizer infrastructure that was designed for standard transformer architectures. The discovery of these bugs through GDN training validates the principle that **novel architectures stress existing infrastructure in ways that standard benchmarks cannot**.

---

## 8. Key Takeaways

### 8.1 For #5400 (GDN Muon Routing)

1. **Heterogeneous weight fusion is fundamentally incompatible with whole-matrix orthogonalization** -- not a bug, but a semantic mismatch. Orthogonalizing q/k/v/gate/beta/alpha together produces mathematically valid but semantically meaningless update directions.

2. **Removing safety guards without adding routing creates silent correctness regressions** -- Megatron's removal of the `assert` blocks made GDN+Muon worse: from "explicitly blocked" to "silently incorrect."

3. **The `skip_orthogonalization` attribute is a good reusable pattern** -- generic, semantically clear, no checkpoint coordination needed. Other fused modules (SSM variants, gated attention) can use the same mechanism.

4. **#5400 and #5395 are complementary fixes** -- both needed for GDN+Muon to work. Without #5400, in_proj gets meaningless orthogonalization. Without #5395, even correctly-routed Muon groups stall from clipping. Both are 0-review stalled PRs.

5. **This is the 6th Muon blocker** -- 3 infrastructure, 2 clipping, 1 structural. Muon remains NOT viable on RTX 4090.

### 8.2 For #5401 (MoE z-loss CUDA Graph)

6. **CPU-to-CUDA tensor creation breaks CUDA graph capture** -- `torch.tensor(python_int, device=cuda)` records a host-to-device transfer that PyTorch rejects during graph capture. The fix: keep dynamic values as Python scalars that are baked into the graph.

7. **This is a Level 1 State Lifecycle Mismatch** -- the mismatch between dynamic batch size (Python int) and static graph (captured operation) causes an explicit crash. This is the least dangerous level of the pattern family because it is immediately visible.

8. **The CUDA graph + dynamic state principle applies broadly** -- any Python-derived value that changes between steps must remain as a Python scalar during graph capture, not be converted to a GPU tensor. This applies to all CUDA graph usage on RTX 4090.

9. **z-loss + CUDA graph is a specific MoE training concern** -- not relevant to inference (vLLM/SGLang don't use z-loss). Relevant to Megatron MoE training with TE CUDA graph capture.

10. **#5401's branch target is `dev` (not `main`)** -- follows the standard Megatron contribution flow for smaller fixes.

### 8.3 Combined Takeaways

11. **Both PRs were opened on the same day (June 18)** -- reflecting rapid bug discovery and fix submission from the community.

12. **Both PRs are community-request label with 0 human reviews** -- standard for external contributor PRs in Megatron, which has a rigorous vetting process.

13. **#5395 has genuine reviewer engagement (ShauryaaSharma, 4 detailed comments)** -- unlike most stalled community PRs. Two of the four issues are CRITICAL (flag propagation, optimizer allowlist). This gives #5395 a realistic merge path.

14. **GDN is a systematic bug-magnet architecture** -- novel architectures stress existing optimizer infrastructure in ways standard benchmarks cannot. The 4-bug cluster (#5394/#5395/#5396/#5400) from one training setup validates this.

15. **RTX 4090 GRPO ranking unchanged** -- both PRs address concerns that are either multi-GPU-only (#5400) or Megatron-specific (#5401). The practical ranking remains verl CPPO+bypass > DeepSpeed ZeRO-2 > others.

---

## 9. Monitor Items

**Active tracking for these PRs**:
- #5400: Watch for reviewer assignment and first human review. Current: 0 reviews, DRAFT, blocked merge. Expected timeline: 1-2 weeks for NVIDIA vetting.
- #5401: Watch for CI results (Victarry triggered "/ok to test"). Watch for reviewer engagement on the `dev` branch. Expected timeline: shorter than #5400 since it's a simpler fix.
- #5395: CRITICAL -- ShauryaaSharma's 4 review issues must be addressed before merge. Two are blockers (flag propagation, allowlist). Watch for author's response.
- #5396: DRAFT, questioning maintainers about CP/packed-sequence implications. Watch for maintainer response on the gating question.

**Cross-PR dependencies**:
- #5400 and #5395 must BOTH merge for GDN+Muon correctness
- #5401 is independent (different subsystem, different author, different branch)
- #5396 is independent (perf optimization, not a correctness fix)

---

## References

- PR #5400: https://github.com/NVIDIA/Megatron-LM/pull/5400 (GDN in_proj Adam routing)
- PR #5401: https://github.com/NVIDIA/Megatron-LM/pull/5401 (MoE z-loss CUDA graph fix)
- Issue #2885: https://github.com/NVIDIA/Megatron-LM/issues/2885 (GDN + Muon compatibility question)
- PR #5395: https://github.com/NVIDIA/Megatron-LM/pull/5395 (skip_grad_norm_clip fix)
- Issue #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (ChainedOptimizer clipping bug)
- PR #5396: https://github.com/NVIDIA/Megatron-LM/pull/5396 (GDN L2-norm fold)
- Issue #5179: https://github.com/NVIDIA/Megatron-LM/issues/5179 (emerging_optimizers PyPI stub)
- DeepSpeed #7939: https://github.com/microsoft/DeepSpeed/pull/7939 (Muon CPU offload)
- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/issues/8068 (gradient_clipping default)
- NeMo #229: https://github.com/NVIDIA-NeMo/Emerging-Optimizers/issues/229 (NS degeneration)
- NeMo #230: https://github.com/NVIDIA-NeMo/Emerging-Optimizers/pull/230 (NS eps floor fix)
- vLLM #45683: https://github.com/vllm-project/vllm/issues/45683 (Deterministic MoE combine)
- vLLM #46085: https://github.com/vllm-project/vllm/pull/46085 (aot_eager piecewise compilation)
- SGLang #28676: https://github.com/sgl-project/sglang/issues/28676 (MXFP8 MoE cache clobber)
- SGLang #28679: https://github.com/sgl-project/sglang/issues/28679 (GDN intermittent degeneracy)
- Megatron #5317: https://github.com/NVIDIA/Megatron-LM/issues/5317 (DSV4-Hybrid rope fusion NaN)
- ShauryaaSharma review on #5395: 4 detailed comments (June 18, 2026)
- State Lifecycle Mismatch Pattern Family derivation: notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md
