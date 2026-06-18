# Megatron #5400 — GatedDeltaNet in_proj Routing to Adam (6th Muon Blocker)

> 2026-06-18 | PR #5400 (DRAFT, OPEN) | Author: yuchenwang3 | Community-request label | +14/-1 lines
> ★★★★★★★★ 6th Muon blocker: structural incompatibility between heterogeneous fused projections and matrix orthogonalization
> ★★★★★★★★ TWO Muon failure modes now documented: structural (#5400, #7939) + clipping degeneration (#5394, #8068)
> ★★★★★★★★ Optimizer-architecture compatibility is NOT just about parameter shapes — it is about SEMANTIC homogeneity of fused weight groups

---

## 1. PR Summary

**Title**: fix(optimizer): route GatedDeltaNet in_proj to Adam instead of orthogonalizing it (Muon)

**Core change**: Tag `GatedDeltaNet.in_proj.weight` with `skip_orthogonalization = True` so it falls back to the standard (Adam) optimizer instead of being sent to Muon's Newton-Schulz orthogonalization. The `out_proj` weight intentionally stays on Muon.

**Status**: DRAFT (auto-converted), 0 reviews, 0 review comments, blocked merge state (needs vetting). Created 2026-06-18 by yuchenwang3. Refs issue #2885.

**Files changed** (2):
- `megatron/core/ssm/gated_delta_net.py`: +9/-0 (add `skip_orthogonalization = True` attribute to `in_proj.weight` after the module is created, with detailed comment explaining the heterogeneous fusion issue)
- `megatron/core/optimizer/emerging_optimizers.py`: +5/-1 (extend `_is_nonlinear_or_embedding` predicate to also check `getattr(param, 'skip_orthogonalization', False)`)

---

## 2. Root Cause: Heterogeneous Fused Projection Cannot Be Orthogonalized as One Matrix

### 2.1 GatedDeltaNet in_proj Architecture

GatedDeltaNet is a linear-attention variant (paper: "Gated DeltaNet: Linear Attention with Gated Deltas" by Songlin Yang, Jan Kautz, Ali Hatamizadeh, NVIDIA). It replaces softmax quadratic attention with O(n) recurrent computation, using gated delta rules to control information retention/forgetting.

The critical architectural feature is `in_proj` — a single fused linear projection that computes **six semantically distinct outputs** in one matrix multiplication:

```
in_proj_dim = qk_dim * 2 + v_dim * 2 + num_value_heads * 2

Components fused into in_proj:
  q (query)        → qk_dim      (attention query projection)
  k (key)          → qk_dim      (attention key projection)
  v (value)        → v_dim       (attention value projection)
  z (gate)         → v_dim       (output gating — controls retention)
  beta             → num_value_heads (delta rule intensity)
  alpha            → num_value_heads (forgetting/decay rate)

In forward:
  qkvzba, _ = self.in_proj(hidden_states)
  qkv, gate, beta, alpha = torch.split(qkvzba, [...split_sizes...], dim=-1)
```

The fused design is motivated by GPU kernel efficiency — one large GEMM replaces six small ones, maximizing arithmetic intensity. But this creates a fundamental tension with Muon.

### 2.2 Why Orthogonalization of Heterogeneous Fusion Is Meaningless

Muon's Newton-Schulz orthogonalization treats the entire `in_proj.weight` matrix (shape `[hidden_size, in_proj_dim]`) as a single 2D weight and orthogonalizes its momentum update as one matrix. This is the correct behavior for a semantically homogeneous projection (e.g., a standard MLP `nn.Linear`), but catastrophically wrong for a heterogeneous fusion:

**The mathematical argument**:

Newton-Schulz iteration computes the orthogonal polar factor of a matrix `M`:
- Input: momentum matrix `M` of shape `[m, n]`
- Output: orthogonal matrix `U` such that `M = U * S` (polar decomposition)
- The orthogonalization operates on the **correlation structure of rows** of `M`

When `M` is a heterogeneous fused weight containing q, k, v, gate, beta, and alpha projections:
- The row correlation structure mixes **six semantically unrelated** gradient flows
- Query gradients have no meaningful correlation with gate gradients
- Key update directions should not be forced orthogonal to beta update directions
- The resulting orthogonal matrix treats the entire fused gradient as one statistical object, producing update directions that are mathematically coherent but **semantically incoherent**

**Analogy**: Orthogonalizing q/k/v/gate/beta/alpha together is like computing a single PCA across temperature readings, stock prices, and pixel intensities — the orthogonal directions are mathematically valid but physically meaningless.

### 2.3 Why Standard Attention QKV Is Different

Standard transformer attention also fuses q, k, v into one `linear_qkv` weight. But Megatron has a **dedicated split path** for this:

```python
# --muon-split-qkv flag
# TensorParallelMuon.orthogonalize():
#   When is_qkv_fn(p) returns True for a parameter:
#     1. Split the weight into Q, K, V sub-blocks by qkv_split_shapes
#     2. Orthogonalize each sub-block independently
#     3. Concatenate the orthogonalized sub-blocks back
```

This works because:
- Q, K, V are **semantically related** (all are attention projections)
- Each sub-block is independently orthogonalized — no cross-correlation pollution
- The split shapes are well-defined and uniform across heads

`in_proj` cannot use this path because:
1. It fuses **more than just q/k/v** — conv, gate, and beta are structurally different
2. The sub-block shapes are heterogeneous (qk_dim vs v_dim vs num_value_heads)
3. Even if you split, the conv/gate/beta sub-blocks should NOT be orthogonalized at all (they are scalar/gate operations, not attention projections)
4. No `--muon-split-in-proj` infrastructure exists, and building it would be invasive

---

## 3. The skip_orthogonalization Mechanism

### 3.1 Implementation

The PR introduces a generic `skip_orthogonalization` attribute on parameter tensors:

```python
# gated_delta_net.py — after in_proj creation:
if hasattr(self.in_proj, "weight") and self.in_proj.weight is not None:
    self.in_proj.weight.skip_orthogonalization = True

# emerging_optimizers.py — predicate expansion:
def _is_nonlinear_or_embedding(param):
    return (
        getattr(param, 'is_embedding_or_output_parameter', False)
        or getattr(param, 'skip_orthogonalization', False)    # NEW
        or len(param.shape) != 2
    )
```

### 3.2 Design Rationale

The author chose `skip_orthogonalization` over overloading `is_embedding_or_output_parameter` for semantic clarity:
- `is_embedding_or_output_parameter` = this param is an embedding or output layer
- `skip_orthogonalization` = this param should not be orthogonalized, for any reason

The generic flag allows future fused modules (e.g., other SSM/linear-attention variants) to reuse the same routing without claiming they are "embeddings." The naming is accurate — the reason is orthogonalization incompatibility, not parameter type.

### 3.3 Existing Optimizer Routing Predicate Flags

Before this PR, Megatron's Muon routing had three predicate mechanisms:

| Attribute | Purpose | Origin |
|---|---|---|
| `is_embedding_or_output_parameter` | Route embeddings/output to Adam | Megatron model code (set on embed/output params) |
| `is_qkv` | Identify QKV weights for split orthogonalization | Muon kwargs config |
| `len(param.shape) != 2` | Exclude non-matrix params (1D biases, norms) | Implicit shape check |

After this PR, there are four:

| Attribute | Purpose | Origin |
|---|---|---|
| `is_embedding_or_output_parameter` | Route embeddings/output to Adam | Model code |
| `skip_orthogonalization` | Route heterogeneous fused params to Adam | **NEW: #5400** |
| `is_qkv` | Enable per-sub-block orthogonalization | Muon config |
| `len(param.shape) != 2` | Exclude non-matrix params | Implicit |

### 3.4 Alternative Approaches Discussed

The PR description explicitly considers alternatives:

1. **Split-and-orthogonalize**: Like `--muon-split-qkv`, split `in_proj` into q/k/v sub-blocks and orthogonalize those, while keeping conv/gate/beta on Adam. More invasive — requires new split infrastructure and shape metadata for a module-specific decomposition.

2. **Name-based ParamKey override**: Instead of an attribute, use a name pattern (e.g., regex matching `in_proj`) in the optimizer config. Less modular, harder to maintain across model variants.

3. **Overload `is_embedding_or_output_parameter`**: Semantically wrong — `in_proj.weight` is not an embedding. Would confuse future readers and make the predicate harder to extend.

The PR chose the conservative approach (attribute tagging + Adam fallback) because it reuses existing infrastructure without new split groups or checkpoint coordination.

---

## 4. Issue #2885 Context: The Road to #5400

### 4.1 Original Question (January 2026)

Issue #2885 asked: "Is there a hard blocker preventing Muon to be used with gated deltanet and gated attention?"

The original code had two hard guard assertions:
```python
assert args.linear_attention_type is None, "Muon optimizer does not support linear attention type for now."
assert not args.attention_output_gate, "Muon optimizer does not support attention output gate for now."
```

### 4.2 CodersAcademy006's Analysis (January 2026)

CodersAcademy006 analyzed the code and concluded there was "no hard theoretical incompatibility" — the restrictions were safety guards due to unvalidated behavior. They noted that routing gated/linear-attention params to Adam "would require explicit parameter tagging, split optimizer param groups, and coordinated stepping and checkpointing" — but this was an implementation limitation, not a fundamental blocker.

### 4.3 The Silent Problem (June 2026)

By June 2026, the hard guards had been **removed from main** during Muon optimizer-validation refactoring. But the routing that #2885 discussed was **not added**. This created a worse situation: GatedDeltaNet + Muon was no longer explicitly blocked, but was now **silently incorrect**.

The Muon routing predicate `_is_nonlinear_or_embedding` only excluded:
- Embedding/output params (`is_embedding_or_output_parameter`)
- Non-2D params (shape check)

`in_proj.weight` is 2D, untagged, and not an embedding → it passes the predicate → goes to Muon → gets orthogonalized as one heterogeneous matrix → **semantically meaningless updates**.

### 4.4 The Fix Path

The author noted that CodersAcademy006's concern about needing "split optimizer param groups + coordinated stepping/checkpointing" was already solved: the `default_param_overrides` / `ParamKey` mechanism that routes embeddings and non-2D params to Adam is exactly that infrastructure, and it is checkpoint-supported. So routing `in_proj` to Adam is a small localized change, not new infrastructure.

---

## 5. Two Muon Failure Modes: Taxonomy

This PR, combined with the other Muon blockers, reveals that Muon incompatibility manifests in **two distinct failure modes**:

### 5.1 Structural Incompatibility (Architecture-Level)

**Definition**: The parameter's semantic structure is incompatible with matrix-level orthogonalization. Orthogonalizing the parameter produces mathematically valid but semantically meaningless updates.

| Issue | Framework | Parameter | Root Cause | Status |
|---|---|---|---|---|
| **#5400** | Megatron | GatedDeltaNet `in_proj` | Heterogeneous fusion (q/k/v/conv/gate/beta) | DRAFT, 0 reviews |
| **#7939** | DeepSpeed | Muon CPU offload | ZeRO-2 CPU offload incompatible with NS iteration on momentum (different memory space, synchronization requirements) | CLOSED without merge |

**Pattern**: Muon assumes each 2D weight matrix is a **semantically homogeneous** object whose row correlations are meaningful to orthogonalize. When the matrix fuses heterogeneous projections, this assumption breaks. The fix is routing: send incompatible params to Adam.

### 5.2 Clipping Degeneration (Optimizer-Level)

**Definition**: Global gradient clipping scales Muon's per-matrix inputs below Newton-Schulz's internal `F.normalize(eps=1e-7)` floor, causing orthogonalization to silently degenerate into near-zero, non-orthogonal updates.

| Issue | Framework | Root Cause | Status |
|---|---|---|---|
| **#5394** | Megatron | ChainedOptimizer applies global `grad_norm` clip to Muon group | OPEN (reported June 17) |
| **#8068** | DeepSpeed | `gradient_clipping` default 0→1.0 silently clips Muon | OPEN, 0 reviews, stalled |
| **#7776** | DeepSpeed | Adam/Muon ordering causes stale gradients | OPEN |
| **NeMo #229** | Emerging-Optimizers | Newton-Schulz `F.normalize(eps=1e-7)` floor degenerates for small inputs | OPEN |

**Pattern**: Muon's orthogonalization is **scale-invariant in principle** (NS discards gradient magnitude), but **NOT scale-invariant in implementation** (the `F.normalize(eps)` floor creates a degenerate regime). Global clipping pushes Muon's inputs into this regime. The fix is either: (a) skip clipping for orthogonalizing optimizers (#5395), or (b) fix the NS eps floor (NeMo #230).

### 5.3 The Taxonomy in One Table

```
Muon Failure Mode          | Symptom                    | Root Cause Layer    | Fix Type
────────────────────────── | ────────────────────────── | ─────────────────── | ──────────────
Structural Incompatibility | Meaningless updates        | Architecture design | Route to Adam
Clipping Degeneration      | Silent training stall      | Optimizer pipeline  | Skip clip / fix NS eps
```

The two failure modes are **independent** — a model can hit both simultaneously (GatedDeltaNet has structural incompatibility AND clipping degeneration from large `grad_norm`). This is exactly what the Qwen3.5-35B-A3B (GatedDeltaNet hybrid) training on 16xB200 encountered.

---

## 6. The 6th Muon Blocker: Complete List

| # | Issue | Framework | Type | Blocker | Status |
|---|---|---|---|---|---|
| 1 | **#5179** | Megatron | Infrastructure | `emerging_optimizers` is PyPI stub (999.9.9), can't install → Muon completely unusable | OPEN |
| 2 | **#7939** | DeepSpeed | Structural | ZeRO-2 CPU offload incompatible with Muon NS iteration | CLOSED without merge |
| 3 | **#5394** | Megatron | Clipping | ChainedOptimizer clips Muon group → NS degenerates → training stalls | OPEN |
| 4 | **#5395** | Megatron | Clipping | `skip_grad_norm_clip` attribute for orthogonalizing optimizers | OPEN, 0 reviews |
| 5 | **#8068** | DeepSpeed | Clipping | `gradient_clipping` default changed to 1.0 → clips Muon | OPEN, 0 reviews, stalled |
| 6 | **#5400** | Megatron | Structural | GatedDeltaNet `in_proj` heterogeneous fusion orthogonalized as one matrix | DRAFT, 0 reviews |

**Note**: #5394 and #5395 are the same underlying problem reported as bug (#5394) and fix (#5395). They can be counted as one blocker with two tracking issues. Either way, the total is 5-6 distinct Muon blockers.

---

## 7. GatedDeltaNet Source-Level Detail

### 7.1 in_proj Construction (gated_delta_net.py)

```python
# Dimension calculation — six outputs fused:
self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
# Expanded: q(qk_dim) + k(qk_dim) + v(v_dim) + z/gate(v_dim) + beta(num_value_heads) + alpha(num_value_heads)

# Module creation:
self.in_proj = build_module(
    submodules.in_proj,
    self.hidden_size,           # input dimension
    self.in_proj_dim,           # output dimension (6 fused outputs)
    ...
)
```

### 7.2 Forward Split

```python
# Single GEMM produces all six outputs:
qkvzba, _ = self.in_proj(hidden_states)

# After CP all-to-all and transpose, split into components:
qkv, gate, beta, alpha = torch.split(
    qkvzba,
    [
        (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,  # q+k+v
        self.v_dim_local_tp // self.cp_size,                                # gate (z)
        self.num_value_heads // self.tp_size // self.cp_size,               # beta
        self.num_value_heads // self.tp_size // self.cp_size,               # alpha
    ],
    dim=-1,
)
```

### 7.3 Conv1d Sub-Fusion

Even within the fused output, there's a secondary fusion: `conv1d` operates on just the QKV portion:

```python
self.conv_dim = self.qk_dim * 2 + self.v_dim
# conv1d processes: q + k + v (not gate, beta, alpha)
```

### 7.4 Sharded State Dict Names

```python
# The six fused components are named in order:
["query", "key", "value", "z", "beta", "alpha"]
```

---

## 8. Broader Pattern: Optimizer-Architecture Compatibility

### 8.1 The General Principle

Muon (and any matrix-level preconditioner) makes an **implicit assumption** about the parameters it processes: that each 2D weight matrix is a **semantically coherent** linear transformation whose row-wise gradient correlations encode meaningful update structure.

This assumption holds for:
- Standard MLP projections (`nn.Linear`): one semantic transformation
- Homogeneous QKV projections (with split): three related attention transformations
- MoE expert weights: per-expert transformations (orthogonalized independently)

It **breaks** for:
- Heterogeneous fused projections (q/k/v/gate/beta/alpha): six unrelated transformations
- Gated SSM projections (query/key/gate/decay): mixed attention and gating semantics
- Any "efficiency kernel" that packs unrelated operations into one GEMM

### 8.2 The Architecture-Optimizer Tension

Modern GPU-optimized model architectures increasingly use **weight fusion** for kernel efficiency:
- GatedDeltaNet: 6-way fusion in `in_proj`
- Standard transformers: 3-way QKV fusion (handled by split)
- MoE: expert weight packing (handled by EP/grouping)
- Conv+BN fusion, gate+up fusion in SwiGLU MLPs

Each fusion pattern needs a **corresponding optimizer disaggregation strategy**:
- QKV fusion → `--muon-split-qkv` (split, orthogonalize each, reassemble)
- Heterogeneous fusion → `skip_orthogonalization` (route to Adam, leave semantics intact)
- Expert packing → EP-aware orthogonalization (per-expert NS)

**The pattern**: every efficiency-driven weight fusion creates a potential optimizer incompatibility that must be explicitly addressed. The default behavior (orthogonalize every 2D weight as one matrix) is correct only for the simplest architectures.

### 8.3 Implications for RTX 4090 GRPO Training

On RTX 4090 with limited memory, weight fusion is even more attractive (reduces kernel launch overhead and intermediate memory). But this means:
- Any model using GatedDeltaNet-style architectures MUST configure optimizer routing
- Muon is NOT a "set `--optimizer muon` and forget" option — it requires architecture-aware param group configuration
- The RTX 4090 GRPO ranking remains: AdamW (safe, no routing needed) > Muon (potentially faster, but needs careful routing)
- DeepSpeed ZeRO-2 + CPU_Adam remains the safest single-GPU choice (no Muon, no routing, no clipping issues)

### 8.4 A Design Rule for Future Architectures

From this analysis, a general design rule emerges:

> **For any new model architecture that fuses semantically heterogeneous projections into a single weight matrix, the architect MUST either:**
> (a) Provide a Muon split path (like `--muon-split-qkv`) that disaggregates the fusion for orthogonalization, OR
> (b) Tag the fused weight with `skip_orthogonalization` so it falls back to a magnitude-based optimizer, OR
> (c) Document the incompatibility in the optimizer validation guards (like the old `assert` blocks)

Failure to do any of these results in **silently incorrect training** — the model trains, loss decreases, but the orthogonalized updates for fused weights are semantically meaningless, potentially degrading convergence quality without any visible error.

---

## 9. Cross-Framework Comparison: Muon Handling

| Framework | Muon Support | Heterogeneous Fusion Handling | QKV Split | Clipping Handling | Package Availability |
|---|---|---|---|---|---|
| **Megatron** | `--optimizer muon/dist_muon` | **None** (→ #5400 fixes) | `--muon-split-qkv` | ChainedOptimizer global clip (→ #5394/#5395) | `emerging_optimizers` PyPI stub (#5179) |
| **DeepSpeed** | `--optimizer muon` in ZeRO-2 | Not addressed | Not addressed | `gradient_clipping` default 1.0 (→ #8068) | Bundled in DeepSpeed |
| **verl** | No Muon support | N/A | N/A | Standard clipping | N/A |
| **rLLM** | No Muon support | N/A | N/A | Standard clipping | N/A |
| **SGLang** | Inference only | N/A | N/A | N/A | N/A |
| **vLLM** | Inference only | N/A | N/A | N/A | N/A |

Megatron is the only framework with both Muon support AND heterogeneous fusion models (GatedDeltaNet, MoE), making it the primary site for optimizer-architecture compatibility bugs.

---

## 10. Status and Outlook

**PR #5400 status**: DRAFT, blocked merge (needs NVIDIA vetting), 0 human reviews, 0 review comments. Auto-converted to draft per Megatron policy. The change is minimal (+14/-1) and reuses existing infrastructure.

**Related PRs/issues to watch**:
- #2885: Original question — now has the #5400 cross-reference in comments
- #5395: `skip_grad_norm_clip` fix for clipping degeneration — also 0 reviews, stalled
- #5179: PyPI stub blocker — Muon literally cannot be installed from pip
- NeMo Emerging-Optimizers #229/#230: NS eps floor fix — the clipping degeneration's upstream dependency

**Critical dependency**: #5400 (structural routing) and #5395 (clipping skip) are both needed for GatedDeltaNet + Muon to work correctly. Without #5400, `in_proj` gets meaningless orthogonalization. Without #5395, even correctly-routed Muon groups stall from clipping degeneration. Both are 0-review stalled PRs.

**RTX 4090 implication**: Muon remains NOT viable on RTX 4090 for GRPO training. Six blockers (3 infrastructure, 2 clipping, 1 structural) make it a research-only optimizer. CPU_Adam with ZeRO-2 remains the only practical choice. The GatedDeltaNet finding reinforces this: even if Muon were installable and clipping were fixed, heterogeneous architectures would need explicit per-module routing configuration — adding complexity to already complex GRPO setups.

---

## 11. Key Takeaways

1. **Heterogeneous weight fusion is fundamentally incompatible with whole-matrix orthogonalization** — not a bug, but a semantic mismatch between the optimizer's assumption (one coherent transformation per 2D weight) and the architecture's reality (six unrelated transformations packed for GPU efficiency).

2. **Muon has TWO distinct failure modes**: structural incompatibility (wrong params sent to orthogonalizer) and clipping degeneration (correct params but input scaled below NS floor). Both can hit simultaneously, as in GatedDeltaNet models.

3. **The `skip_orthogonalization` attribute is a good pattern** — generic, semantically clear, reusable across fused modules. It should be adopted by other frameworks as they add Muon support.

4. **Removing validation guards without adding routing is dangerous** — Megatron's removal of the `assert` blocks created a silent correctness regression. The old guards were ugly but safe; the current state is clean but wrong.

5. **Optimizer-architecture compatibility is a design concern, not just an implementation detail** — future model architects should document optimizer routing requirements alongside their fused projection designs.

6. **Six Muon blockers confirm: Muon is NOT production-ready** — even on high-end hardware (16xB200), Muon + GatedDeltaNet required manual intervention to avoid training collapse. On RTX 4090, the barriers are even higher.

---

## References

- PR #5400: https://github.com/NVIDIA/Megatron-LM/pull/5400
- Issue #2885: https://github.com/NVIDIA/Megatron-LM/issues/2885
- PR #5395 (skip_grad_norm_clip): https://github.com/NVIDIA/Megatron-LM/pull/5395
- Issue #5394 (clipping bug): https://github.com/NVIDIA/Megatron-LM/issues/5394
- Issue #5179 (PyPI stub): https://github.com/NVIDIA/Megatron-LM/issues/5179
- NeMo #229 (NS degeneration): https://github.com/NVIDIA-NeMo/Emerging-Optimizers/issues/229
- NeMo #230 (NS eps floor fix): https://github.com/NVIDIA-NeMo/Emerging-Optimizers/pull/230
- DeepSpeed #7939 (Muon CPU offload): https://github.com/microsoft/DeepSpeed/pull/7939
- DeepSpeed #8068 (gradient_clipping): https://github.com/microsoft/DeepSpeed/issues/8068
- DeepSpeed #7776 (ordering): https://github.com/microsoft/DeepSpeed/issues/7776
- GatedDeltaNet paper: Yang, Kautz, Hatamizadeh — "Gated DeltaNet: Linear Attention with Gated Deltas"
- GatedDeltaNet source: `megatron/core/ssm/gated_delta_net.py` on Megatron-LM main
- Emerging optimizers source: `megatron/core/optimizer/emerging_optimizers.py` on Megatron-LM main
