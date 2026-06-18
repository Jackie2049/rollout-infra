# Megatron-LM #5400 — GatedDeltaNet in_proj Muon Routing to Adam

> 2026-06-18 | OPEN | +14/-1 lines
> ★★★★★★★★ 6th Muon blocker: architecture-level incompatibility → GatedDeltaNet in_proj can't be orthogonalized
> ★★★★★★★★ skip_orthogonalization=True attribute → generic opt-out mechanism for fused modules
> ★★★★★★★★ Two Muon failure modes identified: structural (#5400/#7939) + clipping (#5394/#8068)

---

## What #5400 Does

```
★★★★★★★★★ Problem:
  → GatedDeltaNet uses recurrent gating with fused in_proj
  → in_proj combines q/k/v/conv/gate/beta into ONE weight matrix
  → Muon's Newton-Schulz orthogonalizer assumes "orthogonal update"
  → But: gated recurrence is NOT orthogonal → heterogeneous fusion → can't orthogonalize as one matrix
  → → Newton-Schulz on in_proj → WRONG rotation → training instability → NaN or divergence

★★★★★★★★★ Fix (+14/-1 lines):
  → Add skip_orthogonalization attribute to in_proj.weight
  → ChainedOptimizer checks skip_orthogonalization before applying Muon
  → If True → route to Adam instead → standard SGD-style update → safe for heterogeneous weights
  → → Generic opt-out: other fused modules can reuse this attribute → future-proof
```

---

## GatedDeltaNet Architecture Analysis

```
★★★★★★★★★ Why in_proj is incompatible with Muon:

GatedDeltaNet in_proj fuses 6 components:
  → q (query): standard attention component → orthogonalizable
  → k (key): standard attention component → orthogonalizable
  → v (value): standard attention component → orthogonalizable
  → conv (convolution): recurrence kernel → NOT orthogonalizable
  → gate (gating): sigmoid/softmax activation → NOT orthogonalizable
  → beta (decay): exponential decay → NOT orthogonalizable

★★★★★★★★★ The core incompatibility:
  → Muon orthogonalizes the ENTIRE in_proj matrix as one unit
  → But: conv + gate + beta are recurrent → their "update" is NOT a rotation
  → → Newton-Schulz sees a matrix that is PART rotation + PART recurrence
  → → Orthogonalizing the whole thing → distorts recurrence dynamics → training breaks
  → → This is NOT a bug → it's a fundamental architecture-optimizer mismatch

★★★★★★★★★ Resolution:
  → Route in_proj to Adam → standard gradient descent → handles heterogeneous weights
  → Route other components (q/k/v/out_proj) to Muon → orthogonal updates → faster convergence
  → → Hybrid optimizer: Muon for attention + Adam for recurrence → BEST for GatedDeltaNet
```

---

## Two Muon Failure Modes

```
★★★★★★★★★ Pattern: Muon fails in TWO distinct ways:

Failure Mode 1: Structural Incompatibility
  → Some weight structures can't be orthogonalized at all
  → → GatedDeltaNet in_proj (#5400): heterogeneous fusion → can't be one orthogonal matrix
  → → MoE expert weights (#7939): ZeRO-2 sharded → can't compute Newton-Schulz on partial params
  → → Fix: route incompatible weights to Adam → skip_orthogonalization attribute

Failure Mode 2: Clipping Degeneration
  → Global gradient clipping destroys Muon's orthogonalization
  → → DeepSpeed #8068: default clipping 0→1.0 → clips ALL groups → Muon clipped too
  → → DeepSpeed #7776: orthogonalization-before-clipping → near-zero vectors → degenerate
  → → Megatron #5394: ChainedOptimizer forces shared config → can't skip clipping per-sub-optimizer
  → → Fix: per-optimizer clipping control → #5395 skip_grad_norm_clip + #7776 per-group clipping

★★★★★★★★★ Universal insight:
  → Muon is NOT a universal optimizer → some architectures need Adam fallback
  → → Attention layers: Muon OK → orthogonal updates → faster convergence
  → → Recurrence/gating layers: Adam ONLY → Newton-Schulz incompatible
  → → MoE experts: Muon OK if NOT sharded → but CPU offload blocks it
  → → ★★★★★★★★ Hybrid optimizer = future → Muon where safe + Adam where not → need per-group routing
```

---

## Megatron Muon Blocker Dependency Chain

```
★★★★★★★★★ All 4 Megatron Muon blockers MUST resolve for production RTX 4090:

1. #5219 (crash fix) → MUST merge FIRST → enables initialization on single GPU
   → dp_cp_params_list = None → TypeError → None guard fix → Final Review → stalled

2. #5395 (clipping skip) → MUST merge SECOND → enables correct training
   → skip_grad_norm_clip attribute → +15/-1 → 0 reviews → stalled!
   → ★★★★★★★★ Without this: global clipping → Muon stalls → training divergence

3. #5400 (GatedDeltaNet routing) → MUST merge → prevents architecture-level incompatibility
   → skip_orthogonalization attribute → +14/-1 → OPEN
   → → Prevents NaN/divergence from Newton-Schulz on heterogeneous weights

4. #5391 (compact DDP) → SHOULD merge → memory efficiency
   → Removes dp_size padding on dp=1 → +218/-58 → DRAFT → significant change

★★★★★★★★★ + #5179 (Muon PyPI stub v999.9.9) → can't even install Muon! → 4th blocker
```

---

## Key Findings

★★★★★★★★★ #5400: GatedDeltaNet in_proj INCOMPATIBLE with Muon → 6th Muon blocker
★★★★★★★★★ skip_orthogonalization=True → generic opt-out mechanism → future-proof
★★★★★★★★★ Two Muon failure modes: structural (#5400/#7939) + clipping (#5394/#8068)
★★★★★★★★★ Muon is NOT universal → hybrid optimizer (Muon+Adam per-group) needed
★★★★★★★★★ +14/-1 lines → simple fix → but part of 4-blocker dependency chain
★★★★★★★★★ RTX 4090: AdamW + CPU_Adam remains ONLY safe optimizer → Muon NOT viable

---

## References

- Megatron #5400: https://github.com/NVIDIA/Megatron-LM/pull/5400 (GatedDeltaNet→Adam routing)
- Megatron #5219: https://github.com/NVIDIA/Megatron-LM/pull/5219 (single-GPU crash fix)
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (clipping stalls Muon)
- Megatron #5395: https://github.com/NVIDIA/Megatron-LM/pull/5395 (skip_grad_norm_clip)
- DeepSpeed #7939: https://github.com/microsoft/DeepSpeed/issues/7939 (CPU offload BLOCKED)
- DeepSpeed #7776: https://github.com/microsoft/DeepSpeed/issues/7776 (clipping bug)
- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/issues/8068 (default clipping)
- Cross-framework Muon synthesis: notebook/fundamentals/cross-framework-muon-optimizer-status-synthesis.md
