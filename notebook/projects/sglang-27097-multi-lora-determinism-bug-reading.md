# SGLang #27097 — Concurrent Multi-LoRA Determinism Bug Source Reading

> 2026-06-18 | Issue #27097 (OPEN, June 3) | Author: glaziermag (CONTRIBUTOR)
> ★★★★★★★★ Concurrent multi-LoRA requests diverge under deterministic inference
> ★★★★★★★★ Sequential is stable → concurrency is the problem → 4 interacting factors
> ★★★★★★★★ HIGH severity for correctness → MEDIUM-HIGH for GRPO training

---

## 1. Bug Description

```
★★★★★★★★★ The bug:
  → --enable-deterministic-inference + temperature=0 + concurrent multi-LoRA
  → Same (adapter, prompt) returns DIFFERENT text when concurrent requests
  → Sequential baseline: perfectly stable → identical outputs every time
  → Concurrent: 146 mismatches out of 204 audit items (even with no eviction!)

★★★★★★★★★ Reproduction:
  → A100-SXM4-80GB → verified on both max_loras_per_batch=4 (eviction) and 16 (no eviction)
  → No-eviction control STILL diverges → root cause NOT just eviction/slot reuse
  → ★★★★★★★★ Even with all adapters pre-loaded → concurrent batches diverge
```

---

## 2. Root Cause: 4 Interacting Factors

```
★★★★★★★★★ Factor 1 — SGMV Triton kernel dynamic routing (batch-dependent):
  → _sgemm_lora_a_kernel/_sgemm_lora_b_kernel use tl.load(weight_indices + batch_id)
  → This is DYNAMIC load → NOT tl.constexpr
  → weight_indices mapping depends on which adapters occupy which buffer slots
  → Concurrent requests change batch composition → same adapter → different slot index
  → Segment grouping, seg_indptr offsets, padding patterns differ between batches
  → ★★★★★★★★ Numerical divergence in float32 accumulation loop

★★★★★★★★★ Factor 2 — CUDA graph metadata replay with variable mappings:
  → prepare_lora_batch() (lora_manager.py:305-337) maps LO RA ID → current buffer slot
  → Under concurrent load → this mapping changes per-batch
  → CUDA graphs replay with dynamically-written metadata → same adapter → different routes
  → ★★★★★★★★ Same adapter may be routed through different slot indices in different batches

★★★★★★★★★ Factor 3 — KV cache stale state under concurrent traffic:
  → ReqToTokenPool.free returns freed slots WITHOUT zeroing previous occupant token indices
  → Racy alloc/use ordering → exposes stale KV state
  → Overlap scheduler exacerbates this interaction
  → ★★★★★★★★ Independent of LoRA but compounds divergence under concurrent load

★★★★★★★★★ Factor 4 — flashinfer attention batch-dependency:
  → flashinfer attention backend → non-deterministic under concurrent shared-prefix burst + reuse
  → Triton attention backend → STABLE → no batch-dependent behavior
  → ★★★★★★★★ Another independent batch-dependent path → compounds multi-LoRA divergence
```

---

## 3. RTX 4090 GRPO Implications

```
★★★★★★★★★ Impact: MEDIUM-HIGH for GRPO correctness

1. Single-adapter GRPO still affected:
  → GRPO generates n=8 responses per prompt → batched together in rollout
  → SGMV kernel processes as multiple segments → batch composition changes between steps
  → ★★★★★★★★ Creates exactly the concurrent scenario described in #27097

2. verl HYBRID + SGLang:
  → Multiple prompts from different trajectory groups → batched together
  → With LoRA → concurrent multi-LoRA batches
  → Divergence bug → reward computation inconsistent across runs

3. temperature>0 masks but doesn't fix:
  → GRPO uses temperature>0 for exploration → sampling noise masks divergence
  → BUT underlying computation error persists → subtle logprob drift → accumulates

★★★★★★★★★ Workaround:
  → Run rollout sequentially (concurrency=1) for deterministic verification
  → For training: noise may be tolerable → but validate on RTX 4090 empirically
  → Triton attention backend (not flashinfer) → eliminates Factor 4

★★★★★★★★★ SGLang SM89 specific:
  → SGMV kernel uses tl.constexpr for BLOCK sizes → BUT dynamic routing for weight_indices
  → NOT the same as Inductor batch-invariant gap (autotuned Triton)
  → ★★★★★★★★ Different class of batch-dependent behavior in SGLang's own kernels
```

---

## 4. Related Issues

```
★★★★★★★★★ Related SGLang issues (ALL unresolved):

#24845 (OPEN): Base-mode non-deterministic under cancel/retry + concurrent
  → KV cache stale state → freed slots not zeroed → overlap scheduler interaction

#24853 (OPEN): flashinfer attention non-deterministic under concurrent
  → Triton attention stable → flashinfer diverges → same concurrency root cause class

#25864 (OPEN): LoRARegistry counter negative → concurrent LoRA accounting fragility

#25307 (OPEN): PiecewiseCudaGraphRunner drops dummy lora_ids during forced capture

#24638 (OPEN): Base-model request can collide with LoRA radix-cache namespace

★★★★★★★★★ Pattern: 5+ concurrent LoRA/CUDA graph bugs → systematic fragility → needs architectural fix
```

---

## Key Findings Summary

★★★★★★★★★ #27097: concurrent multi-LoRA diverges under deterministic → 4 interacting factors
★★★★★★★★★ Root cause: SGMV dynamic routing + CUDA graph replay + KV stale state + flashinfer batch-dependency
★★★★★★★★★ No-eviction control STILL diverges → NOT just slot reuse → dynamic routing is core issue
★★★★★★★★★ RTX 4090 GRPO: single-adapter GRPO affected → n=8 responses create concurrent batches
★★★★★★★★★ Workaround: sequential rollout (concurrency=1) or Triton attention backend
★★★★★★★★★ 5+ related issues → systematic concurrent LoRA fragility → needs architectural fix
★★★★★★★★★ SGLang team: zero response in 15 days → no fix proposed → needs community engagement

---

## References

- SGLang #27097: https://github.com/sgl-project/sglang/issues/27097
- SGLang #24845: https://github.com/sgl-project/sglang/issues/24845
- SGLang #24853: https://github.com/sgl-project/sglang/issues/24853
- SGLang #25864: https://github.com/sgl-project/sglang/issues/25864
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- SGLang deterministic source: notebook/projects/sglang-deterministic-inference-source-reading.md
