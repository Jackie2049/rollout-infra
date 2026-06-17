# SGLang #28499/#28371 — csgmv CUDA Graph Segment Replay Fix Deep Reading

> 2026-06-18 | PR #28371 MERGED June 14, #28499 cherry-pick MERGED June 17
> ★★★★★★★★ DIRECTLY addresses #27097 Factor 2 (CUDA graph metadata replay)
> ★★★★★★★★ But NOT a complete fix → Factors 1,3,4 still unfixed → systematic fragility
> ★★★★★★★★ Grid covers full segment buffer → stale tails neutralized → zero-length early-exit

---

## 1. Bug Mechanism: CUDA Graph Baked Segment Count

```
★★★★★★★★★ The bug in ChunkedSgmvLoRABackend under CUDA graph:

Before fix:
  → ChunkedSgmvLoRABackend sorts tokens by adapter → contiguous segments
  → Number of segments depends on how many distinct adapters appear in batch
  → CUDA graph capture (warmup) → single dummy adapter → 1 segment
  → Real decode steps → multiple distinct adapters → N segments
  → ★★★★★★★★ Triton kernels sized launch grid to CAPTURE-TIME segment count (num_segments / bs)
  → On replay with more segments than captured → extra segments' work SILENTLY SKIPPED
  → Stale tail metadata from previous batch → WRONG LoRA output
  → ★★★★★★★★ Only manifests with multiple resident adapters under CUDA graph
  → Single-adapter and eager runs → look fine → why sequential baseline is stable

★★★★★★★★★ This is exactly Factor 2 of #27097:
  → prepare_lora_batch() maps LoRA ID → current buffer slot
  → Under concurrent load → this mapping changes per-batch
  → CUDA graphs replay with dynamically-written metadata → same adapter → different routes
```

---

## 2. The Fix: Two-Part Solution

```
★★★★★★★★★ Part 1 — Grid covers full segment buffer:

Before:
  → Each kernel's grid sized to capture-time num_segments / bs
  → Only covers segments present at capture → dummy adapter → 1 segment

After:
  → Grid sized to weight_indices.shape[0] (pre-allocated max-segment buffer)
  → ★★★★★★★★ Covers EVERY segment that replay might populate
  → No segments silently skipped regardless of how many adapters are resident

★★★★★★★★★ Part 2 — Stale tail segments neutralized:

  → ChunkedSgmvLoRABackend.prepare_lora_batch now:
    1. Zeroes unused tail of weight_indices
    2. Fills seg_indptr tail with final offset → zero-length (seg_start == seg_end) entries
  → ★★★★★★★★ Kernels gain early-exit on zero-length segments
  → Extra grid blocks from Part 1 → do NO work → cheap skip
  → Masked permutation loads use other=0 → stay in-bounds

★★★★★★★★★ Additional: do_not_specialize annotation:

  → chunked_embedding_lora_a marked @triton.jit(do_not_specialize=["num_segments"])
  → Prevents kernel re-specialization on varying segment count
  → ★★★★★★★★ Same principle as tl.constexpr vs tl.load for determinism!
  → do_not_specialize = num_segments treated as runtime value → NOT baked into kernel
```

---

## 3. Modified Files

```
★★★★★★★★★ Files changed (from PR):

1. python/sglang/srt/lora/backend/chunked_backend.py
  → prepare_lora_batch: tail neutralization (zero weight_indices tail, fill seg_indptr tail)

2. python/sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py
  → Grid sizing: weight_indices.shape[0] instead of num_segments
  → Early-exit: zero-length segment check

3. python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py
  → Same grid sizing + early-exit pattern

4. python/sglang/srt/lora/triton_ops/chunked_embedding_lora_a.py
  → Grid sizing + early-exit + do_not_specialize annotation

5. python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py
  → Grid sizing for KV-B absorbed LoRA path
```

---

## 4. Relationship to #27097 Multi-LoRA Determinism Bug

```
★★★★★★★★★ How #28499 relates to #27097's 4 factors:

Factor 1 — SGMV Triton kernel dynamic routing (tl.load not tl.constexpr):
  → ★★★★★★★★ NOT fixed by #28499
  → tl.load(weight_indices + batch_id) remains DYNAMIC
  → Same adapter → different slot index depending on batch composition
  → Numerical divergence in float32 accumulation loop
  → This is the CORE issue → root cause of divergence even without eviction

Factor 2 — CUDA graph metadata replay with variable mappings:
  → ★★★★★★★★ PARTIALLY fixed by #28499
  → Grid now covers full segment buffer → no silent segment skipping
  → Stale tails neutralized → no wrong LoRA output from stale metadata
  → BUT: weight_indices mapping still varies per-batch → same adapter → different slot
  → ★★★★★★★★ #28499 fixes the SEGMENT SKIPPING problem, NOT the ROUTING problem
  → Routing (which slot = which adapter) is still dynamic → still divergence source

Factor 3 — KV cache stale state under concurrent traffic:
  → NOT fixed by #28499 (independent issue)
  → ReqToTokenPool.free returns freed slots WITHOUT zeroing

Factor 4 — flashinfer attention batch-dependency:
  → NOT fixed by #28499 (independent issue)
  → Triton attention stable → flashinfer diverges

★★★★★★★★★ Net effect of #28499 on #27097:
  → Fixes the most SEVERE manifestation (silent segment skipping → garbled output)
  → Does NOT fix the numerical divergence from dynamic routing (Factor 1)
  → ★★★★★★★★ Still diverges under deterministic inference → still 146/204 mismatches
  → But now divergence is SMALLER (no garbled segments → only float32 accumulation order differences)
```

---

## 5. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 GRPO impact:

Before #28499:
  → LoRA csgmv under CUDA graph → segments skipped → garbled LoRA output
  → CRITICAL for GRPO → wrong LoRA deltas → wrong rewards → training instability

After #28499:
  → LoRA csgmv CUDA graph stable → no segment skipping → correct LoRA output
  → ★★★★★★★★ Still has Factor 1 divergence (dynamic routing) → but smaller magnitude
  → For GRPO training → temperature>0 masks small divergence → likely acceptable
  → For deterministic verification → still not fully reproducible

★★★★★★★★★ Config recommendation:
  → MUST use SGLang version with #28499 (v0.5.13+ or later)
  → For training GRPO: csgmv backend + CUDA graph = now SAFE for output correctness
  → For deterministic verification: sequential rollout (concurrency=1) or Triton attention backend
```

---

## Key Findings Summary

★★★★★★★★★ #28499/#28371: csgmv CUDA graph segment replay fix → PARTIALLY fixes #27097 Factor 2
★★★★★★★★★ Fix: grid covers full segment buffer + stale tails neutralized + do_not_specialize annotation
★★★★★★★★★ Factor 1 (SGMV dynamic routing) → NOT fixed → still divergence source
★★★★★★★★★ Factor 2 (CUDA graph replay) → PARTIALLY fixed → no segment skipping → but routing still dynamic
★★★★★★★★★ Factors 3,4 → NOT fixed → independent issues
★★★★★★★★★ RTX 4090 GRPO: csgmv CUDA graph now SAFE for output correctness → still minor divergence
★★★★★★★★★ Same problem class as P9 Inductor Fusion Guard: batch-dependent kernel parameters

---

## References

- SGLang #28371: https://github.com/sgl-project/sglang/pull/28371
- SGLang #28499: https://github.com/sgl-project/sglang/pull/28499
- SGLang #27097: https://github.com/sgl-project/sglang/issues/27097
- #27097 source reading: notebook/projects/sglang-27097-multi-lora-determinism-bug-reading.md
- P9 Fusion Guard: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
