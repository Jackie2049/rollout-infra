# SGLang #28676 — MXFP8 Flashinfer TRTLLM Routed MoE for V4: RL Weight Update Cache Bug Fix + Routed Scaling Enablement

> 2026-06-18 | PR #28676 OPEN | Author: xiuhu17 (zhihaow6@illinois.edu) | +28/-2, 3 files
> ★★★★★★★★ CRITICAL BUG FIX: MXFP8 MoE weight shuffle index cache gets CLOBBERED on RL weight reload → 64x accuracy blowup (0.06→3.83)!
> ★★★★★★★★ DSV4 enablement: routed_scaling_factor on output now implemented → required for V4 routed kernel path
> ★★★★★★★★ Pattern: GPU-resident cache clobbered by weight-region reuse → SAME class as DSV4 systematic instability (per-step dynamic data MUST NOT persist stale)
> ★★★★★★★★ Affects ANY FP8 MoE on flashinfer-trtllm path going through weight update → not just V4!

---

## 1. Issue/PR Metadata

```
PR Number:        #28676
Title:            [RL] MXFP8 flashinfer_trtllm_routed MoE for V4
Author:           xiuhu17 (zhihaow6@illinois.edu — zhihaow6)
State:            OPEN (draft=false)
Created:          2026-06-18T20:01:42Z
Updated:          2026-06-18T21:17:59Z
Merged:           NULL (not yet merged)
Additions:        +28
Deletions:        -2
Changed Files:    3
Commits:          1 (5c474840 — "update")
Labels:           NONE
Assignees:        NONE
Mergeable State:  blocked

Requested Reviewers (8):
  HaiShaw, Ying1123, merrymercy, Edwardf0t1, ispobock, BBuf, ch-wan, Fridge003
  → ★★★★★★★★ Heavy reviewer set → MoE quantization + RL experts → needs thorough review

CI Status:        FAILED
  - Base test: Run #27785909669 (RED X)
  - Extra test: Run #27785909304 (RED X)
  → ★★★★★★★★ CI failing → needs fixes before merge

Review Comments:  0 inline code comments
Issue Comments:   1 (gemini-code-assist bot quota warning)
Reviews:          0 formal reviews submitted
```

---

## 2. Full Description (PR Body)

```
## Motivation

DeepSeek-V4 (MXFP8) on the `flashinfer_trtllm_routed` MoE path breaks after the
first RL weight update: `train_rollout_logprob_abs_diff` jumps from ~0.06 to
**~3.83**. Steady-state is fine — the bug is specific to the weight-reload path.

## Root Cause

`align_mxfp8_moe_weights_for_flashinfer_trtllm` shuffles MoE weights/scales into
the kernel layout using row-permutation index tensors that depend only on shape,
so they're memoized in the GPU cache `_flashinfer_trtllm_shuffle_row_indices_cache_mxfp8`
(added in #21280).

These cached tensors are GPU-resident. On a weight update, sglang reuses the
weights-region GPU memory, **clobbering the cached index tensors**. The cache
still hits (same shape) but the contents are now garbage, so the post-update
`align` permutes the new weights with stale indices → corrupted layout → the 3.83
blowup. Affects any FP8 MoE on the flashinfer-trtllm path that goes through a
weight update, not just V4.

## Changes

1. **Bug fix** — add `clear_mxfp8_shuffle_index_cache()` in `flashinfer_trtllm.py`
   and call it from the weight-load funnel in `fused_moe_triton/layer.py`
   (`_weight_loader_impl` + `weight_loader_fused`, gated on `Fp8MoEMethod`). This
   funnel covers both the colocate and distributed/EP update paths, so the next
   `align` recomputes correct indices after any reload.

2. **V4 enablement** — implement `apply_routed_scaling_factor_on_output` in
   `hash_topk.py` (store the flag, drop the `not implemented` assert, apply
   `topk_weights *= routed_scaling_factor` in `forward`), as required by the
   routed kernel.

## Testing

5-step DeepSeek-V4 MXFP8 RL run (weight update before each step):

| step | 0 | 1 | 2 | 3 | 4 |
|------|------|------|------|------|------|
| `train_rollout_logprob_abs_diff` | 0.069 | 0.062 | 0.068 | 0.065 | 0.062 |

All steps healthy (~0.06), no 3.83 blowup.
```

---

## 3. Comments and Reviews

```
★★★★★★★★★ Current review status:

Issue Comments:
  - gemini-code-assist[bot] (2026-06-18T20:01:46Z): quota limit warning
    → Bot auto-comment → NO substantive content

Review Comments (inline): 0
  → ★★★★★★★★ NO inline code review comments → PR just opened

Formal Reviews: 0
  → ★★★★★★★★ NO submitted reviews → 8 reviewers requested but none responded yet

Timeline Events:
  - review_requested × 8 (all reviewers: HaiShaw, Ying1123, merrymercy, etc.)
  - renamed event (2026-06-18T21:17:59Z) — title was updated after initial submission

★★★★★★★★★ PR is NEW (< 24 hours) → reviews expected to come in over next days
★★★★★★★★★ CI is failing → this is the first hurdle to address
★★★★★★★★★ 8 reviewers = high-stakes PR → MoE quantization + RL path requires expert scrutiny
```

---

## 4. Technical Analysis

### 4.1 MXFP8 Quantization for MoE — What It Is

```
★★★★★★★★★ MXFP8 = OCP Microscaling FP8 format:

Format:        FP8 E4M3 elements + FP8 E8M0 block scales (block_size=32)
Block Scale:   E8M0 = 8 exponent bits, 0 mantissa → pure power-of-2 scale
Quantization:  Per-block scale × FP8 element → higher accuracy than plain FP8

Why MXFP8 for MoE?
  → MoE experts have diverse weight distributions → per-tensor FP8 too coarse
  → MX block_size=32 → finer granularity → per-expert quantization accuracy
  → DeepSeek V3/V4 uses MXFP8 for MoE expert weights → OCP standard alignment
  → ★★★★★★★★ MXFP8 is the quantization format for DSV4 MoE RL training!

From quantization theory note (quantization-theory-mathematical-derivation.md):
  → MX block_size=32 → overhead = 32/32 * 1 byte scale = 3.125% overhead
  → E8M0 scale = pure power-of-2 → hardware-friendly → no mantissa cost
  → ★★★★★★★★ This is MORE accurate than block-wise FP8 (block_size=128) by 4x granularity!
  → But: MXFP8 requires 1D quantization → contraction axis matters → layout shuffle needed

★★★★★★★★★ MX vs plain FP8 for MoE:
  | Format    | Block Size | Granularity | Overhead | DSV4 Use    |
  |-----------|-----------|-------------|----------|-------------|
  | FP8       | 128       | coarse      | 1.6%     | V3/V4 attn  |
  | MXFP8     | 32        | 4x finer    | 3.1%     | V4 MoE      |
  → MXFP8 = better accuracy for MoE (diverse expert distributions)
  → But: 1D quantization → weight shuffle required → THIS is where the bug lives!
```

### 4.2 Flashinfer TRTLLM Routed MoE — How It Works

```
★★★★★★★★★ Flashinfer TRTLLM routed MoE path:

Architecture:
  → FlashInfer + TensorRT-LLM MoE kernel → hybrid approach
  → "routed" = per-token top-k expert selection → dynamic routing
  → FP8/MXFP8 quantized expert weights → flashinfer_trtllm kernel layout

Weight Alignment Pipeline:
  1. MoE weights arrive in standard HuggingFace layout
  2. `align_mxfp8_moe_weights_for_flashinfer_trtllm` SHUFFLES into kernel layout
  3. Shuffle uses row-permutation index tensors (depend on shape only)
  4. Index tensors cached in `_flashinfer_trtllm_shuffle_row_indices_cache_mxfp8`
  5. On inference: cached indices → fast shuffle → correct kernel layout

★★★★★★★★★ Why shuffle is needed:
  → MXFP8 = 1D quantization → weights need specific row ordering
  → TRTLLM kernel expects specific layout → row-permutation reshuffle
  → Permutation depends on expert count, hidden dim, top-k → SHAPE-based
  → → SHAPE doesn't change across weight updates → natural to cache

★★★★★★★★★ The BUG: cached index tensors are GPU-resident!
  → Same GPU memory region as weights → weight reload CLOBBERS the cache!
  → Cache dict still has the key (same shape) → "cache hit" → stale indices
  → Stale indices → wrong permutation → corrupted layout → 3.83 blowup

★★★★★★★★★ This is the SAME class of bug as ALL DSV4 failures:
  → Per-step dynamic data persists stale references → weight update invalidates
  → vLLM #45309 cudagraph → replay with stale metadata → garbage output
  → vLLM #45979 sparse cache → stale cache entries → GSM8K regression
  → SGLang #28612 C128 state → stale SWA mapping → accuracy degradation
  → ★★★★★★★★ UNIVERSAL pattern: DYNAMIC data MUST NOT survive weight reload boundary!
```

### 4.3 Root Cause Deep Dive — GPU Memory Region Reuse

```
★★★★★★★★★ The root cause is GPU memory management:

Weight lifecycle in RL training:
  1. Initial load → weights in GPU memory region A
  2. align_mxfp8 → shuffle indices computed → cached as GPU tensors in region A
  3. RL training step → gradients computed → optimizer updates
  4. Weight update → sglang reloads weights → SAME GPU memory region A reused
  5. ★★★★★★★★ Region A now has NEW weights BUT indices from step 2 are ALSO in region A!
  6. New weights OVERWRITE the cached indices → indices are now garbage
  7. Cache dict still maps shape → (shape) → "hit" → returns garbage indices
  8. align_mxfp8 with garbage indices → wrong permutation → 64x error (0.06→3.83)

★★★★★★★★★ Why this only affects RL path (not static inference):
  → Static inference: weights loaded ONCE → never updated → cache persists correctly
  → RL training: weight update EVERY step → cache invalidated but not cleared
  → ★★★★★★★★ RL weight reload = the boundary condition that exposes this bug!

★★★★★★★★★ Memory region reuse pattern:
  → sglang weight-loader reuses GPU memory for efficiency (no realloc)
  → This is CORRECT for weights (they need the same shape)
  → But: it CLOBBERS adjacent cached tensors that share the same memory pool
  → ★★★★★★★★ Cache invalidation MUST happen at weight-update boundary!
  → → This is the SAME lesson as verl sleep/wake: weight sync must invalidate stale refs
```

### 4.4 The Fix — Source-Level Analysis

```
★★★★★★★★★ Fix 1: clear_mxfp8_shuffle_index_cache() (8 lines, flashinfer_trtllm.py)

FILE: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-0)

Implementation:
  def clear_mxfp8_shuffle_index_cache() -> None:
      """Drop the cached MXFP8 MoE row-index permutations.
      The cached index tensors are GPU-resident; sglang reuses the weights-region
      memory across weight-update cycles
      """
      _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8.clear()

★★★★★★★★★ Simple dict.clear() → removes all cached entries → next align recomputes
★★★★★★★★★ This is the MINIMAL fix → no over-engineering → just clear the cache!

★★★★★★★★★ Fix 2: call clear in weight-load funnel (14 lines, layer.py)

FILE: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-0)

Two insertion points:
  A. _weight_loader_impl (line ~804):
     elif isinstance(method, Fp8MoEMethod):
         from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
             clear_mxfp8_shuffle_index_cache,
         )
         clear_mxfp8_shuffle_index_cache()

  B. weight_loader_fused (line ~1033):
     if isinstance(method, Fp8MoEMethod):
         from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
             clear_mxfp8_shuffle_index_cache,
         )
         clear_mxfp8_shuffle_index_cache()

★★★★★★★★★ Gated on Fp8MoEMethod → only affects FP8/MXFP8 MoE paths → clean
★★★★★★★★★ Covers BOTH update paths:
  → _weight_loader_impl: colocate path (single GPU weight update)
  → weight_loader_fused: distributed/EP path (multi-GPU weight update)
★★★★★★★★★ ALL weight reload funnels covered → no path missed!

★★★★★★★★★ Fix 3: routed_scaling_factor_on_output (6 lines, hash_topk.py)

FILE: python/sglang/srt/layers/moe/hash_topk.py (+6/-2)

Changes:
  1. Store apply_routed_scaling_factor_on_output flag in __init__:
     self.apply_routed_scaling_factor_on_output = (
         apply_routed_scaling_factor_on_output
     )

  2. REMOVED assert not apply_routed_scaling_factor_on_output, "not implemented"
     → ★★★★★★★★ This was a BLOCKER → V4 routed kernel REQUIRED this feature!

  3. Apply in forward:
     if self.apply_routed_scaling_factor_on_output:
         topk_weights = topk_weights * self.routed_scaling_factor

★★★★★★★★★ Why routed_scaling_factor matters for V4:
  → DSV4 MoE uses routed_scaling_factor = scaling applied AFTER expert combination
  → Previous assert blocked this → V4 couldn't use flashinfer_trtllm_routed path
  → Now: topk_weights *= routed_scaling_factor → correct V4 MoE output scaling
  → ★★★★★★★★ This unblocks DSV4 MXFP8 MoE on the routed kernel path!
```

---

## 5. RTX 4090 Impact Assessment

```
★★★★★★★★★ RTX 4090 relevance of #28676:

DIRECT IMPACT:
  → This bug affects ANY FP8 MoE on flashinfer_trtllm_routed path
  → RTX 4090 GRPO training uses verl → SGLang rollout → FP8 MoE models
  → ★★★★★★★★ If doing MoE GRPO on RTX 4090 with FP8 weights → THIS BUG WILL HIT YOU!
  → Without this fix: first weight update → 64x accuracy blowup → training BROKEN

★★★★★★★★★ Practical RTX 4090 scenarios:
  1. DeepSeek-V2-Lite (16B) + MXFP8 → RTX 4090 fits → RL training with weight update
     → Bug: shuffle cache clobbered → 3.83 abs_diff → training unusable!
     → Fix: clear cache → 0.06 abs_diff → training WORKS!

  2. Qwen3-30B-A3B (MoE) + FP8 → RTX 4090 barely fits with ZeRO-2+CPU_Adam
     → Same bug would hit on weight reload → MUST use this fix!

  3. Any future MoE model + FP8/MXFP8 + RL training → ALL affected!

★★★★★★★★★ Impact WITHOUT vs WITH fix:

WITHOUT #28676:
  → Step 0: abs_diff = 0.069 (healthy)
  → Step 1: abs_diff = 3.83 (BROKEN — 64x blowup!)
  → Step 2-4: abs_diff = ??? (garbage → training dead)

WITH #28676:
  → Step 0: abs_diff = 0.069
  → Step 1: abs_diff = 0.062 (healthy!)
  → Step 2: abs_diff = 0.068 (healthy!)
  → Step 3: abs_diff = 0.065 (healthy!)
  → Step 4: abs_diff = 0.062 (healthy!)
  → ★★★★★★★★ ALL steps healthy → RL training viable!

★★★★★★★★★ RTX 4090 GRPO training impact:
  → verl GRPO+bypass → SGLang rollout → MoE model → FP8/MXFP8
  → Weight update every RL step → MUST have cache invalidation
  → ★★★★★★★★ Without this fix → RTX 4090 MoE GRPO = IMPOSSIBLE!
  → ★★★★★★★★ With this fix → RTX 4090 MoE GRPO = VIABLE!
  → Combined with verl #6512 (per-unit LoRA) → memory feasible
  → Combined with verl #6699 (detach fix) → memory feasible

★★★★★★★★★ Connection to verl weight sync mechanism:
  → verl sleep/wake: weight reload via ZMQ IPC / NCCL → same boundary
  → SGLang weight update: weight reload from disk/IPC → same boundary
  → ★★★★★★★★ BOTH need cache invalidation at weight-reload boundary!
  → verl FSDP weight sync: already has summon/unload cycle → BUT:
    → Does verl's SGLang weight update path call this cache clear?
    → ★★★★★★★★ NEED TO VERIFY: verl weight sync → SGLang update_weights → does it go through
      the weight-load funnel that calls clear_mxfp8_shuffle_index_cache?
    → If verl uses update_weights_from_disk → YES → this funnel is covered!
    → If verl uses custom weight transfer → NEED TO CHECK!
```

---

## 6. Connection Map

### 6.1 Connection to SGLang #28618/#28620 (SM89 DSV4-Flash-FP8)

```
★★★★★★★★★ #28676 vs #28618/#28620 — complementary DSV4 fixes:

#28618/#28620: SM89 kernel fallback → SM90 crashes on SM89 → Triton fallback
  → Problem: HARDWARE capability → SM89 lacks TMA/wgmma/clusters
  → Fix: Triton paged MQA logits + FlashMLA fallback
  → Scope: SM89-specific → only affects SM89/RTX 4090/L20

#28676: MXFP8 MoE weight cache clobber → RL weight update breaks cache
  → Problem: MEMORY MANAGEMENT → GPU cache clobbered by weight reuse
  → Fix: clear cache on weight reload boundary
  → Scope: ALL hardware → affects SM89 AND SM90 AND SM100 AND SM120!

★★★★★★★★★ Together they complete the DSV4 MXFP8 MoE RL path:
  → #28620: makes DSV4 inference WORK on SM89 (kernel level)
  → #28676: makes DSV4 RL training WORK on all hardware (weight update level)
  → ★★★★★★★★ BOTH needed for RTX 4090 DSV4 MXFP8 GRPO training!
  → #28620 alone → inference works but RL weight update breaks
  → #28676 alone → RL works on SM90 but SM89 still crashes on inference
  → BOTH → inference + RL = complete DSV4 deployment pathway!

★★★★★★★★★ Sequential dependency:
  → #28676 builds on #21280 (MXFP8 support, MERGED April 4)
  → #21280 added the shuffle index cache → #28676 fixes the cache invalidation
  → #28620 builds on existing DSV4 Flash attention path → adds SM89 fallback
  → ★★★★★★★★ #28676 = incremental fix on #21280 → smaller PR → faster review likely
```

### 6.2 Connection to Quantization Theory (MX OCP Standard, MXFP8 Format)

```
★★★★★★★★★ From quantization-theory-mathematical-derivation.md:

MXFP8 Format:
  → Element type: FP8 E4M3 (4 exponent, 3 mantissa bits)
  → Block size: 32 (vs plain FP8 block_size=128)
  → Scale type: FP8 E8M0 (pure power-of-2, no mantissa)
  → Range: [-448 × 2^s, 448 × 2^s] where s = E8M0 scale
  → Granularity: 4x finer than plain FP8 block-wise → better for MoE

★★★★★★★★★ Why MXFP8 needs shuffle (and plain FP8 doesn't as much):
  → MXFP8 = 1D quantization → specific contraction axis → row ordering matters
  → Plain FP8 = block-wise (block_size=128) → 2D quantization → layout less critical
  → TRTLLM kernel expects MXFP8 in specific row-major layout → shuffle required
  → ★★★★★★★★ The shuffle index cache BUG is specific to MXFP8 → not plain FP8!
  → Plain FP8 MoE path → different alignment → no shuffle index cache → not affected

★★★★★★★★★ MX format cross-framework status:
  → NVIDIA Blackwell (SM100): native MX support in hardware
  → RTX 5090 (SM120): MX-capable → P3 FP4/MXFP4 target
  → vLLM-Ascend #10730: MX quant fusion for DSV4 → Ascend pathway
  → SGLang #21280 (MERGED): MXFP8 support for DeepSeek V3
  → SGLang #28676: MXFP8 MoE RL weight update fix
  → ★★★★★★★★ MX is becoming the standard quant format for MoE → cross-framework!
```

### 6.3 Connection to DeepSpeed #8066 (Per-Policy Dtype)

```
★★★★★★★★★ DeepSpeed #8066 (MERGED June 16) per-policy dtype → CAUSED #8072 regression:

#8066: Allow different dtype policies per parameter type
  → Mixed-precision: some params FP8, some BF16, some FP32
  → For MoE: expert weights FP8 → router BF16 → correct mixed precision
  → ★★★★★★★★ BUT: ZeRO-3 partition dtype mismatch → regression!

#28676: MXFP8 MoE weight shuffle → cache invalidation on reload
  → Mixed-precision: expert weights MXFP8 → kv_b_proj BF16 (#21280 design decision)
  → ★★★★★★★★ #21280 explicitly chose: "keep kv_b_proj always bf16" for MLA!

★★★★★★★★★ Connection: mixed-precision MoE + weight reload = double risk:
  → #8066: dtype mismatch on ZeRO-3 partition → regression
  → #28676: cache clobber on weight reload → accuracy blowup
  → ★★★★★★★★ BOTH bugs expose at weight-update boundary → RL training reveals them!
  → ★★★★★★★★ LESSON: mixed-precision MoE + RL weight update = vulnerability cluster!
  → → Need both dtype consistency (#8073 fix) AND cache invalidation (#28676 fix)
```

### 6.4 Connection to vLLM #45683 (MoE Combine Determinism)

```
★★★★★★★★★ vLLM #45683 — Deterministic MoE combine under batch-invariant mode:

#45683: Cross-rank summation order in MoE combine is NOT stable under DP+EP
  → Breaks bit-for-bit reproducibility
  → Fix: deterministic reduce_scatterv → stable combination order

#28676: MXFP8 MoE weight cache clobber → accuracy blowup after weight update
  → Different bug but SAME domain: MoE correctness

★★★★★★★★★ Connection: MoE correctness requires BOTH:
  → #45683: deterministic combine → stable expert output aggregation
  → #28676: valid weight layout → correct expert computation
  → ★★★★★★★★ WRONG weights + deterministic combine = DETERMINISTICLY WRONG!
  → ★★★★★★★★ Correct weights + nondeterministic combine = RANDOMLY WRONG!
  → BOTH must be fixed for MoE GRPO training correctness!

★★★★★★★★★ For RTX 4090 GRPO with MoE:
  → #45683 needed for DP+EP → dp=1 RTX 4090 → EP=1 → less critical
  → #28676 needed for ALL configs → dp=1 RTX 4090 → CRITICAL!
  → ★★★★★★★★ #28676 is MORE critical for single-GPU RTX 4090 than #45683!
```

### 6.5 Connection to vLLM #45656 (MoE is_sym Guard Regression)

```
★★★★★★★★★ vLLM #45656 — GPTQ/CT MoE is_sym guard regression:

#45656: symmetric vs asymmetric quantization config NOT checked per-expert
  → GPTQ MoE models → is_sym guard → regression → garbage output
  → Fix: restore per-expert is_sym check

#28676: MXFP8 MoE shuffle cache → stale indices after weight update
  → Different quantization format but SAME domain: quantized MoE correctness

★★★★★★★★★ Pattern: quantized MoE has MORE correctness bugs than dense models!
  → Router sensitivity (discrete decisions)
  → Expert weight layout (shuffle/reorder)
  → Quantization format (MXFP8 vs FP8 vs GPTQ → different alignment needs)
  → ★★★★★★★★ MoE + quantization = COMPOUND risk → each format has unique pitfalls
  → → MXFP8: shuffle cache clobber
  → → GPTQ: is_sym guard
  → → FP8: per-block scale alignment
  → → ALL need careful handling on weight update!
```

### 6.6 Connection to DSV4 Systematic Instability Pattern

```
★★★★★★★★★ #28676 is the 10th DSV4-related issue/fix:

FAILURE TRACK (updated from dsv4-systematic-instability-pattern-synthesis.md):
  1. vLLM #45309→#45972: cudagraph → garbage output → MERGED revert
  2. SGLang #26471→#28591: MTP online compress → accuracy regression → OPEN revert
  3. SGLang #27749→#28575: MTP weight update distributed → refactor needed
  4. SGLang #28569: EAGLE3 CUDA graph → ILLEGAL MEMORY ACCESS
  5. vLLM #45979: sparse cache → GSM8K regression → CLOSED (false alarm)
  6. SGLang #28520: MTP swa_loc cache → EAGER mode bug (NOT CUDA graph!)
  7. vLLM-Ascend #10645: DSV4 chat template → FIXED
  8. vLLM-Ascend #10724: DSV4 PD-Mix crash → OPEN
  9. SGLang #28612: C128 state mapping lifecycle → OPEN fix
  10. ★★★★★★★★ SGLang #28676: MXFP8 MoE shuffle cache clobber → OPEN fix

★★★★★★★★★ COMMON THREAD across ALL 10:
  → Per-step dynamic data MUST NOT persist stale references across weight updates
  → Cache invalidation MUST happen at weight-reload boundary
  → ★★★★★★★★ UNIVERSAL RULE: ANY cached GPU tensor that depends on weight data
    → MUST be invalidated when weights change → RL training exposes this!

★★★★★★★★★ #28676 pattern classification:
  → Class: GPU memory reuse clobber (weight region → adjacent cache destroyed)
  → NOT: CUDA graph replay (no CUDA graph involved!)
  → NOT: stale Python dict cache (GPU tensors clobbered, not just stale references!)
  → ★★★★★★★★ NEW subclass: physical memory clobber vs logical stale reference
    → Most DSV4 bugs: logical stale reference (cache hit returns old data)
    → #28676: PHYSICAL clobber (memory overwritten → data is GARBAGE, not just stale!)
    → → This is WORSE than stale reference → garbage data vs outdated data!
```

### 6.7 Connection to DeepSeek V4 Architecture and Architecture Evolution

```
★★★★★★★★★ From llm-architecture-evolution.md:

Architecture generations:
  V2: MLA + MoE → KV压缩 + 计算稀疏
  V3: MLA + MoE + MTP → KV + sparse + multi-token
  V4: MLA + MoE + MTP + DSA + OnlineCompress → 5层动态路由!

★★★★★★★★★ #28676 specifically affects V4's MoE layer:
  → V4 MoE uses MXFP8 (not plain FP8) → finer quantization granularity
  → V4 MoE uses routed_scaling_factor → expert output scaling after combine
  → V4 MoE uses flashinfer_trtllm_routed → specific kernel layout

★★★★★★★★★ Why V4 MoE is MORE fragile than V3 MoE:
  → V3 MoE: FP8 + standard layout → simpler alignment → fewer bugs
  → V4 MoE: MXFP8 + shuffle layout + routed scaling → MORE alignment steps
  → → MORE steps = MORE caching = MORE potential stale data bugs!
  → ★★★★★★★★ V4's complexity makes it the most fragile production MoE model!

★★★★★★★★★ Architecture evolution lesson:
  → Each generation adds a new dynamic routing layer
  → Each new layer adds a new caching/staleness risk
  → V4 = 5 dynamic routing layers = 5 failure points
  → ★★★★★★★★ The lesson for future architectures: MORE dynamic layers = MORE bugs!
  → → Need systematic cache invalidation at ALL weight-update boundaries
```

---

## 7. Key Takeaways

```
★★★★★★★★★ 10 key takeaways from #28676:

1. MXFP8 MoE shuffle cache gets CLOBBERED by GPU weight-region reuse → 64x accuracy blowup!
2. Bug is specific to RL weight-update path → static inference unaffected
3. Fix is MINIMAL: dict.clear() on cache + call from weight-load funnel (28 lines total)
4. Fix covers BOTH colocate AND distributed/EP weight update paths → no path missed
5. routed_scaling_factor_on_output unblocks V4 MoE on flashinfer_trtllm_routed path
6. ★★★★★★★★ This is the 10th DSV4 issue → confirms systematic instability pattern
7. ★★★★★★★★ NEW subclass: PHYSICAL memory clobber (not just logical stale reference!)
8. ★★★★★★★★ UNIVERSAL RULE: ANY GPU-resident cache MUST be invalidated at weight-reload boundary
9. ★★★★★★★★ RTX 4090 MoE GRPO WITHOUT this fix = IMPOSSIBLE → WITH this fix = VIABLE
10. ★★★★★★★★ #28676 + #28620 together = complete DSV4 MXFP8 MoE RL pathway for RTX 4090!
```

---

## 8. Monitoring Status

```
★★★★★★★★★ PR #28676 monitoring items:

MUST WATCH:
  → CI status: currently FAILED → needs author fix or maintainer CI trigger
  → Review progress: 0 reviews submitted, 8 reviewers requested → needs attention
  → Merge timeline: blocked state → CI must pass first
  → ★★★★★★★★ This is a CRITICAL RL bug → merge should be prioritized!

CROSS-REFERENCE MONITORING:
  → #21280 (MXFP8 support, MERGED April 4): foundation that #28676 builds on
  → #28618/#28620 (SM89 DSV4, OPEN): complementary → BOTH needed for RTX 4090
  → #28612 (C128 state mapping, OPEN): same DSV4 pattern → stale data bug
  → vLLM #45683 (MoE combine determinism, OPEN): MoE correctness → same domain
  → DeepSpeed #8072/#8073 (ZeRO-3+PEFT regression, OPEN): mixed-precision MoE + RL

VERL INTEGRATION CHECK:
  → ★★★★★★★★ MUST verify: does verl's SGLang weight update path go through the funnel
    that calls clear_mxfp8_shuffle_index_cache?
  → If verl uses update_weights_from_disk → YES → covered by #28676
  → If verl uses custom ZMQ IPC weight transfer → NEED TO CHECK!
  → This is critical for RTX 4090 verl GRPO pipeline safety!

PRIORITY: HIGH
  → Direct RTX 4090 impact → MoE GRPO training BLOCKED without this fix
  → Small PR (+28/-2) → should merge quickly once CI passes
  → ★★★★★★★★ This is NOT optional for any MoE RL training with MXFP8!
```

---

## References

- SGLang PR #28676: https://github.com/sgl-project/sglang/pull/28676
- SGLang PR #21280 (MXFP8 original, MERGED): https://github.com/sgl-project/sglang/pull/21280
- SGLang RFC #28618: https://github.com/sgl-project/sglang/issues/28618
- SGLang PR #28620: https://github.com/sgl-project/sglang/pull/28620
- SGLang #28612 (C128 state mapping fix): https://github.com/sgl-project/sglang/pull/28612
- vLLM #45683 (MoE combine determinism): https://github.com/vllm-project/vllm/pull/45683
- vLLM #45656 (MoE is_sym guard): https://github.com/vllm-project/vllm/pull/45656
- vLLM-Ascend #10730 (MX quant fusion): https://github.com/vllm-project/vllm-ascend/pull/10730
- DeepSpeed #8066 (per-policy dtype, MERGED): https://github.com/microsoft/DeepSpeed/pull/8066
- DeepSpeed #8072/#8073 (ZeRO-3+PEFT regression): OPEN
- verl #6512 (per-unit LoRA summon, MERGED June 18): 10x memory reduction
- verl FSDP weight sync mechanism: 902-line deep reading
- DSV4 systematic instability pattern: 10 failures across 3 frameworks
- Quantization theory: MXFP8 OCP standard (block_size=32, E8M0 scale)
- Architecture evolution: V4 = 5-layer dynamic routing → most fragile model
- P9 Inductor SM<90 Fusion Guard: same SM89 capability gap root cause
