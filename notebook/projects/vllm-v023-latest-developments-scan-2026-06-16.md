# vLLM Post-v0.23.0 Developments Scan (2026-06-16)

v0.23.0 released June 15 with 408 commits / 200 contributors. This scan covers activity from June 15-16.

---

## RTX 4090 / SM89 Impact (Priority Order)

### 1. FP8 KV Guard PR #45038 -- SM89 VALIDATED, NOT MERGED
- **Status**: OPEN (not merged into v0.23.0)
- **Key update**: Community member samimh23 tested fix on L4 (SM89, 24GB) with NuExtract3-FP8 model -- confirmed it works under stress test
- **Author response**: devangpratap added compute capability info to warning message (e.g., "compute capability 8.9")
- **RTX 4090 impact**: HIGH. This is the guard that prevents compressed-tensors FP8 KV crash on SM89. Still needs merge.
- **Related bug**: #44879 remains OPEN with 0 comments since June 9

### 2. Watermark PR #44594 -- MERGED (in v0.23.0)
- **Merged**: June 11, included in v0.23.0 release
- **RTX 4090 impact**: HIGH. Preemptions -82%, ITL p99 -56%, throughput +5.1%. Must set watermark=0.05.

### 3. INT4 Triton Fallback #43731 -- MERGED (in v0.23.0)
- **Merged**: May 27, included in v0.23.0
- **RTX 4090 impact**: HIGH. W4A16 Triton fallback for non-Marlin-aligned shapes on SM89. Closes SM89 quantization gap.

### 4. HMA-by-Default #41847 -- MERGED (in v0.23.0)
- **Merged**: May 26, included in v0.23.0
- **RTX 4090 impact**: MEDIUM. 8 HMA connectors, startup OOM prevention for 24GB GPUs. SlidingWindow prefix caching caveat.

### 5. Batch Invariance #39096 -- NO ACTIVITY
- **Status**: OPEN, last updated April 17
- **RTX 4090 impact**: CRITICAL but stalled. No upstream progress. Inductor Fusion Guard (PyTorch upstream) remains our best path.

### 6. modelopt_mixed on SM80/86 (#45306) -- OPEN
- **Opens NVFP4 + FP8 on Ampere (SM80/86)** using Marlin W4A16 which needs cc >= 7.5 only
- Does NOT directly cover SM89 but shows trend: quant support expanding below SM90
- **RTX 4090 impact**: LOW directly, but signals that SM89 quant gaps are being filled incrementally

### 7. TurboQuant FP8 v4 Store (#45748) -- OPEN (just created June 16)
- Native CUDA store for Hopper+ (SM90+). Triton fallback remains for SM89.
- **RTX 4090 impact**: NONE directly (SM90+ only). Triton fallback path confirmed still available.

---

## MRv2 Model Runner Developments

### MRv2 DEFAULT for Llama + Mistral dense (in v0.23.0, #43458)
- MRv2 now covers: Qwen3, Llama, Mistral dense models by default
- Qwen2.5 still uses MRv1 (not in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES)

### GraniteMOE MRv2 enable (#45461) -- OPEN
- Next step in MRv2 expansion. Plenty of tests for GraniteMOE.
- Part of MRv2 tracking issue #41286

### DSV4 FlashMLA tile metadata fix (#44069) -- CLOSED
- Fixed MRv2 + DSv4 + MTP crash. FlashMLA tile schedule reset inside captured forward.
- By WoosukKwon. Merged.

### Fuse block table staged writes (#44944) -- OPEN
- V2 model runner performance: single kernel for multi-KV-group block table updates instead of per-group
- Uses VLLM_USE_V2_MODEL_RUNNER=1

### verl MRv2 interaction
- MRv2 two-step execute+sample handled internally by AsyncLLM.generate() -- verl may be safe
- Conservative fallback: VLLM_USE_V2_MODEL_RUNNER=0 still available

---

## BudgetRefiner / SLO-Aware Scheduling

**NO BudgetRefiner PR exists in vLLM upstream.** No SLO-aware scheduling PRs found.

This confirms:
- Our planned BudgetRefiner SLO contribution remains UNIQUE
- RTX 4090 profile data = no other contributor has this
- BudgetRefiner ranks #1 contribution priority (confirmed by this scan)

---

## verl / RL Integration Developments

### Illegal memory access during partial wake_up (#44483) -- OPEN
- Fixes race condition: forward during weights-only wake while scheduler still paused
- verl sleep/wake GPU time-multiplexing affected
- Sleep/Wake = verl HYBRID mode core feature

### LoRA shrink buffer fix (#45715) -- OPEN
- MergedColumnParallelLinearWithLoRA forces all_gather on shrink buffer for TP>1
- Only affects TP>1. RTX 4090 single GPU = NOT affected.

### KV-Offloading CPU cache metric (#45737) -- OPEN
- Exposes `vllm:kv_offload_cpu_cache_usage_perc` metric
- Matches GPU counterpart naming. Useful for monitoring offload health.

### Defer block freeing (#45357) -- MERGED June 15
- Fixes async scheduling + PD KV consumer race: defer freed blocks until in-flight step finishes
- Important for async scheduling stability (verl async rollout)

---

## New Model Support

### MiniMax M3 (#45381) -- MERGED June 15
- NOT in v0.23.0 (released same day, noted as not supported in v0.23.0 release notes)
- Full support: config, processors, MTP, sparse attention, warmup, reasoning/tool parsers
- Heavy model (likely too large for RTX 4090 single GPU)

### MiniMax M3 FP8 sparse GQA (#45744) -- OPEN
- FP8 sparse GQA: Q not quantized, only KV. Triton decode kernel, MSA for SM100+.
- SM89: Triton decode path would be used. Potential RTX 4090 benefit for M3 inference.

---

## Performance / Kernel Developments

### DBO++ TP all-reduce overlap (#44677) -- OPEN
- Overlap TP all-reduce with compute: -7.5% prefill wall on Qwen3-32B TP=4
- RTX 4090 impact: NONE (single GPU, no TP)

### Helion RMSNorm kernel (#36895) -- OPEN
- Helion kernel for rms_norm_per_block_quant (1/N series)
- AMD/ROCm focused. Potential SM89 Triton alternative long-term.

### gridDim.y overflow fix (#45255) -- OPEN
- Fixes CUDA grid dimension overflow for large row counts (row->gridDim.x, cap at 2^31-1)
- Generic kernel bug fix. SM89 relevant if row counts hit 65535 gridDim.y limit.

### FusedMoE scale coercion (#43362) -- OPEN
- NVFP4 per-tensor scales as shape-(1,) coerced to 0-D scalar
- MoE quantization correctness fix.

---

## PD Disaggregation & Spec Decoding

### PD role-aware spec decoding (#45280) -- OPEN
- Phase-2 of PD+SD role optimization. Auto-detects P vs D roles.
- Targeting v0.24/0.25 landing.

### PD skip speculator on P (#45283) -- OPEN
- Skips draft token sampling on Prefill instance in PD flow
- Synthetic tensor for token_ids since dropped anyway.

### CP-scaled scheduler block accounting (#45340) -- OPEN
- NIXL/Mooncake PD with aligned context parallelism
- Block math corrected for CP interleaving. RTX 4090: no CP/PD benefit.

---

## Upcoming: PyTorch 2.13.0

### PyTorch 2.13.0 update (#45731) -- OPEN
- torch -> 2.13.0, torchvision -> 0.28.0, triton -> 3.7.1
- Test channel build. If merged: Inductor behavior may change, impacting batch invariance root cause.
- Monitor this PR for SM89 impact on Triton autotuning.

---

## Quantization Expansion Trend

Multiple PRs expanding quant below SM90:
- #45306: modelopt_mixed on SM80/86 (NVFP4 + FP8 on Ampere via Marlin)
- #45735: ModelOpt mixed precision + NVFP4 runtime formats extension
- #45744: MiniMax M3 FP8 sparse GQA (Triton decode for non-SM100)
- #45738: NVFP4 clamped SwiGLU on FlashInfer-CUTLASS MoE
- #45739: NVFP4 scale buffer zero-init (Blackwell regression fix)

**Trend**: Quantization support actively expanding to sub-SM90 architectures. SM89 gap narrowing incrementally.

---

## RTX 4090 Action Items (Updated)

1. **FP8 KV guard #45038**: Monitor for merge. SM89 validated. Comment with RTX 4090-specific testing data.
2. **Batch invariance #39096**: Still stalled. Push Inductor Fusion Guard via PyTorch upstream.
3. **BudgetRefiner SLO**: No upstream competitor confirmed. Proceed with vLLM PR draft.
4. **PyTorch 2.13.0 #45731**: Monitor. Triton 3.7.1 may change autotuning behavior affecting SM89.
5. **MiniMax M3 FP8 sparse GQA #45744**: Evaluate Triton decode path on SM89 for future M3 inference.
6. **modelopt_mixed #45306**: Watch for SM89 extension possibility (trend is expanding below SM90).
7. **Watermark**: Already in v0.23.0. Set `watermark=0.05` in RTX 4090 config.
8. **INT4 Triton fallback**: Already in v0.23.0. Available for RTX 4090 quantization.
