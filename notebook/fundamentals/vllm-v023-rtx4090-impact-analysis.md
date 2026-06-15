# vLLM v0.23.0 RTX 4090 Upgrade Impact Analysis

> 2026-06-16 | RTX 4090 consulting reference — v0.23.0 release impact
> v0.23.0 released 2026-06-15 — significant RTX 4090 impact (both positive and negative)
> Reference: vllm_v023_upgrade_guide.py, sm89_compatibility_checker.py

---

## 1. v0.23.0 RTX 4090 Positive Changes

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 1.1 INT4 Triton Fallback (#43731)

★★★★★★★★★ Before v0.23.0: INT4 inference on SM89 depended exclusively on Marlin kernel. If a model didn't have Marlin support → INT4 inference FAILED.

After v0.23.0: Triton INT4 fallback kernel added. When Marlin unavailable → automatically falls back to Triton INT4 → INT4 inference ALWAYS works on SM89.

Impact: **More models can use INT4 inference on RTX 4090.** Previously, only models with pre-converted Marlin format worked. Now any GPTQ-Int4 model works.

### 1.2 HMA-by-default (#41847)

★★★★★★★★★ HMA (Hardware-aware Memory Allocation) is now DEFAULT in v0.23.0.

What HMA actually does: per-attention-type KV cache grouping (NOT host-memory offloading as the name suggests). 8 HMA connectors organize KV cache allocation by attention type → better memory utilization.

RTX 4090 impact: Startup OOM prevention for 24GB GPUs. HMA-by-default helps RTX 4090 avoid OOM during initial KV cache allocation by grouping attention types more efficiently.

### 1.3 FP8 Fail-Fast Guards

★★★★★★ v0.23.0 added fail-fast checks for FP8 operations on SM<90:

```python
# Before v0.23.0: FP8 operations could silently fail or produce wrong results on SM89
# After v0.23.0: FP8 operations fail-fast with clear error message
```

Impact: **Better error messages for FP8 on SM89.** Instead of silent crashes or wrong results, users now get immediate, clear error messages. This helps diagnose FP8 issues quickly.

---

## 2. v0.23.0 RTX 4090 Unfixed Issues

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 2.1 compressed-tensors FP8 KV Crash (#44879/#45038)

★★★★★★★★★★★★★★★★★★★★★★★★★★★ STILL UNFIXED in v0.23.0.

Bug: compressed-tensors quantization config overrides kv_cache_dtype to FP8, bypassing FlashInfer's SM90 gate → FlashInfer tries to run FP8 on SM89 → CRASH.

Workaround: Never use FP8 KV cache on SM89. Use INT8 KV instead.

Our contribution opportunity: Comment on #44879/#45038 → explain the 3 FP8 KV path distinction → suggest fail-fast guard in compressed-tensors override → Tier 1 comment draft ready.

### 2.2 SM<90 Batch Invariance (#39096)

★★★★★★★★★★★★★★★★★★★★★★★★★★★ STILL UNFIXED in v0.23.0.

Bug: torch.compile with RMSNorm on SM89 produces batch-dependent results due to Inductor vertical fusion (pow2+mean+rsqrt+mul → one kernel → tl.sum() inline → XBLOCK varies → accumulation order varies).

Workaround options:
1. `enforce_eager=True` → no CUDA graphs → slower but correct
2. `VLLM_USE_V2_MODEL_RUNNER=0` → conservative fallback (may not be necessary)
3. SGLang deterministic inference → alternative serving framework
4. Our proposed fix: Inductor SM<90 Fusion Guard (PyTorch upstream PR)

Our contribution: Inductor Fusion Guard PR draft ready → Tier 2 contribution → needs GPU validation.

### 2.3 Prefix Hash Collision (#44701)

★★★★★ Minor issue, unfixed. Prefix caching hash collisions can cause incorrect KV reuse.

Workaround: Disable prefix caching if seeing unexpected results.

---

## 3. v0.23.0 RTX 4090 MRv2 Impact

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.1 MRv2 Default for Dense Models

v0.23.0 defaults MRv2 (Model Runner V2) for dense models: Qwen3, Llama, Mistral, DeepseekV2, Qwen2Moe.

Key changes:
- `execute_model()` → returns None (no direct output)
- `sample_tokens()` → separate async output
- AsyncLLM.generate() handles two-step internally

### 3.2 verl Impact Assessment

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

CRITICAL CORRECTION from source-level analysis:

**MRv2 is SAFE for verl!** AsyncLLM.generate() internally calls EngineCore.step() which first execute_model() then sample_tokens(). This means verl's call to `engine.generate()` still works — the two-step split is handled inside vLLM's engine, NOT exposed to verl.

But: needs GPU verification to confirm. Conservative fallback: `VLLM_USE_V2_MODEL_RUNNER=0`.

### 3.3 RTX 4090 MRv2 Recommendation

★★★★★★★ Use default MRv2 (don't override). If issues occur, use `VLLM_USE_V2_MODEL_RUNNER=0` as fallback.

---

## 4. Complete v0.23.0 RTX 4090 Checklist

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Must-Do

1. ✅ Use INT8 KV cache (FlashInfer backend) — no FP8 KV on SM89
2. ✅ Use INT4 Marlin/Triton for 7-8B inference — BF16 for 1.7B
3. ✅ Use bypass_mode=True for all GRPO/RL training — no ref model
4. ✅ Use LoRA rank=32 for all training — full model training impossible
5. ✅ Use enforce_eager=True with torch.compile — until batch invariance fixed

### Must-Avoid

1. ✗ Never use FP8 KV cache on SM89 (all 3 paths: Triton FP8 slower, FlashInfer FP8 blocked, compressed-tensors FP8 crashes)
2. ✗ Never use FP8 weight quantization on SM89 (fails or falls back to BF16)
3. ✗ Never use DeepSpeed ZeRO-3 or FSDP2 on single GPU (pure overhead)
4. ✗ Never use Megatron DistributedOptimizer on single GPU (no benefit, potential crash)
5. ✗ Never use full model training on 24GB (18Ψ ≈ 61GB for 1.7B)

### Optional-But-Recommended

1. ★ Use SGLang deterministic inference for GRPO rollout (--enable-deterministic-inference)
2. ★ Use INT4 Triton fallback for models without Marlin format
3. ★ Use coalesce_grad_reduction=True with DeepSpeed ZeRO-2 (even on single GPU)
4. ★ Use HMA-by-default (already default in v0.23.0)
5. ★ Use INT8 KV cache via FlashInfer for all inference on SM89

---

## 5. OSS Contribution Opportunities from v0.23.0

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| # | Issue | Contribution | Priority | Status |
|---|-------|-------------|----------|--------|
| 1 | #44879/#45038 | Comment: 3 FP8 KV path distinction + suggest fail-fast guard | Tier 1 | Draft ready |
| 2 | #39096 | PyTorch PR: Inductor SM<90 Fusion Guard | Tier 2 | PR draft ready |
| 3 | #44701 | Comment: prefix hash collision analysis | Tier 1 | Draft ready |
| 4 | #44594 | Comment: BudgetRefiner complementary to Watermark | Tier 1 | Draft ready |
| 5 | N/A | vLLM PR: BudgetRefiner SLO + RTX 4090 profile data | Tier 2 | PR draft + integration analysis ready |
| 6 | #32268 | vLLM PR: QuantKey refactor (SM89 guard foundation) | Tier 2 | Needs implementation |
| 7 | N/A | vLLM PR: SM120 FP4/MXFP4 kernel (NEXT-PHASE) | Tier 2 | Future window |

★★★★★★★★★ Top 2 contributions:
1. BudgetRefiner SLO → vLLM upstream (most novel + RTX 4090 profile data unique)
2. Inductor SM<90 Fusion Guard → PyTorch upstream (most direct RTX 4090 benefit)
