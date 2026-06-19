# 7-Framework Developments Tracker — June 19, 2026

**Generated: 2026-06-19 | Cross-framework comprehensive check**

---

## 1. DeepSpeed

| Issue | Status | Key Update |
|-------|--------|-----------|
| #8072/#8076 | OPEN, 0 reviews | ZeRO-3+PEFT regression STILL unfixed. Independent confirmation by whyseu (4xH100, LLaMA Factory) |
| #8073 | OPEN, 0 reviews | 2-line dtype fix, stalled with zero reviews |
| #8075 | OPEN, 0 reviews | fd leak fix (+1/-1), external contributor, requested reviewer: tjruwase |
| #8068 | OPEN, 0 reviews | gradient_clipping default 0→1.0, stalled |
| #8066 | MERGED June 16 | Root cause of #8072 regression — per-policy dtype |

★★★★★★★★★ DeepSpeed pattern: ALL community PRs stalled with 0 reviews. v0.19.2 regression cluster unresolved.

---

## 2. Megatron-LM

| Issue | Status | Key Update |
|-------|--------|-----------|
| #5394 | OPEN, 4 comments | ★★★★★★★★ BREAKTHROUGH: AdamW ALSO stalls under global clipping! Optimizer-agnostic bug confirmed. Volunteer stepping up to fix |
| #5395 | OPEN, 2 comments | skip_grad_norm_clip (+15/-1). yuchenwang3 posted thorough review response. Progressing |
| #5387 | OPEN, CI triggered | MFSDPv2 APPROVED. `/ok to test` triggered by wujingyue. CI running |
| #5398 | OPEN, 3 comments | NVIDIA reviewer asolergi-nv asked to "drop the tests" — reviewer engagement! |
| #5396 | OPEN, 1 comment | GDN L2-norm fold, bot-only comment, no reviews yet |
| #5400 | OPEN (draft) | 6th Muon blocker, auto-converted to draft per NVIDIA policy |
| #5401 | OPEN | `/ok to test` triggered by Victarry |

★★★★★★★★★ #5394 breakthrough broadens bug scope: not just Muon, ALL optimizers can stall under global clipping.

---

## 3. vLLM

| Issue | Status | Key Update |
|-------|--------|-----------|
| #45972 | MERGED June 18 | 2nd DSV4 revert — eager_break garbage output |
| #45979 | CLOSED (no merge) | FALSE ALARM — sparse cache VINDICATED, cudagraph was culprit |
| #45656 | ★★★★★★★★ MERGED June 18 | MoE is_sym guard regression fix NOW MERGED! |
| #45819 | OPEN, 7 comments | GDN progressing. yuvalluria posted local testing results. Blocked by MoE limitation |
| #46007 | OPEN, CHANGES_REQUESTED | Orthrus spec decode. benchislett asked for DFlash benchmark |
| #46085 | OPEN, 2 comments | aot_eager backend. First-time contributor requesting ready label |
| #46088 | ★★★★★★★★ NEW | MTP + kv-cache-dtype auto = garbage under batching → GRPO rollout concern! |
| #46105 | NEW | DFlash Bring-Up Tracker — next-gen spec decode paradigm |
| #46106 | NEW | Sync KV num_gpu_blocks across DP replicas |
| #46107 | NEW | RFC: Heterogeneous TP→DP (MoRIIO) disaggregated serving |
| #46083 | NEW | MoE coredump during FlashInfer autotune on B200 |

★★★★★★★★★ #45656 MERGED = MoE quantization regression FIXED! #46088 is new GRPO concern. DFlash emerging as spec decode future.

---

## 4. verl

| Issue | Status | Key Update |
|-------|--------|-----------|
| #6512 | MERGED June 18 | Per-unit LoRA summon → 10x memory reduction |
| #6791 | MERGED June 18 | DSV4/GLM5/KimiK2.5 via Megatron Lite |
| #6790 | MERGED June 18 | Separate async trainer (multi-GPU only) |
| #6794 | OPEN, 2 comments | Delta weight sync. CLA not signed yet. e2e example WIP |
| #6468 | OPEN, 3 comments | ★★★★★★★★ FSDP2 leak scaling confirmed: 0.6-5.3 GiB/step by model size |
| #6782 | OPEN, 1 comment | LoRA rank=64 breaks EOS, image evidence posted |
| #6795 | NEW | Remove invalid single_turn_response_length override |
| #6796 | NEW | Align aggregated metrics logging with current step |
| #6786 | NEW | dynamic-cp batch split bug |

★★★★★★★★★ #6468 leak confirmed multi-user. #6794 e2e progressing. #6782 still unfixed.

---

## 5. SGLang

| Issue | Status | Key Update |
|-------|--------|-----------|
| #28676 | ★★★★★★★★ NEW, OPEN | MXFP8 MoE shuffle cache CLOBBERED → 10th DSV4 failure! 64x accuracy blowup |
| #28618/#28620 | OPEN | SM89 DSV4-Flash-FP8. CI blocked on run-ci label, not test failures |
| #28679 | ★★★★★★★★ NEW | GDN intermittent decode degeneracy — worsens over uptime, clears on restart |
| #28653/28654 | NEW | HiCache ENOSPC at scale → hash-prefix subdirs fix |
| #28669 | NEW | Disaggregated request preprocessing RFC |
| #28591 | OPEN, 2 comments | DSV4 MTP revert. Fix PR #28612 linked |

★★★★★★★★★ #28676 = 10th DSV4 failure (physical clobber, worse than stale ref). #28679 GDN degeneracy = new worrying pattern.

---

## 6. rLLM

| Issue | Status | Key Update |
|-------|--------|-----------|
| #605 | ★★★★★★★★ OPEN 18+ days | CRITICAL GRPO grouping bug, STILL ZERO comments |
| #663 | MERGED June 17 | Step.output fix (rewards were all 0.0) |
| #665 | MERGED June 17 | Fireworks live model catalog |
| #666 | MERGED June 18 | Fireworks SWE-RL |

★★★★★★★★★ #605: Zero engagement after 18+ days. P9 UNIQUE contribution unchanged.

---

## 7. PyTorch

| Issue | Status | Key Update |
|-------|--------|-----------|
| #187620 | OPEN (DRAFT) | PartialOffloadPolicy — CI awaiting approval |
| #187636 | ★★★★★★★★ CI triggered June 18 | autotune_at_compile_time flips default → complements P9 |
| #187653 | CI running | NanDetectMode for NaN/Inf detection |
| #184119 | OPEN, 10 comments | SM89 fp8 guard — broadened per feedback, still progressing |

★★★★★★★★★ #187636 CI triggered → autotune default change complements P9 on RTX 4090.

---

## 8. vLLM-Ascend

| Issue | Status | Key Update |
|-------|--------|-----------|
| #10684 | OPEN, 1 comment | DSA Hadamard — reviewer pinged SOMEONEUNSEEN |
| #10730 | ★★★★★★★★ NEW | MX quant fusion — avg 7.541→3.525us for AddRMSNorm+DynamicMxQuant |
| #10733 | NEW | Layerwise KV pool + prefill layer reuse (builds on #10077) |

---

## Key Takeaways for RTX 4090

1. ★★★★★★★★ **Megatron #5394**: AdamW clipping stalls = optimizer-agnostic bug → affects ALL optimizers, not just Muon
2. ★★★★★★★★ **vLLM #45656**: MERGED → MoE quantization regression FIXED
3. ★★★★★★★★ **SGLang #28676**: 10th DSV4 failure → MXFP8 cache CLOBBERED
4. ★★★★★★★★ **vLLM #46088**: NEW MTP batching bug → potential GRPO concern
5. ★★★★★★★★ **SGLang #28679**: GDN intermittent degeneracy → non-deterministic over time
6. ★★★★★★★★ **verl #6468**: FSDP2 leak scaling confirmed 0.6-5.3 GiB/step
7. ★★★★★★★★ **PyTorch #187636**: CI triggered → autotune complement to P9
8. ★★★★★★★★ **DFlash ecosystem emerging**: vLLM #46105 tracker, #46104 SWA+DFlash

---

*End of developments tracker. Generated 2026-06-19.*
