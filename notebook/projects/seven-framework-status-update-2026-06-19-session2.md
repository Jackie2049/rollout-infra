# 7 Framework Status Update — 2026-06-19 Session 2

> Key findings from automated framework scan + deep readings

---

## Critical New Findings

### 1. vLLM #46125 NEW — Revert stale encoder cache fix
★★★★★★★★ VERY relevant to RLHF/GRPO! Reverts PR #45093 which was silently resetting prefix/encoder cache after every weight update. The revert author argues this decision should be left to the user. This is directly related to our State Lifecycle Mismatch pattern — weight update boundary must invalidate GPU-resident caches.

### 2. verl #6792 NEW — OPD teacher model OOM
Qwen3-235B teacher model OOM even with 2-machine deployment. RTX 4090 concern: OPD distillation requires teacher+student memory → even Qwen2.5-0.5B teacher + student model may be tight on 24GB.

### 3. Megatron #5400 NEW — 6th Muon blocker!
Routes GDN in_proj to Adam fallback. Created June 18. 2 comments. This further confirms Muon is NOT viable for RTX 4090.

### 4. Megatron #5401 NEW — MoE router z-loss + TE CUDA Graph
Fix for CUDA graph capture failure with MoE router z-loss. Created June 18.

### 5. SGLang #28706 NEW — TF32 n_splits warmup JIT stall
Incomplete warmup causes runtime JIT stalls on MHC models. June 19. Performance issue.

### 6. SGLang #28709 NEW — Double sparsity v2
DSV4-related quant feature. June 19.

### 7. SGLang #28705/#28704/#28700 NEW — AMD DSV4 platform PRs
ROCm/aiter fusion for DSV4 on AMD platform. Multiple PRs for different components.

### 8. vLLM-Ascend #10733 NEW — Layerwise KV cache pool with prefill reuse
★★★★★★★★ Very relevant for Ascend RLHF memory optimization. June 18.

### 9. vLLM-Ascend #10730 NEW — MX quant fusion
RMSNorm dynamic MX quant fusion pass. June 18.

### 10. vLLM-Ascend #10727 NEW — async scheduling perf fix
return router experts async scheduling. June 18.

---

## Status Updates on Tracked Issues

### DeepSpeed — ALL STALLED
- #8072: 0 comments, 0 reviews → COMPLETELY STALLED
- #8073: 0 comments, 0 reviews → COMPLETELY STALLED (per-param dtype fix)
- #8075: 0 review_comments → STALLED (fd leak)
- #8068: 0 review_comments → STALLED (gradient_clipping)
- #8076: 0 comments → NEW independent confirmation of #8072
- #8058: 2 comments, Antlera responding → slow progress
- #8061: ★★★★★★★★ 4 comments now! Maintainers hwchen2017/cx2009 engaged. Reporter confirmed production workload.

### Megatron — ACTIVE
- #5395: ★★★★★★★★ yuchenwang3 addressed ALL 4 ShauryaaSharma review findings June 18! Verified on 8-GPU sm90/H100. SIGNIFICANT progress.
- #5387: APPROVED by shjwudp, CI triggered → progressing
- #5398: Reviewer asked to DROP tests → same pattern as #5398
- #5317: Still OPEN, no resolution for DSV4 rotary NaN

### vLLM — MIXED
- #45819: ★★★★★★★★ ACTIVE! GDN_ATTN batch invariance verified on NVIDIA GPU. CI fixed. Progressing.
- #46085: First-time contributor asking for "ready" label → needs maintainer attention
- #46118: 0 comments → no progress on MTP+grammar FSM fix
- #46125: NEW — Revert encoder cache fix → RLHF/GRPO relevant!

### verl — ACTIVE
- #6794: e2e example WIP → progressing (delta sync)
- #6468: ★★★★★★★★ NEW confirmations: 0.6 GiB/step (qwen3.5-2B), 5.3 GiB/step (qwen2.5-3B), 6.3 GiB/step (Qwen3-35B). Leak scales with model size. STILL NO FIX.
- #6731: Progressing with nan handling, torch.clamp (CPPO)

### SGLang — MANY NEW PRs
- #28680: ★★★★★★★★ CI PASSING on 1-gpu-5090! Progressing (DFlash grammar)
- #28679: 0 comments → CRITICAL silent corruption ignored
- #28676: 0 human reviews → MoE cache clobber STALLED
- #28703: 0 human reviews → DSA LoRA targets NEW

### PyTorch — SLOW
- #187653: CI running, "topic: not user facing" label added. No formal reviews yet.
- #187620: CI awaiting approval. DRAFT.
- #184119: 7 COMMENTED reviews (none APPROVED). Last activity June 2. Slow progress.

### rLLM — STALLED
- #605: 0 comments for 19+ days → COMPLETELY STALLED
- #667: CLOSED per user mandate

---

## Key Cross-Framework Pattern Updates

1. **★★★★★★★★ GRPO Singleton Degeneration**: ALL frameworks (verl, rLLM, TRL) use mean=0, std=1 for singleton groups → REINFORCE degeneration. See cross-framework-grpo-advantage-comparison.md for full analysis.

2. **★★★★★★★★ verl V1 Trainer**: 3 types via @register_trainer + 6 checkpoint engines. RTX 4090 = sync + naive + FSDP + bypass_mode=True. See verl-v1-trainer-architecture-deep-reading.md.

3. **★★★★★★★★ vLLM #46125**: Encoder cache revert → RLHF weight update boundary concern. Same pattern family as State Lifecycle Mismatch.
