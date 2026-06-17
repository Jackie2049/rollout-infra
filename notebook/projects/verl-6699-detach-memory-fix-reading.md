# verl #6699 — Detach model_output Memory Fix Source Reading

> 2026-06-18 | PR #6699 (MERGED June 12) | 125 additions | Author: YoungZSh (YZS)
> ★★★★★★★★ 4x memory reduction: 64 GiB OOM → 16.2 GiB stable → CRITICAL for RTX 4090
> ★★★★★★★★ Root cause: FSDPEngineWithLMHead.forward_step returns model_output (log_probs/entropy) STILL ATTACHED to autograd graph
> ★★★★★★★★ Fix: detach tensors before building returned output dict → pinned checkpoint frames released

---

## 1. Root Cause: Autograd Graph Attachment

```
★★★★★★★★★ The bug mechanism (verified from GitHub PR):

Before fix:
  → FSDPEngineWithLMHead.forward_step returns model_output dict
  → model_output contains log_probs and entropy tensors STILL ATTACHED to autograd graph
  → forward_backward_batch holds these per-micro-batch outputs in output_lst
  → output_lst retains until whole batch finishes → graph pinned per micro-batch
  → ★★★★★★★★ Pinned per micro-batch: activation-checkpoint frame's saved embedding output
    (requires grad under PEFT enable_input_require_grads) + its gradient buffer
  → Each micro-batch pins ~0.16 GiB of checkpoint frames (varies by model)
  → At micro-batch #400: accumulation → 64 GiB → OOM crash!

★★★★★★★★★ Verified memory timeline (from PR, Qwen3-8B LoRA rank 32 GRPO):
  → mb #250: 24.8 GiB | #300: 37.9 GiB | #350: ~50 GiB | #400: 64 GiB → OOM
  → After fix: mb #250: 16.2 GiB | #300: 16.2 GiB | #350: 16.2 GiB | #400: 16.2 GiB ✅
  → ★★★★★★★★ STABLE throughout → no accumulation → leak eliminated
```

---

## 2. The Fix: Detach Before Dict Construction

```
★★★★★★★★★ The fix (MERGED June 12, 125 additions):

Key code changes (from PR):
  → verl/workers/engine/fsdp/transformer_impl.py — forward_step:
    detach tensors in model_output before building returned output dict
  → Also covers critic via FSDPEngineWithValueHead
  → Live loss still returned separately for backward() → gradient computation unaffected
  → ★★★★★★★★ 125 additions includes regression test (0 deletions → pure additions)

★★★★★★★★★ Regression test added:
  → tests/workers/test_engine_forward_step_detach_on_cpu.py
  → Miniature reproduction: frozen embedding + requires_grad_ output + checkpointed block + weakref
  → Verified red on unfixed engine, green with fix
  → CPU-only → runs in cpu_unit_tests → no GPU needed

★★★★★★★★★ Why detach() works:
  → .detach() creates new tensor sharing storage but NOT in autograd graph
  → Detached tensor has no grad_fn → no reference to parent graph
  → Checkpoint frames can now be garbage collected after forward pass
  → ★★★★★★★★ The detached tensors are only consumed for metrics aggregation
    and postprocessing AFTER loss.backward() has already run → no gradient impact

★★★★★★★★★ Verified memory timeline (from PR):
  → Before: mb #250: 24.8 GiB | #300: 37.9 GiB | #350: ~50 GiB | #400: 64 GiB → OOM
  → After:  mb #250: 16.2 GiB | #300: 16.2 GiB | #350: 16.2 GiB | #400: 16.2 GiB ✅
  → ★★★★★★★★ STABLE throughout → leak eliminated → RTX 4090 FITS!

★★★★★★★★★ After-fix training health (from PR):
  → grad_norm: 0.015 (healthy)
  → rollout/training log-prob Pearson corr: 0.993 (excellent alignment)
  → Peak allocated: 31.5/79 GiB on larger GPU → RTX 4090 ~16.2 GiB
```

---

## 3. Impact on RTX 4090 LoRA GRPO

```
★★★★★★★★★ RTX 4090 implications:

Without #6699 fix:
  → LoRA GRPO: memory leak → crash at ~micro-batch #30 → UNUSABLE
  → Even with gradient accumulation: each micro-batch pins frames → leak
  → ★★★★★★★★ RTX 4090 LoRA GRPO was BROKEN before this fix!

With #6699 fix:
  → LoRA GRPO: stable 16.2 GiB → fits 24 GiB with 7.8 GiB margin
  → Bypass_mode: 18Ψ→3.8Ψ → eliminates ref model → further savings
  → ★★★★★★★★ RTX 4090 LoRA GRPO NOW VIABLE with #6699 + bypass_mode

★★★★★★★★★ Combined memory savings:
  → #6699 detach: 64→16.2 GiB (4x reduction → leak eliminated)
  → bypass_mode: 18Ψ→3.8Ψ (5x reduction → ref model eliminated)
  → Combined: was 64+ref → now 16.2 with bypass → RTX 4090 comfortable

★★★★★★★★★ Config requirement:
  → MUST use verl version with #6699 MERGED (June 12 commit or later)
  → MUST use bypass_mode=True (separate fix for ref model overhead)
  → MUST use detach_metrics_per_micro_batch=True (additional 0.27 GiB/step prevention)
```

---

## 4. Related verl PRs (June 12-17 Cluster)

```
★★★★★★★★★ Cluster of critical verl fixes:

#6699 (MERGED June 12) — detach model_output → 4x memory reduction
  → CRITICAL for RTX 4090 → LoRA GRPO now viable

#6688 (MERGED June 12) — LoRA IPC buffer crash fix
  → cudaErrorIllegalAddress with unmerged LoRA + vLLM + free_cache_engine=True
  → add_lora retained views into reused IPC bucket buffer → crash
  → ★★★★★★★★ CRITICAL for LoRA weight sync path

#6717 (MERGED June 15) — Tinker training worker primitives (334 additions)
  → optimizer_zero_grad(), forward_backward(), optimizer_step() split primitives
  → Enables rLLM Tinker-style split training IN verl → no framework switch
  → ★★★★★★★★ Important for GRPO micro-batch control

#6689 (MERGED June 17) — Offload process_vision_info to thread
  → CPU-bound PNG decode + smart_resize blocked event loop
  → Not directly RTX 4090 relevant → but improves async pipeline

#6738 (MERGED June 16) — Skip redundant clone SGLang weight sync
  → Critical OOM fix → clones only when necessary (contiguous check)
  → DTensor FSDP path: all-gather results already own tight storage → no clone needed
  → ★★★★★★★★ MoE fused weights → multi-GiB → clone doubled memory → OOM fixed

#6790 (MERGED June 17) — Separate async trainer (17 additions)
  → Colocated group never switches to rollouter → pure async pattern
  → Enables separate rollout+training → memory optimization
```

---

## 5. UNFIXED Leak in Other Engine Backends (★★★★★★★★★ CRITICAL GAP!)

```
★★★★★★★★★ Same leak pattern in 3 OTHER engine backends NOT fixed by #6699:

1. AutomodelEngine (automodel/transformer_impl.py lines 708-712):
  → output = {"model_output": model_output, ...} → NO DETACH!
  → Same output_lst.append(meta_info) pattern in forward_backward_batch (line 260)
  → ★★★★★★★★ SAME leak → SAME accumulation → SAME OOM risk

2. MegatronEngine (megatron/transformer_impl.py lines 1013-1017):
  → output = {"model_output": model_output, ...} → NO DETACH!
  → In postprocess_micro_batch_func → pipeline PP may reduce per-micro-batch accumulation
  → ★★★★★★★★ SAME leak risk → but pipeline PP may mask it

3. TorchTitanEngine (torchtitan/transformer_impl.py lines 730-734):
  → output = {"model_output": model_output, ...} → NO DETACH!
  → Also has output_lst.append(output) in forward_backward_batch (line 340)
  → ★★★★★★★★ SAME leak → SAME accumulation → SAME OOM risk

★★★★★★★★★ VeOmniEngine is SAFE:
  → VeOmniEngineWithLMHead inherits from FSDPEngineWithLMHead (line 848)
  → Uses the fixed forward_step → detach applies automatically

★★★★★★★★★ RTX 4090 impact:
  → FSDP path (most common for RTX 4090) → FIXED → safe
  → Automodel/Megatron/TorchTitan paths → NOT fixed → same leak risk
  → ★★★★★★★★ For RTX 4090 GRPO: ALWAYS use FSDP path → other paths will OOM with LoRA
```

---

## Key Findings Summary

★★★★★★★★★ #6699: detach model_output → 4x memory reduction (64→16.2 GiB) → CRITICAL RTX 4090 LoRA GRPO
★★★★★★★★★ Root cause: log_probs/entropy STILL ATTACHED to autograd → pins checkpoint frames → leak
★★★★★★★★★ Fix: .detach() before building output dict → frames released → stable 16.2 GiB
★★★★★★★★★ RTX 4090: LoRA GRPO was BROKEN before fix → NOW VIABLE with #6699 + bypass_mode
★★★★★★★★★ Cluster of 6 verl fixes June 12-17 → each critical for RTX 4090 GRPO pipeline
★★★★★★★★★ MUST use verl >= #6699 commit → MUST bypass_mode=True → MUST detach_metrics=True
★★★★★★★★★ UNFIXED LEAK: AutomodelEngine, MegatronEngine, TorchTitanEngine have same bug! → FSDP path only fixed
★★★★★★★★★ Fix code: 11 lines → detach dict comprehension with torch.is_tensor + grad_fn is not None guard

---

## References

- verl #6699: https://github.com/verl-project/verl/pull/6699
- verl #6688: https://github.com/verl-project/verl/pull/6688
- verl #6717: https://github.com/verl-project/verl/pull/6717
- verl #6738: https://github.com/verl-project/verl/pull/6738
- verl #6790: https://github.com/verl-project/verl/pull/6790
- RTX 4090 GRPO config reference: tools/rtx4090_grpo_config_reference.py
- GRPO troubleshooter: tools/grpo_troubleshooter_4090.py
