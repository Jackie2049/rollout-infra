# Cross-Framework Muon Optimizer Status Synthesis — RTX 4090 Consultant Guide

> 2026-06-18 | Comprehensive Muon optimizer status across DeepSpeed, Megatron, verl, rLLM
> ★★★★★★★★ RTX 4090 Muon: CURRENTLY NOT VIABLE → 3 blockers remain
> ★★★★★★★★ AdamW + CPU_Adam remains ONLY safe optimizer for RTX 4090 GRPO
> ★★★★★★★★ Tracking 6 open issues/prs that must resolve for Muon to work on single GPU

---

## Executive Summary: Muon NOT Ready for RTX 4090

```
★★★★★★★★★ Muon optimizer status on RTX 4090: NOT VIABLE

3 BLOCKERS:
  1. Muon CPU offload BLOCKED (#7939 closed without merge) → can't fit 24GB
  2. Global gradient clipping DESTROYS Muon (#7776/#8068/#5394) → stalls Newton-Schulz
  3. Single-GPU Muon CRASHES (#5219 still OPEN) → dp_cp_params_list=None → TypeError

★★★★★★★★★ Current RTX 4090 optimizer recommendation:
  → AdamW with CPU_Adam → ZeRO-2 + param_offload + optimizer_offload
  → ★★★★★★★★ ONLY safe path → verified → stable → no blockers

★★★★★★★★★ If ALL 3 blockers resolve:
  → Muon + LoRA natural combo → rotation-based → no need for momentum
  → But timeline: weeks-to-months → no rush to adopt Muon now
```

---

## DeepSpeed Muon Blockers

```
★★★★★★★★★ 3 DeepSpeed Muon blockers (all OPEN/CLOSED-without-merge):

1. #7939 — Muon CPU offload CLOSED WITHOUT MERGE:
  → Author: Brybry14 (community)
  → Tried to add Muon to ZeRO-2 CPU optimizer schedule
  → Reviewer (tjruwase, DeepSpeed maintainer): closed without merge
  → Reason: Muon's orthogonality step needs ALL parameters simultaneously
    → CPU offload + ZeRO-2 shard = can't compute Newton-Schulz on partial params
  → ★★★★★★★★ BLOCKER: Muon+ZeRO-2+CPU_offload = architecturally impossible (for now)
  → Alternative: CPU_Adam remains only viable CPU optimizer for ZeRO-2

2. #7776 — Muon gradient clipping WRONG:
  → orthogonalization-before-clipping = mathematically WRONG
  → Newton-Schulz on near-zero vectors → degenerates → stalls
  → ★★★★★★★★ BLOCKER: DeepSpeed default gradient_clipping 0→1.0 → ALWAYS set 1.0 for GRPO
  → But for Muon: MUST gradient_clipping=0 → or skip clipping for Muon groups

3. #8068 — gradient_clipping default 0→1.0:
  → Default changed in v0.19.x → silently applies global clipping to ALL optimizers
  → ★★★★★★★★ BLOCKER for Muon: global clipping → same as #7776 → stalls Newton-Schulz
  → MUST: gradient_clipping=1.0 for AdamW groups, 0 for Muon groups → per-group config needed

★★★★★★★★★ Additional DeepSpeed Muon issues:
  → #7878: Muon MUST NOT use reduce_scatter → but ZeRO-2 uses reduce_scatter → conflict!
  → #7919: Muon lr overrides → MERGED → lr scheduling works
  → #7953: Muon momentum scheduling → MERGED → muon_lr/adam_lr overrides
  → #8047: Muon lr override tests → MERGED → test coverage added
```

---

## Megatron-LM Muon Blockers

```
★★★★★★★★★ 3 Megatron Muon blockers (2 OPEN, 1 DRAFT):

1. #5219 — Single-GPU Muon CRASH (Final Review, OPEN):
  → dp_cp_params_list = None on single GPU → zip(bucket_params_list, None) → TypeError
  → Fix: None guard + else branch → identical to expt_dp_params_list pattern
  → ★★★★★★★★ IRONY: expt_dp_params_list ALREADY had correct guard → dp_cp path was missing
  → Status: Final Review → no technical objections → likely days-weeks to merge
  → ★★★★★★★★ +14/-7 → simple fix → but NVIDIA CI gate blocks merge

2. #5394/#5395 — ChainedOptimizer global clipping STALLS Muon:
  → Same pattern as DeepSpeed #8068/#7776 → global grad clipping across sub-optimizers
  → ChainedOptimizer forces ALL sub-optimizers to share OptimizerConfig
  → Config constraint: can't set different clipping per sub-optimizer
  → ★★★★★★★★ Fix PR #5395: skip_grad_norm_clip attribute (+15/-1)
  → Bypasses global clipping for specific sub-optimizers → Muon gets skip=True

3. #5391 — Compact LayerWise DDP (DRAFT, OPEN):
  → Removes dp_size padding on dp=1 → per-buffer use_distributed_optimizer
  → ★★★★★★★★ NOT a blocker but HIGHLY beneficial → Muon buffers compact (all-reduce)
  → +218/-58 → significant change → DRAFT status → needs more review

★★★★★★★★★ Megatron Muon dependency chain:
  → #5219 (crash fix) MUST merge FIRST → enables initialization
  → #5395 (clipping skip) MUST merge SECOND → enables correct training
  → #5391 (compact DDP) SHOULD merge → memory efficiency
  → ★★★★★★★★ All 3 needed for production RTX 4090 Muon on Megatron
```

---

## verl Muon Status

```
★★★★★★★★★ verl Muon: NOT natively supported but Tinker path enables it

verl doesn't implement Muon directly → uses external optimizer configs
  → ZeRO-2 + CPU_Adam → current RTX 4090 path
  → ★★★★★★★★ #6717 Tinker training worker → split primitives → enables custom optimizer stepping
  → #6765 per-step optimizer overrides → OptimStepParams → enables Muon LR scheduling!

★★★★★★★★★ Potential verl Muon path (when blockers resolve):
  1. Use verl HYBRID mode for rollout (bypass_mode=True)
  2. Use DeepSpeed ZeRO-2 backend for training
  3. Configure Muon via DeepSpeed config (but need #7939 or alternative)
  4. Use Tinker worker to drive optimizer stepping with per-group params
  5. ★★★★★★★★ This is FUTURE path → needs all DeepSpeed blockers to resolve first

★★★★★★★★★ verl #6765 OptimStepParams (NEW):
  → OptimStepParams = lightweight runtime payload for optimizer param-group values
  → lr, betas, eps, weight_decay per step
  → Scoped to TinkerTrainingWorker.optimizer_step() only
  → ★★★★★★★★ Enables Muon-style LR scheduling without modifying engine API
  → Example: worker.optimizer_step(optim_step_params={"lr": 5e-4})
```

---

## rLLM Muon Status

```
★★★★★★★★★ rLLM Muon: NO support → not even on roadmap

rLLM uses standard AdamW → no Muon implementation
  → Focus is on Tinker-style split training → not optimizer innovation
  → ★★★★★★★★ No Muon-related issues or PRs → no plans visible

★★★★★★★★★ If Muon becomes viable in DeepSpeed:
  → rLLM could leverage DeepSpeed backend for Muon training
  → But #605 GRPO grouping bug makes rLLM unusable for GRPO regardless
  → ★★★★★★★★ Muon won't help rLLM until #605 is fixed
```

---

## Cross-Framework Pattern: Gradient Clipping Bug

```
★★★★★★★★★ SAME bug found independently by 3 groups:

DeepSpeed #8068: default gradient_clipping 0→1.0 → global clipping → stalls Muon
  → Author: community
  → Root cause: gradient_clipping applies to ALL optimizer groups uniformly

DeepSpeed #7776: orthogonalization-before-clipping → mathematically wrong
  → Author: community
  → Root cause: Newton-Schulz on near-zero vectors → degenerates

Megatron #5394: ChainedOptimizer global clipping → same pattern
  → Author: factnn (community)
  → Root cause: config constraint forces shared OptimizerConfig across sub-optimizers

★★★★★★★★★ Universal insight:
  → Scale-invariant optimizers (Muon, Adam with large eps) MUST NOT be globally clipped
  → Global clipping → positive feedback loop → near-zero vectors → stall
  → ★★★★★★★★ Fix pattern: per-optimizer clipping control
    → DeepSpeed: per-group gradient_clipping → needs #7776 resolution
    → Megatron: skip_grad_norm_clip attribute → #5395 (+15/-1)
    → verl: OptimStepParams → #6765 (per-step override)

★★★★★★★★★ This is a Tier 1 OSS contribution (C8):
  → Cross-framework evidence → 3 independent discoveries → same root cause
  → Universal insight → any scale-invariant optimizer needs clipping exemption
  → ★★★★★★★★ Comment draft ready for Megatron #5394
```

---

## RTX 4090 Muon Viability Timeline

```
★★★★★★★★★ When Muon might become viable on RTX 4090:

Short term (1-2 weeks):
  → #5219 likely merges → Megatron single-GPU crash fixed
  → #5395 likely merges → Megatron clipping skip added
  → ★★★★★★★★ But: DeepSpeed #7939 still blocked → Muon+CPU_offload impossible
  → ★★★★★★★★ Result: Megatron Muon works on single GPU → but DeepSpeed doesn't

Medium term (1-2 months):
  → #5391 might merge → Megatron compact DDP → better memory efficiency
  → Someone might resurrect #7939 with different approach → watch for new PR
  → ★★★★★★★★ ZenFlow #8058 might provide alternative → native CPU optimizer → not Muon-specific

Long term (3-6 months):
  → verl Tinker + per-step overrides → full Muon scheduling control
  → DeepSpeed might add per-group clipping → enables Muon+AdamW hybrid
  → ★★★★★★★★ RTX 4090 Muon GRPO becomes viable → but AdamW remains safer

★★★★★★★★★ Recommendation: DON'T WAIT for Muon → use AdamW + CPU_Adam NOW
  → AdamW is proven, stable, no blockers
  → Muon will be better WHEN it works → but timeline uncertain
  → ★★★★★★★★ Start with AdamW → switch to Muon when all blockers resolve
```

---

## Key Findings Summary

★★★★★★★★★ Muon RTX 4090 status: NOT VIABLE → 3 blockers → AdamW remains ONLY option
★★★★★★★★★ DeepSpeed: #7939 CPU offload BLOCKED → #7776/#8068 clipping bug → #7878 reduce_scatter
★★★★★★★★★ Megatron: #5219 crash fix Final Review → #5395 skip_grad_norm_clip → #5391 compact DDP
★★★★★★★★★ verl: #6765 OptimStepParams → enables Muon LR scheduling → but needs DeepSpeed backend
★★★★★★★★★ rLLM: NO Muon support → not on roadmap → #605 blocks GRPO anyway
★★★★★★★★★ Cross-framework pattern: 3 independent discoveries of global clipping bug → same root cause
★★★★★★★★★ Timeline: 1-2 weeks (Megatron fix) → 1-2 months (DeepSpeed alternative) → 3-6 months (full viable)
★★★★★★★★★ Recommendation: use AdamW + CPU_Adam NOW → switch to Muon when blockers resolve

---

## References

- DeepSpeed #7939: https://github.com/microsoft/DeepSpeed/issues/7939 (CPU offload BLOCKED)
- DeepSpeed #7776: https://github.com/microsoft/DeepSpeed/issues/7776 (clipping bug)
- DeepSpeed #8068: https://github.com/microsoft/DeepSpeed/issues/8068 (default clipping)
- DeepSpeed #8058: https://github.com/microsoft/DeepSpeed/pull/8058 (ZenFlow native CPU optimizer)
- Megatron #5219: https://github.com/NVIDIA/Megatron-LM/pull/5219 (crash fix)
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (clipping stalls)
- Megatron #5395: https://github.com/NVIDIA/Megatron-LM/pull/5395 (skip_grad_norm_clip)
- Megatron #5391: https://github.com/NVIDIA/Megatron-LM/pull/5391 (compact DDP)
- verl #6717: https://github.com/verl-project/verl/pull/6717 (Tinker primitives)
- verl #6765: https://github.com/verl-project/verl/pull/6765 (OptimStepParams)
- rLLM #605: https://github.com/rllm-org/rllm/issues/605 (GRPO grouping bug)
- Cross-framework clipping: notebook/fundamentals/cross-framework-muon-gradient-clipping-bug.md
- DeepSpeed Muon source: notebook/projects/deepspeed-muon-optimizer-source-reading.md
