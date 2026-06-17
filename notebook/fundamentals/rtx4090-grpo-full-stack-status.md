# RTX 4090 GRPO Full Stack Status — 7 Framework Deep Research Summary

> 2026-06-18 | Cross-framework status check | Where does each framework stand for RTX 4090 GRPO training?
 What gaps remain? What's ready?
 What contribution opportunities?
> ★★★★★★★★ Comprehensive synthesis from 263+ project notes across 7 frameworks
> ★★★★★★★★ NEW: vLLM batch_invariant.py SM89 gap — RMSNorm NOT aten override → P9 fills gap
> ★★★★★★★★ NEW: verl #6572 validates VLLM_BATCH_INVARIANT as production mechanism
> ★★★★★★★★ NEW: PyTorch #187435 no_fuse_region per-op fusion barrier → complementary to P9

 Complements PR tracker and decision guides

---

## Framework Status Matrix

```
★★★★★★★★★ RTX 4090 GRPO stack status — 7 frameworks:

| Framework | GRPO Viable? | Gap Status | Key Blocker | Contribution Path |
|-----------|-------------|-------------|-------------|-----------------|
| rLLM Tinker | ★★★★★★★★ #1 BEST | Checkpoint export (NOT standard PEFT) → Tinker→HF→vLLM/SGLang bridge needed | P4 checkpoint-to-PEFT export |
 | verl HYBRID | ★★★★★★★★ #2 | vLLM batch invariance (SM89) → enforce_eager fallback | P9 Fusion Guard + P6 Triton swiglu |
 | DeepSpeed ZeRO-2 | ★★★★ #2.5 | #8061 NaN bug (overlap_comm+compile) → overlap_comm=False mandatory | Tier 2: OPD+LoRA ~15 LOC + RouterReplay ~300 LOC | | Megatron-LM | ★★★★ #3 | TE SM89 not guaranteed → RouterReplay MoE-only | P9 Fusion Guard ( Tier 2: comment on #5349 QB routing | | SGLang | ★★★★★★★★ BEST for inference | No gaps → KERNEL-level built-in | Already best — MoE LoRA+deterministic UNIQUE | | vLLM | ★★★★★★★★ #2 serving | Batch invariance bug SM89 | Fusion Guard not ready | P9 Fusion Guard + P6 Triton swiglu + P10 BudgetRefiner | | MindIE | ★★ Ascend-only | Not RTX 4090 | Lessons portable | Triton constexpr = AscendC deterministic tiling | | PyTorch | ★★★★★★★★ Inductor | SM<90 batch invariance | Fusion Guard PR ready | P9 Fusion Guard issue + PR |
```

---

## Priority Action Items

```
★★★★★★★★★ Top 3 priority actions across ALL 7 frameworks:

1. P10 BudgetRefiner SLO → vLLM upstream (UNIQUE RTX 4090 profile data)
   → GPU needed: collect profile_table.csv on RTX 4090
   → NO other contributor has this data → UNIQUE contribution
   → When GPU available → highest priority experiment

2. P9 Inductor SM<90 Fusion Guard → PyTorch upstream
5-line guard)
   → Issue draft READY → file issue on GitHub BEFORE PR
   → Pre-step mandatory per PyTorch community process
   → When issue filed → submit PR

3. P6 Triton dequant_swiglu_quant → vLLM/SGLang (MindIE port)
   → GPU validation needed → correctness + benchmark
   → Complementary to Fusion Guard → provides GOOD fused path
   → When GPU available → validate kernel on RTX 4090
```

---

## Key Blockers per Framework

```
★★★★★★★★★ Framework-specific blockers preventing GRPO on RTX 4090:

rLLM Tinker:
  → ★★★★★★★★ #605 CRITICAL: GRPO grouping bug → trajectory.uid vs task_ids → group size 1 → GRPO BROKEN!
  → Fix: 1 line for enable=False → few lines for per_step → MUST fix before training
  → Checkpoint NOT standard PEFT → need export bridge → Tinker→HF format
  → PR #576 MergedSegment OPEN → track for merge → enable backend swap

verl HYBRID:
  → vLLM batch invariance SM89 → vLLM batch_invariant.py EXISTS but has RMSNorm gap
  → P9 Fusion Guard fills SM89 gap → makes #6572 work on SM89
  → verl #6572 (full determinism) validates VLLM_BATCH_INVARIANT production use
  → CPPO #6731 + bypass + GRPO = RTX 4090 best trust region

DeepSpeed ZeRO-2:
  → #8061 overlap_comm+compile NaN → MUST overlap_comm=False → zero penalty on single GPU
  → #8068 gradient_clipping default 0 → MUST set gradient_clipping=1.0
  → No inference mechanism → use vLLM/SGLang for inference, DeepSpeed for training only

Megatron-LM:
  → TE SM89 not guaranteed → same Fusion Guard needed
  → RouterReplay MoE-only → not general inference determinism
  → NOT recommended for RTX 4090 inference (ranking #3)

SGLang:
  → NO blockers → BEST for inference → built-in deterministic
  → MoE LoRA + deterministic = UNIQUE capability
  → Recommendation: SGLang for RTX 4090 GRPO rollout

vLLM:
  → Batch invariance SM89 → vLLM batch_invariant.py (984 lines) EXISTS but has RMSNorm gap
  → RMSNorm: Triton kernel defined but NOT registered as aten override → Inductor can fuse
  → On SM89: matmul only CUBLASLt workspace config → NO Triton override
  → BudgetRefiner P10 → unique RTX 4090 profile data → highest contribution priority
  → Triton swiglu P6 → complementary to Fusion Guard
  → P9 + vLLM batch_invariant.py + verl #6572 = complete SM89 determinism
```

---

## GPU Availability Impact

```
★★★★★★★★★ GPU availability determines which experiments can proceed:

GPU ONLINE:
  → P10 BudgetRefiner profile_table.csv collection (RTX 4090 specific)
  → P9 SM89 batch invariance reproduction + Fusion Guard validation
  → P6 Triton dequant_swiglu_quant correctness + benchmark
  → DeepSpeed AutoEP MoE smoke test
  → verl HYBRID GRPO end-to-end test

GPU OFFLINE (current state):
  → CPU-only: framework research, source reading, documentation
  → Prepare GPU experiments → ready when GPU available
  → Create diagnostic tools → validate when GPU returns
  → Track PRs and releases → update PR tracker
```

---

## Contribution Readiness Checklist

```
★★★★★★★★★ Pre-submission readiness for each contribution:

P10 BudgetRefiner:
  → [ ] GPU profile data (need RTX 4090)
  → [x] SLO design document (58 lines GPU-generic)
  → [x] Integration points verified (scheduler.py 407/430/629)
  → [ ] profile_table.csv collected (need GPU)
  → [x] Draft PR body ready

P9 Fusion Guard:
  → [x] Issue draft ready (pytorch-inductor-sm89-fusion-guard-issue-draft.md)
  → [ ] Filed on GitHub (need to file BEFORE PR)
  → [x] PR draft ready (5-line choices.py guard)
  → [x] vLLM SM89 gap analysis confirmed (RMSNorm NOT aten override, matmul only CUBLASLt)
  → [x] verl #6572 validates VLLM_BATCH_INVARIANT as production mechanism
  → [x] PyTorch #187435 no_fuse_region complementary mechanism identified
  → [x] Integration path synthesis (P9 + #187435 + #6572) documented
  → [ ] GPU validation (need RTX 4090 to reproduce bug)
  → [x] Triton 3.7.1 check → vLLM #45731 OPEN → REVIEW_REQUIRED → makes P9 MORE needed

P6 Triton swiglu:
  → [x] Kernel prototype ready (triton_dequant_swiglu_quant_prototype.py)
  → [x] Design document ready (triton-dequant-swiglu-quant-sm89-design.md)
  → [ ] GPU correctness test (need RTX 4090)
  → [ ] GPU benchmark (need RTX 4090)
  → [x] MoE path comparison documented

Tier 2 DeepSpeed:
  → [x] OPD+LoRA gap analysis (~15 LOC)
  → [ ] RouterReplay prototype (~300 LOC, after CUDA graph support)
  → [x] freeze_moe_router = 0 LOC immediate solution documented
```

---

## Next Steps (GPU Offline)

```
★★★★★★★★★ While GPU offline, priority actions:

1. File PyTorch issue (P9 Fusion Guard pre-step)
   → Create GitHub issue using draft from pytorch-inductor-sm89-fusion-guard-issue-draft.md
   → This is MANDATORY before submitting PR to PyTorch

2. Update DeepSpeed checkout to v0.16.4 for controlled comparison
   → Current checkout is v0.19.2 (development branch)
   → v0.16.4 = latest stable release → controlled testing environment
   → Compare: MoE changes, ZeRO changes, Muon changes

3. Create SGLang + verl integration test plan
   → SGLang integration (#6117) → check stability
   → Tinker + SGLang → simplest deterministic path
   → Need: SGLang checkout + verl checkout integration testing

4. Prepare BudgetRefiner profile collection script improvements
   → profile_vllm_budget.py → add RTX 4090-specific profiling modes
   → Pre-compute expected profile_table.csv rows → verify estimates match reality

5. Track 7 framework PRs continuously
   → Update seven-framework-pr-tracker.md as PRs evolve
   → Weekly scan of all 7 frameworks' PR lists
```

---

## References

- Full stack decision guide: notebook/fundamentals/rtx4090-deterministic-inference-decision-guide.md
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- Triton kernel comparison: notebook/fundamentals/cross-framework-triton-kernel-comparison.md
- PR tracker: notebook/projects/seven-framework-pr-tracker.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- DeepSpeed ZeRO: notebook/fundamentals/single-gpu-ddp-vs-zero-architecture-comparison.md
- FSDP2 analysis: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- RTX 4090 decision: notebook/fundamentals/rtx4090-grpo-training-decision-flowchart.md
- vLLM batch_invariant SM89 gap: notebook/projects/vllm-batch-invariant-source-reading.md
- verl #6572 full determinism: notebook/projects/verl-6572-full-determinism-source-reading.md
- P9 integration path: notebook/projects/p9-fusion-guard-integration-path-synthesis.md
