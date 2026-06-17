# RTX 4090 OSS Contribution Opportunity Analysis — 5 Framework Agent Results

> 2026-06-18 | Contribution opportunity analysis from 5 research agents | Where are the unique gaps no one else is filling?
> ★★★★★★★★ Based on latest DeepSpeed v0.19.2, SGLang deterministic, vLLM batch invariance, verl CPPO, PyTorch #187275

---

## P10: BudgetRefiner SLO — vLLM upstream (★★★★★★★★★ UNIQUE, NO competing PR)

```
★★★★★★★★★ BudgetRefiner status check:

  vLLM search result: NO PRs or issues with "BudgetRefiner" found
  → This means NO other contributor is working on this
  → Our RTX 4090 profile data remains UNIQUE
  → NO competing effort → highest priority contribution

★★★★★★★★★ BudgetRefiner integration points (VERIFIED):
  → token_budget: scheduler.py line 407
  → decode-first reorder: before line 430 RUNNING loop (4 lines)
  → dynamic max_seqs: line 629
  → 7 files ~300 LOC total

★★★★★★★★★ What we need to submit:
  → GPU profile data (profile_table.csv on RTX 4090) → ~340 rows
  → BudgetRefiner SLO code (58 lines GPU-generic)
  → vLLM config/arg_utils additions
  → Need: GPU ONLINE for profile data collection
```

---

## P9: Inductor SM<90 Fusion Guard — PyTorch upstream (★★★★★★★★★ NO competing PR)

```
★★★★★★★★★ Fusion Guard status check:

  PyTorch search: NO SM89 Fusion Guard PR exists
  → ZERO results for "SM89 fusion guard", "WhyNoFuse", "choices.py SM89"
  → Our 5-line choices.py guard remains UNsubmitted
  → NO competing effort → strong contribution opportunity

★★★★★★★★★ Related PyTorch developments:
  → #187275 PR OPEN: combo kernel dynamic reduction fix → addresses root cause
  → BUT: #187275 requires torch._check bounds → not automatic
  → Our Fusion Guard: automatic → props.major<9 → no bounds needed
  → Complementary: #187275 fixes combo path, our guard blocks fusion path

★★★★★★★★★ PyTorch 2.12 changes affecting Fusion Guard:
  → max_autotune layout deferral NOW opt-in → reduces default SM89 issues
  → BUT: opt-in path still accessible → Fusion Guard still needed
  → Triton 3.7 shipped with 2.12 → 3.7.1 with 2.13 → new autotuning behaviors
  → vLLM #45731: Triton 3.7.1 upgrade → ZERO reviews → makes Fusion Guard MORE needed

★★★★★★★★★ Pre-step for PR submission:
  → File PyTorch issue FIRST (issue draft ready)
  → Issue title: "Vertical reduction fusion produces batch-dependent numerical results on SM<90 GPUs"
  → Required per PyTorch community process
  → Then submit 5-line choices.py guard PR
```

---

## P6: Triton dequant_swiglu_quant — vLLM/SGLang (★★★★★★★★★ MindIE port, complementary)

```
★★★★★★★★★ Triton swiglu status check:

  Kernel prototype ready (triton_dequant_swiglu_quant_prototype.py)
  → tl.constexpr BLOCK_M=32, BLOCK_N=64 → deterministic → SGLang-compatible
  → MoE W8A8 decode: 6→1 kernels per expert → 48→8 total

★★★★★★★★★ SGLang developments complementing Triton swiglu:
  → #27869 MERGED: Qwen3.5 deterministic batch-invariant logprobs → FLA + MoE top-k fix
  → #28063 OPEN: FA3 chunked-prefill deterministic alignment → 2-line fix
  → #26627 OPEN: Kimi-K2.5 MoE deterministic → atomic_add + dual-stream bypass
  → #28466 OPEN: bf16 MoE-LoRA trtllm two-stream overlap → 68-73% ceiling
  → → SGLang continues advancing MoE deterministic → our Triton swiglu fits perfectly

★★★★★★★★★ vLLM developments:
  → #42120 FP8 MoE+LoRA close to merge → our Triton swiglu complementary
  → #39096 SM<90 batch invariance still unfixed → Triton swiglu helps
  → Need: GPU for correctness + benchmark validation
```

---

## Tier 1 Comment Opportunities — SGLang/vLLM/verl (★★★★★★★★★ 3 NEW opportunities)

```
★★★★★★★★★ New Tier 1 comment opportunities from agent results:

  1. SGLang #28063 (FA3 chunked-prefill deterministic alignment):
     → 2-line fix → critical for SM89 → Triton backend alignment
     → Comment: confirm RTX 4090 relevance → SM89 Triton path
     → Our Fusion Guard would also block this → complementary

  2. SGLang #26627 (Kimi-K2.5 MoE deterministic):
     → atomic_add bypass + dual-stream bypass → ~9% throughput cost
     → Comment: RTX 4090 impact analysis → cost acceptable for GRPO determinism
     → Our Triton constexpr approach achieves same determinism with less cost

  3. verl #6572 (full determinism for vLLM rollout):
     → bitwise-aligned reward curves → directly relevant to GRPO
     → Comment: RTX 4090 SM89 batch invariance root cause → reference #39096
     → Our Fusion Guard as complementary solution

★★★★★★★★★ Existing Tier 1 comments (6 drafts ready):
  → #44879/#45038/#44701/#39096 → vLLM batch invariance comments
```

---

## DeepSpeed Contribution Opportunities (★★★★★★★★★ 3 gaps identified)

```
★★★★★★★★★ DeepSpeed gaps from agent results:

  1. Muon CPU offload for ZeRO-2 (#7939 closed without merge):
     → CRITICAL blocker: Muon+CPU_Adam NOT available on single GPU
     → Opportunity: resurrect and fix #7939 → add CPU offload for Muon
     → ~50 LOC → enable Muon memory advantage on RTX 4090

  2. OPD+LoRA integration gap (~15 LOC):
     → #8027 OPD Draft open → full trainer → NOT minimal LoRA path
     → Opportunity: add LoRA adapter path to OPD trainer
     → ~15 LOC → enable LoRA student in OPD

  3. RouterReplay equivalent (~300 LOC):
     → NO RouterReplay PR or issue in DeepSpeed → completely absent
     → Opportunity: implement RouterReplay for AutoEP TokenChoiceTopKRouter
     → ~300 LOC → enable CUDA graph MoE → 1.8-2.3x throughput
```

---

## rLLM Contribution Opportunities (★★★★★★★★★ checkpoint export gap)

```
★★★★★★★★★ rLLM checkpoint export gap:

  → Tinker checkpoint NOT standard PEFT → no export to vLLM/SGLang
  → NO progress found → no PRs or issues addressing this
  → Opportunity: implement Tinker→HF PEFT conversion tool
  → Would enable: Tinker train → PEFT export → vLLM/SGLang serve
  → Critical for RTX 4090 deployment pipeline
```

---

## Megatron Contribution Opportunities (★★★★★★★★★ R3 training integration)

```
★★★★★★★★★ Megatron RouterReplay R3 gap:

  → #4256 CLOSED (not merged) → training-side R3 integration did NOT land
  → #5386 OPEN → DSA/DSv4 Indexer Replay → extends replay concept
  → Opportunity: comment on #5386 → suggest RTX 4090 single-GPU R3 path
  → Megatron #5219: single-GPU Muon crash fix → Final Review → RTX 4090 relevant
```

---

## Contribution Priority Ranking (Updated)

```
★★★★★★★★★ Updated contribution priority from agent results:

  P10 BudgetRefiner SLO (★★★★★★★★★ UNIQUE, NO competing PR):
    → Need: GPU profile data → highest priority when GPU available
    → vLLM: NO other contributor working on this

  P9 Fusion Guard (★★★★★★★★★ NO competing PR):
    → Need: file PyTorch issue first → then submit PR
    → PyTorch: ZERO SM89-specific PRs exist
    → #187275 complementary but different approach

  P6 Triton swiglu (★★★★★★★★★ MindIE port, complementary):
    → Need: GPU validation → after P9 filed
    → SGLang: advancing MoE deterministic → our kernel fits

  Tier 1: 3 NEW comment opportunities (★★★★★★★★★):
    → SGLang #28063, #26627, verl #6572

  Tier 2 DeepSpeed (★★★★★★★★★ 3 gaps):
    → Muon CPU offload (#7939 resurrection), OPD+LoRA (~15 LOC), RouterReplay (~300 LOC)

  Tier 2 rLLM (★★★★★★★★★ checkpoint export):
    → Tinker→PEFT conversion → critical for deployment pipeline
```

---

## References

- BudgetRefiner SLO: notebook/fundamentals/watermark-budgetrefiner-complementary-synthesis.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard issue: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Triton swiglu: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- Full stack status: notebook/fundamentals/rtx4090-grpo-full-stack-status.md
- RouterReplay design: notebook/projects/deepspeed-router-replay-equivalent-design.md
- PR tracker: notebook/projects/seven-framework-pr-tracker.md
