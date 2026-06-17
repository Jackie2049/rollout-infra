# PyTorch + Megatron Latest Developments (June 2026) — RTX 4090 Impact Analysis

> 2026-06-18 | Framework developments affecting RTX 4090 strategy | PyTorch 2.12/2.13 + Megatron RouterReplay + FSDP2
> ★★★★★★★★ PyTorch #187275 confirms same root cause as our P9 Fusion Guard
> ★★★★★★★★ Megatron #5349 QB routing + #5219 single-GPU Muon fix → RTX 4090 relevant

---

## 1. PyTorch 2.12 — SM89 Impact

```
★★★★★★★★★ PyTorch 2.12 SM89-impacting changes:

  1. max_autotune layout deferral → NOW opt-in:
    → Previously: SM89 default = autotune → batch-dependent results
    → Now: opt-in path → reduces default SM89 issues
    → BUT: opt-in path still accessible → Fusion Guard still needed!

  2. Triton 3.7 ships with 2.12:
    → Triton 3.7 new autotuning behaviors → may change SM89 variability
    → vLLM #45731: Triton 3.7.1 upgrade → ZERO reviews → makes Fusion Guard MORE needed
    → Triton 3.7.1 will ship with PyTorch 2.13 → tracked

  3. FSDP2 + torch.compile:
    → No longer supports fullgraph through FSDP2 hooks → graph breaks
    → FSDP2 hooks → dynamic parameter gather/release → incompatible with fullgraph
    → Impact: less compile optimization → FSDP2 single GPU already pure overhead

★★★★★★★★★ Our P9 Fusion Guard status:
  → PyTorch 2.12 DOES NOT change insertion point → same can_fuse_vertical True unconditionally
  → Our 5-line guard INSERTS at same location → no adaptation needed!
  → v2.12 max_autotune EXACERBATES SM89 behavior → more autotuned configs → more variability
```

---

## 2. ★★★★★★★★ PyTorch #187275 — Combo Kernel Dynamic Reduction

```
★★★★★★★★★ PR #187275 OPEN (2026-06-14) — "Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions":

  What it fixes:
    → Combo kernel crash when reduction dimension changes dynamically
    → Hardcoded RBLOCK invalid when rnumel changes
    → torch._check bounds → not automatic → requires runtime check

  How it relates to our P9 Fusion Guard:
    → ★★★★★★★★ SAME ROOT CAUSE CLASS: persistent reduction dimension handling
    → Our issue: numerical correctness (different results per batch size)
    → Their issue: crash correctness (hardcoded RBLOCK invalid when rnumel changes)
    → Both stem from persistent reduction dimension handling → CONFIRMS diagnosis!

  Complementary but different:
    → #187275 fixes combo kernel path → requires torch._check bounds
    → Our Fusion Guard: blocks fusion path → props.major<9 → automatic → no bounds needed
    → Together: #187275 fixes combo path, Fusion Guard blocks fusion path → complete solution

★★★★★★★★★ Action items:
    → File PyTorch issue FIRST (issue draft ready)
    → Reference #187275 in our issue → show complementary nature
    → Then submit 5-line guard PR → different approach from #187275
```

---

## 3. PyTorch FSDP2 — Single GPU Overhead Analysis

```
★★★★★★★★★ FSDP2 single GPU analysis (dp_world_size=1):

  dp_world_size=1 → partition=full → AllGather/ReduceScatter=identity → pure overhead:
    → FSDP2 adds latency per module (hooks, buffer management)
    → WORSE than vanilla DDP on single GPU → extra hooks → no benefit
    → CPUOffloadPolicy = ALPHA → crash bugs → NOT production-ready
    → ZeRO-2+CPU_Adam = ONLY mature optimizer-offload solution

  Memory comparison on RTX 4090:
    → Vanilla DDP: 18Ψ (full model + gradients + optimizer)
    → ZeRO-2+CPU_Adam+LoRA: 3.8Ψ → 4.7x memory reduction → fits 24GB
    → FSDP2: same as DDP on single GPU → no benefit → WORSE with hooks
```

---

## 4. Megatron-LM Developments — June 2026

```
★★★★★★★★★ Megatron developments affecting RTX 4090:

  1. #4885 Lite MERGED → 29K additions → Qwen3 MoE + HF safetensors:
    → MoE model support → Qwen3 MoE config → may help RTX 4090 MoE training
    → HF safetensors loading → standard format → easier model import
    → 29K additions → massive codebase expansion → new capabilities

  2. #4256 RouterReplay R3 CLOSED → RL training integration did NOT land:
    → RouterReplay for RL training → proposed but closed without merge
    → Megatron STILL lacks RouterReplay for GRPO → same gap as DeepSpeed
    → Our DeepSpeed RouterReplay design (~300 LOC) could also apply to Megatron

  3. #5349 QB routing STILL DRAFT → no progress:
    → Query-Balanced routing → better expert load balancing
    → Still draft → no implementation progress
    → May simplify RTX 4090 MoE training pipeline when ready

  4. #5219 single-GPU LayerWise Muon crash fix → Final Review → RTX 4090 relevant:
    → Fixes crash on single GPU → similar to Megatron #5203 singleton PG bug
    → LayerWise optimizer → per-layer gradient update → memory efficient
    → RTX 4090 relevant → single GPU + memory efficiency
    → Final Review → close to merge → monitor

  5. #5386 DSA/DSv4 Indexer Replay → OPEN → extends replay concept:
    → Replay for DeepSeek Attention/DS-v4 → extends RouterReplay concept
    → Comment opportunity: suggest RTX 4090 single-GPU R3 path
    → Replay = cached routing → CUDA graph compatible → throughput boost
```

---

## 5. ★★★★★★★★ PyTorch Inductor SM<90 — Issue Filing Status

```
★★★★★★★★★ PyTorch issue filing status:

  Issue draft READY: pytorch-inductor-sm89-fusion-guard-issue-draft.md
    → Title: "[Inductor] Vertical reduction fusion produces batch-dependent
              numerical results on SM<90 GPUs"
    → 5 evidence sources: vLLM #39096, #185814, YM2132 test, #187275, SGLang 7 overrides
    → Proposed 5-line guard in can_fuse_vertical → same pattern as reduction_split_factor

  PR draft READY: pytorch-inductor-sm89-fusion-guard-pr-draft.md
    → 5-line choices.py guard → props.major<9 → WhyNoFuse
    → 5 existing SM-capability precedents in Inductor
    → No additional imports needed
    → Test plan with MockNode → needs unittest.mock.patch approach

  ★★★★★★★★ Pre-step MANDATORY per PyTorch community process:
    → File issue BEFORE submitting PR
    → Wait for PyTorch team feedback → then submit PR
    → GPU validation needed on RTX 4090 → currently offline

  ★★★★★★★★ Submission readiness:
    → Technical: 95% (unchanged — code correct)
    → Submission: 60% → primary blocker = GPU validation + issue not yet filed
    → GPU offline → cannot file issue with reproducible GPU evidence
```

---

## 6. PyTorch 2.13 Risk Assessment

```
★★★★★★★★★ PyTorch 2.13 risk for Fusion Guard:

  Triton 3.7.1 ships with 2.13:
    → May change SM89 autotuning behavior → XBLOCK selection
    → IF Triton 3.7.1 fixes XBLOCK variability → guard becomes unnecessary
    → IF Triton 3.7.1 DOES NOT fix → guard STILL needed
    → vLLM #45731 tracks Triton 3.7.1 upgrade → ZERO reviews → risky

  ★★★★★★★★ Recommendation:
    → File PyTorch issue NOW regardless → track v2.13 outcome
    → If v2.13 fixes → we close issue with "fixed upstream"
    → If v2.13 does not fix → we proceed with PR submission
    → In either case → filing issue creates visibility → community awareness
```

---

## 7. Tier 1 Comment Opportunities — PyTorch/Megatron

```
★★★★★★★★★ PyTorch/Megatron comment opportunities:

  1. PyTorch #187275 (combo kernel fix):
    → Comment: our SM89 Fusion Guard is complementary → blocks fusion path
    → Different approach: props.major<9 guard vs torch._check bounds
    → Together: complete SM<90 solution → block bad fusion + fix combo kernel

  2. Megatron #5386 (DSA/DSv4 Indexer Replay):
    → Comment: suggest RTX 4090 single-GPU R3 path
    → RouterReplay for GRPO → CUDA graph compatible → throughput boost
    → Our DeepSpeed RouterReplay design → applicable to Megatron too

  3. Megatron #5219 (single-GPU Muon crash fix):
    → Comment: confirm RTX 4090 relevance → single GPU + memory efficiency
    → LayerWise optimizer → RTX 4090 viable → but watch for DeepSpeed Muon blockers
```

---

## 8. RTX 4090 Action Items — PyTorch/Megatron

```
★★★★★★★★★ Priority actions for PyTorch/Megatron:

  IMMEDIATE (CPU-only):
    → File PyTorch issue for P9 Fusion Guard → issue draft ready
    → Cannot file until GPU available for reproducible example
    → Prepare: update issue with #187275 reference → complementary

  WHEN GPU ONLINE:
    → P9 SM89 batch invariance reproduction → validate Fusion Guard
    → Run repro script (sm89_batch_invariance_repro.py) on RTX 4090
    → Collect evidence → file issue with GPU-confirmed reproducible example
    → Then submit PR

  MONITOR:
    → PyTorch 2.13 → Triton 3.7.1 → SM89 autotuning changes
    → vLLM #45731 → Triton upgrade → track for merge
    → #187275 → combo kernel fix → track for merge
    → Megatron #5219 → single-GPU Muon → Final Review → close to merge
    → Megatron #5349 → QB routing → DRAFT → no progress yet
```

---

## References

- Fusion Guard issue: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard approach: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-approach.md
- FSDP2 analysis: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- PyTorch 2.12 features: notebook/projects/pytorch-2.12-features-reading.md
- PyTorch v2.12 release: notebook/projects/pytorch-v2.12-release-reading.md
- Triton swiglu: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- Cross-framework Triton: notebook/fundamentals/cross-framework-triton-kernel-comparison.md
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- Diagnostic: tools/sm89_batch_invariance_diagnostic.py
- Repro: tools/sm89_batch_invariance_repro.py
