# 7-Framework Active PR Tracker — RTX 4090 Relevant Developments

> 2026-06-18 | Tracking all open/recent PRs affecting RTX 4090 strategy across 7 frameworks
> ★★★★★★★★ Updated every session — check for merges, new comments, status changes

---

## DeepSpeed

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #8061 | overlap_comm + torch.compile = NaN | OPEN | ★★★★★★★★ MUST overlap_comm=False on single GPU | Track for fix → remove our workaround when merged |
| #8068 | gradient_clipping default 0→1.0 | OPEN | ★★★★★★★★ ALWAYS set gradient_clipping=1.0 for GRPO | Track for fix |
| #8027 | OPD+LoRA integration discussion | OPEN | ★★★★ ~15 LOC gap → 60x optimizer reduction | Tier 2 comment when ready |
| #6816 | CUDA graph support for AutoEP | OPEN | ★★★★★ Future: RouterReplay needed when merged | Track → our Tier 2 RouterReplay PR targets this |
| Muon optimizer | Exact PR TBD (experimental) | EXPERIMENTAL | ★★★★ Muon+LoRA natural combo for single GPU | Source reading in progress |

---

## Megatron-LM

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #5349 | QB routing (replaces aux loss) | OPEN/EVOLVING | ★★★★★ May simplify RTX 4090 MoE pipeline | Track evolution → update RouterReplay design |
| #4168 | RouterReplay 3-mode | MERGED | ★★★★★★★★ Design reference for DeepSpeed equivalent | Source reading complete |
| #5203 | Singleton ProcessGroup crash | OPEN | ★★★ Single GPU crash bug | Track for fix |
| #4885 | Megatron-Lite + LoRA | OPEN | ★★★★ LoRA support in core → bridge alternative | Track for merge |

---

## vLLM

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #39096 | Batch invariance on SM89 | OPEN | ★★★★★★★★ ROOT CAUSE of our P9 Fusion Guard PR | Track comments → GPU validation |
| #45731 | PyTorch 2.13.0 Triton 3.7.1 | OPEN | ★★★★★★★★ May change SM89 autotuning behavior | Critical monitor → update Fusion Guard if needed |
| #43914 | Triton FP8 KV SM89 ALLOWED | MERGED | ★★★ Triton path for SM89 FP8 KV | Track |
| #45720 | v0.23.0 INT4 Triton+HMA | MERGED | ★★★★ INT4 quantization for RTX 4090 | Update deployment pipeline |
| #45038 | BudgetRefiner SLO (our target) | OPEN | ★★★★★★★★ P10 contribution | Track for comment opportunity |
| #44879 | Batch invariance Inductor fix | OPEN | ★★★★★ Alternative to our approach | Tier 1 comment ready |
| #44701 | Triton deterministic | OPEN | ★★★ SGLang-style approach | Tier 1 comment ready |
| BudgetRefiner upstream | Our PR (7 files ~300 LOC) | DRAFT (needs GPU data) | ★★★★★★★★ UNIQUE contribution | Submit when GPU online → collect profile_table.csv |

---

## verl

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #6713 | Megatron LoRA adapter export for vLLM | DRAFT | ★★★★★★★★ EP gather + 3D MoE pack | Critical review: requires_grad on non-leaf → track for fix |
| #6731 | CPPO (Cumulative Prefix-divergence) | OPEN | ★★★★★★★★ bypass_mode REQUIRED → tighter trust region | Track for merge → update RTX 4090 guide |
| #6736 | off_policy staleness metrics | OPEN | ★★★★ Async GRPO staleness tracking | Source reading in progress |
| #6735 | OOM cap memory | OPEN | ★★★ RTX 4090 memory cap | Track |
| #6729 | Memory relief | OPEN | ★★★ Memory optimization | Track |
| verl-omni #169 | Omni-model support | DRAFT | ★★★★ Coordinated with #6713 | Track |
| vllm-omni #4388 | Omni-model vLLM | DRAFT | ★★★★ Coordinated with #6713 | Track |

---

## MindIE

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #8550 | DeepEP-Ascend (fused_deep_moe) | PLANNED | ★★★ Ascend-only → Triton porting opportunity | Track → our Triton dequant_swiglu_quant kernel |
| CANN 9.0 MXFP4 | MX FP4 quantization | RELEASED | ★★★ Ascend-only → RTX 5090 FP4 future | Next-phase contribution window |

---

## rLLM

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #576 | MergedSegment + TokenOps Protocol | PROPOSED | ★★★★ Backend-agnostic step merge | Track |
| Tinker #1 | RTX 4090 GRPO guide | PUBLISHED | ★★★★★★★★ Best single GPU GRPO path | Reference implementation |
| SyncCoordinator | Async trainer architecture | IN MAIN | ★★★ asyncio staleness management | Source reading in progress |

---

## SGLang

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #28354 | NVFP4 (RTX 5090 SM120) | CONFIRMED | ★★★★★★★★ NEXT-PHASE contribution window | Track → FP4/MXFP4 kernel P3 |
| 7 aten overrides | Deterministic batch invariance | IN MAIN | ★★★★★★★★ KERNEL-level → bypasses Inductor | Source reading complete |
| Triton MoE backend | Triton MoE for SM89 | IN MAIN | ★★★★★★★★ Triton recommended for SM89 | Our Triton swiglu kernel fits here |
| murmur_hash32 | Gumbel-max deterministic sampling | IN MAIN | ★★★★★★★★ Gold standard for GRPO | Reference for deterministic sampling |

---

## PyTorch

| PR/Issue | Title | Status | RTX 4090 Impact | Our Action |
|----------|-------|--------|-----------------|------------|
| #185814 | XBLOCK derivation RMSNorm backward | OPEN | ★★★★★ Complementary to our Fusion Guard | Track |
| #187275 | Combo kernel crash fix | OPEN (2026-06-14) | ★★★★★ Confirms same root cause class | Track → strengthens our case |
| Our Fusion Guard PR | 5-line can_fuse_vertical guard | DRAFT (95%/60%) | ★★★★★★★★ P9 contribution | File PyTorch issue first → then submit PR |
| Our Triton swiglu kernel | dequant_swiglu_quant | PROTOTYPE | ★★★★★★★★ P6 contribution (after Fusion Guard) | Validate on GPU → submit to vLLM/SGLang |
| v2.13 Triton 3.7.1 | Autotuning changes | UPCOMING | ★★★★★★★★ May fix SM89 XBLOCK variability | Track → may make Fusion Guard unnecessary |

---

## Summary: Critical Action Items

★★★★★★★★★ IMMEDIATE (next session):
1. File PyTorch issue (SM<90 batch invariance) — pre-step for Fusion Guard PR
2. When GPU online → collect BudgetRefiner profile_table.csv (P10 UNIQUE data)

★★★★★★★★★ SHORT-TERM (this week):
3. DeepSpeed #8061/#8068 — prepare Tier 1 comments if not fixed
4. verl #6713 — track critical review issue (requires_grad on non-leaf)
5. verl #6731 CPPO — track for merge, update RTX 4090 guide

★★★★★★★★★ MEDIUM-TERM (next 2 weeks):
6. Triton dequant_swiglu_quant kernel → GPU validation → vLLM/SGLang PR
7. DeepSpeed RouterReplay equivalent → after CUDA graph support (#6816) merged
8. OPD+LoRA ~15 LOC comment → DeepSpeed #8027

★★★★★★★★★ LONG-TERM (RTX 5090 era):
9. FP4/MXFP4 Triton kernels → SGLang #28354
10. BudgetRefiner vLLM upstream PR → when profile data collected
