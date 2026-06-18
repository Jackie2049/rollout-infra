# 7-Framework Latest Developments — June 18 Update

> 2026-06-18 | Comprehensive status update across all 7 frameworks
> ★★★★★★★★ Source: background research agent (88 tool calls, 9 min)
> ★★★★★★★★ 9 cross-framework critical alerts for RTX 4090 GRPO

---

## Cross-Framework Critical Alerts (RTX 4090 GRPO)

| # | Framework | Issue | Impact | Status |
|---|-----------|-------|--------|--------|
| 1 | DeepSpeed | v0.19.2 ZeRO-3+PEFT regression #8072 | LoRA training BROKEN on ZeRO-3 | OPEN 0 reviews STALLED |
| 2 | rLLM | #605 GRPO grouping bug | GRPO completely non-functional | OPEN 18+ days ZERO comments |
| 3 | rLLM | #663 Step.output was None | ALL prior rewards = 0.0 | MERGED June 17 |
| 4 | verl | #6782 LoRA rank=64 breaks EOS | MUST rank=32/alpha=64 | OPEN with image evidence |
| 5 | verl | #6468 FSDP2 CPU memory leak | 6.3 GiB/step on Qwen3-35B | OPEN multi-user confirmed |
| 6 | Megatron | #5394 Muon clipping stalls | Cross-framework pattern w/ DeepSpeed | OPEN 0 reviews |
| 7 | vLLM | #45972 REVERT DSV4 cudagraph | Garbage output from cudagraph+DSV4 | MERGED June 18 |
| 8 | SGLang | #28582 RCE CVSS 9.8 | Unauthenticated LoRA endpoint | OPEN 0 maintainer response |
| 9 | Cross-framework | DSV4 systematic instability | 4 failures in 4 days → enforce_eager=True MANDATORY | vLLM #45972 + SGLang #28591/#28575/#28569 |

---

## 1. DeepSpeed

**v0.19.2 Patch Release (June 16)** — includes #8066 (per-policy dtype — CAUSED #8072 regression!)

| Issue | Status | Change | RTX 4090 Impact |
|-------|--------|--------|----------------|
| #8072 ZeRO-3+PEFT LoRA regression | OPEN, 0 comments, STALLED | None since June 16 | DANGEROUS for ZeRO-3+LoRA |
| #8073 fix PR for #8072 | OPEN, 0 reviews, STALLED | None | 2-line fix, stalled |
| #8068 gradient_clipping default | OPEN, 0 reviews, STALLED | None | MUST set clip_grad=1.0 |
| #8061 overlap_comm NaN | OPEN, NEW activity | hwchen2017/cx2009/cjxjxjx confirmed production, multi-stream root cause | MUST overlap_comm=False on single GPU |
| #8058 ZenFlow | OPEN, progressing | Antlera responding to delock reviews | CPU optimizer progressing |
| #8064 AutoEP+AutoTP folding | OPEN, updated June 18 | MoE+dense share same rank set | TP=1+EP=1 = identity on RTX 4090 |
| #8074 type hints for API | NEW June 17 | Low impact | No RTX 4090 relevance |

★★★★★★★★★ #8061 NEW detailed root cause analysis:
  → `reduction_stream` reads IPG bucket before all producer streams complete
  → `torch.compile=False` or `overlap_comm=False` both avoid NaN
  → Confirmed production workload → MUST overlap_comm=False on single GPU

---

## 2. Megatron-LM

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #5387 MFSDPv2 | OPEN DRAFT, CI triggered | wujingyue ran /ok June 17 | Future multi-GPU path |
| #5395 skip_grad_norm_clip | OPEN, 0 reviews | Requires CI validation June 17-18 | CRITICAL for Muon |
| #5394 ChainedOptimizer Muon clipping | NEW ISSUE June 17 | yuchenwang3 confirmed optimizer-agnostic scope | ★★★★★★★★ Cross-framework Muon pattern! |
| #5219 Muon crash | OPEN, progressing | guihong-nv reviewed June 11 | RTX 4090 Muon blocker |
| #5391 compact LayerWise DDP | OPEN June 17 | Experimental per-buffer DDP | RTX 4090 memory efficiency |
| #5396 GDN L2-norm fold | NEW June 17 | q/k L2-norm into gated_delta_rule kernel | Qwen3.5-35B training |
| #5389 GDN THD all-to-all | MERGED June 17 | Fused GDN THD restored on dev | ✓ |
| #5392 GDN A2A refactor | OPEN June 17 | Helper flow refactor for #5389 | ✓ |
| #5393 skip permute valid_tokens=0 | NEW June 17 | Skip kernel when no tokens | ✓ |
| #5397 RADIO vision encoder | NEW June 18 | MIMO example vision encoder | Not RTX 4090 relevant |
| #5384 DSA/DSv4 Indexer Replay | OPEN June 16 | RL training stability feature request | ★★★★★★★★ Directly relevant to verl rollout! |

★★★★★★★★★ #5394 NEW CRITICAL: yuchenwang3 confirmed optimizer-agnostic Muon clipping scope
  → Global grad-norm clipping → clip coefficient c = clip_grad/grad_norm → tiny when global norm large
  → Newton–Schulz orthogonalization degenerates below F.normalize(eps=1e-7) floor → near-zero non-orthogonal update
  → ★★★★★★★★ SAME pattern as DeepSpeed #8068/#7776 → cross-framework confirmation!

---

## 3. vLLM

**v0.23.0 released June 15** — 408 commits, 200 contributors

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #45972 REVERT DSV4 cudagraph | MERGED June 18 | REVERTED #45309 → garbage output! | ★★★★★★★★ cudagraph + DSV4 = correctness regression |
| #45964 MLA DCP query replication | OPEN | 2-5% TPOT improvement | RTX 4090 NOT affected (DCP=1) |
| #45683 MoE deterministic combine | OPEN | No activity since June 15 | CRITICAL for GRPO MoE |
| #45819 GDN batch invariance | OPEN, progressing | yuvalluria fixing CI, GDN_ATTN added | Relevant for deterministic |
| #45731 PyTorch 2.13 | OPEN DRAFT | No new comments | ★★★★★★★★ BLOCKED by #187484 |
| #45966 DCP A2A workspace | OPEN June 17 | Pre-reserve packed workspace | DCP infrastructure |
| #45976 w4a8 MoE Oracle OOP | OPEN June 18 | Refactor | ✓ |
| #45968 NVFP4 MoE/DSV4 test | OPEN June 17 | Real DeepSeek-V4-Flash weights | ✓ |

★★★★★★★★★ #45972 CRITICAL: REVERTED DSV4 cudagraph optimization
  → #45309 cudagraph + DeepSeek-V4 → GARBAGE OUTPUT → correctness regression!
  → Single-commit bisect confirmed → REVERTED June 18
  → Same CUDA graph fragility pattern → enforce_eager=True recommended on RTX 4090

---

## 4. verl

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #6765 per-step optimizer overrides | MERGED June 18 | OptimStepParams for Tinker | Enables Muon LR scheduling per step |
| #6791 DSv4/GLM5/KimiK2.5 via Lite | MERGED June 18 | Megatron Lite documentation | Future mlite path |
| #6731 CPPO | OPEN, progressing | Config restructuring per wuxibin89 | ★★★★★★★★ RTX 4090 #1 config |
| #6782 LoRA rank=64 breaks EOS | OPEN | rongkunxue posted image evidence | MUST rank=32/alpha=64 |
| #6512 per-unit LoRA summon | OPEN | wuxibin89 commented June 17 on VERL_USE_EXTERNAL_MODULES hook | ★★★★★★★★ 10x memory reduction progressing |
| #6468 FSDP2 CPU memory leak | OPEN, confirmed | cben484: 6.3 GiB/step on Qwen3-35B | ★★★★★★★★ Devastating for 24 GiB GPU |
| #6779 agentic rollout | OPEN | Continuous Token mechanism | Future RTX 4090 agent training |
| #6790 async trainer | MERGED June 17 | Runnable separate async trainer | ✓ |
| #6789 vision_info offload | MERGED June 17 | Thread executor offload | ✓ |
| #6784 CI OOM fix | MERGED June 18 | fullyasync+fsdp test | ✓ |
| #6738 SGLang clone skip | MERGED June 16 | Skip redundant clone → OOM reduction | ✓ |

★★★★★★★★★ #6468 confirmed multi-user scaling:
  → 0.6 GiB/step (2B) → 5.3 GiB/step (3B) → 6.3 GiB/step (35B)
  → Suspected: FSDP2 DTensor full_tensor_gloo() CPU staging buffers not released

---

## 5. MindIE/vLLM-Ascend

**v0.21.0rc1 released June 16**

| Issue/PR | Status | Change | Notes |
|----------|--------|--------|-------|
| #10488 W4A4 INT4 | OPEN | Merge conflicts June 15 | 910B only |
| #9922 MoE LoRA Ascend | OPEN | Under review | AscendC LoRA |
| #10225 DyCP | OPEN | Merge conflicts | Dynamic CP |
| #10519 MC2 accuracy regression A2 | OPEN | Qwen3-235B accuracy 0-3%! | Force ALLGATHER to isolate |
| #10579 MoE allgather NaN | OPEN | Remove torch.abs on expanded_row_idx | Ascend MoE |
| #10642 DSA CP over TP | NEW June 18 | Feature | DSA + TP coexistence |
| #10631 Kimi-Linear-48B-A3B | NEW June 17 | Model support | MLA + KDA linear attention |
| #10628 DSV4 chat failure | NEW June 17 | Main branch failure | Production issue |
| #10621 spec decoding determinism | NEW June 17 | Non-deterministic with spec on | ★★★★★★★★ Same determinism concern |
| #10622 dflash spec performance regression | NEW June 17 | NPU idle rate spikes | Performance issue |
| #10626 LoRA+tower+Qwen2VL fail | NEW June 17 | aclnnUniqueConsecutive error | LoRA + Ascend issue |
| #10640 MTP startup failure 300i | NEW June 18 | v0.21.0rc1 issue | Startup crash |

★★★★★★★★★ #10621: speculative decoding non-determinism → mirrors SGLang/vLLM CUDA concerns

---

## 6. rLLM

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #605 GRPO grouping bug | OPEN, 18+ days | ZERO comments → STILL BROKEN | ★★★★★★★★ GRPO COMPLETELY BROKEN |
| #663 Step.output fix | MERGED June 17 | ALL rewards were 0.0 before fix | ★★★★★★★★ ALL prior training INVALID |
| #665 Fireworks live catalog | MERGED June 17 | 283 public models filtered | ✓ |
| #666 Fireworks SWE-RL v2 | OPEN June 18 | Another SWE-RL variant | ✓ |
| #660 Runner export fix | MERGED June 17 | Invalid __all__ + metric aggregation | ✓ |
| #658 cumulative token Tinker | MERGED June 16 | Token ID preservation | ✓ RTX 4090 relevant |
| #661 sandbox concurrency 64 | MERGED June 16 | Default 4→64 | ✓ |

★★★★★★★★★ #605: 18+ days, ZERO developer response → community pressure needed

---

## 7. SGLang

**v0.5.13 (June 13)**

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #28582 RCE security fix | OPEN, June 17 | CVSS 9.8, 0 maintainer response | ★★★★★★★★ CRITICAL for exposed servers |
| #28588 image decompression bomb | NEW June 18 | Guard against oversized image decode | ★★★★★★★★ 2nd security issue this week! |
| #28591 DSV4 MTP revert | NEW June 18 | Revert #26471 (DSV4 online compress MTP) for testing | ★★★★★★★★ 4th DSV4 failure this week! |
| #28569 EAGLE3 CUDA graph crash | OPEN | Illegal memory access when batch shrinks | ★★★★★★★★ Another CUDA graph replay fragility! |
| #27097 multi-LoRA determinism | OPEN | No activity since June 3 | 4-factor determinism bug |
| #28566 sentinel-pad | OPEN June 17 | DP-attention foreign token mapping | Essential for DP LoRA |
| #28564 MoE LoRA gate_up fusion | OPEN June 17 | Length-2 lora_b for gate_up case | MoE LoRA throughput |
| #28574 MoE deferred finalize | OPEN June 18 | AR+residual+RMSNorm fusion for K2.5 | MoE optimization |
| #28580 FA3 skip KV opt-in | OPEN June 18 | Makes experimental path opt-in | ★★★★★★★★ Safety improvement for SM89 |
| #28583 revert head_dim regression | MERGED June 18 | Revert assignment causing test failure | ✓ |
| #28577 EAGLE regression test | MERGED June 17 | Hidden states test in spec suite | ✓ |
| #28576 bench seed default | MERGED June 17 | Unify to seed=42 | ✓ |
| #28589 docs sync | MERGED June 18 | Blog cards | ✓ |
| #28575 MTP weight update reimpl | OPEN June 17 | Revert #27749 + re-impl distributed | Spec decode weight |
| #28579 hybrid linear attention overlap | OPEN June 17 | Spec decoding fully overlap | ✓ |

★★★★★★★★★ #28588 NEW: image decompression bomb guard → 2nd security issue same week as #28582 RCE!
★★★★★★★★★ #28591 DSV4 MTP revert → 4th DSV4 failure this week (vLLM #45972, SGLang #28591, #28575, #28569)
★★★★★★★★★ See: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md

---

## 8. PyTorch (Bonus Tracking)

| Issue/PR | Status | Change | RTX 4090 Impact |
|----------|--------|--------|----------------|
| #187620 PartialOffloadPolicy | OPEN DRAFT | CI labels added June 17 | Multi-GPU ONLY, NOT dp=1 RTX 4090 |
| #184119 SM89 guard | OPEN, progressing | jansel pushing CI, latest June 15 | ★★★★★★★★ Validates P9 thesis |
| #187484 vLLM Inductor breaks on 2.13 | OPEN | frgossen bisected root cause to #184193 | Blocks vLLM #45731 |
| #187581 revert #184193 | CLOSED | NOT accepted → #187484 stays OPEN | |
| #187634 torch.where casting | NEW June 18 | Out-of-range scalar → +/-Inf for fp16 | ★★★★★★★★ fp16 NaN propagation risk on SM89! |
| #187631 dynamo segfault fix | NEW June 18 | List comprehension graph break | torch.compile stability |

---

## References

- DeepSpeed #8061 detailed analysis: notebook/projects/deepspeed-overlap-comm-compile-nan-source-reading.md
- Megatron #5394 Muon clipping: notebook/projects/megatron-5394-chained-optimizer-muon-clipping-reading.md
- vLLM #45972 cudagraph revert: notebook/projects/vllm-cuda-graph-reading.md (Section 13)
- verl #6468 memory leak: notebook/projects/verl-fsdp2-source-deep-reading.md
- SGLang #28582 RCE: notebook/projects/sglang-28582-rce-security-vulnerability-reading.md
- rLLM #605 grouping bug: notebook/projects/rllm-605-grpo-grouping-bug-source-reading.md
- MindIE deep reading: notebook/projects/mindie-vllm-ascend-source-deep-reading.md
- DSV4 systematic instability: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
