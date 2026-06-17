# MindIE Latest Developments — June 2026 Reading

> 2026-06-18 | Comprehensive research synthesis: GitHub/Gitee issues, PRs, releases, web search
> Focus: 10 topics (DeepEP-Ascend, CANN MXFP4, ATB compose fusion, MindIE Turbo DeepSeek, BudgetRefiner SLO, vLLM-Ascend integration, releases, MoE, deterministic inference, SGLang comparison)
> Data sources: GitHub `gh api` (deepseek-ai/DeepEP, vllm-project/vllm-ascend), web search, Gitee (limited access), existing project notes
> RTX 4090: MindIE/vLLM-Ascend is Ascend-only, but architectural lessons and BudgetRefiner SLO are GPU-generic

---

## 1. DeepEP-Ascend Issue #8550 — Status and Analysis

### vllm-ascend #8550: OPEN (Created 2026-04-22, last updated 2026-04-23)

```
★★★★★★★★ vllm-ascend issue #8550 = DeepEP for MoE collective communication feature request!

Title: [Feature]: Support DeepEP for MoE collective communication in vLLM-Ascend?
Author: 525309178
State: OPEN (no progress in ~8 weeks)
Question: Does vLLM-Ascend plan to support DeepEP as an optional MoE collective
  communication backend? What is the expected timeline? What technical blockers exist
  (HCCL vs NCCL incompatibility, hardware/software stack mismatch)?

★★★ Official response (yiz-liu, vllm-ascend maintainer):
  "Yes, we have this plan, could you please share a detailed roadmap here @zuje123?"
  → Acknowledged plan exists but NO detailed roadmap shared yet
  → Still awaiting @zuje123's response (8+ weeks with no follow-up)
```

### DeepEP upstream Ascend-related issues (deepseek-ai/DeepEP repo)

```
★★★★★★★★ DeepEP-Ascend saga: 3 issues reveal the full story

Issue #169: CLOSED (2025-05-16)
  → "Is there any plan for Ascend chips?"
  → Closed quickly → community interest in Ascend port

Issue #269: OPEN (2025-06-30 → updated 2025-07-03)
  → "If the Ascend team integrates EP-related capabilities into DeepEP, will the community welcome it?"
  → Author: Yael-X (Huawei Ascend Ecosystem Team)
  → DeepEP maintainer LyricZhao responded: suggested a brand-new repo (like vllm-project/vllm-ascend)
  → "the whole repo should be refactored, as NVIDIA and Ascend devices are totally different"

Issue #332: CLOSED (2025-07-28 → updated 2025-08-08) — THE DEFINITIVE PROPOSAL
  → Title: "Proposal: Create DeepEP-Ascend Extension for Seamless NPU Integration"
  → Author: Yael-X (Huawei Ascend Ecosystem Team)
  → ★★★★★ KEY CLAIM: "We have successfully implemented an Ascend-native EP communication backend
     that fully complies with DeepEP's low-latency mode API. This adaptation enables end-to-end
     Expert Parallelism for DeepSeek-V3 inference on Ascend clusters via SGLang framework."
  → ★★★ Urgent needs cited: ByteDance, Tencent require unified EP solutions
  → ★★★ Proposal: Step 1: Create DeepEP-Ascend repo under deepseek-ai.
     Step 2: Jointly design abstract multi-backend interfaces.

★★★★★★★★ DeepEP MAINTAINER REJECTION (sphish, 2025-08-05):
  1. "The EP backend for Ascend is totally different from the NVIDIA one,
     so there's no real need for unified maintenance."
  2. "DeepEP's interface is already quite stable. If you want to adapt
     it for Ascend, it's straightforward for your team to do it independently."
  3. "Deepseek-ai isn't an open community like vllm; putting it here
     would create maintenance burden."

★★★★★★★★ Huawei's graceful acceptance (Yael-X, 2025-08-07):
  "We fully respect DeepEP's architectural direction... We remain committed
   to advancing the EP ecosystem alongside you—albeit on separate tracks."

★★★★★★★★ CONCLUSION: DeepEP-Ascend will NOT be hosted under deepseek-ai.
  → Huawei must create and maintain their own Ascend EP backend independently
  → Pattern follows vllm-ascend model (separate repo, same org as framework)
  → DeepEP-Ascend exists in sgl-kernel-npu (confirmed in our prior reading)
```

### Current DeepEP repo stats

```
deepseek-ai/DeepEP: 9,737 stars, 271 open issues, language=Cuda, last updated 2026-06-17
  → Very active repo, but CUDA/NVIDIA-centric
  → Ascend port lives in SGLang ecosystem (sgl-kernel-npu), NOT in DeepEP repo
```

### RTX 4090 Relevance

```
★★★ RTX 4090: DeepEP-Ascend is NPU-only → NOT directly applicable
  BUT: Understanding EP communication patterns → informs MoE training on RTX 4090
  → DeepEP low-latency mode API compliance → confirms API is stable →
     vLLM MoE EP path can reference same API without needing DeepEP
  → DeepSpeed AutoEP #7938 already merged → alternative MoE path for RTX 4090
```

### 7-Framework Implications

```
★★★★★★★★ DeepEP-Ascend separation = confirmation that NVIDIA and Ascend MoE EP
  are architecturally different ecosystems → no unified MoE EP across all frameworks

Framework EP strategy:
  vLLM: DeepEP (NVIDIA) + MC2 (Ascend) → separate backends
  SGLang: DeepEP-Ascend via sgl-kernel-npu → independent Ascend EP
  MindIE: MC2 + EPLB → Ascend-native EP (no DeepEP dependency)
  DeepSpeed: AutoEP → config-only MoE → works on RTX 4090 (EP=1)
  Megatron: QB routing + EP → NVIDIA-only
  verl: relies on vLLM/SGLang rollout → inherits their EP
  rLLM: no MoE EP yet
```

---

## 2. CANN 9.0 MXFP4 Quantization Progress

### vLLM-Ascend Confirms: CANN 9.0 is NOW in production

```
★★★★★★★★ CANN 9.0.0 is the current production version for A2/A3/Ascend 950!

Source: vllm-ascend v0.20.2rc1 and v0.21.0rc1 release notes
  → CANN 9.0.0 for A2/A3/Ascend 950 (unchanged since v0.20.2rc1)
  → 310P uses CANN 9.1.0 beta
  → FULL_AND_PIECEWISE requires HDK 25.5.1+ / CANN 8.5.0+ for stream-budget fix

★★★★★★★★ MXFP4 quantization in vLLM-Ascend v0.21.0rc1:

  #8265: W4A8 MXFP4 quantization support for Ascend 950 → MERGED
  #9391: MXFP4 flatquant with row parallelism for Ascend A5 → MERGED
  #9365: MC2 dispatch and combine support for MXFP4/MXFP8 on Ascend A5 → MERGED
  #9328: MC2 combine MXFP4/MXFP8 for Ascend A5 → MERGED
  #9671: MXFP8 FlashCommV3 support on Ascend 950 → MERGED
  #9625: NZ layout support for W4A8 MoE compressed tensors → MERGED
  #10153: Fixed W4A8 MXFP quantization in shared experts → BugFix

★★★★★★★★ MXFP4 quantization matrix:

| Format        | Hardware   | Status          | PR/Issue     |
|---------------|------------|-----------------|--------------|
| W4A8 MXFP4    | Ascend 950 | MERGED          | #8265        |
| W4A8 MXFP4    | Ascend A5  | MERGED (flatquant + row parallel) | #9391 |
| MXFP4 MC2     | Ascend A5  | MERGED (dispatch + combine)      | #9365, #9328 |
| MXFP8 FlashComm | Ascend 950 | MERGED          | #9671        |
| W4A8 MoE NZ   | A2/A3/950  | MERGED          | #9625        |
| MXFP4 shared experts | All  | BugFix (was broken) | #10153  |

★★★★★★★★ Comparison with NVIDIA MXFP4:

  → NVIDIA: RTX 5090 SM120 supports NVFP4 (similar to MXFP4)
  → SGLang #28354: NVFP4 confirmed for RTX 5090
  → Ascend: MXFP4 W4A8 deployed FIRST on Ascend 950/A5
  → ★★★★★★★ Ascend BEAT NVIDIA to production MXFP4 deployment!
     vLLM-Ascend has W4A8 MXFP4 working while NVIDIA vLLM still developing NVFP4
```

### CANN Version Requirements

```
★★★★★ vLLM-Ascend dependency stack (v0.21.0rc1):
  CANN: 9.0.0 (A2/A3/Ascend 950), 9.1.0 beta (310P)
  PyTorch / torch_npu: 2.10.0
  triton-ascend: 3.2.1
  Mooncake: v0.3.9
  HDK: 25.5.1+ required for FULL_AND_PIECEWISE graph mode

★★★★★ FULL_AND_PIECEWISE requires CANN 8.5.0+:
  → Removes old stream-budget limitation
  → Enables ~32K graphs on A3, ~64K on Ascend 950
  → Older stacks fall back to PIECEWISE mode
```

### RTX 4090 Relevance

```
★★★★★★★★ MXFP4/FP4 is NEXT-PHASE contribution window for RTX 5090
  → Ascend MXFP4 deployment proves the format is viable → validates RTX 5090 FP4 path
  → W4A8 MoE MXFP4 = same architecture as our Triton dequant_swiglu_quant design!
  → ★★★ MindIE's npu_dequant_swiglu_quant = dequant+SwiGLU+quant 1 kernel
     Our P6 Triton design = same pattern but for NVIDIA GPUs
  → Ascend A5 MXFP4 → potential cross-platform contribution (QuantKey refactor P4)
```

---

## 3. ATB Compose-Level Fusion Updates

### New PRs and Kernel Additions (v0.21.0rc1)

```
★★★★★★★★ vLLM-Ascend v0.21.0rc1 adds significant ATB/compose-level updates:

New fused custom operators:
  #9382 + #9601: fused_gdn_gating AscendC operator for Ascend 950
    → Custom fused GDN gating → composes gating computation into single op
    → ★★★ GDN = Gated Depth Network → new architecture component for Qwen3.5
  #9350: A2/A3 and Ascend 950 compressor operator paths
    → Compressor ops for MLA/DSA → compose-level fusion extends to new hardware
  #9491 + #9825: LightningIndexer + SparseFlashAttention ACLNN ops
    → Sparse attention fused → compose-level: indexer + sparse FA = 1 dispatch
  #9789: Rehash for AscendStore grouped keys (DeepSeek V4 + compressed layouts)
    → KV cache management fused into compose pipeline

★★★★★★★★ FULL_AND_PIECEWISE graph mode = compose-level fusion breakthrough!

  #9572 + #9962: FULL_AND_PIECEWISE hybrid graph compilation mode
    → Combines full-graph and piecewise strategies
    → ★★★★★★ This is the compose-level fusion evolution:
       Before: full-graph capture (limited to small graphs) OR piecewise (many small captures)
       Now: FULL_AND_PIECEWISE → hybrid → large sub-graphs as compose-level units
       → ~32K graphs on A3, ~64K on Ascend 950
    → ★★★ Requires HDK 25.5.1+ / CANN 8.5.0+ → removes stream-budget limitation

★★★★★★★★ MXFP4 quantization extends compose-level fusion scope:

  #9625: NZ format for W4A8 MoE compressed tensors
    → Compose-level: quantize + matmul + dequantize fused for MoE experts
    → NZ (Narrow-Z) layout = Ascend-specific memory layout for better access patterns

★★★★★★★★ DSA (DeepSeek Sparse Attention) compose-level fusion evolution:

  #9385: DeepSeek V4 DSA attention backend end-to-end
    → DSA KV cache management → distributed inference → MTP → compose-level
  #9450 + #9441 + #9433 + #9504: DSA multistream overlap optimizations
    → Compressor overlap, indexer-select overlap, CV parallel, compute-communication overlap
    → ★★★★★★ compose-level + multistream = BOTH fusion AND parallelism in one compose unit
  #9390: IndexCache reuse DSA topk_indices across decode steps
    → ★★★ Compose-level: indexer computation persisted → reduces repeated work
```

### Compose-Level Fusion Architecture Summary

```
★★★★★★★★★ ATB compose-level fusion hierarchy (updated):

| Layer          | Mechanism                        | Scope                                     |
|----------------|----------------------------------|-------------------------------------------|
| Kernel-level   | Fused micro-kernels              | 2-3 ops (dequant+SwiGLU+quant)           |
| Graph-level    | ATB Graph.Run()                  | Inter-op memory + stream scheduling       |
| Compose-level  | Operation::Compose()             | Transformer sub-graphs as atomic units    |
| Hybrid-graph   | FULL_AND_PIECEWISE               | Large sub-graphs + piecewise fallback     |

★★★★★★★★★ Compose-level now supports:
  → MLA attention compose (npu_mla_preprocess, npu_kv_rmsnorm_rope_cache)
  → MoE expert compose (npu_dequant_swiglu_quant, npu_grouped_matmul)
  → DSA compose (DeepSeek Sparse Attention + multistream overlap)
  → Spec decode compose (draft + verify in MLA latent space)
  → Sparse attention compose (LightningIndexer + SparseFlashAttention)
  → GDN compose (fused_gdn_gating for Qwen3.5)
  → KV cache compose (AscendStore rehash + compressor)

★★★★★★★★★ NVIDIA comparison:
  → NVIDIA has NO compose-level API → closest = TensorRT (closed-source)
  → torch.compile = graph-level only, no compose-level
  → Compose-level = Ascend unique advantage confirmed
```

---

## 4. MindIE Turbo DeepSeek Inference — Updates Since Last Reading

### DeepSeek-V4 Support (Major New Development)

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ DeepSeek-V4 on Ascend 950 = the biggest new development!

v0.20.2rc1 (2026-06-03):
  → DeepSeek V4 end-to-end support: model architecture, DSA attention backend,
    KV cache management, distributed inference, tool-call parser, MTP support,
    KV Pool adaptation, custom operator enablement
  → PRs: #9270, #9385, #9228

v0.21.0rc1 (2026-06-16):
  → DeepSeek-V4 for Ascend 950: piecewise graph mode, DSA attention,
    KV cache management, MTP
  → PRs: #9757, #9935, #9473

★★★★★★★★★ DeepSeek-V4 specific features:
  → Compressor block size [32,64,128] support (#10354)
  → DSA-CP decoupled from FlashComm1 (#9697, #9910)
  → DeepSeek-V4 compressor operator paths (#9350)
  → DSA multistream overlap (#9450, #9441, #9433, #9504)
  → IndexCache reuse for DSA (#9390)
  → DSA compressed idle dummy graph fix (#9818)
  → DeepSeek V4 PP fixes (#9473)
  → DeepSeek-V4 KV Pool (requires --no-disable-hybrid-kv-cache-manager, #9975)

★★★★★★★★★ Known Issues for DeepSeek-V4:
  → KV Pool stores all states for all compression ratio families
     → 1M token sequence = ~300GB storage (same as upstream vLLM)
  → FULL_AND_PIECEWISE needs HDK 25.5.1+ / CANN 8.5.0+
```

### DeepSeek Turbo Inference Architecture (Updated)

```
★★★★★★★★ MindIE Turbo 3 pillars still valid, now extended to DeepSeek-V4:

Pillar 1: Speculative Decoding
  → NEXTN/EAGLE-style in MLA latent space → NEW: Eagle3 for Ascend
  → DeepSeek-V4 MTP support → KV Pool for draft models (#9893)

Pillar 2: Compose-Level Kernel Fusion
  → FULL_AND_PIECEWISE hybrid graph mode → extends compose to DeepSeek-V4
  → DSA compose: entire sparse attention pipeline as atomic unit
  → Multistream overlap within compose unit → fusion + parallelism combined

Pillar 3: MLA Native Kernel Acceleration
  → Still 4 compose-level MLA kernels (unchanged from prior reading)
  → NEW: DeepSeek-V4 compressor + indexer compose extensions
  → NEW: SparseFlashAttention compose for sparse attention patterns

★★★★★★★★★ DeepSeek model coverage on Ascend:

| Model           | Status            | Key Feature                  |
|-----------------|-------------------|------------------------------|
| DeepSeek-V2/V3  | Production        | MLA compose + spec decode    |
| DeepSeek-V3.1   | Production        | W8A8 + PD disaggregation     |
| DeepSeek-V3.2   | Supported         | DSA-CP decoupled             |
| DeepSeek-V4     | NEW (v0.21.0rc1)  | DSA + piecewise + MTP + KV Pool |
| DeepSeek-R1     | Production        | MLA compose + PD             |

★★★★★★★★★ 5 portable lessons from MindIE Turbo (unchanged + V4 additions):
  1. Compose-level fusion = unique scheduling granularity (Ascend-only, no NVIDIA equivalent)
  2. Spec decode in MLA latent space = reduced draft cost (portable concept)
  3. BudgetRefiner SLO = 58 lines GPU-generic (our P10 contribution)
  4. npu_dequant_swiglu_quant = dequant+SwiGLU+quant → MoE 6 kernels→1 (our P6 design)
  5. DSA multistream overlap within compose = fusion + parallelism (new lesson from V4)
```

---

## 5. BudgetRefiner SLO Implications for MindIE (58 Lines GPU-generic)

### vLLM-Ascend Integration Status (Updated)

```
★★★★★★★★★ BudgetRefiner is PRODUCTION-READY on Ascend NPU!

Source: vllm_ascend/core/scheduler_dynamic_batch.py (58 lines core)
Config: AscendConfig.SLO_limits_for_dynamic_batch (default = -1 → disabled)
Integration: 3 lines in platform.py → scheduler_cls override + force chunked_prefill + pass SLO_limits

★★★★★★★★★ Three integration points (verified at current checkout):
  Point A: token_budget → BudgetRefiner.refine_budget()
  Point B: decode-first reordering → d_lst + p_lst
  Point C: dynamic max_seqs → fewer prefill admissions under decode pressure

★★★★★★★★★ Portability: 100% GPU-generic (only profile_table.csv is GPU-specific)
  → RTX 4090 profile data = NO OTHER vLLM CONTRIBUTOR HAS THIS
  → P10 contribution priority unchanged

★★★★★★★★★ Complementarity with ATB compose-level fusion:
  → BudgetRefiner handles compute time pressure (SLO-aware scheduling)
  → Compose-level fusion handles kernel-level optimization
  → Together: SLO-aware scheduling + compose-level fusion = zero preemptions
  → This is our "watermark-budgetrefiner-complementary-synthesis" finding
```

### Implications for vLLM Upstream Contribution

```
★★★★★★★★★ BudgetRefiner SLO upstream path:

Step 1: Collect RTX 4090 profile_table.csv (profile_vllm_budget.py tool exists)
Step 2: Write vLLM PR adding BudgetRefiner to V1 scheduler
  → 3 integration points: token_budget, decode-first, dynamic max_seqs
  → Only profile_table.csv needs GPU-specific data
  → 58 lines core logic = direct copy from vLLM-Ascend
Step 3: Include RTX 4090 + H100 + A100 profile data
  → RTX 4090 = our UNIQUE contribution data
  → No other contributor has consumer GPU profile data for BudgetRefiner

★★★★★★★★★ BudgetRefiner validates MindIE's scheduling innovation:
  → MindIE solved compute pressure → BudgetRefiner captures it generically
  → vLLM upstream needs this → MindIE proved it works in production
  → Our contribution = bridge from MindIE's NPU insight to vLLM's GPU ecosystem
```

---

## 6. MindIE-vLLM-Ascend Integration Progress

### Repository and Release Status

```
★★★★★★★★★ vllm-ascend = production-grade, rapidly evolving!

Repository: vllm-project/vllm-ascend
  Stars: 2,258 | Open Issues: 2,010 | Language: C++
  Last updated: 2026-06-17 (very active)
  Description: "Community maintained hardware plugin for vLLM on Ascend"

★★★★★★★★★ Release timeline (3 releases in 2 months!):

| Release       | Date        | Key Highlight                    |
|---------------|-------------|----------------------------------|
| v0.18.0       | 2026-04-30  | Kimi-K2.x support               |
| v0.19.1rc1    | 2026-04-30  | DFlash attention, Eagle3, C8 INT8 KV |
| v0.20.2rc1    | 2026-06-03  | DeepSeek-V4, MXFP4 on A5, FA3 ready |
| v0.21.0rc1    | 2026-06-16  | DeepSeek-V4 on 950, FULL_AND_PIECEWISE, MXFP4 W4A8, batch_invariant_ops |

★★★★★★★★★ vllm-ascend release velocity:
  → v0.18.0 → v0.21.0rc1 in ~2 months (3 major version jumps)
  → Tracking upstream vLLM v0.21.0
  → CANN 9.0.0 for main hardware, 9.1.0 beta for 310P
  → PyTorch/torch_npu 2.10.0, triton-ascend 3.2.1
```

### Architecture: 5-Layer Bridge

```
★★★★★★★★★ vLLM-Ascend 5-layer architecture (unchanged from prior reading):

  Layer 1: Platform → AscendPlatform → CANN init, device management
  Layer 2: Device → AscendDevice → HCCL communicator (replaces NCCL)
  Layer 3: Op → Operation-level patches → CANN/ATB kernels
  Layer 4: Model → Model runner → Architecture detection → Backend selection
  Layer 5: Worker → Distributed worker → HCCL process groups

★★★★★★★★★ vLLM-Ascend vs SGLang-Ascend:
  vLLM-Ascend: op-level patch → fine-grained → most flexible serving path
  SGLang-Ascend: graph-level → MindIE as black box → less customizable
  → vLLM-Ascend = better for production deployment + tuning
```

### New Integration Features (v0.21.0rc1)

```
★★★★★★★★★ v0.21.0rc1 major integration features:

  #10034: batch_invariant_ops setup for RL scenarios → MERGED
    → VLLM_BATCH_INVARIANT=1 enables installation of batch invariance custom ops
    → A2/A3 NPU support for batch invariance
    → ★★★★★★ CRITICAL: batch invariance on Ascend for RL training!

  #9533: Hybrid & Mamba align prefix cache → improved cache hit rates

  #9572 + #9962: FULL_AND_PIECEWISE graph mode → hybrid compilation

  #9560: GLM4.7-Flash model support with Flash Attention backend

  #8743: CPU KV Cache Offloading support

  #9731: Mooncake SSD offload for large-scale KV cache storage

  #9638: Prefix caching with PCP/DCP → KV reuse across prefill/decode

  #9468: Layerwise KV cache event callbacks → per-layer observability

  #9765: torch reserved/allocated memory profiling in execute_model()

  #9558: Python 3.12 official support (all Docker images upgraded)
```

### Mooncake KV Transfer Ecosystem

```
★★★★★★★★★ Mooncake = Ascend's KV transfer and disaggregation infrastructure:

  #9058: DeepSeek PCP/DCP adaptation for disaggregated deployments
  #8850: Mooncake Connector hybrid attention support
  #7820: Mooncake KV pool usage optimization
  #9809: Mooncake Connector hybrid PCP/DCP for Qwen3.5
  #9646: Memfabric transfer engine backend for Mooncake KV pool
  #8394: Mooncake dummy client mode for KV transfer
  #9822: Mooncake protocol and device name configurable
  #10590: Mooncake support hybrid attention with PCP/DCP
  #10568: Group semantics for KVPool MooncakeBackend (reduce eviction fragmentation)
  #10563: RFC: Refactor Mooncake Connector (decompose bloated single file)
  #10437: KV transfer DFX (diagnostics)
  #10225: Dynamic Context Parallelism for long-context serving

★★★★★★★★★ Mooncake is Ascend's answer to vLLM's KV transfer connector
  → PCP (Prefill Context Parallel) + DCP (Decode Context Parallel)
  → Enables PD (Prefill/Decode) disaggregation on Ascend
  → Similar to vLLM's KV transfer connector but Ascend-native
```

### DeepEP Integration Path (vllm-ascend #8550)

```
★★★★★★★★★ vllm-ascend #8550 remains OPEN with no detailed roadmap:

  Current MoE EP on Ascend: MC2 (MindIE Communication Collective) + EPLB
    → MC2 = Ascend's collective communication library (replaces DeepEP)
    → EPLB = Expert Parallel Load Balancing
    → #9536: EPLB experts hotness metrics and time consumption data exposure

  ★★★★★★★★ MC2 vs DeepEP comparison on Ascend:
    → MC2 is Ascend-native → works on all Ascend hardware
    → DeepEP-Ascend exists in sgl-kernel-npu → SGLang-Ascend only
    → vLLM-Ascend uses MC2 → no DeepEP dependency needed
    → Issue #8550 asks about DeepEP integration → maintainer says "yes, we have this plan"
    → But no timeline or technical details yet

  ★★★★★★★★ MC2 quantization support expanding:
    → #9365 + #9328: MC2 dispatch/combine for MXFP4/MXFP8 on Ascend A5
    → #9625: NZ format for W4A8 MoE compressed tensors
    → #9908: NPU MoE quantization methods support TP-only (BugFix)
    → #9105: 310P MoE routing path optimization
```

---

## 7. New Releases and Version Updates

### vLLM-Ascend Release Summary

```
★★★★★★★★★ v0.21.0rc1 (2026-06-16) — THE BIGGEST RELEASE YET:

Highlights:
  → DeepSeek-V4 on Ascend 950 (piecewise + DSA + KV + MTP)
  → FULL_AND_PIECEWISE graph mode (32K/64K graphs)
  → W4A8 MXFP4 quantization for Ascend 950
  → MXFP8 FlashCommV3 on Ascend 950
  → batch_invariant_ops setup for RL (#10034)
  → Python 3.12 official support
  → Mooncake SSD offload, CPU KV offload
  → SparseFlashAttention compose
  → GDN compose (fused_gdn_gating)

Dependencies:
  → CANN 9.0.0 (A2/A3/950), 9.1.0 beta (310P)
  → torch_npu 2.10.0
  → triton-ascend 3.2.1
  → Mooncake v0.3.9
  → vLLM baseline: v0.21.0

★★★★★★★★★ Known Issues (v0.21.0rc1):
  → GLM5/GLM5.1 W4A8 advanced config issues (#9395, #9658, #9655)
  → GLM-5.1 MoE EP + FULL graph mode failures (#9503)
  → Qwen3.6-35B-A3B MTP shutdown (#9956)
  → GLM-5.1 200K long-sequence AICore timeout (#9958)
  → GLM5 W4A8 + MTP3 + FlashComm low acceptance rate (#9803)
  → DeepSeek-V4 KV Pool OOM without --no-disable-hybrid-kv-cache-manager (#9975)

★★★★★★★★★ Breaking Changes (v0.21.0rc1):
  → VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL REMOVED → migrate to AscendConfig (#9668)
  → DSA-CP decoupled from FlashComm1 → must explicitly enable both (#9697, #9910)
  → All Docker images now py3.12 (#9558)
```

### MindIE-LLM on Gitee (Limited Access)

```
★★★★★★★★★ MindIE-LLM on Gitee (gitee.com/ascend/MindIE-LLM):
  → Main open-source repository for MindIE-LLM
  → NOT accessible via GitHub (Ascend/MindIE = 404 on GitHub)
  → WebFetch blocked for gitee.com → limited direct access
  → Mirror repos exist on GitHub but with 0-1 stars (auto-created)

★★★★★★★★★ MindIE-LLM release history (from search):
  → 1.0.RC3: initial MLA support, KV cache decompression/recompression
  → 9.0.RC1: MLA KV cache compression, optimized PagedAttention for MLA
  → 9.0.0: Full DeepSeek MLA inference pipeline, continuous batching with MLA-aware scheduling

★★★★★★★★★ MindIE-LLM is part of a larger suite:
  → MindIE-LLM (LLM inference)
  → MindIE-Turbo (high-performance serving)
  → MindIE-Service (deployment/serving)
  → MindIE-Motor / MindIE-PyMotor (general model inference)
  → MindIE-SD (stable diffusion inference)
```

---

## 8. MoE-Related Improvements (MLA, DeepSeek MoE Serving)

### DeepSeek-V4 MoE on Ascend

```
★★★★★★★★★ DeepSeek-V4 MoE serving = the biggest MoE advancement:

  → DSA (DeepSeek Sparse Attention) backend → native sparse attention compose
  → Compressor block size [32,64,128] → variable compression ratios
  → IndexCache reuse → topk_indices persisted across decode steps
  → DSA multistream overlap → compute-communication overlap within compose
  → KV Pool for MoE → stores all compression ratio families
  → MTP (Multi-Token Prediction) → speculative decoding for MoE

★★★★★★★★★ MoE quantization on Ascend:

  → W4A8 MXFP4 MoE compressed tensors (#9625)
  → NZ format for MoE memory access patterns
  → MC2 dispatch/combine for MXFP4/MXFP8 MoE (#9365, #9328)
  → C8 INT8 KV cache for GQA (DeepSeek-V3.1 PD, #7474)
  → EPLB experts hotness metrics (#9536)
  → 310P MoE routing optimization (#9105)
  → MoE TP-only quantization fix (#9908)

★★★★★★★★★ MoE serving pipeline on Ascend (full stack):

  1. MC2 dispatch → expert routing + token distribution
  2. npu_moe_init_routing_v2 → routing + quantize
  3. npu_grouped_matmul → gate_up GEMM per expert
  4. npu_dequant_swiglu_quant → dequant + SwiGLU + quant = 1 kernel
  5. npu_grouped_matmul → down GEMM with swiglu_out_scale
  6. MC2 combine → expert output aggregation
  7. EPLB → load-balanced expert distribution
```

### MLA Attention Evolution

```
★★★★★★★★★ MLA attention backend evolution on Ascend:

| Backend   | Model            | Status      | Key Feature               |
|-----------|------------------|-------------|---------------------------|
| MLA       | DeepSeek-V2/V3   | Production  | W_kc absorption compose   |
| MLA W8A8  | DeepSeek-V3.1    | Production  | Quantized MLA compose     |
| DSA       | DeepSeek-V4      | NEW         | Sparse attention compose   |
| SFA       | DeepSeek-V3.2/GLM5.1 | Planned | Split FA batch invariant   |

★★★★★★★★★ MLA compose-level kernel count (unchanged, confirmed):
  → npu_kv_rmsnorm_rope_cache: RMSNorm(K) + RoPE + cache_write = 4→1
  → npu_mla_preprocess: entire MLA pre-processing (14+ inputs) → 1 op
  → npu_mla_prolog_v3: W8A8 quantized MLA prolog → more aggressive fusion
  → npu_dequant_swiglu_quant: dequant + SwiGLU + quant = 6→1 per expert

★★★★★★★★★ Speculative decoding MLA updates:
  → #9703: Fixed Eagle3 MLA shape mismatch → DeepSeek V2 Eagle3 support
  → #7619: Eagle3 + MiniMax-M2.5 (v0.19.1rc1)
  → DeepSeek-V4 MTP support (v0.20.2rc1/v0.21.0rc1)
```

---

## 9. Deterministic Inference: MindIE vs SGLang Comparison

### MindIE/vLLM-Ascend Batch Invariance

```
★★★★★★★★★ vLLM-Ascend batch invariance RFC #5487 (OPEN, 2025-12-29):

Title: [RFC]: implement batch invariant for reinforcement learning
Author: Ronald1995
State: OPEN (actively progressing)

★★★★★★★★★ RFC checklist status:
  [DONE] Basic framework for batch invariant (#5517)
  [DONE] Fix mm op error (#6107)
  [DONE] Integrate ascendc matmul and fia operator (#6590)
  [DONE] Support qwen3-32B (#6910)
  [DONE] Support qwen3-30B (#6910)
  [DONE] Feature doc (#6910)
  [MERGED] batch_invariant_ops setup (#10034) → VLLM_BATCH_INVARIANT=1
  [TODO] Support acl graph
  [TODO] DeepSeek V3.1 MLA attention batch invariant
  [TODO] DeepSeek V3.2/GLM5.1 SFA attention batch invariant
  [TODO] Qwen3.5 support
  [TODO] DeepSeek-V4 support
  [TODO] Accelerate batch invariant triton kernels

★★★★★★★★★ #10034 merged (v0.21.0rc1):
  → Adds setup/installation of batch_invariant_ops for A2/A3 NPUs
  → VLLM_BATCH_INVARIANT=1 enables installation
  → Updates op_api_common.h for AscendC custom operator loading
  → ★★★★★★ BATCH INVARIENCE ON ASCEND FOR RL TRAINING IS NOW STARTING!

★★★★★★★★★ MindIE deterministic inference (from web search):
  → MINDIE_DETERMINISTIC=1 env var → forces deterministic kernel execution
  → Performance cost: ~10-30% throughput reduction
  → ~15% for batch sizes >=8, negligible for single-request
  → Recommended for testing/debugging/compliance, not production serving
  → Nondeterministic ops: parallel reductions, softmax, certain attention kernels
  → Da Vinci architecture threading = different from GPU → distinct reproducibility challenge
```

### SGLang Deterministic Inference Comparison

```
★★★★★★★★★ SGLang deterministic = KERNEL-level gold standard:

  → 7 aten overrides via constexpr → KERNEL-level batch-invariant
  → Source-verified: mm/addmm/_log_softmax/mean.dim/rms_norm/mm.dtype/bmm
  → murmur_hash32 Gumbel-max float64 sampling → BLOCK_SIZE=1024
  → MoE LoRA + deterministic = unique capability (no other framework has both)
  → ★★★★★★ KERNEL > COMPILE > NONE (3-layer deterministic comparison)

★★★★★★★★★ MindIE/vLLM-Ascend vs SGLang deterministic comparison:

| Aspect              | SGLang (NVIDIA)           | MindIE/vLLM-Ascend          |
|---------------------|---------------------------|------------------------------|
| Approach            | KERNEL-level constexpr     | AscendC custom ops           |
| Scope               | 7 aten overrides           | matmul + fia (2 ops so far)  |
| MoE deterministic   | murmur_hash32 Gumbel-max   | Not yet (TODO in #5487)      |
| MLA deterministic   | Not applicable             | TODO in #5487 (DSV3.1/V3.2)  |
| Sampling            | float64 Gumbel-max         | Not specified yet             |
| RL training         | verl integration           | VLLM_BATCH_INVARIANT=1 start |
| Performance cost    | ~5-10%                     | ~10-30%                       |
| Maturity            | Production (7 overrides)   | Early (2 ops, many TODOs)    |
| Deterministic level | KERNEL (highest)           | KERNEL (AscendC)              |

★★★★★★★★★ Key comparison insights:
  → SGLang's deterministic is MORE mature (7 ops production vs 2 ops early)
  → Both use KERNEL-level approach (constexpr for SGLang, AscendC for vLLM-Ascend)
  → SGLang has MoE deterministic → vLLM-Ascend does NOT yet (critical gap)
  → SGLang has float64 sampling → vLLM-Ascend has no equivalent yet
  → vLLM-Ascend's RFC #5487 has clear roadmap → will catch up over time
  → ★★★★★★ Both frameworks confirm: KERNEL-level deterministic is the correct approach
     (COMPILE-level torch.compile is NOT sufficient for batch invariance)
```

---

## 10. Cross-Framework Synthesis and Priority Actions

### 7-Framework MindIE Position (Updated)

```
★★★★★★★★★ MindIE/vLLM-Ascend in 7-framework comparison (updated rankings):

| Framework     | MindIE Relevance                    | RTX 4090 Relevance               |
|---------------|--------------------------------------|-----------------------------------|
| vLLM          | vllm-ascend = production bridge      | BudgetRefiner SLO P10 contribution|
| SGLang        | sgl-kernel-npu = DeepEP-Ascend host  | Deterministic gold standard       |
| MindIE        | PRIMARY subject of this reading      | Architectural lessons portable    |
| DeepSpeed     | No Ascend bridge                     | AutoEP MoE on RTX 4090            |
| Megatron      | No Ascend bridge                     | QB routing NVIDIA-only            |
| verl          | vllm-ascend rollout path             | CPPO+bypass RTX 4090              |
| rLLM          | No Ascend bridge                     | Tinker #1 RTX 4090                |

★★★★★★★★★ MindIE's UNIQUE contributions to the 7-framework landscape:
  1. Compose-level fusion (Ascend-only, no NVIDIA equivalent)
  2. MXFP4 W4A8 production deployment (beats NVIDIA to market)
  3. BudgetRefiner SLO (58 lines GPU-generic → our P10 contribution)
  4. npu_dequant_swiglu_quant (6→1 MoE kernel → our P6 design reference)
  5. FULL_AND_PIECEWISE hybrid graph mode (compose-level evolution)
  6. DeepSeek-V4 DSA compose (sparse attention + multistream overlap)
  7. Mooncake PD disaggregation (Ascend-native KV transfer)
  8. batch_invariant_ops for RL (AscendC custom ops → KERNEL-level)
```

### Critical RTX 4090 Findings (MindIE-Related)

```
★★★★★★★★★ MindIE findings that directly impact RTX 4090 work:

  ★★★★★★★★★ BudgetRefiner SLO = #1 vLLM upstream contribution
    → 58 lines 100% GPU-generic → RTX 4090 profile data UNIQUE
    → vLLM-Ascend proved it works → we port it to vLLM upstream

  ★★★★★★★★★ Triton dequant_swiglu_quant (P6)
    → MindIE's npu_dequant_swiglu_quant = same pattern (6→1 MoE kernel)
    → Our Triton design = NVIDIA GPU port of same concept
    → After Inductor Fusion Guard (P9) → P6 can proceed

  ★★★★★★★★★ Inductor SM<90 Fusion Guard (P9)
    → SGLang confirmed: KERNEL > COMPILE > NONE for deterministic
    → vLLM-Ascend batch_invariant_ops confirms KERNEL-level approach
    → MindIE compose-level = KERNEL-level → validates our Fusion Guard approach

  ★★★★★★★★★ MXFP4/FP4 (P3 NEXT-PHASE)
    → Ascend deployed MXFP4 W4A8 production → proves viability
    → RTX 5090 FP4 = corresponding NVIDIA path
    → SGLang #28354 NVFP4 confirmed → our NEXT-PHASE contribution window
```

### Monitor Items (Updated)

```
★★★★★★★★★ New monitor items from this reading:

  vllm-ascend #8550: DeepEP for MoE collective communication → OPEN
    → Maintainer acknowledged plan → awaiting @zuje123 detailed roadmap
    → 8+ weeks with no progress → may be slow

  vllm-ascend #5487: batch invariant for RL → OPEN (RFC)
    → 2 ops done, many TODOs (acl graph, DSV3.1 MLA, DSV4, triton kernels)
    → Actively progressing → watch for MLA batch invariant

  vllm-ascend #10034: batch_invariant_ops setup → MERGED (v0.21.0rc1)
    → VLLM_BATCH_INVARIANT=1 → A2/A3 support → RL training starting

  vllm-ascend #9503: GLM-5.1 MoE EP + FULL graph failures → OPEN
    → MoE EP + FULL graph incompatible → workaround: PIECEWISE/eager

  vllm-ascend #9975: DeepSeek-V4 KV Pool issues → OPEN
    → Requires --no-disable-hybrid-kv-cache-manager → OOM without it
    → 1M tokens = ~300GB storage → same as upstream vLLM

  DeepEP #332: DeepEP-Ascend proposal → CLOSED (rejected by DeepEP)
    → Huawei will maintain DeepEP-Ascend independently
    → Follows vllm-ascend model (separate repo)

  DeepEP #269: Ascend EP integration welcome? → OPEN
    → Community discussion ongoing → no resolution

  MindIE-LLM Gitee: releases continuing → 9.0.0 stable
    → MLA inference pipeline production → DeepSeek-V2/V3/V4 full support
    → Limited direct access → monitor via vllm-ascend releases

  vllm-ascend releases: v0.21.0rc1 (2026-06-16) → latest
    → Watch for v0.21.0 stable release → may include more batch invariance

  Mooncake: v0.3.9 → active development
    → PD disaggregation key technology for Ascend
    → KV transfer + hybrid attention + SSD offload → rapidly evolving

  CANN 9.1.0 beta for 310P → production CANN 9.0.0 for A2/A3/950
    → Watch for CANN 9.1.0 stable → may bring more quantization features
```

---

## Summary: MindIE June 2026 State

```
★★★★★★★★★ MindIE/vLLM-Ascend state as of 2026-06-18:

  PRODUCTION-READY:
    → vLLM-Ascend v0.21.0rc1 → DeepSeek-V2/V3/V4/R1 full support on Ascend
    → CANN 9.0.0 → MXFP4 W4A8 quantization deployed
    → MC2 + EPLB → MoE expert parallelism on Ascend
    → FULL_AND_PIECEWISE → compose-level graph compilation
    → Mooncake PD disaggregation → KV transfer infrastructure
    → BudgetRefiner SLO → 58 lines GPU-generic scheduling

  IN PROGRESS:
    → batch_invariant_ops for RL → 2 ops done, many TODOs (#5487)
    → DeepEP-Ascend for vllm-ascend → acknowledged, no roadmap (#8550)
    → SparseFlashAttention compose → expanding
    → MXFP4/FP8/FP4 quantization → Ascend A5 expansion

  BLOCKERS:
    → GLM-5.1 MoE EP + FULL graph incompatibility (#9503)
    → DeepSeek-V4 KV Pool OOM without special flag (#9975)
    → FULL_AND_PIECEWISE needs HDK 25.5.1+ / CANN 8.5.0+

  RTX 4090 IMPACT:
    → BudgetRefiner SLO = P10 contribution (unchanged priority)
    → Triton dequant_swiglu_quant = P6 (after P9 Inductor Guard)
    → MXFP4/FP4 = P3 NEXT-PHASE (RTX 5090)
    → Batch invariance = validates KERNEL-level approach
    → Compose-level fusion = architectural lesson (no NVIDIA port yet)
```

---

## Sources

- GitHub `gh api`: deepseek-ai/DeepEP (issues #169, #269, #332, #662-#667), vllm-project/vllm-ascend (releases, issues #8550, #5487, #10034, #10516, #9975, #9503)
- vllm-ascend release notes: v0.21.0rc1, v0.20.2rc1, v0.19.1rc1, v0.18.0
- Web search: MindIE Turbo DeepSeek, CANN MXFP4, BudgetRefiner SLO, ATB compose fusion, batch invariance
- Existing project notes: mindie-turbo-deepseek-inference-source-reading.md, mindie-atb-compose-fusion-deep-reading.md, mindie-budgetrefiner-slo-source-reading.md, mindie-vllm-ascend-production-reading.md, mindie-cann-9-latest-developments-reading.md, deepep-ascend-reading.md
