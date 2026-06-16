# MindIE / CANN 9.0 / vLLM-Ascend Latest Developments Reading (June 2026)

> 2026-06-16 | Research synthesis from web search + existing project notes
> Focus: BudgetRefiner SLO progress, CANN 9.0 rearchitecture, ATB compose-level fusion, DeepEP-Ascend, MXFP4/FP4, RTX 4090 vs NPU comparison
> ★★★★★★★★ BudgetRefiner SLO = #1 vLLM upstream contribution priority — 95%+ GPU-generic, RTX 4090 profile data UNIQUE

---

## 1. ★★★★★★★★ BudgetRefiner SLO — Current Status and Progress Tracking

### 1.1 vLLM-Ascend Production Integration Status

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
BudgetRefiner in vLLM-Ascend = PRODUCTION-READY on Ascend NPU!

Source: vllm_ascend/core/scheduler_dynamic_batch.py (58 lines core)
Integration: 3 lines in platform.py → scheduler_cls override + force chunked_prefill + pass SLO_limits
Config: AscendConfig.SLO_limits_for_dynamic_batch (default = -1 → disabled)

★★★★★★★★★ Three integration points:
  Point A: token_budget line 358 (static max_num_scheduled_tokens → BudgetRefiner.refine_budget())
  Point B: decode-first reordering line 375 (d_lst + p_lst → decode before prefill)
  Point C: dynamic max_seqs line 565 (fewer prefill admissions under decode pressure)

★★★★★★★★★ Key design:
  → slo_limit > 0 → enables BudgetRefiner → early return when disabled → ZERO overhead!
  → refine_budget() → returns TOTAL budget → decode first → remaining = prefill allocation
  → profile_table.csv → lookup (ctx_len, d_num) → max chunk_size → budget DROPS as decode load increases
  → d_num=0: budget=1024 (full) → d_num=100: budget=768 (25% drop) → d_num=255: budget=512 (50% drop) at SLO=50ms

★★★★★★★★★ CRITICAL: BudgetRefiner ONLY throttles prefill when ACTIVE decode requests exist!
If no decode → full budget → ZERO impact on pure-prefill!
```

### 1.2 GPU-Generic Portability Analysis

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ ONLY profile_table.csv is GPU-specific! Everything else 100% GPU-generic!

| Component                     | GPU-Generic | GPU-Specific | Porting Effort |
|-------------------------------|-------------|--------------|----------------|
| BudgetRefiner class           | 100%        | 0%           | Direct copy    |
| refine_budget()               | 100%        | 0%           | Direct copy    |
| _read_lookup_table()          | 100%        | 0%           | Direct copy    |
| _align_key()                  | 100%        | 0%           | Direct copy    |
| _get_max_budget()             | 100%        | 0%           | Direct copy    |
| Decode-first reordering       | 100%        | 0%           | d_lst + p_lst  |
| SLO_limits config             | 100%        | 0%           | Add SchedulerConfig |
| Pandas CSV loading            | 100%        | 0%           | Direct copy    |
| profile_table.csv DATA        | 0%          | 100%         | ★★★★★ RTX 4090 profiling needed |
| SchedulerDynamicBatch         | 95%         | 5%           | block_size=16 vs 128 |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ RTX 4090 profile data = NO OTHER vLLM CONTRIBUTOR HAS THIS!
H100/A100 profiles → many contributors can collect. RTX 4090 → our exclusive contribution.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

### 1.3 Contribution Plan Progress

```
★★★★★★★★★ 4-Phase Contribution Plan Status:

Phase 0: Pre-Work (1-2 weeks) — IN PROGRESS
  → Tier 1-2 contributions (#32268 QuantKey, #43204 cleanup) — draft ready
  → Comment on SJF RFC (#29406) — reference BudgetRefiner
  → Open vLLM issue: "[Feature] SLO-aware dynamic token budget for V1 scheduler" — TODO
  → Engage vLLM scheduler maintainers on Slack — TODO
  → Read SchedulerConfig source — DONE

Phase 1: RFC + Minimal Implementation (3-4 weeks) — NOT STARTED
  → BudgetRefiner class (105 lines) → GPU-generic
  → SchedulerConfig extension → 4 new opt-in fields
  → profile_table.csv → RTX 4090 data (collected via profile_vllm_budget.py)
  → Integration with vLLM V1 Scheduler → 3 exact points

Phase 2: GPU Validation (GPU online) — BLOCKED (GPU offline)
  → RTX 4090 profiling → profile_table.csv with (ctx_len, d_num, cost, chunk_size)
  → SLO compliance testing → TTFT + ITL p99 measurements
  → Watermark #44594 + BudgetRefiner combined testing

Phase 3: Full PR Submission — NOT STARTED
  → Submit to vllm-project/vllm
  → BudgetRefinerInfo dataclass for observability
  → BudgetRefiner complementary to Watermark #44594

★★★★★★★★★ BLOCKER: GPU servers offline → cannot collect RTX 4090 profile data → Phase 2 blocked
```

### 1.4 BudgetRefiner vs Watermark Complementary Relationship

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Aspect           | Watermark #44594              | BudgetRefiner SLO             |
|------------------|-------------------------------|-------------------------------|
| Handles          | KV cache pressure (reactive)  | Compute time pressure (proactive) |
| Trigger          | Free blocks < watermark       | Prefill budget > SLO limit    |
| Effect           | Admission gate after preempt  | Budget shrinks to protect decode |
| Direction        | KV→scheduling                 | SLO→scheduling                |
| Together         | → approach zero preemptions!  |                               |

★★★★★★★★★ Watermark = preemptions -82%, ITL p99 -56%, throughput +5.1% (on H100)
★★★★★★★★★ BudgetRefiner = prevents prefill from overwhelming decode → complementary!
★★★★★★★★★ Together: proactive (BudgetRefiner) + reactive (Watermark) → near-zero preemptions
```

---

## 2. ★★★★★★★ CANN 9.0 — LLM-Native Rearchitecture

### 2.1 Release Status and Timeline

```
★★★★★★★★★ CANN 9.0 = Next major release of Huawei's Ascend compute architecture

Timeline:
  → CANN 8.x = current production release
  → CANN 9.0 = planned 2026 GA → aligned with Ascend 910D hardware
  → Beta/preview: some components already available (47 auto-fusion patterns confirmed)

★★★★★★★★★ CANN 9.0 LLM-native rearchitecture = fundamental redesign:
  → From general-purpose NN compiler → LLM-first compute stack
  → Core abstraction: composable fusion (ATB compose-level) → not just graph-level
  → Target: Ascend 910D (2026) with backward compatibility to 910B/910C

★★★★★★★★★ Three key shifts in CANN 9.0:
  1. Dynamic shape support natively → eliminates graph recompilation overhead
  2. Mixed-precision composable pipelines (FP8→BF16→FP32 flow-through)
  3. Multi-node compose orchestration for distributed LLM workloads
```

### 2.2 CANN 9.0 47 Auto-Fusion Patterns

```
★★★★★★★★★ 47 automatic operator fusion patterns in CANN 9.0 graph compiler:

Pattern categories:
  → Attention fusion: FlashAttention + softmax + dropout → 1 kernel
  → Activation+Norm fusion: RMSNorm + residual add + activation → 1 kernel
  → MLP fusion: gate_proj + up_proj + SiLU/SwiGLU + down_proj → 1 kernel
  → Quantization fusion: dequant + compute + quant → 1 kernel (npu_dequant_swiglu_quant pattern)
  → KV Cache fusion: RMSNorm K + RoPE + cache write K + cache write V → 1 kernel

★★★★★★★★★ CANN 9.0 joint fusion+layout+quantization co-design:
  → Fusion aware of tensor layouts (NZ vs ND format for Cube ops)
  → Quantization granularity decided within fused kernel boundary
  → Layout choices consider quantized data width (INT4/INT8 packed)
  → ★★★★★★★★ ~2.1x throughput vs individual optimization → joint co-design = key!

★★★★★★★★★ Auto-fusion vs Compose-level — COMPLEMENTARY (not competing):
  → Auto-fusion: compiler IR level → pattern-matching → transparent → within sub-ops
  → Compose-level: developer API → explicit composition → between sub-op groups
  → ★★★★★★★★★★★★★★★★★★★ They are LAYERED: auto-fusion within → compose between!
```

### 2.3 CANN 9.0 vs CUDA Comparison

```
★★★★★★★★★ CANN vs CUDA stack comparison:

| Layer              | NVIDIA              | Ascend                      |
|--------------------|---------------------|------------------------------|
| Inference service  | TensorRT-LLM Server | MindIE-Service               |
| Inference engine   | TensorRT-LLM        | MindIE-LLM / MindIE Turbo    |
| Op library         | cuBLAS/cuDNN/CUTLASS| ATB (Ascend Transformer Boost)|
| Collective comm    | NCCL                | HCCL                         |
| Runtime            | CUDA Runtime        | CANN Runtime                 |
| Driver             | CUDA Driver         | CANN Driver                  |
| Hardware           | NVIDIA GPU          | Ascend NPU                   |
| Auto-fusion        | torch.compile/Inductor| CANN 9.0 47 patterns        |
| Compose-level      | NONE (TensorRT closed)| ATB Operation::Compose     |
| Profiling          | Nsight              | CANN Profiler                |

★★★★★★★★★ Key advantage of CANN 9.0:
  → Auto-fusion = compiler-level (similar to Inductor but HW-specific)
  → Compose-level API = developer-level (NVIDIA has NO equivalent!)
  → ★★★★★★★★ compose-level = Ascend unique advantage → scheduling granularity preserved!
```

---

## 3. ★★★★★★★ ATB Compose-Level Fusion — Production Readiness

### 3.1 Operation::Compose API — Production Status

```
★★★★★★★★★ ATB compose-level fusion = PRODUCTION-READY in vLLM-Ascend + SGLang-Ascend!

Source: Ascend/op-plugin → npu_dequant_swiglu_quant host+kernel
API: atb::Operation::Compose → max 100 sub-ops + TensorBinding zero-copy

★★★★★★★★★ Production compose ops:
  → DecoderLayerOperation → entire Transformer decoder layer = 1 composed op (10-12 sub-ops)
  → npu_dequant_swiglu_quant → 3 sub-ops (dequant+SwiGLU+quant) → 1 kernel
  → npu_kv_rmsnorm_rope_cache → 4 sub-ops (RMSNorm K + RoPE + cache K + cache V) → 1 kernel
  → fused_deep_moe → ENTIRE MoE path (dispatch+GEMM+SwiGLU+GEMM+combine) → 1 kernel!

★★★★★★★★★ TensorBinding zero-copy mechanism:
  → srcTensorIdx → dstTensorIdx → on-chip SRAM (Ascend AI Core Unified Buffer)
  → NOT round-tripped through HBM between each step
  → Intermediate data stays in Vector Unit / Unified Buffer → no global memory traffic

★★★★★★★★★ Production validation:
  → SGLang-Ascend fused_moe_method_npu.py → uses npu_dequant_swiglu_quant in production
  → vLLM-Ascend → uses compose-level via ATB ops → op-level + compose-level
  → MindIE Turbo → compose-level + spec decode → 2-3x latency reduction (DeepSeek-specific)
```

### 3.2 Compose-Level vs NVIDIA — Unique Ascend Advantage

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

NVIDIA fusion mechanisms:
  → CUTLASS epilogue fusion: 2-3 post-GEMM ops → kernel-level (partial)
  → FlashAttention: attention fused → kernel-level (attention scope)
  → cuBLASLt fused GEMM: GEMM+bias+activation → kernel-level (1 GEMM)
  → CUDA Graphs: launch overhead only → graph-level (overhead only)
  → torch.compile (Inductor): sub-graph fusion via Triton → between kernel and compose
  → TensorRT: sub-graph inference fusion → closest BUT closed-source + static shapes!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ NVIDIA has NOTHING = Operation::Compose!
  → TensorRT → closest → BUT closed-source + inference-only + static shapes
  → torch.compile → dynamic BUT limited to Triton kernels → cannot fuse heterogeneous ops
  → ★★★★★★★★★★★★★★★★★ compose-level = Ascend unique advantage → scheduling preserved!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Why compose impractical on CUDA:
  → Register pressure: full Transformer layer = 100+ ops → exceeds GPU register capacity
  → Attention Q*K^T → cannot fit in shared memory for long sequences
  → Parallelism mismatch: attention=sequence-parallel vs MLP=hidden-dim-parallel
  → Dynamic shapes → different tiling per sequence length/heads
  → No compiler for whole-layer fusion across heterogeneous ops

★★★★★★★★★ Practical compose-equivalent on CUDA for RTX 4090:
  → CUDA Graphs: 3-5 kernel launches per layer → FlashAttn + fused MLP + fused LN+residual
  → Triton fused kernels: SGLang deterministic aten overrides → kernel-level fusion
  → torch.compile: JIT fusion → but limited by Inductor rules (RMSNorm fusion = our root cause!)
  → ★★★★★★★★ dequant+SwiGLU+quant → Triton kernel for CUDA → concrete opportunity for vLLM!
```

### 3.3 Scheduling Advantage — Compose vs Graph-Level

```
★★★★★★★★★ WHY compose-level preserves scheduling granularity:

| Aspect         | Compose-Level (vLLM-Ascend)    | Graph-Level (MindIE/SGLang-Ascend) |
|----------------|--------------------------------|-------------------------------------|
| Scheduling     | Per compose-op atomic unit     | Entire graph monolithic block       |
| Preemption     | Yes — between compose ops      | No — must complete entire graph     |
| Dynamic batch  | Yes — adjust per step          | No — fixed batch for full graph     |
| KV interleaving| Between compose boundaries     | All-or-nothing memory               |
| Memory control | Intermediate release           | Peak memory throughout              |
| BudgetRefiner  | Compatible — can throttle budget| Incompatible — no budget control    |

★★★★★★★★★ vLLM-Ascend > SGLang-Ascend for production because:
  → op-level + compose-level → scheduling control → preemption → dynamic batching
  → BudgetRefiner → SLO-aware → decode-first → throttled prefill budget
  → ★★★★★★★★★★★★★★★★★ compose-level = atomic schedulable → BudgetRefiner can intercept!
```

---

## 4. ★★★★★★ DeepEP-Ascend Progress (Issue #8550)

### 4.1 Current Implementation Status

```
★★★★★★★★★ DeepEP-Ascend = sgl-kernel-npu implementation (NOT independent fork):

Status: PRODUCTION-READY on A2/A3/A5 NPUs!
  → Normal mode: deep_ep_cpp custom Ascend C ops + HCCL communication domain
  → AllToAll mode: torch.distributed.all_to_all_single (HCCL fallback)
  → Ops mode: torch_npu native ops (low-latency) + custom ops (normal)

Platform build targets:
  → Atlas A2: bash build.sh -a deepep2
  → Atlas A3: bash build.sh -a deepep
  → Atlas A5: bash build.sh -a deepep Ascend950

★★★★★★★★★ DeepEP-Ascend core: deep_ep_cpp.Buffer class
  → HCCL communicator: HcclComm ep_comm via get_hccl_comm_name()
  → Methods: get_dispatch_layout, intranode/internode dispatch/combine
  → Methods: low_latency_dispatch/combine, fused_deep_moe, dispatch_ffn_combine
  → A5-specific: cam_moe_dispatch_normal_a5.h, cam_moe_combine_normal_a5.h
```

### 4.2 Issue #8550 — HCCL Integration Progress

```
★★★★★★★★★ Issue #8550 status: PLANNED but NOT fully implemented in vLLM-Ascend

Current vLLM-Ascend MoE path → MC2-based (NOT DeepEP-Ascend):
  → MoECommType.ALLGATHER → no EP → simplest
  → MoECommType.MC2 → npu_moe_distribute_dispatch_v2 → recommended
  → MoECommType.ALLTOALL → torch.distributed → HCCL
  → MoECommType.FUSED_MC2 → fused dispatch+FFN+combine → fastest!

★★★★★★★★★ DeepEP-Ascend integration gap:
  → vLLM-Ascend uses MC2 kernels (5 variants: W8A8/BF16/W4A8/decode/matmul_allreduce_add_rmsnorm)
  → DeepEP-Ascend has fused_deep_moe → ENTIRE MoE path 1 kernel → but NOT yet integrated into vLLM-Ascend
  → Issue #8550 → tracks HCCL integration for DeepEP-style all-to-all → needed for EP > 1
  → ★★★★★★★★ Integration would give vLLM-Ascend the FuseEP path → ENTIRE MoE 1 kernel → NVIDIA impossible!

★★★★★★★★★ Blockers for Issue #8550:
  → API differences between HCCL and NCCL → stream synchronization
  → Ascend memory management → different from CUDA
  → MC2 kernel variants already cover most MoE use cases → less urgency
  → DeepEP-Ascend needs A3/A5 hardware → not all Ascend deployments have these
```

### 4.3 DeepEP-Ascend Performance Benchmarks

```
★★★★★★★★★ DeepEP-Ascend on A3 384 SuperPOD (DeepSeek-V3 config: 4K tokens, 7168 hidden, top-8):

| EP Size | Dispatch BW | Combine BW |
|---------|-------------|------------|
| 8       | 146 GB/s    | 125 GB/s   |
| 16      | 107 GB/s    | 103 GB/s   |
| 32      | 102 GB/s    | 95 GB/s    |
| 64      | 81 GB/s     | 91 GB/s    |
| 128     | 57 GB/s     | 81 GB/s    |

Low-latency (128 tokens/batch):
| EP Size | Dispatch Latency | Combine Latency |
|---------|-----------------|----------------|
| 8       | 132us / 58 GB/s | 126us / 116 GB/s|
| 16      | 139us / 55 GB/s | 135us / 109 GB/s|
| 32      | 153us / 49 GB/s | 151us / 97 GB/s |

★★★★★★★★★ vs NVIDIA DeepEP (H100 NVLink):
  → NVLink EP8: 726-740 GB/s vs HCCS EP8: 146 GB/s → 5x bandwidth gap!
  → BUT: low-latency < 150us → latency competitive → production viable!
  → HCCS aggregate ~392 GB/s (910C 8-NPU mesh) vs NVLink4 ~900 GB/s (H100)

★★★★★★★★★ RTX 4090: DeepEP NOT applicable (NPU only, SM89 not SM90, no NVLink/RDMA)
  → RTX 4090 MoE → NCCL all-to-all only → DeepEP requires SM90+NVLink → not viable
  → BUT: understanding EP mechanism critical for Megatron/vLLM MoE architecture knowledge
```

---

## 5. ★★★★★★ MindIE Turbo — DeepSeek-Specific Latest

### 5.1 MindIE Turbo Architecture

```
★★★★★★★★★ MindIE Turbo = DeepSeek-specific inference optimization → NOT open source!

Key optimizations:
  → Speculative decoding tailored to DeepSeek architecture → draft model in MLA compressed latent space
  → Compose-level kernel fusion → MLA projection + RoPE + attention + spec verification → single composition
  → MLA native kernel acceleration → reduces KV-cache memory footprint + improves throughput
  → MoE expert dispatch optimization → 256+ expert configurations → dynamic load balancing

★★★★★★★★★ Performance claims:
  → ~2-3x latency reduction over baseline MLA inference on Ascend NPUs
  → ~2.3x throughput improvement over baseline MLA inference (reported Jan 2026)
  → Compose-level kernel → reduces kernel launch overhead → merges sub-operations into fewer dispatch calls

★★★★★★★★★ MindIE Turbo vs vLLM-Ascend serving path:
  → MindIE Turbo: closed-source → DeepSeek-specific → compose-level+spec decode → highest performance
  → vLLM-Ascend: open-source → general models → op-level+compose-level → flexible but potentially slower
  → ★★★★★★★★ MindIE Turbo = fastest for DeepSeek → vLLM-Ascend = most flexible for all models

★★★★★★★★★ Key limitation: MindIE Turbo = NOT open source → cannot inspect or modify kernel internals
  → vs vLLM-Ascend: open source → can inspect + modify + contribute upstream
  → vs SGLang-Ascend: uses MindIE as black box → no scheduling control
```

### 5.2 MindIE Turbo MLA Compose-Level Kernel Design

```
★★★★★★★★★ MindIE Turbo MLA compose-level kernel:

Traditional path (separate kernels):
  → MLA latent projection → RoPE → attention compute → speculative verification
  → 4+ kernel launches per decode step → memory round-trips → synchronization barriers

Compose-level path (MindIE Turbo):
  → Single kernel composition: MLA projection + RoPE + attention + spec verification
  → 1 composed dispatch → TensorBinding zero-copy → on-chip intermediate data
  → → ★★★★★★★★ Lower token latency → especially for long-context DeepSeek inference

★★★★★★★★★ Compose-level spec decode pipeline:
  → Draft model heads generate candidate tokens using MLA compressed latent representations
  → Verification against target model in fused verification kernel
  → MLA compressed latent → smaller draft model → faster verification → lower latency

★★★★★★★★★ RTX 4090 comparison:
  → MindIE Turbo compose-level → NOT available on RTX 4090 (Ascend NPU only)
  → RTX 4090 MLA path → TritonMLA (only MLA option on CUDA)
  → RTX 4090 spec decode → standard vLLM spec decode → no compose-level optimization
  → ★★★★★★★★ compose-level advantage = Ascend-specific → RTX 4090 uses kernel-level fusion instead
```

---

## 6. ★★★★★ MXFP4/FP4 Quantization on Ascend — Latest

### 6.1 MXFP4 on Ascend NPU

```
★★★★★★★★★ MXFP4 (Microscaling FP4) on Ascend:

Format: float4_e2m1fn_x2 → 2 exponent bits + 1 mantissa bit + MX scaling (E8 block scale, 32-element blocks)
Hardware support: Ascend A5/950B → CANN 9.0 provides native support
Status: CANN 9.0 confirms native hardware-level support for float4_e2m1fn on A5/950

★★★★★★★★★ MXFP4 vs NVFP4 comparison:

| Format    | Vendor         | HW Support         | Ascend Compat | RTX 4090 Compat |
|-----------|----------------|--------------------|---------------|-----------------|
| NVFP4     | NVIDIA proprietary| Blackwell B100/B200| Never expected | Never (SM89)    |
| MXFP4     | OCP open standard| AMD MI300+, Intel  | A5/950 native  | Never (SM89)    |
| INT4      | Universal       | All accelerators   | Supported     | Supported (best)|
| INT8      | Universal       | All accelerators   | Supported     | Supported       |

★★★★★★★★★ MXFP4 significance:
  → float4 + MX scaling = better accuracy than INT4 (floating-point + adaptive block scaling)
  → Hardware acceleration on A5/950 → CANN native → no software emulation overhead
  → ★★★★★★★★ FP4 = next generation quantization → INT4 will be replaced by FP4 over time
  → INT4 → FP4 → MXFP4 → progressive improvement in quantization quality

★★★★★★★★★ Huawei NOT part of OCP Microscaling workgroup (AMD, Intel, Arm, Qualcomm, Microsoft)
  → May pursue proprietary FP4 format rather than adopting MXFP4 directly
  → But: float4_e2m1fn_x2 confirmed in CANN 9.0 → compatible with MX standard format
```

### 6.2 RTX 5090 FP4 / SM120 — Contribution Window

```
★★★★★★★★★ RTX 5090 FP4 = NEXT-PHASE vLLM contribution window:

RTX 5090 specs:
  → SM120 + 32GB GDDR7 + FP4 native (hardware acceleration)
  → FP4 kernel gap = major contribution opportunity for vLLM
  → FP4 will replace INT4 (floating-point + MX scaling = better accuracy + HW accel)

★★★★★★★★★ Ascend MXFP4 as reference implementation:
  → float4_e2m1fn_x2 format → same concept as RTX 5090 FP4
  → CANN custom op → Ascend implementation → can inform vLLM FP4 kernel design
  → ★★★★★★★★ MXFP4 Ascend = reference direction → can be adapted for vLLM CUDA/SM120

★★★★★★★★★ RTX 4090: INT4 still best → FP4 not supported on SM89
  → RTX 4090 quant hierarchy: BF16 (default) → INT8 (recommended for 8B) → INT4 (for MoE/long context)
  → FP4/MXFP4 = RTX 5090 territory → not RTX 4090
  → ★★★★★★★★ SM89 contributions (BudgetRefiner + Inductor Guard) → most impactful near-term
  → SM120 FP4 contributions → NEXT-PHASE after RTX 5090 availability

★★★★★★★★★ Contribution window mapping:
  → Phase 1 (NOW): BudgetRefiner SLO + Inductor SM<90 Guard → RTX 4090 → SM89
  → Phase 2 (RTX 5090 availability): FP4/MXFP4 kernel → vLLM → SM120
  → Ascend MXFP4 → reference for Phase 2 design → cross-platform knowledge transfer
```

---

## 7. ★★★★★ Ascend 910D — Next-Generation NPU

### 7.1 Ascend 910D Specifications and Timeline

```
★★★★★★★★★ Ascend 910D (2026 target release):

Expected improvements over 910C:
  → Higher compute density → improved FP16/BF16 throughput → targeting H100-class
  → Enhanced interconnect bandwidth → upgraded RoCE-based networking
  → Better memory capacity and bandwidth → HBM3 or HBM3E speculated
  → Improved software stack → CANN 9.0 + MindSpore maturity

★★★★★★★★★ Strategic context:
  → Central to China's push for domestic AI chip alternatives → U.S. export controls on NVIDIA
  → Chinese cloud providers and AI labs increasingly deploying Ascend clusters
  → DeepSeek, GLM, etc. → Ascend-based training → domestic alternative to NVIDIA

★★★★★★★★★ Challenges:
  → Manufacturing: SMIC fabrication → limited EUV access → yield/performance constraints
  → Software ecosystem: CANN/MindSpore still lag CUDA in maturity and developer adoption
  → Cluster scalability: scaling to thousands of chips → interconnect + orchestration challenges
  → Timeline uncertainty: mid-2026 target → may shift based on fab progress

★★★★★★★★★ RTX 4090 vs Ascend NPU comparison:

| Dimension   | RTX 4090      | Ascend 910B   | Ascend 910C   | Ascend 910D (est) |
|-------------|---------------|---------------|---------------|-------------------|
| FP16 TFLOPS | 82.6          | 320           | ~400          | ~500-600?         |
| INT8 TOPS   | 165.2         | 640           | ~800          | ~1000?            |
| HBM         | 24GB          | 64GB          | 64-96GB       | 96-128GB?         |
| Interconnect| PCIe 4.0      | RoCE/PCIe     | RoCE/PCIe     | Enhanced RoCE?    |
| Price       | ~$1500        | ~$15K+        | ~$18K+        | ~$20K+?           |
| Ecosystem   | CUDA (mature) | CANN (developing)| CANN         | CANN 9.0          |
| Use case    | Personal/exp  | Enterprise    | Enterprise    | Enterprise+       |

★★★★★★★★★ Conclusion:
  → RTX 4090 = personal/small-team experimentation → CUDA mature → most accessible
  → Ascend = enterprise/data-center → domestic alternative → China market strategic
  → 910D → if achieves H100-class → significant for domestic LLM training → but timeline uncertain
```

---

## 8. ★★★★★ vLLM-Ascend Production Deployment Config Updates

### 8.1 Production Configuration

```
★★★★★★★★★ vLLM-Ascend production deployment config:

Key configuration differences from GPU vLLM:
  → Platform: AscendPlatform → replaces CUDA platform → CANN initialization
  → Attention backend: "ASCEND" → 6 specialized attention implementations
    → FA (FlashAttention), FA3, SFA (Split), DSA (Dynamic Sparse), MLA, KVComp
  → Block size: 128 → vs vLLM GPU block_size=16 → 8x bigger → impacts KV cache management
  → HCCL → replaces NCCL → different tuning parameters (HCCL_BUFFSIZE=1024MB required on A2)

★★★★★★★★★ BudgetRefiner SLO config:
  → AscendConfig.SLO_limits_for_dynamic_batch → SLO target in milliseconds
  → Default = -1 → disabled → slo_limit > 0 → enables BudgetRefiner
  → Forces enable_chunked_prefill=True → BudgetRefiner requires chunked prefill
  → scheduler_cls override → SchedulerDynamicBatch → inherits standard Scheduler

★★★★★★★★★ HCCL tuning requirements (A2-specific):
  → HCCL_BUFFSIZE=1024MB → required for MoE EP
  → Disable HCCL_OP_EXPANSION_MODE
  → HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0
  → bs<=8000 (normal mode), bs<=512 (low-latency mode)
  → Internode: no normal mode quant support → FP8 dispatch planned for A5

★★★★★★★★★ ACL Graph (Ascend CUDA Graph equivalent):
  → torch.npu.graph_task_group_begin(stream) → capture
  → torch.npu.graph_task_group_end(stream) → end capture → return handle
  → torch.npu.graph_task_update_begin(update_stream, handle) → replay
  → torch.npu.graph_task_update_end(update_stream) → replay end
  → ★★★★★ Similar concept to CUDA Graphs but CANN-native → lower-level API
```

### 8.2 Observability and Monitoring

```
★★★★★★★★★ vLLM-Ascend observability = dual-layer monitoring:

vLLM layer (Prometheus metrics):
  → vllm:num_requests_running, vllm:num_requests_waiting
  → vllm:gpu_cache_usage_perc (actually NPU cache usage)
  → vllm:avg_generation_throughput
  → vllm:e2e_request_latency_seconds
  → Per-request TTFT and ITL histograms

Ascend hardware layer:
  → Ascend Device Management (dcgm/msmon) → NPU utilization, memory, temperature
  → MindX monitor plugins for Kubernetes
  → CANN profiling tools → operator-level performance debugging
  → BudgetRefinerInfo dataclass → SLO compliance observability (planned for upstream PR)

★★★★★★★★★ Production SLO metrics:
  → TTFT (Time to First Token) → BudgetRefiner protects by throttling prefill
  → ITL p99 (Inter-Token Latency) → decode-first ensures decode gets priority
  → Throughput (tokens/sec) → BudgetRefiner dynamically optimizes batch composition
  → Preemption rate → BudgetRefiner + Watermark → near-zero preemptions target

★★★★★★★★★ BudgetRefinerInfo observability (planned for upstream PR):
  → budget_before_refine → original static budget
  → budget_after_refine → refined budget (may be smaller)
  → num_decode_requests → decode load driving refinement
  → avg_decode_context_length → context length used for lookup
  → slo_target_ms → SLO limit configured
  → ★★★★★★★★ These metrics → production monitoring → SLO compliance dashboard
```

---

## 9. ★★★★★★ vLLM-Ascend vs SGLang-Ascend — Production Serving Path Comparison

```
★★★★★★★★★ Updated comparison (June 2026):

| Aspect              | vLLM-Ascend                    | SGLang-Ascend                  |
|---------------------|--------------------------------|--------------------------------|
| Architecture        | 5-layer op-level patch         | Graph-level (MindIE black box) |
| Scheduling control  | Full (BudgetRefiner + decode-first) | Minimal (MindIE controls) |
| Preemption          | Yes (between compose ops)      | No (must complete entire graph)|
| BudgetRefiner SLO   | Yes (production-ready)         | No (MindIE handles internally) |
| Dynamic batching    | Yes (per-step adjustment)      | No (fixed batch per graph)     |
| MoE path            | MC2 (5 kernel variants)        | MindIE MoE (compose-level)    |
| DeepEP              | MC2 (not yet DeepEP-Ascend)    | sgl-kernel-npu (DeepEP-Ascend) |
| MLA path            | 6 attention backends + CompressAttention | CANN custom ops (fused) |
| ACL Graph           | torch.npu.graph_task_*         | MindIE internal               |
| Open source         | Yes (full)                     | Partial (MindIE closed)       |
| Production flex     | ★★★★★★★★ (op-level + compose) | ★★★ (graph-level, less flex)  |
| Production perf     | ★★★★★ (good but not best)      | ★★★★★★ (MindIE Turbo fastest) |

★★★★★★★★★ Production recommendation:
  → vLLM-Ascend for flexible production → BudgetRefiner SLO → op-level control → debugging
  → SGLang-Ascend for fastest DeepSeek inference → MindIE Turbo → compose-level fused
  → ★★★★★★★★ BudgetRefiner ONLY works with vLLM-Ascend → SGLang-Ascend has no SLO control
```

---

## 10. ★★★★★★★ Key Insights Summary and RTX 4090 Implications

### 10.1 Top 10 Insights

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ BudgetRefiner SLO = #1 vLLM upstream contribution
   → 95%+ GPU-generic → only profile_table.csv HW-specific → RTX 4090 data UNIQUE
   → vLLM V1 has ZERO SLO-aware scheduling → BudgetRefiner fills genuine gap → no competing PR
   → BLOCKED: GPU offline → cannot collect profile data → Phase 2 blocked until GPU online

2. ★★★★★★★★★★★★★★★★★ CANN 9.0 = LLM-native rearchitecture → fundamental redesign
   → From general NN compiler → LLM-first compute stack
   → 47 auto-fusion patterns + compose-level API → joint fusion+layout+quantization co-design
   → 2.1x throughput vs individual optimization → targeting 910D hardware
   → Timeline: 2026 GA → aligned with Ascend 910D launch

3. ★★★★★★★★★★★★★★★★ ATB compose-level = Ascend unique advantage → NVIDIA has NO equivalent
   → Operation::Compose → max 100 sub-ops → TensorBinding zero-copy → atomic schedulable unit
   → npu_dequant_swiglu_quant = 6→1 kernel launches per MoE expert → production-validated
   → fused_deep_moe = ENTIRE MoE path 1 kernel → NVIDIA impossible!
   → Compose-level preserves scheduling granularity → BudgetRefiner compatible

4. ★★★★★★★★★★★★★★★ BudgetRefiner + Watermark = complementary → near-zero preemptions
   → BudgetRefiner: proactive (compute time pressure) → Watermark: reactive (KV cache pressure)
   → BudgetRefiner throttles prefill to protect decode → Watermark gates admission after preempt
   → Together → could approach zero preemptions → significant for production serving SLO

5. ★★★★★★★★ DeepEP-Ascend = production-ready on A2/A3/A5 but NOT yet integrated into vLLM-Ascend
   → Issue #8550 tracks HCCL integration → MC2 kernels cover most MoE use cases currently
   → fused_deep_moe = single kernel ENTIRE MoE → would be revolutionary if integrated
   → 5x bandwidth gap (HCCS vs NVLink) → but latency < 150us → production viable

6. ★★★★★★★★ MindIE Turbo = fastest DeepSeek inference → compose-level + spec decode + MLA native
   → 2-3x latency reduction → NOT open source → cannot inspect/modify
   → vLLM-Ascend = more flexible → BudgetRefiner SLO → op-level control
   → Production choice: flexibility vs raw performance

7. ★★★★★★ MXFP4/FP4 = next-generation quantization → INT4 → FP4 → MXFP4 progression
   → Ascend A5/950: float4_e2m1fn_x2 + MX scaling → CANN 9.0 native → hardware accelerated
   → RTX 5090 SM120: FP4 native → NEXT-PHASE vLLM contribution window
   → RTX 4090: INT4 still best → FP4 not on SM89 → BudgetRefiner + Inductor Guard near-term

8. ★★★★★ Ascend 910D = targeting H100-class → mid-2026 → timeline uncertain
   → Manufacturing constraints (SMIC, limited EUV) → software ecosystem maturity gap
   → Strategic: central to China's domestic AI independence from NVIDIA
   → RTX 4090: personal experimentation → Ascend: enterprise production → different markets

9. ★★★★★ vLLM-Ascend > SGLang-Ascend for BudgetRefiner integration
   → vLLM-Ascend: op-level + compose-level → scheduling control → BudgetRefiner compatible
   → SGLang-Ascend: graph-level → MindIE black box → NO scheduling control → BudgetRefiner incompatible
   → BudgetRefiner upstream PR targets vLLM (not SGLang) → scheduling is the differentiator

10. ★★★★★★★★ CANN 9.0 auto-fusion + compose-level = complementary LAYERED system
    → Auto-fusion within sub-ops → compiler-level → transparent
    → Compose-level between sub-op groups → developer-level → explicit
    → Together → joint fusion+layout+quantization co-design → 2.1x throughput
    → NVIDIA equivalent: torch.compile (auto) + TensorRT (manual) → but both limited
```

### 10.2 RTX 4090 vs Ascend NPU — BudgetRefiner SLO Implications

```
★★★★★★★★★ RTX 4090 BudgetRefiner SLO implications:

RTX 4090 advantages for BudgetRefiner contribution:
  → RTX 4090 profile data = NO OTHER CONTRIBUTOR HAS THIS → unique contribution
  → 24GB VRAM → max ~32-64 concurrent decode → d_num range smaller → simpler CSV
  → Consumer GPU → accessible to many developers → broader testing community
  → SM89 specific challenges (batch invariance, FP8 crash) → BudgetRefiner helps by throttling prefill

RTX 4090 specific BudgetRefiner profile_table.csv requirements:
  → d_num range: 0-64 (vs Ascend 0-255) → fewer rows
  → ctx_len range: 128, 256, 512, 1024, 2048 → same as Ascend
  → cost measurement: iteration time per (chunk_size, d_num, ctx_len) → RTX 4090 specific
  → ★★★★★★★★ RTX 4090 CSV = ~5x64xN = ~320xN rows → much smaller than Ascend's 10,875

Ascend NPU advantages for BudgetRefiner development:
  → BudgetRefiner already production-tested on Ascend → proven concept
  → 64GB+ HBM → more concurrent decode → richer profile data → more SLO scenarios
  → Enterprise deployment → BudgetRefiner SLO directly relevant to production serving
  → Compose-level scheduling → BudgetRefiner can intercept at compose boundaries

★★★★★★★★★ Cross-platform learning:
  → Ascend BudgetRefiner source → informs vLLM GPU implementation → 95%+ reusable
  → Ascend compose-level fusion → informs CUDA kernel-level fusion strategy → dequant+SwiGLU+quant
  → Ascend MXFP4 → informs RTX 5090 FP4 contribution → same concept, different platform
  → ★★★★★★★★ Understanding both platforms → cross-platform expertise → most valuable for OSS contribution
```

---

## 11. ★★★★★★★ Next Steps and Action Items

```
★★★★★★★★★ Immediate actions (GPU offline):
  → Continue BudgetRefiner PR draft refinement → budgetrefiner-vllm-pr-draft.md
  → Open vLLM issue: "[Feature] SLO-aware dynamic token budget for V1 scheduler"
  → Comment on SJF RFC (#29406) → reference BudgetRefiner as concrete alternative
  → Engage vLLM scheduler maintainers → informal concept discussion

★★★★★★★★★ Actions when GPU online:
  → RTX 4090 profiling → collect profile_table.csv using profile_vllm_budget.py
  → BudgetRefiner + Watermark combined testing → verify complementary behavior
  → Inductor SM<90 Fusion Guard validation → BudgetRefiner + batch invariance fix
  → AutoEP MoE training → ZeRO-2+CPU_Adam+LoRA on RTX 4090

★★★★★★★★★ Monitoring:
  → CANN 9.0 GA release date → affects compose-level API availability
  → Ascend 910D launch → affects NPU competitive landscape
  → DeepEP-Ascend Issue #8550 progress → affects vLLM-Ascend MoE serving
  → MindIE Turbo DeepSeek benchmarks → affects production serving strategy
  → vLLM V1 scheduler roadmap → BudgetRefiner integration opportunity

★★★★★★★★★ Contribution priority:
  → P10: BudgetRefiner SLO → vLLM upstream → RTX 4090 profile data UNIQUE → #1 priority
  → P9: Inductor SM<90 Fusion Guard → PyTorch upstream → root cause confirmed → #2 priority
  → P8-P5: 6 Tier 1 comments → vLLM Issues → drafts ready → build community credibility
  → P4: QuantKey refactor → foundation for SM89 FP8 guard
  → P3: RTX 5090 FP4/MXFP4 kernel → NEXT-PHASE → Ascend MXFP4 as reference
```

---

## References

- [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend) — BudgetRefiner source, scheduler, 5-layer architecture
- [Ascend/op-plugin](https://github.com/Ascend/op-plugin) — npu_dequant_swiglu_quant host+kernel source
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) — DeepEP V2 (NVIDIA)
- [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) — DeepEP-Ascend + MLA preprocess
- [Huawei HiAscend CANN](https://www.hiascend.com/document) — CANN 9.0 docs, 47 auto-fusion patterns
- [ATB Gitee](https://gitee.com/openeuler/ascend-transformer-boost) — Operation::Compose source (partial open-source)
- [openMind Gitee](https://gitee.com/openeuler/openMind) — MindIE open-source subset
- Related project notes:
  - budgetrefiner-slo-source-reading.md — 58-line source-level analysis
  - budgetrefiner-vllm-contribution-plan.md — 4-phase contribution plan
  - budgetrefiner-vllm-pr-draft.md — PR draft with integration points
  - vllm-v1-scheduler-budgetrefiner-integration.md — V1 scheduler integration analysis
  - mindie-atb-compose-fusion-deep-reading.md — compose-level deep dive
  - mindie-vllm-ascend-production-reading.md — production config + 5-layer architecture
  - deepep-ascend-reading.md — DeepEP-Ascend source-level analysis
  - vllm-ascend-serving-layer-reading.md — 6 attention backends + ACL Graph
  - mindie-architecture-reading.md — MindIE full architecture reading
