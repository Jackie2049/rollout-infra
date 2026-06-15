# MindIE ATB Compose-Level Fusion Deep Dive — Source-Level Analysis

> 2026-06-16 | Ascend/op-plugin | ATB Operation::Compose | npu_dequant_swiglu_quant | TensorBinding | MoE compose
> ★★★★★ ATB 3层fusion: kernel→graph→compose → compose = atomic schedulable unit → scheduling granularity preserved!
> ★★★★★ npu_dequant_swiglu_quant = dequant+SwiGLU+quant 1 kernel → MoE 6 kernels→1 → 5x launch reduction
> ★★★★★ TensorBinding zero-copy → srcTensorIdx→dstTensorIdx → on-chip SRAM → no HBM round-trip
> ★★★★★★★ NVIDIA has NO compose-level API → closest = TensorRT (closed-source) → compose-level = Ascend独有优势!

## 1. ★★★★★ Operation::Compose API — 3-Layer Fusion Hierarchy

```
★★★★★★★ ATB fusion hierarchy:

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| Kernel-level | Fused micro-kernels | 2-3 basic ops (dequant+SwiGLU+quant) |
| Graph-level | ATB Graph.Run() | Inter-op memory optimization + stream scheduling |
| Compose-level | Operation::Compose() | Entire Transformer sub-graphs as atomic dispatchable units |

★★★★★★★ Operation::Compose C++ API:

// 1. Create composite operation
atb::Operation *compositeOp = atb::CreateOperation(
    atb::OperationDesc{atb::OperationType::OP_COMPOSE});

// 2. Add sub-operations sequentially (max 100 per compose!)
compositeOp->AddOperation(matmulOp);      // e.g., Linear
compositeOp->AddOperation(activationOp);  // e.g., SiLU/SwiGLU
compositeOp->AddOperation(normOp);        // e.g., RMSNorm

// 3. TensorBinding for zero-copy
atb::TensorBinding binding;
binding.srcTensorIdx = 0;    // Output of producer sub-op
binding.dstTensorIdx = 0;    // Input of consumer sub-op
binding.subOpIdx = 0;
binding.bindingType = INPUT;
compositeOp->SetTensorBinding(binding);

// 4. Build → finalize + validate + optimize
compositeOp->Build();

// 5. Execute as single dispatchable unit
compositeOp->Run(context, variantPack);

★★★★★★★ Limits:
  → Max 100 sub-operations per OperationCompose
  → Max 3 composition depth (nested compose)
  → Typical DecoderLayerOperation = 10-12 sub-ops (RMSNorm+Attention+Residual+RMSNorm+Linear+SwiGLU+Linear+Residual)

★★★★★★★ Key compose ops in production:
  → DecoderLayerOperation → entire Transformer decoder layer = 1 composed op
  → EncoderLayerOperation → encoder-side equivalent
  → npu_dequant_swiglu_quant → 3 sub-ops at kernel level
  → npu_kv_rmsnorm_rope_cache → 4 sub-ops (RMSNorm K + RoPE + cache write K + cache write V)
```

## 2. ★★★★★ npu_dequant_swiglu_quant — Flagship Compose Fusion

```
★★★★★★★ SGLang-Ascend production source code (fused_moe_method_npu.py):

# W8A8 MoE decode path:
sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
    torch.ops.npu.npu_moe_init_routing_v2(...)   # routing+quant
)
hidden_states = torch.ops.npu.npu_grouped_matmul(...)  # gate_up GEMM

# THE FLAGSHIP FUSION: dequant + SwiGLU + quant = 1 kernel!
hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
    hidden_states, quant_mode=1, activate_left=True
)

output = torch.ops.npu.npu_grouped_matmul(...)  # down GEMM with swiglu_out_scale

★★★★★★★ Extended signature (DeepEP path):
torch.ops.npu.npu_dequant_swiglu_quant(
    x=hidden_states,
    weight_scale=layer.w13_weight_scale,      # weight dequant scale
    activation_scale=hidden_states_scale,      # activation dequant scale
    bias=None,
    quant_scale=None,                          # output requant (auto-computed)
    quant_offset=None,
    group_index=group_list,                    # expert group for MoE
    activate_left=True,                        # gate on left side of SwiGLU
    quant_mode=1,                              # per-token dynamic quant
)

★★★★★★★ 5x kernel launch reduction mechanism:
  → Without fusion per expert: dequant_weight + matmul_gate + dequant_activation + SiLU + multiply + quant_output = 6 kernels
  → With fusion: npu_dequant_swiglu_quant = 1 kernel → 6→1 per expert → 8 experts → 48→8 = 6x!
  → "5x" accounts for routing/init/finalize kernels that remain unfused

★★★★★★★ SwiGLU formula in 1 kernel:
  dequant_x = x_quant * activation_scale
  dequant_gate = x_quant * weight_scale
  result = SiLU(dequant_x) * dequant_gate   # SwiGLU activation
  output = result / quant_scale + offset     # requantize for next GEMM

★★★★★★★ How fusion works:
  → Intermediate data stays in Ascend AI Core on-chip buffers (Vector Unit / Unified Buffer)
  → NOT round-tripped through HBM global memory between each step
  → AscendC kernel uses tiling across Cube+Vector+Scalar engines
  → Tiling host code (npu_dequant_swiglu_quant_tiling.cpp) computes optimal tile sizes

★★★★★★★ Source locations:
  → Host: op-plugin/op_plugin/ops/op_host/npu_dequant_swiglu_quant/ (tiling + registration)
  → Kernel: op-plugin/op_plugin/ops/op_kernel/npu_dequant_swiglu_quant/ (AscendC kernel)
  → Python: op-plugin/op_plugin/ops/op/npu_dequant_swiglu_quant.py
  → Repo: https://github.com/Ascend/op-plugin
```

## 3. ★★★★★ Compose-Level vs Graph-Level — Scheduling Advantage

```
★★★★★★★ CRITICAL: compose-level preserves scheduling granularity!

| Aspect | Compose-Level (ATB/vLLM-Ascend) | Graph-Level (MindIE/SGLang-Ascend) |
|--------|----------------------------------|-------------------------------------|
| Scheduling | Per compose-op atomic unit | Entire graph monolithic block |
| Preemption | Yes — between compose ops | No — must complete entire graph |
| Dynamic batch | Yes — adjust per step | No — fixed batch for full graph |
| KV interleaving | Between compose boundaries | All-or-nothing memory |
| Memory control | Intermediate release | Peak memory throughout |
| Latency | Bounded per compose unit | Variable, long single block |
| Debugging | Per-op inspection | Black box |

★★★★★★★ WHY vLLM-Ascend > SGLang-Ascend for production:
  → ★★★★★ vLLM-Ascend → op-level + compose-level → scheduling control → preemption → dynamic batching
  → ★★★★★★★★ SGLang-Ascend → graph-level → MindIE black box → NO scheduling control → NO preemption
  → → ★★★★★★★★★★★★★★★★★★★★★ compose-level = atomic schedulable → scheduler can preempt → production-safe!
```

## 4. ★★★★★★★ RTX 4090 Comparison — NVIDIA Has NO Compose-Level API

```
★★★★★★★★★ NVIDIA fusion vs Ascend compose:

| NVIDIA Mechanism | Scope | Equivalent ATB Layer |
|-----------------|-------|---------------------|
| CUTLASS epilogue fusion | 2-3 post-GEMM ops | Kernel-level (partial) |
| FlashAttention | Attention fused | Kernel-level (attention scope) |
| cuBLASLt fused GEMM | GEMM+bias+activation | Kernel-level (1 GEMM) |
| CUDA Graphs | Launch overhead only | Graph-level (overhead only) |
| torch.compile (Inductor) | Sub-graph fusion via Triton | Between kernel and compose |
| TensorRT layer fusion | Sub-graph inference fusion | Closest BUT closed-source! |

★★★★★★★★★ NVIDIA has NOTHING = Operation::Compose:
  → ★★★★★★★ TensorRT → closest → BUT closed-source + inference-only + static shapes!
  → ★★★★★★★★★★★★★ torch.compile → Inductor → dynamic BUT limited to Triton kernels → cannot fuse heterogeneous ops!
  → → ★★★★★★★★★★★★★★★★★★★★★★ compose-level = Ascend独有优势 → NVIDIA没有等效API!

★★★★★★★★★ Why compose-level impractical on CUDA:
  → 1. Register pressure: full Transformer layer = 100+ ops → exceeds GPU register capacity
  → 2. Attention Q*K^T → cannot fit in shared memory for long sequences
  → 3. Parallelism mismatch: attention=sequence-parallel vs MLP=hidden-dim-parallel → different thread configs
  → 4. Dynamic shapes → different tiling per sequence length/heads
  → 5. No compiler for whole-layer fusion across heterogeneous ops

★★★★★★★★★ Practical compose-equivalent on CUDA:
  → CUDA Graphs → 3-5 kernel launches per layer → FlashAttention + fused MLP + fused LN+residual
  → Triton fused kernels → SGLang deterministic aten overrides → kernel-level fusion
  → torch.compile → JIT fusion → but limited by Inductor rules
  → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ dequant+SwiGLU+quant → Triton kernel for CUDA → concrete opportunity!
```

## 5. ★★★★★ MoE Compose Fusions — Production Paths

```
★★★★★★★ MoE kernel launch comparison:

| Path | NVIDIA (unfused) | Ascend (compose-fused) |
|------|-----------------|------------------------|
| W8A8 decode | ~24 (routing+6ops*8exp+combine) | 4 (routing+2gmm+swiglu_quant+unpermute) |
| W8A8 DeepEP | ~24+ | 3 (2gmm+dequant_swiglu_quant) |
| FuseEP (full) | ~24+ | 1 (fused_deep_moe = ENTIRE MoE path in 1 kernel!) |
| W4A4 | ~24 | 6 (no dequant_swiglu_quant for INT4) |

★★★★★★★ FuseEP = most aggressive fusion:
  → fused_deep_moe → MC2 dispatch + GEMM1 + swiglu + GEMM2 + MC2 combine → 1 kernel!
  → → ★★★★★★★★★★★★★★★★★ ENTIRE MoE path → single kernel → NVIDIA不可能!

★★★★★★★ Additional compose-level fusions (SGLang source):
  → npu_moe_init_routing_v2 → routing + dynamic quant + token permutation → 1 op
  → npu_moe_finalize_routing → combine + weight multiply + unpermute → 1 op
  → npu_moe_token_unpermute → token unpermute + weighted sum → 1 op
  → npu_moe_gating_top_k_softmax → gating + topk + softmax → 1 op
  → npu_kv_rmsnorm_rope_cache → RMSNorm K + RoPE + cache write → 1 op (MLA path)
```

## 6. ★★★★★ CANN 9.0 Auto-Fusion vs Compose-Level — Complementary!

```
★★★★★★★ Relationship: Complementary, NOT competing!

| Aspect | CANN 9.0 Auto-Fusion | ATB Compose-Level |
|--------|----------------------|-------------------|
| Level | Compiler IR (automatic) | Developer API (explicit) |
| Control | Pattern-matching (transparent) | Manual composition |
| Target | Generic NN ops (47 patterns) | Transformer-specific |
| Scheduling | No hooks | Atomic schedulable units |
| Quantization | Fixed patterns | Quantization-aware within compose |

★★★★★★★ CANN auto-fusion → low-level patterns (MatMul+Bias+Activation)
  → ATB compose-level → high-level patterns (Transformer layers, MoE paths)
  → ★★★★★★★★★★★★★★★★★★★★ They are LAYERED: auto-fusion within sub-ops → compose-level between sub-op groups!

★★★★★★★ CANN 9.0 joint fusion+layout+quantization co-design:
  → Fusion aware of tensor layouts (NZ vs ND format for Cube ops)
  → Quantization granularity decided within fused kernel boundary
  → Layout choices consider quantized data width (INT4/INT8 packed)
  → ★★★★★★★★ ~30% end-to-end throughput vs sequential optimization → 2.1x vs individual!
```

## 参考
- Ascend op-plugin: github.com/Ascend/op-plugin → npu_dequant_swiglu_quant host+kernel source
- Ascend pytorch: github.com/Ascend/pytorch → torch_npu with npu_* ops
- ATB Gitee: gitee.com/openeuler/ascend-transformer-boost → OperationCompose source (partial open-source)
- SGLang-Ascend: fused_moe_method_npu.py → all MoE compose paths with npu_dequant_swiglu_quant
- SGLang-Ascend: fuseep.py → fused_deep_moe single-kernel MoE path
- SGLang-Ascend: mla_preprocess.py → npu_kv_rmsnorm_rope_cache compose fusion
- vLLM-Ascend: op-level + compose-level → recommended for production over SGLang-Ascend graph-level
- Related notes: mindie-atb-kernel-architecture-reading.md, mindie-vllm-ascend-production-reading.md, mindie-production-deployment-reading.md
