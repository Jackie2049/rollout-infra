# MindIE ATB Kernel Architecture 源码级深度阅读

> 2026-06-15 | 源码: sglang/srt/hardware_backend/npu/ + ATB Gitee + CANN文档 + vLLM-Ascend
> 核心: ATB=昇腾Transformer专用kernel库(Cube/Vector/Scalar三引擎) → 3层融合(kernel+graph+compose) → torch.ops.npu.*命名空间 → npu_dequant_swiglu_quant三融合(unique!) → FP8 910C=全支持/910B=E4M3 only

## 1. ATB操作类型全景

| 类别 | ATB OpTypes | NVIDIA等价 | 备注 |
|------|-------------|-----------|------|
| **Linear/Matmul** | OP_LINEAR, OP_MATMUL | cuBLAS GEMM | Cube Unit执行; FP16/BF16/INT8/INT4 |
| **Attention** | OP_FLASH_ATTENTION, OP_PAGED_ATTENTION, OP_WINDOW_ATTENTION | FlashAttention-2/3 | Q*K^T+softmax+dropout+attn*V; MHA/GQA/MQA |
| **Normalization** | OP_RMS_NORM, OP_LAYER_NORM | Triton RMSNorm | 融合norm+residual |
| **Embedding** | OP_EMBEDDING, OP_POSITION_EMBEDDING | torch embedding | Lookup+position融合 |
| **Activation** | OP_RELU, OP_GELU, OP_SILU, OP_SIGMOID | cuDNN activation | SiLU→SwiGLU关键 |
| **Quantization** | OP_QUANT, OP_DEQUANT | Custom CUTLASS/Triton | INT8/INT4/FP8融合进计算 |
| **RoPE** | (custom fused) | Custom CUDA RoPE | 融合rotary embedding |
| **Sampling** | OP_TOP_K, OP_TOP_P (fused) | Custom CUDA sampling | TopK+TopP+temperature融合 |
| **Communication** | OP_ALL_GATHER, OP_ALL_REDUCE, OP_REDUCE_SCATTER | NCCL | HCCL集成进ATB graph |

**关键差异**: cuBLAS/cuDNN覆盖所有NN ops(CNN/RNN等); ATB专注Transformer → 缺CNN但更深Transformer融合(PagedAttention/SwiGLU/MLA/MoE routing)

## 2. SGLang-Ascend实际使用的ATB ops

```python
# sglang/srt/hardware_backend/npu/ 实际生产使用:

# Attention
torch.ops.npu.npu_fused_infer_attention_score  # 替代FlashInfer/FA2

# MoE
torch.ops.npu.npu_moe_gating_top_k_softmax      # MoE gating
torch.ops.npu.npu_moe_gating_top_k               # MoE topk
torch.ops.npu.npu_grouped_matmul                 # grouped GEMM for MoE experts
torch.ops.npu.npu_moe_init_routing_v2            # MoE dispatch
torch.ops.npu.npu_moe_finalize_routing           # MoE finalize
torch.ops.npu.npu_moe_token_unpermute            # MoE unpermute

# Quantization
torch.ops.npu.npu_dynamic_quant                  # per-token动态量化
torch.ops.npu.npu_quant_matmul                   # quantized linear
torch.ops.npu.npu_convert_weight_to_int4pack     # INT4 weight packing

# Norm+Activation
torch.ops.npu.npu_rms_norm                       # 融合RMSNorm
torch.ops.npu.npu_swiglu                         # 融合SwiGLU (gate*SiLU(up))
torch.ops.npu.npu_dequant_swiglu_quant           # ★ 三融合: dequant+SwiGLU+quant!

# RoPE
torch.ops.npu.npu_interleave_rope                # interleave+RoPE
torch.ops.npu.npu_kv_rmsnorm_rope_cache          # KV RMSNorm+RoPE+cache写入

# MLA (DeepSeek-V2 specific)
mla_preprocess                                    # MLA专用预处理 (NVIDIA无等价!)

# LoRA
sgmv_shrink / sgmv_expand                         # LoRA SGMV kernels

# DeltaNet
recurrent_gated_delta_rule                        # Gated DeltaNet attention
```

## 3. 三层融合架构

```
ATB 3层融合层次:

1. Kernel-level融合: 微内核合并基本ops → 单次kernel launch
   例: npu_dequant_swiglu_quant → 3 ops→1 kernel → 省3次memory round-trip!

2. Graph-level融合: ATB Graph对象 → ops组成执行子图 → inter-op内存优化+stream调度
   → Graph.Run() → 一次执行整组ops → 减少launch overhead

3. Operation composition: Operation::Compose() → ops可嵌套子ops
   → DecoderLayerOperation = RMSNorm + Attention + Residual + SwiGLU MLP + ...
   → 一层transformer = 1个compose op → 极简调用!

→ vs NVIDIA: CUDA/Triton只有kernel级融合 → CUTLASS epilogue融合(2-3 ops)
→ ATB compose级融合 → 整层Transformer = 1 op → 更彻底!
```

## 4. Transformer融合kernel详解

| 融合模式 | ATB op | 融合内容 | NVIDIA等价 |
|----------|--------|---------|-----------|
| RMSNorm+Residual | `npu_rms_norm` | norm+residual add | Triton RMSNorm(单独) |
| SwiGLU MLP | `npu_swiglu` | gate*SiLU(up) | 3个单独kernel |
| **Dequant+SwiGLU+Quant** | `npu_dequant_swiglu_quant` | weight dequant+SwiGLU+requant | 5+单独kernel! |
| KV RMSNorm+RoPE+Cache | `npu_kv_rmsnorm_rope_cache` | norm K+RoPE+写KV cache | 3单独ops |
| Interleave+RoPE | `npu_interleave_rope` | Q/K interleave+rotary | 单独ops |
| MLA Preprocess | `mla_preprocess` | MLA Q/K/V compact+RoPE | **NVIDIA无等价!** |
| MoE Dispatch+Compute+Combine | `npu_moe_*` + `npu_grouped_matmul` | routing+GEMM+reduce | FlashInfer单独步骤 |
| Quantized Linear | `npu_quant_matmul` | dequant+GEMM+bias 1op | cuBLAS+单独dequant |
| LoRA SGMV | `sgmv_shrink`+`sgmv_expand` | LoRA A/B batch segmented | FlashInfer SGMV |

**npu_dequant_swiglu_quant = ATB独特创新!**
```
→ NVIDIA没有这个三融合 → MoE expert compute路径需要5+单独kernel
→ ATB: dequant(FP8/INT4→BF16) → SwiGLU(gate*SiLU(up)) → quant(BF16→FP8/INT4)
→ 全部1个kernel → 省3-4次memory round-trip → MoE推理更快!

→ MoE expert路径: dequant+SwiGLU+quant × 8 experts = 24+单独kernel(NVIDIA)
→ ATB: 1 kernel per expert × 8 = 8 kernel → 3x fewer launches!
```

## 5. FP8支持对比

| 方面 | NVIDIA (H100) | Ascend 910B | Ascend 910C |
|------|---------------|-------------|-------------|
| **E4M3** | Full Tensor Core | Partial Cube Unit | Full Cube Unit |
| **E5M2** | Full Tensor Core | **Limited(部分fallback FP16)** | Full Cube Unit |
| **格式组合** | All combos | E4M3×E4M3 only | All combos |
| **Accumulator** | FP32 | FP32 or FP16 | FP32 |
| **Scaling粒度** | Per-block(128) | Per-tensor/channel/group | Per-tensor/channel/group |
| **Rounding** | Stochastic | **Deterministic RNE** | Deterministic RNE |
| **Transpose** | FP8 transpose | Limited | Full FP8 transpose |

```
关键差异:
1. 910B E5M2受限 → backward pass FP8 → fallback FP16 → 训练精度风险!
2. Huawei scaling粒度更细(per-channel/group) → 权重量化更精确
3. Deterministic RNE → 可复现但可能不如stochastic精确
4. MXFP8 → 仅910D(A5)支持 → 910B/910C不支持
```

## 6. HCCL集成

```
ATB+HCCL分布式推理:

1. HCCL ops as graph nodes → OP_ALL_GATHER/OP_ALL_REDUCE → 直接嵌入ATB Graph!
2. 三种overlap策略:
   → Stream-based: HCCL在独立Ascend stream → 与计算并发
   → Pipeline: micro-batch N通信 overlap micro-batch N+1计算
   → Graph-embedded: ATB Graph SetCommOverlapMode() → AscendCL自动调度

3. 分布式模式:
   → TP(attention): HCCL AllReduce after QKV+attn+O_proj → 融合进ATB Graph
   → EP(MoE): AllGather→ExpertCompute→ReduceScatter → 融合
   → PD分离: HCCL Send/Recv + MemFabric RDMA → KV transfer
   → CP: HCCL AllGather for KV → K/V merge on feature dim → single AllGather

4. ★ HCCL缺失BF16 → 重大限制!
   → BF16 AllReduce → cast FP16 + AllReduce + cast BF16 → 精度损失!
   → vs NCCL: BF16原生支持 → 无需cast

5. DeepEP-Ascend: sgl-kernel-npu → drop-in replacement → ascend_fuseep backend
```

## 7. CUDA→ATB映射 (vLLM-Ascend + SGLang-Ascend)

| NVIDIA组件 | ATB替代 | 机制 |
|-----------|--------|------|
| FlashInfer/FA2 | `npu_fused_infer_attention_score` | AscendNPUAdaptor替换attention backend |
| cuBLAS GEMM | ATB Linear (Cube Unit) | torch.nn.Linear→ATB via adaptor |
| Triton RMSNorm | `npu_rms_norm` | RMSNorm layer自动替换 |
| vLLM RotaryEmbedding | `npu_interleave_rope` | Rotary layer自动替换 |
| PagedAttention | PagedAttention-Ascend | Block table适配Ascend memory allocator |
| NCCL AllReduce | HCCL AllReduce | torch.distributed backend→HCCL |
| vLLM sampler | ATB TopK/TopP | 融合sampling kernel |

**映射模式**: `torch.ops.npu.*`命名空间 → torch_npu Python binding → 与CUDA `torch.ops.cuda.*`类似

```
两种适配模式:
  vLLM-Ascend: AscendNPUAdaptor → intercept标准ops → dispatch到atb_ops
  SGLang-Ascend: NPU backend实现 → AscendAttnBackend/AscendLoRABackend → 注册为alternative backend

→ 两种模式都可行 → SGLang-Ascend更模块化 → vLLM-Ascend更direct
```

## 8. RTX 4090影响

```
RTX 4090: ATB/MindIE完全不适用!

  → ATB = Ascend NPU专用 → NVIDIA GPU无法使用 → 只能NCCL+cuBLAS+FlashInfer
  → HCCL = Ascend专用 → RTX 4090只能用NCCL
  → MindIE = Ascend推理栈 → RTX 4090用vLLM/SGLang

  → 但ATB设计理念可借鉴:
    → 三融合(dequant+SwiGLU+quant) → NVIDIA没有 → 但Triton可以实现类似!
    → compose级融合 → 整层Transformer → torch.compile可以做到类似!
    → Per-channel/group scaling → 更精确量化 → NVIDIA也可学!

  华为昇腾集群(910C+HCCS):
    → ATB+HCCL+MindIE → 原生优化 → 但HCCL缺BF16 → 训练受限
    → 推理: ATB INT4+FP8 → 910C全支持 → 可行
    → MoE推理: ATB MoE fusion → 比NVIDIA更融合 → 但生态不成熟

  结论: RTX 4090→vLLM/SGLang最优; Ascend 910C→MindIE+ATB原生;
  → 作为AI infra工程师→需要理解ATB设计理念→跨平台思维!
```

## 9. 关键设计洞察

```
1. Transformer专用kernel库 → 比cuBLAS/cuDNN更专注 → 更深融合
   → cuBLAS: 通用矩阵 → cuDNN: 通用NN → ATB: Transformer专用
   → 趋势: 专用kernel库 > 通用 → vLLM FlashInfer也是Transformer专用!

2. 三融合(dequant+SwiGLU+quant) → ATB独特 → NVIDIA没有
   → MoE推理的关键路径 → 省3-4次memory round-trip → 启发Triton实现!

3. compose级融合 → 整层Transformer = 1 op → 比kernel级更彻底
   → torch.compile也有类似效果 → FSDP2+compile = 融合+分布式
   → ATB compose = 静态图 → torch.compile = 动态JIT → 各有优劣

4. HCCL缺BF16 → 训练最大限制 → 推理FP16可 → 但BF16训练需要cast
   → vs NCCL: BF16原生 → NVIDIA训练优势明显

5. 910C FP8 = 全支持 → 910B = 仅E4M3 → 910D = MXFP8
   → 910B推理: E4M3 only → forward pass FP8 → backward需FP16
   → 910C: 全FP8 → 可训练FP8 → 但生态不如NVIDIA

6. torch.ops.npu.* → 昇腾op命名空间 → 与torch.ops.cuda.*对称
   → 跨平台设计: 同一Python接口 → 不同底层实现 → 桥接模式!
```

---

Sources:
- sglang/python/sglang/srt/hardware_backend/npu/ — SGLang-Ascend ATB ops (attention/quantization/lora/MoE)
- sglang/docs/platforms/ascend/ — Ascend量化+特征支持文档
- ATB Gitee Repository (openeuler/ATB) — ATB op类型定义
- CANN Documentation — Da Vinci架构+FP8+HCCL
- notebook/projects/mindie-architecture-reading.md + hccl-vs-nccl-reading.md
- vLLM-Ascend GitHub (vllm-project/vllm-ascend)
