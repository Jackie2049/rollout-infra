# DeepEP-Ascend 源码级分析 (2026-06-15补充)

> ★★★★ DeepEP-Ascend = sgl-kernel-npu实现 → HCCL通信 → 3种策略 → fused_deep_moe ~70us省
> ★★★ vs NVIDIA DeepEP: HCCS(146GB/s) vs NVLink(726GB/s) → 5x带宽差但latency competitive
> ★★★ RTX 4090: DeepEP不适用(NPU) → 但理解EP机制对Megatron/vLLM MoE关键

## 1. DeepEP-Ascend 实现

```
★★★★★ DeepEP-Ascend = sgl-kernel-npu仓库 (非独立fork):

策略选择: DEEP_USE_MODE环境变量
  → default: deep_ep_cpp custom Ascend C ops + HCCL通信域
  → alltoall: torch.distributed.all_to_all_single (HCCL fallback)
  → ops: torch_npu native ops (low-latency) + custom ops (normal)

平台:
  → Atlas A2: bash build.sh -a deepep2
  → Atlas A3: bash build.sh -a deepep
  → Atlas A5: bash build.sh -a deepep Ascend950

★★★★ C++核心: deep_ep_cpp.Buffer类
  → 初始化HCCL通信域: HcclComm ep_comm via get_hccl_comm_name()
  → 方法: get_dispatch_layout, intranode_dispatch/combine, internode_dispatch/combine
  → 方法: low_latency_dispatch/combine, fused_deep_moe, dispatch_ffn_combine
  → ops/ → normal mode kernels, ops2/ → low-latency/v2 kernels
  → A5专属: cam_moe_dispatch_normal_a5.h, cam_moe_combine_normal_a5.h
```

## 2. fused_deep_moe — 核心创新

```
★★★★★ fused_deep_moe = Dispatch + Expert FFN + Combine → 单kernel调用!

融合: 2x Grouped MatMul + SwiGLU + quant/dequant → 全融合
效果: 单层通信延迟 ~70us 减少 (155us→85us)
E2E: 推理延迟 ~4ms 减少

★★★★ FuseMode:
  → FUSED_DEEP_MOE (1): full fusion → dispatch+FFN+combine → 最快!
  → DISPATCH_FFN_COMBINE (2): separate dispatch → then fused FFN+combine

量化: INT8 (quant_mode=1, default) → FP8 planned for A5

★★★★ Python API: deep_ep.Buffer类
  → 构造HCCL communicator → 初始化NormalStrategy + LowLatencyStrategy
  → ep_strategy.py → 策略注册 + 抽象基类
  → strategies/normal_strategy.py, low_latency_strategy.py
  → utils.py → EventOverlap utility
```

## 3. EP Pipeline (4步)

```
★★★★★ Ascend EP pipeline = 4步 (与NVIDIA DeepEP概念相同但ops不同):

Step 1: get_dispatch_layout(topk_idx, num_experts)
  → num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank
  → aclnn_dispatch_layout custom op

Step 2: dispatch(x, topk_idx, topk_weights, ...)
  → Intranode: aclnn_cam_moe_dispatch_normal
  → Internode: aclnn_moe_distribute_dispatch_v2 (CCU variants)
  → INT8/FP8 quant → 减少内存带宽
  → Returns: recv_x, recv_topk_idx, recv_topk_weights, handle

Step 3: Expert Computation
  → Grouped MatMul + SwiGLU + quant/dequant
  → fused_deep_moe → 全融合 → 单次调用
  → vLLM-Ascend: npu_moe_distribute_dispatch_v2 → CANN ops

Step 4: combine(x, topk_idx, topk_weights, src_idx, send_head, handle)
  → Intranode: aclnn_cam_moe_combine_normal
  → Internode: aclnn_moe_distribute_combine_v2
  → 多轮combine → 处理load imbalance
```

## 4. Low-latency Mode

```
★★★★ Low-latency mode (decode phase):

dispatch: aclnn_moe_distribute_dispatch_v2 → fixed buffer per rank
combine: aclnn_moe_distribute_combine_v2 → zero-copy option

★★★★★ ops策略 → comm_alg选项 (A3 only):
  → hierarchy → 分层通信 → 类似A2策略
  → fullmesh_v1 → 全mesh V1 → 默认
  → fullmesh_v2 → 全mesh V2 → 优化版
  → ccu → CCU tiling → topology-aware → ★★★ 最优!

★★★★ A2限制:
  → bs<=8000 (normal), bs<=512 (low-latency)
  → 必设HCCL_BUFFSIZE=1024MB
  → 禁HCCL_OP_EXPANSION_MODE
  → Internode不支持normal mode quant
  → HCCL_INTRA_PCIE_ENABLE=1, HCCL_INTRA_ROCE_ENABLE=0
```

## 5. 性能基准

```
★★★★★ DeepEP-Ascend on A3 384 SuperPOD (DeepSeek-V3 config: 4K tokens, 7168 hidden, top-8):

| EP Size | Dispatch BW | Combine BW |
|---------|-------------|------------|
| 8 | 146 GB/s | 125 GB/s |
| 16 | 107 GB/s | 103 GB/s |
| 32 | 102 GB/s | 95 GB/s |
| 64 | 81 GB/s | 91 GB/s |
| 128 | 57 GB/s | 81 GB/s |

★★★★ Low-latency (128 tokens/batch):
| EP Size | Dispatch Latency | Combine Latency |
|---------|-----------------|----------------|
| 8 | 132us / 58 GB/s | 126us / 116 GB/s |
| 16 | 139us / 55 GB/s | 135us / 109 GB/s |
| 32 | 153us / 49 GB/s | 151us / 97 GB/s |

★★★★ vs NVIDIA DeepEP (H100 NVLink):
  → NVLink EP8: 726-740 GB/s → vs HCCS EP8: 146 GB/s → ★★★ 5x差距!
  → 但: low-latency <150us → latency competitive → 生产级可用!
  → HCCS aggregate ~392 GB/s (910C 8-NPU mesh) vs NVLink4 ~900 GB/s (H100)
```

## 6. SM概念 → AIC Core映射

```
★★★★ Ascend无SM概念 → Da Vinci架构:

  AIC (AI Core) → Cube Engine (矩阵乘) + Vector Engine + Scalar Engine
  DeepEP-Ascend Buffer类: num_sms=20 → ★★★ 兼容placeholder → 不反映真实SM调度!
  Config constructor → num_sms映射为AIC core分配参数 → 通信/计算overlap

★★★★ vs NVIDIA:
  → NVIDIA: SM调度 → 4-12 SMs → analytical计算
  → Ascend: AIC core分配 → 配置参数 → 不需要SM-style thread调度
```

## 7. FlashMLA on Ascend

```
★★★★★ 无独立FlashMLA-Ascend → 但两个框架实现MLA:

A) sgl-kernel-npu MLA:
  → csrc/mla_preprocess/ → CANN custom op → 全融合MLA预处理
  → RMSNorm → Dequant → MatMul(wdqkv) → RoPE → ReshapeAndCache → ★★★ 单kernel!
  → cache_mode 0-3: combined, split krope+ctkv, NZ+INT8, NZ only
  → quant_mode: PER_TENSOR_QUANT_ASYMM, PER_TOKEN_QUANT_SYMM
  → A2/A3 support → Atlas Inference Series不支持
  → tokenNum<=1024, blockSize<=128/256

B) vllm-ascend MLA:
  → vllm_ascend/ops/mla.py → AscendMultiHeadLatentAttention
  → IndexerWrapper → DeepSeek-V3.2 fp8 routing variant
  → csrc/mla_preprocess/ → 类似融合预处理
  → csrc/attention/ → sparse_flash_attention → arch22(A2) + arch35(A3)

★★★★★ MLA预处理融合 = FlashMLA的Ascend等价 → CANN custom ops → CUDA kernels
```

## 8. vLLM-Ascend MoE Serving

```
★★★★★ vLLM-Ascend EP架构:

MoECommType:
  → ALLGATHER → 无EP → 最简
  → MC2 → npu_moe_distribute_dispatch_v2 → ★★★ 推荐
  → ALLTOALL → torch.distributed → HCCL
  → FUSED_MC2 → fused dispatch+FFN+combine → ★★★★ 最快!

TokenDispatcherWithMC2:
  → HCCL communicator via get_mc2_group()
  → enable_dispatch_v2 → A3/A5 need extra args
  → enable_mc2_hierarchy_comm → hierarchical → comm_alg parameter

★★★★ MC2 C++ kernels:
  → dispatch_ffn_combine (INT8 quantized)
  → dispatch_ffn_combine_bf16 (BF16)
  → dispatch_ffn_combine_w4_a8 (W4A8)
  → dispatch_gmm_combine_decode (decode-optimized)
  → matmul_allreduce_add_rmsnorm (fused!)
  → moe_init_routing_quant_v2 (routing+sort+quant → single kernel)

★★★★ EPLB: 5 policies → default_eplb, flashlb, random, swift_balancer, dynamic
```

## 9. NVIDIA DeepEP vs Ascend对比

```
★★★★★ 关键差异:

| 维度 | NVIDIA DeepEP V2 | DeepEP-Ascend |
|------|-----------------|---------------|
| Backend | NCCL Gin + NVSHMEM | HCCL |
| Intra-node | NVLink (~740 GB/s) | HCCS (~146 GB/s) |
| Inter-node | RDMA (IB/RoCE) | HCCS fullmesh/RDMA |
| SM调度 | Analytical 4-12 SMs | AIC core allocation |
| JIT | CUDA JIT runtime | Pre-compiled aclnn_* |
| Buffer API | ElasticBuffer (V2) | Buffer (compatible) |
| Quantization | FP8 dispatch | INT8 dispatch, FP8 planned |
| Max EP | EP2048 | EP128 tested |

★★★★ 结论: 带宽5x差距 → 但latency competitive → 生产可用 → 继续优化
```

## 10. 关键洞察

1. ★★★★★ **DeepEP-Ascend = sgl-kernel-npu** → 不是独立fork → HCCL → 3种策略
2. ★★★★ **fused_deep_moe** → Dispatch+FFN+Combine → 单层~70us省 → E2E ~4ms省
3. ★★★★ **HCCS vs NVLink 5x带宽差** → 但latency<150us → 生产可用 → competitive
4. ★★★★ **FlashMLA on Ascend** → MLA预处理全融合 → CANN custom ops → = FlashMLA等价
5. ★★★ **num_sms=20 = placeholder** → Ascend无SM → AIC core allocation → 不同调度模型
6. ★★★ **MXFP4 on A5/950** → FP8 planned → 类似RTX 5090 FP4方向 → 未来
7. ★★★★ **vLLM-Ascend MC2** → FUSED_MC2 = 最快 → INT8/W4A8 quant → decode optimized
8. ★★★ **RTX 4090: DeepEP不适用** → NPU only → 但EP理解对MoE推理关键

---

Sources:
- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu)
- [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- [Huawei HiAscend CANN](https://www.hiascend.com/document)
- ★★★ DeepEP V2 reading: `notebook/projects/deepep-v2-reading.md`
- ★★★ DeepEP Megatron integration: `notebook/projects/deepep-megatron-integration-latest.md`
- ★★★ vLLM-Ascend serving: `notebook/projects/vllm-ascend-serving-layer-reading.md`
