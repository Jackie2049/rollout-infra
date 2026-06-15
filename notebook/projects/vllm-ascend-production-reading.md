# vLLM-Ascend 生产级深度分析 (2026-06-15补充)

> ★★★★ vLLM-Ascend = vLLM scheduling + CANN/torch_npu ops → GPU与NPU共享调度层
> ★★★★ vs SGLang-Ascend: vLLM-Ascend(op-level patch) vs SGLang-Ascend(MindIE graph-level)
> ★★★ RTX 4090: vLLM-Ascend不适用(NPU only) → 但理解对MindIE对比关键

## 1. 版本跟踪 (2025-2026)

```
★★★★★ vLLM-Ascend版本 = 上游vLLM版本精确对应:

| 版本 | 类型 | 发布日期 | 跟踪vLLM | CANN | torch_npu |
|------|------|---------|----------|------|-----------|
| v0.20.2rc1 | 最新RC | 2026-06-03 | v0.20.2 | 9.0.0 | 2.10.0 |
| v0.18.0 | 最新Stable | 2026-04-30 | v0.18.0 | 8.5.1 | 2.9.0 |
| v0.13.0 | Stable | 2026-02-05 | v0.13.0 | 8.5.1 | - |
| v0.7.3 | 首次官方 | 2025-05-08 | v0.7.3 | 8.1.RC1 | 2.5.1 |

依赖: Python 3.10-3.11, CANN 9.0.0, PyTorch 2.10.0, torch-npu 2.10.0, triton-ascend 3.2.1
硬件: Atlas 800I A2(910B), A3(910C), 300I Duo, A5(950)
```

## 2. 架构: 5层桥接机制

```
★★★★★ vLLM-Ascend 5层桥接 = GPU→NPU适配:

Layer 1: Platform Patching (vllm_ascend/patch/platform/)
  → Monkey-patch torch.accelerator → torch.npu
  → patch_torch_accelerator.py: memory_stats, empty_cache → NPU
  → patch_distributed.py: NCCL → HCCL
  → patch_kv_cache_interface.py: KV cache管理 → Ascend
  → patch_mla_prefill_backend.py: MLA → Ascend MLA

Layer 2: Device Abstraction (vllm_ascend/device/device_op.py)
  → BaseDeviceAdaptor: reshape_and_cache, moe_init_routing, moe_gating_top_k
  → AscendDeviceType: 910B/A2, 910C/A3, 310P, A5/950
  → MXFP兼容层: mxfp_compat.py → float8_e8m0fnu, float4_e2m1fn_x2, hifloat8

Layer 3: Op Registration (direct_register_custom_op, dispatch_key="PrivateUse1")
  → unquantized_gemm, rope_forward_oot, fused MoE ops
  → ★★★ PrivateUse1 = PyTorch的NPU dispatch key → 正确路由!

Layer 4: Model Patching (vllm_ascend/models/)
  → DeepSeek-V4: DSA attention, IndexerWrapper
  → Activation overrides: npu_swiglu, npu_fast_gelu
  → Per-model attention classes

Layer 5: Worker Adaptation (vllm_ascend/worker/)
  → worker.py, model_runner_v1.py → NPU生命周期
  → block_table.py → Ascend KV cache块管理
  → pcp_utils.py → Prefill Context Parallel

★★★★★ vs SGLang-Ascend:
  → vLLM-Ascend: op-level patch → 保留vLLM调度 → 逐步替换CUDA→NPU
  → SGLang-Ascend: MindIE graph-level → 整图优化 → 但丢失SGLang调度控制!
  → ★★★ vLLM-Ascend更灵活 → SGLang-Ascend更黑盒
```

## 3. 支持模型

```
★★★★ Production级: Qwen2/2.5/2.5-VL, DeepSeek-V2/V2.5/V3/V3.1/V3.2/R1, LLaMA-2/3/3.1, Mixtral
★★★ Experimental: Qwen3.x, Qwen3-Next, DeepSeek-V4, GLM-4/4.7/5, Kimi-K2, MiniMax-M2.5, InternVL

★★★★★ 关键模型特性:
  → DeepSeek-V3.1/V3.2: W8A8 + C8(INT8 KV) + PD disaggregation → ★★★ Production!
  → DeepSeek-V4: DSA attention + custom ops + KV pool + MTP → Experimental
  → Qwen3.5 MoE: Flash Comm V1, W8A8 dynamic linear on 310P
  → GLM-5: W4A8, MTP3, FlashComm → known issues with PD
  → Kimi-K2: Eagle3 speculative decoding, FlashComm
```

## 4. KV Cache + Prefix Caching

```
★★★★★ Ascend KV Cache ≠ CUDA PagedAttention:

写入: torch_npu._npu_reshape_and_cache → 替代reshape_and_cache
读取: _npu_flash_attention_unpad → 替代npu_fusion_attention → A2/A3更好性能
块管理: block_table.py → logical→physical → Ascend内存

★★★★ Prefix Caching:
  → --enable-prefix-caching → 支持
  → Mamba prefill prefix caching (layerwise connector) → v0.18.0+
  → CANN 8.1.RC1+ 必需 → 自动prefix caching

★★★★★ Ascend Attention Backend多样性:
  → attention_v1.py: Standard flash attention
  → dsa_v1.py: DeepSeek Sparse Attention
  → mla_v1.py: Multi-head Latent Attention
  → sfa_v1.py: Sparse Flash Attention
  → fa3_v1.py: Flash Attention 3 (awaiting FA3 package)
  → context_parallel/: PCP/DCP
```

## 5. LoRA Serving on Ascend

```
★★★★ LoRA supported since v0.7.3 (#700, China Merchants Bank贡献):

punica_npu.py: PunicaWrapperNPU → Multi-LoRA状态管理
  → 310P or max_lora_rank>=128 → PyTorch-native bgmv/sgmv (slower)
  → Otherwise → Ascend-specific _C_ascend bgmv/sgmv (faster)
  → ★★★ bgmv_shrink, bgmv_expand, sgmv_shrink, sgmv_expand → C++ custom ops

★★★★ 限制:
  → 310P必须PyTorch-native → 较慢
  → 高LoRA rank(>=128) → fallback → 更慢
  → Multi-LoRA动态 → "will be improved in next release"
  → Qwen3.5 dense LoRA → v0.20.2rc1+ (#9023)
```

## 6. MoE + Expert Parallelism

```
★★★★★ vLLM-Ascend MoE通信方式 (MoECommType):

ALLGATHER → 无EP → 全部token allgather → 最简
MC2 → MindSpore Communication 2 → npu_moe_distribute_dispatch_v2 → ★★★ 推荐
ALLTOALL → torch.distributed all_to_all_single → HCCL
FUSED_MC2 → fused dispatch+FFN+combine → ★★★★ 最快!

★★★★ EPLB (Expert Parallel Load Balancing):
  → policy_default_eplb, policy_flashlb, policy_random, policy_swift_balancer
  → Dynamic EPLB: _DYNAMIC_EPLB group → runtime expert redistribution
  → ★★★ 不能同时支持minimax_m2 + W4A8 (#8341)

★★★★ MC2 C++ kernel (csrc/mc2/dispatch_ffn_combine/):
  → dispatch_ffn_combine (INT8 quantized GMM + SwiGLU)
  → dispatch_ffn_combine_bf16 (BF16 variant)
  → dispatch_ffn_combine_w4_a8 (W4A8 per-channel)
  → dispatch_gmm_combine_decode (decode-specific)
  → matmul_allreduce_add_rmsnorm (fused matmul+AR+norm)
  → moe_init_routing_quant_v2 (routing+sort+quant → single kernel)
  → HCCL shmem + topology info → NPU-aware communication
```

## 7. FP8/MXFP量化 (910C vs RTX 4090)

```
★★★★★ Ascend vs NVIDIA量化对比:

| 维度 | Ascend 910C (W8A8) | RTX 4090 (INT4) |
|------|---------------------|----------------|
| 内存压缩 | ~2x (W8A8) | ~4x (INT4/GPTQ) |
| 精度保持 | ★★★ FP8保持动态范围 | ★★ INT4更多退化 |
| KV Cache | C8(INT8 KV) → 2x省内存 | INT8 KV → 2x省内存 |
| 计算吞吐 | npu_quant_matmul | Marlin INT4 kernel |
| 量化管线 | CANN校准+AscendFp8Config | GPTQ/AWQ/GGUF |
| MXFP | MXFP4/MXFP8 on A5/950 | ✗ N/A (SM89) |
| 软件成熟度 | CANN 9.0(发展中) | CUDA(成熟) |

★★★★ MXFP4 on A5/950:
  → float4_e2m1fn_x2 + float8_e8m0fnu scale → ★★★ 类似RTX 5090 FP4!
  → torch_npu 2.10.0 尚未原生暴露MXFP dtype → mxfp_compat.py shim
  → FP8 quantization planned for A5 release → 未来!
```

## 8. HCCL分布式

```
★★★★★ HCCL ProcessGroup结构 (parallel_state.py):

标准: TP, DP, EP → 与vLLM GPU版类似
特殊:
  → _MC2: MC2 dispatch/combine专用组
  → _MLP_TP, _OTP, _LMTP, _EMBED_TP: 模块特定TP组
  → _FLASHCOMM2_OTP/ODP: FlashComm2优化AR序列
  → _SHARD_WEIGHT: 权重分片跨rank
  → _P_TP: Prefill TP (PD disaggregation)
  → _DYNAMIC_EPLB: 动态EP负载均衡

★★★★ HCCL优化:
  → ProcessGroup reuse (#7654) → 减少初始化开销
  → CPU binding → ARM-only A3 policy (#6686)
  → --tensor-parallel-size N --device ascend → 启动命令
```

## 9. 910B vs 910C vs 950 性能对比

```
★★★★★ Ascend NPU推理性能 (vs RTX 4090):

| 维度 | 910B (64GB) | 910C (64GB+) | RTX 4090 (24GB) |
|------|-------------|-------------|----------------|
| FP16 | ~280 TFLOPS | ~320 TFLOPS | ~330 TFLOPS |
| 内存 | 64GB HBM2e | 64GB+ HBM | 24GB GDDR6X |
| 带宽 | ~1.2 TB/s | ~1.5 TB/s | ~1.0 TB/s |
| 小模型 | 竞争但较慢(SW不成熟) | 竞争 | ★★★ 更快 |
| 大模型 | ★★★ 单卡可行! | ★★★★ 单卡舒适! | ✗ 需多GPU |
| 批量serving | ★★★ 更大batch | ★★★★ 更大batch | ★★ 受限 |

★★★★ 关键洞察:
  → 70B+模型 → 910C单卡可行 → RTX 4090需3+GPU → ★★★ 质变!
  → 小模型 → RTX 4090 CUDA成熟 → 更快 → 但Ascend正在追赶
  → MindIE Turbo → DeepSeek-V3/R1/Qwen-2 → kernel级优化 → vLLM-Ascend集成!
```

## 10. 关键洞察

1. ★★★★★ **vLLM-Ascend = op-level patch** → 保留vLLM调度逻辑 → vs SGLang-Ascend(MindIE graph-level) → 更灵活
2. ★★★★★ **5层桥接** → Platform→Device→Op→Model→Worker → CUDA→NPU系统化替换
3. ★★★★ **MXFP4 on A5/950** → 类似RTX 5090 FP4 → float4_e2m1fn_x2 → 未来量化方向
4. ★★★★ **MC2 + EPLB** → MoE EP完整支持 → DeepSeek-V3/V3.2 Production级
5. ★★★★ **910C 64GB** → 70B+单卡推理 → vs RTX 4090 24GB质变 → 但SW不成熟
6. ★★★ **LoRA on Ascend** → bgmv/sgmv custom ops → 310P fallback → 有限制
7. ★★★★ **C8 INT8 KV** → DeepSeek-V3.1+ → 2x KV省内存 → 与RTX 4090 INT8 KV等价
8. ★★★ **RTX 4090: vLLM-Ascend不适用** → NPU only → 但理解对MindIE对比关键

---

Sources:
- [vllm-project/vllm-ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vllm-ascend v0.20.2rc1 Release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.20.2rc1)
- [vllm-ascend v0.18.0 Release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)
- [vLLM Hardware Plugin Blog](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)
- [ascend-sglang](https://github.com/qinsir5522/ascend-sglang)
- ★★★ vLLM-Ascend serving layer: `notebook/projects/vllm-ascend-serving-layer-reading.md`
- ★★★ MindIE architecture: `notebook/projects/mindie-architecture-reading.md`
