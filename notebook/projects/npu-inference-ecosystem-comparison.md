# NPU推理生态对比: MindIE vs vLLM-Ascend vs SGLang-Ascend (2026-06-15)

> ★★★★★ 3种NPU推理路径 → op-level vs graph-level vs hybrid → 不同trade-off
> ★★★ RTX 4090: NPU推理不适用 → 但理解生态对MindIE顾问角色关键
> ★★★★ vLLM-Ascend最灵活 → MindIE最高性能 → SGLang-Ascend用MindIE黑盒

## 1. 3种NPU推理路径

```
★★★★★ Ascend NPU推理的3种路径:

| 路径 | 框架 | 计算层 | 调度层 | 灵活性 | 性能 |
|------|------|--------|--------|--------|------|
| ★★★ vLLM-Ascend | vLLM | torch_npu+CANN | vLLM scheduler | ★★★★ 高 | ★★★ 中高 |
| ★★★★ MindIE | MindIE | ATB+CANN | MindIE scheduler | ★★ 低(黑盒) | ★★★★ 高 |
| ★★ SGLang-Ascend | SGLang | MindIE(ATB) | MindIE scheduler | ★★ 低 | ★★★★ 高 |

★★★★★ 核心差异:
  → vLLM-Ascend: op-level patch → CUDA→NPU逐op替换 → 保留vLLM调度逻辑
  → MindIE: graph-level优化 → ATB构建整Transformer计算图 → 全栈优化
  → SGLang-Ascend: 用MindIE做推理 → SGLang只做HTTP serving wrapper → 黑盒!
```

## 2. vLLM-Ascend架构详解

```
★★★★★ vLLM-Ascend = vLLM GPU版的NPU移植 → op-level替换:

5层桥接:
  Layer 1: Platform Patching → torch.accelerator→torch.npu
  Layer 2: Device Abstraction → BaseDeviceAdaptor → 910B/910C/310P/A5
  Layer 3: Op Registration → PrivateUse1 dispatch key → custom ops
  Layer 4: Model Patching → DeepSeek-V4 DSA, activation overrides
  Layer 5: Worker Adaptation → NPU lifecycle, block_table

★★★★ 优势:
  → ★★★★ 保留vLLM scheduler → continuous batching → preemption → prefix caching
  → ★★★ 灵活 → 可逐步替换 → 新模型只需新patch → 不需重建整图
  → ★★★ 与vLLM上游同步 → 版本精确匹配 → 功能parity高

★★★★ 代价:
  → ★★ op-level → 无整图优化 → 比ATB graph-level慢 → 10-30%
  → ★★★ 逐op替换 → 有些op无NPU等价 → fallback → 性能gap
  → ★★ CANN版本依赖 → torch_npu版本锁定 → 升级慢
```

## 3. MindIE架构详解

```
★★★★★ MindIE = 华为全栈推理引擎 → 黑盒 → 最高性能:

ATB计算层:
  → OP_LINEAR/OP_MATMUL → Cube Unit → FP16/BF16/INT8/INT4/FP8
  → OP_FLASH_ATTENTION/OP_PAGED_ATTENTION → ★★★ PagedAttention native!
  → OP_RMS_NORM + residual → 融合
  → OP_QUANT/OP_DEQUANT → INT8/FP8融合进计算
  → npu_dequant_swiglu_quant → ★★★★ 三融合(unique!) → NVIDIA无等价!
  → OP_ALL_GATHER/OP_ALL_REDUCE → HCCL → 通信融合进图

★★★★ 优势:
  → ★★★★ graph-level → 全栈优化 → kernel fusion → 最大吞吐
  → ★★★★ ATB Transformer专用 → 比cuBLAS/cuDNN更深的融合
  → ★★★ PD disaggregation → prefill/decode分离 → 大规模serving
  → ★★★ FP8 on 910C → 全支持 → E4M3+E5M2 → 最高性能

★★★★ 代价:
  → ★★ 黑盒 → 调度不可控 → 无法自定义batching策略
  → ★★ 模型支持依赖ATB → 新模型需等ATB更新 → 慢
  → ★★ 不开源 → 无法debug → 无法贡献 → 依赖华为
  → ★★ 只支持华为NPU → 无GPU fallback → 硬件绑定
```

## 4. SGLang-Ascend架构

```
★★★ SGLang-Ascend = SGLang HTTP wrapper + MindIE推理引擎:

  → SGLang处理HTTP serving → 请求路由 → token streaming
  → MindIE处理实际推理 → model lifecycle → KV cache → batching
  → ★★★ SGLang只是wrapper → 不控制调度 → MindIE黑盒

★★★★ 优势:
  → ★★★ MindIE最高性能 → graph-level → 全栈优化
  → ★★★ SGLang HTTP接口 → 与GPU版SGLang兼容 → 客户端无差异

★★★★ 代价:
  → ★★ 丢失SGLang调度控制 → 无法实现SGLang-specific优化
  → ★★ 黑盒 → 无法自定义 → 完全依赖MindIE版本更新
  → ★★ SGLang RadixAttention → 无用 → MindIE有自己的prefix caching
```

## 5. 模型支持对比

```
★★★★★ 模型支持矩阵 (2026-06):

| 模型 | vLLM-Ascend | MindIE | SGLang-Ascend |
|------|------------|--------|---------------|
| Qwen2/2.5 | ★★★★ Production | ★★★★ Production | ★★★(via MindIE) |
| DeepSeek-V3/R1 | ★★★★ Production | ★★★★ Production+Turbo | ★★★(via MindIE) |
| DeepSeek-V4 | ★★ Experimental(DSA) | ★★★ ATB support | ★★(via MindIE) |
| LLaMA-2/3/3.1 | ★★★ Production | ★★★ Production | ★★★(via MindIE) |
| GLM-4/5 | ★★ Experimental | ★★★ Production | ★★(via MindIE) |
| Kimi-K2 | ★★ Experimental | ★★★ Production | ★★(via MindIE) |
| MoE(Mixtral) | ★★★ MC2/EPLB | ★★★ ATB MoE | ★★★(via MindIE) |
| LoRA | ★★★ bgmv/sgmv | ★★★ ATB LoRA | ★★(limited) |

★★★★ 关键洞察:
  → ★★★★ vLLM-Ascend支持更多experimental模型 → op-level → 易扩展
  → ★★★★ MindIE对production模型性能更高 → graph-level → ATB优化更深
  → ★★★ SGLang-Ascend模型支持 = MindIE子集 → 受MindIE版本约束
```

## 6. 性能对比 (910C vs H100 vs RTX 4090)

```
★★★★★ 跨硬件性能对比:

| 场景 | 910C+MindIE | 910C+vLLM-Ascend | H100+vLLM | RTX 4090+vLLM |
|------|-------------|-----------------|-----------|---------------|
| 7B BF16 serving | ★★★★ 快(graph优化) | ★★★ 中高 | ★★★★ 快 | ★★★ 竞争 |
| 70B W8A8 serving | ★★★★ 单卡可行(64GB) | ★★★ 可行(64GB) | ★★★★ 单卡(H100 80GB) | ✗ 需3+GPU |
| DeepSeek-V3 | ★★★★ MindIE Turbo | ★★★ MC2+MLA | ★★★★ vLLM V1 | ✗ 不fit |
| MoE EP | ★★★ ATB MoE | ★★★ MC2/EPLB | ★★★★ DeepEP NVLink | ✗ 无EP |
| Prefix Caching | ★★★ ATB native | ★★★ vLLM PagedAttention | ★★★★ vLLM V1 | ★★★ vLLM |

★★★★ 关键洞察:
  → 910C MindIE > 910C vLLM-Ascend → graph vs op → 10-30%性能差
  → H100 vLLM > 910C MindIE → CUDA成熟 + NVLink带宽 → 全面领先
  → RTX 4090 → 小模型竞争 → 大模型不行 → 但CUDA生态最成熟
  → ★★★★ 中国场景: 910C MindIE = 最佳NPU推理 → 但不如H100 CUDA
```

## 7. DeepEP/MoE通信对比

```
★★★★★ MoE EP通信路径:

| 方案 | 通信库 | 带宽 | EP规模 | 特性 |
|------|--------|------|--------|------|
| NVIDIA DeepEP | NCCL+NVSHMEM | 726GB/s(NVLink) | EP2048 | ★★★★ 最高带宽 |
| DeepEP-Ascend(sgl-kernel-npu) | HCCL | 146GB/s(HCCS) | EP128 | ★★★ latency competitive |
| vLLM-Ascend MC2 | HCCL+MC2 | ~146GB/s(HCCS) | EP64+ | ★★★ fused dispatch+FFN+combine |
| MindIE ATB MoE | HCCL+ATB | ~150GB/s(HCCS) | EP128+ | ★★★★ graph-level融合 |

★★★★ 关键差异:
  → NVLink vs HCCS → 5x带宽差 → 但latency competitive → ~150us EP8
  → fused_deep_moe → ~70us单层省 → E2E ~4ms省 → 重要优化
  → MC2 FUSED_MC2 → dispatch+FFN+combine → INT8/W4A8 quant → decode optimized
  → ★★★ MindIE ATB MoE → graph融合 → 最高性能 → 但黑盒不可控
```

## 8. RTX 4090影响

```
★★★★ RTX 4090 = CUDA生态 → MindIE/vLLM-Ascend不适用:

  → MindIE: NPU only → RTX 4090完全不可用 → 只能用vLLM GPU版
  → vLLM-Ascend: NPU only → 同上
  → SGLang-Ascend: NPU only → 同上
  → ★★★ RTX 4090最优: vLLM V1 + INT4 + INT8 KV + prefix caching → CUDA

★★★★ 中国用户影响:
  → 有910C → MindIE最优推理 → 70B单卡 → 胜过RTX 4090
  → 有RTX 4090 → vLLM GPU版最优 → 小模型竞争 → CUDA成熟
  → ★★★★ 理解两者 → 作为AI infra顾问 → 需跨平台知识!
```

## 9. 关键洞察

1. ★★★★★ **3种NPU推理路径** → vLLM-Ascend(op-level) vs MindIE(graph-level) vs SGLang-Ascend(MindIE wrapper)
2. ★★★★★ **vLLM-Ascend最灵活** → 保留vLLM调度 → 可自定义 → 易扩展新模型
3. ★★★★★ **MindIE最高性能** → ATB graph-level → 全栈优化 → npu_dequant_swiglu_quant三融合(unique)
4. ★★★★ **SGLang-Ascend = MindIE wrapper** → 丢失SGLang调度控制 → 黑盒 → 受MindIE版本约束
5. ★★★★ **HCCS vs NVLink 5x带宽差** → 但latency competitive → ~150us → 生产可用
6. ★★★ **vLLM-Ascend模型支持更广** → experimental模型 → op-level易扩展 → 新模型不需重建图
7. ★★★ **MindIE模型性能更高** → production模型 → ATB优化更深 → 但新模型需等ATB更新
8. ★★★★ **fused_deep_moe** → sgl-kernel-npu创新 → Dispatch+FFN+Combine → ~70us省 → E2E ~4ms省
9. ★★★ **RTX 4090: CUDA生态** → NPU推理不适用 → 但跨平台理解对AI infra顾问关键

---

Sources:
- [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu)
- [Huawei HiAscend MindIE](https://www.hiascend.com/document)
- [vLLM Hardware Plugin Blog](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)
- ★★★ vLLM-Ascend production: `notebook/projects/vllm-ascend-production-reading.md`
- ★★★ MindIE architecture: `notebook/projects/mindie-architecture-reading.md`
- ★★★ MindIE ATB kernel: `notebook/projects/mindie-atb-kernel-architecture-reading.md`
- ★★★ DeepEP-Ascend: `notebook/projects/deepep-ascend-reading.md`
