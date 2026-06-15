# NPU Inference Ecosystem: 3 Paths to Ascend Serving (2026-06-15)

> ★★★ 3种Ascend推理路径: vLLM-Ascend(op-level,灵活★★★★) / MindIE(graph-level,最快★★★) / SGLang-Ascend(MindIE wrapper★★)
> ★★★ RTX 4090: 全部不适用(NPU only) → 但理解对中国AI infra战略关键

## 1. 3种路径架构对比

```
★★★★★ Ascend NPU推理生态系统3条路径:

Path 1: vLLM-Ascend (★★★★ 推荐 — 最灵活)
  → 继承vLLM scheduling+batching+KV cache管理 → op-level CUDA→NPU替换
  → 5层桥接: Platform→Device→Op→Model→Worker → 系统化替换
  → ★★★★ 保留vLLM调度逻辑 → 可控 → 生产运维友好
  → ★★★ 自定义模型配置 → per-model patch → 灵活
  → ★★★ LoRA支持(bgmv/sgmv Ascend custom ops) → Multi-LoRA动态
  → ✗ 性能不如MindIE → op-level vs graph-level → 单op优化不如整图

Path 2: MindIE (★★★ 性能最高 — 黑盒)
  → 华为官方推理引擎 → ATB graph-level → 整Transformer=1 op → 最深融合
  → ★★★ npu_dequant_swiglu_quant → 三融合 → NVIDIA无等价!
  → ★★★ MindIE Turbo → DeepSeek-V3/R1/Qwen-2 → 额外kernel级优化
  → ✗ 核心不开源 → 黑盒 → 无法自定义 → 无法debug → 调度不可控
  → ✗ 版本绑定CANN → 升级困难 → 与vLLM生态脱节

Path 3: SGLang-Ascend (★★ 中间 — MindIE wrapper)
  → SGLang HTTP → MindIE internal → SGLang只做HTTP serving wrapper
  → ★★★ 零代码改推理 → MindIE性能直接获得 → 最快部署
  → ✗ 丢失SGLang调度控制 → 无法RadixAttention → prefix caching依赖MindIE
  → ✗ 与SGLang GPU版功能差距大 → prefix scheduling不同
  → ✗ ★★★ SGLang优势(RadixAttention+overlap)→全部丢失→用MindIE替代→不是真SGLang!

★★★★★ 关键选择:
  → 需要灵活+可控 → vLLM-Ascend → ★★★★
  → 需要最高性能+不想自定义 → MindIE → ★★★
  → 需要快速部署+不关心调度 → SGLang-Ascend → ★★
  → ★★★★ GRPO rollout → prefix caching关键 → vLLM-Ascend更可控!
```

## 2. MoE EP通信对比

```
★★★★★ Ascend MoE通信架构:

vLLM-Ascend:
  → ALLGATHER(无EP) / MC2(推荐) / ALLTOALL(HCCL) / FUSED_MC2(最快)
  → MC2: torch_npu.npu_moe_distribute_dispatch_v2 → CANN ops
  → EPLB: 5 policies → expert load balancing
  → ★★★ FUSED_MC2: dispatch+FFN+combine → INT8/W4A8 → decode optimized

DeepEP-Ascend (sgl-kernel-npu):
  → 3种策略: default(alltoall HCCL) / ops(torch_npu native) / alltoall(fallback)
  → ★★★★ fused_deep_moe → 单层~70us省 → E2E ~4ms省 → ★★★ 性能优势
  → INT8 dispatch → FP8 planned for A5/950
  → HCCS EP8: 146 GB/s → NVLink EP8: 726 GB/s → 5x差距
  → 但latency<150us → competitive → 生产可用

★★★★★ 对比:
  → vLLM-Ascend MC2 → 通用 → 适合所有MoE模型
  → DeepEP-Ascend → 专用 → 针对DeepSeek MoE → 更快但限制多
  → ★★★ vLLM-Ascend + MC2 = 生产首选 → 广泛模型支持
  → ★★★ DeepEP-Ascend + fused_deep_moe = DeepSeek专用最优
```

## 3. FP8/MXFP量化路径对比

```
★★★★★ Ascend量化 vs NVIDIA量化:

| 维度 | Ascend 910C | Ascend A5/950 | RTX 4090 | RTX 5090 SM120 |
|------|-------------|---------------|----------|----------------|
| FP8 | ✓ npu_quant_matmul | ✓ | ✗ E5M2 crash | ✗ 不适用(消费级不同) |
| MXFP8 | ✓ float8_e8m0fnu | ✓ | ✗ | ✗ |
| MXFP4 | ✓ float4_e2m1fn_x2 | ★★★ NEW | ✗ | ★★★★ native FP4 |
| INT4 | ✗(Ascend无INT4) | ✗ | ★★★ GPTQ Marlin | ★★★→FP4替代 |
| INT8 KV | ✓ C8 | ✓ | ✓ (唯一可行) | ✓ FP8 KV(SM120) |

★★★★ 关键差异:
  → Ascend: FP8/W8A8为主 → MXFP4 on A5 → 无INT4 GPTQ → 不同量化路径!
  → NVIDIA: INT4(GPTQ/AWQ)为主 → FP8(SM90+) → FP4(SM120) → 不同路径!
  → ★★★ Ascend量化更"浮点化" → FP8+MXFP4 → 保持动态范围
  → ★★★ NVIDIA量化更"整数化" → INT4+INT8 → 但精度损失更多
  → ★★★★ MXFP4 = 未来统一方向 → Ascend A5 + RTX 5090 → 都支持!
```

## 4. 生产部署决策树

```
★★★★★ Ascend推理生产决策树:

Q1: 模型是DeepSeek系列?
  → YES → Q2
  → NO → vLLM-Ascend + MC2 (通用最佳)

Q2: 需要最高吞吐?
  → YES → MindIE Turbo (DeepSeek专用最优)
  → NO → vLLM-Ascend (灵活+可控)

Q3: 需要自定义模型/LoRA?
  → YES → vLLM-Ascend (LoRA支持+per-model patch)
  → NO → MindIE (黑盒但更快)

Q4: 需要GRPO rollout + prefix caching?
  → YES → vLLM-Ascend (可控prefix caching + enable_prefix_caching)
  → NO → MindIE/SGLang-Ascend

★★★★★ 结论:
  → ★★★★ vLLM-Ascend = 通用推荐 → 最灵活+可控+LoRA+prefix caching
  → ★★★ MindIE = DeepSeek专用最优 → 黑盒但性能最高
  → ★★ SGLang-Ascend = 快速部署 → 但丢失核心优势 → 不推荐!
```

## 5. 关键洞察

1. ★★★★★ **3条路径各有定位** → vLLM-Ascend(灵活★★★★) / MindIE(性能★★★) / SGLang-Ascend(wrapper★★)
2. ★★★★ **vLLM-Ascend推荐** → 保留vLLM调度 → 可控 → 生产运维友好 → prefix caching可控
3. ★★★★ **SGLang-Ascend不推荐** → 丢失RadixAttention+overlap → 不是真SGLang → 用MindIE代替
4. ★★★★ **MXFP4 = 未来统一** → Ascend A5 + RTX 5090 → 都支持 → 不同硬件相同量化方向
5. ★★★ **DeepEP-Ascend vs MC2** → DeepEP(专用,fused) vs MC2(通用,广泛) → 不同MoE策略
6. ★★★ **RTX 4090: 全部不适用** → NPU only → 但理解对中国AI infra战略关键
7. ★★★★ **量化路径差异** → Ascend FP8→MXFP4(浮点) vs NVIDIA INT4→FP4(整数→浮点) → 都走向MXFP4

---

Sources:
- [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu)
- [Huawei MindIE](https://www.hiascend.com/document)
- ★★★ vLLM-Ascend production: `notebook/projects/vllm-ascend-production-reading.md`
- ★★★ DeepEP-Ascend: `notebook/projects/deepep-ascend-reading.md`
- ★★★ MindIE architecture: `notebook/projects/mindie-architecture-reading.md`
- ★★★ MindIE ATB kernel: `notebook/projects/mindie-atb-kernel-architecture-reading.md`
