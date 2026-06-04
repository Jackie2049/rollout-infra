# TensorRT-LLM vs vLLM vs SGLang: 推理框架深度对比

> 作者: Claude + jackie2049 | 日期: 2026-06-05
> 版本: TensorRT-LLM v1.3.0 (rc17) / vLLM v0.9.x+ / SGLang v0.4.x+
> 源码: https://github.com/NVIDIA/TensorRT-LLM (TensorRT-LLM)
> 用途: AI Infra 工程师学习笔记, 覆盖架构/性能/优化/特性/生态五个维度

---

## 目录

1. [TensorRT-LLM 架构概述](#1-tensorrt-llm-架构概述)
2. [核心优化技术](#2-核心优化技术)
3. [量化支持详解](#3-量化支持详解)
4. [并行策略](#4-并行策略)
5. [Speculative Decoding](#5-speculative-decoding)
6. [特性兼容性矩阵](#6-特性兼容性矩阵)
7. [性能对比](#7-性能对比)
8. [三框架横向对比](#8-三框架横向对比)
9. [最新版本与 2025-2026 重大更新](#9-最新版本与-2025-2026-重大更新)
10. [总结与选型建议](#10-总结与选型建议)

---

## 1. TensorRT-LLM 架构概述

### 1.1 定位与设计哲学

TensorRT-LLM 是 NVIDIA 推出的 LLM 推理优化库, 2023 年底开源, 现已发展为 **PyTorch-native** 架构:

- **核心思路**: 将 LLM 推理视为一个图优化 + kernel fusion 问题, 通过 TensorRT 编译器生成高度优化的推理引擎
- **硬件绑定**: 深度绑定 NVIDIA GPU (Ampere/Ada/Hopper/Blackwell), 充分利用每一代架构特性
- **编译期优化**: 模型在部署前经过完整的图编译 (build engine), 运行时只执行优化后的 kernel 序列
- **PyTorch-native**: 支持直接从 PyTorch 模型构建引擎, 也支持 AutoDeploy (beta) 零代码部署

### 1.2 核心架构组件

```
┌─────────────────────────────────────────────────────┐
│                  trtllm-serve (API Server)           │
│              OpenAI-compatible REST API              │
├─────────────────────────────────────────────────────┤
│              Disaggregated Serving                   │
│         (Prefill/Decode 分离, KV Cache Transfer)     │
├─────────────────────────────────────────────────────┤
│           TensorRT Engine (编译产物)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ FMHA     │ │ XQA      │ │ DeepGEMM / CuTe DSL  │ │
│  │ (Flash   │ │ (Decode  │ │ (量化 GEMM, FP4/FP8) │ │
│  │  Attn)   │ │  Attn)   │ │                      │ │
│  └──────────┘ └──────────┘ └──────────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ CUTLASS  │ │ PDL      │ │ CUDA Graph           │ │
│  │ (GEMM)   │ │ (Kernel  │ │ (Kernel Launch       │ │
│  │          │ │  Overlap)│ │  优化)               │ │
│  └──────────┘ └──────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────┤
│             In-flight Batching (IFB)                 │
│          Paged KV Cache + KV Cache Reuse             │
├─────────────────────────────────────────────────────┤
│   Parallelism: TP / PP / EP / CP / DP / Wide-EP     │
│         Helix / Attention DP / Hybrid ETP            │
├─────────────────────────────────────────────────────┤
│              KV Cache Transceiver                    │
│         (NIXL / RDMA / 分布式 KV 传输)               │
└─────────────────────────────────────────────────────┘
```

### 1.3 关键架构决策

| 设计决策 | TensorRT-LLM | vLLM V1 | SGLang |
|----------|-------------|---------|--------|
| **引擎类型** | 编译式 (AOT graph optimization) | 解释式 (JIT eager) | 解释式 (JIT eager) |
| **Kernel 策略** | 自研 CUDA/CUTLASS/CuTe | Triton + FlashInfer | FlashInfer + Triton |
| **硬件支持** | NVIDIA only | NVIDIA/AMD/TPU/Intel | NVIDIA/AMD/TPU/Intel/Ascend |
| **模型支持** | ~50 模型 (需 build engine) | 200+ 架构 (开箱即用) | ~100 模型 |
| **KV Cache** | Paged Attention + KV Reuse | V1 BlockPool + BlockHash | RadixAttention (基数树) |
| **量化** | 11 种格式 (FP4/FP8/AWQ/GPTQ) | 30+ 量化方法 | FP8/AWQ/GPTQ/INT8 |

### 1.4 两种运行模式

1. **trtllm-serve** (推荐): 高级 API server, OpenAI 兼容接口, 一键启动
   - 支持 Disaggregated Serving, Speculative Decoding, LoRA, Guided Decoding
   - YAML 配置文件驱动

2. **PyTorch Backend** (灵活): 直接在 PyTorch 中使用 TensorRT-LLM 的优化 kernel
   - 支持自定义模型, Torch Compile 集成
   - 但特性支持有限 (Speculative Decoding 仅支持 EAGLE-3)

---

## 2. 核心优化技术

### 2.1 In-flight Batching (IFB)

TensorRT-LLM 的 IFB 实现:

- **Paged KV Cache**: 类似 vLLM 的 PagedAttention, 以固定大小 block 管理 KV cache
- **KV Cache Reuse**: 跨请求复用 KV cache block (prefix caching)
- **Dynamic Batching**: 每一步动态调整 batch 组成, 新请求插入/prefill, 完成请求移除
- **Chunked Prefill**: 将长 prompt 分块处理, 避免 prefill 阻塞 decode

与 vLLM/SGLang 对比:
- vLLM V1: `Scheduler` (~2400 行) FCFS/PRIORITY 调度, `BlockPool` 双向链表 + LRU 驱逐
- SGLang: Mixin 架构 (6 个 Mixin), Merge-based 批处理, 缓存感知调度 (LPM)

### 2.2 Tensor Core 利用

TensorRT-LLM 深度优化 Tensor Core 利用率:

- **FMHA (Flash Multi-Head Attention)**: Prefill 阶段使用, Tiling + Online Softmax
- **XQA (Decode Attention)**: Decode 阶段专用 kernel, 针对 batch decode 优化
  - 官方声称: XQA 比 FMHA 在 LLaMA-70B 上带来 **2.4x 吞吐提升**
- **DeepGEMM**: 自研量化 GEMM kernel, 专为 FP4/FP8 设计
- **CuTe DSL**: 使用 CUTLASS 的 CuTe (CUDA Templates) 编写灵活的 tile 算法
- **PDL (Programmatic Dependent Launch)**: Hopper+ 架构上, kernel 间自动 overlap, 减少空闲

### 2.3 CUDA Graph

- **Piecewise CUDA Graph**: 将 decode 步骤拆分为多个子图, 允许在子图之间执行动态操作
- **Torch Compile 集成**: 支持 `torch.compile` 生成的图
- 与 vLLM 类似: vLLM V1 使用 `CudagraphDispatcher` + 5 种模式 (FULL_AND_PIECEWISE 默认)

---

## 3. 量化支持详解

### 3.1 量化格式总览

TensorRT-LLM 支持 11 种量化格式:

| 格式 | 精度 | 典型压缩比 | 适用场景 |
|------|------|-----------|---------|
| **NVFP4** | 4-bit 浮点 (Blackwell) | ~8x | 最大压缩, Blackwell 推理 |
| **MXFP4** | 4-bit 浮点 (Block scaling) | ~8x | Blackwell sm120 |
| **FP8 Per Tensor** | 8-bit 浮点 (逐张量) | ~2x | Hopper 通用推理 |
| **FP8 Block Scaling** | 8-bit 浮点 (Block 缩放) | ~2x | Hopper 高精度场景 |
| **FP8 Rowwise** | 8-bit 浮点 (逐行) | ~2x | Hopper 精细量化 |
| **FP8 KV Cache** | 8-bit KV cache | ~2x KV | 减少 KV 内存 |
| **NVFP4 KV Cache** | 4-bit KV cache | ~4x KV | Blackwell 极限 KV 压缩 |
| **W4A8 AWQ** | 4-bit 权重 8-bit 激活 | ~4x | 通用 GPU 量化 |
| **W4A16 AWQ** | 4-bit 权重 16-bit 激活 | ~4x | 通用 GPU 量化 |
| **W4A8 GPTQ** | 4-bit 权重 8-bit 激活 | ~4x | GPTQ 校准量化 |
| **W4A16 GPTQ** | 4-bit 权重 16-bit 激活 | ~4x | GPTQ 校准量化 |

### 3.2 模型 x 量化支持矩阵

| 模型 | NVFP4 | MXFP4 | FP8(per tensor) | FP8(block scaling) | FP8(rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ | W4A16 AWQ | W4A8 GPTQ | W4A16 GPTQ |
|------|-------|-------|-----------------|-------------------|-------------|-------------|----------------|----------|-----------|-----------|------------|
| DeepSeek-R1 | Y | . | . | Y | . | Y | . | . | . | . | . |
| LLaMA 4 | Y | . | Y | . | . | Y | . | . | . | . | . |
| Qwen-3 | Y | . | Y | . | . | Y | Y | . | Y | . | Y |

### 3.3 硬件 x 量化支持矩阵

| 硬件 | NVFP4 | MXFP4 | FP8(per tensor) | FP8(block scaling) | FP8(rowwise) | FP8 KV Cache | NVFP4 KV Cache | W4A8 AWQ | W4A16 AWQ | W4A8 GPTQ | W4A16 GPTQ |
|------|-------|-------|-----------------|-------------------|-------------|-------------|----------------|----------|-----------|-----------|------------|
| Blackwell (sm120) | Y | Y | Y | . | . | Y | . | . | . | . | . |
| Blackwell (sm100/103) | Y | Y | Y | Y | . | Y | Y | Y | Y | Y | Y |
| Hopper | . | . | Y | Y | Y | Y | . | Y | Y | Y | Y |

**关键洞察**: NVFP4/MXFP4 是 Blackwell 独占特性, Hopper 用户主要使用 FP8 + AWQ/GPTQ。Ampere (A100/A16) 量化支持非常有限。

### 3.4 量化工具链: ModelOpt

- **离线量化**: 使用 NVIDIA ModelOpt (`nvidia-modelopt`) 进行校准量化
- **预量化模型**: 支持 HuggingFace 上的预量化 checkpoint 直接加载
- **FP8 KV Cache**: 运行时配置, 无需离线校准

---

## 4. 并行策略

### 4.1 支持的并行策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **TP** (Tensor Parallelism) | 权重切分到多个 GPU | 单节点多 GPU |
| **PP** (Pipeline Parallelism) | 层切分, 流水线执行 | 大模型跨节点 |
| **DP** (Data Parallelism) | 完整模型副本 | 吞吐优先 |
| **EP** (Expert Parallelism) | MoE 专家分布到 GPU | MoE 模型 |
| **CP** (Context Parallelism) | 序列长度切分 | 长上下文 |
| **Wide-EP** | 专家槽与专家解耦, 热专家复制 | DeepSeek-V3/R1 级大 MoE |
| **Attention DP** | Attention 部分用 DP 替代 TP | 减少 Attention 通信 |
| **Hybrid ETP** | EP x TP 混合 | MoE 模型精细并行 |
| **Helix Parallelism** | KV cache 分片, 支持百万 token decode | 超长序列 decode |
| **DWDP** | 分布式权重数据并行 | NVL72 系统 |

### 4.2 Wide-EP 详解 (DeepSeek-R1 优化)

Wide-EP 是 TensorRT-LLM 对 DeepSeek-R1/V3 级大 MoE 模型的关键优化:

1. **专家槽解耦**: 每个 GPU 有固定数量的专家槽, 但不绑定特定专家
2. **热专家复制**: 高频调用的专家可以在多个 GPU 上复制
3. **负载均衡**:
   - **离线 EPLB** (Expert Parallelism Load Balancing): 预先分析路由分布, 安排专家布局
   - **在线 EPLB**: 运行时动态调整专家分配
4. **专用 EP 通信 Kernel**: GB200 MNNVL 架构上的定制 All-to-All kernel
5. **FP4 GEMM**: 使用 NVFP4 权重进一步减少通信量

### 4.3 Helix Parallelism

- 专为百万 token 级 decode 设计
- KV cache 在多个 GPU 间分片 (sharding)
- 解决单 GPU KV cache 容量瓶颈

### 4.4 与 vLLM/SGLang 并行对比

| 并行策略 | TensorRT-LLM | vLLM V1 | SGLang |
|----------|-------------|---------|--------|
| TP | Y | Y | Y |
| PP | Y | Y | Y |
| DP | Y | Y | Y |
| EP | Y | Y | Y |
| Wide-EP | Y (DeepSeek 专用) | Y (v0.7+) | Y |
| CP | Y | . | . |
| Attention DP | Y | . | DP-Attention |
| Helix | Y | . | . |
| Hybrid ETP | Y (EP x TP) | . | EPxTP |
| DWDP | Y (NVL72) | . | . |

---

## 5. Speculative Decoding

### 5.1 支持的方法

TensorRT-LLM 支持 7 种 speculative decoding 方法 (trtllm-serve):

| 方法 | 类型 | 特点 |
|------|------|------|
| **Draft/Target** | 经典双模型 | 小模型起草, 大模型验证 |
| **EAGLE-3 (One-Model)** | 单模型推测 | 使用自身特征, 无需额外模型 |
| **EAGLE-3 (Two-Model)** | 双模型推测 | 独立 draft 模型, 更高接受率 |
| **NGram** | Prompt Lookup | 从 prompt 中查找重复 n-gram |
| **MTP (DeepSeek)** | Multi-Token Prediction | DeepSeek 风格多 token 预测, 支持 `relaxed_acceptance_for_thinking` |
| **PARD** | 并行起草 | K 个 draft token **并行生成** (非串行) |
| **User-Provided** | 自定义 | 用户指定 draft token 来源 |

### 5.2 Suffix Automaton (SA) 增强

SA 可以与多种推测方法组合使用, 提升在重复性内容上的接受率:

- **SA + MTP**: `use_sa_spec=True` + `sa_spec_threshold`
- **SA + EAGLE-3**: 同上配置
- **SA + PARD**: 同上配置
- **SA 独立**: 可作为独立推测方法使用

### 5.3 关键配置参数

```yaml
# EAGLE-3 单模型示例
speculative_config:
  method: eagle
  max_draft_tokens: 5
  speculative_decoding_fast_logits: true

# MTP (DeepSeek) 示例
speculative_config:
  method: mtp
  max_draft_tokens: 4
  use_relaxed_acceptance_for_thinking: true
```

### 5.4 与 vLLM/SGLang 对比

| 特性 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| 经典 Draft/Target | Y | Y | Y |
| EAGLE / EAGLE-2/3 | EAGLE-3 | EAGLE-1/2 | EAGLE-2/3 |
| Medusa | . | Y | Y |
| MTP (DeepSeek) | Y (relaxed) | Y | Y |
| NGram | Y | Y | Y |
| PARD | Y | . | . |
| Suffix Automaton | Y (可组合) | . | . |
| 用户自定义 Draft | Y | Y (LogitsProcessor) | Y |
| PyTorch Backend 支持 | 仅 EAGLE-3 | 全部 | 全部 |

---

## 6. 特性兼容性矩阵

TensorRT-LLM 官方提供 20x20 特性兼容性矩阵, 关键结论:

### 6.1 兼容性总结

| 特性组合 | 兼容? | 备注 |
|---------|------|------|
| LoRA + TP/PP/EP | Y | 完全兼容 |
| LoRA + CUDA Graph | Y | 完全兼容 |
| LoRA + Helix / Attention DP | 未测试 | 官方未验证 |
| Guided Decoding + 所有特性 | Y | **全兼容** |
| MTP + PP | N | 不兼容 |
| EAGLE-3 变体之间 | N | 互不兼容 |
| MTP/EAGLE-3 + Disaggregated | 未测试 | 官方未验证 |
| Speculative Decoding + Guided Decoding | Y | 支持 |
| Speculative Decoding + LoRA | Y | 包括 FP8 LoRA |
| CUDA Graph + Chunked Prefill | Y | 完全兼容 |
| Disaggregated Serving + TP/PP/EP | Y | 完全兼容 |

### 6.2 限制

- **MTP/EAGLE-3 不支持 Pipeline Parallelism**: 如果需要 PP, 只能用经典 Draft/Target 或 NGram
- **EAGLE-3 三种变体互斥**: 一次只能选一种 (one-model / two-model / standalone)
- **PyTorch Backend 限制**: 仅支持 EAGLE-3, 其他推测方法只在 trtllm-serve 可用

---

## 7. 性能对比

### 7.1 官方基准数据 (注意局限性)

> **重要说明**: TensorRT-LLM 官方性能页面数据来自 v0.8.0 (2024 年), 已严重过时。
> 以下数据仅作参考, 当前版本 (v1.3.0) 性能应有显著提升。

**TensorRT-LLM v0.8.0 (H200, FP8)**:
| 模型 | Batch Size | 输入/输出 | 吞吐 (tok/s) |
|------|-----------|----------|-------------|
| LLaMA-7B | 1 | 128/32 | ~20,548 |
| LLaMA-70B | 1 | 128/32 | ~3,844 |

**TensorRT-LLM v1.x 官方声明 (2025-2026)**:
- Llama 4 on B200: **40,000+ tok/s**
- Llama 4 Maverick on Blackwell: **1000 TPS/user**
- XQA kernel vs FMHA: LLaMA-70B **2.4x 吞吐提升**
- DeepSeek-R1 on GB200 NVL72: FP4 + Wide-EP, 显著优于 EP-only

### 7.2 第三方对比数据

由于 WebSearch API 不可用, 无法获取最新第三方基准测试。以下基于已知信息:

| 维度 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| **单 GPU Decode** | 最快 (XQA + 编译优化) | 快 (V1 优化) | 快 (FlashInfer) |
| **多 GPU 并行效率** | 高 (NVLink 优化) | 高 | 高 |
| **MoE 大模型** | 最快 (Wide-EP + FP4) | 好 | 好 |
| **长上下文** | 好 (Helix) | 好 | 好 |
| **Speculative Decoding 加速** | 高 (EAGLE-3 + PARD) | 高 | 高 |
| **Prefix Caching** | KV Cache Reuse | BlockHash Prefix Caching | RadixAttention (最灵活) |
| **量化推理** | 最广 (11 种格式) | 广 (30+ 方法) | 中等 |

### 7.3 性能数据可靠性评估

| 数据来源 | 可靠性 | 时效性 |
|---------|-------|-------|
| TensorRT-LLM 官方 benchmark 页 | 高 | 低 (v0.8.0, 严重过时) |
| TensorRT-LLM 官方声明/README | 中 | 高 (v1.3.0) |
| vLLM 官方 benchmark | 高 | 中 |
| SGLang 官方 benchmark | 高 | 中 |
| 第三方对比 (MLPerf 等) | 高 | 需查最新 |

---

## 8. 三框架横向对比

### 8.1 吞吐与延迟

| 维度 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| **原始吞吐** | 最高 (编译优化 + 专用 kernel) | 高 | 高 |
| **延迟 (TTFT)** | 低 (编译优化) | 低 (Chunked Prefill) | 低 (Chunked Prefill) |
| **延迟 (TPOT)** | 最低 (XQA decode kernel) | 低 | 低 |
| **MoE 吞吐** | 最高 (Wide-EP + FP4 + EPLB) | 高 | 高 |
| **长上下文** | 好 (Helix + CP) | 好 | 好 |
| **Prefix Caching 效率** | 好 (KV Reuse) | 好 (BlockHash) | 最好 (RadixAttention) |

**实际差异**: 在 NVIDIA GPU 上, TensorRT-LLM 通常有 10-30% 的原始性能优势, 但这个优势正在缩小。vLLM V1 和 SGLang 的激进优化正在快速追赶。

### 8.2 特性完整性

| 特性 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| In-flight Batching | Y | Y | Y |
| Paged KV Cache | Y | Y | Y |
| Prefix Caching | Y (KV Reuse) | Y (BlockHash) | Y (RadixAttention) |
| Chunked Prefill | Y | Y | Y |
| CUDA Graph | Y (Piecewise) | Y (5 模式) | Y |
| Speculative Decoding | 7 方法 | 8+ proposer | 多种 |
| Quantization | 11 格式 | 30+ 方法 | FP8/AWQ/GPTQ |
| LoRA | Y (含 FP8 LoRA) | Y (Punica) | Y |
| Multi-Modal | Y (Vision) | Y (丰富) | Y |
| Structured Output | Y (Guided Decoding) | Y (xGrammar/Outlines) | Y (Compressed FSM) |
| Disaggregated Serving | Y | Y (P/D 分离) | Y |
| Pipeline Parallelism | Y | Y | Y |
| Expert Parallelism | Y (Wide-EP) | Y | Y |
| Continuous Batching | Y (IFB) | Y | Y |
| Multi-LoRA | Y | Y (Punica) | Y |
| RL/Post-training | . | Y (verl) | Y (主要骨干) |
| Visual Generation | Y (Diffusion) | . | . |
| AutoDeploy | Y (Beta) | . | . |
| Semantic Router | . | Y | . |
| KV Cache Offload | . | Y (CPU/SSD) | Y |
| Skip Softmax Attention | Y (长上下文) | . | . |
| Scaffolding | Y (推理时计算) | . | . |

### 8.3 易用性

| 维度 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| **入门门槛** | 高 (需 build engine) | 低 (pip install 即用) | 低 (pip install 即用) |
| **配置复杂度** | 高 (YAML + engine build) | 低 (CLI 参数) | 低 (CLI 参数) |
| **调试难度** | 高 (编译后难以调试) | 中 (Python 代码可读) | 中 (Python 代码可读) |
| **新模型支持** | 慢 (需适配 + 编译) | 快 (社区快速添加) | 快 |
| **文档质量** | 好 (NVIDIA 官方) | 好 (社区驱动) | 好 |
| **部署灵活性** | 中 (NVIDIA only) | 高 (多硬件) | 高 (多硬件) |
| **API 兼容性** | OpenAI 兼容 | OpenAI 兼容 | OpenAI 兼容 + DSL |

**TensorRT-LLM 的易用性挑战**:
1. **Engine Build 阶段**: 部署前需要编译模型, 耗时 10-60 分钟
2. **模型支持**: 新模型需要等待官方适配, 不像 vLLM/SGLang 可以快速添加
3. **硬件绑定**: 只能在 NVIDIA GPU 上运行
4. **调试困难**: 编译后的引擎难以 debug

**TensorRT-LLM 的易用性改进 (2025-2026)**:
- AutoDeploy (beta): PyTorch 模型零代码部署
- PyTorch Backend: 直接在 PyTorch 中使用优化 kernel
- trtllm-serve: 简化的一键部署命令

### 8.4 社区与生态

| 维度 | TensorRT-LLM | vLLM V1 | SGLang |
|------|-------------|---------|--------|
| **GitHub Stars** | ~12K | ~55K | ~28K |
| ** Contributors** | ~300 (NVIDIA 为主) | ~2000+ | ~500+ |
| **开源时间** | 2023 年底 | 2023 年初 | 2024 年初 |
| **背后组织** | NVIDIA (商业) | 社区 + 企业 | LMSys/UC Berkeley |
| **企业采用** | NVIDIA 生态企业 | 广泛 | xAI/Cursor/LinkedIn 等 |
| **MLPerf 参赛** | Y (NVIDIA 提交) | . | . |
| **RL 训练集成** | . | Y (verl) | Y (主要骨干) |
| **NVIDIA Dynamo** | Y (深度集成) | . | . |
| **生产部署规模** | 大 (NVIDIA 生态) | 最大 (通用) | 400,000+ GPUs |

**生态差异**:
- **TensorRT-LLM**: NVIDIA 第一方, 深度集成 NVIDIA Dynamo (数据中心级编排), 但社区较封闭
- **vLLM**: 最大最活跃的社区, 企业支持广泛 (AWS/GCP/Azure 均有托管服务), verl 集成
- **SGLang**: 学术出身, 快速增长, RL/post-training 领域的主力框架, xAI/Cursor 等头部公司采用

---

## 9. 最新版本与 2025-2026 重大更新

### 9.1 TensorRT-LLM 版本历史

| 版本 | 时间 | 关键变更 |
|------|------|---------|
| v0.x | 2023-2024 | 初始版本, IFB, 基础量化 |
| v1.0 | 2025 | PyTorch-native 架构, trtllm-serve |
| v1.1-1.2 | 2025 | Wide-EP, Helix, 更多量化格式 |
| v1.3.0 (rc17) | 2026.06 | 最新版, FP4/NVFP4 支持, EAGLE-3, PARD |

### 9.2 2025-2026 重大技术进展

**TensorRT-LLM**:
1. **Blackwell 全面支持**: NVFP4/MXFP4 量化, FP4 GEMM, NVFP4 KV Cache
2. **Wide-EP**: DeepSeek-R1/V3 级大 MoE 模型的专家并行优化
3. **EAGLE-3 + PARD**: 新一代推测解码方法
4. **DeepSeek Sparse Attention (DSA)**: 支持稀疏注意力
5. **Multi-head Latent Attention (MLA)**: DeepSeek-V2 MLA 支持
6. **Disaggregated Serving**: Prefill/Decode 分离部署
7. **LoRA + Speculative Decoding**: 支持 FP8 LoRA 与推测解码组合
8. **Guided Decoding + Speculative Decoding**: 结构化输出与推测解码同时使用
9. **Helix Parallelism**: 百万 token 级 decode 的 KV cache 分片
10. **Diffusion 模型支持**: 扩展到视觉生成
11. **AutoDeploy (Beta)**: 零代码 PyTorch 模型部署
12. **Ray Orchestrator (Prototype)**: Ray 分布式编排
13. **NVIDIA Dynamo 集成**: 数据中心级服务编排
14. **Skip Softmax Attention**: 长上下文推理优化
15. **Scaffolding**: 推理时计算框架
16. **Day-0 支持**: GPT-OSS-120B 等新模型首发支持

**vLLM (同期)**:
1. **V1 架构完成**: v0.11.0 (2025.12) 完成 V1 迁移, 全面替代 V0
2. **Wide-EP Serving**: DeepSeek 大 MoE 模型服务 (v0.7+)
3. **Semantic Router**: 系统级 Mixture-of-Models 路由
4. **DeepSeek-V3.2 on GB300**: 2026.02, GB300 平台优化
5. **MLA Backend**: FlashMLA/FlashInfer/Triton/Cutlass 四种实现
6. **UBatch (Micro-Batching)**: 通信-计算重叠, 双 CUDA 流
7. **DP Serving**: DPEngineCoreProc, Wave 协调
8. **verl 集成**: PPO/GRPO 训练, RL 训练首选

**SGLang (同期)**:
1. **RadixAttention 持续优化**: 基数树 KV cache 共享
2. **零开销 CPU 调度器**: 调度从关键路径移除
3. **多硬件支持**: NVIDIA/AMD/TPU/Intel/Ascend
4. **400,000+ GPUs 部署**: xAI/Cursor/LinkedIn 等头部采用
5. **RL/Post-training 骨干**: 成为 RL 训练领域的主要推理框架

### 9.3 TensorRT-LLM 技术博客系列

官方 2024-2026 技术博客:
1. In-Flight Batching 介绍
2. 高吞吐量 GPT-J 推理
3. GPT-J TP/PP 配置
4. BLOOM 量化部署
5. Llama-2 70B 部署
6. BLOOM 176B TP/PP
7. 多 GPU 多节点推理
8. IFB 深度解析
9. LoRA 服务
10. Multi-Query Attention 优化
11. FP8 量化推理
12. Llama-2 70B FP8
13. KV Cache 优化
14. v0.9 发布
15. v0.10/v0.11 发布
16. Llama-3.1 Int8 AWQ
17. PyTorch 工作流
18. trtllm-serve 介绍
19. Expert Parallelism Part 1-3 (MoE 优化系列)
20. Blackwell FP4 量化
21. Disaggregated Serving
22. EAGLE-3 Speculative Decoding
23. PARD Parallel Drafting

---

## 10. 总结与选型建议

### 10.1 各框架优势总结

**TensorRT-LLM 适合**:
- NVIDIA GPU 上的极致性能需求 (尤其是 Hopper/Blackwell)
- MoE 大模型 (DeepSeek-R1/V3) 的生产部署
- 需要最新量化格式 (FP4/NVFP4) 的场景
- NVIDIA 生态深度绑定 (Dynamo, MLPerf)
- 对延迟极度敏感的在线服务

**vLLM V1 适合**:
- 通用 LLM 推理, 需要快速迭代
- 多硬件环境 (AMD/Intel/TPU)
- 最广泛的模型支持 (200+ 架构)
- RL 训练集成 (verl)
- 活跃社区, 快速问题解决
- 云服务部署 (AWS/GCP/Azure 托管)

**SGLang 适合**:
- RL/post-training 场景 (主要骨干框架)
- 需要 prefix caching 的多轮对话/RL
- 结构化生成 (DSL + Compressed FSM)
- 多硬件异构部署
- 学术研究 + 快速原型

### 10.2 选型决策树

```
需要极致性能?
├── 是 → NVIDIA GPU only?
│   ├── 是 → TensorRT-LLM
│   └── 否 → vLLM 或 SGLang
└── 否 → 需要 RL 训练集成?
    ├── 是 → SGLang (主要) 或 vLLM (verl)
    ├── 否 → 需要最多模型支持?
    │   ├── 是 → vLLM
    │   └── 否 → SGLang (prefix caching) 或 vLLM (通用)
```

### 10.3 关键差异一句话总结

- **TensorRT-LLM**: 编译期优化的极致性能, 但牺牲灵活性和硬件选择
- **vLLM V1**: 最大社区和模型支持, 通用性最强的推理引擎
- **SGLang**: RadixAttention + RL 生态, 后发先至的推理框架

### 10.4 趋势观察

1. **性能差距在缩小**: vLLM V1 和 SGLang 的优化正在快速追赶 TensorRT-LLM
2. **硬件多样性在增加**: AMD/Intel/TPU 支持成为 vLLM/SGLang 的差异化优势
3. **RL 训练成为关键场景**: SGLang 在 RL/post-training 领域建立了优势
4. **MoE 模型推动并行创新**: Wide-EP/Helix 等新并行策略是前沿
5. **Blackwell 是性能新前沿**: FP4/NVFP4 让 TensorRT-LLM 在 Blackwell 上有独特优势
6. **Disaggregated Serving 成为主流**: 三个框架都在推进 P/D 分离

---

## 参考

- TensorRT-LLM GitHub: https://github.com/NVIDIA/TensorRT-LLM
- TensorRT-LLM 文档: https://nvidia.github.io/TensorRT-LLM/ (v1.3.0rc16)
- vLLM GitHub: https://github.com/vllm-project/vllm
- SGLang GitHub: https://github.com/sgl-project/sglang
- TensorRT-LLM 量化文档: https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- TensorRT-LLM Speculative Decoding: https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html
- TensorRT-LLM 并行策略: https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html
- TensorRT-LLM 特性兼容性矩阵: https://nvidia.github.io/TensorRT-LLM/features/feature-combination-matrix.html
- TensorRT-LLM 性能基准: https://nvidia.github.io/TensorRT-LLM/performance.html (v0.8.0, 过时)
