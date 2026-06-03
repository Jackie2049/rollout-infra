# AI Infra 工程师知识地图

> 核心技能树与学习资源索引

## 1. GPU 计算与 CUDA

### 核心概念
- CUDA 编程模型：线程层次 (grid/block/thread)、内存层次 (global/shared/local)
- GPU 架构：A100/H100/B200 关键参数、Tensor Core、NVLink
- Kernel 优化：occupancy、coalesced access、shared memory bank conflict
- OpenAI Triton：Python-like GPU 编程，硬件无关 kernel 编写

### 关键资源
- [NVIDIA CUDA 入门指南](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [Awesome GPU Engineering](https://gpuengineering.com/) — 结构化学习路线
- [Triton 官方文档](https://openai.com/index/triton/)
- NVIDIA GTC 2025: Blackwell Programming with Triton

### 实践目标
- [ ] 编写 vector add / matrix mul CUDA kernel
- [ ] 用 Triton 写 FlashAttention 简化版
- [ ] 理解并 benchmark bandwidth vs compute bound

---

## 2. 分布式训练

### 2.1 数据并行
- DP → DDP → FSDP (ZeRO Stage 3) 演进
- 梯度同步：AllReduce (Ring, NCCL)
- 混合精度：AMP / BF16 / FP8

### 2.2 张量并行 (TP)
- Megatron-LM 1D TP：列并行 + 行并行
- 通信：AllReduce → AllGather + ReduceScatter
- 2D/2.5D/3D TP 概念

### 2.3 流水线并行 (PP)
- GPipe：所有 micro-batch 前向后向
- 1F1B：交替前向后向，减少显存
- Interleaved 1F1B：virtual stages 减少气泡

### 2.4 序列并行 (SP) / 上下文并行 (CP)
- DeepSpeed Ulysses：沿 sequence 维度切分 attention head
- Ring Attention：环形通信实现无限长序列
- Context Parallelism (Megatron)：prefill 阶段序列切分
- USP：统一序列并行框架

### 关键论文
- Megatron-LM 系列 (2020-2024)
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- PipeDream, GPipe
- Ring Attention with Blockwise Parallel FlashAttention

### 关键项目
| 项目 | 维护方 | 说明 |
|------|--------|------|
| [Megatron-LM](https://github.com/nvidia/megatron-lm) | NVIDIA | TP+PP+SP+CP，v0.11.0 支持多数据中心 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO 优化器、offloading |
| [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) | Meta/PyTorch | 原生 PyTorch 分片并行 |
| [Megatron-FSDP](https://docs.nvidia.com/megatron-core/) | NVIDIA | Megatron 并行 + FSDP 融合 |

---

## 3. 显存管理与优化

### 核心概念
- 模型参数 / 梯度 / 优化器状态 / 激活的显存分解
- Gradient Checkpointing / Activation Recomputation
- ZeRO Stage 1/2/3 分片策略
- FlashAttention 1/2/3：IO-aware tiling，减少 HBM 访问

### KV Cache（推理侧）
- PagedAttention：虚拟内存分页管理 KV Cache
- 传统系统浪费 60-80%，vLLM 降至 < 4%
- Prefix Caching：跨请求复用公共前缀 KV

### 关键论文
- FlashAttention 1/2/3
- PagedAttention (vLLM, SOSP 2023)
- ZeRO (SC 2020)

---

## 4. 集合通信

### 核心概念
- 集合通信原语：AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv
- Ring AllReduce 算法原理
- NCCL 架构：channel, proxy, ring/tree 算法选择
- 硬件互联：NVLink (900GB/s), InfiniBand (400Gb/s), RoCE, PCIe

### 2025 新进展
- Meta 开源 NCCLX/CTran：面向 100K+ GPU 规模
- AutoCCL：自动化通信调优 (NSDI 2025)

### 关键资源
- [NCCL 源码](https://github.com/NVIDIA/nccl)
- [NCCL 开发者页面](https://developer.nvidia.com/nccl)

---

## 5. 模型推理与服务

### 核心概念
- Continuous Batching (iteration-level scheduling)
- Speculative Decoding (draft model + verify)
- 量化推理：GPTQ, AWQ, FP8
- Prefix Caching / RadixAttention

### 推理引擎对比

| 引擎 | 特点 | 适合场景 |
|------|------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention, 生态成熟 | 通用推理、生产部署 |
| [SGLang](https://github.com/sgl-project/sglang) | 最高吞吐量, 400K+ GPU | 高性能生产环境 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 极致优化 | NVIDIA GPU 最佳性能 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ 纯实现, GGUF 量化 | 本地推理、边缘部署 |

### 关键资源
- vLLM Architecture Overview: https://docs.vllm.ai/
- SGLang Learning Materials: https://github.com/sgl-project/sgl-learning-materials
- Mini-SGLang (教学版): https://github.com/sgl-project/mini-sglang

---

## 6. RL 训练 Infra

### 核心概念
- RLHF / PPO / GRPO / RLOO 训练流程
- Actor-Critic 架构与并行策略
- Rollout 引擎：vLLM 集成、generate → train pipeline
- Reward Model / Verifiable Reward

### 关键项目

| 项目 | 特点 |
|------|------|
| [verl (HybridFlow)](https://github.com/verl-project/verl) | 字节跳动开源, Ray+vLLM+DeepSpeed+Megatron |
| [OpenRLHF](https://github.com/openrlhf/openrlhf) | Ray+vLLM+DeepSpeed, PPO/GRPO/REINFORCE++ |
| [TRL](https://github.com/huggingface/trl) | HuggingFace, SFT/DPO/GRPO, 易用 |

---

## 7. 存储与 IO

### 核心概念
- Checkpoint 策略：同步 vs 异步保存、分布式保存
- 本地优先 → 异步上传分布式存储
- DataLoader 优化：prefetch, memory mapping, DALI
- 分布式文件系统：Lustre, GPFS, 对象存储

### 关键资源
- [ByteCheckpoint](https://arxiv.org/html/2407.20143v4) — 统一 checkpoint 系统
- [MLPerf Storage v2.0](https://mlcommons.org/2025/08/storage-2-checkpointing/)

---

## 8. 调度与编排

### 核心概念
- Kubernetes + GPU 调度（MIG, time-sharing）
- Slurm 集群管理
- Ray 分布式框架（actor model, placement group）
- 作业排队、优先级、抢占

---

## 优先级排序

基于学习价值和当前项目需求（prefix-sharing / RL训练）：

1. **分布式训练** (TP/PP/SP/CP) — 最核心，直接相关
2. **显存管理** — 紧密关联分布式训练
3. **RL 训练 Infra** — 当前项目方向
4. **推理引擎** — 了解即可，后续深入
5. **集合通信** — 与分布式训练同步学习
6. **GPU 计算** — 长期深入，先掌握概念
7. **存储/调度** — 按需学习
