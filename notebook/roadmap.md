# AI Infra 工程师学习路线图

> 目标：系统掌握 AI 训练与推理基础设施的核心技术，能独立设计和优化大规模分布式系统

## 阶段一：基础夯实（第 1-2 周）

### GPU 计算基础
- [x] CUDA 编程模型：线程层次、内存层次、流与事件 → `fundamentals/cuda-basics.md`
- [x] cuBLAS / cuDNN / Triton 基础使用 → `fundamentals/cuda-basics.md`
- [x] GPU 架构认知：A100/H100 关键参数、Tensor Core 原理 → `fundamentals/cuda-basics.md`
- [x] Triton 编程：kernel、autotune、GEMM、FlashAttention → `fundamentals/triton-programming.md`
- [x] torch.compile / Inductor：Dynamo 图捕获、Triton 代码生成 → `fundamentals/torch-compile.md`
- [x] 实践：GPU profiling 五类操作 (GEMM/带宽/Attention/Kernel Launch/Reduction) → `experiments/gpu-profiling-a16.md`
- [x] 实践：CUDA Graph 实验 (9.37x 加速 500 kernels) → `tools/cuda_graph_demo.py`
- [x] 实践：Flash Attention 微基准 (9.16x 加速, 249x 显存节省) → `tools/flash_attention_bench.py`
- [x] 实践：量化推理 benchmark (FP16 5.5x, INT8 cos_sim 0.993) → `tools/quantization_bench.py`
- [x] 实践：FP8 量化模拟 (FP8 15x robust vs INT8 w/ outliers) → `tools/fp8_simulation.py`
- [ ] 实践：Triton fused LayerNorm kernel benchmark (待 GPU) → `tools/triton_layernorm_bench.py`

### 分布式训练基础
- [x] 数据并行 (DP / DDP) 原理与实现 → `fundamentals/distributed-training.md`
- [x] FSDP 全分片数据并行 — FlattenParameter、Sharding 策略、Prefetching → `fundamentals/fsdp.md`
- [x] 实践：FSDP 内存模拟器（DDP vs FSDP 对比, 76% 显存节省） → `tools/fsdp_memory_sim.py`
- [x] 混合精度训练 (AMP / BF16 / FP8) → `fundamentals/mixed-precision.md`
- [x] 梯度累加、梯度裁剪 → `tools/ddp_train_demo.py` (已实践)
- [x] 实践：PyTorch DDP 训练 demo → `tools/ddp_train_demo.py`

### 通信基础
- [x] 集合通信原语：AllReduce, AllGather, ReduceScatter, Broadcast → `tools/collective_ops_viz.py`
- [x] NCCL 架构与调优基础 → `fundamentals/nccl.md`
- [x] InfiniBand / RoCE / NVLink 概念 → `fundamentals/distributed-training.md`
- [ ] 实践：用 torch.distributed 做通信 benchmark (待 GPU)

## 阶段二：核心进阶（第 3-4 周）

### 模型并行
- [x] 张量并行 (TP)：Megatron-LM 的 1D/2D/3D TP → `projects/megatron-tp-reading.md`
- [x] 流水线并行 (PP)：GPipe, 1F1B, interleaved → `projects/megatron-pp-reading.md`
- [x] 序列并行 (SP)：Ring Attention, Context Parallelism → `fundamentals/sequence-parallelism.md`
- [x] ZeRO 优化器：ZeRO-1/2/3 原理与适用场景 → `fundamentals/zero-optimizer.md`
- [x] 实践：阅读 Megatron-LM 核心代码，理解 TP 通信模式

### 显存优化
- [x] KV Cache 管理与 PagedAttention → `fundamentals/kv-cache.md`
- [x] Gradient Checkpointing / Activation Recomputation → `fundamentals/gradient-checkpointing.md`
- [x] FlashAttention 系列原理 → `fundamentals/flash-attention.md`
- [x] GPU 显存优化总览 → `fundamentals/gpu-memory-optimization.md`
- [ ] 实践：分析模型训练时的显存占用 breakdown (待 GPU)

### RL 训练 Infra
- [x] RLHF / PPO / GRPO 训练流程 → `fundamentals/rlhf-training-infra.md`
- [x] Actor-Critic 并行架构 → `projects/verl-architecture.md`
- [x] Rollout 引擎设计（vLLM 集成） → `projects/verl-architecture.md`
- [x] 实践：阅读 verl 源码，理解 rollout-worker 交互

## 阶段三：推理优化（第 5-6 周）

### 推理引擎
- [x] Continuous Batching / Iteration-level Scheduling → `fundamentals/continuous-batching.md`
- [x] Speculative Decoding → `fundamentals/speculative-decoding.md`
- [x] 量化推理 (GPTQ, AWQ, FP8) → `fundamentals/quantization.md`
- [x] FP8 量化与训练：E4M3/E5M2、Transformer Engine、vLLM 实现 → `fundamentals/fp8-quantization.md`
- [x] Disaggregated Serving：P/D 分离、KV Transfer、Mooncake/Splitwise → `fundamentals/disaggregated-serving.md`
- [x] vLLM / TensorRT-LLM / SGLang 架构对比 → `projects/vllm-architecture.md`
- [x] Tokenizer 与推理管线 → `fundamentals/tokenizer.md`
- [x] 实践：推理延迟分解 (LM Head #1 瓶颈, 层内 89-97%) → `tools/inference_latency_breakdown.py`
- [ ] 实践：部署 vLLM，benchmark 不同配置的吞吐量 (进行中)

### 模型服务
- [x] Serving 架构：API 设计、负载均衡、多副本 → `fundamentals/model-serving.md`
- [x] KV Cache 跨请求复用（prefix caching） → `projects/vllm-architecture.md` + `projects/verl-architecture.md`
- [ ] 实践：搭建简单的模型推理服务 (待 GPU)

## 阶段四：系统级能力（第 7-8 周）

### 存储与 IO
- [x] 分布式文件系统（Lustre, GPFS） → `fundamentals/distributed-fs.md`
- [x] 高效 Checkpoint：异步保存、分布式保存 → `fundamentals/checkpoint.md`
- [x] DataLoader 优化：prefetch, memory mapping → `fundamentals/dataloader.md`
- [ ] 实践：实现异步 checkpoint 保存

### 调度与编排
- [x] Kubernetes + GPU 调度基础 → `fundamentals/k8s-gpu-scheduling.md`
- [x] Slurm 集群管理 → `fundamentals/slurm.md`
- [x] Ray 分布式框架 → `fundamentals/ray-framework.md`
- [ ] 实践：用 Ray 搭建一个分布式训练任务

## 阶段五：开源贡献与实战（持续）

- [x] 选择一个核心项目深入贡献（vLLM / Megatron / verl） → 聚焦 vLLM
- [x] 阅读 issue list，找到 good first issue → `projects/vllm-contribution-plan.md`
- [ ] 提交 PR，参与 code review (PR #44439, #44444 待 DCO 修复)
- [ ] 形成个人技术博客或分享

## 资源汇总

### 必读论文
- Megatron-LM 系列 (2020-2024)
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- FlashAttention 系列
- PagedAttention (vLLM)
- DeepSpeed 系列

### 必读代码库
- PyTorch `torch.distributed`
- Megatron-LM / Megatron-Core
- vLLM
- verl

### 在线课程
- CSCS/ETH GPU Programming Course
- Andrej Karpathy 的视频教程
- ColossalAI tutorials
