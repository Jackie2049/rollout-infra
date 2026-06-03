# rollout-infra — AI Infra 工程师成长之路

> 从零开始，系统学习 AI 基础设施，成为一名强悍的 AI Infra 工程师

## 仓库运作规则

## 规划与路线

- [学习路线图](notebook/roadmap.md) — Phase 1-5 学习计划，从分布式训练到 vLLM 开源贡献
- [知识地图](notebook/knowledge-map.md) — AI Infra 8 大领域技能树，标注掌握程度
- [GPU 实验计划](notebook/gpu-experiment-plan.md) — A16 GPU 上的实验清单和优先级

## 分布式训练

- [分布式训练总览](notebook/fundamentals/distributed-training.md) — DP/TP/PP/SP 演进逻辑，通信原语图解
- [ZeRO 优化器](notebook/fundamentals/zero-optimizer.md) — ZeRO 1/2/3 原理、显存公式、适用场景
- [混合精度训练](notebook/fundamentals/mixed-precision.md) — FP16/BF16/FP8、Loss Scaling、Tensor Core 利用
- [梯度检查点](notebook/fundamentals/gradient-checkpointing.md) — 激活重计算策略，用计算换显存
- [NCCL 通信库](notebook/fundamentals/nccl.md) — Ring/Tree AllReduce、调优参数、网络拓扑
- [通信与计算重叠](notebook/fundamentals/comm-compute-overlap.md) — DDP Bucket Overlap、1F1B、ZeRO-3 Prefetch
- [序列并行](notebook/fundamentals/sequence-parallelism.md) — DeepSpeed Ulysses、Ring Attention、USP 统一框架
- [Ray 分布式框架](notebook/fundamentals/ray-framework.md) — Actor/Task 模型、Placement Group、RLHF 调度
- [分布式 Checkpoint](notebook/fundamentals/checkpoint.md) — 分片保存、异步 IO、mmap 加速加载
- [DataLoader 优化](notebook/fundamentals/dataloader.md) — Prefetch、mmap、WebDataset 流式加载

## GPU 编程与优化

- [CUDA 编程基础](notebook/fundamentals/cuda-basics.md) — 线程层次、内存模型、流、Tensor Core
- [Triton 编程](notebook/fundamentals/triton-programming.md) — OpenAI Triton：kernel、autotune、GEMM、FlashAttention
- [FlashAttention](notebook/fundamentals/flash-attention.md) — 1/2/3 深度解析，tiling、online softmax、IO 复杂度
- [RoPE 旋转位置编码](notebook/fundamentals/rope.md) — 数学推导、四种实现、五种长度扩展方法、推理优化
- [KV Cache 深度解析](notebook/fundamentals/kv-cache.md) — 显存估算、PagedAttention、GQA/MQA、Prefix Caching、量化
- [GPU 显存优化](notebook/fundamentals/gpu-memory-optimization.md) — 训练显存估算公式、OOM 排查、ZeRO、碎片化
- [CUDA Graphs](notebook/fundamentals/cuda-graphs.md) — 推理场景 kernel launch 开销优化，录制重放机制
- [GPU 性能分析](notebook/fundamentals/gpu-profiling.md) — 四层方法：nvidia-smi → nsys → ncu → ncu detail

## RL 训练基础设施

- [RLHF/GRPO 训练基础设施](notebook/fundamentals/rlhf-training-infra.md) — PPO 四模型架构、GRPO 无 Critic、verl/OpenRLHF 框架、Prefix Sharing

## 推理与部署

- [Speculative Decoding](notebook/fundamentals/speculative-decoding.md) — Draft-Verify、Medusa、Eagle 投机采样
- [Continuous Batching](notebook/fundamentals/continuous-batching.md) — Static→Continuous 演进、Chunked Prefill、调度策略
- [推理量化](notebook/fundamentals/quantization.md) — GPTQ/AWQ/SmoothQuant/FP8 量化方法对比
- [模型服务架构](notebook/fundamentals/model-serving.md) — API 网关、负载均衡、HPA 自动扩缩容

## 集群与基础设施

- [K8s GPU 调度](notebook/fundamentals/k8s-gpu-scheduling.md) — Device Plugin、MIG、Gang Scheduling
- [Slurm 集群管理](notebook/fundamentals/slurm.md) — sbatch 提交、NCCL 配置、抢占式调度
- [分布式文件系统](notebook/fundamentals/distributed-fs.md) — Lustre 条带化、IO 优化、并行读写

## 开源项目源码阅读

- [vLLM V1 架构](notebook/projects/vllm-architecture.md) — V1 整体架构、Scheduler/Executor 分离、混合推理
- [vLLM Prefix Caching V1](notebook/projects/vllm-prefix-caching-v1.md) — Hash chain、block 生命周期、四种 attention 缓存策略
- [vLLM 开源贡献计划](notebook/projects/vllm-contribution-plan.md) — 11 个 good first issue 筛选和分析
- [vLLM 贡献准备](notebook/projects/vllm-contribution-prep.md) — FlashInfer 后端分析、开发环境搭建
- [Megatron-LM 张量并行](notebook/projects/megatron-tp-reading.md) — ColumnParallelLinear/RowParallelLinear 源码
- [Megatron-LM 流水线并行](notebook/projects/megatron-pp-reading.md) — 1F1B 调度、气泡分析、通信 overlap
- [verl 架构](notebook/projects/verl-architecture.md) — 三级前缀缓存（系统级+进程级+请求级）
- [verl PrefixGrouper](notebook/projects/verl-prefix-grouper.md) — RL 训练中 prompt 复用的分组与调度
- [分布式训练排错指南](notebook/projects/troubleshooting-guide.md) — NCCL 超时、OOM、梯度异常等常见问题

## GPU 实验记录

- [A16 基准测试](notebook/experiments/gpu-a16-benchmark.md) — GEMM 14.64 TFLOPS、HBM 76 GB/s、混合精度 3.9x 加速

## 工具脚本

| 工具 | 说明 | 状态 |
|------|------|------|
| [training_recipe.py](tools/training_recipe.py) | 输入模型规格和 GPU 配置，生成 Megatron/DeepSpeed/DDP 训练启动命令 | 已验证 |
| [gpu_benchmark.py](tools/gpu_benchmark.py) | GPU 环境验证 + GEMM TFLOPS / HBM 带宽 / 混合精度 / Attention 性能测试 | 已验证 |
| [training_config_guide.py](tools/training_config_guide.py) | 根据模型大小和硬件推荐并行策略、batch size、梯度累积 | 已验证 |
| [ddp_train_demo.py](tools/ddp_train_demo.py) | PyTorch DDP 完整训练示例（多进程、梯度同步、checkpoint） | 已验证 |
| [comm_benchmark.py](tools/comm_benchmark.py) | 集合通信性能测试（AllReduce/AllGather/ReduceScatter 延迟和带宽） | 已验证 |
| [ai_infra_cheatsheet.py](tools/ai_infra_cheatsheet.py) | AI Infra 常用命令速查表（nvidia-smi/nccl/pytorch/triton） | 已验证 |
| [memory_estimator.py](tools/memory_estimator.py) | 训练显存估算（参数/梯度/优化器/激活值逐项计算） | |
| [scaling_simulator.py](tools/scaling_simulator.py) | 多卡扩展效率模拟（通信占比、线性度、最优卡数） | |
| [gpu_monitor.py](tools/gpu_monitor.py) | GPU 实时监控（利用率/显存/温度/功耗） | |
| [triton_examples.py](tools/triton_examples.py) | Triton kernel 示例集（vector_add/GEMM/softmax/flash_attn） | |
| [ddp_cpu_demo.py](tools/ddp_cpu_demo.py) | PyTorch DDP CPU 模式 demo（无 GPU 也可跑） | |
| [collective_ops_viz.py](tools/collective_ops_viz.py) | 集合通信原语可视化（AllReduce/AllGather/Broadcast 数据流） | |

## 工作日志

- [2026-06-03](diary/2026-06-03.md) — GPU 基准测试、vLLM 安装、HuggingFace 推理实验、笔记体系搭建

### 日志记录规范

- **文件**: `diary/YYYY-MM-DD.md`，每天一个文件
- **颗粒度**: 每完成一件事（学一个知识点、做一个实验、写一个工具）就记录一条
- **格式**: `HH:MM — 标题` + 2-3 句话，包含做了什么、为什么做、得出了什么结论或思考
- **顺序**: 按时间正序排列（最早的在最前面）
- **要求**: 不要太简略（不能只列名字），也不要太冗长；要能看到进展、结论和思考

### 笔记记录规范

- **基础知识**: `notebook/fundamentals/` — 每篇包含目标、核心概念、实践记录、参考资料
- **项目源码**: `notebook/projects/` — 源码阅读和分析笔记
- **实验记录**: `notebook/experiments/` — GPU 实验数据和结论
- **规划文档**: `notebook/roadmap.md`, `notebook/knowledge-map.md`

### 工具沉淀规范

- 每个工具独立可运行（有 `if __name__ == "__main__"` 入口）
- 包含 docstring 说明用法
- 验证状态标注在上方的目录树中

### Git 管理规范

- 每完成一个有意义的任务就 commit + push
- commit message 简要说明变更内容
- 所有变更通过 git 管理，不手动管理文件

### 环境信息

- **本地 conda**: `ai-infra` (Python 3.11, macOS)
- **GPU 服务器**: matpool.com A16 15GB, `myconda` 环境 (Python 3.9, PyTorch 2.0+cu117)
- **GPU 连接**: `source /root/miniconda3/bin/activate myconda`（必须激活，不能用系统 Python）
- **镜像源**: pip 用清华/阿里云镜像（中国大陆网络）

## 统计

| 类别 | 数量 |
|------|------|
| 基础知识笔记 | 27 篇 |
| 项目源码阅读 | 9 篇 |
| 实验记录 | 1 篇 |
| 实用工具脚本 | 12 个 |
| **总计** | **45 项** |

## GitHub

https://github.com/Jackie2049/rollout-infra
