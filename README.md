# rollout-infra — AI Infra 工程师成长之路

> 从零开始，系统学习 AI 基础设施，成为一名强悍的 AI Infra 工程师

## 仓库运作规则

### 目录结构

```
rollout-infra/
├── notebook/                           # 学习笔记 (含项目源码分析)
│   ├── [roadmap.md](notebook/roadmap.md)                      # 学习路线图
│   ├── [knowledge-map.md](notebook/knowledge-map.md)                # 知识地图 (8 大领域技能树)
│   ├── [gpu-experiment-plan.md](notebook/gpu-experiment-plan.md)          # GPU 实验计划
│   │
│   ├── fundamentals/                   # 基础知识笔记 (22 篇)
│   │   ├── [distributed-training.md](notebook/fundamentals/distributed-training.md)     # 分布式训练 (DP/TP/PP/SP/ZeRO/通信原语)
│   │   ├── [flash-attention.md](notebook/fundamentals/flash-attention.md)          # FlashAttention 1/2/3 深度解析
│   │   ├── [mixed-precision.md](notebook/fundamentals/mixed-precision.md)          # 混合精度训练 (FP16/BF16/FP8/Loss Scaling)
│   │   ├── [zero-optimizer.md](notebook/fundamentals/zero-optimizer.md)           # ZeRO 优化器 (1/2/3 原理+显存公式)
│   │   ├── [cuda-basics.md](notebook/fundamentals/cuda-basics.md)              # CUDA 编程基础 (线程/内存/流/Tensor Core)
│   │   ├── [triton-programming.md](notebook/fundamentals/triton-programming.md)       # Triton 编程 (kernel/autotune/GEMM/FlashAttn)
│   │   ├── [nccl.md](notebook/fundamentals/nccl.md)                     # NCCL 通信库 (Ring/Tree/调优)
│   │   ├── [speculative-decoding.md](notebook/fundamentals/speculative-decoding.md)     # Speculative Decoding (Draft-Verify/Medusa/Eagle)
│   │   ├── [quantization.md](notebook/fundamentals/quantization.md)             # 推理量化 (GPTQ/AWQ/SmoothQuant/FP8)
│   │   ├── [gradient-checkpointing.md](notebook/fundamentals/gradient-checkpointing.md)   # 梯度检查点/激活重计算
│   │   ├── [ray-framework.md](notebook/fundamentals/ray-framework.md)            # Ray 分布式框架
│   │   ├── [checkpoint.md](notebook/fundamentals/checkpoint.md)               # 分布式 Checkpoint (分片/异步/mmap)
│   │   ├── [dataloader.md](notebook/fundamentals/dataloader.md)               # DataLoader 优化 (prefetch/mmap/WebDataset)
│   │   ├── [gpu-memory-optimization.md](notebook/fundamentals/gpu-memory-optimization.md)  # GPU 显存优化 (估算/OOM/ZeRO/碎片化)
│   │   ├── [comm-compute-overlap.md](notebook/fundamentals/comm-compute-overlap.md)     # 通信与计算重叠 (Bucket/1F1B/Prefetch)
│   │   ├── [cuda-graphs.md](notebook/fundamentals/cuda-graphs.md)              # CUDA Graphs (推理加速/录制重放/vLLM)
│   │   ├── [gpu-profiling.md](notebook/fundamentals/gpu-profiling.md)            # GPU 性能分析 (nsys/ncu/PyTorch Profiler)
│   │   ├── [model-serving.md](notebook/fundamentals/model-serving.md)            # 模型服务架构 (API/负载均衡/HPA)
│   │   ├── [k8s-gpu-scheduling.md](notebook/fundamentals/k8s-gpu-scheduling.md)       # K8s GPU 调度 (Device Plugin/MIG/Gang)
│   │   ├── [slurm.md](notebook/fundamentals/slurm.md)                    # Slurm 集群管理 (sbatch/NCCL/抢占)
│   │   └── [distributed-fs.md](notebook/fundamentals/distributed-fs.md)           # 分布式文件系统 (Lustre/条带化/IO优化)
│   │
│   ├── projects/                       # 开源项目源码阅读 (9 篇)
│   │   ├── [vllm-architecture.md](notebook/projects/vllm-architecture.md)        # vLLM V1 架构深度分析
│   │   ├── [vllm-prefix-caching-v1.md](notebook/projects/vllm-prefix-caching-v1.md)   # vLLM Prefix Caching 源码分析
│   │   ├── [vllm-contribution-plan.md](notebook/projects/vllm-contribution-plan.md)   # vLLM 开源贡献计划 (11 个 good first issue)
│   │   ├── [vllm-contribution-prep.md](notebook/projects/vllm-contribution-prep.md)   # vLLM 贡献准备 (FlashInfer 后端分析)
│   │   ├── [megatron-tp-reading.md](notebook/projects/megatron-tp-reading.md)      # Megatron-LM 张量并行源码阅读
│   │   ├── [megatron-pp-reading.md](notebook/projects/megatron-pp-reading.md)      # Megatron-LM 流水线并行源码阅读
│   │   ├── [verl-architecture.md](notebook/projects/verl-architecture.md)        # verl 架构深度分析 (三级前缀缓存)
│   │   ├── [verl-prefix-grouper.md](notebook/projects/verl-prefix-grouper.md)      # verl PrefixGrouper 源码分析
│   │   └── [troubleshooting-guide.md](notebook/projects/troubleshooting-guide.md)    # 分布式训练排错指南
│   │
│   └── [experiments/](notebook/experiments/)                    # GPU 实验记录
│       └── [gpu-a16-benchmark.md](notebook/experiments/gpu-a16-benchmark.md)        # A16 基准测试数据
│
├── [tools/](tools/)                              # 工具与脚本 (12 个)
│   ├── [ddp_cpu_demo.py](tools/ddp_cpu_demo.py)                 # PyTorch DDP CPU demo
│   ├── [collective_ops_viz.py](tools/collective_ops_viz.py)           # 集合通信原语可视化
│   ├── [ddp_train_demo.py](tools/ddp_train_demo.py)               # DDP 完整训练 demo (已验证)
│   ├── [comm_benchmark.py](tools/comm_benchmark.py)               # 通信 benchmark (已验证)
│   ├── [memory_estimator.py](tools/memory_estimator.py)             # 训练显存估算器
│   ├── [scaling_simulator.py](tools/scaling_simulator.py)            # 扩展效率模拟器
│   ├── [gpu_monitor.py](tools/gpu_monitor.py)                  # GPU 实时监控脚本
│   ├── [training_config_guide.py](tools/training_config_guide.py)        # 训练配置决策工具 (已验证)
│   ├── [training_recipe.py](tools/training_recipe.py)              # 训练启动命令生成器 (已验证)
│   ├── [triton_examples.py](tools/triton_examples.py)              # Triton kernel 示例集
│   ├── [gpu_benchmark.py](tools/gpu_benchmark.py)                # GPU 快速基准测试 (已验证)
│   └── [ai_infra_cheatsheet.py](tools/ai_infra_cheatsheet.py)          # AI Infra 命令速查表 (已验证)
│
└── [diary/](diary/)                              # 工作日志 (每天一个文件)
    └── [2026-06-03.md](diary/2026-06-03.md)                   # 按时间正序记录进展和思考
```

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
| 基础知识笔记 | 22 篇 |
| 项目源码阅读 | 9 篇 |
| 实验记录 | 1 篇 |
| 实用工具脚本 | 12 个 |
| **总计** | **45 项** |

## GitHub

https://github.com/Jackie2049/rollout-infra
