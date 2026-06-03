# AI Infra Engineer Learning Hub

> 成为一名强悍的 AI Infra 工程师 — 学习笔记与实战记录

## 仓库结构

```
rollout-infra/
├── notebook/                          # 学习笔记 (含项目源码分析)
│   ├── README.md                      # 本文件
│   ├── roadmap.md                     # 学习路线图
│   ├── knowledge-map.md               # 知识地图
│   ├── gpu-experiment-plan.md         # GPU 实验计划
│   │
│   ├── fundamentals/                  # 基础知识笔记 (22 篇)
│   │   ├── distributed-training.md    # 分布式训练
│   │   ├── flash-attention.md         # FlashAttention 1/2/3
│   │   ├── mixed-precision.md         # 混合精度训练
│   │   ├── zero-optimizer.md          # ZeRO 优化器
│   │   ├── cuda-basics.md             # CUDA 编程基础
│   │   ├── triton-programming.md      # Triton 编程
│   │   ├── nccl.md                    # NCCL 通信库
│   │   ├── speculative-decoding.md    # Speculative Decoding
│   │   ├── quantization.md            # 推理量化
│   │   ├── gradient-checkpointing.md  # 梯度检查点
│   │   ├── ray-framework.md           # Ray 分布式框架
│   │   ├── checkpoint.md              # 分布式 Checkpoint
│   │   ├── dataloader.md              # DataLoader 优化
│   │   ├── gpu-memory-optimization.md # GPU 显存优化
│   │   ├── comm-compute-overlap.md    # 通信与计算重叠
│   │   ├── cuda-graphs.md             # CUDA Graphs
│   │   ├── gpu-profiling.md           # GPU 性能分析
│   │   ├── model-serving.md           # 模型服务架构
│   │   ├── k8s-gpu-scheduling.md      # K8s GPU 调度
│   │   ├── slurm.md                   # Slurm 集群管理
│   │   └── distributed-fs.md          # 分布式文件系统
│   │
│   ├── projects/                      # 开源项目源码阅读 (9 篇)
│   │   ├── vllm-architecture.md       # vLLM V1 架构分析
│   │   ├── vllm-prefix-caching-v1.md  # vLLM Prefix Caching 源码分析
│   │   ├── megatron-tp-reading.md     # Megatron-LM 张量并行
│   │   ├── megatron-pp-reading.md     # Megatron-LM 流水线并行
│   │   ├── verl-architecture.md       # verl 架构分析
│   │   ├── verl-prefix-grouper.md     # verl PrefixGrouper 分析
│   │   ├── vllm-contribution-plan.md  # vLLM 开源贡献计划
│   │   ├── vllm-contribution-prep.md  # vLLM 贡献准备
│   │   └── troubleshooting-guide.md   # 分布式训练排错指南
│   │
│   └── experiments/                   # GPU 实验记录
│       └── gpu-a16-benchmark.md       # A16 基准测试数据
│
├── tools/                             # 工具与脚本 (12 个)
│   ├── ddp_cpu_demo.py                # DDP CPU demo
│   ├── collective_ops_viz.py          # 集合通信原语可视化
│   ├── ddp_train_demo.py              # DDP 完整训练 demo
│   ├── comm_benchmark.py              # 通信 benchmark
│   ├── memory_estimator.py            # 训练显存估算器
│   ├── scaling_simulator.py           # 扩展效率模拟器
│   ├── gpu_monitor.py                 # GPU 实时监控脚本
│   ├── training_config_guide.py       # 训练配置决策工具
│   ├── training_recipe.py             # 训练启动命令生成器
│   ├── triton_examples.py             # Triton kernel 示例集
│   ├── gpu_benchmark.py               # GPU 快速基准测试
│   └── ai_infra_cheatsheet.py         # AI Infra 命令速查表
│
└── diary/                             # 日志 (每天一个文件)
    └── 2026-06-03.md                  # 按时间顺序记录进展
```

## 统计

| 类别 | 数量 |
|------|------|
| 基础知识笔记 | 22 篇 |
| 项目源码阅读 | 9 篇 |
| 实验记录 | 1 篇 |
| 实用工具脚本 | 12 个 |
| 日记 | 1 天 |
| **总计** | **45 项** |

## 约定

- **notebook/**: 学习笔记和源码分析，Markdown 格式，每篇包含目标、核心概念、实践记录、参考资料
- **tools/**: 可运行的脚本和工具，注重实用性和可扩展性
- **diary/**: 每天一个 Markdown 文件 (YYYY-MM-DD.md)，按时间顺序记录进展和思考
- 为未来转换为 Claude Code Skills 做准备（结构化、模块化）
