# AI Infra Engineer Learning Hub

> 成为一名强悍的 AI Infra 工程师 — 学习笔记与实战记录

## 笔记结构

```
notebook/
├── README.md                          # 本文件 — 导航索引
├── roadmap.md                         # 学习路线图（含完成标记）
├── knowledge-map.md                   # 知识地图（8 大领域技能树）
├── daily-log.md                       # 每日进展日志（时间倒序）
├── gpu-experiment-plan.md             # GPU 到位后的实验计划
│
├── fundamentals/                      # 基础知识笔记（22 篇）
│   ├── distributed-training.md        # 分布式训练（DP/TP/PP/SP/ZeRO/通信原语）
│   ├── flash-attention.md             # FlashAttention 1/2/3 深度解析
│   ├── mixed-precision.md             # 混合精度训练（FP16/BF16/FP8/Loss Scaling）
│   ├── zero-optimizer.md              # ZeRO 优化器（1/2/3 原理+显存公式）
│   ├── cuda-basics.md                 # CUDA 编程基础（线程/内存/流/Tensor Core）
│   ├── triton-programming.md          # Triton 编程（kernel/autotune/FlashAttn）
│   ├── nccl.md                        # NCCL 通信库（Ring/Tree/调优）
│   ├── speculative-decoding.md        # Speculative Decoding（Draft-Verify/Medusa/Eagle）
│   ├── quantization.md                # 推理量化（GPTQ/AWQ/SmoothQuant/FP8）
│   ├── gradient-checkpointing.md      # 梯度检查点/激活重计算（sqrt(n) 策略）
│   ├── ray-framework.md               # Ray 分布式框架（Actor/Task/Placement Group）
│   ├── checkpoint.md                  # 分布式 Checkpoint（分片/异步/mmap）
│   ├── dataloader.md                  # DataLoader 优化（prefetch/mmap/WebDataset）
│   ├── gpu-memory-optimization.md     # GPU 显存优化（估算/OOM/ZeRO/碎片化）
│   ├── comm-compute-overlap.md        # 通信与计算重叠（Bucket/1F1B/Prefetch）
│   ├── cuda-graphs.md                 # CUDA Graphs（推理加速/录制重放/vLLM）
│   ├── gpu-profiling.md               # GPU 性能分析（nsys/ncu/PyTorch Profiler）
│   ├── model-serving.md               # 模型服务架构（API/负载均衡/多副本/HPA）
│   ├── k8s-gpu-scheduling.md          # K8s GPU 调度（Device Plugin/MIG/Gang）
│   ├── slurm.md                       # Slurm 集群管理（sbatch/NCCL/抢占处理）
│   └── distributed-fs.md              # 分布式文件系统（Lustre/条带化/IO优化）
│
├── projects/                          # 开源项目源码阅读（8 篇）
│   ├── vllm-architecture.md           # vLLM V1 架构深度分析
│   ├── megatron-tp-reading.md         # Megatron-LM 张量并行源码阅读
│   ├── megatron-pp-reading.md         # Megatron-LM 流水线并行源码阅读
│   ├── verl-architecture.md           # verl 架构深度分析（三级前缀缓存）
│   ├── verl-prefix-grouper.md         # verl PrefixGrouper 源码分析
│   ├── vllm-contribution-plan.md      # vLLM 开源贡献计划（11 个 good first issue）
│   ├── vllm-contribution-prep.md      # vLLM 贡献准备（源码定位+FlashInfer 分析）
│   └── troubleshooting-guide.md       # 分布式训练排错指南
│
└── ../tools/                          # 工具与脚本（11 个）
    ├── ddp_cpu_demo.py                # PyTorch DDP CPU demo
    ├── collective_ops_viz.py          # 集合通信原语可视化
    ├── ddp_train_demo.py              # PyTorch DDP 完整训练 demo（已验证）
    ├── comm_benchmark.py              # 通信 benchmark（已验证）
    ├── memory_estimator.py            # 训练显存估算器
    ├── scaling_simulator.py           # 扩展效率模拟器
    ├── gpu_monitor.py                 # GPU 实时监控脚本（待 GPU 验证）
    ├── training_config_guide.py       # 分布式训练配置决策工具（已验证）
    ├── training_recipe.py             # 训练启动命令生成器（已验证）
    ├── triton_examples.py             # Triton kernel 示例集（待 GPU 验证）
    └── ai_infra_cheatsheet.py         # AI Infra 常用命令速查表（已验证）
```

## 统计

| 类别 | 数量 |
|------|------|
| 基础知识笔记 | 22 篇 |
| 项目源码阅读 | 8 篇 |
| 实用工具脚本 | 11 个 |
| **总计** | **41 项** |

## 约定

- 所有笔记以 Markdown 格式记录
- 每篇笔记包含：目标、核心概念、实践记录、参考资料
- 注重可运行的代码片段和脚本
- 为未来转换为 Claude Code Skills 做准备（结构化、模块化）
