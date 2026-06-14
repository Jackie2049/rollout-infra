# 7框架深度对比: DeepSpeed / Megatron-LM / vLLM / verl / MindIE / rLLM / PyTorch

> 2026-06-15 | 综合源码阅读+文档研究+实测数据
> 目标: 成为这7个框架的贴身顾问

## 1. 框架定位总览

| 框架 | 类型 | 核心场景 | Stars | 语言 | 开源 | 硬件 |
|------|------|----------|-------|------|------|------|
| DeepSpeed | 训练 | ZeRO分布式训练 | ~35K | Python+C++ | ✅ | NVIDIA GPU |
| Megatron-LM | 训练 | 3D并行(TP+PP+DP) | ~10K | Python+C++ | ✅ | NVIDIA GPU |
| vLLM | 推理 | 通用推理服务 | ~55K | Python+C++ | ✅ | NVIDIA GPU |
| verl | RL训练 | RLHF/GRPO训练 | ~5K | Python | ✅ | NVIDIA GPU |
| MindIE | 推理 | 昇腾推理服务 | ~3K | C++/Python | ❌部分 | Ascend NPU |
| rLLM | RL训练 | Agentic RL编排 | ~5.6K | Python | ✅ | NVIDIA GPU |
| PyTorch | 基础 | 通用训练+推理 | ~85K | Python+C++ | ✅ | NVIDIA+其他 |

## 2. 架构对比图

```
┌─────────────────── 基础层 ───────────────────┐
│                PyTorch                         │
│  distributed/FSDP2/compile/custom_op/AMP      │
└────────────────────────────────────────────────┘
         ↑                    ↑            ↑
┌────────│──── 训练层 ────────│────────────│────┐
│ DeepSpeed│   Megatron-LM   │   verl     │rLLM │
│  ZeRO    │   TP+PP+DP      │   GRPO    │Agent │
│  Offload │   SP+MoE        │   PPO     │  RL  │
│  Infinity│   Pipeline      │   Rollout │编排  │
└──────────│──────────────────│───────────│─────┘
                                    ↑    ↑
┌──────────────── 推理层 ──────────│────│──────┐
│              vLLM               │   │MindIE │
│  PagedAttention/Continuous Batch│   │ATB    │
│  Prefix Cache/Speculative Decode│   │CANN   │
│  PD Disaggregation              │   │昇腾   │
└─────────────────────────────────│───│───────┘
                                   │   │
                    ┌────── RL训练层 ────┐
                    │     rLLM          │
                    │  @rollout/@eval   │
                    │  Gateway+Trace    │
                    │  Backend:verl     │
                    └──────────────────┘
```

## 3. 核心技术对比

### 3.1 分布式训练策略

| 技术 | DeepSpeed | Megatron-LM | PyTorch | verl | rLLM |
|------|-----------|-------------|---------|------|------|
| ZeRO-1/2/3 | ✅核心 | ❌ | ❌(有FSDP) | ❌ | ❌ |
| TP | ❌ | ✅核心 | ❌ | ❌ | ❌ |
| PP | ❌ | ✅核心 | ❌ | ❌ | ❌ |
| DP | ✅ | ✅ | ✅ | ✅ | ✅ |
| SP | ❌ | ✅ | ✅(新) | ❌ | ❌ |
| FSDP/FSDP2 | ❌ | ❌ | ✅核心 | ✅ | ❌(用verl) |
| Expert Parallel | ❌ | ✅(MoE) | ❌ | ❌ | ❌ |
| ZeRO-Offload | ✅ | ❌ | ❌ | ❌ | ❌ |
| CPU Adam | ✅ | ❌ | ❌ | ✅ | ✅(via verl) |

### 3.2 内存优化

| 技术 | DeepSpeed | Megatron | vLLM | verl | rLLM |
|------|-----------|----------|------|------|------|
| ZeRO分片 | ✅(Ψ/N) | ❌ | ❌ | ❌ | ❌ |
| Gradient Checkpoint | ✅ | ✅ | ❌ | ✅ | ✅(via verl) |
| Paged KV Cache | ❌ | ❌ | ✅核心 | ❌ | ❌ |
| Prefix Caching | ❌ | ❌ | ✅(block级) | ❌ | ❌ |
| Quantization | ✅ | ✅ | ✅(INT4/INT8/FP8) | ✅ | ✅ |
| Activation Offload | ✅ | ✅ | ❌ | ❌ | ❌ |

### 3.3 推理优化

| 技术 | vLLM | MindIE | SGLang |
|------|------|--------|--------|
| Continuous Batching | ✅ | ✅ | ✅ |
| PagedAttention | ✅(block级) | ✅(block级) | ✅(token级Radix) |
| Speculative Decode | ✅(5种) | ✅(roadmap) | ✅(6种+Plugin) |
| Prefix Caching | ✅(hash block) | ✅ | ✅(Radix Tree) |
| PD Disaggregation | ✅(实验性) | ✅(roadmap) | ✅(完整) |
| Overlap Scheduling | ❌ | ❌ | ✅(核心创新) |
| HiCache | ❌ | ❌ | ✅(3层) |
| Quantization | ✅ | ✅ | ✅ |

### 3.4 RL训练流水线

| 技术 | verl | rLLM |
|------|------|------|
| Rollout | vLLM/SGLang backend | AgentFlow(any harness) |
| Token Capture | 直接从vLLM logprobs | Gateway透明捕获 |
| Reward | 用户提供函数 | @rllm.evaluator |
| Advantage | GRPO/RLOO内置 | 注册式(GRPO/RLOO/REINFORCE/++) |
| Multi-turn Drift | 无特殊处理 | TokenAccumulator (drift-free!) |
| Trajectory Group | DP-style grouping | task_id:traj_name grouping |
| Backend | 自身 | verl/tinker/fireworks 一行切换 |
| Eval | 无内置 | 60+ benchmark |
| Sandbox | 无 | Docker/Daytona/Modal |

## 4. 通信与同步

| 框架 | 通信库 | 主要模式 | 跨节点 |
|------|--------|----------|--------|
| DeepSpeed | NCCL | AllReduce/ReduceScatter/AllGather | ✅ |
| Megatron-LM | NCCL | AllReduce(TP)+P2P(PP)+AllGather/RS(SP) | ✅ |
| vLLM | NCCL | AllReduce(TP)+P2P(KV Transfer) | ✅ |
| verl | Ray+NCCL | ActorRollout→Critic→Reward→Update | ✅ |
| rLLM | Ray | TaskRunner→WorkerRollout→Gateway→Trainer | ✅ |
| MindIE | HCCL | AllReduce(TP)+P2P(PD Transfer) | ✅ |
| PyTorch | NCCL/gloo | ProcessGroup NCCL/gloo/MPI | ✅ |

## 5. 代码质量与设计

| 维度 | DeepSpeed | Megatron | vLLM | verl | rLLM | PyTorch |
|------|-----------|----------|------|------|------|---------|
| 代码量 | ~100K | ~50K | ~200K | ~30K | ~50K | ~500K+ |
| 设计模式 | Init metaclass注入 | 简洁TP/PP | 双进程+ZMQ | Ray actor | 装饰器+Protocol | C++核心+Python绑定 |
| 类型系统 | 弱 | 中 | 中(V1) | 中(dataclass) | 强(Pydantic) | 强(C++模板) |
| 测试覆盖 | 中 | 低 | 中 | 低 | 中 | 高 |
| 文档 | 好 | 中 | 好 | 中 | 好 | 最好 |
| 扩展性 | 高(ZeRO stage配置) | 中(TP/PP固定) | 高(Pluggable) | 中 | 最高(BackendProtocol) | 高 |

## 6. RTX 4090 适用性

| 框架 | 适用性 | 最佳配置 | 关键限制 |
|------|--------|----------|----------|
| DeepSpeed | ⭐⭐⭐ | ZeRO-2+CPU Adam, bf16 | ZeRO-3慢(offload开销) |
| Megatron-LM | ⭐⭐ | TP=1(单GPU), 小模型 | TP>1无意义(PCIe瓶颈) |
| vLLM | ⭐⭐⭐⭐⭐ | INT4+INT8KV+GQA-8+FlashInfer | decode=memory-bound |
| verl | ⭐⭐⭐ | GRPO+LoRA+CPU Adam | 7B bf16=20GB刚好 |
| MindIE | ❌ | Ascend专用, RTX 4090不支持 | 硬件绑定昇腾 |
| rLLM | ⭐⭐⭐ | tinker backend(单GPU) | 需vLLM/SGLang作为推理后端 |
| PyTorch | ⭐⭐⭐⭐ | FSDP2+bf16+compile | 分布式无NVLink=瓶颈 |

### RTX 4090 7B最优配置

```
推理: vLLM INT4+INT8KV+GQA-8+FlashInfer → 4,791 tok/s
推理+EAGLE: vLLM INT4+EAGLE d=5 → 9,088 tok/s
RL训练: verl GRPO+LoRA+bf16 → 20.04GB fits
RL编排: rLLM tinker backend + vLLM推理 → same code eval+train
ZeRO训练: DeepSpeed ZeRO-2+CPU Adam+bf16 → 有效但不如verl
```

## 7. 框架间依赖关系

```
PyTorch (底层)
    ├── DeepSpeed 依赖 PyTorch + CUDA
    ├── Megatron-LM 依赖 PyTorch + NCCL
    ├── vLLM 依赖 PyTorch + CUDA + FlashInfer
    ├── verl 依赖 PyTorch + vLLM/SGLang + Ray
    └── rLLM 依赖 verl (可选) + vLLM/SGLang + Ray

MindIE (独立栈)
    └── CANN + ATB + HCCL (昇腾专用, 不依赖PyTorch)
```

## 8. 学习路线建议 (按依赖深度排序)

1. **PyTorch** (底层基础) → distributed/FSDP2/compile/custom_op
2. **DeepSpeed** (训练优化) → ZeRO-1/2/3/Offload/Infinity
3. **Megatron-LM** (并行策略) → TP+PP+SP+MoE
4. **vLLM** (推理核心) → PagedAttention/Continuous Batching/Prefix Cache
5. **verl** (RL训练) → GRPO/PPO/Rollout Pipeline
6. **rLLM** (RL编排) → AgentFlow/Gateway/TokenAccumulator
7. **MindIE** (昇腾推理) → ATB/CANN/HCCL/vLLM-Ascend

## 9. 贡献机会

| 框架 | 可能贡献点 | 难度 |
|------|-----------|------|
| DeepSpeed | ZeRO-3 offload优化 | 高 |
| Megatron-LM | MoE serving优化 | 高 |
| vLLM | PR #45157 (NIXL metrics docs, OPEN) | 低 |
| verl | prefix sharing + trajectory grouping | 中 |
| rLLM | RTX 4090 tinker backend优化 | 中 |
| MindIE | vLLM-Ascend ATB ops贡献 | 高(需昇腾硬件) |
| PyTorch | custom_op/FSDP2改进 | 高 |

## 10. 各框架详细笔记链接

- DeepSpeed → `notebook/fundamentals/zero-optimizer.md` + MEMORY
- Megatron-LM → `notebook/projects/megatron-tp-source-reading.md` + MEMORY
- vLLM → `notebook/projects/vllm-*.md` (15+篇)
- verl → `notebook/projects/verl-*.md` (12篇)
- MindIE → `notebook/projects/mindie-architecture-reading.md`
- rLLM → `notebook/projects/rllm-architecture-reading.md`
- PyTorch → `notebook/fundamentals/fsdp.md` + MEMORY (ProcessGroupNCCL/DDP/FSDP2)
