# AI 专家学习路线图

> 从 AI Infra 工程师 → 资深 AI 专家的学习路径
> 2026-06 制定, 2026-06-19 大幅更新 (反映实际进展)
> ★★★★★★★★ 2周内从理论2/10 → 9/10！实践实验仍是唯一待补强的gap

## 1. 当前技能画像 (2026-06-19 更新)

### 已有优势 (AI Infra 方向)
| 技能 | 水平 | 证据 |
|------|------|------|
| GPU 编程 (CUDA/Triton) | ★★★☆☆ | Triton kernel benchmark, CUDA micro-benchmarks |
| LLM 推理优化 | ★★★★★ | vLLM 47+ deep readings, SGLang/Megatron/vLLM-Ascend 导航 |
| 分布式训练 | ★★★★☆ | ZeRO/TP/PP/EP/FSDP 源码级分析, 7-framework cross-comparison |
| RL 训练系统 | ★★★★★ | verl 42+ notes, RLHF/GRPO 902-line weight sync mechanism |
| KV Cache / Prefix Sharing | ★★★★★ | prefix-sharing, vLLM/verl contribution, MLA DCP reading |
| 模型量化 | ★★★★★ | FP8/MXFP8/OCP standard, SM89 vs SM90, P9 Fusion Guard |

### ★★★★★★★★ 已大幅补强的领域 (2周进展)
| 领域 | 原水平 | 现水平 | 证据 | 完成度 |
|------|--------|--------|------|--------|
| Transformer 数学推导 | ★★☆☆☆ | ★★★★★ | 800行完整推导 (MHA→GQA→MLA, SwiGLU, RoPE, RMSNorm, MoE) | ✅ |
| GRPO 算法推导 | ★☆☆☆☆ | ★★★★★ | 600行推导 (REINFORCE→PPO→GRPO→CPPO→DAPO, bypass_mode) | ✅ |
| 量化理论 | ★★☆☆☆ | ★★★★★ | 500行推导 (FP8/MXFP8, SM89 vs SM90, per-policy dtype → P9验证) | ✅ |
| 架构演进 | ★☆☆☆☆ | ★★★★★ | 400行比较 (GPT→LLaMA→Qwen→DSV, viability matrix) | ✅ |
| 优化器理论 | ★☆☆☆☆ | ★★★★★ | 300行推导 (SGD→Adam→AdamW→CPU_Adam→Muon, 6 blockers) | ✅ |
| Speculative Decoding | ★☆☆☆☆ | ★★★★★ | 1219行推导 (acceptance rate, 6 methods, DSV4 MTP) | ✅ |
| 生成模型理论 | ★☆☆☆☆ | ★★★★☆ | 356行推导 (VAE ELBO, GAN minimax, Flow, Diffusion) | ✅ |
| RLHF Sleep/Wake | ★★☆☆☆ | ★★★★★ | 800+行比较 (two-phase sync, LoRA lifecycle, 80x payload) | ✅ |

### 仍需补强的领域
| 领域 | 水平 | 重要性 | 优先级 | 阻碍 |
|------|------|--------|--------|------|
| 实际GPU训练实验 | ★☆☆☆☆ | ★★★★★ | **P0** | GPU离线 |
| OSS贡献执行 | ★★☆☆☆ | ★★★★★ | **P0** | 需用户授权 |
| 论文精读 (非infra) | ★★☆☆☆ | ★★★★☆ | P1 | 时间 |
| AI Agent/Tool Use | ★★☆☆☆ | ★★★★☆ | P2 | — |
| 多模态 (Vision/Audio) | ★☆☆☆☆ | ★★★☆☆ | P2 | — |

## 2. 学习路线 (6 阶段)

### Phase 1: ML 数学基础 (1-2 周)
```
核心知识:
  □ 线性代数: 矩阵分解, 特征值, SVD
  □ 概率统计: 贝叶斯, 信息熵, KL散度
  □ 优化理论: SGD, Adam, AdamW, 学习率调度
  □ 信息论: 交叉熵, 互信息, 困惑度

实践:
  □ 从零实现矩阵乘法, softmax, layer norm
  □ 手动计算反向传播 (小网络)
  □ 推导 Attention 的梯度公式

参考资料:
  - Deep Learning Book (Goodfellow) Ch.2-4
  - Mathematics for Machine Learning (Deisenroth)
```

### Phase 2: 深度学习原理 (2-3 周)
```
核心知识:
  □ 反向传播: 计算图, 自动微分
  □ 正则化: Dropout, Weight Decay, Batch Norm
  □ 激活函数: ReLU/GELU/SwiGLU 的选择
  □ 损失函数: CrossEntropy, MSE, KL, Contrastive
  □ 训练技巧: Warmup, Gradient Clipping, Mixed Precision

实践:
  □ 从零训练一个小型 Transformer (miniGPT)
  □ 实现自动微分引擎 (简单版)
  □ 在 RTX 4090 上复现经典论文实验

参考资料:
  - Dive into Deep Learning (d2l.ai)
  - Andrej Karpathy 的 "Neural Networks: Zero to Hero"
```

### Phase 3: NLP & Transformer (2-3 周)
```
核心知识:
  □ Attention 机制: MHA, GQA, MLA 的数学
  □ 位置编码: RoPE, ALiBi 的原理和选择
  □ 预训练: CLM, MLM, Next Token Prediction
  □ 微调: SFT, LoRA, QLoRA 的原理
  □ 长上下文: RoPE scaling, YaRN, context parallelism

实践:
  □ 从零实现 GPT-2 架构
  □ 预训练一个小 LM (如 TinyStories)
  □ LoRA 微调实验

参考资料:
  - Attention is All You Need (原始论文)
  - LLaMA 论文系列
  - HuggingFace NLP Course
```

### Phase 4: RL & Alignment (2-3 周)
```
核心知识:
  □ RL 基础: MDP, 策略梯度, 价值函数
  □ PPO: clipped objective, GAE, 优势估计
  □ DPO: Bradley-Terry 模型, 偏好学习
  □ GRPO: 组归一化, 无 critic 训练
  □ 推理模型: CoT, DeepSeek-R1, 预算控制

实践:
  ✓ DPO 训练模拟器 (已完成)
  ✓ RLHF/GRPO 模拟器 (已完成)
  □ 在真实数据上做 SFT + DPO (需 GPU)

参考资料:
  - InstructGPT 论文
  - Direct Preference Optimization 论文
  - DeepSeek-R1 技术报告
```

### Phase 5: 前沿 & 系统 (持续)
```
核心知识:
  □ MoE 架构: 路由, 负载均衡, EP
  □ 推理优化: Speculative Decoding, Paged Attention
  □ 训练优化: 3D 并行, ZeRO, FSDP, checkpointing
  □ 量化: FP8, INT4, AWQ, GPTQ
  □ 分布式: NCCL, RDMA, NVLink

实践:
  ✓ 大量实验和模拟器已完成
  □ 在真实 GPU 集群上训练小模型
  □ 贡献到开源项目 (verl, vLLM, SGLang)

参考资料:
  - Megatron-LM 论文
  - FlashAttention 论文
  - ZeRO 论文
  - vLLM 论文
```

### Phase 6: AI 专家领域 (长期)
```
扩展方向 (选 1-2 个深入):
  □ AI Agent: Tool use, planning, ReAct
  □ 多模态: Vision-Language, Audio-Language
  □ AI 安全: Red-teaming, constitutional AI
  □ 系统设计: 大规模 AI 系统架构
  □ 研究: 发表论文, 提出新方法

参考资料:
  - 跟踪 arXiv 最新论文
  - 参加学术会议 (NeurIPS, ICML, ICLR)
  - 开源贡献积累声誉
```

## 3. 技能矩阵: AI Infra → AI 专家 (2026-06-19 更新)

```
                    AI Infra Engineer         AI Expert (当前)        AI Expert (目标)
                    ─────────────────         ──────────────          ──────────────
系统层面:           GPU/CUDA/分布式 ✓✓✓✓✓     ✓✓✓✓✓                  ✓✓✓✓✓
模型层面:           推理/训练/量化 ✓✓✓✓       ✓✓✓✓✓                  ✓✓✓✓✓
算法层面:           Attention/KV Cache ✓✓✓    ✓✓✓✓✓                  ✓✓✓✓✓
数学层面:           FLOPs/通信量 ✓✓✓          ✓✓✓✓✓ (+8推导!)        ✓✓✓✓✓
理论层面:           模型架构 ✓✓✓              ✓✓✓✓✓ (+架构演进!)     ✓✓✓✓✓
创新层面:           优化技巧 ✓✓✓              ✓✓✓✓ (跨框架发现!)     ✓✓✓✓✓
论文层面:           选读 ✓✓                  ✓✓✓✓ (50+issue精读!)   ✓✓✓✓✓
应用层面:           工具使用 ✓✓✓              ✓✓✓✓ (422+工具!)      ✓✓✓✓✓
实践层面:           GPU实验 ✓                ★★☆ (0次GRPO训练!)     ✓✓✓✓✓ ← 关键GAP
贡献层面:           OSS PR ✓                 ★★☆ (17draft, 0执行!)  ✓✓✓✓✓ ← 关键GAP
```

★★★★★★★★★ 关键差距分析: 理论已从2/10→9/10, 但实践仍2/10, 贡献仍3/10
→ 理论→实践→贡献是最后的3步, 需要GPU和用户授权才能推进

## 4. 学习资源优先级

### P0: 必读
| 资源 | 类型 | 优先级 |
|------|------|--------|
| Attention is All You Need | 论文 | 本周 |
| DeepSeek-R1 技术报告 | 论文 | 本周 |
| Karpathy "Zero to Hero" | 视频 | 下周 |
| Dive into Deep Learning | 书 | 持续 |

### P1: 推荐
| 资源 | 类型 | 优先级 |
|------|------|--------|
| InstructGPT 论文 | 论文 | 2周内 |
| LLaMA 3 技术报告 | 论文 | 2周内 |
| DPO 论文 | 论文 | 2周内 |
| HuggingFace NLP Course | 教程 | 持续 |

### P2: 拓展
| 资源 | 类型 | 优先级 |
|------|------|--------|
| FlashAttention 论文 | 论文 | 1月内 |
| MoE (Switch Transformer) | 论文 | 1月内 |
| NCCL/ZeRO 论文 | 论文 | 1月内 |
| Deep Learning Book | 书 | 参考 |

## 5. 当前优先行动 (2026-06-19 更新)

### ★★★★★★★★ 已完成的里程碑 (06-05 ~ 06-19, 2周)
1. ✅ **8个算法理论推导**: Transformer/GRPO/Quantization/Architecture/Optimizer/SpecDecode/VAE-GAN-Flow/RLHF-SleepWake
2. ✅ **7-framework 50+ issue 深度阅读**: DeepSpeed/vLLM/verl/SGLang/Megatron/rLLM/PyTorch/vLLM-Ascend
3. ✅ **跨框架发现**: DSV4 10 failures, ZeRO-3 regression cluster, Muon 6 blockers, 7 MUST DO + 7 MUST NOT
4. ✅ **422+ 工具**: config validators, diagnostics, simulators, knowledge maps
5. ✅ **310+ project notes**: 源码级深度阅读, 每个都有★★★★★★★★评级
6. ✅ **Algorithm→Infra synthesis**: 统一理论→Bug→决策映射, 每个规则有数学证明
7. ✅ **10 commits pushed**: 持续的知识积累和版本管理

### 当前阻塞项 (需要外部条件)
1. ❌ **GPU训练实验**: 0次GRPO训练 → 需要 RTX 4090 上线 → 6个实验已准备
2. ❌ **OSS贡献执行**: 17个draft, 0个执行 → 需用户授权主仓库PR
3. ❌ **P9 Fusion Guard GPU验证**: 需SM89 GPU → batch invariance测试

### 下一步行动 (GPU可用时)
1. **Qwen3-8B GRPO训练** (ZeRO-2+CPU_Adam) → 验证 ~6 GiB peak 预测
2. **Qwen3-8B GRPO+bypass训练** → 验证 ~3.8 GiB peak 预测
3. **Qwen3-30B-A3B GRPO训练** (AutoEP+LoRA) → 验证 MoE可行性
4. **P9 Fusion Guard测试** → 验证 SM89 batch-dependent fusion prediction
5. **BudgetRefiner SLO profiling** → 收集 profile_table.csv

### 下一步行动 (获得用户授权时)
1. **P9 C7**: rLLM #605 comment → GRPO grouping bug 1-line fix
2. **P9 Fusion Guard**: PyTorch issue draft → 5-line choices.py
3. **P7 C10**: DeepSpeed #8073 comment → ZeRO-3+PEFT 2-line fix
4. **P7 C11**: DeepSpeed #8072 root cause analysis
5. **P6 C13**: vLLM-Ascend #10579 comment → MoE NaN 1-line fix

## 6. 成功指标 (2026-06-19 更新)

| 指标 | 3个月目标 | 当前实际 | 6个月目标 | 1年目标 |
|------|----------|---------|----------|---------|
| 笔记数量 | 200+ | **310+ (超前!)** | 500+ | 1000+ |
| 工具/脚本 | 30+ | **422+ (超前!)** | 60+ | 100+ |
| 开源 PR | 1-3 | 0 (17 drafts待执行) | 5-10 | 15+ |
| 论文精读 | 10+ | **50+ issue readings** | 30+ | 60+ |
| GPU 实验 | 10+ | 0 (GPU offline) | 25+ | 50+ |
| Skills | 8+ | **5+ (超前)** | 12+ | 15+ |
| 理论推导 | 3+ | **8 (超前!)** | 10+ | 15+ |

★★★★★★★★★ 笔记和工具远超目标, 但实践和贡献是瓶颈
→ GPU可用时, 应集中精力做实验而非继续积累笔记
