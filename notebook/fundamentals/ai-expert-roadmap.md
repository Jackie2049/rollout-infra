# AI 专家学习路线图

> 从 AI Infra 工程师 → 资深 AI 专家的学习路径
> 2026-06 制定, 基于当前技能画像和行业趋势

## 1. 当前技能画像

### 已有优势 (AI Infra 方向)
| 技能 | 水平 | 证据 |
|------|------|------|
| GPU 编程 (CUDA/Triton) | ★★★☆☆ | Triton kernel benchmark, CUDA micro-benchmarks |
| LLM 推理优化 | ★★★★☆ | vLLM 源码深度阅读, SGLang/Megatron 导航 |
| 分布式训练 | ★★★☆☆ | ZeRO/TP/PP/EP 分析, FSDP 模拟器 |
| RL 训练系统 | ★★★☆☆ | verl 源码分析, RLHF/GRPO 模拟器 |
| KV Cache / Prefix Sharing | ★★★★★ | prefix-sharing 项目, vLLM/verl 贡献 |
| 模型量化 | ★★★☆☆ | FP8/INT4 分析, 量化实验 |

### 需要补强的领域
| 领域 | 水平 | 重要性 | 优先级 |
|------|------|--------|--------|
| 深度学习理论 | ★★☆☆☆ | ★★★★★ | **P0** |
| NLP/Transformer 原理 | ★★★☆☆ | ★★★★★ | **P0** |
| ML 数学 (线代/概率/优化) | ★★☆☆☆ | ★★★★☆ | **P0** |
| RL 理论 (MDP/策略梯度) | ★★☆☆☆ | ★★★★☆ | P1 |
| 模型架构设计 | ★★☆☆☆ | ★★★★☆ | P1 |
| 论文阅读能力 | ★★☆☆☆ | ★★★★★ | P1 |
| AI 安全 / Alignment | ★★☆☆☆ | ★★★☆☆ | P2 |
| 多模态 (Vision/Audio) | ★☆☆☆☆ | ★★★☆☆ | P2 |
| AI Agent / Tool Use | ★★☆☆☆ | ★★★★☆ | P2 |

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

## 3. 技能矩阵: AI Infra → AI 专家

```
                    AI Infra Engineer         AI Expert
                    ─────────────────         ─────────
系统层面:           GPU/CUDA/分布式 ✓✓✓✓✓     ✓✓✓✓✓
模型层面:           推理/训练/量化 ✓✓✓✓       ✓✓✓✓
算法层面:           Attention/KV Cache ✓✓✓    ✓✓✓✓✓
数学层面:           FLOPs/通信量 ✓✓✓          ✓✓✓✓✓
理论层面:           模型架构 ✓✓✓              ✓✓✓✓✓
创新层面:           优化技巧 ✓✓✓              ✓✓✓✓✓
论文层面:           选读 ✓✓                  ✓✓✓✓✓
应用层面:           工具使用 ✓✓✓              ✓✓✓✓
```

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

## 5. 当前优先行动

### 本周 (2026-06-05 ~ 06-12)
1. **论文精读**: Attention is All You Need + DeepSeek-R1
2. **开源贡献**: 在 verl #6401 评论, 或提交小 PR
3. **实践**: 在 RTX 4090 上训练一个 miniGPT (从头)
4. **数学**: 推导 Attention 梯度, 手写反向传播

### 下周 (06-12 ~ 06-19)
1. **论文精读**: InstructGPT + DPO
2. **实践**: SFT + DPO 完整流程 (真实数据)
3. **系统**: 阅读 NCCL 源码, 理解通信优化
4. **开源**: 提交 vLLM 或 SGLang PR

### 月度目标 (06 月底)
1. 精读 5+ 核心论文, 每篇写笔记
2. 3+ 个 GPU 实验 (miniGPT, SFT+DPO, 分布式)
3. 1+ 个开源 PR 被合并
4. notebook/ 累计 200+ 笔记
5. 完成路线图 Phase 1-3

## 6. 成功指标

| 指标 | 3个月目标 | 6个月目标 | 1年目标 |
|------|----------|----------|---------|
| 笔记数量 | 200+ | 500+ | 1000+ |
| 工具/脚本 | 30+ | 60+ | 100+ |
| 开源 PR | 1-3 | 5-10 | 15+ |
| 论文精读 | 10+ | 30+ | 60+ |
| GPU 实验 | 10+ | 25+ | 50+ |
| Skills | 8+ | 12+ | 15+ |
