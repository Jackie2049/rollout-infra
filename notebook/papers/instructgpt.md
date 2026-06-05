# Paper Reading: InstructGPT — Training language models to follow instructions

> Ouyang et al., NeurIPS 2022 | OpenAI
> 精读日期: 2026-06-05
> 优先级: P1 (AI Expert Roadmap Phase 4)

## 1. 论文概要

**核心贡献**: 建立了 **RLHF (Reinforcement Learning from Human Feedback)** 的标准 pipeline, 让 GPT-3 从 "续写文本" 变成 "遵循指令" 的助手.

**历史意义**:
- ChatGPT 的技术基础
- 定义了 SFT → RM → PPO 三步对齐 pipeline
- 首次大规模证明 RLHF 可以显著改善 LLM 的有用性和安全性

## 2. 方法详解

### 2.1 三步 Pipeline

```
Step 1: SFT (Supervised Fine-Tuning)
┌───────────────────────────────────────────┐
│ 数据: ~13K (prompt, response) 对          │
│   - 人工编写的回答 (demonstrations)       │
│   - API 用户提交的 prompts               │
│   - 标注者写出理想回答                    │
│                                           │
│ 训练: 标准 cross-entropy loss             │
│   GPT-3 (175B) 在这些数据上微调           │
│   结果: SFT 模型 (已学会跟随指令格式)     │
└───────────────────────────────────────────┘
         ↓
Step 2: Reward Model Training
┌───────────────────────────────────────────┐
│ 数据: ~33K 比较 (prompt, response_A,      │
│        response_B, preference)             │
│   - SFT 模型对同一 prompt 生成多个回答     │
│   - 人工标注者比较哪个更好                 │
│                                           │
│ 训练: Bradley-Terry 偏好模型               │
│   RM(y_w > y_l) = σ(r(x,y_w) - r(x,y_l)) │
│   Loss = -E[log σ(r(x,y_w) - r(x,y_l))]  │
│                                           │
│ 架构: SFT 模型 + 去掉 unembedding         │
│       + 加一个 scalar output head          │
│   → 6B 参数 (175B 太大且不稳定)           │
└───────────────────────────────────────────┘
         ↓
Step 3: PPO (Reinforcement Learning)
┌───────────────────────────────────────────┐
│ 目标: 最大化 reward 同时不偏离 SFT 太远    │
│                                           │
│ objective = E[r(x,y)] - β·KL(π||π_ref)   │
│   其中:                                   │
│   - r(x,y) = RM 给出的标量奖励            │
│   - π_ref = SFT 模型 (参考策略)           │
│   - β = KL 惩罚系数                       │
│                                           │
│ 使用 PPO (Proximal Policy Optimization):  │
│   - 4 个模型: π, π_ref, V, RM             │
│   - rollout: π 生成回答                   │
│   - reward: RM 评估                       │
│   - update: PPO clipped surrogate loss    │
│                                           │
│ 训练数据: ~31K prompts                    │
└───────────────────────────────────────────┘
```

### 2.2 数据组成

| 数据类型 | 数量 | 用途 |
|---------|------|------|
| SFT demonstrations | ~13K | Step 1: 微调 |
| RM comparisons | ~33K | Step 2: 训练 RM |
| PPO prompts | ~31K | Step 3: RL 训练 |

**Prompt 类别分布**:
- Generation (45%): 写作、创作、摘要
- QA (25%): 问答、解释
- Brainstorming (15%): 头脑风暴、建议
- Chat (10%): 对话、闲聊
- Other (5%): 代码、数学

### 2.3 Bradley-Terry 模型 (Reward Model)

```
偏好模型假设:
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

  其中 σ(z) = 1 / (1 + e^(-z)) (sigmoid)
  r(x, y) 是 reward model 给出的标量分数

训练 loss:
  L = -E[log σ(r(x, y_w) - r(x, y_l))]

直觉:
  - y_w (preferred) 的 reward 应该高于 y_l (dispreferred)
  - 差值越大 → loss 越小 (σ 接近 1)
  - 差值越小 → loss 越大 (σ 接近 0.5)

为什么用 6B 而不是 175B?
  1. 175B RM 在训练中不稳定 (loss 震荡)
  2. 6B RM 足够准确 (人类协议率 ~73%)
  3. PPO 需要 rollout → RM 需要频繁推理 → 小模型更快
```

### 2.4 PPO 目标函数详解

```
完整目标:
  maximize E_x~D, y~π_θ [r_φ(x,y) - β · log(π_θ(y|x) / π_ref(y|x))]

第一项: reward
  - RM 给出的标量奖励
  - 希望 reward 越高越好

第二项: KL 惩罚
  - 防止策略偏离 SFT 模型太远
  - β 控制偏离程度
  - 太大: 模型不敢探索 (退化为 SFT)
  - 太小: reward hacking (生成高 reward 但无用的文本)

PPO 具体:
  L_CLIP = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]

  r_t(θ) = π_θ(y_t|·) / π_old(y_t|·)  # 当前 vs 旧策略
  Â_t = R_t - V(s_t)                     # 优势估计
  ε = 0.2 (clip range)

  GAE (Generalized Advantage Estimation):
  Â_t = Σ (γλ)^l · δ_{t+l}
  δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

## 3. 实验结果

### 3.1 人类评估

| 模型 | 有用性 | 真实性 | 无害性 |
|------|--------|--------|--------|
| GPT-3 (175B) | 22% | 37% | 39% |
| SFT (175B) | 62% | 56% | 47% |
| **InstructGPT (PPO)** | **85%** | **73%** | **74%** |

**关键**: 1.3B InstructGPT 的有用性就超过 175B GPT-3!

### 3.2 Alignment Tax

```
对齐代价: RLHF 可能降低模型的基础能力

衡量:
  - SQuAD: PPO 略低于 SFT
  - HellaSwag: PPO ≈ SFT
  - Writing: PPO > SFT (更有用)

缓解方法:
  - 混合预训练数据到 PPO (pretraining mix)
  - 在 RL 目标中加入 NLL loss
  → 基本消除 alignment tax
```

### 3.3 Scaling 效果

```
模型大小对 RM 准确率的影响:
  1.3B: 63.4%
  6B:   67.2%  ← 选这个 (性价比最优)
  175B: 68.4%

模型大小对 InstructGPT 效果:
  1.3B InstructGPT >> 175B GPT-3 (零样本)
  → 对齐比规模更重要!
```

## 4. 与现代方法的对比

| 方法 | 模型数 | 数据需求 | 优点 | 缺点 |
|------|--------|----------|------|------|
| **InstructGPT (PPO)** | 4 (π, π_ref, V, RM) | 大量人工标注 | 灵活, 强 | 复杂, 不稳定 |
| **DPO** | 2 (π, π_ref) | 偏好数据 | 简单, 稳定 | reward 利用率低 |
| **GRPO** | 2 (π, π_ref) | 规则 reward | 无 RM, 稳定 | 仅限可验证任务 |
| **RLHF (verl)** | 2-4 | 灵活 | 统一框架 | 需要集群 |

### InstructGPT → DPO 的演进

```
DPO 的核心洞察:
  Bradley-Terry 模型可以解析地消除 reward function

  r(x,y) = β log(π_θ(y|x) / π_ref(y|x)) + β log Z(x)

  代入 BT loss → 直接得到策略梯度:
  L_DPO = -E[log σ(β(log π_θ(y_w|·)/π_ref(y_w|·))
                  - β(log π_θ(y_l|·)/π_ref(y_l|·)))]

  → 不需要显式训练 RM!
  → 不需要 RL rollout!
  → 只需要偏好数据, 直接优化策略
```

## 5. 对 AI Infra 的影响

### 5.1 RLHF 的系统挑战

```
1. 多模型协调:
   PPO: 4 个模型同时需要 (Actor, Ref, Critic, RM)
   → 需要多 GPU 分配策略
   → ZeRO / FSDP 分片优化器状态

2. Rollout 服务:
   Actor 生成回答 → 需要 inference engine
   → vLLM continuous batching
   → Prefix caching (同 prompt 生成多个回答)

3. Reward 计算:
   RM 推理 → 需要 GPU
   规则 reward → 需要 CPU 沙箱

4. 训练循环:
   rollout → reward → advantage → PPO update
   → 需要 Ray 分布式调度
   → verl: ActorRolloutRefWorker 混合架构
```

### 5.2 verl 中的实现

```
verl RayPPOTrainer:
  - 统一 PPO/GRPO/REINFORCE via adv_estimator
  - core_algos.py: 2488 lines, 10+ advantage estimators
  - PrefixGrouper: GRPO n=8 prefix caching 节省 58-76% KV
  - ActorRolloutRefWorker: 混合训练+推理
  - Ray 调度: 自动 GPU 分配
```

## 6. 核心学习

1. **RLHF 是 ChatGPT 的核心技术**: SFT → RM → PPO 三步定义了行业标准
2. **对齐 > 规模**: 1.3B InstructGPT > 175B GPT-3 → 对齐是关键
3. **Bradley-Terry 模型**: 偏好学习的数学基础 → DPO 直接从这推导
4. **Reward Model 是瓶颈**: 人工标注贵, 不一致 → GRPO 用规则 reward 解决
5. **Alignment Tax**: 对齐可能降低基础能力 → 混合预训练数据缓解
6. **DPO > PPO 趋势**: 简化 RLHF, 消除 RM 和 critic → 但灵活性降低
7. **verl 统一框架**: 一个 trainer 处理所有 RL 算法 → Infra 角色的价值
