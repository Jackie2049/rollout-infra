# DPO (Direct Preference Optimization) 深度解析

> 跳过 Reward Model 和 RL，直接用偏好数据优化策略——DPO 如何简化 LLM 对齐

## 1. 背景：RLHF 的复杂性问题

### 1.1 RLHF 的四模型困境

```
RLHF (PPO):
  1. 训练 Reward Model (需要人类偏好数据)
  2. 训练 Actor + Critic (PPO)
  3. 保持 Reference Model (KL 约束)
  → 四个模型, 显存爆炸, 超参数敏感, 训练不稳定
```

### 1.2 DPO 的核心洞察

```
RLHF 的优化目标:
  max π_θ(y|x) s.t. KL[π_θ || π_ref] ≤ β  (带 KL 约束的策略优化)

DPO 的关键数学推导:
  上述问题存在闭式解:
  π_θ(y|x) / π_ref(y|x) = (1/β) × exp(r(x,y))

  → Reward 可以用策略比表示:
  r(x,y) = β × log(π_θ(y|x) / π_ref(y|x))

  → 将 r 代入 Bradley-Terry 偏好模型:
  P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))

  → 得到 DPO loss:
  L_DPO = -E[log σ(β × (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]
```

**核心简化**：不需要显式的 Reward Model！奖励函数隐含在策略比 `π_θ/π_ref` 中。

## 2. DPO 算法详解

### 2.1 训练数据

```
DPO 只需要偏好对 (preference pairs):
  x:  prompt
  y_w: 胜出回复 (chosen/preferred)
  y_l: 落败回复 (rejected)

不需要:
  ✗ Reward Model 标注分数
  ✗ Critic 估计价值函数
  ✗ PPO 的 rollout 循环
```

### 2.2 训练流程

```
1. 准备偏好数据: (prompt, chosen, rejected) 三元组
2. 加载 SFT 模型作为 π_ref (冻结)
3. 初始化 π_θ = π_ref 的副本 (可训练)
4. 对每个偏好对计算 DPO loss:
   chosen_reward  = β × log π_θ(y_w|x) - β × log π_ref(y_w|x)
   rejected_reward = β × log π_θ(y_l|x) - β × log π_ref(y_l|x)
   loss = -log σ(chosen_reward - rejected_reward)
5. 标准 SGD/Adam 更新 π_θ
```

### 2.3 直觉理解

```
DPO 在做什么?
  - 增加 chosen 回复的概率 (相对 π_ref)
  - 降低 rejected 回复的概率 (相对 π_ref)
  - 差距越大越好 (σ 函数确保饱和)

类比:
  RLHF = 先学一个评分标准 (RM) → 再用 RL 优化策略
  DPO  = 直接从偏好对比中学习，跳过评分标准

  类似于:
  RLHF: 老师先制定评分标准 → 学生根据标准练习
  DPO:  老师直接告诉学生"A 比 B 好" → 学生自己调整
```

## 3. DPO vs RLHF 对比

### 3.1 系统/工程复杂度

```
| 维度         | RLHF (PPO)          | DPO                 |
|-------------|----------------------|---------------------|
| 模型数量     | 4 (Actor+Crit+RM+Ref) | 2 (Policy+Ref)     |
| 显存需求     | ~4x 模型大小          | ~2x 模型大小        |
| 训练循环     | Rollout→Score→Train   | 单次前向+反向       |
| 超参数       | ε,β,γ,λ,lr...        | β, lr              |
| 训练稳定性   | 不稳定（需大量调参）  | 稳定（类似 SFT）    |
| 实现复杂度   | 高（需要推理引擎集成） | 低（标准训练循环）  |
```

### 3.2 质量对比

```
DPO 的优势:
  - 训练稳定，不会 reward hacking
  - 实现简单，复用 SFT 代码即可
  - 数据效率高（直接用偏好对）

DPO 的劣势:
  - 无法进行 online 数据生成（离线方法）
  - 对数据质量更敏感（坏数据直接污染策略）
  - 缺乏 token 级别的信用分配

实践结论:
  - 简单对齐场景: DPO ≥ PPO
  - 复杂推理场景: PPO/GRPO 可能更好
  - 实际中经常先用 DPO 再用 PPO 精调
```

## 4. DPO 变体

### 4.1 IPO (Identity Preference Optimization)

```
问题: DPO 偏好模型假设 Bradley-Terry，可能导致过拟合
IPO: 使用更保守的偏好模型

L_IPO = E[(chosen_reward - rejected_reward - 1/(2τ))²]

更鲁棒，不会过度放大 chosen/rejected 的差距
```

### 4.2 KTO (Kahn-Tucker Optimization)

```
问题: DPO 需要成对偏好数据 (chosen + rejected)
KTO: 只需要单个回复 + 好/坏标签

数据: (prompt, response, good/bad label)
优势: 数据收集更容易（不需要配对）

loss:
  good: -log σ(β × (log π_θ/π_ref - z_good))
  bad:  -log σ(β × (z_bad - log π_θ/π_ref))

z_good, z_bad 是可调阈值
```

### 4.3 ORPO (Odds Ratio Preference Optimization)

```
将 SFT loss 和偏好优化合并:

L_ORPO = L_SFT + λ × L_OR

L_OR = -log σ(log(odds_ratio))
odds_ratio = (π_θ(y_w)/π_θ(y_l)) / (π_ref(y_w)/π_ref(y_l))

不需要单独的 Reference Model！只需要一个模型
→ 更省显存
```

### 4.4 SimPO (Simple Preference Optimization)

```
去掉 Reference Model:
  用 response length 归一化的 log probability 代替 π_ref

reward = (1/|y|) × log π_θ(y|x)

L_SimPO = -log σ(β × (reward_chosen - reward_rejected) - γ)

γ 是奖励差距阈值（防止 chosen/rejected 太接近时仍强行优化）
```

### 4.5 Online DPO

```
将 DPO 从离线转为在线:
  1. 用当前策略生成两个回复
  2. 用 RM 或 LLM-as-judge 排序
  3. 用 DPO loss 更新

好处: 数据始终来自当前策略，避免分布偏移
代价: 需要在线生成和判断，但仍比 PPO 简单
OpenRLHF 已实现
```

## 5. DPO 的实践要点

### 5.1 数据质量

```
DPO 对数据质量比 RLHF 更敏感:

好的数据:
  - chosen 明显优于 rejected（有信息量）
  - 覆盖多种场景和任务类型
  - rejected 不是完全垃圾（需要有一定的挑战性）

差的数据:
  - chosen 和 rejected 差异太小 → 模型学不到信号
  - rejected 是随机噪声 → 模型只学会避免噪声
  - 偏好标注不一致 → 模型学到矛盾信号
```

### 5.2 超参数

```
β (KL 惩罚系数):
  - β 太小: 策略偏离 π_ref 太远 → 可能不连贯
  - β 太大: 策略几乎不动 → 学不到偏好
  - 典型值: 0.1 - 0.5
  - 建议从 0.1 开始，逐步增大

学习率:
  - 通常 1e-6 ~ 5e-7（比 SFT 小）
  - DPO 容易过拟合，建议早停

训练轮数:
  - 1-3 epochs 通常足够
  - 监控 chosen/rejected reward 的差距
```

### 5.3 与 LoRA 结合

```
DPO + LoRA:
  π_ref: 冻结的 SFT 模型（共享权重）
  π_θ: SFT 模型 + LoRA adapter

  只训练 LoRA 的 A, B 矩阵
  显存需求: 模型权重(冻结) + LoRA(~0.1%) + 优化器状态

  7B 模型 DPO + LoRA (r=16):
    LoRA 参数: ~5 MB
    总额外显存: ~50 MB (含优化器)
    → 单卡即可
```

## 6. 关键要点

1. **DPO 跳过 Reward Model 和 RL** — 直接从偏好对优化策略，将四模型简化为两模型
2. **数学等价性** — DPO 在理论上与 RLHF 有相同的优化目标，只是用闭式解代替了 RL 求解
3. **实现极其简单** — 只需要标准训练循环 + DPO loss，不需要推理引擎集成
4. **数据质量是关键** — 离线方法无法在线生成数据，偏好对的质量直接决定效果
5. **变体不断演进** — IPO/KTO/ORPO/SimPO 在不同维度简化或改进 DPO，KTO 不需要配对数据

## 参考

- 论文: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (DPO 原始论文)
- 论文: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (IPO)
- 论文: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- 论文: [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)
- 论文: [SimPO: Simple Preference Optimization](https://arxiv.org/abs/2405.14734)
- 博客: [The N implementers' guide to DPO](https://huggingface.co/blog/dpo-impl-notes)
