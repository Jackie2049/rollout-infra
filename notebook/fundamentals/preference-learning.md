# Preference Learning & Bradley-Terry Model

> 基础知识 | 2026-06-05
> 关联论文: InstructGPT, DPO, RLHF

## 1. 偏好学习问题

```
输入: 人类对两个回复的偏好比较 (y1 > y2 | x)
输出: 能预测人类偏好的奖励函数 R(x, y)

核心假设: Bradley-Terry 模型
  P(y1 ≻ y2 | x) = σ(R(x,y1) - R(x,y2))

其中 σ 是 sigmoid 函数
```

## 2. Bradley-Terry 模型

### 2.1 历史背景

```
Bradley-Terry (1952):
  原始用途: 体育比赛胜负预测
  核心思想: 每个 "选手" 有一个实力分数 β_i
  P(选手i 胜 选手j) = β_i / (β_i + β_j)

现代 NLP 应用:
  选手 = 给定 prompt x 的两个回复 y1, y2
  实力分数 = 奖励模型 R(x, y)
  P(y1 ≻ y2 | x) = exp(R(x,y1)) / (exp(R(x,y1)) + exp(R(x,y2)))
                  = σ(R(x,y1) - R(x,y2))
```

### 2.2 Reward Model 训练

```
训练数据: D = {(x_i, y_w, y_l)} where y_w ≻ y_l (y_w 是人类选择的)

损失函数 (负对数似然):
  L(R_φ) = -E_{(x,y_w,y_l)~D} [log σ(R_φ(x,y_w) - R_φ(x,y_l))]

等价于: 二元交叉熵 (Binary Cross-Entropy)
  - label = 1 (y_w 被选择)
  - logit = R_φ(x,y_w) - R_φ(x,y_l)

梯度:
  ∂L/∂R_φ(x,y_w) = -(1 - σ(R_w - R_l))
  ∂L/∂R_φ(x,y_l) = (1 - σ(R_w - R_l))

→ R_w 增大, R_l 减小, 直到差距足够大
```

### 2.3 理论性质

```
1. 对称性: P(A≻B) + P(B≻A) = 1 (完备概率)
2. 传递性 (近似): A≻B, B≻C → A≻C (但不是严格保证)
3. 可加性: R 的常数偏移不影响偏好概率
   → 奖励函数只在不同回复的相对大小上有意义

局限性:
  - 假设偏好是确定的 (无噪声), 实际人类偏好有随机性
  - 不建模偏好强度 (只建模方向, 不建模 "好多少")
  - 平局 (tie) 无法处理
```

## 3. 从 Reward Model 到 RLHF/DPO

### 3.1 RLHF 路径 (间接)

```
1. 训练 Reward Model R_φ (用 Bradley-Terry loss)
2. 用 R_φ 作为奖励信号, PPO 训练 LLM 策略

PPO 目标:
  max E[R_φ(x,y)] - β * KL(π_θ || π_ref)

问题: 需要在线采样 → 训练不稳定 → reward hacking
```

### 3.2 DPO 路径 (直接)

```
关键洞察: Bradley-Terry + KL约束的封闭解

RL 目标: max E[R(x,y)] - β * KL(π_θ || π_ref)
最优解: π*(y|x) ∝ π_ref(y|x) * exp(R(x,y)/β)

推导:
  R(x,y) = β * log(π*(y|x) / π_ref(y|x))

代入 Bradley-Terry:
  P(y_w≻y_l|x) = σ(β * log(π*(y_w|x)/π_ref(y_w|x))
                  - β * log(π*(y_l|x)/π_ref(y_l|x)))

DPO 损失函数:
  L_DPO(θ) = -E[log σ(β * (log(π_θ(y_w|x)/π_ref(y_w|x))
                            - log(π_θ(y_l|x)/π_ref(y_l|x))))]

关键: 不需要显式训练 Reward Model!
  直接用策略自身的 log-probability 比来隐式表示奖励
```

### 3.3 对比

```
           RLHF               DPO
模型数量    4 (Actor+Critic+Ref+RM)  2 (Policy+Ref)
训练方式   在线 (PPO采样)         离线 (静态数据)
稳定性     低 (reward hacking)     高 (无在线交互)
灵活性     高 (奖励可动态调整)      低 (固定偏好数据)
理论保证   无 (RL 不保证收敛)       有 (BT 模型封闭解)

实际使用:
  - ChatGPT/Claude: RLHF (PPO) — 更灵活, 持续迭代
  - 开源模型: DPO 更流行 — 简单, 稳定, 不需要 RL
  - 混合: 先 DPO 打底, 再 RLHF 微调 (verl 支持)
```

## 4. 核心学习

1. **Bradley-Terry 是偏好学习的数学基础**: 将比较偏好转化为概率模型
2. **Reward Model = 二元分类器**: 训练目标是区分 "chosen" vs "rejected"
3. **DPO 的优雅**: 将 RL 问题转化为监督学习 — 不需要 RL 训练循环
4. **偏好数据质量 > 算法选择**: 好的偏好标注比算法改进更重要
5. **KL 约束是关键**: 没有 KL → reward hacking; KL 太强 → 不学
6. **温度 β 控制 "保守程度"**: β 大 → 更保守 (接近 reference); β 小 → 更激进
