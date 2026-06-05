# Paper Reading: DPO — Direct Preference Optimization

> Rafailov et al., NeurIPS 2023 | Stanford
> 精读日期: 2026-06-05
> 优先级: P1 (AI Expert Roadmap Phase 4)

## 1. 论文概要

**核心贡献**: 将 RLHF 的 reward model + PPO 两步合并为一步, 直接用偏好数据优化策略, **不需要训练 reward model, 不需要 RL**.

**关键洞察**: Bradley-Terry 偏好模型可以解析地消除 reward function, 得到 closed-form 的最优策略.

**影响**: 大幅简化对齐流程, 成为许多项目 (Zephyr, Tulu, etc.) 的首选方法.

## 2. 核心数学推导

### 2.1 从 KL-constrained RL 到 DPO

```
Step 1: KL-constrained reward maximization
  max_π E_x~D,y~π[r(x,y) - β·KL(π(·|x) || π_ref(·|x))]

  最优解 (closed form):
  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

  其中 Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β) (配分函数)

Step 2: 反解 reward
  π*(y|x) / π_ref(y|x) = exp(r(x,y)/β) / Z(x)
  r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

  → reward 可以用策略的 log-ratio 表示!

Step 3: 代入 Bradley-Terry 模型
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

  r(x, y_w) - r(x, y_l)
  = β[log(π*(y_w|x)/π_ref(y_w|x)) - log(π*(y_l|x)/π_ref(y_l|x))]
  = β · log[π*(y_w|x)π_ref(y_l|x) / (π*(y_l|x)π_ref(y_w|x))]

  注意: Z(x) 被消掉了! 这是 DPO 的关键!

Step 4: DPO 目标函数
  L_DPO(θ) = -E_(x,y_w,y_l) [log σ(β · log(π_θ(y_w|x)/π_ref(y_w|x))
                                - β · log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

### 2.2 DPO 梯度分析

```
对 π_θ 求梯度:

∂L/∂θ = -β · E[σ(·) · (1-σ(·)) · ∂/∂θ(h_θ(x, y_w, y_l))]

其中 h_θ = log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))

展开:
  ∂L/∂θ ∝ σ(r̂) · [∇_θ log π_θ(y_l|x) - ∇_θ log π_θ(y_w|x)]

直觉:
  - 增加 y_w (preferred) 的概率
  - 减少 y_l (dispreferred) 的概率
  - 调整幅度由 σ(r̂) 控制:
    - 如果模型已经正确区分 (r̂ 大) → σ ≈ 1 → 梯度大 → 强更新
    - 如果模型区分困难 (r̂ ≈ 0) → σ ≈ 0.5 → 温和更新

  与 PPO 的区别:
  PPO: 需要估算 advantage → 需要 critic → 需要额外模型
  DPO: 直接比较 y_w vs y_l → 不需要 critic → 不需要额外模型
```

## 3. 实验结果

### 3.1 对话任务 (Reddit TL;DR)

| 方法 | Reward | 人类胜率 |
|------|--------|----------|
| SFT | -1.35 | 26% |
| PPO (RLHF) | -0.56 | 54% |
| **DPO** | **-0.47** | **58%** |
| Preferred-SFT | -0.62 | - |

### 3.2 NLP Benchmark

| Method | NLP Benchmark Avg |
|--------|------------------|
| GPT-J (6B) | 61.3 |
| SFT | 62.5 |
| PPO | 63.1 |
| **DPO** | **64.1** |

### 3.3 效率对比

| 特性 | PPO | DPO |
|------|-----|-----|
| 需要训练 RM | 是 | **否** |
| 需要 Critic | 是 | **否** |
| 需要 RL 循环 | 是 | **否** |
| 模型数量 | 4 | **2** |
| GPU 内存 | 高 | **低** |
| 训练稳定性 | 需要调参 | **稳定** |
| 数据需求 | RM + RL prompts | **仅偏好数据** |

## 4. DPO 的局限和改进

### 4.1 已知局限

```
1. Reward 利用率低:
   PPO: 可以迭代 rollout → 探索 reward landscape
   DPO: 只用给定的偏好对 → 不能主动探索

2. 偏好数据质量要求高:
   PPO: RM 可以从噪声数据学到合理 reward
   DPO: 坏偏好直接导致坏策略

3. Length exploitation:
   DPO 模型倾向于生成更长的文本
   → 需要 length normalization 或 penalty

4. 不适合复杂 reward:
   规则 reward (如数学正确性) 难以用偏好对表示
   → GRPO 更适合
```

### 4.2 改进方向

```
1. IPO (Identity Preference Optimization):
   不用 Bradley-Terry 假设, 用更一般的偏好模型

2. KTO (Kahneman-Tversky Optimization):
   不需要成对偏好, 只需 good/bad 标签
   → 数据收集更简单

3. ORPO (Odds Ratio Preference Optimization):
   将 SFT 和偏好学习合并为一步

4. SimPO (Simple Preference Optimization):
   用 sequence-level average log probability 替代 reference model
   → 不需要 π_ref, 更简单
```

## 5. 实践经验 (从 DPO 训练 Simulator 验证)

| 论文发现 | 实测验证 |
|---------|---------|
| DPO 收敛快 | ✅ loss 0.62→0.14, 99.7% accuracy |
| β=0.3 最优 | ✅ β=0.3 → 100% acc, margin=4.05 |
| Length normalization 有效 | ✅ ↓42% loss |
| Alignment tax 存在 | ✅ NLL 1.17→1.44 |
| 需要 2 个模型 | ✅ π + π_ref, 295MB |

### DPO 训练实操 (RTX 4090 实测)

```
模型: 838K params MiniGPT
数据: 合成偏好对 (good/bad responses)
训练: 3步 (Pretrain → SFT → DPO)

结果:
  DPO loss: 0.62 → 0.14 (10 epochs)
  Accuracy: 65% → 99.7%
  Margin: 0.5 → 2.49
  Speed: 229K tok/s, 295MB

最优参数:
  β = 0.3 (balance alignment vs diversity)
  Length normalization: ON (防止长度利用)
  Learning rate: 1e-4 (比 SFT 小 3x)
```

## 6. 对 AI Infra 的影响

```
DPO 对训练系统的影响:

1. 简化 Infra:
   PPO: Actor + Ref + Critic + RM = 4 个模型
   DPO: Policy + Ref = 2 个模型
   → 内存需求减半
   → 不需要 rollout engine

2. 数据 Pipeline:
   PPO: 需要 prompt → rollout → RM 评分 → RL update 循环
   DPO: 一次性偏好数据 → 直接训练
   → 更适合离线批量处理

3. 训练速度:
   PPO: 需要多次 rollout-iteration 循环
   DPO: 单次 forward (类似 SFT)
   → 训练速度 2-3x

4. 但 GRPO 可能是更好的选择:
   推理任务: GRPO (规则 reward, 不需要偏好数据)
   通用对齐: DPO (偏好数据, 简单稳定)
   最高性能: PPO (最灵活, 但最复杂)
```

## 7. 核心学习

1. **DPO 的数学之美**: 从 BT 模型解析消除 reward → closed-form 最优策略
2. **RL 不是必须的**: 偏好学习可以通过监督学习完成 (只要数学形式正确)
3. **DPO vs PPO 是 trade-off**: 简单性 vs 灵活性, 稳定性 vs reward 利用率
4. **β 是关键超参**: 控制偏离 reference 的程度 (类似 KL 惩罚系数)
5. **Length exploitation**: DPO 的已知问题, 需要 normalization
6. **现代趋势**: GRPO (推理) + DPO (对齐) 组合使用, 如 DeepSeek-R1 pipeline
