# Paper Reading: PPO — Proximal Policy Optimization Algorithms

> Schulman et al., 2017 (OpenAI) | arXiv: 1707.06347
> 精读日期: 2026-06-05
> 优先级: P0 (RLHF/ChatGPT 的核心算法)

## 1. 论文概要

**核心贡献**: 用简单的 clipping 操作实现 TRPO 级别的训练稳定性, 但只需一阶优化 (SGD/Adam).

**解决的问题**: Vanilla Policy Gradient 训练不稳定 — 单步大梯度可能摧毁好策略.

**影响**: 成为 RLHF 的标准算法 (InstructGPT, ChatGPT, Claude).

## 2. 核心公式: PPO-Clip

```
L^{CLIP}(θ) = E_t [ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]
```

其中:
- `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)` — 新旧策略概率比
- `Â_t` — 优势函数估计 (advantage)
- `ε` — clip 范围 (典型 0.1-0.2)

### Clipping 机制详解

```
Case 1: Â_t > 0 (好动作, 想增加概率)
  → r_t 应增大, 但 clip 限制在 [1-ε, 1+ε]
  → r_t > 1+ε 时梯度为零 → 不再鼓励概率继续增大
  → 防止 "好动作概率被推到极端"

Case 2: Â_t < 0 (差动作, 想减少概率)
  → r_t 应减小, 但 clip 限制在 [1-ε, 1+ε]
  → r_t < 1-ε 时梯度为零 → 不再鼓励概率继续减小
  → 防止 "差动作概率被压到极端"

关键: clip 不硬约束 r_t 的范围, 而是移除优化激励 (零梯度)
→ 这是 Schulman 称为 "pessimistic lower bound" 的原因
```

## 3. 为什么需要 PPO? (进化路线)

```
Vanilla PG (REINFORCE)
  问题: 高方差, 无策略变化约束, 学习率敏感
  → 一步大更新可能摧毁好策略 (catastrophic forgetting)

TRPO (2015, Schulman)
  改进: 硬 KL 约束 D_KL(π_old || π_new) ≤ δ
  方法: 共轭梯度 + Fisher 信息矩阵 (二阶优化)
  问题: 计算昂贵, 难以扩展到大模型 (LLM)

PPO-Clip (2017, Schulman)
  改进: 用 clip 创建隐式 trust region
  方法: 标准 SGD/Adam (一阶优化)
  优势: 简单 + 快速 + 可扩展到数十亿参数
  代价: 放弃 TRPO 的理论单调改进保证
  实际: 经验上 PPO ≥ TRPO 性能
```

## 4. GAE (Generalized Advantage Estimation)

```
Â_t^{GAE(γ,λ)} = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}^V

其中 TD 残差: δ_t^V = r_t + γV(s_{t+1}) - V(s_t)

λ 的 bias-variance 权衡:
  λ=0:  Â_t = δ_t^V  (低方差, 高偏差 — 单步 TD)
  λ=1:  Â_t = MC return - V(s_t)  (高方差, 低偏差)
  λ=0.95: 实用折中
```

## 5. 完整 PPO 损失函数

```
L(θ) = -L^{CLIP}(θ) + c_1 * L^{VF}(θ) - c_2 * S[π_θ]

三项:
  1. L^{CLIP}: 策略梯度 (带 clip)
  2. L^{VF} = (V_θ(s_t) - V_t^{target})²: Value function MSE
  3. S[π_θ] = -Σ π(a|s) log π(a|s): 熵奖励 (鼓励探索)
```

**超参数**:
| 参数 | 符号 | 典型值 |
|------|------|--------|
| Clip ratio | ε | 0.1-0.2 |
| Discount | γ | 0.99 |
| GAE lambda | λ | 0.95 |
| Value loss 权重 | c_1 | 0.5 |
| 熵权重 | c_2 | 0.01 |
| 学习率 | α | 3e-4 |

## 6. PPO 在 RLHF 中的适配

```
RLHF Pipeline:
  Step 1: SFT → π^{SFT} (监督微调)
  Step 2: Reward Model → R_φ(x,y) (奖励模型)
  Step 3: PPO → π_θ (策略优化)

RLHF 目标:
  max E_{x~D, y~π_θ}[R_φ(x,y) - β * KL(π_θ || π^{ref})]

关键适配:
  1. 策略 = LLM: π_θ(y|x) 生成回复
  2. Reward = R_φ(x,y) 替代环境奖励
  3. KL penalty = -β * KL(π_θ || π^{ref}) 防止 reward hacking
  4. 每个 episode = 一条 prompt + response (单轮生成)
  5. Value function = 独立的 value head 或 value model
  6. 奖励白化 (whitening) 跨 batch 归一化

token 级奖励:
  r_t = R_φ(x,y) - β * (log π_θ(y_t|...) - log π^{ref}(y_t|...))
  Reward model 分数给到序列末尾, KL penalty 逐 token 计算
```

## 7. GRPO vs PPO (现代对比)

```
PPO (InstructGPT/ChatGPT):
  - 需要价值函数 (Value Model) — 多一个模型要训
  - 复杂: clip + GAE + value loss + entropy + KL
  - 4 个模型: Actor + Critic + Ref + Reward

GRPO (DeepSeek-R1):
  - 无需价值函数! 用组内归一化替代
  - 简单: advantage = (R_i - mean(R)) / std(R)
  - 3 个模型: Actor + Ref + Reward (无 Critic)
  - 训练速度更快, 内存更省
```

## 8. 核心学习

1. **Clip 是核心创新**: 简单但有效 — 零梯度 > 硬约束
2. **一阶优化即可**: 不需要二阶 (TRPO 的 Fisher 矩阵), SGD/Adam 就行
3. **RLHF 的基石**: PPO 的可扩展性使其成为 LLM 对齐的默认选择
4. **GAE 的 bias-variance**: λ=0.95 是实用折中, 不是理论最优
5. **KL penalty 必要**: 没有 KL 约束 → reward hacking (模型找 reward model 漏洞)
6. **超参数敏感性**: ε=0.2 是经验值, 不同任务可能需要调整
7. **GRPO 简化 PPO**: 去掉价值函数, 但保持 clip 机制 → 更适合 LLM 场景
