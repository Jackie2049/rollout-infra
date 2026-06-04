# DPO / RLHF 对齐算法详解

> 对齐 (Alignment) 是 LLM 训练的最后一步: 让模型行为符合人类偏好
> 关键文件: `tools/dpo_train_pipeline.py` (从零实现的 DPO 训练)

## 1. 对齐方法谱系

```
LLM 训练流程:
  Pretraining (next-token prediction)
  → SFT (instruction following)
  → Alignment (human preferences)  ← 我们在这里
      ├── RLHF (PPO): 需要 Reward Model + RL 训练循环
      ├── DPO: 直接从偏好对学习, 无需 RM
      ├── GRPO: 简化版 PPO, 无需 Critic
      ├── RLAIF: 用 AI 代替人类标注偏好
      └── Constitutional AI: 自我对齐
```

## 2. RLHF (Reinforcement Learning from Human Feedback)

### 2.1 传统 RLHF 三步流程

```
Step 1: 收集偏好数据
  对同一 prompt, 生成多个 response
  人类标注: response_A > response_B (偏好排序)

Step 2: 训练 Reward Model (RM)
  RM(prompt, response) → scalar reward
  Bradley-Terry 模型: P(y1 > y2) = σ(r(x,y1) - r(x,y2))
  Loss: -E[log σ(r(x,y_w) - r(x,y_l))]

Step 3: PPO 训练
  Actor: π_θ(y|x) — 策略模型
  Critic: V_φ(x,y) — 价值函数
  Reference: π_ref(y|x) — SFT 冻结模型 (KL 约束)
  Reward: r(x,y) — 步骤2训练的 RM

  Objective: max E[r(x,y) - β·KL(π_θ || π_ref)]
  PPO clip: L_clip = min(r_t·A_t, clip(r_t,1-ε,1+ε)·A_t)
```

### 2.2 RLHF 的问题

1. **4 个大模型**: Actor + Critic + Reference + Reward Model → 显存爆炸
2. **训练不稳定**: RL 本身不稳定, reward hacking, mode collapse
3. **超参敏感**: KL 系数, clip ε, GAE λ/γ, reward scaling
4. **工程复杂**: rollout → reward → advantage → actor update → critic update → sync

## 3. DPO (Direct Preference Optimization)

### 3.1 核心思想

DPO 的关键洞察: RLHF 的最优策略有闭式解。

在 RLHF 中, 最优策略是:
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```

反解 reward:
```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

代入 Bradley-Terry 偏好模型, Z(x) 被消去:
```
P(y_w > y_l | x) = σ(β · [log(π*(y_w|x)/π_ref(y_w|x)) - log(π*(y_l|x)/π_ref(y_l|x))])
```

直接优化策略参数 θ:
```
L_DPO = -E[log σ(β · (log(π_θ(y_w|x)/π_ref(y_w|x))
                        - log(π_θ(y_l|x)/π_ref(y_l|x))))]
```

### 3.2 DPO 实现细节

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # Log-ratio: 策略偏离参考的程度
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # 隐式奖励 (用于监控)
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_rewards, rejected_rewards
```

### 3.3 超参数 β 的影响

**β 控制策略偏离参考模型的程度:**

| β | 行为 | 适用场景 |
|---|------|---------|
| 0.05 | 温和对齐, 接近 SFT | 偏好数据质量低, 避免过拟合 |
| 0.1 | 标准对齐 (论文默认) | 通用场景 |
| 0.3 | 强对齐, 快速收敛 | 偏好明确, 需要显著行为变化 |
| 0.5 | 很强对齐 | 对齐需求极端 |
| 1.0 | 极强对齐, 可能 reward hack | 需谨慎, 可能降低生成质量 |

**实测结论 (RTX 4090, 838K model)**:
- β=0.3: 100% accuracy, margin=4.05 — **推荐**
- β=1.0: loss 最低 (0.018), 但 margin 过大 (9.04) 可能过拟合

### 3.4 Length Normalization

标准 DPO 使用完整序列的 log-prob 之和:
```
log π(y|x) = Σ_t log π(y_t | x, y_<t)
```

长序列的 log-prob 绝对值更大, 导致 loss 被长序列主导。

**Length-normalized DPO**:
```
log π(y|x) = (1/|y|) Σ_t log π(y_t | x, y_<t)
```

**实测**: length normalization 降低 loss 42% (0.081 → 0.046), 对变长 response 场景尤其重要。

## 4. GRPO (Group Relative Policy Optimization)

### 4.1 GRPO 简化

GRPO 是 DeepSeek 提出的简化版 RLHF:
- 移除 Critic (不需要价值函数)
- 用组内相对 reward 替代绝对 reward
- 同一 prompt 采样 n 个 response, 组内排名

```
对 prompt x, 采样 n 个 response {y_1, ..., y_n}
计算 reward r(x, y_i) (规则或 RM)
归一化: Ẽ_i = (r_i - mean(r)) / std(r)
GRPO loss: -E[Ẽ_i · log π_θ(y_i|x)]
```

### 4.2 GRPO vs DPO vs PPO

| Aspect | PPO | DPO | GRPO |
|--------|-----|-----|------|
| Models | 4 | 2 | 2-3 |
| Online/Offline | Online | Offline | Online |
| Reward Model | 需要 | 不需要 | 可选 |
| Critic | 需要 | 不需要 | 不需要 |
| 稳定性 | 低 | 高 | 中 |
| 数据 | prompt + reward | 偏好对 | prompt + 多response |
| 代表实现 | verl/OpenRLHF | TRL | verl/DeepSeek |
| verl 支持 | ✅ | ✅ | ✅ |

## 5. 实战经验总结

### 5.1 从 Pretrain → SFT → DPO 的完整流程

```
1. Pretrain: 学习语言规律 (next-token prediction)
   - 数据: 大量无标注文本
   - Loss: cross-entropy
   - 耗时最长 (7B 模型 ~36天 A100×256)

2. SFT: 学习指令跟随
   - 数据: (instruction, response) 对
   - Loss: cross-entropy (只对 response 部分)
   - 相对快 (几千条数据, 几小时)

3. DPO: 学习人类偏好
   - 数据: (prompt, chosen, rejected) 偏好对
   - Loss: DPO loss (log-sigmoid)
   - 最快 (几百到几千条数据, 几十分钟)
   - ⚠️ 必须从 SFT checkpoint 开始, 不能从 pretrain 开始!
```

### 5.2 关键教训

1. **SFT 是 DPO 的前提**: DPO 需要策略已经能生成合理 response, 否则 log-ratio 无意义
2. **Reference model 必须冻结**: 它提供 KL 约束, 防止策略退化
3. **β 需要调参**: 太小对齐不够, 太大过拟合
4. **Length normalization 对变长 response 重要**: 42% loss 降低
5. **Alignment tax**: DPO 后 SFT loss 可能略增 (1.17→1.44), 这是对齐的代价
6. **小模型对齐有上限**: 838K 模型能学到偏好 (99.7% accuracy) 但生成质量受限

### 5.3 扩展到真实模型

| Model | DPO Memory | 训练时间 (A100) | 偏好数据量 |
|-------|-----------|----------------|-----------|
| 838K (本次) | 295 MB | <1 min | 300 pairs |
| 7B | ~28 GB (2x model) | ~2 hrs | 10K pairs |
| 70B | ~280 GB | ~20 hrs (8×A100) | 100K pairs |
| 405B | ~1.6 TB | ~5 days (64×A100) | 500K pairs |
