# Reward Function Design — RL训练的关键设计决策

> 2026-06-07 | 奖励函数设计对GRPO训练的影响: 稀疏vs密集vs塑形vs课程

## 概述

奖励函数是RL训练的核心接口——它定义了"什么是好"。不同奖励函数设计对训练收敛速度、稳定性和泛化有深远影响。本文从理论和实验角度分析4种奖励设计。

## 一、奖励函数设计的4种范式

### 1.1 Binary (稀疏奖励)

```
r(y) = 1 if correct, 0 otherwise

优点: 目标明确 → 正确=1, 错误=0 → 无歧义
缺点: 稀疏信号 → 大多数response reward=0 → 梯度信号弱
  → GRPO组归一化: 所有reward=0 → A_i=(0-0)/0 → undefined!
  → 只在少数response正确时才有梯度 → 学习慢

类比: 学生只被告知"对或错" → 不知道为什么错 → 难改进

→ DeepSeek-R1使用outcome-only reward(本质是binary) → 但大模型(n=64)采样足够 →
  好response概率>0 → 总有梯度信号 → 小模型n=8 → 采样太少 → binary困难
```

### 1.2 Graded (密集奖励)

```
r(y) = 1.0 if correct, 0.3 if ±1, 0.1 if ±2, 0 otherwise

优点: 密集信号 → 即使不正确也有reward → 梯度信号更强
  → GRPO组内: reward有分布 → A有差异 → 更有效梯度
缺点: 次优奖励 → ±1也得到奖励 → 可能学到"接近但不精确"
  → reward hacking风险: 模型可能稳定在±1 → reward=0.3 → 但不是最优

类比: 考试给部分分 → 学生可能满足于"接近正确" → 不追求完全正确

→ 我们的实验默认使用graded → 这是之前GRPO 79%/SFT→GRPO 93%的reward函数
```

### 1.3 Shaped (塑形奖励)

```
r(y) = 1.0 if correct, 0.5 if ±1, 0.3 if ±2, 0.2 if any digit

最密集的信号 → 即使生成随机数字也有0.2的奖励
→ 探索奖励: 0.2鼓励模型生成数字(而非垃圾token) → 引导探索方向

优点: 最大化梯度信号 → 每个response都有非零reward → 无zero-gradient问题
缺点: reward hacking风险最高 → 模型可能稳定在"生成任何数字"(0.2) → 不追求正确

类比: 每次尝试都有奖励 → 学生可能满足于"做了"而非"做对了"

→ 塑形奖励来自reward shaping理论(Ng et al., 1999):
  F(s,a,s') = r(s,a,s') + γΦ(s') - Φ(s)
  → 如果Φ是potential function → F与r等价(不改变最优策略)
  → 但我们不是严格reward shaping → 可能改变最优策略!
```

### 1.4 Curriculum (课程学习)

```
阶段1(0-33%): shaped reward → 引导模型探索 → 学到"生成数字"
阶段2(33-67%): graded reward → 提高要求 → 学到"接近正确"
阶段3(67-100%): binary reward → 严格要求 → 只奖励正确答案

优点: 渐进学习 → 先探索 → 再细化 → 最后精确 → 类似人类学习
缺点: 需要设计阶段切换时间 → 过早切换 → 模型还没探索好 → 过晚 → reward hacking
  → 切换点不准确 → 可能退步

类比: 先教加法概念(部分分) → 再要求精确(只对错)
```

## 二、Reward Shaping理论

### 2.1 Ng et al. (1999) — 塑形不改变最优策略的条件

```
定理: 如果F(s,a,s') = r(s,a,s') + γΦ(s') - Φ(s)
  则最优策略π*_F = π*_r → 塑形不改变最优策略!

→ 证明: Q_F(s,a) = Q_r(s,a) + γΦ(s') - Φ(s)
  → V_F(s) = V_r(s) + γΣπ(a|s)Φ(s') - Φ(s)
  → 最优策略只取决于Q的相对排序 → Φ只是偏移 → 不影响排序

→ 但我们的shaped reward不是严格reward shaping!
  shaped: r=0.2 for "any digit" → 这不是F = r + γΦ' - Φ的形式
  → 0.2是额外的bonus → 改变了最优策略的定义!
  → 可能导致模型"满足"于0.2 → 不追求1.0 → reward hacking!
```

### 2.2 潜在的Reward Hacking

```
Reward hacking = 模型学到取巧策略而非正确策略

binary reward: hacking = 生成某个固定数字 → 有时碰巧正确 → reward≈0.2
  → 但大多数时候错误 → reward低 → hacking不有效 → GRPO会纠正

shaped reward: hacking = 生成任何数字 → reward=0.2 → 策略稳定在0.2
  → GRPO组归一化: all reward=0.2 → A=(0.2-0.2)/σ → σ≈0 → 无梯度!
  → 模型"满足"于0.2 → 不探索更高的reward → 学习停止!

→ 关键: shaped reward可能让模型更快开始学习
  但也可能让模型更早停止学习 → 在非最优解停滞!
```

## 三、RTX 4090实验结果 — 意外发现!

```
所有4种reward函数都达到100% eval!

Reward Function | Eval Acc | Peak Reward | Final Reward
----------------|----------|-------------|-------------
binary          | 100%     | 1.0         | 1.0
graded          | 100%     | 1.0         | 1.0
shaped          | 100%     | 1.0         | 1.0
curriculum      | 100%     | 1.0         | 1.0

→ 为什么? SFT warmup已经达到100% eval → GRPO phase无学习(zero-gradient)!
→ 与SFT→DAPO实验一致: 模型已经完美 → 无改进空间 → reward设计不重要!

→ 但这次SFT达到100%(之前只有50%) → 因为lr=2e-3(之前lr不同)
→ SFT阶段的好坏决定了后续一切!
```

### 3.1 核心发现: 当SFT完美时, reward设计无关紧要

```
推理链:
1. SFT warmup → 100% eval → 模型学到了正确算法
2. GRPO phase → 所有response都正确 → reward=1.0 for ALL
3. 组归一化 → (1.0 - 1.0) / 0 → undefined → 无梯度
4. 无论reward是binary/graded/shaped → 都得到reward=1.0 → 无差别

→ reward hacking不可能 → 因为模型已经完美 → 无"取巧空间"
→ reward shaping无效果 → 因为已经到达最优 → 无"改善空间"

→ 结论: reward function design只在模型不完美时才重要!
  → 模型不完美时 → 不同reward给出不同的梯度信号 → 收敛速度不同
  → 模型完美时 → 所有reward给相同信号(全部正确) → 无差异

→ 但"模型完美"需要好的SFT → 回到核心结论: SFT warmstart是最关键决策!
```

### 3.2 需要补充实验: 无SFT暖启动

```
当前实验: SFT→GRPO → 所有reward达到100% → reward设计不重要

需要: 纯GRPO(无SFT) → reward设计才有差异!
  → GRPO + binary: 稀疏信号 → 收敛慢 → 可能更低eval
  → GRPO + graded: 密集信号 → 收敛中等 → 可能中等eval
  → GRPO + shaped: 最密集信号 → 可能最快但可能hacking
  → GRPO + curriculum: 可能最佳权衡

→ 但之前实验已经显示纯GRPO的eval低(52%) → reward设计可能影响52→60?
→ 需要进一步实验验证!
```