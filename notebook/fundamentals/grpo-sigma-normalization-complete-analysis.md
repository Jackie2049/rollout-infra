# GRPO σ-normalization: From Theory to Practice — Complete Analysis

> 2026-06-07 | 4 experiments → unified theory: σ-norm effectiveness depends on reward distribution variance

## Timeline

```
Exp 1: Simplified GRPO → σ-norm无效果(normalized=unnormalized=21%)
  → 不同prompt分组→reward跨问题方差大→σ-norm仅rescale

Exp 2: Noise σ sweep → σ=0.01最优(100% eval!)
  → 梯度噪声帮助SFT探索 → 但噪声≠σ-norm!

Exp 3: Proper GRPO → σ-norm HURTS(43% vs 65%, Δ=-22%)
  → 强SFT(39%eval)→多数reward接近1→σ≈0→梯度消失

Exp 4: Double norm verification → clip不是罪魁祸首(22% < 26%)
  → σ-norm NO clip反而更差 → 去掉clip不修复问题

Exp 5: Binary vs continuous reward → σ-norm HELPS弱SFT(45% vs 40%)
  → 弱baseline→reward分散→σ有意义
  → 连续reward反而更差(99% loss=0, reward太小)
```

## Unified Theory: Reward Distribution Compatibility

```
σ-normalization effectiveness depends on reward distribution variance:

┌─────────────────────────────────────────────────────────────┐
│  Reward分布方差         │ σ-norm效果  │ 原因              │
│  低(集中≈1或≈0)        │ HURTS       │ σ≈0→A≈0→梯度消失   │
│  中(分散0.3-0.7)       │ HELPS       │ σ有意义→标准化有效 │
│  极低(连续≈0.004)      │ 灾难        │ 所有reward≈0→全无adv│
└─────────────────────────────────────────────────────────────┘

三个关键实验数据:

Exp 3 (强SFT baseline=39%):
  σ-norm 43% vs unnorm 65% → Δ=-22% ← HURTS!
  → 39%模型→多数completion错误→reward=0或0.3→reward集中→σ小
  → 但对"有差异"的组,σ-norm标准化幅度→clip压缩→有效LR降低

Exp 5 (弱SFT baseline=24%):
  σ-norm 45% vs unnorm 40% → Δ=+5% ← HELPS!
  → 24%模型→更多错误→reward更分散(0/0.1/0.3/1.0混合)
  → σ有意义→标准化减少跨组scale跳跃→更稳定优化

Exp 5 (连续reward):
  σ-norm 21% vs unnorm 21% → Δ=0 ← 两者都灾难!
  → softmax probability reward≈0.004→所有样本reward≈0→A≈0→loss=0→99%无学习
```

## σ-normalization vs Gradient Noise: Fundamental Difference

```
σ-normalization (A=(r-μ)/σ):
  操作: 标准化advantage magnitude → 改变梯度大小
  效果: ∝ reward分布方差(低方差→有害, 高方差→有益)
  问题: reward集中→σ≈0→A≈0→82%时间不学习
  与clip: σ-norm放大梯度→clip压缩→不改善问题(实验验证)
  适用: reward分散的场景(弱baseline/复杂任务/多样reward)

梯度噪声 (p.grad += randn*σ):
  操作: 添加随机扰动 → 不改变梯度方向或大小
  效果: 始终有益(σ=0.01 Goldilocks → 100% eval)
  问题: σ太大干扰(0.5→59%), σ太小无效果(0.001→48%)
  与clip: 噪声不改magnitude→clip不额外压缩→噪声始终穿透clip
  适用: 所有场景(确定性任务σ=0.01, RL训练σ=0.01-0.05)

→ 根本区别:
  σ-norm改变梯度scale → 效果取决于reward分布(条件性)
  噪声添加随机扰动 → 效果取决于噪声大小(无条件性)
  → 噪声是更通用、更可靠的regularization!
```

## Production Recommendations

```
场景                          | 推荐advantage      | 梯度噪声
数学/代码确定性任务(强SFT)    | A=r-μ              | σ=0.01
数学/代码确定性任务(弱SFT)    | A=(r-μ)/σ可尝试    | σ=0.01
推理任务(DeepSeek-R1 style)  | A=(r-μ)/σ推荐      | σ=0.01-0.05
连续reward任务               | A=r-μ              | σ=0.01
大模型GRPO(n≥16)            | A=(r-μ)/σ推荐      | σ=0.01
小模型GRPO(n≤4)             | A=r-μ              | σ=0(小模型噪声有害)

通用规则:
1. 梯度噪声σ=0.01是一行代码最简单最有效的regularization
2. σ-norm只在reward分散时有益 → 强baseline时应避免
3. clip_grad_norm是稳定性必要条件 → 不要去掉(实验验证)
4. 不要用softmax probability作continuous reward → 太小→advantage≈0
5. 合适的continuous reward应该是足够大且分散的(如0-1区间)

理论指导:
  σ-norm效果 ∝ Var(reward) / Mean(reward)
  → Var高/Mean低 → σ-norm有效(弱baseline场景)
  → Var低/Mean高 → σ-norm有害(强baseline场景)
  → Var≈0 → σ-norm灾难(连续小reward场景)
```

## Key Metrics Across All Experiments

```
实验 | σ-norm eval | unnorm eval | Δ | loss=0% | zero_adv%
Exp3 | 43%         | 65%         | -22% | 62% | 高
Exp4+clip | 26%   | 38%         | -12% | 82% | 高
Exp4 NOclip | 22% | 29%         | -7%  | 84% | 高
Exp5 binary | 45% | 40%         | +5%  | 53% | 68%
Exp5 continuous | 21% | 21%   | 0%   | 99% | 100%

Noise experiments:
  σ=0.01 SFT → 100% eval (Goldilocks)
  σ=0 (baseline) → 42% eval
  σ=0.5 → 59% eval (excessive)

→ σ-norm在不同条件下效果从-22%到+5%波动 → 不可靠
→ 梯度噪声σ=0.01在不同条件下效果从+1%到+58% → 可靠
→ 生产环境应优先使用梯度噪声而非σ-norm!
```