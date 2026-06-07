# 概率论与贝叶斯推断 — AI训练的数学根基

> 2026-06-07 | 从MLE到MAP到Bayesian: 理解"为什么Adam比SGD好"的概率论基础

## 概述

概率论是AI训练的底层数学。本文从最大似然估计(MLE)出发，推导出MAP→贝叶斯推断→Adam优化器→模型校准，连接到LLM训练的每一个决策。

## 一、最大似然估计(MLE) — 训练的数学动机

### 1.1 MLE定义

```
给定数据 D = {x₁, ..., x_N}, 模型参数θ:
  L(θ) = P(D|θ) = Π P(x_i|θ)  ← 似然函数

MLE目标: θ_MLE = argmax L(θ)

对数似然(更方便):
  log L(θ) = Σ log P(x_i|θ)

→ 最大化似然 = 最大化对数似然 = 最小化负对数似然
→ 负对数似然 = 交叉熵(当P(x_i|θ) = softmax(z_i))
→ 所以LLM训练(SFT) = MLE!
```

### 1.2 为什么MLE = Cross-Entropy?

```
SFT目标: min -Σ log P(y_i|θ, x_i)

对于softmax输出:
  P(y_i|θ) = softmax(z_i)[y_i] = exp(z_y_i) / Σ exp(z_j)

负对数似然:
  -log P(y_i) = -z_y_i + log Σ exp(z_j)

→ 这就是Cross-Entropy loss!
→ SFT训练 = MLE on next-token prediction
→ 最小化CE = 最大化数据似然 = 找最"解释"数据的参数

→ 与我们之前的连接:
   MiniGPT SFT: loss 5.03→1.47 → 似然↑ → 模型更好地解释数据
   SFT→GRPO: 100% eval → SFT找到好的MLE → GRPO在此基础上优化
```

### 1.3 MLE的性质

```
MLE估计器的性质(统计学):
1. 一致性: θ_MLE → θ_true 当 N→∞ (数据越多→估计越准确)
2. 渐近正态: θ_MLE ~ N(θ_true, I(θ_true)^(-1)) (大样本时接近正态)
3. 渐近有效: Cramér-Rao下界 → MLE是最优无偏估计(大样本)

→ 但实际LLM训练:
   数据量有限(N不是∞) → MLE过拟合 → 需要regularization
   → L2正则 → MAP估计(见下节)
   → 数据增强 → 增加有效N
   → 早停 → 防止MLE过拟合
```

## 二、MAP估计 — MLE + 正则化

### 2.1 贝叶斯框架

```
贝叶斯定理:
  P(θ|D) = P(D|θ) × P(θ) / P(D)
  后验 = 似然 × 先验 / 证据

→ MLE: 只看似然P(D|θ) → 忽略先验P(θ) → 可能过拟合
→ MAP: 看似然+先验 → θ_MAP = argmax P(D|θ) × P(θ)
→ Bayesian: 看完整后验P(θ|D) → 考虑参数不确定性

MAP = argmax [log P(D|θ) + log P(θ)]
     = argmax [Σ log P(x_i|θ) + log P(θ)]

→ log P(D|θ): 数据拟合项(似然) = CE loss的负数
→ log P(θ): 先验约束项 = regularization!
```

### 2.2 L2正则 = Gaussian先验

```
如果P(θ) = N(0, σ²) ← 参数服从Gaussian先验:

log P(θ) = Σ -θ_i²/(2σ²) + const
         = -(1/2σ²) × ||θ||² + const

→ MAP目标 = Σ log P(x_i|θ) - (1/2σ²) × ||θ||²
         = -CE_loss - (λ) × ||θ||²  ← λ = 1/(2σ²)

→ **L2正则化 = Gaussian先验的MAP估计!**
→ λ越大 → σ越小 → 先验认为参数应接近0 → 更强正则化
→ λ越小 → σ越大 → 先验宽容 → 更弱正则化
```

**我们实验验证**: L2在Adam中是灾难(loss飙升47x) → AdamW的decoupled wd正确

### 2.3 L1正则 = Laplace先验

```
如果P(θ) = Laplace(0, b) ← 参数服从Laplace先验:

log P(θ) = Σ -|θ_i|/b + const

→ MAP目标 = Σ log P(x_i|θ) - (1/b) × ||θ||₁

→ L1正则化 = Laplace先验的MAP估计!
→ L1 → 稀疏解(很多参数=0) → Laplace先验在0处密度最高 → 推动参数归零

→ 我们的SAE实验: L1稀疏 = Laplace先验 → λ=3所有feature死亡 → 先验太强!
→ TopK SAE: 不用L1 → 不用Laplace先验 → 直接选top-K → 无feature死亡
```

## 三、Adam优化器 — 贝叶斯视角

### 3.1 SGD vs Adam

```
SGD: θ = θ - lr × ∇L(θ) ← 固定学习率 → 所有参数同等对待

Adam: θ = θ - lr × m/(√v + ε)
  m = β₁ × m + (1-β₁) × ∇L ← 一阶矩(动量)
  v = β₂ × v + (1-β₂) × ∇L² ← 二阶矩(自适应学习率)

→ Adam对每个参数有不同的学习率: lr_i = lr / (√v_i + ε)
→ v_i大 → 梯度波动大 → 学习率小 → 防止跳过最优
→ v_i小 → 梯度稳定 → 学习率大 → 加速收敛

→ Adam = "自适应梯度缩放" → 与GRPO的σ归一化异曲同工!
```

### 3.2 Adam与自然梯度

```
自然梯度(Natural Gradient): 考虑参数空间的几何 → 用Fisher信息矩阵修正方向

∇_natural L = F⁻¹ × ∇L

→ F = Fisher信息矩阵 → 衡量"参数改变对分布的影响程度"
→ F⁻¹修正 → 使参数更新方向考虑参数空间的曲率 → 更高效

Adam与自然梯度的联系:
  v ≈ E[∇L²] → 对角近似Fisher信息 → v⁻¹/² ≈ F⁻¹/²
  → Adam ≈ 自然梯度的对角近似 → 每参数独立 → 简化版

→ 这解释了为什么Adam比SGD好: SGD忽略参数空间几何 → Adam近似考虑
→ 但Adam只是对角近似 → 非对角耦合未考虑 → 更完整的自然梯度方法(K-FAC)更好但太贵
```

### 3.3 我们的实验验证

```
AdamW(lr=0.001, wd=0.1)最优 → loss↓6.5%
SGD(lr=0.1) → ↓3.3%但收敛差
L2在Adam中 → loss飙升47x → wd被自适应lr缩放 → 正则化强度不一致!
AdamW decoupled wd → wd独立于lr → 正则化一致 → 安全

→ 概率论解释:
   L2(=Gaussian先验)在Adam中: wd × lr_adam → lr_adam变化 → wd强度变化 → 不一致
   AdamW decoupled wd: wd × θ → wd强度恒定 → 等价于固定σ²的Gaussian先验 → 一致
```

## 四、Gaussian分布 — AI的核心分布

### 4.1 Gaussian性质

```
N(μ, σ²): p(x) = (1/(√2π σ)) × exp(-(x-μ)²/(2σ²))

关键性质:
1. 中心极限定理: 任何分布的样本均值→Gaussian(当N→∞)
   → 神经网络输出 = 大量神经元加和 → 接近Gaussian → 这是LN归一化的基础!

2. 最大熵性质: 固定均值和方差 → Gaussian是最大熵分布
   → 如果只知道均值和方差 → Gaussian是"最少假设"的分布
   → 这解释了为什么噪声假设常用Gaussian → 不引入额外假设

3. 线性变换不变性: 如果X~N(μ,σ²) → aX+b ~ N(aμ+b, a²σ²)
   → 神经网络的线性层 → Gaussian→Gaussian → 只改变μ和σ → LN恢复μ=0,σ=1
```

### 4.2 为什么LN/BN用Gaussian假设?

```
LayerNorm: y = (x-μ)/σ → 归一化到μ=0, σ=1

假设: x ≈ Gaussian → 减均值除方差 → 输出~N(0,1) → 稳定

但x不一定Gaussian! → LN仍有效 → 因为:
  1. 无论分布 → 减均值除方差 → 零均值单位方差 → 数值稳定
  2. 大量ReLU加和 → 中心极限定理 → 接近Gaussian → LN假设大致成立
  3. learnable γ,β → 可以恢复任意分布 → 不是严格Gaussian约束

→ LN = "weak Gaussian assumption + learnable correction"
→ 实践中有效 → 即使分布不是Gaussian
```

## 五、概率论与训练方法

### 5.1 RL训练的概率论视角

```
Policy Gradient:
  ∇J = E[∇logπ(a|s) × R]
     = Σ_a π(a|s) × ∇logπ(a|s) × R(a)

→ 这是期望值的梯度 → 期望 = 概率加权平均
→ 高概率action贡献大 → 低概率action贡献小 → 但梯度∇logπ∝1/π抵消!

→ ∇logπ(a|s) = ∇π(a|s) / π(a|s)
→ 高概率: ∇π大但1/π小 → 梯度适中
→ 低概率: ∇π小但1/π大 → 梯度适中(!)

→ Policy gradient对所有action有适中梯度 → 不偏向高/低概率 → 好!
→ 但需要采样估计期望 → 样本少 → 估计不准确 → variance大

→ baseline减方差:
  ∇J ≈ E[∇logπ × (R-b)] → b接近E[R] → (R-b)波动小 → variance↓
  → 我们实测: variance↓66.5% ✓

→ GRPO组baseline: μ ≈ E[R] → (R-μ)/σ → variance≈1 ✓
```

### 5.2 Dropout = Bernoulli采样

```
Dropout: 每个神经元以概率p被"关闭" → Bernoulli(p)采样

训练: y = (1-p) × Σ active_neurons × x ← 每次不同的子网络
推理: y = Σ all_neurons × x ← 使用全部神经元(但权重×(1-p))

→ Dropout = 贝叶斯模型的近似!
  → 每次训练用不同子网络 → 类似"从后验分布采样不同模型"
  → 推理时平均 → 类似"贝叶斯模型平均"
  → Dropout不确定性 = 模型不确定性 → 可以用来估计confidence!

→ Gal & Ghahramani (2016): Dropout as Bayesian Approximation
  → 多次dropout推理 → 输出方差 ≈ 模型不确定性 → 不确定性估计!
```

### 5.3 温度采样 = Boltzmann分布

```
LLM推理采样:
  P(y) = exp(z_y/T) / Σ exp(z_j/T) ← Temperature-softmax

→ T=1: 标准Boltzmann分布 → 概率∝exp(logit)
→ T→0: argmax → 确定性选择(最低能量)
→ T→∞: uniform → 随机选择(所有状态等概率)

→ 物理类比: Boltzmann分布描述粒子在不同能级的概率
  → 低能级(高logit)→高概率 → 高能级(低logit)→低概率
  → T=温度 → 高温→粒子可以跳到高能级 → 更多随机性
  → 低温→粒子困在低能级 → 确定选择

→ LLM推理: T↑ → 更有创意(探索高logit以外的区域) → 但可能不准确
           T↓ → 更准确(选择最高logit) → 但可能无聊(永远选最可能的)

→ DeepSeek-R1: 高T采样 → 探索推理路径 → "aha moment"涌现 → 发现低概率但正确的反思策略
→ SFT: T=1训练 → 学习标准推理 → 但推理模式有限 → 需RL探索更多模式
```

## 六、模型校准 — 概率论的应用

### 6.1 校准定义

```
模型校准(Model Calibration):
  P(y=correct) = p → 模型说"90%确定" → 实际90%正确 → 校准好
  P(y=correct) ≠ p → 模型说"90%确定" → 实际只有70%正确 → 校准差

→ 校准好的模型: 输出概率 = 实际正确率 → 可以信任模型的confidence
→ 校准差的模型: 输出概率 ≠ 实际正确率 → 不能信任confidence → 可能过自信

→ LLM通常over-confident → P(correct)高但实际accuracy低
→ 我们的RL实验: GRPO训练reward 87.5% → eval 50% → 模型over-confident!
→ SFT→GRPO: 训练100% → eval 93% → 近似校准 → 可以信任confidence
```

### 6.2 Temperature Scaling校准

```
最简单的校准方法: 改变softmax的温度

校准前: P(y) = softmax(z)[y] → 过自信
校准后: P(y) = softmax(z/T_cal)[y] → T_cal>1 → 更谦逊

→ T_cal在验证集上优化 → 找使P(y)=实际正确率的T

→ 与推理温度不同:
   推理温度: 控制创造性/确定性 → 用户选择
   校准温度: 使模型confidence与实际accuracy匹配 → 自动优化

→ 连接到我们的GRPO实验:
   GRPO-only模型: confidence高但accuracy低 → T_cal>1 → 校准
   SFT→GRPO模型: confidence≈accuracy → T_cal≈1 → 已校准 → 更好!
```

## 七、贝叶斯深度学习 — 前沿方向

### 7.1 贝叶斯神经网络(BNN)

```
标准神经网络: θ是固定值 → 一个模型 → 一个预测

贝叶斯神经网络: θ是分布 → P(θ|D) → 多个模型 → 预测有不确定性

→ BNN预测: P(y|x) = Σ P(y|x,θ) × P(θ|D) ← 对所有可能参数做加权平均

→ 问题: P(θ|D)难以精确计算(高维积分) → 需要近似:
   1. variational inference: 用简单分布近似后验 → VI loss = ELBO
   2. MCMC: 蒙特卡洛采样 → 采样P(θ|D)中的参数 → 多次预测
   3. Dropout近似: 多次dropout推理 → 近似后验采样(Gal & Ghahramani)

→ BNN的优势: 不确定性估计 → 知道模型"不确定什么"
→ BNN的劣势: 计算成本高(多次推理) → LLM太大 → BNN不实用

→ 但思想重要: 理解模型不确定性 → RL训练中的探索-利用权衡 → 贝叶斯视角
```

### 7.2 ELBO — 变分推断

```
ELBO(Evidence Lower Bound):
  log P(D) ≥ E_q[log P(D|θ)] - KL(q(θ) || P(θ))
           = E_q[log P(D|θ)] + E_q[log P(θ)] - E_q[log q(θ)]
           = 重构质量 + 先验匹配 - q的熵

→ ELBO = 似然(数据拟合) + 先验(正则化) - 熵(分布复杂度)
→ 最大化ELBO = 最小化KL(q||P) + 最大化似然
→ 变分推断: 找q使ELBO最大 → q≈P(θ|D) → 近似后验

→ 与VAE连接:
  VAE loss = CE(重构) + KL(q(z|x) || P(z)) ← ELBO的负数
  → 重构损失 = 似然 → KL散度 = 先验-后验差异 → 正则化

→ 与我们的蒸馏连接:
  蒸馏KL = KL(P_student || P_teacher) → 类似VAE的KL → 使student分布接近teacher
```

## 八、总结: 概率论→AI训练的一切

```
MLE → CE loss → SFT训练 → 最大似然
MAP → CE + L2 → 正则化训练 → Gaussian先验
Adam → 自然梯度近似 → 自适应学习率
Gaussian → LN归一化 → 中心极限定理
Policy gradient → 期望梯度 → 采样估计
Dropout → Bernoulli → 贝叶斯近似
Temperature → Boltzmann分布 → 创造性/确定性
校准 → confidence=accuracy → 可信任模型
ELBO → 变分推断 → 不确定性估计

→ 所有LLM训练方法都是概率论的应用:
   SFT = MLE
   GRPO = policy gradient with group baseline
   DPO = Bradley-Terry preference model
   正则化 = MAP with Gaussian/Laplace prior
   Adam = natural gradient diagonal approximation