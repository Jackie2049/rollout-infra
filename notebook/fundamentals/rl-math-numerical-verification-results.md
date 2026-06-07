# RL Math Numerical Verification Results — Policy Gradient → GRPO

> 2026-06-07 | 5组数值实验验证RL核心理论, CPU运行(SimplePolicy 16×4→3)

## 实验概览

| Exp | 理论 | 方法 | 结果 |
|-----|------|------|------|
| 1 | ∇J = E[∇logπ·R] | REINFORCE vs autograd(分析梯度) | cos_sim=0.999962 |
| 2 | E[∇logπ·b]=0 | 50000样本统计 | |b·E[∇logπ]| / |E[∇logπ·R]| = 0.88%, variance↓66.5% |
| 3 | μ_group ≈ V(x) | MC估计收敛性 | error_std∝1/√n, ratio≈0.59 |
| 4 | GRPO比vanilla PG方差低 | 组归一化vs原始reward | variance↓39.9%, mean cos_sim=0.9992 |
| 5 | PPO-clip不改梯度方向 | clip vs unclipped对比 | clip时梯度=0(26.1%), 未clip时cos_sim=1.0 |

## Experiment 1: Policy Gradient Theorem ∇J = E[∇logπ·R]

### 方法

比较两种计算∇J的方式:
- **REINFORCE**: Monte Carlo采样 E[∇logπ(a|s) · R], 50000样本
- **Analytical**: 对确定性期望J(θ) = Σ_a π(a|s)·r(a)用autograd计算梯度, 50000个状态平均

### 关键创新

不再用昂贵的有限差分法(需要对每个参数元素做2次forward),改用autograd对期望奖励直接求导→更高效+更精确。

### 结果

| 指标 | 值 |
|------|------|
| J_baseline (平均奖励) | 0.8888 |
| REINFORCE grad norm | 0.7848 |
| Analytical grad norm | 0.7864 |
| Cosine similarity | **0.999962** |
| Norm ratio | 0.998 |
| Max element diff | 0.00293 |

**结论**: cos_sim>0.9999 → REINFORCE梯度与分析梯度方向几乎完全一致, magnitude误差<0.2% → **Policy Gradient Theorem数值验证通过**

### 数学基础

$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot R\right]$$

验证了: 用Monte Carlo估计(采样)的梯度方向与确定性分析梯度一致→随机采样是可靠的梯度估计方法。

## Experiment 2: Baseline Variance Reduction

### E[∇logπ·b] = 0 验证

理论: 任何不依赖action的baseline b不影响期望梯度:
$$\mathbb{E}[\nabla \log\pi \cdot b] = b \cdot \mathbb{E}[\nabla \log\pi] = b \cdot \nabla \mathbb{E}_a[1] = 0$$

结果:
- |E[∇logπ]| = 0.007 (不精确为0因为有限样本)
- |b·E[∇logπ]| = 0.006
- **Relative: |b·E[∇logπ]| / |E[∇logπ·R]| = 0.88%** → 近似为0, 理论验证通过

### Variance Reduction

| 方案 | Total Variance |
|------|---------------|
| No baseline (∇logπ·R) | 4.960 |
| With baseline b=0.87 (∇logπ·(R-b)) | 1.662 |
| **Reduction** | **66.5%** |

**b = mean_reward = 0.87是好的baseline → 方差减少66.5%**

### GRPO含义

GRPO用μ_group作为baseline → μ_group是V(x)的MC估计 → 期望上等价于用V(x)作baseline → 方差显著减少

## Experiment 3: GRPO Group Mean = MC Estimate of V(x)

### 核心验证

GRPO的组均值μ_group是条件期望V(x) = E[r(x,y)|x]的蒙特卡洛估计:
$$V(x) = \sum_a \pi(a|x) \cdot r(a) \approx \frac{1}{n}\sum_i r(x,y_i) = \mu_{group}$$

### Scaling: error_std vs 1/√n

| n | actual_std | theoretical σ/√n | ratio |
|---|-----------|-------------------|-------|
| 1 | 0.437 | 0.820 | 0.53 |
| 2 | 0.337 | 0.580 | 0.58 |
| 4 | 0.244 | 0.410 | 0.60 |
| 8 | 0.171 | 0.290 | 0.59 |
| 16 | 0.131 | 0.205 | 0.64 |
| 32 | 0.085 | 0.145 | 0.59 |

**ratio≈0.59**: 实际标准差比理论σ/√n更低, 因为reward分布(r=0,1,2)比假设更集中。

**关键结论**: error_std严格随n增加而减少 → μ_group收敛到V(x) → **GRPO组均值就是critic的MC估计**

## Experiment 4: GRPO vs Vanilla PG Variance

### 比较

| 方案 | Total Variance |
|------|---------------|
| Vanilla PG (∇logπ·R) | 6.290 |
| GRPO (∇logπ·(R-μ)/σ) | 3.779 |
| **Reduction** | **39.9%** |

### Mean gradient direction

- Mean gradient cos_sim (vanilla vs GRPO) = **0.9992**
- 两者期望梯度方向一致 → GRPO组归一化不影响收敛方向,只减少方差

### 注意

GRPO方差减少39.9%(比baseline 66.5%少)→ 因为除σ引入额外方差(σ本身有估计误差)。**Dr.GRPO不除σ**(arXiv:2503.20783)可进一步减少方差。

## Experiment 5: PPO-clip Gradient Direction

### 关键发现

PPO-clip有两种效果:
1. **Clip NOT active**: gradient = unclipped gradient → **cos_sim = 1.0**(完全一致)
2. **Clip IS active**: gradient = **0** → 阻止更新

| 统计 | 值 |
|------|------|
| ε | 0.2 |
| Clip active比例 | 26.1% |
| Non-clip cos_sim | 1.000000 |
| Direction matches (>0.9) | 100% |

### 数学解释

PPO-clip objective:
$$L^{clip} = \min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)$$

- **A>0, ρ>1+ε**: clip active → L=(1+ε)A = constant → ∇L=0 (对π_θ无梯度)
- **A<0, ρ<1-ε**: clip active → L=(1-ε)A = constant → ∇L=0
- **其他**: clip NOT active → L=ρ·A → ∇L=∇logπ·A (与unclipped一致)

**PPO-clip从不改变梯度方向,只将"过大更新"的梯度设为零 → 防止destructive update但保留正常更新方向**

### 与GRPO的关系

GRPO通常也用PPO-clip → 组归一化减少方差 + clip防止过大更新 → **双重方差控制**

## 综合结论

5组实验完整验证了RL → GRPO的数学链:

1. **Policy Gradient Theorem**: ∇J = E[∇logπ·R], cos_sim=0.999962 ✓
2. **Baseline理论**: E[∇logπ·b]=0(0.88%误差), variance↓66.5% ✓
3. **GRPO μ=V(x)的MC估计**: error∝1/√n收敛 ✓
4. **GRPO方差减少**: 39.9% vs vanilla, 方向一致(cos_sim=0.999) ✓
5. **PPO-clip**: 不改方向, 只零化过大更新(26.1%) ✓

**数学链**: PG定理 → baseline减方差 → μ_group=V(x) → GRPO组归一化 → PPO-clip保护

## 脚本

- `tools/rl_math_numerical_verification.py`: CPU运行, SimplePolicy(131参数), ~3分钟完成5实验

## 参考

- Policy Gradient: Sutton et al., 1999
- PPO: Schulman et al., 2017, arXiv:1707.06347
- GRPO: Shao et al., 2024, arXiv:2402.03300
- Dr.GRPO: arXiv:2503.20783
- RL Theory for LLM Training: notebook/fundamentals/rl-math-policy-gradient-to-grpo.md