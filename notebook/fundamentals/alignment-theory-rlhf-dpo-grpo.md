# AI Alignment 理论深度: RLHF → DPO → GRPO 统一数学框架

> 2026-06-07 | 基于 Rafailov et al. NeurIPS 2023 + DeepSeek-R1 2025 + verl源码分析

## 一、KL约束策略优化 — 所有方法的共同基础

所有alignment方法都基于同一个优化目标:

$$\max_{\pi_\theta} \mathbb{E}_{x,y \sim \pi_\theta}[r(x,y)] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$$

- $r(x,y)$: reward函数 (人类偏好信号的数学表达)
- $\beta$: KL惩罚系数 (控制策略偏离参考的程度)
- $\pi_{ref}$: 参考策略 (通常是SFT后的模型)

**这个目标有闭式最优解**:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$ 是配分函数.

**关键洞察**: 最优策略将reward信息编码到策略比率中 → **策略本身就是reward model**!

## 二、RLHF (PPO路径) — 传统方法

### 2步pipeline

**Step 1: 训练reward model**

$$\max_{r_\phi} \mathbb{E}_{(x,y_w,y_l)} [\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]$$

Bradley-Terry模型: 人类偏好 $y_w \succ y_l$ → reward差值越大越好.

**Step 2: PPO训练policy**

$$\max_{\pi_\theta} \mathbb{E}_{x,y \sim \pi_\theta}[r_\phi(x,y)] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$$

PPO clip + value function + GAE → 标准RL优化.

### 问题

1. **奖励黑客(reward hacking)**: policy学会exploit reward model的漏洞 → 高reward但低质量
2. **PPO不稳定**: 超参数敏感(clip_ratio/KL_coef/learning_rate)
3. **4个模型**: actor + critic + ref + reward → 内存4x
4. **训练循环复杂**: rollout→reward→advantage→actor→critic→weights

## 三、DPO — 直接偏好优化 (Rafailov et al., 2023)

### 核心推导: 隐式reward函数

从闭式最优策略出发:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp(r(x,y)/\beta)$$

**反转**: 将reward表示为策略比率的函数:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**关键**: $Z(x)$ 只依赖$x$, 不依赖$y$ → 在Bradley-Terry配对比较中**完全消除**!

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### DPO Loss

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**直观理解**:
- $\log \frac{\pi_\theta}{\pi_{ref}}$ = 隐式reward (策略偏离参考的程度)
- 优化目标: 让winner的隐式reward比loser大 → 直接最大化偏好概率

### DPO梯度分析

$$\nabla_\theta \mathcal{L}_{DPO} = -\beta \mathbb{E}\left[\hat{r}(x,y_w) - \hat{r}(x,y_l) \cdot (1 - \sigma(\cdot)) \cdot \nabla_\theta \log \pi_\theta(y|x)\right]$$

其中 $\hat{r}(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ 是隐式reward.

**直觉**: 模型越"错"(1-σ大) → 梯度越大 → 纠正越强. reward差值越大 → 梯度越大.

### DPO vs RLHF对比

| | RLHF (PPO) | DPO |
|---|-----------|-----|
| Pipeline | 2步(reward→PPO) | **1步**(直接优化) |
| Reward model | 需要训练 | **不需要**(隐式) |
| RL训练 | 需要(PPO/GAE) | **不需要**(监督学习式) |
| 数据格式 | reward标注 | **偏好对**(y_w, y_l) |
| 模型数量 | 4(actor+critic+ref+reward) | **2**(policy+ref) |
| 稳定性 | 不稳定(超参敏感) | **稳定**(类似分类loss) |
| Reward hacking | 有风险 | **无**(无显式reward) |

### DPO局限

1. **数据质量关键**: 噪声偏好 → 模型学习错误信号
2. **分布偏移**: 训练数据分布≠模型生成分布 → off-policy问题
3. **长度偏差**: DPO倾向生成更长文本(更多token→更高隐式reward累积)
4. **无需在线采样**: 不能在训练中评估新输出质量

## 四、GRPO — 组相对策略优化 (DeepSeek-R1, 2025)

### 数学公式

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{old}(o_{i,t}|...)} \hat{A}_i, \text{clip}(...) \hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

**组相对优势**:

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

其中 $\mathbf{r} = \{r_1, ..., r_G\}$ 是同一prompt的G个response的reward.

### 为什么组归一化有效?

**理论解释1: Baseline 减方差**

在REINFORCE中, 优势函数 $A = r - b$ 减少梯度方差.
组均值 $\text{mean}(\mathbf{r})$ 是天然的baseline:
- 同一prompt的G个response共享相同的prompt → baseline精确
- 不同于全局baseline(不区分prompt质量差异)

**理论解释2: 对比学习视角**

GRPO ≈ contrastive learning:
- "比组内平均好的" → 正例 → 增强
- "比组内平均差的" → 反例 → 抑制
- 组内归一化 → 相对质量而非绝对质量

**理论解释3: Leave-One-Out Baseline**

组均值baseline与RLOO(Leave-One-Out)理论一致:
$$\hat{A}_i = r_i - \frac{1}{G-1}\sum_{j \neq i} r_j$$

GRPO用组均值代替LOO均值 → 近似但更高效(无需逐个排除).

### GRPO vs PPO vs DPO 三方对比

| | PPO | DPO | GRPO |
|---|-----|-----|------|
| **Critic** | 需要(V网络) | 不需要 | **不需要**(组baseline) |
| **Reward来源** | 训练的RM | 偏好对 | **规则函数**(DeepSeek-R1) |
| **数据格式** | prompt+response | 偏好对(y_w,y_l) | **prompt+G个response+reward** |
| **Advantage** | GAE(γ,λ) | 隐式reward差 | **组归一化** |
| **在线采样** | 需要(rollout) | 不需要 | **需要**(每步G个采样) |
| **KL约束** | 独立KL penalty | 内嵌在loss | **独立KL penalty** |
| **模型数量** | 4 | 2 | **2**(actor+reward_fn) |
| **训练稳定性** | 不稳定 | 最稳定 | **较稳定**(组baseline) |
| **长度偏差** | 有 | **严重** | **轻微**(组内长度接近) |

### GRPO的"涌现推理"现象

DeepSeek-R1-Zero: 纯GRPO RL(无SFT) → 模型**自然涌现**推理行为:
- 自发产生"aha moment"(突然找到解题思路)
- 逐步推理链(chain-of-thought)自然出现
- **无需人工标注推理数据**

原因: outcome reward(答案正确性) + 组相对比较 → 模型自发探索更优推理路径.

## 五、统一框架: 所有方法都是KL约束优化的不同实现

```
KL约束优化: max E[r] - β KL(π_θ || π_ref)
│
├── PPO路径 (显式reward + RL)
│   训练reward model → PPO优化 → GAE advantage
│   优点: 灵活, 在线学习
│   缺点: 不稳定, 4个模型, reward hacking
│
├── DPO路径 (隐式reward + 监督学习)
│   闭式最优策略 → 反转得到隐式reward → 直接优化偏好对
│   优点: 简单稳定, 2个模型, 无RL
│   缺点: offline only, 长度偏差, 数据质量关键
│
└── GRPO路径 (组baseline + outcome reward)
│   组归一化替代critic → outcome-only reward → PPO clip
│   优点: 无critic, 规则reward可行, 推理涌现
│   缺点: 需在线采样(G个), 规则reward覆盖范围有限
│
三种方法都是同一个数学问题的不同解法!
```

### 数学等价性

**DPO隐式reward = PPO显式reward (在最优解处)**:

$$r^{DPO}(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

$$r^{PPO}(x,y) = r_\phi(x,y)$$

当 $\pi^*$ 是KL约束优化的最优解时, 两者数学等价.

**GRPO advantage = GAE advantage (在γ=1,λ=0时)**:

GAE: $A_t = r_t + \gamma V_{t+1} - V_t$ → 需要V函数
GRPO: $\hat{A}_i = (r_i - mean)/std$ → 组baseline替代V

GRPO的组均值 ≈ GAE中V函数的蒙特卡洛估计 → 在outcome-only reward下两者近似等价.

## 六、理论延伸

### IPO (Identity Preference Optimization)

解决DPO的overfitting问题:

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

用MSE替代log-sigmoid → 对偏好噪声更鲁棒.

### KTO (Kahneman-Tversky Optimization)

只需二元反馈(good/bad), 不需要偏好对:

$$\mathcal{L}_{KTO} = \begin{cases} \lambda_w (1 - \sigma(\hat{r}(x,y) - z_{ref})) & \text{if } y \text{ is desirable} \\ \lambda_l (1 - \sigma(z_{ref} - \hat{r}(x,y))) & \text{if } y \text{ is undesirable} \end{cases}$$

不对称loss权重 → Kahneman-Tversky损失厌恶理论.

### ORPO (Odds Ratio Preference Optimization)

$$\mathcal{L}_{ORPO} = -\log \sigma\left(\log \frac{OR(y_w|x)}{OR(y_l|x)}\right)$$

其中 $OR(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$ — 似然比(odds ratio).

### verl中的实现 (2026)

verl core_algos.py 已包含14种advantage estimator + 9种policy loss:
- 14种advantage: GAE/GRPO/GRPO_VECTORIZED/RLOO/REINFORCE_PLUS_PLUS/REMAX/OPO/GRPO_PASSK/GPG/OPTIMAL_TOKEN_BASELINE/TIR_OPTIMAL_TOKEN_BASELINE/GDPO
- 9种policy loss: vanilla(PPO clip)/dppo_tv/dppo_kl/gspo/sapo/gpg/cispo/geo_mean/bypass_mode
- 注册表式扩展: `@register_adv_est` / `@register_policy_loss`

## 七、实用结论

1. **DPO最简单**: 偏好对数据 → 直接训练 → 适合快速实验和小规模对齐
2. **GRPO最强**: 规则reward → 推理涌现 → DeepSeek-R1级别的推理能力
3. **PPO最灵活**: 训练的RM → 任何reward信号 → 适合复杂多维度reward
4. **三者数学等价**: 同一个KL约束优化问题的不同实现
5. **数据格式选择**: 有偏好对→DPO, 有规则函数→GRPO, 有复杂reward→PPO
6. **模型数量**: GRPO/DPO=2模型 vs PPO=4模型 → GRPO/DPO更适合有限GPU
7. **DeepSeek-R1证明**: outcome reward + GRPO → 推理自然涌现, 无需标注推理链
8. **长度偏差**: DPO最严重 → GRPO轻微 → PPO可控(加入长度惩罚)

Sources:
- [DPO Paper (Rafailov et al., NeurIPS 2023)](https://arxiv.org/abs/2305.18290)
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)
- [IPO: Identity Preference Optimization](https://arxiv.org/abs/2310.12043)
- [KTO: Model Alignment as Prospect Theory Optimization](https://arxiv.org/abs/2402.01306)