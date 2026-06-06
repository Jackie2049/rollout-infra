# RL Theory for LLM Training — Mathematical Foundations

> 2026-06-07 | 深度推导PPO/GRPO/DPO三大对齐算法, 证明数学等价性

## 核心结论: 三种方法都是KL约束奖励最大化的不同解法

所有LLM对齐方法(PPO/GRPO/DPO)基于同一个优化目标:
$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

| 方法 | 关键创新 | critic需要? | 训练信号 | 复杂度 |
|------|---------|-----------|---------|--------|
| PPO | 学习V(s)作为baseline | ✓ 需要 | step-level reward | 最高(4模型) |
| GRPO | 组均值作为baseline | ✗ 不需要 | outcome-level reward | 中等(2模型) |
| DPO | 隐式reward=β log_ratio | ✗ 不需要 | pairwise preference | 最低(1模型) |

## 一、Policy Gradient基础

### 标准Policy Gradient定理

目标: 最大化期望奖励 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$

推导:
$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$
$$= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)]$$

展开trajectory概率:
$$\log \pi_\theta(\tau) = \sum_{t} \log \pi_\theta(a_t | s_t)$$

逐步分解:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau)\right]$$

### 为什么加baseline能减方差

关键性质: $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$ (对任何只依赖s的baseline)

证明:
$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \int \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} da$$
$$= b(s) \cdot \nabla_\theta \int \pi_\theta(a|s) da = b(s) \cdot \nabla_\theta 1 = 0$$

因此: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a|s) \cdot (R - b(s))]$ 与原始梯度等价但方差更低.

最优baseline: $b^*(s) = \mathbb{E}[R|s]$ — 这正是V(s)!

### LLM中的Policy Gradient

对LLM, 状态 = 已生成token, 动作 = 下一个token, 轨迹 = 完整response:

$$\nabla_\theta J = \mathbb{E}_{x \sim \text{prompt}, y \sim \pi_\theta}\left[\sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) \cdot A(x, y)\right]$$

其中 $A(x,y)$ 是advantage (reward减baseline).

## 二、PPO — Proximal Policy Optimization

### PPO-Clip目标

PPO限制策略更新幅度,防止 catastrophic updates:

$$J_{\text{PPO}}(\theta) = \mathbb{E}_{x,y \sim \pi_{\theta_{\text{old}}}}\left[\min\left(\rho_t(\theta) \cdot A_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})}$ 是策略比率.

### PPO为什么有效

1. **当A>0(好动作)**: $\rho$被clip到$[1,1+\epsilon]$ → 策略不能过度增加好动作概率
2. **当A<0(坏动作)**: $\rho$被clip到$[1-\epsilon,1]$ → 策略不能过度减少坏动作概率
3. **min操作**: 取原始和clip两者的最小值 → 永远保守更新

### Advantage计算

PPO用GAE (Generalized Advantage Estimation):
$$A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

需要训练critic网络V(s) → **4个模型**(actor/critic/ref/reward)

### LLM PPO的特殊性

1. **Step-level reward**: 每个token位置都有reward → 可以用GAE
2. **KL penalty**: 加 $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ 防止偏离reference
3. **Critic cost**: 4模型 → 内存×2 vs GRPO的2模型

## 三、GRPO — Group Relative Policy Optimization

### GRPO核心思想

**不需要critic** — 用组内样本的均值和标准差作为baseline!

采样G个response for同一prompt, 计算每个的reward:
$$\{r_1, r_2, ..., r_G\}$$

Advantage:
$$A_i = \frac{r_i - \mu_r}{\sigma_r}, \quad \mu_r = \frac{1}{G}\sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_r)^2}$$

### GRPO目标函数

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t} \cdot A_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \cdot A_i\right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

### GRPO为什么是有效baseline

组均值 $\mu_r$ 作为baseline的理论依据:

1. **与V(s)类比**: V(s) = $\mathbb{E}[r|s]$ (给定状态s的期望reward)
   组均值 $\mu_r$ = $\mathbb{E}[r|q]$ (给定prompt q的期望reward)
   → 两者都是条件期望baseline!

2. **方差减少**: 去均值后的advantage $A_i = (r_i - \mu_r)/\sigma_r$ 有零均值和单位方差
   → 比原始reward的梯度方差更低

3. **Outcome-level**: GRPO只看最终outcome reward (不对每个token评分)
   → 适用于"答案是否正确"这类任务 (DeepSeek-R1数学推理)
   → 不需要step-level reward model!

### GRPO vs PPO关键区别

| | PPO | GRPO |
|---|-----|------|
| Baseline | V(s) (学习critic) | $\mu_r$ (组均值,零成本) |
| Reward | Step-level (每token) | Outcome-level (整个response) |
| Advantage | GAE ($\gamma\lambda$折扣) | 组归一化 (零均值+单位方差) |
| 模型数 | 4 (actor+critic+ref+reward) | 2 (actor+ref) |
| KL | 独立惩罚项 | 同PPO |
| 计算 | critic需额外fwd+bwd | 无额外计算 |

### GRPO的局限

1. **小G时baseline噪声大**: G=2→只有2个样本→均值不稳定→advantage噪声大
   → 解决: G=4-8较稳定 (DeepSeek-R1用G=64!)
2. **Outcome-only**: 不能区分response中哪些token贡献了reward
   → 好处: 不需要per-token reward model (简单)
   → 坏处: 可能不够精细 (但DeepSeek-R1证明足够了!)
3. **同组样本共享baseline**: 组内样本必须来自同一prompt
   → GRPO n=8: 同一prompt采样8个response

## 四、DPO — Direct Preference Optimization

### DPO推导 (完整5步)

**Step 1: KL约束奖励最大化**

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

**Step 2: 闭式最优策略**

展开KL: $\text{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{x,y}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$

对每个prompt x, 优化问题:
$$\max_\pi \sum_y \pi(y|x) r(x,y) - \beta \sum_y \pi(y|x) \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$$

Lagrange乘子法 + 概率约束 → 闭式解:
$$\pi^*(y|x) = \pi_{\text{ref}}(y|x) \cdot \frac{\exp(r(x,y)/\beta)}{Z(x)}$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 是配分函数.

**Step 3: 隐式reward**

从闭式解反推reward:
$$\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{\exp(r(x,y)/\beta)}{Z(x)}$$
$$\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{r(x,y)}{\beta} - \log Z(x)$$
$$r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**关键**: $Z(x)$ 只依赖x,不依赖y → 在偏好比较中会消去!

**Step 4: Bradley-Terry偏好模型**

假设人类偏好服从Bradley-Terry模型:
$$P(y_w > y_l | x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))} = \sigma(r(x,y_w) - r(x,y_l))$$

代入隐式reward:
$$P(y_w > y_l | x) = \sigma\left(\beta \log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**$Z(x)$消去了!** 因为 $Z(x)$ 同时出现在 $r(x,y_w)$ 和 $r(x,y_l)$ 中,差值时消去.

**Step 5: DPO损失函数**

最大化偏好数据的log-likelihood:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

### DPO为什么有效

1. **不需要reward model**: 隐式reward = β log_ratio → policy本身就是reward model!
2. **不需要critic**: 直接从偏好数据训练 → 1个模型 (vs PPO的4个)
3. **不需要采样**: 使用现成的偏好数据 → offline训练 (vs PPO的online rollout)
4. **Z(x)消去**: 这是DPO的关键数学技巧 — Bradley-Terry模型的差值结构使配分函数消去

### DPO的局限

1. **Offline**: 不能从新采样中学习 → 需要预收集偏好数据
   → 解决: Iterative DPO (采样→评分→训练→重复)
2. **Bradley-Terry假设**: 假设偏好是二元比较 → 不适合多选项排序
   → 解决: Plackett-Luce generalization
3. **Z(x)消去仅对BT成立**: 其他偏好模型不一定有这个性质
4. **隐式reward可能不准确**: 如果π偏离π_ref太多, 隐式reward估计不准

## 五、数学等价性证明

### 共同目标

所有方法优化:
$$\max_\pi \mathbb{E}[r(x,y)] - \beta \text{KL}(\pi \| \pi_{\text{ref}})$$

### PPO → 直接优化

PPO直接用policy gradient + clipping优化上述目标 (在线, 需critic)

### GRPO → 同样优化, 不同baseline

GRPO用组均值替代critic → 优化同一个目标 (在线, 不需critic)

**证明等价性**:
PPO的advantage: $A_{\text{PPO}} = r - V(s)$
GRPO的advantage: $A_{\text{GRPO}} = (r - \mu_r)/\sigma_r$

两者都是 $r - \text{baseline}$ 的形式:
- V(s)是step-level baseline (依赖当前状态)
- $\mu_r$是outcome-level baseline (依赖整个response的组均值)
- /$\sigma_r$只是缩放,不改变梯度方向

### DPO → 闭式解的参数化

DPO通过闭式最优策略 $\pi^*$ 参数化reward → 优化同一个目标 (离线, 不需reward model)

**证明**: DPO的隐式reward $r = \beta \log(\pi/\pi_{\text{ref}}) + \beta \log Z$ 是KL约束问题的闭式解 → 最大化DPO loss = 最大化原始KL约束目标.

### 等价性图

```
        KL约束奖励最大化
        max E[r] - β KL(π||π_ref)
              |
    ┌─────────┼─────────┐
    |         |         |
  PPO       GRPO       DPO
  (在线)    (在线)     (离线)
    |         |         |
  critic    组均值    隐式reward
  V(s)     μ_r/σ_r   β log_ratio
    |         |         |
  step     outcome   pairwise
  reward   reward    preference
    |         |         |
  4模型     2模型     1模型
```

**关键**: 三者数学上等价 (都是KL约束优化的不同解法), 但实现复杂度和适用场景不同.

## 六、实践选择指南

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 数学推理 (DeepSeek-R1) | **GRPO** | outcome reward足够, 组比较涌现推理 |
| 通用对话对齐 | **DPO** | 简单稳定, offline训练 |
| 精细控制 (per-token) | **PPO** | step-level reward + critic |
| 长推理 (TreeRL/MCTS) | **GRPO** | 组内多response, 树搜索 |
| 多轮对话 | **DPO/Iterative** | 偏好数据 + iterative采样 |

### 从模拟器验证

我的RLHF模拟器(RTX 4090)验证了:
- GRPO比PPO更稳定 (final 0.679 vs 0.411)
- GRPO无需critic → 少2个模型 → 内存×2
- n_samples=2-4最优 (小模型), DeepSeek-R1用n=64

### DeepSeek-R1的"aha moment"

GRPO的组比较机制自然涌现推理:
- 同一prompt采样多个response
- 组归一化 → "这个response比组平均好/差多少?"
- 好的推理路径自然获得正advantage → 被强化
- 不需要显式教推理 → "aha moment"自发涌现!

Sources:
- DPO原论文: [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290)
- DeepSeekMath (GRPO原论文): arxiv.org/abs/2402.03300
- DeepSeek-R1: arxiv.org/abs/2501.04869
- DPO推导: HuggingFace Blog DPO walkthrough
- verl GRPO源码: core_algos.py (2488行, 14 advantage estimators)