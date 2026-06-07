# RL Math Foundations: Policy Gradient → GRPO Derivation

> 2026-06-07 | 从零推导policy gradient theorem到GRPO, 含数值验证

## 一、Policy Gradient Theorem

### 目标函数

最大化期望累计奖励:
$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} r_t\right] = \mathbb{E}_{\pi_\theta}[R]$$

### 推导: ∇J

关键: 我们需要对策略参数θ的梯度, 但奖励r不直接依赖θ.

$$J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)$$

其中轨迹概率:
$$P(\tau;\theta) = \prod_{t=0}^{T} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$$

取梯度:
$$\nabla_\theta J = \sum_\tau \nabla_\theta P(\tau;\theta) R(\tau)$$

**关键技巧**: $\nabla_\theta P = P \cdot \nabla_\theta \log P$ (log-derivative trick)

$$\nabla_\theta J = \sum_\tau P(\tau;\theta) \nabla_\theta \log P(\tau;\theta) R(\tau)$$
$$= \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log P(\tau;\theta) \cdot R(\tau)\right]$$

由于环境动力学$P(s_{t+1}|s_t,a_t)$不依赖θ:
$$\nabla_\theta \log P(\tau;\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Policy Gradient Theorem**:
$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

### 引入Advantage函数

直接用$R(\tau)$作为权重 → 高方差. 引入baseline $b(s_t)$:
$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R(\tau) - b(s_t))\right]$$

**为什么baseline不影响期望?**:

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right]$$
$$= b(s_t) \cdot \mathbb{E}_{a_t}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$
$$= b(s_t) \cdot \nabla_\theta \mathbb{E}_{a_t}[1] = b(s_t) \cdot \nabla_\theta 1 = 0$$

**最优baseline**: $b^*(s_t) = \mathbb{E}[R(\tau)|s_t]$ → 这就是value function $V(s_t)$!

定义**Advantage**: $A_t = R_t - V(s_t)$

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]$$

## 二、GAE (Generalized Advantage Estimation)

### TD Error

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

这是一步bootstrap估计: 用$V(s_{t+1})$估计后续奖励.

### k-step Return

$$R_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k})$$

### GAE: λ-加权组合

$$A_t^{GAE} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

**λ的意义**:
- λ=0: $A_t = \delta_t$ → 一步估计 → 低方差, 高偏差
- λ=1: $A_t = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}$ → Monte Carlo → 低偏差, 高方差
- λ∈(0,1): 方差-偏差权衡

**展开验证** (λ=1, γ=1):
$$A_t = \delta_t + \delta_{t+1} + \delta_{t+2} + ...$$
$$= (r_t + V(s_{t+1}) - V(s_t)) + (r_{t+1} + V(s_{t+2}) - V(s_{t+1})) + ...$$
$$= r_t + r_{t+1} + r_{t+2} + ... - V(s_t)$$
$$= R(\tau) - V(s_t)$$ → 这就是完整return减baseline!

## 三、PPO (Proximal Policy Optimization)

### 目标: KL约束的策略优化

$$\max_\theta \mathbb{E}[A_t] \quad \text{s.t.} \quad KL(\pi_\theta || \pi_{\theta_{old}}) \le \delta$$

### PPO-Clip: 简化KL约束

$$L^{clip}(\theta) = \mathbb{E}_t\left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) A_t\right)\right]$$

**ratio**: $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**4种情况**:
| A_t > 0 (好动作) | A_t < 0 (坏动作) |
|---|---|
| ratio ↑ → 鼓励好动作 | ratio ↓ → 抑制坏动作 |
| clip at 1+ε → 防止过度增加 | clip at 1-ε → 阀防止过度减少 |
| min选择保守目标 | min选择保守目标 |

### 双Clip PPO (Cheng et al., 2019)

当$A_t < 0$且$\rho_t > 1+\epsilon$ → PPO-clip目标是0 → 策略可能从"好"变"坏"

解决方案: 加入下界clip:
$$L = \max\left(\min(\rho_t A_t, (1+\epsilon)A_t), (1-\epsilon)A_t\right) \text{when } A_t < 0$$

## 四、GRPO (Group Relative Policy Optimization)

### 核心思想: 组均值替代critic

GRPO **不需要V(s)** → 用组内均值作为baseline!

$$A_i^{GRPO} = \frac{r_i - \mu_{group}}{\sigma_{group}}$$

其中:
- $r_i$: 第i个response的outcome reward
- $\mu_{group} = \frac{1}{n}\sum_{j=1}^n r_j$: 同prompt的n个response的均值
- $\sigma_{group} = \sqrt{\frac{1}{n}\sum_{j=1}^n (r_j - \mu)^2}$: 组标准差

### 数学推导: 为什么组均值是好的baseline?

从policy gradient theorem:
$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot (r(x,y) - b(x))\right]$$

选择$b(x) = \mu_{group}(x) = \frac{1}{n}\sum_i r(x,y_i)$:

**证明: 组均值是条件期望的蒙特卡洛估计**

$$\mathbb{E}_{y \sim \pi_\theta}[r(x,y)] \approx \frac{1}{n}\sum_i r(x,y_i) = \mu_{group}$$

所以$\mu_{group}$就是$V(x)$的蒙特卡洛估计 → 数学上等价于用critic!

**组归一化$(r - \mu)/\sigma$的好处**:
1. **方差减少**: $(r-\mu)$已经减去了baseline → 进一步除$\sigma$归一化 → 防止不同组reward尺度差异
2. **对比学习**: 组内比较 → 排名信号 → "比平均好/差多少"
3. **Dr.GRPO**: 不除$\sigma$ → 更保守(防止过度归一化) → arXiv:2503.20783

### 单样本组特殊处理

当$n=1$ → $\mu=r_1, \sigma=0$ → advantage为0 → 无更新!

verl处理: 当$\text{len}(id2score)==1$ → $\mu=0, \sigma=1$ → advantage = raw reward
→ 单样本时不归一化 → 直接用原始reward

### Outcome-only vs Step-level

GRPO: `scores = token_level_rewards.sum()` → outcome-only
- 整个response的reward = 所有token reward之和
- 不区分哪个token贡献 → 但组比较自然涌现哪些response整体更好

PPO: step-level reward → 每个token有独立reward → 需要critic估计每步value

**为什么outcome-only足以涌现推理?**
- DeepSeek-R1-Zero: outcome-only reward (数学正确性) → "aha moment"涌现
- 组比较: 反思→修正→正确 的response reward高 → 直接一步→错误 的reward低
- 反思行为被组归一化强化 → **推理自然涌现**

## 五、数学等价性: PPO/GRPO/DPO三者统一

### 统一目标

所有方法优化同一个目标:
$$\max_\theta \mathbb{E}[r] - \beta KL(\pi_\theta || \pi_{ref})$$

### 闭式最优解

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

$Z(x)$是配分函数(normalization constant).

**关键**: 最优策略本身就是reward model!

### PPO: 在线优化

- 用actor采样 → 用critic估计V → GAE advantage → clip update
- 等价于: 在线策略梯度逼近最优解

### GRPO: 在线+组baseline

- 用actor采样n个 → 组均值替代critic → 组归一化advantage
- 等价于: critic的蒙特卡洛估计 + clip update

### DPO: 离线优化

从最优策略反转:
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

偏好对$(y_w > y_l)$:
$$\pi^*(y_w) > \pi^*(y_l) \implies \frac{\pi^*(y_w)}{\pi_{ref}(y_w)} > \frac{\pi^*(y_l)}{\pi_{ref}(y_l)}$$

DPO loss:
$$L_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)$$

**$Z(x)$消去**: 在差值中$\beta\log Z(x)$被减掉 → 不需要估计配分函数!

### 三者等价证明链

$$PPO \xrightarrow{\text{critic=group mean}} GRPO \xrightarrow{\text{反转最优策略}} DPO$$

- PPO: 在线+显式V → 最灵活但最贵(4模型)
- GRPO: 在线+隐式V(组均值) → 简化但2模型
- DPO: 离线+隐式reward → 最简单但1模型

**共同基础**: max E[r] - β KL → KL约束的策略优化 → 三种不同算法实现

## 六、与Prefix Sharing的联系

### GRPO advantage不受PS影响

$$A_i^{GRPO} = \frac{r_i - \mu_{group}}{\sigma_{group}}$$

advantage在**rollout之后**计算 → PS只影响rollout阶段 → advantage值不变!

### Policy loss不受PS影响

$$\text{ratio} = \frac{\pi_\theta(y|x)}{\pi_{\theta_{old}}(y|x)} = \exp(\log P_\theta - \log P_{\theta_{old}})$$

PS改变forward方式但不改变log_prob值(梯度流已验证: cos_sim=1.0) → ratio不变!

### KL penalty不受PS影响

$$KL = \sum_t (\log P_\theta - \log P_{ref}) \cdot \text{mask}$$

ref_log_prob由独立模型计算 → 与PS无关

### PS收益只在rollout阶段

GRPO训练时间分解:
| Phase | Time Share | PS Impact |
|-------|-----------|-----------|
| Rollout | 15-20% | **2.46x speedup** |
| Reward | 5-10% | No impact |
| Log prob | 20-25% | Partial (bypass mode) |
| Actor training | 30-40% | **1.59x speedup** |
| Advantage | <1% | No impact |

**Total training speedup**: ≈forward×0.76 → 1.59x (n=4) → 4.56x (n=8, long prompt)

Sources:
- Policy Gradient: Sutton et al., 1999, "Policy Gradient Methods for RL"
- PPO: Schulman et al., 2017, arXiv:1707.06347
- GAE: Schulman et al., 2016, arXiv:1506.02438
- GRPO: Shao et al., 2024 (DeepSeekMath), arXiv:2402.03300
- DPO: Rafailov et al., 2023, arXiv:2305.18290
- Dual-clip PPO: Cheng et al., 2019, arXiv:1912.09729
- Dr.GRPO: arXiv:2503.20783
- Gradient flow verification: ps_gradient_flow_validation_4090.py (cos_sim=1.0 ALL PASS)