# verl GRPO Training Core Algorithms — core_algos.py 源码深度阅读

> 2026-06-07 | 2487行源码分析: 14种Advantage Estimator + 10种Policy Loss + 注册表架构

## 核心架构: 注册表式扩展

### 双注册表系统

```python
# Advantage Estimator 注册表
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}
@register_adv_est(AdvantageEstimator.GRPO)  # 装饰器注册

# Policy Loss 注册表
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}
@register_policy_loss("vanilla")  # 装饰器注册
```

**设计优势**: 新算法只需写函数+装饰器 → 零修改框架代码 → 可插拔扩展

### 调用路径

```
RayPPOTrainer.fit()
  → compute_advantage()  # 根据 adv_estimator 名从注册表查找
  → compute_policy_loss()  # 根据 loss_name 名从注册表查找
```

## 一、14种 Advantage Estimator

### 1. GAE (PPO标准, 行215)

```python
A_t^{GAE} = Σ(γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

- 需要 critic V(s) → 4模型
- γ=discount, λ=GAE parameter
- 最灵活但最昂贵

### 2. GRPO (组相对, 行267) — **核心!**

```python
scores = token_level_rewards.sum(dim=-1)  # outcome-only → 整个response的总reward
# 组内统计
id2mean[idx] = mean(scores_for_same_prompt)
id2std[idx] = std(scores_for_same_prompt)
# 组归一化
scores[i] = (scores[i] - id2mean) / (id2std + epsilon)
# broadcast到token level
advantages = scores.unsqueeze(-1) * response_mask
```

**关键设计决策**:
1. **Outcome-only**: `scores = rewards.sum()` → 只看最终reward, 不区分哪个token贡献
2. **组归一化**: `(r_i - μ_group) / σ_group` → 同prompt的n个response互为baseline
3. **单样本组特殊处理**: `len(id2score)==1 → mean=0, std=1` → 单样本时advantage=原reward
4. **Dr.GRPO选项**: `norm_adv_by_std_in_grpo=False` → 不除std → 更保守(防止过度归一化)

### 3. GRPO_VECTORIZED (行334)

```python
# 用torch-level group操作替代Python for循环
g = as_torch_index(index, device=scores.device)
mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0)
scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
```

**优化**: Python循环→torch向量操作 → 大batch显著加速

### 4. GDPO (组奖励解耦归一化, 行361)

```python
# 不是先sum再归一化(GRPO), 而是每维度独立归一化再sum!
A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)  # 每个reward维度独立
A = Σ_k A_k  # 维度间解耦
```

**优势**: 防止主导reward信号淹没弱信号 → 多reward维度场景

### 5. GRPO_PASSK (行471)

```python
# top-k samples only → pass@k 评估导向
# 只保留每组最好的k个response → 更高信号质量
```

### 6. REMAX (行732)

```python
# 在当前策略上重新采样 → 用最优response的reward作为baseline
# baseline = max(reward from current policy)
```

### 7. GPG (直接策略梯度, 行768)

```python
# 无clipping, 无ratio → 直接∇logπ · A
# 最简单的REINFORCE variant
```

### 8-14. 其他

- **REINFORCE_PLUS_PLUS**: reward baseline归一化
- **REINFORCE_PLUS_PLUS_BASELINE**: reward + learned baseline
- **RLOO**: Leave-One-Out baseline (组内其他样本均值)
- **OPO**: Online Policy Optimization
- **RLOO_VECTORIZED**: 向量化RLOO
- **OPTIMAL_TOKEN_BASELINE**: token-level最优baseline(理论推导)
- **TIR_OPTIMAL_TOKEN_BASELINE**: TIR(Trust-region)优化token baseline

## 二、10种 Policy Loss

### 1. Vanilla (PPO-clip, 行1278) — **标准!**

```python
ratio = exp(log_prob - old_log_prob)  # π_new / π_old
pg_losses1 = -advantages * ratio  # 无clip
pg_losses2 = -advantages * clamp(ratio, 1-ε, 1+ε)  # clip
clip_pg_losses1 = max(pg_losses1, pg_losses2)

# 双clip (Cheng et al., 1912.09729)
pg_losses3 = -advantages * clip_ratio_c  # 下界clip (A<0时)
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)

pg_losses = where(A < 0, clip_pg_losses2, clip_pg_losses1)
```

**特性**: 标准PPO clip + 可选双clip(防止策略在A<0时过度偏离)

### 2. DPPO-TV (行1372)

```python
# Total Variation distance constraint
# TV(π_new, π_old) ≤ ε → 替代PPO的ratio clip
# 更理论化的约束
```

### 3. DPPO-KL (行1453)

```python
# KL divergence constraint
# KL(π_new || π_old) ≤ δ → 替代PPO的ratio clip
# 更严格的策略变化限制
```

### 4. GSPO (行1538) — 序列级clip

```python
# 序列级ratio → 不按token clip, 按整个序列clip
# 对长序列更稳定(避免逐token clip过度限制)
```

### 5. SAPO (行1614) — 平滑advantage

```python
# 平滑advantage → 避免极端advantage值
# 用soft clipping或log变换
```

### 6. GPG (行1699) — 直接梯度

```python
# 无ratio, 无clip → 直接 A · ∇logπ
# 最简单的REINFORCE → 用于GPG advantage estimator
```

### 7. Clip-Cov (行1735) — 带协方差的clip

```python
# 在clip之上加协方差约束
# 防止高variance的ratio导致不稳定
```

### 8. KL-Cov (行1840) — KL+协方差双约束

### 9. Geo-Mean (行1920) — 几何均值ratio

```python
# ratio = (π_new/π_old)^(1/N) → 几何均值而非逐token
# 序列级ratio → 更稳定
```

### 10. CISPO (行2006) — Clipped IS Policy Optimization

```python
# Clipped Importance Sampling → 在IS权重上clip而非ratio上
# 用于bypass mode (off-policy场景)
```

### 11. Bypass Mode (行2351) — **Off-policy关键!**

```python
# In bypass mode: old_log_prob = rollout_log_prob (不是actor_log_prob)
# 两种子模式:
#   "ppo_clip": PPO clip with IS ratio π_current/π_rollout (no additional IS weights)
#   "reinforce": REINFORCE with IS weights w = π_current/π_rollout
# IS aggregation: token-level, sequence-level, or None
# IS threshold: truncation at rollout_is_threshold (default 2.0)
```

## 三、Loss Aggregation (agg_loss, 行1138)

| Mode | 计算 | 用途 |
|------|------|------|
| token-mean | Σ(loss·mask)/Σ(mask_tokens) | 标准PPO |
| seq-mean-token-sum | Σ(Σ(loss·mask))/batch_size | 序列级 |
| seq-mean-token-sum-norm | seq-mean-token-sum/horizon | 归一化 |
| seq-mean-token-mean | Σ(Σ(loss·mask)/tokens)/batch_size | per-seq mean |

**关键**: `dp_size`参数 → 确保FSDP/Megatron DP下loss等价 → 分布式一致性

## 四、GRPO vs PPO 生产对比

| 方面 | PPO (GAE + vanilla) | GRPO (GRPO + vanilla) |
|------|---------------------|----------------------|
| 模型数 | 4 (actor+critic+ref+reward) | 2 (actor+ref) |
| Advantage | GAE(γλ折扣) | 组归一化(outcome-only) |
| Reward | Step-level (每token) | Outcome-level (整个response) |
| Critic | 必须训练V(s) | **不需要** → 省50%内存 |
| 长推理 | 不擅长 (step reward难以定义) | **擅长** (outcome-only自然涌现) |
| 稳定性 | clip_ratio参数敏感 | 组归一化更鲁棒 |
| 前缀共享 | 不适用 | n=8共享prefix → 58%KV节省 |

## 五、与我们的工作联系

### Prefix Sharing对GRPO的优势

GRPO n=8 → 同prompt采样8个response → prefix共享:
- PrefixGrouper (PR #4368): 仅attention层 → 0.99x加速
- Full-model PS: Provider完整forward + Reuser suffix-only → 2.46x加速
- 训练加速: 1.59x (×0.76打折)

### core_algos.py对PS的影响

1. **advantage计算不受PS影响**: GRPO advantage = (r - μ_group)/σ_group → 在reward计算之后 → PS只影响rollout阶段
2. **policy loss不受PS影响**: ratio = π_new/π_old → 与forward方式无关
3. **KL penalty**: `r(x,y) - β KL(π_new || π_ref)` → ref_log_prob需要独立计算 → **但ref_in_actor=True可以省一个模型**

### verl #6401贡献机会

core_algos.py的注册表设计意味着:
- 新的advantage estimator: 只需写函数 + `@register_adv_est`
- 新的policy loss: 只需写函数 + `@register_policy_loss`
- PS相关的修改: 不在core_algos.py → 在rollout层(ActorRolloutRefWorker)

Sources:
- verl core_algos.py: verl/trainer/ppo/core_algos.py (2487行)
- GRPO原论文: DeepSeekMath (Shao et al., 2024), arXiv:2402.03300
- PPO-clip: Schulman et al., 2017, arXiv:1707.06347
- Dual-clip PPO: Cheng et al., 2019, arXiv:1912.09729
- Dr.GRPO: arXiv:2503.20783 (norm_adv_by_std_in_grpo=False)
- GDPO: Group reward-Decoupled Normalization