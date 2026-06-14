# verl GRPO 源码深度阅读

> verl GRPO 实现的深度源码分析
> 关键发现: verl 没有独立的 GRPO trainer, 而是统一的 RayPPOTrainer 通过 `adv_estimator=grpo` 切换

## 1. 统一架构: RayPPOTrainer

verl 不包含独立的 `RayGRPOTrainer`。PPO 和 GRPO (以及 RLOO, REINFORCE++, REMAX, GDPO) 都由**统一的** `RayPPOTrainer` 处理。

**关键文件**:
| 文件 | 行数 | 作用 |
|------|------|------|
| `verl/trainer/ppo/ray_trainer.py` | 1771 | 主训练循环 |
| `verl/trainer/ppo/core_algos.py` | 2488 | 所有优势估计器 + 策略损失 |
| `verl/trainer/config/algorithm.py` | 670 | AlgoConfig 数据类 |

**算法切换**: `algorithm.adv_estimator` 配置:
- `grpo`: GRPO (Group Relative Policy Optimization)
- `gae`: PPO (Generalized Advantage Estimation)
- `rloo`: RLOO (REINFORCE Leave-One-Out)
- `reinforce_plus_plus`: REINFORCE++
- `remax`: ReMax
- `gdpo`: Group DPO

## 2. ActorRolloutRefWorker

**文件**: `verl/workers/engine_workers.py:434`

在一个 Ray worker 中组合三个角色:
- **Actor** (训练): FSDP/Megatron 训练 worker
- **Rollout** (推理): vLLM/SGLang/HF 推理引擎
- **Ref Policy** (可选): KL 惩罚参考模型

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config, role, ...):
        self.role = role  # "actor_rollout_ref"
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
```

**LoRA 优化**: `ref_in_actor=True` 时, ref policy 使用 actor 权重 (不应用 LoRA), 避免加载第二个模型。

## 3. GRPO 训练循环

### Step 1: 批处理准备 (ray_trainer.py:1435)
```python
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))])
rollout_n = self.config.actor_rollout_ref.rollout.n
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```
- 每个 prompt 获得唯一 UUID 作为 uid
- 批处理重复 `rollout_n` 次, **交错排列** (prompt_i 的所有 n 个采样相邻)

### Step 2: 生成 (ray_trainer.py:1467)
```python
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
```
- 使用 vLLM/SGLang 异步 rollout 引擎
- `rollout.n` 控制每个 prompt 的采样数

### Step 3: 奖励计算 (ray_trainer.py:1518)
```python
reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```
- 通过 `extract_reward()` 提取
- 来自奖励模型分数 (`rm_scores`) 或自定义奖励函数
- `rm_scores` shape: `(batch_size, response_length)` — 末尾 token 处有非零奖励

### Step 4: KL 惩罚
两种模式:
- **in-reward KL**: `algorithm.use_kl_in_reward=True` — 从 token 级奖励中减去 β×KL
- **in-loss KL** (GRPO 推荐): `actor.use_kl_loss=True` — 直接作为正则化项加入策略损失

### Step 5: 优势计算 (GRPO 核心)
```python
batch = compute_advantage(batch, adv_estimator="grpo", ...)
```
分派到 GRPO 特定函数 — 见 core_algos.py 分析。

## 4. core_algos.py — GRPO 优势计算

### compute_grpo_process_reward
**关键**: 组内奖励归一化

```python
# 按 uid 分组 (相同 prompt = 同一组)
uid_list = batch.non_tensor_batch['uid'].tolist()
# 对每个组:
for uid in unique_uids:
    group_indices = [i for i, u in enumerate(uid_list) if u == uid]
    group_rewards = rewards[group_indices]
    # 组内归一化
    mean_r = group_rewards.mean()
    std_r = group_rewards.std()
    advantages[group_indices] = (group_rewards - mean_r) / (std_r + eps)
```

### GRPO 策略损失
与 PPO 类似的 clipped surrogate, 但:
- 使用组归一化的 advantages (而非 GAE)
- KL 惩罚在 loss 中 (而非 reward 中)
- 无 value function

```python
# 简化版 GRPO loss
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1-clip_range, 1+clip_range) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
loss = policy_loss + kl_coef * kl_divergence
```

## 5. Prefix Sharing in GRPO Rollout

### Rollout 阶段的 Prefix Caching
```python
# 相同 prompt × n 个采样 → vLLM prefix caching
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
```
- vLLM 的 `enable_prefix_caching=True` (默认)
- 相同 prompt 复制 n 次 → KV cache 自动复用
- 每 weight update 后 `reset_prefix_cache()`

### Batch 均衡
```python
get_group_balanced_partitions()  # 按 uid 分组
```
- Karmarkar-Karp 分区算法保持 uid 组完整
- 训练时 PrefixGrouper (可选) 进一步合并前缀

## 6. GRPO vs PPO 在 verl 中的实际差异

| 方面 | PPO (adv_estimator=gae) | GRPO (adv_estimator=grpo) |
|------|------------------------|--------------------------|
| Value function | 需要 (critic 网络) | 不需要 |
| 优势估计 | GAE (λ-回报) | 组统计 (mean/std) |
| 模型数量 | 4 (π, π_ref, V, RM) | 2 (π, π_ref) + RM |
| 内存 | ~2x (critic + optimizer) | ~1x |
| KL 惩罚 | 通常 in-reward | 通常 in-loss |
| 采样数 | n=1 | n=4-64 |

## 7. 关键配置参数

```yaml
algorithm:
  adv_estimator: grpo  # 切换为 GRPO
  use_kl_in_reward: false  # GRPO 推荐 in-loss KL

actor_rollout_ref:
  rollout:
    n: 8  # 每个 prompt 采样数
  actor:
    use_kl_loss: true  # GRPO 推荐
    kl_loss_coef: 0.05  # KL 惩罚系数
    ppo_epochs: 1  # GRPO 通常 1 epoch
    clip_range: 0.2

  model:
    use_prefix_grouper: true  # 启用前缀共享
```

## 8. 核心洞察

1. **统一框架**: verl 用一个 trainer 支持所有 RL 算法, 通过 config 切换 → 优雅设计
2. **GRPO 本质**: 用组统计替代 learned baseline → 简单但有效
3. **Prefix Sharing**: GRPO n=8 天然适合 prefix caching (相同 prompt × 8 response)
4. **交错排列**: `repeat(interleave=True)` 使同一 prompt 的 response 相邻 → 适合 PrefixGrouper
5. **异步 Rollout**: rollout 和 actor 训练可以异步 (vLLM/SGLang 引擎)
6. **LoRA 优化**: ref_in_actor=True 避免加载第二个模型 → 50% 内存节省

## 9. 所有可用优势估计器 (core_algos.py)

| 估计器 | 全称 | 需要Critic | 说明 |
|--------|------|-----------|------|
| GAE | Generalized Advantage Estimation | ✓ | 标准 PPO |
| GRPO | Group Relative Policy Optimization | ✗ | 组内归一化 |
| GRPO_VECTORIZED | GRPO (向量化) | ✗ | 更高效的组操作 |
| GRPO_PASSK | Pass@k GRPO | ✗ | 基于 top-k |
| GDPO | 解耦归一化 | ✗ | 每维度独立归一化 |
| REINFORCE_PLUS_PLUS | 增强版 REINFORCE | ✗ | 折扣回报 |
| RLOO | REINFORCE Leave-One-Out | ✗ | 从其他样本剔除基线 |
| REMAX | ReMax | ✗ | 贪婪基线 |
| GPG | 广义策略梯度 | ✗ | 无裁剪 |

## 10. 所有策略损失函数 (core_algos.py:50)

| 损失 | 说明 |
|------|------|
| vanilla | 标准 PPO 裁剪损失 |
| gpg | 无裁剪 REINFORCE |
| bypass_mode | 2策略模式 (rollout=old policy) |
| dppo_tv / dppo_kl | 总变差 / KL 约束 |
| gspo | 几何均值序列级比率 |
| sapo | 平滑策略优化 |
| cispo | 裁剪重要性采样 |
| clip_cov / kl_cov | 协方差裁剪损失 |
| geo_mean | GMPO 几何均值策略 |

## 11. 权重同步机制

训练后 actor 权重需同步到 rollout 引擎:
```python
self.checkpoint_manager.update_weights(self.global_steps)
```
支持三种后端: naive (保存+加载) / NCCL (直接传输) / NIXL (RDMA)

---

## 12. GRPO Advantage 源码级深度 (core_algos.py line 267-358)

### Outcome vs Process Reward

verl GRPO 主要用 **outcome reward**:
```python
scores = token_level_rewards.sum(dim=-1)  # 每序列: sum所有token reward → 1 scalar
# outcome: 只EOS位置非零 → sum = 该序列的最终reward
```

### `compute_grpo_outcome_advantage` (line 267-331) — loop版本

```python
for each group g (identified by uid):
    μ_g = mean(scores in group g)
    σ_g = std(scores in group g)  # or 1.0 for singleton group
    for each sample i in group g:
        if norm_adv_by_std_in_grpo:    # default=True → 原始GRPO公式
            scores[i] = (scores[i] - μ_g) / (σ_g + ε)
        else:                           # Dr.GRPO variant
            scores[i] = scores[i] - μ_g  # 只减均值, 不除std

advantages = scores.unsqueeze(-1) * response_mask  # scalar broadcast到所有tokens
```

关键:
- Singleton (n=1): μ=0, σ=1 → advantage=raw reward → 无normalization
- advantage是scalar per sequence → broadcast到所有response tokens
- `norm_adv_by_std_in_grpo=True` → 标准GRPO; `False` → Dr.GRPO

### `compute_grpo_vectorized_outcome_advantage` (line 334-358) — 纯PyTorch版

```python
g = as_torch_index(uid)          # UUIDs → contiguous 0..G-1
mean_g, std_g, _ = group_mean_std(scores, g)  # scatter-add, 无Python loop!

if norm_adv_by_std_in_grpo:
    scalars = (scores - mean_g[g]) / (std_g[g] + ε)
else:
    scalars = scores - mean_g[g]
advantages = scalars.unsqueeze(-1) * response_mask
```

**`group_mean_std` (groupwise.py line 164)**: 纯PyTorch scatter
- `torch.zeros(G).index_add_(0, gidx, values)` → sum + sum_sq
- `count.clamp_min(1)` → mean; `(count-1).clamp_min(1)` → Bessel-corrected variance
- Singleton fallback: mean=0, std=1

---

## 13. KL Penalty 详细类型 (core_algos.py line 2126)

| Type | 公式 | 特性 |
|------|------|------|
| `kl` / `k1` | `logprob - ref_logprob` | 简单log-ratio |
| `abs` | `|logprob - ref_logprob|` | 绝对差 |
| `mse` / `k2` | `0.5*(logprob-ref)^2` | **无偏梯度** |
| `low_var_kl` / `k3` | `exp(ref-log) - (ref-log) - 1` | **低方差估计**(clipped) |
| `k3+` / `k1+` | fwd用k1/k3值+bwd用k2梯度 | Straight-through trick |

默认: `use_kl_in_reward=False`, `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl`

---

## 14. PPO-Clip + Dual-Clip Loss (core_algos.py line 1279)

```python
ratio = exp(log_prob - old_log_prob)  # importance ratio

pg_losses1 = -advantages * ratio              # unclipped
pg_losses2 = -advantages * clip(ratio, 1-ε, 1+ε)  # clipped
clip_pg_losses1 = max(pg_losses1, pg_losses2)      # PPO-clip

# Dual-clip (负advantage时防止ratio过小):
pg_losses3 = -advantages * clip_ratio_c     # c > 1.0 (默认3.0)
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)

pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

Dual-clip来源: https://arxiv.org/pdf/1912.09729 → 防止负advantage时ratio过小

IS weights可选: `pg_losses * rollout_is_weights`

---

## 15. Loss Aggregation 四模式 (core_algos.py line 1138)

| Mode | 计算 | 使用场景 |
|------|------|----------|
| `token-mean` | Sum所有masked / batch_num_tokens | **verl GRPO默认** |
| `seq-mean-token-sum` | Sum per seq → mean across seqs | - |
| `seq-mean-token-sum-norm` | 同上 / loss_scale_factor | **Dr.GRPO** (防length bias) |
| `seq-mean-token-mean` | Mean per seq → mean across seqs | 原始论文, verl警告不稳定 |

---

## 16. GRPO 变体详解

### Dr.GRPO (`norm_adv_by_std_in_grpo=False`)
- 只mean-centering → 不除std → 消除length bias
- 配合 `loss_agg_mode=seq-mean-token-sum-norm` + `use_kl_loss=False`

### GRPO-passk (`adv_estimator=grpo_passk`)
- 只有best response per group有非零advantage: `r_max - r_second_max`
- 其他completions advantage=0

### GDPO (Group reward-Decoupled Normalization)
- 每reward维度独立normalization → weighted sum
- 需要 `gdpo_reward_keys` (如 ["format_reward", "accuracy_reward"])
- reward function返回per-dimension scores

### GSPO loss
- Geometric-mean sequence-level importance ratio
- `s_i = (π_θ/π_old)^{1/|y_i|}` (length-normalized)
- combined ratio = `sg[s_i] * ratio_token / sg[ratio_token]`
- 推荐 `loss_agg_mode=seq-mean-token-mean`

### Vectorized GRPO (`grpo_vectorized`)
- Python loop → pure PyTorch scatter (`index_add_`) → 大batch更快

### Filter Groups (DAPO)
- `FilterGroupsConfig`: 过滤group by metric (acc/score/seq_reward)
- `max_num_gen_batches` 控制上限
- DAPO reward manager: overlong response penalties

---

## 17. Rollout Correction System

- **2-policy bypass**: `π_rollout = π_old` (同一policy生成+训练)
- **3-policy decoupled**: `π_rollout` (生成), `π_old` (PPO参考), `π_θ` (当前训练)
- IS weights: token-level or sequence-level importance sampling
- Rejection sampling: k1/k2/k3/geometric 配置

---

## 18. Balance Batch & Dynamic BSZ

```python
# trainer.balance_batch=True → 重新分配数据跨DP ranks → 等化valid token counts
# → 改变data order但uid-based advantage不受影响

# use_dynamic_bsz=True → 调整per-GPU micro-batch size → 防variable-length OOM
```
