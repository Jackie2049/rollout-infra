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
