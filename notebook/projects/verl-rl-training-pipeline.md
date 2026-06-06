# verl RL Training Pipeline 深度阅读

> 2026-06-07 | verl/verl/trainer/ppo/ + verl/verl/workers/engine_workers.py

## 核心架构概览

verl RL 训练使用 **RayPPOTrainer** 作为统一入口, 通过 Ray RPC 驱动分布式 worker 组完成 PPO/GRPO 训练循环.

### Worker 架构: ActorRolloutRefWorker (三合一)

```python
class ActorRolloutRefWorker(Worker):
    # role: "actor" | "rollout" | "ref" | "actor_rollout" | "actor_rollout_ref"
    self.actor: TrainingWorker   # 训练引擎 (FSDP/Megatron/etc)
    self.ref: TrainingWorker     # 参考策略 (用于KL惩罚)
    self.rollout: BaseRollout    # 生成引擎 (vLLM/SGLang/TRT-LLM async server)
```

**关键设计**: Actor + Rollout + Ref 在同一 Worker 内组合 → **colocation 模式** 节省资源 (共享 GPU).
- `actor` 和 `rollout` 共享模型权重 → sleep/wake 机制切换
- `ref_in_actor=True` → ref 模型省略, actor 自身提供 ref log_prob → **省50%内存**

### Rollout 后端选择

```python
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "vllm_rollout.ServerAdapter",
    ("sglang", "async"): "sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "trtllm_rollout.ServerAdapter",
    ("vllm", "sync"): "vllm_rollout.vllm_rollout",
    ("sglang", "sync"): "sglang_rollout.sglang_rollout",
    ("hf", "sync"): "hf_rollout.HFRollout",
    ("naive", "sync"): "naive_rollout.NaiveRollout",
}
```

**async模式**: 用 vLLM/SGLang 作为独立推理服务, 通过 RPC 通信
**sync模式**: 同进程内调用模型, 适合小规模实验

## fit() 主训练循环 (完整流程)

```
RayPPOTrainer.fit()
│
├── 初始化: _load_checkpoint() → update_weights() → _validate()
│
├── for epoch in total_epochs:
│   for batch_dict in train_dataloader:
│       │
│       ├── 1. 准备数据
│       │   batch = DataProto.from_single_dict(batch_dict)
│       │   gen_batch_output = gen_batch.repeat(rollout_n, interleave=True)
│       │   # GRPO: 每个prompt生成n个response → interleave排列
│       │
│       ├── 2. Rollout 生成 ← async_rollout_manager.generate_sequences()
│       │   # 通过vLLM/SGLang server生成response
│       │   # sleep_replicas(): 释放rollout GPU内存(给训练用)
│       │   # 产出: input_ids + responses + rollout_log_probs
│       │
│       ├── 3. 数据组装
│       │   batch = batch.repeat(rollout_n, interleave=True).union(gen_output)
│       │   # prompt × n responses 合并
│       │   compute_response_mask()
│       │   balance_batch(): 跨DP rank均衡valid token数
│       │
│       ├── 4. Reward 计算
│       │   if use_rm: _compute_reward_colocate()
│       │   reward_tensor, reward_extra_infos = extract_reward(batch)
│       │   # reward_function 或 reward_model → token_level_scores
│       │
│       ├── 5. Old Log Prob 计算 (两种模式)
│       │   ├── Bypass模式: old_log_probs = rollout_log_probs
│       │   │   # 2个策略: π_rollout(生成), π_θ(当前训练)
│       │   │   # 不需要重新计算, 直接用rollout时的log_prob
│       │   │
│       │   ├── Decoupled模式 (默认): old_log_probs = π_old(batch)
│       │   │   # 3个策略: π_rollout, π_old(锚点), π_θ(当前)
│       │   │   # π_old在mini-batch更新期间不变 → 稳定参考
│       │   │   # compute_log_prob() → actor.infer_batch()
│       │
│       ├── 6. Ref Log Prob (如果use_reference_policy)
│       │   ref_log_prob = _compute_ref_log_prob()
│       │   # ref.infer_batch() → 参考策略的log_prob
│       │
│       ├── 7. Critic Values (如果use_critic)
│       │   values = _compute_values()
│       │   # 仅PPO用, GRPO不需要critic!
│       │
│       ├── 8. KL Penalty (如果use_kl_in_reward)
│       │   apply_kl_penalty(batch, kl_ctrl, kl_penalty)
│       │   # KL divergence加入reward: r_total = r + β * KL(π_θ || π_ref)
│       │
│       ├── 9. Advantage 计算 ← 在driver进程(轻量)
│       │   compute_advantage(batch, adv_estimator, ...)
│       │   ├── GAE: compute_gae_advantage_return(values, rewards, γ, λ)
│       │   │   # A_t = Σ(γλ)^l * (r_{t+l} + γV_{t+l+1} - V_{t+l})
│       │   │
│       │   ├── GRPO: compute_grpo_outcome_advantage(rewards, mask, uid)
│       │   │   # outcome-only reward → 组内归一化
│       │   │   # A_i = (r_i - mean(r_group)) / std(r_group)
│       │   │   # 同一prompt的n个response共享advantage baseline
│       │   │
│       │   ├── GRPO_VECTORIZED: vectorized group normalization
│       │   │   # 用 group_mean_std() 高效计算组统计量
│       │   │
│       │   ├── REINFORCE_PLUS_PLUS: token-level REINFORCE + baseline
│       │   ├── RLOO: REINFORCE Leave-One-Out baseline
│       │   ├── REMAX: reward最大化(需要greedy baseline)
│       │   └── 其他: GDPO, GPG, OPO, GRPO_PASSK, etc.
│       │
│       ├── 10. Update Critic (如果use_critic)
│       │   _update_critic(batch)
│       │   # 仅PPO, critic_value_loss
│       │
│       ├── 11. Update Actor ← 核心训练步骤
│       │   _update_actor(batch)
│       │   # → actor_rollout_wg.update_actor(data)
│       │   # → TrainingWorker.train_mini_batch(data)
│       │   # → ppo_loss(config, model_output, data)
│       │   # → compute_policy_loss_{vanilla|dppo|gspo|...}
│       │
│       ├── 12. Update Weights → Rollout
│       │   checkpoint_manager.update_weights()
│       │   # 从训练引擎同步最新权重到rollout引擎
│       │   # sync模式: 直接内存拷贝
│       │   # async模式: checkpoint engine发送
│       │
│       ├── 13. 验证 (每test_freq步)
│       │   _validate()
│       │
│       └── 14. Metrics收集 + Logging
│           compute_data_metrics() + compute_timing_metrics()
│           compute_throughout_metrics() + compute_variance_proxy_metrics()
│           logger.log(metrics, step)
```

## Policy Loss Registry (9种策略损失)

```python
POLICY_LOSS_REGISTRY = {
    "vanilla":     compute_policy_loss_vanilla,    # 标准PPO clip
    "dppo_tv":     compute_policy_loss_dppo_tv,    # DPPO-Binary-TV
    "dppo_kl":     compute_policy_loss_dppo_kl,    # DPPO-Binary-KL
    "gspo":        compute_policy_loss_gspo,        # GSPO序列级重要性
    "sapo":        compute_policy_loss_sapo,        # SAPO平滑策略
    "gpg":         compute_policy_loss_gpg,         # GPG直接策略梯度
    "clip_cov":    compute_policy_loss_clip_cov,    # Clip-Covariance
    "kl_cov":      compute_policy_loss_kl_cov,      # KL-Covariance
    "geo_mean":    compute_policy_loss_geo_mean,    # GMPO几何均值
    "cispo":       compute_policy_loss_cispo,        # CISPO clipped IS
    "bypass_mode": compute_policy_loss_bypass_mode,  # off-policy IS/rejection
}
```

### Vanilla PPO Loss (核心)

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, config):
    # ratio = π_θ(a|s) / π_old(a|s) = exp(log_prob - old_log_prob)
    ratio = torch.exp(log_prob - old_log_prob)

    # PPO clip: L = -min(r*A, clip(r, 1-ε, 1+ε)*A)
    pg_losses1 = -advantages * ratio                            # 无clip
    pg_losses2 = -advantages * torch.clamp(ratio, 1-ε, 1+ε)   # 有clip
    pg_loss = max(pg_losses1, pg_losses2)  # 取更差(更大)的

    # 双clip (可选): 优势为负时额外clip ratio下限到c(>1)
    # 防止策略更新过大 → 训练稳定性

    # aggregation: token-mean 或 seq-mean-token-mean
    pg_loss = agg_loss(pg_loss, response_mask, loss_agg_mode)
```

### GRPO vs PPO 关键区别

| | PPO | GRPO |
|---|-----|------|
| Critic | 需要(value网络) | **不需要** |
| Advantage | GAE(γ,λ时序) | 组归一化(outcome-only) |
| Reward | token-level+KL | **outcome-only** |
| 模型数量 | 4(actor+critic+ref+reward) | **2**(actor+reward) |
| 内存 | 高 | **低50%+** |
| Advantage公式 | A_t=GAE递推 | A_i=(r_i-μ_group)/σ_group |
| 训练稳定性 | critic引导 | 组baseline+KL约束 |

## Advantage Estimator 全景 (14种)

```
AdvantageEstimator:
├── GAE              — Generalized Advantage Estimation (PPO标配)
│   A_t = Σ_l (γλ)^l δ_t+l, δ = r + γV_{t+1} - V_t
│
├── GRPO             — Group Relative Policy Optimization
│   outcome-only: A = (r - mean_group) / std_group
│
├── GRPO_VECTORIZED  — 高效GRPO (group_mean_std函数)
│
├── REINFORCE_PLUS_PLUS — token-level REINFORCE + value baseline
│
├── REINFORCE_PLUS_PLUS_BASELINE — 同上, 带额外baseline
│
├── REMAX            — Reward Maximization (greedy baseline对比)
│   A = r_sampled - r_greedy
│
├── RLOO             — REINFORCE Leave-One-Out
│   A_i = r_i - mean(r_j, j≠i)  # 避免自biased baseline
│
├── RLOO_VECTORIZED  — 高效RLOO
│
├── OPO              — Offline Policy Optimization
│
├── GRPO_PASSK       — GRPO with pass@k evaluation
│
├── GPG              — Group Policy Gradient (直接-log_prob*A)
│   L = -log_prob * A (无PPO clip, 无ratio)
│
├── OPTIMAL_TOKEN_BASELINE — 最优token级baseline
│
├── TIR_OPTIMAL_TOKEN_BASELINE — TIR版最优baseline
│
└── GDPO             — Group DPO (多维度reward)
```

## Sleep/Wake 权重同步机制

Colocation模式下, actor和rollout共享GPU → 需要权重同步:

```
训练前:
  rollout.sleep(level=1)     → 释放权重到CPU内存
  actor.train_mini_batch()   → 使用GPU训练

训练后:
  checkpoint_manager.update_weights()
  → rollout.wake_up()        → 从actor拷贝权重回GPU
  → rollout准备好下一轮生成
```

**Sleep 3级** (与vLLM V1一致):
- Level 0: 暂停调度(不释放内存)
- Level 1: offload权重到CPU
- Level 2: 丢弃全部GPU内存

**权重同步路径**:
- sync(naive): `actor.engine.get_per_tensor_param()` → `rollout.update_weights()`
- async: `checkpoint_engine.send_weights()` → 远程rollout server接收

## LoRA 支持

```python
# LoRA merge模式 (推荐, 零推理开销)
if model_config.lora.merge:
    # 合并LoRA到base权重 → 推理时无额外计算
    per_tensor_param = actor.engine.get_per_tensor_param(merge=True)

# LoRA adapter模式 (Punica segmented matmul)
# per_tensor_param含base+adapter → rollout需要Punica backend
```

## 数据流: DataProto + TensorDict

```python
class DataProto:
    self.batch: TensorDict       # 所有tensor数据
    self.non_tensor_batch: dict  # 非tensor数据(uid, reward_model, etc)
    self.meta_info: dict         # 元信息(eos_token_id, temperature, etc)

    # 核心操作:
    .repeat(n, interleave=True)  # 每个prompt × n responses
    .union(other)                # 合并两个DataProto
    .slice(start, end)           # 切片
    .split(n)                    # 分成n份
```

**interleave=True**: prompt_A_resp1, prompt_A_resp2, ..., prompt_B_resp1, ...
→ 同一prompt的n个response相邻排列 → GRPO组归一化高效

## PrefixGrouper (GRPO KV Cache优化)

GRPO `n` 次采样时, 同一prompt的前缀完全相同 → PrefixGrouper 分组:

```
n=8, prompt_len=512:
  无分组: 8 × 512 = 4096 prefill tokens
  有分组: 1 × 512 prefill + 8 × 64(不同response suffix) = 1024 tokens
  → 节省 75% prefill计算 + 88% KV cache内存
```

PrefixGrouper 3层优化:
1. **Prefix caching**: 共享prefix的KV只读1次
2. **Batch balancing**: 跨DP rank均衡token数
3. **Training sharing**: FSDP/Megatron共享prefix参数

## Bypass Mode vs Decoupled Mode

### Decoupled Mode (默认, 3策略)
```
π_rollout → 生成response (off-policy, 可能是旧策略)
π_old     → 计算锚点log_prob (每batch重新计算1次, mini-batch更新期间不变)
π_θ       → 当前训练策略 (mini-batch多次更新)
```
ratio = π_θ/π_old → 标准PPO重要性采样

### Bypass Mode (2策略, 更高效)
```
π_rollout → 生成response + 记录log_prob
π_θ       → 当前训练策略
```
old_log_prob = rollout_log_prob → 无需额外推理步骤
但需要IS校正 (rollout和当前策略可能不同)

## 关键代码路径索引

| 功能 | 文件 | 行号/方法 |
|------|------|-----------|
| 主训练循环 | `ray_trainer.py` | `fit()` L1362 |
| Advantage计算 | `ray_trainer.py` | `compute_advantage()` L185 |
| GRPO advantage | `core_algos.py` | `compute_grpo_outcome_advantage()` L267 |
| GAE advantage | `core_algos.py` | `compute_gae_advantage_return()` |
| Policy loss注册 | `core_algos.py` | `POLICY_LOSS_REGISTRY` L50 |
| Vanilla PPO loss | `core_algos.py` | `compute_policy_loss_vanilla()` L1278 |
| Worker类 | `engine_workers.py` | `ActorRolloutRefWorker` L434 |
| Worker init | `engine_workers.py` | `init_model()` L500 |
| Actor update | `engine_workers.py` | `update_actor()` L652 |
| Log prob计算 | `engine_workers.py` | `compute_log_prob()` L644 |
| Rollout生成 | `rollout/base.py` | `generate_sequences()` L71 |
| Rollout registry | `rollout/base.py` | `_ROLLOUT_REGISTRY` L83 |
| Weight同步 | `engine_workers.py` | `update_weights()` L667 |
| KL惩罚 | `ray_trainer.py` | `apply_kl_penalty()` L76 |
| DataProto | `utils/data_proto.py` | 核心数据容器 |
| Losses | `workers/utils/losses.py` | `ppo_loss()` L57 |

## 与 vLLM/SGLang 的集成

### vLLM Async Rollout
- vLLM 作为独立推理服务运行
- `ServerAdapter` 通过 RPC 调用 vLLM generate
- sleep/wake: vLLM 权重offload → actor训练 → vLLM权重恢复

### SGLang Async Rollout
- SGLang 作为推理服务 + RadixAttention
- PrefixGrouper + SGLang RadixAttention → 自动prefix caching
- `rollout_device_mesh`: (dp, infer_tp, infer_pp) 三维mesh

### TRT-LLM Async Rollout
- TensorRT-LLM 推理服务 (高吞吐, NVIDIA优化)
- 适合生产环境大规模部署

## 实用洞察

1. **GRPO训练只需2个模型** (actor + reward_model) → 比 PPO 省50%内存
2. **Outcome-only reward + 组归一化** → GRPO 无需token-level reward标注
3. **PrefixGrouper**: n=8/prompt=512 → 58%计算节省, 88% KV内存节省
4. **ref_in_actor=True**: actor自供ref log_prob → 省一个模型
5. **Sleep/Wake**: colocation模式核心 → 权重GPU↔CPU来回搬运
6. **14种advantage estimator**: 注册表式扩展 → 添加新算法只需`@register_adv_est`
7. **9种policy loss**: 注册表式 → 添加新loss只需`@register_policy_loss`
8. **Bypass mode**: 用rollout log_prob替代π_old → 省1次推理, 但需IS校正
9. **balance_batch**: 跨DP rank均衡 → 防止某些rank训练更多token
10. **DataProto.repeat(interleave=True)**: 同prompt response相邻 → GRPO组操作高效