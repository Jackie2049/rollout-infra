# verl GRPO Training Loop Internals — Source Code Level Deep Reading

> 源码版本: verl main branch (2026-06), v0.8.0+
> 阅读: 2026-06-15
> 重点关注: GRPO training step, compute_advantage → ppo_loss 数据变换链, CoreActorRolloutRefWorker.update_policy(), LoRA adapter管理, RewardManager, bypass_mode

---

## 1. GRPO Training Step 全流程 (两代 Trainer)

### 1.1 Legacy RayPPOTrainer (`verl/trainer/ppo/ray_trainer.py`)

`RayPPOTrainer.fit()` 中的 `_train_step` 核心:

```
rollout → reward → balance_batch → old_log_prob → ref_log_prob → compute_advantage → update_actor → update_weights
```

详细步骤 (ray_trainer.py ~L1500-1565):

| 步骤 | 代码位置 | 说明 |
|------|----------|------|
| 1. Rollout生成 | rollout_wg.generate_sequences() | vLLM/SGLang生成responses+rollout_log_probs |
| 2. Reward计算 | `_compute_reward_colocate()` 或 `_compute_reward()` | NaiveRewardManager或BatchRewardManager |
| 3. Balance batch | `_balance_batch()` | seqlen balancing跨DP组 |
| 4. old_log_prob | `_compute_old_log_prob()` 或 **bypass_mode** | 关键分叉点! 见第6节 |
| 5. ref_log_prob | `_compute_ref_log_prob()` | 可选; LoRA时用actor+disable_adapter |
| 6. KL penalty | `apply_kl_penalty()` | 可选; use_kl_in_reward=True时 |
| 7. Advantage | `compute_advantage()` | GRPO: group-relative; GAE: value-based |
| 8. Update actor | `_update_actor()` | ppo_epochs × mini_batch_size |
| 9. Update weights | `checkpoint_manager.update_weights()` | trainer→rollout权重同步 |

### 1.2 V1 PPOTrainer (`verl/trainer/ppo/v1/trainer_base.py`)

V1用**TransferQueue**替代DataProto直传, 更高效的KV存储:

```python
# trainer_base.py L384-430 step()方法
def step(self, metrics, timing_raw):
    # 1. add batch to generate
    self._add_batch_to_generate()
    # 2. sample from replay_buffer
    batch = self.replay_buffer.sample(partition_id="train")
    # 3. reward (colocated or reward_loop)
    batch = self._compute_reward_colocate(batch)
    # 4. balance batch
    batch = self._balance_batch(batch, metrics)
    # 5. old_log_prob (bypass or recompute)
    batch = self._compute_old_log_prob(batch, metrics)
    # 6. ref_log_prob (optional)
    batch = self._compute_ref_log_prob(batch, metrics)
    # 7. values (optional, GAE only)
    batch = self._compute_values(batch, metrics)
    # 8. advantage
    batch = self._compute_advantage(batch, metrics)
    # 9. update critic (optional)
    batch = self._update_critic(batch, metrics)
    # 10. update actor
    batch = self._update_actor(batch, metrics)
    return batch
```

**关键区别**: V1用`tq.kv_batch_get/put`存取中间数据, 避免大tensor反复拷贝. `KVBatchMeta`只存key+partition_id, 实际数据在TransferQueue中.

---

## 2. compute_advantage → ppo_loss 数据变换链

### 2.1 compute_advantage (ray_trainer.py L187-350)

入口函数, 根据`adv_estimator`分发:

```python
def compute_advantage(data, adv_estimator, gamma, lam, num_repeat, norm_adv_by_std_in_grpo, config):
    # 确保response_mask存在
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == AdvantageEstimator.GAE:
        # 需要 values, gamma, lambda
        advantages, returns = core_algos.compute_gae_advantage_return(...)
    elif adv_estimator == AdvantageEstimator.GRPO:
        # 只需要 token_level_rewards, response_mask, uid
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],  # ← group key!
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        # 通用分发器, 其他13种adv estimator
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        advantages, returns = adv_estimator_fn(**adv_kwargs)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data
```

### 2.2 GRPO Advantage核心计算 (core_algos.py L275-340)

**compute_grpo_outcome_advantage** 的完整数据流:

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index,
                                    epsilon=1e-6, norm_adv_by_std_in_grpo=True, config=None):
    # Step 1: Sum rewards across tokens → scalar per trajectory
    scores = token_level_rewards.sum(dim=-1)  # (bs,) outcome reward

    # Step 2: Group by uid (prompt index), compute mean/std per group
    id2score = defaultdict(list)  # uid → [score_1, score_2, ..., score_n]
    id2mean = {}
    id2std = {}
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = 0.0; id2std[idx] = 1.0  # 单样本: baseline=0, 不除std
        else:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    # Step 3: Normalize within group
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:  # 原始GRPO
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # = (r_i - μ_g) / (σ_g + ε)
        else:  # Dr.GRPO
            scores[i] = scores[i] - id2mean[index[i]]
            # = r_i - μ_g  (不除std, 防止小group梯度消失)

    # Step 4: Broadcast to token dimension
    scores = scores.unsqueeze(-1) * response_mask  # (bs, response_len)

    return scores, scores  # advantage = return for GRPO (outcome-only)
```

**数学公式**:
- GRPO: `a_i = (r_i - μ_g) / (σ_g + ε)`, 其中 μ_g/σ_g 是同prompt所有response的group统计量
- Dr.GRPO: `a_i = r_i - μ_g`, 不除σ_g, 防止小group(n=2)时σ→0导致梯度消失
- 最终: `advantage = a_i * response_mask`, 非EOS位置均为同一个scalar值

### 2.3 GRPO_VECTORIZED (core_algos.py L341-370)

向量化版本, 用`group_mean_std`替代Python loop, **10-100x faster**:

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(token_level_rewards, response_mask, index, ...):
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)  # (bs,)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages
```

**group_mean_std**用scatter操作一次计算所有group的统计量, 完全GPU端, 无CPU round-trip.

### 2.4 其他13种Advantage Estimator

| Estimator | 类别 | 核心公式 | 特点 |
|-----------|------|----------|------|
| GAE | value-based | δ_t + γλδ_{t+1} | 需critic; TD(λ) |
| GRPO | outcome | (r-μ)/σ | group baseline; ★推荐 |
| GRPO_VECTORIZED | outcome | 同GRPO但GPU端 | 10-100x快; ★推荐 |
| GRPO_PASSK | outcome | r_max - r_2nd_max | 只best response; n≥2 |
| GDPO | per-dimension | 每维度独立normalize+加权聚合 | 防dominant reward |
| REINFORCE++ | token-level | γ-discounted return | 无group baseline |
| REINFORCE++_BASELINE | outcome | r-μ_g + whiten | RF++ + group baseline |
| RLOO | outcome | n/(n-1)*r_i - n/(n-1)*μ_g | leave-one-out baseline |
| RLOO_VECTORIZED | outcome | 同RLOO但向量化 | GPU端 |
| OPO | outcome | length-weighted baseline | 防length hack |
| REMAX | outcome | G_t - baseline | greedy baseline |
| GPG | outcome | α*(r-μ)/f_norm | controlled norm |
| OPTIMAL_TOKEN_BASELINE | token-level | G_t - B_t* (path-variance) | per-step最优baseline |

### 2.5 ppo_loss → policy_loss 数据流 (losses.py L47-110)

`ppo_loss()`是wrapper, 内部调用`compute_policy_loss_vanilla`或`bypass_mode`:

```python
def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = no_padding_2_padding(model_output["log_probs"], data)  # 当前policy的log prob
    entropy = model_output.get("entropy", None)

    # 设置global_batch_info用于agg_loss
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]

    # 选择fields
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data: fields.append("rollout_is_weights")
    if "ref_log_prob" in data: fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    # 选择policy loss function
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)  # POLICY_LOSS_REGISTRY查找

    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,    # π_old的log prob (或π_rollout if bypass)
        log_prob=log_prob,             # π_θ当前forward的log prob
        advantages=advantages,         # 从compute_advantage得到
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,  # IS correction weights (可选)
    )

    # 加entropy bonus
    if entropy is not None:
        entropy_loss = agg_loss(entropy, response_mask, loss_agg_mode, **config.global_batch_info)
        policy_loss -= config.entropy_coeff * entropy_loss

    # 加KL loss (loss端, 不是reward端)
    if config.use_kl_loss:
        kld = kl_penalty(log_prob, ref_log_prob, config.kl_loss_type)
        kl_loss = agg_loss(kld, response_mask, loss_agg_mode, **config.global_batch_info)
        policy_loss += config.kl_loss_coef * kl_loss

    return policy_loss, metrics
```

**最终loss公式**:
```
total_loss = pg_loss - entropy_coeff * entropy_loss + kl_loss_coef * kl_loss
```
其中:
- `pg_loss` = PPO clipped objective (vanilla/bypass_mode)
- `entropy_loss` = -avg(entropy) → 减去 → 鼓励探索
- `kl_loss` = avg(KL(π_θ || π_ref)) → 加入 → 防止偏离ref

### 2.6 compute_policy_loss_vanilla (core_algos.py L1278-1372)

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    # Step 1: Compute ratio
    negative_approx_kl = log_prob - old_log_prob  # log(π_θ/π_old)
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # 数值稳定
    ratio = torch.exp(negative_approx_kl)  # π_θ/π_old

    # Step 2: PPO clipped objective
    pg_losses1 = -advantages * ratio  # -ratio * A
    pg_losses2 = -advantages * torch.clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)  # -clip(r)*A

    # Step 3: Dual-clip (for negative advantages)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c  # 下界
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    # Step 4: Choose based on advantage sign
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Step 5: Aggregate
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode, **config.global_batch_info)
    return pg_loss, pg_metrics
```

**Dual-clip PPO** (https://arxiv.org/pdf/1912.09729):
- A>0时: max(-r*A, -clip(r,1-ε,1+ε)*A) → 标准PPO clip, 防止r过大
- A<0时: min(-clip_ratio_c*A, max(-r*A, -clip(r,1-ε,1+ε)*A)) → 双clip, 防止r过小导致意外大update
- clip_ratio_c默认3.0, 必须>1.0

### 2.7 agg_loss 聚合模式 (core_algos.py L1138-1205)

4种聚合方式保证**FSDP/Megatron DP不变性**:

| 模式 | 公式 | 特点 |
|------|------|------|
| `token-mean` | Σ(loss*mask) / global_tokens * dp_size | ★默认; 全局token平均 |
| `seq-mean-token-sum` | Σ_seq(Σ_tok(loss*mask)) / global_bsz * dp_size | 先token-sum再seq-mean |
| `seq-mean-token-sum-norm` | 上者 / loss_scale_factor | 加horizon normalization |
| `seq-mean-token-mean` | Σ_seq(Σ_tok/valid_tok) / global_bsz * dp_size | 先token-mean再seq-mean |

关键: `global_batch_info`确保DP多GPU时loss值一致 — `dp_size`因子补偿每GPU只看到1/dp_size的数据.

---

## 3. CoreActorRolloutRefWorker — update_policy()详解

### 3.1 类层次

```
ActorRolloutRefWorker (verl/workers/engine_workers.py L434)
    extends Worker + DistProfilerExtension
    contains: self.actor (TrainingWorker), self.ref (TrainingWorker), self.rollout (BaseRollout)
```

**注意**: ActorRolloutRefWorker **没有独立的 `update_policy()` 方法**! 它通过`update_actor()` → `self.actor.train_mini_batch()` → `self.actor.train_batch()` → `engine.train_batch(data, loss_function=self.loss_fn)` 完成训练.

### 3.2 update_actor调用链

```
ray_trainer._update_actor(batch)
    → batch.to_tensordict() → left_right_2_no_padding(batch_td)
    → assign metadata: calculate_entropy, global_batch_size, mini_batch_size, epochs, seed, shuffle
    → actor_rollout_wg.update_actor(batch_td)     # Ray RPC
        → ActorRolloutRefWorker.update_actor()     # engine_workers.py L650
            → self.actor.train_mini_batch(data)    # TrainingWorker.train_mini_batch() L234
                → make_iterator(data, mini_batch_size, epochs, seed)
                → for mini_batch in iterator:
                    → self.train_batch(mini_batch)  # TrainingWorker.train_batch() L325
                        → self.engine.train_batch(data, loss_function=self.loss_fn)
                            → engine内部forward → compute log_prob → ppo_loss → backward → optimizer step
```

### 3.3 TrainingWorker.train_mini_batch (engine_workers.py L234-324)

```python
def train_mini_batch(self, data: TensorDict):
    mini_batch_size = data.pop("mini_batch_size")
    epochs = data.pop("epochs", default=1)
    seed = data.pop("seed", default=42)

    # 计算mini_batch_size_per_gpu (除以dp_size)
    mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

    # 构造迭代器: epochs × mini_batch_size_per_gpu
    dataloader = make_iterator(data, mini_batch_size_per_gpu, epochs, seed)

    with self.engine.train_mode():
        for batch_idx, mini_batch_td in enumerate(dataloader):
            # 收集global_token_num (跨DP all_gather)
            tu.assign_non_tensor(mini_batch_td,
                global_token_num=NonTensorData(global_token_num),
                update_lr_scheduler=(batch_idx == total_num_iterations - 1),  # 最后一步才更新lr
            )
            actor_output = self.train_batch(mini_batch_td)  # 单mini_batch训练
            output_lst.append(actor_output)

    # 聚合所有mini_batch的metrics
    metrics = aggregate_metrics(output_lst)
    return output
```

**关键**: `update_lr_scheduler`只在最后一个mini_batch更新 → 避免中间step误更新lr.

### 3.4 TrainingWorker.train_batch (engine_workers.py L325-380)

```python
def train_batch(self, data: TensorDict):
    # 1. forward + backward + optimizer step
    with self.engine.train_mode():
        output = self.engine.train_batch(data, loss_function=self.loss_fn)
        # output包含: loss, model_output, metrics

    # 2. lr scheduler update (only if update_lr_scheduler=True)
    if update_lr_scheduler:
        lr = self.engine.lr_scheduler_step()
    else:
        lr = None

    # 3. 只保留metrics, 丢弃model_output
    output.pop("model_output")
    if lr is not None:
        output["metrics"]["lr"] = lr
    return self._postprocess_output(output).cpu()
```

### 3.5 Loss Function注入

```python
# engine_workers.py init_model() L560-565
if self.distillation_enabled:
    self.loss_fn = partial(distillation_ppo_loss, config=actor_config, distillation_config=...)
else:
    self.loss_fn = partial(ppo_loss, config=actor_config)  # ← 标准PPO loss
self.actor.set_loss_fn(self.loss_fn)
```

`ppo_loss`来自`verl/workers/utils/losses.py`, 见第2.5节详解.

### 3.6 compute_log_prob (engine_workers.py L644-650)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def compute_log_prob(self, data: TensorDict) -> TensorDict:
    output = self.actor.infer_batch(data)
    return output.cpu() if output is not None else None
```

`infer_batch`调用`engine.infer_batch()` → 只forward不backward → 返回log_probs + entropy.

### 3.7 compute_ref_log_prob (engine_workers.py L637-643)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
    output = self.ref.infer_batch(data=data)
    return output.cpu() if output is not None else None
```

如果`ref_in_actor=True` (LoRA模式), **不调用此方法**, 而用actor + disable_adapter, 见第4节.

---

## 4. LoRA Adapter管理 — enable/disable for ref vs actor

### 4.1 ref_in_actor判断 (ray_trainer.py L128-131)

```python
lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
if lora_rank <= 0:
    lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
```

当LoRA rank>0或lora_adapter_path存在时 → **ref_in_actor=True** → 用同一个actor模型计算ref_log_prob.

### 4.2 _compute_ref_log_prob中的LoRA处理

**Legacy** (ray_trainer.py L1229-1251):
```python
def _compute_ref_log_prob(self, batch):
    batch_td = batch.to_tensordict()
    batch_td = left_right_2_no_padding(batch_td)
    metadata = {"calculate_entropy": False, "compute_loss": False}
    if self.ref_in_actor:
        metadata["no_lora_adapter"] = True  # ← 关键flag!
    tu.assign_non_tensor(batch_td, **metadata)
    if self.ref_in_actor:
        # 用actor模型 + disable_adapter → 得到base model的log_probs → 就是ref!
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
    else:
        # 用独立的ref model
        output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
```

**V1** (trainer_base.py L1195-1210):
```python
def _compute_ref_log_prob(self, batch, metrics):
    metadata = {
        "calculate_entropy": False,
        "compute_loss": False,
        "temperature": self.config.actor_rollout_ref.rollout.temperature,
    }
    if self.ref_in_actor:
        metadata["no_lora_adapter"] = True  # ← 同样flag
    batch.extra_info.update(metadata)
    if self.ref_in_actor:
        output = self.actor_rollout_wg.compute_log_prob(batch)  # actor + LoRA disabled
    else:
        output = self.ref_policy_wg.compute_ref_log_prob(batch)  # 独立ref worker
```

### 4.3 disable_adapter在engine层实现 (engine_workers.py L385-407)

```python
# TrainingWorker.infer_batch() 中的LoRA处理:
no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
with self.engine.eval_mode():
    adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
    with adapter_ctx:
        output = self.engine.infer_batch(data, loss_function=loss_function)
```

**关键**: `self.engine.disable_adapter()` 是PEFT的context manager, 临时禁用所有LoRA adapter → forward用**base model权重** → 得到的log_probs就是π_ref.

### 4.4 LoRA + GRPO的ref_in_actor优势

```
无LoRA: 需要2个GPU → actor GPU + ref GPU (或ref shard)
有LoRA + ref_in_actor:
    actor forward (LoRA enabled) → π_θ的log_probs
    actor forward (LoRA disabled) → π_ref的log_probs  ← 同一GPU! 零额外GPU!
```

**RTX 4090关键**: `ref_in_actor=True` → **单GPU可以做actor+ref** → GRPO唯一可行路径!

### 4.5 Weight Sync时的LoRA处理 (engine_workers.py L700-758)

```python
async def update_weights(self, global_steps=None, mode="auto"):
    effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

    if effective_mode != "naive":
        # 异步模式: checkpoint engine send_weights
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)
        return

    # 同步co-located模式: 直接从actor engine→rollout
    # 1. resume rollout weights (sleep后恢复)
    await self.rollout.resume(tags=["weights"])

    # 2. 获取参数
    per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=True
    )

    # 3. LoRA: 如果peft_merge=True, merge LoRA到base → rollout收到merged权重
    do_lora_base_sync = False
    if not self.peft_merge and peft_config is not None:
        do_lora_base_sync = not self.base_sync_done  # 只sync一次base weight

    if do_lora_base_sync:
        # 第一次: 先sync base weights
        per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=False
        )
        await self.rollout.update_weights(per_tensor_param_base, peft_config=peft_config, base_sync_done=False)

    # 第二次: sync merged weights (或adapter weights)
    await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True)
    self.base_sync_done = True
```

**peft_merge**:
- `True`: LoRA merge到base → rollout收merged HF权重 → 标准weight update
- `False`: LoRA不merge → rollout需base+adapter分开sync → SGLang/vLLM的Punica机制

---

## 5. RewardManager — Outcome Reward注入

### 5.1 RewardManager架构

```
AbstractRewardManager (verl/workers/reward_manager/abstract.py)
    ├── NaiveRewardManager (naive.py) — 逐样本串行计算, CPU端
    ├── BatchRewardManager (batch.py) — 批量计算, 但仍串行verify
    ├── DAPORewardManager (dapo.py) — DAPO特定
    └── PrimeRewardManager (prime.py) — PRIME特定
```

注册机制: `@register("naive")` → `RewardManagerRegistry` → 配置选择.

### 5.2 NaiveRewardManager核心 (naive.py)

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", compute_score_timeout=None):
        self.compute_score = compute_score or default_compute_score
        self.compute_score_timeout = compute_score_timeout  # SIGALRM timeout!

    def __call__(self, data: DataProto, return_dict=False):
        # 如果已有rm_scores → 直接返回 (来自reward model)
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            # 解码prompt和response
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            try:
                with _score_timeout(self.compute_score_timeout):  # SIGALRM防卡死!
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
            except TimeoutError:
                score = 0.0  # timeout → 默认reward=0

            if isinstance(score, dict):
                reward = score["score"]  # GDPO: 多维度reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score  # 标准outcome reward

            # ★ 关键: reward只放在最后一个token位置!
            reward_tensor[i, valid_response_length - 1] = reward  # (bs, response_len) 全0 except最后一个位置

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
```

### 5.3 Outcome Reward的token分布

**GRPO用outcome reward**: 只有response**最后一个有效token**有reward值, 其他位置全0.

```
reward_tensor[i, :] = [0, 0, 0, ..., 0, 0.73, 0, 0]  # 只有EOS位置有值
```

`compute_grpo_outcome_advantage`中:
```python
scores = token_level_rewards.sum(dim=-1)  # sum across tokens → 只EOS位置贡献
# → outcome reward = r_i (scalar per trajectory)
```

然后broadcast: `scores.unsqueeze(-1) * response_mask` → **所有有效token共享同一个advantage值**.

### 5.4 Reward注入到训练数据的完整流程

```
1. Rollout生成 → batch.batch["responses"] + batch.batch["rollout_log_probs"]
2. RewardManager.__call__(batch) → reward_tensor (只有EOS位置有值)
3. extract_reward(batch) → reward_tensor, reward_extra_infos_dict
4. batch.batch["token_level_scores"] = reward_tensor
5. (可选) apply_kl_penalty → batch.batch["token_level_rewards"] = token_level_scores - β*KL
6. (默认GRPO) batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
7. compute_advantage → batch.batch["advantages"] + batch.batch["returns"]
```

### 5.5 RewardManager选择 (ray_trainer.py reward.py)

```python
def load_reward_manager(config, tokenizer, **reward_kwargs):
    compute_score = get_custom_reward_fn(config)  # 用户自定义reward function
    final_compute_score = compute_score
    if compute_score is None:
        # 使用default_compute_score (rule-based, CPU端)
        final_compute_score = default_compute_score

    reward_manager_cls = resolve_reward_manager_cls(config)  # 从registry获取class
    return reward_manager_cls(config=config, tokenizer=tokenizer, compute_score=final_compute_score)
```

**两种reward来源**:
1. **Rule-based** (NaiveRewardManager/BatchRewardManager): `compute_score`在CPU端执行, 不需GPU → ★ GRPO默认选择
2. **Reward Model** (colocated RM): 需额外GPU做RM inference → RTX 4090不可行

### 5.6 RM Score路径 (reward.py extract_reward)

```python
def extract_reward(batch: DataProto):
    reward_tensor = batch.batch["rm_scores"]  # 来自reward model或reward_manager
    reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
    reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
    return reward_tensor, reward_extra_infos_dict
```

如果RM预先计算了rm_scores → `_extract_reward_from_rm_scores()`直接返回, 不再调用compute_score.

---

## 6. bypass_mode详解 — 省略old_log_prob重算

### 6.1 核心概念: 3-Policy vs 2-Policy

```
标准PPO (3-Policy):
    π_rollout: 生成trajectory的policy (vLLM/SGLang inference)
    π_old:     训练anchor policy (FSDP FP32, recompute一次)
    π_θ:       当前training policy (每次mini-batch forward)

    ratio = π_θ / π_old → PPO clipping基于π_old
    需要额外forward pass计算old_log_probs → 费计算!

bypass_mode (2-Policy):
    π_rollout: 生成trajectory的policy
    π_θ:       当前training policy

    old_log_probs = rollout_log_probs  ← 直接复用! 零额外forward!
    ratio = π_θ / π_rollout → PPO clipping基于π_rollout
```

### 6.2 apply_bypass_mode (rollout_corr_helper.py L1102-1130)

```python
def apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
    if "rollout_log_probs" not in batch.batch:
        raise ValueError("bypass_mode=True requires rollout_log_probs")

    # ★ 核心操作: old_log_probs = rollout_log_probs (零成本替换!)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    # 设置policy_loss_config
    policy_loss_config["rollout_correction"] = rollout_corr_config
    policy_loss_config["loss_mode"] = "bypass_mode"  # ← 切换到bypass_mode loss!
```

**三行代码**: 把rollout_log_probs赋给old_log_probs + 切换loss_mode → **省掉一整个forward pass**!

### 6.3 compute_policy_loss_bypass_mode (core_algos.py L2351-2420)

```python
@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(old_log_prob, log_prob, advantages, response_mask, ...):
    # old_log_prob = rollout_log_prob (bypass mode语义)

    rollout_log_prob = old_log_prob  # 明确语义

    # 计算IS weights和rejection mask
    rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
        compute_rollout_correction_and_rejection_mask(
            old_log_prob=log_prob,      # π_θ (当前policy)
            rollout_log_prob=rollout_log_prob,  # π_rollout
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
        )
    )

    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None
    effective_mask = modified_response_mask  # rejection sampling后的mask

    # 根据loss_type分发:
    if loss_type == "reinforce":
        # REINFORCE: 显式IS weights
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            rollout_is_weights=computed_is_weights,  # IS weights乘到loss上
        )
    elif loss_type == "ppo_clip":  # ★ 默认
        # PPO-clip: ratio π_θ/π_rollout 自带IS → 不需要额外IS weights!
        pg_loss, pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=rollout_log_prob,  # π_rollout
            log_prob=log_prob,              # π_θ
            advantages=advantages,
            response_mask=effective_mask,
            rollout_is_weights=None,  # ← 显式None! 防止double-counting
        )
```

**PPO-clip + bypass_mode的关键**: ratio = π_θ/π_rollout 已经包含了IS信息, clipping自然约束了IS ratio → **不需要额外IS weights** → 防止double-counting!

### 6.4 compute_policy_loss_reinforce (core_algos.py L2271-2350)

```python
def compute_policy_loss_reinforce(rollout_log_prob, log_prob, advantages, response_mask, ...,
                                   rollout_is_weights=None):
    if rollout_is_weights is not None:
        # IS-corrected: L = -E[w * log π(a|s) * A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π(a|s) * A]
        pg_losses = -advantages * log_prob

    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode, **config.global_batch_info)
    # KL: π_θ vs π_rollout
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, {"actor/ppo_kl": kl_divergence.detach().item()}
```

**关键区别**: REINFORCE用`log_prob`直接(不是ratio), IS weights作为乘法因子; PPO-clip用ratio(π_θ/π_rollout), clipping自动约束.

### 6.5 bypass_mode节省的计算量

```
标准模式 (每training step):
    1. actor forward (compute old_log_prob) → ~0.3-0.5s (7B model, single GPU)
    2. actor forward (compute current log_prob in ppo_loss) → ~0.3-0.5s
    3. actor backward → ~0.5-1.0s
    总计: 2个forward + 1个backward

bypass_mode:
    1. 省掉old_log_prob forward → 直接复用rollout_log_probs
    2. actor forward (compute current log_prob in ppo_loss) → ~0.3-0.5s
    3. actor backward → ~0.5-1.0s
    总计: 1个forward + 1个backward → ★ 省~30-40%训练时间!
```

### 6.6 Rollout Correction子系统 (rollout_corr_helper.py)

`compute_rollout_correction_and_rejection_mask` (L779-870) 支持:

**IS (Importance Sampling)**:
- `token`: per-token IS weights = π_θ(t)/π_rollout(t)
- `sequence`: per-sequence IS weights = Σ log ratio → 单一权重
- Threshold: TIS (Truncated IS, 上界如2.0) 或 IcePop (lower+upper bounds)

**RS (Rejection Sampling)**:
- `token_k*`: per-token ratio阈值 → mask掉极端token
- `seq_sum_k*`: sequence-level ratio sum阈值
- `seq_mean_k*`: sequence-level ratio mean阈值
- `seq_max_k*`: sequence-level ratio max阈值

**Off-policy Metrics**:
- KL divergence, Perplexity, χ² divergence, log-PPL difference

### 6.7 V1 trainer中的bypass_mode实现 (trainer_base.py L1134-1170)

```python
def _compute_old_log_prob(self, batch, metrics):
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

    if bypass_recomputing_logprobs:
        # 从TransferQueue取出rollout_log_probs → 重命名为old_log_probs
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                select_fields=["rollout_log_probs"])
        data["old_log_probs"] = data.pop("rollout_log_probs")
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
        return batch  # ← 零forward pass!

    # 否则: 正常recompute
    batch.extra_info.update({"calculate_entropy": True, "compute_loss": False, ...})
    output = self.actor_rollout_wg.compute_log_prob(batch)  # actor forward
    ...
```

---

## 7. Policy Loss Registry — 8种loss模式

### 7.1 注册机制 (core_algos.py L50-85)

```python
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

@register_policy_loss("vanilla")       → compute_policy_loss_vanilla    # ★ 默认PPO clip
@register_policy_loss("dppo_tv")       → compute_policy_loss_dppo_tv    # DPPO total variation
@register_policy_loss("dppo_kl")       → compute_policy_loss_dppo_kl    # DPPO KL divergence
@register_policy_loss("gspo")          → compute_policy_loss_gspo       # GSPO (gradient scaling)
@register_policy_loss("sapo")          → compute_policy_loss_sapo       # SAPO (self-adaptive)
@register_policy_loss("gpg")           → compute_policy_loss_gpg        # GPG (group policy gradient)
@register_policy_loss("clip_cov")      → compute_policy_loss_clip_cov   # Clip+COVariance
@register_policy_loss("kl_cov")        → compute_policy_loss_kl_cov     # KL+COVariance
@register_policy_loss("geo_mean")      → compute_policy_loss_geo_mean   # Geometric mean ratio
@register_policy_loss("cispo")         → compute_policy_loss_cispo      # CISPO (clipped IS)
@register_policy_loss("bypass_mode")   → compute_policy_loss_bypass_mode # ★ bypass mode!
```

通过`config.policy_loss.loss_mode`选择 → `get_policy_loss_fn(loss_mode)`查找.

### 7.2 KL Penalty实现 (core_algos.py L2126-2270)

5种KL计算方法:

| 方法 | 公式 | 特点 |
|------|------|------|
| k1 | (π_ref - π_θ) | 无偏估计, 但方差大 |
| k2 | 0.5*(log_diff)² | 有偏, 方差小; ★ forward+backward同值 |
| k3 | (π_ref/π_θ - 1) | 无偏估计 |
| k3+ | k3 forward + k2 backward | ★ straight-through: forward无偏+backward有偏 |
| mse | 0.5*(log_diff)² | 同k2 |

---

## 8. RTX 4090 Impact Analysis

### 8.1 GRPO Training配置

| 项目 | RTX 4090最优配置 | 原因 |
|------|-----------------|------|
| Algorithm | GRPO | 不需要critic → 省GPU内存 |
| Advantage | GRPO_VECTORIZED | GPU端向量化 → 10-100x快 |
| norm_adv_by_std | False (Dr.GRPO) | 小group(n=2-4)防梯度消失 |
| ref_in_actor | True (LoRA) | 单GPU做actor+ref → 零额外GPU |
| Reward | Rule-based | 不需RM GPU → CPU端compute_score |
| Rollout | vLLM INT4 | 省内存 → 更多KV空间 |
| LoRA rank | 32 | 平衡: 足够表达力+小内存开销 |
| bypass_mode | True | 省old_log_prob forward → ~30-40%加速 |
| loss_type | ppo_clip | IS handled by ratio → 不需额外weights |
| DP size | 1 | PCIe scaling灾难 → 单GPU最优 |

### 8.2 内存预算 (7B LoRA GRPO)

```
Base model (INT4):             ~3.5GB
LoRA adapters (rank=32):       ~64MB
Optimizer state (CPU_Adam):    CPU端 → 0 GPU占用
KV cache (INT8):               ~2GB (推理时)
Training buffers:              ~1-2GB
Total peak:                    ~7-8GB → 24GB内16-17GB headroom ✓
```

### 8.3 bypass_mode的RTX 4090收益

```
标准模式每step时间 (7B INT4, 单GPU):
    Rollout: ~2s (vLLM INT4 decode)
    old_log_prob forward: ~0.5s
    ref_log_prob forward: ~0.5s (LoRA disable → 基本同old)
    ppo forward+backward: ~1.5s
    Total: ~4.5s

bypass_mode:
    Rollout: ~2s (但rollout_log_probs已在batch中)
    省old_log_prob: 0s (复用rollout_log_probs)
    省ref_log_prob: 0s (LoRA disable in same forward → 但GRPO不需要!)
    ppo forward+backward: ~1.5s
    Total: ~3.5s → 省~22%!

★ GRPO + bypass_mode: 不需要ref_log_prob (GRPO无KL in reward by default)
    → 省old forward + 省ref forward → ~40%加速!
```

### 8.4 不可行配置

| 配置 | 不可行原因 |
|------|-----------|
| Full PPO (GAE + Critic) | 需额外critic GPU (~17GB); RTX 4090不够 |
| Reward Model (RM) | RM inference需额外GPU; 单GPU无空间 |
| FSDP2/Megatron TP>1 | 需多GPU; PCIe scaling灾难 |
| ZeRO-3 | 3Ψ通信; PCIe带宽瓶颈 |
| NVLink/RDMA依赖功能 | RTX 4090无NVLink, 无SM90 |

---

## 9. 源码文件索引

| 文件 | 关键内容 | 行号 |
|------|----------|------|
| `verl/trainer/ppo/core_algos.py` | 15种advantage estimator + 8种policy loss + kl_penalty + agg_loss | L50-2487 |
| `verl/trainer/ppo/ray_trainer.py` | Legacy PPO trainer: compute_advantage, _update_actor, bypass_mode entry | L1-1768 |
| `verl/trainer/ppo/v1/trainer_base.py` | V1 PPO trainer: step(), _compute_advantage, TransferQueue | L1-1400+ |
| `verl/trainer/ppo/v1/trainer_sync.py` | V1 sync trainer (colocated) | L1-30 |
| `verl/trainer/ppo/rollout_corr_helper.py` | apply_bypass_mode, compute_rollout_correction, IS/RS/metrics | L779-1130 |
| `verl/trainer/ppo/reward.py` | RewardManager加载, extract_reward, custom_reward_fn | L1-120 |
| `verl/trainer/ppo/utils.py` | Role enum, need_critic(), need_reference_policy() | L1-150 |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker, TrainingWorker, LoRA disable_adapter, weight sync | L1-758 |
| `verl/workers/utils/losses.py` | ppo_loss wrapper, value_loss | L1-150 |
| `verl/workers/reward_manager/abstract.py` | AbstractRewardManager ABC | L1-60 |
| `verl/workers/reward_manager/naive.py` | NaiveRewardManager (串行CPU reward) | L1-140 |
| `verl/workers/reward_manager/batch.py` | BatchRewardManager | L1-120 |
| `verl/trainer/ppo/v1/utils.py` | compute_advantage_for_multi_trajectories | L1-80 |

---

## 10. 关键发现总结

### ★★★ 核心发现

1. **GRPO bypass_mode = 最优RTX 4090路径**: 省old_log_prob forward + GRPO不需要ref → 省2个forward → ~40%加速
2. **Outcome reward只放在EOS位置**: `reward_tensor[i, valid_length-1] = score` → GRPO sum后等于trajectory total reward
3. **LoRA ref_in_actor = 单GPU可行**: `disable_adapter()` → 同一模型计算π_ref → 不需额外GPU
4. **GRPO_VECTORIZED = GPU端向量化**: `group_mean_std`一次计算所有group → 10-100x比Python loop快
5. **PPO-clip + bypass_mode不加IS weights**: ratio π_θ/π_rollout自带IS → clipping约束 → double-counting风险
6. **Dr.GRPO (norm_adv_by_std=False)**: 小group时σ→0 → 除std导致梯度消失 → 不除std是正确做法
7. **V1 trainer用TransferQueue**: 中间数据存KV store → 避免大tensor拷贝 → 更高效
8. **Dual-clip PPO**: A<0时下界clip_ratio_c=3.0 → 防止ratio过小导致意外大gradient update
9. **agg_loss保证DP不变性**: `dp_size`因子 + `global_batch_info` → 多GPUloss值一致
10. **NaiveRewardManager有timeout**: SIGALRM防单个compute_score卡死整个训练loop
