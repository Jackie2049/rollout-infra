# verl GRPO End-to-End Data Flow 源码级阅读

> 2026-06-15 | 源码: verl/verl/trainer/ppo/core_algos.py + ray_trainer.py + workers/utils/losses.py + utils.py
> 核心: GRPO=PPO损失+group-relative advantage → need_critic自动False → 3种advantage变体 → rollout_n interleave → ref_in_actor=LoRA → Dr.GRPO=不除std → total_loss=pg+0.001*KL-entropy_coef*H

## 1. GRPO vs PPO: 只改advantage,不改loss

```
PPO: loss = clipped_surrogate(advantage=GAE(λ-return))
GRPO: loss = clipped_surrogate(advantage=group_relative(r_i - μ_g / σ_g))

→ 同一个"vanilla" loss function! (core_algos.py:1278-1371)
→ GRPO不是新loss,是新advantage计算方式!
→ 关键差异: need_critic() → adv_estimator!=GAE → critic自动disabled (utils.py:96-107)
→ GRPO跳过: critic forward + critic backward + value loss + GAE
→ PPO需要: actor + critic + ref → 3个模型 → 3倍GPU内存!
→ GRPO需要: actor + ref → 2个模型 → 但ref_in_actor(LoRA)=共享 → 1个GPU!

→ Total GRPO loss = pg_loss + kl_loss_coef * KL(pi || pi_ref) - entropy_coef * H
→ kl_loss_coef=0.001 (default) → KL在loss端加 → 不在reward端减!
→ GRPO example: use_kl_in_reward=False + use_kl_loss=True → KL只在loss端
```

## 2. GRPO Advantage: 3种变体

### GRPO (loop-based) — core_algos.py:268-331

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)  # outcome reward: 只取最后一个token的reward!

    # 按prompt index分组 → 同一prompt的n个response形成1个group
    id2score = defaultdict(list)  # {prompt_id: [r_1, r_2, ..., r_n]}
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # 计算group mean和std
    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)

    # ★ GRPO advantage核心公式:
    # norm_adv_by_std_in_grpo=True (原始GRPO):
    #   a_i = (r_i - μ_g) / (σ_g + ε)
    # norm_adv_by_std_in_grpo=False (Dr.GRPO, arxiv 2503.20783):
    #   a_i = r_i - μ_g  → 不除std → 防止梯度消失!

    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]

    scores = scores.unsqueeze(-1) * response_mask  # broadcast到每个token
    return scores, scores  # advantage = returns (outcome-only)
```

**关键细节**:
- 单样本group(len=1): μ=0, σ=1 → advantage=score本身 → 无group normalization
- outcome reward = token_level_rewards.sum(dim=-1) → 只关心最终结果,不是per-token
- advantage broadcast: 每个token获得相同advantage → 乘response_mask

### GRPO_VECTORIZED — core_algos.py:335-358

```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
    # 同样公式,但vectorized → 不需要loop → 更快!
    scores = token_level_rewards.sum(dim=-1)
    g = as_torch_index(index, device=scores.device)
    mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)

    if norm_adv_by_std_in_grpo:
        scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
    else:
        scalars = scores - mean_g[g]

    advantages = scalars.unsqueeze(-1) * response_mask
    return advantages, advantages
```

**性能差异**: GRPO_VECTORIZED用torch操作代替Python loop → 大batch快10-100x!

### GRPO_PASSK — core_algos.py:472-530

```python
@register_adv_est(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(...):
    # ★ 只有最佳response获得非零advantage!
    # a_best = r_max - r_second_max / (σ + ε)
    # 其他response → advantage = 0 → 不参与训练!

    for idx in id2scores:
        rewards = torch.stack(id2scores[idx])  # (k,)
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0].item()]
        advantage = r_max - r_second_max
        if norm_adv_by_std_in_grpo:
            std = torch.std(rewards)
            advantage = advantage / (std + epsilon)
        advantages[i_max] = advantage  # 只有best获得!
```

**GRPO_PASSK特点**:
- 只有best response参与训练 → 极大减少compute!
- advantage = r_max - r_second_max → 鼓励保持领先
- 至少需要2个样本/group → 不适用于n=1

## 3. GDPO: Decoupled Normalization (新增!)

```python
@register_adv_est(AdvantageEstimator.GDPO)  # core_algos.py:361-468
def compute_gdpo_outcome_advantage(...):
    # ★ 不是先sum再normalize → 先per-dimension normalize再sum!
    # Step 1: 每个reward dimension独立normalize → 防止dominant dimension淹没弱信号
    # A_k = (r_k - μ_group(r_k)) / (σ_group(r_k) + ε)
    # Step 2: 加权聚合 → A_sum = Σ_k w_k · A_k
    # Step 3: batch-level whiten → A_final = whiten(A_sum, response_mask)
```

**GDPO vs GRPO**:
- GRPO: sum all dimensions → normalize → dominant dimension控制advantage
- GDPO: normalize each dimension → sum → balanced across dimensions
- 适用: 多维度reward(格式+准确性+推理) → 防止格式reward淹没准确性

## 4. PPO/GRPO Loss: Clipped Surrogate (不变!)

```python
@register_policy_loss("vanilla")  # core_algos.py:1278-1371
def compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask, ...):
    clip_ratio = config.clip_ratio  # ε for PPO clipping
    clip_ratio_low = config.clip_ratio_low  # asymmetric lower bound
    clip_ratio_high = config.clip_ratio_high  # asymmetric upper bound
    clip_ratio_c = config.get("clip_ratio_c", 3.0)  # dual-clip lower bound

    # ratio = π_new(a|s) / π_old(a|s) = exp(log_prob - old_log_prob)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)

    # ★ Standard PPO clipped loss:
    # L = max(-A * ratio, -A * clip(ratio, 1-ε_low, 1+ε_high))

    pg_losses1 = -advantages * ratio  # unclipped
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # ★ Dual-clip PPO (for negative advantages):
    # When A < 0: L = min(-A * clip_ratio_c, max(-A*ratio, -A*clip(ratio)))
    # → 防止negative advantage时ratio无限大 → 稳定训练!

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**loss聚合**: agg_loss(loss, response_mask, mode="token-mean")
- token-mean: 所有token平均 → 标准PPO
- token-sum: token sum → 大batch更稳定

## 5. End-to-End Data Flow: 从prompt到update

```
ray_trainer.py 全流程:

Step 1: gen_batch = prompts.repeat(rollout_n, interleave=True)
  → rollout_n = config.actor_rollout_ref.rollout.n
  → interleave=True → [p1_r1, p1_r2, ..., p1_rn, p2_r1, p2_r2, ..., p2_rn, ...]
  → 同一prompt的n个response连续排列 → group index = prompt_id

Step 2: rollout → actor_rollout_wg.generate_sequences(gen_batch)
  → vLLM/SGLang inference → 生成response
  → output: {responses, log_probs, input_ids, ...}

Step 3: reward → reward_manager.compute_reward(rollout_output)
  → rm_scores = outcome reward (每个response一个score)
  → reward写入batch.batch["rm_scores"]

Step 4: extract_reward → reward.py:160-167
  → reward_tensor = batch.batch["rm_scores"]
  → 可能包含per-token rewards或outcome-only

Step 5: compute_advantage → core_algos.py
  → GRPO: group_relative normalization
  → index = prompt grouping (uid) → 同一prompt的response归入同一group
  → a_i = (r_i - μ_g) / (σ_g + ε) → broadcast到每个token

Step 6: ref_log_prob → ref_policy_wg.compute_ref_log_prob()
  → ref_in_actor=True(LoRA): 同一个worker → no_lora_adapter=True → 用base model
  → ref_in_actor=False: 独立ref worker → 需要额外GPU!

Step 7: actor update → ppo_loss(losses.py:57-)
  → log_prob = current policy → old_log_prob = rollout policy
  → ratio = exp(log_prob - old_log_prob)
  → loss = clipped_surrogate(ratio, advantage)
  → + KL penalty: kld = kl_penalty(log_prob, ref_log_prob) → kl_loss * 0.001
  → backward → optimizer.step()

★ GRPO跳过: critic forward + critic backward + value loss + GAE
→ 只有actor update → 简单2/3 compute+memory!

★ need_critic() (utils.py:96-107):
  → adv_estimator != GAE → critic disabled → no CriticWorker spawned
  → unless critic.enable=True explicit override → 通常不需要!

★ Data shape through pipeline:

| Stage | Shape | Key Fields |
|-------|-------|------------|
| Dataloader | [train_batch_size] | raw_prompt, reward_model, uid |
| After repeat | [train_batch_size * rollout_n] | Same, interleaved by uid |
| After rollout | [train_batch_size * rollout_n, prompt_len + response_len] | prompts, responses, attn_mask |
| After reward | Same | + rm_scores, token_level_scores |
| After old_log_prob | Same | + old_log_probs, entropys |
| After ref_log_prob | Same | + ref_log_prob |
| After advantage | Same | + advantages, returns |
| Actor input | [ppo_mini_batch * rollout_n] per micro-batch | All fields, no-padding |

★ ppo_mini_batch_size multiplied by rollout_n (ray_trainer.py:1310-1311)
  → effective mini-batch = ppo_mini_batch * rollout_n
```

## 6. LoRA Integration in GRPO

```
ray_trainer.py:358-360:
  self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path")

→ LoRA: ref和actor共享同一GPU!
→ ref_in_actor=True → self.ref_policy_wg = self.actor_rollout_wg (同一WorkerGroup)
→ compute_ref_log_prob时: no_lora_adapter=True → disable LoRA → 用base model weights
→ compute_actor_log_prob时: LoRA enabled → 用LoRA-adapted weights
→ 无需额外GPU → RTX 4090可行!

→ 无LoRA: ref和actor分开 → 需2个GPU → RTX 4090不可行!
→ COLOCATED: actor+ref在同一PG → 可2GPU但内存紧张
→ STANDALONE: actor+ref分离 → 需要≥3个GPU!
```

## 7. AdvantageEstimator完整注册表

```
core_algos.py AdvantageEstimator enum:

GRPO            → loop-based group-relative
GRPO_VECTORIZED → vectorized group-relative (推荐!)
GRPO_PASSK      → only best response (arxiv 2503.19595)
GDPO            → decoupled per-dimension normalization (arxiv 2601.05242)
REINFORCE_PLUS_PLUS         → REINFORCE++ baseline
REINFORCE_PLUS_PLUS_BASELINE → outcome-only REINFORCE++
REMAX                        → REMAX with greedy baseline
RLOO                         → Leave-One-Out baseline
PWIL                         → Wasserstein distance
IS                           → Importance Sampling correction
```

## 8. RTX 4090 GRPO可行配置

```
RTX 4090 (24GB) GRPO配置:

1. HYBRID mode → actor+ref+rollout同一进程 → 1 GPU
2. LoRA rank=16-64 → ref_in_actor=True → 无需额外ref GPU
3. naive weight sync → Python generator zero-copy → 不需要NCCL/IPC
4. GRPO_VECTORIZED → 最快advantage计算
5. 7B BF16 LoRA: ~14GB(base) + ~1GB(LoRA) = 15GB → 24GB可行!
6. rollout_n=4-8 → group足够大 → std有意义 → advantage有效

→ RTX 4090唯一可行RL方案: GRPO + LoRA + HYBRID + naive
→ PPO: 需要3个模型 → 3×14GB=42GB → RTX 4090不可行!
→ rLLM TinkerBackend: 也可行 → 但verl生态更成熟
```

## 9. 关键设计洞察

```
1. GRPO不是新loss → 是新advantage → loss完全不变!
   → PPO和GRPO用同一个clipped_surrogate → 只是advantage来源不同
   → 这意味着: PPO→GRPO迁移 → 只需改adv_estimator → 其他代码不变!

2. Dr.GRPO(不除std) → 防止梯度消失 → 当σ_g接近0时
   → 小group(n=2-4): σ可能很小 → 除σ → advantage极端 → 梯度不稳定
   → Dr.GRPO: a_i = r_i - μ_g → 更稳定 → 但可能不够normalized
   → 推荐: n≥8时用GRPO(除std), n≤4时用Dr.GRPO(不除std)

3. GRPO_PASSK → 极致compute节省 → 只有best response训练
   → 但需要n≥2 → 至少2个response/group
   → 适合: 大batch+小GPU → 只训最有价值的样本

4. GDPO → 多维度reward的关键改进 → 防止dominant dimension
   → 数学推理reward: 正确率>格式 → GRPO可能只优化格式
   → GDPO: 先per-dimension normalize → 平衡优化所有维度
   → 生产级reward需要多维度 → GDPO比GRPO更好!

5. ref_in_actor = LoRA的杀手级优化 → 省1个GPU!
   → 无LoRA: ref需要独立GPU → 2GPU起步 → RTX 4090完全不行
   → LoRA: ref和actor共享 → disable_adapter()切换 → 只1GPU
   → 这不是hack → 是verl正式特性 → ray_trainer.py明确支持!

6. interleave=True → group内response连续 → advantage计算正确
   → gen_batch.repeat(n, interleave=True)
   → 排列: [p1_r1, p1_r2, ..., p1_rn, p2_r1, ...]
   → index = prompt_id → 自然分组 → 不需要额外sort

7. 10种policy loss → 不只clip!
   → vanilla(clip) / dppo_tv / dppo_kl / gspo / sapo / gpg / clip_cov / kl_cov / geo_mean / cispo / bypass_mode
   → GRPO推荐: vanilla(标准clip) → 最简单最稳定
   → 高级: cispo/geo_mean → 更复杂的clip策略 → 可能更好但需调参
```

---

Sources:
- verl/verl/trainer/ppo/core_algos.py — GRPO advantage (3 variants) + GDPO + 10 policy losses
- verl/verl/trainer/ppo/ray_trainer.py — rollout_n interleave + ref_in_actor + advantage routing
- verl/verl/workers/utils/losses.py — ppo_loss + sft_loss implementation
- verl/verl/trainer/ppo/reward.py — extract_reward logic
- notebook/projects/verl-worker-lifecycle-ray-weight-sync-reading.md — WorkerGroup/RolloutMode
- notebook/projects/verl-multi-turn-agent-loop-reading.md — ToolAgentLoop
