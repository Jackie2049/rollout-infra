# verl PPO vs GRPO 训练Loop对比

> 核心差异: 1个config开关(adv_estimator) → critic开关 → 完全不同的数据流
> 源码: verl/trainer/ppo/ray_trainer.py + core_algos.py + losses.py + utils.py

## 1. Config开关: `need_critic()` 决定一切

```python
# utils.py:96
def need_critic(config):
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True   # PPO → critic ON
    else:
        return False  # GRPO → critic OFF
```

## 2. PPO (GAE) 数据流

```
prompt_batch → rollout(generate_sequences)
→ gen_batch_output (responses, log_probs)
→ reward computation → token_level_scores
→ [KL-in-reward?] → token_level_rewards
→ old_log_prob (actor recomputed)
→ ref_log_prob (ref_policy, optional)
→ ★ CRITIC FORWARD: critic_wg.infer_batch → values tensor [bsz, resp_len]
→ ★ GAE advantage:
    delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)  (TD residual)
    A_t^GAE = delta_t + gamma*lam*A_{t+1}       (recursive lambda-weighted)
    → 每个token位置advantage不同!
    returns = advantages + values (value residual)
    advantages = masked_whiten(advantages)  (全局零均值单位方差)
→ ★ CRITIC UPDATE: critic_wg.train_mini_batch
    value_loss = 0.5*max((vpred-returns)^2, (vpred_clipped-returns)^2)
→ ACTOR UPDATE: ppo_loss (same code as GRPO, different advantages)
```

## 3. GRPO 数据流

```
prompt_batch → rollout(generate_sequences with rollout_n repeats)
→ gen_batch_output (each prompt × N completions)
→ reward computation → token_level_scores
→ [KL-in-reward?] → token_level_rewards
→ old_log_prob (actor recomputed)
→ ref_log_prob (ref_policy, optional)
→ ✗ NO CRITIC FORWARD — SKIPPED
→ ★ GRPO advantage:
    scores = token_level_rewards.sum(dim=-1)  → 1 scalar per response
    Group by uid (same prompt = same group):
      singleton (n=1): mean=0, std=1 → advantage=0 → no learning signal!
      n>1: mean=mean(scores), std=std(scores)
      A_i = (score_i - mean) / (std + eps)  [norm_adv=True, 原始GRPO]
      A_i = score_i - mean                   [norm_adv=False, Dr.GRPO]
    advantages = A_i.unsqueeze(-1) * response_mask → scalar broadcast到所有tokens!
    returns = advantages (same tensor, no value baseline)
→ ✗ NO CRITIC UPDATE — SKIPPED
→ ACTOR UPDATE: ppo_loss (IDENTICAL code, different advantages)
```

## 4. Side-by-Side对比表

| 维度 | PPO (GAE) | GRPO |
|------|-----------|------|
| **Config** | `adv_estimator="gae"` | `adv_estimator="grpo"` |
| **Critic** | REQUIRED (独立模型+优化器) | NOT USED |
| **Advantage输入** | `values` tensor from critic | `uid` index for grouping |
| **Advantage计算** | GAE: 递归TD-lambda跨token | Group mean/std跨responses |
| **Advantage粒度** | Token-level (每token不同A) | Response-level (同一A广播到所有tokens) |
| **Returns定义** | `returns = A + V` (value residual) | `returns = A` (同advantage) |
| **Advantage归一化** | masked_whiten(全局零均值单位方差) | per-group mean/std归一化 |
| **Singleton行为** | 不适用(单轨迹GAE正常) | mean=0,std=1 → advantage=0 → 无学习信号! |
| **rollout_n效果** | 更多轨迹,每条独立GAE | **算法必需**: group大小=N, N≥4 |
| **额外forward** | Critic infer + Critic train | None |
| **内存: 模型权重** | Actor+Critic = 2×模型 | Actor only = 1×模型 |
| **内存: 优化器** | 2×Adam(actor+critic) | 1×Adam(actor only) |
| **内存: 激活** | Critic fwd+bwd激活 | None extra |
| **总内存估算(7B)** | ~112-116GB | ~56GB → **省50%!** |
| **训练速度** | ~1.5-2×慢(critic fwd+bwd) | ~1× (只actor bwd) |
| **Loss函数** | ppo_loss + value_loss | ppo_loss only (同一代码!) |
| **gamma/lam** | 两者都用 (GAE时间分解) | 不相关 (outcome-only) |
| **KL处理** | kl-in-reward 或 kl-in-loss | kl-in-loss推荐(coef=0.001, type=k3) |

## 5. GAE详解 (core_algos.py:216-263)

```python
# 从 t = gen_len-1 → 0 逆向迭代:
for t in reversed(range(gen_len)):
    delta_t = reward[t] + gamma * nextvalues - values[t]  # TD residual
    lastgaelam = delta_t + gamma * lam * lastgaelam       # 递归lambda加权
    advantages_t.append(lastgaelam)
    nextvalues = values[t]  # 为下一个t准备

advantages = torch.stack(advantages[::-1])  # 恢复正向顺序
returns = advantages + values               # value residual form
advantages = masked_whiten(advantages, mask) # 全局归一化
```

**关键特性**: 每个token位置advantage不同 → early tokens汇总多个future deltas → late tokens≈immediate delta

## 6. GRPO详解 (core_algos.py:268-331)

```python
scores = token_level_rewards.sum(dim=-1)  # outcome: 1 scalar per response

# 按uid分组:
for uid in unique_uids:
    group = scores[uid_matches]
    if len(group) == 1:
        mean, std = 0.0, 1.0  # singleton → advantage=0!
    else:
        mean = torch.mean(group)
        std = torch.std(group)
    advantages[uid_matches] = (scores[uid_matches] - mean) / (std + eps)

advantages = advantages.unsqueeze(-1) * response_mask  # scalar → broadcast
returns = advantages  # no value baseline
```

**关键特性**: 同一response所有tokens共享同一advantage → 纯相对性能(组内好→正,差→负)

## 7. Singleton Group问题

```python
# n=1时: mean=0, std=1 → advantage = (score-0)/(1+ε) = score
# → 无baseline → 无归一化 → 可能不稳定!
→ rollout_n MUST > 1 for GRPO!
→ rollout_n ≥ 4 是默认推荐 → group normalization质量依赖N
→ rollout_n=2 → normalization存在但噪声大(std from 2 samples)
```

## 8. Loss函数: 同一代码,不同输入

PPO和GRPO都用**同一** `ppo_loss`:

```python
# losses.py:57-144
ratio = exp(log_prob - old_log_prob)
pg_loss = -advantages * clip(ratio, 1-eps, 1+eps)  # dual-clip variant
```

差异仅在于`advantages`内容:
- PPO: token-level GAE values (每位置不同)
- GRPO: response-level group-normalized values (跨位置恒定)

PPO额外有`value_loss` (只有critic存在时):
```python
# losses.py:147-186
vpred_clipped = clamp(vpred, values - cliprange, values + cliprange)
vf_loss = 0.5 * max((vpred-returns)^2, (vpred_clipped-returns)^2)
```

## 9. 代码路径切换详解

当`adv_estimator="grpo"`时, 以下代码被**跳过**:

| 跳过的步骤 | ray_trainer.py位置 | PPO执行 | GRPO跳过 |
|-----------|-------------------|---------|----------|
| Critic worker创建 | line 801-826 | ✅ | ❌ |
| `_compute_values()` | line 1583-1586 | ✅ critic infer | ❌ |
| `_update_critic()` | line 1636-1640 | ✅ critic train | ❌ |
| `value_loss`计算 | losses.py 147-186 | ✅ | ❌ |

新启用:
| 新启用的配置 | 说明 |
|------------|------|
| `norm_adv_by_std_in_grpo` | True=原始GRPO, False=Dr.GRPO |
| `uid` grouping | uuid生成(line 1439) → 核心分组依据 |
| `rollout_n` repeat | line 1447-1448 → 算法必需(N>1) |

## 10. RTX 4090实战建议

```
RTX 4090 24GB (7B模型):
  PPO: Actor(14GB)+Critic(14GB)=28GB → ✗ 不fit单GPU!
  GRPO: Actor only(14GB)+LoRA params(~0.3GB)+activations(~3GB)=17GB → ✓ fit!

  PPO内存拆分:
    模型: 28GB (actor+critic BF16)
    优化器: 84GB (2×Adam FP32)
    → 即使ZeRO-2+CPU Adam: 28GB on GPU → ✗ 不fit!

  GRPO内存拆分:
    模型: 14GB (actor only BF16)
    LoRA: 0.3GB (r=16)
    Activations: ~3GB (单microbatch)
    → 14+0.3+3=17.3GB → ✓ fit 24GB!

  → RTX 4090最优: GRPO+LoRA(r=16)+CPU Adam+KL-in-loss
  → rollout_n=4 (每prompt 4 completions → group normalization)
  → 单GPU无需分布式 → rLLM TinkerBackend最快路径
```

## 11. 决策树: 何时用PPO vs GRPO

```
条件判断:
  - GPU内存 < 2×模型大小 → 只能用GRPO (critic太大)
  - rollout_n=1 → 不能用GRPO (singleton无归一化) → 必须PPO
  - 需要token-level advantage → PPO (GAE每token不同)
  - 需要response-level advantage → GRPO (组内相对)
  - 多GPU+大内存 → PPO可选 (critic有空间)
  - 单GPU+小内存 → GRPO必选 (省50%内存)
  - MoE模型 → GRPO更优 (critic难以处理expert routing)
  - 长CoT任务 → GRPO更优 (outcome reward更合适)
```
