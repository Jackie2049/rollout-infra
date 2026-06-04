# verl RL Infra 架构源码阅读

> verl: 基于 Ray 的分布式 RL 训练框架，支持 PPO/GRPO 等算法

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    verl 训练架构 (main_ppo_sync.py)                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  RayPPOTrainer / SyncPPOTrainer              │   │
│  │  (Driver Process — 单节点 CPU/GPU)                           │   │
│  │                                                             │   │
│  │  训练主循环:                                                  │   │
│  │  1. DataLoader → sample prompts                             │   │
│  │  2. RolloutWorker.generate_sequences() → responses          │   │
│  │  3. RewardManager → compute rewards                         │   │
│  │  4. compute_advantage() → GAE / GRPO                        │   │
│  │  5. ActorWorker.update_policy() → PPO loss                  │   │
│  │  6. CriticWorker.update_critic() → value loss (PPO only)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │ Ray RPC                              │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ ActorRollout  │   │   Critic     │   │   Reward     │            │
│  │ RefWorker     │   │   Worker     │   │   Worker     │            │
│  │              │   │  (PPO only)  │   │              │            │
│  │ - Actor (训) │   │ - Critic (训)│   │ - RM / Rules │            │
│  │ - Rollout (推)│   │              │   │              │            │
│  │ - Ref (固定) │   │              │   │              │            │
│  │   [vLLM]    │   │   [FSDP]    │   │              │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│                                                                     │
│  关键: ActorRolloutRefWorker 是混合 Worker                           │
│  Colocate 模式: Actor+Rollout+Ref 共享同一组 GPU                    │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 核心文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `verl/trainer/main_ppo_sync.py` | ~1866 | 推荐 Trainer, TransferQueue 零拷贝 |
| `verl/trainer/main_ppo.py` | ~1200 | 原始 Trainer (deprecated → v0.8.0) |
| `verl/trainer/ppo/ray_trainer.py` | ~1770 | RayPPOTrainer 基类 |
| `verl/trainer/ppo/core_algos.py` | ~800 | GAE/GRPO/REINFORCE++ 算法 |
| `verl/workers/engine_workers.py` | ~758 | TrainingWorker + ActorRolloutRefWorker |
| `verl/workers/rollout/base.py` | ~400 | BaseRollout + get_rollout_class |
| `verl/workers/reward_manager/` | ~300 | Reward 计算管理器 |
| `verl/protocol.py` | ~500 | DataProto 数据协议 |

## 3. 训练主循环

### 3.1 SyncPPOTrainer (推荐)

```python
# main_ppo_sync.py 的训练循环
class SyncPPOTrainer:
    def fit(self):
        for epoch in range(total_epochs):
            for batch_dict in train_dataloader:
                # === Phase 1: Rollout ===
                # 生成 responses (vLLM/SGLang inference)
                gen_batch = self.actor_rollout_wg.generate_sequences(batch)

                # === Phase 2: Reward ===
                # 计算 reward (规则/RM)
                batch = self._compute_reward_colocate(batch)
                # 或: batch = self.rm_wg.compute_rm_score(batch)

                # === Phase 3: Reference Log Prob ===
                # 计算 ref policy 的 log prob (for KL)
                if self.use_reference_policy:
                    batch = self.ref_wg.compute_ref_log_prob(batch)

                # === Phase 4: Advantage ===
                # GAE (PPO) 或 Group Relative (GRPO)
                batch = compute_advantage(batch, adv_estimator, ...)

                # === Phase 5: Update Critic ===
                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(batch)

                # === Phase 6: Update Actor ===
                actor_output = self.actor_rollout_wg.update_policy(batch)

                # === Phase 7: Sync Weights ===
                # Actor → Rollout (vLLM) weight sync
                self.checkpoint_manager.update_weights(global_steps)
```

### 3.2 关键区别: main_ppo_sync vs main_ppo

| 特性 | main_ppo (旧) | main_ppo_sync (推荐) |
|------|--------------|---------------------|
| 数据传输 | Ray ObjectRef | TransferQueue 零拷贝 |
| Replay Buffer | 无 | ReplayBuffer 轮询 TQ |
| n sampling | 固定 n | 每个 prompt 不同 n |
| Agent Loop | 不支持 | 支持 Multi-turn |
| 启动方式 | `run_ppo(config)` | Hydra YAML |

## 4. Worker 架构

### 4.1 ActorRolloutRefWorker

```python
# engine_workers.py
class ActorRolloutRefWorker(Worker):
    """混合 Worker: Actor (训练) + Rollout (推理) + Ref (固定)"""

    def __init__(self, config):
        # 训练引擎 (FSDP/Megatron)
        self.engine = EngineRegistry.new(
            backend=engine_config.strategy,  # "fsdp" or "megatron"
            ...
        )

        # Rollout 引擎 (vLLM/SGLang/TRT-LLM)
        self.rollout = get_rollout_class(rollout_config)

        # Reference model (frozen copy)
        if need_reference_policy:
            self.ref_engine = EngineRegistry.new(...)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def generate_sequences(self, data):
        """生成 responses (rollout)"""
        return self.rollout.generate_sequences(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def update_policy(self, data):
        """PPO 策略更新"""
        # 1. Forward pass → compute new log probs
        # 2. PPO clip loss: L = -min(r*A, clip(r,1-ε,1+ε)*A)
        # 3. Backward + optimizer step
        return self.engine.train_batch(data, loss_fn=ppo_loss)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def compute_ref_log_prob(self, data):
        """计算 reference policy log prob"""
        with torch.no_grad():
            return self.ref_engine.forward_batch(data)
```

### 4.2 TrainingWorker

```python
# engine_workers.py
class TrainingWorker(Worker):
    """通用训练 Worker (for Critic)"""

    def __init__(self, config: TrainingWorkerConfig):
        self.engine = EngineRegistry.new(
            model_type=config.model_type,
            backend=engine_config.strategy,
            ...
        )
        # 注册 dispatch 信息 (DP rank 等)
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )
```

### 4.3 @register 装饰器 — Dispatch 模式

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)     # 广播到所有 worker
@register(dispatch_mode=Dispatch.DP_COMPUTE)      # DP 并行计算 + all-reduce
@register(dispatch_mode=Dispatch.MEAN)            # 取所有 worker 均值
```

## 5. Advantage 计算

### 5.1 GAE (PPO)

```python
# core_algos.py
def compute_gae_advantage_return(token_level_rewards, values,
                                  response_mask, gamma, lam):
    """Generalized Advantage Estimation

    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
    """
    # Reverse iteration through sequence
    advantages = torch.zeros_like(token_level_rewards)
    last_gae_lam = 0
    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * values[:, t+1] - values[:, t]
        advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam
    returns = advantages + values
    return advantages, returns
```

### 5.2 GRPO (Group Relative)

```python
# core_algos.py
def compute_grpo_outcome_advantage(token_level_rewards, response_mask,
                                    index, norm_adv_by_std_in_grpo):
    """Group Relative Policy Optimization

    对同一 prompt 的 n 个 response:
    A_i = (R_i - mean(R_group)) / std(R_group)
    """
    # Group by uid (same prompt)
    for uid in unique_uids:
        group_rewards = rewards[group_mask]
        mean_r = group_rewards.mean()
        std_r = group_rewards.std() + 1e-8
        # 标准化 advantage
        advantages[group_mask] = (rewards[group_mask] - mean_r) / std_r
    return advantages, returns
```

### 5.3 AdvantageEstimator 枚举

```python
class AdvantageEstimator:
    GAE = "gae"                    # PPO: Actor+Critic
    GRPO = "grpo"                  # GRPO: 无 Critic
    REINFORCE = "reinforce"        # REINFORCE++
    REINFORCE_BASELINE = "reinforce_baseline"
    GDPO = "gdpo"                  # Generalized DPO
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
```

## 6. KL 惩罚机制

```python
# ray_trainer.py: apply_kl_penalty()
def apply_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
    """KL(π_θ || π_ref) 惩罚"""
    kld = core_algos.kl_penalty(
        data["old_log_probs"],     # 当前 policy
        data["ref_log_prob"],      # reference policy
        kl_penalty=kl_penalty,
    )
    beta = kl_ctrl.value  # 自适应 KL 系数
    token_level_rewards = token_level_scores - beta * kld

    # 自适应更新
    current_kl = masked_mean(kld, mask=response_mask)
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
```

**KL 控制器**:
- `AdaptiveKLController`: 自动调整 β 维持 target KL
- 初始 KL 较大 → β 增大 → 约束更强
- 初始 KL 较小 → β 减小 → 允许更多探索

## 7. Rollout 引擎

### 7.1 架构

```python
# rollout/base.py
class BaseRollout(ABC):
    def generate_sequences(self, prompts) -> DataProto: ...
    def update_weights(self, weights) -> None: ...

# 3 种 Rollout 实现
class VLLMRollout(BaseRollout):      # vLLM (推荐)
class SGLangRollout(BaseRollout):    # SGLang
class TRTLLMRollout(BaseRollout):   # TensorRT-LLM
```

### 7.2 vLLM Server 模式

```python
# rollout/llm_server.py
class LLMServerManager:
    """管理 vLLM server 生命周期"""
    def __init__(self, config):
        # 启动 vLLM server 进程
        self.server_process = subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", tp_size,
            ...
        ])

class LLMServerClient:
    """向 vLLM server 发送推理请求"""
    def generate(self, prompts):
        # HTTP 请求 → vLLM OpenAI API
        response = requests.post("/v1/completions", json={
            "prompt": prompts,
            "max_tokens": max_tokens,
            "n": n,  # GRPO: 每个 prompt n 个 response
        })
```

### 7.3 Weight Sync

```python
# Actor → Rollout 权重同步
def update_weights(self, global_steps):
    """将训练后的 Actor 权重同步到 Rollout 引擎"""
    # 方式 1: 直接内存拷贝 (colocate 模式)
    actor_state_dict = self.actor_engine.state_dict()
    self.rollout.update_weights(actor_state_dict)

    # 方式 2: 通过 checkpoint 文件 (分布式模式)
    self.checkpoint_manager.save(actor_state_dict, global_steps)
    self.rollout.load_weights(checkpoint_path)
```

## 8. Reward 计算

```python
# reward_manager/abstract.py
class AbstractRewardManager(ABC):
    @abstractmethod
    def __call__(self, data: DataProto) -> DataProto: ...

# 实现
class NaiveRewardManager(AbstractRewardManager):
    """逐条计算 reward"""
    def __call__(self, data):
        for i in range(batch_size):
            prompt_str = tokenizer.decode(prompt_ids)
            response_str = tokenizer.decode(response_ids)
            ground_truth = data.non_tensor_batch["ground_truth"]
            score = compute_score(data_source, response_str, ground_truth)
            reward_tensor[i] = score
        return data

class BatchRewardManager(AbstractRewardManager):
    """批量 reward 计算 (RM)"""
    def __call__(self, data):
        # 使用 Reward Model 批量打分
        scores = self.reward_model(data)
        return data
```

## 9. 数据协议: DataProto

```python
# protocol.py
class DataProto:
    """verl 的数据传输协议

    封装了 TensorDict + 非 tensor 元数据
    支持 zero-copy 传输 (通过 TransferQueue)
    """
    batch: TensorDict        # tensor 数据 (input_ids, log_probs, ...)
    non_tensor_batch: dict    # 非 tensor 数据 (uid, ground_truth, ...)
    meta_info: dict           # 元信息 (batch_size, eos_token_id, ...)

    def select_idxs(self, indices): ...     # 选择子集
    def split(self, n): ...                 # 分割为 n 份
    def to(self, device): ...               # 设备转移
```

## 10. TransferQueue 零拷贝

```python
# main_ppo_sync.py 独有
# TransferQueue 实现跨 Worker 的零拷贝数据传输

# Producer (Rollout Worker)
tq.kv_put(partition_id="train", key=uid, data=batch_data)

# Consumer (ReplayBuffer)
data = tq.kv_get(partition_id="train", key=uid)

# ReplayBuffer 轮询
class ReplayBuffer:
    def _poll_from_transfer_queue(self):
        while not stopped:
            data = tq.kv_list()  # 非阻塞查询
            self.add(partition_id, data)
            time.sleep(poll_interval)
```

## 11. 分布式配置

### 11.1 ResourcePoolManager

```python
# Ray 资源管理
resource_pool_manager = ResourcePoolManager(
    resource_pool={
        "pool_0": [GPU_ids...],  # Actor+Rollout+Ref 共享
        "pool_1": [GPU_ids...],  # Critic (PPO)
        "pool_2": [GPU_ids...],  # Reward
    }
)

# Colocate 模式 (推荐): 所有角色共享同一组 GPU
worker_cls = create_colocated_worker_cls(
    RayClassWithInitArgs(ActorRolloutRefWorker, ...),
    RayClassWithInitArgs(CriticWorker, ...),
    ...
)
```

### 11.2 典型配置示例

```yaml
# 7B GRPO, 8 GPU
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B
  actor:
    strategy: fsdp
  rollout:
    name: vllm
    tensor_parallel_size: 8
  ref:
    strategy: fsdp

algorithm:
  adv_estimator: grpo
  grpo_n: 8
  kl_ctrl:
    type: kl
    kl_coef: 0.02

reward_model:
  enable: false  # 使用规则 reward

trainer:
  total_epochs: 1
  rollout_batch_size: 512
```

## 12. 关键洞察

1. **Hybrid Worker**: Actor+Rollout+Ref 共享 GPU, 减少 weight sync 开销
2. **TransferQueue**: 零拷贝跨 Worker 数据传输, 替代 Ray ObjectRef
3. **Dispatch 装饰器**: @register 自动处理 DP 并行 + all-reduce
4. **GRPO 无 Critic**: 少一个模型, 节省 ~25% 显存, verl 默认推荐
5. **vLLM 作为 Rollout**: 利用 continuous batching + prefix caching 加速推理
6. **Weight Sync**: Colocate 模式直接内存拷贝; 分布式模式通过 checkpoint
7. **Adaptive KL**: 自动调整 KL 系数, 防止 reward hacking 或 policy collapse
8. **Agent Loop**: Multi-turn 对话支持, session-based advantage 计算
9. **DataProto**: 统一数据协议, tensor + non-tensor + meta_info
10. **FSDP 作为默认训练引擎**: Sharded data parallel, 支持 gradient checkpointing

## 参考资料

- `verl/trainer/main_ppo_sync.py` — Sync PPO Trainer (推荐)
- `verl/trainer/ppo/ray_trainer.py` — RayPPOTrainer 基类
- `verl/trainer/ppo/core_algos.py` — GAE/GRPO/REINFORCE++ 算法
- `verl/workers/engine_workers.py` — TrainingWorker + ActorRolloutRefWorker
- `verl/workers/rollout/base.py` — BaseRollout 抽象
- `verl/workers/rollout/llm_server.py` — vLLM Server 集成
- `verl/workers/reward_manager/` — Reward 计算管理器
- `verl/protocol.py` — DataProto 数据协议
- 相关: [verl 架构](verl-architecture.md), [verl PrefixGrouper](verl-prefix-grouper.md)
