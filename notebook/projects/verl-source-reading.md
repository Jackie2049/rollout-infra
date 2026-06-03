# verl 源码架构阅读

> HybridFlow (EuroSys 2025) 开源实现：字节跳动 Seed 团队的 RL 训练框架

## 1. 框架定位

verl 是一个灵活的、生产就绪的 RL for LLM 训练框架，支持 PPO, GRPO, REINFORCE++, RLOO, ReMax, DAPO, GSPO 等多种算法。

**核心特点**:
- 单控制器 + Ray 分布式执行
- 训练 (FSDP/Megatron) + 推理 (vLLM/SGLang) 混合引擎
- 注册表模式实现算法可扩展性
- Prefix Grouper 实现训练时前缀共享

## 2. 顶层架构

```
verl/
├── protocol.py           # DataProto 数据交换协议
├── base_config.py        # 不可变配置
├── single_controller/    # Ray 分布式控制器
│   ├── base/            # Worker, WorkerGroup, ResourcePool
│   └── ray/             # Ray actor 管理
├── workers/             # 核心引擎
│   ├── engine/          # FSDP/Megatron 训练引擎
│   ├── rollout/         # vLLM/SGLang/TRT-LLM 推理后端
│   ├── reward_manager/  # 奖励计算
│   └── engine_workers.py # TrainingWorker, ActorRolloutRefWorker
├── trainer/             # 入口脚本
│   └── ppo/             # RayPPOTrainer, SyncPPOTrainer
│       ├── ray_trainer.py  # PPO 循环编排
│       ├── core_algos.py   # 优势估计 + 策略损失
│       └── prefix_grouper_utils.py  # 前缀共享
├── utils/               # 工具集
└── models/              # HF 模型 monkey-patch
```

## 3. 核心数据协议: DataProto

**文件**: `verl/protocol.py` (~1346 行)

`DataProto` 是贯穿所有 worker 的数据交换对象：

```python
class DataProto:
    batch: TensorDict         # 张量数据 (input_ids, attention_mask, ...)
    non_tensor_batch: dict    # 非张量数据 (uid, reward_fn, ...)
    meta_info: dict           # 元信息

    # 支持操作
    def chunk(self, n)        # 分成 n 份 (DP 分片)
    def split(self, sizes)    # 按指定大小分割
    def concat(self, others)  # 合并
    def select(self, keys)    # 选择子集
    def repeat(self, n)       # 重复 (用于 GRPO 多 response)
    def make_iterator(bs)     # 创建 mini-batch 迭代器
```

**DataProtoFuture**: 惰性 Ray future，支持异步执行和调度。

## 4. 分布式控制: 单控制器 + Ray

### 4.1 Worker 层次

```
Driver (主进程)
  └── WorkerGroup (管理一组 Ray Actor)
        ├── Worker 0 (GPU 0)
        ├── Worker 1 (GPU 1)
        └── ...
```

### 4.2 调度装饰器

```python
@register(dispatch_mode=...)  # 自动数据分片/收集
```

调度模式:
- `ONE_TO_ALL`: 广播相同数据到所有 worker
- `DP_COMPUTE`: 按 DP 维度分片，分别计算，收集结果
- `DP_COMPUTE_PROTO`: 同上，但用于 DataProto
- `RANK_ZERO`: 只在 rank 0 执行
- `ALL_TO_ALL`: 全互通信

### 4.3 资源管理

`ResourcePoolManager` 创建 Ray placement groups，将角色 (Actor, Critic, Reward, Ref) 映射到资源池。

## 5. PPO 训练循环

**文件**: `verl/trainer/ppo/ray_trainer.py` (1771 行)

### 5.1 完整流程

```
每个 epoch:
  1. DataLoader → batch + uid
  2. async_rollout_manager.generate_sequences()  ← 生成 rollout
  3. 奖励计算 (奖励模型 or 函数奖励)
  4. old_log_prob = actor.forward(batch)          ← 重新计算 log prob
  5. ref_log_prob = ref.forward(batch)            ← 参考 log prob (可选)
  6. values = critic.forward(batch)               ← 值预测 (可选)
  7. KL 惩罚加入奖励 (可选)
  8. 优势估计 (GAE / GRPO / ...)                  ← 在 Driver 上计算
  9. critic.update(batch)                         ← 更新 Critic (可选)
  10. actor.update(batch)                         ← 更新 Actor
  11. rollout.update_weights(actor.weights)       ← 同步权重到推理引擎
  12. 验证 (定期)
```

### 5.2 条件角色

| 算法 | Actor | Critic | Ref Policy | Reward Model |
|------|-------|--------|------------|-------------|
| PPO | 需要 | 需要 (GAE) | 可选 (KL) | 可选 |
| GRPO | 需要 | 不需要 | 可选 (KL) | 可选 |
| REINFORCE++ | 需要 | 不需要 | 可选 | 可选 |
| RLOO | 需要 | 不需要 | 可选 | 可选 |

## 6. GRPO vs PPO 核心差异

**文件**: `verl/trainer/ppo/core_algos.py` (2488 行)

### 6.1 优势估计

**PPO (GAE)**:
```python
def compute_gae_advantage_return(token_level_rewards, values, gamma, lam):
    # TD residual: delta = r + gamma * V(s+1) - V(s)
    # GAE: A = sum(gamma*lam)^i * delta(s+i)
    # 需要学习的 Value 函数
```

**GRPO**:
```python
def compute_grpo_outcome_advantage(token_level_rewards, uid):
    # 1. 每个 response 的总奖励 = sum(token_rewards)
    # 2. 按 uid (prompt ID) 分组
    # 3. advantage = (score - group_mean) / (group_std + eps)
    # 4. 广播到 token-level
    # 不需要 Value 函数!
```

### 6.2 关键差异

| 方面 | PPO (GAE) | GRPO |
|------|-----------|------|
| Critic | 需要 | 不需要 |
| 优势类型 | Token-level (dense) | Outcome (scalar per response) |
| 基线 | 学习的 Value | 组均值 |
| 分组 | N/A | 按 uid (相同 prompt) |
| 模型数量 | 4 (Actor+Critic+Ref+RM) | 2-3 (Actor+Ref+RM optional) |
| GPU 需求 | 更多 | 更少 |

## 7. 引擎层: FSDP + 混合推理

### 7.1 Engine Registry

```python
@EngineRegistry.register(model_type, backend, device)
# backend: fsdp, fsdp2, megatron, automodel, torchtitan, ...
# model_type: actor, value_model, reward_model
# device: cuda, npu
```

### 7.2 FSDP Engine

**文件**: `verl/workers/engine/fsdp/transformer_impl.py`

- 用 FSDP 包装 HF 模型
- 支持 LoRA (PEFT), 激活 offload, 序列并行 (Ulysses), 混合精度
- FSDP1 + FSDP2 双引擎支持
- 2D Device Mesh (DDP + FSDP)

### 7.3 混合引擎 (Colocate)

训练和推理共享 GPU:
1. `sleep_replicas()` — 释放推理引擎的 GPU 内存
2. 训练步骤 (forward + backward + optimizer)
3. `update_weights()` — 同步训练权重到推理引擎
4. 推理步骤 (rollout generation)

这允许 70B GRPO 在 8 GPU 上跑 (vs PPO 需要 16 GPU)。

## 8. Rollout 基础设施

### 8.1 推理后端

```python
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): ServerAdapter,
    ("sglang", "async"): ServerAdapter,
    ("trtllm", "async"): ServerAdapter,
}
```

### 8.2 LLM Server Manager

- `GlobalRequestLoadBalancer`: Ray actor，粘性会话 + 最小负载均衡
- `LLMServerManager`: 管理 rollout replicas 生命周期
- 支持权重热更新 (训练后同步到推理引擎)

## 9. Prefix Grouper: 训练时前缀共享

**文件**: `verl/trainer/ppo/prefix_grouper_utils.py` (236 行)

### 9.1 核心机制

```python
def pg_forward(model, batch, prefix_grouper):
    # 1. 按 uid 分组 (相同 prompt 的多个 response)
    # 2. 提取 prefix tokens (共享部分)
    # 3. 模型只 forward 一次 prefix
    # 4. 对每个 response 的 suffix 分别 forward
    # 5. 合并 log_probs 和 entropy
```

### 9.2 启用条件

- 配置: `use_prefix_grouper = True`
- 批处理平衡: `get_group_balanced_partitions()` 确保相同 uid 的样本在同一 DP rank
- 约束: `num_uid_groups % dp_size == 0`

### 9.3 与服务时 Prefix Caching 的区别

| 特性 | 服务时 (vLLM) | 训练时 (verl PG) |
|------|--------------|-----------------|
| 目的 | 减少 KV Cache 存储 | 减少前向计算 |
| 机制 | Block hash 复用 | Attention 分解 |
| 触发 | 自动检测 | 显式配置 |
| 数据流 | KV Cache blocks | Forward pass |

## 10. 算法扩展: 注册表模式

### 10.1 优势估计器

```python
@register_adv_est("grpo")
@register_adv_est("gae")
@register_adv_est("reinforce_plus_plus")
@register_adv_est("rloo")
@register_adv_est("remax")
# 12+ 种
```

### 10.2 策略损失

```python
@register_policy_loss("vanilla")   # PPO-clip
@register_policy_loss("dppo_tv")
@register_policy_loss("gspo")
# 12+ 种
```

新算法只需注册装饰器 + 实现函数，无需修改核心代码。

## 11. 关键架构洞察

1. **单控制器设计**: Driver 只做轻量级操作 (优势计算)，重计算在 worker 上。这避免了 Driver 成为瓶颈。

2. **混合引擎**: 训练和推理共享 GPU，通过 sleep/wake 机制管理内存。这是 verl 能用更少 GPU 跑 70B RL 的关键。

3. **GRPO 简化**: 去掉 Critic 后，模型数量从 4 减到 2-3，GPU 需求减半。代价是用组均值替代学习的 Value 函数。

4. **Prefix Grouper 是训练优化**: 与服务时 KV Cache 复用不同，这里是在前向传播中直接复用 prefix 的计算。

5. **注册表可扩展**: 算法、后端、奖励函数都通过注册表模式添加，核心代码无需修改。

## 参考资料

- 源码路径: `verl/` (shallow clone)
- 论文: HybridFlow (EuroSys 2025)
- 相关笔记: [RLHF/GRPO 训练基础设施](../fundamentals/rlhf-training-infra.md), [verl 架构](verl-architecture.md), [Prefix Caching](../fundamentals/prefix-caching.md)
- 相关工具: `tools/ray_schedule_sim.py`
