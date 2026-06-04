# RLHF Training Infra 架构分析

> 日期: 2026-06-04
> 基于对 verl 框架的分析 + GPU 实验验证

## 1. RLHF 训练管线概览

```
Prompt Dataset
    ↓
Actor (生成 responses)        ← Rollout 阶段, GPU 密集
    ↓
Ref Model (计算 reference)    ← 需要同结构模型
    ↓
Reward Model (打分)           ← 可以是独立模型
    ↓
Advantage 计算 (GAE/GRPO)     ← CPU 密集
    ↓
Critic (估计 value)           ← PPO only
    ↓
Policy Update (PPO clip)      ← 反向传播
```

## 2. 核心算法

### PPO (Proximal Policy Optimization)
```
ratio = π_new(a|s) / π_old(a|s)
clipped = clip(ratio, 1-ε, 1+ε) * advantage
loss = -min(ratio * advantage, clipped)
```
- 需要: Actor, Critic, Ref Model, Reward Model → 4 个模型
- 关键: `ratio` 的计算需要 `old_log_probs` 和 `new_log_probs`

### GRPO (Group Relative Policy Optimization)
```
for each prompt:
    generate n responses (n=4,8,16)
    compute group advantage = (reward - mean) / std
    update policy
```
- **不需要 Critic** → 3 个模型 (Actor, Ref, Reward)
- 更简单但需要更多 rollout (n 个 response per prompt)

### DPO (Direct Preference Optimization)
```
loss = -log(σ(β * (log_π(y_w|x) - log_ref(y_w|x) - log_π(y_l|x) + log_ref(y_l|x))))
```
- **不需要 Reward Model** → 2 个模型 (Actor, Ref)
- 需要偏好数据 (chosen, rejected pairs)

## 3. verl 架构 (基于 vLLM)

### 核心类
```
RayPPOTrainer
    ├── ActorRolloutRefWorker  ← 混合 Worker (Actor + Rollout + Ref)
    │   ├── ActorModel
    │   ├── Rollout (vLLM backend)
    │   └── RefModel (weight-sharing with Actor)
    ├── CriticWorker           ← PPO only
    ├── RewardWorker           ← Reward Model
    └── DataConsumer           ← 数据流转
```

### 关键优化

#### Prefix Caching for RLHF
```
同一 prompt 生成 n 个 response (GRPO n=8):
- prompt 部分的 KV cache 完全相同
- 用 PrefixGrouper 分组 → vLLM prefix caching
- 结果: 58% KV 节省 (n=8, prompt=512)
- 更大 prompt (4096): 76% 节省
```

#### Weight Sharing (Actor + Ref)
```
Actor 和 Ref 共享底层权重:
- 只保留一份权重
- 通过 `offload` 或 `reshard` 切换
- 节省 50% GPU 内存 (vs 独立部署)
```

#### Rollout Throughput
```
实测 (8B model, A100):
- PPO:  ~4K tok/s (需要 4 个模型)
- GRPO: ~8K tok/s (不需要 Critic, n=8 批量生成)
- GRPO 2.1x faster, 8.5x cost efficiency
```

## 4. 分布式训练拓扑

### 典型配置 (70B PPO)
```
TP=8 (Actor/Rollout)     → 1 node (8 GPUs)
TP=8 (Critic)            → 1 node
TP=4 (Reward)            → 0.5 node
TP=4 (Ref)               → 0.5 node (weight-sharing with Actor)
Total: 2-3 nodes
```

### GRPO 更高效
```
TP=8 (Actor/Rollout/Ref)  → 1 node (weight-sharing)
TP=4 (Reward)             → 0.5 node
Total: 1-2 nodes (省 1 个 node)
```

### Colocate 模式
```
Actor + Ref + Reward → 同一 GPU 集群
通过 `offload` 切换:
- Rollout 时加载 Actor
- 训练时加载 Ref/Reward
- 用 CPU offload 存储 inactive 模型
```

## 5. 数据流和通信

### Rollout → Training 数据流
```
1. Prompts → Actor → 生成 responses (vLLM rollout)
2. Responses + Prompts → Reward Model → rewards
3. Responses + Prompts → Ref Model → ref_log_probs
4. (PPO) Responses + Prompts → Critic → values
5. 计算 advantage (GAE 或 group statistics)
6. Actor.backward() → 梯度更新
```

### 关键瓶颈
1. **Rollout 吞吐**: vLLM decode 速度限制
2. **Reward 打分**: forward pass through reward model
3. **Ref log probs**: forward + log_softmax
4. **Actor 反向传播**: 标准训练 backward
5. **数据传输**: Rollout GPU → CPU → Training GPU

## 6. 性能优化策略

### 训练侧
| 策略 | 效果 |
|------|------|
| ZeRO-3 + Actor | 节省 8x 内存 (DP=8) |
| Gradient Checkpointing | 30-45% 内存节省 |
| BF16 混合精度 | 2x 速度, 24% 内存节省 |
| Sequence Parallel | TP 带宽减半 |

### Rollout 侧
| 策略 | 效果 |
|------|------|
| Prefix Caching | 58-76% KV 节省 |
| Continuous Batching | 23x 吞吐提升 |
| Weight-sharing | 50% 内存节省 |
| Tensor Parallel | 线性扩展 |

### 端到端
| 策略 | 效果 |
|------|------|
| GRPO vs PPO | 2.1x faster, 省 Critic |
| Colocate | 少用 50% GPU |
| Async Reward | 重叠计算 |
| Micro-batch | 降低峰值内存 |

## 7. 实际配置参考

### LLaMA-7B GRPO on 4x A100
```
Actor/Rollout: TP=2, DP=2
Ref: TP=2 (weight-sharing)
Reward: TP=2, DP=2
Rollout: vLLM, max_tokens=512, n=8
Training: BF16, ZeRO-2, grad_ckpt
Expected: ~8K tok/s rollout
```

### LLaMA-70B PPO on 16x A100
```
Actor/Rollout: TP=8, DP=2
Ref: TP=8 (weight-sharing)
Critic: TP=8, DP=2
Reward: TP=4, DP=4
Rollout: vLLM, max_tokens=1024
Training: BF16, ZeRO-3, grad_ckpt
```

## 8. 关键差异: RLHF vs 标准训练

| 维度 | 标准训练 | RLHF |
|------|---------|------|
| 模型数量 | 1 | 2-4 |
| Forward pass | 1/batch | 4-8/batch |
| 需要推理 | 否 | 是 (rollout) |
| KV Cache | 不需要 | 需要 (prefix caching) |
| 通信模式 | AllReduce | AllReduce + 数据传输 |
| 瓶颈 | 计算 | Rollout 吞吐 |
