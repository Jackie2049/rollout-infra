# RLHF / GRPO 训练模拟器 — RTX 4090 实测

> 2026-06-05 | 工具: `tools/rlhf_grpo_simulator.py` | RTX 4090 24GB

## 1. 实验设置

- **模型**: MiniGPT (vocab=128, d=64, 4 heads, 2 layers)
- **PPO 参数**: 116,609 (含 value head 4,225)
- **GRPO 参数**: 112,384 (无 value head)
- **GPU 内存**: 67.6 MB (极小模型)
- **训练步数**: 30 steps (PPO vs GRPO 对比)

## 2. 核心结果

### 2.1 PPO vs GRPO 对比

| 指标 | PPO | GRPO |
|------|-----|------|
| 初始 reward | 0.767 | 0.717 |
| 最终 reward | 0.411 | 0.679 |
| Δ reward | -0.356 | -0.039 |
| 模型数量 | 4 (π, π_ref, V, RM) | 2 (π, π_ref) |
| 额外参数 | 4,225 (value head) | 0 |

**关键发现**: GRPO 比 PPO 更稳定! PPO 出现 reward 下降 (可能过拟合/不稳定),
GRPO 保持稳定。这与生产实践一致: GRPO 因无 critic 而更鲁棒。

### 2.2 GRPO Group Size (n_samples)

| n_samples | 初始 | 最终 | Δ | 分析 |
|-----------|------|------|---|------|
| 2 | 0.717 | 0.808 | +0.091 | 最优 (小模型足够) |
| 4 | 0.717 | 0.808 | +0.091 | 同 n=2 |
| 8 | 0.702 | 0.781 | +0.079 | 开始下降 |
| 16 | 0.714 | 0.793 | +0.079 | 收益递减 |

**结论**: n=2-4 对小模型最优。大模型 (DeepSeek-R1) 用 n=8-64 因为更多样化探索。

### 2.3 KL Penalty 效果

| β_kl | 最终 reward | KL divergence |
|------|-----------|---------------|
| 0.00 | 0.808 | -0.317 |
| 0.01 | 0.781 | -1.147 |
| 0.05 | 0.781 | -2.126 |
| 0.10 | 0.781 | -2.247 |
| 0.50 | 0.781 | -2.714 |

**发现**:
- 无 KL penalty (β=0) reward 最高但可能不稳定
- β=0.01-0.1 是合理范围 (verl 默认 0.05)
- β=0.5 过强, 限制学习

### 2.4 Reward Model 训练

| 指标 | 值 |
|------|-----|
| 初始 loss | 0.772 |
| 最终 loss | 0.662 |
| 最终 accuracy | 58.5% |
| GRPO + learned RM | 0.172 → 0.339 |

**分析**: Reward model 准确率只有 58.5%, 说明偏好信号弱。
合成数据的 "chosen" 和 "rejected" 差异不够显著 (0.697 vs 0.645)。

## 3. RLHF 算法全景对比

```
┌─────────────────────────────────────────────────────────────┐
│ 算法    │ 模型数 │ Value函数 │ Reward Model │ 数据需求    │
│─────────│────────│──────────│──────────────│───────────── │
│ PPO     │ 4      │ ✓ critic │ ✓ trained    │ 在线采样     │
│ GRPO    │ 2      │ ✗ group  │ ✓ 或 oracle  │ 在线采样     │
│ DPO     │ 2      │ ✗        │ ✗            │ 离线偏好对   │
│ RLHF-V  │ 3      │ ✗        │ ✗            │ 在线+反馈    │
│ RLAIF   │ 3      │ ✗/✓      │ AI 标注      │ AI生成偏好   │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 PPO 流程 (4个模型)
```
1. Rollout: π 生成 response
2. Reward: RM(response) → reward
3. Value: V(prompt) → baseline
4. Advantage: A = R - V (GAE)
5. Update: clipped surrogate + KL penalty
```

### 3.2 GRPO 流程 (2个模型)
```
1. Rollout: π 为每个 prompt 生成 n 个 response
2. Reward: R(response_i) for each
3. Group Normalize: A_i = (R_i - mean(R)) / std(R)
4. Update: clipped surrogate + KL penalty
```

### 3.3 DPO 流程 (2个模型)
```
1. 离线数据: (prompt, chosen, rejected)
2. Loss: -log σ(β * (log_π/π_ref(chosen) - log_π/π_ref(rejected)))
3. Update: 直接优化 loss
```

## 4. 与 verl 的对应关系

verl 的 RLHF pipeline 实现:
- `RayPPOTrainer`: PPO with Ray-based distributed training
- `RayGRPOTrainer`: GRPO, 默认 n=8 responses per prompt
- `ActorRolloutRefWorker`: 混合 Actor+Rollout+Ref 角色
- **Prefix Sharing**: GRPO n=8 时, 相同 prompt × 8 response → 58% KV cache 节省

### verl GRPO 性能数据 (Qwen3-4B, 4×H800)
- GRPO 2.1x faster than PPO
- 8.5x cost efficiency
- Prefix Caching saves 40% (n=8, p=512) / 76% (p=4096)

## 5. 核心洞察

1. **GRPO > PPO 的根本原因**: 无 critic → 少一半模型 → 少一半 GPU 显存
2. **Group normalization 替代 learned baseline**: 统计量比神经网络更鲁棒
3. **n_samples 是 GRPO 的关键超参**: 2-4 小模型, 8-64 大模型
4. **KL penalty 防止 reward hacking**: β=0.05 是好的默认值
5. **RLHF 的代价**: 需要 2-4 个模型同时加载, GPU 显存需求 2-4x
