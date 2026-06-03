# Gradient Checkpointing / Activation Recomputation

> 用计算换显存 — 从 O(n) 激活到 O(sqrt(n))

## 1. 核心问题：激活值的显存占用

训练时，前向传播的中间激活需要保存在显存中，用于反向传播计算梯度。

### 1.1 激活显存估算

对于一个 Transformer 层：

```
每层的激活大小 ≈ 11 × b × s × h  bytes (FP16)

b = batch size
s = sequence length
h = hidden dimension

例如: b=8, s=4096, h=4096, L=32 层
每层激活 ≈ 11 × 8 × 4096 × 4096 × 2 bytes = ~2.75 GB
32 层总计 ≈ 88 GB  ← 仅仅激活值！
```

### 1.2 激活 vs 参数的显存占比

| 模型 | 参数显存 | 激活显存 (b=8, s=2K) | 激活占比 |
|------|---------|---------------------|---------|
| 1.3B | 2.6 GB | ~16 GB | 86% |
| 7B | 14 GB | ~60 GB | 81% |
| 13B | 26 GB | ~90 GB | 78% |

**结论**：激活值占训练显存的大部分，尤其是长序列和大 batch 时。

## 2. Gradient Checkpointing 原理

### 2.1 基本思想

不保存所有中间激活，只保存部分**检查点**。反向传播时，从最近的检查点重新计算需要的激活。

```
无 Checkpointing:
  前向: [A1] → [A2] → [A3] → [A4] → [A5] → [A6] → Loss
  保存: A1, A2, A3, A4, A5, A6  ← 全部保存
  显存: O(n) 激活
  计算: 1x 前向

标准 Checkpointing (sqrt(n) 个检查点):
  前向: [A1] → A2 → [A3] → A4 → [A5] → A6 → Loss
  保存: A1, A3, A5              ← 只保存检查点
  反向时:
    需要A6: 从A5重算 → A6 ← 用完释放
    需要A4: 从A3重算 → A4, A5 → A6 ← 用完释放
    需要A2: 从A1重算 → A2, A3 → A4 ← 用完释放
  显存: O(sqrt(n)) 激活
  计算: ~2x 前向 (多一次重计算)
```

### 2.2 数学分析

对于 n 层网络：

| 策略 | 保存的激活 | 重计算次数 | 总计算量 |
|------|-----------|-----------|---------|
| 无 checkpointing | n | 0 | 1x |
| 保存 sqrt(n) 个检查点 | sqrt(n) | sqrt(n) | 2x |
| 每层都 checkpoint | 0 (或 1) | n | n+1 x |

**最优策略**：保存 sqrt(n) 个检查点，以 O(sqrt(n)) 显存换取 2x 计算。

### 2.3 显存节省公式

```
无 checkpointing: Memory_act = n × (每层激活大小)
标准 checkpointing: Memory_act = sqrt(n) × (每层激活大小) + sqrt(n) × (每段重计算)

节省因子: 从 n 降到 ~2×sqrt(n)
对于 32 层: 从 32x 降到 ~11.3x → 节省约 65%
```

## 3. 选择性激活重计算 (Selective Recomputation)

### 3.1 核心观察

并非所有激活都一样大！Transformer 层中：

```
Attention 计算:
  Q, K, V 矩阵: 3 × b × s × h → 相对小
  Attention scores: b × heads × s × s → 巨大！(与 s² 成正比)
  Attention output: b × s × h → 相对小

MLP:
  中间激活: b × s × 4h → 较大 (4x hidden)

Dropout mask:
  b × s × h → 小但需要保存
```

### 3.2 选择性策略

```python
# Megatron-LM 的选择性重计算
只重计算 Attention scores (与 s² 成正比的部分)
保存 MLP 中间激活、LayerNorm 输出等

收益:
  - 显存节省: 比全 checkpointing 更多 (因为保存的是较小的部分)
  - 计算开销: 比全 checkpointing 更少 (只重计算 attention)
```

### 3.3 三种策略对比

| 策略 | 激活显存 | 额外计算 | 实现复杂度 |
|------|---------|---------|-----------|
| 无 checkpointing | O(n) | 0 | 低 |
| 标准 (sqrt(n) 检查点) | O(sqrt(n)) | ~1x | 中 |
| 选择性 (只重算 attention) | O(n) 但常数小 | ~0.2x | 高 |

**实践推荐**：长序列 (s > 4K) 时用选择性重计算，短序列时可以不用。

## 4. Megatron-LM 中的实现

### 4.1 配置选项

```bash
# 完全不 checkpoint
--recompute-granularity null

# 按层 checkpoint (每层一个检查点)
--recompute-granularity full
--recompute-method block  # 或 uniform

# 选择性 checkpoint (只重算 attention)
--recompute-granularity selective
```

### 4.2 实现机制

```python
# PyTorch 的 gradient checkpointing 实现
from torch.utils.checkpoint import checkpoint

def custom_forward(layer, x):
    # 这段代码在前向时不保存中间激活
    # 反向时从 x 重新计算
    return layer(x)

# 使用 checkpoint
output = checkpoint(custom_forward, layer, input_tensor)
```

### 4.3 Megatron 的 Pipeline Parallel 交互

```
在 PP 中，checkpointing 的行为:
  - 每个 pipeline stage 独立决定是否 checkpoint
  - warmup 阶段: 需要保存更多激活 (等待反向传播)
  - steady state 阶段: 可以更激进地释放

Megatron 优化:
  num_microbatches_with_partial_activation_checkpoints
  → 在 warmup microbatch 中对部分层做 checkpoint
  → 在 steady state 中保存完整激活
  → 平衡显存和计算
```

## 5. 与其他技术的交互

### 5.1 与 TP 的交互

```
TP 中每个 GPU 只计算一部分 head 的 attention:
  - 激活值已经减少 (只存部分 head)
  - checkpointing 的收益相对减少
  - 但 MLP 部分的激活仍然完整 → 仍有收益
```

### 5.2 与 Sequence Parallel 的交互

```
SP 中激活沿序列维度切分:
  - 每个 GPU 只存 s/TP 的激活
  - checkpointing 仍有效，但收益减少
  - Attention scores: (s/TP)² 而非 s² → 更小
```

### 5.3 与 ZeRO 的交互

```
ZeRO 分区参数和优化器状态:
  - 激活值不被 ZeRO 分区
  - checkpointing 是减少激活显存的主要手段
  - 两者互补: ZeRO 减少参数/梯度/优化器，checkpointing 减少激活
```

## 6. 性能影响

### 6.1 吞吐量影响

```
标准 checkpointing (sqrt(n)):
  吞吐量下降: ~20-35%
  显存节省: ~60-70%

选择性 checkpointing:
  吞吐量下降: ~5-15%
  显存节省: ~40-50%

不做 checkpointing:
  吞吐量: 最高
  显存: 需要最多
```

### 6.2 实际测量 (参考数据)

| 配置 | 显存 (GB) | 吞吐量 (tokens/s) |
|------|----------|-------------------|
| 无 checkpoint | 75 | 10000 |
| 选择性 | 50 | 9200 |
| 完全 | 35 | 7500 |

## 7. 学习要点

1. **核心权衡**：用计算换显存 — 重计算 vs 存储
2. **sqrt(n) 策略**是最优 — O(sqrt(n)) 显存，2x 计算
3. **选择性重计算**是工程最优 — 只重算 attention scores (s² 相关)
4. **Megatron 的分层策略** — warmup 时部分 checkpoint，steady state 完整保存
5. **与 ZeRO 互补** — ZeRO 处理参数/优化器，checkpointing 处理激活
6. **长序列训练的必需品** — s=128K 时激活可达数百 GB

## 参考

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (Megatron-LM, Korthikanti et al., 2022)
- [PyTorch Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
