# RTX 4090 Knowledge Distillation 实测

> GPU: NVIDIA GeForce RTX 4090, 24GB
> Teacher: 4.8M params (6层, 8头, d=256)
> Student: 444K params (2层, 4头, d=128)
> 压缩比: 10.9x
> 日期: 2026-06-05

## 1. 模型对比

| Model | Params | Layers | Heads | d_model |
|-------|--------|--------|-------|---------|
| Teacher | 4,823,552 | 6 | 8 | 256 |
| Student | 443,648 | 2 | 4 | 128 |
| Ratio | 10.9x | 3x | 2x | 2x |

## 2. 训练结果

| Model | Train Loss | Test Loss | Test PPL | Top-5 Agreement |
|-------|-----------|-----------|----------|-----------------|
| Teacher | 1.28 | 2.063 | 7.87 | — |
| Student (baseline) | 2.60 | **2.492** | **12.08** | **0.405** |
| Student (distilled, T=4) | 1.68 | 2.677 | 14.54 | 0.366 |

**反直觉发现**: 蒸馏学生比从头训练的学生**更差** (PPL 14.54 vs 12.08)!

## 3. 为什么蒸馏失败了?

### 3.1 负面结果分析

这是一个有价值的负面结果。蒸馏效果不好的原因:

1. **合成数据太简单**: 数据是简单的模式 (重复/计数/斐波那契), Teacher 的 "dark knowledge" (软标签中的细粒度信息) 在简单数据上没有额外价值

2. **容量差距过大**: 10.9x 压缩, Student 太小无法拟合 Teacher 的知识分布

3. **KL loss 误导**: KL 散度让 Student 去模仿 Teacher 的输出分布, 但 Student 容量不足以同时拟合硬标签和软标签, α=0.7 给 KL 太高权重

4. **Temperature 过高**: T=4 使 Teacher 的分布过于平滑, Student 无法从中提取有用信号

### 3.2 Temperature Sweep 验证

| Temperature | Test Loss | Test PPL | Top-5 Agreement |
|-------------|-----------|----------|-----------------|
| 1.0 | **2.684** | **14.64** | **0.383** |
| 2.0 | 2.806 | 16.55 | 0.355 |
| 4.0 | 3.006 | 20.20 | 0.327 |
| 8.0 | 3.212 | 24.82 | 0.310 |
| 16.0 | 3.308 | 27.34 | 0.291 |

**结论**: Temperature 越高, 效果越差! T=1 (最低) 效果最好, 但仍不如 baseline。

### 3.3 Alpha Sweep 验证

| α (KL weight) | Test Loss | Test PPL |
|---------------|-----------|----------|
| 0.0 (纯 CE) | **2.700** | **14.87** |
| 0.3 | 2.748 | 15.62 |
| 0.5 | 2.807 | 16.56 |
| 0.7 | 3.011 | 20.31 |
| 1.0 (纯 KL) | 3.459 | 31.80 |

**结论**: α 越低越好! α=0 (纯 CE, 无蒸馏) 效果最好, 进一步验证了在这个设置下蒸馏是有害的。

## 4. 什么时候蒸馏有效?

根据文献和实践经验, 蒸馏在以下条件有效:

1. **真实数据**: ImageNet, Wikipedia 等复杂分布, Teacher 的 dark knowledge 有价值
2. **适中压缩比**: 2-4x 最佳, 超过 10x 收益递减
3. **大 Teacher**: Teacher 需要足够大, 其软标签才包含有用信息
4. **对齐任务**: 蒸馏用于对齐 (如 Alpaca) 比用于预训练更有效
5. **中间层蒸馏**: 不只蒸馏 logits, 还蒸馏 hidden states 和 attention maps

### 4.1 经典成功案例

| 论文 | Teacher | Student | 压缩比 | 效果 |
|------|---------|---------|--------|------|
| DistilBERT (2019) | BERT-base | 6→6层 | 2x | 97% 性能 |
| TinyBERT (2020) | BERT-base | 12→4层 | 7.5x | 96.8% |
| MiniLM (2020) | BERT-large | 24→6层 | 12x | 95% |
| Alpaca (2023) | GPT-4 | LLaMA-7B | N/A | 指令跟随 |

## 5. 实用建议

1. **先试 baseline**: 蒸馏不是万能的, 小模型从头训练可能已经够好
2. **从 α=0.5, T=2 开始调**: 这是文献中最常见的起点
3. **监控两个 loss**: 如果 KL loss 下降但 CE loss 上升, 说明 Student 在牺牲硬标签拟合
4. **考虑中间层蒸馏**: 仅 logits 蒸馏效果有限
5. **真实数据是关键**: 合成/简单数据上蒸馏收益很小

## 6. Compute

| Component | Value |
|-----------|-------|
| Peak GPU memory | 767 MB (Teacher + Student) |
| Throughput | ~570K tok/s |
| Training time | ~3 min (RTX 4090) |
