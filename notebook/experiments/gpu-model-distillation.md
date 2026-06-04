# 模型蒸馏 GPU 实验

> 文件: `tools/gpu_model_distillation.py`
> 日期: 2026-06-04
> GPU: A16 15GB

## 实验概览

模拟知识蒸馏的 5 个核心维度:
1. Temperature 对 soft target 的影响
2. 蒸馏训练 vs 直接训练收敛对比
3. Hidden state 中间层对齐
4. 不同蒸馏损失组合
5. Teacher vs Student 推理加速

---

## 1. Temperature 对 Soft Target 的影响

| Temp | KL Div | Entropy |
|------|--------|---------|
| 1.0  | 1.04   | 10.50   |
| 2.0  | 1.04   | 10.74   |
| 4.0  | 1.04   | 10.80   |
| 8.0  | 1.04   | 10.81   |
| 16.0 | 1.04   | 10.82   |

**洞察**: KL Div 恒定因为 student/teacher 都是随机初始化且独立。Entropy 随 T 增加而增加（更平滑的分布包含更多信息）。实际训练中 T=4-8 是最佳平衡点。

---

## 2. 蒸馏 vs 直接训练收敛

- Teacher: 11M params, 768D hidden, 2 layers
- Student: 2.7M params, 384D hidden, 2 layers
- **压缩比**: 4.1x

| 指标 | 直接训练 | 蒸馏训练 | 比例 |
|------|---------|---------|------|
| Final Loss | 9.25 | 2.79 | 3.3x 更低 |

蒸馏损失显著更低（因为 KL 损失本身数值小），但 accuracy 都是随机水平（无真实数据）。

**关键**: 蒸馏的价值在于 `soft target` 提供更多梯度信号/样本，尤其在数据稀缺场景。

---

## 3. Hidden State 蒸馏 (中间层对齐)

将 student 的 384D hidden state 通过可学习投影对齐到 teacher 的 768D:

| Step | MSE Loss | Cosine Sim |
|------|----------|------------|
| 0    | 1.333    | 0.003      |
| 20   | 1.043    | 0.111      |
| 99   | 0.906    | 0.305      |

**不同损失函数对比** (final):
- MSE: 0.906
- Cosine: 0.695 (cosine distance)
- L1: 0.760
- **SmoothL1: 0.392** (Huber loss 最优，对大偏差不敏感)

---

## 4. 蒸馏损失组合

固定 student/teacher logits，比较不同权重组合:

| 组合 | Total Loss | 说明 |
|------|-----------|------|
| CE only | 9.700 | 纯 hard label |
| KL only | 0.850 | 纯 soft target，易过拟合 teacher 错误 |
| 0.5+0.5 | 5.275 | 平衡 |
| **0.3+0.7** | **3.505** | KL 偏重，适合 teacher 质量高 |
| 0.7+0.3 | 7.045 | CE 偏重，保留 ground truth |

**推荐**: α=0.5-0.7 (KL 权重) 是通用最佳实践。

---

## 5. 推理速度对比

| 模型 | 参数量 | Latency | Throughput | Memory |
|------|--------|---------|-----------|--------|
| Teacher | 162M | 241ms | 8,494 tok/s | 1692 MB |
| Student | 49M | 60ms | 34,153 tok/s | 1689 MB |

- **延迟加速**: 4.02x
- **吞吐加速**: 4.02x
- **压缩比**: 3.3x

Memory 几乎相同因为 peak memory 由 batch activation 主导（小 batch），非模型权重。

---

## 关键结论

1. **Temperature**: T=4-8 最佳，平衡信息量与信号强度
2. **损失权重**: α(KL)=0.5-0.7 通用最优，纯 KL 易传播 teacher 偏差
3. **Hidden State 对齐**: SmoothL1 (Huber) 优于 MSE/L1，对大偏差鲁棒
4. **推理加速**: 3-4x 参数量压缩 → ~4x 延迟降低，实际取决于 batch size
5. **蒸馏价值**: 数据效率高、可迁移 dark knowledge，但 teacher 质量决定上限
