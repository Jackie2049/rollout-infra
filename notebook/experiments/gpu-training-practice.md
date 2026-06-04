# GPU 实验记录 — 训练实战

> 2026-06-04 | NVIDIA A16 15GB | MiniGPT 训练 + AMP 对比

## 实验 1: AMP 混合精度训练对比 ⭐

模型: MiniGPT (16M params, FP32 master weights), 100 steps

| 模式 | ms/step | Loss | Peak Mem | Speedup |
|------|---------|------|----------|---------|
| FP32 (baseline) | 267.2 | 10.89 | 3765 MB | 1.0x |
| **FP16 + AMP (GradScaler)** | **128.6** | **10.89** | **2856 MB** | **2.08x** |
| FP16 no AMP | 124.8 | **11.83** | 2856 MB | 2.14x |
| BF16 + AMP | 127.4 | 10.90 | 2856 MB | 2.10x |
| **BF16 native (no AMP)** | **127.0** | **10.89** | **2856 MB** | **2.10x** |

### 关键发现

1. **FP16 必须配合 GradScaler**: 无 AMP 时 loss 发散到 11.83 (vs 10.89)
   - 原因: FP16 指数范围太小 (5 bit), 梯度容易下溢 (变成 0)
   - GradScaler 通过放大 loss 来防止梯度下溢
2. **BF16 不需要 GradScaler**: native BF16 就能稳定训练
   - BF16 指数范围与 FP32 相同 (8 bit), 不会下溢
   - 精度略低于 FP16 (mantissa 7 bit vs 10 bit), 但训练中足够
3. **BF16 native 是最佳选择**: 无需额外配置, 速度和 FP16+AMP 相同
4. **AMP 内存节省 24%**: 3765→2856 MB, 因为中间激活用 FP16

### AMP 工作原理

```
FP32 master weights → autocast → FP16/BF16 forward → FP16 loss
                                                    ↓
                                            GradScaler.scale(loss)
                                                    ↓
                                            FP16 gradients
                                                    ↓
                                            GradScaler.unscale_()
                                                    ↓
                                            FP32 gradients (master)
                                                    ↓
                                            optimizer.step() (FP32)
```

## 实验 2: MiniGPT 训练 (3.3M, 有模式数据)

使用带重复模式的数据训练，模型可以学到规律。

| Step | Loss | ms/step | tok/s | Peak MB | LR |
|------|------|---------|-------|---------|-----|
| 0 | 5.028 | 224.9 | - | 221 | 5e-4 |
| 50 | 4.386 | 26.5 | 154K | 247 | 4.9e-4 |
| 100 | 2.523 | 26.7 | 153K | 247 | 4.5e-4 |
| 150 | 1.616 | 26.6 | 154K | 247 | 4.0e-4 |
| 200 | 1.553 | 26.8 | 153K | 247 | 3.3e-4 |
| 300 | 1.490 | 26.7 | 154K | 247 | 1.7e-4 |
| 499 | 1.468 | 26.3 | 156K | 247 | 0 |

### Loss 曲线分析

- Step 0-100: 快速下降 (5.03→2.52), 模型学习基础 token 分布
- Step 100-200: 减速 (2.52→1.55), 开始学到重复模式
- Step 200-500: 收敛 (1.55→1.47), 模式已基本掌握
- 理论下限: log(pattern_vocab) ≈ log(80) ≈ 4.38... 实际更复杂

### 评估: Next-token 预测准确率

| Pattern Length | Accuracy | 说明 |
|---------------|----------|------|
| 4 | 31.7% | 短模式可学 |
| **8** | **34.3%** | **训练时使用的长度** |
| 16 | 11.0% | 超出模型能力 |
| 32 | 3.9% | 几乎随机 |

**分析**: 模型成功学到了 pattern_len=8 的重复规律 (34.3% >> 1/128=0.8% 随机基线)。更长的模式需要更多参数或更长训练。

## 实战经验总结

1. **BF16 > FP16**: 无需 GradScaler, 训练更稳定
2. **模型保持 FP32**: AMP 通过 autocast 做前向 FP16, 梯度仍累加到 FP32
3. **显存计算**: AdamW 训练 = 8×params (master+grads+m+v) + activations
4. **A16 训练能力**: 3.3M 模型仅用 247MB, 可训练到 ~1.3B 参数
5. **Cosine LR Schedule**: 稳定收敛, 不会震荡
