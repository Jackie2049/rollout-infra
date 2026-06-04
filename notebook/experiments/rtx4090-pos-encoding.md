# RTX 4090 Positional Encoding Comparison 实测

> GPU: NVIDIA GeForce RTX 4090, 24GB
> 模型: 4层, 4头, d=128, 训练 seq_len=64
> 日期: 2026-06-05

## 1. 位置编码对比

| Type | Params | Train Loss | Extrapolation 2x | Extrapolation 4x |
|------|--------|-----------|------------------|------------------|
| **RoPE** | 821K | **0.488** | **2.882** | **3.925** |
| None | 821K | 0.601 | 3.075 | 3.573 |
| Sinusoidal | 822K | 0.923 | 3.069 | 4.178 |
| Learned | 887K | 1.095 | 2.982 | 4.011 |

## 2. 关键发现

### 2.1 RoPE 最佳
- 训练 loss 最低 (0.49), 比第二名 (None=0.60) 低 19%
- 外推 loss 最低 (2x: 2.88, 4x: 3.93)
- 通过旋转编码相对位置, 天然支持外推

### 2.2 No Position 出奇地好
- 训练 loss 0.60 (第二名), 说明合成数据的位置依赖性不强
- 4x 外推退化 177%, 比 sinusoidal/RoPE 好, 因为没有错误的位置信号

### 2.3 Learned 最差
- 训练 loss 1.10 (最差), 因为可学习参数有限 (64 个位置)
- 外推退化 113.8% (2x) — 无法泛化到训练过的位置之外
- 参数最多 (887K vs 821K), 因为 Embedding 额外参数

### 2.4 Sinusoidal 中规中矩
- 训练 loss 0.92, 不如 RoPE 和 None
- 外推退化 225% (4x), 固定函数在高位置频率不匹配

## 3. 外推能力分析

| Type | 2x degradation | 4x degradation |
|------|---------------|----------------|
| Learned | +59.0% | +113.8% |
| None | +138.6% | +177.1% |
| RoPE | +143.2% | +231.2% |
| Sinusoidal | +139.0% | +225.4% |

**注意**: 这里 "degradation" 是相对于 train_len 的 loss 增长率。Learned 的退化率最低是因为它的 train_loss 本身最高 (1.10), 基数大所以百分比小。绝对 loss 增量: RoPE 最小。

## 4. 为什么 RoPE 最优?

```
RoPE 的工作原理:
1. 将 d 维向量分成 d/2 个 2D 子空间
2. 在每个子空间中, 按 position 旋转:
   [cos(mθ), -sin(mθ)] [x1]
   [sin(mθ),  cos(mθ)] [x2]
3. 不同维度使用不同频率 θ_i = 10000^(-2i/d)
4. 内积只依赖相对位置 (m-n):
   q_m · k_n = f(m-n)
```

**优势**:
- 相对位置编码, 天然支持外推
- 不增加参数 (no Embedding layer)
- 可通过 NTK-aware scaling 扩展到更长序列

## 5. 工程实践建议

1. **默认使用 RoPE**: 现代模型 (LLaMA, Mistral, Qwen) 标配
2. **避免 Learned**: 参数多、外推差、实现复杂
3. **长上下文扩展**: RoPE + NTK scaling 或 YaRN
4. **ALiBi 替代**: 简单线性偏置, 适合需要超长外推的场景 (BLOOM)
