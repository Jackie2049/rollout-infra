# 混合精度训练深度解析

> FP32 → FP16 → BF16 → FP8：为什么降低精度反而更快

## 1. 浮点数表示

### IEEE 754 格式

```
浮点数 = (-1)^sign × 2^exponent × 1.mantissa

sign: 符号位
exponent: 指数位（决定动态范围）
mantissa: 尾数位（决定精度）
```

### 格式对比

| 格式 | Sign | Exponent | Mantissa | 总位数 | 动态范围 | 最小正规数 |
|------|------|----------|----------|--------|----------|-----------|
| FP32 | 1 | 8 | 23 | 32 | ±3.4×10³⁸ | 1.2×10⁻³⁸ |
| FP16 | 1 | 5 | 10 | 16 | ±65504 | 6.1×10⁻⁵ |
| BF16 | 1 | 8 | 7 | 16 | ±3.4×10³⁸ (同FP32!) | 1.2×10⁻³⁸ |
| FP8 E4M3 | 1 | 4 | 3 | 8 | ±448 | 2⁻⁶ ≈ 0.0156 |
| FP8 E5M2 | 1 | 5 | 2 | 8 | ±57344 | 2⁻¹⁴ ≈ 6.1×10⁻⁵ |
| INT8 | - | - | - | 8 | [-128, 127] | 1 |

### 关键区别

```
FP16 vs BF16:
  FP16: 更多 mantissa (10位) → 更精确，但范围小（最大65504）
  BF16: 更多 exponent (8位) → 范围=FP32，但精度低

  FP16 容易 overflow (>65504 就溢出) 和 underflow (<6.1e-5 变为0)
  BF16 几乎不会 overflow/underflow，但表示精度较低

FP8 两种模式:
  E4M3: 更多精度，范围小 → 前向传播（权重、激活）
  E5M2: 更多范围，精度低 → 反向传播（梯度）
```

### 数值精度对比（π 的表示）

```
FP32: 3.1415927...  (7 位有效数字)
BF16: 3.140625      (3 位有效数字)
FP16: 3.140625      (3-4 位有效数字)
FP8 E4M3: 3.125     (2 位有效数字)
```

## 2. 为什么混合精度能加速

### Tensor Core 加速比

GPU 有两种计算单元：
- **CUDA Core**: 通用标量计算
- **Tensor Core**: 专用矩阵乘法单元

| GPU | FP32 | FP16 Tensor Core | BF16 Tensor Core | FP8 Tensor Core |
|-----|------|-----------------|-----------------|----------------|
| A100 | 19.5 TFLOPS | 312 TFLOPS (**16x**) | 312 TFLOPS | N/A |
| H100 | 67 TFLOPS | 990 TFLOPS | 990 TFLOPS | 1979 TFLOPS (**30x**) |
| H200 | 67 TFLOPS | 990 TFLOPS | 990 TFLOPS | 3958 TFLOPS |

**核心原因**：Tensor Core 一次处理矩阵块而非单个元素，低精度让寄存器/SMEM 放更多数据。

### 显存节省

```
参数 + 梯度:
  FP32: 4 + 4 = 8 bytes/param
  FP16: 2 + 2 = 4 bytes/param  (节省 50%)
  BF16: 2 + 2 = 4 bytes/param  (节省 50%)

注意: Adam 优化器状态仍需 FP32 (主参数 + m + v = 12 bytes/param)
所以混合精度训练总显存 ≈ 16Ψ (同纯FP32!) 而非 8Ψ

实际节省主要来自:
  1. 激活值减半（前向中间结果用 FP16 存储）
  2. 通信量减半（梯度以 FP16 传输）
  3. Tensor Core 加速计算
```

## 3. FP16 混合精度训练 (AMP)

### 3.1 核心机制

```
┌─────────────────────────────────────────────┐
│ FP32 主参数 (master weights)                 │  ← 始终保持 FP32
│         ↓ cast to FP16                       │
│ FP16 前向计算 (Tensor Core 加速)             │  ← 计算快 16x
│         ↓                                    │
│ FP16 Loss                                    │
│         ↓ × Loss Scale                      │  ← 放大 loss 防梯度下溢
│ FP16 反向计算                                 │  ← 计算快 16x
│         ↓ FP16 梯度                          │
│         ↓ ÷ Loss Scale + cast to FP32        │
│ FP32 梯度                                    │
│         ↓                                    │
│ FP32 参数更新 (Adam)                          │
└─────────────────────────────────────────────┘
```

### 3.2 Loss Scaling 原理

**问题**：FP16 的最小正规数是 6.1×10⁻⁵。深度网络的梯度通常远小于此值，导致 **underflow**（变为 0）。

**解决**：在前向计算完成后，将 loss 乘以一个缩放因子 S（如 S=65536）：

```python
# 无 scaling:
loss = model(inputs)              # FP16 loss
grad = autograd.backward(loss)    # FP16 grad → 可能 underflow!

# 有 scaling:
loss = model(inputs)
scaled_loss = loss * S            # 放大 loss
scaled_grad = autograd.backward(scaled_loss)
grad = scaled_grad / S            # 缩放回来，此时 grad 有足够的精度
```

**效果**：梯度值被放大 S 倍，远离 FP16 的 underflow 阈值。

### 3.3 动态 Loss Scaling

静态缩放的问题：S 太大 → 梯度 overflow → NaN；S 太小 → 仍有 underflow。

```python
# PyTorch GradScaler 的动态调整逻辑
class GradScaler:
    def __init__(self):
        self.scale = 65536.0    # 初始缩放因子
        self.growth_factor = 2.0
        self.backoff_factor = 0.5
        self.growth_interval = 2000

    def step(self):
        # 检查梯度是否有 inf/nan
        if has_inf_or_nan(grads):
            self.scale *= self.backoff_factor  # 减半
            skip_optimizer_step()              # 跳过此步更新
        else:
            optimizer.step()
            if step_count % growth_interval == 0:
                self.scale *= self.growth_factor  # 翻倍
```

### 3.4 PyTorch AMP 使用

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():              # 自动混合精度上下文
        output = model(data)      # 部分算子在 FP16，部分在 FP32
        loss = criterion(output, target)

    scaler.scale(loss).backward() # 缩放 loss → 反向传播
    scaler.step(optimizer)        # 检查梯度 → 更新参数
    scaler.update()               # 更新缩放因子
```

### 3.5 自动精度选择规则

PyTorch autocast 自动决定每个操作的精度：

| FP16/BF16 计算 | FP32 计算 |
|---------------|-----------|
| matmul, linear | Loss functions |
| conv1d/2d/3d | Softmax |
| attention (QKV) | LayerNorm |
| element-wise (+, *, relu) | Binary cross entropy |
| | Bias add |

**规则**：计算密集型（GEMM/Conv）用低精度，精度敏感型（归约、归一化）用 FP32。

## 4. BF16 训练

### 4.1 优势

```
1. 无需 Loss Scaling！
   - BF16 的动态范围 = FP32 → 梯度不会 underflow/overflow
   - 简化训练代码，消除 scaler 调参

2. 训练更稳定
   - 不受缩放因子选择影响
   - 极少出现 NaN

3. 同样享受 Tensor Core 加速 (312 TFLOPS on A100)
```

### 4.2 PyTorch BF16 训练

```python
# 方法 1: 自动混合精度
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()  # 无需 GradScaler!

# 方法 2: 模型完全 BF16
model = model.to(torch.bfloat16)

# 方法 3: DeepSpeed BF16
ds_config = {
    "bf16": {"enabled": True},
}
```

### 4.3 BF16 的代价

```
精度降低:
  FP32: 7 位有效数字
  BF16: 3 位有效数字

影响:
  - 微小的梯度更新可能被舍入掉
  - 累加操作精度下降（需要更高精度的规约）
  - 通常对训练最终精度影响很小 (<0.1%)

结论: 对于大多数 LLM 训练，BF16 是最佳选择
```

## 5. FP8 训练

### 5.1 两种 FP8 格式

```
E4M3 (前向传播):
  - 4 位指数，3 位尾数
  - 范围: [-448, 448]
  - 更精确 → 适合权重和激活

E5M2 (反向传播):
  - 5 位指数，2 位尾数
  - 范围: [-57344, 57344]
  - 更大范围 → 适合梯度（值分布更宽）
```

### 5.2 FP8 训练流程

```
┌──────────────────────────────────────────┐
│ FP32/BF16 主参数                          │
│         ↓ quantize to FP8 E4M3           │
│ FP8 前向 (Tensor Core, 2x BF16速度)      │
│         ↓                                │
│ FP8 Loss                                 │
│         ↓                                │
│ FP8 反向 (E5M2 梯度)                      │
│         ↓ dequantize to FP32/BF16        │
│ FP32/BF16 参数更新 (Adam)                 │
└──────────────────────────────────────────┘
```

### 5.3 延迟缩放 (Delayed Scaling)

FP8 量化需要确定缩放因子。常用策略：

```python
# 延迟缩放: 用前一步的统计量确定当前步的缩放因子
scale_factor = max_abs_value(history_gradients) / FP8_MAX
quantized = float_to_fp8(gradient, scale_factor)
```

### 5.4 当前状态 (2024-2025)

- **H100/H200**: 硬件支持 FP8 Tensor Core
- **Transformer Engine**: NVIDIA 库，自动管理 FP8 量化/反量化
- **Megatron-LM**: 支持 FP8 训练
- **挑战**:
  - 训练稳定性（某些模型/任务精度下降）
  - 量化噪声累积
  - 需要精细的缩放策略
  - 收敛可能需要调参

```python
# Transformer Engine FP8 使用
import transformer_engine as te

model = te.Linear(768, 768, bias=False)

with te.fp8_autocast():
    output = model(input_bf16)  # 自动 FP8 计算
```

## 6. 实践建议

### 选择决策树

```
训练 LLM?
  ├── 训练精度要求高 → BF16 (推荐)
  ├── 推理速度优先 → FP8 量化推理
  ├── 硬件不支持 BF16 (旧 GPU) → FP16 + Loss Scaling
  └── 最新硬件 (H100+) → FP8 训练 (尝鲜)

微调?
  ├── LoRA/QLoRA → BF16 或 FP16
  └── Full fine-tune → BF16

推理?
  ├── FP16/BF16 → 通用部署
  ├── INT8 (GPTQ/AWQ) → 量化推理
  └── FP8 → H100 部署
```

### 精度 vs 速度 vs 显存

| 格式 | 速度 (相对FP32) | 显存 (相对FP32) | 训练稳定性 |
|------|----------------|----------------|-----------|
| FP32 | 1x | 1x | 最高 |
| FP16 + AMP | ~3-8x | ~50-70% | 需调参 |
| BF16 | ~3-8x | ~50-70% | 高 |
| FP8 | ~6-16x | ~30-50% | 中等 (发展中) |

## 7. 学习要点

1. **混合精度的核心**：计算用低精度（快），参数更新用高精度（准）
2. **FP16 需要 Loss Scaling** — 防止梯度 underflow
3. **BF16 不需要 Loss Scaling** — 动态范围与 FP32 相同
4. **FP8 是前沿方向** — 需要 H100+ 硬件支持，稳定性仍在改进
5. **Tensor Core 是加速的根本原因** — FP16/BF16 在 Tensor Core 上快 16x
6. **显存节省主要来自激活值** — 模型参数仍需 FP32 主副本

## 参考

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR 2018)
- [A Study of BF16 for Deep Learning](https://arxiv.org/abs/1904.06950)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
