# FP8 量化与训练 — 8-bit 浮点格式深度解析

> FP8 是当前 LLM 训练和推理量化的前沿标准，由 NVIDIA/AMD/Intel/Arm 联合制定

## 1. FP8 格式定义

### 1.1 两种编码

```
E4M3 格式 (Forward Pass):
  ┌─────┬──────────┬───────────┐
  │ S   │ E[4位]   │ M[3位]    │
  │ 1bit│ 4 bits   │ 3 bits    │
  └─────┴──────────┴───────────┘
  范围: 2^-9 ~ 2^9 (±448)
  精度: 3 bit mantissa = 8 级
  不支持 Inf (E4M3 没有 Inf 编码)

E5M2 格式 (Backward Pass):
  ┌─────┬──────────┬───────────┐
  │ S   │ E[5位]   │ M[2位]    │
  │ 1bit│ 5 bits   │ 2 bits    │
  └─────┴──────────┴───────────┘
  范围: 2^-14 ~ 2^16 (±57344)
  精度: 2 bit mantissa = 4 级
  支持 Inf 和 NaN (IEEE 754 语义)
```

### 1.2 为什么需要两种格式

```
Forward Pass (权重 + 激活):
  - 数值范围适中: 大部分在 [-10, 10]
  - 需要精度: 3 bit mantissa 保证乘法精度
  → E4M3: 牺牲范围，换取精度

Backward Pass (梯度):
  - 数值范围大: 链式法则导致梯度分布更宽
  - 可以容忍精度: 梯度本身就是近似值
  → E5M2: 牺牲精度，换取动态范围

混合策略 (HYBRID):
  Forward:  E4M3 权重 × E4M3 激活 → E4M3 输出
  Backward: E5M2 梯度 → E4M3 权重梯度 + E5M2 输入梯度
```

### 1.3 与其他格式对比

| 格式 | 位数 | 范围 | 精度 | 内存 | TFLOPS (H100) |
|------|------|------|------|------|---------------|
| FP32 | 32 | ±3.4e38 | 24 bit | 4B | 67 |
| FP16 | 16 | ±65504 | 10 bit | 2B | 1979 |
| BF16 | 16 | ±3.4e38 | 7 bit | 2B | 1979 |
| FP8 E4M3 | 8 | ±448 | 3 bit | 1B | 3958 |
| FP8 E5M2 | 8 | ±57344 | 2 bit | 1B | 3958 |
| INT8 | 8 | ±128 | 均匀 | 1B | 3958 |
| FP4 E2M1 | 4 | ±6 | 1 bit | 0.5B | ~7916 |

关键: H100 FP8 Tensor Core 吞吐量 = 2x FP16/BF16

## 2. 缩放因子策略

### 2.1 为什么需要缩放因子

```
FP8 只有 8 bit 表示，动态范围有限。

问题: 如果权重矩阵的值分布在 [0.001, 0.01]，
但 FP8 E4M3 的精度在 [0.00390625, 0.0078125] 之间只有 8 个离散值
→ 直接表示精度不够

解决: 缩放因子 (scale factor)
  scale = fp8_max / amax    (amax = max|x|)
  x_scaled = x * scale      (放大到 FP8 范围)
  x_fp8 = quantize(x_scaled)
  x_dequant = x_fp8 / scale (恢复时除以 scale)

实际中的缩放因子策略决定了 FP8 的精度表现。
```

### 2.2 常见策略

```
1. Per-tensor scaling (最简单):
   整个张量共用一个 scale
   scale = fp8_max / max(|tensor|)
   优点: 硬件友好，只需一个 scale 参数
   缺点: outlier 影响所有值

2. Per-channel scaling (平衡):
   每行/列一个 scale
   scale[i] = fp8_max / max(|row_i|)
   优点: 精度更好
   缺点: 需要更多 scale 参数

3. Per-block scaling (DeepSeek-V3):
   每 128×128 block 一个 scale
   精度最好，但 scale 参数量大

4. Delayed scaling (Transformer Engine):
   用上一步的 amax 计算当前步的 scale
   零额外计算开销
   一步延迟，对突变不敏感

5. E8M0 scaling (Blackwell):
   Scale 本身也是 FP8 格式 (仅指数位)
   强制 scale 为 2 的幂次
   硬件最友好，DeepGEMM 使用
```

## 3. FP8 训练

### 3.1 混合精度训练流程

```
                    FP32 Master Weights
                          │
                    ┌─────┴─────┐
                    │ Optimizer  │ (FP32)
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │ Cast to   │
                    │ FP8 (E4M3)│
                    └─────┬─────┘
                          │
  ┌───────────────────────┼──────────────────────┐
  │ Forward Pass (E4M3)   │                      │
  │                       │                      │
  │  FP8 Weight × FP8 Act → FP32/BF16 Output    │
  │                       │                      │
  │  注意: LayerNorm, Softmax 保持在 FP32/BF16    │
  └───────────────────────┼──────────────────────┘
                          │ Loss
                          │
  ┌───────────────────────┼──────────────────────┐
  │ Backward Pass         │                      │
  │                       │                      │
  │  E5M2 Gradient × FP8  → FP32 Weight Grad    │
  │  E5M2 Gradient        → E5M2 Input Grad     │
  │                       │                      │
  │  梯度 AllReduce: 可以在 FP8 精度下通信        │
  └───────────────────────┼──────────────────────┘
                          │
                    ┌─────┴─────┐
                    │ Update    │
                    │ FP32 Master│
                    └───────────┘
```

### 3.2 NVIDIA Transformer Engine

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# 配置 FP8 recipe
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    fp8_format=recipe.Format.HYBRID,  # E4M3 forward + E5M2 backward
)

# 使用 autocast
model = te.Linear(768, 768, bias=False)

for data in dataloader:
    with te.autocast(enabled=True, recipe=fp8_recipe):
        output = model(data)
    loss = compute_loss(output)
    loss.backward()
    optimizer.step()

# 内部工作:
# 1. 权重从 FP32 → FP8 (E4M3)，记录 amax
# 2. 激活从 BF16 → FP8 (E4M3)，记录 amax
# 3. FP8 × FP8 GEMM → FP32 累加 → BF16 输出
# 4. 反向传播梯度在 E5M2
# 5. 下一次迭代使用本次记录的 amax 计算 scale
```

### 3.3 FP8-LM 三级优化

```
微软 FP8-LM 论文 (arXiv 2310.18313) 提出:

Level 1: FP8 GEMM only (基础)
  权重 + 激活在 FP8，其余 FP32/BF16
  加速: ~37%

Level 2: FP8 gradients + all-reduce (进阶)
  梯度也用 FP8，减少通信量
  加速: ~50%

Level 3: FP8 optimizer states (极致)
  优化器状态也量化到 FP8
  内存节省: 39%
  总加速: 75% (vs BF16 Megatron-LM)

关键数据 (GPT-175B, H100):
  BF16: 1.0x 吞吐, 1.0x 显存
  FP8-LM: 1.75x 吞吐, 0.61x 显存
```

## 4. FP8 推理

### 4.1 vLLM 的 FP8 实现

```
vLLM 支持两种 FP8 推理模式:

1. 动态量化 (最简单):
   python -m vllm.entrypoints.openai.api_server \
     --model model-name --quantization fp8

   权重 scale: 加载时计算 (一次性)
   激活 scale: 每次 forward 计算 (per-token)
   无需校准数据

2. 静态量化 (llm-compressor):
   先用校准数据量化，保存 FP8 checkpoint
   推理时直接加载

3. Block-wise (DeepSeek-V3 style):
   每 128×128 block 一个 scale
   精度最高，使用 DeepGEMM 或 CUTLASS kernel

Kernel 后端自动选择:
  CUTLASS     → FP8 GEMM (SM 89+)
  Marlin      → W8A16 (SM 80+，无 FP8 硬件时)
  DeepGEMM    → Block-wise (Hopper)
  torch._scaled_mm → PyTorch 原生
  Triton      → Fallback
```

### 4.2 vLLM 源码分析

```
关键源码路径:
  vllm/model_executor/layers/quantization/fp8.py
    → Fp8Config, Fp8LinearMethod, Fp8MoEMethod

  vllm/model_executor/kernels/linear/scaled_mm/
    → cutlass.py: CUTLASS FP8 kernel
    → torch.py: torch._scaled_mm wrapper
    → marlin.py: W8A16 Marlin kernel

核心量化函数:
  ops.scaled_fp8_quant(x, scale)
    → x * scale → clamp → cast to float8_e4m3fn

  torch._scaled_mm(A, B, scale_a, scale_b, out_dtype)
    → A(FP8) × B(FP8) × scale_a × scale_b → BF16/FP16

  注意: torch._scaled_mm 在 batch > 16 时更高效
  vLLM 会自动 pad 到 17 tokens 优化性能
```

### 4.3 FP8 KV Cache

```
vLLM 也支持 FP8 KV Cache:
  - K/V 从 BF16 量化到 E4M3
  - 每 tensor 一个 scale (k_scale, v_scale)
  - 推理时反量化后用于 attention
  - 显存节省: 2x (vs BF16 KV Cache)
  - 对长上下文推理特别重要
```

## 5. 精度影响

### 5.1 模拟实验数据

```
FP8 模拟实验结果 (tools/fp8_simulation.py):

矩阵乘法精度 (1024×1024):
  FP8 全量化:     余弦相似度 0.999224
  FP8 仅权重量化: 余弦相似度 0.999613
  → 精度可接受

Outlier 影响:
  正常数据:      FP8 误差 0.019, INT8 误差 0.006
  含 outlier:    FP8 误差 0.022, INT8 误差 0.291
  → INT8 正常值误差增加 15.2x！FP8 仅 1.1x
  → FP8 浮点格式对 outlier 天然鲁棒

12 层误差传播:
  Layer 1: 3.8% → Layer 12: 4.1%
  → 误差缓慢累积，不会爆炸
```

### 5.2 实际模型精度

```
Transformer Engine 官方数据:
  - FP8 vs BF16 训练 loss 曲线几乎重合
  - 在 175B 参数规模验证

FP8 推理:
  - W8A8 动态量化: benchmark 精度下降 0.1-0.3%
  - lm_head (最终投影层) 通常排除在量化之外 (敏感)
  - Per-token 激活量化优于 per-tensor

关键规则:
  - LayerNorm/Softmax 保持 FP32/BF16
  - lm_head 通常不量化
  - 注意 BOS token 对量化模型评估的影响
```

## 6. 硬件支持

```
                    FP8 Tensor Core    内存带宽    适用场景
──────────────────────────────────────────────────────────────
NVIDIA Hopper H100      ✅ 3958 TFLOPS  3350 GB/s  FP8 训练+推理
NVIDIA Ada L40S         ✅ (较少 SM)     864 GB/s   FP8 推理
NVIDIA Blackwell B200   ✅ 2x Hopper    8000 GB/s  FP4+FP8
AMD MI300X              ✅ (FNUZ格式)   5300 GB/s  FP8 推理
NVIDIA Ampere A100      ❌              2039 GB/s  W8A16 (Marlin)
NVIDIA Turing/Volta     ❌                        不支持 FP8

Compute Capability 要求:
  FP8 GEMM:    SM >= 89 (Ada Lovelace)
  torch._scaled_mm: SM >= 89
  W8A16 Marlin: SM >= 80 (Ampere)
  DeepGEMM Block-wise: SM 90 (Hopper)

Blackwell 新特性:
  - MXFP8: 32 值/block, E8M0 scale
  - NVFP4: E2M1 格式 (2x FP8 压缩)
  - 2代 Transformer Engine: FP8 吞吐翻倍
```

## 7. 实践指南

### 7.1 什么时候用 FP8

```
适合 FP8:
  ✅ 大模型推理 (7B+): 减少显存，提高吞吐
  ✅ 大模型训练 (70B+): 减少显存，加速训练
  ✅ 有 H100/L40S 硬件: 可以获得 FP8 加速
  ✅ 长上下文: FP8 KV Cache 减少显存

不适合 FP8:
  ❌ 小模型 (<1B): 量化开销 > 收益
  ❌ 没有 FP8 硬件: 只能用 W8A16，加速有限
  ❌ 精度敏感任务: 某些数值计算密集的任务
  ❌ 激活 outlier 严重: 需要 block-wise 量化
```

### 7.2 最佳实践

```
训练:
  1. 使用 Transformer Engine 的 DelayedScaling
  2. LayerNorm/Softmax 保持高精度
  3. Master weights 保持 FP32
  4. 监控 amax history，确保 scale 稳定
  5. 前几步可以 warmup with BF16

推理:
  1. 动态量化最简单: --quantization fp8
  2. lm_head 排除量化
  3. 如果精度不够，尝试 block-wise
  4. 如果没有 FP8 硬件，用 W8A16 Marlin
  5. FP8 KV Cache 对长上下文有帮助
```

## 参考

- [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433) — E4M3/E5M2 格式标准
- [FP8-LM: Training FP8 Large Language Models (arXiv 2310.18313)](https://arxiv.org/abs/2310.18313) — 三级 FP8 训练优化
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — 生产级 FP8 训练库
- [vLLM FP8 Documentation](https://docs.vllm.ai/en/latest/features/quantization/fp8.html)
- [vLLM FP8 Source](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization)
