# 推理量化技术深度解析

> GPTQ / AWQ / SmoothQuant / FP8 — 如何在几乎不损失精度的前提下压缩模型

## 1. 为什么需要量化

### 1.1 推理瓶颈

```
FP16 7B 模型推理:
  模型大小: 14 GB (2 bytes/param)
  GPU 需要: ≥16 GB VRAM
  推理速度: 受限于显存带宽 (memory-bound)

INT4 7B 模型推理:
  模型大小: 3.5 GB (0.5 bytes/param)
  GPU 需要: ≥6 GB VRAM  ← 便宜 GPU 也能跑！
  推理速度: 带宽需求减半 → 更快
```

### 1.2 量化收益

| 指标 | FP16 | INT8 | INT4 |
|------|------|------|------|
| 模型大小 | 1x | 0.5x | 0.25x |
| 显存带宽需求 | 1x | 0.5x | 0.25x |
| 推理吞吐量 | 基准 | ~1.5-2x | ~2-3x |
| 精度损失 | 0% | ~0.1-0.5% | ~0.5-2% |

## 2. 量化基础

### 2.1 均匀量化

```
量化: x_q = round(x / scale + zero_point)
反量化: x ≈ (x_q - zero_point) * scale

其中:
  scale = (x_max - x_min) / (2^b - 1)  # b = 量化位数
  zero_point = round(-x_min / scale)
```

### 2.2 对称 vs 非对称量化

```
对称量化 (INT8):
  范围: [-127, 127]
  scale = max(|x_max|, |x_min|) / 127
  zero_point = 0
  适合: 权重 (近似对称分布)

非对称量化 (UINT8):
  范围: [0, 255]
  scale = (x_max - x_min) / 255
  zero_point = round(-x_min / scale)
  适合: 激活值 (ReLU 后非负)
```

### 2.3 量化粒度

| 粒度 | 说明 | 精度 | 效率 |
|------|------|------|------|
| Per-tensor | 整个张量一个 scale | 最低 | 最高 |
| Per-channel | 每个输出通道一个 scale | 中等 | 高 |
| Per-group | 每 128 个元素一个 scale | 最高 | 中等 |
| Per-element | 每个元素一个 scale | 最高 | 最低（不可行） |

**GPTQ 和 AWQ 都使用 per-group 量化**（group size = 128），精度和效率的最佳平衡。

## 3. GPTQ: 基于近似二阶信息的后训练量化

### 3.1 核心思想

逐列量化权重，利用 **Fisher 信息矩阵**（Hessian）的近似来最小化量化误差对输出的影响。

```
目标: 量化权重 W → W_q，使得 W_q * X ≈ W * X
即最小化 ||WX - W_q X||²

关键洞察: 不是所有权重同等重要！
Hessian H = 2X * X^T 告诉我们每个权重对输出的影响程度
```

### 3.2 算法流程

```python
# GPTQ 逐列量化（简化版）
def gptq_quantize(W, H, block_size=128, bits=4):
    """
    W: 权重矩阵 [out_features, in_features]
    H: Hessian 逆矩阵 [in_features, in_features]
    """
    W_q = W.clone()
    Q = torch.zeros_like(W)  # 量化误差累积

    for col_start in range(0, W.shape[1], block_size):
        col_end = min(col_start + block_size, W.shape[1])

        # 1. 量化当前 block
        W_block = W_q[:, col_start:col_end]
        W_block_q = quantize(W_block, bits)  # 逐列量化

        # 2. 计算量化误差
        Err_block = (W_block - W_block_q) / H[col_start:col_end, col_start:col_end].diag()

        # 3. 用 Hessian 逆补偿误差到后续列
        W_q[:, col_end:] -= Err_block @ H[col_start:col_end, col_end:]

        Q[:, col_start:col_end] = W_block_q

    return Q
```

### 3.3 关键特点

- **需要校准数据**：需要少量 (128-256 条) 样本计算 Hessian
- **一次性量化**：量化完成后不再需要校准数据
- **逐列处理**：利用误差补偿确保量化精度
- **时间复杂度**：O(d²) per column，整体 O(d³)
- **精度**：INT4 下 perplexity 仅增加 ~0.1-0.3

### 3.4 GPTQ 的局限

- **权重量化，激活不量化**（Weight-Only Quantization）
- 激活值仍然是 FP16 → 显存带宽收益有限
- 量化过程需要 Hessian 计算（内存密集）
- 对某些异常通道敏感

## 4. AWQ: 激活感知权重量化

### 4.1 核心观察

```
关键洞察: 不是所有权重通道同等重要！
重要的通道 = 权重值大 AND 对应激活值大的通道

AWQ 发现:
  1. ~1% 的通道是"显著通道" (salient channels)
  2. 这些通道的权重对量化误差极其敏感
  3. 保护这些通道就能大幅提升量化精度
```

### 4.2 算法流程

```python
def awq_quantize(W, X, bits=4, group_size=128):
    """
    W: 权重 [out, in]
    X: 激活 [in, seq_len] (校准数据)
    """
    # 1. 计算每个通道的重要性
    importance = W.abs().mean(dim=0) * X.abs().max(dim=0)  # [in]

    # 2. 找到显著通道
    salient_threshold = importance.quantile(0.99)
    salient_mask = importance > salient_threshold

    # 3. 对显著通道应用缩放因子
    # 搜索最优缩放因子 s
    best_s = grid_search_optimal_scale(W, X, salient_mask)

    # 4. 缩放权重
    W_scaled = W * best_s.unsqueeze(0)

    # 5. 量化（per-group）
    W_q = per_group_quantize(W_scaled, bits, group_size)

    return W_q, best_s
```

### 4.3 缩放原理

```
原始: Y = W * X
缩放: Y = (W * s) * (X / s)  ← 缩放权重，反缩放激活

对非显著通道: s ≈ 1 (不改变)
对显著通道: s > 1 (放大权重，缩小激活)

量化误差:
  量化后的误差 = |W - W_q| * |X|
  = |W*s - quantize(W*s)| / s * |X/s|

  当 s 增大: 权重的量化相对误差减小 (更大值的量化比例误差更小)
  代价: 激活值变小，可能导致精度损失

  AWQ 找到最优的 s 平衡两者
```

### 4.4 AWQ vs GPTQ

| 特性 | GPTQ | AWQ |
|------|------|-----|
| 方法 | Hessian-based 误差补偿 | 激活感知缩放 |
| 速度 | 量化较慢 (O(d³)) | 量化较快 |
| 精度 | INT4: 很好 | INT4: 相当或更好 |
| 推理速度 | 快 (GPU kernel 优化) | 快 (Tinychat kernel) |
| 内存开销 | 需要存储 Hessian | 只需要少量校准数据 |
| 稳定性 | 依赖 Hessian 质量 | 更鲁棒 |

## 5. SmoothQuant: 激活+权重量化

### 5.1 核心问题

```
为什么只量化权重不量化激活？
因为激活值有离群点 (outliers)！

某些通道的激活值可能比其他通道大 100x
导致激活量化精度极差
```

### 5.2 SmoothQuant 的解决

```
核心思想: 把激活的难度转移到权重上

Y = X * W^T
  = (X * s^(-1)) * (s * W^T)  ← 数学等价

平滑前: X 有 outlier, W 较均匀
  → X 难量化, W 容易量化

平滑后: X * s^(-1) 较均匀, s * W^T 较大
  → X 容易量化, W 也容易量化
  → 两者都可以 INT8 量化！

s 的选择: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
α ≈ 0.5 (平衡迁移强度)
```

### 5.3 W8A8 量化

```
SmoothQuant 实现了 W8A8 (Weight INT8 + Activation INT8):
  - 权重: INT8 per-channel 对称量化
  - 激活: INT8 per-token 对称量化
  - 推理: INT8 GEMM → 速度提升 ~2x

对比:
  W16A16 (FP16): 基准
  W4A16 (GPTQ/AWQ): 模型缩小 4x，但激活仍 FP16
  W8A8 (SmoothQuant): 模型缩小 2x，全 INT8 计算 → 推理更快
```

## 6. FP8 量化推理

### 6.1 硬件支持

```
H100/H200:
  FP8 E4M3: 3958 TFLOPS (dense)
  FP16: 990 TFLOPS
  → FP8 推理快 4x!

A100:
  不支持 FP8 Tensor Core
  FP8 只能用 CUDA Core → 无加速
```

### 6.2 FP8 推理流程

```
FP16 权重 → 量化为 FP8 E4M3
        ↓
FP8 Matrix Multiply (Tensor Core)
        ↓
FP8 结果 → 反量化为 FP16/BF16
        ↓
继续下一层
```

### 6.3 vs INT8

| 特性 | FP8 E4M3 | INT8 |
|------|----------|------|
| 动态范围 | ±448 | ±127 |
| 精度 | 3 bit mantissa | 整数 |
| 量化复杂度 | 简单 (缩放) | 复杂 (校准) |
| 硬件支持 | H100+ | 通用 GPU |
| 精度损失 | ~0.1% | ~0.3-0.5% |

## 7. 实践建议

### 7.1 选择决策

```
推理场景?
  ├── 延迟敏感，显存充足 → FP16/BF16 (最简单)
  ├── 显存不够，需要压缩 → W4A16 (GPTQ/AWQ)
  │   ├── 想要最佳精度 → AWQ
  │   └── 已有 GPTQ 工作流 → GPTQ
  ├── 有 H100，要最快推理 → FP8
  ├── 需要 W8A8 全量化 → SmoothQuant
  └── 边缘设备，极致压缩 → INT4 + 知识蒸馏
```

### 7.2 精度 vs 速度 vs 大小

```
精度排序 (高→低):
  FP16 > FP8 > INT8 (SmoothQuant) > INT4 (AWQ) > INT4 (GPTQ)

显存排序 (大→小):
  FP16 > INT8 > FP8 > INT4

速度排序 (快→慢, 假设 H100):
  FP8 > INT8 W8A8 > INT4 W4A16 > FP16

注意: INT4 W4A16 虽然模型小，但 GEMM 不是 4-bit native，
      实际加速依赖特定 kernel 实现
```

## 8. vLLM 量化支持

```python
# GPTQ
from vllm import LLM
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ", quantization="gptq")

# AWQ
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")

# FP8 (H100)
llm = LLM(model="meta-llama/Llama-2-7b", quantization="fp8")

# INT8
llm = LLM(model="meta-llama/Llama-2-7b", quantization="int8")
```

## 9. 学习要点

1. **量化本质**：用更少的位数表示数值，牺牲少量精度换取显存和速度
2. **GPTQ**：Hessian-based 逐列量化，精度高但需要校准数据
3. **AWQ**：保护显著通道，更鲁棒的 INT4 量化
4. **SmoothQuant**：把激活 outlier 迁移到权重，实现 W8A8 全量化
5. **FP8**：硬件原生支持，H100 上推理最快
6. **Per-group 量化**（group=128）是精度与效率的最佳平衡
7. **量化与 KV Cache 无关** — KV Cache 的量化是另一个独立话题

## 参考

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [vLLM Quantization Documentation](https://docs.vllm.ai/en/latest/quantization/)
