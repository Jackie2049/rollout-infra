# ZeRO 优化器：零冗余优化深度解析

> ZeRO (Zero Redundancy Optimizer) — 如何将 3x 冗余变为 1/N 显存占用

## 1. 核心问题：DP 中的显存冗余

标准数据并行 (DDP) 中，每个 GPU 都持有模型参数的**完整副本**：

```
GPU 0: [完整参数 Θ, 完整梯度 ∇Θ, 完整优化器状态]
GPU 1: [完整参数 Θ, 完整梯度 ∇Θ, 完整优化器状态]  ← 全部冗余！
...
GPU N-1: [完整参数 Θ, 完整梯度 ∇Θ, 完整优化器状态]
```

### 训练时显存占用 breakdown（以 7B 模型 + Adam 为例）

| 组件 | 计算公式 | 7B FP16 模型 |
|------|----------|-------------|
| 模型参数 | 2Ψ bytes (FP16) | 14 GB |
| 梯度 | 2Ψ bytes (FP16) | 14 GB |
| Adam 优化器状态 | 12Ψ bytes (FP32 主参数 + FP32 m + FP32 v) | 84 GB |
| 激活值 | 取决于 batch/seq/model | ~10-30 GB |
| **总计（不含激活）** | **16Ψ** | **~112 GB** |

> Ψ = 参数数量。Adam 需要 FP32 主参数副本 (4Ψ) + 一阶动量 m (4Ψ) + 二阶动量 v (4Ψ) = 12Ψ。

**关键洞察**：优化器状态占用了 75% 的非激活显存！

## 2. ZeRO 的三阶段优化

### ZeRO Stage 1：优化器状态分区

```
每个 GPU 只存储 1/N 的优化器状态
GPU i 存储: 参数 Θ (全部), 梯度 ∇Θ (全部), 优化器状态的第 i 个分片
```

**显存节省**：
- 优化器状态：12Ψ/N（从 12Ψ 降为 12Ψ/N）
- 每个GPU显存：2Ψ + 2Ψ + 12Ψ/N = 4Ψ + 12Ψ/N

**通信**：与 DDP 相同（1次 AllReduce），**无额外通信开销**。

**7B 模型示例** (N=64 GPU):
- Stage 0 (DDP): 112 GB/GPU
- Stage 1: 4×14 + 84/64 = 56 + 1.3 = **57.3 GB/GPU**

### ZeRO Stage 2：梯度分区

```
在 Stage 1 基础上，每个 GPU 也只存储 1/N 的梯度
反向传播时: ReduceScatter 而非 AllReduce
```

**显存节省**：
- 优化器状态：12Ψ/N
- 梯度：2Ψ/N（从 2Ψ 降为 2Ψ/N）
- 每个GPU显存：2Ψ + 2Ψ/N + 12Ψ/N = 2Ψ + 14Ψ/N

**通信**：用 ReduceScatter 替代 AllReduce，通信量不变，但分布方式不同。

**7B 模型示例** (N=64):
- Stage 2: 14×2 + 14×14/64 = 28 + 3.06 = **31.06 GB/GPU**

### ZeRO Stage 3：参数分区

```
在 Stage 2 基础上，参数也分区！
前向/反向时按需 AllGather 参数
```

**显存节省**：
- 每个GPU显存：(2Ψ + 2Ψ + 12Ψ)/N = 16Ψ/N

**通信**：
- 前向：每层 AllGather 参数 → 额外通信
- 反向：每层 AllGather 参数 + ReduceScatter 梯度 → 额外通信
- **总通信量 ≈ DDP 的 1.5x**

**7B 模型示例** (N=64):
- Stage 3: 16×14/64 = **3.5 GB/GPU** ← 理论上！

## 3. 显存节省公式总结

| Stage | 参数 | 梯度 | 优化器状态 | 总显存 (不含激活) | 7B/N=64 |
|-------|------|------|-----------|-----------------|---------|
| DDP | 2Ψ | 2Ψ | 12Ψ | 16Ψ | 112 GB |
| ZeRO-1 | 2Ψ | 2Ψ | 12Ψ/N | 4Ψ + 12Ψ/N | 57.3 GB |
| ZeRO-2 | 2Ψ | 2Ψ/N | 12Ψ/N | 2Ψ + 14Ψ/N | 31.1 GB |
| ZeRO-3 | 2Ψ/N | 2Ψ/N | 12Ψ/N | 16Ψ/N | 3.5 GB |

> 公式基于 FP16 参数 + FP32 Adam。Ψ = 参数量（元素数）。

## 4. 通信量对比

```
DDP:     反向 1x AllReduce(Ψ)
ZeRO-1:  反向 1x AllReduce(Ψ)                    ← 同 DDP
ZeRO-2:  反向 1x ReduceScatter(Ψ)                 ← 同通信量
ZeRO-3:  前向 1x AllGather(Ψ) per layer
         反向 1x AllGather(Ψ) + 1x ReduceScatter(Ψ) per layer
         ← 约 1.5x DDP 通信量
```

### AllReduce = ReduceScatter + AllGather

```
AllReduce 通信量 = 2 × (N-1)/N × Ψ × sizeof(element)
ReduceScatter 通信量 = (N-1)/N × Ψ × sizeof(element)
AllGather 通信量 = (N-1)/N × Ψ × sizeof(element)

所以 AllReduce = ReduceScatter + AllGather（通信量相同）
```

**ZeRO-3 的额外开销**：参数按需 AllGather，每层多一次通信。假设模型有 L 层，则：
- 前向额外：L × AllGather
- 反向额外：L × (AllGather + ReduceScatter)

但可以通过**预取 (prefetch)** 重叠通信与计算。

## 5. ZeRO-Infinity：Offload 到 CPU/NVMe

当 GPU 显存不够时，将数据 offload 到 CPU 内存或 NVMe SSD：

```
GPU: 当前正在计算的层参数
CPU 内存: 优化器状态、不活跃层的参数
NVMe: 极大规模模型时，CPU 内存也不够用
```

### Offload 策略

```
ZeRO-Offload (Stage 2 + CPU offload):
  GPU: 前向/反向计算
  CPU: 优化器更新 (参数、梯度、m、v 全在 CPU)

ZeRO-Infinity (Stage 3 + CPU/NVMe offload):
  GPU: 当前活跃层计算
  CPU: 参数/优化器状态缓存
  NVMe: 完整参数/优化器状态存储
```

### 带宽瓶颈

| 路径 | 带宽 | 延迟 |
|------|------|------|
| GPU HBM → GPU HBM | ~2 TB/s (A100) | ~ns |
| CPU DDR → GPU (PCIe 4.0) | ~64 GB/s (16x) | ~μs |
| NVMe → CPU | ~7 GB/s (PCIe 4.0 SSD) | ~μs |

**关键**：Offload 后计算变为 bandwidth-bound，速度大幅下降。仅在"否则无法训练"时使用。

## 6. ZeRO vs TP vs PP：何时用哪个

### 对比表

| 方案 | 显存节省 | 通信开销 | 计算效率 | 适用场景 |
|------|---------|---------|---------|---------|
| DDP | 无 | 1x AllReduce | 最高 | 模型能放下单 GPU |
| ZeRO-1/2 | 线性 (1/N) | 同 DDP | ~95%+ | 中等模型，多 GPU |
| ZeRO-3 | 线性 (1/N) | 1.5x DDP | ~70-80% | 大模型，参数放不下 |
| TP | 线性 (1/N) | 每层 2x AllReduce | ~85-90% | 单节点内，NVLink |
| PP | 线性 (1/N) | 点对点 | ~80-90% | 跨节点，高延迟 |

### 组合策略

```
小模型 (<7B, 单 GPU 放得下):
  → DDP (最简单，最高效)

中等模型 (7B-70B):
  → ZeRO-2 或 TP (单节点内用 TP，多节点用 ZeRO)

大模型 (70B-175B):
  → TP + PP (3D 并行)
  或 TP + ZeRO-3 (如 Megatron + ZeRO)

超大模型 (175B+):
  → TP + PP + ZeRO-3 + CPU offload
  或 TP + EP (Expert Parallelism，MoE)
```

### Megatron + ZeRO 组合

```
Megatron-LM 的推荐组合:
  TP: 单节点内 (NVLink 高带宽)
  PP: 跨节点 (点对点通信，容忍高延迟)
  DP + ZeRO-1: 在 TP×PP 组之外 (如果有多个 replica)
```

## 7. DeepSpeed 使用示例

```python
import deepspeed

# ZeRO-2 配置
ds_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,           # ZeRO Stage 2
        "offload_optimizer": {
            "device": "cpu",  # 优化器状态 offload 到 CPU
        },
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": True},
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

## 8. PyTorch FSDP vs DeepSpeed ZeRO

| 特性 | FSDP | DeepSpeed ZeRO |
|------|------|----------------|
| Stage 3 | 默认行为 | Stage 3 |
| Stage 2 | use_orig_params=True | Stage 2 |
| Offload | CPU offload 支持 | CPU + NVMe offload |
| 生态 | PyTorch 原生 | 需要安装 DeepSpeed |
| 社区 | Meta, PyTorch 团队 | Microsoft |
| 混合精度 | 原生支持 | 原生支持 |

FSDP 本质上实现了 ZeRO-3 的功能，是 PyTorch 原生的方案。

## 9. 学习要点

1. **优化器状态是显存大头** — Adam 的 12Ψ bytes（FP32 主参数 + m + v）占 75%
2. **ZeRO 的本质是消除数据并行的冗余** — 把完整副本变成分片
3. **通信量不增加**（Stage 1/2）或增加有限（Stage 3 约 1.5x）
4. **ZeRO-3 的 AllGather 是按需的** — 可以预取重叠
5. **ZeRO 和 TP/PP 正交** — 可以组合使用
6. **Offload 是最后手段** — 带宽瓶颈使训练变慢 3-10x

## 参考

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (SC 2020)
- [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [DeepSpeed Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
