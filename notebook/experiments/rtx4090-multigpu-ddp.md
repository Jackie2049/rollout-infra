# 多 GPU 分布式训练 DDP 实测 — 8× RTX 4090
# Multi-GPU DDP Training Benchmark — 8× RTX 4090 PCIe

> 2026-06-05 | 工具: `tools/multigpu_ddp_benchmark.py` | 8× RTX 4090 24GB PCIe
> 分布式后端: NCCL, 通信: PCIe Gen4 (无 NVLink)

## 1. 实验设计

**DDP (Distributed Data Parallel)**:
- 每个 GPU 持有完整模型副本
- 每个 GPU 处理不同数据子集
- 反向传播后 AllReduce 同步梯度
- 最简单的分布式训练方法

**测试环境**: 8× RTX 4090 PCIe (无 NVLink!)
- PCIe Gen4 x16 理论带宽: 12.7 GB/s (双向)
- NVLink (A100): 300 GB/s (24x 更快)

## 2. 核心结果

### 2.1 DDP 扩展效率

| GPUs | 吞吐量 (tok/s) | 每GPU吞吐 | 加速比 | 效率 |
|------|---------------|----------|--------|------|
| 1 | 249.7K | 249.7K | 1.00x | 100% |
| 2 | 451.2K | 225.6K | **1.81x** | **90.4%** |
| 4 | 806.3K | 201.6K | **3.23x** | **80.7%** |

**分析**:
- 2 GPU 效率 90% — 非常好! PCIe 延迟 ~0.15ms/AllReduce
- 4 GPU 效率 81% — 衰减明显 (4 节点 ring 有 3 跳)
- 对比 A100 NVLink: 效率通常 95-98% (通信快 40x)

### 2.2 AllReduce 通信性能

| GPUs | 1MB | 10MB | 50MB | 100MB |
|------|-----|------|------|-------|
| 2 | 0.16ms / 6.7 GB/s | 1.42ms / 7.4 GB/s | 7.01ms / 7.5 GB/s | 13.95ms / 7.5 GB/s |
| 4 | 0.36ms / 4.4 GB/s | 3.23ms / 4.9 GB/s | 16.2ms / 4.9 GB/s | 31.8ms / 4.9 GB/s |

**发现**:
- **2 GPU AllReduce ~7.5 GB/s**: PCIe 实际带宽的 ~60%
- **4 GPU AllReduce ~4.9 GB/s**: 更低 (ring 有更多跳)
- 大消息 (>10MB) 带宽稳定, 小消息延迟主导

### 2.3 Batch Size 扩展

| BS/GPU | 1 GPU | 2 GPU | 4 GPU | 4GPU加速比 |
|--------|-------|-------|-------|-----------|
| 8 | 127K | 228K | 396K | 3.1x |
| 16 | 250K | 457K | 774K | 3.1x |
| 32 | 250K | 451K | 806K | 3.2x |
| 64 | 247K | 468K | 883K | 3.6x |
| 128 | 244K | 475K | 923K | 3.8x |

**关键发现**: **Batch 越大, 扩展效率越高!**
- BS=8: 3.1x (通信占比高)
- BS=128: 3.8x (计算时间长, 通信被隐藏)
- 大 batch 让计算/通信比提高 → AllReduce 可以和计算重叠

### 2.4 模型大小扩展 (4 GPUs)

| Model | Params | 1 GPU | 4 GPUs | 加速比 | 效率 |
|-------|--------|-------|--------|--------|------|
| small | 0.9M | 949K | 3141K | 3.31x | 82.7% |
| medium | 6.4M | 250K | 804K | 3.22x | 80.4% |
| large | 21.4M | 121K | 411K | 3.41x | 85.2% |
| xlarge | 50.6M | 67K | 229K | 3.44x | **86.0%** |

**发现**: **大模型扩展效率更高!**
- 原因: 大模型 GEMM 更大 → 计算时间长 → 通信占比下降
- 这和理论一致: 效率 ≈ 1 / (1 + comm_time / compute_time)
- 50M 模型效率 86% vs 0.9M 模型 83%

## 3. 与理论对比

```
RTX 4090 PCIe DDP:
  AllReduce 带宽: 5-7.5 GB/s
  NVLink (A100): 300 GB/s → 40-60x 差距
  扩展效率: 81-90% (4 GPU)

理论推导:
  DDP 每步通信量 = 2 × model_size × (N-1)/N
  medium (6.4M) FP16 = 12.8 MB
  通信时间 = 12.8 / 7.5 = 1.7ms (2 GPU)
  计算时间 ≈ 2.0ms (per step)
  通信占比 ≈ 1.7 / (1.7+2.0) = 46%

  但 PyTorch DDP 有 gradient bucketing → 通信和计算重叠
  → 实际通信开销被部分隐藏
```

## 4. 核心学习

1. **PCIe DDP 可行**: 90% (2 GPU) 和 81% (4 GPU) 效率 — 适合训练
2. **PCIe 不适合推理**: 之前实测 TP=4 AllReduce 仅 5-6 GB/s, 推理通信无法被隐藏
3. **大 batch + 大模型**: 最大化计算/通信比, 提高扩展效率
4. **Ring AllReduce**: 2 GPU 最优 (单跳), 更多 GPU 效率递减
5. **梯度分桶 (bucketing)**: PyTorch DDP 自动将梯度分成小桶, 流水线通信
6. **对比 A100 NVLink**: NVLink 300 GB/s → 通信 <2%, PCIe → 通信 >10%
