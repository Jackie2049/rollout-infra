# RTX 4090 Tensor Parallelism 通信基准测试

> 2026-06-05 | 4x RTX 4090, NCCL, PCIe (无 NVLink)
> 脚本: `tools/tp_benchmark_4090.py`

## 背景

RTX 4090 是消费级 GPU, **不支持 NVLink**, GPU 间通信走 PCIe。这是与数据中心 GPU (A100/H100) 的关键区别。

## 通信带宽测试 (TP=4)

| 操作 | 数据大小 | 延迟 (ms) | 带宽 (GB/s) |
|------|----------|-----------|-------------|
| AllReduce | 1024² (2 MB) | 0.67 | 4.7 |
| AllReduce | 4096² (32 MB) | 10.12 | 5.0 |
| AllReduce | 16384² (512 MB) | 160.53 | 5.0 |
| AllGather | 1024² (2 MB) | 0.37 | 5.7 |
| AllGather | 4096² (32 MB) | 5.50 | 6.1 |
| AllGather | 16384² (512 MB) | 87.75 | 6.1 |
| ReduceScatter | 1024² (2 MB) | 0.37 | 5.7 |
| ReduceScatter | 4096² (32 MB) | 5.70 | 5.9 |
| ReduceScatter | 16384² (512 MB) | 90.68 | 5.9 |

## TP MLP 模拟

- **配置**: B=16, H=4096, TP=4 (Column+Row Parallel)
- **延迟**: 0.108 ms
- **TFLOPS**: 9.9

## 对比分析

### 通信带宽对比

| 平台 | 互连 | AllReduce BW | 来源 |
|------|------|-------------|------|
| **RTX 4090** | **PCIe 4.0** | **~5 GB/s** | **本次实测** |
| A100 | NVLink 3.0 | ~300 GB/s | 文献 |
| H100 | NVLink 4.0 | ~900 GB/s | 文献 |
| A100 | PCIe 4.0 | ~25 GB/s | 文献 |

### 关键发现

1. **RTX 4090 PCIe BW 仅 ~5-6 GB/s**: 远低于理论 PCIe 4.0 x16 的 ~25 GB/s
   - 原因: 4 GPU 共享 PCIe 带宽 + 可能的 PCIe switch 拓扑
   - Ring AllReduce: 每步需 GPU→CPU→GPU (跨 NUMA), 损失大

2. **NVLink vs PCIe 差距巨大**: A100 NVLink 是 4090 PCIe 的 **60x**
   - NVLink: GPU 直连, 无需经过 CPU/PCIe switch
   - PCIe: 数据需经过 CPU, 多次 DMA 转发

3. **TP 扩展效率受限**:
   - 假设 7B 模型 decode: HBM 读 ~14 GB (模型), 通信 ~0.5 GB (hidden)
   - Decode 延迟: 计算 ~1.4 ms (14 GB / 890 GB/s) + 通信 ~100 ms (0.5 GB / 5 GB/s)
   - 通信占比: **>99%!** — 完全不可行

4. **结论**: RTX 4090 多卡 TP **不适合推理**
   - 仅适合计算密集场景 (大 batch training, 通信占比 <5%)
   - 推理应使用单卡 + Continuous Batching

### 模拟: 7B 模型推理 TP 分析

```
7B Decode (FP16, B=32):
  - 计算: 2 × 7B × 32 / 890 GB/s ≈ 0.5 ms
  - 通信 (TP=2 AllReduce): 4096 × 32 × 2B / 5 GB/s ≈ 0.05 ms
  - 通信占比: ~10%

7B Decode (FP16, B=1):
  - 计算: 2 × 7B × 1 / 890 GB/s ≈ 0.016 ms
  - 通信 (TP=2 AllReduce): 4096 × 1 × 2B / 5 GB/s ≈ 0.0016 ms
  - 通信占比: ~10%
  - 但: 单卡内存够 (14GB), TP 无必要

70B Decode (FP16, B=1):
  - 单卡内存不够 (140 GB), 必须 TP
  - TP=8: 每卡 17.5 GB, 可以放进 24 GB
  - 计算: 2 × 70B × 1 / 890 GB/s ≈ 0.16 ms
  - 通信 (TP=8 AllReduce): ~2 ms
  - 通信占比: ~93% — 极低效!
  - 对比 A100 NVLink: 通信 <0.1 ms, 占比 <40%
```

## 教训

1. **消费级 GPU 多卡只适合 training** (大 batch, 计算主导)
2. **推理多卡 TP 需要 NVLink** (否则通信 >90%)
3. **单卡推理 + Continuous Batching** 是消费级 GPU 的正确策略
4. **大模型推理 (>24GB)** 需要量化 (FP8/INT4) 或 offload, 不是 TP
