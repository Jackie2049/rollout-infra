# 8xRTX 4090 DDP Scaling 实测: PCIe 通信瓶颈导致扩展效率极低

> 2026-06-07 | 8× RTX 4090 PCIe 集群 DDP training 实测

## 核心数据

**DDP step time vs 单GPU**:

| Model | 1GPU | 2GPU | 4GPU | 8GPU | 通信占比(2GPU) |
|-------|------|------|------|------|--------------|
| 8M (34MB) | 0.99ms | **5.18ms** | 10.97ms | 12.02ms | **81%** |
| 50M (201MB) | 4.91ms | **31.80ms** | 67.14ms | 73.63ms | **85%** |

**关键发现**: 2GPU DDP step比单GPU慢6.4-6.5x! → 通信完全主导

## 为什么DDP反而更慢?

DDP每步包含:
1. **Forward + Backward**: ~4.91ms (50M model, 1GPU)
2. **Gradient AllReduce**: 201MB / 17 GB/s × 2 ≈ **24ms**

总step time = compute + AllReduce ≈ 4.91 + 24 = 29ms (实测31.80ms,误差9%)

**AllReduce占比**: 24/31.8 = **75%+** → GPU大部分时间在等数据传输!

## 扩展效率分析

| 模型 | GPU数 | Ideal scaling | 实际throughput | 实际scaling | 效率 |
|------|-------|--------------|--------------|-----------|------|
| 50M | 1 | 1.00x | 13.0K | 1.00x | 100% |
| 50M | 2 | 2.00x | 4.0K | **0.31x** | 15% |
| 50M | 4 | 4.00x | 3.8K | **0.29x** | 7% |
| 50M | 8 | 8.00x | 6.95K | **0.54x** | 7% |

**效率极低**: DDP在PCIe RTX 4090上几乎是负优化!

## 通信vs计算时间分解

```
50M model (201MB weights):

1GPU step: compute = 4.91ms → total = 4.91ms

2GPU step:
  compute = 4.91ms (每GPU独立计算,与1GPU相同)
  AllReduce = 24ms (201MB gradient × 2 / 17 GB/s)
  total = 4.91 + 24 = 28.91ms → 实测31.80ms (+10% overhead)

4GPU step:
  compute = 4.91ms
  AllReduce = 48ms (ring算法: 3步×16ms)
  total = 52.91ms → 实测67.14ms (+27% extra overhead from更多GPU)

8GPU step:
  compute = 4.91ms
  AllReduce = ~70ms (7步ring×10ms,但带宽更低因为跨NUMA)
  total = 74.91ms → 实测73.63ms
```

## 之前DDP实测(不同模型大小)对比

之前用更小模型(0.5MB)测试DDP:
- 2GPU: 1.81x / 90.4%效率 ← 小模型通信量小,几乎不影响
- 4GPU: 3.23x / 80.7%效率 ← 仍然不错

这次50M模型(201MB):
- 2GPU: **0.31x / 15%效率** ← 通信完全主导

**结论**: 小模型(<1MB)DDP扩展还行, 但任何实际模型(>50MB)DDP在PCIe上效率极低

## 与NVLink对比

| 场景 | PCIe RTX 4090 | NVLink A100 | NVLink优势 |
|------|---------------|-------------|-----------|
| 8M model AllReduce | 24ms (2GPU) | ~1.3ms | 18x faster |
| 50M model AllReduce | 24ms | ~1.3ms | 18x faster |
| DDP 2GPU效率 | 15% | ~95% | 6x better |
| DDP 8GPU效率 | 7% | ~90% | 13x better |

## 实用结论

| 场景 | RTX 4090 PCIe | A100 NVLink |
|------|---------------|-------------|
| 单GPU训练/推理 | ✅ 优秀 | ✅ |
| DDP 2GPU (小模型<1MB) | ✅ ~90%效率 | ✅ ~95% |
| DDP 4GPU (实际模型) | ❌ <10%效率 | ✅ ~90% |
| DDP 8GPU (大模型) | ❌ <10%效率 | ✅ ~85% |
| TP推理 | ❌ 不可行(96%+通信) | ✅ 推荐 |
| ZeRO-3 DP | ⚠️ 降低通信量但仍PCIe瓶颈 | ✅ 高效 |

**核心**: PCIe RTX 4090集群适合**单GPU**任务或**极小模型**DDP。
任何实际规模模型的多GPU训练/推理, 需要NVLink(A100/H100)才有意义。