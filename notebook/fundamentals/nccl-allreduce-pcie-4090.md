# 8xRTX 4090 NCCL AllReduce Bandwidth: PCIe Ring 性能衰减

> 2026-06-07 | 8× RTX 4090 PCIe 集群 NCCL AllReduce 实测

## 核心数据

**NCCL AllReduce Peak Effective Bandwidth (256MB data):**

| GPUs | per_iter_ms | eff_bw (GB/s) | alg_bw (GB/s) | Ring steps (2×(N-1)) |
|------|-------------|----------------|----------------|----------------------|
| 2    | 35.38       | **7.59**       | 15.18          | 2                    |
| 4    | 81.05       | **3.31**       | 13.24          | 6                    |
| 8    | 89.18       | **3.01**       | 24.08          | 14                   |

**关键发现**: eff_bw 从 2GPU→4GPU 下降 **2.3x**, 从 4GPU→8GPU 仅下降 1.1x — 4GPU以上跨NUMA瓶颈已饱和。

## 数据大小 vs 带宽

| Size | 2GPU ms | 2GPU BW | 4GPU ms | 4GPU BW | 8GPU ms | 8GPU BW |
|------|---------|---------|---------|---------|---------|---------|
| 1KB  | 0.037   | 0.03    | 0.037   | 0.03    | 0.072   | 0.01    |
| 64KB | 0.036   | 1.81    | 0.044   | 1.47    | 0.072   | 0.91    |
| 256KB| 0.057   | 4.56    | 0.130   | 2.02    | 0.158   | 1.66    |
| 1MB  | 0.157   | 6.69    | 0.356   | 2.95    | 0.404   | 2.60    |
| 4MB  | 0.581   | 7.22    | 1.318   | 3.18    | 1.440   | 2.91    |
| 16MB | 2.263   | 7.42    | 5.106   | 3.29    | 5.665   | 2.96    |
| 64MB | 8.898   | 7.54    | 20.159  | 3.33    | 22.453  | 2.99    |
| 256MB| 35.381  | 7.59    | 81.048  | 3.31    | 89.176  | 3.01    |

**三阶段特性**:
1. **延迟主导** (<64KB): 36μs 固定开销, 带宽利用率 <2%
2. **过渡区** (256KB-1MB): 带宽开始发挥作用, 4-7 GB/s
3. **带宽饱和** (>4MB): 峰值带宽稳定在 3.0-7.6 GB/s

## 理论分析

### Ring AllReduce 带宽模型

Ring AllReduce: 2×(N-1) 步, 每步发送 size/(N-1) 数据

**理论时间**: T = 2 × size / BW_link (与GPU数无关!)

**实测不符**:
- 2GPU: T = 2 × 256MB / 15.2 GB/s ≈ 34ms → 实测 35.38ms ✓ (same NUMA)
- 4GPU: T = 2 × 256MB / 15.2 GB/s ≈ 34ms → 实测 81.05ms ❌ (2.4x slower!)
- 8GPU: T = 2 × 256MB / 15.2 GB/s ≈ 34ms → 实测 89.18ms ❌ (2.6x slower!)

### 为什么不符合理论?

**NUMA 跨界开销**: Ring 必须经过跨NUMA链路

GPU拓扑: GPU0-3 NUMA0, GPU4-7 NUMA1

| Ring 路径 | NUMA跨界次数 | 跨界开销 |
|-----------|-------------|---------|
| 2GPU (0↔1) | 0 | 最低 |
| 4GPU (0-1-2-3) | 0 (全NUMA0) | 中等 (更多同步) |
| 4GPU (0-1-4-5) | 2 | 高 (跨NUMA) |
| 8GPU (0-1-2-3-4-5-6-7) | 2 | 最高 |

NCCL默认ring顺序可能跨越NUMA→额外同步延迟

### 有效带宽 vs GPU数

```
2GPU: eff_bw = size / time ≈ PCIe_BW / 2 = 7.59 GB/s
4GPU: eff_bw ≈ PCIe_BW / 4.6 = 3.31 GB/s  (2.3x衰减, 跨NUMA)
8GPU: eff_bw ≈ PCIe_BW / 5.1 = 3.01 GB/s  (瓶颈饱和)
```

**衰减因子**: 2→4 GPU 衰减 2.3x (NUMA跨界), 4→8 GPU 衰减 1.1x (已饱和)

### 算法带宽 (总吞吐)

alg_bw = eff_bw × world_size:
- 2GPU: 15.18 GB/s (2×7.59)
- 4GPU: 13.24 GB/s (4×3.31)
- 8GPU: 24.08 GB/s (8×3.01)

**总吞吐随GPU数增加**, 但单节点带宽反而降低 — 这就是为什么DDP扩展效率极差!

## 与DDP实测数据验证

DDP 50M model (201MB gradient):

| 预估 | 实测 |
|------|------|
| 2GPU AllReduce: 201MB / 7.59 = 26.6ms | DDP total: 31.80ms |
| Compute: 4.91ms | 通信占比: 24/31.8 = **75%+** ✓ |

完全吻合! NCCL实测带宽直接解释了DDP的低效率。

## 与NVLink对比

| 场景 | PCIe RTX 4090 | NVLink A100 (300GB/s) | NVLink优势 |
|------|---------------|----------------------|-----------|
| 2GPU AllReduce 256MB | 35.4ms (7.59 GB/s) | ~1.7ms (150 GB/s) | **21x faster** |
| 4GPU AllReduce 256MB | 81.0ms (3.31 GB/s) | ~1.7ms (150 GB/s) | **48x faster** |
| 8GPU AllReduce 256MB | 89.2ms (3.01 GB/s) | ~1.7ms (150 GB/s) | **52x faster** |
| DDP 50M 2GPU效率 | 15% | ~95% | **6x better** |
| DDP 50M 8GPU效率 | 7% | ~90% | **13x better** |

NVLink AllReduce 不受NUMA影响 (GPU间直连, 无PCIe/CPU中转)

## 实用结论

| GPU数 | AllReduce BW | 适用场景 |
|-------|-------------|---------|
| 1 | N/A | ✅ 单GPU训练/推理最佳 |
| 2 | 7.59 GB/s | ⚠️ 小模型(<1MB) DDP勉强可行 |
| 4 | 3.31 GB/s | ❌ 实际模型DDP效率<10% |
| 8 | 3.01 GB/s | ❌ 实际模型DDP效率<10% |

**核心**: PCIe无NVLink的RTX 4090集群, AllReduce带宽随GPU数急剧衰减。
任何需要多GPU通信的操作(DDP/TP/EP), 都需要NVLink才有实际意义。

## NCCL 小数据延迟

| Size | 2GPU | 4GPU | 8GPU |
|------|------|------|------|
| 1KB  | 37μs | 37μs | 72μs |
| 4KB  | 36μs | 36μs | 72μs |
| 16KB | 36μs | 36μs | 70μs |

- 2-4GPU: ~36μs 固定延迟 (NCCL kernel launch + sync)
- 8GPU: ~72μs (翻倍, 更多GPU需要更多同步)
- 对于TP推理(每层AllReduce ~2MB): 0.15-0.4ms → 每层增加15-40%延迟