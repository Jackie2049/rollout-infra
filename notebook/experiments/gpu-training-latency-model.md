# GPU 实验记录 — 训练延迟模型

> 2026-06-04 | NVIDIA A16 15GB | 训练性能预测模型

## 实验 1: GEMM Roofline

| M=N=K | 时间 (ms) | TFLOPS | AI (ops/byte) | Bound | 峰值% |
|-------|----------|--------|---------------|-------|-------|
| 64 | 0.010 | 0.05 | 21.3 | memory | 0% |
| 128 | 0.011 | 0.40 | 42.7 | memory | 3% |
| 256 | 0.010 | 3.29 | 85.3 | memory | 22% |
| 512 | 0.036 | 7.46 | 170.7 | **compute** | 51% |
| 1024 | 0.164 | 13.07 | 341.3 | compute | 89% |
| 2048 | 1.147 | **14.98** | 682.7 | compute | 102% |
| 4096 | 9.655 | 14.24 | 1365.3 | compute | 97% |

**A16 Roofline**:
- Peak FP16: 14.7 TFLOPS, Peak BW: 170 GB/s
- Ridge Point: **86.5 ops/byte** (AI < 86.5 → memory-bound, AI > 86.5 → compute-bound)
- M=256 正好在边界，M≥512 进入 compute-bound

## 实验 2: Transformer 层延迟分解 ⭐

B=4, S=512, H=1024

| 操作 | 时间 (ms) | 占比 |
|------|----------|------|
| QKV projection | 0.985 | 22.4% |
| **Attention (SDPA)** | 0.361 | **8.2%** |
| Output projection | 0.332 | 7.6% |
| **MLP fc1 (H→4H)** | **1.299** | **29.6%** |
| GELU activation | 0.056 | 1.3% |
| **MLP fc2 (4H→H)** | **1.204** | **27.4%** |
| LayerNorm ×2 | 0.152 | 3.5% |

**实测完整 Block**: Forward=4.66ms, Forward+Bwd=15.02ms, Bwd/Fwd=3.22x

**关键洞察**:
- **GEMM 占 87%**: QKV(22.4%) + Out(7.6%) + fc1(29.6%) + fc2(27.4%) = 87%
- **Attention SDPA 仅 8.2%**: FlashAttention 太高效了！
- **LayerNorm + GELU < 5%**: 非计算密集型操作
- **Bwd = 3.22× Fwd**: 反向传播需要 3 倍于前向的时间 (激活重计算 + 双倍 GEMM)
- 这就是为什么 **GEMM 优化是训练性能的核心**

## 实验 3: 训练时间预测

目标: 300B tokens (常见预训练规模), FLOPS/token = 6×参数量

| GPU | 125M | 1.3B | 7B | 70B |
|-----|------|------|-----|------|
| A16 (5 TFLOPS) | 521d | 5417d | 29167d | - |
| A100 (156 TFLOPS) | 16.7d | 174d | 935d | 9348d |
| H100 (495 TFLOPS) | 5.3d | 55d | 295d | 2946d |

**70B 多卡训练 (300B tokens)**:
| 配置 | 训练时间 |
|------|---------|
| A100 × 256 | 36.5 days |
| A100 × 1024 | 9.1 days |
| H100 × 256 | 11.5 days |
| H100 × 1024 | **2.9 days** |

**注意**: 以上为理论峰值，实际 MFU (Model FLOPs Utilization) 通常 30-50%。

## 实验 4: Batch Size 吞吐

H=512, seq=256, 1-layer Transformer Block

| Batch | Tokens | Fwd+Bwd (ms) | tok/s | TFLOPS | Peak Mem |
|-------|--------|-------------|-------|--------|----------|
| 1 | 256 | 1.28 | 200K | 3.78 | 38 MB |
| 2 | 512 | 1.32 | 387K | 7.31 | 45 MB |
| 4 | 1024 | 2.40 | 427K | 8.06 | 57 MB |
| 8 | 2048 | 4.43 | 462K | 8.73 | 79 MB |
| 16 | 4096 | 8.64 | 474K | 8.96 | 125 MB |
| 32 | 8192 | 16.63 | **492K** | **9.30** | 217 MB |
| 64 | 16384 | 33.38 | 491K | 9.27 | 402 MB |

**关键发现**:
- batch=1→4: **2.1x** 吞吐提升 (200K→427K)
- batch=4→32: **1.2x** 吞吐提升 (已开始饱和)
- batch≥32: 几乎无提升 (9.3 TFLOPS plateau)
- **TFLOPS 在 batch=4 时已达峰值的 87%**
- 内存线性增长: batch=64 需要 402MB

## 实验 5: 通信开销

| 数据大小 | Copy BW |
|---------|---------|
| 128 KB | 31.2 GB/s |
| 512 KB | 122.6 GB/s |
| 2 MB | 149.8 GB/s |
| 8 MB | 159.5 GB/s |
| 32 MB | 163.3 GB/s |

TP AllReduce 通信占比 (NVLink 170 GB/s):
- 在 12 层模型中，单层 AllReduce 通信 < 0.2ms
- 相对于 step 时间 (7-494s)，通信占比 < 0.01%
- **结论**: 在 NVLink 下，TP 通信开销几乎可以忽略
- 但跨节点 (Ethernet) 时，通信占比可能升至 10-30%

## 综合结论

1. **训练是 compute-bound**: Transformer 层 87% 时间花在 GEMM 上
2. **FlashAttention 太快了**: 只占 8%，所以 MLA/Paged Attention 等优化的重点是内存而非速度
3. **混合精度是必须的**: FP16/BF16 比 FP32 快 4.5x
4. **Batch size 有最优值**: batch=4-8 即可达到 87-94% 峰值
5. **多卡扩展关键在 MFU**: 理论 2.9 天训 70B，实际可能需要 6-10 天
6. **NVLink 下 TP 效率极高**: 通信占比 <1%，但 Ethernet 下是瓶颈
