# GPU Profiling 实验 — NVIDIA A16

> PyTorch 原生 GPU profiling 实验，测量 GEMM、内存带宽、Attention、kernel launch overhead

## 环境配置

```
GPU:        NVIDIA A16 (Ampere, SM 8.6)
SM 数量:    10
显存:       15.6 GB
L2 Cache:   2 MB
Warp Size:  32
CUDA:       11.8
PyTorch:    2.5.1+cu118
```

## 关键实验结果

### 1. GEMM 吞吐量

| 数据类型 | 矩阵大小 | 耗时 (ms) | 吞吐 (TFLOPS) |
|----------|----------|----------|---------------|
| FP16 | 1024x1024 | 0.435 | 4.93 |
| FP16 | 4096x4096 | 21.05 | 6.53 |
| FP16 | **8192x8192** | **98.17** | **11.20** |
| FP32 | 1024x1024 | 0.912 | 2.35 |
| FP32 | 4096x4096 | 53.60 | 2.56 |

**关键发现**:
- FP16 峰值 ~11.2 TFLOPS（大矩阵），FP32 峰值 ~2.56 TFLOPS → 混合精度加速 **4.4x**
- 小矩阵 (1024) 效率低：仅 4.93 TFLOPS，说明 **kernel launch overhead 占主导**
- A16 理论 FP16 Tensor Core 峰值约 70 TFLOPS (with Tensor Core)，实际只有 ~16% 利用率
  - A16 是 GA100 的低端版，SM 只有 10 个，不是满血 Ampere

### 2. 内存带宽

| 数据量 | 耗时 (ms) | 带宽 (GB/s) |
|--------|----------|-------------|
| 1 MB | 0.008 | 242.6* |
| 10 MB | 0.131 | 152.3 |
| 100 MB | 1.281 | 156.2 |
| 500 MB | 6.378 | 156.8 |
| 1000 MB | 12.753 | **156.8** |

*小数据量带宽虚高，因为缓存效应（L2 cache 2MB 可缓存 1MB 数据）

**关键发现**:
- 实际稳定带宽 ~**157 GB/s**（>= 10MB 数据量）
- A16 理论 HBM 带宽 ~300 GB/s (320-bit bus, ~1.6 Gbps)，实测约 52% 利用率
- clone 操作 = 读取 + 写入 = 2x 数据量，所以实际读取带宽 ~78.4 GB/s

### 3. Kernel Launch Overhead

| 数据量 | 耗时 (us) | 等效带宽 (GB/s) |
|--------|----------|----------------|
| 1 element | **9.2** | — |
| 1K | 8.6 | 1.4 |
| 4K | 8.4 | 5.9 |
| 16K | 8.4 | 23.4 |
| 64K | 8.5 | 92.2 |
| 256K | 8.3 | 380* |
| 1M | 52.2 | 241 |
| 4M | 202.4 | 249 |
| 16M | 806.8 | 250 |

*256K 处带宽虚高，可能是 L2 cache 命中

**关键发现**:
- **Kernel launch 基础开销 ~8-9 us**（与数据量无关）
- < 64K 元素（256KB）时，kernel launch overhead 占总时间的 >90%
- 需要至少 ~256K 元素（1MB）才能有效利用 GPU
- **这是 CUDA Graph 的核心动机**：推理 decode 阶段每步操作小，kernel launch 成本占比高

### 4. Attention 性能 (SDPA, FP16)

| 序列长度 | 耗时 (ms) | 吞吐 (TFLOPS) |
|---------|----------|---------------|
| 512 | 0.661 | 13.00 |
| 1024 | 2.557 | 13.44 |
| 2048 | 9.918 | **13.86** |
| 4096 | 38.864 | **14.15** |
| 8192 | 157.694 | 13.94 |

配置: B=4, H=32, D=64

**关键发现**:
- SDPA (Flash Attention) 在 A16 上稳定 ~**14 TFLOPS**
- 比 GEMM 的 11.2 TFLOPS 高 ~25%，因为 attention 是 memory-bound 操作，Flash Attention 的 tiling 优化了内存访问
- 序列长度翻倍，耗时 ~4x（O(n²) 复杂度），验证了 attention 的二次复杂度

### 5. Reduction 操作

| 数据量 | 耗时 (ms) | 带宽 (GB/s) |
|--------|----------|------------|
| 2^20 (4MB) | 0.031 | 136.0 |
| 2^22 (16MB) | 0.117 | 143.5 |
| 2^24 (64MB) | 0.389 | **172.6** |

**关键发现**:
- Sum reduction 带宽 ~172 GB/s（接近内存带宽上限）
- Reduction 是典型的 memory-bound 操作

### 6. GEMM Batch Scaling

| Batch | 耗时 (ms) | 吞吐 (TFLOPS) |
|-------|----------|--------------|
| 1 | 9.758 | 14.08 |
| 4 | 50.826 | 10.82 |
| 16 | 167.511 | 13.13 |
| 64 | 670.564 | 13.12 |

矩阵: [1x4096] x [4096x4096]，B=batch (bmm)

**关键发现**:
- Batch=1 吞吐 14 TFLOPS（意外高，可能是编译器优化）
- Batch=4 吞吐下降（可能受限于显存分配策略）
- Batch >= 16 吞吐稳定 ~13 TFLOPS

## A16 vs A100 性能对比

| 指标 | A16 | A100-80G | 比值 |
|------|-----|----------|------|
| SM 数量 | 10 | 108 | 1:11 |
| 显存 | 15.6 GB | 80 GB | 1:5 |
| FP16 GEMM | ~11 TFLOPS | ~130 TFLOPS | 1:12 |
| 内存带宽 | ~157 GB/s | ~1555 GB/s | 1:10 |
| SDPA | ~14 TFLOPS | ~150 TFLOPS | 1:11 |
| Kernel launch | ~9 us | ~5-7 us | ~1.5x |

A16 约为 A100 的 **1/10 - 1/12** 性能，适合小模型实验和学习，不适合大模型训练。

## 推理性能估算（基于实验数据）

以 GPT-2 (124M, FP16) 为例：
- 参数量: 124M × 2 bytes = 248 MB
- 每层 GEMM: [1x768] × [768x3072] (MLP) = 2 × 768 × 3072 × 768 ≈ 1.8G FLOPs
- 12 层总 FLOPs ≈ 12 × 3.6G ≈ 43.2G FLOPs per token
- 理论延迟 (11 TFLOPS): 43.2G / 11T = ~3.9 ms/token
- 考虑 kernel launch overhead (+8us × ~50 ops): ~4ms/token
- 实测 decode 速度: ~575 tok/s (batch=4)，即 ~1.7 ms/tok/batch
  - 比 theoretical 快，说明 batch=4 时 GPU 利用率好

## 结论

1. **A16 是入门级 GPU**: 10 SM, 15.6 GB, ~11 TFLOPS FP16，适合学习和实验
2. **Kernel launch overhead 是关键瓶颈**: ~9 us，decode 阶段每步小操作受影响最大
3. **Flash Attention 高效**: SDPA 达到 ~14 TFLOPS，接近内存带宽上限
4. **混合精度 4.4x 加速**: FP16 vs FP32 GEMM
5. **推理优化方向**: CUDA Graph (减少 kernel launch)、KV Cache (减少重复计算)、Continuous Batching (提高 GPU 利用率)

## 参考

- 实验脚本: `tools/gpu_profile_experiment.py`
- GPU 基准测试: `tools/gpu_benchmark.py`
- GPU 性能分析笔记: `notebook/fundamentals/gpu-profiling.md`
