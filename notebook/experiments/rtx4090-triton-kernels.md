# Triton Kernel Benchmark — RTX 4090 实测

> 2026-06-05 | 工具: `tools/triton_kernel_benchmark.py` | RTX 4090 24GB
> Triton 3.5.0 + PyTorch 2.9.0 + CUDA 12.8

## 1. 实验结果

### 1.1 Fused RMSNorm (Triton vs PyTorch)

| Hidden Dim | Batch | PyTorch (ms) | Triton (ms) | 加速比 |
|------------|-------|-------------|-------------|--------|
| 256 | 32 | 0.052 | 0.028 | **1.84x** |
| 512 | 16 | 0.051 | 0.029 | **1.79x** |
| 1024 | 8 | 0.051 | 0.027 | **1.88x** |
| 2048 | 4 | 0.051 | 0.028 | **1.84x** |
| 4096 | 2 | 0.051 | 0.027 | **1.86x** |

**结论**: Triton fused RMSNorm 稳定 1.8x 加速。原因: PyTorch 需要 3 次 kernel launch (mean→rsqrt→mul), Triton 只需 1 次。

### 1.2 Fused Softmax (Triton vs PyTorch)

| Seq Len | Batch | PyTorch (ms) | Triton (ms) | 加速比 |
|---------|-------|-------------|-------------|--------|
| 256 | 32 | 0.007 | 0.025 | 0.28x |
| 1024 | 8 | 0.007 | 0.039 | 0.18x |
| 4096 | 2 | 0.007 | 0.026 | 0.27x |
| 8192 | 1 | 0.008 | 0.024 | 0.29x |

**结论**: 自定义 Triton softmax 比 PyTorch 慢 3-4x!
- PyTorch 的 softmax 使用 NVIDIA 高度优化的 cuDNN kernel
- 小 seq_len (256-8192) 时, PyTorch 已极致优化
- 自定义 kernel 只在 PyTorch 没有优化的操作上有意义
- **教训**: 不要盲目写自定义 kernel, 先 benchmark PyTorch baseline

### 1.3 Fused QKV Projection

| Hidden Dim | Separate (ms) | Stacked (ms) | 加速比 |
|------------|-------------|-------------|--------|
| 512 | 0.041 | 0.021 | **1.98x** |
| 1024 | 0.042 | 0.021 | **1.97x** |
| 2048 | 0.045 | 0.031 | **1.48x** |

**结论**: Stacked QKV (1个大matmul) 比 separate (3个小matmul) 快 ~2x。
- 减少内核启动开销 (3次→1次)
- 更好的 GPU 利用率 (更大的矩阵→更多并行)
- 这是 vLLM, SGLang 等框架的标准做法

### 1.4 LayerNorm vs RMSNorm

| Dim | LayerNorm (ms) | RMSNorm (ms) | RMSNorm 相对速度 |
|-----|---------------|-------------|-----------------|
| 256 | 0.019 | 0.051 | 0.37x (慢!) |
| 1024 | 0.020 | 0.052 | 0.38x |
| 4096 | 0.019 | 0.051 | 0.37x |

**重要发现**: RMSNorm 比 LayerNorm **慢 2.6x** (在这个实现中)!
- PyTorch 的 LayerNorm 使用 cuDNN 高度优化的 kernel
- 我的 RMSNorm 用纯 PyTorch ops (mean→pow→rsqrt→mul), 3次 kernel launch
- 而上面 Triton RMSNorm (0.028ms) 接近 LayerNorm (0.019ms) 的速度
- **结论**: RMSNorm 的理论优势 (少一个 mean 操作) 在实现层面被 cuDNN 优化抵消

### 1.5 Memory Bandwidth

| 操作 | 数据量 | 时间 (ms) | 有效带宽 | 峰值百分比 |
|------|--------|----------|---------|-----------|
| Vector Add | 64 MB | 0.107 | 945 GB/s | **93.7%** |
| Elem Mul | 64 MB | 0.107 | 944 GB/s | **93.7%** |
| Reduction | 32 MB | 0.013 | 2638 GB/s* | >100% |

*Reduction 超过峰值因为输出很小, 可能有 L2 cache 命中。

## 2. 关键洞察

### 2.1 什么时候自定义 Kernel 有意义?

```
值得:
  ✓ PyTorch 没有优化的操作 (如 fused RMSNorm)
  ✓ 需要减少 kernel launch 次数 (3→1)
  ✓ 需要减少中间内存读写 (SRAM → register)
  ✓ 特殊访问模式 (paged attention, block-sparse)

不值得:
  ✗ PyTorch 已高度优化的操作 (softmax, matmul, LayerNorm)
  ✗ 简单元素操作 (add, mul) — 已接近峰值带宽
  ✗ 计算密集型操作 — cuBLAS 已接近峰值 FLOPS
```

### 2.2 RTX 4090 Kernel 性能特征

```
  Launch 开销: ~5μs (非常快)
  峰值 HBM 带宽: 945/1008 = 93.7% (非常高效)
  Triton vs PyTorch: 只有在融合多个操作时有优势
  小矩阵: launch 开销占比大 → 融合收益大
  大矩阵: 计算主导 → 融合收益小
```

### 2.3 Triton vs CUDA C

```
Triton 优势:
  - Python 写 kernel, 编译时优化
  - 自动处理 memory coalescing
  - 自动处理 bank conflicts
  - 开发速度快 10x

Triton 劣势:
  - 不如 CUDA C 灵活 (shared memory 控制有限)
  - 编译时间较长
  - 调试困难 (编译错误信息不清晰)

结论: Triton 适合快速原型和大多数 kernel,
      CUDA C 适合极致优化的关键路径 (如 FlashAttention)
```
