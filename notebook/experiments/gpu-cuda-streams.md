# CUDA Streams & Events 实验 — A16 实测

> 2026-06-04 | A16 15GB (10 SMs, CUDA 11.8)

## 实验概览

5 个实验深入理解 CUDA 并发编程模型。

## 1. CUDA Event vs time.time() 精度

| 操作 | Event (ms) | time.time (ms) | Ratio |
|------|-----------|----------------|-------|
| empty kernel | 0.0033 | 0.0033 | 1.00x |
| small GEMM 64x64 | 0.031 | 0.031 | 1.00x |
| medium GEMM 512x512 | 0.139 | 0.139 | 1.00x |
| large GEMM 2048x2048 | 7.004 | 7.136 | 1.02x |
| memcpy 64MB | 1.368 | 1.377 | 1.01x |

**结论**: Event 和 time.time() 几乎一致，但 Event 只计 GPU 时间，不受 CPU 干扰。

## 2. 多 Stream 并行 GEMM

GEMM size: 1024x1024

| n_ops | 1-stream (ms) | n-stream (ms) | Speedup | Efficiency |
|-------|--------------|--------------|---------|------------|
| 1 | 0.77 | 0.15 | 5.19x | 519% |
| 2 | 1.58 | 0.18 | 8.99x | 450% |
| 4 | 3.48 | 0.33 | 10.48x | 262% |
| 8 | 7.12 | 6.32 | 1.13x | 14% |

**结论**: A16 只有 10 SMs，4 个并行 GEMM 接近饱和。>4 个操作效率急剧下降。多 stream 加速 >100% 是因为测量包含了单 stream 的 launch 开销累加。

## 3. 通信-计算重叠模拟

关键场景: 计算时间 > 通信时间时，通信 100% 被隐藏。

| Copy MB | GEMM N | Compute (ms) | Copy (ms) | Overlap (ms) | Hidden% |
|---------|--------|-------------|-----------|-------------|---------|
| 1 | 2048 | 6.56 | 0.01 | 0.36 | 100% |
| 16 | 1024 | 0.79 | 0.25 | 0.14 | 100% |
| 64 | 512 | 0.12 | 0.82 | 0.16 | 94.7% |
| 64 | 2048 | 6.55 | 9.67 | 38.40 | 0% |

**结论**: 通信 < 计算 → 完全隐藏；通信 > 计算 → 无法隐藏。这解释了为什么 NVLink (300GB/s) 比以太网 (12.5GB/s) 的重叠效果好得多。

## 4. CUDA Graph 性能

| n_layers | Normal (ms) | Graph (ms) | Speedup | Launch/layer (ms) |
|----------|------------|-----------|---------|-------------------|
| 2 | 0.095 | 0.018 | 5.40x | 0.039 |
| 4 | 0.174 | 0.038 | 4.60x | 0.034 |
| 8 | 0.339 | 0.071 | 4.79x | 0.034 |
| 16 | 0.748 | 0.138 | 5.40x | 0.038 |

**结论**:
- 每个 kernel launch 开销 ~0.034ms
- CUDA Graph 固定 ~5x 加速 (无论层数)
- vLLM 使用 CUDA Graph 加速 decode step (大量小 kernel)

## 5. Stream 同步模式

| 模式 | 时间 (ms) | 相对最优 |
|------|---------|---------|
| Launch all + sync once | 0.59 | 1.00x |
| Launch + sync each | 27.05 | 45.57x |
| Launch + event wait | 26.86 | 45.26x |

**结论**: 逐个同步极其低效 (45x 慢)。分布式训练中应尽量减少同步点，使用 AllReduce 替代多个点对点同步。

## 关键洞察

1. **CUDA Graph 是 decode 推理的关键优化**: 消除每步 ~0.034ms 的 launch 开销
2. **通信-计算重叠需要 NVLink**: 以太网通信太慢，无法被隐藏
3. **多 Stream 受 SM 数限制**: A16 只有 10 SMs，4+ 并行就饱和
4. **批量同步远优于逐个同步**: Launch-all-sync-once 模式

## 相关笔记

- [GPU Micro-Benchmark](gpu-microbenchmark.md) — GEMM Roofline
- [Megatron-LM Reading](../projects/megatron-lm-reading.md) — TP/PP 实际通信模式
- [vLLM Architecture](../projects/vllm-reading.md) — CUDA Graph 在 vLLM 中的使用
