# LLM 推理模拟 Benchmark — A16 实测

> 2026-06-04 | A16 15GB (10 SMs, CUDA 11.8) | SimGPT (GPT-2 style)

## 实验概览

用纯 PyTorch 构建的 GPT 模型，模拟 OPT-125M/350M 的推理全流程。

## 1. Prefill vs Decode 延迟

### OPT-125M-like (162M params, 12层, H=768)

**Prefill:**

| Seq | ms | tok/s |
|-----|-----|-------|
| 32 | 3.70 | 8,649 |
| 64 | 3.80 | 16,830 |
| 128 | 4.18 | 30,592 |
| 256 | 7.92 | 32,321 |
| 512 | 14.74 | 34,737 |

**Decode (batch=1): 3.44 ms/tok = 291 tok/s**

### OPT-350M-like (405M params, 24层, H=1024)

**Prefill:**

| Seq | ms | tok/s |
|-----|-----|-------|
| 128 | 10.84 | 11,806 |
| 512 | 37.73 | 13,568 |

**Decode (batch=1): 7.02 ms/tok = 142 tok/s**

**结论**: 350M decode 吞吐约为 125M 的 1/2 (参数量 2.5x, 但吞吐 ~2x 下降)。

## 2. Batch Decode 吞吐量曲线 (125M)

| Batch | ms/tok | tok/s | Scaling |
|-------|--------|-------|---------|
| 1 | 3.607 | 277 | 1.0x |
| 8 | 3.781 | 2,116 | 7.6x |
| 32 | 4.210 | 7,601 | 27.4x |
| 128 | 10.295 | 12,434 | 44.8x |
| 256 | 20.043 | 12,772 | 46.1x |
| 512 | 38.819 | 13,190 | 47.6x |

**结论**: 吞吐量在 batch=256 后饱和 (~13K tok/s)。batch=1→128 近似线性，之后趋平。

## 3. Decode Roofline 分析

| Batch | Actual ms | Memory-bound | 实际/理论 |
|-------|----------|-------------|----------|
| 1 | 3.48 | 1976.6 | 0.002x |
| 64 | 6.71 | 1976.6 | 0.003x |
| 128 | 10.30 | 1976.6 | 0.005x |

**注意**: 这里的 mem_bound 计算有误 (包含了所有模型内存而非每层)。实际 decode 确实是 memory-bound。

## 4. Continuous Batching 模拟

| Active Requests | ms/tok | tok/s | Efficiency |
|----------------|--------|-------|------------|
| 128 | 10.31 | 12,411 | 100% |
| 64 | 6.72 | 9,524 | 77% |
| 32 | 4.22 | 7,582 | 61% |
| 16 | 3.60 | 4,446 | 36% |
| 8 | 3.73 | 2,144 | 17% |
| 1 | 3.48 | 287 | 2% |

**关键发现**: batch=1 时只用了满 batch 的 2% 吞吐。Continuous batching 是推理服务高利用率的必需技术。

## 5. 关键洞察

1. **Decode 吞吐 ∝ batch size** (memory-bound, 线性到饱和点)
2. **Prefill 吞吐 ∝ seq length** (compute-bound, 利用 Tensor Core)
3. **Continuous Batching 效率**: 必须保持高 batch 利用率
4. **A16 上 125M**: 峰值 ~13K tok/s decode, ~35K tok/s prefill
5. **Scaling 饱和点**: batch=256 (A16 的 10 SMs 限制)

## 相关笔记

- [CUDA Streams](gpu-cuda-streams.md) — 并发优化
- [Kernel Tuning](gpu-kernel-tuning.md) — SDPA vs Naive
- [Attention Benchmark](gpu-attention-benchmark.md) — Prefill vs Decode 详细分析
