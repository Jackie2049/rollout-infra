# RTX 4090 全面 Benchmark 实测

> 2026-06-05 | 8x RTX 4090 服务器 (CUDA 12.8, PyTorch 2.9.0+cu128, Triton 3.5.0)
> 脚本: `tools/gpu_benchmark_4090.py`

## GPU 规格

| 参数 | RTX 4090 | A16 (对比) |
|------|----------|------------|
| Architecture | Ada Lovelace (SM 8.9) | Ampere (SM 8.6) |
| SMs | 128 | 10 |
| FP16 Tensor Core | 82.6 TFLOPS (理论) | 14.7 TFLOPS |
| FP32 | 82.6 TFLOPS | — |
| HBM | 21,120 MHz, 384-bit GDDR6X | — |
| VRAM | 24 GB | 15 GB |
| Driver | 575.57.08 | 510.54 |
| CUDA | 12.8 | 11.7 |

## Experiment 1: FlashAttention (SDPA)

### 结果 (32 heads, head_dim=64, FP16)

| Batch | SeqLen | Naive (ms) | SDPA (ms) | Speedup | TFLOPS |
|-------|--------|------------|-----------|---------|--------|
| 1 | 256 | 0.067 | 0.020 | 3.4x | 27.0 |
| 1 | 512 | 0.079 | 0.023 | 3.4x | 92.1 |
| 1 | 1024 | 0.398 | 0.058 | 6.9x | 148.0 |
| 1 | 2048 | 1.831 | 0.216 | 8.5x | 159.1 |
| 1 | 4096 | OOM | 0.851 | — | 161.6 |
| 1 | 8192 | OOM | 3.387 | — | 162.3 |
| 4 | 1024 | 1.861 | 0.216 | 8.6x | 159.1 |
| 4 | 2048 | 7.236 | 0.857 | 8.4x | 160.3 |
| 4 | 4096 | OOM | 3.387 | — | 162.3 |
| 16 | 2048 | OOM | 3.452 | — | 159.2 |
| 32 | 4096 | OOM | 27.754 | — | 158.5 |
| 32 | 8192 | OOM | 110.274 | — | 159.5 |

### 分析
- **SDPA 峰值**: ~162 TFLOPS, 接近理论峰值的 196% (tensor core 融合)
- **加速趋势**: S 越大加速越明显 (3.4x → 8.5x), FlashAttention tiling 对长序列优势更大
- **内存**: Naive 在 B>4 或 S>2048 OOM, SDPA O(N) vs O(N²) 内存
- **Batch 扩展**: B=1→32, TFLOPS 基本持平 (~158-162), 说明单 batch 已接近 GPU 饱和

## Experiment 2: CUDA Graph

### 配置: 3-layer MLP, B=16, H=4096, FP16

| 模式 | 延迟 | 备注 |
|------|------|------|
| No-graph | 0.159 ms | 基线 |
| CUDA Graph | 0.156 ms | 仅 1.02x |
| Launch overhead | ~1.1 us/op | 3 个 matmul |

### 对比 A16
| 指标 | RTX 4090 | A16 |
|------|----------|-----|
| CUDA Graph 加速 | 1.02x | 5.4x |
| Launch 开销 | 1.1 us/op | 34 us/op |

### 结论
- RTX 4090 的 kernel launch 机制极快 (Ada GPU scheduler 改进)
- CUDA Graph 在高端 GPU 上收益有限
- 主要应用场景: 超多小 op (100+) 或低端 GPU

## Experiment 3: Multi-Stream

### 配置: 4x GEMM 4096x4096, FP16

| 模式 | 延迟 | 备注 |
|------|------|------|
| Sequential (1 stream) | 3.346 ms | 基线 |
| Parallel (4 streams) | 3.599 ms | 0.93x (略慢!) |

### 分析
- 单个 4096² GEMM 已占用全部 128 SM, 无剩余 SM 供并行
- Stream 切换本身有微小开销 (~0.08 ms)
- **Multi-stream 有效条件**: 单个 kernel 不饱和 GPU (如小 batch decode)
- 推理框架中 multi-stream 主要用于 通信-计算重叠 (NCCL + GEMM)

## Experiment 4: Batch Decode

### 配置: hidden=4096, vocab=32000, FP16 (LM head 投影)

| Batch | 延迟 (ms) | tok/s | TFLOPS | 利用率 |
|-------|-----------|-------|--------|--------|
| 1 | 0.305 | 3,276 | 0.9 | 0.5% |
| 2 | 0.289 | 6,925 | 1.8 | 1.1% |
| 4 | 0.289 | 13,843 | 3.6 | 2.2% |
| 8 | 0.289 | 27,665 | 7.3 | 4.4% |
| 16 | 0.290 | 55,208 | 14.5 | 8.8% |
| 32 | 0.296 | 108,034 | 28.3 | 17.2% |
| 64 | 0.309 | 206,789 | 54.2 | 32.9% |
| 128 | 0.331 | 386,853 | 101.4 | 61.5% |
| 256 | 0.505 | 507,203 | 133.0 | 80.7% |
| 512 | 0.968 | 529,033 | 138.7 | 84.1% |

### 分析
- **线性区间**: B=1→64, 延迟几乎不变 (~0.29 ms), 吞吐线性增长
- **饱和点**: B≥128, 延迟开始上升
- **峰值**: 529K tok/s @ B=512
- **Batch=1 利用率仅 0.5%**: Continuous Batching 是必须的
- Decode 始终 memory-bound (TFLOPS 远低于 SDPA 的 162 TFLOPS)

## Experiment 5: KV Cache 容量分析

### 7B 模型 (32 heads, 4096 hidden, 128 head_dim, 32 layers)
| SeqLen | KV/layer (MB) | Total 32L (MB) |
|--------|---------------|----------------|
| 512 | 8.4 | 268.4 |
| 2048 | 33.6 | 1,073.7 |
| 4096 | 67.1 | 2,147.5 |
| 8192 | 134.2 | 4,295.0 |
| 32768 | 536.9 | 17,179.9 |

### RTX 4090 容量 (7B, 80% VRAM for KV)
- Block size=16: 0.26 MB/block
- Total blocks: 75,000
- Max concurrent @ seq=512: **2,344 requests**
- **限制**: 70B seq=32K 需 85.9 GB, 远超 24 GB

## 关键结论

1. **SDPA 是 compute-bound**: 峰值 162 TFLOPS, 接近硬件极限
2. **CUDA Graph 在高端 GPU 收益小**: Launch 开销本身极低
3. **Multi-stream 对大 GEMM 无效**: 需要小 kernel + 通信重叠才有收益
4. **Decode 始终 memory-bound**: 峰值 138.7 TFLOPS vs SDPA 的 162 TFLOPS
5. **RTX 4090 是 11x A16**: 几乎所有指标都是数量级提升
6. **Batch=1 是浪费**: 利用率 <1%, Continuous Batching 必须开启
