# GPU 实验记录 — Attention 实现对比 Benchmark

> 2026-06-04 | NVIDIA A16 15GB | Attention 算法性能分析

## 实验 1: Naive vs SDPA (FlashAttention) ⭐

B=4, H=8, D=64, FP16

| SeqLen | Naive (ms) | SDPA (ms) | Speedup | Naive Mem | SDPA Mem | Mem Saving |
|--------|-----------|----------|---------|-----------|----------|------------|
| 128 | 0.102 | 0.020 | 5.0x | 13 MB | 11 MB | 1.2x |
| 256 | 0.308 | 0.044 | 7.1x | 21 MB | 13 MB | 1.6x |
| 512 | 1.086 | 0.116 | **9.4x** | 51 MB | 17 MB | 3.0x |
| 1024 | 4.449 | 0.389 | **11.4x** | 161 MB | 25 MB | 6.4x |
| 2048 | 35.057 | 1.327 | **26.4x** | 583 MB | 42 MB | **13.9x** |

**核心发现**:
- **SDPA (FlashAttention) 比 Naive 快 5-26x**: 越长的序列加速越明显
- **内存节省 1.2-14x**: Naive 需要 O(N²) 存储 attention 矩阵，FlashAttention 不需要
- S=2048: 583MB → 42MB，这是 FlashAttention 的核心价值 — 使得长序列成为可能

## 实验 2: Prefill vs Decode

| Phase | 特征 | 性能特征 |
|-------|------|---------|
| Prefill | q_len = kv_len | Compute-bound, 最高 23.7 TFLOPS |
| Decode | q_len = 1 | Memory-bound, KV 读取带宽决定性能 |

Prefill TFLOPS 随序列长度增长: 1.7→23.7 (14x)，因为大矩阵更充分使用 Tensor Core。

Decode 带宽利用率: kv_len=1024 时 84.8 GB/s (实测峰值 170 GB/s 的 50%)。

## 实验 3: MHA vs GQA vs MQA

B=4, S=1024, D=64, FP16

| 配置 | Q Heads | KV Heads | 时间 (ms) | KV Cache | 加速 |
|------|---------|----------|----------|----------|------|
| MHA | 16 | 16 | 0.745 | 16.8 MB | 1.00x |
| GQA | 16 | 4 | 0.758 | 4.2 MB | 0.98x |
| GQA | 16 | 2 | 0.768 | 2.1 MB | 0.97x |
| MQA | 16 | 1 | 0.752 | 1.0 MB | 0.99x |

**意外发现**: GQA/MQA 在 Prefill 场景没有加速！
- 原因: Prefill 是 compute-bound，减少 KV heads 不影响 FLOPS
- **真正的收益在 KV Cache 内存**: MQA 减少 16x 内存
- **推理收益**: Decode 时 KV Cache 带宽需求减少 16x

## 实验 4: Batch Decode Throughput

kv_len=1024, H=8, D=64

| Batch | tok/s | KV BW (GB/s) | 效率 |
|-------|-------|-------------|------|
| 1 | 40M | 82.7 | 49% |
| 16 | 49M | 100.6 | 59% |
| 128 | 46M | 94.0 | 55% |

**发现**: 吞吐在 batch=16 达到峰值 49M tok/s，之后反而略有下降。这可能是因为大 batch 下 attention 计算开始从 memory-bound 转向 compute-bound。

## 实验 5: Online Softmax (FlashAttention 核心算法)

| N | Max Error | 说明 |
|---|-----------|------|
| 128 | 3.73e-9 | ✓ |
| 256 | 7.45e-9 | ✓ |
| 512 | 1.86e-9 | ✓ |
| 1024 | 1.40e-9 | ✓ |
| 2048 | 9.31e-10 | ✓ |

**精度验证**: Online softmax 误差 <1e-8，与标准 softmax 几乎一致。

**FlashAttention 算法核心**:
```
标准 Attention: Q×K^T → 存储 [B,H,S,S] → softmax → ×V
FlashAttention: 分块处理, 不存储完整 attention 矩阵

for each tile (Q_block, K_block, V_block):
    1. 计算 Q_block × K_block^T
    2. Online softmax 更新 (running max + running sum)
    3. 累加到输出 O
    4. 不存储中间 attention 矩阵!
```

## 综合结论

1. **FlashAttention 是必须的**: 26x 加速 + 14x 内存节省，没有理由不用
2. **Prefill/Decode 的本质区别**: Prefill compute-bound (FLOPS), Decode memory-bound (带宽)
3. **GQA/MQA 的价值是内存而非速度**: Prefill 速度无差异，但 KV Cache 减少 4-16x
4. **Online Softmax 精度完美**: 分块计算误差 <1e-8，是 FlashAttention 可行的数学基础
5. **Batch Decode 最优 batch=16**: 超过后收益递减，因为开始转向 compute-bound
