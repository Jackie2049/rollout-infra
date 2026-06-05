# Tiled Attention 实现 — FlashAttention 概念验证 (RTX 4090)

> 2026-06-05 | 工具: `tools/tiled_attention.py` | RTX 4090 24GB
> 论文: FlashAttention (Dao et al., NeurIPS 2022)

## 1. 实验设计

**目的**: 用 Python 实现 FlashAttention 的核心算法 (tiling + online softmax), 验证:
1. 数值精确性 (online softmax 是否等价标准 softmax)
2. 内存节省 (O(N²) → O(N))
3. 为什么需要 CUDA kernel 才能真正加速

**算法核心**:
```
外循环: 遍历 Q 块 (B_r rows)
内循环: 遍历 K,V 块 (B_c rows)

Online softmax 更新:
  m_new = max(m_old, max(S_ij))      # running max
  l_new = exp(m_old-m_new)*l_old + Σexp(S_ij-m_new)  # running sum
  O_new = (exp(m_old-m_new)*l_old*O_old + exp(S_ij-m_new)@V_j) / l_new
```

## 2. 数值精确性验证

| Block Size | Max Error | Mean Error | Cosine Similarity |
|-----------|-----------|------------|-------------------|
| 16 | 5.96e-07 | 3.38e-08 | 1.00000000 |
| 32 | 5.36e-07 | 2.35e-08 | 1.00000012 |
| 64 | 4.17e-07 | 2.44e-08 | 1.00000012 |
| 128 | 4.17e-07 | 2.63e-08 | 1.00000000 |

**结论: Online softmax 是数学上精确的!** 误差 < 6e-7 (float32 精度极限).
→ 这和之前 FlashAttention 模拟器的结果一致 (误差 < 3.3e-7)

## 3. 内存节省

| N | Naive | Tiled | 节省比 |
|---|-------|-------|--------|
| 512 | 30.5 MB | 15.8 MB | 1.94x |
| 1024 | 85.1 MB | 21.0 MB | **4.04x** |
| 2048 | 294.8 MB | 31.6 MB | **9.33x** |

**关键发现**: 内存比 ≈ N² / N = N → 线性增长!
- N=512: ~2x (因为 N 较小, 固定开销占比大)
- N=2048: ~9x (更显著的节省)
- N=8192: 预期 ~36x (如果 naive 不 OOM 的话)

## 4. 性能对比

| N | Naive | Tiled (Python) | SDPA (FlashAttn) |
|---|-------|---------------|-----------------|
| 256 | 0.062ms | 3.618ms (58x慢) | 0.065ms |
| 512 | 0.070ms | 13.6ms (193x慢) | 0.072ms |
| 1024 | 0.177ms | 52.5ms (296x慢) | **0.146ms** (1.2x快) |
| 2048 | 0.975ms | 207ms (212x慢) | **0.437ms** (2.2x快) |

**核心教训**:
- Python tiled attention 极慢 (200x+) — Python 循环无法利用 GPU 并行
- SDPA (真实 FlashAttention CUDA kernel) 在 N≥1024 时开始超过 naive
- **N 越大, FlashAttention 优势越大**: 减少 HBM 访问的收益随 N 增长

## 5. Block Size 效果

| BS | 时间 (N=1024) | Max Error |
|----|-------------|-----------|
| 16 | 821.8ms | 7.75e-07 |
| 32 | 207.4ms | 5.36e-07 |
| 64 | 52.5ms | 5.66e-07 |
| 128 | 13.5ms | 6.56e-07 |
| 256 | 3.6ms | 5.36e-07 |
| 512 | 1.0ms | 7.15e-07 |

**发现**:
- 大 block = 快 (更少 Python 循环迭代)
- 精度与 block size 无关 (所有 < 8e-7)
- 实际 CUDA kernel: block size 由 SRAM 大小决定 (A100: ~192KB/SM)

## 6. 核心学习

1. **Online softmax 是精确的**: 通过 running max/sum 逐步更新, 数学等价全局 softmax
2. **内存节省是真实的**: O(N²) → O(N), N=2048 时节省 9.3x
3. **Python tiling 无法加速**: 需要 CUDA/Triton kernel 才能并行化 tile 计算
4. **为什么需要 CUDA kernel**:
   - 所有 tile 可以并行计算 (不同的 Q block 独立)
   - 每个 tile 在 SRAM 中完成 (避免 HBM 读写)
   - Python 循环是串行的, 无法利用 GPU
5. **SDPA 加速比随 N 增长**: N=1024 时 1.2x, N=2048 时 2.2x
6. **Block size 只影响速度, 不影响精度**: 在 SRAM 容量允许范围内越大越好
