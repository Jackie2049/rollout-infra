# Paper Reading: FlashAttention — Fast and Memory-Efficient Exact Attention

> Dao et al., NeurIPS 2022 (v1), 2023 (v2) | Stanford
> 精读日期: 2026-06-05
> 优先级: P2 (但与 AI Infra 极度相关)

## 1. 论文概要

**核心贡献**: 通过 **tiling + online softmax + recomputation** 实现 O(N) 内存的精确 attention, 不需要近似.

**解决的问题**:
- 标准 attention: O(N²) 内存 (存储 N×N attention matrix)
- FlashAttention: O(N) 内存 (只存储 output, 不存储 attention matrix)
- 速度: 比 PyTorch 标准实现快 2-4x (减少 HBM 读写)

## 2. 核心思想

### 2.1 为什么标准 Attention 慢?

```
标准 Attention 实现:
  1. S = Q @ K^T        → 写到 HBM (N×N 矩阵)
  2. P = softmax(S)      → 读 S, 写 P 到 HBM
  3. O = P @ V           → 读 P, 写 O 到 HBM

问题:
  - 3 次 HBM 读写, 每次都是 O(N²) 数据
  - N=2048: 4MB attention matrix
  - N=8192: 256MB attention matrix
  - N=32768: 4GB attention matrix!

  HBM 带宽是瓶颈 (A100: 2TB/s vs SRAM: 19TB/s)
  → 减少 HBM 访问 = 加速
```

### 2.2 Tiling 策略

```
FlashAttention 的核心: 将 Q, K, V 分成小块 (tiles)

Q: (N, d) → 分成 B_r 块, 每块 B_r × d
K: (N, d) → 分成 B_c 块, 描块 B_c × d
V: (N, d) → 分成 B_c 块, 每块 B_c × d

外循环: 遍历 Q 的块 (B_r)
内循环: 遍历 K, V 的块 (B_c)

每次只计算一个 B_r × B_c 的 attention tile:
  - 所有中间结果在 SRAM (on-chip) 中
  - 只把最终输出 O 写回 HBM

内存: 只需要存储 O(N × d), 不需要 O(N²)
```

### 2.3 Online Softmax (关键数学)

```
问题: softmax 需要知道全局 max 和 sum, 但我们只看到局部 tile

标准 softmax:
  P_ij = exp(S_ij - max(S)) / Σ exp(S_ij - max(S))

Online softmax (Milakov & Gimelshein, 2018):
  维护两个 running 统计量:
  - m^(j): 到第 j 个 tile 为止的最大值
  - l^(j): 到第 j 个 tile 为止的归一化常数

更新规则:
  当处理第 j 个 K,V tile 时:
  m_new = max(m_old, max(S_new))
  l_new = exp(m_old - m_new) * l_old + Σ exp(S_new - m_new)
  O_new = (exp(m_old - m_new) * l_old * O_old + exp(S_new - m_new) @ V_j) / l_new

数学证明: 这个逐步更新等价于在完整 softmax 上计算!
  → FlashAttention 是精确的, 不是近似
```

### 2.4 Recomputation (反向传播)

```
前向传播: 不存储 attention matrix P, 只存储 O, m, l

反向传播需要 P 怎么办?
  → 重新计算! (recomputation)

经典权衡: 用计算换内存

反向传播重新计算:
  S = Q @ K^T    (重新计算)
  P = softmax(S) (重新计算)
  dV = P^T @ dO  (需要 P)
  dP = dO @ V^T  (需要 V)
  dS = dP ⊙ (1 - P) ⊙ P  (softmax 的梯度)

额外 FLOPs: ~30% (重新计算 S)
节省内存: O(N²) → O(N) (不存储 P)

总体: 因为减少了 HBM 读写, 即使多算 30% FLOPs,
       实际 wall-clock 时间仍然更快!
```

## 3. IO 复杂度分析

```
SRAM 大小: M (A100: 192 MB per SM)
HBM 大小: H (A100: 80 GB)
HBM 带宽: β_HBM (A100: 2 TB/s)

标准 Attention:
  HBM 访问 = N² + N² + N² = Ω(N²d + N²)  bytes

FlashAttention:
  HBM 访问 = Ω(N²d²/M)  bytes (tile 大小 ~ M/d)

当 d² << M (典型: d=64, M=192KB → d²/M = 0.02):
  FlashAttention HBM 访问 ≈ N² × 0.02 ≈ N²/50
  → 比 标准 Attention 少 ~50x HBM 读写!

实际收益:
  A100 (compute-bound 场景):
    FlashAttention: 70-75% of theoretical peak TFLOPS
    PyTorch naive: 25-40% of theoretical peak TFLOPS
    → 2-3x 加速

  为什么不是 50x?
    因为在 compute-bound 场景, 减少内存访问不会线性加速
    加速来自更好的 GPU 利用率 (减少内存访问延迟)
```

## 4. FlashAttention-2 (2023)

```
改进:

1. 减少 non-matmul FLOPs:
   v1: ~30% non-matmul (indexing, reshape, etc.)
   v2: ~10% non-matmul

2. 更好的 thread 分配:
   v1: 每个 SM 处理 attention tile
   v2: 每个 warp 处理 attention row (更细粒度)

3. 减少 shared memory 读写:
   v1: 中间结果写回 shared memory
   v2: 利用 register 直接传递

性能提升:
  v1 → v2: ~2x faster
  A100: 50-73% MFU (理论峰值的 50-73%)
  H100: 接近峰值

FlashAttention-3 (2024, H100 only):
  利用 H100 的异步 execution + TMA
  → 进一步 ~2x on H100
```

## 5. RTX 4090 实测验证 (之前的实验)

```
SDPA (PyTorch 内置 FlashAttention):
  RTX 4090 峰值: 162.3 TFLOPS @ B=1, S=8192
  加速比: 3.4x (S=256) → 8.5x (S=2048)
  序列越长加速越大 (N² 内存节省越显著)

Naive attention OOM:
  B>4 或 S>2048 → OOM (O(N²) 内存)
  SDPA 在 B=32, S=8192 也不 OOM (O(N) 内存)
```

## 6. 对 AI Infra 的影响

```
1. Prefill 优化:
   FlashAttention 使 prefill 计算 bound (而非 memory bound)
   → 充分利用 GPU FLOPS

2. 长上下文:
   O(N²) → O(N) 内存使 128K+ 上下文成为可能
   → 7B/128K: KV=32GB, 但 attention matrix 从 32GB→0

3. KV Cache 成为主要开销:
   FlashAttention 消除了 attention matrix 的内存瓶颈
   → KV Cache 成为长序列的新瓶颈
   → PagedAttention, MLA, Quantized KV 等技术应运而生

4. 所有推理框架必须支持:
   vLLM: 20+ attention backends
   SGLang: RadixAttention (基于 FlashAttention)
   TensorRT-LLM: NVIDIA 优化版

5. 训练加速:
   2-4x attention 加速 → 整体训练 ~1.5x 加速
   (attention 占 ~30-40% 训练时间)
```

## 7. 核心学习

1. **IO 意识 > 算力意识**: 减少 HBM 读写比减少 FLOPs 更重要
2. **Online softmax 是数学妙招**: 逐步更新等价于全局 softmax
3. **Recomputation 是通用策略**: 用计算换内存, 在 GPU 上通常值得
4. **精确 > 近似**: FlashAttention 不牺牲精度, 比 Sparse Attention 更受欢迎
5. **硬件-算法协同**: 理解 SRAM/HBM 层次结构才能写出快 kernel
6. **Tiling 是 GPU 编程核心**: 所有高效 kernel 都基于 tiling
7. **v2/v3 证明还有优化空间**: 即使 v1 已经很好, 仍有 2-4x 提升
