# FlashAttention Algorithm Deep Dive

> 2026-06-07 | 基于 Dao et al., NeurIPS 2022 + FlashAttention-2 (2023)
> 之前已有模拟器笔记 (误差 <3.3e-7, IO 节省 2.2x), 本文聚焦算法层面深度

## 1. 核心问题: 标准 Attention 的 IO Bottleneck

标准 Attention 计算:
```
S = Q @ K^T          # [B, H, N, N] — 需要 O(N²) 内存!
P = softmax(S)        # [B, H, N, N] — 又 O(N²) 内存!
O = P @ V             # [B, H, N, D] — 最终输出
```

**问题**: 对于 N=8K, H=64, B=1:
- S矩阵: 8K × 8K × 64 × 4bytes = **16.4 GB** (超出 GPU HBM!)
- 这需要从 HBM 读写 S 和 P 各一次 → **4 × N² × H × 4bytes** 的 IO

**FlashAttention 的目标**: 不在 HBM 中存储完整的 S 和 P 矩阵

## 2. Tiling Strategy: Block-Level 计算

**核心思想**: 将 Q, K, V 分成小块 (tiles), 在 SRAM (共享内存) 中计算局部 attention

```
SRAM 大小 ≈ 192 KB (A100) → 每个块大小:
B_r = B_c = √(SRAM / (4 × d)) ≈ √(192K / (4 × 64)) ≈ 22

实际: B_r = 64, B_c = 64 (FlashAttention-2 优化)
```

**算法** (FlashAttention-1, 分块):
```
for i = 0, ..., Tr-1:        # Q 的块
  O_i = 0                     # 累积输出
  l_i = 0                     # 累积 softmax 分母 (sum of exp)
  m_i = -inf                  # 累积 softmax 最大值 (max)

  for j = 0, ..., Tc-1:      # K, V 的块
    S_ij = Q_i @ K_j^T       # 在 SRAM 中计算 [B_r, B_c]
    m_ij = max(S_ij)         # 当前块最大值

    # Online softmax: 更新全局最大值
    m_new = max(m_i, m_ij)
    # 修正因子: exp(m_i - m_new) 和 exp(m_ij - m_new)
    alpha = exp(m_i - m_new)
    beta = exp(m_ij - m_new)

    # 更新累积输出和分母
    P_ij = exp(S_ij - m_new)    # 局部 softmax (未归一化)
    l_i = alpha * l_i + beta * sum(P_ij)
    O_i = alpha * O_i + P_ij @ V_j
    m_i = m_new

  O_i = O_i / l_i              # 最终归一化
```

## 3. Online Softmax: 不需要存储完整 Attention Matrix

**关键数学**: softmax 的 "增量计算" 性质

对于两个块 j₁ 和 j₂:
```
softmax([x₁, ..., x_n, y₁, ..., y_m])
= [exp(x_k - m) / l, exp(y_k - m) / l]

其中 m = max(max(x), max(y))
     l = sum(exp(x - m)) + sum(exp(y - m))
```

**增量更新公式**:
```
已知: m_old, l_old = sum(exp(x - m_old))
新数据: m_new = max(m_old, max(y))
修正: l_new = exp(m_old - m_new) * l_old + sum(exp(y - m_new))
```

这个公式让我们可以逐块累积, **不需要先看到所有数据再计算 softmax**!

**数值稳定性**: 通过始终追踪最大值 `m_i` 并用 `exp(m_i - m_new)` 修正, 避免了 exp 溢出

## 4. IO 分析: 为什么 FlashAttention 省内存带宽

### 标准 Attention IO (HBM → SRAM → HBM):
```
读 Q:     B × H × N × d × 4 bytes
读 K:     B × H × N × d × 4 bytes
写 S:     B × H × N × N × 4 bytes   ← 巨大!
读 S:     B × H × N × N × 4 bytes   ← 又读回来
写 P:     B × H × N × N × 4 bytes   ← 巨大!
读 P:     B × H × N × N × 4 bytes
读 V:     B × H × N × d × 4 bytes
写 O:     B × H × N × d × 4 bytes

总计: 2 × B × H × (2Nd + 4N²) × 4 bytes ≈ 8BHN² (dominated by N² terms)
```

### FlashAttention IO:
```
读 Q:     B × H × N × d × 4 bytes × Tr 次 (每块读一次 Q_i)
          实际: Q 只读一次 (外循环固定 i, 内循环遍历 j)
读 K:     B × H × N × d × 4 bytes × Tr 次 (每块内循环遍历所有 K_j)
          实际: K 被读 Tr 次
读 V:     B × H × N × d × 4 bytes × Tr 次
写 O:     B × H × N × d × 4 bytes × 1 次

总计: 2 × B × H × (N × d + Tr × N × d) × 4 bytes
     = B × H × (2Nd + 2 × (N/B_r) × Nd) × 4 bytes
     ≈ BHN² × (2d/N × B_r + 2d/B_r) bytes

对于 N >> d: ≈ 2BHNd × (N/B_r) ← 比 8BHN² 小得多
```

**关键**: FlashAttention **不读写 S 和 P 矩阵到 HBM**, 省了 ~2N² × H 的 IO

### Arithmetic Intensity 对比
- 标准: `FLOPs/IO ≈ (2Nd² + 2N²d) / (8N²) ≈ d/4 + d²/4N ≈ d/4` (对大N, 主要项)
- Flash: `FLOPs/IO ≈ (2Nd² + 2N²d) / (2Nd × N/B_r) ≈ 2d × B_r/(N + d)`

对 A100 (HBM 2TB/s, FP16 312 TFLOPS):
- Ridge point: 312e12 / (2e12 × 2) = 78 ops/byte
- 标准 Attention (N=8K, d=64): AI ≈ 64/4 = 16 → **memory-bound**
- FlashAttention (B_r=64): AI ≈ 2×64×64/(8000+64) ≈ 1.02 → **still memory-bound!**

Wait, that doesn't seem right. Let me reconsider.

**更精确的分析** (FlashAttention-2 paper):
- Prefill (长序列): compute-bound for N >> 256d (因为 O(N²d) FLOPs >> O(Nd) IO)
- Decode (单token): always memory-bound (因为 O(Nd) FLOPs ≈ O(Nd) IO)

实际上 FlashAttention 的 IO 复杂度:
```
IO = O(Nd²/B_r + N²d²/B_r × (SRAM_size/d)^(-1))
```
简化后 ≈ O(N²d²/M) where M = SRAM size

而标准 ≈ O(N²d + Nd²)

当 N 很大时, FlashAttention IO 从 O(N²) 变为 O(N²/M), M >> d → 更优

## 5. Backward Pass: 不存储 P, 重计算 S

**FlashAttention backward 的关键**: 不存储 P = softmax(S) 到 HBM

而是: 在 backward 时重新计算 S 和 P (从 Q, K, V 在 SRAM 中重算)

```
Backward IO:
- 读 dO: B × H × N × d × 4 bytes
- 读 Q, K, V: 各 B × H × N × d × 4 bytes × Tr 次 (重计算)
- 写 dQ, dK, dV: 各 B × H × N × d × 4 bytes

总计: ≈ O(N²d²/M) (同 forward) vs 标准 O(N²d)
```

**代价**: 多读了 Q, K, V 各一次, 但省了读写 O(N²) 的 P 矩阵
**结论**: 当 SRAM >> d (192KB >> 64×4=256B), 重计算比存储 P 更省 IO

**梯度公式**:
```
dV = P^T @ dO                    # P 需要重计算
dS = (dO @ V^T) × P - (dO @ O^T) × P  # row-wise: dS = P × (dV - sum_row(dO×O))
dQ = dS @ K
dK = dS^T @ Q
```

其中 `dS = P × (dO@V - rowsum(dO@O))` 是 row-wise 计算 (每行独立)

## 6. FlashAttention-2 优化 (Dao, 2023)

**FlashAttention-1 vs FlashAttention-2**:
- FA-1: 逐块串行 (内循环 j 遍历所有块), 每块需要同步
- FA-2: 改进 thread mapping → 减少 non-matmul FLOPs, 2x faster
  - 在 GQA 模型上: 将 Q 的 head 分配到不同 warp, 每个 warp 处理 K/V 的一组 head
  - 减少 warp 间同步开销

**关键优化**:
1. **减少 non-matmul FLOPs**: softmax 和 rescale 操作占比从 ~25% 降到 ~10%
2. **并行化**: 对 Prefill, 并行化 over sequence length; 对 Decode, 并行化 over batch+heads
3. **Work partitioning**: 每个 warp 处理固定 Q rows, 减少共享内存读写

## 7. Triton 实现 (vLLM) 关键点

vLLM 的 Triton decode attention kernel:
- **两阶段**: Split-KV (将 KV 分块并行处理) + Reduce (合并结果)
- **在线 Softmax**: 同 FlashAttention 的 m/l 累积模式
- **Paged KV**: 通过 page_table 间接寻址 KV cache (vLLM 的 block 管理)
- **GQA 支持**: Grouped variant, KV head 共享 → load 降低 87.5% (8 KV heads vs 64 Q heads)
- **MLA variant**: IS_MLA 判断 + BLOCK_DPE 对 DeepSeek-V2 MLA attention

## 8. 关键 Takeaways

1. **Online softmax 是核心**: 不需要完整 S 矩阵就能计算 softmax — 通过 max/l 增量累积
2. **IO 复杂度**: O(N²/M) vs O(N²) — SRAM 大小 M 是关键参数, 越大越省 IO
3. **重计算 vs 存储**: backward 重计算 S/P (多读 Q,K,V) 比存储 O(N²) 的 P 更省 IO
4. **Decode 是 memory-bound**: 每步只计算 1 个 Q token, FLOPs = O(Nd) ≈ IO, 无法优化
5. **Prefill 是 compute-bound** (长序列): FLOPs = O(N²d) >> IO, GPU 利用率高
6. **Tiling 是通用范式**: 不仅 Attention, 任何 O(N²) 算法 (如 batched GEMM) 都可用 tiling
7. **GQA 是 Attention 的内存优化**: 减少 KV 头数 → KV cache 减少, 但计算不变