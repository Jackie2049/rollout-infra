# FlashAttention 原理深度解析

> IO-Aware Tiling：为什么 FlashAttention 比标准 Attention 快 2-4x

## 1. 核心问题：标准 Attention 的瓶颈

标准 attention 计算：
```
S = Q @ K^T     # [N, d] @ [d, N] → [N, N]
P = softmax(S)  # [N, N]
O = P @ V       # [N, N] @ [N, d] → [N, d]
```

**问题**：中间矩阵 S 和 P 都是 N×N，必须写入 HBM 再读回来。
- N=8192, d=128 时，单个 head 的 S 矩阵 = 8192×8192×2B = 256MB
- 标准流程需要 3 次读取 + 3 次写入 N² 规模数据

## 2. 关键洞察：Memory-Bandwidth Bound ≠ Compute Bound

### GPU 内存层次 (A100)

| 层级 | 容量 | 带宽 |
|------|------|------|
| HBM (显存) | 40-80 GB | ~2 TB/s |
| SRAM (每SM) | ~192 KB | ~19 TB/s |
| L2 Cache | ~40 MB | ~数 TB/s |

**A100 FP16 算力：312 TFLOPS**
- 计算 QK^T (2N²d FLOPs, N=8192, d=128): ~0.055 ms
- 从 HBM 读 256MB attention 矩阵 (2 TB/s): ~0.128 ms
- **结论：等待数据传输的时间 > 计算时间**

**FlashAttention 的核心洞察：减少 HBM 访问次数比减少 FLOPs 更重要！**

## 3. FlashAttention-1：IO-Aware Tiling 算法

### 3.1 核心思想

把 attention 切分为小块（tile），每块在 SRAM 中完成计算，**永远不在 HBM 中 materialize N×N 矩阵**。

```
For each query block Q_i (size B_q × d):
    初始化: o_i = 0, l_i = 0, m_i = -inf

    For each KV block (K_j, V_j) (size B_kv × d):
        s_ij = Q_i @ K_j^T              # [B_q × B_kv] — 在 SRAM 中
        m_ij = rowmax(s_ij)              # [B_q]
        p_ij = exp(s_ij - m_ij)          # [B_q × B_kv]
        l_ij = rowsum(p_ij)              # [B_q]

        # 更新运行统计量
        m_new = max(m_i, m_ij)
        l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij

        # 更新运行输出
        o_i = (l_i * exp(m_i - m_new) * o_i + p_ij @ V_j) / l_new

        m_i = m_new
        l_i = l_new

    Write o_i to HBM  # 只写一次最终结果
```

### 3.2 内存占用对比

| 方法 | HBM 读取 | HBM 写入 | 总访问量 |
|------|----------|----------|----------|
| 标准 | O(Nd + N²) | O(N² + Nd) | Θ(N² + Nd) |
| FlashAttention | Q+K+V 各读一次 + O 写一次 | O only | Θ(N²d²/M + Nd) |

加速因子 ≈ M/d² (M = SRAM 大小)，约 12x 更少的 HBM 访问。

### 3.3 SRAM 足迹

```
SRAM ≈ B_q × B_kv + B_q × d + B_kv × d + 3 × B_q
典型值: B_q = B_kv = 64, d = 128 → ~200KB (适合 SRAM)
```

## 4. Online Softmax：数学基础

### 4.1 标准 softmax 需要 3 遍扫描

```
Pass 1: m = max(x)           # 数值稳定
Pass 2: d = sum(exp(x - m))  # 归一化常数
Pass 3: o = exp(x - m) / d   # 输出
```

### 4.2 Online Softmax：单遍扫描

逐元素更新运行统计量：

```python
m_0 = -inf; d_0 = 0; o_0 = 0

for i in range(N):
    m_i = max(m_{i-1}, x_i)
    d_i = d_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
    o_i = o_{i-1} * (d_{i-1}/d_i) * exp(m_{i-1} - m_i) + (exp(x_i - m_i)/d_i) * V[i]

# 最终 o_N = softmax(x) @ V (精确值！)
```

**关键**：当 m 增大时，所有之前累积的指数需要乘以 `exp(m_old - m_new)` 进行 rescale。

### 4.3 块级扩展 (FlashAttention)

将 online softmax 从逐元素扩展到逐块：

```python
s_ij = Q_i @ K_j^T                    # 块级分数
m_ij = rowmax(s_ij)                    # 本块最大值
p_ij = exp(s_ij - m_ij)               # 本块 softmax 分子
l_ij = rowsum(p_ij)                   # 本块归一化常数

# 合并运行统计量
m_new = max(m_i, m_ij)
l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
o_new = [l_i * exp(m_i - m_new) / l_new] * o_i
      + [exp(m_ij - m_new) / l_new] * (p_ij @ V_j)
```

## 5. FlashAttention-2：更好的工作分配

FA2 的三大改进：

1. **减少非矩阵乘法 FLOPs**
   - Tensor Core GEMM 比非 GEMM 快 ~16x
   - 重排操作顺序，最大化 GEMM 占比

2. **沿序列长度并行**
   - FA1 只在 batch 和 head 上并行
   - FA2 额外在序列长度上并行 → 更好的 GPU 占用率

3. **更好的 warp 内工作分配**
   - Q block 保持在寄存器中（最快）
   - K/V block 流过 shared memory
   - 内循环迭代 K/V block 而非 Q block

**性能**：比 FA1 快 2x，达到 A100 理论峰值 50-73%。

## 6. FlashAttention-3：Hopper 专用优化

FA3 针对 H100 的三大特性：

1. **Warp 特化 (异步)**
   - 一个 warpgroup 做 GEMM
   - 另一个做 softmax/rescale
   - 并发运行！

2. **Ping-pong 调度**
   - 两个 warpgroup 交替：
   - Warpgroup A 算 GEMM 时，Warpgroup B 做上一 tile 的 softmax

3. **Warpgroup 内交错**
   - 当前 tile 的 GEMM epilogue 与下一 tile 的 GEMM prologue 重叠

**性能**：比 FA2 快 1.5-2x，H100 FP16 达到 740 TFLOPS/s (75%)。

## 7. 对比总结

| 特性 | FA1 (2022) | FA2 (2024) | FA3 (2024) |
|------|-----------|-----------|-----------|
| 核心 | IO-aware tiling | + 工作分配优化 | + 异步重叠 |
| GPU | Ampere (A100) | Ampere | Hopper (H100) |
| 利用率 | ~25-40% | 50-73% | 75% |
| 内存 | O(N) | O(N) | O(N) |
| 精度 | FP16/BF16 | FP16/BF16 | FP16/BF16 + FP8 |

## 8. 与 AI Infra 的关联

- **vLLM PagedAttention**：在 FA 基础上增加分页 KV Cache 管理
- **Megatron-LM TP**：每个 GPU 独立计算不同 head 的 attention，自然适配 FA
- **Sequence Parallelism**：FA 的 O(N) 内存使超长序列训练成为可能
- **推理优化**：FA 是 continuous batching 和 prefix caching 的基础

## 参考

- [FlashAttention-1](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (ICLR 2024)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024)
- [Tri Dao's Blog](https://tridao.me/blog/2024/flash3/)
- [Online Softmax to FlashAttention 讲义](https://courses.cs.washington.edu/cse599m/)
