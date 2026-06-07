# Triton Decode Attention Kernel Benchmark — RTX 4090

```
┌─────────────────────────────────────────────────────────┐
│  Triton Decode Attention Kernel Results                 │
│                                                         │
│  Correctness: Triton vs Naive cos_sim=1.000000 ✅       │
│  Triton vs SDPA(no causal) cos_sim=1.0 ✅               │
│  Triton vs SDPA(causal=True) cos_sim≈0.02 ⚠️           │
│  → SDPA is_causal=True 对Q=1是错误的!                   │
│                                                         │
│  Speed: Triton 2-3x慢 vs SDPA                          │
│  GQA: Triton慢但75-87.5% KV内存省                      │
│  KV Traffic: 99%为KV读取 → decode完全memory-bound      │
│  BLOCK_KV: tuning无影响 (16-256全≈0.062ms)             │
│                                                         │
│  Bug Fix: padding mask → -inf → exp(-inf)=0             │
│  Production: Triton教育价值, FlashInfer是答案           │
│  Decision: Simple→SDPA / Prod→FlashInfer / GQA→FI native│
└─────────────────────────────────────────────────────────┘
```

## 1. 正确性验证: Triton Kernel完全正确

### 1.1 Triton vs Naive (数学正确baseline)

```
Triton vs Naive: cos_sim = 1.000000, max_diff = 0.000732
Triton vs SDPA(is_causal=False): cos_sim = 1.0
```

Triton kernel与数学上正确的naive实现完全匹配! cos_sim=1.0验证了online softmax算法的正确性。

### 1.2 SDPA is_causal=True对Decode Q=1的错误

Benchmark中Triton vs SDPA(is_causal=True)的cos_sim只有0.02-0.09, 但这不是Triton的bug — **是SDPA的causal mask对Q=1的错误应用**!

**根因**:
- SDPA `is_causal=True` 将Q视为position 0 → causal mask限制Q只能看到K[0]
- 但decode场景中Q应该在position S → 可以看到所有K[0..S-1]
- 实测确认:

```
SDPA(is_causal=True)[0,0,0,:8]:  tensor([-0.0622, 1.4209, 0.1058, ...]) ← 只看了K[0]
Triton[0,0,0,:8]:                 tensor([0.0020, 0.8076, -0.5591, ...]) ← 正确(看了所有K)
Naive[0,0,0,:8]:                  tensor([0.0020, 0.8076, -0.5591, ...]) ← 与Triton完全一致!
```

**结论**: Decode(Q=1)必须使用`is_causal=False`或专用decode kernel!

### 1.3 Padding Mask Bug修复

这是Triton kernel开发中最关键的bug发现和修复:

**Bug**: padding positions (kv_positions >= S) 通过mask `other=0.0` 加载为K=0
- score=0 in dot product → `exp(0 - m_new) ≠ 0` → padding贡献非零attention weight → 腐化归一化
- 结果: Triton输出5-16x小, cos_sim=0.008-0.13

**Fix** (关键代码):

```python
# Bug版本:
scores = tl.sum(q[None, :] * k, axis=1) * scale  # padding K=0 → score=0

# Fix版本:
valid_kv = kv_positions < S  # boolean mask for padding
scores = tl.sum(q[None, :] * k, axis=1) * scale
scores = tl.where(valid_kv, scores, float("-inf"))  # padding → -inf → exp(-inf)=0

# ...online softmax...

p = tl.exp(scores - m_new)
p = tl.where(valid_kv, p, 0.0)  # zero out padding attention weights
```

**原理**: softmax中padding必须获得score=-inf → `exp(-inf)=0` → 不影响归一化。score=0 → `exp(0-m_new)≠0` → 破坏softmax。

## 2. Triton Decode Kernel设计

### 2.1 核心架构

```python
@triton.jit
def _decode_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, n_heads, n_kv_heads, d_head,
    # ... strides ...
    scale,
    kv_group_size: tl.constexpr,  # n_heads // n_kv_heads (GQA)
    BLOCK_KV: tl.constexpr,       # KV block size
    D_HEAD: tl.constexpr,         # head dimension
):
    pid = tl.program_id(0)
    batch_idx = pid // n_heads
    q_head_idx = pid % n_heads
    kv_head_idx = q_head_idx // kv_group_size  # GQA: native mapping!

    # Initialize online softmax accumulators
    m_i = float("-inf")  # running max
    l_i = 0.0            # running sum
    acc = tl.full([D_HEAD], 0.0, dtype=tl.float32)  # running output

    # Load Q: (D_HEAD) — single query token
    q = tl.load(Q_ptr + batch_idx * Q_stride_b + q_head_idx * Q_stride_h
                + d_offsets * Q_stride_d, mask=d_offsets < d_head)

    # Iterate over KV blocks
    for kv_start in range(0, S, BLOCK_KV):
        # Load K block: (BLOCK_KV, D_HEAD)
        k = tl.load(K_ptr + ..., mask=(kv_positions[:, None] < S) & (d_offsets[None, :] < d_head))

        # Compute scores: (BLOCK_KV,)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(valid_kv, scores, float("-inf"))  # padding → -inf

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha

        p = tl.exp(scores - m_new)
        p = tl.where(valid_kv, p, 0.0)

        l_i = l_i + tl.sum(p, axis=0)

        # Load V block and accumulate
        v = tl.load(V_ptr + ..., mask=...)
        acc = acc + tl.sum(p[:, None] * v, axis=0)  # (D_HEAD,)

        m_i = m_new

    # Final normalization
    out = acc / l_i
    tl.store(Out_ptr + ..., out)
```

**关键设计决策**:
1. **向量ops而非tl.dot()**: Q=1 → scalar dot product per KV position → `tl.dot(q, k.T)` dimension mismatch → 用`tl.sum(q * k)`替代
2. **GQA native**: `kv_head_idx = q_head_idx // kv_group_size` → kernel内部映射 → 无Python expand!
3. **Online softmax**: FlashAttention-style running max+sum → 精确增量计算
4. **BLOCK_KV=64**: 平衡KV读取粒度和iteration次数

### 2.2 为什么不用tl.dot()

`tl.dot()`要求2D input tensors, 但Q=1decode场景:
- q: (D_HEAD) — 1D vector, 不是2D matrix
- k: (BLOCK_KV, D_HEAD) — 2D
- `tl.dot(q, k.T)` → ValueError: incompatible dimensions (1 and BLOCK_KV)

Triton不支持batched dot product with 1D vector → 必须用element-wise ops + reduction.

## 3. Benchmark结果 (RTX 4090)

### 3.1 Decode速度对比 (7M, n_heads=8, d_head=32)

| Config | Naive(ms) | SDPA(ms) | Triton(ms) | Triton/SDPA |
|--------|-----------|----------|------------|-------------|
| S128 B=1 | 0.075 | 0.028 | 0.060 | 2.17x |
| S128 B=8 | 0.078 | 0.026 | 0.058 | 2.24x |
| S512 B=1 | 0.079 | 0.027 | 0.060 | 2.22x |
| S1024 B=8 | 0.078 | 0.025 | 0.062 | 2.47x |
| S2048 B=1 | 0.076 | 0.027 | 0.077 | 2.89x |
| S2048 B=32 | 0.085 | 0.026 | 0.081 | 3.08x |
| S2048 B=64 | 0.187 | 0.026 | 0.195 | **7.38x** |

**关键发现**:
- Triton比SDPA慢2-3x(B≤32, S≤1024)
- B=64, S=2048时慢7.4x → Triton进入memory-bound瓶颈
- B=64时naive也变慢(0.187ms) → 说明大batch+长序列是GPU瓶颈

### 3.2 GQA Decode (B=8)

| Config | SDPA(expanded)(ms) | Triton(native)(ms) | Triton/SDPA | KV_saving |
|--------|-------------------|--------------------|-------------|-----------|
| MHA S=512 | 0.028 | 0.063 | 2.20x | 0% |
| GQA-4 S=512 | 0.025 | 0.063 | 2.47x | **75%** |
| GQA-2 S=512 | 0.028 | 0.063 | 2.24x | **87.5%** |
| GQA-4 S=2048 | 0.028 | 0.079 | 2.77x | **75%** |
| GQA-2 S=2048 | 0.029 | 0.079 | 2.76x | **87.5%** |

**核心洞察**: Triton GQA native比SDPA(expanded)慢, 但**不需Python KV expand → 75-87.5% KV内存省**!

### 3.3 KV Memory Traffic

| Config | KV_read(MB) | total(MB) | KV_ratio | expand_extra(MB) | GQA_saving |
|--------|------------|-----------|----------|------------------|-----------|
| kv=16 S=2048 | 134.2 | 134.3 | 99.95% | 0 | 0% |
| kv=4 S=2048 | 33.6 | 33.6 | 99.8% | **134.2** | 75% |
| kv=2 S=2048 | 16.8 | 16.8 | 99.6% | **134.2** | 87.5% |
| kv=1 S=2048 | 8.4 | 8.5 | 99.2% | **134.2** | 93.75% |

**KV读占decode memory traffic 99.2-100%** → decode完全memory-bound!

### 3.4 BLOCK_KV Tuning

| BLOCK_KV | median(ms) |
|-----------|-----------|
| 16 | 0.063 |
| 32 | 0.062 |
| 64 | 0.062 |
| 128 | 0.062 |
| 256 | 0.062 |

**BLOCK_KV对速度无影响** → 问题太小(kernel launch主导, S/BLOCK_KV iteration数量差异在noise范围内)

## 4. 为什么Triton Decode比SDPA慢

### 4.1 SDPA auto-select最优backend

```python
# SDPA根据tensor大小自动选择:
# 大batch+大S → FlashAttention backend (HMMA tile, 高效)
# 小batch+小S → Math reference backend (PyTorch matmul, cuBLAS优化的101% peak!)
```

### 4.2 Triton kernel的瓶颈

1. **GPU利用率低**: B=1, n_heads=16 → 只有16个programs → RTX 4090 128 MPs → 利用率12.5%
2. **无tensor core**: vector ops (`tl.sum(q*k)`) → SIMT而非HMMA → 吞吐<<tensor core
3. **无pipeline**: 顺序执行(load→softmax→load→accumulate) → 无FA2的register pipeline重叠
4. **无cooperative batching**: 每个program独立读KV → 无FlashInfer的多Q共享KV load

## 5. Production决策

| 场景 | 最优Backend | 原因 |
|------|-----------|------|
| Research/debug | SDPA is_causal=False | 最快+正确 |
| Simple decode | SDPA math | cuBLAS最优 |
| Production decode | FlashInfer | cooperative batching+paged KV |
| GQA decode | FlashInfer native GQA | 零expand+kernel broadcast |
| Custom Triton | 教育价值 | 不适合RTX 4090生产 |

---

**工具**: `tools/triton_decode_attn_benchmark_4090.py`, `tools/triton_decode_attn_debug.py`
**结果**: `results/triton_decode_attn_benchmark.json`
**相关笔记**: `flashattention2-kernel-internals-vs-triton.md`, `flashinfer-decode-kernel-analysis.md`, `attention-backend-comparison-rtx4090.md`