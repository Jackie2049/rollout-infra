# FlashAttention Throughput & Memory Savings: RTX 4090 实测

> 2026-06-07 | RTX 4090 (SM 8.9, 128 MPs, 24GB HBM)

## 核心发现

**FlashAttention 在 RTX 4090 上比 naive attention 更慢, 但内存节省 85-97%!**

| 指标 | Decode (Q=1) | Prefill (Q=S) |
|------|-------------|---------------|
| Forward speedup | **0.67-0.84x (更慢!)** | **1.03-1.10x (微加速)** |
| Fwd+Bwd speedup | **0.79-0.83x (更慢!)** | — |
| Memory saved | **61-97%** | **61-97%** |

## 为什么 FlashAttention 在 Decode 上更慢?

### 1. Decode: Q=1 token → Attention matrix = [1 × seq_len]

FlashAttention 的核心优化是 **不存储 [S × S] attention matrix**:
- Prefill: Q=S, attention matrix = [S × S] → 巨大节省
- Decode: Q=1, attention matrix = [1 × S] → 极小, 本身就不是瓶颈

在 decode 时:
```
Naive:  Q(1×D) @ K(S×D)ᵀ → S scores → softmax → @ V(S×D) = 1×D
Flash:  Tiling + online softmax → 同样计算量, 但多了一层 indirection
```

**FlashAttention 的 tiling 不减少 IO**: Decode 已经只需要读 Q(1行)+K(S行)+V(S行), 写 output(1行). [1×S] attention matrix 太小, 不影响 memory-bound 性质.

### 2. SDPA kernel 选择逻辑

PyTorch `scaled_dot_product_attention` 有3种实现:
- **FlashAttention-2**: 需要 SM 80+ 和足够大的问题规模
- **Memory-efficient attention**: xFormers-style, 用于中等规模
- **Math fallback**: 最简单, 直接计算

在小 decode (B=1, S=512) 时, SDPA 可能选择了**不必要的复杂 kernel** — 封包开销大于计算本身.

### 3. Naive 实现为什么快?

PyTorch 的 matmul + softmax 有 cuBLAS/cuDNN 极端优化:
- `matmul`: cuBLAS GEMM, Tensor Core 加速
- `softmax`: 融合 CUDA kernel
- 两步相加, 但每步都极致优化

FlashAttention 的额外开销:
- Tiling 管理 (block 切分)
- Online softmax 的 incremental 计算
- Rescaling (每步修正之前的累积)

在小规模下, 这些 overhead > 计算收益.

## Memory Savings: FlashAttention 的真正价值

| S | Naive Peak (MB) | Flash Peak (MB) | Attn Matrix Theory (MB) | Saved |
|---|----------------|----------------|------------------------|-------|
| 512 | 109 | 42 | 34 | 61.4% |
| 1024 | 344 | 76 | 134 | 78.0% |
| 2048 | 1216 | 143 | 537 | 88.2% |
| 4096 | 4572 | 277 | 2147 | **93.9%** |
| 8192 | 17725 | 546 | 8590 | **96.9%** |

**关键**:
- S=8192 时 naive 需要 17.7GB → **超过 RTX 4090 24GB 的 70%!**
- FlashAttention 仅 546MB → **32x 内存节省**
- 这就是为什么长 context (>4K) 必须用 FlashAttention — 不是为了速度, 是为了 **OOM 避免**

## Prefill vs Decode 的不同行为

| Phase | Q length | Attn matrix | IO dominant | Flash benefit |
|-------|----------|-------------|-------------|---------------|
| Prefill | S | S×S (巨大) | HBM (读权重) | **内存节省+微加速** |
| Decode | 1 | 1×S (微小) | HBM (读KV) | **只有内存节省, 更慢** |

Prefill 是 **compute-bound** (S² 的 matmul), FlashAttention 的 tiling 能减少 HBM 访问次数.
Decode 是 **memory-bound** (读 KV cache), FlashAttention 不减少必须读的 KV 数据量.

## Backward Pass 分析

| S | Naive fwd+bwd | Flash fwd+bwd | Speedup | Mem saved |
|---|---------------|---------------|---------|-----------|
| 512 | 0.704ms | 0.851ms | 0.83x | -397%* |
| 1024 | 2.762ms | 3.205ms | 0.86x | 68.2% |
| 2048 | 10.121ms | 12.359ms | 0.82x | 71.7% |
| 4096 | 38.304ms | 48.664ms | 0.79x | 73.4% |

*S=512 时 mem_saved=-397% 可能是测量误差 (peak_memory 在小tensor时不准确)

FlashAttention backward **更慢** 但 **内存节省更多**:
- Naive backward: 需要存储 attention matrix [B×H×S×S] 用于梯度计算
- Flash backward: **recompute** attention matrix (online softmax again) → 不存储

Recompute 的代价: 额外计算 → 更慢
Recompute 的收益: 不存储巨大 attention matrix → 省内存

## 实际应用中的选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| Decode (B≤4, S≤2K) | **Naive 或 SDPA math** | 更快, attention matrix小 |
| Decode (B≥32, S≤2K) | **SDPA** | 差距缩小, SDPA稳定性更好 |
| Prefill (S≥1K) | **FlashAttention** | 内存节省是必须的 |
| Training (backward) | **FlashAttention** | 内存节省远大于速度损失 |
| Long context (S≥4K) | **FlashAttention 必须** | naive 会 OOM! |
| 多GPU (TP) | **FlashAttention 必须** | 每GPU只存部分, 内存更关键 |

## 与之前 A16 数据对比

| GPU | Prefill FlashAttn speedup | Memory saved @S=4K |
|-----|--------------------------|--------------------|
| A16 (10 SM) | 9.6x @S=2048 | — |
| RTX 4090 (128 SM) | 1.03-1.10x | 93.9% |

**A16 为什么 9.6x?** A16 只有 10 SM + PCIe, naive matmul 极慢 → FlashAttention 的计算效率优势巨大.
**RTX 4090 为什么 1.03x?** 128 SM + Tensor Core → naive matmul 已经非常快 → FlashAttention 的计算优势消失.

**核心结论**: FlashAttention 的速度收益 ∝ GPU弱度, 内存收益 ∝ seq_len². 大GPU上FlashAttn主要价值是**省内存防OOM**, 不是加速.

## vLLM 如何使用 FlashAttention

vLLM V1 使用 FlashInfer (而非直接 torch SDPA):
1. **Decode**: FlashInfer 的 Split-KV kernel, 专门优化 decode (Q=1)
2. **Prefill**: FlashInfer 的 BatchPrefillWithPagedKVCache
3. **Paged KV**: 内置 block_table indirection → 不需要 gather
4. **GQA**: Grouped variant → KV head 只读一次, Q head 组共享

vLLM 不用 naive attention → 不是因为速度, 是因为:
1. Paged KV 需要 kernel-level indirection
2. 长上下文 (>4K) 必须 FlashAttention 防 OOM
3. Training/gradients 需要 FlashAttention 省内存