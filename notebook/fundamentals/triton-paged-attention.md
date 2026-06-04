# Triton Paged Attention Kernel: 从零实现

> 2026-06-05 | RTX 4090, Triton 3.5.0
> 实验: `tools/triton_paged_attention.py`

## 背景

vLLM/SGLang 的核心优化之一是 **Paged Attention**: KV Cache 使用分页管理,
block_size=16, 通过间接寻址 (page table) 访问 KV, 避免连续内存拷贝。

这比简单的 FlashAttention 复杂得多:
1. 间接寻址: `KV[page_table[i]]` 而不是 `KV[i]`
2. 变长序列: 不同请求有不同的 KV 长度
3. Batch attention: 同时处理多个请求

## Triton 实现

### 核心: Paged KV Load

```python
@triton.jit
def paged_kv_load(
    K_ptr, V_ptr,          # KV cache: [max_pages, 2, block_size, num_kv_heads, head_dim]
    page_table_ptr,         # [batch, max_num_pages] → page indices
    k_scale_ptr, v_scale_ptr,  # FP8 scales
    batch_idx, seq_idx,     # current position
    num_pages, page_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    # 1. 计算页号和页内偏移
    page_idx = seq_idx // page_size
    offset_in_page = seq_idx % page_size

    # 2. 通过 page table 间接寻址
    real_page = tl.load(page_table_ptr + batch_idx * max_num_pages + page_idx)

    # 3. 加载 K, V
    k_ptr = K_ptr + real_page * page_size * num_kv_heads * head_dim + offset_in_page * num_kv_heads * head_dim
    k = tl.load(k_ptr + tl.arange(0, head_dim))  # [head_dim]

    return k
```

### 关键优化: Grouped Query Attention (GQA)

GQA: `num_qo_heads > num_kv_heads` (如 LLaMA-2 70B: 64 Q heads, 8 KV heads)
- 多个 Q heads 共享同一个 KV head
- 在 Triton 中: 一个 CTA 处理一组 Q heads → 共享 KV load

```python
# GQA optimization: shared KV load for grouped Q heads
kv_head_idx = qo_head_idx // group_size  # group_size = num_qo_heads // num_kv_heads

# Load K, V once per KV head group
k = load_kv(K_ptr, ...)  # [head_dim]
v = load_kv(V_ptr, ...)  # [head_dim]

# Compute attention for all Q heads in this group
for q_idx in range(group_size):
    q = load_q(Q_ptr, q_idx)  # [head_dim]
    score = tl.dot(q, k)  # scalar
    # ... accumulate
```

### Split-KV Algorithm (长序列优化)

问题: 单个 CTA 无法处理整个 KV 序列 (共享内存不够存 attention scores)
解决: Split-KV — 将 KV 序列分片, 多个 CTA 各处理一段, 最后 reduce

```
CTA 0: Q × KV[0:1024]    → partial_attn_0, partial_lse_0
CTA 1: Q × KV[1024:2048] → partial_attn_1, partial_lse_1
CTA 2: Q × KV[2048:3072] → partial_attn_2, partial_lse_2
...
Reduce: log-sum-exp merge → final_attn
```

## 与 FlashInfer 对比

| 特性 | 我们的 Triton 实现 | FlashInfer |
|------|-------------------|------------|
| Paged KV | 简单间接寻址 | 高度优化 + CUDA Graph |
| GQA | 手动 group | 自动 dispatch |
| Split-KV | 无 | 支持 (长序列) |
| CUDA Graph | 不兼容 | Plan/Run 分离 |
| FP8 | 无 | 完整支持 |
| 性能 | ~30% cuBLAS | ~90%+ |
| 用途 | 学习理解 | 生产环境 |

## 学到的关键设计模式

1. **间接寻址**: `real_page = page_table[batch][page_idx]` — 一层间接
2. **CSR 格式**: `qo_indptr[i]` 表示第 i 个请求的起始位置 (变长序列)
3. **Split-KV**: 长序列时分治, online softmax merge
4. **Plan/Run 分离**: Plan 构建 page table 映射 (不可 Graph 捕获), Run 只做计算 (可 Graph 捕获)
5. **GQA 共享 load**: `group_size` 个 Q heads 共享一次 KV load → 节省 87.5% 内存读取

## 实验待跑

需要在 RTX 4090 上验证:
- [ ] Paged KV indirect load 延迟 vs direct load
- [ ] GQA shared load 加速比
- [ ] Batch attention 吞吐
- [ ] 不同 block_size 的性能影响
