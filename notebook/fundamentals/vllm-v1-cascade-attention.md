# vLLM V1 Cascade Attention: Prefix+Suffix 分离计算 + LSE 合并

> 2026-06-07 | vLLM V1 cascade attention 架构与算法分析

## 核心思想

Cascade Attention = 把 KV 序列分成 **Prefix (共享) + Suffix (私有)** 两部分,
分别计算 attention, 然后用 LSE rescale 合并。

**论文**: Section 2.2 of arxiv 2501.01005 (Cascade Attention)

## 算法流程

```
1. Prefix Attention:
   Q × Prefix_KV → prefix_output + prefix_lse (双向, non-causal)

2. Suffix Attention:
   Q × Suffix_KV → suffix_output + suffix_lse (causal)

3. Merge (LSE rescale):
   max_lse = max(prefix_lse, suffix_lse)
   prefix_scale = exp(prefix_lse - max_lse) / (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
   suffix_scale = 1 - prefix_scale
   output = prefix_output × prefix_scale + suffix_output × suffix_scale
```

### 为什么有效?

**Prefix 是所有请求共享的**:
- Prefix KV 只需加载一次, 所有请求共享 → 节省大量内存带宽
- Prefix attention 是 **双向的** (non-causal), 因为prefix token之间没有因果限制
- Prefix block_table 只取第一行 (block_table[:1]) → 所有请求看到相同的prefix blocks

**Suffix 是每个请求私有的**:
- Suffix KV 每个请求不同 → 需要per-request block_table
- Suffix attention 是 **causal** (因果的), 因为suffix token只能看到之前的token
- Suffix block_table 排除prefix blocks → block_table[:, num_common_kv_blocks:]

## 触发条件

`use_cascade_attention()` 的决策逻辑:

1. **Prefix 太短 (<256 tokens)**: 不值得 → return False
2. **ALiBi/sliding_window/chunked**: 不支持 → return False
3. **请求太少 (<8)**: 不值得 → return False
4. **DCP (disaggregated)**: 禁用 → return False
5. **非FlashDecoding场景**: 总是使用cascade → return True
6. **FlashDecoding场景**: 用性能模型比较 CTA 数量:
   - cascade_ctas = num_heads × ceil(num_tokens / q_tile)
   - flash_decoding_ctas = num_reqs × num_kv_heads × ceil(n_q_per_kv / q_tile) × num_prefix_tiles
   - 如果 cascade 更快 → return True

## FlashInfer 实现: MultiLevelCascadeAttentionWrapper

```python
# 2层cascade (prefix + suffix)
cascade_wrapper = MultiLevelCascadeAttentionWrapper(
    num_levels=2,  # prefix(0) + suffix(1)
    workspace_buffer=workspace,
    kv_layout=get_kv_cache_layout()
)

# Plan: 为每层配置索引
cascade_wrapper.plan(
    qo_indptr_arr=[shared_qo_indptr, qo_indptr],      # [prefix, suffix]
    paged_kv_indptr_arr=[shared_kv_indptr, paged_kv_indptr],
    paged_kv_indices_arr=[shared_kv_indices, paged_kv_indices],
    paged_kv_last_page_len=[shared_last_page_len, paged_kv_last_page_len],
)

# Run: 单次调用, 内部自动做 prefix→suffix→merge
output = cascade_wrapper.run(query, kv_cache)
```

**注意**: FlashInfer 的 cascade 当前 **被禁用** (注释掉: `# return use_cascade_attention(...)`)!

## Flash Attention 实现: cascade_attention()

```python
def cascade_attention(output, query, key_cache, value_cache, ...):
    # Step 1: Process shared prefix (non-causal, bidirectional)
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query, k=key_cache, v=value_cache,
        causal=False,                           # ← 双向!
        block_table=block_table[:1],             # ← 只看第一行的prefix blocks
        seqused_k=prefix_kv_lens,               # ← prefix长度
        max_seqlen_k=common_prefix_len,          # ← prefix最大长度
    )

    # Step 2: Process suffix per query (causal)
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query, k=key_cache, v=value_cache,
        causal=True,                            # ← causal!
        block_table=block_table[:, num_common_kv_blocks:],  # ← 排除prefix blocks
        seqused_k=suffix_kv_lens,               # ← suffix长度
        max_seqlen_k=max_kv_len - common_prefix_len,
    )

    # Step 3: Merge prefix and suffix
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

**关键**: Prefix 和 Suffix 使用同一个 Q, 但不同的 KV range 和不同的 causal 设置!

## Triton Backend: 不支持

TritonAttentionBackend 的 `use_cascade_attention()` 始终返回 False。
但 TritonBackend 的 metadata 包含 cascade 相关字段 (use_cascade, common_prefix_len, cu_prefix_query_lens, prefix_kv_lens, suffix_kv_lens), 说明有计划未来支持。

目前 Triton unified kernel 的 3D mode 有类似概念 (分段softmax→全局reduce), 但没有 prefix/suffix 的分离计算优化。

## 与 SGLang RadixAttention 对比

| | vLLM Cascade | SGLang RadixAttention |
|---|---|---|
| **粒度** | 整个batch共享prefix | 每个请求的RadixTree prefix |
| **触发** | common_prefix_len > 256 | 任何prefix匹配 |
| **Prefix类型** | 所有请求的**共同**prefix | 每个请求**独立**的prefix |
| **KV加载** | Prefix只加载1次 (所有请求共享) | Prefix每个请求独立加载 |
| **合并** | merge_attn_states (LSE) | 直接extend KV range |
| **适用** | 多轮对话(相同prompt) | 混合prompt场景 |

vLLM cascade 更激进: 整个batch共享prefix → prefix KV 只需1次加载 → 节省更多带宽。
SGLang RadixAttention 更灵活: 每个请求独立prefix → 支持任何prefix匹配场景。

## 收益分析

**理论**: Prefix复用率 = common_prefix_len × num_requests / (total KV × num_requests)
- 10个请求, prompt=1024 tokens: 复用率 = 1024×10 / (1024×10) = 100% prefix部分
- Prefix只读1次 vs 读10次 → 节省9/10 = **90% prefix带宽**

**前提**: 所有请求必须有相同的prefix → GRPO/RLHF训练场景 (n个请求共享同一个prompt)

## 当前状态

- **FlashInfer**: 有MultiLevelCascadeAttentionWrapper但被禁用 (TODO注释)
- **FlashAttention (fa3)**: 有cascade_attention实现, 通过use_cascade_attention()启发式决定
- **Triton**: 不支持 (计划中)
- **触发阈值**: prefix ≥ 256 tokens, requests ≥ 8

## 数学证明: LSE Merge 等价于全序列Softmax

设 S = {prefix_scores, suffix_scores}, M = max(S):
- prefix_lse = log(Σ_prefix exp(s - M_prefix)) + M_prefix
- suffix_lse = log(Σ_suffix exp(s - M_suffix)) + M_suffix
- M = max(M_prefix, M_suffix)

合并:
- prefix_scale = exp(prefix_lse - M) / (exp(prefix_lse - M) + exp(suffix_lse - M))
  = Σ_prefix exp(s - M) / Σ_all exp(s - M)
  = P_prefix (prefix的softmax权重)

- output = prefix_output × P_prefix + suffix_output × P_suffix
  = Σ_prefix exp(s-M)×V / Σ_all exp(s-M) × Σ_prefix exp(s-M)/Σ_all + ...
  = Σ_all exp(s-M)×V / Σ_all exp(s-M) ✓

**结论**: Merge 等价于对所有token一次性softmax, 数学精确无误。