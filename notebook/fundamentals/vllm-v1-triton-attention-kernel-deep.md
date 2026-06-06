# vLLM V1 Triton Attention Kernel 深度分析

> 2026-06-07 | vLLM V1 两套 Triton attention kernel 架构与算法对比

## 两套 Triton Attention 实现

vLLM V1 有 **两套** Triton attention kernel:

| | Split-KV Decode | Unified Attention |
|---|---|---|
| **文件** | `triton_decode_attention.py` | `triton_unified_attention.py` |
| **来源** | SGLang → LightLLM | IBM Zürich Research Lab |
| **调用者** | TritonMLABackend, TurboQuant | TritonAttentionBackend (主后端) |
| **Prefill+Decode** | 仅Decode | 合一 (2D Prefill / 3D Decode) |
| **算法** | 两阶段: Split-KV → Reduce | 两阶段: 3D Tiled Softmax → reduce_segments |
| **GQA** | 专用grouped kernel (BLOCK_H=16) | BLOCK_M = n_q_per_kv 自动处理 |
| **MLA** | IS_MLA flag (c_kv共享/transpose) | 不专门处理MLA |
| **FP8 KV** | inline dequant (k/v_scale scalar) | 4种量化模式 (per-tensor/per-token-head) |
| **Page size** | PAGE_SIZE constexpr | BLOCK_SIZE from KV cache shape |
| **Block table** | Req_to_tokens (行=request) | block_table (标准vLLM格式) |
| **Q layout** | (batch, heads, dim) | (tokens, heads, dim) — 扁平化 |
| **Softcap** | ✅ (logit_cap) | ✅ |
| **ALiBi** | ❌ | ✅ |
| **Sliding Window** | ❌ | ✅ (tile pruning) |
| **Sinks** | ❌ | ✅ |
| **MM Prefix** | ❌ | ✅ |
| **Chunked Attn** | ❌ | ✅ |
| **Tensor Desc** | ❌ | ✅ (Intel XPU) |

**趋势**: Unified kernel 正在成为主流, Split-KV 仅为 MLA/TurboQuant 特殊场景保留。

---

## Prefill Attention Kernel (SGLang origin)

### 文件: `triton_prefill_attention.py`

**Grid**: `(batch, head, cdiv(max_input_len, BLOCK))`

**BLOCK size**: FP32=32, SM≥80=128, 其他=64

**核心**: 经典FlashAttention但用 `exp2` 替代 `exp`:
```python
sm_scale *= RCP_LN2  # 1/ln(2) → 转为log2域, Triton硬件优化
qk = tl.dot(q, k)
qk = tl.where(mask, qk * sm_scale, -1e8)
p = tl.math.exp2(qk)  # 用exp2代替exp
```

**特点**: 仅Prefill (非paged), Causal+双向Sliding Window, 无FP8/mm_prefix/sinks/softcap

---

## Merge Attention States (Cascade合并)

### 文件: `merge_attn_states.py` (CUDA优先) + `triton_merge_attn_states.py` (Triton fallback)

**算法**: Section 2.2 of arxiv 2501.01005 (Cascade Attention)

**用途**: 合并 Prefix(KV cache) + Suffix(new tokens) 的partial attention:
```python
max_lse = max(prefix_lse, suffix_lse)
p_scale = exp(prefix_lse - max_lse) / (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
s_scale = exp(suffix_lse - max_lse) / (exp(prefix_lse - max_lse) + exp(suffix_lse - max_lse))
output = prefix_output * p_scale + suffix_output * s_scale
```

**关键**: 两个分开的online softmax如何合并 → 全局max rescale → 加权求和 → 数学等价一次softmax

---

## 关键发现总结

## Split-KV Decode Attention (Stage1 + Stage2)

### Stage1: Split-KV 并行计算

**Grid**: `(batch, head_num, NUM_KV_SPLITS)`

每个program处理一个 KV split (即KV序列的一段):
```
kv_len_per_split = ceil(seq_len / NUM_KV_SPLITS)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = min(split_kv_start + kv_len_per_split, seq_len)
```

**核心逻辑 (per split)**:
```python
e_max = -inf
e_sum = 0
acc = zeros(HEAD_DIM)

for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    # 1. Paged KV 间接寻址
    kv_page_number = Req_to_tokens[batch, offs_n // PAGE_SIZE]
    kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

    # 2. 加载 K/V tile
    k = K_Buffer[kv_loc, kv_head, offs_d]  # (BLOCK_N, HEAD_DIM)
    v = V_Buffer[kv_loc, kv_head, offs_dv] # (BLOCK_N, HEAD_DIM_V)

    # 3. FP8 dequant (inline)
    if k.dtype.is_fp8(): k = (k.float * k_scale).to(q.dtype)

    # 4. QK dot + scale
    qk = sum(q * k, dim=-1) * sm_scale

    # 5. Softcap (tanh clamp)
    if logit_cap > 0: qk = logit_cap * tanh(qk / logit_cap)

    # 6. Online softmax (per-split)
    n_e_max = max(e_max, max(qk))
    re_scale = exp(e_max - n_e_max)
    p = exp(qk - n_e_max)
    acc = acc * re_scale + sum(p * v)     # 累积加权V
    e_sum = e_sum * re_scale + sum(p)
    e_max = n_e_max
```

**输出**: 每个split存储 `(acc/e_sum, e_max + log(e_sum))` → Att_Out
- `acc/e_sum` = 该split的加权平均V (用局部max归一化)
- `e_max + log(e_sum)` = 该split的log-sum-exp (用于全局reduce)

### Stage2: Softmax Reduce

**Grid**: `(batch, head_num)`

```python
e_sum = 0
e_max = -inf
acc = zeros(HEAD_DIM)

for split_kv_id in range(NUM_KV_SPLITS):
    tv = Att_Out[batch, head, split_kv_id, :HEAD_DIM]     # 该split的加权V
    tlogic = Att_Out[batch, head, split_kv_id, HEAD_DIM]   # log-sum-exp

    n_e_max = max(e_max, tlogic)
    old_scale = exp(e_max - n_e_max)

    # 全局rescale: 旧结果 × exp(旧max-新max) + 新split × exp(新log-新max)
    acc = acc * old_scale + exp(tlogic - n_e_max) * tv
    e_sum = e_sum * old_scale + exp(tlogic - n_e_max)
    e_max = n_e_max

result = acc / e_sum  # 最终softmax归一化输出
```

**关键数学**: 这是 **online softmax 的分布式版本**!
- 全局 softmax = Σ(exp(S_i - global_max) × V_i) / Σ(exp(S_i - global_max))
- 每个 split 先做局部 online softmax, 存储局部 max + log(sum)
- Stage2 把所有 split 的局部结果做第二次 online softmax → 得到全局结果
- 等价于: global_max = max(all local_max), 然后全局rescale

### Grouped Variant (GQA/MQA/MLA)

**Grid**: `(batch, cdiv(head_num, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)`

与normal variant的区别:
1. **BLOCK_H=16**: 同时处理16个Q头 (共享同一个KV头)
2. `tl.dot(Q, K)` 替代 `tl.sum(q*k)`: 利用Tensor Core做矩阵乘
3. **MLA模式** (`IS_MLA=True`): K和V共享 c_kv buffer, `v = tl.trans(k)` 避免重复加载
4. **BLOCK_DPE**: MLA的decoupled RoPE, 分开加载position encoding部分 (`qpe`)

```
# MLA decode: Q分为两部分
Q = [q_no_pe, q_pe]  # (BLOCK_H, D_qk) 和 (BLOCK_H, D_pe)
K = [k_no_pe, k_pe]  # 从同一个c_kv buffer加载

S = tl.dot(q_no_pe, k_no_pe) + tl.dot(q_pe, k_pe)
V = tl.trans(k)  # MLA: V = c_kv的transpose (不额外加载)
```

---

## Unified Attention Kernel (2D Prefill / 3D Decode)

### 架构: 单一kernel处理 Prefill + Decode

**核心决策**: 根据 batch_size 和 query_len 选择 2D 或 3D 模式:

```python
use_3d = not (
    seq_threshold_3D is None   # 无分配3D buffer
    or max_seqlen_q > 1        # 有prefill请求
    or num_seqs > seq_threshold_3D  # batch太大
    or is_batch_invariant       # batch不变优化
)
```

**seq_threshold_3D** = `MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads` = `128 // num_kv_heads`

| num_kv_heads | threshold | 含义 |
|---|---|---|
| 1 (MQA) | 128 | 3D仅当batch<=128 |
| 4 (GQA) | 32 | 3D仅当batch<=32 |
| 8 (GQA) | 16 | 3D仅当batch<=16 |
| 32 (MHA) | 4 | 3D仅当batch<=4 |

**原理**:
- 2D Grid = `(q_blocks, kv_heads)` → 大batch有足够并行度
- 3D Grid = `(q_blocks, kv_heads, NUM_SEGMENTS)` → 小batch需要额外并行度 (16 segments)
- NUM_PAR_SOFTMAX_SEGMENTS = 16 (固定)

### 2D Mode (Prefill / 大batch Decode)

**Grid**: `(total_q_blocks, num_kv_heads)`

每个program处理一个 Q-block × KV-head:
```
# Q block分配: 扁平化 (seq, q_block_in_seq)
# 通过binary search找到seq_idx和q_block_local_idx
seq_idx = find_seq_idx(query_start_len, q_block_global_idx)
```

**核心循环**:
```python
M = -inf  # running max
L = 1.0   # running sum (init=1, because L stores exp_sum after alpha scaling)
acc = zeros(BLOCK_M, HEAD_SIZE)

for j in range(loop_lo, loop_hi):
    # 1. 加载 KV tile from paged cache
    physical_block = block_table[seq_idx, j*TILE_SIZE // BLOCK_SIZE]
    offset_in_block = j*TILE_SIZE % BLOCK_SIZE
    K = key_cache[physical_block, offset_in_block + offs_t, kv_head, offs_d]  # (HEAD_SIZE, TILE_SIZE)
    V = value_cache[physical_block, offset_in_block + offs_t, kv_head, offs_d]  # (TILE_SIZE, HEAD_SIZE)

    # 2. Causal mask + sliding window + mm_prefix
    seq_mask = compute_kv_seq_mask(query_pos, seq_offset, ...)
    # = (causal AND window) OR mm_prefix

    # 3. Score: S = scale * dot(Q, K)
    S = score_scale * tl.dot(Q, K)  # (BLOCK_M, TILE_SIZE)
    if USE_SOFTCAP: S = apply_softcap(S, softcap)
    S = tl.where(mask & seq_mask, S, -inf)

    # 4. Online softmax step
    M, L, P, alpha = softmax_step(S, M, L)
    # M_new = max(M, max(S, axis=1))
    # alpha = exp(M - M_new)
    # P = exp(S - M_new)
    # L_new = L * alpha + sum(P, axis=1)

    # 5. Rescale accumulator + add new contribution
    acc = acc * alpha[:, None]
    acc += tl.dot(P.to(V.dtype), V)
```

**Epilogue**: 直接输出 `acc / L` (除以running sum)

### 3D Mode (小batch Decode)

**Grid**: `(q_blocks, kv_heads, NUM_SEGMENTS=16)`

与2D的区别:
1. 每个program只处理 `tiles_per_segment` 个tile (而非全部)
2. 输出写入 **per-segment buffer** (而非output)
3. 存储 `M` 和 `L` 到 `segm_max` / `segm_expsum`
4. 需要后续 `reduce_segments` kernel做全局reduce

**reduce_segments kernel**:
```
Grid = (num_tokens, num_query_heads)

# 1. 加载所有segment的max和exp_sum
segm_max = load(segm_max_ptr) → 全局max = max(segm_max)
segm_expsum = load(segm_expsum_ptr) × exp(segm_max - global_max)
global_exp_sum = sum(segm_expsum)

# 2. 加载所有segment的output, rescale, sum
segm_output = load(segm_output_ptr) × exp(segm_max - global_max)
result = sum(segm_output) / global_exp_sum
```

### BLOCK_M 设计 (GQA 处理)

```python
BLOCK_M = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
BLOCK_Q = BLOCK_M // num_queries_per_kv
```

| num_queries_per_kv | BLOCK_M | BLOCK_Q | 说明 |
|---|---|---|---|
| 1 (MHA) | 16 | 16 | 16个token per block |
| 2 (GQA) | 16 | 8 | 8 tokens × 2 heads = 16 rows |
| 4 (GQA) | 16 | 4 | 4 tokens × 4 heads |
| 7 (Qwen2-7B) | 16 → 8* | 1 | 1 token × 7 heads (非pow2) |
| 8 (GQA) | 16 | 2 | 2 tokens × 8 heads |
| 16 (MLA-like) | 16 | 1 | 1 token × 16 heads |

**BLOCK_M=16**: 对GQA, 把共享同一KV头的多个Q头打包到一个block → 减少**K/V加载87.5%**!

### Tile Size 选择

```python
# Prefill: 32 (更多计算, 更大tile更好)
# Decode: 16 (或FP8: 32) — 小tile减少单次处理量, 增加并行度
# Gemma3 SWA=1024: decode也用32 (window小, tile可大)
```

### Helper 函数详解

**`resolve_seq_and_query_len`**: Binary search定位 `(seq_idx, q_block_local_idx)`
- `find_seq_idx`: 在 `query_start_len_ptr` (cumulative lengths) 上做二分查找
- 类似于vLLM scheduler的 `query_start_loc` 但增加了BLOCK_Q分块

**`softmax_step`**: Online softmax的核心
- 输入: S(BLOCK_M, TILE_SIZE), M(BLOCK_M), L(BLOCK_M)
- 输出: M_new, L_new, P, alpha
- `alpha = exp(M_old - M_new)` → 用于rescale之前的accumulator
- 数学等价于FlashAttention的 online softmax

**`compute_kv_seq_mask`**: 掩码构建 (causal + window + mm_prefix)
- 默认: causal (key_pos <= query_pos)
- Sliding window: AND (query_pos - key_pos < SLIDING_WINDOW)
- mm_prefix: OR (bidirectional ranges for multimodal tokens)
- 顺序: `(causal AND window) OR mm_prefix` — 与FlexAttention一致

**`compute_tile_loop_bounds`**: Tile循环边界优化
- Sliding window: 只遍历 [qpos - window, qpos] 范围内的tile → **跳过无关tile**
- 3D mode: 只遍历当前segment范围内的tile
- Chunked attention: align到chunk边界

### FP8 KV Cache Quantization (4种模式)

| KV_QUANT_MODE | 方式 | Kernel处理 |
|---|---|---|
| 0 (NONE) | 不量化 | 直接cast到Q dtype |
| 1 (FP8_PER_TENSOR) | 单个scale | `_cast_kv_tile`: K/V × scale |
| 2 (INT8_PER_TOKEN_HEAD) | 每(token,head)一个scale | 融合: `S = dot(Q,K) × (scale × k_scale)` |
| 3 (FP8_PER_TOKEN_HEAD) | 同上FP8 | 同上, 但V: `P_v = P × v_scale` |

**优化**: Per-token-head不单独做 `×scale` on K/V (BLOCK_M×TILE_SIZE multiply),
而是融合到 `tl.dot(Q,K)` 后乘 `k_scale` (1D multiply on BLOCK_M×1) → **减少87.5%计算**

---

## 算法对比: Split-KV vs Unified 3D

两者在decode时的算法本质相同: **分段softmax → 全局reduce**

| | Split-KV | Unified 3D |
|---|---|---|
| **分段数** | NUM_KV_SPLITS (可调) | NUM_PAR_SOFTMAX_SEGMENTS=16 (固定) |
| **分段策略** | 连续切分KV序列 | tile-per-segment (基于TILE_SIZE) |
| **全局reduce** | 专用Stage2 kernel | reduce_segments kernel |
| **输出格式** | (加权V, log-sum-exp) | (加权V/L, M, L) 3个buffer |
| **并行度** | batch × heads × splits | q_blocks × kv_heads × segments |

Split-KV更灵活 (可调splits数), Unified更标准化 (固定16 segments但更多功能支持)。

---

## Triton vs FlashInfer: vLLM Backend选择

| | FlashInfer (默认) | Triton (fallback) |
|---|---|---|
| **SM要求** | SM 75+ (CUDA ≥7.5) | 无限制 (全SM) |
| **Prefill** | BatchPrefillWithPagedKVCache | unified_attention 2D |
| **Decode** | BatchDecodeWithPagedKVCache | unified_attention 3D |
| **GQA** | 内置支持 | BLOCK_M packing |
| **MLA** | FlashInfer MLA (SM90+) | TritonMLABackend (专用kernel) |
| **FP8 KV** | 内置 | 4种量化模式 |
| **Sliding Window** | 内置 | tile pruning |
| **CUDA Graph** | ✅ | ✅ |
| **性能** | 更快 (C++ cuDNN backend) | 较慢 (纯Triton) |

FlashInfer仍是默认首选; Triton backend作为 **全SM兼容** 的fallback, 且功能覆盖更全 (sinks/mm_prefix/ALiBi)。

---

## 关键发现总结

1. **vLLM V1有两套Triton attention**: Split-KV (SGLang origin, MLA专用) 和 Unified (IBM origin, 主后端)
2. **Decode算法本质**: 都是分段online softmax → 全局reduce, 数学等价于FlashAttention
3. **Unified kernel是趋势**: 功能更全 (ALiBi/Sinks/MM/Chunked/4 quant modes), 一套kernel处理prefill+decode
4. **3D模式选择**: batch_size ≤ threshold时用3D (16 segments增加并行度), 大batch用2D
5. **GQA优化**: BLOCK_M打包多个Q头 → K/V加载减少87.5%
6. **FP8 dequant融合**: Per-token-head模式把scale融合到dot product → 减少87.5%计算
7. **Sliding window tile pruning**: 只遍历window范围内的tile → 跳过无关计算
8. **Per-token-head quant比per-tensor更优**: scale融合到softmax_score而不是K/V tile
9. **Prefill kernel用exp2**: Triton硬件优化, sm_scale乘1/ln(2)转入log2域
10. **Merge attn states (cascade)**: LSE rescale方法合并prefix+suffix partial → 全局max → 加权 → 数学等价一次softmax, 是cascade attention的核心