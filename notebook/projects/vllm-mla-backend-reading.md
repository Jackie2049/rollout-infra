# vLLM MLA Attention Backend 源码阅读

> 基于 vLLM v0.23.0 源码分析 (`rollout-infra/vllm/vllm/v1/attention/backends/mla/`)
> 分析 DeepSeek-V2/V3/V4 的 Multi-head Latent Attention 在 vLLM 中的多种 backend 实现
> 日期: 2026-06-16

---

## 1. 为什么 MLA 需要专门的 Attention Backend

★★★★★ MLA (Multi-head Latent Attention) 和标准 Attention (MHA/GQA) 的核心区别：

```
标准 Attention (MHA/GQA):
  K, V 直接存储在 KV Cache 中: [seq_len, num_kv_heads, head_dim]
  Attention: Q @ K^T / sqrt(d)

MLA Attention:
  KV Cache 只存储压缩的 latent: [seq_len, kv_lora_rank] + [seq_len, qk_rope_head_dim]
  需要从 latent 恢复/投影 K, V → 不同的 attention 计算路径
  关键: 不需要显式恢复完整 K/V（矩阵吸收）
```

★★★★★★★ MLA decode 是 compute-bound: h_q=128 时 compute-memory ratio ≈ 256!
  和标准 MHA decode (memory-bound) 不同! MLA 的压缩让 compute 变成瓶颈
  → DeepSeek-V3 decode 需要专门 kernel 优化

  ★★★★★ DeepSeek-V3 MLA 维让 decode 重新变成 compute-bound!

因此 MLA 需要专门 kernel 实现来:
1. 处理压缩的 KV cache 格式
2. 优化两条计算路径（prefill 和 decode 不同）
3. 支持解耦 RoPE（separate content 和 position 分支）

---

## 2. MLA Backend 体系结构

### 2.1 Decode Backend 优先级 (cuda.py L78-130)

★★★★★★★ SM89 (RTX 4090) 的 MLA backend 选择是**由 hardware capability 严格决定**:

SM100 (Blackwell) 优先级:
  1. FlashInfer MLA ( — head ≤ 16 时最优, FP8 KV cache 优先选 FlashInfer)
  2. Tokenspeed MLA — FP8 KV cache + DeepSeek R1 维度专用, CuTe DSL 优化
  3. Cutlass MLA — CUTLASS SM100 kernel, block_size=128
  4. FlashAttn MLA — SM9 only
  5. FlashMLA — SM90 Hopper 主力
  6. Triton MLA — **通用 fallback, 所有 CUDA SM**

SM90 (Hopper) 优先级:
  1. FlashAttn MLA — FA3 scheduler, 支持 batch invariance
  2. FlashMLA — SM90 主力, FP8 dense decode
  3. FlashInfer MLA — SM10 only, SM90 不支持
  4. Triton MLA — 通用 fallback
  5. FlashMLA Sparse — SM90+SM100, FP8 sparse decode

★★★★★★★ SM89 (RTX 4090) 唯一可用 backend: **Triton MLA**
  FlashMLA: SM90-only → RTX 4090 不可用!
  FlashAttn MLA: SM9-only → RTX 4090 的 SM89 属于 SM9 家族但不支持 FA3 MLA
  FlashInfer MLA: SM100-only → RTX 4090 不可用
  Cutlass MLA: SM100-only → RTX 4090 不可用!
  Tokenspeed MLA: SM100-only → RTX 4090 不可用!

### 2.2 Sparse Backend 优先级

SM90+SM100 优先级:
  - FP8 KV cache: FlashInfer MLA Sparse > FlashMLA Sparse
  - BF16 KV cache: num_heads ≤ 16 → FlashInfer MLA Sparse; num_heads > 16 → FlashMLA Sparse

  - XPU MLA Sparse (Intel GPU), ROCm Aiter MLA Sparse (AMD)

### 2.3 Prefill Backend 优先级 (selector.py L64-75)

SM100 (Blackwell):
  1. FlashAttn — 簡单可靠, 任意维度
  2. TRT-LLM Ragged — 变长序列, 支持 FP8 prefill quant
  3. FlashInfer — SM100 only, 需要 DeepSeek R1 维度
  4. Tokenspeed MLA — FP8 prefill + decode, Blackwell 专用

SM90 (Hopper):
  1. FlashAttn — 仅支持 FA3 (需要 flash_attn_supports_mla())
  其他 CUDA (SM89 等):
  1. FlashAttn — 需要 FA 支持检查

### 2.4 各 Backend 详细对比

| Backend | SM Support | FP8 KV Cache | Batch Invariance | CUDA Graph | Block Size | KV Layout |
|---------|-----------|----------------|-----------------|------------|------------|----------|
| **Triton MLA** | **All CUDA** | **Mode 1 (dequant in kernel)** | **Yes** | UNIFORM_BATCH | MultipleOf(16) | N/A |
| **FlashMLA** | SM90 | FP8 ds_mla format | Yes | UNIFORM_BATCH | 64 | N/A |
| **FlashAttn MLA** | SM9 | **NOT SUPPORTED** | **Yes** | UNIFORM_BATCH | MultipleOf(16) | N/A |
| **FlashInfer MLA** | SM100 | FP8 standard | No LSE return | 32, 64 | **HND** |
| **Cutlass MLA** | SM100 | FP8 (未实现) | No | UNIFORM_SINGLE_TOKEN_DECODE | 128 | N/A |
| **Tokenspeed MLA** | SM100 | **FP8 ONLY** | No LSE return | 32, 64 | **HND** |
| **Aiter Triton MLA** | ROCm | FP8 BMM | No | UNIFORM_BATCH | MultipleOf(16) | N/A |

---

## 3. Triton MLA — SM89 的唯一选择
★★★★★★★ RTX 4090 (SM89) 运行 MLA 模型时， **Triton MLA 是唯一可用的 backend**

原因分析:
- FlashMLA 需要 SM90 (Hopper TMA + WGMMA 特性) → RTX 4090 不可用
- FlashAttn MLA 需要 SM9 (即 SM90), 不是 SM89 → 不可用
- FlashInfer MLA / Cutlass MLA / Tokenspeed MLA 需要 SM100 → 不可用
- Triton MLA `supports_compute_capability` 返回 `True` → **所有 CUDA 设备可用**

### 3.1 Triton MLA FP8 KV Cache: Mode 1 (dequant in kernel)
★★★★★ Triton MLA 的 FP8 KV cache 夯现方式是 "Mode 1":
- KV cache 存储 FP8, 在 Triton kernel 内部 dequantize 为 BF16
- Query 保持 BF16, 不需要额外 FP8 量化
- `supports_quant_query_input = False` (triton_mla.py L131-132)

- 这样做的好处: 逻辑简单, SM89 可用; 不需要 Hopper 专用 dequant 猇令
- 缺点: BF16 计算 → 相比 FlashMLA 的 FP8 ds_mla 献式性能较差

★★★ FlashMLA 的 FP8 ds_mla format (V3.2):
  每个 token 656 bytes:
    - 前 512 bytes: 512 × float8_e4m3 (NoPE 部分)
    - 中间 16 bytes: 4 × float32 scale factor (128 一组)
    - 最后 128 bytes: 64 × bfloat16 (RoPE 部分, 不量化)

★★★★★★★ DeepSeek-V4 fp8_ds_mla format:
  每个 token 584 bytes:
    - 前 448 bytes: 448 × float8_e4m3 (NoPE 部分)
    - 中间 128 bytes: 64 × bfloat16 (RoPE 部分, 不量化)
    - 最后 8 bytes: 7 × ue8m0 scale factor + 1B pad
  压缩比: V3.2 576B → V4 512B → 11.3% 提升!

★★★ FlashMLA Sparse 的 FP8 decode 使用 Crossover 共享 dequantize (flashmla-reading.md):
  2 CTA 组成 cluster, 各负责 64 query heads
  CTA 0 dequantize 前半 → CTA 1 dequantize 后半
  每个 CTA 只需 dequantize 一半 → 25 cycles < 34 cycles → dequantization 不再瓶颈

### 3.2 Triton MLA + VLLM_BATCH_INVARIANT
★★★★★ Triton MLA 设置 `VLLM_BATCH_INVARIANT` 时的行为:
- `num_kv_splits = 1` → 禁用 split-KV,优化, 罚证 reduction (triton_mla.py L158-159)
- 这样保证 attention 计算的确定性 (deterministic)
- ★★★★★★ 但 prefix caching 会被禁用:
  ```python
  # mla_attention.py L428-438
  if (
      cache_config.enable_prefix_caching
      and envs.VLLM_BATCH_INVARIANT
      and (
          self.attn_backend.get_name() == "TRITON_MLA"
          or self.attn_backend.get_name() == "FLASHINFER"
      )
  ):
      logger.warning_once(
          "Disabling prefix caching for TRITON_MLA / FLASHINFER "
          "with batch invariance, as it is not yet supported.",
      )
      cache_config.enable_prefix_caching = False
  ```
  ★★★★★★ 原因: `num_kv_splits=1` 时 Triton decode attention 的 split-reduce
逻辑不支持变前缀共享的 KV block

  → **RTX 4090 上 Triton MLA + batch invariance = 无 prefix caching**

★★★ FlashAttn MLA 的 batch invariance (flashattn_mla.py L157-158, L214-215):
  ```python
  # MetadataBuilder
  if envs.VLLM_BATCH_INVARIANT:
      self.max_num_splits = 1  # 禁用 split

  # _build_decode
  if envs.VLLM_BATCH_INVARIANT:
      max_num_splits = 1  # 禁用 split
  ```
  ★★★★★ FlashAttn MLA 支持 batch invariance + prefix caching (SM90 only)
  Triton MLA 支持 batch invariance 但不支持 prefix caching (SM89+)

  → **RTX 4090 只能同时使用 batch invariance + prefix caching 需要升级 SM90**

---

## 4. MLA KV Cache 压缩比分析

★★★★★★★ MLA 压缩比 vs GQA-8 vs MHA:

  ```
  标准 GQA-8 (per token, per layer):
    K: 64 heads × 128 head_dim = 8192 elements
    V: 64 heads × 128 head_dim = 8192 elements
    总计: 16,384 × 2 bytes = 32,768 bytes

  MLA (DeepSeek-V3) (per token, per layer):
    kv_c: 512 elements (latent)
    k_pe: 64 elements  (RoPE key)
    总计: 576 × 2 bytes = 1,152 bytes

  ★★★★★★★ 压缩比: 32,768 / 1,152 ≈ 28.4x!
  (vs GQA-8)


  MLA (DeepSeek-V4) (per token, per layer, fp8_ds_mla):
    NoPE: 448 × float8_e4m3 = 448 bytes (FP8 量化)
    RoPE: 64 × bfloat16 = 128 bytes (不量化)
    scale: 7 × ue8m0 + 1B pad = 8 bytes
    总计: 584 bytes

  ★★★★★★★ 压缩比: 32,768 / 584 ≈ 56.1x (vs GQA-8, fp8)
  压缩比: 1,152 / 584 ≈ 2.0x (vs V3 MLA bf16)

  V3 MLA bf16 → V4 MLA fp8: 从 576 bytes 降到 584 bytes → 2x 压缩
```

### 4.1 解耦 RoPE 的必要性
★★★★★★★ MLA 的解耦 RoPE 是绝对必要的技术设计:
  ```
  如果对 kv_c 直接施加 RoPE:
    RoPE(kv_c) → W_UK 和 RoPE 矩阵耦合
    → 无法做矩阵吸收 (W_UK 不能被吸收到 W_Q)
    → Decode 时必须从 latent 恢复完整 K → 丧失压缩优势

  解决方案: 解耦 RoPE
    content 部分 (q_nope, k_nope): 不带位置信息 → 可以做矩阵吸收
    position 部分 (q_pe, k_pe): 独立处理 → 所有 head 共享一个 k_pe
  → k_pe shape: [Skv, R] (只有 1 个 KV head, MQA)
  → 每个 token 只额外存储 64 elements (qk_rope_head_dim)
  ```
  (来源: mla_attention.py L156-167, mla.py L156-167)

---

## 5. MLA INT8 KV Cache 支持状态

### 5.1 各 Backend INT8/FP8 KV Cache 支持状态

★★★★★ Triton MLA: 支持 FP8 KV cache (Mode 1)
  - `supports_quant_query_input = False` → query 不量化 FP8
  - Triton kernel 内部 dequantize FP8 KV → BF16
  - 传入 `k_scale=layer._k_scale` (triton_mla.py L210-212)
  - ★★★★★ SM89 可以用 FP8 KV cache (通过 Triton MLA)


  ★★★ FlashMLA: 支持 FP8 KV cache (FP8 ds_mla format)
  - Dense decode: `flash_mla_with_kvcache_fp8` (flashmla.py L306-318)
  - SM90-only → RTX 4090 不支持
 这种 FP8 KV cache
  - 需要 `get_mla_metadata_dense_fp8` 生成 tile scheduler 元数据

  - FP8 decode 有独立的 tile scheduler metadata → 支持完整 CUDA Graph

  ★★★ FlashAttn MLA: **不支持** FP8 KV cache
  - `raise NotImplementedError("FlashAttnMLA V1 with FP8 KV cache not yet supported")` (flashattn_mla.py L304-307)
  - ★★★★★ RTX 4090 不能通过 FlashAttn MLA 使用 FP8 KV cache

  ★★ FlashInfer MLA: 支持 FP8 KV cache (标准 FP8)
  - 使用 `trtllm_batch_decode_with_kv_cache_mla` (flashinfer_mla.py L190-202)
  - SM100-only → RTX 4090 不支持
  - 不返回 LSE → 不支持 speculative decoding 的 logit 禂率

  - 需要 `qk_nope_head_dim ∈ {64, 128, 192}` 维度限制

  ★★ Cutlass MLA: FP8 KV cache **未实现**
  - `_num_kv_splits` 限制， 可能导致 hang (可用 `FORCE_NUM_KV_SPLITS=1`)
  - SM100-only

  ★★ Tokenspeed MLA: **FP8 KV cache ONLY**
  - 不支持 BF16 KV cache (tokenspeed_mla.py L176-181)
  - SM100-only
  - 不返回 LSE
  - 需要 DeepSeek R1 维度 (128, 64, 128)

  ★★★★★★★ MLA INT8 KV Cache 总结: SM89 上只有 Triton MLA 支持 FP8 KV cache (但性能差)


  ★★★★★ INT8 KV cache 是 RTX 4090 上唯一生产可行的 MLA KV 路径 (FlashInfer backend, MHA/GQA 用)

---

## 6. DeepSeek-V4 MLA 扩展
★★★★★★★ DeepSeek-V4 引入了全新的 MLA 扩展机制:

### 6.1 核心概念: Compress Ratio + Sliding Window Attention (SWA)
★★★★★★★ DeepSeek-V4 的 `compress_ratio` 是革命性变化:
  - `compress_ratio = 4` → C4A: 每 4 个 token 崋缩为 1 个 compressed token
  - `compress_ratio = 128` → C128A: 每 128 个 token 压缩为 1 个
  - `compress_ratio ≤ 1` → SWA-only: 只做滑动窗口注意力

  KV cache block_size 被压缩: `storage_block_size = block_size // compress_ratio`

  ★★★★★ C128A 的 Indexer 使用 Triton kernel `_build_c128a_topk_metadata_kernel`:
  - Decode: position → block_table lookup → global slot ids + topk_lens
  - Prefill: position → local indices [0, ..., n-1, -1, ...]
  - Triton kernel 预计算 topk indices, 减少 CPU→GPU sync
  (来源: flashmla_sparse.py L1083-1150)

### 6.2 DeepSeek-V4 Sparse MLA Attention
★★★★★★★ V4 使用专门的 `DeepseekV4MLAAttention` 类:
  - 不使用 v1 framework 的 `forward_mqa` instance方法
  - 而是 `forward_mqa` **classmethod** (Liskov override intentional)
  - 由 layer (DeepseekV4MLAAttention) 而动而非 framework 调度
  - 同时处理 SWA + compressed KV (双缓存机制)
  - Prefill: dequantize + gather → combine topk + SWA indices → flash_mla_sparse_fwd
  - Decode: SWA indices + C4A/C128A indices → flash_mla_with_kvcache (双缓存)
  (来源: attention.py L616-623, nvidia/flashmla.py L130-302)

### 6.3 V4 的 3-Way GEMM Overlap
★★★★★★★ V4 attention 实现了 `attn_gemm_parallel_execute`:
  - Default stream: fused_wqa_wkv (最重的 GEMM)
  - Aux stream 0: compressor kv_score (compress_ratio > 1)
  - Aux stream 1: indexer weights_proj + compressor kv_score (compress_ratio > 1 时)
  - Aux stream 2: reserved
  - 3-way overlap → 1x latency 隐藏 → 3x throughput
  (来源: attention.py L305-363)

### 6.4 V4 的 O 投影: inverse RoPE + FP8 quant
★★★★★★★ V4 的 O 投影使用全新的方式:
  - `fused_inv_rope_fp8_quant`: inverse RoPE + FP8 quantize O (attention.py L276-285)
  - `fp8_einsum`: FP8 einsum for O 投影 (DeepGemm)
  - SM90 recipe: (1, 128, 128) → FP32 block scales
  - SM100 recipe: (1, 1, 128) → INT32 packed scales
  - wo_a: FP8 + inverse RoPE → compressed O → wo_b → final output

### 6.5 V4 的 KV Cache 格式
★★★★★★★ DeepSeek-V4 fp8_ds_mla format (per token 584 bytes):
  - 前 448 bytes: 448 × float8_e4m3 (NoPE 部分, FP8 量化)
  - 中间 128 bytes: 64 × bfloat16 (RoPE 部分, 不量化)
  - 最后 8 bytes: 7 × ue8m0 scale factor + 1B pad
  - V3.2: 512 NoPE + 4 × fp32 scale + 64 bf16 RoPE = 656 bytes
  - V4: 448 NoPE + 64 bf16 RoPE + 7 ue8m0 scale + 1 pad = 584 bytes
  - ★★★★★★★ V3 → V4 压缩: 656 → 584 bytes (2x 提升!) + ue8m0 替代 fp32 scale

  V3.2 scale: 4 × float32 (128-element 分组, 量化)
  V4 scale: 7 × ue8m0 (64-element 分组, 量化) → 更细粒度 + 更紧凑

### 6.6 V4 Indexer
★★★★★★★ V4 的 Indexer (Lightning Indexer):
  - 使用 `SparseAttnIndexer` 选择 top-K compressed tokens
  - `use_fp4_cache`: 支持 FP4 (MXFP4) indexer cache
  - indexer cache head_dim = 128 fp8 + 4 fp32 scale = 132 bytes
  - `skip_topk = True`: 同一 Indexer 在不同 layer 间共享 top-K 结果
  (来源: attention.py L667-807)

---

## 7. Triton MLA 内核架构深度分析

★★★★★★★ Triton MLA 使用 `decode_attention_fwd` 内核:
  - 来源: Lightllm GQA flash decoding (triton_decode_attention.py L1-8)
  - SGLang → vLLM 的移植

### 7.1 两阶段架构
  ```
  Stage 1 (_fwd_kernel_stage1):
    Q @ K^T → per-head per-split 注意力得分
    输出: attn_logits [B, H, S, kv_lora_rank + 1]
    +1 位置存储 LSE (log-sum-exp)
    支持 FP8 KV dequantize (k_scale, v_scale)
    is_mla=True: 使用 kv_lora_rank 代替 head_dim

  Stage 2 (_fwd_kernel_stage2):
    跨 split 合并 partial attention outputs
    softmax(attn_logits) → weighted sum → final output
    ```

### 7.2 Split-KV 优化
★★★★★ Triton MLA 的 `num_kv_splits` 选择逻辑:
  ```python
  # triton_mla.py L157-175
  if envs.VLLM_BATCH_INVARIANT:
      num_kv_splits = 1  # 确定性 reduction
  else:
      min_work_per_split = 512  # 每个 split 最少工作量
      ideal_splits = max(1, max_seq_len // min_work_per_split)
      ideal_splits = triton.next_power_of_2(ideal_splits)  # 2 的幂
      occupancy_multiplier = 2  # 每个 SM 2x blocks
      max_splits = sm_count * occupancy_multiplier
      num_kv_splits = min(ideal_splits, max_splits)
  ```
  ★★★★★★★ `VLLM_BATCH_INVARIANT` 时: num_kv_splits=1 → 禁用 split → 确定性但代价是性能下降
  正常模式: 动态 splits → 平衡负载均衡和 性能优先

  ★★★★★ 关键洞察: `num_kv_splits=1` 导致 prefix caching 不兼容
 见第 3.2 诂

### 7.3 Triton MLA 的维度限制 (PR #41119)
★★★★★★★ Triton MLA 的 `BLOCK_DMODEL` 和 `BLOCK_DV` 逻辑:
  - 旧版 (v0.20.1): DuplicateEntry → 只支持 Lk=576 和 288 → Mistral kv_lora_rank=256 崩溃
  - PR #41119 修复: `next_power_of_2(Lv)` 通用 fallback
  - ★★★★★★★ Mistral Small 4 (kv_lora_rank=256) + 其他非 DeepSeek MLA 维度现在可以工作
  - Issue #45031: Triton MLA grouped-decode fails for kv_lora_rank=256

  - PR #41119: generalized dimension handling (merged 2026-05-11)


  Triton MLA 支持的 KV cache dtype:
  - fp8, fp8_e4m3: FP8 KV cache (Mode 1, dequant in kernel)
  - auto, float16, bfloat16: BF16/FP16 KV cache

---

## 8. MLA 数据流总结

### 8.1 Decode 数据流 (MQA 路径)
★★★★★★★ MLA decode 完整数据流:
  ```
  hidden_states [Sq, H] → MLA wrapper

  MLA Wrapper (mla.py):
    1. 压缩: q_c = hidden_states @ W_DQ, kv_c = hidden_states @ W_DKV
    2. 解压: q_nope = q_c @ W_UQ, q_pe = RoPE(q_c @ W_QR)
    3. RoPE: q[..., qk_nope_head_dim:], k_pe = RoPE(hidden_states @ W_KR)
    4. Indexer (sparse): indexer(hidden_states, q_c, positions) → top-K indices
    5. Attention: mla_attn(q, kv_c_normed, k_pe) → attn_out

  MLAAttention.forward_impl (mla_attention.py):
    6. 矩阵吸收 (decode): ql_nope = q_nope @ W_UK_T → (N,B,P)×(N,P,L)
    7. FP8 quant (if fp8): mqa_q = concat(ql_nope, q_pe) → FP8 quantize
    8. forward_mqa: impl.forward_mqa(mqa_q, kv_cache, metadata, layer) → attn_out
 lse
    9. V 投影: attn_out @ W_UV → output

  Output projection (mla.py):
    10. o_proj(attn_out) → final output
  ```

### 8.2 Prefill 数据流 (MHA 路径)
★★★★★★★ MLA prefill 完整数据流:
  ```
  Prefill 使用 "compute-friendly" MHA 路径:
    1. kv_b_proj: kv_c → kv_nope (k_nope + v) (解压完整 K/V)
    2. concat k_nope + k_pe → K
    3. FlashAttn/FlashInfer: standard MHA attention
    4. 输出 = softmax(Q @ K^T) @ V

  Chunked prefill:
    镱分 context 夘分块处理 → workspace
    每个块: gather KV → compute attention → merge_attn_states 合并结果
  ```

---

## 9. 源码文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `mla_attention.py` | 2327 | MLA 核心: CommonBackend/CommonImpl/MetadataBuilder |
| `mla/triton_mla.py` | 216 | Triton MLA decode (通用 fallback) |
| `mla/flashmla.py` | 335 | FlashMLA decode (SM90 主力) |
| `mla/flashattn_mla.py` | 365 | FlashAttn MLA decode (SM9 only) |
| `mla/flashinfer_mla.py` | 210 | FlashInfer MLA decode (SM100 only) |
| `mla/cutlass_mla.py` | 286 | CUTLASS MLA decode (SM100 only) |
| `mla/tokenspeed_mla.py` | 278 | Tokenspeed MLA decode (SM100 FP8 only) |
| `mla/flashmla_sparse.py` | 1150 | FlashMLA Sparse (SM90+SM100) |
| `mla/aiter_triton_mla.py` | 67 | Aiter Triton MLA (ROCm AMD) |
| `mla/prefill/` | 目录 | Prefill 专用实现 |
| `mla/prefill/registry.py` | 139 | Prefill backend 注册枚举 |
| `mla/prefill/selector.py` | 185 | Prefill backend 自动选择 |
| `mla/prefill/flash_attn.py` | ~100 | FlashAttn prefill (FA3) |
| `mla/prefill/flashinfer.py` | ~100 | FlashInfer prefill (SM100) |
| `mla/prefill/trtllm_ragged.py` | ~100 | TRT-LLM ragged prefill |
| `mla/prefill/tokenspeed_mla.py` | ~100 | Tokenspeed prefill (SM100) |
| `ops/flashmla.py` | 154 | FlashMLA ops 封装 (dense + FP8 + sparse) |
| `ops/triton_decode_attention.py` | ~600 | Triton decode attention 内核 |
| `mla.py` | 182 | MultiHeadLatentAttentionWrapper (MLA 外层) |
| `deepseek_v4/attention.py` | 807 | DeepSeek-V4 MLA attention layer |
| `deepseek_v4/nvidia/flashmla.py` | 425 | V4 FlashMLA Sparse impl |

---

## 10. RTX 4090 (SM89) MLA 宐在总结

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  RTX 4090 MLA 关键结论:
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

  1. ★★★★★★★ **Triton MLA 是 SM89 唯一可用的 MLA decode backend**
     - FlashMLA (SM90-only) / FlashAttn MLA (SM9-only) / FlashInfer MLA (SM100-only) 都不可用
     - Triton MLA 性能显著低于 SM90/SM100 专用 kernel

  2. ★★★★★★★ **FP8 KV cache: Triton MLA (Mode 1) 是 SM89 唯一路径**
     - Triton MLA FP8 KV cache = de朴素 Mode 1 (dequant in kernel)
     - 性能: BF16 计算 + 显式 dequant → 比 FlashMLA FP8 ds_mla 性能差
     - INT8 KV cache (FlashInfer backend) 是生产可行路径 (但不适用于 MLA)

  3. ★★★★★ **Batch invariance + Triton MLA = 无 prefix caching**
     - Triton MLA 支持 batch invariance 但禁用 prefix caching
     - FlashAttn MLA 支持 batch invariance + prefix caching (但需要 SM90)
     - **RTX 4090 不能同时使用 batch invariance + prefix caching + MLA**

  4. ★★★★★ **DeepSeek-V4 MLA 在 RTX 4090 上完全不可用**
     - V4 需要 FlashMLA Sparse (SM90+SM100) + fp8_ds_mla format
     - V4 的 compress_ratio/C128A/Indexer 需要 SM90+ 特性
     - RTX 4090 只能运行 DeepSeek-V3/V3.2 MLA (通过 Triton MLA)

  5. ★★★★★ **Triton MLA 维度限制已修复 (PR #41119)**
     - v0.20.1 只支持 Lk=576/288 → Mistral kv_lora_rank=256 崩溃
     - 主线版本: `next_power_of_2(Lv)` 通用 fallback
     - Mistral Small 4 等非 DeepSeek MLA 维度现在可用

  6. ★★★★★★★ **升级建议: RTX 4090 MLA 的最优策略**
     - 如果只跑 DeepSeek-V3 MLA → Triton MLA + BF16 KV cache (最稳定)
     - 如果需要 FP8 KV → Triton MLA FP8 (Mode 1, 性能差)
     - 如果需要 prefix caching + batch invariance → 忉须升级到 SM90
     - DeepSeek-V4 MLA → 必须使用 SM90+ GPU
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 参考

- vLLM 源码: `vllm/v1/attention/backends/mla/`
- DeepSeek-V2 Paper: https://arxiv.org/abs/2405.04434
- FlashMLA GitHub: https://github.com/deepseek-ai/FlashMLA
- FlashInfer Project: https://github.com/flashinfer-ai/flashinfer
- Tokenspeed MLA PR: https://github.com/vllm-project/vllm/pull/41778
- Triton MLA 维度修复: https://github.com/vllm-project/vllm/pull/41119
- Triton MLA kv_lora_rank bug: https://github.com/vllm-project/vllm/issues/45031
- Batch invariance issue: https://github.com/vllm-project/vllm/issues/40173
- DeepSeek-V4 PR: https://github.com/vllm-project/vllm/pull/43182
- `notebook/fundamentals/mla.md` — MLA 理论笔记
- `notebook/projects/flashmla-reading.md` — FlashMLA kernel 深度笔记
