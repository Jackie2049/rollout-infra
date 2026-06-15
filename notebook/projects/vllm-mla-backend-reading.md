# vLLM MLA Attention Backend 深度源码阅读

> 基于 `rollout-infra/vllm/vllm/v1/attention/backends/mla/` + `model_executor/layers/attention/mla_attention.py` + `platforms/cuda.py` 全量源码分析
>
> 覆盖 DeepSeek-V2/V3/V4 MLA 全 backend 变体, SM89 (RTX 4090) vs SM90+ (H100) vs SM100 (Blackwell) 兼容性矩阵, TritonMLA 作为 SM89 唯一 MLA 选项的关键影响, batch invariance + prefix caching 行为, fp8_ds_mla 格式, INT8 KV cache 支持, 压缩比对比, 以及 DeepSeek-V4 MLA 扩展
>
> 日期: 2026-06-16 (更新版)

---

## 1. MLA Backend 总览 — 10 种 Decode + 5 种 Prefill + 5 种 Sparse

### 1.1 Decode Backend 完整列表

| Backend | 类名 | 目标 SM | KV Cache dtype | Block Size | HND Layout | Key Feature |
|---------|------|--------|----------------|------------|------------|-------------|
| **FlashMLA** | `FlashMLABackend` | SM9, SM10 (major=9 或 10) | BF16/FP16/FP8/FP8_e4m3 | 64 | No | SM90 主力, fp8_ds_mla FP8 KV, Seesaw Scheduling |
| **FlashAttnMLA** | `FlashAttnMLABackend` | SM9 (major=9) | BF16/FP16 only | MultipleOf(16) | No | FA3 MLA varlen, 支持 batch invariance, **不支持 FP8 KV** |
| **FlashInferMLA** | `FlashInferMLABackend` | SM10 (major=10) | BF16/FP16/FP8/FP8_e4m3 | 32, 64 | **Yes (HND)** | Blackwell 专用, trtllm API, qk_nope ∈ {64,128,192} |
| **TritonMLA** | `TritonMLABackend` | **All CUDA** (True) | BF16/FP16/FP8/FP8_e4m3 | MultipleOf(16) | No | ★★★★★ **SM89 唯一 MLA 选项**, FP8 Mode1 (BF16 Q + kernel 内 dequant) |
| **CutlassMLA** | `CutlassMLABackend` | SM10 (major=10) | BF16/FP16/FP8/FP8_e4m3 | 128 | No | Blackwell CUTLASS, q_pad=128 heads, workspace 128MB |
| **TokenspeedMLA** | `TokenspeedMLABackend` | SM10 (major=10) | **FP8/FP8_e4m3 only** | 32, 64 | **Yes (HND)** | Blackwell CuTe DSL, FP8 Q+KV, DSR1 专用维度 |
| **AiterTritonMLA** | `AiterTritonMLABackend` | ROCm (AMD) | - | - | - | AMD GPU 专用, 继承 AiterMLABackend |
| **AiterMLA** | `AiterMLABackend` | ROCm (AMD) | - | - | - | AMD GPU 专用 |

★★★★★ **关键发现**: TritonMLA `supports_compute_capability()` 返回 `True` (源码 `triton_mla.py:78`), 是唯一能在 SM89 上运行的 MLA decode backend。FlashMLA 限制 SM90+, FlashAttnMLA 限制 SM9 (不含 SM89 9.x 子版本的实际 FA3 MLA kernel 支持), 其余均 SM100 only。

### 1.2 Sparse MLA Backend

| Backend | 目标 SM | KV Cache | 特点 |
|---------|--------|----------|------|
| **FlashMLA Sparse** | SM9, SM10 | fp8_ds_mla / BF16 | DeepSeek-V3.2 Sparse Attention, FP8 Crossover dequant |
| **FlashInfer MLA Sparse** | SM10 | FP8 (标准) | Blackwell 专用 |
| **XPU MLA Sparse** | XPU (Intel) | - | Intel GPU |
| **ROCm Aiter MLA Sparse** | ROCm | - | AMD |
| **V4 FlashMLA Sparse** | SM9, SM10 | fp8_ds_mla (584B/token) | ★★★ DeepSeek-V4 专用, C128A compress_ratio=128 |

### 1.3 Prefill Backend

| Backend | 类名 | 目标 SM | 特点 |
|---------|------|--------|------|
| **FlashAttn Prefill** | `FlashAttnPrefillBackend` | All CUDA | 主力, padding 处理不同 head dim |
| **FlashInfer Prefill** | `FlashInferPrefillBackend` | SM100 only | DeepSeek R1 维度 (128,64,128) |
| **TRTLLM Ragged Prefill** | `TrtllmRaggedPrefillBackend` | SM100 only | Triton-based ragged attention |
| **Tokenspeed MLA Prefill** | `TokenspeedMLAPrefillBackend` | SM100 only | CuTe DSL, FP8 Q+KV prefill |

---

## 2. ★★★★★ SM89 (RTX 4090) MLA Backend 选择路径

### 2.1 Backend Priority 逻辑 (源码 `platforms/cuda.py:79-130`)

```
SM89 (capability.major == 9, 非 10):
  MLA decode backend 优先级:
    1. FLASH_ATTN_MLA   — SM9 only, 但 FA3 MLA kernel 实际上仅在 SM90 hopper 上可用
    2. FLASHMLA          — SM90 only (is_flashmla_dense_supported 检查 family==90)
    3. FLASHINFER_MLA    — SM100 only
    4. TRITON_MLA        — All CUDA (supports_compute_capability=True) ← 最终选中!
    5. FLASHMLA_SPARSE   — SM90+ only

  → 当 FLASH_ATTN_MLA 和 FLASHMLA 都因 SM 限制被拒绝后,
    TRITON_MLA 自动成为 SM89 上唯一可用的 MLA decode backend
```

★★★★★ **RTX 4090 影响**: SM89 上的 DeepSeek-V2/V3 MLA 推理 **必须走 TritonMLA decode 路径**。FlashMLA dense kernel 的 `is_flashmla_dense_supported()` 明确检查 `is_device_capability_family(90)` (源码 `ops/flashmla.py:58`), 即 RTX 4090 的 SM89 不满足条件。FlashAttnMLA 的 FA3 MLA kernel 同样依赖 Hopper (SM90) TMA + WGMMA 特性。

### 2.2 TritonMLA 在 SM89 上的关键特性

| 特性 | TritonMLA 实现 | 源码位置 |
|------|----------------|----------|
| **FP8 KV Cache** | Mode 1: BF16 query + kernel 内 dequant FP8→BF16 | `triton_mla.py:128-132` |
| **supports_quant_query_input** | FP8 KV 时设为 False (不量化 Q 到 FP8) | `triton_mla.py:131-132` |
| **batch invariance** | `supports_batch_invariance()` → True | `triton_mla.py:65-66` |
| **VLLM_BATCH_INVARIANT 行为** | num_kv_splits=1, 确保确定性 reduction | `triton_mla.py:158-159` |
| **prefix caching + BI** | **被禁用!** (enable_prefix_caching=False) | `mla_attention.py:428-438` |
| **KV split 计算** | max_seq_len//512 → power_of_2, 限制 SM×2 | `triton_mla.py:160-175` |
| **Kernel** | 2-stage decode_attention_fwd (stage1: Q@K+softmax, stage2: merge splits) | `ops/triton_decode_attention.py` |

### 2.3 ★★★★★ TritonMLA + VLLM_BATCH_INVARIANT + Prefix Caching 交互

```python
# 源码 mla_attention.py:426-438 — 关键交互!
if (
    cache_config is not None
    and cache_config.enable_prefix_caching
    and envs.VLLM_BATCH_INVARIANT
    and (
        self.attn_backend.get_name() == "TRITON_MLA"    # ← SM89 落入这个!
        or self.attn_backend.get_name() == "FLASHINFER"  # 非 MLA 的 FlashInfer
    )
):
    logger.warning_once(
        "Disabling prefix caching for TRITON_MLA / FLASHINFER "
        "with batch invariance, as it is not yet supported.",
    )
    cache_config.enable_prefix_caching = False
```

★★★★★ **RTX 4090 MLA + batch invariant = prefix caching 被禁用**! 这意味着:
- SM89 上 TritonMLA + VLLM_BATCH_INVARIANT 模式下, prefix caching 自动关闭
- 原因: TritonMLA 的 num_kv_splits=1 路径与 prefix cache block 的共享逻辑冲突 (split=1 确保确定性, 但 prefix block 共享需要 ref_cnt 保护, 当前 Triton decode kernel 不支持这个交互)
- FlashMLA 的 `VLLM_BATCH_INVARIANT` 路径则手动构造 tile_scheduler_metadata (源码 `flashmla.py:275-303`), 并不与 prefix caching 冲突 — 但 FlashMLA 在 SM89 上不可用

---

## 3. 各 MLA Backend 详细源码分析

### 3.1 FlashMLA (SM90 主力)

```
源码: vllm/v1/attention/backends/mla/flashmla.py
核心: FlashMLABackend → FlashMLAImpl → forward_mqa()

关键点:
  1. supports_compute_capability: capability.major ∈ {9, 10} (line 74-75)
  2. 实际 dense kernel 限制: is_flashmla_dense_supported() → family==90 (ops/flashmla.py:58)
     → SM89 (RTX 4090) 被拒绝, 即使 major==9 也通过, 但 family check 更严格
  3. FP8 KV Cache 路径:
     - is_quantized_kv_cache → flash_mla_with_kvcache_fp8 (line 305-318)
     - 需要 get_mla_metadata_dense_fp8 预计算 tile scheduler metadata (line 172-194)
     - CUDA graph 时把 FP8 metadata 拷入持久 buffer (line 180-191)
  4. VLLM_BATCH_INVARIANT + FlashMLA:
     - 非 FP8 KV 时: 手动构造 tile_scheduler_metadata (line 275-303)
     - 单 partition, begin_idx=0, end_idx=B-1, end_block_idx=topk//B_TOPK
     - num_splits = zeros of length B+1 (line 300-301)
     - FP8 KV 时: 使用 get_mla_metadata_dense_fp8 预计算, 然后拷入 CG buffer
  5. reorder_batch_threshold = 128 (line 112) — 小 prefill 可以走 decode 路径
  6. query_len_support = UNIFORM (line 111) — 支持 spec decode
```

### 3.2 FlashAttnMLA (SM9, Hopper FA3)

```
源码: vllm/v1/attention/backends/mla/flashattn_mla.py
核心: FlashAttnMLABackend → FlashAttnMLAImpl → forward_mqa()

关键点:
  1. supports_compute_capability: capability.major == 9 (line 72-73)
     ★★★ 但 FA3 MLA kernel 实际需要 SM90 hopper 特性 (TMA, WGMMA)
     → 在 SM89 上 likely 被 flash_attn_supports_mla() 检查拒绝
  2. supports_batch_invariance: True (line 59-61) ← SM89 上唯一 BI-capable 的 MLA 选项
     但如果 FA3 kernel 在 SM89 不可用, TritonMLA 接替 (也支持 BI)
  3. FP8 KV Cache: **不支持!** (line 304-307, 326-327)
     → raise NotImplementedError("FlashAttnMLA V1 with FP8 KV cache not yet supported")
  4. VLLM_BATCH_INVARIANT + FA MLA:
     - max_num_splits = 1 (line 157-158, 214-215)
     - FA3 AOT schedule: get_scheduler_metadata 预计算 (line 170-186)
     - CUDA graph 时拷入持久 scheduler_metadata buffer (line 227-240)
  5. query_len_support = VARLEN (line 108) ← 比 FlashMLA 更灵活!
  6. reorder_batch_threshold = 512 (line 109) — 更宽容的 decode→prefill 转换阈值
  7. 支持 DCP (Decode Context Parallel) with varlen (line 118-126)
```

### 3.3 ★★★★★ TritonMLA (All CUDA, SM89 唯一 MLA 选项)

```
源码: vllm/v1/attention/backends/mla/triton_mla.py
核心: TritonMLABackend → TritonMLAImpl → forward_mqa()
Kernel: vllm/v1/attention/ops/triton_decode_attention.py (2-stage)

关键点:
  1. supports_compute_capability: True (line 77-78)
     → ★★★★★ 在 SM89 上是唯一可选的 MLA decode backend
  2. supported_kv_cache_dtypes: FP16/BF16/FP8/FP8_e4m3 (line 38-44)
     → 支持 FP8 KV Cache!
  3. FP8 KV 实现: "Mode 1" — BF16 query, kernel 内 dequant FP8 KV→BF16
     → supports_quant_query_input = False when FP8 KV (line 131-132)
     → 告知上游 pipeline 不要将 Q 量化为 FP8, kernel 自行处理
  4. ★★★★★ VLLM_BATCH_INVARIANT:
     - supports_batch_invariance(): True (line 65-66)
     - num_kv_splits = 1 (line 158-159) ← 确定性 reduction, 无 split 合并误差
     - 但 prefix caching 被禁用! (mla_attention.py:428-438)
  5. 正常模式 KV split 计算:
     - min_work_per_split = 512 (line 163)
     - ideal_splits = next_power_of_2(max_seq_len // 512) (line 165-168)
     - max_splits = SM_count × 2 (line 173-174)
     - num_kv_splits = min(ideal_splits, max_splits) (line 175)
  6. attn_logits 中间缓冲区:
     - shape: [B, q_num_heads, num_kv_splits, kv_lora_rank+1] (line 178-189)
     - +1 存 LogSumExp (LSE), 供 stage2 kernel 合并 splits
  7. Kernel 调用: decode_attention_fwd(..., is_mla=True) (line 198-213)
     - k_scale/v_scale 从 layer._k_scale 获取 (line 210-211)
     - FP8 KV: kernel 内 k_scale 做 dequant scale
  8. head_size 兼容性: 320, 576 (MLACommonBackend line 1197-1198)
     - Mistral Small 4 kv_lora_rank=256 + rope=64 → 320 ✓ (已通过 #41119 修复)
```

### 3.4 FlashInferMLA (SM100, Blackwell)

```
源码: vllm/v1/attention/backends/mla/flashinfer_mla.py
核心: FlashInferMLABackend → FlashInferMLAImpl → forward_mqa()

关键点:
  1. supports_compute_capability: capability.major == 10 (line 65-66) ← Blackwell only
  2. required_kv_cache_layout: "HND" (line 94-96) ← 需要 Head-Number-Dim layout
  3. supports_combination 检查: qk_nope_head_dim ∈ {64, 128, 192} (line 86-92)
     → 不满足维度则被拒绝
  4. FP8 KV Cache: 支持, bmm1_scale 和 bmm2_scale 包含 q/k scale (line 182-188)
  5. 使用 trtllm_batch_decode_with_kv_cache_mla (FlashInfer API) (line 190-202)
  6. 不返回 LSE (line 207-209) ← pending FlashInfer API support
```

### 3.5 CutlassMLA (SM100, Blackwell CUTLASS)

```
源码: vllm/v1/attention/backends/mla/cutlass_mla.py
核心: CutlassMLABackend → CutlassMLAImpl → forward_mqa()

关键点:
  1. supports_compute_capability: capability.major == 10 (line 65-66) ← Blackwell only
  2. block_size 固定 128 (line 49-50, 75)
  3. q_pad_num_heads = MAX_HEADS = 128 (line 101, 133-134)
  4. workspace: g_sm100_workspace 128MB (line 99)
  5. num_kv_splits: 默认 -1 (auto), 可用 FORCE_NUM_KV_SPLITS 覆盖 (line 152-160)
     ★★★ 当前限制 16 splits 避免挂起! 如遇挂起可设 FORCE_NUM_KV_SPLITS=1
  6. FP8 KV: NotImplementedError "CutlassMLAImpl does not support scaling for q and kv_latent yet" (line 258-261)
     → 实际上不支持 FP8 KV cache!
  7. 输出: MAX_HEADS padding + slice back (line 241-244)
```

### 3.6 ★★★ TokenspeedMLA (SM100, Blackwell FP8-Only)

```
源码: vllm/v1/attention/backends/mla/tokenspeed_mla.py
核心: TokenspeedMLABackend → TokenspeedMLAImpl → forward_mqa()

关键点:
  1. supports_compute_capability: capability.major == 10 (line 82-83) ← Blackwell only
  2. ★★★ supported_kv_cache_dtypes: FP8/FP8_e4m3 ONLY (line 61-64)
     → **必须 FP8 KV cache** (不支持 BF16 KV!)
  3. supports_combination: DSR1 专用维度 (qk_nope=128, rope=64, v=128) (line 118)
  4. required_kv_cache_layout: "HND" (line 127-128)
  5. FP8 Q input: supports_quant_query_input=True, 要求 q dtype=float8_e4m3fn (line 224-228)
     → 上游 pipeline 通过 _DecodeConcatQuantFP8 量化 Q 到 FP8
  6. softmax_scale = scale × q_scale × k_scale (line 247-249)
     output_scale = k_scale (line 250)
  7. tokenspeed_mla_decode CuTe DSL kernel, workspace = SM×heads×max_q_len×(kv_lora_rank+1)×4
  8. 性能: 在 Blackwell 上比 trtllm 快 2-4× (PR #41778 benchmark)
```

---

## 4. MLA Prefill Backend 选择

### 4.1 Prefill 优先级 (源码 `prefill/selector.py:54-76`)

```
SM100 (Blackwell):
  1. FLASH_ATTN    ← 主力
  2. TRTLLM_RAGGED ← Triton ragged
  3. FLASHINFER    ← 需要 DSR1 维度
  4. TOKENSPEED_MLA ← CuTe DSL, FP8 Q+KV

SM90 (Hopper) 及更低:
  1. FLASH_ATTN    ← 只有这一个选项!
```

★★★★★ **SM89 prefill 只有 FlashAttn**: SM89 的 MLA prefill 只能走 FlashAttention (FA3/FA4) prefill 路径。FA3 MLA prefill 使用 padding 处理不同 head dim (将 MLA 的 MQA 转换为 MHA 格式)。

---

## 5. MLA KV Cache 格式与压缩比

### 5.1 标准 Attention vs MLA KV Cache 大小对比

```
标准 MHA KV Cache (per token, per layer):
  K: [num_kv_heads, head_dim]     = 128 × 128 = 16,384 elements
  V: [num_kv_heads, head_dim]     = 128 × 128 = 16,384 elements
  总计: 32,768 elements × 2 bytes  = 65,536 bytes

标准 GQA-8 KV Cache (per token, per layer):
  K: [8, head_dim]    = 8 × 128 = 1,024 elements
  V: [8, head_dim]    = 8 × 128 = 1,024 elements
  总计: 2,048 elements × 2 bytes = 4,096 bytes

DeepSeek-V2/V3 MLA KV Cache (per token, per layer):
  kv_c: [kv_lora_rank]            = 512 elements (latent)
  k_pe: [qk_rope_head_dim]        = 64 elements  (RoPE key)
  总计: 576 elements × 2 bytes    = 1,152 bytes

★★★★★ 压缩比对比:
  MLA vs MHA:  65,536 / 1,152 = 56.9x ← 极端压缩!
  MLA vs GQA-8: 4,096 / 1,152 = 3.6x ← 即使 vs GQA-8 也显著压缩
  MLA vs GQA-1 (MQA): 1,152 / 1,152 = 1.0x ← MLA latent ≈ MQA 1 head 的 KV cache 大小

核心差异: MLA 512 latent + 64 RoPE = 576 维度存储, vs GQA-8 需 8×(128+128)=2048 维度存储
```

### 5.2 fp8_ds_mla 格式详解

★★★★★ **fp8_ds_mla 是 DeepSeek 专有的 MLA FP8 KV cache 格式**, 不是标准 FP8 KV cache。

#### V3.2 fp8_ds_mla (656 bytes/token)

```
源码: flashmla_sparse.py:68-90 (module docstring)

每个 token 的 FP8 KV cache = 656 bytes:
  前 512 bytes: 512 × float8_e4m3 → 量化 NoPE 部分 (kv_lora_rank=512)
  中间 16 bytes: 4 × float32 → tile-wise scale factors (每 128 FP8 值一组)
  最后 128 bytes: 64 × bfloat16 → RoPE 部分 (不量化, 保持精度)

比例: NoPE=512B / total=656B → NoPE 占 78%, RoPE 占 19.5%, scale 占 2.4%
```

#### ★★★ V4 fp8_ds_mla (584 bytes/token)

```
源码: flashmla_sparse.py:80-90 + nvidia/flashmla.py:107-109

每个 token 的 FP8 KV cache = 584 bytes:
  前 448 bytes: 448 × float8_e4m3 → 量化 NoPE 部分 (kv_lora_rank=448)
  中间 128 bytes: 64 × bfloat16 → RoPE 部分 (不量化)
  最后 8 bytes: 7 × ue8m0 + 1B pad → per-64-element scale factors (MX scale)

★★★ 关键差异:
  V3.2: NoPE=512 FP8 + 4×FP32 scale (tile-wise, 128 per group)
  V4:   NoPE=448 FP8 + 7×ue8m0 scale (MX-style, 64 per group)
  → V4 scale 更紧凑 (8B vs 16B), 但量化组更细 (64 vs 128)
  → V4 RoPE 部分位置从尾部移到中间, 排列 [NoPE, RoPE, scale]
```

### 5.3 MLA KV Cache in MLAAttentionSpec

```python
# 源码 kv_cache_interface.py:337-367

class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = None
    compress_ratio: int = 1           # DeepseekV4 only, default=1
    model_version: str | None = None

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio  # C128A: block_size//128

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                return self.storage_block_size * 584   # V4: 584B/token
            return self.storage_block_size * 656       # V3.2: 656B/token
```

---

## 6. ★★★★★ MLA INT8 KV Cache 支持状态

### 6.1 各 Backend 的 INT8/FP8 KV 支持矩阵

| Backend | BF16 KV | FP8 KV (标准) | fp8_ds_mla KV | INT8 KV |
|---------|---------|---------------|---------------|---------|
| **FlashMLA** | ✓ | ✓ (fp8_e4m3) | ✓ (sparse only) | ✗ 无 INT8 dtype |
| **FlashAttnMLA** | ✓ | **✗ NotImplementedError** | ✗ | ✗ |
| **TritonMLA** | ✓ | ✓ (Mode1: BF16 Q+kernel dequant) | ✗ | ✗ |
| **FlashInferMLA** | ✓ | ✓ (FP8 Q+KV) | ✗ | ✗ |
| **CutlassMLA** | ✓ | **✗ NotImplementedError** (scaling not supported) | ✗ | ✗ |
| **TokenspeedMLA** | ✗ (仅 FP8) | ✓ (FP8 Q+KV mandatory) | ✗ | ✗ |

★★★★★ **SM89 MLA FP8 KV 状态**: TritonMLA 支持 FP8 KV, 但使用 "Mode 1" (BF16 query + kernel 内 dequant), 性能不如 FlashMLA 的 "Mode 2" (FP8 Q + FP8 KV + Crossover dequant)。TritonMLA 的 `supports_quant_query_input=False` 意味着 Q 不被量化到 FP8, 所有 FP8→BF16 转换在 Triton kernel 内完成。

★★★ **INT8 KV Cache**: vLLM 的 MLA backend 目前 **没有任何 INT8 KV Cache 支持**。vLLM 的量化系统只提供 FP8 (fp8_e4m3, fp8_e5m2) 和 fp8_ds_mla, 没有 INT8 per-token 或 INT8 block-wise quantization 路径。对于 SM89 (RTX 4090), 标准 FP8 KV cache 的 Triton FP8 path 和 FlashMLA Sparse 的 fp8_ds_mla path 都因 SM 限制不可用, 这使得 **SM89 上 MLA 的唯一 KV cache dtype 是 BF16/FP16**。

---

## 7. MLA Backend 选择 — SM89 vs SM90 vs SM100

### 7.1 完整 SM 兼容性矩阵

| SM | Decode Backend | Prefill Backend | Sparse Backend | FP8 KV | BI Support | Prefix Cache+BI |
|----|----------------|----------------|----------------|--------|------------|-----------------|
| **SM89 (RTX 4090)** | ★★★★★ TritonMLA only | FlashAttn | ✗ (SM90+ only) | Triton Mode1 | ✓ | ✗ (disabled!) |
| **SM90 (H100/H800)** | FlashMLA > FlashAttnMLA > TritonMLA | FlashAttn | FlashMLA Sparse | FlashMLA FP8 | FlashMLA ✓, FlashAttn ✓ | FlashMLA ✓ |
| **SM100 (B200/GB200)** | FlashInferMLA > Tokenspeed > Cutlass > FlashAttnMLA > FlashMLA > Triton | FlashAttn > TRTLLM > FlashInfer > Tokenspeed | FlashMLA Sparse, FlashInfer Sparse | FlashInfer FP8, Tokenspeed FP8, FlashMLA fp8_ds_mla | ✓ (most) | ✓ |

### 7.2 ★★★★★ SM89 (RTX 4090) MLA 推理完整路径

```
SM89 MLA 推理路径 (唯一):

Decode: TritonMLA
  → decode_attention_fwd(..., is_mla=True, num_kv_splits=1 when BI)
  → FP8 KV: Mode1 (BF16 Q, kernel dequant)
  → 无 FlashMLA Seesaw/Crossover 优化
  → 性能: 理论上比 FlashMLA SM90 decode 低 2-3×

Prefill: FlashAttn (FA3/FA4 MLA prefill)
  → padding 将 MLA MQA 转换为 MHA 格式
  → Chunked prefill with workspace
  → 无 TRTLLM/Tokenspeed prefill 优化

Sparse: ✗ 不可用
  → FlashMLA Sparse 需要 SM90+ (is_flashmla_sparse_supported)
  → DeepSeek-V3.2 Sparse Attention 在 RTX 4090 上不可用

FP8 KV: TritonMLA Mode1 only
  → 不如 FlashMLA fp8_ds_mla 的 Crossover 共享 dequant
  → Q 不量化 (supports_quant_query_input=False)
  → 单 token dequant: ~50 cycles, 无 Crossover (SM89 没有 Distributed Shared Memory)

Prefix Cache + BI: ✗ 被禁用!
  → TritonMLA + VLLM_BATCH_INVARIANT → enable_prefix_caching=False
  → 这严重影响长上下文场景的 KV cache 重用效率
```

---

## 8. ★★★★★ DeepSeek-V4 MLA 扩展

### 8.1 V4 MLA 核心创新

```
源码: models/deepseek_v4/attention.py + nvidia/flashmla.py

DeepSeek-V4 MLA 相比 V2/V3 的关键扩展:

1. Compress Ratio (压缩比):
   - compress_ratio ∈ {1, 4, 128}
   - C4A: compress_ratio=4 → 每 4 个 token 压缩为 1 个 compressed token
   - C128A: compress_ratio=128 → 每 128 个 token 压缩为 1 个!
   - storage_block_size = block_size / compress_ratio

2. Sparse Windowed Attention (SWA):
   - 每个请求有独立的 SWA cache (DeepseekV4SWACache)
   - window_size 限制本地关注范围
   - SWA-only 层 (compress_ratio≤1) 不分配主 KV cache

3. Lightning Indexer (索引器):
   - 稀疏注意力模式: indexer 选择每个 query 只关注的 compressed token
   - topk_indices_buffer 存储索引结果
   - 支持 FP8 和 MXFP4 indexer cache
   - 3-way GEMM overlap: default stream (wq_b+kv_insert) + aux stream 0 (indexer) + aux stream 1 (compressor)

4. FP8 O-projection (逆向 RoPE + FP8 einsum):
   - fused_inv_rope_fp8_quant: O → 逆 RoPE → FP8 quant
   - fp8_einsum: FP8 O × FP8 wo_a → BF16 z
   - wo_b: BF16 z → 最终输出
   - SM90 recipe: (1, 128, 128), SM100 recipe: (1, 1, 128)

5. DeepseekCompressor (压缩器):
   - 将 token 压缩为 compressed token
   - rotate=True: 旋转位置编码处理
   - fused_wkv_wgate: 合并 KV 投影 + gating score

6. V4 FlashMLA Sparse Backend:
   - DeepseekV4FlashMLASparseBackend (nvidia/flashmla.py:79-111)
   - get_supported_head_sizes: [512] ← 448 NoPE + 64 RoPE = 512 (vs V3.2 的 576)
   - get_kv_cache_shape: fp8_ds_mla → (num_blocks, block_size, 584)
   - block_size=256 (vs V3.2 的 64)
```

### 8.2 V4 Sparse MLA Decode 路径

```
源码: models/deepseek_v4/nvidia/flashmla.py:130-286

DeepseekV4FlashMLASparseImpl.forward_mqa():

  1. Split prefill + decode
  2. Decode:
     - 获取 SWA indices + compressed topk indices
     - C4A: compute_global_topk_indices_and_lens (per-layer)
     - C128A: pre-computed during metadata build (attn_metadata.c128a_*)
     - flash_mla_with_kvcache:
       q + swa_cache + extra_k_cache (compressed) + extra_indices_in_kvcache
       → SWA window + compressed KV 合并注意力!
  3. Prefill:
     - dequantize_and_gather_k_cache: 从 FP8 compressed KV gather + dequant
     - dequantize_and_gather_k_cache: SWA KV gather
     - combine_topk_swa_indices: 合并 topk + SWA indices
     - flash_mla_sparse_fwd: 稀疏 prefill 注意力
```

### 8.3 ★★★ C128A compress_ratio=128 的含义

```
compress_ratio=128:
  每 128 个 token → 1 个 compressed token
  128K context → 128K/128 = 1K compressed tokens
  → KV cache 大小从 128K×576 bytes = ~90MB → 1K×584 bytes = ~0.6MB

★★★ 但 C128A 有 alignment 约束:
  - _C128A_TOPK_ALIGNMENT = 128 (flashmla_sparse.py:360)
  - c128a_max_compressed = cdiv(max_model_len, compress_ratio)
  - cdiv(c128a_max_compressed, 128) × 128 ← 必须对齐到 128

★★★★★ C128A 是 DeepSeek-V4 的革命性创新:
  - 128 倍压缩 → 极端 KV cache 压缩
  - Sparse Attention + Compression → 2 层筛选
  - Indexer 先选 topk compressed tokens, 再在 SWA window 内精确定位
  → 让 128K+ context 在有限 VRAM 上变得可行
```

---

## 9. MLA 两条计算路径详解

### 9.1 Compute-Friendly (Prefill → forward_mha)

```
源码: mla_attention.py:66-86 (注释)

Prefill 路径 (MHA 模式):
  q_c      = h_t @ W_DQ    → [Sq, Lq]       (query 压缩)
  q_nope   = q_c @ W_UQ    → [Sq, N, P]      (解压 content query)
  q_pe     = RoPE(q_c @ W_QR) → [Sq, N, R]  (解压 RoPE query)
  kv_c     = h_t @ W_DKV   → [Skv, Lkv]      (KV 压缩)
  k_nope   = kv_c @ W_UK   → [Skv, N, P]     (解压 content key)
  v        = kv_c @ W_UV   → [Skv, N, V]     (解压 value)

  → 标准 MHA: Q=[q_nope;q_pe], K=[k_nope;k_pe], V=v
  → Attention output: [Sq, N, V]

特点: 完全解压 → 计算量大但并行度高,适合 prefill 的 compute-bound 特性
实现: Chunked prefill with workspace, merge_attn_states 合并 chunk 输出
```

### 9.2 ★★★★★ Data-Movement Friendly (Decode → forward_mqa)

```
源码: mla_attention.py:94-118 (注释)

Decode 路径 (MQA 模式):
  q_nope   = h_t @ W_DQ @ W_UQ  → [Sq, N, P]
  ql_nope  = einsum(q_nope, W_UK) → [Sq, N, Lkv]  ← ★★★ 矩阵吸收!
  q_pe     = RoPE(q_c @ W_QR)    → [Sq, N, R]
  kv_c     = [Skv, Lkv]            ← 直接用 latent, 不解压!

  → MQA: Q=[ql_nope;q_pe], K=[kv_c;k_pe], V=kv_c
  → Attention output: [Sq, N, Lkv] ← 在 latent 空间做 attention!
  → 最终: einsum(output, W_UV) @ W_O → 完整输出

★★★★★ 关键洞察:
  1. ql_nope = q_nope @ W_UK: 将 Q 的 content 部分直接投影到 latent 空间
     → 矩阵吸收: W_UK 被 "吸收" 到 Q 侧, K 不需要解压
  2. KV cache 只存储 kv_c (latent) + k_pe (RoPE), 不存储完整 K/V
  3. decode 时: MQA 模式, 所有 128 query head 共享 1 个 KV head
  4. 这就是 MLA decode 的 "data-movement friendly" 特性:
     不需要从 latent 恢复完整 K/V, 减少 memory bandwidth
```

### 9.3 Decode 路径的 v_up_proj 实现

```
源码: mla_attention.py:972-994

_v_up_proj(x, out):
  # x shape: [B, N, Lkv] (latent 空间 attention output)
  # → transpose to [N, B, Lkv]
  # → bmm with W_UV [N, Lkv, V] → [N, B, V]
  # → transpose back to [B, N, V]
  # → out shape: [B, N×V]

ROCm paths:
  - aiter FP4 bmm: batched_gemm_a16wfp4 (MXFP4 W_V)
  - aiter FP8 bmm: triton_fp8_bmm (FP8 W_V)
  - Default: torch.bmm (BF16 W_UV)
```

---

## 10. 关键源码文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `mla_attention.py` | ~2327 | ★★★★★ MLA 核心层 + MLACommonBackend/Impl/MetadataBuilder |
| `mla/triton_mla.py` | ~216 | ★★★★★ TritonMLA (SM89 唯一 MLA decode) |
| `mla/flashmla.py` | ~335 | FlashMLA decode (SM90 主力) |
| `mla/flashattn_mla.py` | ~365 | FlashAttn MLA decode (SM9, FA3) |
| `mla/flashinfer_mla.py` | ~210 | FlashInfer MLA decode (SM100) |
| `mla/cutlass_mla.py` | ~286 | CUTLASS MLA decode (SM100) |
| `mla/tokenspeed_mla.py` | ~278 | ★★★ Tokenspeed MLA decode (SM100, FP8 only) |
| `mla/flashmla_sparse.py` | ~1150 | ★★★★★ FlashMLA Sparse (V3.2 + V4 FP8 KV) |
| `ops/flashmla.py` | ~154 | FlashMLA op wrapper, SM support check |
| `ops/triton_decode_attention.py` | ~400+ | ★★★ Triton 2-stage decode kernel |
| `platforms/cuda.py` | ~80-130 | ★★★★★ MLA backend priority 逻辑 |
| `mla/prefill/registry.py` | ~139 | Prefill backend 注册 |
| `mla/prefill/selector.py` | ~185 | Prefill backend 自动选择 |
| `models/deepseek_v4/attention.py` | ~807 | ★★★★★ DeepSeek-V4 MLA 层 |
| `models/deepseek_v4/nvidia/flashmla.py` | ~425 | V4 FlashMLA Sparse impl |
| `kv_cache_interface.py` | ~337-367 | MLAAttentionSpec + fp8_ds_mla layout |
| `mla.py` (model_executor) | ~182 | MultiHeadLatentAttentionWrapper |
| `mla/compressor_utils.py` | - | V4 compressed slot mapping |

---

## 11. 环境变量与配置

| 变量 | 作用 | 相关 Backend |
|------|------|--------------|
| `VLLM_BATCH_INVARIANT` | 启用 batch invariant 模式 | TritonMLA (splits=1), FlashMLA, FlashAttnMLA |
| `VLLM_MLA_DISABLE=1` | 禁用 MLA, 回退到标准 Attention | 所有 MLA backend (workaround) |
| `FORCE_NUM_KV_SPLITS` | 覆盖 KV splits 数量 | CutlassMLA (解决挂起问题) |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | FlashInfer workspace大小 | FlashInferMLA |
| `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` | 多流 GEMM token阈值 | DeepSeek-V4 3-way overlap |

---

## 12. ★★★★★ RTX 4090 MLA 推理实战总结

```
RTX 4090 (SM89) MLA 推理限制:

Decode:
  ✗ FlashMLA (SM90 only)
  ✗ FlashAttnMLA (FA3 kernel 需要 SM90 TMA/WGMMA)
  ✗ FlashInferMLA / CutlassMLA / TokenspeedMLA (SM100 only)
  ✓ TritonMLA ← 唯一选项, 但性能不如 FlashMLA

FP8 KV Cache:
  ✗ FlashMLA fp8_ds_mla (SM90 only)
  ✓ TritonMLA Mode1 (BF16 Q + kernel dequant) ← 性能更低

Sparse Attention:
  ✗ FlashMLA Sparse (SM90+ only)
  → DeepSeek-V3.2 Sparse Attention 不可用

Prefix Caching + Batch Invariant:
  ✗ TritonMLA + BI → prefix caching 被禁用 (mla_attention.py:428-438)
  → 长上下文场景 KV 重用效率大幅下降

INT8 KV Cache:
  ✗ 无任何 MLA backend 支持 INT8 KV

★★★★★ 核心结论:
  SM89 上 MLA 推理的实际可行配置:
  1. KV cache dtype = BF16 (FP8 Mode1 性能不如 BF16, 因无 Crossover)
  2. VLLM_BATCH_INVARIANT = 不建议开启 (会禁用 prefix caching)
  3. 仅支持 DeepSeek-V2/V3 dense MLA (V4 sparse 需要 SM90+)
  4. Triton decode kernel 性能瓶颈: 2-stage reduction, 无 WGMMA 优化
  5. 最大希望: FlashAttnMLA 的 FA3 MLA kernel 如果能在 SM89 上启用
     → 目前 FA3 MLA kernel 需要 SM90, 但 FA3 standard attention 在 SM89 可用
```

---

## 参考

- `notebook/projects/flashmla-reading.md` — FlashMLA kernel 深度笔记
- `notebook/projects/vllm-v1-triton-attention-reading.md` — Triton attention kernel 源码分析
- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434) — MLA 原始论文
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA) — Seesaw Scheduling + Crossover
- vLLM PR #41778 — TokenspeedMLA backend (Blackwell)
- vLLM PR #41119 — Triton MLA generalize dimension (kv_lora_rank=256 fix)
- vLLM Issue #45031 — TritonMLA Mistral Small 4 (kv_lora_rank=256)
- vLLM Issue #40173 — Auto-select batch-invariant backend
- DeepSeek-V4 paper: [arxiv:2603.12201](https://arxiv.org/abs/2603.12201) — C128A + Lightning Indexer
- vLLM 源码: `vllm/v1/attention/backends/mla/`, `platforms/cuda.py`, `models/deepseek_v4/`
