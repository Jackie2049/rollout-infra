# vLLM V1 KV Cache Architecture Source Reading — Multi-Type KV Cache Management

> 2026-06-11 | vLLM V1最新架构: KVCacheSpec 9种类型 + KVQuantMode 5种量化 + NVFP4 KV cache + 3种Coordinator + 可插拔Spec Registry!
> 参考: vllm/v1/kv_cache_interface.py, vllm/v1/core/kv_cache_coordinator.py, vllm/v1/core/single_type_kv_cache_manager.py, vllm/v1/kv_cache_spec_registry.py
> 关联: kv-cache-management-deep-dive.md, scheduler-architecture-deep-dive.md, flashinfer-attention-deep-dive.md

## 0. 核心发现: vLLM V1 KV Cache = 多类型协同管理!

```
旧架构(V0):
  → 单一KV cache类型 → 所有层共享一个block table → 简单但不支持混合模型

V1新架构:
  → KVCacheSpec 9种 → KVQuantMode 5种量化 → 3种Coordinator → 可插拔Registry
  → → 每种类型有独立SingleTypeKVCacheManager → 但共享BlockPool
  → → → HybridKVCacheCoordinator: 迭代固定点算法协调多种类型的前缀匹配
  → → → → DeepSeek-V3/V4: MLA+SlidingWindow+Mamba → 多组KV cache → 生产复杂!

关键洞察:
  → MLAAttentionSpec用FullAttentionManager管理 → MLA不特殊管理, 只是存储layout不同!
  → → SinkFullAttentionSpec用SinkFullAttentionManager → Sink才需要特殊管理!
  → → → NVFP4 KV cache: Blackwell FP4 packed data + fp8 block scales → 下一代量化!
  → → → → SlidingWindow/ChunkedLocal有admission cap → 防止SWA请求过度预留→死锁!
```

## 1. KVCacheSpec 类型层级 — 9种KV cache规格

```
KVCacheSpec (基类)
  │ block_size: int
  │ page_size_bytes: int (property)
  │ storage_block_size: int (property)
  │ max_memory_usage_bytes(vllm_config): int (property)
  │ copy_with_new_block_size(block_size): Self
  │ merge(specs): Self (classmethod)
  │ is_uniform_with_collection(kv_cache_specs): bool
  │
  ├── AttentionSpec (基类, frozen dataclass)
  │   │ num_kv_heads: int
  │   │ head_size: int
  │   │ dtype: torch.dtype
  │   │ kv_quant_mode: KVQuantMode = NONE
  │   │ page_size_padded: int | None = None
  │   │
  │   │ page_size_bytes:
  │   │   → real_page_size_bytes + per-token-head scales (if quantized)
  │   │   → per-token-head: 2 × block_size × num_kv_heads × 4(float32 size)
  │   │   → → 调研: INT8/FP8 per-token-head量化需要额外scale存储!
  │   │
  │   ├── FullAttentionSpec ★★★★★ (最常用!)
  │   │   │ head_size_v: int = head_size (V维度, MLA用)
  │   │   │ sliding_window: int | None = None
  │   │   │ attention_chunk_size: int | None = None
  │   │   │
  │   │   │ real_page_size_bytes:
  │   │   │   → NVFP4: nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)
  │   │   │   → → packed layout per head: fp4 data + fp8 block scales
  │   │   │   → → → fp4 data: head_size//2 bytes (2 fp4 values per byte)
  │   │   │   → → → fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
  │   │   │   → Normal: block_size × num_kv_heads × (head_size + head_size_v) × dtype_size
  │   │   │
  │   │   │ max_memory_usage_bytes:
  │   │   │   → cdiv(max_model_len, block_size) × page_size_bytes
  │   │   │   → → DCP/PCP: max_model_len //= dcp_world_size × pcp_world_size
  │   │   │   → → → 数据并行下每rank只需保存部分tokens!
  │   │   │
  │   │   │ merge:
  │   │   │   → 所有层必须相同spec → 合并为一个manager
  │   │   │   → → sliding_window不能和chunked_local共存
  │   │   │   → → → 验证: "not both sliding_window and chunked_local"
  │   │   │
  │   │   ├── TQFullAttentionSpec
  │   │   │   │ tq_slot_size: int = 0
  │   │   │   │ → TQ-aware page size override (代替 raw head_size × dtype)
  │   │   │   │ → → real_page_size_bytes = block_size × num_kv_heads × tq_slot_size (if > 0)
  │   │   │
  │   │   ├── MLAAttentionSpec ★★★★★ (DeepSeek-V3/V4!)
  │   │   │   │ cache_dtype_str: str | None = None
  │   │   │   │ alignment: int | None = None (padding)
  │   │   │   │ compress_ratio: int = 1 (block size压缩)
  │   │   │   │ model_version: str | None = None
  │   │   │   │
  │   │   │   │ storage_block_size = block_size // compress_ratio
  │   │   │   │ → → 压缩后存储block更小 → 但逻辑block_size不变!
  │   │   │   │
  │   │   │   │ real_page_size_bytes:
  │   │   │   │   → fp8_ds_mla + deepseek_v4: 448B NoPE + 128B RoPE + 8B fp8 scale = **584B/tok**
  │   │   │   │   → → 硬编码layout! → 不用通用公式 → MLA特殊存储!
  │   │   │   │   → fp8_ds_mla + V3.2: **656B/tok** (kv_lora_rank=512 + qk_rope_head_dim=64)
  │   │   │   │   → Normal: storage_block_size × num_kv_heads × head_size × dtype_size
  │   │   │   │
  │   │   │   │ _apply_alignment_padding:
  │   │   │   │   → round_up(actual_page_size, alignment) → padded page
  │   │   │   │   → → FlashMLA需要特定alignment → 硬件对齐!
  │   │   │   │
  │   │   │   │ merge: 验证cache_dtype_str/compress_ratio/model_version一致
  │   │   │   │
  │   │   │   └── HiddenStateCacheSpec
  │   │   │       → Marker class for hidden-state cache layers
  │   │   │       → 用于 extract_hidden_states
  │   │   │
  │   │   └── SinkFullAttentionSpec ★★★ (StreamingLLM!)
  │   │       │ sink_len: int | None = None
  │   │       │ → sink_len tokens永久保留 + 后面是普通full attention
  │   │       │ → → 实现StreamingLLM的"attention sink + rolling window"
  │   │       │ → → → 但用的是SinkFullAttentionManager(不是FullAttentionManager)!
  │   │
  │   ├── SlidingWindowSpec ★★★ (窗口注意力)
  │   │   │ sliding_window: int
  │   │   │ head_size_v: int = head_size
  │   │   │
  │   │   │ max_admission_blocks_per_request ★★★:
  │   │   │   → min(sliding_window - 1 + max_num_batched_tokens, max_model_len)
  │   │   │   → → +1 (block可能不从开头开始)
  │   │   │   → → → 防止SWA请求预留过多blocks → 死锁!
  │   │   │   → → → → 单一真相源: startup pool sizing + runtime admission都用这个!
  │   │   │
  │   │   │ max_memory_usage_bytes:
  │   │   │   → DCP不支持sliding window! (assert dcp_world_size == 1)
  │   │   │   → → 因为DCP切分序列 → sliding window语义不同!
  │   │   │
  │   │   │ NVFP4: 同样支持FP4 packed layout (mirror FullAttentionSpec)
  │   │   │
  │   │   └── SlidingWindowMLASpec
  │   │       │ cache_dtype_str + alignment + compress_ratio + model_version
  │   │       │ → Sliding window + MLA存储格式 → DeepSeek-V4 hybrid!
  │   │       │ → → DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B/tok
  │   │
  │   ├── ChunkedLocalAttentionSpec
  │   │   │ attention_chunk_size: int
  │   │   │ → 也有max_admission_blocks_per_request → 防止过度预留
  │   │   │ → → is_uniform_with_collection: 验证所有层chunk_size相同
  │   │
  │   ├── EncoderOnlyAttentionSpec
  │   │   │ max_memory_usage_bytes = 0 → encoder-only不需要KV cache!
  │   │
  │   └── CrossAttentionSpec ★★ (encoder-decoder)
  │       │ max_memory_usage_bytes:
  │       │ → cdiv(max_encoder_len, block_size) × page_size_bytes
  │       │ → → 用encoder长度而非max_model_len!
  │       │ → → → Whisper等模型: encoder=1500 tokens → decoder逐步生成
  │
  └── MambaSpec ★★★ (SSM state cache!)
      │ shapes: tuple[tuple[int, ...], ...] → 多个state tensor的shape
      │ dtypes: tuple[torch.dtype] → 对应dtype
      │ mamba_type: MambaAttentionBackendEnum (MAMBA2)
      │ mamba_cache_mode: str ("none"/"align"/"all")
      │ num_speculative_blocks: int = 0
      │
      │ page_size_bytes:
      │ → sum(prod(shape) × dtype_size for each (shape, dtype))
      │ → → Mamba不是KV cache → 是SSM state → shape可以多维!
      │
      │ max_memory_usage_bytes:
      │ → "all": cdiv(max_model_len, block_size) + spec_blocks → 全序列缓存
      │ → "align": 2 + spec_blocks → 对齐模式(当前+前一步)
      │ → "none": 1 + spec_blocks → 最小模式(当前步)
      │ → → → Mamba不需要所有历史 → 只需最近几步state!

KVCacheSpecKind (9种enum):
  → full_attention / mla_attention / sliding_window / sliding_window_mla
  → mamba / chunked_local_attention / sink_full_attention
  → encoder_only_attention / cross_attention / unknown
```

## 2. KVQuantMode — 5种KV Cache量化模式

```
KVQuantMode(IntEnum):
  → NONE = 0 → BF16/FP16 无量化
  → FP8_PER_TENSOR = 1 → per-tensor scales (当前fp8路径)
  → → → vLLM现有FP8实现: 一个scale per KV cache tensor
  → → → → 最简单但精度最低(全局scale)
  │
  → INT8_PER_TOKEN_HEAD = 2 → per-token-head动态scales for int8
  → → → 每个token每个head独立scale → 更精确!
  → → → → page_size_bytes额外: 2 × block_size × num_kv_heads × 4(float32)
  → → → → → scale存储: min_scale + max_scale per token-head = 2个float32
  │
  → FP8_PER_TOKEN_HEAD = 3 → per-token-head动态scales for fp8
  → → → 同INT8 per-token-head但用FP8 → 精度更好
  → → → → 同样额外4 bytes per token-head for scales
  │
  → NVFP4 = 4 → packed fp4 data + fp8 block scales ★★★★★ (Blackwell!)
  → → → 下一代量化: FP4权重 + FP8 block scale
  → → → → 不是per-token量化 → 是per-block量化(每16元素1个scale)
  → → → → → head_size//2 bytes for fp4 (2 values packed per byte)
  → → → → → head_size//16 bytes for fp8 block scale
  → → → → → → nvfp4_kv_cache_full_dim(head_size) 计算packed维度

量化模式选择指南(RTX 4090 vs H100 vs B200):
  → RTX 4090: INT8 per-token-head (FP8 per-tensor也行但精度低)
  → → → FP8 per-token-head: SM89支持但不是WGMMA → 用HMMA.16832
  → → → NVFP4: SM89不支持 → Blackwell only!
  → → → → → RTX 4090最优KV量化=INT8 per-token-head+GQA-5→65x并发
  │
  → H100: FP8 per-tensor或per-token-head (WGMMA支持)
  → → → per-tensor足够(大batch下per-token overhead大)
  → → → → → FlashInfer FP8 KV: cos_sim=0.999996 → 精度完美
  │
  → B200: NVFP4 ★★★★★ (终极KV量化!)
  → → → FP4 packed data: 7B → KV 1/4 size → 4x并发!
  → → → → → 但精度需验证 → fp8 block scale补偿
  → → → → → → 生产需fused kernel → 硬件NVFP4解码器

get_kv_quant_mode(kv_cache_dtype)映射:
  → "int8_per_token_head" → INT8_PER_TOKEN_HEAD
  → "fp8_per_token_head" → FP8_PER_TOKEN_HEAD
  → "nvfp4" → NVFP4
  → "fp8*" → FP8_PER_TENSOR (任何fp8前缀)
  → 其他 → NONE
```

## 3. KVCacheCoordinator — 3种协调器

```
get_kv_cache_coordinator()选择逻辑:
  → !enable_caching → KVCacheCoordinatorNoPrefixCache
  → → → 1个group → UnitaryKVCacheCoordinator
  → → → >1个group → HybridKVCacheCoordinator

┌─────────────────────────────────────────────────────────────────┐
│ KVCacheCoordinator (ABC)                                       │
│                                                                 │
│ 核心组件:                                                       │
│   block_pool: BlockPool (共享! 所有manager共用)                │
│   single_type_managers: tuple[SingleTypeKVCacheManager]       │
│   eagle_group_ids: set[int] (EAGLE/MTP draft组)               │
│   retention_interval: int | None (前缀缓存保留间隔)            │
│                                                                 │
│ 核心方法:                                                       │
│   get_num_blocks_to_allocate → sum across all managers         │
│   allocate_new_computed_blocks → per-manager                   │
│   allocate_new_blocks → per-manager                            │
│   cache_blocks → per-manager + retention_interval              │
│   free → per-manager                                           │
│   get_num_common_prefix_blocks → per-manager                   │
│   remove_skipped_blocks → per-manager                          │
│   find_longest_cache_hit → ABC (子类实现)                      │
│                                                                 │
│   CrossAttentionManager特殊处理:                               │
│   → 用num_encoder_tokens而非num_tokens                        │
│   → → encoder-decoder模型: encoder固定长度                    │
└─────────────────────────────────────────────────────────────────┘

┌─── KVCacheCoordinatorNoPrefixCache ───┐
│ find_longest_cache_hit → ([], 0)     │
│ → 不做前缀匹配 → 简单但低效          │
│ → 支持0个group → 无KV cache也可      │
└───────────────────────────────────────┘

┌─── UnitaryKVCacheCoordinator ─────────┐
│ 单一KV cache类型(标准LLM)            │
│ find_longest_cache_hit:               │
│ → single_type_managers[0].find_hit  │
│ → → hash_block_size == block_size     │
│ → → → DCP/PCP: block_size *= world   │
│ → → → → 数据并行下block_size更大     │
│                                       │
│ eagle: single_type_managers[0].use_eagle │
│ → 单组 → eagle标记直接设置           │
└─────────────────────────────────────────┘

┌─── HybridKVCacheCoordinator ─────────────────────────────────┐
│ 多KV cache类型(DeepSeek-V3/V4, Gemma-2等)                  │
│                                                               │
│ SpecGroup(NamedTuple):                                       │
│   spec: KVCacheSpec                                          │
│   group_ids: list[int] ← 相同spec的组batched在一起          │
│   manager_cls: type[SingleTypeKVCacheManager]                │
│   use_eagle: bool ← EAGLE/MTP draft                         │
│                                                               │
│ verify_and_split_kv_cache_groups():                          │
│   → 按spec分组 → 相同spec的group_ids合并                    │
│   → → FullAttentionSpec放第一! (左→右扫描提供tight bound)   │
│   → → → assert len > 1 (至少2组)                             │
│                                                               │
│ find_longest_cache_hit: ★★★★★ 迭代固定点算法!              │
│                                                               │
│   初始: hit_length = max_cache_hit_length                    │
│                                                               │
│   while True:                                                 │
│     for each SpecGroup:                                       │
│       if FullAttentionSpec + already cached:                  │
│         → trim to curr_hit_length (downward-closed!)         │
│         → → full attn不需要重新查找 → 只截断                 │
│         → → → 因为full attn是单调的: 匹配N→匹配任何≤N       │
│                                                               │
│       if drop_eagle_block:                                    │
│         → match 1 more block → pop last → 真正匹配           │
│         → → EAGLE draft需要额外1块验证                       │
│                                                               │
│       hit_blocks = manager_cls.find_longest_cache_hit(...)   │
│       curr_hit_length = len(hit_blocks[0]) × spec.block_size │
│                                                               │
│       if length shrunk:                                       │
│         → clear eagle_verified → 重新验证所有eagle组         │
│         → → → 因为长度变了 → eagle drop也需要重新检查       │
│                                                               │
│     if curr_hit_length >= hit_length:                         │
│       break → 收敛!                                           │
│                                                               │
│     hit_length = curr_hit_length                              │
│     if simple_hybrid (2组: full + 1 other):                   │
│       break → 1次迭代足够!                                    │
│                                                               │
│   收敛保证: 长度单调递减 + 下界≥0 → 必然收敛               │
│                                                               │
│   最后: 截断full attn blocks到最终hit_length                 │
│                                                               │
│ cache_blocks 特殊处理:                                        │
│   → aligned_num_computed_tokens (scheduler_block_size对齐)   │
│   → → EAGLE组: +1 block → 使lookahead block可缓存           │
│   → → → retention_interval传递给每个manager                  │
│                                                               │
│ DCP/PCP不支持! (assert dcp/pcp_world_size == 1)             │
│ → → hybrid + 数据并行 → 太复杂 → 当前不支持                 │
└───────────────────────────────────────────────────────────────┘
```

## 4. Spec Registry — 可插拔架构

```
KVCacheSpecRegistry设计:
  → 全局字典 _REGISTRY_KVCACHESPEC_LIST: dict[type, KVCacheSpecMetadata]
  → → KVCacheSpecMetadata: kvcache_spec_cls + manager_class + uniform_type_base_spec
  │
  │ 核心概念: uniform_type_base_spec ★★★
  │ → 决定KV cache分组 → 相同base spec的层归入一组
  │ → → FullAttentionSpec base: TQ/MLA/HiddenState/Sink都归入FullAttention组!
  │ → → → → 生产意义: DeepSeek-V3 MLA层和普通FullAttention层共享block table!
  │ → → → → → → 但SinkFullAttention有自己的manager (Sink需要特殊处理)
  │
  │ 查找逻辑: MRO walk
  │ → for base in kvcache_spec_cls.__mro__:
  │ → → if base in _REGISTRY_KVCACHESPEC_LIST: return metadata
  │ → → → → 子类继承父类的注册 → 自动路由到正确manager
  │
  │ @register_kv_cache_spec decorator ★★★:
  │ → out-of-tree平台可以注册自定义spec → 不修改vLLM核心代码!
  │ → → platform通过current_platform.register_custom_kv_cache_specs()注入
  │ → → → → Neuwa/Intel等自定义硬件可以添加专属KV cache类型

register_all_kvcache_specs()内置注册:
  → FullAttentionSpec → FullAttentionManager (base=FullAttention)
  → SlidingWindowSpec → SlidingWindowManager (base=SlidingWindow)
  → SlidingWindowMLASpec → SlidingWindowManager (base=SlidingWindowMLA)
  → MambaSpec → MambaManager (base=MambaSpec)
  → ChunkedLocalAttentionSpec → ChunkedLocalAttentionManager (base=ChunkedLocal)
  → CrossAttentionSpec → CrossAttentionManager (base=CrossAttention)
  → TQFullAttentionSpec → FullAttentionManager (base=FullAttention) ★★
  → → → TQ和FullAttention共享manager → 只是存储layout不同
  → MLAAttentionSpec → FullAttentionManager (base=FullAttention) ★★★★★
  → → → MLA用FullAttentionManager管理! → MLA不特殊 → 只是存储不同!
  → → → → → 这解释了为什么vLLM MLA不是独立manager → MLA逻辑在kernel层
  → HiddenStateCacheSpec → FullAttentionManager (base=FullAttention)
  → SinkFullAttentionSpec → SinkFullAttentionManager (base=FullAttention) ★★
  → → → Sink用不同manager! → 因为sink tokens需要特殊保留逻辑
```

## 5. NVFP4 KV Cache Layout — Blackwell下一代量化

```
NVFP4设计(vLLM实现):
  → fp4 packed data: 2 fp4 values per byte → head_size//2 bytes
  → fp8 block scale: 1 scale per 16 elements → head_size//16 bytes
  → → → total per token per head: nvfp4_kv_cache_full_dim(head_size) bytes
  │
  │ FullAttentionSpec.real_page_size_bytes (NVFP4):
  │   → block_size × num_kv_heads × nvfp4_kv_cache_full_dim(head_size + head_size_v) × dtype_size
  │   → → → head_size_v = V维度(MLA用) → K和V都量化!
  │
  │ SlidingWindowSpec/SinkFullAttentionSpec: 同样支持NVFP4
  │ → mirror FullAttentionSpec的NVFP4 layout → 统一设计
  │
  │ 但! AttentionSpec.real_page_size_bytes (NVFP4):
  │   → 2 × block_size × num_kv_heads × full_dim × dtype_size
  │   → → → 注意! 这是基类 → 不区分K/V → 总是×2 → 可能不准确?
  │ → → → → → FullAttentionSpec覆盖了这个 → 用(head_size + head_size_v)
  │ → → → → → → 所以只有基类计算不准确 → 子类覆盖修正!

NVFP4 KV vs 其他量化对比:
  │ QuantMode │ 每tok每head(K) │ Scale存储 │ 7B GQA-5 KV/tok │
  │ FP16(NONE)│ 128B           │ 0         │ 81.92 KB        │
  │ FP8_per_t │ 64B            │ 4B/tensor │ 40.96 KB        │
  │ FP8_per_th│ 64B            │ 8B/t-h    │ 40.96 KB+0.08KB │
  │ INT8_per_t│ 64B            │ 8B/t-h    │ 40.96 KB+0.08KB │
  │ NVFP4     │ ~16B           │ 8B/16el   │ ~10 KB          │ ← 8x省!
  │ → → → NVFP4: 128B→~16B → 8x KV省 → 但仅B200支持!
```

## 6. MLA KV Cache Layout — DeepSeek-V3/V4特殊存储

```
MLA存储(vLLM实现):
  → DeepSeek-V4 (fp8_ds_mla + deepseek_v4):
  │   → 448B NoPE (compressed latent, kv_lora_rank=512)
  │   → 128B RoPE (qk_rope_head_dim=64 × 2 for K+V? → 实际是position encoding)
  │   → 8B fp8 scale (per-token quantization scale)
  │   → → total = 584B/tok → 硬编码! 不用通用公式
  │
  │ DeepSeek-V3.2 (fp8_ds_mla):
  │   → 656B/tok → kv_lora_rank=512 + qk_rope_head_dim=64
  │   → → → 更大的layout → 包含更多RoPE信息
  │
  │ Normal MLA (BF16):
  │   → storage_block_size × num_kv_heads × head_size × dtype_size
  │   → → → 通用公式 → head_size=512 → 1 head × 512 × 2 = 1024B/tok

compress_ratio设计:
  → block_size // compress_ratio = storage_block_size
  → → compress_ratio > 1 → 存储block比逻辑block更小
  → → → → DeepSeek-V4: compress_ratio可能>1 → MLA压缩存储

alignment padding:
  → _apply_alignment_padding(spec):
  │   → round_up(actual_page_size, alignment) → padded page
  │   → → → FlashMLA需要特定alignment → 硬件对齐要求!
  │   → → → → 不aligned → FlashMLA kernel性能差或crash

MLA vs FullAttention管理:
  → MLAAttentionSpec → FullAttentionManager (共享!)
  → → → MLA不需要特殊block管理 → 只是存储layout不同
  → → → → → block allocation逻辑和full attention一样
  → → → → → → MLA的特殊逻辑在FlashMLA kernel层面 → 不在KV cache管理层面
```

## 7. Admission Cap — SWA/Chunked-Local防死锁

```
问题: Sliding Window请求可能预留过多blocks → 死锁!
  → SWA请求只需要sliding_window内的blocks → 但可能预留更多
  → → → 如果所有请求都预留max_model_len → pool耗尽 → 没请求能完成 → 死锁!

解决方案: max_admission_blocks_per_request ★★★
  → 单一真相源: startup pool sizing + runtime admission都用同一个cap
  → → → 确保startup预留和runtime分配一致 → 无死锁!

SlidingWindowSpec.max_admission_blocks_per_request:
  → num_tokens = min(sliding_window - 1 + max_num_batched_tokens, max_model_len)
  → → → sliding_window内历史 + 当前batch → 但不超过max_model_len
  → → → → cdiv(num_tokens, block_size) + 1 ← +1因为block可能不从开头!
  │
  │ 例: sliding_window=4096, max_batched_tokens=2048, block_size=16, max_model_len=8192
  │ → num_tokens = min(4095 + 2048, 8192) = 6143
  │ → blocks = cdiv(6143, 16) + 1 = 384 + 1 = 385 blocks
  │ → → → 远小于8192/16=512 → 留出空间给其他请求!

ChunkedLocalAttentionSpec.max_admission_blocks_per_request:
  → min(attention_chunk_size + max_num_batched_tokens, max_model_len)
  → → → chunked prefill只需1 chunk window + 当前batch

get_manager_for_kv_cache_spec:
  → SlidingWindowSpec/ChunkedLocalAttentionSpec → 自动设置admission cap
  → → → 其他spec → None (无cap) → full attention不过度预留
```

## 8. HybridKVCacheCoordinator Cache Hit算法详解

```
场景: DeepSeek-V3 → FullAttention + MLA + SlidingWindow → 3组KV cache
  → 3个SingleTypeKVCacheManager → 但共享BlockPool
  → → 请求到来 → 需要在所有组中找最长前缀匹配 → 但各组语义不同!

算法: 迭代固定点(fixed-point)
  → 初始candidate: max_cache_hit_length (最大可能匹配长度)
  │
  │ Iteration 1:
  │   FullAttention: 找最长匹配 → hit_length = 4096 tokens
  │   → → downward-closed: 匹配4096 → 匹配任何≤4096
  │   → → → 一次查找即可 → 后续只trim
  │
  │   MLA: 同FullAttention(用FullAttentionManager!) → hit_length可能减少
  │
  │   SlidingWindow: 只匹配窗口内 → hit_length可能进一步减少!
  │   → → → SWA只看最近W tokens → 前面的不匹配 → 长度减少
  │
  │ 如果hit_length减少了 → 重启 → FullAttention trim到新长度 → 再次检查
  │
  │ Iteration 2 (如果需要):
  │   FullAttention: trim到新hit_length → 不重新查找
  │   其他组: 在新长度下查找 → 可能进一步减少
  │
  │ 收敛: 长度单调递减 → ≥0 → 必然收敛
  │ → → simple_hybrid(2组): 1次迭代足够 → full + 1 other
  │ → → → complex hybrid(3+组): 可能需要多次 → 但很少超过2-3次

EAGLE/MTP draft特殊处理:
  → EAGLE组需要match 1 more block → pop last → 真正匹配
  → → → 因为draft model的最后一块是lookahead → 不应该缓存
  → → → → → drop_eagle_block: 验证1 extra block → 然后丢弃
  │
  │ eagle_verified: set[int] → 已验证的eagle组
  │ → → 如果长度减少 → clear eagle_verified → 重新验证
  │ → → → → 防止旧验证在新长度下无效

Simple Hybrid优化:
  → 2组(Full + 1 Other): 1次迭代足够
  → → → break early → 不需要多次 → 性能好!
```

## 9. KVCacheConfig — 模型KV cache配置

```
KVCacheConfig(dataclass):
  → num_blocks: int → 总GPU block数
  → kv_cache_tensors: list[KVCacheTensor] → 初始化方式
  │ → → KVCacheTensor: size(bytes) + shared_by(layer_names)
  │ → → → → 相同KV cache tensor被多层共享 → 省内存!
  │
  → kv_cache_groups: list[KVCacheGroupSpec]
  │ → → KVCacheGroupSpec:
  │ → → → layer_names: list[str] → 组内层名
  │ → → → kv_cache_spec: KVCacheSpec → 组规格
  │ → → → is_eagle_group: bool → EAGLE/MTP draft标记
  │
  │ 单一类型模型(如LLaMA):
  │ → → 1个group → 所有层归入FullAttention → UnitaryKVCacheCoordinator
  │
  │ DeepSeek-V3:
  │ → → 可能3+组 → MLA层 + dense层 → HybridKVCacheCoordinator
  │ → → → → MLA和dense可能共享block table(如果base spec相同)

  has_mamba_layers → bool
  → → any(isinstance(g.kv_cache_spec, MambaSpec))

  needs_kv_cache_zeroing → bool
  → → has_mamba_layers → Mamba需要清零 → SSM state必须初始化为0!
  → → → → vs Attention: 不需要清零 → 直接写入覆盖旧值
```

## 10. vLLM V1 vs V0 KV Cache架构对比

```
V0 (旧架构):
  → 单一KV cache类型 → BlockSpaceManager → 所有层一样
  → → → 不支持混合模型 → DeepSeek-V3/V4无法正确管理
  → → → → MLA/SWA/Mamba → 统一按full attention处理 → 内存浪费!
  │
  │ 前缀匹配:
  │   → BlockHash → hash表 → 简单O(1)查找
  │   → → → 不考虑SWA/MLA差异 → 可能错误匹配

V1 (新架构):
  → Multi-type KV cache → KVCacheCoordinator → 每种类型独立管理
  → → → 支持混合模型 → DeepSeek-V3/V4, Gemma-2等正确管理
  → → → → MLA特殊layout + SWA admission cap + Mamba state cache
  │
  │ 前缀匹配(Hybrid):
  │   → Iterative fixed-point → 跨组协调 → 收敛保证
  │   → → → Full attn first → tight bound → 其他组trim
  │   → → → → EAGLE draft特殊处理 → lookahead block drop
  │
  │ 可插拔:
  │   → KVCacheSpecRegistry → @register_kv_cache_spec → out-of-tree支持
  │   → → → Neuwa/Intel等自定义硬件可以注册专属spec → 不改核心代码

  量化:
  │ → V0: fp8 per-tensor only → 简单但精度低
  │ → V1: 5种量化模式 → NVFP4 for Blackwell → per-token-head for H100/RTX4090
  │ → → → → 精度更高 + 并发更多 + 但实现更复杂

  DCP/PCP数据并行:
  │ → V0: 不支持KV cache数据并行
  │ → V1: max_model_len //= dcp × pcp → 每rank只存部分tokens
  │ → → → → 但Hybrid不支持DCP/PCP → 太复杂
```

## 11. RTX 4090 Implications

```
RTX 4090 KV cache最优配置:
  → KVQuantMode = INT8_PER_TOKEN_HEAD → cos_sim=0.999965 → 精度够
  → → → FP8_PER_TENSOR也行 → cos_sim=0.999996 → 但scale精度差
  → → → → NVFP4不支持 → SM89没有FP4 Tensor Core!
  │
  │ KVCacheSpec = FullAttentionSpec → 标准LLaMA/Qwen
  → → → SlidingWindow: Gemma-2但RTX 4090小模型 → window不重要
  → → → → MLA: DeepSeek-V3 → 但RTX 4090不支持FlashMLA(SM89)!
  → → → → → → RTX 4090最优=GQA-5+INT8 KV → 不是MLA!

  Coordinator选择:
  → → 标准模型 → UnitaryKVCacheCoordinator → 简单高效
  → → → Hybrid模型 → HybridKVCacheCoordinator → 更复杂但支持更多模型类型
  │
  │ Admission Cap:
  → → RTX 4090 24GB → 内存紧张 → admission cap很重要!
  → → → → SWA请求: admission cap防止过度预留 → 防死锁
  → → → → → → full attention请求: 不需要cap → 但也受限于pool大小

  未来方向:
  → NVFP4 → Blackwell only → RTX 4090无法使用
  → → → 但vLLM代码已经准备好 → B200部署时直接启用
  → → → → → FP4 KV: 8x省 → 7B → ~10KB/tok → ~2400并发 → vs INT8 ~240并发
```

## 参考文献

```
1. vllm/v1/kv_cache_interface.py — KVCacheSpec 9种类型 + KVQuantMode + NVFP4 layout
2. vllm/v1/core/kv_cache_coordinator.py — 3种Coordinator + fixed-point算法
3. vllm/v1/core/single_type_kv_cache_manager.py — Manager hierarchy + Registry dispatch
4. vllm/v1/kv_cache_spec_registry.py — 可插拔Spec Registry + @register_kv_cache_spec
5. vllm/utils/torch_utils.py — nvfp4_kv_cache_full_dim() utility

我们的笔记:
- kv-cache-management-deep-dive.md — KV cache理论
- flashinfer-attention-deep-dive.md — FlashInfer实测
- scheduler-architecture-deep-dive.md — vLLM/SGLang调度器
- mla-architecture-deep-dive.md — MLA架构理论