# FlashInfer 架构深度分析

> 源码: https://github.com/flashinfer-ai/flashinfer
> 文档: https://docs.flashinfer.ai
> 版本: 0.6.12
> 论文: arxiv 2501.01005 (MLSys 2025)
> 分析日期: 2026-06-05

## 1. 架构总览

FlashInfer 是专为 LLM 推理服务设计的高性能 GPU attention kernel 库，被 vLLM、SGLang、TensorRT-LLM 等主流框架采用。核心设计理念是 **Plan/Run 两阶段执行模式**，将不可被 CUDA Graph 捕获的辅助数据结构构建 (plan) 与可被捕获的 attention 计算 (run) 分离。

```
FlashInfer 模块结构
├── decode          # Decode attention (1 query token per request)
├── prefill         # Prefill/Append attention (variable query tokens)
├── mla             # Multi-head Latent Attention (DeepSeek MLA)
├── xqa             # TensorRT-LLM XQA decode kernel
├── gemm            # FP8/FP4/BF16 矩阵乘法
├── fused_moe       # 融合 MoE 算子
├── cascade         # Cascade attention (prefix 共享)
├── comm            # 通信 (AllReduce, All-to-All)
├── page            # Paged KV cache 操作
├── sampling        # 采样算子 (top-k, top-p, min-p)
├── topk            # TopK 选择
├── logits_processor # Logits 处理管线
├── norm            # RMSNorm/LayerNorm
├── rope            # RoPE 位置编码
├── activation      # SiLU, GELU 激活函数
├── quantization    # 量化 (packbits)
├── fp4_quantization # NVFP4 量化
├── green_ctx       # Green Context (GPU 资源分配)
└── testing         # Benchmark 工具
```

### Wrapper 层次结构

```
                        ┌─────────────────────────┐
                        │  用户代码 (vLLM/SGLang)  │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  Wrapper (Plan/Run 模式) │
                        │  - plan(): 构建辅助数据   │
                        │  - run(): 执行 attention  │
                        └────────────┬────────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                   │
    ┌─────────────▼─────┐ ┌─────────▼──────────┐ ┌─────▼──────────────┐
    │ BatchDecodeWrapper│ │ BatchPrefillWrapper │ │ BatchMLAWrapper   │
    │ (PagedKV/Ragged) │ │ (PagedKV/Ragged)    │ │ (DeepSeek MLA)    │
    └─────────────┬─────┘ └─────────┬──────────┘ └─────┬──────────────┘
                  │                  │                   │
    ┌─────────────▼──────────────────▼───────────────────▼──────────┐
    │                     Backend 自动选择                           │
    │  fa2 | fa3 | cudnn | trtllm-gen | cute-dsl | cutlass         │
    │  (基于 GPU 架构自动选择最优后端)                                │
    └──────────────────────────────────────────────────────────────┘
```

---

## 2. Attention 后端详解

### 2.1 BatchDecodeWithPagedKVCacheWrapper

Decode attention: 每个请求只有 **1 个 query token**，attention 计算是 memory-bound。

```python
# 典型使用模式
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

# 创建 Wrapper
decode_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
    float_workspace_buffer=workspace_buffer,
    kv_layout="NHD",              # KV cache 布局
    use_cuda_graph=False,         # 是否启用 CUDA Graph
    use_tensor_cores=False,       # 大 GQA group 时用 Tensor Core
    backend="auto",               # auto 自动选择后端
)

# Plan 阶段: 构建 auxiliary data structures
decode_wrapper.plan(
    indptr=kv_page_indptr,        # [batch_size+1] CSR indptr
    indices=kv_page_indices,      # [total_pages] 页索引
    last_page_len=kv_last_page_len, # [batch_size] 最后一页有效长度
    num_qo_heads=64,              # query/output head 数
    num_kv_heads=8,               # KV head 数 (GQA)
    head_dim=128,                 # head 维度
    page_size=16,                 # 页大小
    pos_encoding_mode="NONE",     # 位置编码: NONE/ROPE_LLAMA/ALIBI
    window_left=-1,               # 滑动窗口 (-1=无限制)
    logits_soft_cap=None,         # logits soft cap (Gemma-2 等)
    data_type=torch.float16,
    q_len_per_req=1,              # 每次 query token 数 (默认 1)
)

# Run 阶段: 可被 CUDA Graph 捕获
for layer_idx in range(num_layers):
    q = get_query(layer_idx)                    # [batch, num_qo_heads, head_dim]
    kv_cache = get_kv_cache(layer_idx)          # [max_pages, 2, page_size, num_kv_heads, head_dim]
    output = decode_wrapper.run(q, kv_cache)    # [batch, num_qo_heads, head_dim]
```

**关键特性**:
- **Split-KV 算法**: 长序列 KV cache 被分片到多个 CTA (Cooperative Thread Array) 并行处理，最后 merge 结果
- **GQA 支持**: `num_qo_heads > num_kv_heads` 时自动启用 Grouped Query Attention
- **FP8 支持**: 通过 `q_scale`, `k_scale`, `v_scale` 参数支持 FP8 量化 attention
- **NVFP4 KV Cache**: 通过 `kv_cache_sf` 参数支持 per-block scale factors

### 2.2 BatchPrefillWithPagedKVCacheWrapper

Prefill attention: 每个请求有 **变长 query tokens**，attention 计算是 compute-bound。

```python
prefill_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
    float_workspace_buffer=workspace_buffer,
    kv_layout="NHD",
    use_cuda_graph=False,
    backend="auto",
)

# Plan: 需要额外的 qo_indptr 描述变长 query
prefill_wrapper.plan(
    qo_indptr=qo_indptr,              # [batch_size+1] query indptr
    paged_kv_indptr=paged_kv_indptr,  # [batch_size+1] KV indptr
    paged_kv_indices=kv_page_indices, # [total_pages]
    paged_kv_last_page_len=kv_last_page_len,
    num_qo_heads=64,
    num_kv_heads=16,
    head_dim_qk=128,                  # QK head dim (可与 VO 不同!)
    head_dim_vo=128,                  # VO head dim
    page_size=16,
    causal=True,                      # causal mask
    custom_mask=None,                 # 自定义 mask (packbits 格式)
    window_left=-1,                   # 滑动窗口
    logits_soft_cap=None,             # soft cap
)
```

**关键特性**:
- **异构 head dim**: `head_dim_qk` 和 `head_dim_vo` 可以不同，支持特殊模型架构
- **Custom mask**: 通过 `flashinfer.quantization.packbits()` 编码的位压缩 mask
- **Causal mask**: 内置 causal attention 支持，比 custom mask 更高效
- **Prefix 长度**: `prefix_len_ptr` 参数支持 prefix 共享场景

### 2.3 BatchPrefillWithRaggedKVCacheWrapper

Ragged (非分页) KV cache 的 prefill attention。适用于不需要分页管理的场景 (如 prefill-only 或 KV 连续存储)。

```python
ragged_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
    float_workspace_buffer=workspace_buffer,
    kv_layout="NHD",
)

# Plan: 使用 kv_indptr 代替分页参数
ragged_wrapper.plan(
    qo_indptr=qo_indptr,     # query indptr
    kv_indptr=kv_indptr,     # KV indptr (ragged, 连续存储)
    num_qo_heads=64,
    num_kv_heads=16,
    head_dim_qk=128,
    causal=True,
)

# Run: 直接传入 K, V tensor (不传 paged_kv_cache)
output = ragged_wrapper.run(q, k, v)
```

### 2.4 BatchMLAPagedAttentionWrapper

DeepSeek-V2/V3/R1 的 Multi-head Latent Attention (MLA)，使用 **Matrix Absorption** 技巧。

```python
mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    float_workspace_buffer=workspace_buffer,
    backend="fa2",
)

mla_wrapper.plan(
    qo_indptr=q_indptr,       # decode 时为 [0,1,2,...,batch_size]
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    kv_len_arr=kv_lens,       # [batch_size] KV 长度
    num_heads=128,             # num_local_heads
    head_dim_ckv=512,          # compressed KV dim
    head_dim_kpe=64,           # RoPE K dim
    page_size=1,
    causal=False,
    sm_scale=1.0 / ((128 + 64) ** 0.5),  # 使用吸收前的 head dim
)

# Run: 分离的 q_nope, q_pe, ckv_cache, kpe_cache
output = mla_wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache)
```

**MLA 特殊之处**:
- `q_nope` [batch, num_heads, head_dim_ckv]: 不带 RoPE 的 query 部分
- `q_pe` [batch, num_heads, head_dim_kpe]: 带 RoPE 的 query 部分
- `ckv_cache`: 压缩的 KV cache (head_dim=512 vs 标准 128)
- `kpe_cache`: RoPE 位置的 K cache (head_dim=64)
- sm_scale 使用 Matrix Absorption 前的原始维度计算

### 2.5 XQA Kernel

TensorRT-LLM 的 XQA decode kernel，支持 **Speculative Decoding** (q_seq_len > 1)。

```python
flashinfer.xqa.xqa(
    q, k_cache, v_cache, page_table, seq_lens,
    output, workspace_buffer, semaphores,
    num_kv_heads, page_size,
    q_seq_len=1,                # >1 时启用 speculative decoding
    mask=causal_mask,           # speculative mask (bit-packed uint16)
    sliding_win_size=0,
    kv_layout="NHD",
    enable_pdl=True,            # Hopper+ PDL 优化
)
```

---

## 3. 关键优化技术

### 3.1 Paged KV Cache 页表间接寻址

FlashInfer 的核心创新之一: 通过 **页表间接寻址** 实现 KV cache 的灵活内存管理。

```
传统连续 KV Cache:
  请求 1: [K1_0, K1_1, K1_2, ..., K1_n]  ← 必须连续分配
  请求 2: [K2_0, K2_1, ..., K2_m]        ← 浪费内存碎片

Paged KV Cache (vLLM 风格):
  物理页表:  Page 0  Page 1  Page 2  Page 3  Page 4  Page 5
             [data]  [data]  [data]  [data]  [data]  [data]

  请求 1: indptr[0]=0, indices=[3,1,5], last_page_len=7
           → Page3 + Page1 + Page5(前7个)

  请求 2: indptr[1]=3, indices=[0,2,4], last_page_len=16
           → Page0 + Page2 + Page4(满)
```

**三个核心数据结构**:
- `indptr` [batch_size+1]: CSR 格式的行指针，类似 CuPy 的 indptr
- `indices` [total_pages]: 所有请求使用的页索引
- `last_page_len` [batch_size]: 每个请求最后一页的有效 token 数 (1 <= x <= page_size)

### 3.2 KV Cache 布局

FlashInfer 支持两种 KV cache 内存布局:

```
NHD 布局 (默认): [num_pages, page_size, num_kv_heads, head_dim]
  - 页内 token 连续
  - 适合 decode (访问模式: 读所有 head 的连续 token)

HND 布局: [num_pages, num_kv_heads, page_size, head_dim]
  - Head 维度在前
  - 适合 prefill (访问模式: 按 head 批量读取)

5D 合并格式:
  NHD: [num_pages, 2, page_size, num_kv_heads, head_dim]
  HND: [num_pages, 2, num_kv_heads, page_size, head_dim]
  其中 dim=1 的 0 是 K, 1 是 V
```

### 3.3 Plan/Run 模式与 CUDA Graph 兼容性

这是 FlashInfer 对 LLM serving 的关键优化: **将 attention 执行分为两个阶段**。

```
┌─────────────────────────────────────────────────────┐
│  Plan 阶段 (每步执行一次, 不可被 CUDA Graph 捕获)     │
│                                                      │
│  1. 复制 indptr/indices/last_page_len 到 GPU         │
│  2. 计算每个请求的 KV 长度                            │
│  3. 构建 Split-KV 调度参数                            │
│  4. 分配 workspace buffer                            │
│  5. 选择最优 kernel 变体                              │
│                                                      │
│  输出: 辅助数据结构缓存在 wrapper 内部                  │
│  注意: 对所有层只 plan 一次, run 多次                   │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  Run 阶段 (每层执行一次, 可被 CUDA Graph 捕获)         │
│                                                      │
│  1. 读取缓存的辅助数据结构                             │
│  2. 执行 attention 计算                               │
│  3. 返回输出 tensor                                   │
│                                                      │
│  关键: 输入输出的 tensor 地址/shape 不变                │
│  → 可以被 CUDA Graph 捕获和重放                        │
└─────────────────────────────────────────────────────┘
```

**CUDA Graph 专用 Wrapper**:

```python
# 专用 CUDA Graph Wrapper: 预分配所有 buffer
cg_wrapper = flashinfer.decode.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer=workspace,          # workspace buffer
    indptr_buffer=indptr_buf,            # 预分配 [max_batch+1]
    indices_buffer=indices_buf,          # 预分配 [max_pages]
    last_page_len_buffer=lp_buf,         # 预分配 [max_batch]
    kv_layout="NHD",
)
# 注意: 这个 wrapper "may not be as efficient" 因为不会为不同
# batch size 选择不同的 kernel, 但可以安全地在 CUDA Graph 中使用
```

**使用 `use_cuda_graph=True` 的通用 Wrapper**:
```python
# 也可以在通用 Wrapper 上启用 CUDA Graph
wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer,
    "NHD",
    use_cuda_graph=True,
    paged_kv_indptr_buffer=preallocated_indptr,       # 必须预分配
    paged_kv_indices_buffer=preallocated_indices,     # 必须预分配
    paged_kv_last_page_len_buffer=preallocated_lpl,   # 必须预分配
)
# 启用后 batch_size 不能改变
```

### 3.4 滑动窗口 Attention

通过 `window_left` 参数控制左侧窗口大小:

```python
# 标准全局 attention
wrapper.plan(..., window_left=-1)      # 无限制

# 滑动窗口 attention (Mistral/Gemma 等)
wrapper.plan(..., window_left=4096)    # 只关注最近 4096 个 token

# 也可以在 run() 时覆盖
output = wrapper.run(q, kv_cache, window_left=2048)
```

### 3.5 FP8 和 NVFP4 支持

```python
# FP8 attention: 传入 scale factors
output = wrapper.run(
    q, kv_cache,
    q_scale=0.5,     # query 反量化 scale
    k_scale=0.3,     # key 反量化 scale
    v_scale=0.4,     # value 反量化 scale
)

# NVFP4 KV Cache: per-block scale factors
output = wrapper.run(
    q, kv_cache,
    kv_cache_sf=scale_factors,  # torch.float8_e4m3fn
    # shape: [num_pages, page_size, num_kv_heads, head_dim//16]
)
```

### 3.6 Split-KV 算法

长序列 decode 时，单个 CTA 无法处理所有 KV。Split-KV 将 KV 分片到多个 CTA 并行处理:

```python
# 控制 Split-KV 行为
wrapper.plan(
    ...,
    fixed_split_size=512,    # 固定分片大小 (以 page 为单位)
    disable_split_kv=False,  # 禁用 split-kv (确定性输出)
)
```

- `fixed_split_size`: 设为平均序列长度，可获得 **确定性** softmax 结果 (消除浮点非确定性)
- `disable_split_kv=True`: 禁用 split-kv，保证 CUDA Graph 兼容性

### 3.7 Logits Soft Capping

支持 Gemini/Grok/Gemma-2 的 attention logits soft cap:

```python
# 公式: logits_soft_cap * tanh(x / logits_soft_cap)
wrapper.plan(..., logits_soft_cap=50.0)
```

### 3.8 PDL (Programmatic Dependent Launch)

Hopper (SM90+) GPU 的优化:

```python
# 启用 PDL: 让后续 kernel 在当前 kernel 的 output 就绪时立即启动
output = wrapper.run(q, kv_cache, enable_pdl=True)
```

---

## 4. Backend 自动选择

FlashInfer 支持多种后端，根据 GPU 架构自动选择:

| Backend | 适用 GPU | 特点 |
|---------|---------|------|
| `fa2` | 所有 CUDA GPU | FlashAttention-2 风格，最稳定 |
| `fa3` | Hopper (SM90+) | FlashAttention-3，利用 Hopper 新特性 |
| `cudnn` | 所有 CUDA GPU | cuDNN 后端，某些配置下更快 |
| `trtllm-gen` | Hopper+ | TensorRT-LLM 生成式 decode kernel |
| `cute-dsl` | Blackwell (SM100+) | CuTe DSL GQA decode kernel |
| `cutlass` | SM90+ (MLA) | CUTLASS MLA kernel |
| `auto` | 自动 | 默认值，根据 GPU 架构自动选择 |

```python
# 手动指定 backend
wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer, "NHD", backend="fa3"
)
```

---

## 5. 与 FlashAttention 的比较

| 维度 | FlashAttention (FA2/FA3) | FlashInfer |
|------|-------------------------|------------|
| **定位** | 通用 attention kernel 库 | LLM 推理 serving 专用 |
| **KV Cache** | 连续 tensor | Paged KV (页表间接寻址) |
| **Batch 模式** | 统一 seq_len (padding) | 变长序列 (CSR indptr) |
| **Decode 优化** | 无专门优化 | Split-KV 专用 decode kernel |
| **CUDA Graph** | 不支持 | Plan/Run 分离支持 |
| **GQA/MQA** | 支持 | 专门优化 (use_tensor_cores) |
| **量化** | 不支持 | FP8, NVFP4 内置 |
| **位置编码** | 外部处理 | 内置 RoPE/ALiBi |
| **Soft Cap** | 不支持 | logits_soft_cap |
| **MLA** | 不支持 | 专用 MLA wrapper |
| **JIT** | 编译时配置 | 运行时 jit_args/jit_kwargs |

**何时用哪个**:
- **训练**: 用 FlashAttention (FA2/FA3)，因为是 compute-bound，不需要分页
- **推理 Prefill**: 两者都可以，FlashInfer 更适合变长 batch 和分页 KV
- **推理 Decode**: 必须用 FlashInfer，FA2/FA3 没有针对 decode 的优化
- **CUDA Graph serving**: 必须用 FlashInfer 的 Plan/Run 模式
- **MLA 模型**: 必须用 FlashInfer MLA wrapper

---

## 6. vLLM V1 集成架构

vLLM V1 通过 **AttentionBackend 抽象层** 集成 FlashInfer，核心组件:

```
┌─────────────────────────────────────────────────────────────┐
│                    GPUModelRunner                             │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │  SchedulerOutput │  │      Attention 抽象层             │  │
│  │  - num_scheduled │  │                                  │  │
│  │  - block_ids     │  │  AttentionBackend (接口)         │  │
│  │  - seq_lens      │  │    ├── AttentionMetadata         │  │
│  └────────┬─────────┘  │    ├── AttentionMetadataBuilder  │  │
│           │            │    └── AttentionCGSupport         │  │
│           ▼            │                                  │  │
│  ┌─────────────────┐  │  实现:                            │  │
│  │ _prepare_inputs │──│    ├── FlashInfer Backend         │  │
│  │ - 构建 indptr   │  │    ├── Triton Backend             │  │
│  │ - 构建 block    │  │    ├── cuDNN Backend              │  │
│  │   table         │  │    ├── FlashMLA Backend           │  │
│  │ - 计算 positions│  │    └── XQA Backend                │  │
│  └────────┬─────────┘  └──────────────┬───────────────────┘  │
│           │                           │                      │
│           ▼                           ▼                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              _build_attention_metadata                │    │
│  │                                                       │    │
│  │  1. 构建 CommonAttentionMetadata                     │    │
│  │     - query_start_loc, seq_lens, block_table          │    │
│  │     - positions, slot_mapping                        │    │
│  │                                                       │    │
│  │  2. 对每个 KV Cache Group:                            │    │
│  │     - 获取 AttentionGroup.backend                     │    │
│  │     - 调用 builder.build(common_attn_metadata)       │    │
│  │     - 生成 per-layer AttentionMetadata                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Model Forward                         │    │
│  │                                                       │    │
│  │  for layer in model.layers:                           │    │
│  │    hidden = layer(hidden, attn_metadata[layer_name]) │    │
│  │    # Attention 层内部:                                 │    │
│  │    #   wrapper.run(q, kv_cache)                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### vLLM Attention Backend 接口

```python
class AttentionBackend(ABC):
    """注意力后端抽象基类"""

    @staticmethod
    @abstractmethod
    def get_name() -> str: ...

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type[AttentionMetadata]: ...

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]: ...

class AttentionMetadataBuilder(ABC):
    """从 CommonAttentionMetadata 构建后端特定元数据"""

    @abstractmethod
    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        """构建 AttentionMetadata

        FlashInfer 后端内部:
        1. 创建 FlashInfer wrapper (如果不存在)
        2. 调用 wrapper.plan() 传入 indptr/indices 等
        3. 返回包含 wrapper 引用的 metadata
        """
        ...

class AttentionMetadata:
    """后端特定的 attention 元数据

    FlashInfer 实现:
    - decode_wrapper: BatchDecodeWithPagedKVCacheWrapper
    - prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper
    """
```

### 数据流: Scheduler 到 FlashInfer

```
Scheduler
  │
  │ SchedulerOutput:
  │   - num_scheduled_tokens: {req_id: num_tokens}
  │   - num_common_prefix_blocks: [int]
  │   - scheduled_new_reqs: [NewRequestData]
  │   - scheduled_spec_decode_tokens: {req_id: [token_ids]}
  │
  ▼
GPUModelRunner._update_states()
  - 更新 CachedRequestState
  - 更新 InputBatch (block_table, token_ids, seq_lens)
  - Condense batch
  │
  ▼
GPUModelRunner._prepare_inputs()
  - 构建 query_start_loc (CSR indptr for queries)
  - 构建 seq_lens, num_computed_tokens
  - 构建 positions (每个 token 的绝对位置)
  - 计算 slot_mapping (block_table × page_size + offset)
  - 复制到 GPU
  │
  ▼
GPUModelRunner._build_attention_metadata()
  - 构建 CommonAttentionMetadata (通用数据)
  - 对每个 KV Cache Group 的 AttentionGroup:
    - builder.build() → FlashInfer specific metadata
      内部调用:
        FlashInferAttentionMetadataBuilder.build()
          → 从 block_table 提取 indptr, indices
          → 调用 flashinfer_wrapper.plan(...)
  │
  ▼
Model Forward (在 set_forward_context 中)
  - 每层 Attention 从 forward_context 获取 metadata
  - 调用 impl.forward() → flashinfer_wrapper.run(q, kv_cache)
```

### KV Cache Spec 到 Attention Backend 的映射

vLLM V1 支持多种 AttentionSpec，每种可能映射到不同的 backend:

```python
AttentionSpec 类型:
├── FullAttentionSpec          → FlashInfer (标准全 attention)
├── SlidingWindowSpec          → FlashInfer (window_left 参数)
├── ChunkedLocalAttentionSpec  → 特殊的 local attention
├── CrossAttentionSpec         → encoder-decoder 交叉 attention
└── EncoderOnlyAttentionSpec   → 仅编码器 (pooling 模型)
```

### CUDA Graph 在 vLLM 中的集成

```
vLLM CUDA Graph 模式:
├── NONE: 不使用 CUDA Graph
├── PIECEWISE: 部分操作使用 CUDA Graph (逐层)
└── FULL: 整个 model forward 使用 CUDA Graph
    → FlashInfer wrapper 的 run() 被捕获到 Graph 中
    → plan() 在 Graph 外执行
    → 使用预分配的 buffer (indptr, indices, etc.)

GPUModelRunner.capture_model()
  → _dummy_run() with cudagraph_runtime_mode=FULL
    → _build_attention_metadata(for_cudagraph_capture=True)
      → builder.build_for_cudagraph_capture(common_attn_metadata)
        → 使用 max_model_len 作为 max_seq_len (确保 kernel 选择正确)
```

---

## 7. 性能关键点

### 7.1 Workspace Buffer

```python
# 推荐 128MB workspace buffer
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
```

首次使用必须初始化为 0 (`torch.zeros`)，后续可复用。Workspace 用于:
- Split-KV 中间结果存储
- 辅助数据结构缓存
- 不同 kernel 间的临时存储

### 7.2 层间复用

```python
# Plan 一次, Run 多次 (跨层复用)
wrapper.plan(...)  # 只调用一次!

for layer_idx in range(num_layers):  # 32 层
    q = query[layer_idx]
    kv = kv_cache[layer_idx]
    output = wrapper.run(q, kv)  # 每层调用 run
```

Plan 阶段构建的辅助数据结构对所有层都相同 (因为 indptr/indices 不变)，因此只需调用一次。

### 7.3 Backend 选择策略

```
auto 模式下的选择逻辑:
├── SM100+ (Blackwell)
│   ├── Decode: cute-dsl / trtllm-gen
│   └── Prefill: fa3 / cudnn
├── SM90 (Hopper)
│   ├── Decode: fa2 / trtllm-gen
│   ├── Prefill: fa3 / cudnn
│   └── MLA: cutlass / fa2
└── SM80+ (Ampere/Ada)
    ├── Decode: fa2
    └── Prefill: fa2 / cudnn
```

---

## 8. 论文核心贡献 (arxiv 2501.01005)

FlashInfer 论文发表于 MLSys 2025，主要贡献:

1. **Block-Sparse Format**: 统一的 KV cache 存储格式，支持存储异构性 (Paged/Ragged/Compressed)
2. **Composable Attention Formats**: 可组合的 attention 格式系统，通过 Load-Store abstraction 支持多种 attention 变体
3. **JIT Attention Template**: 运行时 JIT 编译自定义 attention 变体，无需重新编译库
4. **Load-Balanced Scheduling**: Split-KV 算法的负载均衡调度
5. **性能**: 在多种工作负载下实现 29-69% 的 inter-token latency 降低

---

## 9. 总结

```
FlashInfer 核心设计决策:

1. Plan/Run 分离 → CUDA Graph 兼容 → serving 延迟优化
2. Paged KV 页表 → 内存灵活管理 → vLLM PagedAttention 的 kernel 层实现
3. Split-KV → 长 context decode 负载均衡 → 适配任意 KV 长度
4. 多 Backend → 跨 GPU 架构最优 → fa2/fa3/cudnn/trtllm-gen/cute-dsl
5. MLA 支持 → DeepSeek 系列模型 → Matrix Absorption + 压缩 KV
6. FP8/NVFP4 → 量化推理 → 降低 KV cache 带宽需求
```

FlashInfer 在 LLM serving 栈中的定位: 介于框架调度层 (vLLM Scheduler/SGLang Scheduler) 和 GPU 硬件之间，提供 **attention 计算的最优 kernel 实现**，同时通过 Plan/Run 模式和页表间接寻址与上层框架的调度逻辑无缝集成。
