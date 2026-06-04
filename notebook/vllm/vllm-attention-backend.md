# vLLM Attention Backend 架构分析

> 文件路径: `vllm/attention/backends/`
> 分析日期: 2026-06-04

## 后端注册与选择

### 注册机制 (`registry.py`)
- `AttentionBackendEnum`: 枚举所有支持的后端
- `register_backend()`: 装饰器允许覆盖后端
- `_ATTN_OVERRIDES`: 存储自定义后端注册

### 选择机制 (`selector.py`)
- `get_attn_backend()`: 主入口 (L54-110)
- `AttentionSelectorConfig`: 指定需求 (L21-51)
- 平台特定优先级 (`cuda.py` L79-147):
  - **Ampere (SM 8.6)**: FlashInfer > FlashAttention > Triton
  - **Hopper (SM 9.0+)**: FlashAttention > FlashInfer > Triton
  - MLA 有独立优先级

## 抽象接口 (`backend.py`)

### 核心基类
| 类名 | 行号 | 职责 |
|------|------|------|
| `AttentionBackend` | L55-351 | 抽象基类, 定义接口 |
| `AttentionImplBase` | L692-768 | 实现基类 |
| `AttentionImpl` | L770-849 | 标准注意力实现 |
| `MLAAttentionImpl` | L850-938 | Multi-Head Latent Attention |
| `SparseMLAAttentionImpl` | L940-1018 | 稀疏 MLA 变体 |

### 核心方法
- `get_name()`, `get_impl_cls()`, `get_builder_cls()` — 抽象
- `supports_dtype()`, `supports_head_size()` — 配置验证
- Cache shape/layout 方法

## 主要后端实现

### Triton (`triton_attn.py`)
- `TritonAttentionBackend` (L271-395):
  - 支持 FP16/BF16/FP32
  - Block size 必须是 16 的倍数
  - 支持所有 attention 类型
  - KV cache 形状: `(num_blocks, 2, block_size, num_kv_heads, head_size)`
- 支持 cascade attention 和 parallel softmax

### FlashInfer (`flashinfer.py`)
- 使用 FlashInfer 库 kernel
- 支持 FP8 反量化
- Wrapper: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`

## Paged Attention 实现

### Kernel 层面
```python
# KV Cache 结构
shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
```

### 关键操作
- `block_table_tensor`: token 位置 → cache block 映射
- `slot_mapping`: token 在 cache 中的位置
- Cache reshape 和更新
- 支持可变序列长度

## AttentionMetadata (`backend.py` L359-501)

| 字段 | 说明 |
|------|------|
| `query_start_loc` | 每个请求在 query tensor 中的起始位置 |
| `seq_lens` | 每个请求已计算的 token 数 |
| `num_actual_tokens` | batch 中总 token 数 |
| `max_query_len` / `max_seq_len` | 最大维度 |
| `block_table_tensor` | token → cache block 映射 |
| `slot_mapping` | token 位置映射 |

### 后端特定 Metadata
- Triton: cascade attention, parallel softmax segments, multi-modal prefix
- FlashInfer: paged cache wrapper

## 架构流程

```
Backend Selection (平台 → 优先级 → 兼容性)
    ↓
Metadata Building (AttentionMetadataBuilder → per-layer metadata)
    ↓
Implementation Dispatch (AttentionImpl.forward() → kernel)
    ↓
Paged Attention (block_table + slot_mapping → KV cache 访问)
```

## 关键设计特点

1. **Fallback Chain**: 多后端优雅降级
2. **MLA 支持**: 独立实现 (DeepSeek-V2 风格)
3. **量化支持**: FP8/INT8 per-token/per-head scales
4. **CUDA Graph**: 不同支持级别 (ALWAYS/UNIFORM_BATCH/NEVER)
5. **内存布局**: NHD 和 HND 布局优化
