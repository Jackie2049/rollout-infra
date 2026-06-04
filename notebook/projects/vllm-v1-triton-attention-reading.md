# vLLM V1 Triton Attention Backend 源码分析

> 源码路径: `vllm/v1/attention/backends/triton_attn.py`, `ops/triton_unified_attention.py`, `ops/triton_decode_attention.py`
> MLA: `vllm/v1/attention/backends/mla/triton_mla.py`

## 1. 核心类

| 类 | 角色 |
|----|------|
| TritonAttentionBackend | 入口类, 定义后端特性 (dtype/量化/block_size支持) |
| TritonAttentionMetadataBuilder | 构建注意力元数据, 管理3D/2D内核选择 |
| TritonAttentionImpl | 核心实现: 编码器+解码器注意力, KV缓存更新, RoPE |
| TritonMLABackend/Impl | MLA专用后端 (DeepSeek-V2 压缩KV) |

## 2. Triton 内核

### 2.1 unified_attention (统一内核)
- **文件**: `ops/triton_unified_attention.py`
- **功能**: 统一处理 prefill 和 decode
- **关键参数**:
  - `BLOCK_SIZE`: KV 缓存块大小
  - `TILE_SIZE`: 瓦片大小 (2的幂)
  - `HEAD_SIZE/PADDED`: 头维度 (填充到2的幂)
  - `IS_3D`: 3D模式标志 (长序列分段)
  - `KV_QUANT_MODE`: KV量化模式
  - `NUM_SEGMENTS_PER_SEQ`: 每序列分段数

### 2.2 decode_attention_fwd (解码内核)
- **文件**: `ops/triton_decode_attention.py`
- **功能**: 专门用于 decode 的高效注意力
- **关键参数**:
  - `NUM_KV_SPLITS`: KV 分割数 (减少内存占用)
  - `PAGE_SIZE`: 页面大小 (≥1)
  - `BLOCK_DMODEL/BLOCK_DV`: 模型和值维度块大小
  - `kv_group_num`: KV 组数量 (GQA支持)

### 2.3 MLA 特定内核
- **文件**: `backends/mla/triton_mla.py`
- **功能**: MLA 多查询注意力
- **特点**: 多查询头映射到同一KV头, 压缩KV表示, 支持稀疏注意力

## 3. Decode vs Prefill 注意力差异

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 完整多头注意力 | 隐空间注意力 (latent) |
| 内存访问 | 整个 KV 缓存 | 仅最新 token 的 KV |
| 计算量 | ∝ seq² (compute-bound) | ∝ seq (memory-bound) |
| 内核选择 | context_attention_fwd / unified 3D | decode_attention_fwd / unified 2D |
| 序列长度 | 长 (prompt) | 单个或少量 token |

## 4. GQA/MLA 处理

### GQA
- `num_queries_per_kv` 参数控制
- `cur_kv_head = cur_head // kv_group_num` 计算对应KV头
- decode kernel 内置 KV 共享加载 (避免显式 expand)

### MLA (DeepSeek-V2)
- 专用 MLA 后端和内核
- 低秩压缩 (56.9x KV压缩)
- Q 和 KV 使用不同维度
- `kv_c_and_k_pe_cache` 存储压缩KV
- 解耦 RoPE (position encoding 与 content 分离)

## 5. 分页 KV 缓存间接寻址

```
block_table: [num_sequences, max_num_blocks]

寻址过程:
1. kv_page_number = token_id // PAGE_SIZE
2. physical_block = block_table[seq_idx, kv_page_number]
3. offset_in_block = token_id % PAGE_SIZE
4. kv_loc = physical_block * PAGE_SIZE + offset_in_block
```

**Triton 实现**: `tl.load` 间接内存访问, 掩码处理边界, 循环高效分页访问

## 6. 性能优化

### 6.1 Split-KV 优化
- 将 KV 序列分割为多段, 并行计算后合并
- `NUM_KV_SPLITS` 控制分割数
- 长序列下显著提升性能

### 6.2 在线 Softmax
- 每个瓦片维护部分最大值 (`e_max`) 和指数和 (`e_sum`)
- 避免全局 softmax 的内存瓶颈
- 经典 FlashAttention tiling 技术

### 6.3 2D vs 3D 内核选择
- **2D**: 短序列, 单次完整计算
- **3D**: 长序列, 分段计算后合并
- 基于 `seq_threshold_3D` 自动选择

### 6.4 量化支持
- FP8 per-tensor / per-token-head
- INT8 per-token-head
- 内置 dequantization/quantization

### 6.5 滑动窗口
- `compute_tile_loop_bounds` 限制访问范围
- 只在窗口内计算注意力

### 6.6 Tensor Descriptors (Intel XPU)
- `tl.make_tensor_descriptor` 创建 2D 块描述符
- 利用 Xe2/Xe3 硬件 2D 块读取
- 减少内存访问次数
