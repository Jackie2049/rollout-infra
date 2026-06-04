# vLLM Triton Kernel 源码阅读

> 深入理解 vLLM 推理引擎中的关键 Triton GPU Kernel

## 1. Triton 编程模型回顾

```
Triton 三层抽象:
  1. Program (GPU Grid): 多个 program 并行执行
     - tl.program_id(axis) → 获取当前 program 的 ID
     - Grid = (x, y, z) 定义 program 数量

  2. Block (Shared Memory): 每个 program 处理一块数据
     - tl.arange(0, BLOCK_SIZE) → 计算偏移量
     - tl.load/store → 带 mask 的内存访问

  3. Element (Thread): 自动向量化
     - Triton 自动映射到 CUDA 线程
     - tl.dot() → 自动使用 Tensor Core

关键优势 (vs 手写 CUDA):
  - 自动共享内存管理
  - 自动向量化
  - Python 语法
  - JIT 编译优化
```

## 2. Decode Attention Kernel (两阶段架构)

**文件**: `vllm/v1/attention/ops/triton_decode_attention.py` (~791 行)

### 2.1 为什么 Decode Attention 需要特殊 Kernel？

```
Prefill (FlashAttention):
  - Q: [batch, num_heads, seq_len, head_dim]  ← 长序列
  - K,V: [batch, num_kv_heads, seq_len, head_dim]  ← 长序列
  - Q×K^T: [batch, num_heads, seq_len, seq_len]  ← 可以分块 tiling

Decode (单 token):
  - Q: [batch, num_heads, 1, head_dim]  ← 只有 1 个 token!
  - K,V: [batch, num_kv_heads, seq_len, head_dim]  ← 全部历史
  - Q×K^T: [batch, num_heads, 1, seq_len]  ← 瘦长矩阵
  - FlashAttention 的分块策略不适用 (Q 太短)
  - 需要: 遍历所有 K,V, 逐块计算 attention score
```

### 2.2 两阶段设计

```
Stage 1: 分块计算 (Split-KV)
  Grid: (batch, num_heads, NUM_KV_SPLITS)
  每个 program 处理 KV 序列的一个分片

  对每个 KV 分片:
    1. 加载 Q [1, head_dim]
    2. 遍历 KV blocks:
       a. 通过 page_table 查找物理 block 位置
       b. 加载 K [BLOCK_N, head_dim], V [BLOCK_N, head_dim]
       c. 计算 QK^T = sum(Q * K, dim=-1)  ← 注意力分数
       d. 在线 Softmax: 维护 e_max, e_sum, acc
       e. acc += softmax(QK^T) × V
    3. 输出: Mid_O[batch, head, split, head_dim] + log_sum_exp

Stage 2: 合并 (Reduce)
  Grid: (batch, num_heads)
  合并所有 KV split 的结果

  对所有 NUM_KV_SPLITS 个分片:
    1. 加载 Mid_O 和 LSE
    2. 再次在线 Softmax 合并
    3. 最终输出: O[batch, head, head_dim] + LSE[batch, head]
```

### 2.3 在线 Softmax (Online Softmax)

核心算法——避免 materialize 完整的注意力矩阵:

```python
# Stage 1 中每个 KV block 的处理:
e_max = -inf   # 当前最大注意力分数
e_sum = 0.0    # 指数和
acc = zeros    # 累积输出

for each KV block:
    qk = Q × K^T             # 注意力分数
    n_e_max = max(qk, e_max)  # 新的最大值
    re_scale = exp(e_max - n_e_max)  # 旧结果的缩放因子
    p = exp(qk - n_e_max)    # 新的 softmax 概率

    acc = acc * re_scale      # 重新缩放旧累积
    acc += p × V              # 加上新贡献
    e_sum = e_sum * re_scale + sum(p)  # 更新指数和
    e_max = n_e_max           # 更新最大值

# 最终: output = acc / e_sum
```

**关键**: 每次 e_max 更新时，旧的 acc 和 e_sum 都需要重新缩放。这避免了存储完整的 softmax 矩阵。

### 2.4 Paged KV Cache 访问

```python
# 通过 page_table 将逻辑 token 位置映射到物理 block
kv_page_number = load(Req_to_tokens[batch][token_pos // PAGE_SIZE])
kv_loc = kv_page_number * PAGE_SIZE + token_pos % PAGE_SIZE

# 然后用 kv_loc 索引 K_Buffer/V_Buffer
k = load(K_Buffer[kv_loc, kv_head, :])
v = load(V_Buffer[kv_loc, kv_head, :])
```

### 2.5 GQA (Grouped Query Attention) 支持

```
kv_group_num = num_q_heads // num_kv_heads

# 例: Llama-70B, 64 Q heads, 8 KV heads
# kv_group_num = 64 / 8 = 8

cur_kv_head = cur_head // kv_group_num  # Q head 映射到 KV head
```

### 2.6 Grouped Kernel 变体

```python
# _fwd_grouped_kernel_stage1: 一次处理多个 Q head
BLOCK_H = 16  # 一次处理 16 个 head

# Grid: (batch, ceil(head_num/BLOCK_H), NUM_KV_SPLITS)
# 每个 program 处理 BLOCK_H 个 head, 共享 KV 加载

优势:
  - KV 只加载一次, 多个 Q head 共享
  - GQA 场景特别有效 (多个 Q head 对应同一 KV head)
  - 使用 tl.dot() 替代 tl.sum() (利用 Tensor Core)
```

### 2.7 MLA 支持

```python
if IS_MLA:
    # MLA 使用单个 c_kv, 不需要单独的 V
    v = tl.trans(k)  # 直接转置 K 作为 V

# RoPE 分离:
# BLOCK_DMODEL: 无位置编码部分
# BLOCK_DPE: 位置编码部分
# QK = dot(q, k) + dot(qpe, kpe)  # 分两部分计算
```

### 2.8 FP8 KV Cache

```python
# 在 kernel 中即时反量化
if k.dtype.is_fp8():
    k = (k.to(float32) * k_scale).to(q.dtype)
if v.dtype.is_fp8():
    v = (v.to(float32) * v_scale).to(q.dtype)
```

## 3. Reshape and Cache Kernel

**文件**: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

### 3.1 功能

将新计算的 K/V 写入 Paged KV Cache:

```
输入: K/V [num_tokens, num_heads, head_size]
输出: KV Cache [num_blocks, block_size, num_heads, head_size]

关键映射:
  slot_mapping[token_idx] → 物理槽位
  block_idx = slot // block_size
  block_offset = slot % block_size
```

### 3.2 两种布局

```
Head-Major Layout (USE_HEAD_MAJOR_LAYOUT=True):
  Key Cache:   [Block, Head, Dim//x, Slot, x]  ← x=8 用于向量化
  Value Cache: [Block, Head, Dim, Slot]

  优势: 同一 head 的数据连续, 适合 Decode attention 的逐 head 加载
  vLLM V1 默认使用此布局

Token-Major Layout (USE_HEAD_MAJOR_LAYOUT=False):
  Key Cache:   [Block, Slot, Head, Dim]
  Value Cache: [Block, Slot, Head, Dim]

  传统布局, 简单但对 decode 不够友好
```

### 3.3 FP8 量化写入

```python
if FP8_KV_CACHE:
    key_tile = key_load / k_scale  # 缩放到 FP8 范围
    # tl.store 自动 cast 到 FP8

# 另一种变体: Per-token-head 动态量化
# Grid: (num_tokens, num_kv_heads)
# 每个 program:
#   1. 加载一个 head 的 K/V
#   2. 计算 absmax → scale = absmax / QUANT_MAX
#   3. 量化: quantized = round(value / scale)
#   4. 存储: quantized + scale
```

## 4. Kernel 调优参数

### 4.1 BLOCK_N (KV 遍历块大小)

```
Decode attention 中遍历 KV 序列的块大小:
  NVIDIA: BLOCK_N = 64 (stage1) / 32 (grouped)
  AMD ROCm: BLOCK_N = 8 (stage1) / 16 (grouped)

更大的 BLOCK_N:
  + 每步处理更多 KV tokens → 减少循环次数
  + 更好的内存合并
  - 需要更多共享内存
```

### 4.2 NUM_KV_SPLITS

```
KV 序列被分成多少个分片并行处理:
  典型值: 根据 seq_len 动态计算
  seq_len < 256: 1 split
  seq_len < 1024: 2-4 splits
  seq_len >= 4096: 8-16 splits

更多 splits:
  + 更好的 GPU 并行度
  + Stage 2 有额外合并开销
  - 需要更多共享内存存中间结果
```

### 4.3 num_warps 和 num_stages

```
num_warps:
  - 每个 program 使用的 warp 数
  - GQA=1: 4 warps (更多并行)
  - GQA>1: 2 warps (stage1), 1 warp (ROCm)
  - Grouped: 4 warps

num_stages:
  - 流水线阶段数 (软件流水线)
  - 默认: 2 (重叠 load 和 compute)
  - BLOCK_DMODEL >= 1024: 1 (共享内存不足)
  - ROCm: 1
```

## 5. 性能关键洞察

### 5.1 Decode Attention 是 Memory-Bound

```
每次 decode step:
  - 读 Q: 1 token × head_dim = 2KB (7B, 32 heads, 128 dim, FP16)
  - 读 K,V: seq_len × 2 × 32 × 128 × 2 = 16KB × seq_len
  - 写 O: 2KB
  - 计算: 2 × seq_len × head_dim FMA per head

AI (Arithmetic Intensity) ≈ seq_len × head_dim / (seq_len × head_dim × 4 bytes)
                        ≈ 0.5 ops/byte → 极度 memory-bound!

优化重点: 减少内存访问, 而非减少计算
```

### 5.2 分块策略的意义

```
Stage 1 的 Split-KV 设计:
  - 每个 program 只处理部分 KV → 减少共享内存需求
  - 多个 program 并行 → 填满 GPU SM
  - Stage 2 合并代价很小 (只合并 O 和 LSE)

对比不分块:
  - 单个 program 处理全部 KV → 共享内存可能不够
  - GPU SM 利用率低 (程序数太少)
```

### 5.3 GQA 的优化机会

```
kv_group_num = 8 (Llama-70B):
  - 8 个 Q head 共享同一组 KV
  - Grouped kernel: 一次加载 KV, 8 个 Q head 共享
  - 减少 87.5% 的 KV 加载量!
  - 这是 GQA 在推理时的核心优势
```

## 6. 关键洞察

1. **两阶段设计**: Split-KV 并行 + Reduce 合并, 解决 decode Q=1 的并行度不足
2. **在线 Softmax**: 避免 materialize 完整注意力矩阵, O(1) 额外内存
3. **Paged KV**: 通过 page_table 间接寻址, 支持非连续 KV Cache
4. **Grouped 变体**: GQA 场景下 KV 共享加载, 减少 87.5% 内存读取
5. **MLA 支持**: 单 c_kv + RoPE 分离, 通过 IS_MLA/BLOCK_DPE 参数化
6. **FP8 即时反量化**: 在 kernel 内完成 FP8→FP16, 避免额外 kernel launch
7. **Head-Major 布局**: [Block, Head, Dim, Slot] 对 decode 逐 head 加载更友好
8. **软件流水线**: num_stages=2 重叠 load 和 compute, 隐藏内存延迟

## 参考资料

- Triton Language: https://triton-lang.org/
- Triton FlashAttention: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Online Softmax: "Online normalizer calculation for softmax" (M. Milakov et al., 2018)
- vLLM 源码: `vllm/v1/attention/ops/triton_decode_attention.py`
- vLLM 源码: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- SGLang Decode Attention (原始来源): `sglang/srt/layers/attention/triton_ops/`
