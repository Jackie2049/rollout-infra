# FlashMLA — 高效 Multi-head Latent Attention Kernel

> deepseek-ai/FlashMLA | 2025 | MIT License
> DeepSeek-V3 / V3.2 的核心 attention kernel 库

## 1. 概述

FlashMLA 是 DeepSeek 开源的优化 attention kernel 库, 为 DeepSeek-V3 和 V3.2-Exp 模型提供:

- **Dense MLA Decoding**: 3000 GB/s (memory-bound) / 660 TFLOPS (compute-bound) on H800
- **Sparse MLA Decoding**: 410 TFLOPS (FP8 KV cache + bfloat16 计算) on H800
- **Sparse MLA Prefill**: 640 TFLOPS (H800) / 1450 TFLOPS (B200)
- **Dense MHA Prefill**: 1460 TFLOPS forward / 1000 TFLOPS backward (B200)

**核心特点**: MLA (Multi-head Latent Attention) 是 DeepSeek-V3 的核心 attention 机制, 将 KV 压缩到低维 latent space, 大幅减少 KV cache 大小。

## 2. Kernel 类型

### 2.1 Dense MLA Decoding (SM90)

```
特点:
- MQA 模式: 128 query heads, 1 KV head
- head_dim_k = 576, head_dim_v = 512
- Decode 时 s_q=1 (或 MTP 时 s_q>1)
- H800: 3000 GB/s (memory-bound), 660 TFLOPS (compute-bound)
```

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# 一次性获取 tile scheduler 元数据
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv, h_q,
    is_fp8, topk,
)

# 每个 decode step 调用
for i in range(num_layers):
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices,
    )
```

### 2.2 Sparse MLA Decoding (SM90 & SM100)

DeepSeek-V3.2 引入的 **DeepSeek Sparse Attention (DSA)**:
- Token-level sparse attention for decode
- **FP8 KV Cache** — 大幅减少内存占用
- `indices` tensor 指定每个 query 只关注哪些 KV token

```
FP8 KV Cache 格式 (每个 token 656 bytes):
  前 512 bytes: 量化 NoPE 部分 (512 x float8_e4m3)
  中间 16 bytes: 4 x float32 scale factor (每 128 个 FP8 一组)
  最后 128 bytes: RoPE 部分 (64 x bfloat16, 不量化)
```

### 2.3 Sparse MLA Prefill (SM90 & SM100)

```python
# Token-level sparse attention for prefill
out, max_logits, lse = flash_mla_sparse_fwd(
    q,          # [s_q, h_q, d_qk] bfloat16
    kv,         # [s_kv, h_kv, d_qk] bfloat16
    indices,    # [s_q, h_kv, topk] int32 — 每个 query 关注哪些 KV token
    sm_scale,
)
```

等效 PyTorch:
```python
kv = kv.squeeze(1)               # [s_kv, d_qk]
focused_kv = kv[indices]          # [s_q, topk, d_qk]
P = (Q @ focused_kv.T) * sm_scale * log2(e)
max_logits = P.max(dim=-1)
lse = log2sumexp2(P, dim=-1, base=2)
S = exp2(P - lse)
out = S @ focused_kv
```

### 2.4 Dense MHA Prefill (SM100)

标准 MHA forward/backward, 接口兼容 flash_attn:
```python
from flash_mla import flash_attn_varlen_func
# 与 flash_attn 包用法相同
```

## 3. 支持矩阵

| Kernel | GPU | MLA Mode | KVCache |
|--------|-----|----------|---------|
| Dense Decoding | SM90 | MQA (576/512) | BF16 |
| Sparse Decoding | SM90 & SM100 | MQA | FP8 |
| Dense Prefill | SM100 | MHA (192|128/128) | — |
| Sparse Prefill | SM90 & SM100 | MQA | — |

## 4. Seesaw Scheduling — 核心创新

### 4.1 为什么 MLA Decode 是 Compute-Bound?

传统 decode attention 是 memory-bound (KV cache 读取占主导)。但 MLA 不同:

```
Compute-Memory Ratio = h_q * s_q * (d_k + d_v) / d_k ≈ 2 * h_q * s_q

DeepSeek-V3: h_q = 128, s_q = 1
→ Ratio = 256, 远超 H800 的 865 TFLOPS / 3.35 TB/s ≈ 258

结论: MLA decode 是 compute-bound! 需要优化 Tensor Core 利用率
```

### 4.2 Seesaw Schedule

问题: WGMMA 要求输出矩阵在寄存器, 64×512 矩阵占 32,768 registers → 每个 SM 只能放一个输出矩阵 (65,536 total) → FlashAttention-3 的 ping-pong 不适用。

解决方案: **Seesaw Scheduling**

```
将输出矩阵垂直拆分为 O_L 和 O_R (各 64×256)
取两个 KV block (K0,V0) 和 (K1,V1)

Warpgroup 0:                    Warpgroup 1:
  p0 = q @ K0.T                  p1 = q @ K1.T           [Tensor Core]
  softmax(p0)                    softmax(p1)             [CUDA Core]
  O_L += p0 @ V0L                O_R += p1 @ V1R         [Tensor Core]
  O_L += p1 @ V1L                O_R += p0 @ V0R         [Tensor Core]

两个 warpgroup 交错执行, CUDA Core 和 Tensor Core 重叠!
```

关键: 使用共享的 running max `m`, 两个 warpgroup 通过 softmax scale 因子互相协调。

### 4.3 其他优化

- **Fine-grained TMA-GEMM pipelining**: 64×576 K block 拆为 9 个 64×64 TMA copy, GEMM 随到随开始
- **Cache hints**: `EVICT_FIRST` 提升 L2 命中率
- **Programmatic Dependent Launch**: splitkv 和 combine kernel 重叠
- **Tile Scheduler**: 请求和 block 分配到 SM, 负载均衡

## 5. FP8 Sparse Decoding — Crossover 技术

### 5.1 问题: Dequantization Bottleneck

```
FP8 KV cache 需要 dequantize 为 bfloat16:
  FP8 → half → float32 → bfloat16 × scale
  ≈ 50 cycles/token (per CTA)

而 MMA 操作仅需 ≈34 cycles/token

→ Kernel 变成 dequantization-bound! Tensor Core 空闲等待
```

### 5.2 Crossover — 跨 CTA 共享 Dequantize

```
MQA 特性: 同一 query token 的所有 128 head 看同一个 KV head

方案: 2 个 CTA 组成 cluster, 各负责 64 query heads
  CTA 0: dequantize 前半 KV → 写入自己的 shared memory
  CTA 1: dequantize 后半 KV → 写入自己的 shared memory
  同时: 用 st.async 把 dequantize 结果写入对方的 shared memory!

结果: 每个 CTA 只需 dequantize 一半 → 25 cycles < 34 cycles
      Dequantization 不再是瓶颈!

同步: cluster transaction barrier (Hopper 新特性)
```

**灵感来源**: 染色体交叉 (Chromosomal Crossover during Meiosis)!

### 5.3 FP8 Sparse 性能

| 配置 | TFLOPS | 备注 |
|------|--------|------|
| batch=128, topk=2048 | 410 | 默认配置 |
| batch=128, topk=32768 | 460 | 更大 topk |
| 对比 BF16 dense decode | 660 | 理论上界 |

## 6. 与我们项目的联系

| 我们的工作 | FlashMLA 关联 |
|-----------|--------------|
| DeepSeek-V3 Notes | MLA 是 V3 的核心 attention 机制 |
| FlashAttention 笔记 | FlashMLA 基于 FlashAttention 2&3 设计 |
| FP8 量化实验 | FP8 KV cache 量化 (tile-wise, 128 tiles) |
| CUDA Basics | WGMMA, TMA, shared memory, warpgroup |
| Serving 推理优化 | Decode kernel 直接决定推理吞吐 |

## 7. 核心学习

1. **MLA 让 decode 变成 compute-bound**: h_q=128 时, compute-memory ratio ≈ 256, 与标准 MHA decode (memory-bound) 完全不同
2. **Seesaw Scheduling**: 在只有一块输出矩阵寄存器空间的限制下, 创造性地用两个 warpgroup 交错实现 compute-CUDA Core 重叠
3. **Crossover 共享 dequantize**: 利用 MQA 共享 KV 的特性 + Hopper Distributed Shared Memory, 跨 CTA 共享 dequantize 结果
4. **FP8 KV Cache**: 128K context 下, 单请求 KV cache 达 8.72 GiB, FP8 量化是必须的
5. **Sparse Attention 是新范式**: V3.2 的 DSA 用 indices 只关注相关 token, topk=2048 时 410 TFLOPS
6. **Hopper 架构特性**: CTA Cluster, Distributed Shared Memory, cluster transaction barrier 是实现 crossover 的基础

## 参考

- GitHub: https://github.com/deepseek-ai/FlashMLA
- Deep-dive blog: `docs/20250422-new-kernel-deep-dive.md` (Seesaw Scheduling)
- FP8 sparse deep-dive: `docs/20250929-hopper-fp8-sparse-deep-dive.md` (Crossover)
- FlashAttention-3: arXiv:2407.08608
- Hopper Architecture: NVIDIA developer blog
- DeepSeek-V3.2 论文附录: MLA MQA/MHA mode 解释
