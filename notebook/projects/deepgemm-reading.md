# DeepGEMM — 高性能 FP8 GEMM Kernel 库

> deepseek-ai/DeepGEMM | 2025-2026 | MIT License
> 与我们 FP8 量化实验 + MoE 架构研究直接相关

## 1. 概述

DeepGEMM 是 DeepSeek 开源的统一 GPU tensor core kernel 库, 为 LLM 提供:
- **FP8/FP4/BF16 GEMM**: Dense 和 Grouped (MoE)
- **Mega MoE**: Fused EP dispatch + FP8xFP4 linear + SwiGLU + EP combine
- **MQA Scoring**: DeepSeek V3.2 lightning indexer 的加权 ReLU MQA logits
- **JIT 编译**: 运行时编译, 无需安装时 CUDA 编译

**核心设计哲学**: 简洁! 借鉴 CUTLASS/CuTe 概念但避免重度模板依赖, 核心仅少量 kernel 函数。

**性能**: H800 上达 **1550 TFLOPS**, 匹配或超越专家调优库。

## 2. Kernel 类型

### 2.1 Dense GEMM

```python
# D = C + A @ B.T (NT layout)
deep_gemm.fp8_gemm_nt(
    out,        # [M, N] BF16 输出
    a,          # [M, K] E4M3 FP8
    b,          # [N, K] E4M3 FP8 (转置存储)
    a_sf,       # [M, K/tile] FP32 scale factor (SM90) 或 packed UE8M0 (SM100)
    b_sf,       # [N, K/tile] 同上
)
```

### 2.2 Grouped GEMM (MoE Forward)

**Contiguous Layout** (训练 prefill):
```python
# 所有 expert 的 token 拼成连续张量
# M-axis 分组, N 和 K 固定 (所有 expert 共享 weight shape)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    out,          # [total_M, N]
    a,            # [total_M, K] 拼接的 FP8 激活
    b,            # [num_experts, N, K] 每个 expert 的权重
    a_sf, b_sf,
    m_indices,    # [total_M] 每个 token 属于哪个 expert
)
```

**Masked Layout** (推理 decode + CUDA Graph):
```python
# Decode 时 CPU 不知道每个 expert 收到多少 token
# 用 mask 指定哪些 (expert, token) 有效
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    out, masked_m, a, b, a_sf, b_sf, mask, expected_m,
)
```

### 2.3 Mega MoE (V2 核心创新!)

**单 mega-kernel 融合整个 MoE 层**:

```
传统 MoE (5 个独立 kernel):
  1. EP dispatch (All-to-All)          ← 通信
  2. Linear1: x @ W1 (FP8 GEMM)        ← 计算
  3. SwiGLU activation                 ← 计算
  4. Linear2: hidden @ W2 (FP8 GEMM)   ← 计算
  5. EP combine (All-to-All)           ← 通信

Mega MoE (1 个 mega-kernel):
  融合 EP dispatch + FP8xFP4 Linear1 + SwiGLU + FP8xFP4 Linear2 + EP combine
  → 通信与计算在 kernel 内重叠!
```

```python
# 分配对称内存 buffer (多进程共享)
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank,
    num_topk, hidden, intermediate_hidden
)

# 权重转换为 FP4 + UE8M0 scale factor 布局
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    l1_weights, l2_weights
)

# 拷贝输入到 buffer
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 运行 fused mega MoE kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

**关键**: FP8xFP4 混合精度 — activation 用 FP8, weight 用 FP4 (2x 通信减少)。

### 2.4 MQA Scoring (DeepSeek V3.2)

为 lightning indexer 提供加权 ReLU MQA logits 计算:

```python
# 对每个 query token i 和 KV token j:
# kv_j = kv[0][j] * kv[1][j].unsqueeze(1)  # 解量化
# out[i,j] = sum_h(ReLU(q[i,h] @ kv_j[h]) * weights[i,h])
deep_gemm.fp8_mqa_logits(out, q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end)
```

用途: DeepSeek V3.2 的快速检索器, 用 MQA score 做 token-to-token 相关性评估。

## 3. 架构支持

| 特性 | SM90 (H100) | SM100 (B200) |
|------|-------------|--------------|
| FP8 GEMM | NT layout | NT/TN/NN/TT |
| FP4 支持 | 通过 Mega MoE | 原生 |
| Scale Factor | FP32 | Packed UE8M0 |
| CUDA 版本 | ≥12.3 (推荐 12.9) | ≥12.9 |
| 峰值性能 | 1550 TFLOPS (H800) | 更高 |

## 4. JIT 编译系统

```python
# 核心特性:
# 1. 运行时编译 — 无需安装时 CUDA
# 2. 缓存在 $HOME/.deep_gemm
# 3. NVCC 12.9 自动做 FFMA interleaving
# 4. 可选 NVRTC (快 10x 编译, 但性能略低)

# 环境变量:
DG_JIT_CACHE_DIR=~/.deep_gemm        # 缓存目录
DG_JIT_USE_NVRTC=1                   # 使用 NVRTC (更快编译)
DG_JIT_DEBUG=1                       # 调试信息
DG_PRINT_CONFIGS=1                   # 打印每个 shape 的配置
DG_JIT_DUMP_ASM=1                    # dump PTX + SASS
```

## 5. 与我们项目的联系

| 我们的工作 | DeepGEMM 关联 |
|-----------|--------------|
| EP Simulator | DeepEP dispatch/combine 的 GEMM 后端 |
| FP8 量化实验 | FP8 tile-wise scale factor 实现 |
| MoE Layer 实现 | Grouped GEMM (contiguous/masked) |
| SwiGLU 激活 | Mega MoE 融合 SwiGLU |
| DeepSeek-V3 架构 | Mega MoE 是 V3 训练的核心 kernel |
| Top-nσ Triton | 同为 GPU kernel 优化, 可学习编码模式 |

## 6. 核心学习

1. **Kernel Fusion 的终极形态**: Mega MoE 将整个 MoE 层融合为一个 kernel, 通信+计算在 kernel 内重叠
2. **FP4 是新趋势**: weight 用 FP4 (2x 压缩), activation 用 FP8, 通信量减半
3. **JIT 编译是工程最佳实践**: 无需安装时编译, 灵活适配不同 GPU 架构
4. **Masked GEMM**: 解决 decode 时 CUDA Graph 无法预知 token 分配的问题
5. **对称内存**: Mega MoE 需要多进程共享内存 buffer (NVLink domain)
6. **简洁设计**: 核心仅少量 kernel, 比 CUTLASS 模板简单得多, 适合学习

## 参考

- GitHub: https://github.com/deepseek-ai/DeepGEMM
- H800 峰值: 1550 TFLOPS (FP8 dense GEMM)
- Mega MoE PR: #304, benchmark: #316
- MQA Scoring PR: #200
- SM100 Support PR: #112
