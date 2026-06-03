# vLLM MLA Attention Backend 源码阅读

> 基于 `rollout-infra/vllm-latest/vllm/v1/attention/backends/mla/` 源码分析
>
> 分析 DeepSeek-V2/V3 的 Multi-head Latent Attention 在 vLLM 中的多种 backend 实现
>
> 日期: 2026-06-04

---

## 1. 为什么 MLA 需要专门的 Attention Backend

标准 Attention (MHA/GQA/MQA) 和 MLA 的核心区别：

```
标准 Attention:
  K, V 直接存储在 KV Cache 中: [seq_len, num_kv_heads, head_dim]
  Attention: Q @ K^T / sqrt(d)

MLA Attention:
  KV Cache 只存储压缩的 latent: [seq_len, kv_lora_rank] + [seq_len, qk_rope_head_dim]
  需要从 latent 恢复/投影 K, V → 不同的 attention 计算路径
  关键: 不需要显式恢复完整 K/V（矩阵吸收）
```

因此 MLA 需要专门的 kernel 实现来：
1. 处理压缩的 KV cache 格式
2. 优化两条计算路径（prefill 和 decode 不同）
3. 支持解耦 RoPE（separate content 和 position 分支）

---

## 2. MLA Backend 变体一览

### 2.1 Decode Backend（解码阶段）

| Backend | 目标硬件 | Kernel 类型 | 特点 |
|---------|---------|------------|------|
| **FlashMLA** | Hopper (SM90) | Custom C++ | 主力 backend，支持 FP16/BF16/FP8 |
| **FlashInfer MLA** | Blackwell (SM100) | Triton (FlashInfer) | HND KV layout，需特定 head dim |
| **Triton MLA** | All CUDA | Triton | 通用 fallback，动态 KV splits |
| **Cutlass MLA** | Blackwell (SM100) | CUTLASS | 固定 block size=128 |
| **Aiter Triton MLA** | ROCm (AMD) | aiter ops | AMD GPU 专用 |

### 2.2 Sparse 变体

- FlashMLA Sparse, FlashInfer MLA Sparse, XPU MLA Sparse, ROCm Aiter MLA Sparse
- 用于特定稀疏注意力模式（如 sliding window attention）

### 2.3 Prefill Backend（预填充阶段）

| Backend | 目标硬件 | 特点 |
|---------|---------|------|
| FlashAttention Prefill | All CUDA | 用 padding 处理不同 head dim |
| FlashInfer Prefill | SM100 only | 需要 DeepSeek R1 维度 (128,64,128) |
| TritonLLM Ragged Prefill | All CUDA | 支持变长序列 |

---

## 3. MLA 核心数据流

### 3.1 压缩 → 解压 → Attention

```
输入: hidden_states h_t [Sq, H]

压缩 (Down-Projection):
  q_c  = h_t @ W_DQ    → [Sq, Lq]    (query latent, Lq = q_lora_rank)
  kv_c = h_t @ W_DKV   → [Sq, Lkv]   (KV latent, Lkv = kv_lora_rank)

解压 (Up-Projection):
  q_nope = q_c @ W_UQ   → [Sq, N, P]  (content query, P = qk_nope_head_dim)
  q_pe   = RoPE(q_c @ W_QR) → [Sq, N, R] (RoPE query, R = qk_rope_head_dim)
  k_nope = kv_c @ W_UK  → [Skv, N, P]  (content key)
  v      = kv_c @ W_UV  → [Skv, N, V]  (value, V = v_head_dim)
  k_pe   = RoPE(h_t @ W_KR) → [Skv, R]   (shared RoPE key, 所有 head 共享)
```

### 3.2 两条计算路径

```
路径 A: Compute-Friendly (Prefill 阶段)
  完全解压后做标准 MHA:
    Q = [q_nope ; q_pe]  → [Sq, N, P+R]
    K = [k_nope ; k_pe]  → [Skv, N, P+R]
    V = v                 → [Skv, N, V]
    output = Softmax(Q @ K^T / sqrt(P+R)) @ V
  特点: 计算量大但并行度高，适合 prefill 的 compute-bound 特性

路径 B: Data-Movement Friendly (Decode 阶段)
  利用矩阵吸收，在 latent 空间做 attention:
    Q' = q_c (compressed) → [Sq, Lkv]
    K' = kv_c (compressed) → [Skv, Lkv]
    output_compressed = Softmax(Q' @ K'^T / sqrt(Lkv)) @ kv_c
    output = output_compressed @ W_UV @ W_O
  特点: 减少数据搬运，适合 decode 的 memory-bound 特性
```

**关键洞察**: Prefill 和 Decode 用不同路径，因为瓶颈不同！
- Prefill: 瓶颈是计算 → 解压做完整 MHA
- Decode: 瓶颈是带宽 → 在 latent 空间做 attention

---

## 4. KV Cache 格式

### 4.1 标准 Attention vs MLA

```
标准 Attention KV Cache (per token, per layer):
  K: [num_kv_heads, head_dim]     = 64 × 128 = 8192 elements (GQA-8)
  V: [num_kv_heads, head_dim]     = 64 × 128 = 8192 elements
  总计: 16,384 elements × 2 bytes = 32,768 bytes

MLA KV Cache (per token, per layer):
  kv_c: [kv_lora_rank]            = 512 elements (latent)
  k_pe: [qk_rope_head_dim]        = 64 elements  (RoPE key)
  总计: 576 elements × 2 bytes    = 1,152 bytes

压缩比: 32,768 / 1,152 ≈ 28.4x
```

### 4.2 vLLM 中的 KV Cache 格式

```
FlashMLA backend:
  使用 "fp8_ds_mla" 格式 (DeepSeek 压缩格式)
  支持 FP8 KV Cache 进一步压缩

FlashInfer MLA backend:
  使用 HND (Head-Number-Dim) layout
  kv_c 和 k_pe 分开存储

通用格式:
  [kv_lora_rank + qk_rope_head_dim] per token
  block_size = 64 (和标准 attention 相同)
```

---

## 5. 解耦 RoPE 实现

### 5.1 代码中的实现

```python
# 伪代码，基于 flashmla.py 分析
def forward_prefill(hidden_states, ...):
    # 1. 压缩
    q_c = hidden_states @ W_DQ     # [Sq, Lq]
    kv_c = hidden_states @ W_DKV   # [Sq, Lkv]

    # 2. 解压 content 部分
    q_nope = q_c @ W_UQ            # [Sq, N, P]
    k_nope = kv_c @ W_UK           # [Skv, N, P]
    v = kv_c @ W_UV                # [Skv, N, V]

    # 3. RoPE 分支 (解耦位置编码)
    q_pe = apply_rope(q_c @ W_QR)  # [Sq, N, R]
    k_pe = apply_rope(hidden_states @ W_KR)  # [Skv, R] (共享!)

    # 4. 拼接 content + RoPE
    Q = concat(q_nope, q_pe, dim=-1)  # [Sq, N, P+R]
    K = concat(k_nope, k_pe, dim=-1)  # [Skv, N, P+R]

    # 5. 标准 attention
    output = flash_attn(Q, K, V)    # [Sq, N, V]
```

### 5.2 为什么 RoPE 不能直接在 latent 空间施加

```
如果对 kv_c 直接施加 RoPE:
  RoPE(kv_c) → 旋转矩阵作用在 latent 上
  → W_UK 和 RoPE 矩阵耦合
  → 无法做矩阵吸收 (W_UK 不能被吸收到 W_Q)
  → Decode 时必须从 latent 恢复完整 K (失去压缩优势!)

解决方案: 解耦 RoPE
  content 部分 (q_nope, k_nope): 不带位置信息 → 可以做矩阵吸收
  position 部分 (q_pe, k_pe): 独立处理位置 → 所有 head 共享一个 k_pe
```

---

## 6. Backend 选择逻辑

### 6.1 Decode Backend 优先级

```
SM100 (Blackwell) 优先级:
  1. FlashInfer MLA (head ≤ 16 时最优)
  2. Tokenspeed MLA
  3. Cutlass MLA (硬件优化)
  4. FlashAttention MLA
  5. FlashMLA
  6. Triton MLA (fallback)

SM90 (Hopper) 优先级:
  1. FlashAttention MLA
  2. FlashMLA (主力)
  3. FlashInfer MLA
  4. Triton MLA
  5. FlashMLA Sparse

其他 CUDA:
  1. Triton MLA (通用 fallback)
```

### 6.2 Prefill Backend 选择

```
Blackwell:
  FlashAttention → TritonLLM Ragged → FlashInfer → Tokenspeed

Hopper:
  FlashAttention only (稳定可靠)

验证条件:
  FlashInfer Prefill 需要检查 DeepSeek R1 兼容维度:
    qk_nope_head_dim ∈ {64, 128, 192}
    qk_rope_head_dim ∈ {64}
    v_head_dim ∈ {128}
```

---

## 7. 关键参数

### 7.1 DeepSeek-V2/V3 MLA 配置

```python
# DeepSeek-V2 配置
q_lora_rank = 1536         # query 压缩维度
kv_lora_rank = 512         # KV 压缩维度 (核心!)
qk_nope_head_dim = 128     # content head dim (P)
qk_rope_head_dim = 64      # RoPE head dim (R)
v_head_dim = 128           # value head dim (V)
num_heads = 128            # attention heads

# KV Cache 压缩效果
# per token per layer: (512 + 64) × 2 = 1152 bytes
# vs 标准 MHA: 2 × 128 × 128 × 2 = 65536 bytes
# 压缩比: 65536 / 1152 ≈ 56.9x
```

### 7.2 环境变量

```bash
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE  # FlashInfer 工作空间大小
FORCE_NUM_KV_SPLITS                    # Cutlass MLA 分割数覆盖
VLLM_BATCH_INVARIANT                   # 启用 batch-invariant 模式
```

---

## 8. 源码文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `mla/__init__.py` | ~50 | Backend 注册和导出 |
| `mla/flashmla.py` | ~500 | FlashMLA decode (SM90 主力) |
| `mla/flashinfer_mla.py` | ~400 | FlashInfer MLA decode (SM100) |
| `mla/triton_mla.py` | ~300 | Triton MLA decode (通用) |
| `mla/cutlass_mla.py` | ~350 | CUTLASS MLA decode (SM100) |
| `mla/compressor_utils.py` | ~200 | 低秩压缩工具函数 |
| `mla/indexer.py` | ~100 | MLA KV Cache 索引 |
| `mla/prefill/` | 目录 | Prefill 专用实现 |
| `mla/sparse_utils.py` | ~150 | 稀疏注意力工具 |

---

## 9. 与笔记 `fundamentals/mla.md` 的关联

本源码阅读验证了 `notebook/fundamentals/mla.md` 中的理论分析：

| 理论 | vLLM 实现 |
|------|-----------|
| 低秩压缩 c_t^KV = W^DKV × h_t | `kv_c = hidden_states @ W_DKV` |
| 解耦 RoPE | `q_pe = RoPE(q_c @ W_QR)`, `k_pe = RoPE(h_t @ W_KR)` |
| 矩阵吸收 (decode) | FlashMLA/Triton MLA 在 latent 空间做 attention |
| KV Cache = (d_c + d_h^R) | `kv_lora_rank + qk_rope_head_dim` per token |
| Prefill 解压完整 K/V | FlashAttention Prefill 做完整 MHA |

---

## 参考

- `notebook/fundamentals/mla.md` — MLA 理论笔记
- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [FlashInfer Project](https://github.com/flashinfer-ai/flashinfer)
- vLLM 源码: `vllm/v1/attention/backends/mla/`
