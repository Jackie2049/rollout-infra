# Multi-head Latent Attention (MLA) — DeepSeek 的 KV Cache 压缩方案

> 目标：理解 DeepSeek-V2 的 MLA 机制，如何通过低秩压缩实现 93.3% KV Cache 减少

## 1. 问题：KV Cache 是推理瓶颈

```
标准 MHA 的 KV Cache:
  每个 token: 2 × n_h × d_h × l elements (keys + values, 所有层)

LLaMA-7B: 2 × 32 × 128 × 32 = 262,144 elements/token
  → BF16: 512 KB/token → 512 tokens = 256 MB

LLaMA-70B: 2 × 64 × 128 × 80 = 1,310,720 elements/token
  → BF16: 2.5 MB/token → 2048 tokens = 5 GB!

KV Cache 限制了 batch size 和最大序列长度
```

### 已有方案的局限

| 方案 | KV Cache 大小 | 性能 |
|------|-------------|------|
| MHA | 2 × n_h × d_h × l | 最强 |
| GQA | 2 × n_g × d_h × l (n_g < n_h) | 有损 |
| MQA | 2 × d_h × l (n_g=1) | 损失最大 |

GQA/MQA 通过减少 KV head 数量来压缩，但牺牲了性能。

## 2. MLA 核心思想：低秩 KV 联合压缩

### 2.1 压缩机制

```
标准 MHA:
  k_t = W^K × h_t    → [d_h × n_h] 维 (每个 token)
  v_t = W^V × h_t    → [d_h × n_h] 维
  Cache: 2 × d_h × n_h elements

MLA:
  c_t^KV = W^DKV × h_t    → [d_c] 维 (d_c << d_h × n_h)

  推理时只需要 cache c_t^KV!

  k_t^C = W^UK × c_t^KV   (从 latent 恢复 keys)
  v_t^C = W^UV × c_t^KV   (从 latent 恢复 values)

  Cache: d_c elements (比 MHA 少得多!)
```

### 2.2 为什么能压缩？

```
KV 的维度 d_h × n_h 很高 (如 128 × 128 = 16384)
但实际有效信息可能只在一个低秩子空间中 (如 d_c = 512)

类比:
  原始: 128 heads × 128 dim = 16384 维
  压缩: 512 维 latent (32x 压缩)

  信息损失很小，因为 K 和 V 的 effective rank 通常远低于名义维度
```

### 2.3 推理时的矩阵吸收

```
训练时:
  q_t = W^Q × h_t
  k_t = W^UK × W^DKV × h_t
  attn = q_t^T × k_t = (W^Q × h_t)^T × (W^UK × c_t^KV)
       = h_t^T × W^Q^T × W^UK × c_t^KV

推理时 (矩阵乘法结合律):
  可以将 W^UK 吸收到 W^Q 中: W^Q' = W^Q × W^UK
  → 不需要显式计算 k_t^C

  同理: W^UV 可以吸收到 W^O 中
  → 不需要显式计算 v_t^C

结果: 推理时完全不需要从 c_t^KV 恢复 K 和 V
```

## 3. 解耦 RoPE (Decoupled Rotary Position Embedding)

### 3.1 问题：RoPE 与低秩压缩不兼容

```
RoPE 对 Q 和 K 都施加位置相关的旋转矩阵:
  q_t^rot = RoPE(q_t, t)
  k_t^rot = RoPE(k_t, t)

如果对 k_t^C = W^UK × c_t^KV 施加 RoPE:
  RoPE(W^UK × c_t^KV)

  此时 W^UK 和 RoPE 矩阵耦合 → 无法吸收到 W^Q
  → 推理时必须重新计算所有 prefix tokens 的 keys!
```

### 3.2 解耦方案

```
额外的 "decoupled" 分支:
  q_t^R = RoPE(W^QR × c_t^Q)  → 每个头独立的 decoupled query [d_h^R dim]
  k_t^R = RoPE(W^KR × h_t)     → 所有头共享的 decoupled key [d_h^R dim]

最终 attention 计算:
  query_i = [q_t,i^C ; q_t,i^R]     (content 部分 + RoPE 部分)
  key_i   = [k_t,i^C ; k_t^R]       (content 部分 + 共享 RoPE key)

  score = query_i^T × key_i / sqrt(d_h + d_h^R)
        = q_t,i^C^T × k_t,i^C + q_t,i^R^T × k_t^R
          (content attention)    (position attention)
```

### 3.3 Cache 内容

```
推理时需要 cache:
  1. c_t^KV: latent vector (d_c = 512 dim)
  2. k_t^R: decoupled RoPE key (d_h^R = 64 dim)

  总计: (d_c + d_h^R) × l = (512 + 64) × 60 = 34,560 elements/token

  对比 MHA: 2 × 128 × 128 × 60 = 1,966,080 elements/token

  压缩比: 1,966,080 / 34,560 ≈ 56.9x
```

## 4. Query 压缩 (仅训练)

```
训练时还压缩 query (减少 activation memory):
  c_t^Q = W^DQ × h_t       → [d_c'] 维 (d_c' << d_h × n_h)
  q_t^C = W^UQ × c_t^Q     → 恢复到 [d_h × n_h] 维

DeepSeek-V2 参数:
  d_c' = 1536 (query 压缩维度)
  d_c  = 512  (KV 压缩维度)

Query 压缩不影响 KV cache 大小，但减少训练时的显存占用
```

## 5. KV Cache 对比

### 5.1 数值对比

```
DeepSeek-V2 配置:
  n_h = 128 heads, d_h = 128 dim/head
  d_c = 512 (KV latent dim)
  d_h^R = 64 (RoPE decoupled dim)
  l = 60 layers

KV Cache per token:
  MHA:     2 × 128 × 128 × 60 = 1,966,080 elements
  GQA(8):  2 × 8 × 128 × 60   = 122,880 elements
  MLA:     (512 + 64) × 60     = 34,560 elements

  MLA ≈ GQA with 2.25 groups
  MLA > MHA 性能
```

### 5.2 实际推理效果

```
DeepSeek-V2 vs DeepSeek 67B:
  KV Cache 减少: 93.3%
  生成吞吐量: 5.76x 提升
  单节点 (8×H800): >50K tokens/s 生成
                    >100K tokens/s prompt 输入
```

## 6. 完整计算流程

```
Input: h_t ∈ R^d (hidden state)

Step 1: 压缩
  c_t^Q  = W^DQ × h_t       → R^{d_c'}
  c_t^KV = W^DKV × h_t      → R^{d_c}

Step 2: 恢复 Queries
  q_t^C = W^UQ × c_t^Q       → R^{d_h × n_h} (content queries)
  q_t^R = RoPE(W^QR × c_t^Q) → R^{d_h^R × n_h} (RoPE queries)

Step 3: 恢复 Keys (训练时显式计算，推理时通过矩阵吸收)
  k_t^C = W^UK × c_t^KV      → R^{d_h × n_h} (content keys)
  k_t^R = RoPE(W^KR × h_t)   → R^{d_h^R} (shared RoPE key)

Step 4: 恢复 Values (同理)
  v_t^C = W^UV × c_t^KV      → R^{d_h × n_h}

Step 5: Attention (per head i)
  q_i = [q_t,i^C ; q_t,i^R]
  k_i = [k_t,i^C ; k_t^R]
  o_i = Softmax(q_i^T × k_i / sqrt(d_h + d_h^R)) × v_i^C

Step 6: Output
  u_t = W^O × [o_1; o_2; ...; o_{n_h}]

推理时 Cache: { c_t^KV, k_t^R } (只有这两个需要保存)
```

## 7. MLA vs GQA vs MQA 对比

| 维度 | MHA | GQA | MQA | MLA |
|------|-----|-----|-----|-----|
| KV Cache | 2n_h d_h l | 2n_g d_h l | 2d_h l | (d_c+d_h^R)l |
| 性能 | 基准 | 略降 | 降更多 | **优于 MHA** |
| 压缩方式 | — | 减少头数 | 单头共享 | 低秩投影 |
| 额外计算 | — | — | — | 压缩/恢复投影 |
| RoPE 兼容 | 直接 | 直接 | 直接 | 需要解耦 |

MLA 的独特优势：**用更少的 KV Cache 实现比 MHA 更好的性能**。

## 8. vLLM 中的 MLA 实现

```
vLLM 支持 DeepSeek-V2/V3 的 MLA:
  - 优化的 MLA attention kernel (FlashAttention 变体)
  - 支持 PagedAttention 的 block 管理
  - KV Cache 只存储 latent vector (d_c) + decoupled key (d_h^R)
  - 等效 KV Cache 大小远小于 MHA 模型

关键优化:
  - 矩阵吸收: W^UK → W^Q, W^UV → W^O
  - 减少 memory I/O: 不需要从 latent 恢复完整 K/V
  - FlashMLA kernel: 优化的 MLA attention 实现
```

## 参考资料

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [vLLM MLA Implementation](https://github.com/vllm-project/vllm)
- `notebook/fundamentals/kv-cache.md` — KV Cache 基础
- `notebook/fundamentals/rope.md` — RoPE 旋转位置编码
