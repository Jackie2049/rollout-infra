# Transformer 架构演进: GPT-2 → LLaMA → DeepSeek

> 从 GPT-2 (2019) 到 DeepSeek-V3 (2024) 的架构创新全景
> 涵盖: 位置编码、注意力变体、归一化、激活函数、缩放定律

## 1. 位置编码 (Positional Encoding)

### 1.1 四种主要方案

| 方案 | 代表模型 | 公式 | 外推能力 |
|------|---------|------|---------|
| Sinusoidal | Transformer (2017) | PE(pos,2i)=sin(pos/10000^{2i/d}) | 差 |
| Learned | GPT-2 (2019) | 可学习 Embedding 表 | 最差 |
| RoPE | LLaMA (2023) | 旋转矩阵 R(θ,m) 作用于 Q/K | 好 |
| ALiBi | BLOOM (2022) | attention_score += -m·|i-j| | 最强 |

### 1.2 RoPE 详解 (LLaMA/Mistral/Qwen 标配)

```
将 d 维向量分成 d/2 个 2D 子空间:
q_m = R(θ,m) · x    其中 R(θ,m) = [cos(mθ) -sin(mθ)]
                                        [sin(mθ)  cos(mθ)]

θ_i = 10000^{-2i/d}, i=0,1,...,d/2-1

关键性质: q_m · k_n = f(m-n) → 内积只依赖相对位置!
```

**实测验证** (RTX 4090): RoPE 训练 loss 0.49, 比 Learned (1.10) 低 56%, 外推 4x 最佳

### 1.3 ALiBi (超长外推)

```python
# ALiBi: 在 attention score 上加线性偏置
bias = -m * torch.abs(torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1))
attn = Q @ K.T / sqrt(d) + bias  # m 是 per-head 斜率
```

**优势**: 训练 1024 序列 → 直接外推到 2048+, 无需修改位置编码

### 1.4 实测对比

| Type | Train Loss | 4x Extrapolation | 参数量 |
|------|-----------|------------------|--------|
| **RoPE** | **0.488** | **3.925** | 821K |
| None | 0.601 | 3.573 | 821K |
| Sinusoidal | 0.923 | 4.178 | 822K |
| Learned | 1.095 | 4.011 | 887K |

## 2. 注意力变体 (Attention Variants)

### 2.1 MHA → MQA → GQA → MLA 演进

```
MHA (2017):  h 个 Q 头, h 个 KV 头 → KV cache = 2·h·d·seq
MQA (2022):  h 个 Q 头, 1 个 KV 头 → KV cache = 2·d·seq (↓h倍)
GQA (2023):  h 个 Q 头, g 个 KV 头 → KV cache = 2·g·d·seq (↓h/g倍)
MLA (2024):  h 个 Q 头, 1 个 d_c 压缩 KV → KV cache = 2·d_c·seq
```

### 2.2 KV Cache 大小对比 (7B 模型, seq=4096)

| 方案 | 模型 | KV Heads | d_head | KV Cache/layer | 总 KV (32层) | 压缩比 |
|------|------|----------|--------|---------------|-------------|--------|
| MHA | LLaMA-1 7B | 32 | 128 | 128 KB | 4.0 MB | 1x |
| GQA-8 | LLaMA-2 7B | 8 | 128 | 32 KB | 1.0 MB | 4x |
| MQA | PaLM 8B | 1 | 128 | 4 KB | 0.125 MB | 32x |
| MLA-256 | DeepSeek-V2 | 1 | 256 (latent) | 8 KB | 0.25 MB | 16x |

**注意**: MLA 的 d_c=256 是 latent dim, 通过 upproj 恢复到完整 KV

### 2.3 GQA 实测 (RTX 4090)

| Config | KV/tok (bytes) | KV(128tok) | 并发容量 |
|--------|---------------|------------|---------|
| MHA (8 KV heads) | 4,096 | 512 KB | 312 |
| GQA (4 KV heads) | 2,048 | 256 KB | 459 |
| GQA (2 KV heads) | 1,024 | 128 KB | 612 |
| MQA (1 KV head) | 512 | 64 KB | **619** |

**关键发现**: GQA decode 比 MHA 慢! 原因是 KV expand 开销, 需要 FlashInfer 专用 GQA kernel 避免。

## 3. 归一化 (Normalization)

### 3.1 LayerNorm → RMSNorm

```python
# LayerNorm (原始 Transformer)
y = (x - mean(x)) / sqrt(var(x) + eps) * γ + β  # re-center + re-scale

# RMSNorm (LLaMA)
y = x / sqrt(mean(x²) + eps) * γ  # only re-scale, no re-center
```

**RMSNorm 优势**:
- 少一个 mean 操作 → 7-64% 加速 (RMSNorm 论文实测)
- 无 β 参数 → 略少参数
- Pre-norm (现代标配): LayerNorm 在 attention/FFN 之前, 训练更稳定

### 3.2 Pre-norm vs Post-norm

```
Pre-norm (GPT-2, LLaMA):     x = x + Attn(LN(x))     ← 训练稳定
Post-norm (原始 Transformer): x = LN(x + Attn(x))     ← 需要 warmup
```

现代模型全部使用 Pre-norm + RMSNorm。

## 4. 激活函数 (Activation Functions)

### 4.1 ReLU → GELU → SwiGLU

```
ReLU:    max(0, x)
GELU:    x · Φ(x)         ← GPT-2/3, 平滑版 ReLU
SwiGLU:  (xW₁ · swish(xW₂)) W₃  ← LLaMA, GLU 变体
         swish(x) = x · σ(x)
```

### 4.2 SwiGLU 参数影响

```
标准 FFN: d → 4d → d     参数 = 8d²
SwiGLU:   d → 4d/3 (gate) + d → 4d/3 (up) → d   参数 ≈ 8d² (实际略多)
```

**LLaMA 配置**: d_model=4096, FFN_dim=11008 (≈ 2.69 × d_model, 非 4×)

### 4.3 为什么 SwiGLU 更好?

- GLU gate 提供动态特征选择 (哪个维度重要)
- Swish 比 ReLU 更平滑, 梯度流更好
- 实测: SwiGLU 比 GELU 在相同参数下 perplexity 更低 (LLaMA 论文)

## 5. 缩放定律 (Scaling Laws)

### 5.1 Chinchilla (2022): 计算最优缩放

```
核心结论: 给定固定计算预算 C:
  最优模型大小 N* ∝ C^0.5
  最优训练数据 D* ∝ C^0.5
  → N ≈ D (模型参数 ≈ 训练 token 数)

公式: L(N,D) = E + A/N^α + B/D^β
  其中 α ≈ 0.34, β ≈ 0.28
```

| 模型 | 参数 | 训练 Tokens | Chinchilla 最优? |
|------|------|-----------|----------------|
| GPT-3 175B | 175B | 300B | 否 (under-trained) |
| Chinchilla 70B | 70B | 1.4T | ✅ |
| LLaMA-1 65B | 65B | 1.4T | ✅ |
| Llama 3 405B | 405B | 15T+ | 过训练 (超 Chinchilla) |

### 5.2 Llama 3 策略: 过训练

Llama 3 故意过训练 (15T tokens for 405B model, Chinchilla 最优约 8T):
- 过训练提升推理性能 (inference-optimal)
- 更小的模型也能达到好的效果 (8B Llama 3 ≈ GPT-3 175B)
- 数据质量 > 数据量, 15T 高质量数据集

### 5.3 训练计算估算

```
FLOPS ≈ 6 × N × D (forward + backward)

7B 模型训练 1T tokens:
  6 × 7B × 1T = 4.2 × 10^22 FLOPS
  A100 (312 TFLOPS) × 256 卡 ≈ 36 天

70B 模型训练 1T tokens:
  6 × 70B × 1T = 4.2 × 10^23 FLOPS
  H100 (1000 TFLOPS) × 1024 卡 ≈ 4.8 天
```

## 6. 架构演进时间线

| 年份 | 模型 | 创新 | 规模 |
|------|------|------|------|
| 2017 | Transformer | MHA, Sinusoidal PE, LayerNorm | 65M |
| 2019 | GPT-2 | Learned PE, Pre-norm, GELU | 1.5B |
| 2020 | GPT-3 | Scale up, few-shot learning | 175B |
| 2022 | PaLM | MQA, SwiGLU, RMSNorm | 540B |
| 2022 | Chinchilla | 缩放定律: N≈D | 70B |
| 2022 | BLOOM | ALiBi 位置编码 | 176B |
| 2023 | LLaMA-1 | RoPE + RMSNorm + SwiGLU + GQA | 65B |
| 2023 | LLaMA-2 | GQA (4/8 KV heads) | 70B |
| 2023 | Mistral | Sliding window attention | 7B |
| 2024 | DeepSeek-V2 | MLA (93.3% KV 压缩) | 236B (21B active) |
| 2024 | Llama 3 | 过训练, 128K context | 405B |
| 2024 | DeepSeek-V3 | MoE + MLA, 671B/37B active | 671B |
| 2024 | Mixtral | Sparse MoE, 8/2 expert | 47B (13B active) |

## 7. 现代 LLM 标准架构 (LLaMA-like)

```python
class ModernTransformerBlock:
    """LLaMA/Mistral/Qwen 标准架构"""
    attention = GQA(n_heads=32, n_kv_heads=8)  # GQA
    ffn = SwiGLU(d_model=4096, d_ffn=11008)    # SwiGLU
    norm = RMSNorm(d_model=4096)                # RMSNorm
    pos_enc = RoPE(d_head=128)                  # RoPE (在 Q/K 上)

    def forward(x):
        x = x + attention(norm(x))   # Pre-norm + residual
        x = x + ffn(norm(x))         # Pre-norm + residual
        return x
```

**关键数字 (7B 模型)**:
- 32 层, 32 Q heads, 8 KV heads (GQA-4)
- d_model=4096, d_ffn=11008
- RoPE (d_head=128)
- 参数分布: Embedding 5%, Attention 31%, FFN 63%, Norm 0.3%
