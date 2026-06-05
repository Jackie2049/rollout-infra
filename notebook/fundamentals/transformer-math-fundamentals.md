# Transformer 数学基础: 注意力、梯度与缩放

> 从第一性原理推导 Transformer 核心数学
> 目标: 理解每个组件的数学本质, 而非黑盒调用

## 1. Self-Attention 数学推导

### 1.1 缩放点积注意力 (Scaled Dot-Product Attention)

**输入**: Query Q, Key K, Value V
- Q: [B, H, N, d_k]  (batch, heads, seq_len, head_dim)
- K: [B, H, M, d_k]
- V: [B, H, M, d_v]

**推导**:

```
Step 1: 计算 attention scores
  S = Q @ K^T         → [B, H, N, M]

  直觉: S[i,j] = Q[i] · K[j] = Σ_d Q[i,d] * K[j,d]
  这是 query 和 key 的相似度度量 (点积 = cosine × |Q| × |K|)

Step 2: 缩放
  S_scaled = S / √d_k

  为什么除以 √d_k?
  - 假设 Q, K 的每个元素 i.i.d. N(0,1)
  - Q·K = Σ_d x_d·y_d, 每项 E[x_d·y_d] = 0, Var[x_d·y_d] = 1
  - Q·K 的方差 = d_k (d_k 个独立项之和)
  - 标准差 = √d_k
  - 除以 √d_k → 方差归一化为 1
  - 不缩放: d_k=128 时, |Q·K| ≈ √128 ≈ 11.3
  - softmax(11.3) → 极端 one-hot → 梯度消失 (接近 0)

Step 3: Softmax
  A = softmax(S_scaled, dim=-1)   → [B, H, N, M]

  softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

  数值稳定版本:
  softmax(z)_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))

Step 4: 加权求和
  output = A @ V     → [B, H, N, d_v]

  直觉: 每个 token 的输出 = 所有 token 的 value 的加权平均
  权重 = 该 token 的 query 与每个 token 的 key 的相似度
```

**完整公式**:
```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

### 1.2 为什么缩放很重要? (数值证明)

```python
# 不缩放时 softmax 的梯度问题
import torch

d_k = 128
Q = torch.randn(1, 1, 1, d_k)
K = torch.randn(1, 1, 10, d_k)

scores = Q @ K.transpose(-2, -1)
print(f"不缩放: max score = {scores.max():.1f}, std = {scores.std():.1f}")
# max score ≈ 11, std ≈ 11.3

probs = torch.softmax(scores, dim=-1)
print(f"不缩放: max prob = {probs.max():.4f}")
# max prob ≈ 0.9999 → 接近 one-hot

scaled_scores = scores / (d_k ** 0.5)
probs_scaled = torch.softmax(scaled_scores, dim=-1)
print(f"缩放后: max prob = {probs_scaled.max():.4f}")
# max prob ≈ 0.15 → 更均匀的分布
```

### 1.3 Multi-Head Attention (MHA)

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

参数量:
  W^Q: [d_model, d_k × h] = [d_model, d_model]
  W^K: [d_model, d_k × h] = [d_model, d_model]
  W^V: [d_model, d_v × h] = [d_model, d_model]
  W^O: [d_v × h, d_model] = [d_model, d_model]
  总计: 4 × d_model²

对于 7B 模型 (d_model=4096):
  Attention 参数 = 4 × 4096² = 67M (占总参数 ~31%)
```

### 1.4 GQA (Grouped Query Attention)

```
MHA: h 个 Q 头, h 个 KV 头 → 每对独立
GQA: h 个 Q 头, g 个 KV 头 → 每 (h/g) 个 Q 头共享一组 KV

实现:
  # K, V 的 shape: [B, g, M, d_k]  (g < h)
  # 需要扩展到 [B, h, M, d_k]
  K_expanded = K.repeat_interleave(h // g, dim=1)
  V_expanded = V.repeat_interleave(h // g, dim=1)

KV Cache 节省:
  MHA (h=32): 2 × 32 × 128 = 8,192 bytes/tok/layer
  GQA (g=8):  2 × 8  × 128 = 2,048 bytes/tok/layer → 4x 节省
  MQA (g=1):  2 × 1  × 128 = 256   bytes/tok/layer → 32x 节省
```

## 2. 反向传播推导

### 2.1 Attention 的反向传播

```
已知: dL/d(output) = ∂L/∂O   shape: [B, H, N, d_v]

需要: dL/dQ, dL/dK, dL/dV

Step 1: dL/dV
  O = A @ V
  ∂L/∂V = A^T @ ∂L/∂O          shape: [B, H, M, d_v]

Step 2: dL/dA
  O = A @ V
  ∂L/∂A = ∂L/∂O @ V^T          shape: [B, H, N, M]

Step 3: dL/dS (softmax 反向传播)
  S → A = softmax(S)

  softmax 的 Jacobian:
  ∂A_i/∂S_j = A_i(δ_ij - A_j)

  对于每行:
  ∂L/∂S = A ⊙ (∂L/∂A - ( (∂L/∂A ⊙ A) · 1^T ))

  简化:
  dL/dS = A * (dL/dA - sum(dL/dA * A, dim=-1, keepdim=True))

Step 4: dL/dQ 和 dL/dK
  S = Q @ K^T / √d_k

  dL/dQ = (dL/dS @ K) / √d_k    shape: [B, H, N, d_k]
  dL/dK = (dL/dS^T @ Q) / √d_k  shape: [B, H, M, d_k]
```

**FlashAttention 的关键**: 前向时不存储 N×M 的 attention matrix, 而是用 tiling + online softmax 在反向时重新计算。这节省 O(N²) 内存。

### 2.2 Softmax 数值稳定性的反向传播

```
前向:
  m = max(S)                     # 每行最大值
  P = exp(S - m)                 # 减最大值防止溢出
  l = sum(P)                     # 每行的 sum
  A = P / l                      # 归一化

反向:
  dA/dS 的推导 (利用数值稳定形式):
  令 z = S - m, P = exp(z), l = sum(P)

  dL/dz = A * (dL/dA - sum(dL/dA * A))

  注意: 这里 A 和 dL/dA 都已在手, 不需要存储中间 z 或 P
  FlashAttention 正是利用这一点在反向时重算 A
```

### 2.3 LayerNorm / RMSNorm 反向传播

```
LayerNorm:
  y = (x - μ) / √(σ² + ε) * γ + β
  μ = mean(x), σ² = var(x)

  设 x̂ = (x - μ) / √(σ² + ε)

  ∂L/∂γ = Σ ∂L/∂y * x̂
  ∂L/∂β = Σ ∂L/∂y

  ∂L/∂x̂ = ∂L/∂y * γ

  ∂L/∂σ² = Σ (∂L/∂x̂ * (x - μ)) * (-1/2) * (σ² + ε)^(-3/2)

  ∂L/∂μ = Σ (∂L/∂x̂ * (-1/√(σ² + ε))) + ∂L/∂σ² * mean(-2(x - μ))

  ∂L/∂x = ∂L/∂x̂ / √(σ² + ε) + ∂L/∂σ² * 2(x - μ)/d + ∂L/∂μ / d

  → 形状不变, 但计算复杂 (3 个中间梯度)

RMSNorm:
  y = x / √(mean(x²) + ε) * γ

  设 r = √(mean(x²) + ε)

  ∂L/∂γ = Σ ∂L/∂y * (x / r)

  ∂L/∂x = ∂L/∂y * γ * (1/r - x² / (r³ × d))

  → 少了 μ 和 σ² 的交互项, 更简洁
  → 实测快 7-64% (RMSNorm 论文)
```

### 2.4 为什么 FLOPS = 6ND?

```
前向:
  矩阵乘法 A[m,n] @ B[n,p] → 2mnp FLOPs (乘+加)
  一个 Transformer 层:
    QKV projection: 3 × 2d²N = 6d²N
    Attention: 2dN² (QK^T) + 2dN² (AV) = 4dN²
    Output projection: 2d²N
    FFN (SwiGLU): 3 × 2d × 4d × N = 24d²N (3 个矩阵)
    总计 per layer ≈ 36d²N + 4dN²

  对于 N >> d (长序列): attention 项主导
  对于 d >> N (大模型短序列): 线性项主导 (实际通常如此)

  L 层模型: 前向 ≈ L × (36d²N + 4dN²)

反向:
  需要为每个参数计算梯度
  - 每个矩阵乘的反向 = 2 次矩阵乘 (对两个输入求梯度)
  - 反向 ≈ 2x 前向 FLOPs

总计: 前向 (1x) + 反向 (2x) = 3x per parameter per token

FLOPs ≈ 6 × N_params × N_tokens
  对于 7B 模型训练 1T tokens:
  6 × 7e9 × 1e12 = 4.2e22 FLOPs
```

## 3. 位置编码的数学

### 3.1 RoPE (Rotary Position Embedding)

```
核心思想: 在 2D 子空间中旋转 Q 和 K

对于 d 维向量, 分成 d/2 个 2D 子空间:
  x = [(x₁,x₂), (x₃,x₄), ..., (x_{d-1}, x_d)]

对位置 m 的 token:
  R(θ,m) x = [(x₁cos(mθ₁) - x₂sin(mθ₁), x₁sin(mθ₁) + x₂cos(mθ₁)),
              (x₃cos(mθ₂) - x₄sin(mθ₂), x₃sin(mθ₂) + x₄cos(mθ₂)),
              ...]

θ_i = 10000^{-2i/d}  (i = 0, 1, ..., d/2-1)

关键性质:
  R(θ,m) x · R(θ,n) y = x · R(θ, n-m) y

  → 内积只依赖相对位置 (m-n)!
  → 这就是 RoPE 能外推的原因

实现 (高效):
  # 不要真的构造旋转矩阵 (d×d 稀疏), 用复数乘法:
  q_complex = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
  freqs = torch.exp(-math.log(10000) * torch.arange(0, d, 2) / d)
  freqs_complex = torch.polar(torch.ones_like(freqs), m * freqs)
  q_rotated = torch.view_as_real(q_complex * freqs_complex).flatten(-2)
```

### 3.2 ALiBi (Attention with Linear Biabilities)

```
核心思想: 在 attention score 上加线性偏置

  attention_score[i,j] = (Q[i] · K[j]) / √d_k - m · |i - j|

  m 是 per-head 的斜率:
  m_h = 2^(-8h/H)  (h = 0, 1, ..., H-1)

  第一个 head: m = 2^0 = 1.0  (最强衰减)
  最后一个 head: m = 2^{-8} = 0.004  (最弱衰减)

为什么能外推?
  - 训练时学到 "近距离 token 重要, 远距离不重要" 的先验
  - 这个偏置在更长序列上自然延续 (线性衰减)
  - 不需要修改位置编码 → 直接外推

与 RoPE 对比:
  RoPE: 修改 Q 和 K → 只依赖相对位置
  ALiBi: 修改 attention score → 明确的距离衰减
  两者都是相对位置编码, 但作用位置不同
```

## 4. 训练的数学: 学习率与优化

### 4.1 Adam 优化器

```
一阶矩: m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
二阶矩: v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

偏差修正:
  m̂_t = m_t / (1 - β₁^t)
  v̂_t = v_t / (1 - β₂^t)

参数更新:
  θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

默认: β₁=0.9, β₂=0.999, ε=1e-8

内存开销: 每参数需要 3 个额外变量 (m, v, 梯度)
  FP32 训练: 模型权重 4B + 梯度 4B + m 4B + v 4B = 16 bytes/param
  BF16 + Adam: 权重 2B + 梯度 2B + m(FP32) 4B + v(FP32) 4B = 12 bytes/param
  → 训练内存 ≈ 12-16 bytes/param (Adam 是大头, 占 50%)
```

### 4.2 学习率调度

```
Warmup + Cosine Decay (LLaMA 标准):

  lr(t) = lr_max × min(1, t/T_warmup) × (1 + cos(π × min(1, (t - T_warmup) / (T_total - T_warmup)))) / 2

  前 T_warmup 步: 线性增长到 lr_max
  之后: 余弦衰减到 lr_min (通常 = 0.1 × lr_max)

  LLaMA 配置:
    lr_max = 3e-4
    T_warmup = 2000 steps
    T_total = 总步数

为什么需要 warmup?
  - 初始参数随机 → 梯度不稳定
  - 大学习率 → 参数更新过大 → 训练崩溃
  - warmup 让优化器的 m, v 累积足够信息后再加速

为什么 cosine decay?
  - 训练后期需要更小的学习率精细调整
  - cosine 比 step decay 更平滑
  - 实测效果: cosine > linear decay > step decay
```

## 5. 从数学到 Infra: 为什么这些很重要

### 5.1 Attention 的 FLOPs 和 IO

```
Attention 的计算瓶颈:
  - Prefill (长序列): O(N²) 的 attention → compute-bound
  - Decode: O(N) per token (遍历所有 KV) → memory-bound

  FlashAttention 解决的问题:
  - 标准 attention: O(N²) SRAM (存 attention matrix)
  - FlashAttention: O(N) SRAM (tiling, 只存 O(√M) 的 block)
  - 代价: 反向需要重算 attention → 额外 ~1.5x FLOPs
  - 收益: 省了 HBM 读写 → 快 2-4x

  HBM 读写量:
  标准: Q, K, V 读 + S 写 + S 读 + A 写 + O 写
       = 2Nd + 2N² + 2N² + N² + 2Nd ≈ 4N² + 4Nd
  FlashAttention: Q, K, V 读 + O 写 + l, m 写
       = 2Nd + 2Nd + 2N ≈ 4Nd  (省了 N² 项!)
```

### 5.2 KV Cache 的内存数学

```
KV Cache per token per layer:
  2 (K+V) × n_kv_heads × d_head × dtype_size

  7B 模型 (LLaMA-like, BF16):
    2 × 32 × 128 × 2 = 16,384 bytes = 16 KB/token/layer
    32 层: 512 KB/token

  Sequence length = 4096:
    KV Cache = 512 KB × 4096 = 2 GB

  Batch size = 32:
    KV Cache = 64 GB → 可能超过模型权重 (14 GB for 7B BF16)

  → 这就是为什么需要 GQA, Paged Attention, KV Cache 优化!

  GQA (8 KV heads):
    2 × 8 × 128 × 2 = 4,096 bytes = 4 KB/token/layer → 4x 节省
    32 层 × 4096 × 32 = 16 GB (可管理)
```

### 5.3 模型并行的通信量

```
Tensor Parallelism (TP):
  每层 2 次 AllReduce (attention + FFN)
  通信量 = 2 × 2 × B × N × d × (TP - 1) / TP

  TP=4, B=32, N=2048, d=4096, BF16:
    每步 = 4 × 32 × 2048 × 4096 × 2 bytes = 4 GB
    NVLink (300 GB/s): 4/300 = 13ms

Pipeline Parallelism (PP):
  每层 1 次 P2P 通信 (发送 activation)
  通信量 = B × N × d × dtype_size
  = 32 × 2048 × 4096 × 2 = 512 MB
  NVLink: 512/300 = 1.7ms

  → TP 通信量 >> PP 通信量
  → 所以 TP 跨节点 (慢网络) 效果差, PP 跨节点影响小
```

## 6. 数学速查表

```
┌─────────────────────────────────────────────────────────────┐
│ 组件            │ FLOPs          │ 内存 (前向) │ 通信       │
│─────────────────│────────────────│─────────────│────────────│
│ Attention       │ 4dN²           │ 4Nd + N²    │ TP: 2BdN   │
│ FFN (SwiGLU)    │ 24d²N          │ 20dN        │ TP: 2BdN   │
│ LayerNorm       │ 5dN            │ 4dN         │ SP: 2BdN/T │
│ Softmax         │ 3N²            │ N²          │ 无         │
│ KV Cache        │ 无 (数据)       │ 2hNd × L    │ 无         │
│ Embedding       │ dV             │ dN          │ 无         │
│ Adam 优化器     │ 无 (元数据)     │ 12B/param   │ 无         │
└─────────────────────────────────────────────────────────────┘

d = d_model, N = seq_len, B = batch_size, h = n_heads
V = vocab_size, L = n_layers, T = TP degree
```
