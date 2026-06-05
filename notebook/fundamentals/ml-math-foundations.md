# ML 数学基础 — AI Infra 工程师必备

> Phase 1 of AI Expert Roadmap | 2026-06-05
> 重点: 与 AI 系统直接相关的数学, 不是纯数学教材

## 1. 线性代数

### 1.1 矩阵乘法 (GEMM)

```
C = A @ B, A: (M, K), B: (K, N), C: (M, N)
FLOPs = 2MNK (每个输出元素需要 K 次乘加)

GPT 推理的核心:
  Prefill: Q @ K^T → (B, H, T, T), FLOPs = 2BHT²d_head
  Decode:  q @ K^T → (B, H, 1, T), FLOPs = 2BHTd_head (memory-bound!)

为什么 decode 是 memory-bound?
  Decode GEMM: M=1, N=vocab_size, K=d_model
  算术强度 = FLOPs/bytes = 2·1·N·K / (2·K + 2·N) ≈ 2N·K/(2K) = N
  当 N < Ridge Point (A100: 153) → memory-bound
  实际: vocab=32K > 153 → compute-bound? 不, 因为 batch decode M 很小
```

### 1.2 特征值分解 & SVD

```
特征分解: A = QΛQ^(-1) (方阵)
SVD: A = UΣV^T (任意矩阵)

ML 应用:
1. PCA: 数据降维 = SVD of centered data
2. LoRA: W = W_orig + BA, rank-r decomposition (低秩近似)
   → 参数从 d×d 降到 2×d×r (r << d)
3. 权重量化: SVD 找到主方向, 保护重要特征

LoRA 的数学:
  dW = BA, B: (d, r), A: (r, d)
  奇异值分解: dW = Σ σ_i u_i v_i^T
  保留 top-r 奇异值 → 最优 rank-r 近似 (Eckart-Young 定理)
```

### 1.3 矩阵求导

```
标量对矩阵求导:
  ∂(x^T A x)/∂A = xx^T
  ∂(a^T X b)/∂X = ab^T

链式法则 (矩阵版):
  ∂L/∂W = (∂L/∂y)(∂y/∂W)

Linear 层的反向传播:
  y = Wx + b
  ∂L/∂W = ∂L/∂y · x^T    (outer product)
  ∂L/∂x = W^T · ∂L/∂y     (反向传播到输入)
  ∂L/∂b = ∂L/∂y            (sum over batch)

这就是为什么 bwd = 3.22× fwd (Megatron 实测):
  Fwd: 1x GEMM (W @ x)
  Bwd: 2x GEMM (∂L/∂W and ∂L/∂x)
  + weight gradient (∂L/∂W) 需要 allreduce (TP)
```

## 2. 概率与信息论

### 2.1 信息熵

```
H(X) = -Σ p(x) log p(x)

含义: 编码 X 所需的最少 bits (平均)
  公平硬币: H = -0.5×log(0.5) - 0.5×log(0.5) = 1 bit
  不公平硬币 (99:1): H = 0.08 bit (几乎不需要信息)

语言模型中的应用:
  Cross-entropy loss = -Σ p_data(x) log p_model(x)
  → 模型的平均编码长度
  → 越低越好 (模型越准确)
  → 困惑度 PPL = exp(cross_entropy)

初始 loss = log(vocab_size):
  Char vocab=45: log(45) = 3.81 (随机模型)
  BPE vocab=512: log(512) = 6.24 (随机模型)
  → 这就是 MiniGPT 训练开始时的 loss 值!
```

### 2.2 KL 散度

```
KL(p || q) = Σ p(x) log(p(x)/q(x))

性质:
  1. 非负: KL ≥ 0, 等号 iff p = q
  2. 不对称: KL(p||q) ≠ KL(q||p)
  3. 不是距离度量!

RLHF 中的 KL 惩罚:
  L = E[r(x,y)] - β·KL(π_θ || π_ref)

  含义: 策略 π_θ 不应该偏离参考策略 π_ref 太远
  β 越大 → 越保守 (接近 SFT)
  β 越小 → 越激进 (可能 reward hacking)

DPO 中的隐式 KL:
  h_θ = log(π_θ(y_w|·)/π_ref(y_w|·)) - log(π_θ(y_l|·)/π_ref(y_l|·))
  → 这就是 β·(r(x,y_w) - r(x,y_l))
  → DPO 的 log-ratio 天然包含 KL 约束!
```

### 2.3 Softmax 的温度

```
softmax(x/T) = exp(x_i/T) / Σ exp(x_j/T)

T → 0: argmax (greedy)
T → ∞: uniform distribution
T = 1: 标准 softmax

为什么温度影响生成多样性?
  T 小: 概率分布尖锐 → 总是选最可能的 token → 确定性
  T 大: 概率分布平滑 → 更可能选低概率 token → 多样性

MiniGPT 实测:
  T=0.1-0.8: 完全复述训练数据 (过拟合)
  T=1.0: 开始出现变化
  T=1.5: 产生乱码 (过度随机)

数学解释:
  过拟合模型的概率分布本身就极其尖锐 (P(correct) ≈ 1)
  → 需要很高的 T 才能引入足够的随机性
  → 但高 T 也破坏了正确模式 → 乱码
```

## 3. 优化理论

### 3.1 SGD → Adam → AdamW

```
SGD: θ_{t+1} = θ_t - η · g_t
  简单, 但学习率敏感, 容易陷入局部最优

Adam: 自适应学习率
  m_t = β₁·m_{t-1} + (1-β₁)·g_t          # 一阶矩 (momentum)
  v_t = β₂·v_{t-1} + (1-β₂)·g_t²          # 二阶矩 (RMSprop)
  m̂_t = m_t / (1-β₁^t)                     # bias correction
  v̂_t = v_t / (1-β₂^t)
  θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)

  β₁=0.9, β₂=0.98 (Transformer), ε=10⁻⁹

  为什么 Adam 好?
  1. 自适应: 梯度大的参数 → 小学习率, 梯度小的 → 大学习率
  2. Momentum: 平滑噪声梯度 → 更稳定的方向
  3. 不需要手动调学习率 schedule (但 warmup 仍然重要)

AdamW: 解耦 weight decay
  θ_{t+1} = θ_t - η · (m̂_t / (√v̂_t + ε) + λ·θ_t)

  weight decay λ 独立于 Adam 的自适应率
  → 更好的正则化效果
  → 所有现代 Transformer 用 AdamW
```

### 3.2 学习率调度

```
为什么需要 warmup?

Adam 的初始阶段:
  m_0 = 0, v_0 = 0
  前 few steps: 二阶矩估计不准 → 除以很小的 v̂ → 学习率爆炸
  warmup 让模型在初期用小学习率 → 积累准确的矩估计

Cosine Schedule (LLaMA/GPT):
  lr(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(πt/T))

  特点:
  - 开始大, 结束小 (exploration → exploitation)
  - 中间平滑过渡 (cosine curve)
  - η_min 通常 = 0.1 × η_max

MiniGPT 训练验证:
  无 warmup → 前 100 步 loss 震荡
  warmup=100 → 稳定下降
  cos schedule → 比 linear decay 最终 loss 低 ~5%
```

### 3.3 Gradient Clipping

```
clip_grad_norm_(parameters, max_norm=1.0)

为什么需要?
  - Transformer 的梯度可能突然变得很大 (loss spike)
  - 原因: 注意力权重的 softmax 饱和 → 梯度 ×0 → 下一层梯度 ×∞
  - 解决: 裁剪梯度的 L2 norm

  if ||g|| > max_norm:
    g = g × max_norm / ||g||

  不改变方向, 只改变大小
  → 防止参数更新太大导致 loss 爆炸

  实测: 去掉 gradient clipping → loss 偶尔 spike 到 10+
        加上 clipping → loss 平稳下降
```

## 4. 深度学习专用数学

### 4.1 FLOPs 计算

```
单个矩阵乘法: C = A @ B, A:(M,K), B:(K,N)
FLOPs = 2MNK (M×N 个输出, 每个 K 次乘加)

Transformer 层 FLOPs:
  QKV projection: 3 × 2Nd² = 6Nd²
  Attention: QK^T = 2N²d, Attn@V = 2N²d → 4N²d
  Output projection: 2Nd²
  FFN (SwiGLU): 3 × 2Nd × 4d × 2/3 = 16Nd²
  Total per layer ≈ 24Nd² + 4N²d

  当 N < 6d: FFN-dominated (compute-bound, typical for prefill)
  当 N > 6d: Attention-dominated (memory issue)

训练总 FLOPs ≈ 6ND (Chinchilla 推导)
  N = sequence length, D = dataset size in tokens
  6 = forward(2) + backward(4)
```

### 4.2 反向传播的梯度

```
Softmax 的梯度:
  ∂L/∂z_i = p_i - y_i  (p: softmax output, y: one-hot target)

  为什么 attention 的反向传播复杂?
  S = softmax(QK^T/√d)
  A = S @ V
  ∂L/∂S = (∂L/∂A · V^T) ⊙ (S - S²)  ... 更复杂
  实际: ∂L/∂S = A·(dL/dA - Σ(dL/dA·A))

Cross-entropy 的梯度:
  L = -Σ y_i log(p_i)
  ∂L/∂z_i = p_i - y_i  (softmax + CE 的组合梯度!)

  → 非常简洁! 只需要 softmax 输出减去 target
  → 这是为什么 PyTorch 的 CrossEntropyLoss 融合了 softmax
```

### 4.3 KV Cache 的数学

```
每个 token 的 KV cache:
  per_layer = 2 × d_kv × bytes_per_element
  total = per_layer × n_layers × n_kv_heads

LLaMA-7B (GQA, 8 KV heads):
  d_kv = 128, n_layers = 32, n_kv_heads = 8, FP16 = 2 bytes
  per_token_per_layer = 2 × 128 × 2 = 512 bytes
  total = 512 × 32 = 16,384 bytes/token ≈ 16 KB/token

序列长度 128K:
  KV = 16KB × 128K = 2GB per request
  Batch=32: 64GB KV → 远超 24GB GPU!

解决方案:
  1. PagedAttention: 块管理, 无碎片
  2. Quantization: FP8 → 8KB/token (50% 节省)
  3. Offload: CPU/SSD 缓存
  4. MLA: 压缩 KV → 56.9x reduction
```

## 5. 实用公式速查

```
模型参数量:
  Transformer = n_layers × (12d² + 4d × 4d × 2/3 + 4d)
              ≈ 12.67 × n_layers × d²  (不含 embedding)
  Embedding = vocab_size × d
  Total ≈ 12.67 × L × d² + V × d

训练内存 (AdamW):
  per_param = 2 bytes (FP16 weight)
            + 2 bytes (FP16 gradient)
            + 4 bytes (FP32 master weight)
            + 4 bytes (FP32 momentum)
            + 4 bytes (FP32 variance)
            = 16 bytes/param
  7B model = 112 GB (至少)

推理内存:
  Weights: 2 bytes × n_params (FP16)
  KV Cache: 2 × d_kv × n_layers × n_kv_heads × seq_len × batch × 2 bytes

吞吐量估算:
  Decode: tok/s = HBM_BW / (2 × n_params × bytes_per_param / batch)
  Prefill: tokens/ms = TFLOPS × 0.5 / (6 × n_params × seq_len)
```
