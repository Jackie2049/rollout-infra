# Transformer 架构深度解析 — 组件级分析

> 2026-06-07 | 从零拆解Transformer每个组件: embedding, attention, MLP, LN, residual, positional encoding
> 连接RTX 4090实测: FlashAttention, RMSNorm 9x加速, decode memory-bound

## 一、Transformer整体架构

```
标准Transformer Block:
  x → LayerNorm → Attention → +x → LayerNorm → MLP → +x → output

数据流:
  Input: (B, S, D)  [batch, sequence, hidden_dim]
  After Attn: (B, S, D)  [同维度, 但融合了全局信息]
  After MLP: (B, S, D)  [同维度, 但做了非线性变换]
  Output: (B, S, D)  [同维度]

关键设计原则:
  1. 残差连接: 每个子层输出 + 输入 → 防止梯度消失/爆炸
  2. LayerNorm: 在子层之前(Norm-first)或之后(Post-norm)
  3. 位置编码: 注入位置信息(attention本身是位置无关的)
  4. 多头注意力: 多个独立的attention head → 多视角
```

## 二、组件逐一解析

### 2.1 Embedding层

```
功能: 将离散token ID → 连续向量表示

Input Embedding:
  vocab_size → d_model 的映射
  W_embed: (V, D)  [V=vocab_size, D=d_model]
  x_embed = W_embed[token_ids]  → (B, S, D)

参数量: V × D
  GPT-2: 50257 × 768 = 38.6M (7.7% of 124M total)
  LLaMA-2: 32000 × 4096 = 131M (0.9% of 7B → 很小!)

关键设计:
  → 词表大小选择: 32K(LLaMA) vs 50K(GPT-2) vs 100K+(Qwen)
  → 大词表: 更高效编码(更短序列) → 但embedding参数更多
  → Qwen3-27B: vocab=248320 → embedding占248320×4608≈1.15B参数(4.3%)
  → DeepSeek-V3: vocab=129280 → ~0.6B参数(小词表+MLA压缩)

RTX 4090影响:
  → Embedding是lookup操作 → memory-bound(B=1时尤其慢)
  → 但占比小 → 不是瓶颈
```

### 2.2 位置编码 (Positional Encoding)

```
功能: 注入位置信息(Attention是set operation → 无位置感知)

方案对比:
| 方法 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| Sinusoidal(原始) | sin/cos固定编码 | 无参数, 可扩展 | 外推差 |
| Learned | 可学习参数 | 灵活 | 长度限制 |
| RoPE | 旋转位置编码 | 相对位置, 外推好 | 实现复杂 |
| ALiBi | 线性偏置 | 简单, 外推好 | 绝对位置信息弱 |

RoPE (Rotary Position Embedding) — 当前主流:
  → 将位置信息编码为旋转矩阵
  → q_i = W_q × x_i → 旋转 → q_i' = rotate(q_i, pos_i)
  → k_j = W_k × x_j → 旋转 → k_j' = rotate(k_j, pos_j)
  → attn = q_i' · k_j' = (rotate(q_i, pos_i)) · (rotate(k_j, pos_j))
  → = q_i · k_j × cos(pos_i - pos_j) + ... × sin(pos_i - pos_j)
  → 内积只依赖相对位置(pos_i - pos_j)! → 天然相对位置编码

RoPE数学:
  rotate(x, θ) = [[cos θ, -sin θ], [sin θ, cos θ]] × x
  θ_i = θ_base × pos_i  (θ_base = 10000^(-2i/D))

  → 低维度: θ大 → 精细位置区分(局部)
  → 高维度: θ小 → 粗略位置区分(全局)
  → 自然形成了局部+全局的多尺度位置感知!

外推性:
  → RoPE: 需要位置插值(Position Interpolation/YaRN)
  → LLaMA-2 4K → 32K: PI缩放θ → 或NTK-aware缩放
  → DeepSeek-V3: YaRN + 128K context → 位置缩放是关键!

RTX 4090影响:
  → RoPE计算量小(逐元素旋转) → 不是瓶颈
  → 但长序列(128K)时 → RoPE缓存需考虑
```

### 2.3 Multi-Head Attention (核心!)

```
功能: 全局信息聚合 — 每个token关注所有其他token

数学:
  Q = W_q × x  → (B, S, D) → (B, S, D)
  K = W_k × x  → (B, S, D) → (B, S, D)
  V = W_v × x  → (B, S, D) → (B, S, D)

  Multi-Head: 将D拆成H个head, 每个head维度D/H
  Q_i = Q[:, :, i*D/H:(i+1)*D/H]  → (B, S, D/H)
  K_i = K[:, :, i*D/H:(i+1)*D/H]  → (B, S, D/H)
  V_i = V[:, :, i*D/H:(i+1)*D/H]  → (B, S, D/H)

  Attn_i = softmax(Q_i × K_i^T / √(D/H)) × V_i  → (B, S, D/H)
  Output = concat(Attn_1, ..., Attn_H) × W_o  → (B, S, D)

参数量: 4 × D² (W_q, W_k, W_v, W_o)
  7B: 4 × 4096² ≈ 67M per layer × 32 layers ≈ 2.1B (30% of total)

计算量: 2 × B × S² × D (QK^T + Attn×V) + 4 × 2BSD (linear projections)
  → O(S²)是瓶颈! → FlashAttention解决内存, 但compute仍是S²

RTX 4090实测:
  → FlashAttention: 内存省85-97%, decode更慢0.67-0.84x
  → Decode M=1: 0.75 TFLOPS(0.45%peak) → 严重memory-bound
  → GQA-8: KV load降87.5% → 但compute不变
  → MLA: KV压缩56.9x → DeepSeek-V3关键创新

Softmax梯度 (理论连接):
  → softmax BP = A × (dA - Σ(dA·A))  → 我们在backprop理论中推导
  → FlashAttention = online softmax增量更新 → 我们的Ring Attention实测验证(cos_sim=1.0)
  → softmax饱和问题 → √d_k scaling → 我们的attention math笔记详细分析
```

### 2.4 MLP层 (FFN)

```
功能: 非线性变换 — 注意力做信息聚合, MLP做特征变换

标准MLP:
  MLP(x) = W_2 × GELU(W_1 × x + b_1) + b_2
  W_1: (D, 4D) → 升维4倍  (d_model → d_ff)
  W_2: (4D, D) → 降维回D  (d_ff → d_model)

参数量: 2 × D × 4D = 8D²
  7B: 8 × 4096² ≈ 134M per layer × 32 layers ≈ 4.3B (61% of total!)
  → MLP占参数61%! → 是Transformer最大的参数块!

GELU激活函数:
  GELU(x) = x × Φ(x)  (Φ是标准正态CDF)
  ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
  → 平滑版ReLU → 在0附近有渐变 → 梯度更好

SwiGLU (当前主流, LLaMA/DeepSeek使用):
  SwiGLU(x, W, V, b) = Swish(Wx + b) ⊙ (Vx + b)
  → 三个矩阵: W (D→d_ff), V (D→d_ff), W_2 (d_ff→D)
  → 参数量: 3 × D × d_ff → 但d_ff通常 = 8D/3 (而非4D)
  → 效果比GELU-MLP更好(SwiGLU > GELU > ReLU)

RTX 4090实测连接:
  → Prefix Sharing: MLP占计算82%! attn-only仅0.99x → MLP是主导
  → → 这解释了为什么Prefix Sharing(full-model)才有效
  → → Prefix Sharing不是仅省attn → 是省整个block的计算
  → Decode时: MLP = 2 × B × S × D × 4D matmul → memory-bound
  → → MLP推理瓶颈 = weight loading = 8D² bytes per token
```

### 2.5 LayerNorm / RMSNorm

```
功能: 归一化激活值 → 稳定训练(防止梯度爆炸/消失)

LayerNorm (原始):
  LN(x) = γ × (x - μ) / σ + β
  μ = mean(x), σ = std(x)
  γ, β: 可学习参数 (维度D)
  → 中心化(减均值) + 标准化(除标准差) + 缩放+偏移

RMSNorm (LLaMA/DeepSeek使用):
  RMSNorm(x) = γ × x / RMS(x)
  RMS(x) = √(mean(x²))
  → 不减均值! 不除标准差! → 更简单更快
  → 效果与LayerNorm几乎相同(Sqrt均值 ≈ 标准差 when mean≈0)

参数量: D (γ only, 无β)
  7B: 4096 × 2 (per block) × 32 blocks = 262K → 极小!

RTX 4090实测:
  → CUDA RMSNorm: 9x over PyTorch, 5x over Triton!
  → 关键优化: 1 warp/row + butterfly shuffle reduction
  → backward 2.2x (2-pass: 保存inv_rms → 省22%)
  → → RMSNorm是实现层面的优化重点(PyTorch实现太慢!)

理论连接:
  → LN假设: Gaussian分布(中心极限定理) → mean≈0 → RMS≈σ
  → → RMSNorm有效是因为神经网络激活近似Gaussian!
  → → 我们的概率论笔记: LN = Gaussian假设 normalization
```

### 2.6 残差连接 (Residual Connection)

```
功能: 跳跃连接 → 防止深层网络梯度消失

数学:
  y = x + SubLayer(LN(x))  (Pre-Norm设计)

梯度路径:
  ∂L/∂x = ∂L/∂y × (1 + ∂SubLayer/∂x)
  → 梯度至少有∂L/∂y × 1 → 直接传播! → 不会消失!

深层影响:
  N层网络: 梯度 = ∂L/∂y_N × ∏(1 + ∂SubLayer_i/∂x_i)
  → 每层至少×1 → 即使100层, 梯度不会消失到0
  → → 残差连接是深层Transformer可训练的关键!

Pre-Norm vs Post-Norm:
  Pre-Norm: y = x + SubLayer(LN(x))  → 当前主流(LLaMA/DeepSeek)
  Post-Norm: y = LN(x + SubLayer(x))  → 原始Transformer

  Pre-Norm优势:
  → 梯度流更稳定(残差路径无LN)
  → 训练不需要warmup
  → 更容易训练深层网络

  Post-Norm问题:
  → LN在残差之后 → 梯度需经过LN → 不稳定
  → 需要careful warmup + learning rate scheduling

RTX 4090影响:
  → 残差连接增加内存(需保存两个分支的输入)
  → → 但FSDP下残差不增加太多(shard已经降低峰值)
  → → Checkpointing+FSDP下残差反而增内存(需re-gather参数)!
```

### 2.7 Dropout

```
功能: 训练时随机屏蔽部分神经元 → 防止过拟合 → 正则化

数学:
  Dropout(x) = x × mask / (1-p)  (p是dropout概率)
  mask = Bernoulli(1-p) → 每个元素独立随机0或1
  → 训练时: 随机屏蔽 → 推理时: 不屏蔽(已缩放)

理论连接(概率论笔记):
  → Dropout = Bernoulli噪声 → ≈ 贝叶斯近似(模型平均)
  → E[Dropout(x)] = x → 无偏估计
  → Var[Dropout(x)] = p/(1-p) × x² → 增加不确定性

RTX 4090实测:
  → LLaMA/DeepSeek: 训练dropout=0! → RL训练也不需要dropout
  → → 为什么? 大模型+大数据 → 过拟合风险低 → dropout不必要
  → → 我们的小模型实验: dropout降低训练速度但增加稳定性
  → → GRPO训练: dropout=0更好(更多样本→组比较→天然正则化)
```

## 三、Transformer设计演变

```
架构演变时间线:

2017: 原始Transformer (Post-Norm + Sinusoidal + ReLU + 4×FFN)
2018: GPT-2 (Pre-Norm(部分) + Learned PE + GELU)
2019: GPT-3 (Pre-Norm + Sparse Attention + 4×FFN)
2020: BERT变体 (Post-Norm + Learned PE)
2022: LLaMA (Pre-Norm + RoPE + SwiGLU + RMSNorm + 8/3×FFN)
2023: LLaMA-2 (同LLaMA + GQA)
2024: DeepSeek-V3 (MLA + MoE + FP8训练 + MTP)
2025: Qwen3 (RoPE + SwiGLU + GQA + DeltaNet混合)

关键改进:
  → Sinusoidal → RoPE: 位置外推性(4K→128K+)
  → ReLU → GELU → SwiGLU: 激活函数效果提升
  → LayerNorm → RMSNorm: 计算更简单, 效果几乎相同
  → Post-Norm → Pre-Norm: 训练稳定性大幅提升
  → MHA → GQA: KV cache减少(8×更少KV头)
  → Dense → MoE: 参数效率(37B active vs 671B total)
  → Full attention → MLA: KV压缩56.9x → DeepSeek核心创新

每个改进的RTX 4090实测验证:
  → RoPE: 无额外开销(逐元素旋转) ✓
  → SwiGLU: MLP占82%计算 → prefix sharing加速 ✓
  → RMSNorm: CUDA 9x加速 → 实现层面重要 ✓
  → GQA: KV load降87.5% → 推理关键 ✓
  → MoE: Python 6-164x慢 → 必须 fused kernel ✓
  → MLA: KV压缩56.9x → 但RTX 4090推理需vLLM MLA backend
```

## 四、计算/内存分析

```
Transformer训练单步分析 (7B模型, B=8, S=1024):

计算量:
  Attention: 4 × 2BSD = 4×2×8×1024×4096 ≈ 0.27 GFLOPS (projection)
            + 2×BS²D = 2×8×1024²×4096 ≈ 0.17 GFLOPS (QK+AV)
  MLP:      3 × 2BSD×(8D/3) ≈ 3×2×8×1024×4096×10922 ≈ 0.55 GFLOPS
  Total per layer ≈ 1.0 GFLOPS
  × 32 layers ≈ 32 GFLOPS per step

内存 (FP32):
  Params: 7B × 4 = 28GB
  Optimizer (Adam): 7B × 12 = 84GB (params+moment+variance)
  Activation: ≈ 2GB (B=8, S=1024, 32 layers)
  Gradient: 7B × 4 = 28GB
  Peak = 28+84+2+28 = 142GB! → 远超RTX 4090 24GB!

→ ZeRO-3 DP=8: 每GPU仅需142/8 ≈ 17.8GB → fits 24GB ✓
→ BF16+FSDP: 再省50% → 每GPU仅需~9GB → 更安全 ✓

推理 (BF16, decode):
  Weight: 7B × 2 = 14GB → fits 24GB ✓
  KV Cache: B×S×D×2×n_layers×2bytes ≈ B×1024×4096×32×2×2 = 512MB×B
  → B=1: 512MB → 总14.5GB → fits ✓
  → B=64: 32GB → OOM! → 需GQA/MLA/Paged Attention
```

## 五、关键要点

1. **Transformer核心 = Attention(信息聚合) + MLP(特征变换) + 残差(稳定训练)**
2. **MLP占61%参数和82%计算** → 推理瓶颈是MLP而非Attention!
3. **RoPE + SwiGLU + RMSNorm + Pre-Norm** = 当前最优组件组合
4. **GQA减少87.5% KV load** → 推理加速关键
5. **MLA压缩56.9x KV** → DeepSeek-V3革命性创新
6. **CUDA RMSNorm 9x加速** → 实现层面优化很重要
7. **FlashAttention省85-97%内存** → 防OOM而非加速
8. **7B单卡推理可行** → 但需BF16 + KV cache管理(B>64时OOM)
9. **训练需ZeRO-3 DP=8** → 142GB→17.8GB per GPU
10. **组件选择影响深远** → RoPE vs Sinusoidal → 外推4K→128K+