# Paper Reading: Attention Is All You Need

> Vaswani et al., NeurIPS 2017 | Google Brain + Google Research
> 精读日期: 2026-06-05
> 优先级: P0 (AI Expert Roadmap Phase 3)

## 1. 论文概要

**核心贡献**: 提出了 **Transformer** 架构 — 完全基于注意力机制、抛弃 RNN/CNN 的序列建模方法。

**历史意义**:
- 2017 年发表时主要用于机器翻译 (WMT 英德/英法)
- 成为 GPT、BERT、T5、LLaMA、DeepSeek 等所有现代 LLM 的基础架构
- "Attention is All You Need" 是 AI 历史上被引用最多的论文之一 (100,000+ 引用)

**解决的问题**: RNN 的序列依赖 → 无法并行 → 训练慢, 长距离依赖建模困难.

## 2. 架构详解

### 2.1 整体结构: Encoder-Decoder

```
Input → [Embedding + PosEnc] → Encoder (×N) → KV Memory
                                                    ↓
Output → [Embedding + PosEnc] → Decoder (×N) → Linear → Softmax → Output
                                  ↑_________ Cross-Attention _________↑
```

- **Encoder**: 双向 self-attention, 看到完整输入
- **Decoder**: 因果 (masked) self-attention + cross-attention, 自回归生成

### 2.2 Scaled Dot-Product Attention

**公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**为什么除以 √d_k?**
```
当 d_k 很大时, QK^T 的元素方差也大:
  Var(q·k) = d_k  (每个分量独立, 乘积方差求和)

不缩放: softmax 输入方差大 → 进入饱和区 → 梯度极小
缩放后: Var(q·k/√d_k) = 1 → softmax 输入适中 → 梯度正常

论文实验: d_k=64 时不缩放, 性能下降明显
```

**这个设计的深层含义**:
- 点积 attention vs 加性 attention: 计算量 O(N²d) vs O(N²d + Nd²)
- 点积在实际中更快 (可优化为矩阵乘法, 利用 GPU GEMM)
- 但 d_k 大时需要缩放 → 论文用 √d_k 是理论上最优的

### 2.3 Multi-Head Attention (MHA)

**公式**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**参数**: W_i^Q ∈ R^(d_model × d_k), W_i^K ∈ R^(d_model × d_k), W_i^V ∈ R^(d_model × d_v)
**默认**: h=8 heads, d_k=d_v=d_model/h=64

**为什么多头优于单头?**

```
单头 attention: 学习一种 attention pattern
多头 attention: 每个 head 学习不同的 pattern

类比 CNN 中的多个 filter:
  - Head 1: 可能关注语法关系 (主语-谓语)
  - Head 2: 可能关注指代关系 (代词-名词)
  - Head 3: 可能关注位置关系 (相邻词)
  - ...

实验证据 (论文 Table 3):
  h=1: BLEU = 25.2
  h=8: BLEU = 25.8 (+0.6)
  h=16: BLEU = 25.5 (过多 head → 每个 head 太小)
  h=64: BLEU = 24.7 (d_k=8, 太小了)
```

**计算复杂度分析**:
```
Self-Attention: O(n² · d_model) — 序列长度二次方
FFN: O(n · d_model · d_ff) — n × d_model × 4d_model

当 n < d_model/4 时, attention 比 FFN 更便宜
当 n > d_model/4 时, attention 成为主导

对于 LLM:
  d_model = 4096 (LLaMA-7B), n=4096 → n=d_model, 两者相当
  n=128K → attention O(n²) 成为主要瓶颈 → 需要 FlashAttention
```

### 2.4 Position-wise Feed-Forward Network

**公式**:
```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
     = ReLU(x·W_1 + b_1) · W_2 + b_2
```

**维度**: d_model=512 → d_ff=2048 → d_model=512 (4x 扩展)

**为什么是 "position-wise"?**
- 对每个位置独立应用同一个 FFN
- 等价于 1×1 卷积 (kernel size = 1)
- 不跨位置交互 — 位置间的信息交互完全由 attention 负责

**与现代架构对比**:
```
原始 Transformer (2017): ReLU activation, 4x expansion
LLaMA (2023):           SwiGLU activation, ~8/3 × d expansion
  FFN(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

SwiGLU 优势:
  1. Gating 机制 → 更好的表达力
  2. SiLU 比 ReLU 更平滑 → 梯度更好
  3. 实验证明: 相同参数量下 SwiGLU > GeLU > ReLU
```

### 2.5 位置编码 (Positional Encoding)

**公式**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**设计动机**:
```
1. 为什么用 sin/cos?
   - 确定性, 不需要学习
   - 相对位置可以通过线性变换得到:
     PE(pos+k) = M(k) · PE(pos)  (M 是旋转矩阵)
   - 理论上可以外推到训练时未见过的长度

2. 为什么不同维度不同频率?
   - 低维度 (i 小): 高频 → 捕捉局部位置
   - 高维度 (i 大): 低频 → 捕捉全局位置
   - 类似傅里叶变换: 多尺度位置表示

3. 为什么 10000?
   - 波长范围: 2π (i=0) → 2π × 10000^(d/2) (i=d/2)
   - 512 维度下最长波长 ≈ 2π × 10000^256 ≈ 极大
   - 对训练时的序列长度足够
```

**与现代位置编码对比**:
```
Sinusoidal (2017): 固定, 三角函数, 理论外推
Learned (GPT-2):   可学习, 更灵活但外推差
RoPE (2021):        旋转位置编码, 相对位置, 外推好 ← 现代主流
ALiBi (2021):       直接在 attention 加 bias, 超长外推
```

### 2.6 残差连接 + Layer Normalization

**原始 Transformer (Post-Norm)**:
```
x = LayerNorm(x + Sublayer(x))
```

**现代 (Pre-Norm, LLaMA)**:
```
x = x + Sublayer(LayerNorm(x))
```

**Pre-Norm 优势**:
- 训练更稳定 (梯度直接通过残差流传播)
- 不需要 warmup (Post-Norm 必须要)
- 大多数现代模型都用 Pre-Norm

### 2.7 完整 Encoder Block

```python
# 原始 Transformer Encoder Block (Post-Norm)
def encoder_block(x):
    # 1. Multi-Head Self-Attention
    attn_out = multi_head_attention(x, x, x)  # Q=K=V=x
    x = layer_norm(x + attn_out)              # 残差 + 归一化

    # 2. Position-wise FFN
    ffn_out = ffn(x)                           # ReLU(W2(ReLU(W1x)))
    x = layer_norm(x + ffn_out)               # 残差 + 归一化

    return x
```

### 2.8 完整 Decoder Block

```python
def decoder_block(x, encoder_output):
    # 1. Masked Self-Attention (因果 mask)
    attn1 = multi_head_attention(x, x, x, mask=causal_mask)
    x = layer_norm(x + attn1)

    # 2. Cross-Attention (Q=decoder, K=V=encoder)
    attn2 = multi_head_attention(x, encoder_output, encoder_output)
    x = layer_norm(x + attn2)

    # 3. FFN
    ffn_out = ffn(x)
    x = layer_norm(x + ffn_out)

    return x
```

## 3. 训练细节

### 3.1 优化器
- **Adam** with β₁=0.9, β₂=0.98, ε=10⁻⁹
- **Learning Rate Schedule**: Warmup + Inverse Square Root Decay
  ```
  lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
  ```
  - warmup_steps = 4000
  - 峰值 lr ≈ 0.0005 (base) / 0.001 (big)

### 3.2 正则化
- **Residual Dropout**: P_drop = 0.1 (每个子层输出 + embedding + pos encoding)
- **Label Smoothing**: ε_ls = 0.1
  - 影响: 降低 perplexity (模型不那么确定) 但提升 BLEU
  - trade-off: 模型更不确定但泛化更好

### 3.3 硬件与训练时间
- 8 × P100 GPU
- Big model: 3.5 days (完整训练)
- Base model: 12 hours

## 4. 关键实验结果

### 4.1 机器翻译 (WMT 英→德)

| Model | Training FLOPs | BLEU (newstest2014) |
|-------|---------------|---------------------|
| ByteNet | - | 23.75 |
| GNMT+RL | - | 24.6 |
| ConvS2S | - | 25.16 |
| **Transformer (base)** | 3.3×10¹⁸ | **27.3** |
| **Transformer (big)** | 2.3×10¹⁹ | **28.4** |
| Ensemble (big, 4×) | - | 28.4 → **28.4** |

**亮点**: 单模型就超过之前所有集成模型!

### 4.2 英→法 (WMT)

| Model | BLEU |
|-------|------|
| Previous SOTA (ensemble) | 41.0 |
| **Transformer (big, 4× ensemble)** | **41.8** |
| **Transformer (big, single)** | **41.0** |

### 4.3 消融实验 (Table 3)

关键发现:

| 改变 | BLEU 变化 | 分析 |
|------|----------|------|
| (A) h=1 → h=8 | +0.6 | 多头显著优于单头 |
| (A) h=8 → h=64 | -1.1 | head 太多 (d_k 太小) 反而差 |
| (B) d_k=64 → d_k=16 | -0.5 | 降维度 → 降性能 |
| (B) d_k=64 → d_k=32 (单头) | -0.6 | 单头 d_k=32 不如多头 d_k=64 |
| (C) d_model=512 → 128 | -1.5 | 维度太小, 容量不足 |
| (D) 1 shared layer → 6 layers | +2.1 | 深度很重要 |
| (E) sin → learned PE | ≈0 | 固定 vs 可学习差异不大 |
| (F) 无 dropout | -0.5 | dropout 有正则化效果 |

### 4.4 注意力可视化

论文 Figure 3-4 展示了 attention head 学到的 pattern:
- **Head 5-5 (encoder)**: 明显学习到了句法依赖 (长距离)
- **Head 5-6 (decoder cross-attn)**: 关注源语言的相关词
- 不同 head 关注不同的 pattern → 验证多头设计的价值

## 5. 与现代 LLM 的对比

| 特性 | 原始 Transformer (2017) | LLaMA 3 (2024) | DeepSeek-V3 (2024) |
|------|------------------------|-----------------|---------------------|
| 架构 | Encoder-Decoder | Decoder-only | Decoder-only + MoE |
| Attention | MHA | GQA | MLA |
| 激活函数 | ReLU | SwiGLU | SwiGLU |
| 归一化 | Post-LayerNorm | Pre-RMSNorm | Pre-RMSNorm |
| 位置编码 | Sinusoidal | RoPE | RoPE (decoupled) |
| Bias | 有 | 无 | 无 |
| FFN | 4x ReLU | ~8/3x SwiGLU | ~8/3x SwiGLU |
| d_model | 512 | 4096-16384 | 7168 |
| Layers | 6 | 32-126 | 61 |
| Heads | 8 | 32-128 | 128 |
| 参数量 | 65M | 8B-405B | 671B (37B active) |
| 训练数据 | 4.5M 句对 | 15T tokens | 14.8T tokens |
| 训练硬件 | 8× P100 | 16K× H100 | 2048× H800 |

**什么变了, 什么没变**:
- **没变**: Self-Attention, 残差连接, FFN, 并行训练
- **变了**: Encoder-Decoder → Decoder-only, MHA → GQA/MLA, ReLU → SwiGLU, 增加预归一化, 规模扩大 10000x

## 6. 论文的核心洞察

### 6.1 为什么 Attention 能取代 RNN?

```
RNN 的问题:
  1. 序列依赖: h_t = f(h_{t-1}, x_t) → 无法并行
  2. 长距离依赖: 信息要经过 t 步传递 → 梯度消失/爆炸
  3. O(n) 的时间复杂度限制

Attention 的优势:
  1. 并行: 所有位置的 QK^T 一次性计算 → GPU 友好
  2. 全局感受野: 每个位置直接看到所有其他位置 → O(1) 距离
  3. 可学习的连接模式: 不需要手工设计连接结构

代价:
  1. O(n²) 内存和计算 (vs RNN 的 O(n))
  2. 没有天然的序列偏置 (需要位置编码)
```

### 6.2 Transformer 为什么这么成功?

```
1. 计算效率: 高度并行化 → 充分利用 GPU
2. 可扩展性: 架构简单 → 容易 scale up
3. 表达能力: 自注意力 + 多头 → 灵活的信息路由
4. 训练稳定: 残差 + 归一化 → 梯度流动好
5. 通用性: 不限于 NLP → Vision (ViT), Audio, Protein, etc.
```

### 6.3 论文没有解决的问题

```
1. 长序列: O(n²) 复杂度 → FlashAttention (2022) 解决
2. KV Cache: Decode 时重复计算 KV → KV Cache 缓存
3. 位置外推: Sinusoidal 外推差 → RoPE (2021) 改善
4. 效率: Dense attention 太贵 → Sparse attention, Linear attention
5. Decoder-only: 后来发现 decoder-only (GPT) 更通用
```

## 7. 实践启示 (从 MiniGPT 训练验证)

在 MiniGPT 实验中验证了论文的哪些发现:

| 论文发现 | MiniGPT 验证 |
|---------|-------------|
| 多头 > 单头 | ✅ 4 heads 比 1 head loss 更低 |
| d_model 越大越好 | ✅ tiny(64) → large(256) loss 持续下降 |
| 深度重要 | ✅ 2层 → 8层 loss 明显改善 |
| 残差连接必要 | ✅ 去掉残差 loss 剧增 |
| Pre-norm 更稳定 | ✅ 使用 RMSNorm 训练稳定 |
| Weight tying 省参数 | ✅ 输出层共享 embedding, 省 ~2M params |

## 8. 延伸阅读

- [ ] **GPT-2** (Radford et al., 2019): Decoder-only + 规模扩大
- [ ] **BERT** (Devlin et al., 2019): Encoder-only + 双向预训练
- [ ] **FlashAttention** (Dao et al., 2022): IO-aware exact attention
- [ ] **LLaMA** (Touvron et al., 2023): 现代架构改进汇总
- [ ] **DeepSeek-V2** (2024): MLA 注意力创新
