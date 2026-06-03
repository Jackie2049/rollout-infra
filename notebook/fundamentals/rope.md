# Rotary Position Embedding (RoPE) 深度解析

> 旋转位置编码：为什么 LLaMA/Qwen/Mistral 都选择 RoPE 以及它如何影响推理

## 1. 传统位置编码的局限

### 1.1 绝对位置编码 (Learned / Sinusoidal)

原始 Transformer 使用绝对位置编码，直接将位置信息加到 token embedding 上：

```python
# Vaswani et al. 的正弦编码
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

x = token_emb + position_emb  # 直接相加
```

**问题**：

```
1. 无法表达相对位置关系
   - 位置 1 和位置 5 的距离 = 位置 100 和位置 104 的距离
   - 但模型只能学到绝对位置，无法直接利用 token 间的相对距离

2. 长度外推性差
   - 训练时见过 max_pos=2048，推理时遇到 position 3000
   - 学到的位置 embedding 完全没见过这些位置 → 性能急剧下降
   - 正弦编码理论上可以外推，但实际效果不好（高频分量重复）

3. 与注意力计算解耦
   - 位置信息在 attention 计算之前就融合了
   - Attention score QK^T 中位置信息是间接、模糊的
```

### 1.2 相对位置编码 (Shaw et al., T5)

直接在 attention score 中加入相对位置偏置：

```python
# T5 的相对位置 bias
attn_score = Q @ K^T + relative_position_bias[i-j]
```

**问题**：
- 需要额外的参数或 lookup table
- 通常有最大相对距离限制（T5 限制为 128）
- 无法优雅地外推到更长序列

### 1.3 核心诉求

理想的位置编码应该：

```
1. 只依赖相对位置 (m - n)，而非绝对位置
2. 随距离增加而衰减（远处的 token 权重更低）
3. 可以外推到训练时未见过的长度
4. 不增加额外参数
5. 计算高效
```

**RoPE 同时满足了以上所有条件。**

## 2. RoPE 的数学原理

### 2.1 核心思想：用旋转编码位置

RoPE 的核心洞察：**将位置信息编码为向量空间中的旋转**。

对于位置 m 的向量 x，通过旋转矩阵 R(m) 变换：

```
x'(m) = R(m) · x
```

当两个位置 m 和 n 的向量做点积时：

```
R(m) · x · R(n) · y 的内积 = x^T · R(m)^T · R(n) · y
                            = x^T · R(n - m) · y
```

**点积只依赖相对位置 (n - m)，而非绝对位置！**

### 2.2 二维情况的推导

在二维空间中，位置 m 的旋转矩阵：

```
        ┌ cos(mθ)  -sin(mθ) ┐
R(m) =  │                    │
        └ sin(mθ)   cos(mθ) ┘
```

对向量 [x₁, x₂] 应用旋转：

```
R(m) · [x₁] = [x₁ cos(mθ) - x₂ sin(mθ)]
       [x₂]   [x₁ sin(mθ) + x₂ cos(mθ)]
```

验证相对位置性质：

```
⟨R(m)x, R(n)y⟩
= [x₁cos(mθ) - x₂sin(mθ)] · [y₁cos(nθ) - y₂sin(nθ)]
+ [x₁sin(mθ) + x₂cos(mθ)] · [y₁sin(nθ) + y₂cos(nθ)]

= x₁y₁cos(mθ-nθ) + x₂y₂cos(mθ-nθ)
  + x₁y₂sin(mθ-nθ) - x₂y₁sin(mθ-nθ)

= (x₁y₁ + x₂y₂)cos((m-n)θ) + (x₁y₂ - x₂y₁)sin((m-n)θ)
```

**结果仅依赖于 (m-n)！** 这就是 RoPE 的数学基础。

### 2.3 扩展到高维 (d 维)

对于 d 维向量（d 为偶数），将其视为 d/2 个二维子空间的组合：

```
┌ R(mθ₁)    0     ...  0     ┐
│   0     R(mθ₂)   ...  0     │
│  ...     ...     ... ...    │
└  0        0      ... R(mθ_{d/2}) ┘

其中 θ_i = base^(-2i/d), base = 10000 (LLaMA 默认)
```

注意：这是一个分块对角矩阵，每个 2×2 块独立旋转。

**频率的含义**：

```
θ_0 = 10000^0     = 1.0       → 最快旋转 (每 token 旋转一个弧度)
θ_1 = 10000^(-2/d)            → 次快
...
θ_{d/2-1} = 10000^(-1) ≈ 0.0001 → 最慢旋转 (需要 ~62800 个 token 才旋转一圈)

低维度 = 高频 = 捕捉局部位置关系
高维度 = 低频 = 捕捉远程位置关系
```

### 2.4 RoPE 作用于 Attention 的完整公式

```
输入: Q, K ∈ R^{L × d}

步骤 1: 将 Q, K 按维度配对
  q = [q_0, q_1, q_2, q_3, ..., q_{d-2}, q_{d-1}]
  配对: (q_0,q_1), (q_2,q_3), ..., (q_{d-2},q_{d-1})

步骤 2: 对每对应用旋转
  q̂_{2i}   = q_{2i} cos(mθ_i) - q_{2i+1} sin(mθ_i)
  q̂_{2i+1} = q_{2i} sin(mθ_i) + q_{2i+1} cos(mθ_i)

步骤 3: Attention score
  score(m,n) = q̂(m) · k̂(n) = f(q, k, m-n)  ← 仅依赖相对位置
```

## 3. 为什么 RoPE 能做长度外推和相对位置编码

### 3.1 相对位置编码

上一节已证明：⟨R(m)q, R(n)k⟩ 仅依赖 (m-n)。

### 3.2 远距离衰减

RoPE 的 attention score 可以分解为：

```
score(m,n) = Σ_i [q_{2i}k_{2i} + q_{2i+1}k_{2i+1}] cos((m-n)θ_i)
           + Σ_i [q_{2i+1}k_{2i} - q_{2i}k_{2i+1}] sin((m-n)θ_i)
```

当 |m-n| 增大时，不同频率的 cos/sin 项以不同周期振荡，整体呈现**衰减趋势**（因为不同频率的振荡会相互抵消）。这与自然语言中"远距离 token 关联较弱"的先验一致。

### 3.3 长度外推

RoPE 的外推性来源于：

```
1. 纯数学函数编码：位置信息由 cos/sin 函数产生，不需要 lookup table
   - 不存在"训练时没见过的位置"问题

2. 相对位置性质：attention 只看 (m-n)，不关心绝对位置

3. 但有实际限制：
   - 高频分量 (θ_i 接近 1)：旋转很快，外推无问题
   - 低频分量 (θ_i 接近 0)：训练时可能只见过 θ_i × L_train < 2π
     → 超出训练长度后，低频分量进入未见过的新相位 → 性能下降

4. 解决方案：RoPE Scaling（见第 5 节）
```

## 4. 代码实现

### 4.1 Naive 实现：构造旋转矩阵

```python
import torch
import math

def build_rope_cache(seq_len, dim, base=10000):
    """构建 RoPE 的 cos/sin 缓存"""
    # inv_freq: [dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # positions: [seq_len]
    positions = torch.arange(seq_len, dtype=torch.float32)

    # freqs: [seq_len, dim/2]  — 每个位置在每个频率上的弧度
    freqs = torch.outer(positions, inv_freq)

    # emb: [seq_len, dim] — 复制一份，对应 x1 和 x2 的旋转
    emb = torch.cat((freqs, freqs), dim=-1)

    cos_cache = emb.cos()  # [seq_len, dim]
    sin_cache = emb.sin()  # [seq_len, dim]
    return cos_cache, sin_cache


def rotate_half(x):
    """将 x 的后半部分取负并与前半部分交换

    输入:  [x1, x2, x3, x4, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
    输出:  [-x_{d/2+1}, ..., -x_d, x1, x2, ..., x_{d/2}]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos_cache, sin_cache):
    """对输入张量应用 RoPE

    x: [seq_len, batch, heads, dim] 或 [seq_len, dim]
    cos_cache, sin_cache: [seq_len, dim]
    """
    # 核心: R(θ)·x = x * cos(θ) + rotate_half(x) * sin(θ)
    return x * cos_cache + rotate_half(x) * sin_cache
```

### 4.2 为什么 rotate_half 等效于旋转矩阵

```
旋转矩阵作用于二维向量 [a, b]:
  R(θ)·[a] = [a·cos(θ) - b·sin(θ)]
        [b]   [a·sin(θ) + b·cos(θ)]

用 cos/sin 和 rotate_half 表示:
  [a] * cos(θ) + [-b] * sin(θ) = [a·cos(θ) - b·sin(θ)]  ✓
  [b]             [ a]           [a·sin(θ) + b·cos(θ)]  ✓

其中 rotate_half([a, b]) = [-b, a]

推广到 d 维（d/2 个二维子空间）：
  x = [a₁, b₁, a₂, b₂, ..., a_{d/2}, b_{d/2}]
  拆分为前半后半: x1 = [a₁, a₂, ..., a_{d/2}], x2 = [b₁, b₂, ..., b_{d/2}]
  rotate_half(x) = [-x2, x1] = [-b₁, -b₂, ..., -b_{d/2}, a₁, a₂, ..., a_{d/2}]

  x * cos + rotate_half(x) * sin
  = [a₁·cos₁, b₁·cos₁, ..., a_{d/2}·cos_{d/2}, b_{d/2}·cos_{d/2}]
  + [-b₁·sin₁, -b₂·sin₂, ..., a₁·sin₁, a₂·sin₂, ...]  (错位对齐后)
  = [a₁cos₁-b₁sin₁, b₁cos₁+a₁sin₁, ...]  ← 等效于各子空间独立旋转
```

### 4.3 Complex Number 实现（更优雅）

```python
def apply_rope_complex(x, freqs):
    """使用复数运算实现 RoPE

    x: [seq_len, batch, heads, dim]
    freqs: [seq_len, 1, 1, dim/2] — 旋转角度
    """
    # 将相邻维度视为复数: (x[2i], x[2i+1]) → x[2i] + j·x[2i+1]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # freqs → e^{j·θ}
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    # 复数乘法 = 旋转
    x_rotated = x_complex * freqs_complex

    # 转回实数
    return torch.view_as_real(x_rotated).flatten(-2)
```

复数乘法的几何意义就是旋转：`(a + jb) × e^{jθ} = (a+jb)(cos θ + j sin θ)`。

### 4.4 HuggingFace Transformers 生产实现 (LLaMA)

来源: `transformers/src/transformers/models/llama/modeling_llama.py`

```python
class LlamaRotaryEmbedding(nn.Module):
    """HuggingFace 中 LLaMA 的 RoPE 实现 — 工业界参考标准"""

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        """计算默认 RoPE 参数 (inv_freq)"""
        base = config.rope_theta  # 默认 10000
        dim = config.hidden_size // config.num_attention_heads  # head_dim

        # inv_freq: [dim/2]，每维的角频率
        # θ_i = 1 / base^(2i/dim) = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        ))
        return inv_freq, 1.0  # (inv_freq, attention_scaling)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """计算 cos/sin

        x: 用于获取 dtype 和 device
        position_ids: [batch, seq_len]，每个 token 的位置索引
        """
        # inv_freq: [dim/2] → [1, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)

        # position_ids: [batch, seq_len] → [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # freqs: [batch, dim/2, seq_len] → [batch, seq_len, dim/2]
        # freqs[b, s, d] = inv_freq[d] × position_ids[b, s]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)

        # 复制: [batch, seq_len, dim/2] → [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling  # YaRN 的 mscale 在这里
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """应用 RoPE 到 Q 和 K

    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim]
    unsqueeze_dim=1: 在 head 维度扩展 → [batch, 1, seq_len, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**在 LlamaAttention.forward() 中的集成**：

```python
def forward(self, hidden_states, position_embeddings=None, ...):
    # 1. 线性投影 Q, K, V
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # 2. 解包 position_embeddings (在 LlamaModel 中一次性计算，所有层共享)
    cos, sin = position_embeddings

    # 3. 应用 RoPE（只对 Q 和 K，不对 V）
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 4. 更新 KV Cache（RoPE 已应用，缓存的是旋转后的 K 和 V）
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # 5. 标准 Attention 计算
    attn_output = torch.nn.functional.scaled_dot_product_attention(...)
```

关键设计细节：

```
1. position_embeddings 计算一次，所有层共享
   - LlamaModel.forward() 中调用 rotary_emb() 一次
   - 结果作为 position_embeddings 参数传给每一层的 attention
   - 节省重复计算

2. 支持 rope_type 注册表机制
   - ROPE_INIT_FUNCTIONS 注册表支持多种初始化:
     "default", "linear", "dynamic", "yarn", "longrope", "llama3"
   - 通过 config.rope_scaling.rope_type 选择

3. position_ids 的来源
   - Prefill: position_ids = [0, 1, 2, ..., seq_len-1]
   - Decode:  position_ids = [past_seen_tokens]  (单个位置)
   - 支持 flash_attention_2 和 sdpa 两种后端
```

### 4.5 Megatron-LM 的实际实现

来源: `Megatron-LM/megatron/core/models/common/embeddings/rope_utils.py`

```python
def _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=False, mscale=1.0):
    """Megatron-LM 中 RoPE 的应用函数

    t: [seq_len, batch, heads, dim]
    freqs: [seq_len, 1, 1, dim]  — 角度（cat 了两次）
    """
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]  # 只对部分维度应用 RoPE

    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    # 核心公式: x * cos + rotate_half(x) * sin
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)
```

关键细节：

```
1. rotary_percent < 1.0 时，只有部分 head 维度应用 RoPE
   - LLaMA: rotary_percent = 1.0 (全部维度)
   - GPT-NeoX: rotary_percent = 0.25 (只旋转前 25% 维度)

2. 两种排列方式:
   - rotary_interleaved=False (LLaMA 风格): [x₁,x₂,...,x_d] → 前半后半配对
   - rotary_interleaved=True (GPT-NeoX 风格): 相邻元素配对 (x₀,x₁), (x₂,x₃), ...
```

### 4.5 Interleaved vs Half-split 配对

```
Half-split (LLaMA, rotary_interleaved=False):
  x = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]
  x1 = [x₁, x₂, x₃, x₄]   x2 = [x₅, x₆, x₇, x₈]
  rotate_half = [-x₅, -x₆, -x₇, -x₈, x₁, x₂, x₃, x₄]

Interleaved (GPT-NeoX, rotary_interleaved=True):
  x = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]
  奇偶配对: (x₁,x₂), (x₃,x₄), (x₅,x₆), (x₇,x₈)
  rotate_half = [-x₂, x₁, -x₄, x₃, -x₆, x₅, -x₈, x₇]
```

## 5. RoPE Scaling 方法

当需要推理超过训练长度的序列时，直接用 RoPE 会导致低频分量进入未见过的相位区间，性能下降。以下是主要解决方案。

### 5.1 Position Interpolation (PI)

```
思路：不外推，而是内插。
将位置 [0, L_new] 压缩到 [0, L_train]：

  m' = m × (L_train / L_new)

freqs = outer(m × (L_train / L_new), inv_freq)

优点: 简单，不会进入未见过的相位
缺点: 所有频率均匀缩放 → 高频变低频 → 丢失局部位置分辨率
      → 短距离的区分度下降
```

Megatron-LM 中的 `seq_len_interpolation_factor` 参数就是 PI：

```python
# rotary_pos_embedding.py
if self.seq_len_interpolation_factor is not None:
    seq *= 1 / self.seq_len_interpolation_factor  # 压缩位置
```

### 5.2 NTK-aware Scaling

```
思路：不缩放位置，而是缩放频率基底 (base)。

  base_new = base × s^(d/(d-2))    (s = 缩放因子)

等效于: 让频率分量更密（低频更低），不改变高频。
```

**β进制编码视角（苏剑林的深刻洞察）**：

```
RoPE 的本质是位置 n 的 β 进制编码，其中 β = 10000^(2/d):

  RoPE 第 i 维: cos(n / β^i), sin(n / β^i)

  类比 10 进制第 m 位: floor(n / 10^(m-1)) mod 10

两者都有 n / β^(m-1) 的结构！cos/sin 的周期性等价于模运算。

NTK-aware 的本质是进制转换:
  - 3 位 10 进制 → 最大 999
  - 3 位 16 进制 → 最大 4095

  要扩大 k 倍范围: β_new = β × k^(2/d)
  即 base_new = 10000 × k ≈ base × s^(2/d)

  进制转换不改变序关系（875 > 874 在 16 进制下仍然成立），
  所以 NTK-aware 可以不微调就外推！
```

**高频外推、低频内插的统一解释**：

```
提出者 @bloc97 的推导:
  最低频项: n / (β·λ)^(d/2-1) → 设为内插 → λ = s^(2/(d-2))
  最高频项: n / (β·λ) → λ ≈ 1 (因为 d 很大) → 等效于外推

所以 NTK-aware 自然实现了: 高频外推 + 低频内插
任何能实现这种模式的方案都有效，不限于 base 缩放。
```

直觉:
```
  - 高频分量 (θ 接近 1): 不太受影响，保留局部位置精度
  - 低频分量 (θ 接近 0): 被显著缩放，使其在更长序列上才完成一个周期
  - 避免了 PI 的高频退化问题
  - 实测: 不微调时从 512 外推到 4096，准确率从 24%（PI）升到 51%（NTK-aware）
```

### 5.3 Dynamic NTK

```
NTK-aware 的改进：动态调整 base。

  当 current_seq_len ≤ L_train 时: 使用原始 base
  当 current_seq_len > L_train 时:
    base_new = base × (current_seq_len / L_train)^(d/(d-2))

逐 token 动态计算，无需预先设定目标长度。
```

**推理时行为**：

```
Autoregressive 生成中，序列长度逐 token 增长:

  token 1:    seq_len=1    → s = max(1, 1/L) = 1       → 原始 base
  token 2048: seq_len=2048 → s = max(1, 2048/2048) = 1 → 原始 base
  token 2049: seq_len=2049 → s = 2049/2048 ≈ 1.0005    → 微调 base
  ...
  token 8192: seq_len=8192 → s = 8192/2048 = 4.0       → base × 4^(d/(d-2))

特点:
  1. 在训练长度内无任何退化（s=1, 使用原始 base）
  2. 超出训练长度后，graceful degradation 而非突然崩溃
  3. 不需要预设目标长度

重要实现细节（来自 YaRN 论文 §3.4）:
  KV Cache + Dynamic NTK 时，应缓存 RoPE 应用前的 K
  因为每个 token 的 RoPE 参数随 s 变化而变化
  如果缓存了已旋转的 K，后续 token 的 s 变化会导致不一致
```

采用者: Code Llama (NTK-aware, base=1M)、Qwen 7B (Dynamic NTK)

### 5.4 YaRN (Yet another RoPE extensioN)

YaRN 是目前效果最好的方法之一（ICLR 2024），比 PI 少 10x tokens、2.5x 训练步数。
被 Mistral、Megatron-LM 等采用。

**核心思想**：区分对待不同频率维度。

**关键概念 — 波长 (wavelength)**：

```
第 d 维的波长: λ_d = 2π / θ_d = 2π × base^(2d/|D|)

波长 = RoPE 在该维度完成一个完整旋转所需的 token 数

以 LLaMA-7B 为例 (base=10000, dim=128):
  λ_0  = 2π × 1        ≈ 6.28      tokens  (最高频，每 ~6 个 token 旋转一圈)
  λ_32 = 2π × 100      ≈ 628       tokens
  λ_64 = 2π × 10000    ≈ 62832     tokens  (最低频，超过训练长度)
  λ_128= 2π × 100000000 ≈ 6.28亿   tokens  (几乎不旋转)

关键洞察:
  - 如果 λ_d > L_train，该维度在训练期间从未完成完整旋转
    → 包含绝对位置信息（每个 token 到起点的距离唯一）
    → 必须内插，否则外推到新距离会 out-of-distribution
  - 如果 λ_d << L_train，该维度已旋转多圈
    → 只包含相对位置信息
    → 不应修改，否则破坏局部精度
```

**NTK-by-parts 插值**：

```
引入旋转比率: r(d) = L_train / λ_d = L_train / (2π × base^(2d/|D|))
  r(d) = 该维度在训练期间旋转了几圈

两个边界参数 α, β (LLaMA 推荐: α=1, β=32):

  r(d) < α  (低频，旋转 <1 圈):  h(θ_d) = θ_d / s      (PI 缩放，避免外推)
  r(d) > β  (高频，旋转 >32 圈): h(θ_d) = θ_d           (不修改)
  α ≤ r(d) ≤ β  (中频):          线性过渡

  h(θ_d) = (1-γ(r(d))) × θ_d/s + γ(r(d)) × θ_d

  其中 γ(r) = (r - α) / (β - α)  (线性 ramp 函数)
```

**YaRN = NTK-by-parts + Attention Temperature 缩放**：

```
YaRN 观察到: 扩展上下文后 attention 分布变化，entropy 增加
解决方案: 在 attention softmax 中引入温度 t

  attention = softmax(Q'·K'^T / (t × √d))

但直接改 attention 代码不兼容 Flash Attention。
YaRN 的巧妙做法: 缩放 cos/sin 的幅度 = 等效于缩放 Q 和 K

  cos_scaled = cos × √(1/t)
  sin_scaled = sin × √(1/t)

推荐公式: √(1/t) = 0.1 × ln(s) + 1.0

  s=8  → √(1/t) ≈ 1.208
  s=16 → √(1/t) ≈ 1.277
  s=32 → √(1/t) ≈ 1.346

实测在 LLaMA 7b/13b/33b/65b 和 Llama 2 7b/13b/70b 上该公式均适用
→ 说明 entropy 增加的模式具有"普适性"
```

**YaRN 的实验结果**：

```
训练效率（A100 GPU-hours）:
  YaRN 7B 2k→32k:    128 小时 (400 steps)
  PI   7B 2k→16k:    640 小时 (同等质量)
  NTK  7B 4k→50k:  64000 小时

Perplexity（Proof-pile 数据集, 128k tokens）:
  Together PI 32k:     3.50 (8k) | >100 (128k, 崩溃)
  Code Llama NTK 100k: 3.71 (8k) | 2.71 (128k)
  YaRN 128k:           3.56 (8k) | 2.37 (128k) ← 最优

Passkey Retrieval 准确率:
  YaRN 128k: 99.4% (7B), 99.4% (13B)

YaRN 还能 "train short, test long":
  用 64k 数据训练 → 成功外推到 128k
```

Megatron-LM 的 YaRN 实现（`yarn_rotary_pos_embedding.py`）：

```python
def get_emb(self, max_seq_len, offset=0):
    # 计算修正范围
    low, high = _yarn_find_correction_range(
        beta_fast=32, beta_slow=1, dim=self.dim,
        rotary_base=self.rotary_base,
        max_position_embeddings=self.original_max_position_embeddings,
    )
    # 构建掩码: 低频部分为 1，高频部分为 0，中频线性过渡
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2)

    # 低频用缩放后的频率，高频用原始频率，中频混合
    inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

    # Attention temperature 修正 (mscale)
    _mscale = _yarn_get_mscale(self.scaling_factor)
```

**Attention Temperature 修正 (mscale)**：

```
YaRN 的关键额外步骤: 缩放 attention softmax 的 temperature。

直觉: 长序列外推时，attention 分布可能变得过于均匀或过于尖锐。
mscale 修正 cos/sin 的幅度，间接调整 attention score 的分布。

mscale = 0.1 × log(s) + 1.0   (s = scaling_factor)
```

### 5.5 LLaMA 3.x 的 RoPE Scaling

LLaMA 3.x 使用了一种平滑的频率缩放方案（`rotary_pos_embedding.py` 中的 `_apply_scaling`）：

```python
def _apply_scaling(self, freqs, factor=8, low_freq_factor=1, high_freq_factor=4,
                   original_max_position_embeddings=8192):
    wavelen = 2π / freqs
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # 高频 (短波长): 保持不变
    # 低频 (长波长): 除以 factor (缩放)
    # 中频: 平滑过渡
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)

    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    inv_freq_llama = torch.where(is_medium_freq, smoothed, inv_freq_llama)
```

### 5.6 方法对比

| 方法 | 原理 | 局部精度 | 长距外推 | 复杂度 | 典型使用者 |
|------|------|---------|---------|--------|-----------|
| Position Interpolation | 压缩位置 | 下降 | 一般 | 最低 | Meta (原始 PI 论文) |
| NTK-aware | 缩放 base | 保留 | 较好 | 低 | |
| Dynamic NTK | 动态 base | 保留 | 好 | 中 | Code Llama, Qwen |
| YaRN | 分频率处理 + mscale | 保留 | 最好 | 中 | Mistral, Megatron-LM |
| LLaMA 3.x Scaling | 平滑频率过渡 | 保留 | 好 | 中 | LLaMA 3 |

## 6. RoPE 与 Attention 的交互

### 6.1 为什么只作用于 Q 和 K，不作用于 V

```
Attention 计算:
  score = softmax(Q' · K'^T / √d)    ← 位置信息在这里编码
  output = score · V                   ← V 不需要位置信息

原因:
1. RoPE 的目标是在 QK 点积中引入相对位置信息
   - 位置信息影响的是"谁关注谁"（attention weight）
   - 不影响"关注的内容是什么"（value）

2. 数学推导要求:
   - ⟨R(m)q_m, R(n)k_n⟩ = g(q, k, m-n)  ← 只依赖相对位置
   - 如果对 V 也旋转: score · R(n)V → 输出被旋转 → 失去语义一致性
     不同位置的 V 被不同角度旋转后加权平均，语义被破坏

3. 效率考虑:
   - 少旋转一个矩阵 = 少一半的 RoPE 计算开销
   - KV Cache 中不需要额外存储旋转角度信息
```

### 6.2 RoPE 在 Attention 计算流中的位置

```
标准 LLaMA Decoder Layer:
  x → RMSNorm → QKV Projection → [Q, K, V]
                                      │
                            RoPE 应用于 Q 和 K（不作用于 V）
                                      │
                                      ↓
                              Attention(Q', K', V)
                                      │
                                      ↓
                              O Projection → Residual
```

### 6.3 RoPE 对 KV Cache 的影响

```
Prefill 阶段:
  - 对所有 prompt tokens 的 Q 和 K 一次性应用 RoPE
  - 计算量: O(L × d)，L = prompt 长度

Decode 阶段 (逐 token 生成):
  - 只对新生成 token 的 Q 和 K 应用 RoPE
  - KV Cache 中存的是已经旋转过的 K 和 V
  - 不需要重新计算之前位置的 RoPE！

  关键: cos/sin 是位置索引的确定性函数
        存入 KV Cache 的 K 已经是 R(m)·K_m
        新 token 的 Q 是 R(n)·Q_n
        直接计算 QK^T 即得到正确的相对位置编码
```

### 6.4 RoPE 与 Grouped Query Attention (GQA) 的兼容性

```
GQA: 多个 Q head 共享一组 K/V head

RoPE 在 GQA 下的行为:
  - 每个 Q head 独立应用 RoPE
  - 每个 K/V head 也独立应用 RoPE
  - RoPE 是逐 head 的逐元素操作，与 head 间共享完全兼容
  - 无需特殊处理
```

## 7. RoPE 在主流模型中的应用

### 7.1 模型概览

| 模型 | RoPE Base | Head Dim | 上下文长度 | Scaling 方法 | 特殊之处 |
|------|----------|----------|-----------|-------------|---------|
| LLaMA / LLaMA 2 | 10000 | 128 | 2k / 4k | 无 | RoPE 推广的里程碑 |
| LLaMA 3 | 500000 | 128 | 8k→128k | LLaMA 3 Scaling | 大幅提高 base，低高频分区缩放 |
| Qwen 7B | 10000 | 128 | 8k→32k | Dynamic NTK | 推理时动态调整 |
| Qwen 2 | varying | 128 | 32k→128k | YaRN / Dynamic | 多种 scaling 策略 |
| Mistral 7B | 10000 | 128 | 8k→32k | Sliding Window + RoPE | 与 SWA 配合使用 |
| Mixtral 8x7B | 10000 | 128 | 32k | 无 | MoE + RoPE |
| Code Llama | 1000000 | 128 | 16k→100k | NTK-aware (ABF) | base 调至 1M |
| DeepSeek-V2 | 10000 | 192 | 128k | YaRN | MLA + RoPE |
| GPT-NeoX | 10000 | varying | 2k | 无 | rotary_percent=0.25 |
| PaLM | 10000 | 256 | varying | 无 | 大 head dim |

### 7.2 各模型 RoPE 实现差异

```
维度配对方式:
  LLaMA 系列:  half-split (前半/后半配对)
               [x₁,...,x_{d/2}] 和 [x_{d/2+1},...,x_d] 配对
  GPT-NeoX:    interleaved (相邻元素配对)
               (x₀,x₁), (x₂,x₃), ... 配对

RoPE 应用比例:
  LLaMA:  100% (rotary_percent = 1.0)
  GPT-NeoX: 25% (rotary_percent = 0.25，只旋转前 1/4 维度)

Base 频率:
  标准: 10000 (大多数模型)
  LLaMA 3: 500000 (大 50 倍，支持更长上下文)
  Code Llama: 1000000 (大 100 倍，NTK-aware ABF)
```

### 7.3 RoPE 与 Prefix Caching 的交互

```
Prefix Caching: 缓存共享前缀的 KV Cache，避免重复计算

与 RoPE 的兼容性:
  ✅ 完全兼容: KV Cache 中存储的是已旋转的 K
     → 不同请求的相同前缀使用相同的 position_ids [0, 1, ..., prefix_len-1]
     → RoPE 结果确定性: 同一文本 + 同一 position → 同一旋转后 K
     → Prefix Cache 可直接复用

  ⚠️ 注意事项:
     1. position_ids 必须一致
        - 前缀 tokens 必须从 position 0 开始连续编号
        - 如果前缀在不同请求中有不同位置偏移，缓存失效

     2. Dynamic NTK 与 Prefix Caching 冲突
        - Dynamic NTK 的 s 随序列长度变化
        - 前缀 token 的 RoPE 参数在后续 token 增加时会改变
        - 解决: 缓存 RoPE 应用前的 K（而非旋转后的 K）
          但这会增加每次 decode 的 RoPE 重新计算开销

     3. vLLM 的处理方式
        - 默认使用固定 RoPE 参数（非 Dynamic）
        - Prefix Cache 在 prefill 阶段计算并缓存旋转后的 KV
        - 后续请求直接复用，无需重算 RoPE
```

### 7.4 RoPE 在 vLLM 中的实现要点

```
vLLM 使用 FlashAttention 后端的 RoPE:

1. Prefill 阶段:
   - 计算完整 prompt 的 position_ids
   - 一次性对所有 Q, K 应用 RoPE
   - 将旋转后的 K, V 存入 KV Cache (block manager 管理)

2. Decode 阶段:
   - 只计算新 token 的 position_id = past_seen_tokens
   - 只对 1 个 token 的 Q, K 应用 RoPE
   - 新 K, V 追加到 KV Cache

3. PagedAttention 下的 RoPE:
   - KV Cache 按物理 block 存储（不保证逻辑连续）
   - position_ids 在逻辑层面计算（与物理 block 无关）
   - RoPE 应用在 block 写入之前

4. Chunked Prefill:
   - 长 prompt 分块处理
   - 每块的 position_ids 连续但起始位置不同
   - RoPE 参数按块独立计算
```

## 8. 推理性能影响

### 7.1 计算开销分析

```
RoPE 的计算量:
  - 每个 token: 2 × d 次乘法 + 2 × d 次加法 (对 Q 和 K)
  - 相比 QKV projection (2 × 3d² FLOPs): RoPE 开销可忽略
  - RoPE FLOPs = O(d), Projection FLOPs = O(d²)
  - RoPE/Projection ≈ 2/(3d) ≈ 0.5% (d=128)

显存开销:
  - cos/sin 缓存: [max_seq_len, dim] × 2 × sizeof(float16)
  - 例: max_seq_len=8192, dim=128 → 2 × 8192 × 128 × 2B = 4MB
  - 完全可忽略（相比 KV Cache 的 GB 级别）
```

### 7.2 推理时的额外开销

```
Prefill:
  - RoPE 应用于整个 prompt: O(L × d) FLOPs
  - 可以预计算 cos/sin，无额外开销
  - 实测: RoPE 占 prefill 时间 < 1%

Decode (per token):
  - RoPE 只应用于 1 个 token 的 Q 和 K: O(d) FLOPs
  - 开销趋近于零
```

### 7.3 融合优化

Megatron-LM 提供了两种 RoPE 融合路径：

```
1. Transformer Engine 融合 (fused_apply_rotary_pos_emb):
   - 将 RoPE 融入 TE 的 fused attention kernel
   - 避免 cos/sin 的显式物化和中间 tensor 读写
   - 适合 BSHD 格式

2. Triton 融合 (fused_mla_yarn_rope_apply.py):
   - 专为 MLA + YaRN RoPE 设计
   - 将 KV 分割 + RoPE 应用合并为单个 kernel
   - 减少 HBM 访问次数
   - 支持 SBHD 和 THD (packed) 格式
```

Triton 融合 kernel 的核心逻辑：

```python
# 前向: 一次性完成 RoPE 旋转
x_1 = load(x[0::2])  # 偶数维度
x_2 = load(x[1::2])  # 奇数维度
x_left  = x_1 * cos_left - x_2 * sin_left   # 左半部分
x_right = x_2 * cos_right + x_1 * sin_right  # 右半部分
store(x_left, x_right)
```

### 7.4 RoPE vs 其他位置编码的开销对比

| 位置编码方案 | 额外参数量 | 额外 FLOPs | 额外显存 | 对推理延迟影响 |
|-------------|-----------|-----------|---------|-------------|
| Learned Absolute | max_seq × d | 0 | max_seq × d | 近乎为零 |
| ALiBi | 0 | O(L²) (bias) | O(L²) | 小（纯 bias） |
| RoPE | 0 | O(L × d) | cos/sin 缓存 | 近乎为零 |
| RoPE + YaRN | 0 | O(L × d) + mscale | cos/sin 缓存 | 近乎为零 |

**结论：RoPE 的推理开销可以忽略不计。** 它不需要额外参数，不增加 KV Cache 大小，每次 decode 只需 O(d) FLOPs。

## 9. 总结要点

1. **RoPE 的本质**：将位置编码为向量空间中的旋转，使得 QK 点积天然编码相对位置
2. **核心公式**：`R(x) = x * cos(θ) + rotate_half(x) * sin(θ)`，简洁且高效
3. **只作用于 Q 和 K**：位置信息决定"谁关注谁"，不改变"关注什么"
4. **长度外推**：纯函数编码无上限，但低频分量需通过 scaling 方法处理
5. **推理零开销**：计算量 O(d) vs 投影层 O(d²)，可忽略；且 KV Cache 中存储已旋转的 K
6. **Scaling 方法演进**：PI → NTK-aware → Dynamic NTK → YaRN，核心思想都是保护高频分量
7. **工业界标配**：LLaMA、Qwen、Mistral、DeepSeek 等主流 LLM 全部采用 RoPE

## 10. 参考

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) — RoPE 原论文
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023) — Position Interpolation
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (Peng et al., 2024, ICLR 2024) — YaRN, 含 NTK-aware / Dynamic NTK / NTK-by-parts 完整描述
- [NTK-aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) (Reddit @bloc97, 2023) — NTK-aware 首次提出
- [苏剑林博客: RoPE 详解](https://kexue.fm/archives/8265) — 原作者的完整数学推导
- [苏剑林博客: RoPE 是 β 进制编码](https://kexue.fm/archives/9675) — β 进制视角理解 NTK-aware / PI
- [EleutherAI Blog: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) — 英文最佳入门教程
- [Code Llama](https://arxiv.org/abs/2308.12950) — NTK-aware (ABF, base=1M) 的采用
- [HuggingFace Transformers LLaMA](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 生产级 RoPE 实现
- [Megatron-LM RoPE 实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py)
- [FlashAttention RoPE 实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py)
