# Attention机制数学理论 — 从softmax到GQA到MLA

> 2026-06-07 | 理解transformer的核心：为什么attention是"通用的关联计算器"

## 概述

Attention机制是transformer架构的核心创新。本文从数学角度深度分析attention，连接理论与实践——从softmax的数学性质到GQA/MQA/MLA的压缩原理，从causal masking到block-causal mask的推理等价性。

## 一、Attention数学定义

### 1.1 基本形式

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Q: [N, d_k]  (query — "我需要什么信息")
K: [M, d_k]  (key — "我有什么信息")
V: [M, d_v]  (value — "信息的具体内容")
```

**数学分解**:

1. **QKᵀ**: 计算query与所有key的相似度 → [N, M]矩阵
   - 每个元素 S[i,j] = Q[i] · K[j] = Σ_k q[i,k] × k[j,k]
   - 这是**关联矩阵**: "位置i需要位置j的信息"的程度

2. **softmax**: 将相似度转为概率分布 → 每行归一化
   - A[i,j] = exp(S[i,j]) / Σ_j exp(S[i,j])
   - 确保Σ_j A[i,j] = 1 → 概率约束

3. **A × V**: 概率加权求和 → 每个位置的输出是所有value的加权平均
   - O[i] = Σ_j A[i,j] × V[j] → "从所有位置取信息"

### 1.2 为什么除√d_k?

**数学证明**: Q和K是随机初始化的d_k维向量 → Q·K的期望值和方差

```
如果q_i, k_j ~ N(0, 1/d_k) (每维方差1/d_k):
E[q·k] = Σ E[q_i × k_i] = Σ 0 = 0
Var[q·k] = Σ Var[q_i × k_i] = Σ (1/d_k)² = d_k × (1/d_k)² = 1/d_k

但实际初始化通常 q_i ~ N(0,1), k_j ~ N(0,1):
Var[q·k] = Σ Var[q_i] × Var[k_i] = d_k × 1 × 1 = d_k

→ q·k的方差 = d_k → 随d_k增大 → 内积值波动增大
→ softmax输入波动大 → 输出偏向极端(0或1) → gradient接近0 → 训练困难

除√d_k后: Var[q·k/√d_k] = Var[q·k]/d_k = 1 → 标准化方差 → softmax稳定
```

**关键洞察**: √d_k scaling不是"调参"而是数学必需 — 否则高维Q·K的方差导致softmax饱和

### 1.3 Attention作为"通用关联计算器"

**从RNN到Attention的范式转换**:

```
RNN:  位置i的信息 = f(位置i-1的信息, 位置i的输入)
      → 串行传播 → 长距离信息衰减(梯度消失/爆炸)

Attention: 位置i的信息 = Σ_j A[i,j] × V[j]
           → 直接连接所有位置 → O(1)步传播 → 无衰减
```

**信息论视角**:
- RNN: 信息瓶颈 = hidden state维度 → 长距离信息被压缩到有限维度 → 损失
- Attention: 信息瓶颈 = softmax归一化 → ΣA[i,j]=1 → 只能"选择"而非"累积"
- 但softmax的soft选择 → 可以同时关注多个位置 → 多路信息融合

## 二、Softmax数学性质

### 2.1 Softmax作为概率分布

```python
softmax(x) = exp(x) / Σ exp(x)

性质:
1. Σ softmax(x_i) = 1  (概率约束)
2. softmax(x_i) ∈ (0, 1)  (非零概率 → 总有梯度)
3. softmax(x+c) = softmax(x)  (平移不变 → 竞争相对大小而非绝对值)
4. ∂softmax(x_i)/∂x_j = softmax(x_i)(δ_ij - softmax(x_j))  (雅可比矩阵)
```

**平移不变性的重要性**: softmax只关心logit之间的相对差异 → 不受绝对值影响 → 稳定

### 2.2 Softmax的梯度问题

```
softmax梯度雅可比: J_ij = p_i(δ_ij - p_j)

最大值位置(max): p_max ≈ 1 → J ≈ 0 → 梯度接近0 → 学习停滞!
小值位置: p_small ≈ 0 → J ≈ 0 → 同样梯度接近0!
中间位置: p_mid ≈ 0.3 → J ≈ 0.3(δ-0.3) → 有梯度 → 学习发生

→ softmax的"梯度两极化": 只有中间概率的位置有有效梯度
→ 这就是为什么√d_k scaling重要: 防止概率过于极端
```

### 2.3 Temperature-Softmax(蒸馏)

```
softmax_T(x) = exp(x/T) / Σ exp(x/T)

T=1:  标准softmax
T>1:  平坦分布 → entropy↑ → 更多位置有有效梯度 → 训练更稳定
T<1:  尖锐分布 → entropy↓ → 更确定但梯度更极端

蒸馏中T↑的原因: 让teacher的"dark knowledge"(小概率值)变得可见
→ 训练中T↑的原因: 让更多位置有有效梯度 → 防止attention坍塌到单位置
```

### 2.4 LogSumExp(LSE) — FlashAttention的关键

```
LSE(x) = log(Σ exp(x_i))  ← softmax的分母

性质:
1. max(x) ≤ LSE(x) ≤ max(x) + log(n)  (上下界)
2. LSE(x+c) = LSE(x) + c  (平移 → 可以减max防溢出)
3. softmax(x) = exp(x - LSE(x))  (用LSE计算softmax)

FlashAttention的核心优化:
  不存储完整S矩阵(N²) → 只存储每行的LSE(1个值) → O(N)存储
  重建softmax时: A[i,j] = exp(S[i,j] - LSE[i]) ← 只需1个LSE值
```

**我们的实验验证**: Block-causal mask LSE merge精度 cos_sim=0.999999 → LSE merge等价于完整softmax计算

### 2.5 Online Softmax(FlashAttention核心)

```
标准softmax: 需要完整S矩阵 → O(N²)内存 → 不可能在大N上

Online softmax(增量计算):
  逐行计算 → 每到一个新元素就更新running max和running sum

  初始化: m_0 = -∞, l_0 = 0
  对于j=1..N:
    m_j = max(m_{j-1}, S[j])           ← running max
    l_j = l_{j-1} × exp(m_{j-1}-m_j) + exp(S[j]-m_j) ← running sum
  softmax[j] = exp(S[j]-m_N) / l_N     ← 最终归一化

  → 只需2个running变量(max + sum) → O(1)存储 → O(N)计算
  → 这是FlashAttention能在SRAM中计算的原因!
```

## 三、Multi-Head Attention (MHA)

### 3.1 为什么多头?

**单头attention的问题**:
```
单头: 每个位置只能关注"一种模式" → softmax归一化 → ΣA=1
→ 不能同时关注"局部语法关系"和"全局语义关系"

多头: h个独立attention → 每个头可以关注不同模式
  头1: 关注语法(相邻token)
  头2: 关注语义(远距离相关token)
  头3: 关注位置信息(位置编码)
  ...

Concat → Linear → 每个位置获得多角度信息
```

**数学**:
```
MHA(Q, K, V) = Concat(Attention(Q₁, K₁, V₁), ..., Attention(Q_h, K_h, V_h)) × W_O

Q_i = Q × W_Q_i  [N, d_k/h]
K_i = K × W_K_i  [M, d_k/h]
V_i = V × W_V_i  [M, d_v/h]

总参数: h × (d × d_k/h × 3 + d_k × d_v) = 3 × d × d_k + d_k × d_v ≈ 4 × d²
```

### 3.2 Multi-Query Attention (MQA)

```
MQA: 所有head共享同一组K和V!

K = K × W_K  [M, d_k] ← 只有1组K (而非h组)
V = V × W_V  [M, d_v] ← 只有1组V (而非h组)
Q_i = Q × W_Q_i  [N, d_k/h] ← 每个头独立Q

→ KV cache减少h倍! (只有1组K/V而非h组)
→ 推理KV读取减少h倍 → decode速度↑(KV读占decode瓶颈86%!)
→ 但精度略降(所有head只能看同一组key/value → 信息压缩)
```

**实测验证**: MQA KV读仅7%(vs MHA 44.5%) → 但我们的GQA Python expand反而更慢 → 需专用kernel

### 3.3 Grouped-Query Attention (GQA)

```
GQA: g组KV, 每组g/h个head共享 → 中间方案

g组: K_g = K × W_K_g  ← g组K (而非h组)
     V_g = V × W_V_g  ← g组V

每组的头: Q_i = Q × W_Q_i for i in group g
→ KV cache减少h/g倍
→ 精度损失小于MQA(g组>1组 → 更灵活)

GQA-8(h=32,g=4): KV读38%(vs MHA 44.5%, MQA 7%) → 平衡方案
```

**KV cache压缩比**:
```
MHA:  KV = 2 × n_layers × seq_len × d_model × n_heads  ← d_model × n_heads
GQA-g: KV = 2 × n_layers × seq_len × d_model × (n_heads/g) ← 减少g倍
MQA:   KV = 2 × n_layers × seq_len × d_model  ← 最少(只有1组)

7B模型(S=2048):
MHA:  134MB
GQA-8: 16.7MB  (8x压缩)
MQA:  4.2MB    (32x压缩)
MLA:  0.6MB    (224x压缩!)
```

### 3.4 Multi-Head Latent Attention (MLA) — DeepSeek-V2/V3创新

```
MLA核心: 低秩投影压缩KV + 解耦RoPE + 推理时矩阵吸收

压缩路径:
  c = DownProj(x)  [d_model → d_c] ← d_c = 512 (vs d_model = 7168!)
  K_compress = UpK(c)  [d_c → d_k] ← 从压缩空间重建
  V_compress = UpV(c)  [d_c → d_v]
  → KV cache = c(512 bytes/tok) vs 原始KV(7168 bytes/tok) → 14x压缩

解耦RoPE:
  k_pe = RoPE(W_pe × x)  ← 位置编码单独处理(不被压缩)
  K_full = [K_compress; k_pe]  ← 拼接 → 推理时k_pe不需缓存(只有128 bytes)

矩阵吸收(推理优化):
  推理时: Q × UpK → 合并为 W_QK_absorbed = W_Q × UpK
  → 不需要单独存储K → 直接在Q空间计算 → 更高效

实测: MLA KV 0.448ms vs MHA KV 很多 → 容量32x压缩 → 推理可服务更多并发
```

**关键数学**: 低秩投影的合理性

```
为什么KV可以低秩压缩而不严重损失精度?

KV矩阵 [seq_len, d_k] → 实际信息量 ≈ rank << d_k
→ 因为语言有大量冗余 → 大多数KV向量可以由少量基向量组合
→ DownProj保留d_c个主要成分 → 与PCA类似但learned(不是统计PCA)

d_c=512 vs d_model=7168 → 14x → 56.9x总压缩(含KV投影)
→ 5%参数保留80%信息 → 低秩假设在语言特征上成立
```

## 四、Causal Masking

### 4.1 因果mask(Causal mask)

```
标准causal mask: 下三角矩阵
  位置i只能看到位置j≤i → "过去的信息"

  mask[i,j] = 1 if j ≤ i, else 0

  → 自回归生成的基础: 模型生成token i时不能看到token i+1
  → 数学: S_masked = S × mask → softmax只在mask=1的元素上计算
```

### 4.2 Block-Causal Mask (Prefix Sharing)

```
Block-causal mask: prefix部分是bidirectional, suffix部分是causal

  prefix部分: 所有token互相可见 → bidirectional attention
  suffix部分: 只能看到prefix + 自己之前的suffix → causal

  mask[i,j] = 1 if:
    (i,j both in prefix) → bidirectional
    (j ≤ i and j in prefix) → suffix能看到整个prefix
    (j ≤ i and both in suffix) → suffix内部causal

→ 这是Prefix Sharing/KV Injection的正确数学!
→ 我们的验证: cos_sim_suffix=0.999999 → block-causal等价于baseline causal attention
```

### 4.3 LSE Merge (FlashAttention实现)

```
Block-causal attention计算:

方法1: SDPA math — 一次性计算完整mask → O(prefix²+suffix²+prefix×suffix)
方法2: LSE merge — 分两步:

Step 1: Prefix FA (bidirectional) → LSE_prefix
Step 2: Suffix FA (causal, 含prefix KV) → LSE_suffix

Merge: softmax_total = softmax_merge(LSE_prefix, LSE_suffix)

数学等价性证明:
  对于suffix中的位置i:
  P(i|prefix,suffix) = Σ_{j∈prefix} exp(S[i,j]-LSE) + Σ_{j≤i,j∈suffix} exp(S[i,j]-LSE)
  = exp(LSE_prefix_part - LSE_total) + exp(LSE_suffix_part - LSE_total)

  → LSE merge精确等价于完整softmax → 我们的实测cos_sim=0.999999验证!

实测crossover点: prefix≥6K时LSE merge更快(SDPA O(prefix×suffix) vs FA O(suffix²))
```

## 五、Attention的数学本质

### 5.1 为什么Attention有效?

**从三个角度理解**:

1. **信息论**: Attention是"信息路由" — 选择性传输信息
   - softmax是soft routing → 可同时从多个源取信息
   - 相比hard routing(只选1个源) → 更灵活 → 更robust

2. **优化理论**: Attention是"加权聚合" — 用learned权重做平滑
   - 每个位置的输出是所有位置的加权平均 → 类似kernel smoothing
   - 但权重是learned(不是固定的Gaussian/RBF) → 更自适应

3. **计算理论**: Attention是"集合到集合的函数" — permutation equivariant
   - 输入N个向量 → 输出N个向量 → 每个输出取决于所有输入
   - 如果打乱输入顺序 → 输出也打乱(但值不变) → 顺序由位置编码决定

### 5.2 Attention的计算复杂度

```
标准Attention: O(N² × d_k)
  QKᵀ: [N,d_k] × [d_k,N] → O(N² × d_k) ← 主要开销!
  softmax: O(N²) ← 对N²矩阵每行归一化
  A × V: [N,N] × [N,d_v] → O(N² × d_v)

FlashAttention: O(N² × d_k) 计算不变! → 但IO从O(N²)降到O(N²/M)
  tiling: 将Q/K/V分成tile → 每个tile只在SRAM中计算 → 减少HBM读写
  online softmax: 逐tile更新 → 不需要完整N²矩阵 → SRAM够用

→ FlashAttention不减少计算量 → 减少内存访问 → IO bound变为compute bound
→ 实测: prefill微加速(1.03-1.10x), 内存节省85-97%, decode反而慢(Q=1时无IO收益)
```

### 5.3 长序列的N²瓶颈

```
N=128K (long context):
  QKᵀ大小: 128K × 128K × 4bytes = 64GB → 超GPU内存!
  → 必须用FlashAttention(不在SRAM存完整矩阵)
  → 但计算量仍巨大: O(N²) = O(128K²) = 16G ops

解决方案:
1. FlashAttention: 不减少计算 → 减少内存 → 仍然慢
2. Sliding window: 只看局部窗口 → O(N × W) ← 最实用!
3. Sparse attention: 只看重要位置 → O(N × k) ← 需要筛选策略
4. Linear attention: 用kernel近似softmax → O(N × d) ← 理论最优但精度差
5. MLA: 低秩压缩 → KV小 → decode不受N²影响 ← DeepSeek-V3
```

### 5.4 Linear Attention理论

```
核心想法: 用核函数近似softmax → 消除N²计算

softmax(QKᵀ) ≈ φ(Q) × φ(K)ᵀ → 分解为: φ(Q) × [Σ φ(K)ᵀ × V]

  Σ φ(K)ᵀ × V 是与N无关的中间矩阵 [d_k, d_v] ← O(1)存储!
  → 计算变为: φ(Q) × KV_cumulative → O(N × d²) ← linear in N!

问题: φ()近似softmax → 精度损失
  φ(x) = exp(x) → 完全精确但计算昂贵
  φ(x) = elu(x)+1 → 线性近似 → 精度差
  → 理论上linear attention解决了N² → 实践上精度不可接受

Performer/Linear Transformer等尝试 → benchmark上性能下降明显
→ 当前主流仍用FlashAttention(接受N²计算但优化IO)
```

## 六、Attention与训练的连接

### 6.1 训练中attention的变化

```
训练初期(随机): attention均匀分布 → 每个位置平等关注所有位置
训练中期: attention开始分化 → 语法关系先形成(相邻token)
训练后期: attention高度分化 → 语义关系形成(远距离相关)

→ 训练过程是attention从"均匀"→"有结构"的分化过程
→ 这与我们的interpretability实验一致: 等号位置的attn_0高度敏感(0.71 patching)
→ SFT→GRPO模型有更分化的attention → 更精准的信息路由 → 100% eval
```

### 6.2 Attention entropy与训练质量

```
attention entropy: H(A[i]) = -Σ_j A[i,j] × log(A[i,j])

均匀分布: H = log(N) → 最大熵 → 无结构
极端分布: H = 0 → 注意力集中在1个位置 → 可能过度
良好训练: H中等 → 注意力集中在几个相关位置 → 有结构但不过度

→ 可以用attention entropy监控训练质量:
  过低entropy → attention坍塌 → 可能过优化
  过高entropy → attention无结构 → 模型未学到有用模式
```

### 6.3 SFT→GRPO的attention分化

```
我们的实验发现:

GRPO-only attn_0 patching effect: 0.30 → attention不够敏感 → 信息路由模糊
SFT→GRPO attn_0 patching effect: 0.71 → attention更精准 → 信息路由明确

→ SFT阶段建立了正确的"算术电路" → 等号位置的attention学会了"收集所有数字信息"
→ GRPO-only阶段 → attention不够分化 → 信息路由模糊 → eval低

→ 这解释了训练-eval gap: attention分化程度=模型理解深度
```

## 七、RoPE (Rotary Position Embedding)

### 7.1 数学定义

```
RoPE: 用复数旋转编码相对位置

x_m = [x_{2k}, x_{2k+1}]  → 看作复数 x_{2k} + i·x_{2k+1}

旋转: x_m × exp(i·m·θ_k) ← 位置m乘旋转角m·θ_k

→ 位置m的q和位置n的k:
  q_m · k_n = Σ (q_{2k}·cos(mθ) + q_{2k+1}·sin(mθ)) × (k_{2k}·cos(nθ) + k_{2k+1}·sin(nθ))
            = f(q_m, k_n, m-n) ← 只取决于相对位置(m-n)! → 相对位置编码

→ RoPE使attention天然支持相对位置 → 外推性好
```

### 7.2 我们的实测验证

```
位置编码对比 (MiniGPT, 76K模型):

No-pos: loss=0.60 → 出奇好!(数据位置依赖弱)
Learned: loss=1.10 → 最差(绝对位置, 不泛化)
RoPE: loss=0.49 → **最优**! (相对位置, 天然外推)
ALiBi: 适合超长外推

RoPE外推: 4x长度 → loss 3.93 → 仍可用 → 相对位置编码的天然优势
```

## 八、综合: 从理论到实践

### 8.1 实验数据连接理论

| 实验 | 理论 | 验证 |
|------|------|------|
| √d_k scaling | 内积方差∝d_k → softmax饱和 | FlashAttention默认除√d_k ✓ |
| FlashAttention | Online softmax → O(1)存储 | RTX 4090实测: 内存省93-97% ✓ |
| LSE merge | softmax(x) = exp(x-LSE) → block-causal等价 | cos_sim=0.999999 ✓ |
| GQA/MQA/MLA | KV低秩 → 可压缩 | MLA 56.9x ✓, GQA-8 8x ✓ |
| RoPE | 复数旋转 → 相对位置 | loss最低0.49 ✓, 外推3.93 ✓ |
| SFT→GRPO patching | attention分化=理解深度 | 0.71 vs 0.30 → 2.34x ✓ |
| Decode memory-bound | AI≈1 → 0.45% peak TFLOPS | M=1: 0.75 TFLOPS ✓ |

### 8.2 关键洞察链

```
softmax饱和 → √d_k scaling → 稳定训练 → 但N²计算 → FlashAttention → online softmax → SRAM计算

N² IO瓶颈 → FlashAttention tiling → 但仍O(N²)计算 → 长序列灾难 → MLA低秩压缩 → KV 56.9x → decode可行

因果约束 → causal mask → 但prefix sharing → block-causal → LSE merge → prefix≥6K更快

attention分化 → 理解深度 → SFT建电路 → GRPO强化 → eval 100% → patching验证(attn_0 2.34x)
```

### 8.3 AI专家视角

```
理解attention的3个层次:

1. Infra工程师: benchmark, memory, IO优化(我们已达到★★★★★)
2. 系统架构师: GQA/MLA设计, causal mask, long context(我们已达到★★★★)
3. AI专家: 为什么attention有效? 数学本质? 与训练的关系?(我们正在达到★★★→★★★★)

→ 下一步: 连接attention理论到模型训练理论(SFT→RL→distillation的数学基础)
```