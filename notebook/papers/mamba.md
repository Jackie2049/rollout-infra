# Paper Reading: Mamba — Linear-Time Sequence Modeling with Selective State Spaces

> Gu & Dao, 2023 | arXiv: 2312.00752
> 精读日期: 2026-06-05
> 优先级: P1 (Transformer 替代架构的重要尝试)

## 1. 论文概要

**核心创新**: 让 SSM (State Space Model) 的参数成为输入的函数 → **选择性状态空间模型**
- 打破 LTI (线性时不变) 约束
- 模型可以区分 "记住什么" vs "遗忘什么"
- 保持线性时间复杂度 O(L)

**影响**: 第一个在语言建模质量上匹配 Transformer 的线性时间模型

## 2. 从标准 SSM 到选择性 SSM

### 2.1 连续时间 SSM

```
h'(t) = A h(t) + B x(t)       (状态演化)
y(t)  = C h(t)                 (输出生成)
```

### 2.2 离散化 (ZOH)

```
A_bar = exp(Δ * A)
B_bar = (Δ * A)^{-1} * (exp(Δ * A) - I) * Δ * B

递归形式:
  h_t = A_bar * h_{t-1} + B_bar * x_t
  y_t = C * h_t
```

### 2.3 标准 SSM vs 选择性 SSM

```
参数对比:
            S4 (标准)         Mamba (选择性)
  A        (D, N) 固定       (D, N) 固定
  B        (D, N) 固定       (B, L, N) 输入相关!
  C        (D, N) 固定       (B, L, N) 输入相关!
  Δ        (D) 固定          (B, L, D) 输入相关!

关键区别: B, C, Δ 是 x 的函数
  B = Linear_N(x)          — 输入是否进入状态
  C = Linear_N(x)          — 状态是否进入输出
  Δ = softplus(param + Linear_1(x))  — 重置 vs 保持
```

## 3. 选择机制的物理解释

### 3.1 Δ (最重要的参数)

```
大 Δ → 重置状态, 关注当前输入 ("选择" 它)
小 Δ → 保持状态, 忽略当前输入

本质: RNN 门控的推广

Theorem 1: 当 N=1, A=-1, B=1 时:
  g_t = σ(Linear(x_t))
  h_t = (1 - g_t) * h_{t-1} + g_t * x_t
  → 等价于经典 RNN 门控!
```

### 3.2 三大效果

```
1. 可变间距 (Variable Spacing):
   过滤无关噪声 token (语言中的填充词)

2. 上下文过滤 (Filtering Context):
   在任何时刻重置状态, 移除无关历史

3. 边界重置 (Boundary Resetting):
   拼接多条序列时避免信息泄漏
```

## 4. 硬件感知实现

### 4.1 为什么不能用卷积?

```
标准 SSM (LTI): 可以用 FFT 卷积 → O(BLD log L)
选择性 SSM (非 LTI): 每个时间步参数不同 → 只能用递归

朴素递归: O(BLDN) FLOPs, 但 Python 循环太慢
解决方案: 并行扫描 (Parallel Scan)
```

### 4.2 三大优化

```
1. 内核融合 (Kernel Fusion):
   不在 HBM 中准备中间结果 (A_bar, B_bar)
   直接从 HBM 加载参数到 SRAM, 在 SRAM 中计算
   只写最终输出回 HBM → 减少 IO

2. 并行扫描 (Parallel Scan):
   O(N log N) 复杂度
   非线性递归仍可并行化

3. 重计算 (Recomputation):
   反向传播不存储中间状态
   在 backward 时从输入重新计算
   内存需求 = FlashAttention 的 Transformer
```

## 5. 性能对比

### 5.1 语言建模 (Pile)

| 模型 | 参数量 | PPL | LAMBADA | HellaSwag |
|------|--------|-----|---------|-----------|
| Pythia-2.8B | 2.8B | 6.73 | 64.7 | 59.3 |
| RWKV-3B | 3B | 7.00 | 63.9 | 59.6 |
| **Mamba-2.8B** | **2.8B** | **6.22** | **69.2** | **66.1** |
| GPT-J-6B | 6B | - | 68.3 | 66.3 |

**Mamba-2.8B 匹配 GPT-J-6B** (2x 更小!)

### 5.2 推理速度

```
训练: SSM scan 在 L>2K 时比 FlashAttention-2 更快
推理: 比 Transformer 快 5x (无 KV cache)
Mamba-6.9B 吞吐 > Transformer-1.3B (5x 更小的模型!)
```

### 5.3 长序列能力

```
Induction Heads: 训练 L=256, 泛化到 L=1M (4000x 外推!)
DNA 建模: 利用 1M 长上下文, 比 Transformer++ 好 3-4x
```

## 6. 局限性

```
1. 规模验证有限: 最大 2.8B (论文发表时)
   - 7B+ 的行为未验证
   - 后续 Jamba (AI21) 混合了 Mamba + Attention

2. 精确回忆能力:
   - 有限状态维度 N → 无法完美存储所有细节
   - Needle-in-a-haystack 可能不如 Transformer

3. 生态系统:
   - Transformer 的微调/RLHF/量化生态成熟
   - SSM 的这些方面仍是开放问题

4. 连续模态:
   - 选择性机制有利于离散数据 (文本/DNA)
   - 连续数据 (音频/视频) 可能需要调整
```

## 7. 对 AI Infra 的影响

```
1. 无 KV Cache: 推理内存 ∝ 1 (vs Transformer ∝ L)
   → 长序列推理成本大幅降低
   → 对 serving 框架 (vLLM/SGLang) 是新挑战

2. 线性训练: O(L) vs O(L²)
   → 长上下文训练成本降低
   → 但 kernel 需要自定义实现 (parallel scan)

3. 混合架构趋势:
   Jamba: Mamba 层 + 少量 Attention 层
   → 兼顾效率和精确回忆

4. 硬件适配:
   Mamba 的 scan kernel 需要类似 FlashAttention 的优化
   → 对 GPU SRAM 的利用方式不同
```

## 8. 核心学习

1. **选择机制是关键**: 输入相关参数让 SSM 摆脱 "无法内容推理" 的限制
2. **Δ 是 RNN 门控的推广**: 大 Δ = 选择, 小 Δ = 忽略
3. **并行扫描替代卷积**: 非 LTI 仍可高效计算
4. **硬件感知实现是必需**: 类似 FlashAttention 的 SRAM 优化
5. **线性时间 ≠ 一定更好**: Transformer 的精确回忆仍有优势
6. **混合架构可能是未来**: Mamba 的效率 + Attention 的精确回忆
7. **推理 5x 加速**: 无 KV cache 是最大卖点, 对 serving 成本影响巨大
