# Speculative Decoding Algorithm Deep Dive

> 2026-06-07 | 基于 Leviathan et al. (ICML 2023) + Chen et al. (2023, Medusa) + Li et al. (2023, Eagle)
> 已有模拟器笔记 (sharp=10→5.1x, 最优K=3-5), 本文聚焦算法理论 + vLLM实现

## 1. 核心问题: LLM Decode 的 Memory-Bound 性质

**单步 Decode**:
- FLOPs = 2 × d × V (一次矩阵乘: hidden→vocab) ≈ O(d × V)
- HBM IO = 2 × d × V (权重读取) ≈ 同上
- AI ≈ 1.0 → 始终 memory-bound

**后果**: GPU 利用率极低 (A100 上 decode 仅 ~0.7% peak FLOPS)

**投机解码的思路**: 既然 GPU 计算力有余但被内存带宽限制, 用更小的模型 (或同一模型的历史信息) "免费"生成多个候选 token, 然后一次验证所有候选

## 2. Draft-then-Verify 范式

### 基本流程:
```
1. Draft: 用小模型 q(x) 生成 K 个候选 token: [x₁, x₂, ..., x_K]
   (小模型的计算成本很低, memory-bound 但权重更小 → HBM IO 更少)

2. Verify: 用大模型 p(x) 一次前向计算, 同时获取所有 K 个位置的概率
   (因为 p(x) 是 memory-bound, 计算 K 个 token 的概率 ≈ 1 个 token 的成本)
   → 对同一 batch 的多个位置做一次 matmul (利用 GPU 的计算余量)

3. Accept/Reject: 对每个候选 x_i, 用拒绝采样决定是否接受
   如果接受: 输出 x_i, 继续检查下一个
   如果拒绝: 从修正分布采样一个替代 token, 停止后续验证

4. Continue: 从接受的最后一个 token (或替代 token) 继续 Draft-Verify 循环
```

### 为什么 Verify "免费"?

**单步大模型 decode**: 读取权重 W [d, V] + 计算 hidden @ W → 一次 IO = 2dV

**K 步验证**: 对 K 个位置的 hidden vectors, 一次 matmul:
```
H_batch = [h₁, h₂, ..., h_K]   # [K, d]
probs = H_batch @ W              # [K, V] — 一次 matmul!

IO = 2 × d × V (权重只读一次!) + 2 × K × d (hidden vectors)
FLOPs = 2 × K × d × V

额外成本: 2 × K × d (hidden vectors IO) ≈ 很小 (K << V)
```

**关键**: 权重矩阵只读一次! 从 IO = 2dV 变为 2dV + 2Kd ≈ 2dV (当 K << V)
→ **验证 K 个 token 的 IO ≈ 验证 1 个 token 的 IO**

这就是投机解码的理论基础: **利用 memory-bound 的 "计算余量" 来并行验证**

## 3. 拒绝采样理论 (Rejection Sampling)

### 采样算法:
```
对每个候选 x_i (从 q 采样):
  如果 p(x_i) / q(x_i) ≥ r, 其中 r ~ Uniform(0, 1):
    接受 x_i
  否则:
    拒绝 x_i
    从修正分布采样: x ~ p(x) - q(x) for all x where p(x) > q(x)
```

**修正分布**:
```
x ~ Normalize(max(0, p(x) - q(x)))  # 剩余概率质量
```

### 接受概率推导:

对于单个 token x_i:
```
P(accept x_i) = P(r ≤ p(x_i)/q(x_i)) = p(x_i)/q(x_i) × 期望值

总接受概率 = E_q[p(x)/q(x)] = Σ q(x) × p(x)/q(x) = Σ p(x) = 1 (!)
```

**惊人结论**: 单步接受概率 = **1.0**! (理论上)

但实际中:
- p 和 q 不完全匹配 → 某些 token 的 p/q < 1 → 某些拒绝
- 更精确的分析: 对于分布 p 和 q:

```
E[accept] = min(1, p(x)/q(x)) 对所有 x
总接受率 = Σ min(p(x), q(x))  ← 取 p 和 q 的重叠面积

理想情况: q ≈ p → 接受率 ≈ 1
现实: q ≠ p → 掓受率 < 1
```

### K 个候选的期望接受数:

假设每步独立 (实际不独立, 但近似):
```
E[accepted tokens] = Σ_{i=1}^{K} P(step 1..i all accepted)
                   = Σ_{i=1}^{K} α^i   (α = per-step acceptance rate)
                   = α × (1 - α^K) / (1 - α)   (α < 1)
                   ≈ α / (1 - α)   (α^K → 0 for large K)
```

**最优 K**: 取 max of speedup(K) = (E[accepted] + 1) / (1 + cost_draft/cost_verify)

对 α=0.8: E[K=3]=2.44, E[K=5]=3.36, E[K=7]=3.81 → K=3-5 通常最优 (模拟器验证)

### Temperature 的影响:

高 Temperature → 分布更均匀 → q 更接近 p → 接受率更高
低 Temperature → 分布更尖锐 → q 和 p 差异更大 → 接受率更低

**实证**: 模拟器显示 Temperature=1.0 时接受率最高, T=0.1 时最低

## 4. Proposer 类型对比

### Self-Speculative Decoding (Draft = 目标模型自身)
- 用同一模型, 但跳过某些层 (如只跑前半部分) → 低成本 draft
- 无需额外模型, 但需要修改模型推理流程
- 接受率: ~0.8-0.9 (因为是同一模型的近似)

### N-gram Speculative Decoding
- 用历史 token 的 n-gram 统计来预测下一个 token
- 零计算成本! 纯 lookup
- 接受率: ~0.3-0.5 (受限于 n-gram 的信息量)
- 适合: 低 Temperature + 重复性文本 (code, structured data)

### Eagle (Draft = 轻量自回归模型)
- 训练一个小的 draft model, 从目标模型的中间特征 (而不是原始输入) 开始
- 特征级 draft → 更高效 (避免 embedding 层开销)
- 接受率: ~0.85-0.92 (训练良好时)
- vLLM 推荐的 proposer

### Medusa (Multi-token Prediction)
- 不用单独的 draft model, 在目标模型头上添加多个预测头
- 每个 head 预测不同位置的 token → 并行生成多个候选
- 无自回归依赖 → 但每个 head 的准确率较低
- 接受率: ~0.6-0.7 per head (但多个 head 独立)

### 对比总结:

| Proposer | 计算成本 | 接受率 | 依赖模型 | 适合场景 |
|----------|---------|--------|---------|---------|
| Self-spec | 低 | 0.8-0.9 | 需要 | 通用 |
| N-gram | 零 | 0.3-0.5 | 不需要 | 重复性文本 |
| Eagle | 中 | 0.85-0.92 | 需要 draft | 通用 (推荐) |
| Medusa | 低 | 0.6-0.7 | 需要微调 | 通用 |

## 5. Speedup 性能模型

### 理论加速比:
```
S = (1 + E[accepted]) / (1 + C_draft/C_target + C_verify/C_target)

其中:
- E[accepted] = 期望接受的 token 数
- C_draft = draft 模型单步成本
- C_verify = 验证一步的成本 ≈ C_target (memory-bound)

简化 (假设 C_verify ≈ C_target):
S = (1 + E[accepted]) / (1 + C_draft/C_target)

对 Eagle (C_draft ≈ 0.1 × C_target, α=0.85):
S = (1 + 3.36) / (1 + 0.1) = 3.97x

对 N-gram (C_draft ≈ 0, α=0.4):
S = (1 + 1.67) / 1 = 2.67x (但 K 较小时)
```

### 实际加速比 (文献报告):
- LLaMA-70B + Eagle-7B: 2.7x (A100)
- OPT-13B + N-gram: 1.4-2.0x
- LLaMA-2-70B + Medusa: 2.3x
- 模拟器 RTX 4090 实测: sharp=10 → 5.1x (理论上限)

### 加速比极限分析:
```
最大 S = O(C_target/C_draft) × α/(1-α)
       ≈ O(模型大小比例) × 接受率/(1-接受率)

对 70B + 7B draft: S_max ≈ 10 × 0.85/0.15 ≈ 56.7x
实际: ~2.7x (因为 draft 成本 > 理论, 和其他开销)
```

## 6. vLLM 实现 (待补充)

根据 MEMORY 中的笔记:
- **Proposer-Scorer 双层**: SpecDecodeProposer (生成 draft) + RejectionSampler (验证)
- **8+ proposer**: Eagle (推荐), N-gram (零开销), Medusa, etc.
- **Triton 拒绝采样 kernel**: 6 个 Triton kernel (greedy 直接 argmax / random 概率比)
- **Rejection Sampler**: 6 个 Triton kernel 处理不同的 sampling 模式

(vLLM 源码详细分析待背景 agent 返回后补充)

## 7. 关键 Takeaways

1. **投机解码的本质**: 利用 memory-bound 的 IO 特性 — 权重只读一次, 计算 K 个位置的概率
2. **接受率 = 1.0 (理论)**: 但实际 α < 1, 取决于 draft 和 target 的分布匹配度
3. **Temperature 影响**: 高 T → 高接受率, 低 T → 低接受率 (但对 greedy decoding 不适用)
4. **最优 K = 3-5**: 不是越大越好, 超过接受率衰减点后浪费验证成本
5. **Eagle 是当前最佳**: 特征级 draft, 高接受率, vLLM 推荐
6. **N-gram 是零成本方案**: 适合重复性文本, 不需要额外模型
7. **Medusa 是多头方案**: 无自回归依赖, 但每头准确率较低
8. **加速比上限**: O(模型大小比 × α/(1-α)), 实际 ~2-3x due to overhead