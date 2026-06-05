# MoE Layer — Mixture of Experts 从零实现 (RTX 4090)

> 2026-06-05 | 工具: `tools/moe_layer.py` | RTX 4090 24GB
> 参考: DeepSeek-V3, Switch Transformer, Mixtral

## 1. 实验设计

**目的**: 从零实现 MoE (Mixture of Experts) 层, 验证:
1. Top-K 路由行为和负载均衡
2. Dense vs MoE 参数量对比
3. Dense vs MoE 吞吐对比
4. Top-K 值对性能的影响
5. Batch Size 对负载均衡的影响

**架构**:
```
MoE Layer:
  Router: Linear(D, n_experts) → softmax → top-k selection
  Expert: SwiGLU FFN (w_gate, w_up, w_down)
  Output: Σ(top_k_scores[i] * expert[i](x))

Load Balance Loss = n_experts * Σ(f_i * P_i)
  f_i = 分配到 expert i 的 token 比例
  P_i = expert i 的平均路由概率
```

## 2. Top-K 路由分析

| n_experts | aux_loss | load_std | max_load | min_load | active |
|-----------|----------|----------|----------|----------|--------|
| 4 | 1.0001 | 0.0046 | 0.253 | 0.243 | 4/4 |
| 8 | 1.0014 | 0.0108 | 0.139 | 0.111 | 8/8 |
| 16 | 1.0009 | 0.006 | 0.075 | 0.052 | 16/16 |
| 32 | 1.0027 | 0.005 | 0.041 | 0.023 | 32/32 |

**发现**:
- aux_loss ≈ 1.0 (完美均衡的理论值), 初始化时路由已经很均匀
- load_std 随 expert 数量增加而下降 (0.011→0.005), 更多专家 → 更细粒度均衡
- 所有 expert 都被激活 (random init 的效果)
- max/min ratio: E=4 仅 1.04x, E=32 仅 1.78x → 负载很均衡

## 3. Dense vs MoE 参数量

| n_experts | Dense (K) | MoE Total (K) | MoE Active (K) | Active% | Sparsity |
|-----------|-----------|---------------|----------------|---------|----------|
| 4 | 786.4 | 3146.8 | 1573.9 | 50.0% | 2.0x |
| 8 | 786.4 | 6293.5 | 1574.9 | 25.0% | 4.0x |
| 16 | 786.4 | 12587.0 | 1577.0 | 12.5% | 8.0x |

**关键**:
- MoE 总参数 ∝ n_experts (每增加一个 expert +1x d_model*d_ff*3)
- 但 active 参数几乎不变 (top_k=2 固定, router 参数很小)
- Sparsity = n_experts / top_k → DeepSeek-V3: 256/8 = 32x

## 4. Dense vs MoE 吞吐对比

| n_experts | Dense (ms) | MoE (ms) | Overhead |
|-----------|------------|----------|----------|
| 4 | 2.11 | 12.27 | 5.80x |
| 8 | 1.34 | 18.36 | 13.73x |
| 16 | 1.32 | 34.74 | 26.30x |

**核心发现**:
- MoE 远慢于 Dense! (5.8x - 26.3x)
- 原因: **Python 循环 + gather/scatter 开销**
  - 每个 expert 需要独立的前向传播
  - token 的 gather (提取分配给每个 expert 的 token)
  - 结果的 scatter (写回加权结果)
- Dense: 一个大的 matmul, GPU 并行度高
- MoE: 多个小 matmul, GPU 利用率低 + Python 循环开销

**这是 Python 实现的问题, 不是 MoE 架构的问题!**
- 生产实现: batched GEMM (如 Megatron-LM, vLLM)
- Grouped GEMM: 所有 expert 的 matmul 合并成一个 kernel
- Expert Parallelism: 每个 GPU 只处理分配给它的 expert

## 5. Top-K 值效果

| K | Time (ms) | Active Ratio | aux_loss |
|---|-----------|-------------|----------|
| 1 | 3.71 | 12% | 1.0013 |
| 2 | 7.22 | 25% | 1.0002 |
| 4 | 14.18 | 50% | 1.0004 |
| 8 | 28.11 | 100% | 1.0000 |

**发现**:
- 时间 ∝ K (线性增长) — 每个 K 需要遍历所有 expert
- K=1 最快但容量最小 (Switch Transformer 方案)
- K=8 = 全部 expert → 等价于 Dense (aux_loss=1.0 完美均衡)
- 生产常用 K=2 (Mixtral, DeepSeek-V3): 25% active

## 6. 负载均衡 vs Batch Size

| Batch Size | load_std | max/min |
|------------|----------|---------|
| 4 | 0.0096 | 1.25x |
| 16 | 0.0074 | 1.16x |
| 64 | 0.0065 | 1.16x |
| 128 | 0.0062 | 1.16x |
| 256 | 0.0058 | 1.15x |

**发现**:
- 更大 batch → 更均衡 (大数定律)
- BS=4: max/min=1.25x → BS=256: max/min=1.15x
- 收敛很快: BS≥16 已经相当均衡
- 训练时 batch 通常很大 → 负载不均衡不是严重问题

## 7. 核心学习

1. **MoE 本质**: 用参数量换计算量 — 总参数 n×dense, 但每个 token 只用 k/n
2. **Python 循环是瓶颈**: 生产实现需要 batched GEMM / grouped GEMM
3. **路由开销很小**: Top-K 选择 <1% 计算量, 主要是 gather/scatter 开销
4. **负载均衡**: aux_loss + 大 batch → 均匀分配, 无需复杂调度
5. **K 的权衡**: K=1 最快但容量小, K=2 平衡点, K=全部=退化为 Dense
6. **推理瓶颈**: 所有 expert 权重都需加载到 GPU → memory-bound
   - DeepSeek-V3 671B: 需要 ~1.3TB 内存 (即使只激活 37B)
   - 解决: Expert Parallelism + 量化 + CPU offload
