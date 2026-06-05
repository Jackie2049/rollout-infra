# MoE (Mixture of Experts) 架构深度解析

> 从数学原理到工程实现: DeepSeek-V3, Mixtral, Switch Transformer
> 关键问题: 为什么 MoE 能在相同计算量下获得更好效果?

## 1. MoE 核心思想

### 1.1 Dense vs Sparse 模型

```
Dense Model (如 LLaMA):
  每个 token 激活所有参数
  7B model: 每个前向 ≈ 7B 参数的计算
  → 计算 ∝ 总参数量

Sparse MoE Model (如 Mixtral):
  每个 token 只激活部分参数 (top-K experts)
  Mixtral 47B: 每 token 只激活 13B (8 个 expert 选 top-2)
  → 计算 ∝ 激活参数量, 不 ∝ 总参数量

关键洞察:
  模型容量 (总参数) ≠ 计算量 (激活参数)
  MoE: 用 3-18x 更多参数获得相同计算量的更好效果
```

### 1.2 MoE Layer 结构

```
标准 Transformer FFN:
  y = W₂ · activation(W₁ · x)       d → 4d → d

MoE FFN:
  y = Σ_k g_k(x) · E_k(x)

  其中:
    E_k(x) = W₂ᵏ · activation(W₁ᵏ · x)  (第 k 个 expert)
    g_k(x) = softmax(top_k(router(x)))_k   (gating score)

Router:
  router(x) = W_router · x    [d → num_experts]

Top-K 选择:
  scores = softmax(router(x))
  top_k_indices = topk(scores, K)
  g_k = scores[k] / Σ_{j∈topK} scores[j]  (重新归一化)
```

### 1.3 参数量对比

```
Mixtral 8x7B:
  每个 expert: ~7B FFN 参数 (d=4096, ffn_dim=14336)
  8 个 expert: 56B FFN 参数
  Attention + Embedding: ~7B (共享, 不复制)
  总参数: 47B (但只激活 top-2 = 13B)

DeepSeek-V3:
  256 个 router expert + 1 个 shared expert
  每 expert: ~0.5B 参数
  总参数: 671B, 激活: 37B (18x 稀疏比)

Switch Transformer (Google, 2021):
  首次提出 top-1 routing
  简化 → 更快 → 略有精度损失
```

## 2. Router 的数学

### 2.1 负载均衡 (Load Balancing)

**问题**: Router 可能把所有 token 都发给同一个 expert (坍塌)

```
解决: Auxiliary Loss

  L_aux = α × Σ_k f_k · p_k

  其中:
    f_k = batch 中分配给 expert k 的 token 比例
    p_k = batch 中 router 对 expert k 的平均概率

  这个 loss 鼓励:
  - f_k 均匀 (每个 expert 收到相同数量的 token)
  - p_k 均匀 (router 的概率分布均匀)

  α 通常 = 0.01 (太大会损害主 loss)
```

### 2.2 Expert Choice Routing (Google, 2022)

```
传统 Token Choice: token 选 expert (top-k)
  问题: 每个 expert 的负载不均

Expert Choice: expert 选 token
  每个 expert 选 top-m 个 token (m = N_tokens / N_experts)
  → 完美负载均衡!
  → 但某些 token 可能被多次选择, 某些被忽略
```

### 2.3 DeepSeek-V3 的无辅助损失负载均衡

```
核心创新: 用 bias 替代 auxiliary loss

  router(x) = softmax(W_router · x + b)

  b 是可学习的 per-expert 偏置, 但:
  - b 不参与梯度计算 (只用于 routing 决策)
  - 每个 step 后根据负载更新 b:
    如果 expert_k 过载: b_k -= γ (降低选中概率)
    如果 expert_k 欠载: b_k += γ (增加选中概率)

  优势: 不需要 auxiliary loss → 不干扰训练
```

## 3. MoE 训练的工程挑战

### 3.1 内存挑战

```
Dense 7B (BF16):
  模型权重: 14 GB
  优化器 (Adam, FP32): 28 GB
  梯度: 14 GB
  激活: ~8 GB
  总计: ~64 GB → 单卡 A100 80GB 可行

MoE 47B (Mixtral, BF16):
  模型权重: 94 GB (47B × 2)
  优化器: 188 GB
  梯度: 94 GB
  激活: ~15 GB
  总计: ~391 GB → 需要 5× A100 80GB (EP)

  关键: 虽然 token 只激活 13B, 但优化器需要所有 47B 的参数!
  → MoE 训练内存 ∝ 总参数, 不 ∝ 激活参数
```

### 3.2 Expert Parallelism (EP)

```
EP: 每个 GPU 存储不同的 expert 子集

实现:
  1. All-to-All dispatch: token → 对应 expert 所在的 GPU
  2. Expert 计算: 每个 GPU 计算自己负责的 expert
  3. All-to-All gather: 结果 → 回到原始 GPU

通信量:
  A2A 通信 = B × N × d × 2 (dispatch + gather)
  = 32 × 2048 × 4096 × 2 × 2 = 1 GB

  NVLink (300 GB/s): 3.3 ms
  Ethernet (25 Gbps): 320 ms → 太慢!

  → EP 需要高速互联 (NVLink / InfiniBand)
```

### 3.3 训练效率

```
MoE 的 MFU (Model FLOPs Utilization) 通常比 Dense 低:
  Dense: 50-60% MFU
  MoE: 30-45% MFU

原因:
  1. All-to-All 通信开销
  2. 负载不均 → 有些 GPU 空闲等待
  3. Expert 容量限制 (capacity factor) → token dropping
  4. 额外的 router 计算

优化:
  - DeepEP: 分层 A2A (NVLink 组内 + 跨节点)
  - 容量因子: 1.0-1.5 (太小 → drop tokens, 太大 → 浪费)
  - Expert 数量: 8-256 (8 最常见, 256 见于 DeepSeek)
```

## 4. MoE 推理优化

### 4.1 推理 vs 训练的区别

```
训练: 所有 expert 都需要前向+反向 (即使 token 没选中)
  → 内存 ∝ 总参数

推理: 只需要激活的 expert
  → 计算 ∝ 激活参数
  → 但需要所有 expert 权重在内存中 (用于不同 token 的 routing)

MoE 推理的内存瓶颈:
  Mixtral 47B: 需要 94 GB 权重在 GPU 上
  但每 token 只用 13B 的计算
  → 计算/内存比 = 13/47 = 28% (Dense = 100%)
  → 更严重的 memory-bound!
```

### 4.2 Expert Offloading

```
将不活跃的 expert 权重 offload 到 CPU/SSD:
  1. 预测下一个 token 的 expert 需求
  2. 异步预加载 expert 权重到 GPU
  3. 用完后释放 GPU 内存

挑战:
  - Router 输出在运行时才知道 → 难以预测
  - PCIe 带宽: ~32 GB/s (CPU→GPU)
  - 一个 expert: 7B × 2 bytes = 14 GB → 0.44s 加载 (太慢!)

解决:
  - Expert 并行 (EP): 每个 GPU 存一部分 expert
  - Speculative expert prefetch: 预测 + 提前加载
  - Expert quantization: FP8/INT4 减小 expert 大小
```

### 4.3 MoE Serving 的 Batch 调度

```
挑战: 同一个 batch 中不同 token 选不同 expert
  → 每个 expert 的 batch size 不同
  → GPU 利用率不均

优化:
  1. Expert batching: 按目标 expert 分组 token
     → 每个 expert 有自己的 mini-batch
     → 矩阵乘形状: [n_k, d] × [d, ffn_dim]

  2. Padding + Grouped GEMM:
     将所有 expert 的 batch pad 到相同大小
     → 一次 grouped GEMM 调用处理所有 expert
     → Megatron-LM 和 vLLM 使用此方法

  3. Token routing 统计:
     实际中 expert 负载偏差 ≈ ±30%
     Top-2 routing 比.Top-1 负载更均衡
```

## 5. DeepSeek-V3 的 MoE 创新

### 5.1 架构细节

```
总参数: 671B (256 个 routed expert + 1 个 shared expert)
激活参数: 37B (top-8 routed + 1 shared)

每层:
  Attention: 共享 (所有 token 使用相同的 attention)
  FFN:
    shared_expert(x) → 所有 token 都经过 (不路由)
    Σ_{k∈top8} g_k(x) · routed_expert_k(x) → top-8 路由
    y = shared_expert(x) + routed_output

  为什么有 shared expert?
    - 知识分两类: 通用知识 + 专业知识
    - Shared expert 学习通用知识 (所有 token 共享)
    - Routed expert 学习专业知识 (特定领域)
    - 避免每个 expert 都重复学习通用知识
```

### 5.2 FP8 训练

```
DeepSeek-V3 使用 FP8 (E4M3) 训练:
  - 权重: FP8 (每 128 个 token 量化一次)
  - 激活: FP8
  - 累积: FP32 (在 GEMM 内部)
  - Router: BF16 (不量化)

  精度损失: < 0.3% (perplexity 差异)
  训练速度: 1.8x faster than BF16
  内存节省: 50% (通信量减半)

  关键技巧:
  - 细粒度量化: per-tile (128×128) 而非 per-tensor
  - 延迟缩放: 用前一步的 max 值做当前步的缩放因子
```

## 6. MoE vs Dense: 何时选择?

```
┌──────────────────────────────────────────────────────────────┐
│ 方面       │ Dense           │ MoE                    │
│────────────│─────────────────│────────────────────────│
│ 训练效率   │ MFU 50-60%      │ MFU 30-45%             │
│ 推理效率   │ 简单, 均匀      │ 负载不均, memory-bound │
│ 内存需求   │ ∝ 激活参数      │ ∝ 总参数 (训练)        │
│ 效果       │ 较差 (同计算量)  │ 更好 (2-3x 容量)       │
│ 通信需求   │ TP: AllReduce   │ EP: All-to-All         │
│ 实现复杂度 │ 低              │ 高 (routing, load bal) │
│ 适用场景   │ 小模型, 资源有限 │ 大模型, 高端集群       │
└──────────────────────────────────────────────────────────────┘

推荐:
  - 模型 < 30B: Dense (简单, 高效)
  - 模型 30-100B: MoE 8 expert (Mixtral 风格)
  - 模型 > 100B: MoE 64-256 expert (DeepSeek 风格)
  - 推理服务: Dense 通常更经济 (MoE 的内存开销大)
```
