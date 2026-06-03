# 序列并行 (Sequence Parallelism) 深度解析

> 长序列训练的核心技术：从 DeepSpeed Ulysses 到 Ring Attention 到 USP 统一框架

## 1. 为什么需要序列并行

### 1.1 长序列训练的显存瓶颈

```
Transformer Attention 显存: O(N²)
  N = 1024:  1M entries
  N = 4096:  16M entries
  N = 32768: 1B entries
  N = 131072: 17B entries (128K context)

FP16 下一个 attention matrix:
  N=32768 → 64 × 32768² × 2 bytes = 128 GB (仅 attention 中间结果)

这远超单 GPU 显存 → 必须沿序列维度切分
```

### 1.2 现有并行策略的局限

```
数据并行 (DP): 切 batch → 不解决单序列显存问题
张量并行 (TP): 切 hidden_dim → Attention 的 O(N²) 仍在每个 GPU
流水线并行 (PP): 切 layers → 单层 Attention 的 O(N²) 无法拆分

序列并行 (SP): 切 sequence → 直接解决 O(N²) 显存瓶颈
```

### 1.3 两种核心思路

```
1. DeepSpeed Ulysses: 沿 head 维度切分序列 → All-to-All 通信
2. Ring Attention: 沿 sequence 维度切分 → P2P 环形通信
3. USP: 统一框架，两者可组合
```

## 2. DeepSpeed Ulysses

### 2.1 核心思想

```
将 sequence 维度切到不同 GPU，每个 GPU 处理完整的 head 子集

输入: X [batch, seq_len, hidden_dim]
切分: X_i = X[:, seq_len/P : seq_len*(i+1)/P, :]  (沿 seq 切)

但 attention 需要完整的 sequence → All-to-All 通信重排:
  切分后的 X_i: [batch, seq_len/P, hidden_dim]
  All-to-All 后: X_i': [batch, seq_len, hidden_dim/P]

→ 每个 GPU 有完整的 seq_len，但只有 1/P 的 heads
→ Attention 计算完整，只是 head 数减少
```

### 2.2 通信模式

```
Step 1: Input [B, S/P, H] → All-to-All → [B, S, H/P]
Step 2: Attention on [B, S, H/P] (完整 seq, 部分 heads)
Step 3: Output [B, S, H/P] → All-to-All → [B, S/P, H]
Step 4: MLP on [B, S/P, H] (部分 seq, 完整 hidden)

每个 Transformer 层需要 2 次 All-to-All
通信量: O(B × S × H / P) per All-to-All
```

### 2.3 优缺点

```
优点:
  - 实现简单（只需插入 All-to-All）
  - Attention 计算语义不变（完整 seq × 部分 heads）
  - 与 TP/PP 可组合

缺点:
  - 扩展受 head 数限制（P ≤ num_heads）
  - 序列并行度不能超过 head 数（通常 32-128）
  - 通信量与序列长度成正比
```

## 3. Ring Attention

### 3.1 核心思想

```
将序列分成 P 块，每个 GPU 持有一个 block 的 Q, K, V
通过环形通信传递 K, V blocks，逐块计算 attention

关键: 利用 FlashAttention 的分块计算特性
  FlashAttention 已经把 Q, K, V 切成 tiles 逐块计算
  Ring Attention 把这些 tiles 分布到不同 GPU 上
  每一步：计算本地 block 的 attention + 接收下一个 block 的 K, V
```

### 3.2 算法流程

```
初始状态: GPU_i 持有 Q_i, K_i, V_i (序列的第 i 块)

Step 0: 计算 Attn(Q_0, K_0, V_0) → 本地 attention
Step 1: K, V 环形传递 (GPU_i → GPU_{(i+1)%P})
        计算 Attn(Q_0, K_1, V_1) → 用 online softmax 合并
Step 2: K, V 再次传递
        计算 Attn(Q_0, K_2, V_2) → 继续合并
...
Step P-1: 最后一个 block，合并完成

每一步:
  计算: Attn(Q_i, K_j, V_j) — 用 FlashAttention kernel
  通信: 发送 K_j, V_j 到下一个 GPU — 用 P2P (ISend/IRecv)
  计算和通信可以 overlap!
```

### 3.3 关键技术：Blockwise FlashAttention + Online Softmax

```
问题: 完整 softmax 需要所有 blocks 的 max 和 sum
解决: FlashAttention 的 online softmax 天然支持增量更新

对每个 Q block:
  running_max = -inf
  running_sum = 0
  output = 0

  for each K,V block:
    local_attn = Q @ K.T / sqrt(d)
    local_max = max(local_attn)
    new_max = max(running_max, local_max)
    correction = exp(running_max - new_max)
    local_sum = sum(exp(local_attn - new_max))
    output = output * correction * running_sum + exp(local_attn - new_max) @ V
    running_sum = running_sum * correction + local_sum
    running_max = new_max

每处理一个 K,V block，online softmax 增量更新结果
→ 不需要等所有 blocks 到齐
→ 天然适合 Ring Attention 的逐 block 处理
```

### 3.4 计算通信重叠

```
Timeline (4 GPUs):

GPU 0: [Compute Attn(Q0,K0,V0)] [Send K0,V0] [Compute Attn(Q0,K1,V1)] [Send K1,V1] ...
GPU 1: [Compute Attn(Q1,K1,V1)] [Send K1,V1] [Compute Attn(Q1,K2,V2)] [Send K2,V2] ...
GPU 2: ...
GPU 3: ...

关键: 发送当前 K,V 的同时可以计算下一个 block
→ 计算和通信 overlap → 额外通信开销几乎为零

条件: 通信时间 < 计算时间
  通信量 per step: 2 × B × S/P × H × dtype_bytes
  计算量 per step: 2 × B × (S/P)² × H FLOPs
  当 S/P 足够大时，计算 > 通信 → overlap 有效
```

### 3.5 优缺点

```
优点:
  - 理论上无限扩展（P 可以任意大）
  - 不受 head 数限制
  - 计算通信可 overlap → 高效
  - 支持极长序列 (100K+ tokens)

缺点:
  - 实现复杂（需要自定义 FlashAttention kernel）
  - 需要 P2P 通信（NVLink 或 IB 直连）
  - 序列长度必须远大于 P 才有足够计算量 overlap
  - Causal attention 下负载不均衡（前面的 block 计算量小）
```

### 3.6 Causal Attention 的负载均衡

```
问题: Causal mask 下，position i 只 attend 到 0..i
  Block 0: 只 attend 自己 → 计算量 1/P²
  Block P-1: attend 所有 blocks → 计算量最大

解决: Stripe Attention (条纹注意力)
  将序列按条纹模式分配，使每个 GPU 的计算量均衡

或: 动态分配 block 大小，前面的 block 大，后面的小
```

## 4. Megatron Context Parallelism (CP)

### 4.1 与 Ring Attention 的关系

```
Megatron CP 基于 Ring Attention，但做了工程优化:

1. 支持 FlashAttention 融合
2. 与 Megatron TP/PP/DP 无缝集成
3. 支持 GQA (KV heads < Q heads)
4. 优化的 P2P 通信 schedule

主要用于 prefill 阶段的长序列处理
(Decode 阶段序列增长缓慢，通常不需要 CP)
```

### 4.2 CP + TP 组合

```
典型配置 (H100 8-GPU):
  TP = 4 (张量并行)
  CP = 2 (上下文并行)

每 4 GPU 做 TP (NVLink 高带宽)
TP 组之间做 CP (也通过 NVLink)

通信: CP 需要 P2P → TP 组内用 AllReduce
两层通信独立，可以分别优化
```

## 5. USP: 统一序列并行框架

### 5.1 设计思路

```
观察: Ulysses 切 head 维度，Ring Attention 切 sequence 维度
     两者正交 → 可以组合!

USP = Ulysses (沿 head 切) × Ring (沿 seq 切)

二维序列并行:
  U: Ulysses 并行度 (head 切分数)
  R: Ring 并行度 (seq 切分数)
  总并行度 = U × R

通信模式:
  Ulysses: All-to-All (collected heads)
  Ring: P2P (streaming K,V blocks)
```

### 5.2 配置灵活性

```
示例: 16 GPU, 32 heads, seq = 128K

纯 Ulysses: U=16, R=1
  每个 GPU: 2 heads, 完整 seq
  限制: U ≤ 32 (head 数)

纯 Ring: U=1, R=16
  每个 GPU: 32 heads, seq/16
  无 head 数限制

混合: U=4, R=4
  每个 GPU: 8 heads, seq/4
  平衡通信模式和扩展性
```

## 6. 实际应用场景

### 6.1 长上下文训练

```
场景: 训练支持 128K context 的模型
挑战: Attention O(N²) 显存 + O(N²) 计算

解决方案:
  CP=8: 序列切 8 份，每个 GPU 处理 16K
  配合 FlashAttention: 本地 16K 的 attention 已经被优化
  Ring CP: K,V 环形传递，8 步完成完整 attention

效果:
  单 GPU 显存需求: 原来的 1/8
  训练吞吐: 计算通信 overlap 后几乎无损失
```

### 6.2 RLHF 中的长 Prompt

```
场景: RLHF 的 prompt 可能数千 tokens，batch 内 prompt 长度差异大
挑战: Prefill 阶段长 prompt 的 attention 计算瓶颈

解决: Prefill 阶段用 CP 切分序列
  Decode 阶段关掉 CP (每步只算一个 token，不需要)
  Prefill→Decode 切换时需要 all-gather KV Cache
```

### 6.3 多模态训练

```
场景: 图像/视频 token 序列极长
  一张高分辨率图片: 数千 tokens
  一段视频: 数万 tokens

解决: 视觉编码器输出 → CP 切分 → LLM 处理
  视觉部分的 attention 用 Ring Attention
  文本部分可以不用 CP (序列较短)
```

## 7. 通信开销分析

### 7.1 通信量对比

```
假设: B=batch, S=seq_len, H=hidden, P=parallel_size

Ulysses:
  通信量 = 2 × B × S × H / P (2 次 All-to-All)
  带宽需求 = 正比于 S (序列越长通信越大)

Ring Attention:
  通信量 = 2 × (P-1) × B × S/P × H (P-1 次 P2P)
         = 2 × B × S × H × (P-1)/P ≈ 2 × B × S × H
  但可以与计算 overlap!

TP (对比):
  通信量 = 2 × B × S × H / P (2 次 AllReduce per layer)
```

### 7.2 何时选择哪种策略

```
| 条件 | 推荐策略 |
|------|---------|
| P ≤ num_heads, 中等序列 | Ulysses |
| P > num_heads, 超长序列 | Ring Attention |
| 超大规模 + 超长序列 | USP (混合) |
| 需要与 TP 组合 | Ring CP (Megatron) |
```

## 8. 关键要点

1. **序列并行直接解决 O(N²) 显存瓶颈** — TP/PP/DP 都无法拆分单层 attention 的显存
2. **Ring Attention 理论无限扩展** — 不受 head 数限制，支持任意长序列
3. **计算通信 overlap 是关键** — Ring Attention 天然支持，额外开销几乎为零
4. **USP 统一框架最灵活** — 正交组合 Ulysses 和 Ring，适应不同硬件和序列长度
5. **主要用于 prefill** — Decode 阶段序列增长缓慢，通常不需要序列并行

## 参考

- 论文: [DeepSpeed Ulysses](https://arxiv.org/abs/2304.14977)
- 论文: [Ring Attention with Blockwise Parallel FlashAttention](https://arxiv.org/abs/2310.01889)
- 论文: [USP: A Unified Sequence Parallelism Approach](https://arxiv.org/abs/2405.07747)
- Megatron-LM Context Parallelism: https://docs.nvidia.com/megatron-core/
- 博客: [Sequence Parallelism in Long Context Training](https://pytorch.org/blog/sequence-parallelism/)
