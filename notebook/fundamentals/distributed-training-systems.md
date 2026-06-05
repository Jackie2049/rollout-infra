# 分布式训练系统: 从 DDP 到 3D 并行

> 深度解析 AI 训练的分布式策略: DP/TP/PP/SP/EP 的原理、组合与工程实践
> 目标: 理解何时用哪种并行, 以及它们如何组合

## 1. 并行策略全景

```
┌──────────────────────────────────────────────────────────────────────┐
│ 策略             │ 切分维度     │ 通信模式          │ 适用场景      │
│──────────────────│──────────────│───────────────────│───────────────│
│ DP (Data Par.)   │ 数据 (batch) │ AllReduce (梯度)  │ 小模型多数据  │
│ ZeRO             │ 优化器状态   │ AllGather/ReduceSc │ 大模型 DP     │
│ FSDP             │ 模型层       │ AllGather (前向)  │ 大模型通用    │
│ TP (Tensor Par.) │ 权重 (列/行) │ AllReduce (每层)  │ 大模型单节点  │
│ PP (Pipeline)    │ 层 (深度)    │ P2P (activation)  │ 超大模型      │
│ SP (Sequence)    │ 序列长度     │ AllGather/RS      │ 长序列        │
│ EP (Expert Par.) │ Expert       │ All-to-All        │ MoE 模型      │
│ CP (Context Par.)│ 序列 (ring)  │ P2P (KV chunks)   │ 超长上下文    │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. Data Parallelism (DP)

### 2.1 DDP (Distributed Data Parallel)

```
每个 GPU 持有完整模型副本, 处理不同数据:

GPU 0: Model Copy → Forward(batch_0) → Backward → grad_0 ─┐
GPU 1: Model Copy → Forward(batch_1) → Backward → grad_1 ─┤→ AllReduce → avg_grad → Update
GPU 2: Model Copy → Forward(batch_2) → Backward → grad_2 ─┤
GPU 3: Model Copy → Forward(batch_3) → Backward → grad_3 ─┘

内存: 模型权重 + 优化器 + 梯度 = 每GPU完整副本
通信: AllReduce(梯度) = 2×模型大小 × (N-1)/N (Ring算法)

限制: 模型必须 fit 进单GPU内存
  7B BF16: 14GB权重 + 14GB梯度 + 28GB优化器 = 56GB → A100 80GB 勉强
  70B BF16: 560GB → 需要 8×A100 但每卡仍需完整模型 → 不可能!
```

### 2.2 ZeRO (Zero Redundancy Optimizer)

```
核心洞察: DDP 在每个 GPU 上冗余存储了 3 类数据:
  1. 优化器状态 (Adam: m, v) — 占 50%+ 内存
  2. 梯度 — 占 ~25%
  3. 模型权重 — 占 ~25%

ZeRO 分阶段消除冗余:

ZeRO-1 (优化器分片):
  每GPU只存 1/N 的优化器状态
  → 内存 = 权重 + 梯度 + 优化器/N
  → 70B (DP=8): 140GB + 140GB + 560GB/8 = 350GB → 44GB/GPU

ZeRO-2 (优化器 + 梯度分片):
  每GPU只存 1/N 的优化器状态 + 1/N 的梯度
  → 内存 = 权重 + 梯度/N + 优化器/N
  → 70B (DP=8): 140GB + 140GB/8 + 560GB/8 = 245GB → 31GB/GPU

ZeRO-3 (全部分片):
  每GPU只存 1/N 的所有东西
  → 前向时 AllGather 权重, 反向后丢弃
  → 内存 = (权重 + 梯度 + 优化器) / N
  → 70B (DP=8): 840GB/8 = 105GB → 13GB/GPU

通信量:
  ZeRO-1: 与 DDP 相同 (AllReduce 梯度)
  ZeRO-2: ReduceScatter 梯度 (与 AllReduce 通信量相同)
  ZeRO-3: AllGather 权重 (前向+反向各一次) = 3x DDP 通信量
```

### 2.3 FSDP (Fully Sharded Data Parallel)

```
PyTorch 的 ZeRO-3 实现:

关键概念:
  - Sharding: 将参数按 GPU 数量分片
  - Forward: AllGather 每层的权重 → 计算 → 丢弃
  - Backward: AllGather 权重 → 计算梯度 → ReduceScatter 梯度 → 丢弃

  FSDP Unit (分片粒度):
    - 默认: 按层 (每个 Transformer 层是一个 unit)
    - wrap_strategy: size_based (按参数量) / transformer_auto_wrap (按层类型)

  代码模式:
    model = FSDP(model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        device_id=local_rank,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ))

  比手动 ZeRO 的优势:
    - 自动分片和管理
    - 支持混合精度
    - 支持激活检查点
    - 与 torch.compile 兼容
```

## 3. Tensor Parallelism (TP)

### 3.1 原理

```
将权重矩阵按列或行切分到多个 GPU:

Column Parallel (切分输出维):
  Y = XW = X[W₁ | W₂ | ... | Wₙ] = [XW₁ | XW₂ | ... | XWₙ]
  每GPU计算 1/N 的输出列
  → 前向: 无通信 (各算各的)
  → 反向: AllReduce (梯度需要完整输入)

Row Parallel (切分输入维):
  Y = XW = [X₁, X₂, ..., Xₙ] [W₁; W₂; ...; Wₙ] = X₁W₁ + X₂W₂ + ... + XₙWₙ
  每GPU持有 1/N 的输入列和对应权重行
  → 前向: AllReduce (需要求和所有部分结果)
  → 反向: 无通信 (梯度是局部的)

标准 Transformer 层的 TP:
  Attention:
    QKV proj: Column Parallel → 每GPU有 h/N 个 head
    Attn compute: 独立 (每个 head 独立)
    Output proj: Row Parallel → AllReduce
  FFN:
    Up/Gate proj: Column Parallel
    Down proj: Row Parallel → AllReduce
```

### 3.2 通信模式

```
每层的通信:
  Attention output: 1次 AllReduce
  FFN down: 1次 AllReduce
  总计: 2次 AllReduce per layer

AllReduce 通信量:
  Ring AllReduce: 2 × (N-1)/N × message_size
  = 2 × (TP-1)/TP × B × d × dtype_size

  TP=4, B=32, d=4096, BF16:
    = 2 × 3/4 × 32 × 4096 × 2 = 393 KB per layer
    32 层 × 2 = 25 MB per step

  NVLink (300 GB/s): 25/300 = 0.08ms → 可以忽略
  Ethernet (25 Gbps): 25/3.125 = 8ms → 很显著!
```

### 3.3 Megatron-LM TP 实现

```python
# ColumnParallelLinear: 切分输出维
class ColumnParallelLinear:
    def forward(self, x):
        # x: [B, d_in], weight: [d_in, d_out/N]
        output = F.linear(x, self.weight)  # [B, d_out/N]
        # 不通信, 返回局部结果
        return output

# RowParallelLinear: 切分输入维
class RowParallelLinear:
    def forward(self, x):
        # x: [B, d_in/N], weight: [d_in/N, d_out]
        output = F.linear(x, self.weight)  # [B, d_out]
        output = all_reduce(output)  # 求和所有GPU的部分结果
        return output
```

## 4. Pipeline Parallelism (PP)

### 4.1 三种调度

```
模型按层切分到不同 GPU (stage):

Stage 0: Layer 0-7    → GPU 0
Stage 1: Layer 8-15   → GPU 1
Stage 2: Layer 16-23  → GPU 2
Stage 3: Layer 24-31  → GPU 3

GPipe (同步, 简单):
  Forward: F0 → F1 → F2 → F3
  Backward: B3 → B2 → B1 → B0
  气泡: (P-1)/(P-1+M) → M=batch数
  M=8, P=4: 气泡 = 3/11 = 27%

1F1B (交错, 减少内存):
  F0 F1 F2 F3 B0 F0 F1 F2 F3 B1 B0 ...
  每完成一个 forward, 立即开始对应 backward
  气泡: (P-1)/(M+P-1)
  M=8, P=4: 气泡 = 3/10 = 30% (略高但内存少)

Interleaved 1F1B (设备交错):
  将模型分成 V 个 virtual stages:
  Stage 0: Layer 0-3, 16-19  (2个virtual stages)
  Stage 1: Layer 4-7, 20-23
  Stage 2: Layer 8-11, 24-27
  Stage 3: Layer 12-15, 28-31
  气泡: (P-1)/(M×V+P-1), V=virtual stages
  M=8, P=4, V=2: 气泡 = 3/18 = 17% (减半!)
```

### 4.2 PP 的通信

```
通信: P2P (点对点), 只在相邻 stage 之间
  通信量: B × d × dtype_size (activation tensor)

  B=32, d=4096, BF16: 32 × 4096 × 2 = 256 KB
  NVLink: 0.001ms (可以忽略)

PP vs TP 通信:
  PP: P2P, 256KB, 相邻GPU
  TP: AllReduce, 393KB per layer × 2, 所有GPU
  → PP 通信量远小于 TP!
  → PP 适合跨节点 (慢网络), TP 适合节点内 (NVLink)
```

## 5. Sequence Parallelism (SP)

### 5.1 原理 (Megatron-LM)

```
TP 中, LayerNorm 和 Dropout 在每个 GPU 上重复计算 (因为它们在 AllReduce 之前):
  标准 TP:
    x → LayerNorm(x) → Attention → AllReduce → LayerNorm → FFN → AllReduce

Sequence Parallelism:
  将 LayerNorm/Dropout 的 activation 按序列长度切分:
    每个 GPU 只存 1/TP 的序列

  通信替换:
    AllReduce = ReduceScatter + AllGather
    在 SP 中:
      前向: ReduceScatter (聚合后切分) → 计算 → AllGather (恢复)
      反向: AllGather → 计算 → ReduceScatter

  关键: 总通信量不变 (仍然是 AllReduce), 但:
    1. LayerNorm/Dropout 内存省 TP 倍 (只存 1/TP 的序列)
    2. LayerNorm/Dropout 计算量省 TP 倍
    3. 对于长序列特别有效
```

## 6. 3D 并行: DP + TP + PP 组合

### 6.1 组合策略

```
总 GPU 数 = DP × TP × PP

示例: 1024 GPU 训练 175B 模型
  TP = 8  (节点内, NVLink)
  PP = 8  (跨节点, InfiniBand)
  DP = 16 (剩余 GPU)
  总计: 8 × 8 × 16 = 1024 ✓

通信拓扑:
  节点内 (8 GPU, NVLink 300 GB/s):
    - TP 通信: AllReduce (2次/层)
  跨节点 (InfiniBand 200 Gbps):
    - PP 通信: P2P (activation)
    - DP 通信: AllReduce (梯度)

  关键: TP 通信量最大 → 必须在节点内 (NVLink)
        PP 通信量小 → 可以跨节点
        DP 通信量中等 → 跨节点, 但可以和计算重叠
```

### 6.2 内存分配

```
175B BF16 模型, TP=8, PP=8, ZeRO-1:

每个 GPU:
  模型参数: 175B × 2B / (TP × PP) = 350GB / 64 = 5.5GB
  梯度: 350GB / 64 = 5.5GB
  优化器: 175B × 12B / (TP × PP × DP) = 2.1TB / (64 × 16) = 2.0GB
  激活: ~2GB (activation checkpointing)
  总计: ~15GB → A100 80GB 充裕
```

## 7. 工程实践: 如何选择并行策略

### 7.1 决策树

```
模型 < 单GPU内存?
  ✓ → 纯 DP (最简单, 最高效)
  ✗ → 继续

模型 < 节点内 GPU 总内存?
  ✓ → TP + DP
     - TP=2-8 (根据模型大小)
     - DP = 剩余GPU
  ✗ → 继续

模型 < 所有GPU总内存?
  ✓ → TP + PP + DP (+ ZeRO)
     - TP = 节点内GPU数
     - PP = 需要跨节点数
     - DP = 剩余GPU
     - + ZeRO-1/2/3 (进一步节省)
  ✗ → 需要 ZeRO-3 + TP + PP + DP + activation checkpointing
```

### 7.2 通信瓶颈

```
不同网络的通信延迟 (微秒):
┌──────────────────────────────────────────────────────────┐
│ 类型           │ 带宽       │ 延迟    │ 适合的并行      │
│────────────────│────────────│─────────│─────────────────│
│ NVLink (节点内)│ 300 GB/s   │ ~1μs    │ TP, EP          │
│ PCIe (节点内)  │ 32 GB/s    │ ~5μs    │ 有限 TP         │
│ InfiniBand     │ 200 Gbps   │ ~2μs    │ PP, DP, ZeRO    │
│ Ethernet       │ 25 Gbps    │ ~50μs   │ DP only         │
└──────────────────────────────────────────────────────────┘

规则:
  AllReduce/All-to-All → 必须在 NVLink/IB 上
  P2P (小数据) → 可以跨 Ethernet
  TP 每层2次AllReduce → 必须NVLink
  PP 每stage 1次P2P → 可以跨IB
  DP 每step 1次AllReduce → 可以跨IB+overlap
```

### 7.3 实际框架对比

```
┌─────────────────────────────────────────────────────────────────┐
│ 框架          │ TP │ PP │ DP/ZeRO │ SP │ EP │ 特点            │
│───────────────│────│────│─────────│────│────│─────────────────│
│ Megatron-LM   │ ✓  │ ✓  │ ✓       │ ✓  │ ✓  │ 最成熟, 研究    │
│ DeepSpeed     │ ✓  │ ✓  │ ✓(ZeRO) │ ✗  │ ✓  │ ZeRO 先驱      │
│ FSDP          │ ✗  │ ✗  │ ✓(内建) │ ✗  │ ✗  │ PyTorch 原生   │
│ verl          │ ✓  │ ✗  │ ✓(FSDP) │ ✗  │ ✗  │ RL 训练专用    │
│ Megatron-CC   │ ✓  │ ✓  │ ✓       │ ✓  │ ✓  │ Megatron + RL  │
└─────────────────────────────────────────────────────────────────┘
```

## 8. 最新发展: Context Parallelism (CP)

```
问题: 序列长度 > 单GPU内存 (如 128K, 1M tokens)

解决方案: Ring Attention (CP)
  将序列切分成 N/P 个 chunk, 分配到 P 个 GPU
  每个 GPU 计算 1/P 的 attention, 通过 P2P 传递 KV blocks

  通信: P2P 传递 KV chunks → 与计算重叠
  限制: 需要 causal mask 变通 (对左侧 block 需要特殊处理)

  实际效果:
    128K 序列, 8 GPU: 每GPU只需处理 16K 的 KV cache
    → 内存节省 8x, 通信被计算隐藏
```

## 9. 关键数字速查

```
7B 模型 (BF16):
  权重: 14 GB
  Adam 优化器: 28 GB (FP32 的 m, v)
  梯度: 14 GB
  总训练内存: ~56 GB (单GPU, 无ZeRO)

70B 模型 (BF16):
  权重: 140 GB → 需要 ZeRO-3 或 TP
  训练内存: ~560 GB

通信量估算:
  TP AllReduce/层: 4 × B × d × 2 × (TP-1)/TP bytes
  PP P2P/stage: B × d × 2 bytes
  DP AllReduce: 2 × (权重+梯度) × (DP-1)/DP bytes

  规律: 通信量 ∝ batch × hidden_dim × dtype_size
```
