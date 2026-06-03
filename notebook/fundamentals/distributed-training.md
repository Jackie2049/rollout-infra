# 分布式训练核心概念

> 从单卡到万卡：理解大模型分布式训练的并行策略

## 1. 为什么需要分布式训练？

一个 7B 参数的模型：
- **模型参数**: 7B × 2 bytes (BF16) = 14 GB
- **梯度**: 同样 14 GB
- **优化器状态 (AdamW)**: 7B × 8 bytes (m + v, FP32) = 56 GB
- **激活值**: 取决于 batch size 和序列长度，可能 10-50 GB
- **总计**: ~100+ GB

单个 A100 只有 80GB HBM。训练 70B+ 模型必须分布式。

---

## 2. 数据并行 (Data Parallelism)

### 原理
- 每张 GPU 持有模型的**完整副本**
- 数据被切分为 N 份，每张 GPU 处理不同的 micro-batch
- 前向独立计算，反向需要同步梯度

### 梯度同步：AllReduce
```
GPU 0: grad_0 ──┐
GPU 1: grad_1 ──┤── AllReduce ──→ avg_grad ──→ 所有 GPU 更新参数
GPU 2: grad_2 ──┤
GPU 3: grad_3 ──┘
```

### Ring AllReduce 算法
1. **Reduce-Scatter 阶段**：N-1 步，每步每个 GPU 发送 1/N 数据给下一个
2. **AllGather 阶段**：N-1 步，每步每个 GPU 发送聚合后的 1/N 数据

通信量：`2 × (N-1)/N × 模型参数量 × sizeof(dtype)`

### DDP (DistributedDataParallel)
- PyTorch 原生实现
- 每个 rank 独立前向，反向时通过 AllReduce 同步梯度
- 使用 buckets 优化：将梯度分组，计算完一组就通信一组（计算通信重叠）

### 局限
- 每张 GPU 必须装下完整模型 + 优化器状态 + 激活
- 通信量与 GPU 数量线性相关（但 per-GPU 通信量恒定）

---

## 3. ZeRO 优化器 (Zero Redundancy Optimizer)

### 核心思想：消除数据并行中的冗余

DP 的冗余：每张 GPU 都存一份完整的优化器状态、梯度、参数。

| ZeRO Stage | 分片内容 | 每卡显存 | 通信量 |
|-----------|---------|---------|--------|
| Stage 0 | 无（纯 DP） | 全量 | 1× |
| Stage 1 | 优化器状态 | ~1/4 全量 | 1× |
| Stage 2 | 优化器状态 + 梯度 | ~1/8 全量 | 1.5× |
| Stage 3 | 优化器状态 + 梯度 + 参数 | ~1/N 全量 | 1.5× (param gather) |

### ZeRO-3 的工作流程
1. **前向**：需要某一层参数时，通过 AllGather 从其他 rank 收集 → 用完释放
2. **反向**：同上，逐层收集参数计算梯度
3. **梯度**：ReduceScatter 到拥有该参数分片的 rank
4. **更新**：每个 rank 只更新自己负责的参数分片

### ZeRO vs FSDP
- FSDP 是 PyTorch 原生实现的 ZeRO-3
- 接口更简洁，与 PyTorch 生态更好集成
- 性能相当，但 FSDP 对 transformer 模型优化更好

---

## 4. 张量并行 (Tensor Parallelism)

### 核心思想：把单个矩阵乘法切到多张 GPU

### Megatron-LM 的列并行 + 行并行

**Attention 层的切分：**
```
Q = X @ Wq  →  [Q1, Q2] = X @ [Wq1, Wq2]  (列并行，切 output)
K = X @ Wk  →  同上
V = X @ Wv  →  同上

# 每个 GPU 独立计算 attention（不同的 head）
Attn_i = softmax(Qi @ Ki^T / sqrt(d)) @ Vi

# 输出投影（行并行，切 input）
Out = [Attn_1, Attn_2] @ [Wo_1; Wo_2]  (行并行)
# 每个 GPU 算部分结果，然后 AllReduce 求和
```

**MLP 层的切分：**
```
# 第一层线性：列并行
[h1, h2] = X @ [W1, W2]

# 每个 GPU 独立做激活函数
a_i = gelu(h_i)

# 第二层线性：行并行
out = [a1, a2] @ [U1; U2]  + AllReduce
```

### 通信模式
- 每个 Transformer 层需要 **2 次 AllReduce**
- AllReduce = ReduceScatter + AllGather
- 通信量：`4 × batch_size × seq_len × hidden_dim × sizeof(dtype) × (TP-1)/TP`

### 关键约束
- TP 需要高带宽互联（NVLink），通常在**同一节点内**
- 超过节点边界时，通信延迟会成为瓶颈

---

## 5. 流水线并行 (Pipeline Parallelism)

### 核心思想：按层切分模型到不同 GPU

```
GPU 0: Layer 0-5     ──→ GPU 1: Layer 6-11  ──→ GPU 2: Layer 12-17 ──→ GPU 3: Layer 18-23
       Stage 0               Stage 1                 Stage 2                  Stage 3
```

### GPipe
- 将 batch 切分为 M 个 micro-batch
- 所有 micro-batch 做完前向，再依次做反向
- 气泡（bubble）比例：`(P-1)/P`（P = pipeline stages 数）

```
时间 →
GPU 0: [F0][F1][F2][F3]           [B3][B2][B1][B0]
GPU 1:    [F0][F1][F2][F3]     [B3][B2][B1][B0]
GPU 2:       [F0][F1][F2][F3]  [B3][B2][B1][B0]
GPU 3:          [F0][F1][F2][F3][B3][B2][B1][B0]
                    ↑ 气泡 ↑
```

### 1F1B (One Forward One Backward)
- 交替执行一个 micro-batch 的前向和反向
- 减少显存占用（不需要同时保存所有 micro-batch 的激活）
- 气泡比例仍然是 `(P-1)/P`，但峰值显存更低

```
时间 →
GPU 0: [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 1:    [F0][F1][F2][B0][F3][B1][B2][B3]
GPU 2:       [F0][F1][B0][F2][B1][F3][B2][B3]
GPU 3:          [F0][B0][F1][B1][F2][B2][F3][B3]
```

### Interleaved 1F1B (Virtual Pipeline Stages)
- 每个 GPU 负责**不连续**的多组层（如 GPU 0 负责 layer 0-2 和 12-14）
- 更小的 virtual stage → 更小的气泡
- 气泡比例降至 `(P-1)/(M+P-1)`

---

## 6. 序列并行 / 上下文并行

### 序列并行 (Sequence Parallelism)
- Megatron-LM 的 SP：在 TP 基础上，把 LayerNorm 和 Dropout 的激活也按序列维度切分
- 这些操作不需要跨 GPU 通信，切分后减少激活显存
- 在需要做 attention 时用 AllGather 收集完整序列

### 上下文并行 (Context Parallelism)
- 专门解决**超长序列**训练问题
- 将序列按 token 切分到不同 GPU
- 关键技术：**Ring Attention** — 环形传递 Q/K/V blocks，实现线性复杂度的注意力计算

```
GPU 0: tokens [0:T/4]    ──→ Q0, K0, V0
GPU 1: tokens [T/4:T/2]  ──→ Q1, K1, V1
GPU 2: tokens [T/2:3T/4] ──→ Q2, K2, V2
GPU 3: tokens [3T/4:T]   ──→ Q3, K3, V3

Ring Attention: 每步环形传递 KV block，每个 GPU 计算部分 attention
Step 0: GPU_i 用 (Q_i, K_i, V_i)
Step 1: GPU_i 用 (Q_i, K_{(i-1)%4}, V_{(i-1)%4})
... 直到所有 KV pair 都被计算
```

---

## 7. 混合并行策略

实际训练大模型时，需要组合多种并行策略：

```
                    节点 0 (8 × A100, NVLink)
            ┌──────────────────────────────────────┐
            │  TP = 2     │  TP = 2     │  TP = 2  │  ← 张量并行(节点内)
            │  PP=2       │  PP=2       │  PP=2    │  ← 流水线并行
            │  DP=3       │  DP=3       │  DP=3    │  ← 数据并行
            └──────────────────────────────────────┘
                  ↕ InfiniBand
            ┌──────────────────────────────────────┐
            │  节点 1 (同上结构)                      │
            └──────────────────────────────────────┘

总 GPU = TP × PP × DP = 2 × 2 × 3 × 2 nodes = 24 GPUs
```

### 典型配置示例
| 模型大小 | TP | PP | DP | 总 GPU |
|---------|----|----|----|----|
| 7B | 1 | 1 | 8 | 8 |
| 70B | 8 | 1 | 16 | 128 |
| 175B | 8 | 4 | 32 | 1024 |
| 1T+ | 8 | 8 | 128+ | 8192+ |

### 选择原则
- **TP**：用于节点内（NVLink 高带宽），通常不超过 8
- **PP**：当模型太大一个节点装不下时使用
- **DP**：尽可能大，提供最佳吞吐量扩展
- **ZeRO/FSDP**：与 DP 结合，减少每卡显存需求

---

## 8. 通信原语速查

| 原语 | 功能 | 典型用途 |
|------|------|----------|
| **AllReduce** | 所有 rank 求和后广播给所有 rank | DP 梯度同步、TP 层间通信 |
| **ReduceScatter** | 求和后每个 rank 只保留自己的分片 | ZeRO 梯度分片、TP 优化通信 |
| **AllGather** | 所有 rank 的分片拼接后广播给所有 | ZeRO 参数收集、SP 序列收集 |
| **Broadcast** | 一个 rank 的数据广播给所有 | 参数初始化、配置分发 |
| **Send/Recv** | 点对点通信 | PP 层间传递激活/梯度 |
| **AllToAll** | 每个 rank 给每个其他 rank 发不同数据 | MoE expert 通信、SP |

---

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473) (Megatron 1F1B)
- [Ring Attention with Blockwise Parallel FlashAttention](https://arxiv.org/abs/2310.01889)
- [PyTorch FSDP 论文](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf)
