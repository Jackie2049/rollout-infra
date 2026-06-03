# Megatron-LM 张量并行源码阅读笔记

> 基于 `megatron/core/tensor_parallel/` 的深度分析

## 1. 核心架构

Megatron-LM 的张量并行实现由两个关键文件组成：
- **`layers.py`** — 并行线性层（ColumnParallelLinear, RowParallelLinear）
- **`mappings.py`** — 通信原语（基于 autograd Function 的前向/反向通信）

## 2. 通信原语 (`mappings.py`)

### 核心设计：自定义 autograd Function

所有通信操作都实现为 `torch.autograd.Function` 的子类，**前向做通信，反向做对应的逆操作**：

| Autograd Function | Forward | Backward |
|---|---|---|
| `_CopyToModelParallelRegion` | Copy (无操作) | **AllReduce** |
| `_ReduceFromModelParallelRegion` | **AllReduce** | Copy (无操作) |
| `_ScatterToModelParallelRegion` | **Split (last dim)** | AllGather (last dim) |
| `_GatherFromModelParallelRegion` | AllGather (last dim) | **Split (last dim)** |
| `_ScatterToSequenceParallelRegion` | **Split (first dim)** | AllGather (first dim) |
| `_GatherFromSequenceParallelRegion` | AllGather (first dim) | **ReduceScatter (first dim)** |
| `_ReduceScatterToSequenceParallelRegion` | **ReduceScatter** | AllGather (first dim) |

**关键洞察**：前向和反向的通信操作是互补的！
- 如果前向做了 AllGather，反向就做 ReduceScatter（或 Split）
- 如果前向做了 Split，反向就做 AllGather
- 如果前向做了 Copy，反向就做 AllReduce

### 底层通信实现

```python
# 使用 PyTorch 原生的高效集合通信
dist_all_gather_func = torch.distributed.all_gather_into_tensor
dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
```

### AllReduce = ReduceScatter + AllGather

Megatron 优化了通信：不直接用 AllReduce，而是分解为 ReduceScatter + AllGather，可以与计算重叠。

## 3. 列并行线性层 (`ColumnParallelLinear`)

### 原理
```
Y = XA + b, 其中 A = [A_1, A_2, ..., A_p] 沿第二维度切分

GPU i: Y_i = X @ A_i + b_i  (局部计算)
如果 gather_output=True: Y = AllGather([Y_1, Y_2, ..., Y_p])
```

### 权重切分
```python
# A 被切分为 output_size_per_partition × input_size
self.output_size_per_partition = divide(output_size, world_size)
# rank i 拥有 A 的第 i 块
```

### 前向流程

```
输入 X (完整)
  │
  ├─ 如果需要 TP 通信: copy_to_tensor_model_parallel_region(X)
  │   (前向: copy, 反向: AllReduce dX)
  │
  ├─ 如果启用 SP: gather_from_sequence_parallel_region(X)
  │   (前向: AllGather(first dim), 反向: ReduceScatter(first dim))
  │
  ├─ X @ A_i + b_i → output_parallel (局部矩阵乘)
  │
  └─ 如果 gather_output=True: gather_from_tensor_model_parallel_region(output)
      (前向: AllGather(last dim), 反向: Split(last dim))
```

### 关键参数
- `gather_output`: 是否 AllGather 输出（让所有 GPU 都有完整输出）
- `sequence_parallel`: 启用序列并行时的输入处理
- `allreduce_dgrad`: 是否在反向时 AllReduce 输入梯度

## 4. 行并行线性层 (`RowParallelLinear`)

### 原理
```
Y = XA + b, A 沿第一维度切分, X 沿第二维度切分
A = [[A_1], [A_2], ..., [A_p]]  (按行切)
X = [X_1, X_2, ..., X_p]        (按列切)

GPU i: partial_i = X_i @ A_i  (部分结果)
Y = AllReduce(partial_1 + partial_2 + ... + partial_p) + b
```

### 前向流程

```
输入 X
  │
  ├─ 如果 input_is_parallel=False: scatter_to_tensor_model_parallel_region(X)
  │   (前向: Split(last dim), 反向: AllGather(last dim))
  │
  ├─ X_i @ self.weight (局部矩阵乘, weight 是 [output, input/TP])
  │
  └─ ReduceScatter 或 AllReduce 聚合结果
      ├─ SP: reduce_scatter_to_sequence_parallel_region(output)
      │   (前向: ReduceScatter(first dim), 反向: AllGather(first dim))
      └─ 非 SP: reduce_from_tensor_model_parallel_region(output)
          (前向: AllReduce, 反向: copy)
```

### 关键点
- `input_is_parallel=True` 时，输入已经是切分好的（来自上一步 ColumnParallelLinear）
- bias 只在 rank 0 上（不被切分）

## 5. Transformer 层中的 TP 通信模式

一个典型的 Transformer 层使用 2 次 AllReduce（或 ReduceScatter+AllGather）：

```
输入 X (所有 GPU 有完整副本)
  │
  ├─ QKV Linear (ColumnParallelLinear, gather_output=False)
  │   前向: 无通信
  │   反向: AllReduce dX
  │
  ├─ Attention (每 GPU 独立计算不同的 head)
  │
  ├─ Output Linear (RowParallelLinear, input_is_parallel=True)
  │   前向: AllReduce (或 ReduceScatter)  ← 通信点 1
  │   反向: 无通信
  │
  ├─ MLP: Up/Gate Linear (ColumnParallelLinear)
  │   前向: 无通信
  │   反向: AllReduce dX
  │
  ├─ Activation (gelu/silu, 每GPU独立)
  │
  ├─ MLP: Down Linear (RowParallelLinear, input_is_parallel=True)
  │   前向: AllReduce (或 ReduceScatter)  ← 通信点 2
  │   反向: 无通信
  │
  └─ 输出 Y (所有 GPU 有完整副本)
```

**每个 Transformer 层需要 2 次集合通信**（前向）。

## 6. 序列并行 (SP) 的通信变化

启用 SP 后：
- 输入按序列维度切分到各 GPU
- ColumnParallelLinear 前向：AllGather(seq_dim) 收集输入
- RowParallelLinear 前向：ReduceScatter(seq_dim) 分发输出
- LayerNorm / Dropout 在切分后的数据上操作，节省显存

通信量从 AllReduce 变为 AllGather + ReduceScatter，总量相同但可更好重叠。

## 7. AllToAll 通信

```python
# SP ↔ HP (Sequence Parallel ↔ Hidden Parallel) 转换
all_to_all_sp2hp: [num_tokens/TP, H] → [num_tokens, H/TP]
all_to_all_hp2sp: [num_tokens, H/TP] → [num_tokens/TP, H]
```

用于 2D/2.5D 张量并行或专家并行场景。

## 8. 关键文件索引

| 文件 | 内容 |
|------|------|
| `megatron/core/tensor_parallel/layers.py` | ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding |
| `megatron/core/tensor_parallel/mappings.py` | 所有通信 autograd Function |
| `megatron/core/tensor_parallel/utils.py` | split_tensor_along_last_dim 等工具函数 |
| `megatron/core/tensor_parallel/random.py` | TP 场景下的 CUDA RNG 状态管理 |
| `megatron/core/parallel_state.py` | Process Group 管理 |

## 9. 学习要点

1. **自定义 autograd Function 是实现通信-计算感知的关键** — 让 PyTorch autograd 自动在正确时机触发通信
2. **前向的逆操作就是反向** — AllGather ↔ ReduceScatter/Split, AllReduce ↔ Copy
3. **Column → Row 的组合是 TP 的精髓** — Column 并行无通信计算，Row 并行聚合结果
4. **异步通信与计算重叠** — `CUDA_DEVICE_MAX_CONNECTIONS=1` 确保 NCCL 通信先于计算 kernel 调度
5. **SP 的本质** — 用 ReduceScatter 替代 AllReduce，保持总通信量但改变分布方式
