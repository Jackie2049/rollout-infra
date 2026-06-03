# Megatron-LM 张量并行源码深度阅读

> Megatron-Core TP 实现的完整数据流、通信原语、权重分发、3D 并行交互

## 1. 核心文件

**路径**: `megatron/core/tensor_parallel/`

| 文件 | 职责 |
|------|------|
| `mappings.py` | 通信原语 (AllReduce/AllGather/ReduceScatter/AlltoAll)，autograd 集成 |
| `layers.py` | ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding |
| `random.py` | TP 感知 RNG 状态管理 (CudaRNGStatesTracker) |
| `cross_entropy.py` | VocabParallelCrossEntropy (分片 logits 的 softmax) |

## 2. 通信原语 (mappings.py)

每个通信操作是 `torch.autograd.Function` 子类，正向/反向自动配对：

| 原语 | 正向 | 反向 | 用途 |
|------|------|------|------|
| `_CopyToModelParallelRegion` | Identity | All-Reduce | ColumnParallel 输入 |
| `_ReduceFromModelParallelRegion` | All-Reduce | Identity | RowParallel 输出 |
| `_ScatterToModelParallelRegion` | Split(last_dim) | All-Gather(last_dim) | RowParallel 输入 |
| `_GatherFromModelParallelRegion` | All-Gather(last_dim) | Split(last_dim) | ColumnParallel 输出 |

**序列并行 (SP) 扩展**:
| 原语 | 正向 | 反向 |
|------|------|------|
| `_ScatterToSequenceParallelRegion` | Split(first_dim) | All-Gather(first_dim) |
| `_GatherFromSequenceParallelRegion` | All-Gather(first_dim) | Reduce-Scatter(first_dim) |

## 3. TP Transformer 层数据流

### 3.1 Attention 块

```
输入: [seq, batch, hidden_size]
         │
    ┌────┴────┐
    │ QKV 投影 │  ColumnParallelLinear (gather_output=False)
    │ 权重分割  │  权重: [(Q+K+V)/TP, hidden] per rank
    └────┬────┘
         │ [seq, batch, (Q+K+V)/TP]
    ┌────┴────┐
    │ Split QKV │  分离 Q, K, V
    │ + GQA处理 │  KV heads < TP 时 all-gather
    └────┬────┘
         │
    ┌────┴────┐
    │ Attention │  FlashAttention, 纯本地计算
    │ (本地 heads)│  无 TP 通信
    └────┬────┘
         │
    ┌────┴────┐
    │ Output 投影│  RowParallelLinear (input_is_parallel=True)
    │ All-Reduce │  或 Reduce-Scatter (SP)
    └────┬────┘
         │ [seq, batch, hidden_size]  ← 恢复完整维度
```

### 3.2 MLP 块

```
输入: [seq, batch, hidden_size]
         │
    ┌────┴────┐
    │ FC1 (gate+up)│ ColumnParallelLinear (gather_output=False)
    │ 权重: [2*ffn/TP, hidden] │ SwiGLU: stride=2 交错
    └────┬────┘
         │
    ┌────┴────┐
    │ SwiGLU 激活│  本地计算
    └────┬────┘
         │
    ┌────┴────┐
    │ FC2 (down)  │ RowParallelLinear (input_is_parallel=True)
    │ All-Reduce  │  或 Reduce-Scatter (SP)
    └────┬────┘
         │ [seq, batch, hidden_size]
```

### 3.3 每层通信总结

| 模式 | 每层通信 |
|------|---------|
| TP-only | 2× All-Reduce (attention output + MLP FC2) |
| TP+SP | 2× Reduce-Scatter (forward) + 2× All-Gather (backward) + 2× All-Gather (ColumnParallel forward) + 2× Reduce-Scatter (backward) |

## 4. ColumnParallelLinear vs RowParallelLinear

### 4.1 ColumnParallelLinear

```
权重形状: [output_size/TP, input_size]  — 沿输出维度分割
输入: 完整 hidden
输出: 每卡 output_size/TP 列

Forward:
  1. copy_to_tensor_model_parallel_region(input)  # fwd: id, bwd: All-Reduce
  2. output = input @ weight.T  # 本地 matmul
  3. if gather_output: gather_from_tp_region(output)  # All-Gather

Backward:
  1. grad_input = grad_output @ weight  # 本地
  2. if SP: Reduce-Scatter grad_input
  3. grad_weight = grad_output.T @ input  # 本地
```

### 4.2 RowParallelLinear

```
权重形状: [output_size, input_size/TP]  — 沿输入维度分割
输入: 每卡 input_size/TP
输出: 部分和 → All-Reduce

Forward:
  1. if not input_is_parallel: scatter input along last_dim
  2. output_partial = input @ weight.T  # 本地 matmul
  3. All-Reduce (或 SP: Reduce-Scatter) → 完整输出

Backward:
  1. grad_input = grad_output @ weight  # 每卡获取自己的梯度
  2. 反向自动由 autograd Function 处理
```

## 5. 通信-计算重叠

关键优化: **`CUDA_DEVICE_MAX_CONNECTIONS=1`** 强制串行化 NCCL 操作，使异步 All-Reduce/Reduce-Scatter 在权重梯度计算之前被调度:

```
Backward (ColumnParallel):
  ┌───────────────────┐
  │ Async All-Reduce   │ ← grad_input 通信
  │ grad_input (启动)  │
  └────────┬──────────┘
           │ 与计算重叠
  ┌────────┴──────────┐
  │ grad_weight = ... │ ← 权重梯度计算
  └───────────────────┘
```

## 6. 权重初始化

### 6.1 CPU 初始化 (调试用)

```python
initialize_affine_weight_cpu(weight, ...)
  # 1. CPU 上创建完整权重 [output_size, input_size]
  # 2. init_method 初始化
  # 3. 按 partition_dim 分割
  # 4. 复制本地分片到 GPU
```

### 6.2 GPU 初始化 (生产用)

```python
initialize_affine_weight_gpu(weight, ...)
  # 1. 设置 TP 元数据: partition_dim, partition_stride
  # 2. fork RNG state (TP rank → 不同种子)
  # 3. 每个 GPU 独立初始化自己的分片
```

### 6.3 TP 元数据属性

```python
weight.tensor_model_parallel = True
weight.partition_dim = 0  # Column=0, Row=1
weight.partition_stride = 1  # SwiGLU 交错时为 2
```

## 7. TP 感知随机状态

### 7.1 三类 RNG

| 状态 | 种子公式 | 范围 |
|------|---------|------|
| `data-parallel-rng` | `seed` | TP 组内相同, DP 间不同 |
| `model-parallel-rng` | `seed + 2718 + tp_rank` | TP 组间不同 |
| `expert-parallel-rng` | `seed + 1024 + 100*ep_rank + etp_rank` | Expert 层专用 |

**设计**: TP 区域外的 dropout 用 DP RNG (相同 → All-Reduce 正确)。TP 区域内的 dropout 用 TP RNG (不同 → 不影响 All-Reduce)。

### 7.2 激活检查点

```python
CheckpointFunction:
  forward: 保存 RNG state → 正常计算 → 丢弃中间激活
  backward: 恢复 RNG state → 重计算 → 确保 dropout 掩码一致
```

## 8. 3D 并行交互

### 8.1 Rank 映射

```
global_rank = tp_rank + dp_rank × tp_size + pp_rank × tp_size × dp_size
```

正交分解:
- **TP 组**: 同一 PP stage + DP 副本内的 GPU
- **PP 组**: 同一 TP rank + DP 副本内的 GPU
- **DP 组**: 同一 TP rank + PP stage 内的 GPU

### 8.2 TP-PP 交互

- **PP stage 内**: 完整 TP 通信 (2× All-Reduce/layer)
- **跨 PP stage**: `isend/irecv` 点对点, TP 通信仅在 stage 内
- **虚拟流水线 (VP)**: 多个 model chunk 共享一个 PP stage, 交错 1F1B 减少气泡

### 8.3 Context Parallel (CP)

- `_TENSOR_AND_CONTEXT_PARALLEL_GROUP` 结合 TP + CP
- CP 沿序列维度分割, CP ranks 间需要 KV 交换
- 与 Ring Attention / Ulysses 等序列并行策略配合

## 9. VocabParallelEmbedding + CrossEntropy

### 9.1 Embedding

```
词表分片: 每个 GPU 持有 vocab_size/TP 个 embedding
Forward: 本地查找 + 置零越界索引 + All-Reduce
Backward: 只为本地分片累积梯度
```

### 9.2 CrossEntropy

```
分片 logits: [seq, batch, vocab_size/TP]
3× All-Reduce:
  1. logits_max (MAX 操作, 数值稳定)
  2. predicted_logits (SUM)
  3. sum_exp_logits (SUM)
→ 正确计算完整 softmax + CE, 无需 gather 全部 logits
```

## 10. 关键洞察

1. **Column+Row 配对**: ColumnParallel 产生分片结果, RowParallel 消费分片并 All-Reduce, 自然的通信优化
2. **Autograd 集成**: 通信操作作为 Function 子类, 自动处理反向传播, 开发者无需手动管理
3. **通信-计算重叠**: `CUDA_DEVICE_MAX_CONNECTIONS=1` 强制异步通信先调度, 与梯度计算重叠
4. **TP RNG 分叉**: 确保 TP 区域内的 dropout 不破坏 All-Reduce 的正确性
5. **SP 扩展**: 将 All-Reduce 替换为 All-Gather + Reduce-Scatter, LayerNorm 在分片上独立运行

## 参考资料

- 源码路径: `Megatron-LM/megatron/core/tensor_parallel/`
- 相关笔记: [张量并行](../fundamentals/distributed-training.md), [NCCL](../fundamentals/nccl.md), [通信重叠](../fundamentals/comm-compute-overlap.md)
