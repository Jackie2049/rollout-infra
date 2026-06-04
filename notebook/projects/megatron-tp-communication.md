# Megatron-LM Tensor Parallelism 通信分析

> 文件: `megatron/core/tensor_parallel/layers.py`
> 分析日期: 2026-06-04

## ColumnParallelLinear vs RowParallelLinear

TP 将单个 Linear layer 沿列或行切分到多个 GPU:

```
Linear: Y = X @ W    [M,K] @ [K,N] → [M,N]

ColumnParallel (切 W 的列):
┌───────────────┐
│ [M, K]        │  X (全部在每个 rank)
│ @ [K, N/P]    │  W 按列切分
│ = [M, N/P]    │  Y_i = 局部输出
└───────────────┘
  → AllGather → [M, N]  (如果需要完整输出)
  → 无 AllReduce (fwd: 0 通信)!

RowParallel (切 W 的行):
┌───────────────┐
│ [M, K/P]      │  X_i = 输入已经切分 (来自 ColumnParallel 或 SP)
│ @ [K/P, N]    │  W 按行切分
│ = [M, N]_i    │  每个 rank 产生部分和
└───────────────┘
  → AllReduce → [M, N]  (fwd: 1 次 AllReduce)
```

## 通信模式

### ColumnParallelLinear.forward()
```
if sequence_parallel 或 allreduce_dgrad:
    input_parallel = input_  # 已经切分好
else:
    input_parallel = copy_to_tp_region(input)  # 前向: identity, 反向: AllReduce

output_parallel = X @ W_i  # 本地 GEMM, 无通信

if gather_output:
    output = AllGather(output_parallel)  # 1 次 AllGather
else:
    output = output_parallel            # 保持切分 (给下一个 RowParallel)
```

### RowParallelLinear.forward()
```
output_partial = X_i @ W_i  # 本地 GEMM, 无通信
output = AllReduce(output_partial)  # 1 次 AllReduce (必须, 因为是部分和)
```

### 组合: ColumnParallel → RowParallel

```
[fwd]
ColumnParallel:    X @ W_col_i    → output_col_i  (0 通信)
  ↓ (保持切分, gather_output=False)
RowParallel:       output_col_i @ W_row_i  → partial_sum
  ↓
AllReduce(partial_sum)  → final_output  (1 次 AllReduce)

总计: 每 Transformer 层 1 次 AllReduce (在 RowParallel 输出处)
```

### 反向传播通信

```
[bwd]
RowParallel:
  输出 grad 已经 AllReduce → 无额外通信
  输入 grad: copy_to_tp_region 的反向 = AllReduce

ColumnParallel:
  输入 grad: AllReduce (allreduce_dgrad=True 时)
  或: copy_to_tp_region 的反向

总计: 每层 fwd 1 通信 + bwd 1 通信 = 每层 2 次 AllReduce
```

## Sequence Parallelism 通信

```
[fwd]
[序列切分输入] → AllGather → ColumnParallel → RowParallel → ReduceScatter → [序列切分输出]

通信: 1 AllGather + 1 ReduceScatter = 1 AllReduce 当量
      (但内存节省 TPx 倍!)
```

## 梯度同步优化

### allreduce_dgrad (ColumnParallel)
```python
if allreduce_dgrad:
    # 前向: 不做 copy_to_tp_region (input 保持不切分)
    # 反向: 手工 AllReduce 输入梯度
    input_parallel = input_  # no-op copy
```

### gradient_accumulation_fusion
```python
# 将 AllReduce gradient + GEMM backward 合并成单 CUDA kernel
# 节省 roundtrip latency
```

### disable_grad_reduce (LoRA)
```python
# 延迟 AllReduce, 让 LoRA adapter 自己管理梯度归约时机
# 与 LoRA gradient 合并 → 更少同步点
```

## 异步通信优化

```python
# async_op=True: AllReduce 和下一个计算 overlap
handle = AllReduce(tensor, async_op=True)
# ... 其他计算 ...
handle.wait()  # 确保通信完成
```

需要 `CUDA_DEVICE_MAX_CONNECTIONS=1` 确保通信先调度到 stream。

## TP 通信开销分析

对于 70B 模型, TP=8:

```
每层参数: ~12B / 80 layers = 150M
输出维度: hidden=8192, batch=1, seq=2048

ColumnParallel 输出: [2048, 8192/8]=[2048, 1024] × FP16 = 4MB
RowParallel AllReduce: [2048, 8192] × FP16 = 32MB

AllReduce 时间:
- NVLink (300GB/s, 8-GPU ring): 32MB × 2 × (8-1)/8 / 300e9 = ~0.19ms
- 每层计算: ~7ms (70B on A100)
- 通信占比: 0.19 / 7 = 2.7%
```

结论: TP 通信开销 <5% (NVLink 场景)，在小 batch 下略高。

## 代码位置速查

| 类 | 行号 | 通信 |
|----|:---:|------|
| `ColumnParallelLinear` | L770- | fwd: AllGather (optional) |
| `RowParallelLinear` | L1134- | fwd: AllReduce (required) |
| `copy_to_tp_region` | mappings.py | fwd: identity, bwd: AllReduce |
| `reduce_from_tp_region` | mappings.py | fwd: AllReduce, bwd: identity |

## 关键映射函数

| 函数 | Fwd 行为 | Bwd 行为 |
|------|---------|---------|
| `copy_to_tp_region` | copy (no-op if 1 rank) | **AllReduce** grad |
| `reduce_from_tp_region` | **AllReduce** | copy (no-op) |
| `gather_from_tp_region` | **AllGather** | **ReduceScatter** grad |
| `scatter_to_tp_region` | 切分输入 | **AllGather** grad |
