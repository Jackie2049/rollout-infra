# Megatron-LM 流水线并行源码阅读笔记

> 基于 `megatron/core/pipeline_parallel/` 的深度分析

## 1. 核心架构

流水线并行将模型按层切分为多个 stage，每个 GPU 负责一个 stage。数据以 micro-batch 为粒度在 stage 间流动。

### 目录结构

```
pipeline_parallel/
├── schedules.py              # 调度算法核心（1F1B, interleaved, no-pipelining）
├── p2p_communication.py      # 点对点通信（send/recv）
├── combined_1f1b.py          # Combined 1F1B 调度（优化版）
├── hybrid_cp_schedule.py     # CP+PP 混合调度
├── bridge_communicator.py    # Bridge 通信器
├── multimodule_communicator.py  # 多模块通信器
├── fine_grained_activation_offload.py  # 激活细粒度 offload
└── utils.py                  # 工具函数
```

## 2. 调度算法选择

**入口函数**: `get_forward_backward_func()` (schedules.py:48)

```python
def get_forward_backward_func(pp_size, vp_size):
    if pp_size > 1:
        if vp_size is not None:
            # Interleaved PP (VPP > 1)
            return forward_backward_pipelining_with_interleaving
        else:
            # Standard 1F1B PP
            return forward_backward_pipelining_without_interleaving
    else:
        # No pipelining (single GPU)
        return forward_backward_no_pipelining
```

三种调度模式：
1. **No pipelining**: 单 GPU，无通信
2. **Non-interleaved 1F1B**: 每个 GPU 一个 stage，标准调度
3. **Interleaved 1F1B**: 每个 GPU 多个 model chunks (VPP)，减少 bubble

## 3. Non-Interleaved 1F1B 调度

**函数**: `forward_backward_pipelining_without_interleaving()` (schedules.py:2074)

### 3.1 Warmup 计算

```python
# Warmup microbatches 数量
num_warmup_microbatches = total_stages - current_stage - 1
# 例如: 4 stages, stage 0: 3 warmup, stage 3: 0 warmup
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```

### 3.2 三阶段流程

```
Phase 1: Warmup (纯前向)
┌─────────────────────────────────────────────┐
│ for i in range(num_warmup_microbatches):     │
│     input_tensor = recv_forward()            │
│     output_tensor = forward_step(input)      │
│     send_forward(output)                     │
│     save input_tensor, output_tensor          │
└─────────────────────────────────────────────┘

Phase 2: Steady State (1F1B — 1 前向 + 1 反向)
┌─────────────────────────────────────────────┐
│ input_tensor = recv_forward()                │
│ for i in range(num_microbatches_remaining):  │
│     output = forward_step(input)             │
│     output_grad = send_forward_recv_backward()│
│     push(input, output) to save list         │
│     pop(input, output) from save list        │
│     input_grad = backward_step(input, output)│
│     input_tensor = send_backward_recv_forward()│
└─────────────────────────────────────────────┘

Phase 3: Cooldown (纯反向)
┌─────────────────────────────────────────────┐
│ for i in range(num_warmup_microbatches):     │
│     input, output = pop from save list       │
│     output_grad = recv_backward()            │
│     input_grad = backward_step()             │
│     send_backward(input_grad)                │
└─────────────────────────────────────────────┘
```

### 3.3 时间线可视化 (4 stages, 8 microbatches)

```
Stage 0: [F0][F1][F2][F3][F4][B0][F5][B1][F6][B2][F7][B3][B4][B5][B6][B7]
Stage 1:    [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][B4][B5][B6][B7]
Stage 2:       [F0][F1][F2][B0][F3][B1][F4][B2][F5][B3][B4][B5][B6][B7]
Stage 3:          [F0][F1][B0][F2][B1][F3][B2][F4][B3][B4][B5][B6][B7]

Legend: Fn = forward microbatch n, Bn = backward microbatch n
```

**Pipeline Bubble**: warmup/cooldown 期间的空闲时间
- Bubble ratio ≈ (pp_size - 1) / num_microbatches
- 更多 microbatches → 更小的 bubble ratio

### 3.4 显存管理

```python
# 前向后立即释放 output tensor 的 .data（只保留 .grad_fn）
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
# output.data = torch.empty((1,))  # 释放大量显存！
```

这个优化很重要：output tensor 发送给下一个 stage 后，只需要它的 `grad_fn`（计算图），不需要 `.data`（实际张量数据）。将其替换为标量可节省大量显存。

### 3.5 梯度同步控制

```python
# 在 pipeline 执行期间禁用异步梯度同步
disable_grad_sync()

# 在最后一个 microbatch 的反向时启用
if last_iteration:
    enable_grad_sync()

# 确保所有梯度同步完成
if no_sync_context is not None:
    enable_grad_sync()
```

## 4. Interleaved 1F1B 调度 (Virtual Pipeline Parallelism)

**函数**: `forward_backward_pipelining_with_interleaving()` (schedules.py:931)

### 4.1 核心思想

将模型分为 **VPP × PP** 个 model chunks，每个 GPU 持有 VP 个 chunks：

```
模型有 16 层, PP=4, VPP=2:
  GPU 0: chunk_0 (L0-3), chunk_4 (L8-11)
  GPU 1: chunk_1 (L4-7), chunk_5 (L12-15)
  GPU 2: chunk_2 (L8-11 → 已分配给 GPU 0)

修正: 实际是交错分配
  GPU 0: chunk_0, chunk_2  (浅层 + 中层)
  GPU 1: chunk_1, chunk_3  (浅中层 + 中深层)

VPP 使 bubble 更小，因为每个 stage 可以更频繁地交替执行
```

### 4.2 关键变量

```python
num_microbatches    # 每 pipeline stage 的 microbatch 数
num_model_chunks    # = VPP (virtual pipeline size)
total_num_microbatches = num_microbatches * num_model_chunks

# 索引
microbatch_id       # [0, num_microbatches)
model_chunk_id      # [0, num_model_chunks)
virtual_microbatch_id  # [0, total_num_microbatches)
```

### 4.3 通信模式

Interleaved PP 需要更复杂的 P2P 通信：
- 每个 model chunk 都需要与前后的 chunk 通信
- 使用 `overlap_p2p_comm` 选项可以重叠通信与计算

```python
# Interleaved PP 支持的通信优化
config.overlap_p2p_comm  # 通信与计算重叠
config.batch_p2p_comm    # 批量 P2P 操作
# 注意: 两者不能同时使用
```

## 5. 点对点通信 (P2P)

**文件**: `p2p_communication.py`

### 5.1 通信原语

```python
# 批量 P2P 操作（同时发送和接收）
def _batched_p2p_ops(
    tensor_send_prev,     # 发送给前一个 stage
    tensor_recv_prev,     # 从前一个 stage 接收
    tensor_send_next,     # 发送给下一个 stage
    tensor_recv_next,     # 从下一个 stage 接收
    group,                # ProcessGroup
    prev_pipeline_rank,   # 前一个 stage 的 rank
    next_pipeline_rank,   # 下一个 stage 的 rank
):
    ops = []
    if tensor_send_prev is not None:
        ops.append(P2POp(isend, tensor_send_prev, prev_pipeline_rank, group))
    if tensor_recv_prev is not None:
        ops.append(P2POp(irecv, tensor_recv_prev, prev_pipeline_rank, group))
    # ... send_next, recv_next
    reqs = batch_isend_irecv(ops)  # 批量提交所有操作
    return reqs
```

### 5.2 偶奇交替发送

```python
# 为避免死锁，偶数 rank 先发送，奇数 rank 先接收
if group.rank() % 2 == 0:
    send_first, then recv
else:
    recv_first, then send

# 特殊情况: group.size() == 2 时
# 使用 global ProcessGroup 让两个方向同时进行
```

### 5.3 P2PCommunicator 封装

```python
class P2PCommunicator:
    def recv_forward(self, shapes, is_first_stage)
    def send_forward(self, tensor, is_last_stage)
    def recv_backward(self, shapes, is_last_stage)
    def send_backward(self, tensor, is_first_stage)

    # 组合操作（减少同步等待）
    def send_forward_recv_backward(self, ...)
    def send_backward_recv_forward(self, ...)
```

## 6. Pipeline Bubble 分析

### 6.1 Standard 1F1B

```
Bubble ratio = (pp_size - 1) / num_microbatches

例如: pp_size=4, num_microbatches=8
Bubble = 3/8 = 37.5%

pp_size=8, num_microbatches=16
Bubble = 7/16 = 43.75%
```

### 6.2 Interleaved 1F1B

```
Bubble ratio = (pp_size - 1) / (num_microbatches × num_model_chunks)

例如: pp_size=4, num_microbatches=8, VPP=2
Bubble = 3/16 = 18.75%  ← 比 standard 减半！
```

**代价**: 通信次数增加 VPP 倍，需要更高带宽。

## 7. 显存优化

### 7.1 Activation Checkpointing

```python
# 部分激活检查点：只在部分 microbatch 上做完整 checkpoint
if config.num_microbatches_with_partial_activation_checkpoints is not None:
    max_outstanding_backprops = num_warmup_microbatches + 1
    checkpoint_activations_microbatch = (
        i % max_outstanding_backprops
        >= config.num_microbatches_with_partial_activation_checkpoints
    )
```

### 7.2 Deallocate Output

```python
# 发送后立即释放 output tensor 的数据
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
# custom_backward() 绕过 shape 检查，允许用标量 grad_fn 反向传播
```

### 7.3 Fine-Grained Activation Offload

支持将激活值 offload 到 CPU，进一步减少 GPU 显存占用。

## 8. 关键文件索引

| 文件 | 内容 |
|------|------|
| `pipeline_parallel/schedules.py` | 1F1B 调度（non-interleaved + interleaved + no-pipelining） |
| `pipeline_parallel/p2p_communication.py` | P2P 通信原语（isend/irecv/batch） |
| `pipeline_parallel/combined_1f1b.py` | Combined 1F1B 优化调度 |
| `pipeline_parallel/hybrid_cp_schedule.py` | Context Parallel + PP 混合调度 |
| `pipeline_parallel/utils.py` | 工具函数（is_pp_first/last_stage） |
| `transformer/pipeline_parallel_layer_layout.py` | 自定义层分配配置 |
| `pipeline_parallel/multimodule_communicator.py` | 多模块（encoder+decoder）通信 |

## 9. 学习要点

1. **1F1B 三阶段**：Warmup (纯前向) → Steady State (1前向+1反向) → Cooldown (纯反向)
2. **Pipeline Bubble 是主要开销** — 更多 microbatches 和 interleaving 可减少
3. **Deallocate output 是关键显存优化** — 只保留 grad_fn，释放 .data
4. **偶奇交替避免死锁** — P2P 通信的 rank ordering
5. **Interleaved PP 减少 bubble 但增加通信** — VPP=2 时 bubble 减半，但通信次数翻倍
6. **Combined 1F1B** 是新优化版本，进一步减少 bubble
7. **PP 与 TP 正交互补** — TP 切单层，PP 切层间

## 参考

- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (Interleaved PP)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
