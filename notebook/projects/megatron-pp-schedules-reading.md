# Megatron-LM Pipeline Parallel 调度源码阅读

> 文件: `megatron/core/pipeline_parallel/schedules.py` (~2444行)
> 分析日期: 2026-06-04

## 三种 Pipeline 调度

| 调度 | 函数 | 适用场景 |
|------|------|---------|
| **Non-Interleaved 1F1B** | `forward_backward_pipelining_without_interleaving` | 标准 PP，模型不切分 |
| **Interleaved 1F1B** | `forward_backward_pipelining_with_interleaving` | VP (Virtual Pipeline)，模型切为多个 chunk |
| **Combined 1F1B** | `combined_1f1b_schedule_*` | 混合调度，支持更灵活的重叠 |

---

## Non-Interleaved 1F1B (标准 Pipeline)

### 核心三阶段

```
Warmup Forward        1F1B Steady State          Cooldown Backward
├─ F0 ┤              ├─ F3 + B0 ┤               ├─ B1 ┤
├─ F1 ┤              ├─ F4 + B1 ┤               ├─ B2 ┤
├─ F2 ┤              ├─ F5 + B2 ┤               ├─ B3 ┤
     ↑                    ↑                         ↑
  纯forward        forward + backward 交替        纯backward
```

### 代码实现 (L2268-2410)

```python
# 1. Warmup: 纯 forward
num_warmup_microbatches = total_stages - current_stage - 1
for i in range(num_warmup_microbatches):
    input_tensor = p2p.recv_forward(...)
    output_tensor = forward_step(...)
    p2p.send_forward(output_tensor)
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)

# 2. Steady State: 1 Forward + 1 Backward
for i in range(num_microbatches_remaining):
    output_tensor = forward_step(...)
    output_tensor_grad = p2p.send_forward_recv_backward(output_tensor)

    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)
    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

    if last_iteration:
        p2p.send_backward(input_tensor_grad)
    else:
        input_tensor = p2p.send_backward_recv_forward(input_tensor_grad)

# 3. Cooldown: 纯 backward
for i in range(num_warmup_microbatches):
    input_tensor = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)
    output_tensor_grad = p2p.recv_backward(...)
    input_tensor_grad = backward_step(...)
    p2p.send_backward(input_tensor_grad)
```

### 关键公式

- **Warmup microbatches** = `total_stages - current_stage - 1`
  - 第0stage (first): warmup = stages - 1 (最多)
  - 最后stage: warmup = 0
- **Steady state microbatches** = `num_microbatches - num_warmup_microbatches`
- **Bubble** ≈ `(stages - 1) / (microbatches + stages - 1)`
  - 当 `microbatches >> stages` 时 bubble → 0

### P2P 通信原语

| 函数 | 操作 | 适用阶段 |
|------|------|---------|
| `recv_forward` | 从上一stage接收activation | Warmup |
| `send_forward` | 向下一stage发送activation | Warmup |
| `send_forward_recv_backward` | 同时发送fwd + 接收bwd grad | 1F1B steady |
| `send_backward_recv_forward` | 同时发送bwd + 接收fwd activation | 1F1B steady |
| `send_backward` | 向上一stage发送grad | Cooldown/最后 |
| `recv_backward` | 从下一stage接收grad | Cooldown |

**组合通信**减少同步点: `send_forward_recv_backward` 和 `send_backward_recv_forward` 将两次P2P合并为一次，降低latency。

### Activation Checkpointing 策略

```python
max_outstanding_backprops = num_warmup_microbatches + 1
checkpoint = (microbatch_id % max_outstanding_backprops) >= num_partial_checkpoints
```

- 基于当前 "in-flight" 的反向传播数量动态决定是否checkpoint
- 参考论文: [Megatron-LM Appendix C](https://arxiv.org/pdf/2205.05198.pdf)

### 内存优化

```python
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

- 在warmup期间立即释放output_tensor（因为已经send_forward出去了）
- 减少peak activation内存

---

## Interleaved 1F1B (Virtual Pipeline)

### 概念

将模型沿layer维度切分为 `num_model_chunks` (virtual pipeline size)，每个GPU持有多个不相交的layer chunk:

```
Stage 0: [Chunk 0: layers 0-3] [Chunk 2: layers 8-11]
Stage 1: [Chunk 1: layers 4-7] [Chunk 3: layers 12-15]
```

### 优势

- **更小的bubble**: 因为 `effective_microbatches = num_microbatches * num_model_chunks`
- **更好的内存平衡**: 每个chunk更小，activation内存更均匀

### 复杂度

- 需要维护 `model_chunk_id` 和 `virtual_microbatch_id`
- 每个chunk有独立的 `data_iterator`
- 调度顺序更复杂 (chunk 0 forward → chunk 1 forward → chunk 0 backward → chunk 1 backward)

---

## P2P Communication 实现

文件: `megatron/core/pipeline_parallel/p2p_communication.py`

### ring_exchange (最快)

```python
# 同时send和recv，避免死锁
send_op = dist.P2POp(dist.isend, tensor, next_rank)
recv_op = dist.P2POp(dist.irecv, tensor, prev_rank)
dist.batch_isend_irecv([send_op, recv_op])
```

### 张量形状协商

```python
recv_tensor_shapes = get_tensor_shapes(
    seq_length=seq_length,
    micro_batch_size=micro_batch_size,
    config=config,
    tp_group=tp_group,
    cp_group=cp_group,
)
```

- 考虑 TP/CP 切分后的实际形状
- 支持 variable_seq_lengths (不同microbatch不同长度)

---

## 与 UBatch (vLLM) 的对比

| 维度 | Megatron PP | vLLM UBatch |
|------|-------------|-------------|
| **切分对象** | Model layers | Batch tokens |
| **目的** | 大模型跨设备 | 通信-计算重叠 |
| **同步方式** | P2P send/recv | CUDA Event + Threading |
| **Bubble** | 有 (warmup/cooldown) | 无 (完全重叠设计) |
| **适用** | 训练 (大模型) | 推理 (DP/EP通信) |
| **代码位置** | `schedules.py` | `gpu_ubatch_wrapper.py` |

---

## 代码位置速查

| 文件 | 行号 | 内容 |
|------|------|------|
| `schedules.py` | L48-100 | `get_forward_backward_func` 入口 |
| `schedules.py` | L2074- | `forward_backward_pipelining_without_interleaving` |
| `schedules.py` | L931- | `forward_backward_pipelining_with_interleaving` |
| `schedules.py` | L2268-2303 | Warmup Forward 循环 |
| `schedules.py` | L2313-2383 | 1F1B Steady State 循环 |
| `schedules.py` | L2385-2409 | Cooldown Backward 循环 |
| `p2p_communication.py` | - | P2PCommunicator 类 |
