# Megatron-LM 源码阅读笔记

> Megatron-LM 是 NVIDIA 开发的大规模分布式训练框架，是工业标准的大模型训练基础设施。
> 源码版本: main branch (2026-06)

## 目录

1. [架构概览](#架构概览)
2. [Tensor Parallelism (TP)](#tensor-parallelism)
3. [Pipeline Parallelism (PP)](#pipeline-parallelism)
4. [Data Parallelism (DP)](#data-parallelism)
5. [Sequence Parallelism (SP)](#sequence-parallelism)
6. [关键通信原语](#关键通信原语)
7. [性能优化技巧](#性能优化技巧)

---

## 架构概览

```
megatron/core/
├── tensor_parallel/       # 张量并行
│   ├── layers.py          # ColumnParallelLinear, RowParallelLinear
│   ├── mappings.py        # AllReduce, AllGather, ReduceScatter
│   └── utils.py           # split/gather 工具函数
├── pipeline_parallel/     # 流水线并行
│   ├── schedules.py       # 1F1B, GPipe, Interleaved 调度
│   └── p2p_communication.py  # 点对点通信
├── parallel_state.py      # 进程组管理
├── process_groups_config.py  # ProcessGroupCollection
└── optimizer/             # 分布式优化器
```

### 并行维度组合

| 并行类型 | 切分维度 | 通信模式 | 典型规模 |
|---------|---------|---------|---------|
| TP (Tensor) | 权重矩阵 | AllReduce (层内) | 2-8 GPUs |
| PP (Pipeline) | 模型层 | P2P (层间) | 4-16 stages |
| DP (Data) | 数据批次 | AllReduce (梯度) | 任意 |
| SP (Sequence) | 序列长度 | AllGather/ReduceScatter | 配合TP使用 |
| EP (Expert) | MoE专家 | All-to-All | MoE模型 |
| CP (Context) | 长序列 | AllGather/Ring Attention | 超长序列 |

---

## Tensor Parallelism

### 核心思想

将一个大的矩阵乘法 Y = XA 切分到多个 GPU：

```
原始: Y = X @ A        [B,S,H] @ [H,H] = [B,S,H]

TP=2 列切分 (ColumnParallel):
  A = [A₁ | A₂]
  Y₁ = X @ A₁          每个 GPU 算一半输出
  Y = [Y₁ | Y₂]        AllGather 合并

TP=2 行切分 (RowParallel):
  X = [X₁; X₂]          每个 GPU 有部分输入
  Y = X₁@A₁ + X₂@A₂     AllReduce 求和
```

### ColumnParallelLinear (`layers.py`)

```python
# 权重形状: [output_size_per_partition, input_size]
# 每个 TP rank 持有输出维度的一个分片

class ColumnParallelLinear(torch.nn.Module):
    def forward(self, input_):
        # 1. 如果 sequence_parallel: AllGather 输入
        # 2. 本地矩阵乘: output = input @ weight.T + bias
        # 3. 如果 gather_output: AllGather 输出
        # 4. 返回 output (和 bias 如果 skip_bias_add)
```

**前向传播**:
- 输入 X 复制到所有 TP rank (或通过 SP 已经是全局的)
- 每个 rank 计算 Yᵢ = X @ Aᵢ (本地计算，无需通信)
- 如果需要完整输出: AllGather [Y₁, Y₂, ..., Yₚ]

**反向传播**:
- 权重梯度: ∂L/∂Aᵢ = Xᵀ @ ∂L/∂Yᵢ (本地计算)
- 输入梯度: ∂L/∂X = Σ(∂L/∂Yᵢ @ Aᵢ) → AllReduce

### RowParallelLinear (`layers.py`)

```python
# 权重形状: [output_size, input_size_per_partition]
# 每个 TP rank 持有输入维度的一个分片

class RowParallelLinear(torch.nn.Module):
    def forward(self, input_):
        # 1. 如果 !input_is_parallel: Scatter 输入
        # 2. 本地矩阵乘: output = input_local @ weight.T
        # 3. AllReduce 求和 (或 ReduceScatter 如果 SP)
        # 4. 加 bias
```

**前向传播**:
- 输入 X 切分到各 rank: Xᵢ = X[:, start:end]
- 每个 rank 计算 Yᵢ = Xᵢ @ Aᵢ
- AllReduce: Y = ΣYᵢ (通信!)

**反向传播**:
- 权重梯度: ∂L/∂Aᵢ = Xᵢᵀ @ ∂L/∂Y (本地计算)
- 输入梯度: ∂L/∂Xᵢ = ∂L/∂Y @ Aᵢ (本地计算，无需通信)

### MLP 层的 TP 组合

```
            ColumnParallel          RowParallel
MLP:  X → [Linear1ₚ] → GeLU → [Linear2ₚ] → Y
         ↑ 列切分输出            ↑ 行切分输入
         ↑ 无需通信              ↑ AllReduce
```

- Linear1 (ColumnParallel): 切分输出维度，每个 rank 算一半隐藏维度
- GeLU: 本地激活，无通信
- Linear2 (RowParallel): 切分输入维度，AllReduce 合并结果
- **每层只需 1 次 AllReduce** (在 RowParallel 输出处)

### Attention 的 TP

```
Q, K, V = ColumnParallel(X)    # 注意力头切分到各 rank
                                 # 每个 rank 有 H/P 个头
Attention(Qₚ, Kₚ, Vₚ)          # 本地注意力计算
Output = RowParallel(Attn_out)  # AllReduce 合并
```

- 多头注意力天然适合 TP: 每个 TP rank 处理一组注意力头
- QKV 投影用 ColumnParallel, Output 投影用 RowParallel

---

## Pipeline Parallelism

### 核心思想

将模型按层切分到不同 GPU，形成流水线：

```
GPU 0: Layer 0-5     Stage 0
GPU 1: Layer 6-11    Stage 1
GPU 2: Layer 12-17   Stage 2
GPU 3: Layer 18-23   Stage 3
```

### 调度策略

#### 1. GPipe (朴素流水线)

```
Time →
GPU0: F0 F1 F2 F3          B0 B1 B2 B3
GPU1:    F0 F1 F2 F3       B0 B1 B2 B3
GPU2:       F0 F1 F2 F3    B0 B1 B2 B3
GPU3:          F0 F1 F2 F3 B0 B1 B2 B3
              ↑ 气泡 ↑
```

- 先做所有 Forward，再做所有 Backward
- 气泡比例: (P-1)/(P-1+M)，M=microbatch 数
- 内存峰值高: 需要保存所有 microbatch 的激活

#### 2. 1F1B (One Forward One Backward)

```
Time →
GPU0: F0 F1 F2 F3 B0 B1 B2 B3
GPU1:    F0 F1 F2 B0 F3 B1 B2 B3
GPU2:       F0 F1 B0 F2 B1 F3 B2 B3
GPU3:          F0 B0 F1 B1 F2 B2 F3 B3
```

- 交替执行 Forward 和 Backward
- 稳定阶段: 1 个 Forward + 1 个 Backward
- 气泡不变，但内存峰值降低: 只需保存 1 个 microbatch 的激活

#### 3. Interleaved 1F1B (虚拟流水线)

```
将每个 Stage 进一步切分为多个 model chunks:
GPU0: Chunk0(L0-2), Chunk4(L12-14)
GPU1: Chunk1(L3-5), Chunk5(L15-17)
GPU2: Chunk2(L6-8), Chunk6(L18-20)
GPU3: Chunk3(L9-11), Chunk7(L21-23)
```

- 气泡减少到原来的 1/V (V=虚拟 stage 数)
- 需要更多 P2P 通信 (2x)
- 适合高带宽互连 (NVLink)

### P2P 通信 (`p2p_communication.py`, 667行) — 源码级深度

#### P2PCommunicator 核心架构

```python
class P2PCommunicator:
    pp_group: ProcessGroup           # PP进程组
    next_rank: int                   # 下一个stage的global rank
    prev_rank: int                   # 前一个stage的global rank
    virtual_pipeline_model_parallel_size: int  # VP size

    # 核心: _communicate() (line 275-421) — 所有P2P方法的基础
    # 参数: tensor_send_next/tensor_send_prev + recv_prev/recv_next + tensor_shape
```

#### 通信模式

| 模式 | 实现 | 特性 |
|------|------|------|
| `_batched_p2p_ops` | `torch.distributed.batch_isend_irecv()` | 所有send/recv一次batch → config.batch_p2p_comm=True |
| `_p2p_ops` | 逐个`isend`/`irecv` | Even rank:先send→后recv; Odd rank:先recv→后send → 避免deadlock |
| Two-group trick | PP group + WORLD group | PP=2时用两个不同communicator → NCCL真正overlap |

#### 9种P2P方法

| 方法 | Send方向 | Recv方向 | 使用场景 |
|------|----------|----------|----------|
| `recv_forward` | None | prev(fwd) | Warmup forward recv |
| `recv_backward` | None | next(bwd) | Cooldown backward recv |
| `send_forward` | next(fwd) | None | Warmup forward send |
| `send_backward` | prev(bwd) | None | Cooldown backward send |
| `send_forward_recv_backward` | next(fwd) | next(bwd) | 1F1B steady: 发fwd收bwd |
| `send_backward_recv_forward` | prev(bwd) | prev(fwd) | 1F1B steady: 发bwd收fwd |
| `send_forward_recv_forward` | next(fwd) | prev(fwd) | Interleaved warmup |
| `send_backward_recv_backward` | prev(bwd) | next(bwd) | Interleaved cooldown |
| `send_fwd_bwd_recv_fwd_bwd` | both | both | Interleaved steady(无overlap): 一次batched call |

#### Shape Communication (动态序列长度)

```
variable_seq_lengths=True时:
1. 先发送shape: 创建int64 tensor size=3 (编码维度)
2. ring_exchange/batch_isend_irecv 交换shape
3. cuda.synchronize() → 确保shape可用
4. 接收方用学到的shape allocate empty tensor
5. 再发送/接收实际数据

→ 支持不同microbatch有不同序列长度!
```

#### Receive Buffer Allocation

```python
# 接收tensor分配为requires_grad=True → autograd graph跨stage!
recv_buffer = torch.empty(shape, requires_grad=True, dtype=pipeline_dtype, device="cuda")
```

### 1F1B Without Interleaving (schedules.py, line 2092-2462) — 源码级

#### 三阶段详细流程

**Phase 1: Warmup** (num_warmup = PP-1-rank, 最多PP-1)

```
for i in range(num_warmup_microbatches):
    input_tensor = recv_forward()     # 从前stage接收激活
    output_tensor = forward_step()    # 本stage forward计算
    send_forward(output_tensor)       # 发送到后stage
    deallocate_output_tensor()        # 释放output.data→1元素scalar!
    → 只有forward方向P2P, 无backward通信
```

**Phase 2: 1F1B Steady** (num_remaining = M - num_warmup)

```
for i in range(num_microbatches_remaining):
    forward_step(input_tensor)        # 1个forward
    send_forward_recv_backward()      # 同时: 发fwd输出+收bwd梯度 → 通信减半!
    backward_step(pop oldest)         # 1个backward
    send_backward_recv_forward()      # 同时: 发bwd梯度+收fwd输入 → 通信减半!
    → 每个iteration: 1 forward + 1 backward
    → 通信overlap: send+recv同时进行
```

**Phase 3: Cooldown** (drain pipeline, num_warmup步)

```
for i in range(num_warmup_microbatches):
    recv_backward()                   # 从后stage收梯度
    backward_step(pop oldest)         # backward计算
    send_backward()                   # 发梯度到前stage
    → 只有backward方向P2P
```

#### 气泡分析

```
气泡大小 = 2*(PP-1)/M
  warmup: PP-1步forward无backward (stage 0视角)
  cooldown: PP-1步backward无forward (最后stage视角)
  → 总idle = (PP-1)*(fwd_time+bwd_time)
```

### Interleaved 1F1B (schedules.py, line 949-2057) — 源码级

#### Virtual Pipeline Model

```
PP=2, VP=2:
GPU 0: chunk_0(early layers) + chunk_1(mid layers)
GPU 1: chunk_2(later mid) + chunk_3(late layers)

Forward路径: chunk_0(GPU0) → chunk_1(GPU0) → chunk_2(GPU1) → chunk_3(GPU1)
→ P2P只在PP boundary(chunk_1→chunk_2)
```

#### Warmup公式 (更复杂!)

```python
num_warmup = (PP - rank - 1) * 2  # 管线填充×2(每个microbatch穿PP stage两次)
num_warmup += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage  # VP深度buffer
if moe_overlap: num_warmup += 1
```

#### Schedule Table

```
schedule_table[virtual_microbatch_id] → (microbatch_id, model_chunk_id)
Example PP=2, VP=2, M=5:
v_id:  0 1 2 3 4 5 6 7 8 9
m_id:  0 1 2 0 1 2 3 4 3 4
c_id:  0 0 0 1 1 1 0 0 1 1
→ 每组先chunk_0再chunk_1 → 多次穿越PP boundary
```

#### 通信Overlap机制 (Interleaved独有!)

```
steady state with overlap_p2p_comm:
  pp_pre_forward:  wait on posted recv handle (sync recv before compute)
  pp_post_forward: async send_fwd_recv_fwd for next iteration (overlap!)
  pp_pre_backward: wait on posted recv handle for backward
  pp_post_backward: async send_bwd_recv_bwd for next backward

  fwd_recv_buffer: size = microbatch_group_size_per_vp_stage - PP + 1 (leading stage)
  bwd_recv_buffer: same size
  → 通信与计算overlap → NVLink高带宽场景最有效
```

#### 气泡缩减

```
Interleaved气泡 = 2*(PP-1)/(M*VP)
  vs Non-interleaved = 2*(PP-1)/M
  → 气泡减少VP倍!

  约束: microbatch_group_size_per_vp_stage >= PP
  M % microbatch_group_size_per_vp_stage >= PP or == 0
```

### 通信模式对比表

| 阶段 | Non-Interleaved | Interleaved(无overlap) | Interleaved(overlap) |
|------|-----------------|----------------------|---------------------|
| Warmup | recv_fwd+send_fwd | send_fwd_recv_fwd | async send_fwd_recv_fwd+buffers |
| Steady 1F1B | send_fwd_recv_bwd+send_bwd_recv_fwd | send_fwd_bwd_recv_fwd_bwd(batched) | pre/post hooks+async handles |
| Cooldown | recv_bwd+send_bwd | send_bwd_recv_bwd | async send_bwd_recv_bwd+buffers |
| 气泡 | 2*(PP-1)/M | 2*(PP-1)/(M*VP) | 2*(PP-1)/(M*VP) |

### Deallocate Output Tensor (line 157-188)

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    if out is None or not deallocate_pipeline_outputs:
        return
    # 只保留grad_fn → autograd graph跨stage
    # 释放实际data → 省[seq,batch,hidden]大小的内存!
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
    # 断言: out._base is None → 不能是view!
```

### Custom Backward Function (line 190-219)

```python
def custom_backward(output, grad_output):
    # 为什么需要? deallocate后output shape=(1,)但grad_output shape=(seq,batch,hidden)
    # torch.autograd.backward()检查shape匹配 → 会报错!
    # C++ engine (Variable._execution_engine.run_backward) 不检查shape → 直接用grad_fn链
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False, create_graph=False,
        inputs=tuple(), allow_unreachable=True, accumulate_grad=True,
    )
```

### ScheduleNode (utils.py, MoE overlap)

```
ScheduleNode: 每个node有自己的CUDA stream + event
stream_acquire_context: wait event → run on stream → record event → 跨stream依赖
free_input flag: forward后立即释放input storage → input.untyped_storage().resize_(0)
→ combined_1f1b path: MoE overlap调度
```

### Grad Sync管理

```
disable_grad_sync()/enable_grad_sync():
  warmup和1F1B期间 → grad sync disabled → 避免不必要的synchronization
  只在最后iteration启用 → DP AllReduce/ReduceScatter只在训练末尾执行
  grad_sync_func/param_sync_func: 允许overlap grad/param sync与compute
```

### RTX 4090 PP影响

```
RTX 4090 + PP:
  - PCIe inter-GPU: ~12GB/s (vs NVLink 300+GB/s) → P2P通信瓶颈严重
  - PP=2: 气泡=2*(1)/M → M=4时50%气泡 → 加上PCIe延迟更糟
  - Interleaved PP=2,VP=2: 气泡减半但2x P2P → PCIe更慢
  - 结论: RTX 4090不适合PP → 单GPU最优(PP=1)
```

### 关键优化

1. **Deallocate Output**: 发送完激活后立即释放，节省显存
2. **Custom Backward**: 绕过 PyTorch shape checking，节省开销
3. **Shape Communication**: 先传 shape 再传数据，支持动态序列长度

---

## Data Parallelism

### 简单但有效的并行方式

```
GPU 0: 完整模型 + batch_0 → grad_0 ─┐
GPU 1: 完整模型 + batch_1 → grad_1 ─┤ AllReduce → 平均梯度
GPU 2: 完整模型 + batch_2 → grad_2 ─┤
GPU 3: 完整模型 + batch_3 → grad_3 ─┘
```

- 每个 GPU 持有完整模型副本
- 每个 step 后 AllReduce 梯度
- 通信量: 2×model_size (ReduceScatter + AllGather)

### ZeRO 优化

| Stage | 切分内容 | 通信量 | 内存节省 |
|-------|---------|--------|---------|
| ZeRO-1 | 优化器状态 | 不变 | 4x |
| ZeRO-2 | + 梯度 | 不变 | 8x |
| ZeRO-3 | + 模型参数 | 1.5x | N× (N=DP size) |

---

## Sequence Parallelism

### 解决 TP 的 Activation 内存问题

在标准 TP 中，每个 rank 持有完整的 activation 副本 (只有权重被切分)。

SP 将 activation 沿 sequence 维度切分：

```
标准 TP:
  Rank 0: X[full_seq, full_hidden]  ← activation 重复!
  Rank 1: X[full_seq, full_hidden]

SP:
  Rank 0: X[seq/P, full_hidden]     ← activation 切分
  Rank 1: X[seq/P, full_hidden]
```

### SP 的通信模式

```
LayerNorm/Dropout (不需要跨 token):
  → 本地计算，无需通信

ColumnParallelLinear:
  → AllGather(seq_dim) 输入 → 本地计算 → 输出已经是切分的

RowParallelLinear:
  → 输入已经是切分的 → 本地计算 → ReduceScatter(seq_dim) 输出
```

**关键**: AllGather + ReduceScatter = AllReduce
但 Activation 内存节省 P 倍!

### SP 源码级实现 (mappings.py, 617行)

```
核心autograd Functions:

  _GatherFromSequenceParallelRegion:
    Forward: AllGather(seq_dim) → 拼接所有rank的1/P分片
    Backward: ReduceScatter(seq_dim) → 梯度1/P分片回去

  _ReduceScatterToSequenceParallelRegion:
    Forward: ReduceScatter(seq_dim) → 分片结果
    Backward: AllGather(seq_dim) → 拼接梯度恢复完整

  → 对称设计: forward=AG↔backward=RS; forward=RS↔backward=AG
  → autograd自动处理 → 用户只需调用gather/reduce_scatter函数!
```

### ColumnParallelLinear中的SP (layers.py, 1402行)

```
linear_with_frozen_weight (SP模式):

  if sequence_parallel:
    input = gather_from_sequence_parallel_region(input)
    → AllGather: [seq/P, hidden] → [seq, hidden] → 恢复完整序列

  output = LinearWithFrozenWeight.apply(input, weight, bias, ...)
  → 正常计算 → 输出已在TP维度切分

  → backward:
    if sequence_parallel:
      grad_input = reduce_scatter_along_first_dim(grad_output)
      → ReduceScatter: [seq, hidden] → [seq/P, hidden] → 回到SP分片
```

### RowParallelLinear中的SP

```
RowParallelLinear (SP模式):

  forward:
    input已在SP维度切分 [seq/P, hidden/P]
    → 本地计算: matmul → [seq/P, hidden_full] (但还需reduce)
    → ReduceScatter(seq_dim): [seq/P, hidden_full] → [seq/P, hidden/P]
    → 不需要AllGather! → 输入已经是切分维度

  backward:
    → AllGather(seq_dim): 梯度恢复完整序列 → 再计算
```

### SP vs TP 通信量对比

```
TP(无SP):
  ColumnParallel: 输入copy → forward无通信 → backward AllReduce(grad)
  RowParallel: forward AllReduce → backward copy
  → 总通信: 2x AllReduce per layer = 2Ψ

TP+SP:
  ColumnParallel: forward AllGather(Ψ) → backward ReduceScatter(Ψ/N)
  RowParallel: forward ReduceScatter(Ψ) → backward AllGather(Ψ/N)
  → 总通信: AllGather(Ψ) + ReduceScatter(Ψ) = AllReduce(2Ψ) — 等效!
  → 但activation内存省P倍! → 核心优势!

  关键: AllGather + ReduceScatter = AllReduce (数学等价)
  → 通信量相同, 但内存更省 → 这是SP的精髓!
```

### RTX 4090 SP影响

```
RTX 4090 + SP:
  - TP>1无意义(PCIe瓶颈) → SP也不适用(依赖TP)
  - 单GPU: SP=1 → 无通信 → 无分片 → 不节省内存
  - 结论: RTX 4090不适合SP → 单GPU下无TP

  8x RTX 4090 (高校服务器):
  - TP=2: SP省2x activation → 7B可能fit每GPU
  - TP=4: SP省4x → 但PCIe瓶颈严重
  - 建议: TP=2+SP → 最大内存节省+最小PCIe瓶颈

---

## 关键通信原语 (`mappings.py`)

### Autograd Function 包装

每个通信操作都封装为 autograd function，支持反向传播：

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Forward: copy (identity), Backward: AllReduce"""
    @staticmethod
    def forward(ctx, input_):
        return input_
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Forward: AllReduce, Backward: copy"""
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Forward: AllGather(last_dim), Backward: ReduceScatter(last_dim)"""

class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Forward: Scatter(last_dim), Backward: AllGather(last_dim)"""
```

### 异步通信

```python
# 使用 async_op=True 实现通信-计算重叠
handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
# ... 做其他计算 ...
handle.wait()
```

### CUDA_DEVICE_MAX_CONNECTIONS=1

关键环境变量: 限制 GPU 到单个 CUDA 连接，确保通信 kernel 优先调度到 NCCL stream，从而实现通信与计算的 overlap。

---

## 性能优化技巧

### 1. 通信-计算重叠

```
Timeline:
NCCL Stream: [AllGather input_]
Compute:            [matmul forward]  [matmul backward]
NCCL Stream:                          [ReduceScatter grad]
```

### 2. Gradient Accumulation Fusion

```python
# 将梯度计算和累加融合为一个 kernel
fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(...)
# 避免了: compute_grad → add_to_accumulator 两步
```

### 3. Skip Bias Add

```python
# 返回 (output, bias) 而不是 output + bias
# 允许后续操作 (如 GeLU) 与 bias add 融合
return output, bias  # skip_bias_add=True
```

### 4. Memory Buffer Reuse

```python
# 使用全局内存缓冲区，避免动态分配
buffer = get_global_memory_buffer().get(..., dtype)
```

### 5. Pipeline Bubble 优化

- Interleaved 1F1B: 气泡减少 1/V
- V (虚拟 stage 数) 越大越好，但通信量也增加
- 需要 NVLink 等高带宽互连

---

## 与 vLLM 推理框架的对比

| 特性 | Megatron-LM (训练) | vLLM (推理) |
|------|-------------------|-------------|
| TP | Column/RowParallel | TP1D (类似) |
| PP | 1F1B 调度 | PipelinePrefill (新) |
| SP | Activation 切分 | Ring Attention |
| 通信 | AllReduce/AllGather | AllReduce (decode) |
| 内存 | Activation + 优化器 | KV Cache |
| Batch | Micro-batch | Continuous Batching |

---

## 实战练习

### TP 模拟实验 (单 GPU 模拟)

```python
import torch

# 模拟 ColumnParallelLinear + RowParallelLinear
# TP=2, hidden=512

H = 512
TP = 2
x = torch.randn(4, H)  # [batch, hidden]

# Column split: 权重切分输出维度
W1 = torch.randn(H, H // TP)  # rank 0 的权重
W2 = torch.randn(H, H // TP)  # rank 1 的权重
y1 = x @ W1  # [4, 256]
y2 = x @ W2  # [4, 256]
y_col = torch.cat([y1, y2], dim=-1)  # [4, 512] = AllGather

# Row split: 输入和权重都切分
x1, x2 = x.chunk(2, dim=-1)  # 切分输入
A1 = torch.randn(H // TP, H)  # rank 0 的权重
A2 = torch.randn(H // TP, H)  # rank 1 的权重
y_row = x1 @ A1 + x2 @ A2  # AllReduce = sum
```

### 验证

```python
# 完整计算
W_full = torch.cat([W1, W2], dim=1)  # [512, 512]
y_full = x @ W_full

# Column parallel 结果应该与完整计算一致
assert torch.allclose(y_col, y_full)  # ✓
```

---

## Data Parallelism (DP) & 分布式优化器

### Megatron-LM 优化器架构

```
MegatronOptimizer (abstract base)
  ├── MixedPrecisionOptimizer
  │     ├── Float16Optimizer (FP16/BF16)
  │     └── DistributedOptimizer (ZeRO-DP)
  └── TorchFullyShardedDataParallel (FSDP2 集成)
```

### DistributedOptimizer (ZeRO-DP 实现)

```python
# 核心思想: 将优化器状态分片到各 DP rank
# 每个 rank 只存储 1/N 的参数、梯度、优化器状态

class DistributedOptimizer:
    """
    分片策略:
    1. dp_zero_gather_scatter: 朴素 gather/scatter
    2. fully_reshardable: 规范化状态表示
    3. dp_reshardable: 基于桶的分片
    4. fully_sharded_model_space: 按参数分片
    """
```

### 混合精度训练流程

```
1. 前向传播: FP16/BF16 autocast
2. 损失计算: FP32
3. 反向传播: FP16 梯度 → 累加到 FP32 main_grad
4. 梯度同步: AllReduce / ReduceScatter
5. 梯度裁剪: FP32 clip_grad_norm
6. 优化器更新: FP32 参数更新 → 复制回 FP16 模型
```

### 关键内存管理

```python
# _ParamAndGradBuffer: 管理连续参数和梯度缓冲区
# - 将参数分组为桶(buckets)以提高通信效率
# - 64 字节对齐优化 GPU 内存访问
# - 支持 FP8/FP4 量化参数
```

## Gradient Accumulation Fusion

### 自定义 autograd 函数

```python
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    关键优化: 将梯度计算和累加融合为一个 CUDA kernel

    标准:
      grad = input.T @ grad_output  # 计算梯度
      main_grad += grad             # 累加到 main_grad

    融合:
      fused_wgrad_gemm_accum_fp32(input, grad_output, main_grad)
      # 一个 kernel 完成计算+累加, 省去一次内存读写
    """
```

### 梯度延迟 (wgrad_deferral_limit)

```python
# 允许延迟权重梯度计算到多个 microbatch 之后
# 减少大嵌入层的内存带宽使用

wgrad_deferral_limit = N  # 延迟 N 个 microbatch
grad_output_buffer = []   # 缓存梯度输出
embedding_activation_buffer = []  # 缓存输入激活
```

## 与 PyTorch FSDP2 的集成

```python
class TorchFullyShardedDataParallel:
    """
    使用 PyTorch FSDP2 API (需要 PyTorch >= 2.4.0)
    - 支持混合精度和 FP8 张量
    - Just-in-time 参数获取
    - 自定义属性保留 (FP8 transpose cache)
    """
```

---

## MoE Expert Parallel 源码阅读 (2026-06-15)

> 源码: megatron/core/transformer/moe/moe_layer.py (799行)

### MoE Forward 4步Pipeline

```
MoELayer.forward():
    1. route: router(hidden) → (probs, routing_map)
       - TopKRouter: top-k选择 + softmax归一化
       - 支持padding_mask (排除padding token)

    2. dispatch: token_dispatcher.dispatch(hidden, probs)
       - 3种dispatcher策略:
         a) AllGatherTokenDispatcher → AllGather→local compute→ReduceScatter
         b) AlltoAllTokenDispatcher → AlltoAll(v→expert_rank)→compute→AlltoAll(expert_rank→v)
         c) FlexTokenDispatcher → 动态负载均衡

    3. expert_compute: experts(dispatched_input, tokens_per_expert, permuted_probs)
       - 每个rank持有 num_moe_experts/ep_size 个local expert
       - grouped_gemm_backend: auto(FlashInfer)/torch(PyTorch 2.10+)/te(TransformerEngine)/vllm(Triton)

    4. combine: token_dispatcher.combine(output) + shared_expert_output
       - 逆向通信(dispatch→combine)
       - 加上shared_expert_output(如果配置了shared expert)
```

### 3种Token Dispatcher对比

| Dispatcher | 通信模式 | 适用场景 | 内存 |
|------------|----------|----------|------|
| AllGather | AG→local→RS | EP size小, token多 | 高(全聚合) |
| AlltoAll | A2A(v→e)+A2A(e→v) | EP size大, expert多 | 低(只发分配的) |
| Flex | 动态负载均衡 | load imbalance严重 | 中 |

### Latent MoE (DeepSeek-V3风格)

```
hidden_states → fc1_latent_proj → latent_dim → dispatch → expert → combine → fc2_latent_proj → hidden_dim
                    ↓                                                           ↑
                moe_latent_size                                             残差连接
```

- moe_latent_size: hidden_dim→latent_dim投影 → 大幅减少通信量
- fc1/fc2_latent_proj: TELinear (TransformerEngine) → FP8 GEMM
- 与DeepSeek-V3的MLA思路类似: 压缩维度→减少通信→节省内存

### Shared Expert Overlap (推理优化)

```
主stream: route → preprocess → dispatch → expert_compute → combine → postprocess
侧stream(SharedExpertMLP.stream): shared_experts_compute → latent_proj → join+add
```

- shared_expert在独立CUDA stream上与主forward并行
- NVLS(NVLink Shared) dispatcher时自动overlap
- inference_optimized模式: NCCL dispatcher或NVLS dispatcher
- latent+shared overlap: fc1_latent_proj先→shared_expert在侧stream→postprocess join+add

### Delayed Wgrad (反向优化)

```
backward:
    expert_dgrad → record event → ...
    _RegisterDelayedWgradForExperts:
        wgrad_stream.wait_event(dgrad_event)
        expert_wgrad on wgrad_stream (并行!)
        current_stream.wait_event(wgrad_done)
        → post_wgrad_grad_acc_hook (FSDP reduce-scatter)
```

- expert weight梯度在独立stream上计算 → 与其他backward overlap
- 类似DeepSpeed的overlap_grad_reduce → 减少backward耗时

### CUDA Graph Partial Capture

```python
fwd_execution_map = ["route", "expert_compute", "postprocess"]
# 可选择性capture部分forward → 减少CUDA Graph size
# route部分: router + preprocess → 小操作 → 值得capture
# expert_compute: 大GEMM → 可能不值得capture
```

### 与7框架的关系

| 框架 | MoE实现 | EP支持 |
|------|---------|--------|
| Megatron-LM | ✅ 3种dispatcher+latent+shared+overlap | ✅(核心) |
| DeepSpeed | ❌ (无MoE层) | ❌ |
| vLLM | ✅ (MoE serving via expert_parallel) | ✅ |
| verl | ✅ (MoE actor via Megatron strategy) | ✅(via Megatron) |
| rLLM | ✅ (routing_matrices字段) | ✅(via verl) |
| MindIE | ✅ (roadmap) | 🔄 |
| PyTorch | ❌ | ❌ |

---

## 参考资料

- [Megatron-LM Paper (2020)](https://arxiv.org/abs/1909.08053)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- vLLM 源码中的 TP 实现对比
- 本地源码路径: `Megatron-LM/megatron/core/`
