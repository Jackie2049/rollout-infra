# 通信与计算重叠

> 分布式训练的核心优化 — 让 GPU 不等网络

## 1. 问题：通信成为瓶颈

```
分布式训练中，GPU 之间需要交换数据:
  DDP: AllReduce 梯度 (每 step 一次)
  TP:  AllReduce / AllGather / ReduceScatter (每层两次)
  PP:  点对点传输激活值

典型通信量:
  TP-4 (7B): ~512 MB / layer × 2 = ~1 GB / layer
  TP-8 (70B): ~2 GB / layer × 2
  DDP AllReduce (7B): ~14 GB / step

如果通信和计算串行执行:
  时间 = 计算时间 + 通信时间
  GPU 利用率 < 50% (通信时 GPU 空闲!)

目标: 通信和计算同时进行
  时间 ≈ max(计算时间, 通信时间)
  GPU 利用率 → 接近 100%
```

## 2. DDP 中的梯度通信重叠

### 2.1 基本原理

```
无重叠:
  Forward → Backward → AllReduce → 更新参数
                       ↑ GPU 空闲

有重叠 (Gradient Bucketing):
  Forward → Backward:
    Layer 32 反向完成 → 立即 AllReduce bucket_1 (异步)
    Layer 31 反向完成 → 立即 AllReduce bucket_2 (异步)
    ...
    Layer 1  反向完成 → AllReduce bucket_N
  → 通信在反向传播中逐步进行

PyTorch DDP 默认开启 bucket overlap!
```

### 2.2 Bucket 策略

```python
# PyTorch DDP 自动 bucket 化梯度
model = DDP(model, device_ids=[0], bucket_cap_mb=25)

# 工作原理:
# 1. 按参数反向传播顺序分配到 bucket (先算的先进 bucket)
# 2. 每个 bucket 满了就异步启动 AllReduce
# 3. 反向传播结束前大部分梯度已同步完成

# 调优:
# bucket_cap_mb=25  → 默认值, 每桶 25MB
# bucket_cap_mb=200 → 更大的桶, 减少 AllReduce 次数, 但等待更久
# bucket_cap_mb=1   → 极小桶, 最快启动通信, 但 AllReduce 开销大

# 最佳实践:
#   小集群 (4-8 GPU): bucket_cap_mb=25 (默认)
#   大集群 (32+ GPU): bucket_cap_mb=100-200
```

## 3. Tensor Parallelism 中的重叠

### 3.1 Megatron-LM 中的通信重叠

```
标准 TP (无重叠):
  每层 Transformer:
    Self-Attention:
      AllReduce (列切分 → 合并)    ← 等待
      计算继续
    MLP:
      AllReduce (列切分 → 合并)    ← 等待
      计算继续
  每层 2 次 AllReduce, 全串行

优化: 与前向传播重叠
  利用当前层的计算来掩盖前一层的通信
  需要 "等待梯度" 的异步化处理

关键配置:
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  → Megatron 推荐设置为 1!
  → 为什么: 强制 CUDA stream 串行调度, 确保通信和计算交错
  → 多个连接会导致通信和计算竞争带宽
```

### 3.2 序列并行 (SP) 的通信

```
Sequence Parallelism (Ring Attention):
  将序列维度切分到不同 GPU
  需要在 attention 计算中传递 Q/K/V

通信模式:
  Ring Communication: Send/Recv 循环
  每步传递一个 chunk

优化:
  重叠: 在计算当前 chunk 时异步传递下一个 chunk
  使用 NCCL 的 P2P 通信 (send/recv)
  torch.distributed.send/recv 在不同 stream 上
```

## 4. Pipeline Parallelism 中的通信重叠

### 4.1 1F1B 调度

```
GPipe (无重叠):
  Forward 所有 stage → Backward 所有 stage
  Bubble 率: (P-1)/P

1F1B (One Forward One Backward):
  Stage 1: F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 ...
  Stage 2:    F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 ...
  Stage 3:       F1 F2 B1 F3 B2 F4 B3 F5 B4 ...
  Stage 4:          F1 B1 F2 B2 F3 B3 F4 B4 ...

  Bubble 率: (P-1)/(P + micro_batches × 2)

Interleaved 1F1B (Megatron):
  每个 GPU 负责多个不连续的 stage chunk
  减少气泡: (P-1)/(P + micro_batches × M)  (M = chunks per GPU)
```

### 4.2 通信与计算重叠机制

```
PP 中的通信: Send/Recv 激活值和梯度

重叠策略:
  1. 异步 P2P 通信:
     torch.distributed.isend() / irecv()
     在不同 CUDA stream 上执行

  2. 双缓冲:
     缓冲区 A: 当前在通信
     缓冲区 B: 当前在计算
     乒乓切换

  Megatron-LM 代码:
    torch.distributed.isend(tensor, dst=next_stage)
    → 返回 Work object, 可 wait()
    → 在等待的同时, 当前 stage 可以开始计算
```

## 5. ZeRO 中的通信重叠

### 5.1 ZeRO-1: 优化器状态分片

```
通信: AllReduce 梯度 (与 DDP 相同)
重叠: 与 DDP 的 bucket 化一样, 天然支持
额外: 优化器状态分片不需要额外通信
```

### 5.2 ZeRO-2: 梯度分片

```
通信: ReduceScatter 替代 AllReduce
  每个 GPU 只接收自己负责的分片
  通信量不变, 但每张卡只存 1/N 的梯度

重叠: ReduceScatter 也支持 bucket 化
  DeepSpeed 默认开启 overlap_comm
```

### 5.3 ZeRO-3: 参数分片

```
通信量大幅增加:
  Forward: AllGather 参数 (每层)
  Backward: AllGather 参数 + ReduceScatter 梯度 (每层)

优化策略:
  1. 预取 (Prefetch):
     当前层在计算时, 预取下一层的参数
     torch.distributed.all_gather() 异步执行

  2. 连续内存:
     contiguous_gradients=True
     将分散的梯度收集到连续内存再通信

  3. 通信重叠:
     overlap_comm=True
     在 backward 计算时异步通信

DeepSpeed 配置:
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "prefetch_bucket_size": 5e7
  }
```

## 6. NCCL 通信优化参数

```
# 调整 NCCL 以实现更好的通信/计算重叠

# 1. 通道数
export NCCL_MIN_NRINGS=4
# 更多通道 → 更好地利用带宽, 但可能增加延迟

# 2. Buffer 大小
export NCCL_BUFFSIZE=4194304  # 4MB
# 影响每次通信的最大数据量

# 3. 算法选择
export NCCL_ALGO=Ring    # Ring (默认), Tree, CollNet
export NCCL_PROTO=Simple # Simple (低延迟), LL (大消息)

# 4. Net (网络)
export NCCL_NET_GDR_LEVEL=5  # GPUDirect RDMA
# 跳过 CPU, GPU 直接通过网卡通信
# 需要 IB/RoCE 网卡支持

# 5. SHM (节点内)
export NCCL_SHM_DISABLE=0  # 启用共享内存 (同节点内)
# 同节点内 GPU 通过 SHM 通信更快
```

## 7. 实践检查清单

```
□ 确认 DDP bucket overlap 已开启 (默认)
□ TP 训练设置 CUDA_DEVICE_MAX_CONNECTIONS=1
□ PP 使用 1F1B 调度 (比 GPipe 更少气泡)
□ ZeRO-3 开启 overlap_comm 和 prefetch
□ NCCL 开启 GPUDirect RDMA (如果硬件支持)
□ 使用 Nsight Systems 分析通信/计算重叠情况
□ 检查 GPU 利用率: nvidia-smi dmon 或 ncu
□ 检查网络带宽利用率: ibstat, ibv_devinfo
```

## 参考

- [Megatron-LM: Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- [DeepSpeed ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Making Megatron-LM communications efficient](https://developer.nvidia.com/blog/making-megatron-lm-communications-efficient/)
