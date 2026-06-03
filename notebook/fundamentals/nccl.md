# NCCL 通信库架构与调优

> NVIDIA Collective Communications Library — 分布式训练的通信基石

## 1. NCCL 概述

### 1.1 定位

NCCL 是 NVIDIA 提供的集合通信库，专门为 GPU 集群优化：

```
应用层:     PyTorch / Megatron-LM / DeepSpeed
            ↓
通信框架:   torch.distributed / MPI
            ↓
通信库:     NCCL (GPU), Gloo (CPU), MPI (通用)  ← 本篇重点
            ↓
网络硬件:   NVLink / InfiniBand / RoCE / TCP
```

### 1.2 支持的操作

| 操作 | 描述 |
|------|------|
| AllReduce | 所有 rank 数据求和，结果广播给所有 rank |
| ReduceScatter | 求和后每个 rank 只保留自己的分片 |
| AllGather | 收集所有 rank 的分片，拼接成完整数据 |
| Broadcast | 一个 rank 的数据广播给所有 rank |
| Reduce | 所有 rank 数据求和，结果发给一个 rank |
| AllToAll | 每个 rank 给每个其他 rank 发送不同数据 |
| Send/Recv | 点对点通信 |

## 2. 通信算法

### 2.1 Ring Algorithm

```
经典 Ring AllReduce:
  数据分为 N 个 chunk，沿环形拓扑流动

  Phase 1: Reduce-Scatter (N-1 步)
    Step 1: rank 0→1, 1→2, ..., N-1→0
    Step 2: rank 0→1, 1→2, ..., N-1→0
    ...

  Phase 2: AllGather (N-1 步)
    Step 1: 已完成的 chunk 在 ring 上传播
    ...

  总通信量: 2 × (N-1)/N × data_size × sizeof(element)
  带宽利用率: (N-1)/N → 接近 1.0
```

**优点**：带宽利用率高，算法简单
**缺点**：延迟 = 2(N-1) × 单步延迟，大规模时延迟增加

### 2.2 Tree Algorithm

```
Double Binary Tree (NCCL 2.4+):
  构建 2 棵二叉树，交替使用

  Tree 1: Reduce from leaves to root, then Broadcast
  Tree 2: 同样操作，但路径不同

  每棵树的深度 = log2(N)
  总延迟 = 2 × log2(N) × 单步延迟
```

**优点**：延迟随 GPU 数量对数增长（vs Ring 的线性）
**缺点**：根节点带宽压力大

### 2.3 CollNet (Collective Offload)

```
利用 InfiniBand 交换机的集合计算能力:
  交换机直接在网络上做 Reduce 操作
  延迟 = 2 × 单步延迟（与 GPU 数量无关！）

需要: InfiniBand SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
适用: 大规模集群 (>64 GPU)
```

### 2.4 算法选择策略

```
NCCL 自动选择算法:
  小数据量 → Tree (低延迟)
  大数据量 → Ring (高带宽利用)
  SHARP 可用 → CollNet (最低延迟)

也可手动指定:
  NCCL_ALGO=Ring   # 强制 Ring
  NCCL_ALGO=Tree   # 强制 Tree
  NCCL_ALGO=CollNet # 强制 CollNet
```

## 3. Channel 架构

### 3.1 Channel 概念

```
NCCL 使用多个并行 channel 提高吞吐:

每个 Channel = 一条独立的通信路径
  - 独立的 CUDA Stream
  - 独立的 proxy thread
  - 独立的网络连接

Channel 数量:
  NVLink: 通常 6-12 个 channel
  PCIe: 通常 4-8 个 channel
  InfiniBand: 通常 4-8 个 channel
```

### 3.2 Channel 调优

```bash
# 设置 channel 数量 (默认自动检测)
NCCL_MIN_NRINGS=2      # 最少 channel 数
NCCL_MAX_NRINGS=8      # 最多 channel 数

# 更多 channel = 更高吞吐，但:
#   - 更多 GPU 显存 (每个 channel 有 buffer)
#   - 更多 CPU 线程
#   - 更多网络连接
```

## 4. 网络拓扑与协议

### 4.1 GPU 间互连

| 互连方式 | 带宽 (双向) | 延迟 | 距离 |
|---------|-----------|------|------|
| NVLink 4.0 | 900 GB/s | ~1 μs | GPU↔GPU (同节点) |
| NVLink 3.0 | 600 GB/s | ~1 μs | GPU↔GPU (同节点) |
| NVSwitch | 900 GB/s | ~2 μs | GPU↔GPU (同节点，全互联) |
| PCIe 5.0 | 128 GB/s | ~5 μs | GPU↔CPU (同节点) |
| InfiniBand NDR | 400 Gb/s | ~1 μs | 节点↔节点 |
| InfiniBand HDR | 200 Gb/s | ~2 μs | 节点↔节点 |
| RoCEv2 | 100-200 Gb/s | ~5 μs | 节点↔节点 |
| TCP/Ethernet | 25-100 Gb/s | ~50 μs | 节点↔节点 |

### 4.2 NCCL 拓扑检测

```bash
# 查看 NCCL 检测到的拓扑
nccl-topo.xml  # 自动生成

# 常见拓扑:
#   DGX A100: 8 GPU, NVLink + NVSwitch
#   HGX H100: 8 GPU, NVLink + NVSwitch + IB
#   自建: GPU 通过 PCIe 连接，节点间 IB 或以太网
```

### 4.3 网络协议选择

```bash
# NCCL 优先级:
#   1. NVLink (节点内)
#   2. InfiniBand (节点间)
#   3. RoCE (节点间)
#   4. TCP Socket (最后手段)

# 强制使用特定网络:
NCCL_P2P_DISABLE=1      # 禁用 NVLink P2P
NCCL_SHM_DISABLE=1      # 禁用共享内存
NCCL_NET_GDR_LEVEL=5    # GPU Direct RDMA 级别
```

## 5. 关键调优参数

### 5.1 性能相关

```bash
# Buffer 大小 (影响通信效率)
NCCL_BUFFSIZE=8388608    # 8MB per channel (默认)
                          # 大 buffer = 大块传输 = 更高效
                          # 但占用更多显存

# 协议选择
NCCL_PROTO=Simple        # 简单协议 (默认)
NCCL_PROTO=LL            # Low-Latency 协议 (小数据量)
NCCL_PROTO=LL128         # Low-Latency 128字节 (最常用)

# 算法选择
NCCL_ALGO=Ring           # 强制 Ring
NCCL_ALGO=Tree           # 强制 Tree

# 融合操作
NCCL_MAX_OPS=2048        # 最大可融合操作数
```

### 5.2 稳定性相关

```bash
# 超时设置
NCCL_COMM_BLOCKING=1     # 阻塞式初始化 (更稳定)
NCCL_MIN_TIMEOUT=1800    # 最小超时 (秒)
NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理

# 避免挂起
NCCL_LAUNCH_MODE=PARALLEL    # 并行启动
NCCL_DEBUG=INFO               # 调试日志
NCCL_DEBUG_SUBSYS=ALL         # 调试所有子系统
```

### 5.3 Megatron-LM 推荐

```bash
# Megatron-LM 的关键 NCCL 设置
CUDA_DEVICE_MAX_CONNECTIONS=1  # 限制 CUDA 连接，确保通信先调度
                                 # 这让 NCCL 通信能与其他操作重叠！

export NCCL_IB_DISABLE=0       # 启用 InfiniBand
export NCCL_IB_GID_INDEX=3     # IB GID index
export NCCL_NET_GDR_LEVEL=5    # GPU Direct RDMA
```

## 6. 常见问题排查

### 6.1 NCCL Timeout

```
症状: NCCL error: timeout
原因:
  1. 网络不通 (firewall, routing)
  2. GPU 挂起 (OOM, CUDA error)
  3. 负载不均衡 (某些 rank 慢)

排查:
  NCCL_DEBUG=INFO
  NCCL_DEBUG_SUBSYS=INIT,COLL
  检查 nccl-topo.xml
```

### 6.2 NCCL Slow

```
症状: 通信速度远低于理论值
原因:
  1. 使用 TCP 而非 IB
  2. NVLink 未启用
  3. Buffer 太小
  4. Channel 数量不足

排查:
  NCCL_DEBUG=INFO | grep "Channel"
  NCCL_DEBUG=INFO | grep "BusId"
  nvidia-smi topo -m  # 查看拓扑
```

### 6.3 NCCL Uninitialized

```
症状: NCCL error: unhandled system error
原因:
  1. 未设置 MASTER_ADDR/MASTER_PORT
  2. 进程数不匹配
  3. NCCL 版本不兼容

排查:
  检查环境变量:
  MASTER_ADDR=...
  MASTER_PORT=...
  WORLD_SIZE=...
  RANK=...
  LOCAL_RANK=...
```

## 7. 性能基准参考

### 7.1 AllReduce 带宽 (A100, NVLink)

```
数据大小    Ring AllReduce    Tree AllReduce
1 KB        ~10 GB/s         ~15 GB/s
10 KB       ~50 GB/s         ~60 GB/s
100 KB      ~150 GB/s        ~130 GB/s
1 MB        ~280 GB/s        ~250 GB/s
10 MB       ~350 GB/s        ~300 GB/s
100 MB      ~370 GB/s        ~320 GB/s

NVLink 理论峰值: 600 GB/s (双向 300 GB/s × 2)
实际利用率: ~60-80% (受协议开销和延迟影响)
```

### 7.2 AllReduce 带宽 (跨节点, HDR IB)

```
数据大小    AllReduce (4 节点, 32 GPU)
1 MB        ~25 GB/s
10 MB       ~45 GB/s
100 MB      ~50 GB/s

HDR IB 理论峰值: 200 Gb/s = 25 GB/s (单向)
实际聚合带宽: ~50 GB/s (多 IB 卡 + Ring)
```

## 8. 学习要点

1. **NCCL 是 GPU 间通信的核心库** — PyTorch 的分布式训练默认使用
2. **Ring 算法适合大数据，Tree 适合小数据** — NCCL 自动选择
3. **Channel 是并行度** — 更多 channel = 更高吞吐
4. **NVLink 是节点内通信的关键** — 比 PCIe 快 5-10x
5. **InfiniBand 是跨节点通信的关键** — 比 TCP 快 10-100x
6. **CUDA_DEVICE_MAX_CONNECTIONS=1** — Megatron 的关键优化，让通信与计算重叠
7. **调试第一步：NCCL_DEBUG=INFO** — 查看拓扑、算法、带宽

## 参考

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Megatron-LM Performance Optimization Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html#performance-optimization)
