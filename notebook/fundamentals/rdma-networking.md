# RDMA 与高性能网络：分布式 AI 训练与推理的网络基石

> Remote Direct Memory Access — 绕过操作系统内核，实现零拷贝、低延迟、高吞吐的网络通信

## 1. RDMA 基础概念

### 1.1 什么是 RDMA？

RDMA (Remote Direct Memory Access) 是一种允许一台机器直接访问另一台机器内存的技术，无需两端操作系统的参与：

```
传统 TCP/IP 网络栈:
  App → Socket Buffer → TCP/IP Stack → Kernel → NIC → ... → NIC → Kernel → TCP/IP → Socket → App
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         CPU 参与，多次数据拷贝，上下文切换                                    CPU 参与，多次数据拷贝

RDMA 网络栈:
  App → 直接读写 RDMA NIC → ... → RDMA NIC → 直接写入远端内存
         ^^^^^^^^^^^^^^^^^^^^                  ^^^^^^^^^^^^^^^^^^^^
         绕过 OS 内核，零拷贝                    绕过 OS 内核，零拷贝
```

### 1.2 RDMA 的核心优势

| 特性 | TCP/IP | RDMA |
|------|--------|------|
| 数据拷贝 | 3-4 次 (App↔Socket↔Kernel↔NIC) | 0 次 (App↔NIC 直通) |
| CPU 参与 | 高 (协议栈处理、中断、拷贝) | 极低 (只发起传输请求) |
| 上下文切换 | 是 (用户态/内核态切换) | 否 (完全在用户态) |
| 延迟 | ~50-100 μs | ~1-5 μs |
| 带宽开销 | TCP/IP 头 + 拷贝损失 | 接近线速 (>95%) |
| 消息速率 | ~1M msg/s | ~100M+ msg/s |

### 1.3 RDMA 工作原理

```
1. 应用注册内存区域 (Memory Registration)
   → 告诉 NIC: 这块物理内存可以被直接访问
   → OS 将内存页锁定 (pin)，防止被换出到磁盘

2. 应用创建 Queue Pair (QP)
   → Send Queue: 存放待发送的工作请求 (WR)
   → Receive Queue: 存放接收 buffer 的描述符

3. 应用提交工作请求 (Work Request)
   → 把 WR 放到 Send Queue (称为 "post send")
   → 硬件自动处理，不需要 CPU 参与

4. NIC 直接 DMA 读写内存
   → 本地 NIC DMA 读取用户态 buffer
   → 通过网络发送到远端 NIC
   → 远端 NIC DMA 写入远端用户态 buffer

5. 完成通知 (Completion Queue)
   → NIC 完成后产生 Completion Queue Entry (CQE)
   → 应用轮询 (poll) CQ 获取完成状态
```

---

## 2. RDMA 三大协议对比：InfiniBand vs RoCE vs iWARP

### 2.1 InfiniBand

```
特点:
  - 专用网络架构（非 Ethernet）
  - 从物理层到传输层全部重新设计
  - 自带可靠传输、流控、QoS
  - 需要 InfiniBand 交换机和 HCA (Host Channel Adapter)

优势:
  - 延迟最低 (~0.5-1 μs)
  - 带宽最高 (NDR 400 Gb/s, XDR 800 Gb/s)
  - 原生支持 SHARP (交换机内 Reduce)
  - 无损传输 (Credit-based 流控)
  - 自适应路由 (Adaptive Routing)

劣势:
  - 成本高 (专用交换机、线缆)
  - 不兼容现有 Ethernet 基础设施
  - 供应商锁定风险 (Mellanox/NVIDIA 主导)
```

### 2.2 RoCE (RDMA over Converged Ethernet)

```
特点:
  - 在标准 Ethernet 上运行 RDMA
  - RoCEv1: 只在同一个二层网络 (Layer 2) 工作
  - RoCEv2: 封装在 UDP/IP 中，可路由 (Layer 3) ← 主流

优势:
  - 使用标准 Ethernet 交换机，成本更低
  - 兼容现有网络基础设施
  - RoCEv2 支持路由，部署灵活

劣势:
  - 需要额外配置实现无损:
    - PFC (Priority Flow Control): 基于 802.1p 优先级的流量控制
    - ECN (Explicit Congestion Notification): 拥塞信号
    - DCQCN (Congestion Control): RoCE 拥塞控制协议
  - 延迟略高于 InfiniBand (~2-5 μs vs ~0.5-1 μs)
  - 调优更复杂 (PFC 优先级映射、ECN 阈值等)
```

### 2.3 iWARP (Internet Wide Area RDMA Protocol)

```
特点:
  - RDMA over TCP/IP
  - 使用标准 TCP 握手和拥塞控制

优势:
  - 完全兼容现有 IP 网络
  - 可穿越路由器和广域网
  - 不需要特殊交换机支持

劣势:
  - 性能最低 (TCP 开销)
  - 延迟最高 (~5-10 μs)
  - 社区支持最少
  - AI 训练场景几乎不使用
```

### 2.4 协议选择决策

```
适用场景:

InfiniBand:
  ✅ 大规模 GPU 集群 (>64 GPU)
  ✅ 对延迟极度敏感 (训练大模型)
  ✅ 预算充足 (DGX/HGX 系统)
  → NVIDIA DGX/H100/H200 标配

RoCEv2:
  ✅ 中等规模集群
  ✅ 已有 Ethernet 基础设施
  ✅ 想要 RDMA 性能但预算有限
  → 部分云厂商 (Azure HBv3, GPU 云)

iWARP:
  ❌ AI 训练场景几乎不使用
  → 主要用于存储 (iSER, SMB Direct)
```

### 2.5 版本与带宽演进

| 世代 | InfiniBand | 带宽 (单向) | 年份 |
|------|-----------|------------|------|
| SDR | 1X / 4X / 12X | 2.5 / 10 / 30 Gb/s | 2002 |
| DDR | 1X / 4X / 12X | 5 / 20 / 60 Gb/s | 2005 |
| QDR | 1X / 4X / 12X | 10 / 40 / 120 Gb/s | 2008 |
| FDR | 1X / 4X / 12X | 14.06 / 56.25 / 168.75 Gb/s | 2011 |
| EDR | 1X / 4X | 25 / 100 Gb/s | 2014 |
| HDR | 1X / 2X / 4X | 50 / 100 / 200 Gb/s | 2018 |
| NDR | 1X / 4X / 8X | 100 / 400 / 800 Gb/s | 2023 |
| XDR | - | 800+ Gb/s | 2025+ |

---

## 3. RDMA 核心概念详解

### 3.1 Queue Pair (QP)

```
每个 RDMA 通信端点需要一个 Queue Pair:

  ┌─────────────────────────────────────────────┐
  │              Queue Pair (QP)                 │
  │  ┌─────────────────┐ ┌──────────────────┐   │
  │  │   Send Queue     │ │  Receive Queue   │   │
  │  │   (SQ)           │ │  (RQ)            │   │
  │  │                   │ │                   │   │
  │  │  WR → [操作描述]  │ │  WR → [buffer]   │   │
  │  │  WR → [操作描述]  │ │  WR → [buffer]   │   │
  │  │  WR → [操作描述]  │ │  WR → [buffer]   │   │
  │  └─────────────────┘ └──────────────────┘   │
  └─────────────────────────────────────────────┘

QP 类型 (Transport Service):
  - RC (Reliable Connected):   一对一可靠连接，最常用
  - UC (Unreliable Connected): 一对一不可靠
  - UD (Unreliable Datagram):  一对多不可靠
  - DCT (Dynamic Connected Transport): 按需建立连接，节省资源
```

**NCCL 中的 QP 使用**:
- NCCL 为每对通信 rank 建立 RC QP
- 支持 `NCCL_IB_QPS_PER_CONNECTION` (1-128) 配置每连接多个 QP
- 多 QP 提供路由熵 (entropy)，提高多路径利用率

### 3.2 Completion Queue (CQ)

```
  ┌─────────────────────────┐
  │    Completion Queue (CQ) │
  │                          │
  │  CQE: [状态, 字节数, ...] │  ← NIC 完成操作后写入
  │  CQE: [状态, 字节数, ...] │
  │  CQE: [状态, 字节数, ...] │
  │                          │
  └─────────────────────────┘
         ↑ poll (轮询)

应用通过 ibv_poll_cq() 检查完成状态:
  - 同步: 轮询直到完成 (最低延迟)
  - 异步: 使用完成通道 (Completion Channel) + 中断
  - NCCL 使用轮询模式，获得最低延迟
```

### 3.3 Memory Registration (MR)

```
RDMA 要求所有参与通信的内存必须注册:

1. 注册过程:
   ibv_reg_mr(pd, buffer, length, access_flags)
     → OS 锁定 (pin) 物理内存页
     → NIC 获得虚拟地址→物理地址映射
     → 返回 lkey (local) 和 rkey (remote)

2. 访问权限:
   IBV_ACCESS_LOCAL_WRITE   - 本地写
   IBV_ACCESS_REMOTE_WRITE  - 远端写 (RDMA Write)
   IBV_ACCESS_REMOTE_READ   - 远端读 (RDMA Read)
   IBV_ACCESS_REMOTE_ATOMIC - 远端原子操作

3. 为什么需要注册?
   - RDMA 绕过 OS，NIC 直接 DMA 访问内存
   - 如果内存页被 swap 到磁盘，NIC 无法访问
   - 注册 = 告诉 OS "这块内存不能换出" (mlock)
   - 同时建立 NIC 的 IOMMU/DMA 映射

4. 重要限制:
   - 注册大量内存有开销 (页表遍历)
   - 系统需配置 memlock unlimited:
     /etc/security/limits.conf:
       * soft memlock unlimited
       * hard memlock unlimited
```

### 3.4 RDMA 操作类型

```
1. RDMA Write (单向，最常用):
   发送方指定远端地址 + rkey，直接写入远端内存
   远端 CPU 完全不参与

   本地 App → [addr, rkey, data] → NIC → 网络 → NIC → 写入远端内存

2. RDMA Read (单向):
   发送方指定远端地址 + rkey，直接读取远端内存
   远端 CPU 完全不参与

   本地 App → [addr, rkey] → NIC → 网络 → NIC → 从远端内存读

3. Send/Recv (双向):
   发送方 Send，接收方必须预先 Post Receive Buffer
   接收方 CPU 需要预先准备 buffer (但收数据时 CPU 不参与)

4. Atomic (CAS / FetchAdd):
   远端原子操作，用于同步原语
```

### 3.5 Verb API 层次

```
RDMA 编程接口 (Verbs):

  libibverbs (用户态):
    ibv_open_device()      - 打开 RDMA 设备
    ibv_alloc_pd()         - 分配 Protection Domain
    ibv_reg_mr()           - 注册内存区域
    ibv_create_cq()        - 创建完成队列
    ibv_create_qp()        - 创建队列对
    ibv_post_send()        - 提交发送请求
    ibv_post_recv()        - 提交接收请求
    ibv_poll_cq()          - 轮询完成队列
    ibv_modify_qp()        - 修改 QP 状态 (RTS 等)

  librdmacm (连接管理):
    rdma_create_id()       - 创建 RDMA 标识符
    rdma_bind_addr()       - 绑定地址
    rdma_listen()          - 监听连接
    rdma_connect()         - 发起连接
    rdma_accept()          - 接受连接

  rdma-core 项目 (GitHub): 开源用户态 RDMA 库
```

---

## 4. NCCL 与 RDMA

### 4.1 NCCL 如何使用 RDMA

```
NCCL 通信路径 (跨节点):

  GPU 0 (本地)                              GPU 0 (远端)
  ┌──────┐                                  ┌──────┐
  │CUDA  │                                  │CUDA  │
  │Buffer│                                  │Buffer│
  └──┬───┘                                  └──▲───┘
     │ GPU Direct RDMA / DMA-BUF               │ GPU Direct RDMA
     ▼                                         │
  ┌──────┐    QP (RC)     ┌──────┐    QP       │
  │ IB   │───────────────→│ IB   │─────────────┘
  │ NIC  │   InfiniBand   │ NIC  │
  └──────┘   或 RoCEv2    └──────┘

通信流程:
  1. NCCL 初始化时:
     - 检测 RDMA 设备 (ibv_get_device_list)
     - 为每对 rank 创建 QP
     - 注册 CUDA 内存 (或使用 GPU Direct)

  2. 集合通信时:
     - GPU kernel 通过 NCCL Device API 发起通信
     - NCCL proxy thread 管理 RDMA 操作
     - 通过 ibv_post_send() 提交 RDMA Write/Read
     - 轮询 CQ 确认完成

  3. 多 Channel 并行:
     - NCCL 创建多个独立的通信 channel
     - 每个 channel 有自己的 QP 和 proxy thread
     - 充分利用网络带宽
```

### 4.2 NCCL RDMA 关键环境变量

```bash
# === InfiniBand/RoCE 基础设置 ===

# 禁用/启用 IB (默认启用)
NCCL_IB_DISABLE=0

# 选择 IB 接口 (冒号后数字 = 端口号)
NCCL_IB_HCA=mlx5_0:1,mlx5_1:1

# IB 超时 (默认 20, 约 4.3 秒)
# 值 = 4.096 μs × 2^timeout
# 范围 0-31
NCCL_IB_TIMEOUT=20

# IB 重试次数 (默认 7)
# 总超时 ≈ NCCL_IB_TIMEOUT × NCCL_IB_RETRY_CNT
NCCL_IB_RETRY_CNT=7

# === RoCE 专用 ===

# GID Index (RoCE 需要选择正确的 GID)
# 默认 -1 = 自动选择 (NCCL 2.21+)
NCCL_IB_GID_INDEX=-1

# Traffic Class (QoS 标记)
NCCL_IB_TC=0

# 自适应路由 (IB 默认启用, RoCE 默认禁用)
NCCL_IB_ADAPTIVE_ROUTING=1

# === GPU Direct RDMA ===

# GPU Direct RDMA 级别
# LOC=1: 同 PCI 设备 (GPU↔NIC 同卡)
# PIX=2: 同 PCI 交换机
# PXB=3: 同 PCI 桥下不同交换机
# PHB=4: 同 NUMA 节点
# SYS=5: 跨 NUMA 节点
NCCL_NET_GDR_LEVEL=5

# GPU Direct RDMA 读操作 (默认 1, NVLink 平台自动启用)
NCCL_NET_GDR_READ=1

# DMA-BUF (新型 GPU Direct 机制, Linux 内核 6.x+)
NCCL_DMABUF_ENABLE=1

# === Path Crossing NVLink (PXN) ===

# 利用 NVLink 访问非本地 NIC
# 例: GPU 0 通过 NVLink 借用 GPU 3 的 IB NIC
# 默认启用
NCCL_PXN_DISABLE=0

# === NVLink SHARP ===

# NVLink 上的网络内 Reduce
# 0=禁用, 1=启用, 2=自动 (默认)
NCCL_NVLS_ENABLE=2

# === 多 QP 优化 ===

# 每个连接的 QP 数量 (1-128)
# 多 QP 利用交换机多路径
NCCL_IB_QPS_PER_CONNECTION=4

# 合并双端口 IB NIC (默认启用)
NCCL_IB_MERGE_NICS=1

# 跨 NIC 策略 (默认 2 = 优先同 NIC, 允许跨 NIC)
NCCL_CROSS_NIC=2

# === Multi-Node NVLink ===

# MNNVL 支持 (NVLink 跨节点扩展)
NCCL_MNNVL_ENABLE=1
```

### 4.3 NCCL 通信算法与 RDMA

```
NCCL 在 RDMA 网络上的算法选择:

  小数据 (<128 KB):
    Tree (LL/LL128 协议)
    → 延迟 = 2 × log2(N) × 单跳延迟
    → RDMA Write 小包，延迟关键

  大数据 (>128 KB):
    Ring (Simple 协议)
    → 带宽利用率 = (N-1)/N → 接近 100%
    → RDMA Write 大块，带宽关键

  SHARP 可用:
    CollnetChain / CollnetDirect
    → 交换机做 Reduce，延迟与 GPU 数无关
    → 需要 InfiniBand SHARP 交换机

  NVLink SHARP:
    NVLS / NVLSTree
    → NVLink Switch 做网络内 Reduce
    → 需要 NVSwitch with SHARP engine

  H100+ 新算法:
    PAT (Parallel AllReduce Tree)
    → 多棵树并行，提高 NVLink 带宽利用率
```

### 4.4 NCCL 通信协议

```
NCCL 内部协议 (影响 RDMA 传输方式):

  Simple:
    - 直接传输数据
    - 适合大数据量
    - 带宽效率高

  LL (Low Latency):
    - 数据 + 校验和一起发送
    - 接收方无需等待完整数据即可校验
    - 适合小数据量

  LL128:
    - 128 字节对齐的 Low Latency
    - 兼顾延迟和带宽
    - NCCL 推荐的默认协议
```

---

## 5. GPUDirect RDMA

### 5.1 原理

```
传统方式 (无 GPUDirect):
  GPU Memory → CPU Memory (DMA) → NIC (DMA) → 网络 → NIC → CPU Memory → GPU Memory
               ^^^^^^^^^^^^^^^^^^^^                                    ^^^^^^^^^^^^^^^^^^^^
               PCIe 传输 1: GPU→CPU                                    PCIe 传输 3: CPU→GPU
                                  CPU→NIC (DMA)             NIC→CPU (DMA)

  总共: 4 次 DMA + 1 次网络传输

GPUDirect RDMA:
  GPU Memory → NIC (直接 DMA) → 网络 → NIC → GPU Memory (直接 DMA)

  总共: 0 次 CPU 内存拷贝，NIC 直接 DMA GPU 内存
```

### 5.2 GPUDirect 技术 family

```
┌──────────────────────────────────────────────────────┐
│                   GPUDirect 家族                      │
├──────────────────────────────────────────────────────┤
│                                                       │
│  GPUDirect P2P (Peer-to-Peer)                        │
│    GPU ↔ GPU 直接通信 (同一节点, 通过 NVLink/PCIe)    │
│    cudaDeviceEnablePeerAccess()                       │
│                                                       │
│  GPUDirect RDMA                                       │
│    NIC 直接 DMA GPU 内存 (跨节点)                     │
│    需要: nvidia-peermem 内核模块                       │
│    或者: DMA-BUF (Linux 6.x+)                         │
│                                                       │
│  GPUDirect NVLink                                     │
│    NVLink 连接 GPU 和 NIC (部分架构)                  │
│    例: BlueField DPU + NVLink                         │
│                                                       │
│  GPUDirect Storage                                    │
│    存储设备直接 DMA GPU 内存                           │
│    GDS / cuFile API                                   │
│    避免存储读取经过 CPU                                │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 5.3 nvidia-peermem 模块

```bash
# GPUDirect RDMA 需要内核模块让 NIC 访问 GPU 内存

# 检查是否加载:
lsmod | grep nvidia_peermem

# 手动加载:
sudo modprobe nvidia-peermem

# 持久化 (写入 /etc/modules-load.d/):
echo "nvidia-peermem" | sudo tee /etc/modules-load.d/peermem.conf

# 工作原理:
#   1. 注册为 IB peer memory client
#   2. 当 ibv_reg_mr() 请求 GPU 内存时
#   3. peermem 获取 GPU 页面的 DMA 地址
#   4. NIC 可以直接 DMA 这些 GPU 页面
```

### 5.4 DMA-BUF (新型替代方案)

```
Linux 内核 6.x+ 的 DMA-BUF 机制:
  - 开源 NVIDIA 驱动 (open GPU kernel modules) 原生支持
  - 不需要 nvidia-peermem 模块
  - NCCL 2.20+ 自动检测和使用
  - NCCL_DMABUF_ENABLE=1 (默认启用)
  - 更安全: 内核管理 DMA 映射，避免驱动冲突
```

### 5.5 ACS (Access Control Services) 问题

```bash
# ACS 可能阻止 GPUDirect RDMA (IOMMU DMA 重映射)
# 检查 ACS 状态:
sudo lspci -vvv | grep ACSCtl

# 如果看到 ACS Ctrl: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+
# 需要禁用 ACS (BIOS 或 setpci):

# 查找 NIC 的 PCI 地址:
lspci | grep -i mellanox
# 例: 03:00.0

# 禁用 ACS:
sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000

# 或者通过 GRUB 参数:
# 在 /etc/default/grub 添加:
GRUB_CMDLINE_LINUX="... pci=assign-busses,realloc pcie_acs_override=downstream,multifunction"
```

---

## 6. NVLink vs InfiniBand 对比

### 6.1 带宽与延迟对比

```
┌───────────────────────────────────────────────────────────────┐
│                    互连技术性能对比                             │
├──────────────────┬──────────────┬──────────┬──────────────────┤
│ 技术             │ 双向带宽      │ 延迟     │ 适用范围         │
├──────────────────┼──────────────┼──────────┼──────────────────┤
│ NVLink 6 (Rubin) │ 3,600 GB/s   │ ~0.5 μs  │ GPU↔GPU (同节点) │
│ NVLink 5 (B200)  │ 1,800 GB/s   │ ~0.5 μs  │ GPU↔GPU (同节点) │
│ NVLink 4 (H100)  │   900 GB/s   │ ~1 μs    │ GPU↔GPU (同节点) │
│ NVLink 3 (A100)  │   600 GB/s   │ ~1 μs    │ GPU↔GPU (同节点) │
│ PCIe 5.0 x16     │   128 GB/s   │ ~5 μs    │ GPU↔CPU (同节点) │
│ PCIe 4.0 x16     │    64 GB/s   │ ~8 μs    │ GPU↔CPU (同节点) │
│ IB XDR (800G)    │    100 GB/s  │ ~0.5 μs  │ 节点↔节点        │
│ IB NDR (400G)    │     50 GB/s  │ ~1 μs    │ 节点↔节点        │
│ IB HDR (200G)    │     25 GB/s  │ ~2 μs    │ 节点↔节点        │
│ RoCEv2 (200G)    │     25 GB/s  │ ~3-5 μs  │ 节点↔节点        │
│ Ethernet (100G)  │     12.5 GB/s│ ~50 μs   │ 节点↔节点        │
└──────────────────┴──────────────┴──────────┴──────────────────┘

关键对比:
  NVLink 4.0 vs IB NDR:  带宽 18x (900 vs 50 GB/s)
  NVLink 4.0 vs PCIe 5:  带宽 7x  (900 vs 128 GB/s)
  IB NDR vs Ethernet:     延迟 50x (1 μs vs 50 μs)
```

### 6.2 NVLink 详细规格

```
NVLink 世代:

  NVLink 1.0 (P100, 2016):   160 GB/s, 4 links
  NVLink 2.0 (V100, 2017):   300 GB/s, 6 links
  NVLink 3.0 (A100, 2020):   600 GB/s, 12 links
  NVLink 4.0 (H100, 2022):   900 GB/s, 18 links
  NVLink 5.0 (B200, 2024): 1,800 GB/s, 36 links (2x 改进)
  NVLink 6.0 (Rubin, 2026): 3,600 GB/s, 72 links (2x 改进)

  NVLink 6 对比:
    - vs NVLink 5: 2x 带宽
    - vs PCIe Gen6: 14x 带宽
    - vs NVLink 1: 22.5x 带宽

  NVLink Switch:
    - 使同一机架内所有 GPU 全互联
    - Vera Rubin NVL72: 72 GPU 全互联
    - 总带宽: 260 TB/s
    - 算力: 3.6 exaFLOPS
    - 内置 SHARP 引擎 (网络内 Reduce)
    - NVLink 6 Switch 新增: 控制面容错、部分机架部署、热插拔
```

### 6.3 NVLink vs InfiniBand 使用场景

```
节点内通信 (同服务器):
  ✅ NVLink (首选)
    → 8 GPU 全互联 (通过 NVSwitch)
    → 带宽远超任何网络技术
    → 用于 TP (Tensor Parallel)、EP (Expert Parallel)

节点间通信 (跨服务器):
  ✅ InfiniBand (首选)
    → 大规模训练标配
    → NVLink 无法跨节点 (除 MNNVL)
    → 用于 PP (Pipeline Parallel)、DP (Data Parallel)

特殊场景:
  MNNVL (Multi-Node NVLink):
    → NVLink 延伸到节点间
    → 需要特殊硬件 (NVLink Cabinet)
    → NCCL_MNNVL_ENABLE=1

  PXN (Path Crossing NVLink):
    → GPU 通过 NVLink 借用其他 GPU 的 IB NIC
    → 优化 NUMA 感知的网络访问
    → 默认启用
```

---

## 7. 性能数据参考

### 7.1 延迟数据

```
端到端延迟 (应用层测量的 round-trip):

  NVLink (同节点):        ~1-2 μs
  InfiniBand NDR:         ~1-2 μs
  InfiniBand HDR:         ~2-3 μs
  RoCEv2:                 ~3-6 μs
  TCP/IP (同机房):        ~50-100 μs

RDMA 操作延迟 (硬件级):

  RDMA Write:             ~0.5-1 μs (取决于网络)
  RDMA Read:              ~1-2 μs
  Send/Recv:              ~1-2 μs
  Atomic (CAS):           ~1-3 μs
```

### 7.2 带宽数据

```
NCCL AllReduce 有效带宽:

  8×A100 (NVLink, 单节点):
    1 MB:    ~280 GB/s
    10 MB:   ~350 GB/s
    100 MB:  ~370 GB/s
    理论峰值: 600 GB/s (NVLink 3.0)
    利用率:   ~60-80%

  32×A100 (HDR IB, 4 节点):
    1 MB:    ~25 GB/s
    10 MB:   ~45 GB/s
    100 MB:  ~50 GB/s
    理论峰值: 200 Gb/s = 25 GB/s per NIC
    聚合:     ~50 GB/s (多 NIC + Ring)

  64×H100 (NDR IB, 8 节点):
    1 MB:    ~60 GB/s
    10 MB:   ~120 GB/s
    100 MB:  ~160 GB/s
    理论峰值: 400 Gb/s = 50 GB/s per NIC

  ib_write_bw 基准 (原始 RDMA 带宽):
    HDR IB:  ~23 GB/s (接近 25 GB/s 理论)
    NDR IB:  ~48 GB/s (接近 50 GB/s 理论)
    RoCEv2:  ~23 GB/s (200G RoCE)
```

### 7.3 训练场景通信占比

```
典型 LLM 训练 (TP=8, PP=1, DP=N):

  TP 通信 (NVLink):
    AllReduce per layer:  ~128 MB (hidden_size=8192, seq_len=4096, BF16)
    延迟: ~1 μs (NVLink)
    占训练时间: ~5-10%

  DP 通信 (IB/RoCE):
    AllReduce 梯度: ~模型参数量 × 2 bytes (BF16)
    7B 模型: ~14 GB per step
    70B 模型: ~140 GB per step
    带宽需求: 14 GB / 0.5s = 28 GB/s (7B, 目标 step time ~0.5s)
    占训练时间: ~10-25%

  通信瓶颈分析:
    NVLink 内: 通信 < 10% 训练时间 ✅
    跨节点 IB: 通信 10-25% ⚠️ → 需要与计算重叠
    以太网:    通信 > 50% ❌ → 严重瓶颈
```

---

## 8. 实战调优指南

### 8.1 InfiniBand 基础检查

```bash
# 1. 检查 IB 设备
ibv_devinfo
# 确认: 状态 ACTIVE, 速率正确 (HDR: 200 Gb/s, NDR: 400 Gb/s)

# 2. 检查 IB 端口状态
ibstat
# Port state 应为 Active

# 3. 测试节点间连通性
# Server:
ib_write_bw -d mlx5_0
# Client:
ib_write_bw -d mlx5_0 <server_ip>

# 4. 测试延迟
ib_write_lat -d mlx5_0 <server_ip>

# 5. 列出 IB 接口
ibv_devinfo -l
# 例: mlx5_0, mlx5_1 (双端口 NIC)
```

### 8.2 NCCL 配置最佳实践

```bash
# === 大规模训练 (64+ GPU, InfiniBand) ===
export NCCL_IB_DISABLE=0              # 启用 IB
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1 # 使用双端口
export NCCL_IB_GID_INDEX=-1           # 自动选择 GID (NCCL 2.21+)
export NCCL_NET_GDR_LEVEL=5           # GPU Direct RDMA
export NCCL_NET_GDR_READ=1            # GPU Direct 读
export NCCL_IB_QPS_PER_CONNECTION=4   # 多 QP 提高利用率
export NCCL_IB_MERGE_NICS=1           # 合并双端口 NIC
export NCCL_PXN_DISABLE=0             # 启用 PXN (NVLink 穿越)
export NCCL_IB_TC=106                 # Traffic Class (RoCE QoS)
export NCCL_SOCKET_IFNAME=eth0        # 辅助通信接口
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 通信/计算重叠

# === 中等规模 (RoCEv2) ===
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3            # RoCE GID (需确认)
export NCCL_IB_TC=106                 # RoCE DSCP 标记
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_ADAPTIVE_ROUTING=0     # RoCE 通常禁用自适应路由

# === 调试配置 ===
export NCCL_DEBUG=INFO                # 详细日志
export NCCL_DEBUG_SUBSYS=ALL          # 所有子系统
export NCCL_DEBUG_FILE=/tmp/nccl.log  # 日志文件
```

### 8.3 RoCEv2 无损网络配置

```bash
# RoCE 需要无损网络，必须配置以下三项:

# 1. PFC (Priority Flow Control)
# 在交换机上:
#   - 启用 PFC on priority 3 (或 106 >> 2)
#   - 对应 NCCL_IB_TC 的优先级

# 2. ECN (Explicit Congestion Notification)
# 在交换机上:
#   - 启用 ECN 标记
#   - 阈值: min ~150 KB, max ~3 MB, probability 100%

# 3. DCQCN (Data Center QCN)
# 在 NIC 上:
sysctl -w net.ipv4.tcp_ecn=1

# 验证 RoCE 配置:
ibv_devinfo -d mlx5_0 -v
# 检查 GID 表:
show_gids
```

### 8.4 Docker 环境配置

```bash
# Docker 运行分布式训练需要:

docker run --gpus all \
  --net=host \
  --shm-size=1g \           # 共享内存 (NCCL 进程内通信)
  --ulimit memlock=-1 \     # 无限内存锁定 (RDMA 注册)
  --device=/dev/infiniband \ # IB 设备访问
  --cap-add=IPC_LOCK \      # 内存锁定权限
  -v /tmp/nccl.log:/tmp/nccl.log \
  your_image

# 或者使用 NVIDIA Container Runtime:
docker run --runtime=nvidia ...
# NVIDIA Container Toolkit 自动挂载 IB 设备和库
```

### 8.5 常见问题排查

```
问题 1: NCCL 使用 TCP 而非 IB
  症状: NCCL_DEBUG 日志显示 "NET/Socket"
  排查:
    - ibv_devinfo 确认 IB 设备可用
    - 检查 libibverbs 是否安装
    - 检查 NCCL_IB_DISABLE 是否被设为 1
    - 检查 nvidia-peermem 是否加载

问题 2: GPUDirect RDMA 未启用
  症状: 通信带宽远低于理论值
  排查:
    - lsmod | grep nvidia_peermem
    - NCCL_DEBUG=INFO | grep "GPU Direct"
    - 检查 ACS: sudo lspci -vvv | grep ACSCtl
    - 检查 NUMA 感知: GPU 和 NIC 应在同一 NUMA 节点

问题 3: RoCE GID 错误
  症状: "ibv_modify_qp failed with error Invalid argument"
  排查:
    - NCCL 2.21+ 自动选择 (NCCL_IB_GID_INDEX=-1)
    - 旧版本手动设置: show_gids 查看 RoCE v2 对应的 GID index
    - 通常: 纯 IB=0, RoCEv1=1, RoCEv2=2 或 3

问题 4: memlock 不足
  症状: ibv_reg_mr 失败, "Cannot allocate memory"
  解决:
    - /etc/security/limits.conf:
      * soft memlock unlimited
      * hard memlock unlimited
    - 重新登录 (或 ulimit -l unlimited)

问题 5: NCCL 初始化超时
  症状: "NCCL error: unhandled system error"
  排查:
    - 检查所有节点的 IB 连通性: ib_write_bw
    - 检查防火墙: NCCL 使用随机高端口
    - 增加 NCCL_COMM_BLOCKING=1 (阻塞初始化, 更稳定)
    - 增加 NCCL_MIN_TIMEOUT=1800
```

### 8.6 性能调优 Checklist

```
□ 确认 IB 设备状态 (ibv_devinfo → ACTIVE)
□ 确认 NVLink 可用 (nvidia-smi topo -m → NV 连接)
□ 启用 GPU Direct RDMA (nvidia-peermem / DMA-BUF)
□ 禁用 ACS (如果需要)
□ 设置 memlock unlimited
□ 配置双端口 IB (NCCL_IB_HCA, NCCL_IB_MERGE_NICS)
□ 启用 PXN (NCCL_PXN_DISABLE=0)
□ 多 QP (NCCL_IB_QPS_PER_CONNECTION=4-8)
□ RoCE: 配置 PFC + ECN + DCQCN
□ 设置 CUDA_DEVICE_MAX_CONNECTIONS=1
□ 启用通信/计算重叠 (NCCL_ALGO=Ring for 大数据)
□ 监控: ibstat, nccl-topo.xml, NCCL_DEBUG=INFO
```

---

## 9. 前沿技术

### 9.1 SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)

```
InfiniBand SHARP:
  - 在交换机内执行 Reduce 操作
  - 数据不需要到达 GPU 就被聚合
  - 延迟: O(1) 与 GPU 数量无关
  - 带宽节省: Reduce 在网络中完成，减少 GPU 间数据量

  NCCL 支持:
    NCCL_ALGO=CollnetChain   - 链式 SHARP
    NCCL_ALGO=CollnetDirect  - 直连 SHARP

  需要:
    - 支持 SHARP 的 IB 交换机 (Mellanox Quantum)
    - SHARP 软件: sharp_daemon 运行

NVLink SHARP (NVLS):
  - NVSwitch 内置 SHARP 引擎
  - H100 NVSwitch 支持
  - NCCL_NVLS_ENABLE=2 (自动检测)
  - 算法: NVLS, NVLSTree
```

### 9.2 NCCL Device API (Device-initiated Communication)

```
传统方式:
  GPU kernel 完成计算 → CPU 发起通信 → GPU 等待

Device API (NCCL 2.19+):
  GPU kernel 直接发起通信，无需 CPU 参与
  → 减少 GPU→CPU→GPU 往返延迟
  → 更好的通信/计算重叠

  API:
    ncclCommGetPeerRank()      - 获取对端 rank
    ncclCommRegister()         - 注册 buffer
    ncclSend() / ncclRecv()    - Device 端点对点
    ncclAllReduce()            - Device 端集合通信
    ncclPutSignal()            - 单侧写入 + 信号 (2.28+)
    ncclWaitSignal()           - 等待信号 (2.28+)
```

### 9.3 MNNVL (Multi-Node NVLink)

```
将 NVLink 扩展到节点间:
  - 多个服务器的 GPU 通过 NVLink Cabinet 连接
  - 对软件透明：看起来像一个超大节点
  - NCCL_MNNVL_ENABLE=1

  潜力:
    → 消除 IB 瓶颈
    → 节点间带宽接近 NVLink 级别
    → 但需要特殊硬件 (NVLink Cabinet, 光纤 NVLink)
```

---

## 10. 学习要点总结

1. **RDMA = 绕过 OS + 零拷贝** — CPU 几乎不参与网络传输，延迟 ~1 μs vs TCP ~50 μs
2. **InfiniBand 是 AI 训练标配** — NDR 400 Gb/s, 原生无损, SHARP offload
3. **RoCEv2 是平替方案** — 在 Ethernet 上实现 RDMA，需配置 PFC/ECN/DCQCN 无损
4. **QP/CQ/MR 是 RDMA 三要素** — Queue Pair 通信, Completion Queue 通知, Memory Registration 内存注册
5. **NCCL 深度集成 RDMA** — 自动检测、多 QP、GPU Direct、PXN、SHARP
6. **GPUDirect RDMA 消除 CPU 拷贝** — NIC 直接 DMA GPU 内存，需要 nvidia-peermem 或 DMA-BUF
7. **NVLink 是节点内，IB 是节点间** — NVLink 900+ GB/s vs IB 50 GB/s，各司其职
8. **PXN 是关键优化** — GPU 通过 NVLink 借用最优路径的 IB NIC
9. **SHARP 是大规模利器** — 交换机内 Reduce，延迟与 GPU 数无关
10. **调试第一步**: ibv_devinfo + NCCL_DEBUG=INFO + ib_write_bw

## 参考

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
- [Introduction to InfiniBand White Paper (Mellanox)](https://www.mellanox.com/pdf/whitepapers/IB_Intro_WP_190.pdf)
- [rdma-core (GitHub)](https://github.com/linux-rdma/rdma-core)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Perftest (ib_write_bw)](https://github.com/linux-rdma/perftest)
