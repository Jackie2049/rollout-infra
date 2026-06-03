# CUDA 编程基础

> GPU 并行计算的核心 — 理解硬件才能写出高效的 kernel

## 1. GPU 硬件架构

### 1.1 A100 关键参数

```
NVIDIA A100-SXM4-80GB:
  CUDA Cores:          6,912
  Tensor Cores (3rd):  432
  SM 数量:             108
  GPU 基频:            1.41 GHz
  峰值 FP32:           19.5 TFLOPS
  峰值 TF32:           156 TFLOPS (with Tensor Core)
  峰值 FP16:           312 TFLOPS
  峰值 BF16:           312 TFLOPS
  峰值 FP8:            624 TFLOPS (H100 only)
  HBM2e 带宽:          2,039 GB/s
  HBM2e 容量:          80 GB
  L2 Cache:            40 MB
  TDP:                 400W
  NVLink:              600 GB/s (第三代)
  PCIe Gen4:           64 GB/s

关键比率:
  算力/带宽 = 312 TFLOPS / 2039 GB/s ≈ 153 FLOP/Byte
  → 计算密集型操作才能充分利用 GPU
  → AI training 的矩阵乘法: ~100 FLOP/byte (正好在甜点区)
```

### 1.2 H100 升级要点

```
NVIDIA H100-SXM5-80GB:
  SM 数量:             132
  Tensor Cores (4th):  528
  峰值 FP16:           990 TFLOPS
  峰值 FP8:            1,979 TFLOPS ← 2x FP16
  HBM3 带宽:           3,350 GB/s
  HBM3 容量:           80 GB
  NVLink:              900 GB/s (第四代)
  TDP:                 700W

核心变化:
  1. FP8 支持 — 训练和推理都能用
  2. TMA (Tensor Memory Accelerator) — 异步内存搬运
  3. Thread Cluster — 编程模型新增层次
  4. 分布式共享内存 — 跨 SM 共享内存访问
```

### 1.3 Tensor Core 原理

```
传统 CUDA Core: 一次做 1 个乘加 (FMA)
  D = A * B + C  → 1 clock cycle, 1 element

Tensor Core: 一次做矩阵乘加 (MMA)
  D[4×4] = A[4×8] × B[8×4] + C[4×4]  → 1 clock cycle, 32 elements!

A100 Tensor Core (TF32):
  每个 SM: 1 个 MMA 指令 → 16×8×8 矩阵块
  108 个 SM × 1.41 GHz = 156 TFLOPS

为什么 AI 训练能利用 Tensor Core:
  矩阵乘法 GEMM: C[M,N] = A[M,K] × B[K,N]
  大量独立的 MMA 操作 → 充分填满所有 Tensor Core
```

## 2. CUDA 编程模型

### 2.1 线程层次

```
Grid → Block → Thread 三级层次:

Grid (整个 kernel 启动的线程集合)
  ├── Block 0 (在一个 SM 上执行)
  │     ├── Thread 0
  │     ├── Thread 1
  │     ├── ...
  │     └── Thread 255  (最多 1024 threads/block)
  ├── Block 1
  │     └── ...
  └── Block N-1

关键约束:
  - Block 内线程可以同步 (__syncthreads) 和共享内存
  - Block 间不能直接同步 (只能通过 kernel 结束隐式同步)
  - 每个 Block 映射到 1 个 SM
  - 1 个 SM 可同时运行多个 Block (提高 occupancy)
```

### 2.2 内存层次

```
从快到慢:

Registers (线程私有)
  延迟: ~1 cycle
  带宽: ~20 TB/s
  容量: 255 registers/thread (A100)
  用途: 局部变量、循环计数器

Shared Memory (Block 内共享)
  延迟: ~20-30 cycles
  带宽: ~19 TB/s
  容量: 164 KB/SM (可配置, A100)
  用途: Block 内数据共享、缓存热点数据
  声明: __shared__ float tile[256];

L1 Cache (SM 私有)
  与 Shared Memory 共享物理存储 (可配置分配比例)

L2 Cache (所有 SM 共享)
  延迟: ~200-300 cycles
  容量: 40 MB (A100)
  用途: 自动缓存全局内存访问

Global Memory (HBM)
  延迟: ~400-800 cycles
  带宽: 2,039 GB/s (A100)
  容量: 80 GB
  用途: 主要数据存储

优化原则:
  1. 尽量用 Registers 和 Shared Memory
  2. Global Memory 访问要合并 (coalesced)
  3. 利用 L2 Cache 局部性
```

### 2.3 合并访问 (Coalesced Access)

```
好的模式 — 连续线程访问连续地址:
  Thread 0: addr[0]
  Thread 1: addr[1]
  Thread 2: addr[2]
  ...
  → 一次内存事务完成 128 字节

坏的模式 — 随机或跨步访问:
  Thread 0: addr[0]
  Thread 1: addr[128]  (stride=128)
  Thread 2: addr[256]
  ...
  → 每个线程一次内存事务 → 带宽利用率 1/32!

矩阵转置的经典优化:
  __global__ void transpose(float* out, float* in, int N) {
      __shared__ float tile[TILE][TILE+1]; // +1 避免 bank conflict
      // 1. 从 global 读入 shared memory (coalesced 读)
      // 2. __syncthreads()
      // 3. 从 shared memory 写入 global (coalesced 写)
  }
```

## 3. 流与事件

### 3.1 CUDA Stream

```
Stream: GPU 上的命令队列
  - 同一 Stream 内操作串行执行
  - 不同 Stream 的操作可以并行执行

Default Stream (Stream 0):
  - 与所有其他 Stream 同步
  - 默认所有操作在此 Stream

多 Stream 实现重叠:
  Stream 1: [H2D copy] → [Kernel A] → [D2H copy]
  Stream 2: [H2D copy] → [Kernel B] → [D2H copy]
  Stream 3: [H2D copy] → [Kernel C] → [D2H copy]
  → 拷贝和计算并行!

PyTorch 中使用:
  s1 = torch.cuda.Stream()
  s2 = torch.cuda.Stream()
  with torch.cuda.stream(s1):
      a = data.to('cuda', non_blocking=True)
  with torch.cuda.stream(s2):
      b = data.to('cuda', non_blocking=True)
  torch.cuda.synchronize()
```

### 3.2 CUDA Event

```
Event: GPU 时间线上的标记点
  - 精确计时 (GPU 时间)
  - Stream 间同步

用途 1: 计时
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  # ... GPU 操作 ...
  end.record()
  torch.cuda.synchronize()
  elapsed_ms = start.elapsed_time(end)

用途 2: Stream 间同步
  s1 = torch.cuda.Stream()
  s2 = torch.cuda.Stream()
  with torch.cuda.stream(s1):
      a = compute_a()
      event = torch.cuda.Event()
      event.record()
  with torch.cuda.stream(s2):
      event.wait()  # 等 s1 完成
      b = compute_b(a)
```

## 4. Occupancy 与性能

### 4.1 Occupancy

```
Occupancy = 活跃 warp 数 / SM 最大 warp 数

影响 Occupancy 的因素:
  1. 每个 Block 的线程数 (建议 128 或 256)
  2. 每个 Block 的 Shared Memory 用量
  3. 每个线程的 Register 用量

A100 限制:
  每个 SM: 最多 64 warp (2048 threads)
  每个 SM: 最多 32 Block
  每个 SM: 最多 164 KB Shared Memory
  每个 SM: 最多 65536 Registers

计算 Occupancy:
  threads_per_block = 256
  regs_per_thread = 32
  shared_per_block = 8192  # 8 KB

  blocks_per_sm = min(
      2048 / 256,          # 8 (thread limit)
      32,                  # block limit
      65536 / (256 * 32),  # 8 (register limit)
      164 * 1024 / 8192    # 20 (shared memory limit)
  ) = 8

  occupancy = (8 * 256 / 32) / 64 = 100% ✓

工具: ncu (NVIDIA Nsight Compute) — 分析 kernel 性能
命令: ncu --set full -o profile python train.py
```

### 4.2 带宽利用率

```
判断 kernel 是否受带宽限制:

理论带宽: 2039 GB/s (A100)
实际带宽测量: bytes_moved / elapsed_time

Roofline Model:
  | 运算强度 (FLOP/Byte)
  |     /
  |    /  ← 带宽限制区 (slope = peak_bandwidth)
  |   /
  |  /_______________ ← 计算限制区 (flat = peak_flops)
  |___________________
       FLOP/Byte

AI kernel 的运算强度:
  Element-wise (add, relu):  ~1 FLOP/Byte → 带宽限制
  GEMM (matmul):             ~100 FLOP/Byte → 计算限制
  Reduction (sum, max):      ~0.25 FLOP/Byte → 带宽限制
  Attention (FlashAttn):     ~5-20 FLOP/Byte → 中间地带
```

## 5. 常用 CUDA 库

### 5.1 库概览

```
cuBLAS — 线性代数 (矩阵乘法)
  核心: GEMM (C = αAB + βC)
  Tensor Core 自动利用
  PyTorch 的 matmul 底层调用 cuBLAS

cuDNN — 深度学习算子
  卷积 (convolution forward/backward)
  池化 (pooling)
  归一化 (batch norm, layer norm)
  RNN / Attention

NCCL — 集合通信
  AllReduce, AllGather, ReduceScatter, Broadcast
  多 GPU / 多节点通信
  详见 fundamentals/nccl.md

Triton — OpenAI 的 GPU 编程语言
  比 CUDA 更高层次的抽象
  自动内存管理、自动调优
  FlashAttention 的 Triton 实现
  PyTorch 2.0 compile 后端之一
```

### 5.2 cuBLAS GEMM 示例

```
GEMM: C[M,N] = A[M,K] × B[K,N]

A100 性能 (FP16 Tensor Core):
  M=K=N=4096: ~155 TFLOPS (接近峰值 156)
  M=K=N=1024: ~120 TFLOPS
  M=K=N=256:  ~50 TFLOPS
  M=K=N=64:   ~5 TFLOPS (小矩阵开销大)

Megatron TP 的通信-计算重叠:
  ColumnParallelLinear:
    1. 计算 Y = XA (本地 GEMM)
    2. AllGather(Y) (通信)
  用 CUDA Stream 让 1 和 2 重叠
  需要 CUDA_DEVICE_MAX_CONNECTIONS=1 确保通信优先
```

## 6. GPU 监控与调试

### 6.1 nvidia-smi

```bash
# 基础监控
nvidia-smi                    # GPU 使用概况
watch -n 1 nvidia-smi         # 每秒刷新

# 详细信息
nvidia-smi -q -d MEMORY       # 显存详情
nvidia-smi -q -d UTILIZATION  # 利用率
nvidia-smi -q -d TEMPERATURE  # 温度

# 进程级信息
nvidia-smi pmon -c 5          # 进程级 GPU 使用 (5次采样)

# 关键指标:
#   GPU-Util: GPU 计算利用率 (不是显存利用率!)
#   Memory-Usage: 显存使用
#   GPU-Util 高但训练慢 → 可能通信瓶颈
#   GPU-Util 低 → 数据加载或 CPU 瓶颈
```

### 6.2 Nsight 工具

```bash
# Nsight Systems — 时间线分析
nsys profile -o trace python train.py
# 在 Nsight Systems GUI 中查看:
#   - Kernel 执行时间线
#   - 内存拷贝时间线
#   - CPU-GPU 重叠情况
#   - NCCL 通信时间

# Nsight Compute — 单 kernel 分析
ncu --set full -o kernel_profile python train.py
# 分析:
#   - Occupancy
#   - 带宽利用率
#   - 计算吞吐量
#   - warp stall 原因
#   - 内存访问模式
```

## 7. 学习要点

1. **Tensor Core 是 GPU 的核心** — AI 训练的 GEMM 操作正是 Tensor Core 的最佳场景
2. **内存层次决定性能** — Shared Memory 比 Global Memory 快 10x，用 tile 减少 Global 访问
3. **Coalesced Access 是必须的** — 连续线程访问连续地址，否则带宽暴跌
4. **Occupancy 不是越高越好** — 100% occupancy 不一定最快，但太低肯定慢
5. **Stream 实现并行** — 多 Stream 让拷贝和计算重叠，DataLoader 预取的底层机制
6. **小矩阵是大问题** — GEMM 在小尺寸时效率很低，是 TP/PP 切分的约束

## 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA A100 Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/a100/)
- [OpenAI Triton](https://github.com/openai/triton)
- [NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
