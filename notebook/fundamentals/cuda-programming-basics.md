# CUDA 编程基础 — AI Infra 工程师必备

> 理解 GPU 编程模型是推理优化和 kernel 开发的基础

## 1. 执行模型层次

```
Grid (整个 GPU)
  └── Block[0]     Block[1]     ...    Block[N-1]    ← 独立调度到不同 SM
        └── Warp 0   Warp 1   ...   Warp M          ← 32 threads, SIMT 执行
              └── Thread 0  Thread 1  ...  Thread 31 ← 最小执行单元
```

### 关键约束

| 资源 | A100 (SM 8.0) | H100 (SM 9.0) |
|------|---------------|---------------|
| SM 数量 | 108 | 132 |
| 最大线程/SM | 2048 | 2048 |
| 最大线程/Block | 1024 | 1024 |
| Shared Memory/SM | 164 KB | 228 KB |
| 寄存器/SM | 65536 | 65536 |
| 最大 Block/SM | 32 | 32 |

### Warp 是调度基本单位

- 1 Warp = 32 个 Thread, **同时执行相同指令** (SIMT)
- 如果 Thread 走不同分支 → **两个分支串行执行** (Warp Divergence)
- Block Size 应该是 32 的倍数 (通常 128/256/512)

## 2. 内存层次

```
寄存器 (Thread 私有, 1 cycle, ~0.3 TB/s)
  ↓
Shared Memory (Block 共享, ~5 cycles, ~19 TB/s, 可编程)
  ↓
L1 Cache (SM 内, ~30 cycles, 与 SMEM 共享物理空间)
  ↓
L2 Cache (全局, ~200 cycles, ~3 TB/s)
  ↓
HBM Global Memory (全局, ~400 cycles, ~2 TB/s, 80GB)
```

### 数据复用: GEMM 的核心

矩阵乘法 C[M,N] = A[M,K] × B[K,N], tile=64:

| 策略 | AI (ops/byte) | 相比朴素 |
|------|--------------|---------|
| 朴素 (无分块) | ~250M (因为 N²/K 大) | 1x |
| Global Memory 分块 | 更优 | 64x |
| Shared Memory Tiling | 更优 | 64x (SMEM 复用) |

**核心思想**: 每个 tile 从 Global Memory 读入 Shared Memory 后, Block 内所有 Thread 共享复用。

## 3. Coalesced Memory Access

Warp 内 32 个 Thread 的 memory access 合并为尽可能少的 transaction:

- **Coalesced**: Thread 0 读 addr[0], Thread 1 读 addr[1], ... → 1 次 128B transaction
- **Strided-2**: 每 2 个元素读 1 个 → 2x transactions, 有效 BW 减半
- **Random**: 完全随机 → 32x transactions

**实践**: 数据布局要匹配访问模式 (AoS → SoA)

## 4. Occupancy 分析

Occupancy = Active Threads / Max Threads per SM

### 三种限制因素

```python
blocks_by_threads = max_threads_per_sm // threads_per_block
blocks_by_regs = reg_per_sm // (regs_per_thread * threads_per_block)
blocks_by_shared = shared_mem_per_sm // shared_mem_per_block
active_blocks = min(blocks_by_threads, blocks_by_regs, blocks_by_shared)
```

### 典型 Kernel Occupancy (A100)

| Kernel | Threads/Block | Regs/Thread | SMEM/Block | Occupancy | 瓶颈 |
|--------|-------------|------------|-----------|-----------|------|
| 向量加法 | 256 | 32 | 0 | 100% | threads |
| 矩阵乘法 | 256 | 64 | 0 | 50% | registers |
| Conv | 128 | 48 | 16KB | 62.5% | registers |
| Flash Attention | 256 | 32 | 64KB | 25% | shared_mem |
| Flash Attn (大 SMEM) | 256 | 64 | 100KB | 12.5% | shared_mem |

**关键**: 100% Occupancy 不一定最优。Memory-bound 需要 high occupancy 隐藏延迟; compute-bound 可以用更多资源 per thread。

## 5. Roofline Model

### Arithmetic Intensity (AI)

```
AI = FLOPs / Bytes_Accessed  (ops/byte)
```

### Ridge Point

```
Ridge Point = Peak_FLOPS / Peak_BW
A100: 312 TFLOPS / 2035 GB/s ≈ 153 ops/byte
```

- **AI < Ridge Point → Memory-Bound**: 性能受限于带宽, 优化重点是减少内存访问
- **AI > Ridge Point → Compute-Bound**: 性能受限于算力, 优化重点是增加计算吞吐

### LLM 推理的 Roofline 分析

| Kernel | AI (ops/byte) | Bound | Peak 利用率 |
|--------|-------------|-------|-----------|
| Vector Add | 0.1 | memory | 0.1% |
| Softmax | 0.8 | memory | 0.5% |
| LayerNorm | 0.8 | memory | 0.5% |
| GEMM (128³) | 16 | memory | 10.4% |
| GEMM (4096³) | 512 | compute | 100% |
| **Decode Step (7B)** | **~1.0** | **memory** | **0.7%** |
| Flash Attn (4K) | 1.3 | memory | 0.9% |

**核心洞察**: LLM Decode 始终 memory-bound, 优化重点是 HBM 带宽而非算力。这就是为什么:
- H200 (4800 GB/s) 比 H100 (3350 GB/s) 对推理提升巨大
- 量化 (FP8/INT4) 直接减少内存访问量
- Speculative Decoding 通过增加 compute 比例来提高算力利用率

## 6. Flash Attention 的 CUDA 优化

Flash Attention 是 GPU kernel 优化的教科书级案例:

### 问题: 标准 Attention

```
Q [B,H,N,D] × K^T [B,H,D,N] → S [B,H,N,N]  → P [B,H,N,N] → O [B,H,N,D]
                              ↑
                     N² 内存, 写回 Global Memory
```

### 解决: Tiling + Online Softmax

1. **外层循环**: 遍历 K/V 的 block (Br rows at a time)
2. **内层循环**: 遍历 Q 的 block (Bc rows at a time)
3. **Online Softmax**: 不需要全部 S, 用 running max/sum 累积
4. **SRAM 利用**: S 和 P 只在 Shared Memory/Register, 不写回 Global

### 内存访问分析

```
标准: O(N²) Global Memory writes (S 矩阵)
Flash: O(N×d) Global Memory reads/writes (只有 Q, K, V, O)
节省: N²/d 倍 (seq=4K, d=128 → 128x)
```

## 7. 实践: 编写第一个 CUDA Kernel

### Vector Add 示例

```cuda
__global__ void vectorAdd(float* C, const float* A, const float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // Coalesced access ✓
    }
}

// Launch: <<<grid, block>>>
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;
vectorAdd<<<gridSize, blockSize>>>(C, A, B, N);
```

### Matrix Multiply (Naive → Tiled)

```cuda
// Naive: 每个线程读整行 A + 整列 B = O(K) global reads
__global__ void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];  // K 次 global read
    }
    C[row * N + col] = sum;
}

// Tiled: Shared Memory 复用
__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 协作读入 tile 到 Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();  // Block 内同步

        // 在 Shared Memory 上计算
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];  // SMEM read!
        }
        __syncthreads();  // 确保 tile 计算完成再覆盖
    }
    C[row * N + col] = sum;
}
```

### 性能对比

| 版本 | Global Memory 读取 | Shared Memory 读取 | AI (ops/byte) |
|------|-------------------|-------------------|--------------|
| Naive | 2K per output | 0 | 1 (memory-bound) |
| Tiled (T=64) | 2×(M×N×K/T) | 2×T per output per step | ~T (64x 更优) |

## 8. CUDA → LLM 推理的联系

```
CUDA 概念          → LLM 推理应用
─────────────────────────────────────
Coalesced Access   → 权重矩阵行优先存储
Shared Memory      → Flash Attention tiling
Warp Divergence    → MoE expert routing 优化
Occupancy          → Batch size 调优
Roofline Model     → Decode memory-bound 优化策略
Register Pressure  → Attention kernel 寄存器分配
```

## 参考资料

- CUDA C++ Programming Guide (NVIDIA docs)
- GPU 并行编程 (UC Davis ECS 290A)
- FlashAttention 论文 (Dao et al., 2022)
- `tools/cuda_kernel_simulator.py` — 本笔记配套模拟器
