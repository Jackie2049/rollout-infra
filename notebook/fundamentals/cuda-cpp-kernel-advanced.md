# CUDA C++ Kernel Programming — Advanced Topics

> 从 Triton (Python 层) 升级到 CUDA C++ (生产级) 的深度学习笔记
> 2026-06-06 | 专注 AI Infra 工程师必备技能

## 为什么要从 Triton 升级到 CUDA C++

| 维度 | Triton | CUDA C++ |
|------|--------|----------|
| 抽象层级 | Python, 自动内存管理 | C++, 手动内存管理 |
| 性能上限 | ~98% cuBLAS (大矩阵) | 100%+ (手写可超越库) |
| 生产部署 | vLLM 部分用 Triton | DeepSeek (FlashMLA/DeepGEMM) 全 C++ |
| 灵活性 | 受限于 Triton op set | 完全自由 (warp/shmem/cp.async) |
| SM 控制 | 无法直接控制 | 可精确分配 SM 给通信/计算 |
| 学习曲线 | 低 (Python) | 高 (C++ + GPU 架构) |

**结论**: Triton 适合快速原型, CUDA C++ 是生产级 kernel 的唯一选择。AI Infra 工程师必须掌握两者。

## 1. CUDA 内存层级

```
┌─────────────────────────────────────────┐
│ Global Memory (VRAM/HBM)               │  ← 所有线程可见, 最慢 (~900 GB/s)
│   A100: 80GB, RTX 4090: 24GB           │
│   延迟: ~400-800 cycles                │
├─────────────────────────────────────────┤
│ L2 Cache                               │  ← 自动管理, ~4-40MB
│   延迟: ~200-300 cycles                │
├─────────────────────────────────────────┤
│ Shared Memory / L1 Cache               │  ← 手动管理, 每个 SM 独立
│   默认: 48KB shmem + 16KB L1           │
│   可配置: 0-100KB shmem carveout       │
│   延迟: ~30 cycles (比 global 快 10x) │
│   带宽: 同一 bank 无冲突时 ~数 TB/s     │
├─────────────────────────────────────────┤
│ Registers                              │  ← 每线程私有, 最快
│   每个 SM: 65536 个 (Ada/Hopper)       │
│   延迟: ~1 cycle                       │
├─────────────────────────────────────────┤
│ Local Memory                           │  ← spill 到 global, 极慢
│   (register overflow → L1/L2/global)   │
├─────────────────────────────────────────┤
│ Constant Memory                        │  ← 64KB, 只读, broadcast
│   延迟: ~1 cycle (cache hit)           │
└─────────────────────────────────────────┘
```

**关键原则**: 最热数据放 registers, 次热放 shared memory, 其余留 global memory。

### Shared Memory Carveout 配置

```c
// 动态调整 shmem/L1 比例 (Ampere+, SM 8.0+)
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedMemCarveoutMaxShared  // 100KB 给 shmem, 最少给 L1
);
// 或 cudaSharedMemCarveoutMaxL1 (最多给 L1)
// 或 cudaSharedMemCarveoutAuto (驱动自动选择)
```

**实际影响**: RTX 4090 每个 SM 最大 100KB shmem。shmem 越大 → occupancy (并发线程数) 越低。需要用 CUDA Occupancy Calculator 平衡。

## 2. Shared Memory Bank Conflicts

### 32-Bank 架构

```
Bank 0  Bank 1  Bank 2  ... Bank 31
  4B     4B      4B    ...   4B

地址映射: bank_id = (地址 / 4) % 32

float shmem[128];  // 128 × 4B = 512 bytes
// shmem[0]  → Bank 0
// shmem[1]  → Bank 1
// shmem[2]  → Bank 2
// ...
// shmem[31] → Bank 31
// shmem[32] → Bank 0  ← 回到 Bank 0!
// shmem[33] → Bank 1
```

**冲突规则**:
- 同一 warp 中, 多个线程访问 **同一 bank 的不同地址** → 串行化 (n-way conflict)
- 同一 warp 中, 多个线程访问 **同一 bank 的同一地址** → broadcast, 无冲突
- 不同 warp 之间无 bank conflict 影响

### 常见冲突模式

#### 模式 1: Stride-1 访问 (无冲突)
```c
__shared__ float data[256];
// thread i 读 data[i]: 每 thread 不同 bank → 0 conflict
float val = data[threadIdx.x];  // ✅ 完美
```

#### 模式 2: Stride-32 访问 (32-way conflict!)
```c
__shared__ float data[1024];
// thread i 读 data[i * 32]: 所有映射到 Bank 0 → 32x 串行!
float val = data[threadIdx.x * 32];  // ❌ 最差
```

#### 模式 3: 2D 矩阵列访问 (冲突)
```c
__shared__ float tile[32][32];
// 列访问: tile[col][row], row 增 1 → 地址增 32*4B
// 32*4B / 4B = 32 → 映射到同一 bank!
float val = tile[threadIdx.x][0];  // ❌ 列访问 = stride-32
```

### 解决方案: Padding

```c
// 无 padding: 列访问有 32-way conflict
__shared__ float tile_bad[32][32];

// 有 padding: 列访问无冲突
__shared__ float tile_good[32][32 + 1];  // +1 列打破 stride-32 映射
// tile_good[col][row]: 地址 = (row * 33 + col) * 4B
// bank = ((row * 33 + col) * 4 / 4) % 32 = (row * 33 + col) % 32
// 不同 row → 不同 bank! ✅
```

**Padding 原则**: `+1` 列是标准做法, 打破 stride=32 的 bank 映射循环。

### 经典例子: Matrix Transpose

```c
__global__ void transpose_naive(float *out, const float *in, int N) {
    // 无 padding → 列读取有 bank conflict
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * N + x];  // 行写入: 无冲突 ✅
    __syncthreads();
    out[x * N + y] = tile[threadIdx.x][threadIdx.y];  // 列读取: 32-way conflict ❌
}

__global__ void transpose_optimized(float *out, const float *in, int N) {
    // 有 padding → 列读取无冲突
    __shared__ float tile[TILE][TILE + 1];  // ← 关键: +1 padding
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * N + x];  // 行写入: 无冲突 ✅
    __syncthreads();
    out[x * N + y] = tile[threadIdx.x][threadIdx.y];  // 列读取: 无冲突 ✅
}
```

### Bank Conflicts 检测

```bash
# ncu (NVIDIA Compute Profiler) 检测 bank conflicts
ncu --metrics shared_load_transactions_per_request,shared_store_transactions_per_request \
    ./my_kernel_app

# 值 > 1.0 表示有 bank conflict
# 值 = 1.0 表示无冲突
# 值 = 32.0 表示 32-way conflict (最差)
```

## 3. Warp-Level Primitives

### Warp 基础

- Warp = 32 个线程, 同步执行同一指令 (SIMT)
- RTX 4090: 128 SM × 4 warp/SM = 512 warp 并发 (理论)
- Warp 内通信无需 __syncthreads (隐式同步)

### Warp Shuffle 指令 (Legacy API)

```c
// intra-warp 数据交换, 无需 shared memory, 无 bank conflict

// 直接交换: thread i 获得 thread srcLane 的 var 值
float val = __shfl_sync(0xffffffff, var, srcLane);

// 下移: thread i 获得 thread (i - delta) 的 var 值 (用于 reduction)
float val = __shfl_down_sync(0xffffffff, var, delta);

// 上移: thread i 获得 thread (i + delta) 的 var 值
float val = __shfl_up_sync(0xffffffff, var, delta);

// 异或: thread i 获得 thread (i ^ laneMask) 的 var 值 (用于 butterfly reduction)
float val = __shfl_xor_sync(0xffffffff, var, laneMask);

// 0xffffffff = 全 warp 参与的 mask
```

### Warp Reduction Pattern (经典)

```c
// 32 元素求和 reduction, 5 步完成 (log2(32) = 5)
__device__ float warp_reduce_sum(float val) {
    // Step 1: 16 pairs → 16 values  (i XOR i+16)
    val += __shfl_xor_sync(0xffffffff, val, 16);
    // Step 2: 8 pairs → 8 values   (i XOR i+8)
    val += __shfl_xor_sync(0xffffffff, val, 8);
    // Step 3: 4 pairs → 4 values   (i XOR i+4)
    val += __shfl_xor_sync(0xffffffff, val, 4);
    // Step 4: 2 pairs → 2 values   (i XOR i+2)
    val += __shfl_xor_sync(0xffffffff, val, 2);
    // Step 5: 1 pair → 1 value     (i XOR i+1)
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;  // 所有 warp thread 现在持有相同 sum
}
```

**为什么 XOR 而非 DOWN**: Butterfly network 最优 — 每步最大通信距离减半, 模式更均匀。

### Block-Level Reduction (Warp + Shared Memory)

```c
__device__ float block_reduce_sum(float val) {
    // 每个 warp 先内部 reduce
    val = warp_reduce_sum(val);

    // warp 0 的 lane 0 把 warp sum 写入 shmem
    __shared__ float warp_sums[32];  // 最多 32 warp/block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();  // 跨 warp 同步

    // 只用 warp 0 做 final reduction
    if (wid == 0) {
        val = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;  // block 内所有线程可获得最终 sum
}
```

### Cooperative Groups API (现代推荐)

```c
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ float block_reduce_sum_cg(float val) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);  // 32-thread sub-group

    // Warp-level reduce
    val = cg::reduce(warp, val, cg::plus<float>());

    // Warp 0 lane 0 写入 shmem
    __shared__ float warp_sums[32];
    if (warp.meta_group_rank() == 0 && warp.thread_rank() == 0) {
        warp_sums[block.meta_group_rank()] = val;
    }
    block.sync();  // 替代 __syncthreads()

    // Final reduce in warp 0
    if (block.meta_group_rank() == 0) {
        auto warp0 = cg::tiled_partition<32>(block);
        val = (warp0.thread_rank() < block.meta_group_size())
              ? warp_sums[warp0.thread_rank()] : 0.0f;
        val = cg::reduce(warp0, val, cg::plus<float>());
    }
    return val;
}
```

**Cooperative Groups 优势**:
- 类型安全, 不需要手写 mask
- 支持 `tiled_partition<N>` 分任意大小 sub-group
- `block.sync()` 比 `__syncthreads()` 更灵活
- 未来硬件 (Hopper/Blackwell) 新功能只支持 CG API

## 4. 内存合并 (Memory Coalescing)

### 规则

```
Global memory 访问效率取决于 warp 内 32 个 thread 的访问模式:

✅ 合并 (Coalesced):
   - 连续 128 字节 (32 × 4B float) 在一次事务中完成
   - thread i 读 data[i] → 完美合并
   - 带宽利用率: ~90-95%

❌ 非合并 (Uncoalesced):
   - Stride > 1: thread i 读 data[i * stride]
   - 随机地址: 32 次独立事务
   - 带宽利用率: ~5-10%

⚠️ 部分合并:
   - Stride = 2: 2 次事务 (64B each)
   - 带宽利用率: ~50%
```

### 实际影响

```c
// ✅ 合并读取: 性能 ~900 GB/s (RTX 4090 HBM BW)
for (int i = tid; i < N; i += stride) {
    sum += data[i];  // 连续地址
}

// ❌ 随机读取: 性能 ~50-100 GB/s
for (int i = tid; i < N; i += stride) {
    sum += data[random_index[i]];  // 随机地址
}
```

**LLM decode 场景**: Attention 的 KV cache 读取本质是随机访问 (page table 间接寻址), 这就是为什么 decode 是 memory-bound 且难以优化的根本原因。

## 5. Kernel Launch 配置与 Occupancy

### Grid/Block 配置

```c
// 1D 配置
int threads_per_block = 256;  // 每个 block 的线程数
int num_blocks = (N + threads_per_block - 1) / threads_per_block;
my_kernel<<<num_blocks, threads_per_block>>>(...);

// 2D 配置 (适合矩阵操作)
dim3 threads(32, 8);  // 32×8 = 256 threads/block
dim3 blocks((W + 31) / 32, (H + 7) / 8);
my_kernel<<<blocks, threads>>>(...);
```

### Occupancy 计算

```
Occupancy = active_warps_per_SM / max_warps_per_SM

限制因素:
1. Registers per thread: 越多 → occupancy 越低
2. Shared memory per block: 越多 → occupancy 越低
3. Threads per block: 影响但不直接限制

RTX 4090 (SM 8.9, Ada):
- Max warps/SM: 64
- Max threads/SM: 2048
- Max blocks/SM: 32
- Max registers/SM: 65536
- Max shmem/SM: 100KB (可配置)

计算示例:
- Kernel 用 64 regs/thread, 0 shmem
- threads/block = 256 → 8 warps/block
- regs/block = 256 × 64 = 16384
- max blocks = min(32, 65536/16384, 2048/256) = 4 blocks
- active warps = 4 × 8 = 32
- occupancy = 32/64 = 50%
```

### Occupancy 不是越高越好

```
高 occupancy 优势:
- 更多 warp 隐藏内存延迟 (latency hiding)
- 更多并发提高吞吐

高 occupancy 劣势:
- 更多 register pressure → 可能 spill
- L1 cache 减少 → cache hit rate 降低
- shmem 减少 → 无法用足够 shmem

最优 occupancy 通常 50-75%, 不是 100%
```

**vLLM decode kernel**: 低 occupancy (batch=1 可能只用 1 warp) → 极低 GPU 利用率, 这就是为什么 continuous batching 必需。

## 6. Asynchronous Memory Operations

### cp.async (Ampere+, SM 8.0+)

```c
// 异步从 global memory 拷贝到 shared memory
// 在拷贝进行时, GPU 可以执行其他计算

// 单元素异步拷贝 (4B/8B/16B)
__cp_async(&shmem[dst_idx], &global[src_idx], sizeof(float));

// Bulk 异步拷贝 (256B 对齐, Hopper+)
// cp.async.bulk 从 global → shared, 无 CPU 介入
```

**实际应用**: FlashAttention 用 cp.async 预加载下一块 KV 到 shmem, 同时当前块在做 attention 计算 → 通信/计算重叠。

### Double Buffering Pattern

```c
__shared__ float buffer[2][TILE_SIZE];  // 两个缓冲区
int current = 0, next = 1;

// 第一步: 异步加载到 next buffer
__cp_async(&buffer[next][0], &global[offset], TILE_SIZE * sizeof(float));

for (int tile = 0; tile < num_tiles; tile++) {
    // 等待 next buffer 数据到达
    __cp_async_commit_group();
    __cp_async_wait_group<0>();  // 等待所有 pending async copy 完成

    // 交换 current 和 next
    current = 1 - current;
    next = 1 - next;

    // 处理 current buffer (计算)
    process_tile(buffer[current]);

    // 同时异步加载下一块到 next buffer
    if (tile + 1 < num_tiles) {
        __cp_async(&buffer[next][0], &global[offset + (tile+1)*TILE_SIZE],
                    TILE_SIZE * sizeof(float));
    }
}
```

## 7. Tensor Core Integration

### WMMA (Warp Matrix Multiply-Accumulate)

```c
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// 从 global/shared memory 加载到 fragment
load_matrix_sync(a_frag, a_ptr, 16);  // 16×16 half matrix
load_matrix_sync(b_frag, b_ptr, 16);

// 初始化 accumulator
fill_fragment(c_frag, 0.0f);

// Tensor Core MMA: 16×16 × 16×16 → 16×16
mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
store_matrix_sync(c_ptr, c_frag, 16, fragment<accumulator, 16, 16, 16, float>);
```

### MMA.sync (Hopper+, 更灵活)

```c
// Hopper SM90: mma.sync 支持 16×8×16, 16×16×16 等多种形状
// FP8 输入: FP8 E4M3 × FP8 E5M2 → FP32/FP16 accumulator
// 这就是 DeepGEMM 在 H800 上达到 1550 TFLOPS 的基础
```

**关键**: DeepSeek 生态 (FlashMLA/DeepGEMM) 都大量使用 mma.sync + FP8, 这是未来 kernel 的主流。

## 8. PyTorch C++ Extension 工作流

### 项目结构

```
my_kernel/
├── setup.py               ← CUDAExtension 配置
├── csrc/
│   ├── my_kernel.cpp      ← pybind11 绑定
│   └── my_kernel_cuda.cu  ← CUDA kernel 实现
└── my_kernel/
    └── __init__.py         ← Python 接口
```

### setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_kernel',
    ext_modules=[
        CUDAExtension(
            name='my_kernel._C',
            sources=[
                'csrc/my_kernel.cpp',
                'csrc/my_kernel_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

### C++ 绑定 (csrc/my_kernel.cpp)

```cpp
#include <torch/extension.h>

// CUDA forward/backward 声明
torch::Tensor my_kernel_forward(torch::Tensor input, torch::Tensor weight, float eps);
std::vector<torch::Tensor> my_kernel_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, float eps);

// pybind11 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &my_kernel_forward, "My kernel forward (CUDA)");
    m.def("backward", &my_kernel_backward, "My kernel backward (CUDA)");
}
```

### Python 接口 (my_kernel/__init__.py)

```python
import torch
from my_kernel._C import forward as _forward, backward as _backward

class MyKernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        output = _forward(input, weight, eps)
        ctx.save_for_backward(input, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = _backward(grad_output, input, weight, ctx.eps)
        return grad_input, grad_weight, None

def my_kernel(input, weight, eps=1e-6):
    return MyKernelFunction.apply(input, weight, eps)
```

### 编译与使用

```bash
# 开发模式 (就地编译, 不安装)
python setup.py build_ext --inplace

# 安装模式
pip install .

# 使用
import my_kernel
output = my_kernel.my_kernel(input_tensor, weight_tensor, 1e-6)
```

### 关键 API

```cpp
// Tensor 操作
torch::Tensor input = ...;  // 来自 Python
auto sizes = input.sizes();  // shape
auto dtype = input.dtype();  // 类型
auto device = input.device();  // 设备

// 检查
TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
TORCH_CHECK(input.dim() == 2, "input must be 2D");

// 分配输出
auto output = torch::empty_like(input);  // 同 shape/dtype/device
auto output = torch::empty(sizes, input.options());  // 显式

// 数据指针
float* out_ptr = output.data_ptr<float>();
const float* in_ptr = input.data_ptr<float>();

// Kernel launch
int threads = 256;
int blocks = (N + threads - 1) / threads;
my_kernel_cuda<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    out_ptr, in_ptr, weight_ptr, N, hidden_size, eps);
```

## 9. Fused Kernel 设计模式

### 为什么 Fusion 有效

```
单 op: [load] → [compute] → [store] → [load] → [compute] → [store] → ...
融合:   [load] → [compute1] → [compute2] → [compute3] → [store]

收益:
1. 减少 global memory 访问 (中间结果只在 registers/shmem 中)
2. 减少 kernel launch 开销 (~6-34 us/launch)
3. 减少 L2 cache 压力
4. 更好的 GPU 利用率 (更大计算粒度)
```

### Fused RMSNorm + Residual Add

```
单独:
  RMSNorm: load x → compute → store norm_x  (1 global read + 1 global write)
  Add:     load norm_x + residual → add → store y (2 global reads + 1 global write)

融合:
  FusedRMSNormAdd: load x + residual → rms_norm → add → store y
  (2 global reads + 1 global write, 中间结果只在 registers)
  → 节省 1 global read + 1 global write = 节省 ~50% memory traffic
```

### Fused Bias + ReLU (已在 A16 实测 2.32x 加速)

```
单独:
  Linear: GEMM → store output
  Bias:   load output → add bias → store
  ReLU:   load → relu → store

融合:
  FusedBiasReLU: GEMM → bias → relu → store (1 write instead of 3)
```

### DeepGEMM Mega MoE (终极融合)

```
融合整个 MoE 层为一个 kernel:
  EP dispatch → FP8 dequant → FP8×FP4 GEMM → SwiGLU → FP8 requant → EP combine

收益: 通信+计算在一个 kernel 内重叠, 无中间 global memory 读写
```

## 10. RTX 4090 CUDA 编程要点

### 硬件特性 (SM 8.9, Ada Lovelace)

| 特性 | 数值 |
|------|------|
| SM 数 | 128 |
| CUDA Cores/SM | 128 |
| Warp/SM | 48 (max) |
| Registers/SM | 65536 |
| Shared Memory/SM | 100KB (可配置) |
| Tensor Cores/SM | 4 (4th gen) |
| HBM BW | 1008 GB/s (实测 945 GB/s) |
| FP16 TFLOPS | 82.6 (理论), 169.6 (实测 GEMM) |

### 编译注意事项

```bash
# nvcc 编译目标架构
nvcc -arch=sm_89  # RTX 4090

# PyTorch C++ Extension 自动设置 -arch
# 但可通过 TORCH_CUDA_ARCH_LIST 指定
export TORCH_CUDA_ARCH_LIST="8.9"  # 只编译 4090
```

### Performance Checklist

1. ✅ Memory coalescing: 确保 global memory 访问是 stride-1
2. ✅ No bank conflicts: shmem 用 padding 或优化访问模式
3. ✅ Minimize register usage: 用 `--ptxas-options=-v` 检查
4. ✅ Appropriate occupancy: 50-75% 通常最优
5. ✅ Use Tensor Cores: FP16/BF16 矩阵操作用 WMMA/mma.sync
6. ✅ Double buffering: 用 cp.async 预加载下一块数据
7. ✅ Warp-level primitives: 用 shuffle 替代 shmem 做 intra-warp 通信
8. ✅ CUDA Graph: 减少 launch 开销 (但 RTX 4090 launch 仅 1.1us, 收益有限)
9. ✅ Avoid branch divergence: warp 内分支导致串行执行
10. ✅ Minimize __syncthreads: 只在必要时同步, 过多同步浪费

## 11. 从 Triton 到 CUDA C++ 的对照

| Triton 概念 | CUDA C++ 对应 |
|-------------|--------------|
| `tl.program_id(0)` | `blockIdx.x` |
| `tl.arange(0, BLOCK)` | `threadIdx.x` (需要手动范围管理) |
| `tl.load(ptr, mask)` | 直接指针访问 + 条件判断 |
| `tl.store(ptr, val, mask)` | 直接指针写入 + 条件判断 |
| `tl.sum(val, axis=0)` | Warp/Block reduce (手写) |
| 自动 SRAM 管理 | 手动 `__shared__` + bank conflict 管理 |
| 自动边界处理 | 手动 boundary check |
| 自动类型推断 | 手动 template `<typename T>` |

**Triton 优势**: 自动化太多细节, 生产力高
**CUDA C++ 优势**: 完全控制, 可达极致性能

## 12. 实战 Kernel 写作流程

```
1. Profile → 找瓶颈 (ncu/nsight)
2. 设计 kernel → 选择 fusion scope
3. 写 Triton prototype → 快速验证正确性
4. 写 CUDA C++ → 精细优化 (shmem/warp/tensor core)
5. Benchmark → 对比 cuBLAS/PyTorch/Triton
6. Integrate → PyTorch C++ Extension 或直接 vLLM backend
```

**当前项目中的示例**:
- Top-nσ: Triton 实现完成 (2.09x 加速 on 128K vocab)
- Fused RMSNorm + Residual Add: 下一步用 CUDA C++ 实现 (预计更大收益)

---

## 参考资料

- NVIDIA CUDA C++ Best Practices Guide (2025)
- NVIDIA CUDA Programming Guide
- Professional CUDA C Programming (Cheng et al.)
- FlashAttention paper (tiling + online softmax)
- DeepGEMM source code (Mega MoE fusion, FP8 GEMM)
- FlashMLA source code (Seesaw scheduling, Crossover dequantize)
- PyTorch C++ Extension tutorial
- GTC 2025 sessions on Hopper/Blackwell optimization