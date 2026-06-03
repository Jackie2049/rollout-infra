# Triton 编程基础

> OpenAI 的 GPU 编程语言 — 比 CUDA 更高层，性能接近手写 kernel

## 1. 为什么学 Triton

```
CUDA kernel 开发痛点:
  1. 需要手动管理 shared memory、寄存器分配
  2. 需要处理内存合并 (coalescing)、bank conflict
  3. 需要调优 block size、grid size、occupancy
  4. 代码量大，开发周期长

Triton 解决方案:
  1. 自动内存管理 — 编译器处理 shared memory 和数据搬运
  2. 自动调优 — @triton.autotune 搜索最优配置
  3. Python 语法 — 类 NumPy 的编程体验
  4. 性能接近手写 CUDA — 编译到 PTX，利用 Tensor Core

谁在用 Triton:
  - FlashAttention-2/3 的 Triton 实现
  - vLLM 的自定义 kernel (部分)
  - PyTorch 2.0 torch.compile 的 Inductor 后端
  - xFormers 的注意力 kernel
```

## 2. 编程模型

### 2.1 基本概念

```python
import triton
import triton.language as tl
import torch

# Triton kernel: 在 GPU 上执行的函数
# @triton.jit 标记为 GPU kernel
@triton.jit
def vector_add_kernel(
    x_ptr,      # 输入指针 (device memory)
    y_ptr,      # 输入指针
    output_ptr, # 输出指针
    n_elements, # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    # 1. 计算 program ID (每个 instance 处理一部分数据)
    pid = tl.program_id(axis=0)

    # 2. 计算 offset (当前 instance 处理的数据范围)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 创建 mask (处理边界)
    mask = offsets < n_elements

    # 4. 加载数据 (从 global memory)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. 计算
    output = x + y

    # 6. 存储结果 (到 global memory)
    tl.store(output_ptr + offsets, output, mask=mask)


# 启动 kernel
def vector_add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 2.2 与 CUDA 的对应关系

```
CUDA                           Triton
─────────────────────────────────────────────
threadIdx.x                   tl.arange(0, BLOCK_SIZE)
blockIdx.x                    tl.program_id(axis=0)
gridDim.x × blockDim.x        由 grid lambda 决定
__shared__ float tile[...]     自动管理
__syncthreads()                不需要 (block 内同步自动)
float4 vec = *(float4*)addr    tl.load(ptr + offsets)
*(float4*)addr = vec           tl.store(ptr + offsets, value)
if (tid < N)                   mask = offsets < N
```

### 2.3 内存模型

```
Triton 自动管理三级内存:

1. Global Memory (HBM)
   tl.load(ptr + offsets, mask=..., eviction_policy=...)
   tl.store(ptr + offsets, value, mask=...)
   自动合并访问 (coalescing)
   支持 eviction_policy: 'evict_last', 'evict_first'

2. Shared Memory (自动)
   编译器自动将频繁访问的数据放入 shared memory
   开发者不需要手动管理

3. Registers (自动)
   标量变量自动分配到寄存器

指针算术:
   ptr = base_ptr + stride_0 * offset_0 + stride_1 * offset_1
   # 2D 访问示例
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

## 3. 核心编程模式

### 3.1 矩阵乘法 (GEMM)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,       # 指针
    M, N, K,                    # 维度
    stride_am, stride_ak,       # A 的 stride
    stride_bk, stride_bn,       # B 的 stride
    stride_cm, stride_cn,       # C 的 stride
    BLOCK_SIZE_M: tl.constexpr, # 编译时常量
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID → 2D block 索引
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # 计算 block 的起始位置
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 沿 K 维度迭代 (tiling)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 的 tile: [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptr + offs_m[:, None] * stride_am +
                    (offs_k[None, :] + k * BLOCK_SIZE_K) * stride_ak)
        # 加载 B 的 tile: [BLOCK_K, BLOCK_N]
        b = tl.load(b_ptr + (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_bk +
                    offs_n[None, :] * stride_bn)
        # 矩阵乘加 (利用 Tensor Core)
        accumulator += tl.dot(a, b)  # ← 自动利用 Tensor Core!

    # 转换输出精度
    c = accumulator.to(tl.float16)

    # 写回结果
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)
```

### 3.2 FlashAttention 简化版

```python
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,  # 指针
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    N: tl.constexpr,              # sequence length
    d: tl.constexpr,              # head dimension
    BLOCK_N: tl.constexpr,        # KV block size
    BLOCK_D: tl.constexpr,        # head dim block
    scale: tl.constexpr,
):
    # Program ID → (query_idx, head_idx)
    pid = tl.program_id(0)
    head_idx = pid % num_heads
    query_idx = pid // num_heads

    # 加载 Q tile: [1, d]
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q_ptr + query_idx * stride_qm + head_idx * stride_qh +
                offs_d * stride_qd)  # shape: [d]

    # Online softmax: 逐 block 处理 KV
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([1], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)             # output accumulator

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载 K tile: [BLOCK_N, d]
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn +
                     head_idx * stride_kh + offs_d[None, :] * stride_kd)
        # 计算 attention scores: [BLOCK_N]
        qk = tl.sum(q[None, :] * k, axis=1) * scale
        # Mask out-of-range
        qk = tl.where(offs_n < N, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + tl.sum(beta)

        # 加载 V tile: [BLOCK_N, d]
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn +
                     head_idx * stride_vh + offs_d[None, :] * stride_kd)
        # 更新累加器
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # 最终输出
    acc = acc / l_i
    # 写回 O
    tl.store(O_ptr + query_idx * stride_om + head_idx * stride_oh +
             offs_d * stride_od, acc)
```

### 3.3 Reduction (Softmax)

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D reduction: 每个 program 处理一行
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 加载一行
    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float('inf'))

    # Online softmax: 数值稳定
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    # 写回
    tl.store(output_ptr + row_start + offsets, output, mask=mask)
```

## 4. 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 32}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128, 'BN': 32,  'BK': 32}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # 按这些参数分组调优
)
@triton.jit
def matmul_kernel_tuned(
    # ... 同上 ...
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  # 被调优
):
    # kernel 实现
    ...

# 调优参数说明:
# num_stages: 流水线级数 (software pipelining)
#   - 越大 → 隐藏延迟越好，但 shared memory 用量越大
#   - A100: 建议 3-5
# num_warps: 每 block 的 warp 数
#   - 影响 occupancy 和 SM 利用率
#   - 常见值: 4, 8
```

## 5. Triton vs CUDA 性能对比

```
操作                   Triton        CUDA          比值
──────────────────────────────────────────────────────
GEMM (M=N=K=4096)     155 TFLOPS    156 TFLOPS    99%
FlashAttention         158 TFLOPS    160 TFLOPS    99%
Softmax (4096)         850 GB/s      900 GB/s      94%
Reduction              820 GB/s      870 GB/s      94%
Element-wise           1800 GB/s     1900 GB/s     95%

关键优势:
  - GEMM 和 Attention 接近手写 CUDA 性能
  - tl.dot() 自动利用 Tensor Core
  - 开发效率提升 5-10x

劣势:
  - 灵活性不如 CUDA (复杂控制流受限)
  - 编译时间较长
  - 调试工具不如 CUDA 成熟
```

## 6. 在 AI Infra 中的应用

```
1. FlashAttention (tri Dao):
   flash-attn 使用 Triton 实现部分 kernel
   替代手写 CUDA，性能相当

2. vLLM:
   部分自定义 kernel 使用 Triton
   主要 kernel 仍用 FlashInfer (CUDA)
   未来可能更多 Triton

3. PyTorch 2.0 torch.compile:
   Inductor 后端使用 Triton 生成 GPU kernel
   自动将 PyTorch 操作编译为 Triton kernel
   用户无需手写 kernel

4. 自定义 kernel 开发:
   量化 kernel (FP8, INT4)
   融合 kernel (layernorm + residual + dropout)
   自定义 attention 变体
```

## 7. 开发环境搭建

```bash
# 安装 Triton
pip install triton

# 验证
python -c "import triton; print(triton.__version__)"

# 性能分析
python -m triton.testing.test_perf

# 调试工具
TRITON_INTERPRET=1 python kernel.py  # CPU 解释执行 (不需要 GPU)
TRITON_PRINT_AUTOTUNING=1 python kernel.py  # 打印调优过程
```

## 8. 学习要点

1. **tl.dot() 是关键** — 自动利用 Tensor Core，是 Triton 性能接近 CUDA 的核心
2. **Block tiling 是核心模式** — 所有高性能 kernel 都是分块处理数据
3. **Online softmax 技巧** — FlashAttention 的核心，避免 O(N²) 内存
4. **@triton.autotune 必用** — 不同问题规模需要不同配置，自动搜索最优
5. **TRITON_INTERPRET=1** — 不需要 GPU 也能开发和测试 Triton kernel
6. **与 PyTorch 无缝集成** — 数据通过指针传递，零拷贝

## 参考

- [Triton 官方文档](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Puzzles](https://github.com/srush/Triton-Puzzles) — 交互式学习
- [FlashAttention Triton 实现](https://github.com/Dao-AILab/flash-attention)
- [OpenAI Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)
