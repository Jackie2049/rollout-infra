# Triton Kernel Programming: AI Infra 工程师实战指南

> 从 CUDA 到 Triton — 理论模型、核心模式、vLLM 实战 kernel 分析

## 1. Triton vs CUDA: 编程模型对比

### 1.1 根本差异: Scalar vs Blocked

```
CUDA:  "Scalar Program, Blocked Threads"
  - 每个 thread 处理 1 个元素
  - 程序员管理 shared memory、同步、合并访问
  - threadIdx.x, blockIdx.x, blockDim.x

Triton: "Blocked Program, Scalar Threads"
  - 每个 program instance 处理一个 BLOCK 的数据
  - 编译器自动管理 shared memory、同步、合并访问
  - tl.program_id(), tl.arange(), tl.load()/tl.store()
```

### 1.2 编译器自动优化的 7 件事

Triton 编译器自动处理以下 CUDA 程序员需要手动优化的任务:

1. **Memory coalescing** — 自动合并相邻 thread 的内存访问
2. **Shared memory allocation** — 自动将 tl.load 的数据放入 SMEM
3. **Thread synchronization** — block 内同步自动插入
4. **Pre-fetching** — 软件 pipelining (num_stages 参数)
5. **Vectorization** — 自动向量化 load/store
6. **Tensor Core dispatch** — tl.dot() 自动映射到 mma 指令
7. **L2 cache optimization** — 通过 grouped ordering 优化

### 1.3 性能对比 (A100 实测)

```
操作                   Triton          cuBLAS/cuDNN    比值
──────────────────────────────────────────────────────────
GEMM (4096x4096)       218 TFLOPS      220 TFLOPS      99%
Fused Softmax           850 GB/s       900 GB/s        94%
Layer Norm              820 GB/s       870 GB/s        94%
Element-wise            1800 GB/s      1900 GB/s       95%
```

GEMM 和 Attention 接近手写 CUDA, 开发效率提升 5-10x.

### 1.4 已有笔记覆盖 vs 本笔记重点

已有 `triton-programming.md` 覆盖: 基本概念、编程模型、GEMM/FlashAttention/Softmax、autotune、开发环境.
**本笔记新增**: 内存层次深入分析、更多 kernel 模式、vLLM 真实 kernel 分析、性能调优技巧、局限性讨论.

## 2. 核心编程模型

### 2.1 Kernel 结构模板

```python
import triton
import triton.language as tl
import torch

@triton.jit
def my_kernel(
    x_ptr,          # 输入指针 (device memory)
    y_ptr,          # 输出指针
    N,              # 元素总数 (运行时参数)
    stride_x,       # stride (支持非连续 tensor)
    BLOCK_SIZE: tl.constexpr,  # 编译时常量 (block 大小)
):
    # Step 1: 确定 program 处理的数据范围
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 2: 创建 mask (处理边界)
    mask = offsets < N

    # Step 3: 加载数据 (global memory -> registers)
    x = tl.load(x_ptr + offsets * stride_x, mask=mask)

    # Step 4: 计算
    y = x * 2.0 + 1.0

    # Step 5: 存储结果 (registers -> global memory)
    tl.store(y_ptr + offsets, y, mask=mask)


def launch_kernel(x):
    y = torch.empty_like(x)
    N = x.numel()
    # Grid: 每个 program 处理 BLOCK_SIZE 个元素
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    my_kernel[grid](x, y, N, x.stride(0), BLOCK_SIZE=1024)
    return y
```

### 2.2 关键原语速查

```
原语                    作用                      对应 CUDA
─────────────────────────────────────────────────────────────
tl.program_id(axis)     获取 program 索引         blockIdx.x/y/z
tl.arange(start, end)   生成等差数列              threadIdx.x + offset
tl.constexpr            编译时常量                template<int>
tl.load(ptr, mask)      从 DRAM 加载到寄存器      global -> register
tl.store(ptr, val, mask) 从寄存器写入 DRAM        register -> global
tl.dot(a, b)            分块矩阵乘法 (Tensor Core) wmma::mma_sync
tl.sum(x, axis)         规约求和                  warp reduce
tl.max(x, axis)         规约求最大                warp reduce
tl.where(cond, a, b)    条件选择                  ?: operator
tl.zeros(shape, dtype)  创建零张量                __shared__ + memset
tl.full(shape, val)     创建常量张量              -
```

### 2.3 启动 Grid 的三种模式

```python
# 1D grid (最常见: element-wise, reduction)
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

# 2D grid (矩阵操作: GEMM, Attention)
grid = lambda meta: (
    triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
)

# 3D grid (batch + head + split: decode attention)
grid = (batch_size, num_heads, num_kv_splits)
```

## 3. 内存层次

### 3.1 Triton 的两级抽象

Triton 对开发者只暴露两级内存:

```
┌─────────────────────────────────────────────────┐
│  DRAM (Global Memory / HBM)                     │
│  - 容量大: 40-80 GB                              │
│  - 带宽有限: ~2 TB/s (A100)                      │
│  - 通过 tl.load() / tl.store() 访问              │
│  - 指针算术: base_ptr + offset * stride           │
├─────────────────────────────────────────────────┤
│  SRAM (On-chip: Registers + Shared Memory)       │
│  - 容量小: ~48 KB/SM (shared) + ~256 KB/SM (reg) │
│  - 带宽高: ~19 TB/s                               │
│  - Triton 自动分配, 开发者无需管理                 │
│  - tl.load 的数据自动缓存在 SRAM                   │
└─────────────────────────────────────────────────┘
```

### 3.2 指针算术: 多维 Tensor 访问

```python
# 1D: 简单偏移
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=offsets < N)

# 2D: stride 计算行和列
row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
# 广播: row_offs[:, None] 为列向量, col_offs[None, :] 为行向量
ptr = base_ptr + row_offs[:, None] * row_stride + col_offs[None, :] * col_stride
tile = tl.load(ptr, mask=(row_offs[:, None] < M) & (col_offs[None, :] < N))

# Paged KV Cache 间接寻址 (vLLM decode attention 模式)
# 先从 page table 获取物理页号, 再计算页内偏移
kv_page_number = tl.load(Req_to_tokens + batch_idx * stride + offs_n // PAGE_SIZE)
kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE  # 物理地址
k = tl.load(K_Buffer + kv_loc[:, None] * stride_kbs + head_idx * stride_kh
            + offs_d[None, :] * stride_kd)
```

### 3.3 Cache Modifier

Triton 支持 load 的 cache hint, 对性能敏感场景有用:

```python
# .ca = cache all levels (默认, 适合会被重复使用的数据)
q = tl.load(Q + offs, mask=mask, cache_modifier=".ca")

# .cg = cache global only (适合只读一次的大数据, 如 KV cache)
k = tl.load(K + offs, mask=mask, cache_modifier=".cg")
```

### 3.4 内存优化的关键: BLOCK_SIZE 选择

```
BLOCK_SIZE 选择原则:
  - 太小 (<64):   每个元素的分摊 launch 开销高, occupancy 低
  - 太大 (>4096):  寄存器溢出, SMEM 不够, 并行度不足
  - 甜蜜点:        128-1024 (element-wise), 64-128 (matmul tile)
  - 必须是 2 的幂:  triton.next_power_of_2(dim) 自动对齐
```

## 4. 核心 Kernel 模式

### 4.1 Reduction: Online Softmax

关键技巧 — 在线 softmax 避免 O(N^2) 内存:

```python
@triton.jit
def online_softmax_kernel(
    input_ptr, output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=-float('inf'))

    # 数值稳定的 softmax: 先减 max 再 exp
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    tl.store(output_ptr + row_start + offsets, output, mask=mask)
```

### 4.2 Blocked Matrix Multiply with L2 Cache Optimization

这是 Triton 性能的核心 — 分块 GEMM + grouped ordering:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  # L2 cache 优化参数
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # L2 cache 优化: grouped ordering
    # 同一组内的 program 共享 A 的 tile, 提高缓存命中率
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维 tiling
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am +
                    (offs_k[None, :] + k * BLOCK_K) * stride_ak)
        b = tl.load(b_ptr + (offs_k[:, None] + k * BLOCK_K) * stride_bk +
                    offs_n[None, :] * stride_bn)
        accumulator += tl.dot(a, b)  # 自动利用 Tensor Core!

    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)
```

### 4.3 FP8 即时反量化 (vLLM 模式)

vLLM 中大量使用 FP8 量化, 在 kernel 内即时反量化:

```python
@triton.jit
def per_token_group_quant_fp8(
    y_ptr, y_q_ptr, y_s_ptr,
    group_size, y_num_columns, y_row_stride, eps,
    fp8_min: tl.constexpr, fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g_id = tl.program_id(0)
    row = g_id // (y_num_columns // group_size)
    row_g_id = g_id % (y_num_columns // group_size)

    y_ptr += row * y_row_stride + row_g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale = _absmax * (1.0 / fp8_max)  # 用乘法代替除法, 匹配 PyTorch 精度
    y_q = tl.clamp(y / scale, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, scale)
```

反量化在 decode attention kernel 中:
```python
# vLLM triton_decode_attention.py 中的即时反量化
if k.dtype.is_fp8():
    k = (k.to(tl.float32) * ks).to(q.dtype)  # ks 是 scale, 加载一次复用
```

### 4.4 分块规约 + Split-K (Decode Attention 两阶段)

vLLM 的 decode attention 采用 Split-KV 模式, 将 KV 序列切成多个 split, 每个 program
处理一个 split, 最后在 stage2 合并:

```
Stage 1: (batch, head, NUM_KV_SPLITS) 个 program 并行
  每个 program 处理 seq_len / NUM_KV_SPLITS 个 KV token
  输出: partial_o[batch, head, split, dim] + log_sum_exp[batch, head, split]

Stage 2: (batch, head) 个 program
  合并所有 split 的 partial_o, 用 online softmax 合并 log_sum_exp
```

```python
# Stage 2 合并 (简化版)
@triton.jit
def reduce_stage2(Mid_O, o, NUM_KV_SPLITS: tl.constexpr, ...):
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    for split_kv_id in range(0, NUM_KV_SPLITS):
        tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os)
        tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
        n_e_max = tl.maximum(tlogic, e_max)

        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        acc += tl.exp(tlogic - n_e_max) * tv

        e_sum = e_sum * old_scale + tl.exp(tlogic - n_e_max)
        e_max = n_e_max

    result = acc / e_sum
    tl.store(o + offs, result)
```

### 4.5 Multi-LoRA Segmented Matmul (Punica 模式)

vLLM 的 LoRA serving 使用 Punica 风格的 Triton kernel. 核心挑战:
同一个 batch 中不同 token 属于不同 LoRA adapter, 需要分组处理.

```python
@triton.jit
def lora_shrink_kernel(
    input_ptr, lora_ptr, out_ptr,
    M, N, K,
    token_indices_sorted_by_lora_ids,  # 按 lora_id 排序的 token 索引
    num_tokens_per_lora,               # 每个 lora 有多少 token
    lora_token_start_loc,              # 每个 lora 的 token 起始位置
    lora_ids,                          # lora id 列表
    scaling,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ...
):
    lora_idx = tl.program_id(axis=2)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return  # 无 LoRA 的 token 直接跳过

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    # 获取该 lora 分组的 token 索引
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    ram = tl.load(token_indices_sorted_by_lora_ids + lora_m_indices_start + offset_m)

    # 用 ram 间接寻址 input, 不同 lora 用不同的权重矩阵
    # ... 执行分块 matmul ...
```

关键设计:
- 预排序: token 按 lora_id 排序, 避免分支发散
- 间接寻址: 通过 `ram` 数组映射到实际 token 位置
- Early exit: `if lora_id == -1: return` 无 LoRA 时跳过

## 5. vLLM 中的 Triton Kernel 实战分析

### 5.1 Kernel 分布 (30+ 文件)

vLLM 代码库中有 114 个文件使用 `@triton.jit`, 涵盖:

| 类别            | 代表文件                                       | 用途                     |
|-----------------|------------------------------------------------|--------------------------|
| Decode Attention | `v1/attention/ops/triton_decode_attention.py`  | 两阶段 Split-KV decode   |
| Prefill Attn    | `v1/attention/ops/triton_prefill_attention.py` | Prefill attention         |
| MoE Experts     | `layers/fused_moe/fused_moe.py`                | 分块 expert GEMM          |
| LoRA Serving    | `lora/ops/triton_ops/lora_shrink_op.py`         | Multi-LoRA segmented mm  |
| Quantization    | `layers/quantization/utils/fp8_utils.py`        | FP8 per-token-group quant|
| Sampling        | `v1/sample/ops/topk_topp_triton.py`             | Qrita top-k + top-p      |
| Activation      | `model_executor/layers/activation.py`           | SiLU/Mul, Gelu fusion    |

### 5.2 Decode Attention Kernel 深入

文件: `vllm/v1/attention/ops/triton_decode_attention.py` (~792 行)

**架构**: 三层设计

```
decode_attention_fwd()           # 入口: 根据 GQA/MHA 分发
  ├── decode_attention_fwd_normal()    # MHA 路径
  │     ├── _decode_att_m_fwd()       # Stage 1: 分 split 计算
  │     │     └── _fwd_kernel_stage1  # @triton.jit kernel
  │     └── _decode_softmax_reducev_fwd()  # Stage 2: 合并 splits
  │           └── _fwd_kernel_stage2  # @triton.jit kernel
  └── decode_attention_fwd_grouped()  # GQA/MLA 路径
        ├── _decode_grouped_att_m_fwd() # Stage 1 grouped heads
        │     └── _fwd_grouped_kernel_stage1
        └── _decode_softmax_reducev_fwd()  # Stage 2 复用
```

**关键设计决策**:

1. **Paged KV 间接寻址** — 支持 page_size >= 1:
   ```python
   kv_page_number = tl.load(Req_to_tokens + batch_idx * stride + offs_n // PAGE_SIZE)
   kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
   ```

2. **FP8 即时反量化** — 在 kernel 内完成, 避免额外 kernel launch:
   ```python
   if k.dtype.is_fp8():
       k = (k.to(tl.float32) * ks).to(q.dtype)
   ```

3. **GQA Grouped Processing** — 多个 Q head 共享 KV head, BLOCK_H=16:
   ```python
   cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
   qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_H, BLOCK_N] 矩阵乘
   ```

4. **MLA (DeepSeek-V2) 支持** — IS_MLA 模式下, V 不单独存储, 转置 K 作为 V:
   ```python
   if IS_MLA:
       v = tl.trans(k)  # MLA 用同一个 c_kv, transpose 即可
   ```

5. **ROCm 适配** — HIP 需要特殊参数:
   ```python
   if is_hip_:
       extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
       num_stages = 1  # ROCm 不支持多 stage pipelining
   ```

### 5.3 Qrita Top-K/Top-P Sampling Kernel

文件: `vllm/v1/sample/ops/topk_topp_triton.py` (~960 行)

基于论文 "Qrita: High-performance Top-k and Top-p Algorithm for GPUs using
Pivot-based Truncation and Selection".

**核心算法**: Ternary search + Pivot-based truncation

```
1. Zeroth pass: 计算统计量 (mean, std) 用于估计初始 pivot
2. First pass:  遍历 logits, 收集 > outlier_pivot 的值到 buffer
3. Second pass: 在 buffer 上做 ternary search 找到 top-k pivot
4. Third pass:  (如需 top-p) 计算概率, 收集到 buffer
5. Fourth pass: 在 buffer 上做 binary search 找到 top-p pivot
6. Final pass:  应用 mask, 居中 logit 设为 -inf
```

**关键 Triton 技巧**:

- **Persistent program**: `for row_id in tl.range(pid, BATCH_SIZE, num_programs)` —
  每个 program 处理多行, 减少启动开销
- **Compile-time conditional**: `TOPK_ENABLED: tl.constexpr`, `TOPP_ENABLED: tl.constexpr`
  — 编译器为不同组合生成不同 kernel, 避免运行时分支
- **Duplicate logit handling**: top-k 截断边界上可能有多个相同值的 token,
  需要精确计数保留多少个
- **统计表预计算**: `_NORMAL_CDF_TO_SIGMA_TABLE` 200 个元素, 用查表代替 erf 计算

### 5.4 MoE Fused Expert GEMM

文件: `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`

**双 GEMM 流水线**: MoE 需要 w1 (gate+up projection) 和 w2 (down projection):
```
hidden_states --[w1]--> intermediate --[SiLU+Mul]--> activated --[w2]--> output
```

**关键优化**:

1. **Block alignment + padding**: token 按 expert 分组对齐到 BLOCK_SIZE_M 的整数倍
2. **FP8 SiLU+Mul+Quant fusion**: 将激活函数和 FP8 量化融合为一个 kernel:
   ```python
   qintermediate_cache2, a2q_scale = ops.silu_and_mul_per_block_quant(
       intermediate_cache1.view(-1, N), group_size=128, ...)
   ```
3. **LoRA 双流并行**: base GEMM 和 LoRA delta 分别在两个 CUDA stream 执行:
   ```python
   maybe_execute_in_parallel(_base_w13_fn, _lora_w13_fn, ...)
   intermediate_cache1.add_(lora_delta_w13)  # 合并结果
   ```

## 6. 性能调优与调试

### 6.1 @triton.autotune

Autotune 是 Triton 最重要的性能工具. 在不同 shape 下, 最优 block size/warp/stage 不同:

```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 64}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 64}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_stages=4, num_warps=4),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],       # 根据这些参数分组调优
    reset_to_zero=['c_ptr'],    # 每次 benchmark 前清零的输出
)
@triton.jit
def tuned_matmul_kernel(...):
    ...
```

**调优参数含义**:

| 参数         | 含义                          | 推荐范围      |
|--------------|-------------------------------|---------------|
| num_warps    | 每 block 的 warp 数           | 2, 4, 8       |
| num_stages   | 软件 pipelining 级数          | 2-5 (A100)    |
| BLOCK_M/N/K  | Tile 大小                     | 32-256        |
| GROUP_SIZE_M | L2 cache 分组大小             | 1-8           |

**注意**: num_stages 越大占用 SMEM 越多, 可能反而降低 occupancy. 大 BLOCK_DMODEL
时需要降低 num_stages 避免溢出 (vLLM 实例: BLOCK_DMODEL=1024 + BLOCK_H=16 需要
num_stages=1).

### 6.2 num_stages (Software Pipelining)

```
num_stages=1: 无 pipelining, load 和 compute 串行
num_stages=2: load 下一块数据和计算当前块重叠
num_stages=3+: 多级流水线, 更好的延迟隐藏, 但 SMEM 用量线性增长

SMEM 用量估算:
  per_stage = BLOCK_M * BLOCK_K * sizeof(dtype) * 2 (A tile + B tile)
  total = per_stage * num_stages * num_warps
  限制: ~101 KB/SM (A100), 超过则自动降 occupancy
```

vLLM 中的选择:
- Decode attention: `num_stages=2` (默认)
- ROCm: `num_stages=1` (不支持多 stage)
- 大 BLOCK_DMODEL: `num_stages=1` (避免 SMEM 溢出)

### 6.3 tl.range vs range

```python
# range: 编译器展开为静态循环, 循环次数必须是编译时常量
for i in range(0, NUM_TILES):  # NUM_TILES: tl.constexpr
    ...

# tl.range: 支持动态循环次数, 可以利用 software pipelining
for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
    ...  # 编译器自动插入 pre-fetch 指令
```

在 vLLM decode attention 中, grouped kernel 使用 `tl.range` 来支持动态
序列长度, 而普通 kernel 使用 `range`.

### 6.4 调试技巧

```bash
# 1. CPU 解释执行 — 不需要 GPU 也能开发和测试
TRITON_INTERPRET=1 python kernel.py

# 2. 打印 autotune 结果
TRITON_PRINT_AUTOTUNING=1 python kernel.py

# 3. 查看 PTX/LLVM IR (调试编译问题)
triton.compile(kernel_fn, ...)  # 返回 .asm 字段

# 4. NaN/Inf 调试 — 在 kernel 中打印
# Triton 不直接支持 print, 但可以用 tl.store 写到输出 tensor 后检查

# 5. 性能分析
torch.cuda.profiler.initialize(...)
# 然后用 ncu / nsight systems profiling

# 6. 精度验证 — 始终和 PyTorch reference 比较
ref = torch.softmax(x, dim=-1)
triton_out = triton_softmax(x)
torch.testing.assert_close(triton_out, ref, atol=1e-5, rtol=1e-5)
```

### 6.5 常见性能陷阱

1. **BLOCK_SIZE 不是 2 的幂**: 编译器无法优化, 性能可能下降 30%+
   ```python
   BLOCK_SIZE = triton.next_power_of_2(N)  # 始终对齐
   ```

2. **stride 传递错误**: 必须传 PyTorch tensor 的实际 stride, 不能假设 contiguous
   ```python
   # 正确: 传递实际 stride
   kernel[grid](x, y, N, x.stride(0), x.stride(1))
   # 错误: 假设行优先连续
   kernel[grid](x, y, N, N, 1)  # 如果 x 是转置的就错了
   ```

3. **Mask 缺失**: 越界访问导致 UB 或性能下降
   ```python
   mask = offsets < N
   x = tl.load(ptr + offsets, mask=mask, other=0.0)  # other 提供默认值
   ```

4. **Atomic 操作过度**: `tl.atomic_add` 比普通 store 慢 5-10x,
   仅在多 program 写同一位置时使用

5. **编译缓存失效**: Triton 缓存编译结果, 改 BLOCK_SIZE 会重新编译.
   对于 production, 预热所有 shape 避免首次延迟.

## 7. Triton 的局限性

### 7.1 当前不擅长的场景

```
1. 复杂控制流
   - 只支持 for/while + if, 不支持递归
   - 不支持 dynamic parallelism (kernel 内启动 kernel)
   - 不支持 function pointer / virtual dispatch

2. 跨 block 同步
   - 只能通过 atomic 或 kernel 分多次启动实现
   - 无法做 global barrier (CUDA 的 cooperative groups 可以)

3. 自定义 SMEM 布局
   - 无法控制 shared memory 的 bank conflict 优化
   - 某些极端场景下可能不如手写 CUDA

4. 硬件特定指令
   - 无法直接使用 cp.async, TMA, WGMMA 等 H100 专有指令
   - 编译器可能在某些场景下无法自动利用最新硬件特性

5. 调试体验
   - 没有 GDB/cuda-gdb 级别的调试器
   - 错误信息有时不够明确
   - TRITON_INTERPRET 模式速度很慢
```

### 7.2 何时选择 CUDA 而非 Triton

```
选 Triton:
  - 标准分块操作 (GEMM, Attention, LayerNorm, Softmax)
  - Element-wise + Reduction 的融合
  - 快速原型验证
  - PyTorch torch.compile 后端

选 CUDA:
  - 需要精细控制 SMEM 布局和 bank conflict
  - 需要跨 block 同步 (如 AllReduce kernel)
  - 需要最新硬件特性 (TMA, WGMMA, Cluster)
  - 需要极致性能的最后 5%
  - CUDA 11.x 兼容性需求 (Triton 3.x 需要 CUDA 12+)
```

### 7.3 版本兼容性注意事项

```
Triton 2.x: CUDA 11.x 兼容, 但性能较差, 功能不全
Triton 3.x: 需要 CUDA 12+, 性能接近手写 CUDA, API 稳定
Triton 3.2+: 修复 "operation scheduled before its operands" 警告

vLLM 的做法:
  from vllm.triton_utils import tl, triton  # 统一 import, 处理版本差异
```

## 8. 实战学习路径

### 8.1 推荐顺序

```
1. 官方教程 (1-2 天)
   - Vector Add → Fused Softmax → Matrix Multiply → Layer Norm
   - https://triton-lang.org/main/getting-started/tutorials/

2. Triton Puzzles (1 天)
   - 交互式练习, 巩固编程模型
   - https://github.com/srush/Triton-Puzzles

3. 阅读 vLLM kernel (2-3 天)
   - triton_decode_attention.py (Split-KV, Paged KV, FP8)
   - fp8_utils.py (per-token-group quant)
   - topk_topp_triton.py (persistent program, ternary search)

4. 修改/扩展 vLLM kernel (1-2 天)
   - 添加新的 mask 类型
   - 修改 quantization 方案
   - 用 autotune 优化 block size

5. 写自定义 kernel (持续)
   - 从简单 fusion 开始 (bias + activation)
   - 逐步增加复杂度
```

### 8.2 性能 Checklist

写完 Triton kernel 后检查:

- [ ] BLOCK_SIZE 是 2 的幂
- [ ] 有正确的 mask 处理边界
- [ ] 传递了正确的 stride (不是硬编码)
- [ ] 使用 autotune 搜索最优配置
- [ ] 用 tl.dot() 而非手动乘加 (Tensor Core)
- [ ] 大矩阵使用 GROUP_SIZE_M 优化 L2 cache
- [ ] 精度验证通过 (和 PyTorch ref 对比)
- [ ] 检查 SMEM 占用是否超限 (大 BLOCK_SIZE + 多 stage)
- [ ] 避免不必要的 atomic 操作
- [ ] 预热编译缓存, 避免生产环境首次延迟

## 参考

- [Triton 官方文档](https://triton-lang.org/)
- [Triton Language API Reference](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Programming Guide](https://triton-lang.org/main/programming-guide/index.html)
- [Triton Tutorials (10 个)](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton Puzzles](https://github.com/srush/Triton-Puzzles)
- [vLLM Triton Decode Attention](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py)
- [Qrita Top-K/Top-P Paper](https://arxiv.org/abs/2602.01518)
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
