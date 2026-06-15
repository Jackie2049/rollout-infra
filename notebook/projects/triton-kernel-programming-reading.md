# OpenAI Triton — GPU Kernel 编程模型与编译架构深度阅读

> triton-lang/triton v3.7.0 (2026-05-07) | 源码 + Release Notes + Tutorials
> 与我们 vLLM Triton kernel (W4A16 fallback/flash attention/softmax) + PyTorch Inductor codegen + RTX 4090 SM89 直接相关

## 1. 概述

Triton 是 OpenAI 开发的 GPU kernel 编程语言和编译器, 目标是让 Python 程序员写出性能媲美手写 CUDA 的 GPU kernel。

**核心设计哲学**:
- ★ **Block-level 编程**: 不需要管理线程, 只需描述 "block 级" 的计算和内存操作
- ★ **自动优化**: 编译器自动决定 shared memory 管理、线程分配、内存合并
- ★ **Python 前端**: `@triton.jit` 装饰器将 Python AST 编译为 GPU PTX/AMDGCN

**编译流水线**: `Python (@triton.jit) → AST → Triton IR (TTIR) → TritonGPU IR (TTGIR) → PTX/AMDGCN → GPU Execution`

**版本演进**: v3.4→v3.5→v3.6→v3.7.0, 逐步引入 Gluon 框架、2CTA/TMA、warp specialization、dot_scaled(MXFP)、RDNA4 support

## 2. Triton 编程模型: @triton.jit 与 Block-Level

### 2.1 @triton.jit 装饰器工作机制

`@triton.jit` 是 Triton 的核心装饰器, 位于 `python/triton/runtime/jit.py` 的 `JITFunction` 类。

**关键流程**:
1. **AST 解析**: `DependenciesFinder` (ast.NodeVisitor) 遍历 Python AST, 收集依赖和全局变量引用
2. **Hash 计算**: SHA256 hash 基于: 函数源码 + 依赖函数的 cache_key + 全局变量值
3. **Specialization**: 运行时根据参数类型和 constexpr 值生成特化版本 (specialization)
4. **Cache**: 每次调用先查 cache (hash + specialization + options + env_vars), 未命中则编译

```python
# JITFunction.run() 核心逻辑 (python/triton/runtime/jit.py:726)
class JITFunction(JITCallable, KernelInterface[T]):
    def run(self, *args, grid, warmup, **kwargs):
        # 1. 从 args 推断 signature (类型 specialization)
        # 2. 计算 cache_key = hash + specialization + options + env_vars
        # 3. 查 cache → 未命中 → compile()
        # 4. compile() → ASTSource.make_ir() → ast_to_ttir() → TTIR → TTGIR → PTX → cubin
        # 5. 返回 CompiledKernel, 调用 kernel.launch()
```

**constexpr**: `tl.constexpr` 标注的参数在编译时必须确定, 作为 specialization 的 key。block size (BLOCK_M, BLOCK_N 等) 必须是 constexpr。

**与 CUDA 的关键区别**:

| CUDA | Triton |
|---|---|
| 线程级: threadIdx.x/y/z | Block级: tl.program_id(0/1/2) |
| 手动 shared memory (`__shared__`) | 自动 shared memory 管理 |
| 手动内存合并 (coalescing) | 编译器自动优化 |
| 手动 warp 级同步 | 编译器自动插入 barrier |
| 手动 tensor core调度 (mma.sync/wgmma) | tl.dot → 编译器自动 lowering |
| PTX/CUDA C 编写 | Python 编写, JIT 编译 |

### 2.2 Block-Level 编程模型核心概念

**Program**: Triton kernel 的执行单元。每个 "program" 对应 CUDA 的一个 CTA (Cooperative Thread Array / Thread Block)。

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)        # 当前 program 的 ID (类似 blockIdx.x)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)   # block 内的偏移量
    mask = offsets < n_elements           # 边界检查

    x = tl.load(x_ptr + block_start + offsets, mask=mask)  # 加载 block
    y = tl.load(y_ptr + block_start + offsets, mask=mask)
    out = x + y                           # block 级计算 (自动映射到线程)
    tl.store(out_ptr + block_start + offsets, out, mask=mask)  # 存储 block

grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # grid = program 数量
vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
```

**关键**: 用户只需要指定 "每个 block 做什么", 编译器负责:
- 将 block 操作分配到 warp → thread
- 自动内存合并 (确保连续线程访问连续地址)
- 自动 shared memory 分配 (tl.dot 会自动使用 shared memory)
- 自动插入同步 barrier

### 2.3 tl.program_id 与 Grid

```python
pid = tl.program_id(axis=0)       # blockIdx.x
pid_m = pid // grid_n              # 2D grid 的行 ID
pid_n = pid % grid_n               # 2D grid 的列 ID
num_programs = tl.num_programs(0)  # gridDim.x
```

Grid 是 1D/2D/3D 的 program 数量, 在 kernel launch 时通过 `kernel[grid](...)` 指定。

## 3. Triton Autotuning: Config vs Dynamic vs Cache

### 3.1 @triton.autotune 机制

位于 `python/triton/runtime/autotuner.py`, `Autotuner` 类继承 `KernelInterface`。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        # ... 更多 config
    ],
    key=['M', 'N', 'K'],  # cache key 依赖的参数名
)
@triton.jit
def matmul_kernel(...)  ...
```

**Config 参数**:
- `kwargs`: dict of constexpr 值 (BLOCK_M, BLOCK_N 等)
- `num_warps`: 每个 CTA 的 warp 数量 (4/8/16/32)
- `num_stages`: software pipelining 的 stage 数 (影响 shared memory 使用量)
- `num_ctas`: CTA cluster 大小 (SM90+ Hopper, 2CTA mode)

### 3.2 Autotune 工作流程

```
run(args, kwargs)
  → 1. 计算 key = tuple(kwargs[k] for k in self.keys)
  → 2. 查 self.cache[key] → 命中 → 直接用 cached config
  → 3. 未命中 → check_disk_cache() (SHA256 hash, JSON 文件)
  → 4. disk cache 也未命中 → prune_configs(kwargs)
     → 4a. early_config_prune (用户定义的剪枝函数)
     → 4b. perf_model (性能模型预测, top_k 保留)
  → 5. benchmark 每个 pruned config
     → _bench(*args, config=config, **meta)
     → do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
  → 6. 选最快 config → 存入 self.cache[key]
  → 7. 如果 cache_results=True → 存到磁盘 cache
  → 8. 用最优 config 调用 self.fn.run()
```

### 3.3 Config vs Dynamic Block Size 对比

| 方面 | @triton.autotune (Config) | Dynamic Block Size |
|---|---|---|
| **优化方式** | 经验 benchmark, 选最快 | 启发式/逻辑计算 |
| **缓存** | cache per key (内存+磁盘) | 无需缓存 |
| **启动开销** | 首次 benchmark 开销大 | 几乎零开销 |
| **灵活性** | 仅限于预定义 config 组合 | 适应任意输入 shape |
| **适用场景** | 固定/已知输入 shape (GEMM, attention) | 变化/unpredictable shape |
| **Cache miss** | 重新 benchmark 所有 config | 无 cache miss 问题 |

**★ vLLM 实践**: FlashAttention Triton kernel 用 `@triton.autotune` + 多个 Config (BLOCK_M=64/128 等), key=['M', 'N', 'K'], 因为推理 shape 相对固定。W4A16 Triton fallback 也用 autotune。

### 3.4 Cache 机制详解

**两级缓存**:
1. **内存缓存** (`self.cache: Dict[Tuple, Config]`): key = (M, N, K) tuple → 最优 Config
2. **磁盘缓存** (`check_disk_cache`): SHA256 hash → JSON 文件, 跨 session 持久化

**磁盘 cache key 组成**:
```
triton_key() + backend.hash() + fn.cache_key + env_vars + tuning_key + configs_str
→ SHA256 → hexdigest → 查 cache_manager
```

**★ 限制**: disk cache 无法序列化 pre_hook, 所以有 pre_hook 的 config 必须重新 benchmark。

### 3.5 Config 剪枝策略

```python
prune_configs_by={
    'perf_model': lambda configs, named_args, **kwargs: ...,  # 性能模型预测运行时间
    'top_k': 10,            # 只 benchmark 前 K 个最优预测 config
    'early_config_prune': lambda configs, named_args, **kwargs: ...,  # 早期剪枝
}
```

**vLLM 实际使用的剪枝**: 基于 `BLOCK_M * BLOCK_N` 与输入 shape 的关系, 排除明显不适合的 config。

## 4. Triton JIT 编译: Python → GPU Code 的完整流水线

### 4.1 编译流水线总览

```
Python 函数 (@triton.jit)
  ↓ DependenciesFinder (AST visitor, 收集依赖+全局变量)
  ↓ JITFunction.run() → compute_cache_key → 查 cache
  ↓ 未命中 → compile()
  ↓
ASTSource → make_ir()
  ↓ ast_to_ttir() (code_generator.py, Python AST → Triton IR)
  ↓
TTIR (Triton IR, MLIR dialect)
  ↓ backend.add_stages() 注册编译阶段
  ↓ TTIR → TTGIR (TritonGPU IR)
  ↓   - Layout inference (自动 shared memory 管理)
  ↓   - Warp-level operation lowering
  ↓   - Memory coalescing optimization
  ↓   - Software pipelining
  ↓ TTGIR → LLVM IR
  ↓ LLVM IR → PTX (NVIDIA) / AMDGCN (AMD)
  ↓ PTX → SASS (CUDA driver JIT) / HSACO (AMD)
  ↓
CompiledKernel (cubin/hsaco + metadata + launch config)
```

### 4.2 AST → TTIR: code_generator.py

`ast_to_ttir()` 在 `python/triton/compiler/code_generator.py`, 核心是 AST visitor:

- 遍历 Python AST, 将每个操作映射到 Triton IR 操作
- `tl.load` → `tt.load` IR op
- `tl.store` → `tt.store` IR op
- `tl.dot` → `tt.dot` IR op
- `tl.arange` → `tt.make_range` IR op
- 运算符 (+ - * /) → `tt.add/sub/mul/div` IR op
- 条件/循环 → `scf.if/for` IR op

**关键**: `_semantic` 参数在 @triton.jit 内自动注入, 控制所有 builtin 的语义实现。

### 4.3 TTIR → TTGIR: GPU-Specific Lowering

TTGIR 引入 GPU-specific 概念:
- **Memory layout**: `SharedEncodingAttr` (shared memory 布局, 自动 bank conflict 避免和 swizzling)
- **Thread grouping**: `CTALayoutAttr` (CTA/Cluster 分布)
- **Data movement**: `cp_async`, `cp_async_bulk_tensor` (TMA), `wgmma` (warp-group MMA)
- **Layout inference**: 自动推断每个 tensor 的分布式布局 (register/shared/global memory)

**关键优化 pass**:
- **Pipelining**: 将 global memory load 和 compute 重叠, 减少 memory stall
- **Layout propagation**: 从 consumer 向 producer 传播最优布局, 减少 layout convert
- **Warp specialization** (SM90+): 分割 warp 到 producer (load) 和 consumer (compute) 两组

### 4.4 TTGIR → PTX/AMDGCN: Backend Codegen

NVIDIA backend (`lib/TritonNVIDIAGPU/`):
- `wgmma` → `wgmma.mma_async` PTX (SM90+)
- `mma.sync` → `mma.sync.aligned.m8n8k4` PTX (SM80+)
- `cp_async` → `cp.async.cg.shared.global` PTX (SM80+)
- `cp_async_bulk_tensor` → `cp.async.bulk.tensor` PTX (SM90+ TMA)
- `bar.sync` → `bar.sync` PTX (shared memory barrier)

AMD backend (`lib/TritonAMDGPU/`):
- MFMA → `mfma` instruction (CDNA3+)
- WMMA → `wmma` instruction (RDNA3/4+)
- TDM → AMD Tensor Data Movement (RDNA4+)

### 4.5 编译缓存

`compile()` 函数在 `python/triton/compiler/compiler.py:226`:

```python
def compile(src, target=None, options=None):
    backend = make_backend(target)
    key = get_cache_key(src, backend, options, env_vars)
    hash = hashlib.sha256(key.encode()).hexdigest()
    fn_cache_manager = get_cache_manager(hash)

    # Cache hit → 直接返回 CompiledKernel
    if metadata_path is not None:
        return CompiledKernel(src, metadata_group, hash)

    # Cache miss → 编译流水线
    stages = dict()
    backend.add_stages(stages, options, src.language)
    for ext, compile_ir in stages.items():
        module = compile_ir(module, metadata)
    # 写回 metadata 到 cache
```

**Cache key 组成**: `src.hash() + backend.hash() + options + env_vars + triton_version`

**★ AsyncCompile**: Triton 支持 `AsyncCompileMode` 多线程并行编译多个 kernel (v3.4+), 用于减少冷启动编译延迟。

## 5. Triton 内存模型: 指针、tl.load/tl.store、Masking、Tensor Descriptor

### 5.1 Triton 指针模型

Triton 的指针是 **typed pointer** (`triton.PointerType`), 携带数据类型信息:

```python
# 基础指针: 单元素指针
x_ptr: tl.pointer_type  # 从 Python 参数自动推断

# Block of pointers: N 维指针 tensor
# 通过 offset 计算得到
offsets = tl.arange(0, BLOCK_SIZE)            # [0, 1, ..., BLOCK_SIZE-1]
ptrs = x_ptr + offsets                        # pointer + offset → block of pointers

# 2D block of pointers
offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
```

**★ 关键**: Triton 指针是 element-level 的, 不是 byte-level 的。指针运算自动按 dtype 大小偏移。

### 5.2 tl.load / tl.store

位于 `python/triton/language/core.py:2477/2558`, 三种模式:

**(1) 单元素指针 → scalar load/store**:
```python
val = tl.load(ptr)          # scalar
tl.store(ptr, val)          # scalar
```

**(2) Block of pointers → tensor load/store**:
```python
mask = offsets < n_elements
row = tl.load(ptrs, mask=mask, other=-float('inf'))   # 加载 block, mask 外填充 -inf
tl.store(out_ptrs, result, mask=mask)                  # 存储 block
```

**(3) Block pointer / Tensor Descriptor → tensor load/store**:
```python
# make_block_ptr (★ 已 deprecated, v3.7)
block_ptr = tl.make_block_ptr(base=ptr, shape=[M, N], strides=[N, 1],
                              offsets=[0, 0], block_shape=[BM, BN], order=[1, 0])
data = tl.load(block_ptr, boundary_check=[0, 1])
tl.store(block_ptr, result, boundary_check=[0, 1])

# ★ ★ make_tensor_descriptor (推荐, v3.7+)
desc = tl.make_tensor_descriptor(ptr, shape=[M, N], strides=[N, 1],
                                  block_shape=[BM, BN])
data = desc.load([m_offset, n_offset])
desc.store([m_offset, n_offset], result)
```

**tl.load 参数详解**:
- `mask`: bool tensor, False 位置不加载 (避免 OOB 和无用读取)
- `other`: mask=False 位置的填充值 (默认 undefined, 常用 -inf/0.0)
- `boundary_check`: tuple of dim indices, 哪些维度做边界检查 (仅 block ptr/TD)
- `padding_option`: "zero"/"nan" (仅 block ptr/TD, OOB 填充值)
- `cache_modifier`: ".ca"(cache all)/".cg"(cache global, 不 cache L1)/".cv"(不 cache)
- `eviction_policy`: "evict_first"/"evict_last"
- `volatile`: volatile load

### 5.3 Masking 的重要性

**★ ★ Masking 是 Triton 编程的核心技巧**, 因为 BLOCK_SIZE 通常是 2 的幂, 但实际数据维度不一定:

```python
# 正确的 masking 模式
BLOCK_SIZE: tl.constexpr = 1024
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_cols           # 防止 OOB
data = tl.load(ptr + offsets, mask=mask, other=0.0)  # OOB 位置填充 0
```

**为什么 mask 关键**:
- Triton BLOCK_SIZE 必须是 2 的幂 (硬件对齐要求)
- 没有 mask → OOB 读取会读垃圾数据, 写入会破坏内存
- mask=False 的位置, load 返回 `other` 值, store 不写入 → 安全

### 5.4 make_block_ptr → deprecated → make_tensor_descriptor

**★ v3.7.0 Breaking Change**: `tl.make_block_ptr` 正式 deprecated (#9667), 用户应迁移到 `tl.make_tensor_descriptor`。

**原因**: `make_block_ptr` 有多个限制:
- 软件模拟的 block pointer, 不是硬件 descriptor
- 不支持 TMA (Tensor Memory Accelerator)
- 不支持 2CTA cooperative mode
- boundary handling 有 quirks

**make_tensor_descriptor 的优势**:
- **SM90+**: 自动映射到 TMA descriptor → 硬件级异步 bulk transfer
- **SM89**: 自动 fallback 到软件模拟 (不使用 TMA)
- 支持 `padding_option` ("zero"/"nan")
- 支持 `round_f32_to_tf32=True` (#9295)
- 支持 2-5 维 tensor

```python
# make_tensor_descriptor 示例 (来自 core.py docstring)
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
```

**★ TMA descriptor 需要 allocator**: 用户必须通过 `triton.set_allocator(alloc_fn)` 提供 GPU 内存分配函数, 因为 TMA descriptor 需要 global memory allocation。

## 6. Triton Math Ops: tl.dot / tl.reduce / tl.where / tl.arange / tl.full

### 6.1 tl.dot (Matmul)

位于 `core.py:2353`, Triton 的核心 matmul primitive:

```python
@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None,
        max_num_imprecise_acc=None, out_dtype=None, _semantic=None):
```

**参数详解**:
- `input`, `other`: 2D 或 3D tensor (支持 batched matmul)
- `acc`: 累加器, 如果提供, 结果加到 acc 上 (用于循环累加)
- `input_precision`: `"tf32"`/`"tf32x3"`/`"ieee"` — 控制 f32 matmul 精度
  - **tf32**: 19-bit (默认, 用 tensor core)
  - **tf32x3**: 3次 tf32 累加 (更高精度)
  - **ieee**: 标准 float32 (不用 tensor core, SIMT FMA, 慢但精确)
- `allow_tf32`: deprecated, 等价于 input_precision="tf32"
- `out_dtype`: 输出 dtype (默认与输入一致)
- `max_num_imprecise_acc`: 允许不精确累加的最大次数

**★ SM89 lowering**: RTX 4090 上 tl.dot → `mma.sync` (per-warp tensor core op), 不支持 `wgmma` (per-warp-group, SM90+)。

**★ dot 精度行为**:
- BF16 × BF16 → 直接 tensor core mma.sync
- FP16 × FP16 → 直接 tensor core mma.sync
- FP32 × FP32 → TF32 truncation → tensor core (默认), 或 ieee → SIMT FMA
- INT8 × INT8 → INT8 tensor core mma.sync → INT32 accumulator

### 6.2 tl.dot_scaled (MXFP/Block-Scaled Matmul)

v3.7 新增完整支持 (`core.py:2425`):

```python
@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
               acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True,
               out_dtype=float32, _semantic=None):
```

- `lhs_format`: `"e2m1"`(FP4)/`"e4m3"`(FP8)/`"e5m2"`(FP8)/`"bf16"`/`"fp16"`
- `rhs_format`: 同上
- `lhs_scale`, `rhs_scale`: MX scaling factor (e8m0 类型, uint8 表示)
- **★ SM89**: 软件模拟 → 先 upcast 到 BF16 再做普通 tl.dot
- **SM90+**: 硬件 native `wgmma.mma_async` with MXFP scaling
- **SM120+**: native FP4 scaled dot

### 6.3 tl.arange / tl.full

```python
# tl.arange(start, end) — 生成连续整数序列
offsets = tl.arange(0, BLOCK_SIZE)     # [0, 1, ..., BLOCK_SIZE-1]
# ★ start-end 必须是 2 的幂, 且差值 <= TRITON_MAX_TENSOR_NUMEL

# tl.full(shape, value, dtype) — 生成填充 tensor
acc = tl.full([BLOCK_M, BLOCK_N], 0.0, tl.float32)  # 初始化累加器
```

### 6.4 tl.where

```python
@builtin
def where(condition, x, y, _semantic=None):
```

**★ 重要**: `tl.where` 两个分支都会被计算 (不短路), 如果要避免不必要内存操作, 用 `tl.load/tl.store` 的 `mask` 参数。

**常见用法**:
```python
mask = col_offsets < n_cols
row = tl.where(mask, actual_value, 0.0)       # mask=True 用 actual, False 用 0
```

### 6.5 tl.reduce

```python
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
```

- `axis`: reduce 的维度
- `combine_fn`: 自定义 reduce 函数 (必是 @triton.jit)
- `keep_dims`: 是否保留 reduced 维度

**常用 reduce shortcut** (不直接用 tl.reduce, 用 member function):
```python
row_max = x.max(axis=1)     # 逐行 max
row_sum = x.sum(axis=1)     # 逐行 sum
m_i = tl.max(qk, axis=1)   # flash attention 用
```

### 6.6 Softmax (手动实现, 无内置 tl.softmax)

**★ Triton 没有 `tl.softmax` builtin!** Flash attention 用 **online/incremental softmax**:

```python
# Flash attention 核心: online softmax
# 不是一步 softmax, 而是逐 block 更新
qk = tl.dot(q, k)                         # QK^T
m_ij = tl.max(qk, axis=1)                 # 当前 block max
m_i_new = tl.maximum(m_i, m_ij)           # 更新全局 max
p = tl.exp(qk - m_i_new)                  # 当前 block exp
l_ij = tl.sum(p, axis=1)                  # 当前 block sum
alpha = tl.exp(m_i - m_i_new)             # 前面 block 的修正系数
l_i_new = alpha * l_i + l_ij              # 更新全局 sum
acc = acc * alpha[:, None]                 # 修正前面的累加
acc += tl.dot(p.to(v.dtype), v)           # 累加 PV
```

**为什么不用一次 softmax**: flash attention 必须逐 block 处理 (避免 O(N^2) 内存), online softmax 是唯一可行的方案。

## 7. Triton 3.7.0 重大变更详解

### 7.1 make_block_ptr → deprecated (#9667)

**PR #9667** (merged 2026-03-06): 发出 deprecation warning。

**迁移路径**: `tl.make_block_ptr` → `tl.make_tensor_descriptor`

**差异对比**:

```python
# 旧: make_block_ptr (deprecated)
bp = tl.make_block_ptr(base=ptr, shape=[M,N], strides=[N,1],
                        offsets=[0,0], block_shape=[BM,BN], order=[1,0])
data = tl.load(bp, boundary_check=[0,1])
bp = tl.advance(bp, [1,0])   # 移动 pointer
data = tl.load(bp, boundary_check=[0])

# 新: make_tensor_descriptor (推荐)
desc = tl.make_tensor_descriptor(ptr, shape=[M,N], strides=[N,1],
                                  block_shape=[BM,BN])
data = desc.load([offset_m, offset_n])
data = desc.load([offset_m+BM, offset_n])  # 直接用新 offset
```

**★ ★ RTX 4090**: `make_tensor_descriptor` 在 SM89 上不使用 TMA, 软件模拟, 行为与 make_block_ptr 等价。迁移是安全的。

### 7.2 2CTA Mode (SM90+ Hopper)

**2CTA**: 2 个 CTA 组成 cluster, 通过 Distributed Shared Memory (DSM) 共享数据。

**Triton 3.7 2CTA 支持** (#8684, #8874, #8922):
- Gluon multi-cta + 2CTA end-to-end 支持
- M=64 2CTA mode
- 消除不必要的 2CTA MMA 同步
- 正确的 TMEM deallocation timing

**2CTA matmul 工作原理**:
- 2 个 CTA 共同计算同一个 output tile
- 每个 CTA 加载 K 维的不同 slice, 通过 DSM 共享
- 减少 ~50% 的 global memory 读取
- 需要 `cluster_dim` 配置 (Triton Config 的 `num_ctas=2`)

**★ ★ RTX 4090**: 2CTA 需要 SM90 (Hopper) → **不支持!** SM89 没有 cluster hardware, 没有 DSM。

### 7.3 TMA (Tensor Memory Accelerator) 支持

**TMA**: SM90+ 的硬件级 bulk tensor transfer, 从 global memory 到 shared memory, 不需要 warp 参与。

**Triton 3.7 TMA 新增**:
- TMA im2col mode (图像卷积用) (#9202)
- TMA + multicast (cluster 内多 CTA 同时接收同一数据) (#9005)
- TMA descriptor mitigation (防止 descriptor 创建错误) (#9235)
- `tcgen05.mma` + multicast (#9071)

**★ ★ RTX 4090**: TMA 需要 SM90 → **不支持!** SM89 没有 TMA hardware unit。

**SM89 vs SM90 内存对比**:

| Feature | SM89 (Ada/RTX 4090) | SM90 (Hopper/H100) |
|---|---|---|
| **TMA** | ✗ (软件模拟 cp.async) | ✓ (硬件 cp.async.bulk.tensor) |
| **WGMMA** | ✗ (mma.sync per-warp) | ✓ (wgmma per-warp-group) |
| **2CTA Cluster** | ✗ | ✓ (DSM 共享) |
| **FP8 Tensor Core** | ✓ (但 E5M2 训练有限制) | ✓ |
| **Tensor Descriptor HW** | ✗ (软件模拟) | ✓ (TMA descriptor) |

### 7.4 Triton Kernels Matmul Refactor (#8765)

**★ Breaking Change**: `triton_kernels.matmul_*` API 变化 (#8765, "matrix multiplication refactor")。

**triton_kernels** 是 Triton 内置的高性能 matmul kernel 库 (不是 `tl.dot`, 而是 Python-level 封装), 提供:
- `triton_kernels.matmul_ogs` (OGS = One-Group-Stride)
- persistent matmul (stream-k)
- split-K reduction
- MoE expert parallelism
- MXFP/FP8 scaled dot

**重构内容**:
- Tensor/Layout/Distributed 大重构 (#9134)
- Closure-based output mapping (#8999)
- 更好的 API surface

**★ ★ 与我们关系**: vLLM 使用 Triton 内置 matmul (通过 `tl.dot`), 不直接用 `triton_kernels.matmul_*`。重构对 vLLM 影响有限。

### 7.5 Warp Specialization (SM90+)

**AutoWS**: Triton 3.4 引入自动 warp specialization, 3.7 继续完善:

- Nested loop support (#8687)
- 分割 warp: producer (load TMA) vs consumer (compute wgmma)
- `WarpSpecializePartitionsOp` 实现 RegionBranchInterface (#8799)
- Mixed TMA/non-TMA loads 修复 (#9111)

**★ RTX 4090**: Warp specialization 需要 SM90+ WGMMA → **不支持!**

### 7.6 SM89 ptxas Workaround Reverted (#9756)

**★ 重要**: v3.7 移除了 SM89 ptxas workaround (#9756, #7067)。

**历史**: 早期 Triton 版本为 SM89 ptxas bug 添加 workaround。v3.7 确认该 bug 在新版 CUDA driver 已修复, 移除 workaround。

**★ 影响**: 对 RTX 4090 用户, 如果使用旧 CUDA driver (< 12.9), 可能需要 Triton < 3.7 版本的 workaround。推荐使用 CUDA 12.9+。

### 7.7 Default 32-bit Dot Precision

v3.7 **短暂改为 TF32x3 → 又 revert** (#9080 → #9090)。

**最终状态**: v3.7 默认与 v3.6 一致 = TF32 (单次)。

**新增**: `make_tensor_descriptor` 支持 `round_f32_to_tf32=True` (#9295), 用户可显式控制 TF32 rounding 在 descriptor 层面。

### 7.8 Plugin Hooks & Out-of-Tree Dialects (#8401, #8523, #8815)

**★ 重要新特性**: Triton 现在支持用户自定义 TTIR/TTGIR pass 和 Triton Dialect Plugin!

- 用户可以注入自定义 compiler pass
- 可以定义新的 Triton IR dialect
- 有示例文档

**★ 与 Inductor Epilogue Fusion 关系**: 这为用户定义 Triton epilogue fusion 提供了底层基础, 但还不是 PyTorch Inductor 级别的便捷 API。

## 8. 实用示例: Flash Attention / Fused Softmax / Matrix Multiplication

### 8.1 Fused Softmax Kernel (Tutorial 02)

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # --- softmax ---
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator
        # --- store ---
        output_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, output, mask=mask)
```

**★ 关键技巧**:
- `other=-float('inf')`: OOB 位置填充 -inf → exp(-inf)=0 → softmax 分母不受 OOB 影响
- `tl.range()` 带 `num_stages`: software pipelining, overlap load 和 compute
- 每个 program 处理多行 (`row_start` + `row_step`), 提高 occupancy

### 8.2 Matrix Multiplication Kernel (Tutorial 03)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    # --- L2 cache optimized grid mapping ---
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Pointer arithmetic ---
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # --- Inner loop: tiled matmul ---
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Epilogue: store result ---
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

**★ ★ 关键设计**:
- **GROUP_SIZE_M**: L2 cache 优化, super-grouping 让连续 program 共享 A 的行 → 减少 DRAM 读取
- **指针增量**: K 维循环只增量 stride, 不重新计算 → 减少地址计算开销
- **tl.zeros**: 创建零累加器, 类型 float32 (BF16/FP16 matmul 的 accumulator 必须是 float32)
- **tl.dot(a, b)**: 自动使用 tensor core, 编译器决定 mma.sync vs wgmma

### 8.3 Flash Attention Triton Kernel (Tutorial 06)

vLLM V1 的 flash attention kernel 基于 Triton, 核心模式与 Tutorial 06 一致:

```python
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ...):
    # Load K block
    k = tl.load(K_block_ptr)
    # QK^T matmul
    qk = tl.dot(q, k)
    # Online softmax update
    m_ij = tl.max(qk, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    p = tl.exp(qk - m_i_new)
    l_ij = tl.sum(p, 1)
    alpha = tl.exp(m_i - m_i_new)
    l_i_new = alpha * l_i + l_ij
    acc = acc * alpha[:, None]
    # Load V block
    v = tl.load(V_block_ptr)
    # PV matmul
    acc += tl.dot(p.to(v.dtype), v)
    m_i = m_i_new
    l_i = l_i_new
    return acc, l_i, m_i
```

**★ vLLM V1 实际实现**:
- 使用 `make_block_ptr` (v0.23 开始迁移到 `make_tensor_descriptor`)
- 支持 causal masking (`tl.where(mask, qk, -inf)`)
- autotune 多个 BLOCK_M config
- ★ INT8 KV → `tl.dot(q, k.to(q.dtype))` 类型转换

### 8.4 TritonW4A16LinearKernel (vLLM INT4 Fallback, PR#43731)

**与我们直接相关!** vLLM 的 INT4 Triton fallback kernel:

```python
# TritonW4A16LinearKernel — 非 Marlin 的 Triton 实现
# SM89 上之前 crash → PR#43731 correctness fix → 现在可运行
# Triton ~2-5x slower than Marlin → 但之前完全不可能

@triton.jit
def _w4a16_linear_kernel(...):
    # unpack INT4 weights → BF16/FP16
    # tl.dot(weight_unpacked, input) → matmul
    # TritonW4A16LinearKernel = lowest priority safety net
    # ROCm-only gate removed → now CUDA + ROCm (SM80+ SM89 ✓)
```

**★ ★ RTX 4090 影响**: DS-V2-Lite (K=704 non-aligned) 和 Qwen2-MoE (K=2496) 之前 crash, 现在通过 Triton fallback 可以 load → 整体 ~5-15% 性能下降但功能可用。

## 9. PyTorch Inductor Triton Codegen 与 Epilogue Fusion

### 9.1 Inductor 如何生成 Triton Kernel

PyTorch Inductor (`torch._inductor`) 将 PyTorch FX IR 转换为 Triton kernel:

```
PyTorch Model → Dynamo → FX Graph → Inductor Lowering
  → Scheduler (决定 fusion groups)
  → Triton Codegen (生成 @triton.jit kernel)
  → 调用 triton.compile() → PTX → 执行
```

**Inductor 生成的 Triton kernel 特征**:
- 自动 `@triton.jit` 装饰
- 自动 block size 选择 (基于 shape 分析)
- 自动 `tl.load/tl.store` with mask
- 自动 `tl.dot` for matmul
- Epilogue fusion: matmul + pointwise (relu/bias/sigmoid) 融合到一个 kernel

### 9.2 Epilogue Fusion 现状

**Inductor 当前支持**: 自动将 pointwise 操作融合到 reduction/matmul kernel 的 epilogue。

**不支持的**: 用户自定义 Triton 代码注入 epilogue (没有官方 API)。

**已有讨论**:
- Issue #91113 / #88428: Feature request for `register_epilogue_hook`
- `ir.FusedEpilogue`: Inductor 内部 IR node, 扩展点正在讨论
- Triton v3.7 Plugin Hooks (#8401): 提供了底层基础

**★ ★ 与我们 RTX 4090 关系**: Inductor 自动 epilogue fusion 已经很强大 (BF16 训练 → `torch.compile` → Triton kernel 自动融合 activation/bias), 不需要手动 Triton kernel。rLLM Tinker + `torch.compile` 是最优组合。

## 10. RTX 4090 (SM89) 完整 Triton 能力分析

### 10.1 Triton 特性支持矩阵

| Triton 特性 | SM89 (RTX 4090) | 说明 |
|---|---|---|
| **@triton.jit** | ✓ | 核心编程模型, 完全支持 |
| **tl.load/tl.store** | ✓ | 基础内存操作, 完全支持 |
| **tl.dot (mma.sync)** | ✓ | Per-warp tensor core matmul |
| **tl.dot (FP16/BF16)** | ✓ | BF16/FP16 tensor core |
| **tl.dot (FP32→TF32)** | ✓ | TF32 truncation + tensor core |
| **tl.dot (INT8)** | ✓ | INT8 tensor core → INT32 acc |
| **tl.dot_scaled (MXFP)** | ✓ (软件模拟) | Upcast to BF16 then tl.dot |
| **tl.dot (FP8 E4M3)** | ✓ | Ada FP8 tensor core support |
| **tl.dot (FP8 E5M2)** | ✓ (有精度限制) | E5M2 训练不推荐 |
| **tl.arange/tl.full/tl.where** | ✓ | 基础操作, 完全支持 |
| **tl.reduce/tl.max/tl.sum** | ✓ | Reduction 操作, 完全支持 |
| **make_block_ptr** | ✓ (deprecated) | 软件模拟, v3.7 deprecated |
| **make_tensor_descriptor** | ✓ (软件模拟) | SM89 无 TMA, 软件模拟等价 |
| **TMA (硬件)** | ✗ | SM90 专属, Ada Lovelace 没有 |
| **WGMMA** | ✗ | SM90 专属 |
| **2CTA Cluster** | ✗ | 需要 SM90 cluster hardware |
| **Warp Specialization** | ✗ | 需要 WGMMA (SM90+) |
| **tcgen05.mma** | ✗ | Blackwell (SM100+) 专属 |
| **Software Pipelining** | ✓ | cp.async (64B-128B), 非 TMA |
| **CUDA Graph replay** | ✓ | Triton kernel 支持 CUDA graph |
| **Gluon Language** | ✓ (limited) | 无 TMA/warp_spec/2CTA |
| **Plugin Hooks** | ✓ | 用户自定义 pass, 纯软件 |
| **Proton Profiling** | ✓ | GPU profiling, SM89 支持 |
| **Autotune** | ✓ | 完全支持 |
| **CP.Async** | ✓ | Global→Shared async copy |

### 10.2 SM89 Triton 编译路径

```
@triton.jit kernel
  → TTIR → TTGIR
  → mma.sync (per-warp, SM80+ tensor core)
  → cp.async (global→shared, 64B/128B transactions)
  → bar.sync (shared memory barrier)
  → PTX → SASS (SM89 specific)
```

**SM89 不生成的指令**:
- `wgmma.mma_async` (SM90+)
- `cp.async.bulk.tensor` (SM90+ TMA)
- `cluster.barrier.arrive` / `cluster.wait` (SM90+ cluster sync)

### 10.3 RTX 4090 Triton 性能特征

**Memory-bound kernels** (softmax, layernorm, attention):
- Triton 生成高效的 cp.async + shared memory kernel
- 与手写 CUDA 性能接近 (2-5%差距)
- ★ Triton softmax 比 PyTorch native 快 ~4x (fused vs 5次 kernel launch)

**Compute-bound kernels** (matmul):
- Triton matmul tutorial 达到 cuBLAS 性能 ~90% (SM80 A100)
- RTX 4090 上, Triton matmul 受限于:
  - mma.sync (vs Hopper wgmma) → 更小的 tile per instruction
  - cp.async (vs TMA) → 更小的 async copy, 更多 warp 参与 load
  - 无 2CTA → 单 CTA 处理整个 output tile

**★ ★ 实际影响**: RTX 4090 上 Triton kernel 性能主要瓶颈不在 Triton 本身, 而在 SM89 硬件限制:
- 无 WGMMA → matmul tile 更小 → occupancy 更低
- 无 TMA → load 更慢 → pipeline depth 更短
- 但 Triton 生成的代码质量在 SM89 仍然很高, 自动优化大部分能做

### 10.4 RTX 4090 Triton 最佳实践

**1. Block size 选择**:
- SM89: BLOCK_M=64/128, BLOCK_N=64/128, BLOCK_K=32
- num_warps=4 (SM89 warp scheduler 最优)
- num_stages=2-3 (shared memory 有限, 24KB per CTA at 4 warps)

**2. 类型选择**:
- BF16 matmul → tl.dot BF16, accumulator float32 ✓
- INT4 → TritonW4A16LinearKernel (vLLM fallback) ✓
- FP32 matmul → input_precision="tf32" (默认) 或 "ieee" (精确但慢)

**3. 内存优化**:
- 优先用 pointer arithmetic + mask (比 block ptr 更灵活)
- make_tensor_descriptor 在 SM89 等价于 make_block_ptr, 无性能差异
- `cache_modifier=".cg"` 对 decode-heavy kernel 有帮助 (L2 cache, 不 cache L1)

**4. 避免**:
- ✗ 不要用 `num_ctas>1` (SM89 没有 cluster)
- ✗ 不要用 warp specialization 相关 API
- ✗ 不要用 Gluon TMA 相关 API
- ✗ 不要用 tcgen05 相关 API

### 10.5 与 vLLM / verl / rLLM 的 Triton 使用对比

| 框架 | Triton 使用 | RTX 4090 可行性 |
|---|---|---|
| **vLLM** | Flash attention (Triton) + W4A16 fallback + layernorm | ✓ (INT4 Triton fallback 特别重要!) |
| **PyTorch Inductor** | `torch.compile` → Triton kernel 自动生成 | ✓ (SM89 matmul + pointwise fusion) |
| **verl** | 委托 vLLM/PPO, 无直接 Triton | ✓ (间接) |
| **rLLM Tinker** | `torch.compile` → Triton (BF16 LoRA 训练) | ✓ ✓ (最优路径!) |
| **Megatron** | InferenceTopKRouter @torch.compile → Triton | ✓ (SM89 不用 WGMMA/TMA, 但 mma.sync ✓) |

**★ ★ ★ 总结**: RTX 4090 上 Triton 是 **核心工具**, 不是限制。SM89 没有 Hopper 的新硬件特性, 但 Triton 在 SM89 上仍然生成高质量 kernel。关键路径:
- rLLM Tinker + `torch.compile` (Triton) → BF16 LoRA 训练 → merge → INT4 → vLLM (Triton fallback + flash attention) → 4,791→9,088 tok/s

## 11. Triton 编程模型 vs CUDA 完整对比

| 方面 | CUDA C | Triton (@triton.jit) |
|---|---|---|
| **语言** | C++ with PTX inline | Python |
| **执行单元** | Thread (threadIdx.x/y/z) | Program (tl.program_id) |
| **内存层级** | 手动管理 shared/global | 自动 shared memory 管理 |
| **Tensor Core** | 手动 wmma/mma.sync/wgmma | tl.dot → 自动 lowering |
| **内存合并** | 手动设计访问模式 | 编译器自动 coalescing |
| **Synchronization** | 手动 __syncthreads/bar.sync | 编译器自动插入 |
| **Pipeline** | 手动 cp.async + multi-buffer | 编译器自动 pipelining (num_stages) |
| **Block Size** | 手动选择 blockDim | @triton.autotune 或 constexpr |
| **边界检查** | 手动 if (idx < N) | mask 参数 / boundary_check |
| **编译** | nvcc → PTX → SASS (预编译) | JIT: AST → TTIR → TTGIR → PTX (运行时) |
| **调试** | cuda-gdb / Nsight | TRITON_INTERPRET / Proton / Nsight |
| **性能调优** | 手动调 blockDim/tiling | @triton.autotune 自动 benchmark |
| **共享内存** | `__shared__` 声明 | 编译器自动分配和 swizzle |
| **Bank Conflict** | 手动 padding/swizzle | 编译器自动避免 |

**★ 核心差异**: Triton 把 GPU 编程从 **thread-level** 提升到 **block-level**, 让编译器处理 thread-level 优化。代价是灵活性降低 (不能直接控制 thread 行为), 但收益是开发效率大幅提升, 且编译器优化的代码质量通常接近手动调优。

## 12. Triton Gluon 框架 (v3.4+)

**Gluon** 是 Triton 的下一代编程框架 (v3.4 引入, v3.6-3.7 大幅增强):

```python
from triton.experimental.gluon.language import *

@gl.warp_specialize  # warp specialization (SM90+)
@gl.jit
def persistent_matmul(...):
    # producer warp: load TMA
    # consumer warp: compute wgmma
```

**Gluon 新增 (v3.7)**:
- Multi-CTA support (#8468)
- `num_ctas` 实现 (#8602)
- Device-side TMA (#8505)
- Local scatter/gather (#8480)
- `get_view()` (#9270)
- Finer cluster fences (#9076)
- Warp-pipeline on AMD (#8586)

**★ ★ RTX 4090**: Gluon 的 TMA/warp_specialize/2CTA/multi-CTA 功能在 SM89 不可用, 但基础 `gl.jit` 和普通 tl 操作可用。Gluon 对 RTX 4090 用户价值有限 — 继续用标准 `@triton.jit` 更合适。

## 13. Triton 3.6→3.7 版本演进关键 PR

| PR# | 内容 | SM89 影响 |
|---|---|---|
| #9667 | make_block_ptr deprecated | ★ 低 — 迁移到 make_tensor_descriptor (等价) |
| #8765 | triton_kernels matmul refactor (BC-breaking) | 低 — vLLM 不用 triton_kernels 直接 |
| #8684 | 2CTA end-to-end | ✗ — SM89 不支持 |
| #9005 | TMA + multicast | ✗ — SM89 不支持 |
| #9202 | TMA im2col | ✗ — SM89 不支持 |
| #8785 | Returning constexpr from JIT | ✓ — 纯软件, SM89 受益 |
| #8882 | FP8 constants | ✓ — SM89 支持 |
| #9295 | round f32→tf32 in descriptor | ✓ — SM89 软件模拟也可用 |
| #8401 | Plugin Hooks & Out-of-Tree Dialects | ✓ — 纯软件 |
| #9756 | SM89 ptxas workaround reverted | ★ — 需要 CUDA 12.9+ driver |
| #8924 | tl.squeeze/tl.unsqueeze | ✓ — 纯软件 |
| #9080→#9090 | Dot precision: TF32x3→revert to TF32 | ✓ — 默认 TF32, SM89 不变 |

## 14. 总结与 RTX 4090 实战指南

### ★★★ 核心结论

1. **Triton 在 SM89 完全可用**: @triton.jit + tl.dot + tl.load/tl.store + autotune — 核心编程模型不受 SM89 限制
2. **SM89 硬件限制**: 无 TMA/WGMMA/2CTA/warp_spec — 但这些是 Hopper/Blackwell 特性, Triton 自动 fallback 到 SM89 路径 (mma.sync + cp.async)
3. **★ ★ INT4 Triton fallback 是 RTX 4090 生命线**: vLLM PR#43731 让 DS-V2-Lite/Qwen2-MoE 在 SM89 上可运行 (之前 crash), Triton ~2-5x 慢于 Marlin 但 functional
4. **★ ★ torch.compile → Triton 是 RTX 4090 训练最优路径**: BF16 LoRA → compile → Triton kernel → 自动 epilogue fusion → 接近手写 CUDA 性能
5. **make_block_ptr → make_tensor_descriptor**: 迁移对 SM89 无性能差异, 但新 API 更清晰

### ★★★ RTX 4090 Triton 技术栈

```
[训练] rLLM Tinker + torch.compile (Triton) + BF16 LoRA
[推理] vLLM + Triton flash attention + INT4 Triton fallback + INT8KV Triton kernel
[研究] Triton 自定义 kernel (softmax/layernorm/attention) + autotune
[分析] Proton profiling + Nsight Compute
```

### ★★★ 避免

- ✗ 不要在 SM89 用 TMA/warp_spec/2CTA/Gluon advanced features
- ✗ 不要用 num_ctas>1
- ✗ 不要用 tcgen05 (Blackwell)
- ✗ 不要用 FP8 E5M2 训练精度
- ✗ 确保 CUDA driver >= 12.9 (SM89 ptxas workaround 已移除)

---

**参考源码路径** (triton-lang/triton GitHub):
- `python/triton/runtime/jit.py` — @triton.jit, JITFunction, DependenciesFinder
- `python/triton/runtime/autotuner.py` — @triton.autotune, Autotuner, Config
- `python/triton/compiler/compiler.py` — compile(), ASTSource, IRSource
- `python/triton/compiler/code_generator.py` — ast_to_ttir()
- `python/triton/language/core.py` — tl.dot/tl.load/tl.store/tl.arange/tl.full/tl.where/tl.reduce/make_tensor_descriptor
- `python/triton/language/standard.py` — tl.cdiv, tl.sigmoid
- `python/tutorials/01-vector-add.py` — 基础 Triton kernel
- `python/tutorials/02-fused-softmax.py` — Fused softmax
- `python/tutorials/03-matrix-multiplication.py` — Matmul + L2 cache优化
- `python/tutorials/06-fused-attention.py` — Flash attention

**Triton v3.7.0 Release Notes**: https://github.com/triton-lang/triton/releases/tag/v3.7.0
