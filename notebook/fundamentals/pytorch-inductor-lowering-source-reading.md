# PyTorch Inductor Lowering 源码阅读

> 源码: torch/_inductor/lowering.py (5400行)
> 核心: FX graph aten ops → Inductor IR (Pointwise/Reduction/Scatter/Extern/Fallback)

## Lowering 入口: FX Node Dispatch

lowering.py 本身不含 main dispatch — 实际在 `graph.py` 的 `GraphLowering` 类:

```python
# graph.py line 655
def call_function(self, target, args, kwargs):
    if hasattr(target, "_inductor_lowering_function"):
        return target(*args, **kwargs)  # pattern_matcher bypass

    out = lowerings[target](*args, **kwargs)  # 标准 dispatch
```

**3条 dispatch 路径**:
1. `_inductor_lowering_function`: pattern-matcher注册的op → 直接调用，绕过lowerings dict
2. `lowerings[target]`: 标准路径 → dict映射每个aten op到其lowering函数
3. Fallback: 不在lowerings且不在allow list → make_fallback()或报错

**完整流水线**: Dynamo trace → AOT Autograd decompose → FX graph → GraphLowering interpreter逐节点 → lowerings[target] → Inductor IR

---

## Lowering Registry System

### Core Data Structures (lines 61-69)

```python
lowerings = {}          # op → lowering function (主dispatch表!)
layout_constraints = {} # op → layout constraint function
fallbacks = set()       # 使用FallbackKernel的op (无IR codegen)
needs_realized_inputs = set()  # 需要materialized buffer的op
foreach_ops = set()
inplace_foreach_ops = set()
```

### `register_lowering()` (lines 302-317) — 装饰器入口

```python
def register_lowering(aten_fn, broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT):
    return functools.partial(_register_lowering, aten_fn, ...)
```

两种用法:
- 装饰器: `@register_lowering(aten.where, broadcast=False)(def where(...))`
- 直接注册: `register_lowering(aten.clamp_min)(maximum)` — 复用已有lowering

### `_register_lowering()` (lines 252-299) — 注册核心

每个注册的lowering被wrapped:
1. `transform_args()`: broadcast + type promotion预处理
2. `decomp_fn`: 实际lowering实现
3. `validate_ir(out)`: 确保输出是proper IR类型
4. `lowerings.update({fn: wrapped})` — 注册到全局dict

### `transform_args()` (lines 193-224) — 预处理pipeline

两个操作在lowering函数看到args之前:
1. **Type promotion**: 计算promoted dtype → `to_dtype()` 每个TensorBox
2. **Broadcast**: `broadcast_tensors()` 对齐shapes → Constants wrap with `ExpandView`

### `register_pointwise()` (lines 570-609) — 批量Pointwise注册

简单elementwise op一行注册:
```python
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
```

Internal: `ops_wrapper(name)` → `make_pointwise(fn)` → `register_lowering(aten_fn)(fn)`

### `register_lowering_pattern()` (pattern_matcher.py line 1017)

Pattern-based lowerings (mm+mm, fused attention等):
```python
handler._inductor_lowering_function = True  # bypass lowerings dict!
```
在FX graph上pattern-match BEFORE call_function运行 → matched handler直接调用

---

## Key LLM-Relevant Lowering Rules

### Matmul / Linear / addmm

**不在lowering.py** — 注册在 `kernel/mm.py` 和 `kernel/bmm.py`:

```python
@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    choices = [aten_mm] if use_aten_gemm_kernels() else []
    # + Triton template configs
    # + CUTLASS template configs
    return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)
```

**关键**: mm/bmm/addmm都用 **autotune** — 产出IR节点代表ATen cuBLAS / Triton template / CUTLASS之间的选择

**mm+mm fusion** (post_grad.py line 176): pattern-match `aten.add(aten.mm(...), aten.mm(...))` → `tuned_mm_plus_mm` — 专用fused kernel

### Attention (SDPA) — 全是 Fallback!

```python
make_fallback(aten._scaled_dot_product_efficient_attention.default, sdpa_constraint)
make_fallback(aten._flash_attention_forward.default, sdpa_constraint)
make_fallback(aten._efficient_attention_forward.default, sdpa_constraint)
```

**sdpa_constraint** (line 2031): layout约束:
- dense last dimension (stride_order[-1]==0)
- last dim size multiple of 8
- backward需要stride-1 last dim + aligned other strides

还有serialized pattern-based SDPA lowerings: `bmm+softmax+bmm` → fused SDPA (在lowering前transform FX graph)

### SiLU (和 GLU)

**SiLU不在lowering.py** — 通过 **decomposition** 处理:
`aten.silu` → `mul(x, sigmoid(x))` → 两个Pointwise IR节点 → scheduler fuse成1个Triton kernel

**GLU** (line 1242): 直接lowered:
```python
def glu(x, dim=-1):
    a = slice_(x, dim, 0, new_len)
    b = slice_(x, dim, new_len, new_len*2)
    return mul(a, sigmoid(b))
```

### Layer Norm / RMS Norm

**两者都先 decompose** 再 lowering:
`aten.native_layer_norm` → mean→sub→square→sum→div→mul→add→mul序列 → Pointwise/Reduction IR

RMS norm不在Inductor decomp表 → custom pattern或fallback

### Reductions (sum, mean, amax, argmax)

**`make_reduction()`** (lines 4534-4550) — factory:

```python
def make_reduction(reduction_type, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False):
        result = Reduction.create(reduction_type=reduction_type, input_node=x, ...)
        if isinstance(result.data.data, Reduction):
            result.realize()  # force materialization
        return result
    return inner
```

注册:
```python
sum_ = make_reduction("sum")
reduce_amax = make_reduction("max")
reduce_argmax = make_reduction("argmax", override_return_dtype=torch.int64)
```

**mean**: `sum / denom` — 不是简单reduction

**Variance** 两种实现:
- **Two-step** (小reduction): sum of diffs squared → divide
- **Welford** (大reduction): `ir.WelfordReduction.create()` — 单pass数值稳定

---

## IR Node 分类和 Lowering 差异

| IR Node | 产出来源 | Scheduler行为 |
|---------|----------|---------------|
| `Pointwise` | elementwise ops, gather, slice_scatter | Fuse到Triton kernel |
| `Reduction` | sum, mean, amax, argmax, welford | Reduction kernel + epilogue fusion |
| `ir.Scatter` | scatter_, index_put, scatter_reduce | Mutation kernel (可能atomic) |
| `ir.FallbackKernel` | make_fallback ops (SDPA, sort) | ATen kernel直接调用 (无codegen) |
| `ExternKernel` (autotune) | mm, bmm, addmm | Autotuned GEMM kernel |
| `ir.WelfordReduction` | var_mean (大reduction) | 单pass variance kernel |
| `AllReduce/AllGather` | distributed ops | 集合通信 |

### Pointwise — Lazy Composition

```python
def make_pointwise(fn, ...):
    def inner_fn(index):
        return fn(*[load(index) for load in loaders])
    return Pointwise.create(device=device, inner_fn=inner_fn, ranges=ranges)
```

每个pointwise op提供 `inner_fn(index)` → scheduler **fuse** 连续pointwise通过 **compose** inner_fn → 1个Triton kernel

### Scatter/Gather 差异

**Gather**: Pointwise + `ops.indirect_indexing` → 读不规则位置
**Scatter**: `ir.Scatter` + `MutationLayout` + `atomic_add` → 写不规则位置 → 需特殊IR

---

## Dynamic Shapes / Symbolic Variables

### `broadcast_symbolic_shapes()` (lines 320-341)

sympy表达式broadcast:
```python
for x, y in zip_longest(reversed(a), reversed(b), fillvalue=1):
    if y == 1: output.append(x)
    elif x == 1: output.append(y)
    else: V.graph.sizevars.guard_equals(x, y)
```

### `sym_size/sym_stride` (lines 5172-5199)

```python
@register_lowering(aten.sym_size.int)
def sym_size(a, dim):
    val = V.graph.current_node.meta["val"]
    return val.node.expr  # 返回sympy表达式, 不是Python int!
```

**Static vs Dynamic** (graph.py line 638):
- Weights → static shapes (具体整数)
- Dynamic inputs → symbolic shapes (sympy表达式)

### Welford vs Two-step 选择

```python
def use_two_step_variance():
    return isinstance(reduction_numel, sympy.Integer)
        and int(reduction_numel) < config.unroll_reductions_threshold
```

---

## Fallback System

### `make_fallback()` (line 1673)

不可codegen的op:
1. 确认无decomposition (否则应用decomp)
2. 加入 `needs_realized_inputs` (必须materialize输入buffer)
3. 可选 `layout_constraint`
4. `fallback_handler(op)` → `ir.FallbackKernel.create()` → ATen op直接调用 → 无Triton codegen → **打破fusion chain!**

### Key LLM Fallbacks

| Op | 约束 | 影响 |
|----|------|------|
| SDPA (所有变体) | stride_order+alignment | FlashAttention=ATen fallback, 不codegen |
| `aten._scaled_mm` (FP8 matmul) | constrain_to_fx_strides | FP8 GEMM=ATen fallback |
| sort, topk | 无 | 排序=ATen fallback |
| addbmm, addmv | 无 | 稀疏GEMM=ATen fallback |

---

## Collective Ops (Distributed)

```python
@register_lowering(c10d_functional.all_reduce)
def allreduce(input, reduce_op, tag, ranks, group_size):
    return ir.AllReduce.create(input, reduce_op, tag, ranks, group_size)
```

还有: `wait`, `broadcast`, `all_gather_into_tensor`, `reduce_scatter_tensor` → 专用IR节点 → scheduler单独处理

---

## Triton Kernel Wrap (Custom Triton Integration)

```python
@register_lowering(triton_kernel_wrapper_functional)
def triton_kernel_wrap(*, kernel_idx, grid, kwargs, tensors_to_clone):
    kwargs = {key: (clone(x) if key in tensors_to_clone else x) for key, x in kwargs.items()}
    return triton_kernel_wrap_(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
```

允许 `@triton.kernel` 函数与Inductor编译pipeline集成

---

## Architecture Summary: Lowering Pipeline

```
FX Graph Node (aten op)
    │
    ▼
GraphLowering.call_function()
    │
    ├─ Has _inductor_lowering_function? → pattern_matcher (直接调用)
    │
    ├─ In lowerings dict? → lowerings[target](*args, **kwargs)
    │     ├─ transform_args: broadcast + type promotion
    │     ├─ decomp_fn: 实际lowering实现
    │     ├─ validate_ir: 检查输出IR类型
    │     │
    │     ├─ Produces IR nodes:
    │         - Pointwise (elementwise, gather, slice_scatter)
    │         - Reduction (sum, max, welford)
    │         - Scatter (scatter_, index_put)
    │         - ExternKernel+autotune (mm, bmm, addmm)
    │         - FallbackKernel (SDPA, sort, topk)
    │         - UserDefinedTritonKernel (custom Triton)
    │         - AllReduce/AllGather (distributed)
    │
    ├─ Not in lowerings? → make_fallback() 或 raise error
```

**核心设计原则**: Lowering是 **Lazy** — Pointwise/Reduction不立即materialize，只作为computation描述(inner_fn closures)。下游需要realized buffer时才force buffer creation。这种laziness是scheduler fusion的基础!

---

## LLM 推理/训练 Impact

| 模式 | Lowering路径 | Fusion机会 |
|------|-------------|------------|
| 训练(reduce-overhead) | mm→autotune, silu→2 pointwise→fuse, norm→reduction+pointwise→fuse | 高 |
| 推理(max-autotune) | 同上但MaxAutotune选最优config | 高 |
| SDPA | Fallback! FlashAttention直接ATen → 不codegen | 无(ATen内核) |
| FP8 GEMM | Fallback! scaled_mm → 不codegen | 无(TE内核) |
| ZeRO-3+compile | AllGather/ReduceScatter → ir.AllReduce → break fusion | 低 |
| FSDP2+compile | AllGather per-param → compile看到静态图 → fusion正常 | 高 |

**关键结论**:
- SDPA/FP8 GEMM都是fallback → compile不会优化这些 → 依赖ATen/TE原生性能
- FSDP2 per-param AllGather → compile可见静态图 → fusion机会最大化 → FSDP2+compile最优组合
- ZeRO-3 dynamic AllGather → compile看到dynamic → graph break → fusion中断 → ZeRO-3+compile不兼容
