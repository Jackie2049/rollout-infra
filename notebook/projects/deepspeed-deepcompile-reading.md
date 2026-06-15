# DeepSpeed DeepCompile 源码级深度阅读

> 日期: 2026-06-15 | 基于源码: `_temp_deepspeed/deepspeed/compile/` + `csrc/compile/` | arXiv: [2504.09983](https://arxiv.org/abs/2504.09983)
> 作者: Masahiro Tanaka, Du Li, Umesh Chand, Olatunji Ruwase (Microsoft); Ali Zafar, Haiying Shen (UVA)
> 版本: DeepSpeed >= v0.16.6 (2025-06首次发布)

---

## 1. DeepCompile 解决什么问题?

### 1.1 ZeRO-3 + torch.compile 不兼容的根源

传统ZeRO-3实现通过Python hooks实现参数的gather/release:

```
forward前: pre_forward_hook → AllGather参数 → 全尺寸参数可见
forward后: post_forward_hook → Release参数 → param.data = empty(0)
backward:  类似的gather/release + ReduceScatter梯度
```

**torch.compile (Dynamo + AOTAutograd + Inductor) 为什么和ZeRO-3冲突?**

1. **Dynamo tracing时遇到hooks → graph break**: ZeRO-3的`pre_forward_hook`和`post_forward_hook`是Python-level注册的 → Dynamo看到hooks → 无法静态分析 → 触发graph break → 多段编译 → 优化碎片化
2. **参数形状不稳定**: ZeRO-3参数在gather时是全尺寸(如4096), release后变成`empty(0)` → Dynamo的tensor-match guards期望稳定shape → gather时记录的shape(4096) vs release后的shape(0) → guard失败 → recompilation风暴
3. **GatheredParameters context manager**: ZeRO-3用`GatheredParameters`临时恢复参数 → Dynamo无法理解这种动态scope → graph break
4. **条件分支**: ZeRO-3内部有很多条件逻辑(prefetch触发条件、overflow检查等) → 编译器无法resolve → 更多graph break

**blog README原文**: "When the compiler encounters branches that cannot be statically resolved, it splits the computation into multiple graph segments. These fragmented segments can reduce optimization opportunities and introduce additional overheads during execution."

### 1.2 DeepCompile的核心思路: Freeze Schedule

DeepCompile不试图让Dynamo"理解"ZeRO-3 hooks, 而是**在Dynamo tracing之前完全移除hooks**:

```
传统ZeRO-3: 模型初始化 → 注册hooks → forward时hooks动态gather/release
DeepCompile: 模型初始化(ZeRO-3) → compile()调用 → 移除hooks → Dynamo tracing → FX graph → compiler passes在graph中插入gather/release → 编译执行
```

**这本质上是把"运行时动态操作"变成"编译时静态插入"** → 从hook-based变成pass-based。

---

## 2. DeepCompile 初始化流程: init_z3.py (源码级详解)

源码: `deepspeed/compile/init_z3.py` (111行)

### 2.1 移除ZeRO-3 hooks — "freeze"操作

```python
def init_z3(engine, backend, compile_config, compile_kwargs, schedule=None):
    optimizer = engine.optimizer
    use_opt = not isinstance(optimizer, DeepSpeedZeRoOffload)

    # Step 1: 获取C++ deepcompile handle
    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    # ★★★ Step 2: 核心freeze操作 — 恢复原始参数, 移除所有hooks ★★★
    # 2a: 恢复模块的原始参数字典(而非分区后的ds_tensor)
    for m in engine.module.modules():
        m._parameters = m._original_parameters  # ← 从partitioned恢复为full params

    # 2b: 移除parameter_offload的所有module hooks
    if use_opt:
        optimizer.parameter_offload._remove_module_hooks()  # ← pre/post forward hooks
        # 2c: 移除所有gradient accumulation hooks
        for hook in optimizer._grad_acc_hooks:
            hook.remove()
        optimizer._grad_acc_hooks.clear()

    # 2d: 恢复原始linear函数(ZeRO-3曾patch F.linear用于参数替换)
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk
```

**关键洞察**: `_original_parameters`是ZeRO-3在partitioning前保存的原始参数字典 → 此时参数仍以`ds_tensor`(分区形式)存在于GPU → 但`_parameters`指向的是原始full-size引用 → Dynamo tracing时看到full-size参数 → shape稳定 → 不触发graph break

### 2.2 注册Z3参数到C++ handle

```python
    for p in engine.module.parameters():
        # 每个参数的grad_buffer(分区形式, 用于ReduceScatter接收)
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # ★ 禁用persistent(默认全部非persistent → 按需gather)
        p.ds_persist = False

        # 注册到C++ DSParamRegistry
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist,
                             _resolve_expected_grad_dtype(p))
```

C++侧(`csrc/compile/z3.cpp`的`register_z3_param`)做:
- 将参数注册到`DSParamRegistry` → 记录ds_shape、ds_tensor(分区)、grad_buffer
- 验证所有rank的padded shard size一致 → AllGather要求uniform shard size
- persistent=True时立即执行allgather → 之后该参数始终驻留GPU

### 2.3 设置compile pass schedule

```python
    if schedule is None:
        schedule = []
        if compile_config.offload_parameters:
            # offload参数 → 只一个pass: gather/release + offload_parameter_fwd
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]))
        else:
            # ★ 默认schedule: 两阶段 ★
            # 阶段1(step 0): 只插入gather/release → baseline profiling
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            # 阶段2(step WARMUP=5): gather/release + prefetch + selective_gather
            schedule.append(
                (WARMUP, [zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather]))
```

**WARMUP=5的含义**: 前5个step用baseline gather/release → 收集profiling数据(内存使用、通信时间) → 第6个step开始用profiling-guided优化(prefetch + selective_gather)

### 2.4 其他初始化步骤

```python
    # 设置backward前的grad_buffer hook
    if use_opt:
        add_pre_backward_hook(set_grad_buffer)

    # offload optimizer states需要额外setup
    from .passes.offload_adam_states import move_opt_states, move_opt_states_sync, init_offload_opt_states
    for _, passes in schedule:
        if move_opt_states in passes or move_opt_states_sync in passes:
            init_offload_opt_states(optimizer, dc)

    # ★★ patch FakeTensor → 让Dynamo tracing看到full-size而非empty(0) ★★
    patch_fake_tensor()

    # ★ 禁用Inductor size asserts → ZeRO-3参数从分区gather到full → size变化
    torch._inductor.config.size_asserts = False

    # 创建backend → Dynamo compile
    return make_backend(backend, compile_config, compile_kwargs=compile_kwargs)
```

### 2.5 engine.py中的hook管理

源码: `deepspeed/runtime/engine.py` `_set_deepcompile_active()`

```python
def _set_deepcompile_active(self, active: bool):
    if self._deepcompile_active == active:
        return
    if active:
        # DeepCompile激活 → 移除ZeRO-3的forward hooks → 由compiler passes接管
        if self.module_forward_pre_hook is not None:
            self.module_forward_pre_hook.remove()
            self.module_forward_pre_hook = None
        if self.module_forward_post_hook is not None:
            self.module_forward_post_hook.remove()
            self.module_forward_post_hook = None
    else:
        # DeepCompile关闭 → 重新安装ZeRO-3 hooks → 回到eager模式
        if self.module_forward_pre_hook is None:
            self.module_forward_pre_hook = self._create_module_forward_pre_hook()
        if self.module_forward_post_hook is None:
            self.module_forward_post_hook = self._create_module_forward_post_hook()
    self._deepcompile_active = active
```

**关键**: DeepCompile激活时 → ZeRO-3的module-level forward pre/post hooks被移除 → compiler passes在graph中插入的gather/release取代了hook-based的动态gather/release

---

## 3. patch_fake_tensor.py — Dynamo tracing兼容的关键patch

源码: `deepspeed/compile/patch_fake_tensor.py` (77行)

### 3.1 问题: Dynamo guard shape mismatch

PyTorch v2.9+开始严格执行parameter tensor-match guards:

```
Dynamo tracing时:
  ZeRO-3参数暂时all-gathered → size = (4096) → Dynamo记录guard: "size == (4096)"

Guard evaluation时(后续step):
  DeepSpeed已release参数 → param.data = empty(0) → size = (0)
  Guard检查: expected (4096), actual (0) → MISMATCH → recompilation风暴
```

### 3.2 解决方案: patch Dynamo的fake tensor wrapping

```python
def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):  # ← ZeRO-3 partitioned参数
        # 创建dummy tensor → 用ds_shape(全尺寸)而非实际size(可能是0)
        data = torch.rand(t.ds_shape, dtype=t.dtype, ...)  # ← ds_shape是全尺寸!
        if isinstance(t, torch.nn.Parameter):
            t = torch.nn.Parameter(data, requires_grad=t.requires_grad)
    return t
```

**核心**: ZeRO-3参数的`ds_shape`属性记录了**全尺寸** → 但当前`param.data`可能是`empty(0)` → wrap_if_ds_param创建dummy的全尺寸tensor → Dynamo tracing看到稳定shape → 不触发guard mismatch

### 3.3 patch_guard_sizes_strides — 让guard匹配"released"状态

```python
def _get_guard_sizes_strides(t):
    if hasattr(t, "ds_id"):
        # ★ guard应该匹配的是released状态 → empty(0) → 这是"稳定"状态
        released = torch.empty(0, dtype=t.dtype, device=t.device)
        return released.size(), released.stride()
    return t.size(), t.stride()
```

**洞察**: tracing时用全尺寸(dummy) → 但guard匹配时用released状态(empty(0)) → 因为DeepCompile编译后, 参数在大部分时间处于released状态 → guard匹配released才正确

### 3.4 实际patch操作

```python
def patch_fake_tensor():
    # Patch 1: wrap_to_fake_tensor_and_record (Dynamo tracing入口)
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record
    def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
        dummy_tensor = wrap_if_ds_param(t)  # ← 用全尺寸dummy
        ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
        # ★ 保持symbolic context来自dummy → 但guard sizes/strides来自released状态
        size, stride = _get_guard_sizes_strides(t)
        tx.output.input_source_to_sizes_strides[source] = {"size": size, "stride": stride}
        return ret
    torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record_wrapper

    # Patch 2: FakeTensorMode.from_tensor (AOTAutograd入口)
    original_from_tensor = FakeTensorMode.from_tensor
    def from_tensor_wrapper(self, t, *args, **kwargs):
        return original_from_tensor(self, wrap_if_ds_param(t), *args, **kwargs)
    FakeTensorMode.from_tensor = from_tensor_wrapper
```

**这是DeepCompile最精妙的patch**: 让Dynamo"看到"full-size参数 → 产生正确的FX graph → 但guards匹配的是released(empty)状态 → 避免recompilation

---

## 4. backend.py — 编译执行框架

源码: `deepspeed/compile/backend.py` (406行)

### 4.1 整体架构

```
engine.compile() → init_z3() → make_backend() → 返回backend_fn
                                       ↓
                               torch._dynamo.optimize(backend_fn)
                                       ↓
                               Dynamo tracing → backend_fn(gm, real_inputs)
                                       ↓
                               AOTAutograd → partition → fw_compiler / bw_compiler
                                       ↓
                               make_fw_graph / make_bw_graph → run_opt_passes
                                       ↓
                               compiler passes修改graph → Inductor codegen
```

### 4.2 GraphOrder — 跟踪Dynamo产生的多个graph

```python
class GraphOrder:
    def __init__(self):
        self.frames = OrderedDict()  # frame_id → (graph_id, needs_backward)

    def add_graph(self, graph_id, frame_id):
        if frame_id not in self.frames:
            self.frames[frame_id] = (graph_id, None)

    def set_needs_backward(self, frame_id, needs_backward):
        self.frames[frame_id] = (self.frames[frame_id][0], needs_backward)
```

Dynamo可能产生多个编译frame → GraphOrder跟踪每个frame的graph_id和是否需要backward → 用于pass执行时确定graph顺序

### 4.3 launch_compile_passes — 按schedule触发pass

```python
def launch_compile_passes(global_steps: int):
    if len(remaining_schedule) > 0 and global_steps == remaining_schedule[0][0]:
        _, next_passes = remaining_schedule.popleft()
        # ★ 重置所有编译状态 → 重新编译 → 新pass生效
        torch._dynamo.reset()
        get_deepcompile_handle().reset()
        graph_order_with_frame_id.clear()
        profiling_results.clear()
        param_manager.clear()
        frames_partitioned.clear()
```

**关键**: 每次pass schedule触发时 → 全部重置 → 重新trace → 重新编译 → 新的pass组合生效 → profile-guided optimization的循环

### 4.4 backend_fn — Dynamo的backend回调

```python
def backend_fn(gm: GraphModule, real_inputs):
    graph_id = id(gm.graph)
    frame_id = gm.meta["dynamo_compile_id"].frame_id
    graph_order_with_frame_id.add_graph(graph_id, frame_id)

    # ★ 判断是否ZeRO-3 → 检查inputs是否有ds_id
    z3_partition = any(hasattr(v, "ds_id") for v in real_inputs)

    # 获取参数索引 → (input_position, ds_id, ds_shape)
    if z3_partition:
        param_indices = [(i, input_val.ds_id, input_val.ds_shape)
                         for i, input_val in enumerate(real_inputs)
                         if isinstance(input_val, torch.nn.Parameter)]
    else:
        param_indices = [(i, input_val.param_id, input_val.shape)
                         for i, input_val in enumerate(real_inputs)
                         if isinstance(input_val, torch.nn.Parameter)]

    # 创建InputStorage → 保存real_inputs元数据
    input_storage = InputStorage(...)
    input_storage.put(real_inputs)

    # 为这个graph创建DSGraphParamManager
    param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)
```

### 4.5 make_fw_graph / make_bw_graph — forward/backward编译回调

**make_fw_graph**:
```python
def make_fw_graph(gm, sample_inputs):
    # 从InputStorage或fwd_real_inputs获取真实输入
    real_inputs = fwd_real_inputs.pop(0) or input_storage.get()
    real_inputs = set_example_values_to_symints(real_inputs)

    # 创建DSGraphParamManager → 管理参数到graph的映射
    param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)

    # ★ 执行所有compiler passes → 修改FX graph
    run_opt_passes(opt_passes=next_passes, gm=gm, ...)

    # 清空ds_param的meta["val"] → 防止Inductor size validation
    for n in gm.graph.nodes:
        if is_ds_param:
            n.meta["val"] = torch.empty([0], ...)  # ← 告诉Inductor这个参数size是0

    # ★ fast_free_schedule → 内存感知的op调度
    gm.graph = fast_free_schedule(gm.graph, get_accelerator().available_memory(), ...)

    return gm.graph
```

**make_bw_graph**:
```python
def make_bw_graph(gm, sample_inputs):
    # 获取backward的真实输入 → 从patch_compiled_func捕获
    bwd_inputs_stack = get_backward_inputs()
    bwd_real_inputs = bwd_inputs_stack.pop()

    # 执行compiler passes
    run_opt_passes(opt_passes=next_passes, gm=gm, ...)

    # ★ free activation → 在backward中释放大activation
    if free_activation:
        add_free_activations(graph_id, gm.graph, ...)

    # 移除patch → 恢复原始autograd.Function
    frames_needing_bwd.remove(frame_id)
    if len(frames_needing_bwd) == 0:
        unpatch_compiled_func()

    return gm.graph
```

### 4.6 两种backend: eager vs inductor

```python
if backend == "eager":
    # 使用AOTAutograd的aot_module_simplified → fw/bw分开编译
    partition_fn = get_wrapped_partitioner(z3_partition, ...)
    aot_mod = aot_module_simplified(gm, real_inputs,
                                    fw_compiler=..., bw_compiler=...,
                                    partition_fn=partition_fn)
    return torch._dynamo.optimize(**compile_kwargs)(aot_mod)

elif backend == "inductor":
    # ★ Patch Inductor的AOT kwargs → 注入DeepCompile的fw/bw compiler
    patch_create_aot_dispatcher_function(graph_id, z3_partition, ...)
    return torch._inductor.compile(gm, real_inputs)
```

---

## 5. partitioner.py — ZeRO-3参数recompute策略

源码: `deepspeed/compile/partitioner.py` (100行)

### 5.1 问题: min-cut partitioner会保存所有activation

AOTAutograd的`min_cut_rematerialization_partition`将joint graph分成fw/bw → min-cut算法决定哪些activation在fw计算后保存给bw → 最小化总内存

**但在ZeRO-3场景下**: partitioner不知道参数会被gather/release → 它会保存所有"参数aliasing"的activation → 但gathered参数在fw用完后会被release → bw需要时重新gather → 这些activation不需要保存!

### 5.2 解决方案: _recompute_param_aliases

```python
def _recompute_param_aliases(joint_graph, param_indices):
    # 标记aliasing/downcasting参数的节点为MUST_RECOMPUTE
    primal_inputs = list(filter(_is_primal, joint_graph.nodes))
    ds_param_inputs = set([primal_inputs[arg_idx] for arg_idx, _, _ in param_indices])

    for node in joint_graph.nodes:
        if need_recompute(node) and \
            any([(isinstance(a, Node) and (a in ds_param_inputs or a in recomputed_nodes))
                 for a in node.args]):
            # ★ 标记为MUST_RECOMPUTE → bw时重新计算而非保存
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            recomputed_nodes.add(node)
        else:
            node.meta.setdefault("recompute", CheckpointPolicy.MUST_SAVE)
```

**need_recompute判断**: no_copy_ops(如view/copy等alias操作) + type cast → 这些节点alias参数 → 如果保存给bw → 等于保存了整个gathered参数 → 内存浪费 → 标记MUST_RECOMPUTE → bw时重新gather + 计算

**效果**: 让min-cut partitioner知道这些节点应该recompute → 不保存 → fw结束后参数可以被release → bw需要时重新gather → 内存大幅节省

---

## 6. Compiler Passes详解 (9个优化pass)

### 6.1 Pass总览

| # | Pass名 | 源码文件 | 功能 | ZeRO stage |
|---|--------|----------|------|------------|
| 1 | add_z3_gather_release | `passes/zero3_compile.py` | ZeRO-3核心 → FX graph中插入allgather/release/reduce_grad | Z3 |
| 2 | schedule_prefetch | `passes/prefetch.py` | Profile-guided proactive prefetch → 提前启动allgather | Z3 |
| 3 | selective_gather | `passes/selective_gather.py` | 选择性unsharding → 根据内存预算让参数驻留 | Z3 |
| 4 | offload_adam_states | `passes/offload_adam_states.py` | Adaptive optimizer state offloading | Z3 |
| 5 | add_z1_reduce | `passes/zero1_compile.py` | ZeRO-1 ReduceScatter插入 | Z1 |
| 6 | add_z2_reduce | `passes/zero1_compile.py` | ZeRO-2 ReduceScatter插入 | Z2 |
| 7 | sp_compile | `passes/sp_compile.py` | AutoSP → Sequence Parallelism AllGather/RS | SP |
| 8 | offload_parameter_fwd | `passes/offload_parameters.py` | 参数offloading | Z3 |
| 9 | free_activation | `compile/fx.py` | 大activation eager释放 | all |

**默认ZeRO-3 schedule**:
```
Step 0: [add_z3_gather_release] → baseline profiling
Step 5 (WARMUP): [add_z3_gather_release, schedule_prefetch, selective_gather] → profile-guided优化
```

---

### 6.2 Pass 1: add_z3_gather_release (ZeRO-3核心)

源码: `deepspeed/compile/passes/zero3_compile.py` (236行)

#### 6.2.1 Forward pass: add_gather_and_release

```python
def add_gather_and_release(graph_id, graph, param_manager, param_nodes):
    for pn in param_nodes:  # ← 每个ZeRO-3参数节点
        # ★ Type cast fusion: 如果参数的唯一user是type cast(如BF16→FP16)
        # → fuse cast with allgather → 直接gather到target dtype
        fuse_typecast = False
        if len([user for user in pn.users if user.op != "output"]) == 1:
            typecast_node = next(iter(pn.users))
            is_cast, casted_dtype = is_cast_op(typecast_node)
            if is_cast and casted_dtype.itemsize < target_dtype.itemsize:
                fuse_typecast = True
                target_dtype = casted_dtype

        # 1. 插入allgather → 从分区参数gather到全尺寸
        add_allgather(graph_id, graph, pn, ds_id, target_dtype)

        # 2. 插入wait_allgather → 等待NCCL通信完成
        # (allgather是异步的 → wait确保数据可用)

        # 3. 插入release → 在参数的最后一个user之后释放
        for user in users:
            add_release(graph_id, graph, user, pn, ds_id, len(users))
```

**FX graph中插入的custom ops**:
```
param_node → allgather_param(graph_id, ds_id, dtype) → wait_allgather(graph_id, ds_id) → [计算op] → release_param(graph_id, ds_id, n_users)
```

**type cast fusion优化**: 如果参数BF16 → 只有一个user是`convert_element_type(BF16→FP16)` → 直接allgather到FP16 → 通信量减半 → 节省一半AllGather带宽

#### 6.2.2 Backward pass: add_gather_and_reduce

```python
def add_gather_and_reduce(graph_id, graph, param_manager, param_nodes_bw, param_name_to_grad):
    # backward中也需要gather参数(用于梯度计算)
    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    # ★ 为每个参数的梯度插入ReduceScatter
    for param_name in param_manager.param_names:
        if param_name_to_grad[param_name] is not None:
            add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, ds_id)
```

**backward FX graph**:
```
param_node → allgather → wait → [backward计算] → release → ... → grad → reduce_grad(graph_id, ds_id)
```

#### 6.2.3 Profiling + Scheduling

```python
def add_z3_gather_release_fw(gm, ...):
    # 执行add_gather_and_release
    gm.graph = add_gather_and_release(...)

    # ★ 注册graph到C++ executor → 建立通信基础设施
    nz3.register_graph_z3(graph_id, [v[1] for v in param_indices])

    # ★ ProfilingInterpreter → 运行graph收集内存/时间数据
    profiler = ProfilingInterpreter(gm, debug_log=debug_log)
    profiler.run(*real_inputs)

    # ★ 清空ds_param的meta["val"] → 防止Inductor验证参数shape
    for n in gm.graph.nodes:
        if is_ds_param:
            n.meta["val"] = torch.empty([0], dtype=..., device=...)

    # ★ fast_free_schedule → 内存感知的op重排序
    gm.graph = fast_free_schedule(gm.graph, available_memory(), ...)
```

---

### 6.3 Pass 2: schedule_prefetch (Proactive Prefetching)

源码: `deepspeed/compile/passes/prefetch.py` (176行)

#### 6.3.1 算法: 反向遍历 + 通信计算overlap

```
核心思路: 从graph末尾反向遍历 → 遇到allgather时 → 判断是否可以提前prefetch
          → 前面有足够的计算时间 → 提前启动allgather → overlap通信和计算
```

```python
def schedule_prefetch(gm, graph_id, ...):
    # 1. 获取内存限制(跨rank取最小 → 保证所有rank不OOM)
    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)

    # 2. 反向遍历graph → 寻找prefetch机会
    order_rev = list(reversed(graph.nodes))
    for i, node in enumerate(order_rev):
        if node.target == torch.ops.dc.allgather_param.default:
            # ★ 判断fuse: 当前prefetch组 + 新allgather → 是否比分开更快?
            pred_time_current = comm_predictor(current_ag_size)
            pred_time_next = comm_predictor(tensor_size_dict[node.name])
            pred_time_fused = comm_predictor(current_ag_size + tensor_size_dict[node.name])

            do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused \
                      and (current_ag_size + ...) < MAX_FUSE_SIZE

        # ★ 内存检查: prefetch组总大小不能超过可用内存
        while next_peak + ag_tensor_size_sum > max_mem:
            # 提前launch已积累的prefetch组
            fused_ag_nodes = prefetch_ag_groups.pop(0)
```

#### 6.3.2 Prefetch Fusing

```python
    # 将多个allgather fuse成一个prefetch_params_fused
    for ag_group in prefetch_ag_groups:
        param_nodes = [ag_node.args[0] for ag_node in node]
        ds_ids = [get_ds_id(ag_node) for ag_node in node]
        new_graph.call_function(torch.ops.dc.prefetch_params_fused.default,
                                args=(graph_id, param_nodes_copy, ds_ids))
```

**C++侧prefetch_params_fused**: `ncclGroupStart()` → 多个AllGather batch在一起 → `ncclGroupEnd()` → 减少NCCL launch overhead → 一次通信启动多个allgather

**comm_predictor**: 使用profile数据建立通信时间预测模型 → 根据传输大小预测通信延迟 → 用于判断是否值得fuse

---

### 6.4 Pass 3: selective_gather (选择性Unsharding) ★★★

源码: `deepspeed/compile/passes/selective_gather.py` (213行)

**这是DeepCompile最有价值的优化**: 在内存允许时, 让某些参数永久驻留GPU → 不需要反复allgather/release

#### 6.4.1 核心算法

```python
def selective_gather(gm, graph_id, ...):
    # ★ 只在最后一个backward graph上执行(收集了所有profiling数据后)
    if not bwd:
        return gm
    if graph_id != last_backward_graph_id:
        return gm

    # 1. 计算persistence budget → 可用内存中多少可以给persistent参数
    budget = _compute_persistence_budget(all_graph_mem_records, total_mem, MEM_MARGIN)

    # 2. 收集所有allgather的profiling数据 → 每个参数的通信时间
    for n in profile.fwd_graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default:
            ds_id_to_time[n.args[2]] += n.meta["device_time"]

    # 3. ★★★ 关键排序: time_per_size ratio ★★★
    # 通信时间 / 参数大小 → 单位内存节省的通信时间 → 高ratio = 高性价比
    ds_ids = [ds_id for ds_id in ds_id_to_size if ds_id not in persistent_ds_ids]
    ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    # 4. 跨rank取最小内存 → 保证所有rank不OOM
    vals_to_bcast = torch.tensor([total_mem, current_available_mem], device=...)
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)

    # 5. ★★★ Greedy选择: 按ratio从高到低 → 直到内存预算耗尽 ★★★
    available_mem = int(current_available_mem * (1 - MEM_MARGIN))
    persistent_mem = 0
    for ds_id in ds_ids:
        size = ds_id_to_size[ds_id]
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size
        # ★ C++侧set_persistent → 立即gather并注册为persistent
        nz3.set_persistent(ds_id)
```

#### 6.4.2 Persistence Budget Calculation

```python
def _compute_persistence_budget(all_graph_mem_records, total_mem, mem_margin):
    usable_mem = int(total_mem * (1 - mem_margin))  # ← 90% of total GPU mem

    peak_resident_alloc = max(record[1] for ...)  # ← forward/backward峰值内存
    transient_peak = max(record[3] for ...)        # ← transient峰值(包含临时allocation)

    return {
        "usable_mem": usable_mem,
        "peak_resident_alloc": peak_resident_alloc,
        "transient_peak": transient_peak,
        "available_mem": max(0, usable_mem - peak_resident_alloc),  # ← 可以给persistent参数的内存
    }
```

**关键**: `available_mem = usable_mem - peak_resident_alloc` → peak_resident_alloc是forward/backward期间的最大驻留内存 → 差值就是可以安全分配给persistent参数的额外空间 → 不影响计算时的内存需求

#### 6.4.3 为什么selective_gather对gradient accumulation特别有效?

**blog原文**: "The benefit becomes more pronounced at higher accumulation steps, where the reduced frequency of parameter updates makes selective unsharding more effective."

解释: gradient accumulation时 → 一个optimizer step包含多个micro-batch的forward/backward → 每个micro-batch都需要gather相同参数 → 如果参数persistent → 省掉所有micro-batch的重复allgather → gradient_accumulation_steps=4 → 省掉3/4的allgather

**数据**: Mixtral 8x7B → DeepCompile selective unsharding → **1.54x** vs ZeRO-3 baseline

---

### 6.5 Pass 4: offload_adam_states (Adaptive Optimizer Offloading)

源码: `deepspeed/compile/passes/offload_adam_states.py` (550行)

#### 6.5.1 vs 传统offloading

| 特性 | 传统ZeRO-3 offload | DeepCompile adaptive offload |
|------|-------------------|------------------------------|
| Offload对象 | optimizer computation + memory | **只offload memory** (momentum/variance/master weights) |
| 传输时机 | 同步blocking | **异步overlap** → copy_stream + events |
| 选择策略 | 全量offload | **Profile-guided** → 只offload超出内存限制的部分 |
| 计算位置 | CPU | **GPU** → optimizer computation留在GPU |

#### 6.5.2 核心流程

```python
# Forward开始时: offload所有Adam states到CPU
offload_adam_states_sync()  # → move_key(state, "exp_avg") + move_key(state, "exp_avg_sq")
                            # → copy_stream异步传输 → offload_event记录

# Forward过程中: 根据内存压力 → 决定何时offload
while total_mem - peak_mem[node.name] - optim_size < 0:
    task = offload_tasks.pop(0)
    # 插入offload_sync节点 → 等待offload完成 → 删除GPU上的state → 释放内存

# Backward过程中: 根据内存空间 → 决定何时reload
while total_mem > peak_mem[node.name] + total_reload_mem + next_reload_mem:
    # 插入reload节点 → 从CPUreload回GPU → overlap with backward computation
```

**效果**: Llama-3 70B on 16xH100(半数GPU) → **7x** vs ZeRO-3+Eager → 因为传统offload把optimizer computation也搬到CPU → 极慢 → DeepCompile只offload memory → computation留在GPU → 异步overlap

---

## 7. C++底层: 通信内核详解

源码: `csrc/compile/z3.cpp` (680行) + `csrc/compile/deepcompile.cpp` (196行)

### 7.1 Z3CustomOpExecutor — 5个CUDA Stream

```cpp
class Z3CustomOpExecutor : public CustomOpExecutor {
private:
    at::cuda::CUDAStream ag_stream_;      // ★ AllGather专用stream → 通信overlap
    at::cuda::CUDAStream rs_stream_;      // ReduceScatter专用stream
    at::cuda::CUDAStream copy_stream_;    // CPU-GPU copy stream
    at::cuda::CUDAStream offload_stream_; // Adam state offload stream
    at::cuda::CUDAStream reload_stream_;  // Adam state reload stream

    // ★ AllGather event pair → 双event保证正确overlap
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comp_done_events_;  // 计算完成
    std::unordered_map<long, std::shared_ptr<at::cuda::CUDAEvent>> ag_comm_done_events_;  // 通信完成
};
```

**5 stream设计**: 计算在default stream → AllGather在ag_stream → ReduceScatter在rs_stream → offload/reload各自独立 → 5个stream并行 → 最大overlap

### 7.2 allgatherParam — NCCL AllGather实现

```cpp
at::Tensor allgatherParam(long ds_id, std::optional<at::ScalarType> dtype,
                           c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm_mem) {
    const DSParam& param = param_registry_->getParam(ds_id);
    const at::Tensor& ds_tensor = param.getDSTensor();  // ← 分区参数
    const int world_size = process_group_->getSize();
    const int64_t true_numel = productDim(param.getShape());  // ← 全尺寸
    const int64_t padded_per_rank = (true_numel + world_size - 1) / world_size;

    // ★ Persistent参数 → 直接返回已gather的buffer → 省掉AllGather通信!
    if (param_registry_->isValid(ds_id)) {
        auto base = param_registry_->getGatheredParam(ds_id);
        return base.flatten().to(target_dtype).index({Slice(0, true_numel)}).view(param.getShape());
    }

    // 分配output buffer → padded_numel = world_size * padded_per_rank
    output_buf = torch::empty({padded_numel}, ds_tensor.options().dtype(target_dtype));

    // ★ 双event机制 → 确保计算完成后再启动AllGather
    ag_comp_done_events_[ds_id]->record();     // ← default stream记录计算完成
    ag_comp_done_events_[ds_id]->block(ag_stream_);  // ← ag_stream等待计算完成

    // ★ NCCL AllGather → 直接调用ncclAllGather → bypass ProcessGroup overhead
    launchAllGather(output_buf, ds_id, symm_mem);

    // ★ 通信完成event → wait_allgather时检查
    ag_comm_done_events_[ds_id]->record(ag_stream_);

    // 返回view → slice到true_numel → reshape到ds_shape
    return output_buf.flatten().index({Slice(0, true_numel)}).view(param.getShape());
}
```

### 7.3 launchAllGather — NCCL vs Symmetric Memory

```cpp
void launchAllGather(at::Tensor output_buf, long ds_id, ...) {
    if (symm_mem == nullptr) {
        // ★ Fast path: NCCL AllGather → uniform shard size → 直接NCCL API
        ncclResult_t result = ncclAllGather(
            ds_tensor.contiguous().data_ptr(),  // ← send: 本rank的分区
            output_buf.data_ptr(),               // ← recv: 全尺寸output
            shard_elems,                          // ← 每个rank发送的元素数
            get_nccl_data_type(ds_tensor.scalar_type()),
            nccl_comm_,                           // ← 直接NCCL communicator
            ag_stream_);                          // ← 专用stream

    } else {
        // ★ Symmetric Memory path: Mori-style P2P → Hopper NVLink only
        at::Tensor local_buf = symm_mem->get_buffer(rank, ...);
        local_buf.copy_(ds_tensor, true);
        symm_mem->barrier(0, TIMEOUT);
        // Ring-style copy from remote ranks
        for (int step = 0; step < world_size; step++) {
            int remote_rank = (rank - step + world_size) % world_size;
            auto src_buf = symm_mem->get_buffer(remote_rank, ...);
            chunks[remote_rank].copy_(src_buf.flatten(), true);
        }
        symm_mem->barrier(0, TIMEOUT);
    }
}
```

**关键**: DeepCompile直接使用`ncclAllGather` API而非`ProcessGroup::allgather` → bypass PyTorch的ProcessGroup overhead → 更快 → 且在ag_stream上执行 → 和default stream的计算overlap

### 7.4 releaseParam — 智能内存释放

```cpp
void releaseParam(long ds_id, long n_users) {
    param_use_count_[ds_id]--;
    if (param_use_count_[ds_id] == 0 && !param.isPersistent()) {
        // ★ 只有当所有user都完成时才释放 → n_users追踪use count
        at::Tensor gathered_param = param_registry_->getGatheredParam(ds_id);

        // ★ resize_bytes_cuda(storage, 0) → 立即释放内存 → 不是等GC
        at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), 0);

        // ★ record_stream → 防止caching allocator过早reuse
        gathered_param.record_stream(at::cuda::getCurrentCUDAStream());

        param_registry_->unregisterGatheredParam(ds_id);
    }
}
```

**vs 传统ZeRO-3**: 传统方式通过Python hook释放 → DeepCompile在FX graph中精确插入release → **在最后一个user之后立即释放** → 内存利用率最优

### 7.5 flushReduceBucket — 双buffer ReduceScatter

```cpp
void flushReduceBucket(at::ScalarType scalar_type) {
    // ★ ncclGroupStart → 多个ReduceScatter batch在一起
    ncclGroupStart();
    for (const ReduceTask& t : reduce_tasks_.at(scalar_type)) {
        ncclReduceScatter(t.getSendBuf().data_ptr(),
                          recv_buf.data_ptr(),
                          recv_numel,
                          get_nccl_data_type(scalar_type),
                          getReductionOp(),
                          nccl_comm_,
                          rs_stream_);
    }
    ncclGroupEnd();

    // ★ DoubleBufferedReduceBucket → 交替使用两个buffer → overlap reduction和gradient accumulation
}
```

### 7.6 init — 独立NCCL Communicator

```cpp
void init(c10::intrusive_ptr<c10d::ProcessGroup> pg, pybind11::object& config, ...) {
    process_group = pg;

    // ★ 创建独立的NCCL communicator → 不和ProcessGroup共享 → 专用stream
    ncclUniqueId ncclID;
    ncclGetUniqueId(&ncclID);
    // broadcast NCCL ID → ncclCommInitRank → 独立communicator
    ncclCommInitRank(&nccl_comm, process_group->getSize(), ncclID, process_group->getRank());

    param_registry = std::make_shared<DSParamRegistry>();
    reduce_buckets = std::make_shared<DoubleBufferedReduceBucket>(
        initial_reduce_bucket_size, get_config<bool>(config, "double_buffer"));
}
```

**为什么独立NCCL communicator?**: ProcessGroup的通信在default stream → DeepCompile的AllGather在ag_stream → ReduceScatter在rs_stream → 需要独立communicator才能在不同stream上并发执行 → 否则NCCL会串行化

---

## 8. list_schedule.py — fast_free_schedule (内存感知调度)

源码: `deepspeed/compile/list_schedule.py` (442行)

### 8.1 算法: AllGather-aware scheduling

```python
def fast_free_schedule(graph, available_mem, ...):
    # 1. 识别所有allgather节点 → 这是"内存增长点"
    unscheduled_ags = [n for n in unscheduled if n.target == torch.ops.dc.allgather_param.default]

    # 2. 为每个allgather计算"free path" → 从allgather到release需要多少内存
    for ag_node in unscheduled_ags:
        last_use = node_to_last_use[ag_node]
        required_nodes = get_node_requirements(last_use, scheduled)
        ag_nodes_in_path[ag_node] = set(n for n in required_nodes
                                         if n.target == torch.ops.dc.allgather_param.default)

    # 3. ★★★ Greedy scheduling: 优先选择"free_acc_mem == 0"的allgather ★★★
    # → 这种allgather不需要额外的allgather就能释放 → 最小化峰值内存
    while len(unscheduled_ags) > 0:
        runnable_ags = [...]  # ← 所有依赖已满足的allgather
        ags_with_no_additional_ag = [ag for ag in runnable_ags if ag.free_acc_mem == 0]
        if len(ags_with_no_additional_ag) > 0:
            # ★ 选择free_path最短的 → 最快释放gathered参数
            sorted_ags = sorted(ags_with_no_additional_ag, key=_free_path_allgather_key)
            next_ag = sorted_ags[0]
            nodes_to_schedule = next_ag.schedule_until_free  # ← 一次性schedule到释放
        else:
            # fallback: 选择free_acc_mem最小的
            sorted_ags = sorted(runnable_ags, key=_fallback_allgather_key)
            next_ag = sorted_ags[0]
            nodes_to_schedule = next_ag.schedule_until_ag

        # 4. 同时尝试提前schedule reduce节点 → 更早释放梯度内存
        for reduce_node in reduce_nodes:
            if all its prerequisite allgathers are scheduled:
                reduces_to_schedule.append(reduce_node)
```

**核心洞察**: 传统ZeRO-3按module层级gather/release → 参数驻留时间长 → 内存峰值高 → DeepCompile在FX graph粒度调度 → 可以选择最优的allgather顺序 → 最小化同时驻留的gathered参数数量

---

## 9. inductor.py — Inductor集成

源码: `deepspeed/compile/inductor.py` (250行)

### 9.1 Custom Op注册 — 让Inductor理解DeepCompile ops

```python
def register_custom_ops():
    # 所有torch.ops.dc.* 操作注册为Inductor fallback kernel
    register_fallback_no_reuse(torch.ops.dc.allgather_param.default,
                               never_reuse_input=False, never_reuse_output=True)  # ← output不复用
    register_fallback_no_reuse(torch.ops.dc.wait_allgather.default,
                               never_reuse_input=True, never_reuse_output=True)
    register_fallback_no_reuse(torch.ops.dc.release_param.default,
                               never_reuse_input=True, never_reuse_output=False)  # ← input不复用
    register_fallback_no_reuse(torch.ops.dc.reduce_grad.default,
                               never_reuse_input=True, never_reuse_output=True,
                               force_free_input=True)  # ← ★ free input after execution
    register_fallback_no_reuse(torch.ops.dc.free_tensors.default,
                               never_reuse_input=True, never_reuse_output=True)

    # ★ 禁用Inductor的dead_node_elimination → DeepCompile自己管理内存释放
    Scheduler.dead_node_elimination = lambda _: None
```

**关键**:
- `never_reuse_output=True` → Inductor不会把allgather的output buffer复用给其他op → 避免数据冲突
- `force_free_input=True` → reduce_grad执行后立即free input → 释放梯度内存
- `Scheduler.dead_node_elimination = lambda _: None` → 禁用Inductor的自动死节点消除 → DeepCompile的release_param精确控制释放时机

### 9.2 Patch AOTAutograd — 注入DeepCompile的fw/bw compiler

```python
def patch_create_aot_dispatcher_function(graph_id, z3_partition, make_fw_graph, make_bw_graph, ...):
    # ★ Patch AotAutograd.__init__ → 注入DeepCompile的fw/bw compiler
    original_init = AotAutograd.__init__

    @functools.wraps(original_init)
    def patched_init(self, **kwargs):
        _patch_deepcompile_aot_kwargs(kwargs,
                                       graph_id=graph_id, z3_partition=z3_partition,
                                       make_fw_graph=make_fw_graph, make_bw_graph=make_bw_graph, ...)
        original_init(self, **kwargs)

    AotAutograd.__init__ = patched_init
```

**为什么需要patch?**: Inductor内部使用`AotAutograd` → DeepCompile需要将自己的fw/bw compiler注入到AotAutograd的kwargs中 → 否则Inductor会使用默认的compiler → 不执行DeepCompile的pass

### 9.3 ZeRO-3参数shape patching — 防止Inductor size validation

```python
def patch_compiler(original_compiler, dc_compiler, z3_partition, graph_id, ...):
    def wrapped_compiler(gm, fake_inputs):
        mod_graph = dc_compiler(gm, fake_inputs)

        if z3_partition:
            # ★ 将ZeRO-3参数的fake input替换为empty(0) → 防止Inductor验证shape
            for in_node, in_v in zip(input_nodes, fake_inputs):
                if in_node.name in param_names:
                    patched_inputs.append(
                        to_fake_tensor(torch.empty([0], dtype=in_v.dtype, ...), in_v.fake_mode))
                else:
                    patched_inputs.append(in_v)
```

**问题**: Inductor验证输入shape → ZeRO-3参数从分区gather到全尺寸 → shape变化 → validation失败 → 用empty(0)代替 → 绕过validation

---

## 10. patch_compiled_func.py — Backward输入捕获

源码: `deepspeed/compile/patch_compiled_func.py` (94行)

### 10.1 问题: backward的真实输入

DeepCompile的bw compiler需要真实的backward输入(梯度值) → 但AOTAutograd的bw_compiler在编译时只收到fake tensors → 需要捕获运行时的真实backward输入

### 10.2 解决方案: Patch autograd.Function metaclass

```python
# PyTorch >= 2.7: patch _backward_impl
class FunctionMeta(base_meta):
    def __new__(cls, name, bases, dct):
        if name == "CompiledFunction":
            original_backward_impl = dct.get("_backward_impl")

            def wrapped_backward_impl(ctx, all_args):
                if enabled_patched_func:
                    backward_inputs.append(all_args)  # ← ★ 捕获真实backward输入
                    wrapped_backward_impl.owner_class.compiled_bw = None  # ← 清除compiled bw → 下一step重新编译
                return original_backward_impl(ctx, all_args)
            dct["_backward_impl"] = staticmethod(wrapped_backward_impl)

# PyTorch 2.6: patch _backward_prologue (API不同)
class FunctionMeta(base_meta):
    def __new__(cls, name, bases, dct):
        if name == "CompiledFunction":
            original_backward_prologue = dct.get("_backward_prologue")
            def wrapped_backward_prologue(ctx, *grad_outputs):
                all_args = original_backward_prologue(ctx, *grad_outputs)
                if enabled_patched_func:
                    backward_inputs.append(all_args)
                return all_args
```

**关键**: `compiled_bw = None` → 每次backward执行后清除compiled backward → 下一步重新编译 → 因为pass schedule可能改变 → 需要重新trace和编译

---

## 11. DeepCompile vs FSDP2 + torch.compile

### 11.1 FSDP2的compile兼容方案

FSDP2用**intentional graph breaks**解决compile兼容:

```python
# FSDP2: _dynamo_disable → 在FSDP hooks处故意graph break
@torch._dynamo.disable
def _pre_forward_hook(...):
    # unshard parameters → graph break → 后续是新的编译段
```

**问题**: graph break → 多段编译 → 每段独立优化 → 全局优化机会丧失 → 通信和计算无法跨段overlap

### 11.2 DeepCompile的方案

**无graph break**: DeepCompile移除所有hooks → Dynamo tracing完整graph → 一个大FX graph → compiler passes在graph中插入通信op → 全局优化 → 通信和计算overlap

| 特性 | FSDP2 + compile | DeepCompile + ZeRO-3 |
|------|-----------------|----------------------|
| Graph完整性 | 多段(graph breaks at hooks) | **完整graph** |
| 通信插入 | hook-based(eager) | **compiler pass-based** |
| 通信调度 | 按hook触发(固定顺序) | **profile-guided** (可重排序) |
| 内存优化 | reshard_after_forward控制 | **selective_gather + fast_free_schedule** |
| Prefetch | 有限(prefetch only next module) | **proactive prefetch** (多参数fused) |
| Optimizer offload | 无 | **adaptive profile-guided offload** |

---

## 12. PyTorch兼容性状态

### 12.1 PR追踪

| PR | Title | Author | 状态 | 日期 | 内容 |
|----|-------|--------|------|------|------|
| #7951 | Fix DeepCompile+Z3 on PT v2.9/2.10 | tohtana (Masahiro Tanaka) | **MERGED** | 2026-04-11 | patch_fake_tensor.py的guard_sizes_strides patch → PyTorch 2.9开始严格执行parameter tensor-match guards → 需要让guard匹配released(empty(0))状态而非gathered(full)状态 |
| #8024 | Fix DeepCompile AOT kwargs patching for PT >= v2.11 | tohtana | **MERGED** | 2026-05-25 | inductor.py的patch_create_aot_dispatcher_function → PyTorch 2.11改变了AotAutograd的__init__ API → kwargs patching需要适配 |
| #8059 | [DeepCompile] fix gather params in dynamo skipped frames for ZeRO3 | XAheli (Aheli Poddar) | **OPEN** | 2026-06-11 | Dynamo可能skip某些frame → 被skip的frame中的参数没有被gather → 导致运行时参数缺失 |
| #8032 | Fix DeepCompile ZeRO-3 release parameter lifetime | — | — | — | release_param的use count计算修复 |
| #8033 | Fix DeepCompile all-gather scheduler candidate selection | — | — | — | fast_free_schedule的候选选择修复 |
| #8036 | Fix DeepCompile ZeRO-1 grad target lifetime | — | — | — | ZeRO-1梯度生命周期修复 |
| #8038 | Normalize DeepCompile ZeRO-3 grad dtype before reduction | — | — | — | 梯度dtype标准化 |

### 12.2 版本兼容矩阵

| PyTorch版本 | DeepCompile ZeRO-3 | DeepCompile ZeRO-1 | 备注 |
|-------------|--------------------|--------------------|------|
| 2.6 | ✅ (初始发布版本) | ✅ | `_backward_prologue` API |
| 2.7-2.8 | ✅ | ✅ | `_backward_impl` API变更 |
| 2.9-2.10 | ✅ (PR#7951 merged) | ✅ | guard_sizes_strides强制执行 |
| 2.11+ | ✅ (PR#8024 merged) | ✅ | AotAutograd kwargs API变更 |
| 2.12 | ⚠️ (PR#8059 open → 可能有问题) | ✅ | Dynamo skipped frames |

**is_deepcompile_supported()**: `required_torch_version(min_version=2.6) and get_accelerator().device_name() == "cuda"` → 只支持CUDA + PyTorch >= 2.6

---

## 13. 配置参考

源码: `deepspeed/compile/config.py` (60行)

```json
{
  "compile": {
    "deepcompile": true,            // ★ 开启DeepCompile
    "free_activation": false,       // 释放大activation (>10MB阈值)
    "free_activation_threshold": 10485760,  // 10MB
    "offload_activation": false,    // activation offloading
    "offload_opt_states": false,    // optimizer states offloading
    "double_buffer": true,          // ★ ReduceScatter双buffer(默认开启)
    "symmetric_memory": false,      // Mori symmetric memory(需Hopper NVLink)
    "debug_log": false,             // 调试日志
    "offload_parameters": false,    // 参数offloading
    "passes": ["z3"]                // ★ 指定pass → "z1"/"z3"/"autosp"
  }
}
```

**使用方式**:
```python
# DeepSpeed初始化后调用compile()
engine, _, _, _ = deepspeed.initialize(...)
engine.compile(backend="inductor")  # ← "inductor"(推荐) or "eager"
```

---

## 14. RTX 4090分析

### 14.1 DeepCompile + ZeRO-3 + RTX 4090 → ❌ 不可行

**根本原因不变**: ZeRO-3的通信瓶颈在AllGather → RTX 4090是PCIe → AllGather带宽受PCIe限制:

```
ZeRO-3 AllGather: 每层参数forward前AllGather → backward前AllGather
  7B模型 BF16: 14GB参数 → forward: 14GB AllGather → backward: 14GB AllGather → 总28GB
  PCIe带宽: ~12GB/s(bidirectional) → 28GB/12GB/s ≈ 2.3s → 极慢
```

DeepCompile的优化(prefetch, selective_gather)能减少AllGather次数 → 但AllGather本身仍需要 → PCIe带宽瓶颈不变

### 14.2 DeepCompile + ZeRO-2 + RTX 4090 → ✅ 可行

ZeRO-2只分区optimizer states → 参数和梯度全replicate → 通信只有ReduceScatter(梯度):

```
ZeRO-2 ReduceScatter: 每step一次梯度reduction
  7B BF16: 14GB梯度 → ReduceScatter → 14GB/(8 GPUs) = 1.75GB per GPU → 但RTX 4090单GPU...
```

**RTX 4090单GPU**: ZeRO-2没有意义 → 单GPU不分区 → DeepCompile的ZeRO-1/2 pass不适用

### 14.3 DeepCompile + LoRA + ZeRO-3 + RTX 4090 → ❌ 仍不可行

LoRA减少可训练参数 → 但ZeRO-3分区的是**所有参数**(包括冻结的base model) → LoRA只减少optimizer states → 不减少AllGather通信量 → PCIe瓶颈不变

### 14.4 DeepCompile + LoRA + ZeRO-2 + RTX 4090 → ✅ 但意义有限

ZeRO-2 + LoRA → optimizer states分区 → 单GPU → 无意义

### 14.5 RTX 4090唯一可行的DeepCompile场景

**RTX 4090 8卡 PCIe集群**: DeepCompile + ZeRO-1 → ReduceScatter梯度 → 但8卡PCIe scaling灾难(之前分析0.46x) → 不推荐

**结论**: DeepCompile是**多GPU NVLink集群的优化** → RTX 4090 PCIe集群 → 通信瓶颈在任何ZeRO配置下都存在 → DeepCompile的优化无法弥补PCIe带宽限制

| 场景 | RTX 4090可行性 | 原因 |
|------|---------------|------|
| DeepCompile + ZeRO-3 + 单GPU | ❌ | ZeRO-3需要多GPU |
| DeepCompile + ZeRO-3 + 多GPU PCIe | ❌ | AllGather PCIe瓶颈 |
| DeepCompile + ZeRO-2 + 单GPU | ❌ | ZeRO-2单GPU无意义 |
| DeepCompile + ZeRO-1 + 多GPU PCIe | ⚠️ | 仍受PCIe影响 |
| DeepCompile + ZeRO-3 + H100 NVLink | ✅✅✅ | **最佳场景** |

### 14.6 DeepCompile的RTX 4090学习价值

虽然RTX 4090无法利用DeepCompile的性能优化 → 但**学习价值极高**:

1. **ZeRO-3 + torch.compile兼容性**: 这是7框架中唯一解决此问题的方案 → FSDP2用intentional graph break(妥协方案) → DeepCompile用freeze+pass(根本方案)
2. **Compiler-level分布式优化**: 从hook-based → pass-based → 从运行时 → 编译时 → 这是分布式训练系统进化的方向
3. **Profile-guided优化**: selective_gather/prefetch的算法设计 → 内存预算计算 → greedy选择 → 可以迁移到其他系统
4. **C++通信内核**: 5-stream设计 → 双event机制 → 独立NCCL communicator → ncclGroupStart/End → 这些是高性能分布式系统的通用技术

---

## 15. 源码文件结构总览

```
deepspeed/compile/                    # Python编译框架
├── __init__.py                       # 模块入口 → pad_tensors
├── config.py                         # CompileConfig → deepcompile/free_activation/...
├── constants.py                      # AUTOSP_INPUT_ID_KEY等常量
├── backend.py                        # ★ 核心框架: make_backend/launch_compile_passes/init_schedule
│                                     #   GraphOrder + backend_fn + make_fw_graph/make_bw_graph
├── init_z3.py                        # ★ ZeRO-3初始化: unset hooks + register_z3_param + schedule
├── init_z1.py                        # ZeRO-1/2初始化: register_param + schedule
├── init_sp.py                        # AutoSP初始化
├── graph_param.py                    # DSGraphParamManager → 参数到graph的映射
├── partitioner.py                    # ★ _recompute_param_aliases + wrapped partitioner
├── fx.py                             # FX graph操作: add_postprocess/add_end_backward/...
├── inductor.py                       # ★ Inductor集成: register_custom_ops + patch AOT kwargs
├── list_schedule.py                  # ★ fast_free_schedule → 内存感知op调度
├── input_storage.py                  # InputStorage → 保存real_inputs元数据
├── patch_compiled_func.py            # ★ Backward输入捕获 → Patch autograd.Function metaclass
├── patch_fake_tensor.py              # ★★ Dynamo tracing兼容 → wrap_if_ds_param + guard_sizes_strides
├── util.py                           # 工具函数: get_deepcompile_handle/get_last_uses/...
├── profilers/                        # Profiling框架
│   ├── __init__.py                   # ProfilingResult
│   ├── comm_profile.py               # 通信时间预测器
│   └── graph_profile.py              # MemoryProfilingInterpreter
├── custom_ops/                       # AutoSP custom ops
│   ├── all_to_all.py                 # AllToAll
│   ├── sp_compat.py                  # SP兼容性
│   └── sp_dp_registry.py             # SP/DP registry
└── passes/                           # ★★★ Compiler优化passes
    ├── zero3_compile.py              # ★ ZeRO-3核心: add_z3_gather_release
    │                                 #   allgather_param + wait_allgather + release_param + reduce_grad
    ├── zero1_compile.py              # ZeRO-1/2: add_z1_reduce / add_z2_reduce
    ├── selective_gather.py           # ★★ 选择性unsharding: persistence budget + greedy选择
    ├── prefetch.py                   # ★ Proactive prefetching: 反向遍历 + fuse + overlap
    ├── offload_adam_states.py        # ★ Adaptive optimizer offloading: profile-guided
    ├── offload_parameters.py         # 参数offloading
    ├── offload_activation.py         # Activation offloading
    ├── long_context_checkpointing.py # 长上下文checkpointing
    └── sp_compile.py                 # AutoSP compile pass

csrc/compile/                         # ★★ C++通信内核
├── deepcompile.cpp                   # 全局状态: init/reset/cleanup/reduce_grad/free_tensors
│                                     #   DSParamRegistry + DoubleBufferedReduceBucket
│                                     #   独立NCCL communicator + symmetric memory
├── z3.cpp                            # ★★★ Z3CustomOpExecutor: 5 CUDA stream
│                                     #   allgatherParam → NCCL AllGather / Mori symmetric memory
│                                     #   releaseParam → resize_bytes_cuda(0) → 立即释放
│                                     #   flushReduceBucket → ncclGroupStart/End + 双buffer
│                                     #   offload/reload → copy_stream异步传输
│                                     #   prefetchParamsFused → fused AllGather
│                                     #   双event机制(ag_comp_done + ag_comm_done)
├── z3.h                              # C++声明: register_z3_param/allgather_param/set_persistent/...
├── z1.cpp                            # ZeRO-1 ReduceScatter内核
├── z1.h                              # ZeRO-1声明
├── z2.cpp                            # ZeRO-2内核
├── z2.h                              # ZeRO-2声明
├── util.cpp                          # C++工具函数
└── init.cpp                          # C++ pybind11注册 → torch.ops.dc.* namespace
```

---

## 16. 性能数据总结 (来自blog README)

| 模型 | 配置 | DeepCompile vs ZeRO-3+Eager | DeepCompile vs ZeRO-3+Compile |
|------|------|-----------------------------|-------------------------------|
| Llama-3-70B (32xH100) | ZeRO-3+prefetch+selective | **1.28x** | 更好(Compile有overhead) |
| Mixtral 8x7B (32xH100) | ZeRO-3+prefetch+selective | **1.54x** | 更好 |
| Llama-3-70B (16xH100 offload) | adaptive offload | **7x** | ~same as Eager |
| Llama-3-8B (8 GPU) | ZeRO-1 | **1.9x vs Eager, 2.5x vs Compile** | — |

**ZeRO-3+Compile为什么不好**: ZeRO-3的条件分支 → graph break → 多段编译 → 优化碎片化 → 实际overhead

**DeepCompile为什么好**: 无graph break → 全局优化 → profile-guided调度 → selective unsharding → prefetch overlap

---

## 17. 关键洞察总结

### 17.1 DeepCompile的核心创新

1. **Freeze Schedule**: 在Dynamo tracing前移除所有ZeRO-3 hooks → 从"运行时动态操作"变成"编译时静态插入" → 解决ZeRO-3 + torch.compile不兼容的根本问题
2. **patch_fake_tensor**: 让Dynamo看到full-size参数(正确FX graph) → 但guards匹配released状态(避免recompilation) → 最精妙的兼容patch
3. **_recompute_param_aliases**: 标记aliasing参数的节点为MUST_RECOMPUTE → 不保存给backward → fw结束后参数可release → bw时重新gather → 内存大幅节省
4. **selective_gather**: 基于time_per_size ratio的greedy选择 → 让高通信开销的参数驻留 → 省掉重复allgather → gradient accumulation时特别有效
5. **5-stream C++内核**: ag_stream + rs_stream + copy_stream + offload_stream + reload_stream → 5个CUDA stream并行 → 最大通信计算overlap
6. **独立NCCL communicator**: bypass ProcessGroup overhead → 在专用stream上直接调用NCCL API → 更快 → 更灵活的overlap

### 17.2 vs FSDP2的哲学差异

| 维度 | FSDP2 | DeepCompile |
|------|-------|-------------|
| 核心哲学 | 让compile理解FSDP(intentional breaks) | 让FSDP理解compile(freeze hooks) |
| Graph完整性 | 多段 | 完整 |
| 优化层级 | Python-level hooks | Compiler-level passes |
| 未来方向 | 逐渐减少breaks | 扩展pass集合 |

**我的判断**: DeepCompile的方向更正确 → compiler-level优化是终极形态 → FSDP2的intentional breaks是过渡方案 → 但FSDP2已广泛采用 → DeepCompile需要证明足够稳定才能被社区接受

### 17.3 对其他框架的启示

- **verl**: verl使用FSDP2 → intentional breaks → 如果verl转向DeepCompile思路 → 可以获得更好的compile优化
- **rLLM**: TinkerBackend in-process → 无分布式 → DeepCompile不适用 → 但selective_gather的算法思路可用于单GPU内存优化
- **Megatron**: Megatron的Inference Engine已经用compiler-level方式处理通信 → 和DeepCompile思路一致 → Megatron训练端如果也采用类似方法 → 可获得类似优化
