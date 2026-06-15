# PyTorch AOTAutograd 源码级深度阅读

> 2026-06-15 | 源码: pytorch/pytorch main branch
> 核心: AOTAutograd = Dynamo→FX graph 与 Inductor 之间的关键桥梁; 创建 joint fwd+bwd FX graph; min-cut partition 分离 fwd/bwd; 自动 gradient checkpointing 从 partition 决策中涌现
> 关键源文件: torch/_functorch/aot_autograd.py (1959行) + partitioners.py (4046行) + _aot_autograd/graph_capture_wrappers.py + _aot_autograd/graph_capture.py + _aot_autograd/graph_compile.py + _aot_autograd/functional_utils.py + _aot_autograd/runtime_wrappers.py + _aot_autograd/schemas.py

## 1. AOTAutograd 全景: 从 Dynamo FX Graph 到 Inductor 的桥梁

```
torch.compile 完整流水线:

Dynamo符号执行bytecode
  → FX Graph (aten ops)
  → ★ AOTAutograd ★ ← 本文核心
    → Stage 1: Graph Capture (functionalization + joint tracing)
    → Stage 2a: Partition (min-cut → fwd/bwd分离)
    → Stage 2b: Compile (fw_compiler + bw_compiler = Inductor)
    → Stage 2c: Make autograd.Function (CompiledFunction)
  → Inductor lowering→IR→Scheduler→Triton/C++ codegen
  → PyCodeCache → GuardedCode

AOTAutograd 3阶段:
  aot_stage1_graph_capture()  → 收集metadata + trace joint graph
  aot_stage2_compile()        → partition + compile fwd/bwd
  _aot_stage2c_make_autograd_function() → 包装成autograd.Function

★ AOTAutograd名字由来: "AOT" = Ahead-Of-Time
  → 传统autograd: 运行时动态构建backward graph
  → AOTAutograd: 编译时提前(trace)构建joint fwd+bwd graph, 然后partition, 然后compile
  → 结果: backward graph也被Inductor优化(融合+codegen), 不是eager逐op执行!
```

### 1.1 入口: create_aot_state → aot_stage1 → aot_stage2

```
aot_autograd.py: create_aot_state():
  → decompositions = aot_autograd_decompositions + user_decompositions
  → 如果config.functionalize_rng_ops: + rng_decompositions
  → fake_mode + PhiloxStateTracker + preserve_rng_state
  → run_functionalized_fw_and_collect_metadata(flat_fn, flat_args)
  → fw_metadata: ViewAndMutationMeta (所有mutation/alias信息)
  → needs_autograd = any(x.requires_grad for x in flat_args)
  → 如果outputs都不需要grad + inputs无grad mutation → inference path

返回AOTState(needs_autograd, flat_args, fw_metadata, aot_config, ...)

★ fw_metadata是AOTAutograd的灵魂:
  - input_info[i]: 每个输入的mutation/alias/requires_grad信息
  - output_info[i]: 每个输出的alias类型(non_alias/alias_of_input/alias_of_intermediate)
  - mutated_inp_runtime_indices: 哪些输入被mutation了
  - num_intermediate_bases: 输出alias中间tensor的数量
  - static_input_indices: 哪些输入是parameter/buffer(FSDP2关键!)
```

## 2. Joint Forward+Backward Graph 的精确追踪机制

### 2.1 追踪流程: 5层包装 → make_fx

```
graph_capture.py: aot_dispatch_autograd_graph():
  → fn_prepped_for_autograd(flat_fn, args_descs, meta, aot_config)
    → 将mutation转为extra outputs
    → clone requires_grad inputs(autograd需要pre-mutation版本)
    → 返回(fw_outs, out_grad_mask): 哪些outputs需要tangent

  → create_joint(fn_prepped_for_autograd, args_descs, aot_config)
    → ★ 关键: 创建joint(primals, tangents) → 运行forward + autograd.grad()

  → _prepare_graph_capture_tracing(fn_to_trace, flat_args, ...)
    → create_functionalized_fn(fn, args, ...)  ← functionalization包装
    → aot_dispatch_subclass(...)               ← subclass unpack/pack
    → handle_effect_tokens_fn(...)             ← side-effect token处理

  → _create_graph(fn_to_trace, flat_args, ...)
    → FunctionalTensorMode() + make_fx(inner_f)(*args)
    → ★ make_fx就是Dynamo的proxy_tensor tracer: 用FakeTensor执行,记录每个op到FX graph

★ 5层包装顺序(must be in this order!):
  1. FunctionalTensorMode → functionalization(in-place→out-of-place)
  2. fn_prepped_for_autograd → mutation→outputs + clone + grad_mask
  3. create_joint → forward + autograd.grad() → joint graph
  4. subclass unpack/pack → trace plain tensors, wrap at boundaries
  5. effect tokens → side-effect threading
```

### 2.2 create_joint: 精确的joint fwd+bwd tracing

```
graph_capture_wrappers.py: create_joint(fn, primals_descs, *, aot_config):

inner_fn(primals, tangents):
  → outs, tangent_mask = fn(*primals)  ← forward pass
  → 给每个FX node打tag:
      if _is_tangent(node): partitioner_tag = "is_backward"
      else: partitioner_tag = "is_forward"
    ★ 这个tag让partitioner知道: 哪些node是forward产生的,哪些是backward产生的!

  → outs_to_grad = [o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent]
  → grad_primals = [p for p in primals if isinstance(p, Tensor) and p.requires_grad]

  → ★ 核心: autograd.grad(needed_outs, grad_primals, grad_outputs=needed_tangents)
    → PyTorch autograd引擎动态构建backward graph
    → 在make_fx tracing下, 每个backward op也被记录到同一个FX graph!
    → 结果: 一个包含forward ops + backward ops的joint FX graph

  → ★ tangent处理: 每个tangent做view(tangent) → 防止tangent与primal是同一tensor
    (NestedTensor可能有内部alias → make_fx会specialize → 破坏joint graph)

  → final_outs = (outs, [next(backward_out_iter) if i else None for i in inputs_needs_grads])
    → joint graph的output = (forward_outputs, backward_outputs)
    → None表示该input不需要grad → backward graph中无对应gradient

★ joint graph结构:
  inputs: primals_0, primals_1, ..., tangents_0, tangents_1, ...
  ops: forward ops + backward ops (在同一个FX graph中)
  output: (fw_out_0, fw_out_1, ..., bw_out_0, bw_out_1, ...)

★ 为什么不用.backward()而用autograd.grad():
  - .backward()会accumulates gradients到.grad属性 → 不适合tracing
  - autograd.grad()返回gradient tensors → 直接作为graph outputs
  - 更精确: 可以选择哪些outputs需要gradient,哪些inputs需要gradient
```

### 2.3 _create_graph: make_fx tracing的精确机制

```
graph_capture.py: _create_graph(f, args, args_descs, *, aot_config):

  with FunctionalTensorMode(pre_dispatch=..., export=..., _allow_token_discovery=True):
    → FunctionalTensorMode: 开启functionalization dispatch
    → 每个in-place op → 被functionalize成out-of-place版本
    → 每个mutation → 变成额外output + copy_()在runtime epilogue

  fx_g = make_fx(inner_f, decomposition_table=..., record_module_stack=True)(*args)
    → make_fx使用ProxyTensor tracing:
      - 创建FakeTensor proxies
      - 执行inner_f(*args)
      - 每个torch.dispatch op → 记录为FX call_function node
      - 结果: fx.GraphModule containing joint fwd+bwd ops

★ make_fx的symbolic execution本质:
  - 不是静态分析代码 → 而是实际运行代码, 但用FakeTensor替代real tensor
  - FakeTensor: 没有真实数据, 只有shape/dtype/device/stride metadata
  - 每个op通过dispatch mechanism → 被PythonDispatcher + ProxyTorchDispatchMode拦截
  - 拦截后: 创建proxy node → 添加到FX graph → 返回新的FakeTensor proxy
  - symbolically executing = 用metadata推理结果shape → 不做真实计算

★ FunctionalTensorMode与make_fx的关系:
  - FunctionalTensorMode: 全局dispatch key → 所有tensor变成FunctionalTensor
  - FunctionalTensor.to_functional(t): 包装tensor → 记录后续mutation
  - torch.add_(x, y) → functionalize为 x_updated = torch.add(x, y) + 返回x_updated作为output
  - make_fx只看到functionalized后的out-of-place ops → graph无mutation!
```

## 3. Functionalization: Mutation Removal Pipeline

### 3.1 功能化转换: in-place → out-of-place

```
functional_utils.py: to_fun / from_fun / is_fun

to_fun(t):
  → isinstance(t, Tensor): return FunctionalTensor.to_functional(t)
  → isinstance(t, subclass): 递归transform_subclass(t, lambda _, inner: to_fun(inner))
  → else: return t

from_fun(t):
  → isinstance(t, FunctionalTensor): sync_functional_tensor(t) → torch._from_functional_tensor(t.elem)
  → isinstance(t, subclass): 递归transform_subclass(t, lambda _, inner: from_fun(inner))
  → else: return t

★ FunctionalTensor的内部机制(C++层面):
  - torch._C._functionalize: C++ dispatch key → 拦截所有in-place ops
  - torch.add_(self, other):
    → 不修改self的数据 → 而创建新tensor result = torch.add(self_value, other)
    → 记录mutation: self的"current value"更新为result
    → 继续使用self → 实际读取的是result(通过FunctionalTensor的indirection)
  - functionalize完成后:
    → 所有in-place op变成out-of-place
    → 所有mutation通过extra output传递
    → runtime epilogue做copy_()将更新写回原始tensor

★ 3类mutation处理:
  1. Data mutation (torch.mul_(2)):
     → functionalize为 out = torch.mul(x, 2)
     → out作为额外graph output
     → runtime: x.copy_(out)
     → ★ out参与backward graph(因为x的gradient需要flow through updated value)

  2. Metadata mutation (torch.t_()):
     → functionalize为 x_updated = torch.t(x)
     → x_updated作为额外graph output(但不参与backward!)
     → runtime: x.as_strided_(x_updated)
     → ★ metadata mutation不参与backward → 因为view op没有data-level gradient

  3. Storage mutation (set_()):
     → 特殊处理: set_()是FSDP2的关键mutation(参数sharding后unshard)
     → 需要careful ordering: resize_() + set_()顺序很重要
     → ★ 见Note [set_() Input Mutations in AOTAutograd] + Note [Ordering of resize_() and set_()]
```

### 3.2 Mutation检测与分类

```
functional_utils.py: has_data_mutation / has_metadata_mutation / are_all_mutations_hidden_from_autograd

has_data_mutation(t):
  → isinstance(t, FunctionalTensor): torch._functionalize_has_data_mutation(t.elem)
  → subclass: 递归检查内部tensor

has_metadata_mutation(f_arg, arg, *, check_only_storage_mutation):
  → arg_after = torch._from_functional_tensor(f_arg.elem)
  → maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)
  → same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(arg_after.untyped_storage())
  → if check_only_storage_mutation: return maybe_storage_changed and not same_storages
  → else: check sizes/strides/offset是否变化

★ C++ Functionalization API (torch._C._functionalize_*):
  - _functionalize_has_data_mutation(elem): 是否有data-level mutation
  - _functionalize_has_metadata_mutation(elem): 是否有metadata mutation
  - _functionalize_was_storage_changed(elem): 是否有set_()调用
  - _functionalize_are_all_mutations_hidden_from_autograd(elem): mutation是否under no_grad
  - _functionalize_mutation_counter(elem): mutation计数器(用于检测bw中的mutation)

★ Mutation分类的4种情况:
  1. mutations_hidden_from_autograd: under no_grad + 不bump VC → 不影响backward
  2. mutations_under_no_grad_or_inference_mode: under no_grad但bump VC
  3. mutates_data: 数据被修改 → 需要copy_()
  4. mutates_metadata/mutates_storage_metadata: metadata被修改 → 需要as_strided_()
```

### 3.3 create_functionalized_fn: 完整functionalization包装

```
graph_capture_wrappers.py: create_functionalized_fn(fn, args, args_descs, *, meta, aot_config, trace_joint):

_functionalized_f_helper(*args):
  → torch._C._ExcludeDispatchKeyGuard(DispatchKey.Functionalize)  ← 禁止上层functionalize
  → f_args = pytree.tree_map(to_fun, args)  ← 所有tensor变成FunctionalTensor

  → if trace_joint and has_input_mutated_in_graph and joint_fn_handle:
    → 设置joint_fn_handle.post_forward → 记录forward后的mutation状态
    → ★ 这样backward可以检测: 哪些input在forward期间被mutation了

  → f_outs, f_outs_descs = call_and_expect_output_descs(fn, f_args)

  → if trace_joint:
    → 检测backward中的input mutation:
      - joint_mutates_data = has_data_mutation(f_inpt)
      - joint_mutates_metadata = has_metadata_mutation(f_inpt, before)
      - ★ 如果input在backward中被mutation → 设置partitioner_tag = "must_be_in_backward"
      - 用copy_()执行mutation → 但标记为backward-only op
    → 检测backward中的tangent mutation → 大多数情况下会报错

  → if keep_inference_input_mutations:
    → 对于data-only mutation(无metadata mutation):
      → 用copy_()在graph内执行mutation → inductor可以看到copy_()并优化
    → ★ 这是FSDP2的关键: unshard后的copy_()留在graph内 → inductor可以优化它

★ functionalization pipeline 总结:
  Original: def f(x): x.mul_(2); out = x * y; return out
  Functionalized: def f(x): x_updated = x.mul(2); out = x_updated * y; return x_updated, out
  Graph: primals_0=x → mul → mul_1=x_updated → mul → out → output(x_updated, out)
  Runtime: x.copy_(x_updated) → 真正的mutation发生在epilogue

★ ★ functionalization = AOTAutograd最复杂的部分:
  - 8种edge case(见aot_autograd.py顶部long comment)
  - input data mutation → extra output + backward gradient
  - input metadata mutation → extra output + view replay
  - output aliasing input → view replay at runtime
  - output aliasing intermediate → return intermediate base
  - aliased input mutation → synthetic base
  - side-effect tokens → threading through graph
  - subclass mutation → recursive handling
  - backward mutation → special partitioner tag
```

## 4. Min-Cut Partition: 从Joint Graph到Fwd/Bwd的算法核心

### 4.1 Partition概览: 3种策略

```
partitioners.py: 3种partition策略

1. default_partition (行1350-1520):
   → 简单策略: 所有forward nodes → fwd graph
   → 保存所有被backward使用的forward intermediates
   → 无recomputation → 最大memory使用 → 最小compute
   → 如果有recomputable ops → 自动fallback到min_cut!

2. min_cut_rematerialization_partition (行3771-3920):
   → ★ 核心算法: 最小化saved activations memory
   → 通过max-flow/min-cut算法决定: 哪些forward nodes保存,哪些recompute
   → 自动gradient checkpointing从这里涌现!

3. waterloo_partition / greedy_partition:
   → 实验性策略,不常用

★ 实际使用:
  - torch.compile默认 → Inductor的partition_fn → min_cut_rematerialization_partition
  - default_partition有recomputable ops → fallback到min_cut
  - min_cut是production path!
```

### 4.2 classify_nodes: 节点分类的前置步骤

```
partitioners.py: classify_nodes(joint_module, static_lifetime_input_indices, num_fwd_outputs):

  → required_bw_nodes: 从tangent placeholders开始 → 沿users传播 → 所有backward必需的nodes
  → forward_only_graph = _extract_graph_with_inputs_outputs(joint_graph, inputs=primals, outputs=fwd_outputs)
  → required_fw_nodes: forward_only_graph中的所有nodes
  → unclaimed_nodes: 既不在fw也不在bw的nodes (通常是shared计算)

  → tangents_closure: 从tangent开始但未通过users传播的nodes
    → ★ 区别: required_bw_nodes包含所有backward必需的(包括间接依赖)
    → tangents_closure只包含tangent的直接使用者 → 决定哪些可以save vs recompute

  → static_lifetime_input_nodes: parameter/buffer inputs → 对FSDP2特别重要

★ classify_nodes的3类nodes:
  1. required_fw_nodes: forward必须的计算 → 在fwd graph中执行
  2. required_bw_nodes: backward必须的计算 → 在bwd graph中执行(或recompute)
  3. unclaimed_nodes: 两者都可能需要的 → min-cut决定placement

★ ★ tangents_closure vs required_bw_nodes的区别:
  required_bw_nodes = tangents_closure + 间接backward依赖
  → tangents_closure决定: 某个forward output是否"直接"被backward需要
  → 如果node在tangents_closure中 → 可以考虑recompute(因为直接被backward需要)
  → 如果node只在required_bw_nodes中(间接依赖) → 可能必须save
```

### 4.3 ★★★ Min-Cut算法: solve_min_cut 的精确机制

```
partitioners.py: solve_min_cut(joint_graph, node_info, min_cut_options, dont_ban=None):

这是AOTAutograd最核心的算法! 用networkx的minimum_cut实现。

Step 1: 构建二分图(bipartite graph) → 每个FX node变成两个节点
  ┌─────────────────────────────────────────────┐
  │  source                                      │
  │    ↓ (inf edge: cannot recompute)            │
  │  node_in ←→ node_out (capacity = weight)     │
  │    ↓ (inf edge: data dependency)             │
  │  user_in ←→ user_out                         │
  │    ↓                                         │
  │  ...                                         │
  │    ↓ (inf edge: must be in backward)          │
  │  sink                                        │
  └─────────────────────────────────────────────┘

  ★ 每个FX node → node_in + node_out 两个networkx节点
  ★ node_in → node_out 边的capacity = weight(node的memory size)
  ★ source → node_in 边的capacity = inf (表示"不能recompute")
  ★ node_out → user_in 边的capacity = inf (表示"数据依赖")
  ★ node → sink 边的capacity = inf (表示"必须在backward")

  ★ ★ min-cut的含义:
    - source侧 = forward graph (保存这些nodes)
    - sink侧 = backward graph (recompute这些nodes)
    - cut = node_in → node_out边被切断 = 该node需要save for backward!
    - min-cut = 最小化被切断边的总capacity = 最小化saved activations的memory!
```

### 4.3.1 边构建的精确规则

```
1. required_bw_nodes:
   → if node in tangents_closure:
       node_in → sink (inf)  ← "必须在backward中计算"
   → if node NOT in tangents_closure:
       node_out → sink (inf)  ← "output必须可被backward使用"

2. must_recompute nodes (user annotation):
   → node_in → sink (inf)  ← "guaranteed to be in backward"

3. primals/fwd_seed_offset:
   → source → node_in (inf)  ← "不能recompute input"

4. should_ban_recomputation:
   → source → node_in (inf, reason)  ← "不能recompute这个op"
   ★ 5种ban理由:
     a. not in recomputable allowlist → 不可recompute的op类型
     b. random op → dropout/rand_like不能recompute
     c. compute intensive op → mm/conv/bmm/addmm不能recompute
     d. materialized in backward → backward中必须materialize的op
     e. reduction op → output比input小4x → save比recompute便宜

5. node_in → node_out 边:
   → weight = memory size of node's output
   → ★ 如果node是materialized(被非fusible op消费): weight = mem_sz
   → ★ 如果node是fusible(可以被后续op fuse): weight = mem_sz * 2
     (双倍惩罚 → 鼓励recompute fusible chains → 因为fusion可以避免materialization)
   → ★ dist_from_bw heuristic: weight *= 1.1^(min(dist_from_bw, 100))
     (越远离backward → 保存越贵 → 鼓励recompute far-away nodes)

6. node_out → user_in 边:
   → capacity = inf (数据依赖 → 不能切断)

7. static_lifetime_input_nodes (FSDP2 parameters):
   → weight = 0  ← ★ parameters/buffers免费保存! 因为它们始终存在!
   → (config.treat_parameters_as_free_to_save = True)

★ ★ ★ min-cut算法的直觉:
  - 想最小化: forward graph需要保存给backward的activations总量
  - 每个forward node可以选择: SAVE(放在fwd output) vs RECOMPUTE(放在bwd graph)
  - SAVE的代价: node output的memory size
  - RECOMPUTE的代价: 重新计算node(但某些node不能recompute: random/compute-intensive)
  - min-cut找到最优分割: 最小化total saved memory
  - 这就是automatic gradient checkpointing!
```

### 4.3.2 networkx.minimum_cut: max-flow算法

```
nx_graph构建完成后:

cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")

★ networkx.minimum_cut使用max-flow算法(Edmonds-Karp或Dinic):
  - max-flow = 从source到sink的最大流量
  - min-cut定理: max-flow value = min-cut capacity
  - partition = (reachable_from_source, reachable_from_sink)
  - reachable_from_source = forward graph (保存的nodes)
  - reachable_from_sink = backward graph (recompute的nodes)

  - cutset = 从reachable到non-reachable的边
  - ★ 只关心node_in→node_out边(这些边被切断 = 该node要save for backward)
  - cut_nodes = {node_name for node_in, node_out in cutset}

  - saved_values = sorted(cut_nodes, key=original_graph_order)
  → 这就是forward graph的额外outputs → 传递给backward graph!

★ 如果inf-capacity path从source到sink → NetworkXUnbounded异常:
  → 某个node既不能recompute也不能save → 但backward需要它 → BUG!
  → _find_infinite_capacity_path(): BFS沿inf边找path → 报告具体冲突nodes
  → RuntimeError: "AOT Autograd failed to partition..." + 详细诊断信息

★ ★ min-cut的数学保证:
  - 给定所有constraints(哪些op不能recompute,哪些必须save)
  - min-cut找到全局最优的save/recompute分配
  - 不是greedy → 是全局优化!
  - 等价于: 在所有可行的fwd/bwd分割中, 选择activations memory最小的那个
```

### 4.4 choose_saved_values_set: Memory Budget分级策略

```
partitioners.py: choose_saved_values_set(joint_graph, node_info, memory_budget=1):

★ memory_budget参数: 0.0到1.0
  → 1.0 = 默认: min-cut结果(runtime优化,但不是最小memory)
  → 0.0 = 最小memory: 只save inputs → 所有forward ops都recompute
  → 中间值 = trade-off: memory vs compute

执行策略:
  1. solve_min_cut(runtime_optimized_options) → saved_values_1
     → ban_used_far_apart=True, ban_long_fusible_chains=True, ban_materialized_backward=True
     → 最保守的recomputation策略 → 只recompute"安全"的ops

  2. if memory_budget < mem_ratio(saved_values_1): return saved_values_1

  3. solve_min_cut(more_aggressive_options) → saved_values_2
     → 放松前3个ban → 更多ops可以recompute → 更省memory

  4. solve_min_cut(aggressive_options) → saved_values_3
     → 只保留ban_reduction → 几乎所有ops都可以recompute → 最大memory节省

  5. if still too much memory → knapsack solver!
     → _optimize_runtime_with_given_memory():
       → 4种solver: greedy/ilp/dp/dp_knapsack_sliding_hirschberg
       → 输入: banned_nodes的memory weights + runtime weights + max_memory_budget
       → 输出: 哪些banned_nodes应该允许recompute (在budget内)
       → 然后用dont_ban set重新跑solve_min_cut

★ ★ ★ memory_budget的4级递进:
  Level 1: runtime_optimized → 只recompute简单pointwise ops
  Level 2: more_aggressive → 也recompute中等ops(放宽fusible chain/materialized ban)
  Level 3: aggressive → 几乎全recompute(只保留reduction ban)
  Level 4: knapsack → 精确budget约束下的runtime/memory trade-off

★ activation_memory_budget_solver config选项:
  - "dp": 动态规划 → 精确解 → 但O(n*W)复杂度
  - "greedy": 贪心 → 快但可能不最优
  - "ilp": 整数线性规划 → 最精确但可能很慢
  - "dp_knapsack_sliding_hirschberg": 空间优化的DP → O(n)空间
```

### 4.5 ★★★ Gradient Checkpointing从Min-Cut涌现!

```
★ ★ ★ 核心insight:

传统gradient checkpointing:
  → 用户手动标注: torch.utils.checkpoint.checkpoint(layer, ...)
  → 桥接策略: forward只保存input → backward时recompute整个layer

AOTAutograd的automatic checkpointing:
  → min-cut算法自动决定: 哪些forward ops保存,哪些recompute
  → 无需手动标注! 算法找到全局最优分配!
  → ★ 这就是"automatic gradient checkpointing"的来源!

具体机制:
  1. joint graph包含forward + backward ops
  2. min-cut partition: 某些forward nodes → 在backward graph中重新出现(recompute)
  3. _extract_fwd_bwd_modules():
     → fw_module: forward计算 + 输出saved activations
     → bw_module: 包含recomputed forward subgraphs + original backward ops
  4. ★ bw_module的结构: [recomputed_fwd_subgraph_1][original_bwd_1][recomputed_fwd_subgraph_2][original_bwd_2]...

★ ★ 用户也可以手动影响partition:
  → node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE → min-cut优先recompute
  → node.meta["recompute"] = CheckpointPolicy.MUST_SAVE → min-cut必须save(不能recompute)
  → torch.autograd.graph.region_activation_memory_budget(budget) → 设置memory budget

★ checkpointing vs min-cut的数学等价:
  - checkpointing = "save这些, recompute其余" → 一种特定的partition
  - min-cut = "在所有可行partition中, 找memory最小的" → 全局最优
  - min-cut结果 ≥ 任何手动checkpointing方案(如果所有constraints一致)
  - ★ min-cut是gradient checkpointing的"最优版本"!
```

### 4.6 _extract_fwd_bwd_modules: 从joint到fwd/bwd的实际提取

```
partitioners.py: _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes, ...):

  → _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    → fwd_outputs = outputs[:num_fwd_outputs]
    → bwd_outputs = outputs[num_fwd_outputs:]

  → bwd_graph = _extract_graph_with_inputs_outputs(joint_graph, inputs=saved_values+tangents, outputs=bwd_outputs)
    → ★ backward graph的inputs = saved activations from forward + tangent gradients
    → ★ 通过env dict: 只有在saved_values/tangents中的nodes变成placeholder
    → 其他forward nodes → InvalidNode → 不出现在backward graph
    → ★ 但recomputed nodes → 它们在backward中重新计算 → 出现在bwd graph!

  → ★ ★ saved_values分3组:
    1. saved_values_with_vc_check → save_for_backward (autograd version counter check)
    2. saved_values_no_vc_check → 直接stash在ctx (无VC check → 更快但更unsafe)
    3. saved_opaque_objects → FakeScriptObject等非tensor值

  → fwd_graph = _extract_graph_with_inputs_outputs(joint_graph, inputs=primals, outputs=fwd_outputs+saved_values+saved_sym_nodes)
    → ★ forward graph的outputs = user outputs + saved_for_backward activations!
    → ★ 这些saved activations在runtime通过save_for_backward()传递给backward

★ _extract_graph_with_inputs_outputs的机制:
  - 创建new_graph
  - 指定的input nodes → 变成placeholder
  - 遍历joint_graph.nodes:
    - 如果node在env(已被处理) → skip
    - 如果node的所有args都在env → node_copy到new_graph
    - 如果node有arg不在env → InvalidNode → skip
  - ★ 这确保: 只有"可追溯"到指定inputs的nodes出现在子图中
  - ★ recomputed nodes在backward中: 因为saved_values包含它们的上游 → 可追溯

★ ★ fwd/bwd graph的最终signature:
  Forward:
    inputs:  primals_0, primals_1, ..., fwd_rng_state
    outputs: user_out_0, user_out_1, ..., saved_act_0, saved_act_1, ..., saved_sym_0, ...

  Backward:
    inputs:  saved_act_0, saved_act_1, ..., saved_sym_0, ..., tangents_0, tangents_1, ..., bw_rng_state
    outputs: grad_primals_0, grad_primals_1, ...
```

### 4.7 default_partition vs min_cut: 为什么min_cut更优

```
default_partition:
  → 所有forward nodes → fwd graph
  → backward使用的forward intermediates → 全部save
  → 无recomputation → 最大memory → 最小额外compute
  → ★ fallback: 如果有recomputable ops → 自动用min_cut

min_cut_rematerialization_partition:
  → 通过max-flow/min-cut算法 → 最优分配save vs recompute
  → 结果: 某些forward ops出现在both fwd and bwd graphs(recomputed)
  → ★ 大幅减少saved activations memory → 但增加backward compute
  → ★ ★ 这是trade-off: memory bandwidth vs compute → 对decode-bound推理至关重要!

★ 在RTX 4090上:
  - memory-bound推理 → activations memory是瓶颈 → min-cut省memory → 更快!
  - 训练(7B LoRA): activations是主要memory开销 → min-cut减少peak memory
  - ★ ★ min-cut对memory-bound场景 = 性能关键!
```

## 5. Recomputation决策的5种Ban Heuristic

```
partitioners.py: MinCutOptions + should_ban_recomputation

MinCutOptions (5个boolean开关):
  ban_if_used_far_apart           = True (default)
  ban_if_long_fusible_chains      = True (default)
  ban_if_materialized_backward    = True (default)
  ban_if_not_in_allowlist         = True (default)
  ban_if_reduction                = True (default)

★ 5种Ban的理由:

1. ban_if_not_in_allowlist:
   → op不在recomputable_ops list → 不能recompute
   → recomputable_ops = pointwise ops + view ops + simple arithmetic + ...
   → 不在list的: mm, conv, bmm, addmm, SDPA, dropout, rand_like
   → ★ compute_intensive ops不能recompute: 重新计算mm太贵!
   → ★ random ops不能recompute: dropout结果在forward和backward中必须一致!

2. ban_if_reduction:
   → 如果output_size * 4 < input_size → 不允许recompute
   → 例: sum/mean/amax → output比input小很多 → save很便宜 → 不值得recompute
   → ★ 防止recompute normalization ops → save BN/LN的统计值很便宜

3. ban_if_materialized_backward:
   → 如果node的output在backward中必须materialized(不能fuse) → 不recompute
   → materialized = backward consumer不是fusible op → 需要写入memory → recompute没意义
   → ★ 因为: 如果必须materialize → 不管是save还是recompute都要写memory → save更便宜

4. ban_if_used_far_apart:
   → 如果node的output被forward中远处的多个user使用 → 不recompute远处的user
   → find_first_unfusible_use(): 找到第一个不可融合的use
   → 远处的fusible use → ban recomputation → 因为长chain难以fuse
   → ★ 防止: "从头到尾的长fusible chain"(如residual connection) → 保留中间activation

5. ban_if_long_fusible_chains:
   → 从每个forward node开始 → 沿fusible users走 → 如果chain超过100个ops → ban末端
   → ★ 防止degenerate case: 全模型一个长fusible chain → 全recompute → 但memory没省多少

★ ★ aggressive_recomputation config:
  → 设置所有5个ban = False (只保留ban_reduction)
  → 最大recomputation → 最小memory → 但可能增加compute
  → 适合: memory极度受限的场景

★ recomputable_ops list (get_default_op_list):
  核心: add/sub/mul/div/exp/log/sin/cos/gelu/sigmoid/relu/where/clamp/...
  + view ops: view/slice/t/expand/permute/select/...
  + clone/_to_copy/convert_element_type/...
  ★ ★ 不包含: mm/conv/bmm/SDPA/dropout/rand → 这些太贵或不可复现!
```

## 6. CompiledFunction / CompiledBackward 结构

### 6.1 AOTDispatchAutogradCompileSpec: 编译规格

```
runtime_wrappers.py: AOTDispatchAutogradCompileSpec:
  compiled_fw_func: Callable          ← Inductor编译的forward
  compiled_bw_func: Callable | None   ← Inductor编译的backward (lazy!)
  maybe_subclass_meta: SubclassMeta | None
  num_symints_saved_for_bw: int
  backward_state_indices: list[int]
  disable_amp: bool
  indices_of_inps_to_detach: list[int]
  lazy_backward_info: AutogradLazyBackwardCompileInfo  ← ★ lazy backward编译!

★ ★ Lazy Backward Compilation:
  - backward不立即编译 → 只在第一次backward call时编译!
  - 原因: 很多模型只做forward(inference) → 不需要backward
  - 减少编译时间 → 只编译实际需要的部分
  - lazy_backward_info包含: bw_module + compiled_bw_func占位符
```

### 6.2 ★★★ CompiledFunction(torch.autograd.Function): 核心运行时结构

```
runtime_wrappers.py: CompiledFunction(torch.autograd.Function):

class CompiledFunction(torch.autograd.Function):
  compiled_fw = compiled_fw_func       ← Inductor编译的forward kernel
  compiled_bw = compiled_bw_func       ← Inductor编译的backward kernel
  metadata = fw_metadata               ← ViewAndMutationMeta
  maybe_subclass_metadata = maybe_subclass_meta
  num_symints_saved_for_bw = num_symints_saved_for_bw_
  _aot_id = aot_config.aot_id
  _lazy_backward_info = lazy_backward_info

  @staticmethod
  def forward(ctx, *deduped_flat_tensor_args):
    → return CompiledFunction._fwd_fn(ctx, deduped_flat_tensor_args,
        rng_state.add_forward_args,
        saved_state.save_from_forward,
        forward_epilogue.finalize,
        CompiledFunction.compiled_fw)

  @staticmethod
  def backward(ctx, *flat_args):
    → return CompiledFunction._bwd_fn(flat_args, ctx,
        CompiledFunction._bw_prologue_fn,
        rng_state.add_backward_args,
        CompiledFunction._backward_impl,
        CompiledFunction._bw_epilogue_fn,
        CompiledFunction._double_backward)

  @staticmethod
  def _backward_impl(ctx, all_args):
    → saved_tensors_use_once = not torch._C._autograd._get_current_graph_task_keep_graph()
    → compiled_bw = backward_compiler.get_or_compile(saved_tensors_use_once=...)
    → CompiledFunction.compiled_bw = compiled_bw  ← ★ lazy compile backward!
    → call_func_at_runtime_with_args(compiled_bw, all_args, steal_args=True)

★ ★ forward()的5个阶段:
  1. rng_state.add_forward_args → 添加RNG state到forward args
  2. compiled_fw(*args) → 执行Inductor编译的forward kernel
  3. saved_state.save_from_forward → save_for_backward(saved_activations)
  4. forward_epilogue.finalize → copy_() mutations back to inputs
  5. 返回: (user_outputs, saved_for_backward_outputs)

★ ★ backward()的5个阶段:
  1. _bw_prologue_fn → 准备backward inputs(saved activations + gradients)
  2. rng_state.add_backward_args → 添加backward RNG state
  3. _backward_impl → lazy compile + 执行backward kernel
  4. _bw_epilogue_fn → 处理backward的output (detach + reshape)
  5. _double_backward → 如果create_graph=True → 创建CompiledFunctionBackward

★ ★ ★ CompiledFunction = PyTorch autograd的标准接口:
  - 继承torch.autograd.Function → PyTorch autograd引擎知道如何调用它
  - forward() → 替代eager forward → 用Inductor编译的kernel
  - backward() → 替代eager backward → 用Inductor编译的kernel
  - ctx.save_for_backward() → 传递saved activations → 和eager autograd完全兼容!

★ ★ ★ 为什么backward也被编译?
  - 传统: backward由autograd引擎动态构建 → 每个op单独执行 → 无法融合
  - AOTAutograd: backward是整个编译的kernel → 所有ops融合在一起 → Triton kernel!
  - 例: backward中的mm + add + sum → 一个Triton kernel → 大幅加速!
```

### 6.3 SerializableCompiledFunction: 缓存与序列化

```
runtime_wrappers.py: SerializableCompiledFunction:
  compiled_fn: Callable   ← 实际的compiled function
  serialize_fn: Callable  ← 序列化函数(用于AOTAutogradCache)

  → __call__(*args): compiled_fn(*args)
  → serialize(): serialize_fn() → 返回GenericAOTAutogradResult

★ AOTAutogradCache:
  → 本地缓存 + 远程缓存(AOTAutogradRemoteCache)
  → should_use_local_autograd_cache() / should_use_remote_autograd_cache()
  → 避免重复编译相同graph → 大幅减少compile时间
```

## 7. Side-Effect Token Handling

```
aot_autograd.py: Note [Side-Effectful Tokens in AOTAutograd]

★ 问题: 有些ops有side effects(print, torchbind operations) → 不是functional!
★ 解决: 用"effect tokens"显示数据依赖

token机制:
  → token = torch.tensor([]) → dummy value → 只表示"顺序依赖"
  → graph inputs: (*tokens, *params_buffers, *user_inputs)
  → 每个side-effectful op → with_token(ordered_effect_op, args, prev_token)
  → 输出: (new_token, actual_output)
  → graph outputs: (*tokens, *outputs)

★ ★ Inductor不需要tokens → post-processing pass移除:
  → 原始graph: def gm(self, token0, reader): token1, frame = with_token(op, reader, token0); ...
  → 移除后: def gm(self, reader): token0 = prims._make_token(); token1, frame = with_token(op, reader, token0); sink_token = prims._sink_tokens([token1]); ...

★ 2种token处理:
  1. unlift_tokens(fw_module, fw_metadata, aot_config):
     → 将token从graph inputs移除 → 在graph内创建和sink
  2. handle_effect_tokens_fn(fn, args, descs, meta, trace_joint):
     → 为joint tracing设置token → forward tokens + backward tokens分离

★ ★ ★ token = functionalization的"最后防线":
  - functionalization处理tensor mutations → 但print/torchbind不是tensor mutation
  - token threading确保: side-effectful ops的执行顺序被graph结构保证
  - 不是functional → 但token让它们"看起来"functional(有显式数据依赖)
```

## 8. FSDP2与AOTAutograd的交互

### 8.1 static_input_indices: FSDP2参数免费保存

```
schemas.py: AOTConfig:
  static_input_indices: list[int] | None = None

config.py:
  treat_parameters_as_free_to_save = True

★ ★ ★ FSDP2的关键影响:

1. static_input_indices:
   → FSDP2通过TracingContext.fw_metadata.static_input_indices传递
   → 指明哪些inputs是parameters/buffers → 它们始终在GPU上 → "免费"保存
   → get_static_input_idxs() (compile_fx.py:259): 从TracingContext获取

2. treat_parameters_as_free_to_save = True:
   → 在min-cut中: static_lifetime_input_nodes的weight = 0
   → ★ 含义: parameter/buffer activations不需要额外memory → 因为它们始终存在!
   → FSDP2: unsharded parameter通过AllGather → 已经在GPU → 不算peak memory额外开销
   → ★ 这让min-cut更aggressive地recompute → 因为parameter的"saved"是免费的

3. FSDP set_() mutation:
   → FSDP2的unshard = 参数.set_(unsharded_flat_param)
   → 这是storage mutation → functionalize为: param_updated = param.set(unsharded)
   → ★ apply_in_graph_mutations()处理set_() → 在runtime epilogue执行
   → ★ resize_() + set_()顺序很重要 → FSDP2先resize然后set

4. FSDP AllGather in graph:
   → torch.ops.fsdp.all_gather_copy_in → 在forward graph中
   → _is_fsdp_all_gather_copy_in(func): 检测FSDP ops → skip aliasing check
   → ★ AllGather是collective → force_save_collectives → 不允许recompute!
   → 因为: 通信太贵 → recompute AllGather = 重新做AllGather → 不可接受

5. ★ ★ _sync_decision_cross_ranks:
   → 分布式训练中: 不同rank可能有不同的min-cut结果
   → _cone_hashes(): 用structural hash确定graph结构是否一致
   → _canonical_node_names(): Kahn's algorithm → 确定性node命名
   → all_gather_object: 比较所有rank的saved_values → 选最小memory的rank的决策
   → ★ 确保所有rank的partition决策一致 → 防止NCCL同步问题
```

### 8.2 FSDP2对Partition决策的影响

```
★ ★ ★ FSDP2改变min-cut的3个方面:

1. Parameter保存成本 = 0:
   → min-cut认为: 保存FSDP2参数无额外开销 → 更倾向保存
   → 但实际上: FSDP2参数是sharded → AllGather后才能用 → 有通信开销
   → ★ min-cut只考虑local memory → 不考虑通信开销 → 这是局限!

2. Collective ops不能recompute:
   → force_save_collectives(): 所有collective ops → source→node_in inf edge
   → ★ AllGather/ReduceScatter不能recompute → 必须save其output
   → 这增加了saved activations → 但无法避免

3. FSDP2的compile路径:
   → FSDP2注册custom backend → torch._dynamo.optimize(custom_backend)
   → custom_backend = inductor → 但有FSDP-specific config
   → static_input_indices从FSDP2传入 → 告知哪些是parameters
   → ★ FSDP2的intentional graph break: 在FSDP hooks处 → 每个FSDP unit独立compile

★ ★ RTX 4090上的FSDP2 + AOTAutograd:
  → 7B模型单GPU → FSDP2 sharding无意义(单GPU)
  → LoRA训练 → LoRA adapter参数小 → 不需要FSDP2
  → ★ RTX 4090最优: 不用FSDP2 → 用LoRA + CPU_Adam → 单GPU训练
  → 如果多GPU(PCIe): FSDP2可用但性能差(PCIe bandwidth瓶颈)
```

## 9. 数据流全景: 从用户代码到编译结果

```
★ ★ ★ AOTAutograd完整数据流:

用户代码: def f(x): return x.sin().cos()
  ↓
Dynamo: 符号执行 → FX Graph(aten ops)
  ↓
★ AOTAutograd Stage 1: Graph Capture
  ├─ run_functionalized_fw_and_collect_metadata → ViewAndMutationMeta
  ├─ create_functionalized_fn → FunctionalTensorMode包装
  ├─ fn_prepped_for_autograd → mutation→outputs + clone + grad_mask
  ├─ create_joint → forward + autograd.grad → joint graph
  ├─ subclass unpack/pack → trace plain tensors
  ├─ handle_effect_tokens_fn → token threading
  └─ make_fx → FX GraphModule(joint fwd+bwd)
  ↓
★ AOTAutograd Stage 2a: Partition
  ├─ classify_nodes → required_fw/bw/unclaimed
  ├─ ★ solve_min_cut → networkx max-flow/min-cut
  │   ├─ 构建bipartite graph (source/node_in/node_out/sink)
  │   ├─ 边capacity = memory weight / inf constraint
  │   ├─ nx.minimum_cut → (reachable, non_reachable)
  │   └─ cutset → saved_values (forward保存给backward的activations)
  ├─ _extract_fwd_bwd_modules → fw_module + bw_module
  ├─ reordering_to_mimic_autograd_engine → 优化backward执行顺序
  └─ raise_getitems + thread_graphsafe_rng → 后处理
  ↓
★ AOTAutograd Stage 2b: Compile
  ├─ fw_compiler(fw_module, args) → Inductor编译forward
  ├─ bw_compiler(bw_module, args) → Inductor编译backward (lazy!)
  ↓
★ AOTAutograd Stage 2c: Make autograd.Function
  ├─ AOTDispatchAutogradCompileSpec → 编译规格
  ├─ AOTDispatchAutograd.post_compile → CompiledFunction
  ├─ CompiledFunction(torch.autograd.Function):
  │   ├─ forward() → compiled_fw + save_for_backward + epilogue
  │   ├─ backward() → lazy compile bw + compiled_bw
  │   └─ _double_backward → CompiledFunctionBackward(报错!)
  └─ 返回: wrapped callable → 替代原始eager execution

★ ★ ★ 总结:
  用户的f(x) = x.sin().cos()
  → 传统eager: forward=逐op执行, backward=逐op反向传播
  → AOTAutograd: forward=一个融合Triton kernel, backward=一个融合Triton kernel
  → ★ 两者的autograd行为完全一致(gradient值相同)!
  → ★ 但compiled版本更快: 因为fusion + 内存优化 + 代码生成!
```

## 10. 关键配置与调优

```
torch/_functorch/config.py 关键配置:

★ Partitioner相关:
  treat_parameters_as_free_to_save = True
    → FSDP2参数保存成本=0 → 对FSDP2训练重要!
  max_dist_from_bw = 1000
    → ban_if_used_far_apart的距离阈值
  ban_recompute_used_far_apart = True
  ban_recompute_long_fusible_chains = True
  ban_recompute_materialized_backward = True
  ban_recompute_not_in_allowlist = True
  ban_recompute_reductions = True
  aggressive_recomputation = False
    → True = 只保留ban_reduction → 最大recompute
  recompute_views = False
    → True = view ops也可以recompute(通常free)

★ Memory Budget相关:
  activation_memory_budget = 1.0
    → 0.0=最小memory(全recompute), 1.0=默认(runtime优化)
  activation_memory_budget_runtime_estimator = "flops"
    → "flops"/"profile"/"testing"/CustomRuntimeEstimator
  activation_memory_budget_solver = "dp"
    → "dp"/"greedy"/"ilp"/"dp_knapsack_sliding_hirschberg"/CustomKnapsackSolver

★ Functionalization相关:
  functionalize_rng_ops = False
    → True = RNG ops也被functionalize → graph-safe random
  cse = True → Common Subexpression Elimination on joint graph
  unsafe_allow_optimization_of_collectives = False
    → True = 允许recompute collectives → 通常危险!

★ ★ RTX 4090最优配置:
  训练(LoRA):
    activation_memory_budget = 1.0 (默认足够)
    treat_parameters_as_free_to_save = True
    aggressive_recomputation = False (默认)
    → ★ 单GPU + LoRA → activations不大 → min-cut默认策略足够

  推理(INT4):
    不需要AOTAutograd → inference path → 无backward
    → aot_dispatch_base → 只编译forward → 无partition
```

## 11. 与其他框架的关系

```
★ ★ AOTAutograd vs 其他系统:

1. vs DeepSpeed ZeRO-3:
   → ZeRO-3: 参数AllGather→forward→Release→backward时重新AllGather
   → AOTAutograd: 参数通过FSDP2的set_() → 在graph内
   → ★ ZeRO-3不支持torch.compile → AOTAutograd不支持ZeRO-3
   → ★ FSDP2 + AOTAutograd = PyTorch官方推荐的分布式训练路径

2. vs Megatron TP:
   → Megatron: 手动partition forward/backward → PP pipeline
   → AOTAutograd: 自动partition → min-cut全局优化
   → ★ Megatron有自己的backward优化 → 不用AOTAutograd

3. vs vLLM:
   → vLLM推理 → 不需要backward → aot_dispatch_base(inference only)
   → ★ 推理路径: functionalization + make_fx → 只编译forward → 无partition

4. vs verl:
   → verl使用torch.compile → 走AOTAutograd路径
   → ★ GRPO训练: actor模型 → compile → AOTAutograd → min-cut → Inductor
   → ★ ref模型: inference path → 只编译forward

5. vs rLLM:
   → rLLM TinkerBackend: in-process → 单GPU → AOTAutograd正常工作
   → ★ LoRA + AOTAutograd: LoRA merge后 → 无需backward → inference path
   → ★ LoRA训练: AOTAutograd + min-cut → 自动gradient checkpointing
```

## Sources

### 核心源文件 (pytorch/pytorch main branch)
1. `torch/_functorch/aot_autograd.py` (1959行) — AOTAutograd主逻辑, aot_function/aot_module/create_aot_state, Note注释(edge cases)
2. `torch/_functorch/partitioners.py` (4046行) — Partition策略: solve_min_cut, choose_saved_values_set, min_cut_rematerialization_partition, default_partition, classify_nodes, NodeInfo, MinCutOptions, estimate_runtime, _optimize_runtime_with_given_memory
3. `torch/_functorch/_aot_autograd/graph_capture_wrappers.py` — create_joint, create_functionalized_fn, fn_prepped_for_autograd, fn_input_mutations_to_outputs, handle_effect_tokens_fn
4. `torch/_functorch/_aot_autograd/graph_capture.py` — aot_dispatch_autograd_graph, aot_dispatch_base_graph, _create_graph, _prepare_graph_capture_tracing
5. `torch/_functorch/_aot_autograd/graph_compile.py` — aot_stage1_graph_capture, aot_stage2_compile, aot_stage2_autograd, _aot_stage2a_partition, _aot_stage2b_fw_compile, _aot_stage2b_bw_compile, _aot_stage2c_make_autograd_function
6. `torch/_functorch/_aot_autograd/runtime_wrappers.py` — AOTDispatchAutogradCompileSpec, CompiledFunction(torch.autograd.Function), SerializableCompiledFunction, RuntimeWrapper, AOTSyntheticBaseWrapper, AOTDedupeWrapper
7. `torch/_functorch/_aot_autograd/functional_utils.py` — to_fun/from_fun/is_fun, has_data_mutation, has_metadata_mutation, are_all_mutations_hidden_from_autograd, assert_functional_graph
8. `torch/_functorch/_aot_autograd/schemas.py` — AOTConfig, AOTState, ViewAndMutationMeta, GraphSignature, InputAliasInfo, OutputAliasInfo
9. `torch/_functorch/config.py` — treat_parameters_as_free_to_save, activation_memory_budget, aggressive_recomputation, ban_recompute_* heuristics
10. `torch/_inductor/compile_fx.py` — partition_fn (Inductor的入口), get_static_input_idxs, min_cut_rematerialization_partition调用

### 相关源文件
- `torch/_functorch/_aot_autograd/collect_metadata_analysis.py` — run_functionalized_fw_and_collect_metadata
- `torch/_functorch/_aot_autograd/input_output_analysis.py` — create_graph_signature, compute_overlapping_inputs
- `torch/_functorch/_aot_autograd/subclass_utils.py` — unwrap/wrap tensor subclasses
- `torch/_functorch/_aot_autograd/descriptors.py` — AOTInput/AOTOutput descriptor types
- `torch/_functorch/_activation_checkpointing/` — knapsack solvers, AC remat tags, CheckpointPolicy
- `torch/fx/experimental/proxy_tensor.py` — make_fx (ProxyTensor tracer)
- `torch/_subclasses/functional_tensor.py` — FunctionalTensor implementation
- `torch/_C/_functionalize` — C++ functionalization dispatch key

### ★ ★ ★ 核心发现总结

1. **AOTAutograd = 3阶段编译桥梁**: Stage1(trace joint) → Stage2a(min-cut partition) → Stage2b(compile) → Stage2c(wrap autograd.Function)

2. **Joint graph追踪 = forward + autograd.grad()**: 不是静态分析 → 而是symbolic execution with FakeTensor → 每个backward op也记录到FX graph

3. **Functionalization = mutation→outputs pipeline**: 8种edge case → torch.add_变成torch.add + extra output → runtime epilogue做copy_()

4. **★★★ Min-cut = max-flow/min-cut算法**: 构建bipartite graph(source/node_in/node_out/sink) → edge capacity = memory weight → networkx.minimum_cut → 最优save/recompute分配 → 这就是automatic gradient checkpointing!

5. **5种ban heuristic**: ban_not_in_allowlist(compute-intensive)/ban_reduction(小output)/ban_materialized_backward/ban_used_far_apart/ban_long_fusible_chains → 控制哪些ops可以recompute

6. **Memory budget分级**: runtime_optimized → more_aggressive → aggressive → knapsack → 4级递进 → 从最大runtime优化到最小memory

7. **CompiledFunction = torch.autograd.Function**: forward=Inductor kernel + save_for_backward → backward=lazy compile + Inductor kernel → 完全兼容PyTorch autograd引擎

8. **FSDP2影响**: static_input_indices → parameters weight=0 → collective ops不能recompute → _sync_decision_cross_ranks → 确保分布式一致性

9. **★ ★ ★ RTX 4090最优**: 单GPU → LoRA → AOTAutograd + min-cut默认 → 自动gradient checkpointing → inference: 不需要backward → aot_dispatch_base路径
