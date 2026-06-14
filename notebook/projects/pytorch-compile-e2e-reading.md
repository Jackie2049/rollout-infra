# PyTorch torch.compile End-to-End 源码级追踪

> 2026-06-15 | 源码: PyTorch 2.2.2 (ai-infra conda), torch/_dynamo + torch/_inductor + torch/_functorch
> 核心: compile→Dynamo符号执行→FX graph→AOTAutograd joint fwd+bwd→partition→Inductor lowering→IR→Scheduler fusion→Triton/C++ codegen→PyCodeCache→compiled fn
> FSDP2+compile: intentionally graph breaks at FSDP hooks → compile optimizes within each FSDP unit

## 1. 入口: torch.compile(model) → Backend Dispatch

```
torch/__init__.py:1710-1827:

torch.compile(model, backend="inductor"):
  → backend=="inductor": 创建_TorchCompileInductorWrapper(mode, options, dynamic)
  → 否则: 创建_TorchCompileWrapper(backend, mode, options, dynamic)
  → 返回: torch._dynamo.optimize(backend=backend, nopython=fullgraph)(model)

_TorchCompileInductorWrapper (__init__.py:1611-1651):
  → 存储mode(default/reduce-overhead/max-autotune) + options
  → 被调用时 → torch._inductor.compile_fx.compile_fx

Backend Registry (backends/registry.py):
  → _BACKENDS dict → 字符串名→CompilerFn
  → lookup_backend(): _lazy_import() → backends/子模块 → 或entry_points外部backend
  → inductor backend注册 (backends/inductor.py:8-16): lazy import compile_fx

★ torch.compile = 3层包装: wrapper→_dynamo.optimize→OptimizedModule
★ Inductor backend是lazy import → 不用时不加载heavy模块!
```

## 2. Dynamo: C-Level Frame Hook → OptimizedModule

```
torch._dynamo.optimize(backend, nopython, dynamic) (eval_frame.py:728-798):
  → get_compiler_fn(backend) → lookup_backend → wrap_backend_debug
  → nopython=True(fullgraph): optimize_assert() → 必须单图
  → 否则: convert_frame.convert_frame(backend) → 允许graph break

_TorchDynamoContext.__call__(model) (eval_frame.py:340-551):
  → isinstance(fn, nn.Module) → OptimizedModule(mod, self) (line 423)
  → OptimizedModule._initialize(): forward = dynamo_ctx(orig_mod.__call__)
  → __getattr__代理到_orig_mod → 属性访问透传

★ C-Level Eval Frame Hook:
  → set_eval_frame(callback) → torch._C._dynamo.eval_frame
  → Python解释器级别拦截 → 每个frame先经过callback
  → callback决定: trace/cache/skip → 最底层控制!
  → 这是Dynamo的核心机制 → 不是Python层面 → C层面!

OptimizedModule (eval_frame.py:176-237):
  → forward = dynamo_ctx(orig_mod.__call__) → 编译后的forward
  → __getattr__ → _orig_mod → 透明属性代理
  → 编译模型和原始模型看起来一样 → 只是forward变了!
```

## 3. Dynamo Tracer: 符号执行Bytecode → FX Graph

```
convert_frame (convert_frame.py:720-760):
  → 允许graph break → try/except fallback
  → convert_frame_assert (263-406): full compilation → 不fallback

_compile() (convert_frame.py:477-676): 核心编排
  1. transform(instructions, code_options):
     → InstructionTranslator(instructions, code, locals, globals, builtins, ...)
     → tracer.run() → 迭代bytecode指令
     → output = tracer.output → OutputGraph
     → instructions[:] = output.output_instructions → 替换原始bytecode!

  2. transform_code_object(code, transform) → compile_inner:
     → RestartAnalysis → 最多100次restart (569-570)
     → 返回GuardedCode(out_code, check_fn.check_fn) (631)

  3. CheckFunctionManager(output) → 构建guard check函数

InstructionTranslator (symbolic_convert.py:2022-2117): 符号bytecode解释器
  → __init__: OutputGraph + symbolic_locals from f_locals
  → 每个local variable → VariableTracker + Source(LocalSource(k))
  → CALL_FUNCTION (1210): call_function(fn, args, {})
  → 不支持操作 → unimplemented() → graph break

OutputGraph (output_graph.py:222-365): 1:1 with frame
  → tracers: [SubgraphTracer] → 子图追踪
  → ShapeEnv: tracked_fakes, allow_dynamic_output_shape_ops
  → FakeTensorMode: shape_env → 控制symbolic shape
  → 遇到tensor操作 → wrap_fx_proxy → FX proxy node

★ Graph Break:
  → unimplemented() → Unsupported → convert_frame catch → eager fallback
  → Dynamo插入graph break → 编译部分+eager部分+继续编译
  → resume_execution.py → ContinueExecutionCache → break后恢复
  → 最大64个cache entry → 超过fallback eager
```

## 4. AOTAutograd: Joint Forward+Backward → Partition → Compile

```
aot_autograd (backends/common.py:15-62): Dynamo→Inductor桥梁
  → bw_compiler = wrapped fw_compiler + disable()
  → aot_module_simplified(gm, example_inputs)
  → 返回disable(cg) → 防止Dynamo重trace编译输出

create_aot_dispatcher_function (aot_autograd.py:386-505): 主入口
  1. Decompositions: 合并aot_autograd_decompositions + 用户decompositions
  2. Fake mode: FakeTensorMode(shape_env=shape_env)
  3. Input processing: real→fake tensors, SymInt for dynamic shapes
  4. Autograd check: needs_autograd → requires_grad
  5. Joint graph tracing: make_fx → trace joint fwd+bwd graph
  6. Partitioning: partition_fn → default_partition/min_cut_rematerialization
  7. Compilation: fw_compiler(fw_gm) + bw_compiler(bw_gm) → compile_fx
  8. Wrapping: torch.autograd.Function → backward自动调用

★ Partitioners (partitioners.py):
  → default_partition → 简单切分fwd/bwd
  → min_cut_rematerialization_partition → min-cut算法
  → 决定哪些forward节点在backward中recompute → 省内存→费计算
  → 这就是gradient checkpointing的自动版本!

★ FunctionalTensor (functional_tensor.py:13-80):
  → functionalization wrapper → 消除mutation
  → _mode_key = TorchDispatchModeKey.FUNCTIONAL → dispatch key
  → add_() → add() → 不可变操作 → partition正确
  → AOTAutograd需要mutation-free → joint graph才能partition!
```

## 5. Inductor Backend: FX → IR → Scheduler → Codegen → Kernel

```
compile_fx (compile_fx.py:942-991): 主入口
  → config_patches → 递归重调用
  → config.cpp_wrapper → fake inputs + cpp_wrapper=True

compile_fx_inner (compile_fx.py:250-379):
  → count_calls(gm.graph)==0 → make_boxed_func(gm.forward)
  → config.fx_graph_cache → FxGraphCache.load(fx_codegen_and_compile)
  → 否则 → fx_codegen_and_compile(gm, example_inputs)

fx_codegen_and_compile (compile_fx.py:452-563): 核心Pipeline!
  1. view_to_reshape(gm) → .view()→.reshape() → layout兼容
  2. FakeTensorProp(gm, example_inputs) → shape传播
  3. post_grad_passes(gm) → memory planning + folding
  4. GraphLowering(gm, example_inputs, shape_env) → 创建lowering上下文
  5. graph.run(*example_inputs) → FX interpreter → 每op调lowering → IR
  6. graph.compile_to_fn() → Scheduler + codegen → compiled fn

★ GraphLowering (graph.py:111-519): FX Interpreter→IR
  → extends torch.fx.Interpreter → run_node逐node处理
  → call_function → aten op → lowerings[op] → TensorBox/Buffer IR

★ Inductor IR层次 (ir.py:80-139):
  TensorBox → 顶层 → maps to torch.Tensor output
  StorageBox → Layout概念 → points to Buffer
  Buffer → Simple 1D allocation
  View → indirection → TensorBox→View→StorageBox→Buffer
  Chain: TensorBox→StorageBox→Buffer (own storage)
  Chain: TensorBox→View→StorageBox→Buffer (view)
  Mutation functionalized: StorageBox pointer swings → new Buffer → old unchanged

★ Lowering (lowering.py:61-80):
  lowerings = {} → global dict → aten op→lowering function
  make_fallback_handler → ExternKernelNode → 直接调用op
  每个lowering: TensorBox inputs → TensorBox outputs

★ Scheduler (scheduler.py:1184-1244):
  1. nodes = [create_scheduler_node(n)] → IR→scheduler node
  2. compute_dependencies() → 依赖图
  3. topological_sort_schedule() → 排序
  4. dead_node_elimination() → 死代码消除
  5. compute_ancestors() → 祖先追踪
  6. fuse_nodes() → 10轮迭代融合!

  Scheduler.codegen() (2169):
  → Template nodes → codegen_template
  → Extern nodes → codegen_extern_call
  → Fused nodes → codegen_nodes → 单个Triton kernel
  → Normal nodes → TritonScheduling/CppScheduling

★ Codegen:
  Triton (codegen/triton.py:68):
    → TritonPrinter extends PythonPrinter → sympy→Triton表达式
    → @triton.jit kernel + grid配置

  Wrapper (codegen/wrapper.py):
    → WrapperCodeGen → Python wrapper → 调kernel + buffer allocation + CUDA graph

  compile_to_module() (graph.py:1062-1087):
    → codegen() → scheduler → wrapper_code.generate()
    → PyCodeCache.write(code) → 写磁盘
    → PyCodeCache.load_by_key_path() → 加载module
    → 返回module.call → compiled function

★ CompiledFxGraph (compile_fx.py:819-892):
  → compiled_artifact → callable
  → cache_key + artifact_path + cache_linemap → 磁盘缓存
  → guards_expr → FxGraphCache验证 → symbolic shape
  → __call__: get_current_callable()(inputs)
```

## 6. Cache + Guard System

```
★ Dynamo Guard (guards.py:955-1023):
  CheckFunctionManager:
    → GuardBuilder → id_ref + source_ref + lookup_weakrefs + scope
    → guard.create(builder) → 生成guard check code
    → compile_check_fn(builder, guards) → Python guard check函数

  Guard类型:
    → Type: ___check_type_id(arg, <id>)
    → Identity: ___check_obj_id(arg, <id>)
    → Shape: tensor shape/dtype/device
    → Global: grad_mode/deterministic_algorithms/default_device
    → SHAPE_ENV: symbolic shape constraints

★ GuardedCode (types.py:47-49):
  → code: CodeType → 修改后的bytecode
  → check_fn: GuardFn → guard检查函数

  执行流程:
    1. C-level eval_frame → check cache linked list
    2. 每个CacheEntry → check_fn → guards pass/fail
    3. Pass → 执行cached compiled code
    4. Fail → recompile or fallback eager

★ Cache Size (cache_size.py:68-149):
  → CacheSizeRelevantForFrame → same id_match objects
  → compute_cache_size → walk linked list
  → cache_size_limit=64 → 超过→fallback eager

★ FxGraphCache (codecache.py:636-715):
  → Hash GraphModule+inputs+config → key
  → CompiledFxGraph pickle → <cache_dir>/fxgraph/<key>/
  → guards_expr → symbolic shape validation
  → Pass → cached graph → Fail → 重新编译

★ PyCodeCache (codecache.py:1851+):
  → cache: Dict[str, ModuleType] → 内存缓存
  → write(code) → 磁盘 → (key, path)
  → load_by_key_path() → 加载module

★ Cache Invalidation:
  → Guard fail → recompile → new GuardedCode → cache linked list
  → 最多64次 → fallback eager
  → ___compile_config_hash → compile config不变验证
```

## 7. Dynamic Shapes: Symbolic Dimensions + Recompilation

```
★ ShapeEnv in OutputGraph (output_graph.py:275-287):
  → tracked_fakes + allow_scalar_outputs + allow_dynamic_output_shape_ops
  → FakeTensorMode(shape_env=shape_env)
  → Symbolic dimension variables: s0, s1, ... → constraints between them

★ Dynamic Shape Modes:
  dynamic=True: assume_static_by_default=False → 尝试生成dynamic kernel upfront
  dynamic=False: automatic_dynamic_shapes=False → 始终specialize sizes
  dynamic=None(default): current config defaults → automatic detection

★ AOTAutograd Dynamic (aot_autograd.py:449-493):
  → 整数输入: SymInt → shape_env.create_symbol → symbolic
  → Tensor输入: fake_mode.from_tensor(static_shapes=False)
  → Params/buffers: static_shapes=True → weight shape不变!

★ Inductor Symbolic (graph.py:114-147):
  → reuse_shape_env: convert_shape_to_inductor → reuse Dynamo symbolic
  → 否则: ShapeEnv.create_symbolic_sizes_strides_storage_offset → new symbolic
  → 返回sympy expressions → dynamic kernel generation

★ Recompilation Flow:
  1. Guard fail (e.g., tensor.size()[2]==128 → 不再成立)
  2. get_and_maybe_log_recompilation_reason() → log
  3. _compile() → 新shape → 新GuardedCode → cache linked list
  4. automatic_dynamic_shapes=True → 改变维度标记为dynamic
  5. 超过cache_size_limit → fallback eager

★ ★ 关键: recompilation风暴 → LLM训练最大性能杀手!
  → seq_len变化 → 每次recompile → 64次后fallback → 性能暴跌!
  → 解决: dynamic=True → upfront dynamic → 避免recompilation
  → FSDP2+compile → 每个FSDP unit内shape固定 → 减少recompile
  → PyTorch 2.7 → 编译速度2x → 2.8→零配置 → 正式解决!
```

## 8. FSDP2 + compile Integration

```
★ 关键设计: FSDP hooks intentionally create graph breaks!

_dynamo_utils.py (_annotate_modules_for_dynamo):
  → submodule._is_fsdp_managed_module = True → 标记FSDP管理模块
  → submodule._fsdp_use_orig_params = use_orig_params
  → Dynamo skips所有FSDP code → FSDP hooks eager运行
  → FSDP hooks = complex distributed ops (AllGather/ReduceScatter) → 不应被trace
  → FSDP模块 = UnspecializedNNModule → thorough guards
  → 新parameter views (FSDP unshard/reshard每次创建) → 每次fresh capture

fully_shard (composable/fully_shard.py:40-80):
  → @contract(state_cls=_FSDPState) → composable API
  → _annotate_modules_for_dynamo(module, ignored_modules, True)
  → 注册pre/post forward hooks → unshard/reshard

★ ★ 推荐模式: fully_shard(model) → 然后 torch.compile(model)
  → FSDP hooks创建intentional graph breaks
  → compile优化每个FSDP unit内的compute
  → 这样: AllGather/ReduceScatter eager → compute compiled → 最优组合!

★ 为什么不compile FSDP AllGather/ReduceScatter?
  → 这些操作涉及分布式通信 → NCCL → 不应被Triton kernel替换
  → FSDP hooks动态创建parameter views → 每次forward不同 → trace会不稳定
  → 让compile只优化compute → 最安全最稳定!

★ FSDP2+compile实测(H100):
  → compile: +15-16% vs eager
  → compile+Float8: +48-50% vs eager
  → 128GPU: 90% scaling vs ZeRO-3 75-80%
  → ★ 这是PyTorch 2026年最重要的训练优化组合!
```

## 9. 完整执行流程图

```
torch.compile(model, backend="inductor")
│
├─ torch/__init__.py:1710 → _TorchCompileInductorWrapper
├─ torch._dynamo.optimize(backend) → _TorchDynamoContext
├─ _TorchDynamoContext.__call__(model) → OptimizedModule
│   ├─ OptimizedModule._initialize() → forward = dynamo_ctx(orig_mod.__call__)
│   │
│   [第一次forward调用]
│   ├─ C-level set_eval_frame(callback) → 拦截Python frame
│   ├─ _convert_frame_assert(frame, ...)
│   │   ├─ has_tensor_in_frame → 检查torch ops
│   │   ├─ _compile(code, globals, locals, builtins, compiler_fn, ...)
│   │   │   ├─ InstructionTranslator.__init__ → OutputGraph + symbolic_locals
│   │   │   ├─ tracer.run() → 符号执行bytecode
│   │   │   │   ├─ CALL_FUNCTION → call_function → VariableTracker
│   │   │   │   ├─ Tensor ops → wrap_fx_proxy → FX node
│   │   │   │   ├─ 不支持 → unimplemented() → graph break
│   │   │   ├─ OutputGraph.call_user_compiler(gm) → invoke backend
│   │   │   │   ├─ compiler_fn(gm, example_inputs) = aot_autograd(gm, ...)
│   │   │   │   │   ├─ aot_module_simplified(gm, ...)
│   │   │   │   │   │   ├─ create_aot_dispatcher_function(flat_fn, flat_args, aot_config)
│   │   │   │   │   │   │   ├─ make_fx(joint_fn, ...) → trace joint fwd+bwd graph
│   │   │   │   │   │   │   ├─ partition_fn(joint_graph) → split fwd/bwd
│   │   │   │   │   │   │   ├─ fw_compiler(fw_gm, fw_inputs) = compile_fx(fw_gm)
│   │   │   │   │   │   │   │   ├─ compile_fx_inner(gm, ...)
│   │   │   │   │   │   │   │   │   ├─ fx_codegen_and_compile(gm, ...)
│   │   │   │   │   │   │   │   │   │   ├─ view_to_reshape → layout兼容
│   │   │   │   │   │   │   │   │   │   ├─ FakeTensorProp → shape传播
│   │   │   │   │   │   │   │   │   │   ├─ post_grad_passes → memory/folding
│   │   │   │   │   │   │   │   │   │   ├─ GraphLowering(gm, ...) → lowering context
│   │   │   │   │   │   │   │   │   │   ├─ graph.run(*example_inputs) → FX→IR
│   │   │   │   │   │   │   │   │   │   │   ├─ 每个aten op → lowering → TensorBox/Buffer
│   │   │   │   │   │   │   │   │   │   ├─ graph.compile_to_fn()
│   │   │   │   │   │   │   │   │   │   │   ├─ Scheduler(buffers) → fusion + scheduling
│   │   │   │   │   │   │   │   │   │   │   ├─ scheduler.codegen() → kernel generation
│   │   │   │   │   │   │   │   │   │   │   ├─ wrapper_code.generate() → Python wrapper
│   │   │   │   │   │   │   │   │   │   ├─ compile_to_module() → PyCodeCache → load
│   │   │   │   │   │   │   │   │   ├─ CompiledFxGraph(compiled_fn, graph, output_strides)
│   │   │   │   │   │   │   ├─ bw_compiler(bw_gm) → 同上流程
│   │   │   │   │   │   │   ├─ torch.autograd.Function → backward自动调用
│   │   │   │   ├─ CheckFunctionManager(output) → guard check function
│   │   │   │   ├─ GuardedCode(out_code, check_fn) → bytecode+guards
│   │   │   │
│   [后续forward调用]
│   ├─ C-level eval_frame → check guards
│   ├─ Guards pass → 执行cached compiled code
│   ├─ Guards fail → recompile or fallback eager
```

## 10. 关键设计洞察

```
1. C-Level Frame Hook → Python解释器级别拦截 → 最底层控制!
   → 不是Python decorator/wrapper → 是C层面frame evaluation拦截
   → 这使得Dynamo能看到所有Python操作 → 不仅是torch ops
   → 这也使得Dynamo非常脆弱 → Python版本变化可能破坏!
   → PyTorch 2.7 → 更稳定的实现 → 但仍然是C-level

2. Symbolic Bytecode Execution → 不运行代码 → 只模拟!
   → InstructionTracker → VariableTracker → 不执行真实操作
   → 只记录"如果执行会产生什么FX node" → 符号执行
   → Graph break = 遇到无法符号执行的操作 → fallback eager
   → 这是编译器经典技术 → Lisp/ML编译器也用类似方法

3. AOTAutograd joint fwd+bwd → 一次trace → 自动partition!
   → make_fx trace整个forward+backward → joint graph
   → partition → min-cut → 自动gradient checkpointing
   → 这比手动gradient checkpointing更优 → 算出最优recompute策略!
   → 但: partition本身有开销 → 大模型可能慢 → 2.7改进!

4. Inductor IR = TensorBox→StorageBox→Buffer → functionalized mutation!
   → mutation: StorageBox pointer swings → new Buffer → old unchanged
   → 这是functionalization的核心 → 不可变IR → 可优化
   → 与Lua Torch/TorchScript不同 → Dynamo用Python原生 → 不需要IR表示整个程序
   → Inductor IR只表示kernel内部 → 更简单更可优化!

5. Scheduler 10轮迭代融合 → 最大化kernel fusion!
   → can_fuse: 共享数据+同device+顺序正确 → 判断可融合
   → 10轮迭代 → 反复尝试 → 直到收敛
   → GEMM → ExternKernel → 不融合 → 只融合epilogue(RMSNorm+SiLU+Residual)
   → 这就是为什么compile能加速2-3x → 融合减少kernel launch!

6. FSDP2+compile → intentional graph breaks → 最优组合!
   → FSDP hooks(eager) → AllGather/ReduceScatter → distributed ops → 不应compile
   → compute(compiled) → attention+MLP → Triton kernel → 最优
   → graph break不是坏事 → 是intentional design → 安全边界!
   → +15-16% vs eager → +48-50% with Float8 → 最强组合!

7. Dynamic Shapes → recompilation风暴 → LLM训练最大挑战!
   → seq_len变化 → 每次recompile → 64次后fallback → 性能暴跌
   → dynamic=True → upfront dynamic → 避免recompilation → 但kernel可能慢
   → FSDP2 → 每unit内shape固定 → 减少recompile → 间接解决!
   → PyTorch 2.8 → 零配置 → 自动dynamic → 最终解决!

8. PyCodeCache → 磁盘缓存 → 编译结果持久化!
   → 第一次compile → 写磁盘 → 后续加载 → 不重新编译
   → FxGraphCache → FX graph level缓存 → guard验证 → symbolic shape
   → 这使得compile只在shape变化时重新编译 → 大部分时间用缓存!
   → 但: 第一次compile慢 → cold start问题 → 2.7编译速度2x改进!
```

---

Sources:
- PyTorch 2.2.2 source (ai-infra conda env): torch/__init__.py, torch/_dynamo/, torch/_inductor/, torch/_functorch/
- Key files: eval_frame.py, convert_frame.py, symbolic_convert.py, output_graph.py, guards.py, aot_autograd.py, partitioners.py, compile_fx.py, graph.py, ir.py, lowering.py, scheduler.py, codegen/
- torch/distributed/fsdp/_dynamo_utils.py, torch/distributed/_composable/fully_shard.py
- Background agent research (PyTorch compile e2e trace)
