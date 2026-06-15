# PyTorch FX IR 源码级深度阅读

> 2026-06-15 | 源码: torch/fx/graph.py + node.py + interpreter.py + graph_module.py + torch/_dynamo/output_graph.py
> 核心: Node=6种op(placeholder/call_function/call_method/call_module/get_attr/output)+name+target+args+kwargs+users+_input_nodes+meta→Graph=双向链表+_root sentinel+create_node+erase_node+inserting_before/after+CodeGen→Interpreter=逐Node执行→env dict+garbage_collect_values→GraphModule=nn.Module+__new__ singleton+recompile()→OutputGraph=Dynamo输出+1:1 frame+SubgraphTracer+GuardsSet+GraphRegionTracker

## 1. Node — FX IR的基本单元

```
★ ★ Node (node.py:238): 继承_NodeBase → C++ binding → 性能优化

核心字段:
  name: str → 唯一名称 → 自动命名(基于target)
  op: str → 6种操作类型 → _legal_ops dict验证
  target: Target → callable或str → 操作目标
  args: tuple[Argument] → 位置参数 → 可包含其他Node(数据流)
  kwargs: dict[str, Argument] → 关键字参数
  _input_nodes: dict[Node, None] → 所有Node-valued输入 → 有序集(dict as ordered set)
  users: dict[Node, None] → 使用此Node值的所有下游Node → 反向引用!
  meta: dict[str, Any] → 元数据 → 保留跨Node copy → shape/dtype/device/val等
  type: Any | None → 类型表达式 → 用于生成代码的类型注解
  _sort_key: Any → 拓扑排序键 → 确保执行顺序正确

★ ★ 6种op类型详细语义:
  1. placeholder: 函数输入 → name=参数名 → target=参数名 → args=默认值或空
  2. get_attr: 模块属性获取 → target=全限定名(foo.bar.baz) → args/kwargs=空
  3. ★ ★ call_function: 自由函数调用 → target=callable(torch.add/operator.add) → args/kwargs=参数
  4. call_method: 方法调用 → target=方法名(str "relu") → args包含self → kwargs=参数
  5. call_module: 模块forward调用 → target=全限定名("linear") → args不含self → kwargs=参数
  6. output: 函数输出 → args[0]=返回值 → 对应return语句

★ ★ Argument类型系统 (node.py:40-66):
  BaseArgumentTypes = str|int|float|bool|complex|dtype|Tensor|device|SymInt|SymBool|SymFloat
  Argument = Optional[tuple[Argument]|Sequence|Mapping|slice|range|Node|BaseArgumentTypes]
  → ★ ★ Node可以出现在Argument中 → 表示数据流依赖 → 图中的边!
  → ★ SymInt/SymBool/SymFloat → symbolic shapes → torch.compile动态shape支持!

★ ★ Node数据流:
  _input_nodes → 所有作为args/kwargs传入的其他Node → 依赖关系
  users → 使用此Node值的下游Node → 反向依赖
  → ★ _input_nodes + users = 双向依赖图 → 支持拓扑排序和死代码消除!
```

## 2. Graph — FX IR的容器

```
★ ★ Graph (graph.py:1312): 双向链表 → _root sentinel → 增删改查

初始化 (__init__):
  _root: Node = Node(self, "", "root", "", (), {}) → sentinel → 不执行
  _used_names: dict[str, int] → 名称计数 → 唯一命名
  _insert: _root.prepend → 默认插入位置 → prepend到sentinel前 = 添加到末尾!
  _len: int → Node计数
  _graph_namespace: _Namespace → 唯一命名空间
  _codegen: CodeGen → 代码生成器
  _find_nodes_lookup_table: _FindNodesLookupTable → 快速查找

★ ★ Node操作:
  create_node(op, target, args, kwargs, name, type_expr) → 创建并插入Node
  erase_node(to_erase) → 删除Node → 更新所有用户的args → 替换或报错
  inserting_before(n) → _InsertPoint → 临时修改插入位置 → context manager
  inserting_after(n) → _InsertPoint → 同上 → after模式

★ ★ 6种Node创建方法 (便捷API):
  placeholder(name, type_expr, default_value) → create_node("placeholder", ...)
  get_attr(qualified_name, type_expr) → create_node("get_attr", ...)
  call_function(the_function, args, kwargs, type_expr, name) → create_node("call_function", ...)
  call_method(method_name, args, kwargs, type_expr, name) → create_node("call_method", ...)
  call_module(module_name, args, kwargs, type_expr, name) → create_node("call_module", ...)
  output(args, type_expr) → create_node("output", ...)

★ ★ ★ create_size_node / create_stride_node / create_storage_offset_node:
  → torch.ops.aten.sym_size.int → SymInt → 动态shape
  → torch.ops.aten.sym_stride.int → SymInt → 动态stride
  → torch.ops.aten.sym_storage_offset.default → SymInt → 动态offset
  → ★ ★ 这些是torch.compile动态shape的核心 → FX IR用SymInt表示!

★ ★ CodeGen (graph.py:360):
  _sym_repr: Callable → SymNode打印器 → 可自定义
  _body_transformer: TransformCodeFunc | None → 代码变换 → 修改生成代码
  _func_name: str = "forward" → 默认函数名
  → gen_fn_def(free_vars, return_annotation) → 生成函数定义
  → ★ CodeGen = FX Graph → Python代码 → 可执行! → GraphModule用CodeGen生成forward

★ ★ Graph.print_tabular (2525):
  → 打印表格格式 → name|op|target|args|kwargs → 调试用
  → lint() → 验证Graph合法性 → 检查Node依赖、命名、op类型
```

## 3. Interpreter — FX图的执行器

```
★ ★ Interpreter (interpreter.py:52): 逐Node执行 → env dict → 可覆盖方法

核心执行流程:
  run(*args, initial_env, enable_io_processing):
    → 1. args_iter = iter(args) → 位置参数迭代器
    → 2. for node in self.graph.nodes: → 拓扑顺序遍历
    → 3. env[node] = run_node(node) → 执行并存储结果
    → 4. garbage_collect_values → 删除不再使用的值 → 省内存!
    → 5. output节点 → return env[node] → 函数返回

★ ★ run_node(n) → getattr(self, n.op)(n.target, args, kwargs):
  → placeholder → next(args_iter) → 消耗一个参数
  → get_attr → 从self.module获取属性
  → call_function → 直接调用target(*args, **kwargs)
  → call_method → args[0].target(*args[1:], **kwargs) → self方法调用
  → call_module → self.submodules[target](args, kwargs) → 模块forward
  → output → 返回args → 函数结束

★ ★ ★ garbage_collect_values → 关键内存优化!
  → 反向遍历graph → 记录每个Node的最后一次使用 → node_to_last_use dict
  → 执行Node后 → 删除env中不再使用的Node → del env[to_delete]
  → ★ ★ 这确保中间值及时释放 → 执行时内存最优!
  → 可禁用 → garbage_collect_values=False → 检查所有中间值

★ ★ boxed_run → 高效参数传递!
  → args_list直接传入 → 执行后clear → 输入tensor及时释放
  → placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
  → env = dict(zip(placeholder_nodes, args_list)) → 一对一映射
  → ★ ★ boxed_run避免拷贝参数 → 输入tensor更快释放 → 生产级优化!

★ ★ Interpreter扩展模式:
  → 子类化 → 覆盖call_function/call_method → 变换op → 不修改Graph!
  → example: NegSigmSwapInterpreter → swap torch.neg ↔ torch.sigmoid
  → ★ ★ 这是FX的核心设计 → Interpreter=Pass → 不修改IR → 只变换执行!
  → 常见Pass: ShapeProp → 推断shape → 填充Node.meta['val']
```

## 4. GraphModule — FX图的可执行模块

```
★ ★ ★ GraphModule (graph_module.py:511): nn.Module + fx.Graph → 可执行!

__new__ → singleton子类模式:
  → 每个GraphModule实例需要独立的forward方法
  → 创建GraphModuleImpl(cls)子类 → 继承用户定义类
  → 遍历MRO → 找到第一个不是GraphModuleImpl的类 → 正确继承!
  → ★ 解决issue #63883 → 冗余类定义问题 → singleton模式!

__init__(root, graph, class_name):
  → root: nn.Module或dict → 模块或属性映射
  → graph: fx.Graph → 包含要执行的Node
  → 从root复制子模块到GraphModule → _copy_attr
  → self.training = root.training → 保留训练模式

★ ★ ★ recompile() → 核心方法:
  → Graph → Python代码 → forward方法 → self.forward = generated_code
  → 编辑graph后必须调用 → 否则forward不反映最新graph!
  → CodeGen → python_code → self.code → exec → self.forward
  → ★ ★ Graph → code → exec → forward → FX IR变成可执行Python!

★ ★ 关键设计:
  → graph属性赋值 → 自动recompile → 但修改graph内容不触发 → 手动recompile()
  → code属性 → 存储生成的Python代码 → 可读 → 可调试
  → forward属性 → 存储可执行函数 → 从code生成
  → ★ GraphModule = IR → Code → Execution → 完整pipeline!
```

## 5. OutputGraph — Dynamo的输出容器

```
★ ★ OutputGraph (output_graph.py:640): Dynamo输出 → 1:1 with frame → InstructionTranslator写入

★ ★ OutputGraphGuardsState (437): Guard管理 → 重编译触发器
  local_scope: Scope → 局部变量 → 函数内
  global_scope: Scope → 全局变量 → 模块级
  torch_function_mode_stack → torch function mode栈 → __torch_function__覆盖
  guard_on_key_order: set → dict key顺序guard → 防止key顺序变化
  input_source_to_sizes_strides: dict → 输入shape/stride → 动态shape guard
  _guards: GuardsSet → 所有guard集合 → 重编译条件
  _aotautograd_guards: list → AOTAutograd专用guard

★ ★ OutputGraphCommon (565): 最小接口 → 后端兼容
  import_sources: dict → 依赖映射
  shape_env: ShapeEnv → symbolic shape环境 → sympy推理
  export_metadata: ExportMetaData → export专用元数据
  tracked_fakes_id_to_source: dict → fake tensor → Source映射 → trace来源追踪
  output_pycode: list → 生成代码 → Python bytecode

★ ★ OutputGraph初始化 (640+):
  tracers: [SubgraphTracer(self, is_export=export)] → 子图追踪器
  input_source_to_var: dict[Source, VariableTracker] → 输入源→变量追踪器 → 去重!
  leaf_var_creation_order: list → 叶子tensor创建顺序 → backward()自动检测
  cleanup_hooks: list → 清理回调 → 编译后执行
  compile_id: int → 编译ID → 全局计数器
  installed_globals: set → 安装的全局变量 → 避免重复安装
  cudagraph_annotation → CUDA graph标注 → 内联时设置
  region_tracker: GraphRegionTracker → 图区域追踪 → 分段编译

★ ★ ★ Dynamo → OutputGraph → FX Graph 流程:
  1. InstructionTranslator逐指令翻译 → Python bytecode → FX Node
  2. 每个指令 → call_function/call_method等 → 写入OutputGraph.graph
  3. Graph break → 新OutputGraph → 多段 → 分别编译
  4. GuardsSet → 条件集合 → 输入变化时重编译 →最多64次→fallback
  5. OutputGraph → GraphModule → recompile() → 可执行forward

★ ★ Guards机制 — 重编译触发:
  → GuardsSet: 所有guard → 输入shape变化/dtype变化/device变化/key顺序变化
  → 每次编译 → 记录输入特征 → guard
  → 下次输入 → 检查guards → 是否匹配 → 不匹配则重编译
  → ★ ★ 最多64次重编译 → Config.cache_size_limit → fallback到eager模式
  → ★ 这是torch.compile的关键权衡 → 编译开销 vs 执行加速
```

## 6. FX IR与Dynamo/Inductor的交互

```
★ ★ ★ 完整pipeline (torch.compile):

Dynamo → FX Graph → AOTAutograd → Inductor:
  1. Dynamo: eval_frame hook → 符号执行bytecode → OutputGraph → FX Graph
  2. AOTAutograd: FX Graph → joint fwd+bwd graph → min-cut partition → fwd+bwd分离
  3. Inductor: FX Graph → IR(TensorBox→StorageBox→Buffer) → Scheduler → Triton/C++ codegen

★ ★ FX Graph在pipeline中的角色:
  → Dynamo输出: fx.Graph → 包含aten ops → call_function节点
  → AOTAutograd输入: fx.Graph → 添加backward节点 → joint graph
  → Inductor输入: fx.Graph → 转换为Inductor IR → 生成Triton kernel
  → ★ ★ FX Graph = Dynamo和Inductor之间的桥梁 → 标准化IR!

★ ★ FX Graph → Inductor IR转换:
  → call_function(torch.ops.aten.add.Tensor) → Inductor IR → Pointwise kernel
  → call_function(torch.ops.aten.mm.default) → Inductor IR → ExternKernel(cuBLAS)
  → call_function(torch.ops.aten.sum.dim_IntList) → Inductor IR → Reduction kernel
  → ★ ★ Inductor根据op类型 → 选择不同IR类型 → 决定kernel类型

★ ★ FX Graph → Python Code → Execution:
  → GraphModule → CodeGen → python_code → exec → forward函数
  → ★ 这是fallback路径 → 编译失败 → 用Python code执行 → eager模式
  → 正常路径: FX Graph → Inductor → Triton kernel → GPU执行 → 更快!

★ ★ ★ FSDP2 + FX Graph:
  → FSDP2 @contract → MRO insertion → per-param DTensor
  → FSDP2 hooks → RegisterPre/PostBackwardFunction → intentional graph breaks
  → ★ ★ FSDP2故意graph break → 在AllGather/ReduceScatter处break → compile兼容
  → Dynamo → 多段FX Graph → 每段独立编译 → 合并执行
  → 这是FSDP2+compile的核心设计 → graph break不crash → 正确分段!
```

## 7. 关键设计洞察

```
1. Node 6种op → 完整覆盖Python函数语义 → FX IR完备!
   → placeholder/call_function/call_method/call_module/get_attr/output
   → ★ 完全覆盖Python函数 → 输入/计算/属性/输出 → 任何函数都能表示!

2. _input_nodes + users → 双向依赖 → 支持拓扑排序+死代码消除!
   → _input_nodes: 此Node依赖谁 → 上游
   → users: 此Node被谁依赖 → 下游
   → ★ 双向 → 正向遍历=执行 → 反向遍历=死代码消除 → 两用!

3. Interpreter → Pass模式 → 不修改IR → 只变换执行 → 极妙!
   → 子类化 → 覆盖call_function → 变换op → 不修改Graph
   → ShapeProp → 推断shape → 填充meta → 不修改Graph
   → ★ ★ 这是编译器设计经典模式 → IR不变 → Pass变换 → 多Pass流水线!

4. GraphModule → IR→Code→Execution → 完整闭环!
   → Graph → CodeGen → python_code → exec → forward → 可执行!
   → 编辑Graph → recompile() → 新forward → 立即生效
   → ★ ★ IR→Code→Execution = 编译器核心pipeline → FX实现完整!

5. SymInt/SymBool/SymFloat → 动态shape → torch.compile核心!
   → Node.args中SymInt → 代表动态维度 → 运行时才确定
   → Guards → 记录shape特征 → 变化时重编译 →最多64次
   → ★ ★ 动态shape = torch.compile最大挑战 → Guards+重编译 = 解决方案!

6. OutputGraph → 1:1 with frame → InstructionTranslator写入 → Dynamo核心!
   → 每个Python帧 → 一个OutputGraph → 独立编译
   → Graph break → 新OutputGraph → 多段 → 分别编译 → 合并
   → ★ ★ Dynamo分段编译 → 遇到不支持op → break → 继续下段 → 不全失败!

7. boxed_run → 高效参数传递 → 输入tensor及时释放 → 生产级!
   → args_list直接传入 → 执行后clear → 及时释放GPU内存
   → vs normal run → args_iter → 消耗后仍在内存 → 可能泄漏
   → ★ ★ 生产级优化 → 输入tensor更快释放 → 内存更优!

8. garbage_collect_values → Interpreter执行时内存最优!
   → 反向遍历 → 记录最后使用 → 执行后立即删除 → 省内存
   → vs 不删除 → 所有中间值保留 → 内存持续增长
   → ★ ★ 这和vLLM persistent buffer理念类似 → 及时释放无用值 → 内存最优!

9. FSDP2 intentional graph breaks → FSDP hooks处break → compile兼容!
   → RegisterPre/PostBackwardFunction → graph break → 多段编译
   → 每段独立 → 合并执行 → 不crash → compile兼容
   → ★ ★ 这是FSDP2+compile的核心 → vs ZeRO-3不支持compile → FSDP2是未来!

10. FX IR = Dynamo和Inductor之间的桥梁 → 标准化中间表示!
    → Dynamo输出FX → AOTAutograd修改FX → Inductor消费FX → 统一
    → ★ 类似LLVM IR → 多种前端 → 多种后端 → FX = PyTorch的LLVM IR!
    → vs TorchScript → 已deprecated → FX + Dynamo + Inductor = PyTorch未来!
```

---

Sources:
- torch/fx/node.py (Node class, 238-350: op types, fields, _input_nodes, users, meta)
- torch/fx/graph.py (Graph class, 1312+: create_node, erase_node, inserting_before/after, CodeGen, python_code)
- torch/fx/graph.py (CodeGen class, 360+: gen_fn_def, _body_transformer)
- torch/fx/interpreter.py (Interpreter class, 52-170: run, run_node, placeholder, call_function, garbage_collect_values, boxed_run)
- torch/fx/graph_module.py (GraphModule class, 511-580: __new__ singleton, __init__, recompile)
- torch/_dynamo/output_graph.py (OutputGraphGuardsState, 437: guards; OutputGraphCommon, 565: minimal interface; OutputGraph, 640: Dynamo output)
