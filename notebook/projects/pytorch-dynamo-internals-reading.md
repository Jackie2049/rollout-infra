# PyTorch torch._dynamo 内部机制源码级深度阅读

> 研究日期: 2026-06-15 | PyTorch main branch | 源码版本: 最新
> 核心问题: C-level eval_frame hook → VariableTracker → InstructionTranslator → Guard → FX Graph

---

## 1. C-level eval_frame Hook: Python字节码拦截机制

### 1.1 核心拦截: `_PyInterpreterState_SetEvalFrameFunc`

★ ★ ★ **Dynamo的核心不是Python层面的monkey-patch, 而是直接替换CPython解释器的帧评估函数!**

```
// torch/csrc/dynamo/eval_frame.c

static PyObject* dynamo__custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  // 三种状态:
  //   None: 禁用Dynamo → 走默认路径
  //   False: run-only模式 → 只查缓存, 不编译
  //   Python callable: 启用Dynamo → 触发编译
  PyObject* callback = eval_frame_callback_get();

  if (Py_IsNone(callback)) {
    return dynamo_eval_frame_default(tstate, frame, throw_flag);
  }
  return dynamo__custom_eval_frame(tstate, frame, throw_flag, callback);
}

// 安装shim: 替换解释器的全局帧评估函数
static void enable_eval_frame_shim(PyThreadState* tstate) {
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &dynamo_custom_eval_frame_shim) {
    previous_eval_frame =
        _PyInterpreterState_GetEvalFrameFunc(tstate->interp);  // 保存原始函数
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &dynamo_custom_eval_frame_shim);       // 替换!
  }
}

// 卸载shim: 恢复原始帧评估函数
static void enable_eval_frame_default(PyThreadState* tstate) {
  _PyInterpreterState_SetEvalFrameFunc(tstate->interp, previous_eval_frame);
  previous_eval_frame = NULL;
}
```

### 1.2 完整拦截数据流

```
torch.compile(model) 调用链:

① torch.compile → _TorchDynamoContext.__init__(callback=compiler_fn)
② _TorchDynamoContext.__enter__() → set_eval_frame(callback)
③ set_eval_frame(C层):
   - 保存旧callback到eval_frame_callback_key (thread-local storage)
   - increment_working_threads → enable_eval_frame_shim
   - _PyInterpreterState_SetEvalFrameFunc(interp, &dynamo_custom_eval_frame_shim)
④ 之后每个Python帧执行时, CPython调用dynamo_custom_eval_frame_shim而非_PyEval_EvalFrameDefault!

每次Python函数调用:
  CPython: _PyEval_EvalFrameDefault() → 被替换为 dynamo_custom_eval_frame_shim()
  ↓
  dynamo__custom_eval_frame_shim():
    callback = eval_frame_callback_get()  // thread-local
    ↓
    None → 直接走 _PyEval_EvalFrameDefault (原始路径)
    False → 走 dynamo__custom_eval_frame (只查缓存)
    callable → 走 dynamo__custom_eval_frame (可能编译)
```

★ **关键发现**: `set_eval_frame`使用PyThread_tss (thread-local storage)存储callback, 不是全局变量! 这意味着每个线程可以独立启用/禁用Dynamo。

### 1.3 Cache Lookup + Callback调用 (C++层)

```
// torch/csrc/dynamo/eval_frame_cpp.cpp: dynamo__custom_eval_frame()

完整流程:

① 获取ExtraState: extra = get_extra_state(F_CODE(frame))
② 解析strategy: SKIP/RUN_ONLY/DEFAULT
③ 尝试fast-path: try_lookup_without_guard_eval(extra, backend, ...)
④ 如果fast-path失败 → lookup(extra, locals, backend, ...)
   - 遍历cache_entry链表
   - 对每个entry调用guard_manager.check(f_locals)
   - 如果guard通过 → 返回cached_code
⑤ cache hit → eval_custom(): dynamo_eval_custom_code(tstate, frame, cached_code)
⑥ cache miss → dynamo_call_callback(callback, frame, locals, cache_entry, frame_state)
   - callback是Python层的ConvertFrameAssert.__call__
   - 返回ConvertFrameReturn(guarded_code, frame_exec_strategy, apply_to_code)
⑦ 如果guarded_code不为None → create_cache_entry(extra, guarded_code, backend)
⑧ eval_custom(): 用新编译的代码执行frame
```

★ **★ 最重要的发现**: C层的`lookup()`函数在C++中执行guard检查, 不需要回到Python! 这是性能关键 - guard检查完全在C/C++层完成, 只有cache miss才调用Python callback。

### 1.4 Thread-local Callback机制

```python
# torch/_dynamo/eval_frame.py

class _TorchDynamoContext:
    def __enter__(self):
        self.prior = set_eval_frame(None)  # 保存旧callback
        _maybe_set_eval_frame(_callback_from_stance(self.callback))
        # set_eval_frame(C层): PyThread_tss_set(eval_frame_callback_key, callback)
        # 同时: increment_working_threads → enable_eval_frame_shim

    def __exit__(self, ...):
        set_eval_frame(self.prior)  # 恢复旧callback
        # decrement_working_threads → enable_eval_frame_default (如果计数=0)
```

★ **Dynamo启停机制**: 用`active_dynamo_threads`计数器控制。当所有线程都退出Dynamo时, 恢复`_PyEval_EvalFrameDefault`, 完全零开销!

---

## 2. VariableTracker体系: 100+变量追踪类型

### 2.1 VariableTracker基类 (torch/_dynamo/variables/base.py)

```python
class VariableTracker(metaclass=VariableTrackerMeta):
    """所有变量追踪的基类。实例immutable, 修改需要clone()"""

    _cpython_type: type | tuple[type, ...] | None = None  # 对应的CPython类型

    # 核心属性:
    source: Source | None          # 变量来源(LocalSource/GlobalSource/AttrSource)
    source_location: SourceLocation | None  # 源码位置(filename, lineno)
    mutation_type: MutationType | None     # 变更类型

    # 核心方法:
    def as_python_constant(self) -> Any      # 尝试获取Python常量值
    def as_proxy(self) -> Any                # 获取FX proxy
    def reconstruct(self, codegen) -> None   # 重建为Python代码
    def call_function(self, tx, args, kwargs)  # 函数调用
    def call_method(self, tx, name, args, kwargs)  # 方法调用
    def realize(self) -> VariableTracker     # 强制实现(从lazy→real)
    def unwrap(self) -> VariableTracker      # 解包装
    def make_guards(self, guard) -> None     # 创建guards

    # 工厂方法:
    @staticmethod
    def build(tx, value, source=None, realize=False):
        if source is None:
            return SourcelessBuilder.create(tx, value)
        elif realize:
            return VariableBuilder(tx, source)(value)  # 立即构建
        elif type(value) in LazyConstantVariable.supported_types:
            return LazyConstantVariable.create(value, source)  # 延迟常量
        else:
            return LazyVariableTracker.create(value, source, tx=tx)  # 延迟追踪
```

### 2.2 ★ 完整VariableTracker层级树 (100+类)

```
VariableTracker (base.py:320)
├── ConstantVariable (constant.py) — int/float/str/bool/None/tuple常量
├── LazyConstantVariable (lazy.py) — 延迟常量(不被修改时不触发recompile)
├── LazyVariableTracker (lazy.py) — 延迟追踪(首次使用才展开)
│
├── TensorVariable (tensor.py) — torch.Tensor追踪
│   ├── FakeItemVariable — tensor.item()结果
│   ├── DataPtrVariable — tensor.data_ptr()
│   ├── SymNodeVariable — SymInt/SymBool符号值
│   ├── NumpyNdarrayVariable — numpy ndarray
│   ├── UnspecializedPythonVariable — 不特化的Python值(如int→tensor再→int)
│   ├── UntypedStorageVariable — tensor storage
│
├── NNModuleVariable (nn_module.py) — nn.Module追踪
│   ├── UnspecializedNNModuleVariable — 不特化的nn.Module
│   ├── UnspecializedBuiltinNNModuleVariable — 不特化的builtin nn.Module
│   ├── FSDPManagedNNModuleVariable — FSDP管理的nn.Module
│
├── BuiltinVariable (builtin.py) — Python内置函数
│   ├── BaseBuiltinVariable — 基类
│   ├── DictBuiltinVariable — dict()等
│   ├── GetAttrBuiltinVariable — getattr()
│   ├── HasAttrBuiltinVariable — hasattr()
│   ├── IterBuiltinVariable — iter()
│   ├── ListBuiltinVariable — list()等
│   ├── SetAttrBuiltinVariable — setattr()
│
├── UserFunctionVariable (functions.py) — 用户函数
│   ├── BaseUserFunctionVariable — 基类
│   ├── NestedUserFunctionVariable — 内联用户函数
│   ├── UserMethodVariable — 用户方法
│   ├── BoundBuiltinMethodVariable — bound内置方法
│   ├── ClassMethodVariable — classmethod
│   ├── StaticMethodVariable — staticmethod
│   ├── PropertyVariable — property
│   ├── FunctoolsPartialFunction — functools.partial
│   ├── WrapperUserFunctionVariable — 包装函数
│   ├── WrapperUserMethodVariable — 包装方法
│   ├── MethodDescriptorVariable — 方法描述符
│   ├── MethodWrapperVariable — 方法包装器
│   ├── WrapperDescriptorVariable — 包装器描述符
│   ├── ClassMethodDescriptorVariable — 类方法描述符
│   ├── GetSetDescriptorVariable — getset描述符
│   ├── MemberDescriptorVariable — member描述符
│   ├── TupleGetterVariable — operator.itemgetter
│   ├── PolyfilledFunctionVariable — polyfill函数
│   ├── SkipFunctionVariable — 跳过的函数
│   ├── LocalGeneratorFunctionVariable — generator函数
│   ├── LocalGeneratorObjectVariable — generator对象
│   ├── InspectSignatureVariable — inspect.signature
│   ├── CollectionsNamedTupleFunction — namedtuple构造器
│   ├── SparseTensorCreationSkipVariable — sparse tensor创建(跳过)
│   ├── TritonSetAllocatorVariable — Triton allocator设置
│   ├── TMADescriptorExperimental/StableVariable — TMA descriptor
│   ├── CreateTMADescriptorExperimental/StableVariable — TMA创建
│   ├── PyTreeGetNodeTypeFunctionVariable — pytree.get_node_type
│   ├── PyTreeTreeIsLeafFunctionVariable — pytree.tree_is_leaf
│   ├── FunctionDecoratedByContextlibContextManagerVariable — contextlib装饰的函数
│
├── DictVariable体系 (dicts.py):
│   ├── ConstDictVariable — 常量字典(键值已知)
│   ├── DictItemsVariable — dict.items()
│   ├── DictViewVariable — dict view
│   ├── DunderDictVariable — __dict__
│   ├── MappingProxyVariable — MappingProxyType
│   ├── NNModuleHooksDictVariable — nn.Module hooks字典
│
├── ListVariable体系 (lists.py):
│   ├── BaseListVariable — 基类(list/tuple通用)
│   ├── ListVariable — list
│   ├── TupleVariable — tuple
│   ├── ListIteratorVariable — list迭代器
│   ├── TupleIteratorVariable — tuple迭代器
│   ├── RangeVariable — range
│   ├── SliceVariable — slice
│   ├── DequeVariable — collections.deque
│
├── SetVariable体系 (sets.py):
│   ├── SetVariable — set
│   ├── FrozensetVariable — frozenset
│   ├── DictKeySetVariable — dict_keys视图
│   ├── OrderedSetVariable — OrderedSet
│   ├── OrderedSetClassVariable — OrderedSet类
│
├── IteratorVariable体系 (iter.py):
│   ├── IteratorVariable — 通用迭代器
│   ├── CountIteratorVariable — itertools.count
│   ├── FilterVariable — filter
│   ├── MapVariable — map
│   ├── ZipVariable — zip
│   ├── RepeatIteratorVariable — itertools.repeat
│   ├── ItertoolsVariable — itertools模块
│
├── ContextManager体系 (ctx_manager.py):
│   ├── ContextWrappingVariable — 基类
│   ├── GenericContextWrappingVariable — 通用上下文管理器
│   ├── GradModeVariable — torch.no_grad()/torch.enable_grad()
│   ├── InferenceModeVariable — torch.inference_mode()
│   ├── CUDADeviceVariable — CUDA设备上下文
│   ├── XPUDeviceVariable — XPU设备上下文
│   ├── AcceleratorDeviceIndexVariable — accelerator设备索引
│   ├── SDPAKernelVariable — torch.nn.attention.sdpa_kernel
│   ├── CatchWarningsCtxManagerVariable — warnings.catch_warnings
│   ├── DisabledSavedTensorsHooksVariable — no_gradsavedtensorshooks
│   ├── DynamoConfigPatchVariable — Dynamo config patch
│   ├── ErrorOnGraphBreakVariable — error_on_graph_break上下文
│   ├── DualLevelContextManager — dual level上下文
│   ├── CudagraphOverrideVariable — cudagraph override
│   ├── WithEnterFunctionVariable — __enter__函数
│   ├── WithExitFunctionVariable — __exit__函数
│   ├── FSDPParamGroupUseTrainingStateVariable — FSDP训练状态
│   ├── FxTracebackAnnotateVariable — FX traceback注解
│   ├── TemporarilyPopInterpreterStackCtxManagerVariable — 临时pop解释器栈
│   ├── GradIncrementNesting/GradInplaceRequiresGrad — grad嵌套
│   ├── JvpIncrementNestingCtxManagerVariable — JVP嵌套
│   ├── VmapIncrementNestingCtxManagerVariable — vmap嵌套
│   ├── SetFwdGradEnabledContextManager — forward grad
│
├── HigherOrderOp体系 (higher_order_ops.py):
│   ├── TorchHigherOrderOperatorVariable — torch HOP
│   ├── FunctorchHigherOrderVariable — functorch HOP
│   ├── FunctionalCallVariable — functional_call
│   ├── ReparametrizeModuleCallVariable — reparametrize_module_call
│
├── UserDefined体系 (user_defined.py):
│   ├── UserDefinedObjectVariable — 用户自定义对象
│   ├── UserDefinedClassVariable — 用户自定义类
│   ├── UserDefinedConstantVariable — 用户自定义常量
│   ├── UserDefinedDictVariable — 用户自定义字典
│   ├── UserDefinedListVariable — 用户自定义列表
│   ├── UserDefinedSetVariable — 用户自定义集合
│   ├── UserDefinedTupleVariable — 用户自定义元组
│   ├── UserDefinedExceptionClassVariable — 异常类
│   ├── UserDefinedExceptionObjectVariable — 异常对象
│   ├── NamedTupleVariable — namedtuple实例
│   ├── FrozenDataClassVariable — frozen dataclass
│   ├── DefaultDictVariable — defaultdict
│   ├── OrderedDictVariable — OrderedDict
│   ├── InspectVariable — inspect模块
│   ├── RemovableHandleVariable — removable handle
│   ├── StructSequenceVariable — PyStructSequence
│   ├── MutableMappingVariable — MutableMapping
│
├── Misc (misc.py):
│   ├── CellVariable — cell变量(closure)
│   ├── DeletedVariable — 已删除变量
│   ├── ExceptionVariable — 异常追踪
│   ├── GetAttrVariable — getattr结果
│   ├── LambdaVariable — lambda函数
│   ├── NewGlobalVariable — 新全局变量
│   ├── NumpyVariable — numpy模块
│   ├── PythonModuleVariable — Python模块
│   ├── RandomClassVariable — random类
│   ├── RandomVariable — random实例
│   ├── StringFormatVariable — 字符串格式化
│   ├── SuperVariable — super()
│   ├── TracebackVariable — traceback
│   ├── TypingVariable — typing模块
│   ├── UnknownVariable — 未知变量(graph break fallback)
│   ├── WeakRefVariable — weakref
│   ├── AutogradFunctionContextVariable — autograd context
│   ├── AutogradFunctionVariable — autograd function
│
├── Torch体系 (torch.py):
│   ├── TorchInGraphFunctionVariable — torch模块中的函数(在图内)
│   ├── TorchCtxManagerClassVariable — torch上下文管理器类
│
├── SDPAParamsVariable (sdpa.py) — SDPA参数
│
├── Stream体系 (streams.py):
│   ├── StreamVariable — torch.Stream
│   ├── CudaStreamVariable — CUDA stream
│   ├── EventVariable — torch.Event
│   ├── StreamContextVariable — stream上下文
│
├── OptimizerVariable (optimizer.py) — optimizer追踪
│
├── Distributed体系 (distributed.py):
│   ├── DistributedVariable — torch.distributed
│   ├── BackwardHookVariable — backward hook
│
└── ObjectProtocol (object_protocol.py):
    generic_bool / generic_contains / generic_getiter / virtual_iterator_next
```

★ **★ 关键洞察**: Dynamo追踪超过100种Python变量类型! 每种类型都精确镜像CPython的类型系统(C slots: nb_add, nb_bool, sq_repeat等)。这不是简单的"只追踪tensor", 而是对Python对象的全面符号执行。

### 2.3 VariableTracker.build 工厂方法的三条路径

```
VariableTracker.build(tx, value, source, realize):
  │
  ├─ source=None → SourcelessBuilder.create(tx, value)
  │   → 没有Source, 无法创建guards, 只做基本追踪
  │
  ├─ source+realize=True → VariableBuilder(tx, source)(value)
  │   → 立即构建完整VT, 立即创建guards
  │   → 对tensor: 创建FakeTensor + GraphArg + proxy
  │
  ├─ source+realize=False+primitive → LazyConstantVariable.create(value, source)
  │   → 对int/float/str/bool等: 延迟常量
  │   → 只有当值被修改时才触发recompile
  │   → 纯传递(pass-through)的常量永远不会导致recompile!
  │
  └─ source+realize=False+complex → LazyVariableTracker.create(value, source, tx)
      → 首次使用时才展开为具体VT
      → 减少不必要的guard安装
```

★ **★ LazyConstantVariable是性能优化关键**: 纯传递的常量(如只用作索引的int)不会触发guard, 只有被修改时才需要recompile。这大大减少了不必要的recompile次数!

---

## 3. InstructionTranslator: 字节码→FX节点转换

### 3.1 BytecodeDispatchTableMeta: 256项dispatch表

```python
class BytecodeDispatchTableMeta(type):
    """为每个子类安装dispatch_table, 将opcode映射到handler方法"""

    def __init__(cls, name, bases, dct):
        def _missing(opname, *args):
            unimplemented(
                gb_type="Missing bytecode handler",
                context=f"{opname} with args {args}",
                explanation=f"Dynamo不支持bytecode指令 `{opname}`",
                hints=[...],
            )

        dispatch_table = {
            op: getattr(cls, opname, functools.partial(_missing, opname))
            for opname, op in dis.opmap.items()  # Python所有opcode
        }
        cls.dispatch_table = [dispatch_table.get(i) for i in range(2**8)]  # 256项
```

★ **dispatch_table是数组而非字典**: 用opcode数值直接索引(`dispatch_table[inst.opcode]`), 比字典查找快! 这是O(1)的dispatch, 类似CPython自身的opcode dispatch。

### 3.2 InstructionTranslatorBase.step(): 逐指令处理

```python
class InstructionTranslatorBase(metaclass=BytecodeDispatchTableMeta):
    output: OutputGraph                  # FX图构建器
    symbolic_locals: dict[str, VT]       # 局部变量符号表
    symbolic_globals: dict[str, VT]      # 全局变量符号表
    stack: list[VT]                      # Python值栈(符号化)
    instruction_pointer: int | None      # 当前指令位置
    current_instruction: Instruction     # 当前指令
    block_stack: list[BlockStackEntry]   # 异常/with块栈

    def step(self) -> bool:
        """处理一条指令, 返回False表示应退出"""
        ip = self.instruction_pointer
        if ip is None:
            return False
        inst = self.instructions[ip]
        self.instruction_pointer = ip + 1

        # 在栈空时尝试编译子图(speculation机制)
        if not self.stack and self.should_compile_partial_graph():
            self.current_speculation = self.speculate()
            if self.current_speculation.failed(self):
                self.step_graph_break(inst)  # → graph break!
                return False

        # ★ 核心: 一行代码完成dispatch!
        self.dispatch_table[inst.opcode](self, inst)
        return not self.output.should_exit
```

### 3.3 ★ 核心字节码→FX节点映射

```
LOAD_FAST(name):
  symbolic_locals[name].unwrap() → push到stack

LOAD_CONST(value):
  ConstantVariable.create(value) → push到stack

LOAD_GLOBAL(name):
  VariableTracker.build(tx, f_globals[name], GlobalSource(name))
  → 对tensor: 创建FakeTensor + GraphArg + FX proxy
  → 对其他: 创建相应VT (BuiltinVariable/NNModuleVariable等)

CALL_FUNCTION(n_args):
  args = popn(n_args)
  fn = pop()
  fn.call_function(tx, args, kwargs={})
  → TensorVariable.call_function → output.create_proxy("call_function", ...)
  → BuiltinVariable.call_function → 直接计算结果
  → UserFunctionVariable.call_function → inline_user_function_return

LOAD_ATTR(attr):
  obj = pop()
  VariableTracker.build(tx, getattr).call_function(tx, [obj, attrVT], {})
  → TensorVariable: output.create_proxy("call_function", torch.ops.aten.xxx)
  → NNModuleVariable: 返回子模块VT

BINARY_OP(op):
  (3.11+): left, right = popn(2)
  left.call_method(tx, opname, [right], {})  或 nb_add_impl/nb_subtract_impl等
  → TensorVariable: output.create_proxy("call_function", op)

STORE_FAST(name):
  val = pop()
  symbolic_locals[name] = val  → 存入符号表

COMPARE_OP(op):
  compare_op_handlers[op](tx, args, kwargs)
  → TensorVariable: output.create_proxy + SymNodeVariable(符号结果)
```

### 3.4 InstructionTranslator vs InliningInstructionTranslator

```python
class InstructionTranslator(InstructionTranslatorBase):
    """顶层函数翻译器"""
    def __init__(self, instructions, f_code, f_locals, f_globals, ...):
        super().__init__(
            output=OutputGraph(code_options, compiler_fn, self, ...),  # 创建新OutputGraph!
            symbolic_locals={},  # 空符号表, 后面填充
            ...
        )

class InliningInstructionTranslator(InstructionTranslatorBase):
    """内联函数翻译器(函数调用时进入)"""
    @staticmethod
    def inline_call(parent_tx, fn, args, kwargs):
        # 不创建新OutputGraph, 共用parent的OutputGraph!
        # 共享symbolic_locals和symbolic_globals
        # 返回值push回parent的stack
```

★ **★ 关键设计**: InstructionTranslator(顶层)创建新OutputGraph; InliningInstructionTranslator(内联)共用parent的OutputGraph。这就是Dynamo的"函数内联" - 不是真正的函数调用, 而是将函数体展开到同一个FX图中!

### 3.5 CALL_FUNCTION完整处理流程

```python
def CALL_FUNCTION(self, inst):
    args = self.popn(inst.argval)  # 弹出n个参数
    fn = self.pop()                # 弹出函数

    self.call_function(fn, args, {})  # → fn.call_function(tx, args, kwargs)

# fn.call_function dispatch:
fn = TensorVariable:
    → self.call_function → 对tensor方法(如tensor.add): 创建FX节点
fn = BuiltinVariable:
    → self.call_function → 直接计算Python结果
fn = UserFunctionVariable:
    → self.call_function → inline_user_function_return
      → InliningInstructionTranslator.inline_call(parent_tx, fn, args, kwargs)
      → 将函数体展开到当前FX图
fn = NNModuleVariable:
    → self.call_function → 返回子模块属性
fn = UnknownVariable:
    → graph break!
```

---

## 4. Guard系统: 编译代码的有效性保证

### 4.1 Guard创建流程

```
VariableTracker.build(tx, value, source):
  ↓
VariableBuilder(tx, source)(value):
  ↓
  对每种属性创建Guard:
    Guard(source=LocalSource("x"), guard_type="TENSOR_MATCH", ...)
    Guard(source=AttrSource(LocalSource("x"), "dtype"), guard_type="EQUALS_MATCH", ...)
  ↓
install_guard(*guards):
  ↓
  tx.output.guards.append(guard)  → 收集所有guards
  ↓
  编译完成后 → CheckFunctionManager → GuardBuilder → GuardManagerWrapper
```

### 4.2 ★ Guard类型完整列表

```
GuardBuilder的guard方法 (guards.py):

┌──────────────────┬──────────────────────────────────────────────────┐
│ Guard类型         │ 语义                                            │
├──────────────────┼──────────────────────────────────────────────────┤
│ ID_MATCH         │ id(obj) == expected_id → 对象身份相同            │
│ TYPE_MATCH       │ type_id(obj) == expected_type_id → 类型相同      │
│ EQUALS_MATCH     │ obj == expected_value → 值相等(用于常量)         │
│ TENSOR_MATCH     │ ★ 最重要! 检查tensor的shape/stride/dtype/device │
│                  │   + dispatch_keys + dimension marking            │
│ CLOSURE_MATCH    │ 函数closure的code对象相同                         │
│ DICT_VERSION     │ dict.__version__ == expected → 字典未修改        │
│ SHAPE_ENV        │ 符号shape guard (SymInt约束)                     │
│ GLOBAL_STATE     │ PyTorch全局状态(grad mode/dtype等)               │
│ DICT_CONTAINS    │ key in dict == expected                          │
│ SET_CONTAINS     │ item in set == expected                          │
│ LENGTH_CHECK     │ len(obj) == expected                             │
│ DICT_LENGTH      │ len(dict) == expected                            │
│ NONE_MATCH       │ obj is None                                      │
│ NOT_NONE_MATCH   │ obj is not None                                  │
│ TRUE_MATCH       │ obj is True                                      │
│ FALSE_MATCH      │ obj is False                                     │
│ DISPATCH_KEY_SET │ dispatch_key_set(obj) == expected                │
│ NO_HASATTR       │ not hasattr(obj, attr)                           │
│ FLOAT_IS_NAN     │ math.isnan(obj)                                  │
│ COMPLEX_IS_NAN   │ cmath.isnan(obj)                                 │
│ DUAL_LEVEL_MATCH │ C-level dual level match                         │
│ TUPLE_ITER_LEN   │ tuple_iterator length match                      │
│ RANGE_ITER_MATCH │ range iterator的start/stop/step匹配              │
│ DEFAULT_DEVICE   │ 默认设备guard                                     │
│ SYMBOLIC_SHAPE   │ ★ 符号shape guard (C++层执行)                    │
│ DIMENSION_MARKING│ mark_dynamic/mark_static/mark_unbacked约束       │
│ TENSOR_SUBCLASS  │ tensor subclass metadata match                   │
│ DTENSOR_SPEC     │ DTensor spec match                               │
│ OPAQUE_OBJ       │ opaque object guard function match               │
│ NO_ALIASING      │ ★ tensor之间不aliasing                           │
│ STORAGE_OVERLAP  │ storage overlap guard                            │
│ OBJECT_ALIASING  │ 对象aliasing guard                               │
│ MAPPING_KEYS     │ dict.keys()集合不变                              │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 4.3 ★ TENSOR_MATCH Guard详解 (最重要!)

```python
# guards.py: GuardBuilder.TENSOR_MATCH()

def TENSOR_MATCH(self, guard, value=None):
    # 1. 对FSDP模块中的tensor → 跳过(用ID_MATCH代替)
    # 2. 对nn.Module参数 → ID_MATCH(身份guard, 不检查shape)
    # 3. 对普通tensor → ★ TENSOR_MATCH: 检查所有属性!

    # 收集tensor属性:
    pytype = type(value)          # 类型 (torch.Tensor vs nn.Parameter)
    dispatch_keys = torch._C._dispatch_keys(value)  # dispatch key set

    # 从output_graph获取编译时的sizes/strides:
    metadata = output_graph.input_source_to_sizes_strides[guard.originating_source]
    size = convert_to_concrete_values(metadata["size"])
    stride = convert_to_concrete_values(metadata["stride"])

    # ★ 安装C++层guard:
    guard_manager.add_tensor_match_guard(
        value, size, stride, tensor_name,
        verbose_code_parts, user_stack, pytype, dispatch_keys
    )

    # Dimension marking guard:
    for attr_name in ("_dynamo_dynamic_indices", "_dynamo_weak_dynamic_indices",
                      "_dynamo_unbacked_indices", "_dynamo_strict_unbacked_indices",
                      "_dynamo_static_indices"):
        if hasattr(value, attr_name):
            expected_attrs[attr_name] = getattr(value, attr_name)
```

### 4.4 GuardManager树状结构 (C++层)

```
GuardManagerWrapper (Python层, 存储在cache_entry中):
  └── RootGuardManager (C++层根节点)
      ├── framelocals_manager("x", ...) → 检查frame locals中"x"
      │   ├── add_tensor_match_guard(value, size, stride, ...) → LeafGuard
      │   ├── dict_getitem_manager(key, ...) → 检查dict[key]
      │   │   └── add_id_match_guard(id_val, ...) → LeafGuard
      │   └── getattr_manager(attr, ...) → 检查obj.attr
      │       └── add_type_match_guard(...) → LeafGuard
      ├── globals_dict_manager(f_globals, ...) → 检查f_globals
      │   └── dict_getitem_manager(key, ...) → 检查global变量
      └── add_lambda_guard(python_lambda, ...) → 自定义guard

Guard检查流程 (C++层, cache lookup时):
  root.check(f_locals) → 遍历GuardManager树
  → 每个节点调用accessors获取实际值
  → 每个叶子节点调用leaf guard (ID_MATCH/TENSOR_MATCH等)
  → 全部通过 → 返回True → cache hit
  → 任一失败 → 返回False → cache miss → 触发recompile
```

### 4.5 ★ Guard格式可视化示例

```
GuardManager树打印示例 (guards_log.debug):

+- L['x']                       # LocalSource("x")
|  +- tensor_match_guard:       # TENSOR_MATCH
|     size=[2, 3], stride=[3, 1], dtype=torch.float32,
|     device=cuda:0, requires_grad=False, dispatch_keys=...
|  +- __dict__[training]:       # AttrSource + DictGetItem
|     +- id_match_guard: id=...

+- G['torch']                   # GlobalSource("torch")
|  +- getattr[nn]:
|     +- getattr[Module]:
|        +- type_match_guard: type_id=...
```

---

## 5. Graph Break机制

### 5.1 ★ Graph Break触发路径

```
触发graph break的三条路径:

① Unsupported字节码:
   dispatch_table[opcode] = _missing → unimplemented()
   → 没有handler的Python opcode → 直接graph break

② Unsupported Python操作:
   VariableTracker方法抛出Unsupported异常:
   → UnknownVariable.call_function → graph break
   → UserDefinedObjectVariable某些方法 → graph break
   → data-dependent分支(torch.tensor条件) → graph break

③ SpeculationRestartAnalysis:
   → 在栈空时尝试compile子图
   → 下一条指令导致unsupported
   → 回滚到speculation点, 触发graph break
```

### 5.2 ★ 典型Graph Break场景 (来自graph_break_registry.json)

```
GB0000: __torch_function__全部返回NotImplemented
GB0003: Generator函数 (不能直接编译)
GB0004: super().__delattr__() 无mutation tracking
GB0006: 调用super()属性不是函数/方法
GB2861: C-level tp_hash (Dynamo不能trace进C __hash__)
GB0005: str()方法用C实现
... (2800+种已知graph break场景)
```

### 5.3 Graph Break处理流程

```python
def step(self):
    try:
        self.dispatch_table[inst.opcode](self, inst)
    except (Unsupported, UserError) as e:
        if self.one_graph or self.error_on_graph_break:
            raise  # fullgraph模式: 直接报错!
        if self.current_speculation is None:
            log.debug("empty checkpoint - cannot resume from graph break")
            e.skip_frame = True
            raise  # 跳过整个frame

        # ★ 有checkpoint → 回滚到之前的speculation点
        self.current_speculation.fail_and_restart_analysis(self.error_on_graph_break)
        return False  # 重新开始trace!
```

### 5.4 step_graph_break: 子图编译+resume函数生成

```python
def step_graph_break(self, continue_inst):
    """在栈空时触发graph break, 编译当前子图"""

    # 1. 编译当前子图
    all_stack_locals_metadata = self.output.compile_subgraph(
        self, reason=GraphCompileReason("step_unsupported", ...)
    )

    # 2. 生成resume函数
    leaf_resume_code, leaf_resume_name = self.create_resume(
        0, continue_inst, all_stack_locals_metadata[0], [], cg, True, False
    )
    skip_code(leaf_resume_code)  # resume函数不需要再次被Dynamo捕获

    # 3. 生成bytecode: 调用编译函数 → 调用resume函数 → 继续执行
    self.output.add_output_instructions([
        cg.get_instructions(),
        self.parent.create_call_resume_at(...)
    ])
```

★ **★ Graph Break后的执行**: 编译的子图作为一个函数调用, 后续代码作为resume函数继续执行。每个graph break生成两个函数: compiled_subgraph_fn + resume_fn。resume_fn的代码以`__resume_in_`前缀开头, Dynamo通过output_codes追踪避免再次trace这些生成的代码。

---

## 6. Cache/Lookup系统

### 6.1 Cache Entry结构 (C++层)

```cpp
// torch/csrc/dynamo/cache_entry.h

class CacheEntry {
    PyCodeObject* code;             // 编译后的代码对象
    PyObject* backend;              // backend函数引用
    GuardManagerWrapper* guard_manager;  // ★ guard检查器(整个树!)
    int64_t isolate_recompiles_id;  // 隔离recompile ID
    CompileId compile_id;           // 编译ID
    CacheEntry* next;               // 链表next(同一code object多个cache entry)
};
```

### 6.2 ★ Cache Lookup流程 (完全在C/C++层!)

```
dynamo__custom_eval_frame(tstate, frame, throw_flag, callback):

① extra = get_extra_state(F_CODE(frame))
   → 每个code object有一个ExtraState, 包含cache_entry链表

② 尝试fast-path (不需要解frame locals):
   try_lookup_without_guard_eval(extra, backend, isolate_recompiles_id, ...)
   → 如果有且不需要guard → 直接返回cached_code

③ 如果需要guard → 解frame locals:
   locals = FrameLocalsMapping(frame)  → C++层构建locals映射

④ ★ 核心lookup:
   lookup(extra, locals, backend, isolate_recompiles_id, ...)
   → 遍历extra的cache_entry链表
   → 对每个entry:
     guard_manager.check_nopybind(f_locals)  → ★ C++层guard检查!
     → 通过 → 返回cached_code
     → 失败 → 继续下一个entry
   → 全部失败 → 返回None → cache miss

⑤ cache hit → eval_custom(frame, cached_code)
   → dynamo_eval_custom_code(tstate, frame, cached_code, ...)
   → 创建新的shadow frame, 复制localsplus, 执行cached_code

⑥ cache miss → dynamo_call_callback(callback, frame, locals, ...)
   → 回到Python层: ConvertFrameAssert.__call__
   → 触发InstructionTranslator trace → 生成新guarded_code
   → create_cache_entry(extra, guarded_code, backend) → 添加到cache链表
```

★ **★ 核心性能洞察**: Guard检查完全在C++层完成! `guard_manager.check_nopybind()`是C++函数, 不需要回到Python。只有cache miss时才调用Python callback。这使得cache hit几乎零开销(只比原始_PyEval_EvalFrameDefault多一个guard树遍历)。

### 6.3 ConvertFrameAssert.__call__: Python层callback

```python
class ConvertFrameAssert:
    def __call__(self, frame, cache_entry, hooks, frame_state):
        code = frame.f_code
        cache_entries = _get_cache_entries_for_region(code, isolate_recompiles_id)
        cache_size = compute_cache_size(frame, cache_entries, total_count)

        # 检查是否应该trace这个frame:
        if code in output_codes:  → skip (Dynamo生成的代码)
        if not has_tensor_in_frame(frame):  → skip (无tensor)
        if is_generator(code):  → unsupported

        # ★ 触发编译:
        result = _compile(
            frame.f_code, frame.f_globals, frame.f_locals, ...
        )
```

### 6.4 _compile → trace_frame → InstructionTranslator.run

```python
def trace_frame(code, globals, locals, builtins, closure, compiler_fn, ...):
    tracer = InstructionTranslator(
        instructions, code, locals, globals, builtins, closure, ...
    )
    tracer.run()  # ★ 逐指令trace, 构建FX图!

    # 编译完成后:
    tracer_output = DynamoTracerOutput(tracer)
    output = tracer_output.output_graph
    instructions[:] = output.output_instructions  # 生成的bytecode

    # 转换代码对象:
    bytecode = transform_code_object(code, instructions, code_options)
```

---

## 7. Dynamo → FX Graph转换

### 7.1 OutputGraph: FX图构建器

```python
class OutputGraph:
    graph: torch.fx.Graph           # FX图(节点和边)
    nn_modules: dict                # nn.Module映射
    guards: list[Guard]             # 收集的guards
    import_sources: dict            # 导入的模块
    shape_env: ShapeEnv             # 符号shape环境
    side_effects: SideEffects       # 副作用追踪

    def create_proxy(self, kind, target, args, kwargs):
        """★ 创建FX节点"""
        # 1. 在FX graph中创建新node:
        node = self.graph.create_node(kind, target, args, kwargs)
        # 2. 返回proxy (带node引用):
        return proxy_tensor.Proxy(node)
```

### 7.2 ★ 完整Dynamo → FX转换数据流

```
torch.compile(model) 完整流程:

① torch.compile → _TorchDynamoContext(callback=compiler_fn)
② set_eval_frame(callback) → 替换CPython帧评估函数
③ 模型forward被调用 → Python解释器执行forward字节码
④ 但帧评估函数已被替换为dynamo_custom_eval_frame_shim!
⑤ shim → dynamo__custom_eval_frame → cache miss → dynamo_call_callback
⑥ callback → ConvertFrameAssert.__call__ → _compile → trace_frame
⑦ trace_frame → InstructionTranslator(instructions, f_code, f_locals, ...)
⑧ InstructionTranslator.run():
    while self.step():  # 逐指令
        inst = self.instructions[ip]
        self.dispatch_table[inst.opcode](self, inst)

        # 典型tensor操作:
        LOAD_FAST("x") → push TensorVariable(x)
        LOAD_FAST("weight") → push TensorVariable(weight)
        CALL_FUNCTION → TensorVariable.call_function
          → output.create_proxy("call_function", torch.ops.aten.add, (x, weight))
          → FX graph新增node: %add = call_function[target=aten.add](x, weight)

⑨ 编译完成 → OutputGraph包含完整FX图
⑩ CheckFunctionManager → GuardBuilder → 创建GuardManagerWrapper
⑪ GuardedCode(code=transformed_bytecode, guard_manager=wrapper)
⑫ create_cache_entry(extra, guarded_code, backend) → 存入cache
⑬ eval_custom(frame, cached_code) → 执行编译后的代码

下次调用同一函数:
  shim → dynamo__custom_eval_frame → cache lookup → guard_manager.check
  → 全部guard通过 → cache hit → eval_custom → 执行cached_code
  → 无需回到Python! ★
```

---

## 8. ★ ★ 关键洞察总结

### 8.1 Dynamo架构本质

```
Dynamo = Python字节码符号执行器 + C-level拦截器 + FX图生成器

三层架构:
  C层 (eval_frame.c/eval_frame_cpp.cpp):
    - 替换_PyInterpreterState_SetEvalFrameFunc → 拦截所有帧
    - PyThread_tss thread-local callback → 每线程独立控制
    - GuardManager C++检查 → cache hit零Python开销
    - FrameLocalsMapping → 高效locals映射(避免dict构建)

  Python层 (symbolic_convert.py/variables/):
    - InstructionTranslator → 逐指令符号执行
    - 100+ VariableTracker → 精确追踪所有Python类型
    - BytecodeDispatchTableMeta → 256项dispatch数组 → O(1)opcode查找
    - Speculation机制 → 栈空时尝试编译子图

  FX层 (output_graph.py):
    - OutputGraph → torch.fx.Graph节点构建
    - SubgraphTracer → 支持higher-order operators嵌套
    - FakeTensorMode → 符号shape推断
    - GuardBuilder → 从变量属性创建guard树
```

### 8.2 ★ ★ RTX 4090相关关键结论

```
1. torch.compile在RTX 4090上完全可用! Dynamo是纯软件层, 不依赖特定GPU

2. Guard overhead在RTX 4090上:
   - cache hit: C++层guard检查, ~微秒级, 几乎零开销
   - cache miss: 重新trace+compile, 秒级
   - 最多64次recompile (GuardsSet最多64个guard)

3. 最优策略:
   - fullgraph=True → 零graph break → 单次编译
   - dynamic shapes → 减少recompile (允许shape变化)
   - INT4+INT8KV → 无影响(Dynamo不关心量化)

4. 需要避免的graph break:
   - data-dependent分支 (tensor条件if/while)
   - generator函数
   - C-level tp_hash
   - 动态list/dict修改

5. FSDP2 + Dynamo:
   - FSDP2的参数分片 → Dynamo需要guard每个参数
   - ZeRO-3 + compile → intentional graph breaks at FSDP hooks
   - FSDPManagedNNModuleVariable → 专门处理FSDP模块
```

### 8.3 ★ ★ 与已有reading的联系

```
- torch.compile e2e (已有): eval_frame hook → Dynamo → FX → AOTAutograd → Inductor
  本reading补充: Dynamo内部机制(字节码拦截→符号执行→guard→FX)

- PyTorch FSDP2 internals (已有): per-param DTensor + intentional graph breaks
  本reading补充: graph break的C++层处理机制 + FSDPManagedNNModuleVariable

- PyTorch FX IR (已有): Node 6op + Graph双向链表
  本reading补充: Dynamo如何创建FX node (create_proxy → call_function)

- PyTorch compile e2e (已有): GuardedCode最多64次recompile
  本reading补充: GuardManager C++树状结构 + TENSOR_MATCH完整语义

- vLLM V1 GPUModelRunner (已有): CUDAGraphDispatcher + Persistent buffers
  本reading补充: Dynamo的CUDAGraph支持(CudagraphOverrideVariable)
```

---

## 9. Sources: 源码文件与行号

| 文件 | 关键内容 | 行号 |
|------|---------|------|
| `torch/csrc/dynamo/eval_frame.c` | C-level shim安装/卸载, set_eval_frame | 609-656, 230-400 |
| `torch/csrc/dynamo/eval_frame_cpp.cpp` | dynamo__custom_eval_frame完整流程, cache lookup | 343-700 |
| `torch/_C/_dynamo/eval_frame.pyi` | C层API类型定义(set_eval_frame, _CacheEntry, _PyInterpreterFrame) | 全文 |
| `torch/_C/_dynamo/guards.pyi` | C++ GuardManager API(RootGuardManager, LeafGuard, DictGuardManager) | 全文 |
| `torch/_dynamo/eval_frame.py` | _TorchDynamoContext, OptimizedModule, set_eval_frame Python层 | 1-900 |
| `torch/_dynamo/convert_frame.py` | ConvertFrameAssert, _compile, trace_frame | 601-1000 |
| `torch/_dynamo/symbolic_convert.py` | InstructionTranslatorBase, BytecodeDispatchTableMeta, CALL_FUNCTION, LOAD_GLOBAL | 1213-1360, 1347-1800, 2226-2400, 3024-3200, 5298-5700 |
| `torch/_dynamo/variables/__init__.py` | 100+ VariableTracker类导入 | 全文 |
| `torch/_dynamo/variables/base.py` | VariableTracker基类, build工厂方法, make_guards | 320-1900 |
| `torch/_dynamo/variables/builder.py` | VariableBuilder, GraphArg, wrap_fx_proxy | 642-3644 |
| `torch/_dynamo/guards.py` | GuardBuilder, TENSOR_MATCH, CheckFunctionManager | 1018-1219, 1219-1600, 3502-3700, 4400-4600 |
| `torch/_dynamo/output_graph.py` | OutputGraph, SubgraphTracer, CodeOptions | 1-300 |
| `torch/_dynamo/exc.py` | Unsupported, RestartAnalysis, SkipFrame异常 | 157-270 |
| `torch/_dynamo/graph_break_registry.json` | 2800+种已知graph break场景 | 全文 |
| `torch/_dynamo/trace_rules.py` | trace规则(哪些代码应被trace/跳过) | 4234行 |
