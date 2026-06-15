# PyTorch custom_op and torch.library System: Deep Source-Level Analysis

> Source files analyzed from `pytorch/pytorch` (main branch, June 2026)
> Focus: registration pipeline, dispatch key routing, autograd integration, torch.compile compatibility

---

## 1. torch.library.custom_op API -- The Modern Entry Point

### 1.1 Location and Architecture

The production-ready API lives in `torch/_library/custom_ops.py` (1123 lines). The deprecated `torch/_custom_op/impl.py` (718 lines) is a legacy wrapper that redirects to `torch.library`.

Key class: **`CustomOpDef`** (lines 271-1037 of `custom_ops.py`)

### 1.2 custom_op Decorator/Function Signature

```python
# torch/_library/custom_ops.py lines 66-268

@exposed_in("torch.library")
def custom_op(
    name: str,             # e.g. "mylib::numpy_sin"
    fn: Callable | None = None,  /,
    *,
    mutates_args: str | Iterable[str],   # CRITICAL: must be accurate
    device_types: device_types_t = None, # "cpu", "cuda", or None (all devices)
    schema: str | None = None,           # manual schema override
    tags: tags_t = None,                 # torch.Tag.inplace, torch.Tag.out, etc.
) -> Union[Callable, "CustomOpDef"]:
```

**Two usage modes** (decorator vs direct call, see NOTE at lines 1070-1088):

```python
# Decorator usage:
@custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: Tensor) -> Tensor: ...

# Direct call:
numpy_sin = custom_op("mylib::numpy_sin", numpy_sin_impl, mutates_args=())
```

### 1.3 Registration Pipeline (Step-by-Step)

When `custom_op("mylib::numpy_sin", mutates_args=())` is called on a function `fn`, the `inner` function (lines 235-268) executes:

**Step 1: Schema Inference** (lines 239-246)
```python
if schema is None:
    schema_str = torch.library.infer_schema(fn, mutates_args=mutates_args, tags=normalized_tags)
else:
    schema_str = schema
```
The `infer_schema` function (`torch/_library/infer_schema.py`) parses the Python function's type annotations into a C++ schema string. For example, `(Tensor x) -> Tensor` becomes the schema string. It handles:
- `Tensor` -> `Tensor`, `int` -> `int`, `float` -> `float`
- `list[Tensor]` -> `Tensor[]`, `Optional[Tensor]` -> `Tensor?`
- `mutates_args` controls alias annotations: `Tensor(a!)` for mutable args
- `torch.Tag.inplace` forces first arg to be `Tensor(a!)` and return `Tensor(a)`
- `torch.Tag.out` forces mutable kwarg-only args with matching returns

**Step 2: Namespace/Opname Split** (line 248)
```python
namespace, opname = name.split("::")
```

**Step 3: CustomOpDef Creation** (line 249)
```python
result = CustomOpDef(namespace, opname, schema_str, fn, normalized_tags)
```
This triggers the constructor (lines 281-312) which:
- Stores `_namespace`, `_name`, `_schema`, `_tags`, `_init_fn`
- Initializes empty holders: `_backend_fns`, `_abstract_fn`, `_setup_context_fn`, `_backward_fn`
- Creates a Library: `self._lib = get_library_allowing_overwrite(namespace, name)`
- Calls `self._register_to_dispatcher(self._tags)` -- the core registration
- Adds to global registry: `OPDEFS[self._qualname] = self`

**Step 4: Default Kernel Registration** (line 263)
```python
result.register_kernel(device_types)(fn)
```
The initial `fn` is registered as the first kernel for the specified `device_types`.

### 1.4 get_library_allowing_overwrite (lines 1097-1108)

```python
def get_library_allowing_overwrite(namespace, name):
    qualname = f"{namespace}::{name}"
    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()   # destroy old Library
        del OPDEF_TO_LIB[qualname]
    lib = torch.library.Library(namespace, "FRAGMENT")
    OPDEF_TO_LIB[qualname] = lib
    return lib
```
This enables **re-registration** by destroying any previous Library for the same qualname. The `FRAGMENT` kind is crucial -- it bypasses the limitation that only one Library can exist per namespace.

---

## 2. CustomOpDef._register_to_dispatcher -- The Core Dispatch Setup

### 2.1 The 4-Phase Registration (lines 740-797)

```python
def _register_to_dispatcher(self, tags):
    # Phase 1: Define the operator schema
    schema_str = self._name + self._schema
    cpp_schema = _C.parse_schema(schema_str)
    self._validate_schema(cpp_schema, schema_str)
    self._define_dispatcher_op(schema_str, tags)       # -> _lib.define()

    # Phase 2: Check inplace/out semantics
    self._is_inplace = utils.is_inplace(self._opoverload)
    self._is_out = utils.is_out(self._opoverload)
    if self._is_out:
        _, out_kwarg_names = utils.mutated_args_kwargs(self._opoverload._schema)
        self._out_kwarg_names = tuple(out_kwarg_names)

    # Phase 3: Register fake impl (Meta dispatch key)
    self._register_fake_dispatcher_impl()

    # Phase 4: Register autograd impl (Autograd dispatch key)
    self._register_autograd_dispatcher_impl()

    # Phase 5: Register ADInplaceOrView if mutable/view
    self._register_adinplaceorview_dispatcher_impl()
```

### 2.2 _define_dispatcher_op (lines 764-769)

```python
def _define_dispatcher_op(self, schema_str, tags):
    self._lib.define(schema_str, tags=_with_pt2_compliant_tag(tags))
    self._opoverload = utils.lookup_op(self._qualname)
```

**Critical detail**: `_with_pt2_compliant_tag` (lines 33-37) **always adds `torch.Tag.pt2_compliant_tag`**:
```python
def _with_pt2_compliant_tag(tags):
    return [
        _C.Tag.pt2_compliant_tag,
        *(tag for tag in tags if tag != _C.Tag.pt2_compliant_tag),
    ]
```
This means **every custom op created via `torch.library.custom_op` is automatically PT2-compliant**. This tag is what makes custom ops work with torch.compile without graph breaks. The `pt2_compliant_tag` signals to Dynamo that the op has proper Meta/FakeTensor and Autograd registrations.

After defining, `lookup_op` retrieves the `OpOverload` from `torch.ops.{namespace}.{opname}`.

### 2.3 _register_fake_dispatcher_impl (lines 771-786)

```python
def _register_fake_dispatcher_impl(self):
    def fake_impl(*args, **kwargs):
        if self._abstract_fn is None:
            if utils.can_generate_trivial_fake_impl(self._opoverload):
                return utils.generate_trivial_fake_impl(self._opoverload, *args, **kwargs)
            raise RuntimeError(
                f"There was no fake impl registered for {self}. "
                f"Please use `{self._init_fn.__name__}.register_fake` to add one."
            )
        return self._abstract_fn(*args, **kwargs)
    self._lib._register_fake(self._name, fake_impl, _stacklevel=5)
```

**Auto-generated trivial fake impls** (`can_generate_trivial_fake_impl` in `utils.py` lines ~240-270):
- **Tag.inplace ops**: returns `args[0]` (the mutated first arg)
- **Tag.out ops**: returns the `out=` kwargs in declaration order
- **Mutable ops returning nothing**: returns `None`
- **All other ops**: requires explicit `register_fake`

### 2.4 _register_autograd_dispatcher_impl (lines 788-790)

```python
def _register_autograd_dispatcher_impl(self):
    autograd_impl = autograd.make_autograd_impl(self._opoverload, self)
    self._lib.impl(self._name, autograd_impl, "Autograd", with_keyset=True)
```

This creates the **Autograd dispatch key kernel** using `autograd.make_autograd_impl`. See Section 4 for details.

---

## 3. register_kernel -- Device-Specific Kernel Registration

### 3.1 CustomOpDef.register_kernel (lines 391-520)

```python
def register_kernel(self, device_types, fn=None, /):
    def inner(fn):
        # Normalize device_types to list
        if device_types is None or isinstance(device_types, str):
            dtypes = [device_types]
        else:
            dtypes = list(device_types)

        for device_type in dtypes:
            if device_type not in self._backend_fns:
                # First registration for this device_type: register to C++ dispatcher
                def backend_impl(*args, **kwargs):
                    result = self._backend_fns[device_type](*args, **kwargs)
                    # Validate aliasing constraints:
                    # - inplace ops must return args[0]
                    # - out ops must return their out kwargs
                    # - non-view ops must not alias inputs
                    ...
                    return result

                if device_type is None:
                    self._lib.impl(self._name, backend_impl, "CompositeExplicitAutograd")
                else:
                    self._lib.impl(self._name, backend_impl,
                                   _C._dispatch_key_for_device(device_type))

            # Wrap fn with disable_dynamo and kernel-enable/disable logic
            @torch._disable_dynamo
            def wrapped_fn(*args, **kwargs):
                if device_type in self._disabled_kernel:
                    return self._init_fn(*args, **kwargs)  # fallback to default
                else:
                    return fn(*args, **kwargs)

            self._backend_fns[device_type] = wrapped_fn
        return fn
```

**Dispatch key mapping**:
- `device_type=None` -> `CompositeExplicitAutograd` (catches all devices)
- `device_type="cpu"` -> `CPU` dispatch key (via `_C._dispatch_key_for_device`)
- `device_type="cuda"` -> `CUDA` dispatch key
- Other valid types: `"xla"`, `"mps"`, `"ipu"`, `"xpu"`

**Aliasing validation** (lines 448-482): Custom ops have strict aliasing constraints:
- **inplace** (`torch.Tag.inplace`): must return `args[0]`
- **out** (`torch.Tag.out`): must return the mutable kwargs
- **non-view functional**: outputs must not alias inputs (checked via `utils._c_check_aliasing_constraint` which uses `_C._any_output_is_alias_to_input_or_output`)

### 3.2 Factory Functions and BackendSelect (lines 843-862)

For ops with **no Tensor inputs** (factory functions), `register_kernel` also calls `_register_backend_select_dispatcher`:
```python
def _register_backend_select_dispatcher(self, device_arg_index):
    def backend_select(keyset, *args, **kwargs):
        device = args[device_arg_index].type
        if device not in self._backend_fns:
            raise RuntimeError(f"{self._name} does not have a kernel for {device}")
        dispatch_key = _C._dispatch_key_for_device(device)
        dispatch_key = getattr(_C.DispatchKey, dispatch_key)
        return self._opoverload.redispatch(
            _C.DispatchKeySet(dispatch_key), *args, **kwargs)
    self._lib.impl(self._name, backend_select, "BackendSelect", with_keyset=True)
```
This routes factory ops to the correct device based on the `device: torch.device` argument.

### 3.3 set_kernel_enabled Context Manager (lines 322-389)

```python
@contextmanager
def set_kernel_enabled(self, device_type, enabled=True):
    # Temporarily disable/enable a kernel for testing or conditional execution
    ...
    yield
    # Restore original state
```
When a kernel is disabled, `wrapped_fn` falls back to `self._init_fn` (the default implementation).

### 3.4 Library.impl -- The C++ Dispatcher Bridge (torch/library.py lines 438-540)

```python
class Library:
    def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=True):
        name = self._resolve_op_name(op_name, "impl")
        # Collision check via _impls set
        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if (not allow_override) and key in _impls:
            raise RuntimeError(...)
        # Delegate to C++ Library object
        self.m.impl(name, dispatch_key, fn, with_keyset)
        _impls.add(key)
        self._op_impls.add(key)
```
The `self.m` is a `torch._C._dispatch_library` C++ object. The `with_keyset=True` parameter passes the current DispatchKeySet as the first argument to `fn`, needed for redispatch calls in autograd/ADInplaceOrView kernels.

---

## 4. register_fake -- Shape Inference for torch.compile

### 4.1 CustomOpDef.register_fake (lines 522-602)

```python
def register_fake(self, fn, /):
    if self._is_inplace or self._is_out:
        tag = "torch.Tag.inplace" if self._is_inplace else "torch.Tag.out"
        warnings.warn(f"The fake registration for {self} is unnecessary. "
                      f"Custom ops tagged with {tag} already get an autogenerated fake kernel.")
    self._abstract_fn = fn
    return fn
```

Simple -- just stores `fn` in `self._abstract_fn`. The actual dispatch registration happened during `_register_to_dispatcher` (Phase 3), which created a closure `fake_impl` that checks `self._abstract_fn`.

### 4.2 Module-Level register_fake (torch/library.py lines 1150-1310)

For ops not created via `custom_op`:
```python
def register_fake(op, func=None, /, *, lib=None, allow_override=True):
    op = _resolve_op_for_registration(op, "register_fake")
    if isinstance(op, CustomOpDef):
        # Delegate to CustomOpDef.register_fake
        if func is None:
            return op.register_fake
        else:
            return op.register_fake(func)
    # Otherwise, use the Library._register_fake path
    namespace, op_name = torch._library.utils.parse_namespace(op)
    use_lib = _library_for_registration(namespace, lib)
    use_lib._register_fake(op_name, func, _stacklevel=..., allow_override=allow_override)
```

### 4.3 Library._register_fake (torch/library.py lines ~330-380)

```python
def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=True):
    source = torch._library.utils.get_source(_stacklevel + 1)
    qualname = f"{self.ns}::{op_name}"
    entry = torch._library.simple_registry.singleton.find(qualname)
    func_to_register = _check_pystubs_once(fn, qualname, caller_module_name)  # or just fn
    handle = entry.fake_impl.register(func_to_register, source, lib=self, allow_override=allow_override)
    self._registration_handles.append(handle)
```

### 4.4 FakeImplHolder -- The Registration Container (torch/_library/fake_impl.py)

```python
class FakeImplHolder:
    def __init__(self, qualname):
        self.qualname = qualname
        self.kernels: list[Kernel] = []  # ordered by registration time

    @property
    def kernel(self):
        if len(self.kernels) == 0:
            return None
        return self.kernels[-1]  # newest kernel wins

    def register(self, func, source, lib, *, allow_override=True):
        kernel = Kernel(func, source)
        self.kernels.append(kernel)

        meta_kernel = construct_meta_kernel(self.qualname, self)
        lib.impl(self.qualname, meta_kernel, "Meta", allow_override=allow_override)

        handle = RegistrationHandle(deregister_fake_kernel)
        return handle
```

**Key design**: `FakeImplHolder` maintains a **list** of kernels, with the newest one winning. This supports library overwriting -- when a new Library is loaded, it can override the fake impl, and when that Library is destroyed, the previous one is restored.

### 4.5 construct_meta_kernel (torch/_library/fake_impl.py)

```python
def construct_meta_kernel(qualname, fake_impl_holder):
    @functools.wraps(fake_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        source = fake_impl_holder.kernel.source

        def error_on_ctx():
            raise RuntimeError(
                f"{qualname} ({source}): You're trying to run this operator "
                f"with meta Tensors, but this operator may return data-dependent shape. "
                f"Use FakeTensors instead.")

        with set_ctx_getter(error_on_ctx):
            return fake_impl_holder.kernel(*args, **kwargs)
    return meta_kernel
```

The meta kernel wraps the fake impl with `set_ctx_getter`. When running with **meta tensors** (not FakeTensors), calling `get_ctx()` raises an error because meta tensors cannot support data-dependent shapes. With **FakeTensors**, the ctx getter is set to provide `FakeImplCtx` with `new_dynamic_size()` support.

### 4.6 FakeImplCtx and Data-Dependent Shapes (torch/_library/fake_impl.py)

```python
class FakeImplCtx:
    def new_dynamic_size(self, *, min=0, max=None):
        """Constructs a new symint representing a data-dependent value."""
        if self._shape_env is None or not self._shape_env.allow_dynamic_output_shape_ops:
            raise DynamicOutputShapeException(self._op)  # -> graph break in Dynamo
        result = shape_env.create_unbacked_symint()
        torch.fx.experimental.symbolic_shapes._constrain_range_for_size(result, min=min_val, max=max_val)
        return result
```

**Critical for torch.compile**: When a custom op has data-dependent output shape (like nonzero), the fake impl must use `ctx.new_dynamic_size()` to create symbolic integers. Without this, `DynamicOutputShapeException` is raised, causing a **graph break** in Dynamo.

### 4.7 get_ctx (torch/library.py lines 1687-1702)

```python
def get_ctx():
    """Only valid inside a fake impl."""
    return torch._library.fake_impl.global_ctx_getter()
```

The `global_ctx_getter` is a module-level callable that changes based on context:
- During FakeTensor dispatch: returns `FakeImplCtx` with `new_dynamic_size()`
- During meta tensor dispatch: returns `None` or raises error
- Default: returns `None` via `get_none()`

---

## 5. register_autograd -- Custom Backward Passes

### 5.1 CustomOpDef.register_autograd (lines 638-737)

```python
def register_autograd(self, backward, /, *, setup_context=None):
    # Validation:
    if utils.is_out(self._opoverload):
        raise RuntimeError("Cannot register autograd for torch.Tag.out ops")
    if not utils.is_functional_schema(schema, allow_valid_view=True):
        raise RuntimeError("Cannot register autograd for non-functional ops")

    self._backward_fn = backward
    self._setup_context_fn = setup_context
```

Simple storage -- just saves the backward and setup_context functions. The actual dispatch registration happened during `_register_to_dispatcher` (Phase 4).

### 5.2 autograd.make_autograd_impl (torch/_library/autograd.py)

This is the **core mechanism** that bridges custom op backward formulas with PyTorch's autograd engine:

```python
def make_autograd_impl(op, info):
    # info is the CustomOpDef object (or Info dataclass for non-custom_op path)
    # info._backward_fn and info._setup_context_fn

    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    def forward_no_grad(*args):
        metadata = args[-1]  # Metadata is appended as last arg
        args = args[:-1]
        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    def forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        with _C._AutoDispatchBelowAutograd():
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            if info._setup_context_fn:
                args, kwargs = utils.fill_defaults(op._schema, args, kwargs)
                if has_kwarg_only_args:
                    info._setup_context_fn(ctx=ctx, inputs=args,
                                          keyword_only_inputs=kwargs, output=result)
                else:
                    info._setup_context_fn(ctx=ctx, inputs=args, output=result)
            return result

    def backward(ctx, *grads):
        if info._backward_fn:
            prev_needs_input_grad = ctx.needs_input_grad
            ctx.needs_input_grad = prev_needs_input_grad[:-1]  # strip Metadata bool
            result = info._backward_fn(ctx, *grads)
            # Validate result count matches forward input count
            ...
            return (*result, None)  # None is gradient for Metadata arg
        raise RuntimeError(f"Trying to backward through {op} but no autograd formula registered.")

    Generated = type(
        f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}",
        (autograd.Function,),
        {"forward": staticmethod(forward), "backward": staticmethod(backward)},
    )

    if has_tensorlist_like_args:
        Generated = supports_tensorlist(Generated)

    def autograd_impl(keyset, *args, **keyword_only_args):
        if utils.is_out(op):
            # out= ops don't support autograd
            if _C.is_grad_enabled() and _C._any_requires_grad(*args, **keyword_only_args):
                raise RuntimeError("out= ops don't support autograd")
            return forward_no_grad(*args, Metadata(keyset, keyword_only_args))

        if _C.is_grad_enabled() and _C._any_requires_grad(*args):
            result = Generated.apply(*args, Metadata(keyset, keyword_only_args))
        else:
            result = forward_no_grad(*args, Metadata(keyset, keyword_only_args))
        return result

    return autograd_impl
```

### 5.3 Key Design Decisions in Autograd Implementation

**Dynamic `autograd.Function` generation**: A new class is dynamically created via `type()` for each custom op. This class inherits from `torch.autograd.Function` and has `forward`/`backward` methods that call the user-provided functions.

**Metadata as hidden argument**: The `Metadata(keyset, keyword_only_args)` object is appended as the last argument to `Generated.apply()`. This carries:
- `keyset`: the current DispatchKeySet, needed for `redispatch`
- `keyword_only_args`: kwargs that the dispatcher separates from positional args

The Metadata trick is necessary because `autograd.Function.apply()` only accepts positional args, but custom ops may have keyword-only args.

**Two execution paths**:
1. **With grad**: `Generated.apply(*args, Metadata)` -> `forward(ctx, *args_with_metadata)` -> `op.redispatch(after_autograd_keyset)` -> backend kernel
2. **Without grad**: `forward_no_grad(*args, Metadata)` -> `op.redispatch(after_autograd_keyset)` -> backend kernel directly

**redispatch below autograd**: Both forward paths use `op.redispatch(keyset & _C._after_autograd_keyset)` to skip the Autograd key and go directly to backend kernels. This prevents infinite recursion (the Autograd kernel calling itself).

**setup_context timing**: `setup_context(ctx, inputs, output)` runs during the forward pass inside the `forward` method of the generated class. It receives the same `ctx` object used by `torch.autograd.Function`.

**backward validation**: The backward function must return gradients matching the number of forward positional inputs. Extra gradients (beyond `num_actual_inputs`) must all be `None`.

### 5.4 supports_tensorlist -- List[Tensor] Autograd Support (lines 90-168)

Regular `autograd.Function` only supports `Tensor` inputs/outputs. The `supports_tensorlist` wrapper:
- Flattens `List[Tensor]` inputs to flat tensor list before `apply()`
- Unflattens in `forward` and `backward`
- Tracks `TensorListMetadata` on `ctx._pt_metadata` for proper unflattening
- Validates `needs_input_grad` structure matches the input pytree

### 5.5 The Legacy CustomOp Autograd Path (torch/_custom_op/autograd.py)

The deprecated `torch._custom_op` API uses a different pattern:
- **Indirection**: `autograd_kernel_indirection(custom_op)` checks if the user has registered `backward`/`save_for_backward`, and either calls the generated kernel or `autograd_not_implemented` fallback
- **Namedtuple args**: Arguments are wrapped in a namedtuple derived from the schema for cleaner access
- **save_pytree_for_backward/unpack_saved**: Saves a pytree onto `ctx` by separating tensors (via `ctx.save_for_backward`) and non-tensors (via `ctx.saved_non_tensors`)
- **grad_inputs_dict**: The backward must return a dict mapping arg names to gradients, not positional tuple

The new API (`torch.library.custom_op`) uses the simpler pattern from `torch/_library/autograd.py` described above.

---

## 6. torch.library Namespace -- Module-Level APIs

### 6.1 torch/library.py Architecture

The file (75,963 bytes) provides the public API namespace. Key exports:

```python
__all__ = [
    "Library", "impl", "define", "fallthrough_kernel",
    "impl_abstract", "register_autocast", "register_fake",
    "register_torch_dispatch", "register_vmap", "get_ctx",
    "get_kernel", "custom_op", "triton_op", "wrap_triton", "infer_schema",
]
```

### 6.2 Library Class (torch/library.py)

Three kinds of Library:
- **DEF**: Creates a new namespace, can define new ops
- **IMPL**: Overrides existing ops in a namespace
- **FRAGMENT**: Can define and/or override, bypasses "one Library per namespace" restriction

The `FRAGMENT` kind is what `CustomOpDef` uses (via `get_library_allowing_overwrite`). Each CustomOpDef gets its own Library fragment, enabling multiple independent custom ops in the same namespace.

### 6.3 Collision Tracking (_impls and _defs sets)

```python
_impls: set[str] = set()  # "namespace/op_name/dispatch_key"
_defs: set[str] = set()   # "namespace::op_name"
```
These prevent two Libraries from accidentally overriding the same dispatch key for the same op, which would lead to undefined behavior.

### 6.4 Module-Level impl() (torch/library.py lines 767-950)

```python
@functools.singledispatch
def impl(qualname, types, func=None, *, lib=None):
    # Maps device type strings to dispatch keys:
    # "cpu" -> "CPU", "cuda" -> "CUDA", "default" -> "CompositeExplicitAutograd"
    keys = set()
    for typ in types:
        if torch._C._parse_dispatch_key(typ):  # raw dispatch key
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    # For each key, create a Library fragment and register:
    for key in keys:
        use_lib.impl(qualname, func, key)
```

### 6.5 Module-Level register_autograd (torch/library.py lines 1313-1436)

```python
def register_autograd(op, backward, /, *, setup_context=None, lib=None):
    op = _resolve_op_for_registration(op, "register_autograd")
    if isinstance(op, CustomOpDef):
        op.register_autograd(backward, setup_context=setup_context)
        return
    # For non-CustomOpDef ops:
    info = _library.autograd.Info(backward, setup_context)
    autograd_kernel = _library.autograd.make_autograd_impl(op, info)
    namespace, opname = torch._library.utils.parse_namespace(qualname)
    use_lib = _library_for_registration(namespace, lib)
    use_lib.impl(opname, autograd_kernel, "Autograd", with_keyset=True)
```

**Two paths**: If the op is a `CustomOpDef`, delegate to its `register_autograd` method. Otherwise, create an `Info` dataclass and call `make_autograd_impl` directly, then register the resulting kernel to the `Autograd` dispatch key.

---

## 7. Integration with torch.compile (Dynamo Pipeline)

### 7.1 pt2_compliant_tag -- The Gateway to torch.compile

**Every custom op created via `torch.library.custom_op` gets `pt2_compliant_tag` automatically** (see `_with_pt2_compliant_tag` in Section 2.2).

In Dynamo's `output_graph.py` (line 3589-3618):
```python
def check_pt2_compliant_op(output_graph, kind, target, args, kwargs):
    if kind != "call_function":
        return

    if isinstance(target, torch._ops.OpOverload):
        if torch.Tag.pt2_compliant_tag in target.tags:
            # COMPLIANT: track as compliant custom op
            output_graph.compliant_custom_ops.add(target)
            return
        # NON-COMPLIANT: graph break or warning
        output_graph.non_compliant_ops.add(target)
        if config.only_allow_pt2_compliant_ops:
            unimplemented("Encountered non-PT2-compliant op", ...)
```

**Without `pt2_compliant_tag`**: An op causes a graph break unless `config.only_allow_pt2_compliant_ops` is False (default). This is why C++ TORCH_LIBRARY ops that don't have the tag break under torch.compile.

### 7.2 Dynamo Trace Rules for custom_op Modules

From `torch/_dynamo/trace_rules.py`:

**MOD_SKIPLIST** (Dynamo skips/does not trace into these modules):
- `"torch._custom_op"` (line 3585)
- `"torch._custom_ops"` (line 3586)
- `"torch._library"` (line ~3620)
- `"torch.library"` (line ~3640)

**MOD_INLINELIST** (Dynamo inlines into these modules):
- `"torch._library.custom_ops"` (line 3518)
- `"torch._library.autograd"` (line ~3520)

The distinction is important:
- `torch._custom_op` is **skipped** because it's the deprecated wrapper -- Dynamo should not trace into its internal registration logic
- `torch._library.custom_ops` is **inlined** because the `CustomOpDef.__call__` method (which just calls `self._opoverload(*args, **kwargs)`) should be traced through, not graph-broken

### 7.3 How Custom Ops Appear in FX Graphs

When Dynamo encounters a call like `torch.ops.mylib.numpy_sin(x)`, it:
1. Recognizes the `OpOverload` as a `call_function` target
2. Checks `pt2_compliant_tag` -- present (auto-added by `custom_op`)
3. Adds `call_function torch.ops.mylib.numpy_sin.default(x)` to the FX graph
4. The op is **opaque** to Dynamo -- it doesn't peek inside the implementation

This is the primary reason for custom ops: **they create a boundary that torch.compile cannot cross**, preventing the compiler from trying to optimize code it doesn't understand (like CUDA kernels, NumPy calls, etc.).

### 7.4 AOTAutograd -- Joint Forward+Backward Tracing

AOTAutograd traces through the custom op's autograd implementation. When `register_autograd` was called:
- The `Generated` autograd.Function class is used
- `forward` calls `op.redispatch(after_autograd_keyset)` -- this goes to the backend kernel
- `backward` calls `info._backward_fn(ctx, *grads)` -- this is the user's backward formula

Both must be **traceable** by AOTAutograd. If they call non-traceable operations (like `data_ptr()`), the tracing will fail.

### 7.5 Inductor -- Backend Compilation

In Inductor, custom ops appear as `call_function` nodes in the FX graph. Inductor treats them as **opaque operations** (similar to `ExternKernel`). It does not try to decompose or fuse them. The op's fake impl (Meta kernel) is used only for shape inference during tracing.

For custom ops to work efficiently with Inductor, the `register_fake` impl must be correct so that shape inference works, and the `register_autograd` impl must be traceable so that the backward pass can be compiled.

### 7.6 Error Paths in torch.compile

From `torch/_dynamo/utils.py`:

**DataDependentOutputException** (lines 4040-4057):
```python
if isinstance(cause, DataDependentOutputException):
    hints = [
        "Consider wrapping the operator into a PyTorch-understood custom operator "
        "(see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)",
    ]
```

**DynamicOutputShapeException** (lines 4060-4085):
- If `capture_dynamic_output_shape_ops=True`: continues with data-dependent shapes
- If False: graph breaks

**UnsupportedOperatorException** (lines 4090+):
- If op has no Meta/Fake kernel: suggests `register_fake`

**NotImplementedError during FX node execution** (lines 4250-4266):
```python
hints += [
    "If the op is a custom op, did you implement a fake tensor implementation? "
    "(e.g. with `@my_custom_op.register_fake`)",
    "If the op is a PyTorch op, please file an issue to PyTorch.",
]
```

---

## 8. torch.ops Namespace -- How Custom Ops Become Callable

### 8.1 The Three-Level Lazy Binding

From `torch/_ops.py`:

**Level 1: torch.ops** (lines 1455-1480):
```python
class _OpsNamespace(types.ModuleType):
    def __getattr__(self, name):
        # Creates torch.ops.my_namespace
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace
```

**Level 2: _OpNamespace** (lines 1346-1415):
```python
class _OpNamespace(types.ModuleType):
    def __getattr__(self, op_name):
        # Creates torch.ops.my_namespace.my_op
        qualified_op_name = f"{namespace_name}::{op_name}"
        op, overload_names = _get_packet(qualified_op_name, module_name)
        opoverloadpacket = OpOverloadPacket(qualified_op_name, op_name, op, overload_names)
        setattr(self, op_name, opoverloadpacket)
        return opoverloadpacket
```

**Level 3: OpOverloadPacket** (lines 1166-1340):
```python
class OpOverloadPacket:
    def __getattr__(self, key):
        # Resolves torch.ops.my_namespace.my_op.default
        # (or other overload names)
```

The magic: when `torch.ops.mylib.numpy_sin` is first accessed:
1. `torch.ops.__getattr__("mylib")` creates `_OpNamespace("mylib")`
2. `torch.ops.mylib.__getattr__("numpy_sin")` looks up `mylib::numpy_sin` in the C++ dispatcher, creates `OpOverloadPacket`
3. `torch.ops.mylib.numpy_sin(x)` calls `OpOverloadPacket.__call__`, which resolves the default overload and dispatches

### 8.2 CustomOpDef.__call__ (line 864-865)

```python
def __call__(self, *args, **kwargs):
    return self._opoverload(*args, **kwargs)
```

The `CustomOpDef` is callable and simply delegates to its `_opoverload` (an `OpOverload` object). This means:
- `numpy_sin(x)` (via CustomOpDef) and `torch.ops.mylib.numpy_sin(x)` (via OpOverload) are **equivalent**
- Both go through the same C++ dispatcher

### 8.3 _refresh_packet (lines 1424-1435)

When a new overload is added to an existing OpOverloadPacket (e.g., a new Library.define for the same op name with a different overload), `_refresh_packet` re-fetches the op from C++ and updates the packet.

---

## 9. SimpleLibraryRegistry -- The Python-Level Registration Mirror

### 9.1 Architecture (torch/_library/simple_registry.py)

```python
class SimpleLibraryRegistry:
    def __init__(self):
        self._data: dict[str, SimpleOperatorEntry] = {}

    def find(self, qualname):
        # Creates entry if not exists
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        return res

singleton = SimpleLibraryRegistry()
```

### 9.2 SimpleOperatorEntry

```python
class SimpleOperatorEntry:
    def __init__(self, qualname):
        self.fake_impl: FakeImplHolder       # for register_fake
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder  # for register_torch_dispatch
        self.effect: EffectHolder             # for _register_effectful_op
        self.symm_mem_args: SymmMemArgsHolder  # for register_symm_mem_args
```

This registry mirrors registrations that **don't directly correspond to a DispatchKey**. For example, `register_fake` registers to `DispatchKey.Meta` via the Library system, but also tracks the fake impl here for lookup by FakeTensor mode.

---

## 10. The Full Dispatch Key Flow for a Custom Op Call

When you call `numpy_sin(x)` with a CUDA tensor that requires grad:

```
1. CustomOpDef.__call__(x)
2. -> self._opoverload(x)  [OpOverload.__call__]
3. -> C++ Dispatcher routes by tensor dispatch keys:
   - x is CUDA + requires_grad -> DispatchKeySet = {CUDA, Autograd}
4. Dispatcher hits Autograd first (higher priority):
   -> autograd_impl(keyset, x)  [from make_autograd_impl]
5. Since grad is enabled and x.requires_grad:
   -> Generated.apply(x, Metadata(keyset, {}))
6. Generated.forward(ctx, x, Metadata):
   -> with _C._AutoDispatchBelowAutograd():
       -> op.redispatch(keyset & _after_autograd_keyset, x)
       -> this removes Autograd from keyset, leaving {CUDA}
       -> Dispatcher hits CUDA key:
          -> backend_impl(x)  [the register_kernel("cuda") kernel]
       -> result = fn(x)  [user's numpy_sin implementation]
   -> setup_context(ctx, inputs=(x,), output=result)  [if registered]
   -> return result
7. In backward:
   -> Generated.backward(ctx, grad_output)
   -> backward(ctx, grad_output)  [user's backward function]
   -> returns gradient
```

For a **meta/FakeTensor** tensor (torch.compile tracing):

```
1. CustomOpDef.__call__(fake_x)
2. -> self._opoverload(fake_x)  [OpOverload.__call__]
3. -> C++ Dispatcher routes by FakeTensor dispatch keys:
   - fake_x is Meta -> DispatchKeySet = {Meta, ...}
4. Dispatcher hits Meta key:
   -> meta_kernel(fake_x)  [from construct_meta_kernel]
   -> fake_impl_holder.kernel(fake_x)
   -> self._abstract_fn(fake_x)  [user's register_fake function]
   -> returns shape-only result
```

---

## 11. Real-World Usage Patterns

### 11.1 vLLM Custom Op Pattern

vLLM uses `torch.library.custom_op` for Triton kernels:

```python
@torch.library.custom_op("vllm::attention", mutates_args=())
def attention(...) -> Tensor:
    # Triton kernel implementation

@attention.register_fake
def _(...) -> Tensor:
    # Shape inference only

@attention.register_autograd(backward_fn, setup_context=setup_context_fn)
# Or: no autograd (inference-only)
```

### 11.2 Factory Function Pattern

```python
@custom_op("mylib::bar", mutates_args={}, device_types="cpu")
def bar(device: torch.device) -> Tensor:
    return torch.ones(3)
# device_types + no Tensor inputs -> BackendSelect dispatch key registered
```

### 11.3 Inplace Op Pattern

```python
@custom_op("mylib::numpy_sin_", mutates_args={"x"}, device_types="cpu",
           tags=torch.Tag.inplace)
def numpy_sin_(x: Tensor) -> Tensor:
    np.sin(x.numpy(), out=x.numpy())
    return x
# Auto-generates trivial fake impl: returns args[0]
# No autograd support for inplace ops
```

### 11.4 Out= Op Pattern

```python
@custom_op("mylib::numpy_sin_out", mutates_args={"out"}, device_types="cpu",
           tags=torch.Tag.out)
def numpy_sin_out(x: Tensor, *, out: Tensor) -> Tensor:
    np.sin(x.numpy(), out=out.numpy())
    return out
# Auto-generates trivial fake impl: returns kwargs["out"]
# No autograd support for out= ops
```

### 11.5 Data-Dependent Shape Pattern

```python
@custom_op("mylib::nonzero", mutates_args=())
def nonzero(x: Tensor) -> Tensor:
    return torch.tensor(np.stack(np.nonzero(x.numpy()), axis=1), device=x.device)

@nonzero.register_fake
def _(x):
    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()  # symbolic size
    return x.new_empty([nnz, x.dim()], dtype=torch.int64)
```

### 11.6 Triton Op Pattern

```python
@torch.library.triton_op("mylib::triton_kernel", mutates_args=())
def triton_kernel(...):
    # Triton kernel that gets captured during tracing

@torch.library.wrap_triton(triton_kernel)
def wrapped(...):
    # Wrapped version for eager execution
```

`triton_op` and `wrap_triton` (from `torch/_library/triton.py`) are special variants for Triton kernels that get captured as `TorchInGraphFunctionVariable` in Dynamo (line 413 of trace_rules.py).

---

## 12. Key Architectural Insights

### 12.1 Two APIs, One Dispatcher

| Feature | `torch.library.custom_op` (modern) | `torch._custom_op` (deprecated) |
|---------|-------------------------------------|----------------------------------|
| Entry point | `custom_op("ns::name", mutates_args=)` | `custom_op("ns::name")` decorator |
| Schema | Auto-inferred from type hints | Auto-inferred or manual |
| Autograd | `register_autograd(backward, setup_context=)` | `impl_backward` + `impl_save_for_backward` |
| Fake impl | `register_fake(fn)` | `impl_abstract(fn)` |
| Kernel | `register_kernel("cuda", fn)` | `impl("cuda")(fn)` |
| pt2_compliant | **Auto-added** | Not added (graph breaks!) |
| Backward format | `(ctx, *grads) -> tuple of grads | `(ctx, saved, *grads) -> dict of grads |
| Context saving | via `setup_context(ctx, inputs, output)` | via `impl_save_for_backward(inputs, output) -> to_save` |
| Metadata handling | `Metadata(keyset, keyword_only_args)` hidden arg | namedtuple wrapping |
| List[Tensor] | Supported via `supports_tensorlist` | Supported via pytree flatten/unflatten |

### 12.2 Dispatch Key Priority for Custom Ops

```
Functionalize          (highest - for functionalization passes)
ADInplaceOrView        (for inplace/view mutation tracking)
Autograd               (for custom backward formulas)
  -> redispatch to:
    AutocastCUDA       (for AMP autocast)
    CUDA/CPU           (for device-specific kernels)
    CompositeExplicitAutograd (for default kernels)
Meta                   (for shape inference / fake impl)
BackendSelect          (for factory functions)
Python                 (for Python dispatcher modes)
```

Custom ops register kernels at: Autograd, Meta, CUDA/CPU/CompositeExplicitAutograd, BackendSelect, ADInplaceOrView, and optionally Autocast.

### 12.3 Why custom_op is the Right API for torch.compile

1. **pt2_compliant_tag**: Auto-added, no manual configuration needed
2. **Guaranteed Meta kernel**: Even without `register_fake`, trivial impls are auto-generated for inplace/out ops
3. **Proper Autograd**: The generated `autograd.Function` class handles redispatch correctly, preventing infinite loops
4. **Opaque boundary**: Dynamo treats the op as opaque, not tracing into its implementation
5. **Shape inference**: `register_fake` + `get_ctx().new_dynamic_size()` enables data-dependent shape ops

### 12.4 Anti-Patterns and Gotchas

1. **Wrong `mutates_args`**: If you declare `mutates_args=()` but the op actually mutates an input, aliasing constraints are violated -- undefined behavior
2. **Missing `register_fake`**: For non-inplace/non-out ops, `register_fake` is mandatory for torch.compile. Without it, `NotImplementedError` is raised during FakeTensor dispatch
3. **Non-traceable backward**: If `backward_fn` uses `data_ptr()` or depends on global state, AOTAutograd tracing will fail
4. **Using torch._custom_op instead of torch.library.custom_op**: Missing `pt2_compliant_tag` means graph breaks under torch.compile
5. **Registering Autograd for out= ops**: Explicitly forbidden -- out= ops don't support autograd
6. **Inplace + return different tensor**: If `torch.Tag.inplace` is set but the op returns something other than `args[0]`, a RuntimeError is raised

### 12.5 OPDEFS Registry and WeakValueDictionary

```python
OPDEFS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
```

Custom op definitions are stored in a `WeakValueDictionary`, meaning they can be garbage collected if no strong reference exists. The `global_registry` in the deprecated API uses a regular dict (keeping all CustomOps alive forever).

---

## 13. Dispatch Key Routing Summary

| Scenario | Dispatch Path |
|----------|--------------|
| CUDA tensor, requires_grad=True | Autograd -> (redispatch) -> CUDA |
| CUDA tensor, requires_grad=False | CUDA (direct) |
| CPU tensor, requires_grad=True | Autograd -> (redispatch) -> CPU |
| Meta/FakeTensor (torch.compile) | Meta (fake impl) |
| Factory function (no tensor args) | BackendSelect -> (redispatch) -> CPU/CUDA |
| Inplace op on CUDA | ADInplaceOrView -> (version bump) -> Autograd -> CUDA |
| AMP autocast region | AutocastCUDA -> (cast inputs) -> (redispatch without AutocastCUDA) -> Autograd -> CUDA |

---

## 14. File Reference Map

| File | Key Content |
|------|-------------|
| `torch/_library/custom_ops.py` | `CustomOpDef` class, `custom_op()` function, `register_kernel`, `register_fake`, `register_autograd` |
| `torch/_library/autograd.py` | `make_autograd_impl`, `Info`/`InfoProtocol`, `supports_tensorlist` |
| `torch/_library/fake_impl.py` | `FakeImplHolder`, `FakeImplCtx`, `construct_meta_kernel`, `get_ctx` mechanism |
| `torch/_library/simple_registry.py` | `SimpleLibraryRegistry`, `SimpleOperatorEntry`, `singleton` |
| `torch/_library/utils.py` | `Kernel`, `RegistrationHandle`, `lookup_op`, `is_functional_schema`, `can_generate_trivial_fake_impl`, `is_out`, `is_inplace` |
| `torch/_library/infer_schema.py` | `infer_schema()`, type annotation -> C++ schema conversion |
| `torch/_library/effects.py` | `EffectType`, `EffectHolder` |
| `torch/_library/fake_profile.py` | `unsafe_generate_fake_kernels`, `OpProfile`, `_generate_fake_kernel` |
| `torch/library.py` | `Library` class, module-level `impl`, `register_fake`, `register_autograd`, `get_ctx` |
| `torch/_custom_op/impl.py` | Deprecated `CustomOp` class, `custom_op` decorator (redirects to `torch.library`) |
| `torch/_custom_op/autograd.py` | Deprecated `autograd_kernel_indirection`, `construct_autograd_kernel`, `save_pytree_for_backward` |
| `torch/_ops.py` | `OpOverloadPacket`, `_OpNamespace`, `_OpsNamespace`, `_refresh_packet` |
| `torch/_dynamo/output_graph.py` | `check_pt2_compliant_op`, `compliant_custom_ops`, `non_compliant_ops` |
| `torch/_dynamo/trace_rules.py` | `MOD_SKIPLIST` (torch._custom_op), `MOD_INLINELIST` (torch._library.custom_ops) |
| `torch/_dynamo/utils.py` | Error handling for `DataDependentOutputException`, `DynamicOutputShapeException` |

---

## 15. RTX 4090 Implications

- **Custom Triton kernels**: Use `torch.library.custom_op` + `register_kernel("cuda")` for Triton kernels. The `pt2_compliant_tag` enables `torch.compile` compatibility without graph breaks.
- **INT4/INT8 custom ops**: Quantization/dequantization kernels can be wrapped as custom ops with `register_fake` for shape inference and `register_autograd` for straight-through estimator backward.
- **vLLM contribution**: The NIXL KV connector metrics PR uses custom ops for RDMA operations -- these need `register_fake` for torch.compile tracing.
- **rLLM Tinker**: In-process custom ops (LoRA merge, sampling) can use `custom_op` with `device_types="cuda"` for direct CUDA kernel integration.

---

*Analysis completed June 15, 2026. Source from pytorch/pytorch main branch.*
