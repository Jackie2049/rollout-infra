# PyTorch #187653 NanDetectMode — Forward-Pass NaN/Inf Detection

> 2026-06-18 | RTX 4090 GRPO debugging tool | +127 LOC (62 source + 65 tests), OPEN
> Author: rajfirke (Raj Vijay Firke) | Labels: open source, topic: not user facing
> Based on: albanD/subclass_zoo nan_detect.py prototype
> Fixes: #160016 (Requesting improved detect_anomaly for NaN detection)
> ★★★★★★★★ Forward-pass NaN detection → complements detect_anomaly (backward-only)
> ★★★★★★★★ GRPO relevance: locate NaN's FIRST appearance in model forward → pinpoint root cause

---

## PR Summary

Adds `NanDetectMode(TorchDispatchMode)` to `torch/utils/nan_detect.py` that detects NaN (and optionally Inf) in the output of every ATen operation during the forward pass. Raises `RuntimeError` immediately with the offending operation name.

```python
from torch.utils.nan_detect import NanDetectMode

with NanDetectMode():
    out = model(x)
# RuntimeError: Function aten.add.Tensor returned NaN values
```

Key facts:
- +62 LOC in `torch/utils/nan_detect.py` (implementation)
- +65 LOC in `test/test_nan_detect.py` (8 integration tests)
- `check_inf=True` keyword-only option → also detects ±Inf (off by default)
- Uses `tree_flatten` from `torch.utils._pytree` to handle nested outputs
- `try/except NotImplementedError` → gracefully handles meta/fake tensors
- Context manager restores state on exception exit → no corruption
- `torch._disable_dynamo` automatically applied to `__torch_dispatch__` via `__init_subclass__`

---

## Complete Source Code (torch/utils/nan_detect.py, 62 lines)

```python
"""Forward-pass NaN/Inf detection via TorchDispatchMode.

Usage::

    with torch.utils.nan_detect.NanDetectMode():
        out = model(x)
    # RuntimeError raised immediately when any op produces NaN

Based on the prototype at https://github.com/albanD/subclass_zoo/blob/main/nan_detect.py
"""

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class NanDetectMode(TorchDispatchMode):
    """Detect NaN (and optionally Inf) in the output of every operation.

    When enabled, every ATen operation is followed by a check of its outputs.
    If any floating-point output tensor contains NaN (or non-finite values when
    ``check_inf=True``), a ``RuntimeError`` is raised immediately with the name
    of the offending operation.

    This complements :func:`torch.autograd.detect_anomaly`, which only checks
    for NaN during the backward pass.

    Args:
        check_inf (bool): If ``True``, also raise on ``±Inf`` values.
            Default: ``False`` (only check for NaN).

    Example::

        >>> with NanDetectMode():
        ...     x = torch.tensor([1.0, float('nan')])
        ...     y = x + 1  # raises RuntimeError
    """

    def __init__(self, *, check_inf: bool = False) -> None:
        super().__init__()
        self.check_inf = check_inf

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)
        flat_res, _ = tree_flatten(res)
        for t in flat_res:
            if not isinstance(t, torch.Tensor):
                continue
            if not t.is_floating_point() or t.numel() == 0:
                continue
            try:
                if self.check_inf:
                    if not torch.isfinite(t).all():
                        raise RuntimeError(
                            f"Function {func} returned non-finite values"
                        )
                elif torch.isnan(t).any():
                    raise RuntimeError(f"Function {func} returned NaN values")
            except NotImplementedError:
                pass
        return res
```

---

## Original Prototype (albanD/subclass_zoo nan_detect.py)

The prototype was a 25-line bare-bones implementation with key differences:

```python
# Original: used (t != t).any() trick — IEEE 754 NaN self-comparison
# PR version: uses torch.isnan(t).any() — canonical, dtype-safe

# Original: no check_inf option
# PR version: check_inf=True keyword-only, uses torch.isfinite(t).all()

# Original: no dtype/empty tensor filtering
# PR version: t.is_floating_point() and t.numel() == 0 guards

# Original: no docstring, no module-level documentation
# PR version: full docstring, module doc, example

# Original: torch.is_tensor(t) type check
# PR version: isinstance(t, torch.Tensor) — more pythonic
```

Key evolution: `(t != t).any()` → `torch.isnan(t).any()`
- `t != t` is IEEE 754 NaN trick (NaN != NaN), cryptic but fast
- `torch.isnan()` is canonical, handles bfloat16/float16 properly
- Performance: nearly identical (`.any()` reduction dominates cost on both)

---

## TorchDispatchMode Internals — How __torch_dispatch__ Intercepts

### The Dispatch Stack

PyTorch operations go through a multi-layer dispatch system:
```
Python API (torch.add) → Python dispatch key → Autograd → __torch_dispatch__ → Backend (CPU/CUDA)
```

`TorchDispatchMode` intercepts at the `__torch_dispatch__` level — AFTER Python decompositions, AFTER autograd, BEFORE the actual kernel runs. This means:
- You see ATen-level ops (`torch.ops.aten.add.Tensor`), not high-level Python ops
- Decomposed ops (e.g., `torch.nn.functional.gelu` decomposes into multiple ATen ops) are visible individually
- You intercept BEFORE the actual CUDA/CPU computation, enabling pre-computation checks

### Mode Stack Mechanics

The mode stack is managed by C-level functions (`_push_on_torch_dispatch_stack`, `_pop_torch_dispatch_stack`):
1. `__enter__`: Pushes mode onto TLS (thread-local) dispatch stack
2. Every op dispatch: C++ checks stack top → calls mode's `__torch_dispatch__`
3. Mode calls `func(*args, **kwargs)` → op executes → mode inspects result
4. `__exit__`: Pops mode from stack, restores global flags

Three global flags maintained via deque history stacks (supporting nested modes):
- `_is_in_torch_dispatch_mode` — any dispatch mode active?
- `_is_in_non_infra_torch_dispatch_mode` — non-infra (user) mode active?
- `_is_in_any_mode_without_ignore_compile_internals` — should intercept compiled internals?

### __init_subclass__ Auto-Dynamo-Disable

```python
class TorchDispatchMode:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._should_skip_dynamo():  # Default: True
            if "__torch_dispatch__" in cls.__dict__:
                raw = cls.__dict__["__torch_dispatch__"]
                if not isinstance(raw, classmethod):
                    cls.__torch_dispatch__ = torch._disable_dynamo(raw, recursive=True)
```

NanDetectMode's `__torch_dispatch__` is AUTOMATICALLY wrapped with `torch._disable_dynamo` — meaning torch.compile will not trace through it. This prevents FakeTensorMode conflicts (see below).

### Comparison with __torch_function__ / TorchFunctionMode

| Feature | TorchFunctionMode (__torch_function__) | TorchDispatchMode (__torch_dispatch__) |
|---|---|---|
| Intercept level | High-level Python API | Low-level ATen ops |
| Decompositions | NOT visible (single call) | Visible individually |
| Dispatch stage | Before key resolution | After autograd, before backend |
| Use cases | API interception, tensor subclasses | Backend tracing, logging, policy enforcement |
| Coverage | Misses some internal ops | Covers ALL ops including decomposed |

NanDetectMode uses TorchDispatchMode specifically because:
- Forward pass may decompose into MANY ATen ops → need to check EACH one
- `__torch_function__` would miss decomposed ops (e.g., softmax → multiple ATen ops)
- ATen-level granularity = pinpoint exactly which op produced NaN

---

## Pytree Flattening Mechanism

### What tree_flatten Does

```python
from torch.utils._pytree import tree_flatten

# Operations can return complex nested structures:
res = {"logits": tensor, "hidden_states": (tensor1, tensor2), "loss": scalar}
flat_res, spec = tree_flatten(res)
# flat_res = [tensor, tensor1, tensor2, scalar]
# spec encodes: dict{'logits': leaf, 'hidden_states': tuple[2 leaves], 'loss': leaf}
```

`tree_flatten` recursively walks nested containers (dicts, lists, tuples, NamedTuples, custom registered types) and extracts leaf values into a flat list, recording structure in a `TreeSpec`.

### Why This Matters for NanDetectMode

ATen operations can return diverse output types:
- Single tensor: `aten.add.Tensor` → returns `Tensor`
- Multiple tensors: `aten.topk.default` → returns `(Tensor, Tensor)`
- NamedTuples: `aten.max.default` → returns `namedtuple(values, indices)`
- Dict outputs: some custom ops return `Dict[str, Tensor]`

Without pytree flattening, you'd need to handle each case separately. `tree_flatten` handles ALL cases uniformly — iterate flat list, check each element.

### TreeSpec Details

`TreeSpec` encodes structure without leaf values:
- `type`: container type (list, tuple, dict)
- `children_specs`: nested TreeSpec objects
- `num_leaves`: total leaf count
- Dict keys: sorted alphabetically for deterministic traversal

NanDetectMode discards the spec (`flat_res, _ = tree_flatten(res)` — `_` is the spec) because it only needs to check values, not reconstruct structure.

---

## Skip Mechanism — What Gets Checked and What Doesn't

### Three Skip Conditions

```python
for t in flat_res:
    if not isinstance(t, torch.Tensor):  # SKIP 1: non-tensor (scalars, None, strings)
        continue
    if not t.is_floating_point() or t.numel() == 0:  # SKIP 2+3: int/bool/empty tensors
        continue
    try:
        # ... NaN/Inf check ...
    except NotImplementedError:  # SKIP 4: meta/fake tensors
        pass
```

**SKIP 1: `not isinstance(t, torch.Tensor)`**
- Non-tensor leaves: Python ints, floats, bools, None, strings
- These cannot contain NaN by definition
- Example: `aten.size.default` returns `List[int]`

**SKIP 2: `not t.is_floating_point()`**
- Integer tensors (torch.int32, torch.int64, torch.int8)
- Boolean tensors (torch.bool)
- These dtypes have no NaN representation (IEEE 754 NaN only exists for float types)
- Example: `aten.argmax.default` returns LongTensor — no NaN possible

**SKIP 3: `t.numel() == 0`**
- Empty tensors (shape like `(0,)` or `(0, 0)`)
- `torch.isnan(empty_tensor).any()` returns False but unnecessary computation
- `torch.isfinite(empty_tensor).all()` returns True but unnecessary
- Skipping saves dispatch overhead on shape-only operations

**SKIP 4: `except NotImplementedError` (meta/fake tensors)**
- Meta tensors: `torch.empty(2, device='meta')` — no real data storage
- Fake tensors: `FakeTensor` (used by torch.compile tracing) — shape/dtype only
- Both raise `NotImplementedError` when you try `torch.isnan()` or `torch.isfinite()`
- The try/except silently skips them → NanDetectMode works during tracing without crashing
- This is the key design choice for torch.compile compatibility

### Why check_inf=False by Default

Per profPlum's request on issue #160016: many models intentionally use Inf values:
- **Attention masks**: `torch.where(mask, 0.0, float('-inf'))` — standard practice
- **Padding masks**: `-inf` for ignored positions in softmax
- **Loss masking**: `-inf` to zero out unwanted loss terms
- **MoE router logits overflow**: THIS is the RTX 4090 case where Inf detection matters

Default off = NanDetectMode doesn't fire on every attention mask. Opt-in for Inf when debugging overflow scenarios.

---

## Exception Context Handling

### RuntimeError with Operation Identity

```python
raise RuntimeError(f"Function {func} returned NaN values")
raise RuntimeError(f"Function {func} returned non-finite values")
```

`func` is the ATen operation object (e.g., `torch.ops.aten.add.Tensor`). This provides:
- Exact op name: `aten.add.Tensor`, `aten.div.Tensor`, `aten.matmul.default`
- Not a Python traceback — the op name itself identifies the culprit
- In pdb: you can break at the exact line that triggered the RuntimeError

### Context Manager Restoration

The `TorchDispatchMode.__exit__` method handles both normal and exception cases:
```python
def __exit__(self, exc_type, exc_val, exc_tb):
    # Pop mode from dispatch stack
    # Restore global flags from deque history
    # Mode is fully removed regardless of exception type
```

This means:
- After a NaN-triggered RuntimeError, the mode is properly popped
- Subsequent operations run WITHOUT NanDetectMode (no lingering interception)
- No state corruption: global flags restored to pre-mode values
- Deque-based history stack supports nested mode usage correctly

Test verification (`test_context_manager_restores`):
```python
x = torch.tensor([float("nan")])
try:
    with NanDetectMode():
        x + 1  # raises RuntimeError
except RuntimeError:
    pass
y = x + 1  # works normally — NaN propagates silently (mode is gone)
assert torch.isnan(y).any()  # True — mode no longer intercepting
```

---

## Relationship to detect_anomaly — Backward vs Forward

### detect_anomaly Limitations (from issue #160016)

The issue author (profPlum) identified 4 critical problems:

**Problem 1: Un-pdb-able traceback**
- detect_anomaly raises inside C++ autograd internals
- Only provides a Python traceback, can't drop into pdb at the source
- NanDetectMode raises in Python → pdb works directly

**Problem 2: Backward-first detection**
- detect_anomaly flags the FIRST NaN in backward pass (reverse order)
- Forward NaN at sqrt(x-2) → backward NaN detected at next op
- Example: `y = sqrt(x - 2)` → NaN forward → `y = sqrt(y)` → traceback points to SECOND op
- NanDetectMode catches at the EXACT forward op that produced NaN

**Problem 3: Silent NaN pass-through**
- `loss = torch.log(-inp)` → NaN loss value
- `loss.backward()` → detect_anomaly raises NO error despite NaN loss!
- The NaN was in the forward pass, but backward pass had no NaN gradient
- NanDetectMode would catch this immediately in forward

**Problem 4: Backward-only NaN unreported**
- NaN originating ONLY in backward pass can go unreported
- Example: `y = (x-1)**0.5` → forward is finite (sqrt of 0 = 0) → backward is NaN (d/dy = 1/(2*sqrt(0)) = inf)
- But detect_anomaly can miss this in certain optimizer configurations
- NanDetectMode doesn't help here (it's forward-only) → still need detect_anomaly for backward NaN

### Complementary Usage Pattern

```python
# FULL NaN debugging: use BOTH modes
with NanDetectMode():                          # Forward NaN
    with torch.autograd.detect_anomaly():      # Backward NaN
        out = model(x)
        loss = compute_loss(out)
        loss.backward()

# If NaN in forward → NanDetectMode fires FIRST (before backward runs)
# If NaN in backward only → detect_anomaly fires
# If NaN in both → NanDetectMode fires first (forward), prevents backward from running
```

### Timeline Comparison

```
NanDetectMode (forward):    op1 → op2 → op3[NaN!] → RAISE (immediate)
detect_anomaly (backward):  op1 → op2 → op3 → ... → backward: grad_op3 → grad_op2[NaN!] → RAISE (misleading)
```

---

## Integration Tests (test/test_nan_detect.py, 65 lines, 8 tests)

| Test | Purpose | Key Insight |
|---|---|---|
| `test_nan_detected` | NaN in basic op raises RuntimeError | Core functionality |
| `test_clean_tensors_pass` | Clean ops pass through normally | No false positives |
| `test_inf_not_detected_by_default` | Inf passes with check_inf=False | Default behavior safe for attention masks |
| `test_inf_detected_when_enabled` | Inf raises with check_inf=True | Opt-in overflow detection |
| `test_integer_tensors_skipped` | Int tensor ops not checked | dtype guard works |
| `test_empty_tensors_skipped` | Empty tensor ops not checked | numel guard works |
| `test_nn_module` | nn.Linear with NaN input raises | Module integration |
| `test_context_manager_restores` | After exception, mode is gone | No state corruption |

Missing tests (potential follow-up PR additions):
- Multi-output ops (topk returns 2 tensors)
- Dict/namedtuple outputs
- Nested NanDetectMode contexts
- torch.compile interaction (NanDetectMode should be skipped during tracing)
- Mixed-precision (bf16/fp16 NaN detection)
- Gradient computation under NanDetectMode (backward should NOT be intercepted)

---

## Issue #160016 Discussion — Key Comments

**ezyang**: "This isn't possible [to detect backward NaN in forward]. At forward time, we don't know what tangents will participate in the computation." → Confirms forward-only scope for NanDetectMode.

**ezyang**: Also suggested tracing backward into FX graph and pdb-ing that — but acknowledges it's auto-generated code needing interpretation.

**albanD**: Confirmed detect_anomaly limitations are real. Suggested: "your ask is not torch.autograd.detect_anomaly() at all but an extended version of nan_detect.py"

**soulitzer**: "As a first step, we can bring a basic version of nan_detect to core as a mode, off by default. TorchDispatchMode should allow for more fine-grained detection compared to TorchFunctionMode."

**soulitzer**: Also noted: "Integrating into torch.autograd.detect_anomaly() (enablable via a flag) sounds good" — potential future integration path.

**profPlum**: Requested check_inf toggle: "Please also add a toggleable option to raise on infinity (off by default), because #4 is a nasty example if you just get the NaN error inside the optimizer you don't know where it actually came from & it would be consistent with numpy.seterr()." → Implemented as `check_inf=True` keyword.

---

## Performance Overhead Analysis

### TorchDispatchMode Baseline

| Scenario | Overhead per op |
|---|---|
| No mode active (optimized C++ check) | <1 ns |
| Trivial pass-through mode | ~1-3 µs |
| Mode with computation (like NanDetectMode) | ~5-15 µs |
| Nested modes | Cumulative (~1-3 µs per layer) |

### NanDetectMode Specific Overhead

For each operation, NanDetectMode performs:
1. `func(*args, **kwargs)` — original computation (unchanged)
2. `tree_flatten(res)` — flatten nested outputs (~1-2 µs)
3. For each float tensor in flat output:
   - `isinstance(t, torch.Tensor)` check — negligible
   - `t.is_floating_point()` check — negligible
   - `t.numel() == 0` check — negligible
   - `torch.isnan(t).any()` — ~5-10 µs (depends on tensor size)
   - OR `torch.isfinite(t).all()` — ~5-10 µs (slightly more work)

For a model with ~1000 forward ops, typical overhead:
- Small tensors (4x4): ~5 µs per op → ~5 ms total forward → negligible for training
- Large tensors (4096x4096): ~10 µs per op → ~10 ms total → noticeable but acceptable for debugging
- Production: DO NOT leave NanDetectMode on — use only for debugging sessions

### RTX 4090 Impact

For GRPO training with Qwen3-8B on RTX 4090 (24 GiB):
- Forward pass: ~50ms per batch
- NanDetectMode overhead: ~5-10 ms additional → 10-20% slowdown
- Acceptable for debugging (1-2 minutes per step vs 10-20 minutes for manual NaN hunting)
- MUST disable for production training

---

## Practical GRPO Debugging Workflow for RTX 4090

### Step-by-Step NaN Debugging Protocol

```
★★★★★★★★★ RTX 4090 GRPO NaN Debugging Protocol:

Step 1: REPRODUCE NaN
  → Run training until NaN appears in loss/reward
  → Record: which step, which batch, which model state

Step 2: FORWARD PASS NaN DETECTION
  → Wrap model forward in NanDetectMode:
    with NanDetectMode():
        outputs = policy_model(queries)
  → If RuntimeError → know EXACTLY which ATen op produced NaN
  → If no error → NaN originates in BACKWARD pass (skip to Step 4)

Step 3: FORWARD NaN ROOT CAUSE ANALYSIS
  → From RuntimeError: "Function aten.div.Tensor returned NaN values"
  → → Division by zero? Check divisor tensor values before that op
  → → Use pdb: break at NanDetectMode __torch_dispatch__ line
  → → Check inputs to offending op: print(args) before func(*args, **kwargs)

  Common forward NaN patterns on RTX 4090:
  → DSV4 attention indexer → wrong positions → NaN in flash attention
  → SM89 batch invariance → different results → accumulation to NaN
  → LoRA weight corruption → FSDP2 CPU leak → stale weights → NaN in matmul
  → MoE router overflow → logits > 65504 → fp16 overflow → NaN after softmax

Step 4: BACKWARD PASS NaN DETECTION (if forward is clean)
  → Use detect_anomaly:
    with torch.autograd.detect_anomaly():
        loss.backward()
  → Traceback points to backward op with NaN gradient
  → Root cause: gradient computation produces NaN (e.g., sqrt(0) derivative)

Step 5: COMBINED FORWARD+BACKWARD (comprehensive debugging)
  → Use both modes simultaneously:
    with NanDetectMode(check_inf=True):  # check Inf too for overflow
        with torch.autograd.detect_anomaly():
            outputs = model(inputs)
            loss = grpo_loss(outputs, rewards, advantages)
            loss.backward()

Step 6: GRPO-SPECIFIC NaN CHECKS
  → Advantage normalization: std == 0 → division by zero
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
  → Log-prob ratio: exp(new_logp - old_logp) overflow
    ratio = torch.exp(new_logp - old_logp).clamp(0.8, 1.2)
  → KL penalty: large KL divergence → Inf in loss
  → Mixed-precision: bf16 advantage computation → precision loss
  → MUST compute advantages/loss in fp32, even with bf16 model
```

### verl GRPO Integration Example

```python
# verl GRPO training loop with NanDetectMode debugging
from torch.utils.nan_detect import NanDetectMode

def debug_grpo_step(policy_model, ref_model, queries, rewards):
    """Debug a single GRPO step for NaN issues."""
    # Step 1: Check rewards are finite
    assert torch.isfinite(rewards).all(), f"NaN/Inf in rewards: {rewards}"

    # Step 2: Compute advantages with epsilon guard
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    assert torch.isfinite(advantages).all(), f"NaN/Inf in advantages: {advantages}"

    # Step 3: Forward pass with NanDetectMode
    try:
        with NanDetectMode(check_inf=True):  # Catch both NaN and overflow
            new_logps = policy_model(queries).log_probs
            ref_logps = ref_model(queries).log_probs  # reference model
    except RuntimeError as e:
        print(f"Forward NaN detected: {e}")
        # pdb.set_trace()  → inspect offending op inputs
        raise

    # Step 4: Compute ratio and loss in fp32
    ratio = torch.exp(new_logps.float() - ref_logps.float())
    ratio = torch.clamp(ratio, 0.8, 1.2)  # PPO-style clipping

    loss = -(ratio * advantages.float()).mean()
    assert torch.isfinite(loss), f"NaN/Inf in loss: {loss}"

    # Step 5: Backward with detect_anomaly if forward is clean
    with torch.autograd.detect_anomaly():
        loss.backward()
```

### DeepSpeed ZeRO-2 + NanDetectMode Integration

```python
# DeepSpeed ZeRO-2 on RTX 4090 with NaN debugging
import deepspeed
from torch.utils.nan_detect import NanDetectMode

# DeepSpeed config MUST have overlap_comm=False on single GPU
ds_config = {
    "zero_optimization": {
        "stage": 2,           # ZeRO-2 only (ZeRO-3 = pure overhead on single GPU)
        "overlap_comm": False, # MUST False! overlap_comm+compile = NaN (#8061)
        "offload_optimizer": {"device": "cpu"},  # CPU_Adam for 24 GiB
    },
    "gradient_clipping": 1.0,  # MUST set explicitly! default 0 → no clipping (#8068)
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model, config=ds_config
)

# Training loop with conditional NanDetectMode
for step, batch in enumerate(dataloader):
    # Only enable NanDetectMode when debugging (10-20% overhead)
    if DEBUG_NAN:
        with NanDetectMode(check_inf=True):
            loss = model_engine(batch)
    else:
        loss = model_engine(batch)

    model_engine.backward(loss)
    model_engine.step()
```

---

## Known RTX 4090 NaN Scenarios and NanDetectMode Detection

| NaN Source | ATen Op Detected by NanDetectMode | Root Cause | Fix |
|---|---|---|---|
| DSV4 DSA indexer | `aten.flash_attention.default` | Wrong positions from stale cache | enforce_eager=True, per-step dynamic |
| SM89 batch invariance | `aten.mm.default` or fused ops | Different numerical results per batch size | Inductor SM89 fusion guard (P9) |
| overlap_comm (#8061) | `aten.add.Tensor` or gradient ops | Multi-stream race on single GPU | overlap_comm=False |
| FSDP2 CPU leak (#6468) | `aten.mm.default` | Stale weights from leaked parameters | FSDP v1 workaround or fix #6468 |
| LoRA rank mismatch (#6782) | `aten.mm.default` | rank=64 breaks EOS → NaN in weights | Use rank=32, alpha=64 |
| MoE router overflow | `aten.softmax.default` | fp16 logits > 65504 → overflow → NaN | Cast router to fp32 before softmax |
| GRPO zero std (#605) | `aten.div.Tensor` | std=0 → division by zero | Add eps=1e-8 to std |
| gradient_clipping=0 (#8068) | Multiple ops (accumulated NaN) | No clipping → unbounded gradients | Set gradient_clipping=1.0 |

---

## Future Integration Paths

### Potential detect_anomaly Integration (soulitzer suggestion)

soulitzer suggested: "Integrating into torch.autograd.detect_anomaly() (enablable via a flag) sounds good." This could mean:
- `torch.autograd.detect_anomaly(check_forward=True)` → enables NanDetectMode internally
- Single context manager for both forward and backward NaN detection
- Consistent with numpy.seterr() model (profPlum's request)

### torch.compile Compatibility

Current status: NanDetectMode's `__torch_dispatch__` is auto-disabled for Dynamo via `_should_skip_dynamo=True`. This means:
- Under torch.compile, NanDetectMode is silently bypassed
- No NaN detection during compiled execution
- soulitzer noted: "More work may be needed to integrate w/ compile and other things"

Potential future: custom Dynamo rewrite rule that preserves NaN checks in the compiled graph.

### Broader NaN Detection Ecosystem

PyTorch NaN detection evolution:
1. `torch.autograd.detect_anomaly()` — backward only, C++ internals, un-pdb-able
2. `albanD/subclass_zoo/nan_detect.py` — forward only, prototype, TorchDispatchMode
3. `torch.utils.nan_detect.NanDetectMode` — forward only, production, TorchDispatchMode
4. Future: unified `detect_anomaly(check_forward=True)` — forward + backward in one call
5. Future: `torch.compile`-compatible NaN detection — compiled graph with NaN guards

---

## Cross-Framework Connections

| Framework | NaN Issue | NanDetectMode Relevance |
|---|---|---|
| DeepSpeed #8061 | overlap_comm+compile NaN | Can detect NaN at EXACT forward op before backward |
| DeepSpeed #8068 | gradient_clipping=0 → NaN | Forward NaN detection won't help (backward issue) → need detect_anomaly |
| DeepSpeed #8072/#8073 | ZeRO-3+PEFT regression | Forward NaN from dtype mismatch → NanDetectMode catches it |
| verl #6468 | FSDP2 CPU leak → stale weights | Forward NaN from corrupted weights → NanDetectMode catches it |
| verl #6782 | LoRA rank=64 breaks EOS | Forward NaN from weight corruption → NanDetectMode catches it |
| verl #6572 | Determinism issues | NanDetectMode as debugging tool for non-deterministic NaN |
| rLLM #605 | GRPO grouping bug → zero std | Forward NaN from advantage normalization → NanDetectMode catches division |
| rLLM #663 | Step.output=None → rewards=0.0 | Not NaN but related: silent pass-through → NanDetectMode philosophy |
| Megatron #5394 | Muon clipping stalls | Backward NaN → NanDetectMode won't help → need detect_anomaly |
| SGLang #27097 | multi-LoRA determinism | NaN from non-deterministic ops → NanDetectMode catches first occurrence |
| vLLM-Ascend #10579 | MoE NaN from npu_moe_token_unpermute | Forward NaN → NanDetectMode catches exact op |
| vLLM-Ascend #10684 | DSA Hadamard ALL-ZERO after sleep/wake | Not NaN (zero values) but similar detection philosophy |

---

## References

- PyTorch #187653: NanDetectMode PR (OPEN, 2026-06-18, rajfirke)
- PyTorch #160016: original issue — improved detect_anomaly for NaN detection (profPlum)
- albanD/subclass_zoo nan_detect.py: original prototype (25 LOC, `t != t` trick)
- PyTorch `torch/utils/_python_dispatch.py`: TorchDispatchMode implementation
- PyTorch `torch/utils/_pytree/__init__.py`: tree_flatten implementation
- DeepSpeed #8061: overlap_comm+compile NaN → RTX 4090 production example
- DeepSpeed #8068: gradient_clipping=0 → NaN in backward only
- DeepSpeed #8072/#8073: ZeRO-3+PEFT LoRA regression → dtype mismatch NaN
- verl #6468: FSDP2 CPU memory leak → stale weights → forward NaN
- verl #6782: LoRA rank=64 breaks EOS → weight corruption → forward NaN
- verl #6572: full determinism tracking
- rLLM #605: GRPO grouping bug → zero std → division by zero NaN
- rLLM #663: Step.output=None → rewards all 0.0 → silent pass-through
- vLLM-Ascend #10579: MoE NaN from npu_moe_token_unpermute
- vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake
- Megatron #5394/#5395: Muon clipping stalls → backward NaN
- SGLang #27097: multi-LoRA determinism bug
- PyTorch #98849: TorchDispatchMode + torch.compile interaction issues
- PyTorch #28878: torch.isnan vs (t != t) performance comparison
