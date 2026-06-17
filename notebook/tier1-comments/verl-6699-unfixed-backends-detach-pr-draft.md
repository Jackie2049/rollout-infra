# verl C9 PR Draft — Detach model_output in Unfixed Engine Backends

> 2026-06-18 | Tier 2 PR (P7 UNIQUE) | Target: verl-project/verl
> ★★★★★★★★ Apply same detach() fix from #6699 to 3 unfixed engine backends
> ★★★★★★★★ AutomodelEngine + MegatronEngine + TorchTitanEngine → same OOM root cause
> ★★★★★★★★ CRITICAL: MegatronEngine MUST detach AFTER dynamic_cp_merge_output (ordering nuance)

---

## PR Title

`[engine] fix: detach model_output to stop per-micro-batch graph retention in Automodel/Megatron/TorchTitan backends`

---

## PR Body

### What does this PR do?

Applies the same per-micro-batch GPU memory leak fix from #6699 to the three remaining engine backends that still return `model_output` tensors (log_probs/entropy) attached to the autograd graph.

**Root cause** (same as #6698/#6699): `forward_step` returns `model_output` containing tensors still in the autograd graph; `forward_backward_batch` collects these per-micro-batch outputs in `output_lst` until the whole batch finishes. The retained graph pins, per training micro-batch, the activation-checkpoint frame's saved embedding output (which requires grad under PEFT `enable_input_require_grads`) plus its gradient buffer. With LoRA + long sequences + gradient accumulation, this leaks ~0.16-0.27 GiB per micro-batch → OOM at ~400 micro-batches.

**Affected backends**:
1. **AutomodelEngine** (`verl/workers/engine/automodel/transformer_impl.py` lines ~708-712) — `model_output` dict returned without detach
2. **MegatronEngine** (`verl/workers/engine/megatron/transformer_impl.py` lines ~1013-1017 in `postprocess_micro_batch_func`) — `model_output` dict returned without detach
3. **TorchTitanEngine** (`verl/workers/engine/torchtitan/transformer_impl.py` lines ~730-734) — `model_output` dict returned without detach

**Already safe backends** (no change needed):
- FSDPEngine — fixed by #6699 (MERGED June 12)
- VeOmniEngine — inherits from FSDPEngineWithLMHead → fix applies automatically

### Fix pattern (same 1-line comprehension as #6699)

```python
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
```

This creates new tensors sharing storage but NOT in the autograd graph. Detached tensors have no `grad_fn` → no reference to parent graph → checkpoint frames can be garbage collected after the forward pass. The detached tensors are only consumed for metrics aggregation and postprocessing AFTER `loss.backward()` has already run → no gradient impact.

### CRITICAL ordering nuance for MegatronEngine

In `MegatronEngineWithLMHead.postprocess_micro_batch_func`, there is a `dynamic_cp_merge_output` block (lines ~998-1007) that operates on tensors WITH `grad_fn`. **The detach MUST happen AFTER this merge block**, not before. Detaching before would strip `grad_fn` from tensors that the CP merge step needs to operate on correctly.

```python
# In MegatronEngine postprocess_micro_batch_func:

# 1. dynamic_cp_merge_output block (lines ~998-1007) — MUST operate on tensors with grad_fn
#    ... merge logic here ...

# 2. THEN detach — AFTER merge completes
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
```

For AutomodelEngine and TorchTitanEngine, the detach goes right after the `metrics = {}` line, before the return statement — same position as #6699's fix in FSDPEngine.

### Memory impact

Before fix (same pattern as #6699's measured results on Qwen3-8B LoRA rank 32 GRPO):
- mb #250: 24.8 GiB → #300: 37.9 GiB → #350: ~50 GiB → #400: 64 GiB → **OOM**

After fix (FSDPEngine measured):
- 16.2 GiB → 16.2 GiB → 16.2 GiB → 16.2 GiB → **STABLE** ✅

The same 4x memory reduction is expected for all 3 unfixed backends once the detach fix is applied.

### Test

Adapt the existing regression test `tests/workers/test_engine_forward_step_detach_on_cpu.py` from #6699, or create a parametrized version that covers all engine backends. The test reproduces the retention mechanism in miniature (frozen embedding + `requires_grad_` output + checkpointed block + weakref). CPU-only, runs in `cpu_unit_tests`.

### API and Usage Example

No API change. Same as #6699 — detached tensors only consumed for metrics aggregation and postprocessing after `loss.backward()`.

### Design & Code Changes

1. `verl/workers/engine/automodel/transformer_impl.py` — `forward_step`: add detach comprehension for `model_output` after `metrics = {}` line (before return)
2. `verl/workers/engine/megatron/transformer_impl.py` — `postprocess_micro_batch_func`: add detach comprehension for `model_output` AFTER the `dynamic_cp_merge_output` block (CRITICAL ordering)
3. `verl/workers/engine/torchtitan/transformer_impl.py` — `forward_step`: add detach comprehension for `model_output` after `metrics = {}` line (before return)
4. `tests/workers/test_engine_forward_step_detach_on_cpu.py` — extend or parametrize regression test for all engine backends

### References

- Root cause issue: #6698
- FSDPEngine fix (MERGED): #6699
- Memory profiling data: Qwen3-8B LoRA rank 32 GRPO, mb #250→#400 (from #6699)

---

## Checklist Before Submitting

- [ ] Search for similar PRs: no existing PR addresses Automodel/Megatron/TorchTitan detach
- [ ] Format PR title: `[engine] fix: detach model_output to stop per-micro-batch graph retention`
- [ ] Apply pre-commit checks (ruff)
- [ ] Run CPU regression test
- [ ] CI triggered

---

## Implementation Notes

### AutomodelEngine fix location

In `verl/workers/engine/automodel/transformer_impl.py`, around line 705-707 (after `metrics = {}`):

```python
# Before (current):
metrics = {}
return {"model_output": model_output, ...}

# After (fixed):
metrics = {}
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
return {"model_output": model_output, ...}
```

### MegatronEngine fix location

In `verl/workers/engine/megatron/transformer_impl.py`, in `postprocess_micro_batch_func`, after the `dynamic_cp_merge_output` block (lines ~998-1007):

```python
# After dynamic_cp_merge_output completes (DO NOT detach before this block!)
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
```

### TorchTitanEngine fix location

In `verl/workers/engine/torchtitan/transformer_impl.py`, around line 727-729 (after `metrics = {}`):

```python
# Before (current):
metrics = {}
return {"model_output": model_output, ...}

# After (fixed):
metrics = {}
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
return {"model_output": model_output, ...}
```

---

## References

- verl #6698: https://github.com/verl-project/verl/issues/6698 (root cause issue)
- verl #6699: https://github.com/verl-project/verl/pull/6699 (FSDPEngine fix, MERGED)
- Source reading: notebook/projects/verl-6699-detach-memory-fix-reading.md
- Engine backend safety: tools/verl_engine_backend_safety.py
