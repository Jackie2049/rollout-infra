# verl Unfixed Backend Detach Fix Analysis (PR #6699 Extension)

**Date**: 2026-07-14
**Based on**: Background agent deep exploration of verl engine backends
**Related**: verl PR #6699 (MERGED), our planned PR on Jackie2049/verl

## PR #6699 — What It Fixed

In `FSDPEngineWithLMHead.forward_step` (transformer_impl.py:1362-1372):
```python
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
```

**Root cause**: `model_output` tensors (log_probs/entropy) remain attached to autograd graph. `forward_backward_batch` holds these per-micro-batch outputs in `output_lst` until whole batch finishes → retained graph pins activation-checkpoint saved tensors per micro-batch → OOM with PEFT.

Additional safety: FSDP base engine pops `model_output` for training (`meta_info.pop("model_output", None)`) at line 673.

## Backends WITH Fix (in main verl)

1. **FSDP** — `FSDPEngineWithLMHead.forward_step` detaches model_output
   - Covers: `FSDPEngineWithValueHead`, `VeOmniEngineWithLMHead` (inherits via MRO), `VeOmniEngineWithValueHead`

## Backends MISSING Fix (3 backends)

### 2. Megatron — transformer_impl.py:1112-1148

- Fix location: `MegatronEngineWithLMHead.postprocess_micro_batch_func`
- `model_output` built via `self.prepare_model_outputs(output, data)` at line 1118
- Placed into output dict at line 1143 WITHOUT detaching
- **Critical ordering**: detach must happen AFTER `dynamic_cp_merge_output` (lines 1132-1141) because CP merge requires grad_fn
- Covers subclasses: `MegatronEngineWithValueHead`, `MindspeedEngineWithLMHead`, `MindSpeedMegatronEngineWithLMHead`, `MindspeedEngineWithValueHead`

### 3. AutoModel — transformer_impl.py:689-720

- Fix location: `AutomodelEngineWithLMHead.forward_step`
- `model_output` built at line 701, placed into output dict at line 714 WITHOUT detaching
- No `model_output` pop for training in `forward_backward_batch`
- No ValueHead variant exists

### 4. TorchTitan — transformer_impl.py:741-766

- Fix location: `TorchTitanEngineWithLMHead.forward_step`
- `model_output` built at line 749, placed into output dict at line 760 WITHOUT detaching
- No `model_output` pop for training in `forward_backward_batch`
- No ValueHead variant exists

## Additional Gap: Missing model_output Pop for Training

FSDP pops `model_output` entirely during training to prevent nested tensor accumulation even after detach. Missing in:
- **AutoModel**: `forward_backward_batch` line 260 — no pop
- **TorchTitan**: `forward_backward_batch` line 366 — no pop
- **VeOmni**: `forward_backward_batch` line 419 — no pop (tensors already detached though)
- **Megatron**: Different pipeline schedule data flow, pop pattern doesn't directly apply

## Proposed Fix (~15-20 LOC per backend)

For each unfixed backend, add the same detach pattern as FSDP:

```python
# In forward_step or postprocess_micro_batch_func, AFTER model_output computation:
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
```

Plus add `meta_info.pop("model_output", None)` in AutoModel/TorchTitan `forward_backward_batch`.

## Contribution Priority

| Backend | Impact | LOC | Users | Priority |
|---------|--------|-----|-------|----------|
| Megatron | HIGH (multi-GPU MoE) | ~5 | Enterprise | Tier 1 |
| AutoModel | MEDIUM (single GPU fallback) | ~5+2 | RTX 4090 | Tier 2 |
| TorchTitan | MEDIUM (experimental) | ~5+2 | Research | Tier 3 |

**Note**: Must check for existing PRs on verl-project/verl before submitting. This is a follow-up to #6699.

## Status
- Analysis complete: 3 backends need fix
- Need duplicate-work check before creating PR on Jackie2049/verl fork
- Megatron fix has ordering constraint (after CP merge)
