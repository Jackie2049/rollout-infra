# verl #7016 Deep Analysis — Qwen3-MoE FSDP2 Backward Failure

## Overview

**Issue**: https://github.com/volcengine/verl/issues/7016
**Filed**: July 11, 2026
**Severity**: CRITICAL for RTX 4090 GRPO

Two failure modes:
- **A. CheckpointError** (gradient checkpointing ON): data-dependent saved-tensor count
- **B. Native crash** (gradient checkpointing OFF): SIGSEGV in backward, no Python traceback

## Root Cause: FSDP2 MoE Gradient Graph Divergence

**PyTorch PR #174862** (skpark-rh, Feb-June 2026) directly addresses this: https://github.com/pytorch/pytorch/pull/174862

### Mechanism

1. **MoE Router Creates Data-Dependent Execution Paths**:
   - Each token's top-K expert selection varies per rank
   - The forward pass loops over experts, processing only selected tokens per expert via `torch.where(expert_mask[expert_idx])`
   - Results accumulated via `index_add_` → dynamic, data-dependent computation graph

2. **FSDP2 Assumes SPMD**:
   - FSDP2 `fully_shard` assumes Same Program, Multiple Data
   - NCCL reduce-scatter expects identical input **sizes** across all ranks
   - When different ranks activate different experts, different parameters produce gradients → different reduce-scatter input sizes

3. **Failure Mode A (gradient checkpointing ON)**:
   - Activation checkpointing saves intermediate activations during forward
   - During backward, saved-tensor count depends on which experts were visited
   - MoE router takes different expert-dispatch path during recomputation
   - This creates non-deterministic saved-tensor count → `CheckpointError`

4. **Failure Mode B (gradient checkpointing OFF)**:
   - Without checkpointing, the full graph is held in memory
   - FSDP2 tries to all-reduce/reduce-scatter gradients for all parameters
   - Different ranks have different parameter subsets with gradients
   - The NCCL collective sees mismatched input sizes → SIGSEGV

### The Fix (PR #174862)

```python
# In FSDP2 param collection: initialize zero buffer for unused params
# so reduce-scatter gets same input size across all ranks

# Key change in _fsdp_param.py:
self._zero_buf = torch.zeros(
    max_unused_param_size, device=param.device, dtype=param.dtype
)
# Use expand for zero-cost views of unused param gradients
unsharded_grads[idx] = self._zero_buf.expand(param_unsharded_size)
```

Guarded by `reduce_scatter_unused_params` flag (default=False for BC).

### Workaround Already Available

**Transformers PR #41580**: Consolidate all expert parameters into a single large `nn.Parameter`. Any expert computation creates gradients for ALL experts → no unused params → reduce-scatter input sizes match.

### Status

PR #174862: OPEN, last commit June 24, 2026. Reviewer (weifengpy) requested changes multiple times. Issues addressed:
- Zero-buf expansion broken DTensor params (FIXED)
- Local shard-is-zero heuristic incorrect for grad=None detection (FIXED with global mask)
- Need coordination with PR #170667

## Code-Level Analysis of PR #174862 Fix

### 1. Zero Buffer Initialization (`_fsdp_param.py`)

```python
# In init_dtype_attrs(): one-time allocation per FSDPParam
self._zero_buf = torch.zeros(1, dtype=grad_dtype, device=self.sharded_param.device)
```

Single element zero tensor allocated once per param. Used via `expand()` for zero-cost views.

### 2. Unsharded Zero Grad Data (`_fsdp_param.py`)

```python
@property
def unsharded_zero_grad_data(self) -> torch.Tensor:
    if self.is_dtensor:
        return self._get_grad_inner_tensor(torch.zeros_like(self.unsharded_param))
    else:
        # expand() is zero-cost → no memory allocation per call
        return self._get_grad_inner_tensor(self._zero_buf.expand(self._orig_size))
```

- DTensor path: uses `torch.zeros_like` (actual allocation needed for TP)
- Regular FSDP path: uses `_zero_buf.expand()` (zero-cost view, no memory allocation)
- This ensures NCCL reduce-scatter sees the same input size across all ranks

### 3. Post-Backward: Track Locally Used Params (`_fsdp_param_group.py`)

```python
def post_backward(self, *unused: Any):
    ...
    for i, fsdp_param in enumerate(self.fsdp_params):
        ...
        if not hasattr(fsdp_param, "_unsharded_param"):
            continue
        if ...:  # param has grad
            fsdp_params_with_grad.append(fsdp_param)
            unsharded_grads.append(fsdp_param.unsharded_zero_grad_data)
            self._locally_unused_params.add(i)  # Track as "potentially unused globally"
```

When `reduce_scatter_unused_params=True` and param requires_grad:
- If grad exists: normal unsharded_grad (real gradient)
- If NO grad: `unsharded_zero_grad_data` used (zero-buf via expand)
- Index tracked in `_locally_unused_params` for global coordination

### 4. Finalize Backward: Global Coordination (`_fsdp_param_group.py`)

```python
def finalize_backward(self):
    # Global all_reduce to find params unused on ALL ranks
    globally_used = torch.ones(len(self.fsdp_params), ...)
    for i in self._locally_unused_params:
        globally_used[i] = 0  # Mark as potentially unused
    dist.all_reduce(globally_used, op=dist.ReduceOp.MAX, ...)
    globally_used = globally_used.cpu()

    for i, fsdp_param in enumerate(self.fsdp_params):
        if globally_used is not None and not globally_used[i]:
            fsdp_param.sharded_param.grad = None  # Prevent optimizer corruption
```

- `all_reduce(MAX)`: If ANY rank used the param, `globally_used[i] = 1`
- Globally unused params: grad set to None → optimizer skips them
- Prevents Adam momentum/adaptive LR corruption from zero gradients

### Key Insight: Naming Confusion
`_locally_unused_params` actually tracks **locally USED** indices (params that had gradients on this rank). The name is misleading — it should be `_locally_used_params` since these are the params that were locally used, and their complement across all ranks determines global unused status.

## RTX 4090 GRPO Implications

### Single GPU (dp=1) — No cross-rank divergence
Failure mode B (SIGSEGV) likely does NOT occur on single GPU because there's only one rank. No NCCL reduce-scatter needed.

**BUT**: Failure mode A (CheckpointError with grad ckpt) CAN still occur because the data-dependent saved-tensor count is a **local** issue, not a cross-rank one. The autograd graph reconstruction during backward depends on which experts were selected during forward.

### MUST DO for RTX 4090
1. Use **FSDP1** instead of FSDP2 with Qwen3-MoE (confirmed working by reporter)
2. If FSDP2 is required: disable gradient checkpointing AND validate backward stability
3. Consider consolidating expert params into single `nn.Parameter` (transformers PR #41580)
4. Monitor PyTorch PR #174862 for merge

### Silent Corruption Risk
Even when training doesn't crash, the FSDP2+MoE gradient graph mismatch can cause **silent weight corruption** (mentioned in PR discussion — NCCL completes but sends incorrect weights/gradients). This is WORSE than a crash because training appears to succeed but model quality degrades.

## Cross-Framework Pattern

This is a manifestation of a broader pattern:
- **FSDP2 assumes SPMD** — same program, different data
- **MoE violates SPMD** — different programs (expert selection) per rank
- Any framework combining FSDP2 + MoE + data-dependent routing is affected

Frameworks affected: verl, Megatron-LM (when using FSDP2/MFSDP2), DeepSpeed (ZeRO-3)
