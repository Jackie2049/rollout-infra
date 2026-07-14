# Megatron #5789: MegatronFSDP.forward Bypasses Forward Hooks

## Overview
- **Issue**: NVIDIA/Megatron-LM #5789 by yuhezhang-ai, July 13, 2026
- **Status**: OPEN, 0 comments, 0 maintainer response
- **Severity**: ★★★★★★★★ CRITICAL for FSDP correctness
- **Fix**: One-line change: `self.module.forward(...)` → `self.module(...)`

## The Bug

### What Happens
In `MegatronFSDP.forward` (megatron-fsdp v0.5.0, ~line 1459 of `megatron_fsdp.py`):

```python
# BUGGY: calls raw forward(), bypassing __call__()
output = self.module.forward(*inputs, **kwargs)
```

### Why It's Wrong
`nn.Module.__call__()` is the mechanism that dispatches registered `forward_pre_hooks` and `forward_hooks`. By calling `.forward()` directly, all hooks registered on the root module are **silently skipped**.

### Concrete Impact
Megatron-FSDP relies on a pre-forward hook on the root module to **gather its parameters before user code executes**. Since the hook never fires:

1. **Root-owned parameters** point to storage that FSDP already freed after initialization
2. **Child FSDP units** work fine (they manage their own gather/release lifecycle)
3. **Result**: Wrong results or NaN/garbage values specifically from root module parameters

### Why Noticed
This bug is subtle because:
- Partial results are still correct (child units work)
- Only root parameters are wrong
- May manifest as slow training degradation rather than immediate crash
- Model may appear to train but produces incorrect outputs

## The Fix

```python
output = self.module(*inputs, **kwargs)
```

This dispatches through `nn.Module.__call__()`, which fires all registered hooks before entering `forward()`.

## FSDP bug cluster connection

This is part of a **3-bug FSDP cluster** filed by yuhezhang-ai on July 13:

| Bug | Issue | Scope | Fix |
|-----|-------|-------|-----|
| StorageResize use-after-free | #5788 | Bucket allocator frees without `record_stream` | Add `record_stream()` |
| Forward hook bypass | #5789 | `self.module.forward()` → hooks skipped | Use `self.module()` |
| Parameter state loss | #5790 | Plain Python attributes lost on rebuild | Preserve attrs during rebuild |

All three affect Megatron-FSDP v0.5.0's interaction with the rest of Megatron-LM. **ZERO maintainer response on all three as of July 14.**

## Cross-Framework

- Same pattern as: FP8 autocast context manager (`with ctx and fp8_ctx:` bug), context management errors
- Pattern: **P6 (Silent Corruption)** — model trains but produces wrong results from root params
- If using Megatron-FSDP for GRPO training: root module gradients silently wrong

## Monitoring
- 0 comments, 0 assignees, 0 labels
- Author has no upstream PRs linked
- Best path: user contributes fix (one line)
