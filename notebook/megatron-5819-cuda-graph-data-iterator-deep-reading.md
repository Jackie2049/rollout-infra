# Megatron #5819: CUDA Graph Data Iterator Corruption Fix

**Date**: 2026-07-15 (Session 10)
**PR**: NVIDIA/Megatron-LM #5819
**Lines**: +31/-3
**Status**: OPEN (draft, empty description template)
**Pattern**: Static Cache Mutation — same family as DSV4 failures

---

## 1. Root Cause

`StaticBufferLoader.__call__` returns `static_buffers[stage][microbatch]` directly. Callers then **mutate** the contents of these static buffers (e.g., modifying tensors in-place), which corrupts the input data for subsequent CUDA graph replay calls.

CUDA graph captures expect **static input buffers** — same memory locations, same contents (or at least same structure). When the caller mutates the static buffer between graph replays, the graph reads corrupted/stale data.

---

## 2. Fix: `copy_container_shell()`

```python
def copy_container_shell(src):
    """Copy only containers, preserving tensor objects."""
    if isinstance(src, tuple):
        return tuple(copy_container_shell(i) for i in src)
    elif isinstance(src, list):
        return list(copy_container_shell(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_container_shell(src[k]) for k in src}
    else:
        return src  # tensors are NOT copied — same object reference
```

Key insight: This copies **container structure** (tuple/list/dict shells) but preserves **tensor object references**. The caller gets a new container shell they can mutate (add/remove items, replace references) without affecting the static buffer's container structure. But tensor data objects remain shared — CUDA graph still reads from the correct memory locations.

```python
# OLD: direct return (mutable!)
return StaticBufferLoader.static_buffers[stage][microbatch]

# NEW: container shell copy (safe)
return copy_container_shell(StaticBufferLoader.static_buffers[stage][microbatch])
```

---

## 3. Why This Pattern Matters for GRPO

CUDA graph is used in vLLM/SGLang rollout for fast decode. In GRPO training:
- Weight update → reset CUDA graph → recapture at new batch size
- If data_iterator static buffers are mutated between replays → corrupted generation output
- This is the same bug pattern family as vLLM #45552 (cumem sleep/wake) and SGLang #28679 (GDN degeneracy)

### Pattern family: Static Cache Mutation
| Bug | Framework | Cache Type | Mutation Mechanism | Fix |
|-----|-----------|-----------|-------------------|-----|
| #5819 | Megatron | CUDA graph data_iterator | Direct return of static buffer | copy_container_shell |
| #45552 | vLLM | cumem MemPool | Missing cuda.synchronize before free | Add sync |
| #28679 | SGLang | GDN decode state | Intermittent corruption over uptime | Periodic flush (#28695) |
| #28676 | SGLang | MXFP8 MoE shuffle | Weight-reload doesn't invalidate | MERGED July 1 |

---

## Session Stats
- **PR research**: #5819 (+31/-3), copy_container_shell pattern
- **Pattern**: Static Cache Mutation — 4 member bugs across 3 frameworks
- **GRPO relevance**: CUDA graph data corruption → incorrect rollout generation
