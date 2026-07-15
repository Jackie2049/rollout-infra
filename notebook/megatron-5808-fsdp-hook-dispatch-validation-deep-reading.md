# Megatron #5808: FSDP Root Module Hook Dispatch Fix — Validates Our Fork PR #2

**Date**: 2026-07-15 (Session 10)
**PR**: NVIDIA/Megatron-LM #5808 (by wujingyue)
**Lines**: +45/-1 (1 core line + 44 test lines)
**Status**: OPEN
**Significance**: ★★★★★★★★ OFFICIAL fix for #5789 → VALIDATES our Jackie2049/Megatron-LM PR #2

---

## 1. Root Cause (Confirmed by NVIDIA)

`MegatronFSDP.forward` called **`self.module.forward(...)` directly**, bypassing `nn.Module.__call__`.

### Why this is wrong:
- `nn.Module.__call__` invokes: `_call_pre_hooks` → `forward()` → `_call_hooks`
- `self.module.forward()` skips BOTH pre-hooks and post-hooks
- For root-owned parameters, FSDP gather hook is registered as a **forward pre-hook**
- Skipping pre-hook → ungathered/freed parameter storage → wrong forward output or crash

### Our fork PR #2 (exact same fix):
```python
# OLD (bypasses hooks):
self.module.forward(*args, **kwargs)

# NEW (routes through nn.Module.__call__):
self.module(*args, **kwargs)
```

Change: +3/-1 (our PR had slightly different context but same core fix)

### NVIDIA's fix:
```python
# Same fix in megatron_fsdp.py
self.module(*args, **kwargs)  # instead of self.module.forward(*args, **kwargs)
```

Change: +1/-1 core + 44 test lines

---

## 2. NVIDIA Test Coverage (+44 lines)

NVIDIA added a regression test:
```python
def test_root_module_forward_uses_gathered_parameters(self):
    """Root module forward should use gathered parameters via pre-hooks."""
```

This test validates:
- Root-owned parameters under `optim_grads_params` sharding
- Parameters are properly gathered before forward execution
- Output is correct (not reading freed/ungathered storage)

---

## 3. Validation Timeline

| Date | Event |
|------|-------|
| July 13 | #5789 filed (by yuhezhang-ai) |
| July 13 | Our fork PR #2 created (independent fix) |
| July 15 | NVIDIA #5808 opened (official fix, same pattern) |

**Our fix was posted BEFORE the official fix**, demonstrating independent analysis capability.

---

## 4. Cross-Framework Pattern

This is the **Forward Hook Bypass** pattern family:

| Bug | Framework | Bypass Mechanism | Fix | Status |
|-----|-----------|-----------------|-----|--------|
| #5789 | Megatron MFSDP | `module.forward()` skips hooks | `module()` routes through `__call__` | #5808 OPEN |
| — | PyTorch FSDP | Never bypasses (uses `_handle_pre_hook`) | Already correct | — |
| — | DeepSpeed | ZeRO-3 uses separate gather flow | Different pattern | — |

### Lesson for new FSDP implementations:
- ALWAYS route module forward through `nn.Module.__call__`
- NEVER call `module.forward()` directly when hooks are registered
- Add regression tests for root-owned parameters

---

## 5. What This Means for Our PR Strategy

Our Jackie2049/Megatron-LM PR #2 is now **validated by the official NVIDIA fix**:
- Same root cause identification
- Same fix approach (`module()` vs `module.forward()`)
- NVIDIA added test coverage (44 lines) we should reference

### Action:
- Keep our PR #2 open on Jackie2049/Megatron-LM fork
- Add note referencing #5808 as official confirmation
- When #5808 merges to main, our fork branch can rebase

---

## Session Stats
- **PR comparison**: Our PR #2 (+3/-1) vs NVIDIA #5808 (+1/-1 core + 44 test)
- **Same fix**: `module()` replaces `module.forward()` — confirmed independent analysis
