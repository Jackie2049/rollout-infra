# DeepSpeed #8130: ZeRO-3 Frozen Params + Activation Checkpoint Recompte Fix

## Overview
- **PR**: microsoft/DeepSpeed #8130 by qgallouedec (HuggingFace), July 10, 2026
- **Status**: OPEN — delock ✅ approved, tohtana ❌ changes requested, tjruwase/loadams pending
- **Bug lifetime**: Since Sept 2023 (deepspeedai/DeepSpeed#4332)
- **Severity**: ★★★★★★★★ CRITICAL for LoRA/PEFT + ZeRO-3 with gradient checkpointing

## The Bug: CheckpointError with Frozen Params

### What Happens
When using ZeRO-3 + gradient checkpointing (PyTorch **non-reentrant** mode, which is the HF default) + frozen parameters (LoRA adapters, quantized base model):

```
CheckpointError: Recomputed values ... have different metadata ...
```
Saved tensors have shape `[1024]` while recomputed tensors have shape `[0]`.

### Root Cause (3-step chain)

1. **Frozen params aren't detached**: PyTorch checkpoint logic does `x = x.detach() if x.requires_grad else x`. Frozen params (`requires_grad=False`) keep the **original tensor reference** alive.

2. **ZeRO-3 post-forward hook fires during recompute**: During backward's checkpoint recompute, ZeRO-3's forward hooks re-fire. The post-forward hook in `PartitionedParameterCoordinator` partitions params in-place: `param.data = torch.empty(0)`.

3. **Shape mismatch**: The checkpoint validation code compares the saved tensor (still `[1024]`) against the recomputed tensor (now `[0]` because ZeRO-3 shrank it) → `CheckpointError`.

Trainable params are fine because they ARE detached (`requires_grad=True`), so ZeRO-3 partitioning them doesn't corrupt saved references.

### Default Failure Path
Transformers switched `use_reentrant=False` as default → **every** PEFT/LoRA user hitting ZeRO-3 + checkpointing gets this crash:
- huggingface/transformers#47254
- huggingface/trl#5217
- huggingface/trl#4811

Downstream workarounds (trl#4951, trl#6356) force `use_reentrant=True`, which disables the more memory-efficient non-reentrant mode.

## The Fix

**File**: `deepspeed/runtime/zero/partitioned_param_coordinator.py`
**Method**: `PartitionedParameterCoordinator.release_sub_module`

Adds a guard: during a forward release that fires inside a backward pass (checkpoint recompute, detected via `torch._C._current_graph_task_id() != -1`), skip partitioning frozen params (`not param.requires_grad`). The param is "released normally by the ensuing backward."

**Key design**: Only affects frozen params during recompute — trainable params still partition module-by-module. Full finetuning memory profile unchanged.

## Review Status

| Reviewer | Status | Issue |
|----------|--------|-------|
| delock | ✅ Approved Jul 12 | Wanted regression test (added) |
| tohtana | ❌ Changes requested Jul 12 | **Release-lifetime issue**: when checkpoint block receives no-grad inputs, frozen param stays `AVAILABLE` across microbatches instead of releasing → coordinator accounting diverges |

**Author's response (Jul 13)**: Acknowledges tohtana's concern, doesn't fully understand the lifecycle, offers 3 paths:
1. Merge as-is (fixes common HF/PEFT crash)
2. Fold into more complete fix
3. Close in favor of global solution

## GRPO+LoRA Relevance

**Directly relevant for GRPO training on RTX 4090 with ZeRO-3**:
- TRL GRPO + QLoRA + ZeRO-3 = this crash
- RTX 4090's limited 24GB VRAM makes ZeRO-3 + gradient checkpointing essential
- Current workaround (trl#4951 forcing `use_reentrant=True`) wastes memory
- If merged, GRPO+LoRA gets proper non-reentrant checkpointing with frozen base model

## Cross-Framework Connections
- **Same bug family** as DeepSpeed #8072/#8073 (ZeRO-3+PEFT regression from #8066 per-policy dtype)
- **Ecosystem impact**: HuggingFace TRL, Transformers, PEFT all affected
- **Pattern**: P6 (Silent Corruption) — but manifests as hard crash (CheckpointError) rather than silent wrong results

## Monitoring
- tohtana's concern is real but narrow — only affects no-grad inputs to checkpoint blocks
- delock's approval suggests the common case is fixed
- Author from HuggingFace → high motivation to resolve for ecosystem
- 2 code owners pending (tjruwase, loadams)
- If resolved: UNLOCKS ZeRO-3 + LoRA + non-reentrant checkpointing for GRPO
