# rLLM #605 Tier 1 Comment Draft — GRPO Grouping Bug

> 2026-06-18 | Tier 1 comment opportunity | Critical RTX 4090 blocker
> ★★★★★★★★ Source-level analysis confirming the bug + proposed fix with line numbers

---

## Comment Draft

Thanks for the thorough analysis — I've confirmed this bug at the source level and it's critical for GRPO correctness.

### Root cause confirmed

In `agent_workflow_trainer.py`, the line:
```python
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
```

assigns `trajectory.uid` (a unique UUID per trajectory) as the GRPO grouping key. Since each rollout gets a distinct `trajectory.uid`, GRPO groups have size 1, making `advantage ≈ raw reward` — this completely negates GRPO's variance-reduction mechanism.

### Verified impact

With `enable=False` and 4 rollouts of the same prompt, rewards [1, 0, 1, 0]:
- **Current (bug)**: advantages ≈ [1, 0, 1, 0] (group size 1 → no normalization)
- **Expected (fix)**: group mean = 0.5 → advantages ≈ [+0.87, -0.87, +0.87, -0.87]

This affects ALL 3 `stepwise_advantage` configs because the grouping key assignment happens before the config-specific logic.

### Proposed fix

**For `enable=False`** (1-line fix):
```python
# Replace:
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
# With:
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["task_ids"]
```

**For `mode=per_step`** (few lines):
```python
step_indices = batch.non_tensor_batch.get("step_indices", None)
if step_indices is not None:
    batch.non_tensor_batch["uid"] = [
        f"{task_id}_step{step_idx}"
        for task_id, step_idx in zip(batch.non_tensor_batch["task_ids"], step_indices)
    ]
else:
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["task_ids"]
```

**For `mode=broadcast`**: Same as `enable=False` (`task_ids`), plus transform fix for `is_last_step` emission.

### RTX 4090 implications

This bug makes rLLM Tinker GRPO completely unusable for training — any serious GRPO run will produce random advantages instead of normalized group comparisons. Until fixed, rLLM should not be used for GRPO training.

---

## References

- Source reading: rllm-605-grpo-grouping-bug-reading.md
- GRPO config reference: tools/rtx4090_grpo_config_reference.py (rLLM #3 BLOCKED by #605)
