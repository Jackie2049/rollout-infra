# rLLM #605 Tier 1 Comment Draft — GRPO Grouping Bug

> 2026-06-18 | Tier 1 comment opportunity | Critical RTX 4090 blocker
> ★★★★★★★★ Source-level analysis confirming the bug + proposed fix with line numbers

---

## Comment Draft

Thanks for the thorough analysis — I've confirmed this bug at the source level and it's critical for GRPO correctness.

### Root cause confirmed (source-level)

The grouping bug exists in **two locations** depending on the training path:

**Path 1 — Unified trainer (experimental)**: `transform.py:127`
```python
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
```
This groups by `task_id:trajectory.name` where `trajectory.name` comes from `agent.name` (types.py:552). Since each agent/flow has its own name, GRPO groups have size 1 — same prompt but different agent → separate group → no variance reduction.

**Path 2 — AgentWorkflowPPOTrainer (verl backend)**: `uid = step_ids`
As the original issue notes, `batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]` assigns trajectory.uid (unique UUID) as the grouping key → group size 1.

**Both paths**: GRPO groups by a key that's too granular (agent name or trajectory UID) instead of by task/prompt → group size = 1 → advantage ≈ raw reward → GRPO becomes REINFORCE.

### Verified impact

With 4 rollouts of the same prompt, rewards [1, 0, 1, 0]:
- **Current (bug)**: advantages ≈ [1, 0, 1, 0] (group size 1 → no normalization)
- **Expected (fix)**: group mean = 0.5 → advantages ≈ [+0.87, -0.87, +0.87, -0.87]

### Proposed fix

**Path 1 — Unified trainer (1-line fix)**: `transform.py:127`
```python
# Replace:
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
# With:
trajectories_by_name[task_id].append(trajectory)
```

**Path 2 — AgentWorkflowPPOTrainer (1-line fix)**:
```python
# Replace:
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
# With:
batch.non_tensor_batch["uid"] = batch.non_tensor_batch["task_ids"]
```

### Additional critical bug: #663 (MERGED June 17)

Note that #663 just fixed another critical bug: `Step.output` was always `None`, causing **ALL rewards = 0.0** for any training run before June 17. This means any prior rLLM GRPO training was completely invalid — not just grouping wrong, but rewards were literally zero. Combined with #605, GRPO was doubly broken.

### RTX 4090 implications

This bug makes rLLM Tinker GRPO completely unusable — advantages are raw rewards (no variance reduction). Until fixed, rLLM should not be used for GRPO training. verl groups correctly by `task_ids` (prompt) → all responses in one group → proper GRPO.

---

## References

- Source reading: rllm-605-grpo-grouping-bug-source-reading.md (transform.py:127 verified!)
- #663 fix: Step.output was None → ALL rewards = 0.0 → MERGED June 17
- GRPO config reference: tools/rtx4090_grpo_config_reference.py (rLLM #3 BLOCKED by #605)
