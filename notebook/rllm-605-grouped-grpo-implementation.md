# rLLM #605: Grouped GRPO Configurable grouping_key Implementation

## Issue Background

rllm-org/rllm #605 — GRPO grouping bug where `trajectory.name` causes unintended group splitting.

In `_build_trajectory_groups()`, the grouping key was hardcoded as `f"{task_id}:{trajectory.name}"`.
This means:
- For multi-agent workflows (solver+judge): Each role gets a separate group → **correct** behavior
- For single-agent workflows where all trajectories have the same name: All trajectories are grouped together → also works, but the `name` component is redundant
- For mixed workflows where some trajectories have names and others don't: The unnamed ones (defaulting to `"default"`) all get lumped together, creating incorrect groups

**Maintainer feedback**: jeffreysijuntan said trajectory.name is intended for multi-agent grouping. The fix must make grouping **configurable** rather than just removing name.

## Solution Design

### TransformConfig.grouping_key

Added `grouping_key: str = "task_id:name"` field to `TransformConfig`:
- Default `"task_id:name"` preserves current behavior (backwards-compatible)
- `"task_id"` groups all trajectories per task together (GROUPED_GRPO)
- `"name"` groups by trajectory name only, across tasks
- `"task_id:rollout_idx"` groups by task and rollout index
- Any colon-separated combination of supported fields

### _resolve_grouping_key()

New function in `transform.py` that resolves a colon-separated grouping key specification into a concrete group identifier.

Supported field names:
- `task_id` → episode.task_id
- `name` → trajectory.name
- `rollout_idx` → episode.rollout_idx

```python
def _resolve_grouping_key(episode, trajectory, grouping_key):
    field_map = {"task_id": episode.task_id, "name": trajectory.name, "rollout_idx": episode.rollout_idx}
    fields = grouping_key.split(":")
    parts = [str(field_map[f]) for f in fields if f in field_map]
    return ":".join(parts)
```

### GROUPED_GRPO Estimator

Added `GROUPED_GRPO = "grouped_grpo"` to `rLLMAdvantageEstimator` enum and registered via `@register_adv_estimator` hook (PR #742 pattern).

The advantage math is identical to standard GRPO (mean-center + normalize by group std). The difference is entirely in the grouping strategy:
- **GRPO** (grouping_key="task_id:name"): groups by task AND trajectory name → small per-role groups
- **GROUPED_GRPO** (grouping_key="task_id"): groups by task only → larger per-task groups with shared baseline

### Architecture Flow

```
episodes → _impute_trajectory_names → _build_trajectory_groups(grouping_key) → _validate_and_propagate_rewards → TrajectoryGroups → advantage estimator
```

The grouping happens at the transform level (before advantage computation), so the estimator receives whatever groups the transform pipeline produces.

## Files Changed

| File | Change |
|------|--------|
| `config.py` | GROUPED_GRPO enum, grouping_key field in TransformConfig, from_config update |
| `transform.py` | `_resolve_grouping_key()`, `_build_trajectory_groups` uses grouping_key, `_default_traj_grouping_hook` passes transform_config |
| `advantage.py` | `@register_adv_estimator(rLLMAdvantageEstimator.GROUPED_GRPO)` registration |

## Verification

8/8 structural checks passed on H20-3e server:
1. Enum has GROUPED_GRPO ✓
2. Enum has 8+ members ✓
3. TransformConfig has grouping_key ✓
4. grouping_key default="task_id:name" ✓
5. _resolve_grouping_key function exists ✓
6. _build_trajectory_groups calls _resolve_grouping_key ✓
7. GROUPED_GRPO registered in advantage estimators ✓
8. _default_traj_grouping_hook passes transform_config ✓

## Branch/PR

- Branch: `fix/grouped-grpo-adv-estimator-terminal-rl` on Jackie2049/rllm
- Base: `terminal-rl` (upstream active development branch)
- PR: https://github.com/Jackie2049/rllm/pull/2
- Commit: 3412b484 (+256/-5)

## Comparison with Previous Fix (PR #667)

PR #667 was a minimal 2-line fix that just changed the grouping key from `f"{task_id}:{trajectory.name}"` to `f"{task_id}"`. This was CLOSED per user mandate because:
1. It removed `name` from grouping entirely, breaking multi-agent workflows
2. It didn't make grouping configurable

The new PR #2 addresses both concerns:
1. Default grouping_key="task_id:name" preserves multi-agent behavior
2. Users can choose grouping_key="task_id" for single-agent GRPO
3. Any custom grouping strategy is supported via colon-separated fields

## Relationship to Cross-Framework Patterns

This fix addresses the same root issue identified in our cross-framework GRPO advantage comparison:
- All frameworks (verl/rLLM/TRL) use group-based normalization
- Group formation determines which trajectories share a baseline
- Misgrouped trajectories → incorrect advantage → poor training signal
- rLLM #605 is a specific instance of this general problem

See: `notebook/cross-framework-grpo-advantage-comparison.md`
