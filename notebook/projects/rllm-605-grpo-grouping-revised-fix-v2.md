# rLLM #605 GRPO Grouping Bug — Revised Fix Proposal (v2)

> 2026-06-19 | Revised after maintainer jeffreysijuntan feedback
> ★★★★★★★★ V1 fix (remove trajectory.name from grouping key) was INCORRECT — breaks multi-agent use case
> ★★★★★★★★ V2 fix: add configurable `grouping_mode` to `TransformConfig` — preserves both use cases

---

## Background

**Original fix (V1):** Changed grouping key from `task_id:name` to `task_id` only.
- Closed PR #667 on main repo (per user mandate)
- Maintainer jeffreysijuntan feedback: trajectory.name is designed for multi-agent grouping (solver+judge by role). Removing it breaks this use case.

**Revised fix (V2):** Add `grouping_mode` field to `TransformConfig`:
- `"by_task_id"` (default): Group all trajectories for the same task together → optimal for GRPO
- `"by_task_id_and_name"` (alternative): Group by task AND agent name → preserves multi-agent separation

---

## V2 Changes

### 1. config.py — TransformConfig (add grouping_mode field)

```python
class TransformConfig:
    """Configuration for the episode-to-group transformation pipeline."""

    # Name imputation
    impute_missing_names: bool = True
    default_traj_name: str = _DEFAULT_TRAJ_NAME
    drop_unnamed_traj: bool = False

    # Reward configuration
    broadcast: bool = True

    # Grouping configuration (NEW)
    # - "by_task_id": Group all trajectories for the same task together (optimal for GRPO)
    # - "by_task_id_and_name": Group by task AND agent name (multi-agent separation)
    grouping_mode: str = "by_task_id"

    @classmethod
    def from_config(cls, transform_config: DictConfig, *, broadcast: bool = True) -> "TransformConfig":
        return cls(
            impute_missing_names=transform_config.get("impute_missing_names", True),
            default_traj_name=transform_config.get("default_traj_name", _DEFAULT_TRAJ_NAME),
            drop_unnamed_traj=transform_config.get("drop_unnamed_traj", False),
            broadcast=broadcast,
            grouping_mode=transform_config.get("grouping_mode", "by_task_id"),  # NEW
        )
```

### 2. transform.py — _build_trajectory_groups (add grouping_mode parameter)

```python
def _build_trajectory_groups(
    episodes: list[Episode],
    compact_filtering_config: CompactFilteringConfig | None = None,
    grouping_mode: str = "by_task_id",  # NEW parameter
) -> list[TrajectoryGroup]:
    """
    Build TrajectoryGroups from episodes based on the configured grouping strategy.

    Args:
        episodes: List of episodes to group
        compact_filtering_config: Configuration for compact filtering
        grouping_mode: How to group trajectories into TrajectoryGroups.
            - "by_task_id": Group all trajectories for the same task together.
              Optimal for GRPO where all trajectories share a baseline.
            - "by_task_id_and_name": Group trajectories by task AND agent name.
              Preserves multi-agent separation (e.g. solver vs judge).
    """
    trajectories_by_name: dict[str, list[Trajectory]] = defaultdict(list)
    metadata_by_name: dict[str, list[dict]] = defaultdict(list)

    for episode in episodes:
        termination_reason = episode.termination_reason or TerminationReason.UNKNOWN
        if compact_filtering_config and compact_filtering_config.should_mask(termination_reason):
            continue
        task_id = episode.task_id
        for trajectory in episode.trajectories:
            if len(trajectory.steps) == 0:
                continue
            # Compute grouping key based on grouping_mode (NEW)
            if grouping_mode == "by_task_id":
                group_key = task_id
            elif grouping_mode == "by_task_id_and_name":
                group_key = f"{task_id}:{trajectory.name}"
            else:
                raise ValueError(
                    f"Unknown grouping_mode: {grouping_mode}. "
                    "Must be 'by_task_id' or 'by_task_id_and_name'"
                )
            trajectories_by_name[group_key].append(trajectory)
            metadata_by_name[group_key].append(...)
    ...
```

### 3. transform.py — _default_trajectory_grouping_hook (pass grouping_mode)

```python
def _default_trajectory_grouping_hook(...):
    ...
    trajectory_groups = _build_trajectory_groups(
        episodes, compact_filtering_config,
        grouping_mode=transform_config.grouping_mode,  # NEW: pass grouping_mode
    )
    ...
```

---

## Why V2 is Better than V1

| Aspect | V1 (just remove name) | V2 (configurable) |
|--------|----------------------|------------------|
| GRPO scenario | Works | Works (default="by_task_id") |
| Multi-agent scenario | BROKEN | Works (grouping_mode="by_task_id_and_name") |
| Backward compatibility | NO | YES (default preserves existing behavior for non-GRPO users) |
| Maintainer concerns | Not addressed | Fully addressed |
| Config complexity | None | Minimal (1 field, sensible default) |

---

## Application Instructions

**For user to apply manually:**

1. Clone rllm-org/rllm locally:
   ```bash
   cd ~/workspace && git clone https://github.com/rllm-org/rllm.git
   ```

2. Create fix branch:
   ```bash
   cd ~/workspace/rllm && git checkout -b fix/grpo-configurable-grouping-v2
   ```

3. Apply the revised files:
   ```bash
   cp ~/workspace/rollout-infra/notebook/projects/rllm-667-revised-config.py rllm/trainer/algorithms/config.py
   cp ~/workspace/rollout-infra/notebook/projects/rllm-667-revised-transform.py rllm/trainer/algorithms/transform.py
   ```

4. Commit:
   ```bash
   git add rllm/trainer/algorithms/config.py rllm/trainer/algorithms/transform.py
   git commit -m "fix: add configurable grouping_mode to TransformConfig for GRPO grouping (#605)"
   ```

5. Push to fork:
   ```bash
   git remote add fork https://github.com/Jackie2049/rllm.git
   git push fork fix/grpo-configurable-grouping-v2
   ```

6. Create PR on main repo (after user review):
   ```bash
   gh pr create --repo rllm-org/rllm --head Jackie2049:fix/grpo-configurable-grouping-v2 --base main
   ```

---

## Files

- Revised config.py: `notebook/projects/rllm-667-revised-config.py`
- Revised transform.py: `notebook/projects/rllm-667-revised-transform.py`

---

## References

- Issue #605: https://github.com/rllm-org/rllm/issues/605
- PR #667 (CLOSED): https://github.com/rllm-org/rllm/pull/667
- Maintainer feedback: jeffreysijuntan comment on #667
- GRPO advantage computation: notebook/projects/rllm-grpo-advantage-computation-reading.md
