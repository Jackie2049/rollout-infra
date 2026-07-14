"""
rLLM #605 Fix: Configurable grouping key for GRPO advantage estimation

## Problem

On rLLM's main branch (pre-terminal-rl), GRPO groups trajectories by
`trajectory.uid` (a random UUID), which means every trajectory is its own
group → GRPO degenerates to REINFORCE with advantage=0.

On terminal-rl, the grouping is already fixed to `f"{task_id}:{trajectory.name}"`,
but this is hardcoded. Users need the ability to configure the grouping key
for different scenarios:

1. Default: `task_id:name` — group all completions of the same task by agent role
2. `task_id` — group all completions of the same task regardless of agent role
   (useful for single-agent GRPO where name is irrelevant)
3. `metadata.topic` — group by topic/category from trajectory metadata
4. Any custom key from trajectory metadata or episode attributes

## Implementation Plan

### Change 1: Add `grouping_key` to TransformConfig

```python
# rllm/trainer/algorithms/config.py
class TransformConfig:
    # ... existing fields ...
    grouping_key: str = "task_id:name"  # How to group trajectories for advantage computation
    # Options: "task_id:name" (default), "task_id", "metadata:<key>", or custom string
```

### Change 2: Modify _build_trajectory_groups to use grouping_key

```python
# rllm/trainer/algorithms/transform.py
def _build_trajectory_groups(
    episodes: list[Episode],
    compact_filtering_config: CompactFilteringConfig | None = None,
    grouping_key: str = "task_id:name",
) -> list[TrajectoryGroup]:
    trajectories_by_name: dict[str, list[Trajectory]] = defaultdict(list)
    metadata_by_name: dict[str, list[dict]] = defaultdict(list)

    for episode in episodes:
        termination_reason = episode.termination_reason or TerminationReason.UNKNOWN
        if compact_filtering_config and compact_filtering_config.should_mask(termination_reason):
            continue

        # Compute the grouping key for each trajectory
        for trajectory in episode.trajectories:
            if len(trajectory.steps) == 0:
                continue

            # Resolve the grouping key
            key = _resolve_grouping_key(episode, trajectory, grouping_key)
            trajectories_by_name[key].append(trajectory)
            metadata_by_name[key].append({
                "task_id": episode.task_id,
                "rollout_idx": episode.rollout_idx,
                "termination_reason": episode.termination_reason,
                "is_correct": episode.is_correct,
            })

    groups = []
    for name, trajectories in trajectories_by_name.items():
        groups.append(
            TrajectoryGroup(
                trajectories=trajectories,
                group_id=name,
                metadata=metadata_by_name[name],
            )
        )
    return groups


def _resolve_grouping_key(episode: Episode, trajectory: Trajectory, grouping_key: str) -> str:
    """Resolve a grouping key template into an actual key string.

    Supported formats:
    - "task_id:name" → f"{episode.task_id}:{trajectory.name}"
    - "task_id" → episode.task_id
    - "name" → trajectory.name
    - "metadata:<key>" → trajectory.metadata.get(<key>, "unknown")
    - "task_id:metadata:<key>" → f"{episode.task_id}:{trajectory.metadata.get(<key>, 'unknown')}"
    """
    parts = grouping_key.split(":")
    resolved_parts = []
    for part in parts:
        if part == "task_id":
            resolved_parts.append(episode.task_id)
        elif part == "name":
            resolved_parts.append(trajectory.name or "unnamed")
        elif part.startswith("metadata:"):
            meta_key = part[len("metadata:"):]
            resolved_parts.append(str(trajectory.metadata.get(meta_key, "unknown")))
        elif part == "uid":
            resolved_parts.append(trajectory.uid)
        else:
            resolved_parts.append(part)  # literal string
    return ":".join(resolved_parts)
```

### Change 3: Register `grouped_grpo` as an advantage estimator

This is a thin wrapper around GRPO that signals the transform layer to use
a custom grouping key. The actual advantage computation is identical to GRPO.

```python
# rllm/trainer/algorithms/advantage.py
@register_adv_estimator("grouped_grpo")
def calculate_grouped_grpo_advantages(
    rewards: list[np.ndarray],
    algorithm_config: AlgorithmConfig,
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """GRPO advantages with configurable grouping.

    This estimator delegates to standard GRPO advantage computation.
    The grouping is controlled upstream by TransformConfig.grouping_key.
    When grouping_key="task_id:name" (default), this is identical to GRPO.
    """
    return calculate_grpo_advantages(rewards, algorithm_config, **kwargs)
```

### Change 4: Add GROUPED_GRPO to rLLMAdvantageEstimator enum

```python
# rllm/trainer/algorithms/config.py
class rLLMAdvantageEstimator(str, Enum):
    GRPO = "grpo"
    REINFORCE = "reinforce"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    PRPO = "prpo"
    RLOO = "rloo"
    ECHO = "echo"
    GROUPED_GRPO = "grouped_grpo"  # Configurable grouping key
    OTHER = "other"
```

### Change 5: Wire grouping_key through the transform pipeline

In `transform_episodes_to_trajectory_groups`, pass `transform_config.grouping_key`
to `_build_trajectory_groups`:

```python
trajectory_groups = _build_trajectory_groups(
    episodes,
    compact_filtering_config,
    grouping_key=transform_config.grouping_key,
)
```

## Usage Examples

```python
# Default: identical to current behavior
config = AlgorithmConfig(estimator=rLLMAdvantageEstimator.GRPO)

# Single-agent GRPO (group all completions of same task together)
config = AlgorithmConfig(estimator=rLLMAdvantageEstimator.GROUPED_GRPO)
transform_config = TransformConfig(grouping_key="task_id")

# Multi-agent with topic grouping
config = AlgorithmConfig(estimator=rLLMAdvantageEstimator.GROUPED_GRPO)
transform_config = TransformConfig(grouping_key="task_id:metadata:topic")

# By trajectory name only (all tasks with same agent role grouped together)
transform_config = TransformConfig(grouping_key="name")
```

## Risk Assessment

- LOW risk: grouping_key="task_id:name" preserves current behavior exactly
- Backward compatible: TransformConfig.from_config defaults to "task_id:name"
- grouped_grpo estimator delegates entirely to GRPO (no computation change)
- _resolve_grouping_key has a clear mapping from key templates to resolved strings

## Testing

1. grouping_key="task_id:name" → identical to current behavior
2. grouping_key="task_id" → all completions of same task grouped together
3. grouping_key="uid" → each trajectory is its own group (reproduces #605 bug)
4. Custom metadata key → trajectories grouped by metadata attribute

## Issue Reference

https://github.com/rllm-org/rllm/issues/605
"""
