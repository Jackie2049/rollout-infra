# rLLM GRPO Grouping Strategy Revision Note

> 2026-06-20 | Comprehensive revision after PR #667 closure and maintainer feedback
> Priority: P9 C7 (HIGH) -- GRPO grouping bug requires configurable fix, not blanket removal

---

## Context

rLLM #605 (GRPO grouping bug) is a critical issue where `AgentWorkflowPPOTrainer` groups trajectories by `trajectory.uid` (composed as `task_id:trajectory.name`) instead of by prompt/task_id alone. This causes group_size=1 in standard GRPO usage, degrading GRPO to REINFORCE(baseline=0) with no variance reduction.

Our initial PR #667 proposed a 2-line fix that removed trajectory.name from the grouping key entirely. Maintainer jeffreysijuntan closed #667 after identifying that trajectory.name serves a legitimate design purpose: enabling multi-agent grouping where solver and judge trajectories are separated by role. Removing name altogether broke this multi-agent use case.

This note documents the revised approach: making the grouping strategy configurable rather than one-size-fits-all.

---

## 1. Original Bug Analysis (#605)

### 1.1 Root Cause

In `rllm/trainer/algorithms/transform.py` line 127, the grouping key is constructed as:

```python
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
```

The `trajectory.name` field comes from `agent.name` (types.py:552), which defaults to `_DEFAULT_TRAJ_NAME = "default_traj_name"`. In a standard single-agent GRPO setup, each trajectory gets a unique name, causing trajectories that share the same `task_id` (i.e., the same prompt) to be placed in **separate groups**.

Consequence: `group_size = 1` for every group in standard GRPO usage.

### 1.2 Mathematical Proof: group_size=1 Degrades GRPO to REINFORCE(baseline=0)

GRPO advantage formula:

```
A_i = (r_i - mu_g) / (sigma_g + epsilon)
```

Where:
- `mu_g` = mean of group rewards
- `sigma_g` = std of group rewards
- `epsilon` = numerical stability constant

With group_size = 1:
- `mu_g = r_1` (the single reward value)
- `sigma_g = 0` (std of a single value is zero)

The rLLM code handles `std=0` via epsilon-fallback: it substitutes `mean=0` and `std=1` for singleton groups, yielding:

```
A_i = (r_i - 0) / (1 + epsilon) ≈ r_i
```

This is exactly REINFORCE with baseline=0. No variance reduction, no relative comparison between trajectories, no GRPO benefit.

### 1.3 Cross-Framework Comparison

All major GRPO frameworks group by prompt/task_id, not by agent/trajectory name:

| Framework | Grouping key | Code location |
|-----------|-------------|---------------|
| **verl** | uid (prompt ID) via `index` parameter | `core_algos.py` line ~340 |
| **OpenRLHF** | prompt-level grouping | Same pattern as verl |
| **DeepSpeed** | Not built-in (no GRPO), but aligned with prompt grouping | --- |
| **Megatron** | Prompt-level grouping for GRPO | Same pattern |
| **TRL** | prompt in GRPOTrainer | Standard prompt grouping |
| **rLLM** (current, broken) | `task_id:trajectory.name` | `transform.py` line 127 |

The standard GRPO design groups all responses to the same prompt together for relative advantage computation. rLLM is the only framework that includes trajectory.name in the grouping key for GRPO, which is the root cause of #605.

---

## 2. Maintainer Feedback from #667

### 2.1 jeffreysijuntan's Design Question

The maintainer raised a valid design concern: in multi-agent workflows, trajectory.name encodes agent role (e.g., "solver" vs "judge"). Grouping solver and judge trajectories together by task_id alone would mix fundamentally different agent behaviors into the same GRPO group, breaking the multi-agent design where each role should have its own advantage baseline.

### 2.2 Why Our 2-Line Fix Was Wrong

Our PR #667 proposed:

```python
# Before:
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
# After (our V1 fix):
trajectories_by_name[task_id].append(trajectory)
```

This removed trajectory.name from the grouping key entirely. While this fixes standard single-agent GRPO, it breaks multi-agent use cases where solver and judge trajectories should be grouped separately by role.

### 2.3 Valid Objection Summary

The maintainer's objection was correct and well-founded. The problem is not that trajectory.name is wrong for grouping per se; rather, the problem is that the grouping strategy should depend on the algorithm and use case:

- **GRPO (single agent)**: Group by task_id only -- all responses to the same prompt compare against each other
- **Multi-agent (solver/judge)**: Group by task_id + role -- each agent role maintains its own advantage baseline
- **PPO**: Group by task_id + role -- agent role differentiation is important for PPO baselines

The correct fix is making the grouping strategy **configurable**, not **removing name entirely**.

---

## 3. Revised Fix Approaches

### 3.1 Approach A: Add `TransformConfig.grouping_strategy` Parameter

Add a `grouping_strategy` (or `grouping_mode`) field to `TransformConfig` with two options:

- **"by_task_id"**: Group all trajectories by task_id alone. All responses to the same prompt land in one group. This is the standard GRPO approach and provides correct variance reduction when group_size >= 2.
- **"by_task_id_and_name"**: Group by `task_id:trajectory.name`. Preserves multi-agent separation where solver and judge trajectories are grouped by role within the same task.

Default values:
- GRPO default: `"by_task_id"` -- mathematical requirement (sigma needs |G| >= 2)
- PPO/multi-agent default: `"by_task_id_and_name"` -- preserves agent role differentiation

Implementation sketch (already prototyped in `rllm-667-revised-transform.py` and `rllm-667-revised-config.py`):

```python
# TransformConfig addition
grouping_mode: str = "by_task_id"  # or "by_task_id_and_name"

# _build_trajectory_groups modification
if grouping_mode == "by_task_id":
    group_key = task_id
elif grouping_mode == "by_task_id_and_name":
    group_key = f"{task_id}:{trajectory.name}"
else:
    raise ValueError(f"Unknown grouping_mode: {grouping_mode}")
```

Advantages:
- Clean, explicit configuration
- Backward-compatible (multi-agent users set "by_task_id_and_name")
- Easy to extend with future strategies (e.g., "by_task_id_and_role")
- User has full control

Disadvantages:
- Requires user to know which strategy to pick
- Default must be chosen carefully per algorithm

### 3.2 Approach B: Auto-Fallback When group_size=1

When a trajectory group has only 1 member, automatically fall back to task_id-only grouping:

```python
# After initial grouping by task_id:name
# Check for singleton groups and merge them into task_id-level groups
for group_key, trajectories in trajectories_by_name.items():
    if len(trajectories) == 1:
        # Merge singleton into task_id-level group
        task_id = group_key.split(":")[0]
        merged_trajectories_by_task[task_id].extend(trajectories)
```

This preserves multi-agent grouping when |group| > 1 (e.g., 4 solver trajectories + 3 judge trajectories for the same prompt), while automatically fixing the single-response-per-prompt case (standard GRPO).

Advantages:
- No configuration needed -- automatic behavior
- Fixes the degenerate case without user intervention
- Preserves valid multi-agent grouping

Disadvantages:
- Implicit behavior -- user may not understand why grouping changes
- Does not handle the case where multi-agent trajectories legitimately have group_size=1 per role (e.g., 1 solver, 1 judge for the same prompt -- both are singletons and get merged, potentially losing role separation)
- Could produce unexpected grouping for edge cases
- Harder to debug when grouping behavior changes silently

### 3.3 Approach C: Per-Algorithm Default

Set the grouping strategy default based on the algorithm being used:

- **GRPO**: Always default to `"by_task_id"` -- GRPO's mathematical requirement is that sigma needs |G| >= 2 for meaningful normalization. Grouping by task_id alone ensures all responses to the same prompt are compared.
- **PPO/multi-agent**: Always default to `"by_task_id_and_name"` -- PPO with multiple agent roles needs role-based baselines. Solver and judge should each have their own advantage computation.

Implementation:

```python
# In GRPO trainer initialization
transform_config = TransformConfig(grouping_mode="by_task_id")

# In PPO/multi-agent trainer initialization
transform_config = TransformConfig(grouping_mode="by_task_id_and_name")
```

Advantages:
- No configuration needed for standard use cases
- Correct mathematical behavior for each algorithm
- Matches established patterns in other frameworks

Disadvantages:
- Requires modifying each trainer class
- Less flexible if a user wants GRPO with multi-agent grouping (though this is mathematically questionable)

---

## 4. Recommended Approach: A + C Combined

### 4.1 Recommendation

Combine Approach A (configurable parameter) with Approach C (per-algorithm defaults):

1. Add `grouping_mode` field to `TransformConfig` with values `"by_task_id"` and `"by_task_id_and_name"`
2. Set per-algorithm defaults:
   - GRPO trainer: `grouping_mode = "by_task_id"` (mathematical necessity)
   - PPO/multi-agent trainer: `grouping_mode = "by_task_id_and_name"` (preserves multi-agent)
3. Allow user override via config parameter for edge cases

### 4.2 Justification

**Mathematical necessity for GRPO**: GRPO advantage = (r_i - mu_g) / sigma_g. When sigma_g = 0 (group_size = 1), the advantage degenerates to raw reward with no normalization. This is not a preference; it is a mathematical constraint. GRPO **requires** group_size >= 2 for meaningful operation. Grouping by task_id alone (instead of task_id:name) is the only way to ensure group_size > 1 in standard single-agent GRPO.

**Design necessity for multi-agent**: In PPO/multi-agent workflows, solver and judge agents have fundamentally different reward structures and should maintain separate baselines. Mixing solver and judge trajectories into one group would produce meaningless advantages where a judge's reward is compared against a solver's baseline.

**Configurability for flexibility**: Neither "always group by task_id" nor "always group by task_id:name" is universally correct. The right answer depends on the algorithm and use case. Making it configurable with sensible per-algorithm defaults gives users the correct behavior by default while allowing override for edge cases.

### 4.3 Comparison with V1 Fix

| Aspect | V1 (remove name) | V2 (configurable, A+C) |
|--------|------------------|------------------------|
| GRPO scenario | Works | Works (default="by_task_id") |
| Multi-agent scenario | BROKEN | Works (default="by_task_id_and_name") |
| Backward compatibility | NO | YES (defaults preserve existing behavior for PPO/multi-agent) |
| Maintainer concerns addressed | NO | YES (multi-agent use case preserved) |
| Mathematical correctness | YES | YES (GRPO defaults to correct grouping) |
| Config complexity | None | Minimal (1 field, 2 values, sensible defaults) |

---

## 5. Fork Repository Implementation Plan

All changes are made on the **jackie2049/rllm fork**. NEVER push directly to upstream rllm-org/rllm.

### 5.1 Branch Setup

```bash
# Clone fork (if not already cloned)
cd ~/workspace && git clone https://github.com/Jackie2049/rllm.git
cd ~/workspace/rllm

# Create fix branch from main
git checkout main
git checkout -b fix/grpo-grouping-strategy-v2
```

### 5.2 Files to Modify

**File 1: `rllm/trainer/config/transform_config.py`** (or `rllm/trainer/algorithms/config.py`)

Add `grouping_mode` field to `TransformConfig`:

```python
@dataclass
class TransformConfig:
    """Configuration for the episode-to-group transformation pipeline."""

    # Name imputation
    impute_missing_names: bool = True
    default_traj_name: str = _DEFAULT_TRAJ_NAME
    drop_unnamed_traj: bool = False

    # Reward configuration
    broadcast: bool = True

    # Grouping strategy (NEW)
    # - "by_task_id": Group all trajectories for the same task together.
    #   Optimal for GRPO where all trajectories share a baseline.
    #   Default for GRPO algorithm (mathematical requirement: sigma needs |G| >= 2).
    # - "by_task_id_and_name": Group trajectories by task AND agent name.
    #   Preserves multi-agent separation (e.g. solver vs judge get separate baselines).
    #   Default for PPO/multi-agent algorithm (preserves agent role differentiation).
    grouping_mode: str = "by_task_id"

    @classmethod
    def from_config(cls, transform_config: DictConfig, *, broadcast: bool = True) -> "TransformConfig":
        return cls(
            impute_missing_names=transform_config.get("impute_missing_names", True),
            default_traj_name=transform_config.get("default_traj_name", _DEFAULT_TRAJ_NAME),
            drop_unnamed_traj=transform_config.get("drop_unnamed_traj", False),
            broadcast=broadcast,
            grouping_mode=transform_config.get("grouping_mode", "by_task_id"),
        )
```

**File 2: `rllm/trainer/algorithms/transform.py`**

Modify `_build_trajectory_groups` to accept and use `grouping_mode`:

```python
def _build_trajectory_groups(
    episodes: list[Episode],
    compact_filtering_config: CompactFilteringConfig | None = None,
    grouping_mode: str = "by_task_id",
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
            # Compute grouping key based on grouping_mode
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
            metadata_by_name[group_key].append(
                {
                    "task_id": episode.task_id,
                    "rollout_idx": episode.rollout_idx,
                    "termination_reason": episode.termination_reason,
                    "is_correct": episode.is_correct,
                }
            )

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
```

Modify `_default_trajectory_grouping_hook` to pass `grouping_mode`:

```python
def _default_traj_grouping_hook(episodes, transform_config, compact_filtering_config=None):
    trajectory_groups = _build_trajectory_groups(
        episodes, compact_filtering_config,
        grouping_mode=transform_config.grouping_mode,
    )
    # ... rest unchanged ...
```

**File 3: `rllm/trainer/algorithms/grpo_trainer.py`**

Set GRPO-specific default grouping strategy:

```python
# In GRPO trainer initialization
# GRPO requires group_size >= 2 for meaningful normalization
# Default to "by_task_id" grouping strategy
transform_config = TransformConfig(grouping_mode="by_task_id")
```

This ensures that even if the global default changes, GRPO always uses the mathematically correct grouping.

### 5.3 Test Plan

| Test case | grouping_mode | Expected result |
|-----------|--------------|-----------------|
| Standard GRPO (4 trajectories per prompt, single agent) | "by_task_id" | group_size = 4, correct GRPO advantage |
| Standard GRPO (8 trajectories per prompt) | "by_task_id" | group_size = 8, correct GRPO advantage |
| Multi-agent (solver + judge, 4 each) | "by_task_id_and_name" | 2 groups per task: solver (size=4), judge (size=4) |
| Multi-agent (solver + judge, 1 each) | "by_task_id_and_name" | 2 singleton groups per task -- degenerate but preserves role separation |
| Mixed trajectories | "by_task_id" | All trajectories for same prompt in one group, regardless of agent |
| Invalid grouping_mode | "invalid" | ValueError raised |

### 5.4 Documentation

Add docstring explanation for the `grouping_mode` parameter in both `TransformConfig` and `_build_trajectory_groups`, covering:
- What each strategy does
- When to use each strategy
- Mathematical requirement for GRPO (group_size >= 2)
- Multi-agent use case for "by_task_id_and_name"

### 5.5 Commit and Push Plan

```bash
# Stage changes
git add rllm/trainer/algorithms/config.py
git add rllm/trainer/algorithms/transform.py
git add rllm/trainer/algorithms/grpo_trainer.py

# Commit
git commit -m "fix: add configurable grouping_mode for GRPO grouping strategy (#605 v2)

Add grouping_mode parameter to TransformConfig with per-algorithm defaults:
- GRPO: 'by_task_id' (mathematical requirement: sigma needs |G| >= 2)
- PPO/multi-agent: 'by_task_id_and_name' (preserves agent role differentiation)

This replaces the v1 fix that removed trajectory.name from the grouping key
entirely, which broke multi-agent use cases per maintainer feedback on #667."

# Push to fork (NEVER to upstream!)
git push origin fix/grpo-grouping-strategy-v2
```

---

## 6. Cross-Framework Singleton Group Degeneration Update

### 6.1 Numerical Experiment Proof

Our numerical experiment demonstrated that group_size = 1 (gs=1) produces GRPO advantage equivalent to REINFORCE(baseline=0) across ALL four major frameworks:

| Framework | Singleton handling | Resulting advantage | Degeneration |
|-----------|-------------------|--------------------|--------------|
| **verl** | epsilon-fallback: mean=0, std=1 | A = (r - 0) / (1 + eps) ≈ r | REINFORCE(baseline=0) |
| **OpenRLHF** | Same epsilon-fallback | A ≈ r | REINFORCE(baseline=0) |
| **rLLM** | epsilon-division: std=0 + epsilon | A = (r - r) / (0 + eps) ≈ 0 | ZERO gradient signal (worse!) |
| **TRL** | epsilon-fallback: mean=0, std=1 | A ≈ r | REINFORCE(baseline=0) |

### 6.2 Two Distinct Degeneration Modes

There are two different degeneration patterns across frameworks:

**Mode 1: epsilon-fallback (verl, OpenRLHF, TRL)**
- When std=0: substitute mean=0, std=1
- Advantage = (r - 0) / (1 + epsilon) ≈ r
- Result: REINFORCE(baseline=0) -- gradient exists but no variance reduction

**Mode 2: epsilon-division (rLLM, some TRL versions)**
- When std=0: divide by epsilon directly (no substitution)
- Advantage = (r - mean) / epsilon ≈ (r - r) / epsilon ≈ 0
- Result: ZERO gradient signal -- even worse than REINFORCE, training produces no useful updates

### 6.3 This Is a Cross-Framework Design Defect

This singleton group degeneration is NOT rLLM-specific. It is a cross-framework **design defect** that affects every GRPO implementation:

1. All frameworks allow group_size=1 to occur (no enforcement of minimum group_size)
2. All frameworks handle the resulting std=0 via some epsilon-based fallback
3. All fallbacks produce mathematically degenerate advantages (REINFORCE or zero gradient)
4. None of the frameworks warn the user that their training has degraded

### 6.4 Minimum group_size Enforcement Recommendation

All GRPO frameworks should enforce `minimum group_size >= 2`:

- **verl**: Add assertion or warning when `index` produces singleton groups
- **rLLM**: Add assertion or warning in `_build_trajectory_groups` when `group_size < 2`
- **TRL**: Add assertion or warning in `GRPOTrainer` when `n_rollouts < 2`
- **OpenRLHF**: Same pattern

The enforcement could be:
- **Strict**: Raise error if any group has size < 2 (training fails -- user must fix config)
- **Soft**: Log warning if any group has size < 2 (training continues but user is notified)
- **Corrective**: Auto-merge singleton groups into task_id-level groups (Approach B from above)

For rLLM specifically, the `grouping_mode` fix (Approach A+C) addresses the root cause by ensuring GRPO defaults to task_id-only grouping, which naturally produces group_size >= 2 when n_rollouts >= 2. This is the correct solution for rLLM. The cross-framework minimum group_size enforcement is a broader concern that should be raised separately with each framework.

---

## 7. Key Takeaways

1. **Original bug**: rLLM groups by `task_id:trajectory.name` instead of `task_id`, causing group_size=1 in standard GRPO, degrading to REINFORCE(baseline=0).

2. **Maintainer feedback was valid**: trajectory.name serves a legitimate multi-agent purpose (solver/judge role separation). Our V1 fix that removed name entirely was incorrect because it broke multi-agent use cases.

3. **Correct fix**: Make grouping strategy configurable with per-algorithm defaults. GRPO defaults to "by_task_id" (mathematical requirement); PPO/multi-agent defaults to "by_task_id_and_name" (preserves role separation).

4. **Implementation**: Add `grouping_mode` field to `TransformConfig`, modify `_build_trajectory_groups` to use it, set per-algorithm defaults in each trainer.

5. **Cross-framework scope**: Singleton group degeneration affects ALL GRPO frameworks (verl/OpenRLHF/TRL/rLLM). This is a design defect, not a rLLM-specific bug. All frameworks need minimum group_size >= 2 enforcement.

6. **Fork strategy**: All changes on jackie2049/rllm fork, branch `fix/grpo-grouping-strategy-v2`, never push directly to upstream.

---

## References

- rLLM #605: https://github.com/rllm-org/rllm/issues/605
- rLLM #667 (CLOSED): https://github.com/rllm-org/rllm/pull/667
- rLLM #663 (MERGED): Step.output was None -- all rewards = 0.0
- Maintainer feedback: jeffreysijuntan comment on PR #667
- Source reading: `notebook/projects/rllm-605-grpo-grouping-bug-source-reading.md`
- Bug reading: `notebook/projects/rllm-605-grpo-grouping-bug-reading.md`
- Cross-framework comparison: `notebook/projects/cross-framework-grpo-advantage-comparison.md`
- V2 fix draft: `notebook/projects/rllm-605-grpo-grouping-revised-fix-v2.md`
- Revised transform.py prototype: `notebook/projects/rllm-667-revised-transform.py`
- Revised config.py prototype: `notebook/projects/rllm-667-revised-config.py`
- Comment draft: `notebook/projects/rllm-605-grpo-grouping-bug-comment-draft.md`
- GRPO original paper: https://arxiv.org/abs/2402.03300
- Dr.GRPO: https://arxiv.org/abs/2503.20783
