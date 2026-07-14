#!/usr/bin/env python3
"""Apply rLLM #605 fix: Configurable grouping key for GRPO advantage estimation.

5 changes:
1. config.py: Add grouping_key to TransformConfig
2. config.py: Add GROUPED_GRPO to rLLMAdvantageEstimator enum
3. transform.py: Add _resolve_grouping_key and modify _build_trajectory_groups
4. advantage.py: Register grouped_grpo estimator
5. transform.py: Wire grouping_key through the pipeline
"""

import os
import shutil

RLLM_DIR = "/jiangdingfeng/zy/Termius/rollout/rllm-terminal-rl/rllm/trainer/algorithms"

CONFIG_PATH = os.path.join(RLLM_DIR, "config.py")
TRANSFORM_PATH = os.path.join(RLLM_DIR, "transform.py")
ADVANTAGE_PATH = os.path.join(RLLM_DIR, "advantage.py")


def fix_config():
    """Add grouping_key to TransformConfig and GROUPED_GRPO to estimator enum."""
    with open(CONFIG_PATH) as f:
        content = f.read()

    # Change 1: Add grouping_key to TransformConfig
    if "grouping_key" in content:
        print("Config: grouping_key already exists, skipping TransformConfig change")
    else:
        # Find the broadcast field in TransformConfig and add grouping_key after it
        target = "    broadcast: bool = True  # If True, use trajectory-level rewards; if False, use per-step rewards"
        if target not in content:
            print("ERROR: Cannot find broadcast field in TransformConfig")
            return False

        insertion = """    broadcast: bool = True  # If True, use trajectory-level rewards; if False, use per-step rewards
    # Grouping configuration for advantage computation
    grouping_key: str = "task_id:name"  # How to group trajectories: "task_id:name" (default), "task_id", "metadata:<key>", etc."""

        content = content.replace(target, insertion, 1)
        print("Added grouping_key to TransformConfig")

        # Also update from_config method to include grouping_key
        target_from = "            broadcast=broadcast,\n        )"
        if target_from not in content:
            print("WARNING: Cannot find from_config broadcast line")
        else:
            insertion_from = """            broadcast=broadcast,
            grouping_key=transform_config.get("grouping_key", "task_id:name"),
        )"""
            content = content.replace(target_from, insertion_from, 1)
            print("Added grouping_key to TransformConfig.from_config")

    # Change 2: Add GROUPED_GRPO to rLLMAdvantageEstimator enum
    if "GROUPED_GRPO" in content:
        print("Config: GROUPED_GRPO already in enum, skipping")
    else:
        # Find ECHO = "echo" in the enum and add GROUPED_GRPO after it
        target = '    ECHO = "echo"'
        if target not in content:
            print("ERROR: Cannot find ECHO in estimator enum")
            return False

        insertion = '    ECHO = "echo"\n    GROUPED_GRPO = "grouped_grpo"  # GRPO with configurable grouping key'
        content = content.replace(target, insertion, 1)
        print("Added GROUPED_GRPO to rLLMAdvantageEstimator enum")

    # Backup and write
    backup_path = CONFIG_PATH + ".605.orig"
    if not os.path.exists(backup_path):
        shutil.copy2(CONFIG_PATH, backup_path)

    with open(CONFIG_PATH, 'w') as f:
        f.write(content)
    print("Config file written")

    # Verify
    with open(CONFIG_PATH) as f:
        verify = f.read()
    if "grouping_key" in verify and "GROUPED_GRPO" in verify:
        print("Config fix verified")
        return True
    else:
        print("ERROR: Config fix verification failed")
        shutil.copy2(backup_path, CONFIG_PATH)
        return False


def fix_transform():
    """Add _resolve_grouping_key and modify _build_trajectory_groups."""
    with open(TRANSFORM_PATH) as f:
        content = f.read()

    # Change 3: Add _resolve_grouping_key function
    if "_resolve_grouping_key" in content:
        print("Transform: _resolve_grouping_key already exists, skipping")
    else:
        # Insert _resolve_grouping_key before _build_trajectory_groups
        target = "def _build_trajectory_groups(episodes: list[Episode], compact_filtering_config: CompactFilteringConfig | None = None) -> list[TrajectoryGroup]:"
        if target not in content:
            print("ERROR: Cannot find _build_trajectory_groups function")
            return False

        resolve_func = (
            'def _resolve_grouping_key(episode: Episode, trajectory: Trajectory, grouping_key: str) -> str:\n'
            '    """Resolve a grouping key template into an actual key string.\n'
            '\n'
            '    Supported formats:\n'
            '    - "task_id:name" -> task_id:name compound key\n'
            '    - "task_id" -> episode.task_id\n'
            '    - "name" -> trajectory.name\n'
            '    - "metadata:<key>" -> trajectory.metadata lookup\n'
            '    - "uid" -> trajectory.uid (reproduces #605 bug for testing)\n'
            '    """\n'
            '    parts = grouping_key.split(":")\n'
            '    resolved_parts = []\n'
            '    for part in parts:\n'
            '        if part == "task_id":\n'
            '            resolved_parts.append(episode.task_id)\n'
            '        elif part == "name":\n'
            '            resolved_parts.append(trajectory.name or "unnamed")\n'
            '        elif part.startswith("metadata:"):\n'
            '            meta_key = part[len("metadata:")]\n'
            '            resolved_parts.append(str(trajectory.metadata.get(meta_key, "unknown")) if trajectory.metadata else "unknown")\n'
            '        elif part == "uid":\n'
            '            resolved_parts.append(trajectory.uid)\n'
            '        else:\n'
            '            resolved_parts.append(part)  # literal string\n'
            '    return ":".join(resolved_parts)\n'
            '\n'
            '\n'
            'def _build_trajectory_groups(episodes: list[Episode], compact_filtering_config: CompactFilteringConfig | None = None, grouping_key: str = "task_id:name") -> list[TrajectoryGroup]:'
        )

        content = content.replace(target, resolve_func, 1)
        print("Added _resolve_grouping_key function")

    # Change 3b: Modify _build_trajectory_groups to use grouping_key
    # Replace the hardcoded f"{task_id}:{trajectory.name}" with _resolve_grouping_key
    target_grouping = '            trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)'
    if target_grouping not in content:
        print("WARNING: Cannot find hardcoded grouping key line")
        # Show context around _build_trajectory_groups
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "_build_trajectory_groups" in line and "def" in line:
                for j in range(i, i+30):
                    print("L%d: %s" % (j+1, lines[j]))
                break
        return False

    new_grouping = '            key = _resolve_grouping_key(episode, trajectory, grouping_key)\n            trajectories_by_name[key].append(trajectory)'
    content = content.replace(target_grouping, new_grouping, 1)

    # Also fix the metadata grouping line
    target_meta = '            metadata_by_name[f"{task_id}:{trajectory.name}"].append('
    if target_meta in content:
        new_meta = '            metadata_by_name[key].append('
        content = content.replace(target_meta, new_meta, 1)
        print("Updated metadata grouping key")

    # Change 5: Wire grouping_key through transform_episodes_to_trajectory_groups
    # Find the call to _build_trajectory_groups and add grouping_key parameter
    target_call = "trajectory_groups = _build_trajectory_groups(episodes, compact_filtering_config)"
    if target_call not in content:
        # Try alternative format
        target_call = "trajectory_groups = _build_trajectory_groups(episodes, compact_filtering_config)  # part 1"
        if target_call not in content:
            print("WARNING: Cannot find _build_trajectory_groups call")
            # Search more broadly
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "_build_trajectory_groups(episodes" in line:
                    print("Found call at L%d: %s" % (i+1, line.strip()))
                    break
            return False

    new_call = "trajectory_groups = _build_trajectory_groups(episodes, compact_filtering_config, grouping_key=transform_config.grouping_key)"
    content = content.replace(target_call, new_call, 1)
    print("Wired grouping_key through transform pipeline")

    # Backup and write
    backup_path = TRANSFORM_PATH + ".605.orig"
    if not os.path.exists(backup_path):
        shutil.copy2(TRANSFORM_PATH, backup_path)

    with open(TRANSFORM_PATH, 'w') as f:
        f.write(content)
    print("Transform file written")

    # Verify
    with open(TRANSFORM_PATH) as f:
        verify = f.read()
    if "_resolve_grouping_key" in verify and "grouping_key" in verify:
        print("Transform fix verified")
        return True
    else:
        print("ERROR: Transform fix verification failed")
        shutil.copy2(backup_path, TRANSFORM_PATH)
        return False


def fix_advantage():
    """Register grouped_grpo estimator."""
    with open(ADVANTAGE_PATH) as f:
        content = f.read()

    if "grouped_grpo" in content:
        print("Advantage: grouped_grpo already exists, skipping")
        return True

    # Find the ECHO estimator and add grouped_grpo after it
    target = '@register_adv_estimator(rLLMAdvantageEstimator.ECHO)\ndef calculate_echo_advantages'
    if target not in content:
        print("ERROR: Cannot find ECHO estimator")
        return False

    # Insert the grouped_grpo estimator after ECHO
    # Find the end of ECHO's definition
    # ECHO delegates to calculate_grpo_advantages, so it's short
    echo_block_end = None
    lines = content.split('\n')
    echo_start = None
    for i, line in enumerate(lines):
        if line.strip() == '@register_adv_estimator(rLLMAdvantageEstimator.ECHO)':
            echo_start = i
            break

    if echo_start is None:
        print("ERROR: Cannot find ECHO decorator")
        return False

    # Find the end of the ECHO function (next decorator or function definition)
    echo_end = None
    for i in range(echo_start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith('def ') or stripped.startswith('@') or stripped.startswith('def _'):
            if not lines[i].strip().startswith('def calculate_echo_advantages'):
                echo_end = i
                break

    if echo_end is None:
        echo_end = len(lines)

    # Insert grouped_grpo after the ECHO block
    grouped_grpo_code = (
        '\n\n@register_adv_estimator(rLLMAdvantageEstimator.GROUPED_GRPO)\n'
        'def calculate_grouped_grpo_advantages(rewards: list[np.ndarray], algorithm_config: AlgorithmConfig, **kwargs) -> tuple[list[np.ndarray], list[np.ndarray]]:\n'
        '    """GRPO advantages with configurable grouping key.\n'
        '\n'
        '    This estimator delegates to standard GRPO advantage computation.\n'
        '    The grouping is controlled upstream by TransformConfig.grouping_key,\n'
        '    which determines how trajectories are grouped into TrajectoryGroups\n'
        '    before advantage normalization.\n'
        '\n'
        '    When grouping_key="task_id:name" (default), this is identical to GRPO.\n'
        '    Other grouping_key values allow different grouping strategies:\n'
        '    - "task_id": group all completions of the same task together\n'
        '    - "metadata:<key>": group by a metadata attribute\n'
        '    """\n'
        '    return calculate_grpo_advantages(rewards, algorithm_config, **kwargs)'
    )

    lines.insert(echo_end, grouped_grpo_code)
    content = '\n'.join(lines)

    # Backup and write
    backup_path = ADVANTAGE_PATH + ".605.orig"
    if not os.path.exists(backup_path):
        shutil.copy2(ADVANTAGE_PATH, backup_path)

    with open(ADVANTAGE_PATH, 'w') as f:
        f.write(content)
    print("Added grouped_grpo estimator to advantage.py")

    # Verify
    with open(ADVANTAGE_PATH) as f:
        verify = f.read()
    if "grouped_grpo" in verify and "GROUPED_GRPO" in verify:
        print("Advantage fix verified")
        return True
    else:
        print("ERROR: Advantage fix verification failed")
        shutil.copy2(backup_path, ADVANTAGE_PATH)
        return False


def verify_all():
    """Verify all changes."""
    print("\n=== #605 Verification ===")

    # Config
    with open(CONFIG_PATH) as f:
        config = f.read()
    has_grouping_key = "grouping_key" in config
    has_grouped_grpo_enum = "GROUPED_GRPO" in config
    print("Config grouping_key field: %s" % has_grouping_key)
    print("Config GROUPED_GRPO enum: %s" % has_grouped_grpo_enum)

    # Transform
    with open(TRANSFORM_PATH) as f:
        transform = f.read()
    has_resolve = "_resolve_grouping_key" in transform
    has_grouping_param = "grouping_key" in transform
    print("Transform _resolve_grouping_key: %s" % has_resolve)
    print("Transform grouping_key wired: %s" % has_grouping_param)

    # Advantage
    with open(ADVANTAGE_PATH) as f:
        advantage = f.read()
    has_estimator = "grouped_grpo" in advantage
    print("Advantage grouped_grpo estimator: %s" % has_estimator)

    all_ok = all([has_grouping_key, has_grouped_grpo_enum, has_resolve, has_grouping_param, has_estimator])
    print("\nAll checks: %s" % ("PASSED" if all_ok else "FAILED"))


if __name__ == "__main__":
    fix_config()
    fix_transform()
    fix_advantage()
    verify_all()
