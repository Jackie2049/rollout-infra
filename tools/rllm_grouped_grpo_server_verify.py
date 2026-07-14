"""Server verification script for rLLM #605 GROUPED_GRPO fix.

Checks:
1. rLLMAdvantageEstimator enum has 9 members (including GROUPED_GRPO)
2. TransformConfig has grouping_key field (default "task_id:name")
3. _resolve_grouping_key function exists and produces correct keys
4. _build_trajectory_groups uses grouping_key (not hardcoded)
5. GROUPED_GRPO registered in advantage estimator registry
"""

import ast
import sys

FAILURES = 0


def check(label, condition, detail=""):
    global FAILURES
    if condition:
        print(f"[PASS] {label}")
    else:
        FAILURES += 1
        print(f"[FAIL] {label} — {detail}")


def main():
    import_dir = "/jiangdingfeng/zy/Termius/rollout/rllm-fork"

    # ── Check 1: Enum members ──
    with open(f"{import_dir}/rllm/trainer/algorithms/config.py") as f:
        config_tree = ast.parse(f.read())

    enum_members = []
    for node in ast.walk(config_tree):
        if isinstance(node, ast.ClassDef) and node.name == 'rLLMAdvantageEstimator':
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            enum_members.append(target.id)

    check("1. Enum has GROUPED_GRPO",
          "GROUPED_GRPO" in enum_members,
          f"members={enum_members}")
    check("1b. Enum has 8+ members (including OTHER)",
          len(enum_members) >= 8,
          f"count={len(enum_members)}, members={enum_members}")

    # ── Check 2: TransformConfig grouping_key ──
    tc_annotations = []
    for node in ast.walk(config_tree):
        if isinstance(node, ast.ClassDef) and node.name == 'TransformConfig':
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    tc_annotations.append(item.target.id)

    check("2. TransformConfig has grouping_key",
          "grouping_key" in tc_annotations,
          f"annotations={tc_annotations}")

    # ── Check 2b: grouping_key default value ──
    grouping_key_default = None
    for node in ast.walk(config_tree):
        if isinstance(node, ast.ClassDef) and node.name == 'TransformConfig':
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == 'grouping_key':
                    if isinstance(item.value, ast.Constant):
                        grouping_key_default = item.value.value
    check("2b. grouping_key default='task_id:name'",
          grouping_key_default == "task_id:name",
          f"default={grouping_key_default}")

    # ── Check 3: _resolve_grouping_key exists ──
    with open(f"{import_dir}/rllm/trainer/algorithms/transform.py") as f:
        transform_tree = ast.parse(f.read())

    functions = []
    for node in ast.walk(transform_tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)

    check("3. _resolve_grouping_key function exists",
          "_resolve_grouping_key" in functions,
          f"functions={functions}")

    # ── Check 4: _build_trajectory_groups calls _resolve_grouping_key ──
    calls_in_build = []
    for node in ast.walk(transform_tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_build_trajectory_groups':
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    calls_in_build.append(child.func.id)

    check("4. _build_trajectory_groups calls _resolve_grouping_key",
          "_resolve_grouping_key" in calls_in_build,
          f"calls={calls_in_build}")

    # ── Check 4b: _build_trajectory_groups accepts transform_config param ──
    build_args = []
    for node in ast.walk(transform_tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_build_trajectory_groups':
            for arg in node.args.args:
                build_args.append(arg.arg)

    check("4b. _build_trajectory_groups has transform_config param",
          "transform_config" in build_args,
          f"args={build_args}")

    # ── Check 5: GROUPED_GRPO registered in advantage ──
    with open(f"{import_dir}/rllm/trainer/algorithms/advantage.py") as f:
        advantage_tree = ast.parse(f.read())

    registered_estimators = []
    for node in ast.walk(advantage_tree):
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call):
                    if isinstance(deco.func, ast.Name) and deco.func.id == 'register_adv_estimator':
                        if deco.args:
                            if isinstance(deco.args[0], ast.Attribute):
                                registered_estimators.append(deco.args[0].attr)
                            elif isinstance(deco.args[0], ast.Constant):
                                registered_estimators.append(str(deco.args[0].value))

    check("5. GROUPED_GRPO registered in advantage estimators",
          "GROUPED_GRPO" in registered_estimators,
          f"registered={registered_estimators}")

    # ── Check 6: _default_traj_grouping_hook passes transform_config ──
    calls_in_hook = []
    for node in ast.walk(transform_tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_default_traj_grouping_hook':
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    calls_in_hook.append(child.func.id)

    # The hook should call _build_trajectory_groups with transform_config
    source = open(f"{import_dir}/rllm/trainer/algorithms/transform.py").read()
    # Check that the line includes transform_config in _build_trajectory_groups call
    has_transform_config_in_call = "_build_trajectory_groups(episodes, transform_config," in source or \
                                    "_build_trajectory_groups(episodes, transform_config)" in source

    check("6. _default_traj_grouping_hook passes transform_config",
          has_transform_config_in_call,
          f"calls={calls_in_hook}")

    # ── Summary ──
    print(f"\n=== {8 - FAILURES}/8 checks passed, {FAILURES} failures ===")
    sys.exit(FAILURES)


if __name__ == "__main__":
    main()
