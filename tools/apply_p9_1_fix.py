#!/usr/bin/env python3
"""Apply TRL P9-1 fix: Add explicit bypass_mode flag for GRPOTrainer.

Three changes:
1. grpo_config.py: Add bypass_mode field
2. grpo_trainer.py __init__: Add self.bypass_mode = args.bypass_mode
3. grpo_trainer.py _generate_and_score_completions: Change heuristic to use bypass_mode
"""

import os
import sys
import shutil

GRPO_CONFIG_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_config.py"
GRPO_TRAINER_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py"


def fix_config(dry_run=False):
    """Add bypass_mode field to GRPOConfig."""
    with open(GRPO_CONFIG_PATH) as f:
        content = f.read()

    if "bypass_mode" in content:
        print("Config: bypass_mode field already exists, skipping")
        return False

    # Insert bypass_mode after top_n_sigma (added by P7-2)
    # Or if top_n_sigma wasn't added, insert after scale_rewards
    if "top_n_sigma" in content:
        # Find top_n_sigma field end
        target = "    ),\n    loss_type: str = field("
        if target in content:
            new_field = """    ),
    bypass_mode: bool = field(
        default=False,
        metadata={
            "help": "If True, skip computing old_per_token_logps and use per_token_logps.detach() "
            "instead. Saves one forward pass per training step but disables importance sampling "
            "correction. When num_iterations=1 and steps_per_generation=1, this is always safe. "
            "Default is False (current heuristic behavior)."
        },
    )
    loss_type: str = field("""
            fixed = content.replace(target, new_field)
        else:
            print("ERROR: Cannot find insertion point after top_n_sigma")
            return False
    else:
        # Insert after scale_rewards end
        target = "    )\n    loss_type: str = field("
        if target in content:
            new_field = """    )
    bypass_mode: bool = field(
        default=False,
        metadata={
            "help": "If True, skip computing old_per_token_logps and use per_token_logps.detach() "
            "instead. Saves one forward pass per training step but disables importance sampling "
            "correction. When num_iterations=1 and steps_per_generation=1, this is always safe. "
            "Default is False (current heuristic behavior)."
        },
    )
    loss_type: str = field("""
            fixed = content.replace(target, new_field)
        else:
            print("ERROR: Cannot find insertion point after scale_rewards")
            return False

    if not dry_run:
        backup_path = GRPO_CONFIG_PATH + ".p91.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_CONFIG_PATH, backup_path)
            print("Config backup saved to: %s" % backup_path)

        with open(GRPO_CONFIG_PATH, 'w') as f:
            f.write(fixed)
        print("Config fixed: bypass_mode field added")

        # Verify
        with open(GRPO_CONFIG_PATH) as f:
            if "bypass_mode" not in f.read():
                print("ERROR: Config fix verification failed")
                shutil.copy2(backup_path, GRPO_CONFIG_PATH)
                return False
        print("Config fix verified")

    return True


def fix_trainer_init(dry_run=False):
    """Add self.bypass_mode = args.bypass_mode in __init__."""
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    if "self.bypass_mode" in content:
        print("Trainer init: self.bypass_mode already exists, skipping")
        return False

    # Insert after self.multi_objective_aggregation = args.multi_objective_aggregation
    target = "        self.multi_objective_aggregation = args.multi_objective_aggregation"
    if target not in content:
        print("ERROR: Cannot find self.multi_objective_aggregation")
        return False

    insertion = target + "\n        self.bypass_mode = args.bypass_mode"
    fixed = content.replace(target, insertion, 1)

    if not dry_run:
        backup_path = GRPO_TRAINER_PATH + ".p91.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)
            print("Trainer init backup saved to: %s" % backup_path)

        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(fixed)
        print("Trainer init fixed: self.bypass_mode added")

    return True


def fix_trainer_bypass(dry_run=False):
    """Change the old_per_token_logps heuristic to use bypass_mode."""
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    if "not self.bypass_mode and" in content:
        print("Trainer bypass: already modified, skipping")
        return False

    # Change: if self.args.gradient_accumulation_steps % generate_every != 0 or (
    # To:      if not self.bypass_mode and (self.args.gradient_accumulation_steps % generate_every != 0 or (

    old_if = "            if self.args.gradient_accumulation_steps % generate_every != 0 or (\n                self.use_vllm and self.vllm_importance_sampling_correction\n            ):"
    new_if = "            if not self.bypass_mode and (\n                self.args.gradient_accumulation_steps % generate_every != 0\n                or (self.use_vllm and self.vllm_importance_sampling_correction)\n            ):"

    if old_if not in content:
        # Try to find the exact lines by searching for the pattern
        lines = content.split('\n')
        target_start = None
        for i, line in enumerate(lines):
            if "gradient_accumulation_steps % generate_every" in line and "if" in line:
                target_start = i
                print("Found bypass heuristic at L%d: %s" % (i+1, line.strip()))
                # Show next 2 lines
                print("  L%d: %s" % (i+2, lines[i+1].strip()))
                print("  L%d: %s" % (i+2, lines[i+2].strip() if i+2 < len(lines) else ""))
                break

        if target_start is None:
            print("ERROR: Cannot find bypass heuristic")
            return False

        # Build old_if from actual lines
        # The pattern might be slightly different due to formatting
        if_line = lines[target_start]
        next_line = lines[target_start + 1]
        close_line = lines[target_start + 2]

        old_if = if_line + "\n" + next_line + "\n" + close_line
        print("Actual pattern:\n%s" % old_if)

        new_if = (
            "            if not self.bypass_mode and (\n"
            "                self.args.gradient_accumulation_steps % generate_every != 0\n"
            "                or (self.use_vllm and self.vllm_importance_sampling_correction)\n"
            "            ):"
        )

    fixed = content.replace(old_if, new_if, 1)

    if not dry_run:
        backup_path = GRPO_TRAINER_PATH + ".p91.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)

        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(fixed)
        print("Trainer bypass fixed: heuristic changed to use bypass_mode")

        # Verify
        with open(GRPO_TRAINER_PATH) as f:
            if "not self.bypass_mode and" not in f.read():
                print("ERROR: Bypass fix verification failed")
                shutil.copy2(backup_path, GRPO_TRAINER_PATH)
                return False
        print("Trainer bypass fix verified")

    return True


def verify_all():
    """Verify all P9-1 fixes."""
    print("\n=== P9-1 Verification ===")

    with open(GRPO_CONFIG_PATH) as f:
        config = f.read()
    print("Config bypass_mode field: %s" % ("bypass_mode" in config))

    with open(GRPO_TRAINER_PATH) as f:
        trainer = f.read()
    print("Trainer self.bypass_mode init: %s" % ("self.bypass_mode = args.bypass_mode" in trainer))
    print("Trainer bypass heuristic: %s" % ("not self.bypass_mode and" in trainer))

    # Show the modified bypass section
    lines = trainer.split('\n')
    for i, line in enumerate(lines):
        if "not self.bypass_mode and" in line:
            for j in range(i-1, i+5):
                print("  L%d: %s" % (j+1, lines[j].strip()))
            break


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    fix_config(dry_run=dry_run)
    fix_trainer_init(dry_run=dry_run)
    fix_trainer_bypass(dry_run=dry_run)
    verify_all()
