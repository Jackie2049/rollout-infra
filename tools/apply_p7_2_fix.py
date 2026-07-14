#!/usr/bin/env python3
"""Apply TRL P7-2 fix: Add top_n_sigma advantage clipping to GRPOTrainer.

Three changes needed:
1. grpo_config.py: Add top_n_sigma field after scale_rewards
2. grpo_trainer.py: Store self.top_n_sigma after self.scale_rewards
3. grpo_trainer.py: Add clipping logic after nan_to_num(advantages)
"""

import os
import sys
import shutil

GRPO_CONFIG_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_config.py"
GRPO_TRAINER_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py"


def fix_config(dry_run=False):
    """Add top_n_sigma field to GRPOConfig after scale_rewards."""
    with open(GRPO_CONFIG_PATH) as f:
        content = f.read()

    if "top_n_sigma" in content:
        print("Config: top_n_sigma field already exists, skipping")
        return False

    # Find the end of scale_rewards field definition and insert after it
    # The scale_rewards field ends with:    ),
    # )    loss_type starts right after

    target_end = "    )\n    loss_type: str = field("
    if target_end not in content:
        print("ERROR: Cannot find insertion point in config (scale_rewards end)")
        return False

    # Insert top_n_sigma field between scale_rewards and loss_type
    new_field = """    )
    top_n_sigma: float = field(
        default=0.0,
        metadata={
            "help": (
                "Clip advantages to [mean - n*std, mean + n*std] after computation. "
                "Set to 0 to disable (default). Recommended: 3.0 for noisy rewards."
            )
        },
    )
    loss_type: str = field("""

    fixed_content = content.replace(target_end, new_field)

    if not dry_run:
        backup_path = GRPO_CONFIG_PATH + ".p72.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_CONFIG_PATH, backup_path)
            print("Config backup saved to: %s" % backup_path)

        with open(GRPO_CONFIG_PATH, 'w') as f:
            f.write(fixed_content)
        print("Config fixed: top_n_sigma field added")

        # Verify
        with open(GRPO_CONFIG_PATH) as f:
            verify = f.read()
        if "top_n_sigma" in verify:
            print("Config fix verified")
        else:
            print("ERROR: Config fix verification failed")
            shutil.copy2(backup_path, GRPO_CONFIG_PATH)
            return False

    return True


def fix_trainer_init(dry_run=False):
    """Add self.top_n_sigma = args.top_n_sigma after self.scale_rewards."""
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    if "self.top_n_sigma" in content:
        print("Trainer init: self.top_n_sigma already exists, skipping")
        return False

    target = "        self.scale_rewards = args.scale_rewards"
    if target not in content:
        print("ERROR: Cannot find self.scale_rewards in trainer")
        return False

    insertion = "        self.scale_rewards = args.scale_rewards\n        self.top_n_sigma = args.top_n_sigma"
    fixed_content = content.replace(target, insertion, 1)

    if not dry_run:
        backup_path = GRPO_TRAINER_PATH + ".p72.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)
            print("Trainer backup saved to: %s" % backup_path)

        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(fixed_content)
        print("Trainer init fixed: self.top_n_sigma added")

    return True


def fix_trainer_clipping(dry_run=False):
    """Add top_n_sigma clipping logic after nan_to_num(advantages)."""
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    if "Top-n-sigma clipping" in content:
        print("Trainer clipping: already exists, skipping")
        return False

    # Find the nan_to_num line and the subsequent comment
    # The pattern is:
    #   advantages = torch.nan_to_num(advantages, nan=0.0)
    #   [possibly a comment about slicing]
    #   # Slice to keep only the local part of the data

    target_block = """        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Slice to keep only the local part of the data"""

    if target_block not in content:
        # Try variations with comments on same line
        # Check exact content around nan_to_num
        lines = content.split('\n')
        nan_line_idx = None
        for i, line in enumerate(lines):
            if "advantages = torch.nan_to_num(advantages, nan=0.0)" in line and "Slice" not in line:
                nan_line_idx = i
                break

        if nan_line_idx is None:
            print("ERROR: Cannot find nan_to_num advantages line")
            return False

        # Build the target block from the actual lines
        # Show context for debugging
        for j in range(nan_line_idx, nan_line_idx + 5):
            print("  L%d: %s" % (j+1, lines[j].rstrip()))

        # Find the "# Slice" comment
        slice_idx = None
        for j in range(nan_line_idx + 1, nan_line_idx + 5):
            if "# Slice" in lines[j] or "Slice to keep" in lines[j]:
                slice_idx = j
                break

        if slice_idx is None:
            print("ERROR: Cannot find Slice comment after nan_to_num")
            return False

        # Build exact target
        target_lines = lines[nan_line_idx:slice_idx + 1]
        target_block = '\n'.join(target_lines)

    clipping_block = """        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Top-n-sigma clipping: bound extreme advantage values to stabilize training
        if self.top_n_sigma > 0:
            mean_adv = advantages.mean()
            std_adv = advantages.std()
            upper = mean_adv + self.top_n_sigma * std_adv
            lower = mean_adv - self.top_n_sigma * std_adv
            advantages = advantages.clamp(lower, upper)
            self._metrics[mode]["advantages/clip_upper"].append(upper.item())
            self._metrics[mode]["advantages/clip_lower"].append(lower.item())
            self._metrics[mode]["advantages/clipped_fraction"].append(
                ((advantages == upper) | (advantages == lower)).float().mean().item()
            )

        # Slice to keep only the local part of the data"""

    fixed_content = content.replace(target_block, clipping_block, 1)

    if not dry_run:
        # Re-use backup from init fix if it exists
        backup_path = GRPO_TRAINER_PATH + ".p72.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)
            print("Trainer clipping backup saved")

        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(fixed_content)
        print("Trainer clipping fixed: top_n_sigma logic added")

        # Verify
        with open(GRPO_TRAINER_PATH) as f:
            verify = f.read()
        if "Top-n-sigma clipping" in verify:
            print("Trainer clipping fix verified")
        else:
            print("ERROR: Clipping fix verification failed")
            shutil.copy2(backup_path, GRPO_TRAINER_PATH)
            return False

    return True


def verify_all():
    """Verify all fixes are present."""
    print("\n=== Verification ===")

    # Config
    with open(GRPO_CONFIG_PATH) as f:
        config = f.read()
    has_field = "top_n_sigma" in config
    print("Config top_n_sigma field: %s" % has_field)

    # Trainer
    with open(GRPO_TRAINER_PATH) as f:
        trainer = f.read()
    has_init = "self.top_n_sigma = args.top_n_sigma" in trainer
    has_clipping = "Top-n-sigma clipping" in trainer
    print("Trainer self.top_n_sigma init: %s" % has_init)
    print("Trainer clipping logic: %s" % has_clipping)

    if has_field and has_init and has_clipping:
        print("\nAll P7-2 fixes verified successfully")
    else:
        print("\nSome P7-2 fixes missing")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    fix_config(dry_run=dry_run)
    fix_trainer_init(dry_run=dry_run)
    fix_trainer_clipping(dry_run=dry_run)
    verify_all()
