#!/usr/bin/env python3
"""Apply TRL P8-1 fix: Add warning when num_generations=1 in GRPOTrainer.

Bug: When num_generations=1, GRPO degenerates to REINFORCE with zero advantage
(group mean = reward, group std = 0). Training does nothing.

Fix: Add a logger.warning() after self.num_generations assignment.
"""

import os
import sys
import shutil

GRPO_TRAINER_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py"

# The warning code to insert after self.num_generations = args.num_generations
WARNING_CODE = """\
        if self.num_generations == 1:
            logger.warning(
                "num_generations=1 degenerates GRPO to REINFORCE with zero advantage "
                "(group mean = reward, group std = 0). Set num_generations >= 2 "
                "for proper GRPO group normalization."
            )
"""

def apply_fix(dry_run=False):
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    # Check if fix already applied
    if "num_generations=1 degenerates GRPO" in content:
        print("Fix already applied, skipping")
        return False

    # Find the insertion point
    target = "        self.num_generations = args.num_generations  # = G in the GRPO paper"
    if target not in content:
        print("ERROR: Cannot find insertion point")
        return False

    lines = content.split('\n')
    fixed_lines = []
    inserted = False

    for line in lines:
        fixed_lines.append(line)
        if line == target and not inserted:
            # Insert the warning after this line
            for warn_line in WARNING_CODE.split('\n'):
                if warn_line:  # skip empty trailing line
                    fixed_lines.append(warn_line)
            inserted = True
            print("Inserted warning code after line: '%s'" % line.strip())

    if not inserted:
        print("ERROR: Insertion point not found")
        return False

    fixed_content = '\n'.join(fixed_lines)

    if not dry_run:
        # Backup
        backup_path = GRPO_TRAINER_PATH + ".p81.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)
            print("Backup saved to: %s" % backup_path)

        # Write fixed content
        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(fixed_content)
        print("Fixed file written: %d chars" % len(fixed_content))

        # Verify
        with open(GRPO_TRAINER_PATH) as f:
            verify = f.read()
        if "num_generations=1 degenerates GRPO" in verify:
            print("Fix verified successfully")
        else:
            print("ERROR: Fix verification failed")
            shutil.copy2(backup_path, GRPO_TRAINER_PATH)
            print("Original restored from backup")
            return False

    return True


def test_warning():
    """Test that the warning fires when num_generations=1."""
    import logging
    import sys

    # Set up logging to capture warnings
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    # Import the fixed GRPOTrainer
    sys.path.insert(0, "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages")

    print("\n=== Warning Test ===")
    print("This test requires creating a GRPOTrainer instance.")
    print("Since that requires a model and dataset, we verify structurally:")

    # Read the fixed file and show the warning code
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "num_generations == 1" in line:
            print("L%d: %s" % (i+1, line.strip()))
            # Show next 5 lines
            for j in range(i+1, min(i+6, len(lines))):
                print("L%d: %s" % (j+1, lines[j].strip()))
            print("Warning code present: VERIFIED")
            break


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    apply_fix(dry_run=dry_run)
    test_warning()
