#!/usr/bin/env python3
"""Quick fix: Insert bypass_mode field in GRPOConfig after top_n_sigma."""
import os
import shutil

GRPO_CONFIG_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_config.py"

with open(GRPO_CONFIG_PATH) as f:
    content = f.read()

if "bypass_mode" in content:
    print("bypass_mode already in config, skipping")
else:
    # Find the target: closing paren of top_n_sigma field followed by loss_type
    target = "    )\n    loss_type: str = field("
    if target not in content:
        print("ERROR: Cannot find insertion point")
        import sys
        sys.exit(1)

    insertion = """    )
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

    fixed = content.replace(target, insertion, 1)

    # Backup
    backup_path = GRPO_CONFIG_PATH + ".p91b.orig"
    if not os.path.exists(backup_path):
        shutil.copy2(GRPO_CONFIG_PATH, backup_path)

    with open(GRPO_CONFIG_PATH, 'w') as f:
        f.write(fixed)
    print("bypass_mode field inserted")

    # Verify
    with open(GRPO_CONFIG_PATH) as f:
        verify = f.read()
    if "bypass_mode" in verify:
        print("Config fix verified")
    else:
        print("ERROR: Fix verification failed")
        shutil.copy2(backup_path, GRPO_CONFIG_PATH)
