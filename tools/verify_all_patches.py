#!/usr/bin/env python3
"""Comprehensive verification of all TRL patches on the A100 server."""

import os

TRLEX = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl"
TEMPLATES = os.path.join(TRLEX, "chat_templates")
TRAINER_DIR = os.path.join(TRLEX, "trainer")

def verify_6361():
    """#6361: Qwen3.5 chat template elif branch."""
    print("\n=== TRL #6361 ===")
    for name in ["qwen3_5_think_training.jinja", "qwen3_5_nothink_training.jinja"]:
        path = os.path.join(TEMPLATES, name)
        with open(path) as f:
            content = f.read()
        lines = content.split('\n')
        has_if = False
        has_elif = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("{%- if") and "'<think>'" in stripped and "'</think>'" in stripped and "in content" in stripped:
                has_if = True
            if stripped.startswith("{%- elif") and "'</think>'" in stripped and "in content" in stripped:
                has_elif = True
        status = "VERIFIED" if (has_if and has_elif) else "MISSING"
        print("  %s: if=%s, elif=%s -> %s" % (name, has_if, has_elif, status))

def verify_p81():
    """P8-1: num_generations=1 warning."""
    print("\n=== TRL P8-1 ===")
    path = os.path.join(TRAINER_DIR, "grpo_trainer.py")
    with open(path) as f:
        content = f.read()
    has_warning = "num_generations=1 degenerates GRPO" in content
    print("  Warning present: %s -> %s" % (has_warning, "VERIFIED" if has_warning else "MISSING"))

    # Show the warning code
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "num_generations == 1" in line:
            for j in range(i, i+6):
                print("    L%d: %s" % (j+1, lines[j].strip()))
            break

def verify_p72():
    """P7-2: top_n_sigma advantage clipping."""
    print("\n=== TRL P7-2 ===")

    # Config
    config_path = os.path.join(TRAINER_DIR, "grpo_config.py")
    with open(config_path) as f:
        config = f.read()
    has_field = "top_n_sigma" in config
    print("  Config top_n_sigma field: %s -> %s" % (has_field, "VERIFIED" if has_field else "MISSING"))

    # Trainer
    trainer_path = os.path.join(TRAINER_DIR, "grpo_trainer.py")
    with open(trainer_path) as f:
        trainer = f.read()
    has_init = "self.top_n_sigma = args.top_n_sigma" in trainer
    has_clipping = "Top-n-sigma clipping" in trainer
    print("  Trainer init: %s -> %s" % (has_init, "VERIFIED" if has_init else "MISSING"))
    print("  Trainer clipping: %s -> %s" % (has_clipping, "VERIFIED" if has_clipping else "MISSING"))

def verify_p91():
    """P9-1: bypass_mode flag."""
    print("\n=== TRL P9-1 ===")

    # Config
    config_path = os.path.join(TRAINER_DIR, "grpo_config.py")
    with open(config_path) as f:
        config = f.read()
    has_field = "bypass_mode" in config
    print("  Config bypass_mode field: %s -> %s" % (has_field, "VERIFIED" if has_field else "MISSING"))

    # Trainer
    trainer_path = os.path.join(TRAINER_DIR, "grpo_trainer.py")
    with open(trainer_path) as f:
        trainer = f.read()
    has_init = "self.bypass_mode = args.bypass_mode" in trainer
    has_logic = "not self.bypass_mode and" in trainer
    print("  Trainer init: %s -> %s" % (has_init, "VERIFIED" if has_init else "MISSING"))
    print("  Trainer bypass logic: %s -> %s" % (has_logic, "VERIFIED" if has_logic else "MISSING"))

    # Show bypass logic
    lines = trainer.split('\n')
    for i, line in enumerate(lines):
        if "not self.bypass_mode and" in line:
            for j in range(i-1, i+5):
                print("    L%d: %s" % (j+1, lines[j].strip()))
            break

def check_imports():
    """Quick import check to verify TRL is still loadable."""
    print("\n=== Import Check ===")
    try:
        import sys
        sys.path.insert(0, "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages")
        from trl import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
        print("  GRPOTrainer import: OK")
        print("  GRPOConfig import: OK")

        # Check GRPOConfig has new fields
        config = GRPOConfig()
        print("  GRPOConfig.top_n_sigma: %s" % config.top_n_sigma)
        print("  GRPOConfig.bypass_mode: %s" % config.bypass_mode)
    except Exception as e:
        print("  Import error: %s" % e)


if __name__ == "__main__":
    verify_6361()
    verify_p81()
    verify_p72()
    verify_p91()
    check_imports()
    print("\n=== Comprehensive verification complete ===")
