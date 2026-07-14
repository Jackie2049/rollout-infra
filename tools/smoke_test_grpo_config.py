#!/usr/bin/env python3
"""Quick smoke test: Verify GRPOConfig with all new fields loads correctly."""

import sys
sys.path.insert(0, "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages")

from trl import GRPOConfig

cfg = GRPOConfig()
print("top_n_sigma:", cfg.top_n_sigma)
print("bypass_mode:", cfg.bypass_mode)
print("num_generations:", cfg.num_generations)
print("scale_rewards:", cfg.scale_rewards)
print("loss_type:", cfg.loss_type)

# Verify defaults are backward compatible
assert cfg.top_n_sigma == 0.0, "top_n_sigma default should be 0.0"
assert cfg.bypass_mode == False, "bypass_mode default should be False"
print("\nAll backward compatibility checks PASSED")

# Test with non-default values
cfg2 = GRPOConfig(top_n_sigma=3.0, bypass_mode=True)
print("\ncfg2.top_n_sigma:", cfg2.top_n_sigma)
print("cfg2.bypass_mode:", cfg2.bypass_mode)
print("Custom config PASSED")

# Test GRPOTrainer import (without instantiating - needs model)
from trl import GRPOTrainer
print("\nGRPOTrainer import OK")

print("\n=== Smoke test complete ===")
