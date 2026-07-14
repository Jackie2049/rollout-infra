"""
TRL P7-2: Top-n-sigma reward clipping

## Rationale

GRPO advantage values can have extreme outliers, especially with:
1. Small group sizes (num_generations=2-4) where group statistics are noisy
2. Noisy reward functions (e.g., LLM-as-judge with high variance)
3. Early training when policy is still random

These outliers destabilize training by producing extreme policy gradients.
Standard practice in PPO (and adopted by vLLM) is to clip advantages to
a reasonable range.

## Changes

### 1. GRPOConfig: Add `top_n_sigma` field

```python
# In trl/trainer/grpo_config.py
top_n_sigma: float = field(
    default=0.0,
    metadata={
        "help": (
            "Clip advantages to [mean - n*std, mean + n*std] after computation. "
            "Set to 0 to disable (default). Recommended: 3.0 for noisy rewards."
        }
    },
)
```

### 2. GRPOTrainer.__init__: Store the config

```python
# After line 751 (self.scale_rewards = args.scale_rewards)
self.top_n_sigma = args.top_n_sigma
```

### 3. GRPOTrainer._generate_and_score_completions: Apply clipping

After gradient computation (after line 2586), between `nan_to_num` and the process slice:

```python
advantages = torch.nan_to_num(advantages, nan=0.0)

# Top-n-sigma clipping: bound extreme advantage values
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

# Existing code continues:
process_slice = slice(...)
```

## Unified Diff (Trainer)

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -751,6 +751,7 @@ class GRPOTrainer(_BaseTrainer):
         self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0
         self.router_aux_loss_coef = args.router_aux_loss_coef
         self.scale_rewards = args.scale_rewards
+        self.top_n_sigma = args.top_n_sigma
         self.importance_sampling_level = args.importance_sampling_level
         self.off_policy_mask_threshold = args.off_policy_mask_threshold

@@ -2582,6 +2583,18 @@ class GRPOTrainer(_BaseTrainer):
         # Unscorable completions (every reward func returned None) carry no learning signal:
         # their reward is NaN here, so zero their advantage to keep them from moving the policy.
         advantages = torch.nan_to_num(advantages, nan=0.0)
+
+        # Top-n-sigma clipping: bound extreme advantage values to stabilize training
+        if self.top_n_sigma > 0:
+            mean_adv = advantages.mean()
+            std_adv = advantages.std()
+            upper = mean_adv + self.top_n_sigma * std_adv
+            lower = mean_adv - self.top_n_sigma * std_adv
+            advantages = advantages.clamp(lower, upper)
+            self._metrics[mode]["advantages/clip_upper"].append(upper.item())
+            self._metrics[mode]["advantages/clip_lower"].append(lower.item())
+            self._metrics[mode]["advantages/clipped_fraction"].append(
+                ((advantages == upper) | (advantages == lower)).float().mean().item()
+            )
+
         # Slice to keep only the local part of the data
         process_slice = slice(
             self.accelerator.process_index * len(prompts),

## Testing

1. With top_n_sigma=0 (default): no change in behavior
2. With top_n_sigma=3.0: advantages clipped to [mean-3*std, mean+3*std]
3. Extreme reward outliers are bounded, preventing destabilizing gradient updates

## Risk Assessment

- LOW risk: disabled by default (top_n_sigma=0), fully backward compatible
- Common practice: vLLM PPO uses similar clipping, DeepSpeed uses gradient_clipping
- Optional safety net for noisy reward functions
"""
