"""
TRL P8-1: GRPO group_size=1 detection warning

## Rationale

When `num_generations=1`, GRPO degenerates to REINFORCE with zero advantage:

```python
# GRPO advantage formula:
mean_grouped = nanmean(rewards.view(-1, 1), dim=1)  # = rewards (1 item per group)
advantages = rewards - mean_grouped                    # = 0 always
```

This means the policy gradient is identically zero — training does nothing.
The same degenerate case exists across ALL frameworks (verl, rLLM, TRL).

## Fix

Add a warning in `__init__` when `num_generations == 1`:

```python
if self.num_generations == 1:
    logger.warning(
        "num_generations=1 degenerates GRPO to REINFORCE (advantage=0). "
        "Set num_generations >= 2 for proper GRPO group normalization."
    )
```

## Unified Diff

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -708,6 +708,12 @@ class GRPOTrainer(_BaseTrainer):
         # Training arguments
         self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
         self.num_generations = args.num_generations  # = G in the GRPO paper
+        if self.num_generations == 1:
+            logger.warning(
+                "num_generations=1 degenerates GRPO to REINFORCE with zero advantage "
+                "(group mean = reward, group std = 0). Set num_generations >= 2 "
+                "for proper GRPO group normalization."
+            )
         self.max_tool_calling_iterations = args.max_tool_calling_iterations or sys.maxsize
         self.num_generations_eval = args.num_generations_eval or self.num_generations
```

## Testing

1. Create GRPOTrainer with num_generations=1 → warning logged
2. Create GRPOTrainer with num_generations=2 → no warning
3. Training with num_generations=1: loss computed but advantage=0 → no gradient update

## Risk Assessment

- VERY LOW risk: warning-only change, no behavior modification
- 0 reviewers would oppose — pure safety improvement
- Aligns with cross-framework GRPO degeneration analysis
"""
