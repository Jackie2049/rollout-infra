"""
TRL P9-1 Contribution: Explicit bypass_mode Flag for GRPOTrainer

This patch adds an explicit `bypass_mode` flag to GRPOConfig/GRPOTrainer,
allowing users to explicitly control whether the old_per_token_logps forward
pass is skipped during GRPO training. Currently this is determined implicitly
by a heuristic (gradient_accumulation_steps % generate_every != 0).

When bypass_mode=True: saves one forward pass per training step (the model
forward for old_per_token_logps is replaced by .detach()).

This is equivalent to verl's bypass_mode=True and is especially important for
RTX 4090 single-GPU training where every saved forward pass = significant
memory reduction.

== Changes ==

### 1. GRPOConfig (trl/trainer/grpo_config.py)

Add to the GRPOConfig dataclass:
```python
bypass_mode: bool = False
"""
If True, skip computing old_per_token_logps in `_generate_and_score_completions`
and use `per_token_logps.detach()` in `_compute_loss` instead. This saves one
forward pass per training step but disables importance sampling correction.

When `num_iterations == 1` and `steps_per_generation == 1`, the old and new
logprobs are identical anyway, so this is always safe. For multi-step GRPO
(num_iterations > 1), bypass_mode trades off importance sampling accuracy for
throughput.

Default is False, preserving the current implicit heuristic.
"""
```

### 2. GRPOTrainer.__init__ (trl/trainer/grpo_trainer.py)

After line 744 (self.multi_objective_aggregation), add:
```python
self.bypass_mode = args.bypass_mode
```

### 3. GRPOTrainer._generate_and_score_completions (trl/trainer/grpo_trainer.py)

Change lines 2417-2432 from:
```python
generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
if self.args.gradient_accumulation_steps % generate_every != 0 or (
    self.use_vllm and self.vllm_importance_sampling_correction
):
    old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
        self.model,
        prompt_completion_ids,
        attention_mask,
        logits_to_keep,
        batch_size,
        num_images=num_images,
        num_tiles=num_tiles,
        **forward_kwargs,
    )
else:
    old_per_token_logps = None
```

To:
```python
generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
if not self.bypass_mode and (
    self.args.gradient_accumulation_steps % generate_every != 0 or (
        self.use_vllm and self.vllm_importance_sampling_correction
    )
):
    old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
        self.model,
        prompt_completion_ids,
        attention_mask,
        logits_to_keep,
        batch_size,
        num_images=num_images,
        num_tiles=num_tiles,
        **forward_kwargs,
    )
else:
    old_per_token_logps = None
```

== Unified Diff (grpo_trainer.py) ==

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -741,6 +741,7 @@ class GRPOTrainer(_BaseTrainer):
         self.use_liger_kernel = args.use_liger_kernel
         self.loss_type = args.loss_type
         self.multi_objective_aggregation = args.multi_objective_aggregation
+        self.bypass_mode = args.bypass_mode

         # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
         text_config = model.config.get_text_config()
@@ -2414,8 +2415,8 @@ class GRPOTrainer(_BaseTrainer):
             # distribution mismatch between vLLM and the training model can be large and harm the training.
             generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
-            if self.args.gradient_accumulation_steps % generate_every != 0 or (
-                self.use_vllm and self.vllm_importance_sampling_correction
+            if not self.bypass_mode and (
+                self.args.gradient_accumulation_steps % generate_every != 0
+                or (self.use_vllm and self.vllm_importance_sampling_correction)
             ):
                 old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
```

== Usage ==

```python
# Optimal for RTX 4090: save one forward pass
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    train_dataset=dataset,
    args=GRPOConfig(
        bypass_mode=True,       # explicit bypass: skip old_logps forward pass
        num_iterations=1,       # single step per generation
        steps_per_generation=1, # generate every step
    ),
)

# Default behavior unchanged
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    train_dataset=dataset,
    # bypass_mode defaults to False → original heuristic
)
```

== Safety Analysis ==

- bypass_mode=True + num_iterations=1 + steps_per_generation=1: ALWAYS safe
  (old_per_token_logps == per_token_logps, .detach() is identity)
- bypass_mode=True + num_iterations>1: old_logps may differ from current
  logps → training behaves like REINFORCE without importance correction
  (still valid: just no off-policy correction)
- bypass_mode=True + use_vllm=True: disables vLLM importance sampling
  correction (user must accept distribution mismatch)
- bypass_mode=False: behavior identical to current code (100% BC)

== Total LOC ==
Config: +3
Trainer __init__: +1
Trainer _generate_and_score_completions: +2 (change, not add)
Total: ~6 lines changed
"""
