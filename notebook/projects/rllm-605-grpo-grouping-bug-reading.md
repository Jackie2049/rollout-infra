# rLLM #605 GRPO Grouping Bug — Source-Level Reading

> 2026-06-18 | Issue #605 (OPEN, June 2) | CRITICAL correctness bug in GRPO advantage computation
> ★★★★★★★★ trajectory.uid used for GRPO grouping → group size 1 → advantage = raw reward → NO normalization
> ★★★★★★★★ Should use task_ids for prompt-level grouping → group size n → mean/std normalization → correct GRPO
> ★★★★★★★★ Affects agent_workflow_trainer.py + transform.py → 3 stepwise_advantage configs ALL broken

---

## 1. Bug Description

```
★★★★★★★★★ GRPO grouping uses trajectory.uid (per-trajectory UUID) instead of task_ids:

Current behavior:
  → batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
  → step_ids = trajectory.uid → unique UUID per trajectory
  → 4 rollouts of same prompt → 4 DIFFERENT uid values → 4 groups of size 1
  → GRPO advantage = raw reward → NO group normalization → NO relative comparison

Expected behavior:
  → GRPO groups by task_ids (prompt-level identifier)
  → 4 rollouts of same prompt → SAME task_ids → 1 group of size 4
  → GRPO advantage = (reward - group_mean) / group_std → normalized → relative comparison
  → With norm_adv_by_std_in_grpo=true: advantages ≈ ±0.87 → correct GRPO behavior

★★★★★★★★★ Minimal repro (1 prompt, rollout.n=4, rewards [1, 0, 1, 0]):
  Current: advantages ≈ [1, 0, 1, 0] (raw reward, group size 1)
  Expected: advantages ≈ [+0.87, -0.87, +0.87, -0.87] (normalized, group size 4)

★★★★★★★★★ This is a CORRECTNESS bug, NOT a performance bug:
  → Without proper grouping → GRPO advantage = individual reward → no relative comparison
  → GRPO relies on group-relative advantage → same prompt → same group → normalize
  → If grouped by uid → each trajectory unique → group size 1 → advantage = raw reward
  → Training still runs → but GRPO mechanism COMPLETELY broken → effectively = simple reward scaling
```

---

## 2. Per-Config Impact

```
★★★★★★★★★ 3 stepwise_advantage configs ALL affected:

| Config | Expected uid | Current uid | Group Size | Impact |
|--------|-------------|-------------|------------|--------|
| enable=False | task_ids (prompt-level) | trajectory.uid | 1 per rollout | ★★★★★★★★ GRPO BROKEN |
| mode=per_step | {task_id}_step{k} | trajectory.uid | 1 per step | ★★★★★★★★ GRPO BROKEN |
| mode=broadcast | task_ids (last step only) | trajectory.uid | 1 per rollout | ★★★★★★★★ GRPO BROKEN + broadcast ineffective |

★★★★★★★★★ stepwise_advantage only changes reward source, NOT grouping key:
  → enable=False: step_rewards vs traj_rewards → same uid → same grouping bug
  → mode=per_step: reward source changes → uid still trajectory.uid → still bug
  → mode=broadcast: transform emits all rows is_last=True → broadcast largely ineffective
```

---

## 3. Root Cause Code Path

```
★★★★★★★★★ Code path from transform.py → agent_workflow_trainer.py:

1. rllm/experimental/verl/transform.py:
   → trajectory.uid = UUID4 per trajectory → unique per rollout
   → NOT using task_ids (which would group by prompt)

2. agent_workflow_trainer.py (or agent_workflow_trainer_fireworks.py):
   → Before compute_advantage:
     batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
   → step_ids comes from trajectory.uid
   → This assignment OVERWRITES the uid field with trajectory-level UUID

★★★★★★★★★ The fix should be at the uid assignment:
   → enable=False / broadcast: uid = task_ids → prompt-level grouping
   → mode=per_step: uid = {task_id}_step{k} → per-step grouping within prompt
   → broadcast: transform should emit rows with is_last_step=False for intermediate steps

★★★★★★★★★ Additional issue with broadcast:
   → transform emits ALL rows with is_last=True
   → Broadcast path designed to: compute advantage on last step → broadcast to earlier steps
   → But if ALL rows are is_last=True → broadcast has nothing to broadcast to → ineffective
```

---

## 4. Impact on rLLM GRPO Training

```
★★★★★★★★★ RTX 4090 rLLM Tinker GRPO implications:

Training still RUNS → but GRPO mechanism is BROKEN:
  → Loss still computed → gradients still flow → optimizer still updates
  → But: advantage = raw reward → no relative comparison → GRPO = simple reward scaling
  → Model will still learn → but MUCH less efficiently than proper GRPO
  → Comparison: proper GRPO (group=4) vs broken GRPO (group=1) → different learning dynamics

★★★★★★★★★ Why group normalization matters for GRPO:
  → GRPO advantage = (reward - group_mean) / group_std
  → Group mean: average reward across n rollouts of same prompt
  → Group std: standard deviation → normalization scale
  → Without normalization: advantage = raw reward → no baseline → noisy signal
  → With normalization: advantage = relative comparison → cleaner signal → faster convergence

★★★★★★★★★ This bug affects ALL GRPO users on rLLM:
  → Anyone using rllm with GRPO → broken advantage computation
  → Training results will look like they work → but suboptimal
  → Very hard to detect → need to inspect advantage values to see group size
  → ★★★★★★★★ MUST fix before any serious GRPO training on rLLM!
```

---

## 5. Comparison with Other Frameworks

```
★★★★★★★★★ How other frameworks handle GRPO grouping:

verl:
  → GRPO groups by task_ids (prompt-level identifier) → correct
  → group_size = n_rollouts → proper normalization
  → Source: verl/trainer/ppo/advantage.py → GRPO computes group mean/std
  → ★★★★★★★★ verl GRPO grouping = CORRECT → reference implementation

DeepSpeed:
  → No native GRPO → uses verl or custom implementation
  → If using verl backend → inherits verl's correct grouping
  → If custom → need to implement grouping manually → risk of same bug

rLLM Tinker:
  → Uses verl transform.py → but with trajectory.uid bug → BROKEN
  → ★★★★★★★★ This bug ONLY exists in rLLM → verl proper GRPO is correct

★★★★★★★★★ Key insight: the bug is in rLLM's wrapper layer, NOT in the core GRPO algorithm:
  → GRPO algorithm itself (compute_advantage) is correct IF given proper uid
  → The bug is in HOW uid is assigned before compute_advantage is called
  → Fix: change uid assignment → task_ids for enable=False, {task_id}_step{k} for per_step
  → One line fix for enable=False case → slightly more for per_step/broadcast
```

---

## 6. Proposed Fix Analysis

```
★★★★★★★★★ Fix options (from issue author + our analysis):

Option 1: Simple fix for enable=False (1 line):
  → batch.non_tensor_batch["uid"] = batch.non_tensor_batch["task_ids"]
  → task_ids = prompt-level identifier → groups by prompt → correct
  → ★★★★★★★★ Simplest fix → 1 line → addresses the most common case

Option 2: Per-step fix for mode=per_step (few lines):
  → uid = f"{task_id}_step{k}" → groups by step within prompt
  → Need step index from transform → slightly more complex
  → Allows per-step advantage computation → finer-grained

Option 3: Broadcast fix for mode=broadcast (moderate):
  → uid = task_ids → same as Option 1
  → PLUS: fix transform to emit is_last_step=False for intermediate steps
  → Enables proper broadcast → advantage computed on last step → copied to earlier steps

★★★★★★★★★ Recommendation:
  → Option 1 first → 1 line → addresses most common use case
  → Option 2 second → per_step → more sophisticated
  → Option 3 third → broadcast → most complex
  → All three → complete GRPO grouping fix
```

---

## Key Findings Summary

★★★★★★★★★ rLLM #605: CRITICAL GRPO grouping bug → trajectory.uid vs task_ids → group size 1 → advantage = raw reward
★★★★★★★★★ 3 stepwise_advantage configs ALL affected → grouping key unchanged regardless of config
★★★★★★★★★ Bug in uid assignment: batch.non_tensor_batch["uid"] = step_ids → should be task_ids
★★★★★★★★★ Fix: 1 line for enable=False → few lines for per_step → moderate for broadcast
★★★★★★★★★ Training still runs → but GRPO mechanism broken → suboptimal learning dynamics
★★★★★★★★★ verl GRPO grouping = CORRECT → bug is in rLLM wrapper layer, NOT core algorithm
★★★★★★★★★ MUST fix before any serious GRPO training on rLLM Tinker

---

## References

- Issue #605: https://github.com/rllm-org/rllm/issues/605
- rLLM agent research: notebook/projects/rllm-latest-developments-2026-06-agent-research.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
