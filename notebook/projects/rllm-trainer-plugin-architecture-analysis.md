# rLLM Trainer Plugin Architecture — Analysis for #605 Fix

## Active Development: terminal-rl Branch

- `terminal-rl` is **46 commits ahead** of `main` (diverged, July 2026)
- All recent PRs (#713, #741, #742, #743) target `terminal-rl`
- New features are NOT on `main` — our fork's `fix/grpo-configurable-grouping` (based on `main`) is outdated

## Plugin Architecture (Two Hooks)

### 1. `@rllm.register_loss` — Custom Policy Losses (PR #713, MERGED)
- Pluggable policy losses: DPPO, CISPO, GSPO, etc.
- First-class API exposed at `rllm.register_loss`
- Composes with every backend (verl/tinker/fireworks)

### 2. `@rllm.register_adv_estimator` — Custom Advantage Estimators (PR #742, MERGED)
- Mirrors `@rllm.register_loss` API
- 6 built-in estimators registered:
  - `grpo` — standard GRPO
  - `reinforce` — REINFORCE
  - `reinforce++_baseline` — REINFORCE++ with baseline
  - `prpo` — Pairwise Reward PO
  - `rloo` — REINFORCE Leave-One-Out
  - `echo` — ECHO (zero-cost auxiliary loss)
- PKPO (#743) added as first custom estimator using this hook

## #605 Fix: New Direction

**Old approach** (on jackie2049/rllm `fix/grpo-configurable-grouping`):
- Direct code change to `AgentWorkflowPPOTrainer`
- Removes `trajectory.uid` as sole grouping key
- Targets `main` branch

**New approach needed**:
1. Target `terminal-rl` branch
2. Implement as `@rllm.register_adv_estimator("grouped_grpo")` or similar
3. Accept `grouping_key` config param (default: `"trajectory.uid"` for back-compat)
4. Use `rllm.register_adv_estimator` decorator (same pattern as PKPO #743)
5. Preserve `trajectory.name` for multi-agent (solver+judge) grouping

## Implementation Sketch

```python
@rllm.register_adv_estimator("grouped_grpo")
def grouped_grpo_advantages(rewards, grouping_key="trajectory.uid"):
    \"\"\"GRPO advantages with configurable grouping key.\"\"\"
    groups = group_by_key(rewards, grouping_key)
    advantages = []
    for group in groups:
        mean = group.mean()
        std = group.std() + 1e-8
        advantages.append((group - mean) / std)
    return torch.cat(advantages)
```

## Connection to PKPO (#743)

PKPO shows the exact pattern:
```python
# advantage.py
@register_adv_estimator("pkpo")
def calculate_pkpo_advantages_per_group(rewards, config):
    pass_at_k = config.algorithm.pass_at_k
    # ... PKPO math
```

Our fix follows the same registration + config pattern.

## Status
- Old branch: `jackie2049/rllm:fix/grpo-configurable-grouping` (commit f1d1c24, +30/-5)
- Branch based on: `main` (stale, diverged from `terminal-rl`)
- Needs: Rebase to `terminal-rl`, rewrite as `@rllm.register_adv_estimator` plugin
- Waiting for: User authorization to push to fork

