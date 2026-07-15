# rLLM terminal-rl Branch Architecture Deep Reading

**Date**: 2026-07-15 (Session 9 continuation)
**Branch**: terminal-rl (46 commits ahead of main)
**Repo**: rllm-org/rllm

---

## 1. Architecture Overview

rLLM terminal-rl introduces a **UnifiedTrainer** with plugin-based registry architecture for both advantage estimators and policy losses. This mirrors verl V1's registry pattern but with cross-backend portability (verl/tinker/fireworks).

### Key Components
```
rllm/trainer/
  ├── unified_trainer.py     (1152 lines) — UnifiedTrainer, TrainerState, fit() loop
  ├── sync_coordinator.py    (155 lines)  — Async staleness + capacity control
  ├── algorithms/
  │   ├── advantage.py       — 6 registered estimators via @register_adv_estimator
  │   ├── loss.py            (630 lines)  — 9 registered losses via @register_loss
  │   ├── config.py          — AlgorithmConfig, estimator_map, loss_fn_map
  │   ├── transform.py       — Episode → TrajectoryGroup pipeline, grouping_key
  │   ├── rl_algo.py         — GRPO/RLOO advantage math
  │   ├── rejection_sampling.py — RejectionSamplingState
  │   └── performance.py     — simple_timer
  │   ├── metrics.py         — reduce_metrics_lists
  │   └── visualization.py   — print_metrics_table
  ├── backend_protocol.py    — BackendProtocol ABC
  ├── buffer.py              — TrajectoryGroupBuffer
  └── metrics_aggregator.py  — MetricsAggregator
```

---

## 2. Advantage Estimator Registry

### 6 Registered Estimators

| Name | Enum | Math | Notes |
|------|------|------|-------|
| `grpo` | GRPO | group mean±std normalization | Default, same as verl |
| `reinforce` | REINFORCE | A = R (no baseline) | On-policy, simplest |
| `reinforce_plus_plus_baseline` | REINFORCE_PLUS_PLUS_BASELINE | centered reward (mean subtracted) | Control variate baseline |
| `prpo` | PRPO | prompt-level reward | Prompt-dependent |
| `rloo` | RLOO | leave-one-out baseline | Same as verl RLOO |
| `echo` | ECHO | GRPO advantages + env loss | arXiv:2605.24517 |

### Registry Pattern
```python
RLLM_ADV_ESTIMATOR_REGISTRY: dict[str, Callable] = {}

@register_adv_estimator(name)
def my_estimator(rewards, algorithm_config, **kwargs):
    # kwargs includes traj_groups for per-trajectory metadata
    return advantages_by_group, returns_by_group
```

### Estimator-Default Loss Mapping
```python
_ESTIMATOR_DEFAULT_LOSS = {
    rLLMAdvantageEstimator.ECHO: "echo",
}
```
All other estimators default to whatever `algorithm.loss_fn` is set (typically `ppo_clip` or user choice).

---

## 3. Policy Loss Registry

### 9 Registered Losses

| Name | Registration | Math | Aggregation Override | Key Feature |
|------|-------------|------|---------------------|-------------|
| `ppo_clip` | `@register_loss("ppo_clip")` | min(ratio·A, clipped·A) | None (config default) | Standard PPO/GRPO |
| `dppo_tv` | `@register_loss("dppo_tv")` | binary TV divergence mask | None | arXiv:2602.04879, C=inf ratio |
| `dppo_kl` | `@register_loss("dppo_kl")` | binary KL divergence mask | None | arXiv:2602.04879, KL variant |
| `cispo` | `@register_loss("cispo")` | clamp(ratio).detach()·A·logp_curr | None | arXiv:2506.13585, all tokens keep gradient |
| `gspo` | `@register_loss("gspo", agg_mode="seq-mean-token-mean")` | sequence-level ratio s_i = π(y_i)/π_old(y_i)^(1/|y_i|) | **seq-mean-token-mean** | arXiv:2507.18071 |
| `icepop` | `@register_loss("icepop")` | double-sided IS band [alpha, beta] | None | arXiv:2510.18855 |
| `reinforce` | `@register_loss("reinforce")` | -A·logp_curr | None | No ratio, no clip |
| `reinforce_kl` | `@register_loss("reinforce_kl")` | IS-weighted REINFORCE + fwd/bwd KL | None | Requires logp_rollout |
| `echo` | `@register_loss("echo")` | ppo_clip + env_loss_coef·CE(obs) | None | arXiv:2605.24517 |

### LossContext Dataclass
```python
@dataclass
class LossContext:
    logp_curr: torch.Tensor     # requires_grad=True — only differentiable input
    logp_old: torch.Tensor      # behavior/old-policy log-probs (ratio denominator)
    advantages: torch.Tensor    # per-token advantage estimates
    action_mask: torch.Tensor   # 1.0 on assistant/action tokens
    obs_mask: torch.Tensor      # 1.0 on env-observation tokens (for ECHO)
    aggregate: Callable         # injected by backend, (loss, mask, mode?) -> scalar
    logp_ref: torch.Tensor | None     # reference-policy for KL term
    logp_rollout: torch.Tensor | None # true inference/sampling log-probs
    params: dict                # loss hyperparameters
    backend: str                # "verl" | "tinker" | "fireworks"
```

### Aggregation Modes (3 canonical)
```python
LOSS_AGG_MODES = ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum")
DEFAULT_LOSS_AGG_MODE = "seq-mean-token-mean"  # every sequence equal weight
```

- `token-mean`: Σ(loss·mask) / Σ(mask) — every token equal
- `seq-mean-token-mean`: mean within sequence, then mean over sequences — every sequence equal
- `seq-mean-token-sum`: sum within sequence, then mean over sequences

### Native-First Routing
```python
def resolve_loss(algorithm_config, native_losses):
    # If backend has a fused kernel for this loss → use native (None returned)
    # Only custom rLLM losses fall back to forward_backward_custom path
    if native_losses is not None and name in native_losses:
        return None  # prefer backend's native fused kernel
    if not is_custom_loss(name):
        return None  # backend-native name (e.g. verl "vanilla")
    return ResolvedLoss(name=name, fn=get_loss(name), params=params, agg_mode=agg_mode)
```

This means: verl's `dppo_tv` and `cispo` use verl-native kernels when available, falling back to rLLM custom only on tinker/fireworks.

### Entry Point Discovery
```python
def _discover_entry_point_losses():
    # Packages declare in pyproject.toml:
    # [project.entry-points."rllm.losses"]
    # my_dppo = "my_pkg.losses:my_dppo"
    # Lazy: triggered on registry miss, runs once per process
```

### Loss Plugin Loading
```python
def load_loss_plugins(modules: list[str]):
    # algorithm.loss_plugins: ["my_pkg.losses"]
    # Import each module so @register_loss decorators run
```

---

## 4. TrajectoryGroup Grouping (★★★ CRITICAL for PR #2)

### Default Grouping Key
```python
# In _build_trajectory_groups():
for episode in episodes:
    task_id = episode.task_id
    for trajectory in episode.trajectories:
        trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
```

**Default grouping key = `task_id:trajectory.name`**

This is EXACTLY what our PR #2 addresses. The original #605 bug was that `trajectory.name` caused singleton groups when all trajectories in an episode had different names (multi-agent scenario). Our fix makes the grouping_key **configurable**:

- Default: `task_id:name` (preserves multi-agent grouping for solver+judge)
- Single-agent GRPO: `task_id` (groups all trajectories from same task together)

### Trajectory Name Imputation
```python
def _impute_trajectory_names(episodes, config):
    # Unnamed trajectories → "{default_traj_name}_{position}"
    # e.g. "unnamed_0", "unnamed_1"
    # config.impute_missing_names = True (default)
    # config.drop_unnamed_traj = False
```

### TrajectoryGroup Structure
```python
TrajectoryGroup:
    role: str                    # group_role for estimator_map routing
    trajectories: list[Trajectory]
    metadata: list[dict]         # per-trajectory metadata
```

### Multi-Agent Estimator Routing
```python
# AlgorithmConfig.estimator_map:
#   {role: estimator} or {role: (estimator, loss_fn)}
# e.g. {"solver": "grpo", "judge": ("rloo", "ppo_clip")}
# Different roles get different advantage estimators AND loss functions
```

---

## 5. SyncCoordinator (Async Training)

### Capacity Control
```python
class SyncCoordinator:
    def has_capacity(self) -> bool:
        staleness_ok = in_flight < max_in_flight_groups
        concurrency_ok = running_rollouts < max_concurrent_rollouts
        return staleness_ok and concurrency_ok
```

### Staleness Budget
```python
max_in_flight_groups = (1 + staleness_threshold) * trigger_parameter_sync_step * mini_batch_size
```

### Behavior Spectrum (4 modes)
| staleness | sync_step | partial_rollout | Mode |
|-----------|-----------|----------------|------|
| 0 | 1 | — | On-policy (sync every step) |
| 0 | K | — | Stream off-policy (sync every K steps) |
| >0 | — | False | Async with staleness |
| >0 | — | True | Async with partial rollout |

### Event-Driven Dispatch
- `_capacity_event`: asyncio.Event, level-triggered (re-check after wake)
- `_generation_paused`: blocks generation during validation or weight sync
- `_in_flight_tasks`: set of asyncio.Task objects with done callbacks
- `_task_errors`: collects rollout failures, surfaces via raise_if_task_failed()

---

## 6. UnifiedTrainer (1152 lines)

### Architecture
```python
class UnifiedTrainer:
    # Engine paths:
    # 1. agent_flow + evaluator → AgentFlowEngine (gateway-based, local)
    # 2. remote_runtime → RemoteAgentFlowEngine (gateway-based, remote)
    # 3. workflow_class → UnifiedWorkflowEngine (direct)

    def __init__(self, backend_cls, config, workflow_class, ...):
        self.backend = backend_cls(config=config)
        self.async_config = AsyncTrainingConfig.from_config(...)
        # Gateway for agent_flow / remote_runtime paths
        if agent_flow and (evaluator or hooks):
            self._gateway = GatewayManager(config, mode=...)
            self.agent_workflow_engine = AgentFlowEngine(...)
        elif remote_runtime:
            self._gateway = GatewayManager(config, mode=...)
            self.agent_workflow_engine = RemoteAgentFlowEngine(...)
        else:
            self.agent_workflow_engine = UnifiedWorkflowEngine(...)
```

### TrainerState (per-step reset)
```python
@dataclass
class TrainerState:
    rs_state: RejectionSamplingState
    global_step: int = 0
    epoch: int = 0
    total_steps: int = 0
    weight_version: int = 0
    episodes: list[Episode] | None = None
    trajectory_groups: list[TrajectoryGroup] | None = None
    backend_batch: Any | None = None
```

### Key Config Systems
- `CompactFilteringConfig`: 12 termination reasons with per-reason masking
  - Infra errors: verifier_timeout, grading_error, sandbox_error, agent_setup_timeout, env_start_timeout, model_error
  - Trajectory errors: max_prompt_length_exceeded, max_response_length_exceeded, env_done, max_turns_exceeded, timeout, unknown, error

- `TransformConfig`: impute_missing_names, default_traj_name, drop_unnamed_traj

- `AsyncTrainingConfig`: enable, mini_batch_size, fwd_bwd_group_size, staleness_threshold, trigger_parameter_sync_step, partial_rollout, episode_offload_dir

---

## 7. Cross-Framework Comparison: rLLM vs verl V1

| Feature | rLLM terminal-rl | verl V1 |
|---------|-------------------|---------|
| Advantage registry | @register_adv_estimator (6 estimators) | @register_adv_estimator (14 estimators) |
| Loss registry | @register_loss (9 losses) | @register_policy_loss (11 losses) |
| Loss aggregation | 3 modes (token-mean, seq-mean-token-mean, seq-mean-token-sum) | agg_loss (verl-specific) |
| Native-first routing | resolve_loss checks backend fused kernels | All in-process (verl only) |
| Entry-point discovery | rllm.losses entry point group + loss_plugins | N/A (monorepo) |
| LossContext | cross-backend (verl/tinker/fireworks) | verl-specific (advantages, old_log_probs, ...) |
| Grouping | task_id:trajectory.name (configurable) | group_by_prompt (fixed) |
| Async training | SyncCoordinator + 4 staleness modes | 3 trainer types (sync/colocate/separate) |
| Episode filtering | CompactFilteringConfig (12 reasons) | N/A |
| Multi-agent | estimator_map + loss_fn_map per role | N/A (single estimator) |
| Backend | BackendProtocol ABC | verl-specific |

---

## 8. Key Insights for GRPO / RTX 4090

1. **★★★★★★★★ GROUPING KEY = `task_id:name`**: This is the default. For single-agent GRPO, change to `task_id` only. Our PR #2 makes this configurable.

2. **★★★★★★★★ DPPO losses (dppo_tv, dppo_kl)**: Use C=∞ ratio (no truncation) + divergence mask. The mask replaces PPO clipping. For GRPO: dppo_tv may be better than ppo_clip for multi-token trajectories.

3. **★★★★★★★★ CISPO**: All tokens keep gradient (no clipping zeros). `clipped.detach() * A * logp_curr` — the weight is detached (stop-gradient on ratio) but gradient flows through logp_curr. Better gradient utilization than PPO-clip.

4. **★★★★★★★★ GSPO**: Sequence-level ratio `s_i = (π(y)/π_old(y))^(1/|y|)`. Forces `seq-mean-token-mean` aggregation. Length-normalized ratio addresses PPO's long-sequence bias.

5. **★★★★★★★★ IcePop**: Requires `logp_rollout` (true inference log-probs), not `logp_old`. Double-sided IS band [alpha, beta]. Masks out-of-range tokens. Managed backends always provide it; verl needs rollout log-prob capture.

6. **★★★★★★★★ reinforce_kl**: IS-weighted REINFORCE + fwd/bwd KL to behavior policy. Requires `logp_rollout`. Perfect pair with bypass_mode (no proximal forward needed). Score-function vs estimator KL forms.

7. **★★★★★★★★ ECHO**: GRPO advantages + env_loss_coef (0.05) * CE(obs_mask). The auxiliary loss is INSIDE the loss body (no separate framework). Default pairing: estimator="echo" → loss="echo".

8. **★★★★★★★★ seq-mean-token-mean**: rLLM default aggregation. Every sequence gets equal weight, not every token. This matches GRPO philosophy (group-level normalization). Different from verl's token-mean default.

9. **★★★★★★★★ Native-first routing**: If verl has a fused kernel for dppo_tv/cispo → use verl-native (faster). Only custom rLLM losses fall back to forward_backward_custom.

10. **★★★★★★★★ SyncCoordinator**: Async training with continuous capacity control. Level-triggered dispatch (no burst at sync). Staleness budget = (1 + staleness_threshold) × sync_step × mini_batch_size.

---

## 9. Our PR #2 Context

Our Jackie2049/rllm PR #2 (terminal-rl branch) adds `GROUPED_GRPO` as a configurable advantage estimator with a customizable `grouping_key`.

### How it fits into terminal-rl architecture:
- `@register_adv_estimator` hook exists (PR #742 MERGED)
- `AlgorithmConfig.estimator_map` supports per-role routing
- `_build_trajectory_groups()` uses `f"{task_id}:{trajectory.name}"` as default key
- `traj_grouping_hook` in `transform_episodes_to_trajectory_groups()` allows custom grouping

### What our PR #2 adds:
- A new estimator `GROUPED_GRPO` registered via `@register_adv_estimator`
- Configurable `grouping_key` (default `"task_id:name"`, alternative `"task_id"`)
- Same math as GRPO but with configurable grouping

### Maintainer feedback (jeffreysijuntan):
- trajectory.name is intended for multi-agent grouping (solver+judge)
- Fix should make grouping CONFIGURABLE, not just remove name
- Our PR #2 addresses this: both single-agent (`task_id`) and multi-agent (`task_id:name`) preserved

---

## 10. PR #2 vs rLLM Built-in Alternatives

The terminal-rl branch now has its own grouping mechanisms:
1. `traj_grouping_hook` — custom hook function for grouping logic
2. `estimator_map` — per-role estimator routing
3. `TransformConfig.impute_missing_names` — name imputation policy

Our PR #2 adds `GROUPED_GRPO` with configurable `grouping_key` as a **clean, self-contained** solution that works within the existing framework hooks. The `traj_grouping_hook` approach would require writing a full hook function (more complex). Our approach is simpler: just set `grouping_key` in config.

---

## Session Stats
- **Source files read**: 6 (advantage.py, loss.py, config.py, transform.py, sync_coordinator.py, unified_trainer.py)
- **Registry counts**: 6 advantage estimators, 9 policy losses, 3 aggregation modes
- **Key architecture**: UnifiedTrainer + BackendProtocol + plugin registries + native-first routing
- **PR #2 validation**: Default grouping_key = `task_id:name`, our configurable approach fits terminal-rl hooks
