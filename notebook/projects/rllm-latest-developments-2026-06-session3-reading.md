# rLLM Latest Developments (June 2026) -- Session 3 Reading

**Repository**: `rllm-org/rllm` (https://github.com/rllm-org/rllm)
**Stars**: 5,631 | **Language**: Python | **Description**: "Democratizing Reinforcement Learning for LLMs"
**Latest release**: v0.3.0-pre (April 30, 2026)
**Last updated**: June 20, 2026 (actively developed, multiple PRs merged daily)

---

## 1. Recent Issues and PRs (June 2026)

### 1.1 GRPO Grouping Strategy Bug (#605 -> #667)

**Issue #605** (opened June 2, by TtLuckyyy): "AgentWorkflowPPOTrainer: GRPO groups by trajectory.uid instead of prompt/agent key"

This is a critical correctness bug. The core problem:

| Config | Expected GRPO grouping uid | Actual (buggy) uid | Group size |
|--------|---------------------------|---------------------|------------|
| `stepwise_advantage.enable=False` | Prompt-level UUID (`task_ids`) | `trajectory.uid` (unique per trajectory) | 1 (broken) |
| `mode=per_step` | `{prompt_uuid}_step{k}` | `trajectory.uid` | 1 (broken) |
| `mode=broadcast` | Prompt-level UUID; compute advantage on last step only | `trajectory.uid` | 1 (broken) |

**Impact**: With group size = 1, `std(group_rewards) = 0`, so GRPO advantage degrades to raw reward (equivalent to REINFORCE with zero variance reduction). ALL rLLM GRPO training before this fix produced incorrect advantages.

**Minimal repro**: 1 prompt, `rollout.n=4`, rewards `[1, 0, 1, 0]`:
- Current (buggy): advantages = `[1, 0, 1, 0]` (raw reward, no normalization)
- Expected (fixed): group mean = 0.5, advantages = `[+0.87, -0.87, +0.87, -0.87]` with `norm_adv_by_std_in_grpo=true`

**Root cause** (`rllm/trainer/algorithms/transform.py:127`): grouping key was `f"{task_id}:{trajectory.name}"` where `trajectory.name` defaults to `_DEFAULT_TRAJ_NAME` or the agent's name. Different agents have different names, so responses to the same prompt end up in separate groups.

**Fix (PR #667)** (merged June 19, by Jackie2049): Change grouping key from `f"{task_id}:{trajectory.name}"` to `task_id`:

```python
# Before (broken):
trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)

# After (correct):
trajectories_by_name[task_id].append(trajectory)
```

Cross-framework evidence cited in #667: verl, DeepSpeed, and Megatron all group by prompt/task_id for GRPO -- the trajectory/agent name is irrelevant to advantage computation.

**Compounding bug**: PR #667 notes that combined with #663 (Step.output was None -> all rewards = 0.0), pre-#663 training was doubly broken: wrong grouping AND zero rewards.

---

### 1.2 Step.output Population Bug (#663)

**PR #663** (merged June 17, by 87003697): "fix(engine): populate Step.output in trace_record_to_step"

`trace_record_to_step()` set `model_response=content` but omitted `output`, leaving it as `None`. During enrichment, the training Step replaces the agent's original Step, so `output` is lost. Evaluators checking `step.output` always see `None`, so all rewards = 0.0.

Fix: one-line addition of `output=content` alongside `model_response`. Verified on Koala H200x8 with Qwen3.5-9B sandbox_code training: rewards went from all-zero to non-zero.

---

### 1.3 Terminal-RL: Terminus-2 Harness, AgentFlow Redesign, Warm-Pool Training (#650)

**PR #650** (merged June 12, by jeffreysijuntan): Major merge of the `terminal-rl` branch (71 files, +3673/-1413).

Key sub-PRs:
- **#647**: Terminus-2 agent harness -- runs Harbor's Terminus-2 inside the rllm sandbox
- **#649**: pass@k via `--attempts` -- `rllm eval --attempts N` expands each task into N rollouts and reports the unbiased pass@k estimator for k=1..N. Includes three Modal fixes: chunked base64 uploads (Modal argv 64 KiB cap), `RLLM_MODAL_SANDBOX_TIMEOUT_S`, and right-sized snapshot-build sandbox lifetime.
- **AgentFlow redesign phases 0-3 (#636, #637, #638)**: one env-requirement join, always-hooks engine, CLI installs baked into snapshots, and engine ownership of sandbox (flows become plain programs)
- **#632, #635**: Snapshot + warm-pool acceleration for `rllm train` -- train sandboxed agents on separate local train/val benchmarks
- **#644**: Warm-queue / sandboxed-training hardening -- liveness-checked warm-queue pops, config-timeout passthrough
- **#641**: rllm-swesmith integration -- loader/eval/oracle honor task and dataset declarations for harbor-style dirs

---

### 1.4 Gateway Backend Improvements

**PR #658** (merged June 16, by jeffreysijuntan): "feat(gateway): cumulative token mode for the in-process (Tinker) path"

Extends `cumulative_token_mode` (drift-free multi-turn token-forwarding) to the in-process Tinker path. Previously only worked for the HTTP-proxy (vLLM/verl) backend; Tinker rollouts re-rendered + re-tokenized the full conversation every turn, causing token drift.

Key change: `rllm/gateway/tinker_adapter.py` handler now detects pre-tokenized `prompt` (list[int]) and samples straight from it via the engine's `get_token_output_from_token_input`. The `rllm-model-gateway/.../proxy.py` routes to `local_handler` when present (no HTTP worker), with new `_handle_cumulative_streaming_local` for SSE.

Net effect: Tinker rollouts get the same prefix-extension guarantee verl has -- turn N's prompt tokens are byte-for-byte the prior turns' prompt+completion tokens.

**PR #622** (merged June 6, by kylemontgomery1): "fix(rollout,gateway): stamp weight_version at dispatch, not completion"

Under `partial_rollout`, a weight sync landing mid-generation would mislabel old-weights rollout with newer version. Fix captures version at dispatch/request entry, not after generation completes.

**PR #623** (merged June 7, by listar2000): "feat(sampling): gateway-enforced sampling-parameters config for AgentFlow"

Gateway now overwrites configured sampling keys on every outgoing LLM request. `SamplingConfig` resolved at CLI/Hydra level, attached to gateway session, enforced at proxy. Removes `sampling_params_priority` knob. Gateway-always-wins for configured keys.

---

### 1.5 verl 0.8.0 Upgrade (#671, #672)

**PR #671** (opened June 20, by jeffreysijuntan): "feat(verl): upgrade training backend to verl 0.8.0"

Major dependency upgrade:

| Package | Before | After |
|---------|--------|-------|
| verl | 0.7.1 | 0.8.0 |
| transformers | >=4.55.0,<5.0.0 | >=5.5.3 |
| vllm | 0.17.0 | 0.22.1 |
| flash-attn | 2.8.1 | 2.8.3 |
| transformer-engine | 2.10 | 2.11 |
| megatron-core | 0.16.1 | 0.17.1 |

Key changes:
- `AsyncLLMServerManager` replaced with `LLMServerManager` / `LLMServerClient` (verl 0.8.0 promoted unified model engine)
- `use_legacy_worker_impl` removed entirely (only unified model engine path)
- Dropped `apply_all_verl_patches` / `patch_verl_dynamic_batch_sync` hooks -- fixes upstreamed in verl 0.8.0
- Net change: +57/-120 (simplification)

**PR #672** (opened June 20, stacked on #671): "fix(verl): port fully-async/separated rollout path to verl 0.8.0"

The `rllm/trainer/verl/async_agent_loop.py` subclassed verl's pre-0.8 experimental `AsyncLLMServerManager` / `AgentLoopWorker`, which verl 0.8.0 removed. Fix: re-export verl 0.8.0's upstreamed `FullyAsyncLLMServerClient`, `FullyAsyncLLMServerManager`, `FullyAsyncAgentLoopManager`. The dedicated `FullyAsyncAgentLoopWorker` is dropped -- partial-rollout moved into the client.

---

### 1.6 Async Trainer Step Merge and Sync Coordinator (#481)

**PR #481** (merged April 3, by kylemontgomery1): "Integrate fully async training to UnifiedTrainer"

Introduced:
- **SyncCoordinator** (`rllm/trainer/sync_coordinator.py`): manages weight sync between concurrent generation and training
- **TrajectoryGroupBuffer** (`rllm/trainer/buffer.py`): buffers trajectory groups for async pipeline
- **MetricsAggregator** (`rllm/trainer/metrics_aggregator.py`): unified metric aggregation across backends
- Async training loop in UnifiedTrainer with concurrent generation/training
- `on_policy_updated` hook added to backend protocol for weight sync/checkpointing

Currently only for Tinker backend. The sync coordinator manages the lifecycle of weight versions during async training, ensuring the rollout engine always uses the latest policy weights while the optimizer updates happen concurrently.

---

### 1.7 Unified Trainer Graduation (#607)

**PR #607** (merged June 3, by jeffreysijuntan): "refactor: graduate unified trainer out of rllm/experimental"

Key move map:

| Old (experimental) | New (canonical) |
|--------------------|-----------------|
| `rllm/experimental/common/` | `rllm/trainer/algorithms/` |
| `rllm/experimental/unified_trainer.py` | `rllm/trainer/unified_trainer.py` |
| `rllm/experimental/{protocol,buffer,sync_coordinator,metrics}.py` | `rllm/trainer/{backend_protocol,buffer,sync_coordinator,metrics_aggregator}.py` |
| `rllm/experimental/rollout/` | `rllm/engine/rollout/` |
| `rllm/experimental/engine/{gateway_manager,tunnel,tinker_adapter}.py` | `rllm/gateway/{manager,tunnel,tinker_adapter}.py` |
| `rllm/experimental/verl/` | `rllm/trainer/verl/` |
| `rllm/experimental/config/` | `rllm/trainer/config/` |

`from rllm.trainer import AgentTrainer` now returns the unified `AgentTrainer` (canonical import). Legacy `AgentTrainer` preserved for verl + fireworks backends.

---

### 1.8 Other Notable June 2026 PRs

| PR | Title | Key Change |
|----|-------|-----------|
| #669 | feat(advantage): add PRPO advantage | Population-Relative Policy Optimization -- standardizes reward by cross-prompt mean/std, useful for continual learning |
| #668 | feat(algorithm): ECHO env-observation loss | Auxiliary cross-entropy loss on environment-observation tokens (arXiv:2605.24517). Works across verl/tinker/fireworks. Defaults lambda=0.05, zero-cost on verl (shared forward pass), one extra backward on tinker/fireworks |
| #642 | Support Fireworks Training Platform | Third backend (alongside verl, tinker) for training via Fireworks managed service |
| #643 | Train/Inference Discrepancy should be Absolute value | Metric calculation fix |
| #660 | Fix invalid Runner export + incorrect metric aggregation | `__all__` had phantom "Runner" export; `reduce_metrics_by_trajectory_name` overwrote rewards instead of accumulating |
| #630 | deterministic optimizer steps with decoupled batch sizes | Decouples `mini_batch_size` from `global_batch_size`; deterministic `r = train_batch_size // ppo_mini_batch_size` optimizer steps; reduces padding waste (e.g., 298->300 instead of 298->384) |
| #627 | backend-agnostic dataloader | `StatefulTaskDataLoader` -- single, stateful, resumable dataloader owned by trainer; removes per-backend data construction; sample-granular cursor for mid-epoch resume |
| #661 | raise default sandbox concurrency from 4 to 64 | Throughput improvement for sandbox operations |
| #615 | add Tinker as OpenAI-compatible eval provider | Tinker engine now serves as eval backend |
| #618 | environment snapshots for cold-start acceleration | Sandbox pre-warming |
| #629 | warm queue to overlap sandbox creation with rollout | Overlap creation with training |
| #632 | snapshot + warm-pool acceleration for `rllm train` | Fast sandbox boot |
| #649 | pass@k via --attempts | Unbiased pass@k estimator |
| #617 | run harbor SWE-bench via mini-swe-agent on remote sandboxes | SWE-bench eval path |

---

## 2. Architecture Changes Since v0.3

### 2.1 v0.3.0-pre Release (April 30, 2026)

The v0.3.0-pre release was massive, spanning from #356 to #523 (167+ PRs). Highlights from the release body:

- **Tinker backend** moved to `rllm.trainer.tinker` (#409), deprecated legacy trainer APIs
- **Fireworks backend** introduced (#301, #438)
- **Model Gateway & Plugin System** (#412, #438): standalone proxy server for rLLM, gateway-enforced sampling
- **AgentFlow Framework** (#438, #445): `@rllm.rollout` and `@rllm.evaluator` decorators + cookbook examples
- **Fully Async Trainer** (#394, #481): async training loop with concurrent generation/training
- **Unified Trainer** (#398, #401, #408, #410, #416): RL advantage estimators (GRPO, REINFORCE++, RLOO), on-policy distillation
- **verl 0.7.1 upgrade** (#457, #462, #463, #466, #467): megatron-compatible batch padding, rollout log probs, NCCL dynamic batch sync
- **Multi-turn trajectory collapse** (#512): one row per trajectory instead of per step, fixing gradient-weighting bias
- **Training collapse fix** (#511): AgentFlow verl backend had EnrichMismatchError within 3-5 steps

### 2.2 Post-v0.3 Architecture (June 2026)

The current architecture after the June 2026 graduation (#607):

```
rllm/
  trainer/
    __init__.py          # exports unified AgentTrainer
    unified_trainer.py   # the canonical trainer
    backend_protocol.py  # protocol with on_policy_updated hook
    buffer.py            # TrajectoryGroupBuffer
    sync_coordinator.py  # SyncCoordinator for async weight sync
    metrics_aggregator.py # unified metrics
    algorithms/
      advantage.py       # GRPO, REINFORCE++, RLOO, PRPO, ECHO advantage estimators
      config.py          # AlgorithmConfig
      metrics.py         # per-trajectory metrics
      transform.py       # trajectory grouping and advantage computation (THE #605/#667 file)
      rl_algo.py
      performance.py
      rejection_sampling.py
      visualization.py
    config/              # Hydra configs (unified.yaml, rllm/, model_engine/)
    verl/                # verl backend
    tinker/              # Tinker backend
    fireworks/            # Fireworks backend
    deprecated/          # legacy trainer APIs
    distill/             # on-policy distillation
  engine/
    rollout/
      completer.py
      rollout_engine.py
      verl_engine.py
      tinker_engine.py
      fireworks_engine.py
      openai_engine.py
    remote_runtime/
  gateway/
    __init__.py
    manager.py           # GatewayManager (rollout server addresses)
    tinker_adapter.py    # in-process Tinker handler
    tunnel.py            # TCP tunneling
  eval/                  # evaluation harnesses, benchmarks, sandbox
  data/                  # StatefulTaskDataLoader
  cli/                   # rllm train, rllm eval CLI
```

### 2.3 Key Architectural Patterns

1. **Backend-agnostic trainer**: `UnifiedTrainer` owns data loading, metric aggregation, and the training loop. Backends (verl, tinker, fireworks) implement `BackendProtocol` with `on_policy_updated` hook for weight sync.

2. **Gateway as middleware**: The model gateway sits between the training loop and LLM inference. It enforces sampling params, stamps weight versions, manages sessions, and routes requests. For Tinker, the gateway uses an in-process `local_handler` (no separate vLLM worker).

3. **Cumulative token mode**: Drift-free multi-turn token-forwarding across both HTTP (verl) and in-process (Tinker) paths. Turn N's prompt tokens are byte-for-byte the prior turns' sampled tokens, preventing tokenizer drift.

4. **Sync coordinator + async pipeline**: The `SyncCoordinator` manages weight version lifecycle during async training. Concurrent generation and training with coordinator-managed weight sync. Currently Tinker-only, with verl fully-async path (#672) coming.

5. **Sandbox acceleration**: Snapshots, warm pools, and warm queues overlap sandbox creation with rollout generation, reducing cold-start latency.

---

## 3. New Bug Patterns and RTX 4090 Impact Assessment

### 3.1 GRPO Grouping Bug Pattern (#605/#667)

**Pattern**: GRPO advantage computation silently degrades when grouping keys isolate each trajectory into a singleton group. With `std(rewards) = 0` in a singleton, advantage = raw reward, eliminating all variance reduction.

**RTX 4090 Impact**: This bug is device-independent but compounds with RTX 4090 constraints:
- On a single RTX 4090 (24GB VRAM), smaller rollout groups (`rollout.n=2-4`) are common due to memory limits. With the grouping bug, even these small groups had size=1 (each trajectory isolated), so GRPO was completely degenerate.
- The fix (#667) restores proper grouping, but RTX 4090 users still face limited group sizes. With `rollout.n=4` and proper grouping, `std(group_rewards)` can still be low for uniform-reward tasks (e.g., easy math problems all score 1.0), producing near-zero advantages. The `deepscaler_math` dataset (#511) was introduced to reduce this "uniform-group rate."

### 3.2 Step.output Null Bug Pattern (#663)

**Pattern**: `trace_record_to_step()` sets `model_response` but omits `output`. During enrichment, the training Step replaces the agent's original, and `output` is lost. Evaluators see `None` -> rewards = 0.0.

**RTX 4090 Impact**: Device-independent, but on RTX 4090 with limited throughput, all-zero rewards mean entire training runs produce no learning signal, wasting GPU hours.

### 3.3 Multi-Turn Gradient-Weighting Bias (#512)

**Pattern**: Before #512, verl's `_process_trajectory` emitted one batch row per step. Under `seq-mean-token-mean`, a trajectory with N steps got Nx gradient weight, biasing learning toward many-stepped (typically failed) rollouts.

**RTX 4090 Impact**: On RTX 4090, multi-turn rollouts consume more memory per trajectory. The gradient weighting bias means failed (long) trajectories dominate the loss, which can amplify instability on consumer GPUs. After #512, each trajectory contributes one row, equalizing weight.

### 3.4 Padding Waste and Optimizer Step Indeterminacy (#506, #508, #630)

**Pattern**: Multi-turn rollouts produce variable row counts. Previous padding strategies (LCM of dp_size and mini_batch_size) created excessive waste rows. Padded rows with `response_mask=1` inflated the batch_num_tokens denominator, shrinking the effective learning rate.

**RTX 4090 Impact**: Critical. On RTX 4090 with limited VRAM:
- Excessive padding wastes memory. Example: 298 rows padded to 384 (86 wasted) vs. 300 (2 wasted) with #630's fix.
- `response_mask=1` on pad rows (#508) effectively reduced learning rate by `pad_fraction` (~40% in some configs), making training appear unstable on consumer GPUs where every byte of throughput matters.
- #630 decouples mini_batch_size from global_batch_size, giving deterministic optimizer steps.

### 3.5 Training Collapse on AgentFlow+verl (#511)

**Pattern**: AgentFlow training under verl crashed within 3-5 steps with `EnrichMismatchError`. Three compounding issues: per-step ID collisions in advantage scatter, trailing-malformed traces, and contaminated traces from prior retry attempts.

**RTX 4090 Impact**: Consumer GPUs often run smaller-scale configs that are more sensitive to these crashes. The fix adds ID uniqueness, trace robustness, and retry isolation.

### 3.6 Metric Aggregation Overwrite (#659)

**Pattern**: `reduce_metrics_by_trajectory_name` overwrites rewards when multiple trajectories share the same name, retaining only the last reward value instead of accumulating all.

**RTX 4090 Impact**: Makes dashboard metrics unreliable, which can mislead users about training stability on consumer GPUs where careful monitoring is essential.

### 3.7 Cumulative Token Drift (pre-#658)

**Pattern**: Tinker rollouts re-rendered + re-tokenized the full conversation each turn, potentially splitting tokens differently than concatenation of prior sampled tokens. The optimizer trains on a drifted sequence.

**RTX 4090 Impact**: Device-independent, but on consumer GPUs where every training step is precious, drift silently corrupts the training signal over many turns.

### 3.8 RTX 4090 Specific Considerations (No Direct Issues Found)

No issues in the rllm-org/rllm repo explicitly mention RTX 4090, but the following patterns from the broader RL-for-LLM ecosystem apply:

- **bf16 precision edge cases on Ada Lovelace**: Community reports (TRL, OpenRLHF) indicate NaN losses within ~50-100 GRPO steps on RTX 4090, likely related to TF32/bf16 behavior differences vs. A100.
- **24GB VRAM constraint**: Limits rollout.n (group size), batch sizes, and model sizes. Combined with the grouping bug (#605), this made GRPO particularly degenerate on RTX 4090 configs.
- **Recommended mitigations** (from community, not rllm-specific):
  - Reduce `rollout.n` to 4-8 (minimum for meaningful GRPO variance reduction)
  - Use gradient accumulation to compensate for smaller micro-batches
  - Lower `max_grad_norm` (0.5-1.0) for more aggressive clipping
  - Increase KL penalty coefficient to prevent policy drift
  - Consider `torch.backends.cuda.matmul.allow_tf32 = False` for Ada Lovelace precision control

---

## 4. Comparison with verl GRPO Implementation Differences

### 4.1 Grouping Strategy

| Aspect | rllm (pre-#667, buggy) | rllm (post-#667, fixed) | verl |
|--------|------------------------|------------------------|------|
| GRPO grouping key | `f"{task_id}:{trajectory.name}"` (per-trajectory) | `task_id` (per-prompt) | `task_id` / prompt-level UUID |
| Group size with rollout.n=4 | 1 (each trajectory isolated) | 4 (all responses to same prompt) | 4 (standard per-prompt) |
| Advantage computation | Degraded to REINFORCE (std=0) | Proper group-relative | Proper group-relative |

PR #667 explicitly cites verl as the reference: "verl groups by prompt/task_id -- all responses to the same prompt are in one batch group, regardless of which agent produced them."

### 4.2 Multi-Turn Trajectory Representation

| Aspect | rllm (pre-#512) | rllm (post-#512) | verl native |
|--------|-----------------|------------------|-------------|
| Rows per trajectory | N rows (one per step) | 1 row (prefix-merged) | 1 row (verl's own collapse) |
| Gradient weight | Nx weight for N-step trajectory | 1x weight per trajectory | 1x weight |
| response_mask | All tokens masked equally | Action tokens = 1, obs tokens = 0 | verl's own mask |
| Loss aggregation | seq-mean-token-mean (biased) | seq-mean-token-mean (correct) | configurable |

### 4.3 Async Training Architecture

| Aspect | rllm | verl |
|--------|------|------|
| Async path name | `FullyAsyncLLMServerManager` (re-exported from verl 0.8.0 via #672) | `FullyAsyncLLMServerManager` (upstreamed in 0.8.0) |
| Sync coordinator | `rllm.trainer.sync_coordinator.py` (custom) | No equivalent (verl relies on Ray actor coordination) |
| Weight sync mechanism | `on_policy_updated` hook in backend protocol + SyncCoordinator | verl's own `update_weight` RPC |
| Rollout engine ownership | Tinker: in-process; verl: LLMServerManager owns replicas (post-0.8.0) | verl: same ownership model (rllm adopts it) |

### 4.4 Loss Aggregation

rllm supports multiple `loss_agg_mode` options:
- `seq-mean-token-mean`: per-trajectory equal weight (matches Tinker default)
- `token-mean`: per-action-token semantics within a trajectory
- `seq-mean-token-sum`: legacy mode

verl's native GRPO uses its own configurable aggregation. rllm's multi-turn collapse (#512) + response_mask design ensures the two backends produce matching gradient weights.

### 4.5 Advantage Estimator Extensibility

rllm has a richer advantage estimator registry than verl:
- **GRPO** (standard group-relative)
- **REINFORCE++** (with baseline)
- **RLOO** (reinforce with leave-one-out)
- **PRPO** (#669, new): Population-Relative Policy Optimization -- cross-prompt standardization
- **ECHO** (#668, new): auxiliary env-observation loss on top of GRPO advantages

verl's advantage computation is more monolithic, embedded in the trainer class.

### 4.6 Backend Abstraction

rllm's `BackendProtocol` + `UnifiedTrainer` design means the same training loop, dataloader, metrics, and advantage computation work across verl, tinker, and fireworks. verl is a backend choice, not the entire training framework.

verl's native trainer (`PPOTrainer`, `GRPOTrainer`) is a single-path design -- verl IS the training framework, not a backend within a larger framework.

### 4.7 ECHO Algorithm Cross-Backend Implementation

PR #668 demonstrates the backend-agnostic design in practice:
- **verl**: env term is free and exact (single shared forward pass, observation tokens = `(responses != pad) & ~response_mask`)
- **tinker**: gradient-accumulated `cross_entropy` pass with `weights=lambda` on obs tokens, returned separately
- **fireworks**: same second pass, but normalization folded into weights/advantages and `GradAccNormalization` forced to `NONE`

This is a case where the same algorithm has different cost/precision profiles per backend, and rllm's architecture handles it gracefully.

---

## 5. Key Source Code References

### 5.1 GRPO Grouping (the #605/#667 bug and fix)

- `rllm/trainer/algorithms/transform.py:127` -- the grouping key line (bug was here, now fixed)
- `rllm/trainer/algorithms/advantage.py` -- advantage estimator registry (GRPO, REINFORCE++, RLOO, PRPO, ECHO)
- `rllm/trainer/algorithms/config.py` -- `AlgorithmConfig` with `adv_estimator` and `env_loss_coef`

### 5.2 Step.output Bug (#663)

- `rllm/engine/rollout/completer.py` or engine trace converter -- `trace_record_to_step()` (fix adds `output=content`)
- `rllm/trainer/algorithms/transform.py` -- enrichment step that replaces original Step

### 5.3 Sync Coordinator and Async Pipeline

- `rllm/trainer/sync_coordinator.py` -- manages weight version lifecycle during async training
- `rllm/trainer/buffer.py` -- `TrajectoryGroupBuffer` for async pipeline
- `rllm/trainer/metrics_aggregator.py` -- unified metric aggregation
- `rllm/trainer/backend_protocol.py` -- `BackendProtocol` with `on_policy_updated` hook

### 5.4 Gateway Architecture

- `rllm/gateway/manager.py` -- `GatewayManager` (rollout server addresses, weight version)
- `rllm/gateway/tinker_adapter.py` -- in-process Tinker handler with cumulative token mode
- `rllm/gateway/tunnel.py` -- TCP tunneling
- `rllm-model-gateway/.../proxy.py` -- HTTP proxy with cumulative token mode, sampling enforcement

### 5.5 Rollout Engines

- `rllm/engine/rollout/rollout_engine.py` -- base rollout engine with weight_version stamping
- `rllm/engine/rollout/verl_engine.py` -- verl backend engine
- `rllm/engine/rollout/tinker_engine.py` -- Tinker backend engine
- `rllm/engine/rollout/fireworks_engine.py` -- Fireworks backend engine

### 5.6 Trainer Core

- `rllm/trainer/unified_trainer.py` -- canonical `AgentTrainer` (formerly experimental)
- `rllm/trainer/__init__.py` -- exports unified `AgentTrainer`
- `rllm/data/dataloader.py` -- `StatefulTaskDataLoader` (backend-agnostic, stateful, resumable)

### 5.7 verl Backend Specifics

- `rllm/trainer/verl/async_agent_loop.py` -- re-exports from verl 0.8.0 (#672)
- `rllm/trainer/verl/verl_backend.py` -- `VerlBackend` with `LLMServerManager` (0.8.0)
- `rllm/trainer/verl/verl_engine.py` -- `VerlEngine` constructor changes (server_manager vs rollout_manager)

### 5.8 Tinker Backend

- `rllm/trainer/tinker/` -- Tinker-specific training implementation
- `rllm/gateway/tinker_adapter.py` -- in-process handler (key to Tinker's no-vLLM-worker design)

---

## 6. Active Open PRs (June 20, 2026)

| PR | Title | Status | Significance |
|----|-------|--------|-------------|
| #671 | feat(verl): upgrade training backend to verl 0.8.0 | Open | Major dependency upgrade, breaking changes, simplification |
| #672 | fix(verl): port fully-async/separated rollout path to verl 0.8.0 | Open (stacked on #671) | Fixes ImportError for async training, adopts upstreamed verl async |
| #668 | feat(algorithm): ECHO env-observation loss across verl/tinker/fireworks | Open | New algorithm, cross-backend auxiliary loss |
| #605 | AgentWorkflowPPOTrainer: GRPO groups by trajectory.uid instead of prompt/agent key | Open (question, fix in #667) | Critical correctness bug, now fixed |

---

## 7. Timeline Summary (June 2026)

| Date | Key Event |
|------|-----------|
| June 2 | Issue #605 opened (GRPO grouping bug reported) |
| June 3-6 | PR #607 (unified trainer graduation), #619-623 (gateway/sampling/config), #627 (backend-agnostic dataloader) |
| June 7-9 | PR #630 (deterministic optimizer steps), #632-635 (snapshot/warm-pool acceleration) |
| June 10-12 | PR #636-638 (AgentFlow redesign phases 0-3), #649-650 (pass@k, Terminal-RL merge) |
| June 14-15 | PR #651-656 (datasets, sandbox fixes, Dockerfile replay fix) |
| June 16 | PR #658 (cumulative token mode for Tinker), #659/#660 (metric aggregation bugs) |
| June 17 | PR #663 (Step.output fix), #665 (Fireworks model catalog) |
| June 19 | PR #667 (GRPO grouping fix -- THE critical correctness fix) |
| June 20 | PR #669 (PRPO advantage), #671/#672 (verl 0.8.0 upgrade + async path port) |

---

## 8. Key Insights for RTX 4090 Practitioners

1. **The #605/#667 grouping bug made ALL previous rllm GRPO runs on RTX 4090 completely degenerate** -- advantages were raw rewards with zero variance reduction. If you trained with rllm before June 19, 2026, your results are invalid.

2. **The #663 Step.output bug means all rewards were 0.0 before June 17** -- combined with #605, pre-#663 training was doubly broken.

3. **The #512 multi-turn collapse fix addresses gradient weighting bias** that disproportionately affected long rollouts (more steps = more weight). On RTX 4090 where throughput is limited, this bias silently amplified failed trajectories.

4. **The #630 deterministic optimizer step fix reduces padding waste** from 86 rows to 2 rows in typical configs, significantly improving RTX 4090 VRAM utilization.

5. **The #658 cumulative token mode for Tinker** eliminates tokenizer drift in multi-turn rollouts, a subtle correctness issue that compounds over training steps.

6. **ECHO (#668) is particularly interesting for RTX 4090**: it turns failed rollouts into dense supervision at no extra rollout cost. On consumer GPUs where rollout throughput is limited, maximizing signal from every rollout is critical. On verl, the env term is free (shared forward pass); on Tinker, it costs one extra backward but no extra rollout.

7. **PRPO (#669)** standardizes rewards across prompts, which can help with the "uniform-group rate" problem (all rewards = 1.0 for easy tasks) that is more pronounced with small `rollout.n` on RTX 4090.

---

## 9. Comparison Summary: rllm vs verl

| Dimension | rllm | verl |
|-----------|------|------|
| Philosophy | Backend-agnostic framework (verl is a backend choice) | Single-framework trainer (verl IS the framework) |
| GRPO grouping | Was buggy (#605), now fixed (#667) | Always prompt-level (correct) |
| Multi-turn handling | Trajectory collapse (#512) + response_mask | Native single-row trajectory |
| Async training | SyncCoordinator + BackendProtocol hook | Ray actor coordination, LLMServerManager |
| Advantage estimators | Registry (GRPO, REINFORCE++, RLOO, PRPO, ECHO) | Embedded in trainer class |
| Backend support | verl, Tinker, Fireworks | verl only (native) |
| Token drift prevention | cumulative_token_mode across all paths | Native verl completions path (no drift by design) |
| Sandbox/eval | Built-in sandbox, warm pools, snapshots | External evaluation |
| Gateway | Middleware proxy (sampling, routing, weight version) | No equivalent (verl manages internally) |
| Dependency | verl 0.7.1->0.8.0, vllm 0.17->0.22, transformers 4->5 | verl manages its own deps |

rllm's key differentiator is backend-agnosticism: the same training loop, advantage computation, and metrics work across verl, Tinker, and Fireworks. This comes with complexity (gateway, sync coordinator, backend protocol) but enables running the same algorithm on managed services (Fireworks), local GPU clusters (verl), or single-GPU setups (Tinker) without code changes.

verl's advantage is simplicity and production-grade distributed training (FSDP + Megatron hybrid parallelism). But its advantage computation is more monolithic, and its GRPO grouping has always been correct (prompt-level), which rllm only achieved with #667.
