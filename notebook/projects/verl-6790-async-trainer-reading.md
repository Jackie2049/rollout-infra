# verl PR #6790: Separate Async Trainer with Switch+Offload Strategies — Source-Level Reading

> 2026-06-18 | MERGED | Author: Begunner | +17/-16 | 2 files changed
> ★★★★★★★★ CRITICAL: This PR removes the NotImplementedError barrier — separate_async trainer is NOW RUNNABLE
> ★★★★★★★★ Hybrid engine can now DYNAMICALLY switch between trainer/rollout modes (switch_to_trainer / switch_to_rollout)
> ★★★★★★★★ Two independent LLMServerManagers: hybrid (co-located with training) + standalone (disaggregated)
> ★★★★★★★★ `start_rank` parameter avoids Ray named-actor collisions when hybrid and standalone replicas coexist
> ★★★★★★★★ `should_switch_to_rollout()` currently returns False — switch strategies are TODO, coming in future PRs
> ★★★★★★★★ bypass_mode FORCED True — Decoupled PPO not yet supported in separate async
> ★★★★★★★★ Offload strategy TODOs in both switch_to_rollout and switch_to_trainer — "disable auto offload in config and offload according to the switch strategy"
> ★★★★★★★★ RTX 4090 IMPACT: separate_async requires nnodes>0 + n_gpus>0 + non-naive checkpoint backend → NOT viable on single GPU! This is multi-GPU/multi-node only

---

## 1. PR Overview

### 1.1 PR Metadata

| Field | Value |
|-------|-------|
| Title | `[trainer] feat: A runnable separate async trainer` |
| Author | Begunner |
| State | MERGED (closed) |
| Merged at | 2026-06-17T14:25:45Z |
| Base branch | main |
| Head branch | separate-async |
| Additions/Deletions | +17/-16 |
| Files changed | 2 |
| Commits | 2 |

### 1.2 PR Body (verbatim)

> "See above. Currently the colocated group never switches to rollouter, similar to the fully async implementation. Switch and offload strategies will be introduced later."

Key claim: **"currently the colocated group never switches to rollouter"** — the hybrid replicas stay in trainer mode, never switch to rollout mode. Only the standalone replicas handle generation.

### 1.3 What Changed

**File 1: `verl/trainer/ppo/v1/trainer_separate_async.py` (+8/-10)**

| Change | Before | After | Significance |
|--------|--------|-------|-------------|
| NotImplementedError removed | `raise NotImplementedError(...)` | Removed | **NOW RUNNABLE** — was blocked before |
| `super().__init__()` moved | After `bypass_mode` assignment | Before `bypass_mode` assignment | Parent init must complete before config mutation |
| `bypass_mode=True` kept | `self.config.algorithm.rollout_correction.bypass_mode = True` | Same position, moved after super init | bypass_mode STILL forced |
| `LLMServerManager.create()` now passes `start_rank` | No start_rank | `start_rank=hybrid_num_replicas` | Avoid Ray actor name collision |
| `on_validate_end()` removed | Had switch_to_trainer on validate end | Removed | Switch happens in on_sample_end instead |
| `switch_to_rollout()` added TODO | No comment | `# TODO: disable auto offload in config and offload according to the switch strategy` | Offload strategy placeholder |
| `switch_to_trainer()` added TODO | No comment | `# TODO: disable auto offload in config and offload according to the switch strategy` | Offload strategy placeholder |
| `should_switch_to_rollout()` | Not implemented | `return False` | **Always False** — trainer never voluntarily switches to rollout |

**File 2: `verl/workers/rollout/llm_server.py` (+9/-6)**

| Change | Before | After | Significance |
|--------|--------|-------|-------------|
| `LLMServerManager.__init__` adds `start_rank` param | No start_rank | `start_rank: int = 0` | Instance-level default for replica rank offset |
| `_initialize_llm_servers` defaults | `start_rank: int = 0` (parameter default) | `start_rank: int = None` → falls back to `self.start_rank` | Parameter default moves to instance attribute |

---

## 2. Architecture Deep Dive: PPOTrainerSeparateAsync

### 2.1 Class Structure (149 lines)

**File**: `verl/trainer/ppo/v1/trainer_separate_async.py`

```
@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """Asynchronous PPO trainer
    1. Trainer and rollout are separate, trainer may switch to rollout if idle.
    2. Partial rollout is enabled.
    """

    __init__(43-61):         Config validation + super().__init__() + bypass_mode=True
    _setup(63-84):           ★★★ Two LLMServerManagers + CheckpointEngineManagers
    get_llm_client(85-87):   Returns FullyAsyncLLMServerClient from standalone manager
    on_init_end(89-92):      Update weights on both checkpoint managers
    on_train_begin(94-98):   Warmup batches from config
    on_validate_begin(100-103): Switch hybrid→rollout if in TRAINER mode
    on_sample_begin(105-108): Switch hybrid→rollout if needed (currently never triggers)
    on_sample_end(110-113):  Switch hybrid→trainer (always, if in ROLLOUT mode)
    on_step_end(115-119):    Periodic weight sync to standalone replicas
    switch_to_rollout(121-126): Update weights → resume → add replicas to balancer + TODO offload
    switch_to_trainer(128-133): Remove from balancer → abort → sleep + TODO offload
    add_replicas_to_balancer(135-140): Add hybrid replicas to standalone's load balancer
    remove_replicas_from_balancer(142-144): Remove hybrid replicas from standalone's load balancer
    should_switch_to_rollout(146-148): return False — ALWAYS False currently
```

### 2.2 Dual LLMServerManager Architecture

★★★★★★★★★ This is the CORE architectural innovation of #6790:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 PPOTrainerSeparateAsync                                  │
│                                                                         │
│  ┌─────────────────── HYBRID ───────────────────────┐                   │
│  │  LLMServerManager (rank 0..N-1)                   │                   │
│  │  → worker_group = actor_rollout_wg (same process) │                   │
│  │  → In-process: training engine fused with rollout │                   │
│  │  → Mode: TRAINER or ROLLOUT (switchable!)         │                   │
│  │  → CheckpointEngineManager (naive backend)        │                   │
│  │     → sync via in-process update_weights()         │                   │
│  └────────────────────────────────────────────────────┘                   │
│                                                                         │
│  ┌─────────────────── STANDALONE ───────────────────┐                   │
│  │  LLMServerManager (start_rank=N, separate GPUs)   │                   │
│  │  → worker_group = None (disaggregated)            │                   │
│  │  → Separate process: dedicated rollout GPUs        │                   │
│  │  → Always in ROLLOUT mode (never switches!)        │                   │
│  │  → CheckpointEngineManager (nccl/nixl/mooncake)   │                   │
│  │     → sync via NCCL/NIXL/Mooncake weight transfer  │                   │
│  └────────────────────────────────────────────────────┘                   │
│                                                                         │
│  ★★★ Global Load Balancer: routes requests to BOTH sets of replicas     │
│     → Hybrid replicas can be added/removed dynamically                   │
│     → Standalone replicas always serve generation requests               │
└─────────────────────────────────────────────────────────────────────────┘
```

**How it works**:

1. `self.llm_server_manager` — HYBRID: initialized in `super()._setup()` with `worker_group=self.actor_rollout_wg`
   - Replicas start at rank 0
   - Uses naive checkpoint engine (in-process weight sync)
   - Can switch between TRAINER and ROLLOUT modes

2. `self.standalone_server_manager` — STANDALONE: initialized in `PPOTrainerSeparateAsync._setup()` with `start_rank=hybrid_num_replicas`
   - Replicas start at rank `N` (hybrid_num_replicas) — avoids Ray named-actor collisions!
   - Uses nccl/nixl/mooncake checkpoint engine (cross-process weight transfer)
   - Always in ROLLOUT mode (always generating)

3. `self.standalone_checkpoint_manager` — connects actor_rollout_wg to standalone replicas via NCCL/NIXL/Mooncake
   - Periodic weight sync every `parameter_sync_step` steps

### 2.3 Switch Strategy: Current State and Future

★★★★★★★★★ The switch mechanism is a DUAL-MODE dynamic resource allocation system:

**Current behavior** (should_switch_to_rollout returns False):

```
Training step timeline:

  on_sample_begin:
    → should_switch_to_rollout() = False
    → Hybrid stays in TRAINER mode
    → Only standalone replicas generate

  on_sample_end:
    → If hybrid was in ROLLOUT mode → switch_to_trainer()
    → But currently hybrid is ALWAYS in TRAINER mode → switch never triggers

  Result: Hybrid NEVER generates. Only standalone replicas do rollout.
```

**Future behavior** (when switch strategy is implemented):

```
Training step timeline (future, when should_switch_to_rollout = True sometimes):

  on_sample_begin:
    → should_switch_to_rollout() checks replay buffer fill level + switch overhead
    → If True: switch_to_rollout() → hybrid becomes rollout → adds to load balancer
    → If False: hybrid stays trainer → only standalone generates

  on_sample_end:
    → If hybrid was in ROLLOUT mode → switch_to_trainer()
    → Remove from balancer → abort → sleep

  Result: Hybrid DYNAMICALLY joins/leaves generation pool based on need.
```

★★★★★★★★★ **Switch strategy TODO** (from PR body and code comments):

The switch strategy should consider:
1. **Replay buffer fill level**: If the buffer has enough samples, trainer should stay in TRAINER mode. If buffer is empty/starving, trainer should switch to ROLLOUT to help generate more data.
2. **Switch overhead**: Sleep/wake cycle costs time. If overhead > benefit of additional generation, don't switch.
3. **Offload strategy**: When switching to rollout mode, should the trainer offload optimizer states/params to CPU? When switching back to trainer mode, should it reload? Currently this is a TODO in both `switch_to_rollout()` and `switch_to_trainer()`.

### 2.4 start_rank: Avoiding Ray Named-Actor Collisions

★★★★★★★★★ This is a subtle but important detail:

When both hybrid and standalone replicas coexist, they each create Ray named actors for their server processes. If both start at rank 0, the second set would collide with the first.

```python
# trainer_separate_async.py:68-71
hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
    config=self.config, start_rank=hybrid_num_replicas
)
```

Example: If hybrid has 2 replicas (ranks 0, 1), standalone starts at rank 2 (ranks 2, 3, ...).

```python
# llm_server.py:414-416 (in _initialize_llm_servers)
self.rollout_replicas = [
    self.rollout_replica_class(
        replica_rank=start_rank + replica_rank,  # ← offset by start_rank!
        ...
    )
    for replica_rank in range(num_replicas)
]
```

The `start_rank` flows from `LLMServerManager.__init__` → `_initialize_llm_servers` → `RolloutReplica(replica_rank=...)`.

This is the same pattern used by `FullyAsyncLLMServerManager` (a subclass referenced in the old `_initialize_llm_servers` docstring).

---

## 3. Three Trainer Variants Comparison

### 3.1 V1 Trainer Family

| Trainer | Registry Name | Key Feature | Weight Sync | Generation Mode | Rollout Interruption |
|---------|--------------|-------------|-------------|----------------|---------------------|
| **PPOTrainerSync** | `"sync"` | Synchronous, colocated, no partial rollout | naive (in-process) | sleep → update → wake | NO (full generation before training) |
| **PPOTrainerColocateAsync** | `"colocate_async"` | Async, colocated, partial rollout | naive (in-process) | abort → sleep → update → resume | YES (abort ongoing requests, resume later) |
| **PPOTrainerSeparateAsync** | `"separate_async"` | Async, separate rollout, trainer may switch | naive + nccl/nixl/mooncake | Hybrid: switchable; Standalone: always generating | YES (abort + resume on hybrid; standalone always running) |

### 3.2 Lifecycle Hooks Comparison

| Hook | Sync | ColocateAsync | SeparateAsync |
|------|------|---------------|---------------|
| `on_init_end` | update_weights | update_weights | update_weights (both managers) |
| `on_train_begin` | — | warmup batches | warmup batches |
| `on_validate_begin` | — | — | switch hybrid→rollout if in TRAINER |
| `on_sample_begin` | — | — | switch hybrid→rollout if needed |
| `on_sample_end` | sleep_replicas | abort + sleep | switch hybrid→trainer |
| `on_step_end` | update_weights | update_weights + resume | periodic standalone weight sync |
| `on_validate_end` | — | — | removed (switch in on_sample_end) |

### 3.3 Config Requirements

```yaml
# separate_async config (from ppo_trainer.yaml:218-225)
trainer:
  v1:
    trainer_mode: separate_async
    separate_async:
      num_warmup_batches: 4        # More warmup than colocate_async (1)
      parameter_sync_step: 4       # Sync standalone weights every 4 steps

# Mandatory requirements (from trainer_separate_async.py:43-56):
# 1. train_batch_size == ppo_mini_batch_size (no mini-batch splitting in separate mode)
# 2. rollout.nnodes > 0 (standalone replicas need their own nodes!)
# 3. rollout.n_gpus_per_node > 0 (standalone needs dedicated GPUs)
# 4. checkpoint_engine.backend != "naive" (must use nccl/nixl/mooncake for standalone sync)
```

★★★★★★★★★ **Requirement #4 is CRITICAL**: The standalone checkpoint engine MUST use nccl/nixl/mooncake (NOT naive). Naive only works for in-process sync. The hybrid manager still uses naive internally (in-process), but the standalone manager requires cross-process weight transfer.

---

## 4. Global Load Balancer Integration

### 4.1 How Hybrid Replicas Join/Leave the Load Balancer

★★★★★★★★★ The separate_async trainer dynamically manages which replicas serve generation requests:

```python
# switch_to_rollout (lines 121-126):
def switch_to_rollout(self):
    # TODO: disable auto offload in config and offload according to the switch strategy
    self.checkpoint_manager.update_weights(self.global_steps)       # Sync weights
    self.checkpoint_manager.resume_generation_replicas()             # Resume generation
    self.add_replicas_to_balancer()                                  # ★ Add to standalone's LB
    self.current_mode = HybridEngineMode.ROLLOUT

# switch_to_trainer (lines 128-133):
def switch_to_trainer(self):
    # TODO: disable auto offload in config and offload according to the switch strategy
    self.remove_replicas_from_balancer()                             # ★ Remove from standalone's LB
    self.checkpoint_manager.abort_replicas()                         # Abort in-flight requests
    self.checkpoint_manager.sleep_replicas()                         # Sleep (free GPU memory)
    self.current_mode = HybridEngineMode.TRAINER
```

### 4.2 add_replicas_to_balancer / remove_replicas_from_balancer

```python
# lines 135-144:
def add_replicas_to_balancer(self):
    global_load_balancer = self.standalone_server_manager.global_load_balancer
    servers = dict(
        zip(self.llm_server_manager.server_addresses,
            self.llm_server_manager.server_handles, strict=True)
    )
    ray.get(global_load_balancer.add_servers.remote(servers))

def remove_replicas_from_balancer(self):
    global_load_balancer = self.standalone_server_manager.global_load_balancer
    ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))
```

★★★★★★★★★ **Key insight**: The hybrid replicas JOIN the standalone load balancer — not their own! This means when hybrid is in ROLLOUT mode, its replicas are served alongside standalone replicas, and `FullyAsyncLLMServerClient` can route requests to either set.

### 4.3 GlobalRequestLoadBalancer Features

**File**: `verl/workers/rollout/llm_server.py:46-146`

The load balancer supports:
- **Sticky session**: Maps `request_id → server_id` via LRU cache (multi-turn conversations → same server)
- **Least-loaded selection**: When no sticky session, selects server with fewest in-flight requests
- **Dynamic add/remove**: `add_servers()` and `remove_servers()` for elastic scaling
- **Atomic acquire/release**: Single Ray RPC for acquire (server_id + handle), fire-and-forget for release

This is the SAME load balancer used by `FullyAsyncLLMServerClient` for abort-resume generation.

---

## 5. Weight Sync: Dual CheckpointEngineManager Architecture

### 5.1 Two CheckpointEngineManagers

★★★★★★★★★ The separate_async trainer has TWO checkpoint managers:

```
Manager 1: self.checkpoint_manager (HYBRID)
  → backend = "naive" (always, set in trainer_base.py:269)
  → actor_wg = self.actor_rollout_wg
  → replicas = self.llm_server_manager.get_replicas()
  → Method: in-process update_weights() (ColocatedCheckpointEngine)

Manager 2: self.standalone_checkpoint_manager (STANDALONE)
  → backend = nccl/nixl/mooncake (from config)
  → actor_wg = self.actor_rollout_wg
  → replicas = self.standalone_server_manager.get_replicas()
  → Method: NCCL/NIXL/Mooncake CUDA IPC weight transfer
```

**Why two managers?**: The hybrid replicas are in the same process as the training engine — they sync via naive (in-process) mechanism. The standalone replicas are in separate processes on separate GPUs — they need cross-process CUDA IPC (NCCL/NIXL/Mooncake).

### 5.2 Weight Sync Flow in on_step_end

```python
# trainer_separate_async.py:115-119:
def on_step_end(self):
    if self.global_steps % self.config.trainer.v1.separate_async.parameter_sync_step == 0:
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.standalone_checkpoint_manager.update_weights(self.global_steps)
```

★★★★★★★★★ **parameter_sync_step=4 default**: Standalone replicas are updated every 4 steps (not every step). This means standalone replicas run on stale weights for up to 4 training steps. The ReplayBuffer handles staleness via `max_off_policy_threshold=8` and `max_off_policy_strategy=drop`.

### 5.3 CheckpointEngineManager.update_weights() Full Flow

**File**: `verl/checkpoint_engine/base.py:469-515`

When backend is NOT "naive":
```
1. abort_replicas()            — abort all in-flight requests (partial rollout)
2. create RayWorkerGroup        — temporary worker group for all replica workers
3. release_kv_cache_replicas()  — free KV cache, keep weights in place
4. build_process_group()        — NCCL/NIXL/Mooncake communication setup
5. update_weights()             — CUDA IPC weight transfer
6. finalize()                   — cleanup communication
7. resume_kv_cache_replicas()   — restore KV cache
8. resume_generation_replicas() — resume aborted requests
```

★★★★★★★★★ **KV cache preservation**: The `release_kv_cache` + `resume_kv_cache` pattern means weights are written directly into existing weight buffers without full sleep/wake. This is similar to the in-process update path but via CUDA IPC.

---

## 6. June 18 Four-PR Architectural Shift

### 6.1 The Four MERGED PRs

| PR | Title | Author | +/− | Files | Merged |
|-----|-------|--------|------|-------|--------|
| **#6790** | [trainer] feat: A runnable separate async trainer | Begunner | +17/-16 | 2 | June 17 14:25 |
| **#6791** | [megatron,doc] feat: support DSV4/GLM5/KimiK2.5 via Megatron Lite | ISEEKYAN | +864/-9 | 9 | June 18 00:12 |
| **#6765** | [worker] feat: add per-step optimizer param overrides | Luosuu | +110/-8 | 2 | June 18 00:25 |
| **#6789** | [data] fix: offload process_vision_info to thread executor | huaiyizhao | +15/-1 | 1 | June 17 14:30 |

### 6.2 How They Form a Coordinated Architectural Shift

★★★★★★★★★ These four PRs form a COMPLEMENTARY set that enables the **full async training architecture**:

```
#6790 (separate_async trainer):
  → Core infrastructure: separate trainer + rollout workers
  → Enables multi-GPU async training with dynamic resource allocation
  → Switch strategies (TODO) will enable hybrid trainer to join rollout when idle

#6791 (Megatron Lite DSV4/GLM5/KimiK2.5):
  → Model support: enables training DSV4, GLM5, KimiK2.5 on separate infrastructure
  → Megatron Lite = experimental, agent-friendly path → plugs into verl via external mlite
  → 256-GPU configs with PP8/EP8/CP8 → multi-node training

#6765 (per-step optimizer overrides):
  → Training control: enables per-step LR/betas/eps/weight_decay overrides
  → Critical for Muon LR scheduling (#6765 enables #6731 CPPO's Muon)
  → OptimStepParams scoped to Tinker worker → explicit optimizer stepping

#6789 (vision thread executor):
  → Concurrency: fixes event loop blocking in agent-loop multimodal processing
  → CPU-bound vision processing (PNG decode + smart_resize) offloaded to thread
  → Unblocks asyncio.gather for batch agent loops → enables agentic rollout (#6779)
```

**The coordinated shift**: Together, these enable:
1. **Separate training infrastructure** (#6790) — trainer and rollout on different GPUs
2. **New model support** (#6791) — DSV4/GLM5/KimiK2.5 training paths
3. **Per-step optimizer control** (#6765) — fine-grained Muon/Adam scheduling
4. **Async agent-loop concurrency** (#6789) — non-blocking multimodal processing

This is the architecture for **fully disaggregated RLHF training** at scale.

### 6.3 Connection to RolloutMode Architecture

The separate_async trainer maps to the existing RolloutMode architecture:

| RolloutMode | Trainer Variant | Process Layout | Weight Sync |
|-------------|----------------|---------------|-------------|
| **HYBRID** | sync, colocate_async | Same process (training + rollout fused) | naive (in-process) |
| **COLOCATED** | (not yet a v1 trainer) | Same placement group, separate process | NCCL/NIXL/Mooncake |
| **STANDALONE** | separate_async (standalone_server_manager) | Separate GPUs, separate process | NCCL/NIXL/Mooncake |

★★★★★★★★★ **The separate_async trainer uses BOTH HYBRID and STANDALONE modes simultaneously**: The `llm_server_manager` creates HYBRID replicas (in-process with training), and the `standalone_server_manager` creates STANDALONE replicas (disaggregated).

This is a novel combination not seen in the legacy (v0) trainer architecture.

---

## 7. Comparison: verl Separate Async vs rLLM Tinker

### 7.1 Architecture Comparison

| Feature | verl SeparateAsync (#6790) | rLLM Tinker |
|---------|---------------------------|-------------|
| **Trainer-rollout communication** | Ray actors + NCCL/NIXL/Mooncake | Client-server SDK |
| **Weight sync** | CUDA IPC (NCCL/NIXL/Mooncake) | Checkpoint → new SamplingClient (zero-copy) |
| **Generation coordination** | GlobalLoadBalancer (sticky + least-loaded) | SyncCoordinator (asyncio, 172 lines) |
| **Off-policy staleness handling** | ReplayBuffer + max_off_policy_threshold | TrajectoryGroupBuffer + NVMe offload |
| **Dynamic resource allocation** | add/remove replicas from balancer | No dynamic allocation (fixed backend) |
| **Switch strategy** | should_switch_to_rollout (TODO, currently False) | No switch (trainer never becomes rollout) |
| **Serving backend** | vLLM/SGLang/TRT-LLM (RolloutReplica registry) | Tinker (in-process) + vLLM/SGLang (external) |
| **bypass_mode** | Forced True (Decoupled PPO TODO) | Required (bypass_mode=True in RTX 4090 config) |
| **parameter_sync_step** | Every 4 steps (configurable) | Single-step (no staleness) |
| **Partial rollout** | Yes (abort + resume) | No (full generation per step) |
| **Agent loop** | Yes (AgentLoopManagerTQ) | No (simple generate → compute reward) |

### 7.2 Key Architectural Differences

★★★★★★★★★ **verl has dynamic load balancing**: The GlobalRequestLoadBalancer can add/remove hybrid replicas at runtime. rLLM Tinker has fixed backend allocation — no dynamic switching.

★★★★★★★★★ **rLLM has simpler weight sync**: Tinker uses `save_weights → new SamplingClient → set_sampling_client` — zero-copy in-process. verl's separate_async uses NCCL/NIXL/Mooncake — cross-process CUDA IPC.

★★★★★★★★★ **verl has partial rollout**: The abort + resume mechanism allows ongoing generation requests to be interrupted mid-stream and resumed after weight sync. rLLM completes all generation before training.

★★★★★★★★★ **verl is multi-node**: separate_async requires nnodes>0 and n_gpus>0 for standalone replicas. rLLM Tinker is single-GPU by default.

---

## 8. RTX 4090 Impact Analysis

### 8.1 Why Separate Async is NOT Viable on RTX 4090

★★★★★★★★★ **CRITICAL**: The separate_async trainer requires:
- `rollout.nnodes > 0` — standalone replicas need SEPARATE nodes/GPUs
- `rollout.n_gpus_per_node > 0` — standalone needs dedicated GPU allocation
- `checkpoint_engine.backend != "naive"` — must use cross-process weight transfer

On a single RTX 4090 (24 GiB), you CANNOT have both:
1. A hybrid engine (training + rollout in same GPU)
2. A standalone engine (separate GPUs for dedicated rollout)

There are not enough GPUs. This mode is designed for multi-GPU/multi-node setups.

### 8.2 RTX 4090 GRPO Training Mode Ranking (Updated)

| Rank | Mode | Trainer | Viability on RTX 4090 |
|------|------|---------|----------------------|
| **#1** | HYBRID colocate_async + bypass | PPOTrainerColocateAsync | VIABLE — best throughput |
| **#1.5** | Tinker in-process | rLLM TinkerBackend | VIABLE — simpler architecture |
| **#2** | HYBRID sync + bypass | PPOTrainerSync | VIABLE — simpler but no partial rollout |
| **NOT VIABLE** | separate_async | PPOTrainerSeparateAsync | NOT VIABLE — requires separate GPUs |
| **NOT VIABLE** | STANDALONE only | (no v1 trainer variant yet) | NOT VIABLE — no training on single GPU |

### 8.3 When Separate Async Becomes Relevant

★★★★★★★★★ **Future RTX 4090 relevance**: If the user has access to a second GPU (e.g., university GPU + matpool GPU), the separate_async architecture could enable:
1. Training on GPU 1 (RTX 4090) — hybrid mode
2. Rollout on GPU 2 — standalone mode
3. Dynamic switching — when training is idle, GPU 1 joins generation pool

But currently both external GPUs are offline (per MEMORY), so this remains theoretical.

### 8.4 Switch Strategy RTX 4090 Implications

If a switch strategy is implemented in the future:

```
★★★★★★★★★ Potential switch strategy for multi-GPU setups:

  GPU 1 (RTX 4090) — HYBRID mode:
    → Default: TRAINER mode (training + weight updates)
    → When replay buffer starved: switch to ROLLOUT (add to balancer, generate)
    → When buffer full: switch back to TRAINER (remove from balancer, train)

  GPU 2 (any GPU) — STANDALONE mode:
    → Always ROLLOUT mode (always generating)
    → Weight sync every parameter_sync_step=4 steps

  Offload strategy:
    → When GPU 1 switches to ROLLOUT: offload optimizer states to CPU (free GPU for KV cache)
    → When GPU 1 switches to TRAINER: reload optimizer states from CPU
    → This is similar to the existing HYBRID sleep/wake but across mode switches
```

---

## 9. MUST DO / MUST NOT

### 9.1 MUST DO (for separate_async, multi-GPU setups)

- Use `checkpoint_engine.backend = "nccl"` or `"nixl"` or `"mooncake"` for standalone — naive WILL NOT WORK
- Set `parameter_sync_step >= 1` — controls how often standalone replicas get weight updates
- Set `bypass_mode = True` — Decoupled PPO not yet supported in separate_async
- Set `num_warmup_batches >= 4` — replay buffer needs initial data before training starts
- Ensure standalone replicas have sufficient GPU resources (`nnodes > 0`, `n_gpus_per_node > 0`)
- Monitor trajectory staleness metrics: `training/off_policy/trajectory_staleness/mean`

### 9.2 MUST NOT

- Do NOT use `separate_async` on single-GPU RTX 4090 — requires separate GPUs for standalone replicas
- Do NOT use `checkpoint_engine.backend = "naive"` for standalone — requires cross-process sync
- Do NOT set `train_batch_size != ppo_mini_batch_size` — assertion will fail
- Do NOT assume hybrid replicas will switch to rollout — `should_switch_to_rollout()` currently returns False
- Do NOT use `Decoupled PPO` with separate_async — bypass_mode forced True, IS correction not implemented
- Do NOT use `vLLM-Ascend` — sleep_level=1 NOT supported, would break the sleep/wake cycle for hybrid replicas

---

## 10. Future Directions: What #6790 Enables

### 10.1 Switch Strategies (explicitly stated as TODO)

The PR body says: **"Switch and offload strategies will be introduced later."**

Potential switch strategies:
1. **Buffer-based**: Switch to rollout when replay buffer fill level < threshold
2. **Overhead-aware**: Only switch if expected generation throughput > switch overhead
3. **Hybrid adaptive**: Mix of buffer level + overhead calculation
4. **Priority-based**: Switch based on generation request queue depth

### 10.2 Offload Strategies (explicitly stated as TODO)

Both `switch_to_rollout` and `switch_to_trainer` have the same TODO comment:

```python
# TODO: disable auto offload in config and offload according to the switch strategy
```

This suggests:
- Currently auto-offload is handled by the training config (param_offload, grad_offload)
- Future: offload should be CONTROLLED BY THE SWITCH STRATEGY — offload optimizer states when switching to rollout, reload when switching back to trainer
- This connects to the existing HYBRID sleep/wake architecture (sleep_level=1/2)

### 10.3 Connection to verl #6794 (Delta Weight Sync)

★★★★★★★★★ **#6794 delta weight sync + #6790 separate_async = COMPLEMENTARY**:

- #6794 reduces standalone weight sync payload by ~100x (bytewise diff)
- #6790 enables standalone weight sync via NCCL/NIXL/Mooncake
- Combined: standalone replicas get weight updates via ~100x smaller payload every `parameter_sync_step` steps
- BUT: #6794 is currently SGLang-only and deferred for LoRA path

### 10.4 Connection to verl #6512 (Per-Unit LoRA Summon)

★★★★★★★★★ **#6512 per-unit summon + #6790 separate_async = COMPLEMENTARY**:

- #6512 enables per-unit LoRA summon (10x peak memory reduction, 60→6-8 GiB)
- #6790 enables separate async trainer (hybrid + standalone replicas)
- Combined: hybrid replicas can use per-unit summon during training, standalone replicas use LoRA adapter for generation
- This would be the optimal path for multi-GPU LoRA GRPO training

### 10.5 Connection to RolloutMode.COLOCATED

★★★★★★★★★ **There is currently NO v1 trainer variant for COLOCATED mode**:

The existing RolloutMode enum has 3 modes (HYBRID, COLOCATED, STANDALONE), but the v1 trainer registry only has:
- `sync` → HYBRID
- `colocate_async` → HYBRID (with async features)
- `separate_async` → HYBRID + STANDALONE (dual)

A COLOCATED-only trainer variant (separate process, same placement group, no weight sync needed) is NOT yet implemented. This would be useful for GRM (LLM as a judge) scenarios where the rollout engine stays in a separate process but shares the same GPU.

---

## 11. Source Files Read

| File | Key Lines | Purpose |
|------|-----------|---------|
| `verl/trainer/ppo/v1/trainer_separate_async.py` | 1-149 (full) | ★★★ Separate async trainer: dual LLMServerManager, switch strategy, HybridEngineMode |
| `verl/trainer/ppo/v1/trainer_base.py` | 1-1507 (full) | PPOTrainer ABC, step() 10-step data flow, _setup() initialization |
| `verl/trainer/ppo/v1/trainer_sync.py` | 1-43 (full) | Sync trainer variant |
| `verl/trainer/ppo/v1/trainer_colocate_async.py` | 1-58 (full) | Colocate async variant |
| `verl/workers/rollout/llm_server.py` | 1-484 (full) | LLMServerManager, GlobalRequestLoadBalancer, LLMServerClient, FullyAsyncLLMServerClient |
| `verl/checkpoint_engine/base.py` | 1-586 (full) | CheckpointEngine ABC, CheckpointEngineManager, naive backend, weight sync flow |
| `verl/trainer/ppo/v1/replay_buffer.py` | 1-237 (full) | ReplayBuffer, TransferQueue KV storage, off-policy staleness management |
| `verl/trainer/config/ppo_trainer.yaml` | 1-471 (full) | separate_async config: num_warmup_batches=4, parameter_sync_step=4 |

## 12. Related PRs and Notes

| Item | Connection |
|------|-----------|
| **verl #6791** (MERGED June 18) | DSV4/GLM5/KimiK2.5 via Megatron Lite — multi-node model support for separate_async |
| **verl #6765** (MERGED June 18) | Per-step optimizer overrides — enables fine-grained LR/Muon scheduling in separate_async |
| **verl #6789** (MERGED June 18) | Vision thread executor — fixes event loop blocking, enables agentic rollout |
| **verl #6794** (OPEN) | Delta weight sync — ~100x payload reduction for standalone weight updates |
| **verl #6512** (OPEN) | Per-unit LoRA summon — 10x memory reduction for hybrid training |
| **verl #6699** (MERGED) | Detach memory fix — 4x reduction (FSDP only, other backends unfixed) |
| **verl #6782** (OPEN) | LoRA rank=64 breaks EOS — MUST use rank=32/alpha=64 |
| **verl #6731** (OPEN) | CPPO — bypass_mode mandatory, connects to #6765 optimizer overrides |
| **rLLM SyncCoordinator** | 172 lines pure asyncio — simpler than verl GlobalLoadBalancer |
| **rLLM TinkerBackend** | Client-server SDK — simpler weight sync than NCCL/NIXL/Mooncake |
| **verl HYBRID sleep/wake** | sleep_level=1/2 — the offload strategy for hybrid replicas |

---

## Key Findings Summary

★★★★★★★★★ #6790 removes NotImplementedError — separate_async trainer is NOW RUNNABLE (but requires multi-GPU)
★★★★★★★★★ Dual LLMServerManager architecture: hybrid (in-process) + standalone (disaggregated) on separate GPUs
★★★★★★★★★ `start_rank` avoids Ray named-actor collisions between hybrid and standalone replicas
★★★★★★★★★ `should_switch_to_rollout()` returns False — hybrid trainer NEVER switches to rollout currently
★★★★★★★★★ Switch + offload strategies explicitly TODO — future PRs will implement dynamic mode switching
★★★★★★★★★ bypass_mode FORCED True — Decoupled PPO not supported in separate async yet
★★★★★★★★★ parameter_sync_step=4 default — standalone replicas updated every 4 steps (configurable staleness)
★★★★★★★★★ Four June-18 PRs (#6790/#6791/#6765/#6789) form coordinated disaggregated RLHF architecture
★★★★★★★★★ RTX 4090: separate_async NOT viable on single GPU — requires nnodes>0 + n_gpus>0
★★★★★★★★★ Future: switch strategy could enable hybrid trainer to join rollout pool when replay buffer starved
★★★★★★★★★ Offload strategy: optimizer state offload when switching modes — similar to HYBRID sleep/wake
