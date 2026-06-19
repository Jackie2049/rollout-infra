# verl V1 Unified Trainer Architecture Deep Reading

> 2026-06-19 | Source-level analysis of verl V1 trainer, checkpoint engines, GRPO advantage, CPPO/bypass_mode
> ★★★★★★★★ V1 = 3 trainer types via @register_trainer decorator + 6 checkpoint engines via @CheckpointEngineRegistry.register
> ★★★★★★★★ RTX 4090 BEST = sync + naive + FSDP + bypass_mode=True + ppo_clip
> ★★★★★★★★ separate_async FORCES bypass_mode=True but requires multi-GPU → NOT viable on RTX 4090 dp=1

---

## 1. V1 Trainer Registration Pattern

**Location**: `verl/trainer/ppo/v1/trainer_base.py` (1506 lines)

Module-level decorator + registry pattern:

```python
TRAINER_REGISTRY: dict[str, type[PPOTrainer]] = {}

def register_trainer(name: str):
    """Class decorator that registers a PPOTrainer subclass under name."""
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator

def get_trainer_cls(name: str) -> type[PPOTrainer]:
    return TRAINER_REGISTRY[name]
```

Base class `PPOTrainer` is ABC with abstract methods `on_step_end()` and `on_sample_end()`. Each registered subclass only overrides **lifecycle hooks** to inject weight-sync and sleep/wake behavior.

### 3 Registered Trainer Types

| Name | Class | Key Behavior | RTX 4090 |
|------|-------|-------------|----------|
| `"sync"` | `PPOTrainerSync` | Colocated, NO partial rollout. Sleep after sampling, wake+sync after training | ★★★★★★★★ BEST |
| `"colocate_async"` | `PPOTrainerColocateAsync` | Partial rollout ENABLED. FullyAsyncLLMServerClient. Warmup batches | NOT viable single GPU |
| `"separate_async"` | `PPOTrainerSeparateAsync` | Trainer+rollout separate processes. Forces bypass_mode=True. Non-naive checkpoint engine required | NOT viable dp=1 |

**★★★★★★★★ Key**: `fully_async_policy` in `verl/experimental/` is NOT in V1 registry — inherits from legacy, NOT PPOTrainer.

---

## 2. Checkpoint Engine Architecture (6 Backends)

**Registry**: `CheckpointEngineRegistry` in `verl/checkpoint_engine/base.py`

```
Actor side:                Rollout side:
 ME0  ME1  ...  MEn        Replica 0          Replica 1
  |    |         |         [0][1][2][3]        [0][1][2][3]
 CE0  CE1  ...  CEn        CE CE CE CE        CE CE CE CE
  |             |            ^   ^   ^           ^   ^
  +----(nccl/nixl/...)------+---+---+-----------+---+
```

### 6 Backend Comparison

| Backend | Transport | Memory | Platform | RTX 4090 |
|---------|-----------|--------|----------|----------|
| **naive** | In-process Python yield | 0 extra | Any | ★★★★★★★★ BEST — sync trainer uses this |
| **NCCL** | NCCL broadcast + ZeroMQ | 2*bucket_size | NVIDIA GPU | Good for multi-GPU |
| **HCCL** | HCCL broadcast + ZeroMQ | 2*bucket_size | Ascend NPU | NOT usable on RTX 4090 |
| **NIXL** | NIXL p2p RDMA + ZeroMQ | 2*bucket_size | GPU+RDMA | Requires RDMA NIC |
| **KIMI** | ParameterServer+distributed | H2DBucket | NVIDIA GPU | Complex, multi-GPU |
| **Mooncake** | Mooncake TransferEngine p2p RDMA | 2*bucket_size+4KB | GPU/NPU | Requires RDMA |

**★★★★★★★★ RTX 4090**: "naive" backend = zero IPC overhead. Sync trainer forces this.

### CheckpointEngineManager Lifecycle (non-naive)

1. `abort_replicas()` — abort in-flight requests (partial rollout)
2. `release_kv_cache_replicas()` — free KV cache memory
3. `build_process_group()` — NCCL/NIXL/etc. topology setup
4. `actor_wg.update_weights()` + `rollout.update_weights()` — actual transfer
5. `finalize()` — cleanup process groups, free buffers
6. `resume_kv_cache_replicas()` — restore KV cache allocation
7. `resume_generation_replicas()` — restart interrupted requests

---

## 3. GRPO Advantage Computation

**Location**: `verl/trainer/ppo/core_algos.py` (2487 lines)

### 14 Registered Advantage Estimators

GAE, GRPO, REINFORCE_PLUS_PLUS, REINFORCE_PLUS_PLUS_BASELINE, REMAX, RLOO, OPO, GRPO_PASSK, GPG, RLOO_VECTORIZED, GRPO_VECTORIZED, OPTIMAL_TOKEN_BASELINE, TIR_OPTIMAL_TOKEN_BASELINE, GDPO

### GRPO Advantage Path

```
scores = token_level_rewards.sum(dim=-1)   # per-response score
→ group by uid (prompt identity)
→ group_mean, group_std computation
→ singleton: mean=0, std=1 (REINFORCE degeneration)
→ advantage = (score - mean) / (std + ε)   [norm_adv_by_std_in_grpo=True]
→ OR: advantage = score - mean             [Dr.GRPO, norm_adv_by_std_in_grpo=False]
→ broadcast: advantage.unsqueeze(-1) * response_mask
```

### Key GRPO Variants

| Variant | Grouping | Singleton | RTX 4090 |
|---------|----------|-----------|----------|
| GRPO | uid | mean=0, std=1 | ★★★★★★★★ BEST |
| GRPO_VECTORIZED | uid (group_mean_std) | mean=0, std=1 | Efficient GPU |
| GRPO_PASSK | uid | best response only | pass@k optimization |
| GDPO | uid + per-dimension | decoupled normalization | Multi-objective |

---

## 4. CPPO Integration and bypass_mode

### bypass_mode Operating Modes

| Mode | Policies | old_log_prob Source | Computation |
|------|----------|--------------------|------------|
| **Decoupled** (bypass=False) | 3: pi_rollout, pi_old, pi_theta | Recomputed by actor forward | Full forward pass |
| **Bypass** (bypass=True) | 2: pi_rollout = pi_old, pi_theta | Directly from rollout_log_probs | Skip forward pass |

**★★★★★★★★ bypass_mode=True** = skip entire `_compute_old_log_prob` actor forward pass → 18Ψ→3.8Ψ memory reduction on RTX 4090

### Policy Loss Registry (11 types)

vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg, clip_cov, kl_cov, geo_mean, cispo, bypass_mode

**bypass_mode loss dispatch**:
- PPO-clip: `old_log_prob = rollout_log_prob`, ratio handles IS via clipping. NO additional IS weights.
- REINFORCE: explicit IS weights `w = pi_current / pi_rollout`

### CPPO (PR #6731, NOT in main branch)

Position-weighted token-level divergence mask:
- Decreasing `w_t` from 1.0 to `w_min`
- Cumulative prefix-average budget `delta_b`
- Only masking decision changes, surrogate objective unchanged
- 54.79% AIME on Qwen3-30B-A3B-Base without collapse
- Can be used WITH bypass_mode

---

## 5. V1 Advantage Computation Flow

```
TransferQueue → uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values
→ apply KL penalty to rewards (if use_kl_in_reward)
→ compute rollout correction (IS weights + rejection sampling) — SKIPPED if bypass_mode=True
→ compute_advantage_for_multi_trajectories() → delegates to GRPO for multi-trajectory agent loops
→ write advantages and returns back to TransferQueue as nested tensors
```

---

## 6. RTX 4090 Configuration

```yaml
trainer:
  v1:
    trainer_mode: sync                # ★★★★★★★★ MUST be sync for single GPU

actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: naive                   # forced by sync trainer, zero IPC overhead

algorithm:
  adv_estimator: grpo                  # or grpo_vectorized for efficiency
  norm_adv_by_std_in_grpo: true         # original GRPO normalization
  rollout_correction:
    bypass_mode: true                   # ★★★★★★★★ MUST: 18Ψ→3.8Ψ memory reduction
    loss_type: ppo_clip                 # bypass mode loss type

rollout:
  n: 4                                 # ★★★★★★★★ MUST >= 2 for GRPO! 4 optimal for RTX 4090
```

### Trainer Ranking for RTX 4090

1. **sync + naive + FSDP + bypass=True + ppo_clip** = ★★★★★★★★ BEST (GRPO + bypass + PPO-clip)
2. **sync + naive + FSDP + bypass=True + reinforce** = Alternative (explicit IS weights, higher variance)
3. **sync + naive + FSDP + bypass=False** = Standard GRPO (recompute old_log_prob, higher memory)
4. **colocate_async** = NOT viable single GPU
5. **separate_async** = NOT viable dp=1

---

## 7. Source File Map

| File | Lines | Key Content |
|------|-------|------------|
| `verl/trainer/ppo/v1/trainer_base.py` | 1506 | ABC base, TRAINER_REGISTRY, lifecycle, fit(), step(), _compute_advantage, _compute_old_log_prob |
| `verl/trainer/ppo/v1/trainer_sync.py` | ~30 | @register_trainer("sync"), sleep/wake hooks |
| `verl/trainer/ppo/v1/trainer_colocate_async.py` | ~40 | @register_trainer("colocate_async"), warmup |
| `verl/trainer/ppo/v1/trainer_separate_async.py` | 148 | @register_trainer("separate_async"), bypass forced |
| `verl/trainer/ppo/core_algos.py` | 2487 | GRPO advantage, 14 estimators, 11 policy losses, bypass_mode |
| `verl/checkpoint_engine/base.py` | ~350 | CheckpointEngine ABC, CheckpointEngineRegistry, Manager lifecycle |
| `verl/utils/groupwise.py` | ~210 | group_mean_std, as_torch_index, singleton degeneration |
| `verl/trainer/ppo/prefix_grouper_utils.py` | ~200 | PrefixGrouper, shared-prefix optimization |

---

## References

- verl V1 trainer: `verl/trainer/ppo/v1/`
- verl core_algos: `verl/trainer/ppo/core_algos.py`
- verl checkpoint engines: `verl/checkpoint_engine/`
- CPPO PR #6731: https://github.com/verl-project/verl/pull/6731
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- GDPO: https://arxiv.org/abs/2601.05242
- Cross-framework GRPO comparison: notebook/projects/cross-framework-grpo-advantage-comparison.md
