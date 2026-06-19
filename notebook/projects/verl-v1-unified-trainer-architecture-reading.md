# verl V1 Unified Trainer Architecture — Reading Note

**Created: 2026-06-19 | Source: verl-project/verl verl/trainer/ppo/v1/**

---

## Overview

★★★★★★★★★ verl is developing a **V1 unified trainer** that replaces the legacy `main_ppo.py` with a pluggable, registered trainer architecture. This is the most significant architectural change since the HYBRID rollout manager.

## Architecture

### Trainer Registry Pattern

```python
TRAINER_REGISTRY = {}

@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO — trainer+rollout colocated, no partial rollout"""

@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """Async PPO — trainer+rollout colocated, partial rollout enabled"""

@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """Async PPO — trainer+rollout separated, partial rollout enabled"""
```

Configuration selects trainer type: `config.trainer.v1.type = "sync"` etc.

### Lifecycle Hooks

All trainers inherit from `PPOTrainer` and override lifecycle hooks:

| Hook | Purpose | Sync | Colocated Async | Separate Async |
|------|---------|------|-----------------|----------------|
| `on_init_end` | Load checkpoint + update weights | ✓ update_weights | ✓ update_weights | ✓ update_weights |
| `on_train_begin` | Warmup | none | num_warmup_batches | num_warmup_batches |
| `on_step_end` | Update weights after training | update_weights | update_weights + resume_generation | update_weights + resume_generation |
| `on_sample_end` | Sleep/discard after rollout | sleep_replicas | abort + sleep_replicas | abort + sleep_replicas |

### RTX 4090 Best Path

★★★★★★★★★ **`trainer_sync`** is the RTX 4090 best path:

1. **Simple lifecycle**: update_weights → train → sleep_replicas → repeat
2. **No partial rollout**: avoids async complexity on single GPU
3. **Colocated**: trainer and rollout on same GPU (dp=1)
4. **Matches our existing config**: bypass_mode=True, ZeRO-2, FSDP backend

The `trainer_colocate_async` might also work on RTX 4090 for future exploration:
- Warmup batches pre-fill rollout queue
- Abort+sleep+resume cycle enables partial rollout
- BUT: single GPU resource contention between trainer and rollout

### Checkpoint Engine Manager

5 engines available:

| Engine | Backend | RTX 4090 |
|--------|---------|----------|
| nccl_checkpoint_engine | ZMQ + NCCL broadcast + CuPy | ★★★★★★★★ YES (standard CUDA) |
| hccl_checkpoint_engine | HCCL | NO (Ascend NPU only) |
| mooncake_checkpoint_engine | Mooncake | NO (multi-node RDMA) |
| nixl_checkpoint_engine | NIXL | NO (experimental) |
| kimi_checkpoint_engine | Kimi | NO (Moonshot proprietary) |

RTX 4090 uses `nccl_checkpoint_engine` — same as existing weight sync path.

### Checkpoint Engine Internals (NCCL)

★★★★★★★★★ NCCL checkpoint engine source code reveals the **actual weight sync mechanism**:

1. **ZMQ for metadata**: ZMQ pub/sub socket for control messages (weight names, shapes, dtypes, offsets)
2. **NCCL broadcast for weight data**: `ray.util.collective` NCCL broadcast for tensor data transfer
3. **CuPy (cupy) for GPU memory**: CuPy arrays for efficient GPU-to-GPU memory operations
4. **BroadcastOperation**: async broadcast in separate thread → non-blocking weight sync
5. **TensorMeta**: structured metadata for each weight tensor (name, shape, dtype, chunk_offset, chunk_size, offset)
6. **merge_weight_chunks / split_weight_chunks**: chunk-based weight transfer for efficient bandwidth use

Key classes:
- `MasterMetadata(zmq_ip, zmq_port)`: master node metadata for ZMQ communication
- `BroadcastOperation(rank, group_name, bucket, metadata, socket, topic)`: async NCCL broadcast
- `CheckpointEngineRegistry`: registry for checkpoint engines (same pattern as trainer registry)

★★★★★★★★★ This confirms our earlier weight sync analysis:
- ZMQ (not NCCL alone) is used for metadata → validates our IPC comparison
- NCCL broadcast for actual data → efficient on CUDA
- Async broadcast in executor thread → non-blocking
- dp=1 RTX 4090: NCCL broadcast = self-broadcast (identity) → no actual data transfer needed!

### Key New Components

1. **`AgentLoopManagerTQ`**: Transfer queue-based agent loop manager — uses `transfer_queue` library for async data movement
2. **`ReplayBuffer`**: V1 replay buffer for off-policy training (stale data reuse)
3. **`FullyAsyncLLMServerClient`**: Client for async rollout generation
4. **`CheckpointEngineManager`**: Manages weight sync, sleep/wake, and checkpoint persistence

---

## RTX 4090 Implications

★★★★★★★★★ **V1 trainer is the future of verl RTX 4090 GRPO training**

Current status (legacy `main_ppo.py`):
- Uses `main_ppo` with manual backend configuration
- V1 will replace with `config.trainer.v1.type = "sync"` → automatic

Migration path:
1. **Immediate**: Continue using legacy `main_ppo` (production-tested)
2. **Near-term**: Switch to V1 `trainer_sync` when available
3. **Future**: Explore `trainer_colocate_async` for throughput improvement

★★★★★★★★★ **V1 architecture validates our RTX 4090 config stack**:
- `trainer_sync` = HYBRID RolloutMode equivalent (colocated, sync weight sync)
- `nccl_checkpoint_engine` = matches our FSDP backend analysis
- `sleep_replicas()` / `update_weights()` lifecycle = matches our weight sync flow diagram

---

## Connection to Tracked Issues

| Issue | V1 Connection |
|-------|---------------|
| #6790 separate_async trainer | → V1 `trainer_separate_async` (NOW RUNNABLE but multi-GPU only) |
| #6699 detach leak | → V1 `CheckpointEngineManager` handles weight sync properly (FSDP only fixed) |
| #6512 per-unit LoRA summon | → V1 uses same FSDP unit discovery mechanism |
| #6468 FSDP2 CPU leak | → V1 `CheckpointEngineManager` must handle this correctly |
| #6794 delta weight sync | → V1 likely integrates delta sync in future |
| #6731 CPPO | → V1 can register CPPO as another trainer type via `register_trainer` |

★★★★★★★★★ **CPPO integration with V1**: `register_trainer("cppo")` → CPPO as a registered V1 trainer → matches our RTX 4090 #1 BEST approach!

---

## Latest Code Research (June 19)

### Registry Implementation Details

★★★★★★★★★ **`register_trainer` is a module-level dict + decorator** — NOT a class:

The registry pattern is a simple Python dictionary with a decorator function, similar to PyTorch's `@torch.compile` or vLLM's model registry. This means:
- Adding new trainer types is trivial: just write a class + decorate it
- No inheritance from a registry base class needed
- CPPO can be added as a 4th type with a single `@register_trainer("cppo")` decorator
- Registry lives at module scope in the trainer package, not in a separate registry module

### V1 Trainer Source Location

★★★★★★★★★ **V1 trainer lives at `verl/trainer/ppo/v1/`** — NOT `verl/trainer/v1/`:

- The V1 trainer is under the `ppo` subdirectory, not a separate `v1` trainer directory
- This means V1 is specifically a PPO/GRPO trainer upgrade, not a general trainer overhaul
- Path: `verl/trainer/ppo/v1/trainer_base.py`, `verl/trainer/ppo/v1/trainer_sync.py`, etc.

★★★★★★★★★ **`trainer_base.py` = 71KB** — this is the core V1 file containing `PPOTrainer` ABC:

- 71 kilobytes of source code → massive core abstraction
- Contains the full `PPOTrainer` abstract base class with all lifecycle hooks
- Contains `ReplayBuffer`, `CheckpointEngineManager` integration, weight sync orchestration
- This is the single most important file for understanding V1 architecture

### 3 Registered Trainer Types Confirmed

★★★★★★★★★ The 3 registered trainer types are definitively confirmed:

1. **`sync`** — `PPOTrainerSync`: synchronous, colocated, no partial rollout
2. **`colocate_async`** — `PPOTrainerColocateAsync`: async, colocated, partial rollout enabled
3. **`separate_async`** — `PPOTrainerSeparateAsync`: async, separated, partial rollout enabled

★★★ **`fully_async_policy` is NOT part of V1 registry** — it lives in `verl/experimental/`:
- This is an experimental feature, not a production V1 trainer type
- It is NOT registered via `@register_trainer`
- Should be considered unstable and not for RTX 4090 deployment

### 6 Checkpoint Engine Backends

★★★★★★★★★ V1 now has **6 checkpoint engine backends** (expanded from 5 in initial reading):

| # | Engine | Backend | RTX 4090 |
|---|--------|---------|----------|
| 1 | naive_checkpoint_engine | Direct memory copy | ★★★★★★★★ YES (simplest path, dp=1) |
| 2 | nccl_checkpoint_engine | ZMQ + NCCL + CuPy | ★★★★★★★★ YES (standard CUDA) |
| 3 | hccl_checkpoint_engine | HCCL | NO (Ascend NPU only) |
| 4 | nixl_checkpoint_engine | NIXL | NO (experimental, multi-node) |
| 5 | kimi_checkpoint_engine | Kimi | NO (Moonshot proprietary) |
| 6 | mooncake_checkpoint_engine | Mooncake | NO (multi-node RDMA) |

★★★ **naive_checkpoint_engine** added: simplest path, direct memory copy. For dp=1 RTX 4090 this may be even more efficient than NCCL (no broadcast overhead, just in-process memcpy). Worth benchmarking naive vs nccl on single GPU.

### ReplayBuffer Implementation

★★★★★★★★★ **ReplayBuffer uses `TransferQueue` as KV store with GRPO group sampling**:

- TransferQueue (TQ) is the underlying data structure for ReplayBuffer
- TQ acts as a key-value store: episode IDs → trajectory data
- GRPO group sampling: samples entire groups (prompts + all completions) together
- This ensures GRPO reward normalization uses the correct group, not mixed groups
- Off-policy capability: stale data from previous steps can be mixed in (with discount)
- Critical for CPPO: CPPO needs off-policy data from previous policy versions

### Entry Point and Config

★★★★★★★★★ **V1 entry point flow** confirmed:

```
main_ppo.py → TaskRunnerV1 → get_trainer_cls(TRAINER_REGISTRY[config.trainer.v1.type])
→ trainer.init() → trainer.fit()
```

★★★★★★★★★ **Default configuration**:

- `use_v1=false` — V1 is opt-in, legacy `main_ppo` remains default
- `trainer.v1.trainer_mode=sync` — when V1 is enabled, sync is the default trainer mode
- This means: safe migration path, no forced V1 upgrade, gradual adoption

★★★★★★★★★ **trainer_sync confirmed as ONLY viable V1 path for RTX 4090 dp=1**:

| Trainer Type | dp=1 RTX 4090 | Reason |
|--------------|---------------|--------|
| sync | ★★★★★★★★ YES | Simple lifecycle, colocated, no partial rollout needed |
| colocate_async | ⚠ MAYBE | Resource contention on single GPU, warmup batches eat memory |
| separate_async | ✗ NO | Multi-GPU/multi-node ONLY (#6790 confirmed), NOT viable dp=1 |
| fully_async_policy | ✗ NO | Experimental, not in V1 registry, unstable |

### #6794 Critical Review Issues

★★★★★★★★★ **#6794 delta weight sync has 4 critical review issues from gemini-code-assist**:

The delta sync PR (critical for RTX 4090 memory optimization) has 4 flagged issues:
1. LoRA deferred weight handling — incomplete edge cases
2. SGLang-only scope — vLLM backend not supported (delta sync only works with SGLang)
3. Weight shape mismatch risk — delta weights must match exact target shapes
4. Memory lifecycle — when to free delta weight buffers

★★★ These issues are **2 CRITICAL** severity, meaning #6794 may need significant rework before merge. This delays RTX 4090 delta sync optimization.

### CPPO as 4th Trainer Type

★★★★★★★★★ **CPPO can be added as 4th trainer type via `@register_trainer("cppo")`**:

Implementation path:
1. Create `PPOTrainerCPPO` inheriting from `PPOTrainer` in `verl/trainer/ppo/v1/`
2. Add `@register_trainer("cppo")` decorator → automatically registered
3. Override lifecycle hooks for CPPO-specific behavior (off-policy sampling, reward shaping)
4. Use `ReplayBuffer` with TransferQueue for off-policy GRPO data
5. Config: `config.trainer.v1.type = "cppo"` → selects CPPO trainer

This is the simplest path to CPPO integration — no changes to V1 core architecture needed.

---

*Created 2026-06-19. V1 trainer architecture reading. Updated June 19 with latest code research.*
