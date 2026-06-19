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
| nccl_checkpoint_engine | NCCL | ★★★★★★★★ YES (standard CUDA) |
| hccl_checkpoint_engine | HCCL | NO (Ascend NPU only) |
| mooncake_checkpoint_engine | Mooncake | NO (multi-node RDMA) |
| nixl_checkpoint_engine | NIXL | NO (experimental) |
| kimi_checkpoint_engine | Kimi | NO (Moonshot proprietary) |

RTX 4090 uses `nccl_checkpoint_engine` — same as existing weight sync path.

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

*Created 2026-06-19. V1 trainer architecture reading.*
