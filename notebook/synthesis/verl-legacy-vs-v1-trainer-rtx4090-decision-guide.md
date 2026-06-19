# verl Training Architecture Comparison: Legacy vs V1 — RTX 4090 Decision Guide

**Created: 2026-06-19 | Cross-framework synthesis**

---

## Overview

★★★★★★★★★ verl is undergoing a major architectural transition from the legacy `main_ppo.py` trainer to the V1 unified trainer. This note provides a comprehensive comparison for RTX 4090 practitioners to make informed decisions about which path to use.

---

## Legacy Trainer (main_ppo.py)

### Architecture
- Monolithic script: `verl/trainer/main_ppo.py`
- Manual backend configuration: `--trainer.backend fsdp/megatron/etc`
- Hard-coded lifecycle: init → rollout → train → repeat
- RolloutMode enum: HYBRID/COLOCATED/STANDALONE
- Weight sync: manual ZMQ/NCCL configuration

### Known Issues (from 7-framework research)
- **#6699**: 3 backends (megatron/automodel/torchtitan) have UNFIXED detach leak → MUST use FSDP
- **#6468**: FSDP2 CPU memory leak → 0.6-6.3 GiB/step → long-running GRPO at risk
- **#6782**: LoRA rank=64 breaks EOS → MUST use rank=32
- **#605**: rLLM grouping bug → group_size must be ≥2 (different repo but same algorithm)

### RTX 4090 Config
```yaml
algorithm: cppo
bypass_mode: true
backend: fsdp  # ONLY safe backend (#6699)
zero_stage: 2  # NEVER 3 (#8072/#8076)
optimizer: CPU_Adam  # ONLY viable optimizer
rollout.name: sglang
rollout.sleep_level: 1  # LoRA adapter path
rollout.enforce_eager: true  # DSV4 safety
```

---

## V1 Trainer (verl/trainer/ppo/v1/)

### Architecture
- Pluggable registered trainers: `register_trainer("sync")`, `register_trainer("colocate_async")`
- Configuration-driven: `config.trainer.v1.type = "sync"`
- Lifecycle hooks pattern: `on_init_end`, `on_train_begin`, `on_step_end`, `on_sample_end`
- Checkpoint engine registry: `nccl/hccl/mooncake/nixl/kimi`
- Agent loop with transfer queue: `AgentLoopManagerTQ`, `AgentLoopWorkerTQ`

### Key Improvements over Legacy
1. **Modularity**: Trainer type is selectable via config, not hard-coded
2. **Extensibility**: New algorithms (CPPO, DAPO) can register via `register_trainer()`
3. **Checkpoint engine**: Clean weight sync abstraction with 5 backends
4. **Async support**: `colocate_async` enables partial rollout on single GPU
5. **Agent loop**: Transfer queue-based async data movement for better throughput

### RTX 4090 Config (V1 sync)
```yaml
trainer:
  v1:
    type: sync  # ★★★★★★★★ #1 BEST for RTX 4090
    backend: fsdp
algorithm:
  name: cppo
  bypass_mode: true
  clip_ratio: 0.2
  group_size: 2
  position_weight: true
rollout:
  name: sglang
  sleep_level: 1
  lora_rank: 32
  lora_alpha: 64
  merge: false
  enforce_eager: true
checkpoint_engine:
  type: nccl  # Standard CUDA for RTX 4090
```

---

## Comparison Matrix

| Feature | Legacy main_ppo | V1 trainer_sync | V1 colocate_async | V1 separate_async |
|---------|-----------------|-----------------|-------------------|-------------------|
| RTX 4090 viable | ✓ | ✓ | ✓ (experimental) | ✗ |
| Configuration | manual args | config-driven | config-driven | config-driven |
| Weight sync | manual ZMQ/NCCL | CheckpointEngineManager | CheckpointEngineManager | CheckpointEngineManager |
| Algorithm support | GRPO/PPO/CPPO | any registered | any registered | any registered |
| Partial rollout | ✗ | ✗ | ✓ | ✓ |
| Async generation | ✗ | ✗ | ✓ (abort+resume) | ✓ |
| Sleep/wake | HYBRID mode | sleep_replicas | abort+sleep+resume | abort+sleep+resume |
| CPPO integration | manual flags | register_trainer("cppo") | register_trainer("cppo_async") | N/A |
| Maintenance status | maintenance mode | active development | active development | active development |
| Production tested | ✓ (extensive) | limited | limited | limited |
| FSDP detach fix | #6699 MERGED | inherited | inherited | inherited |
| FSDP2 leak risk | #6468 confirmed | inherited | inherited | inherited |

---

## RTX 4090 Decision Flowchart

```
RTX 4090 GRPO Training Decision:
├── Current (immediate): Legacy main_ppo
│   └── Production-tested, known safety rules
│   └── Algorithm: CPPO + bypass_mode
│   └── Backend: FSDP (ONLY safe backend)
│
├── Near-term (V1 stable): V1 trainer_sync
│   └── Same lifecycle, cleaner config
│   └── register_trainer("cppo") available
│   └── CheckpointEngineManager for weight sync
│   └── dp=1: NCCL broadcast = identity → no data transfer!
│
├── Experimental (future): V1 colocate_async
│   ├── Abort+sleep+resume lifecycle
│   ├── Partial rollout on single GPU
│   ├── Resource contention risk (trainer+rollout share GPU)
│   └── Requires careful memory budgeting
│
└── NOT viable: V1 separate_async
    └── Requires 2+ GPUs
    └── RTX 4090 is single GPU (dp=1)
```

---

## Weight Sync Detail (dp=1 RTX 4090)

★★★★★★★★★ Critical insight: On dp=1 (single GPU), NCCL broadcast is **identity operation**:
- No actual data transfer between GPUs (there's only one GPU!)
- ZMQ metadata sync still runs (for control signals)
- `update_weights` = load LoRA deltas from CPU → GPU memory
- `sleep_replicas` = clear KV cache + LoRA adapter memory
- This is why `sleep_level=1` (LoRA adapter path) is optimal: minimum memory transfer

Memory lifecycle per step on RTX 4090:
```
1. update_weights: LoRA δ_t → GPU (~4 MiB for rank=32)
2. rollout: generate responses (KV cache grows)
3. score: compute rewards (small compute, no extra memory)
4. sleep_replicas: clear KV cache + LoRA δ_t (~4 MiB freed)
5. train: compute gradient + optimizer step (CPU_Adam offload → 3.8Ψ on CPU)
6. repeat: load LoRA δ_{t+1} → GPU
```

Peak memory budget:
- Base model weights: 2Ψ = 16 GiB (8B BF16)
- LoRA adapter: ~4 MiB (rank=32)
- KV cache (during rollout): varies by seq_len
- Training gradient: offloaded to CPU (CPU_Adam)
- Total peak: ~16-18 GiB → fits on RTX 4090 24 GiB!

---

## Safety Rules (unchanged across both architectures)

★★★★★★★★★ The 10 MUST DO + 10 MUST NOT rules apply to BOTH legacy and V1 trainers:

1. **ZeRO-2 (NEVER ZeRO-3)** — #8072/#8076 regression
2. **CPU_Adam** — 18Ψ→3.8Ψ optimizer offload
3. **gradient_clipping=1.0** — #8068 default regression
4. **enforce_eager=True** — 11 DSV4 failures
5. **bypass_mode=True** — 18Ψ→3.8Ψ (removes ref model)
6. **LoRA rank=32** — #6782 rank=64 breaks EOS
7. **overlap_comm=False** — #8061 NaN on single GPU
8. **cosine decay + warmup** — standard LR schedule
9. **group_size≥2** — #605 normalization bug
10. **ulimit -n 65536** — #8075 fd leak safety

---

## Migration Path

★★★★★★★★★ Recommended migration sequence:

1. **Now**: Use legacy `main_ppo` — production-tested, safe
2. **When V1 stable**: Switch to `trainer_sync` — same behavior, cleaner config
3. **Future**: Explore `trainer_colocate_async` — potential throughput improvement
4. **CPPO integration**: `register_trainer("cppo")` → seamless in V1

★★★★★★★★★ DO NOT switch to V1 prematurely — legacy is battle-tested. V1 is still in development.

---

*Created 2026-06-19. verl legacy vs V1 trainer comparison for RTX 4090.*
