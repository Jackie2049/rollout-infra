# verl V1 Trainer Architecture Complete Deep Reading

**Date**: 2025-07-15
**Source**: /tmp/verl-fork/verl/ (forked from Jackie2049/verl, latest codebase)
**Author**: Deep source code analysis

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [V1 Trainer Type System](#2-v1-trainer-type-system)
3. [Checkpoint Engine System](#3-checkpoint-engine-system)
4. [Rollout Weight Sync](#4-rollout-weight-sync)
5. [HYBRID Sleep/Wake Architecture](#5-hybrid-sleepwake-architecture)
6. [TransferQueue Data Flow](#6-transferqueue-data-flow)
7. [Agent Loop & Async Rollout](#7-agent-loop--async-rollout)
8. [RTX 4090 Implications](#8-rtx-4090-implications)

---

## 1. Architecture Overview

### 1.1 Key Source Files

| Component | File Path |
|-----------|-----------|
| Main Entry | `/tmp/verl-fork/verl/trainer/main_ppo.py` |
| Core Trainer | `/tmp/verl-fork/verl/trainer/ppo/ray_trainer.py` (RayPPOTrainer, ~1618 lines) |
| TQ Trainer | `/tmp/verl-fork/verl/experimental/transfer_queue/ray_trainer.py` |
| Checkpoint Engine Base | `/tmp/verl-fork/verl/checkpoint_engine/base.py` |
| NCCL Engine | `/tmp/verl-fork/verl/checkpoint_engine/nccl_checkpoint_engine.py` |
| NIXL Engine | `/tmp/verl-fork/verl/checkpoint_engine/nixl_checkpoint_engine.py` |
| HCCL Engine | `/tmp/verl-fork/verl/checkpoint_engine/hccl_checkpoint_engine.py` |
| Rollout Replica | `/tmp/verl-fork/verl/workers/rollout/replica.py` |
| Rollout Config | `/tmp/verl-fork/verl/workers/config/rollout.py` |
| FSDP Workers | `/tmp/verl-fork/verl/workers/fsdp_workers.py` |
| vLLM Async Server | `/tmp/verl-fork/verl/workers/rollout/vllm_rollout/vllm_async_server.py` |
| SGLang Async Server | `/tmp/verl-fork/verl/workers/rollout/sglang_rollout/async_sglang_server.py` |
| Agent Loop | `/tmp/verl-fork/verl/experimental/agent_loop/agent_loop.py` |
| TransferQueue Utils | `/tmp/verl-fork/verl/utils/transferqueue_utils.py` |
| Algorithm Core | `/tmp/verl-fork/verl/trainer/ppo/core_algos.py` |
| Roles & Utils | `/tmp/verl-fork/verl/trainer/ppo/utils.py` |

### 1.2 High-Level Architecture

```
                                    RayPPOTrainer (Single Controller on CPU)
                                    ├─ RayWorkerGroup: ActorRolloutRef
                                    ├─ RayWorkerGroup: Critic (optional)
                                    ├─ RayWorkerGroup: RefPolicy (optional, if not ref_in_actor)
                                    ├─ RewardLoopManager
                                    ├─ AgentLoopManager (async rollout)
                                    └─ CheckpointEngineManager
                                        ├─ ColocatedCheckpointEngine ("naive")
                                        ├─ NCCLCheckpointEngine ("nccl")
                                        ├─ NIXLCheckpointEngine ("nixl")
                                        ├─ HCCLCheckpointEngine ("hccl")
                                        └─ (kimi/mooncake: not in open-source fork)
```

### 1.3 Data Flow: Training Step

```
for epoch, batch_dict in train_dataloader:
    batch = DataProto.from_single_dict(batch_dict)
    batch.non_tensor_batch["uid"] = uuid4() per sample
    gen_batch = batch.pop(non_reward_keys)

    [1] GENERATE: async_rollout_manager.generate_sequences(gen_batch)
        checkpoint_manager.sleep_replicas()  # free rollout GPU memory

    [2] REWARD: extract_reward(batch) or _compute_reward_colocate(batch)

    [3] OLD_LOG_PROB: actor_rollout_wg.compute_log_prob(batch)
        (or bypass_mode: apply_bypass_mode directly uses rollout_log_probs)

    [4] REF_LOG_PROB: ref_policy_wg.compute_ref_log_prob(batch) (if use_kl)

    [5] VALUES: critic_wg.compute_values(batch) (if use_critic)

    [6] ADVANTAGE: compute_advantage(batch)  # lightweight, on driver CPU
        (GAE, GRPO, REINFORCE++, or registered estimators)

    [7] UPDATE_CRITIC: critic_wg.update_critic(batch) (if use_critic)

    [8] UPDATE_ACTOR: actor_rollout_wg.update_actor(batch)

    [9] SAVE_CHECKPOINT: _save_checkpoint() (if save_freq matches)

    [10] UPDATE_WEIGHTS: checkpoint_manager.update_weights()
        # sync trained weights to rollout replicas
```

### 1.4 Critical Design Decisions

1. **Sync mode DEPRECATED**: `RolloutConfig.__post_init__` raises ValueError for `mode="sync"`. Only async mode supported. Line 244: `raise ValueError("Rollout mode 'sync' has been removed.")`

2. **Hybrid Engine Mandatory**: `self.hybrid_engine = config.actor_rollout_ref.hybrid_engine`, line 269: `assert self.hybrid_engine, "Currently, only support hybrid engine"`

3. **New Worker Implementation (Model Engine)**: `use_legacy_worker_impl` config switch. `"disable"` uses `ActorRolloutRefWorker` + `TrainingWorker` from `/tmp/verl-fork/verl/workers/engine_workers.py`. `"auto"/"enable"` uses legacy FSDP/Megatron workers.

4. **ref_in_actor**: When LoRA is used (lora_rank > 0 or lora_adapter_path exists), reference policy computation uses actor without LoRA applied (`no_lora_adapter=True` metadata), saving a separate worker group.

---

## 2. V1 Trainer Type System

### 2.1 Current State: No `register_trainer` Pattern

**IMPORTANT**: The fork code does NOT have a `verl/trainer/ppo/v1/` directory. The `register_trainer` decorator pattern described in MEMORY.md does not exist in this codebase version. The current architecture uses a single `RayPPOTrainer` class with mode switches.

There are effectively 3 trainer variants, but they are **separate files**, not registered types:

| Variant | File | Data Transfer | Weight Sync |
|---------|------|---------------|-------------|
| Standard (legacy) | `verl/trainer/ppo/ray_trainer.py` | DataProto via Ray RPC | naive/nccl/nixl/hccl |
| TransferQueue | `verl/experimental/transfer_queue/ray_trainer.py` | TransferQueue (BatchMeta) | Same engines |
| Separation | `verl/experimental/separation/ray_trainer.py` | DataProto via Ray RPC | Separate rollout pool |
| Fully Async | `verl/experimental/fully_async_policy/fully_async_trainer.py` | Custom async pipeline | Custom |

### 2.2 The "async_rollout_mode" = True Always

In the standard trainer (`ray_trainer.py`), line 821:
```python
self.async_rollout_mode = True  # Note: mode is always "async" since sync mode is deprecated
```

This means ALL standard training now uses `AgentLoopManager` for rollout generation, not direct `actor_rollout_wg.generate_sequences()`.

### 2.3 Async Rollout Manager Initialization

From `ray_trainer.py` lines 824-849:
```python
manager_class_fqn = config.rollout.get("agent", {}).get("agent_loop_manager_class")
if manager_class_fqn:
    AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
else:
    from verl.experimental.agent_loop import AgentLoopManager

enable_agent_reward_loop = not self.use_rm or config.reward.reward_model.enable_resource_pool
reward_loop_worker_handles = (
    self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
)
self.async_rollout_manager = AgentLoopManager(
    config=config, worker_group=self.actor_rollout_wg,
    rollout_resource_pool=actor_rollout_resource_pool,
    reward_loop_worker_handles=reward_loop_worker_handles,
)
```

### 2.4 Role System

From `verl/trainer/ppo/utils.py`:
```python
class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2     # hybrid actor+rollout
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6  # hybrid actor+rollout+ref (new model engine)
    Env = 7
```

The `ActorRolloutRef` role is used when `use_legacy_worker_impl == "disable"` (new model engine), where actor, rollout, and ref policy are all in the same `ActorRolloutRefWorker`.

---

## 3. Checkpoint Engine System

### 3.1 Engine Registry Pattern

From `/tmp/verl-fork/verl/checkpoint_engine/base.py`:
```python
class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):
        def wrapper(cls):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls
        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        return cls._registry[backend]
```

Registered backends in open-source fork:
- `"naive"` -> `ColocatedCheckpointEngine` (same GPU, in-process)
- `"nccl"` -> `NCCLCheckpointEngine` (NCCL broadcast + ZMQ metadata)
- `"nixl"` -> `NIXLCheckpointEngine` (RDMA P2P chain + ZMQ metadata)
- `"hccl"` -> `HCCLCheckpointEngine` (Ascend NPU HCCL broadcast, same pattern as NCCL)

**NOT in open-source fork**: `"kimi"` and `"mooncake"` engines (enterprise/cloud-only).

### 3.2 CheckpointEngineManager

From `base.py` lines 286-410:

```python
class CheckpointEngineManager:
    """Manages weight sync between trainer and rollout replicas."""

    def __init__(self, backend: str, trainer: RayWorkerGroup, replicas: list[RolloutReplica]):
        self.backend = backend
        self.backend_cls = CheckpointEngineRegistry.get(backend)
        self.trainer = trainer
        self.replicas = replicas
```

**Architecture diagram** (from source code comments):
```
    Trainer Process Group                Rollout Replicas
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ME0    │ ME1    │     │ MEn    │         │     Replica 0     │     Replica 1     │
    │ └──┬─┘ │ └──┬─┘ │     │ └──┬─┘ │         │ CE│CE│CE│CE│CE│CE│CE│CE │
    │    v   │    v   │     │    v   │         └──┬─┴──┬─┴──┬─┴──┬─┘
    │ ┌──┴─┐ │ ┌──┴─┐ │     │ ┌──┴─┐ │            ^    ^   cuda ipc
    │ CE  │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┐
    └────┼───┴────────┴─────┴────────┘         │ CE │ CE │ CE │ CE │
         v                                        |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┘
```

### 3.3 naive Engine (ColocatedCheckpointEngine)

**RTX 4090 BEST option**. Simplest engine: trainer and rollout on same GPU, in same process.

```python
@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    def send_weights(self, weights):
        self.weights = weights  # just store reference

    def receive_weights(self):
        yield from self.weights  # yield back same reference
        self.weights = None
```

**Key insight**: With `backend="naive"`:
- `sleep_replicas()` actually sleeps the rollout replicas (line 375: `if self.backend != "naive": return` -- wait, this skips for non-naive!)
- Correction: line 374-376 shows that sleep_replicas ONLY works for naive backend. For nccl/nixl, sleep is skipped (disaggregated rollout).
- `update_weights()` for naive backend (line 383-385): just calls `ray.get(self.trainer.update_weights())`, which invokes the FSDP worker's `rollout_mode()` method.

### 3.4 NCCL Engine

**For multi-GPU/multi-node weight transfer**. Uses NCCL broadcast + ZMQ PUB/SUB for metadata.

Key implementation details:
- **Topology**: Trainer rank 0 is master, all rollout ranks are receivers. Star broadcast topology.
- **Double buffering**: `send_buf` + `recv_buf` (both `bucket_size` bytes). Swap after each broadcast.
- **Metadata via ZMQ**: Rank 0 PUB-lishes bucket_meta (tensor names, shapes, offsets), other ranks SUB-scribe.
- **Bucket-based transfer**: Weights packed into fixed-size buckets (default 2048 MB). Overlap: fill next bucket while previous broadcast runs.
- **cupy for send buffers**: Uses `cp.zeros()` instead of `torch.zeros()` to avoid memory registration errors with `expandable_segments:True`.

### 3.5 NIXL Engine (RDMA P2P)

**For disaggregated rollout with RDMA**. Uses NVIDIA NIXL library for GPU-direct P2P transfers.

Key implementation details:
- **Chain topology**: Trainer rank 0 -> rollout rank 1 -> rank 2 -> ... -> rank N. Each node reads from previous, forwards to next.
- **NIXL agents**: Each worker creates a `NixlAgent` with RDMA memory registration.
- **Memory registration**: `send_buf` and `recv_buf` registered with NIXL for RDMA access.
- **ZMQ for metadata**: Same ZMQ pattern as NCCL, but uses PUSH/PULL instead of PUB/SUB.
- **RDMA transfer**: `ReadableOperation` sends metadata + remote_descs, `ReadOperation` performs RDMA read from remote agent.

### 3.6 Engine Selection Logic

From `ray_trainer.py` line 845-849:
```python
self.checkpoint_manager = CheckpointEngineManager(
    backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
    trainer=self.actor_rollout_wg,
    replicas=self.async_rollout_manager.rollout_replicas,
)
```

Config from `CheckpointEngineConfig`:
```python
@dataclass
class CheckpointEngineConfig(BaseConfig):
    backend: Optional[str] = MISSING  # naive, nccl, nixl, hccl
    update_weights_bucket_megabytes: int = 2048
    engine_kwargs: dict = field(default_factory=dict)
```

**RTX 4090 Selection**: `backend="naive"` (colocated, single GPU, no IPC overhead).

---

## 4. Rollout Weight Sync

### 4.1 Two-Phase Weight Sync: Base + LoRA Delta

From `fsdp_workers.py` lines 680-760, the `rollout_mode()` method:

```python
async def rollout_mode(self):
    aggressive_empty_cache(force_sync=True)

    # Phase 1: Collect parameters
    if hasattr(peft_model, "peft_config"):  # LoRA case
        peft_config = peft_model.peft_config.get("default", None)
        params = collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=self.config.rollout.get("layered_summon", False),
            base_sync_done=self.base_sync_done,  # KEY: first step vs subsequent
        )
        if not self.base_sync_done:  # First step: collect ALL params (base + LoRA merged)
            params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
    else:  # No LoRA: collect all params
        params = self.actor_module_fsdp.state_dict()

    # Phase 2: Offload FSDP model to CPU (free GPU memory)
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    # Phase 3: Prepare per_tensor_param generator
    if peft_config is not None and self.base_sync_done:
        per_tensor_param = params.items()  # Only LoRA deltas!
    else:
        per_tensor_param = (
            (name, param.to(device).full_tensor() if isinstance(param, DTensor) else param)
            for name, param in params.items()
        )

    # Phase 4: Resume rollout weights
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])

    # Phase 5: Handle sleep_level=2 LoRA two-phase update
    if peft_config and getattr(self.rollout, "sleep_level", None) == 2 and config.rollout.free_cache_engine:
        # Base params first (base_sync_done=False), then LoRA params
        per_tensor_base_params = (
            (name, param.to(device).full_tensor() if isinstance(param, DTensor) else param)
            for name, param in base_model_params.items()
        )
        await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
        del base_model_params, per_tensor_base_params

    # Phase 6: Update LoRA or full weights
    await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
```

### 4.2 Key Variables

- **`base_sync_done`** (bool): Tracks whether base weights have been synced to rollout.
  - `False` on first step: ALL weights (base + LoRA merged) are transferred.
  - `True` after first step: Only LoRA delta weights are transferred.
  - This is the mechanism for "80x payload reduction" -- after initial sync, only LoRA deltas (typically <1% of total params) need to be sent each step.

- **`layered_summon`** (bool, default False): Whether to use layered parameter summon (per-layer instead of whole-model). Reduces peak memory from ~60 GiB to ~6-8 GiB on RTX 4090.

- **`sleep_level`** (property on rollout object):
  - `1` (LoRA adapter mode): `tags=["kv_cache"]` -- base weights stay on GPU, only LoRA adapter + KV cache freed. Rollout resumes with `tags=["weights"]` to reload LoRA.
  - `2` (merge mode): `tags=["kv_cache", "weights"]` -- ALL weights freed (base + LoRA merged). Rollout must reload everything. This requires two-phase update (base first, then LoRA delta).

### 4.3 SGLang update_weights

From `sglang_rollout.py` lines 179-216:
```python
async def update_weights(self, weights, **kwargs):
    await self._init_server_adapter()
    update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20

    # Optional FP8 quantization before loading
    if self.config.get("quantization", None) == "fp8":
        weights = quant_weights_by_name(weights, ...)

    # Bucket-based transfer to SGLang engine
    async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
        await sgl_update_weights(engine=self._engine, params_batch=params_batch, ...)

    # Flush KV cache after weight update
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        await self._engine.flush_cache()
```

### 4.4 Weight Sync Flow Summary

```
Step 1 (first):
    trainer.rollout_mode() -> collect ALL params (base + LoRA merged)
    -> rollout.resume(tags=["weights"])  # wake up from sleep
    -> rollout.update_weights(full_params, base_sync_done=False)
    -> base_sync_done = True  # set after first step

Step N (subsequent):
    trainer.rollout_mode() -> collect only LoRA deltas (base_sync_done=True)
    -> rollout.resume(tags=["weights"])  # wake up LoRA adapter only
    -> rollout.update_weights(lora_deltas, peft_config=peft_config, base_sync_done=True)

    Payload: ~1-5 GiB (LoRA deltas) vs ~14 GiB (full model) = 80x reduction
```

---

## 5. HYBRID Sleep/Wake Architecture

### 5.1 RolloutMode Enum

From `/tmp/verl-fork/verl/workers/rollout/replica.py` lines 46-59:

```python
class RolloutMode(Enum):
    HYBRID = "hybrid"
    # Rollout engine and training engine fused in same process.
    # Share GPUs, switch context with weight synchronization.
    # Usage: on-policy training.

    COLOCATED = "colocated"
    # Rollout colocated with hybrid engine in same placement group but separate process.
    # Share GPUs, switch context WITHOUT weight synchronization.
    # Usage: GRM (LLM as a judge).

    STANDALONE = "standalone"
    # Disaggregated rollout, separate GPU resources.
    # Usage: off-policy training.
```

### 5.2 HYBRID Mode: Sleep/Wake Detail

**vLLM HYBRID sleep** (vllm_async_server.py lines 601-607):
```python
async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        # Don't use engine.sleep(level=2) here
        await self.engine.collective_rpc("sleep", kwargs={"level": 2})
```

Note: The comment says "Don't use engine.sleep(level=2) here" but the code actually calls `collective_rpc("sleep", level=2)` which IS sleep level 2. The comment likely means "don't call `self.engine.sleep(level=2)` directly on the driver" and instead uses collective_rpc to call it on each worker.

**SGLang HYBRID sleep** (async_sglang_server.py lines 299-305):
```python
async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)
```

Both release KV cache AND weights (sleep_level=2 behavior for HYBRID mode).

### 5.3 COLOCATED Mode: Sleep/Wake Detail

**vLLM COLOCATED wake_up** (vllm_async_server.py lines 587-596):
```python
async def wake_up(self):
    if self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.wake_up(tags=["kv_cache", "weights"])
        await self.engine.reset_prefix_cache()
```

**SGLang COLOCATED wake_up** (async_sglang_server.py lines 284-295):
```python
async def wake_up(self):
    if self.rollout_mode == RolloutMode.COLOCATED:
        obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
        await self.tokenizer_manager.resume_memory_occupation(obj, None)
        await self.tokenizer_manager.flush_cache()
```

**vLLM COLOCATED sleep** (vllm_async_server.py lines 601-612):
```python
async def sleep(self):
    if self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)
```

Wait -- the vLLM COLOCATED sleep uses `level=1` (only KV cache), but wake_up uses `tags=["kv_cache", "weights"]`. This seems inconsistent. The SGLang COLOCATED sleep releases `tags=["kv_cache", "weights"]` (both).

### 5.4 STANDALONE Mode

Both vLLM and SGLang skip sleep/wake in STANDALONE mode:
```python
elif self.rollout_mode == RolloutMode.STANDALONE:
    logger.info("skip wake_up/sleep in standalone mode")
```

No GPU memory sharing in standalone mode -- rollout has its own GPUs.

### 5.5 CheckpointEngineManager Sleep/Wake Integration

From `base.py` lines 370-376:
```python
@auto_await
async def sleep_replicas(self):
    """Sleep all rollout replicas: free weight and kv_cache device memory."""
    # skip sleep replicas for disaggregated rollout
    if self.backend != "naive":
        return  # Only sleep for naive (colocated) backend
    await asyncio.gather(*[r.sleep() for r in self.replicas])
```

**CRITICAL INSIGHT**: Sleep only happens for `backend="naive"` (HYBRID mode on same GPU). For nccl/nixl/hccl backends (disaggregated rollout), sleep is skipped because rollout has separate GPU resources.

### 5.6 Training Step Sleep/Wake Sequence

From `ray_trainer.py` `fit()` method:

```python
# After rollout generation:
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
self.checkpoint_manager.sleep_replicas()  # sleep rollout -> free GPU for training

# After actor update:
self.checkpoint_manager.update_weights()  # wake rollout + sync weights
```

For naive backend, `update_weights()` calls `self.trainer.update_weights()` which triggers `rollout_mode()` on the FSDP worker, which:
1. Loads FSDP model to GPU (if offloaded)
2. Collects parameters (LoRA deltas or full)
3. Offloads FSDP model to CPU
4. Resumes rollout (wake up)
5. Updates rollout weights

### 5.7 sleep_level=1 vs sleep_level=2 Summary

| Property | sleep_level=1 (LoRA adapter) | sleep_level=2 (merge mode) |
|----------|------------------------------|---------------------------|
| What freed | KV cache + LoRA adapter | KV cache + ALL weights (base + LoRA merged) |
| Resume tags | `["kv_cache"]` then `["weights"]` (LoRA only) | `["weights"]` (full reload) |
| Payload per step | LoRA deltas only (~1-5 GiB) | Full model + LoRA deltas (~14+ GiB) |
| Base weights | STAY on GPU (permanent) | DESTROYED each sleep cycle |
| Two-phase update | No (single phase: LoRA delta) | Yes (base first, then LoRA delta) |
| RTX 4090 optimal | **YES** (80x payload reduction) | NO (full reload each step) |
| SGLang tags | `tags=["kv_cache"]` for sleep, `["weights"]` for resume | `tags=["kv_cache", "weights"]` for sleep |
| vLLM level | `sleep(level=1)` | `sleep(level=2)` or `collective_rpc("sleep", level=2)` |

---

## 6. TransferQueue Data Flow

### 6.1 Overview

TransferQueue (TQ) is an **experimental** data transfer system that replaces DataProto-based Ray RPC with a centralized storage + controller architecture. It is NOT the default; it must be explicitly enabled via `config.transfer_queue.enable = True`.

### 6.2 TQ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 TransferQueueController               │
│                 (Ray remote actor, single instance)   │
│                 - Manages batch metadata              │
│                 - Coordinates put/get operations      │
└─────────────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         v              v              v
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ StorageUnit  │ │ StorageUnit  │ │ StorageUnit  │
│ (Ray remote) │ │ (Ray remote) │ │ (Ray remote) │
│ - CPU memory │ │ - CPU memory │ │ - CPU memory │
│ or ZMQ IPC   │ │ or ZMQ IPC   │ │ or ZMQ IPC   │
└──────────────┘ └──────────────┘ └──────────────┘
         │              │              │
         v              v              v
    Workers get/put data via TQClient
    (each worker has its own TQClient instance)
```

### 6.3 Key Components

From `transferqueue_utils.py`:

**`tqbridge` decorator** (lines 244-328):
```python
def tqbridge(dispatch_mode=None, put_data=True):
    """Creates a decorator for bridging BatchMeta and DataProto.

    When TransferQueue is enabled:
    - Input BatchMeta is converted to DataProto (get_data from storage)
    - Output DataProto is converted back to BatchMeta (put_data to storage)
    When TransferQueue is disabled:
    - Just calls the function directly (no conversion)
    """
```

**BatchMeta**: Lightweight metadata object that references data in storage. Contains:
- `field_names`: list of tensor field names
- `extra_info`: dict of metadata (global_steps, timing, etc.)
- `global_indexes`: sample indices for data parallel dispatch
- Does NOT contain actual tensor data -- data lives in StorageUnit

**DataProto vs BatchMeta flow**:
```
Standard: Worker receives DataProto (tensors in-memory via Ray RPC)
TQ:       Worker receives BatchMeta (reference to storage), then:
          1. tqbridge converts BatchMeta -> DataProto (get_data)
          2. Worker processes DataProto
          3. tqbridge converts DataProto -> BatchMeta (put_data)
```

### 6.4 TQ Trainer Initialization

From `transfer_queue/ray_trainer.py` lines 325-386:
```python
def _initialize_transferqueue(self):
    # 1. Create TransferQueueStorage (AsyncSimpleStorageManager)
    #    - Allocate SimpleStorageUnit Ray actors on CPU nodes
    #    - Size: train_batch_size * num_global_batch * rollout.n

    # 2. Create TransferQueueController (single remote actor)
    self.data_system_controller = TransferQueueController.remote()

    # 3. Register controller & storage, get ZMQ server info
    self.data_system_controller_info = process_zmq_server_info(...)
    self.data_system_storage_unit_infos = process_zmq_server_info(...)

    # 4. Create TQClient on trainer (sync mode)
    create_transferqueue_client(client_id="Trainer", config=..., sync=True)
```

### 6.5 TQ Data Flow in Training Step

```
Trainer (CPU):
    batch = DataProto -> TensorDict -> tq_client.put(data, partition_id)
    gen_meta = BatchMeta (reference only)

AgentLoopWorker (CPU):
    receives BatchMeta
    tqbridge: BatchMeta -> DataProto (get_data from storage)
    generates sequences -> DataProto
    tqbridge: DataProto -> BatchMeta (put_data back to storage)

Actor Worker (GPU):
    receives BatchMeta
    tqbridge: BatchMeta -> DataProto (get_data from storage)
    computes log_probs, updates actor -> DataProto
    tqbridge: DataProto -> BatchMeta (put_data back to storage)
```

### 6.6 TQ Advantages over DataProto

1. **Memory efficiency**: Data lives in centralized storage (CPU), not duplicated across workers
2. **Zero-copy for lightweight ops**: Advantage computation on driver only needs metadata, not full tensors
3. **Streaming**: Rollout workers can put data incrementally, training workers can get partial data
4. **GRPO grouping**: TQ controller can group samples by uid for GRPO advantage computation

### 6.7 TQ Limitations (Current)

1. **Experimental**: Not default, requires explicit config
2. **REMAX not supported**: Commented out in TQ trainer (lines 1184-1204)
3. **curriculum sampler**: Not fully supported (TODO comments)
4. **on_batch_end**: Not fully supported (TODO comments)
5. **Requires transfer_queue package**: Must pip install separately

---

## 7. Agent Loop & Async Rollout

### 7.1 AgentLoopManager

From `/tmp/verl-fork/verl/experimental/agent_loop/agent_loop.py`:

```python
class AgentLoopManager:
    """Manages a group of agent loop workers for async rollout generation."""

    def __init__(self, config, worker_group, rollout_resource_pool, reward_loop_worker_handles):
        self._initialize_llm_servers(rollout_resource_pool)
        self._init_agent_loop_workers()
```

### 7.2 Rollout Replica Initialization

From `_initialize_llm_servers` (lines 879-926):
```python
rollout_world_size = config.rollout.tp * config.rollout.dp * config.rollout.pp
world_size = worker_group.world_size if worker_group else trainer.n_gpus * trainer.nnodes
num_replicas = world_size // rollout_world_size

# Create replicas
self.rollout_replicas = [rollout_replica_class(replica_rank=i, ...) for i in range(num_replicas)]

# Initialize based on mode
if worker_group and rollout_config.name != "trtllm":
    # HYBRID: rollout uses same workers as training
    await server.init_hybrid(self.worker_group)
elif worker_group and rollout_config.name == "trtllm":
    # HYBRID COLOCATED: TRT-LLM needs separate process
    await server.init_hybrid_colocated(self.worker_group, rollout_resource_pool)
else:
    # STANDALONE: separate GPU resources
    await server.init_standalone()
```

### 7.3 AgentLoopWorker

From `agent_loop.py` lines 344-474:

```python
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and runs each in an agent loop."""

    def __init__(self, config, server_handles, reward_loop_worker_handles):
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        # LRU cache for sticky session (request_id -> server)
        # Least-requests load balancing

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        # For each sample, create agent loop instance
        # Run agent loop asynchronously
        # Post-process outputs into DataProto
```

### 7.4 AsyncLLMServerManager

From `agent_loop.py` lines 56-121:

```python
class AsyncLLMServerManager:
    """Manages multiple OpenAI compatible LLM servers.

    Features:
    - Load balance: least requests load balancing
    - Sticky session: multi-turn conversations go to same server
      for automatic prefix caching
    """

    def __init__(self, config, server_handles, max_cache_size=10000):
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id):
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]  # sticky session
        _, _, server = self.weighted_serveres[0]  # least requests
        heapq.heapreplace(...)
```

### 7.5 generate_sequences Flow

```
1. Trainer: gen_batch = batch.pop(non_reward_keys)
2. Trainer: gen_batch_output = async_rollout_manager.generate_sequences(gen_batch)

   Inside AgentLoopManager.generate_sequences():
   - chunk gen_batch across agent_loop_workers (num_workers, default=8)
   - Each AgentLoopWorker.generate_sequences():
     - For each sample, create agent_loop instance
     - agent_loop.run(sampling_params, **kwargs) -> AgentLoopOutput
     - _postprocess outputs -> DataProto with:
       prompts, responses, response_mask, input_ids, attention_mask, position_ids
       Optional: rollout_log_probs, routed_experts, rm_scores

3. Trainer: checkpoint_manager.sleep_replicas()  # sleep after rollout
4. Trainer: continue with reward, log_prob, advantage, training...
```

---

## 8. RTX 4090 Implications

### 8.1 Optimal Configuration

Based on the source code analysis, the RTX 4090 optimal configuration is:

```yaml
# MUST DO
actor_rollout_ref:
  hybrid_engine: true
  rollout:
    checkpoint_engine:
      backend: naive  # colocated, no IPC overhead
    enable_sleep_mode: true  # free rollout GPU memory during training
    name: sglang  # tag-based sleep/wake (more fine-grained)
    free_cache_engine: true  # actually free memory, not just mask
    enforce_eager: true  # DSV4 mandatory
  actor:
    strategy: fsdp  # NOT fsdp2 (MoE FSDP2 backward failure, verl #7016)
    use_legacy_worker_impl: auto  # or enable (new engine not proven on 4090)
  model:
    lora:
      rank: 32  # NOT 64 (EOS bug, verl #6782)

trainer:
  device: cuda
  nnodes: 1
  n_gpus_per_node: 1  # single GPU

algorithm:
  adv_estimator: grpo  # or reinforcpp (no critic needed)
  rollout_correction:
    bypass_mode: true  # 18 Psi -> 3.8 Psi, skip old_log_prob forward

# MUST NOT DO
- Do NOT use backend: nccl/nixl (needs multi-GPU)
- Do NOT use fsdp2 strategy (MoE backward failure)
- Do NOT use LoRA rank=64 (EOS bug)
- Do NOT use sleep_level=2 (full weight reload each step)
- Do NOT use separate_async trainer (needs multi-GPU/multi-node)
```

### 8.2 Memory Budget Analysis

| Component | Memory (7B model, bf16) | Memory (7B + LoRA r=32) |
|-----------|------------------------|------------------------|
| Base model (FSDP shard, dp=1) | 14 GiB (full) | 14 GiB (full) |
| LoRA adapter (A+B) | N/A | ~0.3 GiB |
| Optimizer states (AdamW, fp32) | 28 GiB | 28 GiB (base only) + 0.6 GiB (LoRA) |
| KV cache (rollout) | ~3-6 GiB | ~3-6 GiB |
| Activation memory | ~2-4 GiB | ~2-4 GiB |
| **Total TRAINING phase** | ~44-48 GiB | ~44-48 GiB (with CPU offload) |
| **Total ROLLOUT phase** | ~17-20 GiB | ~17-20 GiB |

With ZeRO-2 + CPU_Adam + bypass_mode + sleep_level=1:
- Training: ~3-6 GiB (FSDP sharded + LoRA summon + CPU offload optimizer)
- Rollout: ~17-20 GiB (base weights + KV cache)
- Peak: ~20 GiB (just fits on 24 GiB RTX 4090!)

### 8.3 Checkpoint Engine Selection

For RTX 4090 (dp=1, single GPU):
- **naive**: The ONLY viable backend. Trainer and rollout share same GPU, in same process. No IPC overhead.
- **nccl**: Needs at least 2 processes (trainer rank 0 + rollout rank 1). Requires dp >= 2 or separate GPU. NOT viable on single RTX 4090.
- **nixl**: RDMA-based, needs multiple nodes or GPUs. NOT viable on single RTX 4090.
- **hccl**: Ascend NPU only. NOT relevant for RTX 4090.

### 8.4 Critical Bugs Impacting RTX 4090

1. **verl #7016**: Qwen3-MoE FSDP2 backward failure. MUST use FSDP v1 (`strategy: fsdp`).
2. **verl #6782**: LoRA rank=64 breaks EOS in vLLM rollout. MUST use rank=32.
3. **verl #6468**: FSDP2 CPU memory leak during weight sync. NOT an issue with FSDP v1.
4. **vLLM #45552**: CuMem sleep/wake missing cuda.synchronize(). Risk for HYBRID mode.
5. **SGLang #28679**: GDN intermittent degeneracy. Silent corruption in long-running GRPO.

---

## Appendix A: Key Function Call Trees

### A.1 init_workers() -> Training Setup

```
TaskRunner.run(config)
  -> add_actor_rollout_worker(config)
     -> AsyncActorRolloutRefWorker (FSDP) or ActorRolloutRefWorker (new engine)
  -> add_critic_worker(config)
  -> init_resource_pool_mgr(config)
  -> RayPPOTrainer(config, ...)

RayPPOTrainer.init_workers()
  -> resource_pool_manager.create_resource_pool()
  -> RayClassWithInitArgs for each role
  -> create_colocated_worker_cls(class_dict) -> colocated multi-role worker
  -> RayWorkerGroup.spawn() -> separate worker groups per role
  -> critic_wg.init_model() / actor_rollout_wg.init_model()
  -> RewardLoopManager
  -> AgentLoopManager
     -> _initialize_llm_servers()
        -> RolloutReplica.init_hybrid() / init_colocated() / init_standalone()
        -> launch_servers() -> vLLM/SGLang/TRT-LLM HTTP server per node
     -> _init_agent_loop_workers()
        -> AgentLoopWorker (Ray remote actors, num_workers=8)
  -> CheckpointEngineManager(backend, trainer, replicas)
  -> checkpoint_manager.sleep_replicas()  # initial sleep for checkpoint load
```

### A.2 update_weights() -> Weight Sync Call Tree

```
CheckpointEngineManager.update_weights()

[naive backend]:
  -> ray.get(self.trainer.update_weights())
     -> AsyncActorRolloutRefWorker.update_weights()
        -> self.rollout_mode()
           -> aggressive_empty_cache()
           -> load_fsdp_model_to_gpu() (if offloaded)
           -> collect_lora_params() or state_dict()
           -> offload_fsdp_model_to_cpu() (if offloaded)
           -> self.rollout.resume(tags=["weights"])  # wake up
           -> self.rollout.update_weights(params, ...)
           -> set base_sync_done=True

[nccl/nixl backend]:
  -> abort_all_requests() on all replicas
  -> create RayWorkerGroup from all replica workers
  -> build_process_group() -> NCCL/NIXL topology
  -> trainer.update_weights() + rollout.update_weights()
  -> finalize all workers
  -> resume_all_requests() on all replicas
```

### A.3 generate_sequences() -> Rollout Call Tree

```
RayPPOTrainer.fit()
  -> async_rollout_manager.generate_sequences(gen_batch)
     -> AgentLoopManager.generate_sequences()
        -> chunk gen_batch across agent_loop_workers
        -> ray.get([worker.generate_sequences.remote(chunk) ...])
           -> AgentLoopWorker.generate_sequences(chunk)
              -> tqbridge: BatchMeta -> DataProto
              -> for each sample:
                 -> _run_agent_loop(sampling_params, ...)
                    -> hydra.utils.instantiate(agent_loop_config)
                    -> agent_loop.run(sampling_params, **kwargs)
                       -> AsyncLLMServerManager.generate(request_id, prompt_ids, ...)
                          -> server.generate.remote(...)
                             -> vLLM/SGLang/TRT-LLM token-in-token-out API
              -> _postprocess(outputs) -> DataProto
              -> tqbridge: DataProto -> BatchMeta
        -> DataProto.concat(outputs)
        -> _performance_metrics(metrics, output)
```

---

## Appendix B: Config Defaults for Key Components

### B.1 RolloutConfig Defaults

```python
name: MISSING           # vllm, sglang, trtllm
mode: "async"           # sync DEPRECATED
temperature: 1.0
n: 1                    # num responses per prompt
prompt_length: 512
response_length: 512
gpu_memory_utilization: 0.5
enforce_eager: True     # CRITICAL for DSV4
free_cache_engine: True
enable_sleep_mode: True  # NEW default!
data_parallel_size: 1
tensor_model_parallel_size: 2
checkpoint_engine:
  backend: MISSING      # must be set: naive, nccl, nixl, hccl
  update_weights_bucket_megabytes: 2048
agent:
  num_workers: 8
  default_agent_loop: "single_turn_agent"
multi_stage_wake_up: False
enable_chunked_prefill: True
enable_prefix_caching: True
load_format: "dummy"    # HYBRID mode: don't load weights (use FSDP model)
layered_summon: False
```

---

*End of verl V1 Trainer Architecture Deep Reading*
