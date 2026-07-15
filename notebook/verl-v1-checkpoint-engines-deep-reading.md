# verl V1 Checkpoint Engines Deep Reading

**Date**: 2026-07-15
**Source**: volcengine/verl main branch
**Focus**: 6 checkpoint engine architecture, naive engine for RTX 4090 dp=1, weight sync interaction with GRPO training

---

## 1. Architecture Overview

### 1.1 Dual "Checkpoint" Concepts in verl

verl uses "checkpoint" in **two distinct senses** that must not be confused:

1. **Checkpoint Engine** (`verl/checkpoint_engine/`): The mechanism for **weight synchronization** between trainer (actor) and rollout workers. This is NOT about saving model state to disk for fault tolerance -- it is about transferring updated model weights from training processes to inference processes during live GRPO training.

2. **Checkpoint Manager** (`verl/utils/checkpoint/`): The mechanism for **saving/loading model state to disk** for fault tolerance and training resumption. This is the traditional checkpoint concept (model weights, optimizer state, lr_scheduler, RNG state saved to filesystem).

The **Checkpoint Engine** is the focus of this deep reading. It is the critical mechanism for GRPO training weight sync, and the choice of backend directly impacts RTX 4090 memory overhead and throughput.

### 1.2 Registry Pattern

All 6 checkpoint engines use a **registry pattern** via `CheckpointEngineRegistry`:

```python
class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):  # decorator
        def wrapper(cls):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls
        return wrapper

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        return cls._registry[backend](*args, **kwargs)
```

Registration calls in each engine file:
- `@CheckpointEngineRegistry.register("naive")` -> `ColocatedCheckpointEngine`
- `@CheckpointEngineRegistry.register("nccl")` -> `NCCLCheckpointEngine`
- `@CheckpointEngineRegistry.register("nccl")` -> `HCCLCheckpointEngine` (note: same registry key as NCCL! HCCL overrides NCCL on Ascend)
- `@CheckpointEngineRegistry.register("nixl")` -> `NIXLCheckpointEngine`
- `@CheckpointEngineRegistry.register("kimi_ckpt_engine")` -> `KIMICheckpointEngine`
- `@CheckpointEngineRegistry.register("mooncake")` -> `MooncakeCheckpointEngine`

All engines except naive are imported conditionally (`try/except ImportError`) so they only register when their dependencies are available.

### 1.3 CheckpointEngineConfig (YAML Config)

```python
@dataclass
class CheckpointEngineConfig(BaseConfig):
    _mutable_fields = {"backend"}
    backend: Optional[str] = "naive"          # naive, nccl, nixl, hccl, kimi_ckpt_engine, mooncake
    update_weights_bucket_megabytes: int = 2048  # bucket size in MB
    engine_kwargs: dict = field(default_factory=dict)  # per-backend extra kwargs
    custom_backend_module: Optional[str] = None  # plugin backend registration
```

Default backend is `"naive"` -- this is the **colocated (in-process) weight pass-through** used for sync training.

---

## 2. The 6 Checkpoint Engines: Detailed Analysis

### 2.1 naive (ColocatedCheckpointEngine) -- BEST for RTX 4090 dp=1

**Class**: `ColocatedCheckpointEngine` registered as `"naive"`

**How it works**: This engine is a **zero-overhead, in-process weight generator**. It does NOT use any distributed communication library (no NCCL, no RDMA, no ZMQ). When actor and rollout are **colocated on the same GPU**, weights are simply passed through as a Python generator object:

```python
class ColocatedCheckpointEngine(CheckpointEngine):
    def send_weights(self, weights, global_steps=None):
        self.weights = weights  # just store the generator reference

    def receive_weights(self, global_steps=None):
        yield from self.weights  # yield from stored generator
        self.weights = None      # cleanup
```

**Key properties**:
- `prepare()`, `init_process_group()`, `finalize()`, `build_topology()` all raise `NotImplementedError` -- no distributed setup needed
- Zero additional GPU memory overhead (no send_buf/recv_buf allocation)
- Zero network/IPC overhead
- Asynchronous but trivially so (just a generator reference)
- Only works when actor and rollout are in the **same process** (colocated)

**Why it is BEST for RTX 4090 dp=1**:
1. **No additional memory**: All other engines allocate 2 * bucket_size bytes of GPU memory for send_buf and recv_buf. With `update_weights_bucket_megabytes=2048`, that is **4 GiB** of extra GPU memory overhead! On a 24 GiB RTX 4090, that is 16.7% of total memory wasted on transfer buffers.
2. **No NCCL process group**: NCCL initialization on dp=1 has overhead and can cause issues (see DeepSpeed #8061 overlap_comm data race pattern).
3. **No ZMQ/RDMA**: No additional CPU threads, no IPC sockets, no RDMA registration.
4. **Fastest possible**: Direct generator yield is O(1) overhead. The actual weight copy happens in the rollout's `update_weights()` method, which uses direct CUDA operations on the same GPU.

**RTX 4090 memory budget impact**:
- With naive: 0 bytes additional overhead. All 24 GiB available for model, KV cache, activations.
- With NCCL/nixl/mooncake: 2 * bucket_size = 2 * 2048 MiB = 4 GiB overhead (send_buf + recv_buf).
- With kimi_ckpt_engine: Additional CPU offload + RDMA registration overhead.

### 2.2 nccl (NCCLCheckpointEngine) -- NVIDIA GPU Off-Policy

**Class**: `NCCLCheckpointEngine` registered as `"nccl"`

**How it works**: Uses NCCL collective broadcast to transfer weights from actor rank 0 to all rollout ranks. Communication uses:
- **NCCL broadcast** for tensor data (bucket-based)
- **ZeroMQ PUB/SUB** for bucket metadata (tensor names, shapes, offsets)

**Architecture**:
```
Actor side:
  ME0 (rank 0) -> CE0 -> NCCL broadcast -> rollout ranks 1..N
  ME1..N (rank -1) -> CE1..N (consume weights, no send)
```

**Topology**: Star topology -- actor rank 0 is the master, all rollout ranks receive via NCCL broadcast. Actor ranks >0 are assigned rank=-1 (spectator mode, they just consume the generator to free memory).

**Bucket protocol**:
1. Actor fills `send_buf` (cupy array) with weight chunks
2. After bucket is full, sync CUDA, start `BroadcastOperation` in separate thread
3. Metadata sent via ZMQ PUB/SUB (bucket_meta dict with TensorMeta per tensor)
4. NCCL broadcast of the bucket buffer
5. Double-buffering: swap send_buf/recv_buf for pipelined transfer
6. Rollout receives in recv_buf, yields tensors from send_buf, pipelines next bucket

**Key properties**:
- `bucket_size`: Default 2048 MiB. Memory overhead = 2 * bucket_size on each process (send_buf + recv_buf)
- Uses **cupy** on master rank to avoid PyTorch expandable_segments memory registration conflict
- `rebuild_group`: Optional rebuild of NCCL group per update (for elastic scaling)
- `rollout_dtype`: Cast weights to target dtype (default bf16)
- ZMQ PUB/SUB for metadata: Master publishes, subscribers receive
- `BroadcastOperation`: Async broadcast in executor thread (non-blocking for the asyncio event loop)

**Elastic scaling**: `rebuild_group=True` destroys and recreates NCCL group each update -- allows adding/removing rollout replicas. Default `False` for fixed clusters.

**Benchmark**: ~7s for Qwen3-30B-A3B, 8.25 GB/s bandwidth on 4x8 H100 with ConnectX-7 400 Gbps InfiniBand.

### 2.3 hccl (HCCLCheckpointEngine) -- Ascend NPU Off-Policy

**Class**: `HCCLCheckpointEngine` registered as `"nccl"` (same registry key! Overrides NCCL on Ascend)

**How it works**: HCCL equivalent of NCCLCheckpointEngine. Uses:
- **HCCL broadcast** (via `StatelessProcessGroup`) for tensor data
- **ZeroMQ PUB/SUB** for bucket metadata
- **torch.npu** device operations

**Key difference from NCCL**: Uses `StatelessProcessGroup` (from vLLM) instead of `ray.util.collective` for HCCL communication. Barrier uses HCCL all_reduce of a signal tensor.

**Note**: This engine registers under the `"nccl"` key, meaning on Ascend NPUs, `backend="nccl"` in YAML actually uses HCCL. This is a design choice for transparent backend selection.

**Benchmark**: ~11s for Qwen3-30B-A3B, 5.3 GB/s on 2x16 Ascend 910C.

### 2.4 nixl (NIXLCheckpointEngine) -- P2P Ring, Elastic, Multi-Backend

**Class**: `NIXLCheckpointEngine` registered as `"nixl"`

**How it works**: Uses NVIDIA's NIXL (NVIDIA Inference Xfer Library) for P2P transfers in a **ring topology**:
- Ring topology: rank 0 -> rank 1 -> rank 2 -> ... -> rank N-1 -> rank N (each rank sends to next, receives from previous)
- Supports multiple transport backends: UCX, UCCL, Mooncake, etc.
- Uses NixlAgent wrapper for registration, metadata exchange, and async transfer operations

**Architecture**:
```
Actor (rank 0) -> NIXL P2P write -> Rollout rank 1 -> NIXL P2P write -> Rollout rank 2 -> ... -> Rollout rank N
                    <- NIXL P2P notification <-         <- NIXL P2P notification <-        <- NIXL P2P notification <-
```

**Key properties**:
- **Ring topology** instead of star/broadcast: Better for elastic scaling (adding/removing nodes only affects local ring segment)
- **High elasticity**: Dynamic ring topology adjustment without rebuilding full process group
- Supports **multiple transport backends** (UCX, UCCL, Mooncake) via NIXL abstraction
- Uses ZMQ async (zmq.asyncio) for metadata exchange between ring neighbors
- `ReadableOperation` and `ReadOperation` for async NIXL RDMA transfers
- Each rank only needs to know prev_agent and next_agent -- minimal topology knowledge
- Double-buffering with send_buf/recv_buf swap for pipelined transfers

**Benchmark**: ~7s for Qwen3-30B-A3B, 8.25 GB/s on 4x8 H100 (same as NCCL on high-bandwidth InfiniBand).

### 2.5 kimi_ckpt_engine (KIMICheckpointEngine) -- Mooncake+Broadcast Hybrid

**Class**: `KIMICheckpointEngine` registered as `"kimi_ckpt_engine"`

**How it works**: Two-phase transfer:
1. **Phase 1 (CPU offload)**: Actor offloads all weights to CPU using ThreadPoolExecutor (32 threads) with non_blocking=True
2. **Phase 2 (P2P + Broadcast)**: Uses Mooncake Transfer Engine's ParameterServer for P2P transfer to a specific rollout worker, then NCCL/HCCL broadcast within rollout subgroup

**Architecture**:
```
Actor ranks: weights -> CPU offload -> register_checkpoint -> gather_metas
Rollout:     parameter_server.receive_tensor -> P2P from actor rank -> broadcast within rollout group
```

**Key properties**:
- **All actor ranks participate** (rank 0..actor_wg_world_size-1), not just rank 0
- Offloads to CPU first: Reduces GPU memory pressure during transfer
- Uses `checkpoint_engine` external package (pip install `checkpoint-engine[p2p]`, version >= 0.4.0)
- Requires Mooncake Transfer Engine (compiled from source for Ascend)
- `ParameterServer` with RDMA registration for P2P transfers
- Bucket-based sharding: weights distributed across actor ranks by tensor_idx % world_size
- Broadcast within rollout subgroup using NCCL/HCCL
- `rollout_ranks` and `rollout_group` for intra-rollout broadcast

**Benchmark**: offload: 7s, update: 3.5s, 16.5 GB/s effective on 2x16 Ascend 910C (highest bandwidth on Ascend due to CPU offload reducing GPU contention).

**Ascend Note**: CANN >= 8.5.0 requires `HCCL_INTRA_ROCE_ENABLE=1` for intra-node RoCE. Not supported on A5 NPUs.

### 2.6 mooncake (MooncakeCheckpointEngine) -- RDMA Ring P2P

**Class**: `MooncakeCheckpointEngine` registered as `"mooncake"`

**How it works**: Uses Mooncake TransferEngine for RDMA P2P transfers in ring topology:
- Ring topology similar to NIXL
- RDMA write for data transfer, RDMA read for receiving
- Uses `StatelessProcessGroup` (from vLLM or SGLang) for topology coordination
- Magic bytes protocol for completion signaling (0xAB, 0xDC, 0xEF, 0x88)

**Architecture**:
```
Rank 0 (actor): fill bucket -> RDMA write to rank 1 -> wait magic completion
Rank 1 (rollout): RDMA read from rank 0 -> forward metadata to rank 2 -> RDMA write magic completion signal to rank 0
...
Rank N-1 (last rollout): RDMA read from rank N-2 -> yield tensors -> RDMA write magic completion
```

**Key properties**:
- Ring P2P topology: rank 0 sends, each subsequent rank reads from previous and forwards metadata to next
- RDMA transfer_sync_read and transfer_sync_write for zero-copy GPU-to-GPU transfers
- Magic completion signal via dedicated `magic_recv` buffer (8 bytes) to prevent data corruption from local GPU side-effects
- Double-buffering: 2 * bucket_size contiguous allocation (`self.buf = torch.empty(2 * bucket_size)`)
- `device_name` filter for Mooncake RDMA device selection
- Supports both NVIDIA (`rdma`) and Ascend (`ascend_direct`) transport
- `StatelessProcessGroup.create` for TCPStore-based coordination

**Benchmark**: 5.93s, 9.44 GB/s on 2x8 H100 ConnectX-7 InfiniBand.

---

## 3. CheckpointEngineManager: The Coordinator

### 3.1 Role in V1 Trainer

`CheckpointEngineManager` is the top-level coordinator that manages the **full weight sync lifecycle** between actor worker group and rollout replicas. It is created in `PPOTrainer._setup()`:

```python
# In trainer_base.py _setup():
checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
checkpoint_engine_config.backend = "naive"  # HARDCODED to naive in base class!
self.checkpoint_manager: CheckpointEngineManager = CheckpointEngineManager(
    config=checkpoint_engine_config,
    actor_wg=self.actor_rollout_wg,
    replicas=self.llm_server_manager.get_replicas(),
)
```

**CRITICAL NOTE**: The base `PPOTrainer._setup()` hardcodes `backend = "naive"`. This is overridden in `PPOTrainerSeparateAsync._setup()`, which does NOT hardcode -- it uses the config value. The sync and colocate_async trainers always use naive because actor and rollout are colocated.

### 3.2 update_weights() Flow

The `update_weights()` method in `CheckpointEngineManager` has a **split path** based on backend:

```python
async def update_weights(self, global_steps=None):
    # 0. NAIVE PATH: direct in-process update
    if self.backend == "naive":
        ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
        return

    # 1-8. DISTRIBUTED PATH for nccl/nixl/kimi/mooncake:
    # 1. abort all unfinished requests (partial rollout)
    await self.abort_replicas()
    # 2. create temporary worker group for all replicas
    rollout = RayWorkerGroup(worker_handles=..., ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
    # 3. release kv_cache before weight sync (weights stay in place)
    await self.release_kv_cache_replicas()
    # 4. build process group (prepare + build_topology + init_process_group)
    self.build_process_group(rollout)
    # 5. update weights of all workers
    ray.get(actor_wg.update_weights(...) + rollout.update_weights(...))
    # 6. finalize all workers
    ray.get(actor_wg.execute_checkpoint_engine(["finalize"]*...) + rollout.execute_checkpoint_engine(["finalize"]*...))
    # 7. restore kv_cache after weight sync
    await self.resume_kv_cache_replicas()
    # 8. resume all unfinished requests (partial rollout)
    await self.resume_generation_replicas()
```

**Naive path**: Just calls `actor_wg.update_weights(mode="naive")` -- a single Ray call. No abort, no process group, no kv_cache release/resume.

**Distributed path**: 8-step pipeline requiring KV cache release before weight sync (to free GPU memory for NCCL buffers), process group construction, weight transfer, finalization, and KV cache restoration.

### 3.3 Sleep/Wake Lifecycle

The `CheckpointEngineManager` also manages sleep/wake lifecycle for memory optimization:

```python
async def sleep_replicas(self):
    """Free weight and kv_cache device memory on all rollout replicas."""
    await asyncio.gather(*[r.sleep() for r in self.replicas])

async def wake_up_replicas(self):
    """Recover kv_cache and weights device memory on all rollout replicas."""
    await asyncio.gather(*[r.wake_up() for r in self.replicas])
```

Each trainer mode uses sleep/wake differently:

| Trainer Mode | Sleep/Wake Pattern | When Sleep | When Wake |
|---|---|---|---|
| **sync** | sleep after sampling, wake+update after step | `on_sample_end()` | `on_step_end()` via `update_weights()` |
| **colocate_async** | abort+sleep after sampling, wake+update+resume after step | `on_sample_end()` | `on_step_end()` via `update_weights()` + `resume_generation_replicas()` |
| **separate_async** | hybrid engine switches modes: sleep when switching to trainer, wake when switching to rollout | `switch_to_trainer()` | `switch_to_rollout()` |

### 3.4 KV Cache Release/Resume (Distributed Engines Only)

For distributed checkpoint engines (nccl/nixl/mooncake), the weight sync needs GPU memory for NCCL/RDMA buffers. The solution is `release_kv_cache_replicas()` / `resume_kv_cache_replicas()`:

```python
async def release_kv_cache_replicas(self):
    """Release kv_cache before NCCL weight sync. Weights stay untouched."""
    await asyncio.gather(*[r.release_kv_cache() for r in self.replicas])

async def resume_kv_cache_replicas(self):
    """Restore kv_cache after NCCL weight sync."""
    await asyncio.gather(*[r.resume_kv_cache() for r in self.replicas])
```

This is a **partial sleep**: only KV cache is freed, model weights remain on GPU. This is different from `sleep_replicas()` which frees BOTH weights and KV cache. The partial release is needed because NCCL/RDMA writes directly into the existing weight buffers, so weights must stay resident.

---

## 4. Weight Sync in GRPO Training: Trainer-Specific Flows

### 4.1 PPOTrainerSync (Sync Trainer)

```python
@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)  # initial weight sync

    def on_step_end(self):
        with marked_timer("update_weights", ...):
            self.checkpoint_manager.update_weights(self.global_steps)  # per-step weight sync

    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas()  # free weights+KV cache for training
```

**Flow per GRPO step**:
1. `on_sample_begin()`: Rollout generates prompts (weights+KV cache are resident from wake)
2. `on_sample_end()`: Sleep replicas (free weights+KV cache) -- GPU memory for training
3. Training: Actor update, critic update, reference compute -- uses freed GPU memory
4. `on_step_end()`: `update_weights()` -> naive path -> `actor_wg.update_weights(mode="naive")`

In the **naive path** within `ActorRolloutRefWorker.update_weights()`:
1. Resume rollout weights (if `free_cache_engine=True`: `rollout.resume(tags=["weights"])`)
2. `actor.engine.get_per_tensor_param()` -- get weights as generator from FSDP/Megatron
3. `rollout.update_weights(per_tensor_param, peft_config, ...)` -- direct CUDA weight copy
4. Offload actor to CPU if param_offload enabled
5. Resume KV cache (`rollout.resume(tags=["kv_cache"])`)
6. `base_sync_done = True` for LoRA two-phase sync

**LoRA two-phase sync** (naive path): If `lora.merge=False` and `peft_config is not None`:
- First call: base weight sync only (`base_sync_done=False`)
- Subsequent calls: LoRA adapter sync only (`base_sync_done=True`)
- `sleep_level=1` for LoRA (only adapter changes, base weights stay resident)

### 4.2 PPOTrainerColocateAsync (Colocate Async Trainer)

```python
@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights", ...):
            self.checkpoint_manager.update_weights(self.global_steps)
            self.checkpoint_manager.resume_generation_replicas()  # resume partial rollout

    def on_sample_end(self):
        self.checkpoint_manager.abort_replicas()    # abort in-flight requests
        self.checkpoint_manager.sleep_replicas()    # free weights+KV cache
```

Same naive backend, but adds **abort/resume** for partial rollout (in-flight requests are paused during weight sync, then resumed).

### 4.3 PPOTrainerSeparateAsync (Separate Async Trainer)

```python
@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    def __init__(self, config):
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive"
        # MUST use nccl/nixl/mooncake for separate async

    def _setup(self):
        super()._setup()
        # Create standalone rollout replicas (NOT colocated with actor)
        self.standalone_server_manager = LLMServerManager.create(...)
        # Create SEPARATE checkpoint engine manager for standalone rollout
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,  # NOT hardcoded to naive!
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )
```

This trainer has **TWO CheckpointEngineManagers**:
1. `self.checkpoint_manager` (naive, base class) -- for **colocated hybrid** replicas
2. `self.standalone_checkpoint_manager` (nccl/nixl/mooncake) -- for **disaggregated standalone** replicas

The hybrid engine switches between trainer and rollout modes:
- `switch_to_rollout()`: wake colocated replicas, update weights, add to load balancer
- `switch_to_trainer()`: remove from load balancer, abort, sleep colocated replicas

---

## 5. Bucket Transfer Protocol Details

### 5.1 Weight Chunking

All engines that transfer over network/IPC use a **bucket-based chunking protocol**. Large tensors that exceed bucket_size are split into chunks:

```python
async def split_weight_chunks(weights, bucket_size):
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(name=name, shape=weight.shape, dtype=weight.dtype,
                                     chunk_offset=chunk_offset, chunk_size=chunk_size, offset=None)
            yield (tensor_meta, buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size
```

And reassembled on the receiving end:

```python
async def merge_weight_chunks(chunks, bucket_size):
    # Small weights: yield directly
    # Large weights: accumulate chunks into pre-allocated buffer, yield when complete
```

### 5.2 TensorMeta

Each weight tensor is described by `TensorMeta`:
```python
@dataclass
class TensorMeta:
    name: str           # weight name (HF key)
    shape: torch.Size   # tensor shape
    dtype: torch.dtype  # tensor dtype (e.g., bf16)
    chunk_offset: int   # byte offset within original tensor
    chunk_size: int     # byte size of this chunk
    offset: int         # byte offset within the bucket buffer
```

### 5.3 Double-Buffering

All network engines use **double-buffering** (send_buf + recv_buf swap):
1. Fill send_buf with current bucket
2. Start transfer of send_buf (broadcast/RDMA/P2P)
3. While transfer is in-flight, swap buffers: recv_buf becomes next send_buf
4. When transfer completes, previous send_buf (now recv_buf) is ready for next bucket
5. Pipeline: transfer bucket N while filling bucket N+1

This reduces total transfer time by overlapping compute and communication, but costs 2 * bucket_size GPU memory.

---

## 6. Engine Comparison Summary

| Engine | Backend | Topology | Hardware | Elastic | Memory Overhead | Best For |
|--------|---------|----------|----------|---------|-----------------|----------|
| **naive** | torch.distributed (in-process) | all_gather (same process) | NVIDIA/AMD/Ascend | NA | **0 bytes** | **RTX 4090 dp=1 colocated GRPO** |
| **nccl** | NCCL broadcast + ZMQ PUB/SUB | Star (rank 0 -> all) | NVIDIA GPU | Low (rebuild_group) | 2 * bucket_size | Multi-GPU off-policy, fixed cluster |
| **hccl** | HCCL broadcast + ZMQ PUB/SUB | Star (rank 0 -> all) | Ascend NPU | Low (rebuild_group) | 2 * bucket_size | Ascend NPU off-policy |
| **nixl** | NIXL P2P ring | Ring (rank 0 -> 1 -> 2 -> ... -> N) | Multi-backend | **High** (dynamic ring) | 2 * bucket_size | Elastic rollout, fault tolerance, heterogeneous hardware |
| **kimi_ckpt_engine** | Mooncake P2P + NCCL/HCCL broadcast | P2P to one + broadcast to all | NVIDIA/Ascend | Low (rebuild_group) | CPU offload + 2 * bucket_size | Ascend high-bandwidth, save checkpoint each time |
| **mooncake** | Mooncake RDMA ring P2P | Ring (rank 0 -> 1 -> ... -> N) | NVIDIA/Ascend | **High** (dynamic ring) | 2 * bucket_size | RDMA clusters, elastic, multi-node |

**Benchmark (Qwen3-30B-A3B)**:
| Hardware | Backend | Time (s) | Bandwidth (GB/s) |
|----------|---------|----------|-------------------|
| 4x8 H100, IB 400Gbps | NCCL | ~7 | 8.25 |
| 4x8 H100, IB 400Gbps | NIXL | ~7 | 8.25 |
| 2x8 H100, IB 400Gbps | mooncake | 5.93 | 9.44 |
| 2x16 Ascend 910C | HCCL | ~11 | 5.3 |
| 2x16 Ascend 910C | kimi_ckpt_engine | offload:7 + update:3.5 | 16.5 |

---

## 7. RTX 4090 dp=1 Recommendations

### 7.1 MUST USE naive Backend

For RTX 4090 dp=1 GRPO training, the **naive (ColocatedCheckpointEngine) is the ONLY viable choice**:

1. **Zero memory overhead**: No send_buf/recv_buf allocation. On 24 GiB GPU, every byte counts.
2. **Colocated architecture**: Actor and rollout share the same GPU process. Sync trainer (`trainer_mode="sync"`) is the optimal mode.
3. **No distributed overhead**: No NCCL initialization, no process groups, no ZMQ/RDMA threads.
4. **Direct weight pass-through**: Generator reference is O(1) overhead. The actual weight copy is done by the rollout server adapter (vLLM/SGLang) using optimized CUDA operations.

**Why NOT use NCCL/nixl/mooncake on RTX 4090 dp=1**:
- `separate_async` trainer mode requires dp>=2 and nnodes>0. The code explicitly asserts:
  ```python
  assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive"
  assert config.actor_rollout_ref.rollout.nnodes > 0
  ```
- Even if you could use NCCL on dp=1, it would allocate 4 GiB for buffers (16.7% of total memory).
- NCCL broadcast on single GPU is meaningless -- there is only one receiver.

### 7.2 Config Recommendation

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: naive            # MUST be naive for dp=1
      update_weights_bucket_megabytes: 2048  # irrelevant for naive, but keep default

trainer:
  v1:
    trainer_mode: sync          # MUST be sync for dp=1 colocated
```

### 7.3 Memory Budget Analysis (naive vs nccl)

For a 7B bf16 model (~14 GiB parameters) on RTX 4090 (24 GiB):

| Component | naive (GiB) | nccl (GiB) |
|-----------|-------------|------------|
| Model weights (actor) | ~14 | ~14 |
| Model weights (rollout, resident during generation) | ~14 (shared process) | ~14 (separate process) |
| KV cache | ~2-4 | ~2-4 |
| Training activations | ~4-6 | ~4-6 |
| **Checkpoint engine buffers** | **0** | **4** (2*2 GiB) |
| **Total** | ~24 (tight fit) | ~28 (OOM!) |

With naive, actor and rollout are colocated in the same process. The weight sync is a generator reference transfer, not a GPU buffer copy. This means during generation, both actor weights and KV cache share the same process, and during training, weights are offloaded/sleep'd and the same GPU memory is reused.

### 7.4 Weight Sync Flow on RTX 4090 (naive)

```
Step N:
  1. on_sample_begin(): Generate prompts (weights + KV cache resident on GPU)
  2. on_sample_end(): sleep_replicas() -> free weights + KV cache
  3. Training phase: actor.update_mini_batch() (uses freed GPU memory)
  4. on_step_end(): checkpoint_manager.update_weights() -> actor_wg.update_weights(mode="naive")
     -> ActorRolloutRefWorker.update_weights():
        a. rollout.resume(tags=["weights"])  # re-allocate weight memory
        b. actor.engine.get_per_tensor_param()  # FSDP summon, yield generator
        c. rollout.update_weights(per_tensor_param)  # direct CUDA copy (same GPU!)
        d. actor offload to CPU (if param_offload)
        e. rollout.resume(tags=["kv_cache"])  # re-allocate KV cache
  5. Step N+1: repeat
```

The key insight: on RTX 4090 with naive backend, the weight sync is a **local GPU memory operation** (allocate -> copy -> free), not a network/IPC transfer. The `rollout.resume(tags=["weights"])` allocates GPU memory for weights, `rollout.update_weights()` copies from FSDP-summoned params into rollout's weight buffers, and then actor can offload its params to CPU.

---

## 8. Checkpoint Recovery (Disk Persistence)

### 8.1 Checkpoint Manager (NOT Checkpoint Engine)

The disk-based checkpoint system (`verl/utils/checkpoint/`) is separate from the checkpoint engine. It handles:

- **Model state**: FSDP sharded state or Megatron dist_checkpointing shards
- **Optimizer state**: Sharded optimizer state
- **Extra state**: LR scheduler, RNG states
- **HF model**: Full HuggingFace format model (via mbridge for Megatron)

`BaseCheckpointManager` provides:
- `save_checkpoint()`, `load_checkpoint()`
- `ensure_checkpoint_capacity()`, `register_checkpoint()` -- manage max_ckpt_to_keep
- `get_rng_state()`, `load_rng_state()` -- RNG state persistence

### 8.2 Resume Modes

```python
# In PPOTrainer._load_checkpoint():
if self.config.trainer.resume_mode == "disable":
    return  # train from scratch
elif self.config.trainer.resume_mode == "auto":
    # find latest checkpoint in default_local_dir
    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
elif self.config.trainer.resume_mode == "resume_path":
    # explicit path to global_step_N directory
    global_step_folder = self.config.trainer.resume_from_path
```

### 8.3 TransferQueue Checkpoint (Async Modes)

For async trainers, PR #7037 (MERGED July 14) adds **TransferQueue checkpointing**:

```python
if self.trainer_mode != "sync" and _tq_supports_checkpoint():
    tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
    if os.path.exists(tq_ckpt_path):
        tq.load_checkpoint(tq_ckpt_path)
```

This allows restarting pending/running prompt groups after resume, preserving finished samples. The `_tq_supports_checkpoint()` function checks for TransferQueue version >= 0.1.9 with `save_checkpoint` and `load_checkpoint` methods.

### 8.4 Reissue Inflight Prompts

After checkpoint recovery, async trainers can **re-issue inflight prompts**:

```python
def _reissue_inflight_prompts(self, partition_id="train"):
    # Find pending/running prompts in TransferQueue
    inflight_uids = [key for key, tag in items.items()
                     if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")]
    # Clear old trajectories for these prompts
    # Re-dispatch as new generation requests
    self.agent_loop_manager.generate_sequences(batch)
```

### 8.5 FSDP Checkpoint Directory Structure

```
checkpoints/${project_name}/${experiment_name}/
  global_steps_${i}/
    actor/
      huggingface/       # config + tokenizer (optional: hf_model weights)
      fsdp_config.json
      model_world_size_N_rank_R.pt
      optim_world_size_N_rank_R.pt
      extra_state_world_size_N_rank_R.pt
    critic/
      ... (same structure)
  latest_checkpointed_iteration.txt
```

---

## 9. Recent Developments

### 9.1 PR #7037 MERGED (July 14): Streaming Dataloader + Async Checkpoint Recovery

- Supports adding arbitrary number of prompts from dataloader without `train_batch_size` limitation
- Improves async trainers with `drop` strategy: dropping stale prompt groups and refilling new prompts
- Allows restarting pending/running prompts after resume via TransferQueue checkpointing
- TransferQueue version >= 0.1.9 required

### 9.2 PR #7041 MERGED (July 15): Qwen2.5-0.5b Fully Async OOM Fix

- Fixes OOM when allocating `recv_buf` in fully async weight sync
- Root cause: recv_buf requests too much memory OR inference engine occupies too much GPU memory
- Solution: Reduce recv_buf size and inference engine memory proportion
- Relevant for RTX 4090: confirms that buffer allocation is a critical memory concern for async modes

### 9.3 Custom Backend Plugin System

The `custom_backend_module` field in `CheckpointEngineConfig` allows registering custom backends:

```python
custom_backend_module: Optional[str] = None  # Python module path
```

Before instantiation, `import_external_libs()` imports this module, which can register additional backends in `CheckpointEngineRegistry`. This enables organization-specific checkpoint engines (e.g., custom RDMA backends, proprietary transfer engines).

---

## 10. Key Findings and RTX 4090 Implications

### 10.1 Naive is the Only Viable RTX 4090 Backend

- The naive engine has **zero memory overhead** vs 4 GiB overhead for all distributed engines
- RTX 4090 dp=1 MUST use sync trainer mode, which requires colocated actor+rollout
- Colocated architecture = naive backend is the natural and optimal choice
- The `separate_async` trainer explicitly **asserts backend != "naive"**, confirming that separate async is not viable on dp=1

### 10.2 Weight Sync Memory Budget is Critical

On RTX 4090 24 GiB:
- Model weights (7B bf16): ~14 GiB
- KV cache (generation): ~2-4 GiB
- Training activations: ~4-6 GiB
- Checkpoint engine buffers: 0 (naive) or 4 GiB (nccl/nixl/mooncake)
- The naive engine's zero overhead makes the difference between viable and OOM

### 10.3 Sleep/Wake Architecture Matters

The sleep/wake lifecycle for naive backend:
- Sleep: Free BOTH weights and KV cache (full GPU memory available for training)
- Wake: Resume weights first, update, then resume KV cache
- This is the same as the HYBRID sleep/wake pattern documented in verl architecture notes

### 10.4 LoRA Two-Phase Sync

For LoRA GRPO on RTX 4090:
- First step: base weight sync (full model weights transferred)
- Subsequent steps: LoRA adapter sync only (sleep_level=1, base weights stay resident)
- This reduces weight sync payload by ~80x (only adapter deltas, not full model)
- The naive engine handles this seamlessly via `base_sync_done` flag in `update_weights()`

### 10.5 PR #7041 OOM Fix Confirms Memory Pressure

The OOM fix for recv_buf allocation in async mode confirms that buffer sizing is critical. On RTX 4090, even the naive backend's `rollout.resume(tags=["weights"])` allocates weight memory, and the timing of allocation/deallocation must be carefully managed.

---

## 11. Source Files Reference

| File | Purpose |
|------|---------|
| `verl/checkpoint_engine/__init__.py` | Registry exports, conditional imports |
| `verl/checkpoint_engine/base.py` | CheckpointEngine ABC, ColocatedCheckpointEngine (naive), CheckpointEngineRegistry, CheckpointEngineManager, CheckpointEngineWorker, TensorMeta, bucket utilities |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | NCCL broadcast + ZMQ PUB/SUB engine |
| `verl/checkpoint_engine/hccl_checkpoint_engine.py` | HCCL broadcast + ZMQ PUB/SUB engine (Ascend) |
| `verl/checkpoint_engine/nixl_checkpoint_engine.py` | NIXL P2P ring engine |
| `verl/checkpoint_engine/kimi_checkpoint_engine.py` | Mooncake P2P + broadcast hybrid engine |
| `verl/checkpoint_engine/mooncake_checkpoint_engine.py` | Mooncake RDMA ring P2P engine |
| `verl/checkpoint_engine/README.md` | Official documentation, benchmark results |
| `verl/workers/config/rollout.py` | CheckpointEngineConfig dataclass |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker.update_weights() -- naive and distributed paths |
| `verl/trainer/ppo/v1/trainer_base.py` | PPOTrainer._setup() (naive hardcoded), _load_checkpoint(), _save_checkpoint() |
| `verl/trainer/ppo/v1/trainer_sync.py` | PPOTrainerSync (sleep after sample, update after step) |
| `verl/trainer/ppo/v1/trainer_colocate_async.py` | PPOTrainerColocateAsync (abort+sleep, update+resume) |
| `verl/trainer/ppo/v1/trainer_separate_async.py` | PPOTrainerSeparateAsync (assert backend!=naive, two CheckpointEngineManagers) |
| `verl/utils/checkpoint/checkpoint_manager.py` | BaseCheckpointManager (disk persistence) |
| `docs/advance/checkpoint.rst` | Official checkpoint documentation |
