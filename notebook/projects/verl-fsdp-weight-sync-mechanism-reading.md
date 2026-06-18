# verl FSDP Weight Sync Mechanism: Source Code Analysis

**Date**: 2026-06-19
**Category**: Source Code Analysis
**Scope**: verl-project/verl main branch + PR #6794 (delta weight sync)
**Purpose**: Understanding trainer->rollout weight synchronization in HYBRID mode on RTX 4090

---

## 1. Architecture Overview: Trainer-to-Rollout Weight Sync Pipeline

verl's weight sync in HYBRID mode follows a precise sequence orchestrated by
`ActorRolloutRefWorker.update_weights()` in `verl/workers/engine_workers.py`.

### 1.1 Call Flow (Full Sequence)

```
ActorRolloutRefWorker.update_weights()
  |
  |-- Step 0: Effective mode resolution
  |     mode="auto" -> config.rollout.checkpoint_engine.backend
  |     mode="naive" -> direct in-process sync
  |     mode=other   -> checkpoint engine send_weights (async/disagg)
  |
  |-- Step 1: resume rollout memory (weights)
  |     await self.rollout.resume(tags=["weights"])
  |     [SGLang: resume_memory_occupation(tags=["weights"]) via HTTP]
  |     [vLLM:  wake_up via engine.wake_up()]
  |
  |-- Step 2: Extract params from FSDP trainer
  |     per_tensor_param, peft_config = engine.get_per_tensor_param(
  |         layered_summon=self.layered_summon,
  |         base_sync_done=True
  |     )
  |
  |-- Step 3: Determine LoRA path (merge vs adapter)
  |     if not self.peft_merge and peft_config is not None:
  |         self.rollout.sleep_level = 1   # adapter path
  |         do_lora_base_sync = not self.base_sync_done
  |     else:
  |         sleep_level stays 2 (default merge/no-lora path)
  |
  |-- Step 3a: Optional base weight sync (adapter path, first step only)
  |     if do_lora_base_sync:
  |         per_tensor_param_base, peft_config = engine.get_per_tensor_param(
  |             layered_summon=self.layered_summon, base_sync_done=False
  |         )
  |         await self.rollout.update_weights(
  |             per_tensor_param_base, peft_config=peft_config,
  |             base_sync_done=False, global_steps=...
  |         )
  |
  |-- Step 3b: Main weight update (LoRA adapter or merged weights)
  |     await self.rollout.update_weights(
  |         per_tensor_param, peft_config=peft_config,
  |         base_sync_done=True, global_steps=...
  |     )
  |
  |-- Step 4: Offload trainer model to CPU
  |     if param_offload_enabled: engine.to("cpu", model=True, ...)
  |     aggressive_empty_cache()
  |
  |-- Step 5: Resume rollout kv_cache
  |     await self.rollout.resume(tags=["kv_cache"])
  |
  |-- Step 6: Mark base_sync_done=True (persists across steps)
```

### 1.2 Key Source Files

| File | Role |
|------|------|
| `verl/workers/engine_workers.py` | **Orchestrator**: `ActorRolloutRefWorker.update_weights()` coordinates the full sync cycle |
| `verl/workers/engine/fsdp/transformer_impl.py` | **FSDP Engine**: `FSDPEngine.get_per_tensor_param()` extracts weights from sharded model |
| `verl/utils/fsdp_utils.py` | **FSDP Utils**: `collect_lora_params()`, `layered_summon_lora_params()`, `merged_lora_context()`, `replace_lora_wrapper()` |
| `verl/workers/rollout/base.py` | **Rollout Interface**: `BaseRollout` defines `update_weights()`, `resume()`, `release()` |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | **SGLang Rollout**: `ServerAdapter` implements SGLang-specific weight update, LoRA adapter load/unload |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | **vLLM Rollout**: `vLLMHttpServer` implements vLLM sleep/wake, ZMQ IPC weight update |
| `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | **IPC Mechanism**: `BucketedWeightSender/Receiver` for ZMQ-based CUDA IPC or shared memory |
| `verl/workers/rollout/sglang_rollout/http_server_engine.py` | **SGLang HTTP**: `AsyncHttpServerAdapter` for HTTP-based weight update, sleep/wake |
| `verl/workers/rollout/sglang_rollout/utils.py` | **SGLang Utils**: `get_named_tensor_buckets()` for bucketed weight streaming |
| `verl/workers/rollout/delta_sync/` | **Delta Sync** (PR #6794): `DeltaState`, `encode.py`, `wrapper.py` for sparse weight updates |

---

## 2. FSDP Summon Mechanics

### 2.1 `get_per_tensor_param()` — The Core Extraction Function

Located in `verl/workers/engine/fsdp/transformer_impl.py` (line 817).

This function is called twice per weight sync cycle when LoRA is in adapter mode:
- First call with `base_sync_done=False`: extracts base model weights for initial sync
- Second call with `base_sync_done=True`: extracts LoRA-only deltas for per-step updates

**Key logic flow**:

```python
def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
    # 1. Load FSDP model to GPU (unless FSDP2 CPUOffloadPolicy manages placement)
    if not self._uses_fsdp2_cpu_offload_policy:
        load_fsdp_model_to_gpu(self.module)

    peft_config = None
    merge_lora = self.model_config.lora.get("merge", False)

    # 2. Branch on LoRA mode
    if hasattr(peft_model, "peft_config"):  # LoRA model
        if not merge_lora:
            # ADAPTER PATH (merge=False)
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
            )
            if not base_sync_done:
                # First sync: keys need replace_lora_wrapper() to map
                # LoRA keys to base_layer keys (e.g., "q_proj.lora_A..." -> "q_proj.base_layer.weight")
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            # MERGE PATH (merge=True)
            with merged_lora_context(self.module, backup_adapters=True):
                params = self.module.state_dict()
                params = normalize_peft_param_name(params)  # strip LoRA keys, convert to HF names
    else:
        # No LoRA: just state_dict()
        params = self.module.state_dict()

    # 3. Convert weight keys to HF names (handles transformers PR #38385 renaming)
    params = convert_weight_keys(params, ...)

    # 4. Offload model back to CPU
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.module)

    # 5. Materialize DTensor -> full tensor, cast to bf16
    if peft_config is not None and base_sync_done:
        # LoRA-only params: already small, can be sent directly
        per_tensor_param = params.items()
    else:
        # Full params or base weights: need DTensor->full_tensor() materialization
        per_tensor_param = (
            (name, param.to(device).full_tensor().to(torch.bfloat16))
            if isinstance(param, DTensor)
            else (name, param)
            for name, param in params.items()
        )

    return per_tensor_param, peft_config_dict
```

### 2.2 Whole-Model Summon vs Per-Unit Summon (#6512)

★★★★★★★★ **Per-unit summon (#6512) is the critical RTX 4090 optimization**: peak memory drops from full model to largest single FSDP unit.

**Whole-model summon** (legacy, `layered_summon=False`):
```python
# In collect_lora_params(), non-layered path:
with FSDP.summon_full_params(module, writeback=False):
    # Entire model is unsharded at once -> peak = full model size
    lora_params = get_peft_model_state_dict(peft_model)
```

**Per-unit summon** (`layered_summon=True`, #6512):
```python
# In layered_summon_lora_params():
for name, submodule in fsdp_module.named_modules():
    if name == "":
        continue  # Skip root: full summon defeats the purpose
    if fsdp_version(submodule) == 0:
        continue  # Skip non-FSDP modules

    # Per-unit summon
    is_fsdp1 = fsdp_version(submodule) == 1
    summon_ctx = FSDP.summon_full_params(submodule, writeback=False) if is_fsdp1 else nullcontext()

    with summon_ctx:
        # Only this unit is unsharded -> peak bounded by largest unit
        sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
        lora_params.update(sub_lora_params)

    torch.cuda.empty_cache()  # Free after each unit
```

★★★★★★★★ **Key FSDP2 difference**: FSDP2 DTensor params all-gather on demand via
`param.full_tensor()` — no explicit `summon_full_params` context needed. This
is why FSDP2 is more memory-friendly for layered summon.

### 2.3 `collect_lora_params()` — LoRA Weight Extraction

Three modes controlled by `layered_summon` and `base_sync_done`:

| Mode | `layered_summon` | `base_sync_done` | Behavior |
|------|-------------------|-------------------|----------|
| Full base + LoRA | False | False | Whole-model summon, extract all non-LoRA params (first sync) |
| Full LoRA only | False | True | Whole-model summon, extract only LoRA delta params |
| Per-unit LoRA | True | True | Per-unit summon, extract LoRA params per FSDP unit (peak bounded) |

★★★★★★★★ **`layered_summon=True` REQUIRES `base_sync_done=True`**: You cannot use per-unit
summon for the initial base weight sync because each unit only sees its own
parameters, not the full model. The base model must be preloaded in the rollout
server (e.g., `rollout.load_format=safetensors`).

### 2.4 Key Name Transformations

**`replace_lora_wrapper()`** (used when `base_sync_done=False`):
Maps LoRA-specific keys to their base layer equivalents:
- `"model.layers.0.q_proj.base_layer.weight"` -> `"model.layers.0.q_proj.weight"`
- This allows the rollout server to load base weights with HF key names

**`normalize_peft_param_name()`** (used when `merge=True`):
Strips PEFT prefixes and removes LoRA keys entirely:
- `"base_model.model.model.layers.0.q_proj.base_layer.weight"` -> `"model.layers.0.q_proj.weight"`
- `"base_model.model.model.layers.0.q_proj.lora_A.default.weight"` -> **REMOVED** (lora_ prefix)

---

## 3. Two-Phase Weight Sync: Base + LoRA Delta

### 3.1 Phase 1: Base Model Weight Sync (One-Time)

When `model.lora.merge=False` (adapter mode), the base model weights must be
synced once at initialization. This is controlled by `base_sync_done`.

★★★★★★★★ **`base_sync_done` initialization**: `"dummy" not in self.config.rollout.load_format`
If the rollout uses `load_format="safetensors"` or any real format, the base
model is already loaded in the rollout server, so `base_sync_done=True` from
the start — no first-phase base sync needed.

If `load_format="dummy"` (lazy loading), `base_sync_done=False` and the first
`update_weights()` call must sync the entire base model.

**First-phase base sync sequence**:
```python
# In ActorRolloutRefWorker.update_weights(), adapter path:
do_lora_base_sync = not self.base_sync_done

if do_lora_base_sync:
    # get_per_tensor_param with base_sync_done=False -> extract ALL params (base + LoRA base)
    per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=False
    )
    # replace_lora_wrapper maps keys to base layer names
    await self.rollout.update_weights(
        per_tensor_param_base, peft_config=peft_config, base_sync_done=False
    )
```

### 3.2 Phase 2: LoRA Delta Weight Sync (Per-Step)

After base sync is done, subsequent steps only sync LoRA deltas:
```python
# get_per_tensor_param with base_sync_done=True -> extract ONLY LoRA params
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True
)
await self.rollout.update_weights(
    per_tensor_param, peft_config=peft_config, base_sync_done=True
)
```

★★★★★★★★ **Payload reduction**: LoRA deltas are tiny compared to full model.
For Qwen3-8B with rank=32 LoRA on 28 target modules, LoRA params are ~2-4 MiB
vs ~16 GiB base model. This is why sleep_level=1 (adapter path) is optimal for
RTX 4090 — only LoRA deltas need to be transferred per step.

---

## 4. Sleep/Wake Integration

### 4.1 Sleep Levels: 1 vs 2

★★★★★★★★ **sleep_level=1** (LoRA adapter path, `merge=False`):
- **Release**: `tags=["kv_cache"]` only — base weights stay on GPU
- **Resume**: `tags=["kv_cache"]` + `tags=["weights"]` (weights never left, but resume restores kv_cache)
- **Per-step payload**: Only LoRA delta params (~2-4 MiB for rank=32)
- **RTX 4090 optimal**: No weight re-transfer, no OOM risk from full model re-load

★★★★★★★★ **sleep_level=2** (merge path or no LoRA, default):
- **Release**: `tags=["kv_cache", "weights"]` — everything released
- **Resume**: `tags=["kv_cache", "weights"]` — full model re-loaded from trainer
- **Per-step payload**: Full model params (~16 GiB for 8B model)
- **RTX 4090 risky**: Must hold full model in trainer + rollout simultaneously during sync

### 4.2 SGLang Sleep/Wake (Tag-Based)

SGLang uses **tag-based** sleep/wake via `release_memory_occupation(tags=...)` and
`resume_memory_occupation(tags=...)` HTTP API calls:

```python
# In ServerAdapter.release():
if self.sleep_level == 1:
    tags = ["kv_cache"]          # LoRA adapter path: keep base weights
else:
    tags = ["kv_cache", "weights"] # Merge path: release everything

await self._engine.release_memory_occupation(tags=tags)
```

```python
# In ServerAdapter.resume():
await self._engine.resume_memory_occupation(tags=tags)
```

### 4.3 vLLM Sleep/Wake (Integer-Based)

vLLM uses **integer-based** sleep levels via `engine.sleep(level=N)`:

```python
# In vLLMHttpServer._sleep_hybrid():
if mtp_rollout_enabled or self.lora_as_adapter or is_npu_available:
    sleep_level = 1  # Keep weights, release kv_cache
else:
    sleep_level = 2  # Release everything

await self.engine.sleep(level=sleep_level)
```

★★★★★★★★ **SGLang tag-based > vLLM integer-based**: SGLang's tag system is more
flexible and explicit. vLLM's integer system conflates multiple behaviors into
a single level number. For RTX 4090 HYBRID mode, SGLang is the preferred
rollout backend precisely because of this control granularity.

### 4.4 vLLM-Ascend Sleep Level Issue

★★★★★★★★ **vLLM-Ascend sleep_level=1 NOT supported**: The code explicitly forces
sleep_level=1 on Ascend but this is a workaround — Ascend NPU does not properly
support sleep/wake at all, and enabling EP during training leads to accuracy
issues. See vLLM-Ascend #10684 (DSA Hadamard ALL-ZERO after sleep/wake).

---

## 5. IPC Mechanism: ZMQ + CUDA IPC / Shared Memory

### 5.1 vLLM: BucketedWeightSender/Receiver (ZMQ)

The vLLM path uses ZMQ REQ/REP sockets with CUDA IPC handles for zero-copy
GPU-to-GPU weight transfer within a single node:

**Sender side** (trainer process):
```python
class BucketedWeightSender:
    def async_send_weights(self, weights):
        self._init_socket()      # ZMQ REQ socket, bind
        self._init_buffer()      # GPU buffer, send CUDA IPC handle to receiver

        async for name, weight in weights:
            # Pack into bucket buffer, send metadata
            if offset + weight.nbytes > bucket_size:
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                self.socket.recv()  # ACK from receiver
            buffer[offset:offset+weight.nbytes].copy_(weight.view(torch.uint8))

        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
```

**Receiver side** (vLLM worker process):
```python
class BucketedWeightReceiver:
    def receive_weights(self, on_bucket_received):
        self._init_socket()      # ZMQ REP socket, connect
        self._init_buffer()      # Receive CUDA IPC handle, rebuild GPU buffer

        while True:
            metadata = self.socket.recv_pyobj()
            # Extract tensors from buffer using metadata offsets
            weights = [(name, buffer[offset:...].view(dtype).view(shape)) for name, meta in metadata.items()]
            on_bucket_received(weights)  # Callback: model.load_weights() or add_lora()
            self.socket.send(b"")  # ACK
```

★★★★★★★★ **Memory lifecycle**: The sender holds the full GPU buffer during
transfer. For RTX 4090, this means trainer weights + bucket buffer + rollout
weights must all fit in 24 GiB simultaneously. Bucket size is configurable
via `update_weights_bucket_megabytes` (default 2048 MiB).

### 5.2 SGLang: HTTP + NCCL Broadcast

The SGLang path uses HTTP API calls for weight update commands, with actual
weight data transferred via NCCL broadcast within the inference TP group:

```python
# In ServerAdapter.update_weights():
async for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
    await sgl_update_weights(
        engine=self._engine,
        params_batch=params_batch,
        device_mesh_key="infer_tp",
        device_mesh=self.device_mesh,
    )
```

The `sgl_update_weights()` function (imported from `sglang.srt.weight_sync.utils`)
uses NCCL broadcast to distribute weights across the TP group. Only TP-rank-0
dispatches the HTTP command; other ranks participate in the NCCL collective.

### 5.3 Shared Memory Fallback (NPU Compatibility)

Both ZMQ and SGLang paths support shared memory fallback for NPU compatibility:
- vLLM: `BucketedWeightSender(use_shm=True)` uses `multiprocessing.shared_memory`
- SGLang: `use_shm` parameter in configuration
- Data flow: tensor -> shared_memory -> receiver -> copy to device

---

## 6. Delta Weight Sync (#6794) — Bytewise Diff Encoding

### 6.1 Overview

★★★★★★★★ **PR #6794** (OPEN, June 18): Opt-in delta weight sync for SGLang rollout.
Instead of broadcasting every parameter every sync, ship only the `(position, value)`
pairs that actually changed since the last broadcast. ~100x payload reduction.

Key config fields (in `CheckpointEngineConfig`):
- `weight_mode`: `"full"` (default) or `"delta"` (sparse updates)
- `weight_transport`: `"nccl"` (IB/RoCE) or `"disk"` (cross-DC shared FS)
- `weight_encoding`: `"indices"` (int32 positions), `"deltas"` (uint16 gap), `"deltas_zstd"` (zstd-compressed)
- `delta_chunk_megabytes`: receiver-side chunk cap (default 512)
- `delta_disk_dir`: shared FS directory for disk transport

### 6.2 DeltaState: Pinned-CPU Snapshot

```python
class DeltaState:
    snapshot: dict[str, torch.Tensor]  # Pinned CPU copy of last broadcast weights
    d2h_stream: torch.cuda.Stream      # GPU->CPU side stream
    h2d_stream: torch.cuda.Stream      # CPU->GPU side stream

    def seed(self, named_tensors):      # First call: populate snapshot, no flush emitted
    def prefetch_snapshot(named_tensors): # Async H2D of snapshot tiles (pipelining)
    def compute_diffs(named_tensors):    # bytewise diff: current != snapshot
    def update_snapshot_async(named_tensors): # D2H copy of new values back to snapshot
```

★★★★★★★★ **Lifecycle**: First sync seeds the snapshot and sends no data (receiver
already has the checkpoint). Each subsequent sync: prefetch -> compute_diffs ->
encode -> send -> update_snapshot_async. The snapshot is additive host RAM cost
(trade: host RAM for cheaper per-step communication).

### 6.3 Bytewise Diff Encoding

```python
def bytewise_diff_mask(current, snapshot):
    # Dtype-agnostic: view as integer, compare bitwise
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[element_size]
    return current.view(int_dtype) != snapshot.view(int_dtype)
```

★★★★★★★★ **Bit-identical, no drift**: Bytewise diff is arithmetic-free
(`view(int) != view(int)`), so the apply is lossless. No per-step accumulation
error. No need for periodic re-syncs to fight drift.

### 6.4 Wire Format: DeltaParam + DeltaFlush

```python
@dataclass
class DeltaParam:
    name: str
    dtype: str
    shape: list[int]
    pos_start: int     # Byte offset in __positions__ blob
    pos_end: int
    pos_width: int     # 2 (uint16) or 4 (uint32)
    val_start: int     # Element offset in __values__ tensor
    val_end: int

@dataclass
class DeltaFlush:
    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor  # Host uint8 blob (byte-packed positions)
    values_gpu: torch.Tensor     # GPU tensor (changed values in original dtype)
    checksum: int                # torch.hash_tensor XOR for corruption detection
```

### 6.5 Transport Dispatchers

**NCCL transport** (`delta_dispatch.py`):
```python
async def dispatch_flush_nccl(engine, flush, *, group_name):
    spec = _build_delta_spec(flush)  # SGLang DeltaSpec metadata
    positions_gpu = flush.positions_cpu.to(device)
    await engine.update_weights_from_distributed(
        names=["__positions__", "__values__"],
        dtypes=[str(positions_gpu.dtype), str(flush.values_gpu.dtype)],
        shapes=[list(positions_gpu.shape), list(flush.values_gpu.shape)],
        delta_spec=spec,
    )
```

**Disk transport** (for cross-DC):
- Writes delta safetensors to shared filesystem
- Engine reads via `update_weights_from_disk` with `load_format="delta"`

★★★★★★★★ **SGLang-only, vLLM deferred**: Delta sync requires SGLang PR #26519
receiver API (`DeltaSpec`, `DeltaParam`, `DeltaEncoding` in `io_struct`). vLLM
does not have a native sparse weight-update API (vllm-project/vllm#31848).

★★★★★★★★ **Incompatible with CUDA IPC colocate**: IPC only crosses a memory
handle — delta would have no bytes to save there. Delta sync is for NCCL or
disk transport only.

### 6.6 SGLang Rollout Integration

```python
# In ServerAdapter.update_weights(), delta branch:
weight_mode = getattr(self.config.checkpoint_engine, "weight_mode", "full")
if weight_mode == "delta":
    await self._update_weights_delta(weights, bucket_bytes, global_steps)
else:
    # Standard full broadcast
    async for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
        await sgl_update_weights(engine=self._engine, params_batch=params_batch, ...)
```

The `_update_weights_delta()` method lazily creates a `DeltaState` instance that
persists across `update_weights` calls, maintaining the snapshot for incremental diffs.

---

## 7. LoRA Adapter Load/Unload (SGLang Path)

### 7.1 Adapter Mode Weight Update

When `peft_config` and `base_sync_done=True` (LoRA adapter path), SGLang uses
a dedicated LoRA adapter load/unload API instead of weight broadcast:

```python
# In ServerAdapter.update_weights(), adapter branch:
if peft_config and base_sync_done:
    # 1. Unload previous LoRA adapter
    models_result = await self._engine.available_models()
    exists = any(item["id"] == SGLANG_LORA_NAME for item in models_result["data"])
    if exists:
        await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)

    # 2. Serialize LoRA params for SGLang
    serialize_peft_config, serialize_named_tensors = self.wrap_lora_params(peft_config, weights)

    # 3. Load new LoRA adapter from tensors
    from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput
    req = LoadLoRAAdapterFromTensorsReqInput(
        lora_name=SGLANG_LORA_NAME,
        config_dict=serialize_peft_config,
        serialized_tensors=serialize_named_tensors,
    )
    await self._engine.load_lora_adapter_from_tensor(req)
```

★★★★★★★★ **SGLang LoRA name**: Fixed as `"verl_actor_lora_name"` (`SGLANG_LORA_NAME`
constant). Only one LoRA adapter is active at a time (max_loras=1).

### 7.2 LoRA Serialization

```python
def wrap_lora_params(self, peft_config, weights):
    # Serialize peft_config JSON
    peft_config_json = asdict(peft_config)
    peft_config_json["task_type"] = peft_config_json["task_type"].value
    peft_config_json["peft_type"] = peft_config_json["peft_type"].value

    # Process LoRA weight tensors via SGLang's _preprocess_tensor_for_update_weights
    processed_weights = {
        name: _preprocess_tensor_for_update_weights(tensor.detach()) for name, tensor in weights
    }

    # Serialize for each TP rank
    serialized_named_tensors = []
    for i in range(infer_tp_size):
        serialized_tensors = MultiprocessingSerializer.serialize(processed_weights, output_str=True)
        serialized_named_tensors.append(serialized_tensors)

    return peft_config_json, serialized_named_tensors
```

★★★★★★★★ **TP serialization**: LoRA tensors are serialized per TP rank using
SGLang's `MultiprocessingSerializer`. Each rank receives its shard of LoRA params.

---

## 8. LoRA Merge Path (merge=True)

### 8.1 merged_lora_context()

When `model.lora.merge=True`, LoRA is merged into base weights before sync:

```python
@contextmanager
def merged_lora_context(actor, backup_adapters=False):
    if backup_adapters:
        base_weights_backup = backup_base_model_weights(actor)

    fsdp_merge_unmerge(actor, do_merge=True)  # Merge LoRA into base
    try:
        yield  # Do work: sync_to_rollout / generate / etc.
    finally:
        if backup_adapters and base_weights_backup is not None:
            restore_base_model_weights(actor, base_weights_backup)  # Restore from CPU backup
            _clean_merged_lora_(actor)
        else:
            fsdp_merge_unmerge(actor, do_merge=False)  # Unmerge numerically
```

★★★★★★★★ **backup_adapters=True is numerically stable**: Backup to CPU then
restore is better than unmerge because floating-point arithmetic in
unmerge can accumulate errors. The backup path copies base weights to CPU
with LoRA disabled, then restores after sync.

### 8.2 FSDP2 Layer-by-Layer Merge

```python
def fsdp_merge_unmerge(module, do_merge):
    version = fsdp_version(module)
    if version == 1:
        # FSDP1: summon entire model, merge all at once
        with FSDP.summon_full_params(module, writeback=True, with_grads=False):
            _merge_or_unmerge_lora_(module, merge=do_merge)
    else:
        # FSDP2: summon layer-by-layer, merge per-unit
        for name, submodule in module.named_modules():
            if isinstance(submodule, FSDPModule) and name != "":
                with FSDP.summon_full_params(submodule, writeback=True, with_grads=False):
                    _merge_or_unmerge_lora_(submodule, merge=do_merge)
```

★★★★★★★★ **FSDP2 layer-by-layer merge avoids OOM**: Unlike FSDP1 which unshards
the entire model at once, FSDP2 summons each unit individually. This keeps peak
memory bounded by the largest unit — critical for RTX 4090.

---

## 9. RTX 4090 Implications

### 9.1 Memory Budget Analysis

| Component | Size (8B model, bf16) | Size (Qwen3-30B-A3B, bf16) |
|-----------|----------------------|----------------------------|
| Base model (FSDP sharded) | ~8 GiB (dp=1, full shard) | ~6 GiB (3B active params, sharded) |
| LoRA params (rank=32) | ~2-4 MiB | ~6-8 MiB (per-unit summon) |
| Rollout weights (GPU resident) | ~16 GiB (full) | ~6 GiB (3B active) |
| KV cache (4K context) | ~0.5 GiB | ~0.5 GiB |
| Weight sync bucket | 2 GiB (configurable) | 2 GiB |
| **Total peak (sleep_level=2)** | ~26 GiB | ~14 GiB |
| **Total peak (sleep_level=1)** | ~24.5 GiB (no weight re-transfer) | ~12 GiB |

★★★★★★★★ **sleep_level=1 is MANDATORY for RTX 4090 with 8B+ models**: The
merge path (sleep_level=2) requires holding full model in both trainer and
rollout simultaneously during sync, which exceeds 24 GiB for any model >7B.
The adapter path (sleep_level=1) only transfers LoRA deltas (~2-4 MiB).

### 9.2 Configuration Recommendations

```yaml
# RTX 4090 optimal HYBRID config
model:
  lora_rank: 32          # MUST use rank=32 (rank=64 breaks EOS, #6782)
  lora_alpha: 64
  lora:
    merge: false          # ADAPTER PATH, not merge

actor:
  strategy: fsdp2         # FSDP v2 for per-unit summon
  param_offload: true     # CPUOffloadPolicy(pin_memory=True) is default=TRUE

rollout:
  name: sglang            # SGLang preferred (tag-based sleep/wake)
  load_format: safetensors  # Preload base -> base_sync_done=True from start
  free_cache_engine: true
  enable_sleep_mode: true
  layered_summon: true    # Per-unit summon (#6512) for peak memory control
```

★★★★★★★★ **Critical: `lora.merge=False` MUST**: Setting merge=True forces
sleep_level=2, requiring full model re-transfer every step. On RTX 4090, this
means holding ~16 GiB base model in trainer GPU + ~16 GiB in rollout GPU
simultaneously during sync = OOM for any model >7B parameters.

★★★★★★★★ **Critical: `layered_summon=True` MUST**: Per-unit summon bounds peak
memory to the largest FSDP unit (~1-2 GiB) instead of the whole model (~16 GiB).
Combined with `base_sync_done=True` (safetensors preload), this gives 10x peak
memory reduction per #6512 (60 -> 6-8 GiB).

### 9.3 Weight Sync Timing (RTX 4090)

| Phase | Payload | Time Estimate |
|-------|---------|---------------|
| Base sync (first step, safetensors preloaded) | 0 bytes (skip) | ~0s |
| LoRA delta (per step, adapter path) | ~2-4 MiB | ~50-100ms |
| Full model (merge path, per step) | ~16 GiB | ~30-60s |

★★★★★★★★ **The adapter path is ~300x faster than the merge path per step**:
2 MiB LoRA delta vs 16 GiB full model. This directly translates to training
throughput improvement on RTX 4090.

---

## 10. Connection Map to Framework Issues

### 10.1 #6794 (Delta Weight Sync)

★★★★★★★★ **Connection to RTX 4090**: Delta sync further reduces per-step payload
from ~2 MiB (LoRA deltas) to potentially ~0 MiB (no change = no flush). But:
- **SGLang-only**: vLLM does not have sparse weight-update API
- **Incompatible with CUDA IPC**: Only works with NCCL/disk transport
- **RTX 4090 with dp=1**: NCCL transport on single GPU = self-communication,
  delta overhead may exceed benefit for LoRA-only sync
- **Optimal use case**: dp>=2 or disaggregated training over network
- **2 CRITICAL review issues**: Snapshots double host RAM; disk path needs
  shared FS reliability guarantees

### 10.2 #6512 (Per-Unit LoRA Summon)

★★★★★★★★ **MERGED June 18**: 10x peak memory reduction (60 -> 6-8 GiB).
- Dynamic FSDP unit discovery replaces 8 hard-coded prefixes
- FSDP1/2 compatibility (summon_full_params for v1, DTensor full_tensor for v2)
- Skip degenerate FSDP units (no lora_* entries)
- **RTX 4090 CRITICAL**: Makes 30B-A3B MoE LoRA training viable on single GPU

### 10.3 #6699 (Detach Memory Fix)

★★★★★★★★ **MERGED June 17**: 4x per-micro-batch memory reduction.
- Detaches model_output and loss metrics from autograd graph
- Prevents per-micro-batch graph retention that OOMs long-sequence LoRA runs
- **UNFIXED**: Automodel, Megatron, TorchTitan backends have same leak
- **MUST use FSDP backend**: Only FSDP engine has the fix

### 10.4 #6782 (LoRA Rank EOS Bug)

★★★★★★★★ **OPEN**: LoRA rank=64 breaks EOS token generation in vLLM rollout.
- **MUST use rank=32/alpha=64** for RTX 4090 GRPO training
- Root cause: vLLM LoRA padding to max_lora_rank exceeds model vocab embedding size

### 10.5 #6468 (FSDP2 CPU Memory Leak)

★★★★★★★★ **OPEN**: FSDP2 CPU memory leak during weight sync.
- Monotonic 0.6-6.3 GiB/step growth scaling with model size
- Ray OOM after many steps
- Confirmed multi-user, 0 reviews
- **RTX 4090 IMPACT**: CPU memory leak fills host RAM, eventually causes Ray failure
- **Mitigation**: periodic restart, or use FSDP v1 with ZeRO-2

---

## 11. Detailed Function Call Tree

```
ActorRolloutRefWorker.update_weights()
  |-- self.rollout.resume(tags=["weights"])                # SGLang/vLLM wake_up
  |     |-- SGLang: _engine.resume_memory_occupation(tags=["weights"])
  |     |-- vLLM: engine.wake_up()
  |
  |-- self.actor.engine.get_per_tensor_param(layered_summon, base_sync_done)
  |     |-- FSDPEngine.get_per_tensor_param()
  |         |-- load_fsdp_model_to_gpu(module)             # Unshard params
  |         |-- if LoRA and not merge:
  |         |     |-- collect_lora_params(module, layered_summon, base_sync_done)
  |         |         |-- if layered_summon: layered_summon_lora_params(module)
  |         |         |     |-- for each FSDP unit:
  |         |         |         |-- summon_full_params(submodule) / nullcontext (FSDP2)
  |         |         |         |-- get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
  |         |         |         |-- param.full_tensor().detach().cpu()
  |         |         |-- else: whole-model summon + get_peft_model_state_dict
  |         |-- if LoRA and merge:
  |         |     |-- merged_lora_context(module, backup_adapters=True)
  |         |         |-- backup_base_model_weights(module)
  |         |         |-- fsdp_merge_unmerge(module, do_merge=True)
  |         |         |-- yield: module.state_dict() + normalize_peft_param_name()
  |         |         |-- restore_base_model_weights(module, backup)
  |         |-- if base_sync_done=False and LoRA:
  |         |     |-- replace_lora_wrapper(k, peft_config) for each key
  |         |-- convert_weight_keys(params, model)
  |         |-- DTensor -> .full_tensor().to(bf16)
  |         |-- offload_fsdp_model_to_cpu(module)
  |
  |-- if do_lora_base_sync:
  |     |-- engine.get_per_tensor_param(layered_summon, base_sync_done=False)
  |     |-- rollout.update_weights(..., base_sync_done=False)
  |         |-- SGLang: sgl_update_weights() via NCCL broadcast (full base model)
  |         |-- vLLM: BucketedWeightSender.async_send_weights() -> ZMQ IPC
  |
  |-- rollout.update_weights(..., base_sync_done=True)
  |     |-- SGLang adapter path:
  |     |     |-- _engine.unload_lora_adapter(SGLANG_LORA_NAME)
  |     |     |-- wrap_lora_params(peft_config, weights)
  |     |     |-- _engine.load_lora_adapter_from_tensor(req)
  |     |-- SGLang merge/no-lora path:
  |     |     |-- get_named_tensor_buckets(weights, bucket_bytes)
  |     |     |-- sgl_update_weights(engine, params_batch, device_mesh)
  |     |-- SGLang delta path (#6794):
  |     |     |-- _update_weights_delta(weights, bucket_bytes, global_steps)
  |     |         |-- DeltaState.seed() or compute diffs
  |     |         |-- iter_delta_flushes(weights, delta_state, encoding, bucket_bytes)
  |     |         |-- dispatch_flush_nccl(engine, flush, group_name) or write_flush_to_disk()
  |     |-- vLLM adapter path:
  |     |     |-- remove_lora(VLLM_LORA_INT_ID)
  |     |     |-- BucketedWeightSender.async_send_weights(weights)
  |     |     |-- vLLMColocateWorkerExtension.update_weights_from_ipc()
  |     |     |-- add_lora(TensorLoRARequest(peft_config, lora_tensors))
  |     |-- vLLM merge/no-lora path:
  |     |     |-- BucketedWeightSender.async_send_weights(weights)
  |     |     |-- model.load_weights(param_updates)
  |
  |-- engine.to("cpu", model=True)  # Offload trainer
  |-- aggressive_empty_cache()
  |-- rollout.resume(tags=["kv_cache"])
  |-- self.base_sync_done = True
```

---

## 12. Key Design Decisions and Trade-offs

### 12.1 Generator-Based Weight Stream

★★★★★★★★ **Why generators?**: `get_per_tensor_param()` returns a generator, not
a dict. This allows streaming weights one tensor at a time, which:
- Reduces peak memory: only one tensor materialized at a time
- Enables bucketing: `get_named_tensor_buckets()` groups tensors into fixed-size buckets
- Enables delta encoding: `iter_delta_flushes()` wraps the generator with diff computation
- But: **all ranks MUST iterate the generator** because DTensor.full_tensor() triggers
  all_gather across the FSDP group. Skipping deadlocks the others.

### 12.2 FSDP1 vs FSDP2 Differences

| Aspect | FSDP1 | FSDP2 |
|--------|-------|-------|
| Summon API | `FSDP.summon_full_params(ctx)` required | `DTensor.full_tensor()` all-gathers on demand |
| State dict | `FSDP.state_dict_type(FULL_STATE_DICT)` | `get_model_state_dict()` with `StateDictOptions` |
| LoRA merge | Summon entire model -> merge | Summon per-unit -> merge (memory-safe) |
| CPU offload | `CPUOffload(offload_params=True)` | `CPUOffloadPolicy(pin_memory=True)` |
| Per-unit summon | Works but needs `_is_root=True` hack | Works naturally (DTensor per-unit) |

★★★★★★★★ **FSDP2 is preferred for RTX 4090**: Per-unit summon is simpler and
more memory-efficient with FSDP2 because DTensor params don't need explicit
summon context managers. Peak memory is naturally bounded.

### 12.3 SGLang vs vLLM for Weight Sync

| Feature | SGLang | vLLM |
|---------|--------|------|
| Sleep/wake | Tag-based (`["kv_cache"]`, `["kv_cache","weights"]`) | Integer-based (`level=1`, `level=2`) |
| LoRA adapter | `load_lora_adapter_from_tensor` / `unload_lora_adapter` | `add_lora` / `remove_lora` |
| Weight transfer | NCCL broadcast via `sgl_update_weights()` | ZMQ + CUDA IPC `BucketedWeightSender` |
| Delta sync | Supported (#6794 + #26519) | Not supported (waiting on vllm#31848) |
| Bucketing | `get_named_tensor_buckets()` + NCCL | `BucketedWeightSender` + ZMQ |
| TP distribution | Automatic via NCCL | Manual per-rank serialization |

★★★★★★★★ **SGLang is the recommended rollout backend for RTX 4090 HYBRID**
because: (1) tag-based sleep/wake gives explicit control over what's released,
(2) delta sync reduces payload ~100x for dp>=2, (3) LoRA adapter load/unload
is native, (4) NCCL broadcast is simpler than ZMQ IPC for single-GPU setups.

---

## 13. Open Issues and Future Work

### 13.1 Delta Sync (#6794) Open Questions

1. **Snapshot memory cost**: Full-coverage pinned CPU snapshot doubles host RAM
   requirements. For 30B model = ~60 GiB host RAM needed for snapshot alone.
2. **Disk transport reliability**: Cross-DC shared FS needs atomicity guarantees.
3. **vLLM support**: Deferred until vllm-project/vllm#31848 lands native sparse API.
4. **LoRA path**: Orthogonal — base sync stays full, adapter sync could be delta later.
5. **Single GPU (dp=1)**: Delta overhead may exceed benefit; LoRA deltas are already tiny.

### 13.2 FSDP2 CPU Memory Leak (#6468)

★★★★★★★★ **Weight sync triggers the leak**: `full_tensor()` or `state_dict()`
operations during `get_per_tensor_param()` leak CPU memory monotonically. Each
step accumulates 0.6-6.3 GiB of unreleased CPU allocations. Long-running GRPO
training (>100 steps) will exhaust host RAM.

### 13.3 Automodel/Megatron/TorchTitan Backend Leaks (#6699)

★★★★★★★★ **Only FSDP backend is safe**: The detach fix in #6699 only applies to
FSDPEngine. The other backends (AutomodelEngine, MegatronEngine, TorchTitanEngine)
still retain per-micro-batch graphs. MUST use FSDP backend for RTX 4090.

### 13.4 vLLM LoRA Rank Constraint (#6782)

★★★★★★★★ **rank=32 MAXIMUM for vLLM**: LoRA rank=64 breaks EOS generation.
Use rank=32 with alpha=64 (effective alpha/rank = 2, same scaling as rank=64/alpha=128).
SGLang does not have this constraint (uses different LoRA padding).

---

## 14. Summary of Critical RTX 4090 Rules for Weight Sync

★★★★★★★★ **MUST**: `model.lora.merge=False` (adapter path, sleep_level=1)
★★★★★★★★ **MUST**: `rollout.name=sglang` (tag-based sleep/wake, delta sync)
★★★★★★★★ **MUST**: `rollout.load_format=safetensors` (preload base -> skip first sync)
★★★★★★★★ **MUST**: `actor.strategy=fsdp2` (per-unit summon, DTensor)
★★★★★★★★ **MUST**: `rollout.layered_summon=True` (per-unit LoRA extraction)
★★★★★★★★ **MUST**: `lora_rank=32` (rank=64 breaks vLLM EOS, #6782)
★★★★★★★★ **MUST**: FSDP backend only (other backends leak, #6699)

★★★★★★★★ **MUST NOT**: `lora.merge=True` (forces sleep_level=2, OOM on 8B+)
★★★★★★★★ **MUST NOT**: `load_format=dummy` (forces full base sync, 16 GiB payload)
★★★★★★★★ **MUST NOT**: `layered_summon=True` without `base_sync_done=True`
★★★★★★★★ **MUST NOT**: Use delta sync on single GPU dp=1 (overhead exceeds benefit)
★★★★★★★★ **MUST NOT**: Run >100 steps without monitoring #6468 FSDP2 CPU leak
