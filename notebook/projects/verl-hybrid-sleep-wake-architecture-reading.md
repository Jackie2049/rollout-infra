# verl HYBRID Sleep/Wake Architecture — Source Reading

> 2026-06-18 | Source Reading | verl v0.3.0-pre
> ★★★★★★★★ 3 RolloutModes: HYBRID (same process), COLOCATED (same placement group), STANDALONE (disaggregated)
> ★★★★★★★★ sleep_level=1 (LoRA adapter): releases KV cache only, keeps base weights → RTX 4090 game-changer!
> ★★★★★★★★ sleep_level=2 (merge/default): releases weights + KV cache → full sleep → requires weight re-transfer
> ★★★★★★★★ Tag-based memory management: `release(tags=["kv_cache"])` vs `release(tags=["weights", "kv_cache"])`
> ★★★★★★★★ RTX 4090 IMPACT: sleep_level=1 + LoRA = OPTYMAL — only sync adapter deltas (~100x smaller payload)

---

## 1. Architecture Overview: RolloutMode Enum

**File**: `verl/workers/rollout/replica.py:54-67`

```python
class RolloutMode(Enum):
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group
    # but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"
```

### Mode Comparison

| Mode | Process Layout | Weight Sync | GPU Sharing | Use Case |
|------|---------------|-------------|-------------|----------|
| **HYBRID** | Same process | Required (sleep→wake→update) | Same GPU, same process | On-policy GRPO |
| **COLOCATED** | Separate process, same PG | Not required (weights stay) | Same GPU, diff process | GRM (judge) |
| **STANDALONE** | Separate GPU | Full transfer via network/ZMQ | Different GPU | Off-policy |

**RTX 4090 Insight**: HYBRID mode is the default and best for single-GPU GRPO. Same-process = no IPC overhead. The sleep/wake mechanism is the memory management backbone.

---

## 2. Sleep/Wake Lifecycle: The Core Training Loop

### 2.1 Training Step Flow (engine_workers.py:695-749)

The weight sync flow in `HybridWorker.update_weights()`:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HYBRID Training Step                               │
│                                                                      │
│  1. set_expandable_segments(False)                                   │
│  2. rollout.resume(tags=["weights"])   ← wake up weights space      │
│  3. get_per_tensor_param()             ← summon from FSDP           │
│  4. if LoRA adapter (merge=False):                                   │
│     a. sleep_level = 1                ← KEY: only release KV later │
│     b. if first time: base_sync (full weights)                       │
│     c. adapter sync (LoRA deltas only)                               │
│  5. if merge mode:                                                    │
│     a. sleep_level = 2                ← release weights + KV later  │
│     b. full weight sync every step                                    │
│  6. offload to CPU if param_offload enabled                          │
│  7. aggressive_empty_cache()           ← reclaim GPU memory         │
│  8. rollout.resume(tags=["kv_cache"])  ← wake up KV space          │
│  9. set_expandable_segments(True)                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Sleep Level Mechanism

**File**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:192-196`

```python
# sleep_level controls what gets released during sleep/release:
#   2 (default) = release weights + kv_cache (full sleep, merge path)
#   1 = release kv_cache only (keep base weights, adapter path)
# Set by engine_workers.update_weights() when lora.merge=False.
self.sleep_level = 2
```

**Dynamic override** in `engine_workers.py:719-720`:

```python
if not self.peft_merge and peft_config is not None:
    self.rollout.sleep_level = 1
    do_lora_base_sync = not self.base_sync_done
```

### 2.3 Release Implementation (SGLang)

**File**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:278-293`

```python
async def release(self):
    """Release weights and kv cache in GPU memory.

    When sleep_level=1 (LoRA adapter mode), only releases kv_cache
    to keep base weights alive across training iterations.
    When sleep_level=2 (default/merge mode), releases everything.
    """
    await self._init_server_adapter()
    if self._engine is None:
        return
    if self._is_server_tp_leader() and self.config.free_cache_engine:
        if self.sleep_level == 1:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]
        await self._engine.release_memory_occupation(tags=tags)
```

### 2.4 SGLang Server Sleep (async_sglang_server.py:469-490)

```python
async def sleep(self):
    if self.node_rank != 0 or not self.config.free_cache_engine:
        return

    # When using LoRA as adapter (merge=False), only release kv_cache —
    # keep base weights in GPU so we only need to sync adapter deltas.
    if self.lora_as_adapter:
        tags = ["kv_cache"]
    else:
        tags = ["kv_cache", "weights"]

    if self.rollout_mode == RolloutMode.HYBRID:
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        await self.tokenizer_manager.release_memory_occupation(obj, None)
    elif self.rollout_mode == RolloutMode.STANDALONE:
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)
```

---

## 3. Memory Budget Analysis: RTX 4090 Impact

### 3.1 Qwen3-8B on RTX 4090 (24 GiB)

| Component | sleep_level=2 (merge) | sleep_level=1 (LoRA) |
|-----------|----------------------|---------------------|
| Model weights (BF16) | ~16 GiB | ~16 GiB (kept!) |
| KV cache (training) | 0 (released) | 0 (released) |
| KV cache (rollout) | ~4-6 GiB | ~4-6 GiB |
| LoRA adapters | N/A | ~200 MiB (rank=32) |
| Optimizer states | Offloaded to CPU | Offloaded to CPU |
| Gradients | Offloaded to CPU | Offloaded to CPU |
| **Training free GPU** | ~8 GiB | ~8 GiB |
| **Rollout free GPU** | ~4-6 GiB | ~4-6 GiB |

**Key Insight**: sleep_level=1 doesn't change the peak memory during rollout (base weights stay). But during training, the base weights are NOT re-transmitted each step — only LoRA deltas (~200 MiB vs ~16 GiB → **80x payload reduction**).

### 3.2 Qwen3-30B-A3B (MoE) on RTX 4090

| Component | sleep_level=2 | sleep_level=1 |
|-----------|-------------|--------------|
| Active params (3B BF16) | ~6 GiB | ~6 GiB (kept!) |
| Expert params (27B) | Offloaded/CPU | Offloaded/CPU |
| KV cache | Released | Released |
| LoRA (rank=32, 3B active) | N/A | ~60 MiB |
| Optimizer (CPU_Adam) | CPU | CPU |

**sleep_level=1 advantage**: For MoE models, the savings are even more dramatic. The active params (3B) are small enough to stay resident, and LoRA deltas for 3B active are ~60 MiB vs ~6 GiB → **100x reduction**.

### 3.3 vLLM Sleep Level

**File**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py:51-95`

```python
def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    return vs.parse(current_version) >= vs.parse(minver)

# In ServerAdapter.__init__:
if config.layered_summon or (config.expert_parallel_size > 1
    and not _check_vllm_version_for_sleep_level()):
    logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
    self.sleep_level = 1
else:
    self.sleep_level = VLLM_SLEEP_LEVEL
```

**vLLM caveat**: EP>1 with old vLLM (<0.11.0) forces sleep_level=1 with a WARNING about potential OOM. New vLLM uses `VLLM_SLEEP_LEVEL` constant.

---

## 4. Base Sync vs Adapter Sync (Two-Phase Weight Transfer)

### 4.1 First Step: Base Sync

When `do_lora_base_sync = not self.base_sync_done` (first iteration only):

```python
if do_lora_base_sync:
    per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
        layered_summon=self.layered_summon, base_sync_done=False
    )
    await self.rollout.update_weights(
        per_tensor_param_base, peft_config=peft_config,
        base_sync_done=False, global_steps=global_steps
    )
```

This transfers the FULL base model weights to the rollout server (one-time cost).

### 4.2 Every Step: Adapter Sync

After base sync, each subsequent step only transfers LoRA deltas:

```python
await self.rollout.update_weights(
    per_tensor_param, peft_config=peft_config,
    base_sync_done=True, global_steps=global_steps
)
```

When `peft_config and base_sync_done`:
- Unload existing LoRA adapter from SGLang
- Load new LoRA adapter by tensor (serialized + MultiprocessingSerializer)
- No full weight transfer needed!

### 4.3 LoRA Adapter Handling in SGLang

**File**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:315-380`

```python
if peft_config and base_sync_done:
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        # 1. unload existing LoRA
        models_result = await self._engine.available_models()
        exists = any(item["id"] == SGLANG_LORA_NAME for item in models_result["data"])
        if exists:
            await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)

        # 2. load new LoRA by tensor
        serialize_peft_config, serialize_named_tensors = self.wrap_lora_params(
            peft_config, weights
        )
        req = LoadLoRAAdapterFromTensorsReqInput(
            lora_name=SGLANG_LORA_NAME,
            config_dict=serialize_peft_config,
            serialized_tensors=serialize_named_tensors,
        )
        await self._engine.load_lora_adapter_from_tensor(req)
```

**Critical detail**: LoRA name is `SGLANG_LORA_NAME` (constant). Only ONE LoRA adapter at a time. Old adapter must be unloaded before loading new one.

---

## 5. Cross-Backend Comparison: SGLang vs vLLM Sleep/Wake

| Feature | SGLang | vLLM |
|---------|--------|------|
| Sleep mechanism | `tokenizer_manager.release_memory_occupation()` | `engine.sleep(level=sleep_level)` |
| Wake mechanism | `tokenizer_manager.resume_memory_occupation()` | `engine.wake_up()` |
| LoRA as adapter | `lora_as_adapter` property → `tags=["kv_cache"]` | sleep_level=1 conditional |
| Weight update | HTTP-based `update_weights` + LoRA `load_lora_adapter_from_tensor` | ZMQ IPC `update_weights` |
| PD disaggregation | SGLangPDReplica with prefill/decode roles | Not supported |
| CUDA IPC | Via `get_named_tensor_buckets` + DTensor | ZMQ handles |
| Tags granularity | `["kv_cache"]`, `["weights"]`, `["kv_cache", "weights"]` | `level=1` or `level=2` |

### 5.1 vLLM Sleep/Wake

**File**: `verl/workers/rollout/vllm_rollout/vllm_async_server.py:969-974`

```python
# vllm_ascend not support sleep_level now
if not is_torch_npu_available(check_device=False):
    sleep_level = 1  # LoRA mode
else:
    sleep_level = 2  # Full sleep
await self.engine.sleep(level=sleep_level)
```

**Note**: vLLM-Ascend (NPU) does NOT support sleep_level=1! Always full sleep on Ascend → weight re-transfer every step → much slower for LoRA adapter path.

---

## 6. Memory Management Tags: SGLang API

The SGLang server exposes tag-based memory management through its tokenizer manager:

- `ReleaseMemoryOccupationReqInput(tags=["kv_cache"])` — release KV cache only
- `ReleaseMemoryOccupationReqInput(tags=["weights"])` — release model weights only
- `ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])` — release everything
- `ResumeMemoryOccupationReqInput(tags=[...])` — resume previously released resources

This is a more fine-grained API than vLLM's integer-based `sleep(level=1|2)`. The tag system allows future expansion (e.g., releasing specific layer groups, or intermediate activations).

---

## 7. RolloutReplica Architecture: Multi-Server Support

**File**: `verl/workers/rollout/replica.py:70-300`

```python
class RolloutReplica(ABC):
    """Rollout replica is an individual server instance, which may be
    deployed on single or multiple nodes."""

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        """Init hybrid rollout server, rollout engine and training engine
        fused in same process."""
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank :
            self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    async def init_colocated(self, resource_pool: RayResourcePool):
        """Init colocated rollout server, in same ray placement group
        but separate processes."""
        self.rollout_mode = RolloutMode.COLOCATED
        # Creates new RayWorkerGroup for rollout process
        ...

    async def init_standalone(self):
        """Init standalone rollout server, create new resource pool."""
        self.rollout_mode = RolloutMode.STANDALONE
        # Creates entirely separate GPU allocation
        ...
```

**Registry pattern**: 3 backends registered (vLLM, SGLang, TRT-LLM):

```python
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)
```

Plus PD-disaggregated SGLang (`SGLangPDReplica`) when `disaggregation.enabled=True`.

---

## 8. RTX 4090 GRPO Training Configuration Recommendations

### 8.1 Optimal Configuration (sleep_level=1 + LoRA + bypass_mode)

```yaml
rollout:
  name: sglang
  mode: hybrid
  free_cache_engine: true
  sleep_level: 1  # LoRA adapter mode — keep base weights!

actor:
  engine:
    backend: fsdp  # FSDP1 or FSDP2
    peft:
      rank: 32     # NOT 64! (VE-1: rank=64 breaks EOS)
      alpha: 64
      merge: false # KEY: LoRA as adapter → sleep_level=1
  optimizer:
    type: cpu_adam # CPU optimizer → 18Ψ→3.8Ψ
  offload:
    param: true    # Offload params to CPU during training
    grad: true     # Offload gradients to CPU
```

### 8.2 Why sleep_level=1 is CRITICAL for RTX 4090

1. **Base weights stay resident**: No 16 GiB weight transfer per step
2. **Only LoRA deltas (~200 MiB)**: 80x payload reduction
3. **Faster cycle time**: Less sleep/wake overhead → more GRPO iterations
4. **Memory stability**: No weight re-transfer fragmentation
5. **Combines with #6794 delta sync**: LoRA deltas + bytewise diff = even smaller (but #6794 is SGLang-only and deferred for LoRA path currently)

### 8.3 MUST DO / MUST NOT

**MUST DO**:
- Set `peft.merge=false` → triggers sleep_level=1
- Use SGLang rollout (best LoRA adapter support)
- Set `lora_rank=32, lora_alpha=64` (NOT 64! VE-1 breaks EOS)
- Use FSDP backend (only backend with detach fix from #6699)
- Offload optimizer + params to CPU (CPU_Adam → 18Ψ→3.8Ψ)

**MUST NOT**:
- Use `peft.merge=true` → forces sleep_level=2 → full weight transfer every step
- Use `lora_rank=64` → breaks EOS (VE-1)
- Use Megatron backend → C9 detach fix not in upstream!
- Use vLLM-Ascend → sleep_level=1 NOT supported!

---

## 9. Connections to Other Framework Issues

| Issue | Connection to sleep/wake |
|-------|------------------------|
| **DS-1 (#8072)** | ZeRO-3+LoRA regression → sleep_level=1 with ZeRO-3 is BROKEN! Must ZeRO-2 |
| **VE-1 (#6782)** | LoRA rank=64 breaks EOS → adapter path unusable at rank=64 |
| **VE-2 (#6468)** | FSDP2 CPU memory leak → 6.3 GiB/step during weight sync → worsens sleep/wake cycle |
| **VE-3 (#6699)** | detach model_output → 4x memory reduction → frees GPU for KV cache during rollout |
| **VE-6512** | per-unit LoRA summon → 10x peak memory → enables LoRA adapter on larger models |
| **SG-3 (#27097)** | multi-LoRA determinism → single LoRA adapter path must be deterministic! |
| **VL-3 (#39096)** | SM89 batch invariance → enforce_eager=True for rollout engine |
| **VA-1 (#10684)** | DSA Hadamard sleep/wake ALL-ZERO → verl Ascend sleep is BROKEN! |
| **DSV4 systematic** | enforce_eager=True → DSV4 models MUST use eager mode in rollout |

---

## 10. Future Directions

1. **#6794 delta weight sync**: Could enable LoRA delta encoding (~100x smaller), but currently deferred for LoRA path
2. **PyTorch #187620 PartialOffloadPolicy**: Fractional CPU offload → more GPU space for KV cache during rollout
3. **verl #6790 separate async trainer**: Switch + offload strategies → more flexible sleep/wake
4. **SGLang PD disaggregation**: Prefill/decode split → can sleep/wake decode separately
5. **vLLM sleep_level=1 for EP>1**: Currently warns about OOM → needs validation on RTX 4090

---

## Source Files Read

| File | Key Lines | Purpose |
|------|-----------|---------|
| `replica.py` | 54-67, 131-226 | RolloutMode enum, RolloutReplica ABC, init_hybrid/colocated/standalone |
| `sglang_rollout.py` | 192-196, 266-293, 295-380 | sleep_level, release/resume tags, update_weights + LoRA adapter |
| `async_sglang_server.py` | 445-490, 496-508 | SGLang server sleep/wake/release_kv_cache |
| `engine_workers.py` | 695-749 | HybridWorker.update_weights() flow, sleep_level=1 override |
| `base.py` | 44-69, 83-87 | BaseRollout ABC, _ROLLOUT_REGISTRY |
| `vllm_rollout.py` | 51-95, 969-974 | vLLM sleep_level version check, engine.sleep() |
