# verl v0.8 Worker Lifecycle + vLLM Integration Source Reading

> 2026-06-16 | verl repo at /Users/jackiemac/workspace/rollout-infra/verl
> Focus: Worker lifecycle from creation to training step, vLLM integration at code level, bypass_mode, TransferQueue+KVBatchMeta zero-copy
> ★★★★★★★ Complete lifecycle traced from init_model → rollout → reward → advantage → actor update → weight sync

---

## 1. Worker Class Architecture

★★★★★★★ Two-layer architecture:

```
Layer 1: Worker (Ray actor per GPU)
  ├── TrainingWorker (engine_workers.py:76)
  │   ├── config: TrainingWorkerConfig
  │   ├── engine: BaseEngine (FSDP/Megatron/VeOmni/FSDP2)
  │   ├── Methods: train_batch, infer_batch, train_mini_batch, save/load_checkpoint
  │   ├── Loss: self.loss_fn = ppo_loss(config=actor_config)
  │   └── Dispatch: ND_compute(mesh="train") for train/infer
  │
  ├── ActorRolloutRefWorker (engine_workers.py:434)
  │   ├── self.actor: TrainingWorker  (actor model)
  │   ├── self.ref: TrainingWorker    (reference model)
  │   ├── self.rollout: BaseRollout   (ServerAdapter for vLLM/SGLang/TRT-LLM)
  │   ├── self.checkpoint_engine: CheckpointEngineRegistry
  │   └── Methods: init_model, compute_log_prob, compute_ref_log_prob,
  │                update_actor, update_weights (async!)

Layer 2: WorkerGroup (Ray driver-side orchestrator)
  ├── RayWorkerGroup._init_with_resource_pool() → creates Ray PG + actors
  ├── execute_all() / execute_all_async() → dispatch to all workers
```

★★★★★★★★★ Key: FusedWorker pattern (HYBRID mode)

In HYBRID mode, `create_colocated_worker_cls()` dynamically creates a `FusedWorker` that inherits from multiple Worker subclasses via `WorkerDict`. Actor+rollout+ref share the SAME Ray actor → zero IPC overhead.

---

## 2. Worker Lifecycle: init_model

★★★★★★★ engine_workers.py:500-632 — Initialization order is INTENTIONAL:

```
1. Ref model (lines 504-539):
   → ref_config → TrainingWorker(config=ref_training_config)
   → CPU offload option → ref doesn't train → can offload to CPU
   → self.set_dispatch_collect(mesh_name="ref")

2. Actor model (lines 542-588):
   → actor_config → TrainingWorker(config=actor_training_config)
   → self.loss_fn = partial(ppo_loss, config=actor_config)
   → self.set_dispatch_collect(mesh_name="actor")

3. Rollout engine (lines 591-616):
   → rollout_config → get_rollout_class("vllm", "async") → ServerAdapter
   → 3D DeviceMesh(dp, infer_tp, infer_pp) → inference parallel topology
   → LoRA flags: base_sync_done, layered_summon, peft_merge

4. Checkpoint engine (lines 619-629)
5. GPU cache clear (line 632)
```

★★★★★★★★★ LoRA: When `lora_rank > 0`, ref model = actor WITHOUT LoRA (`no_lora_adapter=True`) → eliminates separate ref WorkerGroup → saves ~14GB on RTX 4090!

---

## 3. vLLM Integration: ServerAdapter + vLLMHttpServer

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### v0.8: SPMD retired, ServerAdapter replaces HFRollout

PR #4411 retired HFRollout. ALL rollout is now async server-based:

```
ServerAdapter (vllm_rollout.py:61-223):
  → BaseRollout subclass → async-only
  → generate_sequences() RAISES NotImplementedError!
  → ZMQ IPC handle for weight sync

vLLMHttpServer (vllm_async_server.py:84-955):
  → Ray.remote actor → runs per node
  → Launches vLLM serve via CLI → AsyncLLM engine
  → Sleep/Wake: level=1 (LoRA) or level=2 (full weight)
  → worker_extension_cls: vLLMColocateWorkerExtension → bridge to verl

vLLMColocateWorkerExtension (utils.py):
  → monkey_patch_model() → patches compute_logits + MoE weight loader
  → update_weights_from_ipc() → receives weights via ZMQ + CUDA IPC
  → __new__ patches LoRA hijack + FP8 patches at construction time
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ vLLM is NOT part of verl's codebase! Integration via worker_extension_cls → vLLMColocateWorkerExtension → bridge for weight sync + LoRA + FP8 patches
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. Weight Synchronization: 3 Backends

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 4.1 naive backend (HYBRID mode, in-process)

★★★★★★★★★ engine_workers.py:666-747 — weight sync for HYBRID:

```python
async def update_weights(self, global_steps=None, mode="auto"):
    # naive path (HYBRID, in-process):
    # 1. resume rollout memory
    # 2. get per-tensor params from actor engine (FSDP AllGather!)
    # 3. LoRA base sync (adapter path only, two-phase)
    # 4. sync adapter/merged weights
    # 5. offload actor to CPU if param_offload
    # 6. resume KV cache
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ FSDP get_per_tensor_param (fsdp/transformer_impl.py:794-871):

KEY bridge between training (DTensor shards) and rollout (full tensors):

```python
def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
    # DTensor.full_tensor() does AllGather inside FSDP!
    per_tensor_param = (
        (name, param.to(device).full_tensor().to(torch.bfloat16))
        if isinstance(param, DTensor)
        else (name, param)
        for name, param in params.items()
    )
```

★★★★★★★★★ LoRA two-phase sync:
1. First: base weights (model without LoRA) → vLLM loads as standard model
2. Second: LoRA adapter → vLLM loads as TensorLoRARequest (enable_lora=True)

### 4.2 CUDA IPC (COLOCATED, same GPU different process)

BucketedWeightSender → ZMQ REQ + GPU buffer + torch IPC handle
BucketedWeightReceiver → ZMQ REP + rebuild IPC handle

★★★★★★★★★ Bucketed: packs multiple weights into one bucket → reduces ZMQ roundtrips

### 4.3 NCCL/NIXL (STANDALONE, cross-GPU/cross-node)

CheckpointEngineManager → abort replicas → build NCCL ProcessGroup → send weights → restore

---

## 5. TransferQueue + KVBatchMeta (v0.8 Zero-Copy)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### TransferQueue replaces DataProto serialization

```
KVBatchMeta: (keys, tags, partition_id, fields, extra_info)
  → keys: unique identifiers per sample
  → tags: metadata per key (global_steps, status, seq_len)
  → partition_id: "train" or "val"
  → fields: TensorDict → selected fields to read/write

Data flow:
  Rollout → tq.async_kv_put(key=uid, partition_id="train")
  ReplayBuffer → tq.kv_list() → metadata for finished keys
  Trainer → tq.kv_batch_get(keys=batch.keys, select_fields=[...])
  Trainer → tq.kv_batch_put(keys=batch.keys, fields=TensorDict)
  Cleanup → tq.kv_clear(keys=batch.keys)
```

★★★★★★★★★ tqbridge decorator: automatically bridges KVBatchMeta ↔ TensorDict → worker methods don't need to know about TransferQueue!

---

## 6. Complete v0.8 Step() Lifecycle

★★★★★★★★★ main_ppo_sync.py:1688-1753:

```python
def step(self, batch_dict, metrics, timing_raw):
    # 1. Rollout: async via AgentLoopManager → TransferQueue
    # 2. Reward (optional, if colocated)
    # 3. Balance batch across DP groups
    # 4. Compute old_log_prob (bypass_mode skips!)
    # 5. Compute ref_log_prob (bypass_mode skips!)
    # 6. Compute critic values (optional, GRPO skips)
    # 7. Compute advantage
    # 8. Update critic (optional)
    # 9. Update actor
    return batch
```

After step: `checkpoint_manager.update_weights()` → actor→rollout weight sync

---

## 7. bypass_mode in Worker Lifecycle

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### bypass_mode: skip ref model, old_log_probs = rollout_log_probs

```python
def _compute_old_log_prob(self, batch, metrics):
    if bypass_recomputing_logprobs:
        # ★★★★★★★★ Zero-cost: reuse rollout_log_probs as old_log_probs
        data["old_log_probs"] = data.pop("rollout_log_probs")
        return  # SKIP entire actor forward pass!
```

★★★★★★★★★ compute_policy_loss_bypass_mode (core_algos.py:2351-2449):

```
bypass mode: old_log_prob IS rollout_log_prob
IS weights: w = π_current / π_rollout
PPO-clip: r = π_current / π_rollout → clipping handles ratio → NO double IS
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ RTX 4090 impact of bypass_mode:
  → SKIP actor forward for old_log_probs → save ~7GB + computation time
  → SKIP ref model entirely → save ~14GB GPU memory
  → 2-policy system (π_rollout + π_θ) instead of 3-policy
  → GRPO + bypass = RTX 4090 optimal: only actor+rollout → ~10GB peak
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. RolloutMode Architecture

★★★★★★★★★ Three mutually exclusive modes:

```
HYBRID: FusedWorker = actor+rollout+ref in 1 Ray actor → zero IPC → RTX 4090 optimal!
COLOCATED: same PlacementGroup but separate actors → CUDA IPC weight sync
STANDALONE: independent ResourcePool → NCCL/NIXL weight sync → multi-GPU only
```

★★★★★★★★★ HYBRID = RTX 4090 recommended mode: in-process → naive weight sync → minimal overhead

---

## 9. Sleep/Wake: GPU Memory Time-Multiplexing

★★★★★★★★★ HYBRID mode: rollout and training share GPU memory → must time-multiplex:

```
rollout (GPU occupied) → sleep(level=1/2) → training (GPU freed) → weight_sync → wake_up
sleep_level=1: LoRA only → release KV cache, keep base weights
sleep_level=2: full release → release weights + KV cache
```

---

## 10. Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★★★ ServerAdapter-only (v0.8): SPMD retired → ALL rollout async → vLLMHttpServer → worker_extension_cls bridge

2. ★★★★★★★★★ TransferQueue replaces DataProto: KVBatchMeta zero-copy → tqbridge decorator → worker methods unaware of TransferQueue

3. ★★★★★★★★★★ 3 weight sync backends: naive (HYBRID in-process), CUDA IPC (COLOCATED), NCCL/NIXL (STANDALONE) → mutually exclusive

4. ★★★★★★★★★ LoRA two-phase sync: base first → LoRA adapter second → base_sync_done prevents redundant syncs

5. ★★★★★★★★★★ bypass_mode = 2-policy system: skip old_log_probs computation + skip ref model → save 14GB → RTX 4090 MUST

6. ★★★★★★★★ HYBRID mode = FusedWorker → actor+rollout+ref in 1 process → zero IPC → RTX 4090 optimal

7. ★★★★★★★★★ Sleep/Wake = GPU time-multiplexing: rollout→sleep→train→sync→wake → mandatory for single GPU

8. ★★★★★★★★ Dispatch router + ND_compute mesh: @register + make_nd_compute → lazy-compute dispatch → non-uniform DP-TP-PP
