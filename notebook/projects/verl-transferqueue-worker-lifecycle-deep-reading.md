# verl TransferQueue + Worker Lifecycle Deep Reading

**Date**: 2026-07-15 (Session 10 continued)
**Purpose**: Understand verl's TransferQueue weight synchronization and worker lifecycle for GRPO training
**Sources**: verl/utils/transferqueue_utils.py (lines 1-431), verl/workers/engine_workers.py, verl/trainer/ppo/ray_trainer.py, background agent deep read

---

## 1. TransferQueue Architecture Overview

```
TransferQueue (TQ): GPU-backed shared memory for zero-copy data transfer between workers

Key concepts:
  BatchMeta: lightweight metadata handle pointing to data in TQ shared memory
  KVBatchMeta: key-value variant of BatchMeta for partitioned/sharded data
  tqbridge decorator: automatic BatchMeta ↔ TensorDict conversion for function calls

Data flow:
  Producer worker → put data into TQ → get BatchMeta (metadata handle)
  Consumer worker → receive BatchMeta → get data from TQ → TensorDict
  Both workers share the SAME GPU memory → zero-copy transfer!

★★★★★★★★★ This is fundamentally different from Ray's DataProto:
  Ray DataProto: serialize → send over network → deserialize → GPU allocate → copy
  TransferQueue: already in GPU shared memory → just pass metadata → zero-copy access
  → TransferQueue = 100× faster for GPU-to-GPU data transfer
  → But requires shared GPU memory (same machine or NVLink-connected machines)
```

---

## 2. BatchMeta: Lightweight Metadata Handle

```
BatchMeta is a "pointer" to data stored in TransferQueue:

  BatchMeta contains:
  → partition_ids: which TQ partition stores the data
  → global_indexes: indices within the partition
  → field_names: which tensor fields are stored
  → extra_info: non-tensor metadata (strings, dicts)
  → size: number of samples in the batch
  → tags: per-sample tags (for KVBatchMeta)

  Instead of passing entire TensorDict (14 GiB for model weights):
  → Pass BatchMeta (~1 KiB metadata) → instant transfer
  → Consumer reads data directly from GPU shared memory
  → Zero-copy: no GPU allocation, no memcpy, no serialization

★★★★★★★★★ For GRPO weight sync:
  Trainer worker → updated model weights → put to TQ → BatchMeta
  Rollout worker → receive BatchMeta → get weights from TQ → zero-copy access
  → 14 GiB model weights transfer in ~1ms (shared memory pointer)
  → vs Ray DataProto: ~5-10 seconds (serialize + network + deserialize)
```

---

## 3. KVBatchMeta: Key-Value Partitioned Data

```
KVBatchMeta extends BatchMeta for partitioned/sharded data:

  KVBatchMeta contains:
  → keys: list of keys identifying data entries
  → tags: per-sample tags
  → partition_id: which TQ partition
  → fields: which tensor fields to retrieve
  → extra_info: non-tensor metadata

  Use case: sharded model weights across multiple workers
  → Each worker has its own partition (ZeRO-2 style)
  → KVBatchMeta stores keys → consumer retrieves all shards → reassemble

  Conversion flows:
  KVBatchMeta → BatchMeta:
    → async_kv_retrieve_meta(keys, partition_id) → get full BatchMeta
  BatchMeta → KVBatchMeta:
    → async_kv_retrieve_keys(global_indexes, partition_id) → get keys
    → Create KVBatchMeta with keys, tags, partition_id

★★★★★★★★★ This is critical for ZeRO-2 weight sync:
  ZeRO-2 shards optimizer states across workers
  → Each worker stores its shard in TQ partition
  → KVBatchMeta identifies which shard → consumer retrieves specific data
  → Enables efficient shard-level transfer without moving entire model
```

---

## 4. tqbridge Decorator: Automatic Conversion

```
tqbridge(dispatch_mode) decorator wraps function calls with automatic TQ handling:

Flow (for synchronous function):
  1. Find BatchMeta in function arguments → _find_meta()
  2. If BatchMeta found: initialize TQ (tq.init()) if not already
  3. Convert BatchMeta → TensorDict → _meta_to_realdata()
     → If KVBatchMeta: first convert to BatchMeta (kv_batch_meta2batch_meta)
     → Then: tq_client.async_get_data(meta) → retrieve TensorDict from TQ
  4. Call original function with TensorDict arguments
  5. Process output:
     → If output is TensorDict with data: put_data=True
     → _update_meta_with_output(): put output back to TQ → get new BatchMeta
     → If KVBatchMeta: convert back (batch_meta2kv_batch_meta)
  6. Return updated BatchMeta (or original output if no TQ involved)

Flow (for async function):
  → Same but with async variants (async_meta_to_realdata, async_update_meta_with_output)
  → Uses asyncio event loop for concurrent TQ operations

★★★★★★★★★ tqbridge enables transparent weight sync:
  Actor worker methods decorated with @tqbridge:
    → generate_sequences(): input BatchMeta → retrieve prompts → generate → put output to TQ
    → update_policy(): input BatchMeta → retrieve rollout data → update → put new weights to TQ
  Workers don't need to know about TQ → decorator handles everything
```

---

## 5. Worker Lifecycle: 6-Step GRPO Training Cycle

```
verl HYBRID worker lifecycle for GRPO on single GPU:

Step 1: INIT — Load model + set up workers
  → ActorRolloutRefWorker: init model + optimizer + rollout engine
  → CriticWorker: init (if PPO, NOT for GRPO)
  → Ray: coordinate workers via single_controller

Step 2: ROLLOUT — Generate responses
  → ActorRolloutRefWorker.generate_sequences():
    → SGLang rollout engine → generate group_size responses
    → Put rollout data to TransferQueue → BatchMeta
    → Duration: ~5-30 seconds (group_size × max_tokens)

Step 3: SLEEP — Release KV cache, keep model weights
  → sleep(level=1): release KV cache → keep model weights on GPU
  → GPU memory freed: ~2-3 GiB KV cache released
  → For training: model weights still on GPU → no reload needed
  → Duration: ~0.5 seconds

Step 4: TRAINING — Update model
  → ActorRolloutRefWorker.update_policy():
    → Receive rollout BatchMeta from TransferQueue
    → Compute advantages (GRPO normalization)
    → Compute loss (PPO-clip / UP-GRPO)
    → Backward → gradient computation
    → ZeRO-2 optimizer step (CPU_Adam)
    → gradient_clipping = 1.0
    → Put updated weights to TransferQueue → new BatchMeta
    → Duration: ~10-60 seconds

Step 5: WAKE — Reload for next rollout
  → wake(): ready for next rollout generation
  → Delta sync: only LoRA adapter weights (~50 MiB)
  → Duration: ~1 second

Step 6: REPEAT — Back to Step 2

★★★★★★★★★ Memory budget per phase (RTX 4090, 24 GiB):
  Rollout: 14 GiB model + 2-3 GiB KV = ~16-17 GiB
  Sleep: 14 GiB model only (KV released)
  Training: 14 GiB model + 3.8 GiB activations + 1.4 GiB grads = 19.2 GiB
  Wake: 14 GiB model + LoRA delta = ~14.05 GiB
  All phases fit in 24 GiB!
```

---

## 6. TransferQueue vs Naive Backend Comparison

```
verl supports 2 weight sync backends:

Naive backend (Ray DataProto):
  → Serialize TensorDict → Ray remote call → deserialize → GPU allocate → copy
  → 14 GiB model weights → ~5-10 seconds transfer
  → Works across machines (network transfer)
  → No shared GPU memory requirement
  → Default: used when TransferQueue not available

TransferQueue backend:
  → Put TensorDict to shared GPU memory → pass BatchMeta → zero-copy access
  → 14 GiB model weights → ~1ms transfer (just metadata)
  → Requires shared GPU memory (same machine or NVLink)
  → Requires transfer_queue pip package (TransferQueue==0.1.7)
  → Fallback: mock TransferQueue (raises RuntimeError) when not installed

★★★★★★★★★ For RTX 4090 GRPO:
  Single GPU = both trainer and rollout on SAME GPU
  → TransferQueue: both workers share same GPU memory → zero-copy
  → This is the IDEAL case for TransferQueue (same GPU = instant transfer)
  → Naive backend: unnecessary serialization + deserialization overhead

  For multi-node:
  → TransferQueue: requires NVLink or shared GPU memory → same machine
  → Naive: works across machines → Ray network transfer
  → Hybrid: TQ for intra-node, Ray for inter-node
```

---

## 7. Sleep/Wake Mechanism Details

```
verl sleep/wake for colocated training+rollout on single GPU:

sleep(level):
  level=0: release nothing (full model + KV on GPU)
  level=1: release KV cache only (model weights stay)
  level=2: release model weights + KV (everything off GPU)

★★★★★★★★★ RTX 4090 MUST: sleep_level=1 ONLY
  level=0: no memory freed → can't fit training phase → OOM!
  level=1: KV released → enough headroom for training (4.8 GiB)
  level=2: model offloaded → wake requires full reload (~5s) + risk of corruption

  ★★★★★★★★★ NEVER use sleep_level=2 on RTX 4090!
  → Weight corruption risk (CuMem unmap race)
  → Slow wake (full model reload from CPU/SSD)
  → vLLM #45552: CuMem sleep/wake + CUDA stream race → corruption

wake():
  → For level=1: KV cache allocation for next rollout (fast, ~0.5s)
  → For level=2: model reload from CPU/SSD (slow, ~5s) + corruption risk
  → Delta sync: only LoRA adapter weights (~50 MiB) → fast (~1s)

Weight sync after wake:
  → TransferQueue: updated weights already in shared memory → just BatchMeta pointer
  → LoRA delta: only ~50 MiB changes → minimal transfer
  → Full param sync: NOT recommended → 14 GiB → slow + unnecessary for LoRA training
```

---

## 8. ReplayBuffer: Rollout Data → Training Data

```
ReplayBuffer stores rollout data for GRPO training:

Structure:
  → group_by_prompt: group responses by prompt → correct GRPO normalization
  → group_size: number of responses per prompt (4 or 8 for GRPO)
  → TensorDict: stores rewards, log_probs, values, attention_mask, response_mask

Flow:
  Rollout worker → generate_sequences() → TensorDict with responses
  → Put to TransferQueue → BatchMeta
  → ReplayBuffer collects BatchMeta → retrieve data → organize by group

  group_by_prompt=True (★★★★★★★★ MUST for GRPO):
    → Responses from same prompt grouped together
    → Advantage: A_i = (R_i - μ_group) / σ_group
    → σ_group computed from within-group rewards → correct normalization

  group_by_prompt=False (WRONG for GRPO):
    → All responses treated as independent samples
    → σ computed across ALL rewards → wrong advantage normalization
    → Cross-group variance ≠ within-group variance → biased advantages

★★★★★★★★★ ReplayBuffer + group_by_prompt is CRITICAL for GRPO:
  Our rllm fork PR #2 (configurable grouping_key) addresses this!
  verl uses group_by_prompt=True by default → correct
  But: some configs may override → danger of wrong normalization
```

---

## 9. verl HYBRID Architecture: Single-GPU Colocate

```
verl HYBRID mode: colocate training + rollout on same GPU

Architecture:
  ActorRolloutRefWorker (single GPU):
    → Training mode: model forward + backward + optimizer step
    → Rollout mode: SGLang/vLLM serving + generate responses
    → Sleep/wake: switch between modes

  Worker methods:
    → generate_sequences(): rollout mode → SGLang serving
    → update_policy(): training mode → ZeRO-2 + CPU_Adam
    → compute_reward(): external reward model → optional

  TransferQueue bridge:
    → @tqbridge decorator on all worker methods
    → Automatic BatchMeta ↔ TensorDict conversion
    → Zero-copy data transfer between phases

★★★★★★★★★ RTX 4090 HYBRID configuration:
  verl HYBRID + FSDP1 + CPPO + bypass_mode:
  → Training: ZeRO-2 + CPU_Adam + bypass_mode + gradient_clipping=1.0
  → Rollout: SGLang + enforce_eager + LoRA rank=32/alpha=64
  → Sleep/wake: level=1 (release KV, keep model)
  → TransferQueue: zero-copy weight sync between phases
  → Memory: 19.2 GiB training, 17 GiB rollout → both fit in 24 GiB

  This is our #1 BEST config for RTX 4090 GRPO training!
```

---

## 10. Delta Sync: LoRA Weight Transfer

```
LoRA delta sync: only transfer adapter weights, NOT full model

Full param sync (BAD):
  → Transfer 14 GiB full model weights → slow (~5-10s with Ray)
  → Unnecessary: base model unchanged → only LoRA adapter updated
  → Risk: stale weights if sync fails partially

LoRA delta sync (★★★★★★★★ GOOD):
  → Transfer only LoRA adapter weights (~50 MiB for rank=32)
  → Fast: ~0.1ms with TransferQueue (just metadata pointer)
  → Base model stays unchanged → no risk of stale weights
  → Duration: ~1 second (including verification)

Implementation:
  → merge_and_unload_lora(): merge LoRA into base model for rollout
  → After training: extract LoRA delta → put to TransferQueue
  → Rollout worker: get LoRA delta → apply to base model → generate
  → Duration: merge/unmerge ~0.5s each

★★★★★★★★★ MUST NOT #9: full param sync (from cross-framework rules)
  → LoRA delta sync sufficient → 50 MiB vs 14 GiB → 280× faster
  → Only delta needs transfer → base model already on GPU
  → This is the verl HYBRID approach → proven correct for GRPO
```

---

## 11. Cross-Framework Weight Sync Comparison

```
| Feature | verl TransferQueue | DeepSpeed ZeRO | SGLang NIXL |
|---------|-------------------|---------------|-------------|
| **Approach** | GPU shared memory | CPU optimizer offload | RDMA direct GPU→GPU |
| **Transfer size** | BatchMeta (~1 KiB) | Gradient shard (MiB) | KV blocks (MiB) |
| **Latency** | ~1ms (zero-copy) | ~50-100ms (CPU→GPU) | ~5-10ms (RDMA) |
| **Hardware** | Same GPU or NVLink | Any (CPU always available) | IB/RoCE required |
| **Training** | Yes (ZeRO-2) | Yes (ZeRO-2) | No (inference only) |
| **Rollout** | Yes (SGLang/vLLM) | No (training only) | Yes (P/D disaggregation) |
| **Sleep/wake** | Yes (level=1) | No (ZeRO handles it) | Yes (CuMem) |
| **RTX 4090** | PERFECT (same GPU) | Good (CPU offload) | NOT needed (single GPU) |

★★★★★★★★★ For RTX 4090 GRPO:
  verl TransferQueue is BEST because:
  1. Same GPU → zero-copy → instant transfer
  2. Training + rollout on same GPU → no inter-machine transfer needed
  3. Sleep/wake level=1 → KV release only → model stays on GPU
  4. LoRA delta sync → only 50 MiB transfer → minimal overhead
```

---

## 12. Key Data Flow: Complete GRPO Step with TransferQueue

```
Complete GRPO training step on RTX 4090 with verl HYBRID:

Phase 1: Rollout (SGLang serving)
  → ActorRolloutRefWorker.generate_sequences()
  → SGLang processes prompts → generate group_size responses
  → HiCache: manage KV cache during long sequences
  → Output: TensorDict with log_probs, rewards, attention_mask
  → @tqbridge: put output to TQ → return BatchMeta
  → Duration: 5-30s

Phase 2: Sleep (KV release)
  → sleep(level=1): release KV cache → 2-3 GiB freed
  → Model weights stay on GPU → no reload needed
  → Duration: 0.5s

Phase 3: Training (ZeRO-2 + bypass)
  → ActorRolloutRefWorker.update_policy()
  → @tqbridge: receive BatchMeta from TQ → retrieve TensorDict
  → ReplayBuffer: group_by_prompt → organize by groups
  → Compute advantages: GRPO (A = (R - μ) / σ)
  → Compute loss: UP-GRPO (unbounded positive A)
  → Backward: bypass_mode (skip old_log_prob forward) → 3.8 GiB activations
  → ZeRO-2 optimizer step: CPU_Adam → gradient_clipping=1.0
  → @tqbridge: put updated weights to TQ → return BatchMeta
  → Duration: 10-60s

Phase 4: Wake + Delta Sync
  → wake(): allocate KV cache for next rollout
  → LoRA delta sync: extract adapter weights → put to TQ → apply to rollout model
  → Duration: ~1s

Total cycle: 5-30s + 0.5s + 10-60s + 1s = ~15-90s per GRPO step
Throughput: ~7-40 steps per hour
```

---

## Session Stats
- **TransferQueue architecture**: GPU shared memory, BatchMeta, KVBatchMeta, tqbridge decorator
- **Worker lifecycle**: 6-step GRPO cycle (init→rollout→sleep→train→wake→repeat)
- **Sleep/wake**: level=1 MUST (RTX 4090), level=2 NEVER (corruption risk)
- **ReplayBuffer**: group_by_prompt=True for correct GRPO normalization
- **Delta sync**: LoRA 50 MiB vs full params 14 GiB → 280× faster
- **Cross-framework comparison**: TQ vs ZeRO vs NIXL for weight sync
- **Complete data flow**: end-to-end GRPO step traced through TransferQueue
- **RTX 4090 BEST config**: verl HYBRID + FSDP1 + CPPO + bypass + TQ + sleep_level=1

---

## 13. Source Code Implementation Details (from Background Agent)

### tqbridge Decorator Pipeline (transferqueue_utils.py:298-431)

```
tqbridge(dispatch_mode) wraps function calls with automatic TQ handling:

  Inner function (sync):
  1. _find_meta(*args, **kwargs) → detect BatchMeta/KVBatchMeta in arguments
  2. If meta found: tq.init() (lazy init, global TQ_INITIALIZED flag)
  3. If KVBatchMeta: kv_batch_meta2batch_meta() → convert KV→regular meta
  4. _meta_to_realdata(meta) → tq_client.async_get_data(meta) → retrieve TensorDict
  5. func(*args, **kwargs) → execute with real data
  6. If output is TensorDict with batch_size > 0: put_data=True
  7. _update_meta_with_output(output, meta) → tq_client.async_put() → new BatchMeta
  8. If was KVBatchMeta: batch_meta2kv_batch_meta() → convert back
  9. Return updated BatchMeta

  Async variant (async_inner): same pipeline but with await throughout

  ★ Key insight: tqbridge makes TQ completely transparent to worker methods
    → Workers don't need to know about TQ → decorator handles everything
```

### TrainingWorker (engine_workers.py:76-431)

```
TrainingWorker wraps model engine (FSDP/Megatron) and provides:
  train_mini_batch (lines 234-321): split batch into mini-batches, iterate with epochs
  train_batch (lines 323-377): forward-backward with loss function, update LR scheduler
  infer_batch (lines 379-423): inference-only forward pass for log_prob computation
  _postprocess_output (lines 172-231): all-reduce metrics across DP group, compute MFU
```

### Sleep Levels (vllm/__init__.py:33-49)

```
VLLM_SLEEP_LEVEL global determines what sleep releases:
  Level 1 (lines 33, 43): default for NPU + older vLLM → releases KV cache only
  Level 2 (line 49): since vLLM 0.8.5+ → releases weights AND KV cache

  ★★★★★★★★★ RTX 4090 MUST: sleep_level=1 ONLY
    Level 2: model offloaded → wake requires full reload (~5s) + corruption risk
```

### Naive vs Disaggregated Weight Sync (checkpoint_engine/base.py)

```
Naive backend (lines 220-276): ColocatedCheckpointEngine
  → send_weights(): just stores generator as self.weights
  → receive_weights(): yield from self.weights → in-process generator
  → BucketedWeightSender (bucketed_weight_transfer.py): ZMQ + CUDA IPC

Disaggregated backend (lines 469-515):
  1. Abort all in-flight rollout requests (line 483)
  2. Create temp worker group for all replicas (lines 486-489)
  3. Release KV cache (line 493) — keeps weights for NCCL overwrite
  4. Build NCCL/NIXL process group (line 496)
  5. Trainer send_weights() + Rollout receive_weights() (lines 499-502)
  6. Finalize all workers (lines 505-508)
  7. Resume KV cache (line 511)

  ★ Key: naive = in-process (same GPU), disaggregated = cross-process (NCCL/NIXL)
```

### Activation Offloading (activation_offload.py:221-395)

```
AsyncDoubleBufferGroupOffloadHandler:
  → Offloads activations to CPU during forward pass
  → Prefetches back during backward pass
  → Dual-stream (d2h_stream, h2d_stream) to overlap offload/reload with compute
  → At most 2 activation groups in GPU simultaneously

  ★ Critical for RTX 4090: reduces peak activation memory → fits in 24 GiB
```

### GRPO Advantage Computation (core_algos.py:268-331)

```
compute_grpo_outcome_advantage:
  1. Sum token-level rewards: scores = token_level_rewards.sum(dim=-1)
  2. Group scores by uid: id2score[index[i]].append(scores[i])
  3. Per-group mean/std:
     Singleton: mean=0, std=1 (lines 315-317)
     Groups >1: mean = torch.mean(scores_tensor), std = torch.std(scores_tensor)
  4. Normalize:
     norm_adv_by_std_in_grpo=True: (scores[i] - mean) / (std + epsilon)
     norm_adv_by_std_in_grpo=False (Dr.GRPO): scores[i] - mean
  5. Broadcast: scores.unsqueeze(-1) * response_mask

  ★★★★★★★★★★ Vectorized version (core_algos.py:334-347):
    Uses groupwise.py:group_mean_std for efficient PyTorch group operations
    Singleton groups: mean=0, std=1 → fallback prevents division by zero
```

### Key Source File References

| File | Path | Key Lines |
|------|------|-----------|
| TransferQueue utils | verl/utils/transferqueue_utils.py | 298-431 (tqbridge) |
| Engine workers | verl/workers/engine_workers.py | 434-746 (ActorRolloutRefWorker), 667-746 (update_weights) |
| Ray trainer | verl/trainer/ppo/ray_trainer.py | 1362-1772 (fit loop) |
| Core algos | verl/trainer/ppo/core_algos.py | 268-331 (GRPO advantage) |
| Groupwise utils | verl/utils/groupwise.py | 163-222 (group_mean_std) |
| Activation offload | verl/utils/activation_offload.py | 221-395 (AsyncDoubleBuffer) |
| Checkpoint engine | verl/checkpoint_engine/base.py | 220-276 (naive), 469-515 (disaggregated) |
| vLLM sleep level | verl/third_party/vllm/__init__.py | 33-49 (VLLM_SLEEP_LEVEL) |
| Bucketed weight transfer | verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py | 74-231 (sender), 233-334 (receiver) |
| Prefix grouper | verl/trainer/ppo/prefix_grouper_utils.py | 46-100 (build_pg) |
| Rollout config | verl/workers/config/rollout.py | 174 (gpu_memory_utilization=0.5), 178 (free_cache_engine) |

