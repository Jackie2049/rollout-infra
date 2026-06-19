# verl V1 Sync PPOTrainer GRPO Training Loop — End-to-End Deep Reading

> 2026-06-19 | Complete lifecycle tracing from rollout generation to policy update
> RTX 4090 #1 training path: sync trainer + naive checkpoint + FSDP + bypass_mode
> ★★★★★★★★ 10-phase training step lifecycle, TransferQueue-centric data flow
> ★★★★★★★★ bypass_mode = skip entire old_log_prob forward pass (18Ψ→3.8Ψ)

---

## 1. Complete Call Tree

```
PPOTrainerSync.fit(agent_loop_manager)                    [trainer_base.py:299]
  |-- PPOTrainer.init()                                    [trainer_base.py:140-151]
  |     |-- PPOTrainer._setup()                            [trainer_base.py:152-281]
  |     |     |-- _init_tokenizer()                        [trainer_base.py:500-511]
  |     |     |-- _init_dataloader()                       [trainer_base.py:512-559]
  |     |     |-- _init_resource_pool_mgr()                [trainer_base.py:570-624]
  |     |     |-- ActorRolloutRefWorker (Ray remote)       [engine_workers.py:434]
  |     |     |-- TrainingWorker (Ray remote, if critic)   [engine_workers.py:76]
  |     |     |-- RewardLoopManager init                   [experimental/reward_loop/]
  |     |     |-- LLMServerManager.create()                [rollout/llm_server.py]
  |     |     |-- CheckpointEngineManager(config, wg, replicas) [checkpoint_engine/base.py:345]
  |     |     |-- checkpoint_manager.sleep_replicas()      [checkpoint_engine/base.py:432]
  |     |     |-- _load_checkpoint()                       [trainer_base.py:626-674]
  |     |-- PPOTrainerSync.on_init_end()                   [trainer_sync.py:31-33]
  |           |-- checkpoint_manager.update_weights()      [checkpoint_engine/base.py:470-480]
  |
  |-- Training loop (while epoch < total_epochs and steps <= total_training_steps)
  |     [trainer_base.py:345-398]
  |     |-- on_step_begin()                                [trainer_base.py:480-482]
  |     |-- PPOTrainer.step(metrics, timing_raw)           [trainer_base.py:404-456]
  |     |     |-- Step 1: _add_batch_to_generate()         [trainer_base.py:1094-1113]
  |     |     |     |-- train_dataloader next batch
  |     |     |     |-- Assign uid, global_steps
  |     |     |     |-- tq.kv_batch_put(keys, partition="train", tags)
  |     |     |     |-- agent_loop_manager.generate_sequences(batch)
  |     |     |           |-- ray.get([worker.generate_sequences.remote(chunk)])
  |     |     |                 |-- AgentLoopWorkerTQ.generate_sequences()
  |     |     |                       |-- asyncio background tasks per sample
  |     |     |                       |-- _run_prompt()
  |     |     |                             |-- n parallel _run_agent_loop()
  |     |     |                             |-- agent_loop.run(sampling_params)
  |     |     |                             |-- _compute_score(outputs)
  |     |     |                             |-- _agent_loop_postprocess()
  |     |     |                                   |-- tq.async_kv_batch_put(keys, fields, tags)
  |     |     |
  |     |     |-- Step 2: on_sample_begin()                [trainer_base.py:489-491]
  |     |     |-- replay_buffer.sample(global_steps, "train", batch_size)
  |     |     |     |-- _sync_metadata_from_transfer_queue()
  |     |     |     |-- _has_enough_samples() -> poll loop
  |     |     |     |-- _drop_max_off_policy_samples()
  |     |     |-- PPOTrainerSync.on_sample_end()           [trainer_sync.py:40-42]
  |     |     |     |-- checkpoint_manager.sleep_replicas()  [trainer_sync.py:42]
  |     |     |
  |     |     |-- Step 3: [OPTIONAL] _compute_reward_colocate()  [trainer_base.py:1115-1118]
  |     |     |     (raises NotImplementedError -- reward computed in AgentLoop)
  |     |     |
  |     |     |-- Step 4: _balance_batch(batch, metrics)   [trainer_base.py:1139-1163]
  |     |     |     |-- upsample_batch_to_divisible_size()
  |     |     |     |-- get_seqlen_balanced_partitions()
  |     |     |
  |     |     |-- Step 5: _compute_old_log_prob(batch, metrics)  [trainer_base.py:1165-1224]
  |     |     |     |-- IF bypass_mode:
  |     |     |     |     |-- tq.kv_batch_get(["rollout_log_probs"])
  |     |     |     |     |-- data["old_log_probs"] = data.pop("rollout_log_probs")
  |     |     |     |     |-- tq.kv_batch_put(keys, fields=data)
  |     |     |     |-- ELSE (full recomputation):
  |     |     |     |     |-- actor_rollout_wg.compute_log_prob(batch)
  |     |     |     |     |-- tq.kv_batch_get(["entropy", "log_probs", ...])
  |     |     |     |     |-- response_from_nested(log_probs, mask) -> old_log_probs
  |     |     |     |     |-- tq.kv_batch_put(keys, fields={"old_log_probs", "entropy"})
  |     |     |
  |     |     |-- Step 6: [OPTIONAL] _compute_ref_log_prob(batch)  [trainer_base.py:1226-1250]
  |     |     |     |-- IF ref_in_actor (LoRA mode): no_lora=True
  |     |     |     |-- ELSE: ref_policy_wg.compute_ref_log_prob(batch)
  |     |     |     |-- data["ref_log_prob"] = response_from_nested(...)
  |     |     |     |-- tq.kv_batch_put(keys, fields={"ref_log_prob"})
  |     |     |
  |     |     |-- Step 7: [OPTIONAL] _compute_values(batch)  [trainer_base.py:1252-1266]
  |     |     |     |-- critic_wg.infer_batch(batch)
  |     |     |     |-- tq.kv_batch_get + response_from_nested -> values
  |     |     |     |-- tq.kv_batch_put(keys, fields={"values"})
  |     |     |
  |     |     |-- Step 8: _compute_advantage(batch, metrics)  [trainer_base.py:1268-1327]
  |     |     |     |-- tq.kv_batch_get(fields=["uid","response_mask","rm_scores",
  |     |     |     |     "rollout_log_probs","old_log_probs","ref_log_prob","values"])
  |     |     |     |-- IF use_kl_in_reward: apply_kl_penalty()
  |     |     |     |-- ELSE: token_level_rewards = rm_scores
  |     |     |     |-- IF rollout_correction (not bypass): compute_rollout_correction()
  |     |     |     |-- compute_advantage_for_multi_trajectories()  [v1/utils.py:23-92]
  |     |     |     |     |-- IF GRPO: identify final sessions per uid
  |     |     |     |     |     |-- core_algos.compute_grpo_outcome_advantage()
  |     |     |     |     |           |-- scores = rewards.sum(dim=-1)
  |     |     |     |     |           |-- Group by uid -> id2score
  |     |     |     |     |           |-- IF len(group)==1: mean=0, std=1 ★★★★★★★★ DEGENERATION
  |     |     |     |     |           |-- Normalize: (score - mean) / (std + eps)
  |     |     |     |     |           |-- Broadcast: unsqueeze(-1) * mask
  |     |     |     |     |-- Scatter final scores to all rows in batch
  |     |     |     |-- response_to_nested(advantages, returns)
  |     |     |     |-- tq.kv_batch_put(keys, fields={advantages, returns, ...})
  |     |     |
  |     |     |-- Step 9: [OPTIONAL] _update_critic(batch)  [trainer_base.py:1329-1349]
  |     |     |
  |     |     |-- Step 10: _update_actor(batch, metrics)  [trainer_base.py:1351-1381]
  |     |     |     |-- actor_rollout_wg.update_actor(batch)
  |     |     |     |     |-- actor.train_mini_batch(data)
  |     |     |     |           |-- make_iterator(mini_batch_size, epochs, seed)
  |     |     |     |           |-- FOR each mini_batch in dataloader:
  |     |     |     |           |     |-- train_batch(mini_batch_td)
  |     |     |     |           |           |-- engine.train_batch(data, loss_fn=ppo_loss)
  |     |     |     |           |           |     |-- model forward pass
  |     |     |     |           |           |     |-- loss_fn = ppo_loss [losses.py:57-144]
  |     |     |     |           |           |           |-- compute_policy_loss_vanilla() [core_algos.py:1279-1369]
  |     |     |     |           |           |           |     |-- ratio = exp(log_prob - old_log_prob)
  |     |     |     |           |           |           |     |-- pg_losses = max(-A*ratio, -A*clip(ratio))
  |     |     |     |           |           |           |     |-- Dual-clip for negative advantages
  |     |     |     |           |           |           |-- entropy_loss (if enabled)
  |     |     |     |           |           |           |-- KL loss (if use_kl_loss)
  |     |     |     |           |           |           |-- total = pg - ent*coeff + kl*coef
  |     |     |     |           |           |           |-- backward + optimizer step
  |     |     |     |           |-- lr scheduler step (on last mini-batch)
  |     |
  |     |-- _save_checkpoint() (if save_freq triggered)    [trainer_base.py:676-738]
  |     |-- PPOTrainerSync.on_step_end()                   [trainer_sync.py:35-38]
  |     |     |-- checkpoint_manager.update_weights(global_steps)
  |     |           |-- IF backend="naive": actor_wg.update_weights(mode="naive")
  |     |           |     |-- rollout.resume(tags=["weights"]) (if free_cache_engine)
  |     |           |     |-- LoRA base sync (if needed)
  |     |           |     |-- rollout.update_weights(per_tensor_param)
  |     |-- _validate() (if test_freq triggered)           [trainer_base.py:740-891]
  |     |-- _compute_metrics(batch, metrics)               [trainer_base.py:1383-1477]
  |     |-- tq.kv_clear(keys, partition_id)                [trainer_base.py:388]
  |     |-- global_steps += 1
```

---

## 2. Training Step Lifecycle (One Complete GRPO Step)

### Phase 0: Weight Sync from Previous Step (on_step_end from prev iteration)
- `PPOTrainerSync.on_step_end()` calls `checkpoint_manager.update_weights()`
- For sync trainer with `backend="naive"`: in-process weight transfer from actor engine to rollout replicas
- Rollout is in "sleep" state (weights freed); must `resume(tags=["weights"])` before receiving new weights
- If LoRA: two-phase sync (base weights first if first step, then LoRA deltas)

### Phase 1: Rollout Generation (_add_batch_to_generate)
- Sample batch from `train_dataloader` (StatefulDataLoader)
- Assign UUID to each prompt, attach `global_steps`
- Register prompts in TransferQueue (`tq.kv_batch_put`) with `tags={"is_prompt":True, "status":"pending"}`
- Call `agent_loop_manager.generate_sequences(batch)` — **fire-and-forget** (non-blocking)
- Each sample spawns `n` parallel agent loops (rollout.n = GRPO group size)
- Each agent loop generates via LLM server (vLLM/SGLang), computes reward score
- Output key format: `{uid}_{session_id}_{index}`
- Reward computed **inside AgentLoop** by `_compute_score()` using rule-based or reward-model-based reward

### Phase 2: Replay Buffer Sampling
- `replay_buffer.sample(global_steps, "train", batch_size)`
- Polls TransferQueue metadata until enough `finished` prompts accumulate
- Selects oldest prompts first (sorted by global_steps to reduce staleness)
- Applies `max_off_policy_threshold` / `max_off_policy_strategy` ("drop" or "wait")
- Returns `KVBatchMeta(keys, tags)` — keys reference data stored in TransferQueue

### Phase 3: Sleep Replicas (PPOTrainerSync.on_sample_end)
- `checkpoint_manager.sleep_replicas()` — frees rollout replica GPU memory (weights + KV cache)
- ★★★★★★★★ After sampling, rollout replicas sleep to free memory for training

### Phase 4: Reward Computation (OPTIONAL, currently NotImplementedError)
- In practice, rewards are computed **inside the AgentLoop** during Phase 1
- Reward scores stored in TransferQueue fields `rm_scores`, `reward_model`, `extra_fields`

### Phase 5: Batch Balancing (_balance_batch)
- Upsample batch to divisible size (LCM of dp_size, mini_batch sizes)
- Sequence-length balancing across DP ranks for even memory distribution

### Phase 6: Old Log Prob Computation (_compute_old_log_prob)
**Two operating modes:**

- **bypass_mode=True**: Directly reuse `rollout_log_probs` as `old_log_probs`
  - Fetch from TransferQueue, rename, write back
  - **No forward pass needed** — 18Ψ→3.8Ψ savings on RTX 4090

- **bypass_mode=False (decoupled)**: Recompute via actor forward pass
  - `actor.infer_batch(data)` → engine.infer_batch
  - Convert `log_probs` from nested to padded

### Phase 7: Ref Log Prob (OPTIONAL, if use_reference_policy)
- If LoRA: ref log probs from actor with LoRA disabled
- Otherwise: separate ref_policy_wg

### Phase 8: Advantage Computation (_compute_advantage)
- Fetch all fields from TransferQueue
- IF `use_kl_in_reward`: compute KL penalty, adjust token_level_rewards
- ELSE: `token_level_rewards = rm_scores`
- IF rollout_correction (not bypass): compute IS weights
- **GRPO algorithm**:
  - `scores = token_level_rewards.sum(dim=-1)` (outcome reward per trajectory)
  - Group by uid (each group = n trajectories from same prompt)
  - Per-group: mean and std of scores
  - ★★★★★★★★ Singleton group (n=1): mean=0.0, std=1.0 (REINFORCE degeneration)
  - Normalize: `(score - mean) / (std + epsilon)`
  - Broadcast: `advantages = normalized_score.unsqueeze(-1) * response_mask`

### Phase 9: Critic Update (OPTIONAL, not used in pure GRPO)
- GRPO is outcome-only, no critic needed

### Phase 10: Actor Update (_update_actor)
- Mini-batch training loop:
  - For each epoch, for each mini_batch:
    - Forward pass: compute `log_probs`, `entropy`
    - Loss = `ppo_clip(pg_loss) - entropy_coeff*entropy + kl_coef*kl_loss`
    - PPO-clip with dual-clip for negative advantages
    - Backward pass + optimizer step
  - LR scheduler step on last mini-batch

### Post-Step: Weight Sync + Cleanup
- `checkpoint_manager.update_weights()` — wake up replicas, sync weights
- `tq.kv_clear(keys, partition_id)` — cleanup TransferQueue data
- Optional checkpoint save and validation

---

## 3. Data Flow Diagram

```
[Dataloader] --> prompts with uid --> [TransferQueue: "pending" tags]
                                          |
[AgentLoopWorkers] --> generate n trajectories per prompt
  |-- LLM Server (vLLM/SGLang) --> response_ids, logprobs
  |-- _compute_score() --> rm_scores
  |-- _agent_loop_postprocess() --> TransferQueue PUT
                                          |
[ReplayBuffer] --> sample finished groups --> KVBatchMeta(keys, tags)
                                          |
[_balance_batch] --> reorder by seqlen balance --> balanced KVBatchMeta
                                          |
[_compute_old_log_prob] --> TransferQueue GET -> compute/reuse -> PUT
  |-- bypass: GET rollout_log_probs -> rename -> PUT old_log_probs
  |-- full: actor forward -> GET log_probs -> convert -> PUT old_log_probs + entropy
                                          |
[_compute_ref_log_prob] --> (optional) actor/ref forward -> PUT ref_log_prob
                                          |
[_compute_advantage] --> TransferQueue GET all fields -> compute -> PUT
  |-- GET: uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values
  |-- compute: token_level_rewards, GRPO advantages (group normalized), returns
  |-- PUT: advantages, returns
                                          |
[_update_actor] --> TransferQueue GET -> mini-batch training loop
  |-- Each mini_batch: forward -> loss -> backward -> optimizer step
                                          |
[TransferQueue CLEAR] --> tq.kv_clear(keys, partition_id)
                                          |
[Weight Sync] --> actor engine -> rollout replicas (wake up + update_weights)
```

---

## 4. Memory Lifecycle Key Moments

### A. Initialization Phase
- Actor model loaded onto GPU (FSDP sharding)
- Reference model loaded if needed (LoRA-detached actor or separate worker)
- Rollout replicas sleep after init (KV cache + weights freed)

### B. Rollout Generation Phase (GPU memory peak)
- Rollout replicas wake up (resume weights + KV cache)
- LLM server generates n trajectories per prompt — **peak GPU memory for KV cache**
- Rollout replicas sleep after sampling completes
- Agent loop outputs stored in TransferQueue (CPU memory, not GPU)

### C. Old Log Prob Computation
- **bypass_mode=True**: No GPU memory needed (TransferQueue CPU operations)
- **bypass_mode=False**: Actor forward pass needed — model weights + batch data on GPU

### D. Advantage Computation (CPU memory)
- All advantage computation on CPU (trainer controller side)
- Data fetched from TransferQueue (CPU-backed), computed, written back

### E. Actor Update Phase (GPU memory peak for training)
- Mini-batch training loop processes data on GPU
- FSDP: unshard → forward → reshard → unshard → backward → reshard → optimizer
- Per-unit LoRA summon (#6512): only summon LoRA parameters, 60 GiB → 6-8 GiB peak

### F. Weight Sync Phase
- Actor engine: `get_per_tensor_param()` extracts current weights
- Rollout: `resume(tags=["weights"])` allocates weight memory
- Direct in-process transfer (naive backend)
- LoRA: two-phase (base + delta)

### G. Cleanup
- `tq.kv_clear(keys, partition_id)` — frees TransferQueue storage (CPU memory)
- Rollout replicas remain in sleep state until next step

---

## 5. bypass_mode Integration Points

### Integration Point 1: _compute_old_log_prob (trainer_base.py:1165-1224)
```python
# Line 1171-1172: Check bypass_mode
bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
if bypass_recomputing_logprobs:  # SKIP actor forward pass entirely
    data = tq.kv_batch_get(keys, partition, select_fields=["rollout_log_probs"])
    data["old_log_probs"] = data.pop("rollout_log_probs")  # Direct reuse
    tq.kv_batch_put(keys, partition, fields=data)
    return batch  # No actor inference needed!
```

### Integration Point 2: _compute_advantage (trainer_base.py:1290-1297)
```python
# In decoupled mode (not bypass), compute IS weights
rollout_correction = (
    rollout_corr_config is not None
    and "rollout_log_probs" in data.batch
    and not bypass_recomputing_logprobs
)
if rollout_correction:  # Only in decoupled mode
    data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
```
★★★★★★★★ bypass_mode skips IS weight computation since pi_old = pi_rollout (no gap to correct)

### Integration Point 3: Policy Loss (losses.py:57-144)
- In bypass_mode: `old_log_probs` = `rollout_log_probs`
- PPO ratio: `ratio = exp(log_prob_theta - log_prob_rollout) = pi_theta / pi_rollout`
- 2-policy formulation instead of 3-policy

---

## 6. Key Config Parameters (RTX 4090 GRPO)

### Algorithm Config
| Parameter | RTX 4090 Value | Notes |
|-----------|---------------|-------|
| adv_estimator | "grpo" | MUST set to GRPO |
| norm_adv_by_std_in_grpo | True | Original GRPO normalization |
| rollout_correction.bypass_mode | True | Skip old_log_prob forward (18Ψ→3.8Ψ) |
| rollout_correction.loss_type | "ppo_clip" | PPO-clip in bypass mode |

### Actor Config
| Parameter | RTX 4090 Value | Notes |
|-----------|---------------|-------|
| strategy | "fsdp" | FSDP backend for training |
| clip_ratio | 0.2 | PPO clipping epsilon |
| ppo_epochs | 1 | PPO epochs per step |

### Rollout Config
| Parameter | RTX 4090 Value | Notes |
|-----------|---------------|-------|
| n | 8-16 | ★★★★★★★★ GRPO group size, MUST > 1 |
| enforce_eager | True | SM89: no cudagraph with DSV4 |
| free_cache_engine | True | Free KV cache during sleep |
| checkpoint_engine.backend | "naive" | In-process weight sync |

### Trainer Config
| Parameter | RTX 4090 Value | Notes |
|-----------|---------------|-------|
| trainer_mode | "sync" | MUST sync for RTX 4090 #1 |
| use_critic | False | GRPO is outcome-only |

---

## 7. RTX 4090 GRPO Training — Complete Rules

### MUST DO
1. trainer_mode = "sync" (colocated, no disaggregation)
2. rollout.n >= 8 (GRPO group size, n=1 = REINFORCE degeneration)
3. bypass_mode = True (skip old_log_prob forward, 18Ψ→3.8Ψ)
4. enforce_eager = True (SM89, no cudagraph for DSV4)
5. checkpoint_engine.backend = "naive" (in-process, zero IPC overhead)
6. actor.strategy = "fsdp" (FSDP for training)
7. use_critic = False (GRPO is outcome-only)
8. LoRA rank = 32 (NOT 64, see #6782 EOS bug)
9. ref_in_actor = True (LoRA detached as ref, no separate worker)
10. free_cache_engine = True (free KV cache during sleep)

### MUST NOT
1. rollout.n = 1 (REINFORCE degeneration — no variance reduction)
2. use ZeRO-3 on single GPU (pure overhead)
3. use ZeRO-2 + Muon (BLOCKED, #7939 closed)
4. LoRA rank = 64 (breaks EOS in vLLM rollout, #6782)
5. bypass_mode = False (wastes 15Ψ on old_log_prob forward)
6. checkpoint_engine.backend != "naive" (IPC overhead unnecessary for sync)
7. enforce_eager = False (cudagraph fails on SM89 with DSV4)
8. overlap_comm = True (multi-stream NaN on single GPU, #8061)
9. gradient_clipping = 0 (ALWAYS set 1.0, #8068)
10. sleep_level = 2 (full re-transfer, use sleep_level=1 LoRA adapter)

---

## 8. TransferQueue-Centric Architecture

★★★★★★★★ V1 trainer uses TransferQueue (tq) as the central data store

- All intermediate data flows through TransferQueue
- Keys: `{uid}_{session_id}_{index}` format
- Tags: metadata (seq_len, global_steps, status)
- Fields: tensor data (prompts, responses, rm_scores, log_probs, etc.)
- Enables fire-and-forget rollout generation and replay buffer sampling
- Decouples rollout generation from training step — rollout can complete asynchronously

### TransferQueue Operations in One Step
```
PUT: prompts (Phase 1)
PUT: rollout outputs (Phase 1, async)
GET: sample keys (Phase 2)
GET: old_log_probs / rollout_log_probs (Phase 6)
PUT: old_log_probs (Phase 6)
GET: ref_log_probs (Phase 7)
PUT: ref_log_prob (Phase 7)
GET: all advantage fields (Phase 8)
PUT: advantages, returns (Phase 8)
GET: actor update data (Phase 10)
CLEAR: all keys (Post-Step)
```

---

## 9. Sleep/Wake Lifecycle Pattern

```
[Init] → sleep_replicas() → weights freed, KV cache freed
  ↓
[Step N starts] → Phase 1: wake for rollout generation
  ↓ (AgentLoop generates n trajectories per prompt)
  ↓
[Phase 3] → on_sample_end → sleep_replicas() again (free memory for training)
  ↓ (Advantage computation on CPU, actor update on GPU)
  ↓
[Post-Step] → on_step_end → update_weights() → wake + weight sync
  ↓
[Step N+1 starts] → Phase 1 again...
```

★★★★★★★★ This sleep/wake pattern is ESSENTIAL for RTX 4090 GRPO:
- Peak memory during rollout = model weights + KV cache (generation)
- Peak memory during training = FSDP unshard + gradients + optimizer (per-unit LoRA summon)
- By sleeping during training, rollout memory is freed → only one peak at a time

---

## 10. Key Source Files

- `verl/trainer/ppo/v1/trainer_base.py` — PPOTrainer base, step() method (404-456), fit() loop (299-403)
- `verl/trainer/ppo/v1/trainer_sync.py` — PPOTrainerSync (on_init_end, on_step_end, on_sample_end)
- `verl/trainer/ppo/v1/replay_buffer.py` — ReplayBuffer with TransferQueue integration
- `verl/trainer/ppo/v1/utils.py` — compute_advantage_for_multi_trajectories
- `verl/trainer/ppo/v1/agent_loop_tq.py` — AgentLoopWorkerTQ, TransferQueue-based rollout
- `verl/trainer/ppo/core_algos.py` — compute_grpo_outcome_advantage (268-331), compute_policy_loss_vanilla (1279-1369)
- `verl/trainer/ppo/ray_trainer.py` — compute_advantage() dispatcher (187-268)
- `verl/workers/engine_workers.py` — ActorRolloutRefWorker (434+), TrainingWorker (76+)
- `verl/workers/utils/losses.py` — ppo_loss (57-144)
- `verl/trainer/config/algorithm.py` — AlgoConfig, RolloutCorrectionConfig
- `verl/checkpoint_engine/base.py` — CheckpointEngineManager

---

## References

- verl V1 trainer deep reading: notebook/projects/verl-v1-unified-trainer-architecture-reading.md
- verl FSDP weight sync: notebook/projects/verl-fsdp-weight-sync-mechanism-reading.md
- verl bypass mode: notebook/projects/ (core_algos.py lines 2351-2498)
- verl #6512 per-unit LoRA summon: notebook/projects/verl-6512-per-unit-lora-summon-reading.md
- GRPO singleton degeneration: notebook/projects/cross-framework-grpo-advantage-comparison.md
- RTX 4090 GRPO runbook: notebook/projects/rtx4090-grpo-training-runbook.md
