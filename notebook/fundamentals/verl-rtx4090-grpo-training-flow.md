# verl RTX 4090 GRPO Training Complete Flow — Worker Lifecycle + Memory Optimization

> 2026-06-18 | verl-project/verl | RolloutMode=HYBRID + GRPO + LoRA + bypass_mode
> ★★★★★★★★ Complete 11-step training flow for RTX 4090 — memory budget: ~17GB peak → fits 24GB

---

## 1. Three Worker Lifecycle Modes

```
★★★★★★★★★ verl supports 3 RolloutMode values (replica.py:54-68):

HYBRID (★★★★★★★★★ RTX 4090 OPTIMAL):
  → Rollout + training share SAME Ray actor/process per GPU
  → FusedWorker wrapper combines actor+rollout into one process
  → Sleep/Wake GPU memory management:
    → sleep_level=2: full weight sync (releases KV cache + weights + GPU memory)
    → sleep_level=1: LoRA mode (only releases KV cache; base weights remain)
  → Weight sync: naive (in-process Python generator) → ZERO-COPY
  → ★★★★★★★★ BEST for: on-policy PPO/GRPO, single GPU RTX 4090, LoRA training

COLOCATED:
  → Rollout + training share same Ray Placement Group (same GPU) but DIFFERENT processes
  → CheckpointEngineWorker as independent Ray actor per GPU
  → Weight sync: CUDA IPC (ZMQ + torch.multiprocessing) → GPU-direct tensor transfer
  → ★★★★★ BEST for: GRM (LLM-as-judge), MoE models needing separate rollout processes

STANDALONE:
  → Rollout has own ResourcePool → independent GPU set
  → No Sleep/Wake needed → dedicated GPU memory
  → Weight sync: NCCL ProcessGroup or NIXL RDMA
  → ★★★★★ BEST for: off-policy training, disaggregated architecture, multi-GPU clusters
  → NOT feasible on single RTX 4090 (needs 2+ GPU sets)
```

---

## 2. bypass_mode: 18Psi → 3.8Psi Memory Reduction

```
★★★★★★★★★ Standard PPO memory (18Psi):
  → Actor parameters: 2Psi (bf16)
  → Actor gradients: 2Psi (bf16)
  → Optimizer states: 14Psi (fp32) → momentum + variance + master params
  → Reference model: 2Psi (bf16) → KL penalty
  → Total: 18Psi → DOES NOT fit RTX 4090 24GB for 7B

★★★★★★★★★ bypass_mode + LoRA memory (3.8Psi):
  → Frozen base parameters: 2Psi (bf16) → LoRA base weights, not trainable
  → LoRA gradients: ~0.6Psi (bf16) → only adapter params trainable
  → LoRA optimizer states: ~1.2Psi (fp32) → momentum + variance for LoRA only
  → No reference model: 0 → eliminated by bypass_mode!
  → Total: ~3.8Psi → fits RTX 4090 24GB easily

★★★★★★★★★ How bypass_mode eliminates the reference model:
  → apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
  → batch["old_log_probs"] = batch["rollout_log_probs"] ← zero-cost substitution!
  → Policy reduction: 3 policies → 2 policies → pi_old eliminated
  → ★★★★★★★★ On-policy (HYBRID same GPU): pi_rollout = pi_theta → IS ratio = 1.0 → NO correction needed

★★★★★★★★★ bypass_mode loss types:
  → ppo_clip: PPO clipped → ratio = pi_current/pi_rollout → IS correction natural
  → reinforce: REINFORCE → explicit IS weights w = pi_current/pi_rollout
  → ★★★★★★★★ GRPO recommended: ppo_clip + on-policy → IS ratio always 1.0

★★★★★★★★★ Config requirements:
rollout_correction:
  bypass_mode: true
  loss_type: ppo_clip
  rollout_is: null  # on-policy: no IS correction needed
  rollout_rs: null  # on-policy: no RS needed
rollout:
  calculate_log_probs: true  # REQUIRED for bypass_mode
```

---

## 3. Complete GRPO Training Flow on RTX 4090 (11 Steps)

```
★★★★★★★★★ Optimal config: RolloutMode=HYBRID + Weight Sync=naive + GRPO + LoRA + bypass_mode

Step 1: INITIALIZATION (engine_workers.py:500-632)
  → a. Ref model: BYPASS → eliminated entirely (0 memory)
  → b. Actor model: FSDP engine + LoRA + optimizer + LR scheduler → ~0.6Psi trainable
  → c. Rollout engine: vLLM ServerAdapter + 3D DeviceMesh(dp=1, infer_tp=1, infer_pp=1)
  → → lora_as_adapter=True → LoRA not merged → Punica segmented matmul
  → d. Checkpoint engine: naive (in-process Python generator → zero-copy)
  → e. aggressive_empty_cache(force_sync=True) → free GPU memory for vLLM KV cache

Step 2: ROLLOUT PHASE
  → rollout.wake_up(tags=["kv_cache", "weights"]) → vLLM occupies full GPU
  → reset_prefix_cache() → clear stale KV from previous step
  → generate_sequences() via vLLM async server
  → ★★★★★★★★ GRPO n=8: same prompt → 8 responses → sticky session → prefix caching
  → calculate_log_probs=True (bypass_mode requirement)
  → collect: input_ids + responses + rollout_log_probs

Step 3: SLEEP PHASE
  → rollout.sleep(level=1) → LoRA mode: only release KV cache
  → vLLM releases KV cache blocks → ~2-4GB freed
  → Base weights remain on GPU (LoRA mode)
  → GPU memory now available for FSDP training

Step 4: DATA ASSEMBLY
  → batch.repeat(rollout_n=8, interleave=True) → 8 responses per prompt
  → compute_response_mask() → mark valid response tokens
  → balance_batch() → equalize valid token counts across DP ranks

Step 5: REWARD COMPUTATION
  → extract_reward(batch) → rule-based or model-based
  → Outcome-level reward per response (scalar)

Step 6: BYPASS MODE ACTIVATION
  → apply_bypass_mode():
  → batch["old_log_probs"] = batch["rollout_log_probs"] ← zero-cost!
  → policy_loss_config["loss_mode"] = "bypass_mode"
  → No separate ref model forward pass → saves ~2Psi memory

Step 7: KL PENALTY (optional)
  → If use_kl_loss=True (GRPO recommended):
  → KL divergence during actor forward
  → loss = policy_loss + kl_coef * KL(pi_theta || pi_ref)
  → If ref_in_actor=True: actor provides ref log_prob → no separate ref model

Step 8: ADVANTAGE COMPUTATION
  → compute_advantage(batch, adv_estimator="grpo"):
  → Group normalization: A_i = (r_i - mean_group) / (std_group + epsilon)
  → Outcome-only: scores = rewards.sum() → per-sequence scalar
  → ★★★★★★★★ Singleton fallback (n=1): mean=0, std=1 → advantage=raw reward

Step 9: ACTOR UPDATE
  → actor_rollout_wg.update_actor(data):
  → FSDP forward + backward on LoRA params only
  → compute_policy_loss_bypass_mode() → PPO-clip ratio=pi_theta/pi_rollout
  → clip_grad_norm → optimizer.step (LoRA params only)
  → ~0.6Psi gradients + ~1.2Psi optimizer states

Step 10: WEIGHT SYNC (HYBRID naive)
  → checkpoint_manager.update_weights():
  → actor.engine.get_per_tensor_param() → Python generator of (name, tensor)
  → For LoRA: only sync adapter weights (~2.6GB) → or merge LoRA then sync full
  → rollout.update_weights_from_ipc() → receives weights
  → clear_kv_cache() → reset_prefix_cache()

Step 11: WAKE-UP for next rollout
  → rollout.wake_up(tags=["kv_cache"]) → vLLM restores KV cache allocation
  → Ready for next step's generation
```

---

## 4. Memory Timeline Per Step (RTX 4090, 7B LoRA+GRPO+bypass)

```
★★★★★★★★★ Memory budget per training step:

Rollout phase:  ~17GB peak (weights + KV cache + activations)
Sleep phase:    ~12GB (weights remain, KV freed, FSDP uses freed space)
Training phase: ~14GB peak (FSDP forward+backward on LoRA)
Weight sync:    ~2.6GB transfer (LoRA adapter only)
Wake-up:        KV cache reallocated
→ ★★★★★★★★ Total peak: ~17GB → fits 24GB RTX 4090 with 7GB margin!
```

---

## 5. Weight Sync Backend Comparison

```
★★★★★★★★★ Weight sync backend comparison for RTX 4090:

| Backend  | Transfer Method           | Mode     | Overhead | RTX 4090 |
|----------|---------------------------|----------|----------|----------|
| naive    | Python generator (0-copy) | HYBRID   | 0        | ★★★★★★★★ Optimal! |
| CUDA IPC | ZMQ + torch IPC (GPU)    | COLOCATED| Low      | Feasible |
| NCCL     | AllReduce/Broadcast       | STANDALONE| Medium   | Not feasible (PCIe) |
| NIXL     | RDMA P2P ring             | STANDALONE| Lowest   | Not applicable |

★★★★★★★★★ HYBRID + naive = OPTIMAL for RTX 4090 → zero IPC overhead → in-process
```

---

## 6. OOM Cap Mechanism (#6735)

```
★★★★ #6735: OOM when setting cap_memory=True
  → cap_memory limits max memory vLLM rollout worker can consume
  → gpu_memory_utilization=0.5 default → rollout uses 50%, training uses 50%
  → ★★★★★★★★ RTX 4090 LoRA + GRPO: can increase to 0.6-0.7 (smaller training footprint)
  → aggressive_empty_cache() → gc.collect + torch.cuda.empty_cache + sync → 3 retries
  → optimize_memory_for_inference() → set_per_process_memory_fraction(0.95)
  → optimize_memory_for_training() → set_per_process_memory_fraction(0.90)
```

---

## 7. Async Rollout Worker Pool

```
★★★★★★★★★ GlobalRequestLoadBalancer for async rollout:
  → Sticky session: LRU cache → request_id → server_id → prefix caching
  → Least-loaded: min(inflight_requests) → auto load balancing
  → RolloutReplica: per-GPU Ray actor → sleep/wake/generate/update_weights

★★★★★★★★★ RayWorkerGroup management:
  → _init_with_resource_pool() → STRICT_PACK → CPU: max_colocate_count, GPU: 1
  → max_colocate_count=3 → 3 WorkerGroups per GPU (actor+rollout+ref)
  → ★★★★★★★★ GRPO+bypass: max_colocate_count=2 (actor+rollout only, no ref)
  → spawn/fuse: dynamic sub-groups from existing workers
```

---

## Key Findings Summary

★★★★★★★★★ HYBRID = optimal RTX 4090 RolloutMode → in-process → zero IPC overhead → sleep/wake memory management
★★★★★★★★★ bypass_mode: 18Psi→3.8Psi → eliminates ref model → old_log_probs = rollout_log_probs (zero-cost)
★★★★★★★★★ 11-step GRPO flow on RTX 4090: init → rollout → sleep → data → reward → bypass → KL → advantage → update → sync → wake
★★★★★★★★★ Memory timeline: ~17GB peak → fits 24GB RTX 4090 with 7GB margin
★★★★★★★★★ Weight sync: HYBRID+naive = zero-copy Python generator → OPTIMAL
★★★★★★★★★ Sleep levels: level=2 (full release), level=1 (LoRA mode: only KV freed, weights stay)
★★★★★★★★★ LoRA as adapter (not merged) → Punica segmented matmul → rollout efficiency
★★★★★★★★★ GRPO n=8 + sticky session → prefix caching across all group responses
★★★★★★★★★ gpu_memory_utilization=0.5 default → LoRA+GRPO can increase to 0.6-0.7

---

## References

- verl worker lifecycle: notebook/projects/verl-worker-lifecycle-ray-weight-sync-reading.md
- verl async rollout: notebook/projects/verl-async-rollout-architecture-deep-dive.md
- verl GRPO core algos: notebook/projects/verl-grpo-core-algos-reading.md
- verl GRPO source: notebook/projects/verl-grpo-source-reading.md
- Single GPU comparison: notebook/fundamentals/single-gpu-ddp-vs-zero-architecture-comparison.md
- rLLM GRPO guide: notebook/projects/rllm-grpo-practical-training-guide.md
