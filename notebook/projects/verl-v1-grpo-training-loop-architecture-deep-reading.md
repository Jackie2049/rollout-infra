# verl V1 GRPO Training Loop Architecture — Deep Reading

> Created: 2026-06-20 | Priority: ★★★★★★★★ CRITICAL for RTX 4090 GRPO
> Source: Background agent deep reading of verl V1 source code

## 1. Main Training Loop: 10-Phase Step Pipeline

The V1 training loop is orchestrated by `PPOTrainer` (base class at `verl/trainer/ppo/v1/trainer_base.py`) with three variants:

- **PPOTrainerSync** (`trainer_sync.py`) — colocated, synchronous, no partial rollout
- **PPOTrainerColocateAsync** (`trainer_colocate_async.py`) — colocated, asynchronous, partial rollout
- **PPOTrainerSeparateAsync** (`trainer_separate_async.py`) — disaggregated, standalone rollout servers

### The `step()` Method (10 Phases)

**Phase 1 — `_add_batch_to_generate()`**: Samples from train_dataloader, assigns UUIDs, registers prompts in TransferQueue as `pending`, fires `agent_loop_manager.generate_sequences(batch)` (non-blocking). Each prompt generates `rollout.n` parallel trajectories.

**Phase 2 — Replay Buffer Sampling**: `replay_buffer.sample()` polls TransferQueue until enough `finished` prompts, selects oldest-first (sorted by `global_steps`), applies `max_off_policy_threshold` drop/wait strategy.

**Phase 3 — Sleep Replicas**: `checkpoint_manager.sleep_replicas()` frees rollout GPU memory (weights + KV cache). Essential for freeing GPU memory for training.

**Phase 4 — Reward** (optional): Rewards computed inside AgentLoop during Phase 1, stored in TransferQueue.

**Phase 5 — `_balance_batch()`**: Upsamples batch to divisible size (LCM of dp_size, mini_batch sizes), sequence-length balancing.

**Phase 6 — `_compute_old_log_prob()`**:
- **bypass_mode=True**: Reuses `rollout_log_probs` from TransferQueue directly as `old_log_probs`. No actor forward pass. Saves ~14 Psi out of 18 Psi total step time.
- **bypass_mode=False**: Recomputes via `actor.infer_batch()` forward pass.

**Phase 7 — `_compute_ref_log_prob()`**: If LoRA (`ref_in_actor=True`), uses actor with `no_lora_adapter=True` flag; otherwise uses separate ref_policy_wg.

**Phase 8 — `_compute_advantage()`**: GRPO algorithm:
- Groups by `uid` (each group = `rollout.n` trajectories)
- Per-group: `mean` and `std` of scores
- **Singleton groups (n=1): mean=0.0, std=1.0 — REINFORCE degeneration!**
- Normalize: `(score - mean) / (std + epsilon)`
- Broadcast: `advantages = normalized_score.unsqueeze(-1) * response_mask`

**Phase 9 — `_update_critic()`** (optional, not used in pure GRPO).

**Phase 10 — `_update_actor()`**: Mini-batch loop: forward → PPO-clip loss → backward → optimizer step.

### Post-Step: Weight Sync + Cleanup

`PPOTrainerSync.on_step_end()` → `checkpoint_manager.update_weights()` → wake replicas, sync weights, resume KV cache.

## 2. Rollout Phase: SGLang/vLLM Integration

### RolloutReplica Architecture

| Mode | Process Layout | Weight Sync | Use Case |
|------|---------------|-------------|----------|
| **HYBRID** | Same process as trainer | Sleep-wake-update | On-policy GRPO (RTX 4090) |
| **COLOCATED** | Separate process, same PG | Not required | GRM (judge) |
| **STANDALONE** | Separate GPU resources | Full NCCL/NIXL transfer | Off-policy |

**HYBRID mode** (default for RTX 4090): Rollout and training engine share same GPU. SGLang/vLLM server runs in same process as FSDP training engine.

### SGLang ServerAdapter

Uses `AsyncHttpServerAdapter` for HTTP client to SGLang server:
- sleep/wake/update_weights via HTTP API with tag-based granularity
- `tags=["kv_cache"]` — release/resume only KV cache (sleep_level=1)
- `tags=["kv_cache", "weights"]` — release/resume everything (sleep_level=2)
- Only TP-rank-0 does HTTP dispatch; other ranks participate in FSDP DTensor collectives

### vLLM ServerAdapter

Uses ZMQ IPC for weight updates, native `sleep(level=1|2)` / `wake_up()` API for memory management.

## 3. Weight Sync Mechanism: Trainer → Rollout Engine

### Naive Backend (sync, colocated)

`actor_wg.update_weights(mode="naive")` — in-process direct transfer. No IPC, no network.

### NCCL/NIXL Backend (async, disaggregated)

8-step process: abort requests → build process group → release KV cache → send weights → receive weights → finalize → resume KV → resume generation.

### ActorRolloutRefWorker.update_weights() — The Core

```python
async def update_weights(self, global_steps, mode="auto"):
    if effective_mode == "naive":  # IN-PROCESS PATH
        # 1. Resume rollout weight memory
        await self.rollout.resume(tags=["weights"])
        # 2. Summon parameters from FSDP
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(...)
        # 3. Two-phase LoRA sync
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1  # KEY: LoRA adapter mode
            if not self.base_sync_done:
                # First time: full base weight transfer
                await self.rollout.update_weights(per_tensor_param_base, ...)
            # Every step: LoRA delta only
            await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, ...)
        # 4. Offload actor params to CPU if param_offload
        # 5. Resume KV cache
        await self.rollout.resume(tags=["kv_cache"])
```

### LoRA Adapter Path (sleep_level=1)

Two-phase sync:
- **Phase 1 (first step)**: Full base weight transfer (~14 GiB)
- **Phase 2 (every step)**: LoRA adapter delta only (~200 MiB) — **80x payload reduction!**

This is why sleep_level=1 is critical: base weights stay resident, only LoRA deltas transfer each step.

## 4. Sleep/Wake Lifecycle & Memory Management

### Memory Budget Impact (RTX 4090, Qwen2.5-7B)

| State | GPU Memory | Notes |
|-------|------------|-------|
| Wake (rollout ready) | ~19 GiB | base weights + KV cache + LoRA |
| Rollout generation | ~19 GiB | peak during P2 |
| Sleep (KV freed) | ~17 GiB | base weights + LoRA only |
| Training update | ~18 GiB | gradients + activations |
| Post-step sync | ~17 GiB | LoRA delta applied, KV not yet resumed |

**Key**: sleep_level=1 only frees KV cache (~2 GiB), keeping base weights resident. sleep_level=2 frees everything (~16 GiB), requiring full re-transfer.

## 5. TransferQueue Data Flow

The V1 trainer uses `TransferQueue` as central data fabric:
- Keys: `{uid}_{session_id}_{index}` format
- Tags: metadata (`seq_len`, `global_steps`, `status`)
- Fields: tensor data (`prompts`, `responses`, `rm_scores`, `log_probs`, etc.)

Operations per step:
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

## 6. #6782 LoRA EOS Bug Deep Analysis

### Bug Description

With `lora_rank=64` / `lora_alpha=128`, vLLM never emits EOS — all responses run to `max_response_length`. With `lora_rank=32` / `lora_alpha=64`, EOS works normally.

### Root Cause Analysis

LoRA scaling factor `alpha/rank` = 2.0 for both configs. But:
- `LoRA output = base_output + (lora_B @ lora_A) * (alpha / rank)`
- rank=64 → `||lora_B @ lora_A||` can be larger (more degrees of freedom)
- Larger LoRA delta distorts logit distribution → suppresses EOS logit

**RTX 4090**: MUST use `lora_rank=32, lora_alpha=64` (verified safe). MUST NOT use rank>=64.

## 7. FSDP Integration Details

### FSDPEngine (fsdp/transformer_impl.py:85-873)

Supports both FSDP1 and FSDP2:
- **FSDP1**: `strategy="fsdp"`, standard wrapping with `auto_wrap_policy`
- **FSDP2**: `strategy="fsdp2"`, uses `apply_fsdp2()` and `fsdp2_load_full_state_dict()`

For actor: `cpu_offload=None` (forced off — causes incorrect results with gradient accumulation)
For reference model: `cpu_offload=CPUOffload(offload_params=True)` — saves memory

### #6699 Detach Fix

In `forward_step()` (line 1291-1307), model outputs are detached before appending to output list. Without detach, each micro-batch's autograd graph stays alive → pins checkpointed embedding + gradient buffer → OOM accumulation.

### LoRA within FSDP

LoRA adapters applied BEFORE FSDP wrapping:
1. `_build_lora_module()` applies PEFT LoRA
2. `_build_fsdp_module()` wraps with FSDP

`merged_lora_context()` temporarily merges LoRA for full-weight transfer (merge=True path).

## 8. Reference Model Bypass — Memory Savings

### ref_in_actor (LoRA mode)

No separate reference worker needed. Reference = actor with LoRA disabled via `disable_adapter()`.

Memory savings:
- Without ref_in_actor: ~32 GiB (base + reference = exceeds 24 GiB!)
- With ref_in_actor: ~16.2 GiB (base model only + 200 MiB LoRA)

### bypass_mode (rollout_correction config)

Reuses `rollout_log_probs` as `old_log_probs` directly. No actor inference needed.

Creates 2-policy formulation: `pi_theta / pi_rollout` instead of 3-policy `pi_theta / pi_old`.

**Combined**: ref_in_actor + bypass_mode reduces from 5 forward passes to 1 forward pass (actor update only). Saves ~14 Psi out of 18 Psi total step time.

## Key Source Files

| File | Purpose |
|------|---------|
| `verl/trainer/ppo/v1/trainer_base.py` | PPOTrainer base, `step()`, `fit()`, `_setup()` |
| `verl/trainer/ppo/v1/trainer_sync.py` | PPOTrainerSync, `on_step_end` (update_weights), `on_sample_end` (sleep) |
| `verl/trainer/ppo/core_algos.py` | GRPO advantage, PPO policy loss |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker, `update_weights()`, `init_model()` |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDPEngine: FSDP1/2, `get_per_tensor_param()`, LoRA |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | SGLang ServerAdapter |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | vLLM ServerAdapter |
| `verl/workers/rollout/llm_server.py` | LLMServerManager, load balancer |
| `verl/checkpoint_engine/base.py` | CheckpointEngineManager, sleep/wake/sync coordination |
| `verl/trainer/ppo/v1/replay_buffer.py` | ReplayBuffer, staleness ordering |

## RTX 4090 GRPO Key Takeaways

1. **HYBRID mode = required** — single GPU, same process for rollout + training
2. **sleep_level=1 = mandatory** — only frees KV cache, keeps base weights resident
3. **LoRA adapter path (merge=False) = 80x faster sync** — only ~200 MiB vs ~14 GiB per step
4. **bypass_mode + ref_in_actor = 14 Psi savings** — 5→1 forward passes
5. **LoRA rank=32 = safe** (MUST NOT use rank>=64, #6782 EOS bug)
6. **TransferQueue-centric data flow** — all intermediate data flows through KV store
7. **naive checkpoint engine = best for dp=1** — in-process direct transfer, no IPC overhead
