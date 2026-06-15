# rLLM Tinker Backend Training Loop Internals — Source-Level Analysis

> 2026-06-16 | rllm-org/rllm + thinking-machines-lab/tinker | TinkerBackend | UnifiedTrainer | bypass_mode | Async Trainer
> ★★★★★ rLLM Tinker = client-server架构 → TrainingClient(SDK) → forward_backward + optim_step → 不是pure in-process!
> ★★★★★ Zero-copy = Tinker Service内部共享weight memory → save_weights_for_sampler → SamplingClient → 无Python tensor transfer
> ★★★★★ Auto LoRA rank=32 → train_mlp + train_attn + train_unembed → Tinker Service内部自动merge/unmerge
> ★★★★★ bypass_mode=True(Tinker default) → pi_old=rollout logprobs → KL=0 → LossFnType={ppo, importance_sampling, cispo, dro}

## 1. ★★★★★ Training Loop — 8-Stage Pipeline

```
★★★★★★★ UnifiedTrainer._train_batch_async() (unified_trainer.py):

Stage 1: generate_episodes → TinkerBackend.generate_episodes()
  → interleave_tasks(batch, group_size) → 每task重复group_size次 → 相同task_id → GRPO分组!
  → agent_workflow_engine.execute_tasks(interleaved_batch)

Stage 2: transform episodes → trajectory groups (sync)

Stage 3: rejection sampling (sync) → optional filtering

Stage 4: transform_to_backend_batch → TinkerBackend.transform_to_backend_batch()

Stage 5: process_backend_batch → TinkerPolicyTrainer.forward_backward_from_trajectory_groups()
  → 或: fused_forward_backward_and_optim_step → overlap fwd/bwd with optim!

Stage 6: compute_advantages → TinkerBackend.compute_advantages()

Stage 7: update_policy → TinkerPolicyTrainer.optim_step_future()
  → AdamParams(lr, beta1, beta2, eps) → training_client.optim_step_async()

Stage 8: on_policy_updated → save_checkpoint_and_get_sampling_client()
  → save_weights_and_get_sampling_client_async() → zero-copy weight sync!
```

## 2. ★★★★★ In-Process Architecture — Client-Server, NOT Pure In-Process

```
★★★★★★★ CRITICAL: Tinker不是pure in-process!

  → 实际架构: TrainingClient(Tinker SDK) + Tinker Service → client-server!
  → → 但: No Ray → No separate worker processes → No Ray actor overhead → 比verl+Ray更轻
  → → → Key difference vs verl: 一个TrainingClient handles both forward_backward AND optim_step

★★★★★★★ Weight Sync — Zero-Copy:

  → Step 1: training_client.optim_step_async(adam_params) → optimizer update
  → Step 2: save_weights_for_sampler_async() → 导出LoRA weights到sampler路径
  → Step 3: create_sampling_client(sampler_path) → SamplingClient from SAME weights!
  → → ★★★★★★★ Zero-copy: Tinker Service内部共享weight内存 → 无Python tensor serialization!
  → → → ★★★★★★★★★★★★★★★★★ vs verl → Ray actor → separate rollout/training workers → weight transfer over network!

★★★★★★★ SamplingClient flow (tinker_engine.py):

  class TinkerEngine(RolloutEngine):
      self.sampling_client = None  # Set via set_sampling_client()
      async def get_token_output_from_token_input(token_input):
          sample_response = await self.sampling_client.sample_async(
              prompt=model_input, num_samples=1,
              sampling_params=tinker_sampling_params,
          )
          return sample_response.sequences[0]
```

## 3. ★★★★★ Auto LoRA — Tinker SDK Initialization

```
★★★★★★★ TinkerPolicyTrainer.initialize_async():

  self.training_client = await service_client.create_lora_training_client_async(
      base_model=config.model.name,       # e.g., "Qwen/Qwen3-8B"
      rank=config.model.lora_rank,         # default: 32
      train_unembed=train_unembed,         # default: true
      train_attn=train_attn,              # default: true
      train_mlp=train_mlp,               # default: true
  )

★★★★★★★ Tinker SDK LoraConfig (types/lora_config.py):

  class LoraConfig(StrictBase):
      rank: int              # LoRA dimension (default 32)
      seed: Optional[int]    # Reproducible initialization
      train_unembed: bool = True   # LoRA on output embedding
      train_mlp: bool = True       # LoRA on MLP (including MoE)
      train_attn: bool = True      # LoRA on attention

★★★★★★★ LoRA merge/unmerge:
  → ★★★★★★ Tinker Service内部自动处理 → 无Python代码显式merge/unmerge!
  → → Rollout → SamplingClient → LoRA-merged weights → inference
  → → → Training → TrainingClient → forward on LoRA-merged → backward → LoRA gradients only
  → → → → vs vLLM/SGLang推理 → explicit merge → 一次性 → rLLM训练 → Service内部自动!

★★★★★★★ Default config (tinker.yaml):

  model:
    name: "Qwen/Qwen3-8B"
    lora_rank: 32
    train_unembed: true
    train_attn: true
    train_mlp: true
  rllm:
    algorithm:
      rollout_correction:
        bypass_mode: true   # ← Tinker-specific override!
```

## 4. ★★★★★ bypass_mode — Default True for Tinker

```
★★★★★★★ bypass_mode=True → pi_old = rollout logprobs → KL = 0 → no ref model!

  → Loss function selection (ADV_TO_LOSS_FN_AUTO_MAP):
    → GRPO → "ppo" → PPO clip loss with rollout logprobs as pi_old
    → REINFORCE → "importance_sampling" → IS weighting
    → RLOO → "importance_sampling" → IS weighting (leave-one-out)

★★★★★★★ Tinker LossFnType (types/loss_fn_type.py):

  LossFnType: TypeAlias = Literal[
      "cross_entropy",
      "importance_sampling",
      "ppo",
      "cispo",
      "dro",
  ]

★★★★★★★ How bypass works in Tinker:

  Step 1: Rollout → TinkerEngine → sample_async() → logprobs stored in step.logprobs
  Step 2: trajectory_to_datums() → sampled_logprobs → loss_fn_inputs["logprobs"]
  Step 3: forward_backward_async() → Tinker Service:
    → pi_new = current policy logprobs (forward pass)
    → pi_old = loss_fn_inputs["logprobs"] (rollout logprobs)
    → ratio = exp(pi_new - pi_old)
    → PPO clip: clip(ratio, 1-eps, 1+eps) * advantage
    → ★★★★★★ No KL penalty! → bypass → no ref model → KL=0!

★★★★★★★ KL metrics with bypass:
  → compute_kl_and_entropy_metrics() → kl = rollout_logprobs - training_logprobs
  → → ★★★★★ Measures policy drift from rollout weights → NOT from ref model!
  → → → ★★★★★★ This is KL(pi_current | pi_rollout) → monitors policy change → useful diagnostic!
```

## 5. ★★★★★ GRPO Advantage Computation

```
★★★★★★★ calculate_grpo_advantages_per_group (rl_algo.py):

  def calculate_grpo_advantages_per_group(rewards, norm_adv=True, eps=1e-6):
      if len(rewards) <= 1:
          group_mean, group_std = 0.0, 1.0    # ← single sample → std forced to 1!
      else:
          group_mean = np.mean(rewards)
          group_std = np.std(rewards)

      if norm_adv:
          advantages = (rewards - group_mean) / (group_std + eps)
      else:
          advantages = rewards - group_mean
      return advantages, advantages  # ← (advantages, _unused)

★★★★★★★ Advantage flow:
  → TrajectoryGroup → collect rewards → [traj.reward for traj in group.trajectories]
  → → GRPO advantage per group → same task_id → group_size trajectories
  → → → step.advantage = float(advantage) → broadcast → 所有tokens共享同一scalar!
  → → → → ★★★★★★ vs per-token advantage → step.advantage = [list] → different per token

★★★★★★★ interleave_tasks() → group_size mechanism:

  def interleave_tasks(batch, group_size):
      for item in batch:
          uid = str(item.id) if item.id else str(uuid.uuid4())
          for _ in range(group_size):          # ← 每task重复group_size次!
              tasks.append(item)
              task_ids.append(uid)              # ← 相同task_id → GRPO分组!
```

## 6. ★★★★★ Optimizer + Memory Optimization

```
★★★★★★★ Optimizer: Adam (Tinker SDK handles internally)

  tinker.yaml:
    beta1: 0.9, beta2: 0.95, eps: 1e-8, lr: 2e-5

  training_client.optim_step_async(AdamParams(lr=scheduled_lr, beta1, beta2, eps))

★★★★★★★ Learning rate scheduling (tinker_policy_trainer.py):

  def compute_schedule_lr_multiplier(lr_schedule, warmup_steps_ratio, step, total_steps):
      warmup_steps = int(total_steps * warmup_steps_ratio)
      if step < warmup_steps: return step / warmup_steps  # Linear warmup
      step -= warmup_steps
      if lr_schedule == "linear": return 1 - step / total_steps
      elif lr_schedule == "cosine": return 0.5 * (1 + cos(pi * step / total_steps))
      elif lr_schedule == "constant": return 1

★★★★★★★ Memory optimization:
  → ★★★★★ LoRA (rank=32): Only trains ~0.6GB vs 14GB full model → huge savings!
  → ★★★★★ bypass_mode=True: Eliminates ref model (~14GB savings for 7B)
  → ★★★★★ No separate inference processes: Single Tinker service handles both
  → ★★★★★ Fused forward_backward_and_optim_step: Optional overlap → better throughput

★★★★★★★ Mask mechanism (trajectory_to_datums):

  SequenceAccumulator.mask.extend(
      [0.0] * delta_token_input_length +   # ← prompt tokens → mask=0 → not trainable
      [1.0] * len(output_token_ids)         # ← action tokens → mask=1 → trainable!
  )
  → ★★★★★★ Only action tokens trainable → prompt+observation tokens → mask=0 → excluded from loss!
```

## 7. ★★★★★ Async Trainer — Decoupled Rollout/Training

```
★★★★★★★ UnifiedTrainer._fit_fully_async() — Two concurrent loops:

  Generation loop (_generation_loop, lines 596-634):
    → await coordinator.wait_for_generation_allowed()
    → for rollout_idx in range(group_size): → asyncio.create_task(_run_rollout)
    → → buffer.add_episode(task_id, episode) → accumulate until group_size complete

  Training loop (_training_loop, lines 636-778):
    → for pass_idx in range(num_fwd_bwd_passes):
    → → task_batch = await buffer.get() → blocks until available
    → → process_backend_batch → forward_backward
    → optim_step → update_policy
    → coordinator.should_sync() → weight sync

★★★★★★★ TrajectoryGroupBuffer (buffer.py):

  async def add_episode(task_id, episode):
      self._pending.setdefault(task_id, []).append(episode)
      if len(self._pending[task_id]) < group_size: return  # Not complete
      episodes = self._pending.pop(task_id)
      traj_groups = transform_episodes_to_trajectory_groups(...)
      adv_metrics = collect_reward_and_advantage_from_trajectory_groups(...)
      await self._queue.put(TaskBatch(groups=traj_groups, episodes=episodes))

★★★★★★★ SyncCoordinator (sync_coordinator.py):

  max_rollout_quota = int((1 + staleness_threshold) * trigger_sync_step * mini_batch_size)
  should_sync() → self._steps_since_sync >= trigger_parameter_sync_step

★★★★★★★ Async behavior spectrum:
  → staleness_threshold=0, trigger_sync_step=1: On-policy (strict)
  → staleness_threshold=0, trigger_sync_step=K: Stream off-policy
  → staleness_threshold>0: Allows stale data → rollout with older weights
  → ★★★★★★ NOTE: rllm/experimental/fully_async/fully_async_trainer.py → Ray-based → DIFFERENT implementation!
```

## 参考
- rllm/trainer/unified_trainer.py: UnifiedTrainer (8-stage pipeline + async loops)
- rllm/trainer/tinker/tinker_backend.py: TinkerBackend adapter
- rllm/trainer/tinker/tinker_policy_trainer.py: TinkerPolicyTrainer (LoRA init + fwd/bwd + optim)
- rllm/engine/rollout/tinker_engine.py: TinkerEngine (SamplingClient inference)
- rllm/trainer/tinker/transform.py: trajectory_to_datums() + mask mechanism
- rllm/trainer/algorithms/rl_algo.py: GRPO advantage computation
- rllm/trainer/algorithms/advantage.py: Advantage orchestration + diagnostics
- rllm/trainer/algorithms/config.py: AlgorithmConfig + RolloutCorrectionConfig
- rllm/trainer/buffer.py: TrajectoryGroupBuffer (async queue)
- rllm/trainer/sync_coordinator.py: SyncCoordinator (staleness + throttle)
- rllm/trainer/config/rllm/backend/tinker.yaml: Tinker defaults (lora_rank=32, bypass=true)
- rllm/data/utils.py: interleave_tasks() (group_size mechanism)
- thinking-machines-lab/tinker/src/tinker/types/lora_config.py: LoraConfig
- thinking-machines-lab/tinker/src/tinker/types/loss_fn_type.py: LossFnType
- thinking-machines-lab/tinker/src/tinker/lib/public_interfaces/training_client.py: forward_backward + optim_step
- thinking-machines-lab/tinker/src/tinker/lib/public_interfaces/service_client.py: create_lora_training_client
- Related notes: rllm-tinker-backend-deep-reading.md, rllm-architecture-reading.md, rllm-v0.3-latest-developments-2026-06-reading.md
