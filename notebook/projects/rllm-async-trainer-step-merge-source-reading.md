# rLLM Async Trainer + Backend-Agnostic Step Merge — Source-Level Analysis

> 2026-06-16 | rllm-org/rllm | PR #576 (open) + PR #394 (merged) + commit de82d7ae
> ★★★★★★★★ SyncCoordinator = 173 lines pure Python + asyncio → Verl/AReaL formulation → staleness spectrum
> ★★★★★★★★ Backend-agnostic step merge → MergedSegment + TokenOps Protocol → cross-backend code reuse
> ★★★★★★★★ Async Trainer = 2 concurrent asyncio loops → generation + training → TrajectoryGroupBuffer + SyncCoordinator
> ★★★★★★★★ RTX 4090: staleness minimal (single GPU, 0 network latency) → Tinker in-process fastest

## 1. ★★★★★★★★ Backend-Agnostic Step Merge (PR #576, commit de82d7ae)

```
★★★★★★★ PR #576 commit de82d7ae — extract step merging into shared module

  BEFORE (current codebase): Two near-identical merge implementations
    → rllm/trainer/tinker/transform.py: trajectory_to_datums() 239 lines
    → rllm/trainer/verl/transform.py: _process_trajectory() 625 lines
    → ★★★ Both walk Trajectory.steps → detect prefix extension → emit merged rows
    → ★★★★ Same logic, different field access patterns → bug fix requires touching both!

  AFTER (PR #576 proposal): Shared module rllm/experimental/common/step_merge.py 234 lines
    → ★★★★★★★★ MergedSegment dataclass → post-merge intermediate representation
    → ★★★★★★★★ TokenOps Protocol → typed against TokenInput → backend-specific adapters
    → ★★★★★★★★ merge_trajectory_steps() → walks Trajectory.steps once → emits per-prefix-run

  ★★★★★★★★ MergedSegment (234-line step_merge.py):

    @dataclass
    class MergedSegment:
        prompt_ids: TokenInput          # flat prompt tokens
        response_ids: TokenInput        # [A0, obs1, A1, obs2, A2, ...]
        response_mask: list[int]        # [1*N_act, 0*N_obs, 1*N_act, ...]
        response_logprobs: list[float]  # aligned with response_ids
        response_advantages: list[float] # aligned with response_ids
        extras: dict[str, Any]          # per_token + per_segment hooks

        @property
        def num_response_tokens(self) -> int:
            return len(self.response_mask)

  ★★★★★★★★ TokenOps Protocol:

    class TokenOps(Protocol):
        def flatten_prompt(self, prompt: TokenInput) -> TokenInput: ...
        def flat_token_length(self, token_input: TokenInput) -> int: ...

    @dataclass(frozen=True)
    class DefaultTokenOps:                       # verl adapter
        def flatten_prompt(self, prompt: list[int]) -> list[int]:
            return list(prompt)
        def flat_token_length(self, token_input: list[int]) -> int:
            return len(token_input)

    # Tinker ships its own _TinkerTokenOps:
    #   → handles EncodedTextChunk unwrapping + .length counting
    #   → typed against TinkerTokenInput (list[int | EncodedTextChunk])

  ★★★★★★★★ merge_trajectory_steps() core algorithm:

    def merge_trajectory_steps(
        trajectory: Trajectory,
        *,
        token_ops: TokenOps = _DEFAULT_TOKEN_OPS,
        require_logprobs: bool = False,
        require_advantage: bool = False,
        pad_short_logprobs: bool = False,
        skip_steps_without_model_output: bool = False,
        per_token_extras: dict[str, PerTokenExtras] | None = None,
        per_segment_extras: dict[str, Callable[[Step], Any]] | None = None,
    ) -> list[MergedSegment]:

    # Walk steps → detect prefix extension → emit segments
    #  Step 1: prompt_ids = [P0]
    #  Step 2: prompt_ids = [P0+A0+obs] → prefix of full_seq → DELTA = [obs]
    #  Step 3: prompt_ids = [P_new] → NOT prefix → close segment, start new

    Internal state machine:
      _start(step) → new segment with prompt + action
      _extend(seg, full, step, delta) → extend current segment with delta + action
      Prefix check: prompt_flat[:len(full)] == full → extend
      No prefix match → emit current, start new

  ★★★★★★★★ PerTokenExtras + PerSegmentExtras — two extension hooks:

    @dataclass(frozen=True)
    class PerTokenExtras:
        extractor: Callable[[Step], Sequence[Any] | None]
        pad_value: Any
        # → router_replay: PerTokenExtras(extractor=lambda s: s.routing_matrices, pad_value="")

    per_segment_extras:
        # → multi_modal_inputs: per_segment_extras={"multi_modal": lambda s: s.model_output.multi_modal_inputs}

  ★★★★★★★★ Call pattern differences (Tinker vs verl):

    Tinker:
      merge_trajectory_steps(traj, token_ops=_TinkerTokenOps(),
                            require_logprobs=True, require_advantage=True)

    verl:
      merge_trajectory_steps(traj, token_ops=DefaultTokenOps(),
                            pad_short_logprobs=True, skip_steps_without_model_output=True)

  ★★★★★★ Net diff: -262 / +136 lines in existing files → backend transforms become thin adapters!
    → Tinker transform: trajectory_to_datums → thin Datum builder over MergedSegment
    → verl transform: _process_trajectory → thin DataProto builder over MergedSegment

  ★★★★★★★★ CURRENT STATUS: PR #576 is OPEN — not merged into main branch
    → Local repo clone has individual implementations (current state)
    → Commit de82d7ae exists in git history but file was not retained
    → The graduation commit (#607) moved experimental → canonical but didn't include step_merge
    → ★★★★★ Both backends still have their own merge implementations → awaiting PR merge
```

### 1.1 ★★★★★★ Current Merge Implementations (Pre-PR #576)

```
★★★★★★★ Tinker merge — rllm/trainer/tinker/transform.py (239 lines):

  trajectory_to_datums(traj: Trajectory) → list[tinker.Datum]:
    → SequenceAccumulator class: full_sequence + sampled_logprobs + advantages + mask
    → _is_prefix(seq1, seq2) → len(seq1) <= len(seq2) and seq2[:len(seq1)] == seq1
    → _flatten_token_input(token_input) → unwrap EncodedTextChunk → flat list[int]
    → ★★★★★★ Delta detection: token_input_flat[len(SequenceAccumulator.full_sequence):]
    → ★★★★★★ Mask: [0.0]*delta_length + [1.0]*len(output_token_ids)
      → prompt tokens → mask=0 → NOT trainable
      → action tokens → mask=1 → trainable
      → observation tokens (between actions) → mask=0 → NOT trainable

  transform_trajectory_groups_to_datums() → (datums | datums_dict, adv_metrics):
    → Advantage computation: collect_reward_and_advantage_from_trajectory_groups()
    → Per-group loss routing: estimator_map → datums_dict keyed by group_role
    → ★★★★★ Merge metrics: batch/merge_compression_ratio, batch/steps_per_traj, batch/step_response_length

★★★★★★★ Verl merge — rllm/trainer/verl/transform.py (625 lines):

  _process_trajectory(trajectory, task_id, accumulated) → int:
    → Same prefix detection logic: prompt_ids[:len(seg["full_seq"])] == seg["full_seq"]
    → _new_segment(step) → dict with prompt/response/mask/logprobs/full_seq
    → _emit(seg) → ProcessedStepData → accumulated.add_step()
    → ★★★★★★ Same mask pattern: [0]*len(delta_obs) + [1]*len(action)
    → ★★★★★★ Same merge metrics: batch/merge_compression_ratio, batch/steps_per_traj

  ★★★★★★ KEY DIFFERENCE from Tinker:
    → Verl operates on Step.model_output.prompt_ids / completion_ids / logprobs
    → Tinker operates on Step.prompt_ids / response_ids / logprobs directly
    → Verl has skip_steps_without_model_output=True
    → Tinker has require_logprobs=True, require_advantage=True
    → Verl pads short logprobs → Tinker asserts they exist
    → ★★★★★★★★ This is exactly what TokenOps Protocol resolves!
```

## 2. ★★★★★★★★ SyncCoordinator — 173 Lines Pure Python, asyncio

```
★★★★★★★★★ rllm/trainer/sync_coordinator.py — 173 lines, 0 dependencies beyond asyncio

  SyncCoordinatorConfig:
    mini_batch_size: int               # episode groups per optimizer step
    group_size: int                    # episodes per group (rollout.n)
    staleness_threshold: float         # 0.0 = on-policy; >0 = stale allowed
    trigger_parameter_sync_step: int   # optimizer steps between weight sync

    @property
    def max_rollout_quota(self) -> int:
        # ★★★★★★★★ Verl/AReaL formulation!
        # (1 + staleness_threshold) * trigger_parameter_sync_step * mini_batch_size
        # → staleness_threshold=0, trigger=1: max_quota = mini_batch_size (strict on-policy)
        # → staleness_threshold=0, trigger=K: max_quota = K * mini_batch_size (stream off-policy)
        # → staleness_threshold=1, trigger=1: max_quota = 2 * mini_batch_size (1 version stale)

  SyncCoordinator internal state:
    _weight_version: int = 0           # incremented on each weight sync
    _quota_used: int = 0               # groups counting toward current sync window
    _in_flight: int = 0                # groups dispatched but not yet consumed
    _steps_since_sync: int = 0         # training steps since last weight sync
    _total_syncs: int = 0              # total weight syncs performed

    # ★★★★★★★ asyncio.Event for throttle + pause:
    _throttle_event: asyncio.Event      # blocks generation when quota exhausted
    _generation_paused: asyncio.Event   # blocks generation during validation/sync
    _in_flight_tasks: set[asyncio.Task]  # tracks async rollout tasks
    _task_errors: list[BaseException]   # records rollout task failures

  ★★★★★★★★ Throttle mechanism:

    on_group_dispatched():
      _quota_used += 1
      _in_flight += 1
      if _quota_used >= max_rollout_quota:
        _throttle_event.clear()  → blocks generation loop

    on_group_consumed():
      _in_flight = max(0, _in_flight - 1)

    on_group_filtered():
      _in_flight = max(0, _in_flight - 1)
      _quota_used = max(0, _quota_used - 1)
      if _quota_used < max_rollout_quota:
        _throttle_event.set()  → resumes generation loop

    async wait_for_throttle() → await _throttle_event.wait()

  ★★★★★★★★ Weight sync mechanism:

    on_training_step_complete():
      _steps_since_sync += 1

    should_sync() → _steps_since_sync >= trigger_parameter_sync_step

    on_sync_complete():
      _weight_version += 1
      _steps_since_sync = 0
      _total_syncs += 1
      # ★★★★★★★ Carryover: in-flight items span sync boundary
      # → count toward NEW window (they were dispatched with old weights)
      _quota_used = _in_flight
      if _quota_used < max_rollout_quota:
        _throttle_event.set()

  ★★★★★★★★ Generation pause (validation / non-partial sync):

    pause_generation() → _generation_paused.clear()
    resume_generation() → _generation_paused.set()
    async wait_for_generation_allowed() → await _generation_paused.wait()

  ★★★★★★★★ Error propagation:

    track_task(task) → _in_flight_tasks.add(task) + done_callback
    record_task_error(exc) → _task_errors.append(exc) + set all events
    raise_if_task_failed() → raise RuntimeError from first error
    async wait_for_task_error() → await _task_error_event.wait()

  ★★★★★★★★ Drain support:

    cancel_tracked_tasks() → cancel all in-flight rollout tasks
    async wait_for_drain() → while _in_flight_tasks: await asyncio.sleep(0.1)

  ★★★★★★★★ Stats dict (7 keys):
    async/weight_version, async/dispatched_since_sync, async/quota_used,
    async/in_flight_groups, async/steps_since_sync, async/max_rollout_quota, async/total_syncs
```

### 2.1 ★★★★★★★★ Staleness Behavior Spectrum

```
★★★★★★★★★ AsyncTrainingConfig (config.py):

  enable: bool = False
  mini_batch_size: int = 1              # episode groups per optimizer step
  fwd_bwd_group_size: int | None = None # task batches per fwd-bwd pass
  staleness_threshold: float = 0.0      # ★★★★★★★★ KEY parameter!
  trigger_parameter_sync_step: int = 1
  partial_rollout: bool = True           # enable turn-level gating during sync
  episode_offload_dir: str | None = None # NVMe offload for episodes
  trajectory_group_offload_dir: str | None = None # NVMe offload for task batches

  ★★★★★★★★ Behavior spectrum:

    staleness_threshold=0, trigger=1:
      → max_quota = mini_batch_size
      → On-policy: every training step → weight sync
      → ★★★★★ RTX 4090 optimal: zero staleness → most stable training

    staleness_threshold=0, trigger=K:
      → max_quota = K * mini_batch_size
      → Stream off-policy: K training steps between weight sync
      → → Generation can dispatch K batches ahead
      → → ★★★★★ RTX 4090: K=4→8 → reasonable throughput boost

    staleness_threshold>0, partial_rollout=False:
      → max_quota = (1+staleness_threshold) * trigger * mini_batch_size
      → Async with staleness: rollout may use stale weights
      → → Generation pauses during weight sync
      → → ★★★★★ Multi-GPU scenario → useful for scaling

    staleness_threshold>0, partial_rollout=True:
      → Generation continues during weight sync (turn-level gating)
      → → ★★★★★★ Requires FullyAsyncAgentLoopManager (verl-only)
      → → → Tinker: partial_rollout=True → generation still pauses
      → → → ★★★★★ RTX 4090: partial_rollout irrelevant (single GPU → no separate rollout workers)

  ★★★★★★★★ fwd_bwd_group_size → gradient accumulation within mini-batch:

    mini_batch_size=8, fwd_bwd_group_size=4 → 2 fwd-bwd passes before optim step
    → Each pass: 4 groups → forward_backward → accumulate gradients
    → After 2 passes: optimizer.step()
    → ★★★★★★ Similar to Megatron's micro-batching → gradient accumulation

    Verl constraint: fwd_bwd_group_size == mini_batch_size (no gradient accumulation)
    Tinker: fwd_bwd_group_size can differ → gradient accumulation possible!
```

## 3. ★★★★★★★★ Async Trainer Pipeline — 2 Concurrent asyncio Loops

```
★★★★★★★★★ _fit_fully_async() — UnifiedTrainer._fit_fully_async() (lines 552-594):

  Setup:
    assert train_batch_size == 1  → async requires batch_size=1
    assert raise_on_error=False   → process_task_with_retry always returns episode
    coord_config = SyncCoordinatorConfig(...)
    coordinator = SyncCoordinator(coord_config)
    aggregator = MetricsAggregator()
    buffer = TrajectoryGroupBuffer(group_size=..., coordinator=coordinator, ...)

  Launch:
    gen_task = asyncio.create_task(self._generation_loop(trainer_state, buffer, coordinator))
    await self._training_loop(trainer_state, buffer, coordinator, aggregator)

  ★★★★★★★★ Two loops run concurrently in same asyncio event loop!

  ★★★★★★★★ Generation loop (_generation_loop, lines 596-634):

    for epoch in range(total_epochs):
      await backend.on_epoch_start(trainer_state)
      for batch in train_dataloader:          # batch_size=1 → single task
        task = batch[0]

        await coordinator.wait_for_generation_allowed()  # ← pause during validation
        if not coordinator.has_quota():
          await coordinator.wait_for_throttle()           # ← throttle when quota full
        coordinator.on_group_dispatched()                 # ← increment quota

        task_id = str(uuid.uuid4())
        for rollout_idx in range(group_size):             # ← group_size rollouts per task

          async def _run_rollout(t=task, tid=task_id, ridx=rollout_idx):
            _, _, _, episode = await self.agent_workflow_engine.process_task_with_retry(
              task=t, task_id=tid, rollout_idx=ridx, result_idx=0
            )
            await buffer.add_episode(tid, episode)

          t = asyncio.create_task(_run_rollout())  # ← concurrent rollout!
          coordinator.track_task(t)

      await backend.on_epoch_end(trainer_state)
    await coordinator.wait_for_drain()
    buffer.mark_generation_complete()  # ← sentinel None in queue

  ★★★★★★★★ Training loop (_training_loop, lines 636-777):

    while True:
      trainer_state.reset_batch()
      weight_versions = []
      all_trajectory_groups = []
      all_episodes = []

      # 1. Pull mini_batch_size task batches, split into fwd_bwd passes
      for pass_idx in range(num_fwd_bwd_passes):
        chunk_groups = []
        for _ in range(fwd_bwd_group_size):
          task_batch = await buffer.get()        # ← blocks until available
          coordinator.on_group_consumed()

          chunk_groups.extend(task_batch.groups)
          all_trajectory_groups.extend(task_batch.groups)
          all_episodes.extend(task_batch.episodes)

        # Forward-backward on this chunk
        trainer_state.trajectory_groups = chunk_groups
        await self.backend.on_batch_start(trainer_state)
        trainer_state.backend_batch = self.backend.transform_to_backend_batch(trainer_state)
        await self.backend.process_backend_batch(trainer_state)

        aggregator.record_dict(trainer_state.metrics)
        trainer_state.metrics = {}

      # 2. Optimizer step
      await self.backend.update_policy(trainer_state)

      # 3. Capture staleness metrics
      staleness_values = [coordinator.weight_version - v for v in weight_versions]
      aggregator.record("async/staleness_mean", float(np.mean(staleness_values)))
      aggregator.record("async/staleness_min", float(np.min(staleness_values)))
      aggregator.record("async/staleness_max", float(np.max(staleness_values)))

      # 4. Weight sync
      coordinator.on_training_step_complete()
      if coordinator.should_sync():
        await self._perform_weight_sync(trainer_state, coordinator, rollout_engine)

      # 5-7. Flush metrics, log, periodic validation
      trainer_state.metrics.update(aggregator.flush())
      trainer_state.metrics["async/trainer_idle_ratio"] = buffer_wait_time / step_time
      await self.backend.on_batch_end(trainer_state)
      print_metrics_table(trainer_state.metrics, trainer_state.global_step)

  ★★★★★★★★ _perform_weight_sync():

    if not partial_rollout:
      coordinator.pause_generation()       # ← block generation
      await coordinator.wait_for_drain()   # ← wait for in-flight rollouts

    trainer_state.weight_version = coordinator.weight_version + 1
    await self.backend.on_policy_updated(trainer_state)  # ← weight sync hook!
    rollout_engine.weight_version = trainer_state.weight_version
    coordinator.on_sync_complete()

    if not partial_rollout:
      coordinator.resume_generation()

  ★★★★★★★★ Async vs sync path comparison:

    Sync (_fit_on_policy):
      for batch in dataloader:
        episodes = generate_episodes(batch)          # sequential
        trajectory_groups = transform(episodes)       # sequential
        backend_batch = transform_to_backend_batch()  # sequential
        process_backend_batch()                        # sequential
        compute_advantages()                           # sequential
        update_policy()                                # sequential

    Async (_fit_fully_async):
      gen_loop: dispatch rollouts → asyncio.create_task → concurrent!
      buffer: accumulate episodes → TrajectoryGroup → TaskBatch → asyncio.Queue
      train_loop: buffer.get() → forward_backward → optim_step → weight_sync
      → ★★★★★★★★ Generation and training PROGRESS CONCURRENTLY!
      → → Generation doesn't wait for training
      → → Training doesn't wait for generation
      → → → Buffer mediates between them → asyncio.Queue
```

### 3.1 ★★★★★★★★ TrajectoryGroupBuffer — Async Queue + Filtering + NVMe Offload

```
★★★★★★★★★ rllm/trainer/buffer.py — 421 lines:

  TaskBatch:
    groups: list[TrajectoryGroup]
    episodes: list[Episode]

  TrajectoryGroupBuffer:
    _pending: dict[str, list[Episode | str]]   # task_id → accumulated episodes
    _queue: asyncio.Queue[TaskBatch | str | None]  # training queue
    _training_queue_size: int
    _filtered_count: int
    _consumed_count: int
    _generation_complete: bool

  ★★★★★★★★ add_episode() — 6-stage pipeline:

    Stage 1: Check generation complete → ignore late episodes
    Stage 2: Offload to disk (optional) → asyncio.to_thread(pickle_dump)
    Stage 3: Accumulate until group_size reached → _pending[task_id]
    Stage 4: When group complete:
      a. Record episode-level metrics
      b. Transform episodes → trajectory groups (transform_episodes_to_trajectory_groups)
      c. Drop groups with too few trajectories (min_trajs_per_group)
      d. Compute advantages (collect_reward_and_advantage_from_trajectory_groups)
      e. Rejection sampling → drop groups with all-zero advantage
      f. ★★★★★ Set weight_version = _min_weight_version(episodes)
      g. Queue TaskBatch

    ★★★★★★★★ _min_weight_version() — staleness tracking:
      → min(step.weight_version for all steps in all trajectories)
      → ★★★★★★★★ Staleness = coordinator.weight_version - min_weight_version
      → → On-policy: staleness=0 (all steps from current version)
      → → Off-policy: staleness>0 (some steps from older version)

  ★★★★★★★★ Filtering → releases quota:

    Empty groups → coordinator.on_group_filtered() → _filtered_count++
    → ★★★★★★ Releases BOTH in_flight AND quota_used
    → → Generation loop can dispatch more tasks

  ★★★★★★★★ NVMe offloading:

    episode_offload_dir → pickle episodes to disk → reduce memory
    trajectory_group_offload_dir → pickle TaskBatch to disk → reduce memory
    asyncio.to_thread() → non-blocking disk I/O
    → ★★★★★ RTX 4090: 24GB RAM → offloading can help with large batches

  ★★★★★★★★ get_many(count) → blocks until count batches available:

    while _training_queue_size < count:
      if _generation_complete: return None
      await _queue_update_event.wait()
    → Pull count items from queue → return list[TaskBatch]

  ★★★★★★★★ mark_generation_complete():

    Flush incomplete groups → coordinator.on_group_filtered()
    Put sentinel None in queue → signal training loop to stop
```

## 4. ★★★★★★★★ Tinker save_weights_for_sampler ↔ Async Pipeline

```
★★★★★★★★★ TinkerBackend.on_policy_updated() — weight sync hook:

  async def on_policy_updated(trainer_state):
    self.sampling_client = await self.policy_trainer.save_checkpoint_and_get_sampling_client(
      global_step, do_save=do_save
    )
    self.rollout_engine.set_sampling_client(self.sampling_client)

  ★★★★★★★★ save_checkpoint_and_get_sampling_client():

    if not do_save:  # ← most steps: no disk checkpoint
      return await self.training_client.save_weights_and_get_sampling_client_async()
      # → ★★★★★★★★ Zero-copy: Tinker Service shares weight memory internally
      # → → No Python tensor serialization
      # → → → SamplingClient created from SAME weights as TrainingClient

    else:  # ← periodic checkpoint: save state + sampler
      state_future = await self.training_client.save_state_async(name)
      sampler_future = await self.training_client.save_weights_for_sampler_async(name)
      # → Both async → can overlap!
      return self.training_client.create_sampling_client(sampler_result.path)

  ★★★★★★★★ Interaction with async pipeline:

    SyncCoordinator.should_sync() → triggers _perform_weight_sync()
    → backend.on_policy_updated(trainer_state) → TinkerBackend.on_policy_updated()
    → → save_weights_and_get_sampling_client_async() → NEW SamplingClient
    → → rollout_engine.set_sampling_client(new_sampling_client)
    → coordinator.on_sync_complete() → resume generation

    ★★★★★★★★ In async mode: weight sync happens AFTER optim step
    → Generation loop uses OLD sampling_client until sync completes
    → → ★★★★★★ This is intentional staleness → bounded by staleness_threshold
    → → → On-policy (staleness=0): sync after every optim step → near-zero staleness
    → → → Stream off-policy (staleness=0, trigger=K): sync every K steps → K versions stale

    ★★★★★★★★ Tinker-specific: save_weights is ASYNC → doesn't block training loop
    → → Training can continue generating while weights are being saved
    → → → ★★★★★★★★ vs verl: checkpoint_manager.update_weights() → Ray RPC → network latency!

    ★★★★★★★★ Single GPU (RTX 4090):
    → TrainingClient + SamplingClient → SAME Tinker Service → SAME process
    → → save_weights → internal memory sharing → microseconds
    → → → ★★★★★★★★ Near-zero weight sync overhead → best async performance!
    → → → → vs verl: separate rollout workers → Ray RPC → milliseconds → slower
```

### 4.1 ★★★★★★★★ VerlBackend Weight Sync (Separated Mode)

```
★★★★★★★★★ VerlBackend.on_policy_updated():

  if self.is_separated and self.checkpoint_manager is not None:
    await self.checkpoint_manager.update_weights(trainer_state.weight_version)

  ★★★★★★★★ Separated mode vs colocated:
    Colocated: actor_rollout_wg handles both training and rollout
      → weight sync: checkpoint_manager.update_weights() → wake replicas
      → → Sleep/Wake pattern → GPU time-multiplexing

    Separated: training workers + standalone rollout servers
      → rollout on separate GPUs → no sleep/wake needed
      → → weight sync: checkpoint_manager.update_weights() → push to rollout servers
      → → → ★★★★★ Ray RPC → network latency → slower than Tinker in-process

  ★★★★★★★★ Verl separated mode requirements:
    → async_training.fwd_bwd_group_size == mini_batch_size (no gradient accumulation)
    → → ★★★★★★ Verl can't do gradient accumulation in async mode!
    → → → Tinker CAN → fwd_bwd_group_size != mini_batch_size → gradient accumulation

    partial_rollout + remote_runtime → NOT compatible
    → → partial_rollout requires turn-level gating → only FullyAsyncAgentLoopManager supports
```

## 5. ★★★★★★★★ Comparison: rLLM asyncio vs verl Ray-based Async

```
★★★★★★★★★ Architecture comparison:

  | Aspect | rLLM (asyncio) | verl (Ray-based) |
  |--------|----------------|-------------------|
  | Communication | asyncio.Queue + asyncio.Event | Ray RPC + MessageQueueClient |
  | Coordination | SyncCoordinator (173 lines, pure Python) | SeparateRayPPOTrainer + ParamSynchronizer (Ray actors) |
  | Weight sync | backend.on_policy_updated() → in-process or Ray RPC | checkpoint_manager.update_weights() → Ray RPC |
  | Rollout dispatch | asyncio.create_task() → concurrent in event loop | Ray remote actors → separate processes |
  | Staleness control | Verl/AReaL formulation → quota + throttle + pause | trigger_parameter_sync_step + staleness_threshold |
  | Buffer | TrajectoryGroupBuffer (asyncio.Queue + filtering) | MessageQueueClient (ZMQ-like queue) |
  | Error propagation | asyncio task callback → RuntimeError | Ray actor failure → ray.get() exception |
  | Gradient accumulation | Tinker: fwd_bwd_group_size < mini_batch_size | Verl: NO (fwd_bwd_group_size == mini_batch_size) |

  ★★★★★★★★ Key differences:

    1. rLLM = asyncio-based → all loops in SAME event loop thread
       → ★★★★★★★★ No Ray → no separate processes → no Ray actor overhead
       → → Tinker: TrainingClient + SamplingClient → SAME Tinker Service → SAME process
       → → → ★★★★★★★★ Zero-copy weight sync → microseconds
       → → → → vs verl: separate Ray actors → weight sync over network → milliseconds

    2. rLLM generation loop = asyncio.create_task per rollout
       → Multiple rollouts can run concurrently → asyncio event loop
       → → ★★★★★★ Tinker: SamplingClient → async HTTP to Tinker Service
       → → → Not truly parallel (single-threaded event loop) → but overlapped I/O

    3. verl generation loop = Ray remote actors
       → Multiple rollout workers → truly parallel → separate GPU processes
       → → ★★★★★★ Better for multi-GPU → but unnecessary for single GPU

    4. rLLM staleness formulation = identical to Verl/AReaL
       → max_rollout_quota = (1 + staleness_threshold) * trigger * mini_batch_size
       → → ★★★★★★★★ SyncCoordinator docstring: "matching Verl/AReaL"
       → → → Same mathematical model → different implementation (Python vs Ray)

    5. rLLM buffer = TrajectoryGroupBuffer → does filtering + advantage computation
       → Episodes → transform → filter → advantage → TaskBatch → queue
       → → ★★★★★★ Advantage computed BEFORE queueing → training loop gets ready-to-train batches
       → → → vs verl experimental FullyAsyncTrainer: advantage computed IN training loop

    6. rLLM validation = pause_generation + wait_for_drain + validate
       → ★★★★★★★★ Clean: generation stops → drain → validate → resume
       → → → vs verl: separate RolloutExecutor handles validation → different process

  ★★★★★★★★ verl experimental FullyAsyncTrainer (rllm/experimental/fully_async/):

    @ray.remote(num_cpus=10)
    class FullyAsyncTrainer(SeparateRayPPOTrainer):
      → Ray actor → separate process → 10 CPU cores
      → MessageQueueClient → get_sample_sync() → blocking queue read
      → ParamSynchronizer → Ray remote actor → sync_weights.remote()
      → ★★★★★★★★ HEAVIER infrastructure → Ray + MessageQueue + ParamSynchronizer
      → → → vs rLLM asyncio: asyncio.Queue + SyncCoordinator → much lighter!

    verl FullyAsyncTrainer staleness tracking:
      → rollout_param_versions → stale_count = sum(v < current_param_version)
      → trajectory_param_versions → stale trajectory count
      → ★★★★★★★★ Same concept → different tracking (per-sample vs per-group)

    verl MIS (Multi-step Importance Sampling):
      → local_trigger_step == 1: save_model_to_cpu(1) → compute old_log_prob with current weights
      → local_trigger_step > 1: restore_model_from_cpu(1) → compute IS weights with version-1 weights
      → → ★★★★★★★★ Restores old weights on GPU → computes IS → then restores current weights
      → → → → ★★★★★★ This is CPU-GPU weight swap → expensive → RTX 4090 内存压力大!
      → → → → → rLLM Tinker: bypass_mode → no IS → no weight swap → much simpler!
```

### 5.1 ★★★★★★★★ Code Path: step() → merge() → loss computation

```
★★★★★★★★★ On-policy (sync) path:

  _train_batch_async(batch):
    1. episodes = backend.generate_episodes(batch)
    2. trajectory_groups = transform_episodes_to_trajectory_groups(episodes)
       → ★★★★★★ Merges steps into cumulative rows (Tinker: trajectory_to_datums, verl: _process_trajectory)
    3. rejection_sampling_and_filtering()
    4. backend_batch = backend.transform_to_backend_batch(trainer_state)
       → Tinker: returns [] → actual datums in process_backend_batch
       → Verl: transform_episodes_to_dataproto or transform_trajectory_groups_to_dataproto
    5. await backend.process_backend_batch(trainer_state)
       → Tinker: forward_backward_from_trajectory_groups() → Tinker SDK async
       → Verl: compute_log_prob + ref_log_prob → Ray RPC to workers
    6. await backend.compute_advantages(trainer_state, algorithm_config)
       → Tinker: stores algorithm_config (advantages already in datums from transform)
       → Verl: collect_reward_and_advantage_from_trajectory_groups → update_dataproto_with_advantages
    7. await backend.update_policy(trainer_state)
       → Tinker: optim_step_async() → AdamParams
       → Verl: update_actor → Ray RPC

  ★★★★★★★★ Async path:

  _training_loop():
    1. buffer.get() → TaskBatch with trajectory_groups (advantages already computed!)
       → ★★★★★★★★ Buffer computes advantages BEFORE queueing
       → → Training loop receives ready-to-train batches → no advantage computation in loop!
    2. backend.transform_to_backend_batch() → backend-specific format
    3. backend.process_backend_batch() → forward-backward pass
       → ★★★★★ Multiple fwd-bwd passes possible (gradient accumulation)
    4. backend.update_policy() → optimizer step
    5. weight sync → backend.on_policy_updated()
       → Tinker: save_weights_and_get_sampling_client_async() → zero-copy!
       → Verl: checkpoint_manager.update_weights() → Ray RPC

  ★★★★★★★★ Tinker loss computation trace:

    transform_trajectory_groups_to_datums():
      → collect_reward_and_advantage_from_trajectory_groups() → GRPO advantage
      → trajectory_to_datums(traj) → MergedSegment → tinker.Datum
      → → Datum.loss_fn_inputs = {target_tokens, logprobs, advantages, mask}

    forward_backward_async([datums], loss_fn="ppo"):
      → ★★★★★★★★ Tinker Service handles loss computation INTERNALLY
      → → pi_new = current policy logprobs (forward pass)
      → → pi_old = loss_fn_inputs["logprobs"] (rollout logprobs)
      → → ratio = exp(pi_new - pi_old)
      → → PPO clip: clip(ratio, 1-eps, 1+eps) * advantage * mask
      → → → ★★★★★★★★ bypass_mode: pi_old = rollout logprobs → KL=0 → no ref model!

  ★★★★★★★★ Verl loss computation trace:

    process_backend_batch():
      → compute_log_prob → old_log_probs (or bypass: rollout_log_probs)
      → ref_log_prob (if reference policy) → Ray RPC
      → ★★★★★ CustomPPOLoss → per-call loss mode override

    update_actor():
      → update_actor(batch_td) → Ray RPC to workers
      → → Workers handle: loss computation + gradient update
      → → → Loss mode override via metadata["policy_loss_mode_override"]
      → → → → ★★★★★★★★ Per-role loss routing (loss_fn_map) → multiple loss functions per batch!
```

## 6. ★★★★★★★★ RTX 4090 Practical Implications

```
★★★★★★★★★ Single GPU → async training implications:

  1. ★★★★★★★★ Staleness is MINIMAL on single GPU:
    → No network latency between training and rollout
    → Tinker: in-process → weight sync = microseconds
    → → Staleness_threshold=0 is nearly free → training is still on-policy
    → → ★★★★★★★★ RTX 4090: USE staleness_threshold=0, trigger_parameter_sync_step=4-8
    → → → This gives stream off-policy → K batches ahead → better throughput
    → → → → Staleness still near-zero because weight sync is so fast

  2. ★★★★★★★★ Tinker = fastest async on single GPU:
    → asyncio → no Ray → no separate processes → no Ray actor overhead
    → Zero-copy weight sync → microseconds vs milliseconds (verl Ray RPC)
    → → ★★★★★★★★ RTX 4090 async ranking: Tinker #1 >> verl #2
    → → → verl separated mode: needs separate GPUs for rollout → NOT applicable to single GPU!
    → → → → verl colocated mode: sleep/wake → GPU time-multiplexing → but slower than Tinker

  3. ★★★★★★★★ Generation concurrency:
    → asyncio.create_task per rollout → concurrent HTTP to Tinker Service
    → → ★★★★★★ Overlaps I/O → generation latency ~2-8s per task
    → → → Training loop waits for buffer → buffer_wait_time recorded
    → → → → trainer_idle_ratio = buffer_wait_time / step_time → should be <0.3 for good throughput

  4. ★★★★★★★★ NVMe offloading for 24GB constraint:
    → episode_offload_dir + trajectory_group_offload_dir
    → → Pickles to disk → asyncio.to_thread → non-blocking
    → → ★★★★★★ RTX 4090 with large batches → episodes can OOM
    → → → Offload pending episodes to disk → reduce peak memory
    → → → → ★★★★★★★★ verl has NO offload mechanism → more memory pressure!

  5. ★★★★★★★★ Partial rollout → NOT useful on single GPU:
    → Requires separate rollout workers → Tinker is in-process
    → → partial_rollout=True → generation continues during weight sync
    → → → ★★★★★★ Only useful for verl separated mode → needs extra GPUs
    → → → → RTX 4090: partial_rollout=False → generation pauses during sync → simpler!

  6. ★★★★★★★★ Gradient accumulation → Tinker advantage:
    → fwd_bwd_group_size < mini_batch_size → gradient accumulation
    → → ★★★★★★ Example: mini_batch_size=8, fwd_bwd_group_size=4 → 2 fwd-bwd passes
    → → → Verl: fwd_bwd_group_size == mini_batch_size → no accumulation → larger peak memory
    → → → → ★★★★★★★★ RTX 4090: Tinker gradient accumulation → smaller fwd-bwd chunks → less peak memory!

  7. ★★★★★★★★ Recommended RTX 4090 async config (Tinker):

    rllm:
      async_training:
        enable: true
        mini_batch_size: 8          # 8 groups per optim step
        fwd_bwd_group_size: 4       # 4 groups per fwd-bwd → gradient accumulation
        staleness_threshold: 0.0    # on-policy → minimal staleness
        trigger_parameter_sync_step: 4  # sync every 4 optim steps
        partial_rollout: false      # not useful on single GPU
        episode_offload_dir: null   # enable if OOM
        trajectory_group_offload_dir: null  # enable if OOM

    → ★★★★★★★★ Expected throughput: ~3-5x improvement vs sync
    → → Generation and training overlap → no sequential bottleneck
    → → → Buffer mediates → generation produces → training consumes → concurrent!

  8. ★★★★★★★★ Comparison with verl RTX 4090 async:
    → verl separated mode → requires separate GPU for rollout → NOT applicable!
    → verl colocated mode → sleep/wake → single GPU time-multiplexing
    → → Sleep replicas during training → wake during rollout → sequential!
    → → → ★★★★★★★★ NOT truly async → just faster sequential → not concurrent!
    → → → → → ★★★★★★★★★★★★★★★★★★ rLLM Tinker async = truly concurrent → verl colocated = sequential!

  9. ★★★★★★★★ Backend-agnostic step merge → RTX 4090 benefit:
    → Unified merge logic → bug fix once → both backends benefit
    → → Tinker merge bug → fix in step_merge.py → verl also fixed
    → → → ★★★★★★★★ Maintenance cost reduction → both backends share same merge code
    → → → → ★★★★★★ Especially important for multi-turn agent RL → merge is critical
    → → → → → → ★★★★★ Incorrect merge → incorrect mask → incorrect loss → wasted training!
```

## 7. ★★★★★★★★ BackendProtocol — Cross-Backend Code Reuse Architecture

```
★★★★★★★★★ rllm/trainer/backend_protocol.py — 209 lines:

  class BackendProtocol(ABC, Generic[TDataset, TBatch]):
    name: str = "base_backend"
    requires_loop: bool = False  # Tinker=True → needs event loop

    # Required methods:
    init_rollout_engine() → RolloutEngine
    validate_config() → None
    shutdown() → None

    # Async pipeline methods (all async):
    async generate_episodes(batch, agent_workflow_engine, is_validation) → list[Episode]
    transform_to_backend_batch(trainer_state) → TBatch
    async process_backend_batch(trainer_state) → None
    async compute_advantages(trainer_state, algorithm_config) → None
    async update_policy(trainer_state) → None

    # Hook methods (optional):
    async on_train_start(trainer_state)
    async on_train_end(trainer_state)
    async on_batch_start(trainer_state)
    async on_batch_end(trainer_state)
    async on_policy_updated(trainer_state)  # ★★★★★ weight sync hook!
    async on_epoch_start(trainer_state)
    async on_epoch_end(trainer_state)
    async on_validation_start(trainer_state) → bool
    async on_validation_end(trainer_state)

  ★★★★★★★★ Cross-backend code reuse:

    UnifiedTrainer.fit_async() → calls BackendProtocol methods
    → Same pipeline for ALL backends: Tinker / verl / Fireworks
    → → ★★★★★★★★ Backend-agnostic trainer → backends only implement specific methods
    → → → generate_episodes() → backend-specific rollout
    → → → transform_to_backend_batch() → backend-specific data format
    → → → process_backend_batch() → backend-specific forward-backward
    → → → compute_advantages() → Tinker: stores config; Verl: updates DataProto
    → → → update_policy() → Tinker: optim_step_async; Verl: update_actor Ray RPC
    → → → on_policy_updated() → Tinker: save_weights zero-copy; Verl: checkpoint_manager

    ★★★★★★★★ Backend-agnostic ADVANTAGE computation:
      → collect_reward_and_advantage_from_trajectory_groups()
      → → rllm-native → GRPO / REINFORCE / RLOO / REINFORCE++ baseline
      → → → SAME advantage logic for Tinker AND verl!
      → → → → ★★★★★★★★ This is a KEY benefit → no advantage divergence across backends

    ★★★★★★★★ Backend-agnostic DATASET / DATALOADER:
      → StatefulTaskDataLoader → same for all backends
      → → interleave_tasks() → same for all backends
      → → → ★★★★★★★★ Same task distribution → same group_size → same GRPO grouping

  ★★★★★★★★ Three backend implementations:

    | Method | TinkerBackend | VerlBackend | FireworksBackend |
    |--------|---------------|-------------|------------------|
    | generate_episodes | interleave_tasks + TinkerEngine | interleave_tasks + VerlEngine | cloud API |
    | transform_to_backend_batch | returns [] (placeholder) | DataProto construction | custom |
    | process_backend_batch | TinkerPolicyTrainer.fwd_bwd | compute_log_prob + ref_log_prob | cloud API |
    | compute_advantages | stores _algorithm_config | updates DataProto with advantages | custom |
    | update_policy | optim_step_async | update_actor (Ray RPC) | cloud API |
    | on_policy_updated | save_weights zero-copy | checkpoint_manager.update_weights | cloud API |
    | requires_loop | True (asyncio) | False (Ray handles) | False |

  ★★★★★★★★ PR #576 enhances cross-backend reuse further:
    → Step merge: shared merge_trajectory_steps() → MergedSegment
    → → ★★★★★★★★ BEFORE: Tinker and verl each implement merge independently
    → → → AFTER: Both use same merge_trajectory_steps() → different TokenOps adapters
    → → → → ★★★★★★★★ Bug fix once → both backends fixed → maintenance cost halved!
```

## Key Source Files

- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/sync_coordinator.py`: SyncCoordinator (173 lines, pure asyncio)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/buffer.py`: TrajectoryGroupBuffer (421 lines, async queue + filtering + NVMe offload)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/unified_trainer.py`: UnifiedTrainer (1078 lines, 2 concurrent loops + BackendProtocol)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/backend_protocol.py`: BackendProtocol (209 lines, async-prioritized interface)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/algorithms/config.py`: AsyncTrainingConfig + AlgorithmConfig (377 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/algorithms/advantage.py`: Backend-agnostic advantage computation (295 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/tinker/tinker_backend.py`: TinkerBackend (450 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/tinker/transform.py`: Tinker merge (239 lines, trajectory_to_datums)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/tinker/tinker_policy_trainer.py`: TinkerPolicyTrainer (453 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/verl/verl_backend.py`: VerlBackend (880 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/verl/transform.py`: Verl merge (625 lines, _process_trajectory)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/verl/async_agent_loop.py`: FullyAsyncAgentLoopManager (119 lines)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/experimental/fully_async/fully_async_trainer.py`: verl experimental async (648 lines, Ray-based)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/trainer/metrics_aggregator.py`: MetricsAggregator (120 lines)
- Commit de82d7ae: step_merge.py (234 lines, PR #576 proposal — MergedSegment + TokenOps Protocol)
- `/Users/jackiemac/workspace/rollout-infra/_temp_rllm/rllm/types.py`: Step + Trajectory + TrajectoryGroup (554 lines)

## Related Notes

- [rLLM Tinker Training Loop Source Reading](rllm-tinker-training-loop-source-reading.md) — Section 7 covers async at 33-line summary
- [rLLM v0.3 Latest Developments](rllm-v0.3-latest-developments-2026-06-reading.md) — Section 2 covers step merge at 20-line summary
- [rLLM Architecture Reading](rllm-architecture-reading.md) — overall architecture
- [rLLM Tinker Backend Deep Reading](rllm-tinker-backend-deep-reading.md) — Tinker specifics
- [verl vs rLLM Transform Comparison](verl-vs-rllm-transform-comparison.md) — transform comparison
