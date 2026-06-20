# verl V1 Architecture: Critical Bugs & Issues Report (June 8-20, 2026)

**Date**: 2026-06-20
**Source**: verl-project/verl GitHub (issues, PRs, source code, review comments)
**Scope**: GRPO training, weight sync, LoRA, memory, async training, RTX 4090 impact

---

## 1. Executive Summary

verl is undergoing a major architectural transition from the legacy `ray_trainer.py`-based PPO loop to a **V1 unified trainer architecture** (`verl/trainer/ppo/v1/`) that uses a hook-based design pattern with three modes: `PPOTrainerSync`, `PPOTrainerColocateAsync`, `PPOTrainerSeparateAsync`. The legacy trainer is now in maintenance mode and will be deprecated.

**Key themes in June 2026 bugs**:
- Weight sync ordering bugs causing OOM (memory peak overlap between actor all-gather and rollout allocation)
- LoRA-specific failures: IPC buffer aliasing, EOS suppression at high rank/alpha
- Fully async training producing incorrect idle metrics and diverging training behavior even at staleness=0
- Delta weight sync (new PR) with 4 critical review-identified sub-issues

---

## 2. V1 Architecture Overview

### 2.1 Trainer Structure

The V1 trainer lives at `verl/trainer/ppo/v1/`:

| File | Role |
|------|------|
| `trainer_base.py` | `PPOTrainer` base class with hook pattern; uses `transfer_queue` (tq) for async data flow; manages `AgentLoopManager`, `RewardLoopManager`, `MultiTeacherModelManager` |
| `trainer_sync.py` | `PPOTrainerSync` -- colocated, synchronous. Hooks: `on_init_end` -> `update_weights`, `on_step_end` -> `update_weights`, `on_sample_end` -> `sleep_replicas` |
| `trainer_colocate_async.py` | `PPOTrainerColocateAsync` -- colocated, async, partial rollout |
| `trainer_separate_async.py` | `PPOTrainerSeparateAsync` -- disaggregated trainer/rollout, async, partial rollout |
| `agent_loop_tq.py` | `AgentLoopManagerTQ`, `AgentLoopWorkerTQ` -- transfer queue integration for agent loop |

**PR #6710** (`[trainer] feat: add unify trainer abstraction`) was merged on June 12, introducing this architecture. The maintainer (wuxibin89) explicitly stated on issue #6780: *"We encourage users to migrate to the V1 trainer; the legacy trainer is in maintenance mode and will be deprecated in a future release."*

### 2.2 Weight Sync Flow

The `update_weights` method in `verl/workers/engine_workers.py` (lines 670-749) has two paths:

**Naive (colocated/sync) path**:
1. Resume rollout weights (vLLM/SGLang allocates weight memory)
2. `get_per_tensor_param()` -- FSDP2 state_dict all-gather allocates temporary full parameters
3. LoRA base sync if needed
4. Offload actor to CPU if param_offload enabled
5. `aggressive_empty_cache()`
6. Resume kv_cache

**Checkpoint engine (async/disaggregated) path**:
1. `get_per_tensor_param()`
2. `checkpoint_engine.send_weights()` via NCCL/ZMQ/disk

### 2.3 Rollout Engines

| Engine | Key Files | Status |
|--------|-----------|--------|
| vLLM | `verl/workers/rollout/vllm_rollout/vllm_rollout.py`, `utils.py`, `weight_update_utils.py`, `bucketed_weight_transfer.py`, `vllm_async_server.py` | Primary; V1 mode (`VLLM_USE_V1=1`); IPC weight transfer via ZMQ + shared memory |
| SGLang | `verl/workers/rollout/sglang_rollout/sglang_rollout.py`, `utils.py`, `http_server_engine.py`, `async_sglang_server.py` | HTTP-based weight update; tensor bucketing via `get_named_tensor_buckets`; LoRA via `LoadLoRAAdapterFromTensorsReqInput` |
| TRTLLM | `verl/workers/rollout/trtllm_rollout/` | Secondary; async server via `trtllm_async_server.py` |

---

## 3. Critical Issues Deep Read

### 3.1 PR #6794: Delta Weight Sync (4 Sub-Issues)

**Title**: `[rollout][sglang] feat: delta weight sync (sparse trainer->rollout updates)`
**Status**: DRAFT (open), not merged
**Severity**: HIGH (architecture-level change; review revealed 4 critical/high sub-issues)
**Author**: ChangyiYang

**Overview**: Implements sparse/delta weight sync for SGLang rollout, transmitting only changed weight bytes instead of full broadcasts. Based on THUDM/slime and pairs with sgl-project/sglang#26519. Claims ~100x wire payload reduction. New files: `verl/workers/rollout/delta_sync/` (delta_state.py, encode.py, wrapper.py) + `verl/workers/rollout/sglang_rollout/delta_dispatch.py`.

**4 Sub-Issues from gemini-code-assist review**:

#### 3.1.1 record_stream (CRITICAL)
**File**: `verl/workers/rollout/delta_sync/delta_state.py`
**Problem**: When copying tensors asynchronously on `self.d2h_stream`, `tensor.record_stream(self.d2h_stream)` is missing. PyTorch's caching allocator can reclaim/reuse tensor memory before the async copy completes. Since the generator may yield new tensors or FSDP may free/reuse gathered parameter buffers in subsequent steps, omitting `record_stream` leads to **silent data corruption**.
**Root Cause**: Standard CUDA stream synchronization bug -- async D2H copies need `record_stream` to prevent premature memory recycling.
**RTX 4090 Impact**: Directly affects RTX 4090 where CUDA memory pool is smaller (24GB) and allocator pressure is higher, increasing probability of premature recycling.
**Cross-Framework**: Same bug class as vllm-project/vllm#31848 (the native sparse weight-update API that vLLM is waiting on), and is identical to the pattern that caused the IPC buffer aliasing in PR #6688.

#### 3.1.2 disk race (CRITICAL)
**File**: `verl/workers/rollout/sglang_rollout/delta_dispatch.py`
**Problem**: Using `transport='disk'` for delta sync on TP > 1 has two bugs:
1. **Race condition**: The leader rank calls `dispatch_disk_files` and deletes `version_dir` before other ranks finish writing or SGLang finishes reading. Missing `torch.distributed.barrier()`.
2. **Incomplete file list**: `pending_files` is local per-rank, so the leader only dispatches its own files, missing other ranks' flushes.
**Root Cause**: Distributed filesystem operations need explicit synchronization; the current code assumes single-rank semantics.
**RTX 4090 Impact**: Multi-GPU (TP=2 on dual 4090) delta sync via disk would silently lose weight updates, causing training divergence without any error signal.
**Cross-Framework**: vLLM had similar issues in its early distributed checkpoint code (vllm-project/vllm#3082 race on checkpoint directory cleanup). SGLang's own distributed weight loading also uses barriers.

#### 3.1.3 big_values (HIGH)
**File**: `verl/workers/rollout/delta_sync/encode.py`
**Problem**: `_sparse_boundaries` concatenates all parameter values via `big_values = torch.cat(...)`, allocating a massive temporary tensor containing ALL parameter elements (including unchanged ones) for every chunk. This introduces significant GPU memory overhead.
**Root Cause**: The bytewise diff algorithm builds an index into the concatenated `big_values` tensor, but the intermediate allocation defeats the purpose of sparse sync by temporarily requiring full model-size GPU memory.
**RTX 4090 Impact**: On 24GB RTX 4090, allocating a `big_values` tensor that holds all BF16 values for even a 7B model (14GB params) would immediately OOM or leave no room for activations.
**Fix suggestion**: Index into each `d.values` individually using computed `local_idx` on GPU, avoiding the `big_values` allocation entirely.
**Cross-Framework**: DeepSpeed's ZeRO-3 parameter partitioning avoids this by never gathering full parameters. The PULSE paper (arXiv:2602.03839) specifically notes that sparse updates should not require full-model intermediate allocations.

#### 3.1.4 makedirs (HIGH)
**File**: `verl/workers/rollout/sglang_rollout/delta_dispatch.py`
**Problem**: Non-leader ranks might attempt to write flush files to `out_dir` before the leader has created it, leading to `FileNotFoundError`. Missing `os.makedirs(out_dir, exist_ok=True)` inside `write_flush_to_disk`.
**Root Cause**: Missing defensive directory creation before file write in distributed environment.
**RTX 4090 Impact**: Would crash on multi-GPU disk-based delta sync, but is a straightforward fix.

**Overall Assessment**: PR #6794 is architecturally significant but has 4 blocking issues that need resolution before merge. The `record_stream` and `disk race` issues are silent corruption risks -- the worst class of bugs for RL training. The `big_values` issue directly defeats the memory-saving purpose on RTX 4090. Until these are fixed, delta weight sync should NOT be used.

---

### 3.2 Issue #6782: LoRA rank=64/alpha=128 Never Emits EOS

**Title**: `[fsdp, rollout] Qwen3.5-27B LoRA GRPO: vLLM never emits EOS with rank=64/alpha=128`
**Status**: OPEN (bug)
**Severity**: HIGH (completely breaks training -- all responses truncated, reward signal collapses)
**Author**: rongkunxue
**Environment**: 8x NVIDIA H20 (95GB each), FSDP2, vLLM TP=4, Qwen3.5-27B, unmerged LoRA (`lora.merge=False`)

**Root Cause Analysis**:
- The ONLY difference between working (rank=32/alpha=64) and broken (rank=64/alpha=128) runs is LoRA rank/alpha
- Higher alpha (128 vs 64) means the LoRA scaling factor `alpha/rank = 128/64 = 2.0` vs `64/32 = 2.0` -- the ratio is actually the same (2.0), so this is NOT a scaling issue per se
- The actual difference: rank=64 LoRA adapters have 2x more parameters per layer, which changes the **magnitude** of the output perturbation on the base model's logits
- When LoRA weights are applied via `add_lora` (unmerged mode), the larger adapter perturbation likely shifts the logit distribution enough that the EOS token probability drops below sampling threshold, causing infinite generation
- This is a vLLM-specific LoRA application bug, not a verl core bug

**RTX 4090 Impact**:
- RTX 4090 (24GB) with Qwen3.5-7B + LoRA rank=64 GRPO would likely reproduce this bug
- 24GB is insufficient for Qwen3.5-27B, but the same EOS suppression pattern would appear on any model where LoRA rank/alpha perturbation suppresses EOS probability
- On RTX 4090, workaround: use `lora.merge=True` (merged LoRA avoids the vLLM add_lora path) or stick to rank<=32

**Cross-Framework Connections**:
- vLLM's LoRA `add_lora` implementation (`vllm.lora.layers`) applies LoRA at inference time; the adapter magnitude is directly proportional to `lora_alpha / lora_rank * scale`
- SGLang's LoRA handling (`sglang.srt.layers.lora`) uses the same scaling formula; similar EOS suppression is possible
- The bug may be related to the IPC buffer aliasing issue fixed in PR #6688 -- if LoRA weights are views into reused buffers (even after the clone fix), stale or corrupted data could produce wrong logits
- DeepSpeed's LoRA training doesn't face this because it merges LoRA before inference, not at inference time

---

### 3.3 Issue #6786: Dynamic-CP Batch Split Bug

**Title**: `dynamic-cp split the batch into local_cp_size sub-batches bug`
**Status**: OPEN (bug)
**Severity**: MEDIUM (causes data imbalance in Megatron context-parallel training)
**Author**: liujia-cc

**Root Cause**: In `verl/utils/megatron_utils.py` line 1629, when `len(seq_len_effective) = 9` and `local_dp_size = 4`, distributing data leaves a DP rank with no data. Same issue when `len(seq_len_effective) < local_dp_size`.
**RTX 4090 Impact**: Minimal -- this is a Megatron/mcore path, not FSDP. RTX 4090 users typically use FSDP2, not Megatron.
**Cross-Framework**: Similar to DeepSpeed's data partitioning edge case when batch size is not divisible by DP degree. vLLM and SGLang don't have this issue as they don't use context parallelism in the same way.

---

### 3.4 PR #6799: Multimodal Continuous Token Support

**Title**: `feat(ct): Multimodal Continuous Token support for VL model families`
**Status**: DRAFT (open), depends on PR #6779
**Severity**: LOW (new feature, not a bug; but important for VLM GRPO training)
**Author**: Duckycoders (gxlvera + Jianye)

**Overview**: Extends Continuous Token (CT) framework from text-only to multimodal (VL). Adds QwenVL, MiMoVL builders; extends MergeResult with pixel_values, image_grid_thw, image_token_spans fields. 109 tests passing.
**RTX 4090 Impact**: Enables VL model GRPO training on RTX 4090 with smaller VL models (Qwen2.5-VL-3B/7B). No bugs reported yet, but GPU validation pending.
**Cross-Framework**: vLLM's multimodal support is more mature (native VLM inference). SGLang also supports VL models. This PR bridges the gap for agent-loop rollout with VLMs.

---

### 3.5 PR #6798: accumulated_idle_time Fix for Async

**Title**: `[fully_async] fix: introduce accumulated_idle_time to record the actual rollouter idle time`
**Status**: OPEN
**Severity**: HIGH (misleading metrics hide real performance bottleneck)
**Author**: mikequan0425 (jiahao.quan)
**Fixes**: Issue #6693

**Root Cause** (from issue #6693 and comment by TimurTaepov):
- `rollouter/idle_ratio` is computed in `reset_staleness` from `step_start_time` and `idle_start_time`
- This only accounts for ONE pause window in the current parameter version interval
- If the rollouter pauses/resumes through `should_pause_generation` or `resume_event`, repeated pause gaps are **missed or overwritten**
- Result: idle_ratio appears near 0 despite visible pause intervals in logs, masking that the rollout is actually training-bound and wasting GPU time
- The PR adds `accumulated_idle_time` that properly sums all idle intervals

**RTX 4090 Impact**: RTX 4090 async GRPO users would see misleading "near 0 idle" metrics, thinking their config is optimal while actually wasting GPU cycles. This masks the real bottleneck and prevents proper tuning.
**Cross-Framework**: SGLang's async server metrics handle this correctly by tracking cumulative idle time. vLLM's async engine also has idle time tracking, but in its own metrics pipeline. The fix pattern is the same: accumulate all idle windows rather than snapshotting a single one.

---

### 3.6 Issue #6772: FSDP2 update_weights OOM (CLOSED)

**Title**: `[FSDP2] update_weights execution order causes OOM on multi-node training`
**Status**: CLOSED (fixed)
**Severity**: HIGH (OOM on multi-node, critical for production)
**Author**: BeihaiTianming

**Root Cause**: PR #5031 (commit ef6eaa05) refactored `rollout_mode` into `update_weights` in `engine_workers.py`, unintentionally changing execution order. The new order resumes vLLM weights FIRST (allocating ~36.6GB for 27B bf16), then runs `state_dict()` all-gather (allocating ~27GB temporary). Peak overlap: 63.6GB > 61GB NPU limit -> OOM.

The correct order (from old `fsdp_workers.py`):
1. `get_per_tensor_param()` (all-gather peak is temporary, released after)
2. Offload actor + `aggressive_empty_cache()` (releases PyTorch reserved memory)
3. THEN resume vLLM weights (no overlap with all-gather peak)

**Current code state**: The local codebase at `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/engine_workers.py` still shows the PROBLEMATIC order (lines 708-741): resume weights FIRST, then get_per_tensor_param, then offload + cache clear. This means the fix from #6729 (prepare actor weights before rollout wakeup) may not have been applied yet to this local checkout, or the code was only partially reordered.

**RTX 4090 Impact**: CATASTROPHIC for multi-node RTX 4090 (24GB). For Qwen3.5-7B: vLLM weights ~14GB + FSDP2 all-gather peak ~7GB = 21GB peak overlap, which fits in 24GB but leaves only 3GB for activations and KV cache. For any model >7B on RTX 4090, this ordering bug would OOM.
**Cross-Framework**: This is fundamentally the same problem as vLLM's own weight loading + KV cache allocation ordering (vllm-project/vllm#25171 sleep/wake API). SGLang's sleep/wake also separates weight allocation from KV cache resume.

---

### 3.7 Issue #6780: Fully Async On-Policy Mismatch

**Title**: `[fully_async_policy] On-policy mode (trigger=1, staleness=0) didn't match main_ppo training`
**Status**: OPEN (bug)
**Severity**: CRITICAL (fundamental training correctness issue)
**Author**: Mecoli1219

**Root Cause**: With `trigger_parameter_sync_step=1`, `staleness_threshold=0.0`, `partial_rollout=False`, `require_batches=1`, the fully_async on-policy mode should be equivalent to synchronous training. But:
1. `training/rollout_actor_probs_pearson_corr` starts ~0.95 and decays to ~0.5 -- meaning the trainer consumes samples from a rollout weight version that has drifted from the current actor
2. Reward curve spikes then collapses vs stable slow climb in main_ppo
3. Neither `bypass_mode`, `use_kl_loss`, nor explicit importance sampling fixes this
4. There is an implicit lag mechanism even at staleness=0/trigger=1

**Maintainer Response**: wuxibin89 confirmed this is a known issue and directed users to the V1 unified trainer, noting the legacy `fully_async_main` is in maintenance mode. The V1 trainer's `PPOTrainerColocateAsync` and `PPOTrainerSeparateAsync` should resolve this through proper staleness control hooks.

**RTX 4090 Impact**: Any RTX 4090 user attempting async GRPO training with the legacy fully_async code path will get divergent training. Must use V1 trainer instead.
**Cross-Framework**: This is the same class of issue as DeepSpeed's off-policy correction challenges -- the fundamental problem is that async training inherently introduces policy lag even at staleness=0 if the weight sync and sample consumption pipelines are not perfectly synchronized. OpenRLHF and TRL face similar issues in their async implementations.

---

### 3.8 PR #6688: LoRA IPC Buffer Aliasing (MERGED)

**Title**: `[rollout] fix: clone LoRA weights out of the reused IPC buffer before add_lora`
**Status**: MERGED
**Severity**: HIGH (was causing cudaErrorIllegalAddress crash)
**Author**: HaozheZhang6

**Root Cause**: `vLLMColocateWorkerExtension._update_weights` passed per-bucket weight tensors (views into the receiver's single reused IPC bucket buffer from `bucketed_weight_transfer.py`) directly to `add_lora`. Since `add_lora` retains these tensors, later access reads freed/reused memory -> `cudaErrorIllegalAddress`.
**Fix**: Changed `weights = dict(weights)` to `weights = {name: tensor.clone() for name, tensor in weights}` to clone LoRA weights out of the IPC buffer before `add_lora` retains them.
**RTX 4090 Impact**: Directly fixes a crash path for RTX 4090 LoRA GRPO training. This bug would have hit any user running unmerged LoRA + vLLM + free_cache_engine.
**Cross-Framework**: Same pattern as vLLM's own IPC tensor lifetime issues (vllm-project/vllm's `rebuild_cuda_tensor` function also requires careful stream synchronization). The bucketed_weight_transfer.py IPC path mirrors vLLM's RLHF weight transfer example code.

---

### 3.9 PR #6699/6697: Detach model_output Metrics (MERGED)

**Title**: `[fsdp, trainer] fix: detach model_output and loss metrics to stop per-micro-batch graph retention`
**Status**: MERGED
**Severity**: HIGH (OOM in long-sequence LoRA runs)
**Fixes**: Issue #6698

**Root Cause**: `FSDPEngineWithLMHead.forward_step` returns `model_output` (log_probs/entropy) still attached to the autograd graph. `forward_backward_batch` holds these in `output_lst` until the whole batch finishes, pinning per-micro-batch: activation checkpoint saved embeddings + gradient buffers. Under LoRA (`enable_input_require_grads`), this creates a linear memory leak per micro-batch.

**Memory impact**: Before fix: 24.8 GiB -> 37.9 -> 50 -> 64 GiB (OOM). After fix: stable 16.2 GiB.
**RTX 4090 Impact**: Critical for RTX 4090 -- the leak would OOM within ~2 micro-batches on a 24GB card with LoRA + long sequences. The fix prevents this.
**Cross-Framework**: DeepSpeed's ZeRO-3 avoids this by explicitly detaching metrics tensors before accumulation. PyTorch's FSDP documentation warns about retaining computation graph references across micro-batches.

---

### 3.10 PR #6738: Skip Redundant Clone in SGLang Weight Sync (MERGED)

**Title**: `[rollout] fix: skip redundant clone in get_named_tensor_buckets to avoid OOM during SGLang weight sync`
**Status**: MERGED
**Severity**: HIGH (OOM on multi-GiB fused MoE weights)

**Root Cause**: `get_named_tensor_buckets` unconditionally `clone()`d every tensor before bucketing. The clone is only needed for views into larger buffers; DTensor.full_tensor() all-gather results already own tight contiguous storage. Cloning them transiently doubles footprint and OOMs on MoE weights (gate_up_proj/qkv).
**Fix**: Clone only when `not (tensor.is_contiguous() and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size())`.
**RTX 4090 Impact**: Directly fixes OOM for SGLang-based GRPO on RTX 4090 with MoE models. The unnecessary clone would double peak memory during weight sync, easily exceeding 24GB for even modest models.

---

## 4. Additional Recent Issues (June 8-20)

### 4.1 PR #6729: Prepare Actor Weights Before Rollout Wakeup (OPEN)

**Title**: `fix(workers): prepare actor weights before rollout wakeup`
**Severity**: HIGH (fixes Ascend/vLLM-Ascend OOM during wake_up)
**Root Cause**: Same class as #6772 -- the colocated `naive` path resumes rollout weights before extracting actor weights, causing memory peak overlap. This PR reorders the sequence: extract params first, then offload + cache clear, then resume rollout.
**Status**: Still OPEN -- not yet merged. The local codebase still has the problematic order.

### 4.2 PR #6795: Remove Invalid single_turn_response_length Override (OPEN)

**Title**: `[fully_async] fix: remove invalid rollout.single_turn_response_length override`
**Severity**: LOW (config key doesn't exist in RolloutConfig, causing script launch failure)
**Root Cause**: Example script references a Hydra override key that doesn't exist.

### 4.3 PR #6796: Align Aggregated Metrics Logging (OPEN)

**Title**: `[fully_async, trainer] fix: align aggregated metrics logging with current step`
**Severity**: MEDIUM (metrics appear shifted by one sync window)
**Root Cause**: Metrics flushed during parameter sync before current step added to aggregator.

### 4.4 Issue #6792: OPD Teacher Model OOM (OPEN)

**Title**: `opd, fsdp, 910b3, teacher model Qwen3-235B OOM on 2-node deployment`
**Severity**: MEDIUM (configuration issue for distillation, not a verl code bug)

### 4.5 Issue #6703: OOB IndexError in Megatron FP8+CP (OPEN)

**Title**: `[mcore] fix: OOB IndexError in preprocess_thd_engine when FP8 padding + CP > 1`
**Severity**: MEDIUM (crash in specific Megatron configuration)
**Root Cause**: FP8 padding guard `if d.numel() < align_size` was too weak; should check `d.numel() < seqlen_padded_i`.

### 4.6 Issue #6666: Rollout Weight Sync Debug Check (OPEN)

**Title**: `[vllm, checkpoint] feat: add rollout weight sync debug check`
**Severity**: LOW (diagnostic feature, TP=1/DP=1 only)
**Purpose**: Verifies vLLM loaded same weights trainer sent, catching silent sync corruption.

### 4.7 Issue #6656: VLM Text-Only Position IDs Crash (OPEN)

**Title**: `fix(agent_loop): handle text-only input in _compute_position_ids for VLM models`
**Severity**: MEDIUM (crashes when running VL model on text-only tasks like GSM8K)
**Root Cause**: `_compute_position_ids` always calls `get_rope_index` even when no images/videos present; passing `None` for `image_grid_thw` causes crash.

---

## 5. NEW Issues June 18-20 (Untracked Patterns)

These issues opened June 18-20 and may not yet be widely recognized:

| # | Title | Date | Severity | Notes |
|---|-------|------|----------|-------|
| 6794 | Delta weight sync (sparse) | Jun 18 | HIGH | 4 blocking sub-issues from review (see 3.1) |
| 6793 | Open-R1 multimodal dataset support | Jun 18 | LOW | Feature PR |
| 6795 | Invalid config override | Jun 18 | LOW | Script bug |
| 6796 | Async metrics alignment | Jun 18 | MEDIUM | Metrics off-by-one |
| 6798 | accumulated_idle_time fix | Jun 19 | HIGH | Fixes misleading async metrics |
| 6799 | Multimodal CT support | Jun 19 | MEDIUM | Feature; GPU validation pending |

**New untracked pattern**: The combination of PRs #6794 (delta sync) + #6798 (idle time fix) + #6796 (metrics alignment) + #6795 (config cleanup) suggests the fully_async infrastructure is undergoing active stabilization. Users should expect continued fixes and avoid relying on legacy `fully_async_main` for production training.

---

## 6. RTX 4090 GRPO Impact Assessment

### 6.1 Memory Constraints

RTX 4090 has 24GB VRAM. Key memory consumers during GRPO:

| Component | Memory (approx, 7B model) | Notes |
|-----------|--------------------------|-------|
| vLLM weights (bf16) | ~14GB | For 7B params |
| FSDP2 all-gather peak | ~7GB | Temporary during state_dict() |
| LoRA adapter (rank=32) | ~0.2GB | Per adapter, all-linear |
| LoRA adapter (rank=64) | ~0.4GB | 2x rank=32 |
| KV cache | ~2-4GB | Depends on sequence length |
| Activation checkpoints | ~1-3GB | With gradient checkpointing |
| Optimizer states | offloaded to CPU | With param_offload=True |

### 6.2 Critical RTX 4090 Bugs

1. **OOM from weight sync ordering** (#6772, #6729): Peak overlap of vLLM weights + FSDP all-gather = 21GB, leaving only 3GB. Fix requires reorder.
2. **LoRA rank=64 EOS suppression** (#6782): Higher-rank LoRA may suppress EOS probability in vLLM, causing infinite generation.
3. **IPC buffer aliasing** (#6688, MERGED): Already fixed, but verify local checkout has the fix.
4. **Metrics graph retention** (#6699, MERGED): Already fixed, but verify local checkout has the fix.
5. **SGLang redundant clone** (#6738, MERGED): Already fixed, but verify local checkout has the fix.

### 6.3 Recommended RTX 4090 GRPO Configuration

Based on the bug analysis:

```
# For Qwen2.5-7B or similar on RTX 4090
actor_rollout_ref.model.lora_rank=32       # NOT 64 (EOS suppression risk)
actor_rollout_ref.model.lora_alpha=64      # Keep alpha/rank ratio at 2.0
actor_rollout_ref.model.lora.merge=True    # Merged mode avoids add_lora path
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # Conservative for 24GB
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.enforce_eager=True
# Use V1 trainer, NOT legacy fully_async_main
```

---

## 7. Cross-Framework Connection Map

| verl Issue | vLLM Equivalent | SGLang Equivalent | DeepSpeed Equivalent |
|------------|----------------|-------------------|---------------------|
| #6794 record_stream | vllm#31848 (sparse weight API) | sgl#26519 (delta receiver) | N/A (ZeRO avoids this) |
| #6794 disk_race | vllm#3082 (ckpt race) | sgl distributed barrier pattern | DS checkpoint barrier |
| #6794 big_values | vllm weight loader intermediates | sgl bucket allocation | ZeRO-3 partitioning avoids |
| #6782 LoRA EOS | vllm LoRA scaling bug | sgl LoRA same formula | DS merges LoRA before inference |
| #6688 IPC aliasing | vllm RLHF IPC lifetime | sgl IPC not used for weights | DS uses NCCL broadcast |
| #6772 OOM ordering | vllm#25171 sleep/wake API | sgl sleep/wake level system | DS separate train/infer |
| #6699 graph retention | vllm model output detach | sgl similar pattern | DS explicit detach in ZeRO |
| #6738 redundant clone | vllm weight bucket clone | sgl bucketing | DS NCCL no clone needed |
| #6780 async mismatch | vllm async engine lag | sgl async server staleness | DS off-policy correction |
| #6798 idle metrics | vllm async idle tracking | sgl cumulative idle tracking | DS throughput metrics |

---

## 8. Key Source Code Files for GRPO Training Flow

### 8.1 Training Worker & Engine

| File (absolute path) | Role |
|----------------------|------|
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/engine_workers.py` | `TrainingWorker` class; `update_weights()` method (lines 670-749) -- THE critical weight sync path |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/engine.py` | `BaseEngine` and engine registry |

### 8.2 V1 Trainer

| File (absolute path) | Role |
|----------------------|------|
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/trainer/ppo/v1/trainer_base.py` | V1 unified trainer base; hook pattern; transfer_queue integration |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/trainer/ppo/v1/trainer_sync.py` | Synchronous trainer hooks (update_weights on step end, sleep on sample end) |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/trainer/ppo/v1/trainer_colocate_async.py` | Colocated async; partial rollout |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/trainer/ppo/v1/trainer_separate_async.py` | Disaggregated async |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/trainer/main_ppo.py` | Entry point; Hydra config; Ray initialization |

### 8.3 Rollout Engines

| File (absolute path) | Role |
|----------------------|------|
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py` | vLLM rollout adapter; ServerAdapter class |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/vllm_rollout/utils.py` | vLLM colocated worker extension; LoRA add_lora path (line ~308, the clone fix from #6688) |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/vllm_rollout/weight_update_utils.py` | split_buffer_updates + apply_buffer_updates |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` | ZMQ + IPC/shared-memory bucketed weight sender/receiver |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py` | SGLang rollout; update_weights with tensor bucketing; LoRA via LoadLoRAAdapterFromTensorsReqInput |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/workers/rollout/sglang_rollout/utils.py` | get_named_tensor_buckets (the clone optimization from #6738) |

### 8.4 Checkpoint Engine (Weight Sync Backend)

| File (absolute path) | Role |
|----------------------|------|
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/checkpoint_engine/base.py` | CheckpointEngine base; CheckpointEngineRegistry; TensorMeta; async send/receive pattern |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/checkpoint_engine/nccl_checkpoint_engine.py` | NCCL broadcast + ZMQ async; BroadcastOperation with cupy |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/checkpoint_engine/mooncake_checkpoint_engine.py` | Mooncake RDMA transfer |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/checkpoint_engine/nixl_checkpoint_engine.py` | NVIDIA NIXL transfer |

### 8.5 Utility / Memory

| File (absolute path) | Role |
|----------------------|------|
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/utils/memory_utils.py` | `aggressive_empty_cache()` -- gc.collect + empty_cache + synchronize |
| `/Users/jackiemac/workspace/rollout-infra/verl/verl/utils/megatron_utils.py` | Megatron context-parallel batch split (bug in #6786 at line 1629) |

---

## 9. Summary Priority Matrix

| Priority | Issue | Action Required |
|----------|-------|----------------|
| P0 CRITICAL | #6780 async on-policy mismatch | Migrate to V1 trainer; legacy fully_async is unreliable |
| P0 CRITICAL | #6772/#6729 weight sync OOM ordering | Verify local code has reorder fix; patch if not |
| P1 HIGH | #6794 delta sync 4 sub-issues | Do NOT use delta sync until record_stream + disk_race + big_values + makedirs are fixed |
| P1 HIGH | #6782 LoRA EOS suppression | Use lora.merge=True or rank<=32 on RTX 4090 |
| P1 HIGH | #6798 async idle metrics | Apply fix; cannot tune async config without correct idle_ratio |
| P1 HIGH | #6699 graph retention (MERGED) | Verify local checkout includes detach fix |
| P1 HIGH | #6738 SGLang clone (MERGED) | Verify local checkout includes smart clone |
| P1 HIGH | #6688 LoRA IPC clone (MERGED) | Verify local checkout includes clone fix |
| P2 MEDIUM | #6786 CP batch split | Megatron-only; patch if using mcore |
| P2 MEDIUM | #6796 metrics alignment | Apply when using fully_async |
| P2 MEDIUM | #6656 VLM text-only crash | Patch agent_loop.py for text-only VLM tasks |
| P3 LOW | #6799 multimodal CT | Wait for GPU validation before adopting |
| P3 LOW | #6795 config override | Remove invalid key from scripts |
