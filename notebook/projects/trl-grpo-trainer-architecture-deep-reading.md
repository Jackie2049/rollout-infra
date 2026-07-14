# TRL GRPOTrainer Architecture Deep Reading

## Overview

**File**: `trl/trainer/grpo_trainer.py` (3248 lines)
**Class**: `GRPOTrainer(_BaseTrainer)` — HuggingFace TRL's GRPO implementation
**Paper**: DeepSeekMath (2402.03300)
**Status**: Active development, merges daily (6353+ PRs)
**License**: Apache 2.0

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRPOTrainer                                  │
│  (extends _BaseTrainer ← Trainer ← transformers.Trainer)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  training_step() ────→ _prepare_inputs() ────→ _compute_loss()       │
│       │                                         ↑                    │
│       │  (every generate_every steps)           │                    │
│       └─────────────────────────────────────────┘                    │
│                    ┌───────────────────┐                             │
│                    │ _generate_and_    │                             │
│                    │ score_completions │                             │
│                    └────────┬──────────┘                             │
│                             │                                        │
│              ┌──────────────┼──────────────┐                        │
│              ▼              ▼              ▼                         │
│       _generate()   _calculate_rewards()   KL/ref                    │
│              │              │              computation               │
│              │              │                                        │
│       ┌──────┴──────┐      │                                        │
│       ▼             ▼      │                                        │
│  _tokenize_   _generate_   │                                        │
│  prompts()    single_turn()│                                        │
│               (or vLLM)    │                                        │
│                    │        │                                        │
│                    ▼        ▼                                        │
│              _tool_call_loop()                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `__init__` (lines 293-1038) — Configuration and Setup

**Model Loading**:
- Accepts: `str` (model ID/path), `PreTrainedModel`, or `PeftModel`
- When string: loads via `create_model_from_path()` with `model_init_kwargs`
- Distributed: sets `device_map=None` for MULTI_GPU/DEEPSPEED

**PEFT Support**:
- Configurable via `peft_config` parameter (e.g., `LoraConfig`)
- ZeRO-3 + PEFT fix: `autocast_adapter_dtype=False` for non-quantized models (ref: TRL #6089, DeepSpeed #8072)
- Pre-trained adapter handling: creates "ref" adapter as frozen copy of "default" adapter
- When `target_parameters` used in LoRA: falls back to base model for ref logprobs

**Reward Functions** (lines 476-535):
- Multi-reward-function system with weighted aggregation
- Each reward func can be: str (model ID), `nn.Module` (reward model), or custom callable
- Custom functions receive `prompts`, `completions`, `completion_ids`, `trainer_state`, `log_extra`, `log_metric`, `environments`
- Async coroutine functions supported via daemon event loop
- `reward_weights`: configurable per-function weighting
- `None` return values → NaN → excluded from aggregation

**Reference Model** (lines 886-904):
- `beta=0.0` → no ref model (no KL penalty)
- PEFT model → ref model = disabled adapter (no extra model needed)
- Non-PEFT → separate `create_model_from_path()` instance

**vLLM Integration**:
- Mode: `colocate` (same GPU) or `server` (separate process)
- Sleep mode: `vllm_enable_sleep_mode` for memory efficiency
- Importance sampling correction for distribution mismatch

**Loss Configuration**:
- `loss_type`: "grpo", "bnpo", "dr_grpo", "dapo", "luspo", "cispo", "sapo", "vespo"
- `multi_objective_aggregation`: "sum_then_normalize" or "normalize_then_sum"
- `scale_rewards`: "group", "batch", or "none"
- `importance_sampling_level`: "token" or "sequence"
- `use_liger_kernel`: fused GRPO loss via Liger

**Data Flow**:
- `num_iterations` (μ): number of optimization steps per generation
- `steps_per_generation`: how many optimizer steps between generations
- `_buffered_inputs`: reuses generated completions across multiple updates
- `generate_every = steps_per_generation * num_iterations`

### 2. `training_step` (line 1400) — Main Training Step

```python
def training_step(self, model, inputs):
    # Delegates to super().training_step which calls compute_loss
    # Tracks step timing
    return loss
```

### 3. `_prepare_inputs` (line 1412) — Buffered Generation Dispatch

```python
def _prepare_inputs(self, generation_batch):
    # Every generate_every steps: call _generate_and_score_completions
    # Buffer outputs in self._buffered_inputs
    # Split buffered batches for gradient accumulation
    # Return current micro-batch of inputs
```

**Key Logic**:
- Generation frequency: every `steps_per_generation * num_iterations` steps
- Buffered inputs reused across `num_iterations` forward/backward passes
- Each call returns a slice of the buffered batch for gradient accumulation
- At the end of the generation cycle: refills the buffer

### 4. `_generate` (line 2000) — Completion Generation

**Two paths**:
1. **Custom `rollout_func`**: user-provided function receives prompts + trainer instance
   - Must return `prompt_ids`, `completion_ids`, `logprobs`
   - Used with vLLM: syncs weights before generation
2. **Built-in path**: tokenize → `_generate_single_turn` → decode
   - Uses transformers `GenerationConfig` or vLLM backend

**Tool Calling** (lines 2048-2072):
- If tools configured: `_tool_call_loop` extracts and executes tool calls
- Multi-turn: tool responses appended as new messages
- `tool_mask`: marks model-generated tokens (1) vs external tokens (0)
- Tool images merged into forward kwargs

**Return Value**:
```python
(prompt_ids, completion_ids, tool_mask, completions,
 total_completion_tokens, logprobs, extra_fields, images, tool_images)
```

### 5. `_generate_and_score_completions` (line 2128) — Full Pipeline

**Phase 1: Environment Setup** (lines 2136-2156)
- Per-rollout environment instances from pool
- Environment reset with prompt kwargs
- Multi-environment dispatch

**Phase 2: Generation** (lines 2220-2230)
- Calls `_generate(prompts)`
- Tokenizes prompts → generates completions → tool call loop

**Phase 3: Padding & Masking** (lines 2234-2283)
- Left-pad prompts, right-pad completions
- `mask_truncated_completions`: zero out truncated sequences

**Phase 4: Forward Pass for old_logprobs** (lines 2409-2432)
- If generation and optimization misaligned: compute `old_per_token_logps`
- vLLM: always compute for importance sampling correction
- Without misalignment: `old_per_token_logps = per_token_logps.detach()`

***THIS IS THE EQUIVALENT OF verl's `bypass_mode`***:
When `num_iterations == 1` AND `steps_per_generation <= gradient_accumulation_steps`:
- `old_per_token_logps == per_token_logps` → skip forward pass
- `old_per_token_logps = per_token_logps.detach()` — zero-cost

This means **TRL already implements bypass mode implicitly** when generation and optimization are aligned. verl's `bypass_mode=True` is a more explicit version of this optimization.

**Phase 5: vLLM Importance Sampling** (lines 2434-2479)
- Corrects distribution mismatch between vLLM rollout and training model
- Modes: `token_truncate`, `token_mask`, `sequence_truncate`, `sequence_mask`
- Computes `exp(log_ratio)` per token or per sequence
- Clips or masks extreme ratios

**Phase 6: Reference Model Forward** (lines 2481-2511)
- If `beta != 0.0`: compute `ref_per_token_logps`
- Separate ref model or disabled adapter

**Phase 7: Reward Computation** (lines 2513-2620)
- Calls `_calculate_rewards`
- Two aggregation modes for multi-objective rewards

### 6. `_calculate_rewards` (line 1469) — Reward Evaluation

**Three reward function types**:
1. **`nn.Module` (reward model)**: tokenize prompt+completion → forward pass → logits
2. **Async coroutine**: scheduled on daemon event loop, `asyncio.gather` for parallelism
3. **Sync callable**: called directly with `(prompts, completions, **kwargs)`

**Reward Kwargs** (lines 1475-1488):
- All extra dataset columns
- `trainer_state`: for dynamic reward shaping
- `log_extra`: for logging additional completion columns
- `log_metric`: for logging additional scalar metrics
- `environments`: per-rollout environment instances

**Gathering**:
- `rewards_per_func` gathered across all processes via `gather()`
- Critical for correct per-group normalization

### 7. `_compute_loss` (line 2858) — Loss Computation

**Input Preparation**:
- Concatenate prompt + completion IDs
- Handle multimodal inputs (pixel_values, image_grid_thw, etc.)
- Compute `per_token_logps` and `entropies` via forward pass

**Importance Sampling** (lines 2921-2931):
- Two levels: `token` or `sequence`
- `log_ratio = per_token_logps - old_per_token_logps`
- `coef_1 = exp(log_importance_weights)`

**KL Divergence** (lines 2936-2943):
```python
per_token_kl = exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
```
- Optional bias correction (`use_bias_correction_kl`)

**Loss Types** (lines 2947-2976):

| Loss Type | Formula | Notes |
|-----------|---------|-------|
| `grpo` | `-min(coef_1 * A, clip(coef_1) * A)` | Standard PPO-clip |
| `bnpo` | Same as grpo | Batch-normalized |
| `dr_grpo` | Same as grpo | Different normalizer |
| `dapo` | Same as grpo | Different normalizer |
| `luspo` | Same as grpo | Sequence-level |
| `cispo` | `-clamp(coef_1, max=ε_high) * A * logps` | One-sided clip |
| `sapo` | `-sigmoid(T * (coef_1-1)) * 4/T * A` | Soft clip |
| `vespo` | `-phi * A * logps` | Gamma weights |

**Two-Sided Clipping** (for grpo/bnpo/dr_grpo/dapo/luspo):
```python
coef_1 = exp(log_importance_weights)
coef_2 = clamp(coef_1, 1-ε_low, 1+ε_high)
per_token_loss1 = coef_1 * advantages
per_token_loss2 = coef_2 * advantages
per_token_loss = -min(per_token_loss1, per_token_loss2)
```

**Optional `delta` clipping** (line 2953-2954):
```python
coef_1 = clamp(coef_1, max=delta)  # DAPO-style
```

**Loss Normalization** (lines 2990-3017):
- `grpo/sapo`: per-sequence mean, divided by gradient accumulation steps
- `bnpo`: global token mean
- `dr_grpo`: global mean over `(batch * max_completion_length)`
- `cispo/dapo/vespo`: divided by `num_items_in_batch` (DAPO-style)
- `luspo`: masked sequence mean

**Entropy Bonus** (lines 3019-3048):
- Static coefficient or adaptive control
- Adaptive: tracks running window entropy, adjusts coefficient up/down
- Target: maintain `entropy_target` via PID-like controller
- Gates bonus on/off based on `_last_world_entropy <= entropy_target`

### 8. `compute_liger_loss` (line 2711) — Fused GRPO Loss via Liger

**Mechanism**:
- Bypasses `model.forward()` entirely
- Calls backbone directly (gets hidden states)
- Multiplies by `lm_head.weight` (fused op)
- Handles ZeRO-3 gathered lm_head for PEFT compatibility

**Restrictions**:
- PEFT on `lm_head` incompatible (adapter weights invisible to fused op)
- Prompt-learning methods (PromptTuning/PrefixTuning) incompatible
- Entropy bonus not supported
- Off-policy mask not supported

### 9. Training Loop Flow

```
for each optimizer step:
    inputs = _prepare_inputs(generation_batch)
    # Every generate_every steps:
    if step % generate_every == 0:
        self._buffered_inputs = _generate_and_score_completions(inputs)
    # Slice buffered inputs for this micro-batch
    micro_batch = self._buffered_inputs[current_micro_batch]
    loss = _compute_loss(model, micro_batch)
    loss.backward()
    optimizer.step()
```

### 10. Advantage Computation (lines 2537-2586)

**Mode 1: `sum_then_normalize`** (standard GRPO):
```python
rewards = sum(reward_func_i * weight_i)  # weighted sum
mean_grouped = mean(rewards.reshape(-1, num_generations), dim=1)
std_grouped = std(rewards.reshape(-1, num_generations), dim=1)
advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)
```

**Mode 2: `normalize_then_sum`** (per-function normalization):
```python
reward_k = (grouped - mean_k) / (std_k + 1e-4)  # per-func
rewards = sum(reward_k * weight_k)
advantages = (rewards - global_mean) / (global_std + 1e-4)
```

**Scale options**:
- `group`: per-group normalization (standard GRPO)
- `batch`: global batch normalization
- `none`: zero-mean only (no scaling)

**Unscorable completions**: NaN advantages → `torch.nan_to_num(advantages, nan=0.0)`

## Key Differences from verl GRPO

| Feature | TRL | verl |
|---------|-----|------|
| **Architecture** | Extends HF Trainer | Custom training loop |
| **Bypass mode** | Implicit (detach when aligned) | Explicit (`bypass_mode=True`) |
| **Generation** | Transformers or vLLM | vLLM or SGLang |
| **Loss types** | 8 (grpo/bnpo/dr_grpo/dapo/luspo/cispo/sapo/vespo) | ~11 (ppo_clip, clip_high, etc.) |
| **Advantage** | 2 modes, 3 scale options | ~14 estimators |
| **Reward multi-obj** | Sum-then-norm / norm-then-sum | Multiple KL estimators |
| **FSDP2** | Via HF Trainer + Accelerate | Custom FSDP integration |
| **PEFT** | Native via PEFT library | per-unit LoRA summon |
| **Tool calling** | Full multi-turn tool support | Not supported |
| **Environments** | Per-rollout env pool | Not supported |
| **KL penalty** | Per-token, bias-corrected | Per-token, multiple variants |
| **Entropy** | Static + adaptive control | Not explicitly supported |
| **Liger kernel** | Supported | Not supported |

## RTX 4090 Implications

**Strengths**:
1. TRL uses HF Trainer → well-tested with Accelerate + DeepSpeed ZeRO-2
2. Implicit bypass mode when `num_iterations=1` → saves one forward pass
3. Multi-reward func system useful for GRPO on single GPU
4. Liger kernel can reduce memory via fused ops
5. vLLM colocate mode with sleep/wake for weight sharing

**Concerns**:
1. No explicit `bypass_mode` flag — relies on generation alignment detection
2. No per-unit LoRA summon (unlike verl #6512) → full model forward for ref logprobs
3. Larger memory footprint than verl's custom pipeline
4. vLLM importance sampling adds compute cost
5. Buffered generation + multiple iterations may waste GPU memory

**Recommendations for RTX 4090**:
1. Set `num_iterations=1`, `steps_per_generation=1` → implicit bypass
2. Use DeepSpeed ZeRO-2 + CPU_Adam (ZeRO-3 is pure overhead on single GPU)
3. Enable `use_liger_kernel=True` for memory-efficient loss
4. Set `beta=0.0` if KL penalty not needed (saves ref model forward)
5. Use PEFT (LoRA) with `autocast_adapter_dtype=False`
6. Set `scale_rewards="none"` for group_size=1 (avoids division by zero)

## OSS Contribution Opportunities

### P9 — Bypass mode explicit flag
- **Current**: TRL infers bypass opportunity from generation alignment
- **Improvement**: Add explicit `bypass_mode` flag (like verl)
- **Rationale**: Current heuristic may fail with non-standard schedules
- **Complexity**: ~20 LOC in `_generate_and_score_completions` + 2 config params

### P9 — Per-unit LoRA summon for memory reduction
- **Current**: Full model forward for ref logprobs
- **Improvement**: Dynamic FSDP unit discovery (port from verl #6512)
- **Rationale**: 10x peak memory reduction (60→6-8 GiB)
- **Complexity**: ~200-300 LOC, requires FSDP integration

### P8 — GRPO group_size=1 detection and warning
- **Current**: No warning when `num_generations=1` (degenerates to REINFORCE)
- **Improvement**: Log warning + auto-set `scale_rewards="none"`
- **Rationale**: Prevents silent training degradation
- **Complexity**: ~10 LOC

### P8 — Off-policy mask integration for Liger kernel
- **Current**: Liger kernel doesn't support `off_policy_mask_threshold`
- **Improvement**: Implement off-policy masking within Liger fused op
- **Rationale**: Enables off-policy correction with fused kernel
- **Complexity**: ~50 LOC in Liger kernel integration

### P7 — Delta sync for vLLM weight transfer
- **Current**: Full weight sync on every generation step
- **Improvement**: Delta weight sync (port from verl #6794)
- **Rationale**: ~100x payload reduction for LoRA training
- **Complexity**: ~150 LOC in vLLM integration layer

### P7 — DeepSpeed ZeRO-3 + PEFT stability
- **Current**: `autocast_adapter_dtype=False` workaround
- **Improvement**: Proper mixed-dtype handling in DeepSpeed ZeRO-3
- **Rationale**: Fixes root cause, not just symptom
- **Complexity**: ~50 LOC in DeepSpeed engine

### P6 — CUDA graph support for GRPO
- **Current**: No CUDA graph capture for GRPO training
- **Improvement**: Static forward pass capture (when no tool calling)
- **Rationale**: ~30% training throughput improvement
- **Complexity**: ~100 LOC, conditional on no-tool mode

### P6 — Top-n-sigma reward clipping
- **Current**: No reward outlier clipping
- **Improvement**: Port top-n-sigma from vLLM's PPO to TRL GRPO
- **Rationale**: Stabilizes training with noisy reward functions
- **Complexity**: ~30 LOC in advantage computation

## Architecture Summary

```
TRL GRPOTrainer
├── Data Flow: buffered generation → multi-step optimization
├── Generation: transformers or vLLM (colocate/server)
├── Rewards: multi-func, weighted, async-capable
├── Loss: 8 types, PPO-clip core, Liger fused option
├── KL: per-token, bias-corrected, configurable beta
├── Entropy: static or adaptive (PID-like controller)
├── Tools: multi-turn, environment pooling
└── Reference model: PEFT adapter or separate instance
```
