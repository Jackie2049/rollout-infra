# DeepSpeed OPD Trainer + LoRA Gap -- Deep Reading (Updated July 15, 2026)

> **Status Update**: PR #8027 MERGED July 12, but only the rollout engine infrastructure (6 files, +888/-0).
> The actual OPD trainer (losses.py, teacher.py, trainer.py, config.py) remains a follow-up item.
> Q3 roadmap #8104 confirms OPD as a Q3 target: "Systems Design, prototyping and benchmarking."

---

## 1. What Was Actually Merged in PR #8027

### Merged files (6, +888 LOC):

| File | Role | Lines |
|------|------|-------|
| `deepspeed/runtime/rollout/__init__.py` | Factory + exports | 31 |
| `deepspeed/runtime/rollout/base.py` | RolloutEngine ABC + dataclasses | ~120 |
| `deepspeed/runtime/rollout/hybrid_engine_rollout.py` | HybridEngineRollout impl + CUDA graph capture | ~230 |
| `deepspeed/utils/static_cache.py` | DeepSpeedStaticCache for graph capture | ~120 |
| `tests/unit/runtime/rollout/test_hybrid_engine_rollout.py` | HybridEngine rollout tests | ~100 |
| `tests/unit/runtime/rollout/test_rollout_interface.py` | Interface conformance tests | ~100 |

### NOT merged (still follow-up items):
- `opsd/trainer.py` (197 lines) -- Three-phase training loop, OPSDTrainer class
- `opsd/teacher.py` (191 lines) -- TeacherWrapper + TeacherLogitCache
- `opsd/losses.py` (192 lines) -- Forward-KL, reverse-KL, JSD + chunked/streamed loss
- `opsd/config.py` (149 lines) -- OPSDConfig + all sub-configs dataclasses
- `opsd/data.py` (108 lines) -- PromptDataset + LeftPaddedPromptCollator
- `opsd/utils.py` (52 lines) -- build_response_mask + shift_for_next_token_prediction
- vLLM rollout backend
- WeightBridge for TP slicing
- Benchmark scripts
- Example configs

**Key insight**: DeepSpeed merged only the rollout abstraction layer (the reusable infrastructure),
keeping the OPD-specific trainer as a separate follow-up. This is architecturally sound -- the
RolloutEngine ABC serves both OPD and future RL trainers, while the trainer itself is application-specific.

---

## 2. Merged Rollout Engine Architecture (Source-Level Trace)

### RolloutEngine ABC (`deepspeed/runtime/rollout/base.py`):

```python
class RolloutEngine(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        """Run student generate, return prompt+response."""

    def sync_weights(self, step: int) -> None:
        """Push student weights into rollout backend. No-op for hybrid engine."""

    def shutdown(self) -> None:
        """Release backend resources."""
```

### Data classes:
- `RolloutConfig` -- engine type ("hybrid_engine"), use_graph_capture
- `SamplingConfig` -- max_new_tokens, temperature, top_p, top_k, n_samples_per_prompt
- `RolloutRequest` -- prompt_ids [B, T_p] left-padded, prompt_attention_mask [B, T_p]
- `RolloutBatch` -- input_ids [B', T_p+T_r] right-padded, attention_mask, response_start_idx

### HybridEngineRollout (`deepspeed/runtime/rollout/hybrid_engine_rollout.py`):

Two generation paths:
1. **model.generate()** (default): delegates to HuggingFace generate. Supports sampling and greedy.
2. **CUDA graph capture + DeepSpeedStaticCache**: only for greedy (temperature=0). Pre-allocates
   a StaticCache, captures the decode forward pass, replays for each decode step.

**Critical design**: The graph capture path removes forward hooks before capture and restores after.
This is necessary because ZeRO-3 hooks call synchronize(), which is illegal during CUDA graph capture.

**DeepSpeedStaticCache** (`deepspeed/utils/static_cache.py`):
- Write position is supplied externally via shared tensor (not internal counter)
- HuggingFace StaticCache's `cumulative_length` counter freezes during graph capture -> wrong KV positions
- Our cache reads `write_position` from a real tensor at fixed address -> CUDA graph replays read current value
- Caller must call `cache.set_write_position(pos)` before each replay

### `build_rollout` factory (`deepspeed/runtime/rollout/__init__.py`):
Currently only supports `hybrid_engine`. vLLM backend was in the original PR but not merged.
Extensible: adding vLLM, SGLang, or other backends just requires registering with the factory.

---

## 3. The OPD Trainer -- What Still Needs to Be Built

Based on the original PR #8027 draft (from prior deep reading, see `projects/deepspeed-opd-trainer-source-reading.md`),
the OPD trainer has a three-phase architecture:

```
Phase 0: Student Rollout
    prompts -> HybridEngineRollout.generate() -> RolloutBatch (input_ids + attention_mask + response_start_idx)

Phase 1: Teacher Forward + CPU Logit Cache
    input_ids + attention_mask -> TeacherWrapper.forward_to_cache() -> TeacherLogitCache (CPU-resident)

Phase 2: Student Forward + Streamed Divergence + Backward
    input_ids + attention_mask -> student_engine() -> student_logits [B, T, V]
    -> shift for next-token prediction
    -> streamed_distillation_loss(student_logits_shifted, teacher_chunk_fetcher, mask_shifted)
    -> student_engine.backward(loss)
    -> student_engine.step()
    -> teacher_cache.free()
```

### What exists vs. what needs to be implemented:

| Component | Merged? | Location (original PR) | What it does |
|-----------|---------|------------------------|--------------|
| RolloutEngine ABC | YES | `deepspeed/runtime/rollout/base.py` | Abstract interface |
| HybridEngineRollout | YES | `deepspeed/runtime/rollout/hybrid_engine_rollout.py` | In-process generation |
| DeepSpeedStaticCache | YES | `deepspeed/utils/static_cache.py` | CUDA graph-compatible KV cache |
| OPSDTrainer (3-phase loop) | NO | `opsd/trainer.py` | Training orchestration |
| TeacherWrapper + TeacherLogitCache | NO | `opsd/teacher.py` | ZeRO-3 CPU-offloaded teacher + CPU logit cache |
| streamed_distillation_loss | NO | `opsd/losses.py` | Chunked forward-KL/reverse-KL/JSD |
| OPSDConfig | NO | `opsd/config.py` | Dataclass configuration |
| PromptDataset | NO | `opsd/data.py` | Data loading + collation |
| vLLM Rollout | NO | `opsd/rollout/vllm.py` | Disjoint GPU rollout |
| WeightBridge | NO | `opsd/weight_bridge/` | TP slicing for weight sync |

---

## 4. DeepSpeed RLHF Training Pipeline (Existing DeepSpeed-Chat)

DeepSpeed-Chat (microsoft/DeepSpeedExamples) implements the InstructGPT 3-step pipeline:

### Step 1: Supervised Fine-tuning (SFT)
- Pretrained model fine-tuned on human-selected responses
- Standard training (ZeRO + optional LoRA)
- Not the focus of OPD

### Step 2: Reward Model Fine-tuning
- Separate (usually smaller) model trained on human-ranked responses
- Uses pairwise ranking loss
- Standard training

### Step 3: RLHF Training (PPO)
- **This is where HybridEngine is used**
- Each iteration: inference phase (generation) + training phase (PPO update)
- HybridEngine seamlessly transitions between inference and training modes:
  - `model.eval()` -> inference containers activated, LoRA fused for generation
  - `model.train()` -> original forward restored, LoRA unfused for training

### DeepSpeed-Chat API (from blog):
```python
engine = DeepSpeedRLHFEngine(
  actor_model_name_or_path=args.actor_model_name_or_path,
  critic_model_name_or_path=args.critic_model_name_or_path,
  tokenizer=tokenizer,
  num_total_iters=num_total_iters,
  args=args)

trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
  out = trainer.generate_experience(prompt_batch)
  actor_loss, critic_loss = trainer.train_rlhf(out)
```

### How OPD fits into this pipeline:
- OPD replaces Step 3 (PPO) with distillation
- Instead of reward model + PPO, uses teacher model + KL divergence
- The HybridEngineRollout replaces DeepSpeed-Chat's rollout phase
- The student model uses ZeRO-2 + CPU_Adam (or ZeRO-0 for small models)
- The teacher model uses ZeRO-3 + CPU offload (entirely on CPU RAM)

---

## 5. HybridEngine + LoRA Interaction (Source-Level)

### DeepSpeedHybridEngine (`deepspeed/runtime/hybrid_engine.py`):

The HybridEngine inherits from DeepSpeedEngine and adds:
- Inference containers for accelerated generation
- LoRA fuse/unfuse cycle during eval/train transitions
- Performance metrics per phase (gather, generate, train)

### LoRA fuse/unfuse lifecycle:

```
Training step:
  1. model.eval()  -- transition to inference
     -> inference_containers[i].transform_for_inference()
     -> orig_module.forward = inference_container.module.forward
     -> fuse_lora_weight()  -- LoRA params added to base weights for generation

  2. model.generate() -- rollout using fused weights (single forward = base+LoRA)

  3. model.train()  -- transition back to training
     -> inference_container.transform_for_training()
     -> orig_module.forward = orig_fwd (original forward)
     -> unfuse_lora_weight()  -- LoRA params removed from base weights

  4. student_engine.forward() -- training with split forward (base + LoRA separate)
  5. student_engine.backward()
  6. student_engine.step()
```

### Key LoRA interaction code in HybridEngine.generate():

```python
# Z3 + pin_parameters path (lines 176-262):
if len(self.all_lora_params) > 0:
    self.fuse_lora_weight()     # Before generation
    # ... generation ...
    if len(self.all_lora_params) > 0:
        self.unfuse_lora_weight()  # After generation

# Z3 + non-pinned path (lines 146-153):
self.unfuse_lora_weight_non_pinned()  # GatheredParameters + unfuse
```

### HybridEngineContainer fuse/unfuse (`deepspeed/module_inject/containers/features/hybrid_engine.py`):

```python
def fuse_lora(self):
    for maybe_lora_param, param in self.get_lora_matched_pair():
        if len(maybe_lora_param) == 3:
            lora_right_weight, lora_left_weight, lora_scaling = maybe_lora_param
            param.data += lora_scaling * torch.matmul(lora_left_weight.t(), lora_right_weight.t())

def unfuse_lora(self):
    for maybe_lora_param, param in self.get_lora_matched_pair():
        if len(maybe_lora_param) == 3:
            lora_right_weight, lora_left_weight, lora_scaling = maybe_lora_param
            param.data -= lora_scaling * torch.matmul(lora_left_weight.t(), lora_right_weight.t())
```

**Important**: This is the OLD LoRA implementation (HybridEngineContainer-based, used in DeepSpeed-Chat).
The NEW LoRA implementation is `LoRAOptimizedLinear` in `deepspeed/linear/`, which has a different forward
path (`base_weight_output + lora_scaling_factor * lora_output`). These are TWO DIFFERENT LoRA systems.

---

## 6. TWO LoRA Systems in DeepSpeed -- Critical for OPD Integration

### System 1: HybridEngineContainer LoRA (OLD, DeepSpeed-Chat era)
- Location: `deepspeed/module_inject/containers/features/hybrid_engine.py`
- Mechanism: fuse/unfuse cycle (matmul LoRA into base before generation, subtract after)
- Used by: DeepSpeedHybridEngine's generate() method
- Limitation: Only works with models that have DeepSpeed inference policies (GPT2, OPT, BLOOM, LLAMA, etc.)
- Qwen2/3: Falls back to GatheredParameters + HF generate() (no fuse/unfuse for these models)
- Test: `tests/unit/hybrid_engine/test_he_lora.py` (uses convert_linear_layer_to_lora)

### System 2: LoRAOptimizedLinear (NEW, ZeRO-compatible)
- Location: `deepspeed/linear/optimized_linear.py` (223 lines)
- Config: `LoRAConfig` dataclass in `deepspeed/linear/config.py`
- Mechanism: Split forward (base_weight_output + lora_scaling * lora_output)
  - No fuse/unfuse needed -- LoRA is always separate
- Key features:
  - `base_weight_sharding`: Shards base weights across DP ranks for memory savings
  - `offload_ratio`: Offloads a fraction of frozen base weights to CPU
  - `ds_optim_param`: Marks base weights as "no optimizer needed"
  - `delay_lora_init`: Allows manual init after model construction
- Init: `deepspeed.linear.Init` context manager wraps AutoModelForCausalLM.from_pretrained
  - Replaces nn.Linear with LoRAOptimizedLinear during model construction
  - Automatically calls init_lora() after model loading
- Compatible with: ZeRO-2, CPU_Adam, offload_ratio
- NOT compatible with: HybridEngine's inference containers (different forward path)

### OPD must choose which LoRA system to integrate:

| Aspect | HybridEngine LoRA (System 1) | LoRAOptimizedLinear (System 2) |
|--------|-------------------------------|-------------------------------|
| Forward for rollout | fuse -> base+LoRA -> unfuse | base + LoRA split (always) |
| ZeRO-2 compatibility | Only ZeRO-2 with hybrid_engine | ZeRO-2 + CPU_Adam (standard) |
| CPU offload | No base weight offload | offload_ratio 0.0-1.0 |
| Memory savings | LoRA params only | LoRA params + base weight offload |
| Inference acceleration | DeepSpeed inference kernels | HF model.generate() |
| Qwen2/3 support | Fallback path (no acceleration) | Full support (standard HF) |
| RTX 4090 optimal | No (needs inference policy) | YES (offload_ratio + ZeRO-2) |

**Conclusion**: For OPD on RTX 4090, LoRAOptimizedLinear (System 2) is the correct choice.
HybridEngine LoRA (System 1) requires DeepSpeed inference containers that don't support Qwen2/3 well,
and doesn't offer base weight offloading. The OPD trainer should use HybridEngineRollout's standard
generate path (HF model.generate()) with LoRAOptimizedLinear providing the split forward.

---

## 7. The LoRA + OPD Gap -- Where the Integration Breaks

### Current state: Neither LoRA system is wired into the OPD trainer

The original PR #8027 draft loaded the student as:
```python
model = AutoModelForCausalLM.from_pretrained(cfg.student.model_name_or_path)
```

No LoRAConfig, no Init context manager, no LoRAOptimizedLinear. The student trains all parameters.

### The ~15 LOC gap (from prior analysis):

1. **Add LoRAConfig to StudentConfig** (opsd/config.py, ~3 lines):
```python
@dataclass
class StudentConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"
    arch: str = "qwen2"
    lora_config: Optional[LoRAConfig] = None  # NEW
```

2. **Use Init context manager** (main.py, ~5 lines):
```python
if cfg.student.lora_config is not None:
    with deepspeed.linear.Init(lora_config=cfg.student.lora_config):
        model = AutoModelForCausalLM.from_pretrained(...)
else:
    model = AutoModelForCausalLM.from_pretrained(...)
```

3. **ZeRO-2 + CPU_Adam config** (already works, no change needed):
   - LoRAOptimizedLinear is auto-detected by engine._optimized_linear_offload_setup()
   - ds_optim_param=True marks base weights for exclusion from optimizer
   - CPU_Adam only processes LoRA params (~13.5M for rank=32 on 0.5B, vs 614M for full model)

4. **Rollout with LoRA** (HybridEngineRollout, no change needed):
   - HybridEngineRollout.generate() calls module.generate() with the student model
   - With LoRAOptimizedLinear, the forward path is `base_weight_output + lora_scaling * lora_output`
   - This produces correct logits with LoRA applied -- no fuse/unfuse cycle needed

### But there's a subtlety: HybridEngine vs LoRAOptimizedLinear conflict

The old HybridEngine (DeepSpeedHybridEngine) expects HybridEngineContainer LoRA with fuse/unfuse.
If using LoRAOptimizedLinear instead, the HybridEngine's fuse_lora_weight() would fail because
LoRAOptimizedLinear doesn't have `lora_right_weight` / `lora_left_weight` attributes -- it has
`lora_weight_1` / `lora_weight_2`.

**Resolution**: The OPD trainer should NOT use DeepSpeedHybridEngine at all.
Instead, it should use HybridEngineRollout (the newly merged class), which simply calls
`module.generate()` on the HuggingFace model. With LoRAOptimizedLinear injected into the model,
the forward pass naturally includes LoRA contributions. No fuse/unfuse needed.

This is actually the simpler and more correct path. DeepSpeedHybridEngine was designed for the
DeepSpeed-Chat PPO pipeline where inference acceleration matters. For OPD on RTX 4090, the
generation phase is less time-critical (student is small), and the HuggingFace generate() path
is sufficient.

---

## 8. ZeRO-3 Frozen Params + LoRA -- Known Bugs Affecting OPD

### Bug 1: ZeRO-3 Frozen Params + Activation Checkpointing (#8130, OPEN)
- When using ZeRO-3 + gradient checkpointing (non-reentrant) + frozen params (LoRA):
  - Frozen params aren't detached during recompute
  - ZeRO-3 post-forward hook partitions params in-place: `param.data = torch.empty(0)`
  - Saved tensor shape [1024] vs recomputed shape [0] -> CheckpointError
- **Impact on OPD**: If student uses ZeRO-3 + LoRA + activation checkpointing -> crash
- **Workaround**: Use ZeRO-2 (recommended anyway for RTX 4090), or force `use_reentrant=True`
- **Fix PR**: #8130 (delock approved, tohtana requested changes, still OPEN)

### Bug 2: ZeRO-3 + LoRA dtype mismatch (#8072/#8073, still OPEN)
- Per-policy dtype (#8066, merged) caused ZeRO-3 partition dtype mismatch
- LoRA adapter params (bf16) vs base params (different dtype after partitioning)
- **Impact on OPD**: If student uses ZeRO-3 with LoRA -> potential dtype mismatch
- **Workaround**: Use ZeRO-2 (no partitioning, no dtype mismatch)

### Bug 3: overlap_comm + torch.compile = NaN (#8061, CLOSED but unfixed!)
- Multi-stream data race in gradient bucket copy_ operations
- **Impact on OPD**: If student uses ZeRO-2 + overlap_comm=True -> NaN risk
- **Workaround**: overlap_comm=False on single GPU (MUST for RTX 4090)

### Bug 4: gradient_clipping default was 0 (#8068, MERGED, now default=1.0)
- Previously gradient_clipping=0 meant no clipping -> exploding gradients
- **Impact on OPD**: Always set gradient_clipping=1.0 explicitly

### Safe OPD configuration for RTX 4090:
- ZeRO-2 (NOT ZeRO-3 -- avoids all three bugs above)
- CPU_Adam optimizer (offload optimizer to CPU)
- overlap_comm=False (avoid #8061 NaN)
- gradient_clipping=1.0 (always set explicitly)
- LoRAOptimizedLinear with offload_ratio=0.5 (50% base weights offloaded to CPU)

---

## 9. DeepSpeed Q3 Roadmap #8104 -- OPD Status

The roadmap (posted July 2026, 0 comments, 0 reviews) lists:

### OPD section:
```
## On-Policy Distillation Trainer Support (Q3)
- [ ] Systems Design, prototyping and benchmarking (reference: https://arxiv.org/abs/2604.14084)
```

This confirms:
1. OPD is a Q3 target (July-September 2026)
2. The merged PR #8027 was just the rollout infrastructure
3. The actual trainer (3-phase loop, teacher wrapper, distillation loss) is the next milestone
4. Reference paper: TIP (arxiv 2604.14084) -- Token Importance in On-Policy Distillation

### Other Q3 roadmap items affecting OPD:

**AutoEP + AutoTP Folding** (same PR author PKUWZP):
- AutoEP + AutoTP combined, benchmarking TP in attention
- Unified EP kernels for MoE
- Liger Kernel integration (LM head sharding, online softmax)

**DeepCompile** (pass contracts):
- Formal pass contracts and validation
- Composable AutoTP/AutoEP/SP at compiler level
- Activation offloading pass

**Pipeline parallelism with Ray**:
- Ray-backed pipeline stage placement
- Heterogeneous resource allocation

**Tuning guide**:
- Recommended configs for representative models
- Benchmark-backed guidance

**Stability**:
- Performance regression tests
- Nightly full tests (CUDA, AMD, XPU, Gaudi, NPU)

### What's NOT in the roadmap:
- LoRA + OPD integration
- RTX 4090/consumer GPU configs
- LoRAOptimizedLinear + OPD wiring
- Teacher vLLM/SGLang server support

**This is the gap**: The Q3 roadmap targets OPD trainer completion but does NOT mention LoRA
integration. The ~15 LOC gap identified in our prior analysis is not on the roadmap. This
represents a clear contribution opportunity.

---

## 10. OPD Training Data Flow -- Full Pipeline Trace

### With LoRAOptimizedLinear (recommended RTX 4090 path):

```
Step N:
  Phase 0 (Rollout):
    DataLoader -> PromptDataset -> prompts [B, T_p] left-padded
    prompts -> HybridEngineRollout.generate() -> RolloutBatch
    Inside generate():
      module.forward(prompt_ids, attention_mask, ...) with LoRA:
        LoRAOptimizedLinear.forward(input_tensor):
          base_weight_output = F.linear(input_tensor, base_weight)  # base on GPU (or offloaded)
          lora_output = lora_weight_2(lora_weight_1(input_tensor))  # LoRA adapter
          return base_weight_output + lora_scaling * lora_output
      logits -> sampling -> generated tokens
    Output: RolloutBatch { input_ids [B, T_p+T_r], attention_mask, response_start_idx }

  Phase 1 (Teacher Forward + Cache):
    input_ids, attention_mask -> TeacherWrapper.forward_to_cache()
    Teacher on CPU (ZeRO-3 offload):
      ZeRO-3 gathers teacher params to GPU temporarily
      teacher.forward(input_ids, attention_mask) -> teacher_logits [B, T, V] on GPU
      teacher_logits.detach().to(bf16).cpu() -> TeacherLogitCache on CPU (pinned memory)
      ZeRO-3 releases teacher params back to CPU
    Output: TeacherLogitCache { cpu_logits [B, T, V] on CPU host }

  Phase 2 (Student Training):
    student_engine.train()
    student_engine.forward(input_ids, attention_mask) with LoRA:
      LoRAOptimizedLinear.forward(input_tensor):
        base_weight_output = F.linear(input_tensor, base_weight)
        lora_output = lora_weight_2(lora_weight_1(input_tensor))
        return base_weight_output + lora_scaling * lora_output
    student_logits [B, T, V] on GPU

    Shift for next-token prediction:
      student_logits_shifted = student_logits[:, :-1, :]
      mask_shifted = response_mask[:, 1:]

    Streamed distillation loss:
      for each chunk [start:end] along sequence axis:
        teacher_chunk = teacher_cache.chunk_to_device(start, end, GPU, bf16)  # CPU -> GPU
        per_tok = KL_divergence(student_logits[:, start:end], teacher_chunk, temperature)
        total_loss += (per_tok * chunk_mask).sum()
      loss = total_loss / total_tokens

    student_engine.backward(loss) -> gradients flow to LoRA params only (base frozen)
    student_engine.step() -> CPU_Adam updates LoRA params (13.5M, not 614M)
    teacher_cache.free() -> CPU buffer released
```

### Memory lifecycle with LoRA (RTX 4090, Qwen2.5-0.5B student, rank=32):

| Phase | GPU Contents | Peak GPU | CPU Contents |
|-------|-------------|----------|--------------|
| Phase 0 | Student base (~0.6GB with offload_ratio=0.5) + LoRA (~0.03GB) + KV cache | ~2.5GB | Offloaded base (~0.6GB) |
| Phase 1 | Student (~0.63GB) + Teacher params (transient, ~1.4GB per layer) + Teacher logits (transient) | ~4GB | Teacher model (~14GB) + TeacherLogitCache (~2.5GB) |
| Phase 2 | Student (~0.63GB) + Student logits (~0.3-2.5GB) + One teacher chunk (~0.15-1.2GB) + Activations (~0.5GB) + Gradients (~0.03GB) | ~4.5GB | CPU_Adam optimizer (~0.16GB) + TeacherLogitCache (~2.5GB) |

**Peak GPU: ~4.5GB** (well under 24GB RTX 4090)
**Total CPU: ~17.3GB** (fits on 16+ GB RAM systems, unlike 26.5GB without LoRA)

---

## 11. HybridEngineRollout + LoRA Integration -- Technical Details

### How HybridEngineRollout.generate() works with LoRAOptimizedLinear:

The merged HybridEngineRollout calls `module.generate()` on the student engine's module.
With LoRAOptimizedLinear injected into the model (via Init context manager), each Linear layer
becomes a LoRAOptimizedLinear whose forward naturally adds LoRA contributions.

```python
# HybridEngineRollout.generate() (merged code):
output_ids = module.generate(
    prompt_ids,
    attention_mask=prompt_attn,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature if do_sample else 1.0,
    top_p=sampling.top_p if do_sample else 1.0,
    pad_token_id=pad_token_id,
)
```

This is compatible with LoRAOptimizedLinear because:
1. LoRAOptimizedLinear.forward() produces `base + lora_scaling * lora_output`
2. HuggingFace generate() calls forward() repeatedly for each token
3. Each forward naturally includes LoRA contributions
4. No fuse/unfuse cycle needed (LoRA is always active in the forward)

### The CUDA graph capture path with LoRA:

The merged code also includes a `_generate_graph` method for greedy (temperature=0) generation
with CUDA graph capture. This path:
1. Removes forward hooks (ZeRO hooks synchronize -> illegal during graph capture)
2. Uses DeepSpeedStaticCache for KV cache management
3. Captures the decode forward pass as a CUDA graph
4. Replays the graph for each decode step

**LoRA compatibility concern**: CUDA graph capture "freezes" the forward pass.
LoRA adapter weights are modified by the optimizer during training, so the captured
graph would use stale LoRA weights. For OPD, this means:
- Graph capture should be used ONLY within a single step's rollout phase
- After training step (optimizer updates LoRA), the graph must be recaptured
- For temperature > 0 (sampling mode), graph capture is not used anyway

**Recommendation**: For OPD on RTX 4090, use the standard generate() path (temperature > 0).
Graph capture adds complexity without significant benefit for small student models (0.5B).

---

## 12. OPD vs DeepSpeed-Chat PPO -- Architecture Comparison

| Aspect | DeepSpeed-Chat PPO (Step 3) | OPD (future) |
|--------|-------------------------------|---------------|
| Training signal | Reward model score + PPO advantage | Teacher KL/JSD divergence |
| Models needed | Actor + Critic + Reward + Reference | Student + Teacher (2 models) |
| Teacher/reward location | Reward model on GPU or separate server | Teacher on CPU (ZeRO-3 offload) |
| Generation backend | DeepSpeedHybridEngine (inference containers) | HybridEngineRollout (HF generate) |
| LoRA support | HybridEngineContainer fuse/unfuse | LoRAOptimizedLinear split forward |
| Memory (RTX 4090) | ~18GB with bypass, ~35GB full | ~4.5GB with LoRA |
| Quality ceiling | Beyond teacher (if reward allows) | Bounded by teacher |
| DeepSpeed engine | DeepSpeedHybridEngine | Standard DeepSpeedEngine |

### Key difference in HybridEngine usage:

DeepSpeed-Chat PPO uses DeepSpeedHybridEngine (which inherits from DeepSpeedEngine and adds
inference containers). This engine manages the eval/train transitions with LoRA fuse/unfuse.

OPD should NOT use DeepSpeedHybridEngine. Instead:
- Student: Standard DeepSpeedEngine (ZeRO-2 + CPU_Adam + LoRAOptimizedLinear)
- Rollout: HybridEngineRollout (calls module.generate() directly)
- Teacher: ZeRO-3 engine with CPU offload (or plain model on CPU for small teacher)

This is simpler, more compatible with modern models (Qwen2/3), and works with LoRAOptimizedLinear.

---

## 13. Cross-Framework OPD Status

| Framework | OPD Support | LoRA+OPD | RTX 4090 viability |
|-----------|------------|----------|--------------------|
| DeepSpeed | Rollout infrastructure merged; trainer pending | ~15 LOC gap | BEST (ZeRO-2 + CPU_Adam + LoRA offload) |
| verl VeOmni | Full OPD (merged in v0.22+) | LoRA wired (FSDP2 summon) | FSDP2 required, single GPU limited |
| TRL | No OPD trainer | Standard KD + PEFT | No CPU offload optimization |
| rLLM Tinker | No distillation module | GRPO only (LoRA support coming) | In-process, efficient, but no distillation |
| Megatron | No distillation trainer | No LoRA | No path |

### DeepSpeed's competitive advantage for consumer GPU distillation:

1. **ZeRO-3 CPU offload for teacher**: Unique -- no other framework offers this
2. **LoRAOptimizedLinear + offload_ratio**: Base weight partial CPU offload -- unique
3. **TeacherLogitCache**: Host-resident chunk fetch -- shared with verl concept but implementation differs
4. **Streamed distillation loss**: Sequence-axis chunking -- verl uses top-K sparse instead
5. **HybridEngineRollout**: In-process generation, no weight sync needed -- simpler than verl's separate server

The ~15 LOC LoRA integration would make DeepSpeed the ONLY framework with LoRA + CPU-offloaded
distillation on consumer GPUs.

---

## 14. Implementation Roadmap

### Phase 1: OPD Trainer Completion (DeepSpeed team, Q3 2026)
- Implement OPSDTrainer (3-phase loop) as follow-up to merged PR #8027
- Implement TeacherWrapper + TeacherLogitCache
- Implement streamed_distillation_loss (forward-KL, reverse-KL, JSD)
- Implement OPSDConfig dataclass
- Validate on 2xH200 (following original PR's test plan)

### Phase 2: LoRA + OPD Integration (~15 LOC, contribution opportunity)
- Add LoRAConfig to StudentConfig
- Use Init context manager for student loading
- Validate on RTX 4090 (Qwen2.5-0.5B student + LoRA rank=32)
- Profile memory: expect ~4.5GB GPU + ~17.3GB CPU

### Phase 3: RTX 4090 Tuning Guide (follows Q3 roadmap "Tuning guide" item)
- Recommended configs for OPD on consumer GPUs
- Benchmark-backed guidance for student/teacher model size combinations
- offload_ratio sweep (0.0, 0.5, 1.0) for LoRA base weight offloading

### Phase 4: Extended Features
- vLLM/SGLang rollout backend (for multi-GPU setups)
- Reward-weighted distillation (OPSD reward_beta knob)
- Multi-teacher distillation
- Qwen3-MoE student support (needs WeightBridge + MoE slicing)

---

## 15. Key Findings Summary

1. **PR #8027 merged July 12, but only rollout infrastructure**: 6 files (RolloutEngine ABC,
   HybridEngineRollout, DeepSpeedStaticCache). The actual OPD trainer is a follow-up.

2. **Two separate LoRA systems**: HybridEngineContainer (old, fuse/unfuse, for DeepSpeed-Chat)
   and LoRAOptimizedLinear (new, split forward, for ZeRO + modern models). OPD should use
   LoRAOptimizedLinear, NOT the HybridEngineContainer system.

3. **OPD should NOT use DeepSpeedHybridEngine**: The old HybridEngine (with inference containers
   and fuse/unfuse) is designed for PPO. OPD should use standard DeepSpeedEngine (ZeRO-2 + CPU_Adam)
   + HybridEngineRollout (the newly merged class that calls module.generate() directly).

4. **The ~15 LOC LoRA gap remains**: LoRAConfig in StudentConfig + Init context manager for
   student loading. This is NOT on the Q3 roadmap -- a clear contribution opportunity.

5. **ZeRO-2 is safe for LoRA + OPD on RTX 4090**: Avoids all three known LoRA+ZeRO bugs
   (#8130 checkpoint error, #8072 dtype mismatch, #8061 NaN). ZeRO-3 should NOT be used for
   the student on single GPU.

6. **Memory budget with LoRA**: ~4.5GB GPU peak (vs ~7GB without LoRA), ~17.3GB CPU
   (vs ~26.5GB without LoRA). The 60x optimizer reduction (0.16GB vs 9.84GB) is the key win.

7. **Q3 roadmap #8104**: Confirms OPD as a Q3 target but does NOT mention LoRA integration.
   Rollout infrastructure is done; trainer implementation is the next milestone.

8. **CUDA graph capture**: Available in merged HybridEngineRollout for greedy (temperature=0)
   generation. LoRA weights would stale during training -> recapture needed per step. For OPD
   with temperature > 0, standard generate() is sufficient and simpler.

---

## 16. Source File References

### Merged files (in deepspeedai/DeepSpeed master):
| File | Role |
|------|------|
| `deepspeed/runtime/rollout/base.py` | RolloutEngine ABC + dataclasses |
| `deepspeed/runtime/rollout/hybrid_engine_rollout.py` | HybridEngineRollout impl |
| `deepspeed/runtime/rollout/__init__.py` | build_rollout factory |
| `deepspeed/utils/static_cache.py` | DeepSpeedStaticCache |

### Existing LoRA infrastructure (in /tmp/deepspeed-fork/):
| File | Role |
|------|------|
| `deepspeed/linear/optimized_linear.py` | LoRAOptimizedLinear (223 lines) |
| `deepspeed/linear/config.py` | LoRAConfig dataclass |
| `deepspeed/linear/context_manager.py` | Init context manager |
| `deepspeed/runtime/engine.py:377-413` | _optimized_linear_offload_setup |
| `deepspeed/runtime/hybrid_engine.py` | DeepSpeedHybridEngine (OLD, for PPO) |
| `deepspeed/module_inject/containers/features/hybrid_engine.py` | HybridEngineContainer LoRA (OLD) |

### Known bug references:
| Issue | Bug | Impact on LoRA+OPD |
|-------|-----|---------------------|
| #8130 | ZeRO-3 frozen params + checkpoint error | Use ZeRO-2 instead |
| #8072/#8073 | ZeRO-3 dtype mismatch with PEFT | Use ZeRO-2 instead |
| #8061 | overlap_comm + compile NaN | overlap_comm=False on single GPU |
| #8068 | gradient_clipping default 0 | Always set 1.0 explicitly |

### Prior deep readings:
| File | Content |
|------|---------|
| `notebook/projects/deepspeed-opd-trainer-source-reading.md` | Full OPD trainer architecture (draft PR version) |
| `notebook/projects/deepspeed-opd-lora-integration-gap-analysis.md` | ~15 LOC gap + memory budget |
| `notebook/projects/deepspeed-autoep-gap-analysis.md` | 9 AutoEP gaps (OPD not listed there) |
| `notebook/deepspeed-8130-zero3-frozen-params-checkpoint-deep-reading.md` | ZeRO-3 checkpoint error |

---

## 17. References

- OPD PR #8027: deepspeedai/DeepSpeed, MERGED July 12 (6 files only)
- Q3 roadmap: deepspeedai/DeepSpeed #8104
- TIP paper: arxiv:2604.14084 (Token Importance in On-Policy Distillation)
- PACED paper: arxiv:2603.11178 (Distillation at the Frontier of Student Competence)
- Beyond GRPO: arxiv:2605.12483 (RL teacher + distillation)
- LoRAOptimizedLinear: `deepspeed/linear/` (merged in main)
- HybridEngine: `deepspeed/runtime/hybrid_engine.py` (for PPO, NOT for OPD)
- DeepSpeed-Chat: microsoft/DeepSpeedExamples (PPO pipeline, 3-step InstructGPT)
