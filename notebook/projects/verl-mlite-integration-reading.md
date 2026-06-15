# verl + Megatron Lite Integration — mlite_engine.py深度分析

> 2026-06-16 | Megatron-LM PR #4885 (dev) | verl GRPO集成 | mlite_engine.py 779行
> ★★★★★ Bitwise correctness verified (loss=0.0, grad=0.0) → 最高正确性标准!
> ★★★★★ -20.3% memory, ~7% faster → but only MoE models, RTX 4090短期不可行

## 1. ★★★★★ mlite_engine.py架构概要

```
★★★★★ Registration (line 112):
  @EngineRegistry.register(model_type="language_model", backend="mlite", device="cuda")
  class MegatronLiteEngine(BaseEngine):

★★★★★★★ 只注册language_model → 没有value_model → PPO不可用!
  → PPO needs both policy (language_model) + critic (value_model)
  → mlite only has language_model → PPO = ✗
  → GRPO = ✓ (no critic needed) → 与GRPO script一致: critic.enable=False

★★★★★ Core initialization flow:
  1. _build_mlite_config() → MegatronLiteConfig from verl-side config
  2. create_runtime(RuntimeConfig(backend="mlite", hf_path=...))
  3. runtime.build_model() → handle (model + optimizer + parallel state)
  4. _extract_primary_module() → unwrap model from handle
  5. _build_lr_scheduler() → OptimizerParamScheduler if needed
  6. self.to(device="cpu") → if offload enabled, immediately move to CPU

★★★★★ Compile cache isolation (_isolate_compile_cache_per_rank):
  → Append rank_{local_rank} to TORCHINDUCTOR_CACHE_DIR + TRITON_CACHE_DIR
  → Prevent Triton/Inductor cache races in multi-process torchrun
```

## 2. ★★★★★ verl GRPO集成配置

```
★★★★★★★ GRPO launch script: run_qwen3moe_gsm8k_grpo.sh

Algorithm config:
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.rollout_correction.bypass_mode=True
  algorithm.norm_adv_by_std_in_grpo=False

Actor config:
  actor@actor_rollout_ref.actor=mlite_actor
  actor_rollout_ref.actor.engine.impl=lite
  actor_rollout_ref.actor.engine.ep=8
  actor_rollout_ref.actor.engine.tp=2
  actor_rollout_ref.actor.engine.pp=1
  actor_rollout_ref.actor.engine.cp=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm

Critic:
  critic.enable=False → GRPO模式 (no value model)

Rollout (vLLM V1):
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6

★★★★★★★ KL全部禁用 = 语义等同bypass_mode:
  → use_kl_in_reward=False → reward不包含KL
  → kl_coef=0.0 → KL系数为零
  → use_kl_loss=False → actor不包含KL loss
  → kl_loss_coef=0.0 → KL loss系数为零
  → bypass_mode=True → rollout correction跳过ref model
  → ★★★★★ 4个Hydra override + bypass_mode → 全部KL禁用 → 等同bypass_mode!
```

## 3. ★★★★★ THD/no-padding强制要求

```
★★★★★★★ THD = "total-tokens × heads × dim" attention format
  → Packed sequence representation → no padding waste
  → Especially important for MoE → different tokens → different experts

★★★★★★★ 4处强制检查:

1. pad_mode check (forward_backward_batch, line 206-209):
   if pad_mode != DatasetPadMode.NO_PADDING:
       raise NotImplementedError("MegatronLiteEngine only supports pad_mode=no_padding")

2. use_thd forced True (_build_impl_cfg, line 472-476):
   if impl_cfg.get("use_thd", True) is not True:
       raise ValueError("set engine.impl_cfg.use_thd=True")
   impl_cfg["use_thd"] = True

3. NestedTensor requirement (_make_model_inputs, line 629-632):
   if not getattr(input_ids, "is_nested", False):
       raise NotImplementedError("supports only nested no-padding THD batches")

4. Default config (mlite.yaml):
   impl_cfg: use_thd: true

★★★★★★★ THD primitive (primitive/parallel/thd.py, ~300 lines):
  → PackedTHDBatch dataclass: packed input_ids, labels, loss_mask, position_ids, cu_seqlens, packed_seq_params
  → pack_nested_thd(): verl jagged NestedTensor → Megatron packed format
  → TP alignment: align_size = max(tp_size, 1) * (2*cp_size if cp_size>1 else 1)
  → CP zigzag chunking: ring attention across CP ranks
  → roll_packed_thd_left(): next-token-prediction labels without crossing seq boundaries
  → unpack_packed_thd_to_nested(): reverse for model output (log_probs, entropy)
```

## 4. ★★★★★ Offload机制详解

```
★★★★★★★ optimizer_offload=True (默认):

  → offload_fraction=1.0 by default (lines 497-503):
    if offload_fraction is None and self.is_optimizer_offload_enabled:
        offload_fraction = 1.0

  → Context switching:
    train_mode → optimizer state moves to GPU
    exit train → optimizer state moves back to CPU

  → Initialization offload (line 168-173):
    After build_model → if offload enabled → immediately move to CPU
    self.to(device="cpu", model=self.is_param_offload_enabled,
            optimizer=self.is_optimizer_offload_enabled, grad=self.is_param_offload_enabled)

  → GRPO script defaults:
    PARAM_OFFLOAD=False (default)
    OPTIMIZER_OFFLOAD=True (default)
    OPTIMIZER_STATE_OFFLOAD_FRACTION=1.0

★★★★★★★ FSDP2 optimizer backend:
  → MLITE_OPTIMIZER_BACKEND=fsdp2 → impl_cfg.optimizer=fsdp2
  → offload_fraction=1.0 → FSDP2 optimizer keeps update state on CPU
  → Reduces GPU memory → but slows training significantly

★★★★★★★ param_offload=False (默认):
  → When enabled → params move to GPU during train/eval → back to CPU when idle
  → Checkpoint save/load → move params to CUDA temporarily → then back to CPU
```

## 5. ★★★★★ No value_model → PPO不可用

```
★★★★★★★ mlite只注册language_model:
  @EngineRegistry.register(model_type="language_model", backend="mlite", device="cuda")

★★★★ 其他backend对比:
  → megatron: registers language_model + value_model → PPO可用
  → fsdp/fsdp2: registers language_model + value_model → PPO可用
  → veomni: registers both → PPO可用
  → mlite: ONLY language_model → PPO ✗ → GRPO ✓ only!

★★★★★★★ Why intentional:
  → README: "scope to exercise current MLite actor path without expanding scope"
  → value_model → same Lite runtime → but needs value head variant
  → Current Lite model protocols (Qwen3/3.5 MoE) → no value head → need new protocol

★★★★★★★ RTX 4090影响:
  → PPO on RTX 4090 → ✗✗✗ (needs critic → 2x memory → OOM)
  → GRPO on RTX 4090 → ✓ (no critic → save memory → feasible)
  → ★★★★★ GRPO-only = RTX 4090友好 → 不需要PPO!
```

## 6. ★★★★★ No LoRA in Current Integration

```
★★★★★★★ mlite_engine.py has NO LoRA-related code:
  → No LoRA config, no adapter loading, no freeze_non_lora_params
  → BaseEngine.disable_adapter() returns nullcontext() → mlite inherits this

★★★★★★★ Megatron Lite side has LoRA primitives:
  → primitive/modules/lora.py (21,707 bytes) → LinearLoRA, GroupedLinearLoRA
  → model/qwen3_moe/lite/lora_adapter.py → PEFT LoRA adapter
  → BUT → NOT wired into verl integration at all!

★★★★★★★ Why not wired:
  → LoRA crosses multiple layers:
    1. Model protocol build_model → need LoRA adapter injection
    2. Optimizer → freeze_non_lora_params
    3. Checkpoint → LoRA adapter import/export
    4. Weight sync → actor→rollout LoRA update path
  → Same reason as "excluded" in initial PR → too many layers for first integration
  → ★★★★★ 需要dedicated follow-up PR → 中期目标

★★★★★★★ RTX 4090影响:
  → Without LoRA → full model weights → optimizer state for ALL params → OOM!
  → Qwen3.5-35B-A3B → 35B params → >70GB → ✗✗✗ not feasible on 24GB!
  → ★★★★★ LoRA是RTX 4090使用Lite的关键前提!
  → → 当前integration → RTX 4090 ✗✗✗ → 需要LoRA + small model protocol
```

## 7. ★★★★★ Only MoE Models — 为什么?

```
★★★★★★★ Model registry (model/registry.py):
  Only 3 registrations:
  → qwen3 → qwen3_moe (HF: qwen3_moe, qwen2_moe)
  → qwen3_moe → same package
  → qwen3_5 → qwen3_5_moe (HF: qwen3_5_moe)

★★★★★★★ No dense models:
  → No qwen3 dense (HF type: qwen3)
  → No llama, mistral, or any other dense model

★★★★★★★ Why MoE only:
  1. MoE = hardest integration surface → proving correctness validates runtime
  2. MoE has most complex distributed requirements (EP, expert TP, token dispatch)
  3. THD/no-padding particularly important for MoE → different tokens → different experts
  4. Qwen3/3.5 MoE = current generation of open-source MoE → most relevant
  5. Dense models easier to add → once runtime validated on MoE

★★★★★★★ Auto-resolution:
  → model_name="auto" → resolve_model_type_from_hf() → read HF config.json model_type
  → If not in _HF_MODEL_TYPE_MAP → ValueError → can't use Qwen3-1.7B dense!

★★★★★★★ RTX 4090影响:
  → Only MoE models → Qwen3-30B-A3B → ~60GB → ✗✗✗ way too big!
  → Need small model protocol → Qwen3-1.7B dense → community contribution
  → ★★★★★ RTX 4090 GRPO: 等small model + LoRA → 中期可行
```

## 8. ★★★★★ Bitwise Correctness Verification

```
★★★★★★★ 1x GPU Deterministic Correctness (Slurm job 12630675):
  → Setup: 1x GPU, seed=42, Qwen3.5 MoE, seq_len=8, truncate_layers=1
  → Environment: MEGATRON_LITE_DETERMINISTIC=1, CUBLAS_WORKSPACE_CONFIG=:4096:8
  → Comparison: mlite vs mbridge (validated reference)

★★★★★★★ Results:
  → max_loss_abs = 0.0 (BITWISE IDENTICAL!)
  → max_grad_norm_abs = 0.0 (BITWISE IDENTICAL!)
  → mismatches = [] (empty → no mismatches)
  → Step 0 loss = 13.027458190917969
  → Step 0 grad_norm = 120.75512734973202
  → Step 0 weight SHA256 = 1e3176a8...
  → Step 1 loss = 14.698704719543457
  → Eval logits bf16 SHA256 = 2f805802...

★★★★★★★ 8x H100 Performance Comparison:
  → Same data across DP (SAME_DATA_ACROSS_DP=1) → fair comparison
  → Loss matched within atol=0.05, rtol=0.005 → max_abs_diff=0.000500

★★★★★★★ Why this matters:
  → Bitwise identical = highest correctness standard
  → Much stricter than vLLM/verl CI tests (tolerance-based)
  → ★★★★★ MLite runtime = production-grade correctness!

★★★★★★★ Note for verl GRPO path:
  → No bitwise verification published for verl GRPO path
  → Correctness only for pretraining-style forward+backward+optimizer_step
  → GRPO adds policy loss → needs separate verification
```

## 9. ★★★★★ Memory & Performance Advantages

```
★★★★★★★ 8x H100 Benchmark (Qwen3.5-35B-A3B, truncated: 8 layers, 8 experts):

| Runtime | Impl | Optimizer | Step ms | Tok/s | Tok/s/GPU | Peak Mem GB | TFLOPs/GPU |
|---------|------|-----------|---------|-------|-----------|-------------|------------|
| mlite | lite | distopt | 309.4 | 105,897 | 13,237 | 14.324 | 80.444 |
| mbridge | bridge | distopt | 332.2 | 98,639 | 12,330 | 17.987 | 74.931 |

★★★★★★★ MLite advantages:
  → Step speed: -7.0% (309 vs 332 ms)
  → Token throughput: +7.3% (105,897 vs 98,639)
  → GPU throughput: +7.6% (13,237 vs 12,330)
  → Peak memory: -20.3% (14.324 vs 17.987 GB, saves 3.663 GB)
  → TFLOPs: +7.3% (80.444 vs 74.931)

★★★★★★★ Memory savings analysis:
  → 3.663 GB saved per GPU (on 8×H100 with EP=8, TP=2)
  → Likely from: (1) no mbridge wrapper overhead, (2) THD avoids padding memory,
    (3) efficient MoE token dispatch, (4) PackedSeqParams avoids redundant metadata
  → ★★★★★ Proportional savings significant → but absolute model sizes still too large for 24GB

★★★★★★★ Performance analysis:
  → ~7% throughput improvement from: fewer wrapper layers, native THD format,
    more efficient attention with qkv_format="thd"
  → 80.444 TFLOPs → closer to H100 theoretical peak → better compute utilization
  → ★★★★★ Truncated model (8/36 layers, 8/128 experts) → savings may not scale to full model
```

## 10. ★★★★★ Forward/Backward实现

```
★★★★★★★ Two forward_backward_batch paths:

Path A: PP=1 (manual micro-batch loop, lines 199-292):
  → Iterate micro-batches
  → self.module() directly → accumulate loss/metrics
  → _make_model_inputs() → THD/no-padding format required
  → Handles mtp_loss as extra metric when MTP enabled
  → finalize_grads() hook from handle._extras

Path B: PP>1 (runtime-delegated):
  → _use_runtime_forward_backward() → checks handle._parallel_state.pp_size > 1
  → self.runtime.forward_backward() with custom loss function wrapper

★★★★★★★ Weight export (get_per_tensor_param, lines 294-307):
  → If param_offload → first move model to CUDA
  → model_name="qwen3_5" → export_kwargs["target"]="vllm" → vLLM-specific format
  → runtime.export_weights(handle) with optional dtype and target overrides

★★★★★★★ Checkpoint (lines 342-430):
  → save_training_checkpoint / load_training_checkpoint from megatron.lite.primitive.ckpt
  → placement_fn + expert_classifier from handle._extras["protocol"]
  → If offload → move to CUDA before save → back to CPU after
  → LR scheduler state saved separately as lr_scheduler.pt
  → Supports partial saves via checkpoint_config.save_contents
```

## 11. ★★★★★ Context Switching机制

```
★★★★★★★ _MegatronLiteModeCtx (lines 88-109):
  → Wraps BaseEngineCtx with Megatron Lite runtime context managers
  → On __enter__: runtime.train_mode(handle) or runtime.eval_mode(handle)
  → BaseEngineCtx._context_switch():
    → eval_mode → only offload model params to CPU
    → train_mode → offload model + optimizer + grad buffers to CPU
  → ★★★★★ Offload-aware context switching → critical for RTX 4090 memory!

★★★★★★★ Qwen3.5-specific vLLM export:
  → model_name="qwen3_5" → target="vllm" → vLLM compatibility format
  → ★★★★★★ MoE weight sync needs vLLM-specific format → weight export critical path
```

## 12. ★★★★★ RTX 4090可行性评估

```
★★★★★★★ RTX 4090可行性时间轴:

短期 (当前): ✗✗✗ NOT feasible
  → Only MoE models → Qwen3-30B-A3B → 60GB → way too big for 24GB
  → No LoRA in integration → full model weights → OOM
  → No small model protocol → can't use Qwen3-1.7B dense
  → No CPPO → only GRPO vanilla

中期 (需3个贡献): ★★★★ 可能可行
  → 1. Small model protocol → add Qwen3-1.7B dense to registry
  → 2. LoRA integration → wire LinearLoRA into mlite_engine
  → 3. INT8 quantization → MoE params quant → reduce VRAM
  → → After these → Qwen3-1.7B dense + LoRA-32 → ~8-10GB → RTX 4090 feasible!

长期: ★★★★★ 非常可行
  → Lite runtime mature → LoRA + small model standard
  → FSDP2 optimizer → param_offload + optimizer_offload → RTX 4090 optimized
  → Bitwise correctness → production-grade → can trust results
  → verl GRPO integration → mature → CPPO added → best trust region

★★★★★★★ RTX 4090 GRPO排名更新:
  → rLLM Tinker #1 (立即可用, in-process, bypass default)
  → verl + CPPO #2 (立即可用, 但需enforce_eager + MRv2=0)
  → verl + Lite #2.5 (中期可用, 需3个贡献)
  → Megatron core #3 (不可行, 单GPU crash + 无LoRA)

★★★★★★★ RTX 4090 Lite contribution path:
  → 最有价值贡献: small model protocol (Qwen3-1.7B dense)
  → 第二有价值: LoRA integration wiring (LinearLoRA → mlite_engine)
  → 第三有价值: INT8/FP8 quant support for MoE params
  → → ★★★★★ 这3个贡献 → 使Lite成为RTX 4090第三条可行GRPO路径!
```

## 参考
- Megatron-LM dev branch: experimental/lite/examples/verl/ (integration code)
- mlite_engine.py: 779 lines, @EngineRegistry.register("language_model", "mlite", "cuda")
- primitive/parallel/thd.py: THD pack/unpack primitives (~300 lines)
- primitive/modules/lora.py: LoRA primitives (21,707 bytes, NOT wired)
- model/registry.py: only 3 MoE model registrations
- run_qwen3moe_gsm8k_grpo.sh: GRPO launch script
- REQUIRED_VERL.txt: verl v0.8.0 requirement
- Bench README: -20.3% memory, ~7% faster, bitwise correctness verified
- 相关笔记: megatron-lite-reading.md (797行, LoRA correction), rtx4090-grpo-trust-region-comparison.md
