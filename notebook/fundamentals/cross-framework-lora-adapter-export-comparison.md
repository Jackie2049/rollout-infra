# Cross-Framework LoRA Adapter Export Comparison — Training→Inference Bridge

> 2026-06-17 | How each framework exports LoRA adapters from training to inference serving
> ★★★★★★★★ RTX 4090 deployment requires smooth training→serving bridge → LoRA hot-swap = key advantage

---

## 1. The LoRA Lifecycle Problem

```
★★★★★★★★★ LoRA training → inference serving = 6-stage lifecycle:

Stage 1: Train LoRA → adapter weights (PEFT, LoRA rank r=8-32)
Stage 2: Export adapter → portable format (HF safetensors, Megatron format)
Stage 3: Merge adapter → full model weights (optional, zero inference overhead)
Stage 4: Quantize → INT4/AWQ/GPTQ (optional, 3x compression)
Stage 5: Serve → vLLM/SGLang inference (LoRA hot-swap or merged)
Stage 6: Continuous → re-train LoRA on new data → re-deploy or hot-swap

★★★★★★★★★ KEY QUESTION: How does each framework handle Stage 2 (export)?
  → This is the training→inference bridge → the "last mile" gap
  → Different frameworks have DIFFERENT adapter formats → incompatibility!
```

---

## 2. Framework-by-Framework LoRA Export Comparison

### DeepSpeed

```
★★★★ DeepSpeed LoRA Export:
  → DeepSpeed trains LoRA via ZeRO-2/3 + CPU_Adam
  → Export: standard PEFT merge_and_unload() → HF safetensors format
  → ★★★ No special DeepSpeed-specific format → PEFT standard → portable!
  → MoE: AutoEP trains LoRA on active params → experts frozen → LoRA on attention/mlp
  → → Export: PEFT merge → but MoE structure preserved → careful with expert LoRA merging
  → ★★★★★ OPD (#8027): student LoRA + CPU teacher → export student LoRA → standard PEFT
  → ★★★ Gap: DeepSpeed doesn't have RouterReplay → MoE LoRA export doesn't handle routing determinism
```

### Megatron-LM

```
★★★★★★★★★ Megatron LoRA Export = CRITICAL COVERAGE GAP (verl #6713):

verl PR #6713 (DRAFT, 2026-06-12): "Megatron LoRA Adapter Export for Rollout"
  → ★★★★★★★★ CRITICAL CORRECTION: This is a **verl** PR, NOT a Megatron-LM PR!
  → Purpose: Export LoRA adapter-only tensors from verl Megatron backend → vLLM format
  → ★★★★★★★★ Handles MoE LoRA: gather EP-local LoRA → rewrite local→global expert IDs
  → Pack Qwen3-Omni 3D MoE LoRA into vLLM-compatible layout
  → ★★★★★★★★ 3-repo coordinated: verl #6713 + verl-omni #169 + vllm-omni #4388
  → ★★★★★★★★ But: still DRAFT → critical review issue (requires_grad non-leaf) → not production-ready!
  → Detailed reading: notebook/projects/verl-pr6713-megatron-lora-export-reading.md

Megatron-Bridge (NeMo2):
  → LoRA in NeMo2/Megatron-Bridge → separate path from core Megatron
  → Issue #3210: "LoRA only in NeMo2/Megatron-Bridge" → not in Megatron core
  → ★★★ NeMo2 LoRA = Megatron's only native LoRA path → but requires NeMo2 stack

Megatron Lite (#4885):
  → ★★★★★★ LinearLoRA + GroupedLinearLoRA → freeze_non_lora_params → PEFT import/export
  → verl integration → mlite_engine.py → GRPO scripts
  → ★★★★★★★★ Lite has PEFT import/export → bridges to HF format → vLLM compatible!

★★★★★★★★★ RTX 4090 Megatron LoRA export:
  → #5203 crash on single GPU → can't use core Megatron → need DeepSpeed instead!
  → Lite LoRA export → but Lite needs verl → not standalone on RTX 4090
  → ★★★★★★★★ BEST: DeepSpeed trains LoRA → PEFT export → vLLM serve → standard path!
```

### vLLM

```
★★★★★★★★★ vLLM LoRA Serving = MOST MATURE:

Multi-LoRA serving (--enable-lora):
  → Load base model + multiple LoRA adapters → hot-swap per request
  → ★★★★★ LoRA adapter in HF safetensors format → standard PEFT → portable
  → max_loras → limit concurrent active adapters
  → LoRA weights loaded on-demand → not all adapters in GPU memory simultaneously

vLLM LoRA integration (V1):
  → LoRA weights cached → switch between adapters → latency ~0.1s
  → ★★★★★★★★ Hot-swap = A/B testing → multiple GRPO variants → same base model

vLLM LoRA + quantization:
  → INT4 base model + LoRA BF16 adapter → compatible!
  → LoRA on top of quantized weights → adapter weights in BF16 → compute in BF16
  → ★★★★★ INT4 Marlin kernel + LoRA = production viable on RTX 4090

★★★★★★★★★ vLLM LoRA export: NOT needed → vLLM SERVES LoRA, doesn't TRAIN
  → LoRA trained externally (DeepSpeed/rLLM/verl) → PEFT format → vLLM loads directly
  → ★★★★★★★★ vLLM = the SERVING endpoint → not the training start point!
```

### verl

```
★★★★ verl LoRA Export (FSDP backend):
  → verl trains LoRA via GRPO/PPO/CPPO → worker-actor architecture
  → Export: standard PEFT save_pretrained → HF safetensors → portable
  → ★★★★★★★★ verl + rLLM Tinker: LoRA trained in-process → no IPC → simplest path
  → → ★★★★★★★★ Export via PEFT → vLLM serve → end-to-end pipeline = simplest!

★★★★★★★★★ verl LoRA Export (Megatron backend + PR #6713):
  → ★★★★★★★★ NEW: adapter-only export → EP gather + 3D MoE pack → vLLM format
  → ★★★★★★★★ CRITICAL: PR #6713 is in verl repo, NOT Megatron-LM repo!
  → EP=1 (RTX 4090): passthrough → zero overhead → but can't train on Megatron backend (single GPU crash)
  → Multi-GPU training → export adapter → deploy on RTX 4090 vLLM → serving bridge viable
  → ★★★★★★★★ But: DRAFT + critical review issue → not production-ready
  → Detailed reading: notebook/projects/verl-pr6713-megatron-lora-export-reading.md

★★★★ verl ContinuousToken (#6720/#6721 OPEN):
  → Multi-turn agent RL → LoRA mask=[1,0,1,0,...] → action tokens only
  → ★★★ Export: standard PEFT → but response_mask alignment needs careful handling
  → → If using CT for training → export needs mask-aware merge → not yet standard
```

### rLLM

```
★★★★★★★★★ rLLM Tinker LoRA Export = SIMPLEST:

TinkerBackend:
  → ★★★★★★★★ Auto-init LoRA → create_lora_training_client_async → zero manual code
  → LoRA trained in-process → same GPU → no IPC → zero-copy
  → Export: standard PEFT save_pretrained → HF safetensors → ★★★★★★★★ PORTABLE!

★★★★★★★★★ rLLM = SIMPLEST training→serving path:
  → rLLM Tinker trains LoRA → PEFT save → vLLM serve → ★★★★★★★★ 3 commands!
  → No format conversion needed → standard PEFT → vLLM directly compatible

★★★★★★★★★ rLLM LoRA hot-swap path:
  → Train multiple LoRA adapters (different reward functions, different data)
  → PEFT save each → vLLM --enable-lora → hot-swap per request
  → ★★★★★★★★ RTX 4090 A/B testing = train 3 LoRA adapters → serve all from one base model!
```

### SGLang

```
★★★★ SGLang LoRA Serving:
  → SGLang supports LoRA serving → similar to vLLM --enable-lora
  → ★★★★★ Standard PEFT format → HF safetensors → portable
  → ★★★ Deterministic inference → LoRA + constexpr BLOCK_SIZE → batch-invariant

★★★★ SGLang LoRA export: NOT needed → SGLang SERVES LoRA, doesn't TRAIN
  → Same as vLLM: training externally → PEFT format → SGLang loads
```

### MindIE

```
★★★★ MindIE LoRA:
  → MindIE production deployment → model serving → LoRA adapters
  → ★★★★★★★★ vLLM-Ascend = LoRA serving with BudgetRefiner → preserves SLO!
  → Ascend NPU LoRA → same PEFT format → GPU-generic
  → ★★★★★★★★ BudgetRefiner SLO + LoRA hot-swap → dynamic budget per LoRA variant!

★★★★ MindIE Turbo (closed source):
  → No LoRA hot-swap info → scheduling black box → unknown LoRA handling
```

---

## 3. Format Compatibility Matrix

| Train → Serve | DeepSpeed | Megatron Lite | verl/rLLM | verl #6713 (Megatron backend) |
|---------------|-----------|---------------|-----------|-------------------------------|
| **vLLM** | ★★★★★ PEFT→HF→vLLM | ★★★★★ PEFT→HF→vLLM | ★★★★★ PEFT→HF→vLLM | ★★★★★★★★ Direct vLLM format! |
| **SGLang** | ★★★★★ PEFT→HF→SGLang | ★★★★★ PEFT→HF→SGLang | ★★★★★ PEFT→HF→SGLang | ★★ Unknown |
| **MindIE** | ★★★★ PEFT→Ascend | ★★ Megatron→Ascend | ★★★★ PEFT→Ascend | ★★ Direct vLLM-Ascend |

★★★★★★★★★ **Key**: All frameworks except Megatron core use standard PEFT/HF safetensors → format portable → vLLM/SGLang directly compatible. verl #6713 explicitly targets vLLM format (EP gather + 3D MoE pack) → most ambitious bridge but still DRAFT with critical review issue.

---

## 4. RTX 4090 LoRA Training→Serving Decision Tree

```
★★★★★★★★★ RTX 4090 LoRA pipeline — choose based on scenario:

Scenario A: Dense model GRPO (Qwen3-1.7B, Qwen3-4B)
  → Train: rLLM Tinker (#1) → auto-init LoRA → bypass_mode → simplest
  → Export: PEFT save_pretrained → standard → portable
  → Serve: vLLM --enable-lora (multi-LoRA hot-swap) OR SGLang deterministic
  → ★★★★★★★★ TOTAL: 3 commands → simplest end-to-end path!

Scenario B: MoE model GRPO (Qwen3-MoE)
  → Train: DeepSpeed AutoEP + ZeRO-2 + LoRA → EP=1 → single GPU viable
  → Export: PEFT save_pretrained → but MoE structure → careful merge
  → Serve: vLLM INT4 MoE Triton fallback → LoRA on top of quantized base
  → ★★★★★★★★ Gap: DeepSpeed no RouterReplay → MoE CUDA graph stability?

Scenario C: OPD distillation (Qwen2.5-0.5B student)
  → Train: DeepSpeed OPD #8027 (DRAFT) → student GPU + teacher CPU
  → Export: PEFT save → student LoRA → 0.5B → tiny → easy
  → Serve: vLLM/SGLang → 0.5B INT4 → ~0.2GB → massive concurrency
  → ★★★★★★★★ SIMPLEST deployment! → 0.5B INT4 = near-unlimited concurrent requests

★★★★★★★★★ Universal path (all scenarios):
  → Train LoRA → PEFT export → merge_and_unload (zero overhead) → vLLM/SGLang serve
  → OR: Train LoRA → PEFT export → keep separate → vLLM --enable-lora (hot-swap)
  → ★★★★★★★★ merge = zero overhead but irreversible → hot-swap = A/B testing but slight overhead
```

---

## Key Findings Summary

★★★★★★★★★ ALL frameworks (except Megatron core) use standard PEFT/HF safetensors → format portable → vLLM/SGLang directly compatible
★★★★★★★★★ rLLM Tinker = SIMPLEST training→serving path → auto-init LoRA → PEFT save → vLLM serve → 3 commands!
★★★★★★★★★ vLLM multi-LoRA serving (--enable-lora) = MOST MATURE serving endpoint → hot-swap per request
★★★★★★★★★ verl #6713 (DRAFT) = most ambitious bridge → verl Megatron backend LoRA → vLLM format → MoE EP gather + 3D pack → but critical review issue + not merged
★★★★★★★★★ Megatron crashes on single GPU (#5203) → can't use core Megatron → need DeepSpeed/rLLM instead
★★★★★★★★★ DeepSpeed AutoEP LoRA export: standard PEFT but no RouterReplay → MoE deployment gap
★★★★★★★★★ merge_and_unload() = zero inference overhead (5.62ms vs 5.71ms → <2% → effectively free)
★★★★★★★★★ LoRA hot-swap = unique RTX 4090 advantage → multiple adapters → same base model → A/B testing → zero re-merge cost
★★★★★★★★★ INT4 base + LoRA BF16 = production viable → INT4 Marlin kernel + LoRA → RTX 4090 compatible

---

## References

- Training config: notebook/fundamentals/rtx4090-grpo-training-config-card.md
- Deployment pipeline: notebook/fundamentals/rtx4090-training-to-deployment-pipeline.md
- Inference serving: notebook/fundamentals/rtx4090-inference-serving-checklist.md
- rLLM GRPO guide: notebook/projects/rllm-grpo-practical-training-guide.md
- DeepSpeed config generator: tools/deepspeed_config_generator.py
- Megatron LoRA export: notebook/projects/verl-pr6713-megatron-lora-export-reading.md
- Megatron Lite: notebook/projects/megatron-lite-reading.md
- RouterReplay: notebook/projects/megatron-router-replay-source-reading.md
- BudgetRefiner synthesis: notebook/projects/watermark-budgetrefiner-complementary-synthesis.md
