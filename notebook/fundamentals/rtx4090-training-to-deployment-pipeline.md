# RTX 4090 Training → Deployment Pipeline — End-to-End Practical Guide

> 2026-06-17 | LoRA training output → merge → quantize → serve → complete lifecycle
> ★★★★★ First end-to-end guide synthesizing 7-framework knowledge into practical pipeline

---

## 1. Pipeline Overview

```
★★★★★★★★★ RTX 4090 Training → Deployment = 6 stages:

Stage 1: LoRA GRPO Training → LoRA adapter weights
  → DeepSpeed ZeRO-2 / rLLM Tinker / verl HYBRID
  → Output: LoRA adapter checkpoint (~0.3-1.2GB)

Stage 2: Merge LoRA → Full model weights
  → PEFT merge → save_pretrained → HF format
  → Output: merged BF16 model (~3.4-14GB)

Stage 3: Quantize → INT4/AWQ/GPTQ
  → AWQ/GPTQ → INT4 Marlin-compatible → 3x compression
  → Output: quantized model (~1.7-3.5GB)
  → ★★★ MoE: INT4 expert weights → Triton fallback (#43731)

Stage 4: vLLM/SGLang Serve → Production inference
  → vLLM enforce_eager + INT8 KV + prefix caching
  → SGLang deterministic + Triton backend (alternative)
  → Output: OpenAI-compatible API server

Stage 5: Speculative Decoding (optional)
  → EAGLE/N-gram/MTP → 2-4x speedup
  → ★★★ RTX 4090: N-gram simplest (2.14x), EAGLE best (4.2x)

Stage 6: Continuous Training (optional)
  → New data → re-train LoRA → re-merge → re-deploy
  → ★★★ LoRA adapter modular → hot-swap without re-deploying base model!
```

---

## 2. Stage 1: LoRA GRPO Training

### Recommended: rLLM Tinker (#1 for RTX 4090)

```bash
# ★★★ rLLM Tinker = simplest + fastest + bypass built-in
# Install
pip install rllm

# Train (Qwen3-1.7B, LoRA rank=32, GRPO)
python -m rllm.tinker.train \
  --model Qwen/Qwen3-1.7B \
  --algorithm GRPO \
  --lora_rank 32 \
  --bypass_mode true \
  --group_size 4 \
  --batch_size 8 \
  --output_dir ./output/qwen3-1.7b-grpo-lora32
```

### Alternative: DeepSpeed ZeRO-2 (for MoE/OPD scenarios)

```bash
# Generate safe config
python3 tools/deepspeed_config_generator.py --scenario lora-grpo --model qwen3-1.7b \
  --output configs/lora-grpo_rtx4090.json

# Train
deepspeed --num_gpus=1 train.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --peft_method lora --lora_rank 32 \
  --deepspeed configs/lora-grpo_rtx4090.json \
  --output_dir ./output/qwen3-1.7b-grpo-lora32

# ★★★ Safety checks before training:
python3 tools/deepspeed_zero_safety_checker.py --mode check --config configs/lora-grpo_rtx4090.json
```

### Output

```
output/qwen3-1.7b-grpo-lora32/
  adapter_model.safetensors  (~0.3GB) ← LoRA weights
  adapter_config.json        ← LoRA config (rank, target modules)
  trainer_state.json         ← training metrics
```

★★★★★ **Key**: LoRA adapter is SMALL (~0.3GB for rank=32 on 1.7B). Multiple adapters can coexist → switch between them without redeploying base model → A/B testing capability!

---

## 3. Stage 2: Merge LoRA → Full Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, "./output/qwen3-1.7b-grpo-lora32")

# Merge LoRA into base weights
merged_model = model.merge_and_unload()

# Save merged model in HF format
merged_model.save_pretrained("./merged/qwen3-1.7b-grpo", safe_serialization=True)
```

★★★★★★★★★ **merge_and_unload()**: LoRA weights permanently merged → no PEFT dispatch overhead at inference → speed = base model speed (zero overhead from our benchmark data: 5.62ms vs 5.71ms → <2% difference → effectively free)

★★★★★ **WARNING**: merge is IRREVERSIBLE → base model weights modified → keep original base model separately if you want to train new LoRA adapters later!

### For MoE models (Qwen3-MoE)

```bash
# AutoEP training with LoRA → only active params LoRA'd
# Merge: LoRA on active layers → merged into expert weights
# ★★★ MoE merge = more complex → expert weights distributed →
#     use Megatron LoRA adapter export (#6713 DRAFT) when available
#     Currently: manual merge via PEFT → save_pretrained
```

---

## 4. Stage 3: Quantize → INT4/AWQ/GPTQ

### AWQ (recommended for RTX 4090)

```bash
# AWQ quantization → 4-bit → Marlin kernel compatible
pip install autoawq

python -m awq.entrypoint \
  --model_path ./merged/qwen3-1.7b-grpo \
  --w_bit 4 \
  --q_group_size 128 \
  --zero_point True \
  --output_path ./quantized/qwen3-1.7b-grpo-awq-int4

# Result: 3.4GB BF16 → ~1.0GB INT4 → 3.4x compression
# ★★★★★ Must use Marlin kernel for inference → Python dequant = 20x slower!
```

### GPTQ (alternative)

```bash
pip install auto-gptq

# GPTQ with Marlin backend → RTX 4090 compatible
python -m gptq.entrypoint \
  --model_path ./merged/qwen3-1.7b-grpo \
  --bits 4 \
  --group_size 128 \
  --desc_act False \
  --output_path ./quantized/qwen3-1.7b-grpo-gptq-int4
```

★★★★★★★★★ **INT4 Marlin kernel**: Required for RTX 4090 production serving → Python dequant 20x slower → fused Marlin kernel = near-native speed → vLLM auto-detects and uses Marlin when available

### MoE INT4 Quantization

```
★★★★★★★ MoE INT4 considerations (Qwen3-MoE / Mixtral):
  → Expert weights = bulk of model → INT4 savings enormous (~3x)
  → vLLM INT4 Triton fallback (#43731) → non-128-aligned shapes → Triton kernel
  → ★★★ Qwen3-MoE intermediate_size → check if 128-aligned
  → → If non-aligned → TritonW4A16LinearKernel → slower but WORKS on RTX 4090!
  → → ★★★★★ Production viable even with Triton fallback!
```

---

## 5. Stage 4: vLLM/SGLang Serve

### vLLM Production Config (RTX 4090 recommended)

```bash
# ★★★★★ RTX 4090 vLLM serving — production-ready
python -m vllm.entrypoints.openai.api_server \
  --model ./quantized/qwen3-1.7b-grpo-awq-int4 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype int8 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --enforce-eager \
  --watermark 0.05

# ★★★ Key flags:
# --enforce-eager: SM89 batch invariance → avoid Inductor RMSNorm fusion
# --kv-cache-dtype int8: FlashInfer INT8 KV → only viable KV quant on SM89
# --enable-prefix-caching: GRPO prompt reuse → HMA-by-default
# --watermark 0.05: Watermark #44594 → preemptions -82% → ITL p99 -56%
```

### SGLang Alternative (deterministic inference)

```bash
# ★★★★★ SGLang deterministic — batch-invariant by design
python -m sglang.launch_server \
  --model-path ./quantized/qwen3-1.7b-grpo-awq-int4 \
  --enable-deterministic-inference \
  --mem-fraction-static 0.85 \
  --context-length 4096

# ★★★ Key advantage: deterministic inference → 7 aten overrides →
#     tl.constexpr BLOCK_SIZE → NO autotuning → batch-invariant by design!
#     No need for enforce_eager → compile-level batch invariance at kernel level!
```

### Choosing vLLM vs SGLang for RTX 4090 Serving

| Factor | vLLM | SGLang |
|--------|------|--------|
| Batch invariance | enforce_eager (disable compile) | deterministic (kernel-level) |
| KV quant | INT8 FlashInfer | INT8 Triton |
| Prefix caching | HMA-by-default | RadixAttention (tree-based) |
| Spec decode | EAGLE/N-gram/MTP | NEXTN (DeepSeek) |
| SLO control | BudgetRefiner (future) | None (but WAR barrier #28363) |
| MoE support | INT4 Triton fallback | Triton MoE runner |
| RTX 4090推荐 | ★★★★ (enforce_eager safe) | ★★★★★ (deterministic by design!) |

★★★★★★★★★ **RTX 4090 serving recommendation**: SGLang deterministic for quality-sensitive tasks (GRPO rollout, evaluation). vLLM for throughput-focused serving with Watermark + BudgetRefiner (future).

---

## 6. Stage 5: Speculative Decoding (Optional)

### N-gram (simplest, 2.14x speedup)

```bash
# vLLM N-gram spec decode — zero-cost draft model
python -m vllm.entrypoints.openai.api_server \
  --model ./quantized/qwen3-1.7b-grpo-awq-int4 \
  --speculative-algorithm ngram \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4 \
  --enforce-eager
```

### EAGLE (best, 4.2x speedup)

```bash
# EAGLE spec decode — trained draft model → best speedup
python -m vllm.entrypoints.openai.api_server \
  --model ./quantized/qwen3-1.7b-grpo-awq-int4 \
  --speculative-model ./eagle-draft-model \
  --speculative-algorithm eagle \
  --num-speculative-tokens 4 \
  --enforce-eager
```

★★★★★ **RTX 4090 spec decode**: N-gram = simplest (no draft model, 2.14x). EAGLE = best (trained draft, 4.2x). Both tested on RTX 4090 from our benchmark data.

---

## 7. Stage 6: Continuous Training → Hot-Swap LoRA

★★★★★★★★★ **LoRA hot-swap = unique advantage**: Multiple LoRA adapters trained on different data → switch without redeploying base model → A/B testing → incremental updates.

### vLLM Multi-LoRA Serving

```bash
# vLLM multi-LoRA serving — hot-swap adapters
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --enable-lora \
  --lora-modules grpo-v1=./output/qwen3-1.7b-grpo-lora32 \
                 grpo-v2=./output/qwen3-1.7b-grpo-v2-lora32 \
  --max-loras 2 \
  --enforce-eager

# Request with specific LoRA:
curl /v1/completions -d '{"model": "Qwen/Qwen3-1.7B", "lora": "grpo-v2", ...}'
```

★★★★★ **Multi-LoRA = training → serving bridge**: Train multiple LoRA adapters → serve all from same base model → zero re-merge cost → switch instantly.

---

## 8. MoE Full Pipeline (Qwen3-MoE)

```
★★★★★★★★★ Qwen3-MoE RTX 4090 Full Pipeline (DeepSpeed AutoEP):

Stage 1: Train → DeepSpeed AutoEP + ZeRO-2 + LoRA
  → auto_ep_preset: "Qwen3-MoE"
  → EP=1 → all experts single GPU → AutoEP skips AllToAll (#7997)
  → LoRA on active params → expert weights frozen
  → overlap_comm=False (#8061 MUST!) + gradient_clipping=1.0 (#8068)

Stage 2: Merge → PEFT merge_and_unload
  → LoRA merged into active layers → save_pretrained
  → ★★★ MoE merge = careful → expert weight structure preserved

Stage 3: Quantize → GPTQ INT4
  → Expert weights INT4 → 3x compression → bulk savings
  → Triton fallback (#43731) for non-128-aligned → works on RTX 4090!

Stage 4: Serve → vLLM INT4 MoE
  → Triton MoE runner → INT4 with Triton fallback
  → ★★★ MoE serving = expert routing per token → dynamic → can't CUDA graph all experts
  → → BudgetRefiner (future) → SLO-aware → decode pressure → reduce prefill budget

★★★★★★★★★ Alternative serving: SGLang deterministic + Triton MoE
  → Triton persistent matmul → constexpr BLOCK_SIZE → batch-invariant
  → ★★★ SGLang MoE Triton runner = correct choice for SM89 (#28355 validates)
```

---

## 9. OPD Distillation Pipeline (Qwen2.5-0.5B → 7B teacher)

```
★★★★★★★★★ OPD distillation RTX 4090 pipeline (DeepSpeed #8027):

Stage 1: Train student → OPD Trainer
  → Qwen2.5-0.5B student on GPU (~1GB)
  → Qwen2.5-7B teacher on CPU (~16.5GB RAM)
  → 3-phase: student rollout → teacher CPU logit cache → student divergence
  → ZeRO-0 viable! → student so small → no partitioning needed

Stage 2: Merge student LoRA → save_pretrained
  → 0.5B student + LoRA → merge → ~1GB BF16 → tiny!

Stage 3: Quantize → AWQ INT4
  → 0.5B → INT4 → ~0.2GB → incredibly small → massive headroom

Stage 4: Serve → vLLM/SGLang
  → 0.5B INT4 → 24GB headroom → insane concurrency possible
  → ★★★★ OPD = transfer capability → smaller but smarter model!
  → → Combined pipeline: OPD (transfer) → GRPO (explore beyond) → best
```

---

## Key Takeaways

★★★★★★★★★ LoRA hot-swap = unique advantage → train multiple adapters → serve from one base model → A/B testing → incremental updates
★★★★★★★★★ merge_and_unload() → zero inference overhead → same speed as base model (5.62ms vs 5.71ms)
★★★★★★★★★ INT4 Marlin kernel = MUST for RTX 4090 → Python dequant 20x slower → fused kernel near-native speed
★★★★★★★★★ MoE INT4 Triton fallback = production viable even for non-128-aligned shapes (#43731)
★★★★★★★★★ SGLang deterministic → kernel-level batch invariance → better for quality-sensitive tasks (GRPO rollout)
★★★★★★★★★ vLLM enforce_eager → disable compile → batch invariance safe → -10-20% throughput cost
★★★★★★★★★ Watermark 0.05 + BudgetRefiner (future) → preemptions -82% + SLO control → approaching zero preemptions
★★★★★★★★★ OPD → GRPO pipeline: OPD transfers capability (small GPU high CPU) → GRPO explores beyond (high GPU low CPU)

---

## References

- RTX 4090 training config card: notebook/fundamentals/rtx4090-grpo-training-config-card.md
- RTX 4090 inference serving checklist: notebook/fundamentals/rtx4090-inference-serving-checklist.md
- DeepSpeed config generator: tools/deepspeed_config_generator.py
- DeepSpeed ZeRO safety checker: tools/deepspeed_zero_safety_checker.py
- Cross-framework safety matrix: tools/rtx4090_cross_framework_safety_matrix.py
- OPD comparison: notebook/fundamentals/cross-framework-opd-distillation-comparison.md
- SGLang deterministic: notebook/projects/sglang-source-reading.md (section on deterministic inference)
- BudgetRefiner PR draft: notebook/projects/budgetrefiner-vllm-pr-draft.md
