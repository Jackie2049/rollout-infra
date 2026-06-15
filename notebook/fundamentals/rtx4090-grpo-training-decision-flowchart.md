# RTX 4090 GRPO Training Decision Flowchart

> 2026-06-16 | RTX 4090 consulting reference
> Focus: Step-by-step decision guide for choosing GRPO training configuration on RTX 4090
> Reference: single-gpu-ddp-vs-zero-architecture-comparison.md, rtx4090_grpo_config_matrix.py

---

## Decision Flowchart

```
START: Want to train LLM with GRPO on RTX 4090?
│
├── Q1: Model size?
│   ├── ≤1.7B (e.g., Qwen3-1.7B)
│   │   → PROCEED (fits 24GB easily with LoRA+bypass)
│   ├── ~7-8B (e.g., Qwen3-8B, Llama-3.1-8B)
│   │   → PROCEED WITH CARE (need INT4 quantization + LoRA + bypass)
│   │   → Use GPTQ-Int4 model for inference
│   │   → LoRA on BF16 base + INT4 inference = hybrid approach
│   └── ≥14B (e.g., Mistral-7B full, Qwen2.5-14B)
│       → NOT VIABLE on 24GB (even with INT4)
│       → Recommend: rent cloud GPU or use smaller model
│
├── Q2: Training framework preference?
│   ├── Want simplest setup → rLLM Tinker (in-process, auto config)
│   │   → tools/train_tinker_rtx4090.sh --model Qwen3-1.7B --task math
│   │   → LoRA rank=32, group_size=4, batch_size=8, bypass=true
│   │   → LossFnType choices: ppo (default), IS, cispo, dro
│   │
│   ├── Want maximum flexibility → verl (Ray-based, many algorithms)
│   │   → Algorithm choice (Q3):
│   │   ├── CPPO+bypass → provably best bound, near-zero overhead
│   │   ├── GRPO+bypass → simplest, decent results
│   │   ├── ReMax+bypass_ppo_clip → lowest variance BUT needs ref model for canonical!
│   │   │   → bypass ReMax = partial (greedy baseline still works, but use_kl_in_reward ✗)
│   │   ├── IcePop+bypass → most precise IS correction, less stable
│   │
│   ├── Want DeepSpeed integration → DeepSpeed ZeRO-2+LoRA
│   │   → NOT recommended for RL (no bypass_mode native, Ray overhead)
│   │   → Better for supervised fine-tuning only
│   │   → ZeRO-2 + CPU_Adam + LoRAOptimizedLinear + coalesce_grad_reduction
│   │
│   └── Want Megatron → NOT VIABLE on RTX 4090
│       → No LoRA in core, singleton PG bugs, DDP overhead
│       → Use Megatron Lite (LoRA included) only for MoE models
│
├── Q3: RL algorithm choice? (if verl)
│   ├── CPPO (#6731) + bypass + GRPO advantage
│   │   → ★★★★★★★★ RTX 4090 optimal trust region
│   │   → Position-weighted cumulative prefix divergence
│   │   → MUST use bypass_mode (divergence measured against rollout μ)
│   │
│   ├── GRPO (default) + bypass
│   │   → ★★★★★ Simplest, widely validated
│   │   ├── group_size=4 → 4 samples per task
│   │   ├── batch_size=8 → 8 tasks per step
│   │   ├── bypass_mode=True → no ref model → save 14GB
│   │
│   ├── ReMax (#6340) + bypass_ppo_clip
│   │   → ★★★★★ Lowest variance (greedy baseline)
│   │   → BUT: canonical ReMax requires ref model (use_kl_in_reward=True)
│   │   → bypass ReMax = still works for greedy baseline, but no KL penalty
│   │   → Recommended: ReMax + PPO-clip (no KL in reward)
│   │
│   ├── IcePop (#5722) + bypass
│   │   → ★★★★ Most precise importance sampling
│   │   → torch.where(token_kept_mask, weight, 0) → exact population [0.5, 5.0]
│   │   → Less stable for beginners → use with caution
│   │
│   └── MAGI (#6689) prefix-tree KV dedup
│   │   → ★★★★★ 省7/8 prefix KV → RTX 4090直接受益
│   │   → BUT: currently depends on Megatron → needs vLLM/SGLang adapter
│   │   → Medium-term feasible
│
├── Q4: Quantization strategy?
│   ├── BF16 only (≤1.7B model)
│   │   → Simple, no quantization needed
│   │   → INT8 KV cache for inference (FlashInfer backend)
│   │
│   ├── GPTQ-Int4 (7-8B model)
│   │   → Required for inference on 24GB
│   │   → Training still on BF16 base weights (LoRA)
│   │   → Hybrid: LoRA on BF16 base + INT4 inference
│   │   → Marlin/Triton INT4 backend on SM89
│   │
│   ├── FP8 (AVOID on SM89!)
│   │   → Triton FP8 KV: ALLOWED on SM89 (#43914)
│   │   → FlashInfer FP8: NOT supported on SM89
│   │   → compressed-tensors FP8 KV: CRASH on SM89 (#44879/#45038)
│   │   → INT8 KV = only production-viable path on SM89
│   │
│   └── INT4 Marlin/Triton
│   │   → ★★★★★ Only viable inference quantization on SM89
│   │   → vLLM INT4 Triton fallback (#43731) → works on SM89
│   │   → Marlin: faster but needs W4A16 quantized model
│
├── Q5: Sequence length?
│   ├── ≤2048 → most configs work
│   ├── 2048-4096 → 1.7B OK, 8B needs INT4 + smaller batch
│   └── ≥4096 → 8B NOT viable, use 1.7B or smaller
│
├── Q6: Memory optimization?
│   ├── bypass_mode=True → save ~model_size GB (no ref model)
│   │   → MUST for all RTX 4090 RL training
│   ├── LoRA rank=32 → ~0.6GB trainable params
│   │   → train_mlp+attn+unembed (Tinker default)
│   ├── INT8 KV cache → ~50% KV memory savings
│   │   → FlashInfer backend required
│   ├── gradient_checkpointing → saves activation memory
│   │   → recommended for 8B models
│   └── CPU offload (DeepSpeed only) → optimizer on CPU
│       → useful for LoRA (CPU_Adam SIMD 5-7x faster)
│       → but LoRA params small → may not be worth the transfer overhead
│
├── Q7: Batch invariance concern?
│   ├── Using torch.compile → MUST fix batch invariance!
│   │   → Option A: Inductor SM<90 Fusion Guard (our proposed PR)
│   │   → Option B: SGLang deterministic inference (--enable-deterministic-inference)
│   │   → Option C: enforce_eager=True (no CUDA graphs, slower but correct)
│   │   → Option D: VLLM_USE_V2_MODEL_RUNNER=0 (conservative, may not be necessary)
│   │
│   ├── Using vLLM without compile → generally safe
│   │   → But: vLLM V0.23.0 defaults MRv2 for Qwen3/Llama/Mistral
│   │   → MRv2 safe for verl (AsyncLLM.generate handles internally)
│   │   → Still: GPU verification recommended
│   │
│   └── Using SGLang → deterministic inference built-in
│       → --enable-deterministic-inference → batch-invariant ops
│       → Triton backend → constexpr BLOCK_SIZE → no autotuning
│       → Recommended for GRPO inference on SM89!
│
└── Q8: Serving framework for GRPO rollout?
    ├── vLLM (default with verl)
    │   → mature, widely tested
    │   → INT8 KV on SM89 (FlashInfer backend)
    │   → batch invariance: need compile fix or enforce_eager
    │
    ├── SGLang (alternative)
    │   → deterministic inference built-in
    │   → RadixAttention → prefix KV reuse → GRPO benefit
    │   → Triton backend recommended for SM89
    │   → verl SGLang integration available (#6117)
    │
    └── vLLM + SGLang hybrid
        → SGLang for inference (deterministic) + vLLM for training
        → Not yet common, but possible via verl
```

---

## Quick Reference: Recommended Configs

### rLLM Tinker (Easiest, Recommended #1)

```yaml
model: Qwen/Qwen3-1.7B
lora_rank: 32
train_mlp: true
train_attn: true
train_unembed: true
bypass_mode: true  # No ref model
group_size: 4
batch_size: 8
learning_rate: 2e-5
lr_schedule: cosine
warmup_ratio: 0.1
```

### verl + CPPO + bypass (Most Flexible)

```yaml
actor:
  model: Qwen/Qwen3-1.7B
  lora_rank: 32
  bypass_mode: true
algorithm:
  type: cppo
  ppo_clip_ratio: 0.2
rollout:
  backend: vllm
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.85
```

### DeepSpeed ZeRO-2 + LoRA (Fine-tuning Only)

```json
{
  "zero_optimization": { "stage": 2, "offload_optimizer": "cpu" },
  "optimizer": { "type": "CPUAdam", "params": { "lr": 2e-5 } },
  "lora": { "enabled": true, "rank": 32 }
}
```

---

## Memory Budget Table

| Config | Model | LoRA | Bypass | KV | Est Memory | Fits 24GB? |
|--------|-------|------|--------|----|-----------|-----------|
| Tinker+LoRA+bypass | 1.7B BF16 | rank=32 | yes | FP16 | ~9.2GB | YES ✓ |
| Tinker+LoRA+bypass | 1.7B BF16 | rank=32 | yes | INT8 | ~7.2GB | YES ✓ |
| Tinker+LoRA+bypass | 8B BF16 | rank=32 | yes | INT8 | ~24.8GB | TIGHT ⚠️ |
| Tinker+LoRA+bypass | 8B INT4 | rank=32 | yes | INT8 | ~8.5GB | YES ✓ |
| verl+CPPO+bypass | 1.7B BF16 | rank=32 | yes | FP16 | ~10GB | YES ✓ |
| DeepSpeed+LoRA | 1.7B BF16 | rank=32 | ✗ | FP16 | ~23GB | TIGHT ⚠️ |
| DeepSpeed+LoRA+bypass | 1.7B BF16 | rank=32 | yes | FP16 | ~10GB | YES ✓ |
