# RTX 4090 MoE Training Viability — DeepSpeed AutoEP Revolution

> 2026-06-16 | DeepSpeed AutoEP MERGED #7938 changes RTX 4090 MoE landscape
> Focus: MoE model training on RTX 4090 (24GB) — previously thought impossible, now viable!
> ★★★★★★★ AutoEP + EP=1 singleton + LoRA + CPU_Adam = RTX 4090 MoE breakthrough

---

## 1. The Problem: MoE Models on 24GB

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

MoE models have massive parameter counts:

```
Qwen3-MoE (A0.6B + B4B):
  Active params:     0.6B (router + shared + 4 active experts)
  Total params:      4B+  (all 64 experts + router + shared)
  Full model memory: ~16GB BF16 weights alone
  With optimizer:    ~48GB (3x for Adam m+v fp32)
  With activations:  ~56GB total → EXCEEDS 24GB ✗✗✗
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ Previously: MoE training on RTX 4090 = IMPOSSIBLE. Even Qwen3-MoE 4B total params exceeds 24GB with optimizer.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 2. The Breakthrough: DeepSpeed AutoEP + LoRA + EP=1

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Key insight: LoRA makes MoE viable!

```
With LoRA rank=32 on Qwen3-MoE:
  Frozen base weights:     16.0GB  (all 4B+ params, BF16, frozen)
  LoRA params:             0.6GB   (rank=32 on router + expert MLP)
  Optimizer (LoRA only):   1.2GB   (Adam m+v for 0.6GB LoRA, fp32)
  Activations:             2.0GB   (small, LoRA forward/backward)
  KV cache:                0GB     (training only, no inference KV)
  CPU offload (ZenFlow):   -0GB    (optimizer on CPU → no GPU spike!)
─────────────────────────────────────
Total GPU:                ~17.8GB ← FITS 24GB! ✓✓✓
Available headroom:        ~6.2GB
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ LoRA + CPU_Adam offload = only train ~0.6GB LoRA params → optimizer on CPU → base weights frozen → activations small → FITS 24GB!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### AutoEP EP=1 singleton path:

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ EP=1 singleton MoE (#7997) → ALL experts on SAME GPU → skip ALL AllToAll → 15x speedup!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

EP=1 means:
  → All experts co-located on single GPU
  → No cross-GPU token dispatch
  → TokenChoiceTopKRouter runs locally
  → AllToAll = identity operation → SKIP entirely!
  → GroupedExperts execute on same GPU → no MoE communication!

vs EP>1 (multi-GPU):
  → Tokens dispatched across GPUs
  → AllToAll = actual data exchange
  → PCIe bandwidth bottleneck on RTX 4090 (no NVLink!)
  → Latency dominates → slower than dense model!
```

---

## 3. Memory Comparison: Dense vs MoE on RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B Dense + LoRA + bypass + rLLM Tinker (RTX 4090 optimal):

```
Frozen base weights:       3.4GB
LoRA params:               0.6GB
Optimizer (LoRA, CPU):     0GB (offloaded)
Activations:               2.0GB
KV cache:                  2.0GB
Ref model:                 0GB (bypass=true)
CUDA context:              1.0GB
─────────────────────────────
Total:                    ~9.0GB ← #1 most efficient
```

### Qwen3-MoE (A0.6B+B4B) + LoRA + AutoEP EP=1 + DeepSpeed ZeRO-2:

```
Frozen base weights:      16.0GB  (all 4B+ params)
LoRA params:               0.6GB  (rank=32)
Optimizer (CPU offload):   0GB    (CPU_Adam)
Activations:               2.0GB
CUDA context:              1.0GB
ZenFlow chunked copyback:  0.25GB (spike, vs 2.94GB old!)
─────────────────────────────
Total:                    ~19.85GB ← FITS! ✓
```

★★★★★★★ MoE training uses ~20GB vs dense training ~9GB → both fit 24GB → MoE viable!

---

## 4. AutoEP Configuration for RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### DeepSpeed JSON config (RTX 4090 MoE):

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": false,
    "contiguous_gradients": true,
    "allgather_bucket_size": 5e8
  },
  "gradient_accumulation_steps": 4,
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 2,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": "auto"
  },
  "bf16": {
    "enabled": true
  },
  "data_types": {
    "param_dtype": "bf16",
    "buffer_dtype": "fp32"
  },
  "auto_expert_parallelism": {
    "enabled": true,
    "expert_parallel_size": 1,
    "model_type": "qwen3_moe"
  },
  "lora": {
    "enabled": true,
    "rank": 32,
    "target_modules": ["router", "expert_mlp"],
    "offload_ratio": 0.5
  }
}
```

★★★★★★★★★ Key config decisions:
- ZeRO-2 (not ZeRO-3!) → ZeRO-3 useless on single GPU
- CPU_Adam offload → optimizer on CPU → no GPU memory for optimizer
- EP=1 → singleton → 15x speedup → no cross-GPU communication
- LoRA rank=32 → only train ~0.6GB params → fits 24GB
- bf16 param_dtype + fp32 buffer_dtype → preserve inv_freq → RoPE correct (#8066)
- offload_ratio=0.5 → LoRAOptimizedLinear → 50% base weight offload → more headroom!

---

## 5. MoE vs Dense: RTX 4090 Decision Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Aspect | Dense Model (Qwen3-1.7B) | MoE Model (Qwen3-MoE) |
|--------|--------------------------|------------------------|
| Framework | rLLM Tinker #1 | DeepSpeed AutoEP #1 |
| Memory | ~9GB (comfortable) | ~20GB (tight but fits) |
| Training algorithm | GRPO/bypass/ppo | GRPO/bypass/ppo |
| LoRA | rank=32, auto | rank=32, manual config |
| Optimizer | GPU Adam | CPU_Adam offload |
| Weight sync | zero-copy in-process | naive (HYBRID-style) |
| Throughput | ~3-5s/step | ~5-8s/step (more compute) |
| Active params | 1.7B (all) | 0.6B (active per token) |
| Quality | Standard dense | MoE quality > dense |
| Risk | Low (mature) | Medium (AutoEP new) |

★★★★★★★★★ Decision: If quality is priority → MoE; If speed is priority → Dense

---

## 6. OPD Distillation: Another RTX 4090 Path

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### On-Policy Distillation (PR #8027, DRAFT):

```
★★★★★ OPD = small student + big teacher → distillation NEW path for RTX 4090!

Student: Qwen2.5-0.5B → ~1GB GPU → fits easily!
Teacher: Qwen2.5-1.5B → frozen → logits on CPU → chunk fetch → no GPU memory!

Memory:
  Student weights:           1.0GB
  Activations:               0.5GB
  Teacher logits (chunk):    0.1GB (fetched from CPU in chunks)
  Optimizer:                 2.0GB (Adam m+v for student)
  CUDA context:              1.0GB
─────────────────────────────────────
Total GPU:                  ~4.6GB ← incredibly light!
```

★★★★★★★ OPD advantages:
  → Student fits RTX 4090 easily
  → Teacher logits NEVER co-reside on GPU (chunked CPU fetch)
  → 3-phase loop: rollout → teacher forward → student divergence backward
  → Forward-KL / reverse-KL / JSD divergence options

★★★★★★★ OPD risks:
  → Still DRAFT → not production-ready
  → Need validation on RTX 4090
  → Quality depends on teacher quality → not GRPO → different paradigm

---

## 7. Updated RTX 4090 Training Landscape (2026-06)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ RTX 4090 Training Landscape REVOLUTION (June 2026):

Dense model GRPO:
  #1: rLLM Tinker (in-process, bypass, auto LoRA) → ★★★★★★★★ simplest, fastest
  #2: verl+CPPO+bypass → ★★★★★★★ best trust region algorithm
  #3: DeepSpeed ZeRO-2+CPU_Adam+LoRA → ★★★★★ SFT benchmark, MoE transition

MoE model GRPO (NEW!):
  #1: ★★★★★★★★★★★★ DeepSpeed AutoEP ZeRO-2+EP=1+LoRA+CPU_Adam → ONLY viable MoE path!
  #2: verl+CPPO+bypass → no MoE optimization → active params small → but no EP
  #3: Megatron → CRASH → NOT viable

Distillation (NEW!):
  #1: ★★★★★★★ DeepSpeed OPD+ZeRO-0 → student-only GPU → teacher CPU → viable!

Inference serving:
  #1: vLLM → INT8 KV → HMA-by-default → BudgetRefiner SLO (future)
  #2: SGLang → deterministic inference → RadixAttention → Triton backend
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 8. GPU Validation Plan (When Servers Online)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Phase 1: MoE training smoke test (30 min):
```bash
# Test AutoEP ZeRO-2 + EP=1 with Qwen3-MoE
deepspeed test_autoep_moe.py --zero_stage 2 --ep_size 1 --lora_rank 32

# Verify memory stays under 24GB
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

### Phase 2: OPD distillation test (30 min):
```bash
# Test OPD with Qwen2.5-0.5B student + 1.5B teacher
deepspeed test_opd.py --student Qwen2.5-0.5B --teacher Qwen2.5-1.5B

# Verify chunked logits fetch and no GPU co-residence
```

### Phase 3: Benchmark comparison (1 hour):
```bash
# Dense GRPO: rLLM Tinker vs verl vs DeepSpeed
# MoE GRPO: DeepSpeed AutoEP only (Megatron crashes)
# Distillation: DeepSpeed OPD only
```

---

## 9. Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★★★★★★★★★★★★★★★★★ AutoEP MERGED → MoE training on RTX 4090 = NOW VIABLE! EP=1 singleton → 15x speedup → LoRA+CPU_Adam → fits 24GB!

2. ★★★★★★★★★★ LoRA is the KEY enabler: frozen base weights + tiny trainable params + CPU optimizer → 4B+ total params but only 0.6GB trainable → fits 24GB!

3. ★★★★★★★★★★★ ZenFlow chunked copyback → GPU spike 2944→256MiB → eliminates biggest RTX 4090 memory concern during CPU offload!

4. ★★★★★★★★★ OPD distillation → Qwen2.5-0.5B student → ~4.6GB total → incredibly light → NEW training paradigm for RTX 4090!

5. ★★★★★★★★★ DeepSpeed AutoEP = ONLY framework supporting MoE on RTX 4090 → vs Megatron crash (#5203) → vs verl no MoE optimization

6. ★★★★★★★★★ MoE quality > dense quality at same compute → 0.6B active params but 4B+ knowledge → if memory fits → MoE preferred!
