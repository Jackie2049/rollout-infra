# RTX 4090 AI Infra Consulting Quick Reference Card

> 2026-06-16 | Compiled from 227+ notes across 7 frameworks
> Purpose: Quick-access reference for AI infra consulting on RTX 4090
> ★★★★★★ This card = your "贴身顾问" cheat sheet!

---

## Training Quick Reference

### Dense Model GRPO (Qwen3-1.7B)

| Framework | Config | Memory | Step Time | Ranking |
|-----------|--------|--------|-----------|---------|
| rLLM Tinker | LoRA32+bypass+group4 | ~9GB | ~3-5s | ★★★★★★★★★ #1 |
| verl+CPPO | bypass+CPPO+LoRA | ~10GB | ~5-8s | ★★★★★★★ #2 |
| verl+GRPO | bypass+LoRA | ~10GB | ~5-8s | ★★★★★★★ #3 |
| DeepSpeed ZeRO-2 | CPU_Adam+LoRA32 | ~14GB | ~10-15s | ★★★★★ #4 |

### MoE Model GRPO (Qwen3-MoE) — NEW!

| Framework | Config | Memory | Step Time | Ranking |
|-----------|--------|--------|-----------|---------|
| DeepSpeed AutoEP | EP=1+LoRA32+ZenFlow | ~11.8GB | ~5-8s | ★★★★★★★★★★★★ #1! |

### Distillation — NEW!

| Framework | Config | Memory | Ranking |
|-----------|--------|--------|---------|
| DeepSpeed OPD | 0.5B student+1.5B teacher | ~4.6GB | ★★★★★★★★★★★★ #1! |

---

## Inference Quick Reference

### Serving (Qwen3-1.7B)

| Framework | Config | Memory | Throughput | Notes |
|-----------|--------|--------|------------|-------|
| vLLM | INT8 KV+prefix cache+enforce_eager | ~5.4GB | ~1000-1500 tok/s | Batch invariant safe |
| SGLang | deterministic+Triton backend | ~5.4GB | ~800-1200 tok/s | Batch invariant by design! |

### Serving (Qwen3-8B)

| Framework | Config | Memory | Throughput | Notes |
|-----------|--------|--------|------------|-------|
| vLLM | INT4+INT8 KV+enforce_eager | ~6.5GB | ~300-500 tok/s | INT4 Marlin/Triton fallback |
| SGLang | INT4+deterministic | ~6.5GB | ~250-400 tok/s | Deterministic + INT4 |

---

## SM89 Must-Do / Must-Avoid

### MUST DO:
- `--enforce-eager` for batch invariance (vLLM) or `--enable-deterministic-inference` (SGLang)
- INT8 FlashInfer KV cache (only viable KV quantization on SM89)
- INT4 Marlin/Triton for 8B models (v0.23.0 fallback works on SM89)
- LoRA rank=32 for all training (mandatory for 24GB)
- bypass_mode=True for GRPO (save 14GB ref model)
- detach_metrics=True for verl (prevent OOM)

### MUST AVOID:
- FP8 KV cache (3 paths: Triton OK, FlashInfer BLOCKED, compressed-tensors CRASH)
- torch.compile on SM89 (Inductor fuses RMSNorm → batch-dependent results)
- ZeRO-3 on single GPU (meaningless overhead, no sharding)
- Full model training without LoRA (exceeds 24GB)
- Megatron core on single GPU (crash #5203)
- CUDA graphs on SM89 with compile (batch-dependent)

---

## Optimizer Quick Reference

| Optimizer | GPU Memory | Best For | RTX 4090? |
|-----------|-----------|---------|-----------|
| torch.Adam | 2Ψ | Dense LoRA (small Ψ) | ✓✓✓ Default safe |
| ZenFlow CPU_Adam | 0 + 0.25GB spike | MoE LoRA (tight 24GB) | ✓✓✓✓✓✓ BEST |
| DeepSpeed Muon | Ψ (1 state) | Experimental faster convergence | ✓✓✓ New |
| 8-bit Adam | Ψ/2 | Marginal benefit with LoRA | ✓ Not needed |

---

## OSS Contribution Priority

| ID | Priority | Title | Status | Unique? |
|----|----------|-------|--------|---------|
| BR1 | P10 | BudgetRefiner SLO → vLLM | Draft ready (GPU needed) | ★★★★★★★★★★★★★ |
| IF1 | P9 | Inductor SM<90 Fusion Guard → PyTorch | Draft ready (GPU needed) | |
| C1 | P8 | FP8 KV 3-path distinction comment | Draft ready | |
| C2 | P7 | Batch invariance root cause comment | Draft ready | |
| AE1 | P6 | AutoEP RTX 4090 MoE benchmark | GPU validation needed | ★★★★★★★★★★★★★ |
| FP4 | P3 | SM120 FP4/MXFP4 kernel | Future (RTX 5090) | ★★★★★★★★★★★★★ |

---

## GPU Validation Priority (When Servers Online)

1. Batch invariance: sm89_batch_invariance_repro.py (30 min)
2. INT8 KV throughput benchmark (30 min)
3. SGLang deterministic vs vLLM enforce_eager (30 min)
4. INT4 8B serving on SM89 (30 min)
5. BudgetRefiner profile_table.csv (1 hour) — P10 UNIQUE!
6. AutoEP MoE training smoke test (30 min) — P6 UNIQUE!
7. Inductor Fusion Guard validation (1 hour) — P9

---

## Framework Coverage Status

| Framework | Notes | Depth | Key Finding |
|-----------|-------|-------|-------------|
| DeepSpeed | 18 | ★★★★★★★★ Very thorough | AutoEP merged; ZenFlow BEST optimizer; OPD distillation; MoE viable! |
| Megatron-LM | 11 | ★★★★★★★ Very thorough | Singleton PG crash; Lite #4885 bitwise verified but RTX 4090 ✗ |
| vLLM | 42+ | ★★★★★★★★ Very thorough | BudgetRefiner P10 UNIQUE; batch invariance root cause; MRv2 safe for verl |
| verl | 32+ | ★★★★★★★★ Very thorough | CPPO+bypass RTX 4090最优; ServerAdapter-only v0.8; ReMax+IcePop |
| MindIE | 11 | ★★★★★★★ Very thorough | ATB compose unique; BudgetRefiner SLO; Ascend-only |
| rLLM | 11 | ★★★★★★ Good | Tinker in-process #1; client-server not pure in-process; 3 backends |
| SGLang | 9 | ★★★★★★★★ Very thorough | deterministic batch-invariant by design; Triton backend; murmur_hash32 |
| PyTorch | 17 | ★★★★★★★★ Very thorough | Inductor 3-layer fusion gate; SM<90 guard P9; Non-TMA templates |

---

## RTX 5090 Strategic (Future)

- SM120 + 32GB GDDR7 + FP4 native → changes consumer inference
- SM89 still matters (massive 4090 installed base) → continue contributions
- FP4/MXFP4 kernel = next-phase vLLM contribution window (P3)
- RTX 5090 D: inference near-full 5090, compute -33%
