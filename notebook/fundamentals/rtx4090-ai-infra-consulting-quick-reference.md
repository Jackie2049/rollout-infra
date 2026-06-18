# RTX 4090 AI Infra Consulting Quick Reference Card

> 2026-06-18 | Compiled from 678+ notes across 7 frameworks
> Purpose: Quick-access reference for AI infra consulting on RTX 4090
> ★★★★★★ This card = your "贴身顾问" cheat sheet!

---

## Training Quick Reference

### Dense Model GRPO (Qwen3-1.7B)

| Framework | Config | Memory | Step Time | Ranking |
|-----------|--------|--------|-----------|---------|
| verl+CPPO | bypass+CPPO+LoRA32+FSDP | ~16.2GB | ~5-8s | ★★★★★★★★★★★★★★ #1 BEST! |
| verl Tinker | LoRA32+bypass+TinkerPrimitives | ~16.2GB | ~5-8s | ★★★★★★★★★★★★ #1.5 |
| verl+GRPO | bypass+LoRA32+FSDP | ~16.2GB | ~5-8s | ★★★★★★★ #2 |
| DeepSpeed ZeRO-2 | CPU_Adam+LoRA32 | ~14GB | ~10-15s | ★★★★★ #2.5 |
| rLLM Tinker | LoRA32+bypass | ~9GB | ~3-5s | ★★★ #3 BLOCKED (#605 grouping bug → GRPO BROKEN!) |
| Megatron core | — | — | — | ★ #4 (crash on single GPU) |

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
- LoRA rank=32 for all training (mandatory for 24GB, rank=64 breaks EOS #6782!)
- bypass_mode=True for GRPO (save 14GB ref model)
- detach_metrics=True for verl (prevent OOM, #6699 4x reduction)
- overlap_comm=False on single GPU (#8061: overlap_comm+compile=NaN)
- gradient_clipping=1.0 for GRPO (#8068: default 0→stalls)
- FSDP v2 backend for verl LoRA (FSDP v1 whole-model summon=OOM on 4090!)
- verl FSDP backend ONLY (Automodel/Megatron/TorchTitan have detach leak #6699!)

### MUST AVOID:
- FP8 KV cache (3 paths: Triton OK, FlashInfer BLOCKED, compressed-tensors CRASH)
- torch.compile on SM89 (Inductor fuses RMSNorm → batch-dependent results)
- ZeRO-3 on single GPU (meaningless overhead, no sharding)
- Full model training without LoRA (exceeds 24GB)
- Megatron core on single GPU (crash #5203)
- CUDA graphs on SM89 with compile (batch-dependent)
- torch 2.13 upgrade (#187484: vLLM Inductor breaks → 150+ failures!)
- rLLM GRPO training (#605 grouping bug → group size=1 → advantage=raw reward → BROKEN!)
- LoRA rank=64 in verl (#6782: breaks EOS in vLLM rollout)

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
| IF1 | P9 | Inductor SM<90 Fusion Guard → PyTorch | Draft ready (GPU needed) | ★★★★★★★★★★★★ |
| C7 | P9 | rLLM #605 GRPO grouping fix comment | Draft ready | ★★★★★★★★★★★★ UNIQUE |
| C8 | P8 | Megatron #5394 cross-framework Muon clipping | Draft ready | ★★★★★★★★★★★ UNIQUE |
| C1 | P8 | vLLM FP8 KV 3-path distinction | Draft ready | |
| C2 | P7 | vLLM #39096 batch invariance root cause | Draft ready | |
| C3 | P7 | vLLM #44594 BudgetRefiner complementary | Draft ready | |
| C9 | P7 | verl #6699 unfixed backends detach fix | Draft ready (PR) | ★★★★★★★★★★★ UNIQUE |
| C10 | P7 | vLLM #45683 MoE deterministic combine | Draft ready | ★★★★★★★★★★★ UNIQUE |
| AE1 | P6 | AutoEP RTX 4090 MoE benchmark | GPU validation needed | ★★★★★★★★★★★★★ |
| TK1 | P6 | Triton dequant_swiglu_quant kernel | After Fusion Guard | ★★★★★★★★★★★★ |
| QK1 | P4 | QuantKey refactor → vLLM | Foundation work | |
| OPD1 | Tier 2 | DeepSpeed OPD+LoRA gap ~15 LOC | Comment | |
| RR1 | Tier 2 | DeepSpeed RouterReplay ~300 LOC | Design ready | |
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
| DeepSpeed | 28 | ★★★★★★★★ Very thorough | AutoEP merged; ZenFlow BEST optimizer; OPD distillation; MoE viable! ZeRO-3+PEFT regression #8072 |
| Megatron-LM | 24 | ★★★★★★★ Very thorough | Singleton PG crash; MFSDPv2 #5387; skip_grad_norm_clip #5395; Muon 4 blockers |
| vLLM | 47+ | ★★★★★★★★ Very thorough | BudgetRefiner P10 UNIQUE; v0.23.0; MLA DCP #45964; MoE combine #45683 |
| verl | 42+ | ★★★★★★★★ Very thorough | CPPO+bypass #1 RTX 4090; #6699 detach 4x reduction; #6782 rank=64 breaks EOS; #6512 per-unit LoRA |
| MindIE | 15 | ★★★★★★★★ Very thorough | BudgetRefiner origin (P10 source!); ProfilingChunk quadratic; dual-backend deterministic; MXFP4 |
| rLLM | 14 | ★★★★★ Good but BLOCKED | #605 GRPO grouping bug BROKEN; #663 all rewards=0.0 before June 17; Tinker in-process |
| SGLang | 14 | ★★★★★★★★ Very thorough | deterministic KERNEL-level; #28582 RCE source verified; #27097 multi-LoRA fragility |
| PyTorch | 25+ | ★★★★★★★★ Very thorough | #184119 SM89 guard progressing; #187484 torch 2.13 breaks Inductor; #187620 partial offload multi-GPU only |

---

## RTX 5090 Strategic (Future)

- SM120 + 32GB GDDR7 + FP4 native → changes consumer inference
- SM89 still matters (massive 4090 installed base) → continue contributions
- FP4/MXFP4 kernel = next-phase vLLM contribution window (P3)
- RTX 5090 D: inference near-full 5090, compute -33%
