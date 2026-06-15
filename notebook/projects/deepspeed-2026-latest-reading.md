# DeepSpeed 2026 Latest Developments — Comprehensive Update

> 2026-06-16 | Source: GitHub microsoft/DeepSpeed releases v0.18.4-v0.19.2 + 50+ PRs + Q2 Roadmap #7861
> Current version: v0.19.2 | Stars: 42,524 | Open issues: 1,303 | Default branch: master
> 8 releases since January 2026 (v0.18.4 through v0.19.2)
> Core theme: AutoEP merged, AutoSP+multimodal, DeepCompile maturing, ZenFlow native process, Muon optimizer, 9th accelerator (Biren)

---

## 1. Release Timeline (January-June 2026)

| Version | Date | Key Highlights |
|---------|------|---------------|
| v0.18.4 | 2026-01-07 | DeepCompile PT 2.8/2.9 fix, AMD ROCm, BF16 fallback |
| v0.18.5 | 2026-01-30 | MPS fixes, sequential allgather (#7661), gradient checkpointing reentrant fix |
| v0.18.6 | 2026-02-12 | AutoTP custom partitioning (#7806), torch.jit.script replaced with torch.compile, bf16 gradient norm fix |
| v0.18.7 | 2026-03-05 | Z1/2 flatten params on device, EXAONE 4.0 inference, ARM shm_comm, hook count perf regression fix |
| v0.18.8 | 2026-03-13 | AGENTS.md/CLAUDE.md added, Evoformer fixes, fp16 loss_scale validation |
| v0.18.9 | 2026-03-30 | AutoSP merged (#7860), HF tp_plan support (#7901), Muon ZeRO-3 (#7919), ASPLOS 2026 Best Paper |
| v0.19.0 | 2026-05-06 | 28+ PRs: Muon Gram Newton-Schulz (#7953), DeepCompile PT 2.9/2.10 (#7951), AutoSP multimodal (#7984), ZenFlow backward fix (#7980), topkgating bug fix (#7986) |
| v0.19.1 | 2026-05-27 | Singleton MoE (#7997), SDMA allgather (#7999), DeepCompile PT 2.11 (#8024), BF16 optimizer states CPU offload (#8010), torch.func/vmap (#7916/#8023) |
| v0.19.2 | ~2026-06-14 | AutoEP (#7938 merged 6/11), ZeRO-3 grad dtype (#8063) |

---

## 2. ★★★★★★★ AutoEP — Merged! (PR #7938, merged 2026-06-11)

**This is the single most important DeepSpeed feature of 2026. AutoEP is now merged and production-ready for ZeRO-0/1/2.**

### Current scope (merged):
- ZeRO stages 0, 1, and 2 support
- Checkpoint save/load support
- Universal checkpoint conversion support
- 4 preset MoE models: Mixtral, Qwen3-MoE, DeepSeek-V2, DeepSeek-V3
- TorchTitan-derived kernels: TokenChoiceTopKRouter, GroupedExperts, grouped-GEMM, Triton fill-indices
- ~5x throughput improvement over ZeRO-3 baselines (validated on 8xH100)

### ★★★★★ AutoEP ZeRO-3 follow-up (PR #8060, OPEN):
```
★★★★★ PR #8060 — Support AutoEP with ZeRO-3 zero.Init source modules
→ Expert parameters partitioned over expert replica groups (NOT global DP)
→ Router and replicated parameters use global data-parallel group
→ Solves double-partitioning problem correctly!
→ Status: OPEN (created 2026-06-11)
→ This is the correct approach vs. previous workaround attempts
```

### ★★★★★ AutoEP + AutoTP parallel folding (PR #8064, OPEN):
```
★★★★★ PR #8064 — Add AutoEP + AutoTP parallel folding
→ TP(attention/dense) + EP(expert) coexist on same rank set!
→ Dense/attention: stage_size = tp * dp
→ Routed-expert: stage_size = ep * etp * edp
→ Invariant: tp * dp == ep * etp * edp == stage_size
→ 8 GPU example: TP=2 + EP=4 → dense 2-card TP + expert 4-card EP
→ expert_tensor_parallel_size currently MUST be 1 (reserved as follow-up)
→ Depends on #8060 (ZeRO-3 support) being merged first
```

### ★★★★★★★ RTX 4090 AutoEP feasibility:
- EP=1 on single GPU: All experts same GPU, no AllToAll, singleton MoE path (#7997) gives 15x speedup
- Qwen3-MoE (A0.6B + B4B) + LoRA + AutoEP ZeRO-2: RTX 4090's ONLY viable MoE training path
- EP>1 requires cross-GPU: PCIe bandwidth disaster for RTX 4090
- AutoEP ZeRO-3 (#8060) for single GPU: expert params NOT partitioned (singleton replica group), only router params sharded — limited benefit

---

## 3. ★★★★★ AutoSP — Merged + Multimodal Extension

### AutoSP base merged (PR #7860, merged 2026-03-30):
- Compiler-based Sequence Parallelism (ICLR 2026 paper)
- `prepare_autosp_inputs()` API + `pass_shard_seq_dim()` pass
- Auto-detects sequence dimension, shards, inserts AllGather/ReduceScatter
- Prevents graph breaks during `torch.compile()`

### ★★★★★ AutoSP multimodal ViT+LLM (PR #7984, merged 2026-05-01):
```
★★★★★ AutoSP for Multimodal Models (ViT + LLM)
→ Addresses Q2 2026 roadmap item explicitly
→ 3 key features:
  1. AutoSP scaffolding + detector: auto-detect ViT encoder + LLM decoder
  2. ViT Sequence Parallelism (UlyssesSPViTAttention): Gather-Compute-Scatter for non-causal ViT
  3. Cross-modal Fusion Adapters: complex scatter/gather at vision-language boundary
     → LlavaFusionAdapter (visual token splice)
     → InternVLFusionAdapter (IMG_CONTEXT token splice)
     → Qwen2VLFusionAdapter (vision_start/end bounded splice)
→ Significantly reduces ViT FFN + LayerNorm memory across sequence dimension
→ ★★★ RTX 4090: ViT+LLM SP = multi-GPU only → single GPU not applicable
   But: the multimodal architecture knowledge is valuable for understanding model structure
```

---

## 4. ★★★★★ DeepCompile — Maturing (5 fixes merged since April)

DeepCompile is now compatible with PyTorch 2.6 through 2.12 (with one open issue for 2.12).

### Compatibility fixes merged:
| PR | Fix | Date | PyTorch Version |
|----|-----|------|-----------------|
| #7951 | DeepCompile PT 2.9/2.10 guard_sizes_strides | 2026-04-11 | PT 2.9/2.10 |
| #8024 | DeepCompile AOT kwargs PT >= 2.11 | 2026-05-25 | PT >= 2.11 |
| #8032 | ZeRO-3 release parameter lifetime | 2026-06-02 | All |
| #8033 | All-gather scheduler candidate selection | 2026-06-02 | All |
| #8036 | ZeRO-1 grad target lifetime | 2026-06-09 | All |
| #8038 | Normalize ZeRO-3 grad dtype before reduction | 2026-06-09 | All |

### ★★★★ DeepCompile dynamo skipped frames (PR #8059, OPEN):
```
★★★★ PR #8059 — Fix gather params in dynamo skipped frames for ZeRO3
→ Root cause: torch._dynamo may skip entire frames → skipped frames execute eagerly → no gathering mechanism → parameters stay partitioned at shape [0]
→ Validated on 2xH100: Qwen2 MoE (was crashing), LLaMA (regression pass), tied embeddings (pass)
→ 22 compilations unchanged (no guard instability)
→ ★★★★★ This is a CRITICAL fix → Qwen2 MoE was crashing with DeepCompile → now fixed!
→ Status: OPEN → needs review
```

### ★★★★★ DeepCompile compatibility matrix (updated):
| PyTorch Version | ZeRO-3 DeepCompile | ZeRO-1 DeepCompile | Notes |
|----------------|--------------------|--------------------|-------|
| 2.6 | Working | Working | `_backward_prologue` API |
| 2.7-2.8 | Working | Working | `_backward_impl` API |
| 2.9-2.10 | Working (#7951 merged) | Working | guard_sizes_strides enforced |
| 2.11+ | Working (#8024 merged) | Working | AotAutograd kwargs API changed |
| 2.12 | Partial (#8059 open) | Working | Dynamo skipped frames may cause issues |

### RTX 4090 DeepCompile assessment:
- DeepCompile + ZeRO-3 single GPU: NOT viable (ZeRO-3 needs multi-GPU for partitioning)
- DeepCompile + ZeRO-2 single GPU: NOT meaningful (no partitioning on single GPU)
- DeepCompile + H100 NVLink: Best scenario (all optimizations fully utilized)
- Learning value remains extremely high: compiler-level distributed optimization approach

---

## 5. ★★★★★ ZenFlow — Native CPU Optimizer Process (PR #8058, OPEN)

**This is a potentially transformative change for CPU-offloaded training.**

```
★★★★★ PR #8058 — ZenFlow: run the overlapped CPU optimizer in a native process
→ Previously: Python multiprocessing subprocess + pickling Pipe coordination
→ NOW: native CPU optimizer process packaged inside cpu_adam op
→ Shared-memory POSIX-semaphore control block (replaces pickling)
→ Adam state allocated in native process, NUMA-local to optimizer pinned thread pool

Key improvements:
1. Fused multi-tensor CPU Adam (adam_update_multi): whole flattened partition in C++
   → removes per-parameter Python-to-C++ loop + Python-side clone()
2. ZenFlowAdam native class: pinned std::thread pool on dedicated cores
   → driven via shared-memory control block (run_worker / submit / wait)
3. Covers ZeRO stages 1, 2, AND 3 (removes old pickling subprocess entirely)
4. Chunked copyback: streaming fp32 master partition to GPU bit16 in chunks
   → GPU spike: ~2944 MiB → ~256 MiB for 0.75B-param partition!
   → Old fp32.to(device) materialized whole fp32 partition on GPU first
5. ZenFlowCPUAdam now a recognized ZeRO optimizer → zero_allow_untested_optimizer NO longer required!

★★★★★★★★★ RTX 4090 IMPACT:
→ Chunked copyback 2944→256 MiB → avoids GPU memory spike → CRITICAL for 24GB!
→ CPU_Adam offload = real benefit → C++ SIMD AVX512 + OpenMP → 5-7x faster than torch.optim.Adam on CPU
→ Native process → NUMA-local → even better than current subprocess approach
→ ZeRO-2 + CPU_Adam + ZenFlow native → optimizer offload with no GPU memory spike → viable!
```

★★★★ Status: OPEN → still needs review → but direction is very promising
→ Based on top of #7771 (ZenFlow NaN fix) → dependency chain
→ Bit-identical to previous subprocess path → correctness verified
→ Qwen2.5-1.5B end-to-end validated

---

## 6. ★★★★★ Muon Optimizer — ZeRO-3 Support + Gram Newton-Schulz

### ★★★★ Muon Gram Newton-Schulz (PR #7953, merged 2026-04-30):
```
★★★★ PR #7953 — Add Gram Newton-Schulz orthogonalization for Muon optimizer
→ Standard NS iterates on full rectangular matrix X (n x m)
→ Gram NS iterates on much smaller Gram matrix R = X @ X.T (n x n)
→ When m >> n (transformer weight matrices, aspect ratio ~5) → significantly cheaper!
→ Based on: https://tridao.me/blog/2026/gram-newton-schulz/
→ Now DEFAULT orthogonalization method for Muon
→ Configurable ns_method switch to fall back to original iteration

★★★ RTX 4090 impact:
→ Muon → new optimizer → potentially faster convergence than Adam
→ Single GPU Muon → no partitioning → just optimizer → viable!
→ Gram NS → n x n matrix → 5x cheaper → memory efficient → RTX 4090 friendly!
```

### ★★★ Muon ZeRO-3 support (PR #7919, merged in v0.18.9):
- Extends Muon optimizer support to ZeRO Stage 3
- Muon now works across all ZeRO stages (0/1/2/3)

### ★★★★ ASPLOS 2026 Best Paper Award:
```
★★★★ DeepSpeed team received ASPLOS 2026 Best Paper Award
→ PR #7923 added news item for this recognition
→ ★★★ Signals that DeepSpeed research is being recognized at top systems conferences
→ This validates DeepSpeed's continued relevance in the AI systems landscape
```

---

## 7. ★★★★★ coalesce_grad_reduction — Multi-backward Optimization (PR #7992, merged)

```
★★★★★ PR #7992 — Add engine.coalesce_grad_reduction() for ZeRO 1/2/3
→ Context manager that defers gradient reduction across multiple engine.backward() calls
→ N reduce-scatters collapse into 1 → removes communication bottleneck
→ Third step of multi-backward feature:
  - #7665: set_gradient_accumulation_boundary() + manual engine.backward() as first-class API
  - #7981: fix silent gradient-loss bug for ZeRO-1/2 with cpu_offload
  - #7992: make the path efficient → 1 reduction instead of N

★★★★★ RTX 4090 IMPACT:
→ coalesce_grad_reduction → single GPU → no reduce-scatter needed (dp=1)
→ But: API design important → multiple backward calls → contrastive learning!
→ Cached contrastive learning (GradCache): sentence-transformers
→ ★★★★★★★ THIS IS RELEVANT FOR RTX 4090!
→ ZeRO-2 single GPU + coalesce_grad_reduction → gradient accumulation efficiency!
```

---

## 8. ★★★★★ New Accelerators & Hardware Support

### ★★★★ Biren SUPA accelerator (PR #8054, OPEN):
```
★★★★ PR #8054 — Add Biren SUPA accelerator support
→ 9th supported accelerator! (cuda/cpu/xpu/npu/mps/hpu/mlu/sdaa/SUPA)
→ Biren Technology GPU + SUPA software stack
→ Zero intrusion into existing backends
→ torch_supa spoofs torch.cuda → SUPA detection MUST come before CUDA detection!
→ ★★★★★ Biren = Chinese domestic GPU → DeepSpeed supporting Chinese hardware ecosystem
```

### Other hardware:
- MPS (Apple Silicon) fixes — v0.18.5
- ARM shm_comm support — v0.18.7
- Huawei Ascend NPU async_io fix — v0.18.8
- EXAONE 4.0 model support (Korean LG AI) — v0.18.7

---

## 9. ★★★★★ Mixed-Precision Per-Policy (PR #8066, OPEN)

```
★★★★★ PR #8066 — Mixed-precision: per-policy param/buffer dtype cast
→ Add data_types.param_dtype and data_types.buffer_dtype (mirroring FSDP2 MixedPrecisionPolicy)
→ Previously: blanket module.half()/module.bfloat16() → downcasts EVERY floating buffer
→ NOW: parameters go to param_dtype; floating buffers keep loaded dtype unless buffer_dtype set
→ Motivation: bf16 inv_freq loses precision on long contexts → RoPE angles drift → logits diverge
→ ★★★★★ This matches FSDP2 behavior → DeepSpeed aligning with PyTorch standard!
→ ★★★★★ RTX 4090: BF16 inv_freq precision loss → long context training → relevant!
→ buffer_dtype unset → buffers keep loaded dtype (e.g. fp32 inv_freq) → correct RoPE
```

---

## 10. ★★★★ OPD Trainer — On-Policy Distillation (PR #8027, DRAFT OPEN)

```
★★★★ PR #8027 — Add On-Policy Distillation (OPSD) Trainer in DeepSpeed
→ Small student generates rollouts → frozen large teacher scores them
→ Student updated by per-token divergence (forward-KL / reverse-KL / JSD)
→ 3-phase loop: student rollout → teacher forward + CPU logit cache → student forward + streamed divergence + backward
→ Full [B, T, V] teacher tensor never co-resides with student logits → memory efficient!
→ 32 files, ~3.2k LOC
→ Chunked/streamed losses → sequence-axis chunking
→ TeacherLogitCache: host-resident, chunk fetch
→ RolloutEngine ABC + hybrid_engine + vllm backends
→ weight_bridge: per-arch ParallelKind + TP slicer for vLLM weight sync (qwen2/qwen3)
→ 87 CPU-only tests
→ Validated: Qwen2.5-0.5B-Instruct student + Qwen2.5-1.5B-Instruct teacher on 2xH200

★★★★★ RTX 4090 IMPACT:
→ OPD = small student + big teacher → student fits in RTX 4090!
→ Teacher logits on CPU → chunk fetch → no GPU memory for teacher logits!
→ ★★★★★ Qwen2.5-0.5B student → ~1GB → RTX 4090 easily fits!
→ OPD + DeepSpeed ZeRO-0 → no partitioning → student-only on GPU → viable!
→ But: still draft → not production-ready yet
```

---

## 11. ★★★ ZeRO-3 Elastic Checkpoint (PR #8031, DRAFT OPEN)

```
★★★ PR #8031 — Add ZeRO-3 elastic checkpoint save/load support (DRAFT)
→ Currently: ZeRO-3 checkpoints tied to DP world size → different DP size → loading fails
→ NOW: save with one DP world size → resume with another
→ Re-merges and re-partitions optimizer state for current DP world size
→ Auto-detects elastic vs. rigid checkpoint format on load
→ ★★★ RTX 4090: single GPU → DP=1 → checkpoint always compatible → less relevant
→ But: if scaling from single GPU to multi-GPU → elastic checkpoint enables migration!
```

---

## 12. ★★★★★ Comprehensive RTX 4090 Impact Analysis

### ★★★★★★★ NEW RTX 4090-relevant findings (since previous analysis):

| Finding | Rating | RTX 4090 Impact |
|---------|--------|----------------|
| ZenFlow native process (#8058) | ★★★★★★★ | Chunked copyback 2944→256 MiB → avoids GPU spike → CRITICAL for 24GB |
| Mixed-precision per-policy (#8066) | ★★★★★ | BF16 inv_freq precision → long context training → RoPE drift fix |
| coalesce_grad_reduction (#7992) | ★★★★★ | Multi-backward efficiency → contrastive learning → even single GPU benefits |
| OPD Trainer (#8027) | ★★★★★ | Small student + big teacher → Qwen2.5-0.5B fits RTX 4090 → distillation viable! |
| Muon Gram NS (#7953) | ★★★★ | n x n iteration → 5x cheaper → memory efficient optimizer for single GPU |
| AutoEP merged (#7938) | ★★★★★★★ | EP=1 singleton path → 15x speedup → Qwen3-MoE LoRA training viable |

### ★★★★★★★ Updated RTX 4090 optimal DeepSpeed configuration (2026-06):
```
★★★★★★★★ RTX 4090 DeepSpeed BEST configurations (updated):

Training (dense model):
  → PyTorch compile + LoRA → single GPU optimal (no DeepSpeed)
  → DeepSpeed ZeRO-2 + CPU_Adam → multi-GPU optimal

Training (MoE model) — NEW:
  → ★★★★★★★ DeepSpeed AutoEP ZeRO-2 + LoRA + EP=1 + CPU_Adam + coalesce_grad_reduction
  → Qwen3-MoE (A0.6B+B4B) → fits 24GB → singleton MoE path → 15x faster
  → This is NOW the recommended RTX 4090 MoE training approach!

Distillation — NEW:
  → ★★★★★ OPD Trainer + ZeRO-0 + small student (Qwen2.5-0.5B)
  → Teacher logits on CPU → chunk fetch → student-only on GPU
  → Viable on RTX 4090! But still draft → not production-ready yet

CPU Offload improvements — NEW:
  → ★★★★★★★ ZenFlow native process + chunked copyback
  → GPU spike reduction: 2944→256 MiB → massive improvement for 24GB!
  → ZeRO-2 + CPU_Adam + ZenFlow → best CPU offload path for RTX 4090

Long context — NEW:
  → ★★★★★ Mixed-precision per-policy → preserve fp32 inv_freq → correct RoPE
  → BF16 training with fp32 buffers → no RoPE drift → important for Qwen2.5 long context
```

### ★★★★★★★ RTX 4090 DeepSpeed ranking vs. alternatives (updated):
```
★★★★★★★★ RTX 4090 GRPO/Training Framework Ranking (2026-06):

Dense model training:
  #1: PyTorch compile + LoRA (single GPU, no DeepSpeed overhead)
  #2: DeepSpeed ZeRO-2 + CPU_Adam (multi-GPU, offload efficient)
  #3: rLLM Tinker (in-process, bypass_mode, auto LoRA)

MoE model training:
  #1: ★★★★★★★ DeepSpeed AutoEP ZeRO-2 + EP=1 + LoRA + CPU_Adam (NEW! viable now!)
  #2: verl + CPPO + bypass (no MoE-specific optimizations)
  #3: Megatron (CRASH on single GPU → NOT viable)

Distillation:
  #1: ★★★★★ DeepSpeed OPD + ZeRO-0 (NEW! small student viable on RTX 4090)
  #2: Custom implementation (no framework support)

Inference:
  #1: vLLM (serving optimized)
  #2: DeepSpeed Inference (limited SM89 support)
```

---

## 13. ★★★★★★★ Key Insight Summary

### ★★★★★★★ 7 Key Insights (NEW since previous analysis):

1. **★★★★★★★ AutoEP MERGED → DeepSpeed is now the BEST framework for MoE training on HuggingFace**
   - Zero code modification → HuggingFace compatible → just JSON config
   - vs Megatron: manual code changes → FlexTokenDispatcher → crash on single GPU
   - vs verl: no native MoE optimization
   - RTX 4090: EP=1 singleton path → viable!

2. **★★★★★★ ZenFlow native CPU optimizer process → GPU spike reduction is TRANSFORMATIVE for 24GB GPUs**
   - Chunked copyback: 2944 MiB → 256 MiB → 11.5x reduction
   - This directly addresses the biggest RTX 4090 concern: memory spikes during CPU offload
   - If merged, this makes ZeRO-2 + CPU_Adam + ZenFlow the optimal RTX 4090 path

3. **★★★★★★ OPD Trainer → DeepSpeed entering distillation market → RTX 4090 viable**
   - Small student + big teacher → teacher on CPU → student-only on GPU
   - NEW training paradigm that DeepSpeed is uniquely positioned for
   - 3-phase loop avoids GPU memory co-residence of teacher/student logits

4. **★★★★★ Mixed-precision per-policy → DeepSpeed aligning with FSDP2 → long context precision fix**
   - Preserve fp32 buffers (inv_freq) → RoPE angles correct → logits don't drift
   - Matches FSDP2 MixedPrecisionPolicy → DeepSpeed catching up to PyTorch standard
   - RTX 4090 long context training directly benefited

5. **★★★★★ DeepCompile maturing rapidly → 6 fixes merged → PT 2.6-2.11 all working**
   - But RTX 4090 still cannot benefit (needs multi-GPU ZeRO-3)
   - Learning value remains extremely high: compiler-level distributed optimization

6. **★★★★ Muon optimizer → ASPLOS 2026 Best Paper → Gram NS → viable for single GPU**
   - Gram Newton-Schulz: n x n iteration → 5x cheaper than standard NS
   - Muon + single GPU → no ZeRO needed → just use optimizer directly
   - RTX 4090: potentially faster convergence than Adam → worth experimenting!

7. **★★★★★ DeepSpeed expanding hardware ecosystem → 9 accelerators → Biren SUPA**
   - CUDA, CPU, XPU (Intel), NPU (Ascend), MPS (Apple), HPU (Habana), MLU (Cambricon), SDAA, SUPA (Biren)
   - Chinese GPU vendors actively supported → DeepSpeed has broad hardware strategy
   - RTX 4090: NVIDIA CUDA → primary platform → well supported

---

## 14. Source References

- GitHub microsoft/DeepSpeed releases: v0.18.4 through v0.19.2
- PR #7938: AutoEP (merged 2026-06-11)
- PR #8060: AutoEP ZeRO-3 support (OPEN)
- PR #8064: AutoEP + AutoTP parallel folding (OPEN)
- PR #8058: ZenFlow native CPU optimizer process (OPEN)
- PR #8066: Mixed-precision per-policy (OPEN)
- PR #7997: Singleton MoE collectives (merged)
- PR #7992: coalesce_grad_reduction (merged)
- PR #7951: DeepCompile PT 2.9/2.10 (merged)
- PR #8024: DeepCompile PT >= 2.11 (merged)
- PR #8059: DeepCompile dynamo skipped frames (OPEN)
- PR #7953: Muon Gram Newton-Schulz (merged)
- PR #7919: Muon ZeRO-3 support (merged)
- PR #8027: OPD Trainer (DRAFT OPEN)
- PR #8054: Biren SUPA accelerator (OPEN)
- PR #8031: ZeRO-3 elastic checkpoint (DRAFT OPEN)
- PR #7860: AutoSP merged
- PR #7984: AutoSP multimodal ViT+LLM (merged)
- PR #7661: Sequential allgather (merged)
- PR #7923: ASPLOS 2026 Best Paper news
- Issue #7861: DeepSpeed Q2 2026 Roadmap
