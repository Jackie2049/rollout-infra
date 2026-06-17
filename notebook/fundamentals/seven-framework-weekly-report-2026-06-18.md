# 7-Framework Weekly Report — June 18, 2026

> 2026-06-18 | Cross-framework weekly status | All 7 frameworks: latest developments + RTX 4090 implications
> ★★★★★★★★ Source: 6 background research agents (Megatron, verl, DeepSpeed, PyTorch, vLLM, MindIE/vLLM-Ascend)
> ★★★★★★★★ Key: vLLM v0.23.0 Helion rms_norm kernel, DeepSpeed v0.19.2 coalesce_grad, Megatron #5391 compact DDP
> ★★★★★★★★ rLLM #605 GRPO grouping bug = CRITICAL correctness blocker (downgraded from #1 ranking!)

---

## 1. vLLM — v0.23.0 Released June 15

```
★★★★★★★★★ vLLM v0.23.0 — 408 commits, 200 contributors (63 new):

Key RTX 4090 relevant:
  → #34432 MERGED June 17: Helion kernel for rms_norm_dynamic_per_token_quant → 6750 additions
    → Autotuned on H200/B200 → but SM89? → needs checking → may NOT benefit RTX 4090 directly
    → But: proves rms_norm kernel optimization is active area → P9 + Triton swiglu aligns
  → #43731 MERGED: Triton W4A16 CUDA fallback for non-Marlin shapes → SM80/SM89 benefit!
    → Shapes like intermediate_size=2112 → Marlin rejects → Triton W4A16 works → RTX 4090 viable
  → #44220 MERGED: Triton MoE backend default on Hopper → ~3.7% throughput gain
  → #43014 MERGED: MoE-permute buffer pre-allocation → +9-14% gain
  → #43706 MERGED June 1: CUTLASS FP8 scaled-mm padding bypass → +20% kernel perf
  → MRv2 default for Llama/Mistral dense → FlashInfer sampler → breakable CUDA graphs

★★★★★★★★★ BudgetRefiner-related:
  → No direct BudgetRefiner PR → still unique contribution opportunity!
  → Triton MoE backend + MoE-permute pre-allocation = MoE serving improvements
  → vLLM V1 scheduler still uses BudgetRefiner concept → but no OSS PR submitted

★★★★★★★★★ v0.22.1 (June 5): patch release → bug fixes + Mellum v2 + zentorch AMD Zen

★★★★★★★★★ #45731: PyTorch 2.13.0 → Triton 3.7.1 → DRAFT → 0 comments → makes Fusion Guard MORE needed

★★★★★★★★★ RTX 4090 impact:
  → W4A16 Triton fallback → directly benefits SM89 → quantized serving viable
  → Helion rms_norm → may NOT benefit SM89 (autotuned for Hopper) → P9 still needed
  → Triton MoE → SM89 gets Triton backend for MoE → closer to SGLang parity
```

---

## 2. DeepSpeed — v0.19.2 Released June 16

```
★★★★★★★★★ DeepSpeed v0.19.2 — 27 merged PRs in 3 weeks:

Key RTX 4090 relevant:
  → #7992 MERGED May 28: engine.coalesce_grad_reduction() → ZeRO 1/2/3 multi-backward
    → Collapses N reduce-scatters into 1 → removes communication bottleneck
    → ★★★★★★★★ GRPO multi-loss backward patterns benefit!
  → #8066 MERGED June 16: Mixed-precision per-policy dtype cast → preserves fp32 buffers
    → BUT: introduced ZeRO-3+PEFT regression → ★★★★★★★★ #8072/#8073 2-line fix!
  → #8072/#8073 NEW June 17: ZeRO-3+LoRA regression → TypeError in _allgather_params_coalesced
    → Root cause: #8066's dtype cast stopped blanket bf16 → mixed dtypes (bf16 base + fp32 LoRA)
    → ★★★★★★★★ ZeRO-2 UNAFFECTED → RTX 4090 ZeRO-2+LoRA SAFE!
    → Fix: per-param dtype in allgather (2 lines) → workaround: autocast_adapter_dtype=False
  → #8058 OPEN: ZenFlow native CPU optimizer process (954 additions) → reduces GPU spike 2944→256 MiB → chunked copyback → replaces pickling subprocess
    → ★★★★★★★★ CPU_Adam efficiency improvement → RTX 4090 benefit!
  → AutoEP #7938 MERGED June 11 → config-only MoE → RTX 4090 viable!

★★★★★★★★★ #8064 AutoEP+AutoTP folding — OPEN (2690 additions):
  → TP for dense + EP for MoE on same GPU set → preparation for future multi-GPU
  → RTX 4090: TP=1+EP=1 = identity → no change → LOW priority

★★★★★★★★★ #8068 gradient_clipping — OPEN → MUST set explicitly to 1.0
★★★★★★★★★ #8061 overlap_comm NaN — OPEN → MUST overlap_comm=False on single GPU

★★★★★★★★★ RTX 4090 impact:
  → #7992 coalesce_grad_reduction → GRPO benefit → fewer reduce-scatters → faster
  → #8066 dtype preservation → fp32 buffers preserved → numerical correctness
  → ZeRO-3+PEFT regression from #8066 → WATCH → may affect future LoRA+ZeRO-3
```

---

## 3. Megatron-LM — #5391 Compact LayerWise DDP

```
★★★★★★★★★ #5391 OPEN (+218/-58) — Compact LayerWise DDP:
  → Removes dp_size * max(shard_load) padding → per-buffer use_distributed_optimizer
  → RTX 4090: dp=1 → padding = max(shard_load) → wasteful → compact removes entirely
  → ★★★★★★★★ Makes Muon+ZeRO-2 more memory-efficient on RTX 4090

★★★★★★★★★ #5394 NEW (June 17) — ChainedOptimizer Muon clipping stalls:
  → Global grad_norm across all sub-optimizers → Muon orthogonalization degenerates
  → ★★★★★★★★ SAME pattern as DeepSpeed #8068/#7776 → MUST skip global clipping for Muon groups!
  → Positive-feedback stall: layers stop updating → grad_norm grows → clip coefficient shrinks → collapse

★★★★★★★★★ #5386/#5384 NEW (June 17) — DSA/DSv4 Indexer Replay for RL (3404 LOC):
  → Extends RouterReplay concept to sparse attention indexer top-k decisions
  → ★★★★★★★★ For DeepSeek V4 hybrid attention → same train/rollout consistency as MoE RouterReplay!
  → Same singleton-registry architecture as RouterReplay → established pattern

★★★★★★★★★ #5219 Final Review — single-GPU Muon crash fix (+14/-7):
  → None guard for dp_cp_params_list → simple but essential for single GPU

★★★★★★★★★ #4885 Lite — MERGED → 29K additions → Qwen3 MoE + HF safetensors
★★★★★★★★★ core_v0.17.1 (May 28) — latest release

★★★★★★★★★ RTX 4090 impact:
  → #5391 compact DDP → removes wasteful padding on single GPU → memory efficiency
  → #5219 close to merge → single-GPU Muon training viable soon
  → Lite package → agent-friendly path → lighter weight than core Megatron
```

---

## 4. verl — #6790 Separate Async Trainer + #6572 Reviews

```
★★★★★★★★★ verl #6790 MERGED June 17 — Separate async trainer (17 additions):
  → Colocated group never switches to rollouter → pure async pattern

★★★★★★★★★ verl #6699 MERGED June 12 — Detach model_output/loss metrics (★★★★★★★★★ CRITICAL RTX 4090!):
  → FSDPEngineWithLMHead.forward_step returns model_output (log_probs/entropy) STILL ATTACHED to autograd graph
  → forward_backward_batch holds per-micro-batch outputs in output_lst → graph pinned per micro-batch
  → Pinned: activation-checkpoint frame's saved embedding output (requires grad under PEFT enable_input_require_grads)
  → Verified memory timeline (Qwen3-8B LoRA rank 32): mb#250=24.8→#300=37.9→#350=~50→#400=64 GiB→OOM
  → After fix: STABLE 16.2 GiB throughout all micro-batches
  → ★★★★★★★★ 4x memory reduction → CRITICAL for RTX 4090 LoRA GRPO!
  → Regression test: test_engine_forward_step_detach_on_cpu.py → verified red/green

★★★★★★★★★ verl #6688 MERGED June 12 — Clone LoRA weights out of IPC buffer:
  → Fixed cudaErrorIllegalAddress crash with unmerged LoRA + vLLM + free_cache_engine=True
  → add_lora retained views into reused IPC bucket buffer → crash

★★★★★★★★★ verl #6717 MERGED June 15 — Tinker training worker primitives (334 additions):
  → TinkerTrainingWorker: optimizer_zero_grad(), forward_backward(), optimizer_step() split primitives
  → Enables rLLM Tinker-style split training IN verl → no need to switch frameworks!
  → Accumulated gradients across multiple forward_backward calls

★★★★★★★★★ verl #6689 MERGED June 17 — Offload process_vision_info to thread:
  → process_vision_info was CPU-bound (PNG decode + smart_resize) but declared async → blocked event loop

★★★★★★★★★ #6738 MERGED June 16 — Skip redundant clone SGLang weight sync (63 additions):
  → Critical OOM fix → clones only when necessary (contiguous check)
  → DTensor FSDP path: all-gather results already own tight storage → no clone needed
  → MoE fused weights (gate_up_proj/qkv) → multi-GiB → clone doubled memory → OOM fixed

★★★★★★★★★ #6779 OPEN — Continuous Token Agentic Rollout (5024 additions):
  → Still OPEN → not merged → CLA not yet signed
  → Builder layer + 7 model families + boundary handling

★★★★★★★★★ #6572 OPEN — Full determinism vLLM rollout (993 additions):
  → Two critical review concerns raised by wuxibin89 (collaborator)
  → Still OPEN → progressing → but concerns need addressing
  → ★★★★★★★★ On SM89: requires P9 for full determinism → VLLM_BATCH_INVARIANT has RMSNorm gap

★★★★★★★★★ v0.8.0 (June 1) — OPD, TransferQueue, VeOmni, SGLang LoRA, MoE RouterReplay

★★★★★★★★★ RTX 4090 impact:
  → #6738 OOM fix → MoE weight sync on RTX 4090 → critical for MoE GRPO
  → #6790 async trainer → enables separate rollout+training → memory optimization
  → #6572 progressing → BUT reviewer concerns → SM89 gap still needs P9
```

---

## 5. PyTorch — #187275 CHANGES_REQUESTED + #187435 OPEN

```
★★★★★★★★★ #187275 OPEN with CHANGES_REQUESTED (June 16):
  → Fix combo kernel crash with dynamic persistent reduction dimensions
  → Claude code review (requested by jansel): "clean, well-targeted fix"
  → karthickai (Collaborator) requested changes → guard gates both shared-block AND per-subkernel paths
  → CI: "You can merge normally" (4 unrelated failures)
  → ★★★★★★★★ Relevant to SM89: combo kernel affects persistent reduction codegen → RMSNorm-like fused reductions

★★★★★★★★★ #187435 OPEN — no_fuse_region per-op fusion barrier:
  → 804 additions, 7 files → CI currently failing (18 test failures)
  → ★★★★★★★★ Complementary to P9 → P9 global (5 LOC) + #187435 per-op (804 LOC)

★★★★★★★★★ CRITICAL NEW: PyTorch v2.12 max_autotune layout deferral NOW OPT-IN:
  → In v2.11: deferred layout freezing was default → caused SM89 batch-dependent results
  → In v2.12: REVERTED to immediate freezing → ★★★★★★★★ REDUCES SM89 risk by default!
  → Users who opt into deferred mode still need Fusion Guard
  → ★★★★★★★★ Makes P9 less urgent for v2.12+ but still needed for opt-in users

★★★★★★★★★ #181248 NEW: B200 batch invariance nondeterminism (NOT SM89):
  → Triton missing async fence between cp.async and tcgen05.mma on Blackwell
  → Triton commit #9610 fixes it → BUT MISSING from Triton 3.7.x release → cherry-pick needed
  → ★★★★★★★★ Blackwell-only issue → H100 passes → NOT RTX 4090

★★★★★★★★★ Triton 3.7 SM89 ptxas workaround reverted:
  → Older workaround for SM89 ptxas bug removed → now unnecessary with CUDA 13.0+
  → Non-TMA persistent templates for mm/addmm → enables persistent kernels on SM89!

★★★★★★★★★ P9 Fusion Guard issue draft — READY to file on GitHub

★★★★★★★★★ RTX 4090 impact:
  → #187275 progressing → combo kernel fix → affects RMSNorm persistent reduction paths
  → #187435 complementary → per-op fusion barrier → useful on SM90+ where some fusions safe
  → P9 issue draft updated → ready to file → pre-step before PR
```

---

## 6. SGLang — v0.5.13 Released June 13 + #27097 Multi-LoRA Bug

```
★★★★★★★★★ v0.5.13 (June 13) — Spec V2 default, Qwen3.5 Blackwell GDN:
  → Speculative decoding V2 now default → V1 deprecated
  → Qwen 3.5 Blackwell: FlashInfer GDN + CuTeDSL GDN prefill
  → experimental_sgl_trtllm MoE backend for FP8/NVFP4 LoRA (#27329 MERGED)
  → DeepSeek V4: Context Parallel + MTP, sparse FlashMLA, FP4 indexer, SM120
  → Dependencies: transformers 5.8.1, flashinfer 0.6.12, sgl-kernel 0.4.3

★★★★★★★★★ #24459 MERGED May 6 — aten::rms_norm + mm.dtype:
  → KERNEL-level strengthened → 9+ overrides now (was 7)
  → PROVES aten overrides work → P9 guard makes them MORE effective

★★★★★★★★★ #27097 NEW (June 3) — concurrent multi-LoRA diverge under deterministic:
  → Same (adapter, prompt) returns different text when concurrent
  → Sequential is stable → concurrency is the problem
  → Needs watching → may affect GRPO with LoRA

★★★★★★★★★ #27869 MERGED June 14 — Qwen3.5 deterministic batch-invariant logprobs:
  → FLA gated layernorm Triton → MAX_ROWS_PER_BLOCK in batch-invariant
  → MoE top-k combine → disables small-token torch.compile fast path
  → Overhead: 33% at bs=1, 16% at bs=64

★★★★★★★★★ RTX 4090 impact:
  → SGLang remains gold standard for deterministic inference
  → MoE LoRA + deterministic = UNIQUE capability → no other framework has this
  → #27097 multi-LoRA bug → needs watching → may need comment
```

---

## 7. MindIE/vLLM-Ascend — v0.21.0rc1 DeepSeek V4 + #10034 batch_invariant_ops MERGED

```
★★★★★★★★★ vLLM-Ascend v0.21.0rc1 (June 16):
  → DeepSeek-V4 on Ascend 950 → FULL_AND_PIECEWISE graph mode
  → W4A8 MXFP4 quantization on Ascend 950
  → MXFP8 FlashCommV3 on Ascend 950
  → CANN 9.0.0 for A2/A3/950
  → Python 3.12 support
  → batch_invariant_ops RL (#10034) → MERGED June 11 → VLLM_BATCH_INVARIANT=1 on Ascend

★★★★★★★★★ #10034 MERGED — batch_invariant_ops RL:
  → Adds setup and installation of batch_invariant_ops for Ascend A2/A3 NPUs
  → VLLM_BATCH_INVARIANT=1 required → SAME env var as CUDA → consistent!
  → AscendC custom operators for mm, matmul, sum, fused_infer_attention_score, add_rms_norm
  → npu_add_rms_norm = SPLIT add+rms_norm → "AclnnAddRmsNorm can't ensure batch invariant"
  → ★★★★★★★★ SAME root cause reasoning as vLLM on CUDA → Inductor fusion problem → SPLIT approach!

★★★★★★★★★ FULL_AND_PIECEWISE graph mode:
  → Requires HDK 25.5.1+ / CANN 8.5.0+ → removes stream-budget limitation
  → ~32K graphs on A3, ~64K on Ascend 950 → massive scale
  → Hybrid compilation → full-graph + piecewise → flexible scheduling

★★★★★★★★★ RTX 4090 impact:
  → Lessons portable: npu_add_rms_norm SPLIT → Triton add+rms_norm SPLIT → P9 enables this
  → VLLM_BATCH_INVARIANT consistent across CUDA and Ascend → cross-platform mechanism
  → FULL_AND_PIECEWISE → concept portable → BudgetRefiner + piecewise scheduling
```

---

## 8. rLLM — #605 CRITICAL GRPO Grouping Bug

```
★★★★★★★★★ #605 (OPEN, June 2) — CRITICAL GRPO grouping bug:
  → trajectory.uid vs task_ids → group size 1 → advantage = raw reward → GRPO BROKEN!
  → 3 stepwise_advantage configs ALL affected
  → Fix: 1 line for enable=False → few lines for per_step
  → ★★★★★★★★ MUST fix before any serious GRPO training on rLLM
  → ★★★★★★★★ rLLM GRPO ranking DOWNGRADED from #1 to #3 (until bug fixed)

★★★★★★★★★ #658 MERGED June 16 — cumulative token Tinker → drift-free multi-turn
★★★★★★★★★ #630 MERGED June 9 — deterministic optimizer steps → decoupled batch sizes
★★★★★★★★★ #576 OPEN → MergedSegment stalled 6+ weeks

★★★★★★★★★ RTX 4090 impact:
  → #605 CRITICAL → MUST fix → 1 line fix for enable=False → easy fix
  → Until fixed → rLLM Tinker GRPO = BROKEN → use verl HYBRID instead
  → Fixed → rLLM Tinker back to #1 ranking → simplest GRPO story
```

---

## Key Findings Summary

★★★★★★★★★ vLLM v0.23.0: Helion rms_norm kernel (6750 LOC), W4A16 Triton fallback for SM89, Triton MoE default
★★★★★★★★★ DeepSpeed v0.19.2: 27 merged PRs → #7992 coalesce_grad_reduction GRPO benefit, #8066 dtype preservation
★★★★★★★★★ Megatron #5391: compact LayerWise DDP removes dp padding → RTX 4090 memory efficiency
★★★★★★★★★ PyTorch #187275: CHANGES_REQUESTED but progressing → #187435 complementary to P9
★★★★★★★★★ SGLang v0.5.13: #27097 multi-LoRA determinism bug → needs watching
★★★★★★★★★ vLLM-Ascend #10034: batch_invariant_ops MERGED → npu_add_rms_norm SPLIT → same reasoning as Inductor
★★★★★★★★★ rLLM #605: CRITICAL GRPO grouping bug → ranking downgraded from #1 to #3 until fixed
★★★★★★★★★ verl #6699: detach model_output → 4x memory reduction → CRITICAL RTX 4090 LoRA GRPO!
★★★★★★★★★ verl #6717: Tinker training worker primitives → split training IN verl → no framework switch needed
★★★★★★★★★ verl #6688: LoRA IPC buffer crash fix → cudaErrorIllegalAddress resolved
★★★★★★★★★ Megatron #5394: ChainedOptimizer Muon clipping stalls → same pattern as DeepSpeed #8068/#7776
★★★★★★★★★ Megatron #5386/#5384: DSA Indexer Replay → extends RouterReplay to sparse attention (3404 LOC!)
★★★★★★★★★ DeepSpeed #8072/#8073: ZeRO-3+LoRA regression in v0.19.2 → ZeRO-2 unaffected → 2-line fix pending
★★★★★★★★★ DeepSpeed #8058: ZenFlow native CPU optimizer (954 additions) → GPU spike 2944→256 MiB → chunked copyback → fused multi-tensor → POSIX semaphore → NUMA-local
★★★★★★★★★ PyTorch v2.12: max_autotune layout deferral NOW OPT-IN → reduces SM89 fusion risk by default
★★★★★★★★★ PyTorch #181248: B200 Triton async fence → NOT SM89 (Blackwell only) → Triton cherry-pick needed

---

## References

- vLLM v0.23.0: https://github.com/vllm-project/vllm/releases/tag/v0.23.0
- DeepSpeed v0.19.2: https://github.com/microsoft/DeepSpeed/releases
- Megatron #5391: https://github.com/NVIDIA/Megatron-LM/pull/5391
- PyTorch #187275: https://github.com/pytorch/pytorch/pull/187275
- SGLang v0.5.13: https://github.com/sgl-project/sglang/releases/tag/v0.5.13
- rLLM #605: https://github.com/rllm-org/rllm/issues/605
- verl #6572: https://github.com/verl-project/verl/pull/6572
- vLLM-Ascend v0.21.0rc1: https://github.com/vllm-project/vllm-ascend/releases
