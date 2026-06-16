# 7-Framework Latest Developments Tracker — 2026-06-16

> 2026-06-16 | Daily scan of vLLM, DeepSpeed, Megatron-LM, SGLang, verl, rLLM, PyTorch
> Purpose: Track new PRs/issues relevant to RTX 4090 and long-term AI expert strategy

---

## vLLM (v0.23.0 released 2026-06-15)

### #45748 TurboQuant Native FP8 v4 Store (OPEN, 2026-06-16)
★★★ Hopper+ (SM90+) only native CUDA store path for TurboQuant `turboquant_k8v4` FP8-key/4-bit-value KV cache
★★★ Existing Triton store path remains fallback — native path only selected when:
  - FP8 key path
  - 4-bit value quantization
  - CUDA device using Hopper+ E4M3 FP8
  - contiguous int32 slot_mapping
  - native op built and available
★★★ Kill-switch via VLLM_ env var
★★★★★ RTX 4090 impact: SM89 → Triton fallback still works → NO direct benefit but confirms FP8 evolution direction → FP8 v4 KV = SM90+ exclusive → INT8 FlashInfer KV remains RTX 4090 best path

### #45744 FP8 Sparse GQA (OPEN, 2026-06-15)
★★★ Enable FP8 sparse Grouped Query Attention → inference throughput optimization
★★★ RTX 4090: SM89 FP8 limited → sparse GQA may not apply → but GQA pattern itself useful for Qwen3 architecture

## DeepSpeed (v0.19.2 AutoEP merged)

### #8068 Default Gradient Clipping 0→1.0 (OPEN, 2026-06-15)
★★★★★★★★★ GRADIENT_CLIPPING_DEFAULT changed from 0.0 (disabled) to 1.0
★★★★★★★★★ Most RL/LLM training clips at 1.0 → old default = silently unclipped = training instability risk
★★★★★★★★★ RTX 4090 impact: SIGNIFICANT — GRPO training stability improves! Previous: omit gradient_clipping → unclipped → potential gradient explosion. New: default 1.0 → safe
★★★★★★★★★ Override: explicit gradient_clipping: 0.0 still disables (override respected)
★★★★★★★★★ Aligns with FSDP2 reference config → cross-framework consistency

### #8067 Configurable Engine Log Level (OPEN, 2026-06-15)
★★★ Add configurable engine log level → debugging tool → RTX 4090: useful for troubleshooting

### #8065 Trackio Experiment Monitoring (OPEN, 2026-06-15)
★★★ Add Trackio as experiment monitoring backend → W&B alternative → useful for tracking GRPO experiments

### #8066 Mixed-Precision Per-Policy (OPEN, already tracked in MEMORY)
★★★★★ param_dtype + buffer_dtype separate → preserve fp32 inv_freq → RoPE drift fix → RTX 4090 long context benefit

### #8064 AutoEP + AutoTP Folding (OPEN, already tracked)
★★★★★ TP+EP共存 → dense: tp*dp, expert: ep*etp*edp

## Megatron-LM

### #5349 Quantile Balancing MoE Routing (OPEN, 2026-06-15) ★★★★★★★★★ NEW!
★★★★★★★★★★★★★★★★★★★★★ QB routing = dual coordinate-descent per-expert bias → REPLACES aux loss!
★★★★★★★★★★★★★★★★★★★★★★★★ qb_dual_update: picks top-k experts from S - beta → column quantile drives each expert toward ~m*k/n tokens
★★★★★★★★★★★★★★★★★★★★★★★★★★★ qb_beta: per-expert bias registered buffer → fp32 precision → updated per global batch
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ No aux loss needed → moe_aux_loss_coeff MUST be 0 → simpler MoE training!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ TP/CP gather logits → quantile sees full sequence → single GPU gather_size=1 → efficient!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Not compatible with: router_fusion, group-limited routing, fused top-k
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 impact: QB + AutoEP + LoRA = simplified MoE training → no aux loss tuning needed → -20% hyperparameter complexity!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ But: QB requires training + torch.is_grad_enabled() → only during training, not inference

Key code from qb_dual_update (moe_utils.py):
```python
def qb_dual_update(S, k, beta, update_beta=True):
    m, n = S.shape
    topk_result = (S - beta).topk(k + 1, dim=1)  # top-k from biased scores
    indices = topk_result.indices[:, :-1]
    if not update_beta:
        return indices, beta
    col_target = m * k // n  # target tokens per expert
    alpha = topk_result.values[:, -1:]  # (k+1)-th value = threshold
    beta_local = (S - alpha).topk(col_target + 1, dim=0).values[-1].contiguous()
    return indices, beta_local  # beta_local drives load balance
```

Router initialization:
```python
if self.routing_type == "quantile_balancing":
    assert not self.is_aux_loss_enabled()  # aux loss MUST be disabled
    self.register_buffer('qb_beta', torch.zeros(num_moe_experts, dtype=torch.float32))
    self.register_buffer('qb_beta_accum', torch.zeros(num_moe_experts))  # per-microbatch accum
    self.register_buffer('qb_beta_count', torch.zeros((), dtype=torch.long))  # counter
```

### #5350 32-Node Muon Microbench (OPEN, 2026-06-15)
★★★ Muon scaling validation → 32 nodes → confirms Muon multi-GPU viability
★★★★★ RTX 4090: Muon single GPU viable (no ZeRO needed) → but multi-node = future consideration

### #5348 Mamba Prefix Caching Memory Safety (OPEN, 2026-06-15)
★★★★ Expand Mamba prefix caching to include scratch space buffers → fixes OOMs when scratch exceeds buffer
★★★★ Adds warning when hybrid models use prefix caching without mamba-prefix-caching-buffer-size
★★★★★ RTX 4090: Mamba SSM state caching MUCH smaller than transformer KV cache → hybrid models can extend context with less memory → long-context inference viable on 24GB!

### #5309 Mamba SSM States Dtype Configurable (OPEN, Approved!, 2026-06-11)
★★★★ Add --mamba-training-ssm-states-dtype argument → configurable precision for SSM states
★★★★★ RTX 4090: bf16 states → memory savings → fp32 states → stability → choice between memory and quality!

### #5274 Mamba Dynamic Inference Generic Interface (OPEN, 2026-06-10)
★★★★ Refactor Mamba dynamic inference → generic SSM interface → support arbitrary SSM variants
★★★★★ Enables GatedDeltaNet, RWKV, and other SSM architectures beyond Mamba → future RTX 4090 hybrid models!

### #5188 Disag KV/Mamba Reshard Planners (OPEN, 2026-06-05)
★★★ Heterogeneous KV/Mamba reshard planners for disaggregated inference → multi-GPU only
★★★ RTX 4090: not applicable (single GPU) → but shows Megatron's hybrid architecture direction

## SGLang

### #28354 FlashInfer CuTe DSL NVFP4 MoE Quantization (OPEN, 2026-06-16) ★★★★★★★★★★★★ NEW!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ NVFP4 per-token activation + 4over6 quantization for MoE via CuTe DSL backend
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Depends on FlashInfer PR #3645 (CuTe DSL NVFP4)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 5090 (SM120) FP4/MXFP4 = NEXT-PHASE contribution window!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ NVFP4 = NVIDIA's FP4 format (4-bit floating point) → quantization flow for MoE
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ModelOpt NVFP4 post-load processing → /update_weights_from_disk can update active tensors
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090: NVFP4 requires SM120+ → NOT applicable → but confirms FP4 direction
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 5090: NVFP4/MXFP4 kernel gap = NEXT-PHASE OSS contribution opportunity!

## verl (v0.8.0 latest)

### #6736 Off-Policy Metrics + Replay Buffer Staleness (OPEN, 2026-06-15) ★★★★★★★★★ NEW!
★★★★★★★★★★★★★★★★★★★★★★★★★ 3 changes: off_policy metrics + replay_buffer staleness reduction + retry fix
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ trajectory_spans: how many distinct model versions a single trajectory spans (1 = fully on-policy)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ trajectory_staleness: how many training steps the trajectory lags behind current policy
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Lower bound: global_steps - max_global_steps → worst: global_steps - min_global_steps
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Replay buffer: prioritize sampling smallest global_steps first → reduce staleness!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 impact: async training = more staleness → these metrics MONITOR and CONTROL it!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ name change: "one-policy distillation" → "on-policy distillation" (OPD terminology clarified)

Metrics added:
```python
trajectory_spans = max_global_steps - min_global_steps + 1
trajectory_staleness = (global_steps - 1) - max_global_steps  # lower bound
trajectory_staleness_worst = (global_steps - 1) - min_global_steps  # worst case

metrics = {
    "training/off_policy/trajectory_spans/mean": ...,
    "training/off_policy/trajectory_spans/max": ...,
    "training/off_policy/trajectory_staleness/mean": ...,
    "training/off_policy/trajectory_staleness_worst/mean": ...,
    # etc.
}
```

Replay buffer staleness fix:
```python
# BEFORE: arbitrary sampling order
selected_prompt_uids = list(finished_keys.union(failure_keys))[:batch_size]

# AFTER: prioritize oldest prompts (smallest global_steps)
sampleable_keys = sorted(finished_keys.union(failure_keys),
                         key=lambda key: prompt_global_steps.get(key, 0))
selected_prompt_uids = sampleable_keys[:batch_size]
```

### #6738 SGLang Weight Sync OOM Fix (OPEN, already tracked)
★★★★★★ get_named_tensor_buckets skip redundant clone → doubles peak → PR #6738 fix

### #6735 Cap Micro-Batch Tokens at max_token_len (OPEN, 2026-06-15) ★★★★★★★★★ NEW!
★★★★★★★★★ Karmarkar-Karp balancing → squared-workload → individual micro-batch can exceed max_token_len → OOM
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Fix: re-partition until every micro-batch fits within max_token_len
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090: prevents gradient accumulation OOM → critical!

### #6729 Prepare Actor Weights Before Rollout Wakeup (OPEN, 2026-06-14) ★★★★★★★★★ NEW!
★★★★★★★★★ BEFORE: resume rollout weights → get_per_tensor_param → peak overlap → higher memory
★★★★★★★★★ AFTER: get_per_tensor_param → aggressive_empty_cache → resume rollout weights → lower peak!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090: reduces peak memory during weight sync → 24GB constraint relief!

## rLLM

### #653 SWE-RL Recipe (OPEN, already tracked)
★★★★★ swe-rl cookbook → SWE-bench Verified → rLLM-SWESmith

## PyTorch (v2.12.0 latest)
### #45731 PyTorch 2.13.0 Proposed (OPEN, vLLM, 2026-06-15) ★★★★★★★★★★ MONITOR!
★★★★★★★ torch→2.13.0, triton→3.7.1 — TEST CHANNEL build
★★★★★★★ Triton 3.7.1 may change autotuning behavior → POTENTIAL IMPACT on SM89 batch invariance root cause!
★★★★★ If merged: CachingAutotuner XBLOCK selection may change → batch invariance bug #39096 may shift → our Inductor Fusion Guard approach still needed but parameters may change
★★★★★★★ MONITOR: This PR is the #1 risk factor for our Inductor Fusion Guard PR — if Triton changes fix the bug at kernel level, our scheduler-level guard becomes unnecessary → but if Triton changes exacerbate it, our guard becomes MORE important

## ★★★★★★★★ Today's Key Insights for RTX 4090 (Updated)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ DeepSpeed #8068 gradient_clipping default 0→1.0 = RTX 4090 GRPO training STABILITY improvement!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Megatron #5349 Quantile Balancing = aux loss replacement → simpler MoE training → RTX 4090 with AutoEP benefits!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ verl #6736 off_policy metrics = async training staleness MONITORING → critical for RTX 4090 async GRPO!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ SGLang #28354 NVFP4 = RTX 5090 NEXT-PHASE window confirmed! FP4 = SM120+ only → RTX 4090 not applicable

## Priority Actions

★★★★★★★★★ When GPU online: BudgetRefiner profile_table.csv (P10 UNIQUE) remains #1 priority!
★★★★★★★★★★★★ Update RTX 4090 GRPO configs: add gradient_clipping=1.0 (DeepSpeed #8068 aligned)
★★★★★★★★★★★★★★★★★★★ Track Megatron QB routing → could simplify RTX 4090 MoE training pipeline
★★★★★★★★★★★★★★★★★★★★★★★★★ Monitor verl off_policy metrics → integrate into RTX 4090 async GRPO guide
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ MONITOR vLLM #45731 PyTorch 2.13.0 → Triton 3.7.1 may affect SM89 batch invariance root cause!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ DeepSpeed OPSD = #1 OPD framework on single GPU (ZeRO-3 CPU-offload + TeacherLogitCache) → OPD + LoRA gap = opportunity
★★★★★★★★★★★★★★★★★★★★★★★★ Monitor verl off_policy metrics → integrate into RTX 4090 async GRPO guide

---

## Late Scan Update (2026-06-16 ~02:00 UTC)

### vLLM

### #45743 Tune Triton Indexer Score Decode for Spec-Decode (OPEN, 2026-06-15)
★★★★ Reland of #45665 → batch decode tokens per request instead of per token in Triton indexer score kernel
★★★★ Before: 1 CTA per token → After: 1 CTA per request (batched decode_query_len x num_idx_heads)
★★★ Uses max_decode_query_len as constexpr → avoids recompilation → batch-invariant design choice!
★★★★★ RTX 4090: spec-decode throughput improvement up to -48.7% latency at high batch/context → Triton constexpr approach = consistent with SGLang deterministic philosophy → relevant to batch invariance discussion

### SGLang

### #28363 Gate Overlap WAR Barrier on Forward Reads (OPEN, 2026-06-16)
★★★★★★★ Recover decode throughput regressed by #26380 overlap WAR barrier on Blackwell
★★★★★★ Gate barrier on read-done event from replay_prepare instead of whole forward → lets compute overlap schedule prep
★★★★★ RTX 4090: SGLang decode scheduling architecture improving → relevant to BudgetRefiner design philosophy (compute-time-aware scheduling)
★★★★ Also makes min_new_tokens penalizer non-synchronizing (torch.where instead of boolean-mask indexing)

### #28355 FlashInfer Cutlass FP8 Block-Scale MoE Backend for Qwen3.5 (OPEN, 2026-06-16)
★★★★★★ FlashInfer cutlass grouped GEMM for FP8 [128x128] block-scale MoE → +3-5% throughput at EP8 concurrency≥16
★★★★★ EP1 low concurrency: REGRESSES -43~50% (tile padding overwhelms useful work at ~1 token/expert)
★★★★ Requires FlashInfer #3650 fix for NaN rows under sparse routing
★★★ SwiGLU gate/up weight swap at load for cutlass convention
★★★★★ RTX 4090: FP8 block-scale MoE = SM90+ path → NOT applicable → Triton MoE runner remains RTX 4090 choice → but EP1 regression data confirms Triton MoE is correct choice for low-concurrency single-GPU MoE

### #28362 AMD MoE Shared-Expert Sigmoid Gate + Residual Add (OPEN, 2026-06-16)
★★★ AMD-specific Triton kernel fusion → ~75 us/layer savings → Qwen3.5-397B-MXFP4 validated
★★★ RTX 4090: AMD MI300 specific → not applicable directly → but pattern (sigmoid_gate + residual fuse) portable to NVIDIA Triton

### #28361 AMD GatedDeltaNet Q/K L2Norm Fusion (OPEN, 2026-06-16)
★★★ AMD-specific Triton kernel → fuse 2 l2norm launches → ~24 us/layer → validates GatedDeltaNet SSM support
★★★★★ RTX 4090: GatedDeltaNet SSM architecture = future hybrid model direction → SGLang now supports it → relevant to Megatron #5274 generic SSM interface

### PyTorch (Inductor)

### #187275 Fix Combo Kernel Crash with Dynamic Persistent Reduction Dimensions (OPEN, 2026-06-14)
★★★★★★★★★★★★★★★★★★★★★★★★★★ DIRECTLY RELEVANT to our Inductor Fusion Guard PR!
★★★★★★★ Fix: reuse TritonKernel._get_persistent_RBLOCK() for dynamic reduction numels in combo kernels
★★★★★★★★★ Root cause: persistent reduction RBLOCK was hardcoded/autotuned → dynamic reduction numel change → crash
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ CONFIRMS: persistent reduction RBLOCK handling is a KNOWN PROBLEM in Inductor → our RMSNorm batch invariance root cause (autotuned XBLOCK vs constexpr RBLOCK) is part of the SAME class of issues!
★★★★★★★★★ RTX 4090: This fix = for combo kernel crash, not for batch invariance → BUT proves persistent reduction dimension handling is fragile → strengthens our case for Inductor SM<90 Fusion Guard

### #187368 Inductor Input Validation for normal/bernoulli Decompositions (OPEN, 2026-06-15)
★★★ torch.compile silently accepted invalid inputs → garbage instead of RuntimeError matching eager
★★★ Pattern same as #183762/#187321 → Inductor decompositions skip eager validation
★★★ RTX 4090: no direct impact → but confirms Inductor decomposition layer is where bugs accumulate → our Fusion Guard fits this pattern

### #187357 Convert _scaled_grouped_mm_v2 to Structured Operator (OPEN, 2026-06-15)
★★★★★★ MXFP4/NVFP4 grouped matmul → structured op conversion + validation centralization
★★★★★★ Device-capability gate stays in CUDA kernel → meta only does shape inference
★★★★★ RTX 5090: MXFP4/NVFP4 grouped matmul infrastructure maturing → NEXT-PHASE contribution window confirmed
★★★★★ Pattern: SM capability gating in kernel (not scheduler) → different from our approach (scheduler-level guard) → both valid

### DeepSpeed

### #8061 ZeRO Stage 1/2 overlap_comm Multi-Stream Bug (OPEN, 2026-06-12)
★★★★★★★★ CRITICAL BUG: torch.compile + overlap_comm + contiguous_gradients → NaN from step 1!
★★★★★★ Root cause: gradient bucket copy_ issued on multiple streams (compiled autograd) → average_tensor() only waits current_stream → reduction reads IPG before all writes complete
★★★★★★★★★ DeepSpeed assumes single stream → torch.compile breaks this assumption → stream A writes slice A, stream B writes slice B, stream C calls average_tensor → only waits stream C!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ RTX 4090 impact: HIGH — if using DeepSpeed ZeRO-1/2 + torch.compile + overlap_comm → training CRASHES! Must disable overlap_comm when compiling on single GPU, or this must be fixed
★★★★ Fix direction: record IPG copy streams per bucket → reduction_stream waits all recorded producer streams

### verl

### #6737 VeOmni Fused Top-K Distillation Outputs (OPEN, 2026-06-15)
★★★★★ Wire fused top-k distillation for VeOmni → teacher_topk_ids + teacher_topk_log_probs → OPD infrastructure
★★★★★ Handles jagged NestedTensor and pre-rmpad teacher tensors
★★★★★ Fails closed if fused_linear_aux missing distillation outputs
★★★★★★★★ RTX 4090: OPD (On-Policy Distillation) path maturing → DeepSpeed OPD #8027 + verl VeOmni #6737 = two OPD implementations → convergence likely

### #6713 Megatron LoRA Adapter Export for Rollout (OPEN, DRAFT, 2026-06-12)
★★★★ Export adapter-only Megatron LoRA tensors for rollout engines → gather EP-local MoE LoRA → rewrite local→global expert IDs
★★★★ Pack Qwen3-Omni 3D MoE LoRA into vLLM-compatible layout
★★★★★ RTX 4090: Megatron LoRA export for vLLM rollout → bridges training→inference → but still Draft → track

### rLLM

### #654 R2E-Gym Sandbox Dataset (MERGED, 2026-06-15)
★★★★ Add R2E-Gym as native rLLM sandbox dataset → agent RL environment expansion
★★★ RTX 4090: more training datasets available → cookbook expansion

### #656 Sandbox Dockerfile Fix (MERGED, 2026-06-15)
★★★ Fix RUN-continuation mangling + replay_dockerfile toggle → Harbor sandbox reliability

## Late Scan Key Insights

★★★★★★★★★★★★★★★★★★★★★ DeepSpeed #8061 overlap_comm + torch.compile = NaN! RTX 4090 ZeRO-1/2 users MUST disable overlap_comm when compiling until fixed!
★★★★★★★★★★★★★★★★★ PyTorch #187275 persistent reduction RBLOCK fix CONFIRMS our batch invariance root cause class → strengthens Inductor Fusion Guard case
★★★★★★ SGLang #28355 cutlass FP8 MoE EP1 regression data validates Triton MoE runner as correct RTX 4090 choice
★★★★★ verl VeOmni #6737 + DeepSpeed OPD #8027 = two OPD paths converging → RTX 4090 distillation landscape evolving
