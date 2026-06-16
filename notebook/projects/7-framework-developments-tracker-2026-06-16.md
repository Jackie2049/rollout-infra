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
- No new RTX 4090-relevant PRs detected today (Inductor Fusion Guard still our priority)

---

## ★★★★★★★★ Today's Key Insights for RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ DeepSpeed #8068 gradient_clipping default 0→1.0 = RTX 4090 GRPO training STABILITY improvement!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ Megatron #5349 Quantile Balancing = aux loss replacement → simpler MoE training → RTX 4090 with AutoEP benefits!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ verl #6736 off_policy metrics = async training staleness MONITORING → critical for RTX 4090 async GRPO!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ SGLang #28354 NVFP4 = RTX 5090 NEXT-PHASE window confirmed! FP4 = SM120+ only → RTX 4090 not applicable

## Priority Actions

★★★★★★★★★ When GPU online: BudgetRefiner profile_table.csv (P10 UNIQUE) remains #1 priority!
★★★★★★★★★★★★ Update RTX 4090 GRPO configs: add gradient_clipping=1.0 (DeepSpeed #8068 aligned)
★★★★★★★★★★★★★★★★★★★ Track Megatron QB routing → could simplify RTX 4090 MoE training pipeline
★★★★★★★★★★★★★★★★★★★★★★★★ Monitor verl off_policy metrics → integrate into RTX 4090 async GRPO guide
