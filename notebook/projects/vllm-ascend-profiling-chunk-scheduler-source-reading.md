# vLLM-Ascend ProfilingChunkScheduler Source Reading — Runtime-Adaptive Scheduling

> 2026-06-18 | Source: vllm-ascend/core/ (cloned June 18)
> ★★★★★ NEW DISCOVERY: vLLM-Ascend has TWO mutually exclusive scheduling approaches!
> BudgetRefiner (lookup table) vs ProfilingChunkScheduler (runtime profiling + quadratic model)
> ProfilingChunkScheduler is MORE sophisticated but requires pp > 1

---

## 1. ★★★★★ Two Scheduling Approaches — Mutually Exclusive!

```
Platform selection logic (platform.py lines 726-738):

# Approach 1: BudgetRefiner (lookup table)
if ascend_config.SLO_limits_for_dynamic_batch != -1:
    scheduler_cls = SchedulerDynamicBatch  # BudgetRefiner-based
    enable_chunked_prefill = True

# Approach 2: ProfilingChunkScheduler (runtime profiling)
if ascend_config.profiling_chunk_config.enabled:
    scheduler_cls = ProfilingChunkScheduler  # Runtime-adaptive
    # OVERRIDES BudgetRefiner if both configured!

★★★★★★★ Key: ProfilingChunk OVERRIDES BudgetRefiner when both configured
  → The if-check for profiling_chunk comes AFTER SLO_limits check
  → Last writer wins → ProfilingChunk takes priority
  → AscendConfig validates they cannot both be active (mutual exclusion)
```

### Comparison Table

| Aspect | BudgetRefiner | ProfilingChunkScheduler |
|--------|--------------|------------------------|
| **Trigger** | `SLO_limits_for_dynamic_batch != -1` | `profiling_chunk_config.enabled` |
| **Approach** | Offline lookup table (profile_table.csv) | Runtime profiling + quadratic model |
| **Data source** | Pre-built CSV from Huawei OBS bucket | 64 forward pass samples at startup |
| **Model** | Lookup: (ctx_len, d_num) → chunk_size | Quadratic: f(l) = al² + bl + c |
| **Budget control** | token_budget only | token_budget + time_budget (dual!) |
| **Adaptation** | Static table, no runtime update | Online refinement via execution timing |
| **Hardware req** | Currently only 910B3 NPU | Requires pp > 1 (pipeline parallelism) |
| **External deps** | pandas + CSV file download | numpy + model forward at startup |
| **Startup overhead** | Zero (just read CSV) | ~64 forward passes for profiling |
| **Complexity** | 84 lines BudgetRefiner class | 350+ lines ChunkSizePredictor + Manager |
| **Decode-first** | d_lst + p_lst rearrangement | Follows vLLM standard RUNNING/WAITING |

---

## 2. ★★★★★ ChunkSizePredictor — Quadratic Latency Model

```
★★★★★★★ Core model: f(l) = a*l² + b*l + c

Where:
  l = sequence length (total tokens processed)
  a = quadratic coefficient (accounts for attention O(n²) cost)
  b = linear coefficient (accounts for linear compute)
  c = constant offset (kernel launch overhead etc.)

★★★★★★★ Prediction equation:
  Given target latency T and current history length L, solve for chunk x:
  f(L+x) - f(L) = T
  → a*x² + (2aL + b)*x - T = 0
  → Standard quadratic: A*x² + B*x + C = 0

★★★★★★★ Solution:
  A = a
  B = 2a*L + b
  C = -T (negative of target latency)
  discriminant = B² - 4AC
  x = (-B + sqrt(discriminant)) / (2A)  (positive root only)

★★★★★★★ Smoothing: smoothed = base_chunk + smooth_factor * (x - base_chunk)
  → smooth_factor ∈ (0, 1] — prevents wild oscillations
  → Default 0.8 → 80% of prediction + 20% of base
  → Clamped to min_chunk (default 4096)
  → Aligned to page_size or 64 (max(page_size, 64))
```

### History-Aware Model (Online Calibration)

```
★★★★★★★ History-aware: f(C, H) = a*C(C+H) + b*C + c*H

Where:
  C = chunk size (new tokens being prefilled)
  H = history length (tokens already computed)
  More accurate because it accounts for KV cache growth

★★★★★★★ Online calibration:
  1. Startup profiling: 64 forward passes → fit f(l) = al² + bl + c
  2. Runtime: record_batch_execution_time() → accumulate (x1, x2, x3, time_ms)
  3. x1 = Σ(C+H)*C (quadratic feature)
  4. x2 = Σ(C+H) (linear feature)
  5. x3 = batch_size (constant feature)
  6. Fit f(C,H) when 5-30 data points collected → with_history_ready = True
  7. Once history_fitted → disable timing (need_timing = False) → zero overhead

★★★★★★★ Prediction with history:
  A = a (quadratic_chunk_a)
  B = a*H + b
  C = b*H + c - T
  → Solves same quadratic form but accounts for KV cache state
```

---

## 3. ★★★★★ ProfilingChunkScheduler — schedule() Override

```
★★★★★★★ Key modifications from base Scheduler.schedule():

1. Dual budget (lines 248-250):
   target_latency = predictor.target_latency
   time_budget = target_latency if target_latency else float("inf")
   → TWO constraints: token_budget AND time_budget
   → Schedule stops when EITHER budget exhausted

2. RUNNING loop (line 269):
   while req_index < len(self.running) and token_budget > 0 and time_budget > 0:
   → Added "time_budget > 0" to loop condition

3. Dynamic chunk sizing for RUNNING (lines 312-327):
   if profiling_chunk_manager.is_ready and request.num_computed_tokens < num_prompt_tokens:
     predicted_chunk = predict_chunk_size(num_computed_tokens, time_budget)
     num_new_tokens = min(num_new_tokens, predicted_chunk)

4. Time budget accounting (line 387):
   if request.num_computed_tokens < num_prompt_tokens:
     time_budget -= predict_time(num_new_tokens, num_computed_tokens)
   → Prefill requests consume time_budget
   → Decode requests (num_new_tokens=1) SKIPPED → negligible latency

5. WAITING loop (line 432):
   while (self.waiting or self.skipped_waiting) and token_budget > 0 and time_budget > 0:
   → Same dual budget condition

6. Dynamic chunk sizing for WAITING (lines 521-536):
   Same pattern as RUNNING but for new requests entering the scheduler

★★★★★★★ Key insight: decode requests DON'T consume time_budget!
  → Decode latency is negligible (single token)
  → Only prefill latency matters for SLO compliance
  → This prevents decode requests from starving other prefills
```

---

## 4. ★★★★★ Patch Architecture — Monkey-Patching EngineCore

```
★★★★★★★ vLLM-Ascend patches EngineCore via monkey-patching:

patch_profiling_chunk.py (230 lines):

1. Patch EngineCore.__init__ (line 197):
   → After original init, call scheduler.run_profiling_chunk_init(model_executor)
   → This triggers the 64-sample profiling at startup

2. Wrap scheduler.update_from_output (line 146):
   → Before calling original, call _record_execution_timing()
   → Extract execution_time_ms from model_output
   → Feed to profiling_chunk_manager.record_batch_execution_time()
   → Online model refinement!

3. Wrap scheduler.schedule (line 170):
   → After schedule(), propagate timing-done signal
   → scheduler._profiling_timing_done → output.disable_profiling_timing

4. Handle multiprocessing spawn (line 224):
   → Patch EngineCoreProc.run_engine_core
   → Re-apply patches in child process (pickle unpickling triggers module import)
   → Essential for distributed setup

★★★★★★★ Timing lifecycle:
  1. Startup: need_timing=True → collect execution_time_ms each step
  2. After 3 first-chunk samples → set_target_latency from real measurement
  3. After 5-30 history samples → history model fitted → with_history_ready=True
  4. history_fitted + _set_time_done → need_timing=False → zero sync overhead!
  → Total: ~3+30 = ~33 steps before calibration complete → negligible
```

---

## 5. ★★★★★★★ RTX 4090 Implications — BudgetRefiner FIRST, ProfilingChunk SECOND

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Contribution strategy:

Priority 1 (P10): BudgetRefiner → vLLM upstream
  → Simpler: 84 lines core + profile_table.csv
  → No startup overhead: just read CSV
  → RTX 4090 profile data = our UNIQUE contribution
  → Clear integration: 3 points in scheduler.py
  → Already production-tested on Ascend → proven design

Priority 2 (P9→P8): ProfilingChunkScheduler → vLLM upstream (after BudgetRefiner)
  → More sophisticated: runtime profiling + quadratic model
  → Dual budget: token + time → better SLO control
  → History-aware: online calibration → adapts to actual workload
  → BUT: requires startup profiling (64 forward passes) → adds latency
  → BUT: requires pp > 1 on Ascend → need to generalize for single GPU
  → BUT: monkey-patch architecture → need cleaner integration

★★★★★★★★★ Phased approach:
  Phase 1: BudgetRefiner (lookup table) → establishes SLO-aware scheduling in vLLM
  Phase 2: ProfilingChunk (runtime profiling) → adds adaptive capability on top
  Phase 3: Hybrid → BudgetRefiner fallback when ProfilingChunk not calibrated

★★★★★★★★★ RTX 4090-specific:
  → BudgetRefiner: collect profile_table.csv on RTX 4090 → our exclusive data
  → ProfilingChunk: 64 startup forwards on RTX 4090 → ~2-3 min warmup → acceptable
  → Single GPU: ProfilingChunk requires pp > 1 → need to remove this constraint
  → On RTX 4090: pp=1 always → ProfilingChunk can work with single GPU profiling
```

---

## 6. Key Source Files

| File | Lines | Role |
|------|-------|------|
| scheduler_profiling_chunk.py | 729 | ProfilingChunkScheduler override of schedule() |
| profiling_chunk_predictor.py | 428 | ChunkSizePredictor (quadratic model) + ProfilingChunkManager |
| scheduler_dynamic_batch.py | 575 | BudgetRefiner (84 lines) + SchedulerDynamicBatch |
| patch_profiling_chunk.py | 230 | EngineCore monkey-patch for startup profiling + timing |
| ascend_config.py | 824 | AscendConfig + ProfilingChunkConfig + AscendCompilationConfig |
| platform.py | ~752-738 | Scheduler selection (BudgetRefiner vs ProfilingChunk) |

---

## 7. Additional AscendConfig Features Found

```
★★★★★★★ Interesting new configs discovered:

1. AscendCompilationConfig (line 493):
   → enable_npugraph_ex: NPU Graph equivalent of CUDA Graph → True by default!
   → enable_static_kernel: compile op binaries for fixed shapes → False by default
   → fuse_norm_quant: norm + quant fusion → True
   → fuse_qknorm_rope: QK norm + RoPE fusion → True
   → fuse_allreduce_rms: allreduce + RMSNorm fusion → False (PP only)

2. AscendFusionConfig (line 545):
   → fusion_ops_gmmswigluquant: grouped matmul + SwiGLU + quant fusion → True by default!
   → = npu_dequant_swiglu_quant compose-level equivalent → MoE 6→1

3. XliteGraphConfig (line 563):
   → xlite graph mode = extreme NPU graph optimization → entire model as 1 graph
   → Not compatible with spec decode or PP

4. EplbConfig (line 713):
   → Expert Parallel Load Balancing → dynamic expert redistribution
   → eplb_policy_type ∈ {0,1,2,3} → 4 load balancing algorithms
   → num_redundant_experts → hot expert replication

5. RejectionSamplerConfig (line 645):
   → Block Verify: evaluate all draft tokens as cumulative probability block
   → Entropy Verify: adjust acceptance threshold by target distribution entropy
   → posterior_threshold = 0.95, posterior_alpha = 0.4 → tunable

6. WeightPrefetchConfig (line 589):
   → Weight prefetch during decode → reduce stall
   → attn qkv/o: 1.0 ratio, moe gate_up: 0.8, mlp gate_up/down: 1.0

7. enable_async_exponential (line 232):
   → Disabled when batch_invariant mode enabled!
   → SGLang deterministic inference compatibility confirmed

★★★★★★★★★ gmmswigluquant fusion = our Triton dequant_swiglu_quant port target!
  → Ascend has it: torch.ops.npu.npu_dequant_swiglu_quant
  → CUDA needs it: Triton tl.constexpr kernel → P6 contribution
```

---

## Key Findings Summary

★★★★★★★ vLLM-Ascend has TWO scheduling approaches: BudgetRefiner (lookup table) and ProfilingChunkScheduler (runtime profiling + quadratic model)
★★★★★★★ ProfilingChunkScheduler uses DUAL budget: token_budget + time_budget → more SLO control than BudgetRefiner's token-only
★★★★★★★ ProfilingChunkScheduler requires pp > 1 → needs generalization for RTX 4090 single GPU
★★★★★★★ BudgetRefiner FIRST for vLLM upstream (P10) → simpler, proven, RTX 4090 profile data UNIQUE
★★★★★★★ ProfilingChunk SECOND (after BudgetRefiner lands) → adds runtime-adaptive capability
★★★★★★★ gmmswigluquant = Ascend compose-level fusion → Triton port target (P6)
★★★★★★★ enable_async_exponential disabled when batch_invariant → SGLang deterministic compatibility confirmed
★★★★★★★ Ascend has EPLB (expert load balancing) → no equivalent in CUDA frameworks → potential contribution area
