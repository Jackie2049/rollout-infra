# vLLM Latest Developments — June 2026 Reading

> 2026-06-18 | Comprehensive scan of vLLM developments affecting RTX 4090 strategy
> Covers: #39096, #40628, #42120, #43373/#43461, #45731, BudgetRefiner
> Source: GitHub issues/PRs, v0.23.0 release, community activity, diary entries 232-237
> ★★★★★★★★ BudgetRefiner = P10 UNIQUE contribution → NO competing PR confirmed!

---

## 1. #39096: SM<90 Batch Invariance — STILL UNFIXED

### Status and Significance

★★★★★★★★★ **#39096 remains the most critical RTX 4090 correctness bug** — last updated April 17, zero upstream progress.

The Inductor root cause has been confirmed through source-level analysis:
- torch.compile alone breaks batch invariance on SM<90 — not just CUDA graphs
- enforce_eager=True disables BOTH compile and graphs — only known workaround
- Disabling only graphs while keeping compile STILL fails

### Root Cause: Inductor Full-Graph Fusion

```
★★★★★★★★★ Inductor root cause chain:

Layer 1: torch.compile(rms_norm_native) alone → batch invariant ✓
  → RMSNorm itself remains correct after compile
  → bitwise_equal: True, max_abs_diff: 0.0

Layer 2: torch.compile(full Llama forward) → batch NOT invariant ✗
  → divergence at token 80 (20400 != 4324)
  → Different batch sizes → different outputs

Layer 3: Root cause isolation:
  → Inductor fuses RMSNorm + residual add + next linear input prep into one persistent kernel
  → On SM<90 → this fused kernel's reduction operations → batch-dependent!
  → Three mechanisms:
    a) Inductor Triton reduction kernels with batch-dependent configs (different BLOCK_M/BLOCK_N per batch size)
    b) cuBLAS/cuBLASLt GEMM dispatch (partially fixed by #38938 for lm_head)
    c) CUDA graph replay with wrong kernel configuration
```

### SM89-Specific Architecture Factor

```
★★★★★★★ SM89 (Ada Lovelace) is architecturally vulnerable:

  SM89 shared memory: 100KB per SM (vs 164KB on SM80, 228KB on SM90)
  → Triton kernels that fit SM80/SM90 shared memory need different tiling on SM89
  → Different accumulation orders → different results

  SM89 lacks Hopper features:
  → No TMA (Tensor Memory Accelerator)
  → No WGMMA (Warpgroup Matrix Multiply Accumulate)
  → No Distributed Shared Memory (CTA Cluster)
  → Inductor falls back to SM80-style strategy → but SM89 ≠ SM80 → behavior diverges!
```

### Model-Specific Behavior

★★★★★★★ **Qwen3-1.7B passes on SM86 (RTX 3090) — Llama fails on SM89.** This is not a universal SM<90 problem:
- Different model structures → different fusion patterns → different batch invariance behavior
- Qwen3: Inductor fusion does NOT cover torch.mean → override remains effective
- Llama: Inductor fusion covers torch.mean → override bypassed → batch-dependent

### Dual Failure Path

★★★★★★★★★ Both torch.compile AND CUDA graphs independently break batch invariance. Disabling one is insufficient:
- compile=On + graphs=Off → FAILS at token 80
- compile=On + graphs=On → FAILS (original bug)
- compile=Off + graphs=Off → WORKS (enforce_eager=True)

### RTX 4090 Workaround and Impact

```
★★★★★★★★★ Current workaround: enforce_eager=True

  Impact:
  → Speculative decoding (EAGLE/MTP) → unreliable on SM89 → correctness bug
  → ~10-15% throughput loss (no compile acceleration, no graph optimization)
  → GRPO rollout: rLLM Tinker → in-process → NOT affected
  → GRPO rollout: verl HYBRID → vLLM → not using spec decode → NOT directly affected
  → BUT: if verl enables spec decode for rollout acceleration → enforce_eager needed

  Best RTX 4090 config:
  → Serving: INT4 + INT8 KV + prefix caching → normal throughput (no spec decode)
  → If spec decode needed: enforce_eager=True → accept ~10-15% throughput loss
```

### Fix Path: Inductor Fusion Guard (P9)

★★★★★★★★★ The best fix is our proposed PyTorch Inductor SM<90 Fusion Guard (P9):
- 5-line guard in `can_fuse_vertical` → `props.major < 9 → WhyNoFuse`
- Blocks problematic reduction fusions on SM<90 while preserving torch.mean override
- Issue draft ready (pre-step required before PR submission per PyTorch process)
- NO competing PR exists in PyTorch → strong contribution opportunity
- Complementary: Triton dequant_swiglu_quant (P6) provides GOOD fused path → constexpr → deterministic

### vLLM #40628 RFC: Complementary Architecture Fix

#40628 proposes batch-invariant dispatching in vLLM IR with 5 options — Option 2 recommended. This RFC addresses the architectural side (how vLLM dispatches), while our Fusion Guard addresses the compiler side (how Inductor fuses). Both are needed for a complete solution.

### References

- Issue: https://github.com/vllm-project/vllm/issues/39096
- Comment draft: notebook/projects/vllm-39096-batch-invariance-comment-draft.md
- Source reading: notebook/projects/vllm-sm89-batch-invariance-deep-reading.md
- Bug reading: notebook/projects/vllm-sm89-batch-invariance-bug-reading.md
- Diagnostic tool: tools/sm89_batch_invariance_diagnostic.py
- Repro script: tools/sm89_batch_invariance_repro.py

---

## 2. #40628 RFC: Batch-Invariant Dispatching in vLLM IR

### Overview

★★★★★ #40628 is an RFC proposing batch-invariant dispatching within vLLM's internal representation (IR). This addresses the architectural dimension of the batch invariance problem — how vLLM should structure its IR to guarantee batch-independent behavior regardless of backend (Inductor, CUDA graphs, FlashInfer, etc.).

### 5 Options Proposed

```
★★★★★ Five options for batch-invariant dispatching:

Option 1: Disable torch.compile on SM<90
  → Simplest but most aggressive → loses ALL compile benefits → ~10-15% throughput loss
  → Current enforce_eager approach → already in vLLM

Option 2: Batch-invariant IR annotations ★★★★★★★ RECOMMENDED
  → Mark operations in vLLM IR as batch-invariant-sensitive
  → Inductor respects annotations → skips batch-dependent fusion for marked ops
  → Preserves most compile benefits while protecting correctness
  → Requires vLLM + Inductor coordination → but principled and extensible

Option 3: Separate per-batch-size compiled graphs
  → Compile separate graph for each batch size → eliminates cross-batch fusion
  → Correct by construction → but memory overhead (multiple compiled graphs)
  → Startup time penalty → not practical for serving

Option 4: Custom Triton kernels for all sensitive ops
  → Replace Inductor-generated Triton with custom constexpr kernels
  → SGLang approach (tl.constexpr BLOCK sizes → KERNEL-level guarantee)
  → Most principled → but requires significant kernel development effort

Option 5: Hybrid: compile + dynamic fallback
  → Compile normally → but detect batch size changes → fall back to eager for sensitive ops
  → Partial benefit → but detection overhead and complexity
```

### Why Option 2 is Recommended

★★★★★★★★★ Option 2 is recommended because:
- It provides a principled architectural solution — annotations at the IR level
- It preserves most compile benefits — only skips batch-sensitive fusions
- It is extensible — new batch-sensitive ops can be annotated
- It aligns with our P9 Fusion Guard — both target the fusion layer
- It requires minimal vLLM changes — annotations are metadata, not code changes
- The Inductor side would need to respect annotations → this is exactly what our Fusion Guard does (skip fusions on SM<90)

### RTX 4090 Implications

★★★★★★★★★ #40628 Option 2 + our P9 Fusion Guard = complete solution:
- Option 2: vLLM side → annotate batch-sensitive ops in IR
- Fusion Guard: PyTorch side → block batch-dependent fusions on SM<90
- Together: vLLM marks what to protect → Inductor respects annotations → correctness guaranteed
- Without both: either side alone provides incomplete coverage

### References

- RFC: https://github.com/vllm-project/vllm/issues/40628
- Fusion Guard PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard issue draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md

---

## 3. #42120: FP8 MoE+LoRA — Close to Merge (Merge Conflicts)

### Overview

★★★★★★★ #42120 implements FP8 quantization for MoE models with LoRA serving. Currently OPEN with merge conflicts blocking final merge.

### Key Details

```
★★★★★★★ #42120 status:

  → FP8 MoE + LoRA serving → validated on SM 12.0 (Blackwell)
  → Close to merge → merge conflicts need resolution → likely to merge soon
  → MoE quant: FP8 expert weights → W8A8 activation quantization
  → LoRA: LoRA adapters on MoE expert MLPs → quantized LoRA path

★★★★★ RTX 4090 implications:
  → FP8 requires SM90+ → RTX 4090 (SM89) → NOT directly applicable
  → BUT: the LoRA + MoE architecture pattern IS applicable
  → → INT4 MoE + LoRA on SM89 → same pattern but different quantization
  → → Our Triton dequant_swiglu_quant (P6) provides the SM89 quantized MoE path

★★★★★★★ Complementarity with our contributions:
  → #42120 establishes the MoE+LoRA serving architecture → framework pattern
  → Our Triton swiglu kernel provides the SM89-specific MoE quantization path
  → Together: #42120 = SM90+ FP8 path, our P6 = SM89 INT8 path → complete coverage
```

### Significance for BudgetRefiner

★★★★★ #42120 demonstrates vLLM's MoE serving maturity → BudgetRefiner SLO scheduling is increasingly important for MoE models where decode-prefill interference is amplified by expert routing latency.

### References

- PR: https://github.com/vllm-project/vllm/pull/42120
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- MoE serving reading: notebook/projects/vllm-moe-serving-reading.md

---

## 4. #43373/#43461: QuantKey Pivot + MoEKernelOracle ABC Conflict

### Overview

★★★★★★★ The QuantKey refactor (#32268) has evolved into two competing PRs with conflicting approaches. This conflict affects our SM89 FP8 guard strategy.

### #43373: QuantKey Pivot (Oracle Removed)

```
★★★★★ #43373 approach:

  → Removed MoEKernelOracle (the oracle pattern that hardcoded MoE kernel selection)
  → Moved oracle functionality to humming_utils.py → simpler utility functions
  → QuantKey objects → structured keys for quantization method identification
  → → QuantKey(method="fp8_e4m3_kv") → can be extended with requires_sm

★★★★★ Significance:
  → Removing oracle → more modular → QuantKey can carry metadata
  → humming_utils.py → utility layer → not ABC → simpler → easier to extend
  → Aligns with our #32268 plan: Phase 1 pure refactor → Phase 2 add SM requirements
```

### #43461: MoEKernelOracle ABC (Still ABC-based)

```
★★★★★ #43461 approach:

  → Keeps MoEKernelOracle as ABC (Abstract Base Class)
  → Oracle pattern → abstract interface → MoE kernel selection through ABC methods
  → More structured → but more rigid → harder to extend without new ABC subclasses

★★★★★★★ Conflict:
  → #43373 removes oracle → humming_utils.py → simpler
  → #43461 keeps oracle ABC → more structured → but rigid
  → Both modify the same quantization registry → merge conflicts inevitable
  → → Only one can merge → the other must adapt or be abandoned
```

### RTX 4090 Implications

```
★★★★★★★ The outcome of this conflict determines our SM89 guard strategy:

  If #43373 merges (oracle removed → humming_utils.py):
    → QuantKey carries metadata → QuantKey(method="fp8_e4m3_kv", requires_sm=90)
    → Registry-level SM89 guard → systematic → extensible
    → → Our Phase 2 (QuantKey + requires_sm) fits naturally
    → → Better than per-code guard (#45038)

  If #43461 merges (ABC retained):
    → MoEKernelOracle ABC → rigid interface → harder to extend with requires_sm
    → Need to add SM guard differently → possibly as separate ABC subclass
    → → Our Phase 2 requires more structural changes → harder path

★★★★★ Action: Monitor #43373 vs #43461 resolution → adapt our Phase 2 strategy accordingly
```

### References

- Issue #32268: https://github.com/vllm-project/vllm/issues/32268
- Refactor prep: notebook/projects/vllm-32268-quantkey-refactor-prep.md
- SM89 contribution strategy: notebook/projects/vllm-sm89-contribution-strategy.md

---

## 5. #45731: Triton 3.7.1 Upgrade — Zero Reviews → Makes Fusion Guard MORE Needed

### Overview

★★★★★★★★★ #45731 proposes upgrading PyTorch to 2.13.0 with Triton 3.7.1. Currently OPEN with ZERO reviews. This upgrade makes our P9 Fusion Guard MORE needed, not less.

### Key Details

```
★★★★★★★★★ #45731 status:

  → PyTorch → 2.13.0
  → Triton → 3.7.1 (from 3.7 in PyTorch 2.12)
  → torchvision → 0.28.0
  → Currently in test channel build → ZERO reviews
  → No reviewer engagement → uncertain timeline

★★★★★★★★★ Why this makes Fusion Guard MORE needed:

  1. Triton 3.7.1 → new autotuning behaviors → MORE kernel configs per SM architecture
     → More configs → more variability → more batch-dependent behavior on SM89
     → The more autotuning Triton does → the more SM89 gets different configs → more divergence

  2. PyTorch 2.12 max_autotune → layout deferral now opt-in
     → The exacerbating SM89 path was reverted → but opt-in path still accessible
     → Triton 3.7.1 may reintroduce similar behavior → Fusion Guard blocks it

  3. vLLM currently uses Triton 3.3.x → jumping to 3.7.1 = major version gap
     → Many autotuning changes accumulated → significant SM89 behavior shifts
     → Without Fusion Guard → vLLM on SM89 becomes even more unpredictable

★★★★★★★★★ Critical implication:
  → If #45731 merges WITHOUT Fusion Guard → SM89 batch invariance could WORSEN
  → Triton 3.7.1 → more autotuning → more SM89 kernel variability → more divergence
  → Our P9 Fusion Guard must be submitted BEFORE or alongside #45731 merge
  → → P9 is now MORE time-sensitive → not less!
```

### Action Items

★★★★★★★★★ P9 Fusion Guard is now MORE urgent:
- File PyTorch issue (pre-step) → issue draft ready → must submit soon
- If #45731 starts getting reviews → comment on it → raise SM89 concerns
- The Fusion Guard blocks problematic fusions regardless of Triton version → version-agnostic protection

### References

- PR: https://github.com/vllm-project/vllm/pull/45731
- Fusion Guard issue draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- v0.23 scan: notebook/projects/vllm-v023-latest-developments-scan-2026-06-16.md

---

## 6. BudgetRefiner: NO vLLM PR Exists — P10 Remains UNIQUE Contribution Opportunity

### Status Confirmation

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ NO BudgetRefiner PR exists in vLLM upstream. NO SLO-aware scheduling PRs found.
★★★★★★★★★ This confirms our BudgetRefiner SLO contribution remains UNIQUE.
★★★★★★★★★ RTX 4090 profile data = NO other vLLM contributor has this.
★★★★★★★★★ BudgetRefiner ranks #1 vLLM contribution priority (confirmed by multiple scans).
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### BudgetRefiner SLO Integration Points (Verified Against Current Checkout)

★★★★★★★★★ Three exact integration points in vLLM V1 scheduler.py:

```
★★★★★★★★★ Integration Point A: token_budget (line 407)
  → Current: token_budget = self.max_num_scheduled_tokens (STATIC constant)
  → Proposed: BudgetRefiner.refine_budget(count decode → lookup → return adjusted)
  → ★★★★★★★★ Budget becomes DYNAMIC → adapts to decode load
  → Only 3 lines of code change needed at this point

★★★★★★★★★ Integration Point B: decode-first reorder (before line 430)
  → Current: self.running (plain list, FCFS order)
  → Proposed: decode_reqs + prefill_reqs → self.running = d_lst + p_lst
  → ★★★★★★★★ 4 lines change → decode protected from compute starvation
  → self.running is a plain list → trivial to reorder → no structural change needed

★★★★★★★★★ Integration Point C: dynamic max_seqs (line 629)
  → Current: len(self.running) == self.max_num_running_reqs (STATIC limit)
  → Proposed: effective_max_seqs = BudgetRefiner.refine_max_seqs(...)
  → ★★★★★★★★ Reduces concurrent requests when decode-heavy → SLO protection
```

### 7 Files ~300 LOC for Upstream PR

```
★★★★★★★★★ Complete change list (7 files, ~300 lines total):

  File 1: vllm/v1/core/sched/scheduler.py (~20 lines of changes)
    → __init__: add BudgetRefiner instance
    → schedule(): dynamic token_budget (line 407)
    → schedule(): decode-first reordering (before line 430)
    → schedule(): dynamic max_seqs (line 629)
    → schedule(): add budget_refiner_info to SchedulerOutput

  File 2: vllm/v1/core/sched/budget_refiner.py (~105 lines, NEW FILE)
    → BudgetRefiner class → SLO-aware dynamic token budget
    → refine_budget() → main entry: compute optimal budget given SLO
    → _read_lookup_table() → load profile_table.csv
    → _align_key() → match runtime config to profile table entries
    → _get_max_budget() → calculate max tokens within SLO time

  File 3: vllm/v1/core/sched/output.py (~20 lines addition)
    → BudgetRefinerInfo dataclass → observability metadata
    → budget_refiner_info field in SchedulerOutput

  File 4: vllm/config/scheduler.py (~15 lines addition)
    → budget_refiner_enabled: bool = False
    → slo_limits_ms: float = 0.0 (disabled by default)
    → decode_first_enabled: bool = True
    → profile_table_path: str | None = None

  File 5: vllm/v1/core/arg_utils.py (~10 lines addition)
    → CLI arguments for BudgetRefiner config

  File 6: vllm/v1/metrics/stats.py (~15 lines addition)
    → BudgetRefinerStats → metrics/logging

  File 7: profile_table/ directory → RTX 4090 profile data (CSV files)
    → profile_table_rtx4090.csv → our UNIQUE contribution
    → Community can add H100/A100/other GPU profiles
```

### BudgetRefiner Core Logic: 58 Lines, 100% GPU-Generic

★★★★★★★★★ The BudgetRefiner class from vLLM-Ascend is only 58 lines of core logic and 100% GPU-generic:

```
★★★★★★★★★ BudgetRefiner class architecture (scheduler_dynamic_batch.py:33-90):

  __init__ (lines 33-44): slo_limit > 0 → enabled, forces chunked_prefill=True, loads profile_table.csv
  _read_lookup_table() (lines 46-63): Load CSV → group by (ctx_len, d_num) → filter cost ≤ slo_limit → find max chunk_size
  _align_key() (lines 65-68): Aligns runtime value to nearest valid key ≥ value (conservative → never under-estimate!)
  _get_max_budget() (lines 70-82): Align ctx_len and d_num → lookup → fallback to default_budget if miss (3 fallback paths)
  refine_budget() (lines 84-90): If not enabled → return original. Count decode requests → call _get_max_budget → return adjusted budget
```

### BudgetRefiner ONLY Throttles Prefill When ACTIVE Decode Requests

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ CRITICAL INSIGHT: BudgetRefiner ONLY throttles prefill when there are ACTIVE decode requests!
★★★★★★★★★ If no decode requests → full budget available → prefill gets all tokens → ZERO impact on pure-prefill scenarios!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

This means:
- Pure-prefill workloads (batch inference, offline processing) → BudgetRefiner transparent → no overhead
- Mixed decode+prefill workloads (online serving, chat) → BudgetRefiner actively protects decode SLO
- Budget DROPS as decode load increases: d_num=0 → 1024 (full), d_num=100 → 768 (25% reduction), d_num=255 → 512 (50% reduction)

### RTX 4090 profile_table.csv Requirements

```
★★★★★★★★★ profile_table.csv RTX 4090 requirements:

  Required columns (BudgetRefiner uses 4 of 5):
    → chunk_size: prefill token budget (int) → OUTPUT of BudgetRefiner!
    → d_num: number of decode requests (int) → 0-255 range (RTX 4090: 0-64 practical)
    → ctx_len: average decode context length (int) → 128, 256, 512, 1024, 2048
    → cost: measured iteration time in milliseconds (float)
    → p_len: prefill length (int) → NOT used by BudgetRefiner → can be omitted

★★★★★★★★★ RTX 4090 row count estimate:
    → ~340 rows needed
    → 5 ctx_len values × 17 d_num values × 4 chunk_size values = 340
    → MUCH smaller than Ascend's 10,875 rows → feasible to collect in ~2-3 GPU hours

★★★★★★★★★ Three fallback paths → never crashes → production-safe:
    1. Lookup miss → return default_budget → never crashes
    2. No CSV loaded → return default_budget → zero overhead
    3. slo_limit ≤ 0 → disabled → original budget → ZERO impact
```

### BudgetRefiner + Watermark: Complementary, Not Overlapping

```
★★★★★★★★★ Two types of pressure → two complementary mechanisms:

  KV cache pressure (memory bottleneck):
    → KV blocks filling up → no room for new tokens
    → Handled by Watermark #44594 (reactive) → preemptions -82%, ITL p99 -56%

  Compute time pressure (compute bottleneck):
    → Too many prefill tokens → decode ITL exceeds SLO
    → Handled by BudgetRefiner (proactive) → reduces prefill budget → protects decode

★★★★★★★★★ Together they approach ZERO preemptions:
  → BudgetRefiner reduces prefill token count → fewer KV blocks consumed → KV pressure reduced
  → Watermark triggers less → less preemption → fewer WAITING/PREEMPTED requests
  → BudgetRefiner sees fewer requests → budget relaxes → more prefill admitted
  → → Self-regulating feedback loop → system stabilizes!
```

### References

- Contribution plan: notebook/projects/budgetrefiner-vllm-contribution-plan.md
- PR draft: notebook/projects/budgetrefiner-vllm-pr-draft.md
- Source reading: notebook/projects/budgetrefiner-slo-source-reading.md
- Scheduler integration: notebook/projects/vllm-v1-scheduler-budgetrefiner-integration.md
- Watermark synthesis: notebook/projects/watermark-budgetrefiner-complementary-synthesis.md
- Profile tool: tools/profile_vllm_budget.py
- OSS analysis: notebook/fundamentals/rtx4090-oss-contribution-opportunity-analysis.md

---

## 7. Additional Developments from v0.23.0+

### Watermark #44594 — MERGED (in v0.23.0)

★★★★★★★★★ Watermark PR merged June 11, included in v0.23.0 release:
- 4-layer KV cache thrashing prevention mechanism
- Admission gate + watermark headroom + prefix ref_cnt protection + self-preemption detection
- watermark=0.05 → preemptions -82%, ITL p99 -56%, throughput +5.1%
- MUST set watermark=0.05 in RTX 4090 config

### INT4 Triton Fallback #43731 — MERGED (in v0.23.0)

★★★★★★★ INT4 Triton fallback merged May 27 → W4A16 Triton fallback for non-Marlin-aligned shapes on SM89 → closes SM89 quantization gap

### HMA-by-Default #41847 — MERGED (in v0.23.0)

★★★★ HMA merged May 26 → 8 HMA connectors, startup OOM prevention for 24GB GPUs. SlidingWindow prefix caching caveat.

### FP8 KV Guard #45038 — OPEN (not merged into v0.23.0)

★★★★ FP8 KV guard validated on L4 (SM89) by community member → still needs merge. Prevents compressed-tensors FP8 KV crash on SM89.

### MRv2 Developments

★★★★ MRv2 now DEFAULT for Llama + Mistral dense models (in v0.23.0):
- Qwen2.5 still uses MRv1
- GraniteMOE MRv2 enable (#45461) OPEN
- DSV4 FlashMLA tile metadata fix (#44069) CLOSED → fixed crash
- verl MRv2 interaction: AsyncLLM.generate() handles two-step internally → likely safe

### Quantization Expansion Trend

★★★★ Multiple PRs expanding quant below SM90:
- #45306: modelopt_mixed on SM80/86 (NVFP4 + FP8 on Ampere via Marlin)
- #45735: ModelOpt mixed precision + NVFP4 runtime formats
- #45744: MiniMax M3 FP8 sparse GQA (Triton decode for non-SM100)
- #45738: NVFP4 clamped SwiGLU on FlashInfer-CUTLASS MoE
- #45739: NVFP4 scale buffer zero-init (Blackwell regression)
- **Trend**: Quantization support actively expanding to sub-SM90 → SM89 gap narrowing incrementally

### verl Integration Developments

★★★★ vLLM developments affecting verl RTX 4090:
- #44483: Illegal memory access during partial wake_up (verl sleep/wake race condition) — OPEN
- #45715: LoRA shrink buffer fix (TP>1 only, RTX 4090 NOT affected)
- #45357: Defer block freeing — MERGED June 15 → async scheduling + PD KV consumer race fix

### PD Disaggregation + Spec Decoding

★★★★ PD and spec decode developments:
- #45280: PD role-aware spec decoding — OPEN → auto-detects P vs D roles
- #45283: PD skip speculator on Prefill instance — OPEN
- #45340: CP-scaled scheduler block accounting (NIXL/Mooncake PD) — OPEN

---

## 8. Development Interdependency Map

```
★★★★★★★★★ How the 6 key developments connect:

  #39096 (batch invariance) ←──── #45731 (Triton 3.7.1)
    ↑ root cause                      ↑ makes it WORSE
    │                                 │
  #40628 (IR RFC) ←── P9 Fusion Guard ←── Issue draft ready
    ↑ architectural side              ↑ compiler side
    │                                 │
  BudgetRefiner ←── #42120 (MoE+LoRA)
    ↑ scheduling needs                    ↑ MoE serving maturity
    │                                     │
  #43373/#43461 ←── P6 Triton swiglu
    ↑ QuantKey structure           ↑ MoE quantization kernel
    │                              │
  ALL developments ←── RTX 4090 profile data (our UNIQUE contribution)
```

---

## 9. RTX 4090 Action Items

```
★★★★★★★★★ RTX 4090 Action Items (Priority Order):

  1. ★★★★★★★★★★★★★★★★★★★★★★★★★★ P10 BudgetRefiner SLO:
     → GPU MUST come online → collect RTX 4090 profile_table.csv (~340 rows)
     → File RFC issue on vllm-project/vllm → "[Feature] SLO-aware dynamic token budget"
     → Implement BudgetRefiner class (105 lines GPU-generic) + scheduler integration (~20 lines)
     → NO competing PR → UNIQUE contribution → highest priority

  2. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ P9 Inductor Fusion Guard:
     → File PyTorch issue FIRST (issue draft ready) → pre-step before PR
     → Then submit 5-line choices.py guard PR → props.major < 9 → WhyNoFuse
     → Monitor #45731 Triton 3.7.1 upgrade → comment with SM89 concerns if it gets reviews
     → NO competing PR → strong contribution opportunity → MORE urgent due to #45731

  3. ★★★★★★★★★★★★★ Monitor #43373/#43461 conflict:
     → Track which QuantKey PR merges → adapt Phase 2 SM guard strategy
     → If #43373 merges → QuantKey(method="fp8_e4m3_kv", requires_sm=90) → systematic
     → If #43461 merges → need different SM guard approach → harder path

  4. ★★★★★★★★ Tier 1 Comment Posts (4 drafts ready):
     → Post #39096 comment → SM89 dual failure path + Inductor root cause + RTX 4090 perspective
     → Post #44879/#45038 comment → SM89 FP8 limitation matrix + INT8KV alternative
     → Post #44701 comment → LoRA+prefix cache collision + SGLang comparison
     → Establish SM89 expert reputation → prerequisite for P10/P9 PRs

  5. ★★★★★★ Tier 2 Quick PRs:
     → #43204 cleanup PR → simplest first merge → establish credibility
     → #32268 QuantKey refactor → pure refactoring → low risk
     → Execute AFTER Tier 1 comments → build trust before larger PRs

  6. ★★★★★ FP8 KV Guard #45038:
     → Monitor for merge → SM89 validated
     → Comment with RTX 4090-specific testing data when GPU online

  7. ★★★★★ Watermark:
     → Already in v0.23.0 → MUST set watermark=0.05 in RTX 4090 config
     → BudgetRefiner + watermark = complementary → approach zero preemptions

  8. ★★★★ P6 Triton dequant_swiglu_quant:
     → Kernel prototype ready → tl.constexpr → deterministic → batch-invariant
     → Need: GPU for correctness + benchmark validation → after P9 filed
     → Complementary: P9 blocks bad fusions → P6 provides good fused path
```

---

## 10. Key Findings Summary

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

  1. ★★★★★★★★★★★★★★★★★★★★★ #39096 batch invariance STILL UNFIXED
     → Inductor root cause confirmed → torch.compile alone breaks SM<90
     → enforce_eager=True = only workaround → ~10-15% throughput loss
     → P9 Fusion Guard = best fix → 5-line choices.py guard → issue draft ready

  2. ★★★★★★★★ #40628 RFC: batch-invariant dispatching in vLLM IR
     → 5 options → Option 2 (IR annotations) recommended
     → Complementary to P9 Fusion Guard → vLLM marks ops → Inductor respects annotations
     → Together: complete architectural + compiler solution

  3. ★★★★★★★★ #42120 FP8 MoE+LoRA close to merge (merge conflicts)
     → Validated SM 12.0 → MoE+LoRA serving pattern established
     → RTX 4090 NOT directly (FP8 requires SM90+) → but pattern applies to INT4/INT8 MoE+LoRA
     → P6 Triton swiglu provides SM89 quantized MoE path → complementary

  4. ★★★★★★★★ #43373/#43461 QuantKey conflict
     → #43373 removes oracle → humming_utils.py → simpler → better for requires_sm extension
     → #43461 keeps ABC → rigid → harder to extend
     → Outcome determines our Phase 2 SM89 guard strategy → monitor

  5. ★★★★★★★★★★★★★★★★★★★★★★★★★ #45731 Triton 3.7.1 upgrade
     → ZERO reviews → uncertain timeline
     → Makes Fusion Guard MORE needed → more autotuning → more SM89 variability
     → P9 MORE time-sensitive → not less → must submit before or alongside #45731

  6. ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ BudgetRefiner = P10 UNIQUE
     → NO vLLM PR exists → NO SLO-aware scheduling found
     → RTX 4090 profile data = NO other contributor has this
     → 3 integration points verified (lines 407/430/629)
     → 7 files ~300 LOC total → minimal footprint for major feature
     → 58 lines core logic → 100% GPU-generic → only profile_table.csv HW-specific
     → ONLY throttles prefill when ACTIVE decode requests → zero pure-prefill impact
     → ~340 rows RTX 4090 profile_table → feasible in ~2-3 GPU hours

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## References

- v0.23.0 developments scan: notebook/projects/vllm-v023-latest-developments-scan-2026-06-16.md
- BudgetRefiner SLO source reading: notebook/projects/budgetrefiner-slo-source-reading.md
- BudgetRefiner contribution plan: notebook/projects/budgetrefiner-vllm-contribution-plan.md
- BudgetRefiner PR draft: notebook/projects/budgetrefiner-vllm-pr-draft.md
- BudgetRefiner scheduler integration: notebook/projects/vllm-v1-scheduler-budgetrefiner-integration.md
- Watermark+BudgetRefiner synthesis: notebook/projects/watermark-budgetrefiner-complementary-synthesis.md
- SM89 batch invariance deep reading: notebook/projects/vllm-sm89-batch-invariance-deep-reading.md
- SM89 batch invariance bug reading: notebook/projects/vllm-sm89-batch-invariance-bug-reading.md
- Batch invariance comment draft: notebook/projects/vllm-39096-batch-invariance-comment-draft.md
- QuantKey refactor prep: notebook/projects/vllm-32268-quantkey-refactor-prep.md
- SM89 contribution strategy: notebook/projects/vllm-sm89-contribution-strategy.md
- Fusion Guard PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard issue draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- OSS contribution analysis: notebook/fundamentals/rtx4090-oss-contribution-opportunity-analysis.md
- PR tracker: notebook/projects/seven-framework-pr-tracker.md
- Profile tool: tools/profile_vllm_budget.py
- Cross-framework deterministic: notebook/projects/deterministic-inference-cross-framework-comparison.md
- Diary entries: diary/2026-06-18.md (entries 232-237)
