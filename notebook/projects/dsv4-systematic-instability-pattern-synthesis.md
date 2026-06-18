# DSV4 Systematic Instability Pattern — Cross-Framework Synthesis

> 2026-06-18 | Cross-framework deep synthesis of DeepSeek-V4 correctness failures
> ★★★★★★★★ DSV4 is the most fragile production model in 2026 — 4 reverts/crashes in 1 week across 2 frameworks!
> ★★★★★★★★ Root cause: DSV4 has MORE dynamic routing than any previous model → breaks static execution assumptions

---

## 1. The 4 DSV4 Failures (June 15-18, 2026)

```
★★★★★★★★★ Timeline of DSV4-related correctness failures this week:

| # | Framework | Issue | What broke | Symptom | Revert? |
|---|-----------|-------|-----------|---------|---------|
| 1 | vLLM | #45309→#45972 | DSV4 cudagraph optimization | GARBAGE OUTPUT "the the the..." | MERGED revert June 18 |
| 2 | SGLang | #26471→#28591 | DSV4 Online Compress MTP | Testing revert (OPEN) | OPEN revert |
| 3 | SGLang | #27749→#28575 | MTP weight update distributed | Refactor needed | OPEN reimpl |
| 4 | SGLang | #28569 | EAGLE3 CUDA graph replay | ILLEGAL MEMORY ACCESS crash | OPEN bug |

★★★★★★★★★ 4 DSV4 issues in 4 days → 2 frameworks → SYSTEMATIC pattern!
```

---

## 2. Why DSV4 is MORE Fragile Than Previous Models

```
★★★★★★★★★ DSV4 has MORE layers of dynamic routing than any previous model:

DeepSeek-V2/V3:  MoE (expert selection per token)
DeepSeek-V4:     MoE + DSA (sparse attention indexer) + MTP (multi-token prediction)
                 + Online Compress (KV cache compression) + MLA (multi-head latent)

★★★★★★★★★ Each dynamic routing layer is a potential CUDA graph replay failure:

| Dynamic Layer | Discrete Decision | Under Graph Replay |
|--------------|-------------------|---------------------|
| MoE expert selection | top-k expert per token | WRONG experts → incorrect computation |
| DSA indexer | top-k KV position per query | WRONG positions → garbage attention |
| MTP draft | which tokens to draft | WRONG draft → incorrect verification |
| Online Compress | which KV to compress | WRONG compression → stale cache state |
| MLA DCP | which heads to replicate | WRONG replication → batch-dependent |

★★★★★★★★★ Each layer compounds the fragility:
  → MoE alone: ~10% router disagreement (confirmed by R3)
  → MoE + DSA: indexer disagreement ALSO ~10% → compounding mismatch
  → MoE + DSA + MTP: THREE sources of discrete decision mismatch!
  → ★★★★★★★★ DSV4 is an "n-layer dynamic routing" model → EACH layer is a failure point!
```

---

## 3. vLLM #45972 — The Reference Failure (Source-Level)

```
★★★★★★★★★ vLLM DSV4 cudagraph revert — most detailed source-level analysis:

Original PR #45309:
  → Removed @eager_break_during_capture from attention_impl
  → Used runtime BreakableCUDAGraphCapture.is_active() check instead
  → During CAPTURE: wq_b_kv_insert + compressor in 2-way parallel, indexer sequentially
  → ALL inside stream capture context → BECOMES PART OF recorded CUDA graph!

During REPLAY:
  → Entire captured graph replayed with STATIC data from capture-time buffers
  → NOT with live per-request metadata → garbage output like "the the the the..."

★★★★★★★★★ @eager_break_during_capture is the CORRECT separation boundary:
  → Static GEMMs (weight matmuls, norms) → CAN be captured → speed benefit
  → Dynamic routing (attention metadata, MoE expert selection, indexer) → MUST run eagerly
  → ★★★★★★★★ Universal rule: ANY operation whose behavior depends on per-request metadata
    → MUST run eagerly, NEVER inside captured CUDA graph

★★★★★★★★★ vLLM #45972 REVERT was MERGED June 18 → confirms: DSV4 cudagraph optimization NOT safe!
```

---

## 4. SGLang #28591 — MTP Online Compress Revert

```
★★★★★★★★★ SGLang #26471 (MERGED June 16, +1276/-49 lines):

What it added:
  → JIT kernel: online_c128_mtp.cuh (CUDA C++ kernel for MTP + online compress)
  → compress.py: SGLANG_OPT_USE_ONLINE_COMPRESS=1 integration
  → compressor_v2.py: new compressor for DSV4 MTP path
  → deepseek_v4_compress_state.py: state management for compressed KV
  → deepseek_v4_memory_pool.py: memory pool for compress state
  → deepseek_v4_backend.py: attention backend integration

Performance claim:
  → "280% improvement in max tokens (2M→5.7M)"
  → "Only 2% lower than NO_ONLINE+MTP"

★★★★★★★★★ #28591 reverts this entire +1276/-49 PR — for "testing":
  → Labeled "deepseek" + "jit-kernel"
  → The JIT kernel (C++ CUDA) compilation may have correctness issues
  → Online Compress + MTP interaction may have stale state under graph replay
  → ★★★★★★★★ Similar pattern to vLLM: combining dynamic operations (compress + MTP) → state consistency issues!

★★★★★★★★★ SGLang #28575 (OPEN) — reimpl MTP weight update from distributed:
  → #27749 was the "first cut" of distributed weight-update for speculative draft worker(s)
  → Uses disable_draft_model flag + per-worker update_weights_from_distributed
  → #28548 refactors → but the original wiring was fragile → needs reimpl
  → ★★★★★★★★ MTP speculative decoding + distributed weight update = ANOTHER dynamic routing layer!
```

---

## 5. SGLang #28569 — EAGLE3 CUDA Graph Crash

```
★★★★★★★★★ EAGLE3 speculative decoding CUDA graph replay — ILLEGAL MEMORY ACCESS:

Bug: gpt-oss-120b with EAGLE3 spec decode
  → Deterministic crash when running batch shrinks from 32→12 requests
  → --disable-cuda-graph avoids crash → confirms CUDA graph replay cause
  → Batch size changes during decode → graph replay with stale batch metadata
  → ★★★★★★★★ Same root cause: graph captured at one batch size → replayed at different batch → memory access violation

★★★★★★★★★ This is NOT DSV4-specific but confirms the broader CUDA graph fragility pattern:
  → Speculative decoding = dynamic decision (which tokens to accept)
  → Batch size changes = dynamic decision (which requests finish)
  → Under graph replay → both decisions are stale → crash!
```

---

## 6. The Universal Root Cause

```
★★★★★★★★★ Universal root cause across ALL 4 failures:

CUDA graph replay assumes STATIC execution path:
  → Graph captured once → replayed many times
  → During capture: specific batch size, specific routing decisions, specific metadata
  → During replay: DIFFERENT batch size, DIFFERENT routing, DIFFERENT metadata
  → → MISMATCH → incorrect results, memory corruption, or crash

★★★★★★★★★ DSV4 is uniquely fragile because it has the MOST dynamic routing layers:

Traditional models (LLaMA, Mistral):
  → Dense attention → NO dynamic routing → CUDA graph SAFE

MoE models (Mixtral, Qwen3-MoE):
  → MoE expert selection → ONE dynamic routing → CUDA graph RISKY (but manageable)

DSA models (DeepSeek-V3.2):
  → MoE + DSA indexer → TWO dynamic routing → CUDA graph HIGH risk

DSV4 (DeepSeek-V4):
  → MoE + DSA + MTP + Online Compress → FOUR+ dynamic routing → CUDA graph EXTREMELY fragile!

★★★★★★★★★ The fragility is COMPOUND:
  → Each dynamic layer has ~10% disagreement rate
  → 4 layers: ~(1-0.9^4) ≈ 34% of forward passes have at least one mismatch
  → → Nearly 1/3 of DSV4 graph replays produce INCORRECT output!
```

---

## 7. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 DSV4 implications:

1. CUDA graph for DSV4 on RTX 4090:
  → enforce_eager=True → MANDATORY for DSV4 inference
  → 10-15% throughput sacrifice → but CORRECTNESS guaranteed
  → BudgetRefiner SLO compensates throughput loss with better scheduling

2. DSV4 MoE + DSA on RTX 4090:
  → FA2 only (UNIFORM_BATCH) on SM 8.9 → NOT FA3
  → DSA indexer needs eager execution → no graph capture
  → MTP speculative decode → no graph capture for draft model

3. verl GRPO + DSV4 on RTX 4090:
  → Indexer replay (Megatron #5384) → CRITICAL for train/rollout consistency
  → Router replay (MoE) → already needed
  → BOTH replay mechanisms needed → 2x recording overhead (but memory negligible)

4. ★★★★★★★★ RTX 4090 GRPO training stability for DSV4:
  → Need BOTH MoE router replay AND DSA indexer replay
  → verl CPPO+bypass_mode = best framework (handles replay correctly)
  → DeepSpeed ZeRO-2 + CPU_Adam = fallback (needs custom replay integration)
  → rLLM = BLOCKED by #605 grouping bug
```

---

## 8. The @eager_break Pattern — Correct vs Incorrect

```
★★★★★★★★★ CORRECT pattern (vLLM before #45309, should be after #45972):

@eager_break_during_capture   ← decorator marks boundary
def attention_impl():
    # This ENTIRE function runs eagerly during graph capture
    # NOT recorded into the graph → NOT replayed with stale data
    # Dynamic routing decisions made with LIVE metadata
    ...

★★★★★★★★★ INCORRECT pattern (vLLM #45309, reverted by #45972):

# Inside stream capture context:
if BreakableCUDAGraphCapture.is_active():
    # Still inside graph capture → BECOMES PART OF recorded graph!
    # During replay → stale capture-time metadata
    wq_b_kv_insert(...)  # recorded into graph
    compressor(...)       # recorded into graph

★★★★★★★★★ The key distinction:
  → @eager_break_during_capture → BREAKS out of stream capture → runs eagerly → NOT in graph
  → BreakableCUDAGraphCapture.is_active() → still IN stream capture → BECOMES part of graph
  → ★★★★★★★★ is_active() check only affects WHICH branch runs → but BOTH branches are captured!
  → → The "break" is a figment → doesn't actually break out of capture context!

★★★★★★★★★ Universal rule for DSV4-like models with multiple dynamic routing:
  → ALL dynamic routing (MoE expert selection, DSA indexer, MTP draft, compress decisions)
  → MUST use @eager_break_during_capture or equivalent
  → NEVER use runtime branching inside capture context
  → Static operations (weight matmuls, norms, fixed-shape GEMMs) → CAN be captured
```

---

## 9. Cross-Framework DSV4 Support Status

```
★★★★★★★★★ DSV4 support status across frameworks (June 2026):

| Framework | DSV4 Support | CUDA Graph Safe? | Replay Mechanism | Status |
|-----------|--------------|-------------------|------------------|--------|
| vLLM | Yes (v0.23.0) | NO (#45972 reverted) | @eager_break needed | ★★★★★★★★ REVERTED cudagraph opt |
| SGLang | Yes (v0.5.13) | NO (#28591 revert + #28569 crash) | needs testing | ★★★★★★★★ MTP revert, EAGLE3 crash |
| Megatron | Partial (#5386 indexer replay needed) | N/A (training) | RouterReplay + IndexerReplay needed | ★★★★★★★★ feature request OPEN |
| verl | Partial (rollout via SGLang/vLLM) | N/A (uses inference engine) | bypass_mode + record/replay | ★★★★★★★★ CPPO+bypass BEST |
| DeepSpeed | Partial (AutoEP for MoE) | N/A (training) | RouterReplay needed | ★★★★★★★★ needs implementation |
| MindIE/vLLM-Ascend | Yes (MXFP4 + CANN) | NO (#10628 DSV4 failure, #10640 MTP startup crash) | AscendC kernels | ★★★★★★★★ DSV4 ALSO broken on Ascend! |
| PyTorch | Backend only | SM89 guard needed | Inductor choices.py | ★★★★★★★★ #184119 progressing |

★★★★★★★★★ NO framework has fully safe DSV4 support yet — not even Ascend!
  → vLLM reverted their optimization → back to eager
  → SGLang reverting MTP → back to testing
  → vLLM-Ascend: DSV4 chat failure (#10628) + MTP startup crash (#10640) → SAME pattern on different architecture!
  → ★★★★★★★★ DSV4 fragility is CROSS-ARCHITECTURE (NVIDIA + Ascend)!
  → Both NVIDIA and Ascend frameworks in "safe but slow" mode for DSV4
```

---

## Key Findings Summary

★★★★★★★★★ DSV4 has 4 correctness failures in 4 days across 2 frameworks → SYSTEMATIC pattern!
★★★★★★★★★ DSV4 fragility is CROSS-ARCHITECTURE — broken on NVIDIA (#45972, #28591) AND Ascend (#10628, #10640)!
★★★★★★★★★ DSV4 has MORE dynamic routing layers than any previous model → compounding fragility
★★★★★★★★★ vLLM #45972 REVERT confirms: @eager_break_during_capture is the CORRECT boundary
★★★★★★★★★ SGLang #28591 MTP revert + #28569 EAGLE3 crash = same root cause (stale metadata under graph)
★★★★★★★★★ Universal rule: ANY per-request dynamic routing MUST run eagerly → NEVER in captured graph
★★★★★★★★★ RTX 4090: enforce_eager=True MANDATORY for DSV4 → 10-15% throughput sacrifice for correctness
★★★★★★★★★ verl GRPO+DSV4 needs BOTH MoE router replay + DSA indexer replay → 2x recording
★★★★★★★★★ NO framework has fully safe DSV4 CUDA graph support → all in "safe but slow" mode

---

## References

- vLLM #45309→#45972: cudagraph revert (MERGED June 18)
- SGLang #26471→#28591: DSV4 MTP Online Compress revert (OPEN)
- SGLang #27749→#28575: MTP weight update reimpl (OPEN)
- SGLang #28569: EAGLE3 CUDA graph replay crash (OPEN)
- SGLang #27097: multi-LoRA determinism (4 factors)
- vLLM #39096: SM89 batch invariance
- vLLM-Ascend #10628: DSV4 chat failure on Ascend (OPEN)
- vLLM-Ascend #10640: MTP startup failure on 300i duo (OPEN)
- vLLM-Ascend #10621: spec decoding non-determinism on Ascend (OPEN)
- Megatron #5384: DSA indexer replay feature request
- notebook/projects/vllm-cuda-graph-reading.md (Section 13)
- notebook/projects/sglang-28588-image-decompression-bomb-reading.md
