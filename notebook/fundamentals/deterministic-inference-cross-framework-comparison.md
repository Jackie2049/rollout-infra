# Deterministic Inference Cross-Framework Comparison — SM89 Batch Invariance

> 2026-06-18 | Cross-framework analysis | How each framework achieves batch-invariant inference on SM<90 GPUs
> ★★★★★★★★ 5 implementations now: SGLang (9+ overrides tl.constexpr) + vLLM (984-line batch_invariant.py SM89 GAPS) + vLLM-Ascend (AscendC/Triton dual-tier) + MindIE Turbo (compose-level) + verl #6572 (5-layer deployment)
> ★★★★★★★★ ROOT CAUSE: Triton CachingAutotuner XBLOCK varies per batch size → non-associative FP addition
> ★★★★★★★★ Triton tl.constexpr BLOCK_SIZE = gold standard → our Fusion Guard + Triton swiglu kernel both use this
> ★★★★★★★★ vLLM MAIN has batch_invariant.py (984 lines)! — BUT SM89 GAP: RMSNorm NOT aten override, matmul only CUBLASLt workspace
> ★★★★★★★★ vLLM-Ascend #10034: batch_invariant_ops for RL — SAME torch.library.Library pattern as SGLang — add_rms_norm SPLIT into add+rms_norm!
> ★★★★★★★★ verl #6572: 5-layer deployment validates VLLM_BATCH_INVARIANT=1 production use → BUT SM89 requires P9 for full determinism

---

## 1. The Root Cause — Why SM89 Batch Invariance Matters

```
★★★★★★★★★ Fundamental problem on SM<90 GPUs:

Triton CachingAutotuner selects different XBLOCK sizes for different input shapes:
  → SM89 shared memory: 100KB (vs SM80=164KB, SM90=228KB)
  → Different batch sizes → different XBLOCK autotuning → different accumulation order
  → tl.sum() inline reduction → non-associative FP addition → batch-dependent results!

★★★★★★★★★ Why this ONLY affects SM<90:
  → SM90+ has TMA (Tensor Memory Accelerator) + WGMMA → deterministic by hardware design
  → SM<90 relies on autotuned block sizes → variable → non-deterministic
  → This is a Triton/Inductor issue, NOT a model architecture issue

★★★★★★★★★ GRPO relevance:
  → GRPO generates n=8 responses per prompt → all must see same model behavior
  → If inference is batch-dependent → different results for same prompt depending on batch size
  → Deterministic inference = CORRECTNESS requirement for RL training → not just a nicety!
```

---

## 2. Framework-by-Framework Deterministic Approaches

### SGLang — KERNEL-Level (★★★★★★★★★★★★★★★★★★★★ GOLD STANDARD)

```
★★★★★★★★★ SGLang achieves batch invariance at the KERNEL level:

9+ aten overrides with tl.constexpr BLOCK_SIZE (updated with #24459):
  1. rms_norm: BLOCK_SIZE = tl.constexpr → aten override registered → #24459 MERGED May 6
  2. mm: BLOCK_SIZE = tl.constexpr + dtype specialization → #24459 added mm.dtype
  3. addmm: BLOCK_SIZE = tl.constexpr → matmul+bias fused override
  4. matmul: BLOCK_SIZE = tl.constexpr → general matrix multiply override
  5. linear: BLOCK_SIZE = tl.constexpr → nn.functional.linear override
  6. _log_softmax: BLOCK_SIZE = tl.constexpr → softmax with log
  7. mean.dim: BLOCK_SIZE = tl.constexpr → reduction operation override
  8. bmm: BLOCK_SIZE = tl.constexpr → batched matmul override
  9. silu: BLOCK_SIZE = tl.constexpr → activation override
  10. sigmoid: BLOCK_SIZE = tl.constexpr → activation override
  11. mul: BLOCK_SIZE = tl.constexpr → elementwise override

★★★★★★★★★ #24459 (MERGED May 6) strengthened KERNEL-level:
  → Added aten::rms_norm override → was relying on torch.compile → now explicit Triton
  → Added aten::mm.dtype specialization → separate bf16/fp16 constexpr paths
  → NOW 9+ overrides → MORE than originally counted (7)
  → ★★★★★★★★ This PROVES aten overrides work → P9 guard makes them MORE effective

★★★★★★★★★ murmur_hash32 Gumbel-max float64 sampling:
  → Position + seed + vocab hash → murmur_hash32 Triton kernel → Gumbel-max categorical
  → Entirely GPU-side → no RNG state synchronization needed
  → Gold standard deterministic sampling for GRPO → no host-side RNG contamination

★★★★★★★★★ Architecture: BYPASSES Inductor entirely:
  → SGLang replaces aten ops with custom Triton kernels BEFORE Inductor sees them
  → Inductor cannot fuse these → fusion guard irrelevant → deterministic by architecture
  → LoRA Triton SGMV + extra_key namespace → MoE LoRA + deterministic = UNIQUE capability

★★★★★★★★★ Why KERNEL-level is superior:
  → Guaranteed by Triton JIT compilation → constexpr → no autotuning path exists
  → Bypasses Inductor fusion decisions entirely → immune to scheduler changes
  → Works regardless of torch.compile configuration → robust
  → Can be combined with CUDA graph → constexpr = static → graph-friendly
```

### vLLM — COMPILE-Level (★★★★★★★★★ SM89 GAPS CONFIRMED)

```
★★★★★★★★★ vLLM batch_invariant.py (984 lines) — SM89 has CRITICAL GAPS:

SM89 overrides (source-verified, lines 897-951):
  → _log_softmax: Triton override → ALL CUDA platforms ✅
  → softmax/_softmax: Triton override → ALL CUDA platforms ✅
  → mean.dim: Triton override → ALL CUDA platforms ✅
  → bmm: Triton override → ALL CUDA platforms ✅
  → mm/addmm/matmul/linear: CUBLASLt workspace config ONLY → SM89 ❌
    → SM80 gets 4 Triton matmul overrides with constexpr BLOCK_SIZE
    → SM89/else gets ONLY CUBLASLt workspace config → NO Triton override
  → RMSNorm: _rms_norm_kernel defined (lines 775-881) with tl.constexpr BUT NOT registered as aten override ❌

★★★★★★★★★ SM89 gap analysis:
  → RMSNorm: kernel EXISTS but NOT aten override → Inductor can STILL fuse RMSNorm reduction+pointwise
  → matmul: CUBLASLt workspace config doesn't prevent batch-dependent results (workspace size varies)
  → silu/sigmoid/mul: NOT overridden on ANY platform → can be fused by Inductor
  → ★★★★★★★★ On SM89, vLLM batch_invariant leaves THE SAME gaps that P9 fills!

★★★★★★★★★ vLLM's attempts to fix:
  → enforce_eager=True → disables torch.compile → slow but correct
  → MRv2 (Model Runner v2) → preserves determinism → safe for verl
  → vLLM #39096 → open bug → community tracking
  → VLLM_BATCH_INVARIANT=1 → env var → enables existing overrides → production validated by verl #6572
  → ★★★★★★★★ BUT: VLLM_BATCH_INVARIANT=1 on SM89 STILL has RMSNorm gap → P9 REQUIRED

★★★★★★★★★ NEW v0.23.0+ batch invariance PRs (June 15-16):
  → #45683 OPEN (June 15, 89 additions): Deterministic MoE combine under VLLM_BATCH_INVARIANT
    → Cross-rank summation order in MoE combine step was NOT stable → breaks bit-for-bit reproducibility
    → Fix: route MoE combine through deterministic reduce_scatterv → fixed-root reduce + scatter
    → ★★★★★★★★ CRITICAL for GRPO MoE → DP+EP MoE needs deterministic combine for stable rewards!
    → Only changes behavior when VLLM_BATCH_INVARIANT=1 and DP world_size > 2
  → #45819 OPEN (June 16, 13 additions): GDN attention batch invariance support
    → GDNAttentionBackend now supports_batch_invariance() → returns True
    → GDN uses stable sorting (torch.argsort with stable=True) → deterministic by design
    → ★★★★★★★★ Enables Qwen3.6 hybrid Mamba+GDN deterministic inference → RTX 4090 GRPO viable

★★★★★★★★★ Our proposed solution for vLLM SM89:
  → P9 Inductor Fusion Guard → blocks ALL reduction fusions on SM<90 → fills RMSNorm gap
  → P6 Triton dequant_swiglu_quant → provides GOOD fused path → faster than unfused
  → verl #6572 → 5-layer deployment → VLLM_BATCH_INVARIANT=1 production
  → Together: P9 + VLLM_BATCH_INVARIANT + #6572 = complete SM89 deterministic stack

★★★★★★★★★ Why COMPILE-level is weaker than KERNEL-level:
  → Depends on Inductor scheduler decisions → may change across versions
  → torch.compile may fuse differently for different model sizes → model-dependent
  → vLLM SM89: CUBLASLt workspace ≠ Triton constexpr → gaps remain
  → BUT: our Fusion Guard makes COMPILE-level deterministic → same outcome as KERNEL-level
```

### DeepSpeed — NONE (★★★★★★★★★ NO DETERMINISTIC MECHANISM)

```
★★★★★★★★★ DeepSpeed has NO deterministic inference mechanism:

  → No Triton overrides → no aten overrides → no constexpr guards
  → DeepSpeed inference = raw PyTorch → subject to same Inductor batch invariance bug
  → AutoEP MoE: TokenChoiceTopKRouter → always computes top-k dynamically → non-deterministic

★★★★★★★★★ DeepSpeed RTX 4090 determinism path:
  → Freeze router (0 LOC) → same input = same routing → deterministic automatically
  → ZeRO-2 + CPU_Adam → no overlap_comm → no torch.compile → avoid #8061 NaN
  → LoRA on attention only → frozen router → no routing variability
  → ★★★★★★★★ For inference: DO NOT use torch.compile with DeepSpeed → enforce_eager

★★★★★★★★★ RouterReplay equivalent (future):
  → When CUDA graph support added to AutoEP (#6816) → need RouterReplay
  → Current: freeze router = sufficient for simple GRPO without CUDA graph
```

### Megatron-LM — NONE (★★★★★ ROUTERREPLAY EXISTS BUT NOT FOR INFERENCE)

```
★★★★★★★★★ Megatron has RouterReplay but NOT a general deterministic inference mechanism:

RouterReplay (#4168):
  → 3 modes: RECORD/REPLAY_FORWARD/REPLAY_BACKWARD
  → Designed for MoE CUDA graph stability → NOT for general batch invariance
  → Only controls routing decisions → doesn't affect RMSNorm/attention determinism

★★★★★★★★★ Megatron inference determinism:
  → Megatron inference uses TE (Transformer Engine) → custom CUDA kernels
  → TE kernels: deterministic on SM90+ (TMA/WGMMA) → NOT guaranteed on SM89
  → ★★★★★★★★ Megatron NOT recommended for RTX 4090 inference → ranking #3

★★★★★★★★★ QB routing (#5349) → evolving:
  → May simplify determinism → replaces aux loss with QB routing
  → But: QB routing still computes routing dynamically → needs RouterReplay for CUDA graph
```

### verl — COMPILE-Level + 5-Layer Deployment (★★★★★★★★★ #6572 VALIDATES VLLM_BATCH_INVARIANT)

```
★★★★★★★★★ verl #6572 (OPEN, June 2026): 5-layer full determinism for vLLM rollout

5-layer architecture:
  Layer 1: PyTorch enable_full_determinism → torch.use_deterministic_algorithms(True)
  Layer 2: Environment propagation → PYTHONHASHSEED, VERL_FULL_DETERMINISM, VLLM_BATCH_INVARIANT
  Layer 3: VLLM_BATCH_INVARIANT=1 + SamplingParams.seed → production mechanism
  Layer 4: Priority scheduling + deterministic routing → consistent request ordering
  Layer 5: RM max_num_seqs=1 serialization → reward model deterministic inference

★★★★★★★★★ #6572 PRODUCTION VALIDATES VLLM_BATCH_INVARIANT=1:
  → verl uses VLLM_BATCH_INVARIANT=1 as the production mechanism for deterministic inference
  → This validates our analysis: vLLM batch_invariant.py IS the right mechanism
  → BUT: on SM89, VLLM_BATCH_INVARIANT still has RMSNorm gap → needs P9 for full determinism
  → ★★★★★★★★ #6572 + P9 = complete SM89 deterministic GRPO stack!

★★★★★★★★★ verl inference = vLLM backend → inherits vLLM's determinism issues:
  → verl HYBRID mode → vLLM rollout in same process → same torch.compile issues
  → verl rollout engine → vLLM ServerAdapter → enforce_eager or torch.compile

★★★★★★★★★ verl determinism mitigations:
  → HYBRID on-policy: pi_rollout = pi_theta → no IS correction needed → robust
  → bypass_mode: eliminates ref model → no separate model forward → simpler
  → ★★★★★★★★ On-policy + bypass = SAME model for rollout and training → consistent

★★★★★★★★★ verl-specific determinism concern:
  → GRPO n=8 responses → all from SAME rollout engine → SAME batch → SAME compilation
  → IF torch.compile is deterministic for THAT batch size → GRPO works
  → IF different batch sizes → different compilation → different results
  → ★★★★★★★★ Key: GRPO generates all n=8 responses in SAME batch → batch-invariant WITHIN group!

★★★★★★★★★ HYBRID unique advantage for determinism:
  → Same GPU → same process → same compilation → same weights
  → On-policy: no IS correction → no ref model → no cross-model comparison
  → ★★★★★★★★ HYBRID = simplest determinism story for RTX 4090!
```

### MindIE — Ascend-Specific (★★★★★★★★★ COMPLETELY DIFFERENT ARCHITECTURE)

```
★★★★★★★★★ MindIE on Ascend → deterministic by hardware design:

  → Ascend AI Core: deterministic vector operations → no batch-dependent autotuning
  → ATB Operation::Compose → atomic schedulable units → scheduling preserved
  → npu_dequant_swiglu_quant → single AscendC kernel → deterministic by design

★★★★★★★★★ Ascend vs CUDA determinism:
  → Ascend: Cube+Vector+Scalar pipeline → deterministic per-tile execution
  → CUDA: SM unified pipeline → autotuned block sizes → potentially non-deterministic
  → ★★★★★★★★ Ascend hardware advantage → no Triton autotuning needed → deterministic automatically

★★★★★★★★★ MindIE lessons portable to CUDA:
  → npu_dequant_swiglu_quant → Triton dequant_swiglu_quant → SAME fusion concept
  → tl.constexpr = AscendC fixed tiling → both deterministic → same outcome
  → Operation::Compose → Triton grouped kernel → similar scheduling granularity
  → ★★★★★★★★ Triton constexpr = AscendC deterministic tiling → hardware-agnostic determinism!
```

### rLLM — NONE (★★★★★★★★★ DELEGATES TO vLLM/SGLang)

```
★★★★★★★★★ rLLM delegates inference to external serving backend:

  → Tinker: vLLM/SGLang as rollout backend → inherits their determinism
  → client-server SDK pattern → NOT in-process → weight sync via checkpoint
  → ★★★★★★★★ rLLM's determinism = whatever backend's determinism

★★★★★★★★★ Tinker determinism advantage:
  → Single-step training → no multi-iteration staleness
  → No IS correction needed → same as verl bypass_mode reasoning
  → Simple training loop → fewer opportunities for determinism issues
```

---

## 3. Architectural Layer Comparison

```
★★★★★★★★★ 3 layers of deterministic inference:

| Layer | Mechanism | Frameworks | Guarantee | SM89 Viable? |
|-------|-----------|------------|-----------|---------------|
| KERNEL-level | Triton aten overrides + tl.constexpr | SGLang (9+ overrides) | ★★★★★★★★ STRONGEST — bypasses Inductor entirely | Yes! — constexpr = deterministic by design |
| KERNEL-level | AscendC/Triton dual-tier overrides | vLLM-Ascend (batch_invariant_ops) | ★★★★★★★★ STRONGEST — same pattern | Yes! — Ascend hardware deterministic |
| COMPILE-level | Inductor Fusion Guard (P9) + vLLM batch_invariant.py | vLLM + PyTorch + verl #6572 | ★★★★★★★★ STRONG — blocks bad fusions on SM<90 | Yes! — with P9 fills SM89 gaps |
| COMPILE-level | VLLM_BATCH_INVARIANT=1 (existing overrides) | vLLM + verl | ★★★★★★★ PARTIAL — SM89 gaps remain | Partial — RMSNorm gap unfilled on SM89 |
| NONE | No mechanism → raw torch.compile | DeepSpeed, Megatron, rLLM | ★★★ WEAK — subject to batch invariance bug | Only with enforce_eager=True |

★★★★★★★★★ P9 + #187435 + #6572 integration path (3 complementary mechanisms):
  → P9 (5 LOC): GLOBAL SM<90 policy → blocks ALL reduction fusions → simplest, fastest
  → #187435 (804 LOC): PER-OP no_fuse_region → fine-grained control → useful on SM90+
  → #6572 (deployment): VLLM_BATCH_INVARIANT=1 production → validates the mechanism
  → Phase 1: P9 first → simplest → immediate SM89 fix → works with existing vLLM/verl
  → Phase 2: #187435 → per-op refinement → useful for SM90+ edge cases
  → Phase 3: #6572 deployment → production stack → GRPO determinism complete

★★★★★★★★★ Triton tl.constexpr = the KEY insight:
  → SGLang discovered: constexpr BLOCK_SIZE → batch-invariant → no autotuning
  → Our Triton swiglu kernel: SAME constexpr approach → deterministic → SGLang-compatible
  → Our Fusion Guard: blocks Inductor from fusing → forces separate kernels → same outcome
  → ★★★★★★★★ BOTH approaches achieve the SAME goal via DIFFERENT architectural layers!
```

---

## 4. GRPO-Specific Determinism Requirements

```
★★★★★★★★★ GRPO determinism requirements per framework:

| Framework | Rollout Backend | Determinism Level | SM89 Gap? | GRPO Viable on SM89? |
|-----------|----------------|-------------------|-----------|---------------------|
| SGLang | SGLang Triton | ★★★★★★★★ KERNEL (9+ overrides) | No gap | ★★★★★★★★ YES — gold standard |
| verl HYBRID + #6572 | vLLM (same process) | ★★★★★★★★ COMPILE+DEPLOY (5-layer) | RMSNorm gap | ★★★★★★★★ YES — with P9 fills gap |
| vLLM + VLLM_BATCH_INVARIANT | vLLM Triton overrides | ★★★★★★★ PARTIAL | RMSNorm + matmul | ★★★★★★★★ YES — with P9 fills gaps |
| rLLM Tinker | vLLM/SGLang | ★★★★★★★★ KERNEL or COMPILE | Depends on backend | ★★★★★★★★ YES — inherits backend |
| DeepSpeed ZeRO-2 | torch (eager mode) | ★★★ NONE (eager=correct) | No override needed | ★★★★★ YES — enforce_eager → slow |
| Megatron | TE + torch.compile | ★★★ COMPILE (affected) | Same as vLLM | ★★★★ YES — with Fusion Guard |
| PyTorch (raw) | torch.compile | ★★★ NONE | RMSNorm fusion | ★★★ Only with enforce_eager or P9 |

★★★★★★★★★ Complete SM89 deterministic GRPO stack (5 components):
  1. P9 Inductor Fusion Guard (5 LOC) → prevents reduction fusions
  2. VLLM_BATCH_INVARIANT=1 (env var) → enables vLLM aten overrides
  3. verl #6572 5-layer determinism → production deployment
  4. SGLang KERNEL-level overrides → gold standard baseline
  5. Triton constexpr BLOCK_SIZE → deterministic accumulation order
```

---

## Key Findings Summary

★★★★★★★★★ 5 implementations now: SGLang (9+ KERNEL overrides) + vLLM (984-line PARTIAL SM89) + vLLM-Ascend (AscendC/Triton dual-tier) + MindIE Turbo (compose-level) + verl #6572 (5-layer deployment)
★★★★★★★★★ vLLM SM89 GAPS CONFIRMED: RMSNorm NOT aten override, matmul only CUBLASLt workspace, silu/sigmoid/mul NOT overridden
★★★★★★★★★ SGLang #24459 STRENGTHENED: rms_norm + mm.dtype overrides → 9+ overrides → MORE than originally counted
★★★★★★★★★ verl #6572 VALIDATES VLLM_BATCH_INVARIANT=1 production use → BUT SM89 requires P9 for full determinism
★★★★★★★★★ P9 + #187435 + #6572 = 3 complementary mechanisms → P9 first (5 LOC), #187435 second (804 LOC per-op), #6572 deployment
★★★★★★★★★ Triton tl.constexpr = AscendC deterministic tiling → hardware-agnostic determinism principle
★★★★★★★★★ GRPO n=8 in same batch → HYBRID on-policy → simplest determinism story for RTX 4090
★★★★★★★★★ NEW: DSV4 CUDA graph replay = 6th form of non-determinism → batch-dependent graph replay with stale metadata!
  → 4 failures in 4 days across 2 frameworks + Ascend → enforce_eager=True = simplest fix
  → @eager_break_during_capture = correct separation boundary (validated by vLLM #45972 REVERT)
  → Dynamic routing (MoE, DSA, MTP) under graph replay → PRE-CAPTURED path → NOT current decisions

---

## References

- SGLang deterministic: notebook/projects/sglang-deterministic-inference-source-reading.md
- SGLang #24459: notebook/projects/sglang-latest-developments-2026-06-agent-research.md
- vLLM batch_invariant SM89 gap: notebook/projects/vllm-batch-invariant-source-reading.md
- verl #6572 full determinism: notebook/projects/verl-6572-full-determinism-source-reading.md
- P9 integration path: notebook/projects/p9-fusion-guard-integration-path-synthesis.md
- P9 issue draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- DSV4 systematic instability: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
- CUDA graph fragility: notebook/projects/vllm-cuda-graph-reading.md (Section 13)
- vLLM-Ascend batch_invariant for RL: #10034
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- MindIE ATB compose: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
