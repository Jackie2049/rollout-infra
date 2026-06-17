# P9 Inductor Fusion Guard — Integration Path with PyTorch #187435 + verl #6572

> 2026-06-18 | Synthesis note | How P9, PyTorch #187435 (no_fuse_region), and verl #6572 (full determinism) work together
> ★★★★★★★★ P9 = global SM<90 policy, #187435 = per-op mechanism, #6572 = production validation
> ★★★★★★★★ Three approaches are COMPLEMENTARY — each fills a different layer

---

## 1. Three Layers of SM89 Determinism

```
★★★★★★★★★ Three complementary mechanisms for SM89 batch invariance:

Layer A: P9 Inductor Fusion Guard (our contribution)
  → Location: torch/_inductor/choices.py (5 lines)
  → Mechanism: props.major < 9 → WhyNoFuse("batch_invariance")
  → Scope: GLOBAL → blocks ALL reduction fusions on SM<90 GPUs
  → Precision: coarse → blanket prohibition → simple → maintainable
  → Pros: simplest implementation → universal → covers unknown future fusions
  → Cons: blocks even "good" fusions → needs P6 Triton swiglu for performance

Layer B: PyTorch #187435 no_fuse_region (upstream mechanism)
  → Location: torch/_inductor/control_deps.py + scheduler.py (804 additions)
  → Mechanism: mark_no_fuse_region(graph, [ops]) → annotations["no_fuse_region"]
  → Scope: PER-OP → tag specific ops to prevent fusion → fine-grained
  → Precision: fine → only tagged ops blocked → ops in same region can still fuse
  → Pros: precise → doesn't block good fusions → less performance impact
  → Cons: requires identifying and tagging each problematic op → more maintenance

Layer C: verl #6572 VLLM_BATCH_INVARIANT (production deployment)
  → Location: verl/trainer/main_ppo.py + config (993 additions)
  → Mechanism: VLLM_BATCH_INVARIANT=1 + SamplingParams.seed + priority routing
  → Scope: DEPLOYMENT → enables vLLM's existing batch_invariant.py at runtime
  → Precision: runtime → enables/disables entire determinism suite
  → Pros: production-tested → bitwise-aligned reward curves → end-to-end validation
  → Cons: only works on SM90+ without P9 → SM89 still has RMSNorm gap!

★★★★★★★★★ Integration path:
  P9 (global policy) → blocks bad fusions on SM<90 → blanket approach
  #187435 (per-op mechanism) → could tag specific batch-invariant ops → fine-grained
  #6572 (deployment) → enables vLLM overrides at runtime → production mechanism

  → P9 + #6572 = complete SM89 determinism (global blocking + production deployment)
  → #187435 + vLLang overrides = future fine-grained approach (per-op blocking + overrides)
  → All three are COMPLEMENTARY → not competing!
```

---

## 2. P9 vs #187435 — Architectural Comparison

```
★★★★★★★★★ Two approaches to preventing Inductor fusions on SM89:

| Aspect | P9 Fusion Guard | #187435 no_fuse_region |
|--------|----------------|----------------------|
| Location | choices.py (5 lines) | control_deps + scheduler (804 lines) |
| Mechanism | WhyNoFuse("batch_invariance") | annotations["no_fuse_region"] |
| Scope | GLOBAL for SM<90 | PER-OP tagged region |
| Precision | ALL reductions blocked | Only tagged ops blocked |
| Implementation | 5 LOC in choices.py | 804 LOC across scheduler + lowering |
| Review burden | Minimal | Significant |
| Performance | Blocks good fusions too → need P6 | Preserves good fusions |
| Maintenance | Zero → static check | Per-model tagging needed |
| vLLM compatibility | Works with existing batch_invariant.py | Needs vLLM to add no_fuse_region tags |
| SGLang compatibility | Unnecessary (already KERNEL-level) | Could tag SGLang ops too |

★★★★★★★★★ Our strategy: P9 FIRST → #187435 SECOND:

Phase 1: P9 Fusion Guard → PyTorch issue (pre-step) → then PR
  → 5-line implementation → minimal review burden → fast path
  → Global blocking → universal → covers all SM<90 GPUs
  → Combined with P6 Triton swiglu → provides GOOD fused alternative
  → Works with vLLM batch_invariant.py + verl #6572 immediately!

Phase 2: After P9 lands → refine with #187435 mechanism
  → vLLM could use no_fuse_region to tag batch-invariant ops
  → Only block specific problematic fusions → not all reductions
  → Better performance → good fusions preserved → no need for separate kernels
  → But: requires vLLM upstream to adopt no_fuse_region tagging → slower

★★★★★★★★★ Why P9 first is better:
  1. Minimal implementation → 5 LOC → easy to review → fast merge
  2. Universal coverage → ALL SM<90 → not model-specific
  3. Works immediately with vLLM batch_invariant.py → no vLLM changes needed
  4. Works immediately with verl #6572 → makes #6572 work on SM89!
  5. P6 Triton swiglu provides GOOD fused path → performance maintained
  6. #187435 is still OPEN → may change before merge → P9 is stable
```

---

## 3. P9 + #6572 = Complete SM89 Deterministic GRPO

```
★★★★★★★★★ Complete SM89 deterministic GRPO stack:

  1. verl #6572: VLLM_BATCH_INVARIANT=1 → production mechanism
     → Enables vLLM batch_invariant.py overrides at runtime
     → SamplingParams.seed → per-request deterministic sampling
     → Priority routing → deterministic request ordering
     → RM max_num_seqs=1 → serialized reward model

  2. P9 Fusion Guard: WhyNoFuse on SM<90 → makes #6572 work on SM89
     → Blocks ALL reduction fusions → forces separate kernel dispatch
     → vLLM overrides ACTIVATE when dispatched separately → deterministic!
     → RMSNorm gap filled → mean override works → batch-invariant!

  3. vLLM batch_invariant.py: Triton overrides → deterministic when separate
     → softmax, mean.dim, bmm, log_softmax → tl.constexpr → deterministic
     → CUBLASLt workspace config → deterministic matmul when separate
     → NCCL env vars → deterministic communication

  4. P6 Triton dequant_swiglu_quant → GOOD fused path for MoE
     → tl.constexpr → deterministic → same as SGLang approach
     → MoE 6→1 reduction → MindIE-port → GPU-generic
     → Provides performance alternative to unfused path

★★★★★★★★★ Training flow on RTX 4090:
  Step 1: verl HYBRID mode → actor+rollout same process → same GPU
  Step 2: bypass_mode → old_log_probs = rollout_log_probs → no ref model → save 14GB
  Step 3: CPPO mask → position-weighted trust region → better than GRPO
  Step 4: GRPO advantage → group-relative → no critic needed
  Step 5: VLLM_BATCH_INVARIANT=1 → deterministic inference
  Step 6: P9 Fusion Guard → blocks bad fusions → overrides work
  Step 7: P6 Triton swiglu → fast fused path → performance maintained

★★★★★★★★★ Result: deterministic, fast, memory-efficient GRPO on RTX 4090!
```

---

## 4. Contribution Strategy Timeline

```
★★★★★★★★★ Contribution strategy timeline:

Phase 1 (NOW): P9 Issue Draft → PyTorch issue (pre-step before PR)
  → Issue body already drafted → notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
  → Submit as PyTorch issue → get community feedback → validate approach
  → Expected: positive feedback → no competing approach → proceed to PR

Phase 2 (after issue): P9 PR → PyTorch upstream PR
  → 5-line choices.py → WhyNoFuse("batch_invariance") on SM<90
  → Include: verl #6572 as validation → VLLM_BATCH_INVARIANT production use
  → Include: #187435 as complementary → show P9 is simpler first approach
  → Expected: moderate review → 95% confidence based on #187275 progress

Phase 3 (after P9): P6 Triton dequant_swiglu → vLLM/SGLang upstream
  → After P9 lands → blocks bad fusions → need GOOD fused alternative
  → Triton tl.constexpr → deterministic → MoE performance gain
  → Expected: positive review → MindIE port → GPU-generic → unique value

Phase 4 (future): #187435 integration → vLLang adopts no_fuse_region
  → After both P9 and #187435 land → vLLM could use no_fuse_region
  → Fine-grained tagging → only block specific ops → better performance
  → Not urgent → P9 provides immediate solution → #187435 provides future refinement

★★★★★★★★★ GPU experiments needed (when servers online):
  → P9 validation: run sm89_batch_invariance_diagnostic.py + repro.py on RTX 4090
  → BudgetRefiner profile_table.csv collection on RTX 4090 → unique data
  → P6 Triton swiglu benchmark on RTX 4090 → performance validation
  → verl #6572 determinism test → bitwise-aligned reward curves → P9 validation
```

---

## Key Findings Summary

★★★★★★★★★ P9 (global) + #187435 (per-op) + #6572 (deployment) = three complementary layers
★★★★★★★★★ P9 first strategy: simplest → fastest → universal → works with existing vLLM/verl
★★★★★★★★★ #187435 provides future fine-grained mechanism → refine P9 after it lands
★★★★★★★★★ #6572 validates VLLM_BATCH_INVARIANT as production mechanism → P9 fills SM89 gap
★★★★★★★★★ P9 + #6572 = complete SM89 deterministic GRPO → immediate contribution value
★★★★★★★★★ P9 + P6 Triton swiglu = deterministic + fast → complete performance solution

---

## References

- P9 Fusion Guard issue draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- vLLM batch_invariant source: notebook/projects/vllm-batch-invariant-source-reading.md
- verl #6572 full determinism: notebook/projects/verl-6572-full-determinism-source-reading.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- Cross-framework comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- PyTorch #187435: https://github.com/pytorch/pytorch/pull/187435
- verl #6572: https://github.com/verl-project/verl/pull/6572
