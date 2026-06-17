# Deterministic Inference Cross-Framework Comparison — SM89 Batch Invariance

> 2026-06-18 | Cross-framework analysis | How each framework achieves batch-invariant inference on SM<90 GPUs
> ★★★★★★★★ 3 architectural layers: KERNEL-level (SGLang) > COMPILE-level (vLLM) > NONE (DeepSpeed/Megatron/verl)
> ★★★★★★★★ ROOT CAUSE: Triton CachingAutotuner XBLOCK varies per batch size → non-associative FP addition
> ★★★★★★★★ Triton tl.constexpr BLOCK_SIZE = gold standard → our Fusion Guard + Triton swiglu kernel both use this

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

7 aten overrides with tl.constexpr BLOCK_SIZE:
  1. rms_norm: BLOCK_SIZE = tl.constexpr → no autotuning → deterministic
  2. silu: BLOCK_SIZE = tl.constexpr → same
  3. sigmoid: BLOCK_SIZE = tl.constexpr → same
  4. mul: BLOCK_SIZE = tl.constexpr → same
  5. mm (matmul): BLOCK_SIZE = tl.constexpr + dtype specialization → same
  6. bmm: BLOCK_SIZE = tl.constexpr → same
  7. (additional override for elementwise ops)

★★★★★★★★★ 3 NEW overrides added beyond original 4:
  → rms_norm (was relying on torch.compile → now explicit Triton override)
  → mm with dtype specialization (float16/bfloat16 separate constexpr paths)
  → bmm (batched matmul → constexpr block sizes)

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

### vLLM — COMPILE-Level (★★★★★★★★★ AFFECTED BY ROOT CAUSE)

```
★★★★★★★★★ vLLM relies on torch.compile → Inductor → subject to batch invariance bug:

Current status on SM89:
  → torch.compile(vllm_model) → Inductor fuses RMSNorm → batch-dependent
  → vLLM's torch.mean override → WORKS when reduction stays as separate kernel
  → But Inductor FUSES reduction+pointwise → override bypassed → fails!

★★★★★★★★★ vLLM's attempts to fix:
  → enforce_eager=True → disables torch.compile → slow but correct
  → MRv2 (Model Runner v2) → preserves determinism → safe for verl
  → vLLM #39096 → open bug → community tracking
  → vLLM #44879 → Inductor fix attempt → Tier 1 comment ready

★★★★★★★★★ Our proposed solution for vLLM:
  → Inductor Fusion Guard P9 → blocks bad fusions on SM<90 → separate kernels → correct
  → Triton dequant_swiglu_quant P6 → provides GOOD fused path → faster than unfused
  → Together: Fusion Guard blocks bad + Triton provides good = complete SM89 solution

★★★★★★★★★ Why COMPILE-level is weaker than KERNEL-level:
  → Depends on Inductor scheduler decisions → may change across versions
  → torch.compile may fuse differently for different model sizes → model-dependent
  → Cannot guarantee batch invariance across all compilation configurations
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

### verl — COMPILE-Level (★★★★★★★★★ SAME AS vLLM + HYBRID mitigates)

```
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
| KERNEL-level | Triton aten overrides + tl.constexpr | SGLang | ★★★★★★★★ STRONGEST — bypasses Inductor entirely | Yes! — constexpr = deterministic by design |
| COMPILE-level | Inductor Fusion Guard (our P9) | vLLM + PyTorch | ★★★★★★★★ STRONG — blocks bad fusions on SM<90 | Yes! — separate kernels = deterministic |
| NONE | No mechanism → raw torch.compile | DeepSpeed, Megatron, rLLM | ★★★ WEAK — subject to batch invariance bug | Only with enforce_eager=True |

★★★★★★★★★ Our contribution strategy covers ALL 3 layers:
  → P9 Fusion Guard → strengthens COMPILE-level → vLLM + PyTorch
  → P6 Triton swiglu kernel → provides GOOD fused KERNEL-level path → vLLM + SGLang
  → Together: block bad COMPILE-level fusions + provide good KERNEL-level alternatives

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

| Framework | Rollout Backend | Determinism Level | GRPO Viable on SM89? |
|-----------|----------------|-------------------|---------------------|
| SGLang | SGLang Triton | ★★★★★★★★ KERNEL | ★★★★★★★★ YES — gold standard |
| verl HYBRID | vLLM (same process) | ★★★★★★★★ COMPILE (with guard) | ★★★★★★★★ YES — with Fusion Guard or enforce_eager |
| rLLM Tinker | vLLM/SGLang | ★★★★★★★★ KERNEL or COMPILE | ★★★★★★★★ YES — inherits backend determinism |
| DeepSpeed ZeRO-2 | torch (eager mode) | ★★★ NONE (eager=correct) | ★★★★★ YES — enforce_eager → slow but correct |
| Megatron | TE + torch.compile | ★★★ COMPILE (affected) | ★★★★ YES — with Fusion Guard |

★★★★★★★★★ Key insight: enforce_eager = ALWAYS correct but ALWAYS slow
  → Fusion Guard + Triton swiglu = CORRECT and FAST → both needed!
```

---

## Key Findings Summary

★★★★★★★★★ 3 architectural layers: KERNEL > COMPILE > NONE for SM89 determinism
★★★★★★★★★ SGLang KERNEL-level = gold standard → tl.constexpr → bypasses Inductor entirely
★★★★★★★★★ Our P9 Fusion Guard strengthens COMPILE-level → blocks bad fusions → separate kernels → deterministic
★★★★★★★★★ Our P6 Triton swiglu provides GOOD KERNEL-level fused path → tl.constexpr → deterministic + fast
★★★★★★★★★ Triton tl.constexpr = AscendC deterministic tiling → hardware-agnostic determinism principle
★★★★★★★★★ GRPO n=8 in same batch → HYBRID on-policy → simplest determinism story for RTX 4090
★★★★★★★★★ Together: Fusion Guard + Triton swiglu = complete SM89 deterministic inference solution

---

## References

- SGLang deterministic: notebook/projects/sglang-deterministic-inference-source-reading.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- MindIE ATB compose: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
