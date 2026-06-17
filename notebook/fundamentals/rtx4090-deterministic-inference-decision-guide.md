# RTX 4090 Deterministic Inference Decision Guide — SM89 Batch Invariance

> 2026-06-18 | RTX 4090 consulting reference | SM89 batch invariance decision tree
> ★★★★★★★★ 3 solutions: KERNEL-level (SGLang) > COMPILE-level (Fusion Guard) > NONE (enforce_eager)
> ★★★★★★★★ Our contribution: Fusion Guard P9 + Triton swiglu P6 = complete SM89 solution

---

## Decision Tree

```
START: Need deterministic batch-invariant inference on RTX 4090 (SM89)?
│
│ ★★★★★★★★ ROOT CAUSE: Triton CachingAutotuner XBLOCK varies per batch → tl.sum() non-associative FP
│
├── Q1: Serving framework choice?
│   │
│   ├── SGLang → ★★★★★★★★★★★★★★★★★★★ GOLD STANDARD (KERNEL-level)
│   │   ├── --enable-deterministic-inference → 7 aten overrides with tl.constexpr
│   │   │   → rms_norm, silu, sigmoid, mul, mm.dtype, bmm + 1 more
│   │   │   → BLOCK_SIZE = tl.constexpr → no autotuning → deterministic by design
│   │   │
│   │   ├── murmur_hash32 Gumbel-max sampling → GPU-side → no RNG sync
│   │   │   → Position+seed+vocab hash → deterministic categorical sampling
│   │   │
│   │   ├── Triton backend recommended for SM89 → split_tile_size=256
│   │   │   → DeepGEMM disabled → all matmuls _matmul_persistent_triton constexpr
│   │   │
│   │   └── MoE LoRA + deterministic = UNIQUE capability
│   │       → Triton SGMV + extra_key namespace → LoRA + MoE + deterministic
│   │       → ONLY framework with this combo → SGLang MoE GRPO = viable!
│   │
│   │   ★★★★★★★★ NO additional fixes needed → built-in deterministic!
│   │   ★★★★★★★★ Recommendation: SGLang for RTX 4090 GRPO inference
│   │
│   ├── vLLM → ★★★★★★★★ COMPILE-level (needs our Fusion Guard)
│   │   ├── Default: torch.compile → Inductor → batch-dependent on SM89!
│   │   │   → Inductor fuses RMSNorm → tl.sum() varies → non-deterministic
│   │   │
│   │   ├── Fix Option A: Inductor Fusion Guard (our P9 PR)
│   │   │   → 5-line guard in choices.py can_fuse_vertical → blocks SM<90 reduction fusion
│   │   │   → Reductions stay as separate kernels → torch.mean override works
│   │   │   → ★★★★★★★★ CORRECT + FAST (separate kernels still fast)
│   │   │   → Pre-step: file PyTorch issue first (draft ready)
│   │   │
│   │   ├── Fix Option B: Triton dequant_swiglu_quant kernel (our P6)
│   │   │   → Triton fused MoE path → tl.constexpr → deterministic
│   │   │   → Complements Fusion Guard → provides GOOD fused alternative
│   │   │   → ★★★★★★★★ CORRECT + FASTER than unfused reduction path
│   │   │
│   │   ├── Fix Option C: enforce_eager=True
│   │   │   → Disables torch.compile → no Inductor → no fusion → correct
│   │   │   → ★★★★★ CORRECT but SLOW → no CUDA graph → ~2x slower
│   │   │   → Fallback when Fusion Guard not yet merged
│   │   │
│   │   ├── Fix Option D: VLLM_USE_V2_MODEL_RUNNER=0
│   │   │   → MRv1 → more conservative → may avoid some fusion issues
│   │   │   → ★★★★ Uncertain reliability → not guaranteed
│   │   │
│   │   └── ★★★★★★★★ Recommendation: Fusion Guard + Triton swiglu (when merged)
│   │       → Until merged: enforce_eager=True as fallback
│   │
│   ├── DeepSpeed ZeRO-2 → ★★★ NONE (use enforce_eager)
│   │   ├── No deterministic mechanism → raw PyTorch → affected by same bug
│   │   ├── Fix: DO NOT use torch.compile with DeepSpeed → #8061 NaN bug!
│   │   │   → overlap_comm=False → no compile → no batch invariance issue
│   │   │   → ★★★★★★★★ MUST overlap_comm=False on single GPU anyway (zero penalty)
│   │   │
│   │   ├── Inference: use enforce_eager → slow but correct
│   │   ├── MoE: freeze router → same input = same routing → deterministic
│   │   │   → ★★★★★★★★ Freeze router = 0 LOC immediate determinism for MoE
│   │   │
│   │   └── ★★★★★★★★ Recommendation: DeepSpeed for TRAINING only, not inference
│   │       → Use vLLM/SGLang for inference + DeepSpeed for training (HYBRID)
│   │
│   ├── Megatron-LM → ★★★ COMPILE-level (affected, RouterReplay exists for MoE only)
│   │   ├── TE (Transformer Engine) → custom kernels → SM90 deterministic
│   │   │   → SM89: not guaranteed → same autotuning issues possible
│   │   │
│   │   ├── RouterReplay → MoE routing determinism → NOT general inference
│   │   │   → 3 modes: RECORD/REPLAY_FORWARD/REPLAY_BACKWARD
│   │   │   → Only addresses routing → doesn't fix RMSNorm batch invariance
│   │   │
│   │   ├── ★★★★★★★★ NOT recommended for RTX 4090 inference → ranking #3
│   │   │
│   │   └── Fix: Fusion Guard (same as vLLM) + RouterReplay (MoE)
│   │
│   ├── verl HYBRID → ★★★★★★★★ COMPILE-level (inherits from vLLM)
│   │   ├── verl rollout = vLLM backend → same determinism issues
│   │   ├── BUT: HYBRID on-policy → pi_rollout = pi_theta → simplest story
│   │   │   → GRPO n=8 same batch → SAME compilation → batch-invariant WITHIN group!
│   │   │   → ★★★★★★★★ On-policy + same batch = mitigates batch invariance concern
│   │   │
│   │   ├── Fix: inherits vLLM fix (Fusion Guard or enforce_eager)
│   │   └── ★★★★★★★★ Recommendation: verl HYBRID + vLLM Fusion Guard (when merged)
│   │       → Until merged: verl HYBRID + enforce_eager=True
│   │
│   ├── rLLM Tinker → ★★★★★★★★ Delegates to vLLM/SGLang
│   │   ├── Client-server pattern → inherits backend determinism
│   │   ├── Single-step training → no multi-iteration concern
│   │   └── ★★★★★★★★ Use same backend fix (Fusion Guard or SGLang)
│   │
│   └── MindIE → ★★★★★★★★ Ascend-specific (not RTX 4090)
│       → Ascend deterministic by hardware design → no Triton autotuning
│       → npu_dequant_swiglu_quant → deterministic by AscendC design
│       → ATB compose-level = atomic → scheduling preserved
│       → ★★★★★★★★ Lessons portable: Triton constexpr = AscendC deterministic tiling
│       → ★★★★★★★★ Our Triton swiglu kernel = MindIE port to CUDA!
│
├── Q2: Which solution for YOUR setup?
│   │
│   ├── verl HYBRID + vLLM (most common RTX 4090 GRPO)
│   │   → Until Fusion Guard merged: enforce_eager=True
│   │   → After Fusion Guard merged: torch.compile + Fusion Guard = fast + correct
│   │   → Future: Triton swiglu kernel + Fusion Guard = fastest + correct
│   │
│   ├── verl HYBRID + SGLang (best RTX 4090 determinism)
│   │   → SGLang deterministic inference → KERNEL-level → gold standard
│   │   → RadixAttention → prefix KV reuse → GRPO n=8 benefit
│   │   → verl SGLang integration (#6117) → viable path
│   │   → ★★★★★★★★ BEST for determinism → recommended if SGLang integration stable
│   │
│   ├── rLLM Tinker + SGLang (simplest deterministic path)
│   │   → Tinker delegates inference to SGLang → inherits determinism
│   │   → Single-step training → no IS correction → simplest story
│   │   → ★★★★★★★★ SIMPLEST deterministic path for RTX 4090
│   │
│   └── DeepSpeed ZeRO-2 + vLLM (training only, not RL)
│       → DeepSpeed for supervised fine-tuning → no RL → no batch invariance concern
│       → vLLM for inference → Fusion Guard needed if using torch.compile
│       → ★★★★★★★★ For RL: ALWAYS use verl/rLLM, NOT DeepSpeed alone
│
└── Q3: GRPO-specific determinism requirements?
    │
    ├── GRPO n=8 responses per prompt
    │   → ALL 8 responses must be batch-invariant → same prompt = same logits
    │   → ★★★★★★★★ GRPO generates all 8 in SAME batch → SAME compilation
    │   → → If deterministic for THAT batch size → GRPO works within group
    │   → → BUT: across different group sizes → different compilation → concerns
    │
    ├── HYBRID on-policy advantage
    │   → pi_rollout = pi_theta → IS ratio = 1.0 → no correction needed
    │   → ★★★★★★★★ On-policy = simplest determinism story → no cross-model comparison
    │
    ├── bypass_mode simplification
    │   → Eliminates ref model → old_log_probs = rollout_log_probs
    │   → Only 2 policies (pi_theta + pi_rollout) → simpler → more robust
    │   → ★★★★★★★★ bypass_mode = fewer determinism-sensitive comparisons
    │
    └── CPPO (tighter bound) determinism
        → Position-weighted cumulative prefix divergence → stricter trust region
        → bypass_mode MANDATORY → divergence against μ (rollout) → NOT pi_old
        → ★★★★★★★★ CPPO + bypass + SGLang/vLLM Fusion Guard = optimal RTX 4090
```

---

## Quick Reference: Recommended Configurations

### RTX 4090 GRPO #1 — verl + SGLang (Best Determinism)

```
★★★★★★★★★ BEST determinism path:
  Serving: SGLang --enable-deterministic-inference
  Training: verl HYBRID + bypass_mode + CPPO
  Determinism: KERNEL-level → tl.constexpr → gold standard
  MoE: SGLang MoE LoRA + deterministic → unique capability
  Need: verl SGLang integration (#6117) → check stability
```

### RTX 4090 GRPO #2 — verl + vLLM + Fusion Guard (When Merged)

```
★★★★★★★★★ Standard path (when Fusion Guard merged):
  Serving: vLLM + torch.compile + Fusion Guard P9
  Training: verl HYBRID + bypass_mode + CPPO
  Determinism: COMPILE-level → separate reduction kernels → correct
  Need: PyTorch issue filed → PR submitted → merged → wait
```

### RTX 4090 GRPO #3 — verl + vLLM enforce_eager (Current Fallback)

```
★★★★★★★★★ Current fallback (before Fusion Guard merged):
  Serving: vLLM enforce_eager=True
  Training: verl HYBRID + bypass_mode + GRPO/CPPO
  Determinism: NONE (no compile) → enforce_eager → correct but slow
  Trade-off: ~2x slower inference → acceptable for early experimentation
```

### RTX 4090 GRPO #4 — rLLM Tinker + SGLang (Simplest)

```
★★★★★★★★★ Simplest deterministic path:
  Serving: SGLang --enable-deterministic-inference (via Tinker SDK)
  Training: rLLM Tinker single-step + bypass_mode
  Determinism: KERNEL-level (inherited from SGLang)
  Need: Tinker checkpoint → not standard PEFT → export gap
```

---

## Key Findings

★★★★★★★★★ SGLang KERNEL-level = gold standard → tl.constexpr → no autotuning → deterministic by design
★★★★★★★★★ Fusion Guard P9 + Triton swiglu P6 = complete SM89 COMPILE+KERNEL solution
★★★★★★★★★ enforce_eager = ALWAYS correct but ALWAYS slow → fallback only
★★★★★★★★★ verl HYBRID on-policy + GRPO n=8 same batch = simplest determinism story
★★★★★★★★★ DeepSpeed for TRAINING only → vLLM/SGLang for inference → HYBRID pattern
★★★★★★★★★ Triton tl.constexpr = AscendC deterministic tiling → hardware-agnostic principle

---

## References

- Deterministic cross-framework: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- SGLang deterministic: notebook/projects/sglang-deterministic-inference-source-reading.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard issue: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- GRPO decision flowchart: notebook/fundamentals/rtx4090-grpo-training-decision-flowchart.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- MindIE ATB compose: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
