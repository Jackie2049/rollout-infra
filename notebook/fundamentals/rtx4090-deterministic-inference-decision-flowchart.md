# RTX 4090 Deterministic Inference Decision Flowchart — SM89 Batch Invariance

> 2026-06-18 | Decision flowchart | Which deterministic inference approach for each GRPO scenario?
> ★★★★★★★★ 8-level decision tree covering all 7 frameworks + SM89 determinism
> ★★★★★★★★ Complements: rtx4090-grpo-training-decision-flowchart.md (training focus)

---

## Decision Flowchart — SM89 Deterministic Inference

```
START: Need deterministic inference on RTX 4090 (SM89) for GRPO training?
│
├─ YES → Which serving backend?
│  │
│  ├─ SGLang → ★★★★★★★★ USE SGLang Triton defaults
│  │  → 7 aten overrides + tl.constexpr → KERNEL-level → gold standard
│  │  → murmur_hash32 Gumbel-max → deterministic sampling → no RNG sync
│  │  → NO torch.compile needed → NO Inductor issue → bulletproof!
│  │  → ★★★★★★★★ SGLang = BEST choice for deterministic GRPO on RTX 4090
│  │
│  ├─ vLLM → Need torch.compile?
│  │  │
│  │  ├─ NO → enforce_eager=True
│  │  │  → Disables torch.compile → no Inductor fusion → correct but slow
│  │  │  → ★★★★★★★★ Simplest vLLM path → correct → performance penalty
│  │  │
│  │  ├─ YES → Is our Fusion Guard merged? (P9 PR)
│  │  │  │
│  │  │  ├─ YES → ★★★★★★★★ USE torch.compile + Fusion Guard
│  │  │  │  → Inductor blocks bad fusions on SM<90 → separate kernels → deterministic
│  │  │  │  → Triton swiglu kernel P6 → GOOD fused path → fast + deterministic
│  │  │  │  → ★★★★★★★★ Fusion Guard merged = torch.compile viable on SM89!
│  │  │  │
│  │  │  ├─ NO → Is Triton 3.7.1+ (v2.13) available?
│  │  │  │  │
│  │  │  │  ├─ YES → ★★★★ MAY be deterministic (check vLLM #45731)
│  │  │  │  │  → Triton 3.7.1 may fix XBLOCK variability on SM89
│  │  │  │  │  → MUST validate before trusting → test batch invariance
│  │  │  │  │  → ★★★★★★★★ If Triton 3.7.1 fixes → Fusion Guard unnecessary!
│  │  │  │  │
│  │  │  │  ├─ NO → ★★★★★★★★ MUST use enforce_eager=True
│  │  │  │  │  → Without Fusion Guard or Triton fix → batch-dependent!
│  │  │  │  │  → Alternative: custom Triton overrides (SGLang approach)
│  │  │  │  │
│  │  │  │  └─ UNKNOWN → ★★★★★★★★ Test with sm89_batch_invariance_diagnostic.py
│  │  │  │
│  │  │  └─ ← end of vLLM compile branch
│  │  │
│  │  └─ ← end of vLLM branch
│  │
│  ├─ DeepSpeed → ★★★★★★★★ DO NOT use torch.compile for inference
│  │  → DeepSpeed has NO deterministic inference mechanism
│  │  → overlap_comm + torch.compile = NaN (#8061)
│  │  → Use enforce_eager → correct but slow
│  │  → ZeRO-2 + CPU_Adam for training → training determinism is fine
│  │  → ★★★★★★★★ DeepSpeed inference = raw PyTorch → enforce_eager only option
│  │
│  ├─ Megatron → ★★★★ NOT recommended for RTX 4090 inference (ranking #3)
│  │  → TE (Transformer Engine) → SM89 not guaranteed deterministic
│  │  → RouterReplay exists but only for MoE CUDA graph → not general
│  │  → Megatron-Lite may improve → track #4885
│  │
│  ├─ MindIE → Ascend-only → NOT applicable to RTX 4090
│  │
│  ├─ rLLM → ★★★★★★★★ Inherits backend determinism
│  │  → Tinker uses vLLM/SGLang → same determinism as chosen backend
│  │  → ★★★★★★★★ rLLM determinism = backend determinism → choose backend wisely!
│  │
│  └─ OTHER → ★★★★★★★★ Check torch.compile behavior on SM89
│     → Use sm89_batch_invariance_diagnostic.py
│     → If batch-dependent → enforce_eager OR custom Triton overrides
│
├─ NO → Is your GPU SM90+?
│  │
│  ├─ YES → ★★★★★★★★ torch.compile is safe on SM90+
│  │  → TMA + WGMMA → deterministic by hardware design
│  │  → No Fusion Guard needed → Inductor fusions OK
│  │  → ★★★★★★★★ SM90+ = torch.compile fully viable → no restrictions!
│  │
│  ├─ NO (SM89/SM86/SM80/SM75) → ★★★★★★★★ You NEED deterministic inference
│  │  → Follow YES branch above → all SM<90 GPUs affected
│  │  → Triton CachingAutotuner XBLOCK varies on ALL SM<90
│  │  → ★★★★★★★★ Our Fusion Guard P9 protects ALL SM<90 GPUs!
│  │
│  └─ UNKNOWN → ★★★★★★★★ Check with torch.cuda.get_device_capability()
│     → If major < 9 → SM<90 → needs determinism measures
│     → If major >= 9 → SM90+ → torch.compile safe
│
└─ END
```

---

## Quick Decision Matrix — RTX 4090 GRPO Rollout Backend

```
★★★★★★★★★ Recommended rollout backend for each training framework:

| Training Framework | Best Rollout Backend | Determinism Level | Recommended Config |
|-------------------|--------------------|-------------------|-------------------|
| rLLM Tinker #1 | SGLang Triton | ★★★★★★★★ KERNEL | SGLang defaults → deterministic automatically |
| verl HYBRID #2 | vLLM + enforce_eager | ★★★★★★★★ EAGER | enforce_eager=True → slow but correct |
| verl HYBRID #2 (future) | vLLM + Fusion Guard | ★★★★★★★★ COMPILE | torch.compile + P9 guard → fast + correct |
| DeepSpeed ZeRO-2 #2.5 | vLLM/SGLang (external) | ★★★★★★★★ KERNEL | SGLang Triton → deterministic |
| Megatron #3 | SGLang Triton | ★★★★★★★★ KERNEL | SGLang Triton → deterministic |

★★★★★★★★★ UNIVERSAL recommendation: SGLang Triton for RTX 4090 GRPO rollout!
  → KERNEL-level determinism → tl.constexpr → no autotuning → gold standard
  → MoE LoRA + deterministic = UNIQUE capability → no other framework has this
  → murmur_hash32 sampling → deterministic → no RNG sync → GRPO-friendly
```

---

## Tool Reference

```
★★★★★★★★★ Diagnostic and validation tools:

sm89_batch_invariance_diagnostic.py → check batch invariance on any GPU
sm89_batch_invariance_repro.py → reproduce RMSNorm batch dependency
triton_dequant_swiglu_quant_prototype.py → Triton fused kernel (info/benchmark/correctness)

★★★★★★★★★ When GPU available, run validation chain:
  1. sm89_batch_invariance_diagnostic.py → confirm SM89 batch dependency
  2. triton_dequant_swiglu_quant_prototype.py --mode correctness → validate Triton kernel
  3. triton_dequant_swiglu_quant_prototype.py --mode benchmark → measure speedup
```

---

## References

- Deterministic inference comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
- SGLang deterministic: notebook/projects/sglang-deterministic-inference-source-reading.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Fusion Guard issue: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Triton swiglu: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
