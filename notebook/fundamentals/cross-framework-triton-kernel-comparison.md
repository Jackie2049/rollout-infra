# Cross-Framework Triton Kernel Comparison — SM89 Batch Invariance Survey

> 2026-06-18 | Cross-framework Triton kernel survey | Which kernels use tl.constexpr vs autotuning?
> ★★★★★★★★ DeepSpeed 10/11 files ALL constexpr (LOW risk) → SGLang 7 overrides ALL constexpr → vLLM/SGLang/Megatron SAME batch-invariant system

---

## Survey Method

```
★★★★★★★★★ Survey covers Triton kernel files in each framework:
  → DeepSpeed: 11 Triton files in deepspeed/kernels/
  → vLLM: custom Triton kernels for attention/quantization/MoE
  → SGLang: 7 aten overrides + SGMV LoRA + MoE routing
  → Megatron: TE monkey-patching + Triton kernels for SM<90
  → verl: minimal Triton usage → 1-config autotune

★★★★★★★★★ Key metric: tl.constexpr vs autotuned BLOCK_SIZE
  → tl.constexpr → JIT-compiled → fixed → deterministic → batch-invariant
  → autotuned → CachingAutotuner → varies per input shape → batch-dependent on SM<90
  → ALL constexpr = SAFE → deterministic by design
  → ANY autotuned = POTENTIAL RISK → batch invariance concern on SM<90
```

---

## Framework Results

### DeepSpeed — ALL constexpr (★★★★★★★★★ LOW RISK)

```
★★★★★★★★★ DeepSpeed Triton kernel survey — 10/11 files ALL constexpr:

  1. ds_layernorm_kernel.py → BLOCK_M/BLOCK_N = tl.constexpr → deterministic
  2. ds_rmsnorm_kernel.py → BLOCK_M/BLOCK_N = tl.constexpr → deterministic
  3. ds_quantize_kernel.py → BLOCK_SIZE = tl.constexpr → deterministic
  4. ds_softplus_kernel.py → BLOCK_SIZE = tl.constexpr → deterministic
  5. ds_swiglu_kernel.py → BLOCK_M/BLOCK_N = tl.constexpr → deterministic
  6. ds_topk_kernel.py → BLOCK_M/BLOCK_N = tl.constexpr → deterministic
  7. ds_stochastic_rounding.py → BLOCK_SIZE = tl.constexpr → deterministic
  8. ds_bias_add_kernel.py → BLOCK_SIZE = tl.constexpr → deterministic
  9. ds_copy_kernel.py → BLOCK_SIZE = tl.constexpr → deterministic
  10. ds_reduce_kernel.py → BLOCK_M/BLOCK_N = tl.constexpr → deterministic

  1 AUTOTUNED file:
  → ds_scan_kernel.py → autotuned BLOCK_M for scan/reduce → POTENTIAL risk
  → BUT: scan not used in inference → training-only → less concern

★★★★★★★★★ DeepSpeed conclusion:
  → 10/11 ALL constexpr → LOW batch invariance risk
  → 1 autotuned (scan) → training-only → not in inference path
  → DeepSpeed kernels = deterministic by design → safe on SM89
  → Issue is NOT DeepSpeed kernels → it's torch.compile (Inductor) applied on top!
```

### SGLang — ALL constexpr (★★★★★★★★★ GOLD STANDARD)

```
★★★★★★★★★ SGLang Triton override survey — 7 aten overrides ALL constexpr:

  1. rms_norm_override → BLOCK_SIZE = tl.constexpr → deterministic
  2. silu_override → BLOCK_SIZE = tl.constexpr → deterministic
  3. sigmoid_override → BLOCK_SIZE = tl.constexpr → deterministic
  4. mul_override → BLOCK_SIZE = tl.constexpr → deterministic
  5. mm_override → BLOCK_M/BLOCK_K/BLOCK_N = tl.constexpr → deterministic
  6. bmm_override → BLOCK_M/BLOCK_K/BLOCK_N = tl.constexpr → deterministic
  7. (elementwise override) → BLOCK_SIZE = tl.constexpr → deterministic

  Additional SGLang Triton kernels:
  → murmur_hash32 → Triton constexpr → deterministic sampling
  → SGMV LoRA → Triton constexpr → deterministic MoE LoRA
  → MoE routing → Triton constexpr → deterministic expert selection

★★★★★★★★★ SGLang conclusion:
  → ALL overrides constexpr → KERNEL-level → gold standard
  → Bypasses Inductor entirely → immune to scheduler changes
  → MoE LoRA + deterministic = UNIQUE capability → no other framework has this
```

### vLLM — Mixed (★★★★★★★★★ NEEDS Fusion Guard)

```
★★★★★★★★★ vLLM Triton kernel survey — mixed constexpr + autotuned:

  Explicit Triton overrides (batch-invariant):
  → rms_norm override → tl.constexpr → deterministic (like SGLang)
  → attention kernel → BLOCK_SIZE varies → but vLLM controls per-batch
  → MoE top-k → constexpr → deterministic routing

  Inductor-generated Triton (batch-dependent):
  → torch.compile → Inductor → generates Triton kernels
  → CachingAutotuner → XBLOCK varies per batch → NON-deterministic
  → RMSNorm fusion → reduction + pointwise → batch-dependent on SM89

★★★★★★★★★ vLLM conclusion:
  → Custom Triton kernels: mostly constexpr → deterministic
  → Inductor-generated: autotuned → batch-dependent on SM89
  → Need Fusion Guard to block bad Inductor fusions → separate kernels → deterministic
  → vLLM #39096 tracks this issue → community awareness exists
```

### Megatron — TE Monkey-Patching (★★★★★★★★★ UNIQUE)

```
★★★★★★★★★ Megatron Triton kernel survey — UNIQUE TE monkey-patching:

  Transformer Engine (TE):
  → TE provides custom CUDA kernels → replaces standard ops
  → SM90+: TMA/WGMMA → deterministic by hardware → safe
  → SM89: TE not guaranteed deterministic → same autotuning risk

  Megatron-specific Triton:
  → MoE top-k routing → constexpr → deterministic (like DeepSpeed)
  → RouterReplay → MoE CUDA graph stability → deterministic routing
  → BUT: RouterReplay only for MoE → not general inference

★★★★★★★★★ Megatron conclusion:
  → TE monkey-patching = unique approach → replaces standard ops entirely
  → RouterReplay = MoE-only → NOT general inference determinism
  → SM89: TE may not guarantee → same Fusion Guard needed
  → NOT recommended for RTX 4090 inference → ranking #3
```

### verl — Minimal Triton (★★★★★★★★★ 1-Config Autotune)

```
★★★★★★★★★ verl Triton kernel survey — minimal Triton usage:

  verl does NOT write custom Triton kernels → delegates to vLLM
  → Rollout: vLLM backend → inherits vLLM Triton decisions
  → Training: pure PyTorch → no Triton → no autotuning risk

  verl-specific config:
  → 1-config autotune → single Triton config → deterministic by construction
  → No CachingAutotuner → no XBLOCK variation → no batch invariance issue
  → BUT: this is for verl's OWN code → vLLM backend still has issues

★★★★★★★★★ verl conclusion:
  → verl's own Triton: deterministic (1-config)
  → vLLM backend: inherits vLLM's batch invariance issues
  → HYBRID on-policy mitigates → same batch → same compilation → batch-invariant within group
```

---

## Cross-Framework Comparison Matrix

```
★★★★★★★★★ Triton kernel determinism comparison:

| Framework | Custom Triton | Inductor-Gen | constexpr% | Batch Risk | Solution |
|-----------|---------------|-------------|------------|------------|----------|
| DeepSpeed | 10/11 constexpr | None (no compile) | 91% | LOW | enforce_eager (no compile needed) |
| SGLang | 7/7 ALL constexpr | Bypassed entirely | 100% | NONE | Built-in deterministic |
| vLLM | Mixed | Autotuned | ~70% | HIGH | Fusion Guard P9 + Triton swiglu P6 |
| Megatron | TE+constexpr | Possible | ~80% | MEDIUM | Fusion Guard + RouterReplay |
| verl | 1-config | Inherits vLLM | 100% (own) | LOW (own) | HYBRID mitigates vLLM risk |

★★★★★★★★★ KEY INSIGHT:
  → DeepSpeed kernels = ALREADY deterministic (constexpr)
  → The issue is torch.compile + Inductor applied ON TOP → not the kernels themselves
  → SGLang solves this by REPLACING aten ops BEFORE Inductor → bypasses entirely
  → Our Fusion Guard blocks Inductor from fusing → achieves same outcome at COMPILE level
```

---

## SGLang UNIQUE: Deterministic MoE Routing

```
★★★★★★★★★ SGLang murmur_hash32 — UNIQUE deterministic MoE routing:

  murmur_hash32 Triton kernel:
  → Position + seed + vocab size → hash → Gumbel-max → categorical sampling
  → Entirely GPU-side → no RNG state synchronization
  → float64 precision → gold standard deterministic sampling
  → Works for ANY model → not just MoE

  SGMV LoRA kernel:
  → Triton SGMV (Segmented Matrix-Vector) → constexpr → deterministic
  → extra_key namespace → multiple LoRA adapters per expert
  → MoE LoRA + deterministic = UNIQUE capability → no other framework has this

★★★★★★★★★ MoE determinism comparison:
  → DeepSpeed: freeze router (0 LOC) → same input = same routing → deterministic
  → SGLang: murmur_hash32 + constexpr → KERNEL-level → gold standard
  → vLLM: top-k constexpr + Fusion Guard → COMPILE-level → deterministic
  → Megatron: RouterReplay → MoE-only → deterministic routing but not inference
```

---

## Key Findings Summary

★★★★★★★★★ DeepSpeed 10/11 Triton files ALL constexpr → LOW batch invariance risk
★★★★★★★★★ SGLang 7 aten overrides ALL constexpr → gold standard → bypasses Inductor
★★★★★★★★★ vLLM/SGLang/Megatron share SAME batch-invariant override pattern → portable
★★★★★★★★★ verl 1-config autotune → deterministic by construction for own code
★★★★★★★★★ Issue is Inductor-generated Triton (autotuned) → NOT custom kernels (constexpr)
★★★★★★★★★ Fusion Guard blocks bad Inductor fusions → Triton swiglu provides good fused path
★★★★★★★★★ SGLang murmur_hash32 = UNIQUE deterministic MoE routing → KERNEL-level

---

## References

- DeepSpeed kernels: notebook/projects/deepspeed-kernel-source-reading.md
- SGLang deterministic: notebook/projects/sglang-deterministic-inference-source-reading.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Triton swiglu: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
