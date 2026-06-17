# vLLM batch_invariant.py Source-Level Comparison — SM89 Gap Analysis

> 2026-06-18 | Source reading + cross-framework comparison | How vLLM's existing batch_invariant.py relates to SGLang, vLLM-Ascend, and our P9 Fusion Guard
> ★★★★★★★★ KEY FINDING: vLLM batch_invariant.py does NOT override RMSNorm at aten level → SM89 gap confirmed
> ★★★★★★★★ KEY FINDING: On SM89, vLLM only sets CUBLASLt workspace config → no Triton matmul overrides → P9 fills this gap
> ★★★★★★★★ KEY FINDING: vLLM-Ascend npu_add_rms_norm = SPLIT into add+rms_norm → SAME reason as Inductor fusion!

---

## 1. vLLM batch_invariant.py Architecture (984 Lines)

```
★★★★★★★★★ Source: _temp_vllm/vllm/model_executor/layers/batch_invariant.py

Entry point: init_batch_invariance() (line 974)
  → if envs.VLLM_BATCH_INVARIANT: override_envs() + enable_batch_invariant_mode()

★★★★★★★★★ enable_batch_invariant_mode() (lines 897-951):

torch.library.Library("aten", "IMPL") — SAME pattern as SGLang!

SM-level dispatch (lines 907-919):
  SM80: Triton persistent matmul overrides (4 ops)
    → mm: matmul_persistent (tl.constexpr BLOCK_SIZE_M/N/K)
    → addmm: addmm_batch_invariant (→ matmul_persistent with bias)
    → matmul: matmul_batch_invariant (2D/ND/bmm dispatch)
    → linear: linear_batch_invariant (→ matmul + bias add)

  else (SM89/SM90+/SM100+): CUBLASLt workspace config ONLY
    → CUBLAS_WORKSPACE_CONFIG=":16:8" → disables split-k
    → CUBLASLT_WORKSPACE_SIZE="1" → same intent
    → NO Triton matmul overrides!

Common overrides for ALL CUDA (lines 926-938):
  → _log_softmax: Triton constexpr BLOCK_SIZE kernel
  → softmax/_softmax: amax + subtract + exp + sum + divide (multi-step)
  → mean.dim: Triton constexpr BLOCK_SIZE kernel
  → bmm: Triton constexpr BLOCK_SIZE_M/N/K kernel + torch.bmm monkey-patch

★★★★★★★★★ CRITICAL: RMSNorm is NOT registered as aten override!
  → rms_norm_batch_invariant function EXISTS (lines 825-881) — Triton kernel defined
  → _rms_norm_kernel: BLOCK_SIZE: tl.constexpr → deterministic by design
  → BUT: NOT in enable_batch_invariant_mode() aten registrations!
  → vLLM uses this function DIRECTLY in model layers, not via dispatch
  → When Inductor fuses RMSNorm+reduction → bypasses direct call → non-deterministic!
```

---

## 2. SM89 Gap Analysis — Why P9 Fusion Guard is Needed

```
★★★★★★★★★ On RTX 4090 (SM89 = SM capability 8.9):

What vLLM batch_invariant.py provides on SM89:
  ✓ Matmul: CUBLASLt workspace config (disables split-k) → cuBLAS deterministic
  ✓ softmax: Triton constexpr → deterministic
  ✓ mean.dim: Triton constexpr → deterministic
  ✓ bmm: Triton constexpr → deterministic
  ✓ _log_softmax: Triton constexpr → deterministic
  ✗ RMSNorm: NO aten override → Inductor can fuse → batch-dependent!
  ✗ matmul (when fused): CUBLASLt only → Inductor can fuse matmul+reduction → batch-dependent!

★★★★★★★★★ Why SM89 matmul overrides are missing:
  → SM80 (Ampere A100): cuBLASLt NOT deterministic → needs Triton override
  → SM89 (Ada Lovelace RTX 4090): cuBLASLt IS deterministic with workspace config
  → SM90+ (Hopper H100): TMA/WGMMA → hardware deterministic
  → vLLM assumes: cuBLASLt workspace config = sufficient for SM89
  → BUT: Inductor can fuse matmul into composite → bypasses cuBLAS → Triton path → autotuned!

★★★★★★★★★ The Inductor fusion gap on SM89:
  → torch.compile(vllm_model) → Inductor → sees RMSNorm + mean → fuses them
  → RMSNorm is NOT overridden at aten level → Inductor freely fuses it
  → mean IS overridden → but fusion happens BEFORE dispatch → override bypassed!
  → Our P9 Fusion Guard: blocks fusion on SM<90 → forces separate kernels → overrides work!

★★★★★★★★★ RMSNorm gap — 3 levels of concern:
  Level 1: RMSNorm standalone → vLLM's Triton kernel is deterministic
    → When called directly (e.g., model code calls rms_norm_batch_invariant)
    → Deterministic because BLOCK_SIZE: tl.constexpr

  Level 2: RMSNorm + add → fused_add_rms_norm → vLLM uses custom C++ op
    → ops.fused_add_rms_norm → C++ kernel → deterministic
    → NOT subject to Inductor fusion (it's a custom op)

  Level 3: Inductor rewrites RMSNorm → THIS is the problem
    → When torch.compile rewrites RMSNorm as a series of aten ops
    → Inductor sees: x² → mean → add_eps → sqrt → div → mul_weight
    → Fuses mean+div → one Triton kernel → autotuned BLOCK_SIZE → batch-dependent!
    → P9 blocks this fusion → mean stays as separate kernel → override works!
```

---

## 3. Cross-Framework Override Comparison Matrix

```
★★★★★★★★★ Override coverage per SM level per framework:

| Operation      | SGLang (SM89) | vLLM (SM89)    | vLLM (SM80)   | vLLM-Ascend (NPU) | P9 Need? |
|---------------|---------------|----------------|---------------|-------------------|----------|
| mm/matmul     | Triton constexpr| CUBLASLt config| Triton constexpr| AscendC mm       | YES — Inductor bypass |
| addmm         | Triton constexpr| CUBLASLt config| Triton constexpr| —                | YES |
| linear        | Triton constexpr| CUBLASLt config| Triton constexpr| Triton linear   | YES |
| bmm           | Triton constexpr| Triton constexpr| Triton constexpr| Triton bmm      | NO — already covered |
| softmax       | Triton constexpr| multi-step (amax+exp+div)| Triton constexpr| Triton softmax  | NO |
| log_softmax   | Triton constexpr| Triton constexpr| Triton constexpr| —                | NO |
| mean.dim      | Triton constexpr| Triton constexpr| Triton constexpr| torch.sum patch  | NO — but fusion bypasses! |
| rms_norm      | Triton constexpr| ★★★ NOT aten override ★★★| Triton constexpr| npu_add_rms_norm SPLIT | YES — biggest gap! |
| silu          | Triton constexpr| NO override    | NO override   | —                 | YES — fused SwiGLU path |
| sigmoid       | Triton constexpr| NO override    | NO override   | —                 | YES — fused SwiGLU path |
| mul           | Triton constexpr| NO override    | NO override   | —                 | YES — fused SwiGLU path |

★★★★★★★★★ Summary:
  → SGLang: 7 overrides → ALL tl.constexpr → KERNEL-level → strongest
  → vLLM SM89: 5 overrides + CUBLASLt config → BUT RMSNorm/silu/sigmoid/mul NOT overridden
  → vLLM SM80: 9 overrides (4 matmul + 5 common) → stronger than SM89
  → vLLM-Ascend: AscendC+Triton dual-tier → hardware deterministic + Triton fallback
  → Our P9: fills SM89 gaps by blocking Inductor fusion → makes existing overrides work
```

---

## 4. vLLM vs SGLang Override Pattern Comparison

```
★★★★★★★★★ Same torch.library.Library("aten", "IMPL") pattern:

SGLang (sglang/batch_invariant.py):
  lib = torch.library.Library("aten", "IMPL")
  lib.impl("aten::rms_norm", rms_norm_impl, "CUDA")
  lib.impl("aten::silu", silu_impl, "CUDA")
  ... (7 overrides)

vLLM (vllm/model_executor/layers/batch_invariant.py):
  _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
  _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
  ... (9 overrides on SM80, 5 on SM89/else)

★★★★★★★★★ Differences in approach:

1. Registration scope:
   SGLang: ALL overrides apply to ALL CUDA SM levels
   vLLM: SM-level conditional → SM80 gets 4 extra matmul overrides

2. Matmul implementation:
   SGLang: Triton kernel with tl.constexpr → persistent-style
   vLLM: Triton persistent matmul_kernel_persistent → SAME tl.constexpr approach!
   → Both use tl.dot(a, b, accumulator) inner-loop → same accumulation order

3. RMSNorm:
   SGLang: aten override → Inductor NEVER sees RMSNorm → bypassed entirely
   vLLM: function only, NOT aten override → Inductor CAN rewrite RMSNorm

4. Sampling:
   SGLang: murmur_hash32 Gumbel-max float64 → GPU-side deterministic sampling
   vLLM: no deterministic sampling override → relies on torch default

★★★★★★★★★ What vLLM has that SGLang does NOT:
  → matmul_kernel_persistent: sophisticated tiled GEMM with NUM_SMS, GROUP_SIZE_M, LARGE tensors
  → bmm_kernel: explicit batched GEMM Triton kernel (SGLang uses torch.bmm with override)
  → addmm_bias: fused bias in matmul persistent kernel → fewer kernel launches
  → mean_dim: full multi-dimension iterative reduction with sorted dims
  → override_envs_for_invariance(): comprehensive NCCL/CUBLAS/torch.compile env vars
```

---

## 5. vLLM-Ascend batch_invariant_ops — Cross-Platform Pattern

```
★★★★★★★★★ vLLM-Ascend (#10034) — SAME architectural pattern, DIFFERENT hardware:

Source: _temp_vllm-ascend/vllm_ascend/batch_invariant.py (150 lines)

AscendC priority (hardware-native deterministic):
  → mm, matmul, sum, fused_infer_attention_score, add_rms_norm

Triton fallback (software deterministic):
  → mm, matmul, addmm, bmm, softmax, linear

★★★★★★★★★ npu_add_rms_norm = SPLIT into add + rms_norm — WHY?
  → Ascend ATB fuses add + rms_norm into one Operation::Compose
  → On NPU: deterministic by hardware design → no autotuning
  → On CUDA: Inductor fuses add + rms_norm → one Triton kernel → autotuned → batch-dependent!
  → ★★★★★★★★ SAME architectural reason for SPLIT on both platforms!
  → Ascend SPLIT because ATB compose is already deterministic → no need for fused path
  → CUDA SPLIT because Inductor fusion is non-deterministic → MUST block fusion!

★★★★★★★★★ override_envs_for_invariance() comparison:

vLLM (CUDA):
  CUBLAS_WORKSPACE_CONFIG, NCCL_* envs, VLLM_USE_AOT_COMPILE=0, TF32→IEEE

vLLM-Ascend (NPU):
  HCCL_DETERMINISTIC="strict", LCCL_DETERMINISTIC="1", weight_nz_mode=0,
  enable_matmul_allreduce=False

★★★★★★★★★ Cross-platform lesson:
  → Hardware-deterministic (Ascend) + software-deterministic (CUDA tl.constexpr)
  → BOTH need to SPLIT fused ops that mix reduction + pointwise
  → tl.constexpr = software equivalent of hardware-deterministic tiling
  → ★★★★★★★★ Triton tl.constexpr = AscendC deterministic tiling → hardware-agnostic!
```

---

## 6. P9 Fusion Guard — Fills vLLM SM89 Gap

```
★★★★★★★★★ How P9 Inductor Fusion Guard fills the SM89 gap:

vLLM batch_invariant.py on SM89:
  → CUBLASLt workspace config for matmuls → deterministic IF called separately
  → Triton overrides for softmax/mean/bmm → deterministic IF called separately
  → RMSNorm: NOT aten override → Inductor CAN fuse → NOT deterministic
  → Problem: Inductor FUSES operations → overrides bypassed → non-deterministic

P9 Fusion Guard (5-line choices.py):
  → props.major < 9 → WhyNoFuse("batch_invariance") → blocks ALL reductions fusions
  → Forces separate kernels → each goes through aten dispatch → overrides ACTIVATE
  → RMSNorm → mean → sqrt → div → each SEPARATE → mean override works!

★★★★★★★★★ P9 + vLLM batch_invariant.py = COMPLETE SM89 solution:
  → P9 blocks bad fusions → forces separate kernel dispatch
  → vLLM's existing overrides ACTIVATE when dispatched separately
  → CUBLASLt config → deterministic matmuls → separate kernels → works!
  → Triton mean/bmm/softmax → deterministic when separate → works!
  → Together: CORRECT and FAST (separate kernels still use Triton constexpr)

★★★★★★★★★ Why NOT just add RMSNorm aten override to vLLM?
  → Could work → but:
  1. Would need vLLM upstream PR → slow review process
  2. Would need ALL models to call rms_norm_batch_invariant explicitly
  3. Would NOT fix matmul+reduction fusion (still no matmul override on SM89)
  4. Would NOT fix SwiGLU fusion (silu+sigmoid+mul not overridden)
  5. P9 is MORE comprehensive → blocks ALL bad fusions → covers all gaps at once!

★★★★★★★★★ Optimal strategy: P9 FIRST → then extend vLLM overrides
  → P9 Fusion Guard → PyTorch upstream → blocks bad fusions universally
  → Then: add RMSNorm aten override to vLLM → makes explicit call path available
  → Then: Triton dequant_swiglu_quant (P6) → provides GOOD fused path → faster
  → Progressive enhancement: P9 blocks bad → vLLang overrides explicit → P6 provides good fused
```

---

## 7. Triton Kernel Design Comparison

```
★★★★★★★★★ matmul_kernel_persistent (vLLM) vs SGLang matmul:

vLLM matmul_kernel_persistent (lines 41-129):
  → BLOCK_SIZE_M/N/K: tl.constexpr → deterministic tiling
  → GROUP_SIZE_M: tl.constexpr → L-grouped scheduling
  → NUM_SMS: tl.constexpr → persistent across SM count
  → A_LARGE/B_LARGE/C_LARGE: tl.constexpr → int64 offset support
  → HAS_BIAS: tl.constexpr → fused bias addition
  → tl.dot(a, b, accumulator) → inner loop accumulation → deterministic order
  → dtype-specialized configs: bf16/fp16/fp32 separate BLOCK_SIZE profiles
  → FP16: BLOCK_SIZE_N = 256 or 128 (SM-dependent via shared memory check)

SGLang matmul override:
  → BLOCK_SIZE: tl.constexpr → same deterministic principle
  → dtype specialization → float16/bfloat16 separate constexpr paths
  → Same tl.dot inner-loop → same accumulation guarantee

★★★★★★★★★ Both achieve SAME determinism via SAME mechanism:
  → tl.constexpr → no autotuning → fixed block sizes → fixed accumulation order
  → tl.dot(a, b, accumulator) → hardware inner-product → deterministic per tile
  → Persistent-style grid → deterministic tile assignment → same order across batches

★★★★★★★★★ _rms_norm_kernel (vLLM) — lines 775-881:
  → BLOCK_SIZE: tl.constexpr → deterministic
  → Row-wise: each row = one program → independent → batch-invariant
  → Three-step: (1) sum_sq via tl.sum (2) compute inv_rms (3) normalize+weight
  → Float32 accumulation → vals.to(tl.float32) before squaring
  → tl.where(mask, sq_vals, 0.0) → masked accumulation → deterministic padding

★★★★★★★★★ _log_softmax_kernel (vLLM) — lines 341-401:
  → BLOCK_SIZE: tl.constexpr → deterministic
  → Three-pass: (1) max (2) sum_exp (3) output = x - max - log_sum_exp
  → Row-wise: one program per row → batch-invariant
  → tl.where(mask, exp_vals, 0.0) → masked accumulation → deterministic
```

---

## 8. Override Environment Variables Comparison

```
★★★★★★★★★ vLLM override_envs_for_invariance() (lines 953-971):

CUDA-specific:
  CUBLAS_WORKSPACE_CONFIG=":4096:8"  ← NOTE: different from enable_batch_invariant_mode!
    → enable_batch_invariant_mode sets ":16:8" for SM90+
    → override_envs sets ":4096:8" — 4096 workspace = more conservative

NCCL determinism:
  NCCL_LAUNCH_MODE="GROUP"          → synchronous group launches
  NCCL_COLLNET_ENABLE="0"           → disable collective network
  NCCL_NVLS_ENABLE="0"              → disable NVLink SHARP
  NCCL_P2P_NET_DISABLE="1"          → disable P2P network
  NCCL_MIN/MAX_NCHANNELS="1"        → single channel → deterministic
  NCCL_PROTO="Simple"               → Simple protocol → deterministic
  NCCL_ALGO="allreduce:tree"        → tree algorithm → deterministic
  NCCL_NTHREADS/SOCKET_NTHREADS="1" → single thread → deterministic

torch.compile:
  VLLM_USE_AOT_COMPILE="0"          → disable AOT → use JIT → more predictable

★★★★★★★★★ vLLM-Ascend override_envs comparison:

Ascend-specific:
  HCCL_DETERMINISTIC="strict"       → strict HCCL determinism
  LCCL_DETERMINISTIC="1"            → LCCL deterministic mode
  weight_nz_mode=0                  → NZ format disabled → deterministic layout
  enable_matmul_allreduce=False      → disable fused matmul+allreduce

★★★★★★★★★ SGLang override_envs (inferred from source):
  → Less comprehensive env overrides → relies on KERNEL-level overrides more
  → Triton constexpr = primary mechanism → env vars as secondary safety net
```

---

## Key Findings Summary

★★★★★★★★★ vLLM batch_invariant.py (984 lines) EXISTS but has SM89 gaps:
  - matmul: only CUBLASLt workspace config → NO Triton override on SM89
  - RMSNorm: function defined but NOT registered as aten override → Inductor can fuse
  - silu/sigmoid/mul: NOT overridden → SwiGLU fusion non-deterministic

★★★★★★★★★ P9 Fusion Guard fills ALL SM89 gaps at once:
  - blocks Inductor reduction fusions → forces separate kernel dispatch
  - makes vLLM's existing overrides ACTIVATE → deterministic when separate
  - more comprehensive than adding individual overrides → covers unknown future fusions too

★★★★★★★★★ Cross-platform pattern: torch.library.Library("aten", "IMPL") × 3:
  - SGLang: 7 overrides → KERNEL-level → strongest (rms_norm IS overridden)
  - vLLM: 9 overrides (SM80) / 5 overrides (SM89) → COMPILE-level → P9 fills gap
  - vLLM-Ascend: AscendC+Triton dual-tier → hardware+software → deterministic by design

★★★★★★★★★ npu_add_rms_norm = SPLIT → SAME reason as Inductor fusion block:
  - Ascend: SPLIT because ATB compose is deterministic → no fused path needed
  - CUDA: SPLIT (via P9) because Inductor fusion is non-deterministic → must block
  - Both platforms: reduction+pointwise fusion = determinism risk → SPLIT is the answer

★★★★★★★★★ Triton tl.constexpr = hardware-agnostic determinism:
  - vLLM matmul_kernel_persistent tl.constexpr = SGLang tl.constexpr = AscendC fixed tiling
  - Same principle: fixed block sizes → fixed accumulation order → batch-invariant

★★★★★★★★★ Optimal progressive strategy: P9 → vLLang RMSNorm override → P6 Triton swiglu
  - P9 blocks bad fusions universally → immediate SM89 fix
  - Then: add RMSNorm aten override to vLLM → explicit deterministic path
  - Then: P6 Triton dequant_swiglu_quant → GOOD fused path → faster than unfused

---

## References

- vLLM batch_invariant.py: _temp_vllm/vllm/model_executor/layers/batch_invariant.py (984 lines)
- vLLM-Ascend batch_invariant.py: _temp_vllm-ascend/vllm_ascend/batch_invariant.py (150 lines)
- SGLang batch_invariant: notebook/projects/sglang-deterministic-inference-source-reading.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Fusion Guard PR draft: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Triton swiglu design: notebook/projects/triton-dequant-swiglu-quant-sm89-design.md
- Cross-framework comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
