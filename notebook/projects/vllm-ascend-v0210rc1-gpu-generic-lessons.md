# vLLM-Ascend v0.21.0rc1 — June 16 Release Deep Reading (GPU-generic Lessons)

> 2026-06-18 | v0.21.0rc1 released June 16 | 100+ PRs
> ★★★★★★★★ Focus: portable architectural lessons for RTX 4090/NVIDIA ecosystem
> ★★★★★★★★ DeepSeek-V4, FULL_AND_PIECEWISE, MoE LoRA, batch_invariant_ops, EPLB+PP
> ★★★★★★★★ Ascend-only features documented, GPU-generic lessons extracted

---

## 1. DeepSeek-V4 Full Support — Compose-level Fusion Lessons

```
★★★★★★★★★ DeepSeek-V4 on Ascend 950: piecewise graph mode + DSA + KV cache + MTP + distributed

GPU-generic lessons from DSv4 support:
  → piecewise graph mode: FULL_AND_PIECEWISE → hybrid compilation
  → ★★★★★★★★ SAME concept as vLLM's FULL_AND_PIECEWISE cudagraph_mode
  → DSA (DeepSeek Attention): Sparse Flash Attention + LightningIndexer
  → ★★★★★★★★ LightningIndexer = DSA Indexer Replay for RL → enables GRPO RouterReplay
  → MTP (Multi-Token Prediction): spec decode draft models
  → ★★★★★★★★ D2D NetLoader for spec decode → loads draft model directly to device

★★★★★★★★★ Portability assessment:
  → DSA → CUDA equivalent: FlashAttention with sparse mask → vLLM already has this
  → LightningIndexer → CUDA equivalent: needs implementation → RL RouterReplay
  → FULL_AND_PIECEWISE → CUDA: vLLM v0.23.0 already has this
  → MTP → CUDA: vLLM has spec decode → portable
```

---

## 2. FULL_AND_PIECEWISE Graph Mode

```
★★★★★★★★★ Hybrid compilation strategy: full-graph + piecewise approaches

Key: requires HDK 25.5.1+ / CANN 8.5.0+ on older stacks
  → Removes old stream-budget limitation
  → ~32K graphs on A3, ~64K on Ascend 950
  → ★★★★★★★★ GPU-generic lesson: stream budget IS a real constraint
  → NVIDIA CUDA graphs have similar limits → vLLM addresses this

★★★★★★★★★ GPU-generic: piecewise graph decomposition:
  → Small ops → full-graph (captured as single CUDA/NPU graph)
  → Large ops → piecewise (split into sub-graphs → more flexible)
  → ★★★★★★★★ vLLM v0.23.0 MRv2 uses piecewise cudagraph by default
  → Same architectural decision → proven on both GPU and NPU
```

---

## 3. MoE LoRA for Ascend (#9922)

```
★★★★★★★★★ NEW: AscendFusedMoEWithLoRA / AscendFusedMoE3DWithLoRA wrappers (376 lines)

Architecture: bracket-swapping _apply_mlp per-layer LoRA
  → Before: base MLP → after: base MLP + LoRA bracket around each expert
  → ★★★★★★★★ Same concept as SGLang MoE LoRA → bracket approach is universal
  → AscendFusedMoE v1.5 → LoRA applied at expert level → not router level

★★★★★★★★★ GPU-generic lesson:
  → MoE LoRA MUST apply at expert level → NOT router → for correctness
  → Bracket approach (base + LoRA delta) → works on both Ascend and CUDA
  → ★★★★★★★★ SGLang MoE LoRA uses same bracket pattern → validated on both platforms
```

---

## 4. batch_invariant_ops for RL (#10034, MERGED June 11)

```
★★★★★★★★★ SAME pattern as SGLang: torch.library.Library aten overrides

Ascend implementation:
  → npu_add_rms_norm = SPLIT add + rms_norm (same reason as Inductor!)
  → npu operations registered as aten overrides via torch.library.Library
  → ★★★★★★★★ Same architectural approach as SGLang #24459
  → Both override aten operations → KERNEL-level batch invariance

★★★★★★★★★ GPU-generic lesson:
  → batch_invariant_ops = universal pattern → override aten ops → deterministic
  → SGLang: rms_norm + mm.dtype (9+ overrides)
  → vLLM-Ascend: add_rms_norm + more (Ascend-specific ops)
  → ★★★★★★★★ The pattern is the SAME → the specific ops differ by hardware
  → For RTX 4090: P9 Inductor Fusion Guard provides the SM89 equivalent
```

---

## 5. EPLB + PP Combination (#10486)

```
★★★★★★★★★ NEW: EPLB compatible with Pipeline Parallelism

Problem: EPLB previously hardcoded to single-stage → PP missing layers crashed
  → Fix: PPMissingLayer replaces layers → EPLB now works with PP
  → ★★★★★★★★ GPU-generic lesson: Expert Parallel Load Balancing + PP = feasible!
  → For RTX 4090: EP=1 → no EPLB needed → but multi-GPU future needs this
```

---

## 6. W4A4 INT4 MoE Mega Kernel (#10488)

```
★★★★★★★★★ NEW: single-launch fused MoE kernel for Qwen3.x-MoE on 910B

Replaces 5-launch vendor op chain:
  1. block-diagonal Hadamard
  2. INT4 quant+routing scatter
  3. INT4 gate_up
  4. SwiGLU+requant
  5. INT4 down + unpermute/top-k combine

→ ★★★★★★★★ Single-launch → HBM-bandwidth bound → fewer kernel launches → faster
→ ★★★★★★★★ GPU-generic lesson: MoE kernel fusion is CRITICAL for MoE throughput
→ On NVIDIA: vLLM has similar MoE fusion attempts → Triton MoE default Hopper
→ ★★★★★★★★ Our P6 Triton dequant_swiglu_quant follows same fusion philosophy
```

---

## 7. MC2 Accuracy Regression (#10519) and MoE NaN (#10579)

```
★★★★★★★★★ #10519: MC2 accuracy regression on A2 (Qwen3-235B accuracy dropped to 0-3%!)

  → Multi-concurrency + MC2 collective → accuracy collapsed since May 10
  → Fix: force ALLGATHER on A2 → isolate MC2 regression
  → ★★★★★★★★ GPU-generic lesson: collective communication kernels can regress accuracy!
  → On NVIDIA: NCCL updates sometimes introduce similar regressions
  → ★★★★★★★★ WATCH: vLLM #45683 (MoE combine) addresses same class of problem

★★★★★★★★★ #10579: MoE allgather NaN fix (OPEN June 17)

  → torch.abs on expanded_row_idx before npu_moe_token_unpermute → NaN
  → Fix: remove torch.abs → direct row index without modification
  → ★★★★★★★★ GPU-generic lesson: MoE token permutation must NOT modify indices
  → Same principle as SGLang #27097 → indices must be exact → no abs/max/clamp
```

---

## 8. Dynamic Context Parallelism (DyCP) (#10225)

```
★★★★★★★★★ NEW: Dynamic CP for long-context serving on Ascend

  → MLA CP support + Mooncake KV transfer coordination
  → ★★★★★★★★ GPU-generic lesson: CP is becoming standard for long-context serving
  → vLLM #45964 (MLA decode query replication) = same CP direction
  → ★★★★★★★★ RTX 4090: single GPU → no CP needed → but architectural knowledge useful
```

---

## Key Findings Summary

★★★★★★★★★ v0.21.0rc1: 100+ PRs → DeepSeek-V4, FULL_AND_PIECEWISE, MoE LoRA
★★★★★★★★★ GPU-generic: piecewise graph mode = universal pattern → vLLM has same
★★★★★★★★★ GPU-generic: batch_invariant_ops = same torch.library.Library pattern as SGLang
★★★★★★★★★ GPU-generic: MoE LoRA bracket approach = same as SGLang → universal
★★★★★★★★★ GPU-generic: MoE mega kernel fusion = HBM-bandwidth bound → our P6 follows
★★★★★★★★★ GPU-generic: MC2 accuracy regression → collective comms can break MoE
★★★★★★★★★ GPU-generic: MoE token permutation indices → must NOT modify (abs/max/clamp)
★★★★★★★★★ #9922: MoE LoRA Ascend (376 lines) → same bracket architecture as SGLang
★★★★★★★★★ #10488: W4A4 INT4 mega kernel → 5-launch→1-launch fusion philosophy
★★★★★★★★★ #10486: EPLB+PP combination → Expert Parallel + Pipeline feasible
★★★★★★★★★ CANN 9.1.0 beta for 310P → NPU ecosystem expanding

---

## References

- vLLM-Ascend v0.21.0rc1: https://github.com/vllm-project/vllm-ascend/releases
- #9922: MoE LoRA for Ascend
- #10488: W4A4 INT4 mega kernel
- #10486: EPLB+PP
- #10519: MC2 accuracy regression
- #10579: MoE allgather NaN
- #10225: DyCP
- #10034: batch_invariant_ops RL (MERGED)
- SGLang comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
