# MindIE Turbo DeepSeek Inference — Source-Level Analysis

> 2026-06-16 | MindIE Turbo = closed-source DeepSeek-specific inference accelerator | RTX 4090: architectural lessons portable, not directly usable
> Key: compose-level fusion (entire Transformer sub-graph → 1 ATB dispatch) + MLA native kernel + spec decode in latent space

---

## 1. MindIE Turbo Three Pillars

```
★★★★★★★★★ MindIE Turbo = 3 pillars → 2-3x latency reduction over baseline MLA inference on Ascend:

Pillar 1: Speculative Decoding in MLA Latent Space
  → NEXTN/EAGLE-style draft tokens → compressed latent space → reduced draft cost
  → speculative-num-steps=3, eagle-topk=1, speculative-num-draft-tokens=4
  → draft+verify composed into single operation → eliminates sync barriers

Pillar 2: Compose-Level Kernel Fusion
  → entire Transformer sub-graphs → 1 ATB Operation::Compose dispatch
  → MLA projection + RoPE + attention + spec verification → 1 composition
  → TensorBinding zero-copy through on-chip SRAM (AI Core Unified Buffer)

Pillar 3: MLA Native Kernel Acceleration
  → 4 compose-level MLA kernels:
    → npu_kv_rmsnorm_rope_cache: RMSNorm(K) + RoPE(K) + cache_write(K) + cache_write(V) = 4→1
    → npu_mla_preprocess: entire MLA pre-processing pipeline (14+ inputs) → 1 op
    → npu_mla_prolog_v3: W8A8 quantized MLA prolog → even more aggressive fusion
    → npu_dequant_swiglu_quant: dequant + SwiGLU + requant = 6→1 per expert
```

---

## 2. NPU Spec Decode Production Config

SGLang-Ascend production DeepSeek-R1 PD configuration reveals:

```bash
# SGLang-Ascend DeepSeek-R1 PD config
SGLANG_ENABLE_SPEC_V2=1                    # overlap scheduler: draft+verify on separate NPU streams
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1        # stream-level overlap between draft and verification
--speculative-algorithm NEXTN               # DeepSeek MTP as draft method
--speculative-num-steps 3                   # 3 draft generation steps before verification
--speculative-eagle-topk 1                  # single top-k draft token per step (conservative for latency)
--speculative-num-draft-tokens 4            # 4 candidate tokens per iteration
```

### Why spec decode is faster on MindIE Turbo than standard vLLM

On CUDA: spec decode requires separate kernel launches for draft model forward, target model verification, rejection sampling → synchronization barriers between stages.

On MindIE Turbo: entire pipeline (draft generation in MLA latent space + target verification + acceptance/rejection) composed into single operation → eliminates synchronization barriers → dominant overhead removed.

`EAGLEDraftNpuGraphRunner` maps CUDA Graph concepts to NPU Graph (`torch.npu.NPUGraph`), maintaining EAGLE draft structure but replacing CUDA-specific constructs.

---

## 3. MLA Native Kernel — Three Processing Paths

### NPUFusedMLAPreprocess (3 paths)

| Path | Custom Op | Fusion Scope | Quantization |
|------|-----------|-------------|-------------|
| **W8A8 (forward_mlapo)** | `torch.ops.npu.mla_preprocess` | QKV proj + Q norm + Q expand + KV norm + RoPE + W_kc absorption + cache write → 1 op | W8A8 |
| **MLAProlog (forward_mlaprolog)** | `torch.ops.custom.npu_mla_prolog_v3` | token proj + quantize/dequantize + RMSNorm + RoPE + W_kc absorption + cache write → even more aggressive | W8A8 (quantized KV_b_proj excluded) |
| **Absorb (forward_absorb_prepare_npu_rms_norm_cache)** | `torch.ops.npu.npu_kv_rmsnorm_rope_cache` | RMSNorm(K) + RoPE(K) + cache_write(K) + cache_write(V) = 4 sub-ops → 1 kernel | BF16 |

### NPU MLA Attention — Three Modes

| Mode | Description | Key Ops |
|------|-------------|---------|
| **MHA** | Standard multi-head attention | `npu_interleave_rope`, `npu_kv_rmsnorm_rope_cache` |
| **MLA** | Matrix-absorbed attention | `W_kc` absorption via `batch_matmul_transpose`, `W_vc` absorption — avoids explicit KV recovery |
| **DSA** | DeepSeek Sparse Attention | `npu_mla_preprocess` for decode, `fused_split_qk_norm` for Q/K normalization overlap |

### What MLA acceleration does conceptually

On NVIDIA: MLA requires 10+ separate kernel launches:
1. QKV_a projection
2. Q_a norm
3. Q_b projection (expand)
4. KV_a norm
5. RoPE on Q_pe and K_pe
6. W_kc absorption bmm
7. KV cache write
8. attention
9. W_vc absorption bmm
10. o_proj

On Ascend with MLAPO: `torch.ops.npu.mla_preprocess` fuses steps 1-7 → 1 kernel. `batch_matmul_transpose` handles step 9 efficiently. Result: 10+ launches → 3-4.

---

## 4. Compose-Level MoE Fusion: Entire MoE Path → 1 Kernel

### MoE fusion hierarchy on Ascend

| Level | NVIDIA (unfused) | Ascend Path | Kernel Launches | Reduction |
|-------|-------------------|-------------|----------------|-----------|
| W8A8 decode | ~24 (routing+6ops×8exp+combine) | 4 (routing+2gmm+swiglu_quant+unpermute) | 6x | |
| W8A8 DeepEP | ~24+ | 3 (2gmm+dequant_swiglu_quant) | 8x | |
| **FuseEP (full)** | **~24+** | **1 (fused_deep_moe)** | **24x** | ★★★★★★★★ |

### `fused_deep_moe` call signature

```python
buf.fused_deep_moe(
    hidden_states,                  # input tokens
    topk_idx, topk_weights,         # routing decisions
    gmm1_permuted_weight,           # gate+up projection (permuted for grouped GEMM)
    gmm1_permuted_weight_scale,     # gate+up quantization scale
    gmm2_weight,                    # down projection
    gmm2_weight_scale,              # down quantization scale
    num_max_dispatch_tokens_per_rank,
    num_experts,
    fuse_mode=DISPATCH_FFN_COMBINE # full pipeline fusion
)
```

Fuses: MC2 dispatch (token routing) + GEMM1 (gate+up grouped matmul) + SwiGLU activation + GEMM2 (down grouped matmul) + MC2 combine (token reassembly + weighted sum). Everything stays in on-chip SRAM between stages — no HBM round-trips.

### `npu_dequant_swiglu_quant` — flagship sub-kernel

Formula: `result = SiLU(dequant_x) * dequant_gate; output = result / quant_scale + offset`

Fuses: dequantize(weight_scale) + dequantize(activation_scale) + SiLU × gate (SwiGLU) + requantize for next GEMM → 6 kernel launches per expert → 1.

### Why this is impossible on NVIDIA CUDA

1. **Register pressure**: Full MoE path (256 experts in DeepSeek-V3) with dispatch + 2 GEMMs + SwiGLU + combine exceeds GPU register capacity for single kernel
2. **Parallelism mismatch**: dispatch=token-parallel, GEMM=expert-parallel, combine=token-parallel → different thread configs cannot be merged
3. **No Operation::Compose API**: NVIDIA has nothing equivalent. TensorRT is closest but closed-source, static shapes only. torch.compile/Inductor limited to Triton kernels, cannot fuse heterogeneous ops.

---

## 5. Framework Comparison: DeepSeek Inference Paths

| Framework | Platform | MLA | MoE | Spec Decode | Scheduling | RTX 4090? |
|-----------|----------|-----|-----|-------------|-----------|-----------|
| vLLM | CUDA | TritonMLA (SM89) / FlashMLA (SM90+) | Separate kernels (no fusion) | Standard EAGLE/MTP | BudgetRefiner SLO (unique!) | ✓ TritonMLA only |
| SGLang | CUDA | Triton persistent matmul (SM89) | Separate kernels | Standard NEXTN | Deterministic inference (7 aten overrides) | ✓ Triton backend |
| SGLang-Ascend | NPU | MLAPO (`mla_preprocess`) | FuseEP (`fused_deep_moe`) | Spec V2 (NEXTN overlap) | MindIE graph-level (black box) ✗ BudgetRefiner | ✗ NPU only |
| MindIE Turbo | NPU | Compose-level MLA + spec | Compose-level MoE + spec | Fused draft+verify | Closed source, no transparency | ✗ NPU only |
| vLLM-Ascend | NPU | 6 attention backends | MC2 (5 variants, fused_deep_moe pending #8550) | Standard spec | BudgetRefiner SLO (preserved!) | ✗ NPU only |

★★★★★★★★★ **Key**: vLLM-Ascend is the ONLY NPU framework preserving BudgetRefiner SLO scheduling control → MindIE Turbo = black box → no preemption, no SLO control

---

## 6. RTX 4090 Portable Lessons from MindIE Turbo

### Lesson 1: dequant+SwiGLU+quant Triton kernel for CUDA

`npu_dequant_swiglu_quant` pattern (6→1 per expert) is most immediately portable idea. On CUDA/RTX 4090, a Triton kernel fusing dequantize + SwiGLU + requantize for W8A8 MoE experts would reduce kernel launches from ~24 to ~8 for MoE decode. **Concrete vLLM SM89 contribution opportunity.**

### Lesson 2: MLA preprocess fusion is the real bottleneck

On RTX 4090, MLA preprocessing requires 10+ kernel launches. `torch.compile` could fuse some, but **Inductor RMSNorm fusion breaks batch invariance** → Inductor SM<90 Fusion Guard PR is prerequisite to making compile-level MLA fusion viable on RTX 4090.

### Lesson 3: Spec decode needs compose-level thinking

MindIE Turbo insight: draft+verify can run as single composed pipeline without sync barriers. On RTX 4090 with vLLM, spec decode still has separate draft-forward, target-verify, rejection sampling kernels with sync between them. **Overlap scheduling** (SGLang Spec V2 concept) is the practical CUDA equivalent — running draft on separate CUDA stream while previous verification still executing.

### Lesson 4: TritonMLA is RTX 4090's ONLY MLA option

FlashMLA requires SM90 (Seesaw scheduling + Crossover dequant + WGMMA). TritonMLA uses `decode_attention_fwd` which works on SM89 but is memory-bound. RTX 4090 MLA decode will always be slower than H100's FlashMLA because MLA decode is compute-bound on Hopper but memory-bound on SM89 where Tensor Core utilization is lower and no WGMMA.

### Lesson 5: BudgetRefiner SLO applies universally

BudgetRefiner concept (dynamic token budget throttled when decode load high) is 95%+ GPU-generic. Only `profile_table.csv` needs RTX 4090-specific profiling data. **Highest-priority vLLM upstream contribution** — no other framework (including MindIE Turbo) has SLO-aware scheduling compatible with compose-level boundaries.

---

## Key Findings Summary

★★★★★★★★★ MindIE Turbo = compose-level fusion (entire Transformer sub-graph → 1 ATB dispatch) + MLA native kernel + spec decode in latent space → 2-3x latency reduction
★★★★★★★★★ `npu_dequant_swiglu_quant`: dequant+SwiGLU+quant = 6→1 per expert → most portable lesson → Triton kernel for CUDA SM89 = vLLM contribution opportunity
★★★★★★★★★ `fused_deep_moe`: ENTIRE MoE path → 1 kernel → 24x reduction → impossible on CUDA (register pressure + parallelism mismatch + no compose API)
★★★★★★★★★ `npu_mla_preprocess`: 10+ MLA steps → 1 kernel → TritonMLA on RTX 4090 = only option (memory-bound, SM89 no WGMMA)
★★★★★★★★★ Spec V2 overlap scheduling = CUDA practical equivalent of compose-level fusion → separate streams for draft+verify
★★★★★★★★★ Inductor SM<90 Fusion Guard = prerequisite for compile-level MLA fusion on RTX 4090 (RMSNorm fusion breaks batch invariance)
★★★★★★★★★ vLLM-Ascend = ONLY NPU framework preserving BudgetRefiner SLO → MindIE Turbo = scheduling black box
★★★★★★★★★ BudgetRefiner SLO = 95%+ GPU-generic → #1 vLLM upstream contribution priority → unique RTX 4090 profile data

---

## References

- MindIE ATB compose deep dive: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
- MindIE Turbo deployment: notebook/fundamentals/mindie-production-deployment-reading.md
- CANN 9.0 latest: notebook/projects/mindie-cann-9-latest-developments-reading.md
- MLA fundamentals: notebook/fundamentals/mla.md
- FlashMLA reading: notebook/projects/flashmla-reading.md
- NPU MLA preprocess source: sglang/python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py
- NPU MLA attention source: sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
- FuseEP source: sglang/python/sglang/srt/hardware_backend/npu/moe/fuseep.py
- vLLM TritonMLA source: vllm/vllm/v1/attention/backends/mla/triton_mla.py
- SGLang-Ascend DeepSeek PD config: sglang/docs/platforms/ascend/ascend_npu_deepseek_example.md
