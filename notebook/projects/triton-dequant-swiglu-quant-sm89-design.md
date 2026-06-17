# Triton dequant_swiglu_quant Kernel Design — MindIE ATB Fusion Port to CUDA SM89

> 2026-06-18 | Triton kernel design | Port of npu_dequant_swiglu_quant to CUDA
> ★★★★★★★★ MindIE compose-level fusion → Triton kernel → bridges Ascend advantage to NVIDIA
> ★★★★★★★★ MoE W8A8 decode: 6→1 kernel per expert → 48→8 total for 8 experts
> ★★★★★★★★ tl.constexpr BLOCK sizes → DETERMINISTIC → batch-invariant → SGLang-compatible!

---

## 1. MindIE npu_dequant_swiglu_quant — Source Reference

```
★★★★★★★★★ Ascend production source (op-plugin repo):

SwiGLU formula in 1 AscendC kernel:
  dequant_x = x_quant * activation_scale
  dequant_gate = x_quant * weight_scale
  result = SiLU(dequant_x) * dequant_gate   # SwiGLU
  output = result / quant_scale + offset     # requantize

5x kernel launch reduction per expert:
  Without fusion: dequant_weight + matmul_gate + dequant_activation + SiLU + multiply + quant_output = 6 kernels
  With fusion: 1 kernel → 8 experts → 48→8 = 6x!
  "5x" accounts for routing/init/finalize that remain unfused

Source locations:
  → Host: op-plugin/op_plugin/ops/op_host/npu_dequant_swiglu_quant/ (tiling + registration)
  → Kernel: op-plugin/op_plugin/ops/op_kernel/npu_dequant_swiglu_quant/ (AscendC kernel)
  → Python: op-plugin/op_plugin/ops/op/npu_dequant_swiglu_quant.py
```

---

## 2. Triton Port Design — Key Architectural Decisions

```
★★★★★★★★★ Triton kernel: dequant_swiglu_quant_kernel

Why Triton CAN achieve this fusion:
  → All operations are elementwise or row-wise reduction
  → No inter-row dependencies in SwiGLU (per-element activation)
  → Register pressure manageable: N=2048, BLOCK_N=64 → ~256 registers
  → Shared memory: ~4KB for scales → fits SM89 100KB budget
  → Intermediate data stays in GPU registers/shared memory → no HBM round-trip!

★★★★★★★★★ KEY: tl.constexpr BLOCK sizes → DETERMINISTIC → batch-invariant!

  BLOCK_M: tl.constexpr = 32  → fixed across all batch sizes
  BLOCK_N: tl.constexpr = 64  → fixed across all input shapes
  QUANT_MODE: tl.constexpr    → compile-time selection → no runtime branch
  ACTIVATE_LEFT: tl.constexpr → compile-time → matches model convention

★★★★★★★★★ This is SGLang-compatible by design:
  → SGLang's 7 aten overrides use tl.constexpr BLOCK_SIZE → batch-invariant
  → Our Triton kernel uses SAME philosophy → tl.constexpr → no autotuning
  → Contrast with Inductor: autotuned XBLOCK → batch-dependent on SM89
  → Triton kernel bypasses Inductor entirely → deterministic by architecture!

★★★★★★★★★ Why no autotuning (contrast with Inductor CachingAutotuner):
  → Autotuning = XBLOCK varies with input shape → batch-dependent on SM89
  → constexpr BLOCK sizes = fixed → same accumulation order → batch-invariant
  → Trade-off: constexpr may not be "optimal" for every input shape
  → BUT: correctness > performance for GRPO/RL training
  → AND: Triton persistent tiles are fast enough for SM89 anyway
```

---

## 3. Kernel Flow — 5 Steps in 1 Launch

```
★★★★★★★★★ dequant_swiglu_quant_kernel flow:

Step 1: Load INT8 quantized input
  → x_int8 [M, 2*N] gate_up interleaved
  → gate = even columns (offs_gate = 2*offs_n)
  → up = odd columns (offs_up = 2*offs_n + 1)

Step 2: Dequantize
  → gate_val = x_gate * activation_scale * weight_scale
  → up_val = x_up * activation_scale * weight_scale
  → activation_scale: per-token [M] (quant_mode=1) or per-tensor [1] (quant_mode=0)

Step 3: SwiGLU activation
  → SwiGLU = SiLU(gate) * up
  → SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  → activate_left=True: gate*SiLU(up) (DeepSeek/Qwen convention)
  → activate_left=False: up*SiLU(gate)

Step 4: Requantize (per-token dynamic)
  → abs_max = max(|result|) per token
  → quant_scale = abs_max / 127.0
  → output_int8 = round(result / quant_scale) → clamp(-128, 127) → INT8
  → Store quant_scale for down_proj GEMM

Step 5: Store output
  → output_int8 [M, N] → next down_proj grouped GEMM
  → output_scale [M] → per-token requant scale → down_proj uses this
```

---

## 4. MoE W8A8 Decode Path — Full Kernel Launch Comparison

```
★★★★★★★★★ Complete MoE decode path comparison:

| Step | NVIDIA (unfused) | NVIDIA (Triton fused) | Ascend (ATB compose) |
|------|-----------------|-----------------------|---------------------|
| Routing | 1 (top-k) | 1 (top-k) | 1 (npu_moe_init_routing_v2) |
| Gate_up GEMM | 1 (grouped_matmul) | 1 (grouped_matmul) | 1 (npu_grouped_matmul) |
| SwiGLU+quant | 6*E (per-expert) | 1*E (Triton fused) | 1 (npu_dequant_swiglu_quant) |
| Down GEMM | 1 (grouped_matmul) | 1 (grouped_matmul) | 1 (npu_grouped_matmul) |
| Unpermute | 1 | 1 | 1 (npu_moe_finalize_routing) |
| Total (8 experts) | 1+1+48+1+1 = 52 | 1+1+8+1+1 = 12 | 1+1+1+1+1 = 5 |

★★★★★★★★★ Triton fused = 12 vs unfused = 52 → 4.3x launch reduction!
★★★★★★★★★ Ascend compose = 5 → best but requires entire MoE path fused
★★★★★★★★★ Triton approach: incremental improvement → pragmatic → portable!

★★★★★★★★★ Note: grouped_matmul already handles all experts in 1 launch
  → The "6*E" unfused SwiGLU is because each expert needs separate dequant+swiglu+quant
  → Triton fused: handle SwiGLU+quant in 1 kernel with group_index → but current design
    uses per-expert launches (simpler, more portable)
  → Future: Triton grouped SwiGLU kernel with group_index → match Ascend's single-launch
```

---

## 5. SM89-Specific Considerations

```
★★★★★★★★★ RTX 4090 SM89 specifics:

Shared memory: 100KB (vs SM80=164KB, SM90=228KB)
  → BLOCK_M=32 × BLOCK_N=64 → tile sizes manageable
  → Scales storage: activation_scale[M] + weight_scale[N] per block → ~4KB
  → Register usage: ~256 registers per thread → fits SM89 256 register limit

★★★★★★★★★ Why BLOCK_M=32, BLOCK_N=64:
  → 32 rows × 64 columns × 2 halves (gate+up) × 4 bytes = 16KB per tile
  → Fits in SM89 shared memory budget comfortably
  → constexpr → deterministic → SGLang-compatible → batch-invariant
  → No autotuning needed → no XBLOCK variability → no Inductor issues!

★★★★★★★★★ Inductor Fusion Guard relationship:
  → This Triton kernel BY DESIGN avoids the Inductor problem
  → tl.constexpr BLOCK sizes → never autotuned → never batch-dependent
  → The Inductor Fusion Guard (P9) blocks Inductor fusions on SM<90
  → This Triton kernel provides an ALTERNATIVE path: bypass Inductor entirely
  → Together: Fusion Guard blocks bad fusions + Triton provides good fused path
  → ★★★★★★★★ TWO complementary approaches = complete SM89 solution!
```

---

## 6. Contribution Path — vLLM / SGLang Integration

```
★★★★★★★★★ Contribution targets and priority:

Priority P6-P7 (after Inductor Fusion Guard P9 merged):

vLLM integration path:
  → vLLM Triton MoE backend (vllm/attention/backends/triton_moe/)
  → Current: MoE W8A8 uses separate dequant + activation + quant kernels
  → Proposed: dequant_swiglu_quant Triton kernel → 1 kernel per expert
  → W8A8 decode path: grouped_matmul + Triton_swiglu_quant + grouped_matmul
  → ★★★★★★★★ vLLM Triton backend recommended for SM89 (SGLang source confirms)

SGLang integration path:
  → SGLang Triton MoE backend already exists
  → MoE LoRA + deterministic = unique SGLang capability (already documented)
  → Triton swiglu_quant + tl.constexpr = deterministic → natural SGLang fit
  → ★★★★★★★★ SGLang MoE Triton path: most natural integration target!

★★★★★★★★★ Concrete PR targets:
  → vLLM: vllm/model_executor/layers/triton_dequant_swiglu_quant.py (NEW ~200 LOC)
  → SGLang: sglang/layers/triton_dequant_swiglu_quant.py (NEW ~200 LOC)
  → Both: Triton kernel ~100 LOC + host wrapper ~50 LOC + tests ~50 LOC
```

---

## 7. Reference Implementation

```
★★★★★★★★★ See: tools/triton_dequant_swiglu_quant_prototype.py

Features:
  → Triton kernel: dequant_swiglu_quant_kernel with 5 tl.constexpr params
  → Host wrapper: dequant_swiglu_quant_triton() → same API as MindIE
  → Unfused reference: dequant_swiglu_unfused_ref() → 6 operations → correctness baseline
  → Correctness test: Triton vs unfused → ±2 INT8 tolerance
  → Benchmark: Triton vs unfused → expected 2-4x speedup per expert
  → Design info: --mode info → no GPU needed → full design explanation

Modes:
  --mode info: design details (no GPU needed)
  --mode correctness: Triton vs unfused reference
  --mode benchmark: latency comparison
  --mode validate: correctness + benchmark combined

★★★★★★★★★ GPU validation deferred — see rtx4090_gpu_experiment_runner.py P6
```

---

## Key Findings

★★★★★★★★★ Triton dequant_swiglu_quant = MindIE compose fusion port to CUDA SM89
★★★★★★★★★ tl.constexpr BLOCK sizes → DETERMINISTIC → batch-invariant → SGLang-compatible
★★★★★★★★★ MoE W8A8 decode: 6→1 kernels per expert → 48→8 total for 8 experts
★★★★★★★★★ Complementary to Inductor Fusion Guard P9 — Triton provides GOOD fused path while Guard blocks BAD fused path
★★★★★★★★★ SGLang integration: most natural target (already has Triton MoE + deterministic philosophy)
★★★★★★★★★ Priority P6-P7 after Inductor Fusion Guard merged

---

## References

- MindIE ATB compose fusion: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
- SGLang deterministic inference: notebook/projects/sglang-deterministic-inference-source-reading.md
- Inductor Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md
- Triton prototype: tools/triton_dequant_swiglu_quant_prototype.py
- Ascend source: github.com/Ascend/op-plugin → npu_dequant_swiglu_quant
