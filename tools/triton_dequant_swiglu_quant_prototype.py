#!/usr/bin/env python3
"""
★★★★★★★★★ Triton dequant_swiglu_quant Kernel Prototype for SM89 (RTX 4090)
★★★★★★★★★ Port of MindIE ATB compose-level fusion to CUDA via Triton
★★★★★★★★★ Inspiration: npu_dequant_swiglu_quant (Ascend op-plugin)
★★★★★★★★★ Fuses: dequantize_activation + SwiGLU + requantize → 1 kernel launch
★★★★★★★★★ MoE W8A8 decode path: 6→1 kernels per expert → 48→8 total for 8 experts

Reference: notebook/projects/mindie-atb-compose-fusion-deep-reading.md
Reference: Ascend op-plugin npu_dequant_swiglu_quant host+kernel source

USAGE (GPU required):
  python tools/triton_dequant_swiglu_quant_prototype.py --mode benchmark
  python tools/triton_dequant_swiglu_quant_prototype.py --mode validate
  python tools/triton_dequant_swiglu_quant_prototype.py --mode correctness
  python tools/triton_dequant_swiglu_quant_prototype.py --mode info

Note: This requires an NVIDIA GPU (SM89 RTX 4090 preferred).
      If no GPU available, --mode info will show design and expected performance.
"""

import argparse
import sys
import time

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ============================================================
# Triton Kernel: dequant_swiglu_quant
# ============================================================

if HAS_TRITON:
    @triton.jit
    def dequant_swiglu_quant_kernel(
        # Pointers
        x_ptr,           # Input: INT8 quantized activations (gate + up interleaved)
        activation_scale_ptr,  # Per-token activation dequant scale
        weight_scale_ptr,      # Per-expert weight dequant scale (optional)
        output_ptr,      # Output: INT8 requantized result
        output_scale_ptr, # Output: per-token requant scale (for next GEMM)
        # Dimensions
        M,               # Number of tokens
        N_HALF,          # Half of gate_up dimension (hidden_dim for each half)
        # Strides
        stride_x_m, stride_x_n,
        stride_act_m,
        stride_ws_n,     # weight_scale stride (per-expert, 1 value per column group)
        stride_out_m, stride_out_n,
        stride_os_m,
        # Config
        QUANT_MODE: tl.constexpr,   # 0=per-tensor, 1=per-token dynamic
        ACTIVATE_LEFT: tl.constexpr, # True=gate on left (SwiGLU), False=up on left
        HAS_WEIGHT_SCALE: tl.constexpr,  # True if weight_scale provided
        BLOCK_M: tl.constexpr = 32,
        BLOCK_N: tl.constexpr = 64,
    ):
        """
        ★★★★★★★★ Fused dequantize + SwiGLU + requantize kernel

        SwiGLU formula (fused in 1 kernel):
          dequant_x = x_quant * activation_scale   # dequantize
          gate, up = split(dequant_x)              # split gate_up
          result = SiLU(gate) * up                 # SwiGLU activation
          output = quantize(result, quant_scale)   # requantize for next GEMM

        On Ascend: this entire sequence = 1 npu_dequant_swiglu_quant kernel
        On CUDA (Triton): this = 1 Triton kernel → same fusion, different hardware

        Key: intermediate data stays in GPU registers/shared memory → no HBM round-trip!
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Tile offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Mask for boundary handling
        mask_m = offs_m < M
        mask_n = offs_n < N_HALF
        mask = mask_m & mask_n

        # ===== STEP 1: Load INT8 quantized input =====
        # gate_up is interleaved: [gate_col0, up_col0, gate_col1, up_col1, ...]
        # So for gate: index = 2*n, for up: index = 2*n+1
        offs_gate = 2 * offs_n       # gate columns at even positions
        offs_up = 2 * offs_n + 1     # up columns at odd positions

        # Load INT8 values
        x_gate_ptrs = x_ptr + offs_m * stride_x_m + offs_gate * stride_x_n
        x_up_ptrs = x_ptr + offs_m * stride_x_m + offs_up * stride_x_n

        # Cast INT8 to FP32 for computation
        x_gate = tl.load(x_gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_up = tl.load(x_up_ptrs, mask=mask, other=0.0).to(tl.float32)

        # ===== STEP 2: Dequantize =====
        # Load per-token activation scale
        if QUANT_MODE == 1:
            act_scale_ptrs = activation_scale_ptr + offs_m * stride_act_m
            act_scale = tl.load(act_scale_ptrs, mask=mask_m, other=1.0)
            act_scale = act_scale.to(tl.float32)
        else:
            act_scale = tl.load(activation_scale_ptr).to(tl.float32)

        if HAS_WEIGHT_SCALE:
            ws_ptrs = weight_scale_ptr + offs_n * stride_ws_n
            ws = tl.load(ws_ptrs, mask=mask_n, other=1.0).to(tl.float32)
            # Combine: dequant = x_int8 * activation_scale * weight_scale
            gate_val = x_gate * act_scale * ws
            up_val = x_up * act_scale * ws
        else:
            gate_val = x_gate * act_scale
            up_val = x_up * act_scale

        # ===== STEP 3: SwiGLU activation =====
        # SwiGLU = SiLU(gate) * up
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        if ACTIVATE_LEFT:
            # gate on left side (standard DeepSeek/Qwen MoE convention)
            sigmoid_gate = tl.sigmoid(gate_val)
            silu_gate = gate_val * sigmoid_gate
            result = silu_gate * up_val
        else:
            # up on left side
            sigmoid_up = tl.sigmoid(up_val)
            silu_up = up_val * sigmoid_up
            result = gate_val * silu_up

        # ===== STEP 4: Requantize (dynamic per-token) =====
        if QUANT_MODE == 1:
            # Per-token dynamic quant: compute scale from max absolute value
            # abs_max = max(|result|) over N_HALF dimension per token
            abs_result = tl.abs(result)
            # Reduce across N_HALF dimension
            # Triton reduction: use tl.max with axis=1 for row-wise max
            abs_max = tl.max(abs_result, axis=1)  # per-token max over columns

            # quant_scale = abs_max / 127.0 (INT8 range)
            quant_scale = abs_max / 127.0
            # Clamp to avoid division by zero
            quant_scale = tl.maximum(quant_scale, 1e-8)

            # Quantize: output_int8 = round(result / quant_scale)
            output_int = tl.libelement.round(result / quant_scale[:, None])
            # Clamp to INT8 range [-128, 127]
            output_int = tl.maximum(output_int, -128.0)
            output_int = tl.minimum(output_int, 127.0)
            output_int = output_int.to(tl.int8)

            # Store output scale for next GEMM
            os_ptrs = output_scale_ptr + offs_m * stride_os_m
            tl.store(os_ptrs, quant_scale, mask=mask_m)
        else:
            # Per-tensor quant (simpler)
            quant_scale = tl.load(output_scale_ptr).to(tl.float32)
            output_int = tl.libelement.round(result / quant_scale)
            output_int = tl.maximum(output_int, -128.0)
            output_int = tl.minimum(output_int, 127.0)
            output_int = output_int.to(tl.int8)

        # ===== STEP 5: Store output =====
        out_ptrs = output_ptr + offs_m * stride_out_m + offs_n * stride_out_n
        tl.store(out_ptrs, output_int, mask=mask)


# ============================================================
# Host-side wrapper functions
# ============================================================

def dequant_swiglu_quant_triton(
    x_int8: torch.Tensor,           # [M, 2*N] INT8 gate_up interleaved
    activation_scale: torch.Tensor,  # [M] per-token or [1] per-tensor
    weight_scale: torch.Tensor = None,  # [N] per-column or None
    quant_mode: int = 1,            # 0=per-tensor, 1=per-token dynamic
    activate_left: bool = True,     # gate on left (SwiGLU convention)
) -> tuple:
    """
    ★★★★★★★★ Triton equivalent of npu_dequant_swiglu_quant

    Args:
        x_int8: INT8 quantized gate+up activations, shape [M, 2*N]
        activation_scale: dequant scale, per-token [M] or per-tensor [1]
        weight_scale: optional weight dequant scale [N] or None
        quant_mode: 0=per-tensor static, 1=per-token dynamic
        activate_left: True=gate*SiLU(up), False=up*SiLU(gate)

    Returns:
        output_int8: [M, N] INT8 requantized SwiGLU output
        output_scale: [M] per-token requant scale (for down_proj GEMM)
    """
    assert x_int8.dtype == torch.int8, f"Expected int8 input, got {x_int8.dtype}"
    assert x_int8.is_cuda, "Triton kernel requires CUDA tensors"

    M, N_full = x_int8.shape
    N = N_full // 2  # gate_up interleaved: 2*N columns → N output columns
    assert N_full % 2 == 0, "gate_up dimension must be even (interleaved)"

    # Output tensors
    output_int8 = torch.empty((M, N), dtype=torch.int8, device=x_int8.device)
    output_scale = torch.empty((M,), dtype=torch.float32, device=x_int8.device)

    # Launch config
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    dequant_swiglu_quant_kernel[grid](
        x_ptr=x_int8,
        activation_scale_ptr=activation_scale,
        weight_scale_ptr=weight_scale if weight_scale is not None else x_int8,  # dummy if None
        output_ptr=output_int8,
        output_scale_ptr=output_scale,
        M=M, N_HALF=N,
        stride_x_m=x_int8.stride(0), stride_x_n=x_int8.stride(1),
        stride_act_m=activation_scale.stride(0),
        stride_ws_n=weight_scale.stride(0) if weight_scale is not None else 1,
        stride_out_m=output_int8.stride(0), stride_out_n=output_int8.stride(1),
        stride_os_m=output_scale.stride(0),
        QUANT_MODE=quant_mode,
        ACTIVATE_LEFT=activate_left,
        HAS_WEIGHT_SCALE=weight_scale is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return output_int8, output_scale


def dequant_swiglu_unfused_ref(
    x_int8: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor = None,
    quant_mode: int = 1,
    activate_left: bool = True,
) -> tuple:
    """
    ★★★★★★★★ Unfused PyTorch reference: 6 separate operations
    This is what happens WITHOUT our Triton fusion:
      1. dequantize_activation → x * scale → FP32
      2. split → gate, up
      3. SiLU(gate)
      4. multiply → gate_silu * up
      5. compute_quant_scale → max abs per token
      6. requantize → round(result / scale) → INT8

    Total: 6 kernel launches per expert
    With Triton fusion: 1 kernel launch per expert → 6x reduction!
    """
    M, N_full = x_int8.shape
    N = N_full // 2

    # Step 1: Dequantize
    x_fp32 = x_int8.float() * activation_scale.unsqueeze(1)
    if weight_scale is not None:
        x_fp32 = x_fp32 * weight_scale.unsqueeze(0)

    # Step 2: Split gate and up
    gate = x_fp32[:, 0::2]  # even columns
    up = x_fp32[:, 1::2]    # odd columns

    # Step 3+4: SwiGLU
    if activate_left:
        result = torch.nn.functional.silu(gate) * up
    else:
        result = gate * torch.nn.functional.silu(up)

    # Step 5: Compute quant scale (per-token dynamic)
    if quant_mode == 1:
        abs_max = result.abs().amax(dim=1)
        quant_scale = abs_max / 127.0
        quant_scale = quant_scale.clamp(min=1e-8)
    else:
        abs_max = result.abs().max()
        quant_scale = abs_max / 127.0
        quant_scale = torch.tensor([quant_scale], device=x_int8.device)

    # Step 6: Requantize
    output_int8 = (result / quant_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

    return output_int8, quant_scale


# ============================================================
# Validation and benchmarking
# ============================================================

def run_correctness_test():
    """★★★★★★★★★ Validate Triton kernel matches unfused reference"""
    if not torch.cuda.is_available():
        print("  ✗ No CUDA GPU available — cannot run correctness test")
        return False

    device = torch.device("cuda")
    print("  Running correctness test on CUDA...")

    test_configs = [
        (16, 512),    # Small: single token decode
        (32, 1024),   # Medium: small batch decode
        (128, 2048),  # Large: typical MoE decode batch
    ]

    all_pass = True
    for M, N in test_configs:
        x_int8 = torch.randint(-128, 127, (M, 2*N), dtype=torch.int8, device=device)
        act_scale = torch.rand(M, dtype=torch.float32, device=device) * 2.0 + 0.5
        ws = torch.rand(N, dtype=torch.float32, device=device) * 1.5 + 0.5

        # Triton fused
        out_triton, scale_triton = dequant_swiglu_quant_triton(
            x_int8, act_scale, ws, quant_mode=1, activate_left=True
        )

        # PyTorch unfused reference
        out_ref, scale_ref = dequant_swiglu_unfused_ref(
            x_int8, act_scale, ws, quant_mode=1, activate_left=True
        )

        # INT8 comparison: allow ±1 tolerance (rounding differences)
        diff = (out_triton.float() - out_ref.float()).abs().max().item()
        scale_diff = (scale_triton - scale_ref).abs().max().item()

        pass_correctness = diff <= 2 and scale_diff <= 0.01  # ±2 INT8 tolerance
        status = "PASS" if pass_correctness else "FAIL"
        print(f"    M={M}, N={N}: output_diff={diff}, scale_diff={scale_diff:.6f} → {status}")
        all_pass = all_pass and pass_correctness

    return all_pass


def run_benchmark():
    """★★★★★★★★★ Benchmark Triton fused vs PyTorch unfused"""
    if not torch.cuda.is_available():
        print("  ✗ No CUDA GPU available — cannot run benchmark")
        return None

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")

    # Typical MoE decode dimensions
    configs = [
        # (M, N, num_experts) — M=tokens, N=hidden_per_expert
        (8, 512, 8),    # Small: 8 tokens, 512 hidden, 8 experts
        (16, 1024, 8),  # Medium: 16 tokens, 1024 hidden, 8 experts
        (32, 2048, 8),  # Large: 32 tokens, 2048 hidden, 8 experts (DeepSeek-style)
    ]

    for M, N, num_experts in configs:
        x_int8 = torch.randint(-128, 127, (M, 2*N), dtype=torch.int8, device=device)
        act_scale = torch.rand(M, dtype=torch.float32, device=device) * 2.0 + 0.5
        ws = torch.rand(N, dtype=torch.float32, device=device) * 1.5 + 0.5

        # Warmup
        for _ in range(10):
            dequant_swiglu_quant_triton(x_int8, act_scale, ws)
        torch.cuda.synchronize()

        # Triton fused benchmark
        start = time.perf_counter()
        num_iters = 100
        for _ in range(num_iters):
            out_t, scale_t = dequant_swiglu_quant_triton(x_int8, act_scale, ws)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_iters * 1000  # ms

        # Warmup unfused
        for _ in range(10):
            dequant_swiglu_unfused_ref(x_int8, act_scale, ws)
        torch.cuda.synchronize()

        # PyTorch unfused benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            out_r, scale_r = dequant_swiglu_unfused_ref(x_int8, act_scale, ws)
        torch.cuda.synchronize()
        unfused_time = (time.perf_counter() - start) / num_iters * 1000  # ms

        # MoE total: per-expert * num_experts
        triton_moe = triton_time * num_experts
        unfused_moe = unfused_time * num_experts

        speedup = unfused_time / triton_time if triton_time > 0 else float('inf')

        print(f"  M={M}, N={N}, experts={num_experts}:")
        print(f"    Triton fused: {triton_time:.3f} ms per expert, {triton_moe:.3f} ms total MoE")
        print(f"    PyTorch unfused: {unfused_time:.3f} ms per expert, {unfused_moe:.3f} ms total MoE")
        print(f"    ★★★★★ Speedup: {speedup:.2f}x (per expert)")
        print(f"    ★★★★★ MoE kernel launches: Triton={num_experts} vs unfused={6*num_experts}")
        print()


def print_design_info():
    """★★★★★★★★★ Print kernel design and expected performance (no GPU needed)"""
    print("★★★★★★★★★ Triton dequant_swiglu_quant Kernel Design for SM89 ★★★★★★★★★★")
    print()
    print("  Port of MindIE ATB compose-level fusion to CUDA via Triton")
    print("  Reference: npu_dequant_swiglu_quant (Ascend op-plugin)")
    print()
    print("  Fused operations (1 Triton kernel):")
    print("    1. Load INT8 quantized gate_up activations")
    print("    2. Dequantize: x_fp32 = x_int8 * activation_scale * weight_scale")
    print("    3. Split: gate = even columns, up = odd columns")
    print("    4. SwiGLU: SiLU(gate) * up")
    print("    5. Requantize: compute per-token max abs → quant_scale → round → INT8")
    print("    6. Store INT8 output + per-token quant_scale for down_proj GEMM")
    print()
    print("  ★★★★★★★★ MoE W8A8 decode path comparison:")
    print("    Without fusion (per expert): 6 kernel launches")
    print("      → dequant_activation + split + SiLU + multiply + compute_scale + requantize")
    print("    With Triton fusion (per expert): 1 kernel launch")
    print("      → dequant_swiglu_quant_kernel")
    print("    For 8 experts: 48 → 8 kernel launches = 6x reduction!")
    print()
    print("  ★★★★★★★★ Why Triton can achieve this fusion:")
    print("    → All operations are elementwise or row-wise reduction")
    print("    → No inter-row dependencies in SwiGLU (per-element activation)")
    print("    → Register pressure manageable: N=2048, BLOCK_N=64 → ~256 registers")
    print("    → Shared memory: activation_scale + weight_scale per block → ~4KB")
    print("    → SM89 shared memory: 100KB → ample room for BLOCK_M=32 × BLOCK_N=64 tiles")
    print()
    print("  ★★★★★★★★ SM89-specific considerations:")
    print("    → SM89 = 100KB shared memory (vs SM80=164KB, SM90=228KB)")
    print("    → BLOCK_M=32, BLOCK_N=64 → fits in SM89 shared memory budget")
    print("    → Triton constexpr BLOCK sizes → DETERMINISTIC → matches SGLang philosophy!")
    print("    → No autotuning needed → BLOCK sizes are constexpr → batch-invariant!")
    print()
    print("  ★★★★★★★★ Comparison with MindIE/Ascend:")
    print("    → MindIE: npu_dequant_swiglu_quant = AscendC kernel on AI Core Vector Unit")
    print("    → Triton: dequant_swiglu_quant_kernel = Triton JIT kernel on CUDA SM")
    print("    → Both: intermediate data stays on-chip → no HBM round-trip")
    print("    → Key difference: Ascend Cube+Vector pipelining vs CUDA SM unified pipeline")
    print("    → Triton advantage: open-source, JIT, portable across SM versions")
    print()
    print("  ★★★★★★★★ Contribution potential:")
    print("    → vLLM Triton MoE backend: dequant+SwiGLU+quant Triton kernel → portable")
    print("    → SGLang Triton backend: already uses Triton for MoE → natural fit")
    print("    → Priority: P6-P7 (after Inductor Fusion Guard P9 merged)")
    print("    → Target: vLLM W8A8 MoE Triton path for SM89 decode")
    print()
    print("  ★★★★★★★★ Key design choices:")
    print("    → BLOCK_M=tl.constexpr → deterministic → batch-invariant → SGLang-compatible!")
    print("    → BLOCK_N=tl.constexpr → deterministic → consistent across all batch sizes")
    print("    → QUANT_MODE=tl.constexpr → compile-time selection → no runtime branch")
    print("    → ACTIVATE_LEFT=tl.constexpr → compile-time → matches model convention")
    print("    → Per-token dynamic quant (quant_mode=1) → same as MindIE production path")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="★★★★★★★★★ Triton dequant_swiglu_quant prototype for SM89"
    )
    parser.add_argument(
        "--mode", choices=["benchmark", "validate", "correctness", "info"],
        default="info",
        help="benchmark=fused vs unfused timing, validate=correctness check, "
             "correctness=detailed correctness, info=design info (no GPU needed)"
    )
    args = parser.parse_args()

    print("★★★★★★★★★ Triton dequant_swiglu_quant Prototype ★★★★★★★★★★")
    print("  Port of MindIE npu_dequant_swiglu_quant to CUDA/SM89 via Triton")
    print()

    if args.mode == "info":
        print_design_info()
        return

    if not HAS_TRITON:
        print("  ✗ Triton not installed — cannot run kernel. Use --mode info for design.")
        return

    if not torch.cuda.is_available():
        print("  ✗ No CUDA GPU available. Use --mode info for design details.")
        print("  GPU validation deferred — see rtx4090_gpu_experiment_runner.py P6")
        return

    # Check SM capability
    gpu_cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name} (SM{gpu_cap[0]}.{gpu_cap[1]})")
    if gpu_cap[0] < 9:
        print("  ★★★★★★★★ SM<90 — Triton fusion guard applies! (see Inductor Fusion Guard PR)")
    print()

    if args.mode == "benchmark":
        run_benchmark()
    elif args.mode in ["validate", "correctness"]:
        result = run_correctness_test()
        if result:
            print("  ★★★★★★★★ All correctness tests PASSED!")
        else:
            print("  ✗ Some correctness tests FAILED — review kernel implementation")


if __name__ == "__main__":
    main()
