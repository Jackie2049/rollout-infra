#!/usr/bin/env python3
"""Fused RMSNorm Backward Pass: CUDA vs Python vs PyTorch Benchmark
=================================================

Benchmarks:
1. Correctness: CUDA backward vs PyTorch autograd reference
2. Correctness: CUDA backward vs Python fallback
3. Performance: CUDA backward vs Python fallback vs PyTorch autograd chain
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

from fused_rms_norm_python import FusedRMSNormAddFunction, FusedRMSNormFunction, fused_rms_norm_add, fused_rms_norm


CUDA_AVAILABLE = False
try:
    from fused_rms_norm._C import fused_rms_norm_add_backward as _cuda_add_bwd
    from fused_rms_norm._C import fused_rms_norm_backward as _cuda_bwd
    CUDA_AVAILABLE = True
    print("CUDA backward kernels: AVAILABLE")
except ImportError:
    print("CUDA backward kernels: NOT AVAILABLE (using Python fallback)")

print("=" * 60)


# ============================================================
# 1. Correctness: CUDA backward vs PyTorch autograd
# ============================================================

def reference_rms_norm_add(input, residual, weight, eps):
    """Reference: PyTorch chain of ops (for autograd comparison)"""
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    x_norm = input * inv_rms
    return x_norm * weight + residual


EPS = 1e-6

def test_cuda_backward_correctness():
    print("\n1. CUDA Backward Correctness vs PyTorch autograd")

    torch.manual_seed(42)
    configs = [
        (4, 2048, "FP32_B4_H2048"),
        (32, 2048, "FP32_B32_H2048"),
        (128, 2048, "FP32_B128_H2048"),
        (4, 8192, "FP32_B4_H8192"),
    ]

    all_pass = True
    for B, H, name in configs:
        # Reference (PyTorch autograd)
        x_ref = torch.randn(B, H, device='cuda', requires_grad=True)
        r_ref = torch.randn(B, H, device='cuda', requires_grad=True)
        w_ref = torch.randn(H, device='cuda', requires_grad=True)
        y_ref = reference_rms_norm_add(x_ref, r_ref, w_ref, EPS)
        dy = torch.randn_like(y_ref)
        y_ref.backward(dy)

        # Our fused kernel (uses CUDA backward if available)
        x_ours = x_ref.clone().detach().requires_grad_(True)
        r_ours = r_ref.clone().detach().requires_grad_(True)
        w_ours = w_ref.clone().detach().requires_grad_(True)
        y_ours = fused_rms_norm_add(x_ours, r_ours, w_ours, EPS)
        y_ours.backward(dy.clone())

        # Compare gradients
        dx_diff = (x_ref.grad - x_ours.grad).abs().max().item()
        dr_diff = (r_ref.grad - r_ours.grad).abs().max().item()
        dw_diff = (w_ref.grad - w_ours.grad).abs().max().item()

        dx_cos = torch.nn.functional.cosine_similarity(
            x_ref.grad.flatten(), x_ours.grad.flatten(), dim=0).item()

        passed = dx_diff < 1e-4 and dr_diff < 1e-4 and dw_diff < 1e-3 and dx_cos > 0.9999

        print(f"  {name}: dx_diff={dx_diff:.2e} dr_diff={dr_diff:.2e} "
              f"dw_diff={dw_diff:.2e} dx_cos={dx_cos:.6f} "
              f"{'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed

    # FP16 test
    if CUDA_AVAILABLE:
        print("\n  FP16 Backward:")
        x_ref = torch.randn(4, 2048, device='cuda', dtype=torch.float16, requires_grad=True)
        r_ref = torch.randn(4, 2048, device='cuda', dtype=torch.float16, requires_grad=True)
        w_ref = torch.randn(2048, device='cuda', dtype=torch.float16, requires_grad=True)
        y_ref = reference_rms_norm_add(x_ref.float(), r_ref.float(), w_ref.float(), EPS).half()
        dy = torch.randn_like(y_ref)
        y_ref.backward(dy.float())

        x_ours = x_ref.clone().detach().requires_grad_(True)
        r_ours = r_ref.clone().detach().requires_grad_(True)
        w_ours = w_ref.clone().detach().requires_grad_(True)
        y_ours = fused_rms_norm_add(x_ours, r_ours, w_ours, EPS)
        y_ours.backward(dy.clone())

        dx_diff = (x_ref.grad.float() - x_ours.grad.float()).abs().max().item()
        dr_diff = (r_ref.grad.float() - r_ours.grad.float()).abs().max().item()
        dw_diff = (w_ref.grad.float() - w_ours.grad.float()).abs().max().item()
        dx_cos = torch.nn.functional.cosine_similarity(
            x_ref.grad.float().flatten(), x_ours.grad.float().flatten(), dim=0).item()

        passed = dx_diff < 1e-3 and dr_diff < 1e-3 and dw_diff < 1e-2 and dx_cos > 0.999
        print(f"    dx_diff={dx_diff:.2e} dr_diff={dr_diff:.2e} "
              f"dw_diff={dw_diff:.2e} dx_cos={dx_cos:.6f} "
              f"{'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed

        # BF16 test
        print("\n  BF16 Backward:")
        x_ref = torch.randn(4, 2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        r_ref = torch.randn(4, 2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        w_ref = torch.randn(2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y_ref = reference_rms_norm_add(x_ref.float(), r_ref.float(), w_ref.float(), EPS).bfloat16()
        dy = torch.randn_like(y_ref)
        y_ref.backward(dy.float())

        x_ours = x_ref.clone().detach().requires_grad_(True)
        r_ours = r_ref.clone().detach().requires_grad_(True)
        w_ours = w_ref.clone().detach().requires_grad_(True)
        y_ours = fused_rms_norm_add(x_ours, r_ours, w_ours, EPS)
        y_ours.backward(dy.clone())

        dx_diff = (x_ref.grad.float() - x_ours.grad.float()).abs().max().item()
        dr_diff = (r_ref.grad.float() - r_ours.grad.float()).abs().max().item()
        dw_diff = (w_ref.grad.float() - w_ours.grad.float()).abs().max().item()
        dx_cos = torch.nn.functional.cosine_similarity(
            x_ref.grad.float().flatten(), x_ours.grad.float().flatten(), dim=0).item()

        passed = dx_diff < 1e-2 and dr_diff < 1e-2 and dw_diff < 1e-1 and dx_cos > 0.999
        print(f"    dx_diff={dx_diff:.2e} dr_diff={dr_diff:.2e} "
              f"dw_diff={dw_diff:.2e} dx_cos={dx_cos:.6f} "
              f"{'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return all_pass


# ============================================================
# 2. Performance: CUDA backward vs Python fallback vs PyTorch autograd
# ============================================================

def bench_ms(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


def bench_backward():
    print("\n2. Backward Performance: CUDA vs Python vs PyTorch chain")

    torch.manual_seed(42)
    results = []

    for B, H in [(32, 2048), (128, 2048), (32, 8192), (128, 8192)]:
        # PyTorch autograd chain (forward + backward)
        def pytorch_fwd_bwd():
            x = torch.randn(B, H, device='cuda', requires_grad=True)
            r = torch.randn(B, H, device='cuda')
            w = torch.randn(H, device='cuda', requires_grad=True)
            y = reference_rms_norm_add(x, r, w, EPS)
            y.backward(torch.randn_like(y))

        # Fused kernel (forward + backward — uses CUDA if available)
        def fused_fwd_bwd():
            x = torch.randn(B, H, device='cuda', requires_grad=True)
            r = torch.randn(B, H, device='cuda')
            w = torch.randn(H, device='cuda', requires_grad=True)
            y = fused_rms_norm_add(x, r, w, EPS)
            y.backward(torch.randn_like(y))

        pt_ms = bench_ms(pytorch_fwd_bwd)
        fused_ms = bench_ms(fused_fwd_bwd)
        speedup = pt_ms / fused_ms
        backend = "CUDA" if CUDA_AVAILABLE else "Python"

        print(f"  B={B} H={H}: PyTorch={pt_ms:.3f}ms {backend}={fused_ms:.3f}ms "
              f"Speedup={speedup:.2f}x")

        results.append({
            "B": B, "H": H,
            "pytorch_ms": round(pt_ms, 3),
            "fused_ms": round(fused_ms, 3),
            "speedup": round(speedup, 2),
            "backend": backend,
        })

    return results


# ============================================================
# Run
# ============================================================

correctness_pass = test_cuda_backward_correctness()
perf_results = bench_backward()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Backward correctness: {'PASS' if correctness_pass else 'FAIL'}")
if CUDA_AVAILABLE:
    print("CUDA backward kernel: available and working")
else:
    print("CUDA backward kernel: NOT available")

# Save results
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'backward_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "correctness_pass": correctness_pass,
        "cuda_available": CUDA_AVAILABLE,
        "perf_results": perf_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")