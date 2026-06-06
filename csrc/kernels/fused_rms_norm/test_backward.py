#!/usr/bin/env python3
"""Fused RMSNorm Backward Pass Verification + Benchmark
====================================================

1. Verify Python fallback backward is correct vs torch.autograd
2. Implement CUDA C++ backward kernel
3. Benchmark CUDA backward vs Python autograd chain

Runs on RTX 4090 with PyTorch 2.9+cu128.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, '/root/rollout-infra/csrc/kernels/fused_rms_norm')

import torch
import time
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)


# ============================================================
# 1. Verify Python backward correctness vs torch.autograd
# ============================================================

def reference_rms_norm_add(input, residual, weight, eps):
    """Reference: PyTorch chain of ops (for autograd comparison)"""
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    x_norm = input * inv_rms
    return x_norm * weight + residual


def test_backward_correctness():
    print("\n1. Backward Correctness: Python fallback vs torch.autograd")

    torch.manual_seed(42)
    configs = [
        (4, 2048, "FP32_B4_H2048"),
        (32, 2048, "FP32_B32_H2048"),
        (128, 2048, "FP32_B128_H2048"),
        (4, 8192, "FP32_B4_H8192"),
    ]

    all_pass = True
    for B, H, name in configs:
        eps = 1e-6

        # Reference (PyTorch autograd)
        x_ref = torch.randn(B, H, device='cuda', requires_grad=True)
        r_ref = torch.randn(B, H, device='cuda', requires_grad=True)
        w_ref = torch.randn(H, device='cuda', requires_grad=True)
        y_ref = reference_rms_norm_add(x_ref, r_ref, w_ref, eps)
        dy = torch.randn_like(y_ref)
        y_ref.backward(dy)

        # Our Python fallback
        from fused_rms_norm_python import FusedRMSNormAddFunction
        x_ours = x_ref.clone().detach().requires_grad_(True)
        r_ours = r_ref.clone().detach().requires_grad_(True)
        w_ours = w_ref.clone().detach().requires_grad_(True)
        y_ours = FusedRMSNormAddFunction.apply(x_ours, r_ours, w_ours, eps)
        y_ours.backward(dy.clone())

        # Compare gradients
        dx_diff = (x_ref.grad - x_ours.grad).abs().max().item()
        dr_diff = (r_ref.grad - r_ours.grad).abs().max().item()
        dw_diff = (w_ref.grad - w_ours.grad).abs().max().item()

        # Cosine similarity
        dx_cos = torch.nn.functional.cosine_similarity(
            x_ref.grad.flatten(), x_ours.grad.flatten(), dim=0).item()

        passed = dx_diff < 1e-4 and dr_diff < 1e-4 and dw_diff < 1e-3 and dx_cos > 0.9999

        print(f"  {name}: dx_diff={dx_diff:.2e} dr_diff={dr_diff:.2e} "
              f"dw_diff={dw_diff:.2e} dx_cos={dx_cos:.6f} "
              f"{'PASS' if passed else 'FAIL'}")

        all_pass = all_pass and passed

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return all_pass


# ============================================================
# 2. Benchmark: Python fallback backward vs PyTorch autograd chain
# ============================================================

def bench_backward():
    print("\n2. Backward Performance Benchmark: Python vs PyTorch chain")

    torch.manual_seed(42)
    eps = 1e-6

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

    results = []
    for B, H in [(32, 2048), (128, 2048), (32, 8192), (128, 8192)]:
        from fused_rms_norm_python import fused_rms_norm_add

        # PyTorch autograd chain (forward + backward)
        def pytorch_fwd_bwd():
            x = torch.randn(B, H, device='cuda', requires_grad=True)
            r = torch.randn(B, H, device='cuda')
            w = torch.randn(H, device='cuda', requires_grad=True)
            y = reference_rms_norm_add(x, r, w, eps)
            y.backward(torch.randn_like(y))

        # Python fallback (forward + backward)
        def fallback_fwd_bwd():
            x = torch.randn(B, H, device='cuda', requires_grad=True)
            r = torch.randn(B, H, device='cuda')
            w = torch.randn(H, device='cuda', requires_grad=True)
            y = fused_rms_norm_add(x, r, w, eps)
            y.backward(torch.randn_like(y))

        pt_ms = bench_ms(pytorch_fwd_bwd)
        fb_ms = bench_ms(fallback_fwd_bwd)
        speedup = pt_ms / fb_ms

        print(f"  B={B} H={H}: PyTorch={pt_ms:.3f}ms Fallback={fb_ms:.3f}ms "
              f"Speedup={speedup:.2f}x")
        results.append({
            "B": B, "H": H,
            "pytorch_ms": round(pt_ms, 3),
            "fallback_ms": round(fb_ms, 3),
            "speedup": round(speedup, 2),
        })

    return results


# ============================================================
# Run
# ============================================================

correctness_pass = test_backward_correctness()
perf_results = bench_backward()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Backward correctness: {'PASS' if correctness_pass else 'FAIL'}")
print("Next step: Implement CUDA C++ backward kernel for further speedup")

# Save results
out_path = os.path.join('/root/rollout-infra/csrc/kernels/fused_rms_norm',
                        'backward_test_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "correctness_pass": correctness_pass,
        "perf_results": perf_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")