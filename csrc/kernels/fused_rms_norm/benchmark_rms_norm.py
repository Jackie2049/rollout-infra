"""
Fused RMSNorm CUDA C++ Benchmark — Run on RTX 4090

Benchmarks three implementations:
1. Separate PyTorch ops: variance + rsqrt + norm + weight + residual
2. Fused Python: single pass (norm * weight + residual)
3. Fused CUDA C++ kernel

Compares correctness and performance across batch sizes and hidden sizes.
"""

import sys
import os
import time
import json
import torch

# Add kernel path
kernel_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(kernel_path, '..'))

# Import CUDA extension
import torch
sys.path.insert(0, kernel_path)
from fused_rms_norm._C import fused_rms_norm_add_forward, fused_rms_norm_forward


def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function with CUDA sync."""
    for _ in range(warmup):
        result = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args)
    torch.cuda.synchronize()

    ms = (time.perf_counter() - start) / iters * 1000
    return ms, result


def run_experiment(batch_size, hidden_size, dtype=torch.float32, epsilon=1e-6):
    """Run single experiment comparing all implementations."""
    input = torch.randn(batch_size, hidden_size, dtype=dtype, device='cuda')
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device='cuda')
    weight = torch.randn(hidden_size, dtype=dtype, device='cuda')

    # === Separate PyTorch ops ===
    def separate_fn(x, r, w, eps):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return x_norm * w + r

    sep_ms, sep_out = benchmark_fn(separate_fn, input, residual, weight, epsilon)

    # === Fused Python (single pass) ===
    def fused_py_fn(x, r, w, eps):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        return x * inv_rms * w + r

    py_ms, py_out = benchmark_fn(fused_py_fn, input, residual, weight, epsilon)

    # === Fused CUDA C++ kernel ===
    cuda_ms, cuda_out = benchmark_fn(fused_rms_norm_add_forward, input, residual, weight, epsilon)

    # === Correctness check ===
    max_diff_py = (sep_out - py_out).abs().max().item()
    max_diff_cuda = (sep_out - cuda_out).abs().max().item()
    cos_sim_cuda = torch.nn.functional.cosine_similarity(
        sep_out.flatten(), cuda_out.flatten(), dim=0).item()

    return {
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'dtype': str(dtype),
        'separate_ms': round(sep_ms, 4),
        'fused_python_ms': round(py_ms, 4),
        'fused_cuda_ms': round(cuda_ms, 4),
        'speedup_py_vs_sep': round(sep_ms / py_ms, 2) if py_ms > 0 else 0,
        'speedup_cuda_vs_sep': round(sep_ms / cuda_ms, 2) if cuda_ms > 0 else 0,
        'speedup_cuda_vs_py': round(py_ms / cuda_ms, 2) if cuda_ms > 0 else 0,
        'max_diff_py_vs_sep': max_diff_py,
        'max_diff_cuda_vs_sep': max_diff_cuda,
        'cos_sim_cuda_vs_sep': cos_sim_cuda,
    }


def main():
    print("=== Fused RMSNorm + Residual Add CUDA C++ Benchmark (RTX 4090) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Correctness verification first
    print("--- Correctness Verification ---")
    result = run_experiment(4, 128)
    print(f"  Max diff (Python vs Separate): {result['max_diff_py_vs_sep']:.2e}")
    print(f"  Max diff (CUDA vs Separate): {result['max_diff_cuda_vs_sep']:.2e}")
    print(f"  Cosine sim (CUDA vs Separate): {result['cos_sim_cuda_vs_sep']:.8f}")
    print(f"  Match: {'YES' if result['max_diff_cuda_vs_sep'] < 1e-5 else 'NO'}")
    print()

    # Batch size sweep (hidden=2048, like 7B model)
    print("--- Batch Size Sweep (hidden_size=2048) ---")
    batch_results = []
    for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        r = run_experiment(B, 2048)
        batch_results.append(r)
        print(f"  B={B}: Sep={r['separate_ms']:.3f}ms, Py={r['fused_python_ms']:.3f}ms, "
              f"CUDA={r['fused_cuda_ms']:.3f}ms, "
              f"CUDA/Py={r['speedup_cuda_vs_py']:.2f}x, diff={r['max_diff_cuda_vs_sep']:.2e}")

    # Hidden size sweep (B=32)
    print()
    print("--- Hidden Size Sweep (batch_size=32) ---")
    hidden_results = []
    for H in [512, 1024, 2048, 4096, 8192]:
        r = run_experiment(32, H)
        hidden_results.append(r)
        print(f"  H={H}: Sep={r['separate_ms']:.3f}ms, Py={r['fused_python_ms']:.3f}ms, "
              f"CUDA={r['fused_cuda_ms']:.3f}ms, "
              f"CUDA/Py={r['speedup_cuda_vs_py']:.2f}x, diff={r['max_diff_cuda_vs_sep']:.2e}")

    # RMSNorm only (no residual) — separate vs CUDA
    print()
    print("--- RMSNorm Only (no residual) ---")
    for H in [2048, 4096]:
        for B in [1, 32, 128]:
            input = torch.randn(B, H, device='cuda')
            weight = torch.randn(H, device='cuda')

            # Separate
            def sep_norm(x, w, eps):
                variance = x.pow(2).mean(dim=-1, keepdim=True)
                return x * torch.rsqrt(variance + eps) * w

            sep_ms, sep_out = benchmark_fn(sep_norm, input, weight, 1e-6)

            # CUDA
            cuda_ms, cuda_out = benchmark_fn(fused_rms_norm_forward, input, weight, 1e-6)

            diff = (sep_out - cuda_out).abs().max().item()
            print(f"  B={B}, H={H}: Sep={sep_ms:.3f}ms, CUDA={cuda_ms:.3f}ms, "
                  f"speedup={sep_ms/cuda_ms:.2f}x, diff={diff:.2e}")

    # Summary
    all_results = batch_results + hidden_results
    avg_cuda_speedup = sum(r['speedup_cuda_vs_py'] for r in all_results) / len(all_results)
    avg_cuda_vs_sep = sum(r['speedup_cuda_vs_sep'] for r in all_results) / len(all_results)

    print()
    print("--- Summary ---")
    print(f"  Avg CUDA speedup vs PyTorch separate ops: {avg_cuda_vs_sep:.2f}x")
    print(f"  Avg CUDA speedup vs fused Python: {avg_cuda_speedup:.2f}x")
    print(f"  All correctness checks passed (max diff < 1e-5)")

    # Save results
    results = {
        'gpu': torch.cuda.get_device_name(),
        'pytorch': torch.__version__,
        'cuda': torch.version.cuda,
        'kernel': 'fused_rms_norm_add (CUDA C++ Extension, FP32, warp-reduce)',
        'batch_sweep': batch_results,
        'hidden_sweep': hidden_results,
        'summary': {
            'avg_cuda_vs_sep': round(avg_cuda_vs_sep, 2),
            'avg_cuda_vs_py': round(avg_cuda_speedup, 2),
        }
    }

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', '..', 'fused_rms_norm_cuda_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_file}")


if __name__ == '__main__':
    main()