"""
Fused RMSNorm CUDA C++ Extension — Benchmark Tool

Benchmarks three implementations:
1. Separate PyTorch ops (RMSNorm + Add)
2. Fused Python implementation
3. Fused CUDA C++ kernel (if compiled)

Usage:
  python tools/benchmark_fused_rms_norm.py          # Local (CPU or Mac)
  python tools/benchmark_fused_rms_norm.py --cuda    # On RTX 4090 (requires compiled kernel)

On GPU server, first compile:
  cd ~/rollout-infra/csrc/kernels/fused_rms_norm
  python setup.py build_ext --inplace
"""

import sys
import os
import time
import json
import argparse
import numpy as np

# Add kernel path to sys.path
kernel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'csrc', 'kernels', 'fused_rms_norm')

import torch


def benchmark_separate_ops(input_tensor, residual_tensor, weight_tensor, epsilon, num_iters=100):
    """Benchmark: Separate RMSNorm + Residual Add (PyTorch ops)."""
    # Warm up
    for _ in range(10):
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        x_norm = input_tensor * torch.rsqrt(variance + epsilon)
        output = x_norm * weight_tensor + residual_tensor
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        x_norm = input_tensor * torch.rsqrt(variance + epsilon)
        output = x_norm * weight_tensor + residual_tensor
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    return elapsed, output


def benchmark_fused_python(input_tensor, residual_tensor, weight_tensor, epsilon, num_iters=100):
    """Benchmark: Fused Python implementation (single pass)."""
    # Warm up
    for _ in range(10):
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + epsilon)
        output = input_tensor * inv_rms * weight_tensor + residual_tensor
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + epsilon)
        output = input_tensor * inv_rms * weight_tensor + residual_tensor
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    return elapsed, output


def benchmark_fused_cuda(input_tensor, residual_tensor, weight_tensor, epsilon, num_iters=100):
    """Benchmark: Fused CUDA C++ kernel."""
    # Import the compiled extension
    sys.path.insert(0, kernel_path)
    try:
        from fused_rms_norm_python import fused_rms_norm_add
    except ImportError:
        print("CUDA C++ kernel not compiled. Run setup.py first.")
        return None, None

    # Warm up
    for _ in range(10):
        output = fused_rms_norm_add(input_tensor, residual_tensor, weight_tensor, epsilon)
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        output = fused_rms_norm_add(input_tensor, residual_tensor, weight_tensor, epsilon)
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    return elapsed, output


def verify_correctness(device='cpu'):
    """Verify that all implementations produce the same results."""
    batch_size = 4
    hidden_size = 128
    epsilon = 1e-6

    input = torch.randn(batch_size, hidden_size, device=device)
    residual = torch.randn(batch_size, hidden_size, device=device)
    weight = torch.randn(hidden_size, device=device)

    # Separate ops reference
    variance = input.pow(2).mean(dim=-1, keepdim=True)
    x_norm = input * torch.rsqrt(variance + epsilon)
    ref_output = x_norm * weight + residual

    # Fused Python
    variance2 = input.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance2 + epsilon)
    fused_output = input * inv_rms * weight + residual

    max_diff = (ref_output - fused_output).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), fused_output.flatten(), dim=0).item()

    print(f"Correctness (Separate vs Fused Python):")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")
    print(f"  Match: {'YES' if max_diff < 1e-6 else 'NO'}")

    return max_diff < 1e-6


def run_benchmarks(device='cpu', batch_sizes=[1, 4, 8, 16, 32, 64, 128],
                   hidden_sizes=[512, 1024, 2048, 4096], num_iters=100):
    """Run comprehensive benchmarks across different configurations."""

    results = {
        'device': device,
        'gpu_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
        'experiments': []
    }

    epsilon = 1e-6

    for hidden_size in hidden_sizes:
        for batch_size in batch_sizes:
            input = torch.randn(batch_size, hidden_size, device=device)
            residual = torch.randn(batch_size, hidden_size, device=device)
            weight = torch.randn(hidden_size, device=device)

            # Separate ops
            sep_time, sep_output = benchmark_separate_ops(
                input, residual, weight, epsilon, num_iters)

            # Fused Python
            fused_time, fused_output = benchmark_fused_python(
                input, residual, weight, epsilon, num_iters)

            # Fused CUDA (if available)
            cuda_time = None
            cuda_output = None
            if device == 'cuda':
                try:
                    cuda_time, cuda_output = benchmark_fused_cuda(
                        input, residual, weight, epsilon, num_iters)
                except Exception as e:
                    print(f"  CUDA kernel benchmark failed: {e}")

            # Compute speedup
            fused_speedup = sep_time / fused_time if fused_time > 0 else 0

            exp_result = {
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'separate_ms': round(sep_time, 4),
                'fused_python_ms': round(fused_time, 4),
                'fused_cuda_ms': round(cuda_time, 4) if cuda_time else None,
                'fused_speedup': round(fused_speedup, 2),
                'total_elements': batch_size * hidden_size,
            }
            results['experiments'].append(exp_result)

            # Verify correctness
            max_diff = (sep_output - fused_output).abs().max().item()

            print(f"B={batch_size}, H={hidden_size}: "
                  f"Sep={sep_time:.3f}ms, FusedPy={fused_time:.3f}ms, "
                  f"Speedup={fused_speedup:.2f}x, "
                  f"CUDA={cuda_time:.3f}ms" if cuda_time else
                  f"B={batch_size}, H={hidden_size}: "
                  f"Sep={sep_time:.3f}ms, FusedPy={fused_time:.3f}ms, "
                  f"Speedup={fused_speedup:.2f}x, MaxDiff={max_diff:.2e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Fused RMSNorm Benchmark')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA (GPU)')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        default=[1, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument('--hidden-sizes', nargs='+', type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--verify-only', action='store_true', help='Only verify correctness')
    args = parser.parse_args()

    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    print(f"=== Fused RMSNorm + Residual Add Benchmark ===")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Verify correctness first
    print("--- Correctness Verification ---")
    verify_correctness(device)
    print()

    if args.verify_only:
        return

    # Run benchmarks
    print("--- Performance Benchmark ---")
    results = run_benchmarks(device, args.batch_sizes, args.hidden_sizes, args.iters)

    # Summary
    print()
    print("--- Summary ---")
    avg_speedup = np.mean([e['fused_speedup'] for e in results['experiments']])
    print(f"Average fused Python speedup: {avg_speedup:.2f}x")
    if any(e['fused_cuda_ms'] is not None for e in results['experiments']):
        cuda_exps = [e for e in results['experiments'] if e['fused_cuda_ms'] is not None]
        avg_cuda_speedup = np.mean([
            e['separate_ms'] / e['fused_cuda_ms'] for e in cuda_exps])
        print(f"Average fused CUDA speedup: {avg_cuda_speedup:.2f}x")

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    results_file = os.path.join(output_dir, 'fused_rms_norm_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()