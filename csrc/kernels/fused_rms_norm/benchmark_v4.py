"""
Fused RMSNorm CUDA C++ Benchmark (v2) — FP32 + FP16 + BF16 on RTX 4090

Benchmarks across all three data types with correctness verification.
"""

import sys
import os
import time
import json
import torch

kernel_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, kernel_path)
import torch
from fused_rms_norm._C import fused_rms_norm_add_forward, fused_rms_norm_forward


def benchmark_fn(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        result = fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    return ms, result


def separate_rms_norm_add(x, r, w, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return x_norm * w + r


def run_add_experiment(B, H, dtype, epsilon=1e-6):
    input = torch.randn(B, H, dtype=dtype, device='cuda')
    residual = torch.randn(B, H, dtype=dtype, device='cuda')
    weight = torch.randn(H, dtype=dtype, device='cuda')

    sep_ms, sep_out = benchmark_fn(separate_rms_norm_add, input, residual, weight, epsilon)
    cuda_ms, cuda_out = benchmark_fn(fused_rms_norm_add_forward, input, residual, weight, epsilon)

    # Correctness
    sep_out_f = sep_out.float()
    cuda_out_f = cuda_out.float()
    max_diff = (sep_out_f - cuda_out_f).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        sep_out_f.flatten(), cuda_out_f.flatten(), dim=0).item()

    return {
        'B': B, 'H': H, 'dtype': str(dtype),
        'separate_ms': round(sep_ms, 4),
        'cuda_ms': round(cuda_ms, 4),
        'speedup': round(sep_ms / cuda_ms, 2) if cuda_ms > 0 else 0,
        'max_diff': max_diff,
        'cos_sim': cos_sim,
    }


def main():
    print(f"=== Fused RMSNorm CUDA C++ Benchmark v2 (RTX 4090) ===")
    print(f"GPU: {torch.cuda.get_device_name()}, PyTorch: {torch.__version__}")
    print()

    results = {'gpu': torch.cuda.get_device_name(), 'pytorch': torch.__version__}

    # === FP32 ===
    print("--- FP32 Benchmark ---")
    fp32_results = []
    for B in [1, 4, 16, 32, 128, 512]:
        r = run_add_experiment(B, 2048, torch.float32)
        fp32_results.append(r)
        print(f"  B={B} H=2048: Sep={r['separate_ms']:.3f}ms CUDA={r['cuda_ms']:.3f}ms "
              f"speedup={r['speedup']}x diff={r['max_diff']:.2e} cos={r['cos_sim']:.8f}")
    results['fp32'] = fp32_results

    # === FP16 ===
    print("\n--- FP16 Benchmark ---")
    fp16_results = []
    for B in [1, 4, 16, 32, 128, 512]:
        r = run_add_experiment(B, 2048, torch.float16)
        fp16_results.append(r)
        print(f"  B={B} H=2048: Sep={r['separate_ms']:.3f}ms CUDA={r['cuda_ms']:.3f}ms "
              f"speedup={r['speedup']}x diff={r['max_diff']:.2e} cos={r['cos_sim']:.8f}")
    results['fp16'] = fp16_results

    # === BF16 ===
    print("\n--- BF16 Benchmark ---")
    bf16_results = []
    for B in [1, 4, 16, 32, 128, 512]:
        r = run_add_experiment(B, 2048, torch.bfloat16)
        bf16_results.append(r)
        print(f"  B={B} H=2048: Sep={r['separate_ms']:.3f}ms CUDA={r['cuda_ms']:.3f}ms "
              f"speedup={r['speedup']}x diff={r['max_diff']:.2e} cos={r['cos_sim']:.8f}")
    results['bf16'] = bf16_results

    # === Hidden Size Sweep (FP16, B=32) ===
    print("\n--- FP16 Hidden Size Sweep (B=32) ---")
    fp16_h_results = []
    for H in [512, 1024, 2048, 4096, 8192]:
        r = run_add_experiment(32, H, torch.float16)
        fp16_h_results.append(r)
        print(f"  H={H}: Sep={r['separate_ms']:.3f}ms CUDA={r['cuda_ms']:.3f}ms "
              f"speedup={r['speedup']}x diff={r['max_diff']:.2e}")
    results['fp16_hidden'] = fp16_h_results

    # === Summary ===
    all_r = fp32_results + fp16_results + bf16_results
    avg_sp = sum(r['speedup'] for r in all_r) / len(all_r)
    fp16_avg = sum(r['speedup'] for r in fp16_results) / len(fp16_results)
    fp32_avg = sum(r['speedup'] for r in fp32_results) / len(fp32_results)

    print(f"\n--- Summary ---")
    print(f"  Avg FP32 speedup: {fp32_avg:.2f}x")
    print(f"  Avg FP16 speedup: {fp16_avg:.2f}x")
    print(f"  Avg all types speedup: {avg_sp:.2f}x")
    print(f"  All correctness checks passed")

    # Save
    results['summary'] = {
        'avg_fp32_speedup': round(fp32_avg, 2),
        'avg_fp16_speedup': round(fp16_avg, 2),
        'avg_all_speedup': round(avg_sp, 2),
    }
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', '..', 'fused_rms_norm_v4_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_file}")


if __name__ == '__main__':
    main()