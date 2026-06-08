"""
CUTLASS GEMM Benchmark — RTX 4090 (SM89)

Uses PyTorch for baseline BF16/FP16 GEMM timing and theoretical analysis.
Tests GEMM sizes relevant to LLM inference (7B model dimensions).
"""

import os, sys, json, numpy as np, torch, time

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")
print(f"HBM: {props.total_memory/1e9:.2f} GB, MPs: {props.multi_processor_count}")

# RTX 4090 peak specs
fp16_peak_tflops = 82.58  # FP16/BF16 peak without sparse
fp8_peak_tflops = 165.16  # FP8 E4M3 peak (2x FP16)
tf32_peak_tflops = 82.58  # TF32 peak same as FP16
int8_peak_tflops = 165.16  # INT8 peak same as FP8

def benchmark_fn(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        latencies.append(s.elapsed_time(e))
    return np.mean(latencies), np.std(latencies)

# =====================================================================
# SECTION 1: BF16 GEMM Benchmark — LLM-relevant sizes
# =====================================================================
print("\n" + "="*70)
print("SECTION 1: BF16 GEMM Benchmark (LLM-relevant sizes)")
print("="*70)

# 7B model typical dimensions
gemm_configs = [
    ("gate_proj_decode_B1", 1, 10240, 2560),
    ("gate_proj_decode_B4", 4, 10240, 2560),
    ("gate_proj_decode_B8", 8, 10240, 2560),
    ("gate_proj_decode_B16", 16, 10240, 2560),
    ("gate_proj_decode_B32", 32, 10240, 2560),
    ("gate_proj_decode_B64", 64, 10240, 2560),
    ("gate_proj_decode_B128", 128, 10240, 2560),
    ("gate_proj_decode_B256", 256, 10240, 2560),
    ("out_proj_decode_B1", 1, 2560, 10240),
    ("out_proj_decode_B32", 32, 2560, 10240),
    ("out_proj_decode_B128", 128, 2560, 10240),
    ("qkv_proj_decode_B1", 1, 3072, 2560),
    ("qkv_proj_decode_B32", 32, 3072, 2560),
    ("prefill_S128", 128, 10240, 2560),
    ("prefill_S512", 512, 10240, 2560),
    ("prefill_S1024", 1024, 10240, 2560),
    ("prefill_S2048", 2048, 10240, 2560),
    ("square_1024", 1024, 1024, 1024),
    ("square_2048", 2048, 2048, 2048),
    ("square_4096", 4096, 4096, 4096),
    ("square_8192", 8192, 8192, 8192),
]

results_bf16 = []
for name, M, N, K in gemm_configs:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    def gemm_fn():
        return torch.matmul(A, B)

    ms, std = benchmark_fn(gemm_fn, warmup=3, iters=20)

    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    arith_intensity = flops / (2 * (M*K + K*N + M*N))
    is_compute_bound = arith_intensity > 182

    results_bf16.append({
        "name": name, "M": M, "N": N, "K": K,
        "ms": round(ms, 4), "std_ms": round(std, 4),
        "TFLOPS": round(tflops, 2), "peak_pct": round(tflops/fp16_peak_tflops*100, 1),
        "arith_intensity": round(arith_intensity, 1),
        "bound": "compute" if is_compute_bound else "memory",
    })
    print(f"  {name}: M={M} N={N} K={K} | {ms:.3f}ms | {tflops:.2f} TFLOPS ({tflops/fp16_peak_tflops*100:.1f}% peak) | AI={arith_intensity:.0f} | {results_bf16[-1]['bound']}")

# =====================================================================
# SECTION 2: FP16 vs BF16 vs FP32 GEMM Comparison
# =====================================================================
print("\n" + "="*70)
print("SECTION 2: FP16 vs BF16 vs FP32 GEMM Comparison")
print("="*70)

compare_sizes = [(1024, 10240, 2560), (2048, 10240, 2560), (4096, 4096, 4096), (8192, 8192, 8192)]
results_dtype = []

for M, N, K in compare_sizes:
    for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
        torch.cuda.empty_cache()
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)

        def gemm_fn():
            return torch.matmul(A, B)

        ms, std = benchmark_fn(gemm_fn, warmup=3, iters=20)
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12

        peak = fp16_peak_tflops if dtype_name in ["FP16", "BF16"] else 41.29
        results_dtype.append({
            "M": M, "N": N, "K": K, "dtype": dtype_name,
            "ms": round(ms, 4), "TFLOPS": round(tflops, 2),
            "peak_pct": round(tflops/peak*100, 1),
        })
        print(f"  M={M} N={N} K={K} {dtype_name}: {ms:.3f}ms {tflops:.2f} TFLOPS ({tflops/peak*100:.1f}% peak)")

# =====================================================================
# SECTION 3: cuBLAS Batched GEMM (simulates MoE expert parallel)
# =====================================================================
print("\n" + "="*70)
print("SECTION 3: Batched GEMM (MoE-style)")
print("="*70)

results_batched = []
for num_experts in [4, 8, 16, 32, 64]:
    M, N, K = 8, 10240, 2560
    torch.cuda.empty_cache()

    A_batch = torch.randn(num_experts, M, K, dtype=torch.bfloat16, device=device)
    B_batch = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)

    def batched_gemm():
        return torch.bmm(A_batch, B_batch)

    ms, std = benchmark_fn(batched_gemm, warmup=3, iters=20)

    total_flops = 2.0 * num_experts * M * N * K
    tflops = total_flops / (ms * 1e-3) / 1e12

    results_batched.append({
        "num_experts": num_experts, "M": M, "N": N, "K": K,
        "ms": round(ms, 4), "TFLOPS": round(tflops, 2),
        "peak_pct": round(tflops/fp16_peak_tflops*100, 1),
        "total_flops_G": round(total_flops/1e9, 2),
    })
    print(f"  experts={num_experts}: {ms:.3f}ms {tflops:.2f} TFLOPS ({tflops/fp16_peak_tflops*100:.1f}% peak) total_flops={total_flops/1e9:.2f}G")

# =====================================================================
# Save Results
# =====================================================================
all_results = {
    "device": {"name": props.name, "sm": f"{props.major}.{props.minor}", "hbm_GB": round(props.total_memory/1e9,2)},
    "peak_specs": {"fp16_TFLOPS": fp16_peak_tflops, "fp8_TFLOPS": fp8_peak_tflops, "tf32_TFLOPS": tf32_peak_tflops, "int8_TFLOPS": int8_peak_tflops},
    "bf16_gemm_llm": results_bf16,
    "dtype_comparison": results_dtype,
    "batched_gemm_moe": results_batched,
}

os.makedirs("results", exist_ok=True)
with open("results/cutlass_gemm_benchmark.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to results/cutlass_gemm_benchmark.json")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nBF16 GEMM (LLM sizes) — Compute vs Memory Bound:")
for r in results_bf16:
    print(f"  {r['name']}: {r['TFLOPS']} TFLOPS ({r['peak_pct']}% peak) AI={r['arith_intensity']} -> {r['bound']}")

print("\nDtype Comparison:")
for r in results_dtype:
    print(f"  M={r['M']} {r['dtype']}: {r['TFLOPS']} TFLOPS ({r['peak_pct']}% peak)")

print("\nBatched GEMM (MoE):")
for r in results_batched:
    print(f"  {r['num_experts']} experts: {r['TFLOPS']} TFLOPS ({r['peak_pct']}% peak)")
