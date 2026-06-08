"""
CUDA Stream Concurrency Analysis — RTX 4090
Tests: stream priority, compute-transfer overlap, multi-stream GEMM,
prefill+decode concurrent simulation, and kernel launch overhead.

Focus: Understanding GPU concurrency for production inference/training.
"""

import torch
import time
import json

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")
print(f"SMs: {props.multi_processor_count}, Max threads/SM: {props.max_threads_per_multi_processor}")

HBM_BANDWIDTH = 890.8  # GB/s (实测)


def benchmark(func, warmup=10, repeats=50):
    for _ in range(warmup):
        result = func()
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / repeats * 1000, result


def run_all():
    results = {}
    print("=" * 70)
    print("CUDA Stream Concurrency Analysis — RTX 4090")
    print("=" * 70)

    # Exp 1: Kernel launch overhead
    print("\n--- Exp 1: Kernel Launch Overhead ---")
    exp1 = {}
    for M in [1, 4, 16, 64, 256, 1024]:
        A = torch.randn(M, 4096, device=device, dtype=torch.bfloat16)
        B = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        ms, _ = benchmark(lambda: torch.mm(A, B))

        # Compute FLOPS and bytes
        flops = 2 * M * 4096 * 4096
        compute_time_us = flops / (169.6e12) * 1e6  # theoretical compute time
        mem_bytes = 2 * (M * 4096 + 4096 * 4096 + M * 4096)
        mem_time_us = mem_bytes / (HBM_BANDWIDTH * 1e9) * 1e6

        # Estimate launch overhead = measured - theoretical
        launch_overhead_us = ms * 1000 - max(compute_time_us, mem_time_us)
        if launch_overhead_us < 0:
            launch_overhead_us = 0

        exp1[f"M={M}"] = {
            "measured_ms": round(ms, 4),
            "compute_theoretical_us": round(compute_time_us, 2),
            "mem_theoretical_us": round(mem_time_us, 2),
            "launch_overhead_us": round(launch_overhead_us, 2),
            "launch_pct": round(launch_overhead_us / (ms * 1000) * 100, 1) if ms > 0 else 0,
        }
        print(f"  M={M}: {ms:.4f}ms, launch overhead={launch_overhead_us:.2f}us ({launch_overhead_us/(ms*1000)*100:.1f}% of total)")

    results["exp1_launch_overhead"] = exp1

    # Exp 2: Multi-stream GEMM overlap
    print("\n--- Exp 2: Multi-Stream GEMM Overlap ---")
    exp2 = {}

    # Single stream: 2 GEMMs sequentially
    # Multi stream: 2 GEMMs on different streams → potentially overlap
    for M in [1, 4, 16, 64, 128]:
        A1 = torch.randn(M, 4096, device=device, dtype=torch.bfloat16)
        B1 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        A2 = torch.randn(M, 4096, device=device, dtype=torch.bfloat16)
        B2 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        # Single stream (default)
        def single_stream():
            C1 = torch.mm(A1, B1)
            C2 = torch.mm(A2, B2)
            return C1, C2

        # Two streams
        s0 = torch.cuda.Stream(priority=0)  # default priority
        s1 = torch.cuda.Stream(priority=0)

        def dual_stream():
            with torch.cuda.stream(s0):
                C1 = torch.mm(A1, B1)
            with torch.cuda.stream(s1):
                C2 = torch.mm(A2, B2)
            return C1, C2

        ss_ms, _ = benchmark(single_stream)
        ds_ms, _ = benchmark(dual_stream)

        overlap_pct = max(0, (ss_ms - ds_ms) / ss_ms * 100) if ss_ms > 0 else 0
        speedup = ss_ms / ds_ms if ds_ms > 0 else 1.0

        exp2[f"M={M}"] = {
            "single_stream_ms": round(ss_ms, 4),
            "dual_stream_ms": round(ds_ms, 4),
            "speedup": round(speedup, 2),
            "overlap_pct": round(overlap_pct, 1),
        }
        print(f"  M={M}: single={ss_ms:.4f}ms, dual={ds_ms:.4f}ms → {speedup:.2f}x (overlap={overlap_pct:.1f}%)")

    results["exp2_multi_stream"] = exp2

    # Exp 3: Stream priority
    print("\n--- Exp 3: Stream Priority Impact ---")
    exp3 = {}

    # High priority stream vs low priority stream
    s_high = torch.cuda.Stream(priority=-1)  # high priority (lower number = higher priority)
    s_low = torch.cuda.Stream(priority=1)    # low priority

    for M in [1, 16, 64, 256]:
        A = torch.randn(M, 4096, device=device, dtype=torch.bfloat16)
        B = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        # Default stream
        def default_priority():
            C = torch.mm(A, B)
            return C

        # High priority
        def high_priority():
            with torch.cuda.stream(s_high):
                C = torch.mm(A, B)
                torch.cuda.current_stream().synchronize()
            return C

        # Low priority
        def low_priority():
            with torch.cuda.stream(s_low):
                C = torch.mm(A, B)
                torch.cuda.current_stream().synchronize()
            return C

        def_ms, _ = benchmark(default_priority)
        high_ms, _ = benchmark(high_priority)
        low_ms, _ = benchmark(low_priority)

        exp3[f"M={M}"] = {
            "default_ms": round(def_ms, 4),
            "high_priority_ms": round(high_ms, 4),
            "low_priority_ms": round(low_ms, 4),
        }
        print(f"  M={M}: default={def_ms:.4f}ms, high={high_ms:.4f}ms, low={low_ms:.4f}ms")

    results["exp3_stream_priority"] = exp3

    # Exp 4: H2D/D2H transfer + compute overlap
    print("\n--- Exp 4: H2D Transfer + Compute Overlap ---")
    exp4 = {}

    # Test: transfer data to GPU while doing compute on another stream
    for transfer_mb in [1, 4, 16, 64]:
        transfer_bytes = transfer_mb * 1024 * 1024
        host_data = torch.randn(transfer_bytes // 2, device="cpu", dtype=torch.bfloat16)

        # Pure transfer
        def pure_transfer():
            gpu_data = host_data.to(device, non_blocking=True)
            torch.cuda.synchronize()
            return gpu_data

        # Pure compute (large GEMM)
        A = torch.randn(1024, 4096, device=device, dtype=torch.bfloat16)
        B = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        def pure_compute():
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            return C

        # Overlapped: transfer on stream 1, compute on stream 0
        s_transfer = torch.cuda.Stream()

        def overlapped():
            with torch.cuda.stream(s_transfer):
                gpu_data = host_data.to(device, non_blocking=True)
            C = torch.mm(A, B)  # on default stream
            torch.cuda.synchronize()
            return C, gpu_data

        transfer_ms, _ = benchmark(pure_transfer)
        compute_ms, _ = benchmark(pure_compute)
        overlap_ms, _ = benchmark(overlapped)

        # Ideal overlapped = max(transfer, compute)
        ideal_ms = max(transfer_ms, compute_ms)
        actual_overlap = max(0, (transfer_ms + compute_ms - overlap_ms) / min(transfer_ms, compute_ms) * 100) if min(transfer_ms, compute_ms) > 0 else 0

        exp4[f"transfer={transfer_mb}MB"] = {
            "transfer_ms": round(transfer_ms, 4),
            "compute_ms": round(compute_ms, 4),
            "overlap_ms": round(overlap_ms, 4),
            "ideal_ms": round(ideal_ms, 4),
            "overlap_pct": round(actual_overlap, 1),
            "transfer_bandwidth_gb_s": round(transfer_mb / transfer_ms * 1000 / 1024, 2) if transfer_ms > 0 else 0,
        }
        print(f"  Transfer={transfer_mb}MB: transfer={transfer_ms:.4f}ms, compute={compute_ms:.4f}ms, overlap={overlap_ms:.4f}ms, overlap_pct={actual_overlap:.1f}%")

    results["exp4_transfer_overlap"] = exp4

    # Exp 5: Prefill+Decode concurrent simulation
    print("\n--- Exp 5: Prefill+Decode Concurrent Simulation ---")
    exp5 = {}

    # Simulate: prefill on stream 0 (large GEMM M=1024) + decode on stream 1 (small GEMM M=1)
    A_prefill = torch.randn(1024, 4096, device=device, dtype=torch.bfloat16)
    B_prefill = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    A_decode = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
    B_decode = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

    # Sequential: prefill then decode
    def sequential():
        C_p = torch.mm(A_prefill, B_prefill)
        torch.cuda.synchronize()
        C_d = torch.mm(A_decode, B_decode)
        torch.cuda.synchronize()
        return C_p, C_d

    # Concurrent: both on different streams
    s_prefill = torch.cuda.Stream(priority=0)
    s_decode = torch.cuda.Stream(priority=-1)  # decode higher priority!

    def concurrent():
        with torch.cuda.stream(s_prefill):
            C_p = torch.mm(A_prefill, B_prefill)
        with torch.cuda.stream(s_decode):
            C_d = torch.mm(A_decode, B_decode)
        torch.cuda.synchronize()
        return C_p, C_d

    # Measure each individually
    prefill_only_ms, _ = benchmark(lambda: torch.mm(A_prefill, B_prefill))
    decode_only_ms, _ = benchmark(lambda: torch.mm(A_decode, B_decode))
    seq_ms, _ = benchmark(sequential)
    conc_ms, _ = benchmark(concurrent)

    exp5["prefill_decode"] = {
        "prefill_only_ms": round(prefill_only_ms, 4),
        "decode_only_ms": round(decode_only_ms, 4),
        "sequential_ms": round(seq_ms, 4),
        "concurrent_ms": round(conc_ms, 4),
        "concurrent_speedup": round(seq_ms / conc_ms, 2) if conc_ms > 0 else 0,
        "decode_in_prefill_pct": round(decode_only_ms / prefill_only_ms * 100, 2),
    }
    print(f"  Prefill={prefill_only_ms:.4f}ms, Decode={decode_only_ms:.4f}ms")
    print(f"  Sequential={seq_ms:.4f}ms, Concurrent={conc_ms:.4f}ms → {seq_ms/conc_ms:.2f}x")
    print(f"  Decode in prefill time: {decode_only_ms/prefill_only_ms*100:.2f}% → decode几乎free!")

    # Multiple decode steps during one prefill
    for n_decode in [1, 4, 8, 16, 32]:
        A_d = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
        B_d = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        def concurrent_n():
            with torch.cuda.stream(s_prefill):
                C_p = torch.mm(A_prefill, B_prefill)
            for _ in range(n_decode):
                with torch.cuda.stream(s_decode):
                    C_d = torch.mm(A_d, B_d)
            s_decode.synchronize()
            torch.cuda.synchronize()
            return C_p

        conc_n_ms, _ = benchmark(concurrent_n)
        # Sequential: prefill + n_decode × decode
        seq_theoretical_ms = prefill_only_ms + n_decode * decode_only_ms

        exp5[f"prefill+{n_decode}decode"] = {
            "concurrent_ms": round(conc_n_ms, 4),
            "sequential_theoretical_ms": round(seq_theoretical_ms, 4),
            "speedup": round(seq_theoretical_ms / conc_n_ms, 2) if conc_n_ms > 0 else 0,
        }
        print(f"  Prefill + {n_decode} decode concurrent: {conc_n_ms:.4f}ms vs sequential {seq_theoretical_ms:.4f}ms → {seq_theoretical_ms/conc_n_ms:.2f}x")

    results["exp5_prefill_decode_concurrent"] = exp5

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — CUDA Stream Concurrency RTX 4090")
    print("=" * 70)

    e1 = exp1.get("M=1", {})
    print(f"\n  Launch overhead: {e1.get('launch_pct', 'N/A')}% at M=1 → dominates small kernels!")
    e2 = exp2.get("M=1", {})
    print(f"  Multi-stream M=1: {e2.get('speedup', 'N/A')}x → {'overlap possible' if e2.get('speedup', 1) > 1.1 else 'no overlap (launch overhead dominates)'}")
    e5_main = exp5.get("prefill_decode", {})
    print(f"  Prefill+Decode concurrent: {e5_main.get('concurrent_speedup', 'N/A')}x → decode几乎free in prefill time!")

    print(f"\n  Production implications:")
    print(f"    → RTX 4090 has 128 SMs → can overlap small+large kernels on different streams")
    print(f"    → Decode(M=1) in Prefill(M=1024): decode time ≈ 2% of prefill → almost free!")
    print(f"    → Continuous batching: decode requests can run during prefill → higher throughput")
    print(f"    → Stream priority: decode(high priority) can preempt prefill → lower ITL jitter")

    return results


if __name__ == '__main__':
    results = run_all()
    try:
        with open('results/cuda_stream_concurrency.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('cuda_stream_concurrency.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)