"""
GPU Profiling Workshop — RTX 4090
Uses torch.profiler to capture kernel-level traces and measure CPU overhead vs GPU compute.

Focus: Learning GPU profiling as an AI infra production skill.
"""

import torch
import time
import json
import os

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
HBM_GB = props.total_memory / 1024**3
print(f"Device: {props.name}, HBM: {HBM_GB:.2f} GB")

hidden = 4096
inter_dim = 14336
vocab_size = 32000


def benchmark_op(op_fn, warmup=10, repeats=100):
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        op_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return {"avg_us": round(sum(times) / len(times), 2), "min_us": round(min(times), 2)}


def run_all():
    results = {}
    print("=" * 70)
    print("GPU Profiling Workshop — RTX 4090")
    print("=" * 70)

    # ====================================================================
    # Exp 1: CPU Overhead vs GPU Compute (Launch + Dispatch)
    # ====================================================================
    print("\n--- Exp 1: CPU Overhead vs GPU Compute ---")
    exp1 = {}

    # Measure pure wall clock time vs GPU-only time for each op
    ops_config = [
        ("attn_GEMM_1x4096x4096", 1, hidden, hidden),
        ("attn_GEMM_32x4096x4096", 32, hidden, hidden),
        ("mlp_gate_1x4096x14336", 1, hidden, inter_dim),
        ("mlp_gate_32x4096x14336", 32, hidden, inter_dim),
        ("lm_head_1x4096x32000", 1, hidden, vocab_size),
        ("lm_head_32x4096x32000", 32, hidden, vocab_size),
        ("rmsnorm_1x4096", 1, None, None),
        ("rmsnorm_32x4096", 32, None, None),
    ]

    for name, B, K, N in ops_config:
        x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

        if N is not None:  # GEMM
            w = torch.randn(K, N, device=device, dtype=torch.bfloat16)
            # Wall clock time (CPU dispatch + GPU compute)
            wall = benchmark_op(lambda: x @ w, warmup=5, repeats=50)

            # CPU-only time (dispatch overhead): measure with CUDA sync after each op
            torch.cuda.synchronize()
            cpu_times = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = x @ w  # just dispatch, don't sync
                t1 = time.perf_counter()
                cpu_times.append((t1 - t0) * 1e6)
            torch.cuda.synchronize()

            cpu_dispatch_us = sum(cpu_times) / len(cpu_times)
            gpu_compute_us = wall["avg_us"] - cpu_dispatch_us
            overhead_pct = cpu_dispatch_us / wall["avg_us"] * 100

            exp1[name] = {
                "wall_us": wall["avg_us"],
                "cpu_dispatch_us": round(cpu_dispatch_us, 2),
                "gpu_compute_us": round(gpu_compute_us, 2),
                "overhead_pct": round(overhead_pct, 1),
                "batch_size": B,
            }
            print(f"  {name}: wall={wall['avg_us']:.1f}us, CPU dispatch={cpu_dispatch_us:.1f}us({overhead_pct:.1f}%), GPU={gpu_compute_us:.1f}us")
        else:  # RMSNorm
            wall = benchmark_op(lambda: torch.nn.functional.rms_norm(x.float(), (hidden,)).to(torch.bfloat16), warmup=5, repeats=50)
            torch.cuda.synchronize()
            cpu_times = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = torch.nn.functional.rms_norm(x.float(), (hidden,)).to(torch.bfloat16)
                t1 = time.perf_counter()
                cpu_times.append((t1 - t0) * 1e6)
            torch.cuda.synchronize()

            cpu_dispatch_us = sum(cpu_times) / len(cpu_times)
            gpu_compute_us = wall["avg_us"] - cpu_dispatch_us
            overhead_pct = cpu_dispatch_us / wall["avg_us"] * 100

            exp1[name] = {
                "wall_us": wall["avg_us"],
                "cpu_dispatch_us": round(cpu_dispatch_us, 2),
                "gpu_compute_us": round(gpu_compute_us, 2),
                "overhead_pct": round(overhead_pct, 1),
                "batch_size": B,
            }
            print(f"  {name}: wall={wall['avg_us']:.1f}us, CPU dispatch={cpu_dispatch_us:.1f}us({overhead_pct:.1f}%), GPU={gpu_compute_us:.1f}us")

    results["exp1_cpu_overhead"] = exp1

    # ====================================================================
    # Exp 2: Profiler Trace Export — Full Decode Step
    # ====================================================================
    print("\n--- Exp 2: torch.profiler Trace Export ---")
    exp2 = {}

    import torch.profiler as profiler

    B = 16
    x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)
    w_qkv = torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_down = torch.randn(inter_dim, hidden, device=device, dtype=torch.bfloat16)
    w_lm = torch.randn(hidden, vocab_size, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        qkv = x @ w_qkv
        gate = x @ w_gate
        up = x @ w_up
        silu = torch.nn.functional.silu(gate) * up
        down = silu @ w_down
        logits = x @ w_lm
    torch.cuda.synchronize()

    # Profile with torch.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for step in range(3):
            qkv = x @ w_qkv
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down
            logits = x @ w_lm

    # Export Chrome trace
    trace_path = "/tmp/gpu_profiler_trace.json"
    prof.export_chrome_trace(trace_path)
    trace_size_kb = os.path.getsize(trace_path) / 1024

    # Extract key averages using compatible API
    key_averages = prof.key_averages()
    kernel_list = []
    for evt in key_averages:
        # Use cpu_time_total (available in all PyTorch versions)
        cpu_time = evt.cpu_time_total if hasattr(evt, 'cpu_time_total') else 0
        self_cpu = evt.self_cpu_time_total if hasattr(evt, 'self_cpu_time_total') else 0
        count = evt.count if hasattr(evt, 'count') else 0
        key = evt.key if hasattr(evt, 'key') else str(evt)

        if count > 0 and self_cpu > 0:
            kernel_list.append({
                "name": key,
                "count": count,
                "cpu_time_us": round(cpu_time, 1),
                "self_cpu_time_us": round(self_cpu, 1),
            })

    # Sort by self_cpu_time
    kernel_list.sort(key=lambda k: k["self_cpu_time_us"], reverse=True)

    exp2["trace_export"] = {
        "path": trace_path,
        "size_kb": round(trace_size_kb, 2),
        "steps_profiled": 3,
        "total_kernels": len(kernel_list),
    }
    exp2["top_kernels"] = kernel_list[:20]

    print(f"  Trace exported: {trace_path} ({trace_size_kb:.1f}KB)")
    print(f"  Total kernel entries: {len(kernel_list)}")
    print(f"  Top 10 kernels by self CPU time:")
    for k in kernel_list[:10]:
        print(f"    {k['name']}: cpu={k['self_cpu_time_us']/1000:.2f}ms ×{k['count']}")

    results["exp2_trace_export"] = exp2

    # ====================================================================
    # Exp 3: Memory Profile per Operation
    # ====================================================================
    print("\n--- Exp 3: Memory Profile per Operation ---")
    exp3 = {}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    B = 16
    x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

    ops = {
        "qkv_proj": lambda: x @ torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16),
        "gate_proj": lambda: x @ torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16),
        "lm_head": lambda: x @ torch.randn(hidden, vocab_size, device=device, dtype=torch.bfloat16),
        "rmsnorm": lambda: torch.nn.functional.rms_norm(x.float(), (hidden,)).to(torch.bfloat16),
        "silu_mul": lambda: torch.nn.functional.silu(torch.randn(B, inter_dim, device=device, dtype=torch.bfloat16)) * torch.randn(B, inter_dim, device=device, dtype=torch.bfloat16),
    }

    for name, op in ops.items():
        # Warmup
        for _ in range(3):
            op()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_mb = torch.cuda.memory_allocated() / 1024**2

        result = op()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        after_mb = torch.cuda.memory_allocated() / 1024**2
        del result

        exp3[name] = {
            "peak_mb": round(peak_mb, 2),
            "increment_mb": round(peak_mb - before_mb, 2),
            "after_mb": round(after_mb, 2),
        }
        print(f"  {name}: peak={peak_mb:.2f}MB, increment={peak_mb - before_mb:.2f}MB")

    results["exp3_memory_profile"] = exp3

    # ====================================================================
    # Exp 4: CUDA Graph Capture Overhead
    # ====================================================================
    print("\n--- Exp 4: CUDA Graph Capture vs Eager ---")
    exp4 = {}

    B_sizes = [1, 4, 16, 32]

    for B in B_sizes:
        x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)
        w_qkv = torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16)
        w_gate = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
        w_up = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
        w_down = torch.randn(inter_dim, hidden, device=device, dtype=torch.bfloat16)

        def decode_step_eager():
            qkv = x @ w_qkv
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down

        # Warmup eager
        for _ in range(10):
            decode_step_eager()
        torch.cuda.synchronize()

        # Measure eager
        eager_time = benchmark_op(decode_step_eager, warmup=5, repeats=50)

        # CUDA Graph capture
        # Warmup for graph capture (need at least 1 iteration)
        decode_step_eager()
        torch.cuda.synchronize()

        # Capture graph
        g_x = x.clone()
        g_qkv = w_qkv.clone()
        g_gate = w_gate.clone()
        g_up = w_up.clone()
        g_down = w_down.clone()
        g_silu_gate = torch.randn(B, inter_dim, device=device, dtype=torch.bfloat16)
        g_down_result = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

        # Static buffers for graph outputs
        static_qkv = torch.randn(B, 3 * hidden, device=device, dtype=torch.bfloat16)
        static_gate = torch.randn(B, inter_dim, device=device, dtype=torch.bfloat16)
        static_up = torch.randn(B, inter_dim, device=device, dtype=torch.bfloat16)
        static_down = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

        # Warmup for graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            static_qkv.copy_(g_x @ g_qkv)
            static_gate.copy_(g_x @ g_gate)
            static_up.copy_(g_x @ g_up)
            g_silu_gate = torch.nn.functional.silu(static_gate) * static_up
            static_down.copy_(g_silu_gate @ g_down)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        capture_time_start = time.perf_counter()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_qkv.copy_(g_x @ g_qkv)
            static_gate.copy_(g_x @ g_gate)
            static_up.copy_(g_x @ g_up)
            g_silu_gate = torch.nn.functional.silu(static_gate) * static_up
            static_down.copy_(g_silu_gate @ g_down)
        capture_time_ms = (time.perf_counter() - capture_time_start) * 1000

        # Replay graph
        graph_times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            graph_times.append((t1 - t0) * 1e6)

        graph_avg_us = sum(graph_times) / len(graph_times)
        speedup = eager_time["avg_us"] / graph_avg_us if graph_avg_us > 0 else 0
        launch_overhead_saved = eager_time["avg_us"] - graph_avg_us

        exp4[str(B)] = {
            "eager_us": eager_time["avg_us"],
            "graph_us": round(graph_avg_us, 2),
            "capture_ms": round(capture_time_ms, 2),
            "speedup": round(speedup, 2),
            "overhead_saved_us": round(launch_overhead_saved, 2),
        }
        print(f"  B={B}: eager={eager_time['avg_us']:.1f}us, graph={graph_avg_us:.1f}us → speedup={speedup:.2f}x, capture={capture_time_ms:.2f}ms")

    results["exp4_cuda_graph"] = exp4

    # ====================================================================
    # Exp 5: Full Decode Pipeline Profiling Summary
    # ====================================================================
    print("\n--- Exp 5: Decode Pipeline Profiling Summary ---")
    exp5 = {}

    # Combine all data: CPU overhead + CUDA Graph + Memory
    # Estimate full 7B decode with profiling knowledge

    for B in [1, 32, 55]:
        # From Exp 1: CPU overhead for each op
        # 32 layers × (attn_GEMM + mlp_GEMMs + rmsnorm) + lm_head + sampling

        # CPU dispatch per kernel ≈ 5-8us (from Exp 1)
        # Per layer: 7 GEMMs + 2 RMSNorms ≈ 9 kernel launches × 8us = 72us CPU overhead
        # 32 layers = 32 × 72 = 2304us CPU overhead
        # + lm_head (1 launch) + sampling (2 launches) ≈ 24us
        # Total CPU overhead ≈ 2328us

        # GPU compute (from previous benchmark): ~25ms for 7B B=1

        cpu_overhead_us = 32 * 9 * 8 + 3 * 8  # 9 launches per layer × 8us each
        gpu_compute_us = 25603  # from previous decode breakdown (B=1)

        # B scaling: GPU compute scales linearly (memory-bound)
        if B > 1:
            # GPU compute ≈ 13000us for all B (memory-bound → time almost constant!)
            gpu_compute_us = 13000  # from Exp 2 of previous benchmark

        total_us = cpu_overhead_us + gpu_compute_us
        overhead_pct = cpu_overhead_us / total_us * 100

        # CUDA Graph eliminates CPU overhead
        graph_time_us = gpu_compute_us  # only GPU compute, no dispatch
        graph_speedup = total_us / graph_time_us if graph_time_us > 0 else 0

        exp5[str(B)] = {
            "cpu_dispatch_us": round(cpu_overhead_us, 1),
            "gpu_compute_us": round(gpu_compute_us, 1),
            "total_us": round(total_us, 1),
            "overhead_pct": round(overhead_pct, 1),
            "graph_time_us": round(graph_time_us, 1),
            "graph_speedup": round(graph_speedup, 2),
        }
        print(f"  B={B}: CPU={cpu_overhead_us}us({overhead_pct:.1f}%), GPU={gpu_compute_us}us → total={total_us}us → CUDA Graph → {graph_speedup:.2f}x")

    results["exp5_pipeline_summary"] = exp5

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — GPU Profiling Workshop RTX 4090")
    print("=" * 70)

    e1_attn_b1 = exp1.get("attn_GEMM_1x4096x4096", {})
    e1_attn_b32 = exp1.get("attn_GEMM_32x4096x4096", {})
    e1_lm_b1 = exp1.get("lm_head_1x4096x32000", {})
    e4_b1 = exp4.get("1", {})
    e4_b32 = exp4.get("32", {})

    print(f"\n  CPU overhead (launch + dispatch):")
    print(f"    B=1: {e1_attn_b1.get('overhead_pct', 0):.1f}% of wall time → dominates!")
    print(f"    B=32: {e1_attn_b32.get('overhead_pct', 0):.1f}% → smaller for larger ops")
    print(f"    lm_head B=1: {e1_lm_b1.get('overhead_pct', 0):.1f}% → large GEMM still has overhead")
    print(f"\n  CUDA Graph:")
    print(f"    B=1: {e4_b1.get('speedup', 0):.2f}x → eliminates CPU dispatch overhead")
    print(f"    B=32: {e4_b32.get('speedup', 0):.2f}x → less benefit (GPU compute dominates)")
    print(f"\n  Profiling tools learned:")
    print(f"    1. Wall clock vs CPU dispatch vs GPU compute → identify bottlenecks")
    print(f"    2. torch.profiler → Chrome trace export → chrome://tracing visualization")
    print(f"    3. Memory profiling → peak_memory_stats → per-op memory usage")
    print(f"    4. CUDA Graph → eliminate CPU dispatch → speedup B=1 only")
    print(f"    5. Production: Nsight Systems > torch.profiler for deep analysis")

    return results


if __name__ == '__main__':
    torch.cuda.empty_cache()
    results = run_all()
    try:
        with open('results/gpu_profiling_workshop.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('gpu_profiling_workshop.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)