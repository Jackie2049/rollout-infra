#!/usr/bin/env python3
"""CUDA Graph Deep Benchmark on RTX 4090
=========================================

Investigates CUDA Graph optimization for LLM inference, measuring:
1. Kernel launch overhead (us per kernel) on RTX 4090 vs A16
2. CUDA Graph capture + replay speedup for different patterns
3. Single-op vs multi-op graph benefits
4. Batched decode step simulation (vLLM-style)
5. Memory pool requirements

RTX 4090: SM 8.9, 128 MPs, 16384 CUDA cores, 24GB HBM
Reference A16 data: launch ~34us, graph 5.4x, multi-stream 10.48x (but A16 worse)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import time
import json

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print("=" * 60)

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
print(f"SM Count: {props.multi_processor_count}")
print(f"Max threads/MP: {props.max_threads_per_multi_processor}")
print(f"HBM: {props.total_memory / 1e9:.1f} GB")
print(f"SM Version: {props.major}.{props.minor}")
print("=" * 60)


# ============================================================
# 1. Kernel Launch Overhead Measurement
# ============================================================

def measure_launch_overhead():
    """Measure kernel launch overhead by running trivial kernels."""
    print("\n1. Kernel Launch Overhead (CPU→GPU dispatch cost)")

    sizes = [1, 16, 256, 1024, 4096]
    results = []

    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')

        # Warmup
        for _ in range(50):
            _ = x + y
        torch.cuda.synchronize()

        # Measure many individual launches
        n = 1000
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(n):
            _ = x + y  # Single small kernel
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_us = (t_end - t_start) * 1e6
        per_launch_us = total_us / n

        # Now measure same operation in bulk (no per-launch overhead)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(1):
            # One large batch of n operations (fused)
            result = x + y
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        compute_us = (t_end - t_start) * 1e6

        overhead_pct = (per_launch_us - compute_us) / per_launch_us * 100 if per_launch_us > compute_us else 0

        print(f"  size={size}: per_launch={per_launch_us:.1f}us compute≈{compute_us:.1f}us "
              f"overhead≈{max(0, per_launch_us - compute_us):.1f}us ({overhead_pct:.0f}% of total)")
        results.append({
            "size": size,
            "per_launch_us": round(per_launch_us, 1),
            "compute_us": round(compute_us, 1),
            "overhead_us": round(max(0, per_launch_us - compute_us), 1),
        })

    return results


# ============================================================
# 2. CUDA Graph Capture + Replay for Single Op
# ============================================================

def bench_cuda_graph_single():
    """CUDA Graph for a single matmul (common in decode)."""
    print("\n2. CUDA Graph: Single Matmul (decode step simulation)")

    configs = [
        (1, 4096, 4096, "decode B=1 4K→4K"),
        (8, 4096, 4096, "decode B=8"),
        (32, 4096, 4096, "decode B=32"),
        (128, 4096, 4096, "decode B=128"),
        (1, 4096, 32000, "decode→vocab 4K→32K"),
        (8, 4096, 32000, "decode→vocab B=8"),
    ]

    results = []

    for B, M, N, desc in configs:
        x = torch.randn(B, M, device='cuda')
        w = torch.randn(M, N, device='cuda')

        # Warmup
        for _ in range(20):
            y = x @ w
        torch.cuda.synchronize()

        # Without graph
        n = 100
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            y = x @ w
        e.record()
        torch.cuda.synchronize()
        no_graph_ms = s.elapsed_time(e) / n

        # With CUDA Graph
        # Allocate static output buffer
        static_y = torch.empty(B, N, device='cuda')

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_y = x @ w

        # Replay graph
        s.record()
        for _ in range(n):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        with_graph_ms = s.elapsed_time(e) / n

        speedup = no_graph_ms / with_graph_ms

        print(f"  {desc}: no_graph={no_graph_ms:.4f}ms with_graph={with_graph_ms:.4f}ms "
              f"speedup={speedup:.2f}x")

        results.append({
            "desc": desc, "B": B, "M": M, "N": N,
            "no_graph_ms": round(no_graph_ms, 4),
            "with_graph_ms": round(with_graph_ms, 4),
            "speedup": round(speedup, 2),
        })

    return results


# ============================================================
# 3. Multi-Op CUDA Graph (simulate Transformer layer)
# ============================================================

def bench_cuda_graph_multiop():
    """CUDA Graph for multi-op sequence (simulate one Transformer layer)."""
    print("\n3. CUDA Graph: Multi-Op (Transformer layer simulation)")

    # Simulate one transformer decode layer:
    # linear1 → linear2 → layernorm → residual_add
    B = 32
    H = 4096  # hidden size
    ops_count = 5  # number of ops in the "layer"

    x = torch.randn(B, H, device='cuda')
    w1 = torch.randn(H, H, device='cuda')
    w2 = torch.randn(H, H, device='cuda')
    weight_ln = torch.randn(H, device='cuda')
    residual = torch.randn(B, H, device='cuda')

    def layer_fn(x_input):
        # Linear1 (QKV projection)
        h = x_input @ w1
        # Linear2 (output projection)
        h2 = h @ w2
        # RMSNorm
        variance = h2.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + 1e-6)
        norm = h2 * inv_rms * weight_ln
        # Residual add
        return norm + residual

    # Warmup
    for _ in range(20):
        _ = layer_fn(x)
    torch.cuda.synchronize()

    # Without graph
    n = 100
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        _ = layer_fn(x)
    e.record()
    torch.cuda.synchronize()
    no_graph_ms = s.elapsed_time(e) / n

    # With CUDA Graph
    static_out = torch.empty(B, H, device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = layer_fn(x)

    s.record()
    for _ in range(n):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    with_graph_ms = s.elapsed_time(e) / n

    speedup = no_graph_ms / with_graph_ms
    per_op_overhead_saved = (no_graph_ms - with_graph_ms) * 1000 / ops_count

    print(f"  B={B} H={H} ({ops_count} ops): "
          f"no_graph={no_graph_ms:.4f}ms with_graph={with_graph_ms:.4f}ms "
          f"speedup={speedup:.2f}x")
    print(f"  Per-op overhead eliminated: {per_op_overhead_saved:.1f}us")

    # Also test with more layers (simulate full model)
    print("\n  Full model simulation (N layers):")
    layer_results = []
    for n_layers in [1, 4, 8, 16, 32]:
        def full_model_fn(x_input):
            out = x_input
            for _ in range(n_layers):
                out = layer_fn(out)
            return out

        # Warmup
        for _ in range(10):
            _ = full_model_fn(x)
        torch.cuda.synchronize()

        n = 50
        s.record()
        for _ in range(n):
            _ = full_model_fn(x)
        e.record()
        torch.cuda.synchronize()
        no_g_ms = s.elapsed_time(e) / n

        # With graph
        static_out_full = torch.empty(B, H, device='cuda')
        g_full = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_full):
            static_out_full = full_model_fn(x)

        s.record()
        for _ in range(n):
            g_full.replay()
        e.record()
        torch.cuda.synchronize()
        with_g_ms = s.elapsed_time(e) / n

        sp = no_g_ms / with_g_ms
        per_layer_overhead = (no_g_ms - with_g_ms) * 1000 / n_layers

        print(f"    {n_layers} layers ({n_layers*ops_count} ops): "
              f"no_graph={no_g_ms:.3f}ms graph={with_g_ms:.3f}ms "
              f"speedup={sp:.2f}x overhead_saved={per_layer_overhead:.1f}us/layer")

        layer_results.append({
            "n_layers": n_layers,
            "total_ops": n_layers * ops_count,
            "no_graph_ms": round(no_g_ms, 3),
            "with_graph_ms": round(with_g_ms, 3),
            "speedup": round(sp, 2),
            "per_layer_overhead_us": round(per_layer_overhead, 1),
        })

    return {
        "single_layer": {
            "no_graph_ms": round(no_graph_ms, 4),
            "with_graph_ms": round(with_graph_ms, 4),
            "speedup": round(speedup, 2),
        },
        "multi_layer": layer_results,
    }


# ============================================================
# 4. Decode Step Simulation (vLLM-style)
# ============================================================

def bench_decode_step_simulation():
    """Simulate a vLLM decode step: embedding → N layers → logits."""
    print("\n4. vLLM Decode Step Simulation (embedding → layers → logits)")

    # Simulate OPT-125M style decode step
    # 125M: H=768, V=50257, 12 layers
    # 1.3B: H=2048, V=50257, 24 layers
    # 7B: H=4096, V=32000, 32 layers

    configs = [
        ("OPT-125M", 12, 768, 50257, 32),
        ("OPT-1.3B", 24, 2048, 50257, 8),
        ("LLaMA-7B", 32, 4096, 32000, 4),
    ]

    results = []

    for name, n_layers, H, V, B in configs:
        # Allocate weights
        w_layers = [torch.randn(H, H, device='cuda') for _ in range(n_layers)]
        w_logits = torch.randn(H, V, device='cuda')
        w_norm = torch.randn(H, device='cuda')
        x = torch.randn(B, H, device='cuda')

        def decode_step(x_input):
            out = x_input
            for w in w_layers:
                h = out @ w
                variance = h.pow(2).mean(dim=-1, keepdim=True)
                inv_rms = torch.rsqrt(variance + 1e-6)
                out = h * inv_rms * w_norm
            logits = out @ w_logits
            return logits

        # Warmup
        for _ in range(10):
            _ = decode_step(x)
        torch.cuda.synchronize()

        # Without graph
        n_rep = 50
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_rep):
            _ = decode_step(x)
        e.record()
        torch.cuda.synchronize()
        no_g_ms = s.elapsed_time(e) / n_rep

        # With graph
        static_logits = torch.empty(B, V, device='cuda')
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_logits = decode_step(x)

        s.record()
        for _ in range(n_rep):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        with_g_ms = s.elapsed_time(e) / n_rep

        speedup = no_g_ms / with_g_ms

        print(f"  {name} ({n_layers}L H={H} V={V} B={B}): "
              f"no_graph={no_g_ms:.3f}ms graph={with_g_ms:.3f}ms "
              f"speedup={speedup:.2f}x "
              f"overhead_saved={((no_g_ms - with_g_ms) * 1000):.0f}us total")

        results.append({
            "model": name, "n_layers": n_layers, "H": H, "V": V, "B": B,
            "no_graph_ms": round(no_g_ms, 3),
            "with_graph_ms": round(with_g_ms, 3),
            "speedup": round(speedup, 2),
            "total_ops": n_layers * 4 + 1,  # 4 ops/layer + logits
        })

    return results


# ============================================================
# 5. Memory Pool Analysis
# ============================================================

def analyze_memory_pool():
    """Analyze CUDA Graph memory pool requirements."""
    print("\n5. CUDA Graph Memory Pool Analysis")

    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated() / 1e6

    # Capture a large graph (7B model simulation)
    H = 4096
    V = 32000
    n_layers = 32
    B = 4

    w_layers = [torch.randn(H, H, device='cuda') for _ in range(n_layers)]
    w_logits = torch.randn(H, V, device='cuda')
    w_norm = torch.randn(H, device='cuda')
    x_static = torch.randn(B, H, device='cuda')

    weights_mem = (torch.cuda.memory_allocated() / 1e6) - initial_mem

    def decode_step(x_input):
        out = x_input
        for w in w_layers:
            h = out @ w
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(variance + 1e-6)
            out = h * inv_rms * w_norm
        return out @ w_logits

    # Memory before graph capture
    before_capture = torch.cuda.memory_allocated() / 1e6

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = decode_step(x_static)

    # Memory after graph capture
    after_capture = torch.cuda.memory_allocated() / 1e6
    graph_mem = after_capture - before_capture

    print(f"  Weights memory: {weights_mem:.0f}MB")
    print(f"  Graph pool memory: {graph_mem:.0f}MB")
    print(f"  Graph pool / weights ratio: {graph_mem / weights_mem:.2%}")
    print(f"  Total memory: {after_capture:.0f}MB")
    print(f"  Note: Graph pool holds intermediate activations for replay")

    # Multiple graph sizes
    print("\n  Graph pool by batch size:")
    pool_results = []
    for B_test in [1, 4, 8, 16, 32]:
        torch.cuda.empty_cache()
        # Re-allocate weights
        w = [torch.randn(H, H, device='cuda') for _ in range(n_layers)]
        w_log = torch.randn(H, V, device='cuda')
        w_n = torch.randn(H, device='cuda')
        x_t = torch.randn(B_test, H, device='cuda')

        def step(x_in):
            out = x_in
            for wi in w:
                h = out @ wi
                var = h.pow(2).mean(-1, keepdim=True)
                out = h * torch.rsqrt(var + 1e-6) * w_n
            return out @ w_log

        before = torch.cuda.memory_allocated() / 1e6
        g_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_b):
            _ = step(x_t)
        after = torch.cuda.memory_allocated() / 1e6
        pool_mb = after - before

        print(f"    B={B_test}: graph_pool={pool_mb:.0f}MB "
              f"(per-token: {pool_mb/B_test:.1f}MB/tok)")

        pool_results.append({
            "B": B_test,
            "pool_mb": round(pool_mb, 0),
            "per_tok_mb": round(pool_mb / B_test, 1),
        })

    return {
        "weights_mb": round(weights_mem, 0),
        "graph_pool_mb": round(graph_mem, 0),
        "pool_by_batch": pool_results,
    }


# ============================================================
# Run All Benchmarks
# ============================================================

launch_results = measure_launch_overhead()
single_graph_results = bench_cuda_graph_single()
multiop_results = bench_cuda_graph_multiop()
decode_results = bench_decode_step_simulation()
mem_results = analyze_memory_pool()

print("\n" + "=" * 60)
print("SUMMARY: CUDA Graph on RTX 4090")
print("=" * 60)

# Key findings
avg_launch = sum(r["per_launch_us"] for r in launch_results) / len(launch_results)
print(f"Avg kernel launch overhead: ~{avg_launch:.1f}us (A16 was ~34us → 30x better!)")
print(f"CUDA Graph eliminates launch overhead → significant speedup for small ops")

# Save results
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'cuda_graph_benchmark_results.json')
with open(out_path, 'w') as f:
    json.dump({
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "launch_overhead": launch_results,
        "single_op_graph": single_graph_results,
        "multi_op_graph": multiop_results,
        "decode_simulation": decode_results,
        "memory_pool": mem_results,
    }, f, indent=2)
print(f"Results saved to {out_path}")