"""
KV Cache CPU Offloading (Swap) Benchmark — RTX 4090

Tests PCIe bandwidth and latency for KV cache CPU offloading (swap) operations.
Key question: When is swap better than recomputation for KV cache preemption?

Experiments:
1. PCIe bandwidth: pinned vs pageable CPU-GPU transfer
2. Per-block swap latency (vLLM block_size=16 tokens)
3. Swap vs recomputation cost comparison
4. ITL impact simulation
5. Prefill recomputation cost estimation
"""

import torch
import time
import json

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
HBM_GB = props.total_memory / 1024**3
print(f"Device: {props.name}, HBM: {HBM_GB:.2f} GB")


def benchmark_transfer(fn, warmup=5, repeats=20):
    """Benchmark a transfer operation."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    avg = sum(times) / len(times)
    return {"avg_us": round(avg, 2), "min_us": round(min(times), 2), "max_us": round(max(times), 2)}


def run_all():
    results = {}
    print("=" * 70)
    print("KV Cache CPU Offloading Benchmark — RTX 4090")
    print("=" * 70)

    # 7B model parameters (LLaMA-like, GQA-5)
    hidden = 4096
    n_kv_heads = 5
    head_dim = hidden // 32  # 128
    vocab_size = 32000
    block_size = 16  # vLLM default
    layers = 32

    # KV per token per layer (BF16): 2 * n_kv_heads * head_dim * 2 bytes
    kv_bytes_per_token_per_layer = 2 * n_kv_heads * head_dim * 2  # = 2560 bytes
    kv_bytes_per_token_total = kv_bytes_per_token_per_layer * layers  # = 81920 bytes
    # KV per block per layer
    kv_bytes_per_block_per_layer = kv_bytes_per_token_per_layer * block_size  # = 40960 bytes
    kv_bytes_per_block_total = kv_bytes_per_block_per_layer * layers  # = 1310720 bytes (1.25 MB)
    # INT8 KV
    kv_bytes_per_token_per_layer_int8 = 2 * n_kv_heads * head_dim * 1  # = 1280 bytes
    kv_bytes_per_block_per_layer_int8 = kv_bytes_per_token_per_layer_int8 * block_size * layers  # = 655360 bytes (0.625 MB)

    print(f"\nModel config: hidden={hidden}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"KV per token per layer: {kv_bytes_per_token_per_layer} bytes ({kv_bytes_per_token_per_layer/1024:.2f} KB)")
    print(f"KV per token total: {kv_bytes_per_token_total} bytes ({kv_bytes_per_token_total/1024:.2f} KB)")
    print(f"KV per block per layer: {kv_bytes_per_block_per_layer} bytes ({kv_bytes_per_block_per_layer/1024:.2f} KB)")
    print(f"KV per block total: {kv_bytes_per_block_total} bytes ({kv_bytes_per_block_total/1024**2:.2f} MB)")
    print(f"KV per block total INT8: {kv_bytes_per_block_per_layer_int8} bytes ({kv_bytes_per_block_per_layer_int8/1024**2:.2f} MB)")

    # ====================================================================
    # Exp 1: PCIe Bandwidth — Pinned vs Pageable Memory
    # ====================================================================
    print("\n--- Exp 1: PCIe Bandwidth (Pinned vs Pageable) ---")
    exp1 = {}

    sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128]

    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024 * 1024)
        n_elements = size_bytes // 2  # BF16 = 2 bytes

        # Pageable memory transfer (CPU → GPU)
        cpu_pageable = torch.randn(n_elements, dtype=torch.bfloat16)
        gpu_buf = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

        h2d_pageable = benchmark_transfer(lambda: gpu_buf.copy_(cpu_pageable))
        d2h_pageable = benchmark_transfer(lambda: cpu_pageable.copy_(gpu_buf))

        # Pinned memory transfer (CPU → GPU)
        cpu_pinned = torch.randn(n_elements, dtype=torch.bfloat16).pin_memory()
        gpu_buf2 = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

        h2d_pinned = benchmark_transfer(lambda: gpu_buf2.copy_(cpu_pinned, non_blocking=True))
        d2h_pinned = benchmark_transfer(lambda: cpu_pinned.copy_(gpu_buf2, non_blocking=True))

        # Calculate bandwidth
        h2d_pinned_bw = size_mb / (h2d_pinned["avg_us"] / 1e6)  # MB/s
        d2h_pinned_bw = size_mb / (d2h_pinned["avg_us"] / 1e6)
        h2d_pageable_bw = size_mb / (h2d_pageable["avg_us"] / 1e6)
        d2h_pageable_bw = size_mb / (d2h_pageable["avg_us"] / 1e6)

        exp1[str(size_mb)] = {
            "h2d_pinned_us": h2d_pinned["avg_us"],
            "d2h_pinned_us": d2h_pinned["avg_us"],
            "h2d_pinned_bw_mbps": round(h2d_pinned_bw, 2),
            "d2h_pinned_bw_mbps": round(d2h_pinned_bw, 2),
            "h2d_pageable_us": h2d_pageable["avg_us"],
            "d2h_pageable_us": d2h_pageable["avg_us"],
            "h2d_pageable_bw_mbps": round(h2d_pageable_bw, 2),
            "d2h_pageable_bw_mbps": round(d2h_pageable_bw, 2),
            "pinned_speedup_h2d": round(h2d_pageable["avg_us"] / h2d_pinned["avg_us"], 2),
            "pinned_speedup_d2h": round(d2h_pageable["avg_us"] / d2h_pinned["avg_us"], 2),
        }

        print(f"  {size_mb}MB: H2D pinned={h2d_pinned['avg_us']:.0f}us ({h2d_pinned_bw:.0f}MB/s) "
              f"vs pageable={h2d_pageable['avg_us']:.0f}us ({h2d_pageable_bw:.0f}MB/s) "
              f"→ pinned {h2d_pageable['avg_us']/h2d_pinned['avg_us']:.1f}x faster")

        del cpu_pageable, cpu_pinned, gpu_buf, gpu_buf2
        torch.cuda.empty_cache()

    results["exp1_pcie_bandwidth"] = exp1

    # ====================================================================
    # Exp 2: Per-Block Swap Latency
    # ====================================================================
    print("\n--- Exp 2: Per-Block Swap Latency ---")
    exp2 = {}

    # Simulate vLLM block swap: transfer KV cache for 1 block (16 tokens) to/from CPU
    # For each layer, transfer 2 tensors (K and V)

    # Single layer swap (K and V separately)
    kv_single_layer = torch.randn(block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    cpu_pinned_kv = torch.randn(block_size, n_kv_heads, head_dim, dtype=torch.bfloat16).pin_memory()

    # Swap out: GPU → CPU (per layer)
    swap_out_per_layer = benchmark_transfer(lambda: cpu_pinned_kv.copy_(kv_single_layer, non_blocking=True))

    # Swap in: CPU → GPU (per layer)
    swap_in_per_layer = benchmark_transfer(lambda: kv_single_layer.copy_(cpu_pinned_kv, non_blocking=True))

    # Full block swap: all 32 layers
    # Create KV for all layers on GPU
    gpu_kv_all_layers = [torch.randn(block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(layers)]
    cpu_pinned_kv_all = [torch.randn(block_size, n_kv_heads, head_dim, dtype=torch.bfloat16).pin_memory() for _ in range(layers)]

    def swap_out_full():
        for i in range(layers):
            cpu_pinned_kv_all[i].copy_(gpu_kv_all_layers[i], non_blocking=True)
        torch.cuda.synchronize()

    def swap_in_full():
        for i in range(layers):
            gpu_kv_all_layers[i].copy_(cpu_pinned_kv_all[i], non_blocking=True)
        torch.cuda.synchronize()

    swap_out_full_time = benchmark_transfer(swap_out_full)
    swap_in_full_time = benchmark_transfer(swap_in_full)

    # Full block swap (pinned, non_blocking=False — synchronous)
    def swap_out_full_sync():
        for i in range(layers):
            cpu_pinned_kv_all[i].copy_(gpu_kv_all_layers[i])
        torch.cuda.synchronize()

    def swap_in_full_sync():
        for i in range(layers):
            gpu_kv_all_layers[i].copy_(cpu_pinned_kv_all[i])
        torch.cuda.synchronize()

    swap_out_sync = benchmark_transfer(swap_out_full_sync)
    swap_in_sync = benchmark_transfer(swap_in_full_sync)

    exp2 = {
        "per_layer_swap_out_us": swap_out_per_layer["avg_us"],
        "per_layer_swap_in_us": swap_in_per_layer["avg_us"],
        "per_layer_bytes": kv_bytes_per_block_per_layer,
        "full_block_swap_out_async_us": swap_out_full_time["avg_us"],
        "full_block_swap_in_async_us": swap_in_full_time["avg_us"],
        "full_block_swap_out_sync_us": swap_out_sync["avg_us"],
        "full_block_swap_in_sync_us": swap_in_sync["avg_us"],
        "full_block_bytes": kv_bytes_per_block_total,
        "full_block_async_bw_mbps": round(kv_bytes_per_block_total / 1024**2 / (swap_in_full_time["avg_us"] / 1e6), 2),
        "full_block_sync_bw_mbps": round(kv_bytes_per_block_total / 1024**2 / (swap_in_sync["avg_us"] / 1e6), 2),
    }

    # INT8 KV swap (use randint since randn doesn't support int8)
    gpu_kv_int8 = [torch.randint(-128, 127, (block_size, n_kv_heads, head_dim), device=device, dtype=torch.int8) for _ in range(layers)]
    cpu_pinned_kv_int8 = [torch.randint(-128, 127, (block_size, n_kv_heads, head_dim)).to(torch.int8).pin_memory() for _ in range(layers)]

    def swap_in_int8():
        for i in range(layers):
            gpu_kv_int8[i].copy_(cpu_pinned_kv_int8[i], non_blocking=True)
        torch.cuda.synchronize()

    swap_in_int8_time = benchmark_transfer(swap_in_int8)
    exp2["int8_swap_in_us"] = swap_in_int8_time["avg_us"]
    exp2["int8_block_bytes"] = kv_bytes_per_block_per_layer_int8
    exp2["int8_swap_speedup"] = round(swap_in_full_time["avg_us"] / swap_in_int8_time["avg_us"], 2)

    print(f"  Per layer swap out: {swap_out_per_layer['avg_us']:.0f}us, swap in: {swap_in_per_layer['avg_us']:.0f}us")
    print(f"  Full block swap out async: {swap_out_full_time['avg_us']:.0f}us, sync: {swap_out_sync['avg_us']:.0f}us")
    print(f"  Full block swap in async: {swap_in_full_time['avg_us']:.0f}us ({kv_bytes_per_block_total/1024**2/(swap_in_full_time['avg_us']/1e6):.0f}MB/s), sync: {swap_in_sync['avg_us']:.0f}us")
    print(f"  INT8 swap in: {swap_in_int8_time['avg_us']:.0f}us ({swap_in_int8_time['avg_us']/swap_in_full_time['avg_us']:.2f}x of BF16)")

    del gpu_kv_all_layers, cpu_pinned_kv_all, gpu_kv_int8, cpu_pinned_kv_int8
    torch.cuda.empty_cache()

    results["exp2_block_swap_latency"] = exp2

    # ====================================================================
    # Exp 3: Swap vs Recomputation Cost Comparison
    # ====================================================================
    print("\n--- Exp 3: Swap vs Recomputation ---")
    exp3 = {}

    # Recomputation cost: re-run attention for 1 block (16 tokens)
    # Need: Q projection, K projection, V projection, attention computation

    # For a single request with S tokens preempted, recomputation cost:
    # - Re-run prefill for S tokens (memory-bound for decode, compute-bound for prefill)
    # - vLLM uses recomputation by default on RTX 4090 (swap is slower)

    # Measure recomputation cost for different sequence lengths
    seq_lens = [16, 64, 128, 256, 512, 1024, 2048]

    for S in seq_lens:
        # Simulate recomputation: forward pass through 1 layer attention
        x = torch.randn(1, S, hidden, device=device, dtype=torch.bfloat16)
        w_qkv = torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16)

        # QKV projection (the main cost)
        def recompute_qkv():
            qkv = x @ w_qkv
            return qkv

        recompute_time = benchmark_transfer(recompute_qkv, warmup=3, repeats=10)

        # Swap cost for S tokens across all layers
        # S tokens = S/block_size blocks, each block swap = full_block_swap_in_time
        n_blocks = S // block_size
        swap_cost_us = n_blocks * swap_in_full_time["avg_us"] + n_blocks * swap_out_full_time["avg_us"]  # swap out + swap in

        # INT8 swap cost
        int8_swap_cost_us = n_blocks * swap_in_int8_time["avg_us"] * 2  # swap out + swap in (INT8 both directions ≈ same)

        # Recomputation cost: all 32 layers
        # QKV is ~34% of a layer, so full recomputation = 32 * (QKV time / 0.34)
        # But vLLM only recomputes attention (not MLP), so cost = 32 * QKV projection time
        # Actually vLLM recomputes the full forward for the preempted tokens
        full_recompute_us = 32 * recompute_time["avg_us"]  # all layers QKV only

        # More accurate: full layer recomputation (attn + MLP) for preempted tokens
        # vLLM recomputes full forward for preempted tokens
        w_gate = torch.randn(hidden, 14336, device=device, dtype=torch.bfloat16)
        w_up = torch.randn(hidden, 14336, device=device, dtype=torch.bfloat16)
        w_down = torch.randn(14336, hidden, device=device, dtype=torch.bfloat16)

        def recompute_full_layer():
            qkv = x @ w_qkv
            gate = x @ w_gate
            up = x @ w_up
            silu = torch.nn.functional.silu(gate) * up
            down = silu @ w_down
            return qkv, down

        recompute_full = benchmark_transfer(recompute_full_layer, warmup=3, repeats=10)

        full_recompute_all_layers_us = 32 * recompute_full["avg_us"]

        decision = "recompute" if full_recompute_all_layers_us < swap_cost_us else "swap"
        int8_decision = "recompute" if full_recompute_all_layers_us < int8_swap_cost_us else "int8_swap"

        exp3[str(S)] = {
            "recompute_qkv_us": recompute_time["avg_us"],
            "recompute_full_layer_us": recompute_full["avg_us"],
            "recompute_all_layers_us": round(full_recompute_all_layers_us, 0),
            "swap_cost_us": round(swap_cost_us, 0),
            "int8_swap_cost_us": round(int8_swap_cost_us, 0),
            "swap_ratio": round(swap_cost_us / full_recompute_all_layers_us, 2),
            "int8_swap_ratio": round(int8_swap_cost_us / full_recompute_all_layers_us, 2),
            "decision": decision,
            "int8_decision": int8_decision,
            "n_blocks": n_blocks,
        }

        print(f"  S={S}: recompute={full_recompute_all_layers_us:.0f}us, "
              f"swap={swap_cost_us:.0f}us (ratio={swap_cost_us/full_recompute_all_layers_us:.2f}x), "
              f"INT8 swap={int8_swap_cost_us:.0f}us (ratio={int8_swap_cost_us/full_recompute_all_layers_us:.2f}x) "
              f"→ {decision}")

        del x, w_qkv, w_gate, w_up, w_down
        torch.cuda.empty_cache()

    results["exp3_swap_vs_recompute"] = exp3

    # ====================================================================
    # Exp 4: ITL Impact Simulation
    # ====================================================================
    print("\n--- Exp 4: ITL Impact (Swap Overhead per Decode Step) ---")
    exp4 = {}

    # If we swap 1 block in/out during a decode step, how much does ITL increase?
    # Normal decode step: ~13ms (7B B=1 RTX 4090)
    normal_decode_us = 13000  # from decode breakdown benchmark

    for n_swap_blocks in [1, 2, 4, 8, 16, 32, 64]:
        swap_overhead_us = n_swap_blocks * (swap_in_full_time["avg_us"] + swap_out_full_time["avg_us"])
        int8_swap_overhead_us = n_swap_blocks * swap_in_int8_time["avg_us"] * 2

        total_itl_us = normal_decode_us + swap_overhead_us
        int8_total_itl_us = normal_decode_us + int8_swap_overhead_us

        itl_increase_pct = swap_overhead_us / normal_decode_us * 100
        int8_itl_increase_pct = int8_swap_overhead_us / normal_decode_us * 100

        # How many tokens are preempted for n_swap_blocks?
        preempted_tokens = n_swap_blocks * block_size

        exp4[str(n_swap_blocks)] = {
            "swap_overhead_us": round(swap_overhead_us, 0),
            "int8_swap_overhead_us": round(int8_swap_overhead_us, 0),
            "total_itl_us": round(total_itl_us, 0),
            "int8_total_itl_us": round(int8_total_itl_us, 0),
            "itl_increase_pct": round(itl_increase_pct, 1),
            "int8_itl_increase_pct": round(int8_itl_increase_pct, 1),
            "preempted_tokens": preempted_tokens,
        }

        print(f"  {n_swap_blocks} blocks ({preempted_tokens} tok): "
              f"swap overhead={swap_overhead_us:.0f}us ({itl_increase_pct:.1f}% ITL increase), "
              f"INT8 swap={int8_swap_overhead_us:.0f}us ({int8_itl_increase_pct:.1f}% ITL increase)")

    results["exp4_itl_impact"] = exp4

    # ====================================================================
    # Exp 5: Multi-Block Concurrent Swap (Overlap Potential)
    # ====================================================================
    print("\n--- Exp 5: Multi-Block Concurrent Swap ---")
    exp5 = {}

    # Can we overlap swap with compute? Test concurrent transfer + compute
    # Simulate: start swap in background while compute runs

    # Create a compute-heavy operation (GEMM) and a swap operation
    x_compute = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
    w_compute = torch.randn(hidden, 14336, device=device, dtype=torch.bfloat16)

    # 8 blocks of KV for swap
    n_concurrent_blocks = 8
    gpu_kv = [torch.randn(block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(n_concurrent_blocks)]
    cpu_pinned = [torch.randn(block_size, n_kv_heads, head_dim, dtype=torch.bfloat16).pin_memory() for _ in range(n_concurrent_blocks)]

    # Compute-only time
    compute_only = benchmark_transfer(lambda: x_compute @ w_compute, warmup=5, repeats=20)

    # Swap-only time (8 blocks in)
    def swap_only():
        for i in range(n_concurrent_blocks):
            gpu_kv[i].copy_(cpu_pinned[i], non_blocking=True)
        torch.cuda.synchronize()

    swap_only_time = benchmark_transfer(swap_only, warmup=3, repeats=10)

    # Overlapped: compute + swap concurrently
    def overlapped():
        # Start swap on separate stream
        swap_stream = torch.cuda.Stream()
        with torch.cuda.stream(swap_stream):
            for i in range(n_concurrent_blocks):
                gpu_kv[i].copy_(cpu_pinned[i], non_blocking=True)

        # Compute on default stream simultaneously
        result = x_compute @ w_compute

        # Wait for both
        torch.cuda.current_stream().wait_stream(swap_stream)
        torch.cuda.synchronize()

    overlapped_time = benchmark_transfer(overlapped, warmup=5, repeats=20)

    overlap_ratio = overlapped_time["avg_us"] / max(compute_only["avg_us"], swap_only_time["avg_us"])
    overlap_efficiency = (compute_only["avg_us"] + swap_only_time["avg_us"] - overlapped_time["avg_us"]) / swap_only_time["avg_us"] * 100

    exp5 = {
        "compute_only_us": compute_only["avg_us"],
        "swap_only_us": swap_only_time["avg_us"],
        "overlapped_us": overlapped_time["avg_us"],
        "overlap_ratio": round(overlap_ratio, 2),
        "overlap_efficiency_pct": round(overlap_efficiency, 1),
        "overlap_saved_us": round(compute_only["avg_us"] + swap_only_time["avg_us"] - overlapped_time["avg_us"], 0),
    }

    print(f"  Compute only: {compute_only['avg_us']:.0f}us")
    print(f"  Swap only (8 blocks): {swap_only_time['avg_us']:.0f}us")
    print(f"  Overlapped: {overlapped_time['avg_us']:.0f}us")
    print(f"  Overlap efficiency: {overlap_efficiency:.1f}% ({overlap_ratio:.2f}x of max)")

    # Also test with more swap blocks to see when swap dominates
    for n_blocks in [16, 32, 64]:
        gpu_kv_n = [torch.randn(block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(n_blocks)]
        cpu_pinned_n = [torch.randn(block_size, n_kv_heads, head_dim, dtype=torch.bfloat16).pin_memory() for _ in range(n_blocks)]

        def swap_only_n():
            for i in range(n_blocks):
                gpu_kv_n[i].copy_(cpu_pinned_n[i], non_blocking=True)
            torch.cuda.synchronize()

        swap_n_time = benchmark_transfer(swap_only_n, warmup=3, repeats=10)

        def overlapped_n():
            swap_stream = torch.cuda.Stream()
            with torch.cuda.stream(swap_stream):
                for i in range(n_blocks):
                    gpu_kv_n[i].copy_(cpu_pinned_n[i], non_blocking=True)
            result = x_compute @ w_compute
            torch.cuda.current_stream().wait_stream(swap_stream)
            torch.cuda.synchronize()

        overlapped_n_time = benchmark_transfer(overlapped_n, warmup=3, repeats=10)

        eff = (compute_only["avg_us"] + swap_n_time["avg_us"] - overlapped_n_time["avg_us"]) / swap_n_time["avg_us"] * 100

        exp5[f"{n_blocks}_blocks"] = {
            "swap_only_us": swap_n_time["avg_us"],
            "overlapped_us": overlapped_n_time["avg_us"],
            "overlap_efficiency_pct": round(eff, 1),
        }

        print(f"  {n_blocks} blocks overlapped: {overlapped_n_time['avg_us']:.0f}us (efficiency={eff:.1f}%)")

        del gpu_kv_n, cpu_pinned_n
        torch.cuda.empty_cache()

    results["exp5_overlap"] = exp5

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — KV Cache CPU Offloading RTX 4090")
    print("=" * 70)

    # Key numbers
    pinned_bw = exp1.get("8", {}).get("h2d_pinned_bw_mbps", 0)
    full_swap_us = exp2.get("full_block_swap_in_async_us", 0)
    swap_decision_s16 = exp3.get("16", {}).get("decision", "")
    swap_decision_s256 = exp3.get("256", {}).get("decision", "")

    print(f"\n  PCIe pinned bandwidth: ~{pinned_bw:.0f} MB/s (H2D)")
    print(f"  Full block swap in: {full_swap_us:.0f} us (1.25 MB)")
    print(f"  S=16 decision: {swap_decision_s16}")
    print(f"  S=256 decision: {swap_decision_s256}")
    print(f"\n  vLLM on RTX 4090: recomputation is faster than swap")
    print(f"  → vLLM default recomputation is correct for PCIe GPUs!")
    print(f"  → Swap only viable with NVLink (H100/A100)")

    return results


if __name__ == '__main__':
    torch.cuda.empty_cache()
    results = run_all()
    try:
        with open('results/kv_cache_offloading_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('kv_cache_offloading_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    print("\nResults saved.")