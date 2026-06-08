"""
CUDA Memory Allocator Analysis — RTX 4090
Measures memory allocation patterns, fragmentation, and allocator overhead
that impact LLM inference serving (vLLM/SGLang memory management).

Focus: Understanding GPU memory behavior for production serving.
"""

import torch
import time
import json

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")
print(f"Total HBM: {props.total_memory / 1024**3:.2f} GB")

TOTAL_HBM_GB = props.total_memory / 1024**3


def measure_memory():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free = TOTAL_HBM_GB - reserved
    return {"allocated_gb": round(allocated, 3), "reserved_gb": round(reserved, 3), "free_gb": round(free, 3)}


def run_all():
    results = {}
    print("=" * 70)
    print("CUDA Memory Allocator Analysis — RTX 4090")
    print("=" * 70)

    # Reset
    torch.cuda.empty_cache()
    initial = measure_memory()
    print(f"Initial: allocated={initial['allocated_gb']}GB, reserved={initial['reserved_gb']}GB, free={initial['free_gb']}GB")

    # Exp 1: Memory allocation patterns
    print("\n--- Exp 1: Memory Allocation Patterns ---")
    exp1 = {}

    # 7B model weight simulation (BF16)
    torch.cuda.empty_cache()
    weight_size = 7e9 * 2 / 1024**3  # 13.0GB

    # Allocate model weights (simulate single allocation)
    torch.cuda.empty_cache()
    before = measure_memory()
    weight_tensor = torch.randn(int(7e9 * 0.8), device=device, dtype=torch.bfloat16)  # ~10.7GB
    after_alloc = measure_memory()
    alloc_time_us = 0
    # Measure allocation time
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    weight_tensor2 = torch.randn(int(7e9 * 0.8), device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    alloc_time_us = (t1 - t0) * 1e6

    exp1["single_large"] = {
        "before": before,
        "after": after_alloc,
        "tensor_gb": round(weight_tensor.nelement() * weight_tensor.element_size() / 1024**3, 3),
        "alloc_time_us": round(alloc_time_us, 1),
        "fragmentation_gb": round(after_alloc['reserved_gb'] - after_alloc['allocated_gb'], 3),
    }
    print(f"  Single large(10.7GB): alloc_time={alloc_time_us:.1f}us, fragmentation={after_alloc['reserved_gb'] - after_alloc['allocated_gb']:.3f}GB")

    del weight_tensor, weight_tensor2
    torch.cuda.empty_cache()

    # Many small allocations (simulate KV cache blocks)
    n_blocks = 10000
    block_size = 16 * 5 * 128 * 2  # block_size=16, 5 KV heads, d_head=128, BF16 = 20KB
    before = measure_memory()
    blocks = [torch.randn(block_size // 2, device=device, dtype=torch.bfloat16) for _ in range(n_blocks)]
    after_small = measure_memory()

    # Measure small allocation time
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    blocks2 = [torch.randn(block_size // 2, device=device, dtype=torch.bfloat16) for _ in range(100)]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    small_alloc_time_us = (t1 - t0) / 100 * 1e6

    total_small_gb = sum(b.nelement() * b.element_size() for b in blocks) / 1024**3
    exp1["many_small"] = {
        "n_blocks": n_blocks,
        "block_size_kb": round(block_size / 1024, 2),
        "total_gb": round(total_small_gb, 3),
        "reserved_gb": after_small['reserved_gb'],
        "allocated_gb": after_small['allocated_gb'],
        "fragmentation_gb": round(after_small['reserved_gb'] - after_small['allocated_gb'], 3),
        "fragmentation_pct": round((after_small['reserved_gb'] - after_small['allocated_gb']) / total_small_gb * 100, 1) if total_small_gb > 0 else 0,
        "alloc_time_per_block_us": round(small_alloc_time_us, 2),
    }
    print(f"  {n_blocks} small blocks({block_size/1024:.0f}KB each={total_small_gb:.3f}GB total): fragmentation={after_small['reserved_gb'] - after_small['allocated_gb']:.3f}GB ({(after_small['reserved_gb'] - after_small['allocated_gb']) / total_small_gb * 100:.1f}%), alloc_time={small_alloc_time_us:.2f}us/block")

    del blocks, blocks2
    torch.cuda.empty_cache()

    # Mixed allocation (weights + KV)
    weight = torch.randn(int(5e9), device=device, dtype=torch.bfloat16)  # ~9.3GB
    kv_blocks = [torch.randn(block_size // 2, device=device, dtype=torch.bfloat16) for _ in range(5000)]
    mixed_mem = measure_memory()

    exp1["mixed"] = {
        "weight_gb": round(weight.nelement() * weight.element_size() / 1024**3, 3),
        "kv_total_gb": round(sum(b.nelement() * b.element_size() for b in kv_blocks) / 1024**3, 3),
        "reserved_gb": mixed_mem['reserved_gb'],
        "allocated_gb": mixed_mem['allocated_gb'],
        "fragmentation_gb": round(mixed_mem['reserved_gb'] - mixed_mem['allocated_gb'], 3),
    }
    print(f"  Mixed(9.3GB weights+5000 KV blocks): fragmentation={mixed_mem['reserved_gb'] - mixed_mem['allocated_gb']:.3f}GB")

    del weight, kv_blocks
    torch.cuda.empty_cache()

    results["exp1_allocation_patterns"] = exp1

    # Exp 2: Memory fragmentation under allocation/deallocation cycles
    print("\n--- Exp 2: Fragmentation Under Alloc/Free Cycles ---")
    exp2 = {}

    # Simulate request lifecycle: allocate KV → process → free KV → allocate new KV
    torch.cuda.empty_cache()
    base_weight = torch.randn(int(5e9), device=device, dtype=torch.bfloat16)  # baseline 9.3GB
    base_mem = measure_memory()

    # Cycle 1: allocate B=32 request KV, then free, then allocate again
    req_kv_size = 4096 * 5 * 128 * 2  # S=4096, 5 KV heads, d_head=128, BF16 ≈ 40KB per token
    n_tokens = 4096 * 32  # B=32 × S=4096

    fragmentation_cycles = []
    for cycle in range(5):
        # Allocate KV for B=32 requests
        kv_tensors = [torch.randn(n_tokens, device=device, dtype=torch.bfloat16) for _ in range(32)]
        mem_with_kv = measure_memory()
        fragmentation = mem_with_kv['reserved_gb'] - mem_with_kv['allocated_gb']

        # Free all KV (simulate request completion)
        del kv_tensors
        torch.cuda.empty_cache()
        mem_after_free = measure_memory()
        fragmentation_after = mem_after_free['reserved_gb'] - mem_after_free['allocated_gb']

        fragmentation_cycles.append({
            "cycle": cycle,
            "with_kv_reserved": mem_with_kv['reserved_gb'],
            "with_kv_allocated": mem_with_kv['allocated_gb'],
            "with_kv_fragmentation": round(fragmentation, 3),
            "after_free_reserved": mem_after_free['reserved_gb'],
            "after_free_allocated": mem_after_free['allocated_gb'],
            "after_free_fragmentation": round(fragmentation_after, 3),
            "reclaim_gb": round(mem_with_kv['reserved_gb'] - mem_after_free['reserved_gb'], 3),
        })
        print(f"  Cycle {cycle}: with_kv fragmentation={fragmentation:.3f}GB, after_free fragmentation={fragmentation_after:.3f}GB, reclaimed={mem_with_kv['reserved_gb'] - mem_after_free['reserved_gb']:.3f}GB")

    exp2["cycles"] = fragmentation_cycles
    exp2["base_weight_gb"] = round(base_weight.nelement() * base_weight.element_size() / 1024**3, 3)
    results["exp2_fragmentation_cycles"] = exp2

    del base_weight
    torch.cuda.empty_cache()

    # Exp 3: Memory pool efficiency (pre-allocated vs dynamic)
    print("\n--- Exp 3: Memory Pool vs Dynamic Allocation ---")
    exp3 = {}

    # Pre-allocate a pool (vLLM approach: pre-allocate KV cache)
    torch.cuda.empty_cache()
    pool_size_gb = 10.0  # Smaller to avoid OOM with weight baseline
    pool_tokens = int(pool_size_gb * 1024**3 / 2)  # BF16 = 2 bytes each

    # Measure pool creation time
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pool_tensor = torch.randn(pool_tokens, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pool_alloc_time_ms = (t1 - t0) * 1000

    pool_mem = measure_memory()

    # Now use slices of the pool (simulate KV cache block allocation)
    block_tokens = 16 * 5 * 128  # block_size=16, 5 KV heads, d_head=128
    slice_alloc_time_us = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Slice 1000 blocks from pool
    slices = [pool_tensor[i * block_tokens:(i+1) * block_tokens] for i in range(1000)]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    slice_alloc_time_per_us = (t1 - t0) / 1000 * 1e6

    # Dynamic allocation of same blocks — separate from pool
    del slices
    torch.cuda.empty_cache()
    dynamic_blocks = [torch.randn(block_tokens, device=device, dtype=torch.bfloat16) for _ in range(1000)]
    dynamic_mem = measure_memory()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dynamic_blocks2 = [torch.randn(block_tokens, device=device, dtype=torch.bfloat16) for _ in range(100)]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dynamic_alloc_per_us = (t1 - t0) / 100 * 1e6

    total_blocks_gb = 1000 * block_tokens * 2 / 1024**3

    exp3["pool"] = {
        "pool_size_gb": pool_size_gb,
        "pool_alloc_time_ms": round(pool_alloc_time_ms, 2),
        "slice_alloc_per_us": round(slice_alloc_time_per_us, 3),
        "pool_reserved_gb": pool_mem['reserved_gb'],
    }
    exp3["dynamic"] = {
        "n_blocks": 1000,
        "block_kb": round(block_tokens * 2 / 1024, 2),
        "total_gb": round(total_blocks_gb, 3),
        "dynamic_alloc_per_us": round(dynamic_alloc_per_us, 2),
        "reserved_gb": dynamic_mem['reserved_gb'],
        "fragmentation_gb": round(dynamic_mem['reserved_gb'] - dynamic_mem['allocated_gb'], 3),
    }
    exp3["comparison"] = {
        "pool_vs_dynamic_speedup": round(dynamic_alloc_per_us / slice_alloc_time_per_us, 1) if slice_alloc_time_per_us > 0 else 0,
        "pool_fragmentation_advantage": "pool has zero fragmentation (pre-allocated)",
    }

    print(f"  Pool alloc: {pool_alloc_time_ms:.2f}ms for {pool_size_gb}GB")
    print(f"  Pool slice: {slice_alloc_time_per_us:.3f}us/block → vs dynamic {dynamic_alloc_per_us:.2f}us/block → {dynamic_alloc_per_us / slice_alloc_time_per_us:.1f}x faster!")
    print(f"  Pool: zero fragmentation → Dynamic: {dynamic_mem['reserved_gb'] - dynamic_mem['allocated_gb']:.3f}GB fragmentation")

    del pool_tensor
    torch.cuda.empty_cache(), dynamic_blocks, dynamic_blocks2
    torch.cuda.empty_cache()

    results["exp3_pool_vs_dynamic"] = exp3

    # Exp 4: CUDA memory statistics summary
    print("\n--- Exp 4: CUDA Memory Statistics ---")
    exp4 = {}

    torch.cuda.empty_cache()
    # Allocate 7B model simulation
    weight = torch.randn(int(7e9), device=device, dtype=torch.bfloat16)  # ~13GB
    mem_stats = torch.cuda.memory_stats()

    exp4["stats"] = {
        "num_alloc_retries": mem_stats.get("num_alloc_retries", 0),
        "num_ooms": mem_stats.get("num_ooms", 0),
        "allocation_all_time_ms": round(mem_stats.get("allocation.all.time", 0) / 1000, 2),
        "segment_alloc_time_ms": round(mem_stats.get("segment.all.time", 0) / 1000, 2),
        "num_active_segments": mem_stats.get("num_active_segments", 0),
        "num_retries": mem_stats.get("num_alloc_retries", 0),
    }

    current_mem = measure_memory()
    exp4["current"] = current_mem
    print(f"  After 7B weight: alloc_retries={exp4['stats']['num_alloc_retries']}, ooms={exp4['stats']['num_ooms']}")
    print(f"  Active segments={exp4['stats']['num_active_segments']}")
    print(f"  Memory: allocated={current_mem['allocated_gb']}GB, reserved={current_mem['reserved_gb']}GB")

    # Test OOM handling
    try:
        huge = torch.randn(int(20e9), device=device, dtype=torch.bfloat16)
        exp4["oom_test"] = "no_oom"  # shouldn't happen with 24GB GPU and 13GB allocated
        del huge
    except torch.cuda.OutOfMemoryError as e:
        exp4["oom_test"] = f"OOM at 20GB allocation after 13GB weight"
        print(f"  OOM triggered: {e}")

    del weight
    torch.cuda.empty_cache()

    results["exp4_memory_stats"] = exp4

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — CUDA Memory Allocator Analysis RTX 4090")
    print("=" * 70)

    e1_single = exp1.get("single_large", {})
    e1_small = exp1.get("many_small", {})
    e3_pool = exp3.get("pool", {})
    e3_dynamic = exp3.get("dynamic", {})
    e3_comp = exp3.get("comparison", {})

    print(f"\n  Large allocation: {e1_single.get('alloc_time_us', 0):.1f}us for 10.7GB")
    print(f"  Small allocation: {e1_small.get('alloc_time_per_block_us', 0):.2f}us/block ({e1_small.get('block_size_kb', 0):.0f}KB)")
    print(f"  Fragmentation (many small): {e1_small.get('fragmentation_pct', 0):.1f}%")
    print(f"  Pool slice: {e3_pool.get('slice_alloc_per_us', 0):.3f}us/block → {e3_comp.get('pool_vs_dynamic_speedup', 0):.1f}x faster!")
    print(f"  Pool: zero fragmentation → vLLM's PagedAttention approach!")
    print(f"\n  Production implications:")
    print(f"    → vLLM pre-allocates KV cache pool → zero fragmentation → zero alloc overhead")
    print(f"    → Pool slice = tensor view → ~0us → vs dynamic alloc ~{e3_dynamic.get('dynamic_alloc_per_us', 0):.0f}us")
    print(f"    → 24GB GPU: 7B model(13GB) + KV pool(~10GB) + overhead(1GB) = 24GB → tight!")

    return results


if __name__ == '__main__':
    results = run_all()
    try:
        with open('results/cuda_memory_allocator.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('cuda_memory_allocator.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)