#!/usr/bin/env python3
"""CUDA Streams & Events 深度实验

实验:
1. CUDA Stream 并发: 计算重叠
2. CUDA Event 精确计时 vs time.time()
3. 多 Stream 并行 GEMM
4. 通信-计算重叠模拟
5. CUDA Graph 性能

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_cuda_streams.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
import torch.nn as nn
from collections import OrderedDict

GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU: {GPU_NAME}")
print(f"CUDA: {torch.version.cuda}")
print(f"Streams supported: {torch.cuda.get_device_properties(0).multi_processor_count} SMs")


# ============================================================
# 实验 1: CUDA Event vs time.time() 精度对比
# ============================================================

def exp1_timing_precision():
    print("\n" + "=" * 60)
    print("实验1: CUDA Event vs time.time() 精度")
    print("=" * 60)

    results = []
    N = 1024

    # 不同操作耗时对比
    ops = [
        ("empty kernel", lambda: torch.cuda._sleep(1)),
        ("small GEMM (64x64)", lambda: torch.mm(torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda"))),
        ("medium GEMM (512x512)", lambda: torch.mm(torch.randn(512, 512, device="cuda"), torch.randn(512, 512, device="cuda"))),
        ("large GEMM (2048x2048)", lambda: torch.mm(torch.randn(2048, 2048, device="cuda"), torch.randn(2048, 2048, device="cuda"))),
        ("memcpy 1MB", lambda: torch.randn(512*1024//4, device="cuda").clone()),
        ("memcpy 64MB", lambda: torch.randn(16*1024*1024, device="cuda").clone()),
        ("elementwise 1M", lambda: torch.sin(torch.randn(1024*1024, device="cuda"))),
    ]

    print(f"\n  {'Operation':<25} {'Event(ms)':<12} {'time.time(ms)':<15} {'Ratio'}")
    print("  " + "-" * 65)

    for name, op in ops:
        # Warmup
        for _ in range(5):
            op()
        torch.cuda.synchronize()

        # CUDA Event timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        rep = 100
        start_event.record()
        for _ in range(rep):
            op()
        end_event.record()
        torch.cuda.synchronize()
        event_ms = start_event.elapsed_time(end_event) / rep

        # time.time() timing
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(rep):
            op()
        torch.cuda.synchronize()
        wall_ms = (time.time() - t0) / rep * 1000

        ratio = wall_ms / event_ms if event_ms > 0 else float('inf')
        print(f"  {name:<25} {event_ms:<12.4f} {wall_ms:<15.4f} {ratio:.2f}x")
        results.append({
            "op": name, "event_ms": round(event_ms, 4),
            "wall_ms": round(wall_ms, 4), "ratio": round(ratio, 2),
        })

    return results


# ============================================================
# 实验 2: 多 Stream 并行 GEMM
# ============================================================

def exp2_multi_stream_gemm():
    print("\n" + "=" * 60)
    print("实验2: 多 Stream 并行 GEMM")
    print("=" * 60)

    results = []

    # 单 stream vs 多 stream 执行多个独立 GEMM
    N = 1024
    n_ops_list = [1, 2, 4, 8]

    print(f"\n  GEMM size: {N}x{N}")
    print(f"  {'n_ops':<8} {'1-stream(ms)':<14} {'n-stream(ms)':<14} {'Speedup':<10} {'Efficiency'}")
    print("  " + "-" * 60)

    for n_ops in n_ops_list:
        # Prepare data
        As = [torch.randn(N, N, device="cuda") for _ in range(n_ops)]
        Bs = [torch.randn(N, N, device="cuda") for _ in range(n_ops)]

        # Warmup
        for _ in range(3):
            for i in range(n_ops):
                torch.mm(As[i], Bs[i])
        torch.cuda.synchronize()

        rep = 20

        # Single stream (sequential)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            for i in range(n_ops):
                torch.mm(As[i], Bs[i])
        end.record()
        torch.cuda.synchronize()
        single_ms = start.elapsed_time(end) / rep

        # Multi stream (parallel)
        streams = [torch.cuda.Stream() for _ in range(n_ops)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            for i in range(n_ops):
                with torch.cuda.stream(streams[i]):
                    torch.mm(As[i], Bs[i])
        end.record()
        torch.cuda.synchronize()
        multi_ms = start.elapsed_time(end) / rep

        speedup = single_ms / multi_ms
        eff = speedup / n_ops * 100
        print(f"  {n_ops:<8} {single_ms:<14.2f} {multi_ms:<14.2f} {speedup:<10.2f} {eff:.1f}%")
        results.append({
            "n_ops": n_ops, "single_ms": round(single_ms, 2),
            "multi_ms": round(multi_ms, 2), "speedup": round(speedup, 2),
            "efficiency_pct": round(eff, 1),
        })

        del As, Bs, streams
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 3: 通信-计算重叠模拟 (memcpy + compute)
# ============================================================

def exp3_overlap_sim():
    print("\n" + "=" * 60)
    print("实验3: 通信-计算重叠模拟 (memcpy + GEMM)")
    print("=" * 60)

    results = []

    # 模拟: AllReduce通信=memcpy, 计算=GEMM
    # 重叠: 在 stream_A 做 memcpy 的同时, stream_B 做 GEMM
    copy_sizes_mb = [1, 4, 16, 64]
    gemm_sizes = [512, 1024, 2048]

    print(f"\n  模拟通信(PCIe memcpy) + 计算(GEMM) 重叠")
    print(f"  {'Copy MB':<10} {'GEMM N':<8} {'Compute ms':<12} {'Copy ms':<10} {'Overlap ms':<12} {'Overlap%'}")
    print("  " + "-" * 66)

    for copy_mb in copy_sizes_mb:
        for gemm_n in gemm_sizes:
            n_elements = copy_mb * 1024 * 1024 // 4
            copy_data = torch.randn(n_elements, device="cuda")
            A = torch.randn(gemm_n, gemm_n, device="cuda")
            B = torch.randn(gemm_n, gemm_n, device="cuda")

            # Warmup
            for _ in range(3):
                copy_data.clone()
                torch.mm(A, B)
            torch.cuda.synchronize()

            rep = 20

            # Compute only
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                C = torch.mm(A, B)
            end.record()
            torch.cuda.synchronize()
            compute_ms = start.elapsed_time(end) / rep

            # Copy only (simulate communication)
            start.record()
            for _ in range(rep):
                copy_data.clone()
            end.record()
            torch.cuda.synchronize()
            copy_ms = start.elapsed_time(end) / rep

            # Overlapped (different streams)
            stream_compute = torch.cuda.Stream()
            stream_copy = torch.cuda.Stream()
            start.record()
            for _ in range(rep):
                with torch.cuda.stream(stream_compute):
                    C = torch.mm(A, B)
                with torch.cuda.stream(stream_copy):
                    copy_data.clone()
            end.record()
            torch.cuda.synchronize()
            overlap_ms = start.elapsed_time(end) / rep

            # Overlap ratio: how much of the copy was hidden?
            sequential_ms = compute_ms + copy_ms
            overlap_pct = (sequential_ms - overlap_ms) / copy_ms * 100
            overlap_pct = max(0, min(100, overlap_pct))

            print(f"  {copy_mb:<10} {gemm_n:<8} {compute_ms:<12.2f} {copy_ms:<10.2f} {overlap_ms:<12.2f} {overlap_pct:.1f}%")
            results.append({
                "copy_mb": copy_mb, "gemm_n": gemm_n,
                "compute_ms": round(compute_ms, 2), "copy_ms": round(copy_ms, 2),
                "overlap_ms": round(overlap_ms, 2), "overlap_pct": round(overlap_pct, 1),
            })

            del copy_data, A, B
            torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 4: CUDA Graph 性能 (减少 kernel launch 开销)
# ============================================================

def exp4_cuda_graph():
    print("\n" + "=" * 60)
    print("实验4: CUDA Graph 性能 (kernel launch 开销)")
    print("=" * 60)

    results = []

    # 小操作的 graph 优化效果
    N = 512
    model = torch.nn.Sequential(
        torch.nn.Linear(N, N, bias=False).cuda(),
        torch.nn.ReLU(),
        torch.nn.Linear(N, N, bias=False).cuda(),
    ).half()

    x = torch.randn(1, 64, N, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        model(x)
    torch.cuda.synchronize()

    rep = 100

    # Normal execution
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        y = model(x)
    end.record()
    torch.cuda.synchronize()
    normal_ms = start.elapsed_time(end) / rep

    # CUDA Graph capture
    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = model(static_x)

    # Graph replay
    start.record()
    for _ in range(rep):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph_ms = start.elapsed_time(end) / rep

    speedup = normal_ms / graph_ms
    launch_saved = normal_ms - graph_ms

    print(f"\n  Model: 2-layer MLP ({N}), batch=1x64")
    print(f"  Normal:     {normal_ms:.4f} ms")
    print(f"  CUDA Graph: {graph_ms:.4f} ms")
    print(f"  Speedup:    {speedup:.2f}x (saved {launch_saved:.4f} ms/iter)")

    results.append({
        "model": f"2-layer MLP {N}",
        "normal_ms": round(normal_ms, 4),
        "graph_ms": round(graph_ms, 4),
        "speedup": round(speedup, 2),
        "launch_saved_ms": round(launch_saved, 4),
    })

    # Varying operation counts
    print(f"\n  操作数量 vs Graph 加速:")
    print(f"  {'n_layers':<10} {'Normal ms':<12} {'Graph ms':<12} {'Speedup':<10} {'Launch/layer'}")
    print("  " + "-" * 56)

    for n_layers in [2, 4, 8, 16]:
        layers = []
        for _ in range(n_layers):
            layers.extend([
                torch.nn.Linear(N, N, bias=False).cuda().half(),
                torch.nn.ReLU(),
            ])
        model_n = torch.nn.Sequential(*layers)
        x = torch.randn(1, 64, N, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(5):
            model_n(x)
        torch.cuda.synchronize()

        # Normal
        start.record()
        for _ in range(rep):
            model_n(x)
        end.record()
        torch.cuda.synchronize()
        n_ms = start.elapsed_time(end) / rep

        # Graph
        static_x = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_y = model_n(static_x)

        start.record()
        for _ in range(rep):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        g_ms = start.elapsed_time(end) / rep

        sp = n_ms / g_ms
        launch_per_layer = (n_ms - g_ms) / n_layers
        print(f"  {n_layers:<10} {n_ms:<12.4f} {g_ms:<12.4f} {sp:<10.2f} {launch_per_layer:.4f} ms")

        results.append({
            "n_layers": n_layers,
            "normal_ms": round(n_ms, 4), "graph_ms": round(g_ms, 4),
            "speedup": round(sp, 2), "launch_per_layer_ms": round(launch_per_layer, 4),
        })

        del model_n, g
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 5: Stream 优先级 & 同步模式
# ============================================================

def exp5_stream_sync():
    print("\n" + "=" * 60)
    print("实验5: Stream 同步模式")
    print("=" * 60)

    results = []
    N = 2048
    n_ops = 4

    # Create streams
    streams = [torch.cuda.Stream() for _ in range(n_ops)]
    As = [torch.randn(N, N, device="cuda") for _ in range(n_ops)]
    Bs = [torch.randn(N, N, device="cuda") for _ in range(n_ops)]
    Cs = [torch.empty(N, N, device="cuda") for _ in range(n_ops)]

    # Warmup
    for i in range(n_ops):
        with torch.cuda.stream(streams[i]):
            torch.mm(As[i], Bs[i])
    torch.cuda.synchronize()

    rep = 50

    # Pattern 1: Launch all + sync once
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        for i in range(n_ops):
            with torch.cuda.stream(streams[i]):
                Cs[i] = torch.mm(As[i], Bs[i])
    end.record()
    torch.cuda.synchronize()
    launch_all_ms = start.elapsed_time(end) / rep

    # Pattern 2: Launch + sync each (simulate per-layer TP)
    start.record()
    for _ in range(rep):
        for i in range(n_ops):
            with torch.cuda.stream(streams[i]):
                Cs[i] = torch.mm(As[i], Bs[i])
            torch.cuda.synchronize()  # sync after each
    end.record()
    torch.cuda.synchronize()
    sync_each_ms = start.elapsed_time(end) / rep

    # Pattern 3: Launch + event wait (simulate NCCL-style)
    events = [torch.cuda.Event() for _ in range(n_ops)]
    start.record()
    for _ in range(rep):
        for i in range(n_ops):
            with torch.cuda.stream(streams[i]):
                Cs[i] = torch.mm(As[i], Bs[i])
                events[i].record(streams[i])
        # Wait all events on default stream
        for e in events:
            e.synchronize()
    end.record()
    torch.cuda.synchronize()
    event_wait_ms = start.elapsed_time(end) / rep

    print(f"\n  {n_ops}x GEMM ({N}x{N}), 不同同步模式:")
    print(f"  Launch all + sync once:  {launch_all_ms:.2f} ms (best)")
    print(f"  Launch + sync each:      {sync_each_ms:.2f} ms ({sync_each_ms/launch_all_ms:.2f}x)")
    print(f"  Launch + event wait:     {event_wait_ms:.2f} ms ({event_wait_ms/launch_all_ms:.2f}x)")

    results.append({
        "n_ops": n_ops, "gemm_size": N,
        "launch_all_ms": round(launch_all_ms, 2),
        "sync_each_ms": round(sync_each_ms, 2),
        "event_wait_ms": round(event_wait_ms, 2),
        "sync_each_ratio": round(sync_each_ms / launch_all_ms, 2),
        "event_wait_ratio": round(event_wait_ms / launch_all_ms, 2),
    })

    del As, Bs, Cs, streams, events
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CUDA Streams & Events 实验")
    print("=" * 60)

    all_results = OrderedDict()
    all_results["timing_precision"] = exp1_timing_precision()
    all_results["multi_stream_gemm"] = exp2_multi_stream_gemm()
    all_results["overlap_sim"] = exp3_overlap_sim()
    all_results["cuda_graph"] = exp4_cuda_graph()
    all_results["stream_sync"] = exp5_stream_sync()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. CUDA Event 比 time.time() 更精确 (GPU时间 vs 墙钟时间)
  2. 多 Stream 并行: 小 GEMM 加速明显, 大 GEMM 受 SM 限制
  3. 通信-计算重叠: 计算时间 > 通信时间时, 通信几乎完全隐藏
  4. CUDA Graph: 减少 kernel launch 开销, 小操作加速最大
  5. 同步模式: Launch-all-sync-once 最优, 逐个同步最差
""")

    with open("/root/cuda_streams_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to /root/cuda_streams_results.json")
