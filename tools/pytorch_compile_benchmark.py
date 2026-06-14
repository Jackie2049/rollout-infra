#!/usr/bin/env python3
"""PyTorch Compile Benchmark Script.

Benchmarks torch.compile modes (reduce-overhead/max-autotune/default)
against eager mode for a small transformer-like model.

Runs on:
  - Mac (MPS backend, eager only — compile may fail due to C++ compiler)
  - GPU server (CUDA backend, all compile modes)

Usage:
  python3 tools/pytorch_compile_benchmark.py [--device cpu|mps|cuda] [--compile]

Output: CSV results + summary table saved to results/
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


class MiniTransformer(nn.Module):
    """Minimal transformer-like model for benchmarking.
    Mimics the key ops in a real LLM: Linear+ReLU+Linear (MLP block).
    """

    def __init__(self, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # gate/up proj
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),  # down proj
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # residual connection
        x = self.norm(x)
        return self.head(x)


def benchmark_eager(model, inputs, device, n_warmup=5, n_iters=50):
    """Benchmark eager mode."""
    model = model.to(device)
    inputs = inputs.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mode": "eager",
        "mean_ms": sum(times) / len(times) * 1000,
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5 * 1000,
        "n_iters": n_iters,
    }


def benchmark_compile(model, inputs, device, mode="reduce-overhead", n_warmup=10, n_iters=50):
    """Benchmark torch.compile mode."""
    model = model.to(device)
    inputs = inputs.to(device)

    # Compile
    compile_start = time.perf_counter()
    try:
        compiled = torch.compile(model, mode=mode)
    except Exception as e:
        return {"mode": f"compile-{mode}", "error": str(e)[:100], "compile_time_s": -1}

    compile_time = time.perf_counter() - compile_start

    # First run (includes compilation overhead)
    first_start = time.perf_counter()
    try:
        _ = compiled(inputs)
    except Exception as e:
        return {
            "mode": f"compile-{mode}",
            "error": str(e)[:100],
            "compile_time_s": compile_time,
        }
    if device.type == "cuda":
        torch.cuda.synchronize()
    first_time = time.perf_counter() - first_start

    # Warmup (post-compilation)
    for _ in range(n_warmup):
        try:
            _ = compiled(inputs)
        except Exception as e:
            return {"mode": f"compile-{mode}", "error": str(e)[:100], "compile_time_s": compile_time}
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            _ = compiled(inputs)
        except Exception as e:
            return {"mode": f"compile-{mode}", "error": str(e)[:100], "compile_time_s": compile_time}
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mode": f"compile-{mode}",
        "mean_ms": sum(times) / len(times) * 1000,
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5 * 1000,
        "n_iters": n_iters,
        "compile_time_s": compile_time,
        "first_run_s": first_time,
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch compile benchmark")
    parser.add_argument("--device", default="auto", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--compile", action="store_true", help="Run compile benchmarks")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-iters", type=int, default=50)
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Model
    model = MiniTransformer(d_model=args.d_model, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MiniTransformer(d={args.d_model}, layers={args.n_layers})")
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f}MB FP32)")

    # Inputs
    inputs = torch.randn(args.batch_size, args.d_model)

    # Eager benchmark
    print("\n--- Eager benchmark ---")
    eager_result = benchmark_eager(model, inputs, device, n_iters=args.n_iters)
    print(f"  Mean: {eager_result['mean_ms']:.2f}ms")
    print(f"  Median: {eager_result['median_ms']:.2f}ms")

    results = [eager_result]

    # Compile benchmarks
    if args.compile:
        for mode in ["reduce-overhead", "default"]:
            print(f"\n--- Compile ({mode}) benchmark ---")
            compile_result = benchmark_compile(model, inputs, device, mode=mode, n_iters=args.n_iters)
            if "error" in compile_result:
                print(f"  ERROR: {compile_result['error']}")
                results.append(compile_result)
            else:
                print(f"  Compile time: {compile_result['compile_time_s']:.2f}s")
                print(f"  First run: {compile_result['first_run_s']:.2f}s")
                print(f"  Mean: {compile_result['mean_ms']:.2f}ms")
                print(f"  Median: {compile_result['median_ms']:.2f}ms")
                speedup = eager_result["mean_ms"] / compile_result["mean_ms"]
                print(f"  Speedup vs eager: {speedup:.2f}x")
                results.append(compile_result)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "model": f"MiniTransformer(d={args.d_model}, layers={args.n_layers})",
        "n_params": n_params,
        "batch_size": args.batch_size,
        "results": results,
    }

    output_path = os.path.join(OUTPUT_DIR, "pytorch_compile_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"| Mode | Mean(ms) | Median(ms) | Speedup |")
    print(f"|------|----------|------------|---------|")
    for r in results:
        if "error" in r:
            print(f"| {r['mode']} | ERROR | - | - |")
        else:
            speedup = eager_result["mean_ms"] / r["mean_ms"] if r["mode"] != "eager" else 1.0
            print(f"| {r['mode']} | {r['mean_ms']:.2f} | {r['median_ms']:.2f} | {speedup:.2f}x |")


if __name__ == "__main__":
    main()
