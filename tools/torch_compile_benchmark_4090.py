#!/usr/bin/env python3
"""torch.compile Benchmark — RTX 4090

Benchmarks torch.compile (PyTorch 2.x) performance on RTX 4090:
1. Forward pass: compiled vs eager (7M, 25M, 7B-proxy models)
2. Training step: compiled vs eager (7M, 25M models)
3. Compile modes comparison (default/reduce-overhead/max-autotune)
4. Dynamic shapes (variable batch/seq)
5. Memory overhead analysis

Goal: Validate torch.compile speedup claims with real RTX 4090 data.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u tools/torch_compile_benchmark_4090.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleTransformer(nn.Module):
    """Simple transformer for benchmarking — no flash attention dependency."""

    def __init__(self, hidden=512, layers=4, heads=8, vocab=32000, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4*hidden,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x)


def measure_time(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return np.median(times), np.mean(times), np.std(times)


MODEL_CONFIGS = {
    "7M": {"hidden": 512, "layers": 4, "heads": 8, "vocab": 32000},
    "25M": {"hidden": 1024, "layers": 8, "heads": 16, "vocab": 32000},
    "7B-proxy": {"hidden": 2560, "layers": 4, "heads": 20, "vocab": 32000},
}


def benchmark_forward(model_name, config, batch_sizes=[1, 4, 16, 32], seq_len=128):
    """Benchmark forward pass: compiled vs eager."""
    print(f"\n=== Forward Pass: {model_name} ===")

    model = SimpleTransformer(**config, seq_len=seq_len).to("cuda:0").to(torch.bfloat16)
    model.eval()

    # Compile with default mode (max-autotune for best performance)
    compiled_model = torch.compile(model, mode="max-autotune")

    results = []
    for B in batch_sizes:
        input_ids = torch.randint(0, config["vocab"], (B, seq_len), device="cuda:0")

        # Warmup compiled model (first run triggers compilation)
        with torch.no_grad():
            compiled_model(input_ids)
        torch.cuda.synchronize()

        # Eager
        def eager_fwd():
            with torch.no_grad():
                return model(input_ids)

        # Compiled
        def compiled_fwd():
            with torch.no_grad():
                return compiled_model(input_ids)

        eager_time = measure_time(eager_fwd, warmup=5, repeat=20)
        compiled_time = measure_time(compiled_fwd, warmup=5, repeat=20)

        speedup = eager_time[0] / compiled_time[0]
        results.append({
            "batch": B,
            "eager_ms": round(eager_time[0], 4),
            "compiled_ms": round(compiled_time[0], 4),
            "speedup": round(speedup, 2),
        })
        print(f"  B={B}: eager={eager_time[0]:.3f}ms, compiled={compiled_time[0]:.3f}ms -> {speedup:.2f}x")

    del model, compiled_model
    torch.cuda.empty_cache()
    return results


def benchmark_training(model_name, config, batch_sizes=[4, 16, 32], seq_len=128):
    """Benchmark training step: compiled vs eager."""
    print(f"\n=== Training Step: {model_name} ===")

    model = SimpleTransformer(**config, seq_len=seq_len).to("cuda:0").to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    compiled_model = torch.compile(model, mode="max-autotune")
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)

    results = []
    for B in batch_sizes:
        input_ids = torch.randint(0, config["vocab"], (B, seq_len), device="cuda:0")
        targets = torch.randint(0, config["vocab"], (B, seq_len), device="cuda:0")

        # Warmup compiled
        compiled_optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(compiled_model(input_ids).view(-1, config["vocab"]),
                                     targets.view(-1))
        loss.backward()
        compiled_optimizer.step()
        torch.cuda.synchronize()

        # Eager training step
        def eager_train():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(input_ids).view(-1, config["vocab"]),
                                         targets.view(-1))
            loss.backward()
            optimizer.step()

        # Compiled training step
        def compiled_train():
            compiled_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(compiled_model(input_ids).view(-1, config["vocab"]),
                                         targets.view(-1))
            loss.backward()
            compiled_optimizer.step()

        eager_time = measure_time(eager_train, warmup=3, repeat=10)
        compiled_time = measure_time(compiled_train, warmup=3, repeat=10)

        speedup = eager_time[0] / compiled_time[0]
        results.append({
            "batch": B,
            "eager_ms": round(eager_time[0], 4),
            "compiled_ms": round(compiled_time[0], 4),
            "speedup": round(speedup, 2),
        })
        print(f"  B={B}: eager={eager_time[0]:.3f}ms, compiled={compiled_time[0]:.3f}ms -> {speedup:.2f}x")

    del model, compiled_model, optimizer, compiled_optimizer
    torch.cuda.empty_cache()
    return results


def benchmark_compile_modes(model_name="7M", config=None, B=16, seq_len=128):
    """Benchmark different torch.compile modes."""
    if config is None:
        config = MODEL_CONFIGS[model_name]

    print(f"\n=== Compile Modes Comparison: {model_name} B={B} ===")

    modes = ["default", "reduce-overhead", "max-autotune"]
    input_ids = torch.randint(0, config["vocab"], (B, seq_len), device="cuda:0")

    # Eager baseline
    model_eager = SimpleTransformer(**config, seq_len=seq_len).to("cuda:0").to(torch.bfloat16)
    model_eager.eval()
    def eager_fwd():
        with torch.no_grad():
            return model_eager(input_ids)
    eager_time = measure_time(eager_fwd, warmup=5, repeat=20)
    print(f"  eager: {eager_time[0]:.3f}ms (baseline)")

    results = [{"mode": "eager", "time_ms": round(eager_time[0], 4), "speedup": 1.0}]

    for mode in modes:
        model = SimpleTransformer(**config, seq_len=seq_len).to("cuda:0").to(torch.bfloat16)
        model.eval()
        compiled = torch.compile(model, mode=mode)

        # Warmup (compilation happens here)
        with torch.no_grad():
            compiled(input_ids)
        torch.cuda.synchronize()

        def compiled_fwd():
            with torch.no_grad():
                return compiled(input_ids)

        t = measure_time(compiled_fwd, warmup=5, repeat=20)
        speedup = eager_time[0] / t[0]
        results.append({"mode": mode, "time_ms": round(t[0], 4), "speedup": round(speedup, 2)})
        print(f"  {mode}: {t[0]:.3f}ms -> {speedup:.2f}x")

        del model, compiled
        torch.cuda.empty_cache()

    return results


def benchmark_dynamic_shapes(model_name="7M", config=None):
    """Benchmark torch.compile with dynamic shapes (variable batch/seq)."""
    if config is None:
        config = MODEL_CONFIGS[model_name]

    print(f"\n=== Dynamic Shapes: {model_name} ===")

    model = SimpleTransformer(**config, seq_len=128).to("cuda:0").to(torch.bfloat16)
    model.eval()
    compiled = torch.compile(model, mode="max-autotune", dynamic=True)

    # Warmup with different shapes
    for B, S in [(4, 64), (16, 128), (32, 256)]:
        ids = torch.randint(0, config["vocab"], (B, S), device="cuda:0")
        with torch.no_grad():
            compiled(ids)
    torch.cuda.synchronize()

    test_shapes = [(4, 64), (8, 128), (16, 128), (32, 128), (16, 256)]
    results = []
    for B, S in test_shapes:
        input_ids = torch.randint(0, config["vocab"], (B, S), device="cuda:0")

        # Eager
        model_eager = SimpleTransformer(**config, seq_len=S).to("cuda:0").to(torch.bfloat16)
        model_eager.eval()
        def eager_fwd():
            with torch.no_grad():
                return model_eager(input_ids)
        eager_time = measure_time(eager_fwd, warmup=3, repeat=10)

        # Compiled (dynamic)
        def compiled_fwd():
            with torch.no_grad():
                return compiled(input_ids)
        compiled_time = measure_time(compiled_fwd, warmup=3, repeat=10)

        speedup = eager_time[0] / compiled_time[0]
        results.append({
            "batch": B, "seq_len": S,
            "eager_ms": round(eager_time[0], 4),
            "compiled_ms": round(compiled_time[0], 4),
            "speedup": round(speedup, 2),
        })
        print(f"  B={B} S={S}: eager={eager_time[0]:.3f}ms, compiled={compiled_time[0]:.3f}ms -> {speedup:.2f}x")

        del model_eager
        torch.cuda.empty_cache()

    del model, compiled
    torch.cuda.empty_cache()
    return results


def main():
    print(f"=== torch.compile Benchmark -- RTX 4090 ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    results = {
        "device": {
            "name": torch.cuda.get_device_name(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }

    # Section 1: Forward pass speedup by model size
    for model_name, config in MODEL_CONFIGS.items():
        if model_name == "7B-proxy":
            # Only test smaller batch sizes for 7B-proxy to avoid OOM
            results[f"forward_{model_name}"] = benchmark_forward(
                model_name, config, batch_sizes=[1, 4, 8])
        else:
            results[f"forward_{model_name}"] = benchmark_forward(model_name, config)

    # Section 2: Training step speedup
    for model_name in ["7M", "25M"]:
        results[f"training_{model_name}"] = benchmark_training(model_name, MODEL_CONFIGS[model_name])

    # Section 3: Compile modes comparison
    results["compile_modes"] = benchmark_compile_modes()

    # Section 4: Dynamic shapes
    results["dynamic_shapes"] = benchmark_dynamic_shapes()

    # Summary
    print("\n=== Summary ===")
    forward_speedups = []
    training_speedups = []
    for key, data in results.items():
        if key.startswith("forward_"):
            for r in data:
                forward_speedups.append(r["speedup"])
        if key.startswith("training_"):
            for r in data:
                training_speedups.append(r["speedup"])

    avg_fwd = np.mean(forward_speedups) if forward_speedups else 0
    avg_train = np.mean(training_speedups) if training_speedups else 0
    print(f"  Average forward speedup: {avg_fwd:.2f}x")
    print(f"  Average training speedup: {avg_train:.2f}x")
    print(f"  Best forward speedup: {max(forward_speedups):.2f}x")
    print(f"  Best training speedup: {max(training_speedups):.2f}x")

    results["summary"] = {
        "avg_forward_speedup": round(avg_fwd, 2),
        "avg_training_speedup": round(avg_train, 2),
        "best_forward_speedup": round(max(forward_speedups), 2),
        "best_training_speedup": round(max(training_speedups), 2),
    }

    # Memory analysis
    print("\n=== Memory Analysis ===")
    torch.cuda.reset_peak_memory_stats()
    model = SimpleTransformer(**MODEL_CONFIGS["7M"]).to("cuda:0").to(torch.bfloat16)
    input_ids = torch.randint(0, 32000, (16, 128), device="cuda:0")
    with torch.no_grad():
        model(input_ids)
    peak_eager = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    compiled = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        compiled(input_ids)
    peak_compiled = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Eager peak: {peak_eager:.1f}MB")
    print(f"  Compiled peak: {peak_compiled:.1f}MB")
    print(f"  Memory overhead: {(peak_compiled - peak_eager) / peak_eager * 100:.1f}%")

    results["memory"] = {
        "eager_peak_mb": round(peak_eager, 1),
        "compiled_peak_mb": round(peak_compiled, 1),
        "overhead_pct": round((peak_compiled - peak_eager) / peak_eager * 100, 1),
    }

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'results', 'torch_compile_benchmark.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()