#!/usr/bin/env python3
"""TransformerEngine FP8 Training Acceleration Benchmark — RTX 4090

Benchmarks FP8 vs BF16 training throughput on RTX 4090 (SM89).

Tests:
1. FP8 DelayedScaling vs BF16 throughput (GEMM-only + full model)
2. FP8 Float8CurrentScaling vs BF16 throughput
3. FP8 vs BF16 memory usage
4. FP8 accuracy: cos_sim comparison
5. FP8 scaling factor dynamics (DelayedScaling amax history)

Usage: python tools/te_fp8_training_benchmark_4090.py
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check TransformerEngine availability
try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("WARNING: transformer_engine not available. FP8 tests will be skipped.")
    print("Install: pip install transformer-engine")

# Check FlashAttention availability
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def get_device_info():
    """Get GPU device information."""
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_GB": props.total_memory / 1024**3,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
    }


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for benchmarking."""

    def __init__(self, hidden_size=2560, num_heads=20, num_kv_heads=5, seq_len=512, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.seq_len = seq_len

        # QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size + 2 * num_kv_heads * self.head_dim, bias=False)
        # Output projection
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # MLP: up, gate, down
        self.up = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.gate = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        # Norms
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        B, S, H = x.shape

        # Attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q = qkv[:, :, :self.hidden_size]
        k = qkv[:, :, self.hidden_size:self.hidden_size + self.num_kv_heads * self.head_dim]
        v = qkv[:, :, self.hidden_size + self.num_kv_heads * self.head_dim:]
        # Simple attention (no FlashAttention for portability)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # GQA: expand k,v
        if self.num_kv_heads < self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, ratio, S, self.head_dim).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, ratio, S, self.head_dim).reshape(B, self.num_heads, S, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, self.hidden_size)
        x = self.proj(attn)
        x = residual + x

        # MLP (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = self.down(F.silu(self.gate(x)) * self.up(x))
        x = residual + x

        return x


class TETransformerBlock(nn.Module):
    """TransformerEngine FP8 transformer block."""

    def __init__(self, hidden_size=2560, num_heads=20, num_kv_heads=5, seq_len=512, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # TE FP8 Linear layers
        self.qkv = te.pytorch.Linear(hidden_size, hidden_size + 2 * num_kv_heads * self.head_dim, bias=False)
        self.proj = te.pytorch.Linear(hidden_size, hidden_size, bias=False)
        self.up = te.pytorch.Linear(hidden_size, hidden_size * 4, bias=False)
        self.gate = te.pytorch.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down = te.pytorch.Linear(hidden_size * 4, hidden_size, bias=False)
        # Norms
        self.norm1 = te.pytorch.LayerNorm(hidden_size, eps=eps)
        self.norm2 = te.pytorch.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        B, S, H = x.shape

        # Attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q = qkv[:, :, :self.hidden_size]
        k = qkv[:, :, self.hidden_size:self.hidden_size + self.num_kv_heads * self.head_dim]
        v = qkv[:, :, self.hidden_size + self.num_kv_heads * self.head_dim:]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, ratio, S, self.head_dim).reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, ratio, S, self.head_dim).reshape(B, self.num_heads, S, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, self.hidden_size)
        x = self.proj(attn)
        x = residual + x

        # MLP (SwiGLU)
        residual = x
        x = self.norm2(x)
        x = self.down(F.silu(self.gate(x)) * self.up(x))
        x = residual + x

        return x


def benchmark_forward(model, x, num_iters=50, warmup=10):
    """Benchmark forward pass throughput."""
    device = x.device

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / num_iters * 1000
    B, S, H = x.shape
    throughput = B * S / latency_ms * 1000  # tokens/s
    return {
        "latency_ms": latency_ms,
        "throughput_tok_s": throughput,
        "num_iters": num_iters,
    }


def benchmark_training(model, x, num_iters=50, warmup=10):
    """Benchmark training step (forward + backward) throughput."""
    device = x.device
    target = torch.randn_like(x)

    # Warmup
    for _ in range(warmup):
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    torch.cuda.synchronize()

    # Reset gradients
    model.zero_grad()

    # Benchmark
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    total_tokens = 0
    for _ in range(num_iters):
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        total_tokens += x.shape[0] * x.shape[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    latency_ms = elapsed / num_iters * 1000
    throughput = total_tokens / elapsed  # tokens/s

    return {
        "train_latency_ms": latency_ms,
        "train_throughput_tok_s": throughput,
        "peak_memory_GB": peak_mem,
        "num_iters": num_iters,
    }


def benchmark_fp8_accuracy(bf16_model, fp8_model, x, recipe_type="delayed"):
    """Compare FP8 vs BF16 accuracy."""
    with torch.no_grad():
        bf16_out = bf16_model(x)

    if recipe_type == "delayed":
        recipe = DelayedScaling(fp8_format=Format.HYBRID, margin=0, amax_history_len=1024)
    elif recipe_type == "current":
        recipe = Float8CurrentScaling()
    else:
        raise ValueError(f"Unknown recipe: {recipe_type}")

    with te.pytorch.autocast(recipe=recipe):
        fp8_out = fp8_model(x)

    # Compare outputs
    cos_sim = F.cosine_similarity(bf16_out.flatten(), fp8_out.flatten(), dim=0).item()
    max_diff = (bf16_out - fp8_out).abs().max().item()
    mean_diff = (bf16_out - fp8_out).abs().mean().item()
    rel_diff = ((bf16_out - fp8_out).abs() / bf16_out.abs().clamp(min=1e-6)).mean().item()

    return {
        "cos_sim": cos_sim,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
        "bf16_norm": bf16_out.norm().item(),
        "fp8_norm": fp8_out.norm().item(),
    }


def run_benchmarks(hidden_size=2560, num_heads=20, num_kv_heads=5, seq_len=512,
                   batch_sizes=[1, 4, 8, 16, 32], num_iters=50, warmup=10):
    """Run all benchmarks."""
    results = {"device_info": get_device_info(), "te_available": TE_AVAILABLE}
    if TE_AVAILABLE:
        results["te_version"] = te.__version__

    model_params = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
    }
    # Calculate model size
    total_params = 0
    # QKV: H -> H + 2*kv_heads*head_dim
    total_params += hidden_size * (hidden_size + 2 * num_kv_heads * (hidden_size // num_heads))
    # Proj: H -> H
    total_params += hidden_size * hidden_size
    # MLP: up(H->4H) + gate(H->4H) + down(4H->H)
    total_params += hidden_size * hidden_size * 4 * 3
    model_size_MB = total_params * 2 / 1024**2  # BF16 = 2 bytes
    model_params["total_params"] = total_params
    model_params["model_size_MB"] = model_size_MB
    results["model_params"] = model_params

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Create BF16 baseline model
    bf16_model = SimpleTransformerBlock(hidden_size, num_heads, num_kv_heads, seq_len).to(device, dtype)

    # Benchmark BF16 baseline
    print("\n=== BF16 Baseline ===")
    bf16_fwd_results = {}
    bf16_train_results = {}
    for B in batch_sizes:
        x = torch.randn(B, seq_len, hidden_size, device=device, dtype=dtype)
        if x.shape[0] * x.shape[1] * x.shape[2] * 2 > torch.cuda.get_device_properties(device).total_memory * 0.8:
            print(f"  B={B}: Skipping (OOM risk)")
            continue
        fwd = benchmark_forward(bf16_model, x, num_iters, warmup)
        bf16_fwd_results[B] = fwd
        print(f"  B={B}: fwd {fwd['latency_ms']:.2f}ms, {fwd['throughput_tok_s']:.0f} tok/s")

        train = benchmark_training(bf16_model, x, num_iters, warmup)
        bf16_train_results[B] = train
        print(f"  B={B}: train {train['train_latency_ms']:.2f}ms, {train['train_throughput_tok_s']:.0f} tok/s, peak {train['peak_memory_GB']:.2f}GB")

        torch.cuda.empty_cache()

    results["bf16_forward"] = bf16_fwd_results
    results["bf16_training"] = bf16_train_results

    if not TE_AVAILABLE:
        print("\n=== FP8 tests SKIPPED (transformer_engine not available) ===")
        return results

    # Create TE FP8 model
    fp8_model = TETransformerBlock(hidden_size, num_heads, num_kv_heads, seq_len).to(device, dtype)

    # Copy weights from BF16 model to FP8 model
    fp8_model.qkv.weight.data.copy_(bf16_model.qkv.weight.data)
    fp8_model.proj.weight.data.copy_(bf16_model.proj.weight.data)
    fp8_model.up.weight.data.copy_(bf16_model.up.weight.data)
    fp8_model.gate.weight.data.copy_(bf16_model.gate.weight.data)
    fp8_model.down.weight.data.copy_(bf16_model.down.weight.data)
    fp8_model.norm1.weight.data.copy_(bf16_model.norm1.weight.data)
    fp8_model.norm1.bias.data.copy_(bf16_model.norm1.bias.data)
    fp8_model.norm2.weight.data.copy_(bf16_model.norm2.weight.data)
    fp8_model.norm2.bias.data.copy_(bf16_model.norm2.bias.data)

    # FP8 DelayedScaling benchmarks
    print("\n=== FP8 DelayedScaling ===")
    recipe_ds = DelayedScaling(fp8_format=Format.HYBRID, margin=0, amax_history_len=1024)

    fp8_ds_fwd_results = {}
    fp8_ds_train_results = {}
    fp8_ds_accuracy = {}

    for B in batch_sizes:
        x = torch.randn(B, seq_len, hidden_size, device=device, dtype=dtype)
        if x.shape[0] * x.shape[1] * x.shape[2] * 2 > torch.cuda.get_device_properties(device).total_memory * 0.8:
            print(f"  B={B}: Skipping (OOM risk)")
            continue

        # Forward with FP8 autocast
        with te.pytorch.autocast(recipe=recipe_ds):
            fwd = benchmark_forward(fp8_model, x, num_iters, warmup)
            fp8_ds_fwd_results[B] = fwd
            print(f"  B={B}: fwd {fwd['latency_ms']:.2f}ms, {fwd['throughput_tok_s']:.0f} tok/s")

        # Training with FP8 autocast
        fp8_model.zero_grad()
        target = torch.randn_like(x)
        torch.cuda.reset_peak_memory_stats(device)

        with te.pytorch.autocast(recipe=recipe_ds):
            train = benchmark_training(fp8_model, x, num_iters, warmup)
            fp8_ds_train_results[B] = train
            print(f"  B={B}: train {train['train_latency_ms']:.2f}ms, {train['train_throughput_tok_s']:.0f} tok/s, peak {train['peak_memory_GB']:.2f}GB")

        # Accuracy comparison
        with te.pytorch.autocast(recipe=recipe_ds):
            acc = benchmark_fp8_accuracy(bf16_model, fp8_model, x, "delayed")
            fp8_ds_accuracy[B] = acc
            print(f"  B={B}: cos_sim={acc['cos_sim']:.6f}, max_diff={acc['max_diff']:.6f}")

        torch.cuda.empty_cache()

    results["fp8_delayed_forward"] = fp8_ds_fwd_results
    results["fp8_delayed_training"] = fp8_ds_train_results
    results["fp8_delayed_accuracy"] = fp8_ds_accuracy

    # FP8 Float8CurrentScaling benchmarks
    print("\n=== FP8 Float8CurrentScaling ===")
    recipe_cs = Float8CurrentScaling()

    fp8_cs_fwd_results = {}
    fp8_cs_train_results = {}
    fp8_cs_accuracy = {}

    for B in batch_sizes:
        x = torch.randn(B, seq_len, hidden_size, device=device, dtype=dtype)
        if x.shape[0] * x.shape[1] * x.shape[2] * 2 > torch.cuda.get_device_properties(device).total_memory * 0.8:
            print(f"  B={B}: Skipping (OOM risk)")
            continue

        fp8_model.zero_grad()
        with te.pytorch.autocast(recipe=recipe_cs):
            fwd = benchmark_forward(fp8_model, x, num_iters, warmup)
            fp8_cs_fwd_results[B] = fwd
            print(f"  B={B}: fwd {fwd['latency_ms']:.2f}ms, {fwd['throughput_tok_s']:.0f} tok/s")

        fp8_model.zero_grad()
        target = torch.randn_like(x)
        torch.cuda.reset_peak_memory_stats(device)

        with te.pytorch.autocast(recipe=recipe_cs):
            train = benchmark_training(fp8_model, x, num_iters, warmup)
            fp8_cs_train_results[B] = train
            print(f"  B={B}: train {train['train_latency_ms']:.2f}ms, {train['train_throughput_tok_s']:.0f} tok/s, peak {train['peak_memory_GB']:.2f}GB")

        with te.pytorch.autocast(recipe=recipe_cs):
            acc = benchmark_fp8_accuracy(bf16_model, fp8_model, x, "current")
            fp8_cs_accuracy[B] = acc
            print(f"  B={B}: cos_sim={acc['cos_sim']:.6f}, max_diff={acc['max_diff']:.6f}")

        torch.cuda.empty_cache()

    results["fp8_current_forward"] = fp8_cs_fwd_results
    results["fp8_current_training"] = fp8_cs_train_results
    results["fp8_current_accuracy"] = fp8_cs_accuracy

    # Speedup summary
    print("\n=== Speedup Summary ===")
    for B in batch_sizes:
        if B in bf16_train_results and B in fp8_ds_train_results:
            ds_speedup = fp8_ds_train_results[B]["train_throughput_tok_s"] / bf16_train_results[B]["train_throughput_tok_s"]
            ds_mem_ratio = fp8_ds_train_results[B]["peak_memory_GB"] / bf16_train_results[B]["peak_memory_GB"]
            print(f"  B={B}: DelayedScaling {ds_speedup:.2f}x speedup, {ds_mem_ratio:.2f}x memory")

        if B in bf16_train_results and B in fp8_cs_train_results:
            cs_speedup = fp8_cs_train_results[B]["train_throughput_tok_s"] / bf16_train_results[B]["train_throughput_tok_s"]
            cs_mem_ratio = fp8_cs_train_results[B]["peak_memory_GB"] / bf16_train_results[B]["peak_memory_GB"]
            print(f"  B={B}: CurrentScaling {cs_speedup:.2f}x speedup, {cs_mem_ratio:.2f}x memory")

    return results


def main():
    parser = argparse.ArgumentParser(description="TransformerEngine FP8 Training Benchmark — RTX 4090")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument("--num-kv-heads", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/te_fp8_training_benchmark.json")
    args = parser.parse_args()

    print(f"TransformerEngine FP8 Training Benchmark — RTX 4090")
    print(f"Model: {args.hidden_size} hidden, {args.num_heads} heads, {args.num_kv_heads} KV heads")
    print(f"Sequence length: {args.seq_len}")
    print(f"TE available: {TE_AVAILABLE}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    results = run_benchmarks(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        batch_sizes=args.batch_sizes,
        num_iters=args.num_iters,
        warmup=args.warmup,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()