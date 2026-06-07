#!/usr/bin/env python3
"""Training Pipeline Latency Breakdown — RTX 4090
====================================================
Measures each Transformer component individually to reveal
where time is spent at different model sizes and batch sizes.

Components measured:
1. Embedding lookup
2. Q/K/V linear projections (3 × D→D)
3. QK^T matmul (B×H×S×S)
4. Softmax (B×H×S×S)
5. AV matmul (B×H×S×D/H)
6. Output projection (D→D)
7. MLP up-projection (D→4D)
8. Activation (GELU/SwiGLU)
9. MLP down-projection (4D→D)
10. RMSNorm (2 × per block)
11. Residual add (2 × per block)
12. Backward pass (total)
13. Optimizer step (AdamW)
14. Full training step

Also compares:
- Prefill vs Decode (S=1024 vs S=1)
- Different model sizes (7M, 25M, 125M)
- B=1 vs B=8 vs B=64
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import math


def benchmark_component(name, fn, n_runs=50, warmup=5):
    """Benchmark a single component function."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median = sorted(times)[len(times) // 2]
    min_t = min(times)
    return {"name": name, "median_ms": median, "min_ms": min_t, "n_runs": n_runs}


def run_benchmark(model_d, n_heads, n_layers, vocab_size, seq_len, batch_size, dtype=torch.float16):
    """Run comprehensive component benchmark for a given configuration."""
    device = "cuda"
    d_ff = 4 * model_d  # Standard 4x expansion
    d_head = model_d // n_heads

    results = {}

    # Create parameters
    W_embed = torch.randn(vocab_size, model_d, device=device, dtype=dtype)
    W_q = torch.randn(model_d, model_d, device=device, dtype=dtype)
    W_k = torch.randn(model_d, model_d, device=device, dtype=dtype)
    W_v = torch.randn(model_d, model_d, device=device, dtype=dtype)
    W_o = torch.randn(model_d, model_d, device=device, dtype=dtype)
    W_up = torch.randn(model_d, d_ff, device=device, dtype=dtype)
    W_down = torch.randn(d_ff, model_d, device=device, dtype=dtype)
    ln_weight = torch.randn(model_d, device=device, dtype=dtype)
    # For SwiGLU, need a gate projection
    W_gate = torch.randn(model_d, d_ff, device=device, dtype=dtype)

    # Input data
    torch.manual_seed(42)
    if seq_len == 1:
        # Decode: single token
        input_ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    else:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    x_input = torch.randn(batch_size, seq_len, model_d, device=device, dtype=dtype)

    # === Component benchmarks ===
    components = []

    # 1. Embedding lookup
    components.append(benchmark_component(
        "embedding_lookup",
        lambda: W_embed[input_ids]
    ))

    # 2. Q projection
    components.append(benchmark_component(
        "q_projection",
        lambda: x_input @ W_q
    ))

    # 3. K projection
    components.append(benchmark_component(
        "k_projection",
        lambda: x_input @ W_k
    ))

    # 4. V projection
    components.append(benchmark_component(
        "v_projection",
        lambda: x_input @ W_v
    ))

    # Compute Q, K, V for attention ops
    Q = (x_input @ W_q).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    K = (x_input @ W_k).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    V = (x_input @ W_v).view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)

    # 5. QK^T matmul
    scale = 1.0 / math.sqrt(d_head)
    components.append(benchmark_component(
        "qk_matmul",
        lambda: Q @ K.transpose(-2, -1) * scale
    ))

    # 6. Softmax
    S_raw = Q @ K.transpose(-2, -1) * scale
    components.append(benchmark_component(
        "softmax",
        lambda: torch.softmax(S_raw, dim=-1)
    ))

    # 7. AV matmul
    attn_weights = torch.softmax(S_raw, dim=-1)
    components.append(benchmark_component(
        "av_matmul",
        lambda: attn_weights @ V
    ))

    # 8. Output projection
    attn_out = (attn_weights @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, model_d)
    components.append(benchmark_component(
        "output_projection",
        lambda: attn_out @ W_o
    ))

    # 9. MLP up-projection
    components.append(benchmark_component(
        "mlp_up_proj",
        lambda: x_input @ W_up
    ))

    # 10. SwiGLU activation
    up_out = x_input @ W_up
    gate_out = x_input @ W_gate
    components.append(benchmark_component(
        "swiglu_activation",
        lambda: torch.nn.functional.silu(gate_out) * up_out
    ))

    # 11. MLP down-projection
    swiglu_out = torch.nn.functional.silu(gate_out) * up_out
    components.append(benchmark_component(
        "mlp_down_proj",
        lambda: swiglu_out @ W_down
    ))

    # 12. RMSNorm
    components.append(benchmark_component(
        "rmsnorm",
        lambda: x_input / torch.sqrt(torch.mean(x_input ** 2, dim=-1, keepdim=True) + 1e-6) * ln_weight
    ))

    # 13. Residual add
    components.append(benchmark_component(
        "residual_add",
        lambda: x_input + attn_out
    ))

    # Total forward estimate (single block)
    forward_components = ["q_projection", "k_projection", "v_projection",
                          "qk_matmul", "softmax", "av_matmul",
                          "output_projection", "mlp_up_proj", "swiglu_activation",
                          "mlp_down_proj", "rmsnorm", "rmsnorm", "residual_add", "residual_add"]
    # Note: embedding only at first layer, not per-block

    total_forward_estimate = sum(
        c["median_ms"] for c in components if c["name"] in forward_components
    )
    # Add 2x RMSNorm + 2x residual
    rmsnorm_ms = next(c["median_ms"] for c in components if c["name"] == "rmsnorm")
    residual_ms = next(c["median_ms"] for c in components if c["name"] == "residual_add")
    total_forward_estimate += rmsnorm_ms + residual_ms  # Second LN + second residual

    # === Full training step benchmark ===
    # Create a simple model
    class SimpleBlock(nn.Module):
        def __init__(self, d, n_heads, d_ff):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(
                nn.Linear(d, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.mlp(self.ln2(x))
            return x

    model = nn.Sequential(
        nn.Embedding(vocab_size, model_d),
        *[SimpleBlock(model_d, n_heads, d_ff) for _ in range(n_layers)],
        nn.LayerNorm(model_d),
    ).to(device).to(dtype)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Forward + backward
    def train_step():
        optimizer.zero_grad()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    # Warmup
    for _ in range(3):
        train_step()

    torch.cuda.reset_peak_memory_stats()

    # Measure full step
    step_results = benchmark_component("full_train_step", train_step, n_runs=20, warmup=3)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Backward alone
    out = model(input_ids)
    loss_val = out.sum()

    backward_results = benchmark_component(
        "backward_pass",
        lambda: loss_val.backward(retain_graph=True),
        n_runs=20, warmup=3
    )

    # Optimizer alone
    def opt_step():
        optimizer.step()
        optimizer.zero_grad()

    opt_results = benchmark_component("optimizer_step", opt_step, n_runs=20, warmup=3)

    # Compile results
    results = {
        "config": {
            "d_model": model_d, "n_heads": n_heads, "n_layers": n_layers,
            "vocab_size": vocab_size, "seq_len": seq_len, "batch_size": batch_size,
            "dtype": str(dtype), "n_params": n_params, "d_ff": d_ff, "d_head": d_head,
        },
        "components": {c["name"]: {"median_ms": c["median_ms"], "min_ms": c["min_ms"]} for c in components},
        "forward_estimate_ms": total_forward_estimate,
        "full_train_step_ms": step_results["median_ms"],
        "backward_only_ms": backward_results["median_ms"],
        "optimizer_step_ms": opt_results["median_ms"],
        "peak_memory_GB": peak_mem,
        "per_layer_forward_ms": total_forward_estimate / n_layers if n_layers > 0 else 0,
    }

    # Calculate percentages
    total_step = step_results["median_ms"]
    if total_step > 0:
        results["component_pct"] = {}
        for c in components:
            pct = c["median_ms"] / total_forward_estimate * 100 if total_forward_estimate > 0 else 0
            results["component_pct"][c["name"]] = pct

    return results


def main():
    configs = [
        # Small model, prefill
        {"d": 256, "heads": 8, "layers": 4, "vocab": 32000, "seq": 128, "batch": 8, "label": "2.3M_prefill_B8"},
        # Small model, decode
        {"d": 256, "heads": 8, "layers": 4, "vocab": 32000, "seq": 1, "batch": 32, "label": "2.3M_decode_B32"},
        # Medium model, prefill
        {"d": 512, "heads": 16, "layers": 8, "vocab": 32000, "seq": 512, "batch": 8, "label": "25M_prefill_B8"},
        # Medium model, decode
        {"d": 512, "heads": 16, "layers": 8, "vocab": 32000, "seq": 1, "batch": 64, "label": "25M_decode_B64"},
        # Large-ish model, prefill
        {"d": 1024, "heads": 16, "layers": 16, "vocab": 32000, "seq": 512, "batch": 4, "label": "125M_prefill_B4"},
        # Large-ish model, decode
        {"d": 1024, "heads": 16, "layers": 16, "vocab": 32000, "seq": 1, "batch": 128, "label": "125M_decode_B128"},
    ]

    all_results = {}
    for cfg in configs:
        label = cfg["label"]
        print(f"\n=== Benchmarking {label} ===")
        result = run_benchmark(
            model_d=cfg["d"], n_heads=cfg["heads"], n_layers=cfg["layers"],
            vocab_size=cfg["vocab"], seq_len=cfg["seq"], batch_size=cfg["batch"]
        )
        all_results[label] = result

        # Print summary
        print(f"  d={cfg['d']}, H={cfg['heads']}, L={cfg['layers']}, S={cfg['seq']}, B={cfg['batch']}")
        print(f"  Params: {result['config']['n_params']:,}")
        print(f"  Full step: {result['full_train_step_ms']:.2f}ms")
        print(f"  Forward estimate: {result['forward_estimate_ms']:.2f}ms")
        print(f"  Backward: {result['backward_only_ms']:.2f}ms")
        print(f"  Optimizer: {result['optimizer_step_ms']:.2f}ms")
        print(f"  Peak memory: {result['peak_memory_GB']:.3f}GB")

        if "component_pct" in result:
            print(f"  Component breakdown:")
            for name, pct in sorted(result["component_pct"].items(), key=lambda x: -x[1]):
                ms = result["components"][name]["median_ms"]
                if pct > 1:
                    print(f"    {name}: {ms:.3f}ms ({pct:.1f}%)")

    # Save
    with open("results/training_pipeline_breakdown.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/training_pipeline_breakdown.json")

    # Print comparison table
    print("\n=== Prefill vs Decode Comparison ===")
    print(f"{'Label':>20} {'Step(ms)':>10} {'Fwd(ms)':>10} {'Bwd(ms)':>10} {'Opt(ms)':>10} {'Mem(GB)':>10}")
    for label, r in all_results.items():
        print(f"{label:>20} {r['full_train_step_ms']:>10.2f} {r['forward_estimate_ms']:>10.2f} "
              f"{r['backward_only_ms']:>10.2f} {r['optimizer_step_ms']:>10.2f} {r['peak_memory_GB']:>10.3f}")


if __name__ == "__main__":
    main()