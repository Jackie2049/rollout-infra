"""
FP8 GEMM Algorithm Analysis — RTX 4090 (SM89)

Analyzes:
1. Per-layer BF16 vs FP8 forward timing breakdown
2. Quantize overhead ratio per layer
3. Crossover batch size (when FP8 becomes faster)
4. Arithmetic intensity and roofline comparison
5. Full training step: BF16 vs FP8 DelayedScaling vs FP8 CurrentScaling

Key API finding: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
MUST be used alongside te_pytorch.autocast for individual TE Linear layers.
Without torch.autocast, the dtype check in set_activation_dtype fails
because torch.is_autocast_enabled() returns False inside te_pytorch.autocast.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import transformer_engine as te
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("TransformerEngine not available!")
    sys.exit(1)

print(f"TE version: {te.__version__}")
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor} MPs={props.multi_processor_count}")

# Model config
H = 2560
num_heads = 20
num_kv_heads = 5
d_head = H // num_heads
mlp_hidden = 4 * H  # Standard MLP hidden size
S = 512


def benchmark_fn(fn, warmup=5, iters=50):
    """Benchmark a function using CUDA events for accurate timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return np.mean(latencies), np.std(latencies)


# GEMM sizes for each layer
gemm_sizes = {
    "qkv_proj": (H, (num_heads + 2 * num_kv_heads) * d_head),     # K=2560, N=3840
    "out_proj": (num_heads * d_head, H),                             # K=1280, N=2560
    "gate_proj": (H, mlp_hidden),                                     # K=2560, N=10240
    "up_proj": (H, mlp_hidden),                                       # K=2560, N=10240
    "down_proj": (mlp_hidden, H),                                     # K=10240, N=2560
}

# Roofline parameters
HBM_BANDWIDTH = 890.8  # GB/s (measured)
FP16_PEAK = 82.58      # TFLOPS
FP8_PEAK = 165.2       # TFLOPS
RIDGE_AI = FP16_PEAK * 1e12 / (HBM_BANDWIDTH * 1e9)

ds_recipe = DelayedScaling()
cs_recipe = Float8CurrentScaling()

# =====================================================================
# SECTION 1: Per-layer BF16 vs FP8 Forward Timing
# =====================================================================
print("\n" + "=" * 70)
print("Per-layer BF16 vs FP8 Forward Timing Breakdown")
print("=" * 70)

batch_sizes = [1, 2, 4, 8, 16, 32]
layer_results = {}

for layer_name, (K, N) in gemm_sizes.items():
    print(f"\n  {layer_name}: K={K}, N={N}")

    # Create BF16 and FP8 layers
    bf16_layer = nn.Linear(K, N, bias=False).to(device).to(torch.bfloat16)
    fp8_layer = te_pytorch.Linear(K, N, bias=False).to(device)

    # Copy weights
    fp8_layer.weight.data.copy_(bf16_layer.weight.data)

    results_list = []
    for B in batch_sizes:
        M = B * S
        x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)

        # BF16 layer timing
        bf16_mean, bf16_std = benchmark_fn(
            lambda: bf16_layer(x_bf16.detach()), warmup=3, iters=30
        )

        # FP8 layer timing — MUST use torch.autocast + te_pytorch.autocast
        def fp8_fwd():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
                 te_pytorch.autocast(enabled=True, recipe=ds_recipe):
                return fp8_layer(x_bf16.detach())

        fp8_mean, fp8_std = benchmark_fn(fp8_fwd, warmup=3, iters=30)

        # Roofline metrics
        bf16_flops = 2 * M * K * N
        bf16_bytes = M * K * 2 + N * K * 2 + M * N * 2  # BF16: A+W+O all 2 bytes
        fp8_bytes = M * K * 1 + N * K * 1 + M * N * 2   # FP8: A+W 1 byte, O 2 bytes (BF16)
        bf16_ai = bf16_flops / bf16_bytes
        fp8_ai = bf16_flops / fp8_bytes

        speedup = bf16_mean / fp8_mean if fp8_mean > 0 else 0
        fp8_gemm_only_est = bf16_mean * fp8_bytes / bf16_bytes
        quant_overhead_est = max(0, fp8_mean - fp8_gemm_only_est)
        bound = "compute" if bf16_ai > RIDGE_AI else "memory"

        result = {
            "batch": B, "M": M, "K": K, "N": N,
            "bf16_ms": round(bf16_mean, 4), "bf16_std_ms": round(bf16_std, 4),
            "fp8_ms": round(fp8_mean, 4), "fp8_std_ms": round(fp8_std, 4),
            "speedup": round(speedup, 4),
            "bf16_flops": bf16_flops, "bf16_bytes": bf16_bytes, "fp8_bytes": fp8_bytes,
            "bf16_ai": round(bf16_ai, 1), "fp8_ai": round(fp8_ai, 1),
            "bf16_tflops": round(bf16_flops / (bf16_mean * 1e-3) / 1e12, 2),
            "fp8_tflops": round(bf16_flops / (fp8_mean * 1e-3) / 1e12, 2),
            "quant_overhead_est_ms": round(quant_overhead_est, 4),
            "bound": bound,
        }
        results_list.append(result)

        print(f"    B={B}: BF16={bf16_mean:.3f}ms FP8={fp8_mean:.3f}ms "
              f"speed={speedup:.2f}x AI_bf16={bf16_ai:.0f} AI_fp8={fp8_ai:.0f} bound={bound}")

    layer_results[layer_name] = results_list


# =====================================================================
# SECTION 2: Crossover Analysis (down_proj, fine-grained B=1..16)
# =====================================================================
print("\n" + "=" * 70)
print("Crossover Analysis: When does FP8 become faster than BF16?")
print("=" * 70)

crossover_batches = list(range(1, 17))
crossover_results = []

K, N = gemm_sizes["down_proj"]
bf16_down = nn.Linear(K, N, bias=False).to(device).to(torch.bfloat16)
fp8_down = te_pytorch.Linear(K, N, bias=False).to(device)
fp8_down.weight.data.copy_(bf16_down.weight.data)

for B in crossover_batches:
    M = B * S
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    bf16_mean, _ = benchmark_fn(lambda: bf16_down(x_bf16.detach()), warmup=3, iters=30)

    def fp8_fwd():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
             te_pytorch.autocast(enabled=True, recipe=ds_recipe):
            return fp8_down(x_bf16.detach())

    fp8_mean, _ = benchmark_fn(fp8_fwd, warmup=3, iters=30)

    speedup = bf16_mean / fp8_mean
    bf16_flops = 2 * M * K * N
    bf16_bytes = M * K * 2 + N * K * 2 + M * N * 2
    fp8_bytes = M * K * 1 + N * K * 1 + M * N * 2
    fp8_gemm_only_est = bf16_mean * fp8_bytes / bf16_bytes
    quant_overhead = max(0, fp8_mean - fp8_gemm_only_est)

    crossover_results.append({
        "batch": B, "M": M,
        "bf16_ms": round(bf16_mean, 4), "fp8_ms": round(fp8_mean, 4),
        "speedup": round(speedup, 4),
        "fp8_gemm_only_est_ms": round(fp8_gemm_only_est, 4),
        "quant_overhead_est_ms": round(quant_overhead, 4),
        "quant_ratio": round(max(0, quant_overhead) / fp8_mean, 4) if fp8_mean > 0 else 0,
    })

    crossover_str = "FASTER" if speedup > 1.0 else "SLOWER"
    print(f"  B={B}: BF16={bf16_mean:.3f}ms FP8={fp8_mean:.3f}ms "
          f"speed={speedup:.2f}x {crossover_str} "
          f"quant_est={quant_overhead:.3f}ms ({quant_overhead/fp8_mean:.1%} of FP8)")


# =====================================================================
# SECTION 3: Full Training Step (BF16 vs FP8 DS vs FP8 CS)
# =====================================================================
print("\n" + "=" * 70)
print("Full Training Step: BF16 vs FP8 DelayedScaling vs FP8 CurrentScaling")
print("=" * 70)

# Use the model-based approach for training (torch.autocast + te_pytorch.autocast)
class SimpleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(H, (num_heads + 2 * num_kv_heads) * d_head, bias=False)
        self.out = nn.Linear(num_heads * d_head, H, bias=False)
        self.gate = nn.Linear(H, mlp_hidden, bias=False)
        self.up = nn.Linear(H, mlp_hidden, bias=False)
        self.down = nn.Linear(mlp_hidden, H, bias=False)
        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)

    def forward(self, x):
        r = x; x = self.ln1(x)
        qkv = self.qkv(x)
        # Use Q part (first nh*d cols) as attention output proxy
        attn_out = qkv[:, :, :num_heads * d_head]
        out = self.out(attn_out)
        x = r + out
        r = x; x = self.ln2(x)
        h = F.silu(self.gate(x)) * self.up(x)
        x = r + self.down(h)
        return x


class TEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = te_pytorch.Linear(H, (num_heads + 2 * num_kv_heads) * d_head, bias=False)
        self.out = te_pytorch.Linear(num_heads * d_head, H, bias=False)
        self.gate = te_pytorch.Linear(H, mlp_hidden, bias=False)
        self.up = te_pytorch.Linear(H, mlp_hidden, bias=False)
        self.down = te_pytorch.Linear(mlp_hidden, H, bias=False)
        self.ln1 = te_pytorch.LayerNorm(H)  # TE LayerNorm for proper dtype handling
        self.ln2 = te_pytorch.LayerNorm(H)

    def forward(self, x):
        r = x; x = self.ln1(x)
        qkv = self.qkv(x)
        # Use Q part (first nh*d cols) as attention output proxy
        attn_out = qkv[:, :, :num_heads * d_head]
        out = self.out(attn_out)
        x = r + out
        r = x; x = self.ln2(x)
        h = F.silu(self.gate(x)) * self.up(x)
        x = r + self.down(h)
        return x


bf16_model = SimpleBlock().to(device).to(torch.bfloat16)
ds_model = TEBlock().to(device)
cs_model = TEBlock().to(device)

# Copy weights from BF16 to FP8 models
for name, param in bf16_model.named_parameters():
    if hasattr(ds_model, name) and hasattr(getattr(ds_model, name), 'weight'):
        getattr(ds_model, name).weight.data.copy_(param.data)
    if hasattr(cs_model, name) and hasattr(getattr(cs_model, name), 'weight'):
        getattr(cs_model, name).weight.data.copy_(param.data)

training_batches = [1, 4, 8, 16, 32]
training_results = []

for B in training_batches:
    x = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)

    # BF16 training
    bf16_opt = torch.optim.AdamW(bf16_model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()

    def bf16_step():
        bf16_opt.zero_grad()
        out = bf16_model(x)
        F.mse_loss(out, target).backward()
        bf16_opt.step()

    bf16_mean, _ = benchmark_fn(bf16_step, warmup=3, iters=20)
    bf16_mem = torch.cuda.max_memory_allocated() / 1e9

    # FP8 DelayedScaling training — torch.autocast + te_pytorch.autocast
    ds_opt = torch.optim.AdamW(ds_model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()

    def ds_step():
        ds_opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
             te_pytorch.autocast(enabled=True, recipe=ds_recipe):
            out = ds_model(x)
        F.mse_loss(out, target).backward()
        ds_opt.step()

    ds_mean, _ = benchmark_fn(ds_step, warmup=3, iters=20)
    ds_mem = torch.cuda.max_memory_allocated() / 1e9

    # FP8 CurrentScaling training
    cs_opt = torch.optim.AdamW(cs_model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()

    def cs_step():
        cs_opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
             te_pytorch.autocast(enabled=True, recipe=cs_recipe):
            out = cs_model(x)
        F.mse_loss(out, target).backward()
        cs_opt.step()

    cs_mean, _ = benchmark_fn(cs_step, warmup=3, iters=20)
    cs_mem = torch.cuda.max_memory_allocated() / 1e9

    # Accuracy: cosine similarity
    with torch.no_grad():
        bf16_out = bf16_model(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
             te_pytorch.autocast(enabled=True, recipe=ds_recipe):
            ds_out = ds_model(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
             te_pytorch.autocast(enabled=True, recipe=cs_recipe):
            cs_out = cs_model(x)
        ds_cos = F.cosine_similarity(
            bf16_out.flatten().unsqueeze(0).float(),
            ds_out.flatten().unsqueeze(0).float()
        ).item()
        cs_cos = F.cosine_similarity(
            bf16_out.flatten().unsqueeze(0).float(),
            cs_out.flatten().unsqueeze(0).float()
        ).item()

    result = {
        "batch": B,
        "bf16_ms": round(bf16_mean, 4), "bf16_mem_GB": round(bf16_mem, 3),
        "ds_ms": round(ds_mean, 4), "ds_mem_GB": round(ds_mem, 3), "ds_cos_sim": round(ds_cos, 6),
        "cs_ms": round(cs_mean, 4), "cs_mem_GB": round(cs_mem, 3), "cs_cos_sim": round(cs_cos, 6),
        "ds_speedup": round(bf16_mean / ds_mean, 4), "cs_speedup": round(bf16_mean / cs_mean, 4),
    }
    training_results.append(result)

    print(f"  B={B}: BF16={bf16_mean:.2f}ms DS={ds_mean:.2f}ms({bf16_mean/ds_mean:.2f}x) "
          f"CS={cs_mean:.2f}ms({bf16_mean/cs_mean:.2f}x) "
          f"DS_cos={ds_cos:.4f} CS_cos={cs_cos:.4f}")


# =====================================================================
# Save results
# =====================================================================
all_results = {
    "device_info": {
        "name": props.name, "sm": f"{props.major}.{props.minor}",
        "mp_count": props.multi_processor_count, "memory_GB": round(props.total_memory / 1e9, 2),
    },
    "te_version": te.__version__,
    "model_config": {
        "H": H, "num_heads": num_heads, "num_kv_heads": num_kv_heads,
        "S": S, "mlp_hidden": mlp_hidden,
    },
    "roofline_params": {
        "hbm_bw_GB_s": HBM_BANDWIDTH,
        "fp16_peak_tflops": FP16_PEAK,
        "fp8_peak_tflops": FP8_PEAK,
        "ridge_ai": round(RIDGE_AI, 1),
    },
    "layer_timing": layer_results,
    "crossover_analysis": crossover_results,
    "training_analysis": training_results,
}

os.makedirs("results", exist_ok=True)
output_path = "results/fp8_gemm_algorithm_analysis.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")


# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Find crossover
for r in crossover_results:
    if r["speedup"] > 1.0:
        print(f"\nFP8 crossover at B={r['batch']} (M={r['M']}): speedup={r['speedup']:.2f}x")
        print(f"  Quantize overhead estimate: {r['quant_overhead_est_ms']:.3f}ms "
              f"({r['quant_ratio']:.1%})")
        break
else:
    print("\nFP8 never faster in crossover range (B=1..16)")

# Ridge point
print(f"\nRoofline: Ridge AI = {RIDGE_AI:.0f} flops/byte")
for name, results in layer_results.items():
    r_b1 = results[0]
    r_last = results[-1]
    print(f"\n{name}: B=1 AI_bf16={r_b1['bf16_ai']:.0f} speed={r_b1['speedup']:.2f}x; "
          f"B={r_last['batch']} AI_bf16={r_last['bf16_ai']:.0f} speed={r_last['speedup']:.2f}x")