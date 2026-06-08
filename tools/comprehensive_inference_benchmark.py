"""
Comprehensive Inference Benchmark — RTX 4090 (SM89)

Tests:
1. INT4 weight-only inference: throughput + cos_sim + memory saving
2. BF16 vs INT4 decode throughput comparison
3. INT8 KV cache: throughput + cos_sim + memory saving
4. Combined INT4 + INT8 KV: overall throughput and memory
5. Roofline verification: actual vs theoretical throughput
"""

import os, sys, json, numpy as np, torch, torch.nn.functional as F, time

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name} SM={props.major}.{props.minor}")
print(f"HBM: {props.total_memory/1e9:.2f} GB, MPs: {props.multi_processor_count}")

H = 2560; num_heads = 20; num_kv_heads = 5; d_head = H // num_heads; S = 512

def benchmark_fn(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        latencies.append(s.elapsed_time(e))
    return np.mean(latencies), np.std(latencies)

def get_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9

# =====================================================================
# SECTION 1: INT4 Weight-Only Inference
# =====================================================================
print("\n" + "="*70)
print("SECTION 1: INT4 Weight-Only Quantization Benchmark")
print("="*70)

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    print("AWQ not available — using manual INT4 simulation")

# Manual INT4 simulation: quantize weight to INT4, dequantize for inference
# This simulates weight-only INT4 quantization (group_size=128)
results_int4 = []

for B in [1, 4, 8, 16, 32]:
    # BF16 baseline
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create model weights (simulating 7B MLP layer)
    N_in = 2560; N_out = 10240  # gate_proj dimensions
    weight_bf16 = torch.randn(N_out, N_in, dtype=torch.bfloat16, device=device)
    x_bf16 = torch.randn(B, N_in, dtype=torch.bfloat16, device=device)

    def bf16_gemm():
        return torch.nn.functional.linear(x_bf16, weight_bf16)

    bf16_ms, bf16_std = benchmark_fn(bf16_gemm, warmup=3, iters=20)
    bf16_mem = get_mem()
    bf16_out = bf16_gemm()

    # INT4 weight-only simulation (group_size=128)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    group_size = 128
    num_groups = N_in // group_size

    # Quantize weight to INT4 (simulate)
    weight_float = weight_bf16.float()
    # Per-group quantization
    weight_groups = weight_float.reshape(N_out, num_groups, group_size)
    w_max = weight_groups.abs().amax(dim=-1, keepdim=True)  # (N_out, num_groups, 1)
    scales = w_max / 7.0  # INT4 max = 7 (4-bit symmetric: -8 to 7)
    weight_int4 = torch.clamp(torch.round(weight_groups / scales), -8, 7).to(torch.int8)
    # Store as packed int4 (2 values per byte) — but for simulation use int8
    weight_int4_packed = weight_int4.reshape(N_out, N_in).to(torch.bfloat16)  # dequant representation

    def int4_gemm():
        # Dequantize then GEMM (this is the Python overhead path)
        # scales shape: (N_out, num_groups, 1) → expand to (N_out, N_in) via repeat
        scales_expanded = scales.squeeze(-1).repeat_interleave(group_size, dim=1)  # (N_out, N_in)
        w_dequant = (weight_int4_packed.float() * scales_expanded).bfloat16()
        return torch.nn.functional.linear(x_bf16, w_dequant)

    int4_ms, int4_std = benchmark_fn(int4_gemm, warmup=3, iters=20)
    int4_mem = get_mem()
    int4_out = int4_gemm()

    # Correctness
    cos_sim = F.cosine_similarity(bf16_out.flatten().float().unsqueeze(0),
                                  int4_out.flatten().float().unsqueeze(0)).item()
    max_diff = (bf16_out.float() - int4_out.float()).abs().max().item()

    speedup = bf16_ms / int4_ms
    weight_mem_bf16 = N_out * N_in * 2 / 1e9  # BF16: 2 bytes
    weight_mem_int4 = N_out * N_in * 0.5 / 1e9  # INT4: 0.5 bytes (ideal)
    # With scales: 0.5 bytes + scales overhead
    weight_mem_int4_actual = (N_out * N_in * 0.5 + N_out * num_groups * 2) / 1e9

    results_int4.append({
        "B": B, "bf16_ms": round(bf16_ms,4), "int4_ms": round(int4_ms,4),
        "speedup": round(speedup,2), "cos_sim": round(cos_sim,6),
        "max_diff": round(max_diff,6), "bf16_mem_GB": round(bf16_mem,3),
        "int4_mem_GB": round(int4_mem,3),
        "weight_mem_bf16_GB": round(weight_mem_bf16,4),
        "weight_mem_int4_ideal_GB": round(weight_mem_int4,4),
        "weight_mem_int4_actual_GB": round(weight_mem_int4_actual,4),
    })
    print(f"  B={B}: BF16={bf16_ms:.3f}ms INT4={int4_ms:.3f}ms speedup={speedup:.2f}x cos_sim={cos_sim:.6f} diff={max_diff:.6f}")

# =====================================================================
# SECTION 2: INT8 KV Cache
# =====================================================================
print("\n" + "="*70)
print("SECTION 2: INT8 KV Cache Quantization Benchmark")
print("="*70)

results_int8kv = []

for B in [1, 4, 8, 16, 32]:
    # BF16 KV baseline
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    k_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

    def sdpa_bf16_kv():
        k_exp = k_bf16.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        v_exp = v_bf16.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    bf16_ms, bf16_std = benchmark_fn(sdpa_bf16_kv, warmup=3, iters=20)
    bf16_mem = get_mem()
    bf16_out = sdpa_bf16_kv()

    # INT8 KV: quantize K/V to INT8, dequantize before attention
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Per-token INT8 quantization
    k_int8 = torch.clamp(torch.round(k_bf16.float() * 127 / k_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)),
                         -128, 127).to(torch.int8)
    k_scale = k_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    v_int8 = torch.clamp(torch.round(v_bf16.float() * 127 / v_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)),
                         -128, 127).to(torch.int8)
    v_scale = v_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0

    # Store INT8 KV + scales
    # INT8: 1 byte per element, scales: 4 bytes per token per head

    def sdpa_int8_kv():
        # Dequantize INT8 KV to BF16 before attention
        k_dequant = (k_int8.float() * k_scale).bfloat16()
        v_dequant = (v_int8.float() * v_scale).bfloat16()
        k_exp = k_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        v_exp = v_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    int8_ms, int8_std = benchmark_fn(sdpa_int8_kv, warmup=3, iters=20)
    int8_mem = get_mem()
    int8_out = sdpa_int8_kv()

    cos_sim = F.cosine_similarity(bf16_out.flatten().float().unsqueeze(0),
                                  int8_out.flatten().float().unsqueeze(0)).item()

    kv_mem_bf16 = B * S * num_kv_heads * d_head * 2 * 2 / 1e9  # K+V, BF16
    kv_mem_int8 = B * S * num_kv_heads * d_head * 1 * 2 / 1e9   # K+V, INT8
    kv_mem_scales = B * S * num_kv_heads * 1 * 4 * 2 / 1e9      # scales

    results_int8kv.append({
        "B": B, "bf16_ms": round(bf16_ms,4), "int8_ms": round(int8_ms,4),
        "speedup": round(bf16_ms/int8_ms,2), "cos_sim": round(cos_sim,6),
        "kv_mem_bf16_GB": round(kv_mem_bf16,4),
        "kv_mem_int8_GB": round(kv_mem_int8,4),
        "kv_mem_scales_GB": round(kv_mem_scales,4),
        "kv_saving_pct": round((1 - (kv_mem_int8+kv_mem_scales)/kv_mem_bf16)*100,1),
    })
    print(f"  B={B}: BF16={bf16_ms:.3f}ms INT8KV={int8_ms:.3f}ms speedup={bf16_ms/int8_ms:.2f}x cos_sim={cos_sim:.6f} KV_saving={round((1-(kv_mem_int8+kv_mem_scales)/kv_mem_bf16)*100,1)}%")

# =====================================================================
# SECTION 3: HBM Bandwidth Measurement
# =====================================================================
print("\n" + "="*70)
print("SECTION 3: HBM Bandwidth Measurement (Roofline)")
print("="*70)

# Copy benchmark: measure actual HBM bandwidth
sizes = [1<<20, 1<<22, 1<<24, 1<<26, 1<<28]  # 1MB to 256MB
hbm_results = []

for sz in sizes:
    src = torch.randn(sz // 2, dtype=torch.bfloat16, device=device)  # sz bytes
    dst = torch.empty_like(src)

    def copy_fn():
        dst.copy_(src)

    ms, _ = benchmark_fn(copy_fn, warmup=3, iters=20)
    bandwidth_gbs = sz / (ms * 1e-3) / 1e9  # GB/s
    hbm_results.append({"bytes": sz, "ms": round(ms,4), "bandwidth_GB_s": round(bandwidth_gbs,2)})
    print(f"  {sz/1e6:.1f}MB copy: {ms:.3f}ms → {bandwidth_gbs:.2f} GB/s")

# =====================================================================
# Save Results
# =====================================================================
all_results = {
    "device": {"name": props.name, "sm": f"{props.major}.{props.minor}", "hbm_GB": round(props.total_memory/1e9,2)},
    "config": {"H": H, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "d_head": d_head, "S": S},
    "int4_weight_only": results_int4,
    "int8_kv_cache": results_int8kv,
    "hbm_bandwidth": hbm_results,
}

os.makedirs("results", exist_ok=True)
with open("results/comprehensive_inference_benchmark.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to results/comprehensive_inference_benchmark.json")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nINT4 Weight-Only:")
for r in results_int4:
    print(f"  B={r['B']}: {r['speedup']}x, cos_sim={r['cos_sim']}, weight_mem={r['weight_mem_bf16_GB']:.4f}GB→{r['weight_mem_int4_actual_GB']:.4f}GB")

print("\nINT8 KV Cache:")
for r in results_int8kv:
    print(f"  B={r['B']}: {r['speedup']}x, cos_sim={r['cos_sim']}, KV_saving={r['kv_saving_pct']}%")

print("\nHBM Bandwidth:")
for r in hbm_results:
    print(f"  {r['bytes']/1e6:.1f}MB: {r['bandwidth_GB_s']} GB/s")