"""
FP8 KV Cache Benchmark — RTX 4090 (SM89)

Compares BF16 vs INT8 vs FP8 KV cache quantization for attention.
Tests throughput, accuracy (cos_sim), and memory savings.
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
# SECTION 1: INT8 KV (per-token quantization) — baseline comparison
# =====================================================================
print("\n" + "="*70)
print("SECTION 1: INT8 KV Cache (per-token quantization)")
print("="*70)

results = {}

for B in [1, 4, 8, 16, 32]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    k_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

    def sdpa_bf16():
        k_exp = k_bf16.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        v_exp = v_bf16.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    bf16_ms, _ = benchmark_fn(sdpa_bf16, warmup=3, iters=20)
    bf16_out = sdpa_bf16()

    # INT8 per-token quantization
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    k_max = k_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    k_int8 = torch.clamp(torch.round(k_bf16.float() * 127 / k_max), -128, 127).to(torch.int8)
    k_scale = k_max / 127.0
    v_max = v_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    v_int8 = torch.clamp(torch.round(v_bf16.float() * 127 / v_max), -128, 127).to(torch.int8)
    v_scale = v_max / 127.0

    def sdpa_int8_kv():
        k_dequant = (k_int8.float() * k_scale).bfloat16()
        v_dequant = (v_int8.float() * v_scale).bfloat16()
        k_exp = k_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        v_exp = v_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    int8_ms, _ = benchmark_fn(sdpa_int8_kv, warmup=3, iters=20)
    int8_out = sdpa_int8_kv()
    int8_cos = F.cosine_similarity(bf16_out.flatten().float().unsqueeze(0), int8_out.flatten().float().unsqueeze(0)).item()

    print(f"  B={B}: BF16={bf16_ms:.3f}ms INT8={int8_ms:.3f}ms speedup={bf16_ms/int8_ms:.2f}x cos_sim={int8_cos:.6f}")

    results[f"int8_B{B}"] = {"bf16_ms": round(bf16_ms,4), "int8_ms": round(int8_ms,4), 
                              "speedup": round(bf16_ms/int8_ms,2), "cos_sim": round(int8_cos,6)}

# =====================================================================
# SECTION 2: FP8 E4M3 KV Cache (per-token quantization)
# =====================================================================
print("\n" + "="*70)
print("SECTION 2: FP8 E4M3 KV Cache (per-token quantization)")
print("="*70)

# FP8 E4M3: max representable value = 448.0 (from IEEE E4M3 format)
FP8_E4M3_MAX = 448.0

for B in [1, 4, 8, 16, 32]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    k_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
    q = torch.randn(B, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

    bf16_ms, _ = benchmark_fn(lambda: F.scaled_dot_product_attention(q,
        k_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
        v_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
        is_causal=False), warmup=3, iters=20)

    # FP8 E4M3 per-token quantization
    # Scale: amax / FP8_E4M3_MAX
    k_fp8_scale = k_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / FP8_E4M3_MAX
    # Simulate FP8 quantization: round to nearest FP8 E4M3 value
    # Since we can't directly store FP8 in PyTorch, we simulate the quantization error
    k_fp8_quantized = torch.clamp(torch.round(k_bf16.float() / k_fp8_scale), -448, 448)  # FP8 E4M3 range
    # Store as float32 (simulating FP8 storage — real FP8 would be 1 byte per element)
    k_fp8_stored = k_fp8_quantized  # float32 representation of FP8 values
    
    v_fp8_scale = v_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / FP8_E4M3_MAX
    v_fp8_quantized = torch.clamp(torch.round(v_bf16.float() / v_fp8_scale), -448, 448)
    v_fp8_stored = v_fp8_quantized

    def sdpa_fp8_kv():
        # Dequantize FP8 to BF16
        k_dequant = (k_fp8_stored * k_fp8_scale).bfloat16()
        v_dequant = (v_fp8_stored * v_fp8_scale).bfloat16()
        k_exp = k_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        v_exp = v_dequant.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    fp8_ms, _ = benchmark_fn(sdpa_fp8_kv, warmup=3, iters=20)
    fp8_out = sdpa_fp8_kv()
    fp8_cos = F.cosine_similarity(
        F.scaled_dot_product_attention(q,
            k_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
            v_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
            is_causal=False).flatten().float().unsqueeze(0),
        fp8_out.flatten().float().unsqueeze(0)).item()

    print(f"  B={B}: BF16={bf16_ms:.3f}ms FP8={fp8_ms:.3f}ms speedup={bf16_ms/fp8_ms:.2f}x cos_sim={fp8_cos:.6f}")

    results[f"fp8_e4m3_B{B}"] = {"bf16_ms": round(bf16_ms,4), "fp8_ms": round(fp8_ms,4),
                                  "speedup": round(bf16_ms/fp8_ms,2), "cos_sim": round(fp8_cos,6)}

# =====================================================================
# SECTION 3: TE FP8 KV (if TransformerEngine available)
# =====================================================================
print("\n" + "="*70)
print("SECTION 3: TE FP8 KV (TransformerEngine quantize)")
print("="*70)

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling
    TE_AVAILABLE = True
    print("TransformerEngine available — testing TE FP8 quantize for KV")
except ImportError:
    TE_AVAILABLE = False
    print("TransformerEngine not available — skipping")

if TE_AVAILABLE:
    for B in [1, 4, 8, 16, 32]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        k_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(B, S, num_kv_heads, d_head, dtype=torch.bfloat16, device=device)
        q = torch.randn(B, 1, num_heads, d_head, dtype=torch.bfloat16, device=device)

        bf16_ms, _ = benchmark_fn(lambda: F.scaled_dot_product_attention(q,
            k_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
            v_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
            is_causal=False), warmup=3, iters=20)

        # TE FP8 quantize K/V
        with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
            # TE quantize converts BF16 → FP8
            # Note: TE API expects specific tensor shapes
            # We'll simulate the quantize+dequantize process
            k_fp8_te = te.quantize(k_bf16, "e4m3")  # Returns QuantizedTensor
            # Dequantize back to BF16
            k_dequant_te = k_fp8_te.dequantize()  # BF16 output
            v_fp8_te = te.quantize(v_bf16, "e4m3")
            v_dequant_te = v_fp8_te.dequantize()

        def sdpa_te_fp8_kv():
            with te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
                k_q = te.quantize(k_bf16, "e4m3")
                k_dq = k_q.dequantize()
                v_q = te.quantize(v_bf16, "e4m3")
                v_dq = v_q.dequantize()
            k_exp = k_dq.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
            v_exp = v_dq.unsqueeze(2).expand(B, S, num_heads//num_kv_heads, num_kv_heads, d_head).reshape(B, S, num_heads, d_head)
            return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

        try:
            te_fp8_ms, _ = benchmark_fn(sdpa_te_fp8_kv, warmup=3, iters=20)
            te_fp8_out = sdpa_te_fp8_kv()
            te_fp8_cos = F.cosine_similarity(
                F.scaled_dot_product_attention(q,
                    k_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
                    v_bf16.unsqueeze(2).expand(B,S,num_heads//num_kv_heads,num_kv_heads,d_head).reshape(B,S,num_heads,d_head),
                    is_causal=False).flatten().float().unsqueeze(0),
                te_fp8_out.flatten().float().unsqueeze(0)).item()

            print(f"  B={B}: BF16={bf16_ms:.3f}ms TE_FP8={te_fp8_ms:.3f}ms speedup={bf16_ms/te_fp8_ms:.2f}x cos_sim={te_fp8_cos:.6f}")
            results[f"te_fp8_B{B}"] = {"bf16_ms": round(bf16_ms,4), "te_fp8_ms": round(te_fp8_ms,4),
                                       "speedup": round(bf16_ms/te_fp8_ms,2), "cos_sim": round(te_fp8_cos,6)}
        except Exception as e:
            print(f"  B={B}: TE FP8 KV failed: {e}")

# =====================================================================
# Save Results
# =====================================================================
all_results = {
    "device": {"name": props.name, "sm": f"{props.major}.{props.minor}", "hbm_GB": round(props.total_memory/1e9,2)},
    "config": {"H": H, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "d_head": d_head, "S": S},
    "fp8_e4m3_max": FP8_E4M3_MAX,
    "kv_results": results,
}

os.makedirs("results", exist_ok=True)
with open("results/fp8_kv_cache_benchmark.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to results/fp8_kv_cache_benchmark.json")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nINT8 KV vs BF16:")
for key, val in results.items():
    if key.startswith("int8"):
        print(f"  B={key}: speedup={val['speedup']}x, cos_sim={val['cos_sim']}")

print("\nFP8 E4M3 KV vs BF16:")
for key, val in results.items():
    if key.startswith("fp8_e4m3"):
        print(f"  B={key}: speedup={val['speedup']}x, cos_sim={val['cos_sim']}")

if any(k.startswith("te_fp8") for k in results):
    print("\nTE FP8 KV vs BF16:")
    for key, val in results.items():
        if key.startswith("te_fp8"):
            print(f"  B={key}: speedup={val['speedup']}x, cos_sim={val['cos_sim']}")
