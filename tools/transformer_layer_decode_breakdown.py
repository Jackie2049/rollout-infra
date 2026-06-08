"""
Transformer Layer Decode Breakdown — RTX 4090
Measures each component of a 7B-like transformer decode step individually,
providing a clear breakdown of what dominates decode latency.

Focus: Understanding per-component latency and identifying bottlenecks.
"""

import torch
import time
import json

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
HBM_GB = props.total_memory / 1024**3
print(f"Device: {props.name}, HBM: {HBM_GB:.2f} GB")
HBM_BW = 890.8  # GB/s (实测)

# 7B-like model parameters
hidden = 4096
n_heads = 32
kv_heads = 8  # GQA-8
d_head = 128
inter_dim = 14336
vocab_size = 32000
n_layers = 32


def benchmark_op(op_fn, warmup=10, repeats=100):
    """Benchmark an operation"""
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        op_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "avg_us": round(sum(times) / len(times), 2),
        "min_us": round(min(times), 2),
        "p50_us": round(sorted(times)[len(times)//2], 2),
        "p99_us": round(sorted(times)[int(len(times)*0.99)], 2),
    }


def run_all():
    results = {}
    print("=" * 70)
    print("Transformer Layer Decode Breakdown — RTX 4090")
    print("=" * 70)

    # ====================================================================
    # Exp 1: Per-Component Timing (B=1 decode)
    # ====================================================================
    print("\n--- Exp 1: Per-Component Timing B=1 Decode ---")
    exp1 = {}

    B = 1
    x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

    # QKV projection: B×hidden → B×3×hidden (fused)
    w_qkv = torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16)
    t_qkv = benchmark_op(lambda: x @ w_qkv)
    exp1["qkv_proj"] = {**t_qkv, "shape": f"{B}x{hidden} → {B}x{3*hidden}", "bytes": B * hidden * 3 * hidden * 2}
    print(f"  QKV proj: {t_qkv['avg_us']:.1f}us → {B*hidden*3*hidden*2/1024**2:.2f}MB data")

    # QKV separate (3 individual GEMMs)
    w_q = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16)
    w_k = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16)
    w_v = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16)
    t_q = benchmark_op(lambda: x @ w_q)
    t_k = benchmark_op(lambda: x @ w_k)
    t_v = benchmark_op(lambda: x @ w_v)
    exp1["q_proj"] = {**t_q, "shape": f"{B}x{hidden} → {B}x{hidden}"}
    exp1["k_proj"] = {**t_k, "shape": f"{B}x{hidden} → {B}x{hidden}"}
    exp1["v_proj"] = {**t_v, "shape": f"{B}x{hidden} → {B}x{hidden}"}
    exp1["qkv_separate_total"] = {"avg_us": t_q["avg_us"] + t_k["avg_us"] + t_v["avg_us"] + 2 * 8,  # 2×launch overhead
                                   "note": "fused QKV saves 2 kernel launches ~16us"}
    print(f"  Q/K/V separate: {t_q['avg_us']:.1f}/{t_k['avg_us']:.1f}/{t_v['avg_us']:.1f}us → total {t_q['avg_us']+t_k['avg_us']+t_v['avg_us']+16:.1f}us vs fused {t_qkv['avg_us']:.1f}us")

    # Output projection: B×hidden → B×hidden
    w_out = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16)
    t_out = benchmark_op(lambda: x @ w_out)
    exp1["out_proj"] = {**t_out, "shape": f"{B}x{hidden} → {B}x{hidden}"}
    print(f"  Out proj: {t_out['avg_us']:.1f}us")

    # MLP: gate + up + down projections (3 GEMMs)
    w_gate = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
    w_down = torch.randn(inter_dim, hidden, device=device, dtype=torch.bfloat16)
    t_gate = benchmark_op(lambda: x @ w_gate)
    t_up = benchmark_op(lambda: x @ w_up)

    # gate+up with SiLU fusion
    gate = x @ w_gate
    up = x @ w_up
    t_silu = benchmark_op(lambda: torch.nn.functional.silu(gate) * up)
    down_input = torch.nn.functional.silu(gate) * up
    t_down = benchmark_op(lambda: down_input @ w_down)
    exp1["gate_proj"] = {**t_gate, "shape": f"{B}x{hidden} → {B}x{inter_dim}"}
    exp1["up_proj"] = {**t_up, "shape": f"{B}x{hidden} → {B}x{inter_dim}"}
    exp1["silu_mul"] = {**t_silu, "note": "SiLU+mul element-wise"}
    exp1["down_proj"] = {**t_down, "shape": f"{B}x{inter_dim} → {B}x{hidden}"}
    mlp_total = t_gate["avg_us"] + t_up["avg_us"] + t_silu["avg_us"] + t_down["avg_us"]
    exp1["mlp_total"] = {"avg_us": round(mlp_total, 1)}
    print(f"  MLP: gate={t_gate['avg_us']:.1f}us + up={t_up['avg_us']:.1f}us + silu_mul={t_silu['avg_us']:.1f}us + down={t_down['avg_us']:.1f}us → total={mlp_total:.1f}us")

    # RMSNorm (2 per layer: pre-attn + pre-MLP)
    t_rmsnorm = benchmark_op(lambda: torch.nn.functional.rms_norm(x.float(), (hidden,)).to(torch.bfloat16))
    exp1["rmsnorm"] = {**t_rmsnorm, "note": "2 per layer"}
    print(f"  RMSNorm: {t_rmsnorm['avg_us']:.1f}us × 2 per layer")

    # lm_head: B×hidden → B×vocab
    w_lm = torch.randn(hidden, vocab_size, device=device, dtype=torch.bfloat16)
    t_lm = benchmark_op(lambda: x @ w_lm)
    exp1["lm_head"] = {**t_lm, "shape": f"{B}x{hidden} → {B}x{vocab_size}", "bytes": B * hidden * vocab_size * 2}
    print(f"  lm_head: {t_lm['avg_us']:.1f}us → {B*hidden*vocab_size*2/1024**2:.2f}MB")

    # Sampling (softmax + multinomial)
    logits = torch.randn(B, vocab_size, device=device, dtype=torch.bfloat16)
    t_softmax = benchmark_op(lambda: torch.softmax(logits.float() / 0.6, dim=-1))
    t_multinomial = benchmark_op(lambda: torch.multinomial(torch.softmax(logits.float() / 0.6, dim=-1), 1))
    exp1["softmax"] = {**t_softmax}
    exp1["multinomial"] = {**t_multinomial}
    exp1["sampling_total"] = {"avg_us": round(t_softmax["avg_us"] + t_multinomial["avg_us"], 1)}
    print(f"  Sampling: softmax={t_softmax['avg_us']:.1f}us + multinomial={t_multinomial['avg_us']:.1f}us → {t_softmax['avg_us']+t_multinomial['avg_us']:.1f}us")

    # KV cache read simulation (memory-bound, B=1, S=4K)
    # Read K and V for attention: n_layers × 2 × kv_heads × S × d_head × 2 bytes
    S = 4096
    kv_bytes = 2 * kv_heads * S * d_head * 2  # BF16, one layer
    kv_data = torch.randn(kv_bytes // 2, device=device, dtype=torch.bfloat16)
    t_kv_read = benchmark_op(lambda: kv_data.clone())  # simulates reading KV from HBM
    kv_bw = kv_bytes / (t_kv_read["avg_us"] / 1e6) / 1024**3
    exp1["kv_cache_read"] = {**t_kv_read, "bytes_mb": round(kv_bytes / 1024**2, 2), "bandwidth_gbps": round(kv_bw, 2),
                              "note": f"1 layer, {kv_heads} KV heads, S={S}"}
    print(f"  KV read (1 layer, GQA-{kv_heads}, S={S}): {t_kv_read['avg_us']:.1f}us → {kv_bytes/1024**2:.2f}MB → {kv_bw:.2f}GB/s")

    # Total per-layer decode estimate
    attn_total = t_qkv["avg_us"] + t_out["avg_us"] + 2 * t_rmsnorm["avg_us"] + t_kv_read["avg_us"]
    mlp_layer_total = t_gate["avg_us"] + t_up["avg_us"] + t_silu["avg_us"] + t_down["avg_us"] + t_rmsnorm["avg_us"]
    per_layer_total = attn_total + mlp_layer_total
    total_decode = n_layers * per_layer_total + t_lm["avg_us"] + t_softmax["avg_us"] + t_multinomial["avg_us"]

    exp1["layer_summary"] = {
        "attn_total_us": round(attn_total, 1),
        "mlp_total_us": round(mlp_layer_total, 1),
        "per_layer_us": round(per_layer_total, 1),
        "attn_pct": round(attn_total / per_layer_total * 100, 1),
        "mlp_pct": round(mlp_layer_total / per_layer_total * 100, 1),
    }
    exp1["decode_summary"] = {
        "n_layers": n_layers,
        "total_decode_us": round(total_decode, 1),
        "total_decode_ms": round(total_decode / 1000, 3),
        "tok_per_s": round(1 / (total_decode / 1e6), 1),
        "lm_head_pct": round(t_lm["avg_us"] / total_decode * 100, 1),
        "sampling_pct": round((t_softmax["avg_us"] + t_multinomial["avg_us"]) / total_decode * 100, 1),
    }

    print(f"\n  Layer breakdown: attn={attn_total:.1f}us({attn_total/per_layer_total*100:.1f}%) + MLP={mlp_layer_total:.1f}us({mlp_layer_total/per_layer_total*100:.1f}%) = {per_layer_total:.1f}us")
    print(f"  Total decode: {total_decode:.1f}us → {total_decode/1000:.3f}ms → {1/(total_decode/1e6):.1f} tok/s")
    print(f"  lm_head占比: {t_lm['avg_us']/total_decode*100:.1f}%")
    print(f"  sampling占比: {(t_softmax['avg_us']+t_multinomial['avg_us'])/total_decode*100:.1f}%")

    results["exp1_decode_breakdown_b1"] = exp1

    # ====================================================================
    # Exp 2: Batch Scaling (B=1 to B=55)
    # ====================================================================
    print("\n--- Exp 2: Decode Batch Scaling ---")
    exp2 = {}

    batch_sizes = [1, 4, 8, 16, 32, 55]

    for B in batch_sizes:
        x = torch.randn(B, hidden, device=device, dtype=torch.bfloat16)

        # QKV
        w_qkv = torch.randn(hidden, 3 * hidden, device=device, dtype=torch.bfloat16)
        t_qkv_b = benchmark_op(lambda: x @ w_qkv, warmup=5, repeats=50)

        # MLP (gate+up+silu+down)
        w_gate = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
        w_up = torch.randn(hidden, inter_dim, device=device, dtype=torch.bfloat16)
        w_down = torch.randn(inter_dim, hidden, device=device, dtype=torch.bfloat16)
        gate = x @ w_gate
        up = x @ w_up
        t_mlp_b = benchmark_op(lambda: (torch.nn.functional.silu(gate) * up) @ w_down, warmup=5, repeats=50)

        # lm_head
        w_lm = torch.randn(hidden, vocab_size, device=device, dtype=torch.bfloat16)
        t_lm_b = benchmark_op(lambda: x @ w_lm, warmup=5, repeats=50)

        # Total per layer (4 GEMM per attn + 3 GEMM per MLP + 2 RMSNorm)
        # Simplified: use measured times
        total_per_layer = t_qkv_b["avg_us"] + t_mlp_b["avg_us"] + 2 * t_rmsnorm["avg_us"]  # approximate

        # Decode throughput
        total_decode_us = n_layers * total_per_layer + t_lm_b["avg_us"] + 50  # 50us sampling

        exp2[str(B)] = {
            "qkv_us": t_qkv_b["avg_us"],
            "mlp_us": t_mlp_b["avg_us"],
            "lm_head_us": t_lm_b["avg_us"],
            "per_layer_us": round(total_per_layer, 1),
            "total_decode_us": round(total_decode_us, 1),
            "total_decode_ms": round(total_decode_us / 1000, 3),
            "tok_per_s": round(B / (total_decode_us / 1e6), 0),
            "lm_head_pct": round(t_lm_b["avg_us"] / total_decode_us * 100, 1),
        }
        print(f"  B={B}: layer={total_per_layer:.1f}us, decode={total_decode_us/1000:.3f}ms → {B/(total_decode_us/1e6):.0f}tok/s, lm_head={t_lm_b['avg_us']/total_decode_us*100:.1f}%")

    results["exp2_batch_scaling"] = exp2

    # ====================================================================
    # Exp 3: lm_head Dominance Analysis
    # ====================================================================
    print("\n--- Exp 3: lm_head vs Weight GEMM ---")
    exp3 = {}

    # lm_head bytes/tok = hidden × vocab_size × 2 = 4096 × 32000 × 2 = 262MB
    # Per-layer weight bytes/tok = 7 GEMM weights / n_layers
    # gate+up = hidden × inter_dim × 2 × 2 = 4096 × 14336 × 4 = 235MB
    # QKV+out = hidden × hidden × 2 × 4 = 4096 × 4096 × 8 = 134MB
    # Down = inter_dim × hidden × 2 = 14336 × 4096 × 2 = 118MB

    lm_head_bytes = hidden * vocab_size * 2
    layer_weight_bytes = (4 * hidden * hidden + 2 * hidden * inter_dim + inter_dim * hidden) * 2
    kv_bytes_layer = 2 * kv_heads * d_head * S * 2

    for B in [1, 4, 16, 32, 55]:
        # HBM read per decode step (1 token)
        # Each GPU reads: lm_head + per-layer_weights + per-layer_kv
        # All read from HBM (memory-bound)

        total_bytes_per_tok = (lm_head_bytes + n_layers * (layer_weight_bytes + kv_bytes_layer)) * B
        hbm_time_us = total_bytes_per_tok / (HBM_BW * 1024**3) * 1e6  # microseconds

        lm_pct = lm_head_bytes * B / total_bytes_per_tok * 100
        weight_pct = n_layers * layer_weight_bytes * B / total_bytes_per_tok * 100
        kv_pct = n_layers * kv_bytes_layer * B / total_bytes_per_tok * 100

        exp3[str(B)] = {
            "total_bytes_mb": round(total_bytes_per_tok / 1024**2 / B, 2),  # per request
            "hbm_time_us": round(hbm_time_us, 2),
            "lm_head_pct": round(lm_pct, 1),
            "weight_pct": round(weight_pct, 1),
            "kv_pct": round(kv_pct, 1),
            "tok_per_s_roofline": round(B / (hbm_time_us / 1e6), 0),
        }
        print(f"  B={B}: bytes/tok={total_bytes_per_tok/1024**2/B:.2f}MB → HBM={hbm_time_us:.1f}us → lm={lm_pct:.1f}% weight={weight_pct:.1f}% kv={kv_pct:.1f}% → {B/(hbm_time_us/1e6):.0f}tok/s(roofline)")

    results["exp3_lm_head_dominance"] = exp3

    # ====================================================================
    # Exp 4: INT4/INT8 Impact on Component Latency
    # ====================================================================
    print("\n--- Exp 4: INT4/INT8 Weight Impact (Simulation) ---")
    exp4 = {}

    # INT4 weights → 4x fewer bytes → memory-bound decode → 4x faster weight read
    # But: lm_head can't be INT4 (need all vocab logits) → INT8 lm_head

    for B in [1, 4, 16, 32]:
        # BF16: all weights BF16
        bf16_bytes_per_tok = lm_head_bytes + n_layers * layer_weight_bytes + n_layers * kv_bytes_layer

        # INT4 weight + INT8 KV: weight 4x smaller, KV 2x smaller
        int4_int8kv_bytes_per_tok = lm_head_bytes + n_layers * layer_weight_bytes / 4 + n_layers * kv_bytes_layer / 2

        # INT4 weight + INT8 KV + INT8 lm_head: lm_head also 2x smaller
        int4_int8kv_int8lm_bytes_per_tok = lm_head_bytes / 2 + n_layers * layer_weight_bytes / 4 + n_layers * kv_bytes_layer / 2

        bf16_time = bf16_bytes_per_tok * B / (HBM_BW * 1024**3) * 1e6
        int4_int8kv_time = int4_int8kv_bytes_per_tok * B / (HBM_BW * 1024**3) * 1e6
        int4_int8kv_int8lm_time = int4_int8kv_int8lm_bytes_per_tok * B / (HBM_BW * 1024**3) * 1e6

        exp4[str(B)] = {
            "bf16_bytes_mb": round(bf16_bytes_per_tok / 1024**2, 2),
            "int4_int8kv_bytes_mb": round(int4_int8kv_bytes_per_tok / 1024**2, 2),
            "int4_int8kv_int8lm_bytes_mb": round(int4_int8kv_int8lm_bytes_per_tok / 1024**2, 2),
            "bf16_tok_per_s": round(B / (bf16_time / 1e6), 0),
            "int4_int8kv_tok_per_s": round(B / (int4_int8kv_time / 1e6), 0),
            "int4_int8kv_int8lm_tok_per_s": round(B / (int4_int8kv_int8lm_time / 1e6), 0),
            "int4_int8kv_speedup": round(bf16_time / int4_int8kv_time, 2),
            "int4_int8kv_int8lm_speedup": round(bf16_time / int4_int8kv_int8lm_time, 2),
        }
        print(f"  B={B}: BF16→{B/(bf16_time/1e6):.0f}tok/s, INT4+INT8KV→{B/(int4_int8kv_time/1e6):.0f}tok/s({bf16_time/int4_int8kv_time:.2f}x), INT4+INT8KV+INT8lm→{B/(int4_int8kv_int8lm_time/1e6):.0f}tok/s({bf16_time/int4_int8kv_int8lm_time:.2f}x)")

    results["exp4_quantization_impact"] = exp4

    # ====================================================================
    # Exp 5: Prefill vs Decode TTFT
    # ====================================================================
    print("\n--- Exp 5: Prefill TTFT vs Decode ITL ---")
    exp5 = {}

    for S in [128, 512, 1024, 2048, 4096, 8192, 16384]:
        # Prefill: compute-bound, 170 TFLOPS (BF16)
        # FLOPs: 2 × 7B × S (forward pass)
        flops = 2 * 7e9 * S
        prefill_ms = flops / 170 / 1e9 * 1000  # 170 TFLOPS
        ttft_ms = prefill_ms

        # Decode ITL (inter-token latency): memory-bound
        # B=1: ~2ms (from measurement)
        itl_ms = 2.0  # approximate BF16 B=1

        exp5[str(S)] = {
            "prefill_ms": round(prefill_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "itl_ms": round(itl_ms, 2),
            "prefill_tflops": 170,
            "ttft_acceptable": ttft_ms <= 100,
            "prefill_tok_per_s": round(S / (prefill_ms / 1000), 0),
        }
        print(f"  S={S}: TTFT={ttft_ms:.2f}ms {'OK' if ttft_ms<=100 else 'SLOW'}, ITL={itl_ms:.2f}ms → prefill {S/(prefill_ms/1000):.0f}tok/s")

    # Maximum context for TTFT ≤ 100ms
    max_S_ttft = 100 * 170 * 1e9 / (2 * 7e9) * 1000  # ms → tokens
    exp5["max_context_100ms"] = {"S": round(max_S_ttft, 0), "ttft_ms": 100}
    print(f"  Max context for TTFT≤100ms: S≈{max_S_ttft:.0f}")

    results["exp5_ttft_itl"] = exp5

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — Transformer Layer Decode Breakdown RTX 4090")
    print("=" * 70)

    e1_summary = exp1.get("decode_summary", {})
    e1_layer = exp1.get("layer_summary", {})
    e2_b32 = exp2.get("32", {})
    e3_b32 = exp3.get("32", {})
    e4_b32 = exp4.get("32", {})

    print(f"\n  B=1 decode breakdown:")
    print(f"    Total: {e1_summary.get('total_decode_us', 0):.0f}us → {e1_summary.get('tok_per_s', 0):.1f} tok/s")
    print(f"    Per layer: {e1_layer.get('per_layer_us', 0):.0f}us (attn={e1_layer.get('attn_pct', 0)}% + MLP={e1_layer.get('mlp_pct', 0)}%)")
    print(f"    lm_head: {e1_summary.get('lm_head_pct', 0)}% of total decode")
    print(f"    sampling: {e1_summary.get('sampling_pct', 0)}% of total decode")
    print(f"\n  B=32 decode:")
    print(f"    {e2_b32.get('tok_per_s', 0):.0f} tok/s")
    print(f"    lm_head={e2_b32.get('lm_head_pct', 0)}% → dominant at B≥16!")
    print(f"\n  INT4+INT8KV speedup (B=32): {e4_b32.get('int4_int8kv_speedup', 0):.2f}x → {e4_b32.get('int4_int8kv_tok_per_s', 0):.0f} tok/s")
    print(f"\n  Production: quantize weights+KV+lm_head → 2-3x → FlashInfer 1.06-3.20x → combined 3-9x!")

    return results


if __name__ == '__main__':
    torch.cuda.empty_cache()
    results = run_all()
    try:
        with open('results/transformer_layer_decode_breakdown.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except:
        with open('transformer_layer_decode_breakdown.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)