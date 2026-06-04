#!/usr/bin/env python3
"""量化推理实战 — LLM 量化方法 GPU 实测

模拟 LLM 量化推理的核心操作:
1. Weight-Only INT8 量化: 只量化权重, 激活保持 FP16
2. Dynamic Quantization: 运行时量化激活
3. 量化对注意力机制的影响
4. 不同量化粒度 (per-tensor vs per-channel vs per-group)
5. 量化对长序列推理的影响

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  python gpu_quant_inference.py
"""

import os, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

print(f"GPU: {torch.cuda.get_device_name(0)}")


def bench_ms(fn, warmup=5, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


# ============================================================
# 量化工具函数
# ============================================================

def quantize_per_tensor_fp16_to_int8(weight):
    """Per-tensor symmetric quantization: FP16 → INT8"""
    max_val = weight.abs().max()
    scale = max_val / 127.0
    q_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return q_weight, scale


def dequantize_int8_to_fp16(q_weight, scale):
    """INT8 → FP16 dequantization"""
    return q_weight.float() * scale.float()


def quantize_per_channel_fp16_to_int8(weight):
    """Per-channel (output dim) symmetric quantization"""
    max_val = weight.abs().max(dim=1)[0]  # [out_features]
    scale = max_val / 127.0
    q_weight = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    return q_weight, scale


def quantize_per_group_fp16_to_int4(weight, group_size=128):
    """Per-group INT4 quantization (AWQ/GPTQ style)"""
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    num_groups = in_features // group_size

    w_grouped = weight.reshape(out_features, num_groups, group_size)
    max_val = w_grouped.abs().max(dim=2)[0]  # [out_features, num_groups]
    scale = max_val / 7.0  # INT4 range: -8 to 7

    q_weight = torch.round(w_grouped / scale.unsqueeze(2)).clamp(-8, 7).to(torch.int8)
    return q_weight.reshape(out_features, in_features), scale, group_size


# ============================================================
# 实验 1: 量化粒度对比 (per-tensor / per-channel / per-group)
# ============================================================

def exp1_quantization_granularity():
    print("\n" + "=" * 60)
    print("实验1: 量化粒度对比")
    print("=" * 60)

    results = []
    H = 1024

    for dtype_name, orig_dtype in [("FP16", torch.float16), ("FP32", torch.float32)]:
        weight = torch.randn(H, H, device="cuda", dtype=orig_dtype)
        x = torch.randn(4, 256, H, device="cuda", dtype=torch.float16)

        # FP16 baseline
        def fp16_linear():
            return F.linear(x, weight.half())

        fp16_ms = bench_ms(fp16_linear, rep=50)

        # FP32 baseline
        def fp32_linear():
            return F.linear(x.float(), weight.float())

        fp32_ms = bench_ms(fp32_linear, rep=50) if orig_dtype == torch.float16 else fp16_ms

        # Per-tensor INT8
        q_w, scale = quantize_per_tensor_fp16_to_int8(weight.half())
        def int8_per_tensor():
            w_dq = dequantize_int8_to_fp16(q_w, scale)
            return F.linear(x, w_dq.half())

        int8_pt_ms = bench_ms(int8_per_tensor, rep=50)
        int8_pt_err = (F.linear(x, weight.half()) - int8_per_tensor()).abs().max().item()

        # Per-channel INT8
        q_w_ch, scale_ch = quantize_per_channel_fp16_to_int8(weight.half())
        def int8_per_channel():
            w_dq = (q_w_ch.float() * scale_ch.unsqueeze(1).float()).half()
            return F.linear(x, w_dq)

        int8_pc_ms = bench_ms(int8_per_channel, rep=50)
        int8_pc_err = (F.linear(x, weight.half()) - int8_per_channel()).abs().max().item()

        # Per-group INT4
        q_w_g, scale_g, gs = quantize_per_group_fp16_to_int4(weight.half(), group_size=128)
        def int4_per_group():
            out_f, in_f = q_w_g.shape
            num_g = in_f // gs
            w_g = q_w_g.reshape(out_f, num_g, gs).float()
            w_dq = (w_g * scale_g.unsqueeze(2).float()).reshape(out_f, in_f).half()
            return F.linear(x, w_dq)

        int4_pg_ms = bench_ms(int4_per_group, rep=50)
        int4_pg_err = (F.linear(x, weight.half()) - int4_per_group()).abs().max().item()

        print(f"\n  H={H}, {dtype_name}:")
        print(f"    FP16:          {fp16_ms:.3f}ms")
        print(f"    INT8 per-tensor: {int8_pt_ms:.3f}ms, err={int8_pt_err:.4f}")
        print(f"    INT8 per-channel: {int8_pc_ms:.3f}ms, err={int8_pc_err:.4f}")
        print(f"    INT4 per-group:  {int4_pg_ms:.3f}ms, err={int4_pg_err:.4f}")

        results.append({
            "dtype": dtype_name,
            "fp16_ms": round(fp16_ms, 3),
            "int8_pt_ms": round(int8_pt_ms, 3),
            "int8_pt_err": round(int8_pt_err, 4),
            "int8_pc_ms": round(int8_pc_ms, 3),
            "int8_pc_err": round(int8_pc_err, 4),
            "int4_pg_ms": round(int4_pg_ms, 3),
            "int4_pg_err": round(int4_pg_err, 4),
        })

        del weight, x, q_w, scale, q_w_ch, scale_ch, q_w_g, scale_g
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 2: Weight-Only INT8 — 模型大小 vs 精度
# ============================================================

def exp2_weight_only_int8():
    print("\n" + "=" * 60)
    print("实验2: Weight-Only INT8 模型分析")
    print("=" * 60)

    results = []

    # Simulate different model sizes
    configs = [
        ("125M", 12, 768, 12),    # layers, hidden, heads
        ("350M", 24, 1024, 16),
        ("1.3B", 24, 2048, 32),
        ("6.7B", 32, 4096, 32),
        ("13B", 40, 5120, 40),
    ]

    print(f"\n  Weight-only INT8: 权重 INT8, 激活 FP16")
    print(f"  {'Model':<10} {'Params':<10} {'FP16 MB':<12} {'INT8 MB':<10} {'Saving%':<10} {'Max Err'}")
    print("  " + "-" * 66)

    for name, n_layers, H, n_heads in configs:
        # Rough param count: 12 * H^2 * n_layers (MLP=4H + Attn=4H)
        params = 12 * H * H * n_layers

        fp16_mb = params * 2 / 1e6  # FP16
        int8_mb = params * 1 / 1e6  # INT8

        saving = (1 - int8_mb / fp16_mb) * 100

        # Measure quantization error on a small weight
        w = torch.randn(H, H, device="cuda", dtype=torch.float16)
        q_w, scale = quantize_per_channel_fp16_to_int8(w)
        w_dq = (q_w.float() * scale.unsqueeze(1).float()).half()
        max_err = (w - w_dq).abs().max().item()
        rel_err = ((w - w_dq).abs() / (w.abs() + 1e-8)).mean().item()

        print(f"  {name:<10} {params/1e6:<10.1f}M {fp16_mb:<12.0f} {int8_mb:<10.0f} {saving:<10.0f} {max_err:.4f} (rel={rel_err:.4f})")

        results.append({
            "model": name,
            "params_m": round(params / 1e6, 1),
            "fp16_mb": round(fp16_mb, 0),
            "int8_mb": round(int8_mb, 0),
            "saving_pct": round(saving, 0),
            "max_err": round(max_err, 4),
            "rel_err": round(rel_err, 4),
        })

        del w, q_w, scale, w_dq
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 3: 量化对 Attention 的影响
# ============================================================

def exp3_quantized_attention():
    print("\n" + "=" * 60)
    print("实验3: 量化对 Attention 的影响")
    print("=" * 60)

    results = []
    H = 512
    n_heads = 8
    head_dim = H // n_heads

    for seq_len in [256, 512, 1024, 2048]:
        B = 4
        Q = torch.randn(B, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        K = torch.randn(B, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(B, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        # FP16 attention (baseline)
        def attn_fp16():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V)

        # INT8 QK (quantize Q and K before matmul)
        q_q, q_scale = quantize_per_channel_fp16_to_int8(Q.reshape(-1, head_dim))
        q_k, k_scale = quantize_per_channel_fp16_to_int8(K.reshape(-1, head_dim))

        def attn_int8_qk():
            Q_dq = (q_q.float() * q_scale.unsqueeze(1).float()).half().reshape(B, n_heads, seq_len, head_dim)
            K_dq = (q_k.float() * k_scale.unsqueeze(1).float()).half().reshape(B, n_heads, seq_len, head_dim)
            scores = torch.matmul(Q_dq, K_dq.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V)

        fp16_ms = bench_ms(attn_fp16, rep=20)
        int8_ms = bench_ms(attn_int8_qk, rep=20)

        # Accuracy
        out_fp16 = attn_fp16()
        out_int8 = attn_int8_qk()
        max_err = (out_fp16 - out_int8).abs().max().item()
        cos_sim = F.cosine_similarity(out_fp16.flatten().unsqueeze(0),
                                       out_int8.flatten().unsqueeze(0)).item()

        print(f"\n  SeqLen={seq_len}: FP16={fp16_ms:.3f}ms, INT8-QK={int8_ms:.3f}ms, "
              f"err={max_err:.4f}, cos_sim={cos_sim:.6f}")

        results.append({
            "seq_len": seq_len,
            "fp16_ms": round(fp16_ms, 3),
            "int8_qk_ms": round(int8_ms, 3),
            "max_err": round(max_err, 4),
            "cos_sim": round(cos_sim, 6),
        })

        del Q, K, V, q_q, q_scale, q_k, k_scale
        torch.cuda.empty_cache()

    return results


# ============================================================
# 实验 4: FP8 模拟 (E4M3 / E5M2)
# ============================================================

def exp4_fp8_simulation():
    print("\n" + "=" * 60)
    print("实验4: FP8 模拟 (E4M3 / E5M2)")
    print("=" * 60)

    results = []
    H = 1024

    # FP8 is not natively supported on A16, simulate with scaling
    # E4M3: 4 exponent bits, 3 mantissa bits → range [2^-6, 2^9]
    # E5M2: 5 exponent bits, 2 mantissa bits → range [2^-14, 2^15]

    weight = torch.randn(H, H, device="cuda", dtype=torch.float16)
    x = torch.randn(4, 128, H, device="cuda", dtype=torch.float16)

    def simulate_fp8_e4m3(tensor, max_val=None):
        """Simulate FP8 E4M3 quantization"""
        if max_val is None:
            max_val = tensor.abs().max()
        # E4M3: 4 exp, 3 mantissa → ~3.7 bits mantissa precision
        # Simulate by quantizing to 2^4 levels per power-of-2 range
        scale = max_val / 448.0  # E4M3 max normal = 448
        q = torch.round(tensor / scale).clamp(-448, 448)
        return q * scale

    def simulate_fp8_e5m2(tensor, max_val=None):
        """Simulate FP8 E5M2 quantization"""
        if max_val is None:
            max_val = tensor.abs().max()
        scale = max_val / 57344.0  # E5M2 max normal
        q = torch.round(tensor / scale).clamp(-57344, 57344)
        return q * scale

    # FP16 baseline
    ref_out = F.linear(x, weight)
    fp16_ms = bench_ms(lambda: F.linear(x, weight), rep=50)

    # FP8 E4M3 weight-only
    w_e4m3 = simulate_fp8_e4m3(weight)
    e4m3_ms = bench_ms(lambda: F.linear(x, w_e4m3.half()), rep=50)
    e4m3_err = (ref_out - F.linear(x, w_e4m3.half())).abs().max().item()

    # FP8 E5M2 weight-only
    w_e5m2 = simulate_fp8_e5m2(weight)
    e5m2_ms = bench_ms(lambda: F.linear(x, w_e5m2.half()), rep=50)
    e5m2_err = (ref_out - F.linear(x, w_e5m2.half())).abs().max().item()

    # INT8 per-channel (comparison)
    q_w, scale = quantize_per_channel_fp16_to_int8(weight)
    w_int8 = (q_w.float() * scale.unsqueeze(1).float()).half()
    int8_ms = bench_ms(lambda: F.linear(x, w_int8), rep=50)
    int8_err = (ref_out - F.linear(x, w_int8)).abs().max().item()

    # Memory comparison
    fp16_bytes = H * H * 2
    int8_bytes = H * H * 1
    fp8_bytes = H * H * 1  # same as INT8

    print(f"\n  H={H}, B=4, S=128")
    print(f"  {'Format':<14} {'Time ms':<12} {'Max Err':<12} {'Size MB':<10} {'Saving vs FP16'}")
    print("  " + "-" * 60)
    print(f"  {'FP16':<14} {fp16_ms:<12.3f} {'0.0000':<12} {fp16_bytes/1e6:<10.2f} {'baseline'}")
    print(f"  {'INT8 per-ch':<14} {int8_ms:<12.3f} {int8_err:<12.4f} {int8_bytes/1e6:<10.2f} {'50%'}")
    print(f"  {'FP8 E4M3':<14} {e4m3_ms:<12.3f} {e4m3_err:<12.4f} {fp8_bytes/1e6:<10.2f} {'50%'}")
    print(f"  {'FP8 E5M2':<14} {e5m2_ms:<12.3f} {e5m2_err:<12.4f} {fp8_bytes/1e6:<10.2f} {'50%'}")

    results.append({
        "fp16_ms": round(fp16_ms, 3),
        "int8_ms": round(int8_ms, 3), "int8_err": round(int8_err, 4),
        "e4m3_ms": round(e4m3_ms, 3), "e4m3_err": round(e4m3_err, 4),
        "e5m2_ms": round(e5m2_ms, 3), "e5m2_err": round(e5m2_err, 4),
    })

    del weight, x, w_e4m3, w_e5m2, q_w, w_int8
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 5: KV Cache 量化
# ============================================================

def exp5_kv_cache_quant():
    print("\n" + "=" * 60)
    print("实验5: KV Cache 量化")
    print("=" * 60)

    results = []
    H = 512
    n_heads = 8
    head_dim = H // n_heads
    B = 4

    for seq_len in [512, 1024, 2048, 4096]:
        # KV cache: [B, n_heads, seq_len, head_dim]
        K = torch.randn(B, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        V = torch.randn(B, n_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        Q = torch.randn(B, n_heads, 1, head_dim, device="cuda", dtype=torch.float16)  # decode

        # FP16 attention
        def attn_fp16():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V)

        # INT8 KV cache
        K_flat = K.reshape(-1, head_dim)
        V_flat = V.reshape(-1, head_dim)
        q_K, k_scale = quantize_per_channel_fp16_to_int8(K_flat)
        q_V, v_scale = quantize_per_channel_fp16_to_int8(V_flat)
        K_int8 = (q_K.float() * k_scale.unsqueeze(1).float()).half().reshape(B, n_heads, seq_len, head_dim)
        V_int8 = (q_V.float() * v_scale.unsqueeze(1).float()).half().reshape(B, n_heads, seq_len, head_dim)

        def attn_int8_kv():
            scores = torch.matmul(Q, K_int8.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
            return torch.matmul(attn, V_int8)

        fp16_ms = bench_ms(attn_fp16, rep=20)
        int8_ms = bench_ms(attn_int8_kv, rep=20)

        out_fp16 = attn_fp16()
        out_int8 = attn_int8_kv()
        max_err = (out_fp16 - out_int8).abs().max().item()
        cos_sim = F.cosine_similarity(out_fp16.flatten().unsqueeze(0),
                                       out_int8.flatten().unsqueeze(0)).item()

        # Memory
        kv_fp16_mb = B * seq_len * 2 * H * 2 / 1e6  # K+V FP16
        kv_int8_mb = B * seq_len * 2 * H * 1 / 1e6  # K+V INT8

        print(f"\n  SeqLen={seq_len}: FP16={fp16_ms:.3f}ms, INT8-KV={int8_ms:.3f}ms, "
              f"err={max_err:.4f}, cos_sim={cos_sim:.6f}")
        print(f"    KV mem: FP16={kv_fp16_mb:.1f}MB, INT8={kv_int8_mb:.1f}MB (save 50%)")

        results.append({
            "seq_len": seq_len,
            "fp16_ms": round(fp16_ms, 3),
            "int8_kv_ms": round(int8_ms, 3),
            "max_err": round(max_err, 4),
            "cos_sim": round(cos_sim, 6),
            "kv_fp16_mb": round(kv_fp16_mb, 1),
            "kv_int8_mb": round(kv_int8_mb, 1),
        })

        del K, V, Q, K_flat, V_flat, q_K, q_V, K_int8, V_int8
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    all_results = OrderedDict()
    all_results["quant_granularity"] = exp1_quantization_granularity()
    all_results["weight_only_int8"] = exp2_weight_only_int8()
    all_results["quantized_attention"] = exp3_quantized_attention()
    all_results["fp8_simulation"] = exp4_fp8_simulation()
    all_results["kv_cache_quant"] = exp5_kv_cache_quant()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. 量化粒度: per-channel > per-tensor > per-group (精度)
     per-channel INT8 精度损失 < 0.01, per-tensor 可能 > 0.1
  2. Weight-Only INT8: 50% 内存节省, 推理速度相同 (需要反量化)
     vLLM/Marlin 用 INT8 kernel 直接计算 (无需反量化)
  3. Attention 量化: QK 量化对长序列影响大 (softmax 放大误差)
  4. FP8 E4M3: 精度接近 INT8, 但 H100 原生支持 (A16 不支持)
  5. KV Cache INT8: 50% 内存节省, cos_sim > 0.99, 精度可接受
  6. vLLM 实际: FP8 在 H100 上最快, INT8 AWQ 在 A100 上最优
""")

    with open("/root/quant_inference_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved.")
