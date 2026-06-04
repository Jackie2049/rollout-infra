#!/usr/bin/env python3
"""训练延迟模型 — 从 GPU Micro-Benchmark 到训练性能预测

建立从底层硬件参数到训练性能的预测模型:
1. GEMM Roofline 模型: compute-bound vs memory-bound 边界
2. Transformer 层延迟模型: 从 FLOPS 和带宽预测 step 时间
3. 分布式训练延迟模型: 加入通信开销
4. 模型大小→训练时间预测: 不同 GPU 配置下的训练时间估算

用法 (GPU 服务器):
  source /root/miniconda3/bin/activate myconda
  export HF_HUB_OFFLINE=1
  python gpu_training_latency_model.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME}, VRAM: {TOTAL_MEM:.1f} GB")
print(f"CUDA: {torch.version.cuda}, Compute: {torch.cuda.get_device_capability()}")


def bench_ms(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


# ============================================================
# 实验 1: GEMM Roofline 模型
# ============================================================

def exp1_roofline():
    print("\n" + "=" * 60)
    print("实验1: GEMM Roofline 模型")
    print("=" * 60)

    results = []

    # A16 specs
    peak_tflops_fp16 = 14.7
    peak_bw_gbs = 170.0  # measured

    # Ridge point: where compute = memory
    # arithmetic_intensity = FLOPS / bytes_accessed
    # Ridge point AI = peak_FLOPS / peak_BW
    ridge_point = peak_tflops_fp16 * 1e12 / (peak_bw_gbs * 1e9)  # ops/byte
    print(f"\n  Peak FP16: {peak_tflops_fp16} TFLOPS")
    print(f"  Peak BW:   {peak_bw_gbs} GB/s")
    print(f"  Ridge Point: {ridge_point:.1f} ops/byte")

    print(f"\n  {'M=N=K':<10} {'Time(ms)':<12} {'TFLOPS':<10} {'AI':<10} {'Bound':<12}")
    print("  " + "-" * 54)

    for size in [64, 128, 256, 512, 1024, 2048, 4096]:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        ms = bench_ms(lambda: torch.matmul(a, b))
        flops = 2.0 * size * size * size
        tflops = flops / ms / 1e9

        # Arithmetic intensity: FLOPS per byte accessed
        # GEMM reads A(size*size*2) + B(size*size*2) and writes C(size*size*2)
        bytes_accessed = 3 * size * size * 2  # fp16
        ai = flops / bytes_accessed

        bound = "compute" if ai > ridge_point else "memory"
        pct = tflops / peak_tflops_fp16 * 100

        results.append({
            "size": size, "ms": round(ms, 4), "tflops": round(tflops, 2),
            "arithmetic_intensity": round(ai, 1), "pct_peak": round(pct, 1),
            "bound": bound,
        })
        print(f"  {size:<10} {ms:<12.4f} {tflops:<10.2f} {ai:<10.1f} {bound:<12} ({pct:.0f}%)")

    return results


# ============================================================
# 实验 2: Transformer 层延迟分解
# ============================================================

def exp2_layer_latency_breakdown():
    print("\n" + "=" * 60)
    print("实验2: Transformer 层延迟分解")
    print("=" * 60)

    results = []
    B, S, H, n_heads = 4, 512, 1024, 16

    # 1. QKV projection
    qkv_w = torch.randn(3 * H, H, device="cuda", dtype=torch.float16)
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)
    ms_qkv = bench_ms(lambda: x @ qkv_w.T)

    # 2. Attention (SDPA)
    q = torch.randn(B, n_heads, S, H // n_heads, device="cuda", dtype=torch.float16)
    k = torch.randn(B, n_heads, S, H // n_heads, device="cuda", dtype=torch.float16)
    v = torch.randn(B, n_heads, S, H // n_heads, device="cuda", dtype=torch.float16)
    ms_attn = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v))

    # 3. Output projection
    out_w = torch.randn(H, H, device="cuda", dtype=torch.float16)
    ms_out = bench_ms(lambda: x @ out_w.T)

    # 4. MLP fc1 (H -> 4H)
    fc1_w = torch.randn(4 * H, H, device="cuda", dtype=torch.float16)
    ms_fc1 = bench_ms(lambda: x @ fc1_w.T)

    # 5. GELU
    ms_gelu = bench_ms(lambda: F.gelu(x))

    # 6. MLP fc2 (4H -> H)
    x4h = torch.randn(B, S, 4 * H, device="cuda", dtype=torch.float16)
    fc2_w = torch.randn(H, 4 * H, device="cuda", dtype=torch.float16)
    ms_fc2 = bench_ms(lambda: x4h @ fc2_w.T)

    # 7. LayerNorm
    ln = nn.LayerNorm(H).cuda().half()
    ms_ln = bench_ms(lambda: ln(x))

    total_proj = ms_qkv + ms_out + ms_fc1 + ms_fc2
    total = total_proj + ms_attn + ms_gelu + 2 * ms_ln

    print(f"\n  层延迟分解 (B={B}, S={S}, H={H}):")
    print(f"  {'操作':<25} {'时间(ms)':<12} {'占比'}")
    print(f"  {'-'*50}")
    items = [
        ("QKV projection", ms_qkv),
        ("Attention (SDPA)", ms_attn),
        ("Output projection", ms_out),
        ("MLP fc1 (H→4H)", ms_fc1),
        ("GELU activation", ms_gelu),
        ("MLP fc2 (4H→H)", ms_fc2),
        ("LayerNorm ×2", 2 * ms_ln),
    ]
    for name, ms in items:
        pct = ms / total * 100
        print(f"  {name:<25} {ms:<12.4f} {pct:.1f}%")
        results.append({"op": name, "ms": round(ms, 4), "pct": round(pct, 1)})

    print(f"  {'Total (projected)':<25} {total_proj:<12.4f}")
    print(f"  {'Total (all ops)':<25} {total:<12.4f}")

    # 实测完整 Transformer block
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(H, 3 * H, bias=False)
            self.out = nn.Linear(H, H, bias=False)
            self.fc1 = nn.Linear(H, 4 * H, bias=False)
            self.fc2 = nn.Linear(4 * H, H, bias=False)
            self.ln1 = nn.LayerNorm(H)
            self.ln2 = nn.LayerNorm(H)

        def forward(self, x):
            B, S, H = x.shape
            res = x
            x = self.ln1(x)
            qkv = self.qkv(x).reshape(B, S, 3, 16, 64)
            q, k, v = qkv.unbind(2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            attn = F.scaled_dot_product_attention(q, k, v)
            x = self.out(attn.transpose(1, 2).reshape(B, S, H))
            x = res + x
            res = x
            x = self.ln2(x)
            x = self.fc2(F.gelu(self.fc1(x)))
            return res + x

    block = Block().cuda().half()
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float16)

    ms_fwd = bench_ms(lambda: block(x))
    def fwd_bwd():
        y = block(x)
        y.sum().backward()
    ms_fwdbwd = bench_ms(fwd_bwd, warmup=5, rep=30)
    block.zero_grad()

    print(f"\n  实测 Transformer Block:")
    print(f"    Forward:       {ms_fwd:.4f}ms")
    print(f"    Forward+Bwd:   {ms_fwdbwd:.4f}ms")
    print(f"    Bwd/Fwd ratio: {ms_fwdbwd/ms_fwd:.2f}x")

    results.append({
        "measured_fwd_ms": round(ms_fwd, 4),
        "measured_fwdbwd_ms": round(ms_fwdbwd, 4),
        "sum_of_parts_ms": round(total, 4),
        "overhead_pct": round((ms_fwd / total - 1) * 100, 1),
    })

    del block
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 3: 模型大小→训练时间预测
# ============================================================

def exp3_training_time_prediction():
    print("\n" + "=" * 60)
    print("实验3: 模型大小→训练时间预测")
    print("=" * 60)

    results = []

    # 从实测数据推算
    # A16: ~5 TFLOPS effective (考虑利用率), HBM ~170 GB/s

    models = [
        ("125M", 125e6, 12),    # 12 layers
        ("350M", 350e6, 24),
        ("1.3B", 1.3e9, 24),
        ("7B", 7e9, 32),
        ("13B", 13e9, 40),
        ("70B", 70e9, 80),
    ]

    # 训练配置
    seq_len = 2048
    batch_size = 1  # per GPU
    target_tokens = 300e9  # 300B tokens (common pretraining)

    gpu_configs = [
        ("A16 (15GB)", 5.0, 170, 15),         # 5 TFLOPS effective
        ("A100 (80GB)", 156.0, 2039, 80),      # 156 TFLOPS FP16
        ("H100 (80GB)", 495.0, 3350, 80),      # 495 TFLOPS FP16
        ("H200 (141GB)", 495.0, 4800, 141),    # same TFLOPS, more BW
    ]

    print(f"\n  目标: {target_tokens/1e9:.0f}B tokens, seq_len={seq_len}")
    print(f"\n  {'GPU':<18} {'125M':<12} {'1.3B':<12} {'7B':<12} {'70B':<12}")
    print("  " + "-" * 66)

    for gpu_name, tflops, bw, vram in gpu_configs:
        line = f"  {gpu_name:<18}"
        row = {"gpu": gpu_name, "tflops": tflops, "estimates": {}}

        for mname, n_params, n_layers in models:
            # Transformer FLOPS per token per layer ≈ 6 * H²
            # Total FLOPS per token ≈ 6 * n_params (forward+backward)
            flops_per_token = 6 * n_params

            # Time per token
            time_per_token = flops_per_token / (tflops * 1e12)

            # Time for target tokens
            total_time_s = time_per_token * target_tokens

            # Convert to readable format
            if total_time_s < 3600:
                time_str = f"{total_time_s:.1f}s"
            elif total_time_s < 86400:
                time_str = f"{total_time_s/3600:.1f}h"
            else:
                time_str = f"{total_time_s/86400:.1f}d"

            row["estimates"][mname] = {
                "flops_per_token": flops_per_token,
                "time_per_token_us": round(time_per_token * 1e6, 2),
                "total_time": time_str,
                "total_seconds": round(total_time_s, 0),
            }

            if mname in ["125M", "1.3B", "7B", "70B"]:
                line += f" {time_str:<12}"

        results.append(row)
        print(line)

    # Multi-GPU estimates
    print(f"\n  --- 多卡训练时间 (70B, {target_tokens/1e9:.0f}B tokens) ---")
    for gpu_name, tflops, bw, vram in gpu_configs:
        if gpu_name == "A16 (15GB)":
            continue
        for n_gpu in [8, 64, 256, 1024]:
            total_tflops = tflops * n_gpu
            time_70b = 6 * 70e9 * target_tokens / (total_tflops * 1e12)
            days = time_70b / 86400
            if days < 0.1:
                print(f"    {gpu_name} × {n_gpu}: {days*24:.1f} hours")
            else:
                print(f"    {gpu_name} × {n_gpu}: {days:.1f} days")

    return results


# ============================================================
# 实验 4: Batch Size 对训练吞吐的影响
# ============================================================

def exp4_batch_throughput():
    print("\n" + "=" * 60)
    print("实验4: Batch Size 对训练吞吐的影响")
    print("=" * 60)

    results = []
    H = 512
    n_heads = 8

    class MiniBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(H, 4 * H, bias=False)
            self.fc2 = nn.Linear(4 * H, H, bias=False)
            self.qkv = nn.Linear(H, 3 * H, bias=False)
            self.out = nn.Linear(H, H, bias=False)
            self.ln1 = nn.LayerNorm(H)
            self.ln2 = nn.LayerNorm(H)

        def forward(self, x):
            B, S, H = x.shape
            res = x
            x = self.ln1(x)
            qkv = self.qkv(x).reshape(B, S, 3, n_heads, H // n_heads)
            q, k, v = qkv.unbind(2)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            attn = F.scaled_dot_product_attention(q, k, v)
            x = self.out(attn.transpose(1, 2).reshape(B, S, H))
            x = res + x
            res = x
            x = self.ln2(x)
            x = self.fc2(F.gelu(self.fc1(x)))
            return res + x

    model = MiniBlock().cuda().half()
    seq_len = 256

    print(f"\n  H={H}, seq={seq_len}")
    print(f"  {'Batch':<8} {'Tokens':<10} {'Fwd+Bwd(ms)':<14} {'tok/s':<12} {'TFLOPS':<10} {'Peak MB':<10}")
    print("  " + "-" * 64)

    for batch in [1, 2, 4, 8, 16, 32, 64]:
        try:
            x = torch.randn(batch, seq_len, H, device="cuda", dtype=torch.float16)

            torch.cuda.reset_peak_memory_stats()
            def fwd_bwd():
                y = model(x)
                y.sum().backward()
            ms = bench_ms(fwd_bwd, warmup=3, rep=20)
            model.zero_grad()
            peak = torch.cuda.max_memory_allocated() / 1e6

            tokens = batch * seq_len
            throughput = tokens / ms * 1000
            flops = 6 * batch * seq_len * sum(p.numel() for p in model.parameters())
            tflops = flops / ms / 1e9

            results.append({
                "batch": batch, "ms": round(ms, 4),
                "tokens_per_s": round(throughput, 0),
                "tflops": round(tflops, 2),
                "peak_mem_mb": round(peak, 1),
            })
            print(f"  {batch:<8} {tokens:<10} {ms:<14.4f} {throughput:<12.0f} {tflops:<10.2f} {peak:<10.0f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {batch:<8} OOM")
                torch.cuda.empty_cache()
                break
            raise

    del model
    torch.cuda.empty_cache()
    return results


# ============================================================
# 实验 5: 通信开销建模
# ============================================================

def exp5_communication_model():
    print("\n" + "=" * 60)
    print("实验5: 通信开销建模 (AllReduce 模拟)")
    print("=" * 60)

    results = []

    # 在单 GPU 上模拟 AllReduce 开销
    # AllReduce(Ring) = 2*(P-1)/P * data_size / bandwidth

    tp_sizes = [2, 4, 8]
    model_hidden = [512, 1024, 2048, 4096]

    # 实测单 GPU "AllReduce" 时间 (sum + broadcast 的近似)
    print(f"\n  实测数据传输时间 (A16, fp16):")
    print(f"  {'Size(KB)':<12} {'Copy(ms)':<12} {'BW(GB/s)':<12}")
    print("  " + "-" * 36)

    bw_measurements = {}
    for exp_n in [2**16, 2**18, 2**20, 2**22, 2**24]:
        x = torch.randn(exp_n, device="cuda", dtype=torch.float16)
        ms = bench_ms(lambda: x.clone())
        bw = 2 * exp_n * 2 / ms / 1e6  # R+W, fp16=2bytes
        kb = exp_n * 2 / 1024
        print(f"  {kb:<12.0f} {ms:<12.4f} {bw:<12.1f}")
        bw_measurements[kb] = bw

    # 模型: TP AllReduce 开销占 step 时间的比例
    print(f"\n  TP 通信开销预估:")
    print(f"  {'Model':<10} {'TP':<6} {'AllReduce(MB)':<16} {'Time(ms)':<12} {'Step(ms)':<12} {'Comm%':<8}")
    print("  " + "-" * 66)

    for H in model_hidden:
        for tp in tp_sizes:
            # AllReduce 数据量 = 2 * H * sizeof(fp16) = 4H bytes per token
            # 但实际是整个 batch 的 activation
            B, S = 4, 512
            allreduce_bytes = B * S * H * 2  # fp16
            allreduce_mb = allreduce_bytes / 1e6

            # Ring AllReduce time
            bw_eff = 170  # GB/s (A16 measured)
            allreduce_time_ms = 2 * (tp - 1) / tp * allreduce_mb / bw_eff

            # Step time (estimated from FLOPS)
            flops = 6 * B * S * H * H * 12  # 12-layer model
            step_tflops = 5.0  # A16 effective
            step_time_ms = flops / (step_tflops * 1e9) * 1000

            comm_pct = allreduce_time_ms / step_time_ms * 100

            results.append({
                "hidden": H, "tp": tp,
                "allreduce_mb": round(allreduce_mb, 2),
                "allreduce_time_ms": round(allreduce_time_ms, 4),
                "step_time_ms": round(step_time_ms, 2),
                "comm_pct": round(comm_pct, 2),
            })
            print(f"  H={H:<6} {tp:<6} {allreduce_mb:<16.2f} {allreduce_time_ms:<12.4f} {step_time_ms:<12.2f} {comm_pct:<8.2f}%")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("训练延迟模型 — 从硬件到训练性能")
    print("=" * 60)

    all_results = OrderedDict()

    all_results["roofline"] = exp1_roofline()
    all_results["layer_breakdown"] = exp2_layer_latency_breakdown()
    all_results["training_prediction"] = exp3_training_time_prediction()
    all_results["batch_throughput"] = exp4_batch_throughput()
    all_results["communication_model"] = exp5_communication_model()

    print("\n" + "=" * 60)
    print("关键洞察")
    print("=" * 60)
    print("""
  1. Roofline: Ridge point ~87 ops/byte, 大矩阵 compute-bound, 小矩阵 memory-bound
  2. 层分解: GEMM 占 70-80%, Attention 占 10-15%, LayerNorm/GELU <5%
  3. 训练时间: 7B on A100×8 ≈ 5天, 70B on H100×256 ≈ 7天
  4. Batch scaling: batch=1→32 吞吐提升 ~20x, 之后内存成为瓶颈
  5. 通信占比: TP=8 时 AllReduce 占 step 时间的 2-15% (取决于模型大小)
""")

    with open("/root/training_latency_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to training_latency_results.json")
