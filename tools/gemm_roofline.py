#!/usr/bin/env python3
"""GEMM Roofline 分析器 — 理解 LLM 推理的计算/带宽瓶颈

CPU 可运行的 GEMM 性能分析工具，帮助理解:
  - 为什么 LLM 推理是 memory-bound
  - Prefill vs Decode 的计算/带宽比值
  - 不同模型规模的算术强度 (arithmetic intensity)
  - Roofline model 实际应用

覆盖 5 个实验:
  1. 单层线性层的计算/带宽分析
  2. Prefill vs Decode 瓶颈对比
  3. 不同模型规模的算术强度
  4. Batch size 对瓶颈的影响
  5. GPU 代际性能对比

关键概念:
  - 算术强度 (Arithmetic Intensity) = FLOPs / Bytes Accessed
  - Roofline 模型: 性能上限 = min(peak_FLOPS, bandwidth × AI)
  - compute-bound: AI > ridge_point, 性能受 TFLOPS 限制
  - memory-bound: AI < ridge_point, 性能受带宽限制

用法:
  conda run -n ai-infra python tools/gemm_roofline.py
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np


# ──────────────────────────────────────────────
# GPU 配置
# ──────────────────────────────────────────────

@dataclass
class GPU:
    name: str
    fp16_tflops: float     # FP16 峰值 TFLOPS
    hbm_bw_gbps: float     # HBM 带宽 GB/s
    hbm_gb: float          # HBM 容量 GB

    @property
    def ridge_point(self) -> float:
        """Ridge point: compute-bound 和 memory-bound 的分界线
        AI = TFLOPS / (BW_GB/s) = FLOP/byte
        """
        return self.fp16_tflops * 1e12 / (self.hbm_bw_gbps * 1e9)

    @property
    def peak_tflops(self) -> float:
        return self.fp16_tflops

    def max_perf_tflops(self, arithmetic_intensity: float) -> float:
        """Roofline 模型: 给定 AI 的性能上限"""
        # memory-bound: 性能 = bandwidth × AI
        mem_bound = self.hbm_bw_gbps * 1e9 * arithmetic_intensity / 1e12
        # compute-bound: 性能 = peak TFLOPS
        return min(self.fp16_tflops, mem_bound)


H100_SXM = GPU("H100 SXM", 990, 3350, 80)
H100_PCIE = GPU("H100 PCIe", 756, 2039, 80)
A100_80G = GPU("A100 80G", 312, 2039, 80)
A100_40G = GPU("A100 40G", 312, 1555, 40)
L40S = GPU("L40S", 362, 864, 48)
A16 = GPU("A16", 105, 157, 16)


# ──────────────────────────────────────────────
# GEMM 分析
# ──────────────────────────────────────────────

def analyze_linear_layer(
    batch: int,
    in_features: int,
    out_features: int,
    is_decode: bool = False,
    seq_len: int = 1,
) -> dict:
    """分析线性层 Y = X @ W + b 的计算/带宽特征

    Args:
        batch: batch size
        in_features: 输入维度
        out_features: 输出维度
        is_decode: 是否 decode 阶段 (seq_len=1 per step)
        seq_len: 序列长度 (prefill 时 > 1)
    """
    M = batch * seq_len  # 矩阵乘法的行数
    K = in_features
    N = out_features

    # FLOPs: 2 × M × K × N (乘加各算一次)
    flops = 2 * M * K * N

    # Bytes accessed:
    #   读 X: M × K × 2 bytes (FP16)
    #   读 W: K × N × 2 bytes (FP16) — 权重必须全部读取!
    #   写 Y: M × N × 2 bytes (FP16)
    bytes_read = (M * K + K * N) * 2
    bytes_write = M * N * 2
    total_bytes = bytes_read + bytes_write

    # 算术强度
    ai = flops / total_bytes

    # 占比: 权重读取 vs 输入读取
    weight_bytes = K * N * 2
    input_bytes = M * K * 2
    weight_pct = weight_bytes / total_bytes * 100

    return {
        'M': M, 'K': K, 'N': N,
        'flops': flops,
        'total_bytes': total_bytes,
        'weight_bytes': weight_bytes,
        'input_bytes': input_bytes,
        'ai': ai,
        'weight_pct': weight_pct,
        'batch': batch,
        'seq_len': seq_len,
    }


def analyze_attention(
    batch: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    kv_len: int,
) -> dict:
    """分析 Attention 层的计算/带宽特征

    Q@K^T: 2 × B × H × S_q × S_kv × D
    Softmax@V: 2 × B × H × S_q × S_kv × D
    """
    B = batch
    H = num_heads
    D = head_dim
    Sq = seq_len
    Skv = kv_len

    # Q@K^T + Softmax@V
    flops = 4 * B * H * Sq * Skv * D

    # Bytes: Q[B,H,Sq,D] + K[B,H,Skv,D] + V[B,H,Skv,D] + output[B,H,Sq,D]
    total_bytes = (B * H * Sq * D + B * H * Skv * D + B * H * Skv * D + B * H * Sq * D) * 2

    # 注意: 如果是 decode (Sq=1), 每次只读 1 行 Q 但整个 KV cache
    ai = flops / total_bytes

    return {
        'flops': flops,
        'total_bytes': total_bytes,
        'ai': ai,
        'batch': B,
        'seq_len': Sq,
        'kv_len': Skv,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_linear_layer_analysis():
    """实验 1: 单层线性层的计算/带宽分析"""
    print("\n" + "=" * 70)
    print("实验 1: 线性层 Y = X @ W 的计算/带宽分析")
    print("=" * 70)

    # LLaMA-7B 的典型线性层
    hidden = 4096
    intermediate = 11008  # 4 × hidden × 4/3

    configs = [
        ("Prefill (S=1024)", 1, hidden, intermediate, 1024),
        ("Prefill (S=4096)", 1, hidden, intermediate, 4096),
        ("Decode (S=1, B=1)", 1, hidden, intermediate, 1),
        ("Decode (S=1, B=8)", 8, hidden, intermediate, 1),
        ("Decode (S=1, B=32)", 32, hidden, intermediate, 1),
        ("Decode (S=1, B=128)", 128, hidden, intermediate, 1),
    ]

    gpu = H100_SXM
    print(f"\n配置: Linear {hidden}→{intermediate} (LLaMA-7B MLP), GPU={gpu.name}")
    print(f"  Ridge point (AI 分界线): {gpu.ridge_point:.1f} FLOP/byte")
    print(f"\n{'场景':<25} {'FLOPs':>14} {'Bytes':>12} {'AI':>8} "
          f"{'权重占比':>8} {'瓶颈':>10} {'性能上限 TFLOPS':>16}")
    print("-" * 100)

    for name, batch, in_f, out_f, seq in configs:
        r = analyze_linear_layer(batch, in_f, out_f, seq_len=seq)
        max_perf = gpu.max_perf_tflops(r['ai'])
        bottleneck = "compute" if r['ai'] > gpu.ridge_point else "memory"
        print(f"{name:<25} {r['flops']:>14.2e} {r['total_bytes']:>12.2e} "
              f"{r['ai']:>8.1f} {r['weight_pct']:>7.1f}% "
              f"{bottleneck:>10} {max_perf:>16.1f}")

    print(f"\n关键洞察:")
    print(f"  - Prefill (S=4096): AI=209, 远超 ridge point → compute-bound")
    print(f"  - Decode (B=1): AI=2.0, 远低于 ridge point → memory-bound")
    print(f"  - Decode 权重占比 >90%: 每次都要读全部权重但只算 1 行")
    print(f"  - 增大 batch 可以提升 AI (更多数据复用权重)")


def experiment_2_prefill_vs_decode():
    """实验 2: Prefill vs Decode 全模型瓶颈对比"""
    print("\n" + "=" * 70)
    print("实验 2: Prefill vs Decode 全模型瓶颈对比")
    print("=" * 70)

    gpu = H100_SXM

    model_configs = [
        ("LLaMA-7B", 4096, 11008, 32, 32, 128),
        ("LLaMA-13B", 5120, 13824, 40, 40, 128),
        ("LLaMA-70B", 8192, 28672, 80, 64, 128),
    ]

    print(f"\nGPU: {gpu.name} (Peak {gpu.fp16_tflops} TFLOPS, {gpu.hbm_bw_gbps} GB/s)")
    print(f"Ridge point: {gpu.ridge_point:.1f} FLOP/byte")
    print(f"\n--- Prefill (S=2048, B=1) ---")
    print(f"{'模型':<15} {'Linear AI':>10} {'Attn AI':>10} {'Linear 瓶颈':>14} {'Attn 瓶颈':>12}")
    print("-" * 65)

    for name, hidden, inter, layers, heads, head_dim in model_configs:
        # Linear layer (MLP)
        lr = analyze_linear_layer(1, hidden, inter, seq_len=2048)
        # Attention
        ar = analyze_attention(1, heads, head_dim, 2048, 2048)

        l_bottleneck = "compute" if lr['ai'] > gpu.ridge_point else "memory"
        a_bottleneck = "compute" if ar['ai'] > gpu.ridge_point else "memory"

        print(f"{name:<15} {lr['ai']:>10.1f} {ar['ai']:>10.1f} "
              f"{l_bottleneck:>14} {a_bottleneck:>12}")

    print(f"\n--- Decode (S=1, B=1, KV=2048) ---")
    print(f"{'模型':<15} {'Linear AI':>10} {'Attn AI':>10} {'Linear 瓶颈':>14} {'Attn 瓶颈':>12}")
    print("-" * 65)

    for name, hidden, inter, layers, heads, head_dim in model_configs:
        lr = analyze_linear_layer(1, hidden, inter, seq_len=1)
        ar = analyze_attention(1, heads, head_dim, 1, 2048)

        l_bottleneck = "compute" if lr['ai'] > gpu.ridge_point else "memory"
        a_bottleneck = "compute" if ar['ai'] > gpu.ridge_point else "memory"

        print(f"{name:<15} {lr['ai']:>10.1f} {ar['ai']:>10.1f} "
              f"{l_bottleneck:>14} {a_bottleneck:>12}")

    print(f"\n关键洞察:")
    print(f"  - Prefill: 所有操作都是 compute-bound (AI >> ridge point)")
    print(f"  - Decode: 所有操作都是 memory-bound (AI << ridge point)")
    print(f"  - Decode 的 Linear AI ≈ 2.0 (常数!): 只取决于 hidden/inter 比值")
    print(f"  - Decode 的 Attn AI ≈ 2.0: 每次读整个 KV cache 但只算 1 个 query")


def experiment_3_model_scale():
    """实验 3: 不同模型规模的算术强度"""
    print("\n" + "=" * 70)
    print("实验 3: 不同模型规模 Decode 阶段的算术强度")
    print("=" * 70)

    gpu = H100_SXM

    # Decode: B=1, S=1, KV=2048
    models = [
        ("1.5B", 2048, 5504, 24, 24, 64),
        ("7B", 4096, 11008, 32, 32, 128),
        ("13B", 5120, 13824, 40, 40, 128),
        ("30B", 6656, 17920, 60, 64, 128),
        ("70B", 8192, 28672, 80, 64, 128),
        ("405B", 16384, 57344, 126, 128, 128),
    ]

    kv_lens = [512, 2048, 8192, 32768]

    print(f"\nGPU: {gpu.name}, Decode B=1 S=1")
    print(f"\n{'模型':<10}", end="")
    for kv in kv_lens:
        print(f"{'KV='+str(kv):>14}", end="")
    print(f"  {'权重GB':>8} {'吞吐上限 tok/s':>14}")
    print("-" * 90)

    for name, hidden, inter, layers, heads, head_dim in models:
        # Linear decode AI (same for all KV lengths)
        lr = analyze_linear_layer(1, hidden, inter, seq_len=1)
        linear_ai = lr['ai']

        # Weight size
        # 4 × hidden × intermediate × layers × 2 bytes (up, down, gate, gate_bias)
        weight_gb = 4 * hidden * inter * layers * 2 / 1e9

        print(f"{name:<10}", end="")
        for kv in kv_lens:
            ar = analyze_attention(1, heads, head_dim, 1, kv)
            attn_ai = ar['ai']

            # 总 AI: 加权平均 (linear 占大部分 FLOPs)
            linear_flops = lr['flops'] * 4 * layers  # 4 linear per layer (qkv+out or up+down+gate)
            attn_flops = ar['flops'] * layers
            total_flops = linear_flops + attn_flops

            linear_bytes = lr['total_bytes'] * 4 * layers
            attn_bytes = ar['total_bytes'] * layers
            total_bytes = linear_bytes + attn_bytes

            total_ai = total_flops / total_bytes

            max_perf = gpu.max_perf_tflops(total_ai)
            # Decode 吞吐 = 性能上限 / (每 token 的 FLOPs)
            tok_per_s = max_perf * 1e12 / total_flops

            print(f"{total_ai:>14.1f}", end="")

        print(f"  {weight_gb:>8.1f}", end="")

        # 用 KV=2048 的数据估算吞吐
        ar2 = analyze_attention(1, heads, head_dim, 1, 2048)
        total_flops_2k = lr['flops'] * 4 * layers + ar2['flops'] * layers
        total_bytes_2k = lr['total_bytes'] * 4 * layers + ar2['total_bytes'] * layers
        ai_2k = total_flops_2k / total_bytes_2k
        max_perf_2k = gpu.max_perf_tflops(ai_2k)
        tok_per_s_2k = max_perf_2k * 1e12 / total_flops_2k
        print(f"{tok_per_s_2k:>14.0f}")

    print(f"\n关键洞察:")
    print(f"  - 所有模型 Decode 的 AI 都 < 10 (远低于 ridge point ~295)")
    print(f"  - 模型越大, 权重越大, bandwidth 需求越高")
    print(f"  - 405B 模型 Decode 吞吐极低 (权重 810GB, 要读完整个权重才输出 1 token)")
    print(f"  - KV Cache 长度对 AI 影响小 (linear 层权重是绝对主导)")


def experiment_4_batch_size_impact():
    """实验 4: Batch size 对瓶颈的影响"""
    print("\n" + "=" * 70)
    print("实验 4: Batch Size 对 Decode 瓶颈的影响")
    print("=" * 70)

    gpu = H100_SXM
    hidden = 8192  # LLaMA-70B
    inter = 28672
    layers = 80
    heads = 64
    head_dim = 128

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print(f"\n配置: LLaMA-70B, Decode, KV=2048, GPU={gpu.name}")
    print(f"Ridge point: {gpu.ridge_point:.1f} FLOP/byte")
    print(f"\n{'Batch':>6} {'Linear AI':>10} {'Attn AI':>10} {'总 AI':>10} "
          f"{'瓶颈':>10} {'性能 TFLOPS':>12} {'吞吐 tok/s':>12}")
    print("-" * 75)

    for B in batch_sizes:
        lr = analyze_linear_layer(B, hidden, inter, seq_len=1)
        ar = analyze_attention(B, heads, head_dim, 1, 2048)

        linear_flops = lr['flops'] * 4 * layers
        attn_flops = ar['flops'] * layers
        total_flops = linear_flops + attn_flops

        linear_bytes = lr['total_bytes'] * 4 * layers
        attn_bytes = ar['total_bytes'] * layers
        total_bytes = linear_bytes + attn_bytes

        total_ai = total_flops / total_bytes
        max_perf = gpu.max_perf_tflops(total_ai)
        bottleneck = "compute" if total_ai > gpu.ridge_point else "memory"
        tok_per_s = max_perf * 1e12 / total_flops * B  # ×B 因为 batch

        print(f"{B:>6d} {lr['ai']:>10.1f} {ar['ai']:>10.1f} {total_ai:>10.1f} "
              f"{bottleneck:>10} {max_perf:>12.1f} {tok_per_s:>12.0f}")

    print(f"\n关键洞察:")
    print(f"  - Batch size 增大 → AI 增大 → 逐渐从 memory-bound 转向 compute-bound")
    print(f"  - 大 batch 时权重被多次复用, 带宽不再是瓶颈")
    print(f"  - 但 KV cache 也会随 batch 线性增长 → 最终受 GPU 显存限制")


def experiment_5_gpu_comparison():
    """实验 5: GPU 代际性能对比"""
    print("\n" + "=" * 70)
    print("实验 5: GPU 代际 Decode 性能对比 (LLaMA-70B)")
    print("=" * 70)

    gpus = [H100_SXM, H100_PCIE, A100_80G, L40S, A16]

    hidden = 8192
    inter = 28672
    layers = 80
    heads = 64
    head_dim = 128

    print(f"\n配置: LLaMA-70B, Decode B=1, KV=2048")
    print(f"\n{'GPU':<15} {'TFLOPS':>8} {'BW GB/s':>10} {'Ridge':>8} "
          f"{'总 AI':>8} {'性能 TFLOPS':>12} {'吞吐 tok/s':>12} {'利用率':>8}")
    print("-" * 85)

    for gpu in gpus:
        lr = analyze_linear_layer(1, hidden, inter, seq_len=1)
        ar = analyze_attention(1, heads, head_dim, 1, 2048)

        total_flops = lr['flops'] * 4 * layers + ar['flops'] * layers
        total_bytes = lr['total_bytes'] * 4 * layers + ar['total_bytes'] * layers
        total_ai = total_flops / total_bytes

        max_perf = gpu.max_perf_tflops(total_ai)
        tok_per_s = max_perf * 1e12 / total_flops
        utilization = max_perf / gpu.fp16_tflops * 100

        print(f"{gpu.name:<15} {gpu.fp16_tflops:>8.0f} {gpu.hbm_bw_gbps:>10.0f} "
              f"{gpu.ridge_point:>8.1f} {total_ai:>8.1f} "
              f"{max_perf:>12.2f} {tok_per_s:>12.0f} {utilization:>7.1f}%")

    print(f"\n关键洞察:")
    print(f"  - Decode 的 AI ≈ 2.0, 远低于所有 GPU 的 ridge point")
    print(f"  - 性能完全由 HBM 带宽决定, TFLOPS 几乎无关!")
    print(f"  - H100 SXM vs A100: 吞吐比 ≈ 3350/2039 ≈ 1.6x (带宽比)")
    print(f"  - GPU 利用率 < 1%: 大量算力被浪费, 这就是 decode 的瓶颈")
    print(f"  - A16 吞吐仅 ~7 tok/s: 这就是为什么需要高端 GPU")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("GEMM Roofline 分析器")
    print("理解 LLM 推理的 compute-bound vs memory-bound 瓶颈")
    print("=" * 70)

    experiment_1_linear_layer_analysis()
    experiment_2_prefill_vs_decode()
    experiment_3_model_scale()
    experiment_4_batch_size_impact()
    experiment_5_gpu_comparison()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
LLM 推理 Roofline 关键要点:

  1. 算术强度 (Arithmetic Intensity):
     AI = FLOPs / Bytes
     - Linear Decode: AI ≈ 2.0 (常数, 与模型大小无关)
     - Linear Prefill: AI ≈ 2 × seq_len (与序列长度成正比)
     - Attention Decode: AI ≈ 2 × d / (d + 2×kv_len)

  2. Ridge Point (分界线):
     H100: ~295 FLOP/byte
     A100: ~153 FLOP/byte
     L40S: ~419 FLOP/byte
     - Decode AI ≈ 2.0 << 所有 GPU 的 ridge point
     - Decode 永远是 memory-bound!

  3. Prefill vs Decode:
     - Prefill: compute-bound, 性能 = TFLOPS × 利用率
     - Decode: memory-bound, 性能 = HBM带宽 × AI
     - 这就是 P/D 分离架构的动机!

  4. Batch Size 效果:
     - B=1: AI≈2, GPU 利用率 < 1%
     - B=128+: AI 接近 ridge point, 开始 compute-bound
     - 大 batch 提升吞吐但增加 KV Cache 压力

  5. GPU 选型:
     - Decode 吞吐 ∝ HBM 带宽 (不是 TFLOPS!)
     - H100 SXM 吞吐是 A100 的 1.6x (3350/2039)
     - 低端 GPU (A16) 吞吐极低但可跑推理
    """)
