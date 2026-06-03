#!/usr/bin/env python3
"""序列并行模拟器 — Ring Attention / Ulysses / USP

CPU 可运行的序列并行模拟实验，覆盖 4 个实验:
  1. Ulysses vs Ring Attention vs USP 通信模式对比
  2. 序列长度对通信量影响
  3. GPU 数量扩展效率
  4. 不同 SP 策略在长序列训练中的效果

关键概念:
  - Ulysses: 沿 head 切分 + All-to-All 重排, 通信量 O(N×d/P)
  - Ring Attention: 沿 seq 切分 + P2P 环形传递, 通信量 O(N×d/P)
  - USP: 统一框架, 2D 切分 (seq×head), 支持两者组合

用法:
  conda run -n ai-infra python tools/sequence_parallel_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 通信模型
# ──────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """网络配置"""
    name: str
    bandwidth_gbps: float     # 单向带宽 (GB/s)
    latency_us: float = 5.0   # 延迟 (μs)


NVLink = NetworkConfig("NVLink 4.0", 900, latency_us=1.5)
NVLink3 = NetworkConfig("NVLink 3.0", 600, latency_us=2.0)
PCIeGen5 = NetworkConfig("PCIe Gen5", 64, latency_us=5.0)
RoCE400 = NetworkConfig("RoCE 400G", 50, latency_us=10.0)
RoCE200 = NetworkConfig("RoCE 200G", 25, latency_us=15.0)


def all_to_all_time(
    data_per_gpu_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> float:
    """All-to-All 通信时间 (ms)

    All-to-All: 每个 GPU 向其他所有 GPU 发送数据
    总时间 ≈ (P-1) × data_per_pair / bandwidth + P × latency
    """
    if num_gpus <= 1:
        return 0.0
    # 每个 GPU 向 P-1 个其他 GPU 各发送 data / P
    data_per_pair = data_per_gpu_bytes / num_gpus
    total_time_us = (num_gpus - 1) * data_per_pair / (network.bandwidth_gbps * 1e9 / 1e6) \
                    + num_gpus * network.latency_us
    return total_time_us / 1000  # ms


def p2p_time(
    data_bytes: int,
    network: NetworkConfig,
) -> float:
    """点对点通信时间 (ms)"""
    time_us = data_bytes / (network.bandwidth_gbps * 1e9 / 1e6) + network.latency_us
    return time_us / 1000


# ──────────────────────────────────────────────
# 序列并行策略
# ──────────────────────────────────────────────

@dataclass
class ModelShape:
    """模型形状参数"""
    name: str
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_layers: int
    seq_len: int

    @property
    def attention_flops_per_layer(self) -> int:
        """单层 attention FLOPs (forward)"""
        N = self.seq_len
        d = self.head_dim
        h = self.num_heads
        # Q@K: 2*h*N*N*d, Softmax@V: 2*h*N*N*d, Q,K,V proj: 6*N*d*h
        return 4 * h * N * N * d + 6 * N * d * h

    @property
    def attention_memory_per_layer(self) -> int:
        """单层 attention 中间显存 (bytes, FP16)"""
        N = self.seq_len
        h = self.num_heads
        # Attention matrix: [batch=1, h, N, N] × 2 bytes
        return h * N * N * 2


def simulate_ulysses(
    model: ModelShape,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """DeepSpeed Ulysses 模拟

    切分策略:
      1. 输入 X [N, H] 沿 seq 维切 P 份 → 每个 GPU 处理 N/P tokens
      2. All-to-All: 重排为沿 head 维切分 → 每个 GPU 处理 H/P heads × N tokens
      3. 每个 GPU 独立做 attention (完整 seq, 部分 heads)
      4. All-to-All: 重排回沿 seq 切分
    """
    P = num_gpus
    N = model.seq_len
    H = model.hidden_dim

    # 每个 GPU 处理的序列长度
    local_seq = N // P

    # 通信: 两次 All-to-All
    # 数据量: 每个 GPU 有 local_seq × H 的数据要重新分配
    data_per_gpu = local_seq * H * 2  # FP16 bytes
    comm_time = 2 * all_to_all_time(data_per_gpu, P, network)

    # 显存: 每个 GPU 存储 N × H/P 的 attention (完整 seq, 部分 heads)
    # Attention matrix: (H/P) × N × N × 2 bytes
    attn_mem_per_gpu = (model.num_heads // P) * N * N * 2

    # 计算量: 每个 GPU 做完整 attention (H/P heads, N seq)
    local_heads = model.num_heads // P
    flops_per_gpu = 4 * local_heads * N * N * model.head_dim * model.num_layers

    # 每个 GPU 总显存 (简化: 只考虑 attention 中间 + 参数分片)
    param_mem = model.hidden_dim * model.hidden_dim * 2 * model.num_layers / P  # TP-like

    return {
        'strategy': 'Ulysses',
        'local_seq_len': local_seq,
        'local_heads': model.num_heads // P,
        'comm_time_ms': comm_time,
        'comm_type': f'2× All-to-All ({data_per_gpu * P / 1e6:.1f} MB each)',
        'attn_mem_per_gpu_MB': attn_mem_per_gpu / 1e6,
        'flops_per_gpu_TFLOPS': flops_per_gpu / 1e12,
        'seq_per_gpu': N,  # Ulysses 每个 GPU 处理完整 seq!
    }


def simulate_ring_attention(
    model: ModelShape,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """Ring Attention 模拟

    切分策略:
      1. 输入 X [N, H] 沿 seq 维切 P 份 → 每个 GPU 处理 N/P tokens
      2. P2P 环形传递 K, V blocks → 每步处理一个 KV block
      3. 每步做 local attention (N/P query × N/P kv)
      4. P-1 步后完成完整 attention

    关键: 使用 FlashAttention 的 online softmax 累积
    """
    P = num_gpus
    N = model.seq_len
    H = model.hidden_dim

    # 每个 GPU 处理的序列长度
    local_seq = N // P

    # 通信: P-1 步 P2P, 每步传 local_seq × H 的 KV block
    kv_block_bytes = local_seq * H * 2  # FP16
    # 每步: 传 K block + V block
    comm_per_step = 2 * p2p_time(kv_block_bytes, network)
    # 总共 P-1 步
    comm_time = (P - 1) * comm_per_step

    # 显存: 每个 GPU 只存 local_seq tokens 的 KV cache
    # Attention matrix: H × local_seq × local_seq × 2 bytes (per step)
    attn_mem_per_gpu = model.num_heads * local_seq * local_seq * 2

    # 计算量: P 个 step, 每步做 H heads × local_seq × local_seq attention
    # 总: P × H × (N/P)² × d = H × N²/P × d
    flops_per_gpu = 4 * model.num_heads * local_seq * N * model.head_dim * model.num_layers

    return {
        'strategy': 'Ring Attention',
        'local_seq_len': local_seq,
        'local_heads': model.num_heads,
        'comm_time_ms': comm_time,
        'comm_type': f'{P-1}× P2P ring ({kv_block_bytes / 1e6:.1f} MB/step)',
        'attn_mem_per_gpu_MB': attn_mem_per_gpu / 1e6,
        'flops_per_gpu_TFLOPS': flops_per_gpu / 1e12,
        'seq_per_gpu': local_seq,  # Ring 每个 GPU 只处理 N/P!
    }


def simulate_usp(
    model: ModelShape,
    num_gpus: int,
    ulysses_degree: int,
    network: NetworkConfig,
) -> dict:
    """USP (Unified Sequence Parallelism) 模拟

    2D 切分: 同时沿 seq 和 head 维切分
      - Ulysses degree (U): head 维切分
      - Ring degree (R): seq 维切分
      - P = U × R
    """
    U = ulysses_degree
    R = num_gpus // U
    N = model.seq_len
    H = model.hidden_dim

    assert num_gpus % U == 0, f"P={num_gpus} must divide by U={U}"

    local_seq_ring = N // R  # Ring 切分后的 seq 长度
    local_heads = model.num_heads // U  # Ulysses 切分后的 head 数

    # Ulysses 通信: All-to-All (在 U 个 GPU 之间)
    data_ulysses = local_seq_ring * H * 2
    comm_ulysses = 2 * all_to_all_time(data_ulysses, U, network)

    # Ring 通信: P2P (在 R 个 GPU 之间)
    kv_block_bytes = local_seq_ring * (H // U) * 2  # 只传部分 head 的 KV
    comm_ring = (R - 1) * 2 * p2p_time(kv_block_bytes, network)

    total_comm = comm_ulysses + comm_ring

    # 显存
    attn_mem = local_heads * local_seq_ring * local_seq_ring * 2

    # 计算量
    flops = 4 * local_heads * local_seq_ring * N * model.head_dim * model.num_layers

    return {
        'strategy': f'USP (U={U}, R={R})',
        'local_seq_len': local_seq_ring,
        'local_heads': local_heads,
        'comm_time_ms': total_comm,
        'comm_type': f'All-to-All×2(U={U}) + {R-1}×P2P(R={R})',
        'attn_mem_per_gpu_MB': attn_mem / 1e6,
        'flops_per_gpu_TFLOPS': flops / 1e12,
        'seq_per_gpu': local_seq_ring,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_strategy_comparison():
    """实验 1: Ulysses vs Ring Attention vs USP"""
    print("\n" + "=" * 70)
    print("实验 1: Ulysses vs Ring Attention vs USP 对比")
    print("=" * 70)

    model = ModelShape("LLaMA-7B-32K", 4096, 32, 128, 32, seq_len=32768)
    network = NVLink
    P = 8

    print(f"\n配置: {model.name}, N={model.seq_len}, P={P} GPUs, {network.name}")

    results = []
    results.append(simulate_ulysses(model, P, network))
    results.append(simulate_ring_attention(model, P, network))
    results.append(simulate_usp(model, P, 2, network))   # U=2, R=4
    results.append(simulate_usp(model, P, 4, network))   # U=4, R=2

    print(f"\n{'策略':<25} {'Local Seq':>10} {'Local Heads':>12} "
          f"{'Comm (ms)':>10} {'Attn MB':>10} {'通信类型'}")
    print("-" * 100)

    for r in results:
        print(f"{r['strategy']:<25} {r['local_seq_len']:>10d} {r['local_heads']:>12d} "
              f"{r['comm_time_ms']:>10.2f} {r['attn_mem_per_gpu_MB']:>10.1f} "
              f"{r['comm_type']}")

    print(f"\n关键洞察:")
    print(f"  - Ulysses: 每个 GPU 处理完整 seq ({model.seq_len}), 但只有 {model.num_heads//P} heads")
    print(f"    → Attention 矩阵仍然很大 ({results[0]['attn_mem_per_gpu_MB']:.1f} MB)")
    print(f"  - Ring: 每个 GPU 只处理 {model.seq_len//P} seq, 但所有 heads")
    print(f"    → Attention 矩阵很小 ({results[1]['attn_mem_per_gpu_MB']:.1f} MB)")
    print(f"  - USP: 2D 切分, 平衡了两者")


def experiment_2_seq_length_impact():
    """实验 2: 序列长度对通信量影响"""
    print("\n" + "=" * 70)
    print("实验 2: 序列长度对通信量影响")
    print("=" * 70)

    network = NVLink
    P = 8

    seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072]

    print(f"\n配置: P={P} GPUs, {network.name}")
    print(f"\n{'Seq Len':>10} {'Ulysses Comm':>14} {'Ring Comm':>14} "
          f"{'Ulysses Attn':>14} {'Ring Attn':>14}")
    print("-" * 70)

    for N in seq_lengths:
        model = ModelShape("Model", 4096, 32, 128, 32, seq_len=N)
        u = simulate_ulysses(model, P, network)
        r = simulate_ring_attention(model, P, network)
        print(f"{N:>10d} {u['comm_time_ms']:>14.2f} {r['comm_time_ms']:>14.2f} "
              f"{u['attn_mem_per_gpu_MB']:>14.1f} {r['attn_mem_per_gpu_MB']:>14.1f}")

    print(f"\n关键洞察:")
    print(f"  - Ulysses 通信量: O(N×H/P), 与 seq_len 线性增长")
    print(f"  - Ring 通信量: O(N×H/P × (P-1)), 也线性增长但系数更大")
    print(f"  - Ulysses attention 显存: O(H/P × N²), 仍然 O(N²)!")
    print(f"  - Ring attention 显存: O(H × (N/P)²), 缩小 P² 倍!")
    print(f"  - 超长序列 (>32K), Ring Attention 的显存优势决定性")


def experiment_3_gpu_scaling():
    """实验 3: GPU 数量扩展效率"""
    print("\n" + "=" * 70)
    print("实验 3: GPU 数量扩展效率")
    print("=" * 70)

    model = ModelShape("Model-32K", 4096, 32, 128, 32, seq_len=32768)
    network = NVLink

    gpu_counts = [1, 2, 4, 8, 16, 32]

    print(f"\n配置: N={model.seq_len}, {network.name}")
    print(f"\n--- Ulysses ---")
    print(f"{'GPUs':>6} {'Comm (ms)':>10} {'Attn MB':>10} {'Scale Eff':>10}")
    print("-" * 40)

    base_attn = 0
    for P in gpu_counts:
        u = simulate_ulysses(model, P, network)
        if P == 1:
            base_attn = model.num_heads * model.seq_len * model.seq_len * 2 / 1e6
        scale_eff = base_attn / max(1, u['attn_mem_per_gpu_MB'] * P)
        print(f"{P:>6d} {u['comm_time_ms']:>10.2f} {u['attn_mem_per_gpu_MB']:>10.1f} "
              f"{scale_eff:>9.2f}")

    print(f"\n--- Ring Attention ---")
    print(f"{'GPUs':>6} {'Comm (ms)':>10} {'Attn MB':>10} {'Scale Eff':>10}")
    print("-" * 40)

    for P in gpu_counts:
        r = simulate_ring_attention(model, P, network)
        scale_eff = base_attn / max(1, r['attn_mem_per_gpu_MB'] * P)
        print(f"{P:>6d} {r['comm_time_ms']:>10.2f} {r['attn_mem_per_gpu_MB']:>10.1f} "
              f"{scale_eff:>9.2f}")

    print(f"\n关键洞察:")
    print(f"  - Ulysses 显存: O(H/P × N²) → 线性减少但不解决 O(N²)")
    print(f"  - Ring 显存: O(H × (N/P)²) → P² 倍减少! 真正解决长序列")
    print(f"  - Ring 通信量更大, 但用通信换显存是值得的")
    print(f"  - NVLink 下通信开销 < 1ms, 不是瓶颈")


def experiment_4_network_impact():
    """实验 4: 网络带宽对不同策略的影响"""
    print("\n" + "=" * 70)
    print("实验 4: 网络带宽对不同策略的影响")
    print("=" * 70)

    model = ModelShape("Model-32K", 4096, 32, 128, 32, seq_len=32768)
    P = 8

    networks = [NVLink, NVLink3, PCIeGen5, RoCE400, RoCE200]

    print(f"\n配置: N={model.seq_len}, P={P}")
    print(f"\n{'网络':<20} {'Ulysses Comm':>14} {'Ring Comm':>14} "
          f"{'Ulysses/Total%':>14} {'Ring/Total%':>14}")
    print("-" * 80)

    # 估算 attention 计算时间 (假设 H100 ~500 TFLOPS, 40% efficiency)
    # 注意: 这里只算 attention 的计算时间, 不包含 MLP 等
    # 7B 模型单层 attention 的实际时间: attention 部分约占 30-40% 总时间
    # 对于 N=32768, 单层 attention 约 2-5ms, 32 层约 64-160ms
    compute_time_ms = 100.0  # 简化估算: 100ms for attention-only

    for net in networks:
        u = simulate_ulysses(model, P, net)
        r = simulate_ring_attention(model, P, net)
        u_pct = u['comm_time_ms'] / (u['comm_time_ms'] + compute_time_ms) * 100
        r_pct = r['comm_time_ms'] / (r['comm_time_ms'] + compute_time_ms) * 100
        print(f"{net.name:<20} {u['comm_time_ms']:>14.2f} {r['comm_time_ms']:>14.2f} "
              f"{u_pct:>13.1f}% {r_pct:>13.1f}%")

    print(f"\n  参考: Attention 计算时间 ≈ {compute_time_ms:.1f} ms")
    print(f"\n关键洞察:")
    print(f"  - NVLink: 通信占比 < 1%, 几乎可以忽略")
    print(f"  - PCIe Gen5: Ulysses ~1-2%, Ring ~5-10%, 仍可接受")
    print(f"  - RoCE 200G: Ring 通信占比显著增加 (>10%)")
    print(f"  - Ring Attention 更依赖高速互联网络")
    print(f"  - Ulysses (All-to-All) 比 Ring (P2P×P) 对网络更友好")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("序列并行模拟器")
    print("Ulysses vs Ring Attention vs USP")
    print("=" * 70)

    experiment_1_strategy_comparison()
    experiment_2_seq_length_impact()
    experiment_3_gpu_scaling()
    experiment_4_network_impact()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
序列并行关键要点:
  1. Ulysses (DeepSpeed):
     - 切分: 沿 head 维切分, All-to-All 重排
     - 通信: 2× All-to-All, 量 O(N×H/P)
     - 显存: O(H/P × N²) — 仍然 O(N²)!
     - 适合: 中等长度序列, 高速网络

  2. Ring Attention:
     - 切分: 沿 seq 维切分, P2P 环形传递
     - 通信: (P-1)× P2P, 量 O(N×H/P × (P-1))
     - 显存: O(H × (N/P)²) — 缩小 P² 倍!
     - 适合: 超长序列 (>32K), NVLink 集群

  3. USP (统一框架):
     - 2D 切分: 同时沿 head 和 seq 切分
     - P = U × R (Ulysses × Ring)
     - 最灵活, 可以根据硬件和序列长度调整比例

  4. 选择建议:
     - N < 8K: 不需要 SP, FlashAttention 就够了
     - 8K < N < 32K: Ulysses (简单, 通信少)
     - N > 32K: Ring Attention (显存优势决定性)
     - 多种混合: USP 框架自动选择

  5. 实际应用:
     - DeepSeek-V3: 128K context → Ring Attention 必需
     - verl RL 训练: 长 prompt (>4K) → SP 显著加速
     - Megatron-Core: 支持 Ulysses + TP 组合
    """)
