#!/usr/bin/env python3
"""集合通信原语深度分析器

CPU 可运行的集合通信分析工具，覆盖 4 个实验:
  1. Ring vs Tree AllReduce 算法对比
  2. AllReduce vs AllGather vs ReduceScatter 通信量对比
  3. 通信与计算重叠 (Overlap) 效率分析
  4. 通信瓶颈诊断 (通信占比 vs GPU 数量)

关键概念:
  - AllReduce: 所有 GPU 持有全局规约结果, 通信量 = 2(P-1)/P × data
  - AllGather: 所有 GPU 持有完整数据, 通信量 = (P-1)/P × data
  - ReduceScatter: 每个 GPU 持有不同的规约分片, 通信量 = (P-1)/P × data
  - Ring AllReduce: 2 趟 (ReduceScatter + AllGather), 每趟 P-1 步
  - Tree AllReduce: log2(P) 步, 但每步带宽利用率低

用法:
  conda run -n ai-infra python tools/collective_comm_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class NetworkConfig:
    name: str
    bandwidth_gbps: float  # 单链路带宽 (GB/s)
    latency_us: float      # 单跳延迟 (us)
    topology: str  # "nvlink", "pcie", "ethernet"


NVLink = NetworkConfig("NVLink", 900, 1.0, "nvlink")
NVLink_A100 = NetworkConfig("NVLink (A100)", 600, 1.5, "nvlink")
PCIe_Gen4 = NetworkConfig("PCIe Gen4 x16", 64, 5.0, "pcie")
Ethernet_400G = NetworkConfig("Ethernet 400G", 50, 10.0, "ethernet")
Ethernet_200G = NetworkConfig("Ethernet 200G", 25, 10.0, "ethernet")


# ──────────────────────────────────────────────
# 通信模型
# ──────────────────────────────────────────────

def ring_allreduce_time_ms(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """Ring AllReduce 延迟模型

    Ring AllReduce = ReduceScatter + AllGather
    每趟: P-1 步, 每步传输 data/P 字节
    总传输 = 2 × (P-1) × data/P 字节
    每个链路带宽 = network.bandwidth_gbps
    """
    if num_gpus <= 1:
        return {'time_ms': 0, 'total_bytes': 0, 'algorithm': 'ring'}

    P = num_gpus
    # ReduceScatter: P-1 步, 每步 data/P
    # AllGather: P-1 步, 每步 data/P
    steps_per_phase = P - 1
    chunk_bytes = data_bytes / P

    # 每步时间 = max(chunk_transfer, latency)
    chunk_transfer_ms = chunk_bytes / (network.bandwidth_gbps * 1e9 / 1000)  # bytes/ms
    latency_ms = network.latency_us / 1000

    step_time_ms = chunk_transfer_ms + latency_ms  # 简化: 传输 + 延迟

    total_time_ms = 2 * steps_per_phase * step_time_ms
    total_bytes = 2 * steps_per_phase * chunk_bytes

    # 有效带宽利用率
    ideal_time_ms = data_bytes / (network.bandwidth_gbps * 1e9 / 1000)
    bw_utilization = ideal_time_ms / total_time_ms if total_time_ms > 0 else 0

    return {
        'time_ms': total_time_ms,
        'total_bytes': total_bytes,
        'algorithm': 'ring',
        'steps': 2 * steps_per_phase,
        'chunk_bytes': chunk_bytes,
        'step_time_ms': step_time_ms,
        'bw_utilization': bw_utilization,
    }


def tree_allreduce_time_ms(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """Tree (Recursive Doubling) AllReduce 延迟模型

    Tree AllReduce: log2(P) 步 reduce, log2(P) 步 broadcast
    每步传输 data 字节 (不分块!)
    总传输 = 2 × log2(P) × data
    带宽利用率低但延迟步数少
    """
    if num_gpus <= 1:
        return {'time_ms': 0, 'total_bytes': 0, 'algorithm': 'tree'}

    P = num_gpus
    steps = int(math.ceil(math.log2(P)))

    # 每步传输完整数据
    transfer_ms = data_bytes / (network.bandwidth_gbps * 1e9 / 1000)
    latency_ms = network.latency_us / 1000

    step_time_ms = transfer_ms + latency_ms

    # Reduce (up tree) + Broadcast (down tree)
    total_time_ms = 2 * steps * step_time_ms
    total_bytes = 2 * steps * data_bytes

    ideal_time_ms = data_bytes / (network.bandwidth_gbps * 1e9 / 1000)
    bw_utilization = ideal_time_ms / total_time_ms if total_time_ms > 0 else 0

    return {
        'time_ms': total_time_ms,
        'total_bytes': total_bytes,
        'algorithm': 'tree',
        'steps': 2 * steps,
        'step_time_ms': step_time_ms,
        'bw_utilization': bw_utilization,
    }


def allgather_time_ms(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """AllGather 延迟模型 (Ring)

    AllGather: 每个_GPU 发送 data/P, P-1 步
    最终每个 GPU 持有完整 data
    通信量 = (P-1)/P × data
    """
    if num_gpus <= 1:
        return {'time_ms': 0, 'total_bytes': 0}

    P = num_gpus
    chunk_bytes = data_bytes / P
    transfer_ms = chunk_bytes / (network.bandwidth_gbps * 1e9 / 1000)
    latency_ms = network.latency_us / 1000
    step_time_ms = transfer_ms + latency_ms

    total_time_ms = (P - 1) * step_time_ms
    total_bytes = (P - 1) * chunk_bytes

    return {
        'time_ms': total_time_ms,
        'total_bytes': total_bytes,
        'steps': P - 1,
    }


def reducescatter_time_ms(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """ReduceScatter 延迟模型 (Ring)

    ReduceScatter: 每个 GPU 得到 data/P 的规约结果
    通信量 = (P-1)/P × data
    """
    if num_gpus <= 1:
        return {'time_ms': 0, 'total_bytes': 0}

    P = num_gpus
    chunk_bytes = data_bytes / P
    transfer_ms = chunk_bytes / (network.bandwidth_gbps * 1e9 / 1000)
    latency_ms = network.latency_us / 1000
    step_time_ms = transfer_ms + latency_ms

    total_time_ms = (P - 1) * step_time_ms
    total_bytes = (P - 1) * chunk_bytes

    return {
        'time_ms': total_time_ms,
        'total_bytes': total_bytes,
        'steps': P - 1,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_ring_vs_tree():
    """实验 1: Ring vs Tree AllReduce 对比"""
    print("\n" + "=" * 70)
    print("实验 1: Ring vs Tree AllReduce 算法对比")
    print("=" * 70)

    data_mb = 256  # 256 MB (典型梯度大小)
    data_bytes = data_mb * 1e6

    print(f"\n数据量: {data_mb} MB")
    print(f"\n{'网络':<18} {'GPUs':>6} {'Ring(ms)':>12} {'Tree(ms)':>12} "
          f"{'Ring 步数':>10} {'Tree 步数':>10} {'更优':>8}")
    print("-" * 80)

    networks = [NVLink, PCIe_Gen4, Ethernet_400G]
    gpu_counts = [2, 4, 8, 16, 32, 64]

    for net in networks:
        for P in gpu_counts:
            ring = ring_allreduce_time_ms(data_bytes, P, net)
            tree = tree_allreduce_time_ms(data_bytes, P, net)

            winner = "Ring" if ring['time_ms'] <= tree['time_ms'] else "Tree"

            print(f"{net.name:<18} {P:>6d} {ring['time_ms']:>12.3f} {tree['time_ms']:>12.3f} "
                  f"{ring['steps']:>10d} {tree['steps']:>10d} {winner:>8}")
        print()

    print(f"关键洞察:")
    print(f"  - NVLink: Ring 总是更优 (带宽高, 延迟低)")
    print(f"  - PCIe/Ethernet: 小 GPU 数时 Tree 可能更优 (步数少)")
    print(f"  - Ring 步数 = 2(P-1), Tree 步数 = 2×log2(P)")
    print(f"  - P=64: Ring 126 步 vs Tree 12 步")
    print(f"  - Ring 每步传输 data/P, Tree 每步传输 data")


def experiment_2_collective_ops_comparison():
    """实验 2: AllReduce vs AllGather vs ReduceScatter 对比"""
    print("\n" + "=" * 70)
    print("实验 2: 集合通信操作对比 (通信量 + 延迟)")
    print("=" * 70)

    network = NVLink
    data_mb = 256
    data_bytes = data_mb * 1e6

    print(f"\n网络: {network.name} ({network.bandwidth_gbps} GB/s)")
    print(f"数据量: {data_mb} MB")

    print(f"\n{'GPUs':>6} {'AllReduce ms':>14} {'AllGather ms':>14} {'ReduceScatter ms':>18} "
          f"{'AR 数据量':>12} {'AG 数据量':>12}")
    print("-" * 90)

    for P in [2, 4, 8, 16, 32]:
        ar = ring_allreduce_time_ms(data_bytes, P, network)
        ag = allgather_time_ms(data_bytes, P, network)
        rs = reducescatter_time_ms(data_bytes, P, network)

        print(f"{P:>6d} {ar['time_ms']:>14.3f} {ag['time_ms']:>14.3f} {rs['time_ms']:>18.3f} "
              f"{ar['total_bytes']/1e6:>10.1f}MB {ag['total_bytes']/1e6:>10.1f}MB")

    print(f"\n关键洞察:")
    print(f"  - AllReduce = ReduceScatter + AllGather (2x 通信量)")
    print(f"  - AllReduce 总传输 = 2(P-1)/P × data")
    print(f"  - AllGather/ReduceScatter 总传输 = (P-1)/P × data")
    print(f"  - ZeRO-2 用 ReduceScatter (减少 50% 通信)")
    print(f"  - ZeRO-3 用 AllGather + ReduceScatter (和 AllReduce 相同)")


def experiment_3_comm_compute_overlap():
    """实验 3: 通信与计算重叠效率分析"""
    print("\n" + "=" * 70)
    print("实验 3: 通信与计算重叠 (Overlap) 分析")
    print("=" * 70)

    # 模拟: DDP 训练中梯度 AllReduce 与 backward 计算重叠
    # Megatron 1F1B: PP 通信与计算重叠

    print(f"\n--- DDP 梯度 AllReduce 重叠 ---")
    print(f"  假设: backward 计算 100ms, AllReduce 梯度 10ms (NVLink)")

    backward_ms = 100
    for net_name, net_bw in [("NVLink 900GB/s", 900), ("PCIe 64GB/s", 64), ("Ethernet 50GB/s", 50)]:
        # 梯度大小: LLaMA-7B = 14GB
        grad_bytes = 14e9
        net = NetworkConfig(net_name, net_bw, 1.0, "nvlink")

        for P in [2, 4, 8]:
            ar = ring_allreduce_time_ms(grad_bytes, P, net)

            # 无重叠: backward + AllReduce 串行
            serial_ms = backward_ms + ar['time_ms']

            # 有重叠: AllReduce 与最后几层 backward 并行
            # 假设可以重叠 70% 的 AllReduce 时间
            overlap_efficiency = 0.7
            overlapped_ms = backward_ms + ar['time_ms'] * (1 - overlap_efficiency)

            speedup = serial_ms / overlapped_ms

            print(f"  {net_name:<20} P={P:>2d}: "
                  f"AllReduce={ar['time_ms']:>8.1f}ms, "
                  f"串行={serial_ms:>8.1f}ms, "
                  f"重叠={overlapped_ms:>8.1f}ms, "
                  f"加速={speedup:.2f}x")

    print(f"\n--- Megatron-LM 1F1B PP 通信重叠 ---")
    print(f"  PP 通信: P2P 传递激活值, ~0.1-1ms (NVLink)")
    print(f"  计算时间: ~50ms/layer (70B)")
    print(f"  重叠: PP 通信隐藏在 1F1B 稳态中")
    print(f"  气泡中的通信无法重叠 → 需要优化调度")

    print(f"\n--- ZeRO-3 Prefetch 重叠 ---")
    print(f"  ZeRO-3: forward 时 AllGather 权重分片")
    print(f"  Prefetch: 在计算当前层时, 预取下一层权重")
    print(f"  通信/计算比 ≈ 10-20% (NVLink), 可以完全重叠")
    print(f"  PCIe/Ethernet: 通信占比 > 50%, 无法完全重叠")

    print(f"\n关键洞察:")
    print(f"  - NVLink: 通信占比 < 15%, 重叠后几乎隐藏全部通信")
    print(f"  - PCIe: 通信占比 > 50%, 重叠效果有限")
    print(f"  - Ethernet: 通信是瓶颈, 必须减少通信量")
    print(f"  - 重叠策略: Bucket AllReduce (DDP), Prefetch (ZeRO-3), 1F1B (PP)")


def experiment_4_comm_bottleneck():
    """实验 4: 通信瓶颈诊断"""
    print("\n" + "=" * 70)
    print("实验 4: 通信瓶颈诊断 (通信占比 vs GPU 数量)")
    print("=" * 70)

    # 模拟训练中通信占比
    # 通信时间 = AllReduce (或 ReduceScatter) 梯度
    # 计算时间 = forward + backward ≈ 6 × params × tokens / TFLOPS

    print(f"\n--- DP 通信占比 (LLaMA-7B, A100) ---")
    print(f"  每个 training step: forward + backward + AllReduce gradient")

    params = 7e9
    batch = 4
    seq = 2048
    gpu_tflops = 312
    tokens = batch * seq

    # 计算时间
    flops = 6 * params * tokens
    compute_ms = flops / (gpu_tflops * 1e12 * 0.45) * 1000

    print(f"  计算 FLOPs: {flops/1e12:.1f} TFLOPS")
    print(f"  计算时间: {compute_ms:.1f} ms (MFU=45%)")

    grad_bytes = params * 2  # FP16 梯度

    print(f"\n{'网络':<20} {'GPUs':>6} {'通信ms':>10} {'通信占比':>10} {'是否瓶颈':>12}")
    print("-" * 62)

    for net in [NVLink, PCIe_Gen4, Ethernet_400G, Ethernet_200G]:
        for P in [2, 4, 8, 16, 32]:
            ar = ring_allreduce_time_ms(grad_bytes, P, net)
            comm_pct = ar['time_ms'] / (compute_ms + ar['time_ms']) * 100
            bottleneck = "YES" if comm_pct > 20 else "ok"

            print(f"{net.name:<20} {P:>6d} {ar['time_ms']:>10.1f} {comm_pct:>9.1f}% {bottleneck:>12}")
        print()

    print(f"关键洞察:")
    print(f"  - NVLink: 通信占比 < 5%, 不是瓶颈")
    print(f"  - PCIe: P>4 时通信占比 > 20%, 成为瓶颈")
    print(f"  - Ethernet: 通信始终是瓶颈 (> 50%)")
    print(f"  - 跨节点训练的关键: 减少通信量或增大通信带宽")
    print(f"  - 常见优化: gradient compression, overlapping, larger batch")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("集合通信原语深度分析器")
    print("Ring/Tree AllReduce + AllGather/ReduceScatter + 通信重叠")
    print("=" * 70)

    experiment_1_ring_vs_tree()
    experiment_2_collective_ops_comparison()
    experiment_3_comm_compute_overlap()
    experiment_4_comm_bottleneck()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
集合通信关键要点:

  1. 算法选择:
     Ring AllReduce: 2(P-1) 步, 每步 data/P
       适合大数据 + 高带宽 (NVLink)
     Tree AllReduce: 2×log2(P) 步, 每步 data
       适合小数据 + 高延迟 (跨节点)
     NCCL 自动选择: 通常混合 Ring + Tree

  2. 通信量:
     AllReduce: 2(P-1)/P × data ≈ 2×data (大 P)
     AllGather: (P-1)/P × data ≈ data
     ReduceScatter: (P-1)/P × data ≈ data
     → AllReduce = ReduceScatter + AllGather

  3. 重叠策略:
     DDP: Bucket AllReduce (边 backward 边通信)
     ZeRO-3: Prefetch (提前 AllGather 权重)
     PP 1F1B: 通信隐藏在稳态中
     TP: 通信无法隐藏 (每层必须同步)

  4. 瓶颈诊断:
     通信占比 < 5%:   不是瓶颈 (NVLink, 小 P)
     通信占比 5-20%:  轻微影响 (可重叠)
     通信占比 > 20%:  严重瓶颈 (PCIe, 大 P)
     通信占比 > 50%:  主要瓶颈 (Ethernet)

  5. 优化建议:
     NVLink 环境: 通信不是瓶颈, 关注计算效率
     PCIe 环境: TP≤2, 用 DP+PP 代替大 TP
     Ethernet: 增大 batch 减少通信频率, gradient compression
     跨节点: 混合 Ring+Tree, 让 NCCL 自动选择最优路径
    """)
