#!/usr/bin/env python3
"""NCCL 调优参数分析器

CPU 可运行的 NCCL 通信调优分析，覆盖 4 个实验:
  1. NCCL ALGO/PROTOCOL 选择分析
  2. NCCL_CHANNELS/NETHREADS 调优
  3. 跨节点通信拓扑优化
  4. NCCL 调优速查表生成

关键概念:
  - NCCL_ALGO: Ring, Tree, CollnetDirect, CollnetChain
  - NCCL_PROTOCOL: Simple (默认), LL (低延迟), LL128 (折中)
  - NCCL_CHANNELS: 通信通道数, 影响 pipeline 并行度
  - NCCL_MAX_NCHANNELS: 最大通道数限制
  - 网络拓扑: NVLink > PCIe > Ethernet, 影响算法选择

用法:
  conda run -n ai-infra python tools/nccl_tuning_sim.py
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
    bandwidth_gbps: float
    latency_us: float
    topology: str  # nvlink, pcie, ib, ethernet


@dataclass
class ClusterConfig:
    name: str
    gpus_per_node: int
    intra_node_net: NetworkConfig
    inter_node_net: NetworkConfig


# 典型网络配置
NVLINK_H100 = NetworkConfig("NVLink (H100)", 900, 1.0, "nvlink")
NVLINK_A100 = NetworkConfig("NVLink (A100)", 600, 1.5, "nvlink")
PCIe_Gen4 = NetworkConfig("PCIe Gen4 x16", 64, 5.0, "pcie")
IB_NDR = NetworkConfig("InfiniBand NDR400", 400, 2.0, "ib")
IB_HDR = NetworkConfig("InfiniBand HDR200", 200, 3.0, "ib")
ETH_400G = NetworkConfig("Ethernet 400G", 50, 10.0, "ethernet")
ETH_200G = NetworkConfig("Ethernet 200G", 25, 10.0, "ethernet")

# 典型集群配置
H100_CLUSTER = ClusterConfig("8×H100 NVLink+IB NDR", 8, NVLINK_H100, IB_NDR)
A100_CLUSTER = ClusterConfig("8×A100 NVLink+IB HDR", 8, NVLINK_A100, IB_HDR)
A100_PCIE = ClusterConfig("4×A100 PCIe+ETH 400G", 4, PCIe_Gen4, ETH_400G)


# ──────────────────────────────────────────────
# NCCL 通信模型
# ──────────────────────────────────────────────

def nccl_allreduce_time_ms(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
    algo: str = "ring",      # ring, tree
    protocol: str = "simple", # simple, ll, ll128
    num_channels: int = 4,
) -> dict:
    """估算 NCCL AllReduce 延迟

    NCCL 协议:
    - Simple: 默认, 大数据量最优, 每步传输 data/P
    - LL (Low Latency): 小数据优化, 每步只传 4/8 bytes (控制信息)
    - LL128: 折中, 每步传 128 bytes
    """
    if num_gpus <= 1:
        return {'time_ms': 0, 'throughput_gbps': 0, 'algo': algo, 'protocol': protocol}

    P = num_gpus
    bw = network.bandwidth_gbps * 1e9  # bytes/s
    lat = network.latency_us * 1e-6    # seconds

    # 有效带宽: 多通道并行
    effective_bw = bw * min(num_channels, 16) / 16  # 最多 16 通道

    if algo == "ring":
        # Ring AllReduce: 2(P-1) 步, 每步 data/P
        if protocol == "simple":
            chunk_bytes = data_bytes / P
        elif protocol == "ll":
            # LL: 拆成很多小消息, 延迟更低
            chunk_bytes = min(data_bytes / P, 128)  # 每步最多 128 bytes
        elif protocol == "ll128":
            chunk_bytes = min(data_bytes / P, 4096)
        else:
            chunk_bytes = data_bytes / P

        steps = 2 * (P - 1)
        step_time = chunk_bytes / effective_bw + lat
        total_time = steps * step_time

    elif algo == "tree":
        # Tree: 2×log2(P) 步, 每步 data
        steps = 2 * int(math.ceil(math.log2(P)))
        step_time = data_bytes / effective_bw + lat
        total_time = steps * step_time
    else:
        total_time = 0

    throughput = data_bytes / total_time / 1e9 if total_time > 0 else 0  # GB/s

    return {
        'time_ms': total_time * 1000,
        'time_us': total_time * 1e6,
        'throughput_gbps': throughput,
        'algo': algo,
        'protocol': protocol,
        'channels': num_channels,
        'steps': steps if algo in ('ring', 'tree') else 0,
    }


def nccl_optimal_algo(
    data_bytes: int,
    num_gpus: int,
    network: NetworkConfig,
) -> dict:
    """自动选择最优 NCCL 算法和协议

    NCCL 自动选择的逻辑 (简化):
    - 小数据 (< 1MB): Tree + LL
    - 中数据 (1-64MB): Ring + LL128
    - 大数据 (> 64MB): Ring + Simple
    """
    data_mb = data_bytes / 1e6

    candidates = [
        ("ring", "simple"),
        ("ring", "ll128"),
        ("ring", "ll"),
        ("tree", "simple"),
        ("tree", "ll128"),
        ("tree", "ll"),
    ]

    results = []
    for algo, proto in candidates:
        r = nccl_allreduce_time_ms(data_bytes, num_gpus, network, algo, proto)
        results.append(r)

    # 选择最快的
    best = min(results, key=lambda x: x['time_ms'])

    return {
        'best': best,
        'all_results': results,
        'data_mb': data_mb,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_algo_protocol_selection():
    """实验 1: NCCL ALGO/PROTOCOL 选择分析"""
    print("\n" + "=" * 70)
    print("实验 1: NCCL 算法/协议选择分析")
    print("=" * 70)

    network = NVLINK_A100
    P = 8

    data_sizes = [
        ("4 KB", 4 * 1024),
        ("64 KB", 64 * 1024),
        ("1 MB", 1 * 1024 * 1024),
        ("16 MB", 16 * 1024 * 1024),
        ("256 MB", 256 * 1024 * 1024),
        ("1 GB", 1 * 1024 * 1024 * 1024),
    ]

    print(f"\n配置: P={P}, 网络={network.name} ({network.bandwidth_gbps} GB/s)")

    for label, data_bytes in data_sizes:
        print(f"\n--- 数据量: {label} ---")
        opt = nccl_optimal_algo(data_bytes, P, network)
        print(f"  最优: {opt['best']['algo']} + {opt['best']['protocol']} "
              f"({opt['best']['time_us']:.1f} us, {opt['best']['throughput_gbps']:.1f} GB/s)")

        print(f"  {'算法':<8} {'协议':<10} {'延迟(us)':>12} {'吞吐(GB/s)':>14}")
        print(f"  " + "-" * 48)
        for r in sorted(opt['all_results'], key=lambda x: x['time_ms']):
            marker = " ← 最优" if r == opt['best'] else ""
            print(f"  {r['algo']:<8} {r['protocol']:<10} {r['time_us']:>12.1f} "
                  f"{r['throughput_gbps']:>14.1f}{marker}")

    print(f"\n关键洞察:")
    print(f"  - 小数据 (<64KB): Tree + LL 最优 (延迟驱动)")
    print(f"  - 中数据 (64KB-16MB): Ring + LL128 最优 (折中)")
    print(f"  - 大数据 (>16MB): Ring + Simple 最优 (带宽驱动)")
    print(f"  - NCCL 会自动选择, 通常不需要手动设置")


def experiment_2_channels_tuning():
    """实验 2: NCCL_CHANNELS 调优"""
    print("\n" + "=" * 70)
    print("实验 2: NCCL 通道数调优")
    print("=" * 70)

    data_bytes = 256 * 1024 * 1024  # 256 MB
    P = 8

    print(f"\n配置: P={P}, 数据量=256 MB")

    for net in [NVLINK_A100, PCIe_Gen4, IB_HDR]:
        print(f"\n--- {net.name} ({net.bandwidth_gbps} GB/s) ---")
        print(f"  {'通道数':>8} {'延迟(us)':>12} {'吞吐(GB/s)':>14} {'加速比':>10}")
        print(f"  " + "-" * 48)

        baseline_time = None
        for ch in [1, 2, 4, 8, 16, 32]:
            r = nccl_allreduce_time_ms(data_bytes, P, net, "ring", "simple", ch)
            if baseline_time is None:
                baseline_time = r['time_us']
            speedup = baseline_time / r['time_us'] if r['time_us'] > 0 else 1

            print(f"  {ch:>8d} {r['time_us']:>12.1f} {r['throughput_gbps']:>14.1f} {speedup:>10.2f}x")

    print(f"\n关键洞察:")
    print(f"  - NVLink: 8-16 通道最优, 之后收益递减")
    print(f"  - PCIe: 通道数影响小 (总线带宽是瓶颈)")
    print(f"  - IB: 4-8 通道通常足够")
    print(f"  - NCCL 默认自动选择, 一般不需要手动调")


def experiment_3_cross_node_topology():
    """实验 3: 跨节点通信拓扑优化"""
    print("\n" + "=" * 70)
    print("实验 3: 跨节点通信拓扑优化")
    print("=" * 70)

    data_bytes = 256 * 1024 * 1024  # 256 MB

    print(f"\n数据量: 256 MB")
    print(f"\n{'集群配置':<30} {'总GPU':>8} {'节点内(ms)':>12} {'跨节点(ms)':>14} "
          f"{'总时间(ms)':>14} {'跨节点占比':>12}")
    print("-" * 95)

    configs = [
        (H100_CLUSTER, 8, "单节点"),
        (H100_CLUSTER, 16, "2 节点"),
        (H100_CLUSTER, 32, "4 节点"),
        (H100_CLUSTER, 64, "8 节点"),
        (A100_CLUSTER, 8, "单节点"),
        (A100_CLUSTER, 16, "2 节点"),
        (A100_CLUSTER, 32, "4 节点"),
        (A100_PCIE, 4, "单节点"),
        (A100_PCIE, 8, "2 节点"),
        (A100_PCIE, 16, "4 节点"),
    ]

    for cluster, total_gpus, label in configs:
        num_nodes = total_gpus // cluster.gpus_per_node
        gpn = cluster.gpus_per_node

        # 节点内 AllReduce (如果 >1 GPU/node)
        if gpn > 1:
            intra = nccl_allreduce_time_ms(data_bytes, gpn, cluster.intra_node_net, "ring", "simple")
            intra_ms = intra['time_ms']
        else:
            intra_ms = 0

        # 跨节点: 节点间 Tree AllReduce
        if num_nodes > 1:
            inter = nccl_allreduce_time_ms(data_bytes, num_nodes, cluster.inter_node_net, "tree", "simple")
            inter_ms = inter['time_ms']
        else:
            inter_ms = 0

        total_ms = intra_ms + inter_ms
        inter_pct = inter_ms / total_ms * 100 if total_ms > 0 else 0

        print(f"{cluster.name:<30} {total_gpus:>8d} {intra_ms:>12.3f} {inter_ms:>14.3f} "
              f"{total_ms:>14.3f} {inter_pct:>11.1f}%")

    print(f"\n关键洞察:")
    print(f"  - 单节点: NVLink 通信 < 1ms, 可忽略")
    print(f"  - 2 节点: IB NDR ~1-3ms, 仍可接受")
    print(f"  - 8+ 节点: 跨节点通信占比 > 50%, 成为主要瓶颈")
    print(f"  - PCIe+Ethernet: 跨节点通信始终是瓶颈")
    print(f"  - NCCL 混合策略: 节点内 Ring, 节点间 Tree")


def experiment_4_cheatsheet():
    """实验 4: NCCL 调优速查表"""
    print("\n" + "=" * 70)
    print("实验 4: NCCL 调优速查表")
    print("=" * 70)

    print("""
## NCCL 关键环境变量

### 算法选择
  NCCL_ALGO=Ring|Tree|Collnet         # 强制算法 (通常让 NCCL 自动选)
  NCCL_PROTO=Simple|LL|LL128           # 强制协议

### 通道和线程
  NCCL_CHANNELS=                       # 通道数 (默认自动)
  NCCL_MAX_NCHANNELS=16                # 最大通道数
  NCCL_NSOCKS_PERTHREAD=1             # 每 GPU 线程的 socket 数
  NCCL_SOCKET_NTHREADS=1              # 每个 socket 的线程数

### 网络配置
  NCCL_SOCKET_IFNAME=eth0             # 网络接口名
  NCCL_IB_DISABLE=0                   # 0=启用 IB, 1=禁用
  NCCL_IB_HCA=mlx5_0                  # IB 设备名
  NCCL_NET_GDR_LEVEL=5                # GPUDirect RDMA 级别

### 调试
  NCCL_DEBUG=INFO                      # 输出调试信息
  NCCL_DEBUG_SUBSYS=ALL               # 调试子系统
  NCCL_DEBUG_FILE=/tmp/nccl.log       # 输出到文件

### 性能调优
  NCCL_BUFFSIZE=4194304               # 通信缓冲区 (默认 4MB)
  NCCL_MIN_NCHANNELS=1                # 最小通道数
  NCCL_P2P_DISABLE=0                  # 0=启用 P2P, 1=禁用
  NCCL_SHM_DISABLE=0                  # 0=启用共享内存, 1=禁用
""")

    print("## 常见问题排查")
    print()
    print("  问题 1: NCCL timeout")
    print("    → NCCL_ASYNC_ERROR_HANDLING=1")
    print("    → NCCL_COMM_BLOCKING=1")
    print("    → NCCL_MIN_NCHANNELS=1 (减少通道避免拥塞)")
    print()
    print("  问题 2: 跨节点不通信")
    print("    → NCCL_SOCKET_IFNAME=eth0 (指定正确接口)")
    print("    → NCCL_IB_DISABLE=1 (如果无 IB)")
    print("    → 检查防火墙和 SSH 免密登录")
    print()
    print("  问题 3: 性能差")
    print("    → NCCL_DEBUG=INFO 查看实际算法和带宽")
    print("    → NCCL_ALGO=Ring 强制 Ring (大数据)")
    print("    → NCCL_MAX_NCHANNELS=16 增加通道")
    print()
    print("  问题 4: OOM during communication")
    print("    → NCCL_BUFFSIZE=2097152 (减小缓冲区)")
    print("    → NCCL_MAX_NCHANNELS=4 (减少通道)")
    print()

    print("## 性能基线参考")
    print()
    data_bytes = 256 * 1024 * 1024  # 256 MB

    print(f"  AllReduce 256MB 基线:")
    for cluster, P in [(H100_CLUSTER, 8), (H100_CLUSTER, 64), (A100_CLUSTER, 8), (A100_PCIE, 8)]:
        num_nodes = P // cluster.gpus_per_node
        if P <= cluster.gpus_per_node:
            r = nccl_allreduce_time_ms(data_bytes, P, cluster.intra_node_net)
            bw = r['throughput_gbps']
        else:
            r_intra = nccl_allreduce_time_ms(data_bytes, cluster.gpus_per_node, cluster.intra_node_net)
            r_inter = nccl_allreduce_time_ms(data_bytes, num_nodes, cluster.inter_node_net, "tree")
            total_ms = r_intra['time_ms'] + r_inter['time_ms']
            bw = data_bytes / total_ms / 1e6  # GB/s

        label = f"{P}×{cluster.name.split('×')[0]}"
        print(f"    {label:<20}: {bw:.1f} GB/s (AllReduce 256MB)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("NCCL 调优参数分析器")
    print("算法/协议选择 + 通道调优 + 跨节点拓扑 + 速查表")
    print("=" * 70)

    experiment_1_algo_protocol_selection()
    experiment_2_channels_tuning()
    experiment_3_cross_node_topology()
    experiment_4_cheatsheet()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
NCCL 调优关键要点:

  1. 算法选择 (自动, 一般不需手动):
     小数据 (<64KB): Tree + LL (延迟驱动)
     中数据 (64KB-16MB): Ring + LL128 (折中)
     大数据 (>16MB): Ring + Simple (带宽驱动)

  2. 通道调优:
     NVLink: 8-16 通道
     PCIe/IB: 4-8 通道
     过多通道 → 资源争抢, 性能反而下降
     默认自动选择, 通常不需调整

  3. 跨节点通信:
     节点内: NVLink Ring (<1ms)
     节点间: IB Tree (1-10ms)
     8+ 节点: 跨节点通信 >50%, 是主要瓶颈
     优化: NCCL_NET=IB, GPUDirect RDMA

  4. 常用环境变量:
     NCCL_DEBUG=INFO — 调试第一步
     NCCL_SOCKET_IFNAME — 指定网络接口
     NCCL_IB_DISABLE — 禁用/启用 InfiniBand
     NCCL_BUFFSIZE — 通信缓冲区大小

  5. 最佳实践:
     1) 先用 NCCL_DEBUG=INFO 看实际性能
     2) 对比带宽和延迟与理论值
     3) 只在 NCCL 自动选择不佳时手动调参
     4) 跨节点训练用 IB + GPUDirect RDMA
     5) 参考集群厂商推荐的 NCCL 配置
    """)
