#!/usr/bin/env python3
"""NCCL 调优速查表 — 生产环境实用工具

生成基于硬件拓扑和训练配置的 NCCL 调优推荐:
1. 自动检测拓扑并推荐参数
2. AllReduce 算法/协议选择
3. 通道调优
4. 常见问题速查

CPU 可运行，无需 GPU。
"""

import math
from dataclasses import dataclass


# ============================================================
# 拓扑参数模型
# ============================================================

@dataclass
class Topology:
    name: str
    gpu_count: int
    gpu_type: str          # e.g. "A100", "H100"
    interconnect: str      # "nvlink", "nvlink+ib", "pcie+ib", "pcie+eth"
    nvlink_bw_gbps: float  # per GPU bidirectional
    ib_bw_gbps: float      # InfiniBand per link
    nodes: int = 1
    gpus_per_node: int = 8

    @property
    def is_single_node(self):
        return self.nodes == 1

    @property
    def has_nvlink(self):
        return "nvlink" in self.interconnect

    @property
    def has_ib(self):
        return "ib" in self.interconnect


TOPOLOGIES = {
    "单机8×A100-NVLink": Topology("单机8×A100-NVLink", 8, "A100", "nvlink", 300, 0),
    "单机8×H100-NVLink": Topology("单机8×H100-NVLink", 8, "H100", "nvlink", 450, 0),
    "2节点16×A100-NVLink+IB": Topology("2节点16×A100-NVLink+IB", 16, "A100", "nvlink+ib", 300, 200, 2, 8),
    "4节点32×H100-NVLink+IB": Topology("4节点32×H100-NVLink+IB", 32, "H100", "nvlink+ib", 450, 400, 4, 8),
    "8节点64×A100-PCIe+IB": Topology("8节点64×A100-PCIe+IB", 64, "A100", "pcie+ib", 64, 200, 8, 8),
    "2节点16×A100-PCIe+Eth": Topology("2节点16×A100-PCIe+Eth", 16, "A100", "pcie+eth", 64, 12.5, 2, 8),
}


def allreduce_time_bytes(data_bytes: float, topo: Topology, algo: str = "ring",
                          proto: str = "simple") -> float:
    """估算 AllReduce 延迟 (ms)"""
    n = topo.gpu_count

    if algo == "ring":
        # Ring: 2(N-1)/N 次数据传输
        if topo.is_single_node and topo.has_nvlink:
            bw = topo.nvlink_bw_gbps * 1e9
        elif topo.has_ib:
            # 跨节点: 瓶颈在节点间带宽
            bw = topo.ib_bw_gbps * 1e9 * topo.gpus_per_node  # 每节点一条链路
        else:
            bw = topo.nvlink_bw_gbps * 1e9 if topo.has_nvlink else 12.5e9

        transfer_bytes = data_bytes * 2 * (n - 1) / n
        latency_ms = transfer_bytes / bw * 1000
    elif algo == "tree":
        # Tree: log2(N) 步, 每步传输 data_bytes
        if topo.is_single_node and topo.has_nvlink:
            bw = topo.nvlink_bw_gbps * 1e9
        else:
            bw = topo.ib_bw_gbps * 1e9 if topo.has_ib else 12.5e9
        steps = math.ceil(math.log2(n))
        latency_ms = data_bytes * steps / bw * 1000
    else:
        latency_ms = 0.1  # 默认

    # 协议开销
    if proto == "ll":
        latency_ms += 0.05 * math.log2(n)  # Low Latency 协议有额外开销
    elif proto == "simple":
        pass

    return latency_ms


# ============================================================
# 实验
# ============================================================

def experiment1_algorithm_selection():
    """实验 1: 算法选择指南"""
    print("=" * 70)
    print("实验 1: NCCL AllReduce 算法选择指南")
    print("=" * 70)

    # 不同数据大小的推荐
    data_sizes = [
        ("小 (<1MB)", 0.5 * 1e6),
        ("中 (10MB)", 10 * 1e6),
        ("大 (100MB)", 100 * 1e6),
        ("很大 (1GB)", 1 * 1e9),
        ("超大 (10GB)", 10 * 1e9),
    ]

    for topo_name, topo in list(TOPOLOGIES.items())[:3]:
        print(f"\n--- {topo_name} ---")
        print(f"{'数据量':<16} {'Ring (ms)':<12} {'Tree (ms)':<12} {'推荐算法':<14} {'推荐协议'}")
        print("-" * 66)

        for size_name, data_bytes in data_sizes:
            ring_simple = allreduce_time_bytes(data_bytes, topo, "ring", "simple")
            tree_simple = allreduce_time_bytes(data_bytes, topo, "tree", "simple")
            ring_ll = allreduce_time_bytes(data_bytes, topo, "ring", "ll")
            tree_ll = allreduce_time_bytes(data_bytes, topo, "tree", "ll")

            # 选择最优
            candidates = [
                ("Ring+Simple", ring_simple),
                ("Tree+Simple", tree_simple),
                ("Ring+LL", ring_ll),
                ("Tree+LL", tree_ll),
            ]
            best_name, best_time = min(candidates, key=lambda x: x[1])

            algo = "Ring" if "Ring" in best_name else "Tree"
            proto = "Simple" if "Simple" in best_name else "LL"

            print(f"{size_name:<16} {ring_simple:<12.3f} {tree_simple:<12.3f} {algo:<14} {proto}")

    print("\n关键规则:")
    print("  - 小数据 (<1MB): Tree + LL (延迟优先)")
    print("  - 大数据 (>100MB): Ring + Simple (带宽优先)")
    print("  - NVLink 单节点: Ring 几乎总是最优")
    print("  - 跨节点: 取决于 IB 带宽, 数据大时 Ring 更优")


def experiment2_channel_tuning():
    """实验 2: 通道调优"""
    print("\n" + "=" * 70)
    print("实验 2: NCCL 通道调优")
    print("=" * 70)

    print("""
NCCL 通道 (Channel) 决定并行度:
- 每个通道独立传输数据
- 更多通道 = 更高带宽利用率
- 但更多通道 = 更多内存和开销

典型配置:
""")

    configs = [
        ("NVLink (同节点)", "nvlink", 8, 16, "4-16 通道, 默认通常够用"),
        ("InfiniBand (跨节点)", "ib", 2, 8, "2-8 通道, 受限于 IB 链路数"),
        ("Ethernet (跨节点)", "eth", 1, 4, "1-4 通道, 带宽受限"),
        ("RoCE (跨节点)", "roce", 2, 8, "2-8 通道, 配置类似 IB"),
    ]

    print(f"{'互联类型':<24} {'推荐范围':<14} {'默认':<8} {'说明'}")
    print("-" * 66)
    for name, link, min_ch, max_ch, desc in configs:
        default = (min_ch + max_ch) // 2
        print(f"{name:<24} {min_ch}-{max_ch:<10} {default:<8} {desc}")

    print("\n调优命令:")
    print("  # 设置通道数")
    print("  export NCCL_MAX_NCHANNELS=8")
    print("  export NCCL_MIN_NCHANNELS=4")
    print("  # 设置 NVLink 通道数")
    print("  export NCCL_MAX_NVLINK_CHANNEL_COUNT=16")
    print("  # 设置 NET 通道数")
    print("  export NCCL_MAX_NET_CHANNEL_COUNT=8")


def experiment3_multi_node_tuning():
    """实验 3: 多节点调优"""
    print("\n" + "=" * 70)
    print("实验 3: 多节点训练 NCCL 调优")
    print("=" * 70)

    print("""
多节点训练的 NCCL 调优清单:

1. 网络拓扑发现:
   $ nvidia-smi topo -m         # 查看 GPU 拓扑
   $ ibstat                     # 查看 IB 状态
   $ ibv_devinfo                # IB 设备信息

2. 关键环境变量:
""")

    params = [
        ("NCCL_SOCKET_IFNAME", "eth0,ib0", "指定网络接口"),
        ("NCCL_IB_DISABLE", "0", "启用 InfiniBand"),
        ("NCCL_IB_HCA", "mlx5_0,mlx5_1", "指定 IB 设备"),
        ("NCCL_IB_GID_INDEX", "3", "RoCE v2 用 3"),
        ("NCCL_IB_TC", "106", "RoCE DCQCN, DSCP=26"),
        ("NCCL_NET_GDR_LEVEL", "5", "GPUDirect RDMA"),
        ("NCCL_IB_QPS_PER_CONNECTION", "4", "每连接 QP 数"),
        ("NCCL_MAX_NCHANNELS", "8", "最大通道数"),
        ("NCCL_ALGO", "Ring", "强制算法 (调试用)"),
        ("NCCL_PROTO", "Simple", "强制协议 (调试用)"),
        ("NCCL_DEBUG", "INFO", "调试日志级别"),
        ("NCCL_DEBUG_SUBSYS", "INIT,COLL", "调试子系统"),
        ("NCCL_BUFFSIZE", "8388608", "内部缓冲区大小 (8MB)"),
        ("NCCL_IB_RETRY_CNT", "7", "IB 重试次数"),
        ("NCCL_IB_TIMEOUT", "22", "IB 超时 (log2(us))"),
    ]

    print(f"{'变量':<36} {'示例值':<20} {'说明'}")
    print("-" * 80)
    for name, value, desc in params:
        print(f"{name:<36} {value:<20} {desc}")

    # 跨节点通信占比分析
    print("\n--- 跨节点通信占比分析 ---\n")
    scenarios = [
        ("TP=2 (单节点)", 2, 1, True),
        ("TP=8 (单节点)", 8, 1, True),
        ("TP=8 + PP=2 (2节点)", 8, 2, False),
        ("TP=8 + PP=4 (4节点)", 8, 4, False),
        ("DP=4 + TP=8 (4节点)", 8, 4, False),
    ]

    print(f"{'配置':<28} {'跨节点通信':<16} {'通信占比':<10} {'瓶颈'}")
    print("-" * 66)

    for name, tp, nodes, single_node in scenarios:
        if single_node:
            print(f"{name:<28} {'无':<16} {'~2-5%':<10} {'NVLink BW'}")
        else:
            # 跨节点通信
            if "PP" in name:
                print(f"{name:<28} {'PP activation':<16} {'~5-10%':<10} {'IB BW'}")
            else:
                print(f"{name:<28} {'DP gradient':<16} {'~3-8%':<10} {'IB BW'}")


def experiment4_troubleshooting():
    """实验 4: 常见问题速查"""
    print("\n" + "=" * 70)
    print("实验 4: NCCL 常见问题速查")
    print("=" * 70)

    problems = [
        {
            "name": "NCCL 超时 (Timeout)",
            "symptoms": "Watchdog caught timeout, collective hanging",
            "causes": [
                "1. 网络不通: 检查 NCCL_SOCKET_IFNAME 和防火墙",
                "2. IB 链路异常: ibstat 检查 PortState=Active",
                "3. GPU 掉卡: nvidia-smi 检查所有 GPU",
                "4. 负载不均: 某些 rank 处理慢",
            ],
            "fix": "NCCL_DEBUG=INFO 查看 rank 卡在哪步, 检查网络/IB/GPU",
        },
        {
            "name": "NCCL 初始化失败",
            "symptoms": "invalid config, failed to init",
            "causes": [
                "1. NCCL_NET 不支持: 检查 IB driver 版本",
                "2. NCCL_SOCKET_IFNAME 错误: ifconfig 确认接口名",
                "3. 端口冲突: NCCL_SOCKET_PORT 指定其他端口",
            ],
            "fix": "NCCL_DEBUG=TRACE 查看初始化每步, 确认 IB driver 和网络",
        },
        {
            "name": "NCCL 带宽不足",
            "symptoms": "AllReduce 带宽远低于理论值",
            "causes": [
                "1. 通道数不够: 增加 NCCL_MAX_NCHANNELS",
                "2. NUMA 绑定: 确认 GPU 和 NIC 在同一 NUMA",
                "3. IB GID 错误: RoCE 用 NCCL_IB_GID_INDEX=3",
                "4. 拓扑不优: nvidia-smi topo -m 检查",
            ],
            "fix": "nccl-tests 测实际 BW, 对比理论值, 调通道/NUMA/IB参数",
        },
        {
            "name": "NCCL 内存不足",
            "symptoms": "NCCL internal error, out of memory",
            "causes": [
                "1. NCCL_BUFFSIZE 太大: 降低缓冲区",
                "2. 通道数太多: 减少 NCCL_MAX_NCHANNELS",
                "3. 注册内存: GPUDirect RDMA 占用额外内存",
            ],
            "fix": "降低 NCCL_BUFFSIZE/NCCL_MAX_NCHANNELS, 或禁用 GDR",
        },
    ]

    for p in problems:
        print(f"\n### {p['name']}")
        print(f"症状: {p['symptoms']}")
        print("原因:")
        for c in p['causes']:
            print(f"  {c}")
        print(f"解决: {p['fix']}")

    print("\n\n调试命令速查:")
    print("  NCCL_DEBUG=INFO          # 查看通信详情")
    print("  NCCL_DEBUG=TRACE         # 最详细日志")
    print("  NCCL_DEBUG_SUBSYS=ALL    # 所有子系统")
    print("  NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log  # 日志写入文件")
    print("  nvidia-smi topo -m       # GPU 拓扑矩阵")
    print("  ibstat                   # IB 状态")
    print("  ibv_devinfo              # IB 设备详情")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NCCL 调优速查表")
    print("基于硬件拓扑的调优推荐 + 常见问题速查")
    print("=" * 70)

    experiment1_algorithm_selection()
    experiment2_channel_tuning()
    experiment3_multi_node_tuning()
    experiment4_troubleshooting()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
NCCL 调优核心原则:

  1. 算法选择:
     小数据 → Tree + LL (延迟优先)
     大数据 → Ring + Simple (带宽优先)
     NVLink 单节点 → Ring 几乎总是最优

  2. 通道调优:
     NVLink: 8-16 通道
     IB: 2-8 通道
     Ethernet: 1-4 通道

  3. 多节点:
     确认 IB 状态 (ibstat → Active)
     正确设置 NCCL_SOCKET_IFNAME
     GPUDirect RDMA 启用 (NCCL_NET_GDR_LEVEL=5)

  4. 调试:
     NCCL_DEBUG=INFO 查看通信详情
     nvidia-smi topo -m 检查拓扑
     nccl-tests 基准测试验证

  5. 生产建议:
     先用 nccl-tests 测基线
     调优后对比带宽利用率 (>80% 良好)
     记录环境变量到部署脚本
""")
