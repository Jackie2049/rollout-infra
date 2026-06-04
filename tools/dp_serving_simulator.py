#!/usr/bin/env python3
"""Data Parallel Serving 模拟器

模拟 vLLM V1 的 Data Parallel Serving 行为:
1. DP vs TP 扩展对比
2. MoE Expert Parallelism + DP 协调开销
3. Wave-based 协调机制
4. DP Padding 和 CUDA Graph 兼容性
5. 生产部署选型

CPU 可运行，无需 GPU。
"""

import math
from dataclasses import dataclass


@dataclass
class GPUProfile:
    name: str
    hbm_bw_gbps: float  # HBM 带宽 (GB/s)
    hbm_gb: float        # HBM 容量 (GB)
    tflops_fp16: float   # FP16 TFLOPS
    nvlink_bw_gbps: float  # NVLink 双向带宽
    cost_per_hour: float    # $/hour


GPUS = {
    "H100": GPUProfile("H100 SXM", 3350, 80, 990, 900, 3.0),
    "H200": GPUProfile("H200 SXM", 4800, 141, 990, 900, 4.0),
    "A100": GPUProfile("A100 80G", 2039, 80, 312, 600, 2.0),
    "B200": GPUProfile("B200", 8000, 192, 2250, 1800, 8.0),
}

# DeepSeek-V3 MoE: 671B 总参数, 37B 激活参数, 256 experts, top-8
MOE_PROFILE = {
    "name": "DeepSeek-V3",
    "total_params_b": 671,
    "active_params_b": 37,
    "num_experts": 256,
    "top_k": 8,
    "hidden": 7168,
    "layers": 61,
    "weight_gb_fp16": 671 * 2 / 1000,  # ~1342 GB
    "active_weight_gb_fp16": 37 * 2 / 1000,  # ~74 GB
}

# Dense 模型
DENSE_PROFILES = {
    "70B": {"name": "LLaMA-70B", "params_b": 70, "weight_gb_fp16": 130},
    "405B": {"name": "LLaMA-405B", "params_b": 405, "weight_gb_fp16": 780},
}


def decode_throughput(weight_gb: float, hbm_bw: float, batch: int = 1) -> float:
    """估算 decode 吞吐 (tok/s/GPU)"""
    return batch * hbm_bw / weight_gb


def allreduce_latency_gb(data_gb: float, bw_gbps: float, algo: str = "ring") -> float:
    """AllReduce 延迟估算 (ms)"""
    if algo == "ring":
        # Ring: 2(n-1)/n × data/BW, 简化
        return data_gb / bw_gbps * 1000 * 2  # ms
    elif algo == "tree":
        return data_gb / bw_gbps * 1000 * 2 * math.log2(4)
    return data_gb / bw_gbps * 1000


def alltoall_latency_gb(data_gb: float, bw_gbps: float, num_ranks: int) -> float:
    """All-to-All 延迟估算 (ms)"""
    # 简化: data/ranks / bw
    return data_gb / num_ranks / bw_gbps * 1000


# ============================================================
# 实验
# ============================================================

def experiment1_dp_vs_tp():
    """实验 1: DP vs TP 扩展对比"""
    print("=" * 70)
    print("实验 1: Data Parallel vs Tensor Parallel 扩展对比")
    print("=" * 70)

    print("""
Dense 模型扩展策略:
  - TP: 切分模型权重, 减少 per-GPU 显存, 但增加通信
  - DP: 完全独立副本, 线性吞吐扩展, 无通信开销
  - DP+TP: 组合使用

MoE 模型扩展策略:
  - TP: 切分 Expert 权重
  - EP (Expert Parallel via DP): 每个 GPU 放不同 Expert
  - EP+TP: 组合使用
""")

    # Dense 70B
    m = DENSE_PROFILES["70B"]
    gpu = GPUS["H100"]
    print(f"模型: {m['name']} ({m['weight_gb_fp16']}GB FP16)")
    print(f"GPU: {gpu.name} ({gpu.hbm_gb}GB HBM)\n")

    print(f"{'策略':<24} {'GPU数':<8} {'单卡权重GB':<12} {'吞吐 tok/s':<14} {'通信开销':<12} {'线性度'}")
    print("-" * 82)

    strategies = [
        ("TP=1 (单卡)", 1, m["weight_gb_fp16"], 0, "不适用"),
        ("TP=2", 2, m["weight_gb_fp16"] / 2, 10, "~5%"),
        ("TP=4", 4, m["weight_gb_fp16"] / 4, 5, "~11%"),
        ("TP=8", 8, m["weight_gb_fp16"] / 8, 3, "~15%"),
        ("DP=2 (独立副本)", 2, m["weight_gb_fp16"], 0, "~100%"),
        ("DP=4 (独立副本)", 4, m["weight_gb_fp16"], 0, "~100%"),
        ("DP=8 (独立副本)", 8, m["weight_gb_fp16"], 0, "~100%"),
        ("TP=4 + DP=2", 8, m["weight_gb_fp16"] / 4, 5, "~97%"),
    ]

    base_tps = decode_throughput(m["weight_gb_fp16"], gpu.hbm_bw_gbps)
    for name, num_gpus, weight_per_gpu, comm_ms, linearity in strategies:
        tps = decode_throughput(weight_per_gpu, gpu.hbm_bw_gbps)
        total_tps = tps * num_gpus
        dp_count = num_gpus if "DP" in name else 1
        if "TP" in name and "DP" not in name:
            # TP 通信开销
            tp_size = num_gpus
            efficiency = max(0.7, 1.0 - 0.04 * math.log2(tp_size))
            total_tps *= efficiency

        print(f"{name:<24} {num_gpus:<8} {weight_per_gpu:<12.1f} {total_tps:<14.0f} {comm_ms:<12} {linearity}")

    print(f"\n基线单卡吞吐: {base_tps:.0f} tok/s")
    print("\n关键洞察:")
    print("  - Dense 模型: DP 扩展吞吐线性, 但不减少延迟")
    print("  - TP 扩展: 减少延迟, 但通信开销递增 (TP=8 ~15%)")
    print("  - DP=8 吞吐 = 8x 单卡, 完美线性")
    print("  - 选型: 延迟优先→TP, 吞吐优先→DP")


def experiment2_moe_ep_scaling():
    """实验 2: MoE Expert Parallelism + DP"""
    print("\n" + "=" * 70)
    print("实验 2: MoE Expert Parallelism 扩展分析")
    print("=" * 70)

    moe = MOE_PROFILE
    gpu = GPUS["H100"]
    print(f"模型: {moe['name']} ({moe['total_params_b']}B 总参数, {moe['active_params_b']}B 激活)")
    print(f"Expert: {moe['num_experts']} 个, Top-{moe['top_k']} 路由")
    print(f"FP16 权重: {moe['weight_gb_fp16']:.0f}GB (总), {moe['active_weight_gb_fp16']:.1f}GB (激活)\n")

    print(f"{'策略':<28} {'GPU数':<8} {'Expert/GPU':<12} {'A2A通信/ms':<14} {'DP同步/ms':<12} {'总延迟/ms'}")
    print("-" * 84)

    configs = [
        ("TP=8 (无 EP)", 8, moe["num_experts"], 0, 0),
        ("EP=16 (DP=16)", 16, moe["num_experts"] // 16, 2.5, 0.3),
        ("EP=32 (DP=32)", 32, moe["num_experts"] // 32, 1.8, 0.3),
        ("EP=64 (DP=64)", 64, moe["num_experts"] // 64, 1.5, 0.3),
        ("TP=8 + EP=16", 128, moe["num_experts"] // 16, 2.5, 0.3),
        ("TP=8 + EP=32", 256, moe["num_experts"] // 32, 1.8, 0.3),
    ]

    for name, num_gpus, experts_per_gpu, a2a_ms, dp_sync_ms in configs:
        # 激活参数 per GPU
        if "TP=8" in name and "EP" in name:
            # TP 切分 dense 部分, EP 切分 expert 部分
            active_per_gpu = moe["active_weight_gb_fp16"] / 8
            tp_comm = 5  # TP AllReduce
        elif "TP" in name:
            active_per_gpu = moe["weight_gb_fp16"] / num_gpus
            tp_comm = 5
        else:
            # EP only: 每个 GPU 有完整 dense 部分 + 部分 expert
            dense_gb = moe["weight_gb_fp16"] - moe["weight_gb_fp16"] * 0.875
            expert_per_gpu_gb = moe["weight_gb_fp16"] * 0.875 / num_gpus
            active_per_gpu = dense_gb + expert_per_gpu_gb * moe["top_k"]
            tp_comm = 0

        decode_ms = active_per_gpu / gpu.hbm_bw_gbps * 1000  # 简化
        total_ms = decode_ms + a2a_ms + tp_comm + dp_sync_ms

        print(f"{name:<28} {num_gpus:<8} {experts_per_gpu:<12} {a2a_ms:<14.1f} {dp_sync_ms:<12.1f} {total_ms:<.1f}")

    print("\n关键洞察:")
    print("  - EP 扩展: 每多一倍 GPU, Expert 分片更细")
    print("  - All-to-All 通信是 EP 主要开销")
    print("  - DeepEP 低延迟模式: ~1.5ms A2A (NVLink)")
    print("  - DP 同步开销很小 (~0.3ms AllReduce)")
    print("  - TP+EP 组合: 最优扩展 (TP 处理 dense, EP 处理 expert)")


def experiment3_wave_coordination():
    """实验 3: Wave-based 协调机制模拟"""
    print("\n" + "=" * 70)
    print("实验 3: Wave-based DP 协调机制")
    print("=" * 70)

    print("""
Wave 协调机制:
  - 所有 DP rank 在同一 Wave 内步调一致
  - Wave 结束条件: 所有 rank 无未完成请求
  - 每 32 步做一次 AllReduce 检查全局状态
  - 支持 DP Padding (CUDA Graph 兼容)
""")

    # 模拟 3 个 DP rank 的 Wave 处理
    print("模拟: 3 个 DP rank 处理请求\n")

    # 每个 rank 的请求队列 (token 数)
    rank_queues = [
        [256, 128, 64, 32, 16],    # Rank 0: 5 个请求
        [512, 256, 128, 64],        # Rank 1: 4 个请求
        [128, 64, 32, 16, 8, 4],   # Rank 2: 6 个请求
    ]

    wave = 0
    total_ar_count = 0
    step_count = 0

    while any(rank_queues):
        print(f"Wave {wave}:")
        max_tokens = 0
        rank_tokens = []

        for i, q in enumerate(rank_queues):
            if q:
                tokens = sum(q)
                rank_tokens.append(tokens)
                max_tokens = max(max_tokens, tokens)
                print(f"  DP Rank {i}: {len(q)} 个请求, {tokens} tokens → padding 到 {max_tokens}")
            else:
                rank_tokens.append(0)
                print(f"  DP Rank {i}: 空闲 (dummy batch {max_tokens} tokens)")

        # 模拟步数
        steps_this_wave = max(len(q) for q in rank_queues) if rank_queues else 0
        steps_this_wave = max(steps_this_wave, 1)

        # 每 32 步一次 AllReduce
        ar_this_wave = math.ceil(steps_this_wave / 32)
        total_ar_count += ar_this_wave
        step_count += steps_this_wave

        print(f"  → 总步数: {steps_this_wave}, AllReduce 次数: {ar_this_wave}")
        print(f"  → DP Padding 开销: {max_tokens * 3 - sum(rank_tokens)} tokens")

        # 清空已完成的队列
        rank_queues = [[] for _ in rank_queues]
        wave += 1
        print()

    print(f"总计: {wave} waves, {step_count} steps, {total_ar_count} AllReduce")
    print(f"AllReduce 频率: 每 {step_count / max(total_ar_count, 1):.0f} 步一次")

    print("\n关键洞察:")
    print("  - Wave 让所有 rank 同步: 空闲 rank 做 dummy batch")
    print("  - DP Padding 保证 CUDA Graph 兼容")
    print("  - AllReduce 频率: 每 32 步一次, 开销 <0.1ms")
    print("  - 负载不均导致浪费: 空闲 rank 等待忙 rank")


def experiment4_dbo_microbatch():
    """实验 4: Dual Batch Overlap (DBO) 分析"""
    print("\n" + "=" * 70)
    print("实验 4: Dual Batch Overlap (DBO) 分析")
    print("=" * 70)

    print("""
DBO (Dual Batch Overlap):
  - 将一个大 batch 拆成 2 个 microbatch
  - 在 MoE EP 中重叠计算和 All-to-All 通信
  - 条件: decode ≥ 32 tokens, prefill ≥ 512 tokens

原理:
  Microbatch 1: [Compute] → [All-to-All]
  Microbatch 2:            [Compute] → [All-to-All]
                         ↑ 重叠 ↑
""")

    gpu = GPUS["H100"]

    scenarios = [
        ("Decode, 64 tokens", "decode", 64, True),
        ("Decode, 128 tokens", "decode", 128, True),
        ("Decode, 256 tokens", "decode", 256, True),
        ("Prefill, 512 tokens", "prefill", 512, True),
        ("Prefill, 2048 tokens", "prefill", 2048, True),
        ("Decode, 16 tokens", "decode", 16, False),
        ("Prefill, 256 tokens", "prefill", 256, False),
    ]

    print(f"{'场景':<28} {'Token数':<10} {'DBO?':<8} {'无DBO/ms':<12} {'有DBO/ms':<12} {'加速比'}")
    print("-" * 82)

    a2a_ms = 2.0  # All-to-All 延迟

    for name, mode, tokens, use_dbo in scenarios:
        # Decode: memory-bound
        if mode == "decode":
            compute_ms = tokens * 0.05  # 简化: 0.05ms per token decode
        else:
            compute_ms = tokens * tokens * 0.00001  # prefill O(N²)

        no_dbo_ms = compute_ms + a2a_ms

        if use_dbo:
            # 2 microbatches, overlap
            half_compute = compute_ms / 2
            dbo_ms = half_compute + max(half_compute, a2a_ms) + half_compute * 0.1
            speedup = no_dbo_ms / dbo_ms
        else:
            dbo_ms = no_dbo_ms
            speedup = 1.0

        print(f"{name:<28} {tokens:<10} {'✓' if use_dbo else '✗':<8} {no_dbo_ms:<12.1f} {dbo_ms:<12.1f} {speedup:<.2f}x")

    print("\n关键洞察:")
    print("  - DBO 在大 batch 时有效: 重叠计算和通信")
    print("  - 小 batch (<32 decode): DBO 开销大于收益, 不启用")
    print("  - 最佳场景: 大 batch + 高 All-to-All 延迟")
    print("  - DBO 需要 DP rank 协调: AllReduce 决定是否使用")


def experiment5_production_guide():
    """实验 5: 生产部署 DP 选型"""
    print("\n" + "=" * 70)
    print("实验 5: 生产部署 DP 选型指南")
    print("=" * 70)

    scenarios = [
        {
            "name": "Dense 70B 推理",
            "model": "LLaMA-70B",
            "strategy": "TP=2 (单节点)",
            "dp_strategy": "无 DP",
            "reason": "70B FP8 ~65GB, TP=2 放入单节点 H100",
            "hardware": "2×H100",
        },
        {
            "name": "Dense 70B 高吞吐",
            "model": "LLaMA-70B",
            "strategy": "DP=4 (4 副本)",
            "dp_strategy": "Internal LB",
            "reason": "4 副本线性扩展, vLLM 内置 LB",
            "hardware": "8×H100 (2 节点)",
        },
        {
            "name": "MoE (DeepSeek-V3)",
            "model": "DeepSeek-V3 671B",
            "strategy": "TP=8 + EP=16",
            "dp_strategy": "DPEngineCoreProc",
            "reason": "TP 处理 dense, EP 分片 256 experts",
            "hardware": "128×H100 (16 节点)",
        },
        {
            "name": "MoE (Mixtral 8×7B)",
            "model": "Mixtral 8×7B",
            "strategy": "EP=8 (单节点)",
            "dp_strategy": "Internal LB",
            "reason": "8 experts 各放 1 GPU, Top-2 路由",
            "hardware": "8×H100 (1 节点)",
        },
        {
            "name": "MoE K8s 部署",
            "model": "DeepSeek-V3",
            "strategy": "TP=8 + EP=8",
            "dp_strategy": "External LB (K8s)",
            "reason": "每 Pod 1 DP rank, K8s Ingress 负载均衡",
            "hardware": "64×H100 (8 节点)",
        },
        {
            "name": "MoE 弹性扩缩",
            "model": "DeepSeek-V3",
            "strategy": "TP=8 + Elastic EP",
            "dp_strategy": "Elastic EP + EPLB",
            "reason": "动态增减 GPU, Expert 自动重分配",
            "hardware": "64-128×H100",
        },
    ]

    print(f"\n{'场景':<22} {'模型':<20} {'策略':<18} {'DP模式':<20} {'硬件'}")
    print("-" * 100)
    for s in scenarios:
        print(f"{s['name']:<22} {s['model']:<20} {s['strategy']:<18} {s['dp_strategy']:<20} {s['hardware']}")

    print("\n选型决策树:")
    print("  Dense 模型?")
    print("    → 需要低延迟? TP=2/4/8")
    print("    → 需要高吞吐? DP=N (独立副本)")
    print("  MoE 模型?")
    print("    → 单节点? EP=experts (NVLink All-to-All)")
    print("    → 多节点? TP+EP (DeepEP backend)")
    print("    → K8s? External LB + Elastic EP")
    print("    → 需要弹性? Elastic EP + EPLB")

    print("\nDP LB 模式选择:")
    print("  - Internal LB: 默认, vLLM 内置, 适合单集群")
    print("  - External LB: K8s 部署, 每 Pod 1 rank")
    print("  - Hybrid LB: 节点内 vLLM + 节点间外部 LB")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Data Parallel Serving 模拟器")
    print("DP vs TP 扩展 / MoE EP / Wave 协调 / DBO / 生产选型")
    print("=" * 70)

    experiment1_dp_vs_tp()
    experiment2_moe_ep_scaling()
    experiment3_wave_coordination()
    experiment4_dbo_microbatch()
    experiment5_production_guide()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Data Parallel Serving 核心要点:

  1. Dense 模型 DP:
     - 等价于多个独立实例
     - 完美线性吞吐扩展
     - 不减少单请求延迟
     - 推荐场景: 高吞吐 API 服务

  2. MoE Expert Parallelism:
     - DP 维度用于分片 Expert
     - 需要 All-to-All 通信
     - DPEngineCoreProc 处理 DP 协调
     - DeepEP / FlashInfer backend

  3. Wave 协调:
     - 所有 DP rank 同步步进
     - 每 32 步 AllReduce 检查完成状态
     - DP Padding 保证 CUDA Graph 兼容
     - 空闲 rank 做 dummy batch

  4. DBO (Dual Batch Overlap):
     - 重叠计算和 All-to-All
     - 阈值: decode ≥ 32, prefill ≥ 512
     - 所有 rank 通过 AllReduce 一致决策

  5. Elastic EP:
     - 运行时动态扩缩 DP size
     - Expert 自动重分配 (EPLB)
     - 适合云环境弹性需求
""")
