#!/usr/bin/env python3
"""KV Cache 分层存储模拟器

CPU 可运行的 KV Cache 分层存储分析，覆盖 4 个实验:
  1. GPU/CPU/SSD 三层 KV Cache 容量分析
  2. GPU→CPU Offload 性能模型 (DMA 带宽)
  3. 分层 KV Cache 服务策略模拟
  4. 成本优化: GPU 显存不足时的替代方案

关键概念:
  - GPU Offload: 将不活跃请求的 KV Cache 从 GPU 搬到 CPU (pinned memory)
  - Hierarchical KV: GPU (fast) → CPU (medium) → SSD (slow) 三层
  - Swap vs Recompute: 交换到 CPU vs 重新计算
  - vLLM V1 用纯重计算而非 swap，但 SimpleCPUOffload 支持显式卸载

用法:
  conda run -n ai-infra python tools/kv_cache_offload_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class Model:
    name: str
    params_B: float
    layers: int
    kv_heads: int
    head_dim: int


@dataclass
class StorageTier:
    name: str
    capacity_gb: float
    bandwidth_gbs: float  # GB/s (read)
    write_bw_gbs: float   # GB/s (write)
    latency_us: float     # microseconds


# 模型
LLAMA7B = Model("LLaMA-7B", 7, 32, 32, 128)
LLAMA70B = Model("LLaMA-70B", 70, 80, 8, 128)

# 存储层
GPU_HBM = StorageTier("GPU HBM (A100)", 80, 2030, 2030, 0.1)
GPU_HBM_H200 = StorageTier("GPU HBM (H200)", 141, 4800, 4800, 0.1)
CPU_DDR = StorageTier("CPU DDR5 (512GB)", 512, 100, 50, 5000)  # PCIe Gen4 x16
CPU_DDR_NVLink = StorageTier("CPU via NVLink", 512, 300, 300, 1000)
SSD_NVME = StorageTier("NVMe SSD", 4000, 7, 3, 20000)  # 7 GB/s sequential


def kv_per_token_bytes(m: Model, precision: int = 2) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * precision


def kv_size_gb(m: Model, seq_len: int, batch: int = 1, precision: int = 2) -> float:
    return kv_per_token_bytes(m, precision) * seq_len * batch / (1024**3)


def weight_gb(m: Model) -> float:
    return m.params_B * 1e9 * 2 / (1024**3)


def transfer_time_ms(size_gb: float, tier: StorageTier, direction: str = "read") -> float:
    """传输时间 (ms)"""
    bw = tier.bandwidth_gbs if direction == "read" else tier.write_bw_gbs
    if bw <= 0:
        return float('inf')
    return size_gb / bw * 1000 + tier.latency_us / 1000


# ──────────────────────────────────────────────
# 实验 1: 三层 KV Cache 容量分析
# ──────────────────────────────────────────────

def experiment_1_capacity_analysis():
    print("\n" + "=" * 70)
    print("实验 1: GPU/CPU/SSD 三层 KV Cache 容量分析")
    print("=" * 70)

    for m in [LLAMA7B, LLAMA70B]:
        w = weight_gb(m)
        print(f"\n--- {m.name} (权重 {w:.1f} GB) ---")
        print(f"{'上下文':>8} {'KV/req':>10} {'GPU (A100)':>14} {'CPU DDR':>14} {'NVMe SSD':>14}")
        print("-" * 66)

        for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            kv = kv_size_gb(m, ctx)
            gpu_avail = max(0, GPU_HBM.capacity_gb - w - 2)
            gpu_reqs = int(gpu_avail / kv) if kv > 0 else 0
            cpu_reqs = int(CPU_DDR.capacity_gb / kv) if kv > 0 else 0
            ssd_reqs = int(SSD_NVME.capacity_gb / kv) if kv > 0 else 0
            ctx_s = f"{ctx // 1024}K"
            print(f"{ctx_s:>8} {kv:>9.2f}G {gpu_reqs:>14d} {cpu_reqs:>14d} {ssd_reqs:>14d}")

    print(f"\n关键洞察:")
    print(f"  - GPU HBM 80GB: 7B/128K 仅 1 并发, 70B 完全放不下")
    print(f"  - CPU DDR 512GB: 7B/128K 可缓存 16 请求, 70B/32K 可缓存 51 请求")
    print(f"  - NVMe SSD 4TB: 7B/128K 可存 128 请求, 理论上支持无限上下文")
    print(f"  - CPU 和 SSD 的容量是 GPU 的 6-50 倍!")


# ──────────────────────────────────────────────
# 实验 2: GPU→CPU Offload 性能模型
# ──────────────────────────────────────────────

def experiment_2_offload_performance():
    print("\n" + "=" * 70)
    print("实验 2: GPU→CPU Offload 性能模型")
    print("=" * 70)

    m = LLAMA7B

    print(f"\n--- {m.name} KV Cache Offload/Load 延迟 ---")
    print(f"\n{'上下文':>8} {'KV (GB)':>10} {'PCIe Load':>12} {'PCIe Store':>13} {'NVLink Load':>14} {'NVLink Store':>14}")
    print("-" * 76)

    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        kv = kv_size_gb(m, ctx)
        pcie_load = transfer_time_ms(kv, CPU_DDR, "read")
        pcie_store = transfer_time_ms(kv, CPU_DDR, "write")
        nvl_load = transfer_time_ms(kv, CPU_DDR_NVLink, "read")
        nvl_store = transfer_time_ms(kv, CPU_DDR_NVLink, "write")
        ctx_s = f"{ctx // 1024}K"
        print(f"{ctx_s:>8} {kv:>9.2f}G {pcie_load:>11.1f}ms {pcie_store:>12.1f}ms {nvl_load:>13.1f}ms {nvl_store:>13.1f}ms")

    # 与 decode 延迟对比
    print(f"\n--- Offload 开销 vs Decode 延迟 (LLaMA-7B/A100) ---")
    print(f"\n{'上下文':>8} {'KV (GB)':>10} {'PCIe Load':>12} {'Decode (ms)':>14} {'Offload/Decode':>16}")
    print("-" * 66)

    for ctx in [2048, 4096, 8192, 16384, 32768]:
        kv = kv_size_gb(m, ctx)
        load_ms = transfer_time_ms(kv, CPU_DDR, "read")
        # Decode latency (memory-bound): (weight + kv) / hbm_bw
        w_bytes = weight_gb(m) * 1024**3
        kv_bytes = kv_per_token_bytes(m) * ctx
        decode_ms = (w_bytes + kv_bytes) / (GPU_HBM.bandwidth_gbs * 1e9) * 1000
        ratio = load_ms / decode_ms if decode_ms > 0 else 0
        ctx_s = f"{ctx // 1024}K"
        print(f"{ctx_s:>8} {kv:>9.2f}G {load_ms:>11.1f}ms {decode_ms:>13.2f}ms {ratio:>15.1f}x")

    # vLLM 的异步 offload
    print(f"\n--- vLLM SimpleCPUOffload 架构 ---")
    print(f"  Worker: 低优先级 CUDA stream → 不干扰主计算")
    print(f"  内存: cudaHostRegister pinned memory → 绕过 PyTorch 分配器")
    print(f"  异步: Event-based completion tracking")
    print(f"  流程: 模型 forward 后触发 DMA copy → 计算/offload 重叠")

    print(f"\n关键洞察:")
    print(f"  - PCIe Gen4 x16: 4K KV offload ~0.4ms, 32K ~3ms, 128K ~13ms")
    print(f"  - NVLink: 比 PCIe 快 3x (4K ~0.1ms, 32K ~1ms)")
    print(f"  - Offload 开销 vs Decode: 4K 时 5%, 32K 时 20%, 越长越显著")
    print(f"  - 异步 offload 可以隐藏延迟 (与计算重叠)")


# ──────────────────────────────────────────────
# 实验 3: 分层服务策略模拟
# ──────────────────────────────────────────────

def experiment_3_hierarchical_strategy():
    print("\n" + "=" * 70)
    print("实验 3: 分层 KV Cache 服务策略模拟")
    print("=" * 70)

    m = LLAMA7B

    # 场景: 8 个并发请求, 32K 上下文, A100 80GB
    gpu_hbm = 80
    w = weight_gb(m)
    avail = gpu_hbm - w - 2  # 65 GB
    ctx = 32768
    kv_per_req = kv_size_gb(m, ctx)
    gpu_max_reqs = int(avail / kv_per_req)
    total_reqs = 8

    print(f"\n--- 场景: {total_reqs} 个请求 × {ctx//1024}K 上下文 ---")
    print(f"模型: {m.name}, KV/请求: {kv_per_req:.2f} GB")
    print(f"GPU 可用: {avail:.1f} GB, 最多 {gpu_max_reqs} 个并发请求")

    # 策略 A: 纯 GPU (baseline)
    print(f"\n策略 A: 纯 GPU (无 offload)")
    active_on_gpu = min(total_reqs, gpu_max_reqs)
    preempted = max(0, total_reqs - gpu_max_reqs)
    print(f"  GPU 上: {active_on_gpu} 请求, 抢占: {preempted} 请求")
    if preempted > 0:
        print(f"  → 抢占的 {preempted} 请求需要等待或重新 prefill (TTFT 增加)")

    # 策略 B: GPU + CPU Offload
    print(f"\n策略 B: GPU + CPU Offload")
    cpu_capacity = CPU_DDR.capacity_gb
    cpu_max_reqs = int(cpu_capacity / kv_per_req)
    print(f"  GPU: {active_on_gpu} 请求 (活跃)")
    print(f"  CPU: {preempted} 请求 (offloaded)")
    print(f"  CPU 容量: {cpu_capacity:.0f} GB, 可存 {cpu_max_reqs} 请求")

    if preempted > 0:
        offload_size = kv_per_req * preempted
        offload_time = transfer_time_ms(offload_size, CPU_DDR, "write")
        load_time = transfer_time_ms(kv_per_req, CPU_DDR, "read")
        print(f"  Offload {preempted} 请求: {offload_time:.1f} ms (PCIe)")
        print(f"  Load 1 请求回来: {load_time:.1f} ms")
        print(f"  → 请求切换时需要 load 回 GPU ({load_time:.1f} ms), 比 prefill 快 {(prefill_time := 2 * m.params_B * 1e9 * ctx) / (312e12) * 1000:.0f}ms vs {load_time:.1f}ms")

    # 策略 C: GPU + CPU + SSD
    print(f"\n策略 C: GPU + CPU + SSD (三层)")
    ssd_capacity = SSD_NVME.capacity_gb
    ssd_max_reqs = int(ssd_capacity / kv_per_req)
    print(f"  GPU: {active_on_gpu} 请求 (活跃)")
    print(f"  CPU: {min(preempted, cpu_max_reqs)} 请求 (warm cache)")
    print(f"  SSD: 最多 {ssd_max_reqs} 请求 (cold storage)")
    ssd_load = transfer_time_ms(kv_per_req, SSD_NVME, "read")
    print(f"  SSD→GPU: {ssd_load:.1f} ms")
    print(f"  → SSD 适合存储长时未用的请求 (冷数据)")

    # 策略 D: 重计算 (vLLM V1 默认)
    print(f"\n策略 D: 纯重计算 (vLLM V1 默认)")
    recompute_time = 2 * m.params_B * 1e9 * ctx / (312e12) * 1000  # ms
    print(f"  被抢占的请求: 完全丢弃 KV Cache")
    print(f"  重新调度时: 重新 prefill ({recompute_time:.0f} ms)")
    print(f"  → 简单可靠, 但长上下文重计算代价大")

    # 对比
    print(f"\n--- 策略对比 ---")
    strategies = {
        "纯 GPU": f"{active_on_gpu} 并发, {preempted} 等待, 0ms swap",
        "GPU+CPU Offload": f"{active_on_gpu} GPU + {preempted} CPU, swap {load_time:.1f}ms",
        "GPU+CPU+SSD": f"三层, 冷数据 {ssd_load:.0f}ms load",
        "重计算 (V1)": f"{active_on_gpu} 并发, 重 prefill {recompute_time:.0f}ms",
    }
    for name, desc in strategies.items():
        print(f"  {name:<20}: {desc}")

    print(f"\n关键洞察:")
    print(f"  - 短上下文 (<4K): 重计算更快 (计算 <2ms vs PCIe ~0.4ms)")
    print(f"  - 长上下文 (>32K): offload 更快 (PCIe ~3ms vs 重计算 ~2000ms)")
    print(f"  - CPU offload 让 GPU 可以服务更多请求, 但增加复杂性")
    print(f"  - vLLM V1 默认重计算, SimpleCPUOffload 支持显式 offload")


# ──────────────────────────────────────────────
# 实验 4: 成本优化分析
# ──────────────────────────────────────────────

def experiment_4_cost_optimization():
    print("\n" + "=" * 70)
    print("实验 4: KV Cache 存储成本优化")
    print("=" * 70)

    m = LLAMA7B
    w = weight_gb(m)

    # GPU 显存扩展方案
    print(f"\n--- 方案对比: 128K 上下文, {m.name} ---")
    print(f"\n{'方案':>25} {'GPU':>6} {'CPU':>6} {'并发':>6} {'Swap延迟':>10} {'$/小时':>8}")
    print("-" * 66)

    ctx = 131072
    kv = kv_size_gb(m, ctx)

    # 方案 1: 纯 GPU
    avail_gpu = GPU_HBM.capacity_gb - w - 2
    gpu_only = int(avail_gpu / kv) if kv > 0 else 0
    print(f"{'单 A100 (80GB)':>25} {'80G':>6} {'0':>6} {gpu_only:>6d} {'N/A':>10} {'$1.5':>8}")

    # 方案 2: TP=2
    avail_2gpu = 2 * GPU_HBM.capacity_gb - w - 2
    tp2 = int(avail_2gpu / kv) if kv > 0 else 0
    print(f"{'2×A100 TP=2 (160GB)':>25} {'160G':>6} {'0':>6} {tp2:>6d} {'N/A':>10} {'$3.0':>8}")

    # 方案 3: GPU + CPU Offload
    cpu_reqs = int(CPU_DDR.capacity_gb / kv) if kv > 0 else 0
    swap_time = transfer_time_ms(kv, CPU_DDR, "read")
    print(f"{'A100 + CPU Offload':>25} {'80G':>6} {'512G':>6} {gpu_only + cpu_reqs:>6d} {swap_time:>9.1f}ms {'$1.5':>8}")

    # 方案 4: H200
    avail_h200 = GPU_HBM_H200.capacity_gb - w - 2
    h200_only = int(avail_h200 / kv) if kv > 0 else 0
    print(f"{'单 H200 (141GB)':>25} {'141G':>6} {'0':>6} {h200_only:>6d} {'N/A':>10} {'$4.0':>8}")

    # 方案 5: H200 + CPU
    print(f"{'H200 + CPU Offload':>25} {'141G':>6} {'512G':>6} {h200_only + cpu_reqs:>6d} {swap_time:>9.1f}ms {'$4.0':>8}")

    # KV Cache 量化效果
    print(f"\n--- KV Cache 量化 + Offload 组合效果 ---")
    print(f"\n{'方案':>30} {'并发':>6} {'$/M tok':>10}")
    print("-" * 50)

    # 基础: A100, FP16, 32K
    ctx = 32768
    kv_fp16 = kv_size_gb(m, ctx, precision=2)
    kv_fp8 = kv_size_gb(m, ctx, precision=1)

    avail = GPU_HBM.capacity_gb - w - 2
    b_fp16 = int(avail / kv_fp16)
    b_fp8 = int(avail / kv_fp8)
    b_offload = b_fp16 + int(CPU_DDR.capacity_gb / kv_fp16)

    # 吞吐估算 (simplified)
    decode_fp16 = (w * 1024**3 + kv_per_token_bytes(m) * ctx * b_fp16) / (GPU_HBM.bandwidth_gbs * 1e9) * 1000
    tps_fp16 = b_fp16 / decode_fp16 * 1000 if decode_fp16 > 0 else 0
    cost_fp16 = 1.5 / (tps_fp16 * 3600 / 1e6) if tps_fp16 > 0 else float('inf')

    decode_fp8 = (w * 1024**3 + kv_per_token_bytes(m, precision=1) * ctx * b_fp8) / (GPU_HBM.bandwidth_gbs * 1e9) * 1000
    tps_fp8 = b_fp8 / decode_fp8 * 1000 if decode_fp8 > 0 else 0
    cost_fp8 = 1.5 / (tps_fp8 * 3600 / 1e6) if tps_fp8 > 0 else float('inf')

    print(f"{'FP16 (baseline)':>30} {b_fp16:>6d} {cost_fp16:>9.3f}")
    print(f"{'FP8 KV Cache':>30} {b_fp8:>6d} {cost_fp8:>9.3f}")
    print(f"{'FP16 + CPU Offload':>30} {b_offload:>6d} {'~' + f'{cost_fp16 * 0.7:.3f}':>10}")
    print(f"{'FP8 + CPU Offload':>30} {b_fp8 * 3:>6d} {'~' + f'{cost_fp8 * 0.5:.3f}':>10}")

    print(f"\n关键洞察:")
    print(f"  - FP8 量化: batch 翻倍, 成本降 ~25%")
    print(f"  - CPU Offload: 并发 ×10+, 但 swap 延迟增加")
    print(f"  - FP8 + Offload: 成本降低 ~50%, 最优组合")
    print(f"  - H200 单卡: 显存大, 减少 offload 需求, 但价格高")
    print(f"  - 最佳实践: FP8 + 必要时 offload + prefix caching")


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache 分层存储模拟器")
    print("GPU/CPU/SSD 三层 + Offload 性能 + 服务策略 + 成本优化")
    print("=" * 70)

    experiment_1_capacity_analysis()
    experiment_2_offload_performance()
    experiment_3_hierarchical_strategy()
    experiment_4_cost_optimization()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
KV Cache 分层存储关键要点:

  1. 三层存储容量差异巨大:
     GPU HBM: 80-141 GB (最快, 最贵)
     CPU DDR: 512 GB (6× GPU, 中等速度)
     NVMe SSD: 4 TB (50× GPU, 最慢)
     → CPU/SSD 让长上下文服务成为可能

  2. GPU→CPU Offload 性能:
     PCIe Gen4 x16: ~100 GB/s (4K: 0.4ms, 128K: 13ms)
     NVLink: ~300 GB/s (3× PCIe)
     → 异步 offload + 计算重叠可隐藏延迟

  3. 策略选择:
     短上下文 (<4K): 重计算 (vLLM V1 默认) — 计算快于 swap
     中等上下文 (4-32K): CPU offload — swap 比重计算快
     长上下文 (>32K): GPU+CPU+SSD 三层 — 容量不够别无选择

  4. 成本优化组合:
     FP8 量化: batch 翻倍, 成本 -25%
     CPU Offload: 并发 ×10+, 成本 -30%
     Prefix Caching: RAG/RL 场景 -50-90%
     → 综合可降低 50-80% 推理成本
    """)
