#!/usr/bin/env python3
"""CUDA Kernel 执行模拟器

模拟 GPU Kernel 执行的关键概念:
1. Grid/Block/Thread 层次结构
2. 内存层次 (Global/Shared/Register)
3. Coalesced Memory Access
4. Warp Divergence
5. Occupancy 分析
6. Roofline Model

CPU 可运行，无需 GPU。
"""

import math
from dataclasses import dataclass
from typing import List


# ============================================================
# GPU 架构参数模型
# ============================================================

@dataclass
class GPUArch:
    name: str
    sm_count: int           # Streaming Multiprocessor 数量
    max_threads_per_sm: int # 每 SM 最大线程数
    max_threads_per_block: int  # 每 block 最大线程
    shared_mem_per_sm_kb: int   # 每 SM 共享内存 (KB)
    reg_per_sm: int         # 每 SM 寄存器数
    global_bw_gbps: float   # Global Memory 带宽 (GB/s)
    fp16_tflops: float      # FP16 算力 (TFLOPS)
    clock_mhz: int          # 基础频率

GPUS = {
    "A100": GPUArch("A100 (SM 8.0)", 108, 2048, 1024, 164, 65536, 2035, 312, 1410),
    "H100": GPUArch("H100 (SM 9.0)", 132, 2048, 1024, 228, 65536, 3350, 990, 1830),
    "H200": GPUArch("H200 (SM 9.0)", 132, 2048, 1024, 228, 65536, 4800, 990, 1830),
}


def occupancy(sm: GPUArch, threads_per_block: int, regs_per_thread: int = 32,
              shared_mem_per_block_bytes: int = 0) -> dict:
    """计算 Kernel Occupancy"""
    # Block 约束
    blocks_by_threads = sm.max_threads_per_sm // threads_per_block
    blocks_by_regs = sm.reg_per_sm // (regs_per_thread * threads_per_block) if regs_per_thread > 0 else blocks_by_threads
    shared_mem_per_block_kb = shared_mem_per_block_bytes / 1024
    blocks_by_shared = sm.shared_mem_per_sm_kb // shared_mem_per_block_kb if shared_mem_per_block_kb > 0 else blocks_by_threads

    active_blocks = min(blocks_by_threads, blocks_by_regs, blocks_by_shared)
    active_threads = active_blocks * threads_per_block
    occ_pct = active_threads / sm.max_threads_per_sm * 100

    return {
        "active_blocks_per_sm": active_blocks,
        "active_threads_per_sm": active_threads,
        "occupancy_pct": occ_pct,
        "limiting_factor": "threads" if active_blocks == blocks_by_threads
                          else "registers" if active_blocks == blocks_by_regs
                          else "shared_mem",
    }


def memory_access_time(bytes_count: float, bw_gbps: float) -> float:
    """内存访问时间 (ms)"""
    return bytes_count / (bw_gbps * 1e6)  # GB/s → bytes/ms


def compute_time(flops: float, tflops: float) -> float:
    """计算时间 (ms)"""
    return flops / (tflops * 1e9)  # TFLOPS → GFLOPS/ms


# ============================================================
# 实验
# ============================================================

def experiment1_grid_block_hierarchy():
    """实验 1: Grid/Block/Thread 层次结构"""
    print("=" * 70)
    print("实验 1: CUDA Grid/Block/Thread 层次结构")
    print("=" * 70)

    # 模拟向量加法: C = A + B, N = 1M elements
    N = 1_000_000
    bytes_per_element = 4  # float32

    print(f"\n向量加法: N = {N:,} elements, float32")
    print(f"\n{'Block Size':<12} {'Grid Size':<12} {'Total Threads':<16} {'理论效率':<10} {'备注'}")
    print("-" * 66)

    for block_size in [32, 64, 128, 256, 512, 1024]:
        grid_size = math.ceil(N / block_size)
        total_threads = grid_size * block_size
        wasted = total_threads - N
        efficiency = N / total_threads * 100
        note = ""
        if block_size == 32:
            note = "最小 warp, 低效率"
        elif block_size == 128:
            note = "常用配置"
        elif block_size == 256:
            note = "最常用配置"
        elif block_size == 1024:
            note = "最大 block"
        print(f"{block_size:<12} {grid_size:<12} {total_threads:<16} {efficiency:<10.1f}% {note}")

    print("\n关键概念:")
    print("  Grid = N 个 Block, Block = M 个 Thread")
    print("  Thread 是最小执行单元, Block 内可共享 Shared Memory")
    print("  Warp = 32 个 Thread (GPU 调度基本单位)")
    print("  Block Size 通常是 128/256/512 (2/4/8 warps per block)")


def experiment2_memory_hierarchy():
    """实验 2: 内存层次与带宽分析"""
    print("\n" + "=" * 70)
    print("实验 2: GPU 内存层次与带宽分析")
    print("=" * 70)

    # 内存层次 (以 A100 为例)
    mem_levels = [
        ("Register",       "~65536/SM",   "所有",        "~0.3",    "1 cycle",   "最快, 容量最小"),
        ("Shared Memory",  "164 KB/SM",   "Block 内",    "~1.5",    "~5 cycles", "Block 内共享, 可编程"),
        ("L1 Cache",       "192 KB/SM",   "SM 内",       "~1.5",    "~30 cycles","自动缓存, 与 SMEM 共享"),
        ("L2 Cache",       "40 MB",       "所有 SM",     "~3.0",    "~200 cycles","全局共享"),
        ("HBM (Global)",   "80 GB",       "所有 SM",     "2035",     "~400 cycles","大容量, 高延迟"),
    ]

    print(f"\nA100 内存层次:")
    print(f"{'层级':<16} {'容量':<16} {'共享范围':<12} {'带宽 GB/s':<12} {'延迟':<14} {'说明'}")
    print("-" * 86)
    for name, cap, scope, bw, latency, desc in mem_levels:
        print(f"{name:<16} {cap:<16} {scope:<12} {bw:<12} {latency:<14} {desc}")

    # Matrix Multiply 示例: 数据复用分析
    print(f"\n--- 矩阵乘法 C[M,K] × B[K,N] 的数据复用 (Tile=64) ---\n")
    M = N_dim = K = 1024
    tile = 64

    # 不分块: 每个输出元素读 K 个 A + K 个 B
    naive_reads = M * N_dim * K * 2 * 4  # float32
    naive_writes = M * N_dim * 4
    total_ops = M * N_dim * K * 2  # FMA = 2 ops
    arithmetic_intensity_naive = total_ops / ((naive_reads + naive_writes) / 1e9)

    # 分块: 每个 tile 读 tile×K + K×tile = 2×tile×K 个元素, 产生 tile×tile 个输出
    tile_reads = 2 * tile * K * 4  # bytes per tile
    tile_ops = tile * tile * K * 2  # ops per tile
    tiles_count = (M // tile) * (N_dim // tile)
    total_tile_reads = tile_reads * tiles_count
    ai_tiled = tile_ops / (tile_reads / 1e9)

    # Shared Memory 优化: 每个 tile 读 2×tile×tile 个元素 (从 global), 产生 tile×tile 输出
    smem_reads = 2 * tile * tile * 4
    smem_ops = tile * tile * tile * 2
    ai_smem = smem_ops / (smem_reads / 1e9)

    print(f"{'策略':<24} {'Global 读 (GB)':<18} {'AI (ops/byte)':<16} {'复用倍数':<12}")
    print("-" * 70)
    print(f"{'朴素 (无分块)':<24} {naive_reads/1e9:<18.2f} {arithmetic_intensity_naive:<16.1f} {'1x':<12}")
    print(f"{'Global Memory 分块':<24} {total_tile_reads/1e9:<18.2f} {ai_tiled:<16.1f} {ai_tiled/arithmetic_intensity_naive:<12.1f}x")
    print(f"{'Shared Memory Tiling':<24} {smem_reads*tiles_count/1e9:<18.2f} {ai_smem:<16.1f} {ai_smem/arithmetic_intensity_naive:<12.1f}x")

    print("\n关键洞察:")
    print("  - Shared Memory Tiling 将 Arithmetic Intensity 提升 32x (64×tile/2)")
    print("  - 数据从 Global Memory 读入 Shared Memory 后, 在 Block 内复用")
    print("  - 这就是 GEMM Kernel 高效的核心: 数据复用最大化")


def experiment3_coalesced_access():
    """实验 3: Coalesced Memory Access"""
    print("\n" + "=" * 70)
    print("实验 3: Coalesced Memory Access 分析")
    print("=" * 70)

    gpu = GPUS["A100"]
    bw = gpu.global_bw_gbps

    # 模拟: Warp 内 32 个线程读取 1024 个 float32 元素
    N = 1024 * 1024  # 1M elements
    element_bytes = 4  # float32
    total_bytes = N * element_bytes

    print(f"\n数据量: {N:,} float32 = {total_bytes/1e6:.1f} MB")
    print(f"GPU: {gpu.name}, HBM BW: {bw} GB/s\n")

    access_patterns = [
        ("Coalesced (连续)", 32, 1, " threadIdx.x 连续读取 → 1 transaction"),
        ("Strided-2 (间隔1)", 32, 2, " 每隔1个元素读 → 2x transactions"),
        ("Strided-4 (间隔3)", 32, 4, " 每隔3个元素读 → 4x transactions"),
        ("Strided-32 (跨行)", 32, 32, " 跨行读取 → 32x transactions"),
        ("Random (随机)", 32, 32, " 完全随机 → 最坏情况"),
    ]

    print(f"{'访问模式':<22} {'Strides':<8} {'有效 BW':<14} {'耗时 (ms)':<12} {'说明'}")
    print("-" * 80)

    base_time = total_bytes / (bw * 1e6)  # ms
    for name, warp_size, stride, desc in access_patterns:
        efficiency = 1.0 / stride
        effective_bw = bw * efficiency
        time_ms = total_bytes / (effective_bw * 1e6)
        print(f"{name:<22} {stride:<8} {effective_bw:<14.0f} {time_ms:<12.3f}{desc}")

    print("\n关键洞察:")
    print("  - Coalesced: Warp 内连续 128 字节合并为 1 次 memory transaction")
    print("  - Strided: 间隔读取导致有效带宽按比例下降")
    print("  - Random: 完全随机访问, 有效带宽极低")
    print("  - 实践: 数据布局应与访问模式匹配 (AoS → SoA)")


def experiment4_warp_divergence():
    """实验 4: Warp Divergence 分析"""
    print("\n" + "=" * 70)
    print("实验 4: Warp Divergence 分析")
    print("=" * 70)

    print("\nWarp = 32 个 Thread 执行相同指令 (SIMT)")
    print("当 Thread 走不同分支 → 串行执行, 效率下降\n")

    # 模拟: ReLU Kernel
    scenarios = [
        ("ReLU (50% 分支)", 0.5, "if (x > 0) y = x; else y = 0;"),
        ("Sigmoid (1 条路径)", 0.0, "y = 1/(1+exp(-x));  // 无分支"),
        ("Top-K 选择", 0.8, "if (x > threshold) selected++; // 高度发散"),
        ("MoE Router", 0.75, "if (expert_id == my_expert) process(); // EP"),
    ]

    print(f"{'场景':<20} {'分支比例':<10} {'效率':<10} {'代码模式'}")
    print("-" * 70)
    for name, branch_ratio, code in scenarios:
        # Divergence 效率: warp 中需要执行 max(taken, not_taken) 条路径
        taken = branch_ratio
        not_taken = 1 - branch_ratio
        if taken == 0 or not_taken == 0:
            efficiency = 1.0
        else:
            efficiency = 1.0 / (taken + not_taken)  # 简化模型
        print(f"{name:<20} {branch_ratio:<10.1%} {efficiency:<10.2f} {code}")

    print("\n关键洞察:")
    print("  - 无分支 → 100% 效率 (Sigmoid/SiLU 等激活函数)")
    print("  - 二元分支 → 50% 效率 (两个路径串行执行)")
    print("  - MoE Router: 高度发散, 用 Warp-Level Voting 优化")
    print("  - 解决方案: 重排数据使同分支 Thread 相邻 (Warp Aggregation)")


def experiment5_occupancy_analysis():
    """实验 5: Occupancy 分析"""
    print("\n" + "=" * 70)
    print("实验 5: Kernel Occupancy 分析")
    print("=" * 70)

    gpu = GPUS["A100"]
    print(f"\nGPU: {gpu.name}")
    print(f"SM 数: {gpu.sm_count}, 最大线程/SM: {gpu.max_threads_per_sm}")

    print(f"\n{'Kernel 配置':<30} {'Active B/SM':<12} {'Active T/SM':<14} {'Occupancy':<12} {'瓶颈'}")
    print("-" * 82)

    configs = [
        ("向量加法 (128t, 32reg)", 128, 32, 0),
        ("向量加法 (256t, 32reg)", 256, 32, 0),
        ("矩阵乘法 (128t, 64reg)", 128, 64, 0),
        ("矩阵乘法 (256t, 64reg)", 256, 64, 0),
        ("Conv (128t, 48reg)", 128, 48, 0),
        ("Conv+SMEM (128t, 48reg, 16KB)", 128, 48, 16*1024),
        ("Attention (128t, 40reg, 32KB)", 128, 40, 32*1024),
        ("FlashAttn (256t, 32reg, 64KB)", 256, 32, 64*1024),
        ("FlashAttn (256t, 64reg, 100KB)", 256, 64, 100*1024),
    ]

    for name, threads, regs, smem in configs:
        occ = occupancy(gpu, threads, regs, smem)
        print(f"{name:<30} {occ['active_blocks_per_sm']:<12} "
              f"{occ['active_threads_per_sm']:<14} {occ['occupancy_pct']:<12.1f}% "
              f"{occ['limiting_factor']}")

    print("\n关键洞察:")
    print("  - 100% Occupancy 不一定最优: 高占用可能增加 register spill")
    print("  - Memory-bound: 高 occupancy 隐藏延迟 (更多 warp 切换)")
    print("  - Compute-bound: 低 occupancy 可能更好 (更多寄存器/SMEM per thread)")
    print("  - Flash Attention: 用 64KB SMEM → occupancy 降低, 但 IO 大幅减少")


def experiment6_roofline():
    """实验 6: Roofline Model"""
    print("\n" + "=" * 70)
    print("实验 6: Roofline Model 分析")
    print("=" * 70)

    gpu = GPUS["A100"]
    peak_flops = gpu.fp16_tflops * 1e12  # FLOPS
    peak_bw = gpu.global_bw_gbps * 1e9    # bytes/s

    # Ridge point: AI where compute meets memory
    ridge_point = peak_flops / peak_bw  # ops/byte

    print(f"\nGPU: {gpu.name}")
    print(f"Peak FP16: {gpu.fp16_tflops} TFLOPS")
    print(f"Peak BW: {gpu.global_bw_gbps} GB/s")
    print(f"Ridge Point: {ridge_point:.1f} ops/byte (AI > 此值 → compute-bound)")

    # 分析不同 kernel
    kernels = [
        ("Vector Add (elem-wise)", 1, 12),        # 1 op per element, 12 bytes read+write
        ("Vector Mul (elem-wise)", 1, 12),
        ("ReLU", 1, 8),                            # 1 op, 8 bytes (read 4 + write 4)
        ("Softmax (per row)", 6, 8),                # ~6 ops per element (exp + sum + div)
        ("LayerNorm", 10, 12),                      # mean + var + norm + scale + shift
        ("GEMM (M=N=K=128)", 128, 8),               # 2K ops per output, 2K+1 elements read
        ("GEMM (M=N=K=1024)", 1024, 8),
        ("GEMM (M=N=K=4096)", 4096, 8),
        ("Flash Attn (seq=1K)", 8, 24),             # attention ops, QKV+O bytes
        ("Flash Attn (seq=4K)", 32, 24),
        ("Decode Step (7B)", 1, 13e9/13e9),         # weight read dominant
    ]

    print(f"\n{'Kernel':<28} {'AI (ops/byte)':<16} {'Peak %':<10} {'Bound':<12} {'Perf (TFLOPS)'}")
    print("-" * 80)

    for name, ops, bytes_accessed in kernels:
        ai = ops / bytes_accessed if bytes_accessed > 0 else float('inf')

        if ai >= ridge_point:
            # Compute-bound: achieve peak FLOPS
            bound = "compute"
            perf_tflops = gpu.fp16_tflops
            peak_pct = 100.0
        else:
            # Memory-bound: limited by BW
            bound = "memory"
            perf_tflops = ai * peak_bw / 1e12
            peak_pct = perf_tflops / gpu.fp16_tflops * 100

        print(f"{name:<28} {ai:<16.1f} {peak_pct:<10.1f}% {bound:<12} {perf_tflops:<.1f}")

    print(f"\n关键洞察:")
    print(f"  - Ridge Point = {ridge_point:.1f} ops/byte: AI 超过此值才是 compute-bound")
    print("  - Elem-wise ops (ReLU/Softmax): AI < 1, 严重 memory-bound")
    print("  - GEMM: AI = 2K, 小矩阵 memory-bound, 大矩阵 compute-bound")
    print("  - Decode: 权重读主导, AI ≈ 1, 始终 memory-bound")
    print("  - 这解释了为什么推理优化重点是 HBM 带宽而非算力")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA Kernel 执行模拟器")
    print("Grid/Block/Thread + 内存层次 + Roofline 分析")
    print("=" * 70)

    experiment1_grid_block_hierarchy()
    experiment2_memory_hierarchy()
    experiment3_coalesced_access()
    experiment4_warp_divergence()
    experiment5_occupancy_analysis()
    experiment6_roofline()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
CUDA Kernel 优化核心清单:

  1. 执行层次:
     Thread < Warp(32) < Block < Grid
     Warp 是调度基本单位, Block 是同步/共享内存的边界

  2. 内存优化 (最重要):
     Coalesced Access: 连续线程读连续地址
     Shared Memory Tiling: 数据复用, 减少 Global Memory 访问
     Register 优先: 最快的存储, 但数量有限

  3. Occupancy:
     高 Occupancy → 隐藏内存延迟 (memory-bound kernel)
     低 Occupancy → 更多资源 per thread (compute-bound kernel)
     瓶颈: threads / registers / shared memory

  4. Roofline Model:
     AI (Arithmetic Intensity) = ops / bytes
     AI < Ridge Point → memory-bound (优化重点: 减少内存访问)
     AI > Ridge Point → compute-bound (优化重点: 增加计算吞吐)

  5. LLM 推理特殊性:
     Decode 始终 memory-bound (AI ≈ 1)
     Prefill 可能 compute-bound (长序列, 大 batch)
     所以推理优化核心是 HBM 带宽, 不是算力
""")
