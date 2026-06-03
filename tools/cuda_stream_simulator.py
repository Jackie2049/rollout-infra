#!/usr/bin/env python3
"""CUDA Stream 并发模型模拟器

模拟 GPU 并发执行的关键概念:
1. CUDA Stream 与 Event
2. 计算与通信重叠 (Communication-Compute Overlap)
3. NCCL 与 Tensor 并行的重叠策略
4. vLLM 中的多 Stream 架构

CPU 可运行，无需 GPU。
"""

import math


# ============================================================
# GPU 参数模型
# ============================================================

class GPUConfig:
    def __init__(self, name, sm_count, fp16_tflops, hbm_bw_gbps,
                 pcie_bw_gbps=64, nvlink_bw_gbps=300, kernel_launch_us=5):
        self.name = name
        self.sm_count = sm_count
        self.fp16_tflops = fp16_tflops
        self.hbm_bw_gbps = hbm_bw_gbps
        self.pcie_bw_gbps = pcie_bw_gbps
        self.nvlink_bw_gbps = nvlink_bw_gbps
        self.kernel_launch_us = kernel_launch_us


A100 = GPUConfig("A100-80GB", 108, 312, 2035, 64, 300)
H100 = GPUConfig("H100-80GB", 132, 990, 3350, 128, 450)


def compute_time_ms(flops, gpu):
    """计算时间 (ms)"""
    return flops / (gpu.fp16_tflops * 1e9) * 1000


def mem_time_ms(bytes_gb, bw_gbps):
    """内存传输时间 (ms)"""
    return bytes_gb / bw_gbps * 1000


# ============================================================
# 实验
# ============================================================

def experiment1_stream_basics():
    """实验 1: CUDA Stream 基础"""
    print("=" * 70)
    print("实验 1: CUDA Stream 并发模型")
    print("=" * 70)

    print("""
CUDA Stream: GPU 命令队列, 同一 Stream 内命令串行执行, 不同 Stream 并行执行

Stream 0 (Default):
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Kernel A │→ │ Kernel B │→ │ Kernel C │  ← 串行
  └──────────┘  └──────────┘  └──────────┘

Stream 1 + Stream 2 (并行):
  Stream 1: ┌──────────┐  ┌──────────┐
            │ Kernel A │→ │ Kernel C │     ┐
            └──────────┘  └──────────┘     │ 同时执行
  Stream 2: ┌──────────┐  ┌──────────┐     │
            │ Kernel B │→ │ Kernel D │     ┘
            └──────────┘  └──────────┘
""")

    # 模拟: 独立 kernel 在不同 stream 上的并行效果
    gpu = A100

    # 4 个独立的矩阵乘法 kernel
    M = N = K = 2048
    flops = 2 * M * N * K  # FMA = 2 ops
    single_kernel_ms = compute_time_ms(flops, gpu)
    kernel_launch_ms = gpu.kernel_launch_us / 1000

    print(f"GPU: {gpu.name}")
    print(f"矩阵: {M}×{K} × {K}×{N}")
    print(f"单个 kernel: {single_kernel_ms:.3f} ms")
    print(f"Kernel launch 开销: {kernel_launch_ms:.3f} ms\n")

    configs = [
        ("单 Stream (串行)", 1, False),
        ("双 Stream (2 并行)", 2, False),
        ("四 Stream (4 并行)", 4, False),
        ("四 Stream + Overlap", 4, True),
    ]

    print(f"{'配置':<28} {'总时间 (ms)':<14} {'加速比':<10} {'说明'}")
    print("-" * 66)

    for name, streams, overlap in configs:
        if streams == 1:
            total = single_kernel_ms * 4 + kernel_launch_ms * 4
            speedup = 1.0
        else:
            # 每个 stream 负责的 kernel 数
            kernels_per_stream = math.ceil(4 / streams)
            # 并行执行: 时间 = max stream 时间
            launch_overhead = kernel_launch_ms * 4
            if overlap:
                # Overlap: launch 开销被 kernel 计算隐藏
                launch_overhead = 0
            total = single_kernel_ms * kernels_per_stream + launch_overhead
            baseline = single_kernel_ms * 4 + kernel_launch_ms * 4
            speedup = baseline / total

        print(f"{name:<28} {total:<14.3f} {speedup:<10.2f}x", end="")
        if overlap:
            print(" launch 开销被计算隐藏")
        elif streams == 1:
            print(" 无并行")
        else:
            print(f" {streams} kernel 并行")

    print("\n关键洞察:")
    print("  - 多 Stream 允许独立 kernel 并行执行")
    print("  - CUDA_DEVICE_MAX_CONNECTIONS=1 可强制串行化 (有利于 TP overlap)")
    print("  - Stream 间同步通过 Event: record → wait")


def experiment2_comm_compute_overlap():
    """实验 2: 通信与计算重叠"""
    print("\n" + "=" * 70)
    print("实验 2: 通信与计算重叠 (AllReduce + 计算)")
    print("=" * 70)

    gpu = H100

    # 模拟: Transformer 层的 TP 执行
    # 每层: ColumnParallel → AllReduce → RowParallel → AllReduce
    layer_flops = 2 * 8192 * 8192 * 80  # 简化: 80 layers 的 1/80
    comm_bytes_gb = 8192 * 8192 * 2 / 1e9  # FP16 activation

    comp_ms = compute_time_ms(layer_flops, gpu)
    comm_ms_nvlink = mem_time_ms(comm_bytes_gb, gpu.nvlink_bw_gbps)
    comm_ms_pcie = mem_time_ms(comm_bytes_gb, gpu.pcie_bw_gbps)

    print(f"\nGPU: {gpu.name}")
    print(f"层计算: {comp_ms:.3f} ms")
    print(f"通信 (NVLink): {comm_ms_nvlink:.3f} ms")
    print(f"通信 (PCIe): {comm_ms_pcie:.3f} ms\n")

    scenarios = [
        ("无重叠 (串行)", False, "nvlink"),
        ("NVLink 重叠", True, "nvlink"),
        ("PCIe 无重叠", False, "pcie"),
        ("PCIe 重叠", True, "pcie"),
    ]

    print(f"{'场景':<22} {'计算':<10} {'通信':<10} {'总时间':<10} {'通信占比':<10} {'效率'}")
    print("-" * 72)

    for name, overlap, link in scenarios:
        comm = comm_ms_nvlink if link == "nvlink" else comm_ms_pcie
        if overlap:
            total = max(comp_ms, comm)  # overlap: 取最大
            comm_pct = comm / total * 100 if comm > comp_ms else 0
            efficiency = comp_ms / total * 100
        else:
            total = comp_ms + comm  # 串行: 相加
            comm_pct = comm / total * 100
            efficiency = comp_ms / total * 100

        print(f"{name:<22} {comp_ms:<10.3f} {comm:<10.3f} {total:<10.3f} {comm_pct:<10.1f}% {efficiency:.1f}%")

    print("\n关键洞察:")
    print("  - NVLink 重叠: 通信被计算完全隐藏 (通信 < 计算)")
    print("  - PCIe 重叠: 通信大于计算, 即使重叠仍有开销")
    print("  - 这解释了为什么 TP 需要 NVLink: 通信快到可以被计算覆盖")


def experiment3_tp_overlap_patterns():
    """实验 3: TP 层间通信重叠策略"""
    print("\n" + "=" * 70)
    print("实验 3: Tensor Parallel 通信重叠策略")
    print("=" * 70)

    gpu = H100

    print("""
Megatron-LM TP 通信模式:
  每层 Transformer = ColumnParallel + RowParallel

  策略 1: 无重叠 (简单)
    GPU 0: [Comp]→[AllReduce]→[Comp]→[AllReduce]
    总时间 = 2×Comp + 2×Comm

  策略 2: 1F1B 重叠 (Megatron)
    GPU 0: [Comp_F]→[AllReduce ‖ Comp_B]→[AllReduce]
    总时间 = 2×Comp + 1×Comm (一半通信被隐藏)

  策略 3: CUDA_DEVICE_MAX_CONNECTIONS=1 (强制串行 stream)
    确保通信和计算在同一 stream, 方便 overlap
""")

    # 7B 模型 TP=2 的每层参数
    hidden = 4096
    layers = 32
    seq_len = 2048
    batch = 8

    # 每层计算: QKV + Attn + MLP
    layer_flops = 2 * batch * seq_len * hidden * hidden * 4 * 2  # 粗略
    layer_comp_ms = compute_time_ms(layer_flops, gpu)

    # AllReduce 通信量: activation size
    activation_gb = batch * seq_len * hidden * 2 / 1e9  # FP16
    allreduce_ms = mem_time_ms(activation_gb, gpu.nvlink_bw_gbps) * 2  # Ring: 2x data transfer

    print(f"模型: 7B, TP=2, Batch={batch}, Seq={seq_len}")
    print(f"每层计算: {layer_comp_ms:.3f} ms")
    print(f"AllReduce (NVLink): {allreduce_ms:.3f} ms\n")

    # 无重叠
    no_overlap = layers * (2 * layer_comp_ms + 2 * allreduce_ms)

    # 1F1B 重叠: 每层一次通信被计算覆盖
    overlap = layers * (2 * layer_comp_ms + allreduce_ms)

    # 完全重叠 (理论最优)
    perfect = layers * (2 * layer_comp_ms)

    print(f"{'策略':<24} {'每层 (ms)':<14} {'总 (ms)':<14} {'通信占比':<12} {'加速比'}")
    print("-" * 66)
    print(f"{'无重叠':<24} {2*layer_comp_ms+2*allreduce_ms:<14.3f} {no_overlap:<14.1f} {2*allreduce_ms/(2*layer_comp_ms+2*allreduce_ms)*100:<12.1f}% 1.00x")
    print(f"{'1F1B 重叠':<24} {2*layer_comp_ms+allreduce_ms:<14.3f} {overlap:<14.1f} {allreduce_ms/(2*layer_comp_ms+allreduce_ms)*100:<12.1f}% {no_overlap/overlap:.2f}x")
    print(f"{'完全重叠 (理论)':<24} {2*layer_comp_ms:<14.3f} {perfect:<14.1f} {'0.0':<12}% {no_overlap/perfect:.2f}x")

    print("\n关键洞察:")
    print("  - 1F1B 重叠可消除约一半的通信开销")
    print("  - NVLink 下 AllReduce 通信占比 ~11%, 重叠后 ~6%")
    print("  - CUDA_DEVICE_MAX_CONNECTIONS=1 强制单 stream, 确保通信不被打散")


def experiment4_vllm_stream_architecture():
    """实验 4: vLLM 中的多 Stream 架构"""
    print("\n" + "=" * 70)
    print("实验 4: vLLM V1 多 Stream 架构")
    print("=" * 70)

    print("""
vLLM V1 使用多个 CUDA Stream 实现并发:

1. Model Stream (主计算流):
   执行 Transformer 前向传播

2. KV Cache Stream:
   异步拷贝 KV cache (CPU↔GPU, GPU↔GPU)

3. Sampling Stream:
   采样操作可以与下一 batch 的 prefill 重叠

4. CUDA Graph:
   Decode 阶段将整个前向传播录制成 CUDA Graph
   避免 kernel launch 开销 (0.5ms → 0.005ms)
""")

    gpu = A100

    # Decode step 时间分解
    decode_flops = 2 * 7e9 * 1  # 7B model, 1 token per step
    weight_bytes = 13e9 / 1e9  # 13GB FP16
    kv_bytes_per_layer = 2 * 32 * 8 * 128 * 2  # 7B model
    total_kv_bytes = kv_bytes_per_layer * 32 / 1e9  # all layers

    weight_read_ms = mem_time_ms(weight_bytes, gpu.hbm_bw_gbps)
    kernel_launch_ms = gpu.kernel_launch_us / 1000 * 32 * 4  # 32 layers, ~4 kernels each

    print(f"7B Decode Step 分解 (A100):")
    print(f"  权重读取: {weight_read_ms:.3f} ms")
    print(f"  Kernel launch (无 Graph): {kernel_launch_ms:.3f} ms")
    print(f"  Kernel launch (有 Graph): {kernel_launch_ms * 0.01:.3f} ms")
    print(f"  KV Cache 读取: {total_kv_bytes / gpu.hbm_bw_gbps * 1000:.3f} ms")

    no_graph = weight_read_ms + kernel_launch_ms
    with_graph = weight_read_ms + kernel_launch_ms * 0.01

    print(f"\n  无 CUDA Graph: {no_graph:.3f} ms/step")
    print(f"  有 CUDA Graph: {with_graph:.3f} ms/step")
    print(f"  节省: {(1 - with_graph/no_graph)*100:.0f}% (kernel launch 开销)")

    print("\n关键洞察:")
    print("  - CUDA Graph 将 kernel launch 开销降低 ~100x")
    print("  - Decode 阶段每步都相同 (shape 不变), 非常适合 Graph 捕获")
    print("  - Prefill 阶段 shape 变化大, 不适合 CUDA Graph")
    print("  - 多 Stream 允许 KV Cache 操作与计算并行")


def experiment5_async_operations():
    """实验 5: 异步操作与同步机制"""
    print("\n" + "=" * 70)
    print("实验 5: 异步操作与同步机制")
    print("=" * 70)

    print("""
CUDA 异步模型:
  - Kernel launch 是异步的 (CPU 提交后立即返回)
  - cudaMemcpy 是同步的 (阻塞 CPU 直到完成)
  - cudaMemcpyAsync + Stream 可以实现异步传输

同步原语:
  1. cudaStreamSynchronize(stream) — 等待 stream 上所有操作完成
  2. cudaEventRecord(event, stream) — 在 stream 中标记事件
  3. cudaStreamWaitEvent(stream, event) — stream 等待事件完成
  4. cudaMemcpyAsync + stream — 异步内存拷贝
""")

    # 模拟: 异步数据加载 + 计算 overlap
    gpu = A100

    data_size_gb = 10  # 10GB 数据
    compute_flops = 1e12  # 1 TFLOPS 计算

    h2d_ms = mem_time_ms(data_size_gb, gpu.pcie_bw_gbps)  # Host to Device
    comp_ms = compute_time_ms(compute_flops, gpu)

    print(f"数据传输 (PCIe): {h2d_ms:.1f} ms ({data_size_gb} GB)")
    print(f"计算: {comp_ms:.3f} ms\n")

    print(f"{'策略':<28} {'传输':<10} {'计算':<10} {'总时间':<10} {'效率'}")
    print("-" * 68)

    # 同步: 先传完再算
    sync_total = h2d_ms + comp_ms
    print(f"{'同步 (先传后算)':<28} {h2d_ms:<10.1f} {comp_ms:<10.3f} {sync_total:<10.1f} {comp_ms/sync_total*100:.1f}%")

    # 异步 + overlap
    overlap_total = max(h2d_ms, comp_ms)
    print(f"{'异步重叠':<28} {h2d_ms:<10.1f} {comp_ms:<10.3f} {overlap_total:<10.1f} {comp_ms/overlap_total*100:.1f}%")

    # 分块 overlap
    chunks = 10
    chunk_h2d = h2d_ms / chunks
    chunk_comp = comp_ms / chunks
    pipelined = chunk_h2d + (chunks - 1) * max(chunk_h2d, chunk_comp) + chunk_comp
    print(f"{'Pipeline (10 chunks)':<28} {h2d_ms:<10.1f} {comp_ms:<10.3f} {pipelined:<10.1f} {comp_ms/pipelined*100:.1f}%")

    print("\n关键洞察:")
    print("  - 同步模式: 传输和计算串行, 总时间 = 传输 + 计算")
    print("  - 异步 overlap: 传输和计算并行, 总时间 = max(传输, 计算)")
    print("  - Pipeline: 分块 overlap, 接近理论最优")
    print("  - 实践: 预取 (prefetch) 是最常用的 overlap 模式")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA Stream 并发模型模拟器")
    print("Stream/Event + 通信计算重叠 + vLLM 多 Stream")
    print("=" * 70)

    experiment1_stream_basics()
    experiment2_comm_compute_overlap()
    experiment3_tp_overlap_patterns()
    experiment4_vllm_stream_architecture()
    experiment5_async_operations()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
CUDA Stream 并发核心:

  1. Stream 模型:
     - 同一 Stream 串行, 不同 Stream 并行
     - Event 实现跨 Stream 同步
     - CUDA_DEVICE_MAX_CONNECTIONS 控制并发度

  2. 通信计算重叠:
     - 1F1B: Megatron TP 的核心优化, 隐藏 ~50% 通信
     - Pipeline: 分块传输+计算, 接近理论最优
     - 前提: NVLink 带宽 > 计算速度

  3. vLLV 应用:
     - CUDA Graph: kernel launch 开销降低 100x
     - 多 Stream: KV Cache 拷贝与计算并行
     - Prefill/Decode 分离: 计算和传输在不同 GPU 上并行

  4. 最佳实践:
     - TP 需要 NVLink (通信 < 计算, 可以 overlap)
     - Decode 用 CUDA Graph (shape 不变)
     - Prefill 不用 Graph (shape 变化大)
     - Pipeline 分块是最通用的 overlap 模式
""")
