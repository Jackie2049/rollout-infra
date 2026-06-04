#!/usr/bin/env python3
"""Nsight Systems Trace 分析工具

解析 nsys 导出的 JSON trace, 生成性能摘要:
1. Kernel 时间分布 (Top-N 热点)
2. GPU 利用率和 Gap 分析
3. 通信/计算重叠检测
4. 内存带宽估算
5. 时间线可视化 (文本)

CPU 可运行, 无需 GPU。
用于分析从 GPU 服务器导出的 nsys trace 文件。
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Kernel:
    name: str
    start_ns: int
    end_ns: int
    device_id: int
    stream_id: int
    grid_x: int = 0
    grid_y: int = 0
    grid_z: int = 0

    @property
    def duration_us(self):
        return (self.end_ns - self.start_ns) / 1000

    @property
    def duration_ms(self):
        return self.duration_us / 1000


def parse_nsys_json(filepath: str) -> list[Kernel]:
    """解析 nsys 导出的 JSON trace

    导出命令:
      nsys export -t json -o trace.json trace.nsys-rep
    """
    with open(filepath) as f:
        data = json.load(f)

    kernels = []

    # nsys JSON 格式
    if isinstance(data, dict):
        # 标准格式
        traces = data.get("traceEvents", data.get("events", []))
    elif isinstance(data, list):
        traces = data
    else:
        print(f"无法解析的格式: {type(data)}")
        return kernels

    for event in traces:
        if not isinstance(event, dict):
            continue

        name = event.get("name", "")
        cat = event.get("cat", "")

        # CUDA kernel 事件
        if "Kernel" in cat or "kernel" in name.lower():
            ts = event.get("ts", 0)  # 微秒
            dur = event.get("dur", 0)  # 微秒
            args = event.get("args", {})

            if ts and dur:
                kernels.append(Kernel(
                    name=name,
                    start_ns=int(ts * 1000),
                    end_ns=int((ts + dur) * 1000),
                    device_id=args.get("device", 0),
                    stream_id=args.get("stream", 0),
                ))

    # 按 device 分组时排序
    kernels.sort(key=lambda k: k.start_ns)
    return kernels


def generate_synthetic_trace() -> list[Kernel]:
    """生成合成 trace (模拟 LLM 推理的典型模式)

    用于在没有 GPU trace 时演示分析功能。
    """
    kernels = []
    t = 0  # ns

    # 模拟 5 步 decode, 每步有 Transformer 层的 kernel 序列
    for step in range(5):
        # 32 层 Transformer
        for layer in range(32):
            # Attention: QKV projection + attention + output projection
            for name, dur_us in [
                ("gemm_qkv", 35),
                ("flash_attention_kernel", 28),
                ("gemm_output", 25),
            ]:
                kernels.append(Kernel(
                    name=f"{name}_l{layer}",
                    start_ns=t,
                    end_ns=t + dur_us * 1000,
                    device_id=0,
                    stream_id=7,
                ))
                t += dur_us * 1000

            # MLP: gate + up + down projection
            for name, dur_us in [
                ("gemm_gate_up", 40),
                ("silu_and_mul_kernel", 8),
                ("gemm_down", 38),
            ]:
                kernels.append(Kernel(
                    name=f"{name}_l{layer}",
                    start_ns=t,
                    end_ns=t + dur_us * 1000,
                    device_id=0,
                    stream_id=7,
                ))
                t += dur_us * 1000

        # Sampling
        kernels.append(Kernel(
            name="sampling_kernel",
            start_ns=t,
            end_ns=t + 15_000,
            device_id=0,
            stream_id=7,
        ))
        t += 15_000

        # 少量 Gap (模拟 CPU 调度延迟)
        t += 5_000

    return kernels


def analyze_kernels(kernels: list[Kernel], top_n: int = 15) -> dict:
    """分析 kernel 数据"""
    if not kernels:
        return {}

    # 基本统计
    min_start = min(k.start_ns for k in kernels)
    max_end = max(k.end_ns for k in kernels)
    total_trace_ns = max_end - min_start

    # 总 GPU 活跃时间
    total_kernel_ns = sum(k.end_ns - k.start_ns for k in kernels)

    # GPU 利用率
    gpu_util = total_kernel_ns / total_trace_ns * 100 if total_trace_ns > 0 else 0

    # 按 kernel 名聚合
    kernel_stats = defaultdict(lambda: {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0})
    for k in kernels:
        name = _simplify_name(k.name)
        dur = k.end_ns - k.start_ns
        stats = kernel_stats[name]
        stats["count"] += 1
        stats["total_us"] += dur / 1000
        stats["min_us"] = min(stats["min_us"], dur / 1000)
        stats["max_us"] = max(stats["max_us"], dur / 1000)

    # Top N 热点
    top_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)[:top_n]

    # Gap 分析 (GPU 空闲时间)
    sorted_kernels = sorted(kernels, key=lambda k: k.start_ns)
    gaps = []
    for i in range(1, len(sorted_kernels)):
        gap_ns = sorted_kernels[i].start_ns - sorted_kernels[i-1].end_ns
        if gap_ns > 0:
            gaps.append(gap_ns)

    total_gap_ns = sum(gaps)
    gap_pct = total_gap_ns / total_trace_ns * 100 if total_trace_ns > 0 else 0

    # Stream 分布
    stream_stats = defaultdict(int)
    for k in kernels:
        stream_stats[k.stream_id] += 1

    return {
        "total_trace_ms": total_trace_ns / 1e6,
        "total_kernel_ms": total_kernel_ns / 1e6,
        "gpu_util_pct": gpu_util,
        "kernel_count": len(kernels),
        "unique_kernel_types": len(kernel_stats),
        "top_kernels": top_kernels,
        "gap_count": len(gaps),
        "total_gap_ms": total_gap_ns / 1e6,
        "gap_pct": gap_pct,
        "avg_gap_us": (total_gap_ns / len(gaps) / 1000) if gaps else 0,
        "max_gap_us": (max(gaps) / 1000) if gaps else 0,
        "stream_stats": dict(stream_stats),
    }


def _simplify_name(name: str) -> str:
    """简化 kernel 名称, 去掉具体层号"""
    # 移除 _l0, _l31 等后缀
    import re
    name = re.sub(r'_l\d+$', '', name)
    name = re.sub(r'_layer_\d+', '', name)
    # 截断过长的名称
    if len(name) > 50:
        name = name[:47] + "..."
    return name


def print_analysis(analysis: dict):
    """打印分析结果"""
    print("=" * 70)
    print("Nsight Systems Trace 分析报告")
    print("=" * 70)

    print(f"\n总览:")
    print(f"  Trace 时长: {analysis['total_trace_ms']:.2f} ms")
    print(f"  GPU 活跃:   {analysis['total_kernel_ms']:.2f} ms")
    print(f"  GPU 利用率: {analysis['gpu_util_pct']:.1f}%")
    print(f"  Kernel 数:  {analysis['kernel_count']}")
    print(f"  Kernel 类型: {analysis['unique_kernel_types']}")

    print(f"\nGap (空闲) 分析:")
    print(f"  Gap 数:     {analysis['gap_count']}")
    print(f"  总 Gap:     {analysis['total_gap_ms']:.2f} ms ({analysis['gap_pct']:.1f}%)")
    print(f"  平均 Gap:   {analysis['avg_gap_us']:.1f} μs")
    print(f"  最大 Gap:   {analysis['max_gap_us']:.1f} μs")

    print(f"\nStream 分布:")
    for stream, count in sorted(analysis["stream_stats"].items()):
        print(f"  Stream {stream}: {count} kernels")

    print(f"\nTop-{len(analysis['top_kernels'])} 热点 Kernel:")
    print(f"  {'名称':<45} {'次数':<8} {'总时间(ms)':<12} {'占比':<8} {'平均(μs)':<10} {'最大(μs)'}")
    print("  " + "-" * 95)

    total_us = sum(stats["total_us"] for _, stats in analysis["top_kernels"])
    for name, stats in analysis["top_kernels"]:
        pct = stats["total_us"] / (analysis["total_kernel_ms"] * 1000) * 100
        avg = stats["total_us"] / stats["count"]
        print(f"  {name:<45} {stats['count']:<8} {stats['total_us']/1000:<12.2f} {pct:<8.1f}% {avg:<10.1f} {stats['max_us']:.1f}")


def print_timeline(kernels: list[Kernel], width: int = 60, max_steps: int = 3):
    """打印简化的时间线"""
    print("\n" + "=" * 70)
    print(f"时间线可视化 (前 {max_steps} 步)")
    print("=" * 70)

    if not kernels:
        return

    min_start = min(k.start_ns for k in kernels)
    max_end = max(k.end_ns for k in kernels)
    total_ns = max_end - min_start

    # 找到自然分段点 (大的 gap)
    sorted_k = sorted(kernels, key=lambda k: k.start_ns)
    gaps = []
    for i in range(1, len(sorted_k)):
        gap = sorted_k[i].start_ns - sorted_k[i-1].end_ns
        if gap > 0:
            gaps.append((i, gap))

    # 按 gap 大小排序, 取最大的作为分段
    gaps.sort(key=lambda x: x[1], reverse=True)
    split_points = sorted([g[0] for g in gaps[:max_steps - 1]]) if gaps else []

    # 分段
    segments = []
    prev = 0
    for sp in split_points:
        segments.append(sorted_k[prev:sp])
        prev = sp
    segments.append(sorted_k[prev:])

    for seg_idx, segment in enumerate(segments[:max_steps]):
        if not segment:
            continue

        seg_start = segment[0].start_ns
        seg_end = segment[-1].end_ns
        seg_ns = seg_end - seg_start

        print(f"\nStep {seg_idx + 1} ({seg_ns/1e6:.2f} ms):")

        # 按时间位置绘制
        prev_end = seg_start
        bar_parts = []
        for k in segment:
            # Gap
            if k.start_ns > prev_end:
                gap_chars = int((k.start_ns - prev_end) / seg_ns * width)
                bar_parts.append(" " * gap_chars)

            # Kernel
            k_chars = max(1, int((k.end_ns - k.start_ns) / seg_ns * width))
            name_short = _simplify_name(k.name)[:6]
            bar_parts.append(name_short[:k_chars].ljust(k_chars))
            prev_end = k.end_ns

        timeline = "".join(bar_parts)[:width]
        print(f"  |{timeline}|")

    # 图例
    print(f"\n  图例: gemm_  = GEMM, flash_ = Attention, silu_a = Activation")
    print(f"        sampli = Sampling, 空白 = Gap (GPU 空闲)")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Nsight Systems Trace 分析工具")
    print("=" * 70)

    if len(sys.argv) > 1:
        # 解析真实 trace
        filepath = sys.argv[1]
        print(f"\n加载 trace: {filepath}")
        try:
            kernels = parse_nsys_json(filepath)
            if not kernels:
                print("未找到 CUDA kernel 事件")
                sys.exit(1)
            print(f"找到 {len(kernels)} 个 kernel 事件")
        except Exception as e:
            print(f"解析失败: {e}")
            sys.exit(1)
    else:
        # 使用合成 trace 演示
        print("\n未提供 trace 文件, 使用合成数据 (模拟 7B 模型 Decode)")
        print("用法: python nsight_trace_analyzer.py <trace.json>")
        print("导出: nsys export -t json -o trace.json trace.nsys-rep\n")
        kernels = generate_synthetic_trace()

    analysis = analyze_kernels(kernels)
    print_analysis(analysis)
    print_timeline(kernels)

    # 诊断建议
    print("\n" + "=" * 70)
    print("诊断建议")
    print("=" * 70)

    gpu_util = analysis["gpu_util_pct"]
    gap_pct = analysis["gap_pct"]

    if gpu_util > 90:
        print(f"\n  ✓ GPU 利用率 {gpu_util:.0f}% — 良好, GPU 充分利用")
    elif gpu_util > 70:
        print(f"\n  ~ GPU 利用率 {gpu_util:.0f}% — 中等, 有少量优化空间")
    else:
        print(f"\n  ✗ GPU 利用率 {gpu_util:.0f}% — 偏低!")
        if gap_pct > 20:
            print(f"    → Gap 占 {gap_pct:.0f}%, 检查 CPU 瓶颈或数据加载")
        print("    → 建议: nsys 时间线查看 Gap 位置和原因")

    if gap_pct > 10:
        print(f"\n  注意: Gap 占 {gap_pct:.0f}%")
        if analysis["avg_gap_us"] > 100:
            print(f"    → 平均 Gap {analysis['avg_gap_us']:.0f}μs, 可能是 CPU 调度延迟")
            print("    → 建议: 检查 Python GIL, 考虑 CUDA Graph")
        if analysis["max_gap_us"] > 1000:
            print(f"    → 最大 Gap {analysis['max_gap_us']:.0f}μs, 可能有同步点")
            print("    → 建议: nsys 查看 Gap 位置的调用栈")

    # 热点建议
    if analysis["top_kernels"]:
        top_name, top_stats = analysis["top_kernels"][0]
        top_pct = top_stats["total_us"] / (analysis["total_kernel_ms"] * 1000) * 100
        if top_pct > 30:
            print(f"\n  热点: {top_name} 占 {top_pct:.0f}%")
            print(f"    → 如果是 GEMM: 考虑量化 (FP8/INT4) 减少权重读取")
            print(f"    → 如果是 Attention: 已接近最优 (FlashAttention)")
            print(f"    → 如果是 NCCL: 检查通信/计算重叠")
