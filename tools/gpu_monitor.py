#!/usr/bin/env python3
"""GPU Monitoring Script — 训练过程中的实时 GPU 监控

用法:
  # 监控所有 GPU (每秒刷新)
  python gpu_monitor.py

  # 每 2 秒刷新，只看 GPU 0,1
  python gpu_monitor.py --interval 2 --gpus 0,1

  # 记录到 CSV 文件 (用于后续分析)
  python gpu_monitor.py --log gpu_log.csv

  # 查看 GPU 详细信息 (一次性)
  python gpu_monitor.py --info
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUStats:
    gpu_id: int
    gpu_util: float       # GPU 计算利用率 %
    mem_util: float       # 显存利用率 %
    mem_used_mb: int      # 已用显存 MB
    mem_total_mb: int     # 总显存 MB
    temperature: int      # 温度 °C
    power_draw: float     # 功耗 W
    power_limit: float    # 功耗上限 W
    clock_sm: int         # SM 频率 MHz
    clock_mem: int        # 显存频率 MHz


def get_gpu_stats(gpu_ids: Optional[list[int]] = None) -> list[GPUStats]:
    """通过 nvidia-smi 获取 GPU 统计信息"""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
            "clocks.current.sm,clocks.current.mem",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Is NVIDIA driver installed?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr}")
        sys.exit(1)

    stats = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue

        s = GPUStats(
            gpu_id=int(parts[0]),
            gpu_util=float(parts[1]) if parts[1] != "[N/A]" else 0.0,
            mem_util=float(parts[2]) if parts[2] != "[N/A]" else 0.0,
            mem_used_mb=int(float(parts[3])) if parts[3] != "[N/A]" else 0,
            mem_total_mb=int(float(parts[4])) if parts[4] != "[N/A]" else 0,
            temperature=int(parts[5]) if parts[5] != "[N/A]" else 0,
            power_draw=float(parts[6]) if parts[6] != "[N/A]" else 0.0,
            power_limit=float(parts[7]) if parts[7] != "[N/A]" else 0.0,
            clock_sm=int(parts[8]) if parts[8] != "[N/A]" else 0,
            clock_mem=int(parts[9]) if parts[9] != "[N/A]" else 0,
        )

        if gpu_ids is None or s.gpu_id in gpu_ids:
            stats.append(s)

    return stats


def get_gpu_processes() -> dict[int, list[str]]:
    """获取每个 GPU 上的进程信息"""
    try:
        cmd = [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}

        procs: dict[int, list[str]] = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_uuid = parts[0]
                procs.setdefault(0, []).append(
                    f"  PID {parts[1]}: {parts[2]} ({parts[3]})"
                )
        return procs
    except Exception:
        return {}


def print_gpu_info():
    """打印 GPU 详细信息 (一次性)"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu="
             "index,name,driver_version,memory.total,"
             "compute_cap,persistence_mode,power.limit",
             "--format=csv"],
            capture_output=True, text=True, check=True,
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error: {e}")


def format_bar(value: float, max_val: float, width: int = 20) -> str:
    """生成进度条"""
    if max_val == 0:
        return "[" + " " * width + "]"
    ratio = min(value / max_val, 1.0)
    filled = int(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {ratio*100:.0f}%"


def print_stats(stats: list[GPUStats], verbose: bool = False):
    """打印 GPU 统计信息"""
    # Header
    print(
        f"\033[2J\033[H"  # Clear screen and move cursor to top
        f"{'GPU Monitor':^80}\n"
        f"{'=' * 80}\n"
        f"{'Time':} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'-' * 80}"
    )

    for s in stats:
        mem_bar = format_bar(s.mem_used_mb, s.mem_total_mb)
        gpu_bar = format_bar(s.gpu_util, 100)
        power_bar = format_bar(s.power_draw, s.power_limit) if s.power_limit > 0 else "N/A"

        print(
            f"\n  GPU {s.gpu_id}: {s.temperature}°C | "
            f"SM {s.clock_sm}MHz | Mem {s.clock_mem}MHz\n"
            f"  GPU Util:  {gpu_bar}\n"
            f"  VRAM:      {mem_bar} "
            f"({s.mem_used_mb}/{s.mem_total_mb} MB)\n"
            f"  Power:     {s.power_draw:.0f}/{s.power_limit:.0f} W "
            f"({power_bar})"
        )

    if verbose:
        procs = get_gpu_processes()
        if procs:
            print(f"\n{'-' * 80}")
            print("Processes:")
            for gpu_id, proc_list in procs.items():
                for p in proc_list:
                    print(p)

    # Summary
    if len(stats) > 1:
        total_mem = sum(s.mem_used_mb for s in stats)
        total_cap = sum(s.mem_total_mb for s in stats)
        avg_util = sum(s.gpu_util for s in stats) / len(stats)
        total_power = sum(s.power_draw for s in stats)
        print(
            f"\n{'-' * 80}\n"
            f"  Summary: {len(stats)} GPUs | "
            f"AVG Util: {avg_util:.1f}% | "
            f"Total VRAM: {total_mem}/{total_cap} MB | "
            f"Total Power: {total_power:.0f}W"
        )


def log_to_csv(stats: list[GPUStats], csvfile: str):
    """记录到 CSV 文件"""
    import csv
    import os

    write_header = not os.path.exists(csvfile)

    with open(csvfile, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "gpu_id", "gpu_util", "mem_util",
                "mem_used_mb", "mem_total_mb", "temperature",
                "power_draw", "power_limit", "clock_sm", "clock_mem",
            ])
        for s in stats:
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                s.gpu_id, s.gpu_util, s.mem_util,
                s.mem_used_mb, s.mem_total_mb, s.temperature,
                s.power_draw, s.power_limit, s.clock_sm, s.clock_mem,
            ])


def analyze_log(csvfile: str):
    """分析 CSV 日志"""
    import csv

    rows: dict[int, list] = {}
    try:
        with open(csvfile) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gpu_id = int(row["gpu_id"])
                rows.setdefault(gpu_id, []).append(row)
    except FileNotFoundError:
        print(f"File not found: {csvfile}")
        return

    print(f"\n{'=' * 60}")
    print(f"GPU Log Analysis: {csvfile}")
    print(f"{'=' * 60}")

    for gpu_id, entries in sorted(rows.items()):
        utils = [float(e["gpu_util"]) for e in entries]
        mems = [int(e["mem_used_mb"]) for e in entries]
        powers = [float(e["power_draw"]) for e in entries]
        temps = [int(e["temperature"]) for e in entries]

        n = len(entries)
        print(f"\n  GPU {gpu_id} ({n} samples):")
        print(f"    GPU Util:  avg={sum(utils)/n:.1f}%  "
              f"min={min(utils):.0f}%  max={max(utils):.0f}%")
        print(f"    VRAM Used: avg={sum(mems)/n:.0f}MB  "
              f"min={min(mems)}MB  max={max(mems)}MB")
        print(f"    Power:     avg={sum(powers)/n:.0f}W  "
              f"min={min(powers):.0f}W  max={max(powers):.0f}W")
        print(f"    Temp:      avg={sum(temps)/n:.0f}°C  "
              f"min={min(temps)}°C  max={max(temps)}°C")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitor")
    parser.add_argument("--interval", "-i", type=int, default=1,
                        help="刷新间隔 (秒)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="指定 GPU ID (逗号分隔, 如 0,1)")
    parser.add_argument("--log", type=str, default=None,
                        help="记录到 CSV 文件")
    parser.add_argument("--info", action="store_true",
                        help="显示 GPU 详细信息 (一次性)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="分析 CSV 日志文件")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示进程信息")
    args = parser.parse_args()

    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]

    if args.info:
        print_gpu_info()
        return

    if args.analyze:
        analyze_log(args.analyze)
        return

    print(f"Monitoring GPUs (interval={args.interval}s, Ctrl+C to stop)")
    try:
        while True:
            stats = get_gpu_stats(gpu_ids)
            if not stats:
                print("No GPUs found.")
                break
            print_stats(stats, verbose=args.verbose)
            if args.log:
                log_to_csv(stats, args.log)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if args.log:
            print(f"Log saved to: {args.log}")
            analyze_log(args.log)


if __name__ == "__main__":
    main()
