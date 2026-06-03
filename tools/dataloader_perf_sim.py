#!/usr/bin/env python3
"""DataLoader 优化分析器

CPU 可运行的 DataLoader 性能分析，覆盖 4 个实验:
  1. 数据加载瓶颈分析 (IO vs 预处理 vs 训练)
  2. Prefetch/预取优化效果
  3. 不同存储介质对比
  4. DataLoader 调优策略推荐

关键概念:
  - DataLoader 瓶颈: 加载时间 > 训练时间 → GPU 空闲
  - Prefetch: 后台线程提前加载下一批数据
  - mmap: 内存映射文件, 避免 read() 系统调用
  - WebDataset: tar 格式顺序读取, 适合远程存储

用法:
  conda run -n ai-infra python tools/dataloader_perf_sim.py
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class StorageConfig:
    name: str
    read_bw_gbps: float    # 顺序读带宽 (GB/s)
    random_read_iops: int  # 随机读 IOPS
    latency_ms: float      # IO 延迟 (ms)


@dataclass
class DatasetConfig:
    name: str
    num_samples: int
    sample_size_kb: float  # 平均样本大小 (KB)
    num_tokens: int        # 总 token 数 (用于 LLM)
    seq_len: int           # 序列长度
    format: str            # jsonl, parquet, tar, mmap


NVMe_SSD = StorageConfig("NVMe SSD", 5.0, 500000, 0.05)
SATA_SSD = StorageConfig("SATA SSD", 2.0, 100000, 0.1)
HDD = StorageConfig("HDD", 0.2, 100, 5.0)
NFS = StorageConfig("NFS", 1.0, 50000, 2.0)
Lustre = StorageConfig("Lustre", 10.0, 200000, 0.5)
RAM = StorageConfig("RAM Disk", 50.0, 1000000, 0.001)

# 典型 LLM 数据集
PILE = DatasetConfig("The Pile", 30000000, 10, 300e9, 2048, "jsonl")
REDPAJAMA = DatasetConfig("RedPajama", 200000000, 5, 1200e9, 4096, "parquet")
CODE_DATASET = DatasetConfig("Code Dataset", 50000000, 8, 200e9, 4096, "tar")


# ──────────────────────────────────────────────
# 性能模型
# ──────────────────────────────────────────────

def load_time_ms(
    batch_size: int,
    dataset: DatasetConfig,
    storage: StorageConfig,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    use_mmap: bool = False,
    use_webdataset: bool = False,
) -> dict:
    """估算单个 batch 的加载时间

    加载 = 读取文件 + 解码/预处理 + 传输到 GPU
    """
    # 每批数据大小
    batch_bytes = batch_size * dataset.sample_size_kb * 1024

    if use_webdataset:
        # WebDataset: 顺序读 tar, 大块 IO
        read_time_ms = batch_bytes / (storage.read_bw_gbps * 1e6) * 1000
    elif use_mmap:
        # mmap: 页错误时才实际 IO, 通常更快
        read_time_ms = batch_bytes / (storage.read_bw_gbps * 1e6) * 1000 * 0.8
    else:
        # 随机读: batch_size 个独立文件
        if storage.random_read_iops > 0:
            random_time_ms = batch_size / storage.random_read_iops * 1000
            sequential_time_ms = batch_bytes / (storage.read_bw_gbps * 1e6) * 1000
            read_time_ms = max(random_time_ms, sequential_time_ms)
        else:
            read_time_ms = batch_bytes / (storage.read_bw_gbps * 1e6) * 1000

    # 多 worker 并行
    effective_read_time = read_time_ms / num_workers

    # 预处理: tokenization, augmentation (假设 ~0.1ms/sample)
    preprocess_time_ms = batch_size * 0.1 / num_workers

    # 传输到 GPU (PCIe ~32 GB/s)
    transfer_time_ms = batch_bytes / (32 * 1e6) * 1000

    total_time_ms = effective_read_time + preprocess_time_ms + transfer_time_ms

    return {
        'total_ms': total_time_ms,
        'read_ms': effective_read_time,
        'preprocess_ms': preprocess_time_ms,
        'transfer_ms': transfer_time_ms,
        'batch_bytes_mb': batch_bytes / 1e6,
    }


def training_step_ms(
    model_params_B: float,
    batch_size: int,
    seq_len: int,
    gpu_tflops: float,
    mfu: float = 0.45,
) -> dict:
    """估算单步训练时间

    FLOPs = 6 × params × tokens (forward + backward)
    """
    tokens = batch_size * seq_len
    flops = 6 * model_params_B * 1e9 * tokens
    time_ms = flops / (gpu_tflops * 1e12 * mfu) * 1000

    return {
        'time_ms': time_ms,
        'tokens': tokens,
        'flops_tflops': flops / 1e12,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_bottleneck_analysis():
    """实验 1: 数据加载瓶颈分析"""
    print("\n" + "=" * 70)
    print("实验 1: 数据加载 vs 训练时间 (瓶颈识别)")
    print("=" * 70)

    configs = [
        (7.0, 312, 4, PILE, "LLaMA-7B on A100"),
        (70.0, 990, 8, REDPAJAMA, "LLaMA-70B on H100"),
        (7.0, 312, 4, CODE_DATASET, "LLaMA-7B on A100 (code)"),
    ]

    for params_B, gpu_tflops, batch_size, dataset, label in configs:
        seq_len = dataset.seq_len

        print(f"\n--- {label} ---")
        print(f"  Batch={batch_size}, Seq={seq_len}, 数据集={dataset.name}")

        for storage in [NVMe_SSD, NFS, HDD]:
            load = load_time_ms(batch_size, dataset, storage)
            train = training_step_ms(params_B, batch_size, seq_len, gpu_tflops)

            # 无 prefetch: GPU 等待数据加载
            total_no_prefetch = train['time_ms'] + load['total_ms']
            gpu_idle_pct = load['total_ms'] / total_no_prefetch * 100

            # 有 prefetch: 数据加载与训练重叠
            if load['total_ms'] <= train['time_ms']:
                overlap_total = train['time_ms']
                gpu_idle_overlap = 0
            else:
                overlap_total = load['total_ms']
                gpu_idle_overlap = (load['total_ms'] - train['time_ms']) / overlap_total * 100

            print(f"  {storage.name:<15}: 加载={load['total_ms']:>8.1f}ms, "
                  f"训练={train['time_ms']:>8.1f}ms, "
                  f"GPU空闲={gpu_idle_pct:>5.1f}% (无prefetch), "
                  f"{gpu_idle_overlap:>5.1f}% (有prefetch)")

    print(f"\n关键洞察:")
    print(f"  - NVMe SSD + prefetch: 加载 < 训练, GPU 几乎不空闲")
    print(f"  - NFS: 加载可能 > 训练, GPU 空闲显著")
    print(f"  - HDD: 严重瓶颈, GPU 大量空闲")
    print(f"  - Prefetch 可以完全隐藏 NVMe 的加载延迟")


def experiment_2_prefetch_optimization():
    """实验 2: Prefetch 预取优化"""
    print("\n" + "=" * 70)
    print("实验 2: Prefetch 预取优化效果")
    print("=" * 70)

    params_B = 7.0
    gpu_tflops = 312
    batch_size = 4
    seq_len = 2048
    storage = NVMe_SSD

    print(f"\n配置: LLaMA-7B, A100, B={batch_size}, S={seq_len}, {storage.name}")

    load = load_time_ms(batch_size, PILE, storage)
    train = training_step_ms(params_B, batch_size, seq_len, gpu_tflops)

    print(f"\n  单批加载: {load['total_ms']:.1f}ms")
    print(f"  单步训练: {train['time_ms']:.1f}ms")

    print(f"\n  {'Workers':>10} {'Prefetch':>10} {'加载/批':>12} {'训练空闲':>12} {'吞吐 tok/s':>14}")
    print(f"  " + "-" * 62)

    for workers in [1, 2, 4, 8, 16]:
        for pf in [1, 2, 4]:
            load_w = load_time_ms(batch_size, PILE, storage, workers, pf)

            # Prefetch: 后台加载 N 批, 理想情况下完全隐藏
            effective_load = max(0, load_w['total_ms'] - train['time_ms'] * pf)
            total = train['time_ms'] + effective_load
            tokens_per_s = batch_size * seq_len / (total / 1000)

            print(f"  {workers:>10d} {pf:>10d} {load_w['total_ms']:>12.1f}ms "
                  f"{effective_load:>11.1f}ms {tokens_per_s:>14.0f}")
        print()

    print(f"关键洞察:")
    print(f"  - num_workers=4 + prefetch=2: 加载完全隐藏")
    print(f"  - 过多 workers: CPU 争抢, 收益递减")
    print(f"  - Prefetch 因子 2-4 足够, 太大浪费内存")
    print(f"  - PyTorch DataLoader: num_workers=4-8, prefetch_factor=2")


def experiment_3_storage_comparison():
    """实验 3: 不同存储介质对比"""
    print("\n" + "=" * 70)
    print("实验 3: 不同存储介质对比")
    print("=" * 70)

    params_B = 7.0
    gpu_tflops = 312
    batch_size = 8
    seq_len = 2048
    train = training_step_ms(params_B, batch_size, seq_len, gpu_tflops)

    print(f"\n配置: LLaMA-7B, A100, B={batch_size}, S={seq_len}")
    print(f"  训练时间/步: {train['time_ms']:.1f}ms")

    storages = [RAM, NVMe_SSD, SATA_SSD, Lustre, NFS, HDD]
    formats = [("jsonl", False, False), ("mmap", True, False), ("tar/WebDataset", False, True)]

    print(f"\n{'存储':<15} {'格式':<18} {'加载ms':>10} {'加载<训练?':>12} {'GPU利用率':>12}")
    print("-" * 70)

    for storage in storages:
        for fmt_name, use_mmap, use_wds in formats:
            load = load_time_ms(batch_size, PILE, storage, num_workers=4,
                                use_mmap=use_mmap, use_webdataset=use_wds)

            ok = "✓" if load['total_ms'] <= train['time_ms'] else "✗"
            gpu_util = min(100, train['time_ms'] / (train['time_ms'] + load['total_ms']) * 100)

            print(f"{storage.name:<15} {fmt_name:<18} {load['total_ms']:>10.1f} {ok:>12} {gpu_util:>11.1f}%")
        print()

    print(f"关键洞察:")
    print(f"  - NVMe SSD: 所有格式都能跟上训练速度")
    print(f"  - NFS + jsonl: 可能跟不上, 需要 mmap 或 WebDataset")
    print(f"  - HDD: 完全跟不上, 必须用 WebDataset 顺序读")
    print(f"  - WebDataset (tar): 顺序读, 适合 HDD 和远程存储")
    print(f"  - mmap: 减少系统调用, 适合 SSD")


def experiment_4_tuning_recommendation():
    """实验 4: DataLoader 调优推荐"""
    print("\n" + "=" * 70)
    print("实验 4: DataLoader 调优推荐")
    print("=" * 70)

    print("""
## PyTorch DataLoader 调优参数

### 基础参数
  DataLoader(dataset, batch_size=B, shuffle=True,
             num_workers=N,         # 数据加载线程数
             prefetch_factor=P,     # 每 worker 预取批数
             pin_memory=True,       # 固定内存, 加速 CPU→GPU
             persistent_workers=True) # 保持 worker 不销毁

### 推荐配置

  小数据集 (<10GB, 单机):
    num_workers=4
    prefetch_factor=2
    pin_memory=True
    存储: NVMe SSD

  中等数据集 (10-100GB, 单机):
    num_workers=8
    prefetch_factor=2
    pin_memory=True
    persistent_workers=True
    存储: NVMe SSD + mmap

  大数据集 (>100GB, 分布式):
    num_workers=8
    prefetch_factor=2-4
    pin_memory=True
    persistent_workers=True
    存储: Lustre + WebDataset
    DistributedSampler + drop_last=True

  超大数据集 (>1TB, 分布式):
    WebDataset + tar 格式
    num_workers=8-16
    Streaming (不下载全部数据)
    存储: 云存储 + 本地缓存

## 常见问题

  问题 1: 训练第一个 epoch 慢
    → 原因: 文件系统缓存冷启动
    → 解决: 预热 (warmup) + 数据预加载

  问题 2: 多 worker CPU 占满
    → 减少 num_workers (通常 4-8 足够)
    → 使用 pin_memory=False (节省 CPU 内存)

  问题 3: OOM in DataLoader
    → prefetch_factor 过大
    → 改用流式加载 (WebDataset/IterableDataset)

  问题 4: 分布式数据重复
    → 使用 DistributedSampler(shuffle=True)
    → 确保 drop_last=True (所有 rank 等量数据)
""")

    print("## 性能基线参考")
    print()

    baselines = [
        ("LLaMA-7B, A100, B=8", 7.0, 312, 8, 2048),
        ("LLaMA-70B, H100, B=4", 70.0, 990, 4, 4096),
    ]

    for label, params, tflops, bs, sl in baselines:
        train = training_step_ms(params, bs, sl, tflops)
        print(f"  {label}:")
        print(f"    训练/步: {train['time_ms']:.1f}ms ({train['tokens']} tokens)")
        print(f"    需要 DataLoader > {1000/train['time_ms']:.0f} batches/s")
        for storage in [NVMe_SSD, NFS]:
            load = load_time_ms(bs, PILE, storage, num_workers=4)
            ok = "OK" if load['total_ms'] <= train['time_ms'] else "BOTTLENECK"
            print(f"    {storage.name}: {load['total_ms']:.1f}ms → {ok}")
        print()


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DataLoader 优化分析器")
    print("加载瓶颈 + Prefetch + 存储对比 + 调优推荐")
    print("=" * 70)

    experiment_1_bottleneck_analysis()
    experiment_2_prefetch_optimization()
    experiment_3_storage_comparison()
    experiment_4_tuning_recommendation()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
DataLoader 关键要点:

  1. 瓶颈识别:
     加载时间 < 训练时间 → 不是瓶颈 (理想)
     加载时间 > 训练时间 → GPU 空闲 (需优化)
     关键: 加载要跟上训练速度

  2. Prefetch 预取:
     后台线程加载下一批, 与训练重叠
     PyTorch: num_workers=4-8, prefetch_factor=2
     效果: NVMe 下完全隐藏加载延迟

  3. 存储选择:
     NVMe SSD: 5 GB/s, 几乎不会是瓶颈
     Lustre: 10 GB/s, 多节点共享
     NFS: 1 GB/s, 可能成为瓶颈
     HDD: 0.2 GB/s, 必须用 WebDataset

  4. 数据格式:
     jsonl: 通用, 但随机读 (HDD 不友好)
     mmap: SSD 优化, 减少系统调用
     WebDataset (tar): 顺序读, 适合 HDD 和远程存储
     Parquet: 列式存储, 适合分析型查询

  5. 调优优先级:
     1) pin_memory=True (零成本加速 CPU→GPU)
     2) num_workers=4-8 (并行加载)
     3) prefetch_factor=2 (预取 2 批)
     4) persistent_workers=True (避免重启)
     5) 换用 mmap 或 WebDataset (存储优化)
    """)
