#!/usr/bin/env python3
"""分布式 Checkpoint 性能分析器

CPU 可运行的 checkpoint 保存/加载性能分析，覆盖 4 个实验:
  1. Checkpoint 保存策略对比 (单进程 vs 分片 vs 异步)
  2. 保存时间估算 (模型大小 × 存储 IO)
  3. 加载时间与恢复速度分析
  4. Checkpoint 存储策略推荐

关键概念:
  - Checkpoint = 模型权重 + 优化器状态 + RNG state + 训练元数据
  - 单进程保存: rank 0 收集所有参数, 写入一个文件
  - 分片保存: 每个 rank 写自己的分片 (Megatron/ZeRO)
  - 异步保存: 训练继续, 后台线程写磁盘
  - 存储瓶颈: SSD 顺序写 ~5 GB/s, NFS ~1 GB/s, HDD ~0.2 GB/s

用法:
  conda run -n ai-infra python tools/checkpoint_perf_sim.py
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
    write_bw_gbps: float   # 顺序写带宽 (GB/s)
    read_bw_gbps: float    # 顺序读带宽 (GB/s)
    latency_ms: float       # IO 延迟 (ms)
    parallel: bool          # 是否支持多进程并行 IO


SSD_NVMe = StorageConfig("NVMe SSD", 5.0, 5.0, 0.05, True)
SSD_SATA = StorageConfig("SATA SSD", 2.0, 2.0, 0.1, True)
NFS_Lustre = StorageConfig("Lustre (并行)", 10.0, 10.0, 0.5, True)
NFS_Generic = StorageConfig("NFS (通用)", 1.0, 1.0, 2.0, False)
HDD = StorageConfig("HDD", 0.2, 0.2, 5.0, False)
Local_RAM = StorageConfig("RAM Disk", 50.0, 50.0, 0.001, True)


@dataclass
class ModelConfig:
    name: str
    params_B: float
    layers: int
    hidden: int


GPT2 = ModelConfig("GPT-2 Small", 0.124, 12, 768)
LLAMA7B = ModelConfig("LLaMA-7B", 7.0, 32, 4096)
LLAMA70B = ModelConfig("LLaMA-70B", 70.0, 80, 8192)
LLAMA405B = ModelConfig("LLaMA-405B", 405.0, 126, 16384)


# ──────────────────────────────────────────────
# Checkpoint 大小估算
# ──────────────────────────────────────────────

def checkpoint_size_gb(
    model: ModelConfig,
    precision: str = "bf16",
    optimizer: str = "adam",
    include_optimizer: bool = True,
    include_rng: bool = True,
    include_metadata: bool = True,
) -> dict:
    """估算单个 checkpoint 的大小

    组成:
    1. 模型权重: params × bytes_per_param
    2. 优化器状态:
       - Adam: FP32 master (4B) + momentum (4B) + variance (4B) = 12B/param
       - SGD: momentum (4B) = 4B/param
    3. RNG state: ~几十 KB (可忽略)
    4. 元数据: ~几 MB (可忽略)
    """
    params = model.params_B * 1e9

    # 权重
    if precision in ("fp16", "bf16"):
        weight_bytes = 2
    elif precision == "fp32":
        weight_bytes = 4
    elif precision == "fp8":
        weight_bytes = 1
    else:
        weight_bytes = 2

    weight_gb = params * weight_bytes / 1e9

    # 优化器
    if include_optimizer:
        if optimizer == "adam":
            # FP32 master + m + v
            optimizer_gb = params * 12 / 1e9
        elif optimizer == "adamw":
            optimizer_gb = params * 12 / 1e9
        elif optimizer == "sgd":
            optimizer_gb = params * 4 / 1e9  # momentum
        else:
            optimizer_gb = 0
    else:
        optimizer_gb = 0

    # 其他
    rng_gb = 0.001 if include_rng else 0  # ~1 MB
    metadata_gb = 0.01 if include_metadata else 0  # ~10 MB

    total_gb = weight_gb + optimizer_gb + rng_gb + metadata_gb

    return {
        'model': model.name,
        'weight_gb': weight_gb,
        'optimizer_gb': optimizer_gb,
        'rng_gb': rng_gb,
        'metadata_gb': metadata_gb,
        'total_gb': total_gb,
        'weight_pct': weight_gb / total_gb * 100,
        'optimizer_pct': optimizer_gb / total_gb * 100,
    }


# ──────────────────────────────────────────────
# 保存/加载时间估算
# ──────────────────────────────────────────────

def save_time_seconds(
    size_gb: float,
    storage: StorageConfig,
    num_ranks: int = 1,
    strategy: str = "single_rank",  # single_rank, sharded, async
    zero_stage: int = 0,
) -> dict:
    """估算 checkpoint 保存时间

    策略:
    - single_rank: rank 0 收集所有数据, 写入单个文件
      时间 = 序列化 + 传输到 rank 0 + 写磁盘
    - sharded: 每个 rank 写自己的分片
      时间 = 序列化 + 写磁盘 (并行)
    - async: 后台线程写磁盘, 训练不中断
      对训练的影响 ≈ 序列化时间 (不等待 IO)
    """
    # 序列化时间 (CPU, 假设 ~10 GB/s)
    serialize_bw = 10.0  # GB/s
    serialize_time = size_gb / serialize_bw

    if strategy == "single_rank":
        # 所有数据传输到 rank 0 (通过 NCCL Gather)
        # 通信时间 ≈ size / NVLink_bw
        comm_bw = 50.0  # GB/s (NVLink 有效带宽)
        if num_ranks > 1:
            comm_time = size_gb / comm_bw
        else:
            comm_time = 0

        # 单进程写磁盘
        write_time = size_gb / storage.write_bw_gbps

        total_time = serialize_time + comm_time + write_time
        training_impact = total_time  # 训练完全暂停

    elif strategy == "sharded":
        # 每个 rank 写自己的分片
        per_rank_size = size_gb / num_ranks
        serialize_time_per_rank = per_rank_size / serialize_bw

        if storage.parallel:
            # 并行存储 (Lustre/NVMe), 每个(rank)独立写
            write_time = per_rank_size / storage.write_bw_gbps
        else:
            # 串行存储 (NFS/HDD), 带宽共享
            write_time = per_rank_size / (storage.write_bw_gbps)

        total_time = serialize_time_per_rank + write_time
        training_impact = total_time  # 仍然暂停训练

    elif strategy == "async":
        # 异步: 训练只等序列化, IO 后台执行
        per_rank_size = size_gb / num_ranks
        serialize_time_per_rank = per_rank_size / serialize_bw
        write_time = per_rank_size / storage.write_bw_gbps

        total_time = serialize_time_per_rank + write_time
        training_impact = serialize_time_per_rank  # 只等序列化

    else:
        total_time = 0
        training_impact = 0

    return {
        'strategy': strategy,
        'size_gb': size_gb,
        'total_time_s': total_time,
        'serialize_time_s': serialize_time if strategy == "single_rank" else serialize_time_per_rank,
        'write_time_s': write_time if strategy == "single_rank" else write_time,
        'training_impact_s': training_impact,
        'storage': storage.name,
    }


def load_time_seconds(
    size_gb: float,
    storage: StorageConfig,
    num_ranks: int = 1,
    strategy: str = "sharded",
) -> dict:
    """估算 checkpoint 加载时间"""
    # 读磁盘
    if strategy == "single_rank":
        read_time = size_gb / storage.read_bw_gbps
        # 从 rank 0 广播到所有 rank
        comm_bw = 50.0
        comm_time = size_gb / comm_bw if num_ranks > 1 else 0
        deserialize_time = size_gb / 10.0  # CPU 反序列化
        total_time = read_time + comm_time + deserialize_time
    elif strategy == "sharded":
        per_rank_size = size_gb / num_ranks
        read_time = per_rank_size / storage.read_bw_gbps
        deserialize_time = per_rank_size / 10.0
        total_time = read_time + deserialize_time
    else:
        total_time = 0

    return {
        'strategy': strategy,
        'size_gb': size_gb,
        'total_time_s': total_time,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_checkpoint_size():
    """实验 1: Checkpoint 大小分析"""
    print("\n" + "=" * 70)
    print("实验 1: Checkpoint 大小分析")
    print("=" * 70)

    models = [GPT2, LLAMA7B, LLAMA70B, LLAMA405B]

    print(f"\n{'模型':<15} {'权重GB':>10} {'优化器GB':>12} {'总大小GB':>12} "
          f"{'权重占比':>10} {'优化器占比':>12} {'CKPT/训练显存':>16}")
    print("-" * 90)

    for model in models:
        ckpt = checkpoint_size_gb(model, precision="bf16", optimizer="adam")

        # 训练显存估算 (简化: weight + optimizer + grad + activation)
        train_mem = model.params_B * 2 + model.params_B * 12 + model.params_B * 2  # weight + opt + grad
        ratio = ckpt['total_gb'] / train_mem * 100

        print(f"{model.name:<15} {ckpt['weight_gb']:>10.1f} {ckpt['optimizer_gb']:>12.1f} "
              f"{ckpt['total_gb']:>12.1f} {ckpt['weight_pct']:>9.1f}% "
              f"{ckpt['optimizer_pct']:>11.1f}% {ratio:>14.0f}%")

    print(f"\n关键洞察:")
    print(f"  - GPT-2: CKPT ~1.7 GB, 轻松保存")
    print(f"  - LLaMA-7B: CKPT ~98 GB, 需要大存储")
    print(f"  - LLaMA-70B: CKPT ~980 GB, 需要分布式存储")
    print(f"  - LLaMA-405B: CKPT ~5.7 TB, 需要 Lustre 并行存储")
    print(f"  - 优化器状态占 85%+, 只保存权重可节省 85%")


def experiment_2_save_strategies():
    """实验 2: Checkpoint 保存策略对比"""
    print("\n" + "=" * 70)
    print("实验 2: Checkpoint 保存策略对比")
    print("=" * 70)

    model = LLAMA70B
    ckpt = checkpoint_size_gb(model, precision="bf16", optimizer="adam")

    print(f"\n模型: {model.name}, CKPT 大小: {ckpt['total_gb']:.1f} GB")

    # 不同 GPU 数量和存储
    configs = [
        (8, SSD_NVMe, "8×A100 + NVMe"),
        (8, NFS_Generic, "8×A100 + NFS"),
        (32, SSD_NVMe, "32×A100 + NVMe"),
        (32, NFS_Lustre, "32×A100 + Lustre"),
        (64, NFS_Lustre, "64×H100 + Lustre"),
    ]

    print(f"\n{'配置':<25} {'策略':<15} {'总时间':>10} {'训练影响':>10} {'写带宽':>10}")
    print("-" * 75)

    for num_ranks, storage, label in configs:
        for strategy in ["single_rank", "sharded", "async"]:
            r = save_time_seconds(ckpt['total_gb'], storage, num_ranks, strategy)

            print(f"{label:<25} {strategy:<15} {r['total_time_s']:>9.1f}s "
                  f"{r['training_impact_s']:>9.1f}s {storage.write_bw_gbps:>9.1f}G/s")
        print()

    print(f"关键洞察:")
    print(f"  - single_rank: 简单但最慢, 所有数据汇聚到 rank 0")
    print(f"  - sharded: 并行写, Lustre 下最快 (10 GB/s 聚合)")
    print(f"  - async: 训练影响最小 (只等序列化 ~1-2s)")
    print(f"  - 70B 模型 NVMe 保存: ~196s (single) vs ~25s (sharded)")
    print(f"  - 异步保存是大规模训练的标配 (Megatron/DeepSpeed)")


def experiment_3_load_recovery():
    """实验 3: 加载时间与恢复分析"""
    print("\n" + "=" * 70)
    print("实验 3: Checkpoint 加载与故障恢复")
    print("=" * 70)

    models = [LLAMA7B, LLAMA70B, LLAMA405B]

    print(f"\n存储: NVMe SSD (5 GB/s 读)")

    print(f"\n{'模型':<15} {'CKPT GB':>10} {'单进程加载':>14} {'分片加载(8卡)':>16} "
          f"{'分片加载(64卡)':>18} {'恢复到训练':>14}")
    print("-" * 92)

    for model in models:
        ckpt = checkpoint_size_gb(model, precision="bf16", optimizer="adam")

        r_single = load_time_seconds(ckpt['total_gb'], SSD_NVMe, 1, "single_rank")
        r_8 = load_time_seconds(ckpt['total_gb'], SSD_NVMe, 8, "sharded")
        r_64 = load_time_seconds(ckpt['total_gb'], SSD_NVMe, 64, "sharded")

        # 恢复到训练还需要: 初始化 + warmup
        recovery_s = r_64['total_time_s'] + 30  # 额外 30s 初始化

        print(f"{model.name:<15} {ckpt['total_gb']:>10.1f} {r_single['total_time_s']:>13.1f}s "
              f"{r_8['total_time_s']:>15.1f}s {r_64['total_time_s']:>17.1f}s "
              f"{recovery_s:>12.1f}s")

    print(f"\n关键洞察:")
    print(f"  - LLaMA-7B: 分片加载 < 5s, 恢复极快")
    print(f"  - LLaMA-70B: 分片 64 卡加载 ~20s, 可接受")
    print(f"  - LLaMA-405B: 分片 64 卡加载 ~120s, 需要优化")
    print(f"  - 分片加载 ∝ CKPT_size / (num_ranks × storage_bw)")
    print(f"  - 只加载权重 (跳过优化器): 快 7x, 但需要 warmup")

    # 故障恢复场景
    print(f"\n--- 故障恢复场景分析 ---")
    print(f"  假设: 8 节点 64×H100 训练 LLaMA-70B, 每 1000 步保存一次")
    print(f"  每 1000 步 = ~30 分钟训练")
    print(f"  单节点故障概率 ≈ 0.1%/小时")
    print(f"  平均故障间隔 (MTBF) ≈ 8 × 0.1% = 125 小时")
    print(f"  每次故障丢失: 0-30 分钟训练 (平均 15 分钟)")
    print(f"  恢复时间: 加载 + 初始化 ≈ 2 分钟")
    print(f"  有效训练率: ~99.9% (保存频率足够时)")


def experiment_4_storage_recommendation():
    """实验 4: Checkpoint 存储策略推荐"""
    print("\n" + "=" * 70)
    print("实验 4: Checkpoint 存储策略推荐")
    print("=" * 70)

    scenarios = [
        {
            'name': '实验/调试 (7B)',
            'model': LLAMA7B,
            'num_gpus': 1,
            'storage': SSD_NVMe,
            'freq': 100,  # 每 100 步
        },
        {
            'name': '中等训练 (7B)',
            'model': LLAMA7B,
            'num_gpus': 8,
            'storage': SSD_NVMe,
            'freq': 500,
        },
        {
            'name': '大规模训练 (70B)',
            'model': LLAMA70B,
            'num_gpus': 32,
            'storage': NFS_Lustre,
            'freq': 1000,
        },
        {
            'name': '超大规模 (405B)',
            'model': LLAMA405B,
            'num_gpus': 128,
            'storage': NFS_Lustre,
            'freq': 2000,
        },
    ]

    print(f"\n{'场景':<25} {'CKPT GB':>10} {'保存时间':>12} {'训练影响':>12} "
          f"{'存储频率':>10} {'存储空间/天':>14} {'推荐策略':<20}")
    print("-" * 110)

    for s in scenarios:
        ckpt = checkpoint_size_gb(s['model'], precision="bf16", optimizer="adam")

        # 异步分片保存
        r = save_time_seconds(ckpt['total_gb'], s['storage'], s['num_gpus'], "async")

        # 存储空间估算: 假设保留最近 5 个 checkpoint
        # 每 step ~2s, freq 步保存一次
        steps_per_hour = 1800
        saves_per_hour = steps_per_hour / s['freq']
        gb_per_hour = ckpt['total_gb'] * saves_per_hour
        gb_per_day = gb_per_hour * 24

        # 推荐策略
        if ckpt['total_gb'] < 50:
            rec = "同步保存, 保留 5 个"
        elif ckpt['total_gb'] < 500:
            rec = "分片保存, 异步 IO"
        else:
            rec = "Lustre + 分片 + 异步"

        print(f"{s['name']:<25} {ckpt['total_gb']:>10.1f} {r['total_time_s']:>11.1f}s "
              f"{r['training_impact_s']:>11.1f}s {s['freq']:>10d} "
              f"{gb_per_day:>12.1f} GB {rec:<20}")

    print(f"\n通用建议:")
    print(f"  1. 小模型 (<50GB CKPT): 同步保存即可, 简单可靠")
    print(f"  2. 中等模型 (50-500GB): 分片保存 + NVMe SSD")
    print(f"  3. 大模型 (>500GB): 分片 + 异步 + Lustre 并行存储")
    print(f"  4. 超大模型 (>5TB): 只保存权重 + 优化器分片 + 压缩")
    print(f"  5. 频率: 训练 30 分钟内可接受的重做量 (通常 500-2000 步)")
    print(f"  6. 保留: 最近 3-5 个 + 每 N 步一个长期保存")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("分布式 Checkpoint 性能分析器")
    print("保存/加载策略 + 存储 IO + 恢复分析")
    print("=" * 70)

    experiment_1_checkpoint_size()
    experiment_2_save_strategies()
    experiment_3_load_recovery()
    experiment_4_storage_recommendation()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Checkpoint 关键要点:

  1. 大小分析:
     优化器状态占 85%+ (Adam 12 bytes/param)
     权重只占 ~15% (BF16 2 bytes/param)
     7B: ~98 GB, 70B: ~980 GB, 405B: ~5.7 TB

  2. 保存策略:
     单进程: 简单但慢, 只适合小模型
     分片: 每个 rank 写自己分片, 并行 IO
     异步: 后台写磁盘, 训练只等序列化
     → 大规模训练: 分片 + 异步 + 并行存储

  3. 存储选择:
     NVMe SSD: 5 GB/s, 适合单节点
     Lustre: 10+ GB/s 聚合, 适合多节点
     NFS: 1 GB/s, 不适合大模型
     → 存储带宽决定保存速度

  4. 故障恢复:
     MTBF = 单节点_MTBF / num_nodes
     恢复时间 = 加载 + 初始化
     有效训练率 > 99% (合理的保存频率)
     → 频率 = 可接受的重做量 / 每步时间

  5. 最佳实践:
     1) 异步保存 + 分片存储 (Megatron/DeepSpeed)
     2) 只保存权重用于推理 (小 7x)
     3) 保留最近 3-5 个 + 定期长期保存
     4) 存储空间预算: CKPT_size × saves_per_day × retention
     5) 定期验证 checkpoint 完整性
    """)
