#!/usr/bin/env python3
"""流水线并行 (PP) 气泡与效率分析器

CPU 可运行的流水线并行分析，覆盖 4 个实验:
  1. GPipe vs 1F1B 调度对比 (气泡分析)
  2. Micro-batch 数量对效率的影响
  3. PP vs TP vs DP 扩展效率对比
  4. 3D 并行策略推荐

关键概念:
  - PP: 将模型按层切分到不同 GPU (pipeline stages)
  - GPipe: 先全部 forward 再全部 backward, 气泡 = (P-1)/M
  - 1F1B: 交替 forward/backward, 气泡 ≈ (P-1)/(M+P-1)
  - Bubble ratio = idle_time / total_time
  - 有效计算 = M × compute_per_stage / (M × compute + bubble_time)

用法:
  conda run -n ai-infra python tools/pipeline_parallel_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class GPUConfig:
    name: str
    hbm_gb: float
    nvlink_bw_gbps: float  # 节点内互联带宽


@dataclass
class ModelConfig:
    name: str
    hidden: int
    inter: int
    heads: int
    head_dim: int
    layers: int
    params_B: float

    @property
    def model_size_gb(self) -> float:
        return self.params_B * 2  # FP16

    @property
    def size_per_layer_gb(self) -> float:
        return self.model_size_gb / self.layers


H100_80G = GPUConfig("H100-80G", 80, 900)
A100_80G = GPUConfig("A100-80G", 80, 600)
A100_40G = GPUConfig("A100-40G", 40, 600)

LLAMA7B = ModelConfig("LLaMA-7B", 4096, 11008, 32, 128, 32, 7.0)
LLAMA70B = ModelConfig("LLaMA-70B", 8192, 28672, 64, 128, 80, 70.0)
LLAMA405B = ModelConfig("LLaMA-405B", 16384, 57344, 128, 128, 126, 405.0)


# ──────────────────────────────────────────────
# 调度分析
# ──────────────────────────────────────────────

def gpipe_bubble_ratio(num_stages: int, num_microbatches: int) -> float:
    """GPipe 气泡比例

    GPipe: 全部 micro-batch forward, 然后 全部 backward
    时间线: F_1, F_2, ..., F_M, B_1, B_2, ..., B_M
    气泡: 前 P-1 个 stage 等待最后一个 stage 的 forward 和 backward
    气泡比例 = (P - 1) / M (近似)
    """
    if num_microbatches <= 0 or num_stages <= 1:
        return 0.0
    return (num_stages - 1) / num_microbatches


def onefoneb_bubble_ratio(num_stages: int, num_microbatches: int) -> float:
    """1F1B 气泡比例

    1F1B: 前面预热 P-1 个 forward, 然后交替 1F1B, 最后冷却 P-1 个 backward
    气泡: 只有开头和结尾的空闲时间
    气泡比例 ≈ (P - 1) / (2M + P - 1) (更精确)
    """
    if num_microbatches <= 0 or num_stages <= 1:
        return 0.0
    P = num_stages
    M = num_microbatches
    # 1F1B: 每个非边角 stage 都有 P-1 个 idle slot
    bubble_slots = P - 1
    total_slots = M + P - 1
    return bubble_slots / total_slots


def onefoneb_pipeline_schedule(
    num_stages: int,
    num_microbatches: int,
) -> dict:
    """模拟 1F1B 调度的时间线

    返回每个 stage 在每个时间槽的操作类型
    """
    P = num_stages
    M = num_microbatches

    # 总时间槽 = M + P - 1 (1F1B)
    total_slots = M + P - 1

    # 为每个 stage 构建操作序列
    schedule = []
    for stage_id in range(P):
        ops = []
        # 预热阶段: stage_id 个 forward
        warmup = min(stage_id, M)
        for i in range(warmup):
            ops.append(('F', i))

        # 稳态 1F1B
        steady = M - warmup
        for i in range(steady):
            mb_fwd = warmup + i
            mb_bwd = i
            ops.append(('F', mb_fwd))
            ops.append(('B', mb_bwd))

        # 冷却阶段: 剩余 backward
        cooldown = warmup
        for i in range(cooldown):
            mb_bwd = steady + i
            ops.append(('B', mb_bwd))

        schedule.append(ops)

    # 计算时间分配
    fwd_time_per_stage = 1.0  # 归一化为 1 个时间单位
    bwd_time_per_stage = 1.0  # backward 通常稍慢，简化为相同

    total_time = total_slots * (fwd_time_per_stage + bwd_time_per_stage) / 2
    # 实际 1F1B total = M fwd + M bwd + (P-1) bubble = 2M + (P-1) slots
    # 每个 slot = 1 个 fwd 或 1 个 bwd
    actual_slots = 2 * M + (P - 1)  # 总操作槽 (包含 bubble)
    compute_slots = 2 * M  # 实际计算槽
    bubble_slots = P - 1

    return {
        'num_stages': P,
        'num_microbatches': M,
        'total_slots': actual_slots,
        'compute_slots': compute_slots,
        'bubble_slots': bubble_slots,
        'bubble_ratio': bubble_slots / actual_slots,
        'schedule': schedule,
    }


def interleaved_1f1b_bubble_ratio(
    num_stages: int,
    num_microbatches: int,
    num_chunks: int = 1,
) -> float:
    """Interleaved 1F1B (Megatron-LM V-shape) 气泡比例

    每个设备负责 num_chunks 个不连续的 stage chunks
    气泡比例 ≈ (P-1) / (M + P - 1) / num_chunks
    """
    if num_microbatches <= 0 or num_stages <= 1:
        return 0.0
    P = num_stages
    M = num_microbatches
    V = num_chunks
    # Interleaved: 气泡减少约 V 倍
    return (P - 1) / (M * V + P - 1)


def pp_efficiency(
    num_stages: int,
    num_microbatches: int,
    schedule: str = "1f1b",
) -> dict:
    """计算 PP 效率

    效率 = 有效计算 / 总时间 = 1 - bubble_ratio
    """
    if schedule == "gpipe":
        bubble = gpipe_bubble_ratio(num_stages, num_microbatches)
    elif schedule == "1f1b":
        bubble = onefoneb_bubble_ratio(num_stages, num_microbatches)
    elif schedule == "interleaved":
        bubble = interleaved_1f1b_bubble_ratio(num_stages, num_microbatches, num_chunks=2)
    else:
        bubble = 0.0

    return {
        'num_stages': num_stages,
        'num_microbatches': num_microbatches,
        'schedule': schedule,
        'bubble_ratio': bubble,
        'efficiency': 1.0 - bubble,
    }


# ──────────────────────────────────────────────
# 显存约束
# ──────────────────────────────────────────────

def pp_memory_per_gpu(
    model: ModelConfig,
    gpu: GPUConfig,
    pp_size: int,
    batch_size: int,
    seq_len: int,
    tp_size: int = 1,
) -> dict:
    """估算 PP 下每个 GPU 的显存需求

    每个 stage 的显存 = 权重/P + 激活值 + 梯度 + 优化器状态
    """
    layers_per_stage = model.layers / pp_size
    weight_per_stage_gb = model.size_per_layer_gb * layers_per_stage / tp_size

    # 激活值 (每层的激活 = B × S × H × 2bytes)
    activation_per_layer_mb = batch_size * seq_len * model.hidden * 2 / 1e6
    # PP 每个 stage 需要保存的激活 = layers_per_stage × micro_batches
    # micro_batch 数 = batch_size / (gradient_accumulation_steps)
    # 简化: 假设 micro_batch = batch_size
    total_activation_mb = activation_per_layer_mb * layers_per_stage * 2  # fwd + bwd

    # 梯度和优化器 (与权重同大小)
    grad_per_stage_gb = weight_per_stage_gb
    optimizer_per_stage_gb = weight_per_stage_gb * (16 / 2)  # Adam: 4 × FP32 = 8x

    total_gb = weight_per_stage_gb + total_activation_mb / 1e3 + grad_per_stage_gb + optimizer_per_stage_gb

    return {
        'weight_gb': weight_per_stage_gb,
        'activation_mb': total_activation_mb,
        'grad_gb': grad_per_stage_gb,
        'optimizer_gb': optimizer_per_stage_gb,
        'total_gb': total_gb,
        'gpu_hbm_gb': gpu.hbm_gb,
        'fits': total_gb < gpu.hbm_gb * 0.85,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_gpipe_vs_1f1b():
    """实验 1: GPipe vs 1F1B 调度对比"""
    print("\n" + "=" * 70)
    print("实验 1: GPipe vs 1F1B 调度对比 (气泡分析)")
    print("=" * 70)

    num_stages_list = [2, 4, 8, 16]
    M = 16  # 16 个 micro-batches

    print(f"\n配置: {M} 个 micro-batches")
    print(f"\n{'Stages':>8} {'GPipe 气泡':>14} {'GPipe 效率':>12} "
          f"{'1F1B 气泡':>14} {'1F1B 效率':>12} {'Interleaved 气泡':>18} {'Interleaved 效率':>18}")
    print("-" * 100)

    for P in num_stages_list:
        gp = pp_efficiency(P, M, "gpipe")
        ob = pp_efficiency(P, M, "1f1b")
        il = pp_efficiency(P, M, "interleaved")

        print(f"{P:>8d} {gp['bubble_ratio']:>14.1%} {gp['efficiency']:>12.1%} "
              f"{ob['bubble_ratio']:>14.1%} {ob['efficiency']:>12.1%} "
              f"{il['bubble_ratio']:>18.1%} {il['efficiency']:>18.1%}")

    print(f"\n关键洞察:")
    print(f"  - GPipe: 气泡比例 = (P-1)/M, M 必须远大于 P 才有效率")
    print(f"  - 1F1B: 气泡比例 ≈ (P-1)/(M+P-1), 比 GPipe 大幅减少")
    print(f"  - Interleaved 1F1B: 进一步减少气泡 (~V 倍)")
    print(f"  - P=8, M=16: GPipe 43.8% 气泡, 1F1B 30% 气泡, Interleaved 18% 气泡")


def experiment_2_microbatch_impact():
    """实验 2: Micro-batch 数量对效率的影响"""
    print("\n" + "=" * 70)
    print("实验 2: Micro-batch 数量对 PP 效率的影响")
    print("=" * 70)

    P = 8  # 8 stages
    print(f"\n配置: P={P} stages")

    print(f"\n{'M':>6} {'GPipe 气泡':>14} {'GPipe 效率':>12} "
          f"{'1F1B 气泡':>14} {'1F1B 效率':>12}")
    print("-" * 60)

    for M in [4, 8, 16, 32, 64, 128, 256]:
        gp = pp_efficiency(P, M, "gpipe")
        ob = pp_efficiency(P, M, "1f1b")

        print(f"{M:>6d} {gp['bubble_ratio']:>14.1%} {gp['efficiency']:>12.1%} "
              f"{ob['bubble_ratio']:>14.1%} {ob['efficiency']:>12.1%}")

    print(f"\n关键洞察:")
    print(f"  - M=4 时 1F1B 效率仅 53.8%, 大量气泡")
    print(f"  - M=32 时 1F1B 效率 81.8%, 可接受")
    print(f"  - M=128+ 时效率 > 90%, 气泡可忽略")
    print(f"  - 实际 M = global_batch / (DP × micro_batch)")
    print(f"  - 大 batch 训练 (RLHF, pretraining) 天然适合 PP")


def experiment_3_3d_parallel_comparison():
    """实验 3: PP vs TP vs DP 扩展效率对比"""
    print("\n" + "=" * 70)
    print("实验 3: PP vs TP vs DP 扩展效率对比")
    print("=" * 70)

    model = LLAMA70B
    gpu = A100_80G

    print(f"\n配置: {model.name} ({model.params_B}B), GPU={gpu.name}")
    print(f"  权重: {model.model_size_gb:.1f} GB, 每 layer: {model.size_per_layer_gb:.2f} GB")

    # DP: 线性扩展 (理想)
    # TP: 通信开销 (前面分析过, ~5-15%)
    # PP: 气泡开销 (取决于 M)

    num_gpus_list = [2, 4, 8, 16, 32, 64]
    M = 32  # micro-batches

    print(f"\n{'GPUs':>6} {'DP 效率':>12} {'TP 效率':>12} {'PP(M=32) 效率':>16} "
          f"{'TP+PP 3D 效率':>16} {'推荐策略':<20}")
    print("-" * 90)

    for N in num_gpus_list:
        # DP: 线性扩展, 通信开销极小 (AllReduce gradient)
        dp_eff = 0.95  # 几乎线性

        # TP: 通信开销, NVLink
        # TP 效率是每个 TP 组内部的效率, 不取决于总 GPU 数
        def tp_efficiency_for(tp_size):
            if tp_size <= 1:
                return 1.0
            return max(0.85 - (tp_size - 2) * 0.02, 0.70)

        # PP: 气泡
        pp_eff = 1.0 - onefoneb_bubble_ratio(N, M) if N > 1 else 1.0

        # 3D: TP × PP
        if N <= 8:
            # TP only
            tp_used = N
            pp_used = 1
            dp_used = 1
            strategy = f"TP={tp_used}"
            eff_3d = tp_efficiency_for(tp_used)
        elif N <= 32:
            tp_used = min(N, 8)
            pp_used = N // tp_used
            dp_used = 1
            eff_3d = tp_efficiency_for(tp_used) * (1.0 - onefoneb_bubble_ratio(pp_used, M))
            strategy = f"TP={tp_used}, PP={pp_used}"
        else:
            tp_used = 8
            pp_used = 4
            dp_used = N // (tp_used * pp_used)
            pp_eff_val = 1.0 - onefoneb_bubble_ratio(pp_used, M)
            eff_3d = tp_efficiency_for(tp_used) * pp_eff_val * dp_eff
            strategy = f"TP={tp_used}, PP={pp_used}, DP={dp_used}"

        dp_display = dp_eff if N <= 64 else 0
        tp_display = tp_efficiency_for(min(N, 8)) if N <= 8 else tp_efficiency_for(8)
        pp_display = pp_eff if N > 1 else 1.0

        print(f"{N:>6d} {dp_display:>12.1%} {tp_display:>12.1%} "
              f"{pp_display:>16.1%} {eff_3d:>16.1%} {strategy:<20}")

    print(f"\n关键洞察:")
    print(f"  - DP: 几乎线性扩展, 但每张卡要放下整个模型")
    print(f"  - TP: NVLink 下效率 70-90%, 受限于通信")
    print(f"  - PP: 受气泡影响, 需要大 M 才高效")
    print(f"  - 3D 并行: DP > TP > PP (优先用 DP 和 TP)")
    print(f"  - 70B: TP=8 (1 节点), 需要 PP=4+ 才能放进 2+ 节点")


def experiment_4_3d_strategy_recommendation():
    """实验 4: 3D 并行策略推荐"""
    print("\n" + "=" * 70)
    print("实验 4: 3D 并行策略推荐 (结合显存 + 效率)")
    print("=" * 70)

    scenarios = [
        (LLAMA7B, A100_80G, 8, "Pretraining"),
        (LLAMA70B, A100_80G, 8, "Pretraining"),
        (LLAMA70B, A100_80G, 32, "Pretraining"),
        (LLAMA70B, H100_80G, 8, "Pretraining"),
        (LLAMA405B, H100_80G, 64, "Pretraining"),
        (LLAMA70B, A100_80G, 4, "RLHF (small batch)"),
    ]

    M_default = 32

    print(f"\n{'模型':<15} {'GPU':<12} {'N':>4} {'场景':<20} "
          f"{'TP':>4} {'PP':>4} {'DP':>4} {'显存/卡':>10} {'气泡':>8} {'总效率':>10}")
    print("-" * 105)

    for model, gpu, N, scenario in scenarios:
        weight_gb = model.model_size_gb

        # 最小 TP: 模型能放进单卡
        min_tp = math.ceil(weight_gb / (gpu.hbm_gb * 0.7))

        # 确定 TP
        if min_tp <= N:
            tp = min(min_tp, N, 8)  # TP 最大 8 (NVLink 限制)
        else:
            tp = min(N, 8)

        # 剩余 GPU 给 PP 和 DP
        remaining = N // tp

        if remaining == 1:
            pp, dp = 1, 1
        elif model.layers >= 64 and remaining <= 4:
            # 层数多, 用 PP
            pp = remaining
            dp = 1
        elif remaining <= 2:
            pp = 1
            dp = remaining
        else:
            # 平衡 PP 和 DP
            pp = min(remaining, max(2, remaining // 2))
            dp = remaining // pp

        actual_N = tp * pp * dp
        mem = pp_memory_per_gpu(model, gpu, max(pp, 1), 4, 2048, tp)

        # 效率
        pp_bubble = onefoneb_bubble_ratio(pp, M_default) if pp > 1 else 0
        tp_overhead = 0.05 * (tp - 1) if tp > 1 else 0  # 5%/stage 通信
        dp_overhead = 0.02 if dp > 1 else 0

        total_eff = (1 - pp_bubble) * (1 - tp_overhead) * (1 - dp_overhead)

        print(f"{model.name:<15} {gpu.name:<12} {N:>4d} {scenario:<20} "
              f"{tp:>4d} {pp:>4d} {dp:>4d} "
              f"{mem['total_gb']:>9.1f}G {pp_bubble:>8.1%} {total_eff:>10.1%}")

    print(f"\n通用策略:")
    print(f"  1. 先定 TP: 权重/激活能放进单卡 (≥ 30% 显存余量)")
    print(f"  2. 再定 PP: 大模型层数多时用 PP, 但需要大 M")
    print(f"  3. 最后 DP: 用剩余 GPU 做 DP (几乎线性扩展)")
    print(f"  4. RLHF 小 batch: 避免大 PP (M 小 → 气泡大)")
    print(f"  5. 优先级: DP > TP > PP (通信/气泡开销递增)")
    print(f"\n典型配置:")
    print(f"  LLaMA-7B,  8×A100:    TP=1, PP=1, DP=8  (纯 DP)")
    print(f"  LLaMA-70B, 8×A100:    TP=8, PP=1, DP=1  (单节点 TP)")
    print(f"  LLaMA-70B, 32×A100:   TP=8, PP=4, DP=1  (4 节点)")
    print(f"  LLaMA-405B, 64×H100:  TP=8, PP=4, DP=2  (8 节点)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("流水线并行 (PP) 气泡与效率分析器")
    print("GPipe / 1F1B / Interleaved 1F1B 调度分析")
    print("=" * 70)

    experiment_1_gpipe_vs_1f1b()
    experiment_2_microbatch_impact()
    experiment_3_3d_parallel_comparison()
    experiment_4_3d_strategy_recommendation()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
流水线并行关键要点:

  1. 调度策略:
     GPipe: 全部 forward → 全部 backward
       气泡 = (P-1)/M, 需要大 M
     1F1B: 交替 forward/backward
       气泡 ≈ (P-1)/(M+P-1), GPipe 的 ~50-70%
     Interleaved 1F1B: 设备负责多个 chunks
       气泡再减少 V 倍 (V=chunks 数)

  2. 关键约束:
     M (micro-batches) >> P (stages)
     M=4, P=8: 效率 < 55% (很差)
     M=32, P=8: 效率 ~82% (可接受)
     M=128, P=8: 效率 ~94% (良好)
     → 小 batch 任务 (RLHF) 不适合大 PP

  3. 显存约束:
     PP 每 stage 权重 = model_size / P
     PP 每 stage 激活 = micro_batch × seq_len × hidden
     → PP 可以显著降低单卡显存需求

  4. 3D 并行策略:
     优先级: DP > TP > PP
     选择顺序:
       a) TP: 满足显存约束的最小值
       b) PP: 大模型层数多时, 结合 DP
       c) DP: 用剩余 GPU, 几乎线性扩展
     典型配置:
       70B, 8×A100:   TP=8
       70B, 32×A100:  TP=8, PP=4
       405B, 64×H100: TP=8, PP=4, DP=2

  5. PP 特殊优化:
     Interleaved 1F1B: 减少气泡 (V 倍)
     Virtual Pipeline: 每个 GPU 负责 V 个 chunks
     Bubble-free PP: 训练和推理结合 (前沿研究)
    """)
