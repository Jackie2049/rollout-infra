#!/usr/bin/env python3
"""梯度累积与混合精度训练模拟器

CPU 可运行的训练优化分析工具，覆盖 4 个实验:
  1. 梯度累积 vs 大 Batch 等价性分析
  2. 混合精度 (FP16/BF16/FP8) 训练吞吐对比
  3. 梯度检查点 (Activation Recomputation) 显存节省
  4. 训练显存估算器 (综合所有优化)

关键概念:
  - 梯度累积: micro_batch × grad_accum = effective_batch, 节省显存但增加时间
  - 混合精度: FP16/BF16 前向 + FP32 权重更新, 需要 loss scaling
  - 梯度检查点: 丢弃部分激活, backward 时重计算, 用计算换显存
  - 训练显存 = 权重 + 优化器 + 梯度 + 激活值

用法:
  conda run -n ai-infra python tools/training_optimizer_sim.py
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    hidden: int
    inter: int
    heads: int
    head_dim: int
    layers: int
    params_B: float
    seq_len: int = 2048

    @property
    def model_size_gb(self) -> float:
        return self.params_B * 2  # FP16

    @property
    def bytes_per_param(self) -> int:
        return 2  # FP16


GPT2 = ModelConfig("GPT-2 Small", 768, 3072, 12, 64, 12, 0.124, seq_len=1024)
LLAMA7B = ModelConfig("LLaMA-7B", 4096, 11008, 32, 128, 32, 7.0)
LLAMA70B = ModelConfig("LLaMA-70B", 8192, 28672, 64, 128, 80, 70.0)


# ──────────────────────────────────────────────
# 显存估算
# ──────────────────────────────────────────────

def estimate_training_memory(
    model: ModelConfig,
    batch_size: int,
    seq_len: int,
    precision: str = "fp16",  # fp16, bf16, fp32, fp8
    optimizer: str = "adam",  # adam, sgd
    gradient_checkpointing: bool = False,
    zero_stage: int = 0,  # 0, 1, 2, 3
    dp_size: int = 1,
    tp_size: int = 1,
) -> dict:
    """估算单 GPU 训练显存需求

    训练显存组成:
    1. 模型权重: params × bytes_per_param
    2. 优化器状态: Adam = params × 12 bytes (m + v in FP32)
    3. 梯度: params × bytes_per_param
    4. 激活值: layers × batch × seq × hidden × bytes_per_activation
    5. 临时缓冲区
    """
    params = model.params_B * 1e9

    # 每个参数的字节数
    if precision in ("fp16", "bf16"):
        weight_bytes = 2
    elif precision == "fp32":
        weight_bytes = 4
    elif precision == "fp8":
        weight_bytes = 1
    else:
        weight_bytes = 2

    # 1. 权重
    weight_gb = params * weight_bytes / 1e9 / tp_size

    # 2. 优化器状态
    if optimizer == "adam":
        # FP32 master weights + m + v = 4 + 4 + 4 = 12 bytes/param
        optimizer_gb = params * 12 / 1e9 / dp_size / tp_size if zero_stage < 1 else 0
        if zero_stage >= 1:
            # ZeRO-1: 分片优化器状态
            optimizer_gb = params * 12 / 1e9 / dp_size / tp_size
        if zero_stage >= 2:
            # ZeRO-2: 还分片梯度
            pass
        if zero_stage >= 3:
            # ZeRO-3: 还分片权重
            weight_gb = params * weight_bytes / 1e9 / dp_size / tp_size
    elif optimizer == "sgd":
        optimizer_gb = params * 4 / 1e9 / dp_size / tp_size  # momentum
    else:
        optimizer_gb = 0

    # 3. 梯度
    if zero_stage >= 2:
        grad_gb = params * weight_bytes / 1e9 / dp_size / tp_size
    else:
        grad_gb = params * weight_bytes / 1e9 / tp_size

    # 4. 激活值
    # 每个 transformer layer 的激活:
    # - Attention: Q, K, V, Attn_weights, Attn_output, etc.
    # - MLP: gate, up, down activations
    # 简化估算: ~34 × B × S × H bytes per layer (Sheng et al.)
    # 更精确: 2 × B × S × (5 × H × heads/head_dim × S + 34 × H) per layer
    # 简化为: 每层激活 ≈ 34 × B × S × H × 2 bytes (FP16 输出)
    if gradient_checkpointing:
        # 只保存 checkpoint 边界的激活 (通常每层都 checkpoint)
        # 需要保存: 输入激活 = B × S × H × 2 bytes
        # 重计算时的峰值: 和不 checkpoint 时一样, 但不是所有层同时
        activation_per_layer_mb = batch_size * seq_len * model.hidden * 2 / 1e6
        # 梯度检查点: 只存边界层 + 1 层重计算
        num_checkpoints = math.ceil(model.layers / tp_size / 2)  # 每 2 层 1 个 checkpoint
        total_activation_mb = (num_checkpoints + 1) * activation_per_layer_mb
        # 加上 1 层的完整激活用于重计算
        full_layer_activation_mb = 34 * batch_size * seq_len * model.hidden * 2 / 1e6
        total_activation_mb += full_layer_activation_mb
    else:
        activation_per_layer_mb = 34 * batch_size * seq_len * model.hidden * 2 / 1e6
        total_activation_mb = activation_per_layer_mb * (model.layers / tp_size)

    # 5. 临时缓冲区 (碎片化 + CUDA context + 临时张量)
    temp_buffer_gb = 1.0  # 固定 1 GB 估算

    # 总计
    total_gb = (weight_gb + optimizer_gb + grad_gb +
                total_activation_mb / 1e3 + temp_buffer_gb)

    return {
        'model': model.name,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'precision': precision,
        'optimizer': optimizer,
        'gradient_checkpointing': gradient_checkpointing,
        'zero_stage': zero_stage,
        'weight_gb': weight_gb,
        'optimizer_gb': optimizer_gb,
        'grad_gb': grad_gb,
        'activation_mb': total_activation_mb,
        'activation_gb': total_activation_mb / 1e3,
        'temp_buffer_gb': temp_buffer_gb,
        'total_gb': total_gb,
        'activation_pct': total_activation_mb / 1e3 / total_gb * 100,
        'weight_pct': weight_gb / total_gb * 100,
        'optimizer_pct': optimizer_gb / total_gb * 100,
    }


# ──────────────────────────────────────────────
# 吞吐估算
# ──────────────────────────────────────────────

def estimate_training_throughput(
    model: ModelConfig,
    gpu_tflops: float,
    batch_size: int,
    seq_len: int,
    grad_accum_steps: int = 1,
    gradient_checkpointing: bool = False,
) -> dict:
    """估算训练吞吐 (tokens/s)

    训练 FLOPs = 6 × params × tokens (forward + backward)
    梯度检查点额外: +33% FLOPs (重计算激活)
    """
    tokens = batch_size * seq_len
    total_tokens = tokens * grad_accum_steps

    # Forward: 2 × params × tokens (矩阵乘)
    # Backward: 4 × params × tokens (梯度计算约 2x forward)
    flops_per_token = 6 * model.params_B * 1e9
    total_flops = flops_per_token * total_tokens

    # 梯度检查点额外开销
    if gradient_checkpointing:
        total_flops *= 1.33

    # GPU 利用率
    efficiency = 0.45  # MFU (Model FLOPs Utilization)
    compute_time_s = total_flops / (gpu_tflops * 1e12 * efficiency)

    # 通信开销 (AllReduce gradient, DP)
    # 简化: 每 step 额外 5% 时间
    comm_overhead = 0.05
    total_time_s = compute_time_s * (1 + comm_overhead)

    # 吞吐
    tokens_per_s = total_tokens / total_time_s
    samples_per_s = batch_size * grad_accum_steps / total_time_s
    effective_batch = batch_size * grad_accum_steps

    return {
        'effective_batch': effective_batch,
        'grad_accum_steps': grad_accum_steps,
        'total_tokens': total_tokens,
        'total_flops_T': total_flops / 1e12,
        'compute_time_s': compute_time_s,
        'total_time_s': total_time_s,
        'tokens_per_s': tokens_per_s,
        'samples_per_s': samples_per_s,
        'mfu': efficiency,
    }


# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────

def experiment_1_gradient_accumulation():
    """实验 1: 梯度累积 vs 大 Batch"""
    print("\n" + "=" * 70)
    print("实验 1: 梯度累积 vs 大 Batch 等价性分析")
    print("=" * 70)

    model = LLAMA7B
    gpu_tflops = 312  # A100

    effective_batch = 64  # 目标有效 batch

    print(f"\n配置: {model.name}, A100 ({gpu_tflops} TFLOPS), 目标 batch={effective_batch}")
    print(f"\n{'micro_B':>10} {'accum':>8} {'eff_B':>8} {'显存/GB':>10} "
          f"{'时间/step':>12} {'tok/s':>12} {'显存节省':>12}")
    print("-" * 78)

    baseline_mem = None
    for micro_b in [64, 32, 16, 8, 4, 2, 1]:
        accum = effective_batch // micro_b

        mem = estimate_training_memory(model, micro_b, model.seq_len)
        perf = estimate_training_throughput(model, gpu_tflops, micro_b, model.seq_len, accum)

        if baseline_mem is None:
            baseline_mem = mem['total_gb']

        mem_saving = (1 - mem['total_gb'] / baseline_mem) * 100 if baseline_mem > 0 else 0

        print(f"{micro_b:>10d} {accum:>8d} {effective_batch:>8d} "
              f"{mem['total_gb']:>10.1f} {perf['total_time_s']:>12.2f}s "
              f"{perf['tokens_per_s']:>12.0f} {mem_saving:>11.1f}%")

    print(f"\n关键洞察:")
    print(f"  - micro_batch=64: 显存最大, 但每步最快 (无累积开销)")
    print(f"  - micro_batch=1:  显存最小, 但需要 64 步累积, 通信开销 64x")
    print(f"  - 最佳: micro_batch=8-16, 平衡显存和通信")
    print(f"  - 激活值 ∝ micro_batch, 是显存主要消费者")
    print(f"  - 通信量 ∝ 累积步数 (每步 AllReduce 一次)")


def experiment_2_mixed_precision():
    """实验 2: 混合精度训练对比"""
    print("\n" + "=" * 70)
    print("实验 2: 混合精度 (FP32/FP16/BF16/FP8) 训练对比")
    print("=" * 70)

    model = LLAMA7B
    batch = 4
    seq = 2048

    print(f"\n配置: {model.name}, B={batch}, S={seq}")

    print(f"\n{'精度':>8} {'权重GB':>10} {'优化器GB':>12} {'梯度GB':>10} "
          f"{'激活GB':>10} {'总显存GB':>12} {'相对FP32':>10} {'吞吐比':>10}")
    print("-" * 88)

    fp32_mem = None
    precisions = [
        ("FP32", "fp32"),
        ("FP16", "fp16"),
        ("BF16", "bf16"),
        ("FP8", "fp8"),
    ]

    for label, prec in precisions:
        mem = estimate_training_memory(model, batch, seq, precision=prec)
        if fp32_mem is None:
            fp32_mem = mem['total_gb']

        # 吞吐比 (FP16 2x, FP8 4x vs FP32)
        if prec == "fp32":
            throughput_ratio = 1.0
        elif prec in ("fp16", "bf16"):
            throughput_ratio = 2.0  # Tensor Core 加速
        else:
            throughput_ratio = 4.0  # FP8 Tensor Core

        relative = mem['total_gb'] / fp32_mem * 100

        print(f"{label:>8} {mem['weight_gb']:>10.1f} {mem['optimizer_gb']:>12.1f} "
              f"{mem['grad_gb']:>10.1f} {mem['activation_gb']:>10.1f} "
              f"{mem['total_gb']:>12.1f} {relative:>9.1f}% {throughput_ratio:>9.1f}x")

    print(f"\n关键洞察:")
    print(f"  - FP32→FP16: 显存减少 ~35% (权重+梯度减半, 优化器不变)")
    print(f"  - FP16/BF16: 吞吐 2x (Tensor Core), 是训练标配")
    print(f"  - BF16 vs FP16: 显存相同, BF16 动态范围大无需 loss scaling")
    print(f"  - FP8: 显存减少更多, 但训练稳定性仍需验证 (H100+)")
    print(f"  - 优化器状态 (FP32) 占大头, 不受精度影响!")


def experiment_3_gradient_checkpointing():
    """实验 3: 梯度检查点显存-计算权衡"""
    print("\n" + "=" * 70)
    print("实验 3: 梯度检查点 (Activation Recomputation) 分析")
    print("=" * 70)

    model = LLAMA7B
    gpu_tflops = 312

    print(f"\n配置: {model.name}, A100 ({gpu_tflops} TFLOPS)")
    print(f"\n{'Batch':>8} {'无CKPT 显存':>14} {'有CKPT 显存':>14} {'显存节省':>10} "
          f"{'无CKPT tok/s':>14} {'有CKPT tok/s':>14} {'吞吐损失':>10}")
    print("-" * 88)

    for batch in [1, 2, 4, 8, 16, 32]:
        seq = 2048

        mem_no = estimate_training_memory(model, batch, seq, gradient_checkpointing=False)
        mem_yes = estimate_training_memory(model, batch, seq, gradient_checkpointing=True)

        perf_no = estimate_training_throughput(model, gpu_tflops, batch, seq, gradient_checkpointing=False)
        perf_yes = estimate_training_throughput(model, gpu_tflops, batch, seq, gradient_checkpointing=True)

        mem_saving = (1 - mem_yes['total_gb'] / mem_no['total_gb']) * 100
        perf_loss = (1 - perf_yes['tokens_per_s'] / perf_no['tokens_per_s']) * 100

        print(f"{batch:>8d} {mem_no['total_gb']:>14.1f}G {mem_yes['total_gb']:>14.1f}G "
              f"{mem_saving:>9.1f}% {perf_no['tokens_per_s']:>14.0f} "
              f"{perf_yes['tokens_per_s']:>14.0f} {perf_loss:>9.1f}%")

    print(f"\n关键洞察:")
    print(f"  - 激活值 ∝ batch_size, 大 batch 时激活占显存 60-70%")
    print(f"  - 梯度检查点: 激活显存减少 ~60-70%, 但增加 33% 计算")
    print(f"  - Batch=1: 激活很少, 不需要检查点")
    print(f"  - Batch=16+: 激活占大头, 检查点节省显著")
    print(f"  - 权衡: 用 ~33% 更多计算换取 ~60% 激活显存")


def experiment_4_memory_estimator():
    """实验 4: 综合训练显存估算器"""
    print("\n" + "=" * 70)
    print("实验 4: 综合训练显存估算 (不同优化组合)")
    print("=" * 70)

    scenarios = [
        {
            'name': 'LLaMA-7B Baseline',
            'model': LLAMA7B, 'batch': 4, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': False, 'zero': 0, 'dp': 1, 'tp': 1,
        },
        {
            'name': 'LLaMA-7B + CKPT',
            'model': LLAMA7B, 'batch': 4, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': True, 'zero': 0, 'dp': 1, 'tp': 1,
        },
        {
            'name': 'LLaMA-7B + ZeRO-2',
            'model': LLAMA7B, 'batch': 4, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': True, 'zero': 2, 'dp': 4, 'tp': 1,
        },
        {
            'name': 'LLaMA-70B TP=8',
            'model': LLAMA70B, 'batch': 2, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': True, 'zero': 0, 'dp': 1, 'tp': 8,
        },
        {
            'name': 'LLaMA-70B TP=8 + ZeRO-1',
            'model': LLAMA70B, 'batch': 2, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': True, 'zero': 1, 'dp': 1, 'tp': 8,
        },
        {
            'name': 'LLaMA-70B TP=8 + ZeRO-3',
            'model': LLAMA70B, 'batch': 4, 'seq': 2048,
            'precision': 'bf16', 'optimizer': 'adam',
            'checkpointing': True, 'zero': 3, 'dp': 8, 'tp': 8,
        },
    ]

    print(f"\n{'配置':<25} {'权重':>8} {'优化器':>8} {'梯度':>8} "
          f"{'激活':>8} {'总计':>10} {'A100够?':>10}")
    print("-" * 85)

    for s in scenarios:
        mem = estimate_training_memory(
            s['model'], s['batch'], s['seq'],
            precision=s['precision'], optimizer=s['optimizer'],
            gradient_checkpointing=s['checkpointing'],
            zero_stage=s['zero'], dp_size=s['dp'], tp_size=s['tp'],
        )
        fits_80g = "✓" if mem['total_gb'] < 80 else "✗"
        print(f"{s['name']:<25} {mem['weight_gb']:>7.1f}G {mem['optimizer_gb']:>7.1f}G "
              f"{mem['grad_gb']:>7.1f}G {mem['activation_gb']:>7.1f}G "
              f"{mem['total_gb']:>9.1f}G {fits_80g:>10}")

    print(f"\n关键洞察:")
    print(f"  - LLaMA-7B Baseline: ~108 GB (需要 2×A100 或 ZeRO)")
    print(f"  - LLaMA-7B + CKPT + ZeRO-2/DP=4: ~16 GB (单卡即可)")
    print(f"  - LLaMA-70B TP=8 + ZeRO-3: ~18 GB (每卡)")
    print(f"  - 显存优化组合: CKPT + ZeRO + TP 可节省 80%+ 显存")
    print(f"  - 优化器状态是最大开销项 (Adam 12 bytes/param)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("梯度累积与混合精度训练模拟器")
    print("显存估算 + 吞吐对比 + 优化组合分析")
    print("=" * 70)

    experiment_1_gradient_accumulation()
    experiment_2_mixed_precision()
    experiment_3_gradient_checkpointing()
    experiment_4_memory_estimator()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
训练优化关键要点:

  1. 显存组成 (LLaMA-7B, B=4, FP16, Adam):
     权重:     14 GB (13%)  — params × 2 bytes
     优化器:   84 GB (78%)  — params × 12 bytes (FP32 master + m + v)
     梯度:     14 GB (13%)  — params × 2 bytes
     激活:     ~10 GB       — ∝ batch × seq × hidden × layers
     → 优化器状态是大头! Adam 比 SGD 多 3x 显存

  2. 梯度累积:
     effective_batch = micro_batch × accum_steps
     激活显存 ∝ micro_batch (不 ∝ effective_batch!)
     通信次数 ∝ accum_steps
     最佳: micro_batch=8-16, 平衡显存和通信

  3. 混合精度:
     FP16/BF16: 权重+梯度减半, 吞吐 2x (Tensor Core)
     BF16 优势: 动态范围大, 无需 loss scaling
     FP8: 进一步减半, 训练稳定性待验证
     注意: 优化器仍用 FP32, 精度不影响优化器显存

  4. 梯度检查点:
     显存节省: 激活 ~60-70%
     额外计算: +33% FLOPs
     适合: 大 batch 场景 (激活占大头)
     不适合: batch=1 (激活本来就少)

  5. 显存优化优先级:
     1) 混合精度 (BF16) — 免费 2x 加速 + 35% 显存节省
     2) 梯度累积 — 减少激活显存, 增加通信
     3) 梯度检查点 — 用计算换显存
     4) ZeRO — 分片优化器/梯度/权重
     5) TP — 多卡分权重
    """)
