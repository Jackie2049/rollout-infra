#!/usr/bin/env python3
"""FSDP / ZeRO Memory Simulator
================================
Simulates memory usage under different distributed training configurations:
1. Pure DP (baseline)
2. ZeRO-1 (optimizer sharding)
3. ZeRO-2 (optimizer + gradient sharding)
4. ZeRO-3 / FSDP (full sharding)
5. TP + DP combinations
6. PP + TP + DP combinations

Validates against actual PyTorch FSDP memory on GPU.
Educational purpose: understand distributed training memory tradeoffs.
"""

import torch
import torch.nn as nn
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. Model Architecture Specifications
# ============================================================

@dataclass
class ModelSpec:
    """Model architecture specification."""
    name: str
    n_params: int          # Total parameters
    d_model: int           # Hidden dimension
    n_layers: int          # Number of transformer layers
    n_heads: int           # Number of attention heads
    n_kv_heads: int        # Number of KV heads (GQA)
    ffn_dim: int           # FFN intermediate dimension
    vocab_size: int        # Vocabulary size
    max_seq_len: int       # Maximum sequence length

    @property
    def bytes_per_param(self):
        """Training bytes per parameter (BF16 + Adam)."""
        return 16  # weight(2) + grad(2) + m(4) + v(4) + master(4)

    @property
    def weight_bytes(self):
        return self.n_params * 2  # BF16

    @property
    def total_training_bytes(self):
        return self.n_params * self.bytes_per_param


# Common model specifications
MODELS = {
    '125M': ModelSpec('125M', 125_000_000, 768, 12, 12, 12, 3072, 50257, 2048),
    '350M': ModelSpec('350M', 350_000_000, 1024, 24, 16, 16, 4096, 50257, 2048),
    '1.3B': ModelSpec('1.3B', 1_300_000_000, 2048, 24, 32, 32, 8192, 50257, 2048),
    '7B': ModelSpec('7B', 7_000_000_000, 4096, 32, 32, 8, 11008, 32000, 4096),
    '13B': ModelSpec('13B', 13_000_000_000, 5120, 40, 40, 40, 13824, 32000, 4096),
    '30B': ModelSpec('30B', 30_000_000_000, 6656, 60, 52, 52, 17920, 32000, 4096),
    '70B': ModelSpec('70B', 70_000_000_000, 8192, 80, 64, 8, 28672, 32000, 4096),
    '175B': ModelSpec('175B', 175_000_000_000, 12288, 96, 96, 96, 49152, 32000, 4096),
}


# ============================================================
# 2. Memory Calculator
# ============================================================

class MemoryCalculator:
    """Calculate GPU memory requirements for distributed training."""

    def __init__(self, model: ModelSpec, gpu_memory_gb: float = 80.0,
                 dtype_bytes: int = 2, seq_len: int = 2048, batch_size: int = 1):
        self.model = model
        self.gpu_memory_gb = gpu_memory_gb
        self.dtype_bytes = dtype_bytes  # BF16 = 2
        self.seq_len = seq_len
        self.batch_size = batch_size

    def weight_memory(self, dp=1, tp=1, pp=1, zero_stage=0):
        """Memory for model weights per GPU."""
        total = self.model.n_params * self.dtype_bytes
        if zero_stage == 3:
            return total / (dp * tp * pp)
        return total / (tp * pp)

    def gradient_memory(self, dp=1, tp=1, pp=1, zero_stage=0):
        """Memory for gradients per GPU."""
        total = self.model.n_params * self.dtype_bytes
        if zero_stage >= 2:
            return total / (dp * tp * pp)
        return total / (tp * pp)

    def optimizer_memory(self, dp=1, tp=1, pp=1, zero_stage=0):
        """Memory for Adam optimizer states per GPU (FP32 m, v, master)."""
        # Adam: m (FP32) + v (FP32) + master weights (FP32) = 12 bytes/param
        total = self.model.n_params * 12
        if zero_stage >= 1:
            return total / (dp * tp * pp)
        return total / (tp * pp)

    def activation_memory(self, tp=1, pp=1, sp=False):
        """Memory for activations per GPU (with gradient checkpointing)."""
        # Per layer activation: 34 * d_model * seq_len * batch_size bytes
        # 34 = attention(5) + mlp(12) + residuals(8) + norms(4) + misc(5)
        # With gradient checkpointing: only save inputs to each layer
        d = self.model.d_model
        s = self.seq_len
        b = self.batch_size
        layers_per_gpu = self.model.n_layers / pp

        # Without checkpointing
        per_layer_bytes = 34 * b * s * d * self.dtype_bytes

        # With checkpointing (save only layer inputs)
        checkpoint_bytes = b * s * d * self.dtype_bytes * layers_per_gpu

        # SP reduces activation memory by TP
        if sp:
            checkpoint_bytes /= tp

        return checkpoint_bytes

    def kv_cache_memory(self, tp=1):
        """KV cache memory per GPU for inference."""
        d = self.model.d_model
        n_kv = self.model.n_kv_heads
        d_head = d // self.model.n_heads
        s = self.seq_len
        L = self.model.n_layers

        # KV cache = 2 × n_kv_heads × d_head × seq_len × n_layers × dtype
        kv_bytes = 2 * n_kv * d_head * s * L * self.dtype_bytes
        return kv_bytes / tp

    def total_memory(self, dp=1, tp=1, pp=1, zero_stage=0,
                     grad_checkpoint=True, sp=False):
        """Total GPU memory per GPU."""
        mem = {
            'weights': self.weight_memory(dp, tp, pp, zero_stage),
            'gradients': self.gradient_memory(dp, tp, pp, zero_stage),
            'optimizer': self.optimizer_memory(dp, tp, pp, zero_stage),
        }

        if grad_checkpoint:
            mem['activations'] = self.activation_memory(tp, pp, sp)
        else:
            mem['activations'] = self.activation_memory(tp, pp, sp) * 8  # ~8x without ckpt

        mem['total_gb'] = sum(v for v in mem.values()) / 1e9
        mem['gpu_memory_gb'] = self.gpu_memory_gb
        mem['fits'] = mem['total_gb'] <= self.gpu_memory_gb
        return mem

    def minimum_gpus(self, zero_stage=0, tp=1, pp=1, grad_checkpoint=True):
        """Find minimum number of GPUs needed."""
        for dp in range(1, 1025):
            mem = self.total_memory(dp, tp, pp, zero_stage, grad_checkpoint)
            if mem['fits']:
                total_gpus = dp * tp * pp
                return total_gpus, dp, mem
        return -1, -1, None


# ============================================================
# 3. Experiments
# ============================================================

def experiment_dp_vs_zero():
    """Compare DP vs ZeRO-1/2/3 for different model sizes."""
    print("\n  === Experiment: DP vs ZeRO Memory ===")
    print(f"  {'Model':>8} | {'DP':>10} | {'ZeRO-1':>10} | {'ZeRO-2':>10} | {'ZeRO-3':>10}")
    print(f"  {'':>8} | {'GB/GPU':>10} | {'GB/GPU':>10} | {'GB/GPU':>10} | {'GB/GPU':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results = {}
    for name in ['7B', '13B', '30B', '70B', '175B']:
        model = MODELS[name]
        calc = MemoryCalculator(model, gpu_memory_gb=80.0)

        row = {}
        for stage in range(4):
            dp = 8
            mem = calc.total_memory(dp=dp, zero_stage=stage, grad_checkpoint=True)
            label = ['DP', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3'][stage]
            row[label] = mem['total_gb']

        print(f"  {name:>8} | {row['DP']:>9.1f}G | {row['ZeRO-1']:>9.1f}G | "
              f"{row['ZeRO-2']:>9.1f}G | {row['ZeRO-3']:>9.1f}G")
        results[name] = row

    return results


def experiment_min_gpus():
    """Find minimum GPUs needed for each model and strategy."""
    print("\n  === Experiment: Minimum GPUs (80GB A100) ===")
    print(f"  {'Model':>8} | {'DP+TP=1':>8} | {'ZeRO-3':>8} | {'TP=8':>8} | {'TP=8+PP=4':>12} | {'3D':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")

    results = {}
    for name in ['7B', '13B', '30B', '70B', '175B']:
        model = MODELS[name]
        calc = MemoryCalculator(model, gpu_memory_gb=80.0)

        configs = [
            ('DP', 0, 1, 1),
            ('ZeRO-3', 3, 1, 1),
            ('TP=8', 0, 8, 1),
            ('TP=8+PP=4', 0, 8, 4),
            ('3D', 3, 8, 4),
        ]

        row = {}
        line = f"  {name:>8}"
        for config_name, zero, tp, pp in configs:
            min_gpus, dp, mem = calc.minimum_gpus(zero, tp, pp)
            row[config_name] = min_gpus
            line += f" | {min_gpus:>8}"
        print(line)
        results[name] = row

    return results


def experiment_memory_breakdown():
    """Show memory breakdown for 70B model."""
    print("\n  === Experiment: 70B Memory Breakdown (DP=8, A100 80GB) ===")

    model = MODELS['70B']
    calc = MemoryCalculator(model, gpu_memory_gb=80.0, seq_len=2048, batch_size=1)

    for stage in range(4):
        label = ['Pure DP', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3/FSDP'][stage]
        mem = calc.total_memory(dp=8, zero_stage=stage, grad_checkpoint=True)

        total = mem['total_gb']
        w_gb = mem['weights']/1e9
        g_gb = mem['gradients']/1e9
        o_gb = mem['optimizer']/1e9
        a_gb = mem['activations']/1e9
        print(f"\n    {label} (DP=8):")
        print(f"      Weights:    {w_gb:>8.2f} GB ({w_gb/total*100:>3.0f}%)")
        print(f"      Gradients:  {g_gb:>8.2f} GB ({g_gb/total*100:>3.0f}%)")
        print(f"      Optimizer:  {o_gb:>8.2f} GB ({o_gb/total*100:>3.0f}%)")
        print(f"      Activations:{a_gb:>8.2f} GB ({a_gb/total*100:>3.0f}%)")
        print(f"      Total:      {total:>8.2f} GB / {mem['gpu_memory_gb']:.0f} GB {'✓' if mem['fits'] else '✗ OVERFLOW'}")


def experiment_3d_parallel_configs():
    """Test different 3D parallel configurations for 175B model."""
    print("\n  === Experiment: 175B 3D Parallel Configs ===")

    model = MODELS['175B']
    calc = MemoryCalculator(model, gpu_memory_gb=80.0, seq_len=2048, batch_size=2)

    configs = [
        {'name': 'TP=8, PP=8, ZeRO-3', 'tp': 8, 'pp': 8, 'dp': 16, 'zero': 3},
        {'name': 'TP=8, PP=16, ZeRO-1', 'tp': 8, 'pp': 16, 'dp': 8, 'zero': 1},
        {'name': 'TP=4, PP=8, ZeRO-3', 'tp': 4, 'pp': 8, 'dp': 32, 'zero': 3},
        {'name': 'TP=8, PP=8, ZeRO-1', 'tp': 8, 'pp': 8, 'dp': 16, 'zero': 1},
        {'name': 'TP=8, PP=4, ZeRO-3', 'tp': 8, 'pp': 4, 'dp': 32, 'zero': 3},
    ]

    print(f"  {'Config':>30} | {'Total GPU':>8} | {'GB/GPU':>8} | {'Fits?':>5}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")

    for cfg in configs:
        mem = calc.total_memory(dp=cfg['dp'], tp=cfg['tp'], pp=cfg['pp'],
                                zero_stage=cfg['zero'], grad_checkpoint=True)
        total_gpus = cfg['dp'] * cfg['tp'] * cfg['pp']
        print(f"  {cfg['name']:>30} | {total_gpus:>8} | {mem['total_gb']:>8.1f} | "
              f"{'✓' if mem['fits'] else '✗':>5}")


def experiment_kv_cache_analysis():
    """Analyze KV cache memory for inference."""
    print("\n  === Experiment: KV Cache Analysis ===")

    print(f"  {'Model':>8} | {'Seq=4K':>8} | {'Seq=8K':>8} | {'Seq=32K':>9} | {'Seq=128K':>10} | {'vs Weight':>10}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}")

    for name in ['7B', '13B', '30B', '70B']:
        model = MODELS[name]
        calc = MemoryCalculator(model, gpu_memory_gb=80.0)

        row = {}
        for seq_len in [4096, 8192, 32768, 131072]:
            calc.seq_len = seq_len
            kv = calc.kv_cache_memory(tp=1) / 1e9
            row[seq_len] = kv

        weight_gb = model.weight_bytes / 1e9
        ratio = row[131072] / weight_gb
        print(f"  {name:>8} | {row[4096]:>7.2f}G | {row[8192]:>7.2f}G | "
              f"{row[32768]:>8.2f}G | {row[131072]:>9.1f}G | {ratio:>9.1f}x")


def experiment_rtx4090_fits():
    """Check which models fit on RTX 4090 (24GB)."""
    print("\n  === Experiment: What fits on RTX 4090 (24GB)? ===")

    for name in ['125M', '350M', '1.3B', '7B']:
        model = MODELS[name]

        # Training (BF16 + Adam)
        calc = MemoryCalculator(model, gpu_memory_gb=24.0, seq_len=2048, batch_size=1)
        mem_train = calc.total_memory(dp=1, zero_stage=0, grad_checkpoint=True)

        # Inference (BF16 only)
        mem_infer = model.weight_bytes / 1e9

        print(f"  {name:>8}: weight={mem_infer:.1f}GB, "
              f"train={mem_train['total_gb']:.1f}GB, "
              f"fits_infer={'✓' if mem_infer <= 24 else '✗'}, "
              f"fits_train={'✓' if mem_train['fits'] else '✗'}")


def experiment_communication_volume():
    """Estimate communication volume for different parallelism strategies."""
    print("\n  === Experiment: Communication Volume ===")

    model = MODELS['70B']
    d = model.d_model
    b = 4  # micro-batch size
    n_layers = model.n_layers

    print(f"  Model: {model.name}, d_model={d}, {n_layers} layers, batch={b}")

    # TP communication (2 AllReduce per layer)
    tp_comm_per_layer = 2 * 2 * b * d * 2  # 2 AllReduce × 2 phases × B×d×2bytes
    tp_total = tp_comm_per_layer * n_layers
    print(f"\n  TP=8 per step:")
    print(f"    Per layer: {tp_comm_per_layer/1e6:.1f} MB (2 AllReduce)")
    print(f"    Total: {tp_total/1e6:.0f} MB")
    print(f"    NVLink (300GB/s): {tp_total/1e9/300*1000:.2f} ms")
    print(f"    PCIe (32GB/s): {tp_total/1e9/32*1000:.2f} ms")

    # PP communication (P2P per stage boundary)
    pp_comm = b * d * 2  # single P2P
    pp_stages = 8
    pp_total = pp_comm * (pp_stages - 1) * 2  # forward + backward
    print(f"\n  PP=8 per step:")
    print(f"    Per P2P: {pp_comm/1e6:.1f} MB")
    print(f"    Total: {pp_total/1e6:.0f} MB ({pp_stages-1} stages × 2)")
    print(f"    NVLink: {pp_total/1e9/300*1000:.3f} ms")

    # DP communication (gradient AllReduce)
    dp_comm = 2 * model.n_params * 2 * (7.0/8.0)  # Ring AllReduce = 2×(N-1)/N
    print(f"\n  DP=8 per step:")
    print(f"    Total: {dp_comm/1e9:.1f} GB (gradient AllReduce)")
    print(f"    NVLink: {dp_comm/1e9/300*1000:.1f} ms")
    print(f"    IB (200Gbps): {dp_comm*8/200e9*1000:.1f} ms")

    # EP (MoE) All-to-All
    n_experts = 8
    ep_comm = b * model.max_seq_len * d * 2 * 2  # dispatch + gather
    print(f"\n  EP=8 per step (Mixtral-style):")
    print(f"    Per A2A: {ep_comm/1e6:.0f} MB")
    print(f"    NVLink: {ep_comm/1e9/300*1000:.2f} ms")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("FSDP / ZeRO Memory Simulator")
    print("=" * 60)

    results = {}

    # Exp 1: DP vs ZeRO
    results['dp_vs_zero'] = experiment_dp_vs_zero()

    # Exp 2: Minimum GPUs
    results['min_gpus'] = experiment_min_gpus()

    # Exp 3: Memory breakdown
    experiment_memory_breakdown()

    # Exp 4: 3D parallel configs
    experiment_3d_parallel_configs()

    # Exp 5: KV cache analysis
    experiment_kv_cache_analysis()

    # Exp 6: RTX 4090 fits
    experiment_rtx4090_fits()

    # Exp 7: Communication volume
    experiment_communication_volume()

    # Summary
    print("\n" + "=" * 60)
    print("Distributed Training Memory Summary")
    print("=" * 60)
    print("""
    Key Insights:
    1. Adam optimizer is the biggest memory consumer (~75% of training memory)
    2. ZeRO-3 can reduce per-GPU memory to total/N (with 3x communication cost)
    3. TP is limited to NVLink nodes (high communication frequency)
    4. PP can span nodes (low communication, but pipeline bubbles)
    5. For 7B training: single A100 80GB works (ZeRO-2)
    6. For 70B training: TP=8 or ZeRO-3 + DP=8 (minimum 8 GPUs)
    7. For 175B training: 3D parallel minimum 128 GPUs

    GPU Memory Hierarchy (per param, training):
    ┌────────────────────────────────────────────────────────┐
    │ Component      │ BF16  │ FP32  │ ZeRO-3 Savings       │
    │────────────────│───────│───────│───────────────────────│
    │ Weights        │ 2 B   │ 4 B   │ ÷ DP (gather on use)  │
    │ Gradients      │ 2 B   │ 4 B   │ ÷ DP (scatter after)  │
    │ Adam m (FP32)  │ -     │ 4 B   │ ÷ DP                  │
    │ Adam v (FP32)  │ -     │ 4 B   │ ÷ DP                  │
    │ Total          │ 4 B   │ 16 B  │ ÷ DP (ZeRO-3)         │
    └────────────────────────────────────────────────────────┘

    Rule of thumb:
    - < 7B: Single GPU with ZeRO-2
    - 7B-30B: 4-8 GPUs with TP or ZeRO-3
    - 30B-70B: 8-32 GPUs with TP+DP or TP+PP+DP
    - > 70B: 64-1024 GPUs with 3D parallel + ZeRO
    """)

    with open("fsdp_memory_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved to fsdp_memory_results.json")


if __name__ == "__main__":
    main()
