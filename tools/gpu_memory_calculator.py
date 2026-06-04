#!/usr/bin/env python3
"""GPU 显存计算器 — 给定模型配置，精确计算训练/推理所需显存

计算项:
1. 模型参数内存 (FP16/FP32/BF16)
2. 优化器状态 (Adam: m + v, FP32)
3. 梯度内存
4. Activation 内存 (per layer)
5. KV Cache 内存 (推理)
6. ZeRO 优化后的内存
7. 最大可训练模型估算

用法 (GPU 服务器 or 本地):
  python gpu_memory_calculator.py
"""

import math
from collections import OrderedDict

print("=" * 60)
print("GPU 显存计算器")
print("=" * 60)


def format_mem(bytes_val):
    """Format bytes to human readable"""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f} MB"
    elif bytes_val < 1024**4:
        return f"{bytes_val/1024**3:.2f} GB"
    else:
        return f"{bytes_val/1024**4:.2f} TB"


def calc_model_mem(n_params, dtype_bytes=2):
    """Model parameter memory"""
    return n_params * dtype_bytes

def calc_optimizer_mem(n_params, optimizer="adam", fp32=True):
    """Optimizer state memory"""
    if optimizer == "adam":
        # m (FP32) + v (FP32) = 8 bytes/param
        return n_params * 8 if fp32 else n_params * 4
    elif optimizer == "sgd":
        return 0  # no additional state
    return 0

def calc_grad_mem(n_params, dtype_bytes=2):
    """Gradient memory"""
    return n_params * dtype_bytes

def calc_master_params(n_params):
    """FP32 master parameters (for mixed precision)"""
    return n_params * 4

def calc_activation_mem(B, S, H, n_layers, dtype_bytes=2):
    """Activation memory per layer (simplified)"""
    # Each layer stores: input + attention output + MLP intermediate
    # Attention: B * S * H (input) + B * H * S * S (attention matrix, not stored with FlashAttn)
    # MLP: B * S * 4H (intermediate)
    per_layer = B * S * H * dtype_bytes * 4  # input, attn_out, mlp_in, mlp_out
    per_layer += B * S * 4 * H * dtype_bytes  # MLP intermediate
    return per_layer * n_layers

def calc_kv_cache(B, S, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """KV Cache memory for inference"""
    per_layer = 2 * B * n_kv_heads * S * head_dim * dtype_bytes  # K + V
    return per_layer * n_layers


# ============================================================
# GPT 模型配置
# ============================================================

models = OrderedDict([
    ("OPT-125M", {"n_params": 125e6, "n_layers": 12, "hidden": 768, "n_heads": 12, "n_kv_heads": 12, "head_dim": 64}),
    ("OPT-350M", {"n_params": 350e6, "n_layers": 24, "hidden": 1024, "n_heads": 16, "n_kv_heads": 16, "head_dim": 64}),
    ("LLaMA-1.3B", {"n_params": 1.3e9, "n_layers": 24, "hidden": 2048, "n_heads": 32, "n_kv_heads": 32, "head_dim": 64}),
    ("LLaMA-7B", {"n_params": 7e9, "n_layers": 32, "hidden": 4096, "n_heads": 32, "n_kv_heads": 32, "head_dim": 128}),
    ("LLaMA-13B", {"n_params": 13e9, "n_layers": 40, "hidden": 5120, "n_heads": 40, "n_kv_heads": 40, "head_dim": 128}),
    ("LLaMA-70B", {"n_params": 70e9, "n_layers": 80, "hidden": 8192, "n_heads": 64, "n_kv_heads": 8, "head_dim": 128}),
    ("Mixtral-8x7B", {"n_params": 46.7e9, "n_layers": 32, "hidden": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128}),
    ("DeepSeek-V2", {"n_params": 21e9, "n_layers": 60, "hidden": 5120, "n_heads": 128, "n_kv_heads": 1, "head_dim": 512}),
])

gpus = OrderedDict([
    ("A16 15GB", 15.6e9),
    ("A100 40GB", 40e9),
    ("A100 80GB", 80e9),
    ("L40S 48GB", 48e9),
    ("H100 80GB", 80e9),
    ("H200 141GB", 141e9),
])


# ============================================================
# 1. 训练内存分析
# ============================================================

print("\n" + "=" * 60)
print("1. 训练内存分析 (Adam + FP16 mixed precision)")
print("=" * 60)

print(f"\n  内存组成: params(FP16) + grads(FP16) + master(FP32) + Adam(m+v FP32)")
print(f"  = 2 + 2 + 4 + 8 = 16 bytes/param\n")

print(f"  {'Model':<15} {'Params':<10} {'Params MB':<12} {'Total Train GB':<16} {'Adam 占比'}")
print("  " + "-" * 65)

for name, cfg in models.items():
    P = cfg["n_params"]
    params_mb = calc_model_mem(P, 2)
    grads_mb = calc_grad_mem(P, 2)
    master_mb = calc_master_params(P)
    adam_mb = calc_optimizer_mem(P)
    total = params_mb + grads_mb + master_mb + adam_mb
    adam_pct = adam_mb / total * 100

    print(f"  {name:<15} {P/1e9:<10.1f}B {params_mb/1e6:<12.0f} {format_mem(total):<16} {adam_pct:.0f}%")

# ============================================================
# 2. ZeRO 优化效果
# ============================================================

print("\n" + "=" * 60)
print("2. ZeRO 优化 — 每 rank 内存 (DP=8)")
print("=" * 60)

print(f"\n  {'Model':<15} {'No ZeRO':<14} {'ZeRO-1':<14} {'ZeRO-2':<14} {'ZeRO-3':<14}")
print("  " + "-" * 71)

DP = 8
for name, cfg in models.items():
    P = cfg["n_params"]
    params = calc_model_mem(P, 2)
    grads = calc_grad_mem(P, 2)
    master = calc_master_params(P)
    adam = calc_optimizer_mem(P)
    total = params + grads + master + adam

    # ZeRO-1: shard optimizer
    z1 = params + grads + master + adam / DP
    # ZeRO-2: shard optimizer + grads
    z2 = params + grads / DP + adam / DP
    # ZeRO-3: shard everything
    z3 = (params + grads + master + adam) / DP

    print(f"  {name:<15} {format_mem(total):<14} {format_mem(z1):<14} {format_mem(z2):<14} {format_mem(z3):<14}")

# ============================================================
# 3. 推理内存 (Weights + KV Cache)
# ============================================================

print("\n" + "=" * 60)
print("3. 推理内存 — Weights + KV Cache")
print("=" * 60)

B, S = 1, 4096  # batch=1, seq=4096

print(f"\n  Batch=1, Seq=4096, FP16")
print(f"  {'Model':<15} {'Weights':<12} {'KV Cache':<14} {'Total':<14} {'KV/Total':<10}")
print("  " + "-" * 65)

for name, cfg in models.items():
    weights = calc_model_mem(cfg["n_params"], 2)
    kv = calc_kv_cache(B, S, cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], 2)
    total = weights + kv
    kv_pct = kv / total * 100

    print(f"  {name:<15} {format_mem(weights):<12} {format_mem(kv):<14} {format_mem(total):<14} {kv_pct:.0f}%")

# Different seq lengths
print(f"\n  KV Cache vs 序列长度:")
print(f"  {'Model':<15} {'S=512':<10} {'S=2K':<10} {'S=8K':<10} {'S=32K':<10} {'S=128K':<10}")
print("  " + "-" * 65)

for name, cfg in models.items():
    line = f"  {name:<15}"
    for S in [512, 2048, 8192, 32768, 131072]:
        kv = calc_kv_cache(1, S, cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"], 2)
        line += f" {format_mem(kv):<10}"
    print(line)

# ============================================================
# 4. GPU 可训练最大模型
# ============================================================

print("\n" + "=" * 60)
print("4. 各 GPU 最大可训练模型 (FP16 + Adam, ZeRO-3)")
print("=" * 60)

print(f"\n  训练内存 = 16 bytes/param (无 ZeRO)")
print(f"  ZeRO-3(DP=N) = 16/N bytes/param\n")

print(f"  {'GPU':<18} {'VRAM':<10} {'No ZeRO':<14} {'DP=4':<14} {'DP=8':<14} {'DP=64':<14}")
print("  " + "-" * 74)

for gpu_name, vram in gpus.items():
    # Usable VRAM (reserve 10% for system)
    usable = vram * 0.9

    # Max params without ZeRO
    max_no_zero = usable / 16
    # With ZeRO-3
    max_dp4 = usable / (16/4)
    max_dp8 = usable / (16/8)
    max_dp64 = usable / (16/64)

    def fmt_params(p):
        if p >= 1e9:
            return f"{p/1e9:.1f}B"
        return f"{p/1e6:.0f}M"

    print(f"  {gpu_name:<18} {format_mem(vram):<10} {fmt_params(max_no_zero):<14} "
          f"{fmt_params(max_dp4):<14} {fmt_params(max_dp8):<14} {fmt_params(max_dp64):<14}")

# ============================================================
# 5. Activation 内存分析
# ============================================================

print("\n" + "=" * 60)
print("5. Activation 内存 (per layer, B=1)")
print("=" * 60)

print(f"\n  {'Model':<15} {'S=512':<12} {'S=2K':<12} {'S=8K':<12} {'S=32K':<12}")
print("  " + "-" * 63)

for name, cfg in models.items():
    line = f"  {name:<15}"
    for S in [512, 2048, 8192, 32768]:
        act = calc_activation_mem(1, S, cfg["hidden"], 1, 2)  # per layer
        line += f" {format_mem(act):<12}"
    print(line)

# ============================================================
# 6. 实际 GPU 建议
# ============================================================

print("\n" + "=" * 60)
print("6. 训练配置建议")
print("=" * 60)

configs = [
    ("LLaMA-7B", "A100×8", "TP=1, DP=8, ZeRO-3", "14GB/rank"),
    ("LLaMA-13B", "A100×8", "TP=2, DP=4, ZeRO-3", "13GB/rank"),
    ("LLaMA-70B", "H100×64", "TP=8, DP=8, ZeRO-3", "18GB/rank"),
    ("Mixtral-8x7B", "H100×16", "EP=8, DP=2", "active 2/8 experts"),
    ("DeepSeek-V2", "H100×128", "TP=8, EP=8, DP=2", "MLA 56x KV compression"),
]

print(f"\n  {'Model':<18} {'Hardware':<12} {'Strategy':<25} {'Notes'}")
print("  " + "-" * 75)
for model, hw, strategy, notes in configs:
    print(f"  {model:<18} {hw:<12} {strategy:<25} {notes}")

print("\nDone!")
