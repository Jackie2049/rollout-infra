"""
LLM 训练显存估算工具

计算不同配置下模型训练的显存需求，包括：
- 模型参数、梯度、优化器状态
- 激活值（含/不含 checkpointing）
- 支持 TP/PP/ZeRO 各种并行配置
- 给出最小 GPU 配置建议

用法:
    python tools/memory_estimator.py
    python tools/memory_estimator.py --params 70B --tp 4 --zero 2
    python tools/memory_estimator.py --params 7B --seq-len 8192 --batch-size 4
"""

import argparse
import math


# ── 常见模型配置 ──
MODEL_CONFIGS = {
    "1.3B": {"num_params": 1.3e9, "num_layers": 24, "hidden_size": 2048, "num_heads": 16},
    "7B": {"num_params": 7e9, "num_layers": 32, "hidden_size": 4096, "num_heads": 32},
    "13B": {"num_params": 13e9, "num_layers": 40, "hidden_size": 5120, "num_heads": 40},
    "30B": {"num_params": 30e9, "num_layers": 64, "hidden_size": 6656, "num_heads": 52},
    "70B": {"num_params": 70e9, "num_layers": 80, "hidden_size": 8192, "num_heads": 64},
    "175B": {"num_params": 175e9, "num_layers": 96, "hidden_size": 12288, "num_heads": 96},
}

GPU_CONFIGS = {
    "A100-40G": {"memory_gb": 40, "bandwidth_gbs": 2039},
    "A100-80G": {"memory_gb": 80, "bandwidth_gbs": 2039},
    "H100-80G": {"memory_gb": 80, "bandwidth_gbs": 3350},
    "H200-141G": {"memory_gb": 141, "bandwidth_gbs": 4800},
    "A16-15G": {"memory_gb": 15, "bandwidth_gbs": 320},
}


def bytes_to_gb(b):
    return b / (1024**3)


def format_gb(gb):
    if gb >= 1000:
        return f"{gb:.0f} GB"
    elif gb >= 100:
        return f"{gb:.1f} GB"
    else:
        return f"{gb:.2f} GB"


def estimate_memory(
    num_params: float,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    seq_len: int = 2048,
    batch_size: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    zero_stage: int = 0,
    activation_checkpoint: bool = False,
    mixed_precision: str = "bf16",
    optimizer: str = "adam",
):
    """估算训练显存"""

    psi = num_params  # 参数数量

    # ── 参数精度 ──
    param_bytes = 2 if mixed_precision in ("fp16", "bf16") else 4

    # ── 优化器状态 ──
    if optimizer == "adam":
        # FP32 master (4) + momentum (4) + variance (4) = 12 bytes/param
        optim_bytes = 12
    elif optimizer == "sgd":
        optim_bytes = 4  # FP32 momentum only
    else:
        optim_bytes = 0

    # ── ZeRO 分片 ──
    dp_eff = dp if dp > 1 else 1
    if zero_stage >= 3:
        param_shard = dp_eff
        grad_shard = dp_eff
        optim_shard = dp_eff
    elif zero_stage >= 2:
        param_shard = 1
        grad_shard = dp_eff
        optim_shard = dp_eff
    elif zero_stage >= 1:
        param_shard = 1
        grad_shard = 1
        optim_shard = dp_eff
    else:
        param_shard = 1
        grad_shard = 1
        optim_shard = 1

    # ── TP/PP 分片 ──
    tp_shard = tp
    pp_shard = pp

    # 每个 GPU 的参数相关显存
    params_per_gpu = psi / (tp_shard * pp_shard * param_shard)
    grads_per_gpu = psi / (tp_shard * pp_shard * grad_shard)
    optim_per_gpu = psi / (tp_shard * pp_shard * optim_shard)

    params_mem = params_per_gpu * param_bytes
    grads_mem = grads_per_gpu * param_bytes  # 梯度同参数精度
    optim_mem = optim_per_gpu * optim_bytes

    model_mem = params_mem + grads_mem + optim_mem

    # ── 激活值估算 ──
    # 每个 Transformer 层的激活 ≈ 11 * b * s * h bytes (FP16)
    b = batch_size
    s = seq_len
    h = hidden_size

    # TP 减少激活（每个 GPU 只处理部分 head）
    h_per_gpu = h // tp

    act_per_layer = 11 * b * s * h_per_gpu * 2  # 2 bytes (FP16)

    if activation_checkpoint:
        # sqrt(n) 层保存完整激活
        layers_to_save = int(math.sqrt(num_layers))
        # 其余层只保存检查点（很小）
        act_total = layers_to_save * act_per_layer
    else:
        act_total = num_layers * act_per_layer

    # PP 分阶段 — 每个 stage 只有一部分的激活
    layers_per_stage = num_layers // pp
    act_per_gpu = act_total * (layers_per_stage / num_layers)

    # ── 总计 ──
    total_per_gpu = model_mem + act_per_gpu

    return {
        "params_mem": params_mem,
        "grads_mem": grads_mem,
        "optim_mem": optim_mem,
        "model_mem": model_mem,
        "act_mem": act_per_gpu,
        "total_mem": total_per_gpu,
        "params_per_gpu_b": params_per_gpu,
        "num_params": psi,
    }


def suggest_gpu(memory_gb, gpu_configs=GPU_CONFIGS):
    """建议 GPU 配置"""
    suggestions = []
    for name, config in gpu_configs.items():
        if config["memory_gb"] >= memory_gb:
            suggestions.append((name, config["memory_gb"]))
    return suggestions


def main():
    parser = argparse.ArgumentParser(description="LLM 训练显存估算工具")
    parser.add_argument("--params", default="7B", help="模型大小 (1.3B/7B/13B/30B/70B/175B)")
    parser.add_argument("--seq-len", type=int, default=2048, help="序列长度")
    parser.add_argument("--batch-size", type=int, default=1, help="每 GPU batch size")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallelism degree")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallelism degree")
    parser.add_argument("--zero", type=int, default=0, help="ZeRO stage (0/1/2/3)")
    parser.add_argument("--no-ckpt", action="store_true", help="不做 activation checkpointing")
    parser.add_argument("--fp32", action="store_true", help="FP32 训练")
    parser.add_argument("--sgd", action="store_true", help="使用 SGD 而非 Adam")
    parser.add_argument("--compare", action="store_true", help="对比不同配置")
    args = parser.parse_args()

    if args.params in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[args.params]
    else:
        print(f"未知模型: {args.params}")
        print(f"可选: {', '.join(MODEL_CONFIGS.keys())}")
        return

    mp = "fp32" if args.fp32 else "bf16"
    opt = "sgd" if args.sgd else "adam"

    if args.compare:
        print("=" * 80)
        print(f"  模型: {args.params} | Seq: {args.seq_len} | Batch/GPU: {args.batch_size} | {mp}")
        print("=" * 80)
        print()

        configs = [
            ("Baseline (DDP)", 1, 1, 1, 0, False),
            ("+ TP=2", 2, 1, 1, 0, False),
            ("+ ZeRO-1", 1, 1, 8, 1, False),
            ("+ ZeRO-2", 1, 1, 8, 2, False),
            ("+ ZeRO-3", 1, 1, 8, 3, False),
            ("+ Act Ckpt", 1, 1, 1, 0, True),
            ("+ TP=2 ZeRO-2", 2, 1, 8, 2, True),
            ("+ TP=4 ZeRO-2", 4, 1, 8, 2, True),
        ]

        print(f"  {'配置':<25} {'参数':>10} {'梯度':>10} {'优化器':>10} {'激活':>10} {'总计':>10} {'GPU数':>6}")
        print("  " + "-" * 85)

        for name, tp, pp, dp, zero, ckpt in configs:
            r = estimate_memory(
                cfg["num_params"], cfg["num_layers"], cfg["hidden_size"], cfg["num_heads"],
                args.seq_len, args.batch_size, tp=tp, pp=pp, dp=dp,
                zero_stage=zero, activation_checkpoint=ckpt, mixed_precision=mp, optimizer=opt,
            )
            total_gpus = tp * pp * dp
            print(f"  {name:<25} {format_gb(bytes_to_gb(r['params_mem'])):>10} "
                  f"{format_gb(bytes_to_gb(r['grads_mem'])):>10} {format_gb(bytes_to_gb(r['optim_mem'])):>10} "
                  f"{format_gb(bytes_to_gb(r['act_mem'])):>10} {format_gb(bytes_to_gb(r['total_mem'])):>10} "
                  f"{total_gpus:>6}")

        print()
    else:
        # 单一配置估算
        r = estimate_memory(
            cfg["num_params"], cfg["num_layers"], cfg["hidden_size"], cfg["num_heads"],
            args.seq_len, args.batch_size, args.tp, args.pp, args.dp,
            args.zero, not args.no_ckpt, mp, opt,
        )

        total_gb = bytes_to_gb(r["total_mem"])
        total_gpus = args.tp * args.pp * args.dp

        print("=" * 60)
        print(f"  LLM 训练显存估算")
        print("=" * 60)
        print(f"  模型:         {args.params} ({cfg['num_params']/1e9:.1f}B params)")
        print(f"  Layers:       {cfg['num_layers']}")
        print(f"  Hidden:       {cfg['hidden_size']}")
        print(f"  Heads:        {cfg['num_heads']}")
        print(f"  Seq length:   {args.seq_len}")
        print(f"  Batch/GPU:    {args.batch_size}")
        print(f"  Precision:    {mp}")
        print(f"  Optimizer:    {opt}")
        print(f"  TP×PP×DP:     {args.tp}×{args.pp}×{args.dp} = {total_gpus} GPUs")
        print(f"  ZeRO stage:   {args.zero}")
        print(f"  Act Ckpt:     {'No' if args.no_ckpt else 'Yes'}")
        print()
        print(f"  ── 显存 Breakdown (per GPU) ──")
        print(f"  模型参数:     {format_gb(bytes_to_gb(r['params_mem'])):>10}")
        print(f"  梯度:         {format_gb(bytes_to_gb(r['grads_mem'])):>10}")
        print(f"  优化器状态:   {format_gb(bytes_to_gb(r['optim_mem'])):>10}")
        print(f"  ──────────────────────")
        print(f"  模型合计:     {format_gb(bytes_to_gb(r['model_mem'])):>10}")
        print(f"  激活值:       {format_gb(bytes_to_gb(r['act_mem'])):>10}")
        print(f"  ══════════════════════")
        print(f"  总计:         {format_gb(total_gb):>10}")
        print()

        suggestions = suggest_gpu(total_gb)
        if suggestions:
            print(f"  推荐 GPU:")
            for name, mem in suggestions:
                print(f"    {name} ({mem} GB) — 余量 {format_gb(mem - total_gb)}")
        else:
            print(f"  单 GPU 不足，建议增加并行度或使用 ZeRO-3")

        print("=" * 60)


if __name__ == "__main__":
    main()
