#!/usr/bin/env python3
"""训练启动命令生成器 — 一键生成分布式训练配置

用法:
  python training_recipe.py --model 7b --gpus 4 --gpu-type a100
  python training_recipe.py --model 70b --gpus 32 --gpu-type a100 --zero 3
  python training_recipe.py --model 13b --gpus 8 --gpu-type a16 --framework deepspeed
  python training_recipe.py --list-models
  python training_recipe.py --list-gpus
"""

import argparse
import math
import json

# ============================================================
# 模型定义
# ============================================================

MODELS = {
    "1.3b": {
        "name": "GPT-1.3B",
        "num_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_kv_heads": 16,
        "vocab_size": 32000,
        "params_b": 1.3,
        "max_seq_len": 2048,
    },
    "7b": {
        "name": "LLaMA-7B",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "vocab_size": 32000,
        "params_b": 7.0,
        "max_seq_len": 4096,
    },
    "7b-gqa": {
        "name": "LLaMA-7B-GQA",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 32000,
        "params_b": 7.0,
        "max_seq_len": 4096,
    },
    "13b": {
        "name": "LLaMA-13B",
        "num_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "vocab_size": 32000,
        "params_b": 13.0,
        "max_seq_len": 4096,
    },
    "70b": {
        "name": "LLaMA-70B",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "vocab_size": 32000,
        "params_b": 70.0,
        "max_seq_len": 4096,
    },
    "175b": {
        "name": "GPT-175B",
        "num_layers": 96,
        "hidden_size": 12288,
        "num_attention_heads": 96,
        "num_kv_heads": 96,
        "vocab_size": 51200,
        "params_b": 175.0,
        "max_seq_len": 2048,
    },
}

GPU_TYPES = {
    "a100-40": {"name": "A100-40GB", "memory_gb": 40, "bandwidth": 1555},
    "a100":    {"name": "A100-80GB", "memory_gb": 80, "bandwidth": 2039},
    "h100":    {"name": "H100-80GB", "memory_gb": 80, "bandwidth": 3350},
    "a16":     {"name": "A16-15GB",  "memory_gb": 15, "bandwidth": 300},
    "3090":    {"name": "RTX-3090",  "memory_gb": 24, "bandwidth": 936},
    "4090":    {"name": "RTX-4090",  "memory_gb": 24, "bandwidth": 1008},
    "v100":    {"name": "V100-32GB", "memory_gb": 32, "bandwidth": 900},
}

# ============================================================
# 显存计算
# ============================================================

def calc_memory(model, gpu_count, zero_stage, batch_size, seq_len, precision="bf16"):
    """计算训练显存需求"""
    p = model["params_b"] * 1e9
    bpb = 2  # bf16 bytes per param

    # 参数
    params = p * bpb
    # 优化器 (AdamW FP32: m + v)
    optimizer = p * 8
    # FP32 主权重
    master = p * 4
    # 梯度
    grads = p * bpb
    # 激活值 (粗略)
    activation = (batch_size * seq_len * model["hidden_size"] *
                  model["num_layers"] * 4 * bpb)

    # ZeRO 分片
    if zero_stage >= 1:
        optimizer /= gpu_count
    if zero_stage >= 2:
        grads /= gpu_count
    if zero_stage >= 3:
        params /= gpu_count
        master /= gpu_count

    total = params + optimizer + master + grads + activation
    return {
        "params_gb": params / 1e9,
        "optimizer_gb": optimizer / 1e9,
        "grads_gb": grads / 1e9,
        "activation_gb": activation / 1e9,
        "total_per_gpu_gb": total / 1e9,
    }


def find_best_config(model, gpu_type, gpu_count, target_zero=None):
    """搜索最优并行配置"""
    gpu_mem = GPU_TYPES[gpu_type]["memory_gb"]
    best = None

    # 搜索 TP x PP 组合
    for tp in [1, 2, 4, 8]:
        if tp > gpu_count:
            continue
        if model["num_attention_heads"] % tp != 0:
            continue

        for pp in [1, 2, 4, 8]:
            if tp * pp > gpu_count:
                continue
            if model["num_layers"] % pp != 0:
                continue

            dp = gpu_count // (tp * pp)
            if dp < 1:
                continue

            for zero in [0, 1, 2, 3]:
                if target_zero is not None and zero != target_zero:
                    continue

                mem = calc_memory(model, dp * tp * pp, zero,
                                  batch_size=1, seq_len=model["max_seq_len"])

                # 留 10% 余量
                if mem["total_per_gpu_gb"] <= gpu_mem * 0.9:
                    eff = dp / (tp * pp * dp) * 100  # 简化的效率指标
                    score = dp * 10 - tp - pp * 2 - zero  # 偏好高DP, 低TP/PP/ZeRO
                    if best is None or score > best["score"]:
                        best = {
                            "tp": tp, "pp": pp, "dp": dp,
                            "zero": zero,
                            "mem_gb": mem["total_per_gpu_gb"],
                            "efficiency": dp / gpu_count * 100,
                            "score": score,
                            "breakdown": mem,
                        }

    return best


# ============================================================
# 命令生成
# ============================================================

def gen_torchrun_cmd(config, model, gpu_count, nodes=1):
    """生成 torchrun 命令"""
    master_addr = "MASTER_IP" if nodes > 1 else "localhost"
    lines = [
        f"# {model['name']} 训练启动命令",
        f"# GPU: {gpu_count}x, TP={config['tp']}, PP={config['pp']}, DP={config['dp']}, ZeRO-{config['zero']}",
        f"# 预估每卡显存: {config['mem_gb']:.1f} GB",
        "",
    ]

    if nodes > 1:
        lines.append(f"torchrun \\")
        lines.append(f"  --nnodes={nodes} \\")
        lines.append(f"  --nproc_per_node={gpu_count // nodes} \\")
        lines.append(f"  --rdzv_id=job1 \\")
        lines.append(f"  --rdzv_backend=c10d \\")
        lines.append(f"  --rdzv_endpoint={master_addr}:29500 \\")
    else:
        lines.append(f"torchrun --nproc_per_node={gpu_count} \\")

    lines.append(f"  train.py \\")
    lines.append(f"  --tensor-model-parallel-size {config['tp']} \\")
    if config['pp'] > 1:
        lines.append(f"  --pipeline-model-parallel-size {config['pp']} \\")
    lines.append(f"  --num-layers {model['num_layers']} \\")
    lines.append(f"  --hidden-size {model['hidden_size']} \\")
    lines.append(f"  --num-attention-heads {model['num_attention_heads']} \\")
    lines.append(f"  --seq-length {model['max_seq_len']} \\")
    lines.append(f"  --max-position-embeddings {model['max_seq_len']} \\")
    lines.append(f"  --micro-batch-size 1 \\")
    lines.append(f"  --global-batch-size {config['dp'] * 4} \\")
    lines.append(f"  --bf16 \\")

    if config['zero'] > 0:
        lines.append(f"  --use-distributed-optimizer \\")

    lines.append(f"  --vocab-size {model['vocab_size']}")
    return "\n".join(lines)


def gen_deepspeed_cmd(config, model, gpu_count):
    """生成 DeepSpeed 命令"""
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": config["zero"]},
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "train_batch_size": config["dp"] * 4,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 10,
    }
    if config["zero"] == 3:
        ds_config["zero_optimization"]["overlap_comm"] = True
        ds_config["zero_optimization"]["contiguous_gradients"] = True

    lines = [
        f"# {model['name']} DeepSpeed 训练命令",
        f"# ZeRO-{config['zero']}, TP={config['tp']}, PP={config['pp']}",
        "",
        f"# 1. 保存 ds_config.json:",
    ]

    # 格式化 JSON
    lines.append("cat > ds_config.json << 'EOF'")
    lines.append(json.dumps(ds_config, indent=2))
    lines.append("EOF")
    lines.append("")
    lines.append(f"# 2. 启动训练:")
    lines.append(f"deepspeed --num_gpus={gpu_count} train.py \\")
    lines.append(f"  --deepspeed ds_config.json \\")
    lines.append(f"  --num-layers {model['num_layers']} \\")
    lines.append(f"  --hidden-size {model['hidden_size']} \\")
    lines.append(f"  --num-attention-heads {model['num_attention_heads']} \\")
    lines.append(f"  --bf16")

    return "\n".join(lines)


def gen_ddp_cmd(model, gpu_count, gpu_type):
    """生成简单 DDP 命令 (仅适合小模型)"""
    gpu_mem = GPU_TYPES[gpu_type]["memory_gb"]
    mem = calc_memory(model, gpu_count, 0, batch_size=1, seq_len=model["max_seq_len"])

    fits = mem["total_per_gpu_gb"] <= gpu_mem * 0.9

    lines = [
        f"# {model['name']} DDP 训练命令 (单机多卡)",
        f"# 预估显存: {mem['total_per_gpu_gb']:.1f} GB / {gpu_mem} GB",
        f"# {'✓ 可行' if fits else '✗ 显存不足, 需要模型并行或 ZeRO'}",
        "",
    ]

    if fits:
        lines.append(f"torchrun --nproc_per_node={gpu_count} train.py \\")
        lines.append(f"  --bf16 \\")
        lines.append(f"  --gradient-checkpointing \\")
        lines.append(f"  --batch-size 1 \\")
        lines.append(f"  --gradient-accumulation-steps 4")
    else:
        lines.append("# 建议使用 ZeRO-2 或模型并行方案")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练启动命令生成器")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="模型规格")
    parser.add_argument("--gpus", type=int, help="GPU 数量")
    parser.add_argument("--gpu-type", choices=list(GPU_TYPES.keys()), help="GPU 型号")
    parser.add_argument("--zero", type=int, choices=[0, 1, 2, 3], help="指定 ZeRO stage")
    parser.add_argument("--framework", choices=["megatron", "deepspeed", "ddp"],
                        default="megatron", help="训练框架")
    parser.add_argument("--nodes", type=int, default=1, help="节点数")
    parser.add_argument("--list-models", action="store_true", help="列出所有模型")
    parser.add_argument("--list-gpus", action="store_true", help="列出所有 GPU")
    parser.add_argument("--all-configs", action="store_true",
                        help="显示所有可行配置")
    args = parser.parse_args()

    if args.list_models:
        print("可用模型:")
        print(f"  {'Key':<10} {'名称':<20} {'参数量':<10} {'层数':<6} {'隐藏维度':<10} {'注意力头':<8}")
        print("  " + "-" * 70)
        for k, m in MODELS.items():
            print(f"  {k:<10} {m['name']:<20} {m['params_b']:<10.1f} "
                  f"{m['num_layers']:<6} {m['hidden_size']:<10} {m['num_attention_heads']:<8}")
        return

    if args.list_gpus:
        print("可用 GPU:")
        print(f"  {'Key':<10} {'名称':<15} {'显存':<10} {'带宽(GB/s)':<12}")
        print("  " + "-" * 50)
        for k, g in GPU_TYPES.items():
            print(f"  {k:<10} {g['name']:<15} {g['memory_gb']:<10} {g['bandwidth']:<12}")
        return

    if not args.model or not args.gpus or not args.gpu_type:
        parser.print_help()
        print("\n示例:")
        print("  python training_recipe.py --model 7b --gpus 4 --gpu-type a100")
        print("  python training_recipe.py --model 70b --gpus 32 --gpu-type a100 --zero 3")
        print("  python training_recipe.py --model 13b --gpus 8 --gpu-type a16 --framework deepspeed")
        return

    model = MODELS[args.model]
    gpu_type = args.gpu_type
    gpu_count = args.gpus

    print(f"{'='*60}")
    print(f"  {model['name']} 训练配置")
    print(f"  {gpu_count} × {GPU_TYPES[gpu_type]['name']}")
    print(f"{'='*60}")

    if args.all_configs:
        print("\n所有可行配置:")
        print(f"  {'TP':<4} {'PP':<4} {'DP':<4} {'ZeRO':<5} {'显存/卡':<10} {'DP效率':<10}")
        print("  " + "-" * 45)
        for tp in [1, 2, 4, 8]:
            for pp in [1, 2, 4, 8]:
                for zero in range(4):
                    if tp * pp > gpu_count:
                        continue
                    if model["num_attention_heads"] % tp != 0:
                        continue
                    dp = gpu_count // (tp * pp)
                    if dp < 1:
                        continue
                    mem = calc_memory(model, gpu_count, zero, 1, model["max_seq_len"])
                    gpu_mem = GPU_TYPES[gpu_type]["memory_gb"]
                    if mem["total_per_gpu_gb"] <= gpu_mem * 0.9:
                        eff = dp / gpu_count * 100
                        marker = " ←" if zero == 2 else ""
                        print(f"  {tp:<4} {pp:<4} {dp:<4} {zero:<5} "
                              f"{mem['total_per_gpu_gb']:<10.1f} {eff:<10.0f}%{marker}")
        print()

    config = find_best_config(model, gpu_type, gpu_count, args.zero)

    if config is None:
        print("\n⚠ 找不到可行配置! 建议:")
        print(f"  - 增加 GPU 数量 (当前: {gpu_count})")
        print(f"  - 使用更大显存的 GPU (当前: {GPU_TYPES[gpu_type]['memory_gb']}GB)")
        print(f"  - 启用 gradient checkpointing")
        print(f"  - 减小 batch_size 或 seq_len")
        return

    print(f"\n最优配置: TP={config['tp']}, PP={config['pp']}, "
          f"DP={config['dp']}, ZeRO-{config['zero']}")
    print(f"预估每卡显存: {config['mem_gb']:.1f} GB / "
          f"{GPU_TYPES[gpu_type]['memory_gb']} GB "
          f"({config['mem_gb']/GPU_TYPES[gpu_type]['memory_gb']*100:.0f}%)")
    print(f"DP 效率: {config['efficiency']:.0f}%")

    print(f"\n显存分解:")
    bd = config["breakdown"]
    print(f"  参数:     {bd['params_gb']:.1f} GB")
    print(f"  优化器:   {bd['optimizer_gb']:.1f} GB")
    print(f"  梯度:     {bd['grads_gb']:.1f} GB")
    print(f"  激活值:   {bd['activation_gb']:.1f} GB")

    print(f"\n{'─'*60}")

    if args.framework == "megatron":
        print(gen_torchrun_cmd(config, model, gpu_count, args.nodes))
    elif args.framework == "deepspeed":
        print(gen_deepspeed_cmd(config, model, gpu_count))
    elif args.framework == "ddp":
        print(gen_ddp_cmd(model, gpu_count, gpu_type))


if __name__ == "__main__":
    main()
