#!/usr/bin/env python3
"""分布式训练配置决策工具

根据模型大小、GPU 类型和数量，推荐最优的并行策略配置。

用法:
  # 7B 模型，4 张 A100
  python training_config_guide.py --model-size 7B --gpu-type a100-80g --num-gpus 4

  # 70B 模型，64 张 H100
  python training_config_guide.py --model-size 70B --gpu-type h100 --num-gpus 64

  # 自定义模型参数
  python training_config_guide.py --num-layers 32 --hidden-size 4096 --num-heads 32 --gpu-type a100-40g --num-gpus 8

  # 比较不同 GPU 数量的配置
  python training_config_guide.py --model-size 70B --gpu-type a100-80g --compare
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUConfig:
    name: str
    memory_gb: int
    bandwidth_gb_s: float
    tf32_tflops: float
    fp16_tflops: float
    nvlink_bw_gb_s: float


GPU_CONFIGS = {
    "a100-40g": GPUConfig("A100-40G", 40, 1555, 156, 312, 600),
    "a100-80g": GPUConfig("A100-80G", 80, 2039, 156, 312, 600),
    "h100": GPUConfig("H100-80G", 80, 3350, 495, 990, 900),
    "h200": GPUConfig("H200-141G", 141, 4800, 495, 990, 900),
    "a6000": GPUConfig("A6000-48G", 48, 768, 38.7, 77.9, 0),  # PCIe, no NVLink
    "a16": GPUConfig("A16-16G", 16, 400, 74, 148, 0),
    "3090": GPUConfig("RTX3090-24G", 24, 936, 35.6, 71.2, 0),
    "4090": GPUConfig("RTX4090-24G", 24, 1008, 82.6, 165.2, 0),
}


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int = 32000
    ffn_hidden_size: Optional[int] = None
    seq_length: int = 4096
    bytes_per_param: int = 2  # BF16


MODEL_PRESETS = {
    "1.3b": ModelConfig("1.3B", 24, 2048, 16, vocab_size=32000),
    "7b": ModelConfig("7B", 32, 4096, 32, vocab_size=32000),
    "13b": ModelConfig("13B", 40, 5120, 40, vocab_size=32000),
    "30b": ModelConfig("30B", 64, 6656, 52, vocab_size=32000),
    "34b": ModelConfig("34B", 48, 8192, 64, vocab_size=32000),
    "70b": ModelConfig("70B", 80, 8192, 64, vocab_size=32000),
    "175b": ModelConfig("175B", 96, 12288, 96, vocab_size=32000),
}


def calc_params(model: ModelConfig) -> int:
    """估算模型参数量"""
    h = model.hidden_size
    l = model.num_layers
    v = model.vocab_size
    ffn = model.ffn_hidden_size or 4 * h

    # Embedding
    embed = v * h
    # Per layer: qkv + output projection + FFN
    attn = 4 * h * h  # QKV + O
    ffn_params = 3 * h * ffn  # gate, up, down
    layer_norm = 2 * 2 * h  # layernorm gamma+beta × 2
    per_layer = attn + ffn_params + layer_norm
    # Total
    total = embed + l * per_layer + h  # final norm
    return total


def format_params(n: int) -> str:
    """格式化参数量"""
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return str(n)


def calc_memory(model: ModelConfig, tp: int, pp: int, zero_stage: int,
                batch_size: int = 1, seq_len: int = 4096) -> dict:
    """计算每 GPU 显存需求"""
    psi = calc_params(model)  # total params
    psi_per_gpu = psi / tp if pp == 1 else psi / (tp * pp)

    bytes_per_param = model.bytes_per_param

    # Model params + grads + optimizer
    # DDP: 16 * psi_per_gpu bytes
    # ZeRO-1: 4*psi_per_gpu + 12*psi_per_gpu/DP (optimizer sharded)
    # ZeRO-2: 2*psi_per_gpu + 14*psi_per_gpu/DP (grads+optimizer sharded)
    # ZeRO-3: 16*psi_per_gpu/DP (all sharded)
    # Simplified: per-GPU memory for model state
    model_mem = bytes_per_param * psi_per_gpu  # params
    grad_mem = bytes_per_param * psi_per_gpu   # gradients
    optim_mem = 8 * psi_per_gpu                # optimizer (Adam: m + v, FP32)

    total_state = model_mem + grad_mem + optim_mem

    # Activation memory (per GPU, after TP/PP)
    # Approximation: 2 * bsz * seq_len * h * l / (tp * pp) bytes
    act_per_layer = 2 * batch_size * seq_len * model.hidden_size
    activation_mem = act_per_layer * model.num_layers / (tp * pp)

    # With gradient checkpointing, activation = O(sqrt(L)) layers
    # Without: O(L) layers
    # Factor of ~2x overhead for temp buffers, fragmentation
    overhead = 1.5  # 50% overhead

    # ZeRO savings
    dp = 1  # will be set externally
    if zero_stage == 0:
        state_per_gpu = total_state
    elif zero_stage == 1:
        state_per_gpu = model_mem + grad_mem + optim_mem / dp
    elif zero_stage == 2:
        state_per_gpu = model_mem + (grad_mem + optim_mem) / dp
    else:  # ZeRO-3
        state_per_gpu = total_state / dp

    total_gb = (state_per_gpu + activation_mem) * overhead / (1024**3)

    return {
        "total_params": psi,
        "params_per_gpu": psi_per_gpu,
        "model_mem_gb": model_mem / (1024**3),
        "grad_mem_gb": grad_mem / (1024**3),
        "optim_mem_gb": optim_mem / (1024**3),
        "activation_mem_gb": activation_mem / (1024**3),
        "total_per_gpu_gb": total_gb,
    }


def find_configs(model: ModelConfig, gpu: GPUConfig, num_gpus: int) -> list[dict]:
    """搜索所有可行的并行配置"""
    psi = calc_params(model)
    configs = []

    # Try different TP × PP combinations
    for tp in [1, 2, 4, 8]:
        if tp > num_gpus:
            continue
        for pp in [1, 2, 4, 8]:
            if tp * pp > num_gpus:
                continue
            if num_gpus % (tp * pp) != 0:
                continue
            dp = num_gpus // (tp * pp)

            for zero_stage in [0, 1, 2, 3]:
                mem = calc_memory(model, tp, pp, zero_stage,
                                  batch_size=1, seq_len=model.seq_length)
                fits = mem["total_per_gpu_gb"] <= gpu.memory_gb * 0.9  # 90% threshold

                # Communication cost estimation
                # TP: 2 all-reduce per layer, volume = 2*b*s*h/tp
                # PP: pipeline bubble = (pp-1)/num_microbatches
                # ZeRO: all-reduce for grads (stage 2) or all-gather for params (stage 3)

                comm_cost = estimate_comm_cost(model, tp, pp, dp, zero_stage, gpu)

                configs.append({
                    "tp": tp, "pp": pp, "dp": dp,
                    "zero_stage": zero_stage,
                    "memory_per_gpu_gb": mem["total_per_gpu_gb"],
                    "fits": fits,
                    "comm_cost": comm_cost,
                    "efficiency": max(0, 1 - comm_cost),
                    "model_mem": mem,
                })

    return sorted(configs, key=lambda x: (-x["fits"], -x["efficiency"]))


def estimate_comm_cost(model: ModelConfig, tp: int, pp: int, dp: int,
                       zero_stage: int, gpu: GPUConfig) -> float:
    """估算通信开销占比 (0-1, 0=无开销)"""
    h = model.hidden_size
    l = model.num_layers
    bs = 1  # per GPU batch size
    seq = model.seq_length

    # TP communication: 2 all-reduce per layer (forward+backward)
    # Volume per all-reduce: 2 * bs * seq * h / tp
    # Approximated as fraction of compute time
    tp_cost = 0
    if tp > 1:
        # Communication/compute ratio for TP
        # Compute: 2 * bs * seq * h * h / tp per layer
        # Comm: 2 * 2 * bs * seq * h / tp (2 all-reduce, ring has tp-1 steps)
        tp_cost = (2 * (tp - 1) / tp) * (2 * h / (h * 2))  # simplified
        tp_cost = min(0.3, 2 * tp / h * 10)  # heuristic: larger h mitigates

    # PP pipeline bubble
    pp_cost = 0
    if pp > 1:
        # Bubble = (pp-1) / num_microbatches
        # Assume num_microbatches = dp * 4 (gradient accumulation)
        num_mb = max(dp * 4, pp)
        pp_cost = (pp - 1) / num_mb

    # ZeRO communication
    zero_cost = 0
    if zero_stage >= 2 and dp > 1:
        # All-reduce grads (stage 2) or all-gather params (stage 3)
        zero_cost = 0.02 * (zero_stage - 1) / dp  # small overhead

    return min(1.0, tp_cost + pp_cost + zero_cost)


def print_config(model: ModelConfig, gpu: GPUConfig, num_gpus: int, configs: list[dict],
                 show_all: bool = False):
    """打印配置推荐"""
    psi = calc_params(model)

    print(f"\n{'='*70}")
    print(f"  分布式训练配置推荐")
    print(f"{'='*70}")
    print(f"  模型:    {model.name} ({format_params(psi)} params)")
    print(f"  Layers:  {model.num_layers}, Hidden: {model.hidden_size}, Heads: {model.num_heads}")
    print(f"  Seq Len: {model.seq_length}")
    print(f"  GPU:     {gpu.name} × {num_gpus} ({gpu.memory_gb}GB each)")
    print(f"{'='*70}")

    feasible = [c for c in configs if c["fits"]]
    infeasible = [c for c in configs if not c["fits"]]

    if not feasible:
        print("\n  ⚠ 没有可行配置! 建议:")
        print(f"    - 增加 GPU 数量 (当前 {num_gpus})")
        print(f"    - 使用更大显存的 GPU (当前 {gpu.memory_gb}GB)")
        print(f"    - 减小模型大小或序列长度")
        print(f"\n  最接近的配置:")
        if infeasible:
            best = infeasible[0]
            print(f"    TP={best['tp']} PP={best['pp']} DP={best['dp']} "
                  f"ZeRO-{best['zero_stage']}")
            print(f"    需要 {best['memory_per_gpu_gb']:.1f}GB (GPU 只有 {gpu.memory_gb}GB)")
        return

    print(f"\n  ✓ 找到 {len(feasible)} 个可行配置:\n")
    print(f"  {'Rank':<5} {'TP':>3} {'PP':>3} {'DP':>4} {'ZeRO':>5} "
          f"{'显存/GPU':>10} {'效率':>8} {'说明'}")
    print(f"  {'-'*60}")

    recommendations = {
        "high_compute": "高计算效率",
        "balanced": "均衡推荐",
        "low_mem": "省显存",
    }

    for i, c in enumerate(feasible[:10]):
        # Determine recommendation tag
        tag = ""
        if i == 0 and c["efficiency"] == max(x["efficiency"] for x in feasible):
            tag = "★ 最优效率"
        if c["zero_stage"] == 2 and c["tp"] <= 4:
            tag = tag or "◆ 推荐"
        if c["memory_per_gpu_gb"] < gpu.memory_gb * 0.6:
            tag = tag or "  显存充裕"

        zero_str = f"ZeRO-{c['zero_stage']}" if c["zero_stage"] > 0 else "DDP"
        print(f"  {i+1:<5} {c['tp']:>3} {c['pp']:>3} {c['dp']:>4} {zero_str:>7} "
              f"{c['memory_per_gpu_gb']:>8.1f}GB {c['efficiency']:>7.1%} {tag}")

    # Best recommendation
    best = feasible[0]
    print(f"\n{'='*70}")
    print(f"  ★ 推荐配置:")
    print(f"    TP={best['tp']}, PP={best['pp']}, DP={best['dp']}, "
          f"ZeRO-{best['zero_stage']}")
    print(f"    显存使用: {best['memory_per_gpu_gb']:.1f}GB / {gpu.memory_gb}GB "
          f"({best['memory_per_gpu_gb']/gpu.memory_gb*100:.0f}%)")
    print(f"    预估效率: {best['efficiency']:.1%}")

    dp = best["dp"]
    tp = best["tp"]
    pp = best["pp"]
    zero = best["zero_stage"]

    print(f"\n  启动命令示例:")
    print(f"    torchrun \\")
    print(f"      --nnodes=<节点数> \\")
    print(f"      --nproc_per_node={tp * pp // 1 if pp == 1 else tp} \\")
    if pp > 1:
        print(f"      # PP={pp}: 每节点 {tp} GPU, 共 {pp} 节点")
    print(f"      train.py \\")
    if tp > 1:
        print(f"      --tensor-model-parallel-size {tp} \\")
    if pp > 1:
        print(f"      --pipeline-model-parallel-size {pp} \\")
    print(f"      --micro-batch-size 1 \\")
    print(f"      --global-batch-size {dp * 4} \\")
    print(f"      --bf16")
    if zero > 0:
        print(f"      --zero-stage {zero}")

    print(f"\n  关键环境变量:")
    print(f"    export CUDA_DEVICE_MAX_CONNECTIONS=1  # Megatron 必需")
    if num_gpus > 8:
        print(f"    export NCCL_IB_DISABLE=0              # 启用 InfiniBand")
        print(f"    export NCCL_NET_GDR_LEVEL=5            # GPUDirect RDMA")
    print(f"{'='*70}")


def compare_configs(model_name: str, gpu_type: str):
    """比较不同 GPU 数量的配置"""
    model = MODEL_PRESETS.get(model_name.lower().replace("-", "").replace(" ", ""))
    if not model:
        print(f"Unknown model: {model_name}")
        return
    gpu = GPU_CONFIGS.get(gpu_type.lower())
    if not gpu:
        print(f"Unknown GPU: {gpu_type}")
        return

    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    psi = calc_params(model)

    print(f"\n{'='*75}")
    print(f"  扩展性分析: {model.name} ({format_params(psi)}) on {gpu.name}")
    print(f"{'='*75}")
    print(f"  {'GPUs':>6} {'TP':>4} {'PP':>4} {'DP':>5} {'ZeRO':>6} "
          f"{'显存/GPU':>10} {'效率':>8} {'Throughput':>12}")
    print(f"  {'-'*65}")

    for n in gpu_counts:
        configs = find_configs(model, gpu, n)
        feasible = [c for c in configs if c["fits"]]
        if not feasible:
            print(f"  {n:>6}  --- 不可行 ---")
            continue

        best = feasible[0]
        zero_str = f"ZeRO-{best['zero_stage']}" if best["zero_stage"] > 0 else "DDP"
        # Relative throughput = DP × efficiency
        throughput = best["dp"] * best["efficiency"]
        print(f"  {n:>6} {best['tp']:>4} {best['pp']:>4} {best['dp']:>5} "
              f"{zero_str:>7} {best['memory_per_gpu_gb']:>8.1f}GB "
              f"{best['efficiency']:>7.1%} {throughput:>10.1f}x")

    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="分布式训练配置决策工具")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=list(MODEL_PRESETS.keys()),
                        help="模型大小 (1.3b/7b/13b/30b/34b/70b/175b)")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--gpu-type", type=str, default="a100-80g",
                        choices=list(GPU_CONFIGS.keys()))
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--compare", action="store_true",
                        help="比较不同 GPU 数量的扩展性")
    parser.add_argument("--show-all", action="store_true",
                        help="显示所有配置 (包括不可行)")

    args = parser.parse_args()

    # Build model config
    if args.model_size:
        model = MODEL_PRESETS[args.model_size.lower().replace("-", "").replace(" ", "")]
    elif args.num_layers and args.hidden_size and args.num_heads:
        psi = calc_params(ModelConfig("", args.num_layers, args.hidden_size, args.num_heads))
        model = ModelConfig(
            name=format_params(psi),
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            seq_length=args.seq_length,
        )
    else:
        parser.error("需要 --model-size 或 --num-layers/--hidden-size/--num-heads")

    model.seq_length = args.seq_length
    gpu = GPU_CONFIGS[args.gpu_type]

    if args.compare:
        model_name = args.model_size or "custom"
        compare_configs(model_name, args.gpu_type)
    else:
        configs = find_configs(model, gpu, args.num_gpus)
        print_config(model, gpu, args.num_gpus, configs, args.show_all)


if __name__ == "__main__":
    main()
