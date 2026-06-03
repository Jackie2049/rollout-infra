"""
分布式训练 Scaling Efficiency 模拟器

模拟不同并行策略 (TP/PP/DP/ZeRO) 下的训练效率：
- 计算时间 vs 通信时间 vs pipeline bubble
- 理论 scaling efficiency
- 帮助选择最优并行配置

用法:
    python tools/scaling_simulator.py --params 7B --gpus 64
    python tools/scaling_simulator.py --params 70B --gpus 256 --compare
"""

import argparse
import math
from dataclasses import dataclass

from memory_estimator import MODEL_CONFIGS


@dataclass
class GPUConfig:
    tflops: float        # FP16/BF16 TFLOPS
    hbm_bw_gbs: float    # HBM 带宽 GB/s
    nvlink_bw_gbs: float  # NVLink 双向带宽 GB/s
    ib_bw_gbs: float      # InfiniBand 带宽 GB/s


GPU_SPECS = {
    "A100": GPUConfig(tflops=312, hbm_bw_gbs=2039, nvlink_bw_gbs=600, ib_bw_gbs=200),
    "H100": GPUConfig(tflops=990, hbm_bw_gbs=3350, nvlink_bw_gbs=900, ib_bw_gbs=400),
}


def estimate_compute_time(num_params, seq_len, batch_size, gpu_spec, num_gpus):
    """估算单步计算时间 (forward + backward ≈ 6 × forward FLOPs)"""
    # FLOPs per token per layer ≈ 6 × hidden_size² (attention + MLP)
    # 总 FLOPs ≈ 6 × num_params × seq_len × batch_size (forward only)
    # forward + backward ≈ 3x forward
    flops_per_step = 6 * num_params * seq_len * batch_size  # forward + backward
    flops_per_gpu = flops_per_step / num_gpus

    # 实际利用率: ~40-60% (受限于 memory bandwidth 和 kernel 效率)
    mfu = 0.45  # Model FLOPs Utilization
    compute_time = flops_per_gpu / (gpu_spec.tflops * 1e12 * mfu)
    return compute_time


def estimate_comm_time_tp(num_params, seq_len, batch_size, tp, gpu_spec, layers=32):
    """估算 TP 通信时间 (每层 2 次 AllReduce)"""
    if tp <= 1:
        return 0

    # 每层的 hidden activation: [batch, seq, hidden/tp]
    hidden = int(math.sqrt(num_params * layers / (12 * layers)))  # rough estimate
    msg_size_bytes = batch_size * seq_len * hidden * 2  # FP16

    # Ring AllReduce: 2 × (N-1)/N × msg_size / bandwidth
    # TP 走 NVLink
    bw = gpu_spec.nvlink_bw_gbs * 1e9  # bytes/s
    allreduce_time = 2 * (tp - 1) / tp * msg_size_bytes / bw

    # 每层 2 次 AllReduce (attention + MLP)
    return layers * 2 * allreduce_time


def estimate_comm_time_pp(num_params, seq_len, batch_size, pp, gpu_spec, layers=32):
    """估算 PP 通信时间 + bubble"""
    if pp <= 1:
        return 0, 0  # comm_time, bubble_time

    hidden = int(math.sqrt(num_params * layers / (12 * layers)))
    # P2P 通信: 发送 [batch, seq, hidden] 的 activation
    msg_size_bytes = batch_size * seq_len * hidden * 2  # FP16

    # 假设 NVLink (节点内) 或 IB (节点间)
    bw = gpu_spec.nvlink_bw_gbs * 1e9
    p2p_time = msg_size_bytes / bw

    # Pipeline bubble: (pp-1) 个 warmup + cooldown microbatch
    # 假设 num_microbatches = 8
    num_microbatches = 8
    bubble_fraction = (pp - 1) / num_microbatches

    comm_time_per_step = pp * 2 * p2p_time  # forward + backward, pp 次
    bubble_time_per_step = bubble_fraction * estimate_compute_time(
        num_params, seq_len, batch_size, gpu_spec, pp
    )

    return comm_time_per_step, bubble_time_per_step


def estimate_comm_time_zero(num_params, zero_stage, dp, gpu_spec):
    """估算 ZeRO 通信时间"""
    if zero_stage == 0 or dp <= 1:
        return 0

    msg_size_bytes = num_params * 2  # FP16

    if zero_stage <= 2:
        # AllReduce/ReduceScatter + AllGather ≈ 2 × msg_size
        bw = gpu_spec.nvlink_bw_gbs * 1e9 if dp <= 8 else gpu_spec.ib_bw_gbs * 1e9
        comm_time = 2 * (dp - 1) / dp * msg_size_bytes / bw
    else:
        # ZeRO-3: AllGather (forward) + AllGather + ReduceScatter (backward)
        # ≈ 3 × msg_size (per layer)
        bw = gpu_spec.nvlink_bw_gbs * 1e9 if dp <= 8 else gpu_spec.ib_bw_gbs * 1e9
        comm_time = 3 * (dp - 1) / dp * msg_size_bytes / bw

    return comm_time


def simulate_config(
    model_name, num_gpus, tp, pp, dp, zero_stage,
    seq_len=2048, batch_size=1, gpu_type="A100",
):
    """模拟一个并行配置的训练效率"""
    cfg = MODEL_CONFIGS[model_name]
    gpu = GPU_SPECS[gpu_type]
    psi = cfg["num_params"]
    layers = cfg["num_layers"]
    hidden = cfg["hidden_size"]

    # 总 batch = batch_per_gpu × dp
    global_batch = batch_size * dp

    # 计算时间
    compute_time = estimate_compute_time(psi, seq_len, global_batch, gpu, num_gpus)

    # 通信时间
    tp_comm = estimate_comm_time_tp(psi, seq_len, batch_size, tp, gpu, layers)
    pp_comm, pp_bubble = estimate_comm_time_pp(psi, seq_len, batch_size, pp, gpu, layers)
    zero_comm = estimate_comm_time_zero(psi / (tp * pp), zero_stage, dp, gpu)

    total_comm = tp_comm + pp_comm + zero_comm
    total_time = compute_time + total_comm + pp_bubble

    # Efficiency
    ideal_time = estimate_compute_time(psi, seq_len, global_batch, gpu, 1)
    ideal_parallel_time = ideal_time / num_gpus
    efficiency = ideal_parallel_time / total_time if total_time > 0 else 0

    # Throughput
    tokens_per_step = seq_len * global_batch
    throughput = tokens_per_step / total_time

    return {
        "compute_time": compute_time,
        "tp_comm": tp_comm,
        "pp_comm": pp_comm,
        "pp_bubble": pp_bubble,
        "zero_comm": zero_comm,
        "total_comm": total_comm,
        "total_time": total_time,
        "efficiency": efficiency,
        "throughput": throughput,
        "tokens_per_step": tokens_per_step,
    }


def find_best_config(model_name, num_gpus, seq_len=2048, batch_size=1, gpu_type="A100"):
    """搜索最优并行配置"""
    best_config = None
    best_throughput = 0
    results = []

    # 遍历所有可能的 TP × PP × DP = num_gpus 组合
    for tp in [1, 2, 4, 8]:
        if tp > num_gpus:
            break
        for pp in [1, 2, 4, 8]:
            if tp * pp > num_gpus:
                break
            if num_gpus % (tp * pp) != 0:
                continue
            dp = num_gpus // (tp * pp)
            for zero in [0, 1, 2]:
                r = simulate_config(
                    model_name, num_gpus, tp, pp, dp, zero,
                    seq_len, batch_size, gpu_type,
                )
                results.append((tp, pp, dp, zero, r))
                if r["throughput"] > best_throughput:
                    best_throughput = r["throughput"]
                    best_config = (tp, pp, dp, zero, r)

    return best_config, results


def main():
    parser = argparse.ArgumentParser(description="Scaling Efficiency Simulator")
    parser.add_argument("--params", default="7B", help="模型大小")
    parser.add_argument("--gpus", type=int, default=64, help="总 GPU 数量")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2, help="每 GPU batch size")
    parser.add_argument("--gpu-type", default="A100", choices=["A100", "H100"])
    parser.add_argument("--compare", action="store_true", help="对比不同 GPU 数量的 scaling")
    args = parser.parse_args()

    if args.params not in MODEL_CONFIGS:
        print(f"未知模型: {args.params}")
        return

    if args.compare:
        # Scaling curve: 不同 GPU 数量
        print("=" * 80)
        print(f"  Scaling Curve: {args.params} | Seq={args.seq_len} | Batch/GPU={args.batch_size} | {args.gpu_type}")
        print("=" * 80)
        print()

        gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        gpu_counts = [g for g in gpu_counts if g <= args.gpus]

        print(f"  {'GPUs':>6} {'最优配置':>20} {'计算(s)':>10} {'通信(s)':>10} {'Bubble(s)':>10} "
              f"{'总时间(s)':>10} {'效率':>8} {'吞吐(tokens/s)':>15}")
        print("  " + "-" * 95)

        baseline_throughput = None
        for ng in gpu_counts:
            best, _ = find_best_config(
                args.params, ng, args.seq_len, args.batch_size, args.gpu_type
            )
            if best is None:
                continue
            tp, pp, dp, zero, r = best
            config_str = f"TP={tp} PP={pp} DP={dp} Z{zero}"
            if baseline_throughput is None:
                baseline_throughput = r["throughput"]
            scaling = r["throughput"] / baseline_throughput if baseline_throughput > 0 else 0

            print(f"  {ng:>6} {config_str:>20} {r['compute_time']:>10.4f} {r['total_comm']:>10.4f} "
                  f"{r['pp_bubble']:>10.4f} {r['total_time']:>10.4f} {r['efficiency']:>7.1%} "
                  f"{r['throughput']:>15,.0f}")

        print()

    else:
        # 单一配置分析
        best, all_results = find_best_config(
            args.params, args.gpus, args.seq_len, args.batch_size, args.gpu_type
        )

        if best is None:
            print(f"无法为 {args.params} 在 {args.gpus} GPUs 上找到有效配置")
            return

        tp, pp, dp, zero, r = best

        print("=" * 60)
        print(f"  分布式训练 Scaling Efficiency 模拟")
        print("=" * 60)
        print(f"  模型:       {args.params}")
        print(f"  GPU:        {args.gpus} × {args.gpu_type}")
        print(f"  Seq length: {args.seq_len}")
        print(f"  Batch/GPU:  {args.batch_size}")
        print(f"  Global batch: {args.batch_size * dp}")
        print()
        print(f"  ── 最优配置 ──")
        print(f"  TP:         {tp}")
        print(f"  PP:         {pp}")
        print(f"  DP:         {dp}")
        print(f"  ZeRO:       Stage {zero}")
        print()
        print(f"  ── 时间分解 (per step) ──")
        print(f"  计算时间:   {r['compute_time']*1000:>10.2f} ms")
        print(f"  TP 通信:    {r['tp_comm']*1000:>10.2f} ms")
        print(f"  PP 通信:    {r['pp_comm']*1000:>10.2f} ms")
        print(f"  PP Bubble:  {r['pp_bubble']*1000:>10.2f} ms")
        print(f"  ZeRO 通信:  {r['zero_comm']*1000:>10.2f} ms")
        print(f"  ─────────────────────")
        print(f"  总时间:     {r['total_time']*1000:>10.2f} ms")
        print(f"  效率:       {r['efficiency']:>10.1%}")
        print(f"  吞吐量:     {r['throughput']:>10,.0f} tokens/s")
        print("=" * 60)

        # Top 5 配置
        print()
        print(f"  ── Top 5 配置 ──")
        print(f"  {'配置':>20} {'效率':>8} {'吞吐':>15}")
        print("  " + "-" * 50)
        sorted_results = sorted(all_results, key=lambda x: x[4]["throughput"], reverse=True)
        for i, (t, p, d, z, r) in enumerate(sorted_results[:5]):
            config_str = f"TP={t} PP={p} DP={d} Z{z}"
            print(f"  {config_str:>20} {r['efficiency']:>8.1%} {r['throughput']:>15,.0f}")


if __name__ == "__main__":
    main()
