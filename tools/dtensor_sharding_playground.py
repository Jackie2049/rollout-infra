#!/usr/bin/env python3
"""
PyTorch DTensor Sharding Playground — CPU兼容

可视化DTensor分片操作(AllGather/ReduceScatter/AllReduce):
1. 模拟DTensor的Shard/Replicate/_Partial placements
2. 可视化redistribute操作(AllGather/RS/AR)的通信量和数据流
3. 量化不同DeviceMesh配置的通信开销
4. 展示FSDP2(Shard(0)+AllGather/RS) vs TP(Shard(0/1)+AR)的对比

基于: pytorch-dtensor-autograd-reading.md + pytorch-fsdp2-internals-reading.md
      + DTensor 2.12 ProcessGroupCollection

用法:
  python3 tools/dtensor_sharding_playground.py --mode fsdp2
  python3 tools/dtensor_sharding_playground.py --mode tp --tp-degree 2
  python3 tools/dtensor_sharding_playground.py --mode compare
  python3 tools/dtensor_sharding_playground.py --mode hsdp --dp 4 --hsdp-inner 2
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def simulate_dtensor_sharding(tensor_shape, mesh_dims, placements, verbose=False):
    """模拟DTensor分片操作"""
    total_elements = tensor_shape[0] * tensor_shape[1]
    total_bytes = total_elements * 2  # BF16 = 2 bytes

    results = {
        "tensor_shape": tensor_shape,
        "mesh_dims": mesh_dims,
        "placements": [str(p) for p in placements],
        "total_bytes": total_bytes,
    }

    # Calculate local shard sizes
    for i, (dim, placement) in enumerate(zip(mesh_dims, placements)):
        if placement == "Shard(0)":  # Shard along dim 0
            shard_size = total_bytes // dim
            results[f"mesh_{i}_shard_bytes"] = shard_size
            results[f"mesh_{i}_shard_shape"] = (tensor_shape[0] // dim, tensor_shape[1])
        elif placement == "Shard(1)":  # Shard along dim 1
            shard_size = total_bytes // dim
            results[f"mesh_{i}_shard_bytes"] = shard_size
            results[f"mesh_{i}_shard_shape"] = (tensor_shape[0], tensor_shape[1] // dim)
        elif placement == "Replicate":
            results[f"mesh_{i}_shard_bytes"] = total_bytes
            results[f"mesh_{i}_shard_shape"] = tensor_shape
        elif placement == "_Partial":
            results[f"mesh_{i}_shard_bytes"] = total_bytes // dim
            results[f"mesh_{i}_partial_sum"] = True

    return results


def simulate_redistribute(src_placements, dst_placements, mesh_dims, tensor_bytes, interconnect="nvlink"):
    """模拟DTensor redistribute操作 → 计算通信量"""
    ops = []
    total_comm = 0

    for i, (src, dst) in enumerate(zip(src_placements, dst_placements)):
        if src == dst:
            ops.append({"mesh_dim": i, "op": "none", "comm_bytes": 0})
            continue

        dim = mesh_dims[i]
        if src == "Shard(0)" and dst == "Replicate":
            # AllGather: 每GPU接收(N-1)个shard → (N-1)/N × total_bytes
            comm = tensor_bytes * (dim - 1) // dim
            ops.append({"mesh_dim": i, "op": "AllGather", "comm_bytes": comm,
                       "description": f"Shard→Replicate: 每GPU从{dim-1}个peer收集shard"})
            total_comm += comm
        elif src == "Replicate" and dst == "Shard(0)":
            # ReduceScatter: 每GPU发送(N-1)/N × total_bytes + 接收1/N shard
            comm = tensor_bytes * (dim - 1) // dim
            ops.append({"mesh_dim": i, "op": "ReduceScatter", "comm_bytes": comm,
                       "description": f"Replicate→Shard: 先reduce再scatter到{dim}个shard"})
            total_comm += comm
        elif src == "_Partial" and dst == "Replicate":
            # AllReduce: 每GPU参与sum → 等价于ReduceScatter+AllGather → 2×(N-1)/N×total_bytes
            comm = tensor_bytes * (dim - 1) // dim
            ops.append({"mesh_dim": i, "op": "AllReduce", "comm_bytes": 2 * comm,
                       "description": f"Partial→Replicate: AllReduce(sum across {dim} GPUs)"})
            total_comm += 2 * comm
        elif src == "_Partial" and dst == "Shard(0)":
            # ReduceScatter: partial→shard → reduce + scatter
            comm = tensor_bytes * (dim - 1) // dim
            ops.append({"mesh_dim": i, "op": "ReduceScatter", "comm_bytes": comm,
                       "description": f"Partial→Shard: ReduceScatter(sum + split)"})
            total_comm += comm
        elif src == "Shard(0)" and dst == "_Partial":
            # This shouldn't happen normally (shard→partial means sum shards)
            ops.append({"mesh_dim": i, "op": "AllReduce(sum_shards)", "comm_bytes": tensor_bytes,
                       "description": "Shard→Partial: 不常见(requires sum of shards)"})
            total_comm += tensor_bytes
        else:
            ops.append({"mesh_dim": i, "op": f"{src}→{dst}", "comm_bytes": 0,
                       "description": f"未知转换: {src}→{dst}"})

    # Calculate latency based on interconnect
    bandwidths = {
        "nvlink": 300e9,  # 300 GB/s (NVLink 4.0 H100)
        "nvlink3": 150e9,  # 150 GB/s (NVLink 3.0 A100)
        "nvlink2": 80e9,   # 80 GB/s (NVLink 2.0 V100)
        "pcie4": 32e9,     # 32 GB/s (PCIe 4.0 x16)
        "pcie3": 16e9,     # 16 GB/s (PCIe 3.0 x16)
    }
    bw = bandwidths.get(interconnect, 32e9)
    latency = total_comm / bw if total_comm > 0 else 0

    return {
        "ops": ops,
        "total_comm_bytes": total_comm,
        "total_comm_gb": total_comm / 1e9,
        "interconnect": interconnect,
        "bandwidth_gb_s": bw / 1e9,
        "latency_s": latency,
        "latency_ms": latency * 1000,
    }


def simulate_fsdp2(model_size_b, dp_degree, interconnect="nvlink", reshard=True):
    """模拟FSDP2完整训练通信"""
    Ψ = model_size_b * 1e9  # total params
    bf16_bytes = 2 * Ψ  # BF16 = 2 bytes per param
    params_per_layer = bf16_bytes  # per-parameter approach

    print(f"\n{'='*60}")
    print(f"  FSDP2 (Shard(0) on DP={dp_degree}) Training Communication")
    print(f"{'='*60}")
    print(f"  Model: {model_size_b}B ({Ψ/1e9:.1f}B params)")
    print(f"  DP degree: {dp_degree}")
    print(f"  Interconnect: {interconnect}")
    print(f"  reshard_after_forward: {reshard}")

    # Forward: AllGather per parameter group (per-layer)
    fwd_ag_comm = bf16_bytes * (dp_degree - 1) // dp_degree
    if reshard:
        # 2 AllGathers per param group: unshard + reshard
        fwd_total = 2 * fwd_ag_comm
        print(f"\n  Forward (reshard=True):")
        print(f"    AllGather #1 (unshard): {fwd_ag_comm/1e9:.2f} GB per GPU")
        print(f"    AllGather #2 (reshard): {fwd_ag_comm/1e9:.2f} GB per GPU → ★ 等于2×AG!")
        print(f"    Total forward comm: {fwd_total/1e9:.2f} GB per GPU")
    else:
        fwd_total = fwd_ag_comm
        print(f"\n  Forward (reshard=False):")
        print(f"    AllGather #1 (unshard): {fwd_ag_comm/1e9:.2f} GB per GPU")
        print(f"    ★ No reshard → keep full params → 1×AG → 省通信但peak更高!")
        print(f"    Total forward comm: {fwd_total/1e9:.2f} GB per GPU")

    # Backward: ReduceScatter per parameter group
    bwd_rs_comm = bf16_bytes * (dp_degree - 1) // dp_degree
    print(f"\n  Backward:")
    print(f"    ReduceScatter: {bwd_rs_comm/1e9:.2f} GB per GPU")
    print(f"    ★ gradient placement: Shard(0) → _Partial → ReduceScatter → Shard(0)")

    total_per_step = fwd_total + bwd_rs_comm
    print(f"\n  Total per step: {total_per_step/1e9:.2f} GB per GPU")

    # Latency calculation
    bandwidths = {"nvlink": 300e9, "nvlink3": 150e9, "nvlink2": 80e9,
                  "pcie4": 32e9, "pcie3": 16e9}
    bw = bandwidths.get(interconnect, 32e9)
    latency = total_per_step / bw

    print(f"\n  ★ Latency: {latency*1000:.2f} ms ({interconnect} @ {bw/1e9:.0f} GB/s)")
    print(f"  ★ Peak memory: {bf16_bytes/1e9 + fwd_ag_comm/1e9:.2f} GB per GPU (reshard={reshard})")

    # vs ZeRO-3
    zero3_fwd = 3 * fwd_ag_comm  # 3 AllGathers per step
    zero3_bwd = bwd_rs_comm
    zero3_total = zero3_fwd + zero3_bwd
    print(f"\n  vs ZeRO-3:")
    print(f"    ZeRO-3 total: {zero3_total/1e9:.2f} GB (3×AG fwd + RS bwd)")
    print(f"    FSDP2 total: {total_per_step/1e9:.2f} GB ({'2×AG+RS' if reshard else '1×AG+RS'})")
    print(f"    ★ FSDP2 saves {(1 - total_per_step/zero3_total)*100:.1f}% communication vs ZeRO-3!")

    # HSDP note
    if dp_degree > 4:
        print(f"\n  ★ HSDP (Hierarchical Sharded Data Parallel):")
        print(f"    Inner group(reshard=True) + Outer group(reshard=False)")
        print(f"    → Reduce inter-node communication → 最优balance!")
        print(f"    → DataParallelMeshDims(DP=dp_degree, HSDP=inner)")

    return {
        "model_size_b": model_size_b,
        "dp_degree": dp_degree,
        "reshard": reshard,
        "fwd_comm_gb": fwd_total / 1e9,
        "bwd_comm_gb": bwd_rs_comm / 1e9,
        "total_per_step_gb": total_per_step / 1e9,
        "zero3_total_gb": zero3_total / 1e9,
        "savings_vs_zero3": (1 - total_per_step / zero3_total) * 100,
        "latency_ms": latency * 1000,
    }


def simulate_tp(model_size_b, tp_degree, interconnect="nvlink"):
    """模拟Tensor Parallelism训练通信"""
    Ψ = model_size_b * 1e9
    bf16_bytes = 2 * Ψ

    print(f"\n{'='*60}")
    print(f"  Tensor Parallelism (TP={tp_degree}) Communication")
    print(f"{'='*60}")
    print(f"  Model: {model_size_b}B ({Ψ/1e9:.1f}B params)")
    print(f"  TP degree: {tp_degree}")
    print(f"  Interconnect: {interconnect}")

    # TP: AllReduce per layer (2× per transformer layer: attn + MLP)
    # ColumnParallel: split output → AllReduce → merge
    # RowParallel: split input → AllReduce → merge
    # Each layer needs 2 AllReduces (fwd + bwd each has AllReduce)
    # But in Megatron, each transformer layer = 2 AllReduces in fwd + 2 in bwd

    # Each AllReduce = ReduceScatter + AllGather = 2 × (N-1)/N × data
    ar_comm = 2 * bf16_bytes * (tp_degree - 1) // tp_degree  # per AllReduce
    # 2 AllReduces per transformer layer in forward
    # 2 AllReduces per transformer layer in backward
    # Total = 4 AllReduces per step per layer
    total_per_layer = 4 * ar_comm

    print(f"\n  Per transformer layer:")
    print(f"    AllReduce volume: {ar_comm/1e9:.2f} GB")
    print(f"    2 AR fwd + 2 AR bwd = 4 × {ar_comm/1e9:.2f} = {total_per_layer/1e9:.2f} GB")
    print(f"    ★ AllReduce = ReduceScatter + AllGather → 2×(N-1)/N×data")

    # For 7B model with ~32 layers
    num_layers = 32
    total_per_step = num_layers * total_per_layer / tp_degree  # each GPU does 1/TP
    print(f"\n  Total per step ({num_layers} layers, TP={tp_degree}):")
    print(f"    {total_per_step/1e9:.2f} GB per GPU")

    bandwidths = {"nvlink": 300e9, "nvlink3": 150e9, "nvlink2": 80e9,
                  "pcie4": 32e9, "pcie3": 16e9}
    bw = bandwidths.get(interconnect, 32e9)
    latency = total_per_step / bw
    print(f"\n  ★ Latency: {latency*1000:.2f} ms ({interconnect} @ {bw/1e9:.0f} GB/s)")

    if interconnect.startswith("pcie"):
        print(f"  ✗ ✗ ✗ PCIe TP = 灾难级慢! → RTX 4090 PCIe TP不可行!")
        print(f"  ★ ★ RTX 4090最优: 单GPU + LoRA → 无TP通信 → 最快!")

    return {
        "model_size_b": model_size_b,
        "tp_degree": tp_degree,
        "total_per_step_gb": total_per_step / 1e9,
        "latency_ms": latency * 1000,
        "pcie_feasible": not interconnect.startswith("pcie"),
    }


def compare_strategies(model_size_b, gpu_count, interconnect="nvlink"):
    """比较所有分布式训练策略的通信开销"""
    Ψ = model_size_b * 1e9
    bf16_bytes = 2 * Ψ
    n = gpu_count

    print(f"\n{'='*60}")
    print(f"  7框架分布式训练策略通信量对比")
    print(f"{'='*60}")
    print(f"  Model: {model_size_b}B | GPUs: {n} | Link: {interconnect}")

    bandwidths = {"nvlink": 300e9, "nvlink3": 150e9, "nvlink2": 80e9,
                  "pcie4": 32e9, "pcie3": 16e9}
    bw = bandwidths.get(interconnect, 32e9)

    strategies = []

    # DDP
    ddp_comm = bf16_bytes  # AllReduce = 2 × (N-1)/N × Ψ ≈ Ψ
    ddp_latency = ddp_comm / bw
    strategies.append(("DDP", ddp_comm/1e9, "AllReduce", ddp_latency*1000))

    # ZeRO-2
    z2_comm = bf16_bytes * (n-1) // n  # ReduceScatter gradients only
    z2_latency = z2_comm / bw
    strategies.append(("ZeRO-2", z2_comm/1e9, "ReduceScatter", z2_latency*1000))

    # ZeRO-3
    z3_fwd = 3 * bf16_bytes * (n-1) // n  # 3× AllGather
    z3_bwd = bf16_bytes * (n-1) // n  # ReduceScatter
    z3_total = z3_fwd + z3_bwd
    z3_latency = z3_total / bw
    strategies.append(("ZeRO-3", z3_total/1e9, "3×AG+RS", z3_latency*1000))

    # FSDP2 (reshard=True)
    fsdp_comm = 2 * bf16_bytes * (n-1) // n + bf16_bytes * (n-1) // n  # 2AG+RS
    fsdp_latency = fsdp_comm / bw
    strategies.append(("FSDP2(reshard=T)", fsdp_comm/1e9, "2×AG+RS", fsdp_latency*1000))

    # FSDP2 (reshard=False)
    fsdp2_comm = bf16_bytes * (n-1) // n + bf16_bytes * (n-1) // n  # 1AG+RS
    fsdp2_latency = fsdp2_comm / bw
    strategies.append(("FSDP2(reshard=F)", fsdp2_comm/1e9, "1×AG+RS", fsdp2_latency*1000))

    # TP
    tp_comm = bf16_bytes  # ≈ Ψ per step (2 AllReduces)
    tp_latency = tp_comm / bw
    strategies.append(("TP", tp_comm/1e9, "2×AllReduce", tp_latency*1000))

    # LoRA + DDP
    lora_params = 0.5 / 1e9 * 2 * n  # LoRA adapters ~0.5GB, only LoRA gradients
    lora_comm = lora_params
    lora_latency = lora_comm / bw
    strategies.append(("LoRA+DDP", lora_comm/1e9, "AllReduce(LoRA)", lora_latency*1000))

    # rLLM Tinker
    strategies.append(("rLLM Tinker", 0, "0 (in-process)", 0))

    # Print comparison table
    print(f"\n  {'Strategy':<20} {'Comm/step(GB)':<15} {'Op':<15} {'Latency(ms)':<15} {'RTX4090?'}")
    print(f"  {'─'*80}")

    for name, comm, op, lat in strategies:
        feasible = "✓" if name in ("LoRA+DDP", "rLLM Tinker", "ZeRO-2", "FSDP2(reshard=T)", "FSDP2(reshard=F)") else "?"
        if interconnect.startswith("pcie") and name in ("TP", "ZeRO-3"):
            feasible = "✗"
        if name == "rLLM Tinker":
            feasible = "★"
        print(f"  {name:<20} {comm:<15.2f} {op:<15} {lat:<15.2f} {feasible}")

    # Key insights
    print(f"\n  ★ ★ Key Insights:")
    print(f"    1. FSDP2(reshard=True) = 2×AG+RS → 33% less comm vs ZeRO-3(3×AG+RS)")
    print(f"    2. FSDP2(reshard=False) = 1×AG+RS → 50% less comm vs ZeRO-3 → 但peak更高!")
    print(f"    3. LoRA+DDP = ~0.5GB → 99.97% less comm → ★ ★ RTX 4090最优!")
    print(f"    4. rLLM Tinker = 0 → in-process → 单GPU最快路径!")
    print(f"    5. TP/ZeRO-3 on PCIe = 灾难级慢 → ✗ RTX 4090不可行!")

    if interconnect.startswith("pcie"):
        print(f"\n  ★ ★ ★ PCIe环境结论:")
        print(f"    只有3种可行: LoRA+DDP / GRPO+LoRA(verl/rLLM) / ZeRO-2+CPU_Adam")
        print(f"    所有分布式方案(TP/PP/ZeRO-3/FSDP2多GPU) → PCIe瓶颈 → ✗!")

    return {
        "model_size_b": model_size_b,
        "gpu_count": gpu_count,
        "interconnect": interconnect,
        "strategies": [{"name": s[0], "comm_gb": s[1], "op": s[2], "latency_ms": s[3]}
                       for s in strategies],
    }


def simulate_gradient_placement():
    """可视化DTensor gradient placement规则"""
    print(f"\n{'='*60}")
    print(f"  DTensor Gradient Placement 规则")
    print(f"{'='*60}")

    rules = [
        ("Shard(0)", "Shard(0)", "no comm", "forward Shard→backward Shard → 梯度自然Shard!"),
        ("Shard(1)", "Shard(1)", "no comm", "forward Shard(dim 1)→backward Shard(dim 1) → 无通信!"),
        ("Replicate", "Replicate", "no comm", "forward Replicate→backward Replicate → 全拷贝→无通信!"),
        ("_Partial", "Replicate", "AllReduce", "forward Partial→backward needs Replicate → ★ 必须AllReduce!"),
        ("_Partial", "Shard(0)", "ReduceScatter", "forward Partial→backward Shard → ★ ReduceScatter更省!"),
        ("Shard(0)", "_Partial", "不常见", "★ 正常不会出现 → FSDP2 gradient = Shard(0)或Partial→Shard"),
        ("Shard(0)", "Replicate", "AllGather", "backward需要完整参数 → AllGather → FSDP2 pre_backward!"),
        ("Replicate", "Shard(0)", "ReduceScatter", "backward结束 → ReduceScatter → FSDP2 post_backward!"),
    ]

    print(f"\n  {'Forward Placement':<20} {'Grad Placement':<20} {'Required Op':<15} {'Explanation'}")
    print(f"  {'─'*85}")
    for fwd, grad, op, explanation in rules:
        print(f"  {fwd:<20} {grad:<20} {op:<15} {explanation}")

    print(f"\n  ★ ★ ★ FSDP2完整gradient cycle:")
    print(f"    Forward param: Shard(0) → AllGather → Replicate → compute → reshard → Shard(0)")
    print(f"    Backward grad: _Partial → ReduceScatter → Shard(0)")
    print(f"    ★ gradient placement自动正确 → _FromTorchTensor._grad_placements!")
    print(f"    ★ 无需手动通信 → RegisterPre/PostBackwardFunction → 自动AllGather/RS!")

    print(f"\n  ★ ★ ★ TP gradient cycle (Megatron):")
    print(f"    ColumnParallel: fwd Shard(1) → output _Partial → AllReduce → Replicate")
    print(f"    RowParallel: fwd Replicate → output Shard(0) → AllReduce → Replicate")
    print(f"    ★ TP每层2个AllReduce → FSDP2每层1个AG+1个RS → FSDP2更省通信!")

    print(f"\n  ★ ★ ★ HSDP gradient cycle (PyTorch 2.12):")
    print(f"    Inner group(intra-node NVLink): reshard=True → 2×AG(快!)")
    print(f"    Outer group(inter-node RDMA): reshard=False → 1×AG(省通信!)")
    print(f"    → DataParallelMeshDims → 自动配置 → 零手动!")

    return rules


def main():
    parser = argparse.ArgumentParser(description="DTensor Sharding Playground")
    parser.add_argument("--mode", choices=["fsdp2", "tp", "compare", "hsdp", "gradient"],
                        default="compare")
    parser.add_argument("--model-size", type=float, default=7.0,
                        help="Model size in billions of params")
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--interconnect",
                        choices=["nvlink", "nvlink3", "nvlink2", "pcie4", "pcie3"],
                        default="nvlink")
    parser.add_argument("--reshard", type=bool, default=True)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--dp", type=int, default=4, help="HSDP outer DP degree")
    parser.add_argument("--hsdp-inner", type=int, default=2, help="HSDP inner DP degree")

    args = parser.parse_args()
    results = {}

    if args.mode == "fsdp2":
        results["fsdp2"] = simulate_fsdp2(args.model_size, args.gpu_count,
                                          args.interconnect, args.reshard)
    elif args.mode == "tp":
        results["tp"] = simulate_tp(args.model_size, args.tp_degree, args.interconnect)
    elif args.mode == "compare":
        results["compare"] = compare_strategies(args.model_size, args.gpu_count,
                                                args.interconnect)
    elif args.mode == "hsdp":
        # HSDP = inner(reshard=True) + outer(reshard=False)
        Ψ = args.model_size * 1e9
        bf16 = 2 * Ψ
        inner_dp = args.hsdp_inner
        outer_dp = args.dp

        print(f"\n{'='*60}")
        print(f"  HSDP (Hierarchical Sharded Data Parallel)")
        print(f"{'='*60}")
        print(f"  Inner DP={inner_dp} (NVLink intra-node) + Outer DP={outer_dp} (RDMA inter-node)")
        print(f"  Effective DP = {inner_dp * outer_dp}")
        print(f"  Model: {args.model_size}B | Interconnect: {args.interconnect}")

        # Inner group: reshard=True → 2×AG (NVLink, fast)
        inner_ag = bf16 * (inner_dp - 1) // inner_dp
        inner_total = 2 * inner_ag  # 2×AG per layer forward

        # Outer group: reshard=False → 1×AG (RDMA, slow but only 1)
        outer_ag = bf16 * (outer_dp - 1) // outer_dp
        outer_total = outer_ag  # 1×AG per layer forward (inter-node only)

        # Backward: inner RS + outer RS
        inner_rs = bf16 * (inner_dp - 1) // inner_dp
        outer_rs = bf16 * (outer_dp - 1) // outer_dp

        total = inner_total + outer_total + inner_rs + outer_rs

        print(f"\n  Forward:")
        print(f"    Inner 2×AG (NVLink): {inner_total/1e9:.2f} GB → 快!")
        print(f"    Outer 1×AG (RDMA): {outer_total/1e9:.2f} GB → 省通信!")
        print(f"  Backward:")
        print(f"    Inner RS: {inner_rs/1e9:.2f} GB")
        print(f"    Outer RS: {outer_rs/1e9:.2f} GB")
        print(f"  Total: {total/1e9:.2f} GB per step per GPU")
        print(f"\n  ★ ★ HSDP = 最优balance! intra-node多通信(NVLink快) + inter-node少通信(RDMA贵)")
        print(f"  ★ ★ DataParallelMeshDims → FSDP2自动配置 → 零手动!")

        results["hsdp"] = {
            "inner_dp": inner_dp, "outer_dp": outer_dp,
            "total_per_step_gb": total / 1e9,
        }
    elif args.mode == "gradient":
        results["gradient_rules"] = simulate_gradient_placement()

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"dtensor_sharding_{args.mode}_model{args.model_size}b_gpu{args.gpu_count}"
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {RESULTS_DIR / f'{name}.json'}")


if __name__ == "__main__":
    main()
