#!/usr/bin/env python3
"""
Distributed Training Communication Cost Calculator

Quantifies communication overhead (volume, latency, bandwidth requirements)
for different distributed training strategies across 7 frameworks.

Communication patterns:
- AllReduce: 2× data volume (ReduceScatter + AllGather)
- ReduceScatter: 1× data volume
- AllGather: 1× data volume
- P2P send/recv: varies by pipeline stage
- All-to-All: varies by MoE expert routing

Key formulas:
- Ψ = model_size_b × 1e9 (total parameters)
- BF16 model bytes = 2Ψ
- Communication volume depends on parallelism degree and pattern

Usage:
    python3 comm_cost_calculator.py
    python3 comm_cost_calculator.py --model-size 7 --gpu-count 8 --interconnect nvlink
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Hardware bandwidth estimates (GB/s for payload data)
HW_BANDWIDTH = {
    "nvlink_a100": {"bidirectional": 300, "allreduce_per_link": 25},
    "nvlink_h100": {"bidirectional": 450, "allreduce_per_link": 35},
    "nvlink_h200": {"bidirectional": 600, "allreduce_per_link": 45},
    "pcie_gen4_x16": {"bidirectional": 25, "allreduce_per_link": 3},
    "pcie_gen5_x16": {"bidirectional": 32, "allreduce_per_link": 4},
    " roce_200g": {"bidirectional": 25, "allreduce_per_link": 2.5},
    " roce_400g": {"bidirectional": 50, "allreduce_per_link": 5},
    "hccs_910b": {"bidirectional": 196, "allreduce_per_link": 28},
    "hccs_910c": {"bidirectional": 240, "allreduce_per_link": 35},
}

def calc_allreduce_time(data_gb, gpu_count, hw_key, algorithm="ring"):
    """Estimate AllReduce time for given data volume and hardware"""
    bw = HW_BANDWIDTH.get(hw_key, HW_BANDWIDTH["nvlink_a100"])
    # Ring AllReduce: 2*(N-1)/N steps, each step transfers data/N
    # Effective: 2*(N-1)/N * data / (per_link_bw * N/2) ... simplified:
    # For ring: time ≈ 2 * data_gb / (per_link_bw * N)
    # For tree: time ≈ 2 * data_gb / (total_bw)
    if algorithm == "ring":
        # Ring: throughput = per_link_bw * N (N links in parallel)
        throughput = bw["allreduce_per_link"] * gpu_count
        return data_gb / throughput * 1000  # ms
    else:  # tree/hierarchical
        throughput = bw["bidirectional"] * gpu_count / 2
        return data_gb / throughput * 1000  # ms

def calc_comm_costs(model_size_b, gpu_count, interconnect, training_type,
                    lora_rank, tp_size, pp_size, dp_size):
    """Calculate communication costs for all distributed strategies"""

    Ψ = model_size_b * 1e9
    bf16_model_gb = 2 * Ψ / 1e9  # BF16 model size in GB
    fp32_optimizer_gb = 12 * Ψ / 1e9  # Adam states in GB

    # Select hardware key based on interconnect
    if interconnect == "nvlink":
        hw_key = "nvlink_a100"  # default, can override
    elif interconnect == "pcie":
        hw_key = "pcie_gen4_x16"
    elif interconnect == "hccs":
        hw_key = "hccs_910b"
    else:
        hw_key = "pcie_gen4_x16"

    results = {}
    N = gpu_count

    # LoRA reduction
    if lora_rank > 0:
        lora_fraction = 2 * lora_rank / 4096
        lora_param_gb = bf16_model_gb * lora_fraction
        trainable_comm_gb = lora_param_gb  # only LoRA grads need AllReduce
    else:
        lora_fraction = 0
        trainable_comm_gb = bf16_model_gb  # full param grads

    # ===== 1. DDP (AllReduce gradients) =====
    ddp_grad_gb = trainable_comm_gb
    ddp_time_ms = calc_allreduce_time(ddp_grad_gb, N, hw_key)
    results["DDP"] = {
        "pattern": "AllReduce(gradients)",
        "volume_per_step_gb": round(ddp_grad_gb, 2),
        "volume_desc": f"2×{trainable_comm_gb:.2f}GB gradients → AllReduce",
        "time_ms": round(ddp_time_ms, 2),
        "bandwidth_needed_gbs": round(ddp_grad_gb / (ddp_time_ms / 1000), 1) if ddp_time_ms > 0 else 0,
        "frequency": "every backward pass",
        "lora_effect": f"LoRA r={lora_rank}: {lora_fraction*100:.1f}% of full → {lora_param_gb:.2f}GB vs {bf16_model_gb:.2f}GB" if lora_rank > 0 else "none",
    }

    # ===== 2. ZeRO-1 (AllReduce grads + sharded optimizer) =====
    zero1_grad_gb = trainable_comm_gb  # same AllReduce
    zero1_time_ms = calc_allreduce_time(zero1_grad_gb, N, hw_key)
    results["ZeRO-1"] = {
        "pattern": "AllReduce(gradients) [same as DDP]",
        "volume_per_step_gb": round(zero1_grad_gb, 2),
        "time_ms": round(zero1_time_ms, 2),
        "optimizer_comm_gb": round(fp32_optimizer_gb / N, 2),
        "note": "Optimizer sharded → no extra comm, just AllReduce grads",
    }

    # ===== 3. ZeRO-2 (ReduceScatter grads + sharded optimizer+grads) =====
    zero2_volume_gb = trainable_comm_gb  # ReduceScatter ≈ AllReduce in volume
    zero2_time_ms = calc_allreduce_time(zero2_volume_gb, N, hw_key) / 2
    # ReduceScatter is half of AllReduce (just scatter phase)
    # But then need AllGather for updated params → actually same total as AllReduce
    # ZeRO-2: ReduceScatter(grads) = 1× data/N per step per GPU
    # Actually: ReduceScatter volume = data/N per GPU, but total across all GPUs = data
    # Time ≈ similar to AllReduce for ring algorithm
    results["ZeRO-2"] = {
        "pattern": "ReduceScatter(gradients)",
        "volume_per_step_gb": round(zero2_volume_gb, 2),
        "time_ms": round(calc_allreduce_time(zero2_volume_gb, N, hw_key), 2),
        "optimizer_comm_gb": 0,  # optimizer local (sharded)
        "note": "ReduceScatter ≈ AllReduce bandwidth but sharded optimizer saves memory",
    }

    # ===== 4. ZeRO-2 + CPU Adam =====
    zero2_cpu_grad_gb = trainable_comm_gb
    results["ZeRO-2+CPU_Adam"] = {
        "pattern": "ReduceScatter(gradients) + CPU↔GPU optimizer transfer",
        "volume_per_step_gb": round(zero2_cpu_grad_gb, 2),
        "time_ms": round(calc_allreduce_time(zero2_cpu_grad_gb, N, hw_key), 2),
        "cpu_transfer_gb": round(fp32_optimizer_gb * lora_fraction / N if lora_rank > 0 else fp32_optimizer_gb / N, 2),
        "cpu_transfer_time_ms": round(fp32_optimizer_gb * lora_fraction / N * 1000 / 50 if lora_rank > 0 else fp32_optimizer_gb / N * 1000 / 50, 1),
        "note": "CPU optimizer offload → PCIe transfer for optimizer states",
        "rtx4090_warning": "PCIe CPU transfer adds latency but saves GPU memory",
    }

    # ===== 5. ZeRO-3 (3Ψ: AllGather fwd + RS bwd + AllGather bwd) =====
    zero3_volume_gb = bf16_model_gb * 3  # 3× model size communication
    zero3_fwd_ag_gb = bf16_model_gb
    zero3_bwd_rs_gb = bf16_model_gb
    zero3_bwd_ag_gb = bf16_model_gb
    zero3_time_ms = calc_allreduce_time(bf16_model_gb, N, hw_key) * 3
    results["ZeRO-3"] = {
        "pattern": "AllGather(params,fwd) + ReduceScatter(grads,bwd) + AllGather(params,bwd)",
        "volume_per_step_gb": round(zero3_volume_gb, 2),
        "volume_breakdown": {
            "forward_AllGather": round(zero3_fwd_ag_gb, 2),
            "backward_ReduceScatter": round(zero3_bwd_rs_gb, 2),
            "backward_AllGather": round(zero3_bwd_ag_gb, 2),
        },
        "time_ms": round(zero3_time_ms, 2),
        "time_breakdown_ms": {
            "forward_AllGather": round(calc_allreduce_time(zero3_fwd_ag_gb, N, hw_key), 2),
            "backward_ReduceScatter": round(calc_allreduce_time(zero3_bwd_rs_gb, N, hw_key) / 2, 2),
            "backward_AllGather": round(calc_allreduce_time(zero3_bwd_ag_gb, N, hw_key), 2),
        },
        "overlap_potential": "PartitionedParameterCoordinator can prefetch next layer AllGather during compute",
        "note": "3Ψ total communication → high overhead on PCIe → NVLink required for efficiency",
    }

    # ===== 6. FSDP2 (2Ψ: AllGather fwd + ReduceScatter bwd) =====
    fsdp2_volume_gb = bf16_model_gb * 2  # 2× model size
    fsdp2_fwd_ag_gb = bf16_model_gb
    fsdp2_bwd_rs_gb = bf16_model_gb
    fsdp2_time_ms = calc_allreduce_time(bf16_model_gb, N, hw_key) * 2
    results["FSDP2"] = {
        "pattern": "AllGather(params,fwd) + ReduceScatter(grads,bwd)",
        "volume_per_step_gb": round(fsdp2_volume_gb, 2),
        "volume_breakdown": {
            "forward_AllGather": round(fsdp2_fwd_ag_gb, 2),
            "backward_ReduceScatter": round(fsdp2_bwd_rs_gb, 2),
        },
        "time_ms": round(fsdp2_time_ms, 2),
        "vs_zero3": "2Ψ vs 3Ψ → 33% less communication",
        "no_backward_AllGather_reason": "FSDP2 keeps full param after forward → reshard after backward → no 3rd AllGather",
        "compile_compatible": True,
        "note": "torch.compile sees static shapes → max kernel fusion → compensates for 2Ψ overhead",
    }

    # ===== 7. Megatron TP (AllReduce per layer) =====
    if tp_size > 1:
        tp_comm_per_layer_gb = bf16_model_gb / tp_size / model_size_b * 2  # rough per-layer
        # Each transformer layer: ColumnParallel→AllReduce, RowParallel→AllReduce = 2× per layer
        # Number of layers ≈ model_size / (hidden² × 12/7) ... simplified
        num_layers = int(model_size_b * 1e9 / (4096 * 4096 * 12 / 7 * 1e9))  # rough
        tp_total_gb = bf16_model_gb / tp_size * 2 * num_layers * 0.01  # rough per step
        # Actually: each AllReduce is just hidden_dim × seq_len × batch, much smaller than full model
        tp_allreduce_size_gb = (4096 * 2048 * 2) / 1e9 * tp_size  # per layer, BF16
        tp_time_per_layer_ms = calc_allreduce_time(tp_allreduce_size_gb, tp_size, hw_key)
        # Total: 2 AllReduce per layer × num_layers
        # With SP: only 1 AllReduce per layer (ReduceScatter + AllGather = AllReduce but saves memory)
        results["Megatron_TP"] = {
            "pattern": "AllReduce per transformer layer (Column→Row)",
            "volume_per_layer_gb": round(tp_allreduce_size_gb * 2, 3),
            "allreduces_per_layer": 2,
            "time_per_layer_ms": round(tp_time_per_layer_ms * 2, 2),
            "total_layers": num_layers,
            "sp_variant": "Sequence Parallel: ReduceScatter+AllGather = same volume but saves activation memory",
            "note": f"TP={tp_size}: each GPU holds 1/{tp_size} of layer → 2 AllReduce per layer (small volume)",
            "rtx4090_warning": f"TP={tp_size} on PCIe: AllReduce {tp_time_per_layer_ms:.1f}ms per layer → very slow!",
        }
    else:
        results["Megatron_TP"] = {
            "pattern": "TP=1 → no intra-layer communication",
            "volume_per_step_gb": 0,
            "note": "Single GPU → no TP communication needed",
        }

    # ===== 8. Megatron PP (P2P send/recv) =====
    if pp_size > 1:
        # PP: each stage sends activation tensor to next stage
        # Activation size ≈ hidden_dim × seq_len × batch × 2 (BF16)
        pp_activation_gb = 4096 * 2048 * 2 / 1e9  # per microbatch, rough
        pp_p2p_time_ms = pp_activation_gb / HW_BANDWIDTH[hw_key]["bidirectional"] * 1000
        # For interleaved: more P2P but smaller per-send
        results["Megatron_PP"] = {
            "pattern": "P2P send/recv activations between pipeline stages",
            "volume_per_send_gb": round(pp_activation_gb, 3),
            "p2p_time_ms": round(pp_p2p_time_ms, 2),
            "sends_per_microbatch": pp_size - 1,
            "total_fwd_volume_gb": round(pp_activation_gb * (pp_size - 1), 3),
            "bubble_fraction": f"1F1B: {(pp_size-1)/(pp_size)}:{1/(pp_size)} = {round((pp_size-1)/pp_size*100,1)}% bubble",
            "interleaved_bubble": f"VP=2: {round((pp_size-1)/(pp_size*2)*100,1)}% bubble",
            "rtx4090_warning": f"PCIe P2P: {pp_p2p_time_ms:.1f}ms per send → pipeline stall → unusable",
        }
    else:
        results["Megatron_PP"] = {
            "pattern": "PP=1 → no pipeline communication",
            "volume_per_step_gb": 0,
            "note": "Single GPU → no PP communication needed",
        }

    # ===== 9. Megatron MoE EP (All-to-All) =====
    if dp_size > 1 and tp_size == 1 and pp_size == 1:
        # MoE All-to-All: token routing to experts
        # Volume: num_tokens × hidden_dim × 2 per expert pair
        ep_volume_gb = 0  # placeholder
        results["Megatron_MoE_EP"] = {
            "pattern": "All-to-All dispatch + All-to-All combine",
            "volume_approx": "2 × num_tokens × hidden_dim × 2 bytes per dispatch+combine",
            "note": "MoE EP comm depends on routing; DeepEP asymmetric reduces padding waste",
        }

    # ===== 10. verl GRPO (rollout + training comm) =====
    if training_type == "grpo":
        # verl GRPO on single GPU: no distributed comm
        # verl GRPO multi-GPU with FSDP2: same as FSDP2 pattern
        # verl rollout (vLLM): if separate GPU, need to transfer model weights
        results["verl_GRPO"] = {
            "pattern": "GRPO advantage = group mean/std → no critic forward needed",
            "single_gpu": {
                "volume_per_step_gb": 0,
                "note": "Single GPU → no distributed ops → TinkerBackend fastest",
            },
            "multi_gpu_fsdp2": {
                "volume_per_step_gb": round(fsdp2_volume_gb, 2),
                "note": "Same as FSDP2 pattern + rollout generation overhead",
            },
            "rollout_transfer_gb": round(bf16_model_gb, 2),
            "no_critic_saving": f"Saves {bf16_model_gb:.2f}GB critic model + {bf16_model_gb:.2f}GB critic optimizer",
        }

    # ===== 11. rLLM Tinker (in-process, no comm) =====
    results["rLLM_Tinker"] = {
        "pattern": "In-process → no distributed communication",
        "volume_per_step_gb": 0,
        "time_ms": 0,
        "note": "TinkerBackend: same process → LoRA weight sync via checkpoint file → no GPU-to-GPU transfer",
        "sync_mechanism": "save_checkpoint → new SamplingClient → file-based → ~1-2s overhead per sync",
        "rtx4090_optimal": True,
    }

    # ===== Summary =====
    summary = {}
    for name, data in results.items():
        vol = data.get("volume_per_step_gb", 0)
        time = data.get("time_ms", data.get("time_per_layer_ms", 0))
        summary[name] = {
            "volume_gb": vol,
            "time_ms": time,
            "pattern": data.get("pattern", ""),
        }

    results["_summary"] = summary
    results["_config"] = {
        "model_size_b": model_size_b,
        "gpu_count": gpu_count,
        "interconnect": interconnect,
        "training_type": training_type,
        "lora_rank": lora_rank,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "bf16_model_gb": bf16_model_gb,
        "hardware_bandwidth_key": hw_key,
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Communication Cost Calculator")
    parser.add_argument("--model-size", type=float, default=7)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--interconnect", type=str, default="nvlink",
                       choices=["nvlink", "pcie", "hccs", "none"])
    parser.add_argument("--training-type", type=str, default="grpo",
                       choices=["full", "lora", "grpo", "inference", "serving"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=8)
    args = parser.parse_args()

    # Validate parallelism
    assert args.tp_size * args.pp_size * args.dp_size <= args.gpu_count, \
        f"TP({args.tp_size})×PP({args.pp_size})×DP({args.dp_size}) > GPU count({args.gpu_count})"

    results = calc_comm_costs(
        args.model_size, args.gpu_count, args.interconnect,
        args.training_type, args.lora_rank, args.tp_size, args.pp_size, args.dp_size
    )

    output_path = RESULTS_DIR / "comm_cost_calculator.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("=" * 75)
    print(f"Communication Cost Calculator ({args.model_size}B, {args.gpu_count}×GPU, {args.interconnect})")
    print(f"Parallelism: TP={args.tp_size}, PP={args.pp_size}, DP={args.dp_size}")
    print("=" * 75)

    for name, data in results.items():
        if name.startswith("_"):
            continue
        vol = data.get("volume_per_step_gb", 0)
        time = data.get("time_ms", 0)
        pattern = data.get("pattern", "")
        print(f"\n--- {name} ---")
        print(f"  Pattern: {pattern}")
        print(f"  Volume: {vol:.2f} GB/step")
        print(f"  Time: {time:.2f} ms/step")
        if "note" in data:
            print(f"  Note: {data['note']}")

    print(f"\n{'='*75}")
    print("Key Insights:")
    print(f"  1. ZeRO-3 = 3Ψ comm → {results['ZeRO-3']['volume_per_step_gb']:.2f}GB → PCIe disaster")
    print(f"  2. FSDP2 = 2Ψ comm → {results['FSDP2']['volume_per_step_gb']:.2f}GB → 33% less than ZeRO-3")
    print(f"  3. DDP/ZeRO-2 = 1Ψ grad comm → {results['DDP']['volume_per_step_gb']:.2f}GB → LoRA reduces to {results['DDP'].get('lora_effect', 'N/A')}")
    print(f"  4. rLLM Tinker = 0 distributed comm → RTX 4090 single GPU optimal")
    print(f"  5. verl GRPO = no critic → saves entire critic model communication")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
