#!/usr/bin/env python3
"""
Cross-Framework Communication Primitives Comparison

Compares collective communication operations across 7 frameworks:
- NVIDIA GPU: NCCL (standard), DeepEP (MoE-specific), Megatron (autograd wrappers)
- Ascend NPU: HCCL
- General: PyTorch c10d, gloo

Usage:
    python3 comm_primitives_comparison.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def build_comparison():
    """Build comprehensive communication primitives comparison"""

    # 1. Collective Operations
    collective_ops = {
        "AllReduce": {
            "NCCL": "Ring/Tree algorithm; 2×data volume (reduce+broadcast); auto-tuned",
            "HCCL": "Ring/Hierarchical; similar to NCCL; Ascend NPU optimized",
            "c10d": "torch.distributed.all_reduce(); dispatches to NCCL/HCCL/gloo",
            "DeepEP": "Not directly; uses AllGather+ReduceScatter decomposition for MoE",
            "Megatron": "_ReduceFromModelParallelRegion (fwd=AllReduce, bwd=copy)",
        },
        "AllGather": {
            "NCCL": "Ring AllGather; each rank gets full data; N×volume",
            "HCCL": "Ring AllGather; Ascend HCCS/RoCE optimized",
            "c10d": "torch.distributed.all_gather(); dispatches to backend",
            "DeepEP": "ElasticBuffer.dispatch (hybrid mode: NVLink AllGather + RDMA)",
            "Megatron": "_GatherFromSequenceParallelRegion (fwd=AllGather, bwd=ReduceScatter)",
        },
        "ReduceScatter": {
            "NCCL": "Ring ReduceScatter; each rank gets 1/N of reduced data; reduces to 1/N output",
            "HCCL": "Ring ReduceScatter; similar pattern",
            "c10d": "torch.distributed.reduce_scatter(); dispatches to backend",
            "DeepEP": "ElasticBuffer.combine (ReduceScatter with top-k weights)",
            "Megatron": "_ReduceScatterToSequenceParallelRegion (fwd=RS, bwd=AllGather)",
        },
        "All-to-All": {
            "NCCL": "Symmetric; each rank sends equal chunks to all others; padding waste for MoE",
            "HCCL": "Symmetric; same as NCCL pattern on Ascend",
            "c10d": "torch.distributed.all_to_all(); requires equal-sized chunks",
            "DeepEP": "Asymmetric dispatch/combine; MoE-optimized; no padding waste",
            "Megatron": "AlltoAllTokenDispatcher (via NCCL); FlexTokenDispatcher (dynamic)",
        },
        "Send/Recv (P2P)": {
            "NCCL": "isend/irecv; proxy thread model; limited overlap",
            "HCCL": "HcclSend/HcclRecv; similar pattern",
            "c10d": "torch.distributed.isend/irecv(); async ops",
            "DeepEP": "NCCL Gin put/get/signal; RDMA direct from GPU kernel; 0-SM inference",
            "Megatron": "P2PCommunicator (9 methods); batch_isend_irecv; two-group trick",
        },
        "Broadcast": {
            "NCCL": "Tree broadcast; 1×data volume; used for param init",
            "HCCL": "Tree broadcast; similar",
            "c10d": "torch.distributed.broadcast(); param init sync",
            "DeepEP": "NCCL Gin broadcast (experimental); AGRS session-based",
            "Megatron": "broadcast in _initialize_model_parallel_rng; param init",
        },
    }

    # 2. Framework-specific Communication Patterns
    framework_patterns = {
        "DeepSpeed": {
            "primary": "NCCL via torch.distributed",
            "ZeRO-1": "AllReduce gradients (same as DDP)",
            "ZeRO-2": "ReduceScatter gradients → each rank has 1/N grad",
            "ZeRO-3": "AllGather params (forward) + ReduceScatter grads (backward) = 3Ψ comm",
            "NVMe_Swap": "GDS RDMA (NVMe→GPU direct) or standard AIO (NVMe→CPU→GPU)",
            "overlap": "reduce-scatter overlap with backward (DeepSpeed Speed)",
        },
        "Megatron-LM": {
            "primary": "NCCL via torch.distributed + custom autograd wrappers",
            "TP": "AllReduce (1 per layer: RowParallel output)",
            "SP": "AllGather + ReduceScatter = AllReduce equivalent but P× memory savings",
            "PP": "P2P send/recv (9 methods); batched_p2p_ops; two-group trick for PP=2",
            "DP": "AllReduce gradients (standard) or ReduceScatter (ZeRO-compatible)",
            "MoE": "AlltoAll (NCCL) or AllGather+ReduceScatter or FlexTokenDispatcher",
        },
        "vLLM": {
            "primary": "NCCL via torch.distributed (TP only)",
            "TP": "AllReduce (V1: single AllReduce in Attention); TensorParallelLinear",
            "PP": "PipelinePrefill (experimental, vLLM V2)",
            "EP": "All-to-All (expert parallel, MoE serving)",
            "PD": "TCP/RDMA KV transfer (Mooncake/NIXL/fake backend)",
        },
        "verl": {
            "primary": "NCCL via torch.distributed (FSDP/Megatron worker) + vLLM rollout",
            "DP": "FSDP2 ReduceScatter+AllGather or DDP AllReduce",
            "rollout": "vLLM/SGLang async inference → no collective ops during rollout",
            "weight_sync": "naive(save+load) / NCCL direct / NIXL RDMA",
            "GRPO": "No special communication (advantage computed on driver)",
        },
        "rLLM": {
            "primary": "via verl backend (NCCL) or tinker backend (in-process)",
            "on_policy": "Tinker: no distributed ops (in-process); Verl: FSDP/Megatron",
            "async": "Tinker: SamplingClient refresh (weight sync); Verl: NCCL broadcast",
            "IS": "Tinker IS loss (in-process); Verl TIS/rollout_correction (distributed)",
        },
        "MindIE": {
            "primary": "HCCL (Ascend NPU)",
            "EP": "All-to-All via HCCL (expert parallel)",
            "PP": "HCCL Send/Recv (pipeline stages)",
            "PD": "HCCL-based KV transfer (roadmap)",
            "NVIDIA": "Not applicable (Ascend-only)",
        },
        "PyTorch": {
            "primary": "c10d (ProcessGroupNCCL/HCCL/Gloo)",
            "DDP": "AllReduce gradients (bucketed)",
            "FSDP1": "AllGather params + ReduceScatter grads (FlatParameter)",
            "FSDP2": "ReduceScatter grads + AllGather params (DTensor, per-param)",
            "compile": "AllReduce/AllGather as IR nodes → FSDP2 compatible, ZeRO-3 not",
        },
        "DeepEP": {
            "primary": "NCCL Gin (RDMA GPU Direct Access Kernel Interface)",
            "dispatch": "Asymmetric MoE: tokens→expert GPUs; Hybrid(NVLink+RDMA); FP8 halve traffic",
            "combine": "Weighted sum reduction back to original tokens",
            "overlap": "async_with_compute_stream + dual-stream + EventOverlap",
            "engram": "RDMA remote KV fetch (0-SM); ncclGin.get() from CPU segments",
            "AGRS": "AllGather+ReduceScatter (NVLink-only, experimental)",
        },
    }

    # 3. Hardware Backend Comparison
    hw_backends = {
        "NCCL": {
            "hardware": "NVIDIA GPU (NVLink, PCIe, RoCE, TCP)",
            "topo": "Auto-detect GPU topology (NVLink→SHM→NET)",
            "algorithms": "Ring, Tree, CollNet (NVLink-connected)",
            "overlap": "Proxy thread model; limited compute-comm overlap",
            "latency": "AllReduce ~1ms (8 GPU NVLink); P2P ~0.1ms",
            "memory": "Internal buffer management; fixed-size",
        },
        "HCCL": {
            "hardware": "Huawei Ascend NPU (HCCS, RoCE, PCIe)",
            "topo": "Auto-detect NPU topology",
            "algorithms": "Ring, Hierarchical, similar to NCCL",
            "overlap": "Similar to NCCL proxy model",
            "latency": "AllReduce ~similar to NCCL on comparable hardware",
            "memory": "Internal buffer management",
        },
        "NCCL_Gin": {
            "hardware": "NVIDIA GPU (NVLink, RDMA/InfiniBand) — SM90+",
            "topo": "Separate scaleup(NVLink)/scaleout(RDMA) domains",
            "algorithms": "Gin put/get/signal; RDMA direct from GPU kernel",
            "overlap": "0-SM inference kernels; async_with_compute_stream",
            "latency": "RDMA dispatch 77us (EP8); NVLink 726GB/s (EP8)",
            "memory": "ElasticBuffer (dynamic) + SymmetricMemory",
        },
        "Gloo": {
            "hardware": "CPU (TCP/SHM); fallback for non-GPU",
            "topo": "Manual configuration",
            "algorithms": "Ring, simple",
            "overlap": "None (CPU-only)",
            "latency": "10-100x slower than NCCL",
            "memory": "Managed by PyTorch",
        },
    }

    # 4. RTX 4090 Communication Constraints
    rtx4090_constraints = {
        "NVLink": "❌ No NVLink → PCIe only → TP/PP跨GPU瓶颈严重",
        "RDMA": "❌ No RDMA → TCP only → DeepEP Gin backend不可用",
        "PCIe": "12 GB/s Gen4 x16 → 7B模型AllReduce ~2.5s (vs NVLink 0.04s)",
        "SM90": "❌ SM89 (Ada) → DeepEP V2 kernel不可用",
        "single_GPU": "✅ 单GPU最优: 无跨GPU通信 → 无TP/PP/EP",
        "DP_only": "✅ 数据并行: AllReduce梯度 → 但8GPU PCIe scaling灾难",
        "recommended": "LoRA + CPU Adam + GRPO (rLLM TinkerBackend, 单GPU)",
    }

    results = {
        "collective_ops": collective_ops,
        "framework_patterns": framework_patterns,
        "hw_backends": hw_backends,
        "rtx4090_constraints": rtx4090_constraints,
    }

    return results

def main():
    results = build_comparison()

    output_path = RESULTS_DIR / "comm_primitives_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 70)
    print("Cross-Framework Communication Primitives Comparison")
    print("=" * 70)

    print("\n--- Key Collective Operations ---")
    for op, desc in results["collective_ops"].items():
        print(f"\n{op}:")
        for framework, detail in desc.items():
            print(f"  {framework}: {detail[:60]}...")

    print("\n--- Framework-specific Patterns ---")
    for fw, patterns in results["framework_patterns"].items():
        print(f"\n{fw}:")
        for pattern, detail in patterns.items():
            print(f"  {pattern}: {detail[:60]}...")

    print("\n--- RTX 4090 Constraints ---")
    for key, value in results["rtx4090_constraints"].items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
