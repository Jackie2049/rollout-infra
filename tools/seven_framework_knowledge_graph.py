#!/usr/bin/env python3
"""7框架知识图谱生成器 — 从所有reading notes生成跨框架知识关系图

Usage:
    python seven_framework_knowledge_graph.py [mode]

Modes:
    stats        — 统计各框架阅读数量和覆盖度
    connections  — 跨框架连接和依赖关系
    rtx4090      — RTX 4090最优路径生成
    all          — 运行所有模式
"""

import json
import sys
from pathlib import Path

# ============================================================
# 7 Framework Definitions
# ============================================================

FRAMEWORKS = {
    "DeepSpeed": {
        "repo": "deepspeedai/DeepSpeed",
        "category": "training",
        "key_feature": "ZeRO (Zero Redundancy Optimizer)",
        "rtx4090_status": "ZeRO-2+LoRA+CPU_Adam (可行, 但不如rLLM)",
        "readings": [
            "deepspeed-zero-reading.md",
            "deepspeed-zero3-data-flow.md",
            "deepspeed-prefetch-coordinator-reading.md",
            "deepspeed-nvme-swap-reading.md",
            "deepspeed-comm-overlap-reading.md",
            "deepspeed-distributed-optimizer-source-reading.md",
            "deepspeed-latest-developments-2026-06.md",
            "deepspeed-0.19-features-reading.md",
            "zero3-vs-fsdp2-system-comparison.md",
        ],
    },
    "Megatron-LM": {
        "repo": "NVIDIA/Megatron-LM",
        "category": "training+inference",
        "key_feature": "3D parallelism (TP+PP+DP) + MoE EP",
        "rtx4090_status": "TP>1/PP>1 PCIe灾难 (基本不可用)",
        "readings": [
            "megatron-source-reading.md (memory/)",
            "megatron-inference-engine-reading.md",
            "megatron-parallel-state-source-reading.md",
            "megatron-v0.17-latest-reading.md",
            "deepep-source-reading.md",
            "deepep-v2-reading.md",
            "deepep-megatron-integration-latest.md",
            "moe-serving-framework-comparison.md",
        ],
    },
    "vLLM": {
        "repo": "vllm-project/vllm",
        "category": "inference",
        "key_feature": "Paged Attention + Continuous Batching",
        "rtx4090_status": "INT4+INT8KV+EAGLE→9,088 tok/s (最优推理!)",
        "readings": [
            "vllm-source-reading.md (memory/, 7篇)",
            "vllm-v1-gpu-model-runner-reading.md",
            "vllm-speculative-decoding-reading.md",
            "vllm-lora-serving-reading.md",
            "vllm-v1-scheduler-vs-sglang-overlap-scheduling.md",
            "vllm-v1-kv-cache-management-reading.md",
            "vllm-cuda-graph-reading.md",
            "vllm-mrv2-architecture-reading.md",
            "vllm-v0.23-new-features-reading.md",
            "vllm-ascend-serving-layer-reading.md",
            "vllm-pr-45157-resubmission-draft.md",
        ],
    },
    "verl": {
        "repo": "volcengine/verl",
        "category": "rl_training",
        "key_feature": "GRPO/PPO RL training with vLLM/SGLang rollout",
        "rtx4090_status": "HYBRID+naive+GRPO+LoRA (可行)",
        "readings": [
            "verl-grpo-data-flow-reading.md",
            "verl-worker-lifecycle-ray-weight-sync-reading.md",
            "verl-multi-turn-agent-loop-reading.md",
            "verl-checkpoint-management-reading.md",
            "verl-fully-async-policy-reading.md",
            "verl-grpo-training-loop-internals-reading.md",
            "verl-ppo-vs-grpo-training-loop-comparison.md",
        ],
    },
    "MindIE": {
        "repo": "Ascend NPU (Huawei)",
        "category": "inference (Ascend)",
        "key_feature": "ATB kernel + CANN + HCCL",
        "rtx4090_status": "不适用 (NVIDIA GPU → vLLM更优)",
        "readings": [
            "mindie-architecture-reading.md",
            "mindie-atb-kernel-architecture-reading.md",
            "vllm-ascend-serving-layer-reading.md",
            "hccl-vs-nccl-reading.md",
            "openmind-architecture-reading.md",
        ],
    },
    "rLLM": {
        "repo": "rllm-org/rLLM",
        "category": "rl_training+eval",
        "key_feature": "TinkerBackend (in-process) + GRPO",
        "rtx4090_status": "最优! Tinker+GRPO+LoRA+bypass_mode (单GPU最快)",
        "readings": [
            "rllm-architecture-reading.md",
            "rllm-gateway-backend-trainer-source-reading.md",
            "rllm-tinker-backend-deep-reading.md",
            "rllm-v0.3-terminal-rl-reading.md",
        ],
    },
    "PyTorch": {
        "repo": "pytorch/pytorch",
        "category": "foundation",
        "key_feature": "torch.compile + FSDP2 + DTensor",
        "rtx4090_status": "compile✓ / FSDP2需多GPU / LoRA+compile可行",
        "readings": [
            "pytorch-compile-e2e-reading.md",
            "pytorch-dynamo-internals-reading.md",
            "pytorch-fx-ir-source-reading.md",
            "pytorch-aotautograd-internals-reading.md",
            "pytorch-fsdp2-internals-reading.md",
            "pytorch-dtensor-autograd-reading.md",
            "pytorch-dtensor-source-reading.md",
            "pytorch-custom-op-library-system-reading.md",
            "pytorch-inductor-triton-codegen-reading.md",
            "pytorch-compiler-roadmap-26-28.md",
            "pytorch-compile-stack-knowledge-synthesis.md",
        ],
    },
}

# Cross-framework connections
CONNECTIONS = {
    "verl→vLLM": "verl uses vLLM as rollout engine (ServerAdapter+vLLMHttpServer)",
    "verl→SGLang": "verl also supports SGLang as rollout engine",
    "rLLM→vLLM": "rLLM uses vLLM/SGLang via SamplingClient for inference",
    "DeepSpeed→HF": "ZeRO checkpoints→universal→HF format→vLLM serving",
    "FSDP2→HF": "FSDP checkpoints→FSDPModelMerger→HF→vLLM",
    "Megatron→TRT-LLM": "Megatron training→TRT-LLM export→production serving",
    "vLLM→FlashInfer": "vLLM uses FlashInfer for attention+sampling",
    "vLLM→EAGLE": "EAGLE speculative decoding→9,088 tok/s INT4",
    "DeepSpeed→AutoEP": "AutoEP→MoE training→ZeRO-0/1/2→vs Megatron DeepEP",
    "Megatron→DeepEP": "DeepEP V1/HybridEP→asymmetric→4.6x→SM90 only",
    "MindIE→vLLM-Ascend": "vLLM-Ascend=NPU适配→ATB→HCCL→Ascend serving",
    "PyTorch→FSDP2": "FSDP2=DTensor+per-param→compile兼容→未来方向",
    "PyTorch→verl": "verl uses FSDP2 for training→fully_shard→DTensor",
    "PyTorch→rLLM": "rLLM uses torch.compile(reduce-overhead) for training acceleration",
    "PyTorch→DeepSpeed": "DeepCompile→ZeRO-3+compile→分段compile→有值",
    "GRPO→推理scaling": "GRPO rollout_n=best-of-N→推理时compute scaling",
    "LoRA→vLLM": "LoRA merge→INT4→vLLM→4,791 tok/s→最快推理路径",
    "LoRA→rLLM": "LoRA auto-init→TinkerBackend→zero-copy weight sync",
}

# RTX 4090 optimal paths
RTX4090_PATHS = {
    "training": {
        "framework": "rLLM TinkerBackend",
        "algorithm": "GRPO + LoRA-32",
        "key_features": ["bypass_mode=true", "fused fwd-bwd-optim", "zero-copy weight sync"],
        "memory": "~17GB (7B BF16 + LoRA)",
        "throughput": "~19,743 tok/s training",
    },
    "inference": {
        "framework": "vLLM",
        "configuration": "INT4 + INT8KV + GQA-8 + prefix caching",
        "key_features": ["CUDA graph (FULL decode)", "FlashInfer", "EAGLE speculative"],
        "throughput_basic": "4,791 tok/s",
        "throughput_eagle": "9,088 tok/s (8.3x!)",
    },
    "evaluation": {
        "framework": "rLLM",
        "method": "pass@k (--attempts N)",
        "key_features": ["warm-pool (CPU)", "snapshot acceleration", "Terminus-2 harness"],
        "gpu_needed": "No (CPU only)",
    },
    "full_pipeline": "rLLM Tinker+GRPO+LoRA+bypass → merge → HF → INT4 vLLM → 4,791 tok/s → EAGLE → 9,088 tok/s",
}


def compute_stats():
    """Compute reading statistics for each framework."""
    stats = {}
    notebook_dir = Path("notebook/projects")
    fundamentals_dir = Path("notebook/fundamentals")

    for name, fw in FRAMEWORKS.items():
        total_readings = len(fw["readings"])
        found_files = []
        for reading in fw["readings"]:
            # Check in notebook/projects or memory/
            filename = reading.replace(" (memory/, 7篇)", "").replace(" (memory/)", "")
            if notebook_dir.joinpath(filename).exists():
                found_files.append(str(notebook_dir.joinpath(filename)))
            elif fundamentals_dir.joinpath(filename).exists():
                found_files.append(str(fundamentals_dir.joinpath(filename)))

        stats[name] = {
            "total_readings": total_readings,
            "found_files": len(found_files),
            "coverage_pct": round(len(found_files) / total_readings * 100, 1) if total_readings > 0 else 0,
            "category": fw["category"],
            "key_feature": fw["key_feature"],
            "rtx4090_status": fw["rtx4090_status"],
        }

    # Overall stats
    total = sum(s["total_readings"] for s in stats.values())
    found = sum(s["found_files"] for s in stats.values())
    stats["_overall"] = {
        "total_readings": total,
        "found_files": found,
        "coverage_pct": round(found / total * 100, 1),
    }

    return stats


def compute_connections():
    """Compute cross-framework connections."""
    return {
        "connections": CONNECTIONS,
        "framework_roles": {
            "DeepSpeed": "training infra (ZeRO + AutoEP)",
            "Megatron-LM": "training+inference (TP+PP+MoE)",
            "vLLM": "inference serving (Paged Attention)",
            "verl": "RL training orchestration (GRPO+PPO)",
            "MindIE": "inference (Ascend NPU)",
            "rLLM": "RL training+evaluation (Tinker+AgentFlow)",
            "PyTorch": "foundation (compile+FSDP2+DTensor)",
        },
        "data_flow": "HF format → universal checkpoint → vLLM/SGLang/TRT-LLM serving",
        "rl_data_flow": "actor rollout → reward → advantage → policy update → weight sync → next iteration",
    }


def compute_rtx4090():
    """Compute RTX 4090 optimal paths."""
    return {
        "paths": RTX4090_PATHS,
        "key_conclusions": [
            "GRPO > PPO (省50%内存+compute) → RTX 4090唯一可行RL训练方法",
            "rLLM Tinker > verl HYBRID (auto-init LoRA + in-process + bypass_mode)",
            "INT4 + INT8KV + EAGLE → 9,088 tok/s → RTX 4090推理最优",
            "PCIe scaling灾难 → 单GPU最优 → TP>1/PP>1不可行",
            "BF16唯一正确训练精度 → INT4唯一正确推理精度",
            "LoRA merge后推理同(无overhead) → merge→INT4→vLLM→最快",
            "Rule-based reward > RM(14GB GPU) → GRPO典型用math/code correctness",
            "pass@k eval(CPU) + GRPO train(GPU) → 最大化GPU利用率",
        ],
        "memory_budget": {
            "training": "17GB (7B BF16 + LoRA-32) → 24GB ✓",
            "inference": "~11GB (INT4 3.5GB + KV 5GB + graph 2GB + misc 0.5GB) → 24GB ✓",
            "ppo_training": "~48GB → 24GB ✗✗✗",
            "fsdp2_training": "~26GB → 24GB ✗ (需LoRA)",
        },
    }


def print_results(results, mode):
    """Print results in readable format."""
    print("\n" + "=" * 60)
    print(f"  7 Framework Knowledge Graph — {mode.upper()}")
    print("=" * 60)

    for key, value in results.items():
        print(f"\n### {key} ###")
        print(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    args = sys.argv[1:]
    mode = "all" if not args else args[0]

    all_results = {}

    if mode in ["stats", "all"]:
        all_results["stats"] = compute_stats()

    if mode in ["connections", "all"]:
        all_results["connections"] = compute_connections()

    if mode in ["rtx4090", "all"]:
        all_results["rtx4090"] = compute_rtx4090()

    for m, r in all_results.items():
        print_results(r, m)

    # Save
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"seven_framework_knowledge_graph_{mode}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
