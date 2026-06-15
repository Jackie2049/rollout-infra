#!/usr/bin/env python3
"""
RTX 4090 SM89 Compatibility Matrix — vLLM Feature Checker
============================================================
检查SM89(RTX 4090/L4)的vLLM特性兼容性 → 直接支持vLLM #44879/#45038/#44701贡献

★★★ SM89核心限制: FP8 E5M2✗ / FP8训练✗ / FP8通信✗ / NVLS✗ / TMA✗
★★★ SM89可行路径: INT4 GPTQ + INT8 KV + BF16训练 + FlashInfer + CUDA graph(NVLS-free)

Usage:
  python tools/sm89_compatibility_checker.py              # 检查SM89兼容性矩阵
  python tools/sm89_compatibility_checker.py --mode detail  # 详细分析
  python tools/sm89_compatibility_checker.py --mode issues   # 相关vLLM issues
"""

import argparse
import json
from pathlib import Path

# ============================================================
# SM89 (RTX 4090 / L4) Compatibility Matrix
# ============================================================
SM89_COMPAT = {
    "compute_capability": 8.9,
    "gpu_names": ["RTX 4090", "NVIDIA L4"],
    "memory_gb": 24,
    "architecture": "Ada Lovelace",

    # ★★★ Attention Backend Compatibility
    "attention_backends": {
        "flash_attn_2": {"compatible": True, "notes": "✓ SM89支持, head_dim≤256"},
        "flash_attn_3": {"compatible": False, "notes": "✗ FA3需要SM90+ (Hopper)"},
        "flashinfer_decode": {"compatible": True, "notes": "✓ SM89支持FlashInfer decode"},
        "flashinfer_prefill": {"compatible": True, "notes": "✓ SM89支持FlashInfer prefill"},
        "flashinfer_fp8": {"compatible": False, "notes": "✗ FlashInfer FP8需要SM90+ → #44879"},
        "sdpa": {"compatible": True, "notes": "✓ PyTorch SDPA always works"},
        "xformers": {"compatible": True, "notes": "✓ 但不如FlashInfer快"},
    },

    # ★★★ Quantization Compatibility
    "quantization": {
        "bf16": {"compatible": True, "notes": "✓ BF16是唯一正确训练精度"},
        "fp16": {"compatible": True, "notes": "✓ 但BF16更推荐(训练稳定性)"},
        "int4_gptq": {"compatible": True, "notes": "✓ GPTQ INT4 → Marlin kernel on SM89"},
        "int4_awq": {"compatible": True, "notes": "✓ AWQ INT4 → Marlin kernel"},
        "int8_kv_cache": {"compatible": True, "notes": "✓ INT8 KV cache → FlashInfer支持"},
        "fp8_e4m3_kv": {"compatible": False, "notes": "✗ FP8 KV需要SM90+ → #44879/#45038"},
        "fp8_e5m2_weights": {"compatible": False, "notes": "✗ FP8 E5M2 inference需要SM90"},
        "fp8_training": {"compatible": False, "notes": "✗ FP8训练需要SM90+ TE"},
        "int4_triton_fallback": {"compatible": True, "notes": "✓ PR#43731 TritonW4A16 → non-Marlin shapes"},
    },

    # ★★★ CUDA Graph Compatibility
    "cuda_graph": {
        "basic_cuda_graph": {"compatible": True, "notes": "✓ 基本CUDA graph → SM89支持"},
        "nvls_allgather": {"compatible": False, "notes": "✗ NVLS需要SM90+ (Hopper)"},
        "tma_load": {"compatible": False, "notes": "✗ TMA需要SM90+ (Hopper)"},
        "breakable_cg": {"compatible": True, "notes": "✓ BCG(v0.23) → 但MRv2不支持量化"},
        "flashinfer_cg": {"compatible": True, "notes": "✓ FlashInfer FULL_DECODE_ONLY"},
    },

    # ★★★ Communication Compatibility
    "communication": {
        "nccl": {"compatible": True, "notes": "✓ NCCL on SM89 → 但PCIe是瓶颈"},
        "nvlink": {"compatible": False, "notes": "✗ RTX 4090无NVLink → PCIe Gen4 x16"},
        "nvls_sharp": {"compatible": False, "notes": "✗ NVLS SHARP需要SM90+"},
        "fp8_allgather": {"compatible": False, "notes": "✗ NCCL FP8需要SM90"},
        "deepep": {"compatible": False, "notes": "✗ DeepEP需要SM90+NVLink+RDMA"},
        "pcie_gen4": {"compatible": True, "notes": "✓ 32GB/s双向 → 但多GPU灾难(0.46x)"},
    },

    # ★★★ LoRA Compatibility
    "lora": {
        "lora_serving": {"compatible": True, "notes": "✓ Punica SGMV on SM89"},
        "lora_prefix_caching": {"compatible": False, "notes": "✗ #44701 hash collision → LoRA ID不进hash"},
        "lora_cuda_graph": {"compatible": True, "notes": "✓ 固定地址buffer → graph safe"},
        "lora_int4_merge": {"compatible": True, "notes": "✓ merge→INT4→vLLM→4,791 tok/s"},
    },

    # ★★★ Training Compatibility
    "training": {
        "bf16_training": {"compatible": True, "notes": "✓ BF16唯一正确训练精度"},
        "fp8_training": {"compatible": False, "notes": "✗ FP8训练需要SM90+TE"},
        "lora_grpo": {"compatible": True, "notes": "✓ rLLM Tinker + GRPO + LoRA → 17GB/24GB"},
        "single_gpu": {"compatible": True, "notes": "✓ 单GPU最优 → PCIe多GPU灾难"},
        "cpu_optimizer": {"compatible": True, "notes": "✓ CPU_Adam → ZeRO-2 offload"},
    },

    # ★★★ Inference Deployment
    "inference": {
        "int4_vllm": {"compatible": True, "notes": "✓ INT4+INT8KV → 4,791 tok/s"},
        "eagle_speculative": {"compatible": True, "notes": "✓ EAGLE+INT4 → 9,088 tok/s"},
        "mtp_speculative": {"compatible": True, "notes": "✓ MTP → shared layer → 最轻量"},
        "fp8_inference": {"compatible": False, "notes": "✗ FP8推理需要SM90 → #44879"},
        "pd_disaggregation": {"compatible": True, "notes": "✓ 单GPU PD → 但PCIe跨GPU不可"},
    },
}

# ============================================================
# vLLM Issues Related to SM89
# ============================================================
SM89_VLLM_ISSUES = [
    {"number": 44879, "title": "FP8 KV crash on SM89 (L4/RTX4090)", "severity": "★★★★★", "status": "open", "our_draft": "vllm-45038-sm89-fp8-comment-draft.md"},
    {"number": 45038, "title": "Guard FP8 KV override on SM90+", "severity": "★★★★★", "status": "open", "our_draft": "vllm-45038-sm89-fp8-comment-draft.md"},
    {"number": 44701, "title": "LoRA+prefix hash collision", "severity": "★★★★", "status": "open", "our_draft": "vllm-44701-comment-draft.md"},
    {"number": 43731, "title": "INT4 Triton fallback (PR, merged)", "severity": "★★★", "status": "merged", "our_analysis": "vllm-int4-triton-fallback-reading.md"},
    {"number": 45157, "title": "NIXL KV connector metrics (PR, closed)", "severity": "★★", "status": "closed", "our_pr": "vllm-pr-45157-resubmission-draft.md"},
    {"number": 45494, "title": "NIXL KV stats docs (PR, open)", "severity": "★★", "status": "open", "our_draft": "vllm-pr-45157-resubmission-draft.md"},
]


def print_matrix():
    """Print SM89 compatibility matrix"""
    print("=" * 70)
    print("SM89 (RTX 4090 / L4) vLLM Compatibility Matrix")
    print("=" * 70)
    print(f"Compute Capability: {SM89_COMPAT['compute_capability']}")
    print(f"GPU Names: {', '.join(SM89_COMPAT['gpu_names'])}")
    print(f"Memory: {SM89_COMPAT['memory_gb']}GB")
    print(f"Architecture: {SM89_COMPAT['architecture']}")

    categories = [
        ("Attention Backends", SM89_COMPAT["attention_backends"]),
        ("Quantization", SM89_COMPAT["quantization"]),
        ("CUDA Graph", SM89_COMPAT["cuda_graph"]),
        ("Communication", SM89_COMPAT["communication"]),
        ("LoRA", SM89_COMPAT["lora"]),
        ("Training", SM89_COMPAT["training"]),
        ("Inference", SM89_COMPAT["inference"]),
    ]

    for cat_name, cat_dict in categories:
        print(f"\n--- {cat_name} ---")
        for feature, info in cat_dict.items():
            symbol = "✓" if info["compatible"] else "✗"
            print(f"  {symbol} {feature}: {info['notes']}")

    # Summary
    total = sum(len(c) for _, c in categories)
    compatible = sum(1 for _, c in categories for f, i in c.items() if i["compatible"])
    not_compatible = total - compatible
    print(f"\n--- Summary ---")
    print(f"  ✓ Compatible: {compatible} / {total}")
    print(f"  ✗ Not compatible: {not_compatible} / {total}")
    print(f"  ★★★ SM89可行路径: BF16训练 + INT4推理 + FlashInfer + CUDA graph")


def print_detail():
    """Print detailed SM89 compatibility analysis"""
    print_matrix()
    print("\n" + "=" * 70)
    print("★★★ SM89可行 vs 不可行 — 详细分析")
    print("=" * 70)

    feasible = [
        "BF16训练 → LoRA+GRPO → rLLM Tinker → 17GB/24GB → step ~3.5s",
        "INT4推理 → GPTQ+INT8KV → vLLM → 4,791 tok/s",
        "EAGLE speculative → INT4 → 9,088 tok/s",
        "FlashInfer attention → SM89 ✓",
        "CUDA graph → 基本graph ✓ (NVLS-free)",
        "INT4 Triton fallback → PR#43731 → non-Marlin shapes ✓",
        "Prefix caching → 同LoRA adapter时 ✓",
        "LoRA serving → Punica SGMV ✓",
    ]

    not_feasible = [
        "FP8 E5M2推理 → SM89 ✗ → #44879 crash!",
        "FP8 KV cache → SM89 ✗ → #45038 guard needed",
        "FP8训练 → SM89 ✗ → Transformer Engine需要SM90",
        "NVLS AllGather → SM89 ✗ → Hopper-only",
        "TMA load → SM89 ✗ → Hopper-only",
        "DeepEP MoE → SM89 ✗ → 需SM90+NVLink+RDMA",
        "FlashAttention-3 → SM89 ✗ → Hopper-only",
        "Multi-GPU PCIe → 0.46x scaling → 灾难!",
        "LoRA+prefix跨adapter → #44701 hash collision!",
    ]

    print("\n★★★ 可行路径 (8个):")
    for item in feasible:
        print(f"  ✓ {item}")

    print("\n✗✗✗ 不可行路径 (9个):")
    for item in not_feasible:
        print(f"  ✗ {item}")


def print_issues():
    """Print vLLM SM89-related issues"""
    print("=" * 70)
    print("vLLM SM89-Related Issues — Contribution Opportunities")
    print("=" * 70)

    for issue in SM89_VLLM_ISSUES:
        status_sym = "✓" if issue["status"] == "merged" else "◉" if issue["status"] == "open" else "✗"
        draft = issue.get("our_draft", issue.get("our_pr", issue.get("our_analysis", "")))
        print(f"\n  {status_sym} #{issue['number']} [{issue['status']}] {issue['title']}")
        print(f"    Severity: {issue['severity']}")
        if draft:
            print(f"    Our draft: notebook/projects/{draft}")
        print(f"    URL: https://github.com/vllm-project/vllm/issues/{issue['number']}")

    print("\n★★★ 立即可执行的贡献:")
    print("  1. Comment on #45038 → SM89 expertise → our draft ready")
    print("  2. Comment on #44701 → LoRA+prefix fix → our draft ready")
    print("  3. Comment on #45494 → NIXL docstring → our draft ready")
    print("  4. Test #43731 Triton fallback on RTX 4090 → when GPU online")


def main():
    parser = argparse.ArgumentParser(description="SM89 Compatibility Checker")
    parser.add_argument("--mode", choices=["matrix", "detail", "issues"], default="matrix")
    args = parser.parse_args()

    if args.mode == "matrix":
        print_matrix()
    elif args.mode == "detail":
        print_detail()
    elif args.mode == "issues":
        print_issues()


if __name__ == "__main__":
    main()
