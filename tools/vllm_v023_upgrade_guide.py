#!/usr/bin/env python3
"""vLLM v0.23.0 RTX 4090 Upgrade & Migration Guide

Generates a personalized upgrade guide based on current vLLM version
and RTX 4090 workload type, with SM89-specific warnings.

Modes:
  - guide: Full upgrade guide with SM89 warnings
  - checklist: Pre-upgrade checklist (what to check before upgrading)
  - config: Generate v0.23.0-specific vLLM config for your workload
  - migration: Step-by-step migration from v0.22.x to v0.23.0
  - all: All of the above combined

Based on: notebook/projects/vllm-v0.23-rtx4090-impact-reading.md
"""

import argparse
import json
import sys
from pathlib import Path

# v0.23.0 SM89 impact data
POSITIVE_CHANGES = {
    "INT4 Triton fallback": {
        "pr": "#43731",
        "impact": "★★★★★",
        "description": "W4A16 models with non-Marlin-aligned intermediate sizes now work on SM89",
        "before": "ValueError crash",
        "after": "TritonW4A16LinearKernel fallback (slower but functional)",
        "affected_models": ["DeepSeek-V2-Lite (down_proj K=704)", "MoE models with odd shapes"],
    },
    "HMA-by-default": {
        "pr": "#41847",
        "impact": "★★★★",
        "description": "Hybrid KV Cache Manager prevents startup OOM for hybrid-attention models",
        "before": "Startup OOM on 24GB VRAM for DS-V4-Flash/Mixtral/Mamba",
        "after": "HMA auto-detects hybrid attention spec, no collapse to FullAttentionSpec",
    },
    "FP8/NVFP4 fail-fast guards": {
        "prs": ["#43669", "#43914", "#40127"],
        "impact": "★★★",
        "description": "Clear ValueError instead of silent CUDA crash on SM89",
        "before": "Opaque CUDA crash",
        "after": "ValueError with clear message (e.g., 'nvfp4 requires sm_100 or sm_103')",
    },
}

UNFIXED_ISSUES = {
    "compressed-tensors FP8 KV crash": {
        "issue": "#44879",
        "pr": "#45038",
        "status": "OPEN, not merged in v0.23.0",
        "impact": "★★★★★",
        "description": "compressed-tensors models auto-override kv_cache_dtype to FP8 → CUDA crash on SM89",
        "workaround": "--kv-cache-dtype auto or avoid compressed-tensors FP8 checkpoints",
    },
    "SM<90 batch invariance bug": {
        "issue": "#39096",
        "status": "OPEN, no fix PR",
        "impact": "★★★★",
        "description": "CUDA graphs + torch.compile break batch invariance on SM<90 → spec decode incorrect",
        "workaround": "enforce_eager=True (no CUDA graph optimization for spec decode)",
    },
    "MRv2 quantization gap": {
        "status": "Roadmap work, not yet addressed",
        "impact": "★★★",
        "description": "INT4/GPTQ/AWQ models still use MRv1, miss MRv2 features",
        "workaround": "None (MRv1 works correctly for INT4 inference)",
    },
}

FP8_KV_PATHS = {
    "Triton FP8 KV": {
        "sm89_status": "ALLOWED (#43914)",
        "production_ready": "Experimental",
        "performance": "Slower than INT8 KV (Triton JIT overhead)",
        "recommendation": "★★ NOT recommended for RTX 4090 production — INT8 KV faster and safer",
    },
    "FlashInfer FP8 KV": {
        "sm89_status": "NOT supported",
        "root_cause": "flash_attn_varlen_func_fp8_sm90 only compiled for SM90+",
        "recommendation": "✗ Cannot use on SM89",
    },
    "compressed-tensors FP8 override": {
        "sm89_status": "CRASH (#44879)",
        "root_cause": "Auto-overrides kv_cache_dtype to FP8 → uses FlashInfer backend → crash",
        "recommendation": "✗✗✗ Must avoid or use --kv-cache-dtype auto workaround",
    },
    "INT8 KV": {
        "sm89_status": "✓✓✓ Production-ready",
        "performance": "Best for SM89 — FlashInfer backend, well-tested",
        "recommendation": "★★★★★ ONLY production-viable path for RTX 4090 KV cache",
    },
}

DECISION_MATRIX = {
    "INT4 GPTQ inference": {
        "v022": "✅ (Marlin only, odd shapes crash)",
        "v023": "✅✅ (Marlin + Triton fallback, more models work)",
        "upgrade": "★★★★ YES! Triton fallback fixes odd-shape crashes",
        "config_notes": "MRv1 (is_quantized=True), INT8 KV, GQA-8",
    },
    "compressed-tensors FP8": {
        "v022": "❌ crash",
        "v023": "❌ crash (PR #45038 not merged)",
        "upgrade": "★★★ Neutral — still crashes, but clear error message",
        "config_notes": "Avoid FP8 checkpoints, use INT8 KV instead",
    },
    "BF16 dense inference": {
        "v022": "✅ (MRv1)",
        "v023": "✅✅ (MRv2 auto-enabled)",
        "upgrade": "★★★★ YES! MRv2 + FlashInfer sampler + BCG",
        "config_notes": "Auto MRv2 for Llama/Mistral dense",
    },
    "EAGLE spec decode": {
        "v022": "✅ (possibly incorrect)",
        "v023": "⚠️ batch invariance bug (#39096)",
        "upgrade": "★★★ Need enforce_eager=True",
        "config_notes": "enforce_eager=True on SM89 for correctness",
    },
    "GRPO training rollout": {
        "v022": "✅",
        "v023": "✅✅ (MRv1 + INT8 KV + Triton fallback)",
        "upgrade": "★★★★ YES! INT4 Triton fallback helps MoE models",
        "config_notes": "MRv1 (INT4), INT8 KV, enable_prefix_caching=True",
    },
    "Hybrid attention models": {
        "v022": "❌ startup OOM",
        "v023": "✅ HMA-by-default prevents OOM",
        "upgrade": "★★★★★ YES! HMA fix is critical for 24GB VRAM",
        "config_notes": "HMA auto-enabled, multi-tier KV offload optional",
    },
}


def generate_guide(args):
    """Generate full upgrade guide."""
    workload = args.workload or "general"
    current_ver = args.current_version or "v0.22.1"

    lines = []
    lines.append(f"# vLLM Upgrade Guide: {current_ver} → v0.23.0 for RTX 4090")
    lines.append(f"# Workload: {workload}")
    lines.append("")

    lines.append("## 1. What's New in v0.23.0 for RTX 4090")
    lines.append("")
    for name, data in POSITIVE_CHANGES.items():
        pr_str = data.get('pr', data.get('prs', []))
        if isinstance(pr_str, list):
            pr_str = ', '.join(pr_str)
        lines.append(f"### ✓ {name} ({pr_str}) — Impact: {data['impact']}")
        lines.append(f"  Before: {data['before']}")
        lines.append(f"  After:  {data['after']}")
        lines.append(f"  {data['description']}")
        if "affected_models" in data:
            lines.append(f"  Affected models: {', '.join(data['affected_models'])}")
        lines.append("")

    lines.append("## 2. SM89 Unfixed Issues (CRITICAL)")
    lines.append("")
    for name, data in UNFIXED_ISSUES.items():
        status_str = f" ({data['status']})" if "status" in data else ""
        lines.append(f"### ✗ {name} — Impact: {data['impact']}{status_str}")
        lines.append(f"  {data['description']}")
        lines.append(f"  Workaround: {data['workaround']}")
        lines.append("")

    lines.append("## 3. SM89 FP8 KV Cache Paths — MUST DISTINGUISH!")
    lines.append("")
    for name, data in FP8_KV_PATHS.items():
        lines.append(f"  {name}: SM89={data['sm89_status']} → {data.get('recommendation', data.get('root_cause', ''))}")
    lines.append("")
    lines.append("  ★★★★★ Conclusion: INT8 KV = ONLY production-viable path for RTX 4090")

    lines.append("")
    lines.append("## 4. Decision Matrix for Your Workload")
    lines.append("")

    if workload in DECISION_MATRIX:
        data = DECISION_MATRIX[workload]
        lines.append(f"  Workload: {workload}")
        lines.append(f"  v0.22:  {data['v022']}")
        lines.append(f"  v0.23:  {data['v023']}")
        lines.append(f"  Upgrade: {data['upgrade']}")
        lines.append(f"  Config:  {data['config_notes']}")
    else:
        lines.append("  Workload-specific decisions:")
        for name, data in DECISION_MATRIX.items():
            lines.append(f"    {name}: {data['upgrade']}")

    lines.append("")
    lines.append("## 5. Recommended v0.23.0 Config for RTX 4090")
    lines.append("")
    lines.append("  # INT4 GPTQ inference (most common RTX 4090 workload)")
    lines.append("  vllm serve model-int4-gptq \\")
    lines.append("    --kv-cache-dtype int8 \\")
    lines.append("    --enable-prefix-caching=True \\")
    lines.append("    --max-num-seqs 48 \\")
    lines.append("    --gpu-memory-utilization 0.90")
    lines.append("")
    lines.append("  # BF16 dense inference (MRv2 auto-enabled)")
    lines.append("  vllm serve llama-3-8b \\")
    lines.append("    --kv-cache-dtype int8 \\")
    lines.append("    --enable-prefix-caching=True")
    lines.append("")
    lines.append("  # EAGLE spec decode (MUST enforce_eager on SM89)")
    lines.append("  vllm serve model-int4-gptq \\")
    lines.append("    --kv-cache-dtype int8 \\")
    lines.append("    --enforce-eager=True \\     # ★★★ CRITICAL for SM89 spec decode!")
    lines.append("    --speculative-model eagle-model")
    lines.append("")
    lines.append("  # Multi-tier KV offloading (long context)")
    lines.append("  vllm serve model \\")
    lines.append("    --kv-transfer-config '{")
    lines.append("      \"kv_connector\": \"OffloadingConnector\",")
    lines.append("      \"kv_role\": \"kv_both\",")
    lines.append("      \"kv_connector_extra_config\": {")
    lines.append("        \"spec_name\": \"TieringOffloadingSpec\",")
    lines.append("        \"cpu_bytes_to_use\": \"5GB\",")
    lines.append("        \"secondary_tiers\": [{\"type\": \"fs\", \"path\": \"/mnt/nvme/kv_cache\"}]")
    lines.append("      }")
    lines.append("    }'")
    lines.append("")
    lines.append("## 6. Upgrade Checklist")
    lines.append("")
    lines.append("  Before upgrading:")
    lines.append("  1. Backup current vLLM config and benchmark results")
    lines.append("  2. Check if you use compressed-tensors FP8 models → MUST add --kv-cache-dtype auto")
    lines.append("  3. Check if you use EAGLE/MTP spec decode → MUST add enforce_eager=True on SM89")
    lines.append("  4. Verify GPU is SM89 (nvidia-smi → compute capability = 8.9)")
    lines.append("  5. Record current throughput baseline for comparison")
    lines.append("")
    lines.append("  After upgrading:")
    lines.append("  6. Verify MRv2 activation: check logs for 'Using V2 Model Runner'")
    lines.append("  7. Verify INT4 models still use MRv1: check for 'MRv2 does not yet support quantized'")
    lines.append("  8. Test INT4 Triton fallback: load a MoE model with non-aligned intermediate_size")
    lines.append("  9. Test HMA: load Mixtral/Mamba hybrid → should not OOM on startup")
    lines.append("  10. Benchmark throughput: compare with v0.22 baseline")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    return text


def generate_checklist(args):
    """Generate pre-upgrade checklist."""
    lines = []
    lines.append("# vLLM v0.23.0 RTX 4090 Pre-Upgrade Checklist")
    lines.append("")

    checks = [
        ("GPU verification", "nvidia-smi → compute capability = 8.9 → SM89 confirmed", "CRITICAL"),
        ("FP8 models check", "ls *.safetensors → look for compressed-tensors FP8 configs", "CRITICAL"),
        ("Spec decode check", "grep 'speculative' in current config → if present → enforce_eager=True needed", "HIGH"),
        ("INT4 model inventory", "List all INT4/GPTQ/AWQ models → verify MRv1 still used", "MEDIUM"),
        ("Baseline benchmark", "Record current throughput/latency → for post-upgrade comparison", "HIGH"),
        ("Config backup", "cp current_vllm_config.yaml v0.22_config_backup.yaml", "MEDIUM"),
        ("LoRA adapter check", "Verify LoRA adapters → MRv2 LoRA+CUDA graph partially unsupported", "LOW"),
        ("KV cache current", "Check current kv-cache-dtype → if auto → OK; if fp8 → MUST change to int8", "CRITICAL"),
        ("Prefix caching", "Check enable_prefix_caching → MUST be True for GRPO (7x savings)", "HIGH"),
        ("Hybrid model test", "If using Mixtral/Mamba → v0.23.0 HMA fix = critical upgrade", "HIGH"),
    ]

    for i, (name, check, priority) in enumerate(checks, 1):
        lines.append(f"  [{i}] {name} (Priority: {priority})")
        lines.append(f"      Check: {check}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    return text


def generate_config(args):
    """Generate v0.23.0-specific config."""
    workload = args.workload or "int4_inference"

    configs = {
        "int4_inference": {
            "name": "INT4 GPTQ Inference (RTX 4090 most common)",
            "command": "vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "flags": [
                "--kv-cache-dtype int8",
                "--enable-prefix-caching=True",
                "--max-num-seqs 48",
                "--gpu-memory-utilization 0.90",
                "# Note: MRv1 auto-selected (is_quantized=True)",
            ],
            "sm89_warnings": [
                "✗ Do NOT use --kv-cache-dtype fp8 (compressed-tensors override crash)",
                "✗ Do NOT use --kv-cache-dtype nvfp4 (SM89 ValueError)",
            ],
        },
        "bf16_dense": {
            "name": "BF16 Dense Inference (MRv2 auto-enabled)",
            "command": "vllm serve meta-llama/Llama-3.1-8B-Instruct",
            "flags": [
                "--kv-cache-dtype int8",
                "--enable-prefix-caching=True",
                "# MRv2 auto-enabled (dense BF16 Llama)",
                "# FlashInfer sampler + BCG available in MRv2",
            ],
            "sm89_warnings": [
                "✗ Do NOT use --kv-cache-dtype fp8",
                "✓ MRv2 works on SM89 (Triton-dependent, SM89 Triton OK)",
            ],
        },
        "eagle_spec_decode": {
            "name": "EAGLE Speculative Decoding (SM89 CORRECTNESS FIX)",
            "command": "vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "flags": [
                "--kv-cache-dtype int8",
                "--enforce-eager=True",  # ★★★ CRITICAL for SM89!
                "--speculative-model yibinwenlei/EAGLE-Qwen2.5-7B-Instruct",
                "--num-speculative-tokens 5",
                "--enable-prefix-caching=True",
                "# enforce_eager=True = NO CUDA graph = slower but CORRECT on SM89",
                "# Batch invariance bug (#39096): CUDA graphs break spec decode on SM<90",
            ],
            "sm89_warnings": [
                "★★★★★ MUST use enforce_eager=True on SM89 for spec decode correctness",
                "Without enforce_eager: spec decode produces INCORRECT outputs on SM89",
                "Performance impact: ~10-15% slower without CUDA graphs",
            ],
        },
        "grpo_rollout": {
            "name": "GRPO Training Rollout (verl/rLLM)",
            "command": "# vLLM as rollout engine for verl HYBRID mode",
            "flags": [
                "--kv-cache-dtype int8",
                "--enable-prefix-caching=True",
                "--max-num-seqs 48",
                "--gpu-memory-utilization 0.90",
                "# MRv1 (INT4) + INT8 KV + prefix caching = GRPO optimal",
                "# enable_prefix_caching → 7x compute savings for rollout_n=8",
            ],
            "sm89_warnings": [
                "✗ compressed-tensors FP8 → crash → use INT8 KV",
                "✗ enforce_eager=True needed if using spec decode in rollout",
                "✓ INT8 KV + prefix caching = production-ready for GRPO on RTX 4090",
            ],
        },
        "long_context": {
            "name": "Long Context with Multi-Tier KV Offloading",
            "command": "vllm serve model",
            "flags": [
                "--kv-transfer-config '{\"kv_connector\": \"OffloadingConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"spec_name\": \"TieringOffloadingSpec\", \"cpu_bytes_to_use\": \"5GB\", \"secondary_tiers\": [{\"type\": \"fs\", \"path\": \"/mnt/nvme/kv_cache\"}]}}'",
                "--enable-prefix-caching=True",
                "# 24GB → CPU offload 5GB → NVMe offload → ~32K+ context",
            ],
            "sm89_warnings": [
                "✓ FS tier uses local NVMe → available on RTX 4090",
                "✓ HMA-by-default (#41847) prevents startup OOM for hybrid models",
            ],
        },
    }

    if workload not in configs:
        print(f"Unknown workload: {workload}. Options: {', '.join(configs.keys())}")
        return ""

    config = configs[workload]
    lines = []
    lines.append(f"# vLLM v0.23.0 Config: {config['name']}")
    lines.append("")
    lines.append(f"Command:")
    cmd_line = f"  {config['command']}"
    for flag in config["flags"]:
        cmd_line += f" \\\n    {flag}"
    lines.append(cmd_line)
    lines.append("")
    lines.append("SM89 Warnings:")
    for warning in config["sm89_warnings"]:
        lines.append(f"  {warning}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    return text


def generate_migration(args):
    """Generate step-by-step migration guide."""
    lines = []
    lines.append("# vLLM Migration: v0.22.x → v0.23.0 on RTX 4090")
    lines.append("")

    steps = [
        ("1. Pre-flight checks", [
            "nvidia-smi → verify SM89 (compute capability 8.9)",
            "pip show vllm → record current version",
            "Record current benchmark: vllm benchmark --model your-model",
            "Backup config: cp config.yaml config_v022_backup.yaml",
        ]),
        ("2. Install v0.23.0", [
            "pip install vllm==0.23.0 --index-url https://mirrors.aliyun.com/pypi/simple/",
            "Or: conda install vllm=0.23.0 -c conda-forge",
            "Verify: pip show vllm → Version: 0.23.0",
        ]),
        ("3. SM89-specific config changes", [
            "★ CRITICAL: If using EAGLE/MTP spec decode → add enforce_eager=True",
            "★ CRITICAL: If using compressed-tensors FP8 → add --kv-cache-dtype auto (not fp8!)",
            "★ CRITICAL: kv-cache-dtype=int8 → RTX 4090 production path (NOT fp8)",
            "★ Recommended: enable_prefix_caching=True → 7x savings for GRPO",
            "★ Recommended: max-num-seqs=48 → preemption thrashing prevention",
        ]),
        ("4. Verify MRv2 activation (BF16 dense models)", [
            "Launch: vllm serve llama-3-8b",
            "Check logs: 'Using V2 Model Runner' → MRv2 active",
            "If MRv1: 'MRv2 does not yet support...' → quantized model → OK",
        ]),
        ("5. Test INT4 Triton fallback (if using MoE models)", [
            "Load a MoE INT4 model (e.g., Qwen2-MoE INT4)",
            "Previously: ValueError for non-Marlin shapes",
            "v0.23.0: TritonW4A16LinearKernel fallback → loads successfully",
            "Check: model loads without ValueError",
        ]),
        ("6. Test HMA (if using hybrid-attention models)", [
            "Load Mixtral or Mamba hybrid model",
            "Previously: startup OOM on 24GB VRAM",
            "v0.23.0: HMA auto-enabled → no startup OOM",
            "Check: model starts without OOM",
        ]),
        ("7. Post-upgrade benchmark", [
            "vllm benchmark --model your-model --dataset your-data",
            "Compare with v0.22 baseline",
            "Expected: INT4 throughput same (MRv1 unchanged)",
            "Expected: BF16 dense slightly better (MRv2 features)",
            "Expected: Triton fallback models slower but functional",
        ]),
        ("8. Rollback plan (if issues found)", [
            "pip install vllm==0.22.1 --index-url https://mirrors.aliyun.com/pypi/simple/",
            "Restore config: cp config_v022_backup.yaml config.yaml",
            "Known rollback triggers: FP8 crash with no workaround, spec decode incorrect",
        ]),
    ]

    for step_name, items in steps:
        lines.append(f"## {step_name}")
        for item in items:
            prefix = "  ★" if item.startswith("★") else "  -"
            lines.append(f"{prefix} {item}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="vLLM v0.23.0 RTX 4090 Upgrade Guide")
    parser.add_argument("--mode", choices=["guide", "checklist", "config", "migration", "all"],
                        default="guide", help="Output mode")
    parser.add_argument("--workload", default=None,
                        choices=["int4_inference", "bf16_dense", "eagle_spec_decode",
                                 "grpo_rollout", "long_context", "compressed_tensors_fp8",
                                 "hybrid_attention", "general"],
                        help="Target workload type")
    parser.add_argument("--current-version", default="v0.22.1", help="Current vLLM version")
    parser.add_argument("--output", default=None, help="Output file path")

    args = parser.parse_args()

    if args.mode == "guide":
        generate_guide(args)
    elif args.mode == "checklist":
        generate_checklist(args)
    elif args.mode == "config":
        generate_config(args)
    elif args.mode == "migration":
        generate_migration(args)
    elif args.mode == "all":
        generate_guide(args)
        print("\n" + "="*80 + "\n")
        generate_checklist(args)
        print("\n" + "="*80 + "\n")
        generate_config(args)
        print("\n" + "="*80 + "\n")
        generate_migration(args)


if __name__ == "__main__":
    main()
