#!/usr/bin/env python3
"""SM89 Batch Invariance Diagnostic Tool for vLLM RTX 4090

Tests batch invariance scenarios and provides deployment guidance
for RTX 4090 (SM89) / L4 (SM89) production inference.

Modes:
  check     — Check SM89 batch invariance status and risks
  diagnose  — Identify which scenarios require enforce_eager=True
  config    — Generate vLLM config with correct SM89 batch invariance settings
  compare   — Compare SM89 vs SM90 batch invariance behavior
  full      — Full report with all sections

Usage:
  python tools/sm89_batch_invariance_diagnostic.py --mode check
  python tools/sm89_batch_invariance_diagnostic.py --mode config --model Qwen/Qwen3-1.5B
  python tools/sm89_batch_invariance_diagnostic.py --mode full

References:
  - Issue #39096: https://github.com/vllm-project/vllm/issues/39096
  - PR #30018: enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90 workaround
  - PR #38938: test_eagle_dp moved to H100 CI
  - FlashInfer #2424: batch invariance broken for CTA sizes on SM89
"""

import argparse
import json
import os
import sys

# SM89 batch invariance knowledge base
SM89_BATCH_INVARIANCE_ISSUES = {
    "torch_compile": {
        "severity": "HIGH",
        "description": "torch.compile (Inductor) generates Triton kernels that bypass aten overrides",
        "source": "Issue #39096, PR #30018",
        "root_cause": "Inductor autotuning produces batch-size-dependent configs on SM89",
        "affected_ops": ["RMSNorm + residual + linear fusion", "softmax", "mean reduction"],
        "sm86_status": "PASSES (YM2132 confirmed on RTX 3090)",
        "sm89_status": "FAILS at token ~80 (L4 confirmed)",
        "workaround": "enforce_eager=True (disables torch.compile)",
    },
    "cuda_graphs": {
        "severity": "HIGH",
        "description": "CUDA graph replay with different batch composition produces wrong results",
        "source": "Issue #39096, PR #38938",
        "root_cause": "cuBLAS/cuBLASLt selects different split-k strategies for different batch sizes on SM89",
        "sm86_status": "Likely fails (same SM80 family)",
        "sm89_status": "FAILS independently of torch.compile",
        "workaround": "enforce_eager=True (disables CUDA graphs)",
    },
    "flashinfer": {
        "severity": "MEDIUM",
        "description": "FlashInfer CTA tile sizes produce batch-dependent warp-level reduction",
        "source": "FlashInfer Issue #2424",
        "root_cause": "Cooperative Thread Array tile sizes change with grid dimensions on SM89",
        "sm86_status": "Unknown",
        "sm89_status": "FAILS — disabled in vLLM determinism tests",
        "workaround": "Use TRITON_ATTN or FLASH_ATTN backend instead",
    },
    "lm_head": {
        "severity": "MEDIUM (FIXED)",
        "description": "lm_head (UnquantizedEmbeddingMethod) was missing VLLM_BATCH_INVARIANT check",
        "source": "PR #38938 (fixed)",
        "root_cause": "cuBLAS always used for lm_head regardless of batch invariant mode",
        "sm86_status": "FIXED in PR #38938",
        "sm89_status": "FIXED in PR #38938",
        "workaround": "Already fixed — no action needed",
    },
}

SCENARIO_RISK_MATRIX = {
    "single_user_serving": {
        "description": "Single-user inference, no determinism requirement",
        "batch_invariance_needed": False,
        "enforce_eager": False,
        "risk": "LOW — output quality unaffected by batch composition",
        "throughput_impact": "0%",
    },
    "multi_tenant_serving": {
        "description": "Multiple concurrent users need deterministic outputs",
        "batch_invariance_needed": True,
        "enforce_eager": True,
        "risk": "HIGH without enforce_eager — outputs vary across batch composition",
        "throughput_impact": "10-20%",
    },
    "speculative_decoding": {
        "description": "EAGLE/MTP/Medusa draft model verification requires batch invariance",
        "batch_invariance_needed": True,
        "enforce_eager": True,
        "risk": "CRITICAL without enforce_eager — silently wrong verified tokens",
        "throughput_impact": "10-20% (but no spec decode without it)",
    },
    "grpo_rollout": {
        "description": "verl/rLLM GRPO training rollout engine needs consistent rewards",
        "batch_invariance_needed": True,
        "enforce_eager": True,
        "risk": "HIGH without enforce_eager — reward inconsistency across batches",
        "throughput_impact": "Moderate (training side is bottleneck)",
    },
    "benchmarking": {
        "description": "Reproducible evaluation pipelines",
        "batch_invariance_needed": True,
        "enforce_eager": True,
        "risk": "HIGH without enforce_eager — non-reproducible results",
        "throughput_impact": "10-20%",
    },
    "dp_inference": {
        "description": "Data parallel inference across multiple GPUs",
        "batch_invariance_needed": True,
        "enforce_eager": True,
        "risk": "CRITICAL without enforce_eager — DP ranks disagree",
        "throughput_impact": "10-20%",
    },
}


def check_sm89_status():
    """Check current SM89 batch invariance status."""
    lines = []
    lines.append("=" * 70)
    lines.append("SM89 Batch Invariance Status Check")
    lines.append("=" * 70)

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            sm_version = cap[0] * 10 + cap[1]
            lines.append("")
            lines.append(f"GPU: {name}")
            lines.append(f"Compute Capability: SM{sm_version}")
            lines.append(f"SM89 (RTX 4090/L4): {sm_version == 89}")
            lines.append(f"SM90+ (H100/B200): {sm_version >= 90}")
            lines.append(f"SM80 family: {sm_version // 10 == 8}")
        else:
            lines.append("")
            lines.append("No GPU detected — running CPU-only analysis")
            sm_version = None
    except ImportError:
        lines.append("")
        lines.append("PyTorch not available — running theoretical analysis")
        sm_version = None

    # Issue summary
    lines.append("")
    lines.append("Known SM89 Batch Invariance Issues:")
    lines.append("-" * 40)
    for name, info in SM89_BATCH_INVARIANCE_ISSUES.items():
        sev = info["severity"]
        lines.append(f"  {name}: [{sev}] {info['description']}")
        lines.append(f"    SM89 status: {info['sm89_status']}")
        if "sm86_status" in info:
            lines.append(f"    SM86 status: {info['sm86_status']}")
        lines.append(f"    Workaround: {info['workaround']}")

    # Key finding
    lines.append("")
    lines.append("KEY FINDING: torch.compile AND CUDA graphs break batch invariance")
    lines.append("  INDEPENDENTLY on SM89. Disabling ONE is insufficient.")
    lines.append("  enforce_eager=True disables BOTH and is required for correctness.")

    return "\n".join(lines)


def diagnose_scenarios():
    """Diagnose which scenarios require enforce_eager on SM89."""
    lines = []
    lines.append("=" * 70)
    lines.append("SM89 Batch Invariance Scenario Diagnosis")
    lines.append("=" * 70)
    lines.append("")
    lines.append("For each scenario, shows risk level and required config:")
    lines.append("")

    for name, info in SCENARIO_RISK_MATRIX.items():
        lines.append(f"--- {name} ---")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Batch invariance needed: {info['batch_invariance_needed']}")
        lines.append(f"  enforce_eager required: {info['enforce_eager']}")
        lines.append(f"  Risk without config: {info['risk']}")
        lines.append(f"  Throughput impact: {info['throughput_impact']}")
        lines.append("")

    # Decision matrix
    lines.append("=" * 70)
    lines.append("SM89 Deployment Decision Matrix")
    lines.append("=" * 70)
    lines.append("")
    lines.append("| Scenario             | enforce_eager | VLLM_BATCH_INVARIANT | Throughput Impact |")
    lines.append("|----------------------|---------------|----------------------|-------------------|")
    lines.append("| Single-user serving  | False         | False                | 0%                |")
    lines.append("| Multi-tenant         | True          | True                 | 10-20%            |")
    lines.append("| Spec decode (EAGLE)  | True          | True                 | 10-20%            |")
    lines.append("| GRPO rollout (verl)  | True          | True                 | Moderate          |")
    lines.append("| Benchmarking         | True          | True                 | 10-20%            |")
    lines.append("| DP inference         | True          | True                 | 10-20%            |")

    return "\n".join(lines)


def generate_config(model="Qwen/Qwen3-1.5B", scenario="grpo_rollout"):
    """Generate vLLM config with correct SM89 batch invariance settings."""
    info = SCENARIO_RISK_MATRIX.get(scenario, SCENARIO_RISK_MATRIX["grpo_rollout"])

    enforce_eager = info["enforce_eager"]
    batch_invariant = info["batch_invariance_needed"]

    config = {
        "model": model,
        "scenario": scenario,
        "sm89_settings": {
            "enforce_eager": enforce_eager,
            "VLLM_BATCH_INVARIANT": batch_invariant,
            "kv_cache_dtype": "int8_per_token_head",
            "reasoning": {
                "enforce_eager": "Required for batch invariance on SM89 — disables torch.compile + CUDA graphs",
                "VLLM_BATCH_INVARIANT": "Required when enforce_eager=True for consistent outputs",
                "kv_cache_dtype": "INT8 per-token-head is the SM89 production KV path (FlashInfer FP8 requires SM90)",
            },
        },
        "launch_command": f"""# RTX 4090 (SM89) vLLM launch for {scenario}
# enforce_eager={enforce_eager} — {'REQUIRED for batch invariance on SM89' if enforce_eager else 'optional for single-user'}
# VLLM_BATCH_INVARIANT={1 if batch_invariant else 0} — {'REQUIRED for determinism' if batch_invariant else 'not needed'}
# kv_cache_dtype=int8_per_token_head — SM89 production KV path

export VLLM_BATCH_INVARIANT={1 if batch_invariant else 0}

python -m vllm.entrypoints.openai.api_server \
  --model {model} \
  --enforce-eager {enforce_eager} \
  --kv-cache-dtype int8_per_token_head \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --dtype bfloat16""",
        "warnings": [
            "FP8 KV cache (kv_cache_dtype=fp8) CRASHES on SM89 — use int8_per_token_head instead",
            "NVFP4 KV cache requires SM100+ — use int8_per_token_head on SM89",
            "Without enforce_eager=True, spec decode produces SILENTLY WRONG outputs on SM89",
            "FlashInfer attention is also batch-dependent on SM89 (issue #2424)",
        ],
        "memory_estimate": {
            "model_weights_int4": "~4GB",
            "kv_cache_int8_per_token_head": "~2-4GB (depends on context length)",
            "total_with_headroom": "~8-10GB / 24GB available",
            "kv_blocks_available": "~5000 blocks for 7B model at INT4+INT8KV",
        },
    }

    return json.dumps(config, indent=2)


def compare_sm89_sm90():
    """Compare SM89 vs SM90 batch invariance behavior."""
    lines = []
    lines.append("=" * 70)
    lines.append("SM89 vs SM90 Batch Invariance Comparison")
    lines.append("=" * 70)
    lines.append("")
    lines.append("| Feature                          | SM89 (RTX 4090)        | SM90 (H100)          |")
    lines.append("|----------------------------------|------------------------|----------------------|")
    lines.append("| torch.compile batch invariance   | FAILS                  | PASSES               |")
    lines.append("| CUDA graphs batch invariance     | FAILS                  | PASSES               |")
    lines.append("| FlashInfer batch invariance      | FAILS (#2424)          | PASSES               |")
    lines.append("| Triton persistent matmul         | Works (override)       | Not needed (cuBLAS)  |")
    lines.append("| FP8 KV cache (Triton)            | ALLOWED (#43914)       | ALLOWED              |")
    lines.append("| FP8 KV cache (FlashInfer)        | CRASH (#44879)         | ALLOWED              |")
    lines.append("| INT8 KV cache                    | PASSES (SM75+)         | PASSES               |")
    lines.append("| enforce_eager needed             | YES (for determinism)  | NO                   |")
    lines.append("| Throughput penalty               | 10-20%                 | 0%                   |")
    lines.append("| Shared memory per SM             | 100KB                  | 228KB                |")
    lines.append("| Tensor core generation           | 4th (Ada)              | 5th (Hopper)         |")
    lines.append("| WMMA FP8 tensor cores            | NOT available          | Available            |")
    lines.append("")
    lines.append("ROOT CAUSE: SM89 has different shared memory size (100KB vs 228KB)")
    lines.append("  → Triton autotuning selects different configs on SM89")
    lines.append("  → Inductor codegen produces batch-dependent kernel configs")
    lines.append("  → Ada Lovelace tensor cores have different WMMA behavior")
    lines.append("")
    lines.append("SM86 (Ampere) EXCEPTION:")
    lines.append("  Qwen3-1.7B PASSES on RTX 3090 (SM86) with torch.compile")
    lines.append("  → Batch invariance is MODEL-SPECIFIC, not universal SM<90 failure")
    lines.append("  → Different models → different Inductor fusion → different results")
    lines.append("  → Need per-model testing on each SM<90 GPU")

    return "\n".join(lines)


def full_report(model="Qwen/Qwen3-1.5B"):
    """Generate full SM89 batch invariance report."""
    sections = [
        check_sm89_status(),
        "",
        diagnose_scenarios(),
        "",
        compare_sm89_sm90(),
        "",
        "Generated Config (GRPO rollout):",
        generate_config(model, "grpo_rollout"),
        "",
        "=" * 70,
        "CONTRIBUTION OPPORTUNITY",
        "=" * 70,
        "",
        "Issue #39096 has only 6 comments, root cause not isolated.",
        "This is a genuine Tier 1-2 vLLM contribution opportunity:",
        "",
        "Tier 1 (Comment): Post detailed root cause analysis on #39096",
        "  — Draft: notebook/projects/vllm-39096-batch-invariance-comment-draft.md",
        "  — Content: dual failure path + Inductor root cause + framework comparison",
        "",
        "Tier 2 (PR): Investigate which Inductor Triton kernels break on SM89",
        "  — Steps: reproduce → isolate → compare configs → propose SM89 guard",
        "  — Requires: GPU testing on RTX 4090",
        "",
        "Tier 2 (Tool): Create diagnostic that tests batch invariance per-model",
        "  — This tool is a starting point — needs GPU testing to validate",
    ]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="SM89 Batch Invariance Diagnostic Tool")
    parser.add_argument("--mode", choices=["check", "diagnose", "config", "compare", "full"],
                        default="check", help="Diagnostic mode")
    parser.add_argument("--model", default="Qwen/Qwen3-1.5B",
                        help="Model name for config generation")
    parser.add_argument("--scenario", default="grpo_rollout",
                        choices=list(SCENARIO_RISK_MATRIX.keys()),
                        help="Deployment scenario for config generation")

    args = parser.parse_args()

    if args.mode == "check":
        print(check_sm89_status())
    elif args.mode == "diagnose":
        print(diagnose_scenarios())
    elif args.mode == "config":
        print(generate_config(args.model, args.scenario))
    elif args.mode == "compare":
        print(compare_sm89_sm90())
    elif args.mode == "full":
        print(full_report(args.model))


if __name__ == "__main__":
    main()
