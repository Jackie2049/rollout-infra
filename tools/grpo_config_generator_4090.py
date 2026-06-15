#!/usr/bin/env python3
"""
RTX 4090 GRPO Config Generator
================================
Generates optimal GRPO training configurations for RTX 4090 based on
latest research findings (CPPO #6731, Tinker #6717, bypass_mode, detach_metrics).

Modes:
  - generate: Generate config for a specific framework/model combination
  - compare: Compare configs across frameworks
  - recommend: Get recommendation for a specific scenario

Usage:
  python tools/grpo_config_generator_4090.py --mode generate --framework rllm --model qwen3-1.7b
  python tools/grpo_config_generator_4090.py --mode compare --model qwen3-1.7b
  python tools/grpo_config_generator_4090.py --mode recommend --scenario math-reasoning

Reference notes:
  - verl-cppo-algorithm-reading.md (CPPO position-weighted trust region)
  - verl-tinker-worker-primitives-reading.md (split training API)
  - rllm-tinker-backend-deep-reading.md (in-process, bypass default)
  - verl-grpo-detach-metrics-rtx4090-reading.md (memory budget analysis)
  - megatron-lite-reading.md (LoRA primitives, verl integration)
"""

import argparse
import json
import sys

# ============================================================
# Framework configs — RTX 4090 optimal settings
# ============================================================

FRAMEWORKS = {
    "rllm": {
        "name": "rLLM Tinker",
        "rank": 1,
        "viability": "★★★★★★ 立即可用",
        "description": "In-process, bypass default, auto LoRA, zero-copy",
        "key_advantages": [
            "In-process → no Ray overhead → zero-copy weight sync",
            "bypass_mode=True DEFAULT → no ref model → save 14GB",
            "Auto LoRA init → create_lora_training_client_async",
            "detach_metrics → NOT needed (in-process → no IPC)",
            "CPPO not yet supported → use GRPO vanilla",
        ],
        "key_limitations": [
            "Small community → less production validation",
            "No CPPO support yet → GRPO only",
            "Limited model support (Qwen3 dense + MoE)",
            "Single GPU only → no multi-GPU scaling",
        ],
        "configs": {
            "qwen3-1.7b": {
                "model": "Qwen/Qwen3-1.7B",
                "lora_rank": 32,
                "lora_target": "auto (Tinker auto-init)",
                "bypass_mode": "True (default)",
                "batch_size": 4,
                "group_size": 8,
                "max_response_length": 2048,
                "estimated_peak_memory_gb": 18,
                "headroom_gb": 6,
                "loss_mode": "grpo (no CPPO yet)",
                "command": "rllm train --config grpo_qwen3_1.7b.yaml",
            },
            "qwen3-4b": {
                "model": "Qwen/Qwen3-4B",
                "lora_rank": 16,
                "lora_target": "auto",
                "bypass_mode": "True (default)",
                "batch_size": 2,
                "group_size": 8,
                "max_response_length": 1024,
                "estimated_peak_memory_gb": 21,
                "headroom_gb": 3,
                "loss_mode": "grpo",
                "command": "rllm train --config grpo_qwen3_4b.yaml",
            },
        },
    },
    "verl": {
        "name": "verl + vLLM",
        "rank": 2,
        "viability": "★★★★ 立即可用 (需配置)",
        "description": "External engine, bypass_mode+detach_metrics, CPPO available",
        "key_advantages": [
            "Large community → production validated",
            "★★★★★★ CPPO (#6731) → better trust region than GRPO → near-zero overhead",
            "★★★★ Tinker Worker Primitives (#6717) → gradient accumulation",
            "Multi-modal → Gemma4 + Qwen-VL",
            "Multi-GPU → HYBRID/COLOCATED/STANDALONE",
        ],
        "key_limitations": [
            "Ray overhead → cross-process → detach_metrics needed",
            "MRv2 handling = ZERO → must set VLLM_USE_V2_MODEL_RUNNER=0",
            "SM89 batch invariance → enforce_eager=True for spec decode",
            "v0.8.0 pins vllm==0.20.2 (3 versions behind v0.23.0)",
        ],
        "configs": {
            "qwen3-1.7b": {
                "model": "Qwen/Qwen3-1.7B",
                "lora_rank": 32,
                "lora_target": "qkv+proj+fc1+fc2",
                "bypass_mode": "True",
                "detach_metrics": "True",
                "batch_size": 4,
                "group_size": 8,
                "max_response_length": 2048,
                "estimated_peak_memory_gb": 18,
                "headroom_gb": 6,
                "loss_mode": "★★★★★ cppo (recommended) or grpo",
                "mrv2": "VLLM_USE_V2_MODEL_RUNNER=0",
                "command": "python -m verl.trainer.main_ppo_sync --config grpo_qwen3_1.7b.yaml",
                "remax_bypass_config": {
                    "adv_estimator": "remax",
                    "bypass_mode": "True",
                    "use_kl_in_reward": "False",
                    "loss_mode": "ppo_clip",
                    "clip_ratio": "0.15",
                    "use_kl_loss": "True",
                    "kl_loss_coef": "0.05",
                },
                "remax_icepop_config": {
                    "adv_estimator": "remax",
                    "bypass_mode": "True",
                    "use_kl_in_reward": "False",
                    "loss_type": "reinforce",
                    "rollout_is": "token",
                    "rollout_is_threshold": "0.5_5.0",
                    "rollout_rs": "null",
                },
                "cppo_config": {
                    "adv_estimator": "grpo",
                    "bypass_mode": "True",
                    "loss_mode": "cppo",
                    "clip_ratio": "0.15",
                    "cppo_w_min": "0.8",
                    "cppo_delta_b": "0.02",
                    "cppo_delta_b_q": "0.9",
                    "cppo_delta_b_k": "1.0",
                },
            },
            "qwen3-4b": {
                "model": "Qwen/Qwen3-4B",
                "lora_rank": 16,
                "lora_target": "qkv+proj+fc1+fc2",
                "bypass_mode": "True",
                "detach_metrics": "True",
                "batch_size": 2,
                "group_size": 8,
                "max_response_length": 1024,
                "estimated_peak_memory_gb": 21,
                "headroom_gb": 3,
                "loss_mode": "★★★★★ cppo (recommended) or grpo",
                "mrv2": "VLLM_USE_V2_MODEL_RUNNER=0",
                "command": "python -m verl.trainer.main_ppo_sync --config grpo_qwen3_4b.yaml",
            },
        },
    },
    "verl_lite": {
        "name": "verl + Megatron Lite",
        "rank": 2.5,
        "viability": "★★★ 中期可用 (需小模型protocol)",
        "description": "Megatron Lite runtime + vLLM rollout, bitwise correctness verified",
        "key_advantages": [
            "★★★★★ Bitwise correctness verified (loss=0.0, grad=0.0)",
            "★★★★ LoRA primitives included (LinearLoRA + GroupedLinearLoRA)",
            "★★★★ -20.3% memory vs Megatron-Core",
            "KL disabled by default (similar to bypass_mode)",
            "FSDP2 optimizer offload (CPU offload_fraction=1.0)",
        ],
        "key_limitations": [
            "★★★ Only MoE models (Qwen3-MoE 30B-A3B) → too big for 24GB!",
            "★★★ No small/dense model protocol → needs community contribution",
            "★★★ Experimental → dev branch only → not production",
            "★★★★ Need small model (1-7B dense) protocol for RTX 4090",
        ],
        "configs": {
            "qwen3-moe-30b-a3b": {
                "model": "Qwen/Qwen3-30B-A3B",
                "lora_rank": 8,
                "lora_target": "qkv+proj+fc1+fc2",
                "bypass_mode": "KL disabled (equivalent)",
                "batch_size": "N/A (model too big for 24GB)",
                "estimated_peak_memory_gb": "60+ (BF16 weights)",
                "headroom_gb": "N/A → NOT FIT on RTX 4090!",
                "loss_mode": "vanilla (GRPO)",
                "note": "★★★★★ Need small model protocol → add Qwen3-1.7B dense!",
                "command": "experimental/lite/examples/verl/scripts/run_qwen3moe_gsm8k_grpo.sh",
            },
        },
    },
    "megatron": {
        "name": "Megatron-LM (traditional)",
        "rank": 3,
        "viability": "★ 不可行",
        "description": "Singleton PG bug #5203 crash, no LoRA in core",
        "key_advantages": [
            "NVIDIA production quality (multi-GPU training)",
            "DeepEP integration (SM90 only)",
        ],
        "key_limitations": [
            "★★★★★ Single-GPU CRASH (#5203) → singleton PG → TypeError!",
            "★★★★ No LoRA in core → only NeMo2/Megatron-Bridge",
            "★★★★ DeepSeek-V4-Flash = SM90 exclusive",
            "★★★★★ NOT viable for RTX 4090 → use rLLM or verl instead",
        ],
        "configs": {},
    },
}

# ============================================================
# Scenario recommendations
# ============================================================

SCENARIOS = {
    "math-reasoning": {
        "description": "Math/reasoning GRPO training (long responses 4k+ tokens)",
        "recommended_framework": "verl + CPPO",
        "reason": "CPPO position-weighted trust region benefits most from long responses",
        "config_key": "qwen3-1.7b",
        "specific_settings": {
            "max_response_length": 4096,
            "loss_mode": "★★★★★ cppo (critical for long responses!)",
            "cppo_w_min": 0.8,
            "cppo_delta_b": 0.02,
        },
        "alternative": "verl ReMax (lowest variance, needs ref model → RTX 4090 tight)",
        "remax_note": "★★★★★ ReMax canonical: adv_estimator=remax, use_kl_in_reward=True → ref model needed → ~18-20GB on RTX 4090. ReMax-bypass hybrid: undocumented → risky. CPPO+bypass still #1 for RTX 4090.",
    },
    "code-generation": {
        "description": "Code generation GRPO training (medium responses 2k tokens)",
        "recommended_framework": "rLLM Tinker",
        "reason": "Simplest setup, in-process efficiency, auto LoRA",
        "config_key": "qwen3-1.7b",
        "specific_settings": {
            "max_response_length": 2048,
            "loss_mode": "grpo",
        },
        "alternative": "verl + CPPO (better trust region, more overhead)",
    },
    "short-qa": {
        "description": "Short QA/classification GRPO (<1k tokens)",
        "recommended_framework": "rLLM Tinker",
        "reason": "Short responses → CPPO less benefit → rLLM simpler",
        "config_key": "qwen3-1.7b",
        "specific_settings": {
            "max_response_length": 512,
            "loss_mode": "grpo (CPPO less benefit for short responses)",
        },
        "alternative": "verl (larger community, more model support)",
    },
    "vlm-training": {
        "description": "Visual language model GRPO (multimodal)",
        "recommended_framework": "verl",
        "reason": "verl supports Gemma4 + Qwen-VL → most complete VLM RL",
        "config_key": "qwen3-1.7b",
        "specific_settings": {
            "loss_mode": "cppo",
            "multimodal": True,
            "note": "★★★★ rLLM Geo3K Tinker VLM (#357) also available but less mature",
        },
        "alternative": "rLLM Tinker VLM (simpler but less model support)",
    },
    "moe-training": {
        "description": "MoE model GRPO training",
        "recommended_framework": "verl + Megatron Lite (中期)",
        "reason": "Lite has GroupedLinearLoRA + verl GRPO integration",
        "config_key": "qwen3-moe-30b-a3b",
        "specific_settings": {
            "note": "★★★★★ Qwen3-30B-A3B too big for RTX 4090 → need small MoE or multi-GPU",
        },
        "alternative": "rLLM Tinker (MoE support limited) or DeepSpeed (AutoEP + singleton)",
    },
}

# ============================================================
# CPPO vs GRPO vs ReMax decision tree
# ============================================================

CPPO_DECISION = {
    "use_cppo": [
        "Long responses (>4k tokens) → CPPO prefix budget most impact",
        "Math/reasoning tasks → cascading drift risk → CPPO prevents",
        "verl framework → CPPO near-zero overhead → always safe",
        "Want provably better trust region → Theorem 1",
    ],
    "use_grpo_vanilla": [
        "Short responses (<1k) → CPPO less benefit",
        "rLLM Tinker → no CPPO support yet → GRPO only",
        "Simplest setup → one config → GRPO works well for short",
        "Already stable with GRPO → no need to change",
    ],
    "use_remax": [
        "Math/reasoning tasks → greedy baseline → lowest variance → ReMax 97 vs GRPO 89 on GSM8k!",
        "verl TransferQueue sync → ReMax merged (#6340) → canonical config available",
        "Want deterministic baseline → greedy (temperature=0) → more stable than group mean",
    ],
    "remax_rtx4090_warning": [
        "★★★★★★ ReMax canonical requires ref model (use_kl_in_reward=True) → NOT compatible with bypass_mode!",
        "★★★★★★ ReMax-bypass hybrid possible (use_kl_in_reward=False) → undocumented → untested → risky!",
        "★★★★★ ReMax canonical on RTX 4090: param_offload ref model to CPU → ~18-20GB peak → feasible but tight",
        "★★★★★★ RTX 4090推荐: GRPO+bypass_mode (#1) > ReMax-bypass hybrid (risky) > ReMax canonical (tight)",
    ],
    "key_rule": "★★★★★★ CPPO is NEVER worse than GRPO → overhead near-zero → always safe on verl. ReMax has lowest variance but needs ref model → RTX 4090 use GRPO+bypass_mode unless math quality critical enough to accept ref model overhead. IcePop provides exact IS population bounds → more precise than TIS → but requires loss_type=reinforce → less stable than PPO-clip for beginners.",
    "remax_icepop_combination": [
        "★★★★★★ ReMax + IcePop + bypass_pg = theoretically strongest (greedy baseline + exact IS + no ref model)",
        "★★★★★★ But requires loss_type=reinforce → less stable than PPO-clip → NOT recommended for beginners",
        "★★★★★★★ ReMax + bypass + PPO-clip = practical strongest (greedy baseline + trust region + simple)",
        "★★★★★★★ IcePop useful when: high IS weight variance → toxic out-of-range samples → want exact population",
        "★★★★★ IcePop NOT useful when: low IS weight variance → all samples good → IcePop zeros useful data",
    ],
}


def generate_config(framework, model):
    fw = FRAMEWORKS.get(framework)
    if not fw:
        print(f"Unknown framework: {framework}")
        print(f"Available: {', '.join(FRAMEWORKS.keys())}")
        return

    cfg = fw["configs"].get(model)
    if not cfg:
        print(f"Unknown model for {framework}: {model}")
        print(f"Available: {', '.join(fw['configs'].keys())}")
        return

    print(f"=== {fw['name']} — {model} GRPO Config ===")
    print(f"Viability: {fw['viability']}")
    print()

    for k, v in cfg.items():
        if k == "cppo_config" and isinstance(v, dict):
            print(f"\n  CPPO Config (★★★★★ recommended over GRPO):")
            for ck, cv in v.items():
                print(f"    {ck}: {cv}")
        else:
            print(f"  {k}: {v}")

    print(f"\n--- Key Advantages ---")
    for adv in fw["key_advantages"]:
        print(f"  • {adv}")
    print(f"\n--- Key Limitations ---")
    for lim in fw["key_limitations"]:
        print(f"  • {lim}")

    if framework == "verl":
        print(f"\n--- CPPO Decision ---")
        print(f"  Rule: {CPPO_DECISION['key_rule']}")
        for reason in CPPO_DECISION["use_cppo"]:
            print(f"  Use CPPO: {reason}")


def compare_configs(model):
    print(f"=== RTX 4090 GRPO Config Comparison — {model} ===")
    print()

    for fw_key, fw in FRAMEWORKS.items():
        cfg = fw["configs"].get(model)
        if not cfg:
            continue

        print(f"--- #{fw['rank']} {fw['name']} ---")
        print(f"  Viability: {fw['viability']}")
        print(f"  Peak Memory: {cfg.get('estimated_peak_memory_gb', 'N/A')} GB")
        print(f"  Headroom: {cfg.get('headroom_gb', 'N/A')} GB")
        print(f"  LoRA Rank: {cfg.get('lora_rank', 'N/A')}")
        print(f"  Bypass Mode: {cfg.get('bypass_mode', 'N/A')}")
        print(f"  Loss Mode: {cfg.get('loss_mode', 'N/A')}")
        print(f"  Batch Size: {cfg.get('batch_size', 'N/A')}")
        print(f"  Command: {cfg.get('command', 'N/A')}")
        print()

    # Summary recommendation
    print("--- Recommendation ---")
    if model in ["qwen3-1.7b", "qwen3-4b"]:
        print(f"  ★★★★★ Short/medium responses → rLLM Tinker (simpler, in-process)")
        print(f"  ★★★★★★ Long responses (math/code) → verl + CPPO (better trust region)")
        print(f"  ★★★ verl + Megatron Lite → wait for small model protocol")
    else:
        print(f"  ★★★ Check specific model compatibility with each framework")


def recommend_scenario(scenario):
    sc = SCENARIOS.get(scenario)
    if not sc:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    print(f"=== RTX 4090 GRPO Recommendation — {scenario} ===")
    print(f"Description: {sc['description']}")
    print(f"★★★★★ Recommended: {sc['recommended_framework']}")
    print(f"Reason: {sc['reason']}")
    print()

    print(f"Specific settings:")
    for k, v in sc["specific_settings"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Alternative: {sc['alternative']}")

    # Show the actual config
    fw_key = None
    for k, fw in FRAMEWORKS.items():
        if fw["name"] == sc["recommended_framework"]:
            fw_key = k
            break

    if fw_key:
        cfg = FRAMEWORKS[fw_key]["configs"].get(sc["config_key"])
        if cfg:
            print(f"\n--- Config Details ---")
            for k, v in cfg.items():
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GRPO Config Generator")
    parser.add_argument("--mode", choices=["generate", "compare", "recommend"], required=True)
    parser.add_argument("--framework", choices=list(FRAMEWORKS.keys()))
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()))

    args = parser.parse_args()

    if args.mode == "generate":
        if not args.framework:
            print("--framework required for generate mode")
            sys.exit(1)
        generate_config(args.framework, args.model)
    elif args.mode == "compare":
        compare_configs(args.model)
    elif args.mode == "recommend":
        if not args.scenario:
            print("--scenario required for recommend mode")
            sys.exit(1)
        recommend_scenario(args.scenario)


if __name__ == "__main__":
    main()
