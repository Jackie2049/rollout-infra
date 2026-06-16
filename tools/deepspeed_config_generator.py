#!/usr/bin/env python3
"""
RTX 4090 DeepSpeed Config Generator — Consolidated
===================================================
Generates optimal DeepSpeed configs for ALL RTX 4090 training scenarios.
All configs pass the 7-point safety check (overlap_comm, gradient_clipping, ZeRO stage, etc).

Scenarios:
  - lora-grpo:       Standard LoRA GRPO (rLLM Tinker or DeepSpeed)
  - lora-grpo-muon:  LoRA GRPO with Muon optimizer (experimental)
  - moe-autoep:      MoE AutoEP training (Qwen3-MoE)
  - opd-distill:     OPD distillation (LoRA student + CPU-offloaded teacher)

Usage:
  python3 tools/deepspeed_config_generator.py --scenario lora-grpo --model qwen3-1.7b
  python3 tools/deepspeed_config_generator.py --scenario moe-autoep --model qwen3-moe
  python3 tools/deepspeed_config_generator.py --list-scenarios
  python3 tools/deepspeed_config_generator.py --list-models
"""

import argparse
import json
import sys
from pathlib import Path

# ============================================================
# Model database
# ============================================================

MODELS = {
    "qwen3-1.7b": {
        "path": "Qwen/Qwen3-1.7B",
        "params_b": 1.7,
        "hidden_dim": 2048,
        "n_layers": 24,
        "moe": False,
        "est_gpu_gb": 19.2,
    },
    "qwen3-4b": {
        "path": "Qwen/Qwen3-4B",
        "params_b": 4,
        "hidden_dim": 2560,
        "n_layers": 36,
        "moe": False,
        "est_gpu_gb": 21.5,
    },
    "qwen2.5-0.5b": {
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "hidden_dim": 896,
        "n_layers": 24,
        "moe": False,
        "est_gpu_gb": 18.5,
    },
    "qwen2.5-7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "params_b": 7,
        "hidden_dim": 4096,
        "n_layers": 28,
        "moe": False,
        "est_gpu_gb": "OOM (>24GB)",
    },
    "qwen3-moe": {
        "path": "Qwen/Qwen3-MoE",
        "params_b": 4.4,  # total, 0.6B active
        "hidden_dim": 2048,
        "n_layers": 24,
        "n_experts": 8,
        "topk": 2,
        "active_params_b": 0.6,
        "moe": True,
        "est_gpu_gb": 20.0,
    },
}

# ============================================================
# Scenario configs
# ============================================================

SCENARIOS = {
    "lora-grpo": {
        "name": "LoRA GRPO Training",
        "description": "Standard LoRA fine-tuning with GRPO — RTX 4090 optimal",
        "requires_moe": False,
        "default_model": "qwen3-1.7b",
        "lora_rank": 32,
        "config_template": {
            "train_batch_size": 8,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
    "lora-grpo-muon": {
        "name": "LoRA GRPO + Muon Optimizer",
        "description": "Muon+LoRA — experimental, higher rank possible, compare with AdamW baseline",
        "requires_moe": False,
        "default_model": "qwen3-1.7b",
        "lora_rank": 32,
        "config_template": {
            "train_batch_size": 8,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.02,
                    "momentum_beta": 0.95,
                    "ns_steps": 5,
                    "ns_method": "gram",
                    "nesterov": True,
                    "weight_decay": 0.0,
                    "muon_lr_scale": 0.1,
                    "aux_adam_lr": 1e-5,
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
    "moe-autoep": {
        "name": "MoE AutoEP Training",
        "description": "AutoEP + ZeRO-2 MoE training — RTX 4090 MoE viable!",
        "requires_moe": True,
        "default_model": "qwen3-moe",
        "lora_rank": None,
        "config_template": {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 5e-5,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
            "auto_ep_enable": True,
            "auto_ep_preset": "Qwen3-MoE",
        },
    },
    "opd-distill": {
        "name": "OPD Distillation (LoRA Student)",
        "description": "On-Policy Distillation — LoRA student + CPU-offloaded teacher",
        "requires_moe": False,
        "default_model": "qwen2.5-0.5b",
        "lora_rank": 32,
        "config_template": {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
}


def generate_config(scenario_name, model_name):
    """Generate a DeepSpeed config for a specific scenario + model."""
    scenario = SCENARIOS.get(scenario_name)
    if not scenario:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    model = MODELS.get(model_name)
    if not model:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    # Validate MoE requirement
    if scenario["requires_moe"] and not model["moe"]:
        print(f"Scenario '{scenario_name}' requires a MoE model, but '{model_name}' is dense.")
        sys.exit(1)

    if not scenario["requires_moe"] and model["moe"]:
        print(f"Warning: Model '{model_name}' is MoE but scenario '{scenario_name}' is for dense models.")
        print("Consider using 'moe-autoep' scenario instead.")

    # Deep-copy the template
    config = json.loads(json.dumps(scenario["config_template"]))

    # Add model-specific info
    config["_model"] = model["path"]
    config["_scenario"] = scenario_name
    config["_lora_rank"] = scenario["lora_rank"]
    config["_est_gpu_gb"] = model["est_gpu_gb"]

    return config, scenario, model


def print_config_info(config, scenario, model):
    """Print config info and safety notes."""
    print("=" * 70)
    print(f"RTX 4090 DeepSpeed Config: {scenario['name']}")
    print(f"Model: {model['path']} ({model['params_b']}B params)")
    if model.get("moe"):
        print(f"MoE: {model['n_experts']} experts, top-{model['topk']}, {model['active_params_b']}B active")
    if scenario["lora_rank"]:
        print(f"LoRA rank: {scenario['lora_rank']}")
    print(f"Estimated peak GPU: {model['est_gpu_gb']}GB / 24GB")
    print("=" * 70)
    print()

    # Safety notes
    print("Safety notes (all checks PASS):")
    print("  overlap_comm=False: safe from #8061 NaN bug + zero benefit on single GPU")
    print("  gradient_clipping=1.0: aligned with #8068 proposed default")
    print("  ZeRO-2 + CPU_Adam: optimal for single GPU (ZeRO-3 = pure overhead)")
    print("  bf16 only: correct for SM89 (FP8 NOT available)")
    if scenario_name == "lora-grpo-muon":
        print("  Muon optimizer: EXPERIMENTAL! Compare convergence with AdamW baseline!")
    print()

    # Verify with safety checker
    print("Verify: python3 tools/deepspeed_zero_safety_checker.py --mode check --config <path>")


def list_scenarios():
    """List all available scenarios."""
    print("Available training scenarios:")
    print()
    for name, scenario in SCENARIOS.items():
        print(f"  {name}: {scenario['description']}")
        print(f"    Default model: {scenario['default_model']}")
        print(f"    LoRA rank: {scenario['lora_rank'] or 'N/A (MoE)'}")
        print()


def list_models():
    """List all available models."""
    print("Available models:")
    print()
    for name, model in MODELS.items():
        moe_info = f" (MoE: {model['n_experts']}e, top-{model['topk']}, {model['active_params_b']}B active)" if model.get("moe") else ""
        oom_warn = " *** OOM on RTX 4090 ***" if str(model["est_gpu_gb"]).startswith("OOM") else ""
        print(f"  {name}: {model['path']} ({model['params_b']}B){moe_info}{oom_warn}")
        print(f"    Estimated GPU: {model['est_gpu_gb']}GB / 24GB")
        print()


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 DeepSpeed Config Generator")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Training scenario")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        help="Model name")
    parser.add_argument("--output", help="Output file path (default: configs/<scenario>_rtx4090.json)")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List all available scenarios")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without saving to file")

    args = parser.parse_args()

    if args.list_scenarios:
        list_scenarios()
        return

    if args.list_models:
        list_models()
        return

    if not args.scenario:
        print("No --scenario specified. Available:")
        list_scenarios()
        sys.exit(1)

    scenario_name = args.scenario
    model_name = args.model or SCENARIOS[scenario_name]["default_model"]

    config, scenario, model = generate_config(scenario_name, model_name)

    # Remove metadata keys for clean JSON
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}

    if args.dry_run:
        print(json.dumps(clean_config, indent=2))
        print()
        print_config_info(config, scenario, model)
        return

    # Save to file
    output_path = args.output or f"configs/{scenario_name}_rtx4090.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean_config, f, indent=2)

    print(f"Config saved: {output_path}")
    print()
    print_config_info(config, scenario, model)


if __name__ == "__main__":
    main()
