#!/usr/bin/env python3
"""
RTX 4090 DeepSpeed ZeRO Configuration Safety Checker
=====================================================
Validates DeepSpeed ZeRO configs for known RTX 4090 pitfalls.

Checks:
  1. #8061: overlap_comm + torch.compile = NaN (CRITICAL)
  2. #8068: gradient_clipping default 0→1.0 (STABILITY)
  3. ZeRO-3 single GPU = meaningless overhead (CONFIG)
  4. offload_optimizer + ZeRO-2 optimal (CONFIG)
  5. LoRAOptimizedLinear compatibility (CONFIG)
  6. contiguous_gradients + overlap_comm interaction (CONFIG)
  7. bf16/fp16 selection (SM89)
  8. Muon optimizer compatibility (EXPERIMENTAL)

Modes:
  - check: Validate a DeepSpeed config JSON file
  - generate: Generate optimal RTX 4090 config for a training scenario
  - explain: Explain specific check in detail

Usage:
  python tools/deepspeed_zero_safety_checker.py --mode check --config configs/zero2_rtx4090.json
  python tools/deepspeed_zero_safety_checker.py --mode check --config configs/muon_lora_zero2_rtx4090.json
  python tools/deepspeed_zero_safety_checker.py --mode generate --scenario lora-grpo --model qwen3-1.7b
  python tools/deepspeed_zero_safety_checker.py --mode generate --scenario moe-autoep --model qwen3-moe-a2.5b
  python tools/deepspeed_zero_safety_checker.py --mode generate --scenario opd-distill --model qwen2.5-0.5b
  python tools/deepspeed_zero_safety_checker.py --mode explain --check overlap_comm_compile_nan
"""

import argparse
import json
import sys
from pathlib import Path

# ============================================================
# RTX 4090 specs
# ============================================================

RTX4090 = {
    "name": "RTX 4090",
    "vram_gb": 24,
    "sm_version": 89,
    "bf16_tflops": 82.6,
    "hbm_bandwidth_gbps": 1008,
}

# ============================================================
# Safety checks catalog
# ============================================================

CHECKS = {
    "overlap_comm_compile_nan": {
        "severity": "CRITICAL",
        "issue": "DeepSpeed #8061",
        "title": "overlap_comm + torch.compile causes NaN from step 1",
        "description": (
            "ZeRO-1/2 overlap_comm=True with torch.compile creates a multi-stream bug. "
            "DeepSpeed assumes single CUDA stream for gradient copy_, but compiled autograd "
            "dispatches copy_ across multiple streams. average_tensor() only waits current_stream(), "
            "not ALL producer streams → reduction reads incomplete gradient data → NaN.\n\n"
            "Root cause (stage_1_and_2.py line 1234):\n"
            "  stream.wait_stream(get_accelerator().current_stream())  # ONLY waits current!\n"
            "  # Missing: wait for ALL streams that wrote to IPG buckets\n\n"
            "This affects ALL ZeRO-1/2 configs with overlap_comm=True when using torch.compile."
        ),
        "affected_configs": ["ZeRO-1 overlap_comm=True", "ZeRO-2 overlap_comm=True"],
        "workaround": (
            "RTX 4090 workaround: Set overlap_comm=False when using torch.compile.\n"
            "This is safe on single GPU because dp_world_size=1 → reduce_scatter is identity.\n"
            "overlap_comm provides NO benefit on single GPU anyway (no cross-GPU reduction).\n\n"
            "DeepSpeed config fix:\n"
            "  \"zero_optimization\": {\n"
            "    \"overlap_comm\": false,  # MUST be false with torch.compile on single GPU\n"
            "  }\n\n"
            "Proposed upstream fix: record IPG copy streams per bucket → "
            "reduction_stream waits all recorded producer streams before average_tensor()."
        ),
        "rtx4090_impact": (
            "RTX 4090 single GPU: overlap_comm=False has ZERO throughput penalty because "
            "dp_world_size=1 means no cross-GPU communication. The only 'overlap' is "
            "between gradient computation and reduction of SAME GPU's gradients — "
            "which is meaningless on single GPU.\n\n"
            "Therefore: ALWAYS set overlap_comm=False on single GPU. It's both safer "
            "and equally performant."
        ),
    },
    "gradient_clipping_default": {
        "severity": "STABILITY",
        "issue": "DeepSpeed #8068",
        "title": "gradient_clipping default 0.0 (disabled) → training instability risk",
        "description": (
            "DeepSpeed GRADIENT_CLIPPING_DEFAULT = 0.0, meaning gradient clipping is "
            "DISABLED by default. Most RL/LLM training clips at 1.0. Omitting "
            "gradient_clipping in config → silently unclipped → potential gradient "
            "explosion, especially for GRPO training.\n\n"
            "PR #8068 proposes changing default from 0.0 to 1.0.\n"
            "Override: explicit gradient_clipping: 0.0 still disables."
        ),
        "affected_configs": ["Any config without explicit gradient_clipping"],
        "workaround": (
            "Always set gradient_clipping explicitly:\n"
            "  \"gradient_clipping\": 1.0  # CRITICAL for GRPO stability\n\n"
            "For Muon optimizer: gradient_clipping=1.0 still recommended.\n"
            "For OPD distillation: gradient_clipping=1.0 prevents student divergence."
        ),
        "rtx4090_impact": (
            "RTX 4090 GRPO: gradient_clipping=1.0 is CRITICAL. Without it, "
            "GRPO advantage computation can explode → reward hack → training collapse.\n\n"
            "This is the #1 config mistake on RTX 4090: omitting gradient_clipping "
            "and getting silently unclipped gradients."
        ),
    },
    "zero3_single_gpu": {
        "severity": "CONFIG",
        "issue": "Source-level analysis",
        "title": "ZeRO-3 on single GPU = pure overhead, no benefit",
        "description": (
            "ZeRO-3 partitions parameters across GPUs. On single GPU (dp_world_size=1):\n"
            "  - get_data_parallel_partitions() returns [full_tensor] (no sharding)\n"
            "  - all_gather_dp_groups() with dp=1 skips entirely\n"
            "  - ALL partitioning savings = ZERO on single GPU\n\n"
            "ZeRO-3 overhead on single GPU:\n"
            "  - Parameter gathering/scattering: wasted compute\n"
            "  - Extra memory for partition buffers: wasted VRAM\n"
            "  - Contiguous param allocation: constrains memory layout\n\n"
            "ZeRO-2 on single GPU:\n"
            "  - Gradient partitioning also meaningless (dp=1)\n"
            "  - BUT: CPU optimizer offload IS beneficial regardless of dp\n"
            "  - CPU_Adam (C++ SIMD AVX512) → 5-7x faster than torch.optim.Adam on CPU"
        ),
        "affected_configs": ["ZeRO-3 on single GPU"],
        "workaround": (
            "RTX 4090 single GPU: ALWAYS use ZeRO-2 (not ZeRO-3).\n\n"
            "Optimal config:\n"
            "  \"zero_optimization\": {\n"
            "    \"stage\": 2,\n"
            "    \"offload_optimizer\": {\"device\": \"cpu\", \"pin_memory\": true}\n"
            "  }\n\n"
            "With LoRA: ZeRO-0 or ZeRO-2 both viable.\n"
            "  ZeRO-2 + CPU_Adam → optimizer state on CPU → LoRA gradients on GPU\n"
            "  ZeRO-0 → simpler → but no CPU optimizer offload benefit"
        ),
        "rtx4090_impact": (
            "ZeRO-3 wastes ~2-4GB VRAM on partition overhead on single GPU.\n"
            "ZeRO-2 + CPU_Adam + offload saves ~8-10GB optimizer state to CPU.\n"
            "This difference is CRITICAL on 24GB VRAM."
        ),
    },
    "lora_optimized_linear": {
        "severity": "CONFIG",
        "issue": "DeepSpeed LoRAOptimizedLinear (merged)",
        "title": "LoRAOptimizedLinear compatibility with ZeRO configs",
        "description": (
            "LoRAOptimizedLinear uses split forward: base_weight_output + lora_output.\n"
            "Base weight frozen (ds_optim_param=True) → only LoRA params trained.\n"
            "offload_ratio (0→1) cumulatively offloads base weights to CPU.\n\n"
            "Compatible with:\n"
            "  - ZeRO-2 + CPU_Adam: optimizer only for LoRA params (60x reduction)\n"
            "  - Hybrid engine: fuse/unfuse during rollout generation\n"
            "  - OPD trainer: LoRA init B=zeros → initial output=0 → fine for OPD\n\n"
            "NOT compatible with:\n"
            "  - ZeRO-3: parameter partitioning conflicts with frozen base weights\n"
            "  - overlap_comm + torch.compile: same #8061 bug affects LoRA grads"
        ),
        "affected_configs": ["ZeRO-3 + LoRA", "overlap_comm + LoRA + torch.compile"],
        "workaround": (
            "RTX 4090 LoRA config:\n"
            "  \"zero_optimization\": {\n"
            "    \"stage\": 2,  # NOT 3!\n"
            "    \"offload_optimizer\": {\"device\": \"cpu\", \"pin_memory\": true}\n"
            "  }\n\n"
            "LoRAOptimizedLinear config:\n"
            "  offload_ratio: 0.5 → 50% base weights on CPU → ~1GB GPU savings\n"
            "  lora_r: 32 → 13.5M trainable params (2.2% of 0.5B model)\n\n"
            "Init context manager:\n"
            "  with deepspeed.linear.Init(lora_config=LoRAConfig(r=32, offload=True, offload_ratio=0.5)):\n"
            "      model = AutoModelForCausalLM.from_pretrained(...)"
        ),
        "rtx4090_impact": (
            "LoRA reduces optimizer CPU from 9.84GB to 0.16GB (60x).\n"
            "This means LoRA+ZeRO-2 runs on 16GB RAM systems (not 32GB).\n"
            "offload_ratio=0.5 saves ~1GB GPU on 0.5B model."
        ),
    },
    "contiguous_gradients_overlap": {
        "severity": "WARNING",
        "issue": "DeepSpeed stage_1_and_2.py",
        "title": "contiguous_gradients + overlap_comm gradient bucket interaction",
        "description": (
            "When overlap_comm=True and contiguous_gradients=True, gradient buckets "
            "are formed during backward pass. The copy_ operation moves gradients "
            "into contiguous IPG buffer. With torch.compile, copy_ can be issued "
            "on multiple CUDA streams.\n\n"
            "average_tensor() then reads from this buffer on reduction_stream, "
            "but only waits current_stream() — not all streams that wrote to it.\n"
            "This is the #8061 root cause: incomplete gradient data → NaN."
        ),
        "affected_configs": ["ZeRO-1/2 overlap_comm + contiguous_gradients + torch.compile"],
        "workaround": (
            "Set overlap_comm=False on single GPU. contiguous_gradients=True is fine.\n\n"
            "Recommended:\n"
            "  \"zero_optimization\": {\n"
            "    \"all_contiguous_gradients\": true,  # OK on single GPU\n"
            "    \"overlap_comm\": false  # MUST be false with torch.compile\n"
            "  }"
        ),
        "rtx4090_impact": "Same as #8061 — overlap_comm=False eliminates the risk entirely on single GPU.",
    },
    "bf16_vs_fp16": {
        "severity": "CONFIG",
        "issue": "SM89 compatibility",
        "title": "bf16 required for RTX 4090 (SM89) — FP8/FP4 NOT available",
        "description": (
            "RTX 4090 (SM89) supports:\n"
            "  - BF16: full hardware support ✓\n"
            "  - FP16: full support ✓ (but BF16 preferred for training)\n"
            "  - INT8: full support ✓ (FlashInfer INT8 KV cache)\n"
            "  - FP8 (E4M3/E5M2): NOT natively supported ✗ (SM90+ only)\n"
            "  - FP4/MXFP4: NOT supported ✗ (SM120+ only)\n\n"
            "DeepSpeed bf16 config:\n"
            "  \"bf16\": {\"enabled\": true}\n"
            "  \"fp16\": {\"enabled\": false}  # MUST be false when bf16=true\n\n"
            "IMPORTANT: Do NOT enable both fp16 and bf16 simultaneously."
        ),
        "affected_configs": ["fp16 + bf16 simultaneously", "FP8 quantization on SM89"],
        "workaround": (
            "Always use bf16 for RTX 4090 training:\n"
            "  \"bf16\": {\"enabled\": true}\n"
            "  \"fp16\": {\"enabled\": false}\n\n"
            "For inference: INT8 KV cache via FlashInfer is the best quantization path.\n"
            "FP8 Triton KV cache (#43914) works on SM89 but is NOT FlashInfer."
        ),
        "rtx4090_impact": "BF16 is the correct training dtype. FP8 is NOT available on SM89.",
    },
    "muon_optimizer": {
        "severity": "EXPERIMENTAL",
        "issue": "DeepSpeed #7953 (merged)",
        "title": "Muon optimizer experimental — monitor convergence closely",
        "description": (
            "Muon (Momentum Orthogonalized by Newton-Schulz) is merged but EXPERIMENTAL.\n\n"
            "Key properties:\n"
            "  - Operates on 2D matrices (perfect for LoRA!)\n"
            "  - Gram NS iteration: n×n → 5x cheaper than rectangular\n"
            "  - 1 optimizer buffer (momentum) vs Adam's 2 (m+v) → 50% savings\n"
            "  - Prevents LoRA rank collapse via orthogonalization\n\n"
            "RTX 4090 Muon+LoRA:\n"
            "  - LoRA rank=64: Muon fits 19.2GB, AdamW OOM at 24.4GB\n"
            "  - SingleDeviceMuonWithAuxAdam+gram recommended\n"
            "  - aux_adam_lr for 1D params (bias, norm)\n\n"
            "Risks:\n"
            "  - lr=0.02, muon_lr_scale=0.1 may need tuning\n"
            "  - No long-term convergence data on consumer GPUs\n"
            "  - Compare with Adam baseline before committing"
        ),
        "affected_configs": ["Muon optimizer configs"],
        "workaround": (
            "Start with conservative Muon config:\n"
            "  \"optimizer\": {\n"
            "    \"type\": \"Muon\",\n"
            "    \"params\": {\n"
            "      \"lr\": 0.02,\n"
            "      \"momentum_beta\": 0.95,\n"
            "      \"ns_steps\": 5,\n"
            "      \"ns_method\": \"gram\",\n"
            "      \"nesterov\": true,\n"
            "      \"weight_decay\": 0.0,\n"
            "      \"muon_lr_scale\": 0.1,\n"
            "      \"aux_adam_lr\": 1e-5\n"
            "    }\n"
            "  }\n\n"
            "ALWAYS compare with AdamW baseline. Muon is NOT guaranteed better.\n"
            "gradient_clipping=1.0 still recommended (same as AdamW)."
        ),
        "rtx4090_impact": (
            "Muon+LoRA fits 19.2GB → viable on RTX 4090.\n"
            "AdamW+LoRA rank=64 → 24.4GB → OOM.\n"
            "Muon is the ONLY path for LoRA rank≥64 on RTX 4090.\n"
            "But: EXPERIMENTAL — convergence not proven on consumer GPUs."
        ),
    },
}

# ============================================================
# Training scenarios
# ============================================================

SCENARIOS = {
    "lora-grpo": {
        "name": "LoRA GRPO Training",
        "description": "Standard LoRA fine-tuning with GRPO on RTX 4090",
        "config": {
            "train_batch_size": 8,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,  # MUST be False on single GPU
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
    "lora-grpo-muon": {
        "name": "LoRA GRPO Training with Muon Optimizer",
        "description": "Muon+LoRA — experimental, higher rank possible",
        "config": {
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
                "overlap_comm": False,  # MUST be False on single GPU
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
    "moe-autoep": {
        "name": "MoE AutoEP Training",
        "description": "AutoEP+ZeRO-2 MoE training (Qwen3-MoE)",
        "config": {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 5e-5, "weight_decay": 0.01},
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "all_contiguous_gradients": True,
                "overlap_comm": False,  # MUST be False on single GPU
                "reduce_bucket_size": 5e6,
            },
            "bf16": {"enabled": True},
            "fp16": {"enabled": False},
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    },
    "opd-distill": {
        "name": "OPD Distillation (LoRA Student)",
        "description": "On-Policy Distillation with LoRA student + CPU-offloaded teacher",
        "config": {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4, "weight_decay": 0.01},
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

# ============================================================
# Check implementation
# ============================================================


def check_config(config: dict) -> list:
    """Run all safety checks on a DeepSpeed config dict."""
    issues = []

    zero_opt = config.get("zero_optimization", {})
    stage = zero_opt.get("stage", 0)
    overlap_comm = zero_opt.get("overlap_comm", False)
    contiguous_grads = zero_opt.get("all_contiguous_gradients", zero_opt.get("contiguous_gradients", True))
    grad_clipping = config.get("gradient_clipping", 0.0)
    bf16 = config.get("bf16", {}).get("enabled", False)
    fp16 = config.get("fp16", {}).get("enabled", False)
    optimizer_type = config.get("optimizer", {}).get("type", "AdamW")
    offload_opt = zero_opt.get("offload_optimizer", {})
    offload_device = offload_opt.get("device", "none")

    # Check 1: overlap_comm + torch.compile NaN (#8061)
    if overlap_comm:
        issues.append({
            "check": "overlap_comm_compile_nan",
            "severity": "CRITICAL",
            "status": "FAIL",
            "message": (
                f"overlap_comm=True on single GPU is DANGEROUS with torch.compile (#8061).\n"
                f"  average_tensor() only waits current_stream() — not all producer streams.\n"
                f"  This causes NaN from step 1 when using torch.compile.\n"
                f"  On single GPU, overlap_comm provides NO benefit anyway (dp=1 → identity reduce).\n"
                f"  FIX: Set overlap_comm=False"
            ),
        })
    else:
        issues.append({
            "check": "overlap_comm_compile_nan",
            "severity": "CRITICAL",
            "status": "PASS",
            "message": "overlap_comm=False — safe from #8061 NaN bug",
        })

    # Check 2: gradient_clipping default (#8068)
    if grad_clipping == 0.0:
        issues.append({
            "check": "gradient_clipping_default",
            "severity": "STABILITY",
            "status": "WARNING",
            "message": (
                f"gradient_clipping=0.0 (DISABLED) — risk of gradient explosion!\n"
                f"  DeepSpeed default is 0.0 (#8068 proposes 1.0).\n"
                f"  Most RL/LLM training clips at 1.0.\n"
                f"  GRPO is especially vulnerable to unclipped gradients.\n"
                f"  FIX: Set gradient_clipping=1.0"
            ),
        })
    elif grad_clipping >= 1.0:
        issues.append({
            "check": "gradient_clipping_default",
            "severity": "STABILITY",
            "status": "PASS",
            "message": f"gradient_clipping={grad_clipping} — safe (#8068 aligned)",
        })
    else:
        issues.append({
            "check": "gradient_clipping_default",
            "severity": "STABILITY",
            "status": "WARNING",
            "message": (
                f"gradient_clipping={grad_clipping} — unusual value.\n"
                f"  Common values: 1.0 (standard), 5.0 (conservative).\n"
                f"  Verify this is intentional."
            ),
        })

    # Check 3: ZeRO-3 on single GPU
    if stage == 3:
        issues.append({
            "check": "zero3_single_gpu",
            "severity": "CONFIG",
            "status": "FAIL",
            "message": (
                f"ZeRO-3 on single GPU = pure overhead, no benefit!\n"
                f"  partition_size = full model (no sharding with dp=1)\n"
                f"  all_gather_dp_groups() skips entirely with dp=1\n"
                f"  ALL partitioning savings = ZERO on single GPU\n"
                f"  Wasted: ~2-4GB VRAM on partition overhead\n"
                f"  FIX: Set stage=2 + offload_optimizer to CPU"
            ),
        })
    elif stage == 2:
        if offload_device == "cpu":
            issues.append({
                "check": "zero3_single_gpu",
                "severity": "CONFIG",
                "status": "PASS",
                "message": "ZeRO-2 + CPU optimizer offload — optimal for single GPU RTX 4090",
            })
        else:
            issues.append({
                "check": "zero3_single_gpu",
                "severity": "CONFIG",
                "status": "WARNING",
                "message": (
                    f"ZeRO-2 without CPU optimizer offload — optimizer state stays on GPU.\n"
                    f"  For 7B model: ~9.8GB optimizer state on GPU → risky on 24GB.\n"
                    f"  CPU_Adam (C++ SIMD) is 5-7x faster than torch.optim.Adam on CPU.\n"
                    f"  RECOMMEND: offload_optimizer.device='cpu'"
                ),
            })
    elif stage == 0 or stage == 1:
        issues.append({
            "check": "zero3_single_gpu",
            "severity": "CONFIG",
            "status": "INFO",
            "message": f"ZeRO-{stage} — simpler config, no partitioning overhead. Consider ZeRO-2+CPU_Adam for larger models.",
        })

    # Check 4: bf16/fp16
    if bf16 and fp16:
        issues.append({
            "check": "bf16_vs_fp16",
            "severity": "CONFIG",
            "status": "FAIL",
            "message": "Both bf16 and fp16 enabled — NOT allowed! Choose one. RTX 4090: use bf16.",
        })
    elif bf16:
        issues.append({
            "check": "bf16_vs_fp16",
            "severity": "CONFIG",
            "status": "PASS",
            "message": "bf16 enabled — correct for RTX 4090 (SM89) training",
        })
    elif fp16:
        issues.append({
            "check": "bf16_vs_fp16",
            "severity": "CONFIG",
            "status": "WARNING",
            "message": "fp16 enabled — bf16 is preferred for RTX 4090 training (wider dynamic range)",
        })
    else:
        issues.append({
            "check": "bf16_vs_fp16",
            "severity": "CONFIG",
            "status": "WARNING",
            "message": "No mixed precision enabled — training will use fp32 (slow, high memory)",
        })

    # Check 5: Muon optimizer experimental warning
    if optimizer_type == "Muon":
        muon_params = config.get("optimizer", {}).get("params", {})
        ns_method = muon_params.get("ns_method", "gram")
        aux_lr = muon_params.get("aux_adam_lr", None)
        issues.append({
            "check": "muon_optimizer",
            "severity": "EXPERIMENTAL",
            "status": "WARNING",
            "message": (
                f"Muon optimizer is EXPERIMENTAL!\n"
                f"  ns_method={ns_method} {'(recommended)' if ns_method == 'gram' else '(rectangular — 5x more expensive)'}\n"
                f"  aux_adam_lr={aux_lr} " + ("(set for 1D params)" if aux_lr else "(NOT SET — 1D params unoptimized!)") + "\n"
                f"  ALWAYS compare convergence with AdamW baseline.\n"
                f"  Muon+LoRA = natural combo (2D matrices), but results not proven on consumer GPUs."
            ),
        })
    else:
        issues.append({
            "check": "muon_optimizer",
            "severity": "EXPERIMENTAL",
            "status": "INFO",
            "message": f"Using {optimizer_type} optimizer — stable and proven",
        })

    # Check 6: contiguous_gradients + overlap_comm interaction
    if overlap_comm and contiguous_grads:
        issues.append({
            "check": "contiguous_gradients_overlap",
            "severity": "WARNING",
            "status": "FAIL",
            "message": (
                f"overlap_comm=True + contiguous_gradients=True + torch.compile = NaN risk (#8061)\n"
                f"  copy_ on multiple streams → IPG buffer incomplete → average_tensor reads garbage.\n"
                f"  FIX: Set overlap_comm=False on single GPU."
            ),
        })
    else:
        issues.append({
            "check": "contiguous_gradients_overlap",
            "severity": "WARNING",
            "status": "PASS",
            "message": "No overlap_comm+contiguous_gradients+compile interaction risk",
        })

    return issues


def print_results(issues: list, config_path: str = None):
    """Print check results with color coding."""
    severity_colors = {
        "CRITICAL": "\033[0;31m",  # RED
        "STABILITY": "\033[1;33m",  # YELLOW
        "CONFIG": "\033[0;34m",     # BLUE
        "WARNING": "\033[1;33m",    # YELLOW
        "EXPERIMENTAL": "\033[0;35m", # MAGENTA
        "INFO": "\033[0;32m",       # GREEN
    }
    status_icons = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠", "INFO": "ℹ"}

    print("=" * 70)
    print("RTX 4090 DeepSpeed ZeRO Configuration Safety Checker")
    if config_path:
        print(f"Config: {config_path}")
    print("=" * 70)
    print()

    fails = 0
    warnings = 0
    passes = 0

    for issue in issues:
        severity = issue["severity"]
        status = issue["status"]
        color = severity_colors.get(severity, "\033[0m")
        icon = status_icons.get(status, "?")
        nc = "\033[0m"

        print(f"{color}[{severity}] {icon} {issue['check']}{nc}")
        print(f"  {issue['message']}")
        print()

        if status == "FAIL":
            fails += 1
        elif status == "WARNING":
            warnings += 1
        else:
            passes += 1

    print("=" * 70)
    print(f"Results: {passes} PASS, {warnings} WARNING, {fails} FAIL")
    print("=" * 70)

    if fails > 0:
        print()
        print("★★★★★★★★★ CRITICAL ISSUES DETECTED — FIX BEFORE TRAINING! ★★★★★★★★★")
        print("Most common RTX 4090 fix: overlap_comm=False + gradient_clipping=1.0")
        return 1
    elif warnings > 0:
        print()
        print("★★★ WARNINGS DETECTED — Review before training ★★★")
        return 0
    else:
        print()
        print("★★★★★★★★★ All checks PASS — config is safe for RTX 4090 ★★★★★★★★★")
        return 0


# ============================================================
# Generate mode
# ============================================================


def generate_config(scenario: str, model: str) -> dict:
    """Generate optimal DeepSpeed config for a training scenario."""
    scenario_data = SCENARIOS.get(scenario)
    if not scenario_data:
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    config = scenario_data["config"].copy()
    config["_scenario"] = scenario
    config["_model"] = model
    config["_rtx4090_safe"] = True
    config["_notes"] = [
        f"Scenario: {scenario_data['name']}",
        f"Model: {model}",
        "All checks should PASS — optimized for RTX 4090 single GPU",
        "overlap_comm=False: safe from #8061 NaN bug + no benefit on single GPU",
        "gradient_clipping=1.0: aligned with #8068 proposed default",
        "ZeRO-2 + CPU_Adam: optimal for single GPU (ZeRO-3 = pure overhead)",
    ]

    return config


# ============================================================
# Explain mode
# ============================================================


def explain_check(check_name: str):
    """Print detailed explanation of a specific check."""
    check = CHECKS.get(check_name)
    if not check:
        print(f"Unknown check: {check_name}")
        print(f"Available: {', '.join(CHECKS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"DeepSpeed ZeRO Safety Check: {check_name}")
    print("=" * 70)
    print()
    print(f"Severity: {check['severity']}")
    print(f"Issue: {check['issue']}")
    print(f"Title: {check['title']}")
    print()
    print("Description:")
    print(check["description"])
    print()
    print("Affected configs:")
    for ac in check["affected_configs"]:
        print(f"  - {ac}")
    print()
    print("Workaround / Fix:")
    print(check["workaround"])
    print()
    print("RTX 4090 Impact:")
    print(check["rtx4090_impact"])
    print()


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 DeepSpeed ZeRO Safety Checker")
    parser.add_argument("--mode", choices=["check", "generate", "explain"], required=True)
    parser.add_argument("--config", help="Path to DeepSpeed config JSON (for check mode)")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="Training scenario (for generate mode)")
    parser.add_argument("--model", default="qwen3-1.7b",
                        help="Model name (for generate mode)")
    parser.add_argument("--check-name",
                        choices=list(CHECKS.keys()),
                        help="Check name (for explain mode)")
    parser.add_argument("--output", help="Output file for generated config")

    args = parser.parse_args()

    if args.mode == "check":
        if not args.config:
            # Try to find any DeepSpeed config in the project
            config_files = list(Path("configs").glob("*.json"))
            if config_files:
                print("No --config specified. Found configs:")
                for cf in config_files:
                    print(f"  {cf}")
                args.config = str(config_files[0])
                print(f"Using: {args.config}")
            else:
                print("No --config specified and no configs/ directory found.")
                print("Usage: python tools/deepspeed_zero_safety_checker.py --mode check --config <path>")
                sys.exit(1)

        with open(args.config) as f:
            config = json.load(f)

        issues = check_config(config)
        rc = print_results(issues, args.config)
        sys.exit(rc)

    elif args.mode == "generate":
        if not args.scenario:
            print("No --scenario specified.")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)

        config = generate_config(args.scenario, args.model)

        # Remove metadata keys for clean JSON
        clean_config = {k: v for k, v in config.items() if not k.startswith("_")}

        output_path = args.output or f"configs/{args.scenario}_rtx4090.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(clean_config, f, indent=2)

        print(f"Generated config: {output_path}")
        print(f"Scenario: {config['_scenario']}")
        print(f"Model: {config['_model']}")
        print()
        print("Safety notes:")
        for note in config["_notes"]:
            print(f"  {note}")
        print()

        # Run check on generated config
        issues = check_config(clean_config)
        print_results(issues, output_path)

    elif args.mode == "explain":
        if not args.check_name:
            print("Available checks:")
            for name, check in CHECKS.items():
                print(f"  {name}: [{check['severity']}] {check['title']}")
            sys.exit(0)

        explain_check(args.check_name)


if __name__ == "__main__":
    main()
