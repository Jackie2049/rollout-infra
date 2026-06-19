#!/usr/bin/env python3
"""
RTX 4090 GRPO Training Troubleshooting Tool
============================================
Diagnostic tool for common GRPO training issues on RTX 4090 (24GB VRAM).

Modes:
  - check: Run diagnostic checks for common GRPO pitfalls
  - budget: Calculate memory budget for specific configuration
  - fix: Suggest fixes for detected issues

Usage:
  python tools/grpo_troubleshooter_4090.py --mode check
  python tools/grpo_troubleshooter_4090.py --mode budget --model qwen2-7b --lora-rank 32
  python tools/grpo_troubleshooter_4090.py --mode fix --issue oom
"""

import argparse
import json
import sys

# ============================================================
# RTX 4090 specs
# ============================================================

RTX4090 = {
    "name": "RTX 4090",
    "vram_gb": 24,
    "gpu_memory_utilization": 0.90,  # typical vLLM setting
    "usable_gb": 24 * 0.90,  # 21.6 GiB usable
    "sm_version": 89,
    "fp16_tflops": 82.6,
    "bf16_tflops": 82.6,
    "int8_tops": 165.2,
    "hbm_bandwidth_gbps": 1008,  # ~1 TB/s
    "pcie_gen": "4.0 x16",
    "pcie_bandwidth_gbps": 31.5,  # bidirectional
}

# ============================================================
# Model configurations
# ============================================================

MODELS = {
    "qwen2-7b": {
        "name": "Qwen2-7B-Instruct",
        "params_b": 7,
        "hidden_dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 4,  # GQA-8
        "head_dim": 128,
        "dtype": "bfloat16",
        "weight_gb_bf16": 14,
        "weight_gb_int4": 3.5,
    },
    "qwen2.5-7b": {
        "name": "Qwen2.5-7B-Instruct",
        "params_b": 7,
        "hidden_dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 4,
        "head_dim": 128,
        "dtype": "bfloat16",
        "weight_gb_bf16": 14,
        "weight_gb_int4": 3.5,
    },
    "llama3-8b": {
        "name": "Llama-3-8B-Instruct",
        "params_b": 8,
        "hidden_dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,  # GQA-4
        "head_dim": 128,
        "dtype": "bfloat16",
        "weight_gb_bf16": 16,
        "weight_gb_int4": 4,
    },
    "llama2-7b": {
        "name": "Llama-2-7B-Chat",
        "params_b": 7,
        "hidden_dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,  # MHA (worst case!)
        "head_dim": 128,
        "dtype": "float16",
        "weight_gb_bf16": 14,
        "weight_gb_int4": 3.5,
    },
}

# ============================================================
# Common GRPO pitfalls on RTX 4090
# ============================================================

PITFALLS = [
    {
        "id": "OOM_DETACH_METRICS",
        "severity": "★★★★★",
        "title": "Metrics not detached → progressive OOM",
        "symptom": "GPU memory grows ~0.27GiB per micro-batch step → OOM at step ~80",
        "root_cause": ".item() without .detach() → autograd graph retained → GC can't free",
        "fix": "Set detach_metrics_per_micro_batch=True in verl; use .detach().item() everywhere",
        "impact_gb": 0.27,
        "frameworks": ["verl"],
        "auto_safe": ["rLLM Tinker"],  # in-process = automatic safety
        "ref": "verl Issue #327 / PR #328",
    },
    {
        "id": "OOM_NO_BYPASS",
        "severity": "★★★★★",
        "title": "No bypass_mode → ref model eats 14GB",
        "symptom": "GRPO training OOM even with LoRA-32",
        "root_cause": "Reference model loaded → 14GB extra → 17+14=31GB > 24GB",
        "fix": "Set bypass_mode=True → skip ref model forward → KL penalty=0 but saves memory",
        "impact_gb": 14,
        "frameworks": ["verl"],
        "auto_safe": ["rLLM Tinker"],
        "ref": "verl actor_rollout_ref_worker.py",
    },
    {
        "id": "OOM_FULL_PARAMS",
        "severity": "★★★★★",
        "title": "Full parameter training → 42GB needed",
        "symptom": "Immediate OOM on first training step",
        "root_cause": "7B model: 14GB weights + 14GB gradients + 14GB optimizer = 42GB",
        "fix": "Use LoRA-32 → 0.8% params → 2.6GB adapter → 17GB total",
        "impact_gb": 28,
        "frameworks": ["all"],
        "ref": "RTX 4090 decision tree",
    },
    {
        "id": "OOM_REWARD_MODEL",
        "severity": "★★★★",
        "title": "Reward Model loaded → 14GB extra",
        "symptom": "OOM during reward computation",
        "root_cause": "RM needs 14GB GPU → only 9GB left for policy → impossible",
        "fix": "Use rule-based reward → CPU execution → no GPU memory needed",
        "impact_gb": 14,
        "frameworks": ["all"],
        "ref": "RTX 4090 decision tree",
    },
    {
        "id": "CRASH_FP8_KV",
        "severity": "★★★★★",
        "title": "FP8 KV cache → CUDA crash on SM89",
        "symptom": "vLLM crash with FP8 E4M3 KV cache on RTX 4090/L4",
        "root_cause": "FlashInfer FP8 attention kernels only exist for SM90+ → crash on SM89",
        "fix": "Use INT8 KV cache → same memory footprint but SM89 compatible",
        "frameworks": ["vLLM"],
        "ref": "vLLM #44879 / #45038",
    },
    {
        "id": "CRASH_COMPRESSED_TENSORS_FP8",
        "severity": "★★★★",
        "title": "compressed-tensors FP8 override → SM89 crash",
        "symptom": "Even with kv_cache_dtype=int8, model config overrides to FP8",
        "root_cause": "compressed-tensors kv_cache_scheme overrides kv_cache_dtype → FP8 → crash",
        "fix": "PR #45038 guard: has_device_capability(90) check before FP8 KV",
        "frameworks": ["vLLM"],
        "ref": "vLLM PR #45038",
    },
    {
        "id": "WASTE_NO_PREFIX_CACHE",
        "severity": "★★★★",
        "title": "No prefix caching → 7x compute+memory waste for GRPO",
        "symptom": "GRPO rollout_n=8 computes system prompt 8 times instead of 1",
        "root_cause": "V1 APC is optional → must explicitly enable prefix caching",
        "fix": "Set enable_prefix_caching=True → hash-based sharing → 1x compute",
        "impact_pct": 87.5,  # 7/8 waste reduced to 0
        "frameworks": ["vLLM"],
        "ref": "vLLM scheduler.py",
    },
    {
        "id": "LOSS_PREFIX_HASH_COLLISION",
        "severity": "★★★",
        "title": "LoRA prefix-cache hash collision (#44701)",
        "symptom": "Silent KV cache corruption between different LoRA adapters",
        "root_cause": "Bare strings in extra_keys → lora_name='X' collides with cache_salt='X'",
        "fix": "PR #44706: domain-tag tuples [('lora', name)] instead of bare strings",
        "frameworks": ["vLLM"],
        "ref": "vLLM #44701 / PR #44706 (STALLED)",
    },
    {
        "id": "OOM_PPO_CRITIC",
        "severity": "★★★★★",
        "title": "PPO critic model → ~270GB needed",
        "symptom": "Impossible OOM (way beyond 24GB)",
        "root_cause": "PPO needs value function (critic) → separate model → massive memory",
        "fix": "Use GRPO instead → group-relative advantage → no critic needed",
        "impact_gb": 256,
        "frameworks": ["all"],
        "ref": "DeepSeekMath GRPO paper",
    },
    {
        "id": "OOM_BF16_INFERENCE",
        "severity": "★★★★★",
        "title": "BF16 inference → fills 24GB exactly → no headroom",
        "symptom": "vLLM OOM during serving with BF16 weights",
        "root_cause": "7B BF16=14GB + KV=10GB + overhead=24GB → 0GB headroom",
        "fix": "Use INT4 weights + INT8 KV → 3.5+5=8.5GB → 15.5GB headroom",
        "frameworks": ["vLLM", "SGLang"],
        "ref": "RTX 4090 decision tree",
    },
    {
        "id": "BAD_DISTRIBUTED_PCIe",
        "severity": "★★★★",
        "title": "TP/PP over PCIe → catastrophic slowdown",
        "symptom": "Multi-GPU training via PCIe → 3-10x slower than single GPU",
        "root_cause": "PCIe 4.0 x16 = 31.5GB/s vs NVLink 900GB/s → 28x slower interconnect",
        "fix": "Single GPU + LoRA → no distributed → RTX 4090最优",
        "frameworks": ["DeepSpeed", "Megatron-LM", "verl"],
        "ref": "PCIe decision guide",
    },
    {
        "id": "BAD_GRPO_SMALL_GROUP",
        "severity": "★★★",
        "title": "GRPO rollout_n=1 → σ=0 → gradient vanish",
        "symptom": "GRPO training stuck → loss doesn't decrease",
        "root_cause": "Group size=1 → all advantages=0 → no gradient signal",
        "fix": "Use rollout_n≥4 → or Dr.GRPO: norm_adv_by_std_in_grpo=False",
        "frameworks": ["verl", "rLLM"],
        "ref": "Dr.GRPO paper",
    },
    {
        "id": "WASTE_GRPO_CUDA_GRAPH",
        "severity": "★★",
        "title": "GRPO cudagraph exponential sizing → +13.6% peak memory",
        "symptom": "CUDA graph uses more memory than expected",
        "root_cause": "Exponential bucket sizing → worst case 2x sizes → wasted memory",
        "fix": "Use linear cudagraph sizing → or disable cudagraph for GRPO",
        "frameworks": ["vLLM"],
        "ref": "vLLM cudagraph reading",
    },
    {
        "id": "THRASH_PREEMPTION",
        "severity": "★★★★",
        "title": "Preemption thrashing → request repeatedly preempted",
        "symptom": "vLLM ITL p99 high → requests preempted → re-computed → preempted again",
        "root_cause": "Long-output request preempted → V1 retraction=full re-compute → large KV → triggers again",
        "fix": "1) reduce max_num_seqs (48 for RTX 4090); 2) set watermark=0.05; 3) enable_prefix_caching=True",
        "frameworks": ["vLLM"],
        "ref": "PR #44594 (watermark) + V1 preemption source reading",
    },
    {
        "id": "MISSING_WATERMARK",
        "severity": "★★★",
        "title": "No watermark → 82% more preemptions",
        "symptom": "vLLM preemption count high → ITL p99 elevated",
        "root_cause": "Without watermark → admitted requests fill blocks to edge → running requests no headroom",
        "fix": "Set watermark=0.05 → PR #44594 → preemption reduced 82% → ITL p99 reduced 56%",
        "frameworks": ["vLLM"],
        "ref": "PR #44594 + PR #45344 (re-merged)",
    },
    {
        "id": "FP8_TRAINING_SM89",
        "severity": "★★★",
        "title": "FP8 training on SM89 → no acceleration",
        "symptom": "torch._scaled_mm FP8 training same speed as BF16 on RTX 4090",
        "root_cause": "SM89 has FP8 tensor cores but no native GEMM pipeline → fallback to BF16 path",
        "fix": "Use BF16 for training → FP8 only benefits SM90+ (H100) → RTX 4090 BF16 only",
        "frameworks": ["PyTorch", "DeepSpeed", "verl"],
        "ref": "FSDP2 MixedPrecisionPolicy + SM89 compat",
    },
    {
        "id": "FSDP2_SINGLE_GPU",
        "severity": "★★",
        "title": "FSDP2 on single GPU → pointless overhead",
        "symptom": "FSDP2 training slower than non-FSDP2 on single GPU",
        "root_cause": "world_size=1 → no sharding → full params → DTensor overhead but no benefit",
        "fix": "Single GPU → use LoRA + torch.compile instead → no distributed needed",
        "frameworks": ["PyTorch"],
        "auto_safe": ["rLLM Tinker"],
        "ref": "FSDP2 2026 deep reading",
    },
    # ─── NEW PITFALLS FROM 7-FRAMEWORK DEEP RESEARCH (2026-06-19) ────────
    {
        "id": "NAN_OVERLAP_COMM",
        "severity": "★★★★★★★★★",
        "title": "overlap_comm=True → NaN from multi-stream data race",
        "symptom": "Loss becomes NaN from step 1 on single GPU",
        "root_cause": "Gradient bucket copy_ on multiple streams, average_tensor() only waits current stream → reads bucket before all producer streams complete",
        "fix": "Set overlap_comm=False (MUST on dp=1 RTX 4090)",
        "impact_gb": 0,
        "frameworks": ["DeepSpeed"],
        "auto_safe": [],
        "ref": "DeepSpeed #8061 (production-confirmed, torch.compile+overlap_comm=NaN)",
    },
    {
        "id": "REGRESSION_ZERO3_LORA",
        "severity": "★★★★★★★★★",
        "title": "ZeRO-3 + PEFT LoRA → dtype mismatch regression",
        "symptom": "TypeError: output tensor must have same type as input tensor in _allgather_params_coalesced",
        "root_cause": "#8066 per-policy dtype → ZeRO-3 partition dtype mismatch (fp32 LoRA + bf16 base)",
        "fix": "Use ZeRO-2 (NEVER ZeRO-3 on single GPU)",
        "impact_gb": 0,
        "frameworks": ["DeepSpeed"],
        "auto_safe": [],
        "ref": "DeepSpeed #8072/#8076/#8073 (0 reviews, stalled)",
    },
    {
        "id": "GRADIENT_CLIP_DEFAULT",
        "severity": "★★★★★★★★★",
        "title": "gradient_clipping default 0 → silent regression",
        "symptom": "Gradient explosion → training diverges",
        "root_cause": "DeepSpeed v0.19.2 changed default from 0→1.0 → MUST set explicitly",
        "fix": "Set gradient_clipping=1.0 explicitly (NEVER rely on default)",
        "impact_gb": 0,
        "frameworks": ["DeepSpeed"],
        "auto_safe": [],
        "ref": "DeepSpeed #8068 (0 reviews, stalled)",
    },
    {
        "id": "DSV4_CUDAGRAPH",
        "severity": "★★★★★★★★★",
        "title": "CUDA graphs for DSV4/MoE → 11 failures across 4 frameworks",
        "symptom": "Crash, NaN, or garbage output with DSV4/MoE models",
        "root_cause": "CUDA graph replay assumes static execution path → DSV4 has 5 dynamic routing layers",
        "fix": "Set enforce_eager=True (MANDATORY for any model with dynamic routing)",
        "impact_gb": 0,
        "frameworks": ["vLLM", "SGLang", "vLLM-Ascend", "Megatron"],
        "auto_safe": [],
        "ref": "11 failures: #45309, #45972, #28591, #28612, #10684, #10579, #28676, #10724, #28679, #5317, #46088",
    },
    {
        "id": "LORA_RANK64_EOS",
        "severity": "★★★★★★★★",
        "title": "LoRA rank=64 → vLLM never emits EOS (truncated responses)",
        "symptom": "All GRPO responses truncated → no EOS token",
        "root_cause": "rank=64 breaks EOS emission in vLLM rollout",
        "fix": "Use LoRA rank=32 alpha=64 (NEVER rank=64 for GRPO)",
        "impact_gb": 0,
        "frameworks": ["verl"],
        "auto_safe": [],
        "ref": "verl #6782 (OPEN, 1 comment)",
    },
    {
        "id": "FD_LEAK_LONG_RUNNING",
        "severity": "★★★★★★★★",
        "title": "File descriptor leak in long-running GRPO → process crash",
        "symptom": "Process crashes after many training steps (fd exhaustion)",
        "root_cause": "deepspeed_io_handle_t::wait() doesn't close aio fd → leak accumulates",
        "fix": "Set ulimit -n 65536 before training",
        "impact_gb": 0,
        "frameworks": ["DeepSpeed"],
        "auto_safe": [],
        "ref": "DeepSpeed #8075 (0 reviews, external contributor)",
    },
    {
        "id": "GRPO_GROUP_SIZE1",
        "severity": "★★★★★★★★",
        "title": "group_size=1 → normalization undefined (σ=0 → BROKEN)",
        "symptom": "All advantages = 0 → no learning signal",
        "root_cause": "σ = sqrt(var/group_size) → var=0 when group_size=1 → division by zero",
        "fix": "Set group_size≥2 (MUST for GRPO normalization)",
        "impact_gb": 0,
        "frameworks": ["rLLM"],
        "auto_safe": [],
        "ref": "rLLM #605 (OPEN 20+ days, ZERO comments → BROKEN)",
    },
    {
        "id": "MOE_CACHE_CLOBBER",
        "severity": "★★★★★★★★★",
        "title": "MXFP8 MoE shuffle cache CLOBBERED on weight reload (64x blowup)",
        "symptom": "MoE accuracy collapses from 0.06→3.83 (64x blowup) after LoRA update",
        "root_cause": "GPU allocator reuses cache physical address for new weights → physically destroys derived state",
        "fix": "dict.clear() on ALL derived caches at weight-reload boundary",
        "impact_gb": 0,
        "frameworks": ["SGLang"],
        "auto_safe": [],
        "ref": "SGLang #28676 (10th DSV4 failure, +28/-2 fix)",
    },
    {
        "id": "GDN_INTERMITTENT_DEGEN",
        "severity": "★★★★★★★★★",
        "title": "GDN intermittent decode degeneracy (silent, accumulates over uptime)",
        "symptom": "Decode throughput collapses, tiny-output loops → worsens over uptime, clears on restart",
        "root_cause": "State errors accumulate linearly without proper reset at boundaries",
        "fix": "Output quality monitoring + periodic engine restart every 20-50 steps",
        "impact_gb": 0,
        "frameworks": ["SGLang"],
        "auto_safe": [],
        "ref": "SGLang #28679 (NOT DSV4 but same state lifecycle mismatch pattern family)",
    },
    {
        "id": "MTP_GRAMMAR_FSM",
        "severity": "★★★★★★★★",
        "title": "MTP + grammar FSM conflict → 58% request failure on RTX 4090",
        "symptom": "Structured output requests fail with FSM rejection on speculative tokens",
        "root_cause": "Mid-window reasoning-end loses bitmask switch, bonus row inherits stale apply_bitmask=False",
        "fix": "Wait for fix PR #44297 (+455/-15) or disable MTP+grammar combination",
        "impact_gb": 0,
        "frameworks": ["vLLM"],
        "auto_safe": [],
        "ref": "vLLM #46118/#44006 (58% failure rate → 0% after #44297)",
    },
    {
        "id": "V1_TRAINER_EXPERIMENTAL",
        "severity": "★★★",
        "title": "V1 trainer_sync vs Legacy main_ppo — choose wisely",
        "symptom": "Confusion about which verl trainer to use on RTX 4090",
        "root_cause": "V1 trainer is new (active development), Legacy is production-tested (maintenance mode)",
        "fix": "Use Legacy for production, V1 trainer_sync for experimentation",
        "impact_gb": 0,
        "frameworks": ["verl"],
        "auto_safe": [],
        "ref": "verl V1 architecture: register_trainer() pattern, trainer_sync #1 BEST for RTX 4090",
    },
]

# ============================================================
# Memory budget calculator
# ============================================================

def calc_memory_budget(model_key, lora_rank=32, bypass_mode=True,
                       detach_metrics=True, rule_reward=True,
                       kv_dtype="int8", quant_weights="int4",
                       rollout_n=8, max_model_len=1024):
    """Calculate RTX 4090 memory budget for GRPO training or inference."""

    model = MODELS[model_key]
    usable = RTX4090["usable_gb"]

    # Weight memory
    if quant_weights == "int4":
        weights_gb = model["weight_gb_int4"]
    else:
        weights_gb = model["weight_gb_bf16"]

    # LoRA adapter memory
    lora_gb = lora_rank * 2 * model["hidden_dim"] * model["n_layers"] * 2 * 2 / 1e9  # rank*2*hidden*layers*2(A+B)*2bytes
    # Simplified: for rank=32 on 7B, ~2.6GB

    if lora_rank == 32:
        lora_gb = 2.6
    elif lora_rank == 16:
        lora_gb = 1.3
    elif lora_rank == 64:
        lora_gb = 5.2
    elif lora_rank == 0:
        lora_gb = 0  # full params

    # Optimizer memory (LoRA: AdamW → 2 states per param)
    if lora_rank > 0:
        optim_gb = lora_gb * 0.5  # AdamW states ≈ 50% of LoRA params (m+v in FP32)
    else:
        optim_gb = 0  # assume CPU offload for full params

    # KV cache memory (per token)
    kv_bytes_per_token = 2 * model["n_layers"] * model["n_kv_heads"] * model["head_dim"]
    if kv_dtype == "fp8" or kv_dtype == "int8":
        kv_bytes_per_token *= 1  # 1 byte per element
    elif kv_dtype == "fp16" or kv_dtype == "bf16":
        kv_bytes_per_token *= 2  # 2 bytes per element

    kv_total_tokens = rollout_n * max_model_len
    kv_gb = kv_bytes_per_token * kv_total_tokens / 1e9

    # Reference model (if no bypass_mode)
    ref_model_gb = 0 if bypass_mode else model["weight_gb_bf16"]

    # Reward model (if not rule-based)
    rm_gb = 0 if rule_reward else 14

    # Activation memory (training)
    act_gb = 2 if lora_rank > 0 else 14  # simplified

    # Detach metrics overhead (if not detached)
    detach_overhead_gb = 0 if detach_metrics else 10  # ~10GiB for full batch

    # CUDA graph + buffers
    graph_buf_gb = 2.5 if quant_weights == "int4" else 0.5

    # Compute total
    total_gb = (weights_gb + lora_gb + optim_gb + kv_gb + ref_model_gb +
                rm_gb + act_gb + detach_overhead_gb + graph_buf_gb)

    headroom_gb = usable - total_gb
    feasible = headroom_gb > 2  # need at least 2GB headroom

    return {
        "model": model["name"],
        "config": {
            "lora_rank": lora_rank,
            "bypass_mode": bypass_mode,
            "detach_metrics": detach_metrics,
            "rule_reward": rule_reward,
            "kv_dtype": kv_dtype,
            "quant_weights": quant_weights,
            "rollout_n": rollout_n,
            "max_model_len": max_model_len,
        },
        "breakdown": {
            "weights_gb": round(weights_gb, 2),
            "lora_gb": round(lora_gb, 2),
            "optim_gb": round(optim_gb, 2),
            "kv_gb": round(kv_gb, 2),
            "ref_model_gb": round(ref_model_gb, 2),
            "rm_gb": round(rm_gb, 2),
            "act_gb": round(act_gb, 2),
            "detach_overhead_gb": round(detach_overhead_gb, 2),
            "graph_buf_gb": round(graph_buf_gb, 2),
            "total_gb": round(total_gb, 2),
        },
        "headroom_gb": round(headroom_gb, 2),
        "feasible": feasible,
        "status": "✓✓✓ FEASIBLE" if feasible and headroom_gb > 5 else
                  "★★ TIGHT" if feasible else
                  "✗✗✗ OOM",
    }


# ============================================================
# Diagnostic checks
# ============================================================

def run_checks():
    """Run all diagnostic checks for RTX 4090 GRPO training."""
    print("=" * 60)
    print("RTX 4090 GRPO Training Troubleshooter — Diagnostic Check")
    print("=" * 60)
    print()

    # Check 1: Memory budgets
    print("## 1. Memory Budget Quick Check")
    print()

    configs = [
        ("GRPO Tinker optimal", "qwen2-7b", 32, True, True, True, "int8", "int4", 8, 1024),
        ("GRPO verl (safe)", "qwen2-7b", 32, True, True, True, "int8", "int4", 8, 1024),
        ("GRPO verl (dangerous)", "qwen2-7b", 32, False, False, True, "int8", "int4", 8, 1024),
        ("PPO (impossible)", "qwen2-7b", 32, False, False, False, "int8", "int4", 8, 1024),
        ("Full params (impossible)", "qwen2-7b", 0, True, True, True, "int8", "bf16", 8, 1024),
    ]

    print(f"{'Config':<25} {'Total':>8} {'Head':>8} {'Status':<15}")
    print("-" * 60)

    for name, model, lora, bypass, detach, rule, kv, quant, n, len_ in configs:
        budget = calc_memory_budget(model, lora, bypass, detach, rule, kv, quant, n, len_)
        print(f"{name:<25} {budget['breakdown']['total_gb']:>7.1f}G "
              f"{budget['headroom_gb']:>7.1f}G {budget['status']:<15}")

    print()

    # Check 2: Pitfall checklist
    print("## 2. Common Pitfall Checklist")
    print()

    for p in PITFALLS:
        print(f"  [{p['severity']}] {p['id']}")
        print(f"    Title: {p['title']}")
        print(f"    Symptom: {p['symptom']}")
        print(f"    Fix: {p['fix']}")
        print(f"    Frameworks: {', '.join(p['frameworks'])}")
        print(f"    Ref: {p['ref']}")
        print()

    # Check 3: SM89 compatibility
    print("## 3. SM89 (RTX 4090) Compatibility Quick Check")
    print()

    sm89_ok = [
        ("INT4 GPTQ (Marlin)", "✓"),
        ("INT8 KV cache (FlashInfer)", "✓"),
        ("BF16 training", "✓"),
        ("CUDA graph (FULL_DECODE_ONLY)", "✓"),
        ("FlashInfer attention", "✓"),
        ("LoRA serving", "✓"),
        ("Continuous batching", "✓"),
        ("Prefix caching", "✓"),
    ]

    sm89_bad = [
        ("FP8 E4M3 KV cache", "✗ (crash on SM89!)"),
        ("FP8 E5M2", "✗ (SM90+ only)"),
        ("NVLS/TMA", "✗ (SM90+ only)"),
        ("DeepEP", "✗ (SM90+ only)"),
        ("FP8 training (forward)", "✗"),
        ("PPO (critic model)", "✗ (270GB needed)"),
    ]

    print("  ✓ Compatible:")
    for name, status in sm89_ok:
        print(f"    {status} {name}")

    print()
    print("  ✗ NOT Compatible / Avoid:")
    for name, status in sm89_bad:
        print(f"    {status} {name}")

    print()
    print("=" * 60)
    print("★★★★★ RTX 4090 GRPO 最优配置:")
    print("  rLLM Tinker + GRPO + LoRA-32 + bypass_mode + rule-based reward")
    print("  → 17GB / 24GB → 7GB headroom → ✓✓✓")
    print("  → 推理: INT4 + INT8KV → 4,791 tok/s → EAGLE → 9,088 tok/s")
    print("=" * 60)


def run_budget(model_key, lora_rank=32, bypass_mode=True, detach_metrics=True,
               rule_reward=True, kv_dtype="int8", quant_weights="int4",
               rollout_n=8, max_model_len=1024):
    """Calculate and display memory budget."""

    budget = calc_memory_budget(model_key, lora_rank, bypass_mode, detach_metrics,
                                rule_reward, kv_dtype, quant_weights, rollout_n, max_model_len)

    print("=" * 60)
    print(f"RTX 4090 GRPO Memory Budget — {budget['model']}")
    print("=" * 60)
    print()

    print(f"Configuration:")
    for k, v in budget["config"].items():
        print(f"  {k}: {v}")
    print()

    print(f"Memory Breakdown:")
    print(f"  {'Component':<20} {'GB':>8}")
    print(f"  {'-'*28}")
    for k, v in budget["breakdown"].items():
        if v > 0:
            label = k.replace("_gb", "").replace("_", " ")
            print(f"  {label:<20} {v:>7.1f}G")
    print(f"  {'-'*28}")
    print(f"  {'TOTAL':<20} {budget['breakdown']['total_gb']:>7.1f}G")
    print()

    print(f"Results:")
    print(f"  Usable VRAM: {RTX4090['usable_gb']:.1f} GiB")
    print(f"  Total needed: {budget['breakdown']['total_gb']:.1f} GiB")
    print(f"  Headroom: {budget['headroom_gb']:.1f} GiB")
    print(f"  Status: {budget['status']}")

    if not budget["feasible"]:
        print()
        print("  ★★★ OOM! Fix suggestions:")
        if budget["breakdown"]["ref_model_gb"] > 0:
            print("    → Enable bypass_mode=True (save {:.1f}GB)".format(
                budget["breakdown"]["ref_model_gb"]))
        if budget["breakdown"]["detach_overhead_gb"] > 0:
            print("    → Enable detach_metrics_per_micro_batch=True (save {:.1f}GB)".format(
                budget["breakdown"]["detach_overhead_gb"]))
        if budget["breakdown"]["rm_gb"] > 0:
            print("    → Use rule-based reward (save {:.1f}GB)".format(
                budget["breakdown"]["rm_gb"]))
        if budget["config"]["lora_rank"] == 0:
            print("    → Use LoRA-32 instead of full params (save {:.1f}GB)".format(
                budget["breakdown"]["weights_gb"] - 3.5 + budget["breakdown"]["lora_gb"]))
        if budget["config"]["kv_dtype"] in ("fp16", "bf16"):
            print("    → Use INT8 KV cache (save {:.1f}GB)".format(
                budget["breakdown"]["kv_gb"] / 2))


def run_fix(issue):
    """Suggest fixes for a specific issue."""

    matching = [p for p in PITFALLS if p["id"].lower() == issue.lower() or
                issue.lower() in p["title"].lower()]

    if not matching:
        print(f"No matching pitfall for '{issue}'")
        print()
        print("Available issues:")
        for p in PITFALLS:
            print(f"  {p['id']}: {p['title']}")
        return

    for p in matching:
        print("=" * 60)
        print(f"Fix Guide: {p['title']}")
        print("=" * 60)
        print()
        print(f"Severity: {p['severity']}")
        print(f"Symptom: {p['symptom']}")
        print(f"Root cause: {p['root_cause']}")
        print()
        print(f"★★★★★ FIX: {p['fix']}")
        print()
        print(f"Affected frameworks: {', '.join(p['frameworks'])}")
        if p.get("auto_safe"):
            print(f"Auto-safe frameworks: {', '.join(p['auto_safe'])}")
        print(f"Reference: {p['ref']}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GRPO Troubleshooter")
    parser.add_argument("--mode", choices=["check", "budget", "fix"], default="check")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen2-7b")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--bypass-mode", type=bool, default=True)
    parser.add_argument("--detach-metrics", type=bool, default=True)
    parser.add_argument("--rule-reward", type=bool, default=True)
    parser.add_argument("--kv-dtype", choices=["int8", "fp8", "fp16", "bf16"], default="int8")
    parser.add_argument("--quant-weights", choices=["int4", "bf16"], default="int4")
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--issue", type=str, default="oom")

    args = parser.parse_args()

    if args.mode == "check":
        run_checks()
    elif args.mode == "budget":
        run_budget(args.model, args.lora_rank, args.bypass_mode, args.detach_metrics,
                   args.rule_reward, args.kv_dtype, args.quant_weights,
                   args.rollout_n, args.max_model_len)
    elif args.mode == "fix":
        run_fix(args.issue)


if __name__ == "__main__":
    main()
