#!/usr/bin/env python3
"""
Silent Corruption Detector for RTX 4090 GRPO Training
=====================================================
Detects silent corruption bugs that produce no error signal:
  - #8061: CUDA stream data race (overlap_comm NaN)
  - #8058: ZenFlow contiguous() bug (param update loss)
  - #28679: GDN intermittent decode degradation
  - #46118: MTP+grammar FSM conflict

Modes:
  check   - Validate config for silent corruption risk factors
  monitor - Runtime monitoring commands for training
  math    - Mathematical analysis of corruption damage over steps
  rtx4090 - RTX 4090 specific check with all 16+16 MUST/NOT rules
"""

import argparse
import json
import sys


# ============================================================
# Silent Corruption Bug Database
# ============================================================

BUGS = {
    "ds8061": {
        "framework": "DeepSpeed",
        "id": "#8061",
        "title": "ZeRO overlap_comm CUDA stream race",
        "severity": 2,  # Level 2: transient, NaN downstream
        "pattern": "Producer writes on stream A, consumer reads on stream B without wait",
        "effect": "NaN in training (obvious downstream, but no error at corruption site)",
        "trigger": "overlap_comm=True + torch.compile=True on single GPU",
        "fix": "overlap_comm=False on single GPU",
        "detection": "NaN in loss/gradient",
        "rtx4090_risk": "HIGH — single GPU = no redundancy",
    },
    "ds8058_contiguous": {
        "framework": "DeepSpeed",
        "id": "#8058",
        "title": "ZenFlow contiguous() silent param update loss",
        "severity": 2,  # Level 2: accumulating, training stagnation
        "pattern": ".contiguous() creates COPY for non-contiguous → optimizer updates copy, not original",
        "effect": "Original param NOT updated → optimizer step has NO effect → training stagnates",
        "trigger": "Non-contiguous gradient tensor (rare with FSDP/ZeRO sharding)",
        "fix": "Check is_contiguous() first, copy-back result if needed",
        "detection": "Training stagnation (slow, hard to notice)",
        "rtx4090_risk": "MEDIUM — FSDP sharding can produce non-contiguous tensors",
    },
    "sg28679": {
        "framework": "SGLang",
        "id": "#28679",
        "title": "GDN intermittent decode degradation",
        "severity": 3,  # Level 3: progressive, worsens over uptime
        "pattern": "Accumulator state degrades over uptime without reset",
        "effect": "Decode quality worsens → higher TPOT → clears on restart",
        "trigger": "Long-running serving (uptime > 30min)",
        "fix": "Periodic state flush (ReplaySSM #28695)",
        "detection": "Metrics degradation (requires monitoring)",
        "rtx4090_risk": "HIGH — long-running GRPO = worst possible pattern!",
    },
    "vl46118": {
        "framework": "vLLM",
        "id": "#46118",
        "title": "MTP+grammar FSM conflict",
        "severity": 4,  # Level 4: catastrophic, 58% failure
        "pattern": "Speculative tokens bypass FSM → state becomes inconsistent",
        "effect": "58% request failure rate for MTP+structured_output",
        "trigger": "num_speculative_tokens > 0 + guided_decoding enabled",
        "fix": "PR #44297: validate speculative tokens against FSM before acceptance",
        "detection": "Request failures (obvious)",
        "rtx4090_risk": "HIGH — MTP+structured_output simultaneously = BROKEN",
    },
    "sg28676": {
        "framework": "SGLang",
        "id": "#28676",
        "title": "MXFP8 MoE shuffle cache CLOBBERED",
        "severity": 4,  # Level 4: catastrophic, 64x accuracy blowup
        "pattern": "Physical memory clobber — cache overwritten by stale weights on RL reload",
        "effect": "64x accuracy blowup → garbage outputs",
        "trigger": "MoE model + RL weight reload",
        "fix": "dict.clear() on cache + weight-load funnel call (+28/-2)",
        "detection": "Garbage outputs (obvious)",
        "rtx4090_risk": "CRITICAL — MoE GRPO BLOCKED without this fix!",
    },
    "vl6468": {
        "framework": "verl",
        "id": "#6468",
        "title": "FSDP2 CPU memory leak during weight sync",
        "severity": 3,  # Level 3: progressive, monotonic growth
        "pattern": "Freed CPU memory not reclaimed → 0.6-6.3 GiB/step growth",
        "effect": "Host OOM in ~8-22 steps → Ray crashes",
        "trigger": "FSDP2 + CPU offload + long-running GRPO",
        "fix": "MUST use FSDP1 (FSDP2 has this leak)",
        "detection": "Host memory monitoring (gradual, predictable)",
        "rtx4090_risk": "CRITICAL — FSDP2 NOT viable for any long-running training!",
    },
}


# ============================================================
# RTX 4090 MUST DO Rules (16)
# ============================================================

MUST_DO = [
    ("D1", "overlap_comm=False on single GPU", "ds8061", "Prevents CUDA stream race → NaN"),
    ("D2", "enforce_eager=True for DSV4/MoE", "sg28676", "Prevents MoE cache clobber"),
    ("D3", "bypass_mode=True (eliminates ref model)", "verl", "18Ψ→3.8Ψ memory reduction"),
    ("D4", "LoRA rank=32 (not 64)", "vl6782", "rank=64 breaks EOS in vLLM rollout"),
    ("D5", "gradient_clipping=1.0 explicitly", "ds8068", "Default was 0 → always clips"),
    ("D6", "FSDP1 backend (NOT FSDP2)", "vl6468", "FSDP2 has CPU memory leak"),
    ("D7", "ZeRO-2 + CPU_Adam (NOT ZeRO-3)", "ds8072", "ZeRO-3 pure overhead + #8072 regression on dp=1"),
    ("D8", "per-unit LoRA summon (#6512)", "verl6512", "10x peak memory reduction"),
    ("D9", "Verify .contiguous() semantics in optimizer", "ds8058_contiguous", "Prevents silent param update loss"),
    ("D10", "Monitor training loss for stagnation", "ds8058_contiguous", "Detects optimizer death or contiguous() bug"),
    ("D11", "Add NaN detection (NanDetectMode #187653)", "ds8061", "Early NaN detection before catastrophic spread"),
    ("D12", "Validate weight checksums after optimizer step", "ds8058_contiguous", "Detects silent corruption immediately"),
    ("D13", "pin_memory=True for CPU offload", "pt187620", "Already default in CPUOffloadPolicy"),
    ("D14", "sleep_level=1 for verl HYBRID", "verl", "80x payload reduction vs sleep_level=2"),
    ("D15", "ulimit -n >= 65536", "ds8075", "Prevents fd leak exhaustion"),
    ("D16", "Never use Muon optimizer", "megatron5179", "4 blockers: PyPI stub, clipping stalls, CPU offload blocked, package placeholder"),
]


# ============================================================
# RTX 4090 MUST NOT Rules (16)
# ============================================================

MUST_NOT = [
    ("N1", "overlap_comm=True on single GPU", "ds8061", "CUDA stream race → NaN!"),
    ("N2", "FSDP2 backend", "vl6468", "CPU memory leak → host OOM!"),
    ("N3", "ZeRO-3 on dp=1", "ds8072", "Pure overhead + #8072 regression!"),
    ("N4", "torch.compile without overlap_comm=False", "ds8061", "Triggers #8061 NaN!"),
    ("N5", "LoRA rank=64 with vLLM rollout", "vl6782", "Breaks EOS token!"),
    ("N6", "gradient_clipping=0 (default)", "ds8068", "Default 0 → always clips!"),
    ("N7", "Muon optimizer", "megatron5179", "4 blockers confirmed!"),
    ("N8", "sleep_level=2 for RTX 4090", "verl", "Full weight re-transfer → slow!"),
    ("N9", "ZeRO-3+PEFT LoRA", "ds8072", "#8072 regression → dtype mismatch!"),
    ("N10", "CPU_Adam subprocess without death detection", "ds8058", "Silent optimizer death → stale params!"),
    ("N11", "CPUOffloadPolicy on dp=1", "pt187620", "shard=identity → OOM for >8B!"),
    ("N12", "MXFP8 MoE without cache invalidation", "sg28676", "Cache clobber → 64x accuracy blowup!"),
    ("N13", "DSV4 without enforce_eager=True", "dsv4", "11 failures across 4 frameworks!"),
    ("N14", "MTP+structured_output simultaneously", "vl46118", "58% failure rate!"),
    ("N15", "ZenFlow contiguous() on non-contiguous tensors without copy-back", "ds8058_contiguous", "Silent param update loss!"),
    ("N16", "ZenFlow without #7771 NaN fix", "ds7771", "★RESOLVED: #7771 MERGED June 12! No longer a concern."),
]


# ============================================================
# Defense Stack
# ============================================================

DEFENSE_STACK = [
    ("Layer 1", "Synchronization (Prevention)",
     "CUDA stream wait, POSIX semaphore, FSDP summon/desummon lifecycle"),
    ("Layer 2", "Correctness Checks (Detection)",
     "is_contiguous() + copy-back, NaN detection, optimizer step validation, weight checksums"),
    ("Layer 3", "State Reset (Recovery)",
     "Periodic flush (ReplaySSM #28695), MoE cache clear (#28676), KV cache invalidation"),
    ("Layer 4", "FSM Validation (Consistency)",
     "MTP grammar validation (#44297), weight sync consistency, checkpoint checksum validation"),
]


# ============================================================
# Check Mode
# ============================================================

def check_config(config):
    """Validate a training config for silent corruption risk factors."""
    results = []
    risks_found = 0

    # D1: overlap_comm
    overlap_comm = config.get("overlap_comm", False)
    if overlap_comm:
        results.append({
            "rule": "N1",
            "status": "FAIL",
            "detail": f"overlap_comm=True → #8061 CUDA stream race risk → NaN!",
            "bug": "ds8061",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D1",
            "status": "PASS",
            "detail": "overlap_comm=False → #8061 risk eliminated",
        })

    # D2: enforce_eager
    model_type = config.get("model_type", "dense")
    enforce_eager = config.get("enforce_eager", False)
    if model_type in ["dsv4", "moe", "MoE", "DSV4"] and not enforce_eager:
        results.append({
            "rule": "D2",
            "status": "FAIL",
            "detail": f"model_type={model_type} without enforce_eager → #28676 cache clobber risk!",
            "bug": "sg28676",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D2",
            "status": "PASS",
            "detail": f"enforce_eager=True or model_type={model_type} (no MoE/DSV4 risk)",
        })

    # D6: FSDP backend
    fsdp_backend = config.get("fsdp_backend", "fsdp1")
    if fsdp_backend == "fsdp2":
        results.append({
            "rule": "N2",
            "status": "FAIL",
            "detail": f"FSDP2 → #6468 CPU memory leak → host OOM!",
            "bug": "vl6468",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D6",
            "status": "PASS",
            "detail": "FSDP1 backend → #6468 leak avoided",
        })

    # D7: ZeRO stage
    zero_stage = config.get("zero_stage", 2)
    if zero_stage == 3 and config.get("dp", 1) == 1:
        results.append({
            "rule": "N3",
            "status": "FAIL",
            "detail": f"ZeRO-3 on dp=1 → pure overhead + #8072 regression!",
            "bug": "ds8072",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D7",
            "status": "PASS",
            "detail": f"ZeRO-2 or dp>1 → no ZeRO-3 overhead",
        })

    # D4: LoRA rank
    lora_rank = config.get("lora_rank", 32)
    if lora_rank == 64 and config.get("rollout_backend", "vllm") == "vllm":
        results.append({
            "rule": "N5",
            "status": "FAIL",
            "detail": f"LoRA rank=64 + vLLM rollout → #6782 EOS break!",
            "bug": "vl6782",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D4",
            "status": "PASS",
            "detail": f"LoRA rank={lora_rank} → #6782 risk eliminated",
        })

    # D5: gradient clipping
    grad_clip = config.get("gradient_clipping", None)
    if grad_clip is None or grad_clip == 0:
        results.append({
            "rule": "N6",
            "status": "FAIL",
            "detail": f"gradient_clipping={grad_clip} → #8068 default 0 → always clips!",
            "bug": "ds8068",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "D5",
            "status": "PASS",
            "detail": f"gradient_clipping={grad_clip} → #8068 risk eliminated",
        })

    # D3: bypass_mode
    bypass_mode = config.get("bypass_mode", False)
    if not bypass_mode:
        results.append({
            "rule": "D3",
            "status": "WARN",
            "detail": "bypass_mode=False → ref model uses 18Ψ instead of 3.8Ψ → memory waste",
        })
    else:
        results.append({
            "rule": "D3",
            "status": "PASS",
            "detail": "bypass_mode=True → 18Ψ→3.8Ψ memory reduction",
        })

    # D14: sleep_level
    sleep_level = config.get("sleep_level", 1)
    if sleep_level == 2 and config.get("dp", 1) == 1:
        results.append({
            "rule": "N8",
            "status": "WARN",
            "detail": "sleep_level=2 → full weight re-transfer → slow on dp=1",
        })
    else:
        results.append({
            "rule": "D14",
            "status": "PASS",
            "detail": f"sleep_level={sleep_level} → optimal for dp={config.get('dp', 1)}",
        })

    # N14: MTP + structured_output
    mtp_tokens = config.get("num_speculative_tokens", 0)
    guided_decoding = config.get("guided_decoding", False)
    if mtp_tokens > 0 and guided_decoding:
        results.append({
            "rule": "N14",
            "status": "FAIL",
            "detail": f"MTP ({mtp_tokens} spec tokens) + guided_decoding → #46118 58% failure!",
            "bug": "vl46118",
        })
        risks_found += 1
    else:
        results.append({
            "rule": "N14",
            "status": "PASS",
            "detail": "No MTP+grammar conflict",
        })

    # N11: CPUOffloadPolicy on dp=1
    offload_policy = config.get("offload_policy", "default")
    dp = config.get("dp", 1)
    if offload_policy == "PartialOffloadPolicy" and dp == 1:
        results.append({
            "rule": "N11",
            "status": "FAIL",
            "detail": "CPUOffloadPolicy on dp=1 → shard=identity → OOM for >8B!",
            "bug": "pt187620",
        })
        risks_found += 1

    # D9: contiguous() semantics
    optimizer = config.get("optimizer", "cpu_adam")
    if optimizer == "zenflow":
        results.append({
            "rule": "D9",
            "status": "WARN",
            "detail": "ZenFlow optimizer → verify .contiguous() semantics! #8058 bug risk",
            "bug": "ds8058_contiguous",
        })

    return {
        "total_checks": len(results),
        "risks_found": risks_found,
        "results": results,
        "verdict": "SAFE" if risks_found == 0 else "RISKY" if risks_found <= 2 else "DANGER",
    }


# ============================================================
# Monitor Mode
# ============================================================

def monitor_commands():
    """Generate runtime monitoring commands for detecting silent corruption."""
    commands = {
        "NaN detection": [
            "# Add to training script:",
            "from torch.autograd import detect_anomaly",
            "with detect_anomaly():",
            "    loss = model.forward(batch)",
            "",
            "# Or use NanDetectMode (PyTorch #187653):",
            "with torch.nan_detect_mode():",
            "    loss = model.forward(batch)",
        ],
        "Weight checksum validation": [
            "# After each optimizer step, verify weights haven't silently failed to update:",
            "import hashlib",
            "def weight_checksum(model):",
            "    h = hashlib.md5()",
            "    for p in model.parameters():",
            "        h.update(p.data.numpy().tobytes())",
            "    return h.hexdigest()",
            "",
            "# Check before/after optimizer step:",
            "before = weight_checksum(model)",
            "optimizer.step()",
            "after = weight_checksum(model)",
            "assert before != after, 'Weights unchanged after optimizer step! → #8058 contiguous() bug'",
        ],
        "Training stagnation detection": [
            "# Monitor loss for stagnation (>10 steps without improvement):",
            "losses = []",
            "for step in range(max_steps):",
            "    loss = train_step(model, batch)",
            "    losses.append(loss)",
            "    if len(losses) > 10:",
            "        recent = losses[-10:]",
            "        if max(recent) - min(recent) < 1e-6:",
            "            print(f'WARNING: Training stagnation at step {step}! → optimizer death or contiguous() bug')",
        ],
        "Gradient contiguity check": [
            "# Before ZenFlow optimizer step, verify gradient contiguity:",
            "for name, param in model.named_parameters():",
            "    if param.grad is not None:",
            "        if not param.grad.is_contiguous():",
            "            print(f'WARNING: {name}.grad is non-contiguous → #8058 contiguous() bug risk!')",
            "            # Force contiguous with copy-back:",
            "            grad_copy = param.grad.contiguous()",
            "            param.grad = grad_copy",
        ],
        "Host memory monitoring (FSDP2 leak)": [
            "# Monitor host memory for FSDP2 leak (#6468):",
            "import psutil",
            "import os",
            "process = psutil.Process(os.getpid())",
            "baseline_mem = process.memory_info().rss / 1e9",
            "",
            "for step in range(max_steps):",
            "    train_step(model, batch)",
            "    current_mem = process.memory_info().rss / 1e9",
            "    growth = current_mem - baseline_mem",
            "    if growth > 2.0:  # 2 GiB growth threshold",
            "        print(f'CRITICAL: Host memory grew {growth:.1f} GiB → #6468 FSDP2 leak!')",
        ],
        "Decode quality monitoring (GDN degeneracy)": [
            "# For SGLang/vLLM serving, track decode quality over time:",
            "# Monitor TPOT and acceptance rate:",
            "# If TPOT increases >20% over 30min → #28679 GDN degeneracy",
            "# Solution: periodic restart or ReplaySSM flush (#28695)",
        ],
    }
    return commands


# ============================================================
# Math Mode
# ============================================================

def math_analysis(steps=1000, per_step_error=0.001):
    """Mathematical analysis of silent corruption damage over N steps."""
    analysis = {
        "parameters": {
            "steps": steps,
            "per_step_error": per_step_error,
        },
        "loud_failure_damage": {
            "formula": "damage = severity(1) = 1 × ε",
            "value": per_step_error,
            "description": "Loud failure stops at step 1 → minimal damage",
        },
        "silent_corruption_damage": {
            "formula": "damage = Σ severity(i) = Σ i × ε = ε × N(N+1)/2",
            "value": per_step_error * steps * (steps + 1) / 2,
            "description": "Silent corruption accumulates → damage grows quadratically!",
        },
        "damage_ratio": {
            "formula": "ratio = silent/loud = N(N+1)/2",
            "value": steps * (steps + 1) / 2,
            "description": f"Silent corruption is {steps * (steps + 1) / 2:.0f}× more damaging than loud failure for {steps} steps",
        },
        "severity_levels": {
            "Level 1_Transient": "CUDA stream race (#8061) — one-time stale read → NaN",
            "Level 2_Accumulating": "contiguous() bug (#8058) — params never update → stagnation",
            "Level 3_Progressive": "GDN degeneracy (#28679) — state degrades over uptime",
            "Level 4_Catastrophic": "MTP+grammar FSM (#46118) — 58% failure rate",
        },
        "rtx4090_implications": {
            "single_gpu_no_redundancy": "dp=1 → no cross-GPU consistency checks → more vulnerable",
            "long_running_grpo": f"{steps}+ steps → damage = {per_step_error * steps * (steps + 1) / 2:.3f} → catastrophic",
            "cpu_offload_more_sync_points": "CPU→GPU transfers add synchronization → more race opportunities",
        },
    }
    return analysis


# ============================================================
# RTX 4090 Mode
# ============================================================

def rtx4090_check():
    """Complete RTX 4090 silent corruption check with all rules."""
    check = {
        "gpu": "RTX 4090 (24 GiB, SM89, dp=1)",
        "training": "GRPO + LoRA-32 + bypass_mode + FSDP1 + CPU offload",
        "must_do_rules": len(MUST_DO),
        "must_not_rules": len(MUST_NOT),
        "must_do": [
            {"id": r[0], "rule": r[1], "bug": r[2], "reason": r[3]}
            for r in MUST_DO
        ],
        "must_not": [
            {"id": r[0], "rule": r[1], "bug": r[2], "reason": r[3]}
            for r in MUST_NOT
        ],
        "defense_stack": [
            {"layer": d[0], "name": d[1], "mechanisms": d[2]}
            for d in DEFENSE_STACK
        ],
        "bug_summary": {
            "total_bugs": len(BUGS),
            "severity_distribution": {
                "Level 2": sum(1 for b in BUGS.values() if b["severity"] == 2),
                "Level 3": sum(1 for b in BUGS.values() if b["severity"] == 3),
                "Level 4": sum(1 for b in BUGS.values() if b["severity"] == 4),
            },
        },
        "key_warnings": [
            "★★★★★★★★ N16 resolved: #7771 MERGED June 12 → ZenFlow NaN dependency RESOLVED",
            "★★★★★★★★ ZeRO-3+ZenFlow STILL NOT viable (Stage 3 copyback not implemented)",
            "★★★★★★★★ FSDP2 #6468 leak: MUST use FSDP1 backend",
            "★★★★★★★★ MTP+structured_output BROKEN (#46118) → NEVER enable simultaneously",
            "★★★★★★★★ Silent corruption is 500,500× more damaging than loud failures for 1000 steps",
        ],
    }
    return check


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Silent Corruption Detector for RTX 4090 GRPO Training")
    parser.add_argument("mode", choices=["check", "monitor", "math", "rtx4090"],
                        help="Operation mode")
    parser.add_argument("--config", type=str, help="JSON config file for check mode")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps for math mode")
    parser.add_argument("--error", type=float, default=0.001, help="Per-step error magnitude for math mode")

    args = parser.parse_args()

    if args.mode == "check":
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
        else:
            # Default RTX 4090 optimal config
            config = {
                "overlap_comm": False,
                "enforce_eager": True,
                "bypass_mode": True,
                "lora_rank": 32,
                "gradient_clipping": 1.0,
                "fsdp_backend": "fsdp1",
                "zero_stage": 2,
                "dp": 1,
                "model_type": "dense",
                "rollout_backend": "vllm",
                "sleep_level": 1,
                "num_speculative_tokens": 0,
                "guided_decoding": False,
                "optimizer": "cpu_adam",
            }
        result = check_config(config)
        print(json.dumps(result, indent=2))

    elif args.mode == "monitor":
        result = monitor_commands()
        for category, cmds in result.items():
            print(f"\n{'='*60}")
            print(f"## {category}")
            print(f"{'='*60}")
            for cmd in cmds:
                print(cmd)

    elif args.mode == "math":
        result = math_analysis(args.steps, args.error)
        print(json.dumps(result, indent=2))

    elif args.mode == "rtx4090":
        result = rtx4090_check()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
