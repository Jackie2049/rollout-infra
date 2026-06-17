#!/usr/bin/env python3
"""RTX 4090 GPU Experiment Runner — P10→P1 Priority Order

Orchestrates all GPU-dependent experiments in priority order when GPU is online.
P10 BudgetRefiner profile data collection is #1 priority (UNIQUE data).

Usage:
  python3 tools/rtx4090_gpu_experiment_runner.py --check-gpu
  python3 tools/rtx4090_gpu_experiment_runner.py --run p10
  python3 tools/rtx4090_gpu_experiment_runner.py --run p9
  python3 tools/rtx4090_gpu_experiment_runner.py --run all
  python3 tools/rtx4090_gpu_experiment_runner.py --run quick  # P10+P9 only

Reference:
  - tools/gpu_experiment_queue.md (priority list)
  - notebook/projects/budgetrefiner-vllm-pr-draft.md (P10)
  - notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md (P9)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXPERIMENTS = {
    "p10": {
        "name": "BudgetRefiner SLO Profile Data Collection",
        "priority": "UNIQUE — NO other vLLM contributor has RTX 4090 data",
        "script": "tools/profile_vllm_budget.py --mode collect",
        "output": "profile_table_rtx4090.csv",
        "description": "Collect (ctx_len, d_num) → chunk_size data for BudgetRefiner lookup table",
        "estimated_time": "2-4 hours",
        "gpu_required": True,
    },
    "p9": {
        "name": "SM89 Batch Invariance Reproduction",
        "priority": "CRITICAL — validates Fusion Guard PR root cause",
        "script": "tools/sm89_batch_invariance_repro.py",
        "output": "batch_invariance_results.json",
        "description": "Reproduce batch-dependent RMSNorm output on SM89 with 3 configs",
        "estimated_time": "30-60 min",
        "gpu_required": True,
    },
    "p8": {
        "name": "INT8 KV Cache Throughput Benchmark",
        "priority": "CONFIG — INT8 vs BF16 KV throughput measurement",
        "script": "tools/sm89_kv_cache_cost_analyzer.py --benchmark",
        "output": "int8_kv_benchmark.json",
        "description": "Measure INT8 KV throughput vs BF16 on RTX 4090",
        "estimated_time": "30 min",
        "gpu_required": True,
    },
    "p6": {
        "name": "AutoEP MoE Smoke Test",
        "priority": "UNIQUE — ONLY framework supporting MoE on RTX 4090",
        "script": "deepspeed --num_gpus=1 train.py --deepspeed configs/moe-autoep_rtx4090.json",
        "output": "autoep_moe_smoke_test.log",
        "description": "Verify Qwen3-MoE AutoEP training starts without OOM",
        "estimated_time": "10-20 min (just boot + 5 steps)",
        "gpu_required": True,
    },
}


def check_gpu():
    """Check if a CUDA GPU is available and report its specs."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("NO GPU AVAILABLE — torch.cuda.is_available() = False")
            print("Check: 1) nvidia-smi, 2) CUDA driver, 3) PyTorch CUDA build")
            return False

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        memory = torch.cuda.get_device_properties(device).total_mem / (1024**3)

        print(f"GPU FOUND: {name}")
        print(f"  SM version: {capability[0]}.{capability[1]}")
        print(f"  VRAM: {memory:.1f} GB")

        if capability[0] == 8 and capability[1] == 9:
            print("  ★★★★★ RTX 4090 (SM89) DETECTED — PERFECT for all experiments!")
        elif capability[0] >= 9:
            print(f"  ★★★ SM{capability[0]} detected — P9 (batch invariance) may not trigger")
            print(f"  → P10 (BudgetRefiner profile) still valuable for SM{capability[0]}")
        elif capability[0] == 8:
            print(f"  ★★ SM{capability[0]}.{capability[1]} detected — batch invariance repro applicable")
            print("  → BudgetRefiner profile data still collectible")
        else:
            print(f"  ⚠ SM{capability[0]} detected — some experiments may not apply")

        # Check memory headroom
        free_mem = (torch.cuda.get_device_properties(device).total_mem -
                    torch.cuda.memory_allocated(device)) / (1024**3)
        print(f"  Free memory: {free_mem:.1f} GB")

        return True
    except ImportError:
        print("PyTorch NOT installed — install with: pip install torch")
        return False


def check_prerequisites():
    """Check if required packages are installed."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import vllm
    except ImportError:
        missing.append("vllm (needed for P10)")

    try:
        import deepspeed
    except ImportError:
        missing.append("deepspeed (needed for P6)")

    try:
        import pandas
    except ImportError:
        missing.append("pandas (needed for P10)")

    if missing:
        print("MISSING packages:", ", ".join(missing))
        print("Install with: pip install -i https://mirrors.aliyun.com/pypi/simple/ " + " ".join(missing))
        return False
    return True


def run_experiment(exp_id):
    """Run a single experiment by ID."""
    exp = EXPERIMENTS.get(exp_id)
    if not exp:
        print(f"Unknown experiment: {exp_id}")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {exp['name']} [{exp_id}]")
    print(f"Priority: {exp['priority']}")
    print(f"Estimated time: {exp['estimated_time']}")
    print(f"{'='*60}")

    if not check_gpu():
        print("GPU required but not available — skipping")
        return False

    print(f"\nCommand: {exp['script']}")
    print(f"Output: {exp['output']}")
    print(f"\n★★★★★ STARTING EXPERIMENT — DO NOT INTERRUPT GPU WORK ★★★★★")

    # Record start time
    start = time.time()

    # Run the actual script
    result = os.system(exp["script"])
    elapsed = time.time() - start

    success = result == 0
    print(f"\n{'='*60}")
    if success:
        print(f"★★★★★ COMPLETED in {elapsed:.0f}s ★★★★★")
        print(f"Output saved to: {exp['output']}")
    else:
        print(f"★★★ FAILED (exit code {result}) in {elapsed:.0f}s ★★★")
    print(f"{'='*60}")

    return success


def run_all(priority_only=False):
    """Run experiments in priority order."""
    if not check_gpu():
        print("\n★★★★ GPU NOT AVAILABLE ★★★★")
        print("When GPU comes online, run: python3 tools/rtx4090_gpu_experiment_runner.py --run all")
        print("\nPriority order:")
        for exp_id in ["p10", "p9", "p8", "p6"]:
            exp = EXPERIMENTS[exp_id]
            print(f"  {exp_id}: {exp['name']} ({exp['estimated_time']})")
        return

    if not check_prerequisites():
        print("Install missing packages first")
        return

    order = ["p10", "p9"] if priority_only else ["p10", "p9", "p8", "p6"]
    results = {}

    print(f"\n★★★★★★★★★ RUNNING {len(order)} EXPERIMENTS IN PRIORITY ORDER ★★★★★★★★★★")
    for exp_id in order:
        success = run_experiment(exp_id)
        results[exp_id] = {
            "success": success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for exp_id, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"  {exp_id}: {EXPERIMENTS[exp_id]['name']} → {status}")

    # Save results
    results_file = "gpu_experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 GPU Experiment Runner")
    parser.add_argument("--check-gpu", action="store_true",
                        help="Check if GPU is available and report specs")
    parser.add_argument("--check-prereqs", action="store_true",
                        help="Check if required packages are installed")
    parser.add_argument("--run", choices=["p10", "p9", "p8", "p6", "all", "quick"],
                        help="Run experiment(s): p10=BudgetRefiner, p9=batch invariance, "
                             "p8=INT8 KV, p6=AutoEP MoE, all=all in order, quick=p10+p9 only")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments with priorities")

    args = parser.parse_args()

    if args.list:
        print("GPU Experiment Queue (priority order):")
        for exp_id in ["p10", "p9", "p8", "p6"]:
            exp = EXPERIMENTS[exp_id]
            print(f"\n  {exp_id}: {exp['name']}")
            print(f"    Priority: {exp['priority']}")
            print(f"    Time: {exp['estimated_time']}")
            print(f"    Script: {exp['script']}")
            print(f"    Output: {exp['output']}")
        return

    if args.check_gpu:
        check_gpu()
        return

    if args.check_prereqs:
        check_prerequisites()
        return

    if args.run:
        if args.run in ["p10", "p9", "p8", "p6"]:
            run_experiment(args.run)
        elif args.run == "all":
            run_all(priority_only=False)
        elif args.run == "quick":
            run_all(priority_only=True)
        return

    print("No action specified. Use --check-gpu, --run <id>, or --list")
    print("Priority: p10 (BudgetRefiner UNIQUE) → p9 (batch invariance) → p8 (INT8 KV) → p6 (AutoEP MoE)")


if __name__ == "__main__":
    main()
