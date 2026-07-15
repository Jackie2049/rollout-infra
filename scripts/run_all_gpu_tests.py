"""
GPU Test Runner — Execute all PR validation tests on partner H20-3e server.

This script orchestrates all GPU-dependent tests that validate our fork PRs.
Designed to run remotely via SSH on the partner server (10.26.6.88:31954).

Usage:
  # Run all tests
  python scripts/run_all_gpu_tests.py --all

  # Run specific test category
  python scripts/run_all_gpu_tests.py --category up_grpo
  python scripts/run_all_gpu_tests.py --category lora_nan
  python scripts/run_all_gpu_tests.py --category cuda_race

  # Run via SSH
  sshpass -p 'JDF+H200@sribd' ssh root@10.26.6.88 -p 31954 \
    "cd /jiangdingfeng/zy/Termius/rollout && source activate env-rollout && python scripts/run_all_gpu_tests.py --all"
"""

import subprocess
import sys
import json
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

TEST_REGISTRY = {
    "up_grpo": {
        "script": "test_up_grpo_h20.py",
        "description": "UP-GRPO policy loss validation (5 tests: A>0 unbounded, A<0 dual-clip, gradient flow)",
        "pr_ref": "Jackie2049/verl PR #9",
        "timeout": 120,
    },
    "lora_nan_targeted": {
        "script": "test_vllm_lora_nan_targeted_h20.py",
        "description": "vLLM LoRA NaN targeted test (lora_expand block_n=128 vs 32 on sm_90)",
        "pr_ref": "Jackie2049/vllm PR #9",
        "timeout": 180,
    },
    "lora_nan_full": {
        "script": "test_vllm_lora_nan_h20.py",
        "description": "vLLM LoRA NaN full test (multiple configs on sm_90)",
        "pr_ref": "Jackie2049/vllm PR #9",
        "timeout": 300,
    },
    "cuda_race_v1": {
        "script": "cuda_multi_stream_race_demo_h20.py",
        "description": "CUDA stream race demo v1 (basic write/read race)",
        "pr_ref": "Jackie2049/DeepSpeed PR #1 + Jackie2049/Megatron-LM PR #1",
        "timeout": 120,
    },
    "cuda_race_v2": {
        "script": "cuda_multi_stream_race_demo_v2_h20.py",
        "description": "CUDA stream race demo v2 (improved, free/realloc pattern)",
        "pr_ref": "Jackie2049/DeepSpeed PR #1 + Jackie2049/Megatron-LM PR #1",
        "timeout": 180,
    },
    "grpo_smoke": {
        "script": "grpo_smoke_test_h20.py",
        "description": "GRPO training smoke test (basic sanity check)",
        "pr_ref": "General GRPO validation",
        "timeout": 300,
    },
}

def check_gpu():
    """Check GPU availability and print info."""
    result = subprocess.run(
        ["python", "-c",
         "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); "
         "print(f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'); "
         "print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print("GPU NOT AVAILABLE — cannot run tests")
        return False
    print(result.stdout.strip())
    return True

def run_test(category):
    """Run a single test category."""
    info = TEST_REGISTRY[category]
    script_path = SCRIPTS_DIR / info["script"]

    if not script_path.exists():
        print(f"  SKIP: {info['script']} not found")
        return {"category": category, "status": "SKIP", "reason": "script not found"}

    print(f"\n{'='*60}")
    print(f"Running: {info['description']}")
    print(f"PR Ref: {info['pr_ref']}")
    print(f"Script: {info['script']}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True, text=True, timeout=info["timeout"]
    )
    elapsed = time.time() - start

    status = "PASS" if result.returncode == 0 else "FAIL"

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:1000])

    print(f"\nResult: {status} ({elapsed:.1f}s)")

    return {
        "category": category,
        "status": status,
        "elapsed": elapsed,
        "pr_ref": info["pr_ref"],
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--category", choices=list(TEST_REGISTRY.keys()), help="Run specific test")
    parser.add_argument("--check-gpu", action="store_true", help="Only check GPU availability")
    parser.add_argument("--output", default="gpu_test_results.json", help="Output file for results")
    args = parser.parse_args()

    if args.check_gpu:
        check_gpu()
        return

    if not check_gpu():
        sys.exit(1)

    results = []

    if args.all:
        for category in TEST_REGISTRY:
            results.append(run_test(category))
    elif args.category:
        results.append(run_test(args.category))
    else:
        print("Specify --all or --category. Available categories:")
        for cat, info in TEST_REGISTRY.items():
            print(f"  {cat}: {info['description']} ({info['pr_ref']})")
        return

    # Save results
    output_path = SCRIPTS_DIR.parent / args.output
    with open(output_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {pass_count} PASS, {fail_count} FAIL, {skip_count} SKIP")
    print(f"{'='*60}")

    for r in results:
        print(f"  {r['category']}: {r['status']} ({r.get('elapsed', 0):.1f}s) — {r.get('pr_ref', '')}")

if __name__ == "__main__":
    main()
