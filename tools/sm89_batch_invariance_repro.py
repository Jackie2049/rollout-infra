#!/usr/bin/env python3
"""SM89 Batch Invariance Reproduction & Investigation Script

Minimal repro for vLLM Issue #39096: batch invariance breaks with
torch.compile on SM<90 GPUs. This script tests 3 configurations:

1. Baseline (no compile, no graphs) — should pass
2. torch.compile only (no CUDA graphs) — expected FAIL on SM<90
3. torch.compile + CUDA graphs — expected FAIL on SM<90

For each config, compares batch=1 vs batch=4 of the same input row
and checks bitwise equality.

Requires: GPU (CUDA), torch, vllm installed
Target: SM89 (RTX 4090 / L4) but also tests SM90+ for comparison

Reference:
- Issue #39096: https://github.com/vllm-project/vllm/issues/39096
- Minimal repro from issue body
"""

import argparse
import json
import os
import sys
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_sm_info():
    """Get GPU compute capability info."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None, "N/A", "No CUDA GPU available"
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    status = "SM90+" if cap[0] >= 9 else "SM<90"
    return cap, name, status


def test_rms_norm_batch_invariant(compile_mode="none", verbose=False):
    """Test isolated RMSNorm batch invariance (from Issue #39096 repro)."""
    cap, gpu_name, sm_status = get_sm_info()
    if cap is None:
        print("  ✗ No CUDA GPU — cannot test")
        return False

    hidden_size = 4096
    eps = 1e-5

    torch.manual_seed(0)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")

    shared_row = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    x_b1 = shared_row.unsqueeze(0).clone()
    x_b4 = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")
    x_b4[0] = shared_row

    def rms_norm_native(x, w, eps=eps):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(orig_dtype)
        return x * w

    if compile_mode == "compile":
        rms_norm_native = torch.compile(rms_norm_native, dynamic=False)

    # Warmup for compile
    if compile_mode == "compile":
        for _ in range(3):
            _ = rms_norm_native(x_b1, weight)

    out_b1 = rms_norm_native(x_b1, weight)
    out_b4 = rms_norm_native(x_b4, weight)

    bitwise_equal = torch.equal(out_b1[0], out_b4[0])
    max_diff = (out_b1[0].to(torch.float32) - out_b4[0].to(torch.float32)).abs().max().item()

    result = {
        "test": "rms_norm_isolated",
        "compile_mode": compile_mode,
        "gpu": gpu_name,
        "sm": f"{cap[0]}.{cap[1]}",
        "sm_status": sm_status,
        "bitwise_equal": bitwise_equal,
        "max_abs_diff": max_diff,
        "pass": bitwise_equal,
    }

    if verbose:
        status = "✓" if bitwise_equal else "✗"
        print(f"  {status} RMSNorm isolated | compile={compile_mode} | SM={cap[0]}.{cap[1]} | bitwise={bitwise_equal} | diff={max_diff}")

    return result


def test_full_model_batch_invariant(model_name, compile_mode="none",
                                     enforce_eager=True, verbose=False):
    """Test full model forward batch invariance using vLLM.

    This requires vLLM to be installed and a model available.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    except ImportError:
        print("  ✗ vLLM not installed — cannot test full model")
        return None

    cap, gpu_name, sm_status = get_sm_info()
    if cap is None:
        print("  ✗ No CUDA GPU")
        return None

    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    # Configure compilation
    compilation_config = {}
    if compile_mode == "none":
        compilation_config = {"level": 0}
    elif compile_mode == "compile_no_graphs":
        compilation_config = {"level": 1, "cudagraph_mode": "NONE"}
    elif compile_mode == "compile_with_graphs":
        compilation_config = {"level": 1, "cudagraph_mode": "FULL_AND_PIECEWISE"}

    results = []

    try:
        llm = LLM(
            model=model_name,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.90,
            max_model_len=512,
            compilation_config=compilation_config if compilation_config else None,
        )

        # Test prompt
        prompt = "The capital of France is"

        # Single request
        sp1 = SamplingParams(max_tokens=20, temperature=0)
        out1 = llm.generate([prompt], sp1)
        tokens_b1 = out1[0].outputs[0].token_ids[:20]

        # Multiple requests (same prompt + different prompts)
        prompts = [prompt, "The capital of Germany is", "The capital of Italy is", "The capital of Spain is"]
        out4 = llm.generate(prompts, sp1)
        tokens_b4_first = out4[0].outputs[0].token_ids[:20]

        # Compare first prompt across batch sizes
        match = tokens_b1 == tokens_b4_first
        divergence_pos = None
        if not match:
            for i, (t1, t4) in enumerate(zip(tokens_b1, tokens_b4_first)):
                if t1 != t4:
                    divergence_pos = i
                    break

        result = {
            "test": "full_model_forward",
            "model": model_name,
            "compile_mode": compile_mode,
            "enforce_eager": enforce_eager,
            "sm": f"{cap[0]}.{cap[1]}",
            "sm_status": sm_status,
            "batch_invariant": match,
            "divergence_at": divergence_pos,
            "tokens_b1": tokens_b1[:10],
            "tokens_b4_first": tokens_b4_first[:10],
            "pass": match,
        }

        if verbose:
            status = "✓" if match else "✗"
            div_str = f"div@{divergence_pos}" if divergence_pos else "no-div"
            print(f"  {status} Full model | model={model_name} | compile={compile_mode} | eager={enforce_eager} | SM={cap[0]}.{cap[1]} | {div_str}")

        results.append(result)

    except Exception as e:
        result = {
            "test": "full_model_forward",
            "model": model_name,
            "compile_mode": compile_mode,
            "error": str(e),
            "pass": False,
        }
        if verbose:
            print(f"  ✗ Error: {e}")
        results.append(result)

    return results


def run_diagnostic(args):
    """Run full diagnostic across all configurations."""
    cap, gpu_name, sm_status = get_sm_info()

    print("=" * 60)
    print("SM89 Batch Invariance Reproduction & Investigation")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"SM: {cap[0]}.{cap[1]} ({sm_status})")
    print()

    if cap is None:
        print("No GPU available — cannot run reproduction test")
        print("Prepare test for when GPU comes online:")
        print("  1. Install vLLM: pip install vllm==0.23.0")
        print("  2. Download model: huggingface-cli download Qwen/Qwen3-1.7B")
        print("  3. Run: python tools/sm89_batch_invariance_repro.py --mode full")
        return

    all_results = []

    # Phase 1: Isolated RMSNorm test
    print("## Phase 1: Isolated RMSNorm Batch Invariance")
    print("  (Testing: torch.compile(rms_norm_native) alone)")
    print()

    for mode in ["none", "compile"]:
        result = test_rms_norm_batch_invariant(compile_mode=mode, verbose=True)
        all_results.append(result)

    print()

    # Phase 2: Full model test (if model available and vLLM installed)
    if args.model:
        print("## Phase 2: Full Model Batch Invariance")
        print(f"  Model: {args.model}")
        print()

        configs = [
            ("none", True, "Baseline: no compile, no graphs"),
            ("compile_no_graphs", False, "compile ON, graphs OFF (Issue #39096 repro)"),
            ("compile_with_graphs", False, "compile ON, graphs ON (original bug)"),
            ("compile_no_graphs", True, "compile ON, graphs OFF, enforce_eager (workaround)"),
        ]

        for compile_mode, enforce_eager, desc in configs:
            print(f"  Testing: {desc}")
            results = test_full_model_batch_invariant(
                args.model, compile_mode=compile_mode,
                enforce_eager=enforce_eager, verbose=True,
            )
            if results:
                all_results.extend(results)
            print()
    else:
        print("## Phase 2: Full Model — SKIPPED (no --model specified)")
        print("  To test: python tools/sm89_batch_invariance_repro.py --mode full --model Qwen/Qwen3-1.7B")
        print()

    # Summary
    print("=" * 60)
    print("## Summary")
    print()

    passed = [r for r in all_results if r.get("pass", False)]
    failed = [r for r in all_results if not r.get("pass", False)]

    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    print()

    if sm_status == "SM<90":
        print("★★★★★ SM<90 GPU detected — batch invariance bug likely reproducible!")
        print("  Expected results:")
        print("  ✓ RMSNorm isolated (compile=none) → PASS")
        print("  ✓ RMSNorm isolated (compile=compile) → PASS (isolated case works)")
        print("  ✗ Full model (compile+no graphs) → FAIL (this is the bug!)")
        print("  ✗ Full model (compile+graphs) → FAIL")
        print("  ✓ Full model (enforce_eager) → PASS (workaround)")
        print()
        print("  Root cause: Inductor full-graph fusion creates batch-dependent")
        print("  behavior on SM<90. Specifically, RMSNorm + residual add +")
        print("  linear input prep fusion in the full forward graph.")
    elif sm_status == "SM90+":
        print("★★★★ SM90+ GPU detected — batch invariance should work correctly")
        print("  All configurations expected to PASS on SM90+")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


def main():
    if not HAS_TORCH:
        print("✗ torch not installed — cannot run reproduction test")
        print("Install: pip install torch --index-url https://mirrors.aliyun.com/pypi/simple/")
        print("Or with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="SM89 Batch Invariance Reproduction & Investigation Script")
    parser.add_argument("--mode", choices=["quick", "full"],
                        default="quick", help="Test mode")
    parser.add_argument("--model", default=None,
                        help="Model name for full model test (e.g., Qwen/Qwen3-1.7B)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    run_diagnostic(args)


if __name__ == "__main__":
    main()
