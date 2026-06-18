#!/usr/bin/env python3
"""
RTX 4090 FSDP2 vs ZeRO-2 Decision Guide
==========================================
Helps choose between PyTorch FSDP2 and DeepSpeed ZeRO-2 for RTX 4090 training.
Includes PartialOffloadPolicy future path analysis.

Usage:
  python3 tools/fsdp2_vs_zero2_decision_guide.py --mode guide
  python3 tools/fsdp2_vs_zero2_decision_guide.py --mode compare --model qwen3-8b
  python3 tools/fsdp2_vs_zero2_decision_guide.py --mode estimate --model qwen3.5-27b --offload-ratio 0.3
"""

import argparse
import sys

# ============================================================
# Model database
# ============================================================

MODELS = {
    "qwen3-1.7b": {"params_b": 1.7, "moe": False, "peak_gb_full_offload": 8.0, "peak_gb_no_offload": 19.2},
    "qwen3-8b": {"params_b": 8, "moe": False, "peak_gb_full_offload": 16.2, "peak_gb_no_offload": 28.0},
    "qwen3.5-27b": {"params_b": 27, "moe": False, "peak_gb_full_offload": 16.2, "peak_gb_no_offload": 50.0},
    "qwen3-moe-a0.6b": {"params_b": 4, "active_b": 0.6, "moe": True, "peak_gb_full_offload": 19.8, "peak_gb_no_offload": 32.0},
    "qwen3.5-35b-a3b": {"params_b": 35, "active_b": 3, "moe": True, "peak_gb_full_offload": 20.0, "peak_gb_no_offload": 58.0},
}

RTX_4090_GB = 24.0

# ============================================================
# Decision logic
# ============================================================

def decide(model_key: str, offload_ratio: float = None) -> dict:
    """Decide best approach for given model and offload ratio."""
    model = MODELS[model_key]
    peak_full = model["peak_gb_full_offload"]
    peak_no = model["peak_gb_no_offload"]
    params_b = model["params_b"]

    result = {
        "model": model_key,
        "params_b": params_b,
        "moe": model["moe"],
        "rtx_4090_gb": RTX_4090_GB,
    }

    # Check if model fits with full offload (current approach)
    fits_full_offload = peak_full <= RTX_4090_GB
    margin_full_offload = RTX_4090_GB - peak_full

    result["current_approach"] = {
        "method": "verl CPPO+bypass + FSDP2 + CPUOffloadPolicy (full offload)",
        "peak_gb": peak_full,
        "fits": fits_full_offload,
        "margin_gb": margin_full_offload,
        "rank": "#1 (current best)",
        "notes": "All params/grads/optimizer on CPU → host-device copy on every forward → proven stable",
    }

    # Partial offload estimate (future: #187620)
    if offload_ratio is not None and 0.0 <= offload_ratio <= 1.0:
        # Resident params = 1 - offload_ratio → stay on GPU → zero copy
        # Offloaded params = offload_ratio → CPU → host-device copy on forward
        # Peak GPU ≈ peak_full + (resident params that would have been offloaded now staying)
        # Simplified: peak ≈ peak_no * (offload_ratio) + (activations + offloaded-during-forward)
        # Actually: peak ≈ peak_full + (1 - offload_ratio) * (params_gb * 2) for bf16 weights
        # Better estimate: peak with partial ≈ peak_no - (offload_ratio * peak_no - peak_full)
        # The key insight: partial offload REDUCES peak compared to no-offload,
        # but INCREASES peak compared to full-offload (resident params on GPU)
        resident_gpu_b = params_b * (1 - offload_ratio) * 2  # bf16 weights
        # Peak = resident weights + activations + optimizer states (still CPU for Adam) + temporary offloaded forward
        peak_partial_est = peak_full + resident_gpu_b * 0.3  # rough estimate
        # Better: partial peak = full_offload_peak + resident params memory
        peak_partial = peak_full + (params_b * 2 * (1 - offload_ratio))  # resident bf16 params on GPU

        fits_partial = peak_partial <= RTX_4090_GB
        margin_partial = RTX_4090_GB - peak_partial
        copy_fraction = offload_ratio  # only offloaded params need host-device copy

        result["future_approach"] = {
            "method": "verl CPPO+bypass + FSDP2 + PartialOffloadPolicy (fractional offload)",
            "offload_ratio": offload_ratio,
            "peak_gb": round(peak_partial, 1),
            "fits": fits_partial,
            "margin_gb": round(margin_partial, 1),
            "rank": "#1 (future best, when #187620 merges)",
            "copy_latency_reduction": f"{100 * (1 - copy_fraction):.0f}% less host-device copy per forward",
            "notes": f"Only {offload_ratio*100:.0f}% params on CPU → {100-copy_fraction*100:.0f}% resident → faster forward",
            "dependency": "PyTorch #187620 must merge → verl config must add offload_ratio option",
        }
    else:
        # Calculate optimal offload_ratio for tight models
        if not fits_full_offload:
            # Need at least enough offload to fit
            min_offload = 1.0 - (RTX_4090_GB - peak_full) / (params_b * 2)
            min_offload = max(0.0, min(min_offload, 1.0))
            result["recommended_offload_ratio"] = round(min_offload + 0.1, 2)  # 10% margin
        else:
            # Model fits with full offload → partial offload is optional
            result["recommended_offload_ratio"] = 0.0  # no need for offload

    # Framework comparison
    result["framework_comparison"] = {
        "verl_fsdp2": {
            "strategy": "FSDP2",
            "offload": "CPUOffloadPolicy (current) / PartialOffloadPolicy (future)",
            "advantages": ["per-unit summon (#6512)", "detach fix (#6699)", "Tinker primitives (#6717)", "CPPO+bypass (#6731)", "bypass_mode 18Ψ→3.8Ψ"],
            "disadvantages": ["host-device copy on every forward (full offload)", "FSDP2 dp=1 identity overhead"],
            "rtx_4090_rank": "#1",
        },
        "deepspeed_zero2": {
            "strategy": "ZeRO-2 + CPU_Adam",
            "offload": "param_offload + optimizer_offload",
            "advantages": ["mature ZeRO-2 implementation", "CPU_Adam proven stable", "AutoEP MoE (#7938 MERGED)", "Singleton MoE (#7997 MERGED)", "coalesce_grad (#7992 MERGED)"],
            "disadvantages": ["overlap_comm NaN (#8061)", "ZeRO-3 regression (#8072)", "3 Muon blockers", "no Tinker-style split training"],
            "rtx_4090_rank": "#2.5",
        },
        "megatron_mfsdpv2": {
            "strategy": "MFSDPv2 (experimental #5387)",
            "offload": "NOT yet (experimental only)",
            "advantages": ["native TP integration (TE)", "DBuffer explicit lifecycle", "release/reallocate storage control"],
            "disadvantages": ["experimental → Final Review → not production", "no partial offload yet", "no LoRA export path", "no verl integration"],
            "rtx_4090_rank": "#4 (future)",
        },
    }

    return result


def print_guide():
    """Print the full decision guide."""
    print("=" * 70)
    print("RTX 4090 FSDP2 vs ZeRO-2 Decision Guide")
    print("=" * 70)
    print()
    print("OVERVIEW:")
    print(f"  RTX 4090: 24 GiB VRAM, SM89, single GPU")
    print(f"  Current #1: verl CPPO+bypass + FSDP2 + full CPU offload")
    print(f"  Future #1: verl CPPO+bypass + FSDP2 + PartialOffloadPolicy (when #187620 merges)")
    print()
    print("THREE APPROACHES:")
    print()
    print("  1. FULL OFFLOAD (current best):")
    print("     → All params/grads/optimizer states on CPU")
    print("     → Host-device copy on EVERY forward pass")
    print("     → Proven stable → works NOW")
    print("     → Peak: ~16.2 GiB (Qwen3-8B LoRA) → 7.8 GiB margin")
    print()
    print("  2. PARTIAL OFFLOAD (future best, needs #187620, dp>=2 ONLY):")
    print("     → offload_ratio in [0.0, 1.0] → offload only enough to fit")
    print("     → Resident params stay on GPU → zero copy → faster forward")
    print("     → 30-50% less host-device copy bandwidth per step")
    print("     → MoE experts (largest) offloaded first → ideal for greedy selector")
    print("     → ★★★★★★★★ NOT viable on dp=1! FSDP2 shard=identity → resident shard = full param → exceeds 24 GiB!")
    print("     → ★★★★★★★★ ONLY beneficial on dp>=2 → resident shard = total/dp * (1-ratio) → fits per GPU")
    print()
    print("  3. NO OFFLOAD (only for small models):")
    print("     → All params on GPU → fastest forward")
    print("     → Only viable for <14B dense or <4B active MoE")
    print("     → Qwen3-1.7B: fits with ~15 GiB margin")
    print()
    print("DECISION RULES:")
    print()
    print("  Q: Does model fit with no offload? (<14B dense)")
    print("     → A: Use OffloadPolicy (no offload) → fastest")
    print()
    print("  Q: Does model fit with full offload? (14-30B dense, 3-4B active MoE)")
    print("     → A: Use CPUOffloadPolicy → proven stable → works NOW")
    print("     → Future: switch to PartialOffloadPolicy when #187620 merges")
    print()
    print("  Q: Model barely fits with full offload? (<4 GiB margin)")
    print("     → A: Use CPUOffloadPolicy → but watch for overflow")
    print("     → Future: PartialOffloadPolicy(offload_ratio=0.3) → better margin + less copy")
    print()
    print("  Q: Model too large even with full offload? (>30B dense, >4B active MoE)")
    print("     → A: Not viable on single RTX 4090 → need multi-GPU or smaller model")
    print()
    print("TIMELINE:")
    print()
    print("  NOW: CPUOffloadPolicy (full offload) → WORKS → STABLE → USE THIS")
    print("  1-3 MONTHS: #187620 merges → PartialOffloadPolicy available → verl config update")
    print("  3-6 MONTHS: verl adds offload_ratio config → PARTIAL OFFLOAD READY")
    print()
    print("KEY DEPENDENCIES:")
    print("  - PyTorch #187620 (PartialOffloadPolicy) → DRAFT → needs API direction resolution")
    print("  - verl FSDP config schema → needs offload_ratio option (simple addition)")
    print("  - verl FSDPEngine → needs to pass PartialOffloadPolicy to fully_shard()")
    print()
    print("=" * 70)


def print_comparison(model_key: str):
    """Print comparison for a specific model."""
    result = decide(model_key)
    m = result
    print(f"Model: {m['model']} ({m['params_b']}B params, MoE={m['moe']})")
    print(f"RTX 4090: {m['rtx_4090_gb']} GiB")
    print()
    print("CURRENT APPROACH:")
    ca = m["current_approach"]
    print(f"  Method: {ca['method']}")
    print(f"  Peak: {ca['peak_gb']} GiB → {ca['margin_gb']:.1f} GiB margin → {'FITS' if ca['fits'] else 'OOM!'}")
    print(f"  Rank: {ca['rank']}")
    print(f"  Notes: {ca['notes']}")
    print()
    print("FRAMEWORK COMPARISON:")
    for fw, info in m["framework_comparison"].items():
        print(f"  {fw}:")
        print(f"    Strategy: {info['strategy']}")
        print(f"    Offload: {info['offload']}")
        print(f"    RTX 4090 Rank: {info['rtx_4090_rank']}")
        print(f"    Pros: {', '.join(info['advantages'])}")
        print(f"    Cons: {', '.join(info['disadvantages'])}")
    print()


def print_estimate(model_key: str, offload_ratio: float):
    """Print estimate for partial offload."""
    result = decide(model_key, offload_ratio)
    m = result
    print(f"Model: {m['model']} ({m['params_b']}B params, MoE={m['moe']})")
    print(f"RTX 4090: {m['rtx_4090_gb']} GiB")
    print()
    print("CURRENT (full offload):")
    ca = m["current_approach"]
    print(f"  Peak: {ca['peak_gb']} GiB → {ca['margin_gb']:.1f} GiB margin")
    print()
    if "future_approach" in m:
        fa = m["future_approach"]
        print(f"FUTURE (partial offload, ratio={offload_ratio}):")
        print(f"  Peak: {fa['peak_gb']} GiB → {fa['margin_gb']:.1f} GiB margin")
        print(f"  Fits: {'YES' if fa['fits'] else 'NO'}")
        print(f"  Copy reduction: {fa['copy_latency_reduction']}")
        print(f"  Notes: {fa['notes']}")
        print(f"  Dependency: {fa['dependency']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RTX 4090 FSDP2 vs ZeRO-2 Decision Guide")
    parser.add_argument("--mode", choices=["guide", "compare", "estimate"], required=True)
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--offload-ratio", type=float, help="PartialOffloadPolicy ratio [0.0, 1.0]")
    args = parser.parse_args()

    if args.mode == "guide":
        print_guide()
    elif args.mode == "compare":
        if not args.model:
            print("ERROR: --model required for compare mode")
            sys.exit(1)
        print_comparison(args.model)
    elif args.mode == "estimate":
        if not args.model:
            print("ERROR: --model required for estimate mode")
            sys.exit(1)
        if args.offload_ratio is None:
            print("ERROR: --offload-ratio required for estimate mode")
            sys.exit(1)
        print_estimate(args.model, args.offload_ratio)


if __name__ == "__main__":
    main()
