#!/usr/bin/env python3
"""RTX 4090 vLLM Budget Profiler — Collect profile_table.csv for BudgetRefiner SLO

This script profiles vLLM inference performance on RTX 4090 (SM89) across
different model, batch size, and sequence length configurations. The collected
data will be used as RTX 4090 entries in the BudgetRefiner profile_table.csv,
which is our UNIQUE contribution to the vLLM BudgetRefiner SLO upstream PR.

Usage:
  python3 tools/profile_vllm_budget.py --mode collect --models Qwen3-1.7B Qwen3-8B
  python3 tools/profile_vllm_budget.py --mode collect --quantization gptq
  python3 tools/profile_vllm_budget.py --mode validate --csv profile_table_rtx4090.csv
  python3 tools/profile_vllm_budget.py --mode estimate --model Qwen3-8B --batch 8 --seq-len 4096
  python3 tools/profile_vllm_budget.py --mode all

Reference:
  - notebook/projects/budgetrefiner-vllm-pr-draft.md (PR draft)
  - notebook/projects/budgetrefiner-vllm-contribution-plan.md (4-phase plan)
  - MindIE/vLLM-Ascend BudgetRefiner source (95%+ GPU-generic)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# Profile Table Schema — Matches vLLM-Ascend BudgetRefiner
# ============================================================

PROFILE_TABLE_COLUMNS = [
    "model",           # Model name (e.g., "Qwen3-1.7B")
    "quantization",    # Quantization method (BF16, GPTQ-Int4, AWQ)
    "num_layers",      # Number of transformer layers
    "hidden_dim",      # Hidden dimension size
    "num_heads",       # Number of attention heads
    "head_dim",        # Dimension per head
    "seq_len",         # Maximum sequence length
    "batch_size",      # Number of concurrent sequences
    "chunk_size",      # Number of tokens per chunk
    "prefill_time_ms", # Time for prefill (ms)
    "decode_time_ms",  # Time per decode step (ms)
    "total_time_ms",   # Total time per iteration (ms)
    "gpu",             # GPU name (e.g., "RTX 4090")
    "sm_version",      # SM version (e.g., "89")
    "kv_cache_type",   # KV cache dtype (FP16, INT8)
    "gpu_memory_gb",   # Total GPU memory (GB)
    "gpu_memory_used_gb", # GPU memory used (GB)
]

# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "Qwen3-1.7B": {
        "num_layers": 28,
        "hidden_dim": 2048,
        "num_heads": 16,
        "head_dim": 128,
        "hf_name": "Qwen/Qwen3-1.7B",
        "quantization_options": ["BF16", "GPTQ-Int4"],
    },
    "Qwen3-8B": {
        "num_layers": 36,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "hf_name": "Qwen/Qwen3-8B",
        "quantization_options": ["BF16", "GPTQ-Int4", "AWQ"],
    },
    "Llama-3.1-8B": {
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "hf_name": "meta-llama/Llama-3.1-8B",
        "quantization_options": ["BF16", "GPTQ-Int4"],
    },
    "Mistral-7B": {
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "quantization_options": ["BF16", "GPTQ-Int4"],
    },
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
SEQ_LENGTHS = [256, 512, 1024, 2048, 4096]
KV_CACHE_TYPES = ["FP16", "INT8"]

OUTPUT_DIR = Path("results/profile_tables")
CSV_FILENAME = "profile_table_rtx4090.csv"


# ============================================================
# Mode: collect — Run vLLM profiling on GPU
# ============================================================

def run_collect(args):
    """Collect profile data by running vLLM benchmarks on RTX 4090."""
    print("=" * 80)
    print("RTX 4090 vLLM Budget Profiler — Data Collection")
    print("=" * 80)
    print("This mode requires a running RTX 4090 GPU with vLLM installed.")
    print()

    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPU available. Cannot collect profile data.")
            print("Run this script on a GPU server when available.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        sm_ver = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        print(f"GPU: {gpu_name}, VRAM: {gpu_mem:.1f}GB, SM: {sm_ver}")
        if sm_ver != 89:
            print(f"WARNING: SM version is {sm_ver}, not 89 (RTX 4090). Data may differ.")
    except ImportError:
        print("ERROR: PyTorch not installed. Cannot check GPU.")
        sys.exit(1)

    # Select models to profile
    models = args.models or list(MODEL_CONFIGS.keys())
    quants = args.quantization or ["BF16"]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / CSV_FILENAME

    print(f"\nWill profile {len(models)} models × {len(quants)} quantizations × "
          f"{len(BATCH_SIZES)} batches × {len(SEQ_LENGTHS)} seq_lens")
    print(f"Output: {csv_path}")
    print()

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_TABLE_COLUMNS)
        writer.writeheader()

    # Profile each configuration
    rows = []
    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"WARNING: Unknown model {model_name}, skipping")
            continue
        cfg = MODEL_CONFIGS[model_name]
        for quant in quants:
            if quant not in cfg["quantization_options"]:
                print(f"WARNING: {quant} not available for {model_name}, skipping")
                continue
            for batch in BATCH_SIZES:
                for seq_len in SEQ_LENGTHS:
                    print(f"Profiling: {model_name} {quant} batch={batch} seq={seq_len}")

                    # Construct vLLM model name based on quantization
                    if quant == "BF16":
                        hf_model = cfg["hf_name"]
                    elif quant == "GPTQ-Int4":
                        hf_model = f"{cfg['hf_name']}-GPTQ-Int4"
                    elif quant == "AWQ":
                        hf_model = f"{cfg['hf_name']}-AWQ"
                    else:
                        hf_model = cfg["hf_name"]

                    # Run profiling benchmark
                    # This requires vLLM to be installed and GPU available
                    result = profile_vllm_config(
                        hf_model=hf_model,
                        model_name=model_name,
                        quantization=quant,
                        batch_size=batch,
                        seq_len=seq_len,
                        num_layers=cfg["num_layers"],
                        hidden_dim=cfg["hidden_dim"],
                        num_heads=cfg["num_heads"],
                        head_dim=cfg["head_dim"],
                        gpu_name=gpu_name,
                        sm_ver=sm_ver,
                        gpu_mem=gpu_mem,
                        kv_type=args.kv_cache or "FP16",
                    )
                    if result:
                        rows.append(result)
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=PROFILE_TABLE_COLUMNS)
                            writer.writerow(result)

    print(f"\nCollected {len(rows)} profile entries → {csv_path}")
    return rows


def profile_vllm_config(hf_model, model_name, quantization, batch_size, seq_len,
                        num_layers, hidden_dim, num_heads, head_dim,
                        gpu_name, sm_ver, gpu_mem, kv_type):
    """Profile a single vLLM configuration. Returns dict or None."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM not installed. Cannot profile.")
        return None

    # Configure vLLM parameters
    kv_cache_dtype = "int8" if kv_type == "INT8" else None
    max_model_len = seq_len
    gpu_memory_utilization = 0.85

    try:
        llm = LLM(
            model=hf_model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            dtype="bfloat16",
            enforce_eager=True,  # Disable CUDA graphs for accurate timing
        )

        # Generate prompts for batch
        prompts = ["Hello, my name is"] * batch_size
        sampling_params = SamplingParams(
            max_tokens=10,  # Short generation for profiling
            temperature=0.0,  # Deterministic
        )

        # Measure prefill time
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        total_time = time.perf_counter() - start

        # Measure GPU memory used
        import torch
        mem_used = torch.cuda.max_memory_allocated() / 1e9

        # Estimate decode time per step
        # decode_time ≈ total_time / (num_decode_tokens * batch_size)
        num_decode_tokens = 10
        decode_time_per_step = total_time * 1000 / num_decode_tokens

        # Estimate prefill time
        # prefill_time ≈ total_time - decode_time * num_decode_tokens
        prefill_time = total_time * 1000 - decode_time_per_step * num_decode_tokens

        result = {
            "model": model_name,
            "quantization": quantization,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "chunk_size": seq_len,  # For prefill, chunk = full sequence
            "prefill_time_ms": round(prefill_time, 2),
            "decode_time_ms": round(decode_time_per_step, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "gpu": gpu_name,
            "sm_version": str(sm_ver),
            "kv_cache_type": kv_type,
            "gpu_memory_gb": round(gpu_mem, 1),
            "gpu_memory_used_gb": round(mem_used, 2),
        }
        return result
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ============================================================
# Mode: estimate — Estimate budget from existing profile data
# ============================================================

def run_estimate(args):
    """Estimate token budget for a given configuration using quadratic predictor."""
    print("=" * 80)
    print("RTX 4090 Budget Estimator (offline)")
    print("=" * 80)

    # Load profile table if available
    csv_path = OUTPUT_DIR / CSV_FILENAME
    if not csv_path.exists():
        print(f"No profile data found at {csv_path}")
        print("Run --mode collect first when GPU is available.")
        # Use heuristic estimates instead
        return estimate_from_heuristics(args)

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find matching entries
    model_name = args.models[0] if args.models else "Qwen3-8B"
    batch = args.batch or 8
    seq_len = args.seq_len or 4096
    slo_ms = args.slo or 100.0

    matching = [r for r in rows if r["model"] == model_name
                and int(r["batch_size"]) <= batch
                and int(r["seq_len"]) <= seq_len]

    if not matching:
        print(f"No matching entries for {model_name} batch={batch} seq={seq_len}")
        return estimate_from_heuristics(args)

    # Use closest match
    best = matching[-1]  # Largest batch that fits
    decode_ms = float(best["decode_time_ms"])
    total_ms = float(best["total_time_ms"])

    # Budget estimate: max tokens = SLO_time / per-token_time
    slo_budget = int(slo_ms / decode_ms * batch)
    print(f"\nBudget Estimate for {model_name}:")
    print(f"  Batch size: {batch}")
    print(f"  Seq length: {seq_len}")
    print(f"  SLO target: {slo_ms}ms per decode step")
    print(f"  Decode time: {decode_ms}ms per step")
    print(f"  Max token budget: {slo_budget} tokens")
    print(f"  Max sequences: {min(slo_budget // seq_len, batch)}")


def estimate_from_heuristics(args):
    """Estimate budget using heuristics when no profile data available."""
    model_name = args.models[0] if args.models else "Qwen3-8B"
    batch = args.batch or 8
    seq_len = args.seq_len or 4096
    slo_ms = args.slo or 100.0

    # Heuristic decode time estimates (ms per decode step per sequence, RTX 4090)
    heuristic_decode_ms_per_seq = {
        "Qwen3-1.7B": 1.0,   # ~1ms per decode step per seq on RTX 4090
        "Qwen3-8B": 5.0,     # ~5ms per decode step per seq on RTX 4090
        "Llama-3.1-8B": 5.0,
        "Mistral-7B": 4.5,
    }

    decode_ms_per_seq = heuristic_decode_ms_per_seq.get(model_name, 5.0)
    max_seqs = min(int(slo_ms / decode_ms_per_seq), batch)
    max_tokens = max_seqs * seq_len

    print(f"\nHeuristic Budget Estimate for {model_name}:")
    print(f"  Batch size: {batch}")
    print(f"  Seq length: {seq_len}")
    print(f"  SLO target: {slo_ms}ms per decode step")
    print(f"  Estimated decode time: ~{decode_ms_per_seq}ms per step per seq (heuristic)")
    print(f"  Max token budget: {max_tokens} tokens (heuristic)")
    print(f"  Max sequences: {max_seqs} (heuristic)")
    print(f"\n  NOTE: These are rough estimates. Collect actual profile data on GPU for accuracy.")

    # Also fix run_estimate to use model_name
    model_name = args.models[0] if args.models else "Qwen3-8B"


# ============================================================
# Mode: validate — Validate existing profile table
# ============================================================

def run_validate(args):
    """Validate a profile_table.csv file."""
    csv_path = Path(args.csv) if args.csv else OUTPUT_DIR / CSV_FILENAME
    print("=" * 80)
    print(f"Profile Table Validation: {csv_path}")
    print("=" * 80)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Entries: {len(rows)}")
    print(f"Models: {set(r['model'] for r in rows)}")
    print(f"Quantizations: {set(r['quantization'] for r in rows)}")
    print(f"Batch sizes: {sorted(set(int(r['batch_size']) for r in rows))}")
    print(f"Seq lengths: {sorted(set(int(r['seq_len']) for r in rows))}")
    print(f"GPU: {set(r['gpu'] for r in rows)}")

    # Validate required columns
    missing_cols = set(PROFILE_TABLE_COLUMNS) - set(rows[0].keys()) if rows else set(PROFILE_TABLE_COLUMNS)
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    else:
        print("All required columns present ✓")

    # Check for SM89 data
    sm89_rows = [r for r in rows if r.get("sm_version") == "89"]
    print(f"\nSM89 entries: {len(sm89_rows)} (RTX 4090 specific)")
    if len(sm89_rows) == 0:
        print("  WARNING: No SM89 data — collect on RTX 4090 GPU!")


# ============================================================
# Mode: all — Summary of all modes
# ============================================================

def run_all(args):
    """Run validation + estimate modes."""
    run_validate(args)
    print()
    run_estimate(args)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RTX 4090 vLLM Budget Profiler — Collect BudgetRefiner SLO profile data")
    parser.add_argument("--mode",
                        choices=["collect", "validate", "estimate", "all"],
                        default="estimate",
                        help="Mode: collect (GPU needed), validate, estimate (offline), all")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to profile (default: all)")
    parser.add_argument("--quantization", nargs="+", default=["BF16"],
                        help="Quantization methods to profile")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for estimate mode")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length for estimate mode")
    parser.add_argument("--slo", type=float, default=100.0,
                        help="SLO target latency (ms) for estimate mode")
    parser.add_argument("--kv-cache", choices=["FP16", "INT8"], default="FP16",
                        help="KV cache type")
    parser.add_argument("--csv", default=None,
                        help="CSV file path for validate mode")
    args = parser.parse_args()

    modes = {
        "collect": run_collect,
        "validate": run_validate,
        "estimate": run_estimate,
        "all": run_all,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
