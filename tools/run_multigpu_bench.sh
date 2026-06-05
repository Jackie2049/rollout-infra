#!/bin/bash
# Run multi-GPU DDP benchmarks with different GPU counts
# Usage: bash run_multigpu_bench.sh

set -e
cd ~/rollout-infra

source ~/anaconda3/bin/activate llm

echo "========================================"
echo "Multi-GPU DDP Benchmark Runner"
echo "========================================"
date

# Find free GPUs (utilization < 10%)
FREE_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | \
    awk -F', ' '$2+0 < 10 {print $1}' | tr '\n' ',' | sed 's/,$//')
echo "Free GPUs: $FREE_GPUS"
N_FREE=$(echo $FREE_GPUS | tr ',' '\n' | wc -l | tr -d ' ')
echo "Number of free GPUs: $N_FREE"

RESULTS_FILE="multigpu_all_results.json"
echo "{}" > $RESULTS_FILE

# Test with different GPU counts
for N in 1 2 4; do
    if [ $N -gt $N_FREE ]; then
        echo "Skipping $N GPUs (only $N_FREE free)"
        continue
    fi

    # Select first N free GPUs
    GPUS=$(echo $FREE_GPUS | cut -d',' -f1-$N | tr ',' ' ')
    GPU_LIST=$(echo $GPUS | tr ' ' ',')

    echo ""
    echo "========================================"
    echo "Running with $N GPU(s): $GPU_LIST"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --nproc_per_node=$N \
        --master_port=29500 \
        tools/multigpu_ddp_benchmark.py 2>&1 | tee /tmp/ddp_${N}gpu.log

    # Extract results
    if [ -f multigpu_results.json ]; then
        # Merge into combined results
        python3 -c "
import json
with open('$RESULTS_FILE') as f: all_r = json.load(f)
with open('multigpu_results.json') as f: r = json.load(f)
all_r[f'{N}gpu'] = r
with open('$RESULTS_FILE', 'w') as f: json.dump(all_r, f, indent=2)
"
        echo "Results for $N GPU(s) saved."
    fi
done

echo ""
echo "========================================"
echo "All benchmarks complete!"
echo "Results: $RESULTS_FILE"
echo "========================================"
date
