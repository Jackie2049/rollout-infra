#!/bin/bash
# RTX 4090 GRPO Training Recipe — rLLM Tinker (Recommended #1)
#
# This script provides a practical GRPO training configuration for RTX 4090
# using rLLM Tinker backend (our #1 recommended framework for RTX 4090).
#
# Key advantages:
#   - In-process (no Ray overhead) → Zero-copy weight sync
#   - Auto LoRA rank=32 (train_mlp+attn+unembed)
#   - bypass_mode=True DEFAULT → no ref model → save 14GB → KL=0
#   - LossFnType={ppo, IS, cispo, dro} → multiple algorithm options
#
# Reference: notebook/projects/rllm-tinker-training-loop-source-reading.md
#            tools/rtx4090_grpo_config_matrix.py
#
# Usage:
#   bash tools/train_tinker_rtx4090.sh --model Qwen3-1.7B --task math
#   bash tools/train_tinker_rtx4090.sh --model Qwen3-8B --task math
#   bash tools/train_tinker_rtx4090.sh --mode dry-run  # Just print config
#

set -e

# ============================================================
# Default Configuration — RTX 4090 Optimized
# ============================================================

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
LORA_RANK="${LORA_RANK:-32}"
GROUP_SIZE="${GROUP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
BYPASS_MODE="${BYPASS_MODE:-true}"  # DEFAULT true for Tinker
TASK="${TASK:-math}"
NUM_STEPS="${NUM_STEPS:-100}"
SEED="${SEED:-42}"
MODE="${MODE:-run}"  # run or dry-run

# ============================================================
# Memory Budget Estimation
# ============================================================

echo "============================================================"
echo "RTX 4090 GRPO Training — rLLM Tinker Configuration"
echo "============================================================"
echo ""
echo "Model:          $MODEL"
echo "LoRA rank:      $LORA_RANK"
echo "Group size:     $GROUP_SIZE (GRPO: same task × group_size)"
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "LR schedule:    $LR_SCHEDULE"
echo "Warmup ratio:   $WARMUP_RATIO"
echo "Bypass mode:    $BYPASS_MODE (no ref model, KL=0)"
echo "Task:           $TASK"
echo "Num steps:      $NUM_STEPS"
echo "Seed:           $SEED"
echo ""

# Memory estimation
if [[ "$MODEL" == *"1.7B"* ]]; then
    EST_MEM="9.2"
    MODEL_SIZE="1.7B"
    MAX_SEQ="4096"
elif [[ "$MODEL" == *"8B"* ]]; then
    EST_MEM="24.8"
    MODEL_SIZE="8B"
    MAX_SEQ="2048"  # Reduced for 24GB VRAM
else
    EST_MEM="14.0"
    MODEL_SIZE="7B"
    MAX_SEQ="2048"
fi

echo "Estimated memory: ~${EST_MEM}GB (RTX 4090 has 24GB VRAM)"
echo "Max seq length:   ${MAX_SEQ}"
echo ""

if [[ "$EST_MEM" > "23" ]]; then
    echo "WARNING: Estimated memory ${EST_MEM}GB may exceed 24GB VRAM!"
    echo "  Consider using INT4 quantized model or smaller model."
    echo "  Recommended: Qwen3-1.7B for 24GB RTX 4090 GRPO training."
fi

# ============================================================
# Dry-Run Mode — Just print config and exit
# ============================================================

if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry-run mode: No training will be executed."
    echo ""
    echo "To run actual training, execute:"
    echo "  rllm train --config config/rllm/backend/tinker.yaml"
    echo ""
    echo "Key Tinker-specific settings:"
    echo "  model.lora_rank = $LORA_RANK"
    echo "  model.train_mlp = true"
    echo "  model.train_attn = true"
    echo "  model.train_unembed = true"
    echo "  rllm.algorithm.rollout_correction.bypass_mode = $BYPASS_MODE"
    echo "  rllm.algorithm.grpo.group_size = $GROUP_SIZE"
    echo ""
    echo "Memory breakdown (Qwen3-${MODEL_SIZE}):"
    echo "  Base weights (frozen):   ~${MODEL_SIZE}GB"
    echo "  LoRA params (rank=$LORA_RANK):  ~0.6GB"
    echo "  Optimizer (Adam m+v):    ~1.2GB"
    echo "  Activations (per step):  ~2.0GB"
    echo "  Ref model:               0GB (bypass=true)"
    echo "  CUDA context:            ~1.0GB"
    echo "  Total estimated:         ~${EST_MEM}GB"
    echo ""
    echo "Notes:"
    echo "  - bypass_mode=True: no ref model needed → save ~${MODEL_SIZE}GB"
    echo "  - LoRA rank=32: only train ~0.6GB vs full ~${MODEL_SIZE}GB"
    echo "  - In-process: no Ray → no IPC overhead → zero-copy weight sync"
    echo "  - LossFnType: ppo (default), importance_sampling, cispo, dro"
    exit 0
fi

# ============================================================
# Prerequisites Check
# ============================================================

echo "Checking prerequisites..."

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install conda first."
    exit 1
fi

# Check rLLM
if ! command -v rllm &> /dev/null; then
    echo "ERROR: rllm not found. Install rLLM:"
    echo "  pip install rllm  (or from source: github.com/rllm-org/rllm)"
    exit 1
fi

# Check GPU
if ! python3 -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
    echo "ERROR: No CUDA GPU available. This script requires RTX 4090."
    echo "  Run on a GPU server when available."
    echo "  Use --mode dry-run to just print configuration."
    exit 1
fi

# ============================================================
# Training Execution
# ============================================================

echo ""
echo "Starting rLLM Tinker GRPO training on RTX 4090..."
echo ""

# Create config directory if needed
CONFIG_DIR="config/rllm/rtx4090"
mkdir -p "$CONFIG_DIR"

# Write Tinker config for RTX 4090
cat > "$CONFIG_DIR/tinker_grpo.yaml" << EOF
# RTX 4090 GRPO Training Config — rLLM Tinker
model:
  name: "$MODEL"
  lora_rank: $LORA_RANK
  train_unembed: true
  train_attn: true
  train_mlp: true

rllm:
  algorithm:
    type: grpo
    rollout_correction:
      bypass_mode: $BYPASS_MODE  # No ref model → save ~14GB → KL=0
    grpo:
      group_size: $GROUP_SIZE

  training:
    batch_size: $BATCH_SIZE
    learning_rate: $LEARNING_RATE
    lr_schedule: $LR_SCHEDULE
    warmup_steps_ratio: $WARMUP_RATIO
    num_steps: $NUM_STEPS
    seed: $SEED

  backend: tinker  # In-process, no Ray
EOF

echo "Config written to: $CONFIG_DIR/tinker_grpo.yaml"
echo ""
echo "Executing:"
echo "  rllm train --config $CONFIG_DIR/tinker_grpo.yaml"
echo ""

# Execute training
rllm train --config "$CONFIG_DIR/tinker_grpo.yaml"

echo ""
echo "Training complete!"
echo "Check results in: results/rllm/tinker_grpo/"
