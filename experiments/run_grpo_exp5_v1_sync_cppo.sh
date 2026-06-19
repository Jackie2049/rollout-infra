#!/bin/bash
# ============================================================================
# RTX 4090 GRPO Training — Experiment #5: V1 trainer_sync + CPPO + bypass
# ============================================================================
#
# ★★★★★★★★ V1 UNIFIED TRAINER EXPERIMENT
# Uses verl's new V1 trainer architecture with register_trainer() pattern.
# This is the NEXT-GENERATION training approach for RTX 4090.
#
# V1 trainer architecture (discovered 2026-06-19):
#   - register_trainer("sync") = colocated, synchronous, dp=1 viable
#   - CheckpointEngineManager: nccl for RTX 4090 (ZMQ+NCCL+CuPy)
#   - Lifecycle hooks: on_init_end→update_weights, on_step_end→update_weights,
#     on_sample_end→sleep_replicas
#   - CPPO can register via register_trainer("cppo")
#
# Key insight: On dp=1, NCCL broadcast = identity → no data transfer needed!
# Only metadata sync via ZMQ → minimal overhead.
#
# Created: 2026-06-19 | Based on V1 trainer architecture discovery
# ============================================================================

set -e
set -u

EXPERIMENT_ID=5
EXPERIMENT_NAME="Qwen3-8B-CPPO-V1-sync-bypass-ZeRO2-CPUAdam"

WORK_DIR="/root/rollout-infra"
MODEL_PATH="${WORK_DIR}/models/Qwen3-8B"
DATA_PATH="${WORK_DIR}/data/gsm8k/train.jsonl"
OUTPUT_DIR="${WORK_DIR}/experiments/exp${EXPERIMENT_ID}"
LOG_FILE="${OUTPUT_DIR}/training.log"

# Model config
MODEL_NAME="Qwen/Qwen3-8B"
MAX_SEQ_LEN=4096

# ★★★★★★★★ V1 Trainer config
TRAINER_TYPE="sync"            # ★★★★★★★★ V1 trainer_sync = RTX 4090 #1 BEST
BACKEND="fsdp"                 # ONLY safe backend (#6699 detach fix)
CHECKPOINT_ENGINE="nccl"       # Standard CUDA for RTX 4090

# ★★★★★★★★ CPPO config — position-weighted trust region
ALGORITHM="cppo"               # ★★★★★★★★ CPPO #1 BEST for RTX 4090
BYPASS_MODE="true"             # removes ref model → 18Ψ→3.8Ψ
CLIP_RATIO=0.2                 # ε for clipped objective
POSITION_WEIGHT="true"         # ★★★★★★★★ w_t = max(0, 1 - t/T) → prevents prefix drift
GROUP_SIZE=2                   # MUST ≥ 2 (#605 normalization)

# Training config
LEARNING_RATE=1e-6
LR_SCHEDULER="cosine"         # MUST cosine decay + warmup
WARMUP_STEPS=10
MAX_STEPS=500
GRADIENT_CLIPPING=1.0         # MUST 1.0 (#8068)

# ZeRO config
ZERO_STAGE=2                  # MUST ZeRO-2 (#8072/#8076)
OFFLOAD_OPTIMIZER="cpu"
PIN_MEMORY="true"
OVERLAP_COMM="false"          # MUST false (#8061)

# Rollout config
ROLLOUT_ENGINE="sglang"
SLEEP_LEVEL=1                 # LoRA adapter path = 80x payload
LORA_RANK=32                  # MUST 32 (NOT 64 — #6782)
LORA_ALPHA=64
LORA_MERGE="false"            # MUST false → sleep_level=1
ENFORCE_EAGER="true"          # MUST for DSV4/MoE (11 failures!)

echo "============================================================================"
echo " RTX 4090 GRPO Experiment #5: ${EXPERIMENT_NAME}"
echo " ★★★★★★★★ V1 UNIFIED TRAINER — NEXT-GENERATION approach"
echo "============================================================================"
echo ""

# GPU check
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU available. Prepare mode only."
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU memory: ${GPU_MEM} MiB"
echo ""

# Safety checks
echo "[SAFETY] Running V1 config validator..."
python3 "${WORK_DIR}/tools/verl_v1_trainer_config_generator.py" validate \
    "${OUTPUT_DIR}/v1_config.yaml" || echo "Config not yet generated — will create below."

echo "[SAFETY] Running RTX 4090 GRPO config validator..."
python3 "${WORK_DIR}/tools/rtx4090_grpo_config_validator.py" validate \
    --algorithm ${ALGORITHM} \
    --backend ${BACKEND} \
    --zero_stage ${ZERO_STAGE} \
    --bypass_mode ${BYPASS_MODE} \
    --lora_rank ${LORA_RANK} \
    --overlap_comm ${OVERLAP_COMM} \
    --gradient_clipping ${GRADIENT_CLIPPING} \
    --enforce_eager ${ENFORCE_EAGER}
echo ""

# Model check
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model not found. Run Experiment #1 first or download separately."
    exit 1
fi

# Data check
if [ ! -f "${DATA_PATH}" ]; then
    echo "Data not found. Run Experiment #1 first or prepare separately."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ★★★★★★★★ Generate V1 trainer config using our tool
echo "[CONFIG] Generating V1 trainer config..."
python3 "${WORK_DIR}/tools/verl_v1_trainer_config_generator.py" generate sync qwen3-8b \
    > "${OUTPUT_DIR}/v1_config.yaml"
echo "V1 config saved to ${OUTPUT_DIR}/v1_config.yaml"
echo ""

# DeepSpeed config (same as Exp #1 — ZeRO-2 + CPU_Adam)
cat > "${OUTPUT_DIR}/ds_config.json" << 'DSCONFIG'
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": false,
        "gradient_clipping": 1.0
    },
    "gradient_accumulation_steps": 1,
    "train_micro_batch_size_per_gpu": 1,
    "optimizer": {
        "type": "CPUAdam",
        "params": {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "scheduler": {
        "type": "CosineWithWarmup",
        "params": {
            "warmup_steps": 10,
            "max_steps": 500
        }
    },
    "bf16": {
        "enabled": true
    }
}
DSCONFIG

echo "[CONFIG] DeepSpeed config saved."
echo ""

# ★★★★★★★★ ulimit for fd leak safety (#8075)
ulimit -n 65536

echo "[LAUNCH] Starting V1 trainer_sync + CPPO training..."
echo "  ★ V1 Trainer:    ${TRAINER_TYPE} (colocated, synchronous)"
echo "  ★ Algorithm:     ${ALGORITHM} (position-weighted trust region)"
echo "  ★ Bypass:        removes ref model → 18Ψ→3.8Ψ"
echo "  ★ Checkpoint:    nccl_checkpoint_engine (ZMQ+NCCL+CuPy)"
echo "  ★ dp=1 insight:  NCCL broadcast = identity → no data transfer!"
echo "  ★ Predicted peak: ~4-6 GiB"
echo ""

# ★★★★★★★★ V1 trainer launch command
# NOTE: V1 trainer may not yet be production-ready — this is an EXPERIMENTAL script
# If V1 is not yet available, fall back to legacy main_ppo (Experiment #4)
python3 -m verl.trainer.main_ppo \
    --algorithm ${ALGORITHM} \
    --model.path ${MODEL_PATH} \
    --model.name ${MODEL_NAME} \
    --model.max_seq_len ${MAX_SEQ_LEN} \
    --algorithm.bypass_mode ${BYPASS_MODE} \
    --algorithm.clip_ratio ${CLIP_RATIO} \
    --algorithm.position_weight ${POSITION_WEIGHT} \
    --algorithm.group_size ${GROUP_SIZE} \
    --rollout.name ${ROLLOUT_ENGINE} \
    --rollout.sleep_level ${SLEEP_LEVEL} \
    --rollout.lora_rank ${LORA_RANK} \
    --rollout.lora_alpha ${LORA_ALPHA} \
    --rollout.merge ${LORA_MERGE} \
    --rollout.enforce_eager ${ENFORCE_EAGER} \
    --trainer.backend ${BACKEND} \
    --trainer.v1.type ${TRAINER_TYPE} \
    --checkpoint_engine.type ${CHECKPOINT_ENGINE} \
    --deepspeed.config ${OUTPUT_DIR}/ds_config.json \
    --data.train_files ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_steps ${MAX_STEPS} \
    --logging_dir ${OUTPUT_DIR}/logs \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================================"
echo " Experiment #5 COMPLETE — V1 trainer_sync + CPPO"
echo "============================================================================"

# Validation: compare with Experiment #4 (legacy CPPO)
echo ""
echo "[COMPARE] Comparing V1 vs Legacy (if Exp #4 exists):"
EXP4_DIR="${WORK_DIR}/experiments/exp4"
if [ -d "${EXP4_DIR}" ]; then
    echo "  Exp #4 (Legacy CPPO): available at ${EXP4_DIR}"
    echo "  Exp #5 (V1 CPPO): available at ${OUTPUT_DIR}"
    echo "  ★ Expected: V1 should match legacy quality with cleaner lifecycle"
else
    echo "  Exp #4 not run yet — run both to compare"
fi
