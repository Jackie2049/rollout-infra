#!/bin/bash
# RTX 4090 Cross-Framework GRPO Training Runner
# ==============================================
# Unified script to run GRPO training across rLLM Tinker, verl, and DeepSpeed
# with RTX 4090-optimized configurations.
#
# Usage:
#   bash tools/run_grpo_rtx4090.sh --framework rllm --model qwen3-1.7b --dry-run
#   bash tools/run_grpo_rtx4090.sh --framework verl --model qwen3-1.7b --dry-run
#   bash tools/run_grpo_rtx4090.sh --framework deepspeed --model qwen3-moe --dry-run
#   bash tools/run_grpo_rtx4090.sh --framework deepspeed --mode opd --dry-run
#
# Based on: notebook/fundamentals/ (7 framework readings + RTX 4090 references)

set -e

# === Defaults ===
FRAMEWORK="rllm"
MODEL="qwen3-1.7b"
LORA_RANK=32
GROUP_SIZE=4
BATCH_SIZE=8
BYPASS_MODE=true
LEARNING_RATE=2e-5
MAX_STEPS=100
DRY_RUN=false
DEEPSPEED_MODE="grpo"  # grpo | moe | opd

# === Parse Args ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --framework) FRAMEWORK="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --lora-rank) LORA_RANK="$2"; shift 2;;
        --group-size) GROUP_SIZE="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --bypass) BYPASS_MODE="$2"; shift 2;;
        --lr) LEARNING_RATE="$2"; shift 2;;
        --max-steps) MAX_STEPS="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        --mode) DEEPSPEED_MODE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# === Memory Estimation ===
estimate_memory() {
    local model=$1
    local lora=$2
    local bypass=$3
    local framework=$4

    case $model in
        qwen3-1.7b) BASE_GB=3.4;;
        qwen3-8b) BASE_GB=16.0;;
        qwen3-moe) BASE_GB=16.0;;  # total params
        *) BASE_GB=16.0;;
    esac

    # LoRA params
    if [[ $lora -gt 0 ]]; then
        LORA_GB=$(echo "$BASE_GB / 16 * 0.6" | bc -l)
    else
        LORA_GB=0
    fi

    # Ref model
    if [[ "$bypass" == "true" ]]; then
        REF_GB=0
    else
        REF_GB=$BASE_GB
    fi

    # Framework overhead
    case $framework in
        rllm) OVERHEAD_GB=0;;
        verl) OVERHEAD_GB=1.0;;
        deepspeed) OVERHEAD_GB=0.5;;
        *) OVERHEAD_GB=0.5;;
    esac

    # Optimizer (GPU vs CPU)
    case $framework in
        deepspeed)
            OPT_GB=0  # CPU offload
            ;;
        *)
            if [[ $lora -gt 0 ]]; then
                OPT_GB=$(echo "$LORA_GB * 2" | bc -l)
            else
                OPT_GB=$(echo "$BASE_GB * 2" | bc -l)
            fi
            ;;
    esac

    # Activations
    if [[ $lora -gt 0 ]]; then
        ACT_GB=$(echo "$BASE_GB / 16 * 2" | bc -l)
    else
        ACT_GB=$(echo "$BASE_GB / 16 * 8" | bc -l)
    fi

    # KV cache (training only, not serving)
    KV_GB=2.0

    # CUDA context
    CUDA_GB=1.0

    TOTAL=$(echo "$BASE_GB + $LORA_GB + $REF_GB + $OPT_GB + $ACT_GB + $KV_GB + $OVERHEAD_GB + $CUDA_GB" | bc -l)

    echo "Framework: $framework | Model: $model | LoRA: rank=$lora | Bypass: $bypass"
    echo "  Base weights: ${BASE_GB}GB"
    echo "  LoRA params: ${LORA_GB}GB"
    echo "  Ref model: ${REF_GB}GB"
    echo "  Optimizer: ${OPT_GB}GB"
    echo "  Activations: ${ACT_GB}GB"
    echo "  KV cache: ${KV_GB}GB"
    echo "  Overhead: ${OVERHEAD_GB}GB"
    echo "  CUDA: ${CUDA_GB}GB"
    echo "  TOTAL: ${TOTAL}GB / 24GB"
    if [[ $(echo "$TOTAL > 24" | bc -l) -eq 1 ]]; then
        echo "  ✗ EXCEEDS 24GB! NOT VIABLE!"
    else
        HEADROOM=$(echo "24 - $TOTAL" | bc -l)
        echo "  ✓ FITS 24GB! Headroom: ${HEADROOM}GB"
    fi
}

# === Run Framework ===
run_rllm() {
    echo "=== rLLM Tinker GRPO ==="
    echo "Model: $MODEL | LoRA rank: $LORA_RANK | group_size: $GROUP_SIZE | bypass: $BYPASS_MODE"

    estimate_memory "$MODEL" "$LORA_RANK" "$BYPASS_MODE" "rllm"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute:"
        echo "  rllm train --config tinker.yaml"
        echo "  --model.name Qwen/${MODEL}"
        echo "  --model.lora_rank ${LORA_RANK}"
        echo "  --bypass_mode ${BYPASS_MODE}"
        echo "  --group_size ${GROUP_SIZE}"
        echo "  --batch_size ${BATCH_SIZE}"
        echo "  --learning_rate ${LEARNING_RATE}"
        echo "  --max_steps ${MAX_STEPS}"
        return
    fi

    # Check if rllm is installed
    if ! command -v rllm &> /dev/null; then
        echo "ERROR: rllm not installed. Run: pip install rllm"
        exit 1
    fi

    echo "[GPU needed] Running rLLM Tinker GRPO training..."
    # Actual training command (needs GPU)
    rllm train --config tinker.yaml \
        --model.name "Qwen/${MODEL}" \
        --model.lora_rank "$LORA_RANK" \
        --bypass_mode "$BYPASS_MODE" \
        --group_size "$GROUP_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --max_steps "$MAX_STEPS"
}

run_verl() {
    echo "=== verl GRPO ==="
    echo "Model: $MODEL | LoRA rank: $LORA_RANK | group_size: $GROUP_SIZE | bypass: $BYPASS_MODE"

    estimate_memory "$MODEL" "$LORA_RANK" "$BYPASS_MODE" "verl"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute:"
        echo "  python -m verl.trainer.main_ppo"
        echo "  --algorithm grpo"
        echo "  --bypass_mode ${BYPASS_MODE}"
        echo "  --detach_metrics_per_micro_batch True"
        echo "  --model.path Qwen/${MODEL}"
        echo "  --group_size ${GROUP_SIZE}"
        echo "  --batch_size ${BATCH_SIZE}"
        echo "  --max_steps ${MAX_STEPS}"
        echo "  --envs VLLM_USE_V2_MODEL_RUNNER=0  # conservative fallback"
        echo "  --envs ENFORCE_EAGER=1  # batch invariance on SM89"
        return
    fi

    echo "[GPU needed] Running verl GRPO training..."
    export VLLM_USE_V2_MODEL_RUNNER=0  # conservative fallback for MRv2
    python -m verl.trainer.main_ppo \
        --algorithm grpo \
        --bypass_mode "$BYPASS_MODE" \
        --detach_metrics_per_micro_batch True \
        --model.path "Qwen/${MODEL}" \
        --group_size "$GROUP_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --max_steps "$MAX_STEPS"
}

run_deepspeed() {
    local mode=$DEEPSPEED_MODE

    case $mode in
        grpo)
            echo "=== DeepSpeed ZeRO-2 GRPO ==="
            estimate_memory "$MODEL" "$LORA_RANK" "$BYPASS_MODE" "deepspeed"
            ;;
        moe)
            echo "=== DeepSpeed AutoEP MoE ==="
            MODEL="qwen3-moe"
            estimate_memory "$MODEL" "$LORA_RANK" "$BYPASS_MODE" "deepspeed"
            ;;
        opd)
            echo "=== DeepSpeed OPD Distillation ==="
            echo "Student: Qwen2.5-0.5B (~1GB GPU) | Teacher: Qwen2.5-1.5B (CPU logits)"
            echo "Total GPU: ~4.6GB → incredibly light!"
            ;;
        *)
            echo "Unknown DeepSpeed mode: $mode"
            exit 1
            ;;
    esac

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute:"
        case $mode in
            grpo)
                echo "  deepspeed train_grpo.py --deepspeed_config ds_zero2_config.json"
                echo "  --zero_stage 2 --offload_optimizer cpu"
                echo "  --lora_rank ${LORA_RANK} --bypass_mode ${BYPASS_MODE}"
                ;;
            moe)
                echo "  deepspeed train_autoep_moe.py --deepspeed_config ds_autoep_config.json"
                echo "  --zero_stage 2 --offload_optimizer cpu"
                echo "  --autoep_enable True --expert_parallel_size 1"
                echo "  --lora_rank ${LORA_RANK} --offload_ratio 0.5"
                echo "  --use_zenflow True  # chunked copyback, no GPU spike"
                ;;
            opd)
                echo "  deepspeed opd_train.py"
                echo "  --student_model Qwen2.5-0.5B-Instruct"
                echo "  --teacher_model Qwen2.5-1.5B-Instruct"
                echo "  --divergence_type forward_kl"
                echo "  --zero_stage 0  # no partitioning, student-only on GPU"
                ;;
        esac
        return
    fi

    echo "[GPU needed] Running DeepSpeed training..."
    case $mode in
        grpo)
            deepspeed train_grpo.py \
                --deepspeed_config ds_zero2_config.json \
                --zero_stage 2 --offload_optimizer cpu \
                --lora_rank "$LORA_RANK" --bypass_mode "$BYPASS_MODE"
            ;;
        moe)
            deepspeed train_autoep_moe.py \
                --deepspeed_config ds_autoep_config.json \
                --zero_stage 2 --offload_optimizer cpu \
                --autoep_enable True --expert_parallel_size 1 \
                --lora_rank "$LORA_RANK" --offload_ratio 0.5
            ;;
        opd)
            deepspeed opd_train.py \
                --student_model Qwen2.5-0.5B-Instruct \
                --teacher_model Qwen2.5-1.5B-Instruct \
                --divergence_type forward_kl \
                --zero_stage 0
            ;;
    esac
}

# === Main ===
echo "============================================================"
echo "RTX 4090 GRPO Training Runner"
echo "Framework: $FRAMEWORK | Model: $MODEL | Mode: $DEEPSPEED_MODE"
echo "LoRA: $LORA_RANK | Bypass: $BYPASS_MODE | Dry run: $DRY_RUN"
echo "============================================================"
echo

case $FRAMEWORK in
    rllm) run_rllm;;
    verl) run_verl;;
    deepspeed) run_deepspeed;;
    *)
        echo "Unknown framework: $FRAMEWORK"
        echo "Available: rllm, verl, deepspeed"
        echo "  rllm    → rLLM Tinker (in-process, fastest, simplest)"
        echo "  verl    → verl + vLLM (Ray, CPPO+bypass)"
        echo "  deepspeed → DeepSpeed (ZeRO-2 CPU_Adam, AutoEP MoE, OPD distillation)"
        exit 1
        ;;
esac
