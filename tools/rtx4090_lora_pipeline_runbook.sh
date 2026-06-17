#!/bin/bash
# RTX 4090 LoRA Training → Deployment End-to-End Runbook
# ★★★★★★★★ Execute when GPU is available — follow priority order!
# Reference: notebook/fundamentals/cross-framework-lora-adapter-export-comparison.md
# Reference: notebook/fundamentals/rtx4090-training-to-deployment-pipeline.md

set -e

echo "★★★★★★★★★ RTX 4090 LoRA Training → Deployment End-to-End Runbook ★★★★★★★★★★"

# Step 0: Environment check
echo ""
echo "Step 0: Environment check"
echo "-------------------------------------------"

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
    SM_VER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "  SM: $SM_VER"
    if [[ "$SM_VER" == "8.9" ]]; then
        echo "  ★★★★★ RTX 4090 (SM89) DETECTED!"
    else
        echo "  ⚠ SM version $SM_VER — some configs may need adjustment"
    fi
else
    echo "  ✗ No GPU detected — cannot proceed with training"
    exit 1
fi

# Check Python
echo "  Python: $(python3 --version)"
echo "  Conda: $(conda --version 2>/dev/null || echo 'not installed')"
echo ""

# Step 1: Install dependencies (use mirrors for China mainland)
echo "Step 1: Install dependencies"
echo "-------------------------------------------"
PIP_MIRROR="-i https://mirrors.aliyun.com/pypi/simple/"
CONDA_MIRROR="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/main/"

echo "  Install core packages..."
pip install $PIP_MIRROR torch vllm sglang 2>/dev/null || echo "  (may need separate install steps)"
echo ""

# Step 2: Generate safe DeepSpeed config
echo "Step 2: Generate safe DeepSpeed config"
echo "-------------------------------------------"
echo "  Choose scenario:"
echo "    1. lora-grpo       (Dense GRPO, Qwen3-1.7B)"
echo "    2. lora-grpo-muon  (Muon experimental, Qwen3-1.7B)"
echo "    3. moe-autoep      (MoE AutoEP, Qwen3-MoE)"
echo "    4. opd-distill     (OPD student, Qwen2.5-0.5B)"
echo ""

# Example: generate lora-grpo config
python3 tools/deepspeed_config_generator.py --scenario lora-grpo --model qwen3-1.7b --dry-run
echo ""

# Step 3: Safety check BEFORE training
echo "Step 3: Safety check (★★★★★★★★★ MUST DO BEFORE ANY TRAINING!)"
echo "-------------------------------------------"
echo "  Run deepspeed_zero_safety_checker on generated config"
echo "  python3 tools/deepspeed_zero_safety_checker.py --mode check --config configs/lora-grpo_rtx4090.json"
echo ""

# Step 4: Train LoRA (★★★★★★★★★ Priority: P10 BudgetRefiner first, then training)
echo "Step 4: Train LoRA"
echo "-------------------------------------------"
echo ""
echo "  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
echo "  ★★★★★★★★★★ P10 BudgetRefiner profile data collection FIRST! ★★★★★★★★★★★★"
echo "  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
echo ""
echo "  python3 tools/rtx4090_gpu_experiment_runner.py --run p10"
echo ""
echo "  THEN train LoRA:"
echo ""
echo "  # Option A: DeepSpeed ZeRO-2 (for all scenarios)"
echo "  python3 tools/deepspeed_config_generator.py --scenario lora-grpo --model qwen3-1.7b"
echo "  deepspeed --num_gpus=1 train.py --deepspeed configs/lora-grpo_rtx4090.json --output_dir ./output"
echo ""
echo "  # Option B: rLLM Tinker (★★★★★★★★★ SIMPLEST for dense GRPO)"
echo "  python -m rllm.tinker.train --model Qwen/Qwen3-1.7B --algorithm GRPO --lora_rank 32 --bypass_mode true"
echo ""

# Step 5: Export LoRA adapter
echo "Step 5: Export LoRA adapter → PEFT format"
echo "-------------------------------------------"
echo "  # Standard PEFT export — works for ALL frameworks"
echo "  python3 tools/export_lora_adapter.py --input ./output/qwen3-1.7b-grpo-lora32 --format peft"
echo "  # Output: adapter_model.safetensors (~0.3GB) + adapter_config.json"
echo ""

# Step 6: Merge LoRA (optional — zero inference overhead)
echo "Step 6: Merge LoRA → full model (optional, irreversible)"
echo "-------------------------------------------"
echo "  # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ONLY merge if you won't train more LoRA on this base!"
echo "  # merge_and_unload() → zero overhead → but irreversible!"
echo "  python3 tools/merge_lora.py --base Qwen/Qwen3-1.7B --adapter ./output --output ./merged/qwen3-grpo"
echo ""

# Step 7: Quantize (optional — 3x compression)
echo "Step 7: Quantize → INT4/AWQ (optional, 3x compression)"
echo "-------------------------------------------"
echo "  # AWQ INT4 — recommended for RTX 4090"
echo "  python -m awq.entrypoint --model_path ./merged/qwen3-grpo --w_bit 4 --output_path ./quantized/qwen3-grpo-awq"
echo ""
echo "  # ★★★★★★★★★★★★★★★★★★★★★★★★★★★ MUST use Marlin kernel for inference!"
echo "  # Python dequant = 20x slower → fused Marlin = near-native speed"
echo ""

# Step 8: Serve — vLLM or SGLang
echo "Step 8: Serve → vLLM or SGLang"
echo "-------------------------------------------"
echo ""
echo "  # vLLM — throughput-focused (★★★★★★★★★ enforce_eager for SM89 batch invariance)"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model ./quantized/qwen3-grpo-awq \\"
echo "    --gpu-memory-utilization 0.85 \\"
echo "    --kv-cache-dtype int8 \\"
echo "    --enable-prefix-caching \\"
echo "    --enforce-eager \\"
echo "    --watermark 0.05 \\"
echo "    --max-model-len 4096"
echo ""
echo "  # SGLang — quality-focused (★★★★★★★★★★★★★★★★ deterministic inference, batch-invariant by design)"
echo "  python -m sglang.launch_server \\"
echo "    --model-path ./quantized/qwen3-grpo-awq \\"
echo "    --enable-deterministic-inference \\"
echo "    --mem-fraction-static 0.85 \\"
echo "    --context-length 4096"
echo ""
echo "  # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ vLLM multi-LoRA hot-swap:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model Qwen/Qwen3-1.7B \\"
echo "    --enable-lora \\"
echo "    --lora-modules grpo-v1=./output/qwen3-1.7b-grpo-lora32 \\"
echo "                   grpo-v2=./output/qwen3-1.7b-grpo-v2 \\"
echo "    --enforce-eager"
echo ""

# Step 9: Speculative decoding (optional)
echo "Step 9: Speculative decoding (optional, 2-4x speedup)"
echo "-------------------------------------------"
echo "  # N-gram — simplest (2.14x, zero-cost draft)"
echo "  # Add to vLLM: --speculative-algorithm ngram --num-speculative-tokens 5"
echo ""
echo "  # EAGLE — best (4.2x, trained draft model)"
echo "  # Add to vLLM: --speculative-model ./eagle-draft --speculative-algorithm eagle"
echo ""

echo "★★★★★★★★★ End-to-end pipeline complete! ★★★★★★★★★★"
echo "Reference: notebook/fundamentals/rtx4090-training-to-deployment-pipeline.md"
