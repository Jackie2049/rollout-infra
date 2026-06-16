#!/bin/bash
# RTX 4090 GPU Ready — Immediate Action Checklist
# ================================================
# When ANY GPU comes online, run this checklist IMMEDIATELY.
# Priority order from gpu-experiment-readiness-runbook.md.
#
# Usage:
#   bash tools/gpu_ready_checklist.sh --gpu university    # SSH to university server
#   bash tools/gpu_ready_checklist.sh --gpu matpool       # SSH to matpool server

set -e

GPU_TYPE="${2:-university}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo "${BLUE}[INFO]${NC} $1"; }
log_pass() { echo "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo "${RED}[FAIL]${NC} $1"; }
log_warn() { echo "${YELLOW}[WARN]${NC} $1"; }

# GPU SSH config
if [ "$GPU_TYPE" = "university" ]; then
    SSH_CMD="sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no zxw@219.233.198.62"
    CONDA_CMD="source ~/anaconda3/bin/activate llm"
elif [ "$GPU_TYPE" = "matpool" ]; then
    SSH_CMD="ssh -o StrictHostKeyChecking=no -p 28959 root@hz-t3.matpool.com"
    CONDA_CMD="source /root/miniconda3/bin/activate gpu-infra"
else
    log_fail "Unknown GPU type: $GPU_TYPE. Use 'university' or 'matpool'"
    exit 1
fi

echo "============================================================"
echo "RTX 4090 GPU Ready Checklist — GPU: $GPU_TYPE"
echo "============================================================"

# ============================================================
# Phase 0: Verify GPU Online (30 seconds)
# ============================================================

log_info "Phase 0: Verify GPU online and accessible"
GPU_INFO=$(timeout 10 $SSH_CMD "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader" 2>/dev/null)

if [ -z "$GPU_INFO" ]; then
    log_fail "GPU not accessible — SSH or nvidia-smi failed"
    exit 1
fi

GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
GPU_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
GPU_FREE=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')

log_pass "GPU: $GPU_NAME ($GPU_TOTAL total, $GPU_FREE free)"

if [[ "$GPU_NAME" == *"4090"* ]]; then
    log_pass "RTX 4090 confirmed — SM89 target"
else
    log_warn "GPU is $GPU_NAME — may differ from RTX 4090 SM89 target"
fi

# Check conda
CONDA_CHECK=$(timeout 10 $SSH_CMD "$CONDA_CMD && python3 -c 'import torch; print(torch.__version__)'" 2>/dev/null)
if [ -n "$CONDA_CHECK" ]; then
    log_pass "PyTorch: $CONDA_CHECK"
else
    log_fail "PyTorch not available in conda env"
fi

# Check vLLM
VLLM_CHECK=$(timeout 10 $SSH_CMD "$CONDA_CMD && python3 -c 'import vllm; print(vllm.__version__)'" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$VLLM_CHECK" != "NOT_INSTALLED" ]; then
    log_pass "vLLM: $VLLM_CHECK"
else
    log_warn "vLLM not installed — pip install vllm (use mirror: pip install vllm -i https://mirrors.aliyun.com/pypi/simple/)"
fi

echo ""

# ============================================================
# Phase 1: BudgetRefiner Profile Collection — P10 UNIQUE (TOP PRIORITY)
# ============================================================

log_info "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
log_info "Phase 1: BudgetRefiner profile_table.csv — P10 UNIQUE!"
log_info "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"

log_info "This is our TOP-PRIORITY experiment when GPU comes online."
log_info "RTX 4090 profile data = NO OTHER vLLM CONTRIBUTOR HAS THIS!"
log_info ""
log_info "To collect profile data, run on GPU server:"
log_info ""
echo "  $SSH_CMD '$CONDA_CMD && python3 tools/profile_vllm_budget.py --mode collect \\"
echo "    --models Qwen3-1.7B \\"
echo "    --seq-lens 512 1024 2048 \\"
echo "    --batch-sizes 1 4 8 16 32 \\"
echo "    --output results/profile_table_rtx4090.csv'"
log_info ""
log_info "Estimated time: ~1-2 hours"
log_info "After collection: validate with --mode validate"

echo ""

# ============================================================
# Phase 2: Batch Invariance Diagnostic — P9 (5 min)
# ============================================================

log_info "Phase 2: SM89 Batch Invariance Diagnostic — P9"
log_info "Quick check: 5 minutes to confirm batch invariance status"

echo "  $SSH_CMD '$CONDA_CMD && python3 tools/sm89_batch_invariance_diagnostic.py --mode check'"
echo "  $SSH_CMD '$CONDA_CMD && python3 tools/sm89_batch_invariance_repro.py --config none --verbose'"
echo "  $SSH_CMD '$CONDA_CMD && python3 tools/sm89_batch_invariance_repro.py --config compile --verbose'"

echo ""

# ============================================================
# Phase 3: vLLM Serving Quick Test (3 min)
# ============================================================

log_info "Phase 3: vLLM Serving Quick Test"

echo "  timeout 180 $SSH_CMD '$CONDA_CMD && python3 -c \""
echo "  from vllm import LLM, SamplingParams"
echo "  llm = LLM(model=\"Qwen/Qwen3-1.7B\", enforce_eager=True, gpu_memory_utilization=0.90)"
echo "  sp = SamplingParams(max_tokens=32, temperature=0)"
echo "  out = llm.generate([\"Hello\"], sp)"
echo "  print(\"vLLM PASS\", out[0].outputs[0].text[:50])"
echo "  \"'"

echo ""

# ============================================================
# Phase 4: DeepSpeed AutoEP MoE Quick Test — P6 UNIQUE (5 min)
# ============================================================

log_info "Phase 4: DeepSpeed AutoEP MoE Quick Test — P6"

echo "  $SSH_CMD '$CONDA_CMD && python3 -c \""
echo "  import deepspeed"
echo "  print(\"DeepSpeed:\", deepspeed.__version__)"
echo "  from deepspeed.moeh import MoEHConfig"
echo "  print(\"AutoEP available\")"
echo "  \"'"

echo ""

# ============================================================
# Phase 5: SGLang Deterministic Inference — P7 (5 min)
# ============================================================

log_info "Phase 5: SGLang Deterministic Inference Test — P7"

echo "  $SSH_CMD '$CONDA_CMD && pip install sglang[all] && python3 -c \""
echo "  import sglang; print(\"SGLang:\", sglang.__version__)"
echo "  \"'"

echo ""

# ============================================================
# Summary & Immediate Actions
# ============================================================

echo "============================================================"
echo "GPU Ready Checklist — Summary"
echo "============================================================"
echo ""
echo "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
echo "IMMEDIATE ACTION: BudgetRefiner profile_table.csv (P10 UNIQUE)"
echo "★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★"
echo ""
echo "Priority order:"
echo "  1. BudgetRefiner profile (1-2 hours) → P10 UNIQUE contribution"
echo "  2. Batch invariance diagnostic (5 min) → P9 Inductor Guard validation"
echo "  3. vLLM serving test (3 min) → basic functionality"
echo "  4. DeepSpeed AutoEP test (5 min) → P6 UNIQUE MoE viability"
echo "  5. SGLang deterministic test (5 min) → P7 SM89 alternative"
echo ""
echo "Full verification: bash tools/rtx4090_seven_framework_verify.sh --mode quick"
echo "Budget only:       bash tools/rtx4090_seven_framework_verify.sh --mode budget"
echo ""
echo "Reference: notebook/fundamentals/gpu-experiment-readiness-runbook.md"
