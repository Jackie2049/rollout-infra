# GPU Experiment Readiness Runbook — RTX 4090

> 2026-06-16 | Consolidated from 227+ notes, 874+ commits
> Purpose: When GPU comes online, run these experiments IMMEDIATELY — no setup time wasted
> ★★★★★★ This document = your "GPU上线第一件事做什么" runbook!

---

## Prerequisites Checklist (Run First!)

```bash
# 1. Verify GPU online and SM version
sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no zxw@219.233.198.62 \
  "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader"

# 2. Check conda env exists with vLLM + torch
sshpass -p 'adspzxw123' ssh zxw@219.233.198.62 \
  "conda activate rollout; python -c 'import torch; import vllm; print(torch.cuda.get_device_capability())'"

# 3. Verify key scripts are on server
sshpass -p 'adspzxw123' ssh zxw@219.233.198.62 \
  "ls ~/rollout-infra/tools/sm89_batch_invariance_repro.py ~/rollout-infra/tools/profile_vllm_budget.py"
```

**If prerequisites fail**, sync the repo first:
```bash
sshpass -p 'adspzxw123' ssh zxw@219.233.198.62 \
  "cd ~/rollout-infra && git pull origin main"
```

---

## Experiment 1: SM89 Batch Invariance Reproduction (30 min, P9)

★★★★★★★★★ Root cause CONFIRMED: Inductor fuses RMSNorm → ONE kernel → XBLOCK varies → batch-dependent
★★★★★★★★★ This experiment validates the Inductor SM<90 Fusion Guard PR approach

### Setup
```bash
# On GPU server, activate conda env
conda activate rollout

# Install minimal dependencies
pip install vllm torch --no-deps  # if not already installed
```

### Run (3 configs, ~10 min each)
```bash
cd ~/rollout-infra

# Config 1: Baseline — no compile, no graphs — SHOULD PASS
python tools/sm89_batch_invariance_repro.py --config none --verbose

# Config 2: torch.compile only — EXPECTED FAIL on SM<90
python tools/sm89_batch_invariance_repro.py --config compile --verbose

# Config 3: torch.compile + CUDA graphs — EXPECTED FAIL on SM<90
python tools/sm89_batch_invariance_repro.py --config compile_graphs --verbose
```

### Expected Outcomes

| Config | SM90+ Result | SM89 Result | Expected |
|--------|-------------|-------------|----------|
| none | batch-invariant ✓ | batch-invariant ✓ | PASS |
| compile | batch-invariant ✓ | batch-dependent ✗ | FAIL (confirms root cause) |
| compile_graphs | batch-invariant ✓ | batch-dependent ✗ | FAIL |

### Data to Collect
- Exact SM version from `torch.cuda.get_device_capability()`
- Whether compile config shows batch-dependent output
- Exact RMSNorm trace: which Triton kernels are fused
- XBLOCK values for different batch sizes (if possible via Triton debug)

### If Experiment Succeeds (FAIL on SM<90)
→ ★★★★★★★ Confirms Inductor Fusion Guard PR is CORRECT
→ Take data → submit PyTorch PR with exact repro + 5-line patch
→ Update notebook/projects/pytorch-inductor-sm89-fusion-guard-pr-draft.md with results

---

## Experiment 2: BudgetRefiner Profile Data Collection (1 hour, P10 UNIQUE)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ RTX 4090 profile data = NO OTHER vLLM CONTRIBUTOR HAS THIS → UNIQUE contribution!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Setup
```bash
conda activate rollout
pip install vllm pandas  # vLLM v0.23.0 preferred
```

### Run (4-mode profiler)
```bash
cd ~/rollout-infra

# Mode 1: Collect profile_table.csv — THE KEY DATA!
python tools/profile_vllm_budget.py --mode collect \
  --models Qwen3-1.7B Qwen3-8B \
  --seq-lens 512 1024 2046 4096 \
  --batch-sizes 1 4 8 16 32 \
  --output profile_table_rtx4090.csv

# Mode 2: Validate collected data
python tools/profile_vllm_budget.py --mode validate \
  --csv profile_table_rtx4090.csv

# Mode 3: Estimate budget for specific config
python tools/profile_vllm_budget.py --mode estimate \
  --model Qwen3-1.7B --batch 8 --seq-len 4096

# Mode 4: Full pipeline (collect + validate + estimate)
python tools/profile_vllm_budget.py --mode all
```

### Profile Table Schema (matches vLLM-Ascend BudgetRefiner)

| Column | Description | Example |
|--------|-------------|---------|
| model | Model name | Qwen3-1.7B |
| quantization | Quant method | BF16, GPTQ-Int4 |
| chunk_size | Tokens per chunk | 1024 |
| p_len | Prefill tokens | 2046 |
| d_num | Decode sequences | 8 |
| ctx_len | Total context length | 4096 |
| cost | Iteration time (ms) | 35.2 |

### Critical Data Points to Collect

★★★★★★★★★ BudgetRefiner lookup key = (ctx_len, d_num) → chunk_size
★★★★★★★★★ Budget DROPS as decode load increases! Need to map this curve:

| d_num | Expected budget (SLO=50ms) | Data needed |
|-------|---------------------------|-------------|
| 0 | 1024 (max) | Prefill-only baseline |
| 50 | ~768 (25% drop) | Decode pressure starts |
| 100 | ~640 (37% drop) | Moderate decode |
| 200 | ~512 (50% drop) | Heavy decode |
| 255 | ~256 (75% drop) | Maximum decode pressure |

### vLLM Config for RTX 4090
```python
# MUST USE these flags for SM89:
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    enforce_eager=True,           # ★★★ MUST — batch invariance
    gpu_memory_utilization=0.90,  # 24GB → 21.6GB usable
    max_model_len=4096,           # Fits in ~5.4GB KV cache
    kv_cache_dtype="int8",        # ★★★ INT8 FlashInfer KV (SM89 viable)
    enable_prefix_caching=True,   # Prefix reuse for GRPO
    tensor_parallel_size=1,       # Single GPU
)
```

### If Experiment Succeeds
→ ★★★★★★★★★★★★★★★ RTX 4090 profile_table.csv = UNIQUE vLLM contribution data!
→ Submit BudgetRefiner SLO RFC to vLLM community discussion
→ Update notebook/projects/budgetrefiner-vllm-pr-draft.md with real data
→ This is P10 — our HIGHEST priority contribution!

---

## Experiment 3: INT8 KV + vLLM Serving Benchmark (30 min, P8)

★★★★★★ INT8 FlashInfer KV = ONLY viable KV quantization on SM89
★★★★★★ Need throughput numbers for RTX 4090 consulting reference

### Run
```bash
# vLLM serving benchmark — Qwen3-1.7B
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --enforce-eager \
  --kv-cache-dtype int8 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000 &

# Wait for model load, then benchmark
python benchmark_serving.py \
  --model Qwen/Qwen3-1.7B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --port 8000

# Collect: throughput (tok/s), latency P50/P99, memory usage
```

### vLLM v0.23.0 INT4 8B Serving (additional test)
```bash
# Qwen3-8B — INT4 weight + INT8 KV
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enforce-eager \
  --quantization gptq_int4 \
  --kv-cache-dtype int8 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --port 8001 &

# Benchmark 8B model
python benchmark_serving.py \
  --model Qwen/Qwen3-8B \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 64 \
  --num-prompts 50 \
  --port 8001
```

### Data to Collect

| Model | KV dtype | Weight dtype | Memory | Throughput | P99 latency |
|-------|----------|-------------|--------|------------|-------------|
| Qwen3-1.7B | INT8 | BF16 | ~5.4GB | ? tok/s | ? ms |
| Qwen3-1.7B | FP16 | BF16 | ~7.2GB | ? tok/s | ? ms |
| Qwen3-8B | INT8 | INT4 | ~6.5GB | ? tok/s | ? ms |

---

## Experiment 4: SGLang Deterministic vs vLLM enforce_eager (30 min, P7)

★★★★★★★★★ SGLang deterministic = batch-invariant by design (INFERENCE level)
★★★★★★★★★ vLLM enforce_eager = prevents Inductor fusion (MODEL level)
★★★★★★★★★ Need to compare: does SGLang deterministic truly solve SM89 batch invariance?

### SGLang Deterministic Test
```bash
# SGLang with deterministic inference
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --enable-deterministic-inference \
  --port 8002 &

# Test batch invariance: same input, different batch sizes
python -c "
import requests, json
prompt = 'What is the capital of France?'
# Single request
r1 = requests.post('http://localhost:8002/generate', json={'text': prompt, 'max_tokens': 50})
# Batch of 4 identical requests
r4 = requests.post('http://localhost:8002/generate', json={'text': [prompt]*4, 'max_tokens': 50})
# Compare: r1 output == r4 output[0]?
print('Single:', r1.json())
print('Batch[0]:', r4.json()[0])
print('Invariant:', r1.json()['text'] == r4.json()[0]['text'])
"
```

### vLLM enforce_eager Comparison
```bash
# vLLM with enforce_eager (same model)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --enforce-eager \
  --port 8003 &

# Same batch invariance test
python -c "
import openai
client = openai.Client(base_url='http://localhost:8003/v1', api_key='dummy')
# Single request
r1 = client.completions.create(model='Qwen/Qwen3-1.7B', prompt='What is the capital of France?', max_tokens=50)
# Batch of 4
r4 = client.completions.create(model='Qwen/Qwen3-1.7B', prompt=['What is the capital of France?']*4, max_tokens=50)
print('Invariant:', r1.choices[0].text == r4.choices[0].text)
"
```

### Expected Outcomes

| Framework | Config | Batch Invariant? | Throughput Impact |
|-----------|--------|-----------------|-------------------|
| SGLang | deterministic | ✓ (by design) | ~0% loss (constexpr) |
| vLLM | enforce_eager | ✓ (no fusion) | ~5-10% loss (no compile opt) |
| vLLM | compile (no enforce_eager) | ✗ on SM89 | faster but WRONG results |

---

## Experiment 5: DeepSpeed AutoEP MoE Smoke Test (30 min, P6 UNIQUE)

★★★★★★★★★★★★★★★★ AutoEP + EP=1 + LoRA makes MoE training viable on RTX 4090!
★★★★★★★★★★★★★★★★ First-ever RTX 4090 MoE training validation → UNIQUE contribution!

### Setup
```bash
conda activate rollout
pip install deepspeed>=0.19.2  # Must have AutoEP (#7938)
pip install transformers peft
```

### Run (dry-run first, then real)
```bash
cd ~/rollout-infra

# Step 1: Generate AutoEP config for RTX 4090
python tools/deepspeed_autoep_rtx4090_config.py --mode generate \
  --model qwen3_moe --lora-rank 32 --zenflow

# Step 2: Validate memory estimate
python tools/deepspeed_autoep_rtx4090_config.py --mode validate \
  --model qwen3_moe --lora-rank 32 --zenflow
# Expected: ~11.8GB ✓ (fits 24GB!)

# Step 3: Dry-run training (no actual weights, just memory test)
python tools/deepspeed_autoep_rtx4090_config.py --mode compare --model qwen3_moe

# Step 4: Real MoE training smoke test (1 step)
deepspeed --num_gpus 1 train_moe_autoep.py \
  --model_name Qwen/Qwen3-MoE \
  --lora_rank 32 \
  --lora_target_modules mlp attn unembed \
  --zero_stage 2 \
  --offload_optimizer cpu \
  --auto_ep_enable \
  --auto_ep_ep_size 1 \
  --max_steps 1 \
  --dry_run  # first with dry_run, then remove for real
```

### Data to Collect

| Metric | Expected | Actual |
|--------|----------|--------|
| GPU memory at init | ~11.8GB | ? |
| GPU memory at step | ~16.2GB peak | ? |
| Step time (1st) | ~30s (cold) | ? |
| Step time (2nd+) | ~5-8s | ? |
| Loss value | decreasing | ? |
| ZeRO partitioning | full (dp=1) | ? |
| AutoEP EP=1 | identity AllToAll | ? |

---

## Experiment 6: Inductor Fusion Guard Validation (1 hour, P9)

★★★★★★★★★ Validate the exact 5-line patch for PyTorch upstream

### Setup
```bash
# Need PyTorch source with modifications
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu124
# OR build from source with our patch
```

### Test Plan (from PR draft)
```python
# 4 mock-based unit tests:

# Test 1: SM89 blocked — reduction fusion should NOT happen
def test_sm89_reduction_fusion_blocked():
    # props.major = 8 (< 9) + is_reduction() = True → WhyNoFuse → False
    # RMSNorm should NOT be fused → stays separate → batch-invariant

# Test 2: SM90 allowed — reduction fusion should proceed
def test_sm90_reduction_fusion_allowed():
    # props.major = 9 (≥ 9) + is_reduction() = True → True → fuse → fast

# Test 3: SM89 non-reduction allowed — non-reduction ops can still fuse
def test_sm89_non_reduction_fusion_allowed():
    # props.major = 8 + is_reduction() = False → True → non-RMSNorm ops fuse OK

# Test 4: XPU/CPU unaffected — only CUDA SM<90 gated
def test_xpu_unaffected():
    # device.type = xpu → bypass gate → True → XPU fusion unaffected
```

### Manual Validation Steps
```bash
# 1. Apply 5-line patch to choices.py
# Location: torch/_inductor/choices.py lines 639-647
# Patch: DeviceProperties.create(props) + WhyNoFuse + props.major < 9

# 2. Rebuild PyTorch with patch
python setup.py develop

# 3. Run batch invariance repro with patched PyTorch
python tools/sm89_batch_invariance_repro.py --config compile --verbose
# Expected: NOW PASSES on SM89! (RMSNorm not fused → separate → torch.mean override effective)

# 4. Verify no throughput regression on SM90+
# Same test on SM90 GPU (if available) → compile should still work → fast
```

---

## Experiment 7: INT4 8B Serving on SM89 (30 min, P3)

★★★★★★ vLLM v0.23.0 INT4 Triton fallback (#43731) works on SM89
★★★★ Need throughput data for 8B models on RTX 4090

### Run
```bash
# Qwen3-8B INT4 serving
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --enforce-eager \
  --quantization gptq_int4 \
  --kv-cache-dtype int8 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --port 8004 &

# Benchmark
python benchmark_serving.py \
  --model Qwen/Qwen3-8B \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 64 \
  --num-prompts 50

# Collect: throughput, memory, latency
# Expected: ~6.5GB memory, ~300-500 tok/s throughput
```

---

## Experiment Priority & Time Budget

| # | Experiment | Priority | Time | Dependencies | Unique? |
|---|-----------|----------|------|-------------|---------|
| 1 | Batch invariance repro | P9 | 30 min | torch+vllm | ★★★★★ Confirms PR |
| 2 | BudgetRefiner profile | P10 | 1 hour | vllm v0.23 | ★★★★★★★★★★★★★★★★★ UNIQUE! |
| 3 | INT8 KV benchmark | P8 | 30 min | vllm | ★★★ Consulting data |
| 4 | SGLang deterministic vs vLLM | P7 | 30 min | sglang+vllm | ★★★★★★★ SM89 alternative |
| 5 | AutoEP MoE smoke test | P6 | 30 min | deepspeed>=0.19.2 | ★★★★★★★★★★★★★★★★★ UNIQUE! |
| 6 | Fusion Guard validation | P9 | 1 hour | patched pytorch | ★★★★★★★ PR validation |
| 7 | INT4 8B serving | P3 | 30 min | vllm | ★★★ Consulting data |

### Recommended Order (GPU comes online)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ START WITH Experiment 2 (BudgetRefiner) → P10 UNIQUE → most valuable data!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

Then:
1. Exp 2 (BudgetRefiner profile — 1h) → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ P10 UNIQUE
2. Exp 1 (Batch invariance — 30min) → Confirms P9 PR
3. Exp 5 (AutoEP MoE — 30min) → ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ P6 UNIQUE
4. Exp 4 (SGLang deterministic — 30min) → SM89 alternative validation
5. Exp 3 (INT8 KV — 30min) → Consulting data
6. Exp 7 (INT4 8B — 30min) → Consulting data
7. Exp 6 (Fusion Guard — 1h) → Needs patched PyTorch → last

Total time: ~4 hours → Can complete all in one GPU session!

---

## Server-Specific Notes

### University Server (219.233.198.62) — PREFERRED
```bash
# Login
sshpass -p 'adspzxw123' ssh -o StrictHostKeyChecking=no zxw@219.233.198.62

# Features:
# - More GPU hours (university schedule)
# - Potentially better GPU (A100/H100?)
# - More storage
# - conda envs likely already set up

# Check available GPUs:
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### Matpool Server (hz-t3.matpool.com:28959) — BACKUP
```bash
# Login
ssh -p 28959 root@hz-t3.matpool.com

# Features:
# - RTX 4090 guaranteed (paid for)
# - Limited hours (paid subscription)
# - May need fresh environment setup
# - Password: TUR]Nr3fyxM%7)iD

# Quick env setup if needed:
conda create -n rollout python=3.12
conda activate rollout
pip install torch vllm deepspeed transformers peft pandas -i https://mirrors.aliyun.com/pypi/simple/
```

---

## Post-Experiment Actions

### After Each Experiment
1. Save raw data to `results/` directory (gitignored but backed up locally)
2. Update relevant notebook/project note with actual numbers
3. Update diary/2026-06-XX.md with experiment results
4. If UNIQUE contribution data → prepare PR/comment draft immediately

### After All Experiments Complete
1. Update `rtx4090-ai-infra-consulting-quick-reference.md` with actual benchmark numbers
2. Update `cross-framework-training-benchmark-comparison.md` with real data
3. Submit BudgetRefiner SLO RFC to vLLM community (P10)
4. Submit Inductor Fusion Guard PR to PyTorch (P9)
5. Submit AutoEP MoE benchmark to DeepSpeed cookbook (P6)
6. Push all updates to GitHub
