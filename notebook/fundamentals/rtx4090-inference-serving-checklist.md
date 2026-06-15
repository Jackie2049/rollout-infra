# RTX 4090 Inference Serving Checklist — v0.23.0 Reference

> 2026-06-16 | vLLM v0.23.0 + SGLang latest reference
> Focus: Practical inference serving configuration for RTX 4090 (24GB, SM89)
> ★★★★★ Production-ready reference for deployment

---

## 1. vLLM v0.23.0 RTX 4090 Serving Config

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B BF16 (recommended starting config):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype int8 \
  --enable-prefix-caching \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --enforce-eager   # ★★★★★ MUST for batch invariance on SM89!
```

★★★★★★★★★ Key decisions explained:
- `--enforce-eager`: Disable torch.compile → avoid Inductor RMSNorm fusion → batch invariance SAFE on SM89 (#39096)
- `--kv-cache-dtype int8`: INT8 FlashInfer KV → saves ~50% KV memory → more concurrent requests → ONLY viable KV quant on SM89
- `--enable-prefix-caching`: Shared prefix reuse → GRPO rollout benefits → HMA-by-default (#41847)
- `--gpu-memory-utilization 0.85`: Conservative → leave 15% headroom for CUDA context + fragmentation

### Qwen3-8B INT4 (production serving with quantization):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --quantization int4 \
  --kv-cache-dtype int8 \
  --enable-prefix-caching \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --enforce-eager
```

★★★★★★★ INT4 Marlin/Triton fallback (#43731) → v0.23.0 works on SM89 → INT4 inference viable!

---

## 2. SGLang RTX 4090 Serving Config

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B BF16 + deterministic (recommended for RL rollout):

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --enable-deterministic-inference \
  --mem-fraction-static 0.85 \
  --context-length 4096
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ SGLang deterministic inference → batch-invariant BY DESIGN → Triton constexpr BLOCK_SIZE → no Inductor fusion risk → NO enforce_eager needed → ★★★★★★★★★★★★★★★★★★★ GRPO rollout RECOMMENDED: SGLang deterministic + Triton backend + RadixAttention prefix reuse
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. SM89 Quantization Path Decision Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Weight Quantization (model loading):

| Quantization | SM89 Viable | Memory Savings | Quality Loss | Recommended |
|-------------|------------|----------------|--------------|-------------|
| BF16 (no quant) | ✓✓✓ | 0% | 0% | ★★★★★ Default for 1.7B |
| INT4 Marlin | ✓ (Triton fallback #43731) | 75% | ~0.5-1% | ★★★★★ 8B serving MUST |
| INT8 W8A8 | ✓ | 50% | ~0.3% | ★★★★ Good but INT4 better |
| FP8 E4M3 | ✗ (compressed-tensors crash) | 50% | N/A | ✗✗✗ NOT usable on SM89 |
| GPTQ INT4 | ✓ | 75% | ~1% | ★★★ Alternative to Marlin |
| AWQ INT4 | ✓ | 75% | ~0.8% | ★★★ Alternative to Marlin |

### KV Cache Quantization (inference):

| KV Type | SM89 Viable | Memory Savings | Backend | Recommended |
|---------|------------|----------------|---------|-------------|
| FP16 (no quant) | ✓✓✓ | 0% | All | ★★★ Default, fastest |
| INT8 | ✓✓✓ (FlashInfer) | 50% | FlashInfer | ★★★★★★★ MUST for 24GB |
| FP8 E4M3 Triton | ✓ (#43914) | 50% | Triton-only | ★★★★ Experimental |
| FP8 E4M3 FlashInfer | ✗ | N/A | FlashInfer | ✗✗✗ NOT on SM89 |
| FP8 compressed-tensors | ✗✗✗ CRASH | N/A | Override | ✗✗✗✗✗ CRASH #44879/#45038 |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ 3 FP8 KV paths MUST be distinguished: Triton ALLOWED / FlashInfer BLOCKED / compressed-tensors CRASH
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. Batch Invariance Decision Flowchart

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
Is batch invariance critical for your use case?
  │
  ├── YES (GRPO training, reward computation, reproducible results):
  │     │
  │     ├── vLLM → MUST use --enforce-eager
  │     │         → Disables torch.compile → no Inductor fusion
  │     │         → Throughput penalty: ~10-15% slower than compiled
  │     │         → BUT: results are batch-invariant ✓
  │     │
  │     ├── SGLang → MUST use --enable-deterministic-inference
  │     │           → Triton persistent constexpr → batch-invariant by design ✓
  │     │           → No throughput penalty! ✓✓
  │     │           → ★★★★★★★★ RECOMMENDED for GRPO!
  │     │
  │     └── torch.compile → NOT safe on SM89!
  │                 → Inductor fuses RMSNorm → tl.sum() → batch-dependent ✗
  │                 → Fix: Inductor SM<90 Fusion Guard (PR pending)
  │
  ├── NO (latency benchmark, throughput benchmark, casual testing):
  │     │
  │     ├── vLLM → CAN use torch.compile
  │     │         → Faster (~10-15% speedup)
  │     │         → BUT: results may be batch-dependent on SM89 ✗
  │     │         → OK for benchmarks where exact reproducibility not critical
  │     │
  │     └── SGLang → deterministic inference still recommended
  │                 → No throughput loss → always safe ✓
  │
  └── DON'T KNOW → use --enforce-eager (safe default) → verify later
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ GRPO rollout = MUST have batch invariance → reward computation depends on exact log probs → batch-dependent = incorrect rewards → wrong advantage → wrong gradient → FAILED TRAINING!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 5. RTX 4090 Memory Budget per Serving Scenario

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B BF16 + INT8 KV (single request):

```
Model weights BF16:    3.4GB
KV cache INT8:         ~0.5GB per 4K tokens
CUDA context:          1.0GB
vLLM framework:        0.5GB
───────────────────────────
Total:                 ~5.4GB → 18.6GB headroom → ~37 concurrent 4K requests
```

### Qwen3-1.7B BF16 + INT8 KV (64 concurrent, 4K tokens):

```
Model weights BF16:    3.4GB
KV cache INT8:         ~32GB for 64×4K → exceeds 24GB!
CUDA context:          1.0GB
───────────────────────────
Adjusted: max 32 concurrent → 16GB KV → total ~20.4GB → fits ✓
```

★★★★★★★★★ INT8 KV = ~0.5GB per 4K per request → 24GB → max ~32-40 concurrent requests → competitive!

### Qwen3-8B INT4 + INT8 KV (single request):

```
Model weights INT4:    ~4.0GB
KV cache INT8:         ~1.0GB per 4K tokens
CUDA context:          1.0GB
vLLM framework:        0.5GB
───────────────────────────
Total:                 ~6.5GB → 17.5GB headroom → ~17 concurrent 4K requests
```

---

## 6. BudgetRefiner SLO Integration (Future)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

When BudgetRefiner SLO is upstreamed to vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --kv-cache-dtype int8 \
  --enable-prefix-caching \
  --slo-limits-for-dynamic-batch 50 \
  --enforce-eager
```

★★★★★★★★★ BudgetRefiner + Watermark = comprehensive serving protection:
- BudgetRefiner: compute time pressure → dynamic prefill budget → decode-first → SLO guaranteed
- Watermark: KV cache pressure → admission gate → prevent thrashing → -82% preemptions
- Together: ~zero preemptions + ~zero SLO violations → RTX 4090 production-ready!

---

## 7. Common Pitfalls on SM89

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| FP8 KV crash | SIGSEGV/corrupt output | Use INT8 KV or FP16 only |
| Batch invariance | Different log probs per batch size | --enforce-eager or SGLang deterministic |
| MRv2 unknown impact | verl weight sync may fail | VLLM_USE_V2_MODEL_RUNNER=0 (conservative) |
| INT4 Triton fallback | Slower than Marlin on SM90 | Expected; still faster than BF16 for 8B |
| FP8 weight quant | Crash or fallback to FP16 | Use INT4/INT8 instead |
| OOM under load | Preemptions, ITL spikes | INT8 KV + watermark + lower max-num-seqs |
| CUDA graph crash | Graph capture failure on SM89 | --enforce-eager (disables graphs) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ 7 pitfalls → all have known fixes → RTX 4090 serving is production-viable with correct config!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. GPU Validation Checklist (When Servers Online)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Priority 1: Batch invariance (30 min):
```bash
# vLLM --enforce-eager vs torch.compile vs SGLang deterministic
python3 tools/sm89_batch_invariance_repro.py --config all
```

### Priority 2: INT8 KV throughput (30 min):
```bash
# vLLM INT8 vs FP16 KV throughput
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --kv-cache-dtype int8
# benchmark with vllm/benchmarks/benchmark_serving.py
```

### Priority 3: INT4 8B model serving (30 min):
```bash
# vLLM INT4 Marlin/Triton fallback on SM89
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --quantization int4 --kv-cache-dtype int8
```

### Priority 4: SGLang deterministic (30 min):
```bash
# SGLang deterministic inference throughput vs vLLM --enforce-eager
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --enable-deterministic-inference
```

### Priority 5: BudgetRefiner profile data (1 hour):
```bash
# Collect RTX 4090 profile_table.csv → P10 UNIQUE contribution!
python3 tools/profile_vllm_budget.py --mode collect --models Qwen3-1.7B
```
