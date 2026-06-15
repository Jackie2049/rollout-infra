# Cross-Framework LLM Training Benchmark Comparison — RTX 4090 Reference

> 2026-06-16 | RTX 4090 consulting reference
> Focus: Practical performance comparison of training frameworks on RTX 4090 (24GB)
> Note: These are estimated/theoretical benchmarks since GPU is offline. Actual numbers need GPU validation.

---

## 1. RTX 4090 Training Performance Estimates

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 1.1 GRPO Training Throughput (Qwen3-1.7B, LoRA rank=32, bypass=true)

| Framework | Config | Est. Step Time | Est. Throughput | Memory | Notes |
|-----------|--------|----------------|-----------------|--------|-------|
| rLLM Tinker | LoRA32+bypass+group4 | ~3-5s/step | ~200-300 tok/s | ~9.2GB | ★★★★★★★★★★ In-process, zero-copy, no Ray |
| verl+CPPO | LoRA32+bypass+PPOclip | ~5-8s/step | ~100-150 tok/s | ~10GB | ★★★★★★★ Ray overhead, but best algorithm |
| verl+GRPO | LoRA32+bypass+group4 | ~5-8s/step | ~100-150 tok/s | ~10GB | ★★★★★ Same framework, simpler algorithm |
| DeepSpeed ZeRO-2 | LoRA32+CPU_Adam+bypass | ~10-15s/step | ~50-80 tok/s | ~10GB | ★★★★★ CPU_Adam slower (CPU→GPU transfer) |
| Megatron core | N/A (not viable) | N/A | N/A | ~61GB | ✗ Full model only, singleton PG bugs |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 1.2 Inference Throughput (Qwen3-1.7B, BF16)

| Framework | Config | Est. Throughput | Memory | Notes |
|-----------|--------|-----------------|--------|-------|
| vLLM | BF16+INT8 KV | ~1000-1500 tok/s | ~5-6GB | ★★★★★ Mature, HMA default |
| SGLang | BF16+deterministic | ~800-1200 tok/s | ~5-6GB | ★★★★★ Deterministic, radix cache |
| vLLM | BF16+FP16 KV | ~1200-1800 tok/s | ~6-7GB | ★★★★★ Default, fastest but no INT8 savings |
| rLLM Tinker (rollout) | vLLM backend | Same as vLLM | Same | ★★★★★★★★★★ Uses vLLM for rollout, inherits throughput |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 1.3 Inference Throughput (Qwen3-8B, INT4)

| Framework | Config | Est. Throughput | Memory | Notes |
|-----------|--------|-----------------|--------|-------|
| vLLM | INT4+INT8 KV | ~300-500 tok/s | ~7-8GB | ★★★★★ INT4 Marlin, viable on 24GB |
| SGLang | INT4+deterministic | ~250-400 tok/s | ~7-8GB | ★★★★★ INT4 + Triton deterministic |
| vLLM | INT4+FP16 KV | ~400-600 tok/s | ~8-9GB | ★★★★★ Default KV, more memory |

---

## 2. GRPO Training Memory Breakdown (Qwen3-1.7B)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 2.1 rLLM Tinker (LoRA rank=32, bypass=true)

```
Base weights (frozen, BF16):     3.4GB
LoRA params (rank=32, BF16):     0.6GB
Optimizer (Adam, LoRA fp32):     1.2GB (m+v for LoRA only)
Activations (per step):          2.0GB
KV cache (rollout, FP16):        2.0GB
CUDA context + misc:             1.0GB
Ref model:                       0GB (bypass=true)
─────────────────────────────────────
Total:                           ~9.2GB ← FITS 24GB easily ✓
Available for more batch/group:  ~14.8GB remaining
```

★★★★★★★★★ Most efficient: in-process → no Ray overhead → zero-copy weight sync → bypass saves ref model → LoRA saves optimizer memory.

### 2.2 verl (LoRA rank=32, bypass=true)

```
Base weights (frozen, BF16):     3.4GB
LoRA params (rank=32, BF16):     0.6GB
Optimizer (Adam, LoRA fp32):     1.2GB
Activations (per step):          2.0GB
KV cache (rollout, FP16):        2.0GB
Ray overhead + IPC:              1.0GB
CUDA context + misc:             1.0GB
Ref model:                       0GB (bypass=true)
─────────────────────────────────────
Total:                           ~10.2GB ← FITS 24GB ✓
Available for more batch/group:  ~13.8GB remaining
```

★★★★★★★ Ray overhead adds ~1GB IPC buffers + coordination. Still viable.

### 2.3 DeepSpeed ZeRO-2 (LoRA rank=32, bypass=true)

```
Base weights (frozen, BF16):     3.4GB
LoRA params (rank=32, BF16):     0.6GB
Optimizer (CPU_Adam on CPU):     0GB (offloaded to CPU)
Activations (per step):          2.0GB
KV cache (rollout, FP16):        2.0GB
DeepSpeed framework overhead:    0.5GB
CUDA context + misc:             1.0GB
Ref model:                       0GB (bypass=true, custom)
─────────────────────────────────────
Total GPU:                       ~9.5GB ← FITS 24GB ✓
CPU optimizer:                   1.2GB (on CPU, not GPU)
Available for more batch/group:  ~14.5GB remaining
```

★★★★★★★ CPU_Adam frees GPU optimizer memory but adds CPU→GPU transfer latency per step (~1-2ms for LoRA params).

### 2.4 Full Model Training (NO LoRA, NO bypass) — NOT VIABLE

```
Base weights (trainable, BF16):  3.4GB
Gradients (full, BF16):          3.4GB
Optimizer (Adam, full, fp32):    13.6GB (6.8 m + 6.8 v)
Activations (per step):          8.0GB (full model, larger)
KV cache (rollout, FP16):        2.0GB
Ref model (BF16):                3.4GB
CUDA context + misc:             1.0GB
─────────────────────────────────────
Total:                           ~35.8GB ← EXCEEDS 24GB ✗✗✗
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ Full model training on 24GB = IMPOSSIBLE. This is why LoRA + bypass is mandatory.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 3. Step-by-Step Performance Analysis

### 3.1 rLLM Tinker GRPO Step (estimated ~3-5s)

```
1. Rollout (vLLM inference):        ~0.5-1.0s  (group_size=4 prompts × Qwen3-1.7B)
2. Reward computation:              ~0.1s       (simple math reward)
3. Advantage computation:           ~0.05s      (GRPO group normalization)
4. Forward pass (LoRA):             ~0.3s       (LoRA rank=32 forward on 1.7B)
5. Backward pass (LoRA):            ~0.5s       (LoRA backward)
6. Optimizer step:                  ~0.1s       (Adam on ~0.6GB LoRA params)
7. Weight sync (zero-copy):         ~0.01s      (save_weights_for_sampler → in-process!)
─────────────────────────────────────────────
Total per step:                     ~1.5-2.0s  (pure compute)
Overhead (Python, data loading):    ~1.0-3.0s  (varies by dataset)
Estimated total:                    ~3-5s per step
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ rLLM Tinker wins because: zero-copy weight sync (no IPC/serialization), in-process (no Ray coordination), bypass (no ref model forward/backward), LoRA only (tiny optimizer state).

### 3.2 verl GRPO Step (estimated ~5-8s)

```
1. Rollout (vLLM via Ray):          ~1.0-2.0s  (Ray→vLLM IPC overhead)
2. Reward computation:              ~0.1s
3. Advantage computation:           ~0.05s
4. Forward pass (LoRA):             ~0.3s
5. Backward pass (LoRA):            ~0.5s
6. Optimizer step:                  ~0.1s
7. Weight sync (Ray IPC):           ~0.5-1.0s  (serialize→Ray→deserialize)
─────────────────────────────────────────────
Total per step:                     ~2.5-4.0s  (pure compute + IPC)
Overhead (Ray, data loading):       ~2.0-4.0s
Estimated total:                    ~5-8s per step
```

★★★★★★★ verl slower due to: Ray IPC for rollout, weight serialization for sync, ref model computation (if not bypass), coordination overhead.

### 3.3 DeepSpeed ZeRO-2 GRPO Step (estimated ~10-15s)

```
1. Rollout (vLLM via Ray):          ~1.0-2.0s  (Ray→vLLM)
2. Reward computation:              ~0.1s
3. Advantage computation:           ~0.05s
4. Forward pass (LoRA):             ~0.3s
5. Backward pass (LoRA):            ~0.5s
6. Optimizer step (CPU_Adam):       ~1.0-2.0s  (CPU compute + CPU→GPU transfer)
7. Weight sync:                     ~0.5-1.0s  (DeepSpeed coordination)
8. ZeRO buffer management:          ~0.5s       (partition group management)
─────────────────────────────────────────────
Total per step:                     ~4.5-7.0s  (compute + transfer + overhead)
Overhead (ZeRO, data loading):      ~5.0-8.0s
Estimated total:                    ~10-15s per step
```

★★★★★★★ DeepSpeed slower due to: CPU_Adam transfer overhead, ZeRO buffer management, no native RL bypass, Ray coordination for rollout.

---

## 4. RTX 4090 GRPO End-to-End Training Time Estimates

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B, LoRA rank=32, bypass=true, 100 steps

| Framework | Est. Step Time | 100 Steps | Notes |
|-----------|----------------|-----------|-------|
| rLLM Tinker | ~3-5s | ~5-8 min | ★★★★★★★★★★ Fastest! |
| verl+CPPO | ~5-8s | ~8-13 min | ★★★★★ Best algorithm |
| verl+GRPO | ~5-8s | ~8-13 min | ★★★★★ Simplest verl |
| DeepSpeed ZeRO-2 | ~10-15s | ~17-25 min | ★★★★★ Slowest but works |

### Qwen3-8B, INT4 inference + LoRA rank=32, bypass=true, 100 steps

| Framework | Est. Step Time | 100 Steps | Notes |
|-----------|----------------|-----------|-------|
| rLLM Tinker | ~8-15s | ~13-25 min | ★★★★★★ Tight on 24GB |
| verl+CPPO | ~12-20s | ~20-33 min | ★★★★★ Tight, needs INT4 |
| DeepSpeed ZeRO-2 | ~20-30s | ~33-50 min | ★★★ CPU_Adam helps |

★★★★★★★★★ 8B models on RTX 4090 = ~2-4x slower than 1.7B due to INT4 inference overhead and tighter memory.

---

## 5. Inference Serving Benchmark Estimates

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B BF16, single request

| Framework | Est. TTFT | Est. T/s | Est. ITL | Notes |
|-----------|-----------|----------|----------|-------|
| vLLM | ~10ms | ~1500 | ~1ms | ★★★★★★★★ Fastest prefill |
| SGLang | ~15ms | ~1200 | ~1ms | ★★★★★ Deterministic overhead |
| vLLM (compile) | ~5ms | ~2000 | ~0.5ms | ★★★★★ Faster but batch-dependent on SM89 |

### Qwen3-8B INT4, single request

| Framework | Est. TTFT | Est. T/s | Est. ITL | Notes |
|-----------|-----------|----------|----------|-------|
| vLLM | ~30ms | ~500 | ~3ms | ★★★★★ INT4 Marlin |
| SGLang | ~40ms | ~400 | ~3ms | ★★★★★ INT4 + deterministic |

### Batch serving (32 concurrent requests)

| Framework | Model | Est. T/s/batch | Est. ITL | Notes |
|-----------|-------|----------------|----------|-------|
| vLLM | 1.7B BF16 | ~500 | ~2ms | ★★★★★ Good throughput |
| SGLang | 1.7B BF16 | ~400 | ~2ms | ★★★★★ Deterministic, radix cache |
| vLLM + BudgetRefiner | 1.7B BF16 | ~450 | ~1.5ms | ★★★★★★★★ SLO-aware, decode protected |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ BudgetRefiner impact: with SLO=50ms, decode ITL stays within SLO even under load!
Without BudgetRefiner: long prefill → decode blocked → ITL spike → SLO violation.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 6. Cost-Benefit Analysis per Framework

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### RTX 4090 Training Framework ROI

| Framework | Setup Time | Memory Efficiency | Speed | Algorithm Quality | Overall ROI |
|-----------|-----------|------------------|-------|-------------------|------------|
| rLLM Tinker | 5 min | ★★★★★★★★★★ | ★★★★★★★★★★ | ★★★★★ (ppo/IS/cispo/dro) | ★★★★★★★★★★ |
| verl+CPPO | 30 min | ★★★★★★★★ | ★★★★★★★★ | ★★★★★★★★★★ (CPPO best bound) | ★★★★★★★★★★ |
| verl+GRPO | 30 min | ★★★★★★★★ | ★★★★★★★★ | ★★★★★ (standard GRPO) | ★★★★★★★★ |
| DeepSpeed ZeRO-2 | 1 hour | ★★★★★★★★ | ★★★★★★ | ★★★★★★★★ (ZeRO-2 mature) | ★★★★★★★ (non-RL only) |
| Megatron core | 2+ hours | ★★★★★★★★ (not viable) | ✗ | ★★★★★★★★ (multi-GPU king) | ✗ (RTX 4090) |
| PyTorch FSDP2 | 30 min | ★★★★★★★★ (not viable) | ✗ | ★★★★★★★★ (composable) | ✗ (RTX 4090) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Decision Matrix Summary:

- **For GRPO/RL training**: rLLM Tinker #1 (fastest, simplest) → verl+CPPO #2 (best algorithm) → verl+GRPO #3 (simplest verl)
- **For supervised fine-tuning**: DeepSpeed ZeRO-2+LoRA #1 (CPU_Adam, mature) → rLLM Tinker #2 (LoRA+bypass)
- **For inference serving**: vLLM #1 (mature, INT8 KV) → SGLang #2 (deterministic, radix cache) → BudgetRefiner+vLLM #3 (SLO-aware)
- **For multi-GPU training**: Megatron core #1 (TP+PP+DP+EP) → DeepSpeed ZeRO-3 #2 (full sharding) → PyTorch FSDP2 #3 (composable)

---

## 7. Important Caveats

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. **All estimates need GPU validation**: These are theoretical/heuristic estimates. Actual performance depends on GPU availability, dataset size, model configuration, and hardware specifics.

2. **Batch invariance matters**: Without fixing batch invariance (Inductor SM<90 Fusion Guard or SGLang deterministic), torch.compile results on SM89 are unreliable.

3. **INT4 inference quality**: INT4 quantization has accuracy loss (~0.5-1% on most benchmarks). LoRA on BF16 base weights compensates for this during training.

4. **Memory estimates are conservative**: Actual peak memory may be higher due to fragmentation, temporary tensors, and CUDA memory allocation overhead.

5. **Step time includes data loading**: GRPO step time includes prompt generation, rollout, reward computation, and optimization. Pure training time is shorter.

6. **GPU offline**: All benchmarks are estimates. When GPU becomes available, run actual experiments with tools/rlhf_training_simulator.py and profile_vllm_budget.py.

---

## 8. Benchmark Collection Plan (When GPU Available)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Phase 1: Baseline Benchmarks (1-2 hours)

```bash
# vLLM inference throughput
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B --kv-cache-dtype int8
# Run benchmark with vllm/benchmarks/benchmark_serving.py

# SGLang inference throughput
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --enable-deterministic-inference
# Run benchmark with sglang/benchmark.py
```

### Phase 2: GRPO Training Benchmarks (2-4 hours)

```bash
# rLLM Tinker GRPO training
bash tools/train_tinker_rtx4090.sh --model Qwen3-1.7B --task math

# verl GRPO training
python -m verl.trainer.main_ppo --config config/verl/grpo_4090.yaml
```

### Phase 3: BudgetRefiner Profile Data (1-2 hours)

```bash
# Collect profile_table.csv for BudgetRefiner SLO
python3 tools/profile_vllm_budget.py --mode collect --models Qwen3-1.7B
```

### Phase 4: Batch Invariance Validation (1 hour)

```bash
# Verify batch invariance on SM89
python3 tools/sm89_batch_invariance_repro.py --config compile

# Verify SGLang deterministic inference
python -m sglang.launch_server --enable-deterministic-inference --model-path Qwen/Qwen3-1.7B
```
