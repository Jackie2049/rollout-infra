---
name: inference-perf
description: Estimate LLM inference latency, throughput, and memory requirements. Roofline model, batch sizing, KV cache analysis.
triggers:
  - inference latency
  - throughput estimate
  - roofline model
  - inference memory
  - KV cache size
  - 推理性能
  - latency estimate
---

# LLM Inference Performance Estimator

## Key Formulas

### Decode (Memory-Bound, AI ≈ 1.0)

```
Latency per token = (2 × params × bytes_per_param) / HBM_bandwidth
Throughput = batch_size / latency_per_token
```

### Prefill (Compute-Bound)

```
TTFT = (2 × params × seq_len) / (TFLOPS × utilization)
```

### KV Cache Memory

```
KV per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
Total KV = KV_per_token × seq_len × batch_size

Examples (FP16):
  LLaMA-7B:  2×32×32×128×2 = 512KB/token
  LLaMA-70B: 2×80×8×128×2 = 320KB/token (GQA saves 8x!)
  7B@128K:  512KB × 128K = 64GB (> model weight!)
```

### Roofline Model

```
Ridge Point = Peak TFLOPS / HBM BW (ops/byte)
  A100: 312/2.0 = 156 ops/byte
  H100: 990/3.35 = 148 ops/byte
  A16:  14.7/0.170 = 86.5 ops/byte

Arithmetic Intensity:
  Decode: AI ≈ 1.0 (always memory-bound, only 0.7% peak FLOPS on A100)
  Prefill: AI = seq_len (compute-bound when seq > ridge_point)
```

## GPU Reference Data

| GPU | FP16 TFLOPS | HBM GB/s | Memory | NVLink |
|-----|-------------|----------|--------|--------|
| A16 | 14.7 | 170 | 15GB | No (PCIe) |
| A100 | 312 | 2000 | 80GB | 600 GB/s |
| H100 | 990 | 3350 | 80GB | 900 GB/s |
| H200 | 990 | 4800 | 141GB | 900 GB/s |

## Batch Sizing

```
Max batch = available_memory / KV_per_token_per_seq
  A100 80GB, 7B FP16: (80-14)GB / 512KB ≈ 128K tokens
  A16 15GB, 125M FP16: (15-0.25)GB / 128KB ≈ 116K tokens

Throughput vs Batch:
  batch=1:  2% peak throughput
  batch=32: ~50% peak
  batch=128: ~85% peak
  batch=512: ~95% peak (diminishing returns after 128)
```

## TP Communication Overhead

```
Decode TP overhead ≈ 11.5% (constant, independent of batch size)
  Because: AllReduce size = hidden_dim × batch × 2B, ~constant ratio

TP efficiency:
  TP=2:  ~97%
  TP=4:  ~95%
  TP=8:  ~94% (405B)
```

## Speculative Decoding Speedup

```
Speedup ≈ K / (1 + draft_cost_ratio × K)
  K=3, draft=0.1x: 2.3x speedup
  K=5, draft=0.1x: 3.3x speedup
  K=5, draft=0.5x: 2.0x speedup (diminishing returns)

Acceptance rate depends on distribution sharpness, NOT temperature.
```

## Prefix Caching Savings

```
Savings = prefix_hit_rate × (prefix_len / total_len) × 100%

Scenarios:
  System prompt: 100% hit rate (fixed prefix)
  Multi-turn (16 rounds): 80% cumulative hit
  GRPO (n=8, prompt=512): 67% KV saved
  Minimum ROI: prefix > 128 tokens
```

## Cost Estimation

```
$/M tokens = (GPU_cost/hr × 1000) / (throughput_tok/s × 3600)

Examples:
  7B FP8 on A100: ~$0.24/M tok
  Mixtral FP8 on H200: ~$0.09/M tok (MoE sparse)
  A100 most cost-efficient for dense models
```

## Tools Available

- `tools/gpu_inference_sim.py` — Full inference simulation
- `tools/gpu_perf_model.py` — Performance model with roofline
- `tools/gpu_memory_calculator.py` — Memory calculator
- `tools/inference_cost_calculator.py` — Cost calculator
- `tools/llm_latency_estimator.py` — Latency estimator

## Experiment Data (A16 Real)

- OPT-125M batch=1→64: 163→3729 tok/s (23x throughput gain)
- OPT-350M: 665 tok/s
- Max concurrent (512 tok): 699
- Simulator accuracy: 5-29% error
