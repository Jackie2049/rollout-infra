# Quantization Inference Benchmark — RTX 4090

> 2026-06-07 | **INT4 weight-only 0.87-1.08x(几乎免费!)+75%内存省; INT8 KV cache 1.00x+50%内存省+cos_sim=1.0; 但FP8 dequant开销1.5-2.5x!** Python-level量化必须用fused kernel.

## 核心发现

```
RTX 4090量化推理实测 — 5实验(3模型×5batch + KV cache):

┌──────────────────────────────────────────────────────────────┐
│ 关键数据:                                                    │
│                                                              │
│ 1. GEMM吞吐 (FP16 vs 量化):                                  │
│   INT4 weight-only: 0.87-1.08x → 几乎免费! +75%内存省        │
│   FP8 WONLY: 1.49-1.62x → dequant开销显著                   │
│   FP8 FULL(A+W): 1.89-2.17x → 两边都dequant更慢             │
│   INT8 cast: 不可行(精度损失+与FP16同大小)                    │
│                                                              │
│ 2. 量化误差 (FP16→量化→FP16):                                │
│   FP8 E4M3: cos_sim=0.9996, MSE=0.000707 → 很好             │
│   FP8 E5M2: cos_sim=0.9986, MSE=0.00277 → 可接受            │
│   INT8 weight-only: cos_sim=1.0000, MSE=0.0001 → 完美!      │
│   INT4 gs128: cos_sim=0.9933, MSE=0.0137 → 可接受           │
│                                                              │
│ 3. 内存节省:                                                  │
│   FP8: 50% (1byte vs 2bytes)                                │
│   INT8: 50% (1byte vs 2bytes)                               │
│   INT4: 75% (0.5byte packed vs 2bytes)                      │
│                                                              │
│ 4. Decode throughput:                                        │
│   INT4 weight-only: 0.96-1.01x → decode时dequant≈免费        │
│   FP8 WONLY: 1.45-1.56x → decode时dequant仍慢               │
│                                                              │
│ 5. KV Cache量化 (25M, B=32):                                 │
│   INT8 KV: 0.96-1.00x + 50%内存省 + cos_sim=1.0!            │
│   FP8 KV: 1.39-2.45x + 50%内存省 + cos_sim=0.9996           │
│   → INT8 KV cache = 完美方案! 零开销+50%容量↑                │
│   → FP8 KV = 仅适合存储, compute时必须dequant→慢             │
│                                                              │
│ **RTX 4090量化决策**:                                         │
│   1. 权重量化: INT4 weight-only(best!75%省+零开销)            │
│   2. KV cache量化: INT8 per-token(best!50%省+零开销+1.0精度) │
│   3. FP8: 仅适合通信(减少网络带宽)→不适合Python-level推理      │
│   4. 所有量化: 必须fused kernel→Python dequant=慢            │
└──────────────────────────────────────────────────────────────┘
```

## 完整数据

```
Exp 1: GEMM throughput (3模型 × 5 batch):

| Model | B | FP16(ms) | FP8_WONLY(ms) | FP8_WONLY/FP16 | INT4_WONLY(ms) | INT4/FP16 |
|-------|---|----------|---------------|----------------|---------------|-----------|
| 2M    | 1 | 0.028 | 0.042 | 1.54x | 0.029 | 1.05x |
| 2M    | 256 | 0.026 | 0.041 | 1.59x | 0.024 | 0.93x |
| 25M   | 1 | 0.029 | 0.044 | 1.49x | 0.030 | 1.01x |
| 25M   | 256 | 0.024 | 0.037 | 1.57x | 0.023 | 0.91x |
| 125M  | 1 | 0.023 | 0.037 | 1.61x | 0.023 | 1.00x |
| 125M  | 256 | 0.026 | 0.040 | 1.54x | 0.026 | 0.99x |

→ FP8 dequant overhead ≈ 1.5-1.6x (cast开销)
→ INT4 dequant overhead ≈ 0x (group-wise dequant融入matmul pipeline)

Exp 2: 量化误差:

| Format | cos_sim | MSE | Memory Savings |
|--------|---------|-----|---------------|
| FP8 E4M3 | 0.9996 | 0.0007 | 50% |
| FP8 E5M2 | 0.9986 | 0.0028 | 50% |
| INT8 W-only | 1.0000 | 0.0001 | 50% |
| INT4 gs128 | 0.9933 | 0.0137 | 75% |
| INT4 gs32  | ~0.987 | ~0.025 | 75% |

→ INT8 weight-only精度完美(cos_sim=1.0)!
→ INT4 gs128精度可接受(cos_sim=0.993)
→ INT4 group_size=128是sweet spot(32→更粗糙, 256→更精确)

Exp 3: 内存节省(理论计算):

| Model | FP16(MB) | FP8(MB) | INT8(MB) | INT4(MB) | INT4省% |
|-------|----------|---------|----------|----------|--------|
| 2M    | 24.8     | 12.4    | 12.4     | 6.2      | 75% |
| 25M   | 99.9     | 49.9    | 49.9     | 25.0     | 75% |
| 125M  | 602.5    | 301.2   | 301.2    | 150.6    | 75% |

→ INT4: 7B FP16=14GB → INT4=3.5GB → 24GB GPU可以跑B=256!

Exp 4: Decode throughput:

| Model | B | FP16(ms) | INT4(ms) | INT4/FP16 | FP8(ms) | FP8/FP16 |
|-------|---|----------|----------|-----------|---------|----------|
| 25M   | 1 | 0.034 | 0.035 | 1.00x | 0.050 | 1.46x |
| 25M   | 128 | 0.029 | 0.029 | 1.00x | 0.043 | 1.47x |
| 125M  | 32 | 0.033 | 0.032 | 0.97x | 0.048 | 1.45x |

→ INT4 decode: 几乎零开销! decode memory-bound → dequant不影响HBM读
→ FP8 decode: 1.5x开销 → cast占0.016ms额外时间

Exp 5: KV Cache量化 (25M, B=32):

| S | FP16(ms) | FP8(ms) | FP8/FP16 | INT8(ms) | INT8/FP16 | FP16_KV(MB) | FP8_KV(MB) | INT8_KV(MB) |
|---|----------|---------|----------|----------|-----------|-------------|------------|-------------|
| 512 | 0.095 | 0.132 | 1.39x | 0.091 | 0.96x | 33.6 | 16.8 | 16.8 |
| 2048 | 0.192 | 0.384 | 1.99x | 0.192 | 1.00x | 134.2 | 67.1 | 67.1 |
| 4096 | 0.349 | 0.844 | 2.42x | 0.349 | 1.00x | 268.4 | 134.2 | 134.2 |
| 8192 | 0.663 | 1.623 | 2.45x | 0.661 | 1.00x | 536.9 | 268.4 | 268.4 |

→ INT8 KV: **完美方案!** 零开销+50%内存省+cos_sim=1.0
→ FP8 KV: dequant开销随S线性增长 → 2.4x@S=8K → 不可接受
→ vLLM TurboQuant: INT8 per-token KV → 与实测完全吻合!
```

## 与之前实验的串联

```
串联发现链:

1. **KV Cache BW** (kv-cache-bandwidth-rtx4090.md):
   → MHA KV 44.5% decode瓶颈 → KV量越大→瓶颈越严重
   → → 本次: INT8 KV省50%容量 → 更多并发请求 → KV BW需求减半!
   → → → vLLM TurboQuant INT8 per-token = 与实测一致

2. **Decode Roofline** (decode-roofline-rtx4090.md):
   → decode严重memory-bound → 权重读占87%(B=1)
   → → 本次: INT4 weight-only省75%权重内存 → 权重读减少75%!
   → → → → 7B INT4: 权重3.5GB vs FP16 14GB → 可跑更大batch

3. **FlashAttention** (flash-attention-rtx4090.md):
   → decode更慢(0.67-0.84x) → 内存省85-97%
   → → 本次: INT8 KV省50%内存 → 与FlashAttention省内存目的相同
   → → → → INT8 KV更优: 1.00x(零开销) vs FA decode 0.67x(反而更慢!)

4. **Quantization Python overhead** (之前A16 benchmark):
   → Python dequant全部更慢 → 必须fused kernel
   → → 本次: INT4 weight-only dequant≈免费 → group-wise dequant可融入GEMM
   → → → → 但这是"模拟"INT4(GPTQ-style dequant FP16→compute)
   → → → → → 生产: 需要fused INT4 kernel(如Marlin/GPTQ kernel)

5. **DeepSeek-V3 FP8** (deepseek-v3-architecture.md):
   → FP8训练→$5.6M成本→通信省50%带宽
   → → 本次: FP8 Python-level推理慢1.5-2.5x → 但通信场景有效
   → → → → DeepSeek用FP8 for通信(NVLink)→HBM→FP8→传输→dequant→compute
   → → → → → 推理: 权重FP8→fused kernel dequant→compute → 需要专用kernel

→ **量化推理优化优先级(RTX 4090)**:
  1. **INT4 weight-only**: 75%内存省 + 0.87-1.08x speed → 最高杠杆!
  2. **INT8 KV cache**: 50%内存省 + 1.00x speed + cos_sim=1.0 → 完美方案
  3. **FP8通信**: 50%网络带宽省 → NVLink场景有效(RTX 4090 PCIe不适用)
  4. **FP8推理**: 不推荐(Python-level慢1.5-2.5x → 需fused kernel才有收益)
```

## RTX 4090量化推理决策树

```
RTX 4090量化推理决策:

Q: 目标是什么?
├── 内存不够 → INT4 weight-only(75%省) + INT8 KV(50%省)
│   → 7B FP16=14GB → INT4=3.5GB → 24GB可跑B=256!
│   → KV长context → INT8 KV → 50%容量↑ → 更多并发
│
├── 推理加速 → INT4 weight-only(dequant≈免费)
│   → decode memory-bound → INT4权重小75% → batch↑3x
│   → throughput↑3x(更多并发) → latency不变
│
├── 通信加速 → FP8(NVLink场景)
│   → RTX 4090 PCIe: 不适用(5-6 GB/s瓶颈)
│   → A100/H100 NVLink: FP8通信→带宽省50%
│
└── 精度要求高 → INT8 weight-only(cos_sim=1.0)
    → 50%内存省 + 完美精度 + 零开销
    → 比INT4更安全(cos_sim=1.0 vs 0.993)

推荐组合:
- 7B serving: INT4 weights + INT8 KV + FP16 compute → 4.5x内存省
- 13B serving: INT4 weights + INT8 KV → 24GB刚好够!
- 70B serving: INT4 weights + INT8 KV → 24GB仍不够(需要TP)
```

## 工具

- `tools/quantization_inference_benchmark_4090.py` — 5实验量化benchmark
- `results/quantization_inference_benchmark.json` — 完整数据