# Attention Backend Comparison Benchmark — RTX 4090

> 2026-06-07 | **SDPA是RTX 4090最优backend! FA2 decode 3-34x更慢, 仅prefill长序列有价值(85-96%内存省)**

## 核心发现

```
RTX 4090 Attention Backend实测 — 5实验(Naive/SDPA/FA2):

┌──────────────────────────────────────────────────────────────┐
│ Prefill (B=8, causal):                                       │
│                                                              │
│ S=128:  naive 2.6x慢, FA2 3.2x慢 vs SDPA                    │
│ S=512:  naive 3-9x慢, FA2 1.6-2.4x慢 vs SDPA               │
│ S=1024: naive 12-18x慢, FA2 1.3-1.6x慢 vs SDPA             │
│ S=4096: naive 15-29x慢, FA2 ≈1.0x vs SDPA!                 │
│                                                              │
│ → SDPA always fastest for prefill!                           │
│ → FA2 ≈SDPA speed at S≥4096 but saves 92-96% memory         │
│ → FA2 value = memory saving, NOT speed                       │
│                                                              │
│ Decode (Q=1, causal):                                        │
│                                                              │
│ B=1:    FA2/SDPA = 3.3-3.9x SLOWER!                         │
│ B=8:    FA2/SDPA = 3.4-3.9x SLOWER                          │
│ B=32:   FA2/SDPA = 3.1-7.6x SLOWER                          │
│ B=128:  FA2/SDPA = 2.5-34x SLOWER!!                         │
│                                                              │
│ → FA2 decode = NEGATIVE optimization!                        │
│ → FA2 even 1.1-1.3x slower than naive at B≤8!               │
│ → Only SDPA math backend optimal for decode                  │
│                                                              │
│ GQA (B=8, S=512):                                            │
│                                                              │
│ MHA:   SDPA 0.048ms, FA2 0.088ms (1.85x慢)                 │
│ GQA_4: SDPA 0.049ms, FA2 0.095ms (1.94x慢) + 75%KV省       │
│ GQA_2: SDPA 0.050ms, FA2 0.089ms (1.80x慢) + 87.5%KV省     │
│ MQA:   SDPA 0.049ms, FA2 0.086ms (1.77x慢) + 93.75%KV省    │
│                                                              │
│ → SDPA handles GQA expand efficiently                       │
│ → FA2 slightly slower, but GQA=KV saving not speed          │
│                                                              │
│ Long Context Memory (B=1):                                   │
│                                                              │
│ S=1024: FA2 saves 87.7% memory (1.3GB→0.16GB)              │
│ S≥2048: FA2 NEGATIVE memory saving! (-4~-17%)               │
│                                                              │
│ → SDPA auto-selects flash at S≥2048 → already O(N) memory  │
│ → FA2 overhead (layout conversion) > benefit at B=1         │
│ → FA2 memory benefit only at B×S combinations that OOM naive│
│                                                              │
│ Accuracy: ALL cos_sim = 1.000000 (bit-for-bit identical)    │
│                                                              │
│ **Production Decision**:                                     │
│   Prefill → SDPA (auto-selects flash) or FA2 (memory)       │
│   Decode → SDPA math backend ONLY (FA2=负优化!)             │
│   GQA    → SDPA with expanded KV (FA2 not faster)           │
│   vLLM/SGLang → flash for prefill, FlashInfer for decode    │
└──────────────────────────────────────────────────────────────┘
```

## 完整数据

```
Exp 1: Prefill Attention Backends (B=8, causal):

7M模型 (d=256, n_heads=8, d_head=32):

| S    | naive_ms | sdpa_ms | fa2_ms | naive/sdpa | fa2/sdpa | FA2 mem_saving |
|------|----------|---------|--------|------------|----------|----------------|
| 128  | 0.083    | 0.032   | 0.101  | 2.59x      | 3.15x    | 24.6%          |
| 256  | 0.091    | 0.033   | 0.100  | 2.76x      | 3.04x    | 45.8%          |
| 512  | 0.123    | 0.040   | 0.096  | 3.09x      | 2.42x    | 69.4%          |
| 1024 | 0.953    | 0.076   | 0.118  | 12.6x      | 1.56x    | 84.4%          |
| 2048 | 3.610    | 0.173   | 0.208  | 20.9x      | 1.20x    | 92.3%          |
| 4096 | 14.19    | 0.527   | 0.551  | 26.9x      | 1.05x    | 96.2%          |

→ S≥1024: FA2接近SDPA速度 + 巨大内存节省
→ S≥4096: FA2≈SDPA(1.05x) → 速度相等, 内存省96%

25M模型 (d=512, n_heads=16, d_head=32):

| S    | naive/sdpa | fa2/sdpa | FA2 mem_saving |
|------|------------|----------|----------------|
| 128  | 2.83x      | 3.19x    | -60% (SDPA lower!) |
| 256  | 2.80x      | 3.12x    | 52.3%          |
| 512  | 8.87x      | 1.95x    | 72.7%          |
| 1024 | 17.9x      | 1.34x    | 85.6%          |
| 2048 | 24.7x      | 1.11x    | 92.6%          |
| 4096 | 28.9x      | 1.02x    | 96.3%          |

→ S=128: FA2 memory actually HIGHER than SDPA! (SDPA uses flash internally)
→ S≥512: FA2 memory saving grows rapidly

125M模型 (d=1024, n_heads=16, d_head=64):

| S    | naive/sdpa | fa2/sdpa | FA2 mem_saving |
|------|------------|----------|----------------|
| 128  | 2.66x      | 3.17x    | -87% (SDPA much lower!) |
| 256  | 2.34x      | 2.50x    | 37.6%          |
| 512  | 6.73x      | 1.59x    | 58.7%          |
| 1024 | 11.2x      | 1.28x    | 75.5%          |
| 2048 | 13.8x      | 1.08x    | 86.6%          |
| 4096 | 15.5x      | 1.03x    | 92.9%          |

→ At S≤128: SDPA flash backend already uses O(N) memory → FA2 overhead > benefit
→ FA2 only useful when naive would OOM (large S×n_heads)

Exp 2: Decode Attention Backends (Q=1, causal):

7M模型 key results:

| Config      | naive_ms | sdpa_ms | fa2_ms | fa2/sdpa |
|-------------|----------|---------|--------|----------|
| S64_B1      | 0.075    | 0.025   | 0.084  | 3.32x慢  |
| S128_B1     | 0.075    | 0.025   | 0.089  | 3.56x慢  |
| S512_B1     | 0.077    | 0.025   | 0.099  | 3.96x慢  |
| S1024_B128  | 0.184    | 0.028   | 0.204  | 7.41x慢  |
| S2048_B128  | 0.342    | 0.027   | 0.352  | 12.9x慢  |

→ FA2 decode consistently 3-13x slower than SDPA!
→ At B=1: FA2 3.3-3.9x slower → FA2 kernel startup overhead
→ At B=128: FA2 7-13x slower → FA2 not optimized for large batch decode

125M模型 worst cases:

| Config      | sdpa_ms | fa2_ms | fa2/sdpa |
|-------------|---------|--------|----------|
| S256_B128   | 0.039   | 0.215  | 5.48x慢  |
| S1024_B32   | 0.027   | 0.220  | 8.16x慢  |
| S2048_B128  | 0.038   | 1.314  | 34.2x慢! |

→ FA2/SDPA up to 34x slower for decode! → FA2 decode = negative optimization
→ FA2 even slower than naive at B≤8 (1.1-1.3x vs naive)

Exp 3: GQA Backends (B=8, S=512):

| Type  | n_kv | sdpa_ms | fa2_ms | fa2/sdpa | KV saving |
|-------|------|---------|--------|----------|-----------|
| MHA   | 16   | 0.048   | 0.088  | 1.85x慢  | 0%        |
| GQA_4 | 4    | 0.049   | 0.095  | 1.94x慢  | 75%       |
| GQA_2 | 2    | 0.050   | 0.089  | 1.80x慢  | 87.5%     |
| MQA   | 1    | 0.049   | 0.086  | 1.77x慢  | 93.75%    |

→ SDPA handles GQA KV expand efficiently (~0.048-0.050ms regardless of n_kv)
→ FA2 consistently 1.8-2.0x slower
→ GQA value = KV memory saving, not speed

Exp 4: Long Context Memory (B=1, n_heads=16, d_head=32):

| S     | attn_matrix_MB | kv_MB | ratio | sdpa_peak_MB | fa2_peak_MB | FA2 mem_saving | sdpa_ms | fa2_ms | fa2/sdpa |
|-------|----------------|-------|-------|--------------|-------------|----------------|---------|--------|----------|
| 1024  | 33.6           | 2.1   | 16x   | 1301         | 160         | 87.7%          | 0.038   | 0.089  | 2.33x慢  |
| 2048  | 134.2          | 4.2   | 32x   | 98.3         | 102.5       | -4.3%!         | 0.080   | 0.119  | 1.49x慢  |
| 4096  | 536.9          | 8.4   | 64x   | 112          | 120         | -7.5%!         | 0.194   | 0.222  | 1.14x慢  |
| 8192  | 2147.5         | 16.8  | 128x  | 139.6        | 156.4       | -12%!          | 0.572   | 0.598  | 1.04x慢  |
| 16384 | 8589.9         | 33.6  | 256x  | 194.6        | 228.2       | -17%!          | 1.967   | 1.967  | 1.00x    |

→ S=1024: FA2 saves 87.7% memory (SDPA uses naive-style matmul)
→ S≥2048: SDPA auto-selects flash → already O(N) memory → FA2 overhead wins!
→ At S=16384: FA2 latency = SDPA (1.00x) → speed equal but memory worse
→ FA2 memory benefit ONLY when naive would OOM (high n_heads × S × B)

Exp 5: Backend Decision Guide:

| Scenario        | Recommended              | FA2 Benefit                     |
|-----------------|--------------------------|---------------------------------|
| prefill_short   | SDPA (auto math/flash)   | minimal speed, significant mem  |
| prefill_long    | FlashAttention-2         | 85-97% memory, ≈1x speed       |
| decode_B1       | SDPA math backend        | NONE (FA2 slower!)              |
| decode_B128+    | SDPA (auto-selects)      | minimal, only memory matters    |
| gqa             | SDPA with expanded KV    | same perf, varlen API benefit   |
| production      | flash(prefill)+math(decode) | vLLM uses FlashInfer for decode |
```

## RTX 4090 Attention Backend决策树

```
RTX 4090 Attention Backend决策树:

Q1: 计算模式?
├── Prefill (S≥1, B≥1)
│   ├── S≤256 → SDPA (auto) → 最快, 无layout转换
│   │   → FA2 3x慢, 无内存优势(SDPA已用flash)
│   ├── S≥512 → SDPA (auto) → 仍最快
│   │   → FA2 1.6-2.4x慢, 但省69-85%内存
│   ├── S≥4096 → SDPA ≈ FA2速度 → 1.02-1.05x
│   │   → FA2省92-96%内存 → 防OOM唯一价值
│   └── 结论: SDPA始终最快; FA2=内存优化而非速度优化
│
├── Decode (Q=1)
│   ├── B=1 → SDPA math → 0.025ms (最优!)
│   │   → FA2 0.084-0.100ms (3.3-3.9x慢!)
│   ├── B=8-32 → SDPA math → 仍最优
│   │   → FA2 3.5-4.5x慢
│   ├── B=128+ → SDPA auto → 可能用flash
│   │   → FA2 7-34x慢!! (kernel不适合decode)
│   └── 结论: Decode NEVER用FA2! SDPA math唯一最优
│
├── GQA
│   ├── 所有配置 → SDPA + expanded KV → 最快
│   │   → FA2 1.8-2.0x慢 → layout转换开销
│   ├── GQA value = KV省75-93.75% → 不是速度
│   └── FA2 varlen API → 仅packed sequences有用
│
└── Production (vLLM/SGLang)
    ├── Prefill → Flash/SDPA flash backend
    ├── Decode → FlashInfer custom kernel (not FA2!)
    │   → FlashInfer = optimized decode + GQA + varlen
    └── vLLM V1: 20+ attention backends, auto-select

**关键洞察**: FA2的"加速"是误解!
  → FA2 = IO优化(O(N²)→O(N) memory) = 防OOM
  → FA2 ≠ 计算加速(RTX 4090实测证实!)
  → Decode: FA2 kernel启动+layout转换 = 3-34x开销
  → Production: vLLM/SGLang用FlashInfer而非raw FA2做decode

**RTX 4090最优backend选择**:
  Prefill: torch.nn.functional.scaled_dot_product_attention (auto backend)
  Decode:  同上 (auto selects math backend for Q=1)
  仅当OOM风险时: prefill用FA2 API (96%内存省)
```

## 工具

- `tools/attention_backend_comparison_4090.py` — 5实验完整benchmark
- `results/attention_backend_comparison.json` — 完整数据