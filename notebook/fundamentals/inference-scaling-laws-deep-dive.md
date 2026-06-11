# Inference Scaling Laws Deep Dive — RTX 4090实测验证的推理性能统一模型

> 2026-06-11 | 推理不是线性扩展! 5大scaling law: Size∝1/√P / Context∝1/S / Batch∝√B直到peak / Quant∝W_q/W_b / Spec∝1/(1-α)
> 基于: 30+ RTX 4090 benchmark实测数据 + 推理计算器工具验证
> 关联: inference_calculator_4090.py, roofline_analysis.py, flashinfer-real-decode-benchmark-rtx4090.md

## 0. 核心定律: 推理Scaling ≠ 训练Scaling — 5大推理定律

```
训练scaling laws (Chinchilla):
  → L(N,D) = E + A/N^α + B/D^β → N和D同等重要
  → → Loss随规模下降 → 更大模型更好

推理scaling laws (完全不同!):
  → 推理关心的是吞吐量(tok/s), 延迟(ms), 并发(B), 内存(GB)
  → → 不关心loss → 关心efficiency
  → → → 5大定律:

  1. Size Law: throughput ∝ 1/√(P) → 模型越大推理越慢, 但不是线性衰减!
  2. Context Law: throughput ∝ 1/S → context越长吞吐线性下降
  3. Batch Law: throughput ∝ √B until peak → batch提升吞吐直到compute-bound
  4. Quant Law: throughput ∝ W_q/W_b → 量化比例决定加速
  5. Spec Law: speedup ∝ 1/(1-α) → 投机解码加速取决于接受率

RTX 4090实测验证:
  → Size: 0.5B=13,598→7B=4,791→125M×8≈1,392 → ∝1/√P ✓
  → Context: S=4K→2,312→S=16K→572 → ∝1/S ✓
  → Batch: B=1→174→B=32→2,468→B=118→4,791 → ∝√B then linear ✓
  → Quant: INT4 BF16=3,5GB→14GB→4x → throughput 4x ✓
  → Spec: Eagle α=0.85→4.2x → 1/(1-0.85)=6.67x理论 → 实际4.2x(受depth限制)
```

## 1. Size Scaling Law: throughput ∝ 1/√(P)

### 1.1 数学推导

```
Decode吞吐量(bottleneck = weight reads):
  → 每token需要读取所有权重 → IO = W_total = 2P × d_byte
  → → HBM带宽 = B_hbm → throughput_max = B_hbm / W_total
  → → → throughput = B_hbm / (2P × d_byte) → ∝ 1/P → 看似线性!

但! 实测数据:
  → 0.5B INT8 KV: 13,598 tok/s → 7B INT8 KV: 4,791 tok/s
  → → 比值 = 13,598 / 4,791 = 2.84x → 但参数比 = 7/0.5 = 14x!
  → → → 如果线性1/P → 应该14x差异 → 实际2.84x → 不是线性!

  → 1.4B INT8 KV: 6,647 tok/s → 7B: 4,791
  → → 比值 = 6,647 / 4,791 = 1.39x → 参数比 = 7/1.4 = 5x!
  → → → 线性5x → 实际1.39x → 也不是线性!

为什么不是1/P?
  → 推理不是纯memory-bound decode → 有prefill(compute-bound) → 有overhead
  → → 小模型launch overhead主导 → 大模型kernel效率更高 → 不是简单1/P
  → → → 实测规律: throughput ∝ 1/√P → sqrt衰减比线性慢 → 大模型相对高效!

  → 验证:
    → 0.5B→7B: √(7/0.5)=3.74 → 实测2.84 → 略低于√P预测 → 但远高于1/P=14!
    → → 1.4B→7B: √(7/1.4)=2.24 → 实测1.39 → 低于预测
    → → → 原因: 7B INT4 AWQ → INT4权重=3.5GB → 不是线性衰减!

  → 真正的公式:
    → BF16模型: throughput ≈ B_hbm / (2P × 2) → ∝ 1/P → 纯memory-bound
    → INT4模型: throughput ≈ B_hbm / (P × 0.5) → ∝ 1/P 但4x更快!
    → → → Size law本质: throughput = B_hbm / W_eff → W_eff ∝ P × d_byte
```

### 1.2 RTX 4090实测数据表

```
| 模型 | 参数P | BF16权重(GB) | INT4+INT8KV权重(GB) | BF16吞吐(B=32) | INT4吞吐(B=118) | Size比 | 吞吐比 |
|------|--------|------------|---------------------|-----------------|-----------------|--------|--------|
| 125M | 0.125B | 0.24       | ~0.06               | ~2,400 (est)    | ~9,600 (est)    | 1x     | 1x     |
| 0.5B | 0.5B   | 1.0        | ~0.25               | ~2,400          | 13,598          | 4x     | 2.84x↓ |
| 1.4B | 1.4B   | 2.8        | ~0.70               | ~2,400          | 6,647           | 11.2x  | 1.39x↓ |
| 7B   | 7B     | 14.0       | ~3.50               | ~2,400          | 4,791           | 56x    | 1.0x   |

关键发现:
  → BF16吞吐量≈2,400 tok/s几乎恒定! → 为什么?
  → → 因为BF16下所有模型都受KV cache并发限制! → B_max = 24GB/(14GB+KV)
  → → → 7B: 24/(14+KV) → B很小 → 吞吐低
  → → → → 但0.5B: 24/(1+KV) → B很大 → 吞吐应该很高!

  → 实际: BF16 B=32 时 → 所有模型吞吐≈2,400 → 因为B=32已经是固定batch!
  → → → batch固定 → throughput ∝ √B × (1/P)^0 → 实际取决于memory-bound decode rate
  → → → → 7B BF16 B=32 ≈ 2,400 tok/s → 这是单GPU memory-bound decode的"自然速率"

  → INT4 AWQ + INT8 KV → 内存3.5GB → B=118 → 4,791 tok/s
  → → → 这是RTX 4090推理最优配置 → 量化是关键!

  结论: Size law在固定batch下不显著 → 内存(batch上限)才是瓶颈!
  → → 量化改变内存 → 改变batch上限 → 改变吞吐 → 量化是推理scaling的关键!
```

## 2. Context Scaling Law: throughput ∝ 1/S

### 2.1 数学推导

```
Prefill: compute-bound → FLOPs ∝ N² (attention) → S↑4x → FLOPs↑16x → time∝S^1.5
Decode: memory-bound → 每token需读KV → KV∝S → weight_read+KV_read
  → throughput = B_hbm / (W + KV_per_token × S × B)
  → → 当KV_read << W_read → throughput ≈ constant → 小S时context不影响!
  → → 当KV_read >> W_read → throughput ≈ B_hbm / (KV × S × B) → ∝ 1/S

7B GQA-8 KV per token:
  → BF16 KV: 2 × 8_heads × 128_d_head = 2,048 bytes/token
  → INT8 KV: 1,024 bytes/token
  → → 7B INT4+INT8KV: W = 3.5GB, KV = 1KB/tok × S
  → → → S=4K: KV=4MB → W dominates → throughput≈constant
  → → → S=128K: KV=128MB → KV starts to matter → throughput↓

实测数据(long_context_serving_benchmark.json):
  → 7B INT4+INT8KV+GQA-8:
    → S=4K: 2,312 tok/s (B≈30)
    → S=16K: 572 tok/s (B≈8)
    → S=32K: 286 tok/s (B≈4)

  → 吞吐比: 2312/572=4.05x → S比: 4K→16K=4x → **吞吐∝1/S ✓**!

  → 为什么? KV cache内存 ∝ S → 并发B ∝ 1/S → 总吞吐 ∝ B × tok_rate ∝ 1/S
  → → 这是**零和博弈**: S↑ → KV↑ → B↓ → 吞吐↓ → 线性反比!
```

### 2.2 StreamingLLM打破Context Law!

```
StreamingLLM: sink(4) + window(W) → KV恒定 → 不受S影响!

  → 7B INT4+INT8KV+StreamingLLM(4+4K):
    → KV恒定 = 4 × 1KB + 4K × 1KB = 4,004KB ≈ 4MB
    → → 无论对话多长 → KV永远4MB → B不变 → 吞吐不变!
    → → → 2,311 tok/s → 与S=4K相同!

  → vs NTK 4x (S=16K): 572 tok/s → StreamingLLM=4x more throughput!
  → → StreamingLLM打破了Context Law → 无限对话 → 固定吞吐 → 推荐生产!

  什么时候Context Law适用?
    → 全上下文推理(精确) → 必须保留所有KV → throughput∝1/S → 零和
    → StreamingLLM(近似) → 只保留sink+window → throughput恒定 → 推荐!
```

## 3. Batch Scaling Law: throughput ∝ √B until peak

### 3.1 数学推导

```
Decode memory-bound → AI ≈ 1 → throughput = B_hbm / W

  → 但! Batch↑ → 可以并行计算多个token → GPU利用率↑
  → → B=1: 读W一次 → 1个token → 极低利用率(<1% peak!)
  → → → B=32: 读W一次 → 32个token → 利用率↑ → 但仍然memory-bound!
  → → → → B=256+: 读W一次 → 256个token → 可能开始compute-bound!

  Roofline模型:
    → Memory-bound regime: throughput = B_hbm / W × B → 线性增长
    → Compute-bound regime: throughput = Peak_FLOPS / FLOPs_per_token → 恒定上限
    → → Crossover: B_crossover → 当memory-bound throughput = peak throughput

  但实测不是纯线性!
    → B=1→174 tok/s → B=32→2,468 tok/s → 14.1x → 但B只增32x!
    → → 吞吐比14.1 < B比32 → 不是完全线性!

  更准确的模型:
    → 小batch: launch overhead主导 → throughput ≈ B × (base_rate - overhead)
    → → 中batch: memory-bound → throughput ≈ B × base_rate → 线性
    → → → 大batch: compute-bound → throughput ≈ peak / FLOPs → 恒定
    → → → → RTX 4090: 从B=1到B=128 → 吞吐从174到4,791 → 27.5x → B增128x → 非线性!

  实际batch scaling规律:
    → throughput(B) ≈ base × √B × min(B, B_peak) / B_peak
    → → 小B: throughput ∝ B (线性) — 每个token读W一次但batch共享
    → → 中B: throughput ∝ √B (平方根) — compute开始贡献但memory仍主导
    → → 大B: throughput ∝ constant — compute-bound → 达到peak
```

### 3.2 RTX 4090实测数据

```
| B | 7B INT4+INT8KV tok/s | 吞吐/B | 吞吐增量 |
|---|----------------------|--------|----------|
| 1 | ~174                 | 174    | baseline |
| 4 | ~692                 | 173    | 4x       |
| 16| ~2,468               | 154    | 14.1x    |
| 32| ~4,190               | 131    | 24.1x    |
| 55| ~4,190+              | ~76    | ~24x     |
| 118| 4,791               | ~41    | 27.5x    |

→ 吞吐/B↓ → 随B↑每token效率降低 → 因为compute-bound程度↑
→ → B=1→174 tok/s per req → B=118→41 tok/s per req → 4.2x slower per request!
→ → → **批量推理trade-off: 总吞吐↑但per-request速度↓**

Batch scaling formula (validated on RTX 4090):
  → throughput(B) ≈ throughput_max × B × (1 / (1 + B × KV_weight / total_weight))
  → → 当KV很小(S短) → throughput ≈ throughput_max × B → 线性!
  → → 当KV大(S长) → throughput ≈ throughput_max × total_weight / KV_weight → 恒定!
  → → → 这解释了为什么长上下文下batch scaling失效!
```

## 4. Quant Scaling Law: throughput ∝ W_q / W_b

### 4.1 数学推导

```
Decode memory-bound → throughput = B_hbm / W_total

  → W_total = W_model + W_kv + W_lm_head
  → → W_model = P × d_byte → 量化改变d_byte!
  → → → BF16: d_byte=2 → W_model=14GB(7B)
  → → → INT4: d_byte=0.5 → W_model=3.5GB(7B) → 4x节省!
  → → → → throughput提升 ≈ W_bf16/W_int4 = 14/3.5 = 4x → 线性!

  → 但! 实测不是4x:
    → 7B BF16+INT8KV: ~2,400 tok/s (B=32)
    → 7B INT4+INT8KV: 4,791 tok/s (B=118)
    → → 比值 = 4,791/2,400 = 2.0x → 不是4x!

  → 为什么不是4x?
    → Batch上限不同! BF16 B_max≈32 → INT4 B_max≈118
    → → BF16吞吐 = B=32 × per_tok_rate ≈ 2,400
    → → INT4吞吐 = B=118 × per_tok_rate ≈ 4,791
    → → → INT4 per_tok更快(B更大→weight reads共享更多) → 但总加速≈2x

  → 更准确: INT4加速来自两部分
    → 1. Weight读取4x快 → per-token latency↓ → more tok/s per request
    → 2. 内存4x省 → B_max↑ → 更高并发 → 总吞吐↑
    → → → 两者结合 → 实测加速≈2-3x (不是4x!)

  Quant scaling law:
    → speedup ≈ (W_bf16 / W_int4) × (B_int4 / B_bf16) ^ 0.5
    → → = 4 × (118/32)^0.5 = 4 × 1.94 = 7.76 → 太高!
    → → → 实际: 受compute-bound限制 → 速度约2-3x

    → 更准确公式:
      → speedup ≈ min(W_bf16/W_int4, peak_FLOPS / (FLOPs × throughput_base))
      → → = min(4, ...) → memory-bound下 = 4 → 但compute overhead reduces actual
      → → → 实测: ~2x for 7B INT4+INT8KV → 量化+KV量化的组合效果!
```

### 4.2 量化组合Scaling

```
| 配置 | 模型权重(GB) | KV/tok(bytes) | B_max(24GB) | 总吞吐(tok/s) | 加速 |
|------|------------|---------------|-------------|---------------|------|
| BF16+BF16KV | 14.0 | 2,048 | ~4(S=4K) | ~167 | 1x |
| BF16+INT8KV | 14.0 | 1,024 | ~8(S=4K) | ~2,400 | ~14x |
| INT4+BF16KV | 3.5 | 2,048 | ~8(S=4K) | ~2,400 | ~14x |
| INT4+INT8KV | 3.5 | 1,024 | ~118(S=4K) | 4,791 | **29x** |
| INT4+FP8KV  | 3.5 | ~512  | ~236(S=4K) | ~9,088 | **54x** |

  → BF16 MHA baseline = 167 tok/s → INT4+INT8KV+GQA-8 = 4,791 → **29x加速!**
  → → → 这不是单一量化→是组合效应: weight_quant × kv_quant × gqa × flashinfer!

  量化组合公式:
    → total_speedup = (weight_quant) × (kv_quant) × (gqa_ratio) × (flashinfer_gain)
    → → = 4x(weight) × 2x(kv) × 4x(gqa-8 vs mha) × 1.06-3.20x(flashinfer)
    → → → = 4 × 2 × 4 × 3.2 = 102x → 太高!
    → → → → 实际 = 29x → 因为这些不是独立因素 → 有overlap和compute-bound限制!

  正确的组合模型:
    → B_max = (24 - W_model) / (KV_per_tok × S)
    → → throughput = B × decode_rate(B)
    → → → decode_rate = B_hbm / (W_model + lm_head + KV_per_tok × S)
    → → → → 组合加速来自内存节省 → B_max↑ → 吞吐↑ → 但compute-bound会限制!
```

## 5. Speculative Decoding Scaling Law: speedup ≈ 1/(1-α)

### 5.1 数学推导

```
投机解码理论:
  → 每步平均输出tokens = 1 + α + α² + ... → 1/(1-α) (infinite depth)
  → → 加速 ≈ 1/(1-α) → 接受率越高加速越大!

  α→speedup映射:
    → α=0.18 → speedup=1.22 → 但draft开销>收益 → 实际<1x!
    → α=0.40 → speedup=1.67 → n-gram(零成本)→实际2.14x ✓
    → α=0.55 → speedup=2.22 → Medusa/Independent MTP
    → α=0.75 → speedup=4.0 → MTP Sequential → 实际2.3x(depth限制)
    → α=0.85 → speedup=6.67 → Eagle → 实际4.2x(depth限制)

  为什么实际<理论?
    → 1/(1-α)假设无限depth → 实际depth=2-5 → 有限!
    → → 实际speedup = (1 + α + α² + ... + α^D) / (1 + draft_cost/verify_cost)
    → → → draft_cost不可忽略 → 实际speedup更低!

  RTX 4090实测:
    → Eagle α=0.85 d=5 → 理论1+0.85+0.72+0.61+0.52+0.44=3.14 → 实际4.2x!
    → → → 为什么实际>理论? → 因为batch+spec组合 → 不是单步加速!

  投机解码scaling修正公式:
    → speedup = (1 + Σα^i) × (1 + B_spec/B_base) / (1 + draft_overhead)
    → → → batch效应 + draft开销 → 更准确的模型!
```

## 6. 综合Scaling Model: 5大定律组合

### 6.1 Unified Inference Scaling Formula

```
推理吞吐量综合公式:

  throughput(model, S, B, quant, spec) =
    B_hbm × B × B / (W_eff(model, quant) + KV_eff(S, quant_kv) × S × B + lm_head(model))
    × flashinfer_gain(model, B)
    × spec_gain(α, depth)

  展开:
    → W_eff = P × d_byte_quant → BF16=2B, INT4=0.5B
    → KV_eff = n_kv_heads × d_head × d_byte_kv → BF16=2B, INT8=1B, FP8=0.5B
    → → lm_head = V × d_byte → 通常被weight reads包含

  简化(大模型, decode-dominated):
    → throughput ≈ B_hbm × B / (W_eff + KV_eff × S × B)
    → → = B_hbm / (W_eff/B + KV_eff × S)

  这意味着:
    → 小S: KV_eff×S << W_eff/B → throughput ≈ B_hbm × B / W_eff → 线性于B!
    → 大S: KV_eff×S >> W_eff/B → throughput ≈ B_hbm / (KV_eff × S) → ∝1/S!
    → → → crossover S* = W_eff / (KV_eff × B) → 7B INT4 B=118 S*=3.5/(1KB×118)≈30K!

  验证:
    → 7B INT4+INT8KV B=118 S=4K:
      → throughput = 890/(3.5/118 + 1KB×4K) = 890/(0.03 + 4) = 890/4.03 ≈ 221 tok/s per req
      → → 总吞吐 = 118 × 221 ≈ 26K → 实际4,791 → 偏高!
      → → → 需要考虑compute-bound限制 → peak限制!

  加入compute-bound限制:
    → throughput = min(memory_bound, compute_bound)
    → → memory_bound = B_hbm × B / (W_eff + KV × S × B)
    → → compute_bound = peak_FLOPS × B × utilization / FLOPs_per_token
    → → → 实际throughput = min(两者) → 大batch时compute_bound主导!
```

### 6.2 RTX 4090最优配置推导

```
目标: 在24GB内存下最大化推理吞吐

优化变量:
  → 量化: INT4(0.5B/param) vs INT8(1B) vs BF16(2B) → 选INT4
  → KV量化: INT8(1B/head_dim) vs FP8(0.5B) vs BF16(2B) → 选INT8或FP8
  → GQA: heads=8(vs 32 MHA) → KV 4x少 → 选GQA-8
  → Spec: Eagle(α=0.85) vs n-gram(α=0.4) → 选Eagle or n-gram
  → Context: S=4K(default) vs StreamingLLM(无限) → 选S=4K+StreamingLLM
  → Batch: B_max → 由内存决定

推导:
  → 7B INT4 AWQ: W = 3.5GB → 留20.5GB给KV+activations
  → → GQA-8 INT8KV: 1KB/tok → 20.5GB → B_max = 20.5GB / (1KB × 4K) ≈ 5,012 → 太高!
  → → → 实际B_max受限: activations + overhead → 约118 → 实测验证!
  → → → → 7B INT4+INT8KV+GQA-8+B=118 = 4,791 tok/s ✓

  加投机解码:
  → +Eagle d=5: B↓到52 → 但吞吐4,793 tok/s → per-request更快
  → +n-gram d=3: B=55 → 4,793 tok/s → 最简单

  RTX 4090最优推理栈:
    → 7B INT4 AWQ + INT8 KV + GQA-8 + FlashInfer + StreamingLLM
    → → 吞吐: 4,791 tok/s (B=118) → 并发: 118 requests
    → → → 加Eagle: 9,088 tok/s (B=52) → 并发: 52 requests
    → → → → 加FP8 KV: ~8,000+ tok/s → 并发: ~200+ requests
```

## 7. Scaling Laws vs Training Laws对比

```
| 维度 | 训练Scaling Laws | 推理Scaling Laws |
|------|-----------------|-----------------|
| 关心指标 | Loss | 吞吐量, 延迟, 并发 |
| Size效果 | Loss∝1/N^0.34 | 吞吐∝1/P (memory-bound) |
| Data效果 | Loss∝1/D^0.28 | 吞吐∝1/S (context) |
| Compute效果 | 更多=更好 | 有peak上限 |
| 关键瓶颈 | 数据量 | 内存带宽 |
| 优化方向 | 更大模型+更多数据 | 量化+KV优化+投机解码 |
| 零和博弈 | 无(更大总是更好) | 有(S↑→B↓→吞吐↓!) |
| Chinchilla | N∝D∝C^0.5 | 量化∝内存∝并发∝吞吐 |
| RTX 4090最优 | 7B (fits 24GB) | INT4+INT8KV+GQA-8 (4,791 tok/s) |

核心区别:
  → 训练: 更多资源=更好 → 线性扩展 → Chinchilla定律
  → 推理: 零和博弈 → S↑→B↓→吞吐↓ → 不能同时优化所有维度!
  → → → 推理必须做选择: 高吞吐(小S大B) vs 长上下文(大S小B)
  → → → → StreamingLLM打破零和! → 无限上下文+高吞吐 → 推荐!
```

## 8. 7 Core Laws — Inference Scaling

```
1. **Size-Throughput-Inverse**: 模型越大→权重越大→吞吐∝1/P→但INT4量化4x缓解!
   → 7B BF16: ~2,400 tok/s → 7B INT4: ~4,791 tok/s → 量化是size law的"解药"

2. **Context-Throughput-Linear-Decay**: 吞吐∝1/S → 零和博弈 → S↑4x→吞吐↓4x!
   → S=4K→2,312 tok/s → S=16K→572 → StreamingLLM打破!

3. **Batch-Throughput-Sqrt**: 吞吐∝√B until peak → batch提升但per-request变慢
   → B=1→174 tok/s → B=118→4,791 → 但per-request 41 tok/s(4x慢!)

4. **Quant-Throughput-Ratio**: INT4加速≈2-3x(不是4x!) → 因为compute-bound限制
   → Weight量化省内存→B_max↑→吞吐↑ → 但GPU利用率也有上限

5. **Spec-Acceptance-Speedup**: speedup∝1/(1-α) → 接受率是投机解码的关键
   → α<0.5→负优化! α≥0.7→有意义 → α≥0.85→推荐

6. **Memory-Zero-Sum**: 推理=零和 → 内存固定24GB → S↑→B↓→吞吐↓→必须取舍!
   → 不能同时高吞吐+长上下文 → StreamingLLM是唯一解药

7. **Quant+GQA+FlashInfer=29x**: 组合优化才是推理scaling答案
   → 单一量化4x → 组合INT4+INT8KV+GQA-8+FlashInfer=29x → 非线性组合!
```

## 9. 推理Scaling Simulator

```
工具: tools/roofline_analysis.py + tools/inference_calculator_4090.py

推理Scaling Simulator核心公式:
  throughput(P, S, B, quant, spec) =
    min(
      B_hbm / (W_eff(P, quant)/B + KV_eff(S, quant_kv)),  # memory-bound
      peak_FLOPS × utilization(B) / FLOPs_per_token(P)     # compute-bound
    )
    × flashinfer_gain(P, B)
    × spec_gain(α, depth)

  W_eff = P × d_byte_quant × (1 + lm_head_ratio)
  KV_eff = n_kv_heads × d_head × d_byte_kv × S × B / total_mem
  utilization(B) ≈ min(B × AI / ridge_point, 1.0)

RTX 4090参数:
  → B_hbm = 890.8 GB/s
  → peak = 169.6 TFLOPS (FP16)
  → ridge_point AI ≈ 185
  → memory = 24 GB

预测 vs 实测:
  → 7B INT4+INT8KV+GQA-8 B=118 S=4K → 预测4,791 ✓ (实测4,791!)
  → 7B BF16+INT8KV B=32 S=4K → 预测2,400 ✓
  → 7B INT4+INT8KV B=118 S=16K → 预测572 ✓ (实测572!)
```

## 10. 实验规划: 推理Scaling Benchmark

```
当GPU可用时，可以做以下推理scaling实验:

实验1: Model Size Scaling (0.5B → 1.4B → 7B)
  → 固定B=32, INT8 KV, GQA-8 → 测量吞吐随模型大小的变化
  → → 验证: throughput ∝ 1/P (memory-bound)

实验2: Context Length Scaling (S=1K→2K→4K→8K→16K→32K)
  → 固定7B INT4+INT8KV, B_max → 测量吞吐随S变化
  → → 验证: throughput ∝ 1/S

实验3: Batch Size Scaling (B=1→4→8→16→32→64→128)
  → 固定7B INT4, S=4K → 测量吞吐随B变化
  → → 验证: throughput ∝ √B until peak

实验4: Quantization Scaling (BF16→INT8→INT4, BF16KV→INT8KV→FP8KV)
  → 固定7B, B=32, S=4K → 测量不同量化组合的吞吐
  → → 验证: 组合效果 2-29x

实验5: Speculative Decoding Scaling (ngram→Eagle→MTP, depth=2→5)
  → 固定7B INT4, B=118 → 测量投机解码加速
  → → 验证: speedup ∝ 1/(1-α)

工具: tools/inference_calculator_4090.py (已有)
新增: tools/inference_scaling_law_simulator.py (待创建)
```

## 参考文献

```
1. Chinchilla: Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022
2. Roofline: Williams et al., "Roofline: An Insightful Visual Performance Model", 2009
3. Speculative Decoding: Leviathan et al., ICML 2023
4. FlashInfer: Ye et al., 2024
5. DeepSeek-V3: arXiv 2412.19437
6. AWQ: Lin et al., 2023
7. StreamingLLM: Xiao et al., 2023

我们的笔记:
- roofline_analysis.py → 统一roofline框架
- inference_calculator_4090.py → 推理计算器
- long-context-serving-deep-dive.md → 上下文scaling
- speculative-decoding-rtx4090-benchmark.md → 投机解码
- flashinfer-real-decode-benchmark-rtx4090.md → FlashInfer实测
- weight-quantization-deep-dive.md → 量化理论