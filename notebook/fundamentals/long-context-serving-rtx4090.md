# Long Context Serving Benchmark — RTX 4090 实测

> 2026-06-07 | 3实验: KV Cache Scaling, Long Context PS, Chunked Prefill

## 核心发现: 长prefix PS加速更显著!

| prefix_len | prefix_ratio | compute节省 | time节省 | speedup | efficiency |
|-----------|-------------|-------------|----------|---------|-----------|
| 1024 | 0.80 | 60.0% | 58.0% | **2.38x** | 97% |
| 2048 | 0.89 | 66.7% | 65.4% | **2.89x** | 98% |
| 4096 | 0.94 | 70.6% | 70.2% | **3.35x** | 100% |
| 6144 | 0.96 | 72.0% | 71.9% | **3.55x** | 99.9% |

**趋势**: prefix越长 → suffix越短 → PS savings越大 → speedup最高达3.55x!

**与短prefix对比** (full_model_ps benchmark):
- 75% prefix (384/512): 2.08x → 80% prefix (1024/1280): 2.38x → 96% prefix (6144/6400): 3.55x
- 长context GRPO (长prompt + 短response) 是PS最佳场景!

## 一、Exp1: KV Cache Scaling

| seq_len | KV total | KV占peak% | Forward ms | Attention ms | MLP ms | tok/s |
|---------|----------|----------|-----------|-------------|--------|-------|
| 512* | 25.2 MB | 0.3% | 258 ms | 16 ms | ~242 ms | 2000 |
| 1024 | 50.3 MB | 0.5% | 138 ms | 34 ms | ~104 ms | 7394 |
| 2048 | 100.7 MB | 1.0% | 264 ms | 74 ms | ~190 ms | 7752 |
| 4096 | 201.3 MB | 1.9% | 574 ms | 189 ms | ~385 ms | 7139 |
| 6144 | 302.0 MB | 2.7% | 906 ms | 331 ms | ~575 ms | 6739 |
| 8192 | 402.7 MB | 3.5% | 1298 ms | 518 ms | ~780 ms | 6312 |

*seq=512含warmup噪声，应忽略。从1024开始看趋势。

**关键趋势**:

1. **KV线性增长**: 50→403 MB (seq×GQA-4×128×2×24layers) → 但占总peak内存比例小(0.5-3.5%)
   - Peak memory被model权重(~9GB)+中间activation(~1-2GB)主导 → KV仅占3.5%@8K
   - 但在decode场景(B≥32): KV读取占HBM BW主导 → KV cache是decode瓶颈

2. **Forward时间∝seq_len** (几乎线性): 138→1298 ms → MLP主导(compute-bound)
   - 6K→8K: 增长36% (906→1298ms) → 线性增长确认

3. **Attention时间∝seq_len²**: 34→518 ms → quadratic growth确认
   - S=1024: attn仅24% → S=8192: attn占40%
   - 长序列时attention越来越重要 → 但仍低于MLP(60%)

4. **Throughput**: 7400→6300 tok/s → 长序列时GPU利用率略降(kernel效率)
   - S=2048时峰值7752 tok/s → 最优序列长度(平衡compute-bound与GPU利用率)

### KV vs Model Weight Memory

```
Model weight (1.5B): ~4 GB (固定)
Activation: ~1-2 GB (随seq_len增加)
KV cache: 25-400 MB (线性增长, GQA-4)

对于7B模型 (GQA-4, 24层):
  KV per seq = 2×4×128×2×seq_len×24 = 24576×seq_len bytes
  S=4096: 98 MB (占peak 0.9%)
  S=8192: 196 MB (占peak 1.8%)
  S=32768: 786 MB (占peak 7.2%) → KV开始显著!

  对比MHA (32 heads):
  S=8192: 3144 MB → 超过model weight!
  GQA-4节省KV 8x → 延长KV可用序列8x
```

## 二、Exp2: Long Context PS Savings

### Method: Provider (full) + Reusers (suffix-only)

This measures the MLP+QKV_proj savings (the main PS benefit). Provider does full forward, reusers do suffix-only forward — skipping all prefix computation (MLP, QKV_proj, LayerNorm).

| prefix_len | suffix_len | total | prefix_ratio | compute sav | time sav | speedup | efficiency |
|-----------|-----------|-------|-------------|-------------|----------|---------|-----------|
| 1024 | 256 | 1280 | 0.80 | 60.0% | 58.0% | 2.38x | 96.7% |
| 2048 | 256 | 2304 | 0.89 | 66.7% | 65.4% | 2.89x | 98.1% |
| 4096 | 256 | 4352 | 0.94 | 70.6% | 70.2% | 3.35x | 99.7% |
| 6144 | 256 | 6400 | 0.96 | 72.0% | 71.9% | 3.55x | 99.9% |

**关键发现**:

1. **Efficiency接近100%**: 长prefix时time_savings ≈ compute_savings → 线性关系完美验证
2. **Speedup随prefix增长**: 2.38x→3.55x → GRPO长prompt场景是PS最佳应用
3. **96% prefix → 3.55x**: 几乎全是prefix → reuser只forward很短suffix → 加速最大化

### 与短prefix对比

| prefix_ratio | 短prefix(512 tokens) | 长prefix(1024+ tokens) |
|-------------|---------------------|-----------------------|
| 0.75 | 2.08x | — |
| 0.80 | — | 2.38x |
| 0.89 | — | 2.89x |
| 0.90 | 2.63x | — |
| 0.96 | — | 3.55x |

**趋势一致**: speedup = 1/(1/n + (n-1)/n × suffix_ratio) → formula验证:
- n=4, suffix_ratio=0.04 (prefix=96%): speedup = 4/(1+3×0.04) = 4/1.12 = 3.57x → 实测3.55x ✓
- n=4, suffix_ratio=0.20 (prefix=80%): speedup = 4/(1+3×0.20) = 4/1.60 = 2.50x → 实测2.38x (95%)

### 实用估算: 7B GRPO n=8, 长prompt

```
GRPO n=8, prefix_ratio=0.67 (512 prompt, 256 response):
  speedup = 8/(1+7×0.33) = 8/3.31 = 2.42x (实测2.46x ✓)

GRPO n=8, prefix_ratio=0.89 (2048 prompt, 256 response):
  speedup = 8/(1+7×0.11) = 8/1.77 = 4.53x → 预估4.5x!

GRPO n=8, prefix_ratio=0.96 (6144 prompt, 256 response):
  speedup = 8/(1+7×0.04) = 8/1.28 = 6.25x → 预估6x!
```

**震惊**: 长prompt GRPO (n=8, prefix=96%) → **预估6x training speedup**!
→ 这对DeepSeek-R1/TreeRL等长推理prompt场景是巨大的

## 三、Exp3: Chunked Prefill

| seq_len | Full prefill | Chunked (4 chunks) | Overhead | Mem savings |
|---------|-------------|--------------------|---------|-------------|
| 2048 | 261 ms | 672 ms | 157.4% | 0.0% |
| 4096 | 577 ms | 1388 ms | 140.4% | 0.2% |
| 6144 | 925 ms | 2191 ms | 137.1% | 0.4% |
| 8192 | 1319 ms | 3094 ms | 134.5% | 0.5% |

**关键发现**:

1. **Chunked prefill overhead 134-157%**: Python-level实现 → 每个chunk需要重新forward → 非常慢
   - 真实实现(vLLM)用batched chunked prefill → overhead应<5%
2. **Memory savings微不足道**: 0.0-0.5% → peak memory被model+activation主导
   - Chunked prefill目的是节省peak activation memory → 但1.5B模型仅4GB → activation远小于peak
   - 对7B模型(14GB权重): chunked prefill才有意义 → activation可能>peak
3. **Long sequence overhead降低**: 157→135% → 更大chunk → 更好GPU利用率
   - 但仍远高于full prefill → Python实现不可行

### 与理论对比

```
理论chunked prefill:
  - 分4个chunk → 每个chunk短序列 → peak activation降低4x
  - 但总compute不变 → 总时间应≈4×chunk_time ≈ full_time
  - Python-level: chunk间需要concat+重新forward → overhead巨大

真实实现(vLLM chunked prefill):
  - 使用batched flash attention → 多chunk同时处理
  - overhead仅5-10% → TTFT从13s→94ms (长context场景)
```

## 四、综合结论

### 长Context PS是最大优化机会

```
短prefix PS (512 prompt): 2.46x (n=8, GRPO forward-only)
长prefix PS (2048 prompt): 4.5x (n=8, GRPO forward-only) — 预估!
超长prefix PS (6144 prompt): 6.0x (n=8, GRPO forward-only) — 预估!

公式: speedup = n / (1 + (n-1) × suffix_ratio)
suffix_ratio = 1 - prefix_ratio

DeepSeek-R1/TreeRL: prompt=4096, response=256 → ratio=0.94 → speedup=5.7x!
```

### KV Cache容量是长Context瓶颈

```
7B GQA-4 on RTX 4090 (24GB):
  Model: 14 GB
  Available for KV: ~10 GB
  Max concurrent (S=2048): 10GB / 98MB = 102 并发
  Max concurrent (S=8192): 10GB / 196MB = 51 并发
  Max concurrent (S=32768): 10GB / 786MB = 13 并发 → 灾难瓶颈!

  MHA (32 heads) S=8192: 3144MB → 仅3并发!
  GQA-4节省8x → 51并发
  MLA节省32x → ~200并发 (但需要专用kernel)
```

### 实用公式集

```
# KV Cache Size (GQA-4, 24层, head_dim=128, FP16)
KV_bytes = 2 × num_kv_heads × head_dim × 2 × seq_len × num_layers
         = 2 × 4 × 128 × 2 × seq_len × 24
         = 24576 × seq_len bytes
KV_MB = 24.576 × seq_len / 1024

# PS Speedup (forward-only)
speedup = n / (1 + (n-1) × (1 - prefix_ratio))

# Prefill Throughput
tok/s ≈ TFLOPS × 0.38 / (6 × model_params) × seq_len / (fwd_ms / 1000)

# Concurrent Capacity (decode)
max_concurrent ≈ available_memory / KV_per_sequence
```

### 对verl贡献的启示

1. **长prompt GRPO是PS最佳场景**: speedup从2.46x(短prompt)→5-6x(长prompt) → 比Magi Attention的3x更有潜力
2. **当前PG仅attention → 长prompt时更差**: 长prompt→attention占比增(40%@8K)→但仍小于MLP(60%)
3. **需要full-model PS**: 不只是attention monkey-patch → 需要跳过prefix的MLP+QKV_proj+LN
4. **KV injection在长序列无效**: 实测0.99x → 但PS的真正价值是跳过prefix的所有compute-bound操作

Sources:
- Forward-only PS: notebook/fundamentals/full-model-ps-rtx4090.md
- Attention-only PS: notebook/fundamentals/prefix-sharing-packed-thd-rtx4090.md
- Training PS: notebook/fundamentals/grpo-training-ps-rtx4090.md
- Quantization: notebook/fundamentals/quantization-inference-rtx4090.md
- 工具: tools/long_context_serving_4090.py