# Decode Latency Breakdown RTX 4090 实测: 瓶颈随Batch动态转移

> 2026-06-07 | RTX 4090 decode per-layer component breakdown benchmark

## 核心数据

### Per-Layer Breakdown — MHA vs GQA-8

**B=1, S=2048** (single request decode):
| Component | MHA (ms) | GQA-8 (ms) | MHA% | GQA% |
|-----------|----------|------------|------|------|
| Weight GEMM | 0.443 | 0.399 | **87%** | 84% |
| KV cache read | 0.038 | 0.010 | 7.5% | 2.0% |
| Attention (SDPA) | 0.028 | 0.064 | 5.5% | 13.5% |
| **Per-layer total** | 0.510 | 0.473 | | |
| 32 layers + sampling | 16.3ms | 15.1ms | | |
| **Throughput** | 61 tok/s | 66 tok/s | | **1.08x** |

**B=8, S=2048**:
| Component | MHA (ms) | GQA-8 (ms) | MHA% | GQA% |
|-----------|----------|------------|------|------|
| Weight GEMM | 0.394 | 0.383 | 40% | 33.6% |
| KV cache read | 0.306 | 0.077 | **31%** | 6.7% |
| Attention (SDPA) | 0.289 | 0.679 | **29%** | 59.7% |
| Per-layer total | 0.988 | 1.138 | | |
| 32 layers + sampling | 31.6ms | 36.4ms | | |
| Throughput | 253 tok/s | 220 tok/s | | **0.87x** (GQA更慢!) |

**B=32, S=2048**:
| Component | MHA (ms) | GQA-8 (ms) | MHA% | GQA% |
|-----------|----------|------------|------|------|
| Weight GEMM | 0.393 | 0.388 | 14% | 11.6% |
| KV cache read | 1.224 | 0.306 | **44.5%** | 9.1% |
| Attention (SDPA) | 1.134 | 2.661 | **41.2%** | 79.3% |
| Per-layer total | 2.751 | 3.355 | | |
| 32 layers + sampling | 88.1ms | 107.4ms | | |
| Throughput | 363 tok/s | 298 tok/s | | **0.82x** |

**B=128, S=2048**:
| Component | MHA (ms) | GQA-8 (ms) | MHA% | GQA% |
|-----------|----------|------------|------|------|
| Weight GEMM | 0.477 | 0.443 | 4.8% | 3.6% |
| KV cache read | 4.897 | 1.224 | **49.6%** | 10% |
| Attention (SDPA) | 4.509 | 10.547 | **45.6%** | 86.4% |
| Per-layer total | 9.884 | 12.214 | | |
| 32 layers + sampling | 316ms | 391ms | | |
| Throughput | 405 tok/s | 327 tok/s | | **0.81x** |

### KV vs Weight Bottleneck — Seq Length (B=32)

| SeqLen | KV/step(MB) | Weight(MB) | KV% | Weight% | KV>Weight? |
|--------|-------------|-----------|-----|---------|------------|
| 256 | 4295 | 14000 | 23.5% | 76.5% | NO |
| 512 | 8590 | 14000 | 38% | 62% | NO |
| 1024 | 17180 | 14000 | **55.1%** | 44.9% | YES |
| 2048 | 34360 | 14000 | **71.1%** | 28.9% | YES |
| 4096 | 68720 | 14000 | 83.1% | 16.9% | YES |
| 8192 | 137439 | 14000 | 90.8% | 9.2% | YES |
| 16384 | 274878 | 14000 | 95.2% | 4.8% | YES |
| 32768 | 549756 | 14000 | 97.5% | 2.5% | YES |

### Roofline vs Breakdown

| Config | B | Roofline(ms) | Breakdown(ms) | Ratio |
|--------|---|-------------|--------------|-------|
| MHA-32 | 1 | 17.2 | 16.3 | 0.95x |
| GQA-8 | 1 | 16.3 | 15.1 | 0.93x |
| MHA-32 | 8 | 25.8 | 31.6 | 1.23x |
| GQA-8 | 8 | 18.4 | 36.4 | 1.98x |
| MHA-32 | 32 | 55.1 | 88.1 | 1.60x |
| GQA-8 | 32 | 25.8 | 107.4 | 4.17x |

## 关键发现

### 1. 瓶颈随Batch动态转移!

**B=1**: Weight 87% → 权重完全主导 (memory-bound, KV太小)
**B=8**: Weight 40%, KV 31%, Attn 29% → 三者接近均衡
**B=32**: Weight 14%, KV 44.5%, Attn 41% → KV+Attn占86%!
**B=128**: Weight 4.8%, KV 49.6%, Attn 45.6% → KV+Attn占95%!

**递进规律**: 随batch增大, KV和attention占比线性增长, 权重占比快速下降.
原因: 权重大小不变(14GB), 但KV大小∝B×S, attention计算量∝B×S.

### 2. GQA Python-level 实际更慢!

| B | MHA tok/s | GQA-8 tok/s | Speedup |
|---|-----------|------------|---------|
| 1 | 61 | 66 | 1.08x |
| 8 | 253 | 220 | **0.87x** |
| 32 | 363 | 298 | **0.82x** |
| 128 | 405 | 327 | **0.81x** |

**GQA在B≥8时比MHA更慢!** 原因:
- GQA expand (unsqueeze→expand→reshape) 占attention的59-86%
- KV减少(4x)带来的带宽节省被expand开销完全抵消
- 这再次证明: **Python-level GQA不可接受 → 必须用专用kernel**

### 3. Roofline模型低估大batch延迟1.6-4.2x

| | MHA-32 B=32 | GQA-8 B=32 |
|---|-------------|-----------|
| Roofline | 55ms | 26ms |
| Breakdown | 88ms | 107ms |
| Ratio | 1.6x | 4.2x |

**原因**: Roofline只考虑HBM读取, 忽略了:
1. **Attention计算** (SDPA是compute, 不是纯读取)
2. **GQA expand** (创建临时大tensor + reshape)
3. **Kernel launch overhead**

### 4. KV bottleneck 交叉点: S≈1024 (B=32)

KV占比从 S=512(38%) → S=1024(55%) → S=2048(71%)
交叉点在 **S≈1024 tokens** (B=32) — 之前KV读比权重读多!

这个交叉点与之前KV bandwidth benchmark的结果一致.

### 5. Long context MHA灾难: KV占97.5% (S=32K)

S=32K时, KV读55GB vs 权重14GB → KV是权重的4x!
→ MHA long context完全不可行, 必须GQA/MLA/KV offload

### 6. B=1时 Roofline模型准确 (0.95x)

B=1: Roofline=17.2ms, Breakdown=16.3ms → 非常接近!
小batch时权重读主导, compute和expand开销极小 → Roofline准确

## 与之前KV Cache Bandwidth benchmark对比

| | KV BW benchmark | Breakdown benchmark |
|---|----------------|--------------------|
| MHA KV% (B=32) | 71% (Roofline估) | **44.5%** (实测) |
| 瓶颈判断 | "KV是瓶颈" | "KV+Attn共同是瓶颈" |
| 误差原因 | 只测KV读取 | 包含attention计算 |

**修正**: 之前说"KV cache = 71%瓶颈"是Roofline估算.
实际测量: KV读=44.5%, Attention计算=41.2% → **KV+Attn合计86%**, 但KV和Attn大致相等.

这更精确: KV读取和attention计算都是瓶颈, 不只是KV读取.

## 实用结论

1. **B=1**: 权重瓶颈 → 优化方向: weight layout, FP8量化, batch packing
2. **B≤8**: 均衡瓶颈 → 全面优化
3. **B≥32**: KV+Attn瓶颈 → 优化方向: GQA专用kernel, KV cache压缩, attention kernel
4. **Long context (>1K)**: KV灾难性瓶颈 → 必须GQA/MLA/KV offload
5. **Python-level GQA不可用** → 必须Triton/FlashInfer kernel (BLOCK_M打包Q头)
6. **Roofline在B=1准确, B≥8低估1.2-4x** → 需考虑attention和expand开销