# Decode GEMM RTX 4090 实测: 每层延迟 + 吞吐估算

> 2026-06-07 | RTX 4090 decode per-layer GEMM micro-benchmark

## 核心数据

**7B MHA (H=4096, inner=11008) 每层GEMM时间**:

| B | attn (ms) | MLP (ms) | total/layer (ms) | 32层total (ms) | 吞吐 (tok/s) |
|---|-----------|----------|-------------------|----------------|-------------|
| 1 | 0.139 | 0.303 | 0.442 | 14.1 | 71 |
| 4 | 0.080 | 0.303 | 0.383 | 12.3 | 326 |
| 8 | 0.088 | 0.304 | 0.392 | 12.5 | 638 |
| 32 | 0.088 | 0.313 | 0.401 | 12.8 | 2,494 |
| 128 | 0.135 | 0.329 | 0.464 | 14.8 | 8,621 |
| 256 | 0.207 | 0.509 | 0.715 | 22.9 | 11,189 |
| 512 | 0.455 | 0.945 | 1.399 | 44.8 | 11,437 |

**7B GQA (8 KV heads, KV dim=1024)**:

| B | attn (ms) | MLP (ms) | total/layer (ms) | GQA vs MHA attn |
|---|-----------|----------|-------------------|-----------------|
| 1 | 0.092 | 0.301 | 0.393 | **0.66x** (34% faster) |
| 128 | 0.094 | 0.330 | 0.425 | 0.70x |
| 512 | 0.293 | 0.954 | 1.247 | 0.64x |

**70B (H=8192, inner=28672)**:

| B | attn (ms) | MLP (ms) | total/layer (ms) |
|---|-----------|----------|-------------------|
| 1 | 0.337 | 1.541 | 1.878 |
| 32 | 0.341 | 1.505 | 1.846 |
| 256 | 0.586 | 2.530 | 3.116 |
| 512 | 1.088 | 4.505 | 5.593 |

## 关键发现

### 1. MLP是每层瓶颈 (3个GEMM vs 4个attn GEMM)

7B MHA B=1: attn=0.139ms, MLP=0.303ms → **MLP占68%**

原因: MLP的gate+up+down各有 [1,H]×[H,11008] 的GEMM → 权重很大。
Decode时每次只读1 token的激活 (16KB), 但权重4096×11008×2bytes=88MB → 全从HBM读。

### 2. B=1→32 几乎flat (memory-bound验证)

B=1: 0.442ms, B=32: 0.401ms → 几乎不变!

这直接验证了decode memory-bound特性:
- 权重大小不变 (7B=14GB), 全部从HBM读
- 增加batch只增加少量激活数据读取
- 吞吐随batch线性增长 (71→2494 tok/s = 35x)

### 3. B≥256开始compute-bound

B=256: 0.715ms (vs B=32的0.401ms → 1.8x增加)
→ 开始从memory-bound过渡到compute-bound

**Ridge point**: 大约在B=128-256之间
(与之前GEMM Roofline benchmark一致: N=512 ridge point)

### 4. GQA加速decode attention 34%

7B GQA8 B=1: attn=0.092ms vs MHA attn=0.139ms → **34% faster**

原因: KV维度从4096降到1024 (每KV head), K/V proj矩阵小4x → 权重读取减少75%

但MLP不变 → 总层时间仅快11% (0.393 vs 0.442ms)

### 5. 70B单层1.88ms → RTX 4090不适合

70B B=1: 1.88ms/layer × 80层 = 150ms/tok → 6.7 tok/s

70B需要14.7GB×4=58.8GB权重 (FP16) → 超出24GB显存 → **RTX 4090无法跑70B**

7B (14GB FP16) fits 24GB → 是RTX 4090的sweet spot

### 6. B=1 时 吞吐71 tok/s vs 实测 vLLM

之前实测 vLLM OPT-125M RTX 4090 B=1: ~163 tok/s
7B GEMM-only估算: 71 tok/s

差距原因:
- OPT-125M只有125M参数 (vs 7B = 7000M = 56x更小)
- 7B有attention (KV cache读取), sampling, 和kernel launch overhead
- 实际vLLM 7B B=1估计: ~50-60 tok/s (考虑attention+sampling)

## Decode成本分解 (7B MHA B=32)

| Component | ms | 占比 |
|-----------|-----|------|
| GEMM (32层) | 12.8 | 85% |
| Sampling | 0.24 | 2% |
| Attention (KV cache) | ~1.5 | 10% |
| Kernel launch | ~0.3 | 3% |

**GEMM绝对主导** → 优化重点: 1) kernel fusion, 2) weight layout, 3) batch大小