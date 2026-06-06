# KV Cache Memory Bandwidth RTX 4090 实测: Decode瓶颈量化

> 2026-06-07 | RTX 4090 KV cache memory bandwidth micro-benchmark

## 核心数据

### HBM Read Bandwidth — 数据规模缩放

| Size(MB) | Read(ms) | BW(GB/s) | %Peak | 说明 |
|----------|----------|----------|-------|------|
| 1 | 0.013 | 75 | 8.6% | launch主导 |
| 4 | 0.013 | 298 | 33.9% | 部分利用 |
| 16 | 0.013 | 1205 | 137% | L2 cache |
| 32 | 0.013 | 2528 | 288% | L2 cache |
| 64 | 0.019 | 3298 | 376% | L2极限 |
| 128 | 0.141 | 908 | 103.5% | HBM饱和 |
| 256 | 0.274 | 935 | 106.6% | HBM稳定 |
| 512 | 0.548 | 934 | 106.5% | HBM稳定 |

**关键**: 16-64MB 远超 peak BW → 数据在 **L2 cache** (5MB RTX 4090) 中!
64MB以上才到真实 HBM BW (~934 GB/s, 106% peak).

### KV Cache Read — Attention Variant 对比 (B=8, S=2048)

| Config | KV/layer(MB) | Read(ms) | BW(GB/s) | 说明 |
|--------|-------------|----------|----------|------|
| MHA-32 | 268 | 0.296 | 908 | 大数据 → HBM饱和 |
| GQA-8 | 67 | 0.032 | 2107 | 小数据 → L2命中! |
| GQA-4 | 34 | 0.031 | 1069 | L2命中 |
| MQA-1 | 8 | 0.031 | 269 | 太小 → launch主导 |
| MLA-256 | 8 | 0.012 | 699 | 极小 → launch主导 |
| MLA-512 | 17 | 0.012 | 1398 | L2命中 |

**MHA每层KV读0.296ms** — 这是decode attention的数据读取时间。
32层 = 0.296×32 = **9.47ms** (仅KV读取, 不含权重!)

### KV Read vs Batch Size (MHA-32, S=2048)

| Batch | KV/layer(MB) | Read(ms) | BW(GB/s) |
|-------|-------------|----------|----------|
| 1 | 34 | 0.031 | 1064 |
| 4 | 134 | 0.154 | 874 |
| 8 | 268 | 0.294 | 912 |
| 16 | 537 | 0.573 | 936 |
| 32 | 1074 | 1.135 | 946 |

**B=32时每层KV读取1.135ms** — 32层=36.3ms (KV读占总时间的比例随B增大!)

### KV Read vs Seq Length (MHA-32, B=8)

| SeqLen | KV/layer(MB) | Read(ms) | BW(GB/s) |
|--------|-------------|----------|----------|
| 256 | 34 | 0.031 | 1069 |
| 512 | 67 | 0.031 | 2145 |
| 1024 | 134 | 0.153 | 880 |
| 2048 | 268 | 0.294 | 912 |
| 4096 | 537 | 0.573 | 936 |
| 8192 | 1074 | 1.135 | 946 |

**KV读取时间 ∝ seq_len** — 线性增长, long context下KV读成为瓶颈!

### GQA/MQA KV Expansion Overhead

| Config | Expand(ms) | Attn(ms) | Total(ms) | Expand% |
|--------|-----------|----------|-----------|---------|
| MHA-32 | 0 | 0.289 | 0.289 | 0% |
| GQA-8 | 0.381 | 0.301 | 0.682 | **56%** |
| GQA-4 | 0.338 | 0.295 | 0.633 | **53%** |
| MQA-1 | 0.013 | 0.201 | 0.214 | 6% |

**GQA expand占56%开销**! KV head expand (unsqueeze→expand→reshape) 创建临时大tensor → 56%开销.

MQA只有6%因为expand很小 (1→32, 但实际MQA在专用kernel中不做expand).

**结论**: Python-level GQA expand不可接受 → **必须用专用Triton/FlashInfer kernel** (GQA BLOCK_M打包Q头, 无expand).

### MLA Upsample vs MHA Full Read

| Method | KB/tok | ms | BW(GB/s) | vs MHA |
|--------|--------|-----|----------|--------|
| MHA-32 read | 16.0 | 0.146 | 918 | 1.0x |
| MLA-256 upsample | 0.5 | 0.446 | 19 | 32x压缩 |
| MLA-512 upsample | 1.0 | 0.843 | 20 | 16x压缩 |
| MLA-1024 upsample | 2.0 | 1.617 | 21 | 8x压缩 |

**MLA upsample BW仅19-21 GB/s** — 因为是 matmul (compute-bound), 不是纯读取!
- MLA存储小(32x压缩), 但upsample需要 matmul → 从latent维度映射到4096维 → 计算开销
- 19 GB/s BW远低于934 GB/s HBM → upsample matmul是瓶颈
- **关键**: MLA的真正收益不是内存带宽, 而是 **KV cache存储容量** (32x压缩 → 更多并发请求)

### Decode Throughput Model (7B, B=32, S=2048)

| Config | KV/step(MB) | Total(MB) | Est(ms) | tok/s |
|--------|-------------|-----------|---------|-------|
| MHA-32 | 34360 | 48360 | 55.1 | 580 |
| GQA-8 | 8590 | 22590 | 25.8 | 1242 |
| GQA-4 | 4295 | 18295 | 20.9 | 1534 |
| MQA-1 | 1074 | 15074 | 17.2 | 1862 |
| MLA-256 | 2148 | 16148 | 18.4 | 1738 |
| MLA-512 | 4295 | 18295 | 20.9 | 1534 |

**纯Roofline估算** (HBM BW 877 GB/s):
- MHA: 55ms → 580 tok/s → KV cache读73% (34.4GB/48.4GB)
- GQA-8: 25.8ms → 1242 tok/s → KV读38%
- MQA: 17.2ms → 1862 tok/s → KV读7%

**KV cache占比从7%(MQA)到73%(MHA)** → MHA decode是KV cache瓶颈而非权重!

## 关键发现

### 1. MHA decode: KV cache读是瓶颈

7B MHA B=32 S=2048:
- 权重读: 14GB / 877 GB/s = **16ms**
- KV读: 34.4GB / 877 GB/s = **39ms**
- **KV读占71%!** → 权重只占29%

这与之前decode GEMM benchmark的结果不同 — GEMM只测了权重, 没测KV!
加上KV后, MHA decode的瓶颈从权重转移到KV cache.

### 2. GQA/MQA: 减少KV → 从KV瓶颈回到权重瓶颈

GQA-8: KV只占38% → 权重又成主导 (62%)
MQA: KV只占7% → 权重完全主导 (93%)

**递进关系**:
- MHA: KV bottleneck (71%)
- GQA-8: transition zone (38% KV / 62% weight)
- MQA: weight bottleneck (93%)

### 3. GQA Python expand 56% overhead → 专用kernel必需

Python-level GQA expand占56% total time!
这是为什么vLLM Triton attention用 GQA BLOCK_M打包Q头 → 不expand → ↓87.5% KV load.

### 4. L2 cache对小KV cache有巨大影响

GQA-8 KV=67MB → BW=2107 GB/s (2.4x peak!) → L2命中
MHA-32 KV=268MB → BW=908 GB/s → HBM

小KV cache (GQA/MQA) 因L2 cache效果, 比纯HBM理论更快!

### 5. MLA不是带宽节省, 是容量节省

MLA upsample BW=19-21 GB/s (matmul, compute-bound)
→ MLA不节省读取带宽, 但节省 **32x存储容量** → 更多并发请求

### 6. Long context: KV cache线性增长 → 灾难性瓶颈

S=8192时KV/层=1074MB, 32层=34.4GB → 超过权重2.4x!
→ long context decode必须用GQA/MLA/KV offload

## 与之前实验对比

| | GEMM-only估算 | KV+GEMM估算 | 实际差异 |
|---|-------------|------------|---------|
| 7B MHA B=32 | 12.8ms/2494 tok/s | 55ms/580 tok/s | **4.3x更慢!** |
| GQA-8 | ~12ms/2667 tok/s | 25.8ms/1242 tok/s | **2.1x更慢** |

GEMM-only benchmark低估了总延迟2-4倍, 因为忽略了KV cache读取!
MHA的KV读是隐藏的瓶颈 — 权重读+GEMM只占总延迟的29%.