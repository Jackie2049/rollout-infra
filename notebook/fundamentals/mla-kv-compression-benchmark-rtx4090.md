# MLA KV Compression Simulation Benchmark — RTX 4090

> 2026-06-07 | **MLA saves 3.2x KV memory(DS_V3 style) but decode 2.4-7.8x slower! Upsample matmul是瓶颈(compute-bound). 小配置MLA反而不如GQA(rope开销). MLA的价值=容量↑,不是吞吐↑.**

## 核心发现

```
RTX 4090 MLA KV Compression实测 — 6实验(3配置×多batch):

┌──────────────────────────────────────────────────────────────┐
│ 关键数据:                                                    │
│                                                              │
│ 1. KV内存对比 (DS_V3 style, S=2048):                        │
│   MHA: 67.1MB → GQA: 67.1MB → MLA: 21.0MB → 3.2x压缩!     │
│   MLA = compressed(4.2MB) + rope(16.8MB)                    │
│   → MLA容量: 200并发(S=8K) vs MHA 62 → 3.2x并发↑           │
│                                                              │
│ 2. 但小配置MLA反不如GQA:                                     │
│   GQA_4: MLA 1.6MB vs GQA 1.0MB → MLA反而大1.6x!           │
│   → rope开销(n_heads×rope_d)在小d_latent时超过压缩收益       │
│   → MLA只在d_latent足够大(≥n_kv_heads×d_head的一半)时省     │
│                                                              │
│ 3. MLA decode更慢:                                           │
│   DS_V3 style B=1: MLA=0.35ms vs MHA=0.09ms → 3.97x慢!     │
│   DS_V3 style B=32: MLA=18.3ms vs MHA=2.4ms → 7.70x慢!     │
│   → 原因: upsample matmul = compute-bound                    │
│   → 读取compressed KV(21MB) + up-project到full(67MB)        │
│   → matmul比纯attention还重!                                 │
│                                                              │
│ 4. MLA projection开销分解:                                   │
│   down_K+V: 0.068ms → 省了prefill时存储(KV→latent)          │
│   up_K+V: 0.268ms → decode瓶颈!(6x more than attention)     │
│   rope: 0.079ms → 额外开销                                   │
│   → up-project > attention本身 → MLA延迟更高                 │
│                                                              │
│ 5. RoPE decoupled精度:                                       │
│   分数拆分cos=1.0 → nope+rope=full精确等价                   │
│   但MLA+RoPE vs 标准+RoPE cos≈0.6 → 不等价!                 │
│   → 原因: decoupled RoPE只旋转rope_d维度                     │
│   → 标准RoPE旋转全d_head → 两种模式数学不等价                │
│   → DeepSeek-V3的"decoupled RoPE"是有意设计≠标准RoPE        │
│                                                              │
│ 6. Capacity是MLA的真正价值:                                  │
│   DS_V3 S=8192: MHA=62req → MLA=200req → 3.2x并发!          │
│   → 更多并发=更高吞吐(即使单请求更慢)                         │
│   → 这是DeepSeek-V3选择MLA的根本原因:                        │
│     671B模型 → MHA KV极大 → MLA压缩→更多并发→更高吞吐       │
│                                                              │
│ **RTX 4090 MLA决策**:                                        │
│   1. MLA不加速单请求 → decode反而慢2-8x                     │
│   2. MLA加速吞吐 → 并发↑3.2x → 总吞吐可能更高              │
│   3. 小模型: GQA更优(MLA rope开销>压缩收益)                  │
│   4. 大模型(7B+): MLA有价值 → 但需要fused upsample kernel   │
│   5. DeepSeek-V3: MLA必需 → 671B KV无MLA→OOM              │
└──────────────────────────────────────────────────────┘
```

## 完整数据

```
Exp 1: KV内存对比:

| Config | S | MHA(MB) | GQA(MB) | MLA(MB) | MLA/MHA | MLA/GQA |
|--------|---|---------|---------|---------|---------|---------|
| GQA_4  | 2048 | 2.1 | 1.0 | 1.6 | 1.3x | 0.7x |
| GQA_8  | 2048 | 4.2 | 2.1 | 3.1 | 1.3x | 0.7x |
| DS_V3  | 2048 | 67.1 | 67.1 | 21.0 | **3.2x** | **3.2x** |
| DS_V3  | 8192 | 268.4 | 268.4 | 83.9 | **3.2x** | **3.2x** |

→ GQA_4/8: MLA rope开销(n_heads×rope_d) > compressed savings → MLA比GQA更大!
→ DS_V3: d_latent=512远大于n_kv_heads×d_head的一半 → MLA有收益
→ **MLA只在d_latent足够大时有效!**

Exp 2: MLA projection延迟:

| Config | S | down_K | down_V | up_K | up_V | rope | total |
|--------|---|--------|--------|------|------|------|-------|
| GQA_4  | 2048 | 0.029 | 0.029 | 0.029 | 0.029 | 0.030 | 0.145 |
| DS_V3  | 2048 | 0.034 | 0.034 | **0.134** | **0.134** | 0.079 | **0.415** |

→ DS_V3 up-project: 0.134ms × 2 = 0.268ms → 占总65%!
→ up-project是compute-bound → matmul远大于attention本身

Exp 3: Decode throughput (S=2048):

| Config | B | MHA(ms) | GQA(ms) | MLA(ms) | MLA/MHA | MLA/GQA |
|--------|---|---------|---------|---------|---------|---------|
| GQA_4  | 1  | 0.086 | 0.087 | 0.205 | 2.39x | 2.37x |
| GQA_4  | 64 | 0.195 | 0.193 | 0.985 | 5.06x | 5.10x |
| GQA_8  | 1  | 0.085 | 0.086 | 0.198 | 2.31x | 2.29x |
| GQA_8  | 64 | 0.345 | 0.344 | 1.878 | 5.44x | 5.46x |
| DS_V3  | 1  | 0.088 | 0.093 | 0.349 | **3.97x** | 3.76x |
| DS_V3  | 8  | 0.620 | 0.622 | 4.675 | **7.54x** | 7.52x |
| DS_V3  | 64 | 4.668 | 4.669 | 36.511 | **7.82x** | 7.82x |

→ MLA decode 2-8x慢! upsample matmul compute-bound
→ 大batch: MLA overhead更大(up-project matmul随B增长)

Exp 4: Upsample matmul开销分析 (S=2048):

| Config | MHA_read(MB) | MLA_read(MB) | MLA/MHA | up_K(ms) | up_V(ms) | MHA_attn(ms) | MLA_est(ms) |
|--------|-------------|-------------|---------|----------|----------|-------------|-------------|
| GQA_4  | 2.1 | 1.6 | 1.3x省 | 0.027 | 0.026 | 0.086 | 0.055 |
| DS_V3  | 67.1 | 21.0 | **3.2x省** | 0.127 | 0.127 | 0.097 | **0.277** |

→ DS_V3: MLA读取省3.2x → 但up-project总时间0.277ms > MHA attn 0.097ms
→ HBM读取省 → 但compute开销增 → net: MLA更慢
→ 如果batch足够大 → KV cache容量优势 → 更多并发 → 总吞吐↑

Exp 5: RoPE decoupled精度:

| Config | split_cos | output_cos | MLA+RoPE vs Std+RoPE |
|--------|-----------|------------|---------------------|
| GQA_4  | 1.000000 | 1.000000 | 0.594 |
| DS_V3  | 1.000000 | 1.000000 | 0.678 |

→ 分数拆分(nope+rope=full): 精确等价(cos=1.0)
→ **但RoPE应用方式不同**: decoupled RoPE只旋转rope_d=64维度
→ 标准RoPE旋转全d_head=256维度 → 两种模式产生不同attention pattern
→ cos≈0.6 → DeepSeek-V3的decoupled RoPE是有意设计 ≠ 标准RoPE等价
→ 实际应用: decoupled RoPE仍有相对位置编码效果, 但pattern不同

Exp 6: Capacity scaling (24GB GPU, 70% KV):

| Config | avg_S | MHA_req | GQA_req | MLA_req | MLA/MHA | MLA/GQA |
|--------|-------|---------|---------|---------|---------|---------|
| GQA_4  | 2048 | 8010 | 16021 | 10681 | 1.3x | 0.7x |
| DS_V3  | 2048 | 250 | 250 | **801** | **3.2x** | **3.2x** |
| DS_V3  | 8192 | 62 | 62 | **200** | **3.2x** | **3.2x** |
| DS_V3  | 16384 | 31 | 31 | **100** | **3.2x** | **3.2x** |

→ **DS_V3 MLA: 3.2x并发容量** → 671B模型必须MLA(否则KV OOM!)
→ 小配置: MLA不如GQA → rope开销抵消压缩收益
```

## 与之前实验的串联

```
串联发现链:

1. **KV Cache BW** (kv-cache-bandwidth-rtx4090.md):
   → MHA KV 44.5% decode瓶颈 → KV量越大→瓶颈越严重
   → → 本次: MLA省3.2x KV → KV量减少 → KV BW需求减少3.2x
   → → → 但up-project compute-bound → 减少HBM读但增加compute → net延迟更高

2. **Decode Latency Breakdown** (decode-latency-breakdown-rtx4090.md):
   → B=32: KV+Attn 86% → KV是主要瓶颈
   → → 本次: MLA减少KV读 → 但upsample matmul比KV读更耗时
   → → → → MLA单请求延迟↑ → 但并发↑ → throughput可能↑

3. **DeepSeek-V3 Architecture** (deepseek-v3-architecture.md):
   → MLA 56.9x KV压缩 → $5.6M训练成本 → 671B/37B稀疏
   → → 本次: 实测3.2x压缩(小模型) → 56.9x是671B的full-scale数据
   → → → → MLA=容量优化而非速度优化 → 这是DeepSeek选择的根本原因

4. **Quantization** (quantization-inference-benchmark-rtx4090.md):
   → INT8 KV 1.00x+50%省 → 与MLA类似(都省KV)
   → → 本次: MLA省3.2x但慢2-8x → INT8 KV省2x(50%)但1.00x
   → → → → INT8 KV更优! (延迟不变+50%省) vs MLA(延迟↑+3.2x省)
   → → → → → 但MLA省更多(3.2x vs 2x) → 671B必须MLA+INT8

5. **Prefix Sharing** (full-model-ps-rtx4090.md):
   → PS 2.46x加速 → MLP 82%主导
   → → 本次: MLA upsample matmul = MLP-style compute → compute-bound
   → → → → PS和MLA都涉及MLP-level compute → 两者都compute-bound
   → → → → → 但PS省compute → MLA增compute(upsample) → 方向相反!

→ **MLA vs 其他KV优化对比**:

| 方法 | KV省 | 延迟影响 | 适用场景 |
|------|------|---------|---------|
| MLA  | 3.2-56.9x | 2-8x慢(upsample) | 671B大模型 |
| GQA  | 2-8x | 1.00x(零开销) | 中小模型 |
| INT8 KV | 2x | 1.00x(零开销) | 所有模型 |
| INT4 KV | 4x | ~1.0x(fused kernel) | 推理密集 |
| MQA  | n_heads/x | 1.00x | 极端压缩 |
| FlashAttn | 不省KV | decode 0.67x(更慢!) | 内存防OOM |

→ **最优组合**: MLA + INT8 KV → 3.2x × 2x = 6.4x KV容量↑!
→ DeepSeek-V3实际用: MLA + FP8 KV → 类似思路
```

## 工具

- `tools/mla_kv_compression_benchmark_4090.py` — 6实验MLA benchmark
- `results/mla_kv_compression_benchmark.json` — 完整数据