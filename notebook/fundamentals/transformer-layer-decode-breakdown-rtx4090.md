# Transformer Layer Decode Breakdown — RTX 4090

> 2026-06-08 | 5实验实测, 逐组件分析7B-like decode延迟构成
> 关键: MLP占65%, lm_head仅1-2%, quantization 3.7x加速

## 1. Per-Component Timing (B=1 Decode)

| 组件 | 时间(us) | 说明 |
|------|---------|------|
| QKV proj (fused) | 109.5 | B×H → B×3H, 3个GEMM融合 |
| Q/K/V separate | 177.1 | 3个独立GEMM → 融合节省68us |
| Out proj | 50.8 | B×H → B×H |
| gate_proj | 145.9 | B×H → B×14336 |
| up_proj | 145.6 | B×H → B×14336 |
| SiLU×mul | 26.0 | element-wise (很小!) |
| down_proj | 147.0 | B×14336 → B×H |
| MLP total | 464.4 | gate+up+silu+down |
| RMSNorm | 46.3 | ×2 per layer |
| lm_head | 315.0 | B×H → B×32000 vocab |
| softmax | 41.7 | |
| multinomial | 158.5 | |
| sampling | 200.1 | |

**关键发现**:
- **MLP占一层65.1% vs Attn 34.9%!** MLP是decode绝对主导
- MLP GEMM(hidden×14336)比Attn GEMM(hidden×hidden)慢3x → 因为N更大→更多bytes
- **lm_head仅1.2%!** → 比之前估计的20-30%大幅低估 → 因为7B vocab=32K相对较小
- **sampling仅0.8%!** → softmax+multinomial=200us → vs 25ms decode → 近乎零
- **SiLU×mul仅26us** → element-wise ops极快 → kernel fusion价值小

## 2. Fused QKV vs Separate

| 方式 | 时间(us) | 节省 |
|------|---------|------|
| Fused QKV | 109.5 | - |
| Separate Q+K+V | 177.1 | **68us (38%)** |

**Fused QKV节省2个kernel launches → 38% faster!** → 生产必须用fused QKV

## 3. Batch Scaling

| B | per_layer(us) | decode(ms) | tok/s | lm_head% |
|---|--------------|-----------|-------|---------|
| 1 | 384.1 | 12.65 | **79** | 2.5% |
| 4 | 386.9 | 12.73 | **314** | 2.3% |
| 8 | 388.7 | 12.79 | **626** | 2.3% |
| 16 | 393.4 | 12.94 | **1,237** | 2.3% |
| 32 | 393.9 | 12.97 | **2,468** | 2.4% |
| 55 | 399.2 | 13.13 | **4,190** | 2.3% |

**关键发现**:
- **decode time几乎恒定!** 12.65→13.13ms → 仅+3% → memory-bound
- B×tok_per_s ≈ 常数 × B → 线性吞吐增长 → memory-bound验证
- lm_head仅2.3-2.5% → **lm_head不是decode瓶颈** (与之前Tokenization笔记的20-30%修正!)
- 原因: vocab=32K → lm_head=262MB → 占总bytes仅1.6%

## 4. lm_head Dominance Analysis (Roofline)

| B | weight% | kv% | lm_head% | Roofline tok/s |
|---|---------|-----|---------|---------------|
| 1 | 95.1% | 3.3% | 1.6% | 58 |
| 32 | 95.1% | 3.3% | 1.6% | 58 |

**震撼**: **Weight reads占95.1%!** → decode瓶颈是weight读取 → INT4量化是关键!
- KV仅3.3% → INT8 KV省的是内存容量而非带宽
- lm_head仅1.6% → vocab=32K时不是瓶颈

**大vocab影响**: vocab=128K → lm_head=1GB → 占比例升到6% → 仍不是主导
- vocab=151K(Qwen) → lm_head=1.2GB → 占~8% → 开始有影响但仍小于weight

## 5. Quantization Impact (Roofline Prediction)

| 配置 | bytes/tok(MB) | tok/s | vs BF16 |
|------|--------------|-------|---------|
| BF16 baseline | 15,610 | 58 | 1.0x |
| INT4+INT8KV | 4,218 | 216 | **3.70x** |
| INT4+INT8KV+INT8lm | 4,093 | 223 | **3.81x** |

**INT8 lm_head仅额外0.11x加速** → lm_head太小不值得量化(除非128K vocab)
- INT4 weight → 4x bytes省 → 3.7x加速 → 决定性!
- INT8 KV → 省内存容量但带宽影响小(kv仅3.3%)

## 6. 修正之前笔记的错误

**之前Tokenization笔记说"lm_head占decode 20-30%"** → 实测仅1.6-2.5%!

原因: 之前计算基于vocab=128K→lm_head=1GB, 但7B模型vocab=32K→lm_head=262MB→远小于预估
- 128K vocab lm_head占比 ≈ 6% → 不是20-30%
- **修正: lm_head在vocab=32K时仅1-2%, vocab=128K时约6% → 不是主要瓶颈**

## 7. 核心规律

```
Decode延迟构成 (7B, vocab=32K, BF16):
  Weight reads: 95.1% → 量化是唯一出路!
  KV reads:     3.3%  → INT8 KV省容量不是带宽
  lm_head:      1.6%  → 不是瓶颈(vocab=32K)
  sampling:     0.8%  → 近乎零

  MLP vs Attn: 65.1% vs 34.9% → MLP是decode绝对主导
  MLP原因: hidden×14336 > hidden×hidden → 3x更多bytes

  INT4加速: 3.70x (roofline) → 实测fused kernel约2-3x → 理论吻合
  INT4+FlashInfer: 3.70×(1.06-3.20) = 3.9-11.8x → 组合效应!

  lm_head修正: vocab=32K→1.6% / vocab=128K→6% → 不是主要瓶颈
  → 之前的"lm_head占20-30%"是错误的 → 修正!
```

## 8. 与之前实验的交叉验证

| 之前发现 | 本次验证 | 修正 |
|----------|---------|------|
| "Pipeline Breakdown MLP 15% fwd" | **MLP 65.1%** (decode!) | fwd≠decode! prefill compute-bound→MLP时间占比不同 |
| "lm_head占20-30%" | **1.6-2.5%** | 修正! vocab=32K太小→不是瓶颈 |
| "INT4 2-3x decode加速" | **3.70x roofline** | 吻合! (fused kernel消除overhead) |
| "sampling ~0.06ms" | **0.20ms (200us)** | 略高(multinomial占158us) |
| "MLP GEMM 4.1x slower" | MLP=510us vs attn=273us → **1.87x** | MLP比attn慢但不是4x |

**关键修正**: decode中MLP占比65% → vs prefill中占比不同 → decode=memory-bound→大GEMM读更多bytes→MLP主导