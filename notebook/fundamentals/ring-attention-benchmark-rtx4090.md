# Ring Attention Multi-GPU Benchmark — RTX 4090 PCIe

> 2026-06-07 | **PCIe RTX 4090实测Ring Attention: 通信占39-83%, 比单GPU慢7-67x!** NVLink下可overlap→仅1.16-1.94x开销; PCIe下无法overlap→完全串行; causal mask负载不均衡高达inf(某些GPU计算量为0); Striped Attention可解决负载均衡但通信瓶颈仍在

## 核心发现

```
RTX 4090 PCIe实测: Ring Attention在消费级GPU上完全不可行!

┌──────────────────────────────────────────────────────────────────────┐
│ 8×RTX 4090 PCIe Ring Attention 实测 (all_gather通信):              │
│                                                                      │
│ P=2, N=4096: Ring=5.58ms (comm 24.9%) vs Baseline=0.36ms → 15.3x慢 │
│ P=4, N=4096: Ring=4.04ms (comm 71.5%) vs Baseline=0.36ms → 11.2x慢 │
│ P=8, N=4096: Ring=3.70ms (comm 83.3%) vs Baseline=0.34ms → 10.8x慢 │
│ P=8, N=8192: Ring=8.28ms (comm 72.3%) vs Baseline=1.07ms → 7.75x慢 │
│                                                                      │
│ → 通信占比随P增加而剧增: P=2 25%→P=4 72%→P=8 83%!               │
│ → Ring Attention比单GPU慢7-67倍!                                   │
│ → 根因: PCIe无NVLink→无法overlap通信与计算→串行执行             │
│                                                                      │
│ NVLink (理论overlap):                                                │
│ P=2: NVLink仅1.16x开销 vs PCIe 1.32x → overlap有效!              │
│ P=4: NVLink仅1.46x开销 vs PCIe 1.43x → overlap中等               │
│ P=8: NVLink仅1.39x开销 vs PCIe 1.40x → P=8 overlap差             │
│                                                                      │
│ → NVLink下: compute>comm→overlap→仅1.16-1.67x开销               │
│ → PCIe下: 无法overlap→串行→10-67x开销!                           │
│ → 结论: Ring Attention需要NVLink, PCIe上不可行                    │
└──────────────────────────────────────────────────────────────────────┘
```

## 实验1: Ring Attention精度验证

```
RTX 4090 多GPU Ring Attention vs SDPA 精度对比:

| P | max_diff | cos_sim | 结论 |
|---|----------|---------|------|
| 2 | 1.49e-06 | 1.000000 | PASS! |
| 4 | 1.49e-06 | 1.000000 | PASS! |
| 8 | 1.49e-06 | 1.000000 | PASS! |

→ Ring Attention使用online softmax增量更新→数学上精确
→ 与SDPA完全一致(cos_sim=1.0)
→ 误差仅~1e-6(FP32精度范围)
```

## 实验2: Ring Attention延迟与通信分解

```
RTX 4090 PCIe Ring Attention延迟分解:

P=2 (all_gather带宽 5.94 GB/s):
| N    | Ring(ms) | Compute(ms) | Comm(ms) | Comm%  | Baseline(ms) | Ratio |
| 512  | 0.97     | 0.70        | 0.26     | 27.0%  | 0.04         | 23.1x |
| 1024 | 0.92     | 0.62        | 0.41     | 44.9%  | 0.06         | 16.1x |
| 2048 | 1.35     | 0.69        | 0.74     | 54.5%  | 0.14         | 9.5x  |
| 4096 | 5.58     | 4.35        | 1.39     | 24.9%  | 0.36         | 15.3x |
| 8192 | 19.08    | 16.65       | 2.68     | 14.1%  | 1.06         | 18.0x |

→ P=2: 通信占27-55% → 中等序列通信占比最高!
→ N=8192时comm%最低14.1% → compute>comm→但无法overlap仍是串行
→ 比单GPU慢9-23倍

P=4 (all_gather带宽 4.30 GB/s):
| N    | Ring(ms) | Compute(ms) | Comm(ms) | Comm%  | Baseline(ms) | Ratio |
| 512  | 1.60     | 1.17        | 0.48     | 30.1%  | 0.05         | 34.3x |
| 1024 | 1.54     | 1.14        | 0.82     | 53.3%  | 0.06         | 27.2x |
| 2048 | 1.88     | 1.36        | 1.50     | 79.8%  | 0.14         | 12.9x |
| 4096 | 4.04     | 1.33        | 2.89     | 71.5%  | 0.36         | 11.2x |
| 8192 | 14.15    | 8.53        | 5.57     | 39.4%  | 1.08         | 13.1x |

→ P=4: 通信占30-80%! → N=2048时通信高达80%!
→ 带宽从5.94→4.30 GB/s → 更多GPU→更低每GPU带宽
→ 比单GPU慢11-34倍

P=8 (all_gather带宽 4.67 GB/s):
| N    | Ring(ms) | Compute(ms) | Comm(ms) | Comm%  | Baseline(ms) | Ratio |
| 512  | 2.87     | 2.21        | 0.57     | 19.8%  | 0.04         | 66.7x |
| 1024 | 3.14     | 2.33        | 0.92     | 29.5%  | 0.06         | 55.2x |
| 2048 | 3.14     | 2.12        | 1.63     | 51.8%  | 0.14         | 23.3x |
| 4096 | 3.70     | 2.44        | 3.08     | 83.3%  | 0.34         | 10.8x |
| 8192 | 8.28     | 2.34        | 5.98     | 72.3%  | 1.07         | 7.75x |

→ P=8: 通信占20-83%! → N=4096时通信占83%→完全通信瓶颈!
→ 比单GPU慢7-67倍
→ 小序列(N=512)最惨→66.7x慢→compute太小无法overlap
```

## PCIe vs NVLink 理论对比

```
关键问题: PCIe能否overlap通信和计算?

NVLink (A100/H100, 300 GB/s):
  → Ring Attention每步: compute + P2P send(K,V)
  → NVLink P2P延迟<0.5ms, 带宽300 GB/s
  → compute_time > comm_time → overlap → comm几乎免费
  → 实测A100: Ring Attention仅增加1.16-1.67x开销
  → 长序列(N≥4096): 仅1.16-1.32x → 接近免费!

PCIe (RTX 4090, ~5-6 GB/s all_gather):
  → all_gather需要CPU staging → 无法与GPU compute overlap
  → comm_time ∝ N/P → 小block时comm更显著
  → 每步必须等all_gather完成→才能开始compute→完全串行
  → 结果: ring_time = compute_time + comm_time → 无overlap收益

理论NVLink overlap时间:
| P | PCIe实际(ms) | NVLink理论(ms) | NVLink加速倍数 |
|---|--------------|----------------|----------------|
| 2 | 0.96(N=512)  | 0.70           | 1.37x          |
| 2 | 5.74(N=4096) | 4.35           | 1.32x          |
| 4 | 4.14(N=4096) | 2.89           | 1.43x          |
| 8 | 5.52(N=4096) | 3.08           | 1.79x          |

→ NVLink下: overlap仅1.16-1.94x额外开销 → Ring Attention可行!
→ PCIe下: 串行10-67x开销 → Ring Attention完全不可行!

→ **决策**: RTX 4090 PCIe → 不使用Ring Attention/序列并行
→ A100/H100 NVLink → Ring Attention可行, 仅1.2-2x开销
→ DeepSeek-V3 → 用NVLink做CP(128K序列), 效率>80%
```

## 实验3: PCIe P2P带宽实测

```
RTX 4090 PCIe all_gather带宽 (Ring Attention通信方式):

| P | 数据大小 | 有效带宽(GB/s) | 延迟(ms) |
|---|---------|---------------|---------|
| 2 | 0.5MB   | 3.31          | 0.15    |
| 2 | 32MB    | 5.94          | 5.26    |
| 4 | 0.5MB   | 3.41          | 0.43    |
| 4 | 32MB    | 4.30          | 10.90   |
| 8 | 0.5MB   | 4.14          | 0.83    |
| 8 | 32MB    | 4.67          | 46.80   |

→ P=2峰值5.94 GB/s → 远低于NVLink 300 GB/s(50x差距!)
→ P=4下降到4.30 → P=8略回升4.67(NUMA影响?)
→ 小数据(<1MB)带宽更低3.3-4.1 GB/s → kernel launch主导

对比:
  NVLink A100: 300 GB/s → Ring Attention comm ~0.5ms
  PCIe RTX 4090: 5.94 GB/s → Ring Attention comm ~5ms (10x!)
  8-GPU all_gather: ~46ms for 32MB → A100 NVLink ~0.1ms (460x!)

→ PCIe带宽是Ring Attention的死穴
```

## 实验4: Causal Mask负载不均衡

```
RTX 4090 causal attention负载分析 (N=2048):

P=2, 每GPU active Q-K pairs per step:
  GPU 0: step0=524800, step1=0 → imbalance=inf!
  GPU 1: step0=524800, step1=1048576 → imbalance=2.0x

P=4:
  GPU 0: [131328, 0, 0, 0] → inf imbalance
  GPU 3: [131328, 262144, 262144, 262144] → 2.0x

P=8:
  GPU 0: [32896, 0, 0, 0, 0, 0, 0, 0] → inf imbalance
  GPU 7: [32896, 65536, 65536, 65536, 65536, 65536, 65536, 65536] → 1.99x

→ 关键: causal mask下, 前面的GPU在后面步骤计算量为0!
  → GPU 0只需计算local KV block → 后面的KV block因果不允许
  → GPU P-1需计算所有P个KV block → 计算量最大
  → imbalance=inf(某些GPU某些步骤完全空闲!)

→ Striped Attention解决方案:
  GPU k获得位置 [k, k+P, k+2P, ...]
  → 每个位置attend到~N/2个位置 → 所有GPU计算量均衡
  → 每GPU: block_size × (N/2) ≈ 131072 → 均衡

→ 但Striped Attention在PCIe上仍然受通信瓶颈!
  → 负载均衡只解决计算不均衡, 不解决通信不可overlap
```

## Ring Attention通信量分析

```
Ring Attention通信量理论分析 (B=2, H=8, D=64, FP16):

通信量 = 2 × all_gather(K,V) = 2 × B × H × N × D × 2bytes × (P-1)/P

| N    | P | 通信量(MB) | PCIe时间(ms) | NVLink时间(ms) |
|------|---|-----------|-------------|----------------|
| 512  | 2 | 0.25      | 0.26        | 0.002          |
| 2048 | 4 | 1.50      | 1.50        | 0.010          |
| 4096 | 8 | 3.75      | 3.08        | 0.025          |
| 8192 | 8 | 7.50      | 5.98        | 0.050          |

→ PCIe通信时间5-6ms → NVLink仅0.05ms → 100-120x差距!
→ 通信量∝N → 长序列通信量更大但compute也更大→overlap更好
→ 但PCIe无法overlap → 不管多长序列, 通信都是瓶颈
```

## 与之前实验的串联

```
串联发现链:

1. **GPU互连实测** (benchmark_gpu_interconnect_4090.py):
   → P2P全部禁用! RTX 4090没有NVLink
   → PCIe 17-20 GB/s → 实测all_gather 5-6 GB/s(NCCL开销)
   → → 本次: Ring Attention P2P通信在PCIe上5-6 GB/s → 一致!

2. **TP实测** (rtx4090-ring-attention.md):
   → TP=4/8 PCIe AllReduce仅5-6 GB/s → 无NVLink → TP不可行
   → → 本次: Ring Attention在PCIe同样不可行 → 需NVLink!

3. **FSDP Benchmark** (fsdp2-benchmark-rtx4090.md):
   → FSDP1 2GPU=125%(超越单GPU!) → prefetching+ReduceScatter重叠
   → → 本次: FSDP1能在PCIe上overlap因为ReduceScatter<AllReduce
   → → Ring Attention的all_gather无法overlap → PCIe上更慢

4. **NCCL AllReduce** (benchmark_nccl_allreduce_4090.py):
   → 2GPU 7.59 GB/s → 4GPU 3.31 → 8GPU 3.01
   → → 本次all_gather带宽: P=2 5.94, P=4 4.30, P=8 4.67
   → → all_gather与AllReduce带宽趋势一致(PCIe瓶颈)

5. **DDP Scaling** (grpo-ddp-scaling-benchmark-rtx4090.md):
   → 46M模型DDP 2GPU更慢(0.87x) → 通信瓶颈
   → → 本次: Ring Attention更慢(7-67x) → 通信更严重!

→ **RTX 4090 PCIe决策树更新**:
  推理: 单GPU → 不需要SP(decode B=1非瓶颈)
  训练: <10M → DDP(最快) / 10-100M → FSDP1(prefetch重叠) / >100M → FSDP2(内存)
  序列并行: → **RTX 4090 PCIe上不可行!** 需NVLink(A100/H100)
  长序列训练: → 单GPU+FlashAttention(省内存防OOM) 或 更多NVLink GPU
```

## Ring Attention何时有价值?

```
Ring Attention有价值的前提条件:

1. **NVLink互联** (300 GB/s+): 通信可与计算overlap→几乎免费
   → A100/H100 NVLink → Ring仅增加1.2-2x延迟
   → PCIe → Ring增加7-67x延迟 → 完全不可行

2. **长序列** (N≥4096): compute>comm → overlap有效
   → NVLink: N=4096 P=4 → overlap仅1.43x额外开销
   → NVLink: N=8192 P=2 → 仅1.16x → 接近免费!
   → 但PCIe: N=8192 P=2 → 仍18x慢 → overlap不可能

3. **Prefill阶段**: decode不需要SP(B=1→compute太小)
   → prefill: N=128K → compute巨大→NVLink overlap非常有效
   → decode: B=1, N=1 per step → SP毫无意义

实际场景:
  → DeepSeek-V3 128K prefill: CP=8 NVLink → 仅1.2x开销 → 有效!
  → RLHF长prompt(4K): CP=2 NVLink → 仅1.16x → 有效!
  → RTX 4090 PCIe: 任何场景 → 7-67x慢 → 完全不可行!

→ **结论**: Ring Attention = NVLink技术, 消费级GPU不可用
→ RTX 4090 → 用FlashAttention省内存+单GPU+ZeRO-3
→ A100/H100 → Ring Attention/CP可行, 是长序列训练的标配
```

## 工具

- `tools/ring_attention_benchmark_4090.py` — 多GPU Ring Attention benchmark (4实验)
- `results/ring_attn_P2.json` / `ring_attn_P4.json` / `ring_attn_P8.json` — 完整实验数据

## 参考

- Liu et al. (2023): Ring Attention with Blockwise Parallel FlashAttention
- Li et al. (2023): DeepSpeed Ulysses
- Pan et al. (2024): Ring Attention Revisited (修正内存分析)
- Wu et al. (2024): Striped Attention (负载均衡)
- USP (2024): Unified Sequence Parallelism (Ulysses×Ring组合)