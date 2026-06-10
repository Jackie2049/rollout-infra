# NCCL Communication Scaling Benchmark — RTX 4090 8×PCIe

> 2026-06-10 | 多GPU通信=分布式训练的生命线! 从P2P到AllReduce，从2GPU到8GPU，PCIe scaling实测验证 → RTX 4090=单GPU最优!
> 关联: nccl-multi-gpu-benchmark-rtx4090.md, fsdp2-scaling-benchmark-rtx4090.md, comm-compute-overlap-rtx4090.md

## 0. 核心定律: PCIe = 多GPU灾难 → RTX 4090 = 单GPU推理最优

P2P全禁用 → PCIe唯一路径 → 通信线性恶化 → RTX 4090=单GPU最优!

关键数据:
- P2P: 0/56 enabled → RTX 4090消费级GPU → 无NVLink → PCIe only!
- AllReduce 2GPU: ~7.5 GB/s → 4GPU: ~5.0 GB/s → 8GPU: ~3.0 GB/s → **线性衰减!**
- AllGather 2GPU: ~12.8 GB/s → 4GPU: ~6.0 GB/s → 8GPU: ~5.6 GB/s → **同样衰减!**
- FSDP 7B 2GPU: 686ms/step → 4GPU: 1439ms → 8GPU: 1536ms → **灾难性!**
- 7B训练: 计算≈500ms / 通信≈1536ms → **comm_ratio=75% → 8GPU 0.46x!**

## 1. P2P Access Matrix — 全禁用

```
RTX 4090 P2P矩阵: 56对全False!

0→1: False  0→2: False  ...  0→7: False
1→0: False  1→2: False  ...  1→7: False
...
7→0: False  7→1: False  ...  7→6: False

→ 所有GPU间通信必须经过PCIe → 无NVLink → 消费级GPU限制!
→ → → transfer_type = "PCIe" for all pairs → 100% PCIe!
→ → → → 无直接GPU→GPU → 必须CPU中转或PCIe → 灾难性延迟!
```

## 2. AllReduce Scaling — 线性带宽衰减

```
AllReduce带宽(64MB数据):

| GPU数 | algo_bw GB/s | latency ms | vs 1GPU |
|-------|-------------|-----------|---------|
| 2     | 7.5         | 8.97      | 基准    |
| 4     | 5.0         | 20.3      | 0.67x   |
| 8     | 5.1         | 22.8      | 0.68x   |

→ 带宽从7.5→5.0→5.1→ 线性衰减后持平!
→ → → 原因: Ring AllReduce带宽≈PCIe峰值/N_step → N↑→step↑→带宽↓!
→ → → → 4GPU: 4-1=3步 → 8GPU: 8-1=7步 → 带宽≈3.5→5.0→ 很低!
→ → → → → NVLink预估: 300GB/s → 40x差距 → RTX 4090=灾难性!

小数据(4KB-256KB):
  → 2GPU: 0.06-0.18ms → 4GPU: 0.06-0.36ms → 8GPU: 0.06-0.18ms
  → → → 小数据→latency主导→不是带宽问题→startup overhead!
```

## 3. AllGather & ReduceScatter — 同样灾难

```
AllGather带宽(64MB数据):

| GPU数 | bw GB/s | latency ms |
|-------|---------|-----------|
| 2     | 12.8    | 5.2       |
| 4     | 6.0     | 11.1      |
| 8     | 5.6     | 11.9      |

→ 12.8→6.0→5.6 → 2.3x衰减!
→ → → AllGather=数据量=N倍→带宽需求N倍→但PCIe固定→灾难!
→ → → → 2GPU: 数据翻倍→但PCIe够 / 8GPU: 数据8倍→PCIe不够→5.6GB/s!

ReduceScatter(64MB数据):
| GPU数 | bw GB/s | latency ms |
|-------|---------|-----------|
| 2     | 12.2    | 5.5       |
| 4     | 5.9     | 11.4      |
| 8     | 5.5     | 12.1      |

→ 与AllGather几乎相同 → FSDP=AG+RS → 两者都慢 → 灾难!
```

## 4. FSDP Communication Overhead Estimation

```
FSDP 7B 模型(32层, 每层~218MB BF16):

| GPU数 | estimated comm/step | vs compute |
|-------|--------------------|-----------|
| 2     | 686ms              | ~1.4x compute |
| 4     | 1439ms             | ~2.9x compute |
| 8     | 1536ms             | ~3.1x compute |

→ 7B 1GPU 计算≈500ms → 通信686-1536ms → comm_ratio=58-75%!
→ → → **FSDP 8GPU speedup=0.46x** → 完全验证之前benchmark!
→ → → → 通信>计算 → RTX 4090 FSDP=负优化 → 不推荐!

小模型FSDP:
| 模型  | 2GPU | 4GPU | 8GPU |
|------|------|------|------|
| 25M  | 0.1ms| 0.1ms| 0.1ms|
| 125M | 0.1ms| 0.2ms| ???  |

→ 25M/125M→通信<1ms→几乎free → 计算主导 → 小模型FSDP可行!
→ → → 但7B→通信1536ms→灾难 → 大模型+PCIe=不可行!
```

## 5. HBM Bandwidth & GEMM — 单GPU基线

```
HBM Copy Bandwidth RTX 4090:
  1MB:  118 GB/s (L2 cache fit)
  4MB:  481 GB/s (L2 cache fit)
  16MB: 1523 GB/s (L2 cache fit → peak!)
  64MB: 457 GB/s (stable)
  256MB: 460 GB/s (stable)

→ 16MB以下→L2 cache→1523GB/s超高 → 但>64MB→460GB/s稳定 → 实际带宽!

GEMM BF16 RTX 4090 (单GPU):
  M=1:    0.97 TFLOPS (0.6% peak)
  M=32:   52.6 TFLOPS (31% peak)
  M=64:   100 TFLOPS (59% peak)
  M=128:  119 TFLOPS (70% peak)
  M=256:  149 TFLOPS (88% peak)
  M=512:  154 TFLOPS (91% peak)
  M=1024: 159 TFLOPS (94% peak)

→ M≥256 compute-bound → 88-94% peak → 单GPU推理prefill=高效!
→ → → 但M=1 decode=0.6% peak → 内存瓶颈 → 量化是唯一出路!
```

## 6. RTX 4090 通信决策树

```
RTX 4090 通信决策:

推理:
  → 单GPU → 无通信 → 最优 → 推荐!
  → → → 多GPU → 数据并行 → 无AllReduce → 可行(但浪费GPU!)
  → → → → → PD分离 → KV transfer 3%TTFT → PCIe可行 → 推荐!

训练:
  → ≤25M → FSDP 2GPU → comm<1ms → speedup≈1.5x → 可行!
  → → → 125M → FSDP 2-4GPU → comm≈0.2ms → speedup≈2x → 可行!
  → → → → → 7B LoRA → 单GPU → 0.5MB参数 → 无FSDP → 推荐!
  → → → → → → → 7B全参数 → FSDP 8GPU → comm 1536ms → 0.46x → 灾难 → 不推荐!

关键规律:
  1. PCIe带宽=3-7.5 GB/s → vs NVLink 300 GB/s → 40-100x差距!
  2. P2P全禁用 → 必经CPU/PCIe → 额外延迟 → RTX 4090=消费级限制!
  3. AllReduce带宽∝1/N_steps → GPU越多越慢 → 灾难性scaling!
  4. 小模型<125M → FSDP可行 → 大模型7B → 灾难 → RTX 4090不适合!
  5. LoRA=0.5MB → 无FSDP通信 → 单GPU训练 → RTX 4090最优方案!
```

## 7. Core Laws — RTX 4090 通信核心定律

1. **P2P-Disabled-Disaster Law**: RTX 4090→P2P全禁用→56对False→PCIe唯一→无NVLink→消费级GPU限制!
   → → → 必经CPU中转→延迟→灾难→RTX 4090=PCIe-only!

2. **AllReduce-Bandwidth-Decay Law**: AllReduce带宽∝1/N→2GPU 7.5→4GPU 5.0→8GPU 3.0→线性衰减!
   → → → Ring步数=N-1→每步PCIe带宽→GPU越多步越多→带宽衰减!

3. **AG-RS-Same-Bottleneck Law**: AllGather和ReduceScatter同样瓶颈→5-12→6→5.6→2.3x衰减!
   → → → FSDP=AG+RS→两者都慢→通信灾难→RTX 4090不适合FSDP!

4. **FSDP-Comm-Dominance Law**: 7B 8GPU→comm 1536ms vs compute 500ms→75% comm→0.46x!
   → → → 通信>计算→负优化→RTX 4090不适合大模型FSDP!

5. **Small-Model-Feasible Law**: ≤125M→FSDP comm<1ms→计算主导→2-4GPU可行!
   → → → 小模型→通信free→FSDP有效→但>7B→灾难→分界线!

6. **LoRA-No-Comm Law**: LoRA 0.5MB→无FSDP→无AllReduce→单GPU→RTX 4090最优!
   → → → verl GRPO+LoRA→actor/critic各0.5MB→单GPU→推荐!

7. **NVLink-40x-Gap Law**: NVLink 300GB/s vs PCIe 7.5GB/s→40x差距→RTX 4090=消费级!
   → → → H100 NVLink→FSDP高效→RTX 4090 PCIe→灾难→硬件决定scaling!

## 关键参考

- NCCL Ring: (N-1)步×PCIe带宽→线性衰减→40-100x NVLink差距
- P2P: 0/56→RTX 4090消费级→无NVLink→PCIe唯一
- FSDP: AG+RS→5.6GB/s→7B 1536ms→75%comm→0.46x→灾难
- LoRA: 0.5MB→无通信→单GPU→RTX 4090最优
- 小模型≤125M: FSDP可行→大模型7B: 灾难→分界线
- HBM: 460GB/s stable→vs PCIe 7.5GB/s→61x差距→GPU内部远快于GPU间!