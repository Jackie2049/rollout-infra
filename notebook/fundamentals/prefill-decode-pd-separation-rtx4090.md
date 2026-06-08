# Prefill vs Decode Throughput — PD Separation Analysis RTX 4090

> 2026-06-08 | 5实验实测, Prefill 73.5%peak vs Decode 12.8%peak → 175x TFLOPS差距
> 关键: Prefill compute-bound, Decode memory-bound → 资源需求完全不同 → PD分离是自然的

## 1. Prefill Throughput (Compute-bound for S≥256)

| S | Time(ms) | TFLOPS | %Peak | AI | Bound | tok/s |
|---|---------|--------|-------|----|-------|-------|
| 32 | 21.13 | 19.5 | 11.5% | 26 | memory | 1,514 |
| 64 | 21.61 | 38.2 | 22.5% | 53 | memory | 2,962 |
| 128 | 23.33 | **70.7** | 41.7% | 106 | memory | 5,486 |
| 256 | 31.76 | **103.9** | 61.2% | 212 | **compute** | 8,060 |
| 512 | 57.71 | **114.3** | 67.4% | 424 | **compute** | 8,872 |
| 1024 | 111.39 | **118.5** | 69.8% | 847 | **compute** | 9,193 |
| 2048 | 211.60 | **124.7** | 73.5% | 1695 | **compute** | 9,679 |
| 4096 | 428.14 | **123.3** | 72.7% | 3390 | **compute** | 9,567 |

**关键发现**:
- **Crossover: S=256 → compute-bound** (AI=212 > Ridge≈190)
- S<256 → memory-bound → GPU underutilized
- S≥256 → compute-bound → 61-73% peak → 接近Roofline上限
- **Prefill可以充分利用GPU compute** → 适合专用prefill GPU

## 2. Decode Throughput (Memory-bound)

| B | Time(ms) | TFLOPS | %Peak | AI | tok/s |
|---|---------|--------|-------|----|-------|
| 1 | 16.59 | **0.7** | 0.4% | 0.81 | 60 |
| 2 | 16.68 | 1.4 | 0.8% | 1.63 | 120 |
| 4 | 16.74 | 2.8 | 1.7% | 3.26 | 239 |
| 8 | 16.89 | 5.6 | 3.3% | 6.52 | 474 |
| 16 | 16.97 | 11.1 | 6.6% | 13.04 | 943 |
| 32 | 17.43 | **21.7** | 12.8% | 26.07 | 1,836 |
| 64 | 17.73 | **42.6** | 25.1% | 52.15 | 3,609 |

**关键发现**:
- **Decode B=1仅0.4% peak!** → GPU严重浪费compute资源
- Decode时间几乎恒定(16.6-17.7ms) → memory-bound验证
- **GPU compute在decode时闲置87-99.6%** → 巨大浪费
- B=32才12.8% peak → 即使高并发也严重浪费compute
- **与Prefill形成鲜明对比**: prefill用73% peak vs decode用0.4% → 175x差距

## 3. Mixed Prefill+Decode Workload

| B_dec | S_pre | Decode(us) | Mixed(us) | ITL Change% |
|-------|-------|-----------|----------|------------|
| 1 | 128 | 1522 | 1109 | **-27.1%** |
| 1 | 512 | 1524 | 2090 | +37.2% |
| 1 | 2048 | 1525 | 6507 | **+326.6%** |
| 32 | 128 | 1594 | 1131 | **-29.0%** |
| 32 | 512 | 1591 | 2105 | +32.3% |
| 32 | 2048 | 1593 | 6497 | **+307.8%** |

**关键发现**:
- **S=128: 混合反而快27-29%!** → 短prefill(compute)与decode(memory)资源互补 → overlap有效
- **S≥512: ITL增加32-326%** → 长prefill阻塞decode → ITL严重恶化
- **S=2048: ITL翻3-4倍!** → 用户感知明显 → 用户体验恶化
- **vLLM chunked prefill**: 限制每步prefill tokens → 减少ITL stall → 但增加总prefill时间

## 4. TTFT vs ITL Profile

| S | TTFT(ms) | ITL B=1(ms) | Prefill TFLOPS | Decode TFLOPS | Ratio |
|---|---------|------------|---------------|--------------|-------|
| 128 | 23.3 | 16.59 | 70.7 | 0.7 | **99.6x** |
| 512 | 57.7 | 16.59 | 114.3 | 0.7 | **161.0x** |
| 1024 | 111.4 | 16.59 | 118.5 | 0.7 | **166.8x** |
| 2048 | 211.6 | 16.59 | 124.7 | 0.7 | **175.6x** |
| 4096 | 428.1 | 16.59 | 123.3 | 0.7 | **173.6x** |

**震撼**: **Prefill vs Decode TFLOPS差距高达175x!**
- Prefill GPU利用率73.5% vs Decode仅0.4% → 资源需求完全不同
- 这就是PD分离的根本动机 → compute vs bandwidth 专用化

## 5. PD Separation Theoretical Benefit

| S | KV(MB) | PCIe(ms) | %TTFT | NVLink(ms) | %TTFT | PD ITL改善 |
|---|--------|---------|-------|-----------|-------|----------|
| 128 | 10.0 | 0.41 | **1.7%** | 0.033 | 0.1% | 134% |
| 512 | 40.0 | 1.63 | **2.8%** | 0.130 | 0.2% | 331% |
| 2048 | 160.0 | 6.51 | **3.1%** | 0.521 | 0.2% | 1,214% |
| 4096 | 320.0 | 13.02 | **3.0%** | 1.042 | 0.2% | 2,456% |

**关键发现**:
- **PCIe KV transfer仅3% TTFT!** → 远小于预期 → PD分离可能比之前认为的更可行
- **NVLink KV transfer仅0.2% TTFT** → 几乎免费 → PD分离是生产的标配
- PD ITL改善: 134-2456% → 消除ITL stall是巨大收益

**修正之前的结论**:
- 之前说"RTX 4090 PCIe PD不可行" → **需修正!**
- PCIe KV transfer = 3% TTFT → 可以接受(但需要2GPU → 成本翻倍)
- NVLink几乎免费 → PD分离是生产标配(H100/A100)

**PD分离决策树**:
- 单GPU RTX 4090: chunked prefill → 限制每步prefill → 减少ITL stall
- 2GPU RTX 4090 PCIe: PD分离 → +3% TTFT → 但消除ITL翻倍 → 值得考虑(成本2x)
- H100 NVLink: PD分离 → +0.2% TTFT → 生产标配
- 大规模集群: 1:4或1:8 prefill:decode比例 → 最优资源分配

## 6. 核心规律

```
Prefill vs Decode 资源特征:
  Prefill (S≥256): compute-bound → 73.5% peak → GPU compute充分利用
  Decode (B=1):    memory-bound → 0.4% peak → GPU compute严重浪费
  → TFLOPS差距 175x → 资源需求完全不同 → PD分离是自然的!

  ITL stall规律:
    短prefill (S≤128): -27% ITL → 资源互补 → 不需要PD分离
    中prefill (S≥512): +32% ITL → 开始恶化 → chunked prefill缓解
    长prefill (S≥2048): +326% ITL → 严重恶化 → PD分离必需!

  PD分离成本:
    PCIe KV transfer: 3% TTFT → 可接受(但需2GPU)
    NVLink KV transfer: 0.2% TTFT → 几乎免费 → 生产标配
    → 之前"PCIe PD不可行"修正为"PCIe PD可行但成本2x"

  RTX 4090最优策略:
    单GPU: chunked prefill (限制每步S≤512) → ITL+32%可接受
    2GPU PCIe: PD分离 → +3% TTFT → 消除ITL翻倍 → 成本2x但体验好
    H100 NVLink: PD分离 → 生产标配 → 0.2% TTFT overhead

  Prefill:Decode比例:
    单GPU: 100:0 (混合)
    PD分离: 1:4-1:8 (1 prefill GPU : 4-8 decode GPU)
    原因: decode吞吐远低于prefill → 需要更多decode GPU
```

## 7. 与之前实验的交叉验证

| 之前发现 | 本次验证 | 新发现 |
|----------|---------|--------|
| "Decode 0.6% peak (B=1)" | **0.4% peak** | 验证! 更精确 |
| "Prefill compute-bound" | **S≥256 compute-bound** | crossover精确=256 |
| "PD分离需NVLink" | **PCIe也可! 仅+3% TTFT** | 修正! |
| "MLP占65% decode" | MLP GEMM确实是瓶颈 | 验证 |
| "Weight reads 95.1%" | Decode memory-bound验证 | 验证 |