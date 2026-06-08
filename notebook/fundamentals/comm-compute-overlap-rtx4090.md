# Communication-Computation Overlap Benchmark — RTX 4090

> 2026-06-08 | 5实验实测(2 GPU), overlap效率65-84%, FSDP RS overlap 50-79%
> 关键: PCIe RTX 4090上overlap有效! 但NCCL带宽(7GB/s)限制了scaling效率

## 1. Baseline: Compute vs Comm

| Type | Time(us) | Bandwidth |
|------|---------|-----------|
| Compute (MLP B=32) | **426** | — |
| AllReduce 1MB | 183 | 5.45 GB/s |
| AllReduce 4MB | 610 | 6.56 GB/s |
| AllReduce 16MB | 2290 | 6.99 GB/s |
| AllReduce 64MB | 8816 | 7.26 GB/s |
| AllReduce 256MB | 35494 | 7.21 GB/s |

**关键**: 2-GPU AllReduce带宽 5.5-7.3 GB/s → 与8-GPU benchmark(2.76GB/s)不同 → 2GPU带宽更高!

## 2. Compute + AllReduce Overlap

| Size(MB) | Sequential(us) | Overlapped(us) | Saved(us) | Efficiency% |
|----------|---------------|---------------|----------|------------|
| 4 | 1025 | **752** | 273 | **65.9%** |
| 16 | 2723 | **2428** | 295 | **68.1%** |
| 64 | 9319 | **8895** | 425 | **84.4%** |

**关键发现**:
- **65-84% overlap效率!** → PCIe RTX 4090上overlap确实有效
- Compute(426us)比4MB AllReduce(610us)快 → overlap后时间≈610us(comm主导)
- Compute(426us)比16MB AllReduce(2290us)慢 → overlap后时间≈2290+小额外开销
- **Overlap有效但不完美** → 非default stream开销 → 与之前"多stream负优化"发现对比

## 3. FSDP-style Overlap (RS + Compute)

| Size(MB) | RS(us) | AG(us) | Compute(us) | Sequential(us) | Overlapped(us) | Efficiency% |
|----------|--------|--------|------------|---------------|---------------|------------|
| 4 | 356 | 361 | 426 | 780 | **601** | **50.7%** |
| 16 | 1326 | 1327 | 426 | 2135 | **1496** | **79.0%** |

**关键发现**:
- **FSDP overlap: 50-79%效率** → RS(通信)可以与compute重叠!
- 4MB: RS(356us)≈compute(426us) → 时间接近 → overlap效率50%(接近50%理论最小)
- 16MB: RS(1326us)>>compute(426us) → compute几乎完全藏在RS内 → 79%效率
- **FSDP scaling仍受通信主导** → 但overlap减少了30%额外时间

## 4. 核心规律

```
Overlap效率规律:
  Compute < Comm → overlap效率低(comm主导,compute藏入comm)
  Compute ≈ Comm → overlap效率≈50%(两者各藏一半)
  Compute >> Comm → overlap效率高(comm完全藏入compute)

  RTX 4090 PCIe:
    2GPU AllReduce: 5.5-7.3 GB/s → 比8GPU(2.76GB/s)快2.6x!
    → 2GPU是RTX 4090 FSDP的最佳规模(之前FSDP benchmark验证2GPU=1.12x)

    FSDP overlap策略:
    1. RS梯度分片 → 与下一层forward重叠 → 50-79%效率
    2. AG参数收集 → 不能与forward重叠(需要参数才能forward)
    3. → FSDP per-step = max(RS, compute) + AG + compute

    与之前发现交叉验证:
    - FSDP 2GPU=1.12x → 2GPU是勉强可行的(scale小模型)
    - FSDP 4GPU=0.69x → 4GPU通信太多 → overlap帮助有限
    - FSDP 8GPU=0.50x → 灾难性 → NCCL带宽瓶颈 → overlap也无法挽救
    - CUDA_DEVICE_MAX_CONNECTIONS=1 → 确保通信先调度 → 生产必需

    NVLink vs PCIe对比:
    PCIe RTX 4090: AllReduce 7 GB/s → overlap 65-84% → 但scaling仍差
    NVLink H100: AllReduce 300 GB/s → overlap接近100% → scaling可行
    → PCIe overlap有帮助但不够 → NVLink才是scaling的根本解决

  RTX 4090训练决策:
    单GPU: 最优(推理或小模型训练)
    2GPU FSDP1: 勉强可行(≤25M模型, overlap帮助)
    4-8GPU: 不可行(通信>>计算 → overlap无法挽救)
    → 结论与之前FSDP benchmark完全一致!
```