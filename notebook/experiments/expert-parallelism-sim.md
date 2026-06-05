# Expert Parallelism Simulation (CPU)

> 2026-06-05 | 工具: `tools/expert_parallel_sim.py` | CPU 分析模型
> 参考: Mixtral 8x7B (arXiv:2401.04088), DeepSeek-V3 (arXiv:2412.19437)

## 1. EP Scaling (Mixtral 配置: 8 experts, Top-2, B=32)

| GPUs | Latency | Compute | Comm | Mem/GPU | Speedup | Efficiency |
|------|---------|---------|------|---------|---------|-----------|
| 1 | 209.6us | 209.6us | 0 | 2164MB | 1.0x | 100% |
| 2 | 108.3us | 104.8us | 3.5us | 1082MB | 1.94x | 96.8% |
| 4 | 54.2us | 52.4us | 1.7us | 541MB | 3.87x | 96.8% |
| 8 | 27.1us | 26.2us | 0.9us | 271MB | **7.74x** | **96.8%** |

**近乎线性扩展!** Communication overhead < 4% (NVLink 足够快)

## 2. Communication vs Compute

| Batch | Total | Comm | Comm% |
|-------|-------|------|-------|
| 1 | 0.8us | 0.0us | 3.2% |
| 32 | 27.1us | 0.9us | 3.2% |
| 128 | 108.3us | 3.5us | 3.2% |
| 512 | 433.2us | 14.0us | 3.2% |

Communication 比例稳定在 ~3% — NVLink 带宽充足, EP communication 不是瓶颈

## 3. Load Imbalance (8 GPUs)

| Experts | Top-2 | Top-4 | Top-6 |
|---------|-------|-------|-------|
| 8 | 1.00 | 1.00 | 1.00 |
| 64 | 3.50 | 1.88 | 1.33 |
| 256 | 8.00 | 4.50 | 2.89 |

**Expert 越多, 负载不均越严重!** DeepSeek-V3 (256 experts) 需要 bias-based 均衡

## 4. Expert Offloading (单 GPU)

| Hot Experts | Memory Saving | Latency Ratio |
|------------|---------------|---------------|
| 1/8 | -88% | 83.0x |
| 2/8 | -75% | 83.2x |
| 4/8 | -50% | 83.4x |
| 8/8 | 0% | 83.9x |

**CPU offloading 延迟代价巨大 (~83x)!** 适合吞吐优先、延迟不敏感的场景

## 5. Mixtral vs DeepSeek-V3 (8 GPUs EP)

| Model | Experts | Top-K | Latency | Comm% | Mem/GPU | Imbalance |
|-------|---------|-------|---------|-------|---------|-----------|
| Mixtral 8x7B | 8 | 2 | 27.1us | 3.2% | 271MB | 1.00 |
| DeepSeek-V3 | 256 | 6 | 231.9us | 0.7% | 25367MB | 3.11 |

**Mixtral 天然适合 EP (1 expert/GPU, 完美均衡)**
**DeepSeek-V3 需要 EP+TP 混合并行 (32 experts/GPU, 需要负载均衡)**

## 核心学习

1. **EP 几乎线性扩展**: Communication overhead 在 NVLink 下可忽略 (<4%)
2. **Mixtral 是 EP 的最佳场景**: 8 experts = 8 GPUs, 完美 1:1 映射
3. **Expert 越多, 负载均衡越重要**: DeepSeek-V3 的 bias-based 均衡是必要的
4. **CPU Offloading 是最后手段**: 83x 延迟代价, 只适合离线推理
5. **显存是 MoE 推理的核心瓶颈**: 46.7B params 全需加载, 即使只激活 12.9B
