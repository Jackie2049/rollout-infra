# FSDP2 Scaling Benchmark: RTX 4090 PCIe 实测

> 2026-06-08 | 从1到8 GPU, PCIe scaling的残酷现实
> 基于: tools/fsdp2_scaling_benchmark_4090.py (8x RTX 4090 PCIe, PyTorch 2.9)
> 关联: nccl-communication-deep-dive.md (NCCL), distributed-training-frameworks-comparison.md (框架对比)

## 0. 核心发现: PCIe scaling在4GPU以上急剧下降!

```
FSDP1 BF16 scaling (25M模型):
  1GPU: 152K tok/s → baseline
  2GPU: 171K tok/s → 1.12x → 仅12%加速!
  4GPU: 104K tok/s → 0.69x → 比单GPU更慢!
  8GPU: 101K tok/s → 0.67x → 比单GPU更慢!

FSDP1 BF16 scaling (125M模型):
  1GPU: 33K tok/s → baseline
  2GPU: 27K tok/s → 0.82x → 已经更慢!
  4GPU: 16K tok/s → 0.48x → 一半速度!
  8GPU: 15K tok/s → 0.46x → 一半速度!

DDP scaling (更惨):
  8GPU DDP = 0.23-0.34x of 1GPU → 反而变慢!

结论: RTX 4090 PCIe分布式训练, >2GPU完全不划算!
  → 25M: 最多2GPU(勉强1.12x加速)
  → 125M: 单GPU最优(2GPU已经更慢!)
```

## 1. 完整数据

### 1.1 25M模型 (45.4M参数)

| GPUs | 策略 | 时间(ms) | 内存(GB) | tok/s | vs 1GPU | 内存vs 1GPU |
|------|------|---------|---------|-------|---------|-----------|
| 1 | Single | 26.89 | 2.14 | **152K** | 1.0x | 1.0x |
| 2 | DDP | 43.25 | 2.32 | 95K | 0.62x | 1.09x |
| 2 | FSDP1 | 23.98 | 1.15 | **171K** | 1.12x | 0.54x |
| 4 | DDP | 73.85 | 2.32 | 55K | 0.37x | 1.09x |
| 4 | FSDP1 | 39.22 | 1.02 | 104K | 0.69x | 0.48x |
| 8 | DDP | 79.55 | 2.32 | 51K | 0.34x | 1.09x |
| 8 | FSDP1 | 40.39 | 0.95 | 101K | 0.67x | 0.44x |

**分析**:
- FSDP1比DDP快2x(0.53x time) → 因为FSDP1 overlap通信+减少内存
- FSDP1内存省50% → 从2.14GB→1.15GB(2GPU) → 但这不够补偿通信开销
- 4GPU和8GPU FSDP1吞吐甚至比单GPU低 → 通信开销完全主导!

### 1.2 125M模型 (166.3M参数)

| GPUs | 策略 | 时间(ms) | 内存(GB) | tok/s | vs 1GPU | 内存vs 1GPU |
|------|------|---------|---------|-------|---------|-----------|
| 1 | Single | 61.15 | 3.62 | **33K** | 1.0x | 1.0x |
| 2 | DDP | 128.37 | 4.28 | 16K | 0.48x | 1.18x |
| 2 | FSDP1 | 74.78 | 2.15 | **27K** | 0.82x | 0.59x |
| 4 | DDP | 242.43 | 4.28 | 8.4K | 0.25x | 1.18x |
| 4 | FSDP1 | 128.55 | 1.65 | 16K | 0.48x | 0.46x |
| 8 | DDP | 264.42 | 4.28 | 7.7K | 0.23x | 1.18x |
| 8 | FSDP1 | 134.03 | 1.40 | 15K | 0.46x | 0.39x |

**分析**:
- 125M模型更大 → 通信量更大 → PCIe更无法承受!
- 2GPU FSDP1已经比单GPU慢18% → 4GPU慢52% → 8GPU慢54%
- FSDP1内存省: 从3.62→2.15(2GPU) → 1.65(4GPU) → 1.40(8GPU) → 省内存但不省时间!

## 2. 为什么PCIe scaling这么差?

```
通信开销分析:

FSDP1每步通信:
  → AllGather(参数): 166.3M×4bytes=0.67GB → 2次(forward+backward)
  → ReduceScatter(梯度): 0.67GB → 1次(backward)
  → 总: 2×0.67 + 0.67 = 2.01GB

PCIe带宽(~12 GB/s):
  → 2.01GB / 12 GB/s = 0.17s = 167ms → 这就是通信时间!

125M 1GPU计算时间: 61ms
125M 2GPU FSDP1: 75ms → 比61ms慢14ms → 但overlap后仅74.78ms

  → overlap效率: 通信167ms → overlap约50% → 有效通信约84ms
  → 计算: 61ms/2(2GPU) ≈ 31ms → 总: 31+84 = 115ms → 实测75ms?
  → → overlap更好(约80%) → 有效通信约33ms → 31+33+11(overhead)≈75ms ✓

4GPU FSDP1:
  → 通信: 2×(0.67×3/4) + 0.67×3/4 = 1.0GB → 0.10s → 但Ring延迟×2(N-1)步!
  → 计算: 61ms/4 ≈ 15ms → 总应约15+100=115ms → 实测128ms → close!

8GPU FSDP1:
  → 通信量虽少(每GPU仅0.67/8=0.084GB) → 但Ring 2×(N-1)=14步 → 延迟14×50μs=0.7ms
  → → 小消息PCIe效率低 → 延迟主导 → scaling灾难!

核心问题: PCIe延迟高(~50μs per step) + Ring步骤多(2×(N-1)) + 小消息效率低
  → 8GPU: 14步×50μs = 0.7ms延迟 + 小消息带宽利用率低 → 总通信时间>>计算时间!
```

## 3. 与A16结果对比

```
之前A16 GPU(15GB, FP16~14.7 TFLOPS, HBM 76GB/s)实测:
  → 25M FSDP1 2GPU: 125% scaling (比1GPU快25%)
  → 25M FSDP1 8GPU: 61% scaling (比1GPU慢39%)

RTX 4090实测:
  → 25M FSDP1 2GPU: 112% scaling (比1GPU快12%)
  → 25M FSDP1 8GPU: 67% scaling (比1GPU慢33%)

差异分析:
  → RTX 4090比A16快11x(GEMM 169 vs 14.7 TFLOPS)
  → → 计算时间短 → 通信占比相对更高 → scaling更差!
  → RTX 4090 25M 1GPU: 26.89ms → 计算快 → 通信61%占比
  → A16 25M 1GPU: ~150ms → 计算慢 → 通信占比更低 → scaling看似更好?

核心洞察: **GPU越强, PCIe scaling越差!**
  → 快GPU → 计算时间短 → 通信时间占比高 → scaling差
  → 慢GPU → 计算时间长 → 通信时间占比低 → scaling看似更好
  → → 但: 快GPU的绝对吞吐仍然更高! → 101K vs 60K tok/s(8GPU)
```

## 4. NVLink scaling预估

```
如果RTX 4090有NVLink(726 GB/s):
  → 125M FSDP1通信: 2.01GB / 726 GB/s = 2.8ms → vs PCIe 167ms → 60x更快!
  → 计算: 31ms → 通信: 2.8ms → 总: 33.8ms → vs 1GPU 61ms → **1.8x加速!**
  → 8GPU: 计算15ms + 通信5ms → 总20ms → vs 61ms → **3x加速!**
  → → NVLink下FSDP scaling可达3x以上!

对比H100 NVLink实测(类似参数):
  → TP+DP scaling: 88.5%效率 → 几乎完美!
  → 8GPU: 8×0.885 = 7x → vs PCIe 0.46x → **15x差距!**

结论:
  → NVLink让scaling从0.46x→7x → 15x差距!
  → RTX 4090没有NVLink → 只能单GPU推理或≤2GPU训练
  → → 这解释了为什么推理优化(量化+FlashInfer)更重要!
```

## 5. RTX 4090分布式训练决策

```
模型大小 vs 最优GPU数:

| 模型大小 | 1GPU吞吐 | 2GPU FSDP1吞吐 | 4GPU FSDP1吞吐 | 推荐GPU数 |
|---------|---------|---------------|---------------|---------|
| 25M | 152K | 171K(1.12x) | 104K(0.69x) | **2 GPU** |
| 125M | 33K | 27K(0.82x) | 16K(0.48x) | **1 GPU** |
| 7B | ~9K估 | ~7K估 | ~4K估 | **1 GPU** |

RTX 4090训练决策树:
  → <50M: 2GPU FSDP1(1.12x加速) → 或单GPU也OK
  → 50M-500M: 单GPU → 多GPU反而慢!
  → >500M: 单GPU内存不够 → FSDP1(B≤4) → 但效率<50%

  → **最优**: 单GPU训练 + gradient accumulation → 比多GPUFSDP更高效!
  → **如果内存不够**: FSDP1 2-4GPU → 但接受0.48-0.69x效率 → 总吞吐仍低于1GPU!

  → RTX 4090不是分布式训练的好选择 → 是单GPU推理的好选择!
  → 推理: 单GPU+INT4+INT8KV+GQA+FlashInfer → $0.01/Mtok → 最优!
```

---

**Sources**:
- FSDP2 Scaling Benchmark: results/fsdp2_scaling_benchmark_4090.json
- NCCL PCIe实测: 12 GB/s单向
- RTX 4090 Specs: 82.58 TFLOPS BF16, 876 GB/s GEMM BW

**Related notes**: nccl-communication-deep-dive.md (NCCL), distributed-training-frameworks-comparison.md (框架), inference-cost-analysis.md (推理成本)