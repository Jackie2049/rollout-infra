# Distributed Training Scaling Math — Theory + RTX 4090 Data

> 2026-06-08 | 分布式训练scaling的精确数学模型, 整合实测数据
> 关键: PCIe scaling灾难有精确数学解释, NVLink vs PCIe = 6.6x差距

## 1. Scaling Efficiency Model

### 基本scaling公式

```
N GPU ideal speedup = N (完美线性scaling)
实际 speedup = N / (1 + α(N-1))

其中 α = comm_time / compute_time (通信/计算比)

当 α=0: speedup=N (无通信开销 → 完美scaling)
当 α=0.5: speedup=N/(1+0.5(N-1)) → 2GPU=1.33x, 4GPU=1.6x, 8GPU=2x
当 α=1: speedup=N/(1+(N-1))=1 (通信=计算 → 完全无法scaling!)
当 α>1: speedup<1 → 多GPU反而慢! (RTX 4090 PCIe场景)
```

### RTX 4090实测α值

| 模型 | 计算时间(ms) | 通信时间(ms) | α | N=2 | N=4 | N=8 |
|------|------------|------------|-----|-----|-----|-----|
| 25M BF16 | 20 | 2.2 | 0.11 | 1.80 | 3.2 | 5.7 |
| 125M BF16 | 50 | 2.2 | 0.04 | 1.93 | 3.7 | 7.1 |
| 7B BF16 | 200 | 2.2 | 0.011 | 1.98 | 3.9 | 7.7 |

Wait — 这些α值看起来scaling应该好, 但实测灾难性 → 为什么?

**关键修正**: FSDP通信量 = model_size / N × 2 (RS+AG)
- 对于7B BF16: 15.6GB / N × 2 × bandwidth → 通信时间随N变化
- RS+AG per layer per step: shard大小 = model_per_layer / N

```
实际α(N) = comm_time(N) / compute_time(N)

对于FSDP ZeRO-3 per step:
  compute_time ≈ constant (每GPU处理shard → compute∝shard∝1/N)
  comm_time = 2 × shard_size / bandwidth + launch_overhead

  Shard_size = model_size / N
  comm_time(N) = 2 × (model_size/N) / BW + fixed_overhead

  当model小 → compute∝1/N → compute小 → α大 → scaling差
  当model大 → compute∝1/N → compute大 → α小 → scaling好?
  → 但FSDP每GPU仍然做全量forward → compute不随N减少!
  → 实际: FSDP compute_time = full_forward_time (不变!)
  → FSDP comm_time = 2 × shard_size / BW (随N减少)
  → α(N) = comm_time / full_forward_time
```

## 2. FSDP ZeRO-3 Scaling Math (精确)

### Per-step cost分析

```
FSDP ZeRO-3 per step (forward + backward):

Forward:
  AG(param_shard) → 全参数 → compute full forward → RS(activation_shard)

Backward:
  AG(param_shard) → compute backward → RS(grad_shard)

Total per step:
  2×AG + 2×compute + 2×RS

  Without overlap: 2×(AG + compute + RS)
  With overlap:    max(RS, compute) + AG + max(RS, compute) + AG
                   = 2×AG + 2×max(RS, compute)

Overlap效率 = (sequential - overlapped) / (sequential - theoretical_min)
RTX 4090实测: 50-79%
```

### RTX 4090 PCIe实测数据(8GPU)

```
AllReduce带宽:
  8GPU, 100MB: 2.76 GB/s
  2GPU, 16MB: 6.99 GB/s
  → 2GPU比8GPU快2.5x! → 更多GPU = 更多跨GPU通信 = 更慢

FSDP 7B B=32 8GPU:
  compute ≈ 13ms (decode, 每GPU独立)
  comm ≈ 2 × (15.6GB/8) / 2.76GB/s = 2 × 2.0 / 2.76 = 1.45s
  → comm_ratio = 99.4%!
  → speedup = 0.50x → 比1GPU慢2x!

  α = 1.45s / 13ms = 111.5 → α >> 1 → scaling灾难性!

FSDP 25M 2GPU:
  compute ≈ 20ms
  comm ≈ 2 × (100MB/2) / 7GB/s ≈ 0.014ms
  → α = 0.014 / 20 = 0.0007 → 接近0 → scaling接近线性!
  → 实测: 1.12x → 接近1但不够 → 因为还有其他开销(launch, sync)
```

### NVLink vs PCIe对比

```
PCIe RTX 4090:
  AllReduce 8GPU 100MB: 2.76 GB/s
  → α(7B) ≈ 111 → scaling灾难性

NVLink H100:
  AllReduce 8GPU 100MB: ~300 GB/s (NVLink)
  → α(7B) = 2 × (15.6/8) / 300 / 0.013 = 0.16
  → speedup ≈ N / (1+0.16(N-1))
  → 8GPU: 8/(1+1.12) = 3.6x → 实际可达3-4x

差距: NVLink 3.6x vs PCIe 0.50x = **7.2x差距!**
与实测6.6x差距一致(NVLink预估3.3x vs PCIe 0.50x)
```

## 3. Communication-Compute Overlap Math

### Overlap efficiency model

```
Two operations: compute(C) and communication(M)
Sequential: C + M
Overlapped: max(C, M) + δ

其中 δ = overlap overhead (stream切换, sync等)

理论最佳: max(C, M)
实际: max(C, M) + δ

Overlap efficiency = (C+M - (max(C,M)+δ)) / (C+M - max(C,M))

RTX 4090实测:
  C=426us, M=2290us(16MB AllReduce)
  Sequential = 2723us
  Overlapped = 2428us
  δ = 2428 - 2290 = 138us (stream切换开销!)
  Efficiency = (2723-2428)/(2723-2290) = 295/433 = 68.1%

  δ ≈ 138us → 消费级GPU stream切换开销 → 不完美但有效
  NVLink预估: δ ≈ 20us (更快的stream调度) → efficiency接近95%
```

### FSDP overlap model

```
FSDP per-step:
  Without overlap: AG + compute + RS = AG + max(RS, compute) + min(RS, compute)
  With overlap:    AG + max(RS, compute) + δ_overlap

  δ_overlap = min(RS, compute) - (overlap - max(RS, compute))
  → 实际: δ ≈ 100-200us (RTX 4090 stream overhead)

RTX 4090实测:
  RS=1326us(16MB), AG=1327us, compute=426us
  Sequential = 2135us
  Overlapped = 1496us
  → 30% time saved! → overlap有效

  但: AG不能overlap → 必须等AG完成才能compute
  → AG占总时间的60%+ → overlap只能省RS部分 → 限制scaling改善

  AG占比 ∝ 1/bandwidth → PCIe AG占60% → NVLink AG占5% → scaling差距巨大
```

## 4. Optimal N GPU Decision (RTX 4090)

```
scaling_formula: speedup(N) = N / (1 + α(N))

α取决于:
  1. model_size / compute_intensity → 大模型α小 → 小模型α大?
     → 错! FSDP compute不随N减少 → α ∝ comm_time(不变) / compute_time(不变)
     → α ≈ comm_time / compute_time → 通信固定开销决定scaling

  2. bandwidth → NVLink α≈0.16 → PCIe α≈111 → 差距700x!

  3. batch_size → B越大compute越长 → α越小 → scaling越好
     → 但7B decode=13ms(B=32) vs comm=1.45s → α=111 → 仍然灾难性

RTX 4090决策树:
  N=1: 最优(无通信开销) → 推理首选
  N=2: 勉强(25M模型, FSDP1=1.12x) → 训练小模型
  N=4: 不可行(125M FSDP1=0.48x → 比单GPU慢2x!)
  N=8: 灾难性(7B FSDP1=0.50x → 比单GPU慢2x!)

  → RTX 4090 PCIe = 单GPU推理或≤2GPU小模型训练
  → 多GPU训练需要NVLink → H100/A100

NVLink决策树:
  N=2: 可行(7B speedup≈1.7x)
  N=4: 可行(7B speedup≈2.5x)
  N=8: 可行(7B speedup≈3.6x)
  → NVLink scaling可行但非线性 → α≈0.16 → overhead约16%
```

## 5. Core Scaling Laws Summary

```
Scaling Law 1: speedup = N / (1 + α(N-1))
  α = comm/compute → 决定scaling效率
  α<0.1 → 接近线性 → NVLink场景
  α>1 → 灾难性 → PCIe场景

Scaling Law 2: α ∝ model_size / (bandwidth × compute_time)
  大模型 → compute大 → α小 → scaling好 (需要NVLink!)
  PCIe → bandwidth小 → α大 → scaling差 (不论模型大小!)

Scaling Law 3: overlap效率 = 1 - δ/min(C,M)
  δ = stream切换开销 → RTX 4090 δ≈100-200us → overlap 65-84%
  NVLink δ≈20us → overlap接近95%
  → overlap改善有限: 只省min(C,M)部分 → max(C,M)不变

Scaling Law 4: AG不能overlap → AG占总时间比例:
  PCIe: AG占比60%+ → overlap改善<40%
  NVLink: AG占比5% → overlap改善>90%
  → AG是真正瓶颈 → 需要NVLink → overlap只是锦上添花

Scaling Law 5: FSDP scaling灾难根因:
  PCIe bandwidth = 2.76 GB/s (8GPU)
  7B model shard = 15.6GB/8 = 2GB per GPU
  AG time = 2GB / 2.76GB/s = 0.73s per step
  → AG占99%+ → compute只有1% → 完全无法scaling
  → NVLink: AG = 2GB / 300GB/s = 6.7ms → 占5% → 可scaling

  数学结论: PCIe GPU不适合多GPU训练 → 推理是唯一出路
  → RTX 4090 = 推理专用GPU → 训练需H100/A100
```