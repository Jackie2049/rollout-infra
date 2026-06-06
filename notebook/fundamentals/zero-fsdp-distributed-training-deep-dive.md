# ZeRO & FSDP Distributed Training Memory Optimization

> 2026-06-07 | 基于 Rajbhandari et al. (SC 2020, ZeRO) + PyTorch FSDP docs
> 已有模拟器结果 (7B ZeRO-3 DP=8→14.5GB, 70B需128+GPU), 本文聚焦算法层面

## 1. 问题: 大模型训练的内存瓶颈

**单 GPU 训练 70B 模型的内存需求**:
```
参数:       70B × 2 bytes (FP16) = 140 GB
梯度:       70B × 2 bytes = 140 GB
Adam 优化器: 70B × 12 bytes (2×FP32 states + FP32 master) = 840 GB
激活值:     ≈ 数GB (取决于 batch/seq/gradient checkpointing)

总计: ≈ 1120 GB → 远超单 GPU 的 80 GB (A100) 或 24 GB (RTX 4090)
```

**传统 DP (Data Parallel)**: 每个GPU 拥有完整模型副本 → N GPU 训练70B 需 N≥14 (A100)
→ **浪费**: N 个GPU 各存 N份相同的优化器状态 → 总内存 N × 1120 GB

**ZeRO 的洞察**: DP 的冗余内存可以用分片消除!

## 2. ZeRO 三阶段: 逐步消除冗余

### ZeRO Stage 1: 分片优化器状态

```
优化器状态总量: 12 bytes/param × 70B = 840 GB
分片到 N GPU: 每GPU只需 840/N GB

通信: ReduceScatter (梯度聚合后只保留本地分片)
      → 通信量与标准 DP 相同 (AllReduce = ReduceScatter + AllGather)

内存节省: 优化器状态从 840GB → 840/N GB
7B on 8 GPU: 84GB → 10.5GB (但参数+梯度仍是 28GB → 总38.5GB)
```

### ZeRO Stage 2: 分片优化器 + 梯度

```
优化器+梯度: (12 + 2) bytes/param × 70B = 980 GB
分片到 N GPU: 每GPU只需 980/N GB

通信: ReduceScatter (梯度分片聚合)
      保存梯度时只保存本地分片 → 立即释放非本地梯度内存

内存节省: 优化器+梯度从 980GB → 980/N GB
7B on 8 GPU: 98GB → 12.25GB (参数仍是 14GB → 总26.25GB)
```

### ZeRO Stage 3: 分片优化器 + 梯度 + 参数

```
全部: (12 + 2 + 2) bytes/param × 70B = 1120 GB
分片到 N GPU: 每GPU只需 1120/N GB

通信: 前向: AllGather (收集需要的参数层)
      反向: AllGather (收集参数) + ReduceScatter (梯度分片)
      → 通信量增加: 前向和反向各多一次 AllGather

内存节省: 全部从 1120GB → 1120/N GB
7B on 8 GPU: 1120GB → 140GB per GPU? No!
             实际: 70B × 16/N = 16B × 16/8 = 32 bytes/param → 7B × 32/8 = 28GB → 14GB per GPU
```

Wait, 修正计算:

```
ZeRO-3 单GPU内存: 1120/N = 1120/8 = 140GB (70B on 8 A100)
→ 仍超 80GB → 需更多 GPU!

ZeRO-3 on 16 GPU: 1120/16 = 70GB → 可以在 A100 上训练 70B!
ZeRO-3 on 128 GPU: 1120/128 = 8.75GB → 很轻松
```

**关键**: ZeRO-3 的内存 ∝ 1/N → GPU 越多, 每GPU内存越小

### 验证 (7B 模型):
```
DP:         7B × 16 = 112 GB (不能训练!)
ZeRO-1 DP=8: 7B × (2+2+12/8) = 7B × 5.5 = 38.5 GB (仍超 24GB for RTX 4090)
ZeRO-2 DP=8: 7B × (2+2/8+12/8) = 7B × 3.5 = 24.5 GB (刚好!)
ZeRO-3 DP=8: 7B × (2/8+2/8+12/8) = 7B × 2 = 14 GB (OK!)

模拟器实测: 7B ZeRO-3 DP=8 → 14.5GB (基本吻合)
```

## 3. ZeRO 通信开销分析

### 标准 DP (AllReduce):
```
前向: 无额外通信
反向: AllReduce (梯度聚合)
通信量: 2 × M × (N-1)/N bytes (ring AllReduce)

对 70B: 2 × 140GB × (N-1)/N ≈ 280GB/N × (N-1) ≈ 280GB (跨 NVLink 很快)
```

### ZeRO-1/2 (ReduceScatter):
```
反向: ReduceScatter (梯度分片)
通信量: 同 AllReduce! (ReduceScatter = AllReduce 的一半)

对 70B: ≈ 280GB/N × (N-1) ≈ 同 DP
```

### ZeRO-3 (AllGather + ReduceScatter):
```
前向: AllGather (参数层按需收集) — 每层一次
反向: ReduceScatter (梯度分片) + AllGather (反向参数)

通信量: ≈ 3 × M × (N-1)/N bytes (3x vs DP!)

对 70B: ≈ 3 × 280GB/N × (N-1) ≈ 840GB (3x more communication)

但这不是 3x 慢, 因为:
1. AllGather 是 layer-by-layer → GPU 计算和通信可以重叠
2. 参数 AllGather 后立即释放 → 内存峰值低
3. 通信量增加但带宽足够 (NVLink 300 GB/s)
```

### ZeRO-Infinity: GPU → CPU → NVMe 分片

```
当 GPU 内存不够 ZeRO-3 分片:
1. 优化器状态 → CPU (DDR4/5, 100+ GB available)
2. 参数 → NVMe (SSD, 1+ TB available)

代价: 通信延迟增加 (PCIe ~32 GB/s, NVMe ~7 GB/s)
收益: 单GPU 可训任意大模型 (只要有足够 CPU/NVMe)
```

## 4. PyTorch FSDP: ZeRO-3 的 PyTorch 实现

### FSDP 核心 API:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    device_id=local_rank,
)
```

### ShardingStrategy:
```
NO_SHARD:     DP (不分片) → ZeRO-0
SHARD_GRAD_OP: 分片梯度+优化器 → ZeRO-2
FULL_SHARD:   分片全部 → ZeRO-3
HYBRID_SHARD: 组内 ZeRO-3, 组间 DP → ZeRO-3 + DP 混合
```

### FSDP 前向流程:
```
1. AllGather (收集本层的参数 shard) → GPU 有完整层参数
2. 前向计算 (用完整参数)
3. 释放非本地参数 shard → 只保留本地 shard

这实现了 "参数按需收集, 用完释放" 的内存优化
```

### FSDP 反向流程:
```
1. AllGather (收集参数, 同前向)
2. 计算梯度 (用完整参数)
3. ReduceScatter (梯度分片) → 每GPU只保留本地梯度 shard
4. 释放非本地参数和梯度
5. 本地优化器更新本地 shard 的参数

这实现了 "梯度分片聚合, 参数分片更新" 的完整 ZeRO-3 流程
```

### FSDP 关键配置:
```python
# 激活值内存优化
use_activation_checkpointing=True  # gradient checkpointing → 省内存, 增计算

# 模型包装策略
auto_wrap_policy=transformer_auto_wrap_policy  # 按Transformer层分片
# 每层独立分片 → AllGather/ReduceScatter 只收集/释放当前层

# Mixed precision
mixed_precision=MixedPrecision(
    param_dtype=torch.float16,    # 参数 FP16
    reduce_dtype=torch.float16,   # 梯度 reduce FP16
    buffer_dtype=torch.float32,   # buffer FP32
)
```

### FSDP vs DeepSpeed ZeRO:

| 特性 | PyTorch FSDP | DeepSpeed ZeRO |
|------|-------------|---------------|
| 实现级别 | PyTorch native | 第三方库 |
| 分片粒度 | 模块级 (per layer) | 参数级 |
| Offload | CPU offload (有限) | CPU + NVMe offload |
| 通信重叠 | 自动 (CUDA stream) | 手动配置 |
| 易用性 | 高 (PyTorch API) | 中 (需 config) |
| 生态 | PyTorch 官方 | Microsoft/DeepSpeed |

## 5. 选择并行策略: 决策树

```
模型大小 → GPU 数量 → 网络带宽 → 策略选择

单 GPU 内存:
  7B FP16 ≈ 14GB (fits RTX 4090)
  13B FP16 ≈ 26GB (需要 A100)
  70B FP16 ≈ 140GB (绝对需要多GPU)

策略选择 (70B 模型为例):
  8 × A100 (NVLink):
    DP=8 + ZeRO-1: 不够 (38.5GB/GPU)
    DP=8 + ZeRO-2: 不够 (26.25GB/GPU)
    DP=8 + ZeRO-3: 不够 (140GB/GPU)
    → 需 TP+ZeRO 或更多GPU

  16 × A100 (NVLink):
    DP=16 + ZeRO-3: 70GB/GPU → 可以! 但通信 3x DP

  128 × A100 (混合 NVLink+Ethernet):
    TP=8 + DP=16 + ZeRO-3: 每GPU ≈ 8.75GB → 轻松!
    通信: TP (NVLink) + DP AllReduce (Ethernet)

  更大 (175B+):
    TP=8 + PP=16 + DP=8 + ZeRO-3: 3D并行
```

### 实用决策:
```
1. 模型 < 单GPU内存 → DP (不需要 ZeRO)
2. 模型 ≈ 2-4× 单GPU → TP=2-4 (NVLink 必需)
3. 模型 ≈ 8-16× 单GPU → TP+DP 或 ZeRO-3+DP
4. 模型 > 16× 单GPU → TP+PP+DP+ZeRO (3D并行)

NVLink 的重要性:
  TP 需要 NVLink (通信 <5% 训练时间)
  ZeRO-3 不一定需要 NVLink (但 3x 通信量需要高速网络)
  没有 NVLink → TP 效率很低 (5-7.5 GB/s PCIe → 60x slower)
```

## 6. 训练延迟估算

### 简单 Roofline 模型:
```
每步时间 = max(compute_time, communication_time)

Compute: 6 × N × B × seq_len × throughput / (N_GPU × peak_FLOPS × MFU)
Communication: ZeRO-3 ≈ 3 × 2 × model_size × (N_GPU-1) / bandwidth

70B, B=1, seq=4K, A100×128:
Compute = 6 × 70B × 4K × 128 / (128 × 312 TFLOPS × 0.5) ≈ 5.4s/step
Communication = 3 × 140GB × 127 / 300GB/s ≈ 1.4s (NVLink)
→ Compute-dominated → ZeRO-3 通信开销 <30%
```

## 7. 关键 Takeaways

1. **ZeRO 消除 DP 冗余**: DP 每GPU存完整优化器 (78% 内存), ZeRO 分片后 ∝ 1/N
2. **ZeRO-3 内存公式**: 每GPU内存 ≈ (2+2+12)/N × model_params = 16/N × P bytes
3. **ZeRO-3 通信开销**: 3x DP → 但 layer-by-layer 通信可与计算重叠
4. **FSDP = PyTorch native ZeRO-3**: 更易用, 但 offload 能力不如 DeepSpeed
5. **选择策略**: 模型小 → DP, 中 → TP+ZeRO, 大 → TP+PP+ZeRO
6. **NVLink 是 TP 必需**: 无 NVLink → TP 效率暴跌 (实测 RTX 4090 PCIe 仅 5-7.5GB/s)
7. **Gradient checkpointing + ZeRO-3**: 组合使用, 省内存 ≈ 4x (14%计算开销)
8. **ZeRO-Infinity**: CPU/NVMe offload → 单GPU可训任意大模型, 但慢 (PCIe瓶颈)