# Distributed Training Frameworks Comparison: Megatron-LM vs DeepSpeed ZeRO vs FSDP2

> 2026-06-08 | 三大分布式训练框架架构对比+RTX 4090实测+决策树
> 基于: Megatron-LM源码阅读, DeepSpeed文档, PyTorch FSDP2文档, RTX 4090 benchmark数据
> 关联: zero-fsdp-distributed-training-deep-dive.md (ZeRO理论), fsdp.md (FSDP基础), megatron-* (源码阅读)

## 0. 核心结论

```
<10M参数 → DDP(最简单, 无分片开销)
10-100M → FSDP1(内存省49%+1.51x加速, PCIe可用)
100M-10B → FSDP2(per-parameter sharding+torch.compile)
10B-70B → FSDP2+TP(混合并行, 比纯DP更优)
>70B → Megatron-LM(TP+PP+DP, 1000+GPU最优)
极端内存受限 → DeepSpeed ZeRO-3+NVMe offload(唯一能fit的方法)
```

## 1. 三框架架构对比

### 1.1 通信-计算模式

| 特性 | Megatron-LM | DeepSpeed ZeRO | FSDP2 |
|------|-------------|---------------|-------|
| **核心策略** | TP+PP+DP | ZeRO分片(DP变体) | ZeRO-3式分片 |
| **通信模式** | P2P+AllReduce | ReduceScatter+AllGather | ReduceScatter+AllGather |
| **通信量(每层)** | 2×AllReduce(TP) | 1×AR(grad)+2×AG(param) | 1×AR+2×AG |
| **分片粒度** | 无分片(每GPU全参数) | 优化器/梯度/参数分片 | per-parameter分片 |
| **内存省** | TP省(N分之1参数) | ZeRO-3省8x(12+2+2/N) | ZeRO-3等价 |
| **重叠** | 通信与计算异步重叠 | 有限重叠 | backward prefetch |
| **Offload** | 无 | CPU/NVMe offload | 无(纯GPU) |

### 1.2 通信量对比

```
Megatron-LM(TP=N):
  每层: AllReduce(梯度) = 2×(N-1)/N × 模型大小 → N=4: 1.5×模型大小
  → 实测: TP通信占比~11.5%(NVLink), 39-83%(PCIe!)

DeepSpeed ZeRO-3:
  Forward: AllGather(参数) = 1×模型大小
  Backward: AllGather(参数) + ReduceScatter(梯度) = 1.5×模型大小
  → 通信量3x vs DDP, 但内存省8x

FSDP2(per-parameter):
  Forward: AllGather(每层参数) → 层级聚合
  Backward: ReduceScatter(梯度) + AllGather(prefetch下一层参数)
  → 与ZeRO-3等价通信量, 但per-parameter更灵活
```

## 2. Megatron-LM深度分析

### 2.1 核心并行策略

```
TP(Tensor Parallelism):
  → ColumnParallelLinear: 切输出维 → 每GPU算1/N列
  → RowParallelLinear: 切输入维 → 每GPU算1/N行 → 端AllReduce
  → 每层1次AllReduce → NVLink下11.5%通信占比 → 高效!

PP(Pipeline Parallelism):
  → GPipe: 所有micro-batch前向 → 所有反向 → 气泡大!
  → 1F1B: 1个前向+1个反向交替 → 气泡=(P-1)/(M+P-1) → 更小
  → Interleaved: 多虚拟stage → 气泡更小 → 但通信量增加

DP(Data Parallelism):
  → 各pipeline replica独立 →梯度AllReduce → 最外层并行

组合: TP×PP×DP = 总GPU数
  → TP需NVLink → PP需P2P → DP需任意连接
  → RTX 4090 PCIe: TP/PP通信39-83% → 不划算 → DDP/FSDP更优!
```

### 2.2 Megatron源码关键发现

```
已阅读的核心模块(9个笔记):
  - TP: ColumnParallel/RowParallel + 通信映射(_CopyTo/_ReduceFrom)
  - PP: GPipe/1F1B/Interleaved + P2P(ring_exchange最快) + deallocate
  - SP: Sequence Parallelism → activation沿seq切分 → AllGather+RS=AR
  - Async: CUDA_DEVICE_MAX_CONNECTIONS=1 → 通信先调度 → 与计算重叠
  - Optimizer: distributed Adam optimizer + gradient accumulation
  - MoE: expert parallelism + token dispatch + expert buffer管理

关键: async_op=True + CUDA_DEVICE_MAX_CONNECTIONS=1
  → 所有NCCL操作先入队 → GPU scheduler优先执行通信 → 与计算重叠!
  → 这是Megatron性能的关键: 不是减少通信量, 而是重叠!
```

## 3. DeepSpeed ZeRO深度分析

### 3.1 ZeRO三阶段(RTX 4090实测)

```
ZeRO Stage 1 (优化器分片):
  → Adam 12B/param占78% → 分片后每GPU 12/N B/param
  → 通信: ReduceScatter(与DP AllReduce等价) → 无额外通信!
  → RTX 4090实测: 7B/8GPU → 10.5GB optimizer + 14GB param+grad → 24.5GB

ZeRO Stage 2 (优化器+梯度分片):
  → 每GPU: (12+2)/N B/param → 梯度用完立即释放 → 内存省更多
  → RTX 4090实测: 7B/8GPU → 12.25GB + 14GB param → 26.25GB

ZeRO Stage 3 (优化器+梯度+参数分片):
  → 每GPU: (12+2+2)/N = 16/N B/param → 8x总内存省!
  → 但: Forward/Backward各多一次AllGather → 通信3x DP
  → RTX 4090实测: 7B/8GPU → 2GB optimizer+grad+param → 总14GB
```

### 3.2 DeepSpeed Offload

```
CPU Offload: 优化器状态→CPU内存 → GPU只存参数+梯度
  → 70B训练: GPU只需14GB(参数)+14GB(梯度) → 40GB GPU够用!
  → 但: optimizer step在CPU → 每步~1-10s → 比GPU慢10-100x

NVMe Offload: 优化器+梯度→NVMe SSD → GPU只存当前层参数
  → 70B训练: GPU只需~2GB → 24GB RTX 4090够用!
  → 但: SSD读写~1GB/s → 比GPU HBM 900GB/s慢900x → 极慢!

适用场景:
  → CPU offload: 中等模型(7-30B) + GPU内存刚好不够 → 可行
  → NVMe offload: 大模型(>70B) + GPU内存严重不足 → 最后手段
  → RTX 4090: 7B用ZeRO-3+CPU offload → 但实测ZeRO-3无offload更快!
```

### 3.3 DeepSpeed vs FSDP关键差异

| 特性 | DeepSpeed ZeRO-3 | FSDP2 |
|------|-----------------|-------|
| 分片粒度 | flat parameter分片 | per-parameter分片 |
| AllGather策略 | 整层AllGather | 层级+prefetch |
| Offload | CPU/NVMe支持 | 无 |
| torch.compile | 不兼容 | ✅原生兼容 |
| 混合精度 | FP16 AMP + BF16 | BF16原生 |
| 依赖 | 外部包(deepspeed) | PyTorch内置 |
| 配置 | ZeRO config JSON | Python API |
| 通信优化 | 有限 | backward prefetch |

**FSDP2优势**: per-parameter sharding → 更灵活分片 → torch.compile兼容 → 无外部依赖!

## 4. FSDP2深度分析

### 4.1 FSDP1 vs FSDP2对比(RTX 4090实测)

```
FSDP1(flat parameter):
  → 整层参数flat → AllGather整层 → 用完释放 → 内存省但通信粗糙
  → RTX 4090实测: 25M/2GPU → 1.51x加速(vs DDP) + 49%内存省
  → Scaling: 2GPU=125%→8GPU=61%(PCIe瓶颈!)

FSDP2(per-parameter):
  → 每个参数独立分片 → AllGather更灵活 → backward prefetch下一层
  → 与torch.compile兼容 → forward 3.75x + training 1.31x加速!
  → BF16原生 → 不需要AMP(FP16反而更慢!)
  → 2025推荐: 大多数场景用FSDP2
```

### 4.2 FSDP2关键优化

```
1. Backward Prefetch:
   → Backward计算当前层时, AllGather下一层参数 → 计算与通信重叠!

2. Per-Parameter Sharding:
   → 不同参数可以不同分片策略 → 更灵活内存管理
   → 大层细粒度分片 → 小层不分片 → 自适应!

3. torch.compile兼容:
   → FSDP2 + torch.compile → forward 3.75x加速!
   → 但: backward不被compile(动态shape) → training仅1.31x

4. BF16原生:
   → BF16 + FSDP = 最佳组合! 比FP16 AMP更好
   → BF16不需要loss scaling → 更稳定 → RTX 4090实测验证
```

### 4.3 FSDP2 + TP混合

```
2025新趋势: FSDP2 + torch.distributed.tensor.parallel
  → FSDP2做ZeRO-3式分片 + TP做切层并行
  → 70B模型: FSDP2(8GPU) + TP(2) = 16GPU → 更优scaling!
  → 通信: FSDP2 AG/RS + TP AllReduce → 但可重叠!

RTX 4090 PCIe限制:
  → TP通信39-83% → PCIe不适合TP
  → 但FSDP2通信量小(只AllGather参数) → PCIe可以!
  → 所以: RTX 4090用FSDP2, 不用TP!
```

## 5. RTX 4090实测数据汇总

### 5.1 DDP vs FSDP vs ZeRO

| 方法 | 模型 | GPU数 | 加速 | 内存 | 备注 |
|------|------|-------|------|------|------|
| DDP | 25M | 2 | 基准 | 基准 | 最简单 |
| FSDP1 | 25M | 2 | **1.51x** | 49%省 | BF16+FSDP最佳 |
| FSDP1 | 25M | 4 | 1.25x | 62%省 | Scaling下降 |
| FSDP1 | 25M | 8 | 0.61x | 75%省 | PCIe瓶颈! |
| ZeRO-3 | 7B | 8 | — | 8x省 | 通信3x DP |
| TP | 7M | 2 | 0.89x | 50%省 | PCIe TP慢 |
| Ring Attn | 7M | 2-8 | 0.03-0.14x | — | PCIe灾难性慢 |

### 5.2 RTX 4090决策树

```
问题: RTX 4090 PCIe用什么分布式策略?

答案:
  <10M → DDP(最简单, 无分片开销, 2GPU=2x加速)
  10-100M → FSDP1(BF16, 2GPU=1.51x+49%内存省)
  100M-7B → FSDP2(per-parameter+torch.compile)
  7B训练 → BF16+FSDP2 + TE FP8(B≥4时1.48-1.59x额外加速!)
  推理 → 单GPU+INT4+INT8 KV(最优部署组合)

不可行:
  → TP(RTX 4090 PCIe): 39-83%通信 → 不划算
  → PP(RTX 4090 PCIe): P2P慢+气泡大 → 不划算
  → Ring Attention: 7-67x慢 → PCIe灾难
  → ZeRO-3+NVMe offload: SSD太慢 → 最后手段
```

## 6. 极大规模训练(>1000 GPU)

### 6.1 Megatron-LM的优势

```
>1000 GPU场景:
  → DP alone → 通信量∝GPU数 → 不高效
  → TP+PP+DP混合 → 通信量只∝TP×PP → 更小!

  例: 2048 GPU, TP=8, PP=16, DP=16
  → TP通信: 8 GPU内(NVLink) → 11.5%
  → PP通信: 16 stage P2P → ~5%
  → DP通信: 16 replica AllReduce → 与DP=16等价 → 可接受!

  vs 纯DP: 2048 GPU AllReduce → 通信量∝2048 → 灾难性慢!
  vs 纯FSDP2: AllGather/RS ∝ 2048 → 同样慢!

  → 结论: >1000 GPU必须用TP+PP+DP混合!
```

### 6.2 何时选什么框架?

```
决策树:
  GPU < 8 → FSDP2(简单+高效+PyTorch原生)
  GPU 8-32 → FSDP2 + TP(混合并行)
  GPU 32-256 → FSDP2 + TP + PP(或Megatron-LM)
  GPU > 256 → Megatron-LM(TP+PP+DP, 最成熟)
  GPU内存不够 → DeepSpeed ZeRO-3 + CPU/NVMe offload

RTX 4090特例:
  → 无NVLink → TP/PP不可行 → 只能FSDP2/ZeRO
  → PCIe带宽低 → 8GPU以上scaling急剧下降
  → 单GPU性能好(890 GB/s HBM) → 单GPU+量化部署更优!
```

## 7. Checkpointing与FSDP交互

```
RTX 4090实测发现:
  → Gradient checkpointing + FSDP → 反而增内存!
  → 原因: checkpointing保存activation → 但FSDP AllGather后释放参数
     → checkpointing需要更多activation空间 → 参数释放延迟 → 内存冲突!

最优: FSDP + no checkpointing = 0.478GB(最省内存!)
  → 7B小模型 → activation不大 → 不需要checkpointing

大模型(>70B): checkpointing必要 → 但需与FSDP2协调:
  → FSDP2 per-parameter sharding → checkpointing与分片更兼容
  → 选择性checkpointing(只checkpoint大层) → 减少activation内存
```

## 8. 2025-2026趋势

```
1. FSDP2成为默认选择:
   → PyTorch原生 + per-parameter + torch.compile兼容
   → 大多数7-70B训练场景最优

2. TP + FSDP2混合:
   → torch.distributed.tensor.parallel + FSDP2
   → 70B+模型的最佳策略

3. DeepSpeed转向辅助角色:
   → ZeRO-3 offload仍是极端场景唯一选择
   → 但大多数场景FSDP2更优 → DeepSpeed社区活跃度下降

4. Megatron-LM继续极端规模主导:
   → NVIDIA旗舰训练方案 → TP+PP+DP不可替代
   → 但配置复杂 → 中小团队用FSDP2+TP

5. FP8训练加速:
   → TE FP8 → 1.48-1.59x → 与任何框架组合使用!
   → FSDP2 + TE FP8 → RTX 4090最优训练方案

6. verl等RL框架:
   → FSDP训练 + vLLM/SGLang推理 → colocation → 50%GPU省
   → sleep/wake机制 → 推理时训练GPU休眠 → 节省资源
```

---

**Sources**:
- [PyTorch FSDP2 Blog](https://pytorch.org/blog/fsdp2/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [ZeRO Paper (SC 2020)](https://arxiv.org/abs/1910.02054)

**Related notes**: zero-fsdp-distributed-training-deep-dive.md (ZeRO理论), fsdp.md (FSDP基础), fsdp2-benchmark-rtx4090.md (实测), megatron-* (源码), rtx4090-pcie-decision-guide.md (决策树)