# ZeRO Algorithm Deep Dive — 数学原理推导 + 跨框架实现对比(DeepSpeed vs FSDP1 vs FSDP2 vs Megatron-LM) + ZeRO+TP+PP组合 + ZeRO-Infinity + 性能分析 + RTX 4090/A100/H100策略选择

> 2026-06-13 | ZeRO算法深度分析: 内存公式严格推导(ψ→(16+K)ψ/N→(2+12+K)/Nψ→...) + 通信量推导(all-reduce=reduce-scatter+all-gather) + 跨框架对比(DeepSpeed Init metaclass+ZeROOrderedDict vs FSDP1 FlatParameter vs FSDP2 per-parameter+DTensor) + ZeRO+TP+PP组合内存公式 + ZeRO-Infinity扩展 + 3D并行策略选择决策树 + RTX 4090/A100/H100最优配置
> 关联: deepspeed-zero-source-reading.md, pytorch-torch-distributed-internals-deep-dive.md, megatron-source-reading.md
> 参考: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020; Rajbhandari et al., "ZeRO-Infinity", SC 2021

## 0. 核心定律: 冗余消除 = 状态分区 + 按需聚合 + 通信不变

```
ZeRO的根本洞察: 标准DDP存在N倍冗余 → 每rank存完整optimizer+gradient+parameter → N个rank=Nx冗余!

ZeRO消除冗余方法: 每rank只存1/N → 消除(N-1)/N的冗余 → 内存减Nx!

但! 计算需要完整参数 → 必须临时聚合(all-gather) → 用完释放(reshard) → 按需!

关键: 虽然临时聚合 → 但通信量不变! → all-reduce = reduce-scatter + all-gather → 等价!

→ → → ZeRO = 内存减Nx + 通信不变 → 以局部聚合的开销换取内存减Nx → 巨大收益!
```

## 1. ZeRO数学原理 — 内存公式严格推导

```
### 1.1 模型状态定义

令Ψ = 模型参数数 (如7B = 7×10^9)
令N_d = 数据并行度 (DP world size)
令K = 临时buffer大小系数 (通常≈0)

混合精度AdamW状态:
  → FP16参数: 2Ψ bytes → 模型权重(低精度)
  → FP16梯度: 2Ψ bytes → 反向传播梯度
  → FP32 master参数: 4Ψ bytes → 精确权重副本
  → FP32 momentum: 4Ψ bytes → AdamW一阶动量
  → FP32 variance: 4Ψ bytes → AdamW二阶动量
  → FP32 gradient(用于optimizer): 4Ψ bytes → 精确梯度副本
  → → 总计: 2Ψ + 2Ψ + 4Ψ + 4Ψ + 4Ψ + 4Ψ = 16Ψ bytes ≈ KΨ额外
  → → → 简化: 总内存 ≈ (16+K)Ψ bytes per rank (标准DDP)

7B模型实例 (Ψ=7×10^9):
  → 16Ψ = 16 × 7×10^9 × 1byte = 112GB → 但BF16=2bytes → 2×16×7×10^9 = 224bytes
  → → 实际: 2Ψ(params)=14GB + 2Ψ(grads)=14GB + 12Ψ(optimizer)=84GB = 112GB per rank
  → → → 注意: 这里Ψ是元素数×2bytes(BF16) → 总计≈112GB per rank → 远超24GB RTX 4090!

### 1.2 ZeRO-1 (P_os): Optimizer State Partitioning

内存公式推导:
  → 参数: 2Ψ → 不分区 → 每rank存完整副本 → 2Ψ/N_d... 不! 参数完整!
  → → 修正: ZeRO-1只分区optimizer → 参数+梯度仍完整!
  → → → ZeRO-1 per-rank内存:
    | 状态           | 标准DDP  | ZeRO-1       |
    | 参数(FP16)     | 2Ψ      | 2Ψ (不变)    |
    | 梯度(FP16)     | 2Ψ      | 2Ψ (不变)    |
    | Master参数(FP32)| 4Ψ     | 4Ψ/N_d       |
    | Momentum(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Variance(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Grad(FP32)     | 4Ψ      | 4Ψ/N_d       |
    | 总计            | 16Ψ+KΨ  | 2Ψ+2Ψ+(4+4+4+4)Ψ/N_d+KΨ/N_d |
    |                |         | = 4Ψ + 16Ψ/N_d + KΨ/N_d      |

  → → → 精确公式: Memory(ZeRO-1) = 4Ψ + (12+K)Ψ/N_d
  → → → → N_d=1: 16Ψ+KΨ → 同DDP → 无优化!
  → → → → N_d=64: 4Ψ + (12+K)Ψ/64 ≈ 4Ψ → 参数+梯度仍占4Ψ → 大部分!
  → → → → → 7B 8GPU: 14+14+(84/8)=14+14+10.5=38.5GB → 24GB不够!

通信量:
  → ZeRO-1: 梯度all-reduce → 通信量 = 2Ψ (与DDP完全相同!)
  → → 原因: optimizer分区不影响梯度聚合 → all-reduce不变!
  → → → 优势: 通信零额外开销 → optimizer内存省N倍 → 简单!

### 1.3 ZeRO-2 (P_os+g): Gradient Partitioning

内存公式推导:
  → ZeRO-2在ZeRO-1基础上分区梯度 → 每rank只存1/N_d梯度!
  → → 关键: 梯度分区后 → 只保留自己分区的梯度 → 其余释放!
  → → → ZeRO-2 per-rank内存:
    | 状态           | 标准DDP  | ZeRO-2       |
    | 参数(FP16)     | 2Ψ      | 2Ψ (不变)    |
    | 梯度(FP16)     | 2Ψ      | 2Ψ/N_d       |
    | Master参数(FP32)| 4Ψ     | 4Ψ/N_d       |
    | Momentum(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Variance(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Grad(FP32)     | 4Ψ      | 4Ψ/N_d       |
    | 总计            | 16Ψ+KΨ  | 2Ψ+(2+4+4+4+4)Ψ/N_d+KΨ/N_d |
    |                |         | = 2Ψ + (14+K)Ψ/N_d            |

  → → → 精确公式: Memory(ZeRO-2) = 2Ψ + (14+K)Ψ/N_d
  → → → → N_d=1: 16Ψ+KΨ → 同DDP
  → → → → N_d=64: 2Ψ + (14+K)Ψ/64 ≈ 2Ψ → 参数仍占2Ψ → 主要开销!
  → → → → → 7B 8GPU: 14+((14+84)/8)=14+12.25=26.25GB → 24GB不够!

通信量推导 (关键!):
  → ZeRO-2: reduce-scatter替代all-reduce → 通信量 = Ψ (理论!)
  → → 为什么? reduce-scatter = all-reduce的一半! → 但仔细推导:
  → → → all-reduce = reduce-scatter + all-gather (数学等价!)
  → → → → all-reduce通信量 = 2×(N_d-1)×Ψ/N_d ≈ 2Ψ for large N_d
  → → → → reduce-scatter通信量 = (N_d-1)×Ψ/N_d ≈ Ψ for large N_d
  → → → → → ZeRO-2: 只reduce-scatter → 通信量=Ψ → 省一半!
  → → → → → → 但! ZeRO-2不需要all-gather梯度 → 因为每rank只保留自己的分区!
  → → → → → → → optimizer step用本rank梯度分区 → 不需要完整梯度!

  → → → → → → → 实际: reduce-scatter ≈ all-reduce的一半 → 但结果相同!
  → → → → → → → → 因为optimizer只在本地分区step → 不需要全局梯度!

### 1.4 ZeRO-3 (P_os+g+p): Parameter Partitioning

内存公式推导:
  → ZeRO-3分区参数 → 每rank只存1/N_d参数!
  → → 参数按需聚合(all-gather) → 用完释放!
  → → → ZeRO-3 per-rank内存:
    | 状态           | 标准DDP  | ZeRO-3       |
    | 参数(FP16)     | 2Ψ      | 2Ψ/N_d       |
    | 梯度(FP16)     | 2Ψ      | 2Ψ/N_d       |
    | Master参数(FP32)| 4Ψ     | 4Ψ/N_d       |
    | Momentum(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Variance(FP32) | 4Ψ      | 4Ψ/N_d       |
    | Grad(FP32)     | 4Ψ      | 4Ψ/N_d       |
    | 总计            | 16Ψ+KΨ  | (16+K)Ψ/N_d                     |

  → → → 精确公式: Memory(ZeRO-3) = (16+K)Ψ/N_d → 完美线性缩放!
  → → → → N_d=1: 16Ψ+KΨ → 同DDP → 无优化(单GPU!)
  → → → → N_d=64: (16+K)Ψ/64 ≈ Ψ/4 → 线性减!
  → → → → → 7B 8GPU: 112/8=14GB → fits 24GB RTX 4090! → 但有通信开销!

通信量推导 (关键!):
  → ZeRO-3: 每层需要all-gather参数 + reduce-scatter梯度!
  → → Forward: all-gather参数 → 通信量 = Ψ (每层)
  → → Backward: reduce-scatter梯度 + all-gather参数(用于grad计算) → 通信量 = Ψ
  → → → 总通信量/层 = 2Ψ → L层 = 2LΨ → 但! 按层数×每层参数量!
  → → → → 实际: 所有层参数=Ψ → 每层参数=Ψ/L → 每层通信=2×Ψ/L
  → → → → → 总通信量 = 2×L×(Ψ/L) = 2Ψ → 与标准DP相同!
  → → → → → → 关键洞察: ZeRO-3总通信量 = all-reduce = 2Ψ → 不增加!

  → → → → → → → 但! 实际实现: 每层单独all-gather → L次all-gather → vs 1次all-reduce!
  → → → → → → → → 小all-gather比大all-reduce延迟更低 → 但次数更多 → 总延迟可能更高!
  → → → → → → → → → 实际测量: ZeRO-3通信开销 ≈ 1.5x DDP → 不是2x → 因为all-gather更小更快!

### 1.5 内存公式总结表

```
| 状态类型        | DDP      | ZeRO-1    | ZeRO-2    | ZeRO-3    |
| 参数(FP16)      | 2Ψ       | 2Ψ        | 2Ψ        | 2Ψ/N_d    |
| 梯度(FP16)      | 2Ψ       | 2Ψ        | 2Ψ/N_d    | 2Ψ/N_d    |
| Optimizer(FP32) | 12Ψ+KΨ   | (12+K)Ψ/N_d| (12+K)Ψ/N_d| (12+K)Ψ/N_d|
| 总计            | 16Ψ+KΨ   | 4Ψ+(12+K)Ψ/N_d| 2Ψ+(14+K)Ψ/N_d| (16+K)Ψ/N_d|
| 通信量          | 2Ψ       | 2Ψ        | Ψ         | 2Ψ        |
| 通信方式        | all-reduce| all-reduce| reduce-scatter| all-gather+reduce-scatter|
| N_d→∞极限      | 16Ψ+KΨ   | 4Ψ        | 2Ψ        | 0(理论)   |

7B BF16实例 (N_d=8):
| 状态类型        | DDP      | ZeRO-1    | ZeRO-2    | ZeRO-3    |
| 总内存/rank     | 112GB    | 38.5GB    | 26.25GB   | 14GB      |
| fits RTX 4090?  | NO       | NO        | NO        | YES(理论) |
| 通信量          | 14GB×2   | 14GB×2    | 14GB      | 14GB×2    |
| 实际可行?       | PCIe灾难  | PCIe灾难   | PCIe灾难   | PCIe×L层更灾难!|

关键: RTX 4090 PCIe带宽瓶颈 → 任何跨GPU通信都灾难 → 单GPU最优!
```

### 1.6 ZeRO-Infinity扩展

ZeRO-Infinity (CPU/NVMe offload):
  → 在ZeRO-3基础上 → optimizer/gradient/parameter → offload到CPU/NVMe!
  → → GPU内存 ≈ activation only ≈ 2-3GB → 极小!
  → → → CPU内存: optimizer+gradient+parameter → 112GB → 需要64GB+CPU!
  → → → → NVMe: 超大模型 → 1TB+ → NVMe存 → swap到CPU → swap到GPU → 分层!
  → → → → → 通信: CPU↔GPU PCIe ≈12GB/s → NVMe↔CPU ≈2GB/s → 极慢!

内存公式:
  → GPU内存 ≈ Activation_memory ≈ 2-3GB (7B)
  → CPU内存 ≈ (16+K)Ψ/N_d → 分区后CPU存 → 每"GPU"rank的CPU存1/N!
  → → NVMe内存 ≈ rest → 超过max_in_cpu的部分 → swap!
  → → → 有效内存/GPU ≈ Activation ≈ 2-3GB → 几乎任意大模型都可训练!
  → → → → 但吞吐极低 → NVMe带宽2GB/s → vs NVLink 300GB/s → 150x差距!

```

## 2. 跨框架ZeRO实现对比 — DeepSpeed vs FSDP1 vs FSDP2 vs Megatron-LM

```
### 2.1 DeepSpeed ZeRO-3 — Init Metaclass注入

核心设计哲学: "参数出生就分区" → 不让完整参数存在于任何rank!

实现架构:
  → Init类 → InsertPostInitMethodToModuleSubClasses → 替换Module.__init__!
  → → partition_after包装 → 子类__init__完成后 → _post_init_method → partition()
  → → → 全局影响! → 所有Module子类__init__都被替换 → 深度侵入!
  → → → → Tensor构造函数也替换 → torch.tensor/empty/zeros → 自动FP16!

  → ZeROOrderedDict → 替换module._parameters → __getitem__拦截!
  → → forward时: param.ds_status==NOT_AVAILABLE → all_gather → 透明!
  → → → 用户不需要修改代码 → 参数自动聚合 → 透明!

  → PartitionedParameterCoordinator → trace+prefetch → 通信计算重叠!
  → → 第一次: RECORD → 记录参数使用顺序
  → → 后续: COMPLETE → prefetch → overlap!
  → → → 不匹配: INVALID → 重新trace → 自适应!

  → AllGatherCoalescedHandle → 批量all-gather → 单次通信 → 效率!
  → → 多参数合并 → 1次all-gather → vs N次 → 延迟省N倍!

关键特性:
  → 侵入式: 替换Module.__init__ + Tensor构造 → 深度修改Python运行时!
  → → 不兼容torch.compile → compile需要标准Module+Tensor → ZeRO修改了它们!
  → → → 不兼容FSDP → ZeROOrderedDict vs FlatParameter → 不同抽象层!
  → → → → 不兼容quantization → ZeRO替换了Tensor → quantization期望标准Tensor!

  → 优化激进: prefetch+trace+coalesced → 最高性能 → 但最复杂!
  → → ZeRO-Infinity: CPU/NVMe offload → 无竞争! → 模型超GPU内存的唯一方案!

### 2.2 PyTorch FSDP1 — FlatParameter架构

核心设计哲学: "Flatten→Shard→Unshard→Reshard" → 显式生命周期管理!

实现架构:
  → FullyShardedDataParallel → wrapper → 模型先完整初始化 → 然后flatten+shard!
  → → FlatParameter → 模块所有参数拼接成1D flat buffer → 单次all-gather!
  → → → 每FlatParameter = 一个FSDP unit → 一个模块的所有参数 → 一次聚合!
  → → → → shard: flat buffer按rank分区 → 每rank存1/N!
  → → → → → unshard: all-gather → 重建完整flat buffer → 计算!
  → → → → → → reshard: 释放 → 恢复分区状态 → 内存回收!

  → _FlatParamHandle → 状态机 → SHARD→UNSHARD→... → 生命周期管理!
  → → register_forward_hook / register_forward_pre_hook → hook驱动!
  → → → hook与activation checkpointing冲突 → 复杂bug!

关键特性:
  → 单次all-gather → 效率 → 但丧失per-parameter灵活性!
  → → 所有参数同一dtype → 不能per-param混合精度!
  → → → Flat buffer padding → 参数不能自然分 → 需padding → 浪费内存!
  → → → → Debug困难 → flat buffer看不出哪个param出错 → obscured!

  → 与torch.compile冲突 → hook驱动 → compile期望纯forward → hook破坏!
  → → → PyTorch官方已弃用FSDP1 → 推荐FSDP2!

### 2.3 PyTorch FSDP2 — Per-Parameter DTensor架构

核心设计哲学: "参数先完整存在 → 然后显式shard → 按需unshard/reshard" → PyTorch native!

实现架构:
  → fully_shard() API → 后初始化shard → 模型先完整 → 然后shard!
  → → per-parameter sharding → 每个参数独立分区 → 不flatten!
  → → → DTensor (Distributed Tensor) → 核心抽象 → Shard placement!
  → → → → param.data = shard_data → in-place替换 → 保留param identity!

  → FSDPParamGroup → 管理参数生命周期 → shard/unshard/reshard!
  → → pre_forward_hook → UnshardHandle → all-gather参数 → 开始计算!
  → → post_forward_hook → ReshardHandle → 释放参数 → 内存回收!
  → → → backward同理 → pre_backward_hook → unshard → post_backward → reshard!
  → → → → reduce-scatter梯度 → 每rank保留1/N → 与ZeRO-2/3相同!

  → prefetch → 比ZeRO-3简单 → 无trace → 但有basic prefetch策略!
  → → 下一层参数提前all-gather → overlap部分 → 但不如ZeRO-3 trace精确!

关键特性:
  → PyTorch native → torch.compile兼容 → DTensor兼容 → TP兼容!
  → → → per-parameter → 精细控制 → 可以per-param mixed precision!
  → → → → 无padding浪费 → 每参数自然分区 → 不flatten → 内存更省!
  → → → → → Debug简单 → 每参数独立 → 出错可定位 → 清晰!

  → 无ZeRO-Infinity → 无CPU/NVMe offload → 模型必须fits GPU+CPU总量!
  → → → 简单prefetch → 无trace → 性能不如ZeRO-3极致prefetch!
  → → → → 正在开发CPU offload → 2026可能支持 → closing gap!

### 2.4 Megatron-LM — TP+PP+DP (不直接实现ZeRO)

核心设计哲学: "层内切分(TP)+层间切分(PP)+数据复制(DP)" → 3D并行!

实现架构:
  → TP (Tensor Parallel): ColumnParallelLinear + RowParallelLinear → 每层切分!
  → → 每层1次AllReduce → 通信量Ψ/TP_size → 但每层都通信!
  → → → NVLink带宽300GB/s → TP=8 → AllReduce ≈0.08ms → 极快!

  → PP (Pipeline Parallel): GPipe/1F1B/Interleaved → 层间切分!
  → → P2P通信 → 发送activation/gradient → 逐stage → 带宽要求低!
  → → → Pipeline bubble → 1F1B最小bubble → 但仍有~1/PP浪费!

  → DP (Data Parallel): 标准DDP → 数据复制 → AllReduce!
  → → Megatron DP = standard → 不分区 → 完整optimizer+gradient+parameter!

  → → → DeepSpeed集成: Megatron 3D + ZeRO → ZeRO作用于DP维度 → 分区DP冗余!
  → → → → TP=8+PP=2+DP=64 → ZeRO-3 → 每GPU内存 = Ψ/(TP×PP×DP)!

内存公式(Megatron + ZeRO-3):
  → Per-GPU参数: 2Ψ/(TP×PP×DP)
  → Per-GPU梯度: 2Ψ/(TP×PP×DP)
  → Per-GPU optimizer: (12+K)Ψ/(TP×PP×DP)
  → → 总计: (16+K)Ψ/(TP×PP×DP) → 完美缩放!
  → → → 7B 8GPU(TP=2,PP=1,DP=4): 112/(2×1×4)=14GB → fits 24GB!
  → → → → 100B 512GPU(TP=8,PP=8,DP=8): 1600/(8×8×8)=3.12GB/GPU → fits 80GB A100!

### 2.5 跨框架对比总结表

```
| 特性            | DeepSpeed ZeRO-3 | FSDP1            | FSDP2          | Megatron 3D    |
| 初始化方式       | Init metaclass   | FSDP wrapper     | fully_shard()  | Megatron init  |
| 分区粒度         | per-parameter    | FlatParameter    | per-parameter  | per-layer(TP)  |
| 聚合方式         | ZeROOrderedDict  | FlatParam unshard| DTensor unshard| AllReduce(TP)  |
| Prefetch        | trace+coalesced  | 无               | basic prefetch  | 无             |
| CPU/NVMe offload| ZeRO-Infinity    | 无               | 无(开发中)     | 无             |
| torch.compile   | 不兼容           | 不兼容           | 兼容!          | 不兼容         |
| TP兼容          | 部分             | 部分             | DTensor TP     | 完美           |
| 混合精度per-param| 困难             | 困难             | 自然!          | 困难           |
| Debug难度       | 中(全局修改)     | 高(flat obscured)| 低(per-param) | 中             |
| 2026趋势        | offload场景      | 弃用→FSDP2       | 主流方向       | 大模型标准     |
| RTX 4090推荐    | ZeRO-2+offload   | 不推荐           | 不推荐(单GPU)  | 不推荐(PCIe)   |
```

### 2.6 哲学差异

```
DeepSpeed ZeRO-3: "参数出生就分区" → 全局侵入 → 深度优化 → 但不兼容!
  → 像手术 → 修改Python运行时 → 所有Module/__init__/Tensor → 深度!
  → → 优势: 最早分区 → 内存最小 → 不浪费任何空间!
  → → → 劣势: 不兼容 → torch.compile/quantization/FSDP → 侵入式!

FSDP1: "Flatten再Shard" → flat buffer → 单次通信 → 简单!
  → 像打包 → 所有参数打包成flat → 整体分区 → 整体聚合!
  → → 优势: 单次all-gather → 通信最少 → 简单!
  → → → 劣势: per-param灵活性差 → padding浪费 → hook冲突 → 弃用!

FSDP2: "参数先完整再Shard" → PyTorch native → DTensor → 兼容!
  → 像整理 → 每参数独立 → 自然分区 → 自然聚合 → PyTorch原生!
  → → 优势: torch.compile/TP/quantization兼容 → per-param灵活性!
  → → → 劣势: 无ZeRO-Infinity(offload) → 简单prefetch → 性能不如ZeRO-3!

Megatron 3D: "TP+PP+DP" → NVLink → 高带宽 → 不分区DP → 标准!
  → 像分工 → 层内TP切 → 层间PP切 → DP复制 → NVLink支撑!
  → → 优势: NVLink下最快 → TP/PP无额外通信 → 极致性能!
  → → → 劣势: DP无分区 → 内存不减 → 需ZeRO补充 → 无offload!
```

## 3. ZeRO+TP+PP组合 — 3D并行策略选择

```
### 3.1 组合内存公式

令TP = Tensor Parallel度
令PP = Pipeline Parallel度
令DP = Data Parallel度 (ZeRO作用于DP维度)
令N = TP×PP×DP (总GPU数)

ZeRO-3 + TP + PP per-GPU内存:
  → 参数: 2Ψ/(TP×PP×DP)
  → 梯度: 2Ψ/(TP×PP×DP)
  → Optimizer: (12+K)Ψ/(TP×PP×DP)
  → Activation: A/(PP) → PP分担activation! → TP不分担activation!
  → → 总计: (16+K)Ψ/N + A/PP → DP分区+TP分区+PP分担activation!

ZeRO-2 + TP + PP per-GPU内存:
  → 参数: 2Ψ/(TP×PP) → TP+PP分区 → DP不分区参数!
  → 梯度: 2Ψ/(TP×PP×DP) → DP分区梯度!
  → Optimizer: (12+K)Ψ/(TP×PP×DP) → DP分区optimizer!
  → → 总计: 2Ψ/(TP×PP) + (14+K)Ψ/N + A/PP

### 3.2 通信量公式

TP通信 (每层):
  → AllReduce: 2×(Ψ/TP) per layer → L层 → 2×L×Ψ/TP
  → → 需要: NVLink → 300GB/s → 极快 → TP=8 → 每层≈0.08ms!

PP通信 (每stage):
  → P2P: activation + gradient → 小量 → 跨节点IB → 可接受!

ZeRO-3 DP通信 (每step):
  → All-gather参数: Ψ/DP per layer → L层 → L×Ψ/DP
  → Reduce-scatter梯度: Ψ/DP per layer → L层 → L×Ψ/DP
  → → 总: 2×L×Ψ/DP → 但! 参数已由TP+PP缩小 → 实际每层=Ψ/(TP×PP×DP)

ZeRO-2 DP通信:
  → Reduce-scatter梯度: Ψ/(TP×PP×DP) per layer → L层 → L×Ψ/(TP×PP×DP)

### 3.3 策略选择决策树

```
模型大小 → GPU类型 → GPU数量 → 最优策略

<7B, RTX 4090, 1 GPU:
  → LoRA + gradient checkpointing → 14.6GB fits → 单GPU最优!
  → → ZeRO无用 → 单GPU无DP → 不分区!
  → → → 如果全参数微调 → ZeRO-2 + CPU optimizer offload → 22.75GB→offload→16.35GB fits!

<7B, RTX 4090, 8 GPU:
  → 不推荐多GPU → PCIe灾难 → DDP 0.46x → ZeRO更慢!
  → → 结论: 单GPU最优 → 不要跨GPU通信!

7-13B, A100 80GB, 8 GPU:
  → TP=1, ZeRO-2 → 8×80=640GB → 13B BF16≈104GB → ZeRO-2 8GPU: 2×13+(14×13)/8≈35.5GB → fits!
  → → ZeRO-3: (16×13)/8=26GB → fits → 但通信开销大!
  → → → 推荐: ZeRO-2 → 通信不变 → 内存够 → 简单!

7-70B, A100 80GB, 64 GPU:
  → TP=8(intra-node) + PP=2 + ZeRO-2(DP=4):
  → → 70B BF16=140GB → TP=8 → 每GPU参数=140/8=17.5GB → PP=2 → 8.75GB
  → → → ZeRO-2 DP=4: optimizer+grad /4 → 8.75+8.75/4+140/32≈20GB → fits 80GB!
  → → → → 通信: TP AllReduce(NVLink) + PP P2P(IB) + ZeRO-2 reduce-scatter → OK!

>100B, H100 80GB, 512 GPU:
  → TP=8 + PP=8 + ZeRO-3(DP=8):
  → → 100B BF16=200GB → TP=8→25GB/PP→PP=8→3.125GB→DP=8→0.39GB → fits!
  → → → ZeRO-Infinity如果模型超GPU+CPU总量 → NVMe offload → 极慢但唯一!

决策规则:
  1. 有NVLink → TP优先(intra-node) → TP=8 → 最快AllReduce!
  2. 跨节点 → PP优先 → P2P小量 → IB带宽够!
  3. DP维度 → ZeRO-2优先 → 通信不变 → 内存减 → 够用!
  4. DP不够 → ZeRO-3 → 但通信增加 → 需高带宽!
  5. ZeRO-3不够 → ZeRO-Infinity → CPU/NVMe offload → 极慢但唯一!
  6. 无NVLink(RTX 4090) → TP不推荐 → PCIe太慢 → 单GPU最优!
```

## 4. RTX 4090 vs A100 vs H100 — ZeRO最优配置对比

```
| GPU          | NVLink | 带宽      | ZeRO-1       | ZeRO-2       | ZeRO-3       | 推荐策略         |
| RTX 4090     | 无     | PCIe      | 不推荐(38.5GB)| 不推荐(26GB) | 理论14GB但PCIe灾难| LoRA+单GPU       |
| A100 80GB    | 300GB/s| NVLink+IB | OK但不够     | 推荐!         | 可选           | TP=8+ZeRO-2      |
| H100 80GB    | 726GB/s| NVLink+IB | OK但不够     | 推荐!         | 推荐(高带宽)   | TP=8+ZeRO-3      |

RTX 4090关键结论:
  → PCIe带宽瓶颈 → 任何跨GPU通信灾难 → ZeRO-3更灾难(L次all-gather)!
  → → 单GPU ZeRO-2+CPU offload → optimizer在CPU → 14.6GB fits → 最优(全参数)!
  → → → LoRA+gradient checkpointing → 不需要ZeRO → 更简单更快!
  → → → → 结论: RTX 4090 = 单GPU最优 → ZeRO只在offload场景有用!

A100关键结论:
  → NVLink 300GB/s → TP AllReduce极快 → ZeRO-2+TP最优!
  → → 7B A100 8GPU: ZeRO-2 → 26GB fits 80GB → TP不需要 → 简单DDP+ZeRO-2!
  → → → 13B A100 8GPU: TP=2+ZeRO-2 → 7×80=560GB → fits → 推荐!

H100关键结论:
  → NVLink 726GB/s → 更快 → ZeRO-3可行 → 高带宽支撑all-gather开销!
  → → 70B H100 64GPU: TP=8+ZeRO-3(DP=8) → fits 80GB → 推荐!
  → → → ZeRO-Infinity: 模型超GPU+CPU → NVMe offload → 极慢 → 最后手段!
```

## 5. ZeRO与FSDP2未来趋势 (2026)

```
1. FSDP2成为主流:
  → PyTorch官方推荐 → torch.compile兼容 → DTensor TP兼容 → HuggingFace集成!
  → → DeepSpeed维护缓慢 → 新PyTorch版本兼容问题 → 社区转向FSDP2!
  → → → 但ZeRO-Infinity(offload)仍是独有优势 → 模型超GPU唯一方案!

2. 2D/3D并行标准化:
  → FSDP2 + DTensor TP → PyTorch原生2D并行 → 不需要DeepSpeed!
  → → FSDP2 + TP + PP → 3D并行 → DTensor统一 → 2026标准!

3. CPU offload差距:
  → DeepSpeed: ZeRO-Infinity → CPU/NVMe offload → 成熟!
  → FSDP2: 无offload → 2026开发中 → closing gap!
  → → 过渡期: 模型超GPU → 用DeepSpeed → 其他场景用FSDP2!

4. LoRA/PEFT改变游戏:
  → LoRA微调 → 不需要ZeRO → 单GPU即可 → 7B LoRA=14.6GB → fits 24GB!
  → → ZeRO价值下降 → 大模型全参数训练需要 → 但LoRA更常见!
  → → → RTX 4090: LoRA最优 → ZeRO只在全参数微调(offload)有用!
```

## 参考文献

```
1. ZeRO论文:
   - Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020 (arxiv.org/abs/1910.02054)
   - Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall", SC 2021

2. FSDP论文/文档:
   - PyTorch FSDP文档 — pytorch.org/docs/stable/fsdp.html
   - FSDP2 RFC — Per-Parameter Sharding Architecture
   - Wei et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"

3. Megatron-LM:
   - Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models", SC 2019
   - Korthikanti et al., "Efficient Large-Scale Language Model Training on GPU Clusters", 2022

4. 我们的笔记:
   - deepspeed-zero-source-reading.md → DeepSpeed源码结构
   - pytorch-torch-distributed-internals-deep-dive.md → ProcessGroupNCCL+FSDP2
   - megatron-source-reading.md → Megatron TP+PP
   - nccl-internals-architecture-deep-dive.md → NCCL通信原理
