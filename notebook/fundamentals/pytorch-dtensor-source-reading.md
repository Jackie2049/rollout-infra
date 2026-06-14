# PyTorch DTensor (Distributed Tensor) 源码级深度阅读

> 2026-06-15 | 源码: torch/distributed/tensor/ (api.py + placement_type.py + device_mesh.py + sharding_propagation.py)
> 核心: DTensor是PyTorch分布式训练的基础抽象 → DeviceMesh+Placement(Shard/Replicate/Partial) → SPMD模型 → FSDP2/TP/EP统一表达

## 1. DTensor核心概念

```
DTensor = 本地tensor + 分布式元信息
  ├── local_tensor: 本GPU上的实际数据片段
  ├── device_mesh: ND设备网格 → 定义物理拓扑
  ├── placements: 每个mesh维度上的Placement → 定义数据分布方式
  └── spec: DTensorSpec(mesh+placements) → 完整分布描述

三类Placement:
  Shard(dim): 沿指定维度分片 → 每个设备持有1/N
  Replicate(): 完整复制 → 每个设备持有全量
  _Partial(): 内部状态 → 部分计算/未完成AllReduce → 需reduce才能使用
```

### 基本API

```python
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor

# 1D DeviceMesh (数据并行)
mesh_1d = DeviceMesh("cuda", list(range(world_size)))
dt = distribute_tensor(tensor, mesh_1d, [Shard(0)])  # 沿dim=0分片
dt = distribute_tensor(tensor, mesh_1d, [Replicate()])  # 全量复制

# 2D DeviceMesh (TP+DP混合)
mesh_2d = DeviceMesh("cuda", torch.arange(world_size).reshape(tp_size, dp_size))
dt = distribute_tensor(tensor, mesh_2d, [Shard(0), Replicate()])
# → mesh_dim=0(TP): Shard(0)沿dim=0分片
# → mesh_dim=1(DP): Replicate()完整复制
# → 结果: 每个TP组持有1/tp_size的tensor, DP组间复制!

# 转换
local_tensor = dt.to_local()  # DTensor→本GPU的local tensor
dt = DTensor.from_local(local_tensor, mesh, placements)  # local→DTensor
```

## 2. DeviceMesh设计

```python
# torch/distributed/tensor/device_mesh.py

class DeviceMesh:
    """ND设备网格 → 每个维度对应一个并行组"""

    # 1D mesh: [0, 1, 2, 3, 4, 5, 6, 7] → 1个DP组(8设备)
    # 2D mesh: [[0,1], [2,3], [4,5], [6,7]] → TP组{0,1},{2,3},{4,5},{6,7} + DP组{0,2,4,6},{1,3,5,7}
    # 3D mesh: → TP×PP×DP → 完全灵活

    # 内部: torch.distributed.ProcessGroup per mesh_dim
    # mesh_dim_group(0) → TP ProcessGroupNCCL
    # mesh_dim_group(1) → DP ProcessGroupNCCL
    # → 每个维度独立通信 → 不互相阻塞!
```

### ND Mesh实例

```
8 GPU, TP=2, DP=4:
  mesh_2d = DeviceMesh("cuda", [[0,1],[2,3],[4,5],[6,7]])

  mesh_dim=0 → TP组:
    {0,1}, {2,3}, {4,5}, {6,7} → 4组×2GPU

  mesh_dim=1 → DP组:
    {0,2,4,6}, {1,3,5,7} → 2组×4GPU

  一个tensor的placement=[Shard(0), Replicate()]:
    → TP组内Shard(0): GPU0持有前半,GPU1持后半
    → DP组间Replicate: 每组都有完整副本(shard后)
    → GPU0实际持有: tensor[0:N/2], GPU2也有tensor[0:N/2]
```

## 3. Placement详解

### Shard(dim)

```python
class Shard(Placement):
    """沿指定维度分片"""

    # Shard(0): 沿dim=0(row维度)分片 → 每个设备持有1/N的行
    #   weight矩阵 [out_features, in_features]:
    #     GPU0: [0:out/tp_size, in_features]
    #     GPU1: [out/tp_size:2*out/tp_size, in_features]
    #   → ColumnParallelLinear! (输出沿列切分)

    # Shard(1): 沿dim=1(column维度)分片 → 每个设备持有1/N的列
    #   weight矩阵 [out_features, in_features]:
    #     GPU0: [out_features, 0:in/tp_size]
    #     GPU1: [out_features, in/tp_size:2*in/tp_size]
    #   → RowParallelLinear! (输入沿列切分,输出需AllReduce)
```

### Replicate()

```python
class Replicate(Placement):
    """完整复制 → 每个设备持有全量"""

    # 用于: bias向量(太小不值得分片) + DP组间复制
    # 在2D mesh [Shard(0), Replicate()]中:
    #   → TP组Shard, DP组Replicate → 每DP rank有完整shard副本
```

### _Partial() (内部)

```python
class _Partial(Placement):
    """部分计算状态 → 需reduce才能使用"""

    # 出现场景:
    # 1. RowParallelLinear输出: 每个TP rank有部分结果 → 需AllReduce→Replicate
    # 2. 梯度聚合前: 每个DP rank有部分梯度 → 需AllReduce→_Partial→Replicate/Shard
    # 3. 优化器step后: 每个shard有部分参数更新 → 需AllGather→完整

    # 自动redistribute: DTensor检测_Partial → 自动插入reduce操作!
```

## 4. Sharding Propagation (分片传播)

```
核心算法: 给定input placements → 推导output placements
→ SPMD模型: 用户只指定input的分片方式 → DTensor自动推导所有op的output分片!

规则示例:

1. Element-wise op (add, mul, relu):
   output_placement = input_placement → 传播不变

2. MatMul: Einsum-based推导
   A: [Shard(0), Replicate()] × B: [Replicate(), Shard(1)]
   → C: [Shard(0), Shard(1)] → TP行×TP列 → TP输出!

   A: [Shard(0), Replicate()] × B: [Replicate(), Replicate()]
   → C: [Shard(0), Replicate()] → ColumnParallel!

   A: [Replicate(), Replicate()] × B: [Replicate(), Shard(1)]
   → C: [Replicate(), Shard(1)] → RowParallel!(需AllReduce→Replicate)

3. Reduction op (sum, mean):
   沿sharded维度reduce → Replicate()
   沿replicated维度reduce → Shard(dim-1) or _Partial()

4. Softmax/CrossEntropy:
   softmax(dim=-1): input[Shard(1)] → 需AllGather→Replicate→softmax→Shard(1)
   → 自动redistribute!

5. Embedding:
   [Replicate()] → 输出[Replicate()] (vocab完整在每个GPU)
   或 [Shard(0)](vocab分片) → 输出[Shard(0)](但需AllReduce for DP)
```

### Custom Propagation Rules

```python
from torch.distributed.tensor import register_sharding_propagation_rule

@register_sharding_propagation_rule(my_custom_op)
def my_sharding_rule(op, input_placements):
    # 自定义推导规则
    if input_placements[0] == Shard(0):
        return [Shard(0)]  # output也Shard(0)
    else:
        return [Replicate()]  # 其他情况Replicate
```

## 5. Redistribution (Resharding)

```
当op的输入分片不满足要求 → DTensor自动插入redistribute操作!

例: softmax需要完整序列 → input是Shard(1)(序列分片)
  → redistribute: Shard(1) → AllGather → Replicate → softmax → Shard(1)

redistribute实现:
  DTensor.redistribute(target_placements):
    分析source→target的变化:
      Shard→Replicate: AllGather
      Replicate→Shard: 新分片(本地slice)
      Shard(0)→Shard(1): reshard(沿不同维度分片)
      _Partial→Replicate: AllReduce
      _Partial→Shard: ReduceScatter

    自动插入通信操作 → autograd兼容!
    → backward时redistribute的反向操作自动插入:
      AllGather backward → ReduceScatter
      ReduceScatter backward → AllGather
      AllReduce backward → 无(梯度自然分布)
```

## 6. DTensor Autograd

```
关键设计: DTensor的backward是forward propagation的镜像!

Forward:
  input_DTensor[Shard(0)] → op → output_DTensor[Shard(1)]
  → redistribute: Shard(0)→AllGather→Replicate→op→Shard(1)

Backward (梯度传播方向=反向):
  grad_output_DTensor[Shard(1)] → op.backward → grad_input_DTensor
  → 需要redistribute grad_output: Shard(1)→ReduceScatter→Shard(0)
  → 或根据数学推导直接计算!

关键规则:
  Forward AllGather → Backward ReduceScatter (聚合→散开→散开→聚合)
  Forward ReduceScatter → Backward AllGather (散开→聚合→聚合→散开)
  Forward AllReduce → Backward: 无额外通信(梯度自然已分布)

  → 与Megatron SP的数学等价!
  _GatherFromSequenceParallelRegion(fwd=AllGather, bwd=ReduceScatter)
  _ReduceScatterToSequenceParallelRegion(fwd=ReduceScatter, bwd=AllGather)
```

## 7. FSDP2如何使用DTensor

```python
# FSDP2: 每个参数变成DTensor with Shard(0)

model = MyModel()
# FSDP2 wrap:
for module in model.modules():
    for param in module.parameters():
        # Before: param.data = local_tensor (full)
        # After: param.data = DTensor.from_local(
        #     local_shard,  # 本GPU的1/N分片
        #     mesh,         # DeviceMesh
        #     [Shard(0)],   # 沿dim=0分片
        # )
        # → param现在是DTensor!

# Forward:
#   FSDP2 module hook: DTensor.redistribute([Replicate()])
#   → AllGather → 完整参数 → compute forward
#   → 保留完整参数用于backward!

# Backward:
#   gradient是DTensor with _Partial() → 需reduce
#   → DTensor.redistribute([Shard(0)])
#   → ReduceScatter → 每个GPU只持自己shard的梯度

# Optimizer:
#   param.data = DTensor with Shard(0) → 本GPU只更新1/N
#   → optimizer只处理local shard → 省12Ψ/N内存!
```

### FSDP2 vs FSDP1 DTensor差异

```
FSDP1 (FlatParameter):
  - 所有参数打包成1个FlatParameter → 固定大小分片
  - padding waste: 小参数被padding到固定大小
  - compile不兼容: FlatParameter动态gather→graph break
  - backward需AllGather(FlatParameter gather→backward→ungather)

FSDP2 (DTensor per-parameter):
  - 每个参数独立DTensor → 精确分片(无padding!)
  - compile兼容: DTensor静态shape→compile看到固定shapes
  - backward不需要AllGather(保留完整参数→backward后ReduceScatter)
  - 通信: 2Ψ(AllGather fwd + ReduceScatter bwd) vs FSDP1 3Ψ!
```

## 8. TP如何使用DTensor

```python
# Tensor Parallelism: DTensor表达Column/Row Parallel

# ColumnParallelLinear: weight Shard(0)
mesh = DeviceMesh("cuda", list(range(tp_size)))
linear = nn.Linear(hidden, hidden)
linear.weight = distribute_tensor(linear.weight, mesh, [Shard(0)])
# → 每个TP rank持有 [hidden/tp_size, hidden] 的weight
# → 输出沿output维度分片 → AllReduce或直接传下一个Shard(0) op

# RowParallelLinear: weight Shard(1)
linear.weight = distribute_tensor(linear.weight, mesh, [Shard(1)])
# → 每个TP rank持有 [hidden, hidden/tp_size] 的weight
# → 输出沿input维度分片 → 需AllReduce→Replicate

# DTensor自动处理通信:
#   Shard(0)→Shard(0) matmul → 无额外通信(连续ColumnParallel)
#   Shard(0)→Shard(1) → 需AllReduce→Replicate→新Shard(1)
#   → DTensor.redistribute自动插入!
```

## 9. Compositional Parallelism (组合并行)

```
2D DeviceMesh → TP+DP组合:

mesh_2d = DeviceMesh("cuda", [[0,1],[2,3],[4,5],[6,7]])  # TP=2, DP=4

weight placements = [Shard(0), Replicate()]
  → mesh_dim=0(TP): Shard(0) → TP组内分片
  → mesh_dim=1(DP): Replicate → DP组间复制

forward:
  TP组: AllGather→完整weight→compute→ReduceScatter→1/tp_size weight
  DP组: 无通信(Replicate→计算独立)

backward:
  TP组: ReduceScatter(grads) → 本TP rank的grad shard
  DP组: AllReduce(grads) → 跨DP组梯度聚合

→ 自动表达TP+DP → 无需手动管理ProcessGroup!

3D DeviceMesh → TP+PP+DP:
  mesh_3d = DeviceMesh("cuda", reshape(world_size, [tp, pp, dp]))
  placements = [Shard(0), Replicate(), Replicate()]
  → TP Shard(0) + PP Replicate(各stage独立) + DP Replicate(DP组间复制)
  → backward: TP ReduceScatter + DP AllReduce + PP P2P
```

## 10. DTensor vs ZeRO-3 ds_tensor对比

| 维度 | DTensor | ZeRO-3 ds_tensor |
|------|---------|------------------|
| **抽象层** | PyTorch核心(2.6+稳定) | DeepSpeed专用 |
| **分片粒度** | per-parameter | sub-partition(多参数打包) |
| **Placement表达** | [Shard(dim), Replicate()] → 声明式 | ds_status枚举 → 状态机 |
| **通信插入** | 自动redistribute → autograd兼容 | PartitionedParameterCoordinator手动trace |
| **Padding浪费** | 无(DTensor精确) | 有(NCCL对齐+分区对齐) |
| **compile兼容** | ✓(静态shape) | ✗(动态AllGather→graph break) |
| **ND Mesh** | ✓(任意维度组合TP+DP+PP) | ✗(只有1D DP) |
| **Autograd集成** | ✓(backward是forward镜像) | ✗(手动backward hooks) |
| **生态整合** | FSDP2+TP+EP统一 | DeepSpeed专用 |

## 11. RTX 4090影响

```
RTX 4090 24GB (7B模型):
  DTensor Shard(0): 每参数1/N → 8GPU: 每GPU14/8=1.75GB
  → 但AllGather需完整参数=14GB → peak=1.75+14+7+1.75+1.4=32.9GB ✗
  → 单GPU24GB不够!

  DTensor不适用RTX 4090单GPU:
    DeviceMesh size=1 → Shard=N/A → DTensor退化为普通tensor
    → 单GPU: DTensor无任何分布式功能!

  RTX 4090 PCIe 8GPU:
    DTensor FSDP2: 28GB/step通信 → 1.17s → 计算0.5s → 通信>60%
    → PCIe灾难 → DTensor/FSDP2不适合RTX 4090!

  RTX 4090最优:
    单GPU+LoRA+CPU_Adam → 无DTensor需求 → 无分布式通信!
    → rLLM Tinker or verl GRPO+LoRA → 0通信开销

  NVLink集群:
    DTensor+FSDP2+compile → 最优路径!
    → 2Ψ通信+2.5x compile加速 → 2-3x训练速度
    → DTensor+TP → Megatron-level性能但代码更简单
```

## 12. DTensor关键设计洞察

```
1. 声明式分片 → 用户只说"Shard(0)", DTensor自动处理通信
   → 不需要手动AllGather/ReduceScatter → SPMD编程模型!

2. Autograd兼容 → backward是forward镜像 → 自动梯度分片
   → 不需要手动backward hooks → 简单!

3. ND DeviceMesh → 任意组合TP+DP+PP → 统一表达
   → 不需要15+复合ProcessGroup → 简洁!

4. Sharding Propagation → 输入分片→自动推导所有op输出分片
   → 不需要手动指定每个op的通信 → 自动!

5. compile兼容 → DTensor静态shape → torch.compile看到固定sizes
   → 不需要担心graph break → FSDP2+compile=最佳组合!

6. 未来方向: DTensor成为PyTorch分布式训练的唯一抽象
   → FSDP2/TP/EP都基于DTensor → 统一API
   → ZeRO-3的ds_tensor → 将被DTensor替代(长期)
   → Megatron的ProcessGroup → 也将迁移到DTensor(ProcessGroupCollection已经在迁移!)
```

---

Sources:
- [PyTorch DTensor Official Documentation](https://pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch Blog: Scaling with DTensor](https://pytorch.org/blog/scaling-pytorch-with-dtensor/)
- [PyTorch DTensor Tutorial](https://pytorch.org/tutorials/advanced/dtensor.html)
- [GitHub: pytorch/torch/distributed/tensor](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor)
- notebook/projects/zero3-vs-fsdp2-system-comparison.md
- notebook/fundamentals/megatron-parallel-state-source-reading.md
