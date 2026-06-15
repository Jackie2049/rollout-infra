# PyTorch FSDP2 源码级深度阅读 (2026-06-15更新版)

> 2026-06-15 | 源码: torch/distributed/_composable/fsdp/ + torch/distributed/fsdp2/ + torchtitan
> ★★★ FSDP2 = PyTorch原生分布式训练未来 → ZeRO-3淘汰 → DTensor per-parameter → torch.compile兼容
> ★★★ RTX 4090: FSDP2无用(单GPU) → 但理解至关重要(多GPU未来) → BF16唯一正确精度

## 1. FSDP2 vs FSDP1 架构革命

```
★★★★★ 3个根本问题驱动FSDP1→FSDP2重设计:

Problem 1: FlatParameter → 参数身份被摧毁
  → FSDP1: flatten所有参数→单个FlatParameter buffer → 名字/形状丢失 → debug不透明
  → FSDP2: 每个nn.Parameter独立DTensor.from_local(Shard(0)) → 保留名字+形状!

Problem 2: Class Wrapper → 灵活性差
  → FSDP1: FullyShardedDataParallel(module) → 单一类封装 → 无法与TP/AC组合
  → FSDP2: fully_shard(module) → functional API → hooks → 可组合TP+AC!

Problem 3: torch.compile不兼容
  → FSDP1: Python hooks → graph breaks → flatten/unflatten → dynamic shape → compile失败
  → FSDP2: compiler-friendly hooks → torch.compiler.disable on collectives → 减少graph break!

★★★★ 迁移映射:
  FSDP(model) → fully_shard(model)
  MixedPrecision → MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype)
  summon_full_params() → unshard() context manager
  FSDP1 flat state dicts → 标准per-parameter state dicts
```

## 2. 3种分片策略

```
★★★★★ FSDP2 per-parameter DTensor → 模型内混合策略!

| Strategy | Params | Grads | Optim | Memory/GPU | Communication |
|----------|--------|-------|-------|------------|---------------|
| ★★★ FULL_SHARD | Shard(DTensor) | Shard(RS) | Shard | 最低(1/N) | AllGather+RS per layer |
| ★★ SHARD_GRAD_OP | Replicated | Shard(RS) | Shard | 中(全params+shard grad/opt) | RS only(no AG) |
| ★ NO_SHARD | Replicated | Replicated | Replicated | 最高(DDP) | AllReduce(DDP) |

★★★ FULL_SHARD = ZeRO-3 → 最省内存 → 但通信多(2×AG per param)
★★★ SHARD_GRAD_OP ≈ ZeRO-2 → 省3x内存 → 通信少(1×RS) → 宯用最优!
★★★ ★★★ 模型内混合: 大层FULL_SHARD + 小层NO_SHARD → FSDP1做不到!
```

## 3. Mixed Precision: MixedPrecisionPolicy

```python
★★★★★ 三轴精度控制:

from torch.distributed.fsdp2 import MixedPrecisionPolicy, Bfloat16MixedPrecisionPolicy

# 精细控制:
policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # 分片参数存储精度 → 灁持久内存
    reduce_dtype=torch.float32,     # 梯度通信精度 → 数值稳定性
    output_dtype=torch.bfloat16,   # forward计算输出精度 → 计算精度
)

# 便利预设: BF16 params + FP32 reduction + BF16 output → ★★★ 推荐!
policy = Bfloat16MixedPrecisionPolicy()

★★★★ RTX 4090:
  → BF16: SM89 tensor core支持 → ★★★ 唯一正确训练精度
  → FP8: SM89 tensor core存在但无native GEMM pipeline → 无性能优势 → ✗ 不用!
  → FP8需要SM90(Hopper/H100) → torch._scaled_mm需要SM90 → SM89 fallback无加速
  → ★★★ 结论: RTX 4090 FSDP2训练 → Bfloat16MixedPrecisionPolicy → BF16 only!
```

## 4. Resharding生命周期 (4 hook phases)

```
★★★★★ 每module的reshard lifecycle:

| Phase | Hook | Action | Memory |
|-------|------|--------|--------|
| Forward pre | register_forward_pre_hook | AllGather shard→full params | Peak: 1 module full params |
| Forward post | register_forward_hook | Free full params→restore shard | Drop back to shard size |
| Backward pre | grad_fn.register_hook() | AllGather again for bwd | Peak again |
| Backward post | grad_fn.register_hook() | ReduceScatter grads; free full | Drop; shard grad stored |

★★★★ reshard_after_forward选项:
  → RESHARD_AFTER_FORWARD(default): 参数forward后释放→backward重新AG → 2×AG但最低peak mem
  → 不reshard: 参数保留→backward不需要AG → 1×AG但peak mem高 → 所有forward层参数同时存活!

★★★★★ FSDP2 vs ZeRO-3通信量:
  → ZeRO-3: 3Ψ per step (AG fwd + AG bwd + RS grads)
  → FSDP2 FULL_SHARD: 2Ψ per step (AG fwd + RS grads → bwd AG included in fwd)
  → FSDP2 SHARD_GRAD_OP: 1Ψ per step (RS grads only)
  → ★★★ FSDP2比ZeRO-3少33%通信!
```

## 5. 2D DeviceMesh: HSDP (Hybrid Sharded DP)

```python
★★★★★ HSDP = 2D mesh → FSDP + Replication → inter-node FSDP + intra-node replicate:

from torch.distributed.device_mesh import DeviceMesh

# 2D mesh: 2 replica groups × 4 shard groups
mesh_2d = DeviceMesh("cuda", (2, 4), mesh_dim_names=('replicate', 'shard'))

# FSDP uses 'shard' dimension → inter-node sharding
# 'replicate' dimension → intra-node replication → gradient sync within group

model = FSDPModule(model, mesh=mesh_2d)

★★★★ HSDP优势:
  → intra-node: NVLink → fast gradient sync → replicate → 无参数AG
  → inter-node: network → slow → shard → 参数AG needed
  → → 比纯FULL_SHARD: intra-node更快; 比DDP: inter-node更省内存!

★★★★ 3D mesh (FSDP + TP):
  mesh = DeviceMesh("cuda", (1, 2, 4), mesh_dim_names=('replicate', 'tp', 'shard'))
  → FSDP(shard) + TP(tp) + replicate → torchtitan uses this!
```

## 6. Prefetching: 双stream通信-计算重叠

```
★★★★★ FSDP2 prefetch = 通信-计算overlap → 双CUDA stream:

核心机制:
  → 默认stream计算layer i → NCCL stream发起layer i+1的AllGather
  → CUDA event同步 → 确保prefetched参数在计算前ready
  → ★★★ 类似FlashAttention的overlap → 但在参数通信层面!

Forward prefetch: 计算layer i时，prefetch layer i+1 → 正序
Backward prefetch: 计算bwd layer i时，prefetch layer i-1 → 逆序

★★★ prefetch_policy配置:
  → None: 顺序(无overlap) → 最简单 → 最低吞吐
  → Backward-only: 只backward prefetch → forward轻量时有用
  → Always: forward+backward都prefetch → 最大overlap → 最高吞吐

★★★★ Reduce-scatter overlap:
  → backward RS(gradient sharding)与backward计算重叠 → 进一步减少通信延迟!

★★★★ 执行顺序感知:
  → FSDP2预注册模块执行顺序 → 知道prefetch哪个模块
  → Transformer层 → 自然顺序 → FSDP2自动处理 → 无需手动!

★★★★ vs ZeRO-3 prefetch:
  → ZeRO-3: PartitionedParameterCoordinator → 多参数合并1次AllGather
  → FSDP2: per-parameter → 小参数独立通信 → NCCL channel利用更有效!
```

## 7. 2D DeviceMesh: HSDP

```python
★★★★★ composable API → AC+FSDP完美组合:

from torch.distributed.algorithms._checkpoint import activation_checkpoint
from torch.distributed.fsdp import FSDPModule

# ★★★ 正确顺序: AC first, then FSDP on same module
model.layer = activation_checkpoint(model.layer)  # ← 先AC
model.layer = FSDPModule(model.layer)             # ← 后FSDP

# 或者: apply to specific module types
activation_checkpoint.apply(
    model,
    check_fn=lambda m: isinstance(m, TransformerLayer)
)

★★★★ AC+FSDP2 vs AC+ZeRO-3:
  → FSDP2: composable → 一行apply → 无冲突
  → ZeRO-3: partition_activations → 需要DeepSpeed config → 复杂
```

## 7. Checkpointing: local_state_dict

```python
★★★★★ FSDP2 checkpointing (推荐模式):

# ★★★ Memory-efficient: each rank saves its own shard
local_state = model.local_state_dict()  # 只返回本rank的shard
torch.save(local_state, f"checkpoint_rank_{rank}.pt")

# Load: each rank loads its own shard
local_state = torch.load(f"checkpoint_rank_{rank}.pt")
model.load_local_state_dict(local_state)

# ★★★ 全量checkpoint (需要unshard → 高内存!)
with model.unshard():
    full_state = model.state_dict()  # 所有参数临时unshard → 高peak mem!
    torch.save(full_state, "checkpoint_full.pt")

★★★★ vs ZeRO-3:
  → ZeRO-3: complex state dict mapping → flat buffer → 不易读
  → FSDP2: per-parameter state dict → 人类可读 → 更简单!
```

## 8. FSDP2 + torch.compile

```
★★★★ FSDP2 + torch.compile → PyTorch未来方向:

  → FSDP2 hooks redesigned for compiler → torch.compiler.disable on collectives
  → 减少graph breaks → 但fullgraph=True仍然挑战性
  → ★★★ 实际: compile+FULL_SHARD → 需要reshard hooks → graph breaks不可避免
  → ★★★ 实际: compile+SHARD_GRAD_OP → 无reshard → 更少graph breaks → 更好!

  ★★★ RTX 4090: compile overlay → FSDP2不适用(单GPU) → 但compile单独有用!
  → → torch.compile(model, mode='reduce-overhead') → 单GPU训练加速 → 有效!
```

## 9. FSDP2 vs DeepSpeed ZeRO对比

```
★★★★★ 决定性对比:

| 维度 | FSDP2 | ZeRO-3 | ZeRO-Infinity |
|------|--------|--------|---------------|
| 分片单位 | ★★★ per-param DTensor | sub-partition(粗粒度) | 同ZeRO-3 |
| Padding浪费 | ★★★ 无(精确shard) | 有(NCCL对齐+分区) | 同ZeRO-3 |
| 通信量/step | ★★★ 2Ψ | 3Ψ | 3Ψ(+offload overhead) |
| torch.compile | ★★★ 兼容 | ✗ 不兼容 | ✗ |
| NVMe Offload | ✗ | ✗ | ★★★★ 支持(唯一!) |
| CPU Offload | ★★★ CPUOffload config | ★★★ offload_optimizer | ★★★★ 全offload |
| State Dict | ★★★ per-param(可读) | flat buffer(不可读) | 同ZeRO-3 |
| 模型内混合策略 | ★★★★ YES! | ✗ NO | ✗ |
| HSDP(2D) | ★★★★ DeviceMesh | ✗ | ✗ |
| AC组合 | ★★★ composable | ★★ 需config | ★★ |
| PyTorch原生 | ★★★★ YES | ✗ 外部依赖 | ✗ |
| 配置复杂度 | ★★ 低 | ★★★ 中高 | ★★★★ 高 |

★★★★★ 结论:
  → 不需要NVMe offload → FSDP2更好(更快+更简+compile兼容)
  → 需要NVMe offload → ZeRO-Infinity唯一选择(100B+模型)
  → ★★★ ZeRO-3注定淘汰 → FSDP2是PyTorch路线 → 但ZeRO-Infinity niche存在
```

## 10. RTX 4090影响

```
★★★★★ RTX 4090 SM89:

FSDP2训练 → BF16唯一正确精度
  → FP8: SM89 tensor core存在但无加速 → 不用
  → BF16: SM89 tensor core → 82.6 TFLOPS → 有效

FSDP2单GPU → 无意义!
  → world_size=1 → 不分片 → 全参数 → DDP模式
  → FSDP2只有在≥2GPU时才有意义
  → ★★★ RTX 4090单GPU → 不需要FSDP2 → 但LoRA+compile更有效!

★★★★ 多GPU未来:
  → RTX 4090 × 2 (PCIe) → SHARD_GRAD_OP → 灁3x内存 → 但PCIe瓶颈
  → RTX 4090 × 8 (PCIe) → FULL_SHARD → 灁8x内存 → 但PCIe通信慢
  → ★★★ PCIe瓶颈 → FSDP2多GPU收益有限 → 但内存节省仍然有价值!

★★★★★ RTX 4090推荐路径:
  → 单GPU: LoRA-32 + BF16 + torch.compile → 不需要FSDP2
  → 多GPU(未来): SHARD_GRAD_OP + BF16 + AC → FSDP2有意义
  → ★★★ 当前最优: rLLM Tinker GRPO LoRA → 不用分布式 → 最简!
```

## 11. 7框架定位

```
★★★★★ FSDP2在7框架中的定位:

训练框架:
  → PyTorch FSDP2 → 基础 → ZeRO替代 → torch.compile兼容 → 未来方向
  → DeepSpeed ZeRO → 当前主流 → NVMe offload niche → 但注定淘汰
  → Megatron-LM → TP+PP+DP → 大规模集群 → FSDP2不替代(不同定位)
  → verl → RL训练 → 使用FSDP2或Megatron作为actor策略 → 下层依赖!

推理框架:
  → vLLM → 通用推理 → 不用FSDP2(推理不需要sharding)
  → SGLang → 高吞吐推理 → 同上
  → MindIE → 昇腾推理 → HCCL(不是NCCL/FSDP2)

RL框架:
  → rLLM → 单GPU最优 → FSDP2不需要 → 但多GPU未来可切换verl+FSDP2
  → verl → 多GPU RL → FSDP2/Megatron → 下层分布式策略

★★★★★ 结论:
  → FSDP2是PyTorch路线 → 所有使用PyTorch分布式训练的框架都将迁移
  → verl FSDP2 backend → 直接依赖 → v0.8已经支持!
  → Megatron → TP+PP → FSDP2不替代 → 不同粒度
  → ★★★ 作为AI infra工程师 → 必须理解FSDP2 → 未来所有训练都基于它!
```

## 12. 关键洞察

1. ★★★★★ **FSDP2 = PyTorch分布式训练的未来** → ZeRO-3注定淘汰 → 但ZeRO-Infinity niche存在(NVMe offload)
2. ★★★★★ **DTensor per-parameter** → 参数身份保留 → 无padding → 人类可读state dict → 模型内混合策略
3. ★★★★ **3轴MixedPrecisionPolicy** → param_dtype/reduce_dtype/output_dtype → BF16推荐 → FP8=SM90 only
4. ★★★★ **2Ψ通信量** → 比ZeRO-3少33% → 无backward AllGather → 更高效
5. ★★★★ **composable API** → AC+FSDP+TP组合 → 一行apply → 无class wrapper冲突
6. ★★★★ **HSDP 2D DeviceMesh** → inter-node shard + intra-node replicate → 最优multi-node策略
7. ★★★ **RTX 4090: FSDP2单GPU无意义** → LoRA+compile更有效 → 多GPU未来才需要FSDP2
8. ★★★ **SHARD_GRAD_OP ≈ ZeRO-2** → 实用最优 → 3x内存节省 + 1Ψ通信 → RTX 4090多GPU首选
9. ★★★★ **verl已支持FSDP2** → --actor.strategy=fsdp2 → 直接依赖FSDP2作为分布式策略
10. ★★★ **local_state_dict** → 每rank保存自己的shard → 最内存高效checkpoint → 比ZeRO-3更简单

---

Sources:
- [PyTorch FSDP2 Blog](https://pytorch.org/blog/fsdp2-next-generation/)
- [PyTorch FSDP2 Getting Started](https://pytorch.org/blog/getting-started-with-fsdp2/)
- [torchtitan GitHub](https://github.com/pytorch/torchtitan)
- [PyTorch DeviceMesh Docs](https://pytorch.org/docs/stable/distributed.device_mesh.html)
- [Hugging Face Accelerate FSDP2](https://huggingface.co/docs/accelerate/fsdp2)
- ★★★ Zero3-vs-FSDP2 comparison: `notebook/projects/zero3-vs-fsdp2-system-comparison.md`
- ★★★ PyTorch internals: `notebook/projects/pytorch-fsdp2-internals-reading.md`
- ★★★ Custom op system: `notebook/projects/pytorch-custom-op-library-system-reading.md`
- ★★★ SM89 Compatibility: `tools/sm89_compatibility_checker.py`
