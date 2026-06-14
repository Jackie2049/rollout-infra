# Megatron-LM parallel_state.py 源码深度阅读

> 源码: megatron/core/parallel_state.py (2239行) + process_groups_config.py (719行)
> 核心: 5维并行(TP×CP×EP×DP×PP)的进程组管理 → mixed-radix rank映射 → ProcessGroupCollection迁移

## 1. 全局变量架构

`parallel_state.py` 使用 **module-level全局变量** (singleton)管理所有进程组:

```
_TENSOR_MODEL_PARALLEL_GROUP      # TP组
_PIPELINE_MODEL_PARALLEL_GROUP    # PP组
_MODEL_PARALLEL_GROUP             # TP+PP组合组
_DATA_PARALLEL_GROUP              # DP组
_CONTEXT_PARALLEL_GROUP           # CP组
_EXPERT_MODEL_PARALLEL_GROUP      # EP组
+ 15+复合组 (tp-cp, tp-dp, tp-ep-pp, dp-cp, ...)
```

每组还存 `_GLOBAL_RANKS` (组内world rank列表) + `_MPU_*_RANK`/`_MPU_*_WORLD_SIZE` (可运行时override的缓存值)

## 2. 两个RankGenerator

`initialize_model_parallel()` 创建 **两个** RankGenerator:

**1. decoder_rank_generator** (dense层):
- tp=TP, ep=1, dp=DP, pp=PP, cp=CP
- **ep强制=1** (EP>1和CP>1在同一个RankGenerator互斥!)

**2. expert_decoder_rank_generator** (MoE expert层):
- tp=expert_TP, ep=EP, dp=expert_DP, pp=PP, cp=1
- **cp强制=1** (同理)

**关键约束**: `decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp")` → PP组在dense和expert间完全相同

## 3. Mixed-Radix Rank映射

### 核心数学

```
global_rank = tp_rank × stride_tp + cp_rank × stride_cp + ep_rank × stride_ep + dp_rank × stride_dp + pp_rank × stride_pp
```

stride = **prefix products** 由 `order` 参数决定

### 默认 order="tp-cp-ep-dp-pp"

例: tp=2, cp=1, ep=1, dp=4, pp=2, world_size=16:
- strides: [1, 2, 2, 2, 8] (prefix products of [2,1,1,4,2])
- tp_rank = global_rank % 2
- dp_rank = (global_rank // 2) % 4  (当cp=1, ep=1)
- pp_rank = (global_rank // 8) % 2

### `generate_masked_orthogonal_rank_groups`

给定 `parallel_size` 和 `mask`:
1. **分解** group_index → unmasked dimensions (识别world的哪个"切片")
2. **分解** rank_in_group → masked dimensions (切片内的位置)
3. global_rank = inner_product(decomposed_rank, masked_stride) + inner_product(decomposed_group, unmasked_stride)

`mask` 由 `RankGenerator.get_mask()` 从 `"tp-dp"` 等token生成 → bool array标记哪些维度"在组内"

### Concrete Example (tp=2, pp=4, dp=3, order="tp-pp-dp")

- TP groups: mask=[True, False, False] → size=2 → [[0,1], [2,3], ...]
- DP groups: mask=[False, True, True] → group_index=(tp_rank, pp_rank), rank=dp_rank
- PP groups: mask=[False, True, False] → group_index=(tp_rank, dp_rank), rank=pp_rank

## 4. Getter函数设计

每个getter有 `check_initialized` 参数:

```python
def get_tensor_model_parallel_group(check_initialized=True):
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None
    return _TENSOR_MODEL_PARALLEL_GROUP
```

`check_initialized=False` → **ProcessGroupCollection迁移的关键** → 允许构建collection时不触发不存在组的assertion

### `get_data_parallel_group()` — 最复杂的getter

```python
def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
    if with_context_parallel:
        if partial_data_parallel:
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
        return _DATA_PARALLEL_GROUP_WITH_CP
    if partial_data_parallel:
        # 纯DP+partial → 不允许! → 需要CP
    return _DATA_PARALLEL_GROUP
```

### Rank Getter的二级fallback

```python
# 1. 检查_MPU_*_RANK缓存override值 (可动态设置)
# 2. 无override → .rank()/ .size() on ProcessGroup
→ 支持运行时rank重映射(virtual PP/interleaved调度)
```

## 5. ProcessGroupCollection迁移

### 问题

模块级全局变量 = **隐式全局状态** → 紧耦合 → 不可:
- 同一GPU集上运行多个模型(不同并行配置)
- 通过clean API传递explicit ProcessGroup
- 测试组件无需初始化全局状态

### 解决: `ProcessGroupCollection` (dataclass)

```python
pgs = ProcessGroupCollection()
pgs.tp = tp_group
pgs.pp = pp_group
pgs.dp = dp_group
model = TransformerModel(..., pg_collection=pgs)
```

关键设计:
- 所有字段 `field(init=False)` → 必须先创建再设置属性 → 防止init顺序错误
- 字段 **不required** → `hasattr(pg_collection, 'dp')` 检查组是否存在 → 支持partial collection

### `use_mpu_process_groups()` — 桥接

```python
pgs = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'dp', 'pp'])
# 调用 parallel_state.get_*_group(check_initialized=False) → 从全局状态拉取
# check_initialized=False → 只拉取需要的组, 不触发不需要组的assertion
```

### CLAUDE.md迁移指导

> 在megatron/core生产代码中, **避免新的parallel_state.get_*_group()直接读取**
> 优先接受ProcessGroupCollection或explicit ProcessGroup参数
> 允许的兼容点: parallel_state.py, process_groups_config.py, init/bootstrap, tests, migration fallback

## 6. 复合组: 跨维度通信

| 组 | 维度 | 大小 | 用途 |
|----|------|------|------|
| TP | tp | tp_size | AllReduce层内(Column+Row) |
| PP | pp | pp_size | P2P send/recv |
| DP | dp | dp_size | AllReduce梯度 |
| CP | cp | cp_size | Send/recv KV/dKV (seq切分) |
| MP | tp+pp | tp×pp | 梯度AllReduce(无DP) |
| tp+cp | tp+cp | tp×cp | 注意力计算(全序列+TP) |
| dp+cp | dp+cp | dp×cp | **权重梯度AllReduce** (CP piggybacks on DP!) |
| tp+dp | tp+dp | tp×dp | FP8 amax reduction |
| tp+dp+cp | tp×dp×cp | FP8 amax+CP |
| EP | ep | ep_size | All-to-all dispatch(MoE) |
| expert_tp | etp | etp_size | Expert层TP |
| tp+ep | tp×ep | Expert层通信 |
| tp+ep+pp | tp×ep×pp | MoE梯度sync(不含DP) |
| expert_dp | edp | edp_size | Expert梯度AllReduce(可能≠dense DP!) |

### CP piggybacks on DP: 关键设计

CP切分序列长度(不切分权重) → 权重在CP rank间重复 → 梯度需在DP+CP组内AllReduce
→ `_DATA_PARALLEL_GROUP_WITH_CP` (size=dp×cp) vs `_DATA_PARALLEL_GROUP` (size=dp) → 两个不同communicator!

### Expert DP ≠ Dense DP

当 expert_tp ≠ tp 或 ep > 1:
- expert_dp_size = world / (expert_tp × ep × pp) ≠ data_parallel_size = world / (tp × pp × cp)
→ 需要独立的expert DP组!

## 7. 初始化顺序 (严格!)

1. **dp+cp groups FIRST** → NCCL SHARP只能用于第一个创建的communicator!
   - `NCCL_COLLNET_ENABLE=1` 设置 → 创建dp+cp组 → 删除环境变量 (防泄漏)
2. Hybrid DP+CP sub-groups (power-of-2 sizes)
3. Pure DP groups (SHARP if enabled)
4. CP groups (含hierarchical CP)
5. MP (tp+pp) groups
6. TP groups
7. PP groups (含embedding sub-groups; UCC backend可选)
8. tp+dp+cp / tp+dp / tp+cp groups (FP8)
9. Expert parallel groups (EP, expert_tp, tp+ep, ...)
10. Distributed optimizer instance groups (intra/inter)
11. Global memory buffer

## 8. 3D/4D/5D并行实例

### 3D: TP=2, PP=4, DP=2, order="tp-dp-pp" (16 GPUs)

- 8 TP groups: [0,1], [2,3], ..., [14,15]
- 4 PP groups: [0,4,8,12], [1,5,9,13], ...
- 8 DP groups: [0,8], [1,9], ..., [7,15]
- Rank 5 = tp=1, dp=0, pp=1

### 5D: TP×CP×EP×DP×PP (full stack)

约束: `ep==1 or cp==1` per RankGenerator → 不能同时EP>1和CP>1

实际:
- Dense层: tp × cp × dp × pp (decoder_rank_generator, ep=1)
- Expert层: expert_tp × ep × expert_dp × pp (expert_rank_generator, cp=1)
- 共享PP维度

### Distributed Optimizer Instance Sharding

`num_distributed_optimizer_instances=N` → 层次化DP:
- `intra_partial_dp_cp`: size = (dp×cp)/N → 一个optimizer实例内的梯度shard
- `inter_dist_opt`: size = N → 实例间同步

### UCC Backend for PP

PP组可选UCC backend → zero-SM通信 + 更好IB带宽利用 → 需 `CUDA_DEVICE_MAX_CONNECTIONS > 1`

### AllGather/ReduceScatter Overlap Groups

`create_all_gather_groups()` → 同ranks的独立NCCL communicator → 不同CUDA streams上overlap AG和RS

---

## 9. RTX 4090影响

```
RTX 4090 (8×GPU, PCIe only):
- TP>1: PCIe瓶颈(12GB/s vs NVLink 300+) → ❌
- PP>1: PCIe P2P慢+气泡大 → ❌
- DP>1: AllReduce梯度 → PCIe慢但可行(带宽≈24GB/s for 8-GPU)
- EP>1: AlltoAll → PCIe瓶颈 → ❌
- CP>1: 需TP → ❌

→ RTX 4090最优: TP=1, PP=1, EP=1, CP=1, DP=1 → 单GPU!
→ 8-GPU方案: DP=8 (梯度AllReduce) → 但scaling差(PCIe)
→ 推荐: 单GPU + LoRA + GRPO (rLLM TinkerBackend)

进程组映射:
  TP group = [rank] (size=1, 无通信)
  PP group = [rank] (size=1, 无通信)
  DP group = [rank] (size=1, 无通信)
  → parallel_state.py 退化为 trivial case → 所有group都是singleton!
```

## 10. 与7框架对比

| 框架 | 进程组管理 | 迁移方向 |
|------|-----------|----------|
| Megatron-LM | 5D mixed-radix + global singleton → ProcessGroupCollection | **迁移中** |
| PyTorch | c10d ProcessGroupNCCL (单维度) → FSDP2 DTensor | 稳定 |
| DeepSpeed | ZeRO groups via torch.distributed | 无迁移 |
| vLLM | TP only → ProcessGroupNCCL | 简单 |
| verl | FSDP/Megatron backend → 同Megatron | 同backend |
| rLLM | Tinker(in-process) → 无分布式组; Verl(Ray) → Ray PG | 不同架构 |
| MindIE | HCCL HcclComm → 类似NCCL pattern | 稳定 |

---

Sources:
- `_temp_megatron/megatron/core/parallel_state.py` (2239 lines)
- `_temp_megatron/megatron/core/process_groups_config.py` (719 lines)
- CLAUDE.md migration guidance
