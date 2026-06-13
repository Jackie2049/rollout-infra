# DeepSpeed ZeRO Source Code Deep Dive — Stage 1+2 (Optimizer Partition + Reduce-Scatter) + Stage 3 (Parameter Partition + All-Gather + Prefetch) + ZeRO-Offload + ZeRO-Infinity + PartitionedParameterCoordinator + Init Metaclass Injection + RTX 4090

> 2026-06-13 | DeepSpeed ZeRO源码全链路分析: ZeRO-1(optimizer state分区+all-reduce)+ZeRO-2(gradient分区+reduce-scatter+IPGBucket)+ZeRO-3(parameter分区+on-demand all-gather+PartitionedParameterCoordinator+prefetch+trace)+ZeRO-Offload(CPU/NVMe offload+pin memory)+ZeRO-Infinity(NVMe swapping)+Init metaclass注入+AllGatherCoalescedHandle+ZeRO-3 vs FSDP2对比+RTX 4090
> 源码: github.com/microsoft/DeepSpeed — deepspeed/runtime/zero/stage_1_and_2.py (3038行), stage3.py (3579行), partition_parameters.py (2376行), partitioned_param_coordinator.py, parameter_offload.py, offload_config.py
> 关联: pytorch-torch-distributed-internals-deep-dive.md, nccl-internals-architecture-deep-dive.md, distributed-nav SKILL

## 0. 核心定律: 分区消除冗余 + 按需聚合 + Prefetch重叠通信计算

```
ZeRO三阶段分区定律:

ZeRO-1: Optimizer State Partitioning
  → 每rank只存1/N optimizer states → 内存从4ψ降到4ψ/N
  → 7B BF16: 14GB params + 56GB Adam → 4x70=70GB → 4GPU: 70/4=17.5GB per rank
  → 梯度: all-reduce → 每rank仍有完整梯度副本 → 内存不减!
  → → 通信: 与DDP相同(all-reduce) → 不增加通信量!
  → → → 优势: optimizer内存减N倍 → 通信不变 → 最简单!

ZeRO-2: Gradient Partitioning (在ZeRO-1基础上)
  → 每rank只存1/N gradient → 内存从4ψ+2ψ降到4ψ/N+2ψ/N=(2+4/N)ψ
  → 梯度reduce-scatter → 每rank只收自己分区的梯度 → 不是all-reduce!
  → → 7B BF16: 14GB+14GB+56GB → 4GPU: (14+56)/4+14=32.5GB → vs DDP 84GB!
  → → → 通信: reduce-scatter代替all-reduce → 通信量相同! → 但内存省!
  → → → → 关键: reduce-scatter = all-reduce的等价操作 → 但每rank只保留1/N结果!

ZeRO-3: Parameter Partitioning (在ZeRO-2基础上)
  → 每rank只存1/N parameter → 内存从(2+4/N)ψ降到(2+4/N+2/N)ψ/N ≈ (14/N)ψ/N
  → 参数all-gather on-demand → 前向/反向时临时聚合 → 用完即释放!
  → → 7B BF16: 4GPU → 14/4+14/4+56/4=21GB → vs DDP 84GB → 省4x!
  → → → 通信: 每层2次all-gather(前向+反向) → 通信量增加4x! → 通信vs内存权衡!
  → → → → 关键: 参数不持久 → 用时聚合 → 用完分区 → 内存最小但通信最大!

ZeRO-Infinity: NVMe Offload (在ZeRO-3基础上)
  → optimizer/gradient/parameter → 全offload到NVMe → 内存几乎为0!
  → → 7B BF16: GPU内存≈0 → NVMe存70GB → 理论上可训练任意大模型!
  → → → 通信: NVMe带宽≈2GB/s → 比PCIe慢4x → 比NVLink慢150x → 极慢!
  → → → → → 适用: 模型远大于GPU+CPU内存 → 唯一选择 → 但吞吐极低!

关键设计定律:
  → ZeRO-1: 通信不变 + optimizer内存省N倍 → 最简单 → 推荐起步!
  → ZeRO-2: 通信不变 + optimizer+gradient内存省 → 推荐大部分场景!
  → ZeRO-3: 通信增加4x + 全状态内存省 → 大模型必须 → 但通信开销大!
  → ZeRO-Offload: CPU offload → 算力在CPU → GPU利用率低 → 但GPU内存省!
  → → → → RTX 4090: ZeRO-2+Offload = 单GPU最优 → optimizer在CPU → 14.6GB fits 24GB!
```

## 1. ZeRO Stage 1+2 — DeepSpeedZeroOptimizer (stage_1_and_2.py, 3038行)

```
文件: deepspeed/runtime/zero/stage_1_and_2.py
类: DeepSpeedZeroOptimizer(ZeROOptimizer)

### 1.1 初始化架构

__init__(init_optimizer, param_names, timers, optimizer_params, ...):
  → partition_gradients: bool → False=ZeRO-1, True=ZeRO-2 → 关键区分!
  → reduce_scatter: bool → True时用reduce-scatter而非all-reduce → ZeRO-2必须!
  → cpu_offload: bool → optimizer states offload到CPU → ZeRO-Offload
  → contiguous_gradients: bool → 梯度contiguous buffer → 减少碎片 → 默认True
  → reduce_bucket_size: int → 500M → 梯度reduce桶大小 → 类似DDP bucket_cap_mb
  → overlap_comm: bool → 通信重叠 → 需要独立CUDA stream → 默认False
  → round_robin_gradients: bool → round-robin梯度分区 → 负载均衡

关键数据结构:
  → bit16_groups: List[List[Tensor]] → FP16/BF16参数按组排列
  → bit16_groups_flat: List[Tensor] → 每组参数flatten成单tensor → 类似DDP flat buffer!
  → parallel_partitioned_bit16_groups: List[List[Tensor]] → 按DP度分区 → 每rank一份
  → single_partition_of_fp32_groups: List[Tensor] → 本rank的FP32 master weight分区
  → ipg_buckets: Dict[dtype, IPGBucket] → Independent Partition Gradient桶 → 梯度reduce用!
  → params_in_partition / params_not_in_partition → 分区信息 → 哪些param属于本rank
  → param_to_partition_ids: Dict → 每param属于哪些partition → 跨partition参数!

### 1.2 参数flatten + 分区

初始化流程(关键!):
  1. 按param_group遍历 → trainable_parameters → 排除不需要grad的param
  2. round_robin_reorder → 梯度分区负载均衡 → round-robin排序参数!
     → 原因: 参数按层顺序排列 → 同层参数连续 → 同层梯度归同一rank → 不均衡!
     → → round-robin: param_i → rank i%N → 均衡分布 → 每rank梯度量相近!
  3. flatten_dense_tensors_aligned → 压成flat buffer → NCCL 4-byte alignment!
     → alignment = nccl_start_alignment_factor × world_size → NCCL all-gather需要!
     → → _flatten_dense_tensors → PyTorch工具 → 参数拼接 → 单tensor → 高效!
  4. get_data_parallel_partitions → 按DP度split → 每rank一个分区
     → partition_size = flat_buffer_size / world_size → 等分
     → → 注意: 参数边界可能不与分区边界对齐 → 需first_offset记录偏移!
  5. single_partition_of_fp32_groups → 本rank的FP32分区 → optimizer更新用!
     → .detach().clone() → 从FP16分区 → 转FP32 → 独立副本!
     → → cpu_offload: 转CPU + pin_memory → CPU optimizer offload!
  6. optimizer.param_groups[i]['params'] = [single_partition_of_fp32_groups[i]]
     → 关键! → 原optimizer只看到本rank的flat partition → 只更新本rank参数!
     → → → AdamW: m和v也只存本rank分区 → optimizer内存减N倍!

### 1.3 IPGBucket — 梯度reduce桶

@dataclass IPGBucket:
  → buffer: List[torch.Tensor] → contiguous梯度buffer → 最多2个(index 0/1) → ping-pong!
  → params: List[Tuple(group_idx, param_idx, param_id)] → 哪些参数在桶里
  → grads: List[Tensor] → 梯度引用 → 用完清空
  → elements: int → 桶中元素总数 → 超过reduce_bucket_size时触发reduce!
  → has_moe_params: bool → MoE参数标记 → 用不同process group!

IPGBucket工作流程:
  1. backward → autograd_hook触发 → reduce_independent_p_g_buckets_and_remove_grads(param, i)
  2. 检查: bucket.elements + param.numel() > reduce_bucket_size → 超阈值!
     → → 触发reduce_ipg_grads() → reduce当前桶 → 清空 → 重新填!
     → → → overlap_comm: bucket.index = 1-index → ping-pong → 一个reduce一个填!
  3. 填桶: param梯度 → contiguous拷贝到bucket.buffer → narrow(offset, param.numel())
     → → grad_reduc.data = new_grad_tensor.data → 替换grad → 等reduce!
  4. 桶满 → reduce → 梯度聚合 → 分区 → 本rank保留1/N → 其余释放!

### 1.4 梯度Reduce-Scatter (ZeRO-2核心!)

average_tensor(tensor, communication_data_type):
  → reduce_scatter=False → gradient_reduction_w_predivide → all-reduce → ZeRO-1!
  → reduce_scatter=True → ZeRO-2核心! → reduce-scatter!

ZeRO-2 reduce-scatter流程(关键创新!):
  1. 遍历ipg_buckets中的params → 计算每个param的partition_ids和offset
     → param_to_partition_ids[param_id] → 哪些partition包含这个param
     → → 参数可能跨partition边界 → 一个param可能属于2个partition!
  2. 计算rank_and_offsets → 每个梯度slice的(dst_rank, offset_in_bucket, numel)
     → grad_slice = tensor.narrow(offset, numel) → 梯度切片!
  3. tensor.div_(world_size) → 预除 → reduce-scatter不除 → 需手动!
  4. 按bucket_key分组 → use_multi_rank_bucket_allreduce=True → 混合桶!
     → → allreduce_and_scatter → 多rank桶 → reduce-scatter一次!
  5. reduce-scatter → 每rank只收自己分区的梯度 → 不存完整梯度 → 内存省!

ZeRO-1 all-reduce流程(简单):
  → gradient_reduction_w_predivide → all-reduce完整梯度 → 每rank保留完整副本
  → → predivide: tensor.div_(predivide_factor) → 防溢出!
  → → all_reduce → sum → postdivide → 每rank完整梯度!
  → → → 内存: 每rank存完整梯度 → 不省 → 但简单!

### 1.5 ZeRO-Offload (CPU optimizer offload)

CPU Offload关键设计:
  → self.device = 'cpu' → optimizer在CPU → GPU内存省!
  → DeepSpeedCPUAdam → 自定义CPU Adam → 多核并行 → 比torch.optim.Adam快!
     → → cpuadam_cores_perc=0.8 → 80% CPU核 → 并行optimizer step!
  → pin_memory=True → page-locked → CPU→GPU DMA更快 → 但CPU内存开销!
  → param_buffer_of_bit16_for_cpu_offload_groups → pinned buffer → GPU→CPU传输!

CPU Offload梯度流程:
  1. GPU backward → 梯度在GPU → reduce-scatter → 本rank梯度分区
  2. 梯度分区 → copy到CPU pinned buffer → async DMA → 不阻塞GPU!
  3. CPU optimizer step → AdamW → 更新FP32 master weight → 在CPU!
  4. FP32→FP16 cast → 在CPU → copy到GPU → 更新模型参数!
  → → → 关键: GPU只做forward+backward → optimizer全在CPU → GPU内存最小!

### 1.6 ZeRO-1 vs ZeRO-2 对比

```
特性         ZeRO-1                  ZeRO-2
参数         每rank完整副本           每rank完整副本
梯度         all-reduce→完整副本     reduce-scatter→1/N分区
Optimizer    1/N分区                  1/N分区
通信         all-reduce              reduce-scatter
通信量       ψ (同DDP)               ψ (等价!)
GPU内存      2ψ+4ψ/N                 (2+4/N)ψ+2ψ/N
RTX 4090     optimizer省4x           optimizer+grad省4x → 推荐!
```
```

## 2. ZeRO Stage 3 — DeepSpeedZeroOptimizer_Stage3 (stage3.py, 3579行)

```
文件: deepspeed/runtime/zero/stage3.py
类: DeepSpeedZeroOptimizer_Stage3(ZeROOptimizer)

### 2.1 初始化架构

__init__(module, init_optimizer, param_names, timers, ds_config, ...):
  → parameter_offload: DeepSpeedZeRoOffload → 参数分区管理! → ZeRO-3核心组件!
  → prefetch_bucket_size: 50M → prefetch大小 → 通信计算重叠!
  → max_reuse_distance: 1B → 参数重用距离 → 超过就释放 → 内存控制!
  → max_live_parameters: 1B → 最大同时live参数 → 内存上限!
  → offload_param_config: → 参数CPU/NVMe offload → ZeRO-Infinity!
  → all2all_process_group: → all-to-all quantized reduce → 通信优化!

关键数据结构:
  → fp16_groups: List[List[Parameter]] → 分区后的参数组
  → fp16_partitioned_groups: List[List[Tensor]] → 参数分区(ds_tensor) → 每rank1/N!
  → fp16_partitioned_groups_flat: List[Tensor] → 分区flat buffer → GPU/CPU/NVMe!
  → fp32_partitioned_groups_flat: List[Tensor] → FP32 master weight分区 → optimizer用!
  → param_coordinator: PartitionedParameterCoordinator → prefetch+trace → 核心调度!
  → grad_partitions_flat_buffer: Tensor → 梯度分区flat buffer → 单tensor!
  → ipg_buckets: Dict[dtype, IPGBucketZ3] → ZeRO-3梯度桶 → 和ZeRO-2不同!

### 2.2 参数分区 — partition_parameters.py

Init类(2376行) — Metaclass注入! → 最巧妙的设计!

InsertPostInitMethodToModuleSubClasses:
  → __enter__ → patch_init_and_builtins → 狙击torch.nn.Module.__init__!
  → → 遍历所有torch.nn.Module子类 → 替换__init__ → partition_after包装!
  → → → 每个Module初始化后 → 自动调用_post_init_method → 参数分区!

Init._post_init_method(module):
  → 遍历module.parameters() → 每个param → 调用param.partition()
  → → param.ds_tensor = param.data.narrow(0, rank*partition_size, partition_size) → 只存1/N!
  → → param.data = torch.empty(0) → 释放完整参数! → 内存减N倍!
  → → → 关键: Module.__init__创建完整参数 → Init立即分区释放 → GPU内存最小!

参数状态管理(ZeroParamStatus):
  → AVAILABLE = 1 → 参数完整可用 → 前向/反向时
  → NOT_AVAILABLE = 2 → 参数分区/不在本rank → 释放状态
  → INFLIGHT = 3 → 参数正在all-gather → 过渡状态

参数生命周期(关键!):
  → partitioned → NOT_AVAILABLE → 空tensor → 不占内存!
  → forward需要 → all_gather → INFLIGHT → 正在聚合!
  → all_gather完成 → AVAILABLE → 完整参数 → 计算!
  → backward完成 → partition → NOT_AVAILABLE → 释放! → 内存回收!

### 2.3 All-Gather机制

NoGatherHandle (单GPU场景):
  → rank=0 → 不需要all-gather → ds_tensor直接to(device) → 参数可用!
  → → 单GPU ZeRO-3: 没有通信 → 参数分区→聚合=CPU→GPU copy!

NoGatherCoalescedHandle (单GPU批量):
  → 多参数同时 → 批量CPU→GPU → 单次synchronize → 更高效!

AllGatherHandle (多GPU场景):
  → _dist_allgather_fn → dist.allgather_fn(output, input, async_op=True) → 异步!
  → → wait() → handle.wait() → 同步 → 参数聚合完成!
  → → → param.data = torch.cat(partitions).view(param.ds_shape) → 重组完整参数!

AllGatherCoalescedHandle (多GPU批量 — ZeRO-3核心!):
  → 多参数同时all-gather → 单次通信 → 减少通信次数!
  → → coalesced all-gather → 拼接多个分区 → 一次all-gather → 效率!
  → wait() → 拼接结果 → split回各参数 → status=AVAILABLE!
  → → → 关键优化: 合并通信 → N个参数1次all-gather → vs N次 → 延迟省N倍!

### 2.4 PartitionedParameterCoordinator — Prefetch调度

文件: partitioned_param_coordinator.py

核心职责:
  → 管理参数的fetch(prefetch)+release → 通信计算重叠 → 内存控制!

__ParamInTrace:
  → param: Parameter → 参数引用
  → step_id_last_used_at: int → 上次使用的step → 用于prefetch调度!

ZeRoTraceMode:
  → RECORD → 第一次forward+backward → 记录参数使用顺序!
  → COMPLETE → 后续step → 按trace预取 → 通信计算重叠!
  → INVALID → trace与实际不符 → 重置 → 重新trace!

Prefetch机制(核心优化!):
  1. 第一次forward+backward → RECORD → 记录每个module的参数使用顺序
     → → record_module → 添加到__submodule_order
     → → record_parameters → 添加到__param_order → 含step_id!
  2. 后续step → COMPLETE → 按trace提前prefetch!
     → → forward_fetch_submit → 提交当前module参数的all-gather!
     → → forward_prefetch_submit → 提交下一个module参数的prefetch!
     → → → → 计算+通信重叠! → 当前module计算时 → prefetch下一个module参数!
  3. backward同理:
     → → backward_fetch_submit → 提交反向参数all-gather!
     → → backward_prefetch_submit → prefetch下一个反向module!

内存控制:
  → __max_n_available_params → 最大live参数数 → 超过就release!
  → __max_reuse_dist_in_numel → 参数重用距离 → 超过就release!
  → → release_param → free_param → param.data = torch.empty(0) → 释放!
  → → → 关键: 内存有上限 → 超过就释放旧参数 → 保证不OOM!

__ongoing_fetch_events限制:
  → __max_ongoing_fetch_events = 2 → 最多2个未完成的fetch event!
  → → 原因: fetch在allgather_stream → host线程分配内存 → GPU还没用 → 内存压力!
  → → → 限制并发fetch → 减少内存峰值 → 防OOM!

Thread安全(多线程backward):
  → __leaf_module_lock → threading.Lock → leaf module fetch互斥!
  → → __ongoing_fetch_leaf_module_events → threading.Event → 多线程hook同步!

### 2.5 ZeRO-3 Forward/Backward流程

Forward:
  1. module.__parameters__ → ZeROOrderedDict → __getitem__ → 触发fetch!
     → → 如果param.ds_status == NOT_AVAILABLE → register_external_parameter → all_gather!
  2. param_coordinator.fetch_sub_module(module) → 提交all-gather
     → → 所有参数 → coalesced all-gather → 单次通信 → 高效!
  3. prefetch → 下一个module的参数 → 通信计算重叠!
  4. forward计算 → 使用AVAILABLE参数
  5. backward → gradient hook → reduce-scatter梯度分区!

Backward:
  1. autograd_hook → reduce_and_partition_stream → 梯度reduce-scatter!
  2. 梯度分区 → copy到grad_partitions_flat_buffer → 本rank的梯度分区!
  3. 参数释放 → free_param → NOT_AVAILABLE → 内存回收!

### 2.6 ZeRO-3 Optimizer Step

_step(self):
  1. 遍历fp16_groups → 每sub_group → _optimizer_step(sub_group_id)
  2. fp32_partitioned_groups_flat[i] → 本rank的FP32分区 → optimizer更新!
     → → optimizer.param_groups['params'] = [fp32_param] → 只看到本rank分区!
  3. offload_optimizer → CPU step → DeepSpeedCPUAdam → 多核并行!
  4. FP32→FP16 cast → 更新fp16_partitioned_groups_flat → 本rank分区!
  5. NVMe swap → swap_out_partitioned_params → 写回NVMe!

_optimizer_step:
  → offload → CPU optimizer → cpuadam → parallel across cores
  → GPU optimizer → torch.optim → GPU step → 快但占GPU内存
  → → partial offload → ratio → 混合 → 一部分CPU一部分GPU!

### 2.7 ZeRO-3 vs ZeRO-1/2 vs FSDP2 对比

```
特性                ZeRO-1         ZeRO-2         ZeRO-3         FSDP2
参数                完整副本       完整副本       1/N分区         1/N分区(shard)
梯度                完整副本       1/N分区        1/N分区         1/N分区(reduce_scatter)
Optimizer           1/N分区        1/N分区        1/N分区         1/N分区
参数聚合             不需要         不需要         all-gather      all_gather(unshard)
参数释放             不需要         不需要         partition       reshard
Prefetch             无             无             trace+prefetch  无(简单策略)
梯度聚合             all-reduce     reduce-scatter reduce-scatter  reduce_scatter
通信量/step          ψ             ψ             2ψ(layer数×2)   2ψ(layer数×2)
GPU内存/rank         2ψ+4ψ/N       (2+4/N)ψ+2/Nψ ~14ψ/N        ~14ψ/N
Init方式             无             无             Metaclass注入   wrapper module
RTX 4090推荐         不推荐        推荐+offload   不推荐          不推荐
```

ZeRO-3 vs FSDP2 关键区别:
  → ZeRO-3: Init metaclass → 狙击Module.__init__ → 参数创建后立即分区!
  → FSDP2: _fsdp_param_group → wrapper → shard/unshard生命周期 → 更PyTorch native!
  → → ZeRO-3: PartitionedParameterCoordinator → trace+prefetch → 通信计算重叠!
  → → FSDP2: 简单策略 → 无prefetch → 通信不重叠 → 但简单!
  → → → ZeRO-3: ZeROOrderedDict → __getitem__触发all-gather → 侵入式!
  → → → FSDP2: param.data swap → pre_forward_hook/post_forward_hook → 非侵入!
  → → → → ZeRO-3: 更激进优化(prefetch+trace+coalesced) → 但更复杂!
  → → → → FSDP2: 更简单 → PyTorch官方 → 但性能可能不如ZeRO-3(prefetch)!
```

## 3. Partition Parameters — Init Metaclass注入 (partition_parameters.py, 2376行)

```
文件: deepspeed/runtime/zero/partition_parameters.py

### 3.1 Init类 — ZeRO-3最巧妙设计!

class Init(InsertPostInitMethodToModuleSubClasses):
  → ZeRO-3的核心 → 不是包装module → 而是狙击Module.__init__!

patch_init_and_builtins:
  1. 遍历所有torch.nn.Module子类 → get_all_subclasses → 递归!
  2. 每个子类 → cls._old_init = cls.__init__ → 备份原始__init__!
  3. cls.__init__ = partition_after(cls.__init__) → 替换! → 包装原始__init__!
  4. Module.__init_subclass__ → 也替换 → 新定义的子类也拦截!
  5. torch.Tensor.__new__ → 替换 → dtype和device控制!
  6. torch.nn.functional.linear → zero3_linear_wrap → 内存高效Linear!

partition_after包装函数:
  → 调用原始__init__ → 创建完整参数 → 正常初始化!
  → 检查is_child_module → 只有最终子类才触发_post_init → 不在父类!
  → → 原因: 子类可能调整权重 → 如custom init → 需要完整参数!
  → → → 只有子类__init__完成 → 才分区 → 保证初始化正确!
  → 调用self._post_init_method(module) → 分区+释放!

Module.apply包装:
  → apply_with_gather → 包装fn_to_apply → 先all-gather → 再apply → 再partition!
  → → 原因: apply可能修改权重 → 需要完整参数 → 分区后无法修改!
  → → → 先聚合 → 再修改 → 再分区 → 保证正确!

Tensor构造函数替换:
  → torch.tensor/empty/zeros/ones/full/arange/eye/randn → 全替换!
  → → target_fp_dtype → 强制BF16/FP16 → 不创建FP32临时tensor!
  → → target_device → 强制GPU → 不在CPU创建 → 减少CPU→GPU copy!
  → → → 关键: 所有tensor创建 → 自动BF16+GPU → 不浪费内存!

### 3.2 参数分区方法

param.partition():
  → param.ds_tensor = param.data.narrow(rank*partition_size, partition_size)
  → → 只保留本rank的1/N → ds_tensor存分区数据!
  → param.ds_numel = param.numel() → 原始大小 → 聚合时需要!
  → param.ds_shape = param.shape → 原始形状 → 聚合时需要!
  → param.ds_status = ZeroParamStatus.NOT_AVAILABLE → 释放状态!
  → param.data = torch.empty(0) → 释放完整参数 → GPU内存回收!
  → → → 关键: ds_tensor在GPU/CPU/NVMe → 视offload配置!
  → → → → 不offload → GPU分区 → 快! → offload → CPU/NVMe → 省GPU但慢!

param.all_gather():
  → _dist_allgather_fn → dist.allgather_fn(output, input, async_op=True)
  → → 每rank发送自己的ds_tensor分区 → 聚合成完整参数!
  → → → param.data = torch.cat(partitions).view(param.ds_shape) → 重组!
  → → → → param.ds_status = ZeroParamStatus.AVAILABLE → 可用!

free_param(param):
  → param.data.record_stream(current_stream) → 等stream用完 → 安全释放!
  → param.data = torch.empty(0) → 释放! → GPU内存回收!
  → param.ds_status = ZeroParamStatus.NOT_AVAILABLE → 分区状态!

### 3.3 ZeROOrderedDict — 参数访问拦截

class ZeROOrderedDict(OrderedDict):
  → 替换module._parameters → __getitem__拦截参数访问!

  __getitem__(key):
    → 如果param.ds_status == NOT_AVAILABLE → 检查_in_forward!
    → → _in_forward=True → register_external_parameter → all_gather → 触发聚合!
    → → → 关键: forward时访问参数 → 自动all-gather → 透明!
    → → → → 用户不需要手动all-gather → ZeRO-3自动管理!

  ZeROOrderedDict._in_forward → FWD_MODULE_STACK → 标记forward阶段!
  → → forward → _in_forward=True → 自动all_gather!
  → → backward → _in_forward=False → 不自动all_gather → hook处理!
```

## 4. ZeRO-Offload + ZeRO-Infinity

```
文件: deepspeed/runtime/zero/offload_config.py + parameter_offload.py

### 4.1 Offload设备层级

OffloadDeviceEnum:
  → none → 不offload → 全在GPU → 默认
  → cpu → offload到CPU → page-locked可选 → ZeRO-Offload
  → nvme → offload到NVMe → 磁盘 → ZeRO-Infinity

DeepSpeedZeroOffloadParamConfig (ZeRO-3专用):
  → device: OffloadDeviceEnum → offload目标设备
  → nvme_path: Path → NVMe文件路径 → ZeRO-Infinity
  → buffer_count: 5 → NVMe buffer池大小
  → buffer_size: 100M → NVMe buffer大小
  → max_in_cpu: 1B → CPU最大参数数 → NVMe→CPU→GPU分层!
  → pin_memory: False → page-locked → DMA加速 → 但内存开销!

DeepSpeedZeroOffloadOptimizerConfig (ZeRO-1/2/3通用):
  → device: OffloadDeviceEnum → optimizer offload目标
  → nvme_path: Path → NVMe路径 → optimizer state到磁盘!
  → buffer_count: 4 → Adam有4个state → 需要4个buffer!
  → pipeline_read/write: False → tile-based overlapping → NVMe预取!
  → fast_init: False → NVMe快速初始化 → 不预读!
  → ratio: 1.0 → CPU offload比例 → partial offload!
  → super_offload: False → Superchip高性能offload → 特定硬件!
  → cpuadam_cores_perc: 0.8 → CPU Adam核数占比!

### 4.2 DeepSpeedZeRoOffload — ZeRO-3参数管理

class DeepSpeedZeRoOffload:
  → module → 原始模型
  → param_coordinator → PartitionedParameterCoordinator → prefetch调度!
  → __allgather_stream → 独立CUDA stream → 通信重叠!
  → __inflight_param_registry → InflightParamRegistry → 正在all-gather的参数!
  → persistent_parameters → 持久参数 → 不分区 → 小参数留在GPU!
  → → param_numel_persistence_threshold → 100K → 小于此阈值不分区 → 减少通信!

初始化:
  1. _convert_to_zero_parameters → 所有参数→ZeRO参数 → ds_tensor+ds_status!
  2. _init_external_params → 添加ds_external_parameters → 外部参数注册!
  3. _inject_parameters → module._parameters = ZeROOrderedDict → 拦截访问!
  4. mark_persistent_parameters → 标记持久参数 → 不分区 → 减少通信!
  5. PartitionedParameterCoordinator → prefetch+trace+release → 调度!

### 4.3 NVMe Offload层级 (ZeRO-Infinity)

三级存储层次:
  → NVMe → slowest → 2GB/s → 参数和optimizer state持久存储!
  → CPU → medium → DDR bandwidth → 活跃参数和梯度暂存!
  → GPU → fastest → HBM bandwidth → 当前计算需要的参数!

数据流:
  → NVMe → CPU → GPU (swap_in) → 计算用
  → GPU → CPU → NVMe (swap_out) → 释放存
  → → max_in_cpu → CPU最大参数数 → 超过就swap到NVMe!
  → → → Pipeline: pipeline_read → 预读下一tile → overlap NVMe→CPU!

PartitionedParamStatus:
  → AVAILABLE → 在CPU/GPU → 可用
  → NOT_AVAILABLE → 在NVMe → 需swap_in → 不可用!
  → → ZeRO-Infinity: 参数可能在NVMe → 需要swap → 延迟大!
```

## 5. RTX 4090 ZeRO推荐

```
1. ZeRO-1对RTX 4090:
  → 7B BF16: 14GB+14GB+56GB=84GB → 8GPU ZeRO-1: 14+14+56/8=35GB → 24GB不够!
  → → optimizer省8x → 但参数+梯度仍14+14=28GB → >24GB → OOM!
  → → → 结论: ZeRO-1对4090无意义 → optimizer省不够 → 参数+梯度仍大!

2. ZeRO-2对RTX 4090:
  → 7B BF16: 14+56/8+14/8=22.75GB → 接近24GB → 可能fits!
  → → 但! 激活内存≈2.1GB → 总≈24.85GB → OOM风险!
  → → → ZeRO-2+Offload: optimizer→CPU → 14+14/8+2.1=16.35GB → fits 24GB!
  → → → → 但! DDP需要8GPU → PCIe灾难 → 0.46x → 不推荐!
  → → → → → 单GPU ZeRO-2+Offload: 14+2.1+CPUoptimizer=16.1GB → fits!
  → → → → → → 推荐: 单GPU ZeRO-2+CPU optimizer offload → 7B BF16 fits 24GB!

3. ZeRO-3对RTX 4090:
  → 7B BF16单GPU: 14/1+14/1+56/1=14+14+56 → ZeRO-3不省内存!
  → → 单GPU → world_size=1 → 分区=不分区 → ZeRO-3无意义!
  → → 8GPU ZeRO-3: 14/8+14/8+56/8=10.5GB → fits → 但通信4x → PCIe更灾难!
  → → → 结论: ZeRO-3对4090不推荐 → 通信开销远大于内存节省!

4. LoRA+Gradient Checkpointing对RTX 4090:
  → 7B BF16+LoRA: 14.6GB → 单GPU → 不需要ZeRO → 最简单最优!
  → → gradient checkpointing → 激活2.1→0.5GB → 总≈15.1GB → 安全!
  → → → 结论: RTX 4090最优策略 = LoRA+grad checkpointing → 不需要ZeRO!

5. ZeRO-Offload对RTX 4090:
  → 单GPU+CPU offload → optimizer→CPU → GPU内存≈14+0.5=14.5GB → fits!
  → → 但! CPU optimizer慢 → CPU Adam ≈10-30ms/step → vs GPU Adam ≈1ms!
  → → → LoRA+grad checkpointing → 不需要offload → 更快!
  → → → → 全参数微调+ZeRO-2+Offload → optimizer在CPU → 22.75-14=8.75GB GPU → fits!
  → → → → → 全参数微调场景: ZeRO-2+Offload → 唯一选择 → 但LoRA更优!

6. RTX 4090最终推荐:
  → 7B LoRA微调 → 单GPU → LoRA+grad checkpointing → 14.6GB → 最优!
  → 7B 全参数微调 → 单GPU → ZeRO-2+CPU optimizer offload → 最优!
  → → → 8GPU DDP → 不推荐 → PCIe灾难 → 0.46x!
  → → → ZeRO-3 → 不推荐 → 通信4x → PCIe更灾难!
```

## 6. DeepSpeed ZeRO关键数据结构索引

```
stage_1_and_2.py (ZeRO-1+2):
  → DeepSpeedZeroOptimizer → ZeRO-1/2 optimizer类
  → IPGBucket → 梯度reduce桶(dataclass)
  → bit16_groups_flat → 参数flat buffer(List[Tensor])
  → parallel_partitioned_bit16_groups → 参数按DP分区(List[List[Tensor]])
  → single_partition_of_fp32_groups → 本rank FP32分区(List[Tensor])
  → param_to_partition_ids → 参数→分区映射(Dict)
  → averaged_gradients → 梯度分区(Dict[int, List[Tensor]])
  → reduction_stream → 独立CUDA stream → overlap_comm时

stage3.py (ZeRO-3):
  → DeepSpeedZeroOptimizer_Stage3 → ZeRO-3 optimizer类
  → IPGBucketZ3 → ZeRO-3梯度桶(dataclass) → 和ZeRO-2不同!
  → parameter_offload → DeepSpeedZeRoOffload → 参数管理
  → fp16_partitioned_groups_flat → 参数分区flat buffer
  → fp32_partitioned_groups_flat → FP32分区flat buffer
  → grad_partitions_flat_buffer → 梯度分区flat buffer → 单tensor!
  → reduce_and_partition_stream → 梯度reduce stream

partition_parameters.py:
  → Init → Metaclass注入类 → 狙击Module.__init__ → 分区
  → AllGatherHandle → 单参数all-gather handle
  → AllGatherCoalescedHandle → 批量all-gather handle → 核心!
  → NoGatherHandle → 单GPU no-gather handle
  → NoGatherCoalescedHandle → 单GPU批量no-gather handle
  → ZeroParamType → NORMAL/PARTITIONED/REMOTE
  → ZeroParamStatus → AVAILABLE/NOT_AVAILABLE/INFLIGHT
  → CUDAQuantizer → ZeRO-3量化 → 8bit symmetric → reduce通信量!
  → free_param → 释放参数存储 → param.data=torch.empty(0)

partitioned_param_coordinator.py:
  → PartitionedParameterCoordinator → prefetch调度核心!
  → ZeRoTraceMode → RECORD/COMPLETE/INVALID → trace模式
  → InflightParamRegistry → 正在all-gather的参数注册表
  → __ParamInTrace → trace中的参数+step_id

parameter_offload.py:
  → DeepSpeedZeRoOffload → ZeRO-3参数offload管理
  → ZeROOrderedDict → 替换module._parameters → __getitem__拦截!
  → _inject_parameters → module._parameters → ZeROOrderedDict → 注入!

offload_config.py:
  → OffloadDeviceEnum → none/cpu/nvme → offload目标
  → DeepSpeedZeroOffloadParamConfig → 参数offload配置
  → DeepSpeedZeroOffloadOptimizerConfig → optimizer offload配置
  → OffloadStateTypeEnum → optim_states/hp_params/lp_params/lp_grads/contiguous_grad_buffer
```

## 7. ZeRO-3源码洞察 → 关键发现

```
1. Init metaclass注入 → 最巧妙的设计!
  → 不是包装module → 而是狙击Module.__init__ → 创建后立即分区!
  → → vs FSDP2: wrapper方式 → 更显式 → 但不够透明!
  → → → ZeRO-3: 用户不需要修改代码 → 自动分区 → 透明!
  → → → → 但! __init__替换 → 全局影响 → 其他代码也受影响 → 潜在风险!

2. ZeROOrderedDict → 参数访问拦截 → 最侵入式设计!
  → module._parameters → ZeROOrderedDict → __getitem__ → 自动all_gather!
  → → 用户访问param → 自动触发all-gather → 透明 → 但不可控!
  → → → vs FSDP2: hook方式 → pre_forward_hook → 更可控 → 更PyTorch native!

3. PartitionedParameterCoordinator → 最优化prefetch!
  → trace → 记录参数使用顺序 → 预取 → 通信计算重叠!
  → → 3种trace模式 → RECORD→COMPLETE→INVALID → 自适应!
  → → → 第一次记录 → 后续优化 → 不匹配重记录 → 自适应!
  → → → → vs FSDP2: 无prefetch → 顺序all_gather → 通信不重叠 → 性能差!

4. IPGBucket → ping-pong buffer → 通信重叠!
  → 2个buffer(index 0/1) → 一个reduce → 一个填 → 交替!
  → → overlap_comm → reduction_stream → 独立stream → GPU计算+通信并行!
  → → → vs DDP: 单buffer → 不重叠 → 通信阻塞计算 → 慢!

5. CUDAQuantizer → ZeRO-3量化通信!
  → 8bit symmetric quantize → all-gather数据量化 → 通信量减2x!
  → → target_group_size=8000 → 最优group_size=4k → 量化精度vs通信权衡!
  → → → vs FP16 all-gather → 14GB参数→all-gather→7GB量化→通信减半!

6. round_robin_gradients → 负载均衡!
  → 参数按层顺序排列 → 同层归同rank → 不均衡!
  → → round-robin: param_i → rank i%N → 均衡 → 每rank梯度量相近!
  → → → 关键: reduce-scatter后 → 每rank只保留1/N → 均衡减少等待!

7. ZeRO-3的__ongoing_fetch_events限制 → 内存控制!
  → 最多2个并发fetch → 防止内存峰值 → host分配→GPU还没用!
  → → → 理由: all-gather async → host分配output buffer → GPU还没消费 → 内存压力!
  → → → → 限制并发 → host最多分配2个buffer → 内存可控!

8. ZeRO-Infinity的NVMe offload → 三级存储!
  → NVMe → CPU → GPU → 分层 → max_in_cpu限制 → 超过就NVMe!
  → → pipeline_read/write → 预读下一tile → overlap NVMe→CPU!
  → → → 适用: 模型远大于GPU+CPU → 唯一选择 → 但吞吐极低!
```

## 参考文献

```
1. DeepSpeed ZeRO论文:
   - Rajbhandari et al., "ZeRO: Memory Optimization Towards Training A Trillion Parameter Models", 2020
   - Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Optimizing Extreme-Scale Deep Learning Models", 2021

2. DeepSpeed源码:
   - github.com/microsoft/DeepSpeed/deepspeed/runtime/zero/
   - stage_1_and_2.py (3038行) → ZeRO-1+2 optimizer
   - stage3.py (3579行) → ZeRO-3 optimizer
   - partition_parameters.py (2376行) → 参数分区+Init metaclass
   - partitioned_param_coordinator.py → prefetch调度
   - parameter_offload.py → ZeRO-Offload
   - offload_config.py → offload配置枚举

3. FSDP2对比:
   - pytorch-torch-distributed-internals-deep-dive.md → FSDP2生命周期
   - torch/distributed/fsdp2/_fsdp_param_group.py → FSDP2源码

4. 我们的笔记:
   - pytorch-torch-distributed-internals-deep-dive.md → ProcessGroupNCCL+DDP+FSDP
   - nccl-internals-architecture-deep-dive.md → NCCL Channel+Proxy+Protocol
   - distributed-nav SKILL → torch.distributed+NCCL导航
