# DeepSpeed 分布式优化器源码深度阅读

> 源码: deepspeed/runtime/base_optimizer.py + stage_1_and_2.py + stage3.py + bf16_optimizer.py + partitioned_param_coordinator.py + contiguous_memory_allocator.py
> 核心: 优化器分区算法 + 梯度缩减流程 + 连续内存管理 + 混合精度集成

## 1. 优化器类层次结构

```
DeepSpeedOptimizer (base_optimizer.py:17)        -- empty base class
    |
    +-- ZeROOptimizer (base_optimizer.py:250)     -- shared ZeRO infrastructure
    |       |                                       (BackwardHookStateManager, _configure_master_weights)
    |       +-- DeepSpeedZeroOptimizer (stage_1_and_2.py:126)  -- ZeRO Stage 1 & 2
    |       +-- DeepSpeedZeroOptimizer_Stage3 (stage3.py:136)  -- ZeRO Stage 3
    |       +-- BF16_Optimizer (bf16_optimizer.py:36)           -- BF16 variant
    |
    +-- FP16_Optimizer (fused_optimizer.py:33)               -- Legacy fused fp16
    +-- FP16_UnfusedOptimizer (unfused_optimizer.py:24)      -- Legacy unfused fp16
```

**ZeROOptimizer** (base_optimizer.py:250) 包含 `BackwardHookStateManager` — 管理:
- 多阶段backward状态(重入梯度检查 + 嵌套深度追踪 + epilogue触发逻辑)
- `_configure_master_weights()` (line 467-511): 所有ZeRO阶段共享
  - fp32 master(默认) / fp16 master(需CPU offload+DeepSpeedCPUAdam) / bf16 master(需CPU offload+DeepSpeedCPUAdam fp32_optimizer_states=False)

## 2. ZeRO-1/2 分区算法 (stage_1_and_2.py:359-510)

### 初始化流程

```
1. 扁平化参数 (line 424-441):
   bit16_groups_flat[i] = concat所有fp16参数为1个连续张量
   对齐: nccl_start_alignment_factor * dp_world_size (fp16=2*dp, 4字节NCCL对齐)

2. 分区扁平化张量 (line 455):
   get_data_parallel_partitions() (line 1826-1844):
   base_size = total_num_elements // dp
   remaining = total_num_elements % dp
   Ranks 0..(remaining-1): base_size+1 elements; rest: base_size

3. FP32主权重分区 (line 477-494):
   weights_partition = parallel_partitioned_bit16_groups[i][partition_id].clone().to(fp32)
   → single_partition_of_fp32_groups.append(weights_partition)

4. 优化器指向分区 (line 501):
   param_group['params'] = [single_partition_of_fp32_groups[i]]
   → Adam只为1/N的参数创建momentum/variance!

5. 分区信息跟踪 (line 504):
   get_partition_info() (line 1846-1875): 记录哪些参数属于哪个分区
   处理跨分区边界的参数

6. 轮询梯度重排 (line 403-408):
   round_robin_gradients: 重排参数使连续梯度属于不同秩
   → 更好的并行梯度复制到CPU
```

### 关键: 优化器只为 1/N 参数创建状态

Adam状态: momentum(fp32) + variance(fp32) = 8Ψ/N 字节
加上FP32主权重: 4Ψ/N 字节
总计: 12Ψ/N → 8GPU上每GPU仅1.75GB(vs 84GB全量)

## 3. ZeRO-1/2 参数更新流程 (stage_1_and_2.py:2140-2269)

```
step():
  1. Check overflow (line 2149)
  2. If overflow → zero_grad, skip (line 2154-2170)
  3. Compute scaled_global_grad_norm (line 2174)
  4. For each param_group:
     a. Free gradients NOT in this partition (line 2205)
     b. Flatten averaged_gradients IN this partition (line 2216-2219)
     c. Assign flattened grad as fp32_partition.grad (line 2226)
     d. Free gradient copies IN this partition (line 2228)
     e. Unscale and clip grads (line 2233)
     f. optimizer.step() on single fp32 partition (line 2239)
     g. Copy fp32→fp16 partition (line 2245):
        bit16_partitions[partition_id].data.copy_(fp32_partition.data)
  5. AllGather updated fp16 weights (line 2255-2259):
     all_gather_dp_groups → 重建完整权重
  6. Update model fp16 params to point to gathered buffer (line 2264)
```

**关键** (line 2245): 只更新本秩的fp16分区 → AllGather重建完整权重 → 所有GPU看到一致更新

## 4. ZeRO-3 分区算法 (stage3.py)

### 参数分区 (partition_parameters.py)

每个参数获得 `ds_tensor` 属性:
- `ds_tensor`: 1/N分区存储在GPU/CPU/NVMe
- `partition_numel()`: ceil(ds_numel / dp_world_size)
- `padding_size()`: 对齐填充
- 参数状态: AVAILABLE(完全存在) / NOT_AVAILABLE(已分区) / INFLIGHT(正在AllGather)

### FP32分区创建 (stage3.py:919-1046, `_create_fp32_partitions`)

```
- NVMe swap: 创建空FP32张量,稍后swap in
- CPU offload: 部分子组分配CPU,部分GPU(部分offload比例)
- 标准GPU: 从fp16分区clone为fp32
- optimizer param_groups初始化后清空(line 1045-1046)
  → 每步骤为每个子组动态填充
```

## 5. ZeRO-3 步骤流程 (stage3.py:2424-2468)

```
step():
  1. _pre_step(): reset micro_step_id, clear stale ipg buckets (line 2185)
  2. _partition_all_parameters(): 所有参数回分区状态 (line 2870)
  3. Overflow check + loss scale update (line 2432)
  4. Compute norm_groups + scaled_global_grad_norm (line 2437-2441)
  5. For each sub_group:
     a. _prepare_sub_group(): swap in或准备fp32 grad
        - Non-offload: flatten averaged_gradients→fp32_partition.grad (line 2218)
        - Swap: optimizer_swapper.swap_in_optimizer_state (line 2259)
     b. unscale_and_clip_grads (line 2455)
     c. _optimizer_step(): param_group→fp32_partition, optimizer.step(), clear (line 2458)
     d. _reassign_or_swap_out_partitioned_parameters (line 2461):
        Copy fp32→fp16_partitioned_groups_flat (line 2409)
        Unflatten partitioned parameters (line 2413)
     e. _release_sub_group(): clear fp32 grad, swap out optimizer states (line 2464)
  6. _post_step(): gather persistent parameters, invalidate secondary tensors (line 2468)
```

**关键区别**: ZeRO-3 步骤不执行AllGather! 参数在下次forward时按需gather(PartitionedParameterCoordinator)

## 6. 梯度缩减流程

### ZeRO-1 (无梯度分区, partition_gradients=False)

```
reduce_gradients():
  For each param in bit16_groups:
    reduce_ready_partitions_and_remove_grads(param)  # AllReduce
  overlapping_partition_gradients_reduce_epilogue()   # finalize
```

标准AllReduce → 每秩提取自己分区的梯度份额

### ZeRO-2 (梯度分区, partition_gradients=True)

```
backward期间:
  1. 参数梯度hook触发 → 梯度进入ipg_buckets(dtype分组)
  2. bucket达reduce_bucket_size → ReduceScatter(只传本秩梯度分区)
  3. averaged_gradients[i][j] = 仅本秩参数的梯度
  4. 其他秩参数的梯度→丢弃

overlap_comm=True时:
  单独CUDA stream(reduction_stream, line 516) overlap梯度缩减与backward计算
```

### ZeRO-3 梯度流

```
1. Forward: PartitionedParameterCoordinator.fetch_sub_module() → 异步AllGather参数
2. Forward后: 参数保留AVAILABLE用于backward
3. Backward: 计算使用gathered参数
4. Backward hook触发 (stage3.py:1279, create_reduce_and_remove_grad_hooks):
   reduce_scatter_coalesced() / all_to_all_quant_reduce() / reduce_scatter()
   → 每秩仅接收梯度分区
5. 参数释放: release_sub_module() → NOT_AVAILABLE
6. grad_partitions_flat_buffer (stage3.py:637): 所有梯度分区1个连续张量
```

### BF16 梯度流

```
1. backward_prologue(): clear_lp_grads() (line 337)
2. backward: 梯度累积到bf16参数.grad
3. backward_epilogue()→update_hp_grads(): bf16→fp32梯度 (line 345-355)
   hp_grad.data.add_(lp.grad.data.to(hp_grad.dtype))
4. immediate_grad_update模式: backward期间立即触发
   → lp.grad→hp_grad→清lp.grad → 更及时但可能碎片化
```

## 7. 连续内存管理

### ContiguousMemoryAllocator (contiguous_memory_allocator.py:16-288)

```
初始化: torch.zeros(size, dtype, device) + 空闲地址映射 + 张量地址追踪 + 参数关联

allocate_tensor (line 51-78):
  找最小contiguous_size ≥ requested_size → narrow创建子张量
  碎片化阻止分配 → _defragment_memory()

_defragment_memory (line 179-230):
  所有活动张量→缓冲区开头压缩 → 合并空闲空间
  复制后_reset_param_data() (line 138-141) → 修复参数.data指针

release_tensor (line 97-106):
  删除张量追踪 → _consolidate_address()合并相邻空闲区域

assign_to_param (line 83-94):
  张量缓冲区view→参数.data via narrow(0,0,numel).view(shape)
```

### ZeRO-1/2 连续梯度缓冲区

- `IPGBucket.buffer` (line 112): 连续梯度防止backward期间碎片化
- `contiguous_gradients=True` (line 249): 强制连续缓冲区(CPU offload必须)

### ZeRO-3 连续缓冲区

- `grad_partitions_flat_buffer` (line 637-647): 所有梯度分区1个连续张量 → 消除碎片化
- `lp_param_buffer` (line 811): fp16分区张量的单个连续缓冲区
- `defragment()` (utils.py:207-235): GPU→CPU→清缓存→GPU → 一次性碎片整理

### BF16 连续缓冲区

- `fp32_groups_gradients_flat[i]` (line 173): 每组1个连续fp32梯度缓冲区
- 所有fp32梯度视图是此缓冲区的narrow视图 (line 181-193)

## 8. 混合精度集成

### FP16 流程

```
模型参数=fp16 → 主权重=fp32 → 梯度fp16→fp32(步骤前)
Loss scaling: 动态或静态 → 防溢出
backward: loss.float() * loss_scale → scaled_loss.backward()
step: fp16 grads→flatten→cast fp32→optimizer.step()→copy fp32→fp16
```

### BF16 流程

```
模型参数=bf16 → 主权重=fp32分区 → 梯度累积dtype可选
grad_acc_dtype=torch.float32: fp32累积 → 推荐(精度好)
grad_acc_dtype=torch.bfloat16: bf16累积 → 省内存但精度差
BF16模式无loss scaling (line 63-66): custom_loss_scaler=False
immediate_grad_update: backward期间立即bf16→fp32 → 更及时
```

### ZeRO-3 混合精度

```
gradient_accumulation_dtype (line 176): fp32或fp16/bf16
communication_data_type (line 177): ReduceScatter通信dtype
bf16_optimizer_states选项 (line 180): Adam momentum/variance也bf16
  → 需CPU offload + DeepSpeedCPUAdam fp32_optimizer_states=False
```

## 9. ZeRO-3 vs ZeRO-1/2 关键差异表

| 维度 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| **分区对象** | 优化器状态 | 优化器+梯度 | 优化器+梯度+参数 |
| **参数位置** | 全量每GPU | 全量每GPU | 1/N分区每GPU |
| **梯度缩减** | AllReduce | ReduceScatter | ReduceScatter(coalesced) |
| **参数AllGather** | 无 | 无(参数完整) | 每层forward+backward |
| **通信总量/步** | 1Ψ(grad AllReduce) | 1Ψ(grad RS≈AR) | 3Ψ(AG+RS+AG) |
| **步骤后** | AllGather更新参数 | AllGather更新参数 | 不AllGather(下次forward按需) |
| **内存峰值** | ~16Ψ/N | ~14Ψ/N | ~2Ψ/N+Ψ(AllGather temp) |
| **优化器指向** | 1/N FP32分区 | 1/N FP32分区 | 动态填充子组 |
| **连续缓冲区** | IPGBucket | IPGBucket+ipg_buffer | grad_partitions_flat_buffer+lp_param_buffer |
| **compile兼容** | ✓ | ✓ | ✗(动态AllGather→graph break) |

## 10. RTX 4090 实战分析

```
内存计算 (7B模型):
  Ψ = 7e9, bf16_model=14GB, fp32_optimizer=84GB(Adam)

  ZeRO-1 8GPU: 14GB(model)+14GB(grads)+10.5GB(optimizer)=38.5GB ✗>24GB
  ZeRO-2 8GPU: 14GB(model)+1.75GB(grads shard)+10.5GB(optimizer shard)=26.25GB ✗>24GB
  ZeRO-2+CPU_Adam: 14GB(model)+1.75GB(grads shard)+0(optimizer on CPU)+1.4GB(activation)=17.15GB ✓
  LoRA r16+CPU_Adam: 14GB(base)+0.26GB(LoRA)+0(optimizer on CPU)+1.4GB(activation)=15.66GB ✓
  GRPO LoRA r16: 14GB(base)+0.26GB(LoRA)+0(optimizer on CPU)+2.8GB(activation×2)=17.06GB ✓

  ZeRO-3 8GPU PCIe: 1.75GB(model shard)+1.75GB(grads shard)+10.5GB(optimizer shard)+14GB(AllGather temp)=27.75GB ✗
  → 加上AllGather时间: 14秒/步(PCIe) → 完全不可用!

  关键: RTX 4090单GPU最优=LoRA+CPU_Adam → 无分布式通信开销!
  多GPU PCIe: DDP+LoRA AllReduce 0.11GB/step → 4.56ms → 可行但scaling差
```

## 11. BF16训练选择

```
RTX 4090 (SM 8.9 Ada):
  BF16硬件支持: ✓ (GEMM 169.6 TFLOPS BF16)
  推荐: bf16 master weights + fp32 optimizer states + fp32 gradient accumulation
  不推荐: fp16(需loss scaling,溢出风险大) + bf16 optimizer states(精度差)

  BF16优势 vs FP16:
  1. 无loss scaling → 简化流程 → 无溢出跳步
  2. 更大动态范围 → 训练更稳定
  3. immediate_grad_update → backward期间bf16→fp32 → 减少epilogue开销
```

Sources:
- `deepspeed/runtime/base_optimizer.py` (511 lines)
- `deepspeed/runtime/engine.py` (gradient reduction hooks)
- `deepspeed/runtime/zero/stage_1_and_2.py` (2269 lines)
- `deepspeed/runtime/zero/stage3.py` (2870 lines)
- `deepspeed/runtime/zero/bf16_optimizer.py` (571 lines)
- `deepspeed/runtime/zero/partitioned_param_coordinator.py` (634 lines)
- `deepspeed/runtime/zero/partition_parameters.py` (316 lines)
- `deepspeed/runtime/zero/contiguous_memory_allocator.py` (288 lines)
- `deepspeed/runtime/zero/config.py` (90 lines)
