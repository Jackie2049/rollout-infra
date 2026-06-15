# PyTorch DTensor Autograd 源码级深度阅读

> 2026-06-15 | 源码: torch/distributed/tensor/_api.py + _redistribute.py + device_mesh.py + op.py + api.py + FSDP flat_param.py
> 核心: DTensor autograd=__torch_dispatch__+C++ fast path+_ToTorchTensor/_FromTorchTensor→Redistribute(NestedRedistribute二次可微)+梯度placement规则(Shard→Shard/Replicate→Replicate/Partial→Replicate)+FSDP2 RegisterPre/PostBackwardFunction→AllGather/ReduceScatter→OpDispatcher+register_sharding_propagation_rule→split_group vs new_group(ProcessGroupCollection)+CuTe _MeshLayout layout algebra

## 1. DTensor Autograd 整体架构

```
★ ★ ★ DTensor autograd = 3层 dispatch + autograd Function 包装:

Layer 1: C++ Fast Path (python_variable.cpp):
  → torch._C._distributed.DTensor → C++ binding → 快速类型检查
  → 比Python dispatch快 → 减少overhead → 关键性能优化

Layer 2: Python __torch_dispatch__ (op.py):
  → OpDispatcher.__torch_dispatch__(func, types, args, kwargs)
  → DTensor输入→sharding propagation→输出DTensor
  → 非DTensor输入→fallback到local tensor→正常执行

Layer 3: Autograd Function 包装 (_api.py):
  → _ToTorchTensor: DTensor→local tensor → forward时redistribute到目标placement
  → _FromTorchTensor: local tensor→DTensor → backward时redistribute回原placement
  → ★ ★ 这些autograd Function确保梯度placement正确!

★ ★ Autograd流程:
  Forward: DTensor input → _ToTorchTensor → local tensor → op执行 → _FromTorchTensor → DTensor output
  Backward: DTensor grad_output → _ToTorchTensor → local grad → op backward → _FromTorchTensor → DTensor grad_input
  → ★ 每次DTensor→local转换都是autograd Function → 梯度自动正确!
```

## 2. DTensor → Local Tensor: _ToTorchTensor

```
★ ★ _ToTorchTensor (_api.py): DTensor→local tensor的autograd Function

Forward:
  → 输入: DTensor with placements (e.g. [Shard(0)])
  → redistribute到目标placement → full_tensor() if Replicate → shard提取 if Shard
  → 输出: local tensor → 给下游op用

★ ★ Backward:
  → 输入: local grad tensor → 来自下游op的梯度
  → _FromTorchTensor.backward() → redistribute回原DTensor placement
  → ★ gradient placement规则:
    Shard(0) → Shard(0) → 梯度保持分片 → 无需通信!
    Replicate → Replicate → 梯度保持复制 → 无需通信!
    Partial → Replicate → 梯度需要AllReduce → 从Partial变成Replicate!

★ ★ ★ Gradient Placement数学保证:
  1. Shard(dim) → Shard(dim): 梯度在dim维度保持分片 → 无通信 → 最高效!
  2. Replicate → Replicate: 梯度保持复制 → 无通信 → 但每rank有完整梯度
  3. Partial → Replicate: 梯度从Partial变成Replicate → 需要AllReduce → 通信!
  → ★ ★ FSDP2 backward ReduceScatter = 先AllReduce(Partial→Replicate) + 再ReduceScatter(Replicate→Shard)!
  → ★ ZeRO-3 backward = 直接ReduceScatter(各rank的Partial→各rank的Shard shard)
```

## 3. Redistribute — DTensor分布转换的autograd

```
★ ★ ★ Redistribute (_redistribute.py): DTensor placement转换的autograd Function

Forward:
  → 输入: DTensor with current_placements (e.g. [Shard(0)])
  → 目标: target_placements (e.g. [Replicate()])
  → 执行: redistribute(current→target) → AllGather/ReduceScatter/AllReduce
  → 输出: DTensor with target_placements

★ ★ Backward:
  → 输入: grad DTensor with target_placements
  → 执行: _redistribute_backward(grad, current_placements, target_placements)
  → 输出: grad DTensor with current_placements

★ ★ _redistribute_backward 核心逻辑:
  → gradient placement normalization → _normalize_placements_for_grad()
  → Replicate→Shard: ReduceScatter → 把复制梯度分片到各rank
  → Shard→Replicate: AllGather → 把分片梯度收集成完整
  → Partial→Replicate: AllReduce → 把部分梯度求和成完整
  → ★ ★ Partial normalization: backward中Partial→Replicate → 通信必须!

★ ★ NestedRedistribute (_redistribute.py): 二次可微!
  → Redistribute.forward的输出也是DTensor → 需要二次backward → NestedRedistribute
  → NestedRedistribute保存(ctx): current_placements, target_placements
  → backward时: 再次调用_redistribute_backward → 支持高阶梯度!
  → ★ 这是PyTorch autograd兼容的关键 → 否则二阶优化(如K-FAC)不支持!
```

## 4. Sharding Propagation — OpDispatcher

```
★ ★ ★ OpDispatcher (op.py): DTensor op的sharding规则引擎

__torch_dispatch__(func, types, args, kwargs):
  → 1. 检查输入是否是DTensor → 全部local→fallback
  → 2. 混合输入(DTensor+local) → local→replicate DTensor → 然后统一处理
  → 3. 全DTensor → sharding propagation → 输出DTensor

★ ★ Sharding propagation两种来源:
  1. register_sharding_propagation_rule(op, rule):
     → 用户自定义sharding规则 → 优先级最高
     → 规则: 输入placements → 输出placements
     → ★ FSDP2用此机制注册自定义规则!

  2. 默认einop_rule:
     → 自动推断输出sharding → 基于输入placements
     → Element-wise: 输出placement = 输入placement → 传播
     → Matmul: Shard(0)×Shard(1) → Partial → 需要AllReduce
     → ★ einop_rule是通用规则 → register_sharding_propagation_rule是专用规则

★ ★ DTensor dispatch完整流程:
  1. C++ fast path检查 → DTensor类型 → 快速路由
  2. OpDispatcher.__torch_dispatch__ → sharding propagation
  3. redistribute输入到统一placement → op执行
  4. 输出DTensor → 正确placement → 下一步继续
  → ★ ★ 整个过程: detect→propagate→redistribute→execute→output → 全自动!
```

## 5. FSDP2 + DTensor Autograd交互

```
★ ★ ★ FSDP2 = DTensor + RegisterPre/PostBackwardFunction → 自动通信!

FSDP2 Forward (flat_param.py):
  → RegisterPreBackwardFunction(all_gather_handle) → 注册在参数上
  → 参数是DTensor with Shard(0) → forward时需要AllGather→Replicate
  → ★ PreBackwardFunction: 下次backward开始时触发AllGather → 按需gather!

FSDP2 Backward:
  → 1. autograd触发 → RegisterPreBackwardFunction → AllGather(Shard→Replicate)
  → 2. backward计算 → 使用Replicate参数 → 正确梯度
  → 3. RegisterPostBackwardFunction → ReduceScatter(Replicate→Shard)
  → ★ PostBackwardFunction: backward结束后触发ReduceScatter → 释放参数!

★ ★ ★ FSDP2 autograd完整序列:
  Forward:
    → @contract(FSDPState) → MRO插入 → per-param DTensor(Shard(0))
    → RegisterPreBackwardFunction(AllGather handle) → 附加到每个参数
    → reshard_after_forward=True → 参数reshard回Shard → forward后立即释放!

  Backward:
    → autograd引擎触发 → RegisterPreBackwardFunction.pre_backward()
    → AllGather → 参数变成Replicate → backward使用完整参数
    → backward计算 → 梯度Partial → 需要normalize
    → RegisterPostBackwardFunction.post_backward()
    → ReduceScatter → 梯度从Replicate→Shard(0) → 参数shard
    → ★ ★ backward: AllGather(in) + ReduceScatter(out) = 2次通信!

★ ★ reshard_after_forward 3种模式:
  1. True: forward后reshard→Shard → 2次AllGather(forward+backward) → 省内存(2Ψ/module)
  2. False: forward后保持Replicate → 1次AllGather(只forward) → 省通信但peak高
  3. Int: 子组设置 → HSDP可per-module → 混合策略
  → ★ ★ True=省内存+2通信; False=省通信+1内存; 3-stream AllGather重叠通信!
```

## 6. ProcessGroupCollection — DeviceMesh PG管理

```
★ ★ ★ DeviceMesh._init_one_process_group (device_mesh.py): 每mesh维度创建PG

两种PG创建方式:
  1. ★ ★ split_group (ncclCommSplit):
     → 条件: bound_device_id存在 或 use_torchcomms=True
     → NCCL communicator已被eagerly initialized → 可split
     → 优势: 只需N次API调用(N=mesh维度数) → 2×4 mesh只需2次!
     → ncclCommSplit = 在已初始化communicator上split → 极快!
     → ★ 这是PyTorch 2.7+的新机制 → ProcessGroupCollection→split_group→0配置!

  2. ★ new_group (传统方式):
     → 条件: 无bound_device_id → NCCL未初始化
     → 每个子组一次new_group调用 → 2×4 mesh需要2×4=8次调用!
     → 更慢 → 但更灵活 → 支持不同backend

★ ★ ★ PG创建对比:
  → split_group: 1次/维度 → 极快 → ncclCommSplit → 无需额外初始化
  → new_group: N次/子组 → 较慢 → 需要独立初始化NCCL communicator
  → ★ ★ PyTorch 2.7 ProcessGroupCollection默认用split_group → 0配置!
  → ★ 2.8→零配置: DeviceMesh自动split_group → 用户不需要手动new_group!

★ ★ DeviceMesh._dim_group_names:
  → 每mesh维度一个PG name → mesh_dim_dp, mesh_dim_tp等
  → _pg_registry: {dim_name: ProcessGroup} → 查找PG
  → __getstate__: pop _pg_registry → pickle不保存PG
  → __setstate__: 从_dim_group_names重建_pg_registry → _resolve_process_group()
  → ★ ★ PG是进程本地资源 → 不能跨进程pickle → 必须重建!
```

## 7. CuTe Layout Algebra — _MeshLayout

```
★ ★ ★ _MeshLayout (device_mesh.py): CuTe-inspired layout algebra → rank映射!

核心概念:
  → _FlatLayout: 单维度layout → 描述一个mesh维度的rank排列
  → _MeshLayout: 多维度layout → list[_FlatLayout] → ND mesh描述
  → remap_to_tensor(rank_map): layout→tensor → 生成mesh tensor
  → collapse(): 多维→单维 → 用于验证和排序

★ ★ Layout操作:
  → _get_slice_mesh_layout(mesh_dim_names): 验证slice维度 → layout_sliced
  → check_sorted(): 排序检查 → dim indices必须升序 → CuTe约束
  → check_orthogonal(): 正交检查 → slice维度不能重叠 → 防止PG冲突
  → ★ ★ CuTe layout algebra = NVIDIA CUTLASS CuTe → 借鉴其tensor layout概念!

★ ★ vs 传统DeviceMesh:
  → 传统: mesh tensor直接存储 → reshape → 简单但不灵活
  → CuTe: _MeshLayout + rank_map → layout algebra → 更灵活
  → → 支持flatten/slice/remap → 不需要存储大tensor → 内存更少
  → ★ 2.7+用CuTe layout → 2.8→零配置 → DeviceMesh自动推断layout!

★ ★ from_group (device_mesh.py):
  → 从已有ProcessGroup构建DeviceMesh → 反向创建
  → 1D: 单PG → 1D mesh → 简单
  → ND: 多PG → ND mesh → mesh+mesh_dim_names必须提供
  → ★ 这是ProcessGroupCollection迁移的桥梁 → global→explicit!
```

## 8. DTensor C++ Implementation

```
★ ★ C++ DTensor fast path (python_variable.cpp):

torch._C._distributed:
  → Placement → C++ placement基类 → Python继承
  → Shard(dim) → extends torch._C._distributed.Shard → C++实现
  → Replicate() → extends torch._C._distributed.Replicate → C++实现
  → _Partial() → extends torch._C._distributed._Partial → C++实现

★ ★ C++ dispatch fast path:
  → DTensor.__torch_dispatch__ → 先C++检查 → 快速路由
  → 比纯Python快 → 减少每次op dispatch overhead
  → ★ 这是性能关键 → 训练中每个op都经过dispatch → overhead必须最小!

★ ★ DTensor属性:
  → _local_tensor: 本地分片的tensor → 实际数据
  → _placements: list[Placement] → 分布描述 → [Shard(0)]等
  → _device_mesh: DeviceMesh → mesh描述 → 哪些rank参与
  → ★ DTensor = local_tensor + placements + device_mesh → 声明式分布!
```

## 9. 关键设计洞察

```
1. DTensor autograd = autograd Function包装 → 梯度placement自动正确!
   → _ToTorchTensor/_FromTorchTensor → redistribute→local→op→local→redistribute→DTensor
   → 梯度placement规则: Shard→Shard/Replicate→Replicate/Partial→Replicate
   → ★ ★ 这是DTensor的核心设计 → autograd Function确保梯度placement → 无手动通信!

2. Redistribute autograd → DTensor placement转换 → 通信操作!
   → forward: redistribute(current→target) → AllGather/RS/AR
   → backward: _redistribute_backward(grad, target→current) → 反向通信
   → NestedRedistribute → 二次可微 → 高阶优化支持
   → ★ ★ DTensor placement转换 = 通信 → autograd自动插入 → 无手动AllGather!

3. FSDP2 = DTensor + Pre/PostBackwardFunction → 自动AllGather/ReduceScatter!
   → RegisterPreBackwardFunction → forward后注册 → backward时AllGather
   → RegisterPostBackwardFunction → backward后ReduceScatter
   → ★ ★ FSDP2通信完全自动 → 用户代码无手动通信 → 和ZeRO-3一样但更简洁!

4. split_group vs new_group → ProcessGroupCollection革命!
   → split_group: ncclCommSplit → 1次/维度 → 极快 → 2.7+默认
   → new_group: 独立初始化 → N次/子组 → 较慢 → 传统方式
   → ★ ★ PyTorch 2.8→零配置 → split_group→0配置 → DeviceMesh自动!

5. Gradient placement数学保证 → 无需手动通信!
   → Shard→Shard: 无通信 → 梯度保持分片 → 最高效
   → Replicate→Replicate: 无通信 → 梯度保持复制
   → Partial→Replicate: AllReduce → 从Partial变成Replicate → 必须通信
   → ★ ★ 这些规则是DTensor autograd的核心 → 数学保证 → 用户不操心!

6. Sharding propagation → register_sharding_propagation_rule → 自定义规则!
   → 默认: einop_rule → 自动推断 → 通用但非最优
   → 自定义: register_sharding_propagation_rule → 用户指定 → 更精确
   → ★ FSDP2用自定义规则 → 覆盖默认 → 更精确的sharding!

7. CuTe layout algebra → _MeshLayout → 更灵活的rank映射!
   → vs 传统: mesh tensor直接存储 → 不灵活
   → CuTe: layout algebra → flatten/slice/remap → 更灵活
   → ★ 借鉴NVIDIA CUTLASS CuTe → tensor layout概念 → 数学化!

8. C++ DTensor fast path → dispatch overhead最小!
   → C++ placement类型 → Python继承 → 快速路由
   → C++ dispatch检查 → 比Python快 → 性能关键
   → ★ ★ 训练中每个op都经过 → overhead必须最小 → C++ fast path必需!

9. RTX 4090 + DTensor/FSDP2:
   → PCIe → 无多GPU → DTensor Shard无意义(只有1rank)
   → 单GPU → FSDP2无意义 → 无分片 → 无通信 → ZeRO-3也无效
   → ★ ★ RTX 4090只能LoRA+CPU_Adam → DTensor/FSDP2/ZeRO-3都需要多GPU!
   → 但: DTensor概念理解 → 为多GPU集群做准备 → 知识储备!

10. DTensor vs ZeRO-3 data flow:
    → ZeRO-3: 手动AllGather→forward→释放→手动AllGather→backward→ReduceScatter
    → FSDP2/DTensor: 自动(Pre/PostBackwardFunction)→AllGather→forward→释放→AllGather→backward→ReduceScatter
    → ★ 两者data flow相同 → 但FSDP2/DTensor自动化 → compile兼容 → 未来!
    → ★ ZeRO-3: ds_tensor(pad) → 不compile兼容 → 被DTensor取代 → 趋势明确!
```

---

Sources:
- torch/distributed/tensor/_api.py (_ToTorchTensor, _FromTorchTensor, from_local)
- torch/distributed/tensor/_redistribute.py (Redistribute, NestedRedistribute, _redistribute_backward)
- torch/distributed/tensor/op.py (OpDispatcher, __torch_dispatch__)
- torch/distributed/device_mesh.py (_MeshLayout, _FlatLayout, _init_one_process_group, from_group)
- torch/distributed/tensor/api.py (register_sharding_propagation_rule)
- torch/_C/_distributed (C++ Placement, Shard, Replicate, _Partial)
- pytorch/torch/csrc/distributed/c10d/python_variable.cpp (C++ DTensor dispatch)
- torch/distributed/fsdp/flat_param.py (RegisterPreBackwardFunction, RegisterPostBackwardFunction)
- Background agent research (PyTorch DTensor autograd internals)
