# PyTorch FSDP2 (fully_shard) 源码级内部原理深度阅读

> 2026-06-15 | 源码: torch/distributed/fsdp/_fully_shard/ + torch/distributed/_composable/fsdp/
> 核心: fully_shard→@contract(FSDPState)+MRO insertion→FSDPParam per-parameter DTensor→AllGather(foreach_all_gather+copy_in/copy_out)→reshard_after_forward=True释放/False保留→backward ReduceScatter(foreach_reduce+dtype转换)→torch.compile兼容(UnspecializedNNModule+_dynamo_disable intentional graph breaks)→FSDP1 FlatParameter vs FSDP2 DTensor per-param

## 1. fully_shard() 实现 — 当你调用fully_shard(module)时

```
_fully_shard.py (98-296):

★ ★ @contract(state_cls=FSDPState)装饰器(97):
  → _composable framework → 为module创建FSDPState对象 → fully_shard.state(module)访问
  → 状态与module 1:1对应 → 插入module内部状态字典

关键流程:
  1. mesh初始化(242): mesh or _init_default_mesh() → 1D CUDA DeviceMesh
  2. mesh验证(243-244): _validate_mesh → FSDPMeshInfo(1D)/HSDPMeshInfo(2D)/DDPMeshInfo
  3. reshard_after_forward默认(246-265): None→True → lazy init时root模块覆盖为False
  4. 模块/参数收集(266-268): _get_modules_and_states() DFS → 收集可管理模块 → _move_states_to_device()
  5. State初始化(269-270): state.init(modules, device, mp_policy, auto_reshard_after_forward)
  6. 参数组初始化(272-285): _init_param_group() → FSDPParamGroup → 参数分片→DTensor
  7. Dynamo注解(288-290): _is_fsdp_managed_module=True + _fsdp_use_orig_params=True
  8. ★ ★ MRO类修改(293-295): _apply_to_module() → FSDP<OrigClass>(FSDPModule, OrigClass) → 替换module.__class__

★ ★ 参数→DTensor转换 (FSDPParam.__init__, _fsdp_param.py:192-232):

_init_sharded_param() (228-332):
  → Plain tensor: DTensorSpec(mesh, Shard(0)) 或 (Replicate(), Shard(0))
  → DTensor(TP): DP+TP mesh concatenate → _StridedShard interleaved sharding
  → DTensor(SPMD): 验证Replicate() on DP dims → 改为Shard(0)或_StridedShard
  → torch.chunk(param_data, shard_world_size, dim=shard_dim) → chunks[shard_rank]
  → ★ 不均匀分片: zero-pad到padded_sharded_size(304-311) → AllGather需要等量
  → CPU offload: 移动到CPU+可选pin_memory(312-315)
  → nn.Parameter(to_sharded_dtensor(sharded_local), requires_grad=...)
  → unsafe_setattr_param() → 直接写入module._parameters → 绕过nn.Module.__setattr__
```

## 2. FSDP2 Forward — Unshard/Reshard生命周期

```
★ ★ Forward路径: _pre_forward → unshard → wait → compute → _post_forward → reshard

Pre-forward (unshard):
  FSDPState._pre_forward() (_fsdp_state.py:158-189):
    → _lazy_init()(首次): 确定root模块, 设置reshard_after_forward=False for root, 共享comm_ctx
    → _root_pre_forward(): 等optimizer stream, 移动输入到device
    → _cast_forward_inputs(): 转换浮点输入到param_dtype
    → fsdp_param_group.pre_forward()

  FSDPParamGroup.pre_forward() (_fsdp_param_group.py:541-557):
    → TrainingState.FORWARD
    → self.unshard() → foreach_all_gather
    → self.wait_for_unshard() → copy到unsharded params
    → RegisterPostBackwardFunction autograd节点

  ★ ★ foreach_all_gather() (_fsdp_collectives.py:325-378):
    → all_gather_copy_in_stream: shard data→dtype转换→copy到AllGather输出buffer
    → 等copy_in_stream完成
    → all_gather_stream: all_gather_comm(output, input, group)
    → 返回AllGatherResult(output, event, work, dtypes, numels, split_sizes)

  wait_for_unshard() (418-490):
    → 等AllGather完成
    → foreach_all_gather_copy_out() → 拆分到每个param的all_gather_outputs
    → ★ init_unsharded_param():
      → Plain: torch.as_strided(all_gather_output, _orig_size, _contiguous_orig_stride) → ★ 视图无额外分配!
      → DTensor: _from_local_no_grad(unsharded_param, unsharded_dtensor_spec) → 正确placements
    → _to_unsharded() → unsafe_setattr_param() → 注册unsharded param到module

Post-forward (reshard):
  FSDPParamGroup.post_forward() (559-571):
    → self.reshard() → 释放unsharded/注册sharded(如果reshard_after_forward=True)
    → _record_post_forward() → 加入comm_ctx.post_forward_order → backward prefetching
```

## 3. reshard_after_forward — True vs False代码级详解

```
★ ★ ★ 3种reshard_after_forward → 根本不同的内存/通信行为!

reshard() (_fsdp_param_group.py:500-510):
  if not _reshard_after_forward: → return (no-op) → keep unsharded params!
  if _use_post_forward_mesh: → _to_sharded_post_forward() → reshard到子组
  else: → _to_sharded() → 完全reshard

★ ★ True: post_forward_mesh_info = mesh_info (完全reshard)
  → _to_sharded() (866-870):
    → fsdp_param.to_sharded() → setattr(sharded_param) + free_storage(unsharded→0)
    → ★ free_storage(): 将unsharded param的storage调整为0 → 释放GPU内存!
    → 下次unshard: alloc_storage()重新分配 → 不是malloc → reuse已分配的storage对象

★ ★ False: post_forward_mesh_info = None (no reshard)
  → reshard()提前return → unsharded params保持 → storage保持分配
  → backward unshard(): 检查is_unsharded → return (no-op) → 无AllGather!
  → 内存: 峰值=shard_size + full_unsharded_size (始终保留)

★ ★ Int: post_forward_mesh_info = smaller_mesh (部分reshard)
  → _to_sharded_post_forward() (872-876):
    → narrow到子shard: all_gather_outputs.narrow(0, sharded_numel*shard_rank, sharded_numel)
    → .clone() → 与AllGather输出分离 → 可释放AllGather输出
    → DTensorSpec(post_forward_mesh_info.mesh, (Replicate(), Shard(0)))
    → free_unsharded_param() → 释放完整AllGather输出
  → backward: AllGather从子shard→在较小子组上运行→更少通信!

★ 内存对比:
  → True: forward峰值=shard+unsharded(临时); forward后=shard仅; backward峰值=shard+unsharded(临时)
  → False: forward峰值=shard+unsharded; forward后=shard+unsharded(保留); backward=shard+unsharded(保留)
  → Int: forward峰值=shard+unsharded(临时); forward后=shard+子shard; backward=shard+子unsharded(子组)
  → ★ False省通信(1 AllGather)但内存高(始终保留full params)
  → ★ True省内存(释放)但通信多(2 AllGather)
  → ★ HSDP per-module设置: 中间层True(省内存), root False(省通信backward)

★ ★ Auto(default): 非root=True, root=False → 最优组合!
  → 非root: 释放→省内存→backward需要重新AllGather→但prefetch可以overlap
  → root: 保留→backward立即开始→无AllGather→省通信→但内存更高
```

## 4. FSDP2 Backward — Gradient ReduceScatter

```
★ ★ Backward路径: _pre_backward→unshard→compute→RegisterPostBackwardFunction.backward→post_backward→ReduceScatter

Pre-backward (unshard):
  FSDPState._register_pre_backward_hook() (_fsdp_state.py:229-249):
    → register_hook() on all gradient-receiving output tensors
    → functorch: RegisterPreBackwardFunction.apply() → autograd.Function backward调用

  FSDPState._pre_backward() (212-222):
    → TrainingState.PRE_BACKWARD
    → 注册root模块post-backward final callback(queue_callback)
    → fsdp_param_group.pre_backward()

  FSDPParamGroup.pre_backward() (_fsdp_param_group.py:581-593):
    → self.unshard() → if reshard_after_forward=True → 再次AllGather; False → no-op
    → self.wait_for_unshard()
    → _backward_prefetch() → 从post_forward_order预取下一个backward模块

★ ★ Post-backward (ReduceScatter):

  FSDPParamGroup.post_backward() (596-776) — 核心梯度处理!

  1. 梯度累积(612-614): accumulate_unsharded_grad_if_needed() → 加reduce_dtype累积grad
  2. 梯度提取(624-647): per FSDPParam:
     → unsharded_accumulated_grad / unsharded_param.grad / zero_grad
     → DTensor gradient redistribution: DP维度保持Partial→FSDP ReduceScatter处理→避免冗余all-reduce
  3. Reshard(648-649): if reshard_after_backward → reshard()
  4. Buffer回收(658-671): if buffer上限max_input_buffers → 等最旧RS完成→释放引用
  5. ★ ★ foreach_reduce() (_fsdp_collectives.py:522-546):
     → 梯度除法因子: ReduceScatter pre_div=1 post_div=1/world_size; HSDP pre=1/replicate_size post=1/shard_size
     → reduce_scatter_stream: torch.ops.fsdp.chunk_cat → concatenate grads → reduce_scatter_comm(output, input, group)
     → HSDP: all_reduce_stream上也all_reduce
     → CPU offload: RS输出→D2H→CPU→可选pin_memory()

  ★ Root模块post-backward final callback (_fsdp_state.py:125-147):
    → Variable._execution_engine.queue_callback() → 所有组post_backward()
    → finalize_backward() → 等post-reduce events → 清除RS state → 清除post_forward_order
```

## 5. FSDP2 + torch.compile集成

```
★ ★ ★ FSDP2 = torch.compile兼容 → FSDP1 = 不兼容 → 这是FSDP2核心优势!

Dynamo注解(_dynamo_utils.py):
  → _annotate_modules_for_dynamo(): _is_fsdp_managed_module=True + _fsdp_use_orig_params=True
  → Dynamo → UnspecializedNNModule → thorough guards → 参数作为每次forward输入 → 不捕获到compiled图
  → ★ 这对backward overlap关键: 参数视图每次forward重新创建 → backward hooks per-layer interleaved

★ ★ _dynamo_disable装饰器 (_fsdp_common.py:1-15):
  → torch._dynamo.disable(func, recursive=True, reason="skipping FSDP hooks")
  → 应用到所有FSDP hooks: _pre_forward, _post_forward, _pre_backward, reset_iter_state, lazy_init
  → ★ ★ Intentional graph breaks at FSDP module boundaries!
    → FSDP hooks → eager → comm ops(AllGather/RS) → 不compile
    → 模块内compute → compiled → Triton kernels → 最优
    → 这是想要的: comm=eager + compute=compiled → 安全+高效!

★ ★ FSDP1 FlatParameter vs FSDP2 original params:
  → FSDP1: FlatParameter → 捕获到compiled图 → static views → backward interleaving破坏
  → FSDP2: per-param DTensor → 不捕获 → views每次重建 → backward interleaving保持
  → 这是FSDP2+compile兼容的根本原因 → original params → dynamic → compile safe!
```

## 6. FSDP2 Mixed Precision

```
★ ★ MixedPrecisionPolicy (_fsdp_api.py:25-59):
  param_dtype: forward/backward compute dtype → AllGather在此dtype通信
  reduce_dtype: gradient ReduceScatter dtype → 如果≠param_dtype → grad cast
  output_dtype: forward output dtype
  cast_forward_inputs: True → 转换输入到param_dtype

★ ★ 关键: sharded params始终保持orig_dtype!
  → FSDPParam.init_dtype_attrs() (_fsdp_param.py:586-601):
    → orig_dtype = sharded_param.dtype → 保持原始dtype!
    → param_dtype只在AllGather时cast → sharded→param_dtype→AllGather→unsharded是param_dtype
  → AllGather输入: _to_dtype_if_needed(sharded_data, param_dtype) → cast到param_dtype
  → ReduceScatter前: to_accumulated_grad_if_needed() → grad.cast到reduce_dtype

★ ★ Optimizer step: sharded params在orig_dtype → FP32 shard→FP32 optimizer → 不需要loss scaling!
  → vs FSDP1: 需要loss scaling for FP16 → FSDP2 BF16不需要 → 更安全!

★ 推荐配置:
  → param_dtype=BF16 → compute BF16 → shard FP32 → optimizer FP32 → 最优!
  → reduce_dtype=FP32 → gradient reduction高精度 → 更稳定
  → ★ BF16 compute + FP32 optimizer = 无loss scaling + 高精度 = FSDP2推荐!
```

## 7. FSDP2 CPU Offload (Alpha)

```
★ CPUOffloadPolicy (_fsdp_api.py:88-99):
  pin_memory: bool = True → CPU tensor pin → H2D non_blocking overlap

★ ★ CPU offload数据流:
  1. Shard param存储在CPU(pin_memory) → size=param_size/world_size
  2. Pre-forward: H2D copy(to(device, non_blocking=True)) → pin_memory允许overlap
  3. AllGather: GPU上运行 → GPU output
  4. Forward: compute用GPU unsharded params
  5. Post-forward: 释放GPU unsharded → 重新注册CPU shard
  6. Backward: 再次H2D + GPU AllGather + GPU compute
  7. Post-backward: GPU ReduceScatter → D2H到CPU → grad在CPU
  8. Optimizer step: CPU上运行 → CPU optimizer states

★ ★ 内存节省: GPU内存 = AllGather临时 + 活跃compute层 → params+optim在CPU!
  → vs ZeRO-Offload: 类似但FSDP2 alpha → ~20-30%比FSDP1快 → 但~50-60%比pure GPU慢
  → pin_memory → H2D/D2H non_blocking → overlap with compute → 减少stall!
```

## 8. FSDP2 State Dict — DTensor Native

```
★ ★ FSDP2 state dict = DTensor native → 无需ShardedStateDictConfig!

FSDPParamGroup._register_state_dict_hooks() (_fsdp_param_group.py:924-946):
  → register_state_dict_pre_hook(to_sharded_hook) → save前确保sharded状态
  → _register_load_state_dict_pre_hook(to_sharded_hook) → load前确保sharded状态

★ Save: module.state_dict() → 每param是DTensor → _local_tensor = 本地shard → per-rank save
★ Load: 每rank加载本地shard → DTensor.from_local() → 恢复DTensor

★ ★ vs FSDP1:
  → FSDP1: FlatParameter + ShardedStateDictConfig + FullStateDictConfig → 复杂配置
  → FSDP2: DTensor._local_tensor = shard → standard state_dict → 简单自然!
  → 全参数gather: DTensor.full_tensor() → 所有shard gather → 完整参数 → rank0 only
  → 不需要FSDP1的config system → DTensor本身表达sharding → 更优雅!
```

## 9. FSDP2 vs FSDP1 内部原理对比

```
★ ★ ★ 10个关键代码级差异:

| Feature | FSDP1 | FSDP2 |
|---------|-------|-------|
| 参数表示 | FlatParameter(flatten所有params→1D) | per-param DTensor(independent) |
| 分片方式 | FlatParamHandle per unit | FSDPParam per parameter |
| Padding | FlatParameter整体pad→world_size倍 → 可能大量浪费 | per-param pad到shard slot → 极小浪费 |
| Hook机制 | FullyShardedDataParallel wrapper → override forward() | _composable @contract + MRO insertion |
| 状态管理 | _FSDPState attached to wrapper | FSDPState via _composable_state |
| 通信 | per FlatParamHandle AllGather/RS | per FSDPParamGroup foreach_all_gather/RS |
| State dict | ShardedStateDictConfig/FullStateDictConfig | DTensor native → _local_tensor |
| compile兼容 | ❌ FlatParameter捕获到图 → static views | ✅ original params → dynamic → UnspecializedNNModule |
| Backward | FlatParamHandle hooks | RegisterPostBackwardFunction + pre_backward hooks |
| Mixed precision | loss scaling for FP16 | BF16 orig_dtype + param_dtype cast → no loss scaling |

★ ★ Padding waste举例(8 GPU):
  → FSDP1: 7B模型flatten所有参数→1D→pad到8倍 → 可能浪费数MB
  → FSDP2: [4096,4096] weight → shard=[512,4096] → 无pad(均匀可分)
  → FSDP2: [7,4096] bias → shard=[1,4096](rank0-6) or [0](rank7) → pad仅7元素
  → ★ FSDP2 per-param sharding → 几乎无padding浪费 → 比FSDP1显著省内存!

★ ★ 通信效率:
  → FSDP1: 1次AllGather per FlatParamHandle → 整个flatten param → 大但少
  → FSDP2: 1次foreach_all_gather per FSDPParamGroup → 所有param shard concat → 1次通信但split回来
  → 两者通信量相同 → 但FSDP2 per-param → 更灵活(TP/EP mesh组合)
```

## 10. 关键设计洞察

```
1. MRO insertion vs wrapper module → 更透明!
   → FSDP1: FullyShardedDataParallel(module) → wrapper → 新类 → 丢失原始类
   → FSDP2: FSDP<OrigClass>(FSDPModule, OrigClass) → MRO → 继承原始类 → __getattr__透传
   → ★ 这是_composable framework的优势 → 修改类但不wrapper → 更少surprise

2. per-param DTensor vs FlatParameter → 更灵活+更少浪费!
   → FSDP1 flatten: cat所有params → 1D → pad → 不能per-param操作 → 不灵活
   → FSDP2 DTensor: 每param独立 → 可以TP+DP mesh组合 → _StridedShard → interleaved
   → Padding: per-param几乎无浪费 → FSDP1可能浪费数MB → 7B模型差异显著!

3. reshard_after_forward 3种模式 → 内存vs通信tradeoff!
   → True: 省内存(释放unsharded) 但2 AllGather → 最小GPU内存占用
   → False: 省通信(1 AllGather) 但保留full params → 最快backward
   → ★ Auto: 非root True + root False → 最优组合 → 中间层省内存 + root省通信
   → HSDP: per-module Int值 → 更细粒度控制 → 子组reshard → 灵活!

4. torch.compile intentional graph breaks → 安全边界!
   → FSDP hooks → _dynamo_disable → eager → AllGather/RS → 不compile
   → 模块内compute → compiled → Triton kernels → 最优
   → ★ 这不是bug → 是intentional design → comm ops不应被Triton kernel替换!
   → +15-16% vs eager / +48-50% with Float8 → 最佳训练优化组合!

5. DTensor native state dict → 简单自然!
   → save: DTensor._local_tensor → per-rank shard → torch.save
   → load: per-rank load → DTensor.from_local() → 恢复DTensor
   → ★ 无需ShardedStateDictConfig/FullStateDictConfig → DTensor本身表达sharding!
   → vs FSDP1: 复杂config → 3种StateDictType → 容易出错 → FSDP2更安全!

6. AllGather copy_in/copy_out → 多stream overlap!
   → copy_in_stream: dtype转换+buffer准备 → CPU→GPU copy → non_blocking
   → all_gather_stream: 实际AllGather通信 → GPU collective
   → copy_out_stream: 拆分到per-param outputs → as_strided(view) → 无额外分配!
   → ★ 3 stream pipeline → copy_in→all_gather→copy_out → overlap!

7. ReduceScatter buffer回收 → 避免内存峰值!
   → max_input_buffers → 限制in-flight RS数量 → 等最旧完成→释放 → O(1) buffers
   → vs 无回收: 每层1个RS buffer → 层数×buffer大小 → 内存峰值高!
   → ★ 这个设计类似DeepSpeed ZeRO-3的max_ongoing_backpressure(2 events) → 共同pattern!

8. BF16 orig_dtype + param_dtype cast → 无loss scaling → 更安全!
   → shard params FP32 → optimizer FP32 → 不需要loss scaling
   → AllGather: cast到BF16 → compute BF16 → 高效
   → ReduceScatter: grad FP32 → reduction FP32 → 精确
   → ★ vs FSDP1 FP16: loss scaling → 不稳定 → FSDP2 BF16 → 简单稳定!

9. FSDPParam状态机 → SHARDED/UNSHARDED/SHARDED_POST_FORWARD
   → 状态决定行为 → unshard/reshard/grad处理 → 状态驱动 → 清晰!
   → vs FSDP1: FlatParamHandle Free/Unshard/Auto → 类似但flatten粒度不同

10. backward prefetch → 从post_forward_order预取下一层!
    → 记录forward order → backward逆序 → prefetch下一backward模块的AllGather
    → ★ overlap: 当前层backward compute + 下一层AllGather → 减少stall!
    → 与DeepSpeed ZeRO-3 trace-based prefetch类似 → 但FSDP2用forward order → 更简单
```

---

Sources:
- torch/distributed/fsdp/_fully_shard/_fully_shard.py (fully_shard function + FSDPModule)
- torch/distributed/fsdp/_fully_shard/_fsdp_init.py (mesh validation + module collection)
- torch/distributed/fsdp/_fully_shard/_fsdp_param.py (FSDPParam + DTensor sharding + state machine)
- torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py (FSDPParamGroup + forward/backward lifecycle)
- torch/distributed/fsdp/_fully_shard/_fsdp_state.py (FSDPState + hooks + lazy init)
- torch/distributed/fsdp/_fully_shard/_fsdp_api.py (MixedPrecisionPolicy + CPUOffloadPolicy)
- torch/distributed/fsdp/_fully_shard/_fsdp_common.py (MeshInfo + TrainingState)
- torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py (AllGather + ReduceScatter)
- torch/distributed/fsdp/_dynamo_utils.py (Dynamo integration)
- Background agent research (PyTorch FSDP2 internals)
