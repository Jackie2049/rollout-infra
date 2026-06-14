# DeepSpeed PartitionedParameterCoordinator 源码深度阅读

> 2026-06-15 | 源码: deepspeed/runtime/zero/partitioned_param_coordinator.py (634行)
> 核心: 3阶段trace机制(RECORD→COMPLETE→INVALID) + prefetch队列 + max_reuse_distance释放策略 + NVMe swap预取
> 与FSDP2对比: ZeRO-3用trace驱动精确prefetch; FSDP2用basic prefetch(无trace)

## 1. 3阶段Trace机制 (ZeRoTraceMode)

```python
class ZeRoTraceMode(Enum):
    RECORD = 1    # 第一次forward+backward → 记录模块执行顺序
    COMPLETE = 2  # 后续 → 使用记录的顺序 → 精确prefetch
    INVALID = 3   # 记录不匹配 → 重新trace
```

### 1.1 状态转换

```
首次迭代: INVALID → RECORD → (完成forward+backward) → COMPLETE
后续迭代: COMPLETE → (如果模块顺序匹配) → COMPLETE
           COMPLETE → (如果模块顺序不匹配) → INVALID → 下一轮 → RECORD
```

### 1.2 RECORD阶段: 记录模块执行顺序

```python
def record_module(self, sub_module):
    """adds sub module to trace"""
    self.__submodule_order.append(sub_module)
    self.__step_id_module_fetched_for[sub_module.ds_id].append(self.__step_id)

def record_parameters(self, sub_module):
    """adds parameters to trace"""
    step_id = self.__step_id_module_fetched_for[sub_module.ds_id].popleft()
    for param in sorted(set(iter_params(sub_module)), key=lambda p: p.ds_id):
        self.__param_order.append(__ParamInTrace(
            param=param,
            step_id_last_used_at=step_id  # ← 记录此参数最后在哪个step使用
        ))
```

**关键**: `step_id_last_used_at` 记录每个参数最后被哪个模块使用 → 用于prefetch计算

### 1.3 COMPLETE阶段: 校验模块顺序

```python
def trace_prologue(self, sub_module):
    if self.is_complete_trace():
        # 模块顺序必须匹配 → 否则invalidate
        if sub_module != self.__submodule_order[self.__step_id]:
            self._invalidate_trace()  # → 重新trace
```

**为什么需要invalidate?**
- 动态模型(如条件分支) → 模块执行顺序可能变化
- evaluate vs train → 模块可能不同
- 一旦invalidate → 下一轮RECORD → 重新建立trace

### 1.4 reset_step: 完成一个forward+backward

```python
def reset_step(self):
    """indicate that we have completed one fwd+bwd for the model"""
    if self.is_record_trace():
        # 第一次 → 构建参数trace → freeze
        self.construct_parameter_trace_from_module_trace()
        # 所有rank验证顺序一致性!
        assert_ints_same_as_other_ranks([m.ds_id for m in self.__submodule_order])
        assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
        self.__submodule_order = tuple(self.__submodule_order)  # freeze → tuple不可变
        self.__param_order = tuple(self.__param_order)          # freeze
        self.__trace_mode = ZeRoTraceMode.COMPLETE
    else:
        self.__trace_mode = ZeRoTraceMode.RECORD  # enable recording for next pass

    self.__param_queue = collections.deque(self.__param_order)  # reset fetch queue
    self.__step_id = 0
    self.__n_available_params = 0
```

**关键**: 所有rank必须顺序一致 → 否则prefetch会不同步!

## 2. fetch_sub_module: 核心fetch+prefetch+wait流程

```python
def fetch_sub_module(self, current_submodule, forward):
    """3步流程:
    1. AllGather当前模块的参数 (立即需要)
    2. Prefetch后续模块的参数 (提前获取)
    3. Wait当前模块参数完成 (阻塞直到可用)
    """
```

### 2.1 步骤1: AllGather当前模块参数

```python
params_to_fetch = set(iter_params(current_submodule))
fetch_numel = sum([p.partition_numel() for p in params_to_fetch
                   if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])

if fetch_numel > 0:
    self.__all_gather_params(params_to_fetch, forward)
    # → 内部: AllGatherCoalescedHandle → 多参数合并一次AllGather
```

### 2.2 步骤2: Wait参数可用

```python
for param in params_to_fetch:
    param.ds_active_sub_modules.add(current_submodule.ds_id)
    if param in self.__inflight_param_registry:
        # 在allgather_stream上等待
        with get_accelerator().stream(self.__allgather_stream):
            # Backpressure控制: 限制并发fetch事件数
            while self.__ongoing_fetch_events and self.__ongoing_fetch_events[0].query():
                self.__ongoing_fetch_events.popleft()
            if len(self.__ongoing_fetch_events) > self.__max_ongoing_fetch_events:
                self.__ongoing_fetch_events.popleft().synchronize()
            self.__inflight_param_registry.pop(param).wait(handle_dependency=not fast_fetch)
```

**Backpressure机制**: `__max_ongoing_fetch_events = 2`
- 限制同时inflight的AllGather事件数 → 防止内存压力过大
- 每个fetch分配buffer → 太多同时 → GPU内存爆!
- 类似TCP拥塞控制 → 简单但有效

### 2.3 步骤3: Prefetch后续参数

```python
if self.is_complete_trace():
    # 从prefetch queue中弹出当前模块参数(不再需要prefetch)
    discarded_from_prefetch_queue = set()
    params_not_already_fetched = set(
        filter(lambda p: self.__most_recent_step_id_param_fetched_for[p] < self.__step_id,
               params_to_fetch))
    while self.__param_queue and len(discarded_from_prefetch_queue) < len(params_not_already_fetched):
        param_in_trace = self.__param_queue.popleft()
        discarded_from_prefetch_queue.add(param_in_trace.param)

    # Prefetch下一批参数
    if self.__prefetch_bucket_sz > 0:
        max_params_to_prefetch = min(
            self.__max_n_available_params - self.__n_available_params,
            self.__prefetch_bucket_sz  # ← prefetch桶大小限制
        )
        params_to_prefetch = set()
        numel_prefetching = 0
        while self.__param_queue and numel_prefetching < max_params_to_prefetch:
            param_in_trace = self.__param_queue.popleft()
            do_prefetch = param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            if do_prefetch:
                params_to_prefetch.add(param_in_trace.param)
                numel_prefetching += param_in_trace.param.ds_numel

        if numel_prefetching > 0:
            self.__all_gather_params(params_to_prefetch, forward)
```

**关键**:
- `prefetch_bucket_sz`: 控制prefetch参数总量 → 防止prefetch占用太多内存
- `max_n_available_params - n_available_params`: 当前可用参数的剩余空间
- 参数从`__param_queue`按顺序弹出 → 确保prefetch顺序与执行顺序一致

## 3. release_sub_module: 参数释放策略

```python
def release_sub_module(self, submodule, forward=False):
    params_to_release = self.__params_to_release(submodule, self.__step_id)
    # ↑ 有trace: 只释放满足条件的参数
    # 无trace: 释放所有参数

    for param in iter_params(submodule):
        param.ds_active_sub_modules.discard(submodule.ds_id)
        if param.ds_id in params_to_release:
            self.__release_param(param)
```

### 3.1 __params_to_release: 精确释放决策

```python
@functools.lru_cache(maxsize=None)
def __params_to_release(self, submodule_to_release, step_id):
    params_to_release = set(p.ds_id for p in iter_params(submodule)
                           if not p.ds_persist)  # 持久化参数不释放

    # 条件1: 如果prefetch已跳过此参数的后续使用 → 不释放
    for param in iter_params(submodule):
        if self.__most_recent_step_id_param_fetched_for[param] > step_id:
            params_to_release.discard(param.ds_id)

    # 条件2: 如果参数在max_reuse_dist内会再次使用 → 不释放
    params_traversed = 0
    for module in self.__submodule_order[step_id:]:
        if params_traversed >= self.__max_reuse_dist_in_numel:
            break
        for param in iter_params(module):
            params_to_release.discard(param.ds_id)
            params_traversed += param.ds_numel

    return params_to_release
```

**`max_reuse_distance_in_numel`**: 核心参数!
- 控制参数释放的激进程度
- 小值 → 激进释放(更多内存节省, 但更多AllGather)
- 大值 → 保守释放(更少AllGather, 但更多内存占用)
- 类似CPU cache的替换策略: LRU vs 预测式释放

**lru_cache**: 缓存释放决策 → 相同(submodule, step_id) → 不重复计算!

## 4. __all_gather_params: AllGather实现细节

```python
def __all_gather_params(self, params, forward):
    # 分组: quantized vs non-quantized
    quantized_params = [p for p in params if hasattr(p.ds_tensor, 'ds_quant_scale')]
    nonquantized_params = [p for p in params if not hasattr(p.ds_tensor, 'ds_quant_scale')]
    # 量化参数 → 单独AllGather(解量化需要)
    # 非量化参数 → 合并AllGather(coalesced)

def __all_gather_params_(self, params, forward, quantize=False):
    partitioned_params = [p for p in params if p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
    if partitioned_params:
        self.__n_available_params += all_gather_numel
        # 分组: 有secondary_tensor vs 无
        # secondary_tensor = 双分片的第二份 → 不需AllGather!
        with get_accelerator().stream(self.__allgather_stream):
            handle = partitioned_params[0].all_gather_coalesced(param_group, quantize=quantize)
        for param in param_group:
            self.__inflight_param_registry[param] = handle
```

**secondary_tensor**: ZeRO优化 → 小参数(如bias)双份 → 一个常驻GPU → 不需AllGather!
- `ds_persist = True` → 持久化在GPU → 不释放 → 节省AllGather开销
- 只有大参数才需要AllGather → 进一步减少通信

## 5. NVMe Prefetch (异步磁盘预取)

```python
def __prefetch_nvme_param_partitions(self):
    """在GPU prefetch的同时, 从NVMe预取下一批参数的分区"""
    if not self.is_complete_trace():
        return

    numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)
    swap_in_params = []
    for param_in_trace in self.__param_queue:
        param = param_in_trace.param
        if (numel_considered > 2 * numel_in_flight
                or len(swap_in_params) >= param.nvme_swapper.available_swap_in_buffers()):
            break
        if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
            swap_in_params.append(param)
        numel_considered += param.ds_numel

    if swap_in_params:
        swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)
```

**3级Prefetch**:
```
1. GPU prefetch: AllGather下一层参数 → overlap计算
2. NVMe→CPU prefetch: swap_in参数分区 → overlap GPU计算+AllGather
3. CPU→GPU AllGather: GPU prefetch等待 → 但NVMe预取已准备好
```

**`2 * numel_in_flight`**: 预取距离 = 2倍当前inflight → 平衡提前量和内存

## 6. 线程安全: Leaf Module Backward

```python
def fetch_sub_module(self, current_submodule, forward):
    # Backward时leaf module可能被多线程并发触发
    is_leaf = z3_leaf_module(current_submodule)
    needs_sync = is_leaf and not forward  # 只backward需要同步

    if needs_sync:
        with self.__leaf_module_lock:
            event = self.__ongoing_fetch_leaf_module_events.get(current_submodule.ds_id)
            if event is not None:
                event_to_wait = event  # 其他线程正在fetch → 等它完成
            else:
                new_event = threading.Event()
                self.__ongoing_fetch_leaf_module_events[current_submodule.ds_id] = new_event

        if event_to_wait is not None:
            event_to_wait.wait()  # 阻塞直到fetch完成
            return  # 不重复fetch!

    # ... fetch完成后 ...
    if needs_sync:
        event = self.__ongoing_fetch_leaf_module_events.pop(current_submodule.ds_id)
        event.set()  # 通知等待的线程
```

**问题**: 为什么backward需要线程安全?
- `torch.autograd` 在backward时可能多线程执行hooks
- 当一个module返回多个tensor → 每个tensor的grad_fn可能在不同线程执行
- → 多个线程同时触发同一个leaf module的pre-backward hook
- → 需要同步! 否则参数状态会冲突

## 7. 与FSDP2 Prefetch对比

| 方面 | DeepSpeed ZeRO-3 | FSDP2 |
|------|-----------------|-------|
| Trace机制 | 3阶段(RECORD→COMPLETE→INVALID) | 无trace |
| Prefetch驱动 | 记录的参数顺序 → 精确prefetch | 简单: 下一层FSDPParamGroup |
| Prefetch粒度 | prefetch_bucket_sz → 可配置桶大小 | fixed: 下一组参数 |
| 释放策略 | max_reuse_distance → 智能释放 | 固定: forward后立即reshard |
| Backpressure | max_ongoing_fetch_events=2 → 限并发 | 无explicit限制 |
| NVMe支持 | swap_in async预取 → 3级pipeline | 无NVMe |
| 线程安全 | leaf module backward threading lock | 无(单线程backward) |
| 参数持久化 | ds_persist → 小参数常驻GPU | 无(所有参数同等对待) |
| 量化兼容 | quantized vs non-quantized分组AllGather | 无量化 |
| lru_cache | 释放决策缓存 → 不重复计算 | 无(固定策略) |

**关键差异**:
- ZeRO-3: **trace驱动** → 动态自适应 → 条件分支模型也能处理(INVALID→重新trace)
- FSDP2: **固定策略** → 下一层 → 简单但不够灵活 → 条件分支可能prefetch不需要的参数

## 8. RTX 4090影响

```
ZeRO-3 prefetch在RTX 4090单GPU:
  - 单GPU无DP → prefetch无意义(没有其他rank的参数要提前gather)
  - ZeRO-3单GPUpeak = 16Ψ/N+4Ψ = 280GB → 完全不可能
  - → RTX 4090: 用verl GRPO代替, 不用ZeRO-3

FSDP2 prefetch在RTX 4090单GPU:
  - 单GPU: FSDP2也不分片(N=1) → prefetch无意义
  - FSDP2单GPUpeak = 同ZeRO-3 → 也不行
  - → RTX 4090: LoRA+FSDP2+CPU Adam → 20.04GB → 可行!

多GPU(8×4090)时:
  - ZeRO-3 N=8: 16Ψ/8=2Ψ+4Ψ → peak=84GB → 仍超24GB单卡!
  - FSDP2 N=8: 同 → 但overlap更好
  - PCIe scaling灾难 → 实际通信远慢于NVLink环境
  - → 8×4090: 实际不可行(PCIe瓶颈)
```

## 9. 关键配置参数

| 参数 | 含义 | 推荐值 | 影响 |
|------|------|--------|------|
| prefetch_bucket_sz | prefetch桶大小(参数数) | 5e7-1e8 | 大→prefetch更多→overlap更好但内存多 |
| max_reuse_distance_in_numel | 参数释放最大reuse距离 | 1e8-1e9 | 大→保留更多→少AllGather但内存多 |
| max_available_parameters_in_numel | 最大可用参数总数 | 取决于GPU内存 | 限制prefetch总量→防止OOM |
| max_ongoing_fetch_events | 并发fetch事件限制 | 2 | 防止内存压力→类似拥塞控制 |
| prefetch_nvme | 是否NVMe预取 | True(有NVMe时) | 3级pipeline→磁盘→CPU→GPU |

## 10. 下一步

- [ ] 研究 ZeRO-Infinity NVMe swap完整数据流(AsyncPartitionedParameterSwapper)
- [ ] 研究 AllGatherCoalescedHandle内部实现(coalesced buffer管理)
- [ ] 对比 ZeRO-3 trace vs Megatron 1F1B调度 vs vLLM scheduler的三种"预知"机制
- [ ] 在GPU可用时实测ZeRO-3 prefetch性能 vs FSDP2 basic prefetch
