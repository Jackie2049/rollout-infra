# PyTorch torch.distributed Internals Deep Dive — ProcessGroupNCCL + DDP Reducer + FSDP Lifecycle + Backend Architecture

> 2026-06-12 | PyTorch分布式训练全链路源码分析: ProcessGroupNCCL(ncclComm生命周期+WorkNCCL异步模型+Watchdog线程+CUDA Stream管理+grouped ops+ncclCommSplit)+DDP Reducer(梯度分桶+autograd_hook+flat buffer+通信计算重叠)+FSDP2(参数分片生命周期+unshard/reshard+param.data swap)+ProcessGroup抽象接口+后端注册
> 源码: torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp (~1400行), ProcessGroupNCCL.cpp (~3500+行), reducer.h/reducer.cpp, torch/distributed/fsdp2/_fsdp_param_group.py
> 关联: distributed-systems-deep-dive.md, zero-fsdp-distributed-training-deep-dive.md, nccl-multi-gpu-benchmark-rtx4090.md

## 0. 核心定律: 4层架构 + 流式异步 + 分桶重叠 + 分片生命周期

```
PyTorch torch.distributed 4层架构:

Layer 1: ProcessGroup (抽象接口)
  → c10d/ProcessGroup.hpp → 定义collective+p2p操作契约
  → Backend类 → 工厂注册(Gloo/NCCL/MPI/自定义)
  → → init_process_group(backend="nccl") → 查找注册的creator → 创建ProcessGroupNCCL

Layer 2: ProcessGroupNCCL (NCCL后端实现)
  → 56KB头文件 + 3500+行cpp → 最复杂最关键的后端!
  → ncclComm按需创建+缓存 → devNCCLCommMap_[deviceKey] → 一次创建复用
  → CUDA Stream per-device → ncclStreams_[deviceKey] → 专用NCCL流
  → CUDA Event → ncclEvents_[deviceKey] → 流间同步(double-barrier)
  → WorkNCCL → 异步Work对象 → 封装ncclResult_t+CUDA event → wait()/isCompleted()
  → Watchdog线程 → 监控超时 → ncclCommAbort → 3级错误处理(NoHandling/TearDown/CleanUpOnly/SkipCleanUp)
  → HeartbeatMonitor → 监控Watchdog线程心跳 → 防止Watchdog本身hang
  → ncclGroupStart/End → coalescing → 多个collective合并一次proxy setup
  → ncclCommSplit → 子组通信器 → split(新PG) → 不需要重新ncclCommInitRank!

Layer 3: DDP Reducer (梯度同步引擎)
  → reducer.cpp → C++核心 → 梯度分桶+autograd_hook+flat buffer
  → bucket_cap_mb=25 → 默认25MB桶大小 → 按参数大小逆序分组
  → autograd_hook → backward时触发 → mark_variable_ready → mark_bucket_ready → allreduce
  → 通信-计算重叠 → 后层梯度先ready → 先allreduce → 前层backward继续 → overlap!
  → flat buffer → 单连续tensor → 单次allreduce → 避免多次小通信

Layer 4: FSDP (参数分片训练)
  → FSDPParamGroup → per-parameter sharding → 不再FlatParameter!
  → 生命周期: shard→unshard(all_gather)→compute→reshard(free)→backward→reduce_scatter
  → reshard_after_forward 3策略: NO_SHARD/SHARD_GRAD_OP/FULL_SHARD
  → param.data swap → Python层面直接swap → 无额外内存 → 最快

关键设计:
  → ProcessGroupNCCL: 流式异步模型 → 不阻塞主线程 → overlap通信+计算
  → DDP: 分桶重叠 → 后层先allreduce → 前层继续backward → 利用反向传播天然顺序
  → FSDP: 分片生命周期 → 用时gather→不用时free → 最小化GPU内存占用
  → Watchdog+HeartbeatMonitor: 双层监控 → Watchdog监控Work超时 → HeartbeatMonitor监控Watchdog
```

## 1. ProcessGroup 抽象接口 — 后端注册契约

```
文件: torch/csrc/distributed/c10d/ProcessGroup.hpp

ProcessGroup抽象类:
  → 纯虚方法 → collective操作(allreduce/allgather/broadcast/reduce_scatter/...) + p2p(send/recv)
  → 返回Work对象 → 异步操作 → wait()/isCompleted()/getFuture()
  → Options → Backend::Options → 后端配置(timeout/is_high_priority_stream等)

Backend注册:
  → 工厂模式 → static std::map<std::string, ProcessGroupCreator> backendCreators
  → → init_process_group(backend="nccl") → 查找"nccl"creator → 创建ProcessGroupNCCL
  → 内置后端:
    → Gloo: ProcessGroupGloo → CPU+GPU → 注册为"gloo"
    → NCCL: ProcessGroupNCCL → GPU专用 → 注册为"nccl" → 最重要!
    → MPI: ProcessGroupMPI → MPI封装 → 注册为"mpi" → 需USE_MPI=1
  → 自定义后端:
    → 实现ProcessGroup接口 → 编译为.so → torch.ops.load_library()
    → Backend.register_backend("mybackend", creator) → Python注册

Python级调用链:
  init_process_group(backend="nccl")
    → Backend(value="nccl") → 查找注册表 → ProcessGroupNCCL
    → store=TCPStore → rendezvous → rank/size协商
    → ProcessGroupNCCL(store, rank, size, options) → C++构造
```

## 2. ProcessGroupNCCL — NCCL后端核心实现

```
文件: torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp (~1400行)
      torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp (~3500+行)

### 2.1 ncclComm生命周期

devNCCLCommMap_: unordered_map<string, shared_ptr<NCCLComm>>
  → key=deviceKey → 设备标识字符串 → "0" 或 "0,1,2,3,4,5,6,7"
  → value=shared_ptr<NCCLComm> → ncclComm_t封装 → 缓存复用!

生命周期:
  1. 构造时不创建ncclComm! → lazy initialization → 按需创建
  2. 首次collective → initNCCLComm(deviceKey, device, opType)
    → broadcastUniqueNCCLID → rank 0生成ncclUniqueId → Store广播到所有rank
    → ncclCommInitRank → NCCL创建通信器 → 所有rank同步!
    → 缓存到devNCCLCommMap_ → 后续相同deviceKey复用
  3. split → ncclCommSplit → 从parent通信器创建子组 → 不需要重新init!
  4. 错误 → ncclCommAbort → 强制终止 → 通信器进入error state → 不可复用
  5. 正常关闭 → shutdown() → ncclCommDestroy → 清理所有缓存通信器

关键设计:
  → 按需创建 → 不浪费资源 → 只有实际用到的设备才有通信器
  → 缓存复用 → 同deviceKey不重复创建 → 节省ncclCommInitRank开销(秒级!)
  → split/merge → ncclCommSplit/ncclCommMerge → 子组创建更快(毫秒级)
  → → 支持: supportsSplitting()=true, supportsCoalescing()=true
  → suspend/resume → NCCL 2.29.7+ → 内存卸载 → vLLM sleep/wake!
```

### 2.2 WorkNCCL — 异步Work对象

```
class WorkNCCL : public Work:

核心属性:
  → device_: at::Device → 操作所在设备
  → ncclComm_: shared_ptr<NCCLComm> → 使用的NCCL通信器
  → ncclStartEvent_: shared_ptr<CUDAEvent> → NCCL操作开始事件(可选)
  → ncclEndEvent_: shared_ptr<CUDAEvent> → NCCL操作结束事件
  → seq_: uint64_t → 操作序列号(全局唯一)
  → opTimeout_: chrono::milliseconds → 操作超时(默认10分钟)
  → workStartTime_: chrono::time_point → 操作开始时间
  → outputs_: shared_ptr<vector<Tensor>> → 输出tensor引用
  → stashed_for_allocator_safety_: shared_ptr<TensorShelf> → 缓存分配器安全!
  → future_: intrusive_ptr<Future> → Python级Future对象
  → blockingWait_: bool → 阻塞等待模式

状态检查:
  → startedGPUExecutionInternal() → ncclStartEvent是否完成 → GPU是否开始执行
  → finishedGPUExecutionInternal() → ncclEndEvent是否完成 → GPU是否完成执行
  → isCompleted() → finishedGPUExecution + 无异常 → 全部完成
  → checkTimeout() → 检查是否超时 → 超时设exception

关键方法:
  → wait() → 阻塞等待完成 → 或timeout异常
  → getFuture() → 返回Future → Python级异步等待
  → result() → 返回输出tensor → 需wait后调用
  → checkAndSetException() → 检查NCCL错误 → 设exception_ptr
  → setException(exception_ptr) → 外部设异常

设计亮点:
  → stashed_for_allocator_safety_ → 避免recordStream → 保持tensor引用直到流同步
  → → TORCH_NCCL_AVOID_RECORD_STREAMS → 替代recordStream → 更安全!
  → CUDAEvent → GPU事件 → 无CPU轮询 → 精确同步
  → future → 与Python异步框架桥接 → torch.futures兼容
```

### 2.3 CUDA Stream管理 — 流式异步模型

```
ncclStreams_: unordered_map<string, at::cuda::CUDAStream>
ncclEvents_: unordered_map<string, at::cuda::CUDAEvent>

Double-Barrier事件模型:

  User Compute Stream (default)
    → [用户计算] → recordEvent(compute_done) → [等待nccl完成]
    → → compute_done事件 → 表示输入tensor已准备好

  NCCL Stream (per device)
    → waitEvent(compute_done) → [NCCL collective] → recordEvent(nccl_done)
    → → nccl_done事件 → 表示NCCL输出tensor已准备好

  User Compute Stream (default) 继续
    → waitEvent(nccl_done) → [使用输出tensor]

正确性保证:
  → compute→NCCL→compute → 两道屏障 → 保证tensor就绪 → 无竞态!
  → → 但NCCL流和计算流可以并行 → overlap通信+计算!

is_high_priority_stream:
  → TORCH_NCCL_HIGH_PRIORITY → NCCL流用高优先级 → 减少延迟
  → → 但可能抢占计算流 → 需权衡!
  → → 生产: 大部分不用 → 默认优先级足够

CUDAEventCache:
  → TORCH_NCCL_CUDA_EVENT_CACHE → 缓存CUDAEvent → 避免频繁创建销毁
  → → cudaEvent创建需要CUDA global lock → 可能hang → 缓存避免!
```

### 2.4 Watchdog线程 — 超时监控+错误处理

```
class Watchdog:
  → ncclCommWatchdogThread_: std::thread → 后台监控线程
  → pg_: ProcessGroupNCCL* → 原指针(线程在PG生命周期内)
  → heartbeat_: atomic_uint64_t → 心跳计数 → HeartbeatMonitor检查
  → workMetaListCV_: condition_variable → 等待条件
  → rethrowCUDAErrors_: bool → 是否重新抛出CUDA错误
  → propagatePgError_: bool → 是否通过TCPStore传播错误到所有rank
  → desyncDebugger_: DesyncDebugger → 反同步调试器

Watchdog.runLoop()核心逻辑:
  while (!terminateProcessGroup_):
    1. 检查completedWorkList → 清理已完成Work → 移出workMetaList
    2. 检查所有WorkNCCL → checkTimeout() → 超时→设exception
    3. 检查ncclComm错误 → checkForNCCLErrors() → NCCL internal error
    4. 如果发现错误:
      → ErrorHandlingMode决定行为:
        → NoHandling: 不处理 → 忽略错误
        → TearDown: 终止进程 → std::abort → 最激进!
        → CleanUpOnly: ncclCommAbort → 清理通信器 → 不终止进程
        → SkipCleanUp: 不清理 → 直接终止 → ncclCommAbort本身hang时用
    5. 传播错误 → TCPStore写入signal → 其他rank也能检测
    6. 更新heartbeat_ → HeartbeatMonitor知道Watchdog还活着
    7. sleep → workMetaListCV_.wait_for → 等待新Work或超时

ErrorHandlingMode 4级:
  → 默认: SkipCleanUp(3) → 保守 → 防止ncclCommAbort本身hang
  → 生产推荐: CleanUpOnly(2) → 清理但不终止 → 可以恢复
  → → 但恢复需要重新ncclCommInitRank → 秒级 → 可能不划算
```

### 2.5 HeartbeatMonitor — 监控Watchdog的Watchdog

```
class HeartbeatMonitor:
  → ncclHeartbeatMonitorThread_: std::thread → 第二层监控线程
  → pg_: ProcessGroupNCCL* → 原指针
  → heartbeatTimeoutInSec_: int → 心跳超时秒数
  → terminateHeartbeatMonitorThread_: atomic<bool> → 终止标志

runLoop()逻辑:
  while (!terminateHeartbeatMonitorThread_):
    1. 检查Watchdog的heartbeat_ → 上次更新时间
    2. 如果超过heartbeatTimeoutInSec_ → Watchdog可能hang!
    3. dumpDebuggingInfo() → 飞行记录器+stack trace → 保存调试信息
    4. terminateProcess() → std::abort → 终止整个进程!
    5. sleep → coordCheckIntervalMilSec → 定期检查

双层监控设计:
  → Watchdog → 监控WorkNCCL超时+NCCL错误 → 第一层
  → HeartbeatMonitor → 监控Watchdog心跳 → 第二层 → 防Watchdog本身hang
  → → Watchdog可能hang → 调用cudaEventElapsedTime → 需CUDA global lock → 可能死锁!
  → → → HeartbeatMonitor → 不调用CUDA API → 只检查atomic heartbeat → 安全!
  → → → → HeartbeatMonitor终止进程 → 调试信息已保存 → 可事后分析

配置:
  → TORCH_NCCL_ENABLE_MONITORING → 启用HeartbeatMonitor
  → TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC → 心跳超时(默认300秒)
  → TORCH_NCCL_ASYNC_ERROR_HANDLING → 启用异步错误处理
```

### 2.6 ncclGroupStart/End — Coalescing合并操作

```
Coalescing机制:

startCoalescing():
  → coalescing_state_ += 1 → 进入coalescing模式
  → ncclGroupStart() → NCCL开始合并模式

endCoalescing() → endCoalescing(OpType):
  → coalescing_state_ -= 1 → 退出coalescing模式
  → ncclGroupEnd() → NCCL结束合并 → 执行所有合并操作!
  → → 返回单个WorkNCCL → 包含所有合并操作 → 单次proxy setup!

关键设计:
  → ncclActiveGroupCounter_: thread_local uint64_t → 嵌套coalescing支持
  → coalescedDevice_: at::Device → 合并操作的设备
  → coalescedComm_: shared_ptr<NCCLComm> → 合并操作使用的通信器
  → coalescedTensors_: TensorShelf → 合并操作的输入输出tensor暂存
  → → endCoalescing时 → 转移到WorkNCCL.stashed_for_allocator_safety_

用途:
  → FSDP all_gather + reduce_scatter → 合并 → 单次proxy → 减少延迟
  → 多个小allreduce → 合并 → 单次大allreduce → 更高效
  → → 但! 合合操作不overlap → 一次执行 → 可能不如分桶overlap

RTX 4090:
  → PCIe带宽低 → 合合操作可能更优 → 减少proxy setup次数
  → 但大合并 → 不overlap → 需权衡!
```

### 2.7 ncclCommSplit — 子组通信器

```
split(store, ranks, opts):
  → opts.split_from → parent ProcessGroupNCCL → 从哪个PG分出来
  → opts.split_color → 分组颜色 → NCCL_SPLIT_NOCOLOR(-1)=不在组
  → ncclCommSplit(parent_ncclComm, color, &new_ncclComm)
  → → color相同的rank在同一子组 → 新ncclComm只包含这些rank
  → → 不需要ncclCommInitRank! → 从parent直接split → 毫秒级!

merge(store, opts, rank, size):
  → 合并子组 → ncclCommMerge → 多个split出来的子组重新合并
  → → NCCL 2.21.5+ → ncclCommCreateFromRanks → 更高效

shrink(ranks_to_exclude, shrink_flags, opts):
  → 缩减 → 排除指定rank → 动态缩容
  → → NCCL_HAS_COMM_SHRINK → ncclCommShrink → 弹性训练!

设计意义:
  → MoE Expert Parallel → split → 每组expert独立通信 → 不需要全局AllReduce
  → Pipeline Parallel → split → 每stage独立通信 → P2P更高效
  → Elastic Training → shrink/merge → 动态调整GPU数量 → 容错+弹性

RTX 4090:
  → 8 GPU → split为2组4GPU → 每组独立AllReduce → 通信量减半!
  → → 但! PCIe无P2P → split性能提升有限 → 全局AllReduce已经慢
```

## 3. DDP Reducer — 梯度分桶同步引擎

```
文件: torch/csrc/distributed/c10d/reducer.h + reducer.cpp

### 3.1 Reducer核心架构

Reducer在DDP.__init__()中创建:
  → _c10dReducer = Reducer(params, process_group, bucket_cap_mb)
  → 核心职责: 梯度分桶 + autograd_hook + flat buffer + allreduce调度

核心数据结构:
  → buckets_: vector<Bucket> → 桶列表(逆序!后层先)
  → → Bucket: variables(list) + flat_buffer(Tensor) + offset_per_var + num_gradients_not_ready(atomic)
  → variable_locators_: vector<VariableLocator> → 每参数→(bucket_idx, intra_bucket_idx)
  → grad_accumulators_: vector<Tensor> → 每参数的梯度累加器
  → next_bucket_index_: int → 下一个要allreduce的桶(贪心调度)
  → num_gradients_ready_: atomic<int> → 已ready的梯度数
  → has_rebuilt_bucket_: bool → 是否重建过桶(动态图模式)

### 3.2 initialize_buckets() — 桶构建

算法:
  1. 按参数大小降序排列 → 大参数优先 → 桶更均匀
  2. 逆序 → 后层参数在前面 → backward先算后层 → 先ready!
  3. 每桶总大小≤bucket_cap_mb(25MB) → 控制allreduce粒度
  4. 计算flat_buffer大小 → 每桶一个连续tensor → 单次allreduce
  5. 计算offset → 每参数在flat_buffer中的偏移 → copy用

关键设计:
  → 逆序 → Layer6→bucket0, Layer5→bucket0, Layer4→bucket1...
  → → backward从Layer6开始 → bucket0先ready → 先allreduce → overlap!
  → → → 顺序allreduce → Layer1→bucket0 → backward最后才ready → 无overlap!
  → bucket_cap_mb → 大→少allreduce但少overlap / 小→多overlap但多allreduce
  → → 生产: 25MB → 经验最优 → 大模型可能需要调整
  → bucket_cap_mb=0 → 单桶 → 等backward全部完成再allreduce → 无overlap → 最慢!

### 3.3 autograd_hook → mark_variable_ready → mark_bucket_ready

autograd_hook(variable_index):
  → 注册在每个参数的AccumulateGrad上
  → backward计算该参数梯度 → hook触发
  → → 调用mark_variable_ready(variable_index)

mark_variable_ready(variable_index):
  1. 获取locator → variable_locators_[variable_index] → (bucket_idx, intra_bucket_idx)
  2. 复制梯度到flat_buffer → flat_buffer[offset:end].copy_(grad.view(-1))
  3. 桶计数器减1 → num_gradients_not_ready -= 1 (atomic)
  4. 清空原始grad → grad.zero_() → 防止重复累加
  5. 如果num_gradients_not_ready == 0 → mark_bucket_ready(bucket_idx)

mark_bucket_ready(bucket_index):
  1. 记录已ready → ready_bucket_indices_.push(bucket_index)
  2. 贪心调度 → 检查是否可以立即allreduce
  3. → 调用allreduce(bucket.flat_buffer) → 异步!
  4. → 记录Work → 等完成 → 复制回grad

贪心调度逻辑:
  → 不等所有桶ready → 桶ready就allreduce → overlap!
  → → bucket0 ready → 立即allreduce → 同时bucket1仍在计算
  → → → 后层梯度: ready快→allreduce快 → 前层梯度: 正在计算→不等待
  → → → → 整体时间 = max(backward_time, allreduce_time) → overlap!

### 3.4 Flat Buffer设计

flat_buffer:
  → 单连续tensor → 包含桶内所有参数梯度
  → → 7B模型: 每桶25MB → 单次allreduce 25MB → 比250次100KB allreduce更快!
  → → PCIe: 25MB allreduce → ~10ms → vs 250×100KB → ~250×1ms=250ms → 25x更快!

内存布局:
  flat_buffer = [param0_grad | param1_grad | param2_grad | ...]
  → offset[i] = sum(param_size[0:i]) → 每参数偏移
  → → view(-1) → 展平 → 连续复制 → 最快

生命周期:
  → 初始化 → 分配flat_buffer → 每迭代
  → backward → grad复制到flat_buffer → 桶ready
  → allreduce → flat_buffer → AllReduce → 平均梯度
  → 复制回 → flat_buffer[offset:end] → param.grad → 更新参数
  → → 全部在同一tensor上 → 无额外内存分配 → 最省!

### 3.5 通信-计算重叠时间线

```
时间线(逆序桶):

Backward Layer6 → grad6 ready → bucket0 allreduce开始!
Backward Layer5 → grad5 ready → bucket0 already allreducing
Backward Layer4 → grad4 ready → bucket1 allreduce开始!
Backward Layer3 → grad3 ready → bucket1 already allreducing
Backward Layer2 → grad2 ready → bucket2 allreduce开始!
Backward Layer1 → grad1 ready → bucket2 already allreducing

同时:
  GPU计算: backward Layer4,3,2,1... (继续!)
  GPU通信: allreduce bucket0,1,2... (并行!)
  → → overlap! → 总时间 = max(compute, comm) → 不是sum!

对比无overlap:
  Backward全部完成 → 所有allreduce → 总时间 = compute + comm → 更慢!

RTX 4090:
  → PCIe allreduce慢 → 25MB ~10ms → vs NVLink ~1ms
  → → overlap更重要! → 减少总时间 → 但仍有瓶颈
  → → → bucket_cap_mb调小 → 更overlap → 但更多allreduce → 需benchmark!
```

### 3.6 静态图 vs 动态图

```
静态图(find_unused_parameters=False + static_graph):
  → initialize_buckets只调用一次 → 不重建 → 梯度顺序不变
  → → 每迭代同一桶结构 → 同一allreduce顺序 → 最优overlap!
  → → 大部分模型是静态图 → 生产推荐!

动态图(find_unused_parameters=True):
  → 每迭代重建桶 → 参数可能变化 → 桶结构不稳定
  → → unused参数 → 梯度不ready → 桶不完成 → DDP等待 → 可能hang!
  → → → find_unused_parameters=True → 检测unused → 跳过 → 不hang
  → → → 但开销! → 每迭代重建桶 → 每迭代reinitialize → 省不了!

Communication Hook:
  → DDPCommunicationHook → 自定义allreduce行为
  → → FP16CompressHook → 梯度FP16压缩 → 通信量减半 → 但精度损失
  → → PowerSGDDHook → 低秩压缩 → 通信量更大缩减 → 但需额外计算
  → → → RTX 4090 PCIe慢 → FP16Compress → 25MB→12.5MB → allreduce快2x!
```

## 4. FSDP2 — 参数分片生命周期

```
文件: torch/distributed/fsdp2/_fsdp_param_group.py

### 4.1 FSDPParamGroup — 核心管理类

关键变化(FSDP1→FSDP2):
  → FSDP1: FlatParameter → 多参数合并→一个flat tensor → 复杂!
  → FSDP2: per-parameter sharding → 每参数独立 → 更灵活!

FSDPParamGroup核心:
  → _sharded_params: list[_ShardedParam] → 分片参数列表
  → process_group: ProcessGroup → 通信组
  → device: Device → GPU设备
  → reshard_after_forward: ReshardAfterForward → 重分片策略

_ShardedParam:
  → orig_param: Parameter → 原始未分片参数
  → sharded_data: Tensor → 本rank的分片(只存1/N)
  → unsharded_data: Tensor → 全参数(all_gather结果)
  → _shard_state: ShardState → SHARD/UNSHARD → 当前状态
  → _unshard_state: UnshardState → NOT_UNSHARDED/UNSHARDING/UNSHARDED → 生命周期

### 4.2 参数生命周期 — Forward

```
Forward前: pre_forward_unshard()
  → 检查_unshard_state → NOT_UNSHARDED → 需要all_gather
  → all_gather(sharded_data) → 重建完整参数 → 各rank同步
  → → process_group.allgather → WorkNCCL → 异步!
  → param.data = unsharded_data → Python直接swap → 无额外内存!
  → → → orig_param.data → sharded_data → unsharded_data → 指针swap!
  → _unshard_state = UNSHARDED → 标记完成

Forward计算:
  → 使用完整参数 → 正常forward → 无分片开销

Forward后: post_forward_reshard()
  → 检查reshard_after_forward策略:
    → NO_SHARD: 保持unsharded → 不释放 → 内存高但backward不用再gather!
    → SHARD_GRAD_OP: reshard → 释放 → backward时再unshard → 标准FSDP
    → FULL_SHARD: reshard → 释放 → backward也要unshard → 最省内存!
  → reshard → param.data = sharded_data → swap回去 → 释放unsharded_data
  → _unshard_state = NOT_UNSHARDED → 标记sharded
```

### 4.3 参数生命周期 — Backward

```
Backward前: pre_backward_unshard()
  → 同forward前 → all_gather → param.data = unsharded_data
  → → 如果NO_SHARD → 已经unsharded → 不需要再gather → 最快!

Backward计算:
  → 使用完整参数 → 正常backward → 计算梯度

Backward后: post_backward_reshard()
  → reduce_scatter(grad) → 梯度分片 → 每rank只存自己的1/N梯度
  → → process_group.reduce_scatter → WorkNCCL → 异步!
  → reshard → param.data = sharded_data → swap回去
  → _unshard_state = NOT_UNSHARDED → 标记sharded
```

### 4.4 3种Reshard策略

```
NO_SHARD (不分片=DDP):
  → forward前: 不all_gather(已完整) → forward计算
  → forward后: 不reshard(保持完整)
  → backward前: 不all_gather(已完整) → backward计算
  → backward后: allreduce梯度 → 每rank完整梯度
  → → 内存: 每rank存完整模型 → 和DDP一样!
  → → 适用: 小模型 → 内存够 → 通信最少

SHARD_GRAD_OP (标准FSDP):
  → forward前: all_gather → forward
  → forward后: reshard → 释放
  → backward前: all_gather → backward
  → backward后: reduce_scatter → 分片梯度
  → → 内存: 每rank只存1/N参数+1/N梯度 → 标准ZeRO-3
  → → 通信: 2次all_gather+1次reduce_scatter → 标准开销
  → → 适用: 大模型 → 内存不够 → 标准选择

FULL_SHARD (最省内存):
  → 同SHARD_GRAD_OP → 但优化器状态也分片
  → → 每rank只存1/N参数+1/N梯度+1/N优化器状态 → ZeRO-3!
  → → 内存: 1/N of full → 最省!
  → → 适用: 极大模型 → 70B+ → 必须分片

RTX 4090 FSDP策略选择:
  → 7B BF16 → 14GB参数 → 单GPU 24GB → 不需要FSDP!
  → → NO_SHARD → DDP → 最快 → 通信最少
  → 但! FSDP+FSDP offload → CPU存优化器状态 → GPU省56GB → 有意义!
  → → 7B ZeRO-2 → GPU存参数+梯度 → CPU存优化器 → 最优!
```

### 4.5 FSDP通信-计算重叠

```
FSDP overlap设计:
  → all_gather异步 → WorkNCCL → 不阻塞backward
  → → forward时: all_gather(param) → 同时计算前面的层 → overlap!
  → → backward时: all_gather(param) → 同时计算后面的层 → overlap!

但FSDP比DDP更难overlap:
  → DDP → backward时梯度逐层ready → 自然overlap → 分桶即可
  → FSDP → forward/backward前需要all_gather → 必须等完成才能计算 → 不overlap!
  → → → 解决: prefetch → 预取下一层的参数 → 当前层计算时gather下一层 → overlap!
  → → → → FSDP2: _pre_forward_hook → 可以指定prefetch顺序 → 手动overlap!

RTX 4090:
  → PCIe all_gather慢 → 7B all_gather ~1.5s → 不能overlap → 灾难性!
  → → → FSDP 8GPU: 0.46x → 比单GPU慢2x → PCIe瓶颈!
  → → → → RTX 4090: 单GPU最优 → FSDP不适用 → 但CPU offload有意义!
```

## 5. ProcessGroupNCCL关键环境变量

```
错误处理:
  → TORCH_NCCL_ASYNC_ERROR_HANDLING → 启用异步错误处理
  → TORCH_NCCL_BLOCKING_WAIT → 阻塞等待模式(调试用)
  → TORCH_NCCL_PROPAGATE_ERROR → 传播错误到所有rank(TCPStore)
  → TORCH_NCCL_DESYNC_DEBUG → 反同步调试(找超时根因)

监控:
  → TORCH_NCCL_ENABLE_MONITORING → 启用HeartbeatMonitor
  → TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC → 心跳超时(默认300s)
  → TORCH_NCCL_DUMP_ON_TIMEOUT → 超时时dump调试信息

性能:
  → TORCH_NCCL_HIGH_PRIORITY → NCCL流高优先级
  → TORCH_NCCL_AVOID_RECORD_STREAMS → 避免recordStream → 用stashed_ref
  → TORCH_NCCL_ENABLE_TIMING → 启用collective计时
  → TORCH_NCCL_TRACE_BUFFER_SIZE → 飞行记录器buffer大小
  → TORCH_NCCL_CUDA_EVENT_CACHE → CUDAEvent缓存
  → TORCH_NCCL_NAN_CHECK → NaN检查(慢但安全)

调试:
  → TORCH_NCCL_RANKS_PER_ROOT → ncclCommInit时每root覆盖rank数
  → NCCL_DEBUG=INFO → NCCL日志 → 详细通信信息
  → NCCL_DEBUG_SUBSYSTEM=ALL → 所有子系统日志
```

## 6. RTX 4090 torch.distributed Implications

```
1. ProcessGroupNCCL对RTX 4090:
  → 8×4090 → devNCCLCommMap_["0,1,2,3,4,5,6,7"] → 单通信器覆盖所有GPU
  → → PCIe无P2P → NCCL必须走PCIe → AllReduce慢2.76GB/s → vs NVLink 9.2GB/s
  → → → ncclCommInitRank → 秒级 → 但只初始化一次 → 可接受
  → → → → single GPU最优 → 不需要跨GPU通信器!

2. DDP Reducer对RTX 4090:
  → bucket_cap_mb=25 → 25MB AllReduce → PCIe ~10ms → 算大!
  → → 7B backward ~200ms → allreduce ~80ms(4桶×20ms) → 不完全overlap!
  → → → 调小bucket → 更多overlap → 但更多allreduce → 需benchmark
  → → → → 实测: 8GPU DDP=0.46x → 通信瓶颈太严重 → 不如单GPU!

3. FSDP对RTX 4090:
  → all_gather → 7B/8=~1.75GB per rank → PCIe ~650ms → 灾难!
  → → reduce_scatter → 同样650ms → 每步1.3s通信 → 不overlap!
  → → → FSDP 8GPU=0.46x → 和DDP一样慢 → PCIe瓶颈!
  → → → → RTX 4090: 单GPU+CPU offload → 更优! → 不跨GPU通信

4. ncclCommSplit对RTX 4090:
  → MoE Expert Parallel → split → 2组4GPU → 组内AllReduce → 通信量减半
  → → 但4GPU PCIe AllReduce仍然慢 → 不如单GPU!
  → → → RTX 4090: MoE 7B → 单GPU足够 → 不需要EP!

5. Watchdog对RTX 4090:
  → 8GPU训练 → 任一GPU NCCL错误 → Watchdog检测 → abort → 所有GPU终止
  → → PCIe不稳定 → 可能NCCL hang → Watchdog保护 → 防止无限等待
  → → → 生产: TORCH_NCCL_ASYNC_ERROR_HANDLING+TORCH_NCCL_ENABLE_MONITORING → 必须!

6. 生产最优分布式配置:
  → RTX 4090单GPU → NO_SHARD → DDP → 不跨GPU → 最快
  → → CPU Adam offload → GPU省56GB → 7B训练24GB可fit!
  → → → ZeRO-2 + offload → 最优配置 → 不牺牲速度
  → → → → 不用FSDP跨GPU → 不用AllReduce → 单GPU训练!
  → → → → → 但! 数据并行需要多GPU → 8GPU异步→独立batch→梯度CPU同步?
  → → → → → → 实测: 8GPU DDP=0.46x → PCIe灾难 → 不值得!
  → → → → → → → 结论: RTX 4090训练=单GPU+offload → 或者NVLink GPU(A100/H100)
```

## 参考文献

```
1. PyTorch torch.distributed 源码:
   - torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp (~1400行) — ProcessGroupNCCL类声明
   - torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp (~3500+行) — ProcessGroupNCCL实现
   - torch/csrc/distributed/c10d/ProcessGroup.hpp — ProcessGroup抽象接口
   - torch/csrc/distributed/c10d/reducer.h + reducer.cpp — DDP Reducer
   - torch/nn/parallel/distributed.py — DDP Python wrapper
   - torch/distributed/fsdp2/_fsdp_param_group.py — FSDP2核心

2. NCCL API:
   - ncclCommInitRank → 通信器初始化
   - ncclCommSplit → 子组创建
   - ncclCommAbort → 错误终止
   - ncclGroupStart/End → 合合操作
   - NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/

3. PyTorch Blog:
   - "Deep Dive into ProcessGroupNCCL" — 官方详解
   - "FSDP Internals" — FSDP2生命周期

4. 我们的笔记:
   - distributed-systems-deep-dive.md — 分布式系统理论
   - zero-fsdp-distributed-training-deep-dive.md — ZeRO/FSDP算法
   - nccl-multi-gpu-benchmark-rtx4090.md — NCCL实测数据
   - fsdp2-scaling-benchmark-rtx4090.md — FSDP2实测数据