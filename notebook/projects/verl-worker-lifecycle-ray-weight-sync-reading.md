# verl Worker 生命周期与 Ray 分布式架构源码阅读

> 2026-06-15 | 源码: verl/single_controller/ + verl/workers/engine_workers.py + verl/checkpoint_engine/ + verl/workers/rollout/
> 核心: WorkerGroup=Ray actor管理+Dispatch路由 → RolloutMode(HYBRID/COLOCATED/STANDALONE) → Sleep/Wake GPU内存切换 → 3种Weight Sync后端(naive/NCCL/NIXL)

## 1. WorkerGroup 类层次

```
WorkerGroup (base/worker_group.py:123-255)
  ├── _workers: list[worker_handle] → Ray actor引用
  ├── _worker_names: list[str] → worker标识
  ├── _dispatch_info / _collect_info: dict → lazy dispatch注册
  ├── world_size = len(self._workers)
  ├── _bind_worker_method(): 扫描@register装饰的方法 → 动态绑定到WorkerGroup

RayWorkerGroup (ray/base.py:416-909)
  ├── _init_with_resource_pool(): 创建Ray PG → 每bundle 1 Ray actor
  ├── execute_all() / execute_all_async() / execute_all_sync()
  ├── execute_rank_zero() / execute_rank_zero_async()
  ├── spawn(prefix_set): 从现有workers创建detached子WorkerGroup
  ├── spawn_fused(prefix_set): 为FusedWorker创建WorkerGroup(colocated)
  ├── fuse(prefix_set): 合并多个WorkerGroup → ColocateWorkerGroup
```

### Worker创建流程

```python
# RayWorkerGroup._init_with_resource_pool()
for bundle in placement_group.bundles:
    worker = RayClassWithInitArgs.__call__()  # 创建Ray actor
    # num_gpus = 1 / resource_pool.max_colocate_count  → 分数GPU for colocation!
    # max_colocate_count=3 → 每actor得1/3 GPU → 3个WorkerGroup共享同一GPU
```

## 2. Dispatch 路由机制

### Dispatch 模式枚举

```python
# decorator.py:26-47
Dispatch.register("RANK_ZERO")        # 只有rank 0执行
Dispatch.register("ONE_TO_ALL")       # 相同参数广播到所有workers
Dispatch.register("ALL_TO_ALL")       # 参数直接传递(已per-worker)
Dispatch.register("DP_COMPUTE")       # 参数已按DP rank分好; 输出per-rank
Dispatch.register("DP_COMPUTE_PROTO") # 拆分DataProto by DP size, 自动padding, concat结果
Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC") # 拆分DataProto, 传function到每个worker
Dispatch.register("DP_COMPUTE_METRIC") # 拆分DataProto, 收集per-rank(不concat)
Dispatch.register("DIRECT_ROLLOUT_METHOD") # 禁止! 必须通过rollout replica调用
```

### @register 装饰器工作流

```python
# decorator.py:398-444
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=True)
def generate_sequences(self, data: DataProto):
    ...

# _bind_worker_method()处理:
# 1. 读取MAGIC_ATTR → {dispatch_mode, execute_mode, blocking}
# 2. 从DISPATCH_MODE_FN_REGISTRY获取 dispatch_fn + collect_fn
# 3. 选择execute_fn (execute_all / execute_rank_zero)
# 4. 创建Functor: dispatch_fn(args) → execute_fn(remote_call) → collect_fn(output)
```

### 关键Dispatch实现

```
ONE_TO_ALL (行120-123):
  args = tuple([arg] * world_size for arg in args) → 所有worker相同参数
  → 用于: init_model, reset, save_checkpoint (所有DP worker做相同事)

DP_COMPUTE (行148-156):
  args必须已是list[world_size] → 每worker拿对应slice
  → 用于: 已经按DP rank组织的数据

DP_COMPUTE_PROTO (行167-199):
  DataProto.split(world_size) → 自动padding确保均匀 → 每worker1块
  collect: _concat_data_proto_or_future → 合并回完整DataProto
  → 用于: generate_sequences, update_policy, compute_log_prob (最常用!)

ND Compute (行266-304):
  make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")
  → 查询worker_group._dispatch_info[mesh_name]
  → 根据dp_rank_mapping映射DataProto到正确worker
  → 支持非均匀DP-TP-PP拓扑!
```

### engine_workers.py中的Dispatch使用

| 方法 | Dispatch模式 | 原因 |
|------|-------------|------|
| `init_model` | ONE_TO_ALL | 所有worker初始化相同模型 |
| `compute_log_prob` | ND_compute(mesh="actor") | 按actor DP拓扑分数据 |
| `update_actor` | ND_compute(mesh="actor") | 按actor DP拓扑 |
| `compute_ref_log_prob` | ND_compute(mesh="ref") | 按ref DP拓扑 |
| `train_mini_batch` | ND_compute(mesh="train") | 按训练引擎DP拓扑 |
| `save/load_checkpoint` | ONE_TO_ALL | 所有worker保存/加载相同 |

## 3. RolloutMode: 3种部署模式

```python
# replica.py:54-68
class RolloutMode(Enum):
    HYBRID = "hybrid"       # Rollout+Training 同进程
    COLOCATED = "colocated"  # Rollout+Training 同PG但不同进程
    STANDALONE = "standalone" # 分离GPU资源, disaggregated
```

### HYBRID模式 (init_hybrid, 行131)

```
特征: Rollout引擎和FSDP训练引擎共享同一Ray actor/进程
  → FusedWorker包装actor+rollout进1个进程per GPU

GPU内存管理:
  → Sleep/Wake必需: rollout必须sleep(释放GPU) → training使用GPU → wake_up恢复
  → sleep_level=2(full weights) 或 sleep_level=1(LoRA adapters)

Weight Sync:
  → in-process → naive/ColocatedCheckpointEngine → Python generator!
  → self.actor.engine.get_per_tensor_param() → generator of (name, tensor)
  → yield from self.weights → 同进程零拷贝!

适用: on-policy PPO训练 (最高效, 最简单)

RTX 4090:
  → HYBRID最优! 单进程→无IPC/Ray overhead
  → LoRA→sleep_level=1→只释放KV cache→更快wake_up
  → GRPO→无critic→actor+rollout共用GPU→HYBRID完美!
```

### COLOCATED模式 (init_colocated, 行160)

```
特征: Rollout和训练引擎共享同一Ray Placement Group(同GPU)但不同进程
  → CheckpointEngineWorker作为独立Ray actor per GPU

GPU内存管理:
  → sleep_level=1 only (只释放KV cache → 为NCCL weight buffer腾空间)
  → 不需要sleep_level=2! (rollout进程保留weights → NCCL直接写入)

Weight Sync:
  → CUDA IPC → BucketedWeightSender/Receiver → ZMQ socket + torch.multiprocessing IPC
  → 流程: CheckpointEngine进程 → ZMQ IPC → vLLM进程 → GPU tensor直传
  → bucket_size=2048MB → 多个weight打包成1个bucket → 减少传输次数

适用: GRM(LLM as judge), rollout需要独立进程但同GPU

RTX 4090:
  → 可行但不如HYBRID→IPC开销(虽然同一GPU但跨进程)
  → 适合需要vLLM独立进程的场景(如MoE模型)
```

### STANDALONE模式 (init_standalone, 行189)

```
特征: Rollout有自己的ResourcePool → 独立GPU集
  → 新ResourcePoolManager → 独立Placement Groups

GPU内存管理:
  → 无Sleep/Wake! (rollout有自己的GPU → 不需要释放)

Weight Sync:
  → NCCL/NIXL checkpoint engine → 跨GPU集传输
  → NCCL: 建立ProcessGroup → AllReduce/Broadcast weight buckets
  → NIXL: RDMA P2P ring → 双buffer(send_buf+recv_buf) → swap per bucket

适用: off-policy训练, disaggregated架构(推理和训练分离GPU)

RTX 4090:
  → 不适用! 需要至少2×GPU集 → PCIe带宽不够NCCL weight sync
  → NVLink集群→可行但不如HYBRID高效
```

## 4. Sleep/Wake机制详解

### VLLM_SLEEP_LEVEL

```python
# third_party/vllm/__init__.py
VLLM_SLEEP_LEVEL = 1  # 默认
# vLLM >= 0.8.5: VLLM_SLEEP_LEVEL = 2
# Ascend NPU: 强制=1 (sleep_level=2不支持)
```

```
Level 1: 释放KV cache和部分模型权重 → LoRA-safe → 只释放adapter?
  → 保留模型config和部分metadata → wake_up更快

Level 2: 释放所有(KV cache + 全部模型权重 + 更多)
  → 完全清空GPU内存 → training可以全量使用 → wake_up需要重新加载weights
  → HYBRID full weight sync必须level=2 → LoRA mode只需level=1
```

### HYBRID sleep/wake流程

```python
# vllm_async_server.py:937-955
async def _sleep_hybrid(self):
    if self.lora_as_adapter or is_torch_npu_available():
        sleep_level = 1  # LoRA只需level 1 → 不释放base weights
    else:
        sleep_level = 2  # Full weight sync需要level 2 → 完全清空
    await self.engine.sleep(level=sleep_level)

# vllm_async_server.py:604-623
async def wake_up(self):
    # HYBRID: wake_up(tags=["kv_cache", "weights"]) → 恢复全部GPU内存
    # COLOCATED: wake_up(tags=["kv_cache", "weights"]) → weights已通过IPC更新
    # STANDALONE: skip → 不需要wake_up(有自己的GPU)
```

### 完整Rollout-Training切换时序

```
HYBRID模式 (一个step的GPU内存生命周期):

  1. Rollout阶段: vLLM awake → 占据全部GPU(weights+KV cache)
     → generate_sequences → rollout完成 → 收集responses

  2. sleep(level=1 or 2):
     → vLLM释放GPU内存 → FSDP训练引擎可以使用!
     → Level 1: 释放KV cache → FSDP可以做forward+backward(但weights仍在!)
     → Level 2: 释放KV cache+weights → FSDP独占全部GPU!

  3. Training阶段: FSDP引擎活跃
     → compute_log_prob → advantage → update_policy → backward+optimizer

  4. weight_sync: actor→rollout
     → Level 1(LoRA): 只sync adapter → rollout.resume(tags=["kv_cache"])
     → Level 2(full): sync全部weights → rollout.resume(tags=["kv_cache", "weights"])

  5. wake_up: vLLM恢复GPU内存 → 重新分配weights+KV cache → 可以rollout了!

  关键: sleep→training→weight_sync→wake_up = 1个step的完整cycle!
  → GPU时间分时复用: rollout占→sleep→training占→wake→rollout占
  → 这就是为什么HYBRID需要sleep/wake → GPU不能同时被rollout和training使用!
```

## 5. Weight Sync 3种后端

### 5.1 naive (ColocatedCheckpointEngine)

```python
# checkpoint_engine/base.py:220-275
# HYBRID模式 → in-process → Python generator零拷贝!

class ColocatedCheckpointEngine:
    def send_weights(self):
        self.weights = actor.engine.get_per_tensor_param()  # generator
        # 不做任何网络传输! 直接Python引用传递!

    def receive_weights(self):
        yield from self.weights  # 同进程 → Python generator → 零拷贝!

# 优点: 最快 → 无IPC/NCCL/RDMA overhead → Python引用传递
# 缺点: 只能HYBRID模式 → rollout和actor必须同进程
# RTX 4090最优: HYBRID+naive → 零weight sync开销!
```

### 5.2 NCCL

```
# COLOCATED或STANDALONE → 跨进程/跨GPU → NCCL ProcessGroup

流程:
  1. Abort所有in-flight rollout requests → 确保vLLM不在生成
  2. 释放KV cache (保留weights → NCCL写入weight buffer)
  3. 建NCCL ProcessGroup(trainer workers + rollout workers)
  4. Trainer: NCCL AllReduce/Broadcast weight buckets
  5. Rollout: 接收weights → server_adapter.update_weights()
  6. 关闭ProcessGroup → 恢复KV cache → resume generation

bucket_size=2048MB → 多个weight tensor打包成1个bucket → 减少NCCL launch次数

优点: 跨GPU weight sync → 支持STANDALONE模式
缺点: NCCL overhead → PCIe瓶颈 → RTX 4090上很慢!
```

### 5.3 NIXL (RDMA P2P Ring)

```python
# checkpoint_engine/nixl_checkpoint_engine.py
# RDMA → NVIDIA NIXL库 → UCX/UCCL/Mooncake backends

class NIXLCheckpointEngine:
    def send_weights(self):
        # Ring topology: trainer_rank0 → rollout_rank1 → ... → rollout_rankN → back
        # 双buffer: send_buf + recv_buf → swap per iteration
        for weight_chunk in bucketed_weights:
            readable_op = nixl_agent.create_read_operation(send_buf)
            nixl_agent.push(readable_op, next_agent)  # RDMA push到下一个
            swap(send_buf, recv_buf)  # 双buffer交替

    def receive_weights(self):
        # 从prev_agent接收 → yield到vLLM → 同时接收下一个bucket
        yield from send_buf  # 当前bucket → vLLM处理
        recv_buf = nixl_agent.pull(prev_agent)  # RDMA接收下一个

# 优点: RDMA → GPU直传 → 无CPU拷贝 → 最快跨节点weight sync
# 缺点: 需要NIXL库+RDMA网络 → RTX 4090无RDMA → 不适用!
# 适用: NVLink+RDMA集群 → H100/H800跨节点
```

### CUDA IPC (BucketedWeightSender/Receiver)

```python
# rollout/vllm_rollout/bucketed_weight_transfer.py
# COLOCATED模式 → 同GPU但不同进程 → CUDA IPC

class BucketedWeightSender:
    # ZMQ IPC socket: ipc:///tmp/rl-colocate-zmq-...sock
    # 主路径: CUDA IPC handles (torch.multiprocessing.reductions.reduce_tensor)
    #   → tensor留在GPU → 进程间共享GPU内存 → 无拷贝!
    # 备选: Shared memory (Ascend NPU → IPC不支持)

    # Bucketed: 多个weight打包成bucket_size buffer → 减少ZMQ消息次数
    # 流程: send metadata → send CUDA IPC handle → receiver重建tensor

class BucketedWeightReceiver:
    # 接收ZMQ消息 → 从CUDA IPC handle重建tensor → 写入vLLM模型
```

### Weight Sync对比

| 后端 | 传输方式 | 适用模式 | 开销 | RTX 4090 |
|------|---------|---------|------|---------|
| **naive** | Python generator(零拷贝) | HYBRID | 0 | ✅ 最优 |
| **CUDA IPC** | ZMQ+torch IPC(GPU直传) | COLOCATED | 很低 | ✅ 可行 |
| **NCCL** | AllReduce/Broadcast | STANDALONE | 中 | ❌ PCIe慢 |
| **NIXL** | RDMA P2P ring | STANDALONE+跨节点 | 最低(跨节点) | ❌ 无RDMA |

## 6. ResourcePoolManager: GPU资源分配

```python
# ray/base.py:182-239
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]  # {"pool_0": [8], "pool_1": [8]}
    mapping: dict[int, str]                    # {0: "pool_0", 1: "pool_1"}
    max_colocate_count: int = 3                # FSDP: actor+rollout+ref共享

# max_colocate_count=3 → 每Ray actor得1/3 GPU → 3 WorkerGroups共享同一GPU!
#   → actor WorkerGroup + rollout WorkerGroup + ref WorkerGroup
#   → 或 actor + critic + ref (PPO模式)
```

### Ray PlacementGroup创建

```python
# ray/base.py:112-161
class RayResourcePool:
    def get_placement_groups(self):
        # 每bundle: {CPU: max_colocate_count, GPU: 1}
        # STRICT_PACK策略 → 所有bundles强制同一节点 → 确保colocation!
        # PG按node IP排序 → rank稳定性
```

### split_resource_pool: 分割资源池

```python
# ray/base.py:269-315
# 从一个pool创建SubRayResourcePool → 分配不同角色
# → rollout可以用不同的TP/PP配置 → 与训练不同的并行拓扑!
# → 关键: rollout(vLLM TP=4) 和 training(FSDP DP=4) 可以在同一组GPU上!
```

## 7. ActorRolloutRefWorker 初始化顺序

```
init_model (engine_workers.py:500-632) → @register(ONE_TO_ALL)

初始化顺序(有意的!):

1. Ref模型 (行504-539):
   → ref_config: ActorConfig → TrainingWorker → FSDP引擎
   → ref.reset() → 构建模型+FSDP wrap → 只forward(no backward)
   → CPU offload选项 → ref不参与训练 → 可以offload到CPU省GPU内存!

2. Actor模型 (行542-588):
   → actor_config: ActorConfig → TrainingWorker → FSDP引擎
   → actor.reset() → 构建模型+FSDP wrap+optimizer+LR scheduler
   → loss_fn = ppo_loss(config=actor_config) → PPO loss绑定
   → 完整训练引擎 → forward+backward+optimizer

3. Rollout引擎 (行591-616):
   → 3D DeviceMesh(dp, infer_tp, infer_pp) → 推理并行拓扑
   → rollout_cls = vllm.ServerAdapter / sglang.ServerAdapter / trtllm.ServerAdapter
   → LoRA flags: base_sync_done, layered_summon, peft_merge

4. Checkpoint引擎 (行619-629):
   → CheckpointEngineRegistry.new() → weight sync pipeline

5. GPU缓存清空 (行632):
   → aggressive_empty_cache(force_sync=True) → 为vLLM释放GPU内存!

关键: Ref→Actor→Rollout→Checkpoint顺序:
  → Ref先初始化(CPU offload) → Actor占GPU → Rollout接管GPU
  → 清空GPU缓存后 → Rollout引擎可以分配KV cache
```

## 8. FSDP引擎初始化详解

### FSDP2 wrapping流程

```python
# fsdp/transformer_impl.py + fsdp_utils.py

FSDPEngine._build_fsdp_module (行340-428):

FSDP1 (行340-404):
  module = FSDP(module, auto_wrap_policy=..., sharding_strategy=..., ...)

FSDP2 (行405-428):
  # 先保存full state_dict (FSDP2初始化后参数变为DTensor → 需先保存!)
  full_state = module.state_dict()
  apply_fsdp2(module, fsdp_kwargs, engine_config)
  fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)

# apply_fsdp2 (fsdp_utils.py:559-600):
  # 先wrap每个transformer layer → 然后wrap根module
  modules = _select_fsdp2_wrap_targets(model, transformer_layer_cls)
  for module in modules:
      fully_shard(module, **fsdp_kwargs)  # per-layer DTensor sharding
  fully_shard(model, **fsdp_kwargs)       # root module sharding
```

### Sharding策略选择

```python
# fsdp/utils.py:61-92
def get_sharding_strategy(device_mesh, zero3_enable=True):
    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD       # ZeRO-3等价
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD      # HSDP (ZeRO-3+DDP)
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP     # ZeRO-2等价
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        return fsdp_strategy                               # 1D → 全分片
    elif device_mesh.ndim == 2:                             # 2D → HSDP
        return hsdp_strategy                               # (ddp, fsdp) mesh
```

### DeviceMesh创建

```python
# fsdp/utils.py:40-58
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 1D mesh → 全FSDP → ZeRO-3等价
        device_mesh = init_device_mesh("cuda", (world_size,), ["fsdp"])
    else:
        # 2D mesh → HSDP → ZeRO-3+DDP
        device_mesh = init_device_mesh("cuda",
            (world_size // fsdp_size, fsdp_size),
            ["ddp", "fsdp"])
```

## 9. Rollout-Training Sharding转换 (无正式ShardingManager!)

```
verl没有独立的ShardingManager类!
→ 分片转换嵌入在ActorRolloutRefWorker的sleep/wake/update_weights流程中

转换本质: 内存所有权交换 + weight同步

训练阶段:
  FSDP引擎拥有GPU → DTensor shard在FSDP mesh上
  → AllGather→完整参数→forward→backward→ReduceScatter→1/N shard

Rollout阶段:
  vLLM拥有GPU → TP shard在vLLM内部
  → PagedAttention + KV cache + continuous batching

桥接: actor.engine.get_per_tensor_param()
  → FSDP: AllGather所有参数 → yield (name, full_tensor)
  → Rollout: update_weights → vLLM内部做TP分片 → 不需要显式resharding!
  → 关键: FSDP gather完整参数 → vLLM自己做TP分片 → 中间是完整tensor!

RTX 4090影响:
  → 单GPU: TP=1 → FSDP DP=1 → 无任何sharding转换 → 最简单!
  → HYBRID+naive: 同进程 → Python generator → 最快weight sync!
  → 多GPU PCIe: NCCL weight sync → 灾难 → 只能单GPU+LoRA
```

## 10. RTX 4090最佳配置

```
RTX 4090 24GB, 7B模型:

最优: HYBRID + naive + GRPO + LoRA
  → RolloutMode=HYBRID → actor+rollout同进程
  → Weight Sync=naive → Python generator零拷贝 → 0开销
  → Sleep/Wake: sleep_level=1(LoRA) → 只释放KV cache → 快速wake_up
  → GRPO: 无critic → actor+rollout共用GPU → 17GB peak → ✓fit 24GB
  → LoRA: 只sync adapter weights → ~2.6GB → 极快weight sync

时序:
  1. Rollout: vLLM generate(占据GPU, weights+KV cache)
  2. Sleep(level=1): 释放KV cache → FSDP做forward+backward
  3. Weight sync: naive → generator → 只sync LoRA adapters(~2.6GB)
  4. Wake_up: vLLM恢复KV cache → 可以rollout了!

不适合RTX 4090:
  → STANDALONE: 需要多GPU集 → PCIe NCCL weight sync灾难
  → COLOCATED NCCL: 跨进程IPC → 可行但不如HYBRID
  → NIXL: 无RDMA → 不适用
  → PPO: 2×模型内存 → 28GB ✗超出24GB

8×NVLink A100/H100 80GB:
  → HYBRID或COLOCATED均可 → NCCL weight sync很快
  → FSDP2+compile → 2-3x训练加速
  → PPO可用(critic独立GPU池) → 2×模型内存→80GB fit
  → NIXL RDMA → 跨节点weight sync → 最快
```

## 11. 关键设计洞察

```
1. FusedWorker → 同进程colocation → 最极致的GPU共享
   → HYBRID模式的核心 → actor+rollout=1个Ray actor → 0 IPC开销

2. Sleep/Wake → GPU分时复用 → 解决rollout+training争GPU问题
   → vLLM API: engine.sleep(level) / engine.wake_up(tags)
   → Level 1: LoRA-safe → 只释放KV cache → 快
   → Level 2: 完全清空 → 全量weight sync → 慢但必须

3. 3种Weight Sync → 适配3种部署模式
   → naive=HYBRID(in-process) → NCCL=STANDALONE(跨GPU) → NIXL=RDMA(跨节点)
   → CUDA IPC=COLOCATED(同GPU跨进程) → BucketedWeightSender

4. Dispatch路由 → WorkerGroup自动管理DP并行
   → @register(ONE_TO_ALL) → 广播
   → @register(DP_COMPUTE_PROTO) → DataProto自动split/merge
   → ND_compute → 支持非均匀DP-TP-PP拓扑

5. ResourcePoolManager → Ray PG + 分数GPU → 灵活colocation
   → max_colocate_count=3 → 3 WorkerGroups共享同一GPU
   → split_resource_pool → rollout用不同TP/PP → 灵活并行配置

6. 初始化顺序 → Ref→Actor→Rollout→Checkpoint → 有意的
   → Ref可CPU offload → Actor是核心 → Rollout接管GPU → Checkpoint桥接

7. 无ShardingManager → FSDP gather完整参数 → vLLM内部TP分片
   → 中间是完整tensor → 不需要显式resharding → 简化!
```

---

Sources:
- verl/single_controller/base/worker_group.py — WorkerGroup基类
- verl/single_controller/ray/base.py — RayWorkerGroup + ResourcePoolManager + PlacementGroup
- verl/single_controller/base/decorator.py — Dispatch + @register装饰器
- verl/workers/engine_workers.py — ActorRolloutRefWorker + TrainingWorker
- verl/workers/rollout/replica.py — RolloutMode + RolloutReplica sleep/wake
- verl/workers/rollout/vllm_rollout/vllm_async_server.py — vLLM sleep/wake API
- verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py — CUDA IPC weight transfer
- verl/checkpoint_engine/base.py — CheckpointEngine ABC + ColocatedCheckpointEngine
- verl/checkpoint_engine/nixl_checkpoint_engine.py — NIXL RDMA weight sync
- verl/workers/engine/fsdp/transformer_impl.py — FSDPEngine
- verl/utils/fsdp_utils.py — apply_fsdp2 + create_device_mesh
- notebook/projects/verl-rl-infra-reading.md + verl-dataproto-data-flow-reading.md
