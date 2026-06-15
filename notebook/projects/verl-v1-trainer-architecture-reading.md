# verl V1 Trainer Architecture 源码级深度阅读

> 最后更新: 2026-06-15
> verl版本: main branch (commit db33bc8d, v0.8.0+)
> 源码路径: `verl/trainer/ppo/v1/`

## 目录

1. [V1 Trainer 总体架构](#1-v1-trainer-总体架构)
2. [TransferQueue + KVBatchMeta: 避免大tensor拷贝的核心机制](#2-transferqueue--kvbatchmeta)
3. [PPOTrainerBase.step() 数据流](#3-ppotrainerbasestep-数据流)
4. [ReplayBuffer: 分区+采样机制](#4-replaybuffer)
5. [_compute_reward_colocate 和 _compute_old_log_prob: GRPO vs PPO](#5-grpo-vs-ppo-差异)
6. [TrainingWorker + engine.train_batch: 实际训练发生地](#6-trainingworker-训练执行)
7. [三种 Trainer 变体: sync / colocate_async / separate_async](#7-三种trainer变体)
8. [V1 vs Legacy 效率对比](#8-v1-vs-legacy-效率对比)
9. [RTX 4090 实战分析](#9-rtx-4090)

---

## 1. V1 Trainer 总体架构

### 文件结构 (共7个文件, ~2174行)

```
verl/trainer/ppo/v1/
├── __init__.py           (35行)  — 导出PPOTrainer, register_trainer, get_trainer_cls, 3种trainer变体
├── trainer_base.py       (1447行) — ★ PPOTrainer ABC, step()10步数据流, KVBatchMeta操作
├── trainer_sync.py       (42行)  — @register_trainer("sync"), sync模式
├── trainer_colocate_async.py (57行) — @register_trainer("colocate_async"), async+colocate
├── trainer_separate_async.py (150行) — @register_trainer("separate_async"), async+separate(NotImplementedError)
├── replay_buffer.py      (150行) — ReplayBuffer, TransferQueue KV存储, GRPO分区采样
├── agent_loop_tq.py      (237行) — AgentLoopWorkerTQ/AgentLoopManagerTQ, TransferQueue适配
├── utils.py              (92行)  — compute_advantage_for_multi_trajectories, GRPO多轨迹advantage
```

### 类层次

```
PPOTrainer (ABC, trainer_base.py:100)
│   ├── TRAINER_REGISTRY: dict[str, type] — 注册表
│   ├── register_trainer(name) — 类装饰器注册
│   ├── get_trainer_cls(name) — 查询
│   ├── step() → 10步数据流
│   ├── fit() → 主循环
│   ├── _setup() → 初始化10步
│   ├── ReplayBuffer → TransferQueue采样
│   ├── agent_loop_manager → AgentLoopManagerTQ
│   ├── LLMServerManager → RolloutReplica管理
│   ├── CheckpointEngineManager → 权重同步
│
├── PPOTrainerSync (@register_trainer("sync"))
│   on_step_end → update_weights(naive)
│   on_sample_end → sleep_replicas
│
├── PPOTrainerColocateAsync (@register_trainer("colocate_async"))
│   get_llm_client → FullyAsyncLLMServerClient
│   on_train_begin → warmup batches
│   on_step_end → update_weights + resume_generation
│   on_sample_end → abort + sleep
│
└── PPOTrainerSeparateAsync (@register_trainer("separate_async"))
      NotImplementedError — 尚未完成
    设计: hybrid engine mode切换(TRAINER/ROLLOUT)
    LLMServerManager standalone → global_load_balancer add/remove
```

### 初始化流程 (_setup, trainer_base.py:129-258)

```
1. _init_tokenizer → hf_tokenizer + hf_processor
2. _init_dataloader → StatefulDataLoader(train/val)
3. _init_dump_executor → ThreadPoolExecutor(1 worker)
4. _init_resource_pool_mgr → ResourcePoolManager(global_pool + optional reward_pool)
5. ActorRolloutRefWorker → RayClassWithInitArgs → create_colocated_worker_cls
6. [if use_critic] Critic → TrainingWorker → RayClassWithInitArgs → value_loss partial
7. RayWorkerGroup.spawn → all_wg dict
8. [if use_critic] critic_wg.reset() + set_loss_fn(value_loss_)
9. actor_rollout_wg.init_model()
10. RewardLoopManager → rule-based or model-based
11. [if use_teacher] MultiTeacherModelManager
12. ★ LLMServerManager.create(config, worker_group, rollout_resource_pool)
13. ★ CheckpointEngineManager(config, trainer=actor_rollout_wg, replicas=llm_server_manager.get_replicas())
14. checkpoint_manager.sleep_replicas → _load_checkpoint
```

**关键变化**: V1不再像Legacy那样把rollout作为WorkerGroup的一部分, 而是通过LLMServerManager管理独立的RolloutReplica. 这使rollout可以独立sleep/wake, 支持async模式.

---

## 2. TransferQueue + KVBatchMeta: 避免大tensor拷贝的核心机制

### TransferQueue 简介

TransferQueue是华为Ascend团队开源的异步流式数据管理系统:
- GitHub: https://github.com/Ascend/TransferQueue
- 论文: AsyncFlow (arXiv:2507.01663)
- verl集成: PR#5401, 128×H100端到端性能提升49.1%

核心架构: **Control Plane(元数据)** + **Data Plane(存储)** + **User Interface**

### KVBatchMeta: 轻量级元数据引用

`KVBatchMeta`是TransferQueue提供的高层API数据类型, 它只存储**key引用**, 不存储实际tensor数据:

```python
# transfer_queue.KVBatchMeta 的关键属性
class KVBatchMeta:
    partition_id: str    # 分区名("train"/"val")
    keys: list[str]      # key列表, 格式: {uid}_{session_id}_{index}
    tags: list[dict]     # 元数据tag列表(prompt_len, response_len, seq_len, status, global_steps)
    extra_info: dict     # 附加信息(temperature, mini_batch_size等)
```

**★ 核心设计**: V1 trainer中, `batch` 变量几乎全部是 `KVBatchMeta`, 不是 `DataProto`. 所有中间计算结果(old_log_probs, entropy, advantages, values等)直接通过 `tq.kv_batch_put` 写入TransferQueue, 下一步通过 `tq.kv_batch_get(keys=batch.keys, select_fields=[...])` 按需读取特定字段.

### 为什么避免大tensor拷贝?

**Legacy RayPPOTrainer的数据流瓶颈**:
- 所有DataProto必须经过RayPPOTrainer(driver进程) → **单点瓶颈**
- 每一步: worker_group计算 → Ray.get拉回driver → union合并 → 传给下一步worker_group → **全量tensor来回拷贝**
- 一个7B模型的PPO batch (128 prompts × rollout_n=8 × 512 tokens): ~4MB tensor × 6步 × 2方向 = ~48MB网络拷贝

**V1的零拷贝设计**:

```python
# Legacy方式: 全量tensor经过driver
batch = DataProto(...)  # ~4MB tensor在driver内存
output = self.actor_rollout_wg.compute_log_prob(batch)  # 4MB → Ray RPC → workers → 4MB返回
batch = batch.union(output)  # 4MB+4MB = 8MB在driver
ref_output = self.ref_policy_wg.compute_ref_log_prob(batch)  # 8MB → workers → 4MB返回
batch = batch.union(ref_output)  # 8MB+4MB = 12MB在driver

# V1方式: 只有KVBatchMeta在driver, 实际数据在TransferQueue
batch = KVBatchMeta(partition_id="train", keys=[...], tags=[...])  # 几KB!
# 步1: workers直接读TransferQueue, 计算log_prob, 写回TransferQueue
output = self.actor_rollout_wg.compute_log_prob(batch)  # 几KB KVBatchMeta经Ray → workers从TQ读数据 → 计算 → 写回TQ
# 步2: 按需select_fields读取
data = tq.kv_batch_get(keys=batch.keys, select_fields=["log_probs", "response_mask"])  # 只读需要的字段
data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
tq.kv_batch_put(keys=batch.keys, fields=data.select("old_log_probs", "entropy"))  # 写回, 其他字段不动
```

**★ 效率提升核心**:
1. **driver内存**: 从O(N×d) tensor → O(N) KVBatchMeta(keys+tags). 128 prompts的KVBatchMeta ~几KB vs 4MB DataProto
2. **worker通信**: 从全量tensor Ray RPC → KVBatchMeta元数据RPC + TransferQueue本地读写
3. **字段级读写**: `select_fields=[...]` 只读/写需要的列, 不必全量union
4. **生产者-消费者解耦**: AgentLoopWorker直接写入TransferQueue, Trainer从TransferQueue按需读取

### TransferQueue API (V1使用的高层KV接口)

```python
import transfer_queue as tq

# 初始化 (AgentLoopWorkerTQ.__init__中)
tq.init()

# 写入 (AgentLoopWorkerTQ._agent_loop_postprocess)
await tq.async_kv_batch_put(
    keys=keys,                  # ["{uid}_{session_id}_{index}", ...]
    fields=list_of_dict_to_tensordict(fields),  # TensorDict格式
    tags=tags,                  # [{"status":"success","seq_len":1024,...}, ...]
    partition_id="train"        # 分区
)

# 读取 (trainer_base.py多处)
data = tq.kv_batch_get(
    keys=batch.keys,            # KVBatchMeta.keys
    partition_id=batch.partition_id,
    select_fields=["rollout_log_probs", "response_mask"]  # ★ 字段级读取
)

# 更新 (只写特定字段, 不影响其他)
tq.kv_batch_put(
    keys=batch.keys,
    partition_id=batch.partition_id,
    fields=data.select("old_log_probs", "entropy")  # ★ 只更新需要的字段
)

# GRPO组状态管理
await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})  # prompt状态

# 清理
tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

# 列出所有key+tag
tq.kv_list()  # → {partition_id: {key: tag}}
```

### TransferQueue存储后端

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| SimpleStorageUnit | CPU内存, 默认, 分布式(每节点一个) | 小规模训练 |
| openYuanrong | Ascend原生, HBM/DRAM/SSD分层 | 昇腾NPU集群 |
| MooncakeStore | RDMA传输, GPU→DRAM | 大规模GPU集群 |
| RayRDT(alpha) | Ray对象直接传输 | Ray生态 |

**RTX 4090**: SimpleStorageUnit(CPU内存)即可, RDMA不需要(PeCe瓶颈).

---

## 3. PPOTrainerBase.step() 数据流

### step() 10步数据流 (trainer_base.py:381-428)

```
step(metrics, timing_raw) → KVBatchMeta
│
│  1. _add_batch_to_generate()
│     → 从dataloader取batch → 赋uid → tq.kv_batch_put(keys=uids, tags=[{is_prompt:True,status:"pending"}])
│     → agent_loop_manager.generate_sequences(batch)  # ★ fire-and-forget, 不等结果!
│
│  2. replay_buffer.sample(partition_id="train", batch_size=config.data.train_batch_size)
│     → ★ 阻塞等待足够finished/failure prompts → 返回KVBatchMeta
│     → batch.extra_info["temperature"] = rollout.temperature
│     on_sample_begin() / on_sample_end()  # sleep/wake hooks
│
│  3. [OPTIONAL] _compute_reward_colocate(batch)  # reward model在actor worker上计算
│
│  4. _balance_batch(batch)  # seqlen平衡 + upsample到DP divisible
│     → upsample_batch_to_divisible_size → get_seqlen_balanced_partitions
│     → batch.reorder()  # ★ KVBatchMeta.reorder()
│
│  5. _compute_old_log_prob(batch)
│     → [if bypass_mode] tq.kv_batch_get(select_fields=["rollout_log_probs"]) → rename → kv_batch_put
│     → [else] actor_rollout_wg.compute_log_prob(batch) → kv_batch_get → response_from_nested → kv_batch_put
│
│  6. [OPTIONAL] _compute_ref_log_prob(batch)
│     → [if ref_in_actor] actor_rollout_wg.compute_log_prob(no_lora_adapter=True)
│     → [else] ref_policy_wg.compute_ref_log_prob(batch)
│     → kv_batch_get → response_from_nested → kv_batch_put(ref_log_prob)
│
│  7. [OPTIONAL] _compute_values(batch)
│     → critic_wg.infer_batch(batch) → kv_batch_get → response_from_nested → kv_batch_put(values)
│
│  8. _compute_advantage(batch)
│     → kv_batch_get(select_fields=[uid, response_mask, rm_scores, rollout_log_probs, old_log_probs, ref_log_prob, values])
│     → DataProto(batch=data.to_padded_tensor()) → apply_kl_penalty → compute_rollout_correction
│     → ★ compute_advantage_for_multi_trajectories(batch_keys=batch.keys) → GRPO多轨迹
│     → response_to_nested → kv_batch_put(advantages, returns)
│
│  9. [OPTIONAL] _update_critic(batch)
│     → critic_wg.train_mini_batch(batch) → reduce_metrics
│
│ 10. _update_actor(batch)
│     → actor_rollout_wg.update_actor(batch) → reduce_metrics
│
│  return batch  # KVBatchMeta
```

### 关键差异 vs Legacy step()

| 特性 | Legacy RayPPOTrainer | V1 PPOTrainerBase |
|------|---------------------|--------------------|
| **数据类型** | DataProto(全量tensor) | KVBatchMeta(key引用) |
| **rollout** | 同步阻塞generate_sequences | fire-and-forget via AgentLoopManagerTQ |
| **采样** | 直接从gen_batch_output取 | ReplayBuffer从TransferQueue异步采样 |
| **中间数据** | batch.union()逐步累加tensor | tq.kv_batch_put/get按需读写 |
| **padding** | no_padding→padding来回转换 | to_padded_tensor仅在需要时 |
| **数据流方向** | 全量经过driver→worker→driver循环 | Worker直接读写TransferQueue, driver只传KVBatchMeta |
| **GRPO** | batch.repeat(n)+interleave | AgentLoop自动n sessions+uid分组 |
| **multi-turn** | 需要额外配置 | AgentLoop原生支持 |
| **seqlen balance** | DataProto reorder | KVBatchMeta.reorder() |

---

## 4. ReplayBuffer: 分区+采样机制

### ReplayBuffer (replay_buffer.py:30-150)

```python
class ReplayBuffer:
    def __init__(self, poll_interval=2.0):
        self.poll_interval = poll_interval
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)  # partition_id → {key: tag}
        self.pending_keys: dict[str, set] = defaultdict(set)   # prompt已提交但session未开始
        self.running_keys: dict[str, set] = defaultdict(set)   # 所有session正在运行
        self.finished_keys: dict[str, set] = defaultdict(set)  # 所有session完成(无错误)
        self.failure_keys: dict[str, set] = defaultdict(set)   # 有session失败
```

### GRPO组采样控制

ReplayBuffer的核心设计是为GRPO的group sampling服务:

1. **prompt注册**: `_add_batch_to_generate()` 将每个prompt uid注册到TransferQueue, tag=`{is_prompt:True, status:"pending"}`
2. **session状态**: AgentLoopWorkerTQ._run_prompt → `tq.async_kv_put(key=uid, tag={"status":"running"})`
3. **完成标记**: 所有n sessions完成后 → `tq.async_kv_put(key=uid, tag={"status":"finished"})`
4. **采样条件**: `replay_buffer.sample()` 阻塞等待 `len(finished_keys) + len(failure_keys) >= batch_size`

### sample() 详解

```python
def sample(self, partition_id: str, batch_size: int) -> KVBatchMeta:
    # 1. 从TransferQueue同步所有元数据
    self._sync_metadata_from_transfer_queue()  # tq.kv_list()

    # 2. 阻塞等待足够的finished/failure prompts
    while len(self.finished_keys[partition_id]) + len(self.failure_keys[partition_id]) < batch_size:
        time.sleep(self.poll_interval)
        self._sync_metadata_from_transfer_queue()

    # 3. 选择prompt uids (finished + failure)
    selected_prompt_uids = list(finished_keys.union(failure_keys))[:batch_size]
    tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)  # 清除prompt元数据

    # 4. 从partitions中找到所有相关trajectory keys
    # key格式: {uid}_{session_id}_{index}
    keys, tags = [], []
    selected = set(selected_prompt_uids)
    for key, tag in self.partitions[partition_id].items():
        uid = key.split("_")[0]  # ★ 从trajectory key提取uid
        if uid in selected:
            keys.append(key)
            tags.append(tag)

    # 5. 返回KVBatchMeta(只有keys+tags, 不含tensor!)
    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
```

**★ 关键设计**: `tq.kv_clear(keys=selected_prompt_uids)` 清除prompt的元数据(从meta server删除), 但trajectory数据仍在TransferQueue storage unit中, 由后续step中通过keys引用. 这实现了"元数据一次消费, 数据按需读取".

### 分区机制

- `partition_id="train"`: 训练数据分区
- `partition_id="val"`: 验证数据分区
- 每个分区独立管理, 元数据隔离
- 允许train和val并行运行(async模式下)

---

## 5. _compute_reward_colocate 和 _compute_old_log_prob: GRPO vs PPO

### _compute_reward_colocate (trainer_base.py:1084-1087)

```python
def _compute_reward_colocate(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    """Compute the reward with colocate reward model."""
    # TODO: add reward model
    raise NotImplementedError
```

**当前状态**: V1的colocate reward model尚未实现. 这意味着V1目前只支持rule-based reward(RewardLoopManager), 或独立的reward model worker.

**GRPO典型场景**: rule-based reward(custom_reward_function) → 在AgentLoopWorkerTQ._compute_score中计算 → 直接写入TransferQueue的rm_scores字段 → V1 trainer无需额外reward pass.

**PPO需要reward model时**: 需要在step()第3步调用_compute_reward_colocate, 但V1目前NotImplementedError → 只能用独立的RM worker(reward_loop_manager).

### _compute_old_log_prob (trainer_base.py:1134-1193)

这是V1和Legacy差异最大的方法之一:

```python
def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    # ★ 模式选择: bypass vs decoupled
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

    if bypass_recomputing_logprobs:  # ★ GRPO典型: π_rollout直接作为π_old
        data = tq.kv_batch_get(keys=batch.keys, select_fields=["rollout_log_probs"])
        data["old_log_probs"] = data.pop("rollout_log_probs")
        tq.kv_batch_put(keys=batch.keys, fields=data)
        return batch  # ★ 零GPU计算! 只做key重命名

    # Decoupled mode: 重新计算old_log_probs(3 policy: π_rollout, π_old, π_θ)
    batch.extra_info.update({"calculate_entropy": True, "compute_loss": False, "temperature": ...})
    output = self.actor_rollout_wg.compute_log_prob(batch)  # → infer_batch → forward pass

    # 从TransferQueue读取计算结果
    fields = ["entropy", "log_probs", "response_mask"]
    if self.config.actor_rollout_ref.rollout.calculate_log_probs:
        fields.extend(["responses", "rollout_log_probs"])
    data = tq.kv_batch_get(keys=batch.keys, select_fields=fields)

    # ★ response_from_nested: nested tensor → padded tensor转换
    data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
    data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])

    # 只写old_log_probs和entropy回TransferQueue(不影响其他字段)
    tq.kv_batch_put(keys=batch.keys, fields=data.select("old_log_probs", "entropy"))

    # 构造DataProto做metrics计算(只在driver, 不传worker)
    data = DataProto(batch=data.to_padded_tensor())
    # compute entropy metrics, debug metrics...
```

### GRPO vs PPO 在old_log_prob的差异

| 特性 | GRPO (bypass_mode=True) | PPO (bypass_mode=False) |
|------|------------------------|------------------------|
| **π_old来源** | 直接用π_rollout的log_probs | 重新用π_θ计算(old_log_probs) |
| **GPU计算** | 0 (只做KV字段重命名) | 1次forward pass |
| **策略数量** | 2 (π_rollout, π_θ) | 3 (π_rollout, π_old, π_θ) |
| **IS correction** | PPO ratio自然 = π_θ/π_rollout | 需要rollout_correction |
| **数据流** | tq.kv_batch_get → rename → tq.kv_batch_put | actor_rollout_wg.compute_log_prob → tq.kv_batch_get/put |
| **适合场景** | GRPO rollout_n小, stale风险低 | PPO, rollout_n大, stale风险高 |

**★ RTX 4090最优**: GRPO+bypass_mode → 省1次forward pass → 单GPU唯一可行路径.

### _compute_advantage: GRPO多轨迹处理 (utils.py)

```python
def compute_advantage_for_multi_trajectories(data, batch_keys, adv_estimator, ...):
    # ★ 非GRPO直接委托给legacy compute_advantage()
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return compute_advantage(data, ...)

    # ★ GRPO: 只用每个session的final output计算group-relative advantage
    # key格式: {uid}_{session_id}_{index}
    final_sessions: dict[str, tuple[int, int]] = {}  # session_key → (max_index, position)
    for i, key in enumerate(batch_keys):
        uid, session_id, index = key.rsplit("_", 2)
        session_key = f"{uid}_{session_id}"
        if session_key not in final_sessions or final_sessions[session_key][0] < index:
            final_sessions[session_key] = (index, i)

    # 只选final sessions做GRPO advantage
    final_data = compute_advantage(data.select_idxs(final_indices), adv_estimator=GRPO, ...)
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # ★ scatter: 将final session的score广播到同一session的所有output
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]  # 只在response_mask=1处有值
    data.batch["advantages"] = scores
    data.batch["returns"] = scores
```

**★ 关键**: 多turn AgentLoop中, 一个session可能产生多个output(index=0,1,2,...). GRPO只用最后一个output的reward score, 但广播到所有output的response_mask覆盖区域. 这保证了multi-turn场景下loss只在action tokens上计算.

---

## 6. TrainingWorker + engine.train_batch: 实际训练发生地

### TrainingWorker (engine_workers.py:76-430)

```python
class TrainingWorker(Worker, DistProfilerExtension):
    """Tinker-like API as RayWorkerGroup to single controller"""

    def __init__(self, config: TrainingWorkerConfig):
        # ★ EngineRegistry.new() → 根据strategy选择引擎(fsdp/megatron/veomni/...)
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )
        self.loss_fn = None  # ★ 由外部set_loss_fn设置

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        # 分mini-batch, 多epoch迭代
        mini_batch_size = tu.pop(data, "mini_batch_size")
        epochs = tu.pop(data, "epochs", 1)
        dataloader = tu.make_iterator(data, mini_batch_size=mini_batch_size_per_gpu, epochs=epochs, seed=...)

        with self.engine.train_mode():
            for batch_idx, mini_batch_td in enumerate(dataloader):
                actor_output = self.train_batch(mini_batch_td)  # ★ 单个mini-batch训练
                output_lst.append(actor_output)

        # 聚合metrics across mini-batches
        metrics = {}
        for output in actor_output:
            for key, val in output.items():
                if isinstance(val, list):
                    output[key] = Metric.aggregate_dp(val) if isinstance(val[0], Metric) else list(chain.from_iterable(val))
            append_to_dict(metrics, output)
```

### train_batch (engine_workers.py:325-370)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
def train_batch(self, data: TensorDict) -> TensorDict:
    assert self.loss_fn is not None
    # inject engineering parameters
    default_keys = dict(use_remove_padding, use_dynamic_bsz, max_token_len_per_gpu, ...)

    with self.engine.train_mode(), Timer("train_batch"):
        output = self.engine.train_batch(data, loss_function=self.loss_fn)  # ★ 实际训练!

    # update lr scheduler (if last mini-batch)
    if update_lr_scheduler:
        lr = self.engine.lr_scheduler_step()

    # postprocess: all_reduce loss, allgather metrics, compute mfu
    final_output = self._postprocess_output(output, ...)
```

### BaseEngine.train_batch (engine/base.py:113-130)

```python
def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
    maybe_fix_3d_position_ids(data)
    self.optimizer_zero_grad()
    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
    grad_norm = self.optimizer_step()
    outputs["metrics"]["grad_norm"] = grad_norm
    return outputs
```

**★ 完整训练链路**:

```
PPOTrainerBase._update_actor(batch: KVBatchMeta)
  → actor_rollout_wg.update_actor(batch)  # RayWorkerGroup dispatch to all workers
    → ActorRolloutRefWorker.update_actor(data: TensorDict)
      → actor.train_mini_batch(data)
        → TrainingWorker.train_mini_batch(data: TensorDict)
          → make_iterator → for mini_batch_td in dataloader:
            → self.train_batch(mini_batch_td)
              → self.engine.train_batch(data, loss_function=self.loss_fn)
                → BaseEngine.train_batch:
                  → optimizer_zero_grad()
                  → forward_backward_batch(data, loss_function, forward_only=False)
                  → optimizer_step()
                  → return {loss, metrics, grad_norm}
```

**★ V1的数据如何进入TrainingWorker**:

V1的 `_update_actor` 调用 `actor_rollout_wg.update_actor(batch)`, 其中batch是KVBatchMeta. RayWorkerGroup的dispatch机制会:

1. KVBatchMeta → 通过Ray RPC传给每个worker(只有keys+tags+extra_info, 几KB)
2. Worker内部: `tq.kv_batch_get(keys=local_keys, select_fields=[input_ids, position_ids, old_log_probs, advantages, ...])` → 从TransferQueue读取需要的tensor字段
3. 构造TensorDict → 传给engine.train_batch

**★ 这意味着**: 即使V1的actor update API仍然接受TensorDict(因为TrainingWorker不改), 但数据来源从"driver传全量DataProto"变成了"worker从TransferQueue按需读取".

---

## 7. 三种 Trainer 变体

### PPOTrainerSync (trainer_sync.py)

```python
@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """同步PPO trainer
    1. Trainer和rollout在同一GPU(colocated)
    2. 不支持partial rollout
    """

    def on_init_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        with marked_timer("update_weights"):
            self.checkpoint_manager.update_weights(self.global_steps)  # naive模式: 直接in-process sync

    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas()  # sleep所有replica(释放weights+KV cache)
```

**时序**: generate(blocking) → sleep → sample → reward → old_log_prob → advantage → update_actor → update_weights → sleep → next step

### PPOTrainerColocateAsync (trainer_colocate_async.py)

```python
@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    """异步PPO trainer
    1. Trainer和rollout在同一GPU(colocated)
    2. 支持partial rollout(中断续生成)
    """

    def get_llm_client(self):
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_train_begin(self):
        # warmup: 提前提交N个batch到agent_loop_manager
        num_warmup_batches = self.config.trainer.v1.colocate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()

    def on_step_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()  # resume而不是sleep

    def on_sample_end(self):
        self.checkpoint_manager.abort_replicas()  # abort未完成的请求
        self.checkpoint_manager.sleep_replicas()
```

**★ 关键差异**: colocate_async在on_sample_end时abort未完成请求, 但在on_step_end时resume生成. 这意味着:
- generation可以跨越多个training step(partial rollout)
- warmup batches在训练开始前就提交, 减少startup延迟
- FullyAsyncLLMServerClient支持abort→resume续生成

**时序**: [warmup: N batches generating] → step: abort_prev → sample(从replay_buffer) → compute → update_actor → update_weights → resume_generation → step end → next batch submitted → ...

### PPOTrainerSeparateAsync (trainer_separate_async.py) — NotImplementedError

设计方向:
```python
@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    """异步PPO trainer
    1. Trainer和rollout分离(GPU)
    2. Trainer idle时可切换为rollout(hybrid engine)
    3. 支持partial rollout
    """

    # ★ HybridEngineMode切换: TRAINER ↔ ROLLOUT
    class HybridEngineMode(Enum):
        TRAINER = "trainer"
        ROLLOUT = "rollout"

    # standalone_server_manager: 独立rollout集群
    # checkpoint_engine: NCCL/NIXL/Mooncake(非naive)
    # bypass_mode强制=True(π_rollout作为π_old)
    # global_load_balancer.add_servers/remove_servers动态管理
```

**时序设计**: generate(standalone) → switch_to_trainer → train → should_switch_to_rollout → switch_to_rollout → ...

**强制bypass_mode**: separate_async因为weight sync跨GPU有延迟, π_θ和π_rollout之间存在staleness, 所以必须用bypass_mode(π_old=π_rollout), 而不是重新计算old_log_probs.

---

## 8. V1 vs Legacy 效率对比

### 数据传输效率

| 维度 | Legacy (DataProto) | V1 (KVBatchMeta) |
|------|---------------------|--------------------|
| **driver→worker传输** | 全量tensor (~4MB/batch) | KVBatchMeta (~几KB) |
| **worker→driver传输** | 全量tensor (~4MB/step) | KVBatchMeta (~几KB) + select_fields按需 |
| **中间数据存储** | driver内存逐步累加 | TransferQueue分布式存储 |
| **字段读写** | 全量union | select_fields按需 |
| **7B PPO batch (128×8×512)** | ~48MB网络往返 | ~几KB元数据 + TQ本地读写 |

### 计算效率

| 维度 | Legacy | V1 |
|------|--------|-----|
| **rollout** | 同步阻塞 | fire-and-forget(async) |
| **GRPO bypass** | apply_bypass_mode(DataProto) | tq.kv_get→rename→kv_put(零GPU) |
| **padding** | no_padding↔padding来回 | nested tensor, 只在需要时to_padded |
| **seqlen balance** | DataProto reorder | KVBatchMeta.reorder + tags seq_len |
| **reward** | colocate RM(sleep/wake) | rule-based in AgentLoop / separate RM |
| **multi-turn** | 需额外配置 | AgentLoop原生支持 |

### 端到端性能 (TransferQueue官方数据)

- 128×H100集群, 多模态GRPO: **49.1%端到端提升**
- 来源: TransferQueue集成verl PR#5401

### 单步时间估算 (7B GRPO, 8GPU)

| 步骤 | Legacy耗时 | V1耗时 | 原因 |
|------|-----------|--------|------|
| generate | ~5s(同步) | ~0s(fire-and-forget, 由replay_buffer.sample阻塞等) | 异步生成 |
| reward | ~0s(rule-based) | ~0s(in AgentLoop) | reward前置到生成 |
| old_log_prob | ~1.5s | ~0s(bypass) or ~1.5s | bypass省1次forward |
| advantage | ~0.1s | ~0.1s | driver计算 |
| update_actor | ~2s | ~2s | 相同 |
| **driver→worker传输** | ~2s(多步) | ~0.2s(只有元数据) | KVBatchMeta |
| **总计** | ~10s | ~4-5s | ~2x加速 |

---

## 9. RTX 4090 实战分析

### 可行方案

V1 PPOTrainerSync + GRPO + LoRA-32 + bypass_mode + rule-based reward

```
配置:
- trainer_type: "sync"  (单GPU只能sync)
- algorithm.adv_estimator: GRPO
- algorithm.rollout_correction.bypass_mode: True
- actor_rollout_ref.model.lora.rank: 32
- actor_rollout_ref.rollout.n: 8 (rollout_n)
- reward: rule-based (custom_reward_function)
- data.train_batch_size: 16 (prompts per step)
- actor_rollout_ref.actor.ppo_mini_batch_size: 16
```

### 内存估算 (7B INT4 + LoRA-32)

```
模型权重(INT4):          ~3.5GB
LoRA(rank=32):           ~64MB
Optimizer(CPU_Adam):     ~0MB(GPU) / ~14GB(CPU)
KV cache(rollout):       ~2GB(INT8KV+GQA-8)
TransferQueue(SimpleStorageUnit): ~200MB(CPU, per batch)
Gradient+activation:     ~1GB
─────────────────────────
GPU总计:                  ~6.5GB → 24GB RTX 4090有余量!
```

### 为什么V1比Legacy更适合RTX 4090

1. **driver内存节省**: Legacy需要在driver进程累积4MB+的DataProto, V1只需要几KB的KVBatchMeta → driver进程几乎不占GPU内存
2. **GRPO bypass**: 省1次old_log_prob forward pass → 单GPU省~1.5s/step
3. **TransferQueue CPU存储**: SimpleStorageUnit用CPU内存, 不占GPU → RTX 4090的24GB全部留给模型+KV
4. **rule-based reward前置**: 在AgentLoop中计算, 不需要额外GPU pass → 单GPU唯一可行
5. **fire-and-forget generation**: 不阻塞driver → 单GPU可以同时做rollout+train(colocated)

### 不可用方案

- PPOTrainerColocateAsync: 需要partial rollout, RTX 4090单GPU内存不够同时rollout+train
- PPOTrainerSeparateAsync: NotImplementedError + 需要多GPU
- reward model: 需要额外GPU, 单GPU24GB不够
- critic(PPO): 需要value model, 双倍GPU需求 → 单GPU不可行

### 性能估算

```
V1 PPOTrainerSync + GRPO + LoRA-32 + bypass_mode:
- generate: ~1.15s (INT4, 5500tok/s, 128 prompts×8)
- reward: 0s (rule-based, 在AgentLoop中)
- old_log_prob: 0s (bypass_mode)
- advantage: ~0.1s (driver计算)
- update_actor: ~2s (LoRA forward+backward)
- sleep/wake: ~0.3s (weight sync)
- TransferQueue overhead: ~0.05s (CPU内存读写)
─────────────────────────
step总计: ~3.5s → ~3.5 tok/s training throughput
vs Legacy: ~10s → 3.5x加速
```

---

## 附录: 源码关键路径索引

| 文件 | 路径 | 行数 | 关键内容 |
|------|------|------|---------|
| trainer_base.py | `verl/trainer/ppo/v1/trainer_base.py` | 1447 | PPOTrainer ABC, step(), fit(), 10步数据流 |
| replay_buffer.py | `verl/trainer/ppo/v1/replay_buffer.py` | 150 | ReplayBuffer, GRPO分区采样 |
| trainer_sync.py | `verl/trainer/ppo/v1/trainer_sync.py` | 42 | sync模式hooks |
| trainer_colocate_async.py | `verl/trainer/ppo/v1/trainer_colocate_async.py` | 57 | async+colocate模式 |
| trainer_separate_async.py | `verl/trainer/ppo/v1/trainer_separate_async.py` | 150 | async+separate(NotImplemented) |
| agent_loop_tq.py | `verl/trainer/ppo/v1/agent_loop_tq.py` | 237 | AgentLoopWorkerTQ, TransferQueue适配 |
| utils.py | `verl/trainer/ppo/v1/utils.py` | 92 | GRPO多轨迹advantage |
| engine_workers.py | `verl/workers/engine_workers.py` | 758 | TrainingWorker, train_batch, infer_batch |
| engine/base.py | `verl/workers/engine/base.py` | ~230 | BaseEngine ABC, train_batch/infer_batch |
| llm_server.py | `verl/workers/rollout/llm_server.py` | ~500 | LLMServerManager, FullyAsyncLLMServerClient |
| ray_trainer.py | `verl/trainer/ppo/ray_trainer.py` | 1768 | Legacy RayPPOTrainer(对比基准) |
| transfer_queue.md | `docs/data/transfer_queue.md` | ~300 | TransferQueue官方文档 |

### V1核心创新总结

1. **TransferQueue替代DataProto**: KVBatchMeta(key引用) → 避免大tensor经driver来回拷贝 → driver瓶颈解除
2. **生产者-消费者解耦**: AgentLoopWorker写入TransferQueue, Trainer按需读取 → 真正异步
3. **GRPO原生支持**: uid分组+session_id+ReplayBuffer分区 → GRPO group sampling零额外逻辑
4. **bypass_mode零GPU**: GRPO bypass → old_log_probs=rollout_log_probs → 只做KV字段重命名
5. **TrainingWorker新API**: ActorRolloutRefWorker不再直接rollout → LLMServerManager管理独立Replica
6. **3种trainer变体**: sync/colocate_async/separate_async → 统一ABC + hooks → 灵活部署
7. **字段级读写**: select_fields → 只读写需要的列 → 不必全量union → 内存+带宽双省
