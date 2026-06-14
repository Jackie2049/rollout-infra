# verl Fully-Async Policy 源码级深度阅读

> 2026-06-15 | 源码: verl/experimental/fully_async_policy/ + verl/trainer/ppo/rollout_corr_helper.py
> 核心: 3 Ray actors(FullyAsyncRollouter+FullyAsyncTrainer+MessageQueue)→continuous streaming generation→independent training→staleness_threshold+IS/RS correction→partial_rollout resume→gen_batch_size=1→vs sync=sequential→RTX 4090单GPU只能sync

## 1. Fully-Async Policy 三组件架构

```
★ ★ 三组件 = 3个独立Ray actors → concurrent execution!

┌───────────────────────────────────────────────────────────────────┐
│  FullyAsyncTaskRunner (orchestrator)                              │
│    ├── FullyAsyncRollouter (@ray.remote, num_cpus=10)            │
│    │     ├── FullyAsyncLLMServerManager → hybrid+standalone      │
│    │     │     └── GlobalRequestLoadBalancer(sticky session)     │
│    │     │     └── RolloutReplica (Ray actors per GPU)           │
│    │     ├── FullyAsyncAgentLoopManager                          │
│    │     ├── RewardLoopManager (if use_rm)                       │
│    │                                                              │
│    ├── FullyAsyncTrainer (@ray.remote, num_cpus=10)              │
│    │     ├── Actor WorkerGroup (FSDP training)                   │
│    │     ├── CheckpointEngineManager (weight sync)               │
│    │     ├── Hybrid CheckpointEngineManager (validation)         │
│    │                                                              │
│    ├── MessageQueue (@ray.remote, num_cpus=2)                    │
│    │     ├── deque(maxlen=max_queue_size) → FIFO overflow        │
│    │     ├── asyncio.Condition → signal put/get                  │
│    └────────────────────────────────────────────────────────────│

★ 初始化顺序(fully_async_main.py:51-115):
  1. 创建Trainer(78) → 创建Rollouter(84)
  2. 注入Trainer actor worker group到Rollouter → hybrid replicas(81)
  3. 创建MessageQueue(97) → 注入client到Rollouter和Trainer(102-103)
  4. Trainer _fit_update_weights() → sync weights to Rollouter(110)
  5. ray.wait()并行启动 → rollouter.fit() + trainer.fit() → concurrent!
```

## 2. Continuous Streaming Generation — Rollouter

```
fully_async_rollouter.py (392-1046):

★ ★ Streaming Architecture → gen_batch_size=1 → single sample streaming!

_feed_samples() (815-846): dataloader → pending_queue(maxsize=128)
_processor_worker() (848-931): pending_queue → _process_single_sample_streaming()
_streaming_generation_main() (952-1006): asyncio.gather(_feed_samples + _processor_worker)
fit() (1011-1046): asyncio.gather(_streaming_generation_main + _async_monitor_loop)

★ _process_single_sample_streaming() (933-950):
  → async_rollout_manager.generate_sequences_single() → 单sample!
  → ray.cloudpickle.dumps(RolloutSample) → MessageQueue.put_sample()

★ ★ gen_batch_size=1 → 极细粒度 → trainer随时可consume!
  → vs sync: 整batch生成 → trainer等整个batch完成 → 阻塞!
  → async: 逐sample流式 → MessageQueue FIFO → trainer随时取 → overlap!
```

## 3. Independent Training — Trainer

```
fully_async_trainer.py (52-452):

★ ★ Training Loop → 从MessageQueue consume → 不等rollouter!

fit() (376-411): while True → fit_step()
fit_step() (412-452): complete PPO pipeline
  → _fit_generate() (454-463): _get_samples_from_queue() → required_samples
  → reward → log_prob → advantage → actor update

★ ★ _get_samples_from_queue() (275-332):
  → 从MessageQueue取required_samples个样本
  → ray.cloudpickle.loads() → deserialize
  → assemble_batch_from_rollout_samples() → 合成DataProto

★ ★ Trainer从不等Rollouter → 从queue取可用样本 → 训练独立进行!
  → 如果queue空 → trainer等待 → 但rollouter可能在生成更多
  → 如果queue满 → rollouter暂停 → 但trainer继续训练 → 自然overlap!
```

## 4. MessageQueue — Async Buffer

```
message_queue.py (26-234):

★ ★ Core: deque(maxlen=max_queue_size) → FIFO overflow!

max_queue_size计算(529-543):
  max_required_samples = required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step

★ Overflow behavior(68-72):
  → queue full → oldest samples popped → FIFO → 先进先出溢出!
  → 这意味着: 极旧样本自动丢弃 → trainer不处理过旧数据!

★ MessageQueueClient(180-234):
  → put_sample() → ray.cloudpickle.dumps → remote put
  → get_sample() → ray.cloudpickle.loads → remote get
  → get_queue_size() → staleness check
  → get_statistics() → monitoring

★ RolloutSample dataclass(detach_utils.py:28-39):
  → full_batch: Any → DataProto
  → sample_id: str → "sample_{epoch}_{global_steps}"
  → epoch: int
  → rollout_status: dict → statistics
```

## 5. Staleness Handling — 3种机制

```
★ ★ Staleness = async policy核心挑战 → 3种纠正!

A. Staleness Threshold + Pause/Resume(fully_async_rollouter.py):

  staleness_threshold: config, default=0.1(485)
  max_required_samples = required_samples * (staleness_threshold+1) * trigger_parameter_sync_step

  _should_pause_generation() (1077-1099):
    → queue_size >= max_queue_size AND staleness_samples >= max_required_samples
    → True → pause → drain → wait _resume_event

  reset_staleness() (564-595): trainer调用 → weight sync后
    → staleness_samples = active + queue samples → 重置计数
    → paused = False → set _resume_event → resume generation

B. Staleness Metrics Tracking(detach_utils.py):

  assemble_batch_from_rollout_samples():
    → min_global_steps / max_global_steps → weight version range
    → param_version_diff = abs(end - start) → staleness span per trajectory
    → partial_ratio = non-zero-diff / total → 跨weight sync boundary比例

C. Rollout Correction(IS + RS)(rollout_corr_helper.py):

  ★ ★ 16个预设 → 重要性采样+拒绝采样!

  关键预设:
    → TIS(截断IS): token/sequence level IS weights, 阈值上限
    → Icepop: 双边界IS weights, threshold range [lower, upper]
    → Geo-RS: 几何均值拒绝采样, seq_mean_k1 + ratio bounds
    → K3-RS: KL estimator-based rejection
    → ★ bypass_mode: True(默认) → rollout_log_probs = old_log_probs → PPO ratio自然处理IS!

  compute_rollout_correction_and_rejection_mask() (779-894):
    → 计算 IS weights → 应用 rejection masks → 追踪 off-policy metrics

★ ★ bypass_mode=True → 最简单correction → 用rollout log_probs代替old_log_probs → PPO ratio自动纠正!
```

## 6. Partial Rollout Resume

```
★ ★ Weight sync中断generation → partial_rollout resume!

FullyAsyncLLMServerClient.generate() (51-150):
  → stop_reason == "aborted" AND partial_rollout=True → resume!
  → 累积partial outputs到final_output → 多步拼接
  → 追踪per weight version的min/max_global_steps → staleness tracking
  → 调整remaining max_tokens → 不浪费compute

★ 设计: weight sync→abort→partial output保存→resume→继续生成 → 不丢弃!
  → vs sync: sleep/wake → 全部重新生成 →浪费 → async更高效!
```

## 7. Async vs Sync — 3种Training Mode对比

```
★ ★ 3种mode → 根本不同架构!

| Feature | Sync Colocated | Sync Separated | Fully Async |
|---------|---------------|---------------|-------------|
| GPU sharing | Same, time-mux | Separate GPUs | Separate + hybrid for val |
| Generation | Sequential/step | Sequential/step | Continuous streaming |
| Training | After gen completes | After gen+weight sync | Independent, from queue |
| Staleness | None(on-policy) | 0 or 1 step | Configurable threshold |
| Weight sync | In-process | Cross-GPU/step | Cross-GPU/trigger_step |
| Buffer | None | None | MessageQueue(deque) |
| Rollout abort | Sleep/wake | Sleep/wake | Abort+resume(partial_rollout) |
| gen_batch_size | Full batch | Full batch | ★ 1 (single sample) |
| Correction | None | Optional IS | IS+RS+bypass |

★ ★ 关键差异:
  → Sync: generate→train→generate→train → sequential → 简单但低效
  → Async: generate continuously → train continuously → overlap → 高效但复杂
  → Async需要staleness correction → IS/RS → bypass_mode最简单 → 但有理论保证?

★ ★ RTX 4090 implications:
  → 单GPU → 只能sync colocated → hybrid_engine=True → naive generator
  → 2+GPU → fully async → generation-training overlap → 但staleness correction必需
  → 8×RTX 4090 PCIe → NCCL weight sync → 12GB/s → 比NVLink慢 → async overhead高
```

## 8. Weight Sync in Async Mode

```
★ ★ _fit_update_weights() (501-536): trainer→rollouter

1. only if local_trigger_step == 1 → 每trigger_parameter_sync_step步一次
2. checkpoint_manager.update_weights() → NCCL/NIXL push weights to rollout
3. rollouter.reset_staleness.remote() → RPC → 重置staleness → resume generation
4. rollout replicas: abort_all_requests → 中断inflight → resume_generation

★ ★ Trainer-side Validation (557-596): 3-phase
  1. Wake hybrid replicas → update_weights → add_replicas → resume_generation
  2. Run validation
  3. Sleep hybrid replicas → abort_replicas → remove_replicas → sleep_replicas
```

## 9. Ray Actor Topology

```
★ ★ 完整Ray actor拓扑:

FullyAsyncTaskRunner (orchestrator)
  ├── FullyAsyncRollouter
  │     ├── FullyAsyncLLMServerManager
  │     │     ├── GlobalRequestLoadBalancer(@ray.remote, sticky session)
  │     │     ├── RolloutReplica (per GPU node)
  │     ├── FullyAsyncAgentLoopManager
  │     │     ├── AgentLoopWorker (Ray actors)
  │     ├── RewardLoopManager (if use_rm)
  │
  ├── FullyAsyncTrainer
  │     ├── Actor WorkerGroup (RayWorkerGroup, FSDP)
  │     ├── CheckpointEngineManager (weight sync backend)
  │     ├── Hybrid CheckpointEngineManager (validation)
  │
  ├── MessageQueue (@ray.remote, num_cpus=2, max_concurrency=20)

★ 协调机制:
  → Trainer→Rollouter: ray.get() RPC → reset_staleness/add_replicas/remove_replicas
  → Rollouter→MessageQueue: put_sample → async
  → Trainer→MessageQueue: get_sample → blocking wait
  → GlobalRequestLoadBalancer: add_servers/remove_servers → elastic scaling
```

## 10. 关键设计洞察

```
1. gen_batch_size=1 → 极细粒度 → overlap最大化!
   → Sync: 整batch → trainer等batch完成 → GPU空闲 → 浪费!
   → Async: 单sample → trainer随时取可用样本 → overlap自然!
   → 这是async的核心优势 → 细粒度 → 更高GPU利用率!

2. MessageQueue → deque + FIFO overflow → 自然staleness管理!
   → queue满 → 旧样本自动丢弃 → trainer不处理极旧数据 → 安全!
   → max_queue_size = staleness_threshold相关 → 阈值控制overflow → 设计精妙!
   → vs sync: 无buffer → 无staleness → 但无overlap → 低效!

3. Partial Rollout Resume → 不丢弃 → 不浪费!
   → weight sync中断 → partial output保存 → resume → 继续生成
   → vs sync sleep/wake → 全部重新生成 → 浪费compute → async更优!
   → 但: partial_rollout增加complexity → 多weight version → staleness tracking更复杂

4. 3种IS/RS Correction → 从简单到复杂 → bypass最实用!
   → bypass: rollout_log_probs = old_log_probs → PPO ratio自动 → 最简单
   → TIS: token-level IS weights → 更精确但更复杂
   → Geo-RS/K3-RS: rejection sampling → 最精确但最慢
   → RTX 4090单GPU: 不需要correction(sync on-policy) → bypass只用于async多GPU

5. 3种Training Mode → 场景决定选择!
   → 单GPU RTX 4090 → sync colocated → 简单稳定 → naive zero-copy
   → 2+GPU PCIe → sync separated → NCCL weight sync → 无staleness但通信慢
   → 8×H100 NVLink → fully async → NIXL RDMA → maximum overlap → 但需correction
   → ★ 选择取决于GPU数量和interconnect!

6. CheckpointEngineManager in Async → 每trigger_step一次 → 不是每step!
   → Sync: 每step weight sync → overhead高 → 但无staleness
   → Async: 每trigger_parameter_sync_step一次 → 省sync overhead → 但有staleness
   → trigger_step可配置 → 小=更on-policy但更overhead → 大=更off-policy但更高效

7. GlobalRequestLoadBalancer → sticky session → 请求路由 → 动态scaling!
   → add_servers/remove_servers → validation期间弹性scaling
   → sticky session → 同一request始终路由到同一replica → KV cache复用!
   → vs sync: 固定replica → 无弹性 → 简单但不灵活

8. Async vs Sync → 根本架构不同 → 不是简单开关!
   → Sync: RayPPOTrainer单actor → 14步sequential → 简单但低效
   → Async: 3个independent Ray actors → concurrent → 高效但复杂
   → 选择是架构级决策 → 不是config开关 → 需要不同代码路径!
```

---

Sources:
- verl/experimental/fully_async_policy/fully_async_rollouter.py (FullyAsyncRollouter)
- verl/experimental/fully_async_policy/fully_async_trainer.py (FullyAsyncTrainer)
- verl/experimental/fully_async_policy/fully_async_main.py (TaskRunner orchestrator)
- verl/experimental/fully_async_policy/message_queue.py (MessageQueue + Client)
- verl/experimental/fully_async_policy/detach_utils.py (RolloutSample + assemble)
- verl/trainer/ppo/rollout_corr_helper.py (IS/RS correction)
- verl/workers/rollout/llm_server.py (GlobalRequestLoadBalancer)
- Background agent research (verl async rollout)
