# rLLM 源码深度阅读: GatewayManager + BackendProtocol + UnifiedTrainer

> 2026-06-15 | 源码级分析: 505+209+1078行
> 源码: _temp_rllm/rllm/gateway/manager.py + trainer/backend_protocol.py + trainer/unified_trainer.py
> 衔接: notebook/projects/rllm-architecture-reading.md (架构总览)

## 1. GatewayManager — 推理服务生命周期管理

### 1.1 两种模式

```
GatewayManager支持两种执行模式:

1. 'process' (subprocess) → verl/distributed训练
   - subprocess.Popen启动rllm-model-gateway CLI
   - 独立进程 → 可跨节点分布式
   - 健康检查: client.health()轮询(30s超时)

2. 'thread' (background thread) → tinker/single-machine/eval
   - uvicorn.Server在daemon线程中运行
   - create_app() + GatewayConfig → FastAPI app
   - 支持local_handler注入(tinker不需HTTP backend!)
```

### 1.2 TinkerEngine in-process handler

```
TinkerEngine特殊处理:
  - 不启动HTTP backend server → 直接注入local_handler
  - create_tinker_handler(rollout_engine) → in-process调用
  - 避免HTTP overhead → 单GPU最快路径!

vs VerlEngine:
  - 注册vLLM server addresses → HTTP转发
  - _ensure_verl_engine_workers → server_id_to_handle.keys()
  - 每个worker注册到gateway → load-balanced
```

### 1.3 Session API + Trace API

```
Gateway核心API:

  生命周期:
    create_session(session_id, is_validation, sampling_params) → session URL
    get_session_url(session_id, public=True) → 基础URL或tunnel URL
    get_traces(session_id) → list[TraceRecord] ← 关键! 捕获logprobs+token IDs

  异步API:
    acreate_session() / aget_traces() / adelete_session() / adelete_sessions()

  Weight Version:
    set_weight_version(weight_version) ← 权重版本号 → rollout引擎同步

  → Trace = rLLM核心创新: 透明捕获推理引擎的logprobs和token IDs!
```

### 1.4 Cumulative Token Mode (Drift-Free Multi-Turn)

```
cumulative_token_mode: bool = gw_cfg.get("cumulative_token_mode", False)
renderer_family: str = gw_cfg.get("renderer_family", "auto")

关键: 消除multi-turn tokenization drift!
  - 传统: 每turn独立tokenize → 分词器状态不同 → token序列不连续
  - 累积模式: prev_prompt_ids + prev_completion_ids → 增量tokenize
  - renderer_family: "qwen3"/"glm-5"/"auto" → 指定分词器家族

→ 这是rLLM对multi-turn RL的关键贡献: 确保token IDs连续一致!
```

### 1.5 Tunnel (Cloudflared)

```
tunnel支持:
  - parse_tunnel() → CloudflaredTunnel或直接URL
  - 用途: sandbox环境(如Docker)需要从外部访问gateway
  - container_reachable_url(): 127.0.0.1 → host.docker.internal

→ 远程sandbox + cloudflared tunnel → rLLM可以跑远程agent!
```

### 1.6 EvalGatewayManager (评估专用)

```python
class EvalGatewayManager(GatewayManager):
    add_logprobs = False        ← 外部provider不支持return_token_ids
    add_return_token_ids = False ← 关闭vLLM专用注入!

    # 不需要rollout_engine → 静态upstream_url
    # 支持OpenAI/Anthropic/LiteLLM → 评估任意模型!
```

## 2. BackendProtocol ABC — 后端抽象基类

### 2.1 6个核心方法

```
BackendProtocol(ABC, Generic[TDataset, TBatch]):

  必须实现:
  1. init_rollout_engine(**kwargs) → RolloutEngine
     初始化rollout引擎(verl→vLLM server, tinker→in-process)

  2. validate_config() → None
     验证后端配置(verl: Ray配置; tinker: GPU配置)

  3. shutdown() → None
     关闭后端资源

  4. generate_episodes(batch, agent_workflow_engine, is_validation) → list[Episode]
     使用workflow引擎生成episodes (async!)

  5. transform_to_backend_batch(trainer_state) → TBatch
     rLLm-native → 后端专用数据格式 (sync)

  6. process_backend_batch(trainer_state) → None
     计算log probs/critic values/forward pass (async!)

  7. compute_advantages(trainer_state, algorithm_config) → None
     有默认实现! → rLLM-native advantage计算
     后端可override → verl有自己的advantage计算

  8. update_policy(trainer_state) → None
     更新策略参数 (async!)
```

### 2.2 6个Hook方法

```
生命周期hooks:
  on_train_start(trainer_state)
  on_train_end(trainer_state)
  on_batch_start(trainer_state)
  on_batch_end(trainer_state)
  on_epoch_start(trainer_state)
  on_epoch_end(trainer_state)

  on_policy_updated(trainer_state) ← 权重同步hook!
  on_validation_start(trainer_state) → bool ← 控制是否继续验证
  on_validation_end(trainer_state)

→ 类似PyTorch Lightning的设计 → 后端可以在任何阶段注入逻辑!
```

### 2.3 Generic设计

```
TDataset = TypeVar("TDataset", bound=Iterable)  ← 后端专用数据集类型
TBatch = TypeVar("TBatch")                      ← 后端专用batch类型

verl backend: TDataset=DataProto dataset, TBatch=DataProto
tinker backend: TDataset=数据集, TBatch=tinker.Datum

→ 不同后端用不同数据类型 → 但训练流程统一!
```

## 3. UnifiedTrainer — 统一训练器

### 3.1 TrainerState 数据类

```python
@dataclass
class TrainerState:
    rs_state: RejectionSamplingState          ← rejection sampling状态
    global_step: int = 0                      ← 全局步数
    epoch: int = 0                            ← epoch数
    total_steps: int = 0                      ← 总步数
    is_training: bool = True                  ← 训练/验证标志
    weight_version: int = 0                   ← 权重版本号(同步用)
    train_dataloader: StatefulTaskDataLoader | None ← 训练数据加载器

    # 训练pipeline数据流
    episodes: list[Episode] | None = None     ← Stage 1输出
    trajectory_groups: list[TrajectoryGroup] | None = None ← Stage 2输出
    backend_batch: Any | None = None          ← Stage 4输出

    # 辅助
    timing_dict: dict                         ← 计时数据
    metrics: dict                             ← 指标数据
    extra_info: dict                          ← 额外信息
```

### 3.2 三种Engine路径

```
UnifiedTrainer.__init__ → 根据输入选择engine:

1. agent_flow + evaluator → AgentFlowEngine (gateway-based, local)
   - GatewayManager(mode="process" if verl, "thread" if tinker)
   - agent_flow=用户提供的agent代码
   - evaluator=用户提供的reward函数
   - n_parallel_tasks/retry_limit → 并发控制

2. remote_runtime → RemoteAgentFlowEngine (gateway-based, remote)
   - RemoteRuntimeConfig(backend="agentcore")
   - 远程sandbox → 云端agent执行
   - session_timeout → 防止agent超时

3. workflow_class → UnifiedWorkflowEngine (direct)
   - 不用gateway → 直接rollout_engine
   - workflow_cls=用户Workflow类
   - 最简单路径 → 单GPU最优!
```

### 3.3 8-Stage训练Pipeline (核心!)

```
_train_batch_async(batch, trainer_state) → 8个阶段:

  Stage 1: generate_episodes (async)
    → backend.generate_episodes(batch, agent_workflow_engine)
    → 使用workflow/agent执行任务 → 收集episodes

  Stage 2: transform_episodes_to_trajectory_groups (sync)
    → prefix-merge + mask + 分组 → TrajectoryGroup[]
    → transform_config + cf_config → 过滤+compact
    → traj_grouping_hook → 自定义分组逻辑

  Stage 3: apply_rejection_sampling_and_filtering (sync)
    → 过滤低reward轨迹 → 只保留高质量样本
    → rs_state跟踪累计过滤状态

  Stage 4: transform_to_backend_batch (sync)
    → backend.transform_to_backend_batch(trainer_state)
    → TrajectoryGroup → DataProto(verl) / tinker.Datum(tinker)

  Stage 5: process_backend_batch (async)
    → backend.process_backend_batch(trainer_state)
    → 计算log probs / critic values / forward pass

  Stage 6: compute_advantages (async)
    → backend.compute_advantages(trainer_state, algorithm_config)
    → GRPO/RLOO/REINFORCE++ → advantage计算

  Stage 7: update_policy (async)
    → backend.update_policy(trainer_state)
    → PPO/GRPO policy gradient → 参数更新

  Stage 8: cleanup + visualization (sync)
    → visualize_trajectory_last_steps → 展示训练轨迹
    → tokenizer解码 → 人可读输出

→ 8阶段流水线 → 每阶段可由不同backend实现 → 统一接口!
```

### 3.4 On-Policy vs Fully-Async

```
两种训练模式:

  _fit_on_policy (同步):
    - 1 batch → 1 rollout → 1 update → 串行
    - 最vanilla → 最简单 → 单GPU首选
    - warm_queue: 预热sandbox → 减少首次启动延迟

  _fit_fully_async (异步):
    - 多个mini_batch并行 → rollout+update异步overlap
    - AsyncTrainingConfig.enable → 开启
    - 更高吞吐 → 但需要更多GPU资源
    - 类似verl的async rollout模式
```

### 3.5 验证Pipeline

```
_validate_async(trainer_state):
  - is_validation=True → 不更新策略
  - sampling_params切换(val_sampling_params)
  - 评估指标记录 → trainer_state.metrics
  - warm_queue临时detach → 验证task不在warm队列中

val_before_train: 默认True → 先验证再训练 → baseline对比!
val_only: 只评估不训练 → rllm eval模式
test_freq: 每N步验证一次 → 定期检查进展
```

## 4. 跨框架对比

| 维度 | rLLM UnifiedTrainer | verl RayPPOTrainer | PyTorch Lightning |
|------|--------------------|--------------------|--------------------|
| 设计 | 8-stage async pipeline | 4-stage sync pipeline | callback-based |
| 后端切换 | BackendProtocol ABC一行切换 | 紧耦合Ray/vLLM | 无RL后端 |
| 数据类型 | Generic[TDataset, TBatch] | DataProto硬编码 | dict/DataLoader |
| Episode→Trajectory | prefix-merge+mask | 单步DataProto | 无RL概念 |
| Rejection Sampling | Stage 3内置 | 无内置 | 无 |
| Async训练 | fully-async模式 | async rollout模式 | 无RL概念 |
| Gateway | 独立gateway进程/线程 | vLLM server内嵌 | 无 |

## 5. RTX 4090实战路径

```
RTX 4090最优rLLM配置:

  Engine: UnifiedWorkflowEngine (direct) → 不用gateway → 最简单!
  Backend: tinker → in-process → 无HTTP overhead → 单GPU最快
  Mode: _fit_on_policy → 串行 → 内存最小
  Algorithm: GRPO → 无需critic模型 → 省内存
  LoRA: rank=16 → 2.6GB → 7B fits RTX 4090
  CPU Adam: offload optimizer → 20.04GB总内存

  具体配置:
    backend_cls = TinkerBackend
    workflow_class = MyWorkflow
    algorithm.estimator = grpo
    rollout.n = 8 → 8轨迹per task → group advantage

  禁用:
    - async训练(需要更多GPU)
    - gateway process模式(不需要分布式)
    - rejection sampling multiplier=1(省rollout次数)
```

## 6. TinkerBackend 深度实现 (450行)

> 源码: _temp_rllm/rllm/trainer/tinker/tinker_backend.py + tinker_policy_trainer.py (452行)

### 6.1 核心架构

```
TinkerBackend(BackendProtocol[Iterable, list[tinker.Datum]]):
  name = "tinker"
  requires_loop = True  ← 使用async操作 → 需event loop

  关键组件:
    service_client: tinker.ServiceClient(base_url) ← Tinker API客户端
    policy_trainer: TinkerPolicyTrainer ← 前向反向+optimizer
    sampling_client: tinker.SamplingClient ← 推理采样(权重同步!)
    rollout_engine: TinkerEngine ← 推理引擎(in-process)
```

### 6.2 LoRA Training Client 初始化

```
TinkerPolicyTrainer.initialize_async():

  从零开始:
    training_client = service_client.create_lora_training_client_async(
      base_model = config.model.name,     ← 如 Qwen3-8B
      rank = config.model.lora_rank,      ← LoRA rank
      train_unembed = True,               ← 训练输出层?
      train_attn = True,                  ← 训练attention层?
      train_mlp = True,                   ← 训练MLP层?
    )
    → 自动创建LoRA adapter → 保存权重 → 返回sampling_client

  恢复训练(resume):
    training_client = service_client.create_training_client_from_state_async(state_path)
    sampling_client = create_sampling_client(sampler_path)
    → 从checkpoint恢复 → 包括dataloader状态!
```

### 6.3 Loss Function Auto-Mapping

```
ADV_TO_LOSS_FN_AUTO_MAP:
  GRPO            → "ppo"             ← PPO loss with clipped advantages
  RLOO            → "importance_sampling" ← IS correction
  REINFORCE       → "importance_sampling"
  REINFORCE++     → "importance_sampling"

TINKER_KNOWN_LOSSES = {"importance_sampling", "ppo", "cispo", "dro", "cross_entropy"}

→ 5种loss function → tinker API提供 → 用户不需要手写!
→ GRPO用PPO loss(clipped) → 不是IS loss → 这与verl不同!
```

### 6.4 Advantage → Forward-Backward 流程

```
TinkerBackend特殊: advantage计算在process_backend_batch内完成!

  compute_advantages → 只是存储algorithm_config!
    self._algorithm_config = algorithm_config

  process_backend_batch → 实际流程:
    1. TinkerPolicyTrainer.forward_backward_from_trajectory_groups():
       a) transform_trajectory_groups_to_datums() ← 优势计算在这里!
       b) _get_forward_backward_futures() → async forward-backward
       c) asyncio.gather(*fwd_bwd_futures) → 等待完成
       d) 提取logprobs + server metrics

    2. 两种模式:
       - 不fused: forward_backward → 优化器步骤分开
       - fused: 融合前向、反向、优化器步骤 → 1次async → 减少round-trip!
```

### 6.5 Mask处理

```
_remove_mask(datum) → 前向反向前移除mask:

  Datum = tinker.Datum(
    model_input = ...,
    loss_fn_inputs = {...所有字段, 不含mask},
  )

→ Mask只在advantage计算时使用 → forward_backward不需要mask
→ tinker API不接受mask参数 → 优势在transform阶段已baked into loss_fn_inputs!
```

### 6.6 Weight Sync机制

```
on_policy_updated → 关键权重同步流程:

  1. policy_trainer.save_checkpoint_and_get_sampling_client(global_step, do_save)
  2. → 返回新的sampling_client ← 权重更新后的推理客户端!
  3. rollout_engine.set_sampling_client(sampling_client) ← 传播到rollout引擎
  4. → 下次rollout使用更新后的权重!

→ tinker的权重同步: 保存→创建新sampling_client→替换rollout的sampling_client
→ 不需要显式weight sync(如verl的vLLM weight sync) → sampling_client自动使用最新权重!
```

### 6.7 Fused Forward-Backward-Optim优化

```
fuse_forward_backward_and_optim_step=True时:

  policy_trainer.fused_forward_backward_and_optim_step(
    step, total_steps, trajectory_groups,
    learning_rate, beta1, beta2, eps
  )
  → 1次async调用 → forward+backward+optimizer→ 一次搞定!

  vs 分步模式:
    1. forward_backward_from_trajectory_groups (async)
    2. 优化器步骤 (async)

→ fused减少1次async round-trip → 单GPU下更高效!
→ 但调试更难 → 无法分开观察fwd/bwd/optim
```

### 6.8 VLM支持

```
VLM自动检测:
  if "vl" in model_name_lower or "vision" in model_name_lower:
    processor = AutoProcessor.from_pretrained(model_name)
    image_processor = processor.image_processor
    → TinkerEngine需要image_processor → 渲染图片内容!

  例: Qwen3-VL → 自动加载image_processor → 支持多模态推理!
```

## 7. 下一步

- [x] GatewayManager深度阅读 → 本文件Section 1
- [x] BackendProtocol深度阅读 → 本文件Section 2
- [x] UnifiedTrainer 8-stage pipeline → 本文件Section 3
- [x] TinkerBackend深度阅读 → 本文件Section 6
  - LoRA training client自动创建 + mask处理 + weight sync(sampling_client刷新)
  - GRPO→PPO loss + RLOO→IS loss → tinker 5种loss function
  - fused forward-backward-optim → 减少async round-trip → 单GPU更高效
  - advantage在process_backend_batch内计算(不是独立stage)
- [ ] 研究 VerlBackend 中的 compute_advantages override
- [ ] 研究 fully-async训练pipeline细节
- [ ] GPU可用时: 运行rLLM eval+train实测

---

Sources:
- `_temp_rllm/rllm/gateway/manager.py` (505 lines)
- `_temp_rllm/rllm/trainer/backend_protocol.py` (209 lines)
- `_temp_rllm/rllm/trainer/unified_trainer.py` (1078 lines)
- `_temp_rllm/rllm/trainer/tinker/tinker_backend.py` (450 lines)
- `_temp_rllm/rllm/trainer/tinker/tinker_policy_trainer.py` (452 lines)
