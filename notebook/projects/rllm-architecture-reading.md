# rLLM 源码阅读: Agentic RL for LLMs

> 2026-06-15 | 源码分析 + README + 文档
> GitHub: https://github.com/rllm-org/rllm (5.6K stars, Apache-2.0)
> 论文: "rLLM: A Framework for Post-Training Language Agents" (Berkeley Sky Lab)
> 作者: Sijun Tan, Michael Luo, Ion Stoica 等 (Ray/vLLM 团队)

## 1. 定位与核心概念

rLLM = **Agentic RL on any harness, with any backend, on any benchmark**

不是 "Relational LLM"，而是 **Reinforcement Learning for Language Agents**:
- 训练语言智能体(agent)的 RL 框架
- 核心创新: 同一份 agent 代码同时驱动 eval 和 training
- 与 verl 互补: verl=分布式训练引擎, rLLM=agent+rollout+eval编排层

### 与 verl 的关系

| 维度 | rLLM | verl |
|------|------|------|
| 角色 | 编排层(rollout+eval+reward) | 训练引擎(policy update) |
| Agent | 任意代码(@rllm.rollout) | 无agent概念 |
| Backend | verl/tinker/fireworks | 仅自身 |
| Benchmark | 60+内置 | 无 |
| Sandbox | Docker/Daytona/Modal | 无 |
| 关系 | rLLM可选verl作为backend | verl是rLLM的一个backend |

→ rLLM 是 verl 的**上层封装**，类似 Ray RLlib 对 Ray 的关系

## 2. 架构

### 2.1 核心Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Your Agent  │───▶│    Traces     │───▶│   Rewards    │───▶│  RL Update   │
│  (any code)  │    │  (auto-logged)│    │ (your logic) │    │  (GRPO etc.) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

关键: agent 代码不变 → rLLM 的 Model Gateway 自动捕获 token IDs + logprobs

### 2.2 内部组件

```
rLLM Framework
    │
    ├── Workflow Engine (编排层)
    │     ├── 运行 N 个并行 agent instances → 收集 rollouts
    │     ├── Episode = 一组 task 的完整轨迹
    │     ├── Trajectory = 一个 agent run 的所有 steps
    │     └── Step = 一个 LLM call (token_ids + logprobs)
    │
    ├── Model Gateway (网关层)
    │     ├── URL-routed sessions → 透明捕获 LLM calls
    │     ├── 捕获 token IDs + logprobs → 构建 Episodes
    │     ├── eval 时: 直接转发到模型服务
    │     └── train 时: 路由到训练 gateway (捕获梯度)
    │
    ├── Transform Pipeline (转换层)
    │     ├── 组 trajectories → advantage 计算
    │     ├── group trajectories by task → 同组对比
    │     └── 处理多 trajectory episodes
    │
    ├── Training Backend (训练层)
    │     ├── verl (分布式多GPU) → vLLM/SGLang rollout
    │     ├── tinker (单机) → 适合小模型/debug
    │     ├── fireworks (Fireworks平台) → 云端训练
    │     └── 切换只需改一个 flag!
    │
    └── Sandbox System (沙箱层)
          ├── Docker (默认)
          ├── Daytona (云端沙箱)
          ├── Modal (按需沙箱)
          └── local (本地)
          └── snapshot + warm-pool → 加速 rollout
```

### 2.3 数据流详解

```
Task (benchmark)
    │
    ├── @rllm.rollout: solve(task, config) → Episode
    │     ├── Trajectory(name="solver", steps=[...])
    │     ├── artifacts={"answer": "..."}  ← 任意输出
    │     └── config.base_url → 自动指向 gateway
    │
    ├── @rllm.evaluator: score(task, episode) → EvalOutput
    │     ├── reward: float ← RL reward
    │     ├── is_correct: bool ← 是否正确
    │     └── signals: [Signal(name, value)] ← 多维度评估
    │
    └── AgentTrainer(backend="verl/tinker/fireworks")
          ├── 收集 Episodes → 组 trajectories → 计算 advantage
          └── GRPO/REINFORCE/RLOO → policy update
```

## 3. 核心抽象

### 3.1 @rllm.rollout 装饰器

```python
@rllm.rollout
def solve(task: Task, config: AgentConfig) -> Episode:
    # 任意代码! 可以调用 OpenAI, LangGraph, 自定义 agent
    client = OpenAI(base_url=config.base_url, api_key="EMPTY")
    response = client.chat.completions.create(...)
    return Episode(
        trajectories=[Trajectory(name="solver", steps=[])],
        artifacts={"answer": answer},
    )
```

**关键**: `config.base_url` 是 rLLM 自动注入的:
- eval → 直接指向模型服务
- train → 指向训练 gateway (透明捕获 token_ids + logprobs)
- **同一份代码, 两种模式** ← 核心创新!

### 3.2 @rllm.evaluator 装饰器

```python
@rllm.evaluator
def score(task: dict, episode: Episode) -> EvalOutput:
    answer = episode.artifacts.get("answer", "")
    is_correct = answer.strip() == task["ground_truth"].strip()
    return EvalOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
```

### 3.3 核心类型

| 类型 | 含义 | 关键字段 |
|------|------|----------|
| Task | 一个 benchmark 问题 | instruction, ground_truth |
| Episode | 一个 task 的完整求解 | trajectories, artifacts |
| Trajectory | 一个 agent run | name, steps |
| Step | 一个 LLM call | token_ids, logprobs |
| AgentConfig | agent 配置 | base_url, model |
| EvalOutput | 评估结果 | reward, is_correct, signals |

## 4. CLI 使用

```bash
# 1. 配置模型
rllm model setup

# 2. 在 benchmark 上 eval
rllm eval gsm8k

# 3. 用 RL 训练
rllm train gsm8k
```

60+ 内置 benchmark:
- Math: AIME, MATH-500, GSM8K
- Code: SWE-bench, Terminal-Bench 2.0
- MCQ: GPQA
- Agentic: SkillsBench
- VLM, translation, search, QA 等

## 5. Harness 系统 (10+ CLI Harnesses)

rLLM 支持 10+ 现成的 agent harness:
- Claude Code
- Codex
- Terminus-2
- mini-swe-agent
- opencode
- Harbor-compatible task dirs
- 自定义: LangGraph, OpenAI Agents SDK, `openai.OpenAI`

→ 或用 `@rllm.rollout` 包装任何 agent

## 6. 训练方法

| 方法 | 说明 |
|------|------|
| GRPO | Group Relative Policy Optimization (verl 内置) |
| REINFORCE | 经典策略梯度 |
| RLOO | REINFORCE with Leave-One-Out (降低 variance) |
| SFT | 监督微调 |
| On-policy Distillation | 从强 teacher 到弱 student |
| Custom | 自定义 loss function |

## 7. 战绩 (Battle-tested)

| 项目 | 结果 |
|------|------|
| DeepScaleR-1.5B | 超越 O1-Preview |
| DeepCoder-14B | O3-mini 级别 |
| DeepSWE-32B | 开源 SWE agent SOTA |
| FinQA-4B | 超越 235B, 接近 Gemini 2.5 Pro |

## 8. 与 7 框架定位对比

```
训练框架:
  DeepSpeed  ──→ 分布式训练(ZeRO/Offload)
  Megatron-LM──→ 3D并行(TP+PP+DP)
  verl       ──→ RL训练引擎(GRPO/PPO)
  rLLM       ──→ Agentic RL编排(rollout+eval+reward)
  PyTorch    ──→ 基础框架(distributed/FSDP2/compile)

推理框架:
  vLLM       ──→ 通用推理(PagedAttention/continuous batching)
  MindIE     ──→ 昇腾推理(ATB+CANN/Ascend 910B/C)
```

rLLM 的独特价值:
1. **同一份代码 eval+train** → 大幅降低开发成本
2. **任意 agent harness** → 不局限于特定 agent 实现
3. **多 backend 切换** → verl/tinker/fireworks 一行切换
4. **60+ benchmark** → 评估即开即用
5. **Sandbox 加速** → snapshot+warm-pool 降低 rollout 成本

## 9. 源码结构 + 深度阅读 (2026-06-15)

### 9.1 顶层结构

```
rllm/
    ├── agents/          ← 向后兼容 shim
    ├── cli/             ← rllm CLI (eval/train/model setup)
    ├── engine/          ← RolloutEngine, AgentFlowEngine, AgentWorkflowEngine
    ├── eval/            ← @rllm.evaluator + EvalOutput/Signal
    ├── gateway/         ← Model Gateway 客户端
    ├── harnesses/       ← 10+ CLI harnesses (Claude Code, Codex, etc.)
    ├── rewards/         ← Reward functions
    ├── sandbox/         ← Docker/Daytona/Modal/local sandbox
    ├── trainer/         ← AgentTrainer + BackendProtocol + 3 backends
    │     ├── algorithms/ ← transform + advantage (GRPO/RLOO/REINFORCE)
    │     ├── verl/      ← verl backend (Ray remote TaskRunner)
    │     ├── tinker/    ← tinker backend (单机)
    │     ├── fireworks/ ← Fireworks platform backend
    │     ├── distill/   ← On-policy distillation
    │     └── backend_protocol.py ← BackendProtocol ABC (6 abstract methods)
    ├── types.py         ← Task/Step/Trajectory/Episode/TrajectoryGroup/AgentFlow/Evaluator
    ├── workflows/       ← Workflow + TerminationReason
    └── tools/           ← ToolCall/ToolOutput

rllm-model-gateway/ (独立子项目!)
    ├── server.py        ← FastAPI app factory + CLI entrypoint
    ├── proxy.py         ← ReverseProxy (请求转发+trace捕获)
    ├── session_router.py ← SessionRouter + RoutingPolicy + StickyLeastLoaded
    ├── session_manager.py ← Session CRUD
    ├── token_accumulator.py ← drift-free multi-turn token tracking
    ├── middleware.py     ← SessionRoutingMiddleware (注入session_id)
    ├── models.py        ← GatewayConfig, WorkerInfo
    └── store/           ← TraceStore (sqlite/memory)
```

### 9.2 核心类型系统 (types.py, 554行)

**5层嵌套数据模型**:

```
Task (benchmark问题)
  → id, instruction, metadata, dataset_dir, sub_dir
  → task_dir 属性 (per-task-dir or shared-verifier)

Step (一个LLM call)
  → 核心: id, input, output, action, reward, done
  → 训练: prompt_ids, response_ids, logprobs, advantage, weight_version
  → chat_completions: 累积消息列表 (用于cumulative token检测)
  → model_output: ModelOutput (backfill prompt_ids/response_ids/logprobs)

Trajectory (一个agent run的所有steps)
  → uid, name, task, steps[], reward, signals
  → is_cumulative(): 检测每step的chat_completions是否前缀递增

Episode (一组task的完整轨迹)
  → id, task, trajectories[], artifacts{}, is_correct, session_id
  → task_id/rollout_idx 属性 (从id split)
  → to_dict/from_dict 序列化

TrajectoryGroup (用于advantage计算的轨迹组)
  → trajectories[], group_id, metadata[], weight_version
  → group_role/task_id 属性 (从group_id split)
  → 同组trajectory对比 → GRPO/RLOO advantage
```

**3个核心Protocol**:

```python
class AgentFlow(Protocol):
    def run(self, task, config) -> Episode | Trajectory | None: ...

class Evaluator(Protocol):
    def evaluate(self, task, episode) -> EvalOutput: ...

class RoutingPolicy(Protocol):
    def select_worker(self, workers, session_id, active_counts) -> WorkerInfo: ...
```

**AgentConfig**: base_url, model, session_uid, sampling_params
→ base_url 由 gateway 自动注入 → eval直连, train路由到gateway

### 9.3 Model Gateway (rllm-model-gateway, 独立子项目)

**核心创新**: 透明捕获 LLM calls 的 token_ids + logprobs

```
Gateway 工作流:
    1. Agent → OpenAI API call → Gateway (FastAPI proxy)
    2. SessionRoutingMiddleware → 注入 session_id → session_router.route()
    3. StickyLeastLoadedPolicy → LRU cache(session_id→worker) + least-loaded fallback
    4. ReverseProxy → 转发到 worker (vLLM/SGLang)
    5. response → TokenAccumulator → 捕获 prompt_ids + completion_ids
    6. TraceStore (sqlite/memory) → 存储 trace
    7. /sessions/{id}/traces → trainer 读取
```

**SessionRouter** (session_router.py):
- RoutingPolicy protocol → 可插拔路由策略
- StickyLeastLoadedPolicy: LRU cache + least-loaded fallback
- verl AsyncLLMServerManager._choose_server() 的独立实现
- health_check: 10s interval, 3次失败标记dead, dead可恢复
- active_counts: 每worker活跃请求计数 → least-loaded依据

**TokenAccumulator** (token_accumulator.py, 核心创新!):
- 解决 multi-turn RL 的 tokenization drift 问题
- drift: decode→re-encode 产生不同token IDs → 训练梯度错误
- 解法: 维护 prev_prompt_ids + prev_completion_ids → bridge_to_next_turn
- renderers (https://github.com/PrimeIntellect-ai/renderers) 提供:
  - bridge_to_next_turn(): byte-for-byte prefix-extension invariant
  - 支持: qwen3/qwen3.5/qwen3.6/glm-5/deepseek-v3/gpt-oss
- cumulative_ids = prev_prompt_ids + prev_completion_ids
- is_cumulative(): SHA-256 fingerprint 检测 prefix divergence
- should_rewrite(): turn>0 时用 /v1/completions (pre-tokenized) 替代 /v1/chat/completions

**Cumulative Token Mode** (--cumulative-token-mode):
- 开启后, gateway 用 pre-tokenized prompts 替代 text→tokenize
- 消除多轮对话的 tokenization drift → 训练数据更精确
- 需要 --model (HF checkpoint) + renderers 包
- renderer_family: auto/Qwen3/GLM-5/DeepSeek-V3/GPT-OSS

### 9.4 BackendProtocol (trainer/backend_protocol.py)

**6个abstract methods**:
1. `init_rollout_engine()` → 返回 RolloutEngine
2. `validate_config()` → 配置校验
3. `shutdown()` → 清理资源
4. `generate_episodes()` → 用 workflow engine 生成 episodes
5. `transform_to_backend_batch()` → rllm数据→backend格式
6. `update_policy()` → 更新策略

**async hooks**: on_train_start/end, on_batch_start/end, on_epoch_start/end, on_policy_updated

→ 任何新backend只需实现 BackendProtocol → 一行切换!

### 9.5 Advantage计算 (trainer/algorithms/advantage.py)

**注册式架构**: RLLM_ADV_ESTIMATOR_REGISTRY

| 估算器 | 实现 | 说明 |
|--------|------|------|
| GRPO | `(r - mean) / std` per group | normalize by group std |
| REINFORCE | `advantage = reward` | 无baseline |
| REINFORCE++ | `(r - mean) / (batch_std + ε)` | group mean + batch whitening |
| RLOO | leave-one-out baseline | 降低variance |
| Custom | `register_rllm_adv_estimator()` | 自定义 |

**优势计算流程**:
1. Episodes → `_build_trajectory_groups()` → 按 `task_id:trajectory_name` 分组
2. TrajectoryGroup → `collect_reward_and_advantage_from_trajectory_groups()`
3. 按 group_role 分配 estimator → 计算 advantage
4. advantage broadcast 到每个 step → `step.advantage = float(advantage)`
5. 零variance group → 无梯度 (too_easy/too_hard诊断)

### 9.6 Transform Pipeline (trainer/algorithms/transform.py)

**3步pipeline**:
1. `_impute_trajectory_names()` → 未命名trajectory自动命名
2. `_build_trajectory_groups()` → 按 `task_id:traj_name` 分组 → TrajectoryGroup
3. `_validate_and_propagate_rewards()` → broadcast/per-step reward校验

**Compact Filtering**: 按 termination_reason 过滤 episodes (too_easy/too_hard等)

### 9.7 AgentTrainer (trainer/agent_trainer.py)

- 目前只支持 `backend="verl"`
- workflow_class 必需 (旧的 agent_class/env_class 已移除)
- _train_verl(): ray.init → TaskRunner.remote() → runner.run.remote()
- tinker backend 用 unified_trainer.AgentTrainer

### 9.8 VerlBackend (trainer/verl/verl_backend.py, 880行)

**深度集成verl的实现**: 两种worker部署模式

| 模式 | colocated (混合引擎) | separated (异步训练) |
|------|---------------------|----------------------|
| 训练GPU | Actor+Rollout+Ref共享 | Actor+Ref独占训练GPU |
| Rollout | AgentLoopManager(同GPU) | FullyAsyncAgentLoopManager(独立GPU) |
| Weight Sync | checkpoint_manager.update_weights(每batch) | on_policy_updated(异步pipeline) |
| 优势 | 资源共享→成本低 | 训练不阻塞rollout→吞吐高 |

**CustomPPOLoss**: 包装verl的ppo_loss，支持per-call loss mode override
→ `policy_loss_mode_override` 字段 → 同一batch不同role用不同loss

**Rollout Correction** (重要性采样):
- bypass_mode=True: π_old=π_rollout(直接用rollout logprobs → 无IS权重)
- TIS mode: π_old≠π_rollout → importance sampling weights → log_ratio = old_log_probs - rollout_log_probs

**Batch Processing Pipeline**:
1. interleave_tasks(batch, rollout.n) → 每个task重复N次
2. _execute_tasks_async(tasks, task_ids, engine) → agent workflow执行
3. sleep_replicas() → 释放KV cache内存(colocated模式)
4. transform_episodes_to_dataproto → rllm→verl数据格式
5. pad_dataproto_to_world_size → DP对齐
6. balance_batch → 跨DP rank平衡token数
7. compute_log_prob → old_log_probs + entropy
8. compute_ref_log_prob → ref_log_probs(LoRA: ref_in_actor, 无LoRA: ref_policy_wg)
9. collect_reward_and_advantage → GRPO/RLOO/REINFORCE++
10. update_dataproto_with_advantages → broadcast到每个token
11. update_actor → PPO loss → policy update

**Decoupled Batch Sizes**:
- mini_batch_size = m (ROWS per optimizer step) → 控制update数量
- global_batch_size = gbs (ROLLOUTS per update) → seq-mean loss denominator
- m和gbs解耦 → stable loss scale regardless of merge ratio

### 9.9 GRPO/RLOO实现 (trainer/algorithms/rl_algo.py, 28行)

```python
def calculate_grpo_advantages_per_group(rewards, norm_adv_by_std_in_grpo=True):
    group_mean = np.mean(rewards)
    group_std = np.std(rewards)
    advantages = (rewards - group_mean) / (group_std + epsilon)
    return advantages, advantages

def calculate_rloo_advantages_per_group(rewards):
    # Leave-One-Out baseline: advantage = N/(N-1) * (r - mean)
    advantages = num_trajs / (num_trajs - 1) * (rewards - rewards.mean())
    return advantages, advantages
```

→ 极简实现! GRPO=group normalization; RLOO=leave-one-out baseline

## 10. 下一步

- [x] 克隆 rLLM 源码 ✅ → _temp_rllm/
- [x] 深入阅读核心模块 ✅ → types.py/gateway/transform/advantage
- [ ] 对比 rLLM vs verl 的 rollout pipeline (verl transform vs rllm transform)
- [x] 研究 Model Gateway 的 token capture 实现 ✅ → TokenAccumulator + renderers
- [x] 研究 Transform Pipeline 的 trajectory grouping ✅ → _build_trajectory_groups
- [ ] 研究 Advantage 计算的 rl_algo.py (GRPO/RLOO具体实现)
- [ ] 评估 rLLM 对 RTX 4090 的适用性 (tinker backend)
- [ ] 尝试在本地跑 rLLM eval 一个 benchmark
- [ ] 研究 rllm/workflows/ Workflow 实现

---

Sources:
- [rLLM GitHub](https://github.com/rllm-org/rllm)
- [rLLM Documentation](https://docs.rllm-project.com/)
- [rLLM Website](https://rllm-project.com)
- [DeepScaleR Blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- [DeepSWE Blog](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)
