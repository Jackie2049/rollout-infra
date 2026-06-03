# verl (HybridFlow) 架构深度分析

> 基于 verl 源码的架构阅读笔记，重点关注 RL 训练框架设计、vLLM 集成和前缀缓存机制

## 1. 整体架构

verl 是一个面向 RL 训练（PPO/GRPO 等）的分布式框架，核心设计理念：

- **Single Controller 模式**：由一个主控进程编排所有工作流
- **ActorRolloutRef 合体设计**：Actor、Rollout、Ref 模型在同一组 GPU 上复用
- **可插拔 Rollout 引擎**：支持 vLLM、SGLang、TensorRT-LLM
- **DataProto 统一数据协议**：所有组件间通过 DataProto 交换数据

### 核心组件关系

```
                    RayWorkerGroup
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ActorRolloutRef   CriticWorker   RewardWorker
   (训练+推理+参考)    (价值估计)     (奖励计算)
          │
    ┌─────┼─────┐
    │     │     │
  Actor  Rollout Ref
  (训练) (推理) (参考)
         │
    vLLM/SGLang/TRT-LLM
    (推理引擎)
```

## 2. DataProto — 统一数据协议

**文件**: `verl/protocol.py`

```python
class DataProto:
    batch: dict[str, torch.Tensor]       # 张量数据
    non_tensor_batch: dict[str, Any]     # 非张量数据（token IDs, masks 等）
    meta_info: dict[str, Any]            # 元数据
```

关键方法：
- `chunk(n)` — 分割为 n 个子 DataProto
- `concat(list)` — 拼接多个 DataProto
- `to(device)` — 设备迁移
- `fold_batch_dim()` / `unfold_batch_dim()` — 批次维度折叠/展开
- `split(batch_size)` — 按批次大小切分

DataProto 是 verl 中所有数据流动的载体，从 rollout 生成的 trajectories 到训练时的 mini-batch 都用它。

## 3. ActorRolloutRefWorker — 核心 Worker

**文件**: `verl/workers/engine_workers.py`

这是 verl 最关键的设计：将 Actor（训练）、Rollout（推理）、Ref（参考模型）三合一。

### 3.1 角色组合

```python
# 支持的角色组合
roles = ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
```

- **actor_rollout_ref**: 最常用，Actor/Rollout/Ref 共享 GPU
- 好处：权重同步零拷贝（同一进程内的参数更新立即可见）
- Rollout 时用推理引擎（vLLM），训练时用训练框架（FSDP/Megatron）

### 3.2 关键方法

| 方法 | 功能 |
|------|------|
| `compute_log_prob` | Actor 计算当前策略的 log probability |
| `compute_ref_log_prob` | Ref 模型计算参考策略的 log probability |
| `update_actor` | PPO Actor 更新 |
| `update_weights` | 从训练引擎同步权重到推理引擎 |

### 3.3 权重同步机制

```
训练引擎 (FSDP/Megatron) 权重更新
        │
        ▼
update_weights()  ← 权重同步
        │
        ▼
推理引擎 (vLLM) 加载新权重 → 生成新的 rollout
```

## 4. Rollout 引擎集成

**文件**: `verl/workers/rollout/`

### 4.1 架构层次

```
BaseRollout (抽象基类)
    │
    ├── vllm_rollout/ServerAdapter    ← 异步 HTTP 模式
    ├── sglang_rollout/               ← SGLang 后端
    └── trtllm_rollout/               ← TensorRT-LLM 后端
```

### 4.2 vLLM ServerAdapter

**文件**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

verl 使用 vLLM 作为推理引擎的方式有两种：
1. **嵌入式**：直接调用 vLLM Python API（旧模式）
2. **ServerAdapter**：启动 vLLM HTTP Server，通过 API 交互（新模式，推荐）

ServerAdapter 核心流程：

```python
class ServerAdapter:
    def __init__(self):
        # 启动 vLLM HTTP Server
        self.server = vLLMHttpServer(model, ...)
        # 权重传输器
        self.weight_sender = BucketedWeightSender()

    def generate_sequences(self, prompts):
        # 通过 HTTP 发送生成请求
        return self.server.generate(prompts)

    def update_weights(self, model_state_dict):
        # 分桶传输权重到 vLLM
        self.weight_sender.send(model_state_dict)
```

### 4.3 权重传输：BucketedWeightSender

```
训练引擎参数 → 按层分桶 → HTTP/RPC 传输 → vLLM 热更新
```

避免一次性传输整个模型权重，按层分桶传输减少内存峰值。

## 5. 三级前缀缓存机制

**这是 verl 最独特的设计，也是与 prefix-sharing 项目最相关的特性。**

### 5.1 第一级：vLLM 推理层前缀缓存

```yaml
# vLLM 配置
enable_prefix_caching: True
```

- vLLM 的 PagedAttention 支持 prefix caching（Merkle Hash Chain）
- 相同前缀的 KV Cache 块在请求间复用
- 对 RL 训练中相同 prompt 前缀的 rollout 生成直接加速

### 5.2 第二级：负载均衡器 Sticky Sessions

```
请求路由:
  相同 prompt hash → 路由到同一个 vLLM 实例
                     → 复用该实例上的 KV Cache
```

- 当部署多个 vLLM 实例时，负载均衡器使用 sticky session
- 相同前缀的请求总是路由到同一个 GPU
- 确保 KV Cache 前缀复用的命中率

### 5.3 第三级：训练侧 PrefixGrouper

**文件**: `verl/trainer/ppo/prefix_grouper_utils.py`

```python
# 核心功能
class PrefixGrouper:
    """将共享前缀的序列分组，在训练时复用前缀的计算结果"""

    def build_position_ids_for_prefix_grouper():
        """为分组后的序列构建 position IDs"""
        # 共享前缀只计算一次
        # 各序列的不同后缀分别计算

    def build_pg_from_micro_batch():
        """从 micro-batch 构建 prefix group"""
```

训练侧的优化：
- 将共享前缀的 samples 在一个 batch 中分组
- 前向计算时前缀部分只做一次，各样本的 unique suffix 分别计算
- 减少训练时的重复计算

### 5.4 三级缓存协同

```
Prompt: [系统提示 | 用户输入]

推理时 (vLLM):
  ┌─────────────────────────┐
  │ 系统提示 KV Cache 缓存   │ ← 第1级：vLLM prefix caching
  │ 只计算一次，多请求复用    │
  └─────────────────────────┘
  路由到同一 GPU            ← 第2级：sticky session

训练时 (PrefixGrouper):
  ┌───────────┬───────────┐
  │ 共享前缀   │ 后缀 A    │
  │ (计算1次)  │ 后缀 B    │ ← 第3级：训练侧分组
  │           │ 后缀 C    │
  └───────────┴───────────┘
```

## 6. PPO 训练主循环

**文件**: `verl/trainer/main_ppo_sync.py`, `verl/trainer/ppo/ray_trainer.py`

### 6.1 Ray Trainer 流程

```python
# 简化的 PPO 主循环
for epoch in range(num_epochs):
    # 1. 生成 rollout
    trajectories = actor_rollout.generate(prompts)

    # 2. 计算奖励
    rewards = reward_model.compute(trajectories)

    # 3. 计算参考 log prob
    ref_log_probs = actor_rollout.compute_ref_log_prob(trajectories)

    # 4. 计算优势
    values = critic.compute_values(trajectories)
    advantages = compute_gae(rewards, values)

    # 5. PPO 更新
    for ppo_epoch in range(num_ppo_epochs):
        # Actor 更新
        actor_rollout.update_actor(trajectories, advantages, ref_log_probs)
        # Critic 更新
        critic.update_critic(trajectories, values, rewards)

    # 6. 同步权重到推理引擎
    actor_rollout.update_weights()
```

### 6.2 同步模式 vs 异步模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| Sync | Actor 和 Rollout 共置，零拷贝权重同步 | 单节点，GPU 资源充足 |
| Async | Rollout 独立部署，通过 RPC 通信 | 多节点，大规模 |

## 7. 分布式执行模型

### 7.1 Ray 单控制器

**文件**: `verl/single_controller/ray/base.py`

```python
class RayWorkerGroup:
    """管理一组 Ray Actor Worker"""
    def __init__(self):
        self.workers = []  # Ray Actor 列表

    def dispatch(self, data_proto):
        """将 DataProto 分发到各 worker"""
        chunks = data_proto.chunk(len(self.workers))
        futures = [w.process(chunk) for w, chunk in zip(self.workers, chunks)]
        return futures

    def collect(self, futures):
        """收集各 worker 的结果"""
        results = ray.get(futures)
        return DataProto.concat(results)
```

### 7.2 资源管理

```python
class ResourcePoolManager:
    """管理 GPU 资源池，分配 worker 到 GPU"""
    def create_colocated_worker_cls():
        """创建共置 worker（多个角色共享 GPU）"""
```

## 8. 关键文件索引

| 文件 | 内容 |
|------|------|
| `verl/protocol.py` | DataProto 定义 |
| `verl/workers/engine_workers.py` | ActorRolloutRefWorker, TrainingWorker |
| `verl/workers/rollout/base.py` | Rollout 基类与工厂函数 |
| `verl/workers/rollout/vllm_rollout/` | vLLM 推理引擎集成 |
| `verl/trainer/ppo/ray_trainer.py` | PPO Ray 训练器 |
| `verl/trainer/ppo/prefix_grouper_utils.py` | 训练侧前缀分组优化 |
| `verl/trainer/main_ppo_sync.py` | 同步 PPO 训练入口 |
| `verl/single_controller/ray/base.py` | Ray WorkerGroup 与资源管理 |
| `verl/single_controller/base/worker.py` | Worker 基类 |

## 9. 学习要点

1. **ActorRolloutRef 三合一** — 避免权重跨进程传输，训练和推理共享 GPU
2. **DataProto 统一协议** — 类似 gRPC protobuf 的设计，但面向 PyTorch 优化
3. **可插拔 Rollout 后端** — 通过工厂模式支持 vLLM/SGLang/TRT-LLM 切换
4. **三级前缀缓存** — 推理层（vLLM）+ 路由层（sticky session）+ 训练层（PrefixGrouper）
5. **Ray 单控制器** — 简化分布式编排，与 vLLM 的多进程模型互补
6. **权重同步** — BucketedWeightSender 分桶传输，减少内存峰值

## 10. 与 prefix-sharing 项目的关系

verl 的三级前缀缓存机制直接对应 prefix-sharing 的核心需求：
- **第1级**：推理时 KV Cache 复用（vLLM 原生支持）
- **第3级**：训练时前缀计算复用（PrefixGrouper）
- 中间的路由优化（第2级）是系统层面的粘合

要深入理解 prefix-sharing，可以：
1. 研究 `prefix_grouper_utils.py` 的具体实现
2. 分析 vLLM 的 prefix caching 与 verl 的集成方式
3. 理解 Prompt 结构如何影响三级缓存的命中率

## 参考

- [verl GitHub](https://github.com/volcengine/verl)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
