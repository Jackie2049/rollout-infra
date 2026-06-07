# Distributed RL Training Architecture for LLM — verl Deep Dive

> 2026-06-07 | verl RayPPOTrainer 源码分析: 混合引擎+colocation+sleep/wake+多后端rollout

## 一、verl 分布式架构总览

### 单控制器 + Ray WorkerGroup

verl 采用 **单控制器(Single Controller)** 架构:
- **Driver进程**: 运行`RayPPOTrainer`, 调度整个训练loop, 只做轻量计算(advantage)
- **WorkerGroup**: Ray远程进程, 执行重计算(rollout/training/reward)
- **ResourcePoolManager**: 定义GPU资源池, 支持多pool(colocation)

```
Driver (CPU)
  ├── RayPPOTrainer.fit() — 主循环调度
  │     ├── actor_rollout_wg.generate_sequences() — 生成rollout
  │     ├── actor_rollout_wg.compute_log_prob() — 重算log_prob
  │     ├── critic_wg.compute_values() — 计算V(s)
  │     ├── critic_wg.update_critic() — 更新critic
  │     ├── actor_rollout_wg.update_policy() — 更新策略
  │     └── checkpoint_manager.update_weights() — 权重同步
  │
  └── ResourcePoolManager
        ├── pool_actor_rollout_ref: [GPU 0-3] — actor+rollout+ref colocation
        ├── pool_critic: [GPU 4-5] — critic独立
        └── pool_reward: [GPU 6-7] — reward独立
```

### Colocation策略

`create_colocated_worker_cls()` 将多个模型合并到同一WorkerGroup:
- **FSDP后端**: `max_colocate_count=3` → actor_critic_ref合并为一个WorkerGroup
- **Megatron后端**: `max_colocate_count>1` → 不同模型用不同WorkerGroup

**GRPO最简配置**: actor+ref+rollout = 1个pool → 2模型(GRPO无critic) → **节省50%GPU**

```python
# WorkerDict: monkey-patch方法到单一Worker类
class WorkerDict(worker_cls):
    def __init__(self):
        self.worker_dict = {}
        for key, user_defined_cls in cls_dict.items():
            self.worker_dict[key] = user_defined_cls(*args, **kwargs)
```

## 二、RayPPOTrainer.fit() 主循环

### 完整14步Pipeline (ray_trainer.py, line 1362-1761)

| 步骤 | 代码位置 | 操作 | GPU |
|------|---------|------|-----|
| 1 | line 1448 | `gen_batch.repeat(n)` — 重复prompt n次 | CPU |
| 2 | line 1470 | `generate_sequences()` — rollout生成 | GPU(rollout) |
| 3 | line 1471 | `sleep_replicas()` — 释放rollout权重 | GPU(rollout) |
| 4 | line 1518-1526 | `extract_reward()` — reward计算 | GPU(reward)或CPU |
| 5 | line 1542-1558 | `compute_old_log_prob()` — 重算old_log_prob | GPU(actor) |
| 6 | line 1578-1580 | `compute_ref_log_prob()` — ref策略KL | GPU(ref) |
| 7 | line 1584-1586 | `compute_values()` — critic V(s) | GPU(critic) |
| 8 | line 1597-1603 | `apply_kl_penalty()` — KL惩罚 | CPU |
| 9 | line 1625-1633 | `compute_advantage()` — advantage估计 | CPU(driver) |
| 10 | line 1637-1638 | `update_critic()` — critic训练 | GPU(critic) |
| 11 | line 1648-1649 | `update_actor()` — actor训练 | GPU(actor) |
| 12 | line 1674-1675 | `update_weights()` — 权重同步到rollout | GPU(actor→rollout) |
| 13 | line 1686-1693 | `_validate()` — 验证 | GPU |
| 14 | line 1750 | `logger.log()` — metrics记录 | CPU |

### 关键设计

1. **rollout_n**: `gen_batch.repeat(repeat_times=rollout_n)` → 每prompt生成n个response(GRPO核心)
2. **Bypass vs Decoupled**: bypass模式用`rollout_log_prob`代替重算 → 省一次forward
3. **balance_batch**: `_balance_batch()` 确保DP rank间token数均衡
4. **REMAX**: 需要额外greedy baseline → `gen_batch + gen_baseline_batch`合并

## 三、Sleep/Wake机制 — 权重内存优化

### 3种Rollout模式

| 模式 | sleep行为 | wake_up行为 | 适用场景 |
|------|----------|------------|---------|
| **HYBRID** | 释放权重+KV cache | 重载权重+重置prefix cache | actor/rollout不同GPU |
| **COLOCATED** | engine.sleep(level=1) | engine.wake_up() | actor/rollout同一GPU |
| **STANDALONE** | 跳过sleep | 跳过wake_up | rollout独占GPU |

### vLLM sleep/wake实现 (vllm_async_server.py)

```python
async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        await self._sleep_hybrid()  # 释放所有GPU内存
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.sleep(level=1)  # level=1: offload权重到CPU

async def wake_up(self, tags=None):
    if self.rollout_mode == RolloutMode.HYBRID:
        await self.engine.wake_up(tags=tags or self._get_wake_up_tags())
        await self.engine.reset_prefix_cache()  # 清除旧权重对应的KV cache
    elif self.rollout_mode == RolloutMode.COLOCATED:
        await self.engine.wake_up(tags=self._get_wake_up_tags())
        await self.engine.reset_prefix_cache()
```

**vLLM 3级sleep**:
- **Level 0**: 暂停调度 → 请求排队等待
- **Level 1**: offload权重到CPU → 释放GPU内存给训练
- **Level 2**: 丢弃全部GPU内存(权重+KV+activation) → 极端内存释放

### SGLang sleep/wake (async_sglang_server.py)

```python
async def sleep(self):
    if self.lora_as_adapter:
        tags = ["kv_cache"]  # LoRA: 只释放KV cache, 保留base weights
    else:
        tags = ["kv_cache", "weights"]  # 全量: 释放权重+KV

async def wake_up(self, tags=None):
    # SGLang ReleaseMemoryOccupation + ResumeMemoryOccupation
```

**LoRA特殊处理**: LoRA作为adapter(merge=False) → sleep只释放KV cache → wake_up只需同步adapter delta → 极快!

## 四、Multi-Backend Rollout

### 4种Rollout后端

| 后端 | 实现 | 适用场景 |
|------|------|---------|
| **vLLM async** | `vllm_async_server.py` | verl默认, TP inference |
| **SGLang async** | `async_sglang_server.py` | prefix复用强, agentic |
| **TRT-LLM async** | `trtllm_async_server.py` | 延迟敏感, C++ runtime |
| **Naive (HF)** | `naive_rollout.py` | 单GPU调试, 无TP |

### 异步rollout架构

```
LlmServerManager (driver端)
  ├── generate_sequences() → 发请求到async server
  ├── sleep_replicas() → 释放GPU内存
  ├── wake_up_replicas() → 恢复GPU内存+同步权重
  └── abort_all_requests() → 取消生成
```

每个rollout server是独立的Ray Actor:
- vLLM: `AsyncLLM` → `EngineCoreProc` → 双进程ZMQ IPC
- SGLang: `TokenizerManager` → `Scheduler` → Mixin事件循环
- 都支持异步batched generation → 多请求并行

## 五、并行策略

### FSDP后端 (verl default)

```
DP=4, FSDP:
  ├── WorkerGroup (DP rank 0-3)
  │     ├── ActorRolloutRefWorker (colocated)
  │     │     ├── actor: FSDP shard训练
  │     │     ├── rollout: vLLM TP inference
  │     │     └── ref: FSDP shard推理
  │     └── CriticWorker (可选)
```

- **训练**: FSDP(ZeRO-3) → 参数分片 → DP通信
- **推理**: vLLM/SGLang → TP → 权重聚合 → 高吞吐
- **切换**: sleep/wake → offload训练权重 → 装载推理权重 → 同一GPU

### Megatron后端

```
TP=4, PP=2, DP=2:
  ├── WorkerGroup (TP×PP=8 GPU, 2 replicas)
  │     ├── MegatronActorWorker
  │     ├── MegatronCriticWorker
  │     ├── MegatronRewardWorker
  │     └── MegatronRolloutWorker
```

- **训练**: Megatron TP+PP+DP → 通信重叠(1F1B)
- **推理**: vLLM/SGLang TP → 不同WorkerGroup
- **并行**: max_colocate_count>1 → 多WorkerGroup在同一GPU

## 六、Prefix Sharing在verl中的集成路径

### 当前PrefixGrouper (PR #4368已merge)

- **位置**: `prefix_grouper_utils.py` → monkey-patch `ALL_ATTENTION_FUNCTIONS`
- **机制**: 只拦截attention层 → Provider正常forward → Reuser用padded prefix做forward
- **局限**: MLP/QKV_proj无节省 → 0.99x加速(长序列) vs full-model 2.46x

### Full-Model PS集成方案

**核心挑战**: Full-model PS需要model-level修改, 不能仅monkey-patch attention

**方案A: 两遍PS (prefix-0501验证)**:
```
Pass 1 (Provider): 完整forward所有层 → 存储KV+DeltaNet state
Pass 2 (Reuser): suffix-only forward → 注入KV+state → block-causal attention
```
- 优点: DeltaNet+全attention完整支持, 精度验证通过(cos_sim=0.999973)
- 缺点: 两次forward → rollout阶段额外开销

**方案B: 单遍PS (MegatronIntegration)**:
```
一次forward中, 逐序列处理:
  Provider: 正常attention → 存储KV
  Reuser: 加载KV → block-causal attention
```
- 优点: 一次forward, 更简单
- 缺点: 不支持verl model classes, 不支持DeltaNet state injection

**方案C: Hybrid (推荐)**:
```
Rollout阶段: 使用vLLM/SGLang rollout → prefix caching自然支持
Training阶段: 使用full-model PS → Provider+Reuser模式
```
- 优点: rollout用serving引擎(prefix cache零成本), training用PS(MLP savings)
- 缺点: 两个引擎需要权重同步

### 实际集成步骤 (verl #6401)

1. **Phase 1**: PrefixGrouper改进 → 加MLP层skip → 预估1.87x training speedup
2. **Phase 2**: Magi Attention backend → prefix tree + sparse attention
3. **Phase 3**: Full-model PS → model-level修改 → 2.5x forward speedup

**关键**: verl的hybrid engine架构天然支持PS → sleep释放rollout权重 → actor训练用PS → wake恢复rollout

## 七、多节点扩展

### DeepSeek-R1规模 (n=64 GRPO)

```
64 GPU集群:
  ├── 32 GPU: actor_rollout_ref (DP=8, TP=4)
  │     ├── actor: FSDP训练 (DP=8)
  │     ├── rollout: vLLM TP=4 × 8 replica
  │     └── ref: FSDP推理 (DP=8)
  ├── 16 GPU: critic (如果PPO, GRPO不需要)
  ├── 16 GPU: reward (如果RM, 规则reward不需要)
```

**GRPO节省**:
- 无critic → 节省16 GPU → 50%资源节省
- 规则reward(math/code验证) → 节省16 GPU → 进一步50%
- 总计: 4模型→2模型 → 64 GPU→32 GPU → **成本降低50%**

**PS加速 (n=64)**:
- prefix=94% → 10.2x forward speedup → 7.7x training speedup(×0.76打折)
- 全集群: 32 GPU → 4 GPU等效 → **成本降低87%**

## 八、关键数据流

### GRPO数据流 (2模型)

```
Prompt × n → rollout.generate() → [n responses per prompt]
  → extract_reward() → [r_i per response]
  → compute_old_log_prob() → [log π_θ(y|x)]
  → compute_ref_log_prob() → [log π_ref(y|x)]  (ref_in_actor=True时免费)
  → apply_kl_penalty() → [r_i - β KL]
  → compute_advantage(GRPO) → [(r_i - μ_group)/σ_group]
  → update_actor() → [PPO clip with GRPO advantage]
  → update_weights() → [sync π_θ to rollout]
```

### PPO数据流 (4模型)

```
Prompt × n → rollout.generate() → [n responses per prompt]
  → extract_reward() → [r_i per response]
  → compute_old_log_prob() → [log π_θ(y|x)]
  → compute_ref_log_prob() → [log π_ref(y|x)]  (独立ref模型)
  → compute_values() → [V(s_t) per token]  (独立critic模型)
  → apply_kl_penalty() → [r_i - β KL]
  → compute_advantage(GAE) → [Σ(γλ)^l δ_{t+l}]
  → update_critic() → [critic训练]
  → update_actor() → [PPO clip with GAE advantage]
  → update_weights() → [sync π_θ to rollout]
```

## 九、性能瓶颈分析

### GRPO训练步时间分解

| 阶段 | 占比 | 优化方向 |
|------|------|---------|
| Rollout生成 | 15-20% | prefix cache, PS |
| Reward计算 | 5-10% | 规则reward(CPU), batched RM |
| Log prob重算 | 20-25% | bypass mode省一步 |
| Actor训练 | 30-40% | PS forward, gradient accumulation |
| 权重同步 | 5-10% | NCCL优化, sleep/wake |
| Advantage计算 | <1% | CPU, 已最优 |

**最大瓶颈**: Actor训练(30-40%) → PS可节省MLP compute → **1.87x total speedup**

Sources:
- verl ray_trainer.py: verl/verl/trainer/ppo/ray_trainer.py (1770行)
- verl core_algos.py: verl/verl/trainer/ppo/core_algos.py (2487行)
- ResourcePoolManager: verl/verl/single_controller/ray/base.py (line 182)
- create_colocated_worker_cls: verl/verl/single_controller/ray/base.py (line 986)
- vLLM sleep/wake: verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py (line 604-634)
- SGLang sleep/wake: verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py (line 466)
- engine_workers.py: verl/verl/workers/engine_workers.py (TrainingWorker)
- prefix-0501两遍PS: prefix-0501/scripts/run_ps_e2e_twopass_v2.py (cos_sim=0.999973验证通过)