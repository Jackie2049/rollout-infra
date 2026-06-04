# RLHF/GRPO 训练基础设施 2026 技术全景

> 截至 2026 年的 RLHF 训练基础设施综合分析：框架架构、核心优化、大规模扩展、生产系统

## 1. 引言：RL 训练 Infra 的演进

2024-2026 年，RL for LLM 训练基础设施经历了三波关键变革：

```
第一波 (2024): 四模型 PPO 时代
  Actor + Critic + Ref + RM = 4 个大模型共存
  显存爆炸、训练-推理引擎割裂、分布式编排困难

第二波 (2025): GRPO 简化 + 混合引擎
  移除 Critic → 3 模型甚至 2 模型
  verl (HybridFlow) / OpenRLHF 统一训练和推理
  Prefix Caching / PrefixGrouper 大幅减少冗余计算

第三波 (2026): 万亿参数 + MoE + 异构硬件
  verl 支持 671B MoE (DeepSeek-V3), 万亿参数 LoRA GRPO
  FSDP2 + Megatron 双引擎, NVIDIA/AMD/Ascend 多硬件
  Agent Loop Multi-Turn RL, 可验证奖励替代 RM
```

**核心矛盾**：RL 训练同时需要训练引擎（反向传播、优化器）和推理引擎（自回归生成、KV Cache），二者对 GPU 资源的使用模式截然不同。

## 2. verl 框架架构 (ByteDance Seed, EuroSys 2025)

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                     verl 训练架构 (2026)                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │             SyncPPOTrainer (main_ppo_sync.py)                  │  │
│  │             Driver Process — 单节点 CPU/GPU                     │  │
│  │                                                                │  │
│  │  训练主循环:                                                     │  │
│  │  1. DataLoader → sample prompts + uid                          │  │
│  │  2. RolloutWorker.generate_sequences() → responses             │  │
│  │  3. RewardManager → compute rewards (规则/RM)                  │  │
│  │  4. compute_advantage() → GAE / GRPO / REINFORCE++             │  │
│  │  5. ActorWorker.update_policy() → PPO clip loss                │  │
│  │  6. CriticWorker.update_critic() → value loss (PPO only)       │  │
│  │  7. Sync weights → Rollout engine (vLLM/SGLang)                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                            │ Ray RPC / TransferQueue                  │
│          ┌─────────────────┼──────────────────┐                      │
│          ▼                 ▼                  ▼                      │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ActorRolloutRef│ │   Critic     │ │   Reward     │               │
│  │   Worker      │ │   Worker     │ │   Worker     │               │
│  │              │ │  (PPO only)  │ │              │                │
│  │ Actor (训练) │ │ Critic (训练)│ │ RM / Rules   │                │
│  │ Rollout (推) │ │              │ │              │                │
│  │ Ref (固定)   │ │   [FSDP]    │ │              │                │
│  │ [vLLM+FSDP] │ │  [Megatron] │ │              │                │
│  └───────────────┘ └──────────────┘ └──────────────┘               │
│                                                                      │
│  关键: ActorRolloutRefWorker 混合设计 — 训练+推理共享 GPU            │
│  2026 新增: FSDP2 引擎, Megatron 671B MoE, 万亿参数 LoRA GRPO       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心文件索引

| 文件 | 行数 | 作用 |
|------|------|------|
| `verl/trainer/main_ppo_sync.py` | ~1866 | 推荐 Trainer, TransferQueue 零拷贝 |
| `verl/trainer/main_ppo.py` | ~1200 | 原始 Trainer (deprecated) |
| `verl/trainer/ppo/ray_trainer.py` | ~1770 | RayPPOTrainer 基类 |
| `verl/trainer/ppo/core_algos.py` | ~2488 | GAE/GRPO/REINFORCE++ 等算法 |
| `verl/workers/engine_workers.py` | ~758 | TrainingWorker + ActorRolloutRefWorker |
| `verl/workers/rollout/base.py` | ~400 | BaseRollout + 推理后端工厂 |
| `verl/trainer/ppo/prefix_grouper_utils.py` | ~236 | 训练侧前缀共享 |
| `verl/protocol.py` | ~1346 | DataProto 数据协议 |

### 2.3 ActorRolloutRefWorker — 三合一混合 Worker

这是 verl 最关键的设计创新：将 Actor（训练）、Rollout（推理）、Ref（参考模型）合并到同一个 Worker。

```python
# engine_workers.py
class ActorRolloutRefWorker(Worker):
    """混合 Worker: Actor (训练) + Rollout (推理) + Ref (固定)"""

    def __init__(self, config):
        # 训练引擎 (FSDP/FSDP2/Megatron)
        self.engine = EngineRegistry.new(
            backend=engine_config.strategy,  # "fsdp", "fsdp2", "megatron"
        )

        # Rollout 引擎 (vLLM/SGLang/TRT-LLM)
        self.rollout = get_rollout_class(rollout_config)

        # Reference model (frozen copy)
        if need_reference_policy:
            self.ref_engine = EngineRegistry.new(...)
```

**核心收益**：
- 权重同步零拷贝（同进程内参数更新立即可见）
- 训练和推理通过 sleep/wake 机制时分复用 GPU
- 70B GRPO 在 8 GPU 上可跑（vs PPO 需要 16 GPU）

### 2.4 3D-HybridEngine — 零冗余权重重分片

```
问题: 训练和推理对模型分布有不同最优配置
  训练: TP=2, PP=2, DP=4 (Megatron 3D 并行)
  推理: TP=8 (vLLM 张量并行)

传统解决: 全局 all-gather → 按推理配置重分片 → 巨大显存开销

verl 3D-HybridEngine:
  1. 预计算训练→推理的分片映射表
  2. 直接在 GPU 间传输对应分片 (point-to-point)
  3. 无需全局 all-gather
  4. 零额外显存开销

  开销: 权重重分片 < 5% 训练时间
  论文: HybridFlow (EuroSys 2025) 报告 1.53x-20.57x 吞吐提升
```

### 2.5 DataProto — 统一数据协议

```python
class DataProto:
    """verl 的数据传输协议 — 贯穿所有 Worker"""
    batch: TensorDict        # tensor 数据 (input_ids, log_probs, ...)
    non_tensor_batch: dict    # 非 tensor (uid, ground_truth, reward_fn)
    meta_info: dict           # 元信息 (batch_size, eos_token_id)

    # 支持操作
    def chunk(self, n)        # 分成 n 份 (DP 分片)
    def split(self, sizes)    # 按指定大小分割
    def concat(self, others)  # 合并
    def repeat(self, n)       # 重复 (GRPO 多 response)
    def make_iterator(bs)     # 创建 mini-batch 迭代器
```

DataProtoFuture 是其惰性 Ray future 版本，支持异步执行。

### 2.6 Dispatch 装饰器 — 自动数据分片/收集

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)     # 广播到所有 worker
@register(dispatch_mode=Dispatch.DP_COMPUTE)      # DP 并行计算 + all-reduce
@register(dispatch_mode=Dispatch.MEAN)            # 取所有 worker 均值
@register(dispatch_mode=Dispatch.RANK_ZERO)       # 只在 rank 0 执行
```

装饰器自动处理数据分片和结果聚合，Worker 代码无需感知分布式逻辑。

### 2.7 TransferQueue 零拷贝 (main_ppo_sync)

```
旧模式 (main_ppo): Ray ObjectRef → 序列化/反序列化开销
新模式 (main_ppo_sync): TransferQueue → 零拷贝跨 Worker 传输

Producer (Rollout Worker):
  tq.kv_put(partition_id="train", key=uid, data=batch_data)

Consumer (ReplayBuffer):
  data = tq.kv_get(partition_id="train", key=uid)

ReplayBuffer 轮询 TransferQueue, 支持不同 prompt 不同采样数 n
```

### 2.8 2026 新进展

```
verl 2026 关键更新:
  1. FSDP2 引擎: PyTorch 原生 FSDP 第二版, 改进分片效率
  2. Megatron 后端: 支持 671B MoE (DeepSeek-V3) 的 3D 并行训练
  3. 万亿参数 LoRA GRPO: 在 64×H800 上训练 LoRA 适配器
  4. 多硬件: NVIDIA (CUDA) + AMD (ROCm) + Ascend (NPU)
  5. Seed-Thinking-v1.5: 使用 verl 训练, AIME 2024 达 86.7 分
```

## 3. OpenRLHF 架构对比

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    OpenRLHF 架构                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            PPOTrainer / RayPPOTrainer                    │   │
│  │            (编排器 — 调度各 Worker)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                      │                                           │
│     ┌────────────────┼────────────────┐                         │
│     ▼                ▼                ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                  │
│  │vLLM      │  │DeepSpeed │  │DeepSpeed     │                  │
│  │Actor     │  │Critic    │  │Reward + Ref  │                  │
│  │Rollout   │  │Worker    │  │Worker        │                  │
│  │          │  │          │  │              │                  │
│  │[vLLM引擎]│  │[ZeRO-3]  │  │[ZeRO-3]     │                  │
│  │[推理优化]│  │[训练优化]│  │[仅推理]      │                  │
│  └──────────┘  └──────────┘  └──────────────┘                  │
│                                                                  │
│  核心特点:                                                       │
│    - DeepSpeed ZeRO-3 + AutoTP 做训练                           │
│    - vLLM 做推理 (Actor Rollout)                                │
│    - Ray 做分布式调度                                            │
│    - 支持 Agent-based 执行 (token-in-token-out)                 │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 部署模式

| 模式 | 描述 | 适用 |
|------|------|------|
| Colocate-All | 所有模型在同一组 GPU, ZeRO-3 + 权重卸载 | 7B-13B |
| Separate | 模型分布在不同 GPU 组 | 70B+ |

### 3.3 独特特性

```
Agent-based 执行:
  - token-in-token-out 范式: 环境返回 token 而非文本
  - Single-turn executor: 标准单轮 RL
  - Multi-turn executor: 多轮交互 (工具使用、代码调试)
  - 支持自定义环境接口

异步 RL 训练:
  - Actor 和 Trainer 分离部署
  - Actor 持续生成 rollout, Trainer 异步消费
  - 减少 GPU 空闲时间

DAPO 支持:
  - 动态 advantage 策略
  - 基于正确率的 prompt 过滤
  - 适合可验证奖励场景

VLM 支持 (v0.10+):
  - 多模态 RL 训练
  - 图像/视频输入的奖励计算
```

### 3.4 支持的算法

PPO, REINFORCE++, GRPO, RLOO, DAPO, DPO, SimPO, KTO, Online DPO, ReMax, BCO 等 15+ 种。

### 3.5 verl vs OpenRLHF 详细对比

| 维度 | verl | OpenRLHF |
|------|------|----------|
| **开发方** | 字节跳动 Seed | 开源社区 (Google, 字节, 腾讯等贡献) |
| **训练后端** | FSDP/FSDP2 + Megatron | DeepSpeed ZeRO-3 + AutoTP |
| **推理后端** | vLLM/SGLang/TRT-LLM | vLLM |
| **核心创新** | 3D-HybridEngine 零冗余重分片 | Agent-based 执行, 异步 RL |
| **Colocate** | 原生支持 (ActorRolloutRef 三合一) | ZeRO-3 + offload 模拟 |
| **大规模** | 671B MoE (Megatron 3D 并行) | 70B+ (RingAttention + AutoTP) |
| **前缀缓存** | 三级 (vLLM + Sticky Session + PrefixGrouper) | 依赖 vLLM APC |
| **数据传输** | TransferQueue 零拷贝 | Ray ObjectRef |
| **硬件** | NVIDIA + AMD + Ascend | NVIDIA |
| **配置** | Hydra YAML | 命令行参数 |
| **Multi-turn** | Agent Loop (main_ppo_sync) | Multi-turn executor |
| **论文** | HybridFlow (EuroSys 2025) | 无 |
| **生产验证** | Seed-Thinking-v1.5 (86.7 AIME) | Google/字节/腾讯/阿里使用 |

**选型建议**：
- 需要极致性能 + 大规模 (70B+/MoE) → verl
- 需要快速原型 + HF 生态 + Agent RL → OpenRLHF
- 研究实验 + 小模型 → TRL

## 4. TRL (HuggingFace) 定位

### 4.1 定位与特点

```
TRL (Transformer Reinforcement Learning):
  - HuggingFace 官方 RL 训练库
  - 深度集成 HF 生态系统 (transformers, datasets, PEFT)
  - 专注易用性和研究友好性
  - v1 版本已发布, 接口稳定
```

### 4.2 核心训练器

| 训练器 | 用途 |
|--------|------|
| `SFTTrainer` | 监督微调 |
| `GRPOTrainer` | GRPO 训练 (无需 Critic) |
| `DPOTrainer` | 直接偏好优化 |
| `RewardTrainer` | 奖励模型训练 |
| `PPOTrainer` | PPO 训练 (legacy) |
| `KTOTrainer` | Kahneman-Tversky 优化 |

### 4.3 集成生态

```
PEFT: LoRA/QLoRA 低资源 RL 训练
Unsloth: 2x 加速 + 60% 显存节省
CLI: trl sft / trl dpo / trl grpo 命令行接口
WandB: 内置实验追踪
```

### 4.4 局限性

```
TRL 不适合大规模分布式 RL 的原因:
  1. 单机/小规模设计: 不支持 Ray 分布式, 无 WorkerGroup 概念
  2. 无推理引擎集成: 没有 vLLM/SGLang 集成, rollout 用 transformers 原生
  3. 无权重同步: 不支持训练-推理引擎间的零拷贝权重同步
  4. 无前缀缓存: 没有 PrefixGrouper 或 KV Cache 复用
  5. 无混合引擎: 没有 3D-HybridEngine 的训练-推理重分片

适用场景: 单机小模型研究 (≤7B, 1-4 GPU)
不适用: 生产级大规模 RL 训练 (70B+, 多节点)
```

## 5. 关键 RL 训练优化

### 5.1 Reference Model 共享与权重管理

```
┌─────────────────────────────────────────────────────────┐
│            权重管理策略对比                                │
│                                                         │
│ PPO (4 模型):                                           │
│   Actor    ──── 训练 (需要梯度 + 优化器)                  │
│   Critic   ──── 训练 (需要梯度 + 优化器)                  │
│   Ref      ──── 冻结 (仅推理)                            │
│   RM       ──── 冻结/规则                                │
│                                                         │
│ GRPO (3 模型):                                          │
│   Actor    ──── 训练                                     │
│   Ref      ──── 冻结 (可选, KL 约束需要)                  │
│   RM       ──── 规则奖励 (无需 RM 模型)                   │
│                                                         │
│ 可验证奖励 GRPO (2 模型):                                │
│   Actor    ──── 训练                                     │
│   Ref      ──── 冻结 (可选)                              │
│   Reward   ──── 代码/math 验证器 (非神经网络)             │
└─────────────────────────────────────────────────────────┘

Colocate 模式下的显存优化:
  1. Ref 模型与 Actor 共享初始权重, 仅做推理, 无优化器状态
  2. sleep()/wake() 机制: 训练时释放推理引擎显存, 推理时释放训练中间态
  3. FSDP 分片: 每张 GPU 只存 1/N 的参数
  4. 参数 offload + 优化器 offload: 不活跃参数移到 CPU
```

### 5.2 Prefix Caching for Rollout Groups (GRPO n=8)

这是 RL 训练中最有价值的优化之一。GRPO 对每个 prompt 采样 N 个 response，天然存在大量前缀共享。

```
┌─────────────────────────────────────────────────────────────┐
│             GRPO 前缀冗余与优化                               │
│                                                             │
│ 问题: 同一 prompt 生成 N 个 response, prompt 被重复处理 N 次  │
│                                                             │
│ Prompt: "请解释量子计算" (512 tokens)                         │
│   → Response 1: "量子计算是..." (512 tokens)                │
│   → Response 2: "量子计算利用..." (512 tokens)              │
│   ...                                                       │
│   → Response 8: "量子计算与传统..." (512 tokens)            │
│                                                             │
│ 不优化: 8 × (512 + 512) = 8192 tokens 计算                  │
│ 优化后: 512 + 8 × 512 = 4608 tokens 计算                    │
│ 节省: (8-1) × 512 / 8192 = 43.75%                          │
│                                                             │
│ 通用公式: 节省比例 = (N-1) × L_p / (N × (L_p + L_r))       │
│   L_p = L_r, N=8  → 节省 43.75%                            │
│   L_p = 4×L_r, N=8 → 节省 70%                              │
│   L_p = L_r, N=16 → 节省 46.9%                             │
└─────────────────────────────────────────────────────────────┘
```

#### 三级前缀缓存机制

```
┌────────────────────────────────────────────────────────────┐
│                 verl 三级前缀缓存                           │
│                                                            │
│ 第 1 级: vLLM KV Cache 前缀缓存                            │
│   enable_prefix_caching: True                              │
│   PagedAttention Block Hash → 命中则复用 KV Cache          │
│   对 RL 场景: 同 prompt 多 response 自动命中                │
│                                                            │
│ 第 2 级: Sticky Session 路由                               │
│   多 vLLM 实例部署时:                                       │
│   相同 prompt hash → 路由到同一 GPU 实例                    │
│   确保第 1 级 KV Cache 命中率                               │
│                                                            │
│ 第 3 级: PrefixGrouper 训练侧优化                          │
│   训练 forward pass 中:                                     │
│   1. prefix self-attention 只计算一次                       │
│   2. 各 suffix 共享 prefix KV Cache                        │
│   3. 各 suffix 独立计算自己的 attention                     │
│   加速: 1.26x-1.70x (prompt 越长加速越大)                  │
│                                                            │
│ 协同效果:                                                   │
│   推理时: vLLM APC + Sticky Session → Rollout 加速         │
│   训练时: PrefixGrouper → Forward pass 加速                │
│   GRPO n=8, prompt=512: 总体节省 ~58% KV Cache            │
└────────────────────────────────────────────────────────────┘
```

#### PrefixGrouper 核心机制

```python
# verl/trainer/ppo/prefix_grouper_utils.py

# 1. 按 uid 分组 (相同 uid = 相同 prompt)
group_sizes = [count_consecutive_same_uid(uids)]

# 2. 提取每组 prefix (只取第一行, 同组 prompt 相同)
prefix_ids = prompts.index_select(0, prefix_indices)

# 3. 构建 PrefixGrouper 对象
prefix_grouper = PrefixGrouper.from_ungrouped_masks(
    prefix_mask, suffix_mask, group_sizes, padding_mode
)

# 4. 拼接输入: [prefix | resp1 | resp2 | ... | resp8]
concat_input_ids = prefix_grouper.concat_input(prefix_ids, ...)

# 5. Attention 分解:
#    prefix self-attention: Q_p, K_p, V_p → KV_cache_prefix (计算 1 次)
#    suffix concat-attention:
#      resp_i: Q_i × concat(K_p, K_i) → 复用 KV_cache_prefix
```

**限制**：仅支持 FSDP worker，不兼容 dynamic_bsz / remove_padding / Ulysses SP。

### 5.3 Reward Model Serving vs Reward Function

```
┌────────────────────────────────────────────────────────────┐
│             奖励计算方式对比                                │
│                                                            │
│ 方式 A: Reward Model (神经网络)                             │
│   ┌─────────────┐     ┌─────────────┐                     │
│   │ (prompt,    │ ──→ │  RM Model   │ ──→ score: float    │
│   │  response)  │     │  (7B-70B)   │                     │
│   └─────────────┘     └─────────────┘                     │
│   优点: 通用性强, 可捕捉细微偏好                            │
│   缺点: 额外 GPU 开销, 需要 RM 训练数据, 可能 reward hack   │
│   使用: PPO 通用 RLHF, 复杂偏好场景                        │
│                                                            │
│ 方式 B: 规则/函数奖励                                      │
│   ┌─────────────┐     ┌─────────────┐                     │
│   │ (prompt,    │ ──→ │ compute_    │ ──→ score: float    │
│   │  response)  │     │ score()     │     或 dict         │
│   └─────────────┘     └─────────────┘                     │
│   优点: 零 GPU 开销, 精确无 bias, 不会 reward hack         │
│   缺点: 需要明确判断标准 (数学/代码), 不适用开放式生成      │
│   使用: DeepSeek-R1, DAPO, 数学/代码 RL                    │
│                                                            │
│ 方式 C: 混合 (verl 支持)                                   │
│   data_source 路由:                                        │
│     "gsm8k"  → 数学精确匹配奖励                            │
│     "codegen" → 代码执行验证奖励                           │
│     "chat"   → RM 打分                                    │
│   同一训练批次内可混合多种奖励来源                           │
└────────────────────────────────────────────────────────────┘
```

verl 自定义奖励函数：

```python
# my_reward.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """返回 float 或 dict"""
    if data_source == "math":
        # 精确匹配
        return 1.0 if extract_answer(solution_str) == ground_truth else 0.0
    elif data_source == "code":
        # 执行验证
        return run_test_cases(solution_str, ground_truth)
    return 0.0

# 配置:
# reward.custom_reward_function.path=/path/to/my_reward.py
# reward.custom_reward_function.name=compute_score
```

### 5.4 KL 散度计算技巧

```
┌────────────────────────────────────────────────────────────┐
│             KL 惩罚机制                                     │
│                                                            │
│ R(x, y) = r(x, y) - β * KL[π_θ || π_ref]                 │
│                                                            │
│ Token 级 KL 计算:                                          │
│   kl_t = log(π_θ(a_t|s_t)) - log(π_ref(a_t|s_t))         │
│   KL = sum(kl_t for t in response_tokens)                  │
│                                                            │
│ 自适应 KL 控制器 (AdaptiveKLController):                   │
│   target_kl = 0.1 (目标 KL 散度)                           │
│   如果 current_kl > target_kl → β 增大 → 更强约束         │
│   如果 current_kl < target_kl → β 减小 → 允许更多探索     │
│                                                            │
│ 计算优化:                                                   │
│   1. Ref log prob 在 rollout 时一起计算 (避免额外 forward)  │
│   2. 使用 churning: 缓存 ref_log_prob, 不每步重算          │
│   3. Dr.GRPO 变体: 完全不用 KL (use_kl_loss=False)        │
│                                                            │
│ KL 控制器类型:                                              │
│   - 固定 β: 简单但需手动调参                               │
│   - 自适应: 自动调整, verl 默认                            │
│   - 无 KL: GRPO + 可验证奖励时可省去 Ref 模型              │
└────────────────────────────────────────────────────────────┘
```

### 5.5 Sequence Packing

```
┌────────────────────────────────────────────────────────────┐
│             Sequence Packing 优化                           │
│                                                            │
│ 问题: RL 训练中 response 长度差异大                         │
│   短 response: 50 tokens                                   │
│   长 response: 2048 tokens                                 │
│   Padding 到 max_length → 大量浪费                         │
│                                                            │
│ 解决: 将多个短序列 pack 到一个固定长度 slot                 │
│                                                            │
│ 不 Packing:                                                 │
│   [seq1(50) + PAD(1998)]  → 利用率 2.4%                   │
│   [seq2(100) + PAD(1948)] → 利用率 4.9%                   │
│   [seq3(2048)]             → 利用率 100%                   │
│                                                            │
│ Packing:                                                    │
│   [seq1(50) + seq2(100) + seq4(80) + PAD(3)] → ~99.8%    │
│   [seq3(2048)]                                → 100%      │
│                                                            │
│ 实现要点:                                                   │
│   1. position_ids 需要正确处理序列边界                      │
│   2. attention_mask 需要 block-diagonal (序列间不互看)     │
│   3. Flash Attention varlen 前向 (cu_seqlens)              │
│   4. loss 计算按序列边界切分, 不跨序列                      │
│                                                            │
│ verl 支持: use_dynamic_bsz=True 动态调整 batch             │
│ Megatron-LM: 原生 sequence packing 支持                    │
└────────────────────────────────────────────────────────────┘
```

## 6. GRPO vs PPO 深度对比

### 6.1 算法差异

```
┌────────────────────────────────────────────────────────────┐
│             PPO (GAE) vs GRPO 优势估计                     │
│                                                            │
│ PPO — GAE (Generalized Advantage Estimation):              │
│   需要 Critic 模型估计 V(s_t)                              │
│   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)                     │
│   A_t = Σ (γλ)^l * δ_{t+l}                                │
│   优势: Token-level, dense signal                          │
│   代价: 额外 Critic 模型 (~25% GPU)                        │
│                                                            │
│ GRPO — Group Relative:                                     │
│   对同一 prompt 的 N 个 response:                           │
│   1. 每个 response 的总奖励 R_i = sum(token_rewards)        │
│   2. 按 uid (prompt ID) 分组                               │
│   3. advantage = (R_i - group_mean) / (group_std + eps)    │
│   4. 广播到 token-level                                    │
│   优势: 无需 Critic, 更简单                                │
│   代价: Outcome-level (稀疏信号), 理论上有偏               │
└────────────────────────────────────────────────────────────┘
```

### 6.2 资源对比

| 方面 | PPO | GRPO |
|------|-----|------|
| 模型数量 | 4 (Actor+Critic+Ref+RM) | 2-3 (Actor+Ref(opt)+RM(opt)) |
| GPU 显存 (7B FP16) | ~200 GB | ~120 GB |
| 最小 GPU (7B) | 4×A100 | 1-2×A100 |
| 最小 GPU (70B) | 16×H100 | 8×H100 |
| 训练稳定性 | 需要 Critic 收敛 | 更鲁棒 |
| 适用场景 | 通用 RLHF | 推理/代码/可验证 |

### 6.3 其他算法变体

```
AdvantageEstimator 枚举 (verl core_algos.py):
  GAE                       — PPO: Actor+Critic
  GRPO                      — GRPO: 无 Critic
  REINFORCE_PLUS_PLUS       — REINFORCE++: 修正 GRPO 有偏估计
  REINFORCE_BASELINE        — REINFORCE + 简单基线
  RLOO                      — Leave-One-Out 基线
  REMAX                     — ReMax: 最大化奖励
  GDPO                      — Generalized DPO
  OPTIMAL_TOKEN_BASELINE    — Token 级最优基线
  TIR_OPTIMAL_TOKEN_BASELINE — TIR 变体

注册表模式: 新算法只需 @register_adv_est("name") + 实现函数
```

## 7. 大规模扩展：70B+ 模型策略

### 7.1 并行策略组合

```
┌────────────────────────────────────────────────────────────┐
│          70B+ RL 训练的并行策略                              │
│                                                            │
│ 模型大小: 70B FP16 = 140 GB 参数                           │
│ 训练状态 (Adam): 140 GB × 12 bytes/param = 1.68 TB        │
│ 总计: ~1.82 TB → 必须多 GPU 分布式                          │
│                                                            │
│ verl 策略 (3D-HybridEngine):                               │
│   训练: TP=4 + PP=2 + DP=4 → 32 GPU                       │
│   推理: TP=8 → 8 GPU (重分片, 零冗余)                     │
│   Rollout: vLLM TP=8 + prefix caching                     │
│                                                            │
│ OpenRLHF 策略:                                             │
│   DeepSpeed ZeRO-3: 参数/梯度/优化器全部跨 GPU 分片        │
│   AutoTP: 自动选择最优 TP 度                               │
│   RingAttention: 超长序列支持                               │
│   权重卸载: CPU offload 减少显存占用                       │
│                                                            │
│ 671B MoE (DeepSeek-V3):                                    │
│   Megatron 3D 并行: TP=8 + EP=64 + DP                     │
│   verl 已支持 Megatron 后端                                │
│   Active 参数 ~37B → 实际计算量远小于 671B                 │
│                                                            │
│ 万亿参数 LoRA GRPO:                                        │
│   基座模型冻结 + LoRA 适配器训练                            │
│   仅 LoRA 参数需要优化器状态                                │
│   verl 在 64×H800 上实现                                   │
└────────────────────────────────────────────────────────────┘
```

### 7.2 显存优化技巧汇总

| 技术 | 显存节省 | 开销 | 配置 |
|------|---------|------|------|
| 梯度检查点 | 30-45% | 27% 计算 | `enable_gradient_checkpointing=True` |
| 参数 offload | ~50% | CPU-GPU 传输 | `param_offload=True` |
| 优化器 offload | ~30% | CPU-GPU 传输 | `optimizer_offload=True` |
| BF16 混合精度 | ~35% | 无显著 | 默认开启 |
| ZeRO-3 分片 | 按DP比例 | 通信增加 | DeepSpeed/verl |
| LoRA | >90% (仅训 adapter) | 精度略降 | `use_lora=True` |
| Prefix Grouper | -40% 计算 | 兼容性限制 | `use_prefix_grouper=True` |

### 7.3 Colocate vs 分离部署

```
Colocate 模式 (推荐, verl 默认):
  所有角色共享同一组 GPU
  时分复用: Rollout 时用 vLLM, 训练时用 FSDP
  优点: GPU 利用率高, 零拷贝权重同步
  缺点: 显存管理复杂, 需 sleep/wake 切换
  适合: 7B-70B, 单机或小集群

分离模式:
  不同角色 → 不同 GPU 组
  Actor GPU 组: 8 GPU (推理优化)
  Critic GPU 组: 4 GPU
  RM + Ref GPU 组: 4 GPU
  优点: 各角色独立管理, 显存隔离
  缺点: 权重同步需网络传输, 部分时间 GPU 空闲
  适合: 70B+, 多节点
```

## 8. 生产 RLHF 系统

### 8.1 DeepSeek

```
DeepSeek-R1 训练管线:
  预训练基座: DeepSeek-V3-671B (MoE)
  SFT: 仅指令数据, 无推理数据
  RL (GRPO): 可验证奖励 (数学/代码)
  无 Critic, 无 RM → 仅 2 个模型

  关键发现:
    1. 纯 RL 可激发推理能力 ("aha moment")
    2. 模型自主发展 self-reflection, verification, backtracking
    3. 组内标准化 (GRPO) 足够驱动推理能力涌现
    4. 论文发表在 Nature (2025)

  Infra 特点:
    - MoE 架构: 671B 总参数, 37B active
    - EP (Expert Parallelism) 做大规模 MoE 训练
    - 推理时仅激活相关 expert → 计算量可控
    - verl 已支持 Megatron 后端做 MoE RL 训练
```

### 8.2 OpenAI

```
已知信息 (公开论文/博客):
  - InstructGPT (2022): 奠定 PPO RLHF 范式
  - ChatGPT: 在 InstructGPT 基础上规模化
  - o1/o3/o4-mini: 推理模型, 疑似 RL 训练

  推测的 Infra:
    - 大规模 Kubernetes + 自研训练框架
    - 多阶段 RL: 先可验证奖励 (math/code), 再 RM 奖励 (对话)
    - 安全 RL: Constitutional AI 或类似方法
    - 训练集群: 万级 GPU (H100/B200)

  公开已知的技术:
    - PPO 变体 (可能是 GRPO 或自研算法)
    - 大规模分布式训练 (Megatron/自研)
    - 自动化奖励工程 (AI 评判 AI)
```

### 8.3 Anthropic

```
已知信息:
  - Claude 系列使用 RLHF (公开承认)
  - Constitutional AI (CAI): 用 AI 反馈替代人类反馈
  - RLAIF: AI 生成偏好数据 → 训练 RM → RL 优化

  RL 管线推测:
    1. SFT → 2. CAI 偏好数据 → 3. RM 训练 → 4. PPO/变体
    额外: 安全 RL (避免有害输出)

  Infra 特点:
    - 自研训练框架 (非 verl/OpenRLHF)
    - 大规模 GPU 集群
    - 多轮对话 RL 训练
    - 长上下文 RL (Claude 支持 200K context)
```

### 8.4 Google

```
已知信息:
  - Gemini 系列使用 RLHF
  - OpenRLHF 贡献者 (Google Research)
  - TPU 训练 + JAX 生态

  Infra 特点:
    - TPU Pod 做大规模训练
    - JAX + Flax 框架 (非 PyTorch)
    - Pathways 做分布式编排
    - 可能使用 MoE (Mixtral 思路)
```

### 8.5 国内厂商

```
字节跳动:
  - verl 框架作者
  - Seed-Thinking-v1.5: AIME 2024 达 86.7 分
  - HybridFlow 论文 (EuroSys 2025)
  - 671B MoE RL 训练能力

腾讯:
  - OpenRLHF 贡献者
  - 大规模 RL 训练部署

阿里:
  - OpenRLHF 贡献者
  - Qwen 系列使用 GRPO 训练
  - 数学/代码推理 RL
```

## 9. PPO 训练主循环详解

### 9.1 SyncPPOTrainer (verl 推荐)

```
┌────────────────────────────────────────────────────────────┐
│             SyncPPOTrainer 训练循环                         │
│                                                            │
│ for epoch in range(total_epochs):                          │
│   for batch in train_dataloader:                           │
│                                                            │
│     === Phase 1: Rollout ===                               │
│     gen_batch = actor_rollout_wg.generate_sequences(batch) │
│     # vLLM/SGLang 自回归生成 response                      │
│                                                            │
│     === Phase 2: Reward ===                                │
│     batch = _compute_reward(batch)                         │
│     # 规则奖励 或 RM 打分                                  │
│                                                            │
│     === Phase 3: Ref Log Prob ===                          │
│     if use_reference_policy:                               │
│       batch = ref_wg.compute_ref_log_prob(batch)           │
│     # 计算 reference policy log prob (KL 约束)             │
│                                                            │
│     === Phase 4: Advantage ===                             │
│     batch = compute_advantage(batch, adv_estimator)        │
│     # GAE / GRPO / REINFORCE++ 等                          │
│                                                            │
│     === Phase 5: Update Critic (PPO only) ===              │
│     if use_critic:                                         │
│       critic_output = critic_wg.update_critic(batch)       │
│                                                            │
│     === Phase 6: Update Actor ===                          │
│     actor_output = actor_rollout_wg.update_policy(batch)   │
│     # PPO clip loss: L = -min(r*A, clip(r,1-e,1+e)*A)     │
│                                                            │
│     === Phase 7: Sync Weights ===                          │
│     actor_rollout_wg.update_weights()                      │
│     # Actor → Rollout (vLLM) 权重同步                     │
│                                                            │
│     === Phase 8: Validate ===                              │
│     if step % val_interval == 0:                           │
│       validate()                                           │
└────────────────────────────────────────────────────────────┘
```

### 9.2 Rollout 占比分析

```
典型 RL 训练时间分布:

Rollout (推理生成):     ~80% ← 优化重点
  - 自回归 token-by-token 生成
  - 无法充分并行 (sequential decoding)
  - Prefix Caching 可减少 40-80%

Advantage 计算:         ~5%
  - 在 Driver 上计算, 非 GPU 密集

Actor 更新:             ~8%
  - FSDP/Megatron 分布式训练

Critic 更新 (PPO):      ~4%
  - GRPO 不需要此步

权重同步:               ~3%
  - Colocate: 零拷贝
  - 分离: 网络 RPC

优化 ROI 排序:
  1. Prefix Caching for Rollout → 最高 ROI
  2. 混合引擎 (减少 GPU 总量) → 高 ROI
  3. GRPO (去掉 Critic) → 一次性节省 25%
  4. Sequence Packing → 中等 ROI
  5. Kernel 优化 (Flash Attention 等) → 中等 ROI
```

## 10. 训练循环数据流

### 10.1 GRPO 数据流

```
┌──────────────────────────────────────────────────────────────┐
│              GRPO 数据流 (verl)                               │
│                                                              │
│ DataLoader                                                   │
│   │                                                          │
│   ▼                                                          │
│ prompts (batch_size=32)                                      │
│   │                                                          │
│   ├── uid 分配 (每 prompt 唯一标识)                           │
│   │                                                          │
│   ▼                                                          │
│ repeat(n=8) → 256 个 (prompt, response) 对                   │
│   │                                                          │
│   ▼                                                          │
│ vLLM Rollout 生成 response                                   │
│   │ ← Prefix Caching: 同 prompt 共享 KV Cache               │
│   ▼                                                          │
│ Reward 计算                                                  │
│   │ ← 规则: compute_score(data_source, response, truth)     │
│   │ ← RM: reward_model.forward(prompt+response)             │
│   ▼                                                          │
│ Ref Log Prob (可选)                                          │
│   │ ← ref_engine.forward_batch(batch)                       │
│   ▼                                                          │
│ GRPO Advantage                                               │
│   │ ← 按 uid 分组                                            │
│   │ ← A_i = (R_i - group_mean) / group_std                  │
│   ▼                                                          │
│ Actor 更新 (PPO clip loss)                                   │
│   │ ← train_batch(data, loss_fn=ppo_loss)                   │
│   ▼                                                          │
│ 权重同步 → Rollout Engine                                    │
│   │ ← update_weights()                                      │
│   ▼                                                          │
│ 下一 epoch                                                   │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 关键超参数

| 参数 | PPO 典型值 | GRPO 典型值 | 说明 |
|------|-----------|------------|------|
| learning rate | 1e-6 | 1e-6 | 比 SFT 小 10x |
| clip_ratio (ε) | 0.2 | 0.2 | PPO 裁剪范围 |
| rollout.n | - | 5-16 | 每 prompt 采样数 |
| ppo_mini_batch_size | 16-256 | 16-256 | Actor mini batch |
| max_prompt_length | 256-1024 | 256-1024 | Prompt 最大长度 |
| max_response_length | 512-2048 | 512-2048 | Response 最大长度 |
| kl_coef (β) | 0.02-0.1 | 0.001-0.02 | KL 惩罚系数 |
| entropy_coeff | 0 | 0-0.01 | 熵正则系数 |

## 11. 硬件需求与配置参考

### 11.1 模型大小 vs GPU 需求

| 模型大小 | 算法 | 最小 GPU | 推荐 GPU | Offload |
|---------|------|---------|---------|---------|
| 0.5B | GRPO | 1× GPU (24GB) | 1× A100 | 不需要 |
| 7B | GRPO | 1× GPU (80GB) | 4-8× A100 | 推荐 |
| 7B | PPO | 4× A100 | 8× A100 | 需要 |
| 8B | GRPO | 4× GPU | 8× A100 | 推荐 |
| 70B | GRPO | 8× H100 | 16× H100 | 推荐 |
| 70B | PPO | 16× H100 | 32× H100 | 需要 |
| 671B MoE | GRPO | 32× H100 | 64× H100 | 需要 |

### 11.2 典型 YAML 配置 (7B GRPO, 8 GPU)

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    enable_gradient_checkpointing: True
  actor:
    strategy: fsdp
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 16
    fsdp_config:
      param_offload: True
      optimizer_offload: True
    clip_ratio: 0.2
    entropy_coeff: 0
    use_kl_loss: False
  rollout:
    name: vllm
    tensor_model_parallel_size: 8
    gpu_memory_utilization: 0.4
    n: 5              # 每 prompt 5 个 response
  ref:
    strategy: fsdp
    fsdp_config:
      param_offload: True

algorithm:
  adv_estimator: grpo
  grpo_n: 5
  norm_adv_by_std_in_grpo: True

data:
  train_files: ~/data/math/train.parquet
  val_files: ~/data/math/test.parquet
  train_batch_size: 32
  max_prompt_length: 256
  max_response_length: 512

trainer:
  total_epochs: 15
  n_gpus_per_node: 8
  nnodes: 1
```

## 12. 2026 年趋势与展望

### 12.1 算法趋势

```
演进路径:
  PPO (4模型) → GRPO (3模型) → 可验证奖励 (2模型)
      ↓               ↓               ↓
  通用 RLHF      推理/代码 RL    数学/代码 RL
  需要 Critic    无 Critic       无 Critic + 无 RM
  需要 RM        可选 RM         规则奖励

新算法:
  - REINFORCE++: 修正 GRPO 有偏估计, 全局标准化
  - RLOO: Leave-One-Out 基线, 更低方差
  - DAPO: 动态 advantage + prompt 过滤
  - Dr.GRPO: sequence-mean-token-sum-norm, 无 KL
  - GSPO: Generalized DPO 变体
```

### 12.2 Infra 趋势

```
1. 混合引擎统一化
   训练和推理不再割裂 → 同一框架管理
   verl 3D-HybridEngine + OpenRLHF Ray+vLLM

2. MoE RL 训练
   DeepSeek-V3 671B MoE → 只激活 37B 参数
   EP (Expert Parallelism) + TP + DP
   verl Megatron 后端已支持

3. 异构硬件支持
   verl: NVIDIA + AMD + Ascend
   降低对单一硬件供应商依赖

4. Agent Loop / Multi-Turn RL
   工具使用训练 (搜索、代码执行、API 调用)
   变长序列 + 环境交互延迟
   verl main_ppo_sync 和 OpenRLHF multi-turn executor

5. 可验证奖励 → 无 RM 训练
   数学: 答案精确匹配
   代码: 测试用例验证
   减少 RM 训练成本和 reward hacking 风险

6. 万亿参数 LoRA RL
   基座冻结 + LoRA 适配器训练
   大幅降低 RL 训练的显存需求
   verl 在 64×H800 上实现
```

### 12.3 开放问题

```
1. RL 训练稳定性
   - Reward hacking 仍然存在
   - KL 约束 vs 探索的平衡
   - 长训练过程中的 model collapse 风险

2. 大规模 MoE RL 的效率
   - Expert 负载均衡在 RL 场景的特殊挑战
   - EP 的通信开销
   - MoE + Prefix Caching 的兼容性

3. Multi-Turn RL 的工程复杂性
   - 环境沙箱管理 (代码执行安全)
   - 变长序列的 batch 管理
   - 跨 turn 的奖励累积

4. 异构硬件的性能一致性
   - AMD/Ascend 的 kernel 优化成熟度
   - 跨硬件的数值一致性
   - 混合硬件集群的调度策略
```

## 13. 关键要点总结

```
1. Rollout 是瓶颈 (~80% 时间):
   优化推理引擎集成和 prefix sharing 是 ROI 最高的方向

2. 模型数在减少:
   PPO 4 模型 → GRPO 3 模型 → 可验证奖励 2 模型

3. 混合引擎是核心:
   verl 3D-HybridEngine 和 OpenRLHF 解决训练-推理权重同步

4. Prefix Sharing 是免费午餐:
   RL 场景天然大量共享前缀 (GRPO n=8)
   三级缓存: vLLM APC + Sticky Session + PrefixGrouper

5. GRPO 是当前默认选择:
   verl/OpenRLHF 都推荐 GRPO 作为默认算法
   更少 GPU, 更简单, 推理/代码场景效果相当

6. 框架选择:
   大规模生产 → verl (671B MoE, 万亿 LoRA)
   快速原型/研究 → OpenRLHF (Agent RL, 异步训练)
   小模型研究 → TRL (HF 生态集成)
```

## 参考资料

### 论文
- [HybridFlow / verl](https://arxiv.org/abs/2409.19256) (EuroSys 2025)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Nature 2025)
- [RLOO](https://arxiv.org/abs/2402.14740)
- [REINFORCE++](https://arxiv.org/abs/2501.03262)

### 项目
- [verl](https://github.com/volcengine/verl) — 字节跳动 Seed
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 开源社区
- [TRL](https://github.com/huggingface/trl) — HuggingFace
- [PrefixGrouper](https://github.com/johncaged/PrefixGrouper)

### 相关笔记
- [verl 架构深度分析](verl-architecture.md)
- [verl 源码阅读](verl-source-reading.md)
- [verl RL Infra 源码](verl-rl-infra-reading.md)
- [PrefixGrouper 深度阅读](verl-prefix-grouper.md)
- [GRPO 实战指南](grpo-practical-guide.md)
- [RLHF 训练基础设施基础](../fundamentals/rlhf-training-infra.md)
