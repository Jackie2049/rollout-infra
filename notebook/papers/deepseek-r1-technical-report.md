# Paper Reading: DeepSeek-R1 技术报告

> DeepSeek-AI, 2025年1月 | DeepSeek-R1/R1-Zero
> 精读日期: 2026-06-05
> 优先级: P0 (AI Expert Roadmap Phase 4)

## 1. 论文概要

**核心贡献**: 证明了推理能力可以通过纯 RL 训练从基础模型中**涌现**, 无需人工标注推理数据.

**两个模型**:
- **DeepSeek-R1-Zero**: 直接在基础模型上用 GRPO 做 RL, 无 SFT 冷启动
- **DeepSeek-R1**: 加入冷启动 SFT → GRPO RL → 拒绝采样 SFT → DPO 四步 pipeline

**关键现象 — "Aha Moment"**:
```
在 RL 训练过程中, 模型突然开始:
1. 写 "Wait, let me reconsider..."
2. 进行自我验证和纠错
3. 探索多种解题策略
4. 这些行为从未被明确教过 — 是自然涌现的!
```

## 2. 方法详解

### 2.1 GRPO (Group Relative Policy Optimization)

**核心思想**: 用组内统计量替代 learned value function (critic).

```
标准 PPO:
  advantage = reward - V(state)     # V 需要 extra model
  需要 4 个模型: π, π_ref, V, RM

GRPO:
  对同一个问题生成 n 个回答 (n=8-64)
  advantage_i = (reward_i - mean(rewards)) / std(rewards)
  只需要 2 个模型: π, π_ref
```

**GRPO 目标函数**:
```
L_GRPO = E[ Σ min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t)
           - β · KL(π_θ || π_ref) ]

其中:
  r_t(θ) = π_θ(a_t|s_t) / π_ref(a_t|s_t)  # 重要性采样比率
  Â_t = (reward_i - μ_G) / σ_G              # 组归一化优势
  KL 项防止偏离参考模型太远
```

**为什么 GRPO 比 PPO 好?**
```
1. 不需要 critic → 省一半 GPU 内存
2. 组内对比比绝对价值估计更稳定
3. reward 信号更直接 (outcome-based, 无需过程奖励)
4. 实现 simpler (verl 统一 RayPPOTrainer 切换 adv_estimator)
```

### 2.2 训练 Pipeline (DeepSeek-R1 完整版)

```
Step 1: Cold-Start SFT (可选)
  ┌──────────────────────────────────┐
  │ 少量 CoT 数据 (<1000条)         │
  │ 目的: 教模型基本的推理格式       │
  │   [think_start]                  │
  │   reasoning steps here...        │
  │   [think_end]                    │
  │   final answer                   │
  └──────────────────────────────────┘
         ↓
Step 2: GRPO with Outcome Reward
  ┌──────────────────────────────────┐
  │ 大规模 RL 训练                   │
  │ reward = 答案正确性 (自动验证)   │
  │ 关键: 推理能力自然涌现           │
  │ "Aha moment" 出现在此阶段        │
  └──────────────────────────────────┘
         ↓
Step 3: Rejection Sampling + SFT
  ┌──────────────────────────────────┐
  │ 从 RL 模型采样大量回答           │
  │ 保留正确的推理链                 │
  │ 结合 SFT 数据 (写作/翻译/等)    │
  │ 蒸馏到更小模型                   │
  └──────────────────────────────────┘
         ↓
Step 4: DPO (Direct Preference Optimization)
  ┌──────────────────────────────────┐
  │ 用 RL 模型生成 (chosen, rejected)│
  │ DPO 微调最终对齐                 │
  └──────────────────────────────────┘
```

### 2.3 Reward Design

**规则-based reward** (DeepSeek-R1 的关键创新):
```
1. Accuracy Reward: 答案是否正确?
   - 数学: 等价性检查
   - 代码: 执行测试用例
   - 推理: 逻辑一致性检查

2. Format Reward: 格式是否正确?
   - 是否包含 <think_start>...<think_end>?
   - 是否有明确的最终答案?

不需要 learned reward model!
→ 消除了 reward hacking 的风险
→ reward 信号完全透明可解释
```

### 2.4 Budget Forcing (思考预算控制)

```
思考预算 = 最大 thinking tokens

目的: 平衡推理深度和延迟

实现:
  1. 生成 thinking tokens
  2. 达到预算 → 强制插入 [think_end] + 生成 answer
  3. 如果 thinking 自然结束 → 不强制

DeepSeek-R1 的建议预算:
  1.5B: 512 tokens
  7B: 1024 tokens
  70B: 4096 tokens
  671B: 8192+ tokens

关键发现:
  - 小模型不需要长推理链 (4-8 tokens 足够)
  - 长推理链对小模型甚至有害 (噪声增加)
  - Budget forcing 可以作为 API 参数暴露给用户
```

## 3. 实验结果

### 3.1 核心 Benchmark

| Benchmark | DeepSeek-R1-Zero | DeepSeek-R1 | OpenAI o1-1217 |
|-----------|------------------|-------------|----------------|
| MATH-500 | 95.9% | 97.3% | 96.4% |
| AIME 2024 | 73.3% | 79.8% | 79.2% |
| Codeforces (percentile) | - | 96.3 | 96.6 |
| GPQA Diamond | 71.5% | 71.5% | 75.7% |
| MMLU | 87.0% | 90.8% | 92.3% |

**关键发现**:
- R1-Zero (纯 RL) 已经很强 → 推理可以涌现!
- R1 (完整 pipeline) 在大多数任务上接近 o1-1217
- 最大的差距在通用知识任务 (MMLU, GPQA)

### 3.2 Distillation 结果

| Student | MATH-500 | AIME 2024 | GPQA Diamond |
|---------|----------|-----------|--------------|
| Qwen-1.5B | 83.9% | 28.9% | 42.0% |
| Qwen-7B | 92.8% | 55.5% | 49.1% |
| Qwen-14B | 93.9% | 59.4% | 55.2% |
| Qwen-32B | 94.3% | 62.1% | 58.3% |
| **R1-Zero-32B** | **94.6%** | **67.3%** | **63.6%** |

**重要**: 直接蒸馏 R1 的推理到小模型 > 在小模型上做 RL!
→ 推理能力主要来自大模型的 reasoning patterns, 不是小模型自己能学到的

### 3.3 Aha Moment 分析

```
训练过程中观察到的推理能力涌现:

Stage 1 (初期): 直接给答案
  "The answer is 42."

Stage 2 (中期): 开始写简单推理
  "Let me think... 6 × 7 = 42. The answer is 42."

Stage 3 ("Aha" 时刻): 出现自我纠错
  "Let me solve this step by step.
   6 × 7 = 42. Wait, let me verify: 6 + 6 + 6 + 6 + 6 + 6 + 6 = 42. Yes!
   The answer is 42."

Stage 4 (后期): 探索多种策略
  "Method 1: 6 × 7 = 42.
   Method 2: I can factor 42 = 2 × 3 × 7...
   Both confirm the answer is 42."

这个涌现需要:
  - 足够大的基础模型 (>7B)
  - 足够的训练步数
  - 正确的 reward 信号 (outcome-based)
```

## 4. 关键技术洞察

### 4.1 为什么推理能涌现?

```
核心: GRPO 的组归一化提供了正确的学习信号

1. 同一问题生成 n 个回答 (不同推理链)
2. 正确答案 → reward = 1 → 正优势 (Â > 0)
3. 错误答案 → reward = 0 → 负优势 (Â < 0)
4. 模型学到: 哪种推理链导向正确答案
5. 逐渐发现:
   - 更长的推理链 → 更多验证机会 → 更高的正确率
   - 自我纠错 → 修正初始错误 → 更高正确率
   - 多策略验证 → 双重检查 → 更高正确率
→ 这些 "策略" 被正优势强化, 自然涌现!

类比: AlphaGo 的 "Move 37"
  - 从未被人类教过
  - 但 self-play RL 发现了这个妙手
  - 推理模型的 "Aha moment" 同理
```

### 4.2 Cold Start vs Zero

```
R1-Zero (无 SFT):
  ✓ 证明推理可以涌现
  ✗ 可读性差 (推理过程不连贯)
  ✗ 语言混杂 (中英文混合)

R1 (有 SFT 冷启动):
  ✓ 推理格式规范
  ✓ 可读性好
  ✓ 性能更高 (尤其在通用任务)
  → 推荐做法: 少量 SFT + 大规模 GRPO

冷启动数据只需 <1000 条!
→ 关键是格式模板, 不是推理知识
```

### 4.3 与 PPO 的对比

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic model | 需要 | **不需要** |
| GPU 内存 | 4 个模型 | **2 个模型** |
| 优势估计 | V(s) 学习的 | **组统计量** |
| 训练稳定性 | 需要调 V 的 lr | **更稳定** |
| Reward hacking | 风险高 | **规则reward, 低风险** |
| 实现复杂度 | 高 | **低** |
| verl 支持 | RayPPOTrainer | 同一 trainer, adv_estimator=grpo |

### 4.4 Reasoning Model 架构决策

```
基础模型选择:
  DeepSeek-V3 (671B MoE, 37B active)
  → 已经很强的 base → RL 可以快速提升

为什么不用小模型?
  → 蒸馏表显示: 小模型做 RL < 蒸馏大模型的推理
  → 推理能力主要在大模型中涌现

Token 设计:
  [think_start] ... reasoning ... [think_end] answer
  → 特殊 token 让模型区分推理和回答
  → 推理部分可以折叠 (用户不需要看)

长度惩罚:
  太短 → 推理不充分 → 正确率低
  太长 → 浪费 tokens, 可能发散
  → Budget forcing 自动控制
```

## 5. 实践启示 (从 Reasoning Model Simulator 验证)

| 论文发现 | Simulator 验证 |
|---------|---------------|
| GRPO > PPO 稳定性 | ✅ GRPO reward 0.679 vs PPO 0.411 |
| 短 thinking 对小模型最优 | ✅ Think=4-8 tokens 最优 (24+ 无收益) |
| Logic 任务最容易学 | ✅ Logic +0.094 > Arithmetic +0.016 |
| Budget forcing 0.5 最优 | ✅ think_weight=0.5 → acc 0.047 |
| 字符级 tokenizer 太粗糙 | ✅ 无法有效学习算术 |
| 需要 outcome-based reward | ✅ 正确答案 reward 足够 |

## 6. 对 AI Infra 的影响

### 6.1 训练系统需求

```
GRPO 训练的 Infra 挑战:

1. Rollout 服务:
   - 需要高效生成 n 个回答 (n=8-64)
   - vLLM/SGLang 的 continuous batching
   - Prefix caching: n 个回答共享 prompt → 大幅节省

2. Reward 计算:
   - 代码执行沙箱 (安全性)
   - 数学等价性检查
   - 批量评估 (GPU/CPU)

3. 训练框架:
   - verl: 统一 PPO/GRPO/REINFORCE
   - ActorRolloutRefWorker: 复用模型做 rollout
   - Ray 集群调度

4. 内存优化:
   - GRPO 省 50% 内存 (无 critic)
   - Prefix caching 省 40-76% KV cache
   - ZeRO + offload 用于大模型
```

### 6.2 推理服务需求

```
Reasoning model 的推理挑战:

1. 变长序列:
   - thinking 长度从 0 到 8192+ tokens
   - KV cache 需要动态管理
   - Speculative decoding 不适用 (推理过程不可预测)

2. 延迟-准确性权衡:
   - 用户可选择 "思考量" (budget forcing)
   - 简单问题: 短 thinking → 低延迟
   - 复杂问题: 长 thinking → 高准确性

3. 显存管理:
   - thinking tokens 也需要 KV cache
   - 可能需要 KV cache offload (CPU/SSD)
   - Paged Attention 帮助管理碎片
```

## 7. 延伸阅读

- [x] **GRPO 源码**: verl `core_algos.py` (2488 lines, 10+ advantage estimators)
- [x] **Reasoning Simulator**: `tools/reasoning_model_simulator.py`
- [ ] **DeepSeek-V3**: MoE 架构, FP8 训练
- [ ] **OpenAI o1**: 推理模型的开创者
- [ ] **Kimi k1.5**: 另一个推理模型
- [ ] **PR #6401**: Prefix-Tree Shared Attention for GRPO
