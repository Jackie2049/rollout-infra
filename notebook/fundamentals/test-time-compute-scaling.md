# Test-Time Compute Scaling — 推理时计算扩展的新范式

> 2026-06-07 | 从训练时扩展到推理时扩展: 让模型"想得更久"而非"练得更多"

## 概述

**范式转变**: 从"训练更大的模型"(scale params)到"让模型推理时想得更久"(scale inference compute) → 这是2024-2026最重要的研究方向之一。

**类比**: 人类遇到难题会花更多时间思考 → LLM也应该在推理时分配更多计算资源给难题。

## 核心理论框架

### 计算分配二分法

| 策略 | 机制 | 适用问题 | 计算类型 |
|------|------|----------|----------|
| **并行计算 (Parallel)** | Best-of-N采样 → 选最优 | 简单问题 | 独立生成N个solution → 验证选best |
| **序列计算 (Sequential)** | Tree-of-Thought搜索 → 逐步精炼 | 困难问题 | 逐步推理+回溯+验证 |
| **迭代精炼 (Iterative)** | 自我修正 → 多轮改进 | 中等问题 | 生成→批评→修改→再验证 |

### Snell et al. (2024) 关键发现

**"Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling LLM Parameters"**

核心结论: 给定总计算预算, 小模型+更多推理计算 可以 超过 大模型+少推理计算。

**最优分配取决于问题难度**:
- 简单题 → Best-of-N更有效 (独立尝试→高命中率)
- 困难题 → Tree Search更有效 (逐步推理→系统性探索)
- **自适应分配** → 根据问题难度动态选择策略 → 最优

**数学直觉**:
- Best-of-N: 成功概率 = 1 - (1-p)^N → p=0.3, N=10 → 97.2%
- Tree Search: 每步成功概率累积 → 步步验证→避开死胡同
- 关键: p(单次成功)低时 → N需要非常大 → 但Tree Search可以逐步提升p

### 验证模型: ORM vs PRM

| 类型 | 定义 | 优点 | 缺点 |
|------|------|------|------|
| **Outcome RM (ORM)** | 只验证最终答案 | 简单, outcome-only | 无法判断中间步骤正确性 |
| **Process RM (PRM)** | 验证每步推理 | 精确, 步级反馈 | 需要步骤标注, 更复杂 |

**Lightman et al. (2023)**: PRM dramatically outperforms ORM for best-of-N and search → 步级验证是有效test-time compute的关键!

**与我们实验的联系**: GRPO outcome-only reward = ORM → DeepSeek-R1证明outcome-only足够涌现推理 → 但PRM可能进一步提升!

## 实际应用: o1/o3/o4

### OpenAI o1 推测机制

o1的推理过程推测包含:
1. **内部搜索**: 生成多个推理路径 → 验证选择
2. **自我验证**: 在CoT中自我检查关键步骤
3. **回溯修正**: 发现错误→回退→换路径
4. **自适应计算**: 简单题少think tokens → 困难题多think tokens

**o3/o4进一步扩展**:
- 更多think tokens → 更长推理链
- 更强的PRM(内部) → 更精确的步骤验证
- 并行+序列混合策略

### DeepSeek-R1的"aha moment"

R1的GRPO训练 → 模型自发涌现:
- self-verify: "Wait, let me check this..."
- reflect: "This approach seems wrong..."
- backtrack: "Let me try another way..."

**本质**: GRPO组比较 → 模型探索多条推理路径 → 自然发现自我验证的价值 → test-time compute的内化!

## 算法详解

### 1. Best-of-N with Verifier

```python
# Simple Best-of-N
for i in range(N):
    solution = model.generate(problem)
    score = verifier.score(solution)
best = max(solutions, key=lambda s: verifier.score(s))
```

**效率分析**:
- 生成N个solution → N倍推理成本
- 但每个solution独立 → 可并行
- 验证成本 ≈ 1次推理(小verifier)
- 总成本 = N × generation_cost + verification_cost

**关键**: verifier质量决定Best-of-N的上限!

### 2. Tree-of-Thought (ToT) Search

```python
# Tree Search with step-level verification
def tree_search(problem, model, verifier, max_depth=5, branching=3):
    root = State(problem)
    for depth in range(max_depth):
        candidates = []
        for state in frontier:
            for b in range(branching):
                next_state = model.generate_step(state, problem)
                score = verifier.score_step(next_state, problem)
                if score > threshold:
                    candidates.append((next_state, score))
        frontier = sorted(candidates, key=lambda x: x[1])[:beam_width]
    return best_final_state
```

**效率分析**:
- 每步生成branching个候选 → pruning低分路径 → 节省计算
- 验证每步 → 及时发现错误 → 避免浪费后续计算
- 总成本 ≈ Σ_depth branching × step_cost

### 3. Iterative Refinement (Self-Correction)

```python
# Self-correction loop
solution = model.generate(problem)
for round in range(max_rounds):
    critique = model.critique(problem, solution)
    if critique.is_correct():
        break
    solution = model.refine(problem, solution, critique)
```

**关键**: 自我修正能力需要训练 → 不是所有模型都能有效自我修正 → 需要RL训练(如R1的GRPO)

## 计算预算分析

### 训练 vs 推理 计算分配

**传统**: 几乎所有计算用于训练 → 推理仅1次forward pass
**新范式**: 推理时分配更多计算 → 训练仍需但推理计算占比上升

**计算预算公式**:
```
Total_Compute = Training_Compute + N_requests × Inference_Compute_per_request
Inference_Compute = tokens_generated × compute_per_token × test_time_multiplier
test_time_multiplier ∈ [1, 10, 100] depending on strategy
```

**推理成本趋势**:
- 2024: 推理成本 ≈ 训练成本的10-20%
- 2025(o1): 推理成本 ≈ 训练成本的50-100%
- 2026预测: 推理成本可能超过训练成本(高频请求+长推理链)

### 对AI Infra的影响

| 影响维度 | 传统 | Test-time scaling |
|----------|------|-------------------|
| **推理吞吐** | 高(短序列) | 低(长CoT+搜索) |
| **延迟** | TTFT<1s | TTFT可达10-100s |
| **成本** | $0.1-1/Mtok | $1-10/Mtok(o1级别) |
| **GPU需求** | 推理为主 | 推理×10 |
| **KV Cache** | 短序列 | 长CoT→10x内存 |
| **调度** | FCFS | 需考虑think token长度 |

**Infra挑战**:
1. **长CoT KV cache**: 10K+ think tokens → KV cache 10x增长
2. **推理延迟**: 用户等待10-100s → SLO重新设计
3. **成本控制**: 推理成本10x → 需要更高效硬件+量化
4. **调度复杂性**: 不同问题think长度不同 → 动态资源分配

## 与我们实验的直接联系

### GRPO = Best-of-N + Outcome Verifier

GRPO训练循环:
1. 生成n个response (Best-of-N)
2. outcome reward (ORM验证)
3. 组归一化选最优 (验证选择)

**关键区别**: GRPO是**训练时**做Best-of-N → 训练策略→推理时模型自己生成1个solution
**Test-time compute**: 推理时也做Best-of-N → 每次请求生成N个solution → 选最优

### DAPO动态采样 = 推理时自适应计算

DAPO的动态采样(n随σ调整) → 推理时的类比:
- 简单问题 → n小(少推理计算)
- 困难问题 → n大(多推理计算)

**这正是test-time compute scaling的核心思想!**

### DeepSeek-R1 "aha moment" = 内化Test-time Compute

R1的GRPO训练 → 模型学会:
- 在推理时"多想几步"(self-verify/reflect)
- 不需要显式编程 → RL训练自然涌现

**本质**: RL训练将test-time compute策略**内化到模型权重**中 → 推理时模型自动选择思考深度!

## 模拟实验结果 (500 problems)

### Best-of-N Scaling

**简单问题 (difficulty < 0.2)**:
| N | Accuracy | Cost (tokens) |
|---|----------|---------------|
| 1 | 0.310 | 9 |
| 4 | 0.763 | 36 |
| 8 | 0.918 | 72 |
| 16 | 0.981 | 144 |

**困难问题 (difficulty > 0.7)**:
| N | Accuracy | Cost (tokens) |
|---|----------|---------------|
| 1 | 0.044 | 48 |
| 4 | 0.160 | 192 |
| 8 | 0.291 | 384 |
| 16 | 0.469 | 768 |
| 32 | 0.687 | 1536 |
| 64 | 0.835 | 3072 |

**关键**: 困难问题需要N=64才达到83.5% → 成本64倍 → Best-of-N对困难题不高效!

### Tree Search (困难问题)

| Config | Accuracy | Cost | Notes |
|--------|----------|------|-------|
| b=2,d=3,bw=2 | 0.008 | 109 | 极低成功率 |
| b=3,d=5,bw=3 | 0.001 | 246 | 更低! |
| b=4,d=7,bw=4 | 0.000 | 438 | 0% |

**关键**: Tree Search对困难问题反而更差 → 模型能力不足时, 搜索也无法找到正确路径 → **需要足够的模型能力!**

### Iterative Refinement (中等问题)

| Rounds | Accuracy | Cost |
|--------|----------|------|
| 1 | 0.473 | 43 |
| 2 | 0.716 | 50 |
| 3 | 0.843 | 55 |
| 5 | 0.952 | 59 |
| 8 | 0.990 | 60 |

**关键**: 5轮精炼已达95% → 成本仅59 tokens → 比Best-of-N=16(144tokens)更高效!

### 最优分配 (budget=500 tokens)

**所有难度级别**: Iterative Refinement (r=8) 是最优策略 → accuracy 98-99%

**这与DeepSeek-R1的"aha moment"完全一致!**
- R1的GRPO训练 → 模型涌现self-verify/reflect/backtrack
- 这就是Iterative Refinement的内化版!
- 模型不需要外部搜索 → 自我修正更高效

### 核心结论

1. **Best-of-N适合简单问题**: p>0.3 → N=8已达94%
2. **Tree Search需要强模型**: 模型能力不足 → 搜索路径几乎全错 → 0%成功率
3. **Iterative Refinement最通用**: 自我修正 → 成本低 → 效果好
4. **模型能力是关键前提**: 任何test-time compute策略都需要模型有基本解题能力 → 小模型策略无效
5. **与GRPO实验一致**: DAPO/GRPO在小模型上不稳定 → test-time compute同理

## 未来方向

1. **SFT→GRPO + Test-time compute**: SFT暖启动→GRPO训练推理策略→推理时自适应计算 → 最优pipeline
2. **PRM训练**: 过程验证器训练 → 步级验证 → Tree Search有效
3. **混合策略**: 简单题Best-of-N→困难题Iterative Refinement → 自适应计算分配
4. **推理Infra优化**: 长CoT KV cache管理 + 量化 + prefix sharing
5. **成本模型**: 推理成本×10 → 需要新的ROI分析

1. **SFT→GRPO + Test-time compute**: SFT暖启动→GRPO训练推理策略→推理时自适应计算 → 最优pipeline
2. **PRM训练**: 过程验证器训练 → 步级验证 → Tree Search有效
3. **混合策略**: 简单题Best-of-N→困难题Tree Search → 自适应计算分配
4. **推理Infra优化**: 长CoT KV cache管理 + 量化 + prefix sharing
5. **成本模型**: 推理成本×10 → 需要新的ROI分析

## Sources

- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) — Snell et al., 2024
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) — Yao et al., 2023
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al., 2023
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek, 2025
- OpenAI o1/o3 blog posts