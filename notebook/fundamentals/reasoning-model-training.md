# Reasoning Model 训练方法论: DeepSeek-R1 / o1 / o3

> 2026-06-07 | 基于 DeepSeek-R1技术报告 + 公开分析 + 推测

## 一、DeepSeek-R1: 开源的推理训练蓝图

### 4阶段Pipeline

```
DeepSeek-R1 最终训练流程:

Stage 1: Cold-Start SFT
│   小量精心标注的long-CoT数据 → 微调base model
│   目的: 提供格式模板, 防止R1-Zero的语言混用问题
│   数据量: ~数千条高质量推理示例
│
Stage 2: RL Phase 1 (GRPO + Rule-based Reward)
│   GRPO训练 → 规则reward(数学正确性/代码测试)
│   仅reasoning任务(math/code/logic)
│   "Aha moment"在此阶段自然涌现!
│
Stage 3: Rejection Sampling + SFT
│   从RL模型生成 → 过滤(规则+DL reward) → SFT数据
│   只保留正确+可读的回答
│   合并reasoning数据 + non-reasoning数据(写作/QA/翻译等)
│   目的: 恢复通用能力(防止RL specialization)
│
Stage 4: RL Phase 2 (GRPO + All-task Rewards)
│   规则reward + DL reward → 所有任务类型
│   一致性reward: 防止语言混用(中英文交叉)
│   最终模型: DeepSeek-R1
```

### R1-Zero vs R1 最终版

| | **R1-Zero** (纯RL) | **R1** (4阶段) |
|---|-------------------|---------------|
| Cold-start SFT | 无 | **有**(~数千条) |
| 语言混用 | **严重**(中英交叉) | **轻微**(一致性reward) |
| 格式问题 | **严重**(结构混乱) | **良好**(SFT模板) |
| 推理能力 | **强**(涌现) | **更强**(4阶段叠加) |
| 通用能力 | **下降**(RL specialization) | **保持**(non-reasoning数据) |
| 可读性 | **差** | **好** |

### "Aha Moment" 涌现现象

DeepSeek-R1-Zero在GRPO训练过程中自发涌现了**自我反思**行为:

```
模型生成的示例 (训练step ~200):
"Let me calculate this step by step...
Result: 42

Wait, let me double-check that.
Actually, I think I made an error in step 3.
Let me reconsider the approach...

[重新计算]
The correct answer should be 38."
```

**关键特征**:
1. **非编程**: 模型没有被提示"请自我反思" → 自发出现
2. **训练step依赖**: 在某个训练step后才涌现 → 不是初始化时就有的
3. **推理质量提升**: 出现"aha moment"后, 后续推理更准确
4. **跨任务**: 不仅在数学, 在代码和逻辑推理中也出现

**理论解释** (基于GRPO组归一化):
- 组相对优势让模型探索更多推理路径
- 长推理链如果最终正确 → reward高 → 组内优势正 → 被增强
- 模型发现"反思→修正→正确"路径 → 反思行为被强化
- **涌现 = RL探索发现更优策略的自然结果**

### 蒸馏 vs RL训练

| 目标模型 | 直接RL训练 | **从R1蒸馏** |
|----------|-----------|-------------|
| Qwen-1.5B | 性能差 | **显著更好** |
| Qwen-7B | 性能中等 | **更好** |
| Qwen-32B | 性能好 | **更好** |
| Llama-8B | 性能中等 | **更好** |
| Llama-70B | 性能好 | **略好** |

**关键发现**: 蒸馏比在小模型上直接做RL更高效 → 推理模式可迁移 → 大模型的推理能力可"压缩"到小模型.

**原因**: RL训练需要大模型才能充分探索推理空间 → 小模型探索不足 → 但蒸馏直接继承大模型已发现的推理模式.

## 二、OpenAI o1/o3: 推测的训练方法

### 已知事实

1. **RL on Chain-of-Thought**: o1/o3确实用RL训练推理过程
2. **Test-time compute scaling**: 模型动态分配更多"思考时间"给难题
3. **Hidden thinking**: 推理过程不公开(仅显示摘要)
4. **o3-mini**: 小模型也能推理 → 蒸馏是可能的

### 高可信度推测

| 推测 | 可信度 | 依据 |
|------|--------|------|
| RL on CoT推理 | 高 | OpenAI确认 + DeepSeek-R1验证 |
| Process Reward Models | 中高 | Lightman et al. 2023验证PRM有效 |
| Outcome + Process reward混合 | 中高 | o1推理链较长 → 可能有步骤级信号 |
| 自我改进循环 | 中 | 迭代生成→过滤→训练 |
| 蒸馏(o3-mini) | 中高 | o3-mini推理能力强 → 可能蒸馏自更大模型 |
| Curriculum learning | 低中 | 难度递进是常见训练策略 |
| 推理时搜索(beam-like) | 中 | test-time compute = 更多推理token |

### o1/o3 vs DeepSeek-R1 对比

| | **o1/o3** | **DeepSeek-R1** |
|---|----------|----------------|
| **训练方法** | RL + (推测)PRM | GRPO + 规则reward |
| **Reward类型** | (推测)Outcome + Process | **Outcome-only**(规则) |
| **是否需要PRM** | (推测)可能需要 | **不需要**(组baseline替代) |
| **Critic模型** | (推测)可能需要 | **不需要**(GRPO消除) |
| **公开程度** | 不公开 | **完全公开** |
| **模型大小** | 未知(推测>200B) | 67B(DS-V3) + 蒸馏系列 |
| **推理token** | 数千-数万 | 数百-数千 |
| **test-time scaling** | 有(reasoning effort可调) | 有(think token长度随任务变化) |

### 关键差异: PRM vs Outcome-only

**PRM (Process Reward Model)**: 每个推理步骤评分 → 步骤级监督 → 更精细但需要标注
**Outcome-only**: 只看最终答案是否正确 → 简单但推理过程无监督

DeepSeek-R1证明: **Outcome-only + GRPO → 推理自然涌现, 无需PRM!**

这挑战了之前"必须PRM才能训练推理模型"的假设. 可能的解释:
1. GRPO组比较 → 模型自发探索有效推理路径 → 自我监督
2. 大模型有足够的"先验知识" → RL只需引导方向, 不需步骤级信号
3. Outcome reward足够 → 正确答案的推理链通常也是合理的

但PRM可能在以下场景更好:
1. **小模型**: 探索不足 → 需要步骤级引导
2. **复杂推理**: 多步骤任务 → 中间步骤错误导致最终错误
3. **安全性**: PRM可以检测危险推理步骤

## 三、Reasoning Model 训练的5个关键设计决策

### 1. Cold-start vs Pure RL

```
Pure RL (R1-Zero风格):
  优点: 最小人工干预, 推理真正涌现
  缺点: 语言混用, 格式差, 通用能力下降

Cold-start SFT (R1风格):
  优点: 格式好, 语言一致, 通用能力保持
  缺点: SFT数据可能限制推理多样性

推荐: Cold-start → 少量高质量模板 → RL自由探索
```

### 2. Reward类型选择

```
Rule-based reward:
  数学: 答案正确性检查
  代码: 单元测试通过率
  格式: 结构化输出格式
  优点: 精确, 无reward hacking, 无需训练RM
  缺点: 覆盖范围有限

DL reward model:
  人类偏好训练 → 更广泛覆盖
  优点: 覆盖写作/翻译/创意等非规则任务
  缺点: reward hacking风险, 训练成本高

推荐: Rule-based为主(推理) + DL辅助(通用)
```

### 3. 推理长度控制

```
Budget forcing:
  设置最小/最大思考token数
  优点: 防止过短(不充分推理)或过长(浪费时间)
  缺点: 硬限制可能截断有效推理

自适应长度:
  模型自己决定推理长度(简单问题短, 复杂问题长)
  优点: 灵活, 高效
  缺点: 可能对简单问题过度推理

推荐: budget forcing weight=0.5(软约束) + 自适应
```

### 4. 蒸馏 vs 直接RL

```
蒸馏:
  大模型RL → 生成推理数据 → 小模型SFT
  优点: 高效(小模型不需要RL探索), 质量好
  缺点: 小模型推理多样性受限(继承大模型模式)

直接RL (小模型):
  小模型直接GRPO → 自主探索
  优点: 推理多样性, 可能发现新推理路径
  缺点: 探索不足, 性能通常不如蒸馏

推荐: 蒸馏为主(快速获得推理能力) + 小规模RL(增加多样性)
```

### 5. 通用能力保持

```
问题: RL训练后通用能力下降(reasoning specialization)
  写作质量下降, 翻译变差, 创意丢失

解决: Stage 3 non-reasoning数据
  SFT阶段混入写作/翻译/QA数据 → 恢复通用能力
  RL Phase 2加入所有任务类型的reward → 全面优化

推荐: 必须混入non-reasoning数据(占比~30-40%)
```

## 四、verl中的Reasoning训练实现

基于我的源码阅读, verl支持完整的reasoning训练pipeline:

```python
# verl配置示例 (GRPO reasoning训练)
actor_rollout_ref:
  rollout:
    n: 8              # 每prompt 8个response (GRPO组大小)
    temperature: 0.6  # 推理采样温度
  actor:
    clip_ratio: 0.2   # PPO clip
    loss_agg_mode: "seq-mean-token-mean"  # 推理loss聚合

algorithm:
  adv_estimator: grpo  # GRPO组归一化
  gamma: 1.0
  lam: 1.0
  use_kl_in_reward: true  # KL约束
  kl_penalty: kl  # KL惩罚类型

reward_model:
  # Rule-based: math accuracy + code test
  # 或 DL reward model
```

**verl已有**:
- GRPO outcome-only advantage (组归一化)
- PPO clip loss (策略更新安全)
- KL penalty (防止偏离ref)
- PrefixGrouper (n=8 prefix缓存节省58%计算)
- Rollout backend: vLLM/SGLang async server
- Multi-stage training: 可配置cold-start → RL → SFT → RL

## 五、实用结论与行动建议

1. **GRPO是reasoning训练的最佳起点**: 无critic, outcome-only, 推理涌现
2. **Cold-start SFT必要**: ~数千条高质量CoT → 防止格式/语言问题
3. **Rule-based reward优于DL RM**: 精确无hacking, 数学/代码天然适合
4. **4阶段pipeline**: Cold-start → GRPO → Rejection+SFT → GRPO(all tasks)
5. **蒸馏比小模型直接RL更好**: 继承大模型推理模式 → 质量更高
6. **Non-reasoning数据必须混入**: 防止通用能力下降(~30-40%占比)
7. **verl已支持完整pipeline**: 只需配置GRPO + rule reward + n=8
8. **Budget forcing**: think=4-8最优 (软约束weight=0.5)
9. **o1/o3可能用PRM**: 但R1证明outcome-only也能涌现推理 → PRM不是必需的
10. **下一步**: 在RTX 4090上用verl跑7B GRPO reasoning训练(数学任务)

Sources:
- [DeepSeek-R1 Technical Report (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [Lightman et al. "Let's Verify Step by Step" (PRM)](https://arxiv.org/abs/2305.20050)
- [Nathan Lambert analysis of o1 training](https://interconnects.ai/)
- [OpenAI o3 ARC-AGI achievement](https://openai.com/index/introducing-o3/)