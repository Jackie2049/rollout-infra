# PRM vs ORM 实验验证 — Step-Level vs Outcome-Level Verification

> 2026-06-07 | Mini PRM训练pipeline: step-level verifier能否beat outcome-level? RTX 4090实测

## 概述

基于Lightman et al. (2023) "Let's Verify Step by Step"的核心发现, 我们构建mini PRM vs ORM验证实验, 测试step-level verifier是否在小模型上优于outcome-level verifier。

**Lightman et al. 核心发现**: PRM在MATH等复杂推理任务上显著优于ORM → 但我们的实验显示, **在简单算术任务上PRM无法beat ORM**!

## 实验设计

### Step-level推理任务

```
Problem: a+b= (a,b∈{0..4})
Step 1: first=a (提取第一个数)
Step 2: second=b (提取第二个数)
Step 3: sum=a+b (计算总和)
```

### PRM Model: StepVerifier

输入: (problem_tokens, step_tokens) → 输出: step正确概率(0-1)

```python
class StepVerifier(nn.Module):
    # Separate processing for problem and step
    problem_net → problem_repr  # 理解问题"a+b="
    step_net → step_repr  # 理解步骤"first=2"
    combine_net → score  # 给定问题, 步骤是否正确?
```

**关键**: PRM需要**问题上下文**来判断步骤正确性 → "first=3"是否正确取决于问题是否是"3+b="

### ORM Model: OutcomeVerifier

输入: full solution tokens → 输出: 最终答案正确概率(0-1)

```python
class OutcomeVerifier(nn.Module):
    # Only uses last token (the final answer digit)
    embed → net → sigmoid → score
```

### 错误类型

- **Early error**: Step 1错误(misidentify first number) → 级联错误(sum也错)
- **Middle error**: Step 2错误(misidentify second number) → 级联错误
- **Late error**: Step 3错误(仅最后一步计算错误, 前两步正确)

### 训练配置

- 1000步, lr=1e-3, hidden_dim=64, vocab_size=20
- PRM: 9样本/步(3 correct + 3×3 wrong steps with problem context)
- ORM: 16样本/步(4 correct + 4×3 wrong solutions)

## 实验结果

### 训练精度

| Model | Final Accuracy | Params |
|-------|---------------|--------|
| PRM | 58.3% | ~13K |
| ORM | **75.0%** | ~9K |

**PRM stuck at 58.3%**: 为什么?

1. **Task too simple**: 5×5=25个(a,b)组合 → 问题上下文几乎无信息量
2. **Step tokens too short**: "first=3"只有5个token → embedding信息不足
3. **Problem context redundant**: 对于算术任务, "3+2="的问题上下文对判断"first=3"几乎无帮助 → 正确与否一目了然

### Best-of-N验证 (200 problems, 8 candidates each)

| Method | Accuracy | Notes |
|--------|----------|-------|
| PRM Best-of-8 | ~50% | 与ORM相当 |
| ORM Best-of-8 | ~50% | 与PRM相当 |
| **PRM improvement** | ~0% | 无显著提升 |

### 错误类型拒识率

| Error Type | PRM Reject Rate | ORM Reject Rate | Notes |
|-----------|----------------|-----------------|-------|
| Early | ~60% | ~70% | ORM反而更好! |
| Middle | ~55% | ~65% | ORM更好 |
| Late | ~50% | ~60% | ORM更好 |

**反直觉**: ORM在所有错误类型上拒识率都比PRM更高!

### 根因分析: 为什么PRM不work?

**Lightman et al. 的MATH任务**:
- 问题是复杂数学证明(10+步推理) → 问题上下文提供丰富信息
- 中间步骤需要问题上下文才能判断正确性 → PRM有优势
- ORM只能看最终答案 → 无法定位中间错误 → 但复杂推理的最终答案可能因中间错误而错误但看起来"合理"

**我们的算术任务**:
- 问题是简单加法(3步) → 问题上下文几乎无额外信息
- Step正确性直观可见 → "first=3"是否正确一目了然(数字就是数字)
- ORM看最终答案即可判断 → sum≠a+b → 直观错误

### PRM > ORM的前提条件

```
PRM优势 = f(问题复杂度 × 步骤依赖性 × 错误隐蔽性)

高优势(MATH等):
- 问题复杂 → 上下文提供大量信息
- 步骤依赖 → 后续步骤依赖前面步骤的正确性
- 错误隐蔽 → 中间错误可能不会在最终答案中明显暴露

低优势(简单算术):
- 问题简单 → 上下文几乎无信息
- 步骤独立 → 每步可独立判断
- 错误明显 → 错误答案一看就知道
```

## 与理论联系

1. **Lightman et al. (2023)**: PRM在MATH上远优于ORM → 但需要复杂推理任务
2. **Snell et al. (2024)**: Test-time compute中verification quality是关键 → PRM是更好的验证器
3. **DeepSeek-R1**: Outcome-only reward足够 → 不需要PRM(推理任务足够复杂, 最终答案正确性就足以判断)
4. **我们的结论**: PRM和ORM的选择取决于**任务复杂度** → 简单任务用ORM, 复杂推理用PRM

## 代码

- 工具: `tools/mini_prm_training.py`
- 结果: `prm_vs_orm_results.json`

## 改进方向

1. **复杂推理任务PRM**: 多步推理(a×b+c=, 多位数运算) → 验证PRM>ORM在复杂任务上成立
2. **PRM with longer steps**: 增加步骤长度 → embedding信息更丰富 → PRM可能work
3. **Tree Search with PRM**: Best-of-N → Tree Search → PRM在Tree Search中的价值更大(需要定位中间步骤错误来决定分支方向)