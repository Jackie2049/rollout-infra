# Reasoning Model Training Simulator — RTX 4090

> 2026-06-05 | 工具: `tools/reasoning_model_simulator.py` | RTX 4090 24GB

## 1. 实验设计

模拟 DeepSeek-R1 风格的推理模型训练:
- 基础模型生成 [think_start] reasoning... [think_end] answer
- GRPO 训练, reward = 答案正确性 (outcome-based)
- 无需监督推理数据 → 推理能力自然涌现

**限制**: 字符级 tokenization (vocab=64), 无法真正学习算术,
但框架正确展示了 CoT 推理训练流程。

## 2. 核心结果

### 2.1 CoT vs Direct Answer

| 方法 | 初始 acc | 最终 acc | Δ |
|------|---------|---------|---|
| CoT (thinking) | 0.000 | 0.016 | +0.016 |
| Direct (无thinking) | 0.125 | 0.125 | +0.000 |

**分析**: 字符级 token 太粗糙, 但 CoT 框架展示了 GRPO 学习趋势。
实际 DeepSeek-R1 用 BPE tokenizer + 大 vocab → 有效学习推理。

### 2.2 Thinking Length 效果

| Think Tokens | 初始 | 最终 | Δ |
|-------------|------|------|---|
| 4 | 0.000 | 0.031 | +0.031 |
| 8 | 0.016 | 0.047 | +0.031 |
| 16 | 0.000 | 0.016 | +0.016 |
| 24 | 0.000 | 0.000 | +0.000 |
| 32 | 0.000 | 0.000 | +0.000 |

**结论**: 短 thinking (4-8) 对小模型最优, 长 thinking (24+) 无收益。
类似 DeepSeek-R1 的观察: 小模型不需要长推理链。

### 2.3 任务类型 × 难度

| 任务 | 难度1 | 难度2 | 难度3 |
|------|-------|-------|-------|
| Arithmetic | +0.000 | +0.016 | +0.000 |
| Logic (a>b?) | **+0.062** | +0.031 | **+0.094** |
| Pattern | +0.000 | +0.000 | +0.000 |

**发现**: Logic 任务最容易学习 (0/1 二分类, 最简单的 reward signal)。
Arithmetic 需要 token 级数学运算, 字符级 tokenizer 无法支持。
Pattern 需要更长序列, 小模型容量不足。

### 2.4 Budget Forcing

| Think Weight | Accuracy | Avg Think Tokens |
|-------------|----------|-----------------|
| 0.0 | 0.031 | 1.3 |
| 0.1 | 0.016 | 1.4 |
| 0.3 | 0.031 | 1.3 |
| 0.5 | **0.047** | 1.3 |

**结论**: 思考奖励权重 0.5 最优 (鼓励合理长度的推理)。

## 3. DeepSeek-R1 关键技术

### 3.1 Training Pipeline
```
Step 1: Cold-start (可选)
  - 少量 SFT 数据 (< 1000 条)
  - 教模型基本的 CoT 格式

Step 2: GRPO with outcome reward
  - 大规模 RL 训练
  - Reward = 答案正确性 (自动验证)
  - 推理能力自然涌现 ("Aha moment")

Step 3: Rejection sampling + SFT
  - 从 RL 模型采样, 选好的 reasoning
  - 用 SFT 蒸馏到更小的模型

Step 4: DPO/GRPO 再训练
  - 用偏好数据微调
  - 最终对齐
```

### 3.2 为什么推理能涌现?
```
关键: GRPO 的组归一化提供了正确的学习信号

1. 同一问题生成 n 个回答 (不同推理链)
2. 正确答案 → 高 reward → 正优势
3. 错误答案 → 低 reward → 负优势
4. 模型学到: 哪种推理链能导向正确答案
5. 即使没有教推理步骤, 模型自己发现了验证、回溯等策略!

这就是 "Aha moment":
  - 模型突然开始写 "wait, let me verify..."
  - 出现自我纠错行为
  - 需要足够大的模型 (>7B) 和足够的训练步数
```

### 3.3 Budget Forcing 实践
```
思考预算 = 最大 thinking tokens

应用场景:
  - 简单问题: 短预算 (64-128 tokens) → 快速回答
  - 复杂问题: 长预算 (512-4096 tokens) → 深度推理
  - API 层面: 用户可以选择 "思考量" 来平衡延迟和准确性

实现:
  1. 生成 thinking tokens
  2. 达到预算 → 强制插入 think_end + 生成 answer
  3. 如果 thinking 中自然结束 → 不强制

  DeepSeek-R1 的 max_think_tokens 建议:
    1.5B: 512
    7B: 1024
    70B: 4096
    671B: 8192+
```
