# Frontier RL for Reasoning: DAPO + Dr.GRPO — GRPO的两大改进

> 2026-06-07 | 两篇2025前沿论文分析: DAPO(阿里/Qwen)和Dr.GRPO(西北大学/Peng Lab), GRPO的关键缺陷与改进

## 一、DAPO: Dynamic Advantage Policy Optimization (arXiv 2503.14476)

### 作者与背景

- **团队**: 阿里巴巴/Qwen团队 (Yufeng Yuan, Yuqi Huo等)
- **动机**: GRPO在LLM推理训练中有4个关键缺陷 → DAPO提出4项改进
- **地位**: DAPO是Qwen2.5-Max的RL训练算法

### GRPO的4个缺陷

1. **奖励归一化漂移 (Reward Norm Drift)**: 不同难度组的reward分布差异大 → 组归一化后advantage不稳定
   - 简单组: reward=[1,1,1,0.3] → μ=0.825, σ=0.31 → 优势范围[-2.66, 0.56]
   - 困难组: reward=[0.1,0,0,0] → μ=0.025, σ=0.05 → 优势范围[-0.5, 1.5]
   - **不同组的advantage尺度不一致** → 梯度不稳定

2. **熵坍塌 (Entropy Collapse)**: 训练后期策略过于确定性 → 探索不足 → 无法发现更好的推理路径
   - GRPO组归一化 + clip → 策略快速收敛到局部最优 → 停止探索

3. **长度奖励劫持 (Length Reward Hacking)**: 长response → 更多token → 更高总reward → 模型变冗长
   - GRPO的outcome-only reward不惩罚长度 → 模型学会写冗长废话来凑分

4. **梯度不稳定 (Gradient Instability)**: 组内reward全部相同 → σ=0 → advantage=0 → 无梯度 → 跳步
   - 我们实验中也观察到: loss=0.0000 → 组内所有response reward相同 → 无更新

### DAPO的4项改进

#### 1. 动态优势归一化 (Dynamic Advantage Normalization)

不再用组内归一化(r-μ)/σ, 而是**跨组全局归一化**:
- 优势 = (r - μ_global) / σ_global
- μ_global = 全batch所有response的平均reward
- σ_global = 全batch的标准差
- **效果**: 不同难度组的advantage尺度一致 → 梯度稳定

**与我们Exp4的联系**: 我们用组内归一化 → variance↓39.9% → 但组间尺度不一致 → DAPO用全局归一化解决此问题

#### 2. 解耦clip (Decoupled Clip)

PPO-clip对ratio clip at [1-ε, 1+ε], 但DAPO分离上下clip:
- 上clip: ratio > 1+ε_upper → clip (防止过度增加好动作的概率)
- 下clip: ratio < 1-ε_lower → clip (防止过度减少坏动作的概率)
- ε_upper ≠ ε_lower → **不对称clip**
- 允许策略更自由地降低坏动作概率(更大的ε_lower) → 更快修正错误

#### 3. 动态采样 (Dynamic Sampling)

GRPO固定n个response → 当模型变好, 组内reward趋同 → σ=0 → 无更新

DAPO: **动态调整采样数n**
- 当组内reward多样性低(σ小) → 增加n → 更多探索
- 当组内reward多样性高 → 减少n → 更高效
- **效果**: 避免"无梯度"的跳步问题 → 持续学习

**与我们实验的联系**: 我们观察到loss=0.0000 → 组内reward相同 → 无更新 → DAPO的动态采样正是解决此问题!

#### 4. Token-level Loss

GRPO用response-level loss: L = -Σ (advantage × Σ_t log_prob_t) / Σ_t mask_t
→ 长response贡献更多token → 更多梯度 → 隐性鼓励长度

DAPO改为**token-level loss**:
- L = Σ_t (-advantage_t × log_prob_t) / Σ_t mask_t
- advantage_t = advantage (same for all tokens in a response)
- **效果**: 短response和长response贡献相同梯度量 → 不鼓励冗长

### DAPO实验结果

| Benchmark | GRPO | DAPO | Improvement |
|-----------|------|------|------------|
| AIME 2024 | baseline | +3-4pts | 显著提升 |
| AMC 2023 | baseline | +3pts | 显著提升 |
| Minerva Math | baseline | +3-4pts | 显著提升 |

**关键**: DAPO在所有推理benchmark上一致优于GRPO → 4项改进叠加有效

---

## 二、Dr.GRPO: 修复GRPO的两大偏差 (arXiv 2503.20783)

### 作者与背景

- **团队**: 西北大学 Hao Peng Lab (Yushi Hu等)
- **动机**: GRPO有两个系统性偏差 → Dr.GRPO诊断并修复
- **名字**: "The Doctor Is In" → 诊断(diagnose) + 修复(fix)

### GRPO的两个偏差

#### 1. 奖励偏差 (Reward Bias)

**问题**: GRPO组归一化(r-μ)/σ引入虚假方差

推导:
- 设response i的token-level reward为r_i = Σ_t reward_token_t
- 组归一化: advantage_i = (r_i - μ) / σ
- σ² = (1/n)Σ(r_i - μ)² → 包含随机噪声
- **噪声方差**: 即使两个response质量相同, 采样的reward波动 → σ非零 → 产生虚假advantage

**影响**: 模型学到"如何在噪声中获得高advantage"而非"如何真正提高质量"

**Dr.GRPO修复**: reward shaping + token-level indicators
- 给每个token一个indicator: 是否属于response的特定部分
- 根据indicator调整reward → 去除噪声方差的影响

#### 2. 长度偏差 (Length Bias)

**问题**: GRPO的KL惩罚per-token → 短correct response惩罚少, incorrect response惩罚多

推导:
- KL penalty = Σ_t (log π_θ(y_t) - log π_ref(y_t)) × mask_t
- 短correct response (5 tokens): KL = Σ_5 KL_t ≈ 0.05β
- 长incorrect response (100 tokens): KL = Σ_100 KL_t ≈ 1.0β
- **net advantage**: correct短 = reward - 0.05β ≈ 1.0, incorrect长 = reward - 1.0β ≈ 0.0
- → 短correct比长incorrect advantage更高 → **隐性鼓励简短但不惩罚正确长答案**

等等, 这看起来是好事? 但实际问题是反过来的:
- **长response获得更多梯度** → 模型学会写冗长推理来"填充"token → 即使答案错误, 更多token = 更多梯度信号 → 策略被长response引导

**Dr.GRPO修复**: KL惩罚在**序列级**而非token级
- KL_penalty = β × Σ_t KL_t / Σ_t mask_t → **平均per-token KL**
- 短长response的KL惩罚公平 → 不鼓励冗长

### Dr.GRPO实验结果

| Benchmark | GRPO | Dr.GRPO | Improvement |
|-----------|------|---------|------------|
| AIME 2024 | baseline | +3.4pts | 显著 |
| AMC 2023 | baseline | +3.2pts | 显著 |
| Minerva Math | baseline | +3.8pts | 显著 |
| MATH-500 | baseline | +4.0pts | 显著 |
| GPQA Diamond | baseline | +改善 | 科学推理也改善 |

**关键**: 去除两个偏差后 → 一致3-4pts提升 → 说明偏差是GRPO的系统性问题

---

## 三、与我们实验的联系

### 我们观察到的问题

1. **loss=0.0000**: 组内reward全部相同 → σ=0 → advantage=0 → 无更新
   → **DAPO的动态采样正是解决此问题**

2. **偏向重复**: 模型输出`33333333` → 长response相同token
   → **Dr.GRPO的长度偏差解释此问题!** per-token KL → 长response贡献更多梯度 → 偏向重复

3. **SFT→GRPO=100%**: SFT暖启动避免了从随机探索 → 组内reward多样性高 → σ非零 → 有梯度
   → DAPO的动态采样 ≈ SFT暖启动的效果(确保组内多样性)

4. **PPO训练reward高但eval低**: PPO的critic估计不稳定 → 类似DAPO指出的梯度不稳定

### 改进方向

如果将DAPO和Dr.GRPO的技术应用到我们的mini GRPO pipeline:
1. **全局归一化**: 不用组内(r-μ)/σ, 用跨batch全局归一化 → 更稳定的advantage
2. **序列级KL**: KL/num_tokens → 不鼓励冗长 → 防止重复问题
3. **动态采样**: 当σ<阈值时增加n → 确保每步有梯度
4. **Token-level loss**: 短长response贡献相同梯度 → 公平

**预估效果**: 纯GRPO可能从60%提升到80-90%!

---

## 四、理论深度: 三者统一框架的演进

| 世代 | 算法 | 核心 | 改进 |
|------|------|------|------|
| v1 | PPO | Critic V(s)作baseline | 稳定但贵(4模型) |
| v2 | GRPO | 组均值μ替代critic | 简化(2模型)但有偏差 |
| v2.5 | Dr.GRPO | GRPO + 去偏差 | 修复长度/奖励偏差 |
| v3 | DAPO | GRPO + 4项改进 | 动态优势+解耦clip+动态采样+token loss |

**演进方向**: PPO→GRPO(简化)→Dr.GRPO(修复)→DAPO(增强)→未来(混合?)

**数学基础不变**: 所有方法都优化 max E[r] - β KL(π||π_ref)
但实现细节决定实际性能 → **"理论等价≠实践等价"** → 这正是我们从实验中观察到的!

---

## Sources

- [DAPO: An Open-Source LLM RL System](https://arxiv.org/abs/2503.14476) — arXiv 2503.14476
- [Dr.GRPO: Fixing GRPO for Reasoning](https://arxiv.org/abs/2503.20783) — arXiv 2503.20783
- [DeepSeek-R1: GRPO原始论文](https://arxiv.org/abs/2402.03300) — arXiv 2402.03300
- [DAPO blog analysis](https://www.superannotate.com/blog/dapo-reinforcement-learning)
- [DAPO on HuggingFace](https://huggingface.co/papers/2503.14476)