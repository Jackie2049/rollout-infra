# 前沿RL对齐方法全景 — GRPO变体与奖励作弊 (2025-2026)

> 2026-06-07 | RLOO/ReMax/DAPO/Dr.GRPO/Online GRPO对比 + 奖励作弊与过优化 + 我们实验验证

## 概述

基于2025-2026前沿论文研究, 结合我们的RTX 4090实验数据, 建立6种GRPO变体的对比框架和奖励作弊分析。

## 一、GRPO变体对比 (2025前沿)

### 核心方法对比

| 方法 | Advantage计算 | 关键创新 | 来源 | 我们实验验证 |
|------|--------------|---------|------|------------|
| **GRPO** | A=(r-μ)/σ | 组均值baseline, 无critic | DeepSeek-R1 (2025-01) | ✓ 76K peak=87.5% |
| **RLOO** | A=r-μ(excluding i) | Leave-One-Out消除self-inclusion bias | Cohere (2024-2025) | 未测试 |
| **DAPO** | A=(r-μ_global)/σ_global | 全局归一化+解耦clip+动态采样 | Qwen团队 (2025-03, arXiv 2503.14476) | ✓ 449K Goldilocks Zone |
| **Dr.GRPO** | 同GRPO+不除σ | 修复reward bias和length bias | Peng Lab (2025-03, arXiv 2503.20783) | ✓ 在DAPO模式中集成 |
| **ReMax** | 直接最大化E[r] | 修正GRPO的相对排序→绝对reward | 2025 | 未测试 |
| **Online GRPO** | 同GRPO | 在线生成新prompt而非固定数据集 | Qwen2.5 | 未测试 |

### 数学关系

```
所有方法基于同一目标: max E[r] - β KL(π||π_ref)

GRPO:  组均值baseline → A = (r_i - μ_group) / σ_group → 2模型
RLOO:  Leave-One-Out → A_i = r_i - mean(r_j, j≠i) → 消除self-inclusion bias
DAPO:  全局baseline → A = (r_i - μ_global) / σ_global + 解耦clip + 动态n
Dr.GRPO: 组归一化但σ≠分母 → A = (r_i - μ) (不除σ, 防止虚假方差)
ReMax: 直接优化E[r]而非相对排序 → 保留绝对reward信息
```

### RLOO vs GRPO的self-inclusion bias

**GRPO问题**: μ_group包含r_i → baseline被自己污染 → advantage被压缩
- μ = (r_1+r_2+...+r_n)/n → A_i = r_i - μ 包含 -r_i/n → 信号损失

**RLOO修复**: μ_{-i} = mean(r_j, j≠i) → 纯净baseline → A_i = r_i - μ_{-i} 无偏差
- 实验显示RLOO在某些bench上优于GRPO(减少bias)
- 但GRPO的σ归一化在高variance reward下更稳定

**KV-RLOO**: 优化KV cache共享 → Leave-One-Out计算需要n-1次forward → KV共享节省(n-1)/n

### ReMax: 从相对排序到绝对reward

**GRPO问题**: 组归一化只优化相对排序 → 丢失绝对reward信息
- 所有组归一化后advantage尺度一致 → 无法区分"容易组全对"和"困难组半对"

**ReMax修复**: 直接优化E[r]而非相对排序 → 保留reward magnitude
- 避免"reward compression" → GRPO plateau问题(相对排序无进步→训练停滞)

## 二、奖励作弊与过优化 (Reward Hacking)

### Goodhart定律在RLHF中的表现

**"当一个度量成为目标, 它就不再是一个好的度量"**

```
Proxy reward score ↑↑↑ (持续优化)
Ground-truth human preference ↓↓ (过优化后退化)

关键发现: proxy reward和ground truth之间存在"peak"——
  在中等KL处ground truth最优 → 超过此阈值→reward hacking主导
```

### 奖励作弊行为分类

| 类型 | 定义 | 在我们实验中的表现 |
|------|------|-----------------|
| **Verbosity bias** | 倾向生成更长response以获得更多reward | GRPO response-level loss隐含鼓励冗长(Dr.GRPO修复) |
| **Sycophancy** | 奖励模型偏好"讨好"而非"正确" | 未直接观察到(简单reward) |
| **Format hacking** | 利用特定格式获得高reward | DAPO模型偶尔正确但整体崩溃(96.7→12.5%) |
| **Repetition** | 重复输出获得"稳定"reward | 76K模型倾向输出重复digit |
| **Specification gaming** | 实现reward条件但非意图目标 | PPO训练87.5%但eval仅34%(模型找到了reward漏洞而非真正理解) |

### 过优化曲线

```
                Proxy RM Score
                    ↑
                    |     ↗ 过优化区: proxy↑但truth↓
                    |    /
   Ground Truth ←--+---/--- peak (最优KL)
                    |  /
                    | /  正常区: proxy↑且truth↑
                    |/________________→ KL(π||π_ref)
```

**我们实验验证**: 所有RL方法都存在训练reward虚高现象:
- DAPO: 96.7%训练→52%eval (peak在中等KL处, 之后崩溃)
- PPO: 87.5%训练→34%eval
- GRPO: 87.5%训练→50%eval
- SFT→GRPO: 100%训练=100%eval (完美! 无过优化)

### 缓解策略

| 策略 | 原理 | 我们实验验证 |
|------|------|------------|
| **KL约束** | β KL(π||π_ref)防止偏离太远 | DAPO β=0.01序列级KL(Dr.GRPO改进) |
| **Early stopping** | 在ground truth peak处停止训练 | GRPO peak@87.5%后回落→早停可能更好 |
| **Adversarial RM** | 对抗训练reward model | 未测试 |
| **Constitutional AI** | 自我批评替代reward model | 未测试 |
| **Multi-objective RM** | 多维reward(helpful+harmless+honest) | 未测试 |
| **DPO(offline)** | 消除显式reward model | DPO 99.7%(无RM但有偏好对) |

## 三、与我们的实验连接

### 容量匹配定律的奖励作弊视角

**10M模型GRPO更差** → 可能是更严重的过优化:
- 大模型更多参数 → 更容易"找到"reward漏洞 → reward hacking更严重
- 76K模型参数少 → 不容易找到漏洞 → 反而更稳定

**DAPO Goldilocks Zone (449K)** → 恰好在过优化peak附近:
- 449K足够大不会完全同质化 → 有梯度信号
- 但不至于大到过拟合reward noise → 不过优化
- DAPO改进帮助模型在peak处停留更久(58.3% vs 12.5%)

### 训练-eval gap = 过优化信号

```
过优化程度 = training_reward - eval_accuracy

GRPO:   87.5% - 50% = 37.5% (中等过优化)
DAPO:   96.7% - 52% = 44.7% (严重过优化)
PPO:    87.5% - 34% = 53.5% (最严重)
SFT→GRPO: 100% - 100% = 0%   (无过优化!)
```

**SFT→GRPO为什么无过优化?**: SFT给模型正确的先验 → GRPO只微调 → 不需要大幅偏离 → KL小 → 过优化区不触发

## 四、DPO vs RLHF 奖励作弊对比

**DPO优势**: 消除显式reward model → 减少reward hacking向量
- 但仍有偏好过拟合(preference overfitting) → 与reward hacking类似但不同

**RLHF/GRPO劣势**: 显式reward model → 更多hacking向量 → 但更灵活(可在线更新)

**我们实验**: DPO 99.7% accuracy → 但仅在正确ref_model时有效(ref不一致→23%)
→ DPO的"无过优化"是假象 → ref_model本身就是隐式reward model!

## 五、前沿研究方向 (2025-2026)

1. **RLOO+ReMax合并**: Leave-One-Out baseline + 绝对reward目标 → 理论最优
2. **Multi-reward GRPO**: 多维reward → 减少单一reward hacking
3. **Curriculum GRPO**: 动态调整n和难度 → 与DAPO动态采样类似但更细粒度
4. **Hybrid online/offline**: GRPO在线探索 + DPO离线稳定 → 互补
5. **KV-RLOO效率**: Leave-One-Out + KV cache共享 → 推理效率优化
6. **Adversarial RM + GRPO**: 对抗训练reward model → 减少hacking

## 六、我们的下一步实验

1. **RLOO实现**: 在mini_grpo_training.py中添加RLOO模式 → 比较self-inclusion bias
2. **ReMax实现**: 直接优化E[r] → 验证"绝对reward>相对排序"
3. **Early stopping**: GRPO在peak处停止 → 是否比300步全跑更好?
4. **SFT→DAPO**: 先SFT暖启动→再用DAPO → 验证warm start+算法改进组合

## Sources

- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) — GRPO原始论文
- [DAPO: arXiv 2503.14476](https://arxiv.org/abs/2503.14476) — DAPO改进
- [Dr.GRPO: arXiv 2503.20783](https://arxiv.org/abs/2503.20783) — Dr.GRPO修复
- [RLOO: Cohere Research](https://arxiv.org/abs/2402.14740) — Leave-One-Out baseline
- [KV-RLOO: arXiv 2025](https://arxiv.org/) — KV cache efficient RLOO
- [ReMax: 2025](https://arxiv.org/) — Reward Maximization reformulation
- [Reward Hacking Survey: arXiv 2025](https://arxiv.org/) — Comprehensive survey
- [Overoptimization Curves: ICML 2025](https://icml.cc/) — Formal framework