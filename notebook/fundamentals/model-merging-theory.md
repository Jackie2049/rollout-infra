# 模型合并理论 — 从Task Arithmetic到TIES/DARE

> 2026-06-07 | 无需重新训练就能组合模型能力: 权重空间的代数运算

## 概述

模型合并(Model Merging)是直接在权重空间组合多个微调模型的技术——不需要额外训练或数据。从简单的权重平均到SLERP/TIES/DARE，模型合并已成为LLM领域的重要工具。

对AI专家而言，模型合并连接了多个领域：
- 优化理论(权重空间的几何)
- 训练理论(task vector = 微调delta)
- 模型压缩(MoE from merged models)
- 实用性(组合不同能力→更强大模型)

## 一、Task Arithmetic — 基础框架

### 1.1 Task Vector定义

```
Task Vector: Δ_task = W_finetuned - W_base

含义: 微调改变了什么? → 参数差异 = "任务知识"

→ Task vector可以被当作向量运算:
  W_new = W_base + α × Δ_task1 + β × Δ_task2 - γ × Δ_task3

  → 加法: 增加任务能力(如: +数学推理)
  → 减法: 减去 undesired 行为(如: -有毒输出)
  → 混合: 组合多种能力(如: +数学+代码-有毒)
```

**实验验证**(Ilharco et al., 2022):
- T0模型微调多个任务 → task vector可以加减 → 组合能力!
- 但简单加法→任务干扰(task interference) → 简单平均效果差

### 1.2 Task Vector的数学性质

```
Δ_task = W_ft - W_base 的性质:

1. 稀疏性: 大部分参数变化接近0 → 只有少数参数真正改变
   → 这与DARE的发现一致: 90%的delta是冗余的!

2. 方向性: Δ_task有明确的方向 → 代表"从base到task的路径"
   → SLERP沿球面插值 → 保持方向 → 更好

3. 线性假设: W_base + Δ_task1 + Δ_task2 ≈ 同时会两个任务
   → 简单但不够准确 → 任务干扰使得非线性效果
```

## 二、简单权重平均 — 最基础的方法

### 2.1 线性平均

```
W_merged = (1/N) Σ W_i

优点: 简单, 快速, 无额外计算
缺点:
  1. 任务干扰: 不同task vector方向冲突 → 平均化→能力损失
  2. 丢失几何: 高维权重空间中线性平均→破坏方向信息
  3. 无选择性: 所有参数平等参与 → 冗余参数也被平均
```

**Model Soups**(Wortsman et al., 2022):
```
改进: 不一次性平均 → 先单独评估 → 只选最好的权重组合

Soup = average of top-k models on validation set

→ 比简单平均更好 → 但仍受干扰影响
```

## 三、SLERP — 球面线性插值

### 3.1 为什么SLERP比线性平均好?

```
线性插值: W = (1-t)W_A + t W_B
  → 在直线上的点 → 高维空间中直线会穿过球面内部 → 丢失方向

SLERP: 沿球面插值
  W = [sin((1-t)θ) / sin(θ)] × W_A + [sin(tθ) / sin(θ)] × W_B

  θ = arccos(W_A · W_B / (||W_A|| × ||W_B||)) ← 两向量间的角度

  → 沿球面弧线插值 → 保持方向 → 不丢失几何信息

直觉: 两个向量定义一个"方向球面" → 球面插值保持在这个球面上 → 不"掉入"内部
```

### 3.2 SLERP的数学推导

```
给定: v₀ 和 v₁ (两个模型的权重向量)
求: v(t) = 球面上的中间点 (t ∈ [0,1])

设 v(t) = a(t) × v₀ + b(t) × v₁

约束:
  ||v(t)||² = 1 (球面上) → a² + b² + 2ab cos(θ) = 1
  v(0) = v₀ → a=1, b=0
  v(1) = v₁ → a=0, b=1

解:
  a(t) = sin((1-t)θ) / sin(θ)
  b(t) = sin(tθ) / sin(θ)

→ t=0.5 → a=b=sin(θ/2)/sin(θ) = 1/(2cos(θ/2)) → 两向量等权
→ θ接近0 → SLERP≈线性(两向量方向一致)
→ θ大 → SLERP与线性差异大 → 方向差异大时SLERP更重要
```

### 3.3 限制

```
SLERP只能合并两个模型 → 多个模型需要逐对合并(sequential)
  → A+B → Merged_AB → Merged_AB+C → ...

→ 顺序影响结果 → 不是最优 → 但比线性平均好
→ MergeKit支持SLERP → 是最受欢迎的合并方法之一
```

## 四、TIES — Trim, Elect, Sign

### 4.1 问题: 任务干扰

```
两个task vector相加时:
  Δ_1[i] > 0 (任务1使参数i增大)
  Δ_2[i] < 0 (任务2使参数i减小)
  → Δ_1[i] + Δ_2[i] ≈ 0 → 参数i回到base → 两个任务都丢失!

这就是任务干扰: 不同任务的参数更新方向冲突 → 相加后互相抵消
```

### 4.2 TIES三步

```
Step 1: Trim — 删除冗余参数
  对每个task vector, 只保留top-k%最大的参数(绝对值)
  其余设为0 → 消除噪声+减少干扰源

  为什么要trim?
  → 90%的delta参数值很小 → 对任务贡献小 → 但参与合并时引入干扰
  → trim只保留"真正重要的参数" → 减少干扰面

Step 2: Elect — 解决方向冲突
  对于被多个task vector修改的同一参数:
  如果Δ_1[i]>0 且 Δ_2[i]>0 → 方向一致 → 保留
  如果Δ_1[i]>0 且 Δ_2[i]<0 → 方向冲突 → 取多数(sign majority vote)

  → Elect = "民主投票" → 方向冲突时多数决定

Step 3: Sign-gated Merge — 只合入sign一致的参数
  最终合并: 只加入与base方向一致的参数修改
  → 防止"反方向修改"破坏base的知识

  W_merged = W_base + Σ (elected, sign-consistent deltas)
```

### 4.3 TIES vs 线性平均

```
线性平均: 简单但干扰严重 → 合并后各任务性能下降20-30%
TIES: 修剪+投票+sign gating → 合并后各任务性能仅下降5-10%

→ TIES的关键洞察: 不是所有参数都重要 → 不是所有方向都一致 → 需要选择性合并
→ 与我们的RL实验类比: 不是所有reward都可靠 → GRPO组归一化选择性放大好response
```

## 五、DARE — Drop And Rescale

### 5.1 DARE核心发现

```
震惊发现: 90%的fine-tuned delta参数可以随机删除 → 几乎不影响性能!

DARE步骤:
1. 对每个task vector, 随机删除p%的参数(设为0)
   p=0.9 → 只保留10%的delta!

2. Rescale剩余参数: Δ_rescaled = Δ_remaining × 1/(1-p)
   → 确保总magnitude不变: E[|Δ_dare|] ≈ |Δ_original|

   数学: E[Δ_dare[i]] = E[Δ[i] × (1/(1-p) if kept, 0 if dropped)]
         = Δ[i] × (1-p) × (1/(1-p)) + Δ[i] × p × 0
         = Δ[i] ← 期望值完全不变!

   → DARE是delta的"无偏估计" → 随机删除不改变期望值
```

### 5.2 为什么DARE有效?

```
1. Delta稀疏性: 大部分delta参数值很小 → 删除不影响功能
2. 过参数化: LLM参数远多于需要 → 删除90%仍够用
3. Rescaling补偿: 保留的10%参数被放大10x → 信息量保持

→ 与我们之前MoE实验的类比:
   MoE: 每次只激活2/16专家(12.5%) → 但训练在所有专家上 → 等效rescaling
   DARE: 随机删除90% → 保留10%×10x → 信息量不变
```

### 5.3 DARE-TIES组合

```
最佳实践: 先DARE→再TIES → DARE-TIES

Step 1: DARE(随机删除90% + rescale) → 大幅减少参数量
Step 2: TIES(trim+elect+sign) → 在DARE处理后的delta上做选择性合并

→ DARE减少冗余 → TIES解决冲突 → 两步组合 → 最佳效果!

→ MergeKit支持DARE-TIES → 是当前最推荐的合并方法
```

## 六、前沿方向 (2025)

### 6.1 Passthrough/Frankenmerging

```
不是合并权重 → 而是堆叠层!

  Model A: 32层 → 取前16层
  Model B: 32层 → 取后16层
  Frankenmerger: 32层(前16 from A + 后16 from B)

→ 创建更深模型 → 但架构必须兼容(维度一致)
→ "Goliath" merge: 从7B模型堆叠到更大 → 社区流行

→ 与我们之前FSDP内存模拟器连接: 更深模型→更多内存→需要ZeRO-3
```

### 6.2 SVD-based Merging

```
用SVD分解权重矩阵 → 保留主要成分 → 在低维空间合并 → 再重建

  W = U Σ Vᵀ → 保留top-k奇异值 → W_lowrank = U_k Σ_k V_kᵀ

→ 合并W_lowrank_A和W_lowrank_B → 干扰更少(信息集中在主成分)
→ 与我们的MLA实验连接: MLA也是低秩投影 → d_c=512 ← 保留主成分
```

### 6.3 MoE from Merged Models

```
不合并权重 → 而是把多个模型作为MoE的expert!

  Router选择哪个model回答 → 每个model是独立expert
  → 保留各模型完整能力 → 无干扰!

→ 与我们的MoE实验连接:
   MoE Python 6-164x慢于Dense → scatter/gather瓶颈 → 需FusedMoE kernel
   → 合并模型作MoE → expert切换开销 → 需专用实现

→ DeepSeek-V3: 256细粒度expert + 1 shared → 不是从merged model构建
   但原理相似: 多个小expert比1个大model更灵活
```

### 6.4 Representation Merging

```
不在权重空间合并 → 在激活(activation)空间合并!

→ 合并中间层表示而非参数 → 更灵活 → 不要求架构兼容
→ 但需要训练时干预 → 不如权重合并简单

→ 与我们的interpretability实验连接:
   Feature steering: 在等号位置注入activation → 改变输出 → 43%成功率
   → 这就是representation-level的"模型编辑"!
```

## 七、实践指南

### 7.1 合并方法选择

```
两个模型?
├─ 方向差异小(θ<30°) → 线性平均够了(简单快速)
├─ 方向差异大 → SLERP(保持几何)
└─ 需要精确控制 → Task Arithmetic(α调参)

多个模型?
├─ 简单 → 线性平均(效果差)
├─ 推荐 → DARE-TIES(最佳实践)
├─ 不同能力 → TIES(减少干扰)
└─ 架构兼容 → Passthrough(堆叠层)

特殊需求?
├─ 减去有毒输出 → Task Arithmetic(-α×Δ_toxic)
├─ 创建MoE → 把合并模型作expert
└─ 低秩优化 → SVD merging
```

### 7.2 MergeKit YAML示例

```yaml
models:
  - model: meta-llama/Llama-3-8B
    # base model
  - model: meta-llama/Llama-3-8B-Instruct
    parameters:
      density: 0.5  # TIES top-k%
      weight: 0.5   # 合并权重
  - model: some-code-llama
    parameters:
      density: 0.5
      weight: 0.5

merge_method: dare_ties  # 最佳方法
base_model: meta-llama/Llama-3-8B
parameters:
  int8_mask: true  # 量化合并(省内存)
```

## Sources

- [Task Arithmetic](https://arxiv.org/abs/2212.04089) — Ilharco et al., 2022
- [TIES Merging](https://arxiv.org/abs/2306.01708) — Jain et al., 2023
- [DARE Merging](https://arxiv.org/abs/2311.03099) — Yu et al., 2023
- [MergeKit](https://github.com/arcee-ai/mergekit) — 开源合并框架
- [Model Soups](https://arxiv.org/abs/2203.05482) — Wortsman et al., 2022