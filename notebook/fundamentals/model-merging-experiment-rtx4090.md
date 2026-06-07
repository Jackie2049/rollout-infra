# Model Merging Experiment — RTX 4090 实测

> 2026-06-07 | Task Arithmetic / SLERP / DARE-TIES 在76K小模型上的实测验证

## 概述

在RTX 4090上使用MiniGQATransformer(76K params, vocab=20, 算术任务a+b=)测试了5种模型合并方法。关键发现：**DARE 90% drop在小模型上失败**(与大模型文献矛盾), Task Arithmetic SFT完美转移, SLERP(SFT,SFT→GRPO)=100%。

## 一、实验设计

### 1.1 模型配置

```
MiniGQATransformer: hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=2
参数量: ~76K
任务: 单位加法 a+b=? (vocab=20, sum范围0-9)
训练: seed=42统一, 200步SFT, 300步GRPO, n_samples=8
```

### 1.2 四个基础模型

| 模型 | 训练方式 | Eval Accuracy |
|------|----------|---------------|
| Base | 随机初始化 | 0.0% |
| SFT-only | 200步SFT | **100%** |
| GRPO-only | 300步GRPO | 79% |
| SFT→GRPO | 200步SFT + 300步GRPO | **100%** |

## 二、Task Arithmetic — α扫参

### 2.1 SFT Task Vector Sweep

| α | Accuracy | 分析 |
|---|----------|------|
| 0.0 | 0.0% | = base model |
| 0.25 | 25.5% | 开始学到SFT知识 |
| 0.5 | 72.0% | 半步SFT已经很好 |
| 0.75 | **100%** | 最优α! |
| 1.0 | 100% | 完全恢复SFT |
| 1.5 | 88% | 过度放大→退化 |
| 2.0 | 70.5% | 继续退化 |

**关键洞察**: SFT task vector α=0.75已经达到100% → SFT训练形成了**非常紧凑**的知识表示，不需要完整的delta就能恢复！

### 2.2 GRPO Task Vector Sweep

| α | Accuracy | 分析 |
|---|----------|------|
| 0.0 | 0.0% | = base model |
| 0.25 | 21.5% | GRPO知识更难迁移 |
| 0.5 | 54% | |
| 0.75 | 69% | |
| 1.0 | 81% | 恢复GRPO原始性能 |
| 1.5 | 71.5% | 过度放大退化 |
| 2.0 | 47% | 严重退化 |

**对比**: GRPO task vector比SFT更难迁移 — α=0.25时SFT=25.5%而GRPO=21.5%, α=0.5时SFT=72%而GRPO=54%。GRPO训练形成的表示**更分散/更不稳定**。

### 2.3 Combined SFT + GRPO Task Vectors

| α_sft | α_grpo | Accuracy |
|-------|---------|----------|
| 0.5 | 0.5 | 50.5% |
| **1.0** | **0.5** | **87%** |
| 1.0 | 1.0 | 61.5% |
| 0.5 | 1.0 | 82% |

**最优组合**: α_sft=1.0, α_grpo=0.5 → 87%。全SFT delta + 半步GRPO delta → SFT提供稳定基础, GRPO提供小幅增强。α_grpo=1.0反而退化到61.5% → GRPO delta与SFT delta**部分冲突**。

## 三、SLERP vs Linear Average

### 3.1 不兼容模型(SFT vs GRPO)

| 方法 | Accuracy |
|------|----------|
| SLERP(SFT, GRPO, t=0.5) | 56.5% |
| Linear avg(SFT, GRPO) | 52.0% |

SLERP仅比linear好4.5% → SFT和GRPO模型**方向差异大**, 任何合并都会损失大量信息。

### 3.2 SLERP t-Sweep (SFT vs GRPO)

| t | Accuracy | 含义 |
|---|----------|------|
| 0.1 | **100%** | 几乎纯SFT → 保持完美 |
| 0.2 | **100%** | 还是几乎纯SFT |
| 0.3 | 73% | 开始混入GRPO → 退化 |
| 0.4 | 54% | |
| 0.5 | 61% | SLERP非线性导致比0.4好 |
| 0.6 | 71.5% | 更多GRPO反而回升? |
| 0.7 | 73% | |
| 0.8 | 72% | |
| 0.9 | 81% | 几乎纯GRPO → 接近GRPO原始79% |

**洞察**: t=0.1-0.2时100% → 少量混入GRPO不影响; t=0.3-0.4严重退化 → SFT和GRPO电路**冲突区域**; t=0.9回升到81% → 接近纯GRPO。

### 3.3 兼容模型(SFT vs SFT→GRPO)

| 方法 | Accuracy |
|------|----------|
| SLERP(SFT, SFT→GRPO, t=0.5) | **100%** |

**关键发现**: SFT和SFT→GRPO模型完美合并! → 因为SFT→GRPO是从SFT出发训练的, 两者的task vector**高度兼容**。这与SFT vs GRPO(56.5%)形成鲜明对比。

## 四、DARE — Drop and Rescale

### 4.1 DARE SFT Task Vector

| drop_ratio | Accuracy | kept_pct | 分析 |
|------------|----------|----------|------|
| 50% | 85.5% | 49.8% | drop一半仍85.5% |
| 70% | 80% | 30% | drop七成仍80% |
| **90%** | **6.5%** | 10% | **灾难性退化!** |
| 95% | 4.5% | 5% | |
| 99% | 0% | 1% | |

### 4.2 DARE GRPO Task Vector

| drop_ratio | Accuracy | 分析 |
|------------|----------|------|
| 50% | 81.5% | GRPO delta更鲁棒 |
| 70% | 68% | |
| 90% | 60.5% | GRPO drop90%仍有60.5%! |
| 95% | 26.5% | |
| 99% | 4% | |

### 4.3 DARE-TIES (SFT + GRPO)

| drop_ratio | Accuracy |
|------------|----------|
| 90% | 38.5% |
| 95% | 34.5% |

## 五、关键发现

### 5.1 DARE在大模型有效但小模型失败!

```
大模型文献(Yu et al., 2023):
  DARE drop 90% → ≈保留原始性能 → "大部分delta不重要"

我们的76K小模型:
  DARE drop 90% → SFT: 6.5%, GRPO: 60.5%

→ 原因: 大模型有参数冗余 → 76K模型没有!
  大模型: delta中90%确实冗余 → drop后仍好
  小模型: delta中每个参数都重要 → drop=删除关键知识

→ GRPO的delta对drop更鲁棒(60.5% vs 6.5%):
  GRPO训练更"分散" → 删除部分仍有残存
  SFT训练更"紧凑" → 删除就是删除关键电路 → 灾难
```

### 5.2 Task Vector兼容性决定合并效果

```
模型合并的关键不是方法(Task Arithmetic/SLERP/DARE)
而是task vector的兼容性!

兼容性高(SFT vs SFT→GRPO):
  → SLERP t=0.5 = 100% → 任何合并方法都好
  → 因为两个模型的内部表示方向一致

兼容性低(SFT vs GRPO):
  → SLERP t=0.5 = 56.5% → 任何合并方法都差
  → 因为两个模型的内部表示方向冲突

→ 结论: 合并方法的选择远不如模型兼容性重要!
→ 实践: 先做SFT暖启动再做RL → 合并兼容性最高
```

### 5.3 Task Negation可以操控行为

```
SFT→GRPO - 0.5×GRPO = 75.5%

→ 减去GRPO influence → 从100%降到75.5%
→ 说明可以"去除"特定训练的影响
→ 但也说明不能完全去除(75.5%而非100%)

→ 原因: GRPO训练修改了SFT建立的电路
  → 减去GRPO delta → 恢复部分SFT电路
  → 但GRPO修改+SFT电路的交互 → 不能完全恢复
```

## 六、与大模型合并的联系

```
大模型(LLM)的合并为何更成功?

1. 参数冗余: 7B模型有7B参数 → delta中大量冗余 → DARE有效
   76K模型只有76K → delta几乎无冗余 → DARE失败

2. 任务多样性: LLM多任务 → task vector更独立 → 合并冲突少
   算术任务单一 → task vector高度耦合 → 合并冲突多

3. 表示稳定性: SFT→GRPO表示稳定 → 合并兼容
   GRPO-only不稳定 → 与SFT冲突 → 合并困难

→ 预测: LLM中SFT→RL模型的合并应该最兼容!
  → 与小模型实验一致: SFT→GRPO与SFT完美合并(100%)
```

## 七、实验代码

```bash
# 在GPU服务器运行
cd ~/rollout-infra/
source ~/anaconda3/bin/activate llm
python tools/model_merging_experiment.py --device cuda --output model_merging_results.json

# 结果保存在 results/model_merging_results.json
```

工具: `tools/model_merging_experiment.py`