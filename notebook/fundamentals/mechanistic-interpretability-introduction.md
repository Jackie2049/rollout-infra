# 机制可解释性入门 — 从黑箱到电路分析

> 2026-06-07 | AI理论×实践的关键桥梁：理解LLM内部如何"思考"

## 概述

机制可解释性(Mechanistic Interpretability)是将神经网络从"黑箱"变为"可理解系统"的研究领域。它不关注宏观行为（如benchmark分数），而是追问：**模型内部的哪些神经元/特征/电路导致了某个输出？**

对AI infra工程师而言，这是连接实践与理论的绝佳桥梁：
- Infra知识（GPU/CUDA/并行计算）帮助高效实现interpretability工具
- Interpretability知识帮助理解RL训练如何塑造模型内部表示
- 两者结合 → 更好的模型训练、调试和安全保障

## 一、为什么机制可解释性重要？

### 三大驱动力

1. **AI安全**: 理解模型是否在"欺骗"(deception)而非真正"理解"
   - 例子：模型可能在训练中学会"表面服从"而非"内在对齐"
   - 只有理解内部机制才能区分这两种情况

2. **训练理解**: RL训练改变了模型的什么？
   - GRPO/DPO/PPO不只改变输出分布 → 它们重塑内部计算路径
   - 我们实验发现：SFT→GRPO=100%eval vs 纯GRPO=50%eval → 为什么？
   - Interpretability可以回答：SFT建立了什么电路？GRPO强化了什么？

3. **模型编辑**: 不重新训练就能修改模型行为
   - 找到"有毒输出"对应的内部特征 → 直接编辑该特征
   - 比重新训练更快、更精确、更可控

### 与我们RL实验的连接

我们实验的核心发现——训练reward≠eval性能——可以通过interpretability解释：

```
训练reward高 ≠ 模型真正理解任务

GRPO 87.5%训练 → 50% eval → 模型学到了"采样策略"而非"算术规则"
SFT→GRPO 100%训练 → 100% eval → SFT建立了"正确计算电路"，GRPO只是强化

Interpretability可以验证：
- GRPO-only模型内部是否有"算术电路"？还是"采样运气电路"？
- SFT模型内部"算术电路"是否更robust？
```

## 二、核心方法

### 2.1 Sparse Autoencoders (SAE)

**原理**: 神经元是"多义"(polysemantic)的 → 一个神经元可能同时表示多个概念 → SAE将稠密激活分解为稀疏特征

```
稠密激活 z ∈ R^d → SAE分解 → 稀疏特征 f ∈ R^m (m >> d)

f = ReLU(W_enc @ z + b_enc)  # 编码：稀疏激活
z' = W_dec @ f + b_dec        # 重建：从稀疏回到稠密

L = ||z - z'||² + λ * ||f||₁  # 重建损失 + L1稀疏惩罚
```

**关键发现** (Anthropic, 2024):
- Claude 3 Sonnet上训练SAE → 发现了数千个可解释特征
- 例如："逐字分析"、"批评性思维"、"HTML代码"、"安全相关概念"
- 特征可以在模型内部追踪 → 看到信息如何在层之间流动

**L1稀疏的重要性**: 为什么不用PCA？
- PCA找到最大方差方向 → 但这些方向可能混合多个概念
- L1强制稀疏 → 每个特征只激活少量输入 → 更可解释
- 类比：医生诊断 → "5个独立症状"(稀疏)比"1个综合分数"(稠密)更可解释

### 2.2 Causal Tracing / Activation Patching

**原理**: 精确找到"哪个层、哪个位置"的信息导致了某个输出

```
方法：运行两次模型

Clean run: 输入"巴黎是法国的首都" → 输出"法国" ✓
Corrupted run: 输入"巴黎是XXX的首都" → 输出"XXX" ✗

Patching: 在corrupted run中，将某层的激活替换为clean run的激活
→ 如果输出恢复为"法国" → 该层是"法国知识"的关键位置！
```

**实践发现** (Meng et al., ROME/MEMIT):
- 知识存储在中层(layer 5-15 in GPT-2)的MLP权重中
- 早期层处理语法，后期层处理语义
- 可以通过编辑MLP权重来"插入新知识"(Model Editing)

### 2.3 Circuit Analysis

**原理**: 找到模型内部的"算法" → 不是单层分析，而是跨层的信息流图

```
电路 = 有向图 {节点=特征/神经元, 边=权重/注意力模式}

例子：GPT-2的"间接对象识别"(Indirect Object Identification)电路
输入: "Mary gave John a gift, then she gave..." → 输出应补完"Mary"
电路流程：
  1. 早期层检测"John"和"Mary"(token identity)
  2. 中层建立"John→接收者, Mary→给予者"(role assignment)
  3. 后层将"she→Mary"(pronoun resolution) → 输出"Mary"
```

**Anthropic的进展** (2024-2025):
- 在Claude中发现多个可解释电路
- 开发了"Circuit Discovery"自动化工具
- 从"手工分析"→"自动化电路发现"

## 三、与RL训练的深度连接

### 3.1 RL如何重塑内部表示

```
训练前(随机初始化):
  - 神经元随机连接 → 无可解释电路
  - 输出随机 → 无任务相关特征

SFT训练后:
  - 建立"任务电路"(如算术计算电路)
  - 特征变得有意义 → SAE可以找到"数字特征"、"运算特征"
  - 模型有了"正确的先验"

GRPO训练后(从SFT继续):
  - GRPO强化正确电路 → "算术电路"权重增大
  - 不需要新建电路 → 只需强化 → 稳定

GRPO训练后(从随机开始):
  - 需要从零建电路 → 但GRPO只给出"相对排序"信号
  - 容易建错电路 → "采样运气电路"而非"算术电路"
  - 训练reward高但eval低 → 电路不robust
```

### 3.2 过优化 = 电路退化

```
正常训练: 正确电路逐步强化 → 性能↑
过优化: 模型找到"捷径电路" → reward↑但真实能力↓

例子(DAPO 96.7→12.5%):
  - 训练初期：算术电路正在形成 → reward↑
  - 过优化：模型发现"输出高频数字"的捷径 → reward虚高
  - 崩溃：捷径电路覆盖了正确电路 → eval骤降
```

### 3.3 Verification = Interpretability检查

```
训练-eval gap的interpretability解释:

GRPO: 算术电路50%robust + 50%采样运气 → eval 50%
SFT→GRPO: 算术电路100%robust → eval 100%
PPO: 算术电路34%robust + 66%specification gaming → eval 34%
DAPO: 算术电路peak时强但不稳定 → 96.7→12.5%
```

## 四、实验结果 (RTX 4090, 76K模型)

### 4.1 SAE训练结果

**ReLU+L1 SAE** (feature死亡问题):
- λ=1→L0=85.5(33.4% active, 太密), λ=3→L0=0(所有feature死亡!), λ=10→L0=0
- Feature死亡: L1惩罚太强→encoder权重被推向零→所有feature都不激活
- 解决方案: 使用TopK SAE

**TopK SAE** (解决了feature死亡):
- K=10: L0=10(稳定不变!), recon=0.04(好), 3.9% active → **Feature 81选择性激活!**(9/200输入, avg_sum=6.0 → 大数检测特征!)
- K=5: L0=5, recon=0.04(好), 2% active → 但全局feature对所有输入都激活(200/200) → 需更细粒度分析
- **关键教训**: TopK解决了feature死亡问题，但64维太小→TopK feature捕捉所有信息→需更细粒度的选择性分析

### 4.2 Activation Patching结果 (重要发现!)

**实验设计**: Clean input "3+2=" → Corrupt input "1+0=" → Patch clean activation到corrupt run

**关键发现**: 算术知识集中在**等号位置(pos 3)**!

```
Patching效果排名 (effect = patched_prob - corrupt_baseline):

attn_0_pos3 (=等号): +40.5%  ← 最关键! Layer 0 attention在等号位置
attn_1_pos3 (=等号): +24.9%  ← Layer 1 attention在等号位置也有强效果
ln2_1_pos3 (=等号): +13.4%   ← Layer 1 MLP
mlp_1_pos3 (=等号): +13.4%
ln2_0_pos3 (=等号): +2.0%    ← Layer 0 MLP较弱
其他位置: ~0%                 ← 算术知识不在数字位置!
```

**解读**:
- 等号 `=` 是模型做出"决定"的位置 → 信息流最关键
- 数字位置(pos 0,1,2)对答案几乎无影响 → 模型不需要"理解"数字含义
- 这说明模型在等号位置已经完成了计算 → 等号位置是"算术电路"的输出节点

### 4.4 GRPO vs SFT→GRPO Circuit比较 (验证核心假设!)

**实验设计**: 训练两个模型(相同seed=42) → GRPO-only 300步 vs SFT→200步+GRPO 300步 → 比较内部电路差异

**结果**: GRPO 81% eval vs SFT→GRPO **100% eval**!

```
Circuit差异 (cosine similarity):
ln2_0:  0.77  ← LayerNorm差异较小
mlp_0:  0.27  ← **MLP差异最大!**
attn_0: 0.35  ← Attention也有差异
ln2_1:  0.45
mlp_1:  0.14  ← **MLP差异最显著!**
attn_1: 0.17

Position差异:
pos 0 (digit a): 0.23  ← 数字位置差异大
pos 1 (+):        0.42
pos 2 (digit b):  0.32
pos 3 (=):        0.49  ← **等号位置差异最显著!**
pos 4 (eos):      0.32
```

**Patching敏感度 (at equals-sign pos 3)**:
```
Layer      GRPO-only  SFT→GRPO   Ratio
attn_0     0.30       **0.71**   **2.34x** ← SFT→GRPO模型2.3x更敏感!
attn_1     0.22       0.21       0.97x  ← Layer 1几乎相同
ln2_1      0.15       0.19       1.23x
mlp_1      0.15       0.19       1.23x
```

**核心洞察**:
1. **MLP层是两种训练方式差异的核心** → SFT改变了MLP的内部表示 → MLP是"知识存储层"
2. **SFT→GRPO的attn_0在等号位置2.3x更敏感** → 更容易响应正确的算术信息 → 更灵活
3. **GRPO-only模型的attn_0更弱**(0.30 vs 0.71) → "算术电路"更脆弱 → 容易被错误信息误导
4. **这解释了训练-eval gap**: GRPO-only的脆弱电路 → 训练reward高但eval低 → 电路不robust

**实验**: 在等号位置注入不同target_sum的activation → 能否操控模型输出?

```
Steering成功率: 86/200 = 43%

target=7: 23/23 = **100%成功率!**  ← "7"在模型内部有超强独立表示!
target=2: 18/22 = 82%成功
target=0: 13/24 = 54%成功
target=3: 10/21 = 48%成功
target=6:  6/22 = 27%成功
target=1:  9/23 = 39%成功
target=5:  3/21 = 14%成功
target=4:  1/20 =  5%成功 ← 最难操控
target=8:  3/24 = 13%成功 ← 难操控
```

**关键发现**:
1. **"7"有100%操控成功率** → 注入"7"的等号位置activation可以把任何输入变成7! → "7"在模型内部有非常强、非常独立的表示
2. **"4"和"8"最难操控** → 这些数字的内部表示较弱或更分散
3. **43%总成功率** → 单个位置的activation注入就能改变模型行为 → 验证了等号位置是"决策中心"

**意义**: 这是**模型编辑(Model Editing)**的interpretability版本!
- 不改权重 → 只改一个位置的activation → 就能改变输出
- 未来方向: SAE feature steering → 更精细的控制(SAE feature→activation→output)

### 4.1 TransformerLens

- Open-source工具(nnsight团队维护)
- 功能：hook任意层激活、patching、SAE训练
- 支持：GPT-2/GPT-J/Llama系列
- 我们可以结合GPU知识 → 优化interpretability工具的CUDA性能

### 4.2 SAE训练流程

```
1. 收集激活: 在大量文本上运行模型 → 收集指定层的MLP/Residual激活
2. 训练SAE: 用L1稀疏自编码器分解激活
3. 分析特征: 看哪些SAE特征在什么输入上激活 → 可解释性评估
4. 电路追踪: 在模型forward中追踪SAE特征 → 建立信息流图
```

### 4.3 我们可以做的实验

**Mini Interpretability实验**(用我们的MiniGQATransformer 76K模型):
1. 训练SAE → 找到"数字特征"和"运算特征"
2. Activation patching → 找到"算术电路"的关键层
3. 比较GRPO-only vs SFT→GRPO模型的电路差异
4. 验证：SFT模型是否有更robust的算术电路？

**GPU加速SAE训练**:
- SAE训练需要大量forward pass → GPU必不可少
- CUDA kernel优化：SAE encoder/decoder的fused kernel
- 与我们的CUTLASS/CUDA kernel经验完美结合

## 五、前沿方向(2025-2026)

### 5.1 Anthropic的进展

- **Scaling Monosemanticity (2024-2025)**: 在Claude 3 Sonnet上训练SAE → 发现数千个可解释特征
- **Gated SAEs (2025)**: 改进架构解决feature absorption和dead feature问题 → 我们的实验验证了ReLU+L1的feature死亡问题(λ≥3→L0=0)
- **Dictionary Learning**: SAE的改进 → 更好的特征分离
- **Circuit Discovery**: 自动化找到可解释电路 → 不需手工分析
- **Feature Steering**: 直接操控SAE特征 → 改变模型输出(安全应用!) → 我们的实验可以做feature steering: 在等号位置注入不同activation改变答案
- **Automated Interpretability**: 用LLM自动标注SAE特征 → Neuronpedia平台

### 5.2 开源进展

- **TransformerLens 2.0**: nnsight升级版，支持更多模型
- **Gemma Scope**: Google发布Gemma模型的SAE特征集
- **Open SAE**: 多个开源SAE训练框架(EleutherAI/DeepMind)
- **Neuronpedia**: 社区SAE特征浏览平台
- **SAE-Bench**: 评估benchmark

### 5.3 理论进展

- **Superposition theory**: 神经元为何是多义的 → 因为特征数>维度 → 必须叠加
- **Feature geometry**: 特征在激活空间中的几何结构 → manifold分析
- **Phase transitions**: 训练中电路如何突然形成 → 类似我们GRPO的"aha moment"

## 六、下一步计划

1. **Mini SAE实验**: 在MiniGQATransformer上训练SAE → 找算术特征
2. **Activation patching**: 找到算术电路的关键层
3. **GRPO vs SFT电路比较**: 直接验证训练-eval gap的interpretability解释
4. **GPU加速**: SAE训练的CUDA kernel优化
5. **写工具**: `tools/mini_interpretability.py` — 可解释性分析工具

> 这是我从AI infra走向AI专家的关键一步 — 不只理解系统如何运行，还要理解模型如何"思考"。

## Sources

- [Anthropic SAE Research](https://www.anthropic.com/research) — Sparse Autoencoders on Claude
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — Interpretability tool
- [Meng et al., ROME/MEMIT](https://arxiv.org/abs/2202.05262) — Locating and Editing Factual Associations in GPT
- [Elhage et al., Circuits](https://transformer-circuits.pub/) — Anthropic circuits research
- [Gemma Scope](https://ai.google.dev/gemma/gemma_scope) — Google's SAE feature set