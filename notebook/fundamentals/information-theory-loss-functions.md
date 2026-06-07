# 信息论与损失函数 — 从熵到交叉熵到LLM训练的一切

> 2026-06-07 | 理解"为什么CE loss是标准"和"为什么KL divergence是蒸馏核心"

## 概述

信息论是所有ML损失函数的理论基础。本文从香农熵出发，推导出交叉熵→KL divergence→困惑度，连接到LLM训练的每一个loss。

## 一、香农熵 — 信息的度量

### 1.1 熵的定义

```
H(X) = -Σ p(x) × log₂ p(x)

含义: "描述随机变量X需要的平均比特数"

例子:
  抛硬币(公平): H = -0.5×log(0.5) - 0.5×log(0.5) = 1 bit
  → 需要1比特来描述结果

  抛硬币(偏向一面): p(正面)=0.9
  H = -0.9×log(0.9) - 0.1×log(0.1) = 0.47 bit
  → 只需0.47比特 → 结果更确定 → 信息更少

  确定事件: p=1 → H=0 → 无信息(已知答案)
  均匀分布: p=1/N → H=log₂N → 最大信息(完全不确定)
```

### 1.2 熵与softmax

```
softmax输出的熵:

H(softmax(z)) = -Σ p_i × log p_i

→ 高entropy = 均匀分布 = 模型不确定 = "还没学到" → 训练初期
→ 低entropy = 尖锐分布 = 模型确定 = "已经学到了" → 训练后期

→ 训练过程 = entropy从高到低的过程 = 从不确定到确定

  例外: 过优化 → entropy过低 → 模型过度确定 → 泛化差 → 需要entropy bonus
```

## 二、交叉熵 — LLM训练的核心损失

### 2.1 交叉熵定义

```
H(P, Q) = -Σ p(x) × log q(x)

含义: "用分布Q编码分布P需要的平均比特数"

→ 如果Q=P → H(P,P)=H(P) → 最优编码
→ 如果Q≠P → H(P,Q)>H(P) → 需要更多比特 → Q不是P的最佳编码

→ 交叉熵衡量Q与P的差异 → 差异越大 → 交叉熵越高
```

### 2.2 为什么LLM用交叉熵而非MSE?

```
MSE: L = Σ (p_i - q_i)²  ← 回归任务
CE:  L = -Σ p_i × log q_i ← 分类/生成任务

LLM本质是"概率分布预测":
  输入: "猫坐在"
  输出: p(垫子上)=0.8, p(沙发上)=0.15, p(冰箱里)=0.001

  → 不是预测一个值(regression) → 而是预测一个分布(classification)
  → CE衡量"预测分布与真实分布的差异" → 是最自然的选择

MSE问题:
  1. 对小概率事件不敏感: (0.001-0)²=0.000001 → 几乎无梯度 → 不学"垫子上"的竞争对手
  2. 对高概率事件过于敏感: (0.8-0)²=0.64 → 但CE: -log(0.8)=0.097 → 更合理的惩罚
  3. 不满足概率约束: MSE不保证Σq_i=1 → CE与softmax配合保证概率分布

CE的优势:
  1. 对小概率事件更敏感: -log(0.001)=6.9 → 大惩罚 → 促使模型不忽视低概率但重要的token
  2. 与softmax天然配合: CE + softmax → 梯度简单(p_i - q_i → 线性!)
  3. 信息论最优: 最小化CE = 最大化似然 = 用最少比特编码数据
```

### 2.3 CE + Softmax梯度 — 极简公式

```
L = -log softmax(z_correct)  ← 单个正确token的CE

∂L/∂z_i = p_i - δ_i_correct

  如果i是正确token: ∂L/∂z_i = p_i - 1 → 推高正确logit
  如果i是错误token: ∂L/∂z_i = p_i → 降低错误logit

→ 梯度就是"预测概率 - 真实分布" → 极简! → 无需计算softmax雅可比 → 直接用p-δ

→ 这是CE+softmax被广泛使用的原因: 梯度计算简单+数值稳定+信息论最优
```

### 2.4 困惑度(Perplexity)

```
PPL = exp(H(P, Q_model)) = exp(-1/N Σ log q_model(x_i))

含义: "模型对数据的平均不确定性" → PPL越低 → 模型越好

  PPL=1 → 完全确定(每步都正确) → 信息量0
  PPL=10 → 每步有10种等可能选择 → 较不确定
  PPL=100 → 每步有100种等可能选择 → 非常不确定

  → 我们的实验: SFT后PPL降低 → MiniGPT loss 5.03→1.47 → PPL从~150→~4.3

  → PPL与loss的关系: PPL = exp(loss) → loss=log(PPL)
  → PPL更直观(描述"多少种选择") → loss更方便(优化目标)
```

## 三、KL Divergence — 蒸馏的核心

### 3.1 KL Divergence定义

```
KL(P || Q) = Σ p(x) × log(p(x)/q(x))
            = Σ p(x) × log p(x) - Σ p(x) × log q(x)
            = H(P) - H(P, Q) ← 熵减交叉熵!

→ KL(P||Q) = H(P,Q) - H(P) → "用Q编码P比用P编码自己多需要的比特数"

→ KL ≥ 0 ( Gibbs不等式: KL(P||Q) ≥ 0, 当P=Q时=0)
→ KL ≠ 对称: KL(P||Q) ≠ KL(Q||P) → 不是"距离"而是"差异"
```

### 3.2 Forward vs Reverse KL

```
Forward KL(P||Q): Σ p(x) × log(p(x)/q(x))
  → P是teacher, Q是student
  → student必须覆盖teacher的所有高概率区域 → mode-covering
  → student不会遗漏teacher的任何重要输出 → 但可能"平均化"

Reverse KL(Q||P): Σ q(x) × log(q(x)/p(x))
  → Q是student, P是teacher
  → student聚焦teacher最高概率区域 → mode-seeking
  → student更精确但可能遗漏 → MiniLLM推荐(for LLM蒸馏)

直觉:
  Forward KL: "我(teacher)知道什么, 你(student)必须都知道" → 覆盖全面
  Reverse KL: "你(student)只说我(teacher)最确定的东西" → 精确但有遗漏风险
```

### 3.3 KL与训练-eval gap

```
RL训练中的KL约束: max E[r] - β KL(π || π_ref)

为什么需要KL约束?
  → 如果只优化reward → π会偏离π_ref → 过优化 → reward虚高但eval低

  → KL约束确保π不偏离太远 → 保持π_ref的"安全知识"

  → β(KL权重)是关键:
     β太小 → π偏离太多 → 过优化 → 训练reward高但eval低(GRPO 87.5%→50%)
     β太大 → π接近π_ref → reward低 → 训练不充分

  → SFT→GRPO: SFT已经建立正确π → KL偏离小 → 不需要强KL约束 → eval高!
```

## 四、信息论视角下的训练

### 4.1 训练 = 减少交叉熵

```
训练目标: min CE(P_data || P_model)
  → P_data: 数据的真实分布
  → P_model: 模型的预测分布
  → 训练使P_model接近P_data → CE减小 → PPL减小

训练前: P_model接近uniform → H(P_data, P_model) ≈ H(P_data) + log|Vocab| → 高PPL
训练后: P_model接近P_data → H(P_data, P_model) ≈ H(P_data) → 低PPL

→ 训练过程 = 从"uniform猜测"到"精确预测" → 信息量减少 → 不确定性降低
```

### 4.2 过优化 = 信息丢失

```
正常训练: CE(P_data||P_model)↓ → P_model学到P_data的结构 → 泛化好

过优化: CE(P_train||P_model)↓ 但 CE(P_all||P_model)↑
  → P_model只记住训练数据的模式 → 丢失了更广泛的模式 → 泛化差

信息论解释:
  过优化 = 训练数据的信息 → 完全编码到模型 → 但"暗知识"(泛化模式)被训练数据的noise淹没
  → 需要regularization(L2/dropout/early stopping) → 防止模型编码噪声而非信号

→ 我们的RL实验:
  DAPO过优化: 96.7%训练reward → 12.5% eval → reward的noise被编码 → 丢失泛化模式
  SFT→GRPO: 100%训练 → 93% eval → SFT先编码"正确模式" → GRPO只强化 → noise影响小
```

### 4.3 Distillation = 信息压缩

```
蒸馏目标: min KL(P_teacher || P_student)
  → P_student编码P_teacher的信息 → 但用更少的参数 → 信息压缩

  → 如果P_teacher有H(P_teacher)比特的信息
  → P_student容量有限 → 只能编码部分信息 → 哪部分?

  → Hinton: temperature↑ → 小概率事件更可见 → P_student编码更多"暗知识"
  → 但compression ratio太大 → P_student容量不够 → 无法编码足够信息 → 蒸馏失败

  → 我们实验: 10.9x压缩 → student容量不足 → PPL反而更差
  → 合理压缩(≤3x): P_student编码大部分信息 → 蒸馏成功
```

### 4.4 互信息与representation learning

```
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

含义: "知道Y后, X的不确定性减少了多少"

→ 互信息衡量两个变量之间的信息依赖关系
→ 高互信息 → X和Y高度相关 → 知道Y能预测X
→ 低互信息 → X和Y独立 → 知道Y不能预测X

→ InfoNCE loss (对比学习):
  L = -log(exp(sim(x,y+))/Σ exp(sim(x,y_i)))
  → 最大化正样本的互信息 → 最小化负样本的互信息

→ Transformer的attention可以理解为最大化互信息:
  query与key的相似度 → 选择与query最相关的value → 最大化I(query; value)
```

## 五、损失函数对比总结

| Loss | 公式 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| **Cross-Entropy** | -Σ p log q | LLM训练(SFT/CE) | 信息论最优+梯度简单 | 对长尾分布不敏感 |
| **KL Divergence** | Σ p log(p/q) | 蒸馏/RL约束 | 衡量分布差异 | 不对称+数值不稳定 |
| **MSE** | Σ(p-q)² | 回归/feature对齐 | 简单+对称 | 对小误差不敏感+不满足概率约束 |
| **SmoothL1** | f(p-q) | Feature蒸馏 | 对大误差鲁棒 | 不满足概率约束 |
| **Focal Loss** | -Σ(1-p)^γ p log q | 长尾分类 | 关注难分类样本 | 需调γ |
| **InfoNCE** | -log(exp(+)/Σexp) | 对比学习 | 无需负样本标签 | 需大batch |

## 六、与前沿训练方法的连接

### 6.1 DPO Loss的信息论解释

```
DPO loss: -log σ(β log(π(y_w)/π_ref(y_w)) - β log(π(y_l)/π_ref(y_l)))

→ σ内部: β × (隐式reward(y_w) - 隐式reward(y_l))
→ 隐式reward = β log(π/π_ref) = β × (KL偏离)

→ DPO训练偏好对的顺序 → 确保π对y_w的KL偏离 > π对y_l的KL偏离
→ 信息论: π在y_w上编码更多信息(偏离ref) → π在y_l上编码更少信息(接近ref)

→ 与RL的KL约束一致: max E[r] - β KL → DPO也确保偏离ref的程度与reward成正比
```

### 6.2 GRPO组归一化的信息论解释

```
GRPO advantage: A = (r - μ) / σ

信息论视角:
  μ = 组内平均reward → "baseline信息量"
  σ = 组内reward方差 → "信号强度"
  r-μ = 单个response偏离baseline的信息量
  (r-μ)/σ = 标准化的信息量 → 单位方差 → 可比较

  → σ归一化 = 信息标准化 → 使不同组的信息量可比较 → 防止高方差组主导训练
  → 无σ归一化(RLOO): 不同组信号强度不同 → 高方差组梯度大 → 不稳定
```

### 6.3 Perplexity与模型质量的关系

```
PPL vs eval accuracy的关系(我们的实验):

| Model | PPL | Eval Acc | 说明 |
|-------|-----|----------|------|
| Random | ~5.21(BPC) | ~0% | 完全随机 |
| SFT 200步 | ~1.47 | 50% | 学了格式但不精确 |
| SFT→GRPO | ~0.02 | **93%** | 精确掌握了算术 |

→ PPL和accuracy不是线性关系!
→ PPL从5→1.5 → accuracy从0→50%(巨大进步)
→ PPL从1.5→0.02 → accuracy从50→93%(进一步精确)
→ 但PPL接近0 → 模型完全确定 → 可能过优化!

→ 良好的模型: PPL低但不为0 → 有适度的不确定性 → 可以探索
```

## 七、实战应用

### 7.1 选择loss的决策树

```
任务类型?
├─ 分类/生成 → Cross-Entropy (信息论最优)
├─ 回归 → MSE (预测连续值)
├─ 分布对齐 → KL Divergence (衡量分布差异)
├─ Feature对齐 → SmoothL1 (鲁棒+稳定)
├─ 对比学习 → InfoNCE (最大化互信息)
└─ 长尾分布 → Focal Loss (关注难样本)

训练阶段?
├─ SFT → CE loss (学习基本模式)
├─ RL → reward-based + KL constraint (优化对齐)
├─ 蒸馏 → KL(T=4-8) + CE (迁移知识)
└─ DPO → preference-based (偏好对齐)
```

### 7.2 监控指标的信息论含义

| 指标 | 公式 | 信息论含义 |
|------|------|------------|
| Loss | -Σ p log q | 交叉熵 → 用模型编码数据的代价 |
| PPL | exp(loss) | 困惑度 → 模型平均不确定度 |
| Entropy | -Σ p log p | 模型输出的信息量 |
| KL | Σ p log(p/q) | 模型偏离参考的程度 |
| Accuracy | Σ(预测==真实) | 硬标签匹配度(不反映分布质量) |

**关键**: Accuracy≠质量 → 模型可能100% accuracy但entropy极高(说明靠运气而非确定性)

## 八、Sources

- Shannon, 1948 — A Mathematical Theory of Communication (信息论创始论文)
- Hinton et al., 2015 — Distilling the Knowledge in a Neural Network
- Goodfellow et al., 2016 — Deep Learning (Chapter 3: Information Theory)