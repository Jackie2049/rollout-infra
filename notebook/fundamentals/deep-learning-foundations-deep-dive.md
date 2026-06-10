# Deep Learning Foundations Deep Dive

> 2026-06-10 | 深度学习基础=AI的数学根基! 从感知机到Transformer, 从反向传播到损失函数, 从正则化到优化landscape, 基础=理解一切的起点!
> 关联: probability-bayesian-foundations.md, optimization-algorithms-deep-dive.md, generalization-theory.md, attention-sink-streamingllm-deep-dive.md

## 0. 核心定律: 深度学习 = 层次化特征提取 + 梯度优化

单层=线性 → 多层=非线性组合 → 深度=层次化特征 → 梯度=优化路径!

关键数据:
- 1层MLP → XOR解决 → 非线性必需! → 深度→层次化→强!
- Transformer → self-attention → 全连接 → 位置无关 → 通用!
- ResNet残差 → 梯度直传 → 100+层 → 无退化! → Transformer也用!
- Dropout → Bernoulli ≈ 贝叶斯 → 正则化 → 泛化↑!

## 1. Neural Network Basics — 从感知机到深度网络

```
神经网络演进:

1. Perceptron (感知机) — 1957:
   → input × weight → sum → threshold → output → 0/1!
   → → → 数学: y = sign(w^T x + b) → 线性 → 只能线性分类!
   → → → → → 限制: XOR不可解! → 线性分类器→非线性问题→失败!
   → → → → → → → Minsky-Papert 1969: 感知机限制 → AI寒冬!

2. MLP (多层感知机) — 解决非线性:
   → 多层 + 非线性激活 → XOR可解 → 万能!
   → → → 数学: y = f(W2 × f(W1 × x + b1) + b2) → 非线性组合!
   → → → → → XOR: x1→hidden→x2→hidden→AND+NOT→output→XOR解!
   → → → → → → → Universal Approximation: 1隐藏层→任意连续函数→足够宽!

3. Activation Functions — 非线性关键:
   → sigmoid: 1/(1+e^-x) → (0,1) → 梯度消失 → 不推荐!
   → → → tanh: (e^x-e^-x)/(e^x+e^-x) → (-1,1) → 梯度消失 → 不推荐!
   → → → → → ReLU: max(0,x) → 简单 → 不消失 → 推荐! → 但: 死亡ReLU(x<0→0)!
   → → → → → → → GELU: x × Φ(x) → 平滑 → LLaMA用! → 推荐!
   → → → → → → → → → SwiGLU: x × sigmoid(βx) × gate → LLaMA核心 → 推荐!

4. 现代架构:
   → CNN: 局部+池化 → 图像 → ResNet → 残差连接!
   → → → RNN: 序列+隐藏 → 文本 → LSTM → 长依赖 → 但慢!
   → → → → → Transformer: self-attention → 全连接 → 位置无关 → 通用!
   → → → → → → → → → Transformer = 当前AI一切的基础 → LLM/VLM/Diffusion/Agent!

关键发现:
  → 非线性=神经网络必需 → 无非线性=多层=线性=无意义!
  → → → 深度=层次化特征 → 第1层→边缘 → 第2层→纹理 → 第3层→对象!
  → → → → → ResNet残差=梯度高速公路 → 深度不退化 → Transformer也用Pre-Norm!
```

## 2. Backpropagation Math — 反向传播数学

```
反向传播=链式法则 → 梯度计算 → 参数更新 → 训练核心!

Forward pass:
  → x → W1 → h1 → W2 → h2 → W3 → y → loss!
  → → → 每层: h_l = f(W_l × h_{l-1} + b_l) → 激活函数!

Backward pass (链式法则):
  → dL/dW3 = dL/dy × dy/dh2 × dh2/dW3 → 最后一层!
  → → → dL/dW2 = dL/dy × dy/dh2 × dh2/dh1 × dh1/dW2 → 传播!
  → → → → → dL/dW1 = dL/dy × ... × dh1/dW1 → 第一层 → 最远!

梯度消失问题:
  → sigmoid: dy/dx = y(1-y) → 最大0.25 → 每层×0.25 → N层→0.25^N → 消失!
  → → → 10层: 0.25^10 ≈ 10^-6 → 几乎0 → 第一层→无梯度→不学习!
  → → → → → ReLU: dy/dx = 1 (x>0) → 不缩小 → 梯度不消失 → 解决!
  → → → → → → → 但: ReLU x<0 → dy/dx=0 → 死亡 → 永不更新 → 问题!

梯度爆炸问题:
  → W初始化太大 → ×大W → ×大W → ×大W → 梯度→∞ → NaN!
  → → → 解决: gradient clipping → max_norm → 梯度裁剪 → 防爆炸!
  → → → → → 初始化: Xavier → W ~ N(0, 2/(fan_in+fan_out)) → 合适!
  → → → → → → → Kaiming(He) → W ~ N(0, 2/fan_in) → ReLU专用 → 推荐!

Transformer反向传播:
  → Attention: dL/dQ, dL/dK, dL/dV → softmax BP → 复杂!
  → → → softmax BP = A × (dA - ΣdA·A) → 矩阵形式 → 见attention math笔记!
  → → → → → dL/dW = dL/dh × dh/dW → 每层独立 → Transformer=每层梯度!

Layer Norm vs Batch Norm:
  → BatchNorm: μ,σ = batch统计 → 训练不同→推理不同→需要running stats!
  → → → LayerNorm: μ,σ = 单样本统计 → 训练=推理 → 简单 → Transformer用!
  → → → → → RMSNorm: σ only → 不算μ → 更简单 → LLaMA用 → 推荐!
  → → → → → → → RMSNorm BP: dL/dx = dL/dy × (1/σ) × (I - y×y^T/N) → 简单!
```

## 3. Loss Functions — 损失函数

```
损失函数=训练目标 → 不同任务→不同loss → 核心!

1. Cross-Entropy (CE) — 分类/LLM训练:
   → CE = -Σ y_true × log(y_pred) → 真实分布→预测分布→差距!
   → → → LLM: y_true=1(token) → CE = -log(p_token) → 最大化正确token概率!
   → → → → → SFT: CE loss → 最小化 → 正确token→高概率 → 学习!
   → → → → → → → 数学: CE(P,Q) = -Σ P(i) × log Q(i) → 用Q编码P的代价!
   → → → → → → → → → MLE = CE loss → SFT训练 = 最大似然估计 → 见概率论笔记!

2. MSE (Mean Squared Error) — 回归:
   → MSE = Σ(y_true - y_pred)^2 / N → 预测→真实→差距!
   → → → 适合: 数值预测 → 温度/价格/坐标 → 连续值!
   → → → → → 但: LLM→不用MSE → 分类→CE → 文本→token→CE!

3. KL Divergence — 分布距离:
   → KL(P||Q) = Σ P(i) × log(P(i)/Q(i)) → P到Q的散度!
   → → → 非对称: KL(P||Q) ≠ KL(Q||P) → forward/reverse!
   → → → → → RL: KL(π||π_ref) → 新策略→旧策略→偏离 → penalty!
   → → → → → → → GRPO: KL penalty → β × KL → 防过度偏离 → 见RL笔记!

4. Contrastive Loss — 对比学习:
   → 正样本→近 → 负样本→远 → InfoNCE → CLIP用!
   → → → InfoNCE: -log(sim_pos / Σsim) → 正→高→负→低!
   → → → → → VLM: CLIP → 图像+文本 → 对齐 → InfoNCE!

5. Focal Loss — 不平衡分类:
   → CE × (1-p)^γ → 难样本→权重↑ → 易样本→权重↓ → 平衡!
   → → → γ=2 → (1-p)^2 → p=0.9→0.01 → p=0.1→0.81 → 81x权重差!

RTX 4090损失函数:
  → SFT: CE → 最大似然 → 简单 → 不需要GPU特性!
  → → → RL: CE + KL → GRPO → 需要GPU → 但LoRA→单GPU可行!
```

## 4. Regularization — 正则化

```
正则化=防过拟合 → 泛化↑ → 训练必需!

1. Dropout — Bernoulli近似贝叶斯:
   → 训练: 每神经元→随机→0(p=0.5) → 不依赖单神经元 → 集体!
   → → → 推理: 全部激活 → 输出×(1-p) → 缩放 → 等价!
   → → → → → 数学: Dropout ≈ 贝叶斯模型平均 → Bernoulli采样 → 多模型→平均!
   → → → → → → → 效果: 泛化↑ → 过拟合↓ → 简单 → 推荐!
   → → → → → → → → → 但: LLM训练→不太用 → 数据够→自然泛化→SFT→GRPO→0 gap!

2. L2 Regularization (Weight Decay):
   → loss + λ × Σw^2 → 参数→小 → 不极端 → 平滑!
   → → → AdamW: decoupled wd → wd不被√v缩放 → 公平 → 推荐!
   → → → → → 见优化算法笔记 → AdamW > Adam+L2 → 关键差异!

3. Batch Normalization → 间接正则化:
   → μ,σ = batch统计 → 每batch→不同→噪声→间接正则化!
   → → → 但: Transformer→LayerNorm → 不用BatchNorm → 序列!

4. Early Stopping:
   → 验证loss↑ → 停止 → 不过拟合 → 简单!
   → → → 但: RL训练→需要跑完 → 不早停 → 见GRPO笔记!

5. Data Augmentation → 最强正则化:
   → 更多数据 → NLP→不常用 → 图像→常用 → 但: LLM→数据质量>数量!

6. Gradient Clipping → 防爆炸:
   → max_norm → 梯度→裁剪 → 防NaN → 训练稳定!
   → → → PPO: clip → ε范围 → 策略更新→不太大 → RL稳定!

RTX 4090正则化推荐:
  → LLM SFT: CE+AdamW wd → 简单 → 数据够→自然泛化!
  → → → GRPO: KL penalty → RL正则化 → 防reward hacking!
  → → → → → LoRA: 冻结base → 天然正则化 → 不需要额外 → 简单!
  → → → → → → → Dropout: 大模型→不需要 → 数据够 → 小模型→需要!
```

## 5. Optimization Landscape — 优化地形

```
优化地形=loss function空间 → 梯度→寻找最低点!

地形特征:
  → Local minima: 局部最低 → 不是全局 → 但: 高维→局部≈全局 → 鞍点更多!
  → → → Saddle points: 一方向↑另一方向↓ → 高维→主要障碍 → 不是局部最小!
  → → → → → Plateaus: 平坦 → 梯度≈0 → 不前进 → Adam→自适应→帮助!

高维空间特性:
  → 高维(7B=7e9参数) → 局部最小→不严重 → 大部分→鞍点 → 可逃离!
  → → → 原因: 7e9维 → 同时所有维→最小 → 概率≈0 → 实际=鞍点!
  → → → → → Adam: 自适应lr → 鞍点→不同lr → 方向→不同→逃离!

Loss landscape实测:
  → SFT→center loss ≈ 0.001 → 深峡谷 → 正确解 → 泛化好!
  → → → GRPO→center loss ≈ 5.06 → 局部高点 → 不精确 → 泛化差!
  → → → → → SFT盆地: center极低 → 周围极高 → 深但窄 → 精确解!
  → → → → → → → GRPO高地: center偏高 → 附近有更好解 → flatness=-0.215 → 不在最优!

  → → → → → → → → → 关键发现: 泛化好≠平坦盆地 → 而是"解本身正确"(CE≈0→预测精确)!
  → → → → → → → → → → → reward hacking = reward高但CE也高 → 模型没学到正确算法!

Sharp vs Flat Minima:
  → Sharp: 周围↑ → 参数变化→loss↑快 → 不稳定 → 泛化差!
  → → → Flat: 周围平坦 → 参数变化→loss不变 → 稳定 → 泛化好!
  → → → → → 但: SFT→深峡谷→sharp → 但泛化好 → 理论矛盾!
  → → → → → → → 解释: sharp但精确 → 解正确 → 不需要"平坦" → CE≈0→已最优!

RTX 4090优化地形:
  → LoRA: 0.5MB参数 → 低维 → 地形简单 → AdamW→快速找到!
  → → → 全参数: 7e9维 → 高维 → 鞍点多 → AdamW+WD → 缓慢!
  → → → → → SFT暖启动 → 已在好位置 → RL微调 → 不需要探索 → 快速!
```

## 6. Universal Approximation Theorem — 万能逼近定理

```
万能逼近定理:
  → 1隐藏层MLP(足够宽) → 任意连续函数 → 逼近 → 万能!
  → → → 但: 不说→多少神经元 → 可能需要2^N → 实际不可!
  → → → → → 深度网络 → 每层→少量 → 层次化 → 更高效 → 实际可行!

定理限制:
  → 1. 不说→多少神经元 → 可能需要指数级 → 不实际!
  → → → 2. 不保证→可学习 → 存在解→但SGD→不一定找到!
  → → → → → 3. 不保证→泛化 → 训练误差≈0→但测试→可能差!

深度 vs 宽度:
  → 宽度: 2^N神经元 → 任意函数 → 但: 参数→指数 → 不可!
  → → → 深度: N层×O(1)神经元 → 层次化 → 参数→线性 → 可行!
  → → → → → 关键: 深度=层次化特征提取 → 第1层→边缘→第N层→语义!
  → → → → → → → 深度网络→指数级表达能力 → vs宽度→需要指数级参数!

现代架构层次化:
  → Transformer: 层→token→attention→特征→层→层→层→全局!
  → → → ResNet: 层→edge→texture→object→face → 层次化→视觉!
  → → → → → MoE: 不同expert→不同特征→稀疏→层次化→混合!
  → → → → → → → VLM: ViT→visual→projector→LLM→language → 多模态层次!
```

## 7. Modern Architectures — 现代架构

```
现代AI架构3代:

1. ResNet (2015) — 深度革命:
   → 残差连接: y = F(x) + x → 梯度→直传 → 深度不退化!
   → → → 数学: dL/dx = dL/dy × (dF/dx + 1) → +1 → 梯度至少1 → 不消失!
   → → → → → 100+层 → ImageNet→3.57% → 超人类 → 深度=力量!
   → → → → → → → Transformer也用: Pre-Norm → 残差 → 梯度稳定 → 类似ResNet!

2. Transformer (2017) — 当前一切:
   → Self-attention: QK^T/√d × V → 全连接 → 位置无关 → 通用!
   → → → Multi-Head: 多头→不同子空间 → 多视角 → 信息丰富!
   → → → → → Position encoding: RoPE → 位置→频率 → 外推 → 推荐!
   → → → → → → → Layer structure: RMSNorm → Attention → Residual → MLP → Residual!
   → → → → → → → → → MLP: SwiGLU → gate_proj × SiLU(gate) × up_proj → LLaMA核心!

   → → → → → → → → → → → LLaMA架构 = RMSNorm + GQA + SwiGLU + RoPE → 现代!
   → → → → → → → → → → → → → DeepSeek-V3: MLA + MoE + MTP + FP8 → 创新!
   → → → → → → → → → → → → → → → Qwen: GQA-8 + SwiGLU + RoPE → 类LLaMA → 生产!

3. MoE (Mixture of Experts) — 稀疏激活:
   → Router → top-K experts → 稀疏激活 → 参数多→计算少!
   → → → DeepSeek-V3: 671B参数/37B激活 → 18x稀疏 → 高效!
   → → → → → 但: All-to-All瓶颈 → NVLink/RDMA必需 → RTX 4090不行!

架构决策树:
  → 小模型(<1B) → MLP → 简单 → 嵌入/分类!
  → → → 中模型(1-10B) → Transformer dense → LLaMA架构 → 推荐!
  → → → → → 大模型(>10B) → MoE → 稀疏 → DeepSeek → 需要 NVLink!
  → → → → → → → RTX 4090最优: 7B dense + INT4 → 单GPU → 推荐!
```

## 8. Core Laws — 深度学习基础核心定律

1. **Nonlinearity-Essential Law**: 非线性激活=神经网络必需 → 无非线性=多层=线性=无意义 → XOR需要非线性!
   → → → ReLU/GELU/SwiGLU → 不消失 → 推荐 → sigmoid/tanh → 消失 → 不推荐!

2. **Backprop-Chain Law**: 反向传播=链式法则 → 梯度→逐层传播 → sigmoid→0.25^N→消失 → ReLU→1→不消失!
   → → → Kaiming初始化→ReLU专用 → RMSNorm→LLaMA → Pre-Norm→梯度稳定!

3. **CE-MLE-Equivalence Law**: CE loss = MLE → SFT训练 = 最大似然 → 最简单最有效!
   → → → RL: CE + KL → GRPO → KL penalty → 防偏离 → 正则化!

4. **Dropout-Bayesian Law**: Dropout ≈ 贝叶斯模型平均 → Bernoulli采样 → 泛化↑ → 但大模型→不需要!
   → → → LoRA: 冻结base → 天然正则化 → 不需要额外dropout → 简单!

5. **Saddle-Point-Dominance Law**: 高维空间→鞍点主导→局部最小不严重 → Adam自适应→逃离鞍点!
   → → → 7B=7e9维 → 同时所有维最小→概率≈0 → 实际=鞍点→可逃离!

6. **Depth-Efficiency Law**: 深度>宽度 → 深度=层次化特征 → 指数级表达力 → 参数线性 → 可行!
   → → → 万能逼近: 1层足够宽 → 但宽度=指数 → 深度=线性 → 推荐!

7. **Residual-Gradient-Highway Law**: 残差连接 → 梯度直传 → +1 → 不消失 → ResNet/Transformer都用!
   → → → Pre-Norm → RMSNorm+Residual → LLaMA → 梯度稳定 → 推荐!

## 关键参考

- Backprop: 链式法则 → 梯度消失/爆炸 → ReLU解决 → Kaiming初始化
- CE = MLE: SFT训练=最大似然 → 见概率论笔记
- Dropout ≈ 贝叶斯: Bernoulli采样 → 模型平均 → 泛化
- Loss landscape: SFT→深峡谷(0.001) → GRPO→高地(5.06) → 精确解泛化好
- ResNet: 残差→梯度+1 → 深度不退化 → Transformer也用
- Transformer: self-attention → 位置无关 → 通用 → LLaMA=GQA+SwiGLU+RoPE+RMSNorm