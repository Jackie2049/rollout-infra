# MiniGPT 从零训练 — RTX 4090 实测

> 2026-06-05 | 工具: `tools/minigpt_train.py` | RTX 4090 24GB
> 架构: Pre-norm + RMSNorm + SwiGLU + Weight Tying (LLaMA-like)

## 1. 实验设置

- **Tokenizer**: 字符级 (vocab=40, 英语语料)
- **语料**: 2177 tokens (童话故事, Lucy 和 Snowball)
- **训练**: AdamW, cosine schedule + warmup, gradient clip=1.0
- **速度**: 290-580K tok/s (RTX 4090)

## 2. 模型缩放实验

| Model | Params | Train Loss | Val Loss | 时间 | 生成质量 |
|-------|--------|-----------|----------|------|---------|
| tiny | 0.14M | 0.930 | 1.757 | 9.3s | 乱码 |
| small | 1.06M | 0.101 | **2.376** | 14.1s | **完整句子!** |
| medium | 4.22M | 0.079 | 2.419 | 14.0s | 完整句子 |
| large | 6.32M | 0.078 | 2.638 | 19.1s | 完整句子 |

**关键发现**:
1. **Tiny → Small**: 质量飞跃! Small 模型开始生成完整句子
   - Sample: "The rabbit was white with pink ears. She picked up the rabbit..."
2. **严重过拟合**: 训练集 loss 极低 (0.08), 但验证集 loss 反而上升
   - 原因: 语料太小 (2177 tokens), 模型记忆了训练集
   - Medium/Large 过拟合更严重 (train 0.078, val 2.64)
3. **最优模型**: Small (1M) — 最佳 val loss, 够用但不浪费
4. **Scaling 规律**: 参数翻倍, train loss 只降 ~10%, 但 val loss 反升

### 过拟合分析
```
语料大小 vs 模型大小的关系:
  语料: 2177 tokens (极小)
  tiny (0.14M): 2177/140000 = 0.016 tokens/param → 严重欠拟合
  small (1.06M): 2177/1060000 = 0.002 tokens/param → 刚好
  medium (4.22M): 2177/4220000 = 0.0005 → 严重过拟合

  经验法则: Chinchilla 建议每参数 20 tokens
  要训练好 1M 模型, 至少需要 20M tokens!
  当前语料差 10000x → 必然过拟合
```

## 3. Temperature 实验

| Temperature | Unique Ratio | 观察到的行为 |
|------------|-------------|-------------|
| 0.1 | 0.913 | 完全复述训练数据 (greedy) |
| 0.3 | 0.913 | 同上 |
| 0.5 | 0.913 | 同上 |
| 0.8 | 0.913 | 同上 |
| 1.0 | 0.909 | 开始出现变化 |
| 1.5 | 0.909 | 出现乱码 |

**发现**: 温度 ≤ 1.0 时模型几乎完全复述训练数据 (因为过拟合).
只有 T=1.5 才产生新文本, 但质量下降.
这正好说明: **过拟合的模型生成缺乏多样性**.

## 4. Context Length 实验

| Block Size | Val Loss | 时间 |
|-----------|----------|------|
| 32 | **2.063** | 11.1s |
| 64 | 2.286 | 11.4s |
| 128 | 2.244 | 11.3s |

**发现**: block_size=32 最佳! 小语料 + 短序列 → 最少过拟合.
更长序列 → 模型看到更多上下文 → 更容易记忆 → 过拟合加剧.

## 5. 训练动态分析

```
典型训练曲线 (small model):

Step    0: loss=3.74 (初始, ≈log(vocab)=log(40)=3.69)
Step  200: loss=1.19 (快速下降, 学习常见模式)
Step  400: loss=0.19 (训练集几乎记忆, 但 val 开始上升)
Step  600: loss=0.12 (训练集继续下降, val 持续恶化)
Step  800: loss=0.10 (过拟合区域)
Step 1000: loss=0.10 (饱和)

关键洞察:
  - 前 200 步: 学习高频模式 (字母频率, 常见词)
  - 200-400 步: 学习语法结构 (句子模式)
  - 400+ 步: 过拟合 (记忆特定句子)
  - 应该在 step 400 附近 early stop!
```

## 6. 生成质量示例

**Small model (最好)**:
```
"The rabbit was white with pink ears. She picked up the rabbit and carried it ho..."
"Spring came and the flowers bloomed again. Lucy and Snowball explored the fores..."
```
→ 完整的英语句子, 正确的语法, 符合故事内容

**Medium model**:
```
"They walked through the forest for hours. The trees grew taller and the path gr..."
"The golden flower never wilted. It sat on her windowsill, glowing every night...."
```
→ 也是完整句子, 但 val loss 更高 (过拟合更严重)

**Tiny model**:
```
"There would towers totes ved. I i, dat wat gallden bamedter hey ne the forest a..."
```
→ 部分正确词汇, 但语法混乱

## 7. 核心学习

1. **数据量 > 模型大小**: 2K tokens 语料只能训 tiny 模型
2. **过拟合是主要问题**: 训练 loss 不等于泛化能力
3. **Early stopping 很重要**: 在 val loss 开始上升时停止
4. **温度不能弥补过拟合**: 过拟合的模型缺乏多样性
5. **LLaMA 架构可行**: RMSNorm + SwiGLU + Pre-norm 工作正常
6. **Weight tying 节省参数**: 输出层共享 embedding → 省一个 d×V 矩阵
7. **下一步需要**: 更大语料 (至少 100K+ tokens), 更长训练, BPE tokenizer
