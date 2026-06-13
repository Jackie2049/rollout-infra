# NLP Fundamentals Deep Dive — 4大架构(GPT Decoder-only/BERT Encoder-only/T5+BART Encoder-Decoder) + Tokenization(BPE主流+SentencePiece多语言+Unigram+Byte-level) + 语言建模(Autoregressive因果 vs Masked双向 → 2025 Decoder-only主导) + 任务分类(Classification/Generation/Translation/Summarization/QA) + 评估指标(BLEU/ROUGE/Perplexity/F1→COMET/BLEURT/LLM-as-Judge) + 2025趋势(Decoder-only主导+ModernBERT复兴Encoder+Prompt-based/Few-shot+RAG+Agent) + RTX 4090(7B INT4生成+BERT分类+T5翻译) + 与已有知识联系(Attention/Transformer/LLM serving/Quantization)

> 2026-06-14 | NLP基础深度分析: 4大Transformer架构(GPT autoregressive decoder-only→BERT masked encoder-only→T5 text-to-text encoder-decoder→BART denoising encoder-decoder) → Tokenization(BPE迭代合并最频繁→主流→GPT/LLaMA都用+SentencePiece语言无关→多语言+Unigram→概率最优+Byte-level→完全语言无关) → 语言建模(Autoregressive=P(x)=∏P(x_i|x_{<i})→生成→2025主流 vs Masked=15%mask→双向→理解→BERT/ModernBERT) → 任务(Classification/NER/QA→Encoder → Generation/Chat→Decoder → Translation/Summary→Enc-Dec) → 评估(BLEU n-gram precision→ROUGE recall→Perplexity=exp(-ΣlogP/N)→F1 precision×recall→2025: COMET/BLEURT neural+LLM-as-Judge→Chatbot Arena) → 2025趋势(Decoder-only主导+ModernBERT 8192+Prompt-based+RAG+Agent) → RTX 4090(7B INT4 generation+BERT classification+T5 translation)
> 关联: ai-expert-knowledge-map-gap-analysis.md(NLP ★★→★★★★), inference-perf skill(推理性能), evaluation-benchmarking-deep-dive.md(Benchmark+metrics), agent-system-deep-dive.md(Agent NLP)
> 参考: GPT(OpenAI 2018-2024), BERT(Devlin et al. 2018), T5(Raffel et al. 2019), BART(Lewis et al. 2019), ModernBERT(2025), BPE(Sennrich et al. 2016), SentencePiece(Kudo 2018), BLEU(Papineni 2002), ROUGE(Lin 2004)

## 0. 核心定律: 4大架构 × Tokenization × 语言建模 → NLP全景 → 2025 Decoder-only主导!

```
NLP核心架构:

  → GPT(Decoder-only) → Autoregressive → P(x)=∏P(x_i|x_{<i}) → 生成 → 2025主流!
  → → BERT(Encoder-only) → Masked → 双向理解 → 分类+检索 → 生产仍重要!
  → → → T5/BART(Encoder-Decoder) → Text-to-text/Denoising → 翻译+摘要 → 下降趋势!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Decoder-only=2025主流 → 但Encoder-only+Encoder-Decoder仍在特定场景有用!

  Tokenization:
    → BPE → 字符→合并最频繁pair→子词 → GPT/LLaMA → 主流!
    → → SentencePiece → 语言无关→byte stream→多语言 → T5/mT5 → 重要!
    → → → Unigram → 概率模型→最优分割 → SentencePiece内含 → 更优!
    → → → → Byte-level → 完全语言无关 → ByT5 → 前沿!

  语言建模:
    → Autoregressive → 下一个token → 因果 → 生成 → 2025主流!
    → → Masked → 随机mask→预测 → 双向 → 理解 → BERT → 特定场景!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Autoregressive=主流 → 但Masked仍有独特优势(双向上下文)!

  评估指标演进:
    → 传统: BLEU/ROUGE/Perplexity/F1 → n-gram → 快 → 但不完整!
    → → → → → 现代: COMET/BLEURT/LLM-as-Judge → neural+语义 → 更准确 → 2025标准!
```

## 1. Tokenization — 从字符到子词

```
### 1.1 BPE (Byte Pair Encoding) — 主流方法

BPE (Sennrich et al. 2016) → 字符→迭代合并最频繁pair → 子词 → GPT/LLaMA!

  算法:
    → Step 1: 初始词汇=所有字符 → a,b,c,d,e,... → 最小单位!
    → → → → Step 2: 计算所有相邻pair频率 → (a,b)=100, (e,r)=50, ... → 排序!
    → → → → → → → Step 3: 合并最频繁pair → (a,b)→"ab" → 加入词汇 → 更新!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Step 4: 重复 → 直到达到目标词汇量 → 32K/64K/128K → 完成!

  优势:
    → 处理未登录词(OOV) → "unbelievable" → "un"+"believ"+"able" → 子词 → 不遗漏!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 平衡词汇量+覆盖率 → 32K够 → 128K更好 → trade-off!

  2025趋势:
    → 词汇量增大 → GPT-2 50K → LLaMA-3 128K → 趋势增大!
    → → → → → → → 多语言公平 → 英文1.5 byte/token → 中文3-4 byte/token → 不公平!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 压缩率 → bytes/token → 衡量tokenizer效率 → 关键指标!

### 1.2 SentencePiece — 多语言

SentencePiece (Kudo 2018) → 语言无关 → byte stream → 中文/日文无空格 → 处理!

  核心:
    → 不需预分词 → 直接byte stream → 任何语言 → 无空格语言(中文/日文) → 处理!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 实现: BPE模式+Unigram模式 → 两种 → 灵活!

  Unigram Language Model → SentencePiece内含 → 概率最优:
    → 给定词汇 → 计算每子词概率 → 选择最优分割 → Viterbi → 最优!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 优势: 比BPE更优 → 概率模型 → 最优分割 → 但计算更复杂!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → T5/mT5/XLM-R → 用SentencePiece+Unigram → 多语言最优!

### 1.3 Byte-level → 完全语言无关

Byte-level → ByT5 → 每byte=1token → 256词汇 → 完全语言无关 → 但序列太长!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 优势: 256词汇 → 任何语言 → 任何字符 → 完全覆盖!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 劣势: 序列长度3-4x → 推理慢 → 实际不可行(太大)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Byte-level=理论最优 → 但实际不可行 → BPE/SentencePiece=实践最优!

### 1.4 Tokenizer对比

| Tokenizer | 词汇量 | 多语言 | 处理OOV | 压缩率 | 用途 |
|-----------|-------|--------|---------|--------|------|
| BPE | 32K-128K | 中 | ✅ | 好 | GPT/LLaMA主流 |
| SentencePiece+Unigram | 32K-128K | ✅极好 | ✅ | 好 | T5/XLM-R多语言 |
| WordPiece | 30K | 中 | ✅ | 中 | BERT |
| Byte-level | 256 | ✅完美 | ✅完美 | 低(3-4x长) | ByT5研究 |

→ → → → → → → → → → → → → → → → → → → → → → → → → → 结论: BPE=主流生成 → SentencePiece=多语言最优 → Byte-level=理论最优但不可行 → 分场景选择!
```

## 2. 语言建模 — Autoregressive vs Masked

```
### 2.1 Autoregressive (因果语言建模)

Autoregressive LM → P(x) = ∏ P(x_i | x_{<i}) → 左到右 → 下一个token → GPT!

  数学:
    → P("the cat sat") = P("the") × P("cat"|"the") × P("sat"|"the cat")
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 每个token → 只看前面 → 因果 → 不能看后面 → 但生成足够!

  训练目标:
    → NLL(Negative Log-Likelihood) → Cross-Entropy → -Σ log P(x_i | x_{<i})
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 这就是我们之前学的CE loss → NLP训练=CE → 一致!

  注意力:
    → Causal mask → 下三角 → token i只能看token 1..i-1 → 不能看i+1..n → 因果!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 与FlashAttention → causal模式 → 我们已学 → 实测!

  优势: 生成自然 → 训练简单 → 推理方便 → scaling law好 → 2025主流!
  劣势: 不能用双向上下文 → 限制理解能力 → 但large enough model克服!

### 2.2 Masked Language Modeling (双向)

Masked LM → BERT → 15%随机mask → 双向预测 → 理解更强!

  方法:
    → 选择15%token → 80%替换[MASK] → 10%随机替换 → 10%保留 → 多样化!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 模型 → 双向注意力 → 看[MASK]左右所有 → 预测原始token → 上下文丰富!

  数学:
    → L_MLM = -Σ log P(x_i | x_{all except i}) → 双向 → 更丰富上下文!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → vs Autoregressive → P(x_i|x_{<i}) → 只看左边 → 上下文少!

  优势: 双向 → 更丰富上下文 → 理解更强 → 分类/检索/NER更好!
  劣势: 不能自然生成 → [MASK]在推理时不存在 → 不适合生成!

### 2.3 2025趋势 — Decoder-only主导

2025 → Decoder-only主导 → 但Encoder-only+Encoder-Decoder仍有特定场景!

  为什么Decoder-only主导?
    → Scaling law → Decoder-only scaling更好 → 更多参数→更多性能 → 更简单!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 简单架构 → 无cross-attention → 无encoder-decoder分离 → 更易实现!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大模型 → 双向能力可以通过prompt+context弥补 → 不需专门encoder!

  ModernBERT(2025) → Encoder-only复兴:
    → 8192 context → RoPE → GeGLU → 2T tokens → 2025现代encoder!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 优势: 分类+检索+NER+embedding → Decoder-only不如 → 特定场景最优!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Decoder-only=2025主流生成 → Encoder-only=特定场景最优 → 不对立 → 互补!
```

## 3. 4大架构对比 — GPT/BERT/T5/BART

```
### 3.1 架构对比表

| 特征 | GPT | BERT | T5 | BART |
|------|-----|------|-----|------|
| **架构** | Decoder-only | Encoder-only | Encoder-Decoder | Encoder-Decoder |
| **注意力** | Causal(masked) | 双向 | Enc:双向+Dec:因果+Cross | Enc:双向+Dec:因果+Cross |
| **预训练** | Next-token | MLM+NSP | Span corruption | Denoising reconstruction |
| **生成** | ✅ 极强 | ❌ 不能 | ✅ 可以 | ✅ 可以 |
| **理解** | 中 | ✅ 极强 | ✅ 可以 | ✅ 可以 |
| **分类** | 中 | ✅ 极强 | ✅ 可以 | ✅ 可以 |
| **检索** | 中 | ✅ 极强 | 中 | 中 |
| **翻译** | 中 | ❌ | ✅ 极强 | ✅ 强 |
| **摘要** | ✅ 强 | ❌ | ✅ 极强 | ✅ 极强 |
| **2025地位** | 主流(frontier) | 生产(分类+检索) | 下降(被decoder-only替代) | 下降(特定场景) |
| **模型大小** | 7B-1.8T | 340M-1.5B | 220M-11B | 400M-1.5B |

### 3.2 预训练目标详解

GPT → Next-token prediction:
  → P(x_i|x_{<i}) → 自回归 → 简单 → 自然 → scaling好!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 训练数据 → 大量文本 → 每个token都是训练信号 → 高效!

BERT → MLM + NSP:
  → MLM: 15%mask → 预测 → 双向 → 理解信号更强 → 但mask只在训练时!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → NSP: Next Sentence Prediction → 是否连续 → 后来被RoBERTa证明不太有用!

T5 → Span Corruption:
  → 随机mask连续span → "Thank you <X> me to your party <Y> week" → 预测"<X> invited <Y> last"
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Text-to-text → 所有任务都是text→text → 统一框架 → 灵活!

BART → Denoising Autoencoder:
  → 多种噪声: 删除+置换+mask+旋转 → 模型重建原文 → 更鲁棒!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 生成+理解 → 两者兼顾 → 比T5更鲁棒 → 但规模不及GPT!

### 3.3 2025架构趋势

2025趋势 → Decoder-only主导 → 但 Encoder/Enc-Dec 不死 → 特定场景最优!

  → Frontier模型 → 全Decoder-only → GPT-4o/Claude 3.5/Gemini/Llama 3 → 主流!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 生产分类+检索 → Encoder-only(BERT/ModernBERT) → 更高效 → 不用7B做分类!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 翻译+摘要 → 可以用Decoder-only → 但Enc-Dec仍可能更优 → 少场景!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RAG检索 → Encoder-only → embedding → ChromaDB → 最高效 → 不能用7B做embedding!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Decoder-only=主流 → 但 Encoder-only=分类+检索+embedding最优 → 分场景选择!
```

## 4. NLP任务分类

```
### 4.1 任务 → 架构映射

| 任务 | 最优架构 | 原因 | 代表模型 |
|------|---------|------|---------|
| 文本分类 | Encoder-only | 双向+短序列+不需要生成 | BERT/RoBERTa/ModernBERT |
| NER | Encoder-only | 需要理解每个token上下文 | BERT |
| 问答QA | Encoder-only/Decoder | Extractive→Encoder / Generative→Decoder | BERT/GPT |
| 文本生成 | Decoder-only | 自然生成 → scaling好 | GPT/Llama |
| 对话/Chat | Decoder-only | 多轮生成 → 交互 | GPT/Claude |
| 翻译 | Encoder-Decoder | 需理解源+生成目标 → 两种语言 | T5/mT5 |
| 摘要 | Encoder-Decoder/Decoder | 需理解长文+生成短文 → 缩减 | BART/GPT |
| 代码生成 | Decoder-only | 生成 → 代码 → token-by-token | GPT-4/CodeLlama |
| 检索/Embedding | Encoder-only | 双向 → 语义相似度 → embedding | BERT/E5 |
| 情感分析 | Encoder-only | 分类 → 简单 → BERT够用 | BERT |

### 4.2 Prompt-based/Few-shot — 2025 NLP范式转变

传统NLP → 每任务fine-tune → 需标注数据 → 专门模型 → 10+模型!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 2025 NLP → Prompt+Few-shot → 一个模型 → 所有任务 → 通用!

  In-Context Learning:
    → 给模型几个示例 → "Input: X → Output: Y" → 模型模仿 → 不需训练!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Few-shot → 3-5个示例 → 性能接近fine-tune → 但不需训练 → 更灵活!

  Zero-shot:
    → 不给示例 → 直接描述任务 → "Translate to French: ..." → 模型理解 → 简单!
    → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 大模型 → zero-shot效果好 → 不需fine-tune → 通用 → 2025主流!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: Prompt/Few-shot → 一个模型所有任务 → 不需fine-tune → 2025范式转变! → 但特定任务fine-tune仍更精确 → 分场景!
```

## 5. NLP评估指标 — 传统→现代

```
### 5.1 传统指标

| 指标 | 类型 | 用途 | 公式 | 限制 |
|------|------|------|------|------|
| BLEU | 精确率 | 翻译 | n-gram precision | 不捕获语义+忽略recall |
| ROUGE | 召回率 | 摘要 | n-gram recall | 不捕获语义+依赖参考 |
| Perplexity | 困惑度 | 语言模型 | exp(-ΣlogP/N) | 只测流畅性+不测质量 |
| F1 | 精确率×召回率 | 分类/NER | 2×P×R/(P+R) | 分类足够+但不够生成 |

### 5.2 现代指标(2025)

COMET (2020+) → Neural-based → 翻译评估 → 语义+参考+源 → 最准确翻译评估!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → BLEURT → BERT-based → 语义相似 → 比BLEU更准确 → 但需训练!

LLM-as-Judge → 2025主流生成评估 → GPT-4/Claude评判 → 80-90%人类一致!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Pairwise比较 → "哪个更好?" → Bradley-Terry → Elo → Chatbot Arena!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Single-point评分 → "1-5分" → 结构化 → MT-Bench!

  偏见:
    → Position bias → 左右位置 → 随机 → 控制!
    → → → → → Verbosity bias → 长回答≠好回答 → 控制!
    → → → → → → → Self-preference → 自我风格偏好 → 控制!

### 5.3 指标选择决策树

  翻译评估 → BLEU(快速)+COMET(准确) → 组合!
  → → → 摘要评估 → ROUGE(快速)+LLM-as-Judge(准确) → 组合!
  → → → → → 分类/NER → F1 → 足够!
  → → → → → → → 语言模型 → Perplexity → 基础 → 但需下游任务验证!
  → → → → → → → → → 对话/Chat → Chatbot Arena → 人类偏好 → 最真实!
  → → → → → → → → → → → 生成质量 → LLM-as-Judge+人工 → 组合!

→ → → → → → → → → → → → → 结论: 传统指标+现代指标 → 组合 → 不依赖单一指标 → BLEU+COMET+Arena → 三层!
```

## 6. RTX 4090 NLP策略

```
### 6.1 RTX 4090 NLP推理

| 模型 | 类型 | INT4推理 | 任务 | RTX 4090可行性 |
|------|------|---------|------|---------------|
| Llama-3-8B | Decoder-only | ~4800 tok/s | 生成/对话/翻译 | ✅ 极好 |
| BERT-base | Encoder-only | ~100 samples/s | 分类/NER/检索 | ✅ 极好(小模型) |
| ModernBERT | Encoder-only | ~50 samples/s | 分类+检索(8192 ctx) | ✅ 极好 |
| T5-base | Enc-Dec | ~200 tok/s | 翻译/摘要 | ✅ 可行 |
| Phi-3-mini | Decoder-only | ~150 tok/s | 轻量生成 | ✅ 实时 |

关键策略:
  → 分类/NER → BERT/ModernBERT → 小模型 → 快 → RTX 4090绰绰有余!
  → → → → → → 生成/对话 → 7B INT4+INT8KV+SGLang → 我们已测 → 4,791 tok/s!
  → → → → → → → → → → 翻译/摘要 → T5或7B INT4 → 都可行 → 但7B更通用!
  → → → → → → → → → → → → 检索/Embedding → BERT/E5 → 最高效 → ChromaDB+向量搜索 → RAG!

  → → → → → → → → → → → → → → 结论: RTX 4090 → 分类用BERT → 生成用7B INT4 → 检索用BERT → 全链路NLP可行!
```

## 7. 与已有知识的联系

```
Attention → NLP核心:
  → Self-attention → Transformer → NLP基础 → 我们已深度学(FlashAttention/FlashInfer)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Causal attention → GPT → 只看前面 → 我们已实测(causal vs bidirectional)!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Cross-attention → T5/BART → encoder→decoder → 理解 → 生成桥梁!

Transformer → NLP架构:
  → 我们已学: Transformer variants(GQA/MLA/MoE) → 全是Decoder-only → NLP生成!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Encoder-only(BERT) → Transformer+双向 → 分类+检索 → 特定最优!

LLM Serving → NLP推理:
  → vLLM/SGLang → NLP推理 → PagedAttention → KV cache → 我们核心!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → INT4+INT8KV → 7B → 4,791 tok/s → NLP生成最优!

Quantization → NLP部署:
  → INT4 BERT → 推理加速 → 但BERT太小 → INT4无必要 → bf16够快!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → INT4 7B → 生成加速 → 我们已学 → NLP生产部署!

Evaluation → NLP评估:
  → BLEU/ROUGE → NLP经典评估 → 我们evaluation deep dive中已学!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → LLM-as-Judge → NLP现代评估 → Chatbot Arena → 已学!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → Perplexity → 语言模型评估 → 我们已测(RTX 4090 PPL benchmark)!

Agent → NLP应用:
  → Agent = NLP+工具 → 生成+调用 → 我们agent deep dive中已学!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RAG = 检索+生成 → NER+embedding+LLM → 全NLP链路!

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: NLP与已有知识高度关联 → Attention/Transformer/Serving/Quantization/Evaluation/Agent → 全链路!
```

## 8. 核心规律

```
NLP核心规律:

  1. 4架构 → GPT(生成主流)+BERT(理解最优)+T5/BART(翻译/摘要) → 分场景!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 2025: Decoder-only=主流 → 但BERT=分类+检索最优 → 不对立 → 互补!

  2. Tokenization → BPE=主流+SentencePiece=多语言 → 32K-128K词汇 → 子词平衡!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 多语言公平 → 英文1.5 byte/token → 中文3-4 → 需优化 → SentencePiece!

  3. 语言建模 → Autoregressive=生成主流+Masked=理解最优 → 互补 → 不替代!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → ModernBERT(2025) → Encoder复兴 → 8192 ctx → 分类+检索 → 不被Decoder替代!

  4. 评估 → 传统(BLEU/ROUGE/PPL)→现代(COMET/LLM-as-Judge/Arena) → 组合!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → BLEU快速+COMET准确+Arena真实 → 三层 → 不依赖单一!

  5. Prompt/Few-shot → 一个模型所有任务 → 不需fine-tune → 2025范式!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 但特定任务fine-tune仍更精确 → 分类用BERT → 生成用GPT → 分场景!

  6. RTX 4090 → 分类BERT+生成7B INT4+检索BERT+翻译T5 → 全NLP链路可行!
  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → BERT小→快 → 7B INT4→生成 → RAG→检索+生成 → 全链路!

  知识Gap修复:
    → NLP从★★(2/5) → ★★★★(4/5) → 4架构+Tokenization+语言建模+任务+评估+RTX 4090 → 全面!
    → → → → → 但仍需实践 → GPU可用时 → BERT分类+7B生成+RAG检索 → 实测!
```

## 参考文献

```
1. 架构:
   - GPT: OpenAI 2018-2024 → GPT-2/3/4 → Decoder-only → 生成主流
   - BERT: Devlin et al. 2018 → Encoder-only → 双向理解
   - T5: Raffel et al. 2019 → Encoder-Decoder → text-to-text
   - BART: Lewis et al. 2019 → Encoder-Decoder → denoising
   - ModernBERT: 2025 → Encoder-only复兴 → 8192 ctx

2. Tokenization:
   - BPE: Sennrich et al. 2016 → 子词 → GPT主流
   - SentencePiece: Kudo 2018 → 语言无关 → 多语言
   - WordPiece: BERT → 类似BPE

3. 评估:
   - BLEU: Papineni et al. 2002 → 翻译精确率
   - ROUGE: Lin 2004 → 摘要召回率
   - COMET: 2020+ → Neural翻译评估
   - BLEURT: 2020+ → BERT-based评估
   - LLM-as-Judge: Zheng et al. 2023 → Chatbot Arena

4. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → NLP gap评估
   - evaluation-benchmarking-deep-dive.md → BLEU/ROUGE/Perplexity+LLM-as-Judge
   - agent-system-deep-dive.md → Agent NLP应用
   - inference-perf skill → 推理性能+RTX 4090

Sources:
- [Hugging Face NLP Course](https://huggingface.co/course)
- [ModernBERT](https://huggingface.co/blog/modernbert)
- [SentencePiece](https://github.com/google/sentencepiece)
- [BPE Paper](https://arxiv.org/abs/1508.07909)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [BART Paper](https://arxiv.org/abs/1910.13461)
