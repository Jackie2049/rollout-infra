# Data Pipeline & Curation for LLM Training Deep Dive — 数据质量>模型架构(Common Crawl→MinHash LSH去重30-40%→Quality Filtering→PII Removal→DoReMi最优混合+Contamination检测+合成数据Self-Instruct/RL自我博弈+Data Wall 2028) + 生产数据Pipeline(Datatrove+FineWeb+DCLM+Nemotron-CC) + RTX 4090数据策略

> 2026-06-14 | LLM训练数据pipeline深度分析: 数据质量是训练最关键因素(质量×3>数量) → 从数据源(Common Crawl 250B页+ArXiv+GitHub+书籍)到生产pipeline(language detection→MinHash LSH dedup→quality scoring→PII removal→domain labeling) → 数据混合策略(DoReMi minimax excess loss+domain proportion optimization) → 数据污染检测(n-gram overlap 13-token+contamination-benchmark interaction) → 合成数据(Self-Instruct+self-play with verification+model collapse风险) → Data Wall(2024-2026人类高质量文本耗尽→RL自我博弈→无限数据) → RTX 4090数据策略(小模型→高质量子集→LoRA微调)
> 关联: ai-expert-knowledge-map-gap-analysis.md(数据gap★→★★★★), scaling-laws-deep-dive.md(Data Wall+数据质量), ai-safety-guardrails-production-deep-dive.md(数据投毒OWASP#3)
> 参考: FineWeb(EPFL), DCLM(DataComp for Language Models), Nematron-CC(NVIDIA), Datatrove(HuggingFace), DoReMi(Xie et al. 2023), Self-Instruct(Wang et al. 2023), Chinchilla(Hoffmann et al. 2022)

## 0. 核心定律: 数据质量>模型架构 → 质量×3>数量 → curated 5TB≈raw 30TB → 数据是训练最关键因素!

```
数据质量定律:

  质量×3>数量:
    → curated过滤95% → 5-10TB高质量 → 等价30TB原始 → 3x效率!
    → → FineWeb: Common Crawl 250B页 → 过滤 → 1.3TB高质量 → 8B模型+FineWeb=70.5% MMLU
    → → → vs 250B原始 → 8B模型+原始=40-50% MMLU → 质量差距20+百分点!

  数据决定性能上限 → 模型只逼近上限:
    → 好数据+小模型 > 坏数据+大模型 → 反直觉但真实!
    → → Phi-1(1.3B)+高质量代码数据 > Llama 1(7B)+普通数据 → TextBooks>Web!
    → → → DCLM: 7B+curated=64.8% MMLU > 7B+raw=46% → 差距18.8%!

  数据混合比架构更重要:
    → 同一架构+不同混合 → 性能差距可达10+百分点!
    → → DoReMi: 最优混合 vs 默认混合 → 差距5-10% → 不改架构只改数据!

→ → → 结论: 训练优先级 = 数据质量 > 数据混合 > 模型架构 > 训练超参 → 反直觉!
```

## 1. 数据源 — LLM训练数据的来源与规模

```
### 1.1 主要数据源

1. Common Crawl — Web数据(最大源):
   → 规模: 250B+网页 → 20-30PB原始HTML → 最大的公开数据源!
   → → 爬取: 每月1-2B新页 → 持续增长 → 但质量参差!
   → → → 问题: 重复率高(30-40%)→质量低(50%+垃圾)→有害内容→需要大量清洗!
   → → → → → 使用: 所有主流LLM都用CC → Llama/GPT/Claude/Mistral → 基础数据!

2. ArXiv — 学术论文:
   → 规模: 2M+论文 → LaTeX+PDF → 高质量学术!
   → → 优势: 严谨 → 数学公式 → 科学推理 → 高质量!
   → → → 问题: 领域窄 → 数量有限 → 不够训练基础模型!
   → → → → → 使用: 数学推理 → 科学理解 → 辅助数据!

3. GitHub — 代码:
   → 规模: 100M+仓库 → 多语言代码 → 最大代码源!
   → → 优势: 实际代码 → 多语言 → 注释+文档 → 代码理解!
   → → → 问题: 低质量代码多 → 需过滤(stars/language/length) → 重复!
   → → → → → 使用: 代码生成 → 程序理解 → StarCoder/CodeLlama!

4. 书籍 + Wikipedia:
   → 规模: 数万书籍 → 6M Wikipedia → 高质量知识!
   → → 优势: 结构化知识 → 长上下文 → 专业领域 → 高质量!
   → → → 问题: 版权问题 → 数量有限 → 但质量极高!
   → → → → → 使用: 知识理解 → 长文本 → 辅助高质量数据!

5. StackExchange + Reddit:
   → 规模: 50M+问答 → 对话式 → 多领域!
   → → 优势: QA格式 → 人类偏好 → 多视角 → RLHF数据!
   → → → 问题: 低质量回答 → 偏见 → 需过滤(upvotes)!
   → → → → → 使用: QA能力 → 对话 → 偏好对齐!

### 1.2 数据规模与模型参数比例

Chinchilla定律: N参数需要20×N tokens → 最优数据量!
  → 7B模型 → 140B tokens → 可以训练到最优!
  → → 70B模型 → 1.4T tokens → 需大量数据 → CC必须!
  → → → 405B模型 → 8T tokens → 数据量决定大模型可行性!

  → → → → 关键: 数据量不足 → 模型欠训练 → 性能远低于Chinchilla最优!
  → → → → → 例: GPT-3 175B + 300B tokens → 欠训练 → Chinchilla说需要3.5T!
```

## 2. 数据清洗Pipeline — 从原始到高质量

```
### 2.1 生产数据Pipeline步骤(7步)

Step 1 — URL/文档级去重(dedup):
  → 去除完全相同URL/文档 → 减少冗余 → 第一步!
  → → 方法: URL hashing → hash比较 → 完全相同→删除 → 简单但有效!
  → → → 效果: 去除10-15%完全重复 → 减少冗余 → 但近似重复仍需后续!

Step 2 — Language Detection(语言检测):
  → fastText language identifier → 176语言 → 分类每段文本!
  → → 目标: 只保留目标语言(英语为主) → 过滤其他语言 → 减少噪声!
  → → → 效果: 过滤20-30%非目标语言 → 减少噪声 → 提高质量!
  → → → → → 多语言模型 → 保留多语言 → 但每种语言需单独质量过滤!

Step 3 — MinHash LSH Deduplication(近似去重):
  → 核心: MinHash+LSH → 检测近似重复 → 最重要的去重方法!
  → → → → 详细见 Section 3!

Step 4 — Quality Filtering(质量过滤):
  → 核心: heuristic+learned classifiers → 过滤低质量 → 最关键!
  → → → → 详细见 Section 4!

Step 5 — PII Removal(个人信息移除):
  → 检测+移除: email/phone/SSN/address/IP → 保护隐私 → 合规!
  → → 方法: regex patterns → 正则匹配 → 高精度!
  → → → 效果: 移除99%+PII → 但可能误删非PII → 需平衡!
  → → → → → 合规: GDPR/EU AI Act → 必须移除PII → 否则法律风险!

Step 6 — Domain Labeling(领域标注):
  → 分类文本领域 → science/code/math/news/fiction → 混合权重!
  → → 方法: fastText classifier → 预训练领域分类器 → 标注每段!
  → → → 效果: 精确混合 → DoReMi用领域标注 → 最优混合!

Step 7 — Final Curation(最终精选):
  → 人工+模型验证 → 最终精选 → 最高质量!
  → → 方法: 人工抽样审查 → 模型打分 → 双重验证 → 最终集!
  → → → 效果: 确保最高质量 → 但成本高 → 只能抽样!

### 2.2 生产Pipeline对比

FineWeb (EPFL 2024):
  → Common Crawl → 96 snapshots → 250B页 → 1.3TB高质量!
  → → 步骤: URL dedup→language→MinHash→quality→PII→domain→最终!
  → → → 关键: 严格MinHash(Jaccard 0.7) → 去重30-40% → 高质量!
  → → → → → 结果: 8B+FineWeb=70.5% MMLU → 最优web数据集!

DCLM (DataComp for Language Models, 2024):
  → 竞赛框架 → 7B固定架构 → 只变数据 → 证明数据重要性!
  → → → baseline=46% MMLU → curated=64.8% → 差距18.8% → 数据决定性!
  → → → → → 关键: 证明了数据质量>架构 → 不改模型只改数据 → 18.8%提升!

Nemotron-CC (NVIDIA 2024):
  → Common Crawl → NVIDIA优化 → 30TB cleaned → 1.8TB deduped → 高质量!
  → → → 步骤: NVIDIA custom pipeline → 更严格过滤 → PII → dedup!
  → → → → → 特点: 商业级pipeline → 多格式(parquet+TFRecord) → vLLM兼容!

Datatrove (HuggingFace 2024):
  → 开源pipeline框架 → 模块化 → 可组合 → 灵活!
  → → → 步骤: 每步独立pipeline → 自定义 → 可替换 → 灵活!
  → → → → → 特点: 大规模分布式 → Slurm/Dask → 可扩展 → 研究+生产!

→ → → → → → 结论: FineWeb最严格→1.3TB最优; DCLM最证明→数据决定性; Nemotron商业级→30TB清洗!
```

## 3. MinHash LSH Deduplication — 近似去重核心算法

```
### 3.1 MinHash原理

MinHash = Jaccard相似度的近似估计 → 无需比较所有对 → 大规模去重!

Jaccard相似度: J(A,B) = |A∩B| / |A∪B|
  → 精确计算 → 需比较所有元素 → O(|A|×|B|) → 太慢!

MinHash近似: h_min(A) ≈ J(A,B) → 期望值等于Jaccard!
  → → 方法: k个hash函数 → 每个hash取最小值 → k个最小值组成signature!
  → → → → → signature相似度 ≈ Jaccard → 不需要比较原始集合!

### 3.2 LSH(Locality-Sensitive Hashing) — 大规模加速

MinHash解决了近似估计 → 但仍需比较所有pair → N²问题 → LSH解决!

LSH原理:
  → 将signature分成b bands → 每band r rows → b×r=k!
  → → 如果两个文档在至少1个band完全匹配 → 候选相似对 → 需验证!
  → → → → → 概率分析:
    → 单band全匹配概率 = (1/s)^r → s是Jaccard相似度!
    → 至少1band匹配概率 = 1-(1-(1/s)^r)^b → 候选概率!

参数选择(b,r)控制精度:
  → Jaccard阈值=0.7 → b=10,r=10 → (1/0.7)^10≈0.03 → 单band=3%
  → → → → → → → 至少1band = 1-(1-0.03)^10 ≈ 26% → 可能漏!
  → → → → → → → → → 改进: b=20,r=5 → (1/0.7)^5≈0.18 → 至少1band=1-(1-0.18)^20≈99% → 几乎不漏!

  → Jaccard阈值=0.8 → b=10,r=10 → (1/0.8)^10≈0.107 → 至少1band=1-(1-0.107)^10≈69%
  → → → → → → → 改进: b=14,r=7 → 更好!

### 3.3 生产MinHash实践

FineWeb MinHash配置:
  → hash函数: k=128 → 128个MinHash值 → 精确!
  → → LSH bands: b=16, r=8 → 128=16×8 → 阈值≈0.7!
  → → → n-gram: 5-gram(字符级) → "hello world" → {"hell","ello","llo ","lo w","o wo"," wor","worl","orld"}
  → → → → → 效果: 去除30-40%近似重复 → 大规模 → 分布式!

Nemotron-CC MinHash:
  → k=256 → 更精确 → 但更慢 → 成本2x!
  → → Jaccard阈值=0.8 → 更严格 → 只去高度相似 → 保留更多多样性!
  → → → → → 效果: 去除25-35% → 比FineWeb保留更多 → 但质量稍低!

Dedup策略对比:
  → 精确去重(URL/hash) → 去除10-15% → 简单 → 必须第一步!
  → → MinHash LSH(J=0.7) → 去除30-40% → 近似 → 最关键!
  → → → MinHash LSH(J=0.8) → 去除25-35% → 更严格 → 保留更多!
  → → → → → → 结论: 先精确→再MinHash→J=0.7最平衡→去除30-40%→大幅去冗余!

### 3.4 MinHash去重对模型性能影响

去重=必要 → 不去重=灾难:
  → 不去重: 模型记忆重复 → evaluation污染 → 过拟合重复 → 浪费compute!
  → → → 重复数据=浪费 → 10x重复→compute浪费10x → 但性能不提升!
  → → → → → 评估污染: 训练集包含benchmark题 → 模型"作弊" → 虚高!
  → → → → → → → 去重后: 真实性能暴露 → 可能下降5-10% → 但真实!

  → 去重后性能:
    → FineWeb: 去重后+高质量 → 8B=70.5% MMLU → 去重必要!
    → → → 不去重+原始CC → 8B=40-50% → 差距20+百分点 → 去重决定性!
```

## 4. Quality Filtering — 数据质量过滤方法

```
### 4.1 Heuristic Filtering(启发式过滤)

最基础 → 快速 → 但粗糙 → 需配合learned:

1. Length filter:
   → 太短(<50 tokens) → 无信息 → 删除!
   → → 太长(>100K tokens) → 可能噪声 → 删除!
   → → → 效果: 去除5-10% → 简单 → 但可能误删好内容!

2. Language model perplexity:
   → 用小LM(n-gram或小Transformer)计算perplexity → 过滤异常!
   → → → 高ppl → 不自然 → 可能垃圾 → 删除!
   → → → → → 低ppl → 太重复 → 可能模板 → 删除!
   → → → → → → → 效果: 去除10-15% → 比length更精确!

3. Word frequency/ratio:
   → 特殊符号比例 → {}/<> → 代码混入文本 → 过滤!
   → → → 重复词比例 → 太高 → 重复 → 删除!
   → → → → → 效果: 去除5-10% → 简单 → 但领域特定!

4. "Boilerplate" removal:
   → 常见模板 → "点击这里"/"版权声明"/导航 → 去除!
   → → → 方法: regex → 常见模式 → 去除模板内容!
   → → → → → 效果: 去除5-10% → 减少噪声 → 提高有效信息密度!

### 4.2 Learned Quality Filtering(学习型过滤)

更精确 → 但需要训练 → 主流方法:

1. fastText classifier:
   → 训练二分类器 → 高质量vs低质量 → 分类每段文本!
  → → → 训练数据: Wikipedia(高质量)+random web(低质量) → 二分类!
  → → → → → 效果: 去除15-25%低质量 → 比heuristic更精确!

2. Perplexity filtering with GPT-2/3:
   → 用预训练LM计算ppl → 过滤极端值 → 双向过滤!
  → → → 高ppl(>阈值) → 不自然 → 删除 → 去除垃圾!
  → → → → → 低ppl(<阈值) → 重复/模板 → 删除 → 去除boilerplate!
  → → → → → → → 效果: 去除10-20% → 比小LM更精确 → 但成本更高!

3. FineWeb质量分类器:
   → 专用quality classifier → 多维评分 → 综合!
  → → → 特征: length+ppl+language+special_char+重复率 → 综合!
  → → → → → 效果: FineWeb最终1.3TB → 从250B → 过滤99.5% → 极严格!

4. CosmoQualityClassifier (DCLM):
   → 专门为DCLM训练 → 7B评估 → 最优!
  → → → 特点: 人工标注训练数据 → 1000+样本 → 精确分类!
  → → → → → 效果: DCLM curated → 64.8% MMLU → 18.8%提升!

### 4.3 质量过滤对性能影响

质量过滤=性能提升 → 但过度过滤=多样性丧失:

适度过滤(FineWeb/DCLM):
  → 过滤95% → 5TB高质量 → 性能最优 → 多样性保留!

过度过滤:
  → 过滤99% → 1TB → 高质量但太窄 → 多样性不足 → 偏见!

过滤不足:
  → 过滤50% → 15TB → 噪声多 → 性能下降 → compute浪费!

→ → → → → → 结论: 95%过滤 → 5TB → 最平衡 → 高质量+多样性!
```

## 5. Data Mixing Strategies — 数据混合策略

```
### 5.1 默认混合比例

典型LLM混合(参考Llama 2):
  → Web text: 67% → Common Crawl清洗 → 最大比例!
  → → Books: 5% → 高质量长文本 → 知识深度!
  → → → ArXiv: 4% → 学术 → 数学推理!
  → → → → → GitHub: 4.5% → 代码 → 程序能力!
  → → → → → → → Wikipedia: 4.5% → 结构化知识 → 准确性!
  → → → → → → → → → StackExchange: 2% → QA → 对话!
  → → → → → → → → → → → 其他: 13% → 多样性!

### 5.2 DoReMi — 最优数据混合(数据比例最关键!)

DoReMi = Distributional Robustness Optimization → minimax excess loss → 最优混合!

核心思想:
  → 不用固定比例 → 用optimization找到最优比例 → 数据自适应!
  → → → 方法: minimax excess loss → 找让所有domain表现最好→最差domain损失最小!
  → → → → → → 例: 数学弱 → 增加数学比例 → 直到数学不再最弱 → 最优!

DoReMi算法:
  Step 1: 用参考模型(ref)训练 → 在各domain计算loss!
  Step 2: 计算excess loss → L_excess = L_domain - L_ref → 每个domain超出参考多少!
  Step 3: Minimax优化 → 调整domain比例 → 让最大excess loss最小 → 最公平!
  → → → → → 结果: 最优比例 → 最弱domain提升最大 → 整体性能最优!

DoReMi效果:
  → vs 默认混合 → 整体提升5-10% → 最弱domain提升10-20%!
  → → → 例: 数学从60%→72% → 程序从55%→65% → 巨大提升!
  → → → → → 关键: 不改模型 → 只改数据比例 → 5-10%提升 → 简单有效!

### 5.3 数据混合注意事项

1. 领域平衡:
   → 不能只加web → 其他领域太弱 → DoReMi自动平衡!

2. 多语言:
   → 英语过多 → 其他语言弱 → 需专门混合 → 每语言单独优化!

3. 代码比例:
   → 太多 → 模型偏代码 → 其他能力弱 → 适度(4-10%)!
   → → 太少 → 代码能力差 → 适度增加 → 但不要超过10%!

4. 对话/QA:
   → StackExchange/Reddit → 对话能力 → RLHF基础 → 适度!

→ → → → → 结论: DoReMi自动最优 → 但需参考模型 → 训练成本2x → 但数据质量提升5-10% → 值得!
```

## 6. Data Contamination Detection — 数据污染检测

```
### 6.1 什么是数据污染?

数据污染 = 训练集包含benchmark题 → 模型"作弊" → 虚高 → 真实性能被掩盖!

问题严重:
  → 不去重 → 30-40%近似重复 → 包含benchmark → 模型记忆!
  → → evaluation虚高 → 5-10% → 不真实 → 误导决策!
  → → → → → 修正: 去重+污染检测 → 真实性能 → 正确评估!

### 6.2 n-gram Overlap Detection

最实用 → 简单 → 有效 → 主流方法:

方法:
  → 从benchmark提取13-gram → 检查训练集是否包含 → 匹配=污染!
  → → → 13-gram选择: 太短(3-gram)→误报多 → 太长(50-gram)→漏报多 → 13-gram最平衡!
  → → → → → 效果: 检测90%+污染 → 简单 → 可扩展!

GPT-4污染检测:
  → 13-gram overlap → 发现MMLU/GSM8K等benchmark → 训练集包含!
  → → → → → 结果: 污染后性能虚高 → 去污染后真实 → 差距5-10%!

### 6.3 Contamination与Benchmark交互

污染→虚高 → 去污染→真实 → 但模型仍需evaluation:

处理方法:
  1. 去污染 → 从训练集删除benchmark题 → 简单但可能删太多!
  2. → 创建新benchmark → 不在训练集 → 但需要新数据!
  3. → → 用hold-out set → 训练不看 → 但需要提前规划!
  4. → → → 动态benchmark → 定期更新 → 防止污染!

  → → → → → 最佳实践: 去重+去污染+hold-out+定期更新 → 全链路!

### 6.4 训练集污染 vs 数据投毒(OWASP #3)

训练集污染(evaluation):
  → 无意 → 训练集包含benchmark → 虚高 → 误导!

数据投毒(恶意攻击):
  → 有意 → 攻击者注入恶意数据 → 模型偏见/有害 → OWASP #3!
  → → → 方法: 注入偏见数据 → 注入后门 → 注入误导 → 操控模型!
  → → → → → 防御: 数据验证+来源审计+异常检测 → Layer 1 pipeline!
  → → → → → → → 关联: ai-safety-guardrails-production-deep-dive.md(OWASP #3)

→ → → → → → → → 区别: 污染=无意(evaluation虚高) → 投毒=恶意(操控模型) → 都需检测!
```

## 7. Synthetic Data — 合成数据生成

```
### 7.1 Self-Instruct — 模型自己生成训练数据

Self-Instruct(Wang et al. 2023):
  → 用LLM生成instruction→input→output → 自我教学 → 不需人工标注!

流程:
  Step 1: 175个seed任务 → 人工写 → 起点!
  Step 2: LLM生成新instruction → 从seed → 稍微变化 → 新任务!
  Step 3: LLM生成input+output → 给instruction → 生成完整数据!
  Step 4: Filtering → 去除重复+低质量 → 保留高质量!
  → → → → → 结果: 52K指令 → 82K数据 → 训练 → Alpaca!

效果:
  → Alpaca 7B + Self-Instruct → 接近GPT-3.5 → 52K数据 → 低成本!
  → → → 但: 质量不如人工 → 模型偏窄 → 需持续改进!

### 7.2 Self-Play with Verification — RL自我博弈

比Self-Instruct更强 → RL自我博弈 → 验证 → 高质量合成数据!

数学推理(R1/O1模式):
  → 模型生成推理过程 → 验证答案 → 正确→保留 → 错误→丢弃!
  → → → 验证: 数学=答案匹配 → 代码=运行测试 → 严格 → 高质量!
  → → → → → RL: 用正确推理训练 → GRPO → 自我提升 → 闭环!
  → → → → → → → 效果: R1-Zero → 纯RL → 无人工 → 数学接近O1 → 自我博弈成功!

代码生成:
  → 模型生成代码 → 运行+测试 → pass→保留 → fail→分析错误!
  → → → → → 验证: 单元测试 → 多测试 → pass rate → 质量保证!
  → → → → → → → 效果: AlphaCode → 自我博弈 → 竞赛级代码 → 无人工!

### 7.3 Model Collapse风险 — 合成数据的陷阱

合成数据训练合成数据 → 模型退化 → collapse → 必须混合真实数据!

问题:
  → 纯合成 → 模型丢失分布尾部 → 聚焦高频 → 低频消失 → 退化!
  → → → 例: 生成"猫"多 → "稀有物种"消失 → 多样性丧失!
  → → → → → 多代合成 → 逐代退化 → 最终collapse → 模型无用!

防御:
  → 合成+真实混合 → 保持多样性 → 不纯合成!
  → → → 验证 → 数学/代码 → 确保正确 → 高质量合成!
  → → → → → 人工抽查 → 确保质量 → 不完全依赖自动!

→ → → → → 结论: 合成数据有用 → 但不能纯合成 → 必须混合真实 → 否则collapse!
```

## 8. Data Wall — 人类数据耗尽

```
### 8.1 Data Wall时间线

人类高质量文本有限 → 2024-2026可能耗尽 → Data Wall!

估计:
  → 高质量英语文本 → ≈10-20T tokens → 已用大部分!
  → → Common Crawl → 250B页 → 但高质量<5% → ≈1.3TB高质量(FineWeb)!
  → → → 其他源 → ArXiv+GitHub+书籍 → ≈0.5TB → 有限!
  → → → → → 总计高质量 → ≈2TB → ≈200-500B tokens → 已接近Chinchilla最优!

  → → → → → → 大模型需求:
    → 7B → 140B tokens → 可满足 → 小模型OK!
    → → 70B → 1.4T tokens → 环现有数据 → 可满足!
    → → → 405B → 8T tokens → 需更多 → 可能不够!

### 8.2 突破Data Wall的策略

1. 合成数据 + RL自我博弈:
   → Self-Instruct+Self-Play → 模型自生成 → 无限数据!
   → → → 但: collapse风险 → 需验证+混合 → 不能纯合成!

2. 多语言数据:
   → 英语耗尽 → 其他语言 → 中文+法语+... → 新数据源!
   → → → 但: 质量参差 → 翻译损失 → 间接!

3. 专业领域数据:
   → 医学+法律+金融 → 专业数据 → 高质量但窄 → 新数据!
   → → → 但: 版权+隐私 → 获取困难 → 有限!

4. RL自我博弈(最优路径):
   → AlphaProof/AlphaGeometry → RL自我博弈 → 无需人类数据 → 无限!
   → → → → → 数学/代码 → 验证器 → 正确=好数据 → 自我进化!
   → → → → → → → 关键: 有验证器的领域 → RL自我博弈 → 无限数据 → 突破Data Wall!

→ → → → → → → → 结论: Data Wall → RL自我博弈是突破路径 → 有验证器的领域→无限数据!
```

## 9. RTX 4090 数据策略 — 小模型最优

```
RTX 4090数据策略(24GB HBM限制 → 小模型 → 高质量数据):

训练策略:
  → 7B模型 → LoRA微调 → 高质量子集 → 最优!
  → → → 数据: 不需要全量 → 只需要目标领域 → 5-50GB → 高质量子集!
  → → → → → 例: 代码微调 → GitHub高质量 → 10-20GB → LoRA → 高效!

数据准备:
  → MinHash LSH → 本地可行 → CPU即可 → RTX 4090不需要GPU!
  → → → → → Quality filtering → 小分类器 → CPU → 不需GPU!
  → → → → → → → DoReMi → 需参考模型 → RTX 4090训练ref → 可行!

合成数据:
  → Self-Instruct → 用7B模型 → RTX 4090 → 生成 → 可行!
  → → → → → 验证 → 代码=运行 → 数学=答案 → 可行!
  → → → → → → → GRPO → RTX 4090 → 7B+LoRA → RL微调 → 可行!

→ → → → → → → → RTX 4090最优: LoRA微调+高质量子集+Self-Instruct合成+GRPO RL → 小模型高效路径!
```

## 10. 核心规律

```
Data Pipeline核心:

  数据质量>模型架构 → 质量×3>数量 → curated 5TB≈raw 30TB → 3x效率!
  → → FineWeb: 250B→1.3TB → 8B=70.5% MMLU → vs 250B原始=40-50% → 差距20+!
  → → → DCLM: 同架构不同数据 → 46%→64.8% → 18.8% → 数据决定性!

  MinHash LSH去重 → Jaccard阈值=0.7 → 去除30-40%近似重复 → 最关键步骤!
  → → → 不去重=灾难 → 模型记忆重复 → evaluation污染 → compute浪费!

  Quality Filtering → heuristic+learned → 过滤95% → 5TB高质量 → 多样性保留!
  → → → 过度=多样性丧失 → 不足=噪声 → 95%最平衡!

  DoReMi → minimax excess loss → 最优domain比例 → 5-10%性能提升 → 不改架构!
  → → → 最弱domain提升10-20% → 最公平 → 数据自适应!

  Data Contamination → 13-gram overlap → 检测90%+ → 去污染后真实性能 → 5-10%虚高修正!
  → → → 区别: 污染(无意)vs投毒(恶意OWASP#3) → 都需检测!

  Synthetic Data → Self-Instruct+RL自我博弈 → 无限数据 → 但不能纯合成→collapse!
  → → → 验证器领域(数学/代码)→RL→无限 → 突破Data Wall!

  Data Wall → 2024-2026人类高质量文本耗尽 → RL自我博弈是突破路径!
  → → → 有验证器→无限 → 无验证器→collapse风险 → 需混合真实!

  RTX 4090数据策略:
    → LoRA微调+高质量子集+Self-Instruct+GRPO → 小模型高效路径!
    → → MinHash+quality filtering → CPU → 不需GPU!
    → → → → 数据准备本地可行 → 训练GPU可行 → 全链路!

  知识Gap修复:
    → Data Pipeline从★★(2/5) → ★★★★(4/5) → MinHash+Quality+DoReMi+Contamination+Synthetic+Data Wall → 全面!
    → → → → 但仍需实践 → GPU可用时 → Datatrove pipeline → 实际数据清洗 → 实测!
```

## 参考文献

```
1. 数据pipeline框架:
   - FineWeb: huggingface.co/datasets/HuggingFaceFW/fineweb
   - DCLM: datacomp.ai
   - Nemotron-CC: huggingface.co/datasets/nvidia/nemotron-cc
   - Datatrove: github.com/huggingface/datatrove

2. 去重算法:
   - MinHash: Broder 1997, "On the resemblance and containment of documents"
   - LSH: Indyk & Motwani 1998, "Approximate nearest neighbors"
   - ccdedup: github.com/mozilla/ccdedup (Mozilla MinHash实现)

3. 质量过滤:
   - fastText: github.com/facebookresearch/fastText
   - FineWeb quality filter: EPFL 2024 technical report
   - DCLM CosmoQualityClassifier: datacomp.ai

4. 数据混合:
   - DoReMi: Xie et al. 2023, "DoReMi: Optimizing Data Mixtures by Reweighting"
   - Chinchilla: Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"

5. 数据污染:
   - GPT-4 contamination: OpenAI 2023 technical report
   - n-gram overlap: Carlini et al. 2023, "Quantifying and Mitigating Data Contamination"

6. 合成数据:
   - Self-Instruct: Wang et al. 2023, "Self-Instruct: Aligning Language Models"
   - AlphaProof/AlphaGeometry: DeepMind 2024
   - Model Collapse: Shumailov et al. 2023, "The Curse of Recursion"

7. Data Wall:
   - Epoch AI: "Will we run out of data?" 2024 analysis
   - Villalobos et al. 2024, "Running out of data"

8. 我们的笔记:
   - scaling-laws-deep-dive.md → Chinchilla+数据质量+Data Wall
   - ai-safety-guardrails-production-deep-dive.md → OWASP #3数据投毒
   - ai-expert-knowledge-map-gap-analysis.md → 数据gap评估
