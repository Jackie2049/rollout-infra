# Data Pipeline & Curation — 数据质量 > 模型架构

> 2026-06-08 | "数据质量>模型架构" → Chinchilla验证数据量关键, 但质量决定上限
> 关键: 数据是训练最关键因素 → 去重+质量过滤+混合策略+污染检测

## 1. 为什么数据质量 > 模型架构

```
Chinchilla验证:
  → N_opt∝C^0.50, D_opt∝C^0.50 → 模型和数据同等重要!
  → 70B+1.4T > 280B+0.3T → 更小模型+更多数据胜!

但数据不仅仅是"量":
  → 低质量数据 → 训练噪声 → 模型学习垃圾 → loss高
  → 高质量数据 → 有效信号 → 模型学习有用 → loss低

Phi-series证明 (Microsoft 2023-2024):
  → Phi-1 (1.3B): "Textbooks Are All You Need" → 1.3B超越50B!
  → Phi-2 (2.7B): 用高质量教科书级数据 → 2.7B超越25B
  → Phi-3 (3.8B): 高质量+多语言 → 3.8B接近7B性能
  → 关键: 不是参数量 → 是数据质量决定性能上限!

数据质量层级:
  1. 数量(Chinchilla): D_opt∝C^0.50 → 必须有足够数据
  2. 质量(Phi): 高质量>>低质量 → 1.3B超越50B!
  3. 多样性: 不同领域 → 不同推理模式 → 泛化能力
  4. 去重: 重复数据=浪费 → 真实有效数据量=去重后量
  5. 污染: benchmark数据在训练集 → 评估失效 → 必须检测

→ 数据pipeline决定模型上限 → 模型架构只是效率优化!
```

## 2. 数据清洗与去重

```
去重方法层级:

1. 精确去重 (Exact Deduplication)
  → 删除完全相同的文档 → 最简单
  → 方法: hash-based → MD5/SHA256 → O(N)时间
  → 效果: 删除5-15%重复 → 但近似重复更多!

2. 近似去重 (Near Deduplication)
  → 删除"几乎相同"的文档 → 更重要!
  → MinHash + LSH (Locality-Sensitive Hashing):
    → 将文档分成k-grams → 计算MinHash签名 → LSH分桶
    → 相似度>Jaccard阈值 → 标记为重复 → 删除
    → 参数: num_hash=128, bands=32, threshold=0.7
    → 效果: 删除30-50%近似重复! → The Pile去重后减少50%

  →Suffix Array去重:
    → 找到长重复子串 → 精确删除重复段
    → 适合段落级去重 → 比MinHash更精确但更慢

  → 实际数据集去重率:
    The Pile: 30%去重 → 从825GB到575GB
    FineWeb: 50%去重 → 从15TB到7.5TB → 更严格!
    RedPajama: 25%去重 → URL+内容双去重

数学: 有效数据量 = 原始数据量 × (1-去重率)
  → 原始100B tokens → 去重率30% → 有效70B tokens → 30%浪费!
  → 去重后70B有效tokens → 比原始100B更有价值(无重复噪声)
```

## 3. 质量过滤

```
质量过滤方法:

1. 困惑度过滤 (Perplexity Filtering)
  → 用参考模型计算perplexity → 高PPL=低质量 → 删除
  → 参考: GPT-2 Small → 计算PPL → 删除PPL>阈值的文档
  → 效果: 删除垃圾网页 → 但可能删除方言/非标准文本
  → 阈值: 通常PPL>200 → 删除 → 但需调优!

2. 分类器过滤 (Classifier Filtering)
  → 训练质量分类器 → 高质量/低质量 → 删除低质量
  → 方法: fasttext classifier → 训练在维基百科(高) vs 随机网页(低)
  → CCNet (Common Crawl Net): 用KenLM分类 → 高质量选维基级
  → 效果: FineWeb用CCNet → 从1PB → 过滤后7.5TB → 99%删除!

3. 启发式过滤 (Heuristic Filtering)
  → 简单规则 → 删除明显低质量
  → 规则:
    → 文档太短(<50 words) → 删除
    → 文档太长(>100K words) → 删除(可能是垃圾)
    → 重复行>30% → 删除(模板网页)
    → 特殊字符>20% → 删除(乱码)
    → "lorem ipsum" → 删除(占位符)
  → 效果: 删除5-10% → 最简单 → 第一层过滤

4. 语言识别过滤
  → fasttext lid模型 → 识别语言 → 只保留目标语言
  → 多语言LLM → 需要多种语言 → 但要平衡比例
  → 效果: 删除非目标语言 → 减少噪声

质量过滤pipeline:
  Heuristic(5-10%) → Language(5-15%) → Perplexity(20-40%) → Classifier(50-99%)
  → 总过滤率: 70-99%! → 从1PB原始 → 到7.5TB训练数据
  → 说明: 网络数据99%是垃圾 → 需要严格过滤!
```

## 4. 数据混合策略

```
数据混合 = 不同领域数据的比例 → 决定模型能力分布

关键发现 (DoReMi, 2023):
  → 最优比例不是均匀 → 而是按领域难度和重要性分配!
  → DoReMi: 用小模型(proxy)估计各领域难度 → 加权混合
  → → 难领域多数据 → 简单领域少数据 → 更高效!

实际混合比例:
  LLaMA-1 (7B, 1T tokens):
    → CommonCrawl: 67% → 最大的来源
    → C4: 15% → 高质量网页
    → Github: 4.5% → 代码
    → Wikipedia: 4.5% → 知识
    → Books: 4.5% → 长文本理解
    → ArXiv: 2.5% → 学术
    → StackExchange: 2% → QA

  Phi-1 (1.3B, 高质量):
    → 教科书级代码+数学 → 不是随机网页!
    → 量少但质量极高 → 1.3B超越50B!

  数据混合策略:
    1. 降采样(Downsampling): 高质量数据源 → 不需要全部 → 采样足够
    2. 上采样(Upsampling): 低质量数据源 → 需要更多才能有效 → 重复训练
    3. 温度采样(Temperature): 数据源权重∝log(N_source)^α → α<1 → 降采样大源
    4. 课程学习(Curriculum): 从简单→困难 → 先训练简单数据 → 后训练难数据

  RTX 4090最优数据策略(7B模型):
    → Chinchilla: D_opt=140B tokens → 但实际用1T+ → 超最优→更好!
    → 质量优先: 1T高质量 > 10T低质量 → Phi验证!
    → 比例: CCrawl(60%) + 代码(20%) + 知识(20%) → 通用+代码+推理
```

## 5. 数据污染检测

```
数据污染 = benchmark数据出现在训练集 → 评估失效 → 高分不代表真实能力!

检测方法:

1. 字串匹配 (String Matching)
  → 在训练数据中搜索benchmark的prompt/response
  → n-gram overlap > 阈值 → 标记为污染
  → 效果: 简单但不够精确 → 改写后的污染可能漏检

2. N-gram Overlap
  → 计算benchmark和训练数据的n-gram重叠率
  → 重叠>50% → 可疑污染 → 需要进一步验证
  → GPT-4污染检测: 8-gram overlap → 检测率>95%

3. Membership Inference
  → 训练loss模型 → 如果benchmark样本loss异常低 → 可能是训练数据!
  → 方法: 训练一个小模型 → 检测loss分布 → 污染样本loss<阈值

4. Perplexity-based Detection
  → 模型对benchmark的PPL异常低 → 说明模型"见过"→ 污染!
  → → clean PPL=10 → contaminated PPL=2 → 5x差距!

已知污染案例:
  → GPT-3训练数据 → 包含部分benchmark → 评估偏高
  → LLaMA-1 → 检测到少量污染 → 但影响有限
  → 2024年 → 更多模型被检测到污染 → 需要严格检测!

数据污染清除:
  → 从训练数据删除benchmark → 重新训练 → 真实评估
  → 或: 在训练前就排除 → contamination list → 搜索排除

→ 关键: 数据污染 = 评估的最大威胁 → 不检测 = 评估不可信!
```

## 6. 数据Pipeline架构

```
End-to-end数据Pipeline:

Step 1: 数据收集 (Data Collection)
  → Common Crawl → 1PB+ → 每月更新
  → GitHub → 代码数据
  → Wikipedia → 知识数据
  → Books → 长文本
  → ArXiv → 学术

Step 2: 预处理 (Preprocessing)
  → HTML→text提取 → trafilatura → readability
  → 编码标准化 → UTF-8
  → 格式清理 → 删除标记 → 去模板

Step 3: 过滤 (Filtering)
  → 启发式 → 语言 → 困惑度 → 分类器
  → 99%删除 → 从1PB → 到7.5TB

Step 4: 去重 (Deduplication)
  → 精确去重 → 近似去重(MinHash+LSH)
  → 30-50%删除 → 从7.5TB → 到5TB

Step 5: 污染检测 (Contamination Detection)
  → benchmark n-gram matching → 删除污染数据
  → Membership inference → 验证

Step 6: 混合 (Mixing)
  → 数据源比例 → 降采样/上采样 → 课程学习
  → 1T tokens训练数据

Step 7: Tokenization
  → BPE训练 → vocab size决定 → 32K-128K
  → 我们已经深读tokenizer! → vocab决定推理成本

Step 8: 格式转换
  → Parquet → JSON → 适合训练框架读取
  → Streaming dataset → 不需要全部加载到内存

→ 关键瓶颈: Step 3(过滤)和Step 4(去重) → 99%+50% → 大量计算!
→ 数据pipeline是训练最耗时的部分 → 不是模型训练本身!
```

## 7. 核心规律

```
Data Pipeline核心:

1. 数据质量 > 模型架构
  → Phi-1(1.3B)超越50B → 教科书级数据=决定性因素
  → 不是"参数量大=好" → 是"数据质量高=好"
  → Chinchilla验证量重要 → Phi验证质量更重要!

2. 去重是必需的
  → 30-50%数据是近似重复 → 不去重=浪费计算
  → MinHash+LSH → 标准方法 → 必须实现!
  → 有效数据量=原始×(1-去重率) → 实际比想象少!

3. 网络数据99%是垃圾
  → 1PB原始 → 7.5TB训练 → 99%删除!
  → 严格过滤是必需 → 不是可选 → Heuristic+PPL+Classifier三层

4. 混合比例决定能力分布
  → 不是均匀混合 → 而是按难度和重要性加权
  → DoReMi → 用小模型估计难度 → 自动化混合
  → 代码比例决定代码能力 → 数学比例决定数学能力

5. 数据污染=评估最大威胁
  → benchmark在训练集 → 评估失效 → "高分≠好"
  → 必须检测 → n-gram+PPL → 必须清除!

6. 数据pipeline是训练瓶颈
  → 不是GPU训练 → 而是数据准备 → 耗时最长!
  → 7B Chinchilla需要140B tokens → 但1T更好 → 数据量巨大
  → Pipeline自动化 → 工程化 → 才能规模化

RTX 4090数据实践:
  → 训练7B → 需要1T高质量tokens → 数据收集是主要工作
  → 本地: 用HF datasets下载 → FineWeb/EduFineWeb → 高质量预过滤
  → Chinchilla: 7B+1T → 近似最优 → 2T更好(但需要更多数据!)
  → 数据pipeline自动化 → tools/data_pipeline.py → 未来工具