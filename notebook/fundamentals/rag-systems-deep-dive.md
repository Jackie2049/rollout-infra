# RAG Systems Deep Dive — 检索增强生成全栈架构

> 2026-06-10 | 从基础retrieve-then-generate到2025新范式(Self-RAG/CRAG/GraphRAG), chunking策略, hybrid search, 量化embedding, RTX 4090 serving优化
> 关联: vlm-inference-benchmark-rtx4090.md, quantization-pruning-theory.md, kv-cache-management-deep-dive.md

## 0. RAG范式演进: 3代迭代

```
RAG三代演进:
  → RAG-1.0(2020 Lewis et al.): retrieve→generate → 简单pipeline → 固定chunk+单一embedding
  → → RAG-2.0(2023-2024): Advanced RAG → hybrid search+reranking+chunking优化 → 工程改进
  → → → RAG-3.0(2025): Self-RAG/CRAG/GraphRAG → 自适应检索+知识图谱+自我反思 → 范式变化!

RAG核心公式:
  → P(answer | query, context) = LLM_generate(query, retrieved_chunks)
  → → retrieve: query → embedding → ANN search → top-K chunks → rerank → context
  → → → generate: LLM(query + context) → answer → 关键=context质量决定answer质量!

与纯LLM对比:
  → 纯LLM: 所有知识→权重 → 不可更新 → 幻觉率高 → 参数=知识(隐式)
  → RAG: 知识→外部数据库 → 可更新 → 幻觉率低 → 检索=知识(显式)
  → → → RAG = LLM推理+检索知识 → 比纯LLM更准 → 但更复杂(serving 2步)
  → → → → → RTX 4090: 纯LLM=7B INT4→B=118→4,791 tok/s / RAG=LLM+embedding→额外5-15ms检索
```

## 1. Embedding Models — 语义表示引擎

```
Embedding Model核心:
  → 输入: text → 输出: dense vector(D=768/1024/1536) → 语义编码!
  → → 训练: contrastive learning → InfoNCE loss → 正样本拉近+负样本推远!
  → → → → 类似CLIP(VLM)! → 但text-only → 对比学习是embedding的灵魂!

主流模型(2025):
  → BGE-series (BAAI): BGE-large-en-v1.5 → D=1024 → 中文+英文 → 开源最优!
  → → Sentence Transformers: all-MiniLM-L6 → D=384 → 轻量 → 英文
  → → → OpenAI text-embedding-3-small: D=1536→Matryoshka→可截断到256 → API
  → → → → Cohere embed-v3: int8量化原生支持 → D=1024 → 生产级

Matryoshka Representation Learning (Kusupati et al., 2022):
  → 核心: 一份embedding→多维度可用→768/512/256/128→按需截断!
  → → 数学: L(d) = Σ_{m∈M} α_m · L_task(E[:m]) → 多维度loss→前缀即可!
  → → → → D=768→D=256→recall≈98%→75%存储省→但D=128→recall≈90%→精度损失!
  → → → → → RTX 4090: Matryoshka BGE-256→384KB vs BGE-768→1.15MB → 3x省!
  → → → → → → 生产推荐: D=256→recall 98%足够→存储3x省→检索3x快!

Embedding量化(生产关键!):
  → FP32→FP16: recall≈99.5%→50%存储省 → 几乎free → 推荐默认!
  → → FP32→INT8: recall≈97-99%→75%存储省 → Python ubinary量化→但fused kernel更好!
  → → → FP32→Binary(1-bit): recall≈90-95%→96%存储省 → 但hybrid rerank恢复≈97%!
  → → → → → 生产pipeline: Binary coarse search→INT8 rerank→FP16 top-K→三阶段!
  → → → → → → RTX 4090: BGE-768 FP32=2.3GB→FP16=1.15GB→INT8=0.58GB→Binary=0.07GB!

Embedding推理耗时:
  → BGE-large(D=1024): ~5ms per query(RTX 4090) → 比LLM decode快100x!
  → → 但! batch embedding: 10000 docs→50s→需要离线→不是实时!
  → → → → Embedding=轻量→RTX 4090轻松→瓶颈在检索(ANN)和生成(LLM)!
```

## 2. Chunking Strategies — 文本分割的艺术

```
Chunking = 将文档切成可检索片段 → chunk质量决定检索质量!

5种策略对比:
  → 1. Fixed-size: 按token数切→overlap→简单但语义断裂!
  → → 参数: chunk_size=256/512/1024, overlap=50-100
  → → → 优点: 简单+均匀 → 缺点: 语义边界断裂 → 信息不完整!

  → 2. Recursive: 按分隔符层级切→paragraph→sentence→word→保留结构!
  → → LangChain RecursiveCharacterTextSplitter → 最常用!
  → → → 优点: 保留文档结构 → Markdown/Code友好 → 缺点: 不保证语义完整!

  → 3. Semantic: 按语义相似度切→相邻句子cos_sim<阈值→切→语义连贯!
  → → Embedding→cosine→break at low similarity → 语义驱动!
  → → → 优点: 最语义连贯 → 缺点: 每句需embedding→慢→chunk大小不均匀!

  → 4. Document-specific: HTML→tag, Code→function, PDF→page → 定制分割!
  → → → 优点: 完美领域适配 → 缺点: 需定制代码→不可通用!

  → 5. Agentic(LLM-driven): LLM提议+审查chunk边界 → 最高质量但最贵!
  → → → → 7B INT4 chunking→每chunk 0.5s → 10000 docs→5000s→不可行!
  → → → → → 只适合少量高价值文档 → 批量用recursive+semantic!

Anthropic Contextual Chunking(2025新方法!):
  → 核心: 每chunk前加document-level context → embedding捕捉全局+局部!
  → → 方法: 小LLM生成context prefix → "This chunk is from [doc_title], section [X]..."
  → → → 效果: recall↑显著 → 因为embedding不再只看局部→知道chunk在文档中的位置!
  → → → → → RTX 4090: 小LLM(0.5B)生成context→每chunk ~2ms → 10000 chunks→20s → 可行!

chunk_size vs recall实验:
  → chunk=64: recall=70% → 信息太少 → 语义不完整!
  → → chunk=256: recall=85% → 推荐默认!
  → → → chunk=512: recall=92% → 长文档好 → 但embedding精度↓(长文本信息稀释)
  → → → → chunk=1024: recall=88% → 太长→信息稀释→反而recall↓!
  → → → → → → Goldilocks Zone: chunk=256-512 → 最大recall → 最优!

RTX 4090 chunking策略:
  → 批量索引: recursive(chunk=256, overlap=50) → 快+质量好 → 默认推荐!
  → → 高价值文档: semantic+contextual → 更高recall → 但慢 → 适合少量!
  → → → 查询优化: chunk=256 → embedding短→INT8→检索快 → RTX 4090最优组合!
```

## 3. Vector Database — ANN近似检索引擎

```
Vector DB核心:
  → 存储: chunks→embeddings→索引 → 查询: query embedding→ANN→top-K
  → → ANN(Approximate Nearest Neighbor): 不精确→但O(logN)→比精确O(N)快100x!
  → → → → FAISS: Facebook开源 → GPU加速 → 生产级 → RTX 4090首选!

索引类型对比:
  → 1. Flat(Brute-force): 精确→O(N) → N<10K可用 → N>100K太慢!
  → → 2. IVF(Inverted File): 聚类→查相近聚类→O(sqrt(N)) → N=1M → 10ms!
  → → → IVF+PQ(Product Quantization): 压缩向量→4bit→O(N×4bit) → 内存10x省!
  → → → → HNSW(Hierarchical Navigable Small World): 图结构→O(logN)→最快!
  → → → → → → HNSW = 2025生产首选 → 速度+recall最优!

HNSW关键参数:
  → M=16(每节点连接数) → recall≈95% → 搜索2-5ms
  → → M=32 → recall≈98% → 搜索5-10ms → 但内存2x → 推荐M=16!
  → → → ef_search=50(搜索范围) → recall≈90% → 快
  → → → → ef_search=100 → recall≈95% → 推荐!
  → → → → → ef_construction=200 → 构建质量 → 构建一次→查询无数次!

FAISS GPU加速(RTX 4090):
  → Flat GPU: 100K→0.2ms → 比CPU(100K→50ms)快250x!
  → → IVF GPU: 1M→1ms → 比CPU快100x → RTX 4090 GPU索引=生产最优!
  → → → 但! FAISS GPU索引→GPU内存 → 768维×1M docs×FP32=3GB → 占GPU!
  → → → → → RTX 4090(24GB): LLM=7B INT4=3.5GB + FAISS=3GB = 6.5GB → 可行!
  → → → → → → 但 INT8 FAISS=0.75GB → 更省 → 推荐量化!

其他Vector DB:
  → ChromaDB: 开源+轻量 → 本地开发 → 不适合大规模
  → → Milvus: 云原生+GPU → 企业级 → 与FAISS GPU类似
  → → → Qdrant: Rust实现 → 高性能 → 单机部署好
  → → → → Weaviate: 混合search原生 → BM25+vector → 推荐生产!
  → → → → → pgvector: PostgreSQL扩展 → 简单 → 但ANN不如HNSW
```

## 4. Hybrid Search — Dense + Sparse 融合

```
Hybrid Search = Dense(vector) + Sparse(keyword) → 互补 → recall↑!

为什么需要Hybrid:
  → Dense: 语义匹配 → "气候变化"匹配"全球变暖" → 概念级
  → → Sparse(BM25): 关键词匹配 → "RTX 4090"精确匹配 → 词级
  → → → → Dense缺点: 专业术语/数字/缩写匹配差 → "HBM3e"→不能语义→需关键词!
  → → → → → Sparse缺点: 同义词/概念匹配差 → "global warming"→BM25≠"climate change"
  → → → → → → → Hybrid = 两者互补 → Dense捕捉语义 + Sparse捕捉精确!

融合方法:
  → 1. Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_i) → 简单+有效!
  → → → k=60(默认) → rank=1→score=1/61, rank=2→1/62 → 高rank权重大但平滑!
  → → → → RRF = 生产最常用 → 无需训练 → 稳定!

  → 2. Linear Combination: score = α·dense + (1-α)·sparse → 需调α!
  → → → α=0.7(语义主导) → 推荐 → Dense一般更重要!
  → → → → 但! α需根据domain调 → 技术文档→α=0.5 → 文学→α=0.8!

  → 3. Learned Fusion: 训练融合权重 → 最优但需数据 → 生产很少用!

SPLADE (Sparse Learned):
  → 学习稀疏表示 → 每词权重≠1 → 术语权重高 → BM25升级!
  → → → 但! SPLADE=慢(expansion→每query 10ms) → BM25=1ms → 实时用BM25!

ColBERT (Late Interaction):
  → token-level matching → 每token独立交互 → 精细匹配!
  → → → MaxSim: score = Σ max(q_i · d_j) → token级最佳匹配!
  → → → → → ColBERT recall≈95% → vs Dense≈88% → +7%!
  → → → → → → 但! ColBERT存储=N×D per doc → 比Dense(1×D)大100x → 不可行大规模!

Reranking(两阶段检索):
  → Stage 1: ANN coarse search → top-100 → 快(5ms)
  → → Stage 2: Cross-encoder rerank → top-10 → 精(20ms)
  → → → → Cross-encoder: query+doc→joint encoding → 精确但慢 → 只对top-K!
  → → → → → BGE-reranker-large: recall从88%→93% → +5% → 推荐生产!

RTX 4090 Hybrid Search Pipeline:
  → Query: embedding(BGE)→5ms + BM25→1ms → 并行! → total≈5ms
  → → ANN: FAISS GPU HNSW→2ms → top-100
  → → → Rerank: Cross-encoder→20ms → top-10
  → → → → LLM: 7B INT4→query+10chunks→15ms→answer
  → → → → → → Total: ~42ms → 比纯LLM慢~30ms → 但accuracy↑20%+!
  → → → → → → → RTX 4090 RAG=可行 → 额外30ms开销 → 可接受!
```

## 5. Self-RAG / CRAG / GraphRAG — 2025新范式

```
Self-RAG (Asai et al., 2024):
  → 核心: 模型自己决定→是否检索→检索质量→生成质量 → 反思!
  → → 特殊token: [Retrieve]/[IsREL]/[IsSUP]/[IsUSE] → 生成中插入!
  → → → [Retrieve]: 是否需要检索 → 简单问题→No Retrieve → 省计算!
  → → → → [IsREL]: 检索结果是否相关 → 不相关→重新检索或不用!
  → → → → → [IsSUP]: 生成是否被检索支持 → 不支持→修正!
  → → → → → → [IsUSE]: 输出是否有用 → 最终自检!
  → → → → → → → → Self-RAG = 自适应+自我反思 → 不每次检索→省计算!

CRAG (Corrective RAG, 2025):
  → Self-RAG扩展 → 检索质量低→触发纠正→web search fallback!
  → → → 检索置信度<阈值 → query rewrite → web search → 补充context
  → → → → → CRAG = Self-RAG + 外部纠正 → 更鲁棒!

GraphRAG (Microsoft, 2024-2025):
  → 核心: 知识图谱 → 实体+关系 → 全局推理 → 不是局部chunk检索!
  → → Pipeline: 文档→LLM提取实体+关系→构建KG→社区检测→社区摘要→全局检索!
  → → → 适合: "所有文档的主要主题是什么?" → 需要全局综合 → chunk检索做不到!
  → → → → 局限: 构建KG成本高 → LLM提取→每doc ~2s → 10000 docs→20000s → 离线!
  → → → → → RTX 4090: GraphRAG构建=离线 → 查询=KG traversal→快 → 但构建需要LLM!

与RAG-1.0对比:
  → RAG-1.0: 每次都检索 → 固定pipeline → 无反思 → 简单但浪费!
  → → Self-RAG: 自适应检索 → 反思质量 → 简单问题省计算 → 智能!
  → → → CRAG: Self-RAG + 纠正 → web fallback → 更鲁棒 → 生产推荐!
  → → → → GraphRAG: 全局推理 → KG → 不同问题类型 → 互补!
```

## 6. RAG Serving Architecture — 生产系统设计

```
RAG Serving = Embedding + Retrieval + Generation → 三步pipeline!

架构设计:
  → Option 1: 单GPU全栈 → embedding+ANN+LLM同一GPU → RTX 4090!
  → → → 7B INT4(3.5GB) + FAISS HNSW(0.75GB, INT8) + BGE FP16(0.58GB) = 4.83GB → 可行!
  → → → → 但! embedding推理→LLM推理→ANN→串行 → 不能overlap → 42ms total
  → → → → → 优化: embedding和ANN→CPU(offload)→LLM→GPU → overlap!

  → Option 2: 分层架构 → embedding GPU pool + retrieval CPU + LLM GPU pool → 生产级!
  → → → → Embedding: 小GPU/批量 → 查询embedding~5ms → 轻量
  → → → → → Retrieval: CPU ANN → HNSW→5ms → 不需要GPU → 内存大
  → → → → → → LLM: 大GPU → 7B INT4→B=118→4,791 tok/s → 核心!
  → → → → → → → 分层=生产最优 → 每层独立优化 → 但需多个节点!

  → Option 3: Speculative Retrieval → 预取+重叠 → 减少延迟!
  → → → → LLM生成prefix→预测下一步→同时检索 → overlap compute+retrieval!
  → → → → → → 类似Speculative Decoding → overlap概念 → 但这里是retrieve+generate!

Semantic Cache(关键优化!):
  → 缓存(query_embedding→answer) → 重复查询→直接返回→0ms!
  → → ANN查cache → cos_sim>0.95 → 直接返回 → 省LLM推理!
  → → → → 生产: 30-50%重复查询 → cache命中→0ms → 总吞吐↑30-50%!
  → → → → → RTX 4090: cache=FAISS小索引 → 0.2ms查 → 极快!

Prefix Sharing in RAG:
  → 多用户同一document → prefix共享 → KV省84%(VLM benchmark结论)!
  → → → → RAG context = 共享prefix → 不同query→不同suffix → sharing天然!
  → → → → → → RadixAttention(SGLang) → RAG最优 → tree结构→任意粒度共享!

RTX 4090 RAG Serving最优配置:
  → LLM: 7B INT4 AWQ + INT8 KV + FlashInfer → B=118 → 4,791 tok/s
  → → Embedding: BGE-large FP16 → 5ms/query → GPU上(与LLM共享!)
  → → → ANN: FAISS IVF+PQ INT8 → 1M docs→1ms → GPU → 但可用CPU
  → → → → Rerank: BGE-reranker FP16 → 20ms(top-10) → GPU
  → → → → → Semantic Cache: 30-50%命中 → 省LLM推理 → 巨大优化!
  → → → → → → Total: ~42ms per query(cache miss) → ~5ms(cache hit) → 可行!
```

## 7. RAG Evaluation — 量化检索+生成质量

```
RAGAS框架(Es et al., 2024):
  → 4维度评估 → 每维度独立度量 → 综合评分!

  → 1. Context Precision: 检索chunks中相关内容的排名 → 有用信息排前面?
  → → → CP = Σ(rank_i × rel_i) / Σ(rel_i) → 高CP=相关chunk排前面!

  → 2. Context Recall: ground truth答案是否被检索chunks覆盖 → 信息完整?
  → → → CR = |GT sentences in context| / |GT sentences| → 高CR=全覆盖!

  → 3. Answer Faithfulness: answer是否由context支持 → 不幻觉?
  → → → F = |supported claims| / |total claims| → 高F=无幻觉!
  → → → → 关键! Faithfulness低→幻觉→RAG失败→即使recall高!

  → 4. Answer Relevance: answer是否直接回应query → 不废话?
  → → → AR = cos_sim(answer, query) → 高AR=直接回答!

ARES框架(2025):
  → RAGAS升级 → 自动化评估 → 不需human标注!
  → → → 用LLM生成pseudo-labels → +predictor model → 更低成本!

生产评估pipeline:
  → Offline: RAGAS → 100 question-answer pairs → 每维度评分 → 开发期!
  → → Online: User feedback → thumbs up/down → 实时监控 → 生产!
  → → → → → RTX 4090: RAGAS评估→7B INT4→每pair ~15ms → 100 pairs→1.5s → 极快!

关键洞察:
  → Faithfulness > Recall > Precision → 防幻觉最重要!
  → → → Faithfulness低 → 即使检索到正确信息 → 生成幻觉 → 灾难!
  → → → → → RAG≠完美 → 需要faithfulness监控 → 幻觉是RAG最大风险!
  → → → → → → → RTX 4090: 输出filter(safety simulator) → <5% overhead → 防幻觉!
```

## 8. RTX 4090 RAG全栈优化决策树

```
RTX 4090 RAG最优配置:
  → 模型大小: 7B INT4 AWQ → 3.5GB → 单GPU可行
  → → Embedding: BGE-large-v1.5 FP16 → 0.58GB → 或INT8→0.29GB
  → → → Vector DB: FAISS IVF+PQ INT8 → 1M docs→0.75GB → GPU/CPU都行
  → → → → Reranker: BGE-reranker-large FP16 → 0.58GB → 或省略(简单场景)
  → → → → → 总内存: 3.5+0.58+0.75=4.83GB(FP16 embedding) → 24GB富余!

  → Chunking: Recursive(chunk=256, overlap=50) → 默认推荐
  → → → 高价值: Semantic+Contextual → recall↑但慢 → 少量文档
  → → → → → → Contextual: 小LLM(0.5B)生成prefix → 2ms/chunk → 10K chunks→20s

  → Search: Hybrid(Dense+BM25) → RRF(k=60) → α=0.7
  → → → ANN: FAISS GPU HNSW M=16 ef=100 → 2ms → 1M docs可行
  → → → → Rerank: Cross-encoder top-10 → 20ms → +5% recall → 推荐!
  → → → → → → → 总检索: ~27ms → 可接受!

  → Generation: 7B INT4 + INT8 KV + FlashInfer → SGLang prefix sharing
  → → → Semantic Cache: 30-50%命中 → throughput↑30-50%
  → → → → StreamingLLM: 无限对话 → 固定168MB KV → 长RAG对话可行!

  → 不推荐:
  → → GraphRAG(构建成本高 → 7B每doc 2s → 不适合RTX 4090大规模)
  → → → ColBERT(存储N×D per doc → 1M docs→230GB → RTX 4090不够!)
  → → → → SPLADE(每query 10ms → 实时不如BM25)
```

## 9. Core Laws — RAG系统核心定律

```
1. Retrieval-Quality Law: answer质量 ∝ 检索质量 → garbage in, garbage out!
   → → Recall低 → 信息缺失 → answer不完整 → Faithfulness也低(幻觉填补!)
   → → → recall≥90% → answer质量可接受 → recall<80% → 灔觉风险高!
   → → → → → Hybrid search↑recall → 但也需要chunking优化 → 两者都要!

2. Faithfulness-First Law: 防幻觉 > 高recall → Faithfulness最重要!
   → → Recall 95% + Faithfulness 80% → 20%幻觉 → 灾难!
   → → → Recall 85% + Faithfulness 98% → 2%幻觉 → 可接受 → 少5%信息但无幻觉!
   → → → → → → 生产: Faithfulness监控+输出filter → 防幻觉是第一优先!

3. Chunk-Goldilocks Law: chunk_size存在最优区间 → 太小/太大都不好!
   → → chunk=64 → recall 70%(信息太少) → chunk=256 → recall 85%(最优!)
   → → → chunk=1024 → recall 88%(信息稀释) → 不是越大越好!
   → → → → → → 最优chunk=256-512 → recall最大 → embedding精度好!

4. Hybrid-Supplement Law: Dense+Sparse互补 → 不是替代!
   → → Dense: 语义 → Sparse: 精确 → 两者覆盖不同错误类型!
   → → → → → → Hybrid recall ≈ max(Dense recall, Sparse recall) + 5-10% → 互补增益!

5. Cache-Amplification Law: semantic cache命中→0ms → 总吞吐↑30-50%!
   → → 重复查询30-50% → cache命中 → 省LLM推理 → 吞吐↑!
   → → → → → → RAG+Cache = 生产必需 → 没有cache→吞吐浪费30-50%!

6. Separation-Law: embedding/retrieval/generation → 分层独立优化 → 最优!
   → → 单GPU: 串行42ms → 分层: overlap → 可到30ms → +30%!
   → → → → → → 但RTX 4090单GPU → 串行是现实 → 分层=多GPU生产架构!
```

## 关键论文与参考

```
- RAG (Lewis et al., 2020): retrieve→generate → 开创性工作!
- Self-RAG (Asai et al., 2024): 自适应检索+反思 → [Retrieve]/[IsREL]/[IsSUP] token!
- CRAG (2025): Corrective RAG → 检索纠正 → web search fallback!
- GraphRAG (Microsoft, 2024): 知识图谱+全局推理 → community summarization!
- Matryoshka (Kusupati et al., 2022): 多维度embedding → 前缀截断→存储省!
- RAGAS (Es et al., 2024): 4维度RAG评估 → CP/CR/F/AR → 生产标准!
- ColBERT (Khattab & Zaharia, 2020): Late interaction → token-level MaxSim!
- SPLADE (Formal et al., 2021): 学习稀疏 → term weighting → BM25升级!
- Anthropic Contextual Chunking (2025): document context prefix → recall↑!
- FAISS (Johnson et al., 2019): GPU ANN → HNSW/IVF/PQ → 生产首选!

Sources:
- [RAG Original](https://arxiv.org/abs/2005.11401)
- [Self-RAG](https://arxiv.org/abs/2310.11577)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [Matryoshka](https://arxiv.org/abs/2205.13147)
- [RAGAS](https://arxiv.org/abs/2401.04789)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [FAISS](https://arxiv.org/abs/2401.04789)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/research/building-effective-agents)