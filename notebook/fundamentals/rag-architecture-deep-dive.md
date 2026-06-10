# RAG Architecture Deep Dive

> 2026-06-08 | RAG=检索增强生成=最practical的知识更新方式! 从简单pipeline→多级检索→Agentic RAG→Graph RAG 4代演进, 核心挑战=检索质量+chunking+embedding+成本
> 关联: agent-systems-deep-dive.md, llm-evaluation-benchmarking-deep-dive.md, continual-learning-deep-dive.md

## 0. 核心定律: RAG = 外挂知识库 = 零遗忘+无限plasticity

```
RAG公式:
  → Answer = LLM(Q + Context(retrieved from DB))
  → → Query → 检索相关文档 → 注入LLM context → 生成答案

vs 模型内化:
  → RAG: 知识external → 检索 → 不修改参数 → 零遗忘!
  → Fine-tune: 知识internal → 训练 → 修改参数 → 有遗忘风险!
  → → RAG=100% stability, Fine-tune=有plasticity但stability风险
  → → → 最佳: RAG(新知识/事实性) + Fine-tune(推理模式/技能) → hybrid!

RAG挑战:
  → 检索质量: 找到最相关文档? → embedding质量+chunking策略+检索算法!
  → Context window: 检索文档占context → KV cache增大 → 并发降低!
  → → → 7B模型4K context → 检索3K → 只剩1K给对话 → 限制!
  → → → 128K context → 检索100K → 28K对话 → OK!
  → 成本: 检索+推理 → embedding计算+向量搜索+LLM推理 → 3x成本!
  → → → 但不修改模型 → 不需重训练 → 知识更新成本极低 → RAG最practical!

RTX 4090 RAG serving:
  → 7B INT4+INT8KV+FlashInfer → B=118 → 4,791 tok/s → 推理够快
  → ChromaDB本地 → 检索<10ms → 总延迟≈推理50ms+检索10ms=60ms → 可接受!
  → → → RTX 4090 = RAG serving最优平台! (本地+低成本+高吞吐)
```

## 1. 四代RAG架构演进

### 1.1 Naive RAG — 最简单pipeline
```
流程: Query → Embedding → Vector DB检索 → Top-K → 注入Context → LLM生成

问题:
  → 检索不准: embedding可能不匹配 → 返回不相关文档 → LLM生成错误答案!
  → → → "检索噪声" → 比没有检索更糟! → LLM被无关信息误导!
  → chunking不当: 固定512token → 可能切断段落 → 语义不完整!
  → → → 答案跨chunk → 检索只返回一部分 → 信息不完整!

改进: Better embedding → fine-tuned embedding → 领域特定 → 更准!
  → → 但仍依赖embedding质量 → 不根本解决问题!
```

### 1.2 Advanced RAG — 多级检索+重排序
```
改进点:
  → Query Transformation: 原始query可能模糊 → 重写→更清晰→检索更准!
    → → Rewrite: "什么是MoE?" → "什么是Mixture of Experts in LLM serving?"
    → → → HyDE: 先让LLM生成假设答案 → 用假设答案的embedding检索 → 更准!
    → → → → Query Decomposition: 复杂问题 → 分解成子问题 → 分别检索 → 合并!

  → Reranking: 检索Top-100 → Cross-encoder重排序 → Top-5最相关!
    → → Bi-encoder(embedding): 快但粗 → 检索阶段用 → 100候选
    → → Cross-encoder: 慢但精 → 重排序阶段用 → 5精选
    → → → → 两级检索 = 快速粗筛 + 精确精选 → 速度+质量平衡!

  → Chunking改进:
    → → Semantic chunking: 按语义边界切 → 不切断段落 → 完整语义!
    → → → Parent-child: 小chunk(child)检索 → 返回大chunk(parent)作为context!
    → → → → → 检索粒度细(小chunk更易匹配) → context粒度粗(大chunk更完整)!

RTX 4090 Reranking:
  → Cross-encoder(0.5B模型) → 3,5GB INT4 → 24GB内共存3个reranker模型!
  → → → 或7B模型同时做retrieval+reranking+generation → 1模型3用!
  → → → → Reranking overhead: 5ms×100docs=500ms → 可接受!
```

### 1.3 Agentic RAG — 自主检索Agent
```
核心: LLM自主决定是否需要检索+检索什么+是否需要再次检索

流程:
  → Step 1: LLM评估 → "我需要检索吗?" → 如果已知 → 直接回答!
  → Step 2: 如果需要 → 构造query → 检索 → 评估检索结果质量!
  → Step 3: 如果不够 → 改进query → 再次检索 → 直到满意!
  → → → Self-Reflective RAG → LLM自己控制检索循环 → 自适应!

与Agent Systems联系:
  → Agentic RAG = Tool-use Agent → 检索=tool → LLM决定何时用tool!
  → → → 与之前的Agent笔记完全一致! → 检索tool + 自我评估tool + 生成tool
  → → → → LangGraph实现: retrieval_node → evaluate_node → generate_node → loop!
  → → → → → RTX 4090: 7B INT4 Agent → 检索+推理 → 500ms → 可接受!

Self-RAG (2025):
  → 模型自反思 → 3种decision:
    → → Retrieval needed? → [Yes: 检索]/[No: 直接回答]
    → → → Is retrieval relevant? → [Relevant: 用]/[Irrelevant: 丢弃+再检索]
    → → → → Is answer supported? → [Supported: 输出]/[Not supported: 修正]
  → → → → → Self-RAG = 最自适应 → 但需要模型足够smart → 7B勉强 → 70B更好!

RTX 4090 Agentic RAG:
  → 7B INT4 → 每步50ms → 3步(self-reflect) → 150ms + 检索10ms = 160ms → 快!
  → → → 70B INT4 → 需3-4GPU(TP=4) → 更smart → 但成本↑4x → 7B性价比更好!
```

### 1.4 Graph RAG — 知识图谱+向量检索
```
核心: 向量检索(语义相似) + 知识图谱(结构关系) → 双重信息!

知识图谱优势:
  → 关系推理: A→B→C → 检索不只找相似 → 还找关联! → 更全面!
  → → → "MoE→DeepSeek→EP" → 检索不只是MoE相关 → 还能找到DeepSeek和EP!
  → → → → → Graph traversal比embedding检索更结构化 → 适合关系密集领域!

Graph RAG流程:
  → Step 1: Embedding检索 → Top-10语义相似文档
  → Step 2: Graph traversal → 从Top-10 → 找相关节点 → 扩展context!
  → Step 3: 合并 → 语义+结构 → 完整context → LLM生成!

与MoE Serving联系:
  → MoE知识图谱: expert→layer→model→training→serving → 多层关系!
  → → → 检索"MoE serving" → 不仅返回MoE文档 → 还返回EP/A2A/FusedMoE!
  → → → → → Graph RAG = AI Infra知识检索的最有效方式!

RTX 4090 Graph RAG:
  → Neo4j(Neo4j Desktop) → 本地知识图谱 → 图查询<5ms → 极快!
  → → → ChromaDB + Neo4j → 混合检索 → 10ms + 5ms = 15ms → 近零overhead!
```

## 2. Embedding与Chunking策略

```
Embedding选择:
  → General: OpenAI text-embedding-3-large → 3072维 → 通用但不够精准
  → → Domain-specific: fine-tune embedding → 领域适配 → 更准!
  → → → Matryoshka: 自适应维度 → 64/128/256/512/1024 → 灵活精度!
  → → → → 低维度(64) → 快但粗 → 高维度(3072) → 慢但精 → 按需求选择!

Chunking策略对比:
  → Fixed-size: 每512token → 简单 → 但可能切断语义 → 最基础!
  → → Semantic: 按段落/标题/语义边界 → 更完整 → 但chunk大小不均 → 累引复杂!
  → → → Sentence-level: 每句1chunk → 检索极精准 → 但context太碎片 → 需parent-child!
  → → → → Late chunking: 先全文档embedding → 再chunk → 保留全局语义 → 最新!

  Parent-Child策略:
    → Child(chunk_size=128): 检索用 → 精准匹配 → Top-20 child
    → → Parent(chunk_size=512): context用 → 完整段落 → 返回parent docs
    → → → → → 检索精准 + context完整 → 最佳trade-off!

RTX 4090 Embedding计算:
  → Embedding模型(0.5B) → INT4 → 0.175GB → 24GB可同时跑7B+0.5B embedding!
  → → → ChromaDB: embedding计算→向量存储→检索 → 全本地 → 无云成本!
  → → → → → 100万文档 → embedding → 0.5B INT4 → ~30分钟 → 一次完成!
```

## 3. RAG评估 — RAGAS框架

```
RAGAS (Retrieval Augmented Generation Assessment):
  → 4维度评估 → 每个维度独立 → 全面!

  1. Context Precision: 检索文档中相关内容占比 → 有多少噪声?
     → → High precision → 大部分检索相关 → LLM不被误导!
     → → Low precision → 大量噪声 → LLM可能被误导 → 比不检索更糟!

  2. Context Relevance: 检索文档是否包含答案所需信息 → 有没有遗漏?
     → → High relevance → 信息完整 → LLM能准确回答!
     → → Low relevance → 信息不完整 → LLM可能猜测 → 不可靠!

  3. Faithfulness: 生成的答案是否基于检索文档 → 有没有"幻觉"?
     → → High faithfulness → 所有答案claim都能在检索文档找到出处!
     → → Low faithfulness → LLM编造信息 → 不是来自检索 → 幻觉!
     → → → → 这是RAG最重要的指标! → RAG的意义=减少幻觉 → faithfulness必须高!

  4. Answer Relevance: 生成的答案是否回答了问题 → 有没有偏题?
     → → High answer relevance → 直接回答问题 → on-topic!
     → → Low answer relevance → 偏题 → 回答了不相关内容 → off-topic!

与LLM Evaluation联系:
  → RAGAS = HELM for RAG → 专门评估RAG pipeline → 不是评估LLM本身!
  → → → RAG pipeline = 检索+推理+评估 → 3个独立组件 → 各有不同指标!
```

## 4. Core Laws — RAG核心定律

```
1. Retrieval-Quality Law: RAG答案质量 ∝ 检索质量
   → → 检索不准 → 答案必错 → "garbage in, garbage out"!
   → → → Reranking是关键: Top-100粗筛 → Top-5精选 → 精度↑50%!

2. Context-Limit Law: 可用context ∝ total_context - retrieval_context
   → → 检索占context → 对话空间减少 → 长对话受限!
   → → → INT8 KV → context 2x → 检索+对话都更大 → RAG效果更好!
   → → → → StreamingLLM → 固定KV → 无限对话 → RAG检索不影响对话长度!

3. Cost-Speed Law: RAG成本 = 检索成本 + 推理成本 + Embedding成本
   → → Embedding: 一次 → 存储 → 近零(已计算)
   → → 检索: Vector search → <10ms → 近零
   → → 推理: LLM decode → 50ms → 主成本
   → → → → RAG总成本 ≈ LLM推理成本 + 10ms检索 → 检索近零!

4. Stability-Plasticity RAG Law: RAG stability=100%, plasticity=∞
   → → 不修改参数 → 遗忘=0 → stability完美!
   → → 检索库可随时更新 → 新知识即时可用 → plasticity无限!
   → → → 但! plasticity不是"内化" → 只是"查找" → 每次都要检索!
   → → → → → 模型不真正"学会"新知识 → 只能"找到"新知识 → 速度受限于检索!

5. Chunk-Tradeoff Law: 检索精度 ∝ 1/chunk_size, Context完整性 ∝ chunk_size
   → → 小chunk → 检索精准 → 但context不完整 → 需parent-child!
   → → 大chunk → context完整 → 但检索不精准 → 噪声多!
   → → → Parent-child = 解决trade-off → 小检索+大context → 最佳!
```

## 关键论文与参考

```
- RAG (Lewis et al., 2020): 检索增强生成 → 基础架构
- HyDE (Gao et al., 2023): 假设答案检索 → 更准的query representation
- Self-RAG (Asai et al., 2024): 模型自反思 → 是否需要检索+评估结果
- Graph RAG (Microsoft, 2024): 知识图谱+向量 → 结构化检索
- RAGAS (Es et al., 2024): RAG评估框架 → 4维度 → faithfulness最重要!
- ColBERT (Khattab & Zaharia, 2020): Late interaction → 多向量embedding
- Matryoshka Embeddings (2024): 自适应维度 → 灵活精度/速度
- Late Chunking (2025): 先全文档embedding → 再chunk → 保留全局语义

Sources:
- [Self-RAG](https://arxiv.org/abs/2310.05585)
- [Graph RAG](https://microsoft.github.io/graphrag/)
- [RAGAS](https://arxiv.org/abs/2402.05585)
- [HyDE](https://arxiv.org/abs/2212.09541)
- [ColBERT](https://arxiv.org/abs/2004.12832)