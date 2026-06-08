# Attention Sink & KV Cache Eviction Deep Dive: StreamingLLM→H2O→SnapKV→PyramidKV

> 2026-06-08 | Attention Sink=前4个token吸收 disproportionate attention → StreamingLLM保留sink+滑动窗口 → KV eviction从位置保留(H2O)到注意力模式(SnapKV)到层自适应(PyramidKV)
> 基于: StreamingLLM(Xiao 2023), H2O(Zhang 2024), SnapKV(Zang 2024), PyramidKV(2025), Infini-Attention(Google 2024), SnapKV/ClusterKV/GEAR/KIVI
> 关联: kv-cache-management-deep-dive.md, long-context-serving.md, flashinfer-attention-deep-dive.md

## 0. 核心定律: Attention Sink = Softmax归一化副产品 = 前4 token吸收~80% attention

```
Attention Sink现象:
  → 自回归LLM中 → attention大量集中在序列开头的几个token
  → → 无论这些token语义上是否重要 → 都被大量关注 → 称为"attention sink"
  → → → 这些sink token是attention的"锚点" → 稳定softmax计算!

为什么形成Sink?
  → Softmax归一化: softmax(z_i) = exp(z_i) / Σexp(z_j) → 所有概率必须归一化到1
  → → 当当前token对所有已有token的attention score都很低 → softmax仍然要分配概率!
  → → → 哪个token分到概率? → 最前面的token → 因为位置编码赋予它们"默认注意力"
  → → → → 或者: 初始token的KV向量 → 成为"溢出桶" → 接收"无处可去"的attention概率
  → → → → → 这不是语义需要 → 而是数学需要 → softmax必须归一化 → 必须有"接收者"

实测数据 (LLaMA-2 7B):
  → 前4个token: ~80%的attention概率 → 即使是<s>(beginning of sequence)这种无语义token!
  → 中间token: ~15%的attention → 真正的语义信息
  → 最近token: ~5%的attention → local attention pattern
  → → → 如果移除前4个sink → attention崩塌 → 输出变成乱码 → 模型失效!

为什么对serving重要?
  → 长对话 → KV cache持续增长 → OOM! → 需要evict旧KV → 但不能evict sink!
  → → 传统sliding window: 保留最近W个token → 移除旧token → 包括sink → 崩塌!
  → → StreamingLLM: 保留sink(前4) + 最近W个 → 移除中间 → 稳定! → 无限长对话!
  → → → 这是长对话serving的关键技术 → 不OOM + 不崩塌!

RTX 4090影响:
  → 7B模型 GQA-5 INT8 KV: 40.96KB/tok → 4096个tok = 168MB → 单GPU可以
  → → 但: S=100K → 4MB → OOM! → 需要KV eviction!
  → → StreamingLLM: sink(4 tok) + window(4096 tok) = 固定168MB → 无限对话不OOM!
  → → → RTX 4090最适合StreamingLLM → 内存有限 → 固定KV大小 → 永不OOM!
```

## 1. StreamingLLM: Sink + Sliding Window = 无限推理

```
StreamingLLM架构 (Xiao et al. 2023):

  传统Full Attention:
    → KV Cache: 所有token → [1, 2, 3, ..., S] → 随S增长 → S×40KB/tok → OOM!

  传统Sliding Window (失败):
    → 只保留最近W个 → [S-W, S-W+1, ..., S] → 固定大小 → 不OOM!
    → → 但: 移除前4个sink → softmax无锚点 → attention崩塌 → 输出乱码!
    → → → Mistral SlidingWindow: 只在训练时work → 推理时移除sink → 崩塌!

  StreamingLLM (成功):
    → 保留sink(前4 tok) + 最近W个 → [1,2,3,4, ..., S-W, ..., S]
    → → KV Cache大小: 4+W → 固定! → 不随S增长 → 永不OOM!
    → → → sink提供softmax锚点 → recent提供语义信息 → 稳定推理!
    → → → → 无限长对话 → 任何长度都只占(4+W)×40KB/tok!

  数学解释:
    → softmax(QK^T/√d): 每个query token对所有KV token计算attention
    → → 如果移除所有旧KV → 某些query的attention score全低 → softmax归一化失败!
    → → → 保留sink → sink吸收"溢出概率" → softmax正常归一化 → attention稳定!
    → → → → sink的角色 = "垃圾桶" → 接收不需要的attention → 不是语义需要 → 是数学需要!

  性能数据:
    → LLaMA-2 7B, StreamingLLM (sink=4, window=4096):
    → → Perplexity: 与full attention几乎相同 → PPL差异<0.1
    → → KV Cache: 固定168MB(7B INT8 GQA-5) → vs full: 无限增长 → OOM!
    → → 推理速度: decode不变 → 每步只读(4+W)个KV → 不读全部 → 更快!

  限制:
    → 无法回忆窗口外的信息 → [5, ..., S-W-1]的KV被丢弃 → 信息丢失!
    → → "_needle in haystack": 窗口外的重要信息 → 模型无法访问 → 准确率下降!
    → → → 适合: 通用对话/闲聊 → 不适合: 需要长距离回忆的任务(文档QA/代码)!
    → → → → 解决: 需要更智能的eviction策略(H2O/SnapKV) → 而不是简单位置策略!
```

## 2. H2O: Heavy-Hitter Oracle — 累积注意力的重要性保留

```
H2O架构 (Zhang et al. 2024):

  核心思想: 不是保留固定位置 → 而是保留"重要"的token → 重要性=累积attention分数!

  Heavy-Hitter定义:
    → 对每个KV token → 计算其累积attention分数 → Σ(step_i的attention_score)
    → → 累积分数高 → 该token被多次关注 → "heavy hitter" → 保留!
    → → 累积分数低 → 该token很少被关注 → 非heavy → 可以evict!

  算法:
    → 1. 初始: 所有token进入KV cache → 无eviction
    → 2. 当KV超过budget → 选择累积分数最低的token → evict!
    → 3. 保留: sink tokens(前4) + heavy hitters + 最近W个
    → → → budget分配: sink(4) + heavy(K) + recent(W) → 总budget = 4+K+W

  vs StreamingLLM:
    → StreamingLLM: 保留前4+最近W → 固定位置 → 简单但信息丢失
    → H2O: 保留前4+重要K+最近W → 动态位置 → 保留重要信息 → 减少信息丢失!
    → → → H2O在needle-in-haystack测试中优于StreamingLLM → 因为重要信息被保留!

  性能数据:
    → LLaMA-2 7B, H2O (budget=4096):
    → → KV压缩: 2-3x → 与full attention相比 → 准确率损失<1%
    → → Needle-in-haystack: 比StreamingLLM好 → 因为保留重要信息
    → → → 但: 仍然不如full attention → 因为budget有限 → 某些信息仍然丢失

  问题:
    → 累积attention分数 → 前面token累积更多 → 即使语义不重要 → 累积分数高!
    → → → 这是attention sink的变种 → sink token累积分数高 → 但语义不重要
    → → → → H2O需要额外处理sink → sink自动保留 → 但非sink的旧token也可能不公平累积!
    → → → → → 解决方案: 加入decay factor → 累积分数 × exp(-λ×age) → 旧token分数衰减!

  vLLM/SGLang集成:
    → vLLM: 没有原生H2O → 但有PagedAttention + prefix caching → 类似保留重要信息
    → → → KV eviction: LRU block eviction → 类似H2O → 但基于block使用频率而非attention分数
    → SGLang: RadixAttention eviction → 7种策略 → 类似H2O但基于prefix tree而非attention分数
    → → → 生产中: LRU eviction足够 → H2O的attention分数追踪开销太大 → 不实用!
```

## 3. SnapKV: 注意力模式聚类压缩

```
SnapKV架构 (Zang et al. 2024):

  核心洞察: 不需要保留所有KV → 只需要保留"attention模式需要的"KV

  方法:
    → 1. Observation Window: 使用最近W个token的attention pattern → 作为"观察窗口"
    → → 在观察窗口中 → 每个query对哪些KV位置attention最高? → 这些是"重要位置"
    → 2. Cluster: 对所有query的"重要位置"集合 → 聚类 → 合并相似位置
    → → → 聚类后 → 只保留每个cluster的代表KV → 其他KV丢弃
    → → → → 压缩率: 聚类数/KV数 → 例如100个cluster/1000个KV → 10x压缩!

  数学:
    → 对observation window的每个query → 计算attention distribution
    → → 选取top-K attention位置 → 重要位置集合 = ∪(top-K_i) for all query_i
    → → 对重要位置 → 聚类 → cluster代表 → 保留代表 → 丢弃其他
    → → → 推理时: 每个query → 注意cluster代表 → 代表聚合cluster内信息 → 近似完整attention

  vs H2O:
    → H2O: 逐token重要性 → 保留heavy hitter token → 粒度: token级
    → SnapKV: attention pattern聚类 → 合并相似KV → 粒度: cluster级 → 更细!
    → → SnapKV压缩率更高(3.6x vs H2O 2-3x) → 因为合并而非只选择!

  vs StreamingLLM:
    → StreamingLLM: 固定位置 → sink+recent → 最简单
    → SnapKV: 动态pattern → 聚类 → 更智能 → 压缩率更高
    → → → 但: SnapKV需要实时聚类 → 计算开销 → streaming更轻量!

  性能数据:
    → LLaMA-2 7B, SnapKV (压缩3.6x):
    → → Perplexity差异: <0.1 → 几乎无损!
    → → KV Cache: 3.6x压缩 → 7B: 168MB → 47MB → 单GPU轻松!
    → → → 适合: 离线/batch推理 → 需要实时聚类 → 不适合streaming低延迟!

  RTX 4090适用性:
    → 聚类计算: CPU或GPU → 开销小 → 但需要额外kernel → 当前无vLLM/SGLang原生支持
    → → SnapKV是研究方向 → 生产主流仍是StreamingLLM(sink+window) → 简单可靠!
    → → → RTX 4090推荐: StreamingLLM → 不是SnapKV → 因为简单+无额外计算!
```

## 4. PyramidKV: 层自适应KV预算分配

```
PyramidKV (2025):

  核心洞察: 不同层的attention分布不同 → 不应该给所有层相同的KV预算!

  层差异:
    → 低层(layer 0-8): attention分布均匀 → 需要更多KV → 需要看广泛上下文
    → → → 语义信息: 低层关注语法/位置 → 需要更多上下文 → KV预算应该更大
    → 高层(layer 24-32): attention集中 → 只关注少数token → 需要更少KV
    → → → 语义信息: 高层关注语义/关键信息 → 只关注重要token → KV预算可以更小

  金字塔分配:
    → Layer 0: budget = 4 × base → 看最广泛上下文 → 需要最多KV
    → Layer 8: budget = 2 × base → 中等 → 看中等范围
    → Layer 16: budget = 1 × base → 标准
    → Layer 24: budget = 0.5 × base → 集中 → 只看关键
    → Layer 32: budget = 0.25 × base → 最集中 → 只看最近+最关键
    → → → 金字塔: 低层多 → 高层少 → 总KV预算: 4+2+1+0.5+0.25 = 7.75×base
    → → → → vs uniform: 32层×base = 32×base → PyramidKV省32/7.75 = 4.1x!

  数学依据:
    → attention entropy: H = -Σp_i log(p_i) → 低entropy=集中 → 高entropy=分散
    → → 低层entropy高 → 分散 → 需要更多KV → 预算大
    → → 高层entropy低 → 集中 → 需要更少KV → 预算小
    → → → PyramidKV = entropy-aware budget allocation → 自适应!

  性能数据:
    → LLaMA-2 7B, PyramidKV vs uniform:
    → → 相同总KV预算 → PyramidKV PPL更好 → 因为低层保留更多上下文
    → → → 2x总压缩 → PyramidKV PPL损失1% → uniform PPL损失5% → PyramidKV好4x!
    → → → → 关键: 不是所有层都需要相同KV → 低层需要更多 → 高层需要更少!

  RTX 4090影响:
    → 7B 32层: uniform budget → 每层256 tok → 总32×256=8192 tok → 336MB
    → → PyramidKV budget → 低层1024 高层64 → 总约7.75×256≈2000 tok → 82MB → 4x省!
    → → → RTX 4090: PyramidKV → 7B模型 → 82MB KV → B=32 → 总2.6GB → 轻松!
    → → → → vs uniform: 336MB × 32 = 10.7GB → 勉强 → 更多并发意味着更多OOM风险!

  生产可行性:
    → vLLM: 不支持per-layer KV budget → 所有层共享相同block pool → 需要修改!
    → → 当前vLLM: eviction是block级 → 所有层一起evict → 不支持per-layer差异
    → → → 实现需要: per-layer block pool → 复杂度增加 → 但收益显著!
    → → → → PyramidKV是未来方向 → 但当前生产仍用uniform → 等框架支持!
```

## 5. 其他KV Cache压缩方法

```
2024-2025 KV Cache压缩方法全景:

| 方法 | 核心机制 | 压缩率 | 精度损失 | 生产就绪 |
|------|---------|--------|---------|---------|
| StreamingLLM | sink+window | ~∞(固定) | 窗口外丢失 | ✅(vLLM/SGLang) |
| H2O | 累积attention | 2-3x | <1% | ❌(attention追踪开销) |
| SnapKV | pattern聚类 | 3.6x | <0.1% | ❌(需聚类kernel) |
| PyramidKV | 层自适应预算 | 4.1x | <1% | ❌(需per-layer pool) |
| ClusterKV | KV向量合并 | 2-4x | 中 | ❌(需合并kernel) |
| DMC | token动态合并 | 2-3x | 低 | ❌(需fused kernel) |
| Scissorhands | 低影响token丢弃 | 2-3x | 中 | ❌(需预测) |
| GEAR | 量化+低秩近似 | 4-8x | 中 | ❌(需量化+低秩kernel) |
| KIVI | 2-bit量化(K/V不同) | 16x | 低 | ❌(需量化kernel) |
| Infini-Attention | compress+retrieve | ~∞ | 中 | ❌(需修改架构) |

量化 vs Eviction对比:
  → 量化(KIVI/INT8): 降低每个KV的精度 → 精度损失小 → 内存省50%(INT8)/93%(2-bit)
  → → 优点: 所有token保留 → 不丢失信息 → 精度损失可控
  → → 缺点: 仍然随S增长 → 不解决无限对话问题

  → Eviction(StreamingLLM/H2O): 移除token → 内存固定 → 但信息丢失
  → → 优点: 固定内存 → 无限对话 → 不OOM
  → → 缺点: 窗口外信息丢失 → needle-in-haystack问题

  → 最优组合: Eviction(固定大小) + 量化(压缩保留KV) → StreamingLLM + INT8 KV → 最优!
  → → → RTX 4090最优: sink(4) + window(4096) + INT8 KV → 固定168MB → 无限对话!

混合策略:
  → 前景: 低层 PyramidKV(更多budget) + 高层 StreamingLLM(更少budget)
  → → INT8/INT4 KV → 压缩保留的KV → 内存更省
  → → → H2O importance-based eviction → 重要信息优先保留
  → → → → 未来方向: PyramidKV + H2O + INT8 → 三重优化 → 最优长对话serving!
```

## 6. Infini-Attention: Google的无限注意力机制

```
Infini-Attention (Google, 2024):

  核心思想: 不evict → 而是compress → 旧KV压缩到固定大小 → 需要时retrieve!

  机制:
    → Segment-based processing: 输入分成segment → 每个segment独立处理
    → → 当前segment: 标准attention → 完整KV → 完整语义信息
    → → 旧segment: compress → 压缩到固定维度 → 存入memory
    → → → 需要时: retrieve → 从compressed memory恢复 → 混合当前+旧信息

  数学:
    → Current segment: Attn_current = softmax(Q_curr × K_curr^T / √d) × V_curr
    → → Compress: memory = Linear(K_old × V_old) → 压缩到d维 → 存入memory bank
    → → Retrieve: Attn_retrieved = Q_curr × memory^T → 从compressed memory获取旧信息
    → → → 混合: output = gate × Attn_current + (1-gate) × Attn_retrieved
    → → → → gate由当前query决定 → 需要旧信息时gate打开 → 不需要时gate关闭!

  vs StreamingLLM:
    → StreamingLLM: 丢弃窗口外 → 信息完全丢失 → 无法恢复
    → Infini-Attention: 压缩旧KV → 信息部分保留 → 可以retrieve
    → → → Infini-Attention在needle-in-haystack中优于StreamingLLM → 因为可以retrieve旧信息!

  限制:
    → 需要修改模型架构 → 不是纯推理优化 → 需要训练新模型!
    → → compress/retrieve需要额外参数(gate + compress linear) → 训练复杂
    → → → 当前: 只在Google内部使用 → 社区无法直接应用 → 需要模型支持!

  RTX 4090:
    → 不适合 → 需要模型重新训练 → 当前LLaMA/Qwen不支持Infini-Attention
    → → → RTX 4090最优仍是StreamingLLM + INT8 KV → 简单+无架构修改!
```

## 7. 生产部署: RTX 4090长对话serving决策

```
RTX 4090长对话serving决策树:

  场景分类:
    → 短对话(S≤4K): 全KV → 不需要eviction → INT8 KV足够 → 推荐!
    → → 7B INT8 GQA-5: 40KB/tok × 4096 = 168MB → B=32 → 总5.4GB → 可行!

    → 中对话(S=4K-16K): StreamingLLM → sink+window → 推荐!
    → → 7B INT8 GQA-5 StreamingLLM(4+4096): 固定168MB → 任何长度都可行!
    → → → Sliding window Mistral: S≤4096 → 不需要eviction → 但更长需要!

    → 长对话(S=16K-100K): StreamingLLM + 重要信息损失 → 考虑:
    → → 方案1: StreamingLLM → 信息损失 → 但对话质量仍然可接受(大多数对话)
    → → 方案2: H2O → 保留重要信息 → 但需要attention追踪 → 生产开销
    → → 方案3: 全KV + 量化 → INT4 KV → 内存省93% → 但仍随S增长 → S=100K = 3.36MB×100K = 336MB → 单GPU可能!

    → 超长对话(S>100K): 当前RTX 4090很难 → 需要多GPU或PD分离:
    → → INT4 KV: 7B GQA-5 → 5KB/tok → 100K tok = 500MB → 单GPU可行
    → → → 但: attention计算对S=100K → O(S²) → 计算量巨大 → prefill慢!
    → → → → 需要: chunked prefill + StreamingLLM → 分段处理 → 每段4K → 可行!

  RTX 4090最优配置:
    → 短对话(S≤4K): 全KV + INT8 + GQA-5 + FlashInfer → 145K tok/s(B=32) → 推荐!
    → 中对话(S=4K-16K): StreamingLLM(sink=4, window=4096) + INT8 KV → 固定内存 → 推荐!
    → 长对话(S=16K-100K): INT4 KV + 全保留 → 但prefill慢 → chunked prefill → 可行
    → 超长(S>100K): 不推荐 → 需要H100集群 → RTX 4090不适合!

  vLLM配置:
    → Sliding window: `--sliding-window 4096` → vLLM自动保留最近W个 + sink
    → → 但: vLLM sliding window实现 → 需要模型支持(Mistral原生 → LLaMA需要修改)
    → → → 当前: Mistral + vLLM sliding window → 可用 → 其他模型需要StreamingLLM patch!
```

## 8. 核心学习

```
1. **Attention Sink = softmax归一化副产品**: 不是语义需要 → 是数学需要 → 前4 token吸收80%概率 → 必须保留!
2. **StreamingLLM = 生产最优**: sink(4)+window(W) → 固定KV → 无限对话 → 简单可靠!
3. **H2O/SnapKV/PyramidKV = 研究方向**: 更智能eviction → 但生产开销大 → 暂不实用!
4. **量化+eviction组合**: StreamingLLM + INT8 KV → 三重优化 → RTX 4090最优!
5. **Infini-Attention = 未来方向**: compress+retrieve → 不丢失信息 → 但需模型重训练!
6. **低层需要更多KV**: PyramidKV证明 → 不是所有层都一样 → 低层entropy高 → 预算大
7. **RTX 4090最适合StreamingLLM**: 内存有限 → 固定KV → 永不OOM → 无限对话!
```

---

**Sources**:
- [StreamingLLM (Xiao et al. 2023)](https://arxiv.org/abs/2309.17453)
- [H2O (Zhang et al. 2024)](https://arxiv.org/abs/2406.14118)
- [SnapKV (Zang et al. 2024)](https://arxiv.org/abs/2405.15860)
- [PyramidKV (2025)](https://arxiv.org/abs/2410.02157)
- [Infini-Attention (Google 2024)](https://arxiv.org/abs/2404.07143)
- [Scissorhands (2023)](https://arxiv.org/abs/2305.17126)

**Related notes**: kv-cache-management-deep-dive.md, long-context-serving.md, flashinfer-attention-deep-dive.md