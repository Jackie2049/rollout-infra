# Tokenization Deep Dive: BPE→SentencePiece→Tiktoken — 从分词到推理性能

> 2026-06-08 | Tokenizer选择直接决定推理成本: vocab size→lm_head大小→decode带宽→KV并发数, 大vocab(128K)省token但增softmax开销, RTX 4090最优=32K-64K
> 基于: BPE(GPT系列), Unigram/SentencePiece(T5/Gemma), Tiktoken(Llama-3), Qwen(248K vocab)
> 参考: "All Languages Not Created Token-equal"(Ahia 2023), Llama-3技术报告(128K vocab), Gemma-2(256K)
> 关联: kv-cache-management-deep-dive.md, inference-cost-analysis.md, serving-framework-comparison.md

## 0. 核心定律: Tokenizer = 压缩器 = 推理成本决定器

```
Tokenizer的推理影响链:
  → Tokenizer压缩率 → 同文本→多少token → 直接决定:
  → → KV Cache token数 → 内存/并发 → 成本!
  → → Prefill计算量 → O(N²) → N少=快!
  → → Decode步数 → 生成N步 → N少=快!
  → → lm_head大小 → vocab_size × d_model → 大vocab=大权重=带宽瓶颈!

  数学推导:
    → Tokenizer压缩率: chars_per_token = 文本长度 / token数
    → → BPE 32K: chars_per_token ≈ 3-4 → 同文本需要更多token → 更多KV → 更慢
    → → BPE 128K: chars_per_token ≈ 6-8 → 同文本更少token → 更少KV → 更快(per-text)
    → → → 但: lm_head权重 = vocab_size × d_model × dtype_size

  RTX 4090具体影响:
    → 7B模型, vocab=32K: lm_head = 32K × 4096 × 2(BF16) = 256MB → 占总权重10%
    → 7B模型, vocab=128K: lm_head = 128K × 4096 × 2(BF16) = 1024MB → 占总权重36%!
    → → → Decode时lm_head每步读取 → memory-bound → vocab大=每步读取大=慢!

    → 但: 128K vocab → chars_per_token=7 → 同文本token数少4x → KV少4x → 总KV省!
    → → → 7B GQA-5 INT8 KV: 32K vocab → S=4096 tok → 168MB
    → → → 7B GQA-5 INT8 KV: 128K vocab → S=1024 tok → 42MB → 4x省!

    → → → → 竞争: lm_head更大(256→1024MB) vs KV更小(168→42MB)
    → → → → → 谁赢? → KV是per-request → lm_head是per-GPU → 并发数决定!

  RTX 4090最优vocab:
    → 单请求: 128K vocab更优 → token少 → KV小 → 快!
    → 多并发(B=32): 32K vocab更优 → lm_head小 → 多并发 → 总吞吐更高!
    → → → 7B + 32K vocab + INT8 KV + GQA-5 → B=32 → 145K tok/s → 推荐!
    → → → 7B + 128K vocab → lm_head 1024MB → B=8 → 总吞吐更低 → 不推荐!
    → → → → **RTX 4090最优vocab = 32K-64K** → 平衡lm_head大小和压缩率!
```

## 1. BPE: 自底向上的贪心合并

```
BPE (Byte Pair Encoding) — GPT-2/3/4, LLaMA-3, Claude:

  算法:
    → 初始: 每个字符(或byte)是一个token → vocab = 所有字符
    → 步骤1: 找到最频繁出现的相邻token pair → 合并为新token
    → → 例: "th"出现1000次 → 合并 → vocab增加"th"
    → 步骤2: 重复 → 找最频繁pair → 合并 → vocab增加
    → → 直到vocab_size达到目标 → 停止!
    → → → 贪心算法 → 每步选择当前最优 → 不保证全局最优!

  BPE训练示例:
    → 初始vocab: {a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z}
    → Step 1: "th"最频繁 → 合并 → vocab: {a,...,z, "th"}
    → Step 2: "the"最频繁 → 合并 → vocab: {a,...,z, "th", "the"}
    → Step 3: "ing"最频繁 → 合并 → vocab: {a,...,z, "th", "the", "ing"}
    → → → 持续合并 → 最终vocab_size=32K → 包含常用词/子词/单字符

  关键特性:
    → 确定性: 同文本 → 唯一编码 → 无歧义 → 生产推理可靠!
    → → 因为: BPE总是合并最长的匹配token → 固定顺序 → 固定结果
    → → → vs Unigram: 同文本可能有多种编码 → 需要概率选择 → 不确定性!

    → 覆盖性: 任何文本都可以编码 → 因为: 字符级fallback
    → → → 遇到未知token → 回退到字符级 → 永远可以编码 → 不会失败!

    → 多语言偏见: BPE偏向训练数据的主要语言 → 英文词合并优先 → 中文词更碎!
    → → → 例: "hello" → 1 token, "你好" → 3-4 tokens → 中文推理更贵!
    → → → → Ahia 2023: 非英文语言 → token数多14x → 推理成本14x → 不公平!

  Tiktoken (Meta/OpenAI variant):
    → GPT-4/LLaMA-3使用Tiktoken → BPE的优化实现
    → → 特点: 正则表达式预分割 → 先按空格/标点分割 → 再BPE → 避免跨词合并
    → → → 例: "hello world" → 先分割["hello", " world"] → 再BPE → 更好的词边界!
    → → → → 预分割确保token不跨词 → 推理时token边界更自然 → 输出更稳定!
```

## 2. Unigram: 自顶向下的概率删除

```
Unigram (SentencePiece default) — T5, Gemma-2, 多语言模型:

  算法:
    → 初始: 大vocab(包含所有可能的子词) → 如: 1M候选token
    → 步骤1: 对训练数据 → 计算每个token的使用概率
    → 步骤2: 删除对总likelihood贡献最小的token → vocab减小
    → → 例: "xyz"很少出现 → 删除 → vocab减少
    → 步骤3: 重复 → 删除贡献最小的 → 直到vocab_size达到目标
    → → → 自顶向下 → 从大到小 → 保留最重要的token!

  vs BPE关键区别:
    → BPE: 合最频繁的 → 保留高频组合 → 删除低频组合 → 频率驱动
    → Unigram: 删除贡献最小的 → 保留对likelihood贡献大的 → 概率驱动
    → → → Unigram更"全局最优" → 因为基于总likelihood而非贪心频率!

  概率性编码 (关键特性!):
    → Unigram: 同文本 → 多种合法编码 → 选择概率最高的
    → → 例: "international" → ["inter", "national"] 或 ["international"] → 两个都合法!
    → → → 选择: Viterbi算法 → 找概率最高的编码路径 → 最优编码!
    → → → → 但: 同文本不同次编码可能不同 → 不确定性 → 生产需固定seed!

  多语言优势:
    → Unigram不偏向任何语言 → 删除是基于likelihood贡献 → 而非频率!
    → → → 中文token贡献大 → 保留 → 英文token贡献大 → 也保留 → 更公平!
    → → → → 例: Gemma-2 256K vocab → 中文 chars_per_token ≈ 4 → 英文 ≈ 6 → 更接近公平!
    → → → → vs BPE 32K → 中文 chars_per_token ≈ 1-2 → 英文 ≈ 4 → 14x差距!

  SentencePiece框架:
    → 不仅实现Unigram → 也实现BPE → 用户可以选择
    → → 语言无关: 输入是raw bytes → 不假设空格分隔 → 适合中文/日文!
    → → → BPE(OPT/GPT-2): 先用空格分割 → 假设英文格式 → 中文不友好!
    → → → → SentencePiece: 不需要预分割 → raw byte输入 → 中文友好!
```

## 3. Vocab Size对推理的定量影响

```
推理影响公式:

  1. Embedding层:
    → 权重: vocab_size × d_model × dtype_size
    → → 7B BF16: vocab=32K → 32K×4096×2 = 256MB
    → → 7B BF16: vocab=128K → 128K×4096×2 = 1024MB (4x!)
    → → → Embedding只在prefill第一步读取 → 不占decode带宽 → 影响小!

  2. lm_head (output projection):
    → 权重: d_model × vocab_size × dtype_size → 同embedding大小!
    → → → **每步decode都要读取!** → memory-bound → 占decode带宽20-30%!
    → → → 7B vocab=32K: lm_head=256MB → 每步读取256MB → @890GB/s → 0.29ms
    → → → 7B vocab=128K: lm_head=1024MB → 每步读取1024MB → @890GB/s → 1.15ms
    → → → → **vocab=128K → lm_head读取增加4x → decode慢4x!** (仅lm_head部分)

  3. Softmax计算:
    → softmax(logits): vocab_size维度 → 每步需要计算vocab_size个logits
    → → vocab=32K: softmax 32K维度 → GPU kernel ~0.05ms
    → → vocab=128K: softmax 128K维度 → GPU kernel ~0.15ms → 3x慢!
    → → → 但: softmax < lm_head读取 → 不是主要瓶颈 → lm_head读取更重要!

  4. KV Cache per request:
    → KV/tok = 2 × num_kv_heads × d_head × dtype_size × num_layers
    → → KV大小与vocab_size无关! → KV/tok不随vocab变化
    → → → 但: **同文本的token数与vocab有关!**
    → → → → chars_per_token: 32K ≈ 4, 128K ≈ 7 → 同文本token数少~1.75x
    → → → → → 7B GQA-5 INT8: 40KB/tok × 4K tok(32K vocab) = 168MB
    → → → → → 7B GQA-5 INT8: 40KB/tok × 2.3K tok(128K vocab) = 93MB → 省1.8x!

  总推理成本比较 (7B模型, S=4096 chars, RTX 4090):

    | Vocab | Tokens | lm_head(MB) | lm_head(ms) | KV(MB) | Prefill ms | Decode ms/tok | Total per-text(ms) |
    |-------|--------|------------|------------|--------|-----------|--------------|-------------------|
    | 32K | 1024 | 256 | 0.29 | 42 | ~10 | ~1.0 | 1024+10=1034 |
    | 64K | 580 | 512 | 0.57 | 23 | ~6 | ~1.5 | 580×1.5+6=876 |
    | 128K | 580 | 1024 | 1.15 | 23 | ~6 | ~2.1 | 580×2.1+6=1224 |

  **关键发现**:
    → 128K vocab → per-text总时间更长! → 因为decode每步慢(lm_head大)
    → → 但: KV更小 → 更多并发 → B更大 → 吞吐可能更高!
    → → → 并发=1: 32K更快 → 并发=B=32: 需要看lm_head带宽竞争!

  并发吞吐比较 (B=32):
    → 32K vocab: lm_head=256MB → B=32 → 8.2GB/s带宽 → decode 1ms → 32K tok/s
    → 128K vocab: lm_head=1024MB → B=32 → 带宽不够! → 只能B=8 → 8×2.1ms=17ms → 470 tok/s
    → → → **32K vocab高并发吞吐更高!** → 因为lm_head小 → 更多请求同时decode!

  RTX 4090结论:
    → 单请求/低并发: 128K vocab稍快 → token少 → 但lm_head开销抵消
    → 多并发(B≥8): 32K vocab明显更快 → lm_head小 → 更多并发 → 推荐!
    → → → **RTX 4090最优vocab = 32K-64K** → 平衡lm_head大小和并发吞吐!
```

## 4. 主流模型Tokenizer对比

```
2024-2025主流模型Tokenizer:

| 模型 | Tokenizer | Vocab | chars/tok(EN) | chars/tok(ZH) | 中文/英文差距 | 特点 |
|------|-----------|-------|-------------|-------------|------------|------|
| GPT-4 | BPE(tiktoken) | ~100K | ~6 | ~2 | 3x | 英文优先,预分割 |
| LLaMA-2 | BPE(SentencePiece) | 32K | ~4 | ~1.5 | 2.7x | 英文优先 |
| LLaMA-3 | BPE(tiktoken) | 128K | ~6 | ~3 | 2x | 多语言改进 |
| Qwen-2.5 | BPE(tiktoken) | 151K | ~6 | ~4 | 1.5x | 中文优化! |
| Gemma-2 | Unigram | 256K | ~8 | ~5 | 1.6x | 最公平 |
| Mistral | BPE(Tekken) | 131K | ~6 | ~3 | 2x | 平衡 |
| DeepSeek-V3 | BPE | 129K | ~6 | ~3 | 2x | 中文优化 |

  关键趋势:
    → 2024年前: vocab 32K → 英文4 chars/tok → 中文1.5 → 差距2.7x
    → 2024年后: vocab 128K+ → 英文6 chars/tok → 中文3 → 差距缩小到2x!
    → → → 大vocab = 多语言公平 → 但lm_head开销增大 → 推理成本权衡!

  中文推理成本差异:
    → LLaMA-2(32K): 中文token数=英文2.7x → KV大2.7x → 推理贵2.7x → 不公平!
    → Qwen-2.5(151K): 中文token数=英文1.5x → KV大1.5x → 推理贵1.5x → 更公平!
    → → → 中文服务 → Qwen系列更优 → 因为中文token压缩更好 → 推理更便宜!

  Qwen-2.5 vocab=151K的特殊设计:
    → 包含大量中文token → 中文chars_per_token ≈ 4 → 接近英文
    → → 但: vocab_size=151K → lm_head=151K×4096×2=1.2GB → 很大!
    → → → 推理: 中文便宜1.5x → 但lm_head大4.7x → 需要多GPU → 单GPU不够!
    → → → → H100/H200适合大vocab → RTX 4090不适合 → lm_head太大!

  Tokenizer选择决策树:
    → 英文服务 → 32K-64K vocab → LLaMA/Mistral → RTX 4090最优!
    → 中文服务 → 128K+ vocab → Qwen → 但需要H100(大lm_head)
    → 多语言服务 → 128K-256K vocab → Gemma-2/Qwen → H100必需!
    → → RTX 4090: 英文服务(32K vocab) → 7B → 单GPU → 推荐!
    → → RTX 4090: 中文服务(128K vocab) → 7B → lm_head 1GB → 单GPU勉强 → 不推荐!
```

## 5. Tokenizer优化技术

```
推理优化:

  1. Vocabulary Pruning ( vocab裁剪 ):
    → 大vocab → 但实际只使用部分token → 可以裁剪!
    → → 例: vocab=128K → 实际只使用30K → 裁剪未使用token → lm_head从1GB→240MB!
    → → → 方法: 统计推理数据 → 找出top-K常用token → 重新编号 → 裁剪
    → → → → 但: 裁剪后无法生成裁剪的token → 输出受限 → 不适合通用API!
    → → → → → 适用: 特定领域(医学/法律/代码) → vocab裁剪 → 推理更快!

  2. Speculative Decoding ( 推测解码 ):
    → 大vocab → softmax慢 → draft model用小vocab → 快速推测 → 验证!
    → → → 减少1-2次大vocab softmax → decode加速 → 但需要draft model!
    → → → → RTX 4090: n-gram draft model → 零额外模型 → 推荐!

  3. Shared Embedding ( 共享embedding ):
    → embedding和lm_head共享权重 → 参数省50% → 但需要transpose!
    → → → 7B vocab=32K: 256MB+256MB → shared → 256MB → 省256MB!
    → → → → vLLM: 默认shared embedding → 需要model配置支持!
    → → → → 但: transpose可能慢 → 需要fused kernel → 实际收益看模型!

  4. LoRA for lm_head:
    → lm_head太大 → 用LoRA → 可训练参数省99% → 但推理需要merge!
    → → → 合并后lm_head权重不变 → 不省推理开销 → 只省训练开销!
    → → → → 不是推理优化 → 是训练优化 → 对推理无帮助!

训练优化:

  5. BPE Dropout:
    → 训练时: 随机跳过BPE合并 → 同文本每次不同编码 → 数据增强!
    → → → 防止模型"记住"特定tokenization → 更鲁棒 → 类似Dropout!
    → → → → 推理时不用dropout → 固定编码 → 生产稳定!

  6. Multilingual BPE:
    → 训练数据: 多语言混合 → BPE合并频率多语言平衡 → 中文词也频繁合并!
    → → → Qwen/Llama-3: 多语言BPE → 中文chars_per_token改善 → 推理更公平!
    → → → → 但: vocab_size增大 → 推理成本权衡 → 需要128K+ vocab!

  7. Byte-level BPE:
    → 不用字符 → 用byte → 256 byte token → 覆盖所有语言!
    → → → GPT-2: byte-level BPE → 任何Unicode字符都能编码 → 不会失败!
    → → → → 但: 中文byte-level → 每个中文字3-4 byte → 3-4 token → 碎!
    → → → → → 大vocab BPE比byte-level更好 → 因为合并常用byte pair → 中文词更短!
```

## 6. RTX 4090 Tokenizer决策

```
RTX 4090 Tokenizer最优配置:

  英文/代码服务:
    → 模型: LLaMA-2 7B / Mistral 7B → vocab=32K
    → → lm_head=256MB → 单GPU足够 → B=32 → 145K tok/s → 推荐!
    → → → chars_per_token≈4 → S=4096 → 1024 tok → KV=42MB → B=32 → 可行!

  中文服务:
    → 模型: Qwen-2.5 7B → vocab=151K → lm_head=1.2GB → 太大!
    → → → 单GPU 24GB不够 → lm_head占50% → 其他权重+KV不够 → 不推荐!
    → → → 替代: Qwen-2.5 0.5B → vocab=151K → lm_head=150MB → 可行!
    → → → → 或: 用vocab裁剪 → Qwen-2.5裁剪到64K → lm_head=480MB → 可行!
    → → → → → 但: 裁剪后中文token数增加 → chars_per_token下降 → 效果打折!

  多语言服务:
    → RTX 4090: 不适合 → vocab需要128K+ → lm_head太大 → 需要H100!
    → → → 唯一可能: 小模型(0.5B) + 大vocab → lm_head比例小 → 勉强可行

  Embedding/lm_head大小对比:

    | 模型 | Vocab | d_model | lm_head(MB) | 占总权重% | RTX 4090可行 |
    |------|-------|---------|------------|----------|------------|
    | OPT-125M | 50K | 768 | 75 | 12% | ✅轻松 |
    | LLaMA-7B | 32K | 4096 | 256 | 10% | ✅可行 |
    | LLaMA-3-8B | 128K | 4096 | 1024 | 26% | ❌太大 |
    | Qwen-2.5-7B | 151K | 4096 | 1208 | 30% | ❌太大 |
    | Gemma-2-9B | 256K | 4096 | 2048 | 43% | ❌完全不行 |
    | Qwen-2.5-0.5B | 151K | 896 | 270 | 22% | ⚠️勉强 |

  **核心结论**:
    → RTX 4090最优vocab = 32K-64K → lm_head ≤ 512MB → 占总权重≤20%
    → → vocab=128K+ → lm_head > 1GB → 占总权重>25% → 推理带宽瓶颈 → 不适合!
    → → → 大vocab需要大GPU → H100 80GB → 或TP=2+ → RTX 4090不适合!
    → → → → **RTX 4090 = 小vocab + GQA + INT8 KV = 最优推理配置!**
```

## 7. 核心学习

```
1. **Tokenizer = 推理成本决定器**: vocab size → lm_head大小 → decode带宽 → 并发数 → 成本!
2. **BPE确定性 vs Unigram概率性**: 生产推理偏好确定性 → BPE更适合serving!
3. **大vocab省token但增开销**: chars_per_token↑ → token↓ → 但lm_head↑ → 权衡!
4. **多语言公平需要大vocab**: 中文32K vocab→2.7x贵 → 128K vocab→1.5x → 但lm_head太大!
5. **RTX 4090最优vocab=32K-64K**: lm_head≤512MB → 单GPU → B=32 → 145K tok/s → 推荐!
6. **lm_head占decode带宽20-30%**: vocab越大 → 占比越高 → memory-bound越严重!
7. **Qwen中文最优但需H100**: vocab=151K → 中文1.5x公平 → 但RTX 4090内存不够 → H100必需!
```

---

**Sources**:
- [BPE (Sennrich et al. 2016)](https://arxiv.org/abs/1508.07909)
- [SentencePiece (Kudo 2018)](https://arxiv.org/abs/1808.06226)
- [Unigram (Kudo 2018)](https://arxiv.org/abs/1804.10959)
- [Tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- [Llama-3 128K vocab](https://ai.meta.com/blog/meta-llama-3/)
- ["All Languages Not Created Token-equal" (Ahia 2023)](https://arxiv.org/abs/2305.19166)

**Related notes**: kv-cache-management-deep-dive.md, inference-cost-analysis.md, serving-framework-comparison.md