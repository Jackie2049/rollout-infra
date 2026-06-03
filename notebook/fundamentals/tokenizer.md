# Tokenizer 与推理管线

> Tokenization 如何影响模型性能、推理成本和 KV Cache 设计

## 1. Tokenization 基础

### 1.1 什么是 Tokenizer

```
将文本转换为 token ID 序列（模型输入）:

"Hello, how are you?" → [15496, 11, 703, 527, 345, 30]

Tokenizer = 编码器 (文本→ID) + 解码器 (ID→文本)
词表 (Vocabulary): 所有可能的 token 集合, 大小 = vocab_size
```

### 1.2 主流 Tokenizer 类型

```
| 类型             | 代表        | 特点                   |
|-----------------|-------------|------------------------|
| BPE             | GPT-2/3/4   | 从字符逐步合并高频子串   |
| SentencePiece   | LLaMA       | 支持多语言，字节级 BPE  |
| WordPiece       | BERT        | 基于似然的子词选择       |
| Unigram         | T5, Gemma   | 从大词表逐步剪枝         |

现代 LLM 几乎都用 BPE 的变体:
  GPT 系列: tiktoken (BPE, cl100k_base)
  LLaMA: SentencePiece BPE
  Qwen: tiktoken (BPE, cl100k 变体)
```

### 1.3 BPE (Byte Pair Encoding) 工作原理

```
训练阶段:
  1. 初始词表: 所有单个字节 (256 个)
  2. 统计相邻 token 对的频率
  3. 合并频率最高的 token 对 → 新 token 加入词表
  4. 重复直到词表达到目标大小 (如 32000, 128000)

示例:
  初始: "low" "lower" "lowest" → ['l','o','w'], ['l','o','w','e','r'], ...
  合并1: 'l'+'o' → 'lo' → ['lo','w'], ['lo','w','e','r'], ...
  合并2: 'lo'+'w' → 'low' → ['low'], ['low','e','r'], ...
  合并3: 'e'+'r' → 'er' → ['low'], ['low','er'], ...

编码阶段:
  对输入文本，按训练好的合并规则贪心地合并
```

## 2. 词表大小对模型的影响

### 2.1 显存和计算

```
Embedding 层: [vocab_size, hidden_dim]
LM Head:      [hidden_dim, vocab_size] (通常与 Embedding 共享权重)

显存: vocab_size × hidden_dim × dtype_bytes

LLaMA-7B (vocab=32000, hidden=4096, FP16):
  Embedding: 32000 × 4096 × 2 = 256 MB

LLaMA-3-8B (vocab=128000, hidden=4096, FP16):
  Embedding: 128000 × 4096 × 2 = 1 GB  ← 4x 增长

GPT-4 (vocab=100256, hidden≈7168, FP16):
  Embedding: ~1.4 GB
```

### 2.2 计算瓶颈：LM Head

```
每个生成步骤的最后一个操作:
  logits = hidden @ W_lm_head^T    [batch, hidden] × [hidden, vocab]
  → [batch, vocab_size]

vocab=128000 时:
  每个 token 的 LM Head: 4096 × 128000 = 524M FLOPs
  整个 forward 的其他层: ~16B FLOPs (8B model)
  LM Head 占比: 524M / 16B ≈ 3.3%

→ Decode 阶段 LM Head 不是瓶颈（计算量相对小）
→ 但 softmax over 128K classes 需要注意数值稳定性
```

### 2.3 大词表的趋势

```
| 模型      | 词表大小 | 原因                     |
|-----------|---------|--------------------------|
| GPT-2     | 50,257  | 英文为主                 |
| LLaMA-2   | 32,000  | 多语言但压缩率一般       |
| LLaMA-3   | 128,256 | 多语言优化 + 更高压缩率  |
| Qwen-2.5  | 151,936 | 中英日韩等多语言全覆盖   |
| Gemma-2   | 256,000 | 极大多语言支持           |

大词表的好处:
  - 更高的压缩率（每个 token 平均代表更多字符）
  - 多语言支持更好（中文/日文/韩文等高效编码）
  - 减少推理总 token 数 → 降低延迟和成本

大词表的代价:
  - Embedding 显存增大
  - LM Head 计算增加
  - 训练数据需要足够覆盖所有 token
```

## 3. Token 压缩率对推理的影响

### 3.1 不同语言的压缩率

```
"compression ratio" = 字符数 / token 数

英文:
  "Hello, how are you today?" = 26 字符 → 6 tokens
  压缩率 ≈ 4.3 字符/token

中文:
  "你好，今天怎么样？" = 8 字符 → ~10 tokens (LLaMA-2)
  压缩率 ≈ 0.8 字符/token

  同一中文, LLaMA-3 (128K vocab):
  "你好，今天怎么样？" = 8 字符 → ~5 tokens
  压缩率 ≈ 1.6 字符/token → 改善 2x

影响:
  中文在 LLaMA-2 下 token 数 ~英文 5x → 推理成本 5x
  LLaMA-3 大词表显著改善多语言推理效率
```

### 3.2 对推理成本的实际影响

```
场景: 100 万字的中文文本处理

LLaMA-2 (32K vocab):
  ~125 万 tokens → 推理成本 $12.5 (假设 $10/M tokens)

LLaMA-3 (128K vocab):
  ~62.5 万 tokens → 推理成本 $6.25

节省 50% 的推理成本，仅靠更好的 tokenizer!

代码场景也类似:
  Python 代码在 LLaMA-3 下压缩率提升 ~30%
```

## 4. Tokenizer 与 KV Cache 的交互

### 4.1 Token 数 vs 字符数

```
KV Cache 大小取决于 token 数，不是字符数:
  KV per token = 2 × kv_heads × head_dim × dtype_bytes × layers

中文 "你好世界" (4 字符):
  LLaMA-2: ~8 tokens → KV Cache = 8 × per_token_size
  LLaMA-3: ~3 tokens → KV Cache = 3 × per_token_size

→ 更好的 tokenizer 直接减少 KV Cache 占用
```

### 4.2 Prefix Caching 的粒度

```
vLLM Prefix Caching 以 block (16 tokens) 为单位:
  16 tokens 在中文下:
    LLaMA-2: ~13 字符
    LLaMA-3: ~26 字符

Prefix 匹配精度受 tokenizer 影响:
  相同文本不同 tokenizer → 不同 token 序列 → 无法复用 KV Cache
  → 不同模型间 prefix caching 不可用
```

## 5. Tokenizer 实现细节

### 5.1 Hugging Face Tokenizer

```python
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 编码
tokens = tokenizer.encode("Hello, world!")
# → [15496, 11, 995, 0]

# 解码
text = tokenizer.decode(tokens)
# → "Hello, world!"

# 关键属性
print(f"词表大小: {tokenizer.vocab_size}")  # 128256
print(f"模型最大长度: {tokenizer.model_max_length}")  # 8192
print(f"特殊 tokens: {tokenizer.special_tokens_map}")
```

### 5.2 tiktoken (OpenAI)

```python
import tiktoken

# GPT-4 的 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, world!")
# → [15496, 11, 995]

# tiktoken 比 HuggingFace 快 3-6x (Rust 实现)
# 支持并行编码
```

### 5.3 推理管线中的 Tokenizer 位置

```
请求处理流程:
  1. 文本输入 → Tokenizer 编码 → token IDs
  2. token IDs → Embedding 查找 → 向量序列
  3. 向量序列 → Transformer 层 → hidden states
  4. 最后一层 → LM Head → logits
  5. logits → sampling → 下一个 token ID
  6. token ID → Tokenizer 解码 → 文本输出
  7. 重复 5-6 直到 EOS 或达到 max_tokens

Tokenizer 在步骤 1 和 6，是 CPU 操作
Transformer 在步骤 2-5，是 GPU 操作
```

## 6. 特殊 Tokens

### 6.1 常见特殊 Tokens

```
<BOS>   (Beginning of Sequence): 序列起始标记
<EOS>   (End of Sequence):       序列结束标记 (生成停止条件)
<PAD>   (Padding):               批处理时填充到相同长度
<UNK>   (Unknown):               未知 token (现代 tokenizer 通常不需要)
<MASK>  (Mask):                  MLM 任务专用 (BERT)

Chat 模板的特殊 tokens:
  [INST], [/INST]   — LLaMA-2 Chat
  <|im_start|>, <|im_end|> — Qwen / ChatML
  <|user|>, <|assistant| > — LLaMA-3
```

### 6.2 Chat 模板

```python
# LLaMA-3 的 chat template
tokenizer.apply_chat_template([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
])

# 生成:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Hello!<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>

# Chat template 的 tokens 会占用 KV Cache!
# 系统提示 + special tokens 可能耗费 20-50 tokens
# 大 batch 时这些开销不容忽视
```

## 7. 关键要点

1. **Tokenizer 直接影响推理成本** — 更好的压缩率 = 更少 tokens = 更低成本和更低延迟
2. **大词表是多语言支持的关键** — LLaMA-3 的 128K 词表让中文推理成本降低 ~50%
3. **KV Cache 按 token 计费** — token 压缩率改善直接降低显存需求
4. **Chat template 有开销** — special tokens + system prompt 占用 KV Cache 空间
5. **不同 tokenizer 之间不兼容** — 不同模型的 prefix caching 无法复用

## 参考

- 论文: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE 原始论文)
- 库: [tiktoken](https://github.com/openai/tiktoken) (OpenAI 的高性能 tokenizer)
- 库: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- 博客: [Tokenization in Large Language Models](https://medium.com/@gkamal/hands-on-with-llm-tokenization-7a1e7e8e0893)
