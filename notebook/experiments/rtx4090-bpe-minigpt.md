# BPE Tokenizer 从零实现 — RTX 4090 实测

> 2026-06-05 | 工具: `tools/bpe_minigpt_train.py` | RTX 4090 24GB
> 架构: LLaMA-like (RMSNorm + SwiGLU + Pre-norm + Weight Tying)

## 1. 实验设计

**核心问题**: 字符级 tokenizer (vocab=40) vs BPE tokenizer (vocab=512-2048) 对训练的影响?

**BPE 算法**:
1. 将文本转为字节 (base vocab = 256)
2. 统计所有相邻字节对频率
3. 合并最高频对为新 token
4. 重复直到目标 vocab size
5. 推理时: 从左到右贪心应用 merges

**语料**: 7255 字符英文故事 (Lucy & Snowball)
**模型**: 0.14M - 6.46M params, 4层 Transformer

## 2. BPE Tokenizer 分析

### 2.1 压缩率

| Vocab Size | Tokens | 压缩率 | 使用的 unique tokens |
|-----------|--------|--------|---------------------|
| 512 | 3664 | 1.98x | 269/512 |
| 1024 | 2825 | 2.57x | 545/1024 |
| 2048 | 2479 | 2.93x | 554/1370 |
| Char (45) | 7255 | 1.00x | 45/45 |

**发现**: vocab 越大压缩率越高, 但边际收益递减 (1.98→2.57→2.93)

### 2.2 学到的 Merges

前 10 个 merge (vocab=512):
```
he (210), the (119), re (110), in (104), an (90),
er (72), ed (66), en (63), and (62), ll (61)
```
→ 完全符合英语频率统计! 最常见的字母对和子词被优先学习.

后期的 merge:
```
for (6), children (5), their (4)
```
→ 更大的 merge 形成完整单词.

## 3. BPE vs Char 训练对比

**关键**: raw loss 不可直接比较!
- BPE vocab=512: 初始 loss ≈ log(512) = 6.24
- Char vocab=45: 初始 loss ≈ log(45) = 3.81

**公平比较**: 使用 BPC (bits-per-character)

| Tokenizer | Val Loss | PPL | BPC | Compression | Time |
|-----------|----------|-----|-----|-------------|------|
| BPE (512) | 6.087 | 439.9 | **3.99** | 2.20x | 24.3s |
| Char (45) | 3.710 | 40.9 | **5.35** | 1.00x | 23.8s |

**BPE 优势: BPC 低 25%!** (3.99 vs 5.35)

### 为什么 BPE 的 BPC 更低?

```
BPC = loss / ln(2) / compression_ratio

BPE: 6.087 / 0.693 / 2.20 = 3.99 bits/char
Char: 3.710 / 0.693 / 1.00 = 5.35 bits/char

原因: BPE 每个 token 编码更多信息 (2.2 chars/token)
→ 每个预测步骤覆盖更多文本
→ 模型可以学到更高层次的模式 (词级别 vs 字符级别)
```

## 4. Vocab Size 对训练的影响

| Vocab | Val Loss | BPC (bits/char) | Compression | Tokens |
|-------|----------|-----------------|-------------|--------|
| 512 | 6.019 | 3.94 | 2.20x | 3297 |
| 1024 | 6.767 | 3.42 | 2.85x | 2542 |
| 2048 | 7.105 | 3.19 | 3.25x | 2231 |

**发现**:
- 更大 vocab → 更高 raw loss (选择更多)
- 但 BPC 持续下降 (3.94→3.42→3.19)!
- **最优 vocab 取决于语料大小**: 小语料 (7K chars) vocab=512 已足够
- 实际 LLM: GPT-2 用 50K vocab, LLaMA 用 32K

## 5. 模型缩放 (BPE, vocab=512)

| Model | Params | Val Loss | BPC | Time |
|-------|--------|----------|-----|------|
| tiny | 0.14M | 6.227 | 4.08 | 11.2s |
| small | 0.87M | 6.121 | 4.01 | 18.3s |
| medium | 2.78M | 5.947 | 3.89 | 24.2s |
| large | 6.46M | 5.606 | 3.67 | 30.6s |

**观察**: 所有模型 loss 都很高 (>5.6), 因为语料太小 (3.3K BPE tokens)
→ 过拟合问题与上次实验一样严重

## 6. 生成质量

两个 tokenizer 的生成都是乱码, 因为:
- 7255 chars = 3.3K BPE tokens, 严重不足
- Chinchilla 建议每参数 20 tokens, 当前差 ~1000x
- 需要 100K+ 字符语料才能产生有意义的文本

BPE 生成的乱码包含了一些可识别的子词片段 ("enchanted", "Every", "Snowball"),
而 Char 生成的完全是随机字符 — 这说明 BPE 至少学到了一些有意义的表示.

## 7. 核心学习

1. **BPE 的本质**: 学习数据驱动的子词单元, 自动平衡 vocab 大小和序列长度
2. **BPC 是正确的比较指标**: 不同 tokenizer 不能直接比较 loss
3. **BPE 在任何规模都优于字符级**: 即使用 7K 字符, BPC 仍低 25%
4. **数据量是瓶颈**: 即使最好的 tokenizer 也救不了太小的语料
5. **BPE 算法简单但有效**: 仅需频率统计 + 贪心合并
6. **Byte-level 的优势**: 不需要预定义字符集, 天然支持任何语言/Unicode
7. **Real-world**: GPT-2/3/4 使用 byte-level BPE, 50K+ vocab, ~4 chars/token

## 8. 大语料实验 (518K chars)

用合成语料 (300个儿童故事模板, 517,589 chars) 重新训练:

### 8.1 BPE vs Char — 大语料

| Tokenizer | Val Loss | PPL | BPC | Compression | Time |
|-----------|----------|-----|-----|-------------|------|
| BPE (512) | 5.589 | 267.3 | **3.89** | 1.87x | 116s |
| Char (55) | 3.614 | 37.1 | **5.21** | 1.00x | 115s |

**BPE 优势: 25.4%** — 与小语料一致, 说明 BPE 的优势是结构性的.

### 8.2 模型缩放 (BPE vocab=512)

| Model | Params | Val Loss | BPC | PPL | Time |
|-------|--------|----------|-----|-----|------|
| tiny | 0.15M | 6.220 | 4.33 | 502 | 23s |
| small | 0.89M | 6.101 | 4.25 | 447 | 40s |
| medium | 2.80M | 5.651 | 3.93 | 284 | 117s |
| large | 6.49M | 5.287 | 3.68 | 198 | 222s |
| xlarge | 12.54M | **4.949** | **3.44** | 141 | 334s |

**清晰的 scaling 趋势!** 参数 10x → BPC -20%

### 8.3 Vocab Size 效果 (大语料)

| Vocab | BPC | PPL | Compression |
|-------|-----|-----|-------------|
| 512 | 3.99 | 308 | 2.07x |
| 1024 | 3.28 | 528 | 2.75x |
| 2048 | **2.93** | 802 | 3.29x |

**更大 vocab → BPC 更低!** 但 PPL 上升 (因为预测更难).

### 8.4 关键发现

1. **Scaling 规律清晰**: xlarge(12.5M) 比 tiny(0.15M) BPC 低 20%
2. **BPE 优势稳定**: 大小语料都是 ~25% BPC 优势
3. **生成质量仍差**: 模型未收敛, BPE 的 byte-level 输出包含无效 UTF-8
4. **数据瓶颈**: 518K chars / 250K BPE tokens 仍远不够 (Chinchilla: 12.5M params 需 250M tokens)

## 9. 下一步

- [ ] 用 TinyStories 真实数据 (需要更大磁盘空间)
- [ ] 训练到收敛 (10000+ steps)
- [ ] 实现 BPE pre-tokenization (regex split)
- [ ] 在 BPE 模型上做 SFT + DPO 实验
