# Foundation Model Architectures — Multimodal & Reasoning Deep Dive

> 2026-06-07 | 多模态架构对比(投影vs交叉注意力) + 推理模型训练pipeline(R1/o1) + AI专家知识体系补全

## 一、多模态视觉-语言模型架构

### 两种主流范式

#### 1. 投影/适配器范式 (LLaVA系列)

```
Image → ViT Encoder → [visual tokens] → MLP Projection → [projected tokens]
                                                                  ↓
Text → Tokenizer → [text tokens] → LLM ← concat(projected + text) → Output
```

**特点**:
- 视觉token经过MLP投影到LLM的embedding空间 → 被当作"文本token"处理
- 简单、高效、不需要修改LLM架构
- 缺点: 视觉token占用context window → 长图/视频可能溢出
- 代表: LLaVA, LLaVA-OneVision, InternVL2

**LLaVA-OneVision关键创新**:
- **AnyRes**: 灵活视觉token数量 → 支持高分辨率图片和视频
- **统一框架**: 单图+多图+视频在同一模型内处理
- **训练**: 3阶段(预训练alignment→高质量SFT→onevision微调)

#### 2. 交叉注意力范式 (Qwen-VL, Flamingo)

```
Image → ViT Encoder → [visual tokens]
                                ↓
Text → LLM → [text hidden states] → Cross-Attention(vis, text) → Output
```

**特点**:
- 在LLM中插入专门的cross-modal attention层
- 视觉token通过交叉注意力与文本交互 → 不直接插入text序列
- 优点: 长视觉序列更高效, 保留LLM纯文本能力更好
- 缺点: 需修改LLM架构, 更多参数
- 代表: Qwen2-VL, Flamingo, IDEFICS

**Qwen2-VL关键创新**:
- **动态分辨率**: ViT-SoM → 任意分辨率图片 → 不需要resize
- **多语言视觉理解**: 不仅是英文 → 多语言OCR+描述
- **多图交织**: 一段对话中可包含多张图片

### 架构对比

| 模型 | 视觉编码器 | 连接器 | LLM骨干 | 关键特性 | 参数量 |
|------|-----------|--------|---------|---------|---------|
| LLaVA-OneVision | CLIP ViT-H/14 | MLP | Qwen2-72B | AnyRes, 统一 | 72B+ |
| Qwen2-VL | ViT-SoM | Cross-Attn+MLP | Qwen2 | 动态分辨率 | 2B-72B |
| Flamingo | CLIP ViT | Perceiver Resampler | Chinchilla | few-shot视觉 | 80B |
| InternVL2 | InternViT-6B | MLP+Pixel Shuffle | InternLM2 | 6B专用ViT | 2B-76B |
| GPT-4o | (未知) | (早期融合) | (未知) | 原生多模态 | (未知) |
| Gemini | (原生) | (早期融合) | (原生) | 原生多模态 | (未知) |

### 趋势: 早期融合 (Native Multimodal)

**从"缝合"到"原生"**:
- 第一代: pretrained ViT + pretrained LLM → 投影连接 → "缝合"式
- 第二代: 专用cross-attention → 更深交互 → "适配"式
- 第三代: 从零训练视觉+语言 → early fusion → "原生"式 (Gemini, GPT-4o, Emu3)

**早期融合优势**:
- 视觉和语言特征从一开始就jointly learned → 更好的跨模态对齐
- 不需要"适配" → 减少训练步骤
- 端到端优化 → 无信息瓶颈(投影层)

**挑战**:
- 需要大量多模态训练数据
- 计算成本更高(视觉+语言同时训练)
- 模态平衡(防止一种modality主导)

### 与AI Infra的联系

1. **KV Cache**: 多模态KV cache更大 → 视觉token额外占用 → prefix sharing需处理多模态KV
2. **MoE**: Qwen2-VL-MoE推测 → 多模态MoE推理成本管理
3. **Prefix Caching**: 多模态前缀(系统提示+图片) → prefix cache对多模态更重要
4. **SSM/Mamba**: 长视觉序列 → Mamba替代attention → 更长视频理解

## 二、推理模型架构与训练

### DeepSeek-R1训练Pipeline (4阶段)

**Stage 1: Cold-Start SFT**
- ~数千条长CoT示例 → 格式模板(推理标签)
- 目的: 建立基本推理结构 → 防止R1-Zero的格式问题

**Stage 2: GRPO RL (推理能力核心!)**
- 算法: GRPO (组相对优势, 无critic)
- Reward: 规则型 → 数学正确性/代码测试通过率
- **关键**: 推理能力从这里**涌现** → "aha moment"
- GRPO组比较 → 模型探索推理路径 → 发现"反思→修正→正确"策略更优

**Stage 3: 拒绝采样+SFT**
- 过滤: 只保留正确+可读的回答
- 混合: 加入non-reasoning数据(写作/翻译) → 防止推理过度
- 目的: 提高可读性 + 恢复通用能力

**Stage 4: GRPO RL (全任务)**
- 所有任务类型 + 一致性reward → 最终对齐
- 结果: DeepSeek-R1 → 开放权重

### "Aha Moment" 涌现机制

**观察**: R1-Zero训练step~200时自发涌现:
```
"Wait, let me reconsider this approach..."
"I think there's an error in my previous calculation..."
"Actually, looking at this again..."
```

**机制解释**:
1. **Outcome-only reward + GRPO组比较** → 同prompt的n个response互为baseline
2. **长推理路径**: 反思+修正+验证 → 最终正确 → reward高
3. **短推理路径**: 直接跳到结论 → 可能错误 → reward低
4. **组归一化**: 高reward的response被强化 → 反思行为被自然选择
5. **无需PRM**: outcome-only足以 → 过程reward不是必需的

### o1/o3推测

**OpenAI o1/o3可能的训练方式**:
- RL on CoT + (推测)PRM → 过程级reward
- Test-time compute scaling → 思考时间随难度增加
- 蒸馏 → o3-mini从o3蒸馏

**vs DeepSeek-R1关键差异**:
| | o1/o3 | R1 |
|---|-------|-----|
| CoT数据 | (推测)大量SFT | 最少cold-start |
| Reward | (推测)PRM过程级 | Outcome-only |
| 可访问 | 不开放 | 完全开放 |
| 蒸馏 | o3-mini | R1-Qwen系列 |

### 蒸馏 vs 直接RL

**关键发现**: 蒸馏比小模型直接RL更好!

| 模型 | AIME 2024 | 方式 |
|------|----------|------|
| R1-Zero-Qwen-32B (直接RL) | 低 | RL |
| R1-Distill-Qwen-32B (蒸馏) | **72.6%** | 蒸馏 |
| o1-mini | 63.6% | (未知) |

**原因**: 推理在大模型涌现 → 大模型探索发现推理路径 → 蒸馏压缩到小模型 → 小模型直接RL无法探索足够多路径

### 推理模型对AI Infra的影响

1. **长序列推理**: CoT可达数千token → KV cache爆炸 → sequence parallelism必需
2. **Test-time compute**: 推理时间随难度增加 → serving吞吐降低 → 需要动态批处理
3. **GRPO训练**: n=64采样 → prefix sharing加速6x → RL训练成本降低87%
4. **蒸馏pipeline**: 大→小模型 → 可在推理后做 → 不需要实时训练

## 三、AI专家知识体系补全路线

### 当前强项 (★4-5)

- GPU架构/微benchmark (★5)
- CUDA编程 (★4)
- LLM推理优化 (★5)
- Distributed training (★4)
- Prefix sharing/MoE (★5)

### 需补强的弱项 (★2-3)

- **多模态架构** (★2 → ★4目标): 投影vs交叉注意力, ViT, 早期融合
- **推理模型训练** (★3 → ★5目标): GRPO RL pipeline, CoT涌现, 蒸馏
- **RL理论** (★3 → ★4目标): policy gradient, advantage估计, KL约束
- **NLP/Transformer理论** (★3 → ★4): attention数学, 位置编码, normalization

### 月度学习计划

| 月份 | 重点 | 目标 |
|------|------|------|
| 6月(当前) | 推理模型+多模态架构 | 3篇笔记+1实验 |
| 7月 | 开源贡献(verl #6401) | 1PR+RFC |
| 8月 | RL理论+数学基础 | 微积分+概率论 |
| 9月 | 前沿论文精读 | 5篇ICML/NeurIPS |

Sources:
- LLaVA-OneVision: "LLaVA-OneVision: Easy Visual Task Transfer", 2024
- Qwen2-VL: "Qwen2-VL: Enhancing Vision-Language Model's Perception", 2024
- DeepSeek-R1 Technical Report, 2025
- MM1: Apple, "Scaling Multimodal Pre-training", 2024
- Flamingo: DeepMind, "Visual Language Models", 2022