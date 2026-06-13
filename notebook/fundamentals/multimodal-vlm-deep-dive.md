# Multimodal AI (Vision-Language) Deep Dive — VLM架构演进(CLIP双编码器对比损失→LLaVA投影层+指令微调→InternVL InternViT-6B+动态分辨率→Qwen2-VL原生动态分辨率→CogVLM视觉专家模块→Gemini原生多模态) + Vision Encoder设计(patch=14+多分辨率+位置嵌入) + Projection策略(Linear/MLP/Q-Former/Visual Expert/对比) + 三阶段训练(预训练→SFT→DPO) + RTX 4090 VLM推理(7B INT4+INT8 ViT→24GB可行+ViT 15-30ms+FlashAttention-2必须) + 2026趋势(原生多模态+Agent VLM+Token压缩+Video理解)

> 2026-06-14 | VLM架构演进深度分析: 从CLIP(2021)双编码器对比学习→LLaVA(2023)MLP投影+指令微调→InternVL(2024)InternViT-6B+动态分辨率448px→Qwen2-VL(2025)动态分辨率+多语言→CogVLM视觉专家模块→Gemini(2023-2025)原生多模态MoE → 4代架构演进! Vision Encoder从ViT-B/32→ViT-L/14→InternViT-6B→SigLIP-2→原生 → Patch=14标准+多分辨率+动态分辨率 → Projection从linear→MLP→Q-Former→Visual Expert → MLP主流Q-Former over-engineered! → 三阶段训练pretrain→SFT→DPO → RTX 4090 7B INT4+INT8 ViT→7-8GB→24GB可行 → FlashAttention-2必须 → 2026趋势: 原生多模态+Agent VLM+Token压缩+Video
> 关联: ai-expert-knowledge-map-gap-analysis.md(VLM gap★→★★★★), scaling-laws-deep-dive.md(inference scaling), ai-safety-guardrails-production-deep-dive.md(VLM安全部署), data-pipeline-curation-deep-dive.md(VLM训练数据)
> 参考: CLIP(Radford et al. 2021), BLIP-2/Li et al. 2023, LLaVA(Liu et al. 2023-2024), InternVL(Shanghai AI Lab 2024-2025), Qwen2-VL(Alibaba 2024-2025), CogVLM(Tsinghua 2023-2024), Gemini(Google DeepMind 2023-2025), SigLIP(Zhai et al. 2023-2025), NaViT(Dehghani et al. 2024)

## 0. 核心定律: VLM=4代架构演进 → CLIP→LLaVA→InternVL→Gemini → 趋势=原生多模态!

```
VLM架构演进(4代):

Gen 1 — CLIP(2021): 双编码器+对比损失 → 图文对齐 → zero-shot!
  → → ViT+Text Transformer → cosine similarity → 分开编码 → 不生成!

Gen 2 — LLaVA(2023): ViT编码器+投影+LLM解码器 → 图文生成 → 指令微调!
  → → → Linear/MLP projection → 视觉token注入LLM → 生成式!

Gen 3 — InternVL/Qwen2-VL(2024-2025): ViT-6B+动态分辨率+原生多模态 → 高分辨率理解!
  → → → → → 动态分辨率 → 448px tiles → 高分辨率任务(OCR/图表)→突破!

Gen 4 — Gemini(2023-2025): 原生多模态 → 从头训练 → 无投影层 → 端到端!
  → → → → → → → Interleaved训练 → text+image+video+audio → 全模态原生!

→ → → → → → → → 趋势: projection→native → 固定分辨率→动态 → 单模态→全模态 → 2026原生多模态!
```

## 1. CLIP — 对比学习双编码器(Gen 1)

```
### 1.1 CLIP架构

CLIP = Contrastive Language-Image Pre-training → 图文对齐 → zero-shot分类!

架构:
  → Image Encoder: ViT-B/32(151M) / ViT-B/16(151M) / ViT-L/14(428M)
  → → → patch_size=32→7×7=49tokens / 16→14×14=196 / 14→16×16=256
  → → → → → ViT-L/14最优 → 256 tokens → 14×14 patch → 最细粒度!
  → → Text Encoder: Transformer(12层, 77 token max) → 文本编码!

对比损失:
  → N个图文pair → 正pair=N → 负pair=N²-N → 最大化正pair cosine similarity!
  → → L = -log(exp(sim(I_i,T_i)/τ)/Σexp(sim(I_i,T_j)/τ)) → InfoNCE!
  → → → → τ=learnable temperature → 控制分布锐度 → 训练稳定!

训练数据:
  → 400M image-text pairs → 从网络收集 → 规模决定性能!

### 1.2 CLIP zero-shot分类

Zero-shot = 不训练 → 直接分类 → 通用性!

流程:
  → 给类别名 → "a photo of [class]" → text encoder → text embedding!
  → → 给图片 → image encoder → image embedding!
  → → → → → cosine similarity → 最高=预测类别 → 分类!

效果:
  → ImageNet zero-shot → 76.2% → 接近ResNet-50监督训练(76.6%) → 无训练!
  → → → → → 关键: 对比学习 → 图文对齐 → 通用性 → 不需下游训练!

### 1.3 CLIP局限与2025扩展

CLIP局限:
  → 双编码器 → 只能检索/分类 → 不能生成 → 无对话能力!
  → → 固定分辨率 → 224px → 高分辨率任务差 → OCR/图表弱!
  → → → → → 对比损失 → 只对齐语义 → 不对齐细粒度 → 理解不深!

2025扩展:
  → SigLIP(2023-2025): sigmoid loss → 替代softmax → batch独立 → 更高效!
  → → → → → → → 关键: sigmoid(I,T) → 不需同步全局 → 支持更大batch!
  → → EVA-CLIP: 更强ViT → 更好预训练 → 更高质量!
  → → → LongCLIP: 长文本支持 → 扩展到77→248 tokens → 更详细描述!
  → → → → MobileCLIP: 轻量ViT → 移动部署 → hybrid架构!
  → → → → → UniCL: 统一对比+生成 → 检索+生成 → 双重能力!

→ → → → → → → → 结论: CLIP=对齐基础 → 但不能生成 → 需LLM组合(LLaVA)!
```

## 2. LLaVA — 投影层+指令微调(Gen 2)

```
### 2.1 LLaVA架构

LLaVA = Large Language-and-Vision Assistant → ViT+Projection+LLM → 生成式VLM!

架构:
  → Vision Encoder: CLIP ViT-L/14 → 256 visual tokens → 固定!
  → → Projection Layer: Linear→MLP(LLaVA-1.5) → 视觉→语言空间!
  → → → → → → → → → → → → → MLP: W1→ReLU→W2 → 2层 → 更丰富 → ~8M参数!
  → → → → → → → → → → → → → → → → → Linear: 单层W → 最简 → ~2M → 但不够好!
  → → → → → → → → → → → → → → → → → → → → LLM Decoder: Vicuna-7B/13B → 生成文本 → 自回归!

Visual Token注入:
  → ViT → 256 visual tokens → projection → 256 language tokens → 前缀注入!
  → → → → → 替换: <image> → 256 projected tokens → LLM处理 → 生成!

### 2.2 LLaVA演进

| 版本 | ViT | Projection | Visual Tokens | LLM | Resolution |
|------|-----|------------|---------------|-----|------------|
| LLaVA(v1) | CLIP ViT-L/14 | Linear | 256 | Vicuna-7B/13B | 224px |
| LLaVA-1.5 | CLIP ViT-L/14 | 2-layer MLP | 576(multi-crop) | Vicuna-7B/13B | 336px |
| LLaVA-NeXT | CLIP ViT-L/14 | 2-layer MLP | dynamic | Mistral/Qwen | AnyRes |
| LLaVA-OneVision | SigLIP ViT | 2-layer MLP | dynamic(AnyRes) | Qwen2-7B/72B | AnyRes+video |

关键改进:
  → Linear→MLP: 2层MLP投影 → 更丰富 → 性能提升5-10%!
  → → 224→336px: 更高分辨率 → 更多visual tokens → 更精细!
  → → → → → AnyRes: 动态分辨率 → 根据图片大小自适应 → 2025标准!
  → → → → → → → Video: 扩展到视频 → 多帧 → temporal understanding!

### 2.3 指令微调数据

LLaVA指令微调:
  → GPT-4生成 → 基于图片描述 → instruction-following → 80K→1M+!
  → → → → → 类型: 对话/描述/推理/编码 → 多任务 → 通用!
  → → → → → → → 2025改进: 多模态CoT → 推理指令 → chain-of-thought → 更深理解!
  → → → → → → → → → 数据质量>数量 → curated→精选 → 1M→精选→更优!

### 2.4 LLaVA核心洞察

核心洞察:
  → 简单投影层足够! → 不需Q-Former → MLP就够了 → 令人惊讶!
  → → → 原因: LLM本身就强 → 只需要把视觉信息"翻译"到语言空间 → MLP就够!
  → → → → → 关键: 投影层简单 → 但LLM强 → 组合=强VLM → 不需复杂adapter!
  → → → → → → → → → 但: 高分辨率 → 千tokens → 需压缩策略 → 下一问题!

→ → → → → → → → → → → → → → → → → 结论: MLP=2025主流 → 简单+有效 → ViT足够强时MLP够了!
```

## 3. InternVL — InternViT-6B+动态分辨率(Gen 3)

```
### 3.1 InternVL架构

InternVL = InternViT-6B + InternLM2 + dynamic resolution → 高分辨率VLM!

架构:
  → Vision Encoder: InternViT-6B → 6B参数ViT → 比CLIP ViT-L(428M)大14x!
  → → → → → → → → → 6B=更理解 → 更细粒度 → OCR/图表→突破!
  → → → → → → → → → → → MLP Projection → 视觉→语言 → 简单但有效!
  → → → → → → → → → → → → → LLM Decoder: InternLM2-8B/20B → 强语言 → 生成!

### 3.2 动态分辨率机制

InternVL动态分辨率 → 最重要的创新 → 高分辨率任务突破!

机制:
  → 图片 → 按aspect ratio → 分成N个tiles → 每tile 448×448 → 处理!
  → → → → → 例: 1024×512 → 2 tiles → 448×448 each → 再拼接!
  → → → → → → → 例: 2048×1024 → 4-6 tiles → 更多细节 → OCR/图表!
  → → → → → → → → → 每tile → InternViT-6B → token → 拼接 → 全图representation!

  → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关键: thumbnail view → 1个全局+ N个局部 → 全局+局部 → 全图理解!

vs 固定分辨率:
  → CLIP/LLaVA → 224/336px → 下采样 → 丢细节 → OCR差!
  → → → → → InternVL → 448px×N tiles → 保留细节 → OCR/图表→突破!

InternVL 2.5成绩:
  → MMMU: 84.3% → 开源最优 → 动态分辨率关键!
  → → → OCRBench: 领先 → 高分辨率→OCR→突破!
  → → → → → MathVista: 领先 → 图表理解→突破!

### 3.3 InternViT-6B设计

InternViT-6B vs CLIP ViT-L/14:
  → 参数: 6B vs 428M → 14x更大 → 更理解!
  → → → Patch: 14×14 → 标准现代 → 精细!
  → → → → → Resolution: 448px dynamic → vs 224px fixed → 2x+!
  → → → → → → → 训练: 原生多模态 → vs 对比学习 → 更丰富!

→ → → → → → → → 结论: InternViT-6B=更大+动态+原生 → 高分辨率任务突破!

### 3.4 InternVL版本演进

| 版本 | ViT | LLM | 分辨率 | 特色 | MMMU |
|------|-----|-----|--------|------|------|
| InternVL1 | InternViT-6B | InternLM2-7B/20B | 448px | 原生对齐 | ~58% |
| InternVL1.5 | InternViT-6B | InternLM2-8B/20B | 448px动态 | progressive训练 | ~70% |
| InternVL2 | InternViT-6B | 多LLM(2B→76B) | 448px动态 | 多尺寸 | ~78% |
| InternVL2.5 | InternViT-6B-448px-V2 | 多LLM | 448px动态 | 改进ViT | 84.3% |
| InternVL3(2025) | InternViT-6B | 多LLM | 448px动态 | 原生多模态训练 | ~85%+ |

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关键: 动态分辨率+大ViT=持续改进 → OCR/图表领先 → 2025开源最优!
```

## 4. 其他VLM架构 — Qwen2-VL / CogVLM / Gemini

```
### 4.1 Qwen2-VL

Qwen2-VL = ViT-L/14 + Adapter + Qwen2-LLM + 动态分辨率 → 多语言!

架构:
  → Vision Encoder: ViT-L/14 → 动态分辨率 → 多语言支持!
  → → → → → Adapter: ViT→LLM → 连接 → 简单!
  → → → → → → → LLM: Qwen2-7B/72B → 强语言 → 多语言!

特点:
  → Naive动态分辨率 → 不需要固定 → 自适应 → 类InternVL!
  → → → 多语言OCR → 中文+英文+多语言 → 全球化!
  → → → → → Video理解 → temporal encoding → 视频 → 多帧!

### 4.2 CogVLM — 视觉专家模块

CogVLM = Visual Expert Module → 在LLM内部加视觉attention → 不是简单投影!

架构:
  → Vision Encoder: ViT → 视觉特征 → 标准!
  → → → → → Visual Expert: 在LLM每层加dedicated attention → 视觉专用参数!
  → → → → → → → → → 共享: FFN+其他attention → 与LLM共享 → 但视觉attention独立!

设计哲学:
  → 避免"modality alignment gap" → 简单投影→信息丢失 → 视觉需要自己的attention!
  → → → → → 视觉专家 → 独立attention → 不丢失 → 更好理解!

vs LLaVA:
  → LLaVA: MLP projection → 简单 → 但视觉信息压缩 → 可能丢失!
  → → → CogVLM: Visual expert → 独立attention → 不压缩 → 但参数更多!

### 4.3 Gemini — 原生多模态(Gen 4)

Gemini = Natively Multimodal → 从头训练 → 无投影层 → 端到端 → 最先进!

架构:
  → 无单独ViT → 无投影 → 原生 → 从头设计 → 全模态共享!
  → → → MoE(Mixture-of-Experts) → 不同模态→不同expert → 专业化!
  → → → → → Long context → 1M tokens → 图片+视频+文本 → 超长!

训练:
  → Interleaved → text+image+video+audio+code → 混合 → 全模态!
  → → → → → 不需要单独align → 原生对齐 → 最自然!

Gemini 1.5/2.0(2025):
  → 1.5 Pro: MoE+1M context → 超长 → 全模态理解!
  → → → 2.0 Flash: 更快+agentic → GUI agent → 视觉行动!
  → → → → → → → 关键: 原生多模态 → 不需投影 → 端到端 → 2026趋势!

### 4.4 VLM架构对比

| 模型 | ViT | 投影 | 分辨率 | 原生多模态 | 特色 |
|------|-----|------|--------|------------|------|
| CLIP | ViT-L/14(428M) | 对比损失 | 224px固定 | ❌ | zero-shot检索 |
| LLaVA-1.5 | CLIP ViT-L/14 | MLP(2层) | 336px固定 | ❌ | 简单投影够用 |
| LLaVA-OV | SigLIP ViT | MLP | AnyRes动态 | 部分 | video+anyres |
| InternVL2.5 | InternViT-6B | MLP | 448px动态 | 部分 | OCR/图表突破 |
| Qwen2-VL | ViT-L/14 | Adapter | 动态 | 部分 | 多语言+视频 |
| CogVLM | ViT | Visual Expert | 固定 | 混合 | 视觉独立attention |
| Gemini | 原生 | 无需 | 原生 | ✅ | 端到端多模态 |

→ → → → → → → 趋势: 固定→动态→原生 → 投影→adapter→expert→无投影 → 2026原生多模态!
```

## 5. Vision Encoder设计 — ViT选择与优化

```
### 5.1 Patch Size选择

Patch size决定visual token数量 → 影响精度和计算:

  ps=32: 224px→7×7=49tokens → 太粗 → 信息不足 → CLIP-B/32最差!
  → ps=16: 224px→14×14=196tokens → 标准 → 足够 → legacy ViT!
  → → ps=14: 224px→16×16=256tokens → 现代 → 更精细 → 2025标准!
  → → → → → ps=14+384px: 27×27=729tokens → 精细+多 → SigLIP-2!
  → → → → → → → ps=14+896px: 64×64=4096tokens → 太多 → 需压缩!

→ → → → → → → → 结论: ps=14是2025标准 → 但高分辨率需token压缩!

### 5.2 分辨率策略

分辨率策略演进:
  → 2021(CLIP): 224px固定 → 太低 → OCR/图表差!
  → → 2023(LLaVA-1.5): 336px → 稍好 → 但仍固定!
  → → → 2024(InternVL): 448px动态 → tile-based → OCR突破!
  → → → → → 2025(SigLIP-2/NaViT): 多分辨率训练 → 224+384+512+896 → 灵活!

动态分辨率(InternVL):
  → 图片→按aspect ratio→N tiles→448×448→处理→拼接→全图!
  → → → → → 关键: thumbnail view → 1个全局+ N个局部 → 全局+局部!
  → → → → → → → 保留细节 → OCR/图表 → 高分辨率任务 → 突破!

多分辨率训练(SigLIP-2):
  → 训练时随机选分辨率 → 224/384/512/896 → 模型适应任意分辨率!
  → → → → → 位置嵌入: learned+bicubic interpolation → 不同分辨率→插值→适应!

### 5.3 位置嵌入设计

位置嵌入 → ViT如何知道patch位置 → 影响分辨率灵活性:

1. Learned position embedding(原始ViT):
   → 每个位置一个learned vector → 固定grid → 改分辨率→插值!
   → → → 插值: bicubic → 224→336 → 196→576 → 近似 → 不完美!

2. 2D Sinusoidal embedding:
   → sin/cos → 数学公式 → 自然扩展 → 任意grid → 无插值!
   → → → → → 优势: 自然适应 → 但性能略低于learned → trade-off!

3. Factorized 2D learned(NaViT):
   → 分解为x+y → 分别learned → 插值更自然 → 灵活!
   → → → → → → → 优势: 分解 → 自然插值 → 多分辨率 → InternVL用类似!

4. No explicit PE(探索):
   → 靠patch grid顺序 → 无位置 → 最灵活 → 但性能略低!

→ → → → → → → → 结论: Learned+interpolation→2025主流 → Factorized→未来 → Sinusoidal→备选!
```

## 6. Projection策略 — 视觉→语言的连接方式

```
### 6.1 四种Projection策略

Projection = 把视觉信息翻译到语言空间 → 4种策略 → 从简到复杂!

1. Linear Adapter(LLaVA v1):
   → 单层linear → W×visual_tokens → 直接映射 → 最简单!
   → → → 参数: ~2M → 最少 → 但信息保留最差 → 压缩!
   → → → → → 效果: 足用但不够好 → LLaVA-1.5改用MLP!

2. MLP(LLaVA-1.5/InternVL):
   → 2层MLP → W1→ReLU→W2 → 更丰富 → 2025主流!
   → → → 参数: ~8M → 少 → 但信息保留好 → 不压缩token数!
   → → → → → 效果: 简单+有效 → 2025主流 → 大多数VLM用MLP!
   → → → → → → → 关键: 不压缩 → 保留所有visual tokens → spatial info完整!

3. Q-Former(BLIP-2):
   → 32个learnable queries → cross-attention → 与image features交互!
   → → → → → 输出: 固定32 tokens → 不管图片多大 → 压缩!
   → → → → → → → 参数: ~188M → 多 → 但压缩到固定 → 灵活!
   → → → → → → → → → 优势: 固定输出 → 不受分辨率影响 → video友好!
   → → → → → → → → → → → 劣势: 压缩 → 丢失空间信息 → OCR/图表差!

4. Visual Expert(CogVLM):
   → 在LLM内部加dedicated visual attention → 视觉专用参数!
   → → → → → → → → → 不压缩 → 但视觉有独立attention → 不丢失!
   → → → → → → → → → → → → → 参数: 更多 → 但理解最好!

### 6.2 策略对比

| 策略 | 参数 | Token数 | 空间信息 | 训练成本 | 适用场景 |
|------|------|---------|----------|----------|----------|
| Linear | ~2M | 全部256+ | ✅保留 | 最低 | 简单场景 |
| MLP | ~8M | 全部256+ | ✅保留 | 低 | 2025主流 |
| Q-Former | ~188M | 固定32 | ❌压缩 | 中 | video/长序列 |
| Visual Expert | ~100M+ | 全部 | ✅保留 | 高 | 精细理解 |

### 6.3 2025趋势: MLP主流 → Q-Former被认为over-engineered

关键发现:
  → MLP性能≥Q-Former → 当ViT足够强(SigLIP/InternViT-6B) → MLP就够了!
  → → → 原因: ViT已经理解视觉 → MLP只需翻译到语言空间 → 不需复杂交互!
  → → → → → Q-Former优势只在: video → 多帧 → 需压缩 → token budget紧!
  → → → → → → → → → 结论: 大多数VLM用MLP → video才用Q-Former → 2025简单化!
```

## 7. VLM三阶段训练Pipeline

```
### 7.1 Stage 1: Pretraining(图文对齐)

目标: 视觉→语言空间对齐 → 投影层学习 → 基础!

训练配置:
  → Frozen LLM + Frozen ViT → 只训练Projection → 参数最少!
  → → → → → 数据: 100M-1B image-text pairs → caption-style → 大规模!
  → → → → → → → 损失: autoregressive → predict text from image → 语言生成!

关键:
  → VILA论文证明 → Pretraining质量比想象的更重要!
  → → → Interleaved image-text → 不只用caption → 交错数据 → 更好!

### 7.2 Stage 2: Instruction Tuning(SFT)

目标: 指令跟随+推理 → 多任务 → 通用!

训练配置:
  → Unfreeze LLM(partial/full) → ViT frozen或LoRA → Projection训练!
  → → → → → 数据: 10K-100K高质量指令 → GPT-4V生成 → 多任务!
  → → → → → → → 类型: 对话+描述+推理+编码+数学 → 多任务 → 通用!

2025改进:
  → 多模态CoT → 推理指令 → 逐步 → chain-of-thought → 更深理解!
  → → → → → 数据质量>数量 → curated→精选 → 1M→精选→更优!

### 7.3 Stage 3: Preference Alignment(RLHF/DPO)

目标: 减少幻觉 → 提高有用性 → 安全 → 对齐!

2025主流: DPO > PPO → 更简单→更稳定 → 不需reward model!

训练配置:
  → Full model fine-tuning → preference pairs → 人类/AI偏好!
  → → → → → 数据: VLFeedback/Silkie → 偏好对比 → 减少幻觉!
  → → → → → → → 目标: 减少hallucination → 视觉groundedness → 安全!

关键:
  → VLM幻觉特别严重 → "看到"不存在的东西 → 需对齐!
  → → → → → RLHF-V: 专门VLM RLHF → 减少幻觉 → 视觉约束!

### 7.4 RTX 4090训练策略

RTX 4090 VLM训练(24GB HBM):
  → 7B LLM + ViT-L/14 → bf16 → ~18GB → 太紧 → 需量化/LoRA!
  → → → → → → 最优: LLM INT4/LoRA + ViT frozen → 7-8GB → 可训练!
  → → → → → → → → → Stage 1: 只训练MLP projection → ViT+LLM frozen → 最少内存!
  → → → → → → → → → → → Stage 2: LLM LoRA + ViT frozen → 8GB → 可行!
  → → → → → → → → → → → → → Stage 3: DPO → 全模型LoRA → 8-10GB → 紧但可行!
```

## 8. RTX 4090 VLM推理 — 内存与延迟分析

```
### 8.1 内存预算

RTX 4090 24GB → VLM推理内存分析:

| 模型 | LLM | ViT | 总计(bf16) | INT4 LLM+INT8 ViT | 24GB可行? |
|------|-----|-----|-----------|-------------------|-----------|
| LLaVA-1.6-Mistral-7B | 7B | ViT-L/14(300M) | ~18GB | ~7GB+KV | ✅ 可行 |
| InternVL2-8B | 8B | InternViT-6B | ~28GB | ~10GB+KV | ⚠️ 紧 |
| Qwen2-VL-7B | 7B | ViT-L/14 | ~16GB | ~7GB+KV | ✅ 可行 |
| Phi-3.5-Vision-4B | 4B | ViT-L/14 | ~10GB | ~5GB+KV | ✅ 舒适 |

关键:
  → INT4 LLM+INT8 ViT → 7B VLM → ~7-8GB → 24GB可行 → KV cache剩余16GB!
  → → → → → InternViT-6B → INT8→3GB → 总计~10GB → 紧 → 需INT4 ViT或offload!
  → → → → → → → Phi-3.5-Vision-4B → 最舒适 → INT4→5GB → 大量KV空间!

### 8.2 Vision Encoder推理延迟

ViT推理延迟(RTX 4090):
  → ViT-L/14 + 336px单图 → 15-30ms → 快 → 不成瓶颈!
  → → → ViT-L/14 + 896px高分辨率 → 50-150ms → 慢 → 可能瓶颈!
  → → → → → InternViT-6B + 448px → 30-50ms → 6B更大 → 但448px可控!
  → → → → → → → 批量 → N图片 → N×ViT → encoder成为瓶颈 → 需分离!

### 8.3 VLM推理优化

优化策略(RTX 4090):
  1. FlashAttention-2 → 必须! → 不用→activation溢出24GB!
  → → → → → ViT+LLM都需要 → 减少activation 40-60%!

  2. INT4 LLM + INT8 ViT → 量化 → 内存减半+ → 可行!
  → → → → → → → INT4 ViT → 性能下降明显 → 不推荐 → INT8即可!

  3. Token pruning/merging → 减少visual tokens → 40-70% → 计算减!
  → → → → → → → → → ToP/ToMe → 剪枝不重要token → 减少LLM输入!

  4. Encoder-Decoder分离 → ViT单独GPU → LLM单独GPU → 2×4090!
  → → → → → → → → → → → ViT GPU1 → LLM GPU2 → 各自优化 → 最好!
  → → → → → → → → → → → → → → 但: 2GPU成本 → 单GPU也可行(量化)

  5. vLLM/SGLang → PagedAttention+RadixAttention → KV cache优化!
  → → → → → → → → → → → → → → → → → vLLM multimodal支持 → vision offload → 生产!

  6. Pixel shuffle/downsampling → Qwen2-VL/InternVL → 视觉token压缩!
  → → → → → → → → → → → → → → → → → → → → → → 高分辨率→千tokens→压缩→百tokens→LLM友好!

→ → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090最优: 7B INT4+INT8 ViT+FlashAttention-2+vLLM → 单GPU可行!
```

## 9. VLM vs Text-only LLM — 关键差异

```
### 9.1 架构差异

VLM vs LLM关键差异:

  1. 双模型 → ViT+LLM → 两个独立模型 → 内存2x → 管理2x!
     → → → → → LLM: 单模型 → 更简单 → 更容易优化!

  2. Visual tokens → 多模态输入 → 非文本 → 需翻译(projection)!
     → → → → → → → LLM: 只文本 → BPE tokenizer → 标准 → 简单!

  3. 分辨率影响 → 图片大小→token数 → 不固定 → 需动态!
     → → → → → → → → → LLM: 固定vocab → token数确定 → 简单!

  4. 预处理开销 → ViT推理 → 15-30ms → 增加延迟 → 需优化!
     → → → → → → → → → → → LLM: 无预处理 → 直接 → 无额外延迟!

### 9.2 训练差异

  1. 三阶段 → pretrain→SFT→DPO → 比LLM多一阶段(alignment)!
     → → → → → → → LLM: 2阶段 → SFT→RLHF → 简单!

  2. 数据混合 → image+text → 需要多模态数据 → 更复杂!
     → → → → → → → → → LLM: 只text → 简单 → 数据易获取!

  3. 幻觉更严重 → VLM "看到"不存在的东西 → 视觉groundedness!
     → → → → → → → → → → → LLM: 文本幻觉 → 但VLM更严重!

### 9.3 Serving差异

  1. 两阶段推理 → ViT first → LLM second → pipeline!
     → → → → → → → LLM: 单阶段 → tokenizer→model→输出 → 简单!

  2. KV cache管理 → visual tokens+text tokens → 混合 → 更复杂!
     → → → → → → → → → LLM: 只text KV → 简单 → PagedAttention!

  3. 批量挑战 → 不同图片→不同visual token数 → 不固定 → 困难!
     → → → → → → → → → → → LLM: 固定text → batching容易 → 标准!

→ → → → → → → → → → → → → → → → → → → → → → → → → → 结论: VLM比LLM复杂2x → 但趋势=简化→原生多模态→消除差异!
```

## 10. 2025-2026 VLM趋势

```
### 10.1 关键趋势

1. 原生多模态(Gemini方向):
   → 无投影层 → 从头训练 → text+image+video+audio → 端到端!
   → → → → → → → → → 2026: 更多模型走原生路线 → Gemini→GPT-5→开源!
   → → → → → → → → → → → → → → → 关键: 原生=最自然 → 不需翻译 → 最强!

2. Agent VLM(视觉行动):
   → VLM+行动 → GUI导航 → 机器人 → 视觉+行动 → 2025大方向!
   → → → → → → → → → → → Gemini 2.0 → agentic → GUI agent → 视觉行动!
   → → → → → → → → → → → → → → → → → → CogVLM2 → GUI agent → 屏幕理解→行动!

3. Token压缩(高分辨率必需):
   → 动态分辨率→千tokens → 需压缩 → pruning/merging/shuffle → 2025!
   → → → → → → → → → → → → → → → → → → → → → → ToP/ToMe → 40-70%压缩 → 计算减!
   → → → → → → → → → → → → → → → → → → → → → → → → Pixel shuffle → 学习压缩 → 更智能!

4. Video理解(时间维度):
   → 图片→视频 → 多帧 → temporal → 2025扩展 → LLaVA-OneVision!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → 长视频 → 百帧 → 需压缩 → Q-Former/Token pruning!

5. 小型高效VLM:
   → InternVL2-2B/MiniCPM-V → 小模型 → 边缘部署 → 移动!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → RTX 4090友好 → Phi-3.5-Vision-4B → 5GB INT4 → 舒适!

6. DPO替代RLHF:
   → 简单 → 稳定 → 不需reward model → 2025主流 → VLM对齐!
   → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 关联: rl-experiments.md(DPO=PPO等价但更简单)

→ → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → → 结论: 2026→原生多模态+Agent VLM+Token压缩+Video → Gemini方向!
```

## 11. 核心规律

```
VLM核心规律:

  4代架构演进 → CLIP→LLaVA→InternVL→Gemini → 趋势=原生多模态!
  → → Gen1 CLIP: 双编码器+对比 → 检索 → zero-shot → 但不生成!
  → → → Gen2 LLaVA: ViT+MLP+LLM → 生成 → 指令微调 → 简单投影够用!
  → → → → → Gen3 InternVL: ViT-6B+动态分辨率 → OCR/图表突破!
  → → → → → → → Gen4 Gemini: 原生多模态 → 无投影 → 端到端 → 2026趋势!

  Projection策略 → MLP主流 → Q-Former被认为over-engineered → 2025简单化!
  → → MLP: ~8M参数 → 保留所有token → 简单+有效 → 大多数VLM!
  → → → Q-Former: ~188M → 压缩到32token → video友好 → 但空间信息丢失!
  → → → → → 结论: ViT足够强时 → MLP就够了 → 不需Q-Former → 简单化!

  Vision Encoder → patch=14+多分辨率+动态 → 2025标准!
  → → → ps=14 → 256 tokens → 精细 → 比16(196)更好!
  → → → → → 动态分辨率 → 448px tiles → OCR/图表 → 高分辨率突破!
  → → → → → → → 位置嵌入 → learned+interpolation → 2025主流 → factorized→未来!

  三阶段训练 → pretrain(projection only)→SFT(LLM LoRA)→DPO(全模型) → 标准!
  → → → → → VILA: pretraining质量比想象更重要 → interleaved数据→更好!
  → → → → → → → DPO > PPO → 2025主流 → 简单+稳定 → 不需reward model!

  RTX 4090 VLM推理:
    → 7B INT4 + INT8 ViT + FlashAttention-2 → ~7-8GB → 24GB可行!
    → → ViT 15-30ms → 不成瓶颈 → 但高分辨率+批量→可能瓶颈!
    → → → → → Phi-3.5-Vision-4B → INT4→5GB → 最舒适 → 大量KV空间!
    → → → → → → → vLLM multimodal → PagedAttention → vision offload → 生产!

  VLM比LLM复杂2x → 双模型+visual tokens+动态分辨率+幻觉更严重 → 但趋势=简化!
  → → → → → → → → → → → → → → → 原生多模态 → 消除投影 → 端到端 → 简化!

  知识Gap修复:
    → Multimodal从★(1/5) → ★★★★(4/5) → CLIP+LLaVA+InternVL+Qwen2-VL+CogVLM+Gemini+ViT设计+Projection+Training+RTX 4090 → 全面!
    → → → → → 但仍需实践 → GPU可用时 → LLaVA部署+Phi-3.5-Vision推理 → 实测!
```

## 参考文献

```
1. 基础VLM:
   - CLIP: Radford et al. 2021, "Learning Transferable Visual Models"
   - BLIP-2: Li et al. 2023, "BLIP-2: Bootstrapping with Frozen Models"
   - LLaVA: Liu et al. 2023-2024, "Visual Instruction Tuning"
   - LLaVA-OneVision: Liu et al. 2024, "LLaVA-OneVision"

 (arxiv.org/abs/2408.06564)

   - SigLIP: Zhai et al. 2023, "Sigmoid Loss for Language-Image Pre-Training"
   - SigLIP 2: Google Research 2025
   - NaViT: Dehghani et al. 2024, "Native Resolution ViT"

2. 高级VLM:
   - InternVL: arxiv.org/abs/2312.14237
   - InternVL 2.5: arxiv.org/abs/2412.07713
   - Qwen2-VL: Alibaba 2024-2025
   - CogVLM: Tsinghua 2023-2024
   - Gemini: Google DeepMind 2023-2025

3. 训练与对齐:
   - VILA: arxiv.org/abs/2401.01670 "Pretraining Matters"
   - RLHF-V: arxiv.org/abs/2312.03749
   - VLFeedback: VLM preference dataset
   - Silkie: DPO for VLMs

4. ViT基础:
   - ViT: Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"

5. 我们的笔记:
   - ai-expert-knowledge-map-gap-analysis.md → VLM gap评估
   - scaling-laws-deep-dive.md → inference scaling
   - ai-safety-guardrails-production-deep-dive.md → VLM安全部署
   - data-pipeline-curation-deep-dive.md → VLM训练数据
