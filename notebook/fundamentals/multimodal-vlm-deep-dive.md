# Multimodal AI (Vision-Language) — From CLIP to LLaVA to InternVL

> 2026-06-08 | VLM是下一代AI主流, Infra工程师必须懂视觉编码器+跨模态注意力
> 关键: CLIP→对齐空间 → LLaVA→投影到LLM → InternVL→原生多模态

## 1. CLIP (2021) — Contrastive Language-Image Pre-training

```
CLIP核心: 图像和文本在同一向量空间 → 对齐!

架构:
  Image Encoder: ViT(Visual Transformer) → 图像→embedding
  Text Encoder: Transformer → 文本→embedding
  → 两者映射到同一空间 → cosine similarity衡量相似度

训练:
  对比学习(Contrastive Learning):
    → N个(图像,文本)对 → 正确配对应该相似 → 错误配对应该不相似
    → Loss = -log(sim(I_i, T_i)/Σsim(I_i, T_j))
    → = InfoNCE loss → 与NCE(噪声对比估计)一致!

CLIP关键发现:
  → Zero-shot分类: 不需要训练 → 直接用文本描述 → "这是一只猫的照片"
  → 迁移能力: CLIP → 可以对任何新类别分类 → 不需要fine-tune!
  → 图像-文本对齐: 相同概念→相似向量 → "猫"的文本≈猫的图像向量

与已知知识连接:
  → InfoNCE = NCE的扩展 → 我们已经懂信息论!
  → Contrastive = 与RLHF的对比类似 → 正确配对vs错误配对
  → ViT = Transformer处理图像 → patch token化 → 与LLM token化一致!

CLIP局限:
  → 不是生成模型 → 只能分类/检索 → 不能生成图像描述
  → 对齐空间有限 → 复杂推理(如"这个图像里有多少个红色的球?")不够
  → → 需要LLM增强 → LLaVA = CLIP + LLM!
```

## 2. Vision Encoder设计 — ViT到SigLIP

```
ViT (Vision Transformer, 2020):
  → 图像→patch → 每个patch=16×16像素 → 196个patch(224×224图像)
  → Patch embedding = 线性投影 → 类似LLM token embedding
  → Position embedding → 2D位置 → 告诉patch的位置
  → Transformer layers → 与LLM完全相同的架构!

ViT变体:
  ViT-B/16: 86M参数, 12层, 768dim → 小但有效
  ViT-L/14: 307M参数, 24层, 1024dim → CLIP默认
  ViT-H/14: 632M参数, 32层, 1280dim → 大但慢
  ViT-bigG/14: 1.8B参数 → InternVL用!

SigLIP (2024, Google):
  → Sigmoid loss替代softmax → 更高效
  → → sigmoid(I_i,T_i) → 不需要全局normalize → batch内独立
  → → 支持更大batch → 更好对齐 → InternVL用SigLIP!

关键: Vision Encoder决定VLM的视觉理解质量
  → 大encoder(ViT-bigG) → 更细粒度理解 → 但推理成本高!
  → 小encoder(ViT-B) → 粗粒度 → 但推理成本低
  → 投影策略决定信息传递 → 下节详述
```

## 3. LLaVA (2023-2024) — Large Language-and-Vision Assistant

```
LLaVA核心: CLIP vision encoder + LLM = 视觉语言模型!

架构(LLaVA-1.5):
  Vision Encoder: CLIP ViT-L/14 → 图像→196个patch embedding
  Projection: MLP → 196×1024 → 196×4096 → 投射到LLM空间
  LLM: LLaMA-7B/13B → 处理视觉token+文本token

训练:
  Stage 1: 预训练(Pretrain)
    → CC3M数据 → 图像描述对 → 只训练projection MLP → LLM冻结
    → 目标: 让projection学会将视觉信息映射到LLM空间
    → 快! 几小时完成 → projection很小(2层MLP)

  Stage 2: 指令微调(Instruction Tuning)
    → LLaVA-Instruct-150K → 对话格式 → 训练projection+LLM
    → 目标: 让LLM学会理解视觉信息并回答问题
    → 更慢 → 但关键 → 才能对话!

LLaVA-NeXT (2024):
  → AnyRes: 动态分辨率 → 大图像→多个patch → 更细粒度
  → 高分辨率图像 → 分成多个子图 → 每个子图独立编码 → 合并
  → → 但token数增加 → 推理成本↑ → KV cache压力!

关键数学:
  视觉token数 = (image_size / patch_size)² × num_patches
  → 224×224 / 16×16 = 14×14 = 196 tokens (CLIP ViT-L/14)
  → 336×336 / 14×14 = 24×24 = 576 tokens (LLaVA-1.5高分辨率)
  → 高分辨率: 多个子图 → 总token可达1000+!

推理影响(RTX 4090):
  → 7B LLM + 576视觉token → KV cache增加576×81.92KB = 47.3MB → 可管理
  → 高分辨率: 1000+视觉token → KV 82MB → 仍可管理(INT8 KV省50%)
  → 但: 视觉token是一次性的 → 不像文本每步增加 → prefix caching有效!
  → → vLLM/SGLang prefix caching → 视觉token共享 → 多用户同一图像省KV!

与已知Infra连接:
  → LLM推理我们已经深读 → VLM = LLM + 视觉encoder + projection
  → 视觉encoder推理 → 独立计算 → 与LLM可以流水线
  → KV cache → 视觉token固定 → prefix sharing → 复用!
  → INT8 KV → 视觉token也省50% → RTX 4090可行!
  → FlashInfer → 视觉token GQA → 同样加速!
```

## 4. InternVL (2024-2025) — 原生多模态架构

```
InternVL = 不用projection → vision encoder直接输出LLM空间!

架构(InternVL-2):
  Vision Encoder: InternViT-6B → 6B参数ViT → 极细粒度理解
  → SigLIP训练 → sigmoid loss → 更好对齐
  → 6B = 比CLIP ViT-L(307M)大20x → 更强大!

  LLM: InternLM2-7B/20B → 中文+英文多语言
  → 与InternViT同公司 → 更好配合

  连接: 不用MLP projection!
  → InternViT-6B输出 → 直接与InternLM2输入对齐 → native multimodal!
  → 需要特殊训练 → 但不需要projection → 更简单!

关键创新:
  1. 动态分辨率 → 与LLaVA-NeXT类似 → 但InternVL更灵活
  2. 原生对齐 → vision encoder输出=LLM输入 → 不需要projection
  3. 大encoder → 6B ViT → 细粒度理解 → OCR/图表/文档更强!
  4. 中文优化 → InternLM2中文能力强 → 多语言VLM

InternVL性能:
  → OCR: 93.2%(比LLaVA-1.5 75%高18pt!)
  → 图表理解: 88.5%(比LLaVA-1.5 65%高23pt!)
  → 中文VQA: 90.1%(比LLaVA-1.5 72%高18pt!)
  → → 大encoder+原生对齐 = 细粒度任务更强!

推理影响(RTX 4090):
  → InternViT-6B推理: 6B参数 → INT4量化 → ~3GB → RTX 4090可行!
  → InternLM2-7B推理: 与7B LLM相同 → INT4+INT8KV → 4,791 tok/s
  → 两者流水线: ViT→prefill→LLM→decode → 与PD分离类似!
  → 视觉prefill → compute-bound → 快(~10ms)
  → LLM decode → memory-bound → FlashInfer加速
  → 总推理成本 ≈ 7B LLM × 1.5 → 视觉encoder额外50%
```

## 5. 跨模态注意力机制

```
VLM如何处理视觉+文本信息:

方法1: Early Fusion (LLaVA方式)
  → 视觉token + 文本token → 合并 → 拼成一个sequence → 送入LLM
  → LLM看到: [视觉token_1...视觉token_N] + [文本token_1...文本token_M]
  → → LLM自然处理 → 注意力同时看视觉和文本 → 最简单!

  问题:
  → 视觉token占位置 → LLM context更长 → KV更大
  → 576视觉token → 576×81.92KB=47.3MB额外KV → 可管理但非零
  → → INT8 KV + prefix caching → 大幅缓解!

方法2: Cross-Attention Fusion (Flamingo方式)
  → 视觉token → 经过cross-attention → 与文本token交互
  → → 每隔几层插入cross-attention → 视觉信息"注入"到LLM
  → → 视觉token不占LLM context → KV更少!

  问题:
  → 需要新架构设计 → cross-attention层 → 复杂
  → 推理时 → cross-attention额外计算 → 成本↑
  → 但: 视觉KV独立管理 → 不污染文本KV → 可能更高效!

方法3: Native Multimodal (InternVL方式)
  → 视觉encoder输出 = LLM输入维度 → 直接concat → early fusion
  → → 与LLaVA类似但不需要projection → 更简洁
  → → 视觉token仍然是LLM context的一部分

RTX 4090最优策略:
  → Early Fusion (LLaVA方式) → 最简单 → 推理成本可控
  → INT8 KV → 视觉+文本KV都省50%
  → Prefix caching → 视觉token共享 → 同一图像多用户省KV!
  → FlashInfer → 视觉token GQA → 同样加速
  → 视觉encoder: ViT-L INT4 → 150MB → 与LLM并行推理
```

## 6. 图像Token化与分辨率

```
图像→视觉token的数学:

基本公式:
  num_patches = (H / patch_size) × (W / patch_size)
  num_tokens = num_patches (每个patch=1个token)

CLIP ViT-L/14 (224×224):
  → (224/14)² = 16² = 256 tokens → 但实际196(去掉一些padding)
  → patch_size=14 → 比16更细 → 1.3x更多token

高分辨率(LLaVA-NeXT, InternVL):
  → 336×336 → (336/14)² = 24² = 576 tokens → 3x更多!
  → 448×448 → (448/14)² = 32² = 1024 tokens → 5x更多!
  → → AnyRes: 动态切分子图 → 每个子图独立编码 → 合并token

Token压缩策略:
  1. PixelShuffle: 重排像素 → 4个patch合并为1 → 4x压缩 → LLaVA用!
     → (H/2, W/2, 4×C) → reshape → (H/2, W/2, C×4) → 空间4x压缩
  2. Q-Former (BLIP-2): 查询向量 → 从视觉特征提取固定数量token
     → 32个query → 固定32个视觉token → 不随图像大小变化!
     → → KV cache友好! → 但信息损失!
  3. Average Pooling: patch平均 → 简单压缩 → 质量差
  4. Learned Compression: 训练压缩模块 → 质量好 → 但需要额外训练

KV Cache影响:
  → 196 tokens → 196×81.92KB = 16.1MB → 小 → 可管理
  → 576 tokens → 47.3MB → 中 → INT8→23.65MB → OK
  → 1000+ tokens → 82MB → 大 → INT8→41MB + prefix caching → OK!
  → → RTX 4090 24GB → 7B INT4+INT8KV+GQA-8 → 有足够空间放视觉KV!

RTX 4090最优VLM配置:
  → ViT-L/14 INT4 → 150MB → 推理快
  → PixelShuffle → 4x压缩 → 576→144 tokens → KV更小!
  → INT8 KV → 视觉+文本都省50%
  → Prefix caching → 视觉token共享 → 多用户同一图像零额外KV!
  → FlashInfer → 整体1.06-3.20x加速(包括视觉token)
```

## 7. 核心规律

```
VLM核心:

1. CLIP=对齐 → 图像+文本同一空间 → zero-shot能力
   → InfoNCE loss → 与信息论一致(我们已经懂!)
   → 但不是生成模型 → 需要LLM增强

2. LLaVA=投影 → CLIP→MLP→LLM → 最简单最有效!
   → 2层MLP projection → 几小时预训练 → 最小成本
   → 视觉token = LLM context的一部分 → Early Fusion

3. InternVL=原生 → 大encoder+原生对齐 → 细粒度更强!
   → 6B ViT → 20x CLIP → OCR/图表/文档更好
   → 不需要projection → 但需要特殊训练

4. 视觉token = 推理成本增加器 → 但可控!
   → 576 tokens → 47.3MB KV → INT8→23.65MB → 可管理
   → Prefix caching → 视觉token共享 → 多用户同一图像免费!
   → FlashInfer → 视觉token也加速 → 与文本token相同

5. 分辨率 = token数 = KV大小 = 推理成本
   → 224×224 → 196 tokens → 小
   → 336×336 → 576 tokens → 中
   → 448×448 → 1024 tokens → 大 → PixelShuffle压缩!

6. RTX 4090 VLM最优配置:
   → 7B LLM(Vicuna/InternLM) INT4+INT8KV+FlashInfer → 4,791 tok/s
   → ViT-L INT4 → 150MB → 与LLM并行推理 → ~10ms prefill
   → PixelShuffle → 576→144 tokens → KV更小 → 更多并发
   → Prefix caching → 同一图像多用户 → 零额外KV成本
   → 安全guardrails → 视觉输出也要过滤 → xgrammar扩展!
   → → VLM推理成本 ≈ 1.5× LLM推理 → RTX 4090完全可行!

VLM发展趋势 (2025):
  → 视频理解 → 视觉token更多 → 需要更强KV管理
  → 多图像理解 → 多个图像 → prefix sharing更重要
  → 原生多模态 → 不再需要projection → InternVL方向
  → 细粒度 → 大encoder → OCR/图表/文档 → 实用!
  → → VLM = 2025-2026 AI主流方向 → Infra必须支持!