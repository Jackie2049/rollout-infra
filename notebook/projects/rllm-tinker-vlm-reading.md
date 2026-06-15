# rLLM Tinker VLM (PR #357) — Geo3K VLM集成深度分析

> 2026-06-16 | rLLM PR #357 | Geo3K Cookbook | Tinker VLM
> ★★★★★ 首次VLM RL训练 → Tinker VLM支持 → 但verl VLM更成熟
> ★★★★ RTX 4090 VLM可行: Qwen2.5-VL-7B / Qwen3-VL-8B (INT8 KV必须!)

## 1. ★★★★★ PR #357: 3个VLM Bug修复

```
★★★★★ Bug 1: image_processor not passed to TinkerEngine
  → TinkerBackend.init_rollout_engine() → 没有加载image_processor
  → VLM模型在first multimodal message → assertion error → deep inside renderer
  → 修复:
    1. VLM检测: "vl" or "vision" in model_name_lower (substring heuristic!)
    2. AutoProcessor.from_pretrained() → processor.image_processor extracted
    3. image_processor → TinkerEngine.__init__() → renderer
    4. Fail-fast RuntimeError for missing torchvision (Qwen3-VL needs it!)

★★★★★ Bug 2: Message format incompatible with VLM
  → QwenChatTemplateParser.parse_user() → 处理multimodal messages with "images" key
  → Strip leading <image> literal → prevent double-counting
  → Inject <|vision_start|><|image_pad|><|vision_end|> tokens * n_imgs
  → VLM-specific tokens: image_token, vision_start_token, vision_end_token

★★★★★ Bug 3: ImageChunk ValueError
  → TinkerEngine._convert_images_to_content_list() → convert PIL→renderer content
  → _flat_token_input_length() → ImageChunk has .length → correct counting
  → Without this → ValueError when renderer encounters raw PIL images
```

## 2. ★★★★★ VLM检测机制

```
★★★★★ String matching heuristic (tinker_backend.py line 125):
  model_name_lower = self.full_config.model.name.lower()
  if "vl" in model_name_lower or "vision" in model_name_lower:

★★★★ 匹配的模型:
  → Qwen3-VL-30B-A3B-Instruct → "vl" matches
  → Qwen3-VL-8B-Instruct → "vl" matches
  → Qwen2-VL-7B-Instruct → "vl" matches
  → Qwen2.5-VL-3B-Instruct → "vl" matches

★★★★ 潜在false positive:
  → model-v1-large → "vl" in name → incorrectly triggers VLM detection
  → Heuristic only → no architecture-level verification

★★★★ 检测后行为:
  1. AutoProcessor.from_pretrained()
  2. processor.image_processor extracted
  3. image_processor → TinkerEngine → renderer
  4. VLM-specific token injection in ChatTemplateParser
```

## 3. ★★★★★ RTX 4090 VLM VRAM估算

```
★★★★★★ RTX 4090 (24GB) VLM VRAM估算:

| Model | BF16 Weights | LoRA r=32 | KV Cache | Total | Headroom | 可行性 |
|-------|-------------|-----------|----------|-------|----------|---------|
| Qwen2.5-VL-3B | ~6GB | ~0.1GB | ~2GB | ~8GB | ~16GB | ★★★★★ 舒适 |
| Qwen2.5-VL-7B | ~14GB | ~0.3GB | ~3GB | ~17GB | ~7GB | ★★★★ 可行(INT8 KV) |
| Qwen3-VL-8B | ~16GB | ~0.3GB | ~4GB | ~20GB | ~4GB | ★★★ 紧凑(INT8 KV+offload) |
| Qwen3-VL-30B-A3B | ~60GB | N/A | N/A | >60GB | N/A | ✗✗✗ 不可行! |

★★★★★★ RTX 4090 VLM推荐:
  → Qwen2.5-VL-3B → 快速实验 (8GB peak, 16GB headroom)
  → Qwen2.5-VL-7B → 最佳平衡 (17GB peak, 7GB headroom) → INT8 KV必须!
  → Qwen3-VL-8B → 紧凑可行 (20GB peak, 4GB headroom) → INT8 KV + offload必须!
  → Qwen3-VL-30B-A3B → ✗✗✗ 绝对不可行!

★★★★★★ VLM额外内存开销:
  → 图像token膨胀: 1 geometry diagram (384x384) → ~256-768 image tokens
  → 100-token text prompt + 1 image → ~300-1000 total tokens
  → group_size=8 → prompt memory × 8
  → geo3k training: max_prompt_length=4096 → accounts for image inflation
```

## 4. ★★★★★ Geo3K Cookbook VLM集成

```
★★★★★ Geo3K dataset (hiyouga/geometry3K):
  → 2.4K train, 601 test rows
  → Each row: geometry problem (text), answer (string), images ([PIL.Image])

★★★★★ Data transform (geo3k_transform):
  → Normalize HuggingFace row → {question, images: [PIL], ground_truth, data_source}
  → Handles images = list / single / None cases

★★★★★ Plugin registration (pyproject.toml):
  → entry-points."rllm.agents" → geo3k = geo3k_flow:geo3k_flow
  → entry-points."rllm.evaluators" → geo3k_math = geo3k_eval:geo3k_evaluator

★★★★★ Training config (train_tinker.sh):
  → model: Qwen3-VL-30B-A3B-Instruct (默认 → NOT for RTX 4090!)
  → lora_rank: 32
  → group_size: 8
  → temperature: 0.6
  → total_epochs: 3
  → bypass_mode: true (default)

★★★★★ RTX 4090配置修改:
  → model.name → Qwen/Qwen2.5-VL-7B-Instruct (NOT 30B-A3B!)
  → lora_rank: 32
  → group_size: 4 (reduced from 8 → VRAM savings)
  → INT8 KV cache → 必须启用
  → bypass_mode: true (default → no ref model)
```

## 5. ★★★★★ VLM图像流转完整路径

```
★★★★★★★ VLM图像7步流转路径:

Step 1: Dataset loading (geo3k_transform)
  → Raw HuggingFace row → {question, images: [PIL.Image], ground_truth}

Step 2: Materialization (materialize.py)
  → category=="vlm" → PIL Images → PNG files under images/
  → Read time → _build_multimodal_instruction() → OpenAI-format content blocks
  → base64 data URI encoding

Step 3: AgentFlow execution (geo3k_flow.py)
  → _build_vlm_content() → split text on <image> markers
  → Replace <image> → {"type": "image_url", "image_url": {"url": data:image/png;base64,...}}
  → _img_to_data_uri() → handles bytes/dict/str/PIL.Image
  → _detect_mime() → PNG/JPEG/WebP magic bytes detection
  → Messages → AsyncOpenAI.chat.completions.create()

Step 4: Gateway/TinkerEngine
  → bypass_render_with_parser=True → ChatTemplateParser → inject vision tokens
  → bypass_render_with_parser=False → _convert_images_to_content_list() → renderer

Step 5: Tokenizer encoding
  → Rendered prompt (with vision tokens) → token IDs
  → Mix of text token IDs + special vision/image token IDs

Step 6: Tinker sampling
  → _flat_token_input_to_model_input() → ModelInput with EncodedTextChunk + ImageChunk
  → _flat_token_input_length() → text tokens count 1, ImageChunk counts .length

Step 7: Training data transform (transform.py)
  → trajectory_to_datums() → ImageChunk NOT flattened → preserved
  → .length used for token counting in create_rightshifted_model_input_and_leftshifted_targets()
```

## 6. ★★★★★ Image Token膨胀分析

```
★★★★★★★ Qwen VLM token expansion:
  → <|vision_start|><|image_pad|><|vision_end|> per image
  → <|image_pad|> → multiple visual patch tokens after ViT processing
  → 1 geometry diagram (384x384) → ~256-768 image tokens

★★★★★★★ Prompt length估算:
  → Text prompt: ~100-200 tokens
  → 1 image: ~256-768 tokens
  → Total per VLM task: ~300-1000 tokens
  → group_size=8 → × 8 → 2400-8000 total prompt tokens per GRPO step!

★★★★★★★ QwenChatTemplateParser VLM处理:
  → Strip leading <image> literal → prevent double-counting
  → (vision_start + image_pad + vision_end) * n_imgs
  → Multi-image support → multiple <image> markers

★★★★★★★ process_image_data:
  → Uses qwen_vl_utils.fetch_image() → Qwen-specific ViT patch processing
  → image_patch_size from processor.image_processor.patch_size
  → Handles dict/str/PIL.Image input formats
```

## 7. ★★★★★ verl VLM vs rLLM Tinker VLM对比

```
★★★★★★★ VLM支持对比:

| 维度 | rLLM Tinker VLM | verl VLM |
|------|-----------------|----------|
| 支持状态 | ★★★ Experimental/Limited | ★★★★★ Full/Mature |
| 文档 | ★★ Limited | ★★★★ Complete |
| 推荐度 | "Use verl for VLM training" | ★★★★★ 推荐 |
| VLM模型 | Qwen3-VL系列 | Qwen2-VL/Qwen3-VL/Qwen2.5-VL |
| Image处理 | QwenChatTemplateParser | Qwen2VLImageProcessor + mRoPE |
| 位置编码 | 无特殊处理 | get_rope_index() for Qwen3-VL/Qwen2-VL |
| TP支持 | 单GPU only | TP=2 distributed |
| Bug修复 | 3 bugs (PR #357) | patch_verl_qwen3_vl_dummy_inplace() |
| Cookbook | geo3k (实验性) | 多个VLM cookbook |

★★★★★★ RTX 4090 VLM选择指南:
  → 简单VLM实验 → rLLM Tinker → geo3k cookbook → 一行命令
  → 多模态RL production → verl → 更成熟VLM支持 → mRoPE + TP
  → Qwen2.5-VL-3B → rLLM Tinker → 最简单VLM入门
  → Qwen2.5-VL-7B → verl → INT8 KV + offload → 更稳定
```

## 8. ★★★★★ 关键发现与RTX 4090影响

```
★★★★★★ 5个关键发现:

1. ★★★★★ VLM detection = substring heuristic → fragile
   → "vl"/"vision" in model_name → false positives possible
   → No architecture-level verification → future: check model config

2. ★★★★★ Qwen3-VL需要torchvision → fail-fast RuntimeError
   → Even for image-only → video processor imports torchvision at load time
   → uv pip install torchvision → MUST for Qwen3-VL on rLLM

3. ★★★★★ RTX 4090 VLM可行但需要INT8 KV + offload
   → Qwen2.5-VL-7B → 17GB peak → INT8 KV must → 7GB headroom
   → Qwen3-VL-8B → 20GB peak → INT8 KV + offload must → 4GB headroom

4. ★★★★★ Geo3K cookbook = RTX 4090 VLM入门最佳路径
   → 2.4K训练 → geometry problem → 图像+文本
   → 但默认模型30B-A3B不可行 → 必须改为7B/3B
   → Gap: 需要RTX 4090 VLM recipe → 可贡献train_tinker_rtx4090.sh

5. ★★★★★ verl VLM更成熟 → rLLM Tinker仍适合RTX 4090简单VLM
   → verl: mRoPE + TP + 完整VLM pipeline → production grade
   → rLLM: 实验性 → 但in-process → 简单setup → 快速实验
   → RTX 4090 VLM: rLLM #1 for simplicity, verl #2 for maturity
```

## 参考
- rLLM PR #357: Geo3K Tinker VLM integration
- rLLM docs/cookbooks/geo3k.mdx: cookbook documentation
- rLLM docs/backends/comparison.mdx: VLM support comparison (tinker: Limited, verl: Full)
- tinker_backend.py: VLM detection (line 125), image_processor loading (lines 118-157)
- chat_template_parser.py: QwenChatTemplateParser VLM handling (lines 374-593)
- tinker_engine.py: _convert_images_to_content_list (lines 243-259)
- geo3k_flow.py: AgentFlow with _build_vlm_content (lines 94-130)
- 相关笔记: rllm-tinker-backend-deep-reading.md, rllm-architecture-reading.md
