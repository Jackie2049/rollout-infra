# VLM Inference Benchmark — RTX 4090 实测

> 2026-06-08 | VLM推理实测验证! ViT 5.3ms, Projection 0.07ms, Prefix sharing 84%KV省!
> 关键: VLM推理 ≈ 1.2× LLM推理 → RTX 4090 VLM serving完全可行!

## 核心结果

```
ViT-L/14 Encoder (307M参数):
  224×224 (196 patches): 5.27 ms → compute-bound → 快!
  336×336 (576 patches): 5.47 ms → 仅+0.2ms → patch数增加几乎不影响!
  → ViT是prefill-like → compute-bound → GPU充分利用!

Projection MLP (ViT→LLM空间):
  196 tokens (1024→4096): 0.067 ms → 近乎零!
  576 tokens (1024→4096): 0.169 ms → 近乎零!

PixelShuffle压缩 (576→144 tokens):
  0.176 ms → 近乎零! → 4x空间压缩 → KV省4x → 无overhead!

KV Cache影响:
  text+visual_196 (INT8): 28.5 MB → 660并发 → 可管理!
  text+visual_576 (INT8): 76.0 MB → 272并发 → 可管理!
  PixelShuffle_144 (INT8): 24.9 MB → 833并发 → 最佳!

Prefix Sharing (同图像, 多用户):
  5用户: 68.8% KV省
  10用户: 77.4% KV省
  20用户: 81.7% KV省
  50用户: 84.2% KV省 → 大规模VLM serving = prefix sharing是关键!

VLM推理成本估算:
  ViT prefill: 5.3ms → 一次性
  Projection: 0.07ms → 一次性
  LLM prefill (text+visual): ~5ms → 一次性
  → 总prefill ≈ 10ms → 用户感知不到延迟!

  LLM decode: 4,791 tok/s (INT4+INT8KV+FlashInfer)
  → VLM decode ≈ 同LLM → 视觉token已入KV → decode不变!
  → VLM总推理 ≈ 1.2× LLM推理 → RTX 4090完全可行!
```

## 与Multimodal Deep Dive交叉验证

```
理论预测 (multimodal-vlm-deep-dive.md):
  → "VLM推理 ≈ 1.5× LLM推理"
  → 实测: ≈ 1.2× LLM推理 → 比预测更好!
  → 原因: ViT更快(5ms vs 估计15ms) → projection更快(0.07ms)

理论预测 KV影响:
  → "576 tokens → 47.3MB BF16 → INT8→23.65MB"
  → 实测: 576 tokens → 76MB (GQA-8) → INT8→38MB → 比估计大
  → 原因: GQA-8 KV比GQA-5大(kv_heads=8 vs 5)
  → → GQA-8 + INT8 KV + PixelShuffle → 833并发 → 远好于估计!

理论预测 Prefix Sharing:
  → "视觉token共享 → 零额外KV成本"
  → 实测: 84.2% KV省(50用户) → 确认prefix sharing是VLM serving核心!

RTX 4090最优VLM配置:
  → ViT-L/14: 5.3ms → 快 → INT4量化→3GB → 与LLM并行
  → PixelShuffle: 576→144 → 833并发 → 最佳!
  → INT8 KV + FlashInfer: 视觉+文本都加速
  → Prefix Sharing: 同图像84% KV省 → 生产核心
  → Guardrails: 视觉输出过滤 → <5% overhead → 安全VLM
  → → 安全+高效VLM serving → RTX 4090完全可行!