# LLM推理量化基础知识 — RTX 4090 (SM89) 专门分析

> 2026-06-16 | RTX 4090 | SM89 | INT4/INT8/FP8/BF16 | vLLM/SGLang/MindIE | 生产路径
> ★★★★★ RTX 4090 (SM89)量化路径: INT4 Marlin/Triton(推理唯一可行) + INT8 KV cache(唯一可行KV量化) + FP8全路径CRASH
> ★★★★★ 3种FP8 KV路径需严格区分: Triton FP8 SM89 ALLOWED / FlashInfer FP8 NOT / compressed-tensors override CRASH

## 1. ★★★★★ GPU Compute Capability与量化路径映射

```
★★★★★★★ SM版本 → 量化能力映射:

| SM版本 | GPU代表 | FP16 | BF16 | INT8 | INT4 | FP8(E4M3/E5M2) | FP4(E2M1) | MXFP4 |
|--------|---------|------|------|------|------|----------------|-----------|-------|
| SM80   | A100    | ✓    | ✓    | ✓    | ✓(Marlin) | ✗ | ✗ | ✗ |
| SM89   | RTX 4090| ✓    | ✓    | ✓    | ✓(Marlin/Triton) | ✗(no HW) | ✗ | ✗ |
| SM90   | H100    | ✓    | ✓    | ✓    | ✓(Marlin/Triton) | ✓(native) | ✗ | ✗ |
| SM100  | B200    | ✓    | ✓    | ✓    | ✓    | ✓ | ✓(native) | ✗ |
| SM120  | RTX 5090| ✓    | ✓    | ✓    | ✓    | ✓ | ✓(native) | ✓(native) |

★★★★★★★ 关键理解:
  → SM89 RTX 4090 → FP8 = 纯软件模拟 → 无硬件加速 → 比 BF16 更慢 → 不推荐!
  → SM89 RTX 4090 → INT4 = Marlin/Triton kernel → 有硬件INT4 tensor core → 推理最强量化!
  → SM89 RTX 4090 → INT8 = FlashInfer kernel → 有硬件INT8 tensor core → KV cache唯一量化路径
  → SM90 H100 → FP8 = 硬件native → 2x FP16 throughput → 推理首选量化
  → SM120 RTX 5090 → FP4/MXFP4 = 硬件native → 2x FP8 → 未来方向!
```

## 2. ★★★★★ 权重量化 — INT4 Marlin/Triton

```
★★★★★★★ INT4量化 = RTX 4090推理生产唯一路径:

| 方法 | 量化算法 | Kernel | SM89支持 | vLLM支持 | 压缩比 | 精度损失 |
|------|---------|--------|---------|---------|--------|---------|
| GPTQ | 对称4bit+group128 | Marlin | ✓(SM>=8.0) | ✓(#43731 Triton fallback) | ~4x | ~1-2% |
| AWQ  | 激活感知4bit | Marlin | ✓(SM>=8.0) | ✓ | ~4x | ~1-3% |
| SqueezeLLM | 稀疏+量化 | Custom | 部分 | ✓ | ~4x | ~2-5% |
| FP8(compressed-tensors) | E4M3/E5M2 | CUTLASS | ✗(SM>=9.0) | ✓(但SM89 crash!) | ~2x | ~0.5-1% |
| MXFP4 | E2M1+MX scaling | Custom | ✗(SM>=12.0) | ✗ | ~4x | ~0.3-0.5% |

★★★★★★★ vLLM INT4 Triton fallback (#43731, v0.23.0):
  → 之前: GPTQ INT4 → Marlin kernel only → SM89需要SM>=8.0 → 可以运行
  → 现在: v0.23.0新增INT4 Triton fallback → Marlin不可用时自动切换 → 更多SM覆盖
  → BUT: Triton fallback → 比Marlin慢30-50% → Marlin仍是首选 → Triton是备选

★★★★★★★ RTX 4090 INT4推理配置:

  # vLLM INT4 GPTQ推理 (推荐):
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-GPTQ-Int4 \
    --quantization gptq \
    --kv-cache-dtype int8 \               # ← INT8 KV! 不是FP8!
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16

  # vLLM INT4 AWQ推理 (备选):
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-AWQ \
    --quantization awq \
    --kv-cache-dtype int8 \
    --max-model-len 4096 \
    --dtype bfloat16
```

## 3. ★★★★★★★ KV Cache量化 — RTX 4090唯一INT8可行

```
★★★★★★★★★ 3种FP8 KV路径 — MUST区分!

| KV量化方法 | 实现路径 | SM89行为 | vLLM支持 | 生产推荐 |
|------------|---------|---------|---------|---------|
| Triton FP8 KV | vLLM Triton kernel (#43914) | ✓ ALLOWED → SM89可以运行 | ✓ | ★★★ ★★(但不如INT8) |
| FlashInfer FP8 KV | FlashInfer attention kernel | ✗ CRASH → SM89不支持 | ✓(SM>=9.0) | ✗ |
| compressed-tensors FP8 KV | compressed-tensors override | ✗ CRASH (#44879/#45038) → bypass FlashInfer检查 | ✓(but crash!) | ✗✗✗ |

★★★★★★★★★ RTX 4090 KV量化推荐:

  → ★★★★★★★★ INT8 KV (FlashInfer) = RTX 4090唯一生产可行KV量化路径
  → → 2x KV cache压缩 → 更多token → 更长context → GRPO直接受益
  → → FlashInfer INT8 kernel → SM89硬件支持 → 稳定运行 → 无crash风险
  → → --kv-cache-dtype int8 → vLLM参数 → 直接启用

  → ★★★ Triton FP8 KV (#43914) = SM89理论上可运行 → 但:
  → → 无硬件FP8 → 纯软件模拟 → 不如INT8硬件加速
  → → 精度: E4M3(4bit mantissa) vs INT8(8bit) → INT8精度更好
  → → 推荐: 除非特别需要FP8格式 → 否则INT8优先

  → ✗✗✗ compressed-tensors FP8 KV → SM89必CRASH → #44879/#45038 → 绝不使用!

★★★★★★★★★ KV量化内存节省估算:

  # 7B模型, 32K context, group_size=8 GRPO:
  BF16 KV: num_layers * 2 * 32K * hidden_dim * 2bytes = ~4GB per sequence
  INT8 KV: ~2GB per sequence (50% saving)
  FP8 KV:  ~2GB per sequence (50% saving, 但SM89不可用)

  → ★★★★★ RTX 4090 24GB → BF16 KV → 6 concurrent sequences → 32K max
  → ★★★★★ RTX 4090 24GB → INT8 KV → 12 concurrent sequences → 32K max → GRPO group_size翻倍!
```

## 4. ★★★★★ 激活量化 — FP8 vs INT8

```
★★★★★★★ 激活量化 (用于训练中的微批次):

| 激活量化 | 算法 | 精度 | SM89 | 训练用途 | 推理用途 |
|---------|------|------|------|---------|---------|
| FP8(E4M3) | dynamic per-tensor | ~BF16接近 | ✗(no HW) | ✗ | SM90推理 |
| INT8 | dynamic per-token | ~1%loss | ✓(HW) | W8A8 MoE | RTX 4090 MoE |
| BF16 | 无量化 | exact | ✓ | 所有训练 | 所有推理 |

★★★★★★★ RTX 4090训练 → BF16 only:
  → FP8 activation → SM89无硬件 → 量化比不量化更慢 → 无意义
  → INT8 activation → W8A8需要权重也INT8 → 推理可以用 → 训练不推荐
  → BF16 → RTX 4090默认 → 硬件native → 最快 → 训练推理都适用

★★★★★★★ MindIE/Ascend激活量化对比:
  → npu_dequant_swiglu_quant → W8A8 → dequant+SwiGLU+quant → 1 kernel → compose-level!
  → MXFP4 → float4_e2m1fn_x2 → 910C+硬件 → SM120/RTX5090 FP4 equivalent
  → → ★★★★★★★ NVIDIA → 无compose-level → dequant+activation+quant = 3 kernels vs Ascend 1!
```

## 5. ★★★★★ 量化框架对比 — 7框架RTX 4090量化路径

```
★★★★★★★ 7框架量化支持矩阵 (RTX 4090 SM89):

| 框架 | INT4推理 | INT8推理 | FP8推理 | INT8 KV | FP8 KV | INT8训练 | 量化备注 |
|------|---------|---------|---------|---------|--------|---------|---------|
| vLLM | ✓ GPTQ/AWQ Marlin | ✗ | ✗ | ✓ FlashInfer | ✗(crash) | ✗ | INT4 Triton fallback v0.23 |
| SGLang | ✓ GPTQ/AWQ | ✗ | ✗ | ✓ | ✗ | ✗ | DeepGEMM SM89 fallback |
| verl | ✗(rollout用vLLM) | ✗ | ✗ | ✗ | ✗ | ✗ | 依赖vLLM/SGLang量化 |
| MindIE | ✓ MXFP4(910C+) | ✓ W8A8 | ✗ | ✓ CANN | ✓ CANN(910C+) | ✓ W8A8 compose | compose-level量化独有 |
| DeepSpeed | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ZeRO无量化 LoRAOptimizedLinear支持 |
| Megatron-LM | ✗ | ✗ | ✗(SM90only) | ✗ | ✗(SM90) | ✗ | FP8=SM90 only → RTX4090✗ |
| PyTorch | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | Inductor SM<90 fusion root cause! |

★★★★★★★ RTX 4090量化结论:
  → ★★★★★★★★ 推理量化: INT4 (GPTQ/AWQ) → vLLM/SGLang → 唯一生产路径
  → ★★★★★★★★ KV量化: INT8 (FlashInfer) → vLLM → 唯一生产路径
  → ★★★★★ 训练量化: BF16 → 所有框架 → 无量化 → LoRA(0.6GB)代替full(14GB)
  → ✗✗✗ FP8全部: SM89无硬件 → 不适用 → 等RTX 5090 (SM120)
  → ★★★★★★★★ compose-level量化: MindIE独有 → dequant+activation+quant 1 kernel → NVIDIA没有!
```

## 6. ★★★★★ 量化未来趋势 — FP4/MXFP4取代INT4

```
★★★★★★★ 量化演进方向:

  INT4 (当前RTX 4090) → FP8 (SM90 H100) → FP4/MXFP4 (SM120 RTX 5090/B200)

★★★★★★★ FP4 vs INT4 技术对比:
  → INT4: 整数量化 → 对称/不对称 → group scaling → GPTQ/AWQ → 精度1-3%损失
  → FP4(E2M1): 浮点量化 → 2bit exponent + 1bit mantissa → MX scaling → 精度0.3-0.5%
  → MXFP4: Microscaling + FP4 → block scaling + per-block normalization → 更好精度+硬件加速

★★★★★★★ vLLM贡献窗口:
  → SM120 FP4/MXFP4 kernel → RTX 5090 → next-phase → vLLM需要新量化路径
  → 当前vLLM → INT4 Marlin → FP8 compressed-tensors → 无FP4路径
  → → ★★★★★★★★ MXFP4 kernel开发 → vLLM → RTX 5090贡献 → 高优先级!

★★★★★★★ MindIE MXFP4参考:
  → float4_e2m1fn_x2 → 910C+硬件 → compose-level → dequant+activation+quant
  → → Ascend已经走在前面 → NVIDIA需要追赶 → vLLM社区可以贡献SM120 kernel!
```

## 参考
- vLLM INT4 Triton fallback: #43731 (v0.23.0) → SM89 production path
- vLLM FP8 KV crash: #44879/#45038 → compressed-tensors FP8 override → SM89必CRASH
- vLLM Triton FP8 KV: #43914 → SM89 ALLOWED → 但不如INT8
- vLLM QuantKey refactor: #32268 → boolean→QuantKey → foundation for SM89 guard
- SGLang deterministic inference → Triton persistent kernels → constexpr → batch-invariant
- MindIE npu_dequant_swiglu_quant → compose-level → W8A8 MoE → 5x kernel reduction
- MindIE MXFP4 → float4_e2m1fn_x2 → 910C+ → future FP4 direction
- Related notes: vllm-v023-release-reading.md, vllm-v1-quantization-reading.md, mindie-atb-compose-fusion-deep-reading.md, sm89_batch_invariance_diagnostic.py
