# Quantization + Inference + Training 交叉点 — RTX 4090终极路径

> 2026-06-15 | 综合7框架 → 量化是RTX 4090唯一出路 → 推理-训练-量化三合一
> ★ ★ ★ 核心结论: INT4推理(4,791tok/s) + LoRA训练(17GB) + GRPO算法 → 三合一→RTX4090最优

## 1. 为什么量化是RTX 4090唯一出路

```
★ ★ ★ RTX 4090 24GB内存约束 → decode=memory-bound(95.1% weight reads)

BF16推理(7B):
  → weights=14GB → KV=10GB → 总=24GB → ★ 精确占满24GB → 无headroom!
  → → 多轮/长上下文 → KV超10GB → ✗✗✗ 不可行

INT4推理(7B):
  → weights=3.5GB → INT8KV=5GB → graph pool=2GB → buffers=0.5GB → 总=~11GB
  → → ★ ★★ 13GB headroom → 多轮/长上下文/batch/EAGLE全部可行!

★ ★ ★ 结论:
  BF16推理 → 精确占满 → ✗ 无法做任何优化
  INT4推理 → 13GB headroom → ✓✓✓ 所有优化都可行 → ★唯一出路!
```

## 2. 量化方法对比 — 4种量化选择

```
★ ★ ★ 4种量化方法对比:

| 方法 | 权重精度 | KV精度 | 压缩率 | 推理加速 | RTX4090 |
|------|---------|--------|--------|---------|---------|
| ★★★ GPTQ INT4 | 4bit | 8bit | 4x weights | 3.4x throughput | ★★★✓✓✓ |
| AWQ INT4 | 4bit | 8bit | 4x weights | 3x throughput | ✓✓ |
| FP8(E4M3) | 8bit | 8bit | 2x weights | 1.5x throughput | ✗(SM90+?) |
| ★ INT8 Weight | 8bit | 8bit | 2x weights | 1.5x | ✓ |

★ ★ ★ GPTQ INT4最优原因:
  → 4bit → 最大压缩 → 3.5GB vs 14GB → ★ 4x省内存!
  → Marlin kernel → RTX 4090 SM 8.9支持 → ★ 高性能!
  → Triton fallback(v0.23 PR#43731) → non-Marlin shapes也能用 → ★ 更多模型!
  → INT8KV → KV也压缩 → 5GB vs 10GB → ★ 多轮可行!

★ ★ vLLM INT4 Triton fallback关键细节:
  → TritonW4A16LinearKernel.can_implement() → 从ROCm-only→CUDA+ROCm
  → 最低优先级 → 只在Marlin/Machete拒绝shapes时激活
  → intermediate_size not divisible by 128 → Marlin不支持 → Triton兜底
  → ★ ★ ★ RTX 4090: INT4 MoE模型之前ValueError → 现在Triton兜底 → ★ 更多模型可用!
```

## 3. LoRA + 量化 — 训练推理完美循环

```
★ ★ ★ ★ ★ LoRA + 量化 = 训练→推理完美循环:

训练阶段(BF16):
  → 7B BF16 + LoRA-32 → 只训0.8%参数 → ~17GB → 24GB ✓✓✓
  → ★ LoRA rank=32 → attn/mlp/unembed → auto-init(rLLM) → 最简单
  → ★ ★ LoRA是BF16精度训练 → ★★★ BF16唯一正确训练精度!

合并阶段(merge):
  → LoRA weights merge into base → ★ merge后推理与全参数训练完全等价!
  → ★ ★ merge后无LoRA overhead → ★ INT4量化不影响LoRA

量化阶段(INT4):
  → merge后BF16 → GPTQ/AWQ → INT4 → 3.5GB → ★★★ 推理内存可控!
  → ★ ★ ★ 关键: 量化在merge之后 → LoRA信息完全保留在INT4权重中!

推理阶段(INT4):
  → vLLM INT4 + INT8KV + CUDA graph → 4,791 tok/s → ★★★ 最高吞吐
  → EAGLE + INT4 → 9,088 tok/s → ★★★★ 极速推理!

★ ★ ★ ★ ★ 为什么这个循环如此优雅?
  → 训练: LoRA(少量参数) → 0.11GB trainable → ★ 内存省!
  → 合并: merge into base → 等价于全参数训练 → ★ 无损失!
  → 量化: INT4 → 4x压缩 → ★ 推理省内存!
  → 推理: vLLM → continuous batching → ★ 高吞吐!

  → ★★★ 从17GB训练 → 到11GB推理 → 内存一直可控 → ★ 完美循环!
```

## 4. 量化 + CUDA Graph — 推理双重加速

```
★ ★ ★ INT4 + CUDA Graph = 推理双重加速:

Layer 1: INT4量化 → memory-bound→3.4x加速:
  → INT4 weights → 3.5GB vs 14GB → weight reads 4x少 → decode加速
  → INT8KV → 5GB vs 10GB → KV reads 2x少 → 多轮可行
  → GQA-8 → KV heads 8x少 → KV reads 8x少 → ★★★ 组合加速!

Layer 2: CUDA Graph → kernel launch 100x加速:
  → 128 kernels → 0.64ms→0.006ms → ★ CPU launch overhead消除!
  → decode每步shape不变 → ★★★ graph replay最优!
  → graph pool ~2GB → 固定地址 → replay安全!

★ ★ ★ 组合加速计算:
  → INT4: 3.4x (memory-bound → weight reads 4x少)
  → CUDA Graph: ~10% (launch overhead消除)
  → INT8KV+GQA: 2x+8x → ★★★ KV reads 16x少 → 多轮5x加速!
  → EAGLE: 8.3x (speculative → acceptance 70-80%)

  → ★★★★ 总: 4,791 tok/s (INT4 baseline)
  → ★★★★ EAGLE: 9,088 tok/s → ★★★★★ 极速!

★ ★ 内存预算(INT4+INT8KV+EAGLE):
  INT4 weights=3.5GB + INT8KV=5GB + graph pool=2GB + buffers=0.5GB + EAGLE draft=0.5GB
  = ~11.5GB → ★ 12.5GB headroom → ★★★ 完全可行!
```

## 5. 量化 + GRPO RL训练 — 三合一路径

```
★ ★ ★ ★ ★ 量化 + GRPO = 推理-训练三合一:

Phase 1: GRPO训练 → BF16精度
  → rLLM Tinker + GRPO + LoRA-32 + bypass_mode → ~17GB
  → ★ ★ ★ BF16是唯一正确训练精度 → INT4用于推理不用于训练!
  → ★ ★ bypass_mode → pi_old=rollout logprobs → 省forward

Phase 2: LoRA merge → 仍然BF16
  → save_pretrained → merge LoRA → HF format → ★ BF16 weights

Phase 3: INT4量化 → 推理精度
  → GPTQ → INT4 weights → ★ 推理精度4bit → 训练精度BF16

Phase 4: vLLM INT4推理 → 生产部署
  → INT4+INT8KV+GQA-8+prefix caching → 4,791 tok/s
  → EAGLE+INT4 → 9,088 tok/s → ★★★★

★ ★ ★ 关键洞察:
  → 训练用BF16 → 推理用INT4 → ★★ 不同阶段不同精度 → 各最优!
  → LoRA merge后BF16 → INT4量化 → ★★ 精度损失可接受(inference)
  → ★★★ GRPO训练+INT4推理 = 推理-训练-量化三合一 → RTX 4090终极路径!
```

## 6. 量化与SM 8.9限制

```
★ ★ ★ RTX 4090 SM 8.9 量化限制:

可用:
  ✓ INT4 GPTQ → Marlin kernel → Triton fallback → ★★★
  ✓ INT4 AWQ → Marlin kernel → ★✓
  ✓ INT8 weights → cuBLAS → ✓
  ✓ INT8 KV cache → FlashInfer/vLLM → ★✓
  ✓ FP8 E4M3 inference → ★★ 可用但SM89无专用kernel加速
  ✓ BF16 → 全精度 → 训练唯一选择

不可用:
  ✗ FP8 E5M2 → SM 8.9不支持 → SM90+ → ✗✗✗
  ✗ FP8 training → SM89不支持 → ✗✗✗
  ✗ FP8 AllGather(减半通信) → SM90+ → ✗✗✗
  ✗ INT4 MoE专家并行(DeepEP) → SM90+ → ✗✗✗

★ ★ ★ RTX 4090量化最优策略:
  → 训练: BF16 + LoRA → ★★★ 唯一正确训练精度
  → 推理: INT4 + INT8KV → ★★★ 唯一可行推理精度组合
  → 不用: FP8/FP8训练/FP8通信 → SM89不支持
  → ★★★ 结论: BF16训练 + INT4推理 = 各最优 → 分阶段精度!
```

## 参考资料

- vLLM INT4 Triton fallback: notebook/projects/vllm-v0.23-new-features-reading.md
- RTX 4090 benchmark: memory/benchmark-results.md
- RTX 4090 config: notebook/projects/rtx4090-seven-framework-practical-config.md
- RL Training Patterns: notebook/projects/rl-training-design-patterns-comparison.md
- Inference-Training Integration: notebook/projects/inference-training-integration-pipeline-reading.md
- CUDA Graph Memory: tools/cuda_graph_memory_estimator.py results
