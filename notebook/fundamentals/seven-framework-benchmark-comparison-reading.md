# 七框架Benchmark Cross-Comparison — RTX 4090视角

> 2026-06-16 | 7-framework benchmark | DeepSpeed vs Megatron | vLLM vs SGLang vs TRT-LLM | verl vs rLLM
> ★★★★★ 跨框架benchmark对比 → RTX 4090每框架最优配置与性能特征
> ★★★★★ verl GRPO throughput ~2x rLLM Tinker → 但rLLM simplicity + in-process 更优

## 1. ★★★★★ GRPO训练 Throughput对比 (verl vs rLLM vs DeepSpeed)

```
★★★★★★★ GRPO训练throughput对比 (Qwen2.5-7B, GSM8K):

| Framework | Throughput | 简单度 | RTX 4090可行 | 关键优势 |
|-----------|-----------|--------|-------------|---------|
| verl (Ray+vLLM) | ~2x baseline | ★★★ 中 | ★★★★ 可行(bypass+detach) | ★★★★★ CPPO+community |
| rLLM Tinker | ~1x baseline | ★★★★★ 最高 | ★★★★★ 立即可用 | ★★★★★ in-process+bypass |
| DeepSpeed (ZeRO-2) | ~1.5x baseline | ★★★ 中 | ★★★ 有限 | ★★★★★ MoE AutoEP唯一 |
| Megatron core | ✗ crash | ★ 低 | ✗✗✗ 不可行 | ✗ singleton PG crash |

★★★★★★★ verl throughput ~2x rLLM原因:
  → vLLM continuous batching → higher rollout throughput → more samples/min
  → 但Ray overhead → cross-process → detach_metrics MUST → memory overhead
  → → ★★★★★ throughput ≠ efficiency → rLLM in-process → zero Ray overhead → simpler

★★★★★★★ rLLM Tinker throughput相对低原因:
  → HF generate → no continuous batching → sequential rollout → slower
  → 但in-process → zero IPC → zero detach → 简单 → 内存友好
  → → ★★★★★ 简单度 > throughput → RTX 4090内存有限 → 简单路径更可行

★★★★★★★ DeepSpeed throughput ~1.5x原因:
  → ZeRO-2 gradient partitioning → 通信overhead → slower than verl's vLLM rollout
  → 但MoE AutoEP → unique → dense model不如verl/rLLM
  → → ★★★★★ Dense model → rLLM/verl → MoE → DeepSpeed唯一选择

★★★★★★★ RTX 4090 GRPO推荐排序 (综合throughput+可行+简单):
  → #1 rLLM Tinker → simplest → in-process → bypass → 立即可用 → ★★★★★
  → #2 verl ReMax → TransferQueue sync → greedy baseline → 高throughput → ★★★★
  → #2 verl CPPO → best trust region → 高throughput → more config → ★★★★
  → #3 DeepSpeed ZeRO-2 → MoE唯一 → 但dense不如 → ★★★ 有限
  → #4 Megatron → crash → ✗✗✗ not viable
```

## 2. ★★★★★ 推理Serving对比 (vLLM vs SGLang vs TRT-LLM)

```
★★★★★★★ 推理serving benchmark (Llama-70B, 8xA100):

| Framework | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Memory (GB) | Setup |
|-----------|-------------------|-----------|-----------|-------------|-------|
| vLLM V1 | ~2200 | ~120 | ~25 | ~35 | pip install |
| SGLang | ~2400 | ~100 | ~22 | ~33 | pip install |
| TRT-LLM | ~2800 | ~80 | ~18 | ~30 | 复杂setup |
| MindIE (Ascend) | ~2600* | ~90* | ~20* | ~28* | CANN+HCCL |

★★★★★★★ vLLM vs SGLang vs TRT-LLM关键差异:

vLLM:
  → ★★★★★ 最灵活 → CLI flags → community → easiest setup
  → ★★★★★ HMA prefix caching → per-attention-type KV grouping → smart scheduling
  → ★★★ Continuous batching → preemption → watermark → 健壮serving
  → ★★★★ RTX 4090: enforce_eager=True → SM89 batch invariance → throughput ~10-15% lower

SGLang:
  → ★★★★★ RadixAttention → tree-based prefix caching → 多轮对话最优
  → ★★★★ torch.compile → 更低latency → 但SM89 batch invariance risk!
  → ★★★ PD disaggregation → prefill:decode分离 → 独特feature
  → ★★★★★ SGLang router → cache-aware routing → prefix reuse最大化

TRT-LLM:
  → ★★★★★ 最高throughput → kernel-level optimization → NVIDIA production
  → ★★★★★ FP8/INT8 quantization → 最优推理性能 → 但RTX 4090 FP8 fallback to FP16!
  → ★★ Setup复杂 → 需要模型转换 → 重建engine → 不灵活
  → ★★★ INT8 → RTX 4090推荐 → FP16 not FP8 → SM89 specific
  → ★★★ LoRA plugin → inference deployment → max_lora_rank=64

★★★★★★★ RTX 4090推理推荐:
  → 简单serving → vLLM → enforce_eager=True → INT8 KV → 最简单
  → 最高throughput → TRT-LLM INT8 → 需要模型转换 → 复杂setup
  → 多轮对话 → SGLang → RadixAttention → prefix caching最优
  → ★★★★★ RTX 4090: vLLM最实用 → 但SM89 batch invariance MUST enforce_eager!
```

## 3. ★★★★★ Pretraining Throughput对比 (DeepSpeed vs Megatron)

```
★★★★★★★ Pretraining throughput (GPT-7B, multi-GPU):

| Scale | DeepSpeed ZeRO-3 | Megatron-LM | 差异 |
|-------|------------------|-------------|------|
| 8 GPU | ~160 TFLOPS | ~150 TFLOPS | +6.7% |
| 32 GPU | ~620 TFLOPS | ~600 TFLOPS | +3.3% |
| 128 GPU | ~2400 TFLOPS | ~2500 TFLOPS | -4% |
| 512 GPU | ~9000 TFLOPS | ~10000 TFLOPS | -10% |

★★★★★★★ Key insight:
  → 小规模 (≤32) → DeepSpeed略优 → ZeRO-2通信优化 → 更好throughput
  → 大规模 (≥128) → Megatron更优 → TP+PP重叠 → 更好scaling
  → ★★★★★★ Megatron在大规模更优 → DeepSpeed在小规模更优 → 选择取决于规模!

★★★★★★★ RTX 4090 pretraining:
  → 单GPU → 不需要分布式 → DeepSpeed ZeRO-3无意义 → Megatron crash
  → → ★★★★★ RTX 4090 pretraining = LoRA+compile → 不用ZeRO或TP!
  → → DeepSpeed ZeRO-2 for LoRA training → 省optimizer memory → 有价值
  → → 但rLLM Tinker → in-process → 简单 → LoRA training最优路径
```

## 4. ★★★★★ MindIE/vLLM-Ascend vs vLLM (Ascend NPU)

```
★★★★★★★ Ascend NPU推理benchmark (Qwen2-7B, 910B):

| Framework | Throughput | TTFT | TPOT | Setup |
|-----------|-----------|------|------|-------|
| MindIE (official) | ~1800 tok/s | ~110ms | ~25ms | CANN+HCCL |
| vLLM-Ascend | ~1600 tok/s | ~130ms | ~28ms | pip install |
| SGLang-Ascend | ~1500 tok/s | ~140ms | ~30ms | pip install |

★★★★★★★ MindIE最高 → 但closed source → 不灵活
  → ★★★★★ vLLM-Ascend推荐 → op-level patch → 灵活 → BudgetRefiner SLO
  → SGLang-Ascend → graph-level → 黑盒 → 不推荐production

★★★★★★★ BudgetRefiner SLO impact:
  → 无SLO → vLLM V1 → 可能SLA violation → 不稳定
  → 有SLO → BudgetRefiner → decode-first → SLO-guaranteed → ★★★★★ production-grade
  → ★★★★★ BudgetRefiner → 可贡献vLLM upstream → GPU也可用 → portable!

★★★★★★★ RTX 4090影响:
  → Ascend benchmark → 不适用于RTX 4090 → NPU-only
  → 但BudgetRefiner理念 → portable → GPU也可用 → ★★★★★ future contribution
```

## 5. ★★★★★ PyTorch Compile vs Eager (SM89)

```
★★★★★★★ PyTorch compile vs eager benchmark:

| Config | Throughput | Latency | Memory | Batch Invariance |
|--------|-----------|---------|--------|-----------------|
| eager (no compile) | baseline | baseline | baseline | ★★★★★ ✓ invariant |
| compile + graphs (SM90) | +15-25% | -10-20% | ~same | ★★★★★ ✓ invariant |
| compile + graphs (SM89) | +15-25% | -10-20% | ~same | ✗✗✗ ✗ batch-dependent! |
| compile, no graphs (SM89) | +10-15% | -5-10% | ~same | ✗✗✗ ✗ batch-dependent! |

★★★★★★★ ★★★★★★★ Key finding: compile alone breaks batch invariance on SM89!
  → 不只是CUDA graphs → torch.compile alone → Inductor fusion → mean inline → batch-dependent!
  → → SM89 MUST: enforce_eager=True → disable compile → invariant → but -10-15% throughput

★★★★★★★ Inductor SM<90 Fusion Guard impact (if PR merged):
  → compile active → reduction不fusion → mean override有效 → batch invariant ✓
  → → ★★★★★ 可以启用compile → 不需要enforce_eager → +10-15% throughput恢复!
  → → → ★★★★★★ spec decode可用 → inference latency -40-50%!
  → → → ★★★★★★★ Inductor Fusion Guard = RTX 4090最大throughput improvement!
```

## 6. ★★★★★ verl vs rLLM Tinker Detailed Comparison

```
★★★★★★★ verl vs rLLM Tinker GRPO训练详细对比:

| 维度 | verl (Ray+vLLM) | rLLM Tinker (in-process) |
|------|-----------------|--------------------------|
| Setup | ★★★ 中 (Ray+vLLM+CPPO config) | ★★★★★ 最简单 (rllm train) |
| Throughput | ★★★★★ ~2x (vLLM continuous batching) | ★★★ ~1x (HF generate sequential) |
| Memory | ★★★ 中 (detach_metrics MUST) | ★★★★★ 高 (in-process, no detach) |
| Trust Region | ★★★★★ CPPO+bypass (最优) | ★★★ GRPO+bypass (标准) |
| MRv2 Handling | ★★ ZERO → MUST禁用 | N/A (不用vLLM) |
| SM89 Batch | ★★★ MUST enforce_eager | N/A (不用vLLM) |
| LoRA | ★★★ config-based | ★★★★★ auto-init |
| Bypass | ★★★ config flag | ★★★★★ default true |
| Multi-GPU | ★★★★★ HYBRID/COLOCATED | ✗ single GPU only |
| VLM | ★★★★★ mature (mRoPE+TP) | ★★★ experimental (Geo3K) |
| Agent RL | ★★★ ContinuousToken (open) | ✗ not yet |
| ReMax | ★★★★★ merged (#6340) | ✗ not yet |
| CPPO | ★★★★★ open (#6731) | ✗ not yet |
| Community | ★★★★★ Large (21k stars) | ★★★ Small |

★★★★★★★ RTX 4090选择指南:

简单+快速实验 → rLLM Tinker → ★★★★★ 一行命令 → bypass默认 → auto LoRA
高throughput+CPPO → verl → ★★★★★ vLLM continuous batching → CPPO+bypass
多模态 → verl → ★★★★★ VLM成熟
长CoT数学 → verl CPPO → ★★★★★ prefix-weighted trust region → 防drift
Agent RL → verl → ★★★★★ ContinuousToken → multi-turn → 未来

★★★★★★★ 综合RTX 4090推荐:
  → #1 rLLM Tinker → 简单 → 立即可用 → 不需要vLLM → 不受SM89 bug影响
  → #2 verl ReMax → 高throughput → greedy baseline → 简单config → TransferQueue sync
  → #2 verl CPPO → best trust region → long CoT → but more config + enforce_eager
```

## 7. ★★★★★ RTX 4090 Performance Optimization路径

```
★★★★★★★ RTX 4090性能优化路径 (按impact排序):

Phase 1 (当前): 基础GRPO训练
  → rLLM Tinker → Qwen2.5-3B → LoRA-32 → bypass → ~50-80 tok/s training
  → 或 verl → enforce_eager=True → INT8 KV → bypass → detach → ~100-160 tok/s

Phase 2 (Inductor PR后): compile+graphs启用
  → Inductor SM<90 Fusion Guard merged → SM89可以compile → batch invariant ✓
  → → enforce_eager=False → CUDA graphs可用 → vLLM throughput +10-15%
  → → spec decode可用 → inference latency -40-50%
  → ★★★★★★ 这是RTX 4090最大throughput提升!

Phase 3 (MAGI prefix-tree): GRPO KV dedup
  → MAGI prefix-tree merged → GRPO group prefix dedup → KV memory省一半
  → → 更大group_size → 更多rollout per prompt → GRPO quality提升
  → ★★★★★ RTX 4090 memory freed → 可训练更大model

Phase 4 (SM120): RTX 5090 FP4/MXFP4
  → RTX 5090 → SM120 → FP4 native → vLLM FP4 kernel → inference革命
  → → INT4被FP4替代 → better accuracy + HW accel → future contribution window
```

## 参考
- verl benchmarks: https://github.com/verl-project/verl (GRPO throughput examples)
- rLLM Tinker benchmarks: cookbooks/math/ (training time examples)
- DeepSpeed benchmarks: https://www.deepspeed.ai/benchmark/
- Megatron benchmarks: https://github.com/NVIDIA/Megatron-LM#performance
- vLLM benchmarks: https://vllm.readthedocs.io/en/latest/performance.html
- SGLang benchmarks: https://github.com/sgl-project/sglang#performance
- TRT-LLM benchmarks: https://github.com/NVIDIA/TensorRT-LLM#performance
- vLLM #39096: SM89 batch invariance → enforce_eager → throughput impact
- 相关笔记: rtx4090-grpo-trust-region-comparison.md, rtx4090-verl-cppo-grpo-training-guide.md, verl-v080-latest-developments-2026-06-reading.md, seven-framework-advisor.py
