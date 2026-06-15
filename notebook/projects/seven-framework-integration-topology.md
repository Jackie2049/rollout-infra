# 7-Framework Integration Topology (2026-06-15)

> ★★★★★ 生产级7框架集成拓扑 → 训练+推理+RL=3层 → verl=中枢 → HF=桥梁
> ★★★ RTX 4090: rLLM Tinker最简 → verl最成熟 → 其他框架辅助

## 1. 3层集成拓扑

```
★★★★★ 7框架生产集成 = 3层:

Layer 1: ★★★ 训练层 (Training)
  DeepSpeed ZeRO → 通用分布式训练 → ZeRO-2+LoRA → RTX 4090辅助
  PyTorch FSDP2 → 未来标准 → BF16+compile → 多GPU首选
  Megatron-LM → 3D并行 → TP+PP+DP → 大规模集群 → RTX 4090 overkill

Layer 2: ★★★★ RL/训练-推理集成层 (RL Training)
  verl → ★★★★ 中枢 → 训练+vLLM/SGLang推理 → weight sync
  rLLM Tinker → ★★★★ RTX 4090最优 → in-process → GRPO+LoRA
  Megatron GRPO → ★ 过重 → 但推理引擎可用 → recipe有

Layer 3: ★★★★ 推理层 (Inference Serving)
  vLLM → ★★★★ 通用GPU推理 → INT4+INT8KV → 4,791→9,088 tok/s
  SGLang → ★★★ 高吞吐推理 → RadixAttention → GRPO rollout更优
  MindIE → ★★★ Ascend NPU推理 → ATB+FP8 → 中国场景
  vLLM-Ascend → ★★★★ Ascend灵活推理 → op-level → 可控

★★★★★ 桥梁:
  HF format → ★★★★★ 所有训练→推理的桥梁 → AI infra的"HTTP"
  PyTorch → ★★★★ 所有框架的底层 → DTensor+compile+NCCL
```

## 2. verl = RL中枢集成

```
★★★★★ verl连接训练+推理 = RL中枢:

verl + vLLM (★★★★★ 最成熟):
  → HYBRID: 同进程 → naive generator → 零拷贝 → ★★★ RTX 4090最优
  → COLOCATED: 同PG不同进程 → CUDA IPC → 多GPU
  → STANDALONE: 不同GPU → NCCL/NIXL → 需NVLink
  → Sleep/Wake: sleep(level=1/2) → 释放GPU → 训练 → wake → 推理
  → Weight sync: naive(0ms) / CUDA IPC(~100ms) / NCCL(PCIe慢) / NIXL(RDMA)
  → ★★★★★ verl HYBRID + vLLM V1 = RTX 4090 GRPO生产路径!

verl + SGLang (★★★ 新增v0.8.0):
  → SGLang rollout engine option → PD disaggregation → 1P:N decode
  → ★★★ SGLang prefix caching更优 → GRPO rollout_n=8 → 7x省
  → ★ 但: verl+SGLang集成仍在早期 → 不如vLLM成熟
  → ★★★ RTX 4090: HYBRID模式不适用(SGLang无HYBRID) → COLOCATED only

verl + DeepSpeed (★★ 分布式训练backend):
  → verl --actor.strategy=deepspeed → ZeRO-2/3 backend
  → ★★★ 多GPU场景 → ZeRO-2+LoRA → 3x内存省
  → ★ RTX 4090: 单GPU → DeepSpeed backend无用 → LoRA more effective
  → ★★★ DeepSpeed几乎停转 → PR merge极低 → ★★★ 不推荐投入!

verl + Megatron (★★ 实验性):
  → verl --actor.strategy=megatron → Megatron-FSDP mode
  → ★★★ 大规模集群 → TP+PP+FSDP → verl管理RL loop
  → ★ RTX 4090: TP/PP不可用 → Megatron backend无用
  → ★★★ 实验性 → 不如FSDP2成熟 → 不推荐RTX 4090

verl + FSDP2 (★★★ 推荐):
  → verl --actor.strategy=fsdp2 → PyTorch原生 → compile兼容
  → ★★★ 多GPU首选 → 2Ψ通信 → composable → 未来标准
  → ★ RTX 4090: world_size=1 → FSDP2无意义 → LoRA more effective
  → ★★★ NVLink集群 → FSDP2+compile → 最优训练路径
```

## 3. rLLM Tinker = RTX 4090最优

```
★★★★★ rLLM Tinker = RTX 4090唯一最优 → in-process → 最简:

集成拓扑 (单GPU):
  → Policy Model (BF16 LoRA-32) → TinkerEngine → SamplingClient
  → Weight Sync: save_weights_and_get_sampling_client_async → GPU-only merge → <1ms
  → Reward: rule-based → CPU execution → no GPU
  → Evaluation: pass@k → CPU warm-pool → no GPU

vs verl (RTX 4090单GPU):
  → rLLM: in-process → 0 IPC → 最简 → bypass auto-default → detach auto-safe
  → verl: Ray actor → IPC overhead → bypass must config → detach must config
  → ★★★★★ rLLM更简更快 → verl更成熟更灵活 → 各有优势
  → ★★★ 实战: rLLM训练 → verl推理部署 → 两者结合!

部署路径:
  → rLLM训练 → LoRA merge → save_pretrained → HF → INT4 → vLLM → 4,791 tok/s
  → ★★★ 6-step路径 → 从训练到推理 → 完整pipeline
```

## 4. HF Format = 所有桥梁

```
★★★★★ HF format = AI infra的HTTP → 所有训练→推理的桥梁:

训练→HF:
  → DeepSpeed ZeRO → universal checkpoint → FP32 → HF safetensors
  → FSDP2 → FSDPModelMerger → HF safetensors → ★★★ 最简洁
  → Megatron → mbridge HF export → safetensors
  → verl → actor checkpoint = HF → ★★★★ 直接推理模型!
  → rLLM → save_pretrained → HF → ★★★★★ 最简路径!

HF→推理:
  → HF → vLLM INT4 → 4,791 tok/s → ★★★★★ GPU推理最优
  → HF → SGLang INT4 → ~5,000 tok/s → ★★★ 高吞吐
  → HF → TRT-LLM → 高性能 → 但构建复杂
  → HF → MindIE FP8 → ★★★ Ascend推理
  → HF → vLLM-Ascend FP8 → ★★★★ Ascend灵活推理

★★★★★ 关键洞察:
  → HF = 所有框架的统一输出格式 → 训练任何框架 → 推理任何框架
  → ★★★★ verl/rLLM → HF → vLLM = 最简最快完整路径!
  → ★★★ DeepSpeed → universal checkpoint → 多步转换 → 最慢
  → ★★★ Megatron → mbridge → 中等复杂度
```

## 5. RTX 4090生产路径选择

```
★★★★★ RTX 4090 生产集成路径:

★★★★★ Path 1: rLLM Tinker (最简最快):
  → TinkerBackend + GRPO + LoRA-32 + bypass_mode → ~17GB
  → Weight sync: zero-copy → <1ms
  → → save_pretrained → HF → INT4 → vLLM → 4,791 tok/s → EAGLE → 9,088 tok/s
  → ★★★★★ 从训练到推理 → 最简最快 → RTX 4090最优!

★★★★ Path 2: verl HYBRID + vLLM (更成熟):
  → verl HYBRID + GRPO + LoRA + bypass_mode + detach_metrics → ~17.6GB
  → Weight sync: naive → 0ms (same process)
  → → merge → HF → INT4 → vLLM → 4,791 tok/s
  → ★★★★ 更成熟 → Ray生态 → GPU集群支持 → 但单GPU不如rLLM

★★★ Path 3: DeepSpeed ZeRO-2 (辅助训练):
  → ZeRO-2 + LoRA + CPU_Adam → ~17GB → 无GRPO → 需自实现RL
  → → universal ckpt → FP32 → HF → INT4 → vLLM
  → ★★ 辅助 → 非RL → 适合static finetuning → 不适合GRPO

★★ Path 4: Megatron推理 (推理引擎):
  → DynamicInferenceEngine → INT4 → CUDA graph → NCCL → 单GPU
  → ★★ 推理可用 → 但不如vLLM成熟 → SM89缺关键kernel
  → ★★★ 不适合RL训练 → 太重 → 单GPUoverkill

✗✗✗ Path 5: MindIE (NPU only → RTX 4090不适用)
  → NPU推理 → 910C → FP8 → ATB → 高性能 → 但GPU不可用
  → ★★★ 中国NPU场景 → 但RTX 4090完全不行
```

## 6. 跨框架数据流

```
★★★★★ 生产数据流拓扑:

Training → Checkpoint → Conversion → Quantization → Serving
    ↓          ↓           ↓           ↓           ↓
DeepSpeed   universal     FP32→HF    GPTQ INT4   vLLM/SGLang
FSDP2       ModelMerger   per-param  INT8 KV     INT4+EAGLE
Megatron    mbridge       safetensor W4A8 Triton  TRT-LLM
verl        actor ckpt    HF direct  FP8(910C)   MindIE/vLLM-Ascend
rLLM        save_pretrained HF      INT4+INT8KV vLLM GPU

★★★★★ RL Training特殊数据流:
  Prompt → Rollout(vLLM/SGLang/Tinker) → Reward(rule-based) →
  Advantage(GRPO) → Policy Update(LoRA) → Weight Sync →
  New Rollout → 循环!

  ★★★ rLLM: 全in-process → 无跨框架传输 → 最简!
  ★★★★ verl: Rollout(vLLM) → Driver → Advantage → Update → Weight Sync(vLLM)
  ★★★ verl V1: TransferQueue → zero-copy KV reuse → 49.1% improvement!
```

## 7. 关键洞察

1. ★★★★★ **3层拓扑**: 训练层+RL集成层+推理层 → verl=中枢 → HF=桥梁
2. ★★★★★ **verl=RL中枢**: 连接vLLM+SGLang+DeepSpeed+Megatron+FSDP2 → 多backend选择
3. ★★★★★ **rLLM=RTX 4090最优**: in-process → 无跨框架 → 最简最快 → 生产首选
4. ★★★★ **HF=AI infra HTTP**: 所有训练→推理统一格式 → verl/rLLM最简路径
5. ★★★★ **RTX 4090最优路径**: rLLM训练 → HF → INT4 → vLLM推理 → EAGLE → 9,088 tok/s
6. ★★★ **DeepSpeed停转**: PR merge极低 → 不推荐 → FSDP2替代
7. ★★★ **verl+SGLang**: 新增 → PD disaggregation → prefix caching更优 → 但不成熟
8. ★★★ **跨框架数据流**: RL循环 → Prompt→Rollout→Reward→Advantage→Update→Sync → 循环
9. ★★★★ **v0.23.0影响**: MRv2量化不支持 → RTX 4090仍用V1 → FP8 guard仍需

---

Sources:
- ★★★ verl integration: `notebook/projects/verl-worker-lifecycle-ray-weight-sync-reading.md`
- ★★★ rLLM Tinker: `notebook/projects/rllm-tinker-backend-deep-reading.md`
- ★★★ Inference-training integration: `notebook/projects/inference-training-integration-pipeline-reading.md`
- ★★★ Checkpoint deployment: `notebook/fundamentals/checkpoint-to-inference-deployment-guide.md`
- ★★★ NPU ecosystem: `notebook/projects/npu-inference-ecosystem-comparison.md`
- ★★★ vLLM v0.23.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.23.0
- ★★★ verl v0.8.0: https://github.com/volcengine/verl/releases/tag/v0.8.0
