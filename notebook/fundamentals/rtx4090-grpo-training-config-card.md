# RTX 4090 GRPO Training Configuration Quick Card

> 2026-06-15 | 7框架×RTX 4090 GRPO训练实战配置速查
> ★ ★ ★ 基于57+深度阅读 → 实战可用的配置 → GPU上线即可运行

## 1. RTX 4090 约束总结

```
★ ★ ★ 硬件约束 (24GB VRAM):
  SM 8.9 (Ada Lovelace)
  GEMM: 169.6 TFLOPS BF16
  HBM: 890.8 GB/s
  CUDA: 12.x
  ✗: NVLS / TMA / FP8 E5M2 / FP8 training / FP8 AllGather
  ✓: INT4 GPTQ Marlin / INT8 KV / BF16 / CUDA graph / FlashInfer / NCCL

★ ★ ★ 内存约束 (7B模型):
  BF16推理: 14GB weights + 10GB KV = 24GB → ✗ 精确占满 → 无headroom
  INT4推理: 3.5GB weights + 5GB INT8KV + 2GB graph + 0.5GB buf = ~11GB → 13GB headroom ✓✓✓
  BF16训练(LoRA): 14GB base + 2.6GB LoRA + 0.5GB optim = ~17GB → 7GB headroom ✓
  PPO训练: ~270GB → ✗✗✗ 完全不可能
  GRPO训练: 17GB → ✓ 唯一可行RL方案

★ ★ ★ PCIe约束:
  7B 8GPU = 0.46x scaling → ✗✗✗ 灾难级
  单GPU = 最优 → 所有分布式方案不可行
```

## 2. rLLM Tinker — ★★★ RTX 4090最优路径

```yaml
# rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode
# ★★★ 最简最快推理-训练集成 → 单GPU唯一最优

framework: rllm
backend: TinkerBackend  # in-process → 零拷贝 → 无IPC → 最简
algorithm: GRPO  # group-relative advantage → 不需要critic → 省50%
lora_rank: 32  # auto-init → attn/mlp/unembed
lora_target: all  # auto → create_lora_training_client_async
bypass_mode: true  # ★ pi_old=rollout logprobs → 省forward → ~40% faster
fused_fwd_bwd_optim: true  # asyncio.gather overlap → faster
reward: rule_based  # math/code → 无GPU → 确定性

# Memory budget
base_model_bf16: 14GB
lora_weights: ~2.6GB  # rank=32, 0.8% params
optimizer_state: ~0.5GB  # CPU_Adam or AdamW small
gradients: ~0.5GB  # LoRA only
activations: ~2GB  # GRPO rollout_n=8
total: ~17GB / 24GB → 7GB headroom ✓

# Training throughput
estimated_tokens_per_sec: ~19,743  # BF16 decode
rollout_n: 8  # GRPO group size
gen_batch_size: 8  # per prompt

# Weight sync
method: zero_copy  # save_weights → GPU-only merge → new SamplingClient
time: ~0ms  # ★★★ 最快 → 无传输 → 无延迟

# After training: merge → HF → INT4 → vLLM
merge_lora: true  # merge into base → 等价全参数训练
save_format: huggingface  # save_pretrained → 直接HF
quantize: GPTQ_INT4  # 3.5GB → 4x省内存
deploy: vLLM_INT4  # 4,791 tok/s → EAGLE → 9,088 tok/s
```

## 3. verl HYBRID — ★★ 同GPU不同进程

```yaml
# verl HYBRID (colocated) + GRPO + LoRA + bypass_mode
# ★ Ray actor → 同PG → 同GPU → 不同进程 → 有IPC overhead

framework: verl
rollout_mode: HYBRID  # same GPU, different process → naive generator
algorithm: GRPO_VECTORIZED  # ★ 推荐 10-100x faster than loop
lora_rank: 32
lora_config:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
bypass_mode: true  # ★ same as rLLM → 省forward
reward: rule_based  # custom_reward_function → no GPU
ref_in_actor: true  # LoRA → same worker → no_lora_adapter → base model ref → 1GPU!

# Memory budget (similar to rLLM but more overhead)
base_model_bf16: 14GB
lora_weights: ~2.6GB
ref_model_overhead: ~0  # disable_adapter() → same GPU
ray_overhead: ~1GB  # Ray actor + IPC buffers
total: ~17.6GB / 24GB → 6.4GB headroom ✓ (but tighter)

# Weight sync
method: naive  # generator → same process → zero-copy
time: ~0ms  # ★ HYBRID mode → generator in same process

# vs rLLM Tinker:
# +: verl更成熟(5K stars)/Ray生态/GPU集群支持
# -: IPC overhead/Ray complexity/more boilerplate
# ★ 单GPU → rLLM Tinker更优; 多GPU → verl更优
```

## 4. DeepSpeed — ★ ZeRO-2+LoRA (no GRPO built-in)

```yaml
# DeepSpeed ZeRO-2 + LoRA + CPU_Adam
# ★ ZeRO-3不可用(3Ψ AllGather→PCIe灾难); ZeRO-2可行但需LoRA
# ✗ DeepSpeed没有内置GRPO → 需要自己实现RL loop

framework: deepspeed
zero_stage: 2  # ★ ZeRO-2 only → partition gradients+optimizer
offload_optimizer: cpu  # CPU_Adam → 省GPU optimizer state
lora_rank: 32
algorithm: custom_GRPO  # 需自实现 → 或用rLLM/verl wrapper

# Memory budget
base_model_bf16: 14GB
lora_weights: ~2.6GB
optimizer_state: ~0GB  # offloaded to CPU
gradients_partition: ~0.5GB  # ZeRO-2 partitioned
total: ~17.1GB / 24GB → 6.9GB headroom ✓

# ★ ★ AutoEP for MoE (EP=1 on RTX 4090)
auto_ep_preset: Qwen3-MoE  # EP=1 → ZeRO-2 → LoRA → single GPU MoE

# ★★ Muon Optimizer (optional)
optimizer: Muon  # Newton-Schulz → 1 buffer vs Adam 2 → 45%省optimizer state
# production: Kimi-K2/GLM-5/DS-V4 → proven

# Weight sync: 需自实现 → checkpoint save → load → 较慢
# ✗ 不适合RL training loop → 无in-process集成
# ★ RTX 4090: 只适合static training → 不适合dynamic RL loop
```

## 5. Megatron-LM — ★ 推理引擎可做rollout

```yaml
# Megatron DynamicInferenceEngine + GRPO cudagraph
# ★ Megatron推理引擎可以做rollout → 但训练侧需要自定义
# ✗ TP>1不可用; PP不可用(单GPU); 只有单GPU模式

framework: megatron
mode: single_gpu  # all PG = singleton → ProcessGroupCollection degrades
tensor_parallel: 1  # ✗ TP>1 → PCIe不可用
pipeline_parallel: 1  # ✗ PP → 单GPU only
data_parallel: 1

# ★ GRPO cudagraph sizing
cudagraph_sizing: linear  # ★ GRPO: linear → stable → vs exponential → regression
# exponential → +13.6% peak memory → ✗ for RTX 4090

# Memory budget (inference only → no training)
inference_bf16: ~24GB  # ✗ 精确占满 → 无headroom → 单独推理不可行
inference_int4: ~11GB  # ✓ 13GB headroom → 可行

# ★ RTX 4090: Megatron用于推理部署 → GRPO训练用rLLM/verl更优
# ★ 不适合作为RL training framework → 太heavy → SM89缺关键kernel
# ✗ NVLS/TMA/FP8 training → SM89不支持
```

## 6. PyTorch — ★ compile only (no distributed)

```yaml
# PyTorch torch.compile + LoRA + GRPO
# ★ compile是纯软件优化 → RTX 4090 ✓ → 但需要自实现GRPO loop
# ✗ FSDP2/DTensor → 需多GPU → PCIe不可用

framework: pytorch
mode: single_gpu_compile  # no distributed → compile optimization only
compile: true  # ★ +15-16% throughput / +48-50% with Float8
fullgraph: false  # ★ FSDP2 hooks → need fullgraph=False (2.12 breaking change)
lora_rank: 32
algorithm: custom_GRPO  # 自实现

# Memory budget
base_model_bf16: 14GB
lora_weights: ~2.6GB
compile_overhead: ~0  # software only → no extra memory
total: ~17GB / 24GB → 7GB headroom ✓

# ★ RTX 4090: compile ✓ → 但不如rLLM Tinker(更简单)
# ★ compile advantage: throughput +15-16% → worth trying as overlay
# ✗ distributed training → PCIe不可用 → FSDP2需要多GPU
```

## 7. MindIE — ✗ RTX 4090不适用

```
★★★ MindIE = 昇腾专用 → RTX 4090完全不可用
  ✗ NPU only (910B/C/D) → GPU无法使用
  ✗ HCCL → 昇腾通信 → NCCL不可替代
  ✗ ATB kernel → Ascend专用 → CUDA无等价
  ✗ FP8需910C → SM89完全不支持

★★★ MindIE vs vLLM-Ascend 关系:
  → MindIE: graph-level整图优化 → 黑盒 → 最高性能 → 但不可控
  → vLLM-Ascend: op-level patch → 保留vLLM调度 → 灵活但稍慢
  → SGLang-Ascend: 用MindIE作为底层 → 黑盒 → 丢失调度控制
  → ★★★ vLLM-Ascend更灵活 → MindIE性能更高 → 不同定位

替代: vLLM-Ascend(适配层) → 但也是NPU专用 → RTX 4090用vLLM GPU版
★★★ DeepEP-Ascend(sgl-kernel-npu) → HCCL EP → fused_deep_moe → NPU only
```

## 8. 推理部署配置 (训练后)

```yaml
# ★★★ RTX 4090 推理最优配置 (训练→merge→INT4→部署)

# Phase 1: 训练完成 → merge LoRA → save HF
merge_command: "rllm merge → save_pretrained → HF format"

# Phase 2: INT4量化
quantize_command: "auto-gptq --model-path ./merged --quant-method gptq --bits 4 --group-size 128"
# ★ Marlin kernel → SM 8.9 ✓ → Triton fallback for non-Marlin shapes
# ★ INT8 KV → kv_cache_dtype="fp8_e4m3" or "int8"

# Phase 3: vLLM INT4 部署
vllm_config:
  model: "./merged-int4"
  quantization: gptq
  kv_cache_dtype: int8  # ★ INT8KV → 5GB vs 10GB → 多轮可行
  gpu_memory_utilization: 0.90
  max_model_len: 8192
  enable_prefix_caching: true  # ★ GRPO rollout prefix复用
  enable_chunked_prefill: true
  cuda_graph_sizes: "linear"  # ★ GRPO场景 → linear sizing

  # EAGLE speculative decoding (optional → ★★★ 极速)
  speculative_config:
    method: eagle
    model: "./eagle-draft"
    num_speculative_tokens: 5
    acceptance_rate: 0.70-0.80
    # → 9,088 tok/s vs 4,791 tok/s → 1.9x加速

# Memory budget (INT4 inference)
int4_weights: 3.5GB
int8_kv_cache: 5GB
cuda_graph_pool: 2GB
buffers: 0.5GB
eagle_draft: 0.5GB  # if using EAGLE
total: ~11.5GB / 24GB → 12.5GB headroom ✓✓✓
```

## 9. 完整训练→部署流程

```
★ ★ ★ ★ ★ RTX 4090 GRPO训练→推理完整路径:

Step 1: rLLM Tinker训练 (~17GB, BF16)
  → GRPO + LoRA-32 + bypass_mode + rule-based reward
  → ~19,743 tok/s training throughput
  → rollout_n=8, group-relative advantage

Step 2: LoRA merge (GPU, <1ms)
  → save_weights_and_get_sampling_client_async
  → GPU-only merge → zero-copy → new SamplingClient
  → ★ merge后等价全参数训练 → 无损失

Step 3: Save HF format (CPU, <1min)
  → save_pretrained → BF16 weights + LoRA merged
  → ★ 直接HF → 最简路径

Step 4: INT4量化 (CPU, ~1min)
  → GPTQ → INT4 weights → 3.5GB
  → INT8 KV config → 5GB KV
  → ★ 4x压缩 → 推理内存可控

Step 5: vLLM INT4推理 (~11GB)
  → INT4+INT8KV+GQA-8+prefix caching+CUDA graph
  → 4,791 tok/s baseline
  → EAGLE → 9,088 tok/s → ★★★★ 极速!

Step 6: 评估 (CPU warm-pool)
  → rllm eval --attempts N → pass@k
  → ★ 不占GPU → CPU warm-pool → 最大化GPU利用率

★ ★ ★ 循环: 评估→再训练→再推理 → 推理scaling+GRPO完美结合
```

## 10. 7框架对比速查表

```
| 框架 | GRPO训练 | 推理部署 | RTX4090可行 | 推荐度 |
|------|---------|---------|-------------|--------|
| ★★★ rLLM Tinker | ✓✓✓(in-process) | INT4→vLLM | ✓✓✓ | ★★★★ |
| ★★ verl HYBRID | ✓✓(Ray colocated) | INT4→vLLM | ✓✓ | ★★★ |
| ★ DeepSpeed Z2 | ✓(需自实现) | ✗(无推理) | ✓ | ★ |
| ★ Megatron | ✗(太heavy) | ✓(INT4 inference) | 推理✓ | ★ |
| ★ PyTorch compile | ✓(需自实现) | ✗(无推理) | ✓compile | ★ |
| ✗ MindIE | ✗(NPU only) | ✗(NPU only) | ✗✗✗ | ✗ |
| ★ PyTorch | compile overlay | ✗(no serving) | ✓ | ★ |

★ ★ ★ 结论: rLLM Tinker = RTX 4090唯一最优 → 最简最快最可行!
```
