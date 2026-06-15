# verl v0.8.0 Features 源码级深度阅读

> 2026-06-15 | verl版本: v0.8.0 (released 2026-06-01)
> 源码: https://github.com/verl-project/verl/releases/tag/v0.8.0
> 核心: Gemma4多模态GRPO→capability detection; 统一Trainer ABC→sync/colocate_async/separate_async; ROCm HIP→PlatformROCm; detach graph→+0.27GiB/mb→16.2GiB固定; OPD→3 backend; SGLang LoRA+PD disaggregation; TransferQueue→49.2% E2E; RTX 4090分析

## 目录

1. [Gemma4 多模态 GRPO](#1-gemma4-multimodal-grpo)
2. [统一 Trainer 抽象](#2-unify-trainer-abstraction)
3. [ROCm/HIP 支持](#3-rocmhip-platform)
4. [Detach Loss Metrics 修复](#4-detach-loss-metrics-fix)
5. [On-Policy Distillation (OPD)](#5-on-policy-distillation)
6. [SGLang LoRA + PD 分离](#6-sglang-lora--pd-disaggregation)
7. [其他重要特性](#7-other-important-features)
8. [v0.8.0 Breaking Changes](#8-breaking-changes)
9. [RTX 4090 实战分析](#9-rtx-4090)
10. [与 rLLM v0.3 对比](#10-vs-rllm-v03)

---

## 1. Gemma4 Multimodal GRPO

> PR #6715 | 7 additions, 2 deletions | merged 2026-06-12
> 相关: #6341 (issue), #6030 (Gemma4 FSDP SFT), #6406 (Gemma4 tool parser)

### 核心问题

verl的VLM管道是为Qwen-VL设计的 — M-RoPE position ids, `image_grid_thw`, 严格的每图片token验证 — 所以Gemma4在rollout后处理和FSDP forward都失败.

### ★ 三处 Capability Detection 修复

```
★ 设计原则: capability-detected(keyed on processor capabilities, not model name)
  → Qwen-VL路径完全不变 → 新分支只在Gemma4触发

1. verl/utils/tokenizer.py (hf_processor):
   → 添加Gemma4Processor case
   → Gemma用标准1D RoPE(no get_rope_index to bind)
   → Gemma processor严格验证每个图片placeholder → 拒绝verl重建的placeholder-stripped text
   → ★ 禁用Gemma4的image placeholder验证 → image features仍然正常产出

2. verl/experimental/agent_loop/agent_loop.py (_compute_position_ids):
   → 当processor没有get_rope_index → fallback到mask-based 1D positions
   → Gemma用mm_token_type_ids在attention mask中编码图片结构 → 不在M-RoPE中
   → Qwen有get_rope_index → 其分支不变

3. verl/utils/model.py (extract_multi_modal_inputs):
   → 添加mm_token_type_ids到_VARLEN_MULTI_MODAL_KEYS
   → per-sample variable-length token-type mask → pad-batched instead of torch.cat
   → torch.cat在differing sequence lengths失败
   → Qwen把mm_token_type_ids消费进M-RoPE → 永不到达此路径
```

### Gemma4 特殊限制: head_dim=512

```
★ ★ Gemma4 global(full-attention) layers用head_dim=512
  → FlashAttention-2/3限制: head_dim ≤ 256
  → ★ 必须用SDPA(attn_implementation=sdpa)

配置示例:
actor_rollout_ref.model.path=google/gemma-4-E2B-it \
+actor_rollout_ref.model.override_config.attn_implementation=sdpa \
actor_rollout_ref.model.use_remove_padding=False
```

### 多模态 GRPO 数据流变化

```
★ 多模态数据流 vs 纯文本:

[纯文本 GRPO]:
  prompt(text) → tokenizer → input_ids → rollout → response_ids → reward → advantage → update

[多模态 GRPO (Gemma4/Qwen-VL)]:
  prompt(text+<image>) → processor → input_ids + mm_inputs → rollout → response_ids + mm_token_type_ids
                                                    ↓
                                               extract_multi_modal_inputs
                                                    ↓
                                          pad-batched mm_data → FSDP forward
                                                    ↓
                                          mask-based position_ids(Gemma) / M-RoPE(Qwen)

★ ★ 关键差异:
  1. Qwen-VL: M-RoPE → get_rope_index → 3D position_ids
  2. Gemma4: 1D RoPE + mm_token_type_ids → attention mask编码
  3. 两种路径通过capability detection自动切换 → 代码共享
```

### 验证结果

Gemma-4-E2B-it, GRPO, FSDP(actor/ref) + vLLM rollout, attn_implementation=sdpa, H200:
- 训练稳定, reward和eval metrics随训练增长
- Qwen-VL路径不受影响

---

## 2. Unify Trainer Abstraction

> PR #6710 | 2037 additions, 1151 deletions | merged 2026-06-12
> ★ ★ 这是v0.8.0最重要的架构改动

### 核心设计: PPOTrainer ABC + Register Pattern

```
★ ★ 三层抽象:

verl/trainer/ppo/v1/
├── trainer_base.py       (1447行) — ★ PPOTrainer ABC
│   ├── TRAINER_REGISTRY: dict[str, type]
│   ├── @register_trainer(name) — 类装饰器
│   ├── get_trainer_cls(name) — 查询
│   ├── init() → _setup() + on_init_end()
│   ├── fit() → 主循环
│   ├── step() → 10步数据流
│   ├── ReplayBuffer → TransferQueue采样
│   ├── agent_loop_manager → AgentLoopManagerTQ
│   ├── LLMServerManager → RolloutReplica管理
│   ├── CheckpointEngineManager → 权重同步
│
├── trainer_sync.py       (42行) — @register_trainer("sync")
│   → on_init_end → update_weights
│   → on_step_end → update_weights + sleep
│   → on_sample_end → sleep_replicas
│
├── trainer_colocate_async.py (52行) — @register_trainer("colocate_async")
│   → get_llm_client → FullyAsyncLLMServerClient
│   → on_train_begin → warmup batches
│   → on_step_end → update_weights + resume_generation
│   → on_sample_end → abort + sleep
│
├── trainer_separate_async.py (150行) — @register_trainer("separate_async")
│   → NotImplementedError! — 尚未完成
│   → 设计: HybridEngineMode(TRAINER/ROLLOUT)切换
│   → standalone LLMServerManager + GlobalLoadBalancer
│
├── replay_buffer.py      (150行) — TransferQueue KV存储
├── agent_loop_tq.py      (237行) — AgentLoopWorkerTQ/ManagerTQ
├── utils.py              (92行) — compute_advantage_for_multi_trajectories
```

### ★ ★ Hook Pattern: ABC定义生命周期hooks

```
PPOTrainer (ABC):
  init():
    _setup()          → 初始化10步(WorkerGroup/LLMServerManager/Checkpoint/Reward)
    on_init_end()      → ★ 子类override → sync=update_weights; async=warmup

  fit():
    on_train_begin()   → ★ async: warmup batches
    for each step:
      on_sample_begin()  → ★ async: 可能switch_to_rollout
      step() → 10步数据流
      on_sample_end()    → ★ sync: sleep; async: abort+sleep
      on_step_end()      → ★ sync: update_weights; async: update+resume

★ ★ Legacy main_ppo.py → 864行重复逻辑 → V1 PPOTrainer → 3个42-150行子类
  → 每个子类只override关键hooks → 共享逻辑在base中!
```

### PPOTrainerSync vs PPOTrainerColocateAsync 差异

```
★ ★ sync (单GPU最优):
  1. Trainer和rollout colocated
  2. Partial rollout disabled
  3. 顺序: sample → sleep → train → wake → repeat

★ ★ colocate_async (多GPU overlap):
  1. Trainer和rollout colocated (same GPU group)
  2. Partial rollout enabled → streaming generation
  3. warmup: 预填充TransferQueue → 训练开始时已有数据
  4. 顺序: warmup → abort → sleep → train → wake+resume → sample continues

★ ★ separate_async (NotImplementedError, 未来):
  1. Trainer和rollout在不同GPU
  2. HybridEngineMode切换(TRAINER/ROLLOUT)
  3. 空闲trainer GPU可临时做rollout → elastic scheduling
  4. standalone LLMServerManager + GlobalLoadBalancer add/remove
```

### ReplayBuffer: TransferQueue-based GRPO 采样

```
★ ★ ReplayBuffer核心设计 (replay_buffer.py):

TransferQueue KV存储:
  key = {uid}_{session_id}_{index}
  tag = {"is_prompt": True/False, "status": "pending/running/finished/failure"}
  value = {input_ids, response_ids, response_mask, ...}

GRPO group sampling control:
  1. 写入prompt → tag={is_prompt:True, status:"pending"}
  2. 所有session开始 → status="running"
  3. 所有session完成 → status="finished"(成功) / "failure"(有session失败)
  4. ★ 只有finished/failure的prompt可以被sample

sample()流程:
  → _sync_metadata_from_transfer_queue() → tq.kv_list()
  → while finished+failure < batch_size: sleep(poll_interval) → sync again
  → 选择finished+failure prompts → tq.kv_clear() → 返回KVBatchMeta

★ ★ KVBatchMeta只有keys+tags → 不搬运tensor!
  → tensor通过TransferQueue zero-copy传递 → 省去DataProto ~4MB/batch开销
```

### compute_advantage_for_multi_trajectories

```
★ ★ AgentLoop多轨迹 → GRPO advantage处理 (utils.py):

问题: AgentLoop一次prompt可能产出多个trajectory → 每个有自己的response_ids
  → key格式: {uid}_{session_id}_{index}
  → 需要只用最终session的output来计算GRPO advantage

算法:
  1. 找每个session_key的最大index → final_sessions: {uid}_{session_id} → (index, row_index)
  2. select_idxs(final_indices) → 只用最终output → compute_advantage(GRPO)
  3. first_nnz_indices → 取advantages中第一个valid token的score
  4. ★ scatter → scores[row_to_local_index] → broadcast到所有session的轨迹
  5. scores * response_mask → 非LLM token不参与loss

★ 非GRPO estimator(GAE等) → 直接委托给original compute_advantage → 不变
```

### 配置示例

```yaml
# Sync trainer (RTX 4090单GPU)
trainer:
  v1:
    backend: sync  # PPOTrainerSync

# Colocate async trainer (多GPU)
trainer:
  v1:
    backend: colocate_async
    colocate_async:
      num_warmup_batches: 4  # TransferQueue预热

# TransferQueue config
transfer_queue:
  enable: true
```

---

## 3. ROCm/HIP Platform

> PR #6702 | 82 additions | merged 2026-06-12
> CI: PR #6668 (AMD ROCm MI300 e2e_ppo_trainer)

### ★ ★ PlatformROCm: CUDA base + ROCm extensions

```
★ 设计: PlatformROCm(PlatformCUDA) → 继承+override → 不重写!

@PlatformRegistry.register(platform="amd")
class PlatformROCm(PlatformCUDA):

  vendor_name = "amd"  # device_name保持"cuda"(PyTorch ROCm通过hipify暴露torch.cuda.*)

  is_platform_available():
    → torch.version.hip is None → False(排除NVIDIA)
    → torch.version.hip != None → rocm-smi check → /opt/rocm/bin/rocm-smi fallback
    → 最终: torch.cuda.is_available() (CPU-only Ray actor)

  rollout_env_vars():
    → 继承CUDA的env_vars
    → ★ SGLANG_USE_AITER=os.environ.get("SGLANG_USE_AITER", "1")
      → AITER = AMD Inference Transformer Engine Replacement
      → 路由SGLang的RMSNorm/RoPE/MoE/quant kernel到AITER
      → 用户可override: export SGLANG_USE_AITER=0 → fallback到vLLM kernel

  ray_noset_envvars():
    → 继承CUDA的 + HIP/ROCR NOSET vars
    → CUDA_VISIBLE_DEVICES保持primary(不改HIP_VISIBLE_DEVICES) → 避免flip读写冲突

★ PlatformCUDA.is_platform_available():
  → torch.version.hip is not None → 返回False → 自动检测选择ROCm
```

### ROCm 验证

8x gfx942 / MI3xx, ROCm 7.0:
- `get_platform()` → PlatformROCm (不fallback到PlatformCUDA)
- Qwen3-8B, FSDP, GRPO end-to-end → RCCL通信正常
- AITER enabled → SGLang kernel走AITER路径

### RTX 4090 vs ROCm MI300

```
★ RTX 4090: NVIDIA → PlatformCUDA → ROCm不适用
★ AMD MI300: ROCm → PlatformROCm → AITER加速 → RCCL(类似NCCL)

AMD MI3xx优势:
  - HBM3 192GB → 比4090 24GB大8倍 → PPO(critic+actor+ref)可能可行
  - NVLink → PCIe灾难不存在 → 多GPU scaling正常
  - AITER → 专用kernel → 部分场景比vLLM CUDA kernel更快

AMD MI3xx劣势:
  - FP8 E4M3支持 → 但FP8 E5M2不如H100(HW层面)
  - ROCm生态不如CUDA → 某些kernel未优化
  - FlashAttention ROCm → slower than NVIDIA版本
```

---

## 4. Detach Loss Metrics Fix

> PR #6699 | 125 additions | merged 2026-06-11
> Issue #6698: per-micro-batch GPU memory leak → OOM with LoRA + long sequences
> ★ ★ ★ 这是v0.8.0最重要的bug fix!

### ★ ★ ★ 根因分析 (Issue #6698)

```
★ ★ ★ FSDPEngineWithLMHead.forward_step返回model_output(log_probs/entropy)仍然attached to autograd graph!

forward_backward_batch():
  for micro_batch in micro_batches:
    loss, meta_info = forward_step(micro_batch)
    loss.backward()
    output_lst.append(meta_info)  ← ★ meta_info中的model_output仍pin graph!

  → ★ output_lst持有每个micro_batch的meta_info → 直到整个batch完成才释放
  → ★ 每个training micro_batch的autograd graph被retained → 累积OOM!

具体机制:
  1. LoRA → enable_input_require_grads() → embedding output requires_grad=True
  2. Gradient checkpointing → checkpoint frame保存embedding output
  3. loss.backward() → 但graph被output_lst中的model_output anchor → frame不释放
  4. ★ 每个backward micro_batch永久保留: embedding output + gradient buffer
     → ~2 × [total_nnz, hidden] bf16 → ~139 MiB each at 16k tokens

内存证据(torch.cuda.memory_allocated()):
|  | mb #250 | #300 | #350 | #400 |
| before | 24.8 GiB | 37.9 GiB | ~50 GiB | 64 GiB → ★ OOM! |
| after  | 16.2 GiB | 16.2 GiB | 16.2 GiB | 16.2 GiB ★ ✅ |

★ ★ 每个training micro-batch → +0.27 GiB → 400 micro-batches → +108 GiB → OOM!
  → fix后 → 16.2 GiB constant → ✅
```

### 修复方案

```python
# verl/workers/engine/fsdp/transformer_impl.py (FSDPEngineWithLMHead.forward_step):

# ★ ★ 在构建output dict之前detach所有有grad_fn的tensor:
model_output = {
    key: value.detach() if torch.is_tensor(value) and value.grad_fn is not None else value
    for key, value in model_output.items()
}
output = {
    "model_output": model_output,
    "loss": loss.detach().item(),  # ← loss也detach
    "metrics": metrics,
}

return loss, output  # ← live loss仍然返回给backward() → 正确!
```

### ★ 为什么之前没被发现?

```
1. 全参训练(无LoRA) → enable_input_require_grads不触发 → embedding不requires_grad
   → checkpoint frame不保存embedding → graph不pin → 无泄漏!

2. ★ 只有LoRA + gradient checkpointing + long sequences → 才触发!
   → 短序列(~1k) → embedding很小 → 泄漏不明显
   → 长序列(~16k) → embedding ~139 MiB → 400 micro-batches → OOM

3. GRPO rollout_n=8 → 40 prompts × 8 = 320 trajectories → ~400 micro-batches
   → ★ ★ ★ RTX 4090 LoRA GRPO + long sequences → 必然OOM!

★ ★ Metric.append已经做value.detach().item() → scalar metric不泄漏
  → 真正的anchor是model_output中的log_probs/entropy → 非scalar tensor!
```

### Regression Test

```python
# tests/workers/test_engine_forward_step_detach_on_cpu.py:
# → 微型reproduction → frozen embedding + requires_grad_ output + checkpointed block + weakref
# → 验证: unfixed=red, fixed=green
# → CPU-only → cpu_unit_tests → 不需要GPU
```

---

## 5. On-Policy Distillation (OPD)

> PR #5041 | merged 2026-04-10 | 跨FSDP+Megatron+VeOmni
> Doc: https://verl.readthedocs.io/en/latest/algo/opd.html

### ★ ★ OPD = 用teacher model的log_probs来训练student

```
★ ★ 两种loss:
1. Forward top-K KL loss:
   → 用teacher top-k logits估计forward KL
   → student学习teacher的top-k分布 → 直接backprop

2. Reverse KL loss (estimator):
   → 用k1/k3 estimator估计reverse KL
   → → PPO-style → negative distillation loss作为reward

两种update方式:
1. Supervised: distillation loss直接backprop → 如arXiv:2306.13649
2. Policy gradient: negative distillation loss作为reward → 如on-policy distillation

★ ★ ★ Multi-teacher支持 (#6051):
  → 单teacher / 多teacher → 不同维度 → 更强distillation
  → Teacher logprobs用vLLM server计算 → 不占用training GPU
```

### OPD 数据流

```
★ ★ OPD数据流:

┌───────────────────┐
│ Student Model      │ ← FSDP/Megatron/VeOmni训练
│ (actor worker)     │
└───────────────────┘
        ↓ rollout (vLLM/SGLang)
  student trajectories
        ↓
┌───────────────────┐
│ Teacher LLM Server │ ← vLLM独立server → GPU上运行
│ (AsyncTeacher)     │ → 计算teacher log_probs
└───────────────────┘
        ↓ teacher_log_probs
  ★ KL divergence computation
        ↓
  distillation_loss → student update

★ ★ 关键:
  - Teacher可以是更大的模型(7B→0.5B distillation)
  - Teacher server独立 → 不共享training GPU → 不干扰
  - VLM distillation也支持(Qwen3-VL-2B → 4B teacher → Geo3K)
```

### OPD 性能结果

```
Student: Qwen2.5-0.5B, GSM8K:
  - Forward top-K KL + Qwen2.5-3B teacher → GSM8K acc稳步上升
  - Forward top-K KL + Qwen2.5-7B teacher → 更高最终acc
  - k3 KL estimator + Qwen2.5-7B teacher → reward-style → 也不错

VLM Distillation:
  - Student: Qwen3-VL-2B, Teacher: Qwen3-VL-4B, Geo3K → 有效
```

### ★ Fused top-K distillation kernel

```
PR #6511: fused top-K distillation kernel for OPD
  → chunked gather-logsumexp (#6593) → 长context避免OOM
  → overlap metrics (#6469) → 训练+metrics overlap
```

---

## 6. SGLang LoRA + PD Disaggregation

### 6.1 SGLang LoRA Support

> PR #5564 | dual-mode LoRA: merge + native adapter paths

```
★ ★ 双模式 LoRA (actor_rollout_ref.model.lora.merge):

1. merge=True: Merge LoRA into base weights → sync full HF-keyed params to SGLang
   → ★ 今天可用 → 等价全参 → vLLM/SGLang都支持
   → RTX 4090最优路径: merge → INT4 → 推理加速

2. merge=False (default): Native adapter → LoadLoRAAdapterFromTensorsReqInput
   → ★ infrastructure ready → Phase 2 wiring还在开发
   → 未来: 多tenant → 同batch不同adapter → 更灵活

★ ★ 配置示例:
actor_rollout_ref:
  model:
    lora:
      merge: true  # 或 false (native path)
```

### 6.2 SGLang PD Disaggregated Rollout

> PR #6117 | 1 prefill : N decode | per-rank role routing

```
★ ★ PD分离 → 不对称布局:

Prefill (1 GPU): 1P → 处理长prompt → compute-intensive
Decode (N GPU): 3D → 生成response → memory-bound

★ 关键设计: per-rank role routing → CUDA IPC handles stay on same physical GPU
  → 1P:3D, TP=1 → 避免跨GPU KV transfer开销

★ ★ vs vLLM PD分离:
  vLLM: KV connector(NIXL/RDMA) → 跨节点KV transfer → 网络开销
  SGLang: 1P:ND → same-node → CUDA IPC → 极低延迟

Qwen2.5-7B, GSM8K, n=8, response_length=2048, 1P:3D:
  → 比colocated更快 → decode GPU专注decode → 不被打断prefill

★ ★ ★ RTX 4090: 单节点 → PD分离有意义(prefill GPU只做prefill)
  → 但单GPU=24GB → 无法同时放actor+prefill+decode → 至少2GPU
  → PCIe → NCCL IPC慢 → 不如NVLink集群有意义
```

---

## 7. Other Important Features

### 7.1 TransferQueue Sync Trainer (PR #5401)

```
★ ★ ★ TransferQueue → controller只orchestrate metadata(KVBatchMeta)
  → 大tensor zero-copy通过TransferQueue → 消除controller bottleneck

性能(128×H100):
| Platform | Scale | Model | E2E Gain |
| H100 | 16×8 | Qwen3-VL-30B | ★ +49.2% |
| H100 | 2×8 | Qwen3-30B | +14.1% |
| NPU A3 | 2×16 | Qwen3-VL-30B | +20% |

★ ★ 关键: scale越大 → gain越高 → controller bottleneck越严重 → TQ越有效
```

### 7.2 VeOmni Engine

```
VeOmni = 华为Ascend团队贡献的engine backend → NPU专用
  - On-policy distillation support (#6072) → VeOmni-native critic (#6453)
  - MoE router replay R2/R3 (#6325) → load-balance monitoring (#6470)
  - Qwen3-VL-30B-MOE/Qwen3.5-122b-a10b/30b GRPO demos

★ ★ RTX 4090: VeOmni=NPU专用 → ✗ → 不适用
```

### 7.3 Megatron-FSDP Mode (PR #5423)

```
★ Megatron-FSDP = Megatron + FSDP2 混合 → 既用Megatron的TP/PP调度 → 又用FSDP的sharding
  → Dynamic context parallel (#5057) + CP for BSHD format (#5826)
  → Checkpoint save as HF PEFT format (#5575)
  → Qwen3.5 MTP SFT/RL (#5898)
  → NVFP4 QAT training (#5254) → SM90 only

★ RTX 4090: Megatron需要多GPU+NVLink → PCIe disaster → ✗
```

### 7.4 TRT-LLM Async RL (PR #5631, #5992)

```
TensorRT-LLM rollout → async RL支持 → 包括inter-node
  → inference极致优化 → but deployment复杂
  → ★ 只适合生产部署 → 不适合研究/实验
```

### 7.5 MooncakeStoreConnector Hard-Reset (PR #6373)

```
MooncakeStoreConnector → weight update时hard-reset → 清除KV cache
  → vs之前的soft-reset → 可能残留旧KV → 导致inconsistency
```

### 7.6 Agentic RL Training — uni-agent

```
★ verl-project/uni-agent → 新project → building, running, training general agents at scale
  - Coding, search, GUI agent RL training
  - Simpler function-based tool registration (#6189)
  - Per-sample tool environment routing (#5978)
  - Gemma4 tool parser (#6406)
  - Multi-step skip_rollout v2 (#5556)
```

### 7.7 Audio Data Support (PR #6276)

```
★ Audio data → 多模态扩展 → 不仅图片 → 还有音频
  → 未来: video, audio → 完整多模态RL
```

### 7.8 ReMax Support (PR #6340)

```
ReMax = 生成greedy response → 与sampled response对比 → reward as advantage
  → ★ TransferQueue trainer支持ReMax → combine greedy+sampled in one rollout request (#6308)
```

---

## 8. Breaking Changes

```
★ ★ ★ v0.8.0 Breaking Changes:

1. ★ ★ Deprecate legacy FSDP and Megatron workers → migrate to unified engine abstraction (#5604, #6067)
   → main_ppo.py deprecated → warning → favor main_ppo_sync.py (#6384)
   → 旧WorkerGroup path → 移除 → 必须用新engine path!

2. Deprecate verl/interactions (#6074) → 移除旧交互模块

3. Move LLMServerManager out of AgentLoopManager (#6129) → 独立化

4. Migrate Diffusion RL → verl-omni (#6200) → 独立repo
   → experimental/vla → verl-vla standalone (#6162)

5. Remove curriculum sampler + dynamic dataset + tool examples (#6302) → 清理旧代码

★ ★ main_ppo.py → main_ppo_sync.py 迁移:
  → 旧: main_ppo.py → 272行 → legacy code → deprecated warning
  → 新: main_ppo_sync.py → PPOTrainerSync → clean hooks
  → ★ main_ppo_v0.py (233行) → 保存旧代码 → 但不推荐使用

★ 配置迁移:
  trainer.v1.backend: "sync" | "colocate_async" | "separate_async"
  → 通过register_trainer选择 → 不需要改代码 → 只改config!
```

---

## 9. RTX 4090 实战分析

### ★ ★ RTX 4090 v0.8.0 Feature 影响矩阵

```
★ ★ RTX 4090 Feature Impact:

| Feature | RTX 4090适用? | 原因 |
|---------|-------------|------|
| Gemma4 GRPO | ★ ★ ✓ | SDPA可行(FA2/3 head_dim限制→用SDPA); 但E2B模型→INT4后可fit |
| Unify Trainer ABC | ★ ★ ✓ | PPOTrainerSync=单GPU最优; TQ overhead可忽略 |
| ROCm/HIP | ✗ | NVIDIA GPU → PlatformCUDA → ROCm不相关 |
| Detach Fix | ★ ★ ★ ★ ★ | ★ ★ ★ ★ ★ 关键! LoRA+GRPO+long→必然OOM → fix是必须的! |
| OPD | ✓(受限) | Teacher需要额外GPU → 单GPU内存太紧; 但merge=True→teacher=CPU |
| SGLang LoRA merge | ★ ★ ✓ | merge→INT4→4,791 tok/s最优路径 |
| SGLang PD disaggregation | ✗ | PCIe+单GPU → PD不可行 |
| TransferQueue | ✓(小规模) | 单GPU→controller不是瓶颈 → TQ增益小 |
| VeOmni | ✗ | NPU专用 |
| Megatron-FSDP | ✗ | 多GPU+NVLink→PCIe不可用 |
| TRT-LLM async | ✗ | 需要TRT-LLM编译→不适合研究 |
```

### ★ ★ ★ Detach Fix → RTX 4090的关键救星

```
★ ★ ★ ★ ★ v0.8.0的detach fix对RTX 4090是生死级重要!

问题: LoRA rank=32 + GRPO n=8 + use_dynamic_bsz(16k tokens) → ~400 micro-batches/update
  → 每个backward micro-batch → +0.27 GiB → 400 × 0.27 = 108 GiB → OOM!

修复后: 16.2 GiB constant → 24GB headroom → ✅!

★ ★ RTX 4090 GRPO LoRA训练配置(v0.8.0):
  1. ★ ★ 必须: verl >= v0.8.0 → detach fix → 否则OOM
  2. ★ PPOTrainerSync → 单GPU → 最简单
  3. LoRA rank=32 → merge=True → INT4 → 推理加速
  4. use_remove_padding=False → 有些模型需要(Gemma4)
  5. response_length ≤ 2048 → 长序列更安全(detach后内存稳定)

vs rLLM Tinker:
  rLLM → in-process → 无micro-batch累积问题 → graph自然释放
  verl → Ray actor → 需要detach → 但v0.8.0已修复 → ✅
```

### RTX 4090 实战配置示例

```bash
# ★ ★ RTX 4090: verl v0.8.0 GRPO LoRA (Qwen3-8B)

python -m verl.trainer.main_ppo_sync \
  trainer.v1.backend=sync \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.model.lora.merge=true \
  actor_rollout_ref.model.lora.rank=32 \
  algorithm.advantage_estimator=grpo \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.model.use_remove_padding=True \
  reward_model.enable=false \
  +algorithm.rollout_correction.bypass_mode=true \
  transfer_queue.enable=true
```

---

## 10. vs rLLM v0.3

### 统一Trainer ABC vs rLLM BackendProtocol

```
★ ★ verl PPOTrainer ABC vs rLLM BackendProtocol ABC:

| 维度 | verl v0.8.0 | rLLM v0.3 |
|------|-------------|-----------|
| 抽象层 | PPOTrainer ABC + register_trainer | BackendProtocol ABC + @rllm.rollout |
| 注册机制 | TRAINER_REGISTRY dict | BackendProtocol subclass |
| 子类数 | 3(sync/colocate_async/separate_async) | 3(verl/tinker/fireworks) |
| 共享逻辑 | step() 10步在base → hooks override | transform_fn在decorator → backend只管rollout |
| 数据传输 | TransferQueue(KVBatchMeta) | DataProto → tensor copy |
| 权重同步 | CheckpointEngineManager(5种backend) | zero-copy in-process(Tinker) |
| 多模态 | capability detection(Qwen/Gemma) | rLLM暂不支持 |
| Async | colocate_async(separate_async WIP) | asyncio gen_loop+train_loop |

★ ★ 核心差异:
  verl: 大规模分布式优先 → Ray+TransferQueue → 128 GPU
  rLLM: 单GPU研究优先 → in-process → Tinker → 1 GPU

★ ★ ★ RTX 4090最优路径:
  verl: PPOTrainerSync + LoRA + bypass → 需要detach fix(v0.8.0✅)
  rLLM: TinkerBackend + GRPO + LoRA → in-process → 不需要detach → 更简单

  → 但verl v0.8.0的detach fix让两条路径都可行!
```

### Gemma4 GRPO vs rLLM

```
★ ★ verl v0.8.0: Gemma4 GRPO → capability detection → 3处修改 → 7行代码
  → 自动切换Qwen/Gemma路径 → 不需要用户改代码

rLLM v0.3: 暂不支持Gemma4多模态GRPO
  → BackendProtocol只有LLM rollout → VLM需要额外开发
  → TinkerBackend: in-process → 更容易添加VLM支持
  → 但当前无capability detection → 硬编码

★ verl领先: 多模态RL → Qwen-VL + Gemma4 → production-ready
  rLLM落后: 多模态RL → 未实现 → 但Tinker架构更简单→更容易扩展
```

### ROCm vs rLLM

```
★ ★ ROCm → rLLM不涉及(只支持CUDA)
  → verl: PlatformROCm → AMD MI300可用 → 扩展hardware支持
  → rLLM: TinkerBackend → torch.cuda → 不区分NVIDIA/AMD

★ AMD用户 → verl v0.8.0更合适 → ROCm+AITER→优化kernel
```

### Detach Fix → rLLM不需要

```
★ ★ rLLM Tinker → in-process → 不跨Ray actor → 不累积micro-batch output_lst
  → autograd graph自然释放 → 不需要手动detach

verl → Ray actor → forward_backward_batch → output_lst累积 → 需要detach
  → v0.8.0修复后 → 问题消失 → 但设计上仍然是overhead

★ ★ Tinker的in-process优势:
  1. 不需要detach → graph自然释放
  2. 不需要TransferQueue → tensor直接传递
  3. 不需要CheckpointEngine → zero-copy weight sync
  4. 不需要Ray → 进程内 → 低延迟

★ ★ verl的分布式优势:
  1. 128 GPU → 49.2% gain → 大规模
  2. 多模态 → production-ready
  3. ROCm → hardware广泛
  4. OPD → distillation framework
```

---

## v0.8.0 精要总结

```
★ ★ ★ ★ ★ v0.8.0 5大核心变化:

1. ★ ★ ★ Detach Fix → LoRA+GRPO+long sequence → 从OOM到16.2 GiB stable
   → ★ ★ ★ ★ ★ RTX 4090生死级重要!

2. ★ ★ Unify Trainer ABC → PPOTrainer(sync/colocate_async/separate_async)
   → Legacy main_ppo.py deprecated → hooks pattern → 简洁可扩展

3. ★ Gemma4 Multimodal GRPO → capability detection → 3处7行修改
   → Qwen-VL/Gemma4自动切换 → production-ready VLM RL

4. ★ ROCm/HIP → PlatformROCm → AMD MI300 → AITER kernel
   → 扩展hardware → 但RTX 4090不相关

5. ★ OPD + SGLang LoRA/PD → distillation framework + rollout增强
   → 多teacher → PD分离 → 但RTX 4090受限

★★★ RTX 4090最优配置: verl v0.8.0 + PPOTrainerSync + LoRA merge + bypass + GRPO
  → detach fix让LoRA GRPO可行 → INT4推理 → 4,791 tok/s
  → vs rLLM Tinker: verl更复杂但更强大; rLLM更简单但功能少
```
