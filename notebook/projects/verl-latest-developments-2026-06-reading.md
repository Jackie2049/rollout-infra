# verl 最新发展 (2026年6月) 深度阅读

# verl 最新发展 (2026年6月) 深度阅读

> 2026-06-16 | 源码: volcengine/verl GitHub | 最新PR: Gemma4/#ZMQ fix/CUDA graph fix
> 核心: Gemma4多模态RL + ZMQ DP修复 + CUDA graph config兼容
> ★★★★ verl持续快速发展 → 多模态RL支持 → GRPO生态最完整

## 1. ★★★★★ 2026年6月关键合并PR

### 1.1 PR Gemma4: ★★★★★ 多模态RL支持 (2026-06-14 merged)

```
★★★★★ PR — feat: support Gemma4 multimodal models in RL (GRPO)

→ verl的VLM plumbing → 假设Qwen-VL形状的processor → Gemma4失败!
→ Gemma4使用标准1D RoPE (不是M-RoPE) → processor严格验证 → 失败

★★★★★ 修复内容:
  → verl/utils/tokenizer.py (hf_processor): add Gemma4Processor case
    → Gemma4 → standard 1D RoPE → no get_rope_index → 禁用validate check
  → verl/experimental/agent_loop/agent_loop.py (_compute_position_ids):
    → fallback → mask-based 1D positions → when processor没有get_rope_index
  → verl/utils/model.py (extract_multi_modal_inputs):
    → add mm_token_type_ids to _VARLEN_MULTI_MODAL_KEYS → pad-batching

★★★★★ RTX 4090影响:
  → Gemma4 → 2B → 单GPU可以运行 → GRPO训练可行!
  → ★★★★ verl now supports Qwen-VL + Gemma4 → 多模态RL生态更完整!
  → 但Gemma4-2B → head_dim=512 → 需要 SDPA (FlashAttention cap=256)
  → attn_implementation=sdpa → use_remove_padding=False → 配置注意!
```

### 1.2 PR ZMQ Handle: ★★★★ DP>1 colocated修复 (2026-06-08 merged)

```
★★★★ PR — fix: use data-parallel rank in vLLM ZMQ handles

→ 问题: colocated vLLM weight-sync → DP>1 → ZMQ socket rank collision!
  → sender: local_rank = rollout_rank % local_world_size
  → receiver: vLLM worker local_rank → 可能与sender不同 → socket冲突!

★★★★★ 修复:
  → reconstruct sender-side local rank:
    → data_parallel_rank_local * tensor_parallel_size + tensor_parallel_rank
  → 当DP未启用 → existing local-rank behavior preserved
  → ★★★ DP>1时 → 正确的ZMQ handle → 防止socket冲突

★★★★★ RTX 4090影响:
  → DP>1 → 多GPU → verl HYBRID → 可能遇到这个问题
  → 单GPU → 不受影响 → 但如果未来多GPU → 需要这个fix
  → ★★★ verl在积极修复多GPU部署问题
```

### 1.3 PR CUDA Graph Config: ★★★ vLLM 0.11.1兼容 (2026-06-15 merged)

```
★★★ PR — fix: Adjust cuda graph capture sizes config logic for vLLM >= 0.11.1

→ vLLM >= 0.11.1 → cuda_graph_sizes → 不再是CLI参数!
  → 改为 compilation_config.cudagraph_capture_sizes → 配置内参数
  → verl之前使用旧的CLI参数 → vLLM 0.23.0不兼容!

★★★★★ 修复:
  → 版本判断 → vLLM >= 0.11.1 → 用compilation_config参数
  → vLLM < 0.11.1 → 用旧的CLI参数
  → ★★★ 兼容vLLM新版本 → 防止配置错误 → 启动失败

★★★★★ RTX 4090影响:
  → verl + vLLM 0.23.0 → 需要这个fix → 否则配置错误 → 启动失败
  → ★★★★ 这意味着verl可以正确使用vLLM 0.23.0!
  → 但还需要注意v0.23.0的MRv2 interaction → 之前发现的ZERO handling问题
```

## 2. ★★★★ verl持续活跃开发

```
★★★★★ verl 2026年6月活跃开发:

模型支持:
  → Gemma4 多模态 RL (GRPO) → 新增!
  → SGLang LoRA support → 活跃PR
  → 多模态 → Qwen-VL + Gemma4 → 两个VLM框架!

部署修复:
  → ZMQ DP>1 fix → 多GPU部署修复
  → CUDA graph config → vLLM版本兼容
  → vLLM Docker image update → 0.20.2

★★★★★ verl v0.8+ 仍在快速发展:
  → 每周都有新PR merged
  → 多模态 → Gemma4 → 扩展RL训练范围
  → vLLM兼容 → 积极修复版本适配问题
  → ★★★ verl正在成为最完整的RL训练框架!
```

## 3. ★★★★★ verl vs rLLM — RTX 4090 GRPO训练对比更新

```
★★★★★★ 2026年6月 verl vs rLLM对比:

| 维度 | verl | rLLM |
|------|------|------|
| GRPO | ★★★★★ 完整 | ★★★★★ 完整 |
| 多GPU | ★★★★★ HYBRID/COLOCATED | ★★★ Tinker (单GPU) |
| 多模态 | ★★★★★ Qwen-VL + Gemma4 | ★★★ Geo3K Tinker VLM |
| 单GPU | ★★★★ bypass_mode | ★★★★★★ Tinker zero-copy |
| MRv2 | ✗✗✗ ZERO handling | ✗✗ 不用vLLM serving |
| spec decode | ★★★ 可以但需enforce_eager | ★★★ Tinker不用 |
| LoRA | ★★★★ FSDP+LoRA | ★★★★★ LoRA-32+bypass |
| async | ★★★★★ AsyncRolloutManager | ★★★★★ Fully Async Trainer |
| 社区 | ★★★★★ 大社区+活跃开发 | ★★★ 小社区+活跃开发 |
| 易用性 | ★★★ 需要多配置 | ★★★★★ rllm train → 一行命令 |

★★★★★★ RTX 4090推荐:
  → 单GPU GRPO → rLLM Tinker ★★★★★ → 最优 (in-process, zero-copy)
  → 多GPU GRPO → verl HYBRID ★★★★★ → 最优 (dedicated rollout + training)
  → 多模态 → verl ★★★★★ → Gemma4 + Qwen-VL → 更完整
  → 简单部署 → rLLM ★★★★★ → rllm train → 一行命令

★★★★★★ 新变化 (2026年6月):
  → verl + Gemma4 → 多模态RL支持更完整!
  → verl + vLLM 0.23兼容修复 → 可以用最新vLLM
  → rLLM + Fully Async Trainer → 异步训练 → 两者都有async
  → ★★★★★ 两个框架都在快速发展 → 选择取决于场景!
```

## 4. ★★★★★ verl MRv2交互更新

```
★★★★★ verl + vLLM v0.23 MRv2交互:

之前发现:
  → verl ZERO explicit MRv2 handling
  → vLLM 0.23 → MRv2 auto for Llama/Mistral/Qwen3 dense
  → INT4 → MRv1 (safe on RTX 4090)
  → recommend VLLM_USE_V2_MODEL_RUNNER=0 until tested

★★★★★ 新发现 (2026年6月):
  → verl CUDA graph config fix → 兼容vLLM >= 0.11.1
  → ★★★★ 这意味着verl CAN use vLLM 0.23 → 但MRv2仍需注意!
  → ZMQ handle fix → DP>1时 → 正确 → MRv2可能也需要DP适配
  → ★★★★★ verl需要在MRv2交互方面做更多工作:
    1. 检查MRv2是否影响verl的weight sync机制
    2. 检查MRv2是否影响verl的KV cache transfer
    3. 检查MRv2是否影响verl的sampling

★★★★★★ 建议:
  → 仍然推荐 VLLM_USE_V2_MODEL_RUNNER=0 for RTX 4090 GRPO
  → INT4 = MRv1 → safe → 不受MRv2影响
  → BF16 dense → MRv2 auto → 需要verl显式处理 → 否则可能有bug
  → ★★★★ 等verl正式支持MRv2再启用 → 当前建议禁用
```

## 5. ★★★★ verl SGLang集成

```
★★★★★ verl + SGLang → PD disaggregation:

SGLang LoRA support PR → active:
  → SGLang → LoRA adapter → verl可以与SGLang rollout
  → RadixAttention → prefix caching → similar to vLLM
  → ★★★ PD disaggregation → SGLang = decode-only → verl = prefill+train

★★★★★ 对比 verl + vLLM:
  → vLLM → HYBRID mode → rollout + training on same GPU
  → SGLang → PD disaggregation → separate prefill/decode GPUs
  → ★★★ PD disaggregation → 更高throughput → 但需要更多GPU

★★★★★ RTX 4090影响:
  → PD disaggregation → 需要至少2个GPU → RTX 4090单GPU → 不适用
  → ★★★ 单GPU → HYBRID mode → vLLM rollout → 最优
  → ★★★★ 多GPU → PD mode → SGLang decode → 可能更快
```

## 6. ★★★★★ Tinker Worker Primitives (#6717) — verl走向in-process!

```
★★★★★ PR #6717 — Tinker Worker Primitives → verl正在走rLLM-style in-process路线!

→ 新增 TinkerTrainingWorker 和 TinkerActorRolloutRefWorker
→ 分解训练步骤: optimizer_zero_grad() + forward_backward() + optimizer_step()
→ → 可独立调用! → 允许accumulated gradients across multiple forward_backward calls!
→ 默认API不变: worker.train_batch(data) → 保持兼容
→ ★★★★★ 这是verl走向in-process training的第一步!

★★★★★★★ 与rLLM Tinker对比:
  rLLM Tinker: 完整in-process → TinkerBackend → zero-copy → bypass default
  verl Tinker: worker primitives → split API → 但还没完整in-process trainer!
  → ★★★ verl正在缩小与rLLM的差距 → 但rLLM仍然领先!

★★★★★ RTX 4090影响:
  → verl Tinker primitives → 未来可能in-process → 不需要detach → 更高效
  → 但当前版本 → 仍需Ray → 跨进程 → 需要detach_metrics
  → ★★★★ 等下一版本完整Tinker trainer → RTX 4090 GRPO更简洁!
```

## 7. 关键洞察总结

```
★★★★★★ 6个关键洞察:

1. ★★★★★ Gemma4 multimodal RL → verl多模态RL生态最完整
   → Qwen-VL + Gemma4 → 两个VLM框架 → GRPO训练可行!
   → RTX 4090 → Gemma4-2B → 单GPU可行 → 但head_dim=512需要SDPA

2. ★★★★ ZMQ handle fix → DP>1 colocated → 正确
   → 多GPU部署修复 → verl在积极修复多GPU问题

3. ★★★★ CUDA graph config fix → vLLM >= 0.11.1兼容
   → verl CAN use vLLM 0.23 → 但MRv2仍需注意

4. ★★★★★ verl vs rLLM → 选择取决于场景
   → 单GPU → rLLM Tinker → 最优
   → 多GPU → verl HYBRID → 最优
   → 多模态 → verl → Gemma4 + Qwen-VL → 更完整

5. ★★★★★ MRv2交互 → verl仍需显式处理
   → ZERO explicit handling → 需要更多工作
   → INT4 → MRv1 → safe → BF16 → MRv2 → 需要verl处理

6. ★★★★ SGLang PD disaggregation → 需要多GPU
   → RTX 4090单GPU → HYBRID mode → vLLM
   → 多GPU → PD mode → SGLang decode → 可能更快
```

## 参考
- PR Gemma4: feat: support Gemma4 multimodal models in RL (merged 2026-06-14)
- PR ZMQ fix: fix: use data-parallel rank in vLLM ZMQ handles (merged 2026-06-08)
- PR CUDA graph fix: fix: Adjust cuda graph capture sizes config logic (merged 2026-06-15)
- vLLM-Ascend: Ascend/vllm-ascend GitHub
- verl SGLang LoRA support PR
- volcengine/verl GitHub
- 相关笔记: verl-mrv2-interaction-reading.md, verl-v0.23-mrv2-interaction-reading.md, distributed-rl-training-verl-architecture.md
