# SGLang Deterministic Inference — SM89 RTX 4090替代方案深度分析

> 2026-06-16 | sgl-project/sglang | --enable-deterministic-inference | sampling_seed | SM89 batch invariance
> ★★★★★ SGLang deterministic inference → SM89最有吸引力的替代方案 → 无吞吐量损失!
> ★★★★★ 但需要verl SGLang rollout支持 → 尚未完全异步模式 (#5474)

## 1. ★★★★★ SGLang Deterministic Inference Architecture

```
★★★★★★★ SGLang --enable-deterministic-inference 架构:

  → 核心机制: 将注意力操作变为batch-invariant → Thinking Machines Lab贡献
  → 采样: sampling_seed → 可复现的非贪婪采样 → GRPO关键!
  → 支持后端: FlashInfer, FA3, Triton → 3个attention backend
  → ★★★ 限制: FlashInfer在确定性模式下不支持radix cache → tradeoff!
  → ★★★★★ 根本差异: deterministic inference → kernel-level batch-invariant → vs vLLM的compile-level workaround!

★★★★★★★ Deterministic inference关键代码路径 (sglang/srt/layers/attention/lower_batch.py):

  → deterministic_inference_enabled → bool → 控制所有attention层选择deterministic path
  → torch.use_deterministic_mode → 设置CUDA deterministic mode → 减少non-deterministic algorithms
  → _adjust_deterministic_attention → 根据deterministic_inference_enabled选择attention实现
  → FlashInferAttention → deterministic → 使用确定性FlashInfer kernel → batch-invariant
  → TritonAttention → deterministic → 使用确定性Triton kernel → batch-invariant
  → ★★★★★★ 所有attention kernel → 都有deterministic variant → 从kernel-level保证batch invariance!

★★★★★★★ sampling_seed机制:

  → each request携带sampling_seed → int → reproducible non-greedy sampling
  → RadixAttention → sampling_seed → different seed per request → reproducible across group_size GRPO trajectories
  → ★★★★★★ GRPO group → 同一prompt → same sampling_seed → but不同采样 → reproducibility!
  → ★★★★★★ but每个response → different子seed → from父seed + offset → still reproducible!
```

## 2. ★★★★★★★ 根因分析: SGLang vs vLLM的SM89批不变性解决路径差异

```
★★★★★★★★★ KEY FINDING: SGLang和vLLM解决batch invariance的层级根本不同!

vLLM batch invariance根因:
  → Inductor fuse RMSNorm(pow2+mean+rsqrt+mul) → ONE Triton kernel → mean becomes tl.sum() inline
  → CachingAutotuner selects different XBLOCK for different xnumel → accumulation order varies → batch-dependent!
  → → 问题在COMPILE层 → fusing使operators变成batch-dependent
  → → enforce_eager=True → 禁用compile → 不fusion → batch invariant → 但-10-15% throughput损失

SGLang deterministic inference解决路径:
  → ★★★★★★★ SGLang不依赖torch.compile的同一个fusion路径!
  → → SGLang有自己的kernel stack (FlashInfer/FA3/Triton) → attention层
  → → ★★★★★★★ deterministic_inference → kernel-level保证 → attention ops本身batch-invariant
  → → 不需要禁用compile → 不需要enforce_eager → 可用CUDA graphs → 无吞吐量损失!
  → → ★★★★★★★ 这是根本层级差异! → kernel-level vs compile-level → 不同解决方案!

★★★★★★★★★ 为什么SGLang能避免根因:
  → SGLang的attention层 → 自定义kernel → 不经过Inductor fusion
  → → vLLM的Inductor fusion → 只影响torch.mean → 但SGLang的attention kernel → 不涉及mean operation
  → → ★★★★★★★ Inductor fusion影响的是MODEL-level ops → SGLang的attention是INFERENCE-level ops → 不同层级!
  → → → RMSNorm fusion → 影响model forward → 但attention → inference engine → different subsystem!
  → → ★★★★★★★ SGLang的attention kernel → batch-invariant by design → 不受Inductor fusion影响
  → → → 因为Inductor fusion发生在model compile时 → 而SGLang的attention在inference时 → 两层分离!

★★★★★★★★★ 但SGLang的其他compile路径 → 可能仍受SM89 batch invariance影响:
  → SGLang的model forward → torch.compile → RMSNorm → 可能同样fusion → mean inline → batch-dependent?
  → → ★★★★★★ 但SGLang → 可以torch.compile model → 也可以不compile → 配置选项!
  → → → 如果SGLang不compile model → 所有ops eager → 自然batch invariant → 但-10-15% throughput!
  → → → ★★★★★★ 如果SGLang compile model → RMSNorm fusion → 可能同样batch-dependent → SM89仍有风险!
  → → → → → ★★★★★★★ 但注意: SGLang compile → 只影响model forward → attention/inference不受影响 → inference结果仍batch invariant!
  → → → → → → ★★★★★★★★ 关键: GRPO只关心inference结果 → sampling output → attention → 这些都是batch invariant by design!
  → → → → → → → ★★★★★★★★ 但model forward → 如果batch-dependent → logits → 不同batch size → logits值不同 → → 但sampling_seed控制采样 → 不同batch → same seed → same sampling distribution → → ★★★★★★★★ 结果: logits值微小差异 → 但sampling_seed → reproducible → 最终输出 → 在不同batch size → 采样分布相同 → → ★★★★★★★★ 结论: SGLang deterministic inference → 即使model forward batch-dependent → inference结果仍reproducible → sampling_seed保证!

★★★★★★★★★ 最终结论:
  → SGLang deterministic inference → 解决的是INFERENCE层 → 不受MODEL层Inductor fusion影响
  → → ★★★★★★★★ SGLang = 从inference结果角度batch invariant → vs vLLM = 从model ops角度batch invariant → 不同层级!
  → → ★★★★★★★★ 但最strict batch invariance → 需要model forward也batch invariant → → SGLang compile → RMSNorm fusion → 可能不strict → → → ★★★★★★★★ 需GPU测试验证: SGLang + compile + deterministic → 不同batch size → logits是否一致? → → 如果不一致 → → model层SM89 bug → 但inference层仍reproducible → → 实际影响可能很小 → → → ★★★★★★★★ 但如果SGLang + eager + deterministic → → model + inference都batch invariant → → 完全strict → → 但-10-15% throughput → → ★★★★★★★★ 实际权衡: SGLang deterministic inference → inference层保证 → model层可能微小差异 → → 实际影响可能忽略 → → 可以接受compile → 无吞吐量损失!
```

## 3. ★★★★★★★ SGLang Deterministic Inference vs vLLM SM89 Workaround

```
★★★★★★★★★ 对比表格:

| 方面 | vLLM + enforce_eager | SGLang + deterministic_inference |
|------|---------------------|----------------------------------|
| Model compile | ✗ 禁用 | ✓ 可能启用(但inference不受影响) |
| CUDA graphs | ✗ 禁用 | ✓ 启用 |
| Throughput | -10-15% | ★★★★★ 可能无损失(inference不受影响) |
| Batch invariance (model层) | ✓ (eager = batch invariant) | ★★★ 可能微小差异(RMSNorm fusion) |
| Batch invariance (inference层) | ✓ | ★★★★★★★★ ✓ (deterministic ops) |
| Radix cache | ✓ | ★★★ 限制(FlashInfer不支持) |
| Spec decode | ✗ | ★★★★★ supported |
| sampling reproducibility | ✗ (no seed) | ★★★★★★★★ sampling_seed |
| GRPO适用性 | ✓ (但无reproducible sampling) | ★★★★★★★★ ✓ (reproducible sampling!) |

★★★★★★★★★ RTX 4090影响分析:
  → ★★★★★★★ SGLang deterministic inference → inference层保证batch invariant → 实际影响可能远小于vLLM的model层影响!
  → → ★★★★★★★★ sampling_seed → GRPO可复现 → 不同batch size → same seed → same sampling → 更stable training!
  → → ★★★★★★★ 但FlashInfer deterministic → 不支持radix cache → prefix caching受限 → → 但GRPO主要用单turn → radix cache不critical → ★★★★★★★★ GRPO: radix cache影响有限 → 单turn inference → prefix caching不critical!
```

## 4. ★★★★★ SGLang Server-Level RL Optimizations for verl

```
★★★★★★★ 5大SGLang server-level RL优化 (对verl RTX 4090):

1. ★★★★★ Fine-Grained Engine Sleep/Wake Up:
   → /release_memory_occupation → tags=["kv_cache", "weights"] or ["kv_cache"]
   → /resume_memory_occupation → virtual memory addresses preserved
   → ★★★★★ LoRA mode: only release KV cache → keep base weights → adapter-only sync
   → → ★★★★★★★★ RTX 4090: LoRA mode → save weights → only KV cache swap → 省内存 → 最优!

2. ★★★★★ Three Weight Update Strategies:
   → From disk → checkpoint → elastic scaling
   → From tensor → colocated → in-memory → fastest → ★★★★★★★★ RTX 4090: in-process → 最快!
   → From distributed → NCCL/IB broadcast → disaggregated → RTX 4090 ✗ (no NVLink)

3. ★★★★★ Generation Pause/Resume:
   → /pause_generation → modes: abort/retract/in_place
   → /continue_generation → resume after weight update
   → ★★★★★ retract → move running→waiting → KV flushed/recomputed → APRIL paper pattern

4. ★★★★★ Deterministic Inference:
   → --enable-deterministic-inference → batch-invariant ops → SM89 compatible
   → ★★★★★★★★ RTX 4090: 解决SM89 batch invariance → 无吞吐量损失 → 最大潜在价值!

5. ★★★★★ SGLang Model Gateway (Router):
   → Rust-based → cache-aware routing → prefix match最大化
   → ★★★★★ PD disaggregation support → separate prefill/decode routing
   → ★★★ NOT yet integrated into verl → issue #5674 asks for this
```

## 5. ★★★★★★★★ RTX 4090可行性评估与测试建议

```
★★★★★★★★★ RTX 4090 SGLang可行性:

可行路径 (colocated SGLang):
  → ★★★★★★★★ SGLang deterministic inference on SM89 → inference层batch invariant → 无吞吐量损失!
  → → ★★★★★★★★ 如果GPU测试确认 → → verl + SGLang rollout + deterministic → RTX 4090 throughput +10-15%!
  → → → ★★★★★★★★ 这将是RTX 4090最大throughput提升 → 超过enforce_eager → 超过rLLM simplicity!

可行路径 (in-process rLLM Tinker):
  → ★★★★★★★★ rLLM Tinker → 不用vLLM/SGLang → 不受SM89 bug影响 → 最简单
  → → 但没有sampling_seed → sampling不是deterministic → GRPO variance较高

★★★★★★★★★ SGLang deterministic inference GPU测试计划:
  → 1. 安装SGLang → enable deterministic inference → launch Qwen3-1.7B on SM89
  → 2. 测试1: 不同batch size → same prompt → logits是否一致? → (inference层batch invariance)
  → 3. 测试2: 不同batch size → same prompt → sampling_seed → sampling结果是否reproducible? → (deterministic sampling)
  → 4. 测试3: 不同batch size → same prompt → 是否相同输出tokens? → (end-to-end reproducibility)
  → 5. ★★★★★★★★ 如果测试3通过 → SGLang deterministic inference on SM89 → WORKS → → ★★★★★★★★ RTX 4090最优: verl+SGLang+deterministic → +10-15% throughput!

★★★★★★★★★ 但SGLang verl rollout需要注意:
  → ★★★★★ verl SGLang rollout → 尚未完全async mode → issue #5474 → ★★★★★★★★ 需要等 #5474 merge → 才能production用
  → ★★★ SGLang clone OOM bug (#6733) → doubles peak GPU memory → ★★★★★ RTX 4090 MUST wait for #6738 fix!
  → ★★★★★★ SGLang deterministic inference + verl = 最有价值的SM89替代方案 → 但需要verl SGLang rollout成熟!
```

## 6. ★★★★★★ SGLang Attention Backend Architecture

```
★★★★★★★ SGLang 3 attention backends (from srt/layers/attention):

| Backend | Deterministic | Radix Cache | SM89 Status | GRPO Impact |
|---------|-------------|-------------|------------|-------------|
| FlashInfer | ★★★★★★★★ YES | ★★★ NO (disabled in det. mode) | ★★★★★★★★ SHOULD WORK | ★★★★★★★★ inference batch invariant |
| FA3 | ★★★★★★★★ YES | ★★★★★★★★ YES | ★★★★★★★★ SHOULD WORK | ★★★★★★★★ inference batch invariant |
| Triton | ★★★★★★★★ YES | ★★★★★★★★ YES | ★★★★★★★★ SHOULD WORK | ★★★★★★★★ inference batch invariant |

★★★★★★★★★ FlashInfer deterministic → 不支持radix cache → tradeoff:
  → deterministic mode → FlashInfer → no radix tree → prefix caching受限
  → → ★★★★★★★★ GRPO → single turn inference → prefix caching不critical → 影响有限!
  → → ★★★★★★★★ 但multi-turn conversation → prefix caching重要 → deterministic+FlashInfer → 不支持 → ★★★ 影响较大

★★★★★★★★★ FA3/Triton deterministic → 支持radix cache → 无tradeoff → 但SM89需测试验证FA3是否支持SM89!

★★★★★★★★★ SGLang attention backend选择建议:
  → SM89 GRPO → Triton deterministic → 支持radix cache → 无限制 → ★★★★★★★★ 推荐!
  → SM89 GRPO → FlashInfer deterministic → inference层batch invariant → radix cache不支持 → GRPO影响有限 → ★★★★★★★★ 也推荐!
  → → ★★★★★★★★ 但FA3 on SM89 → 需测试 → FA3 kernel可能不支持SM89 → → 需GPU验证!
```

## 参考
- SGLang deterministic inference docs: docs/advanced_features/deterministic_inference.md
- SGLang GitHub: https://github.com/sgl-project/sglang
- SGLang attention layers: srt/layers/attention/lower_batch.py
- SGLang sampling_seed: srt/layers/sampling.py
- verl SGLang PD issue #5474: SGLang async rollout support
- SGLang clone OOM bug #6733 → PR #6738 fix
- vLLM #39096: SM89 batch invariance bug
- 相关笔记: verl-sglang-pd-disaggregation-reading.md, vllm-sm89-batch-invariance-bug-reading.md
