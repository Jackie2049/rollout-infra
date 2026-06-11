# vLLM V1 Speculative Decoding Architecture Source Reading — Proposer + Triton Rejection Sampler

> 2026-06-11 | vLLM V1投机解码全链路源码分析: SpecDecodeBaseProposer → EagleProposer → DraftModelSpeculator → 6 Triton Kernel Rejection Sampler
> 源码: vllm/v1/spec_decode/llm_base_proposer.py (1718行), vllm/v1/spec_decode/eagle.py (23行), vllm/v1/worker/gpu/spec_decode/speculator.py (225行), vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (671行)
> 关联: vllm-v1-kv-cache-architecture-source-reading.md, speculative-decoding-rtx4090-benchmark.md, inference-sampling-deep-dive.md

## 0. 核心定律: Spec Decode = Proposer生成 + Rejection Sampler验证 + KV Cache EAGLE组管理

```
投机解码全链路:
  Target Model Forward → hidden_states + next_token_ids
  → SpecDecodeBaseProposer.propose()
    → set_inputs_first_pass() → copy target outputs to draft buffers
    → build_attn_metadata → draft model forward → sample draft tokens
    → iterative loop: steps 1..N-1 → update positions → forward → sample
  → [batch_size, num_speculative_tokens] draft_token_ids 返回
  → Scheduler调度 → Target Model验证所有draft tokens
  → RejectionSampler (6 Triton kernels) → 拒绝/接受 → 重采样

关键架构决策:
  1. EAGLE/MTP共享target的embed_tokens+lm_head → 节省内存
  2. DeepSeek-V4 MTP: hc_mult扩展hidden_size + index sharing(set_skip_topk)
  3. Parallel Drafting (DFlash/Gemma4): 所有draft token一次forward → 不迭代!
  4. HybridKVCacheCoordinator: EAGLE draft层自成一组 → last-block drop机制
  5. 6 Triton kernel rejection sampler → 分块处理vocab → 贪心/概率双路径
```

## 1. SpecDecodeBaseProposer — 统一投机提议器基类

```
文件: vllm/v1/spec_decode/llm_base_proposer.py (1718行)

类层次:
  SpecDecodeBaseProposer (1718行, 所有投机方法的基类)
  → EagleProposer (pass_hidden_states_to_model=True)
  → DraftModelSpeculator (独立draft模型, 继承BaseSpeculator)

SpecDecodeBaseProposer核心属性:
  → speculative_config: 方法/步数/draft模型配置
  → pass_hidden_states_to_model: True=EAGLE(用target hidden states)/False=独立draft
  → parallel_drafting: True=DFlash(一次forward所有draft tokens)/False=迭代
  → hidden_size: draft模型hidden size; DeepSeek-V4 → hc_mult扩展
  → num_speculative_tokens: 投机步数
  → _share_mtp_indices: DeepSeek-V4 MTP index sharing优化
  → constant_draft_positions: Q-only attention → 固定position → 不递增

关键buffer:
  → input_ids: [max_num_tokens] → draft输入token ids
  → positions/mrope_positions: → draft位置编码
  → hidden_states: [max_num_tokens, hidden_size] → target hidden states传入
  → is_rejected_token_mask: [max_num_tokens] → 标记被拒绝的token
  → is_masked_token_mask: [max_num_tokens] → 标记masked padding tokens
  → cudagraph_dispatcher: PIECEWISE cudagraph管理

DeepSeek-V4 MTP特殊处理:
  → hidden_size *= hc_mult → target的pre-hc_head residual shape是(T, hc_mult * hidden_size)
  → compress_ratios + hc_mult检测 → 自动扩展hidden_states buffer
  → _share_mtp_indices → step 0计算sparse topk indices → steps 1+通过set_skip_topk(True)复用

Parallel Drafting:
  → DFlash/Gemma4 → parallel_drafting=True
  → extra_slots_per_request = num_speculative_tokens (不是1!)
  → 所有draft token在single forward pass生成 → 无迭代循环
  → mask_token_id → 用于masked positions → draft model看到mask而非实际token
```

## 2. propose() 方法 — 投机提议全流程

```
propose()签名:
  target_token_ids: [num_tokens] → target模型所有token ids
  target_positions: [num_tokens] or [3, num_tokens] → 位置编码
  target_hidden_states: [num_tokens, hidden_size] → target模型hidden states
  next_token_ids: [batch_size] → target模型sampled next tokens
  common_attn_metadata → attention元数据(block_table, seq_lens等)
  sampling_metadata → sampling参数(temperature, top_k等)
  num_rejected_tokens_gpu → 上一步被拒绝的token数(用于调整seq_lens)

propose()流程:

Phase 1: EAGLE3/DFlash hidden states合并
  → model.combine_hidden_states(target_hidden_states) → 多hidden state融合

Phase 2: First Pass (set_inputs_first_pass)
  → EAGLE路径(无extra slots):
    input_ids[:num_tokens-1] = target_token_ids[1:]  # shift by 1
    input_ids[token_indices_to_sample] = next_token_ids  # 替换最后token
    hidden_states[:num_tokens] = target_hidden_states
    positions[:num_tokens] = target_positions

  → Draft Model路径(有extra slots):
    copy_and_expand_eagle_inputs_kernel → Triton kernel:
    每个request: copy target tokens → 插入next_token_ids → padding slots
    同时设置is_rejected_token_mask + is_masked_token_mask

Phase 3: Draft Model Forward
  → build_per_group_and_layer_attn_metadata → 构建attention元数据
  → _determine_batch_execution_and_padding → cudagraph padding决策
  → model(**model_kwargs) → draft模型forward
  → Step 0后: if _share_mtp_indices → set_skip_topk(True) → MTP index复用

Phase 4: Sample Draft Tokens
  → _sample_draft_tokens(hidden_states, sampling_metadata)
  → 贪心: compute_logits().argmax() / local_argmax_reduction
  → 概率: compute_probs_and_sample_next_token → 保存draft_probs用于rejection

Phase 5: 如果parallel_drafting或num_speculative_tokens==1 → 直接返回
  → 否则进入迭代循环:

Phase 6: Iterative Loop (steps 1..N-1)
  for token_index in range(num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()  # 上一步的draft token
    positions = _update_positions_dependent_metadata()  # position+1
    # 更新seq_lens: +=1 (新增draft token)
    # 重新构建attn_metadata → draft model forward → sample
    draft_token_ids = _sample_draft_tokens(...)
    draft_token_ids_list.append(draft_token_ids)

  → 返回: [batch_size, num_speculative_tokens] → 每个request的draft token序列
```

## 3. EagleProposer — 最简EAGLE实现

```
文件: vllm/v1/spec_decode/eagle.py (23行)

class EagleProposer(SpecDecodeBaseProposer):
  → __init__: pass_hidden_states_to_model=True → 唯一区别!
  → 其他所有逻辑继承SpecDecodeBaseProposer

为什么pass_hidden_states_to_model=True?
  → EAGLE架构: draft head看到target的hidden_states → 输入是(token_id, hidden_state)对
  → vs 独立draft模型: 只看到token_id → 需要自己的embed_tokens + 全forward

EAGLE3特殊:
  → combine_hidden_states → 融合多层hidden states → 更强的draft质量
  → eagle3_use_aux_hidden_state → 使用aux hidden states
  → 3种EAGLE模型类: Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM
```

## 4. DraftModelSpeculator — 独立Draft模型提议器

```
文件: vllm/v1/worker/gpu/spec_decode/speculator.py (225行)

class BaseSpeculator(ABC):
  → init_cudagraph_manager() → capture() → propose() → 抽象方法

class DraftModelSpeculator(BaseSpeculator):
  → 独立draft模型 → 自己的input_buffers, block_tables, kv_cache_config
  → draft_logits: [max_num_reqs, num_spec_steps, vocab_size] → 概率draft采样
  → draft_sample_method == "probabilistic" → 存储draft logits → rejection sampler需要

关键buffer:
  → idx_mapping: [max_num_reqs] → request到state的映射
  → temperature: [max_num_reqs] → 采样温度
  → seeds: [max_num_reqs] → 采样种子
  → draft_tokens: [max_num_reqs, num_speculative_steps] → draft token ids
  → draft_logits: [max_num_reqs, num_spec_steps, vocab_size] → 概率draft

propose()签名(与SpecDecodeBaseProposer不同!):
  → input_batch: InputBatch → 完整的输入batch
  → last_hidden_states: [num_tokens, hidden_size] → target hidden states
  → aux_hidden_states: list[Tensor] → 辅助hidden states(EAGLE3用)
  → num_sampled/num_rejected/last_sampled → rejection状态
  → next_prefill_tokens → prefill tokens
  → temperature/seeds → 采样参数
```

## 5. 6 Triton Kernel Rejection Sampler — 拒绝采样引擎

```
文件: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (671行)

数学基础:
  拒绝采样 = 确保输出等于target分布!
  → 对每个draft token x:
    → 贪心: target_argmax == draft_token → 接受 / 否则拒绝 → 替换为target_argmax
    → 概率: log_p(x) > log(u) + log_q(x) → 接受
      → p(x) > u × q(x) → 概率形式
      → u是均匀随机数 → 确保p覆盖q → 输出=精确p分布

  重采样: 如果拒绝 → 从residual分布采样
    → residual = max(p(x) - q(x), 0) → target分布去掉draft分布
    → HAS_DRAFT_LOGITS: ratio = exp(log_q - log_p) → residual = log(max(1-ratio, 0)) + log_p
    → One-hot draft: residual = target - rejected_draft_token(零化该token)

6个Triton Kernel:

Kernel 1: _compute_block_stats_kernel
  → 输入: [num_logits, V] target_logits + [max_num_reqs, num_spec_steps, V] draft_logits
  → 输出: per-block max/argmax/sumexp → 用于后续global LSE计算
  → 贪心路径: 只需要target_local_argmax + target_local_max
  → 概率路径: target_local_max + target_local_sumexp + draft stats(如果HAS_DRAFT_LOGITS)
  → Bonus token(draft_step_idx >= num_spec_steps): 跳过 → 不需要stats

Kernel 2: _rejection_kernel
  → 核心拒绝逻辑! → per-request → 迭代检查每个draft token
  → 贪心: target_argmax == draft_sampled → 接受/拒绝 → 拒绝则用target_argmax替换
  → 概率: log_p(x) > log(u) + log_q(x) → u由tl_rand64生成 → 确保精确分布
  → SYNTHETIC_MODE: u < rate → 用于benchmark/simulation
  → 记录: rejected_steps(接受步数), target/draft_rejected_logsumexp(LSE)

Kernel 3: _resample_kernel
  → 重采样被拒绝的位置 → 从residual分布采样
  → Bonus token: residual = target_logits → 直接采样(无拒绝→加一个bonus token!)
  → HAS_DRAFT_LOGITS: ratio = exp(draft_log_prob - target_log_prob)
    → residual = log(max(1-ratio, 0)) + target_log_prob → target去掉draft
  → One-hot: residual = target但rejected_draft_token位置=-inf
  → 采样: gumbel_block_argmax → Gumbel-max trick → 高效采样

Kernel 4: _insert_resampled_kernel
  → 将重采样的token插入sampled tensor
  → 贪心+非bonus: 直接跳过 → target_argmax已在sampled中
  → 概率/bonus: 找到resampled_local_argmax(跨block argmax) → 插入

Kernel 5: _compute_global_lse (inline function)
  → 计算log(sum(exp(logits))) → 跨vocab blocks的global log-sum-exp
  → global_max = max(local_maxes) → global_lse = global_max + log(sum(sumexps * exp(maxes-global_max)))
  → 数值稳定: max-normalization → 避免溢出

Kernel 6: _flatten_sampled_kernel (在RejectionSampler类中)
  → 将[num_reqs, num_spec_steps+1]的sampled tensor扁平化
  → 用于后续处理(scheduler需要flat token list)

常量配置:
  → VOCAB_BLOCK_SIZE = 8192 → vocab分8192大小blocks → 适配GPU并行
  → RESAMPLE_BLOCK_SIZE = 1024 → 重采样用更小block → 减少waste
  → FP64 Gumbel支持 → use_fp64_gumbel → 更精确的采样

关键设计决策:
  → 分块处理vocab → 适配任意vocab大小(32K-151K) → 不需要固定V
  → 贪心路径简化 → 只需argmax → 不需要LSE → 更快!
  → HAS_DRAFT_LOGITS constexpr → 编译时分支 → 零overhead
  → SYNTHETIC_MODE constexpr → 用于benchmark → 不影响生产代码路径
  → Gumbel-max trick → 采样=argmax(logits + Gumbel noise) → 高效!
```

## 6. Weight Sharing — EAGLE/MTP共享Target权重

```
_maybe_share_embeddings():
  → EAGLE模型: has_own_embed_tokens=False → 共享target的embed_tokens
  → EAGLE模型: embed_tokens与target相同 → torch.equal() → 共享!
  → EAGLE模型: embed_tokens不同 → 保持独立 → 更灵活
  → MTP模型: 总是共享 → MTP没有自己的embed_tokens → 必须!

  共享方式: del self.model.model.embed_tokens → self.model.model.embed_tokens = target_embed_tokens
  → Python引用替换 → 同一个GPU内存 → 零额外内存开销!

_maybe_share_lm_head():
  → EAGLE: has_own_lm_head=False → 共享target的lm_head
  → EAGLE: lm_head相同 → torch.equal() → 共享!
  → EAGLE: lm_head不同 → 保持独立

  DeepSeek-V4 MTP特殊:
    → lm_head = shared_head.head → 共享DeepSeek-V4的shared_head
    → topk_indices_buffer sharing → step 0计算 → steps 1+复用

内存节省计算(7B LLaMA):
  → embed_tokens: 32000 × 4096 × 2bytes = 256MB → 共享省256MB!
  → lm_head: 4096 × 32000 × 2bytes = 256MB → 共享省256MB!
  → 总省: 512MB → RTX 4090 24GB → 2.1% → 值得但不是决定性

  → 但! EAGLE draft head很小(~1-4%参数) → 加上共享 → 总draft开销<5%
  → → INT4+INT8KV+EAGLE → 内存开销几乎为零 → RTX 4090最优组合!
```

## 7. KV Cache Integration — EAGLE Draft层自成KV组

```
HybridKVCacheCoordinator中的EAGLE处理:

1. eagle_group_ids:
  → SpecDecodeBaseProposer的draft attention layers → 自成KV cache group
  → 与target的FullAttention组分开 → 不同block_table管理

2. is_eagle_group检查:
  → HybridKVCacheCoordinator.find_longest_cache_hit()中
  → 对EAGLE group: 执行last-block drop
  → 原因: EAGLE最后一步的draft token KV → 未被target验证 → 不应缓存!
  → → 减去eagle_last_block_num → 更保守 → 避免无效prefix hit

3. Draft Attention Backend初始化:
  → SpecDecodeBaseProposer: draft_attn_layer_names = all_attn_layers - target_attn_layer_names
  → → 找到draft模型新增的attention layers → 用自己的KV cache spec
  → initialize_attn_backend(kv_cache_config, active_layer_names=draft_attn_layer_names)
  → → 为draft layers单独初始化attention backend → 与target分开管理

4. Slot Mapping:
  → _get_slot_mapping(): draft layers共享同一个slot_mapping buffer
  → → EAGLE和target使用相同的KV cache物理位置 → 但逻辑上分开管理
  → → PADDING_SLOT_ID用于无效位置 → draft token padding

5. RTX 4090影响:
  → EAGLE draft layers KV极小(~1-4%参数) → KV/tok极少 → 内存影响negligible
  → → EAGLE overhead主要在compute(1次额外forward) → 不是内存
  → → INT4+INT8KV+EAGLE → KV额外开销<2% → 完全可行!
```

## 8. RTX 4090 Implications — 投机解码的硬件约束

```
RTX 4090投机解码架构影响:

1. EAGLE是RTX 4090最优proposer:
  → 接受率α≈0.85 → depth=5 → 4.2x加速 → vLLM原生支持
  → Weight sharing → 内存开销<5% → INT4+INT8KV下几乎free
  → 单GPU → 无多GPU通信 → EAGLE overhead纯compute

2. N-gram是RTX 4090最简proposer:
  → 零compute overhead → 只查token频率表 → 最快proposer
  → α≈0.4 → depth=3 → 2.14x → 不如EAGLE但零开销
  → vLLM Medusa proposer → 多头并行 → 更灵活

3. Draft Model不建议:
  → 独立draft模型需要额外7B→1.4B → 内存占用→挤占target空间
  → 同大小draft = OOM灾难 → 接受率α≈0.95但compute 0.95x慢→负优化!
  → 更小draft → α下降 → 性价比不如EAGLE

4. Triton Rejection Sampler性能:
  → 分块处理 → VOCAB_BLOCK_SIZE=8192 → 32K vocab只需4 blocks → 高效
  → 贪心路径 → 只需argmax → 极快 → 大多数生产场景用贪心
  → 概率路径 → 需LSE+draft_logits → 更多compute → 仅probabilistic sampling时
  → 估计: rejection sampler overhead <0.5ms → << attention 0.22ms → 不是瓶颈

5. 生产最优组合:
  → 7B INT4 AWQ + INT8 KV + GQA-8 + FlashInfer + Eagle d=5
  → → B=52 → 9,088 tok/s → 4.2x over non-spec → RTX 4090极限性能!

6. DeepSeek-V4 MTP在RTX 4090:
  → hc_mult → hidden_size扩展 → 需要更多compute per draft step
  → 但! index sharing(set_skip_topk) → step 0计算sparse indices → steps 1+复用
  → → MTP compute overhead <4% → 可行 → 但EAGLE更适合RTX 4090(接受率更高)
```

## 参考文献

```
1. vLLM V1 Speculative Decoding源码:
   - vllm/v1/spec_decode/llm_base_proposer.py (1718行) — SpecDecodeBaseProposer
   - vllm/v1/spec_decode/eagle.py (23行) — EagleProposer
   - vllm/v1/worker/gpu/spec_decode/speculator.py (225行) — DraftModelSpeculator
   - vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (671行) — 6 Triton kernels

2. EAGLE: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature-Level Compression", 2024
3. DeepSeek-V3/V4 Technical Report — MTP + hc_mult + index sharing
4. Chen et al., "Breaking the Sequential Dependency of LLM Inference", 2024 (Medusa)
5. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023

我们的笔记:
- vllm-v1-kv-cache-architecture-source-reading.md — KV Cache架构
- speculative-decoding-rtx4090-benchmark.md — RTX 4090投机解码实测
- inference-sampling-deep-dive.md — 采样理论+拒绝采样数学
- inference-calculator-4090.md — 推理计算器(含投机解码)