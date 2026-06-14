# vLLM V1 Speculative Decoding 源码级架构分析

> 2026-06-15 | 源码: vllm/v1/spec_decode/ + vllm/model_executor/models/ + vllm/v1/worker/gpu/spec_decode/
> 核心: 8种方法(EAGLE/MTP/Medusa/DFlash/Ngram/Draft/Stuffix/Custom) → EAGLE=hidden state feed → MTP=shared layer → Medusa=multi-head MLP → Rejection=3 Triton kernels → RTX 4090最优=EAGLE+INT4→9,088 tok/s

## 1. Speculative Decoding方法总览

```
vllm/config/speculative.py (59-68): 8种方法

| Method | Proposer | Architecture | 特点 |
|--------|----------|-------------|------|
| eagle | EagleProposer | Draft model fed hidden states | ★ 最高acceptance |
| eagle3 | EagleProposer | Draft fed multi-layer hidden | EAGLE+v3 multi-layer |
| mtp (deepseek/step3.5等) | EagleProposer/Step3p5 | Single transformer layer+shared lm_head | ★ 最轻量 |
| dflash | DFlashProposer | Parallel drafting+cross-attention | 一次forward所有draft |
| medusa | MedusaProposer | Multi-head MLP on target hidden | ★ 最小内存 |
| draft_model | DraftModelProposer | Independent smaller LM | 最大额外内存 |
| ngram | NgramProposer | Pattern matching | 无额外模型 |
| suffix/custom | CustomProposer | User-defined | 灵活 |

★ EAGLE定义特性: pass_hidden_states_to_model=True → 传入target hidden states!
★ Draft Model定义特性: pass_hidden_states_to_model=False → 独立draft LM
```

## 2. EAGLE — Hidden State Feed Draft Model

```
★ ★ 核心洞察: EAGLE把target model的hidden states作为draft model输入
  → 不是用token IDs → 是用内部表示 → 预测质量远高于独立draft model!
  → fc层: torch.cat((input_embeds, hidden_states), dim=-1) → project [2*hidden_size]→hidden_size

EAGLE v1模型架构 (llama_eagle.py:54-116):
  forward(input_ids, positions, hidden_states):
    input_embeds = embed_tokens(input_ids)
    hidden_states = fc(torch.cat((input_embeds, hidden_states), dim=-1))
    → ★ 第1层decoder: qkv_input_size = 2*hidden_size (接收target+embed)
    → 后续层: normal hidden_size
    for layer in self.layers:
      hidden_states, residual = layer(positions, hidden_states, residual)
    return (last_hidden, all_hidden) → ★ all_hidden用于下一步draft!

EAGLE v3 (llama_eagle3.py):
  → ★ 扩展: 使用target model中间层的hidden states → 不只最后一层!
  → combine_hidden_states(): concatenates multiple layer outputs → project
  → eagle_aux_hidden_state_layer_ids → 配置哪些target层提取

★ ★ EagleSpeculator (speculator.py:40): GPU-side autoregressive draft
  Prefill phase (264):
    → target hidden states → EAGLE forward → _sample_draft() → 第1个draft token

  Multi-step decode (299):
    → 每步: generate_draft() → forward + sample + update inputs
    → ★ _update_eagle_draft_inputs_kernel (810-903): Triton kernel
      → fused: copy hidden states + increment positions → 一次kernel → 极快!

  _sample_draft() (239)两种模式:
    → Greedy: logits.argmax → one-hot draft → rejection简单
    → Probabilistic: gumbel_sample() → full draft logits → rejection精确

★ ★ Weight Sharing (1274-1392):
  → embed_tokens: EAGLE shares target when has_own_embed_tokens=False
  → lm_head: EAGLE shares target when has_own_lm_head=False
  → MTP always shares both → 最省内存!

★ Step3.5MTPProposer (step3p5.py:24):
  → extends EagleProposer
  → Per-layer draft-step selection → spec_step_idx → compute_logits()
  → Multi-KV-cache-group support → draft layers可跨多个KV cache组
```

## 3. Medusa — Multi-Head MLP Speculation

```
★ ★ 核心洞察: Medusa不需要autoregressive loop → 所有head并行预测!

Medusa模型架构 (medusa.py:41):
  self.blocks = nn.ModuleList of ResidualBlock → 每head一个
  self.lm_heads = nn.ModuleList of ParallelLMHead → 每head一个

ResidualBlock (19-38):
  → nn.Linear(hidden_size, hidden_size) × N + SiLU + residual
  → x = x + act(layer(x)) → 简单MLP!

Forward (105-106):
  → [block(hidden_states) for block in self.blocks] → 每head独立预测!

★ ★ vs EAGLE关键差异:
  → 无autoregressive loop → 单次hidden states → 所有head并行 → 极快!
  → 无独立KV cache → Medusa heads没有attention层 → 极小内存!
  → 无hidden state传播 → 不sequential forward → 但acceptance更低
  → Optional truncated vocab (97-103): token_map减少draft vocab → 省compute!

MedusaProposer (medusa.py:18):
  propose(target_hidden_states, sampling_metadata):
    blocks = model(target_hidden_states) → 所有head并行
    logits = model.compute_logits(blocks)
    draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
    → ★ 极简 → 单次forward → 无循环!

★ 当前vLLM Medusa用top-1 verification → 不是tree-based → 简单但acceptance不如tree!
  → draft tokens = [head0_token, head1_token, ...] → linear sequence → sequential verify
```

## 4. MTP — Multi-Token Prediction (Layer-Level)

```
★ ★ 核心洞察: MTP是EAGLE简化版 → 1层transformer → shared embed+lm_head → 极轻量!

MTPModelTypes (speculative.py:34-53): 10+模型类型!
  → deepseek_mtp / mimo_mtp / mimo_v2_mtp / glm4_moe_mtp / ernie_mtp
  → nemotron_h_mtp / qwen3_next_mtp / qwen3_5_mtp / step3p5_mtp / gemma4_mtp
  → ★ 所有MTP统一为method="mtp"

MTP vs EAGLE:
  → MTP: 1 transformer layer → shared embed/lm_head → minimal overhead
  → EAGLE: 1-2 transformer layers → fc projection → hidden state feed
  → MTP model_returns_tuple=False → last_hidden_states == hidden_states → 无separate propagation

★ ★ DFlash (dflash.py): Parallel Drafting → 所有draft tokens一次forward!
  → parallel_drafting = True → 自动设置
  → ★ Cross-attention: context K/V from target hidden → Q from query embeddings
  → precompute_and_store_context_kv() → 预插入context KVs → cache
  → use_non_causal=True → 非因果attention → parallel可以!
  → num_query_per_req = 1 + num_speculative_tokens → 一次生成所有!

★ DFlash vs EAGLE:
  → EAGLE: sequential draft → 1 token per step → 多步
  → DFlash: parallel draft → 所有token一次 → 但需要cross-attention
  → DFlash需要NVLink → PCIe RTX 4090不适合 → cross-attention效率低
```

## 5. Rejection Sampling — 3 Triton Kernels

```
★ ★ GPU Model Runner RejectionSampler (rejection_sampler_utils.py):

3个Triton kernel实现rejection sampling:

1. _compute_block_stats_kernel (48-153):
   → 每block: target argmax + max + sumexp
   → 如果HAS_DRAFT_LOGITS=True: 也计算draft logit stats

2. ★ ★ _rejection_kernel (157-303): 核心rejection!
   → Greedy (T=0): accept IFF target_argmax == draft_sampled → 一致才accept!
   → Stochastic: accept IFF target_log_prob > log(u) + draft_log_prob
     → ★ ★ 标准rejection sampling: p(x) >= u*q(x) → 数学证明正确!
   → Synthetic: accept with decaying probability u < rate → conditional rates
   → HAS_DRAFT_LOGITS=False (one-hot draft): draft_log_prob = 0 → p(x) >= u

3. _resample_kernel (306-428): rejection后resample
   → ★ Residual distribution: max(p(x) - q(x), 0)
   → HAS_DRAFT_LOGITS: residual = p + log(1 - exp(q - p)) → 精确residual
   → One-hot draft: target distribution with rejected token zeroed out
   → gumbel_block_argmax → sampling from residual

4. _insert_resampled_kernel (434-490): 插入resampled token

★ ★ V1 Sample RejectionSampler (sample/rejection_sampler.py):

2条路径:

Greedy path (rejection_greedy_sample_kernel, 708-757):
  → 每draft position: accept IFF draft_token == target_argmax
  → Rejection → replace with target_argmax → stop
  → All accepted → append bonus token → ★ bonus!

Stochastic path (rejection_random_sample_kernel, 762-826):
  → Core: target_prob / draft_prob >= uniform_prob
  → NO_DRAFT_PROBS (ngram/medusa): draft_prob = 1 → accept IFF target_prob >= u
  → Rejection → sample_recovered_tokens_kernel (853-921)
    → recovered = argmax of max(p-q, 0) * inv_q → exponential distribution

★ ★ Acceptance Criteria总结:

| Sampling Mode | Acceptance | Recovery Distribution |
|--------------|-----------|---------------------|
| Greedy (T=0) | target_argmax==draft | target_argmax directly |
| Stochastic+draft_logits | p/q >= u | max(p-q, 0) × exp |
| Stochastic+one-hot | p >= u | p with draft zeroed |
| Synthetic | u < conditional_rate | standard target sampling |
```

## 6. Executor Coordination — Data Flow

```
GPU Model Runner协调 (gpu_model_runner.py):

Drafter initialization (541-614):
  → method → 选proposer class → 创建drafter
  → RejectionSampler creation (614) → sampler + speculative_config + device

★ ★ Async spec decode (627):
  → use_async_spec_decode → GPU-side drafting → 无CPU sync → 极快!
  → Deferred corrections (1457-1487): rejection corrections deferred → 不阻塞主路径

SpecDecodeMetadata (2133-2166):
  → logits_indices + bonus_logits_indices → scheduling spec decode tokens

★ ★ Data Flow:

Target Model Forward → Hidden States + Logits
  |
  Hidden States → Drafter.propose() → Draft Tokens (autoregressive loop for EAGLE)
  |
  Target Logits + Draft Tokens → RejectionSampler → Accepted Tokens + Bonus Token

EAGLE/MTP flow:
  1. Target forward → logits AND hidden_states
  2. Hidden → EagleProposer.propose() → autoregressive draft loop → draft tokens
  3. Draft tokens + target logits → RejectionSampler → accepted sequence

Medusa flow:
  1. Target forward → logits AND final hidden state
  2. Single hidden → MedusaProposer.propose() → all heads parallel → draft tokens
  3. Draft tokens + target logits → RejectionSampler → accepted sequence

★ ★ ★ Triton-optimized全路径:
  → _update_eagle_draft_inputs_kernel → copy hidden + increment positions → fused GPU
  → _compute_block_stats + _rejection + _resample → 3 Triton kernels → rejection sampling
  → 无CPU-GPU sync → 全GPU → 极快!
```

## 7. RTX 4090 Speculative Decoding选择

```
★ ★ Memory Constraints (7B BF16 ~14GB target):

| Method | Extra Memory | Description |
|--------|-------------|-------------|
| EAGLE | ~1-2GB | 1-2 transformer layers + fc + shared embed/lm_head |
| EAGLE v3 | ~1-2GB+aux | Same + auxiliary hidden state buffers |
| MTP | ~0.5-1GB | 1 transformer layer + shared embed/lm_head |
| Medusa | ~0.2-0.5GB | Only MLP heads, no attention |
| Draft Model | ~7-14GB | Entire separate LM → ❌ RTX 4090不可能! |
| DFlash | ~1-2GB | Small draft + parallel buffers |

★ ★ Compute Overhead (decode memory-bound):

| Method | Per-Step Compute | Overhead |
|--------|----------------|---------|
| EAGLE | 1-2 small layers | ~10-20% per step |
| MTP | 1 transformer layer | ~5-10% |
| Medusa | MLP heads | ~1-3% negligible! |
| Draft Model | Full LM forward | ~50-100% → ❌ |
| DFlash | 1 forward all tokens | Similar to EAGLE total |

★ ★ Acceptance Rate (7B models):
  EAGLE: 70-80% → avg acceptance length ~3-4 → ★ 最高!
  MTP: 60-75% → 依赖模型架构
  Medusa: 50-70% per head → independent predictions → 较低
  Ngram: 30-50% → pattern matching → 最低

★ ★ ★ RTX 4090推荐: EAGLE + INT4量化 → 9,088 tok/s!
  1. EAGLE shares embed_tokens+lm_head → minimal extra memory (~1GB)
  2. Hidden state feed → higher acceptance → 70-80% → avg 3-4 tokens accepted
  3. INT4量化 → 7B ~3.5GB → 剩余空间足够EAGLE draft (~1GB)
  4. Triton-optimized rejection → 3 kernels → 无CPU-GPU sync → 极快
  5. From memory notes: INT4+INT8KV+GQA-8+FlashInfer = 4,791 tok/s → EAGLE → 9,088 tok/s!

★ ★ Second best: MTP (DeepSeek-V3/V4/Step3.5/Qwen3.5)
  → 模型自带MTP layers → 不需要额外训练
  → Shared所有权重 → minimal overhead (~0.5-1GB)
  → 但acceptance略低于EAGLE → single layer capacity有限

★ ★ Not recommended for RTX 4090:
  → Draft Model: 需要完整LM (~7GB) → 24GB不够target+draft
  → Medusa: acceptance低于EAGLE → parallel head independent → 不利用hidden propagation
  → DFlash: 需NVLink → PCIe RTX 4090不适合 → cross-attention效率低
```

## 8. 关键设计洞察

```
1. pass_hidden_states_to_model → EAGLE定义特性 → 区分所有方法!
   → EAGLE/MTP/DFlash: True → 传入target hidden states → 更高acceptance
   → Draft Model: False → 独立LM → 只用token IDs → lower acceptance
   → Medusa: True但无autoregressive → 单次hidden → 并行heads → 不同模式!

2. Triton kernel 全GPU路径 → 无CPU sync → spec decode极快!
   → _update_eagle_draft_inputs_kernel → fused copy+increment
   → 3 rejection kernels → compute_stats → reject → resample → insert
   → 全GPU → 不回CPU → decode spec decode完全在GPU上!
   → vs 传统: CPU-GPU sync per step → bottleneck → vLLM Triton解决!

3. Weight sharing → 内存效率关键!
   → EAGLE: shared embed+lm_head → 只1-2 extra layers → ~1GB
   → MTP: shared all → 只1 extra layer → ~0.5-1GB → 最省!
   → Draft Model: 无sharing → 全新LM → ~7-14GB → ❌
   → Medusa: shared hidden → 只MLP heads → ~0.2-0.5GB → 最小!

4. Rejection Sampling → 数学保证 → 产出与target分布完全一致!
   → p(x)/q(x) >= u → accept → 否则reject → resample from max(p-q, 0)
   → Greedy: argmax一致 → accept → bonus token → all accepted时额外1 token!
   → ★ 最终分布 = target分布 → spec decode不改变output质量!

5. MTP vs EAGLE → 趋势是MTP!
   → DeepSeek-V3/V4 → MTP baked in → 不需要额外训练draft
   → Step3.5/Qwen3.5 → MTP → 新模型自带
   → EAGLE需要额外训练 → 但acceptance更高
   → 未来: 模型自带MTP → spec decode无需额外步骤 → 更简单!

6. Async Spec Decode → GPU-side only → 不回CPU → 最大性能!
   → use_async_spec_decode → draft和target都在GPU → 不sync CPU
   → Deferred corrections → rejection后corrections推迟 → 不阻塞主路径
   → 这是vLLM V1 spec decode的核心优化 → vs V0 → CPU-GPU sync → 慢!

7. DFlash → Parallel Drafting → 未来方向 → 但需NVLink!
   → 所有draft tokens一次forward → cross-attention → non-causal
   → H100 NVLink: DFlash → 极快 → 1次forward → 所有draft
   → RTX 4090 PCIe: DFlash不适合 → cross-attention效率低 → 不推荐
   → GB200集群 → DFlash + NVLS → 最优推理路径 → 未来!
```

---

Sources:
- vllm/v1/spec_decode/eagle.py (EagleProposer)
- vllm/v1/spec_decode/medusa.py (MedusaProposer)
- vllm/v1/spec_decode/draft_model.py (DraftModelProposer)
- vllm/v1/spec_decode/dflash.py (DFlashProposer)
- vllm/v1/spec_decode/step3p5.py (Step3p5MTPProposer)
- vllm/v1/spec_decode/llm_base_proposer.py (SpecDecodeBaseProposer)
- vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (EagleSpeculator)
- vllm/v1/worker/gpu/spec_decode/rejection_sampler.py + rejection_sampler_utils.py
- vllm/v1/sample/rejection_sampler.py
- vllm/model_executor/models/llama_eagle.py + llama_eagle3.py + medusa.py
- vllm/config/speculative.py
- Background agent research (vLLM speculative decoding)
