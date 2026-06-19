# vLLM #46007 — Orthrus Speculative Decoding Deep Reading

> 2026-06-19 | vLLM PR #46007 | Block-Diffusion Speculative Decoding (Dual-View)
> Author: HaizhouPeng (contributor) | Reviewer: benchislett (CHANGES_REQUESTED)
> Status: OPEN | Labels: documentation, new-model, speculative-decoding, v1
> +825/-24 lines across 13 files | mergeable_state: blocked
> Source: https://github.com/vllm-project/vllm/pull/46007

---

## Issue Metadata

| Field | Value |
|-------|-------|
| Number | #46007 |
| Title | [Features] Add Orthrus speculative decoding support |
| Author | HaizhouPeng |
| State | OPEN |
| Labels | documentation, new-model, speculative-decoding, v1 |
| Created | 2026-06-18T06:46:45Z |
| Updated | 2026-06-18T18:23:22Z |
| Merged | null (not merged) |
| Draft | false |
| Additions | 825 |
| Deletions | 24 |
| Changed files | 13 |
| Mergeable state | blocked |
| Head branch | add-orthrus (from HaizhouPeng/vllm) |
| Base branch | main |
| Commits | 6 (initial by aminous, merges by HaizhouPeng) |
| Review state | CHANGES_REQUESTED (benchislett) |

---

## Full Description

### Summary

This PR adds Orthrus support to vLLM's speculative decoding path. It:
1. Registers the Orthrus model architecture (`OrthrusForCausalLM`)
2. Wires the Orthrus proposer into the v1 GPU model runner
3. Enables the `orthrus` speculative method
4. Adds scheduler/config handling for the extra block-diffusion lookahead token
5. Adds Orthrus coverage to model registry examples
6. Extends the DFlash lookahead tests to cover Orthrus behavior

### Test Environment

- GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM
- Driver: 580.105.08
- CUDA: 13.0

### Launch Commands

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /root/Orthrus-Qwen3-4B \
    --host 0.0.0.0 \
    --port 8898 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.9 \
    --attention-backend flash_attn \
    --speculative-config '{"method": "orthrus", "num_speculative_tokens": 16}'
```

### Benchmark Results (philschmid/mt-bench)

| Model | Concurrency | Prompts | Output tok/s | Req/s | Mean TTFT ms | Mean TPOT ms | Failed | Acceptance % | Acceptance len | Speedup vs Qwen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Orthrus | 1 | 30 | 269.64 | 0.26 | 59.23 | 3.65 | 0 | 17.76 | 3.84 | 1.66x |
| Qwen3-4B | 1 | 30 | 162.55 | 0.16 | 26.81 | 6.13 | 0 | - | - | - |
| Orthrus | 3 | 90 | 703.01 | 0.69 | 52.43 | 4.03 | 0 | 16.79 | 3.69 | 1.59x |
| Qwen3-4B | 3 | 90 | 442.88 | 0.43 | 33.73 | 6.75 | 0 | - | - | - |
| Orthrus | 5 | 150 | 1090.84 | 1.07 | 51.66 | 4.46 | 0 | 16.95 | 3.71 | 1.64x |
| Qwen3-4B | 5 | 150 | 663.19 | 0.65 | 38.66 | 7.51 | 0 | - | - | - |

---

## Comments

### Issue-Level Comments (3 total)

| # | Author | Date | Content |
|---|--------|------|---------|
| 1 | github-actions[bot] | 2026-06-18T06:46:56Z | Standard welcome message + agent guidelines |
| 2 | mergify[bot] | 2026-06-18T06:47:21Z | Documentation preview link |
| 3 | benchislett | 2026-06-18T18:23:22Z | "Have you benchmarked against DFlash?" |

### Review Comments (1 total)

| # | Author | Date | Content |
|---|--------|------|---------|
| 1 | benchislett | 2026-06-18T18:23:22Z | (inline comment, not visible separately) |

### Reviews (1 total)

| # | Author | State | Submitted | Body |
|---|--------|-------|-----------|------|
| 1 | benchislett | CHANGES_REQUESTED | 2026-06-18T18:22:57Z | Two concerns: (1) Additional validations needed: E2E correctness and acceptance rate testing, performance on server-class hardware (H100/H200/B200/B300), and ideally support for a model other than Qwen3 8B. Still somewhat skeptical of the value of this technique. (2) Shifting focus onto ModelRunnerV2 for speculative decoding features. Do not plan to accept any new speculation backends for the V1 gpu_model_runner. |

---

## Technical Analysis

### 1. What is Orthrus Speculative Decoding?

Orthrus is a **block-diffusion speculative decoding** method based on the paper "Fast, lossless LLM inference via dual-view diffusion decoding" (chiennv2000/orthrus on GitHub). The key concept is **dual-view**: the model has two attention paths through the same transformer backbone:

- **AR (autoregressive) path**: Standard causal attention → generates tokens one at a time (the target model)
- **Diffusion path**: Non-causal attention with separate QKV projections → generates a block of tokens in parallel (the draft model)

The diffusion path uses **separate linear projections** (`qkv_proj_diff`, `o_proj_diff`, `q_norm_diff`, `k_norm_diff`) alongside the standard AR projections, but **shares all other weights** (embed_tokens, MLP, lm_head, decoder backbone). This means:
- No separate draft model needed — the target model IS the draft model
- The diffusion attention layer writes to the **shared KV cache** (via `update_kv_cache_when_sharing = True`)
- KV sharing is critical: diffusion pass sees the same KV history as the AR pass, but with bidirectional attention within the draft block

**Architecture summary**:

```
OrthrusForCausalLM:
  → OrthrusModel (extends Qwen2Model)
    → OrthrusDecoderLayer per layer:
      → OrthrusAttention:
        → Standard: qkv_proj + o_proj + attn (AR path)
        → Diffusion: qkv_proj_diff + o_proj_diff + q_norm_diff + k_norm_diff + attn_diff
        → attn_diff uses kv_sharing_target_layer_name → shares KV cache with AR attn!
        → attn_diff.update_kv_cache_when_sharing = True → writes diffusion KV to shared cache
        → is_diffusion_pass flag switches between AR and diffusion paths
      → Qwen3MLP (shared, no separate path)
      → RMSNorm (shared)
  → OrthrusDiffusionModel:
    → Wrapper that calls OrthrusModel.forward(is_diffusion_pass=True)
    → Gives torch.compile a distinct __call__ entry for diffusion
    → Uses object.__setattr__ to avoid duplicate parameter registration

Shared weights: embed_tokens, lm_head, all MLP layers, all RMSNorm layers
Diffusion-only weights: qkv_proj_diff, o_proj_diff, q_norm_diff, k_norm_diff per layer
```

### 2. How Orthrus Differs from EAGLE/EAGLE3/Medusa/Draft-Model/DFlash

| Aspect | Orthrus | EAGLE/EAGLE3 | Medusa | Draft Model | DFlash |
|--------|---------|-------------|--------|-------------|--------|
| **Draft source** | Target model itself (diffusion path) | Lightweight draft head | MLP prediction heads | Separate small LM | Separate non-causal draft model |
| **Topology** | Block (parallel) | Chain/Tree (sequential) | Tree (parallel heads) | Chain (sequential) | Block (parallel) |
| **Feature level** | Feature-level (shared hidden states) | Feature-level (target hidden state feed) | Token-level (parallel heads) | Token-level (independent) | Feature-level (cross-attention) |
| **Extra parameters** | ~4-8% per layer (diff QKV+O proj) | ~0.5GB (1-2 draft layers) | ~1% (MLP heads) | Full model (7-14GB) | Full separate model (~1-2GB) |
| **Weight sharing** | ALL backbone + MLP + embed + lm_head | embed + lm_head (optional) | None (MLP heads separate) | None | None (separate model) |
| **KV sharing** | YES (diffusion shares AR KV) | YES (draft layers own KV group) | NO (no attention in draft) | NO (separate KV cache) | NO (separate KV pool) |
| **Draft generation** | 1 forward pass (parallel block) | K forward passes (autoregressive) | 1 forward pass (parallel heads) | K forward passes (autoregressive) | 1 forward pass (parallel block) |
| **pass_hidden_states** | False (self-model, no external feed) | True (EAGLE feeds target hidden) | True (Medusa reads target hidden) | False (independent) | True (DFlash reads target hidden) |
| **parallel_drafting** | True | False (sequential) | N/A (single pass) | False | True |
| **Memory overhead** | Low (~4-8% more params) | Medium (~0.5-1GB) | Low (~0.2-0.5GB) | Very high (7-14GB) | Medium (~1-2GB) |
| **No separate model repo needed** | YES | NO (need EAGLE draft model) | NO (need Medusa heads) | NO (need separate draft) | NO (need DFlash model) |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Key architectural insight**: Orthrus is the ONLY speculative decoding method where the draft IS the target model with a different attention mode. This eliminates:
1. Draft model loading overhead (no separate model)
2. Draft-target distribution mismatch (same backbone → same representation)
3. Draft model memory overhead (only extra QKV+O projections per layer)
4. Draft-target synchronization issues (shared KV cache)

But the acceptance rate is low (~17-18%) because:
- Non-causal (bidirectional) attention within the block → draft tokens can see future positions → violates causality → target model (causal) disagrees with draft predictions
- Block generation is "coarse" → entire block at once → no sequential conditioning → lower per-token quality
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3. OrthrusProposer Implementation Details

The `OrthrusProposer` (244 lines) extends `SpecDecodeBaseProposer`:

```python
class OrthrusProposer(SpecDecodeBaseProposer):
    # Key properties:
    pass_hidden_states_to_model = False  # target model IS the draft, no external feed
    parallel_drafting = True             # block-diffusion: all tokens in 1 forward pass
    use_local_argmax_reduction = True    # O(2*tp_size) vs O(vocab_size) communication

    # load_model: uses target_model directly (no separate model loading!)
    # → self.model = target_model
    # → Requires model.get_top_tokens() for local argmax reduction
    # → Extracts attn_diff layers as draft attention layers

    # set_inputs_first_pass: uses copy_and_expand_dflash_inputs_kernel (same as DFlash!)
    # → Expands target tokens + mask tokens for parallel drafting
    # → num_query_per_req = 1 + num_speculative_tokens (bonus token included)
    # → causal=False in CommonAttentionMetadata → non-causal diffusion attention
    # → Shifts token_indices_to_sample: skip last mask position, include bonus

    # dummy_run: calls model.forward(is_diffusion_pass=True)
    # build_model_inputs_first_pass: passes is_diffusion_pass=True to model
```

**Critical design decisions**:

1. **Reuses DFlash infrastructure**: `copy_and_expand_dflash_inputs_kernel`, same `parallel_drafting` logic, same scheduler lookahead tokens (`num_lookahead_tokens = num_spec_tokens + 1`). Orthrus is architecturally similar to DFlash but uses the SAME model (not a separate one).

2. **KV cache sharing via `update_kv_cache_when_sharing`**: The `Attention` class gets a new flag `update_kv_cache_when_sharing`. Previously, when `kv_sharing_target_layer_name` was set, KV cache updates were skipped (the layer "borrowed" KV from the target layer). Orthrus's `attn_diff` needs to WRITE its diffusion K/V into the shared KV cache so the block tokens have bidirectional context. This flag is `True` only for Orthrus diffusion layers.

3. **Local argmax reduction**: Orthrus uses `get_top_tokens()` from the model to do local argmax reduction on TP shards → communication cost O(2*tp_size) instead of O(vocab_size). This is a significant optimization for multi-GPU settings.

4. **`model_returns_tuple() = False`**: Unlike EAGLE which returns `(hidden_states, all_hidden_states)` tuple, Orthrus returns plain `hidden_states` — no hidden state propagation needed since the model IS the draft.

5. **No prefix caching**: The PR explicitly uses `--no-enable-prefix-caching` in benchmarks. Prefix caching with block-diffusion speculative decoding is likely problematic (same issue as DFlash prefix-cache corruption #42971).

### 4. Benchmark Data Analysis

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Acceptance rate is only 17.76%** — this is the lowest acceptance rate among all speculative decoding methods in our tracked data!

Comparative acceptance rates:
- Orthrus: ~17.76% (from PR benchmark)
- N-gram: ~30-50% (from theory note)
- Medusa: ~50-65% (from theory note)
- MTP (DSV4): ~70-85% (from theory note)
- EAGLE: ~85-92% (from theory note)
- DFlash: ~80-90% (from theory note)

Why so low? Applying our acceptance rate math:

alpha = 17.76% → TV(p,q) = 82.24% → massive distributional disagreement!

For alpha=0.1776, K=16:
E[output] = 1 + alpha*(1-alpha^K)/(1-alpha) = 1 + 0.1776*(1-0.1776^16)/(1-0.1776) = 1 + 0.1776*1/0.8224 = 1 + 0.2158 = 1.216

But the measured acceptance length is 3.84, NOT 1.216!

This discrepancy suggests:
- The acceptance rate of 17.76% is per-position in the block, but the acceptance length of 3.84 means on average 3.84 tokens are accepted per iteration
- For block-diffusion: K=16 draft tokens generated in parallel → target verifies all → acceptance is sequential but draft generation is parallel
- The speedup formula is different for parallel drafting: S = (1 + avg_accept_len) * t_target / (t_draft_parallel + t_verify)
- Since t_draft_parallel uses the SAME model (just diffusion path), t_draft_parallel ≈ t_target (one forward pass)
- Net speedup: 1.66x with 3.84 avg acceptance → (1+3.84)*t_target / (t_target + t_target) = 4.84/2 = 2.42x theoretical → 1.66x measured (overhead + rejection cost)

The 1.66x speedup is modest compared to:
- EAGLE (2.5-4x on single GPU)
- MTP (1.8-2.4x)
- N-gram (1.5-2x on repetitive content)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## RTX 4090 Impact

### 1. Memory Analysis

Orthrus on RTX 4090 memory budget:

```
RTX 4090 24GB total

For Qwen3-4B (BF16 ~8GB):
  → Base model: ~8GB
  → Diffusion extra weights (qkv_proj_diff + o_proj_diff + q_norm_diff + k_norm_diff per layer):
    → Per layer: 3 * hidden_size^2 * 2/tp (QKV) + hidden_size^2 * 2/tp (O) + 2 * head_dim (norms)
    → Qwen3-4B: hidden_size=2560, 40 layers → ~0.5GB extra
    → Total: ~8.5GB → leaves 15.5GB for KV cache → viable!

For Qwen3-8B (BF16 ~16GB):
  → Base model: ~16GB
  → Diffusion extra: ~1.0GB per layer * ~32 layers? → ~1-2GB extra
  → Total: ~17-18GB → leaves 6-7GB for KV cache → tight but possible!

For Qwen3-14B (BF16 ~28GB):
  → Base model alone exceeds RTX 4090 → NOT viable without quantization
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**RTX 4090 memory viability**: Orthrus is MORE memory-efficient than DFlash/EAGLE/draft-model because it reuses the target model. The extra diffusion projections (~4-8% per layer) are significantly less than loading a separate draft model. However, the low acceptance rate (~17%) makes it less efficient in terms of throughput gain per unit of memory invested.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 2. GRPO Training Relevance

For verl GRPO training rollout phase:

```
Orthrus vs other methods for RTX 4090 GRPO rollout:

| Method | Extra Memory | Acceptance | Speedup | GRPO Viability |
|--------|-------------|------------|---------|----------------|
| N-gram | 0 | 30-50% | 1.5-2x | ★★★★★ BEST (zero overhead) |
| EAGLE | ~0.5-1GB | 85-92% | 2.5-4x | ★★★★ GOOD |
| Orthrus | ~0.5-1GB | 17.76% | 1.66x | ★★ MARGINAL |
| DFlash | ~1-2GB | 80-90% | ? | ★ NOT viable (needs separate model) |

Orthrus is NOT the best choice for RTX 4090 GRPO rollout because:
1. Low acceptance rate (17.76%) → only 1.66x speedup → marginal benefit
2. N-gram gives 1.5-2x with ZERO extra memory → better ROI
3. EAGLE gives 2.5-4x with similar memory (~0.5-1GB) → much better ROI
4. Orthrus requires Qwen3-based checkpoints with Orthrus diffusion weights → limited model availability
5. Orthrus needs flash_attn backend + no prefix caching → more restrictive configuration

However, Orthrus has UNIQUE advantages:
1. No separate model repository → simpler deployment
2. Same backbone = identical representation space → distributional closeness is theoretically high
3. The low acceptance may be due to the "parallel block" approach — sequential conditioning might improve it
4. If Orthrus acceptance can be improved (better diffusion training, adaptive K), it could surpass EAGLE in simplicity
```

### 3. SM89 / Batch Invariance Concerns

Orthrus on RTX 4090 (SM89):

```
Potential SM89 issues:

1. Non-causal attention (diffusion path):
   → attn_diff uses causal=False → different attention mask
   → Flash attention on SM89 supports non-causal mode? → needs verification
   → If flash_attn backend required → SM89 may need enforce_eager=True
   → RTX 5090 (SM100) benchmark works → SM89 support unclear

2. Block-diffusion + CUDA graph:
   → parallel_drafting=True → fixed batch shapes → CUDA graph viable?
   → But benchislett notes ModelRunnerV2 migration → V1 CUDA graphs may change
   → Same concern as EAGLE3 CUDA graph crash (#28569) → batch shrinking = danger

3. torch.compile compatibility:
   → Orthrus uses @support_torch_compile decorator
   → Two separate compiled entry points: OrthrusModel.forward + OrthrusDiffusionModel.forward
   → SM89 Inductor fusion guard (P9) → may block fp8→bf16 fusion → BUT Orthrus uses BF16
   → Dual-view torch.compile → two compilation contexts → potential SM89 batch-dependent fusion risk
   → autotune_at_compile_time=False (PyTorch #187636) → reduces risk

4. KV cache sharing + diffusion update:
   → attn_diff.update_kv_cache_when_sharing = True → writes diffusion KV into shared slots
   → On SM89 → unified_kv_cache_update → Inductor-compiled → SM89 fusion guard applies
   → Potential: diffusion pass overwrites AR KV → verification reads stale KV → same pattern as DSV4 C128 state corruption!

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
SM89 risk assessment: MEDIUM-HIGH
→ Non-causal attention on SM89 needs flash_attn → enforce_eager may be needed
→ KV sharing update during diffusion → potential stale state (same pattern as DSV4 instability!)
→ Dual torch.compile entry points → SM89 batch-dependent fusion risk
→ MUST verify: flash_attn non-causal mode on SM89, diffusion KV overwrite safety
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## Connection Map

### Direct Connections

| Connected Issue/PR | Framework | Connection Type | Relevance |
|---------------------|-----------|-----------------|-----------|
| **#45819** (GDN batch invariance) | vLLM | Architecture overlap | GDN = GatedDeltaNet = Mamba+Attention hybrid → same dual-path concern as Orthrus (AR + diffusion). Orthrus diffusion attention is similar to GDN gated attention in concept (non-causal variant of the same backbone). GDN batch invariance fix → Orthrus may inherit same fix pattern. |
| **#45683** (MoE combine) | vLLM | Determinism overlap | Orthrus uses `use_local_argmax_reduction` → local argmax on TP shards → needs deterministic combine. Same concern as MoE combine: TP-sharded argmax must be deterministic. Orthrus requires `get_top_tokens()` → similar to MoE expert routing determinism. |
| **#45731** (PyTorch 2.13) | vLLM | torch.compile overlap | Orthrus uses `@support_torch_compile` with TWO compiled entry points (OrthrusModel + OrthrusDiffusionModel). PyTorch 2.13 autotune_at_compile_time changes → affects dual-compile-path stability. |
| **#28569** (EAGLE3 CUDA graph crash) | SGLang | CUDA graph risk | Same pattern: spec decode on variable-batch models → CUDA graph crash when batch shrinks. Orthrus uses parallel_drafting → fixed shapes → less risk, but benchislett notes ModelRunnerV2 migration = uncertainty. |
| **#28591/#28612** (DSV4 MTP state mapping) | SGLang | State corruption pattern | Orthrus diffusion writes to shared KV cache → same pattern as DSV4 C128 compress state corruption (constant buffer overwritten during speculative pass). If diffusion KV overwrites AR KV → target verification may see stale KV → same root cause class! |
| **DFlash (vLLM existing)** | vLLM | Architectural sibling | Orthrus reuses DFlash infrastructure (copy_and_expand_dflash_inputs_kernel, parallel_drafting, lookahead tokens). Both are block-diffusion methods. Orthrus = DFlash but with same model (no separate draft). |

### Indirect Connections

| Connected Issue | Framework | Connection Type | Relevance |
|-----------------|-----------|-----------------|-----------|
| **SM89 batch invariance (P9)** | PyTorch | Fusion risk | Orthrus dual torch.compile → two Inductor compilation contexts → SM89 fusion guard applies to BOTH paths. If diffusion path triggers fp8/bf16 fusion → P9 guard blocks it. |
| **#184119** (SM89 fp8 guard) | PyTorch | Validation | Validates P9 thesis: SM89 Inductor class blocks dangerous fusions. Orthrus uses BF16 → no fp8 fusion risk → P9 guard less relevant, but dual-compile-path still triggers SM89 class checks. |
| **verl HYBRID sleep/wake** | verl | Deployment pathway | If Orthrus is used for verl GRPO rollout → needs SGLang/vLLM integration. Currently only vLLM → not yet in SGLang. verl SGLang backend → Orthrus unavailable. |
| **BudgetRefiner (P10)** | vLLM | SLO overlap | Orthrus 1.66x speedup → affects SLO budget. If Orthrus used in rollout → BudgetRefiner must account for variable acceptance rate (17.76% → unstable SLO). |
| **#6794** (delta weight sync) | verl | Weight sync overlap | Orthrus shares target model weights → if used in verl HYBRID, delta sync must include diffusion weights. sleep_level=1 (LoRA adapter) → Orthrus diffusion projections must be synced. |

---

## Key Takeaways

### 1. Orthrus is a Novel Architecture

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
Orthrus represents a genuinely new approach to speculative decoding: **the draft IS the target model** with a different attention mode. This eliminates the draft model entirely and reduces memory overhead to ~4-8% (diffusion projections per layer). However, the low acceptance rate (~17.76%) and modest speedup (1.66x) make it currently inferior to EAGLE (2.5-4x) and N-gram (1.5-2x, zero overhead) for most production use cases.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 2. PR is BLOCKED — Two Major Concerns

benchislett (vLLM reviewer) raised two critical concerns:

**Concern 1**: Additional validations needed:
- E2E correctness and acceptance rate testing beyond mt-bench
- Performance on server-class hardware (H100/H200/B200/B300)
- Support for models other than Qwen3 (currently only Orthrus-Qwen3-1.7B and 4B)
- Skeptical of the technique's value given low acceptance rate

**Concern 2**: ModelRunnerV2 migration:
- vLLM is shifting focus to ModelRunnerV2 for speculative decoding
- No new speculation backends will be accepted for V1 gpu_model_runner
- This effectively BLOCKS the PR from merging in its current form

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**PR assessment**: CHANGES_REQUESTED + ModelRunnerV2 migration = this PR is unlikely to merge in its current V1 form. It would need to be rewritten for ModelRunnerV2, which is a significant architectural change. The PR may need to wait for ModelRunnerV2 spec decode support to mature.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3. Orthrus vs DFlash — Architectural Comparison

Both Orthrus and DFlash are block-diffusion (parallel drafting) methods, but they differ fundamentally:

| Aspect | Orthrus | DFlash |
|--------|---------|--------|
| Draft model | Same as target (diffusion path) | Separate model |
| Weight sharing | ALL (backbone + MLP + embed + lm_head) | None (separate model) |
| KV cache | Shared (diffusion writes to AR KV) | Separate (own KV pool) |
| Extra parameters | ~4-8% per layer (diff projections) | Full separate model (~1-2GB) |
| Memory overhead | Lower (no separate model) | Higher (need separate model weights) |
| Model availability | Only Orthrus-Qwen3 checkpoints | Qwen3-DFlash-0.5B |
| Infrastructure | Reuses DFlash kernel + scheduler | Own implementation |

Orthrus is architecturally simpler (no separate model) but DFlash has higher acceptance (80-90% vs 17.76%). The tradeoff: Orthrus saves memory but sacrifices acceptance.

### 4. KV Cache Sharing Innovation

The `update_kv_cache_when_sharing` flag is a subtle but important change to vLLM's attention infrastructure. Previously, `kv_sharing_target_layer_name` meant "don't update KV cache at all — borrow from target layer." Orthrus needs to both borrow (for history) AND write (for new diffusion positions). This flag enables that dual behavior. This is a general-purpose improvement that could be useful for other KV-sharing scenarios.

### 5. Orthrus Documentation Quote

From `docs/features/speculative_decoding/orthrus.md`:

"Orthrus uses parallel drafting internally. The scheduler reserves KV lookahead slots for the diffusion block, but the Orthrus drafter batch is separate from the target verification batch and does not append extra compute slots to the target batch."

This clarifies that Orthrus's diffusion pass runs as a SEPARATE batch from the target verification — unlike EAGLE/DraftModel which interleave draft and target computation. The scheduler allocates lookahead slots for the diffusion block but the actual compute is separate.

---

## Monitoring Status

### Active Tracking

| Item | Status | Priority |
|------|--------|----------|
| PR #46007 merge status | OPEN, CHANGES_REQUESTED, blocked by ModelRunnerV2 | Watch |
| benchislett's DFlash benchmark question | Unanswered by author | Watch |
| Orthrus acceptance rate improvement potential | Low (17.76%) → needs research | Low priority |
| ModelRunnerV2 spec decode migration | Underway → blocks ALL new V1 spec backends | High priority |
| Orthrus SM89 / RTX 4090 testing | Not tested on SM89 → needs GPU validation | Medium priority |
| Orthrus + prefix caching compatibility | PR uses --no-enable-prefix-caching → likely broken | Watch |
| KV cache sharing update flag generalization | General improvement → may affect other methods | Low priority |

### Update Triggers

- If benchislett approves after ModelRunnerV2 rewrite → reassess for RTX 4090 viability
- If Orthrus acceptance rate improves to >50% → becomes competitive with EAGLE/N-gram
- If Orthrus SGLang implementation appears → verl GRPO rollout pathway opens
- If KV sharing update pattern causes DSV4-style state corruption → CRITICAL blocker
- If SM89 flash_attn non-causal support confirmed → Orthrus RTX 4090 viable

---

## Changed Files Summary

| File | Change | Description |
|------|--------|-------------|
| `vllm/model_executor/models/orthrus.py` | +389 (NEW) | OrthrusForCausalLM model with AR+diffusion dual attention |
| `vllm/v1/spec_decode/orthrus.py` | +244 (NEW) | OrthrusProposer extending SpecDecodeBaseProposer |
| `docs/features/speculative_decoding/orthrus.md` | +49 (NEW) | Orthrus documentation |
| `vllm/config/speculative.py` | +25/-4 | OrthrusModelTypes, use_orthrus(), parallel_drafting, max_num_new_slots |
| `vllm/v1/worker/gpu_model_runner.py` | +30/-6 | OrthrusProposer integration, block-diffusion drafter handling |
| `vllm/model_executor/layers/attention/attention.py` | +19/-6 | update_kv_cache_when_sharing flag for KV sharing writes |
| `vllm/v1/core/sched/scheduler.py` | +3/-7 | Orthrus lookahead tokens (num_spec_tokens + 1) |
| `vllm/v1/spec_decode/llm_base_proposer.py` | +2/-7 | mask_token_id from model_hf_config for Orthrus |
| `vllm/model_executor/models/registry.py` | +2 | Orthrus model registration |
| `tests/models/registry.py` | +2 | Orthrus test registration |
| `tests/v1/spec_decode/test_dflash_lookahead.py` | +55/-3 | Extended DFlash tests for Orthrus behavior |
| `vllm/config/vllm.py` | +3 | Orthrus configuration support |
| `docs/features/speculative_decoding/README.md` | +2/-1 | Orthrus documentation link |

---

## References

### Papers / Repositories
- chiennv2000/orthrus: "Fast, lossless LLM inference via dual-view diffusion decoding" — https://github.com/chiennv2000/orthrus
- Orthrus-Qwen3 models: chiennv/Orthrus-Qwen3-1.7B, Orthrus-Qwen3-4B (HuggingFace)

### Internal Reading Notes
- `/notebook/fundamentals/speculative-decoding-theory-derivation.md` — Spec decode math derivation (acceptance rate, speedup formula)
- `/notebook/fundamentals/vllm-v1-spec-decode-architecture-source-reading.md` — vLLM V1 spec decode architecture (6 Triton kernels, 8 methods)
- `/notebook/projects/vllm-speculative-decoding-reading.md` — vLLM spec decode source reading (Chinese)
- `/notebook/projects/sglang-spec-decode-reading.md` — SGLang spec decode source reading
- `/notebook/projects/vllm-45819-gdn-batch-invariance-reading.md` — GDN batch invariance (dual-path architecture)
- `/notebook/projects/dsv4-systematic-instability-pattern-synthesis.md` — DSV4 state corruption pattern (KV sharing risk)

### Related vLLM Issues
- #45819: GDN batch invariance (dual attention path)
- #45683: MoE combine determinism
- #45731: PyTorch 2.13 + torch.compile
- #46085: aot_eager piecewise compilation backend (dual compile path)
- #42971: DFlash prefix-cache corruption fix (MERGED)
