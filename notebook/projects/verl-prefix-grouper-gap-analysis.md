# verl PrefixGrouper Gap Analysis — From Attention-Only PS to Full-Model PS

> 2026-06-07 | Critical finding: Current PG saves attention only (0.99x), real potential is full-model PS (2.46x)

## Executive Summary

The current verl PrefixGrouper (PR #4368) **only saves attention-level computation** for prefix tokens. RTX 4090 benchmarks show this gives **0.99x speedup** for GRPO n=8 (memory-bound, KV read unchanged). The real speedup potential lies in **full-model forward PS** (skipping MLP + QKV_proj for prefix tokens), which gives **2.46x speedup** — a 2.5x gap.

This gap is the single most impactful optimization opportunity for verl GRPO training acceleration.

## 一、Current Implementation Analysis

### Architecture (PR #4368, merged)

```
PrefixGrouper (external package) → monkey_patch.py → ALL_ATTENTION_FUNCTIONS
                                     ↓
                    intercepts attention forward calls
                                     ↓
            prefix_grouper.forward(attn_func, Q, K, V) → prefix self-attn + suffix concat-attn
```

### What PG Currently Does

1. **Input construction**: `concat_input(prefix_ids, prefix_mask, responses, response_mask)` → creates batch with prefix + suffixes
2. **Attention interception**: Monkey-patch wraps ALL_ATTENTION_FUNCTIONS → PG handles prefix/suffix splitting in attention
3. **Output splitting**: `prefix_grouper.split_output(logits)` → separates prefix and suffix logits
4. **Logprob extraction**: `logprobs_from_logits(suffix_out, completion_ids)` → only suffix logprobs needed for training

### What PG Currently Does NOT Do

| Operation | Provider (full) | Reuser (suffix-only) | Current PG saves? |
|-----------|----------------|---------------------|------------------|
| QKV_proj (prefix) | ✓ computed | ✗ **still computed** (padded input) | ✗ NO |
| Attention (prefix) | ✓ computed | ✗ intercepted by PG → not computed | ✓ YES |
| MLP (prefix) | ✓ computed | ✗ **still computed** (padded input) | ✗ NO |
| LayerNorm (prefix) | ✓ computed | ✗ **still computed** (padded input) | ✗ NO |

**Critical**: The model still does full forward on ALL tokens (including padded prefix in reusers). PG only intercepts attention → the MLP, QKV_proj, and LayerNorm still compute on prefix tokens for every sample → **no compute savings on 91.2% of compute-bound operations**!

## 二、Benchmark Evidence

### Pure Attention PS (what PG does) — RTX 4090

| Config | compute节省% | time节省% | speedup |
|--------|-------------|----------|---------|
| GRPO n=8, prefix=512, suffix=256 | 58.3% | **-0.9%** | **0.99x** |
| GRPO n=4, prefix=512, suffix=256 | 50.0% | 3.7% | 1.04x |
| GRPO n=16, prefix=512, suffix=256 | 62.5% | 12.7% | 1.15x |

**结论**: Attention是memory-bound → PS减少compute但不减KV读取量 → time几乎不变 → 0.99x!

### Full-Model Forward PS (what PG should do) — RTX 4090

| Config | compute节省% | time节省% | speedup |
|--------|-------------|----------|---------|
| GRPO n=8, prefix=512, suffix=256 | 58.3% | **59.3%** | **2.46x** |
| GRPO n=4, prefix=512, suffix=256 | 50.0% | 47.8% | 1.91x |
| GRPO n=16, prefix=512, suffix=256 | 62.5% | 63.6% | 2.75x |

**结论**: MLP占82%+QKV_proj占9.2% → 91.2% compute-bound → PS跳过prefix的全部层 → 2.46x!

### Training PS (forward+backward) — RTX 4090

| Config | speedup (fwd-only) | speedup (training) | ratio |
|--------|--------------------|-------------------|-------|
| n=4, 75% prefix | 2.08x | 1.59x | 0.76x |
| n=8*, 67% prefix | 2.46x | ~1.87x (estimated) | 0.76x |

*Training speedup ≈ forward-only × 0.76 due to backward+optimizer overhead

### Speedup Gap Analysis

```
Current PG (attention-only): 0.99x  ←  what verl currently achieves
Potential (full-model PS):  2.46x  ←  what we could achieve
Gap:                        2.47x  ←  2.5x improvement opportunity!

Source of gap:
  MLP:        82% of layer time, compute-bound → PS saves prefix MLP → biggest win
  QKV_proj:   9.2% of layer time, compute-bound → PS saves prefix QKV_proj
  Attention:  2% of layer time, memory-bound → PS no time savings (current PG focus!)
  Total:      91.2% compute-bound operations not saved by current PG
```

## 三、Proposed Architecture: Full-Model PS for verl

### Three Approaches (Ordered by Impact)

#### Approach 1: Provider-Reuser Forward Splitting (Highest Impact)

```python
# Conceptual implementation
def forward_micro_batch_with_full_model_ps(micro_batch, model):
    # 1. Provider: full forward on prefix+suffix
    provider_logits, provider_hidden = model(input_ids=provider_input_ids)

    # 2. Reusers: suffix-only forward with KV injection
    #    - Inject provider's prefix KV into reuser's attention
    #    - Skip prefix MLP, QKV_proj, LayerNorm entirely
    for reuser in reusers:
        reuser_logits = model(
            input_ids=reuser_suffix_ids,
            prefix_kv=provider_kv_cache,  # inject prefix KV
            prefix_hidden=provider_hidden_at_prefix_end,  # for residual connection
        )

    # 3. Prefix-last logprob restoration
    #    - Provider's prefix-last token logprob is already computed
    #    - Reusers need provider's prefix-last logprob for training correctness
```

**Challenges**:
- Requires model-level modification (not just attention monkey-patch)
- KV injection needs custom attention backend (FlashAttention/Magi Attention)
- Prefix-last restore needs careful logprob handling
- Backward pass needs gradient flow through shared prefix computation

**Estimated speedup**: 1.87x (n=8, training)

#### Approach 2: Gradient Checkpointing for Prefix (Medium Impact)

```python
# Instead of skipping prefix forward, recompute during backward
def forward_with_prefix_checkpointing(model, input_ids, prefix_len):
    # 1. Forward: compute prefix with gradient checkpointing (no activations saved)
    prefix_hidden = checkpoint(model_layers[:prefix_len], input_ids[:prefix_len])

    # 2. Forward: compute suffix normally (save activations)
    suffix_hidden = model_layers[prefix_len:](prefix_hidden, input_ids[prefix_len:])

    # 3. Backward: prefix is recomputed → saves 50% activation memory
    #    - More concurrent samples → bigger batch → better throughput
```

**Benefits**: Saves activation memory (not compute) → allows bigger batches → indirect throughput gain

**Estimated impact**: 30-50% more concurrent → indirect speedup ~1.3-1.5x

#### Approach 3: Tree-based Prefix Sharing (Magi Attention, Medium Impact)

```python
# Current: flat grouping (all samples share same prefix)
# Magi: tree-based grouping (hierarchical prefix sharing within micro-batch)

# Example: 3 prompts with different prefixes
# Prompt A: "What is 2+3?" → responses: [A1, A2, A3, A4]
# Prompt B: "What is 2+4?" → responses: [B1, B2, B3, B4]
# Prompt C: "Explain gravity" → responses: [C1, C2, C3, C4]

# Flat grouping: 3 groups, each shares their own prefix
# Tree grouping: A and B share "What is 2+" → 2 groups (AB, C) + sub-groups
# → More prefix sharing → more savings
```

**Benefits**: Better prefix sharing within heterogeneous micro-batches (different prompts with partial overlap)

**Estimated impact**: For heterogeneous batches with partial prefix overlap → up to 30% more prefix sharing → ~0.3x additional speedup over flat grouping

### Recommended Priority

```
P0: Approach 1 (Full-model PS) — 2.5x gap, highest impact
P1: Approach 3 (Tree-based) — complements P0, additional ~0.3x
P2: Approach 2 (Checkpointing) — indirect, easier to implement first
```

## 四、Concrete Contribution Plan

### Phase 1: RFC + Benchmark Evidence (Immediate)

1. Create RFC documenting:
   - Current PG limitation (attention-only PS, 0.99x)
   - Benchmark evidence (RTX 4090, 0.99x vs 2.46x)
   - Proposed full-model PS architecture
   - Estimated speedup (1.87x training with n=8)

2. Post on verl #6401 discussion:
   - "Current PrefixGrouper only saves attention (0.99x for long sequences). The real win is full-model PS (2.46x). Here's the benchmark evidence and proposed architecture."

### Phase 2: FSDP Compatibility Fix (Easy, Good First PR)

Current PG has FSDP issues (from #6401 discussion):
- PG doesn't work with FSDP2 (sharded parameters)
- PG doesn't work with gradient checkpointing
- These need fixing before full-model PS can be integrated

### Phase 3: Full-Model PS Implementation (Medium-Hard)

Implement provider-reuser forward splitting:
1. Create `verl/trainer/ppo/full_model_ps_utils.py` with:
   - `build_full_model_ps_plan()` — compute prefix_len, suffix_len, provider/reuser assignments
   - `forward_provider()` — full forward with KV cache extraction
   - `forward_reuser()` — suffix-only forward with KV injection
   - `restore_prefix_last_logprobs()` — logprob restoration
2. Custom attention backend for KV injection (FlashAttention-based)
3. Integration with verl's dp_actor.py and ray_trainer.py

## 五、Comparison with Related Work

| Project | Approach | Scope | Speedup |
|---------|----------|-------|---------|
| **verl PrefixGrouper** (PR #4368) | Attention monkey-patch | Attention-only | **0.99x** (long seq) |
| **prefix-0501** (SandAI) | One-Forward + KV Injection + Prefix-Last Restore | Full-model | **2.46x** (fwd) / ~1.87x (training) |
| **Magi Attention** (SandAI, #6401) | Prefix Tree + Sparse Attention | Attention-only + tree | ~3x (claimed, attention level) |
| **vLLM Cascade Attention** | Block-level prefix sharing | Attention-only (inference) | ~1.04x (long seq) |

**Key**: prefix-0501's full-model PS approach is the only one that captures the MLP savings. Magi Attention's 3x claim is at the attention level only, which is 18% of total layer time.

## 六、verl #6401 Discussion Points

### Current Step-2+ Loss Disagreement

The verl #6401 RFC has a "step-2+" issue where PG loss diverges from baseline loss. This is likely related to:
1. **Prefix-last token handling**: PG removes the prefix-last token from suffix output, but the logprob for this token needs special handling
2. **Attention mask differences**: PG's prefix/suffix attention mask vs full causal mask → potential numerical differences
3. **Gradient flow**: When PG intercepts attention, does autograd correctly flow through the prefix computation?

### Suggested Fix for Step-2+ Issue

- Verify that `include_prefix_last=1` correctly restores the prefix-last logprob
- Compare PG loss vs baseline loss on identical inputs (should match exactly)
- If they differ, investigate attention mask semantics (prefix causal vs suffix causal)

Sources:
- RTX 4090 Benchmarks: notebook/fundamentals/prefix-sharing-packed-thd-rtx4090.md (attention-only), full-model-ps-rtx4090.md (full-model), grpo-training-ps-rtx4090.md (training)
- verl PR #4368: PrefixGrouper integration
- prefix-0501 project: notebook/projects/prefix-0501-project-analysis.md
- Magi Attention RFC: notebook/fundamentals/magi-attention-prefix-sharing-comparison.md