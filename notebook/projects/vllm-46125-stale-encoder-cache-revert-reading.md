# vLLM #46125/#45093 — Stale Encoder/Prefix Cache After Weight Update

> 2026-06-19 | Deep reading of encoder cache invalidation vs revert debate
> ★★★★★★★★ CRITICAL for RLHF/GRPO: #45093 added cache reset after weight update → #46125 REVERTS it → DANGEROUS for RL training
> ★★★★★★★★ Pattern Family: State Lifecycle Mismatch (same root as #28676 MoE cache clobber, #28679 GDN degeneracy, #44395 illegal memory)
> ★★★★★★★★ Weight Reload Boundary: ANY GPU-resident cache computed with old weights MUST be invalidated at weight-reload boundary

---

## 1. The Two PRs

### #45093 (MERGED) — Fix Stale Encoder Cache After Weight Update

Added `reset_prefix_cache()` and `reset_encoder_cache()` after every `finish_weight_update()`:

```python
# vllm/entrypoints/llm.py + vllm/v1/engine/async_llm.py
def finish_weight_update(self) -> None:
    """Finish the current weight update."""
    self.llm_engine.collective_rpc("finish_weight_update")
    # Invalidate cached state computed with old weights:
    # - prefix cache: KV blocks computed with old weights
    # - encoder cache: multimodal embeddings keyed only by mm_hash
    self.llm_engine.reset_prefix_cache()
    self.llm_engine.reset_encoder_cache()
```

**Why this is correct**:
- Encoder cache entries keyed only by `mm_hash` (NOT including weight version)
- After weight update, old encoder outputs are stale → serving them = silent corruption
- Prefix cache KV blocks computed with old weights → mixing old/new KV = silent corruption
- Same pattern as SGLang #28676 (MoE shuffle cache clobbered on weight reload)

### #46125 (OPEN) — Revert #45093

Arguments:
1. "The decision to reset prefix cache should be left to the user"
2. "We should not be silently resetting prefix cache after every weight update"
3. "I'm not sure about encoder cache though (is it typical to reset encoder cache for multimodal async RL?)"

**Why this revert is DANGEROUS for RLHF/GRPO**:
- RL training updates weights EVERY step (every 30-60 seconds)
- Without cache reset, stale KV blocks and encoder outputs persist across weight updates
- This creates a **State Lifecycle Mismatch**: GPU-resident state (cache) computed with old weights mixed with new-weight inference
- Silent corruption — no error signal, model outputs just get subtly wrong
- Same pattern family as:
  - SGLang #28676: MoE shuffle cache clobbered on weight reload → 64x accuracy blowup
  - SGLang #28679: GDN intermittent degeneracy → worsens over uptime, clears on restart
  - vLLM #44395: wake_up(tags=["weights"]) + forward → illegal memory access (KV cache still asleep)
  - vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake (class variable not in named_buffers)

---

## 2. Root Cause Analysis

### Encoder Cache Staleness

```
Cache key: mm_hash (hash of multimodal input pixels)
Cache value: encoder_output (computed by vision tower weights)

Before weight update:
  mm_hash_A → encoder_output_A (computed with weights_v1)

After weight update (weights_v1 → weights_v2):
  Same mm_hash_A → encoder_output_A (STALE! was computed with weights_v1)
  New request with same pixels → hits cache → gets old encoder output → wrong logits
```

**The problem**: Cache key doesn't include weight version. Two different weight states produce different encoder outputs for the same input, but the cache can only store one.

**★★★★★★★★ Same pattern as MoE RouterReplay**: Router decisions cached with old weights → new weights produce different routing → stale cache routes tokens to wrong experts → NaN/corruption.

### Prefix Cache Staleness

```
KV block: computed with old weights for prompt tokens
After weight update: new weights produce different KV for same prompt tokens

Result: Mixed KV cache — some blocks from old weights, some from new weights
→ Attention computation mixes old/new KV → incorrect logits
→ Silent corruption (no NaN, just subtly wrong outputs)
```

**★★★★★★★★ This is why vLLM #44395 is a BLOCKER**: `wake_up(tags=["weights"])` only wakes weights, not KV cache. Forward pass with awake weights + asleep (stale) KV → illegal memory access on GPU, silent corruption on CPU.

---

## 3. Why User-Controlled Reset is NOT Sufficient

The revert author argues "let the user decide." But:

1. **RL training is automated**: GRPO/RLHF loops update weights automatically every step. There's no human in the loop to manually call `reset_prefix_cache()`.

2. **Silent corruption is the worst bug pattern**: No error signal, no NaN, just subtly wrong outputs that accumulate over training steps. This is EXACTLY what we documented in the Silent Corruption Pattern Family analysis (notebook/projects/silent-corruption-pattern-family-analysis.md).

3. **verl already handles this correctly**: verl's sync trainer lifecycle:
   - `sleep_replicas()` → frees KV cache (step 2 of CheckpointEngineManager)
   - `update_weights()` → transfers new weights
   - `resume_kv_cache_replicas()` → allocates fresh KV cache (step 6)
   This inherently invalidates stale KV because the cache is freed and re-allocated, not just "reset."

4. **SGLang handles this correctly**: After weight reload, SGLang's dict.clear() on MoE shuffle cache (fix for #28676) and ring cursor reset (fix for #28679) both invalidate GPU-resident caches at the weight-reload boundary.

5. **RTX 4090 GRPO concern**: If #46125 revert is merged, vLLM's RLHF weight-update path will silently serve stale KV/encoder outputs → GRPO advantage computation on wrong logits → training degradation → potentially complete collapse.

---

## 4. Proposed Solution

Instead of reverting #45093 entirely, the correct approach is:

### Option A: Keep #45093 but make it configurable

```python
def finish_weight_update(self, *, reset_cache: bool = True) -> None:
    """Finish the current weight update."""
    self.llm_engine.collective_rpc("finish_weight_update")
    if reset_cache:
        self.llm_engine.reset_prefix_cache()
        self.llm_engine.reset_encoder_cache()
```

For RLHF: `reset_cache=True` (default, safe)
For single-step weight update (e.g. loading a different model): `reset_cache=False` (user can control)

### Option B: Add weight version to cache keys

```python
# Instead of:
encoder_cache_key = mm_hash  # only input-dependent

# Use:
encoder_cache_key = (mm_hash, weight_version)  # input + weight-dependent
```

This is the "correct" architectural fix — cache keys include the weight version, so stale entries are automatically invalidated. But this requires significant refactoring.

### Option C: verl-style lifecycle (free + re-allocate)

Already implemented in verl's CheckpointEngineManager:
- Step 2: `release_kv_cache_replicas()` — free KV cache
- Step 6: `resume_kv_cache_replicas()` — allocate fresh KV cache

This implicitly invalidates stale KV because the cache is destroyed and recreated.

---

## 5. RTX 4090 GRPO Deployment Rules

| Rule | MUST DO | MUST NOT |
|------|---------|----------|
| Weight update boundary | Reset prefix cache + encoder cache after EVERY weight update | Assume caches are still valid after weight update |
| verl RLHF | Use sync trainer (naive checkpoint engine handles lifecycle) | Use vLLM without cache reset in RLHF loop |
| vLLM RLHF | Call reset_prefix_cache() + reset_encoder_cache() after update_weights() | Skip cache reset "for performance" |
| SGLang RLHF | Use weight-load funnel with dict.clear() on MoE cache | Assume MoE cache survives weight reload |
| Cache key design | Include weight version in cache key (or invalidate at boundary) | Use only input-dependent cache keys across weight updates |

---

## References

- vLLM #45093: https://github.com/vllm-project/vllm/pull/45093 (MERGED — cache reset after weight update)
- vLLM #46125: https://github.com/vllm-project/vllm/pull/46125 (OPEN — REVERT of #45093)
- vLLM #44910: https://github.com/vllm-project/vllm/issues/44910 (original stale encoder cache bug)
- vLLM #44395: https://github.com/vllm-project/vllm/issues/44395 (illegal memory access after partial wake)
- SGLang #28676: MoE shuffle cache clobbered on weight reload
- SGLang #28679: GDN intermittent degeneracy (same pattern family)
- vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake
- State Lifecycle Mismatch pattern: notebook/projects/state-lifecycle-mismatch-pattern-family-derivation.md
- Silent Corruption pattern: notebook/projects/silent-corruption-pattern-family-analysis.md
- Cross-framework Partial Wake Safety: notebook/projects/cross-framework-partial-wake-safety-analysis.md
