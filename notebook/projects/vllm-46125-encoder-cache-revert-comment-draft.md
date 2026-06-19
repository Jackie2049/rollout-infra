# vLLM #46125 Comment Draft — P6 C19: Stale Encoder/Prefix Cache After Weight Update Revert

> ★★★★★★★★ DRAFT — NEEDS USER AUTHORIZATION BEFORE POSTING
> Priority: P6 (RTX 4090 RLHF critical — same pattern family as SGLang #28676, #28679, vLLM #44395)
> Target: https://github.com/vllm-project/vllm/pull/46125

---

## Comment Body

This revert raises a critical concern for RLHF/GRPO training workflows where weight updates happen automatically every training step (~30-60 seconds).

**The core issue is a State Lifecycle Mismatch at the weight-reload boundary:**

When weights are updated but GPU-resident caches (prefix KV blocks, encoder outputs) are NOT reset, subsequent inference silently mixes state computed with old weights and new weights. This produces subtly incorrect outputs without any error signal — the worst possible bug pattern for RL training.

**Cross-framework evidence that cache invalidation at weight-reload boundaries is essential:**

1. **SGLang #28676** — MXFP8 MoE shuffle cache was CLOBBERED after RL weight reload (64x accuracy blowup). Fix: `dict.clear()` on cache + weight-load funnel call (+28/-2). If cache isn't invalidated, stale routing decisions produce catastrophic errors.

2. **SGLang #28679** — GDN intermittent decode degeneracy worsens over uptime, clears on restart. Root cause: GPU-resident state lifecycle mismatch. Same pattern family.

3. **vLLM #44395** — `wake_up(tags=["weights"])` + forward → illegal memory access because KV cache was still "asleep" (stale). Blocked, not merged.

4. **vLLM-Ascend #10684** — DSA Hadamard stored as CLASS VARIABLE (not model buffer) → invisible to `named_buffers()` → ALL-ZERO after sleep/wake. verl RLHF BLOCKER.

5. **verl V1 trainer** — Correctly handles this via lifecycle: `release_kv_cache_replicas()` (step 2) → `update_weights()` → `resume_kv_cache_replicas()` (step 6). KV cache is freed and re-allocated, not just "reset" — inherently invalidating stale state.

**The mathematical risk for GRPO:**

In GRPO training, advantage computation relies on correct logprob outputs from the rollout engine. If stale KV blocks produce subtly wrong logits, the advantage estimates are corrupted. Since GRPO normalizes by group mean/std, even small systematic biases compound across training steps — no NaN signal, just gradual training degradation.

**Proposed solution instead of full revert:**

Rather than removing cache reset entirely, make it configurable:

```python
def finish_weight_update(self, *, reset_cache: bool = True) -> None:
    """Finish the current weight update."""
    self.llm_engine.collective_rpc("finish_weight_update")
    if reset_cache:
        self.llm_engine.reset_prefix_cache()
        self.llm_engine.reset_encoder_cache()
```

- RL training: `reset_cache=True` (default, safe — stale state is dangerous)
- One-shot weight load: `reset_cache=False` (user can control if they know weights won't change again)

This preserves the safety guarantee for automated RL workflows while giving manual control for one-shot scenarios.

For encoder cache specifically, a more robust long-term fix would be adding weight version to cache keys (so stale entries are automatically evicted). But the short-term safety measure is cache reset after weight update — removing it creates a RLHF/GRPO regression risk.

---

## Priority Assessment

- **P6**: Not a unique theoretical contribution (pattern family is well-documented), but the cross-framework evidence matrix and RLHF/GRPO-specific risk analysis provide actionable value
- **RTX 4090 impact**: CRITICAL — verl sync trainer + vLLM rollout is the #1 RTX 4090 GRPO deployment path. If cache reset is removed, this path becomes unsafe.
