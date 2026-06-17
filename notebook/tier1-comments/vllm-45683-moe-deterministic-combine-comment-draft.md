# vLLM #45683 Tier 1 Comment Draft — Deterministic MoE Combine

> 2026-06-18 | Tier 1 comment opportunity | CRITICAL for GRPO MoE determinism
> ★★★★★★★★ MoE combine step cross-rank summation order was NOT deterministic → breaks GRPO reward stability

---

## Comment Draft

This is a critical fix for GRPO training with MoE models. A few observations:

### GRPO reward stability implications

In GRPO training, the reward signal must be bit-for-bit reproducible across rollout and training phases. The MoE combine step's cross-rank summation order directly affects:
- **Log probabilities**: If MoE routing affects attention/FFN output, the combine reduction order changes the final token probabilities
- **Reward computation**: Non-deterministic combine → non-deterministic rewards → advantage variance → unstable GRPO training
- **SM89 single GPU**: DP=1 → reduce_scatter = identity → this PR doesn't affect single GPU, but is essential for multi-GPU GRPO MoE

### Alignment with SGLang's approach

SGLang's deterministic inference (#24459) already handles MoE top-k combine deterministically by disabling the small-token torch.compile fast path (#27869). vLLM's approach (fixed-root reduce + scatter) is the NCCL-level equivalent — both ensure the same reduction tree regardless of token routing.

### Suggestion for SM89 coverage

The PR currently only changes behavior when DP world_size > 2. For DP=1 (RTX 4090 single GPU), the deterministic path is never exercised. Consider also:
- Testing the deterministic path on DP=1 to ensure it produces identical results
- Adding a unit test that verifies bit-for-bit reproducibility across different batch sizes (the main SM89 concern from #39096)

### Complementary with VLLM_BATCH_INVARIANT

This PR and VLLM_BATCH_INVARIANT=1 serve complementary roles:
- VLLM_BATCH_INVARIANT: ensures *within-batch* determinism (same batch size = same result)
- This PR: ensures *cross-rank* determinism (same reduction tree = same result)
- Together: complete MoE determinism stack for GRPO

---

## References

- vLLM #45683: https://github.com/vllm-project/vllm/pull/45683
- SGLang #27869: MoE top-k combine deterministic fix
- vLLM #39096: SM89 batch invariance bug
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
