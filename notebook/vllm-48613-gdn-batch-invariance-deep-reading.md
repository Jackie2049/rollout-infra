# vLLM #48613: GDN Batch Invariance for Qwen3.5/Qwen3.6

## Overview
- **Issue**: vllm-project/vllm #48613 by cm2435 (Charlie Masters), July 14, 2026
- **Status**: OPEN, 0 maintainer responses — only author comments
- **Severity**: ★★★★★★★★ CRITICAL for GRPO — GDN batch invariance broken means training-time and inference-time behavior differ with batch size
- **Models affected**: Qwen3.5-0.8B, Qwen3.6-35B-A3B (any GDN architecture)
- **Tests**: FlashInfer: 18 mismatches, Triton/FLA: 22 mismatches at bs=32

## Background: GDN and Batch-Invariant Compute

**GDN (Gated Differential Networks)** is the linear attention architecture used by Qwen3.5/3.6. Unlike softmax attention (which is naturally batch-invariant — each token's attention computes independently of others in the batch), GDN uses recurrent state that can interact across sequences.

**Batch-Invariant Compute (BIC)** mode (`VLLM_BATCH_INVARIANT=1`) guarantees that running a prompt individually (bs=1) produces bitwise-identical results to running it as part of a batch (bs=N). This is critical for:
- **Deterministic debugging**: isolate generation issues without batch-size confounding
- **GRPO training**: rollout generation must be reproducible across batch configurations
- **Testing**: verify model behavior independent of serving batching

## The Bug: Flag-Only Approach Is Insufficient

### Phase 1: Engine Init Failure
With `VLLM_BATCH_INVARIANT=1`, GDN models fail at engine init:
```
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```
This is because `GDNAttentionBackend.supports_batch_invariance()` returns `False` by default — the safe, conservative setting.

### Phase 2: Flag-Only Bypass Fails Validation
Author cm2435 created a spike branch that sets `supports_batch_invariance() -> True` (same approach as PR #45819). Results:

| Backend | Mismatches at bs=32 | Type |
|---------|--------------------|------|
| FlashInfer (default) | **18** | sampled-token divergence + logprob mismatches |
| Triton/FLA (forced) | **22** | selected-token logprob mismatches |

**Conclusion**: The flag is necessary but insufficient. "Appears to be remaining batch-invariance work needed in the GDN execution path and/or recurrent state handling."

## Root Cause Analysis

GDN is a **recurrent linear attention** mechanism. Unlike softmax attention where each query attends independently to keys/values, GDN maintains a recurrent state that accumulates across the sequence. This state is **batch-size-dependent** when:
1. **Recurrent state normalization**: If the state is normalized by total tokens seen (implicit batch-level normalization), batch size affects the normalization factor
2. **Cross-sequence interactions**: If any operation mixes information across sequences in the batch (even inadvertently through kernel tile boundaries)
3. **CUDA kernel tiling**: FlashInfer and Triton GDN kernels tile the recurrent computation differently based on batch dimensions

The 18-22 mismatches at bs=32 suggest the issue manifests in specific tokens (boundary tiles, first/last tokens in sequence), consistent with a **tiling boundary effect** or **state initialization difference**.

## Related Work
- **PR #45819**: The core `supports_batch_invariance() -> True` change. Has "ongoing debate" between @yewentao256 and @yuvalluria about test sufficiency
- **Issue #42960**: Related GDN/BIC support request (found after filing)

## GRPO Impact

For GRPO training on Qwen3.5/3.6 models:
- **Rollout generation** uses vLLM with varying batch sizes (depends on request load)
- **Without batch invariance**: training rollouts are non-deterministic w.r.t. batch size
- **Advantage computation**: If rollout logprobs vary by batch configuration, advantage estimates become inconsistent
- **Loss training**: Model sees different probability distributions depending on serving batching

This is especially critical for:
- **DAPO-style zero/clip mechanisms**: Sensitive to exact probability values
- **UP-GRPO asymmetric clipping**: The self-anchored ratio depends on exact logprobs
- **Multi-node training**: Different nodes may serve different batch sizes

## Fix Approaches (Potential)

From the issue author's proposal:
1. **Validation harness**: `Qwen/Qwen3.5-0.8B` as reference model for deterministic testing
2. **Targeted enablement**: Only enable `supports_batch_invariance()` for GDN paths that pass invariance checks
3. **Documentation**: Explicitly list unsupported GDN backend combinations

The actual code fix likely involves:
- **Recurrent state isolation**: Ensure per-sequence state doesn't leak across batch boundaries in Triton kernels
- **Tile-size invariant computation**: Verify that GDN kernel tiling doesn't introduce batch-size-dependent floating-point ordering
- **State normalization fix**: If GDN state uses count-based normalization, switch to per-sequence counters

## Monitoring
- No maintainer response as of July 14
- Author cc2435 willing to contribute fix
- PR #45819 debate may resolve direction
- **Key trigger**: Any maintainer response indicating acceptance of contribution

## Cross-Framework Connections
- The same GDN batch-invariance issue affects **SGLang** (which also supports GDN models)
- SGLang #28679 (GDN intermittent degeneracy) may be related — non-deterministic behavior that worsens over uptime
- Pattern: **P3 (State Lifecycle Mismatch)** — GDN recurrent state handling crosses the batch-size abstraction boundary
