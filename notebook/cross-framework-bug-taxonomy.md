# Cross-Framework Bug Pattern Taxonomy

A systematic classification of recurring bug patterns across 7 ML training/inference frameworks,
built from 3+ weeks of daily monitoring (June 19 - July 14, 2026).

## Pattern Categories

| ID | Pattern Name | Frameworks Affected | Occurrences |
|----|-------------|-------------------|-------------|
| P1 | `cuda_stream_use_after_free` | DeepSpeed, Megatron, vLLM | 4+ |
| P2 | `incorrect_gradient_normalization_scope` | Megatron, verl, TRL | 4+ |
| P3 | `state_lifecycle_mismatch` | vLLM, SGLang, vLLM-Ascend | 6+ |
| P4 | `dsv4_instability` | vLLM, SGLang, Megatron, vLLM-Ascend | 13+ |
| P5 | `muon_clipping_interaction` | Megatron, DeepSpeed | 2+ |
| P6 | `silent_corruption` | Multiple | 4+ |
| P7 | `weight_reload_cache_staleness` | vLLM, SGLang | 3+ |

---

## P1: CUDA Stream Use-After-Free (`cuda_stream_use_after_free`)

**Root cause**: In overlapped/async parameter or gradient communication, storage/buffer
is released on one CUDA stream while another stream may still be reading from it.
The caching allocator recycles the memory → consumer reads garbage.

### Members

| # | Framework | Bug ID | Symptom | Root Cause | Resolution |
|---|-----------|--------|---------|------------|------------|
| 1 | DeepSpeed | #8061 | overlap_comm+torch.compile=NaN | IPG bucket copy_ on multiple streams, average_tensor() only waits current_stream | #8080: wait ALL producer streams per IPG bucket |
| 2 | Megatron | #5788 | Intermittent numerical corruption, overlapped param gather | StorageResizeBasedBucketAllocator.free() no record_stream | OPEN: add record_stream(current_stream) |
| 3 | vLLM | #45552 | CuMemAllocator sleep/wake CUDART crash | In-flight kernels race cuMemUnmap + cudaMemcpy | OPEN: add cuda.synchronize() before unmap |
| 4 | vLLM | #46125 | Stale KV cache after weight update | Cache kernels race with new weight state | Reverted #45093 |

### Fix Pattern
**Before freeing a buffer that was written on a different stream, synchronize
or record_stream on the consuming stream**.

- DeepSpeed fix (#8080, +96/-1): Wait on ALL recorded copy_streams per IPG bucket
  before `average_tensor()`, not just `current_stream`.
- Megatron fix (proposed, +1 line): `storage.record_stream(torch.cuda.current_stream())`
  before `_free_storage()`.
- vLLM fix (proposed): `torch.cuda.synchronize()` before `cuMemUnmap`.

### Detection
- `compute-sanitizer` can surface cross-stream violations
- Intermittent: depends on exact kernel timing and allocator state
- `torch.compile` triggers because it changes kernel launch ordering

---

## P2: Incorrect Gradient Normalization Scope (`incorrect_gradient_normalization_scope`)

**Root cause**: Loss or gradient normalization computed over a local/per-microbatch scope
instead of the global/training-step scope. Causes MBS-dependent gradient scaling or
gradient bias with variable-length sequences.

### Members

| # | Framework | Bug ID | Symptom | Root Cause |
|---|-----------|--------|---------|------------|
| 1 | Megatron | #4590 | 158% gradient bias, variable-length completions | calculate_per_token_loss=False → local_mean over microbatch, then DP/CP average |
| 2 | Megatron | #5798 | 1/MBS spurious scaling on seq_aux_loss gradient | local_num_tokens = seq_length (ignores bsz) under calculate_per_token_loss=True |
| 3 | verl | #6836 (MERGED) | MoE aux/z-loss grad blowup at CP>1 | calculate_per_token_loss with CP creates per-CP-chunk normalization |
| 4 | TRL | P7-2 | top_n_sigma clipping needs group-level normalization | GRPO advantage normalized per-group, but clipping applied per-token |

### Fix Pattern
**All loss terms (main + auxiliary) must use global-sum/global-sum normalization,
regardless of MBS, CP, or DP configuration.**

### Key Insight
The `calculate_per_token_loss` flag is a DUAL bug:
- **False**: main loss normalization broken (gradient bias) — #4590
- **True**: auxiliary losses still broken (MBS-dependent scaling) — #5798

---

## P3: State Lifecycle Mismatch (`state_lifecycle_mismatch`)

**Root cause**: GPU-resident state (cache, constant tensors, random state) is not
properly invalidated or reinitialized at lifecycle boundaries (weight reload,
sleep/wake, device transfer).

### Members

| # | Framework | Bug ID | Symptom |
|---|-----------|--------|---------|
| 1 | vLLM-Ascend | #10684 | DSA Hadamard ALL-ZERO after sleep/wake — class variable lost during state transfer |
| 2 | SGLang | #28679 | GDN intermittent decode degeneracy — worsens over uptime, clears on restart |
| 3 | SGLang | #28676 | MXFP8 MoE shuffle cache CLOBBERED at weight-reload boundary |
| 4 | vLLM | #44483/#44395 | wake_up(tags=["weights"]) + forward → illegal memory access (KV cache still asleep) |
| 5 | SGLang | #28608 | RolloutKV prefix KV pinning — TTL eviction races with RL training |
| 6 | vLLM | #46118 | MTP+grammar FSM conflict — 58% request failure |

### Fix Pattern
**At every state lifecycle boundary (weight reload, sleep/wake, device transfer),
ALL GPU-resident state must be explicitly invalidated or reinitialized.**

---

## P4: DSV4 Instability (`dsv4_instability`)

**Root cause**: DeepSeek V4 (DSV4) hybrid architecture (MLA + MoE + MTP) creates
complex interactions that fail across frameworks. 13+ distinct failure modes.

### Failure Modes by Framework

| # | Framework | Failure | Root Cause |
|---|-----------|---------|------------|
| 1 | vLLM | MLA/DSV4 crash | Various |
| 2 | SGLang | MXFP8 MoE cache clobbered (#28676) | Cache not invalidated |
| 3 | Megatron | apply_rope_fusion NaN (#5317) | Triton in-place kernel bypasses autograd |
| 4 | vLLM-Ascend | DSV4 chat fix (#10645) | CANN-specific |
| 5 | SGLang | GLM-5.2 FP8 wrong on MI350X (#28685) | aiter gemm incorrect |

### Universal Protection
- `enforce_eager=True` MANDATORY for DSV4
- Any GPU-resident cache MUST be invalidated at weight-reload boundary
- Test gap: existing tests use `.detach()` → verify arithmetic only, NOT autograd

---

## P5: Muon Clipping Interaction (`muon_clipping_interaction`)

**Root cause**: Global gradient norm clipping interacts incorrectly with Muon optimizer,
because Muon's Newton-style preconditioning produces gradients with different magnitude
scales than standard optimizers.

### Members

| # | Framework | Bug ID | Symptom |
|---|-----------|--------|---------|
| 1 | Megatron | #5394 | ChainedOptimizer Muon clipping stalls — AdamW ALSO stalls (optimizer-agnostic) |
| 2 | DeepSpeed | #8068 | gradient_clipping default=0, must be set to 1.0 for GRPO |
| 3 | DeepSpeed | #8141 (NEW) | Muon ZeRO-1/2 reduce_scatter — would unlock Muon+ZeRO-2 on RTX 4090 |

### Fix Pattern
- Skip global clipping for Muon via `skip_grad_norm_clip` attribute
- #5395 fix: +15/-1 lines, adds skip_grad_norm_clip flag to optimizer containers

---

## P6: Silent Corruption (`silent_corruption`)

**Root cause**: No error signal when corruption occurs — model produces wrong outputs
but training continues, making detection extremely difficult.

### Members

| # | Framework | Pattern | Risk |
|---|-----------|---------|------|
| 1 | DeepSpeed #8061 | Overlap_comm NaN | Intermittent NaN that may not crash immediately |
| 2 | Megatron #5788 | Storage use-after-free | Numerical corruption, no error |
| 3 | SGLang #28679 | GDN degeneracy | Worsens over uptime, no error |
| 4 | vLLM #46125 | Stale KV cache | Wrong logprobs, no error |

### Defense Stack (4 layers)
1. **Gradient NaN/Inf guard** (verl #6, TRL built-in)
2. **Forward-pass NaN detection** (PyTorch #187653 NanDetectMode, 500,000× faster than detect_anomaly)
3. **Periodic validation loss** (compare against held-out set, detect divergence)
4. **compute-sanitizer** for pre-production validation

---

## P7: Weight Reload Cache Staleness (`weight_reload_cache_staleness`)

**Root cause**: When model weights are updated (GRPO training step), GPU-resident
caches (KV cache, MoE shuffle cache, constant tensors) contain stale values from
old weights.

### Members

| # | Framework | Bug | Cache Type |
|---|-----------|-----|------------|
| 1 | vLLM | #46125 | KV cache stale after weight update |
| 2 | SGLang | #28676 | MoE shuffle cache clobbered |
| 3 | SGLang | #28679 | GDN state degrades over uptime |

### Fix Pattern
- Full cache invalidation at every weight-reload boundary
- For GRPO: reinitialize all GPU-resident state at each training step

---

## Cross-Cutting Concerns

### RTX 4090 Relevance
| Pattern | RTX 4090 Impact | Priority |
|---------|----------------|----------|
| P1 `cuda_stream_use_after_free` | Weight sync between rollout/training engines uses multiple streams | HIGH |
| P2 `incorrect_gradient_normalization_scope` | Single GPU → no DP/CP, but variable-length completions always matter | MEDIUM |
| P3 `state_lifecycle_mismatch` | Sleep/wake is primary RTX 4090 GRPO mechanism (HYBRID mode) | CRITICAL |
| P4 `dsv4_instability` | DSV4 not primary target for RTX 4090 (SM89) | LOW |
| P5 `muon_clipping_interaction` | Muon is potential RTX 4090 optimizer | MEDIUM |
| P6 `silent_corruption` | Long-running GRPO most vulnerable | CRITICAL |
| P7 `weight_reload_cache_staleness` | Every GRPO step reloads weights | CRITICAL |

### Monitoring Strategy
- **Daily**: Check for new P1, P2, P3, P6 occurrences (new bugs likely follow known patterns)
- **Weekly**: Full 7-framework scan for new pattern emergence
- **On merge**: Track fix patterns — validate our analysis against actual resolved bugs

### Contribution Strategy
- **Tier 1 (UNIQUE)**: Cross-framework comments linking known bugs as same pattern family
  - P1: #5788 ↔ #8061 ↔ #45552 (draft ready, awaiting authorization)
  - P2: #5798 ↔ #4590 ↔ #6836 (draft ready, awaiting authorization)
- **Tier 2 (SMALL FIX)**: One-line fixes, config changes
  - P5: skip_grad_norm_clip for emerging optimizers
- **Tier 3 (FEATURE)**: New tools and detection mechanisms
  - P6: NanDetectMode portable version (grpo_nan_detective.py)
