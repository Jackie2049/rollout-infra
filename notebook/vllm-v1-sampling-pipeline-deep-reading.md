# vLLM V1 Sampling Pipeline — Deep Reading

**Date**: 2025-07-15
**Source**: /tmp/vllm-fork/vllm/v1/sample/ (full directory)
**Purpose**: Understand the complete token-level data flow for GRPO rollout generation

---

## 1. Architecture Overview

The V1 sampling pipeline is a **sequential filter chain** applied to raw logits from the model's last linear layer. It is implemented as `Sampler(nn.Module)` at `/tmp/vllm-fork/vllm/v1/sample/sampler.py`, orchestrated by `GPUModelRunner._sample()` which calls `self.sampler(logits, sampling_metadata)`.

### Top-Level Call Chain

```
GPUModelRunner.execute_model()
  -> model.forward() => hidden_states => logits [num_scheduled_tokens, vocab_size]
  -> GPUModelRunner._sample(logits, sampling_metadata, spec_decode_metadata)
    -> if no spec_decode: Sampler.forward(logits, sampling_metadata)
    -> if spec_decode:    RejectionSampler.forward(metadata, draft_probs, logits, sampling_metadata)
```

For GRPO rollout (no spec decode), the path is always `Sampler.forward()`.

---

## 2. Sequential Sampling Pipeline — 9-Step Order

The `Sampler.forward()` docstring defines the exact order. Each step modifies a shared `logits` tensor (mostly in-place for efficiency).

### Step 1: Compute raw_logprobs (before any modification)

```python
# Lines 79-88 of sampler.py
if num_logprobs is not None or sampling_metadata.logprob_token_ids:
    if logprobs_mode == "raw_logprobs":
        raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
    elif logprobs_mode == "raw_logits":
        raw_logprobs = logits.to(torch.float32)  # clone or cast
```

**CRITICAL for GRPO**: This captures logprobs BEFORE penalties/temperature. This is the `old_logprobs` reference point. The V1 sampler explicitly documents this difference from V0:

> "NOTE(woosuk): Use the original logits (before any penalties or temperature scaling) for the top-k logprobs. This is different from the V0 sampler, which uses the logits that is used for sampling (after penalties and temperature scaling)."

**GRPO implication**: When verl asks vLLM for logprobs during rollout, it gets `raw_logprobs` — i.e., `log_softmax(original_logits)` before any penalty/temperature modification. This matches the GRPO algorithm requirement: `old_log_prob = log_softmax(pi_theta_old(a|s))` at the original (unmodified) logits.

### Step 2: Cast logits to float32

```python
logits = logits.to(torch.float32)  # Line 91
```

### Step 3: Apply logits processors (via `apply_logits_processors`)

This is a compound step with sub-operations in strict order:

#### 3a. Apply allowed token ids whitelist (structured output / grammar FSM)

```python
# Line 391-392
if sampling_metadata.allowed_token_ids_mask is not None:
    logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
```

**How it works**: `allowed_token_ids_mask` is a `[batch_size, vocab_size]` bool tensor. `True` means "NOT allowed" (mask to -inf). This is the grammar/structured output whitelist mechanism.

**Construction**: `InputBatch` maintains `has_allowed_token_ids` set and lazily allocates `allowed_token_ids_mask`. The mask is populated per-request via `SamplingParams.allowed_token_ids`, which grammar FSM engines (xgrammar, Outlines) fill.

**GRPO + #46118 conflict**: This is EXACTLY where the MTP+grammar FSM conflict (#46118) manifests. When grammar FSM restricts allowed tokens, but MTP spec decode proposes tokens outside the grammar, the rejection sampler may fail. For GRPO without spec decode, this is less concerning, but structured output (JSON schema enforcement) via grammar FSM directly modifies this whitelist, potentially restricting the token distribution in ways that affect GRPO advantage computation.

#### 3b. Apply bad words exclusion

```python
# Lines 394-396
if bad_words_token_ids:
    apply_bad_words(logits, bad_words_token_ids, output_token_ids)
```

Bad words are n-gram patterns matched against `past_tokens_ids`. If the prefix matches, the last token ID in the pattern is set to `-inf`. This is per-request, indexed by `dict[int, list[list[int]]]`.

#### 3c. Apply non-argmax-invariant logits processors

```python
# Lines 399-400
for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
    logits = processor.apply(logits)
```

These processors CAN change greedy sampling outcomes. Currently:

- **MinTokensLogitsProcessor**: Sets stop/EOS tokens to `-inf` until `min_tokens` count is reached. `is_argmax_invariant() = False`.
- **LogitBiasLogitsProcessor**: Adds per-token bias values. `is_argmax_invariant() = False`.

#### 3d. Apply penalties

```python
# Lines 403-404 (calls apply_penalties which delegates)
logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
```

Penalties are applied in `apply_all_penalties()` at `ops/penalties.py` which calls `apply_penalties()` from `vllm/model_executor/layers/utils.py`. Three penalty types:

- **Repetition penalty**: Multiplies logits. If `rep_pen > 1`: tokens already seen get `logit / rep_pen`, unseen tokens get `logit * rep_pen`. If `rep_pen < 1`: reverse.
- **Frequency penalty**: `logit -= freq_pen * count(token in output)`. Linear penalty per occurrence.
- **Presence penalty**: `logit -= pres_pen * (1 if token appeared else 0)`. Flat penalty per appearance.

**GRPO relevance**: Penalties modify logits AFTER raw_logprobs were captured (Step 1). This means the sampled token distribution differs from the raw logprob distribution. For GRPO, penalties are typically NOT used during rollout (they would bias the policy), but if they are, the `old_logprobs` (from Step 1) won't match the actual sampling distribution. This is a potential misalignment.

#### 3e. Thinking budget state (optional)

```python
# Lines 404-414
if holder is not None and holder.has_tracked_requests():
    holder.update_state(output_token_ids, ...)
    logits = holder.apply_to_logits(logits, ...)
```

ThinkingBudgetStateHolder forces end-of-thinking tokens when the thinking token budget is exceeded. It sets specific token logits to `1e9` (massive boost) to force the think-end token.

**GRPO relevance**: If the model uses thinking budget (Qwen3-style `<think>...</think>`), this can alter the sampling distribution in a deterministic way. For GRPO, this means the policy is NOT purely determined by the model logits — external forcing overrides the model's preference at budget boundaries.

### Step 4: Convert sampled token IDs to int64

```python
sampled = sampled.long()  # Line 104
```

FlashInfer returns int32, PyTorch argmax returns int64. This standardizes to int64 for gather operations.

### Step 5: Handle logprob_token_ids (generative_scoring API)

```python
# Lines 108-113
if sampling_metadata.logprob_token_ids:
    logprob_token_ids_tensors = self.gather_specific_token_logprobs(...)
```

This is an efficient path for the `generative_scoring` API that gathers logprobs for specific token IDs rather than the full top-k. Pads shorter lists to max length, uses `gather()` instead of full-vocab `topk()`. Not typically used in GRPO.

### Step 6: Gather top-k logprobs

```python
# Lines 115-131
if num_logprobs == -1:
    # Return full unsorted logprobs (OOM risk for large vocab)
    logprobs_tensors = LogprobsTensors(torch.empty(0), raw_logprobs, torch.empty(0))
else:
    logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
```

`gather_logprobs()`:
1. `torch.topk(logprobs, num_logprobs)` — find top-k values and indices
2. `logprobs.gather(-1, sampled.unsqueeze(-1))` — get logprob of the sampled token
3. `batched_count_greater_than(logprobs, token_logprobs)` — compute rank of sampled token
4. Concatenate sampled token logprob + top-k logprobs → final output may have `num_logprobs + 1` entries

**GRPO alignment**: The sampled token's logprob is always included (at position 0). The `rank` tells where the sampled token falls in the overall distribution. This is the data verl consumes as `old_log_probs`.

### Step 7: Convert sampled to int32, unsqueeze to [num_reqs, 1]

```python
sampled = sampled.to(torch.int32)  # Line 134
sampled_token_ids = sampled.unsqueeze(-1)  # [num_reqs, 1]
```

### Step 8: Return SamplerOutput

```python
SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=logprobs_tensors)
```

---

## 3. The `sample()` Method — Temperature + Top-K/Top-P + Random Sampling

The `sample()` method (lines 238-297) handles the actual sampling operation with a mixed greedy/random dispatch.

### 3.1 Greedy vs Random Dispatch

```python
# Lines 251-266
assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
if sampling_metadata.all_random:
    greedy_sampled = None
else:
    greedy_sampled = self.greedy_sample(logits)  # argmax
    if sampling_metadata.all_greedy:
        return greedy_sampled, processed_logprobs  # Early return
```

**Classification**:
- `all_greedy = True` when EVERY request has `temperature < 1e-5` (SamplingType.GREEDY)
- `all_random = True` when EVERY request has `temperature >= 1e-5`
- Mixed batch: both flags are False — compute greedy AND random, then select per-request

**GRPO relevance**: GRPO rollout uses `temperature > 0` (typically 0.6-1.0). So `all_random = True` during rollout. The greedy path is never taken for GRPO. However, if temperature=0 is used for comparison, the `all_greedy` early return skips all temperature/top-k/top-p processing.

### 3.2 Temperature Scaling

```python
# Lines 271-273
logits = self.apply_temperature(logits, sampling_metadata.temperature, sampling_metadata.all_random)
```

```python
# apply_temperature (lines 223-232)
if not all_random:
    temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)  # Avoid div/0 for greedy rows
return logits.div_(temp.unsqueeze(dim=1))  # In-place division
```

**Important**: Temperature is applied AFTER penalties and non-argmax-invariant processors, but BEFORE argmax-invariant processors (min_p) and top-k/top-p.

**GRPO**: `temperature > 0` divides logits, flattening the distribution. The `old_logprobs` (captured at Step 1) are BEFORE temperature scaling. The actual sampling distribution is `softmax(logits / temperature)`. This means:
- `old_log_prob = log_softmax(original_logits)` (what vLLM returns)
- `sampling_prob = softmax(logits_after_penalties / temperature)` (what vLLM actually samples from)

For GRPO, these should be aligned: `log_softmax(original_logits / temperature)` = what the model intended. If penalties are zero (typical for GRPO rollout), `raw_logprobs` = `log_softmax(logits_before_temp)` but sampling uses `softmax(logits_before_temp / temp)`. The logprob returned is `log_softmax(logits_before_temp)`, which is NOT `log_softmax(logits_before_temp / temp)`.

**WAIT — THIS IS A KEY FINDING**: vLLM V1 returns `raw_logprobs = log_softmax(logits)` (before temperature), but the actual sampling probability is `softmax(logits / temperature)`. These are DIFFERENT distributions when `temperature != 1.0`. For GRPO, this means the returned logprobs do NOT represent the actual sampling distribution at the given temperature. They represent the un-scaled (temperature=1.0) distribution.

However, this is INTENTIONAL by vLLM's design (documented in the docstring). The reason is that raw logprobs are more useful for analysis/debugging. For GRPO training frameworks like verl, they need to either:
1. Use the raw logprobs directly as `old_log_probs` (which means the ratio `exp(new_logprob - old_logprob)` implicitly accounts for the log_softmax difference)
2. Or convert: `log_softmax(logits / temp) = raw_logprobs / temp - log(Z_temp)` where `Z_temp = sum(exp(logits / temp))`

This is a potential subtle misalignment that GRPO frameworks must handle.

### 3.3 Argmax-Invariant Processors (Min-P)

```python
# Lines 277-278
for processor in sampling_metadata.logitsprocs.argmax_invariant:
    logits = processor.apply(logits)
```

**MinPLogitsProcessor** (builtin.py lines 22-115):
```python
probability_values = softmax(logits, dim=-1)
max_probabilities = amax(probability_values, dim=-1, keepdim=True)
adjusted_min_p = max_probabilities * self.min_p
invalid_token_mask = probability_values < adjusted_min_p
logits.masked_fill_(invalid_token_mask, -float("inf"))
```

Min-p filters tokens whose probability is below `min_p * max_probability`. This is applied AFTER temperature scaling, so it operates on the temperature-scaled distribution. `is_argmax_invariant() = True` because min-p doesn't change the argmax outcome.

**GRPO relevance**: If `min_p > 0` during GRPO rollout, some tokens are filtered out. The returned `raw_logprobs` (from Step 1) include these tokens, but they could never be sampled. This creates a mismatch between "possible tokens in logprobs" and "tokens that can actually be sampled". For GRPO, `min_p` should typically be 0 during rollout.

### 3.4 Top-K / Top-P Sampling

```python
# Lines 281-286
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits, sampling_metadata.generators, sampling_metadata.top_k, sampling_metadata.top_p
)
```

The `TopKTopPSampler` dispatches to different backends (see Section 4). The sequence is:
1. Apply top-k filter (keep only top k logits)
2. Apply top-p filter (keep smallest set whose cumulative probability >= top_p)
3. Convert to probabilities via `softmax`
4. Sample via Gumbel-max trick: `probs / q.argmax()` where `q.exponential_()`

**GRPO relevance**: Like min-p, top-k/top-p filtering creates a mismatch between the raw logprobs (full vocab) and the actual sampling distribution (filtered vocab). During GRPO rollout, typically `top_k = -1` (disabled) and `top_p = 1.0` (disabled), so this step is a no-op.

### 3.5 Mixed Batch Resolution

```python
# Lines 291-296
if greedy_sampled is not None:
    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,  # Reuse tensor
    )
```

For mixed batches (some greedy, some random), `torch.where` selects per-request. This is O(1) and efficient.

---

## 4. Sampling Backend Selection

### 4.1 Backend Dispatch Architecture

`TopKTopPSampler.__init__()` selects the forward method at construction time:

| Platform | Condition | Forward Method |
|----------|-----------|---------------|
| CUDA + FlashInfer supported + logprobs_mode != processed | Default | `forward_cuda` |
| CUDA + FlashInfer unsupported OR processed logprobs needed | Fallback | `forward_native` |
| CPU + x86/ARM | Default | `forward_cpu` |
| CPU + POWERPC/RISCV | Fallback | `forward_native` |
| XPU + VLLM_XPU_USE_SAMPLER_KERNEL | Optimized | `forward_xpu` |
| ROCm + aiter.ops.sampling available + logprobs_mode != processed | Optimized | `forward_hip` |
| ROCm + aiter unavailable OR processed logprobs needed | Fallback | `forward_native` |
| Other | Fallback | `forward_native` |

### 4.2 FlashInfer Backend (Primary for CUDA)

`forward_cuda()` calls `flashinfer_sample()`:

```python
# Lines 431-468
if k is None:
    # Top-p only: softmax → flashinfer.sampling.top_p_sampling_from_probs
elif p is None:
    # Top-k only: softmax → flashinfer.sampling.top_k_sampling_from_probs
else:
    # Both: flashinfer.sampling.top_k_top_p_sampling_from_logits (no softmax needed)
```

**Fallback triggers**:
1. No top-k AND no top-p → falls back to native (nothing to do)
2. Per-request generators present → FlashInfer 0.2.3+ doesn't support them → falls back
3. `logprobs_mode` requires processed logits/logprobs → FlashInfer doesn't expose them → falls back

**GRPO on RTX 4090**: FlashInfer is the primary backend for SM89 (RTX 4090). However, #48613 reports batch invariance issues with FlashInfer on Qwen3.5/3.6 GDN models. If `bs=1 vs bs=N` produces different outputs, GRPO's per-request sampling may be affected.

### 4.3 Triton Backend (Top-K/Top-P Filtering Only)

The Triton kernel at `ops/topk_topp_triton.py` implements the **Qrita algorithm** (Park et al., arXiv:2602.01518) — a pivot-based truncation and selection method.

**Dispatch logic** (`apply_top_k_top_p()` lines 322-337):
```python
if p is None and k is None:
    return logits  # No-op
if current_platform.is_cpu():
    if HAS_TRITON: return apply_top_k_top_p_triton(logits, k, p)
    else: return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
if HAS_TRITON and logits.shape[0] >= 8:
    return apply_top_k_top_p_triton(logits, k, p)  # Triton for batch >= 8
return apply_top_k_top_p_pytorch(logits, k, p)  # PyTorch for small batches
```

**The Triton kernel**:
- Uses ternary search (2 pivots per iteration, max 18 iterations) to find the exact top-k/top-p threshold
- Handles duplicate logit values correctly (counts and partially keeps/removes)
- Uses Gaussian sigma-truncation for outlier pre-filtering (percentile → sigma lookup tables)
- Operates on vocab in BLOCK_SIZE=8192 tiles (GPU) or 256 tiles (CPU)
- Uses a scratch BUFFER for outlier gathering and probability normalization
- Total: 6 passes over the data (0: stats, 1: max/min/outliers, 2: ternary search, 3: exp/sum, 4: buffer, 5: p-pivot search, 6: apply mask)

### 4.4 PyTorch Backend (Fallback)

`apply_top_k_top_p_pytorch()`:
- Sorts entire vocab descending (O(V log V))
- Applies top-k mask: values below k-th largest → -inf
- Applies top-p mask: cumulative softmax sum < (1 - p) → -inf
- Scatter back to original order

`random_sample()`:
- `softmax(logits)` → probabilities
- Gumbel-max trick: `q.exponential_()` then `probs / q.argmax()` — equivalent to `torch.multinomial` but avoids CPU-GPU sync

### 4.5 Sampling Backend for RTX 4090 GRPO

For GRPO rollout on RTX 4090:
- **Typical config**: top_k=-1, top_p=1.0 → both disabled → `apply_top_k_top_p()` returns logits unchanged → no Triton kernel needed
- **FlashInfer**: Not invoked because no top-k/top-p is set → `forward_cuda` falls back to `forward_native`
- **Native path**: `softmax(logits) → random_sample(probs, generators)` — pure PyTorch, no FlashInfer

This means GRPO rollout on RTX 4090 typically goes through the simplest possible path: softmax + Gumbel-max sampling. No Triton, no FlashInfer.

---

## 5. Logprobs Computation — Complete Data Flow

### 5.1 LogprobsMode — 4 Options

```python
LogprobsMode = Literal[
    "raw_logits",      # Return logits before any processing
    "raw_logprobs",    # Return log_softmax(logits) before processing (DEFAULT)
    "processed_logits", # Return logits after all processing
    "processed_logprobs" # Return log_softmax(logits) after all processing
]
```

Default: `"raw_logprobs"` (configured in `ModelConfig.logprobs_mode`).

### 5.2 Raw vs Processed — The Critical Difference

| Mode | What is captured | When captured | What it represents |
|------|-----------------|---------------|--------------------|
| `raw_logprobs` | `log_softmax(logits)` | Before Step 3 (all processors) | Model's original output distribution |
| `raw_logits` | `logits.clone()` | Before Step 3 | Raw model output (unnormalized) |
| `processed_logprobs` | `log_softmax(logits_after_all)` | After Step 7 (all filters) | Actual sampling distribution |
| `processed_logits` | `logits_after_all` | After Step 7 | Filtered logits (unnormalized) |

**GRPO uses `raw_logprobs`** (default). This gives `log_softmax(original_model_logits)` — the model's raw prediction, before temperature, penalties, grammar, min-p, top-k/top-p.

### 5.3 Top-K Logprobs Structure

`LogprobsTensors` is a NamedTuple with:
- `logprob_token_ids`: `[num_tokens, max_num_logprobs + 1]` int32 — token IDs
- `logprobs`: `[num_tokens, max_num_logprobs + 1]` float32 — logprob values
- `selected_token_ranks`: `[num_tokens]` — rank of sampled/prompt token
- `cu_num_generated_tokens`: `[num_reqs]` (optional) — cumulative token offsets

**Layout**: Column 0 = sampled/prompt token. Columns 1..num_logprobs = top-k tokens. If sampled token is within top-k, it appears both at column 0 and in the top-k portion, and is merged by `LogprobsProcessor` downstream.

### 5.4 Prompt Logprobs vs Output Logprobs

**Prompt logprobs** (`prompt_logprobs`):
- Computed during PREFILL phase
- Separate path: `PromptLogprobsWorker.compute_prompt_logprobs()` at `/tmp/vllm-fork/vllm/v1/worker/gpu/sample/prompt_logprob.py`
- Uses chunked computation (CHUNK_SIZE=1024) to avoid OOM on long prompts
- For each prompt position, computes `log_softmax(logits)` and gathers top-k
- Accumulated across chunked prefill steps via `in_progress_prompt_logprobs[req_id]`
- Returned once at end of prefill via `pop_prompt_logprobs()`

**Output logprobs** (`logprobs` / sample logprobs):
- Computed during DECODE phase — every token generation step
- Direct path through `Sampler.forward()` → `gather_logprobs()`
- Per-step, per-request: `[num_reqs, max_num_logprobs + 1]`
- Accumulated in `LogprobsProcessor._update_sample_logprobs()`

**GRPO uses output logprobs exclusively** for the `old_log_prob` reference. Prompt logprobs are not needed for GRPO training.

### 5.5 Token Rank Computation

```python
# logprobs.py
@torch.compile(backend=current_platform.simple_compile_backend)
def batched_count_greater_than(x, values):
    return (x >= values).sum(-1)
```

This counts how many tokens have logprob >= the sampled token's logprob, giving the rank. `mark_unbacked` prevents Dynamo from specializing on batch_size=1 vs >1 (avoids recompilation).

**GRPO relevance**: Token rank is metadata, not directly used in GRPO loss computation. But it's useful for debugging (checking if sampled tokens are in top-k).

### 5.6 Detokenization and UTF-8 Correction

`LogprobsProcessor._verify_tokens()` handles byte-fallback tokenization where multi-byte UTF-8 characters are split across tokens. Uses preceding sampled tokens as context for `_correct_decoded_token()`. This is purely cosmetic for display; GRPO operates on token IDs and logprob values, not decoded strings.

---

## 6. Batched Sampling — How Multiple Requests Are Processed Together

### 6.1 Per-Request Parameter Tensors

All sampling parameters are batched as GPU tensors:
- `temperature`: `[num_reqs]` float32 — per-request temperature
- `top_p`: `[num_reqs]` float32 — per-request top-p
- `top_k`: `[num_reqs]` int32 — per-request top-k
- `presence_penalties`, `frequency_penalties`, `repetition_penalties`: `[num_reqs]` float32
- `allowed_token_ids_mask`: `[num_reqs, vocab_size]` bool — per-request grammar whitelist
- `generators`: `dict[int, torch.Generator]` — per-request RNG state (for SamplingType.RANDOM_SEED)

### 6.2 Mixed Batch Handling

When the batch contains both greedy and random requests:
- `all_greedy = False`, `all_random = False`
- Greedy samples are computed for ALL requests (even random ones, since argmax is cheap)
- Random samples are computed for ALL requests (even greedy ones, but they're ignored)
- Final selection via `torch.where(temperature < eps, greedy, random)` — per-request dispatch

This is efficient because:
1. `argmax` is always computed (needed for logprob rank computation anyway)
2. Temperature scaling applies uniformly — greedy rows get temp replaced with 1.0
3. Top-k/top-p applies uniformly — greedy rows typically have top_k=vocab_size, top_p=1.0

### 6.3 Speculative Decode Sampling (RejectionSampler)

When `spec_decode_metadata` is present, the sampling path switches to `RejectionSampler.forward()`:

1. Sample bonus token: `self.sampler(bonus_logits, ...)` with `logprobs_mode_override="raw_logits"`
2. Apply logits processors to target logits (with repeat_indices for per-request expansion)
3. Apply sampling constraints (temperature + top-k/top-p) to target logits
4. Rejection sampling: compare draft vs target probabilities, accept/reject/recover
5. Logprob computation: merge bonus + accepted target logits

For GRPO without spec decode, this path is irrelevant. But it's worth noting that spec decode has its own logits processor path (`RejectionSampler.apply_logits_processors`) that handles penalty expansion differently (repeat_indices for per-token rows).

---

## 7. GRPO-Specific Analysis

### 7.1 Complete Token-Level Data Flow for GRPO Rollout

```
1. Model forward: hidden_states → logits [1, vocab_size]  (dp=1, single request)
2. Sampler.forward():
   a. raw_logprobs = log_softmax(logits)  ← THIS IS old_log_probs
   b. logits.to(float32)
   c. allowed_token_ids_mask → grammar whitelist (if structured output)
   d. bad_words (typically none for GRPO)
   e. MinTokens (typically none for GRPO)
   f. LogitBias (typically none for GRPO)
   g. Penalties (typically none for GRPO rollout)
   h. ThinkingBudget (if thinking model, may force think-end)
   i. Temperature: logits /= temperature (e.g., 0.7)
   j. Min-P (typically 0 for GRPO)
   k. Top-K/Top-P (typically disabled for GRPO)
   l. softmax(logits_after_temp) → probs
   m. Gumbel-max: probs / exponential_rand.argmax → sampled_token_id
   n. Gather: raw_logprobs[sampled_token_id] + topk → LogprobsTensors
3. Return: sampled_token_id + raw_logprobs → verl consumes
```

### 7.2 old_logprobs Alignment — The Temperature Problem

**The fundamental question**: Does vLLM's `raw_logprobs = log_softmax(logits_before_temp)` match what verl needs for GRPO's `old_log_prob = log_softmax(pi_old(a|s))`?

**Answer**: NO, if temperature != 1.0.

- vLLM returns: `log_softmax(logits)` = `log_softmax(logits / 1.0)` — the temperature=1.0 distribution
- The actual sampling distribution: `softmax(logits / temperature)` — the temperature-scaled distribution
- GRPO needs: `log(pi_old(a|s))` where `pi_old` is the policy that actually generated the token

The correct `old_log_prob` for GRPO should be `log_softmax(logits / temperature)`, not `log_softmax(logits)`.

**However**: If both old and new policies use the SAME temperature during rollout, then:
```
ratio = exp(new_logprob - old_logprob)
     = exp(log_softmax(new_logits/temp) - log_softmax(old_logits/temp))
```

If both use the same temp, this simplifies to comparing the raw logits (temp cancels out in the ratio). The PPO/GRPO ratio computation is invariant to constant temperature scaling as long as both old and new use the same temperature.

**Practical resolution**: verl's GRPO implementation likely uses `raw_logprobs` directly. Since temperature is the same for all requests in a group, the ratio `exp(new_raw - old_raw)` is equivalent to `exp(new_scaled - old_scaled)` up to a constant that cancels in the advantage normalization. So using `raw_logprobs` is mathematically valid for GRPO as long as temperature is consistent within the group.

### 7.3 Temperature=0 vs Temperature>0 Impact

- **Temperature=0 (greedy)**: `argmax(logits)` — deterministic, no randomness. GRPO CANNOT work with temperature=0 because there's no stochastic policy to compute ratios.
- **Temperature>0**: `softmax(logits/temp)` — stochastic. Higher temperature → flatter distribution → more exploration. GRPO requires temperature>0 to generate diverse completions for advantage estimation.

**vLLM implementation**: If temperature=0, `all_greedy=True`, the sampler early-returns at Step 7b with only argmax. No top-k/top-p processing. The returned `raw_logprobs` still contain the full distribution (for debugging), but the sampled token is always argmax.

### 7.4 Structured Output (Grammar FSM) + GRPO — The #46118 Conflict

**The problem**: Grammar FSM restricts `allowed_token_ids` to a subset (e.g., only tokens that produce valid JSON). This happens at Step 3a BEFORE penalties and temperature.

**For GRPO without grammar**: No conflict. The full vocab is available.

**For GRPO with grammar (JSON schema enforcement)**:
1. The `allowed_token_ids_mask` filters logits to -inf for grammar-invalid tokens
2. `raw_logprobs` (Step 1) are computed BEFORE grammar filtering — they include invalid tokens
3. The actual sampling distribution (after grammar) excludes invalid tokens
4. The sampled token's logprob in `raw_logprobs` is correct (it's from the unfiltered distribution)
5. But if GRPO tries to compute `ratio = exp(new_raw - old_raw)`, both distributions are filtered by grammar during rollout, so the grammar restriction cancels out

**However**: If grammar FSM has a bug (as in #46118 — MTP conflict, 58% request failure rate), the grammar mask may be incorrect, leading to:
- Tokens that should be allowed are blocked → generation fails or produces garbage
- Tokens that shouldn't be allowed pass through → invalid outputs

**Recommendation for GRPO**: Do NOT use grammar/structured output during GRPO rollout. Generate freely, then validate outputs post-generation. Grammar FSM adds complexity and potential bugs to the sampling path without benefit for training.

### 7.5 Logprobs Memory Budget for GRPO on RTX 4090

For a typical GRPO rollout with `num_logprobs=20`:
- `logprob_token_ids`: `[num_reqs, 21]` int32 = 84 bytes per request
- `logprobs`: `[num_reqs, 21]` float32 = 84 bytes per request
- `selected_token_ranks`: `[num_reqs]` int32 = 4 bytes per request
- Total per request: ~172 bytes — negligible

For `num_logprobs=-1` (full vocab):
- `logprobs`: `[num_reqs, vocab_size]` float32 — for 152K vocab (Qwen3) = ~600 KB per request
- This can cause OOM with many concurrent requests — avoid in GRPO

---

## 8. SamplingMetadata — Construction and Lifecycle

### 8.1 InputBatch._make_sampling_metadata()

Located at `/tmp/vllm-fork/vllm/v1/worker/gpu_input_batch.py` lines 831-934.

Key decisions:
- `temperature`: Only copied to GPU if `not all_greedy` (all greedy = no temperature needed)
- `top_p/top_k`: Only copied if `not no_top_p/not no_top_k`
- `penalties`: Only copied if `not no_penalties`
- `prompt_token_ids`: Only built if penalties or logits processors need them
- `output_token_ids`: Only set if penalties, bad_words, logits processors, or thinking budget need them
- `allowed_token_ids_mask`: Only allocated if any request has grammar/structured output

**Lazy allocation**: `allowed_token_ids_mask` and its CPU twin are lazily allocated (only when a request first uses `allowed_token_ids`). This avoids allocating `[max_num_reqs, vocab_size]` bool tensors (~6 GB for 152K vocab) unless needed.

### 8.2 Batch Update Lifecycle

`SamplingMetadata` is reconstructed whenever the batch changes (new requests, removed requests, moved requests). The `BatchUpdateBuilder` tracks changes, generates `BatchUpdate`, and feeds it to logits processors' `update_state()` methods.

For GRPO with a stable batch of rollout requests, `SamplingMetadata` is relatively stable — reconstructed only when new rollout requests are added or completed ones are removed.

---

## 9. Key Findings and GRPO Implications Summary

### 9.1 Architecture Findings

1. **9-step sequential pipeline**: Whitelist → bad_words → non-argmax-invariant processors → penalties → thinking budget → temperature → argmax-invariant processors → top-k/top-p → sample
2. **Raw logprobs captured BEFORE all modifications**: vLLM V1 intentionally returns `log_softmax(logits)` before temperature/penalties/grammar. This differs from V0.
3. **Mixed greedy/random dispatch via torch.where**: Efficient per-request selection in mixed batches.
4. **FlashInfer primary backend for CUDA**: Falls back to native when top-k/top-p are disabled (common for GRPO).
5. **Triton Qrita kernel for top-k/top-p filtering**: 6-pass pivot-based algorithm, used only when batch >= 8.
6. **Grammar FSM via allowed_token_ids_mask**: Boolean mask tensor, applied at Step 3a, before penalties.

### 9.2 GRPO-Specific Implications

1. **Temperature misalignment in raw_logprobs**: vLLM returns `log_softmax(logits)` (temp=1.0), but sampling uses `softmax(logits/temp)`. Mathematically valid for GRPO ratio computation as long as temperature is consistent within the group, because the ratio cancels the constant.

2. **GRPO rollout typically uses the simplest sampling path**: No grammar, no penalties, no min-p, no top-k/top-p. The path is: logits → temperature → softmax → Gumbel-max → sample. No Triton kernel, no FlashInfer invocation.

3. **Avoid grammar FSM during GRPO rollout**: #46118 shows grammar FSM bugs cause 58% failure rates. Grammar adds complexity without training benefit. Validate outputs post-generation instead.

4. **ThinkingBudget forcing**: If the model uses thinking tokens with a budget, the sampler may force `think_end` at budget boundaries. This is external policy override, not purely model-driven. For GRPO, this means the policy distribution is modified at budget boundaries, potentially creating artifacts in the advantage computation.

5. **Per-request generators (SamplingType.RANDOM_SEED)**: FlashInfer doesn't support per-request generators. If GRPO uses seeded generation for reproducibility, the sampler falls back to the native path. This is slower but correct.

6. **FlashInfer batch invariance (#48613)**: bs=1 vs bs=N may produce different outputs with GDN models. For GRPO with per-request rollout (bs=1), this could cause subtle differences compared to batched rollout.

7. **Logprobs memory**: With `num_logprobs=20`, memory overhead is negligible (~172 bytes/request). With `num_logprobs=-1`, full vocab logprobs can cause OOM — avoid for GRPO on RTX 4090.

---

## 10. File Index

| File | Path | Purpose |
|------|------|---------|
| Sampler | `/tmp/vllm-fork/vllm/v1/sample/sampler.py` | Top-level orchestration, 9-step pipeline |
| SamplingMetadata | `/tmp/vllm-fork/vllm/v1/sample/metadata.py` | Per-step sampling parameters dataclass |
| TopKTopPSampler | `/tmp/vllm-fork/vllm/v1/sample/ops/topk_topp_sampler.py` | Backend dispatch (FlashInfer/Triton/Native/XPU/ROCm) |
| Triton topk_topp | `/tmp/vllm-fork/vllm/v1/sample/ops/topk_topp_triton.py` | Qrita pivot-based top-k/top-p kernel |
| Penalties | `/tmp/vllm-fork/vllm/v1/sample/ops/penalties.py` | Repetition/frequency/presence penalties |
| Bad words | `/tmp/vllm-fork/vllm/v1/sample/ops/bad_words.py` | N-gram bad word exclusion |
| Logprobs ops | `/tmp/vllm-fork/vllm/v1/sample/ops/logprobs.py` | batched_count_greater_than (rank computation) |
| RejectionSampler | `/tmp/vllm-fork/vllm/v1/sample/rejection_sampler.py` | Spec decode rejection sampling |
| ThinkingBudget | `/tmp/vllm-fork/vllm/v1/sample/thinking_budget_state.py` | Thinking token budget forcing |
| LogitsProcessor interface | `/tmp/vllm-fork/vllm/v1/sample/logits_processor/interface.py` | ABC for logits processors |
| LogitsProcessor builtin | `/tmp/vllm-fork/vllm/v1/sample/logits_processor/builtin.py` | MinP, LogitBias, MinTokens processors |
| LogitsProcessor state | `/tmp/vllm-fork/vllm/v1/sample/logits_processor/__init__.py` | LogitsProcessors container, BatchUpdateBuilder |
| InputBatch | `/tmp/vllm-fork/vllm/v1/worker/gpu_input_batch.py` | SamplingMetadata construction, per-request state |
| GPUModelRunner | `/tmp/vllm-fork/vllm/v1/worker/gpu_model_runner.py` | _sample() dispatch, sampler invocation |
| LogprobsProcessor | `/tmp/vllm-fork/vllm/v1/engine/logprobs.py` | Output-side logprob processing, UTF-8 correction |
| PromptLogprobsWorker | `/tmp/vllm-fork/vllm/v1/worker/gpu/sample/prompt_logprob.py` | Chunked prompt logprobs computation |
| ModelConfig | `/tmp/vllm-fork/vllm/config/model.py` | LogprobsMode definition (4 modes) |
| Outputs | `/tmp/vllm-fork/vllm/v1/outputs.py` | LogprobsTensors, SamplerOutput data structures |
