# SGLang Speculative Decoding Source Code Analysis

> Date: 2026-06-04 | ~7500 lines across 12 core files
> Codebase: sglang/python/sglang/srt/speculative/ + models/llama_eagle*.py

---

## 1. Architecture Overview

### 1.1 Speculative Decoding Algorithm Registry

SGLang uses a dual-path dispatch system for speculative algorithms:

- **Builtin algorithms** are defined as enum members in `SpeculativeAlgorithm` (in `spec_info.py`): `EAGLE`, `EAGLE3`, `DFLASH`, `STANDALONE`, `NGRAM`, `FROZEN_KV_MTP`, `NONE`.
- **Plugin algorithms** can be registered via `SpeculativeAlgorithm.register("NAME", supports_overlap=True)` decorator, backed by `spec_registry.py`'s `_REGISTRY` dict. Reserved names (`EAGLE`, `DFLASH`, etc.) cannot be re-registered.

Both builtin and plugin algorithms expose a uniform `is_*()` / `create_worker()` interface. The `create_worker` method on `SpeculativeAlgorithm` returns the appropriate worker class based on both the algorithm AND whether overlap scheduling is enabled:

```
EAGLE + overlap=True  -> EAGLEWorkerV2 (spec_v2)
EAGLE + overlap=False -> EAGLEWorker   (spec_v1)
DFLASH                -> DFlashWorker   (spec_v1 only)
NGRAM                 -> NGRAMWorker    (spec_v1 only)
STANDALONE + overlap  -> StandaloneWorkerV2
```

### 1.2 Spec Worker Architecture

`BaseSpecWorker` (base_spec_worker.py) defines two abstract properties:
- `target_worker` -> `TpModelWorker` (the main model)
- `draft_worker` -> `BaseDraftWorker` (has `draft()` and `draft_extend()` methods)

The hook `on_verify_complete_cpu(num_correct_drafts_per_req)` feeds acceptance data back to the adaptive controller without forcing GPU-to-CPU sync in the hot path.

### 1.3 Data Flow Through a Decode Iteration

The spec decode decode path follows a 3-phase pipeline:

```
Phase 1: DRAFT      - Draft model generates tree of candidate tokens
Phase 2: VERIFY     - Target model verifies all candidates in one forward pass
Phase 3: DRAFT_EXTEND - Draft model catches up KV cache for accepted tokens
```

**Spec v1 flow** (`EAGLEWorker.forward_batch_generation`):
1. `draft(batch)` -> produces `EagleVerifyInput` (tree of draft tokens + mask)
2. `verify(batch)` -> target model forward + sampling -> `EagleVerifyOutput`
3. `forward_draft_extend_after_decode(batch)` -> catch up draft KV cache

**Spec v2 flow** (`EAGLEWorkerV2.forward_batch_generation`):
- Same 3 phases but the `verify` and `draft_extend` are interleaved via the overlap scheduler. `draft_worker` is a separate `EagleDraftWorker` (not the worker itself running the draft model). The v2 path uses a `plan_stream` for pipelining verify preparation with draft output computation.

---

## 2. EAGLE v1 vs v2

### 2.1 EAGLE v1 (`EAGLEWorker`, ~1310 lines)

`EAGLEWorker` **extends** `TpModelWorker` -- the worker IS the draft model. It runs both draft and target models:
- `self.model_runner` = draft model runner
- `self.target_worker` = separate target `TpModelWorker`

Key characteristics:
- Shares `req_to_token_pool` and `token_to_kv_pool_allocator` with the target worker
- Draft model uses the target's embedding and lm_head (set via `set_embed_and_head`)
- Supports hot token mapping for vocab reduction (`hot_token_id`)
- Separate `draft_attn_backend` and `draft_extend_attn_backend` for multi-step attention
- CUDA graphs for both draft decode (`EAGLEDraftCudaGraphRunner`) and draft extend (`EAGLEDraftExtendCudaGraphRunner`)

### 2.2 EAGLE v2 (`EAGLEWorkerV2`, ~1472 lines)

`EAGLEWorkerV2` **composes** draft and target workers:
- `self._draft_worker` = separate `EagleDraftWorker` (a `BaseDraftWorker`)
- `self._target_worker` = target `TpModelWorker`

Key differences from v1:
1. **Separation of concerns**: `EagleDraftWorker` is a standalone draft worker with its own model runner (`draft_runner`), attention backends, and CUDA graph runners
2. **Overlap scheduling**: Uses `plan_stream` (separate CUDA stream) to pipeline verify preparation with draft output, and draft extend preparation with verify output
3. **topk=1 fast path**: Pre-allocates `_topk1_parents_prealloc` and `_topk1_score_indices_prealloc` -- when topk=1, the tree degenerates to a chain and parent_list/score_indices become runtime-invariant constants, avoiding expensive `organize_draft_results` sorting
4. **KV cache management**: `move_accepted_tokens_to_target_kvcache` physically copies draft KV entries to target KV slots for tree paths (topk > 1); `_compact_accepted_to_front` rearranges the predict/hidden_states tensor

### 2.3 Forward Mode States

```
DECODE           -> normal decode (non-spec)
TARGET_VERIFY    -> target model verifying draft tokens (batched prefill-like)
DRAFT_EXTEND     -> draft model filling KV cache after v1 verify
DRAFT_EXTEND_V2  -> draft model filling KV cache after v2 verify
IDLE             -> no active requests
```

---

## 3. Draft Tree Construction

### 3.1 Multi-Step Autoregressive Generation

The core draft loop is in `draft_forward` (both v1 and v2 share the same logic):

```
for i in range(speculative_num_steps):
    1. select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, topk)
       -> input_ids, hidden_states, scores, (score, token, parent)
    2. Append to score_list, token_list, parents_list
    3. Skip last step (no forward needed)
    4. Run draft model forward with per-step ForwardContext
    5. Softmax -> topk sampling -> get next topk_p, topk_index, hidden_states
    6. Increment positions by 1
```

At each step, the draft model generates `topk` candidate tokens per request. With topk=5 and 5 steps, this produces a tree of up to 5^5 = 3125 candidate paths.

### 3.2 Tree Flattening and Selection

After the multi-step loop, `organize_draft_results` (eagle_utils.py):
1. Concatenates per-step scores into `[bs, topk * num_steps]`
2. Selects top `num_draft_tokens - 1` by score across all steps
3. Sorts the selected indices (for tree mask construction)
4. Gathers the corresponding draft tokens

The final draft tree has `num_draft_tokens` nodes (= `bonus_token` + `num_draft_tokens - 1` selected tokens).

### 3.3 Tree Mask Construction (`build_tree_kernel_efficient`)

This is a CUDA/Triton kernel (`sgl_kernel.build_tree_kernel_efficient`) that constructs:
- **tree_mask**: boolean attention mask -- each draft token can only attend to its ancestor path in the tree
- **positions**: absolute position IDs for each draft token
- **retrieve_index** `[bs, num_verify_tokens]`: maps each tree node to its position in the flat batch
- **retrieve_next_token** `[bs, num_verify_tokens]`: index of the first child (for tree traversal during verify)
- **retrieve_next_sibling** `[bs, num_verify_tokens]`: index of the next sibling (for BFS traversal)

Three mask modes supported:
- `FULL_MASK`: full `[q_len, kv_len]` boolean mask (default, highest memory)
- `QLEN_ONLY`: only query-length portion of the mask
- `QLEN_ONLY_BITPACKING`: bit-packed query-length mask (most compact)

The tree mask ensures that during target verification, each draft position only attends to its causal ancestors in the draft tree.

---

## 4. Verification Process

### 4.1 Verify Forward Pass

The target model processes ALL draft tokens in a single forward pass (batch prefill). The verify input (`EagleVerifyInput`) provides:
- `draft_token`: flat tensor of all draft tokens
- `custom_mask`: tree attention mask
- `positions`: position IDs
- `retrieve_index/next_token/next_sibling`: tree traversal structures

The target forward runs in `TARGET_VERIFY` mode with `CaptureHiddenMode.FULL` to capture hidden states for the next draft iteration.

### 4.2 Sampling and Rejection

Two verification paths exist:

**Greedy verification** (`verify_tree_greedy_func`, sgl_kernel):
- Compare target argmax with draft candidates along tree paths
- Accept matching tokens, stop at first mismatch
- Append target's prediction as bonus token

**Stochastic verification** (`tree_speculative_sampling_target_only`, sgl_kernel):
- Compute target probabilities: `softmax(logits / temperature)` -> top_k_renorm -> top_p_renorm
- Rejection sampling: accept draft token if `uniform_random < target_prob[token]`
- If rejected, sample from `max(0, target_prob - draft_prob)` (rescaled)
- Uses `threshold_single` and `threshold_acc` acceptance thresholds
- TP sync: broadcast predict/accept_index from rank 0 (float non-determinism)

### 4.3 Post-Verify KV Cache Management

After verification, unaccepted draft tokens' KV cache slots must be freed:

- **page_size=1**: Simple free via boolean mask indexing
- **topk=1 + page_size>1**: Align evict mask to page boundaries
- **topk>1 + page_size>1**: Shift accepted tokens to contiguous front, copy KV cache entries via `move_kv_cache`, free the tail

The v2 path additionally moves accepted tokens to the target KV cache via `move_accepted_tokens_to_target_kvcache`, using a Triton kernel (`assign_extend_cache_locs`) to compute target locations and `fill_accepted_out_cache_loc` to gather source locations.

### 4.4 Structured Output (Grammar) Support

When grammar constraints are active, `generate_token_bitmask` computes a logit mask for each draft position based on the grammar FSM state. The mask is generated on CPU (overlapped with the target forward), then applied to logits before sampling. Each accepted token advances the grammar FSM.

---

## 5. N-gram Speculation

### 5.1 Zero-Overhead Draft Generation

`NGRAMWorker` is fundamentally different from EAGLE -- it requires NO draft model. Instead, it uses a C++ `NgramCorpus` for token matching:

1. **Corpus**: A trie-based data structure built from request token streams. Supports both online insertion (from ongoing requests) and external corpus loading.
2. **Lookup**: `batch_get(req_ids, batch_tokens, total_lens)` performs BFS on the trie with configurable breadth (`min_bfs_breadth` to `max_bfs_breadth`).
3. **Match types**: Configurable via `speculative_ngram_match_type` (exact or fuzzy matching).
4. **External corpus**: Can load pre-tokenized text for domain-specific n-gram matching.

### 5.2 Key Design Choices

- Pre-allocates tensors for all batch sizes (0 to max_batch_size) to avoid dynamic allocation
- Uses `reconstruct_indices_from_tree_mask` (sgl_kernel) to build tree traversal indices from the mask
- Supports FULL_MASK mode (causal mask including prefix) for attention
- Updates corpus after each verify with `batch_put` (appends new token sequences)
- Cleans up match state for finished/retracted requests

### 5.3 Limitations

- Overlap scheduling not supported (spec_v1 only)
- No CUDA graph for draft generation (it's a CPU lookup)
- Always produces `draft_token_num` candidates per request (padded if needed)

---

## 6. DFlash Worker

### 6.1 Dual Forward Pass Architecture

`DFlashWorker` uses a separate draft model (not EAGLE-style) that drafts non-causal blocks:

1. **Draft model**: A separate `TpModelWorker` with its own KV cache pool and attention backend
2. **Block drafting**: Generates a fixed-size block of tokens (`block_size`) in one forward pass using non-causal attention (mask tokens fill non-first positions)
3. **Fused KV materialization**: Optionally uses Triton kernels to project target hidden states directly into draft KV cache, avoiding sequential layer-by-layer computation

### 6.2 Draft Windowing

When `speculative_draft_window_size` is set, DFlash maintains a compact sliding window KV cache for the draft model:
- Only recent tokens are visible to draft attention (saves memory)
- Draft req->token table is a private compact view over the global KV index space
- After each verify, rebuilds the compact view from committed target state

### 6.3 Vocabulary-Parallel Greedy Sampling

`_greedy_sample_from_vocab_parallel_head` handles the challenge of greedy sampling with TP-sharded LM head:
- Each rank computes local max over its vocab shard
- All-gather maxima and global IDs across TP ranks
- Select global argmax via gather
- Handles base + added vocab (e.g., LoRA-added tokens)

---

## 7. Adaptive Speculation

### 7.1 Adaptive Controller

`AdaptiveController` (adaptive_runtime_state.py) manages runtime state switching:

```python
class SpecRuntimeState:
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    draft_attn_backend        # Multi-step draft attention
    cuda_graph_runner         # Draft CUDA graph
    target_attn_backend       # Verify attention
    target_graph_runner       # Target CUDA graph
    draft_extend_attn_backend # Post-verify extend
    cuda_graph_runner_for_draft_extend
```

Each candidate step count has its own pre-built `SpecRuntimeState` (attention backends + CUDA graphs). Switching is an atomic pointer swap.

### 7.2 Adaptive Parameters

`AdaptiveSpeculativeParams` (adaptive_spec_params.py):
- **EMA tracking**: `ema_accept_len = (1 - alpha) * ema + alpha * batch_avg` (alpha default 0.2)
- **Warmup**: Ignores first `warmup_batches` (default 10)
- **Update interval**: Only checks every 5 batches
- **Hysteresis**: `down_hysteresis=-0.25` (need strong evidence to reduce), `up_hysteresis=0.0`
- **Candidate steps**: Discrete set like `[1, 3, 7]`, always includes initial steps

The update logic walks up/down `candidate_steps` based on whether `ema_accept_len` exceeds thresholds. Only topk=1 is supported (topk>1 trees have shape-dependent CUDA graphs that are expensive to pre-build for many configurations).

### 7.3 Constraints

Adaptive is only supported for:
- EAGLE or EAGLE3 algorithms
- topk=1 (chain topology, deterministic CUDA graph shapes)
- Not with DP attention, multi-layer eagle, TBO, or PDMux

---

## 8. CUDA Graph Integration

### 8.1 Draft CUDA Graphs

`EAGLEDraftCudaGraphRunner` captures the entire multi-step draft forward as a single CUDA graph:
- Captures for multiple batch sizes (power-of-2 bucketing)
- Each replay runs all `speculative_num_steps` forward passes in one graph
- Graph includes: embedding, FC projection, all decoder layers, attention, logits, topk sampling
- The per-step `ForwardContext` with `attn_backends[i]` is baked into the graph

### 8.2 Draft Extend CUDA Graphs

`EAGLEDraftExtendCudaGraphRunner` captures the post-verify extend forward:
- Fixed-length extend (num_draft_tokens per request)
- Used to fill draft KV cache after verification
- Supports Triton, TRTLLM MLA, and Tokenspeed MLA backends

### 8.3 Plan Stream Overlap (V2)

V2 uses a separate CUDA stream (`plan_stream`) to overlap:
- Verify preparation runs on `plan_stream` while draft output is still being consumed
- Draft extend preparation runs on `plan_stream` while verify output is being consumed
- `wait_stream` synchronizes before the main compute stream reads plan_stream results

---

## 9. EAGLE3 (Multi-Layer Improvements)

### 9.1 Architecture (`LlamaForCausalLMEagle3`, llama_eagle3.py)

EAGLE3 extends EAGLE with multi-layer auxiliary hidden states:

**Input projection**: Instead of just the final hidden state, EAGLE3 concatenates hidden states from multiple target model layers:
```python
self.fc = nn.Linear(target_hidden_size * num_aux, config.hidden_size)
```
Where `num_aux` defaults to 3 (configurable via `num_aux_hidden_states` or `eagle_aux_hidden_state_layer_ids`).

**Per-aux normalization**: Optional `fc_norm` (list of RMSNorm) normalizes each auxiliary hidden state chunk before concatenation.

**Input layer modification**: The first decoder layer's QKV projection accepts `2 * hidden_size` (concatenation of token embedding + projected target hidden states). Subsequent layers use standard `hidden_size`.

**Draft-to-target vocab mapping**: The `d2d`/`t2d` weight tensors enable vocabulary mapping between draft and target model (hot token selection). `hot_token_id` stores the mapping: `loaded_weight + arange(V)`.

### 9.2 Hidden State Capture

EAGLE3 uses `CaptureHiddenMode.FULL` during prefill and `CaptureHiddenMode.LAST` during decode to capture target hidden states. The `capture_aux_hidden_states` flag on the model tells the model runner to capture intermediate layer outputs.

### 9.3 EAGLE v1 (`LlamaForCausalLMEagle`, llama_eagle.py)

Simpler architecture:
- Single hidden state (no aux layers)
- `fc = Linear(hidden_size * 2, hidden_size)` -- concat embed + target hidden
- First layer skips input_layernorm (replaced with identity)
- Shares target's lm_head for draft logit computation

### 9.4 STANDALONE Algorithm

`STANDALONE` uses a vanilla LLM as the draft model (no hidden state sharing). `CaptureHiddenMode.NULL` is used everywhere -- the draft model operates independently with only token IDs as input. This simplifies the pipeline at the cost of reduced draft quality.

---

## 10. Comparison with vLLM Spec Decode

### 10.1 Architecture

| Aspect | SGLang | vLLM |
|--------|--------|------|
| Worker model | Composed (draft+target) or Inherited (v1) | Separate draft/verify model runners |
| Tree verify | Custom tree mask + single batched prefill | Similar tree attention approach |
| Spec v1/v2 | Two worker classes (v1=inherit, v2=composition) | Single architecture with optional overlap |
| N-gram | C++ NgramCorpus with BFS + external corpus | Python-based n-gram matcher |
| DFlash | Full implementation with fused KV | Not available |
| Adaptive | Pre-built runtime states, atomic swap | N/A |

### 10.2 Key Technical Differences

1. **Plan stream overlap**: SGLang v2 uses a separate CUDA stream for pipelining preparation work (buffer allocation, position computation) with GPU compute, reducing idle time between phases.

2. **topk=1 fast path**: SGLang pre-allocates constant parent/score_index buffers for chain topology, skipping the expensive `torch.topk` + `torch.sort` in `organize_draft_results`. vLLM does not have this optimization.

3. **Fused KV materialization (DFlash)**: Triton kernels that project target hidden states directly into draft KV cache for all layers in one pass, rather than sequential layer-by-layer forward.

4. **TP-aware verification sampling**: Both frameworks sync sampling results across TP ranks, but SGLang explicitly broadcasts `predict`, `accept_index`, and `num_correct_drafts` from rank 0.

5. **Grammar integration**: SGLang generates token bitmask on CPU overlapped with target verify forward, then applies it before sampling. This is similar to vLLM's structured output with bitmask FSM.

6. **Mamba/hybrid model support**: Both frameworks handle hybrid GDN models (Mamba + Attention) by tracking Mamba state updates after verify, with interval-based state checkpoints.

7. **Sliding window draft cache**: DFlash's compact sliding window KV cache for the draft model is unique to SGLang.

---

## 11. Key Insights and Innovations

### 11.1 Tree-Based Speculation vs Chain

The EAGLE tree approach (topk > 1) generates multiple candidate paths per step. The tree mask ensures each candidate only attends to its causal ancestors. This provides higher coverage than chain-based speculation (topk = 1), at the cost of:
- Larger batch sizes for verify (O(topk * steps) vs O(steps))
- More complex KV cache management after verify
- Larger CUDA graph memory footprint

In practice, topk=1 is often preferred because:
- The chain topology has deterministic shapes (enables adaptive speculation)
- CUDA graph memory is linear in steps
- KV cache management is simpler (no tree compaction)
- `num_draft_tokens = steps + 1` (fixed relationship)

### 11.2 Phase Pipeline Efficiency

The 3-phase pipeline (draft -> verify -> extend) has the following latency characteristics:
- **Draft**: O(steps) serial forward passes (can be CUDA-graphed)
- **Verify**: Single forward pass over all draft tokens (memory-bound, batched prefill)
- **Extend**: Single forward pass to fill draft KV cache

The total draft+verify+extend time must be less than `num_accepted_tokens * time_per_token` for speculation to be worthwhile. The acceptance rate is the critical metric.

### 11.3 Hot Token Optimization

Both EAGLE and EAGLE3 support reducing the draft model's vocabulary:
- `hot_token_id` maps draft vocab indices to target vocab indices
- Draft model only outputs logits over the "hot" subset (typically ~10K tokens)
- This reduces the `lm_head` computation from V to hot_vocab_size
- EAGLE3 stores the mapping in model weights (`d2d` tensor)

### 11.4 EMA-Based Adaptive Control

The adaptive speculation system is a pragmatic engineering choice:
- Pre-build CUDA graphs for all candidate step counts at startup (expensive but one-time)
- EMA provides smoothing to avoid oscillation (alpha=0.2 means ~5-batch memory)
- Hysteresis prevents rapid switching: -0.25 down hysteresis means acceptance must drop 0.25 below threshold before reducing steps
- Only applies after warmup (10 batches) and at fixed intervals (every 5 batches)

### 11.5 File Organization

The codebase is organized around the data flow:
- `spec_info.py` - Algorithm enum + SpecInput base class
- `eagle_info.py` - V1 data structures (EagleDraftInput, EagleVerifyInput, EagleVerifyOutput, EagleDraftExtendInput)
- `eagle_info_v2.py` - V2 mixins for prepare/sample methods
- `eagle_utils.py` - Tree construction, verify helpers, utility functions
- `eagle_worker.py` - V1 worker (EAGLEWorker extends TpModelWorker)
- `eagle_worker_v2.py` - V2 worker (EAGLEWorkerV2 composes EagleDraftWorker + target)
- `ngram_worker.py` - N-gram worker (no draft model)
- `dflash_worker.py` - DFlash worker (separate draft model, non-causal block drafting)
- `adaptive_spec_params.py` - EMA-based adaptive parameter controller
- `adaptive_runtime_state.py` - Runtime state container + controller
