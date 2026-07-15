# SGLang Speculative Decoding Architecture Deep Reading

> Date: 2026-07-15 | Source: sgl-project/sglang main branch
> Files: python/sglang/srt/speculative/ (34 files), overlap_utils.py, ragged_verify.py
> Supersedes: notebook/projects/sglang-spec-decode-reading.md (June 4, now outdated -- spec V1 removed)
> Cross-references: sglang-31197-fused-mla-lora, vllm-46105-dflash, sglang-rolloutkv

---

## 1. Architecture Overview

### 1.1 Seven Builtin Algorithms + Plugin Registry

SGLang defines 7 builtin speculative decoding algorithms as enum members in `SpeculativeAlgorithm` (spec_info.py):

```
DFLASH = auto()
DSPARK = auto()       # DFlash variant for DSv4-family MoE
EAGLE = auto()
EAGLE3 = auto()
FROZEN_KV_MTP = auto()  # Frozen-KV Multi-Token Prediction
STANDALONE = auto()     # Independent draft model (no weight sharing)
NGRAM = auto()          # Zero-overhead n-gram draft
NONE = auto()           # No speculative decoding
```

**Plugin registration** via `SpeculativeAlgorithm.register()` decorator:
```python
@SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=True)
def _factory(server_args):
    return MySpecWorker
```

Reserved names (all builtin enum members + "NEXTN" alias) cannot be re-registered. The `_assert_custom_spec_algo_conforms()` check ensures every plugin's `CustomSpecAlgo` subclass implements ALL `is_*()` and `supports_*()` methods from the enum, preventing silent `AttributeError` at dispatch sites.

### 1.2 Spec V2 Only (V1 Removed)

The current codebase is **entirely V2**. The spec V1 worker path was removed. All algorithms now run on the V2 scheduler schema:
- Algorithms with `supports_overlap=True`: run with overlap scheduling (Draft/Verify/DraftExtend interleaved)
- Algorithms with `supports_overlap=False`: run synchronously on V2 schema (deprecated, generates warning)

The `create_worker()` dispatch in `SpeculativeAlgorithm`:

| Algorithm | Worker Class | Overlap Support |
|-----------|-------------|-----------------|
| EAGLE (single-layer) | `EAGLEWorkerV2` | Yes |
| EAGLE (multi-layer) | `MultiLayerEagleWorkerV2` | Yes |
| EAGLE3 | `EAGLEWorkerV2` (same) | Yes |
| FROZEN_KV_MTP | `FrozenKVMTPWorkerV2` | Yes |
| DFLASH | `DFlashWorkerV2` | Yes |
| DSPARK | `DSparkWorkerV2` | Yes |
| STANDALONE | `StandaloneWorkerV2` | Yes |
| NGRAM | `NGRAMWorker` | Yes (different base class) |

### 1.3 Class Hierarchy

```
BaseSpecWorker (ABC)
  |-- EAGLEWorkerV2 (EAGLE/EAGLE3)
  |     |-- StandaloneWorkerV2
  |-- FrozenKVMTPWorkerV2
  |-- DFlashWorkerV2
  |-- DSparkWorkerV2
  |-- NGRAMWorker (direct BaseSpecWorker, no EagleDraftWorkerBase)

EagleDraftWorkerBase (ABC)
  |-- EagleDraftWorker (used by EAGLEWorkerV2._draft_worker)
  |-- StandaloneDraftWorker (used by StandaloneWorkerV2._draft_worker)
  |-- FrozenKVMTPDraftWorker (used by FrozenKVMTPWorkerV2._draft_worker)

TpModelWorker (used as draft_worker for DFLASH/DSPARK via draft_worker_common.py)
```

NGRAM has **no draft worker** (`draft_worker` property returns None). It drafts from a CPU-side corpus lookup.

---

## 2. SpecInput Abstract Class and 9 Subtypes

### 2.1 SpecInputType IntEnum

Phase identification for attention backend dispatching:

```
EAGLE_DRAFT = auto()          # EAGLE draft-decode forward
EAGLE_DRAFT_EXTEND = auto()   # EAGLE draft-extend (KV catch-up) forward
EAGLE_VERIFY = auto()         # EAGLE/EAGLE3/FROZEN_KV_MTP verify
FROZEN_KV_MTP_DRAFT = auto()  # Frozen-KV MTP draft-decode
FROZEN_KV_MTP_VERIFY = auto() # Frozen-KV MTP verify
DFLASH_DRAFT = auto()         # DFLASH/DSPARK draft-decode
DFLASH_VERIFY = auto()        # DFLASH/DSPARK verify
NGRAM_VERIFY = auto()         # NGRAM verify
```

### 2.2 SpecInput ABC (base class)

Key class-level defaults (NOT __init__ assignments, to avoid dataclass __post_init__ clobbering):
- `ragged_verify_layout: Optional[RaggedVerifyLayout] = None` -- per-request verify lengths for DSPark
- `num_tokens_per_req: int = -1` -- uniform per-request token width (-1 = not set)
- `num_tokens_for_logprob_per_req: int = -1`
- `dsa_topk_indices: Optional[torch.Tensor] = None` -- DSA MTP IndexShare seed relay
- `future_dsa_topk_indices_available: bool = False`
- `dsa_seed_topk_capture: Optional[torch.Tensor] = None`

Cross-algorithm phase guards:
- `is_draft_input()` -- checks `spec_input_type` in {EAGLE_DRAFT, EAGLE_DRAFT_EXTEND, FROZEN_KV_MTP_DRAFT, DFLASH_DRAFT}
- `is_verify_input()` -- checks in {EAGLE_VERIFY, FROZEN_KV_MTP_VERIFY, DFLASH_VERIFY, NGRAM_VERIFY}

### 2.3 SpecInput Subtype Map

| Algorithm | Draft Input | Verify Input | Draft-Extend Input |
|-----------|-------------|-------------|-------------------|
| EAGLE/EAGLE3 | `EagleDraftInput` | `EagleVerifyInput` | `EagleDraftExtendInput` |
| FROZEN_KV_MTP | `FrozenKVMTPDraftInput` (extends `EagleDraftInput`) | `FrozenKVMTPVerifyInput` (extends `EagleVerifyInput`) | None (seed-select only) |
| DFLASH/DSPARK | `DFlashDraftInputV2` | `DFlashVerifyInput` | None (uses target model) |
| NGRAM | None (CPU-side) | `NgramVerifyInput` | None |
| STANDALONE | `EagleDraftInput` (hidden_states=None) | `EagleVerifyInput` | `EagleDraftExtendInput` |

---

## 3. Six Algorithm Deep Dive

### 3.1 EAGLE/EAGLE3

**Architecture**: Draft model is a lightweight single-layer (EAGLE) or multi-layer (EAGLE3) autoregressive head that shares the target model's embedding and lm_head. The draft reads the target model's **last hidden state** as its recurrent input.

**Worker class**: `EAGLEWorkerV2` with `EagleDraftWorker` as `_draft_worker`.

**Key properties**:
- Shares `embed` and `lm_head` with target (via `set_embed_and_head()`)
- `hot_token_id`: optional reduced vocabulary mapping for EAGLE3 (built-in in some models like nvidia/gpt-oss-120b-Eagle3)
- `CaptureHiddenMode.LAST` for draft, `CaptureHiddenMode.FULL` for verify (target returns all hidden states)
- DSA IndexShare support: `index_share_for_mtp_iteration` (for GLM-5.2-style MTP models)

**3-Phase Pipeline**:

```
Phase 1: DRAFT (draft_forward)
  - Input: EagleDraftInput (bonus_tokens, topk_p, topk_index, hidden_states, dsa_topk_indices)
  - For each step i in [0, speculative_num_steps-1]:
    1. select_top_k_tokens(i, ...) -> input_ids, hidden_states
    2. Draft model forward (one decode step) on draft_runner
    3. From logits: renorm_draft_probs -> fast_topk(k=topk) -> next topk_p/topk_index
    4. Capture hidden_states for next step
  - topk=1 fast path: chain topology, precomputed parent_list/top_scores_index
  - Output: parent_list, top_scores_index, draft_tokens, draft_probs
  - Then build_tree_kernel_efficient -> EagleVerifyInput

Phase 2: VERIFY (target model forward)
  - Input: EagleVerifyInput (draft_token tree, custom_mask, positions, retrieve_index, ...)
  - Target model forward: ForwardMode.TARGET_VERIFY
  - eagle_sample(): verify each draft token against target logits
  - Output: accept_lens, next_token_ids (bonus tokens), next_draft_input

Phase 3: DRAFT_EXTEND (KV cache catch-up)
  - Input: EagleDraftExtendInput (hidden_states, num_accept_tokens, ...)
  - Draft model forward: ForwardMode.DRAFT_EXTEND_V2
  - Fills draft KV cache for accepted positions
  - Output: next_draft_input with topk_p/topk_index/hidden_states for next iteration
```

**Rejection Sampling**: Optional mode (`speculative_use_rejection_sampling`). Requires topk=1. Draft emits `draft_probs` (full vocab distribution). Verify uses `eagle_sample` with rejection sampling instead of greedy matching.

**Adaptive Steps**: EAGLE supports adaptive speculative decoding via `AdaptiveController`:
- Per-BS EMA tracking of acceptance rates
- Runtime step switching (1, 3, 7, etc.) with atomic state swap
- `SpecRuntimeState` bundles: draft_attn_backend + cuda_graph + target_attn_backend + target_graph + draft_extend_attn_backend + draft_extend_graph

**Memory sharing**: Embeddings/lm_head shared => saves ~2x vocab_size * hidden_size parameters. Draft model itself still occupies GPU memory (one or few layers).

**RTX 4090 impact**:
- EAGLE (single-layer): ~0.5-1 GiB additional draft model (manageable for <=7B targets)
- EAGLE3 (multi-layer): ~1-3 GiB additional draft model (tight for <=7B targets)
- topk=1 chain mode is the RTX 4090 sweet spot (reduced tree width, simpler CUDA graphs)
- Rejection sampling overhead: stores full draft_probs (vocab_size per step per batch), significant memory pressure

### 3.2 DFLASH (Deep Flash / Non-Causal Bidirectional Draft)

**Architecture**: Draft model uses **non-causal (bidirectional) attention** to predict draft tokens in one forward pass, rather than autoregressive multi-step. The draft model sees the entire prefix context simultaneously.

**Worker class**: `DFlashWorkerV2` with `TpModelWorker` as draft (built via `draft_worker_common.build_draft_tp_worker()`).

**Key innovation**: Instead of multi-step autoregressive draft (EAGLE), DFLASH:
1. Runs ONE draft forward pass producing `draft_token_num` tokens per request
2. Uses bidirectional attention in the draft model (draft tokens can attend to each other)
3. Verify is a linear (non-tree) pass: `topk=1` always

**SpecInput types**:
- `DFlashDraftInputV2`: carries future_indices, reserved_seq_lens_cpu, plan stream buffers
  - Has `prepare_for_decode()`: over-allocates KV slots ahead of time for the next step
  - Plan stream optimization: KV allocation runs on a separate CUDA stream to overlap with forward
- `DFlashVerifyInput`: `topk=1` always, carries `ragged_verify_layout` (optional per-request verify lengths)

**3-Phase Pipeline**:

```
Phase 1: DRAFT
  - Over-allocate KV slots (prepare_for_decode with plan_stream)
  - Draft model forward: ForwardMode.DECODE (on draft_runner)
  - Output: draft tokens for next verify block

Phase 2: VERIFY
  - Target model forward: ForwardMode.TARGET_VERIFY
  - Linear verify (not tree): draft_token_num tokens per request verified sequentially
  - Compute accept lengths via _compute_dflash_accept_bonus_triton_unchecked

Phase 3: DRAFT_EXTEND
  - No separate draft_extend forward for DFLASH
  - Instead: target model catches up via normal decode
```

**Over-allocation strategy**: DFLASH reserves `2 * block_size` KV slots per request ahead of time, avoiding allocator backup/restore in the hot path. `kv_allocated_len` tracks the high-water mark; `kv_committed_len` tracks actual committed positions.

**Acceptance rate**: ~17% reported for Qwen3-8B-DFlash-b16 (pos0=77%). Lower than EAGLE's 80-90% because:
- Bidirectional draft is less precise than autoregressive
- But: single forward pass, no multi-step overhead
- Net effect depends on draft/verify latency ratio

**RTX 4090 impact**:
- Draft model memory: same as a full independent model load (no weight sharing)
- No recurrent hidden_states relay (simpler than EAGLE)
- Plan stream optimization critical for reducing verify overhead
- Block-based KV allocation (page_size-aware) reduces fragmentation
- ~17% acceptance means ~1.17x throughput boost (modest)

### 3.3 DSPARK (DeepSeek-V4 MoE variant)

**Architecture**: DSPARK is a DFLASH variant specifically designed for DSv4-family MoE models. It adds:
1. MoE-aware draft attention backend (`DSV4_DRAFT_ATTENTION_BACKEND`)
2. Block-level accept estimation (`DSparkBlockAcceptEstimator`)
3. Ragged verify support (per-request verify lengths, not uniform)
4. Confidence relay for overlap scheduling

**Worker class**: `DSparkWorkerV2` with `TpModelWorker` draft (MoE-specific backend).

**Key components** (in `dspark_components/`):
- `dspark_draft.py`: `DraftBlockProposer` + `make_next_draft_input`
- `dspark_verify.py`: `TargetVerifyExecutor` + `DsparkVerifyEpilogue`
- `dspark_planner.py`: `DSparkVerifyPlanner` + `alloc_verify_window`
- `dspark_block_accept_estimator.py`: per-request accept estimation
- `dspark_kv_inject.py`: `TargetHiddenKvInjector` (injects target hidden states into draft KV)
- `dspark_sps.py`: Single-Phase Speculative (simplified mode)
- `dspark_sts.py`: Two-Phase Speculative (full mode)
- `kernels/`: Custom Triton kernels

**Ragged verify**: DSPark supports per-request verify lengths via `RaggedVerifyLayout`:
- `STATIC`: all requests verify the same number of tokens
- `CAP_ACCEPT`: verify length = min(draft_tokens, accept_estimate)
- `COMPACT`: per-request variable lengths with token-bucket-keyed CUDA graphs

**Confidence relay**: `ConfidenceRelay` ring buffer carries draft confidence scores across overlap iterations, with ring depth = CONFIDENCE_RELAY_RING_LAG + 1 = 3.

**RTX 4090 impact**:
- DSv4 MoE draft is memory-heavy (multiple expert weights)
- DSPARK needs TP>=2 for MoE draft (not viable on single RTX 4090)
- Ragged verify can reduce wasted computation for mixed-accept batches

### 3.4 FROZEN_KV_MTP (Frozen-KV Multi-Token Prediction)

**Architecture**: The draft model **reads the target KV cache read-only** and **owns no KV pool**. It's a lightweight 2-layer MTP head that:
1. Uses the target model's KV cache as its "context" (via `FrozenKVMTPContext`)
2. Does NOT extend the draft KV cache -- instead, seed selection happens at the start of next draft
3. The "draft extend" phase is NOT a model forward: it selects last accepted token + target hidden state

**Worker class**: `FrozenKVMTPWorkerV2` with `FrozenKVMTPDraftWorker` (inherits both `EagleDraftWorkerBase` and `TpModelWorker`).

**Critical memory optimization**: `max_total_num_tokens=64` for draft pool config (dummy -- frozen draft doesn't need real KV allocation). This means almost zero additional memory for KV cache.

**FrozenKVMTPContext**: Maps assistant-logical layer IDs to target-physical layer IDs:
```python
@dataclass(frozen=True)
class FrozenKVMTPContext:
    target_token_to_kv_pool: KVCache
    physical_layer_ids: Dict[int, int]  # assistant logical -> target physical
```

**Draft model reads target KV via**:
- `frozen_kv_target_view()`: wraps the forward batch's token_to_kv_pool with the target pool
- `target_kv_pool_view()`: creates a read-only view of the target KV for the draft model's attention

**Key properties**:
- Shares embedding/lm_head with target (via `set_embed_and_head`)
- No separate KV pool allocation (only 64 token slots, dummy)
- `draft_extend_attn_backend = None` (no extend phase)
- `cuda_graph_runner_for_draft_extend = None`
- `CaptureHiddenMode.LAST` for draft forward (reads target hidden, not full)

**RTX 4090 impact**:
- **BEST option for RTX 4090** among tree-based spec decode algorithms
- Only adds ~2 layer parameters (backbone + MTP head), ~0.1-0.3 GiB
- Zero additional KV cache memory (reads target KV read-only)
- topk=1: only TritonAttnBackend supported for topk > 1
- Shares EAGLE's verify tree mask contract (compatible with existing tree verification kernels)

### 3.5 NGRAM (Zero-Overhead N-Gram Draft)

**Architecture**: NGRAM has **no draft model at all**. It uses a CPU-side `NgramCorpus` trie structure built from request token streams to propose draft tokens. Zero GPU overhead for draft generation.

**Worker class**: `NGRAMWorker` (directly inherits `BaseSpecWorker`, NOT `EagleDraftWorkerBase`).

**NgramCorpus**: CPU-side trie with configurable parameters:
- `min_bfs_breadth`, `max_bfs_breadth`: BFS expansion breadth
- `match_type`: matching strategy
- `capacity`: trie capacity
- `max_trie_depth`: depth limit (default speculative_ngram_max_trie_depth)
- External corpus support: load from file via `speculative_ngram_external_corpus_path`

**Draft generation**:
1. For each request, extract last `max_trie_depth` tokens from `origin_input_ids + output_ids`
2. `ngram_corpus.batch_get()`: BFS search in trie -> (draft_tokens, tree_mask) as numpy arrays
3. `reconstruct_indices_from_tree_mask()`: GPU kernel to build tree structure
4. Always produces exactly `draft_token_num` tokens per request

**Verify**: Uses same tree verification as EAGLE:
- `NgramVerifyInput` carries tree mask + retrieve_index structure
- Target model forward: `ForwardMode.TARGET_VERIFY`
- `eagle_sample()` for verification (same kernel)
- `max_tree_depth = draft_token_num` (corpus BFS is depth-unbounded)
- `tree_topk = -1` (irregular tree -- branching follows corpus matches)

**Ngram corpus update**: After each verify:
1. `_update_ngram_corpus()`: adds accepted token sequences to trie
2. Erases departed requests' match state (requests that left the batch)

**Overlap mode**: In overlap mode, accepted tokens from the previous round are spliced into `output_ids` before corpus lookup (one iteration behind). Grammar batches process results before next draft, so no splicing needed.

**RTX 4090 impact**:
- **ZERO additional GPU memory** -- no draft model, no draft KV cache
- CPU-side trie is negligible (<100 MiB for typical corpora)
- Acceptance rate depends on corpus quality: well-matched domains can achieve 40-70%
- External corpus: can load domain-specific text for higher acceptance
- **RTX 4090 BEST for GRPO rollout** if domain is repetitive (e.g., RLHF reward model responses)
- Grammar/structured output support: `generate_token_bitmask()` for constrained decoding

### 3.6 STANDALONE (Independent Draft Model)

**Architecture**: An independent draft model that does NOT share embeddings/lm_head with the target. It's a separate LLM loaded as a draft worker.

**Worker class**: `StandaloneWorkerV2` (extends `EAGLEWorkerV2`) with `StandaloneDraftWorker` (extends `EagleDraftWorker`).

**Key differences from EAGLE**:
- `init_lm_head()` override: no sharing (`pass` -- draft uses its own lm_head)
- Requires **matching vocabulary** (same vocab_size AND same tokenizer mapping)
- `_validate_vocab_compatibility()`: raises ValueError if vocab mismatch
- `CaptureHiddenMode.NULL` for draft and draft_extend (no hidden states relay)
- `EagleDraftInput.hidden_states = None` always

**Memory impact**: Full independent model load. For a 1B draft model alongside a 7B target:
- ~2 GiB additional (model weights + KV cache)
- Not viable on RTX 4090 for target models >= 3B (total exceeds 24 GiB)

**RTX 4090 impact**:
- **NOT viable** for most RTX 4090 GRPO configurations
- Requires separate draft model weights + separate KV pool
- Only useful when target model is very small (<=1B) and draft is even smaller

---

## 4. 3-Phase Pipeline: Draft -> Verify -> DraftExtend

### 4.1 Prefill Path (Extend)

```
1. Target prefill (ForwardMode.EXTEND)
   - CaptureHiddenMode.FULL (for EAGLE) or NULL (for STANDALONE)
   - Output: logits_output.hidden_states, next_token_ids

2. Publish (overlap fence at target-end)
   - on_publish(new_seq_lens) -- updates FutureMap

3. Draft extend for prefill (EagleDraftWorker._draft_extend_for_prefill)
   - Replaces input_ids with chunked-prefill-aware tail tokens
   - ForwardMode.DRAFT_EXTEND_V2 (on draft_runner)
   - CaptureHiddenMode.LAST (EAGLE) or NULL (STANDALONE)
   - Output: EagleDraftInput (topk_p, topk_index, hidden_states, bonus_tokens, dsa_topk_indices)
```

### 4.2 Decode Path

```
1. Adaptive step selection
   - activate_step_by_batch(batch_size) -- may switch runtime state

2. Draft phase (spec_stage_span "draft")
   - draft_worker.draft(batch) -> EagleVerifyInput / DFlashVerifyInput / etc.
   - For EAGLE: multi-step autoregressive loop (draft_forward)
   - For DFLASH: single forward pass on draft model
   - For NGRAM: CPU-side corpus lookup -> NgramVerifyInput
   - For FROZEN_KV: seed forward then autoregressive loop reading target KV

3. Verify phase (spec_stage_span "verify")
   - batch.spec_info = verify_input from draft
   - target_worker.forward_batch_generation(batch, is_verify=True)
   - ForwardMode.TARGET_VERIFY
   - eagle_sample() / _compute_dflash_accept_bonus_triton / ...
   - Output: accept_lens, next_token_ids, logits_output

4. Publish (overlap fence at verify-end)
   - on_publish(new_seq_lens) -- updates FutureMap before draft_extend

5. Draft extend phase (spec_stage_span "draft_extend")
   - For EAGLE: draft_worker._draft_extend_for_decode(batch, batch_result)
   - For FROZEN_KV: seed-select only (no model forward)
   - For DFLASH: no separate extend (target model catches up)
   - For NGRAM: no extend phase
```

### 4.3 Zero-Step Mode (Adaptive Spec Steps=0)

When adaptive spec selects `steps=0` (high batch size), the worker:
1. Builds a **trivial verify input**: 1-node tree rooted at bonus token (functionally a plain decode)
2. Runs `_build_trivial_verify_input()` -> `EagleVerifyInput(draft_token_num=1, spec_steps=0)`
3. Optionally skips draft_extend via `SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND`

---

## 5. Overlap Scheduling Interaction

### 5.1 FutureMap: Always-On Cross-Iter Relay

`FutureMap` (overlap_utils.py) is the central relay mechanism for spec-v2 overlap scheduling. It's pool-indexed (by `req_pool_indices`) and always-on:

**Key buffers**:
- `output_tokens_buf`: [req_pool_size] int64 -- last sampled token per request slot
- `new_seq_lens_buf`: [req_pool_size] int64 -- last accepted seq_lens per request slot
- `topk_p_buf`: [req_pool_size, topk] float32 -- EAGLE draft top-k probabilities
- `topk_index_buf`: [req_pool_size, topk] int64 -- EAGLE draft top-k indices
- `hidden_states_buf`: [req_pool_size, hidden_size] -- draft hidden states
- `dsa_topk_indices_buf`: [req_pool_size, index_topk] -- DSA indexer seed
- `draft_probs_buf`: [req_pool_size, vocab] float32 -- rejection sampling probs

**Operation flow**:
```
publish(future_indices, new_seq_lens, confidence):
  - Scatter new_seq_lens into new_seq_lens_buf[indices]
  - Record publish_ready event (CUDA Event)
  - Optional: scatter confidence into ConfidenceRelay

stash(RelayPayload):
  - Scatter bonus_tokens, topk_p, topk_index, hidden_states into respective bufs
  - Lazy-init forward bufs on first non-empty stash

resolve_forward_inputs(batch, future_map):
  - Prefill: H2D copy from pinned CPU staging
  - Decode: gather from future_map.output_tokens_buf[req_pool_indices]
  - Spec: _resolve_spec_extras() gathers draft-specific fields

resolve_seq_lens_cpu(batch):
  - Private D2H stream: copy new_seq_lens_buf -> pinned CPU
  - Gate on publish_ready event (non-blocking)
  - Result: batch.seq_lens_cpu = pinned_buf[req_pool_indices_cpu]
```

**Debug mode**: `SGLANG_IS_IN_CI` enables:
- Poisoned init (all buffers filled with -1)
- `_assert_nonneg_and_invalidate()`: fused assert + scatter -1 (compiled via `torch.compile`)
- Consume-once tracking: `_publish_fresh` flag

### 5.2 WAR Barrier

The scheduler's WAR (Write-After-Read) barrier ensures the next iteration's forward doesn't read stale data from the previous iteration's buffers. It waits on the `publish_ready` event from the step's last shared-buffer-reading phase:
- EAGLE: draft_extend is the last phase (runs on draft_runner)
- DFLASH: target verify is the last phase (runs on target runner)
- NGRAM: target verify is the last phase

The `war_fastpath_runner` property identifies which runner owns the read-done event.

### 5.3 Plan Stream Optimization

For DFLASH and EAGLE, a separate "plan stream" runs KV allocation and metadata preparation ahead of the main forward:

```python
plan_stream, plan_stream_ctx = get_plan_stream(self.device)
```

The plan stream:
1. Waits on the scheduler stream (fresh batch data)
2. Runs `prepare_for_decode()` / `prepare_for_draft_extend()`
3. The main stream then waits on the plan stream before the actual forward

This overlaps metadata preparation with the previous batch's forward, improving GPU occupancy.

### 5.4 Confidence Relay (DSPark)

DSPark uses a ring-buffer confidence relay with depth 3 (`CONFIDENCE_RELAY_RING_DEPTH`):
- `scatter(indices, confidence)`: scatter draft confidence into per-slot buffer
- `issue_ring_copy()`: D2H copy into pinned ring buffer (off schedule stream)
- `resolve()`: gather from ring buffer for next iteration's draft decisions

---

## 6. RTX 4090 / H20-3e Applicability Analysis

### 6.1 Memory Budget (RTX 4090 24 GiB)

| Algorithm | Additional Draft Memory | Additional KV Memory | Total Overhead | Viability |
|-----------|------------------------|--------------------|--------------|-----------|
| NGRAM | 0 | 0 | **~0 GiB** | BEST |
| FROZEN_KV_MTP | ~0.1-0.3 GiB (2 layers) | 0 (reads target KV) | **~0.3 GiB** | BEST (tree-based) |
| EAGLE (topk=1, single-layer) | ~0.5-1 GiB | Draft KV (small) | **~1-2 GiB** | Good (for <=7B target) |
| EAGLE3 (multi-layer) | ~1-3 GiB | Draft KV (small) | **~2-4 GiB** | Marginal |
| DFLASH (independent draft) | Full model (1-2 GiB for 0.5B draft) | Draft KV (full) | **~2-4 GiB** | Marginal |
| DSPARK (DSv4 MoE) | MoE weights (huge) | Draft KV (MoE) | **>>4 GiB** | NOT viable (needs TP>=2) |
| STANDALONE | Full model (2+ GiB for 1B) | Draft KV (full) | **~4+ GiB** | NOT viable |

### 6.2 Acceptance Rate Comparison

| Algorithm | Typical Acceptance | Throughput Boost | Notes |
|-----------|-------------------|-----------------|-------|
| NGRAM | 40-70% (domain-matched) | 1.4-1.7x | Depends on corpus quality |
| FROZEN_KV_MTP | ~70-80% (for well-trained MTP) | ~1.7-1.8x | MTP heads are well-calibrated |
| EAGLE (topk=1) | ~80-90% | ~1.8-1.9x | Best acceptance for general domains |
| DFLASH | ~17% | ~1.17x | Low acceptance but single-pass |
| DSPARK | Variable (MoE) | Variable | DSv4-specific |

### 6.3 GRPO Rollout Impact

For GRPO rollout on RTX 4090, the spec decode algorithm choice depends on the rollout model size:

**Target <=3B (e.g., Qwen2.5-1.5B, Llama-3.2-1B)**:
- EAGLE topk=1 viable (1B draft model fits alongside 3B target)
- NGRAM viable (zero overhead, domain-matched RLHF responses)
- FROZEN_KV_MTP viable if model supports it (GLM-5.x, DSv3.x)

**Target 7B (e.g., Qwen2.5-7B)**:
- NGRAM = BEST choice (zero overhead)
- FROZEN_KV_MTP = BEST if supported
- EAGLE topk=1 = marginal (1B draft alongside 7B target leaves ~16 GiB free)
- DFLASH = NOT viable (even 0.5B draft exceeds budget)

**Target >=14B (e.g., Qwen3-14B)**:
- NGRAM = ONLY viable option (all tree-based algorithms OOM)
- Or: no spec decode at all (plain decode)

### 6.4 H20-3e (143 GiB each, 4 cards)

On H20-3e with 4 cards:
- All algorithms viable (143 GiB per card is massive)
- DSPARK + DSv4-family models viable with TP=2+
- EAGLE/EAGLE3 best for general throughput
- DFLASH viable for latency-sensitive scenarios

---

## 7. Key Implementation Details

### 7.1 Tree Verification Kernel (eagle_utils.py)

`build_tree_kernel_efficient()` builds the verification tree from draft outputs:
- Input: bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens, topk, num_steps, num_draft_tokens
- Output: tree_mask, positions, retrieve_index, retrieve_next_token, retrieve_next_sibling, draft_tokens
- `tree_mask_mode`: controls mask generation strategy (default from `default_tree_mask_mode()`)
- CUDA kernel: `sgl_build_tree_kernel_efficient` (from sgl_kernel)
- CPU fallback: `sgl_build_tree_kernel_efficient_cpu`

`eagle_sample()` verifies the tree against target logits:
- Greedy verify: `verify_tree_greedy_kernel_triton`
- Rejection sampling: `speculative_use_rejection_sampling=True`

### 7.2 Cache Location Assignment (spec_utils.py)

Multi-step draft uses `assign_draft_cache_locs_contiguous` (from sgl_kernel):
- For `page_size=1` or `topk=1`: simple contiguous allocation
- For `page_size>1` and `topk>1`: per-branch page-aligned allocation with `duplicate_prefix_tail_to_draft_branches()`

Draft-extend uses `assign_extend_cache_locs`:
- Per-request start_offset to end_offset allocation in req_to_token pool

### 7.3 CUDA Graphs

Three graph types for EAGLE:
1. **Draft decode graph**: `EAGLEDraftCudaGraphRunner` -- multi-step autoregressive draft
2. **Draft extend graph**: `EAGLEDraftExtendCudaGraphRunner` -- KV catch-up after verify
3. **Target verify graph**: `DecodeCudaGraphRunner` (shared with non-spec decode)

Frozen-KV MTP has its own `FrozenKVMTPCudaGraphRunner` for the draft loop.

DFLASH uses the standard `DecodeCudaGraphRunner` for both draft and verify.

NGRAM uses `DecodeCudaGraphRunner` for verify only (no draft model forward).

### 7.4 MoE Backend Context

`speculative_moe_backend_context()` and `speculative_moe_a2a_backend_context()` are context managers that swap the MoE backend for draft model forwards. This is necessary because:
- Draft model may use different MoE routing (fewer experts, different TP)
- DSv4 MoE models need special backend for sparse attention
- A2A (All-to-All) backend needs different configuration for draft vs target

### 7.5 Weight Update Support

`BaseSpecWorker` provides weight update methods:
- `update_weights_from_disk()`: iterates over all `draft_worker.draft_runners`
- `update_weights_from_ipc()`: same pattern
- NGRAM override: `update_weights_from_tensor()` delegates to target worker (NGRAM has no draft weights)

This is CRITICAL for GRPO sleep/wake: when the target model weights are updated, the draft model must also be updated (for EAGLE/FROZEN_KV). NGRAM is unaffected since it has no draft model.

---

## 8. Configuration Parameters

### 8.1 Key Server Args

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speculative_algorithm` | "NONE" | Algorithm name (EAGLE, EAGLE3, DFLASH, DSPARK, FROZEN_KV_MTP, NGRAM, STANDALONE) |
| `speculative_num_steps` | 5 | Number of draft steps (autoregressive iterations) |
| `speculative_num_draft_tokens` | steps+1 | Total draft tokens per request (for tree-based) |
| `speculative_eagle_topk` | 1 | Top-k branching factor for EAGLE tree |
| `speculative_num_draft_tokens` (DFLASH) | 4 | Verify block size |
| `speculative_use_rejection_sampling` | False | Enable rejection sampling (EAGLE topk=1 only) |
| `speculative_adaptive` | False | Enable adaptive step selection |
| `speculative_adaptive_config` | None | Path to adaptive config JSON (default: per-BS EMA) |
| `speculative_ngram_*` | Various | NGRAM-specific params (match_type, capacity, max_trie_depth, external corpus) |
| `speculative_draft_attention_backend` | None | Override draft model attention backend |
| `speculative_token_map` | None | Hot token ID mapping for reduced vocab |

### 8.2 Adaptive Config (Default)

```json
{
  "1": {"candidate_steps": [1, 3, 7], "up_hysteresis": 0.0, "down_hysteresis": -0.25, "ceiling_coeff": 0},
  "8": {"candidate_steps": [0, 1, 3], "up_hysteresis": 0.0, "down_hysteresis": 0.0, "ceiling_coeff": 0},
  "32": {"candidate_steps": [0, 1], ...},
  "64": {"candidate_steps": [0], ...}
}
```

BS=1 can draft up to 7 steps; BS=8 reduces to 3 steps; BS>=32 reduces to 0 or 1 step (no drafting). This is the right trade-off: small batches benefit most from spec decode, large batches are bottlenecked by verify overhead.

---

## 9. Cross-Framework Comparison

| Feature | SGLang | vLLM | verl |
|----------|--------|------|------|
| Spec decode algorithms | 7 (EAGLE/EAGLE3/DFLASH/DSPARK/FROZEN_KV/NGRAM/STANDALONE) | 4 (Eagle/Eagle3/MTP/NGram) | None (uses vLLM/SGLang rollout) |
| Plugin registry | Yes (CustomSpecAlgo) | No | N/A |
| Overlap scheduling | Yes (FutureMap, WAR barrier) | Partial (different mechanism) | Uses rollout server's scheduling |
| Adaptive steps | Yes (AdaptiveController, EMA) | No | N/A |
| Frozen-KV MTP | Yes (unique) | No (has MTP but not frozen) | N/A |
| DFLASH bidirectional | Yes (unique) | vLLM #46105 (in progress) | N/A |
| DSPark MoE-aware | Yes (unique) | No | N/A |
| Ragged verify | Yes (DSPark, 3 modes) | No | N/A |
| Confidence relay | Yes (DSPark ring buffer) | No | N/A |
| Plan stream overlap | Yes | No | N/A |

---

## 10. GRPO Rollout Recommendations

### 10.1 RTX 4090 (24 GiB, dp=1)

**Recommended**: NGRAM or FROZEN_KV_MTP (if model supports it)

**NGRAM config for GRPO**:
```bash
--speculative-algorithm NGRAM
--speculative-num-draft-tokens 5
--speculative-ngram-max-trie-depth 5
--speculative-ngram-external-corpus-path <domain-specific-text>
```

**FROZEN_KV_MTP config** (for GLM-5.x/DSv3.x MTP models):
```bash
--speculative-algorithm FROZEN_KV_MTP
--speculative-num-steps 4
--speculative-num-draft-tokens 5
--speculative-eagle-topk 1
```

**Sleep/wake interaction**:
- NGRAM: unaffected (no draft model, no draft KV). Corpus survives sleep/wake.
- FROZEN_KV_MTP: draft reads target KV. After wake_up, target KV is restored -> draft can continue. CRITICAL: `FrozenKVMTPContext.bind_frozen_kv_context()` must re-bind after weight update (target pool address may change).
- EAGLE: draft model must be updated alongside target. Draft KV cache must be re-allocated after sleep/wake.

### 10.2 H20-3e (143 GiB x4)

**Recommended**: EAGLE topk=1 for general throughput, DSPARK for DSv4-family models

```bash
# EAGLE config
--speculative-algorithm EAGLE
--speculative-num-steps 5
--speculative-num-draft-tokens 6
--speculative-eagle-topk 1
--speculative-adaptive True
```

### 10.3 Weight Update Race Conditions

CRITICAL for GRPO sleep/wake with spec decode:
1. **NGRAM**: No race condition (no draft model). Corpus state persists across weight updates.
2. **EAGLE/FROZEN_KV**: Both draft and target models must be updated atomically. If only target is updated, draft model will produce inconsistent proposals (acceptance rate crashes).
3. **DFLASH**: Draft model must be updated alongside target. Over-allocated KV slots must be invalidated at weight-reload boundary (same pattern as MoE cache clobbering in #28676).
4. **STANDALONE**: Draft model is independent, but vocab must still match after target update.

---

## 11. Known Issues and Bugs

| Issue | Algorithm | Severity | Description |
|-------|-----------|----------|-------------|
| #48613 | DFLASH/EAGLE | CRITICAL | GDN batch invariance broken for Qwen3.5/3.6 -- supports_batch_invariance()=True insufficient, both FlashInfer & Triton fail bs=1 vs bs=N |
| #28679 | DFLASH | CRITICAL | GDN intermittent decode degeneracy -- worsens over uptime, clears on restart, NOT DSV4 but same state lifecycle mismatch pattern |
| #28676 | DSPARK/DFLASH | CRITICAL | MXFP8 MoE shuffle cache CLOBBERED -> 10th DSV4 failure (MERGED July 1) |
| #28685 | DFLASH | CRITICAL | GLM-5.2 FP8 block-fp8 wrong on MI350X -> 12th DSV4 failure |
| #26358 | EAGLE | Medium | ROCm argmax tie-break corrupts MTP draft selection on FP8 logits |
| CUDA stream race | EAGLE (draft_extend) | Low | Plan stream vs main stream cross-stream .to() cast can create data race (handled by casting before entering plan stream) |

---

## 12. Summary

SGLang's spec decode architecture is the most comprehensive in the open-source ecosystem:
- **7 builtin algorithms** + plugin registry
- **Spec V2 only** (V1 removed), all algorithms support overlap scheduling
- **3-phase pipeline** (Draft -> Verify -> DraftExtend) with FutureMap-based cross-iter relay
- **Adaptive step selection** for EAGLE/EAGLE3
- **Ragged verify** for DSPark MoE models
- **Plan stream** optimization for overlapping metadata prep with forward passes

**RTX 4090 GRPO rollout**: NGRAM (zero overhead) and FROZEN_KV_MTP (minimal overhead) are the clear winners. EAGLE topk=1 is viable for small targets. DFLASH/DSPARK/STANDALONE are not viable due to memory constraints.

**H20-3e GRPO rollout**: EAGLE with adaptive steps is the best general choice. DSPARK for DSv4-family models.
