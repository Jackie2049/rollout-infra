# SGLang RolloutKV: Prefix KV Cache Pinning for RL Rollout (PR #28608)

> Deep reading created: 2026-07-14
> PR: https://github.com/sgl-project/sglang/pull/28608
> Author: chengcuiping (also author of previous #24781)
> State: OPEN (replaces closed #24781, rebased 2026-06-18)
> Changes: +768/-5 across 6 files
> Replaces: PR #24781 (closed, same concept but without TTL eviction)

---

## 1. Core Mechanism: How RolloutKV Pins Prefix KV Cache

### The Problem

In RL rollout (GRPO/PPO), the same long prompt is prefilled repeatedly:
- Once per generation sample (G=8 means 8 prefill passes)
- Once for actor logprob scoring (prompt + response)
- Once for reference logprob scoring (prompt + response)
- Once for reward/value scoring (if multi-role)
- Multi-round: PPO recomputes logprobs across multiple update rounds

Each prefill recomputes the entire prompt KV cache from scratch. For long prompts (8K-16K tokens), this dominates wall time.

### Why RadixAttention Alone Does Not Solve This

SGLang's RadixAttention is **reactive**: it inserts a prefix node into the radix tree *after* a request finishes, and that node competes with every other request for the LRU eviction budget. In co-located RL setups (training + rollout sharing the same GPU), the committed prefix is frequently evicted between the generation pass and the logprob pass, forcing a full re-prefill.

Three specific failure modes:
1. **Inter-pass eviction**: Between the commit/generation pass and the logprob pass, other requests (or the scheduler) can evict the prefix node.
2. **Output pollution**: After each generation, the full output (including reasoning/thinking tokens) is inserted into the radix tree, creating dead branches that waste GPU memory (see issue #22373).
3. **No proactive guarantee**: RadixAttention only provides cache hits *after* the first request completes, so the first request in each group always does a full prefill.

### The RolloutKV Solution: Proactive Prefix Commit + Pin + Reuse-Only

RolloutKV makes prefix caching **proactive** through three mechanisms:

1. **Commit before fanout**: The trainer sends a commit request with `rollout_kv_commit=True` that prefills and inserts the prompt prefix *before* any follower requests are dispatched, guaranteeing a cache hit for all G branches.

2. **Pin against eviction**: `inc_lock_ref` moves the committed node from `evictable_size` to `protected_size`, making it immune to LRU eviction until explicitly released after all followers complete.

3. **No output pollution**: Follower requests use `rollout_kv_reuse_only=True` to skip finished-cache insertion, preventing rollout/reasoning suffixes from filling the radix tree with unreachable entries.

### Data Flow: A Complete GRPO Rollout Step

```
Phase 1: COMMIT (trainer)
  Request: prompt_ids, rollout_kv_commit=True, rollout_kv_expected_followers=8
  Result: prefix KV is prefilled, inserted into radix tree, pinned (protected_size += prompt_len)
  Refcount: _rollout_kv_pin_counts[pin_key] = 8

Phase 2: GENERATION (G=8 rollout branches)
  Request: prompt_ids, rollout_kv_reuse_only=True, rollout_kv_auto_unprotect_on_finish=True
  Prefix match: hits pinned prefix node at >99.6% cache hit rate
  Prefill: only the response suffix (zero tokens for generation start)
  On finish: auto_unprotect decrements refcount by 1
  Refcount after all 8: 0 → dec_lock_ref → node moves back to evictable_size
  No insertion into radix tree (reuse_only=True)

Phase 3: ACTOR LOGPROB SCORING
  Request: prompt_ids + response_ids, rollout_kv_reuse_only=True
  Prefix match: hits pinned prefix node (or evictable node if generation already unpinned)
  Prefill: only the response suffix (~128 tokens)
  No insertion (reuse_only=True)

Phase 4: REFERENCE LOGPROB SCORING
  Same as Phase 3, different extra_key or same if sharing the prefix

Phase 5: EXPLICIT RELEASE (trainer, optional cleanup)
  Request: prompt_ids, rollout_kv_unprotect=True, rollout_kv_unprotect_all=True
  Force-releases any residual refs
```

### Physical Block Sharing Verification

The PR verifies that follower requests' `req_to_token_pool` rows match the committed prefix's physical KV indices, confirming **reference reuse** (not copy). This is a critical correctness property: the same physical GPU memory blocks are shared across all follower requests without duplication.

---

## 2. inc_lock_ref / protected_size: The Pinning Lifecycle

### Existing RadixCache Mechanism (Pre-RolloutKV)

The radix cache has a two-zone memory model:

- **evictable_size**: Nodes with `lock_ref == 0`, subject to LRU eviction. When memory pressure rises, the scheduler calls `evict()` which frees these nodes from the bottom of the priority heap.

- **protected_size**: Nodes with `lock_ref > 0`, immune to eviction. Currently used only for active requests (each request increments lock_ref on its matched prefix node during prefill, decrements on finish).

The `inc_lock_ref` / `dec_lock_ref` functions manage the transition:

```python
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    # Walk from node up to root, incrementing lock_ref on each ancestor
    while node != self.root_node:
        if node.lock_ref == 0:
            # Transition from evictable → protected
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent
    return IncLockRefResult(delta=delta)

def dec_lock_ref(self, node: TreeNode):
    # Walk from node up to root, decrementing lock_ref on each ancestor
    while node != self.root_node:
        if node.lock_ref == 1:
            # Transition from protected → evictable
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
        node.lock_ref -= 1
        node = node.parent
    return DecLockRefResult(delta=delta)
```

Key properties:
- lock_ref walks up to root, protecting ALL ancestor nodes (not just the matched leaf)
- When lock_ref transitions 0→1, the node moves from evictable to protected
- When lock_ref transitions 1→0, the node moves back to evictable
- Multiple refs on the same node are cumulative (lock_ref is an integer counter)

### RolloutKV's Extension of lock_ref

RolloutKV leverages this existing mechanism but adds **persistent, trainer-controlled** pinning that is separate from the transient per-request locking:

- **Transient lock** (existing): Each request increments lock_ref on its matched prefix during `match_prefix`, decrements on `cache_finished_req`. This protects the prefix only while the request is active.

- **Persistent pin** (RolloutKV): The commit request calls `inc_lock_ref` on the matched node *after* insertion, with a refcount equal to `rollout_kv_expected_followers`. This refcount is tracked separately in `_rollout_kv_pin_counts`. The node stays in `protected_size` until ALL followers have released their refs.

The separation is important: the persistent pin is *additive* to transient locks. Even after the commit request's transient lock is released (at `cache_finished_req`), the persistent pin keeps the node protected.

### Pin Key Construction

```python
def _rollout_kv_pin_key(self, key: RadixKey):
    return (key.extra_key, key.is_bigram, tuple(key.token_ids))
```

The pin key is a tuple of (extra_key, is_bigram, token_ids). This means:
- Different extra_keys (e.g., "actor_v42" vs "ref_v42") create different pin entries for the same prompt tokens
- The same prompt with the same extra_key shares the same pin
- Bigram (EAGLE spec decode) is separated from regular tokens

### Pin Lifecycle State Machine

```
State: UNPINNED (lock_ref=0, not in _rollout_kv_pin_counts)
  → [commit request with rollout_kv_commit=True, rollout_kv_protect=True]
  → inc_lock_ref(matched_node)
  → _rollout_kv_pin_counts[pin_key] += expected_followers
  → _rollout_kv_pin_nodes[pin_key] = matched_node
  → _rollout_kv_pin_timestamps[pin_key] = time.monotonic()

State: PINNED (lock_ref>=1, in _rollout_kv_pin_counts with count>0)
  → Protected from eviction. All followers hit cache.
  → [follower finishes with auto_unprotect] → _rollout_kv_pin_counts[pin_key] -= 1
  → [explicit unprotect] → _rollout_kv_release_pin(pin_key, release_count=N)
  → [TTL expiry] → _rollout_kv_release_pin(pin_key, release_all=True)

State: RELEASED (lock_ref decremented, back in evictable_size)
  → _rollout_kv_pin_counts[pin_key] = 0 (popped)
  → _rollout_kv_pin_nodes[pin_key] = None (popped)
  → _rollout_kv_pin_timestamps[pin_key] = None (popped)
  → Node is now evictable, can be reclaimed by LRU
```

### Key Design Decision: Re-match vs Stored Node Reference

In the original PR #24781, the unprotect path re-ran `match_prefix` to find the node to `dec_lock_ref`. In the updated PR #28608, the node reference is **stored** in `_rollout_kv_pin_nodes` at commit time, so release does not need to re-match. This is a correctness improvement: if the radix tree structure changes between commit and unprotect (e.g., node splits during subsequent insertions), re-matching could find a different node or a subtree that doesn't correspond to the original commit.

---

## 3. TTL Eviction Mechanism

### Problem: Stale Pins Leak Memory

Two failure modes that leave pins indefinitely:

1. **Follower crash / OOM kill**: If a follower process is killed before calling `cache_finished_req`, its refcount never reaches zero. The pinned KV pages are held indefinitely, gradually exhausting GPU memory.

2. **`rollout_kv_expected_followers` mismatch**:
   - Over-counted (expected_followers=8 but only 6 followers arrive): residual refcount stays positive indefinitely
   - Under-counted (expected_followers=6 but 8 followers arrive): extra unprotect calls are idempotent (return False, no crash)

### Solution: Lazy TTL Eviction

```python
def _rollout_kv_evict_expired_pins(self) -> int:
    if not self._rollout_kv_pin_timestamps or self._rollout_kv_pin_ttl_seconds <= 0:
        return 0
    now = time.monotonic()
    expired = [
        pk for pk, ts in list(self._rollout_kv_pin_timestamps.items())
        if now - ts > self._rollout_kv_pin_ttl_seconds
    ]
    for pk in expired:
        age = now - self._rollout_kv_pin_timestamps.get(pk, now)
        remaining = self._rollout_kv_pin_counts.get(pk, 0)
        logger.warning(
            "RolloutKV: evicting stale pin %s (age=%.1fs > ttl=%.1fs, "
            "remaining_refcount=%d). Likely cause: follower crash or "
            "rollout_kv_expected_followers mismatch.",
            pk, age, self._rollout_kv_pin_ttl_seconds, remaining,
        )
        self._rollout_kv_release_pin(pk, release_all=True)
    return len(expired)
```

Design choices:
- **Lazy evaluation** (not a background thread): Called at the top of every `cache_finished_req` when `enable_rollout_kv=True`. This avoids thread complexity and ensures eviction happens at a natural scheduling point.
- **time.monotonic()** (not wall clock): Ensures TTL is immune to clock adjustments (NTP, suspend/resume).
- **Default TTL = 600 seconds (10 minutes)**: Reasonable for RL steps that typically take seconds to minutes. Can be set to 0 to disable entirely.
- **WARNING log with forensic data**: Pin key, age, TTL, remaining refcount — enables post-mortem debugging of follower crashes.
- **release_all=True**: Force-releases ALL remaining refs in one call, regardless of count. This is the safe choice for stale pins.

### Integration Point

```python
def cache_finished_req(self, req: Req, is_insert: bool = True):
    custom_params = getattr(req.sampling_params, "custom_params", None)
    custom_params = custom_params if isinstance(custom_params, dict) else {}
    if self.enable_rollout_kv:
        self._rollout_kv_evict_expired_pins()  # ← lazy TTL check
        rollout_kv_commit = bool(custom_params.get("rollout_kv_commit", False))
        rollout_kv_reuse_only = bool(custom_params.get("rollout_kv_reuse_only", False))
        ...
```

The TTL check happens *before* any custom_params processing, ensuring stale pins are cleaned up before new request logic runs.

---

## 4. Relation to verl Weight Sync and GRPO Training

### Direct Targeting of verl-Style RL Workflows

The PR benchmarks are explicitly labeled "verl-style RL" and model the exact verl GRPO workflow:
- Generation: G=8 rollout branches from the same prompt
- Actor logprob: scoring prompt+response with the actor policy
- Reference logprob: scoring prompt+response with the reference policy
- Multi-role: actor + ref + reward + value, multiple rounds

### Integration with verl's HYBRID Sleep/Wake Architecture

verl's HYBRID mode uses a three-phase architecture:
1. **Training phase**: FSDP training, rollout engine sleeps (sleep_level=1 releases KV cache)
2. **Rollout phase**: Rollout engine wakes up (weight sync: base weights + LoRA deltas), generates G responses
3. **Training phase**: Rollout engine sleeps again

RolloutKV operates **within the rollout phase**. The lifecycle:

```
verl HYBRID step lifecycle:
  1. wake_up(tags=["kv_cache"])  ← restores KV cache memory pool
  2. weight sync (base weights + LoRA deltas)
  3. RolloutKV commit (prompt prefix)
  4. RolloutKV G=8 generation followers
  5. RolloutKV actor/ref logprob scoring
  6. RolloutKV unprotect/release
  7. sleep(tags=["kv_cache"])  ← releases ALL KV cache including RolloutKV pins
```

Key interaction: RolloutKV pins are **ephemeral within a rollout phase**. They are created after wake_up and destroyed before sleep. This is the correct lifecycle — pinned KV should not persist across sleep/wake boundaries because:
- sleep releases ALL KV cache memory (including RolloutKV pins)
- wake_up re-allocates the KV memory pool from scratch
- Any attempt to preserve RolloutKV pins across sleep/wake would require a persistence mechanism that doesn't exist

### Weight Reload Boundary Concern (Pattern Family: #28679, #28676)

This is a **critical interaction point**. The MEMORY.md documents a pattern family of state lifecycle mismatches where GPU-resident caches are not invalidated at weight-reload boundaries:

- SGLang #28679: GDN intermittent decode degeneracy (worsens over uptime)
- SGLang #28676: MXFP8 MoE shuffle cache Clobbered (merged July 1)
- vLLM #45552: CuMem sleep/wake missing cuda.synchronize()

RolloutKV adds a **new stateful cache entry** (the pinned prefix) that lives in GPU memory during the rollout phase. If weight reload happens *during* a rollout phase (e.g., LoRA delta update for a new policy version), the pinned prefix KV was computed with the OLD weights and would be stale for the NEW weights.

**Current design handles this correctly**: RolloutKV pins are released at the end of each rollout step, and the next step's commit request computes fresh prefix KV with the updated weights. But this requires that the trainer integration strictly follows the commit→fanout→release lifecycle within a single weight version.

**Risk scenario**: If the trainer sends a commit request, then updates weights (LoRA delta), then sends logprob scoring followers — the followers would reuse prefix KV computed with OLD weights, producing incorrect logprobs. This is a **silent corruption** bug (no error signal, wrong logprobs fed to GRPO advantage computation).

**Mitigation**: The verl integration must ensure weight sync happens BEFORE the RolloutKV commit, or the commit request must be sent with the correct weight version's extra_key. The extra_key field in the pin key provides a natural versioning mechanism: `"actor_v42"` vs `"actor_v43"` would create separate pin entries, preventing stale reuse.

### verl Integration API Mapping

The RolloutKV API is exposed through `sampling_params.custom_params`, which maps to verl's rollout engine generate calls:

```python
# verl rollout engine integration
# Step 1: commit prefix
rollout_output = engine.generate(
    prompt_ids,
    sampling_params=SamplingParams(
        custom_params={
            "rollout_kv_commit": True,
            "rollout_kv_expected_followers": group_size * 2,  # G rollout + G logprob
        },
        extra_key=f"actor_step_{step_idx}",
    ),
)

# Step 2: G rollout branches
for i in range(group_size):
    rollout_output = engine.generate(
        prompt_ids,
        sampling_params=SamplingParams(
            temperature=0.8,
            custom_params={
                "rollout_kv_reuse_only": True,
                "rollout_kv_auto_unprotect_on_finish": True,
            },
            extra_key=f"actor_step_{step_idx}",
        ),
    )

# Step 3: actor logprob
logprobs = engine.generate(
    prompt_ids + response_ids,
    sampling_params=SamplingParams(
        custom_params={
            "rollout_kv_reuse_only": True,
        },
        extra_key=f"actor_step_{step_idx}",
    ),
)

# Step 4: ref logprob (same extra_key if same prefix)
ref_logprobs = engine.generate(
    prompt_ids + response_ids,
    sampling_params=SamplingParams(
        custom_params={
            "rollout_kv_reuse_only": True,
        },
        extra_key=f"ref_step_{step_idx}",  # different extra_key for ref model
    ),
)

# Step 5: explicit release
engine.generate(
    prompt_ids,
    sampling_params=SamplingParams(
        custom_params={
            "rollout_kv_unprotect": True,
            "rollout_kv_unprotect_all": True,
        },
        extra_key=f"actor_step_{step_idx}",
    ),
)
```

### RTX 4090 Relevance

For RTX 4090 GRPO training:
- **Memory budget**: 24 GiB total, ~19 GiB for KV cache (with mem_fraction=0.88)
- **RolloutKV adds only +4 MiB peak HBM** (verified in LongBench benchmark)
- **Pinned prefix cost**: 8192 tokens * page_size 32 * KV bytes per token ≈ small relative to 19 GiB pool
- **Generation speedup**: 1.65x for 8K prompt, 1.25x for 16K prompt
- **Logprob scoring speedup**: 22-41x, which is the dominant win

For RTX 4090 with Qwen3-8B (the viable model size):
- 8B model is more memory-efficient, leaving more room for KV cache
- RolloutKV's value scales with prompt length / model size ratio
- Short prompts (<512 tokens) see minimal benefit on RTX 4090
- Long prompts (4K-8K tokens) with G=8 should see 1.5-2x generation speedup

**Key constraint**: RolloutKV requires `--enable-rollout-kv` server flag. This is off by default with "zero intrusion" — no overhead when disabled. The flag must be passed to SGLang server startup in verl's rollout engine configuration.

---

## 5. The 22-41x Logprob Speedup: Benchmark Analysis

### Benchmark Setup

- Hardware: Single A100 80 GB
- Model: Qwen3-32B
- Page size: 32
- No attention kernel or model math changes

### Logprob Scoring Speedup Calculation

The "scoring-only speedup" measures the time spent purely on logprob scoring (excluding the commit request time):

| Input | Output | Baseline total | RolloutKV total | Scoring only | Total speedup | Scoring speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 128 | 7.665 s | 2.813 s | 0.339 s | **2.72x** | **22.64x** |
| 16384 | 128 | 17.329 s | 6.107 s | 0.413 s | **2.84x** | **41.91x** |

Calculation:
- 8192 input: 7.665 / 0.339 = 22.64x scoring-only speedup
- 16384 input: 17.329 / 0.413 = 41.91x scoring-only speedup

The scoring-only number (0.339s / 0.413s) is remarkably small because:
- With RolloutKV, logprob scoring requests hit the pinned prefix at >99.6% cache hit rate
- Only the 128-token response suffix needs to be prefilled
- The 8192/16384-token prompt prefix is reused from the pinned KV cache with zero recomputation

### Cache Hit Evidence

| Input | Cache hit | Cached tokens | protected_size |
|---:|---:|---:|---:|
| 8192 | 99.62% | 65280/65528 | 8192 pinned |
| 16384 | 99.81% | 130816/131064 | 16384 pinned |

The >99.6% cache hit rate means nearly the entire prompt prefix is reused. The small miss (0.38% / 0.19%) is likely due to page-alignment boundary effects (last page of prefix may not be fully utilized).

### Why 41.91x > 22.64x

The speedup scales with prompt length because:
- Longer prompts = more prefill time saved per scoring pass
- 16384-token prompt baseline prefill is ~2x longer than 8192-token prompt
- But the RolloutKV scoring time (0.339s vs 0.413s) only increases by ~22% because it's dominated by the fixed response suffix (128 tokens)
- So the ratio grows: 17.329/0.413 > 7.665/0.339

### Multi-Role Speedup

For 4 roles (actor/ref/reward/value) with 3 rounds, the logprob speedup compounds:

| Input | G | Output | Gen speedup | Logprob speedup | Scoring speedup |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8 | 128 | 1.64x | 8.03x | **22.83x** |
| 8192 | 16 | 128 | 1.56x | 8.05x | **22.97x** |
| 16384 | 8 | 128 | 1.26x | 9.48x | **42.43x** |
| 16384 | 16 | 128 | 1.21x | 9.48x | **42.56x** |

The 8.03x total logprob speedup comes from: 4 roles * 3 rounds = 12 scoring passes, each with 22.83x individual speedup, but the commit overhead (one prefill per step) dilutes the aggregate.

### Full RL Step Speedup

4 prompts * G=4 * 5 phases * prompt=8192 * mem_fraction=0.88, 3-run median:
- Baseline: 45.465 s
- RolloutKV: 35.759 s
- **1.27x full-step speedup**, rollout-phase: 1.83x

The full-step speedup (1.27x) is lower than the per-phase speedups because:
- Training phase is unchanged (FSDP backward, optimizer step)
- Only the rollout phase benefits from RolloutKV
- The rollout phase is ~60% of total step time (estimated from 1.27x vs 1.83x)

### Break-even and Boundary Cases

- **G=4, short output**: Near break-even. Commit overhead (~0.5s) nearly offsets the prefill savings per follower.
- **Decode-dominated** (prompt=512, output=16384): 1.002x total, 3.4x prefill-only. RolloutKV only saves prefill time, not decode time. When decode is >99% of wall time, the end-to-end gain is negligible.

---

## 6. Implementation Scope and Code Changes

### Changed Files (6)

| File | +/− | Change Description |
|---|---|---|
| `python/sglang/srt/mem_cache/radix_cache.py` | +152/-5 | Core: pin tracking dicts, commit/pin/reuse-only/TTL logic in cache_finished_req |
| `python/sglang/srt/managers/scheduler.py` | +253/-0 | Extracted `init_cache_with_memory_pool()` method, passing enable_rollout_kv params to CacheInitParams |
| `python/sglang/srt/mem_cache/cache_init_params.py` | +2/-0 | Added `enable_rollout_kv: bool = False` and `rollout_kv_pin_ttl_seconds: float = 600.0` fields |
| `python/sglang/srt/server_args.py` | +19/-0 | CLI args `--enable-rollout-kv` and `--rollout-kv-pin-ttl-seconds` |
| `test/.../test_radix_cache_unit.py` | +308/-0 | Unit tests: commit/pin/reuse-only/TTL/mismatch idempotency/auto-unprotect/unprotect-all |
| `docs/advanced_features/sglang_for_rl.md` | +34/-0 | Usage documentation |

### No Changes To

- Attention kernels (FA3, FlashInfer)
- CUDA Graph capture/replay
- Sampling logic
- Model math
- `match_prefix` / `req_to_token_pool` physical-index path

This is a **cache-layer-only** change, which is a major design strength. The attention computation reuses the same physical KV blocks through the existing radix cache prefix-matching mechanism — no kernel modifications needed.

### Scheduler.py Refactoring Note

The +253/-0 addition in scheduler.py is a method extraction: `init_cache_with_memory_pool()` pulls the cache initialization logic (previously inline in `init_model_worker()`) into a separate method. This allows the new `enable_rollout_kv` and `rollout_kv_pin_ttl_seconds` fields to be passed cleanly through `CacheInitParams`. The actual RolloutKV-specific lines in this method are just 2:

```python
enable_rollout_kv=server_args.enable_rollout_kv,
rollout_kv_pin_ttl_seconds=server_args.rollout_kv_pin_ttl_seconds,
```

The rest is existing cache initialization code (SWA, Mamba, HiCache, Unified, LMCache, etc.) that was previously in `init_model_worker()`. This refactoring is substantial but tangential to the RolloutKV feature itself.

### Core radix_cache.py Changes

Three new internal data structures:
- `_rollout_kv_pin_counts: defaultdict(int)` — refcount per pin key
- `_rollout_kv_pin_nodes: dict` — TreeNode reference per pin key (avoids re-matching on release)
- `_rollout_kv_pin_timestamps: dict` — commit time per pin key (for TTL eviction)

Four new methods:
- `_rollout_kv_pin_key(key)` — constructs the pin key tuple from RadixKey
- `_rollout_kv_pin_refcount_from_params(custom_params)` — extracts expected_followers/pin_refcount
- `_rollout_kv_release_pin(pin_key, release_count, release_all)` — decrements refcount, calls dec_lock_ref when count hits 0
- `_rollout_kv_evict_expired_pins()` — lazy TTL eviction at cache_finished_req entry

Modified `cache_finished_req()`:
- Added TTL eviction check at entry
- Added RolloutKV-specific parameter extraction (commit, reuse_only, unprotect, auto_unprotect)
- Modified `is_insert` logic: `disable_finished_insert` is bypassed for commit requests; reuse_only and unprotect force `is_insert=False`
- Modified token_ids selection: commit/unprotect use only `origin_input_ids[:commit_len]`, not the full `origin_input_ids + output_ids`
- Added pin creation on commit: `inc_lock_ref` + populate pin tracking dicts
- Added pin release on unprotect: `_rollout_kv_release_pin`
- Added auto-unprotect on finish: `_rollout_kv_release_pin` with release_count=1

---

## 7. API Reference

### Server Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--enable-rollout-kv` | bool | False | Enable RolloutKV prefix pinning. Zero overhead when disabled. |
| `--rollout-kv-pin-ttl-seconds` | float | 600.0 | TTL for stale pins. Set to 0 to disable TTL eviction. |

### Custom Params (per-request)

| Field | Type | Default | Description |
|---|---|---|---|
| `rollout_kv_commit` | bool | False | Commit only the prompt prefix; pin it. |
| `rollout_kv_commit_len` | int | None | Tokens to commit. Defaults to page-aligned len(input_ids). |
| `rollout_kv_protect` | bool | True | Pin the committed node with inc_lock_ref. |
| `rollout_kv_pin_refcount` | int | 1 | Refs to add. Alias: `rollout_kv_expected_followers`. |
| `rollout_kv_reuse_only` | bool | False | Skip finished-cache insertion (prevents suffix pollution). |
| `rollout_kv_auto_unprotect_on_finish` | bool | False | Release one ref when request finishes. |
| `rollout_kv_unprotect` | bool | False | Explicitly release the pin. |
| `rollout_kv_unprotect_count` | int | 1 | Refs to release on explicit unprotect. |
| `rollout_kv_unprotect_all` | bool | False | Force-release all remaining refs. |

### Typical RL Flow (from PR description)

```python
# Step 1: commit prompt prefix once (trainer)
engine.generate(prompt, sampling_params=SamplingParams(custom_params={
    "rollout_kv_commit": True,
    "rollout_kv_expected_followers": 8,  # 8 rollout responses
}, extra_key="actor_v42"))

# Step 2: G=8 rollout branches — all hit the pinned prefix
for seed in seeds:
    engine.generate(prompt, sampling_params=SamplingParams(
        n=1, temperature=0.8,
        custom_params={
            "rollout_kv_reuse_only": True,
            "rollout_kv_auto_unprotect_on_finish": True,
        }, extra_key="actor_v42"))

# Step 3: actor/ref logprob scoring — near-zero prefill
engine.generate(prompt + response, sampling_params=SamplingParams(
    custom_params={"rollout_kv_reuse_only": True},
    extra_key="actor_v42"))
```

---

## 8. Correctness Verification

### Exact Output Match

Qwen3-32B, prompt=4096, output=1028, G=8:
- Output length: 1028 tokens (both baseline and RolloutKV)
- First token IDs: exact match
- Physical block sharing verified: req_to_token_pool rows for followers match committed prefix's physical KV indices

### LongBench-v2 Accuracy

Qwen3-32B, 7 examples, context=8192, G=4:
- LongBench score: 0.4286 (both) — delta = 0.000
- Extracted answer match: 7/7 (both) — same
- Peak HBM: 66.656 GiB → 66.660 GiB — +4 MiB overhead

The +4 MiB overhead comes from the pin tracking data structures (defaultdict + dict + dict), which are negligible.

---

## 9. Pattern Family Analysis

RolloutKV belongs to several documented pattern families:

### State Lifecycle Mismatch Pattern Family

RolloutKV introduces a **new stateful entry** (pinned prefix KV) with a well-defined lifecycle (commit → fanout → release). This is the correct approach — it avoids the intermittent degeneracy seen in #28679 (GDN) and #28676 (MoE shuffle cache) where state is not properly invalidated at boundary transitions.

The key difference: RolloutKV's lifecycle is **trainer-controlled** (explicit commit and release), not **implicit** (relying on request completion to update state). This makes it more robust against the degeneracy pattern.

However, RolloutKV still has a boundary risk: if weight reload happens between commit and release, the pinned prefix KV becomes stale. The TTL mechanism mitigates this by auto-releasing pins after 600 seconds, but a faster weight update cycle (e.g., LoRA delta per step) requires explicit unprotect at the correct boundary.

### Output Pollution Pattern Family (#22373)

The `rollout_kv_reuse_only=True` flag directly addresses issue #22373: reasoning/thinking tokens from rollout output are NOT inserted into the radix tree, preventing dead branches. This is a targeted fix that complements the broader "strip thinking tokens from cache" proposal in #22373.

### Weight Reload Boundary Pattern Family

Related to:
- vLLM #45552: CuMem sleep/wake missing cuda.synchronize()
- SGLang #28679: GDN intermittent degeneracy
- SGLang #28676: MoE shuffle cache clobbered (MERGED)
- verl #6794: delta weight sync

RolloutKV adds a GPU-resident cache entry (pinned prefix KV) that MUST be invalidated at weight-reload boundaries. Current design handles this correctly (pins released before sleep, recomputed after wake), but the integration must ensure strict lifecycle ordering.

---

## 10. Comparison with Previous PR #24781

| Feature | #24781 (closed) | #28608 (current) |
|---|---|---|
| TTL eviction | None | Yes (default 600s, lazy evaluation) |
| Pin refcount | Single ref per commit | N refs via rollout_kv_expected_followers |
| Auto-unprotect | None | rollout_kv_auto_unprotect_on_finish |
| Bulk release | None | rollout_kv_unprotect_all |
| Node reference storage | Re-match on release (dec_lock_ref re-runs match_prefix) | Stored in _rollout_kv_pin_nodes (avoids re-match) |
| Server flag | None (custom_params only) | --enable-rollout-kv (zero intrusion when off) |
| Pin timestamp tracking | None | _rollout_kv_pin_timestamps (for TTL) |
| Scheduler refactoring | Minimal | init_cache_with_memory_pool() extraction (+253 lines) |
| Documentation | None | sglang_for_rl.md section added |

The TTL eviction is the most important addition: it protects against the "stale pin leaks memory forever" failure mode that was unhandled in #24781.

---

## 11. Critical Assessment and Open Questions

### Strengths

1. **Zero intrusion when disabled**: --enable-rollout-kv defaults to False, all RolloutKV code paths are gated. No overhead for non-RL users.
2. **Cache-layer-only change**: No attention kernel or model math changes. The reuse mechanism leverages the existing radix cache prefix matching path.
3. **Physical block sharing**: Verified by req_to_token_pool row comparison. No copies, no duplication.
4. **TTL eviction**: Lazy, monotonic-time-based, with forensic logging. Addresses the stale pin problem comprehensively.
5. **Idempotent unprotect**: Over-counted expected_followers leaves residual refs (handled by TTL); under-counted extra unprotects return False (no crash).
6. **Output pollution prevention**: rollout_kv_reuse_only directly addresses #22373.

### Concerns

1. **Scheduler.py refactoring scope**: The +253/-0 method extraction is substantial and tangential to RolloutKV. Reviewers may push for separating the refactoring from the feature PR.

2. **No review comments yet**: Zero review comments as of 2026-06-18. This is a +768-line PR with significant architectural implications for RL integration — needs thorough review.

3. **CI status**: Both CI runs are failing (x status). The PR mentions `flashinfer.fused_moe` import error blocking local pytest, but CI failures need resolution.

4. **Weight-reload boundary**: RolloutKV does not explicitly handle weight reload during a pinned prefix's lifetime. The extra_key provides implicit versioning, but there's no explicit mechanism to invalidate pins when weights change.

5. **Multi-GPU / TP scenarios**: Benchmarks are on single A100. RolloutKV's behavior under TP (where radix cache is per-device) needs verification. The `tp_cache_group` parameter exists in CacheInitParams but RolloutKV doesn't modify the TP synchronization path.

6. **RadixCacheCpp and UnifiedRadixCache compatibility**: The PR only modifies `radix_cache.py`. The C++ radix tree implementation (`RadixCacheCpp`) and the `UnifiedRadixCache` do not have RolloutKV support. This means `--enable-rollout-kv` is incompatible with `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` and `SGLANG_ENABLE_UNIFIED_RADIX_TREE`.

7. **Hybrid models**: SWA/Mamba models use specialized radix caches (SWARadixCache, MambaRadixCache) that are not modified by this PR. RolloutKV is incompatible with hybrid SWA/SSM models.

### Open Questions for verl Integration

1. How does verl's rollout engine pass custom_params to SGLang? The API uses `sampling_params.custom_params` which is a dict — verl needs to populate this dict in its SGLang integration layer.

2. Does verl's `extra_key` mechanism already provide versioning per step? If so, RolloutKV's pin key naturally tracks step versions, preventing stale reuse.

3. What happens when verl's separate_async trainer sends rollout requests to a remote SGLang engine? RolloutKV's commit/unprotect lifecycle must be managed by the trainer process, which communicates with SGLang via HTTP/ZMQ. The commit request is a generate call with custom_params — this should work with the existing API.

4. For RTX 4090 with Qwen3-8B, what's the practical speedup? The benchmarks use Qwen3-32B on A100. The ratio of prompt prefill time to total step time changes with model size and hardware. Smaller models (8B) have faster prefill, so the relative savings may be smaller.

---

## 12. RTX 4090 Deployment Assessment

### Viability: YES

RolloutKV is viable on RTX 4090 for GRPO training with Qwen3-8B:
- Memory: +4 MiB overhead (negligible on 24 GiB)
- Speedup: Expected 1.3-1.8x generation for 4K-8K prompts with G=8
- Logprob scoring: Expected 10-20x for 4K-8K prompts
- No kernel changes, no model math changes

### Required Configuration

```bash
# SGLang server startup
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --enable-rollout-kv \
    --rollout-kv-pin-ttl-seconds 600 \
    --mem-fraction-static 0.88 \
    --disable-radix-cache=False  # must have radix cache enabled
```

### Constraints

- Incompatible with: RadixCacheCpp, UnifiedRadixCache, SWA models, Mamba models
- Requires: RadixAttention enabled (default)
- Best for: Long prompts (>=2048), short-medium output, G>=4, multi-role scoring
- Minimal benefit: Short prompts (<512), decode-dominated output

### Integration with verl HYBRID mode

```
1. wake_up(tags=["kv_cache"])  # SGLang wakes, KV pool re-allocated
2. RolloutKV commit            # prefix pinned
3. G rollout                   # prefix reused
4. logprob scoring             # prefix reused (22-41x speedup)
5. RolloutKV unprotect_all     # pin released
6. sleep(tags=["kv_cache"])   # ALL KV freed (including former pin)
```

This lifecycle is correct and safe for RTX 4090 HYBRID mode.

---

## Summary

RolloutKV is a well-designed, cache-layer-only feature that directly targets the dominant latency bottleneck in RL rollout (repeated prompt prefill). The core mechanism — proactive prefix commit + lock_ref pinning + reuse-only followers — is architecturally clean, leveraging SGLang's existing radix cache infrastructure without modifying attention kernels or model math. The TTL eviction mechanism addresses the stale-pin failure mode comprehensively. The 22-41x logprob scoring speedup is the headline result, driven by >99.6% cache hit rate on pinned prefix KV.

The main risks are: (1) weight-reload boundary correctness (pins must not persist across weight updates), (2) incompatibility with alternative radix cache implementations (Cpp, Unified, SWA, Mamba), and (3) the large scheduler.py refactoring mixed into the feature PR. For verl RTX 4090 GRPO integration, RolloutKV is viable and beneficial, especially for long-prompt workloads with multi-role logprob scoring.
