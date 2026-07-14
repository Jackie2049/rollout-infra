# SGLang RolloutKV: Prefix KV Cache Pinning for RL Rollout — Deep Reading

Based on PR #28608 by sgl-project/sglang (+768/-5, replaces #24781).  
Server flag: `--enable-rollout-kv`.  
TTL eviction: `--rollout-kv-pin-ttl-seconds`.

## Problem

RL rollouts repeatedly prefill the **same long prompt**:
1. Once per generation sample (G times)
2. Once for actor logprob
3. Once for reference logprob
4. Once per reward/value scoring pass

Without RolloutKV, SGLang's RadixAttention **reactively** reuses prefix KV — but nodes compete with every other request for LRU eviction. In co-located RL (training + rollout on same GPU), the committed prefix is **frequently evicted** between generation pass and logprob pass, forcing full re-prefill.

## Solution: Proactive Commit-and-Pin

RolloutKV makes prefix reuse **proactive** instead of reactive:

1. **Commit before fanout.** Prefix committed AND pinned before any follower request — guarantees cache hit for all G branches.
2. **Pin against eviction.** `inc_lock_ref` moves node into `protected_size` — immune to LRU eviction until explicitly released.
3. **No output pollution.** Followers skip finished-cache insertion — prevents long rollout/reasoning suffixes from filling radix tree with unreachable entries.

## Core Mechanism (RadixCache)

### `inc_lock_ref(node)`:
```
while node != root_node:
    if node.lock_ref == 0:
        evictable_size -= len(node.key)
        protected_size += len(node.key)
    node.lock_ref += 1
    node = node.parent
```

### `dec_lock_ref(node)`:
```
while node != root_node:
    if node.lock_ref == 1:
        evictable_size += len(node.key)
        protected_size -= len(node.key)
    node.lock_ref -= 1
    node = node.parent
```

- Walks from node to root (full prefix path)
- `lock_ref=0 → protected_size` on first increment
- `lock_ref=1 → evictable_size` on last decrement
- Reference counting allows multiple concurrent consumers

## Performance Numbers (Qwen3-32B, A100 80GB)

| Scenario | Input | Output | G | Generation | Logprob Total | Logprob Scoring |
|----------|-------|--------|---|------------|---------------|-----------------|
| actor+ref+rollout | 8192 | 128 | 8 | **1.65x** | **2.72x** | **22.64x** |
| actor+ref+rollout | 16384 | 128 | 8 | **1.25x** | **2.84x** | **41.91x** |
| 4 roles × 3 rounds | 8192 | 128 | 8 | **1.64x** | **8.03x** | **22.83x** |

The logprob scoring-only speedup (22-41x) is transformative — the main GRPO bottleneck is repeated prompt prefill for logprob computation.

## API Design

### Server Flags
- `--enable-rollout-kv` — default off, zero intrusion when disabled
- `--rollout-kv-pin-ttl-seconds` — stale pin TTL (default 600s, 0=disable)

### Custom Params (via `sampling_params.custom_params`)

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `rollout_kv_commit` | bool | False | Commit+pint prompt prefix |
| `rollout_kv_commit_len` | int | None | Tokens to commit (default=page-aligned len) |
| `rollout_kv_protect` | bool | True | Pin committed node with inc_lock_ref |
| `rollout_kv_pin_refcount` | int | 1 | Number of refs (=expected followers) |
| `rollout_kv_reuse_only` | bool | False | Skip finished-cache insertion |
| `rollout_kv_auto_unprotect_on_finish` | bool | False | Release one ref on request finish |
| `rollout_kv_unprotect` | bool | False | Explicitly release pin |
| `rollout_kv_unprotect_count` | int | 1 | Refs to release |
| `rollout_kv_unprotect_all` | bool | False | Force-release all remaining refs |

### Typical RL Flow

```python
# Step 1: commit prompt prefix once
engine.generate(prompt, sampling_params=SamplingParams(custom_params={
    "rollout_kv_commit": True,
    "rollout_kv_expected_followers": 8,
}))

# Step 2: G=8 rollout — all hit pinned prefix
for seed in seeds:
    engine.generate(prompt, sampling_params=SamplingParams(
        n=1, temperature=0.8,
        custom_params={"rollout_kv_reuse_only": True,
                       "rollout_kv_auto_unprotect_on_finish": True}))

# Step 3: actor/ref logprob — near-zero prefill
engine.generate(prompt + response, sampling_params=SamplingParams(
    custom_params={"rollout_kv_reuse_only": True}))
```

## Reliability: TTL-based Stale Pin Eviction

Two failure modes handled:

**1. Follower crash/OOM**: Refcount never reaches zero → pinned pages held indefinitely.  
**Fix**: `_rollout_kv_pin_timestamps` records commit time. `_rollout_kv_evict_expired_pins()` called lazily at top of every `cache_finished_req`. No background thread needed.

**2. `expected_followers` mismatch**:
- Over-counted: residual refcount stays positive → TTL eviction cleans up
- Under-counted: extra unprotect calls are **idempotent** (return False, no crash)

## GRPO on RTX 4090 Implications

- **22-41x logprob speedup** directly translates to faster GRPO training iterations
- **Memory tradeoff**: `protected_size` takes memory from evictable cache. On 24GB 4090, must limit total pinned size
- **TTL mechanism** prevents memory leaks when training crashes
- **Integration with verl/TRL**: The custom_params API needs trainer-side support
- **Comparison to vLLM weight sync**: RolloutKV addresses a different bottleneck (prefill compute) vs weight sync (weight transfer). Both are needed for efficient co-located GRPO
- **rollout_kv_reuse_only** prevents rollout suffix pollution — critical for long-reasoning models (Qwen3.5 think, DeepSeek R1)

## Implementation Status

- PR #28608: OPEN, 1 comment (Gemini code-assist bot quota message)
- No human reviewer comments yet
- Replaces #24781, clean 5-commit history
- TTL eviction added 2026-06-18 (after initial submission)

## Connections

- Same pattern family as **verl HYBRID sleep/wake** (both address GRPO prefill repetition)
- Complements **vLLM #45552 cumem sync** (weight reload memory safety)
- Complements **SGLang #28679 GDN degeneracy** (both are decay-vs-reload lifecycle issues)
- Enables **cross-framework partial-wake safety** (avoids full weight reload in rollout phase)

