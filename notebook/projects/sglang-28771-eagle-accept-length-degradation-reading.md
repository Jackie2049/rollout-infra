# SGLang #28771 EAGLE Accept Length Degradation — Deep Reading

> Created: 2026-06-20 | Priority: ★★★★★★★★ CRITICAL for RTX 4090 GRPO
> Source: Background agent deep reading + cross-framework analysis

## Issue Status
- **Number**: #28771
- **Title**: "EAGLE speculative decoding accept_length continuously degrades over time with HiCache + Mooncake + DSA (GLM-5.1)"
- **Opened**: 2026-06-20
- **State**: OPEN (0 comments, 0 assignees, no triage yet)
- **Framework**: SGLang

## Bug Description

Running GLM-5.1-FP8 with EAGLE speculative decoding + HiCache (Mooncake RDMA) + DSA (NSA), the `accept_length` metric smoothly and monotonically declines from ~3.4 to ~1.9 over ~2 hours. No crashes, no NaN, no garbled output — just silent progressive throughput loss.

### Quantitative Data (2,342 Decode Batches)

| Time Window | Avg Accept Length | Avg Accept Rate | Avg Token Usage |
|:-----------:|:-----------------:|:---------------:|:---------------:|
| 14:49-15:19 | **3.33** | ~78% | 0.36 |
| 15:48-16:11 | **3.07** | ~69% | 0.40 |
| 16:11-16:32 | **2.84** | ~61% | 0.44 |
| 16:32-16:50 | **2.44** | ~48% | 0.46 |
| 16:50-17:02 | **1.90** | ~30% | 0.47 |

**Correlation**: Near-perfect inverse correlation between token_usage and accept_length.

### Reproduction Environment
- SGLang: main branch (2026-06-17)
- Model: GLM-5.1-FP8 (embedded EAGLE)
- Hardware: 4x GPU, TP=4
- Key flags: `--attention-backend nsa`, `--enable-hierarchical-cache`, `--hicache-storage-backend mooncake`, `--speculative-algorithm EAGLE`, `--speculative-num-steps 3`

## Root Cause Analysis (5 hypotheses, ordered by likelihood)

### 1. HiCache async layer transfer race with draft model forward (MOST LIKELY)
Similar to #22811 NSA indexer race, but subtler. When HiCache swaps KV pages between GPU/Host, the draft model may read partially-loaded KV pages during forward pass. Not complete corruption (no garbled text), but slight numerical imprecision → gradually reducing prediction accuracy.

### 2. Draft model KV cache eviction during HiCache pressure
As token_usage increases, HiCache evicts KV pages from GPU memory. If draft model's required KV pages are evicted and need async reload, draft model may compute attention on stale/incomplete state.

### 3. Attention metadata refactor regression (#28767)
The coordinated 6-PR refactor (#28763-28768) changes how attention metadata is computed. PR #28767 affects EAGLE worker metadata handling. If the refactor changed metadata semantics affecting draft model forward pass, accept_length would degrade smoothly.

### 4. NSA/DSA sparse index stale reads
Under increasing memory pressure, NSA indexer reads may get stale data from pages mid-transfer in HiCache. PR #22811 fixed explicit race for target model; draft model may have subtler race.

### 5. Sampling parameter interaction
`temperature=1.0, top_p=0.95` with draft topk=1: deterministic draft vs stochastic target verification → context drift accumulates.

## Key Code References

1. **eagle_worker.py**: EAGLE worker shares `req_to_token_pool` and `token_to_kv_pool_allocator` with target worker (line 140-142). `maybe_evict_swa()` call (line 596) during draft preprocessing may trigger HiCache swaps.

2. **metrics_reporter.py** (line 711): `accept_length = spec_num_accept_tokens / spec_num_forward_ct`

3. **hicache_spec_storage_common.py**: Existing test infrastructure with `min_expected_accept_length = 7.0`. BUT: tests only check point-in-time, NOT degradation over time.

## Connection to #28679 (GDN Intermittent Degeneracy)

Both belong to the **Weight Reload State Lifecycle Mismatch Pattern Family**:

| # | Framework | Issue | Root Cause | Platform |
|---|-----------|-------|-----------|----------|
| 5 | SGLang | #28679 | GDN intermittent decode degeneracy | NVIDIA |
| 7 | SGLang | #28771 | EAGLE accept_length degradation | NVIDIA |

Common theme: **dynamic state degeneracy under sustained workload/memory pressure** — not crashes, not NaN, just degraded quality or throughput.

## RTX 4090 GRPO Impact

**CRITICAL for RTX 4090 GRPO rollout**:
1. Accept_length = throughput. Degradation from 3.4→1.9 = 44% throughput loss
2. GRPO rollout = 69.2% bottleneck → spec decode throughput directly impacts training speed
3. EAGLE worker shares KV pool infrastructure with target → both compete for same 24 GiB space
4. verl HYBRID sleep/wake cycle compounds the issue: each wake cycle starts fresh but KV cache state may persist

## Adjacent PRs

| PR | Title | Status | Relevance |
|----|-------|--------|-----------|
| #22811 | NSA indexer race fix | MERGED | Same pattern family, but this is subtler |
| #28767 | Attention metadata refactor | OPEN | Possible regression source for EAGLE |
| #28752 | DSA indexer memory budgeting | OPEN | OOM on startup with host_to_device_ratio > 1 |
| #28753 | DSA indexer computation order | OPEN | Position scoring before KV commit |
| #28754 | Speculative KV-commit bookkeeping | OPEN | Directly related to #28771 |

## SGLang EAGLE Issue Landscape (20+ issues)

EAGLE speculative decoding faces systematic stability challenges:
- #27367: Triton MLA decode crashes with EAGLE on asymmetric KV dims
- #26399: GLM-5.1 TP8 EAGLE verify hangs when KV pool near-full
- #25563: GLM-5-NVFP4 + EAGLE on B300 crashes at draft graph capture
- #24747: DSv4 attention backend assertion fails on EAGLE draft path
- #24440: Accuracy issue with EAGLE + aiter's unified attention

## Immediate Actions

1. **A/B test**: Disable HiCache → confirm root cause
2. **Add continuous monitoring**: Extend test suite to check accept_length degradation over time
3. **Investigate PR #28767**: Check if metadata refactor introduced regression
4. **Add layer_transfer_counter synchronization**: For draft model KV reads during HiCache swaps
5. **RTX 4090 workaround**: Monitor accept_length during GRPO rollout, restart engine if < 2.0 threshold

## Theoretical Insight

acceptance_rate = 1 - D_TV(p, q) where p = target distribution, q = draft distribution.
Any subtle draft model degradation → increased distribution divergence → reduced acceptance → compound over time.
