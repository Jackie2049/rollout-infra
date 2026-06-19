# SGLang #28695 — ReplaySSM Ring Spec-Verify Deep Reading

> 2026-06-19 | PR #28695 OPEN | Author: yuan-luo (COLLABORATOR) | CI running
> RFC: #28511 (Part B) | Part A: #28451 (decode kernel) | Related: #27658 (compact spec cache)
> ★★★★★★★★ MAJOR GDN spec-decode optimization: eliminates ~12 GB `intermediate_ssm` per-draft snapshots
> ★★★★★★★★ +13.1% throughput, -11.8% TPOT at batch 32 — bandwidth-bound kernel optimization
> ★★★★★★★★ Mathematically lossless (GSM8K parity 0.892 vs 0.887 — within noise)
> ★★★★★★★★ RTX 4090 RELEVANCE: GDN-only, linear-chain only → NEXTN/MTP viable on SM89, but TP4 measurement on H20

---

## 1. PR Metadata

```
PR Number:         #28695
Title:             [GDN] Support ReplaySSM Ring Spec-Verify
Author:            yuan-luo (COLLABORATOR, Dao AI Lab / SGLang team)
State:             OPEN (draft=False, CI running)
Created:           2026-06-19T01:10:53Z
Labels:            speculative-decoding, run-ci, run-ci-extra, linear-attention
Requested Reviewers: HaiShaw, kaixih
Comments:          1 (gemini-code-assist quota message only, no substantive review yet)
Files Changed:     6
  + gdn_replayssm_spec_decode.py     +741/-0 (NEW — core kernel)
  + gdn_backend.py                   +117/-16 (MODIFIED — dispatch + _replayssm_target_verify)
  + memory_pool.py                   +147/-3  (MODIFIED — ring allocation, SpeculativeState fields)
  + model_runner_kv_cache_mixin.py   +2/-0    (MODIFIED — config passthrough)
  + server_args.py                   +33/-0   (MODIFIED — CLI flags)
  + spec_utils.py                    +57/-0   (MODIFIED — commit path)
Total:             +1097/-19 lines
Merge Commit:      9817994661c6d736ba6a4eed7965cf38b9557216 (pending)
```

---

## 2. Technical Design: What is ReplaySSM and How Does It Work

### 2.1 Origin and Context

ReplaySSM originates from **Dao AI Lab (2026)** — [dao-lab.ai/blog/2026/replayssm](https://dao-lab.ai/blog/2026/replayssm). The vLLM reference implementation lives at `github.com/Johnny-Liou/ReplaySSM` (commit `3c85112`), with files `fused_recurrent_replayssm.py` (decode) and `gdn_replayssm_spec_decode.py` (spec). This PR (Part B of RFC #28511) ports the spec-decode portion to SGLang.

### 2.2 The Problem It Solves

GDN (Gated DeltaNet, Qwen3.5/Qwen3-Next lineage) and KDA (Kimi Delta Attention) maintain a fixed-size recurrent state `S` of shape `[HV, V, K]` per slot per layer (~1 MB in bf16 for HV=32, K=V=128). The standard spec-decode **target-verify** path:

1. For **every draft token**, writes the full `[V, K]` recurrent state to `intermediate_ssm`
2. This snapshot buffer enables rollback on rejected drafts
3. But it costs **~12 GB** for Qwen3.5-35B-A3B (70 layers * 64 heads * 4 draft tokens * [128,128] states)
4. And at batch >= 64, the verify forward runs at **75-79% of peak HBM bandwidth** — bandwidth-bound

Both problems (memory footprint and bandwidth saturation) are exactly what ReplaySSM eliminates.

### 2.3 Core Algorithm: ReplaySSM Mathematics

Single head, single decode step (baseline packed decode):
```
alpha = exp(g)
d = beta * (v - alpha * (S * k))       # residual update
o = alpha * (S * q) + d * (k^T * q)   # output
S' = alpha * S + d * k^T              # state update
```

Because the state update is **linear** in `S`, after `m` buffered steps:
```
S = A * S0 + sum_j(w_j * d_j * k_j^T)

where:
  A = prod_i(alpha_i)         # total decay
  w_j = prod_{i>j}(alpha_i)   # replay decay of entry j
```

**Key insight**: The checkpoint `S0` is decayed by ALL buffered `alpha`; each older entry `j` is decayed by every `alpha` that came **after** it. The output needs only `S*q` and `S*k`, both matrix-vector products:
```
S*q = A*(S0*q) + sum_j(w_j * (k_j^T * q) * d_j)
```

`(k_j^T * q)` is a **scalar** computed first — the `[V,K]` outer product `d_j * k_j^T` is **never built in HBM** on a non-flush step. The reconstruction sum is a small GEMM running on **tensor cores**.

### 2.4 Ring Buffer Design

Instead of snapshotting the full state per draft token, ReplaySSM keeps:

1. **Frozen checkpoint `h0`** (`temporal` in SGLang's `ssm_states`, shape `[slots, HV, V, K]`): the last flushed state, read every step but written only every `L` steps
2. **Circular `(d, k, g)` ring**: per-slot ring buffer of the last `L` committed steps' records
   - `d_cache`: `[layers, slots, HV, L, V]` — residual vectors
   - `k_cache`: `[layers, slots, H, L, K]` — key vectors
   - `g_cache`: `[layers, slots, HV, L]` — gate scalars (fp32)
3. **Block-keyed cursors** (per-slot, shared across all layers of a decode step):
   - `write_pos`: `[slots]` int32 — ring write position
   - `cache_base`: `[slots]` int32 — circular origin
   - `is_flush`: `[slots]` int8 — flush flag

On a non-flush step: append `(d, k, g)` to ring, reconstruct output-only (no state write-back). On a flush step (every `L` committed tokens): fold ring into checkpoint, write full state back.

Rejected-draft rollback becomes a **pointer move** on the ring cursors — no state write-back needed.

### 2.5 Spec-Verify Flow

The verify kernel (`gdn_replayssm_spec_decode` in the new file) processes the whole draft window at once:

1. **Reconstruction**: From checkpoint `h0` + ring contents, compute output for all draft tokens via the chunked delta-rule `(I + A)^{-1}` UT-transform
2. **Ring append**: Write this window's `(d, k, g)` records to the ring (they become committed on acceptance)
3. **State is never materialized** on a non-flush step — only the output is computed

After verification, `commit_gdn_replayssm_spec` (in `spec_utils.py`) advances the ring cursors by the accepted count (including bonus token) instead of copying an intermediate state into `temporal`. The CONV state still needs its usual accept-rollback (scatter), which is preserved in the commit path.

### 2.6 Lifecycle Integration

```
Prefill (extend):  Reset cursors (write_pos=0, cache_base=0, is_flush=0) on each slot
                    → checkpoint == temporal, ring empty

Verify (decode):   ReplaySSM kernel dispatches when ring is allocated
                    → (flag on + GDN + topk<=1)
                    → output-only reconstruction from checkpoint + ring
                    → append (d,k,g) to ring

Commit:            commit_gdn_replayssm_spec advances cursors by accepted count
                    → conv state scatter still runs (separate from SSM)
                    → early return (skip SSM intermediate copy)

Flush:             Every L committed tokens, kernel folds ring into checkpoint
                    → device-side, inside CUDA graph capture
                    → full state written back to temporal

COW (copy_from):   Ring + cursors replicated for prefix caching
                    → excluded from RDMA transfer
```

### 2.7 Dispatch Logic (gdn_backend.py)

In `forward_extend`, when `is_target_verify`:
```python
if mamba_cache_params.replayssm_d is not None:
    # ReplaySSM path: circular (d,k,g) reconstruction kernel
    core_attn_out = self._replayssm_target_verify(...)
else:
    # Fallback: recurrent verify with per-draft full-state snapshots
    core_attn_out = self.kernel_dispatcher.target_verify(...)
```

The `_replayssm_target_verify` method reshapes `q/k/v/a/b` into packed token format, calls `gdn_replayssm_spec_decode` with the ring buffers and checkpoint, and returns the output reshaped to match the recurrent verify output shape.

### 2.8 Config Invariants

- `gdn_replayssm_spec_cache_len` (L, default 16) must be a **power of two**
- L >= 2 * `speculative_num_draft_tokens` (early-flush invariant — ensures at least one flush opportunity within any draft window)
- `max_cache_len >= 2 * speculative_num_draft_tokens` enforced in config validation

---

## 3. Memory Savings Analysis (~12 GB intermediate_ssm Elimination)

### 3.1 Baseline Memory Footprint

The recurrent verify path allocates `intermediate_ssm` — a full `[V, K]` state snapshot per draft token:
```
intermediate_ssm per request:
  layers * heads * draft_tokens * [V, K] * dtype_size
  = 70 * 64 * 4 * 128 * 128 * 2 bytes (bf16)
  = 70 * 64 * 4 * 32,768 * 2
  = ~560 MB per request (as stated in #27658)
```

For 32-64 concurrent requests, this scales to ~17-35 GB. The PR states **~12 GB** for this model configuration at typical serving concurrency, which is the bulk of spec-decode memory.

### 3.2 ReplaySSM Ring Memory

The ring replaces the full `[V,K]` state per draft token with tiny `(d,k,g)` records:
- `d`: `[HV, V]` = 32*128 = 4096 elements (bf16 = 8 KB per slot per layer per L-step)
- `k`: `[H, K]` = ~4096 elements (bf16 = ~8 KB)
- `g`: `[HV]` = 32 elements (fp32 = 128 bytes)

Total per slot per layer per L-step: ~16 KB vs the full state ~32 KB (but the ring stores L=16 steps worth, so 16*16KB = 256 KB per slot per layer)

Still much less than intermediate_ssm because:
- intermediate_ssm stores **4 full [V,K] states** per request (draft_token_num=4): 4*32KB = 128 KB per slot per layer
- The ring stores L=16 tiny records: ~256 KB per slot per layer
- But intermediate_ssm is allocated **per-request concurrency slot** while the ring is allocated once and reused

**The key savings**: intermediate_ssm is a separate buffer for each request's draft tokens, requiring allocation for max_concurrency * draft_tokens. The ring is a per-slot persistent structure that doesn't grow with concurrency. The net result is eliminating the ~12 GB that intermediate_ssm required.

### 3.3 OOM Comparison

The PR explicitly states: "the recurrent baseline OOMs at `--mem-fraction-static 0.8` where replayssm fits" — under a fixed HBM budget, ReplaySSM sustains higher concurrency, where the bandwidth win is larger still.

This is a **dual win**: memory elimination + bandwidth reduction compound at high concurrency.

---

## 4. Performance Data

### 4.1 Decision Gate: Bandwidth-Bound Measurement

The RFC gates Part B on whether verify forward is bandwidth-bound at target batch. Measured on **H20-3e** (peak ~3.85 TB/s):

| batch | achieved HBM bandwidth |
|------:|------------------------:|
| 1     | 6.7%                    |
| 16    | 60.8%                   |
| 64    | **74.7%**               |
| 256   | **79.4%**               |

>=70% at batch >=64: the full-state-write amortization is a real speedup, not just a memory win.

### 4.2 Throughput and TPOT Benchmarks

Matched A/B on **Qwen3.5-35B-A3B (TP4 NEXTN, draft=4, custom all-reduce)**, `bench_serving` random 256in/512out:

| concurrency | output throughput delta | median TPOT delta |
|------------:|------------------------:|-------------------:|
| 8           | +6.5%                   | -8.0%              |
| 16          | +5.6%                   | -5.3%              |
| 32          | **+13.1%**              | **-11.8%**         |

The win **grows with batch** because the kernel is bandwidth-bound — reducing bytes moved per step directly translates to speedup at high concurrency.

### 4.3 Correctness Verification

| Layer | Method | Result |
|-------|--------|--------|
| Kernel math | vs `chunk_gated_delta_rule` | ~5e-4 across draft_token_num 1-8 |
| Lifecycle | multi-step ring + commit + flush + reset vs chunk | bounded ~5e-3 (bf16, re-anchored at flushes) |
| **End-to-end** | **GSM8K (400q), Qwen3.5-35B-A3B TP4 NEXTN, CUDA-graph on** | **0.892 (flag on) vs 0.887 (flag off)** — within noise, 0 invalid |

Mathematically lossless but not bit-exact (matches baseline GDN, which is already non-bit-exact across chunk sizes). The degenerate L=1 case (every step is a flush, ring always empty) reduces algebraically and numerically to the baseline packed decode — a bit-exact anchor for testing.

---

## 5. RTX 4090 Implications for GRPO Rollout

### 5.1 Direct Impact Assessment

**ReplaySSM is GDN-only and linear-chain (topk<=1) only.** This means:

- **YES** for NEXTN/MTP-style speculative decoding on GDN models (Qwen3.5-35B-A3B)
- **NO** for EAGLE tree verify (topk>1) — falls back to recurrent verify
- **NO** for KDA — not yet supported (follow-up when KDA spec lands in SGLang)
- **NO** for NPU/CPU — falls back to recurrent verify

### 5.2 RTX 4090 Memory Budget Analysis

For RTX 4090 (24 GB HBM) serving a GDN model with speculative decoding:

- The ~12 GB `intermediate_ssm` elimination is **significant** — 50% of HBM freed up
- This enables higher concurrency under fixed budget, which is critical for GRPO rollout
- However, the benchmarks were measured on **H20-3e (TP4)** — not single-GPU RTX 4090
- On RTX 4090, the bandwidth-bound threshold would be different (SM89, different peak bandwidth)

### 5.3 RTX 4090 Viability Concerns

1. **TP requirement**: The benchmarks used TP4 (Qwen3.5-35B-A3B). On a single RTX 4090:
   - Qwen3.5-35B-A3B at FP8 requires ~35 GB even with expert offloading — exceeds 24 GB
   - Smaller models (DSV2-Lite 16B, Qwen3-8B) would need separate benchmarking
   - The bandwidth-bound threshold likely differs on SM89 vs SM90 (H20)

2. **SM89 compatibility**: ReplaySSM is a Triton kernel. Triton works on SM89. No Inductor dependency. This is good for RTX 4090.

3. **CUDA graph compatibility**: The PR explicitly states "CUDA-graph-safe (cursors are static buffers advanced post-verify outside capture; the flush is device-side inside the kernel)." This is critical for RTX 4090 serving stability.

4. **GRPO rollout context**: In verl's HYBRID mode with SGLang as rollout engine:
   - Weight reload boundary: ReplaySSM ring must be reset/flushed when weights change
   - The COW path handles `copy_from` for prefix caching — but NOT RDMA transfer
   - For GRPO, the weight-reload funnel would need to reset ring cursors (similar to prefill reset)

### 5.4 MUST DO / MUST NOT for RTX 4090 with ReplaySSM

```
MUST DO:
  1. Set --enable-gdn-replayssm-spec when using GDN models with NEXTN/MTP spec decode
  2. Set --gdn-replayssm-spec-cache-len to power of two >= 2*speculative_num_draft_tokens
  3. Use --mamba-radix-cache-strategy no_buffer (ReplaySSM incompatible with extra_buffer)
  4. Reset ring cursors at weight-reload boundary in GRPO HYBRID mode
  5. Test bandwidth-bound threshold on RTX 4090 specifically (may differ from H20)

MUST NOT:
  1. NOT use with EAGLE tree verify (topk>1) — automatic fallback to recurrent
  2. NOT use with --enable-mamba-extra-buffer — raises ValueError, incompatible
  3. NOT use with KDA models — not yet supported
  4. NOT assume RTX 4090 benchmarks from H20-3e TP4 data
  5. NOT use with NPU/CPU — automatic fallback
```

### 5.5 Connection to RTX 4090 GRPO Training Ranking

ReplaySSM strengthens the SGLang rollout path for verl HYBRID mode:
- verl CPPO+bypass #1 → ReplaySSM makes GDN NEXTN spec decode viable on SGLang → better throughput
- But RTX 4090 single-GPU limits model size → smaller GDN models only
- No impact on DeepSpeed/rLLM/Megatron training paths (they don't use SGLang for rollout)

**RTX 4090 GRPO ranking remains unchanged**: verl CPPO+bypass #1, but ReplaySSM improves the SGLang rollout throughput component when GDN models are used.

---

## 6. Connection to GDN (#28679) and Spec Decode Ecosystem

### 6.1 Connection to #28679 GDN Intermittent Degeneracy

#28679 reports a **critical intermittent degeneracy** on Qwen3.6-27B-FP8 (dense hybrid GDN) with `extra_buffer` strategy:
- Tiny-output loop + decode-throughput collapse
- **Worsens over server uptime**, **clears on restart**
- **Silent** — no error logged, no NaN, no OOM

**ReplaySSM connection**:
1. ReplaySSM is **incompatible with extra_buffer** — it requires `no_buffer` strategy. The degeneracy in #28679 manifests under `extra_buffer`. These are different code paths.
2. However, both issues touch the **GDN state management lifecycle**. ReplaySSM's ring cursors and periodic flush mechanism could potentially provide a **more deterministic** state evolution path vs the recurrent verify's per-draft snapshots.
3. ReplaySSM's ring reset at prefill ensures a clean state, and the periodic flush provides a "re-anchor" every L tokens — this is structurally similar to what #28679 needs (the degeneracy worsens over uptime, suggesting state drift).
4. **NOT a fix for #28679**: The degeneracy occurs without speculative decoding (no spec decode in the #28679 setup). ReplaySSM only addresses the spec-verify path, not the base decode path. Part A (#28451) addresses the base decode path.

### 6.2 Connection to #28511 RFC (ReplaySSM Porting)

#28511 is the parent RFC tracking ReplaySSM porting to SGLang in three parts:

| Part | Scope | Resource Won | Status |
|------|-------|-------------|--------|
| **A** | Buffered output-only GDN decode kernel | HBM bandwidth (decode speedup) | #28451 OPEN |
| **B** | ReplaySSM spec-decode verify (ring) | bandwidth + memory | #28695 OPEN (this PR) |
| (rel.) | Compact linear spec cache | spec-decode memory footprint | #27658 OPEN |

Part A (#28451) is the **decode** optimization: per-step state traffic drops from read+write (~8dn) to read-only (~4dn), roughly halved. 12 files, +628 new kernel + supporting infrastructure.

Part B (#28695) is the **spec-verify** optimization: eliminates ~12 GB intermediate_ssm + reduces per-draft bandwidth. 6 files, +741 new kernel.

### 6.3 Connection to #27658 Compact Linear Spec Cache

#27658 (by strgrb) proposes a **delta cache** approach: instead of storing full `[V,K]` states, store the delta vectors `k_t` and `v_t^T - k_t^T * Diag(alpha_t) * S_{t-1}` plus `alpha_t`. Per head per layer: full state [128,128] → delta [3,128] — a ~42x size reduction.

Both #28695 (ReplaySSM) and #27658 (compact spec cache) address the intermediate_ssm memory problem, but with different approaches:
- ReplaySSM: circular ring + checkpoint + output-only reconstruction (tensor-core GEMM)
- Compact spec cache: delta encoding + replay kernel to reconstruct full state on demand

**Complementary or competing?**: The PR states ReplaySSM is Part B of the RFC, while compact spec cache is listed as a "related" piece. They may coexist — ReplaySSM handles the bandwidth optimization, compact cache handles memory footprint for the remaining states. But they're different kernel paths.

### 6.4 Connection to vLLM ReplaySSM

The vLLM reference implementation (`Johnny-Liou/ReplaySSM`, commit `3c85112`) provides:
- `fused_recurrent_replayssm.py` (decode) — Part A equivalent
- `gdn_replayssm_spec_decode.py` (spec) — Part B equivalent

SGLang's port differs in:
- **Split q/k/v** tensors (SGLang passes already-split + post-causal-conv1d tensors) vs vLLM's packed `mixed_qkv`
- **State layout**: SGLang `ssm_states` is `[slots, HV, V, K]` with K contiguous, matching the checkpoint directly
- **Null block sentinel**: SGLang uses -1 (valid slots >= 0) vs vLLM's 0
- **Gate type**: GDN has `head_k_dim == head_v_dim`, so `temporal` (`[slots, HV, K, V]`) maps directly

### 6.5 Connection to DFlash (#46105) and Spec Decode Ecosystem

vLLM #46105 tracks DFlash bring-up. The spec-decode ecosystem now has multiple paradigms:

| Paradigm | Framework | Status | RTX 4090 Viability |
|----------|-----------|--------|---------------------|
| EAGLE tree verify | SGLang/vLLM | Production | topk>1, NOT ReplaySSM compatible |
| NEXTN/MTP linear | SGLang/vLLM | Production | topk<=1, ReplaySSM compatible |
| DFlash | vLLM #46105 | In progress | Standard + Hybrid + MiMo variants |
| Orthrus block-diffusion | vLLM #46007 | CHANGES_REQUESTED | 17.76% acceptance, NOT viable yet |

---

## 7. Comparison with Other Spec Decode Optimizations

### 7.1 ReplaySSM vs EAGLE Tree Verify

| Dimension | ReplaySSM | EAGLE Tree Verify |
|-----------|-----------|-------------------|
| Scope | GDN/KDA SSM layers only | All attention layers |
| Topology | Linear chain (topk<=1) | Tree (topk>1) |
| Memory | Eliminates ~12 GB intermediate_ssm | Uses KV cache per draft token |
| Bandwidth | Output-only reconstruction (no state write) | Standard attention verify |
| Acceptance | MTP typically ~70-80% | EAGLE typically ~80-90% |
| Compatibility | EAGLE tree falls back to recurrent verify | ReplaySSM NOT applicable |

**Key insight**: ReplaySSM and EAGLE are **orthogonal** — ReplaySSM optimizes the SSM (linear attention) component of hybrid models, while EAGLE optimizes the attention component. They don't compete; they address different parts of the model architecture.

### 7.2 ReplaySSM vs Compact Spec Cache (#27658)

| Dimension | ReplaySSM (#28695) | Compact Spec Cache (#27658) |
|-----------|---------------------|------------------------------|
| Mechanism | Circular ring + checkpoint + output-only | Delta encoding + replay reconstruction |
| Memory reduction | Eliminates intermediate_ssm entirely (~12 GB) | Reduces per-head from [128,128] to [3,128] (~42x) |
| Compute overhead | Tensor-core GEMM for reconstruction | Replay kernel to restore full state |
| Bandwidth impact | Removes per-draft full-state writes (bandwidth win) | Smaller footprint but still needs full-state reconstruction |
| Scope | GDN only, linear chain | GDN (implemented), KDA (planned), Lightning Attn (planned) |
| Flush mechanism | Periodic flush every L committed tokens | Not needed (delta approach) |

**Key insight**: ReplaySSM provides a **bandwidth win** in addition to a memory win, which is why it matters at high concurrency. Compact spec cache provides a **pure memory win**. They complement each other.

### 7.3 ReplaySSM vs DFlash (#46105)

| Dimension | ReplaySSM | DFlash |
|-----------|-----------|--------|
| Architecture | GDN/KDA linear attention optimization | Non-causal full attention (all layers) |
| Mechanism | SSM ring buffer + output-only reconstruction | Sliding-window + non-causal attention |
| Applicable models | Qwen3.5 (hybrid GDN), Kimi (KDA) | Qwen3-8B-DFlash, Gemma4-DFlash, MiMo-DFlash |
| RTX 4090 | NEXTN/MTP viable, needs SM89 benchmarking | MiMo-style needs non-causal+SWA support |

**Key insight**: ReplaySSM and DFlash serve **different model families**. ReplaySSM is for hybrid GDN/KDA models (linear attention layers). DFlash is for models with non-causal full attention. They're not competing — they're expanding spec decode coverage to different architectures.

### 7.4 ReplaySSM vs Orthrus (#46007)

| Dimension | ReplaySSM | Orthrus |
|-----------|-----------|---------|
| Mechanism | SSM ring buffer optimization | Block-diffusion speculative decoding |
| Acceptance rate | MTP/NEXTN ~70-80% | 17.76% (very low) |
| Memory sharing | Shares target model weights (no draft model) | Shares target model weights |
| Scope | GDN/KDA SSM verify path | Attention-based spec decode |
| Status | OPEN, CI running | CHANGES_REQUESTED, blocked by ModelRunnerV2 |

**Key insight**: ReplaySSM is a verified, well-benchmarked optimization for an existing spec-decode path. Orthrus is a new spec-decode paradigm with low acceptance rate. ReplaySSM is far more practical for immediate deployment.

---

## 8. Code Architecture Deep Dive

### 8.1 New Kernel File: gdn_replayssm_spec_decode.py (+741)

The core Triton kernel `gdn_replayssm_spec_circular_kernel`:
- Takes split q/k/v/a/b (post causal-conv1d), checkpoint state h0/ht, d/k/g cache rings, and block-keyed cursors
- Parameters: `stride_q_t`, `stride_k_t`, `stride_v_t` etc. as `tl.constexpr` for compilation
- Operates on packed token format with `query_start_loc` cu_seqlens
- Uses `ssm_state_indices` to map requests to physical mamba slots
- Handles the `(I + A)^{-1}` UT-transform for output-only reconstruction

The Python wrapper `gdn_replayssm_spec_decode`:
- Validates input shapes and strides
- Sets up launch grid
- Calls the Triton kernel with all ring buffer + cursor arguments

Cursor management functions:
- `commit_gdn_replayssm_spec`: Advances cursors by accepted count after verification
- Ring reset at prefill: `write_pos=0, cache_base=0, is_flush=0`

### 8.2 Backend Dispatch: gdn_backend.py (+117/-16)

The `forward_extend` method now has a two-path dispatch:
```python
if is_target_verify:
    if mamba_cache_params.replayssm_d is not None:
        # ReplaySSM path
        core_attn_out = self._replayssm_target_verify(...)
    else:
        # Recurrent verify path (fallback)
        core_attn_out = self.kernel_dispatcher.target_verify(...)
```

After prefill (extend, non-verify), cursor reset:
```python
if getattr(mamba_cache_params, "replayssm_write_pos", None) is not None:
    slots = cache_indices[cache_indices >= 0]
    mamba_cache_params.replayssm_write_pos[slots] = 0
    mamba_cache_params.replayssm_cache_base[slots] = 0
    mamba_cache_params.replayssm_is_flush[slots] = 0
```

New method `_replayssm_target_verify`:
- Reshapes q/k/v/a/b into packed token format
- Calls `gdn_replayssm_spec_decode` with checkpoint (`temporal`) + ring buffers
- Returns output reshaped to match recurrent verify output shape
- GDN has K==V (`head_k_dim == head_v_dim`), so `temporal` is consumed directly as the kernel checkpoint

### 8.3 Memory Pool: memory_pool.py (+147/-3)

New `gdn_replayssm_spec_enabled` predicate function:
```python
def gdn_replayssm_spec_enabled(is_npu, is_cpu, speculative_eagle_topk, enabled):
    return enabled and not is_npu and not is_cpu and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
```

New `SpeculativeState` fields (all Optional, None unless flag on):
```python
replayssm_d: Optional[torch.Tensor] = None   # [layers, slots, HV, L, V]
replayssm_k: Optional[torch.Tensor] = None   # [layers, slots, H,  L, K]
replayssm_g: Optional[torch.Tensor] = None   # [layers, slots, HV, L]  fp32
replayssm_write_pos: Optional[torch.Tensor] = None   # [slots] int32
replayssm_cache_base: Optional[torch.Tensor] = None   # [slots] int32
replayssm_is_flush: Optional[torch.Tensor] = None     # [slots] int8
```

Ring allocation in `__init__` when `gdn_replayssm_spec_enabled`:
- Derives `H_` from `conv_dim` = `2*H*K + HV*V` (same formula as conv state)
- Validates `L_` is power of two and >= 2*speculative_num_draft_tokens
- Allocates `nslots = size + 1` ring tensors on CUDA
- Derives `num_k_heads` from conv_dim (GDN conv runs over mixed_qkv = q|k|v)

`get_tensor_size_bytes` fix: Returns 0 for `None` tensors (so Optional ring fields don't break `mem_usage_bytes()`)

`at_layer_idx` fix: Per-slot cursors (`write_pos`, `cache_base`, `is_flush`) are shared across all layers — not sliced by `at_layer_idx`.

### 8.4 Spec Utils: spec_utils.py (+57/-0)

The `commit_mamba_states_after_verify` function now has a ReplaySSM branch:
- If `spec_state.replayssm_d is not None` → ReplaySSM commit path
  - Call `commit_gdn_replayssm_spec` to advance cursors by accepted count
  - Roll back conv state to last accepted draft step (scatter with mask)
  - **Early return** — skip the SSM intermediate copy entirely
- Else → existing recurrent verify commit path

The conv state rollback is preserved because ReplaySSM only replaces the SSM verify; the causal-conv1d state still needs per-draft accept-rollback.

### 8.5 CLI Flags: server_args.py (+33/-0)

```python
enable_gdn_replayssm_spec: bool = False    # opt-in, default off
gdn_replayssm_spec_cache_len: int = 16     # ring length L, power of two
```

Validation:
- Raises ValueError if `enable_gdn_replayssm_spec` + `enable_mamba_extra_buffer` (incompatible)
- Invariant: `gdn_replayssm_spec_cache_len >= 2 * speculative_num_draft_tokens`

---

## 9. Limitations and Follow-Up Work

### 9.1 Current Limitations

1. **GDN only**: KDA, Lightning Attn not yet supported (follow-up when KDA spec is in SGLang)
2. **Linear chain only**: topk<=1 (NEXTN/MTP). EAGLE tree verify (topk>1) falls back to recurrent
3. **Incompatible with extra_buffer**: mamba radix prefix-caching (`--enable-mamba-extra-buffer`) needs device-side force-flush at radix-track boundaries — not yet implemented
4. **Not bit-exact**: Tensor-core reconstruction matches baseline GDN non-bit-exact behavior (lossless but fp32 accumulation → bf16 output)
5. **TP4 measurement**: All benchmarks on H20-3e with TP4; single-GPU RTX 4090 benchmarks needed

### 9.2 Follow-Up Items

1. **KDA ReplaySSM spec verify**: When KDA spec decode is supported in SGLang
2. **Radix prefix-caching compatibility**: Device-side force-flush for extra_buffer
3. **Part A integration**: #28451 (decode kernel) still OPEN — needs to merge before full ReplaySSM benefit
4. **Compact spec cache (#27658) integration**: Whether ReplaySSM + delta cache coexist or one supersedes
5. **RTX 4090/SM89 specific benchmarks**: Verify bandwidth-bound threshold on consumer GPUs
6. **verl HYBRID weight-reload boundary**: Ring cursor reset needed when weights are updated in GRPO

---

## 10. Cross-Framework Synthesis

### 10.1 ReplaySSM Across Frameworks

| Framework | ReplaySSM Status | Notes |
|-----------|-------------------|-------|
| vLLM | Reference implementation (Johnny-Liou/ReplaySSM) | Original port from Dao AI Lab |
| SGLang | Part A (#28451) + Part B (#28695) OPEN | Split q/k/v adaptation, CUDA-graph safe |
| verl | Not applicable (training framework) | Uses SGLang/vLLM as rollout engine |
| DeepSpeed | Not applicable (training framework) | No GDN/KDA model support |
| Megatron | Not applicable | No ReplaySSM kernel |

### 10.2 Pattern: GPU-Resident Cache Invalidation at Weight-Reload Boundary

This PR reinforces the **systematic pattern** identified across multiple frameworks:
- SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED on RL weight update (DSV4)
- vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake
- ReplaySSM: Ring cursors must be reset at prefill/weight-reload boundary

The **principle**: Any GPU-resident cache that depends on model weights must be invalidated/reset when those weights change. ReplaySSM's ring contains `(d, k, g)` records derived from the current model weights — stale ring entries after weight reload would produce incorrect output.

### 10.3 GDN Spec-Decode Landscape

```
                    Spec Decode for GDN Models
                    =========================

Current (before ReplaySSM):
  ┌─────────────────────────────────────────┐
  │ Recurrent Verify                         │
  │   • Full [V,K] state per draft token     │
  │   • ~12 GB intermediate_ssm              │
  │   • 75-79% HBM bandwidth at batch>=64    │
  │   • Rollback = state copy from snapshot   │
  └─────────────────────────────────────────┘

ReplaySSM (this PR):
  ┌─────────────────────────────────────────┐
  │ Ring Verify                               │
  │   • Circular (d,k,g) ring + checkpoint    │
  │   • ~12 GB eliminated                     │
  │   • Output-only reconstruction            │
  │   • Rollback = pointer move on cursors    │
  │   • +13.1% throughput, -11.8% TPOT        │
  │   • Mathematically lossless                │
  └─────────────────────────────────────────┘

Compact Spec Cache (#27658):
  ┌─────────────────────────────────────────┐
  │ Delta Cache                               │
  │   • [3,128] deltas per head vs [128,128]  │
  │   • ~42x per-head reduction               │
  │   • Replay kernel for reconstruction      │
  │   • Compatible with radix prefix caching  │
  └─────────────────────────────────────────┘
```

---

## 11. Key Takeaways

1. **ReplaySSM is the most significant GDN spec-decode optimization** — eliminates the dominant memory cost (~12 GB intermediate_ssm) and provides measurable bandwidth wins at serving concurrency
2. **Mathematically lossless** — GSM8K parity, not an approximation trick
3. **RTX 4090 relevance is MODERATE** — NEXTN/MTP viable on SM89, but benchmarks are TP4 on H20-3e, single-GPU numbers needed. Smaller GDN models (8B-16B) would benefit most
4. **Incompatible with extra_buffer** — must use no_buffer mamba strategy, which is the default
5. **Weight-reload boundary is critical for GRPO** — ring cursors must be reset when model weights change in verl HYBRID mode
6. **Part of a three-piece ecosystem**: Part A (decode, #28451) + Part B (spec verify, #28695) + compact cache (#27658) — all OPEN, all complement each other
7. **Orthogonal to EAGLE/DFlash** — addresses GDN/KDA linear attention specifically, not attention-based spec decode paradigms
8. **Connection to #28679**: ReplaySSM uses a different mamba strategy (no_buffer) than the degeneracy bug (extra_buffer), but the periodic flush mechanism structurally addresses the "worsens over uptime" pattern — needs investigation
