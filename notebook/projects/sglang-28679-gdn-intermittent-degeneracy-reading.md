# SGLang #28679 — GDN Intermittent Silent Decode Degeneracy Reading

> 2026-06-19 | Issue #28679 OPEN | Author: santiwen | 0 comments, 0 reviews
> ★★★★★★★★ MOST DANGEROUS bug pattern for GRPO: silent corruption + worsening over uptime + no error logged
> ★★★★★★★★ NOT a DSV4 failure (Qwen3.6-27B-FP8 is DENSE hybrid GDN) — but shares the broader "state accumulation" pattern
> ★★★★★★★★ BOTH mamba-scheduler strategies (extra_buffer AND no_buffer) now have bugs — #20791 (no_buffer) closed, #28679 (extra_buffer) NEW
> ★★★★★★★★ RTX 4090 CRITICAL: same FlashInfer pool API, same SSM state management, same "worsens over uptime" risk for long-running GRPO

---

## 1. Issue Metadata

```
Issue Number:     #28679
Title:            Qwen3.6-27B-FP8 (dense, hybrid GDN): intermittent silent decode degeneracy
                  — tiny-output loop + decode-throughput collapse (extra_buffer, A100)
Author:           santiwen
State:            OPEN
Created:          2026-06-18T21:24:04Z
Updated:          2026-06-18T21:24:04Z (no updates yet)
Labels:           NONE (no maintainer triage yet)
Comments:         0
Reactions:        0
Assignees:        NONE
```

---

## 2. Full Issue Description (Verbatim Key Sections)

### Model and Hardware

```
Serving Qwen/Qwen3.6-27B-FP8 (dense, hybrid Gated-DeltaNet) on a single A100 80GB PCIe (SM80).
SGLang 0.5.13.post1 (official lmsysorg/sglang:latest image), CUDA 13.x,
FlashInfer attn/sampling, xgrammar.
max_model_len 262144, KV bf16.
Launch: --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --mamba-scheduler-strategy extra_buffer --mem-fraction-static 0.90 --context-length 262144
  (radix cache on; no speculative decoding; disable_piecewise_cuda_graph=True)
```

### Two Manifestations of Degeneracy

```
Manifestation 1: TINY-OUTPUT LOOP
  → Model emits burst of short, non-progressing completions
  → completion_tokens ~80-225 vs healthy ~260-690
  → reasoning_tokens collapsed to ~15-90 vs healthy ~260-690
  → Every one stopping on finish_reason {matched: 248046}
  → num_retractions: 0 (no KV cache retraction)
  → Agentic ReAct loop on top cannot make progress → hits recursion limit in 20-48 s

Manifestation 2: DECODE-THROUGHPUT COLLAPSE
  → Single request decode drops to ~0.02 tok/s
  → GPU pinned at 100% util + power-capped (304W vs 72W idle)
  → clocks_event_reasons.active = 0x4 (SW Power Cap)
  → #running-req: 1, full token usage: 0.06, mamba usage: 0.03
  → cuda graph: True
  → GPU is busy but decode does not advance
```

### The MOST Dangerous Characteristic: SILENT Corruption

```
★★★★★★★★★ Crucially, NOTHING is logged as an error:
  → No OOM
  → No KV-cache retraction
  → No NaN
  → No watchdog fire
  → No traceback
  → No restart

The model SUCCESSFULLY returns degenerate output that looks like a normal stop.
The same prompt that loops in a bad window produces a correct multi-hundred-token
answer minutes earlier/later.

★★★★★★★★★ This is the WORST possible bug pattern for GRPO:
  → No error signal → no retry → no fallback
  → Degenerate output silently propagates into rewards/advantages
  → Long-running GRPO (hours) → worsening accumulation → silently corrupted training
```

### What Was Ruled Out

```
1. GPU hardware: ECC all-zero (volatile + aggregate SRAM/DRAM), no thermal slowdown, 29-39 C
2. Server crash / OOM / KV retraction / NaN: server log clean across hours, no restart
3. KV / mamba pool pressure: full token usage 0.02-0.13, mamba usage ~0.03
   → KV pool ~362K tokens for single request → NO pressure
   → num_retractions: 0 throughout
4. #23687 (FP8 weight_scale_inv dropped at load): does not apply
   → checkpoint loads with no gate_gate_up_proj.weight_scale_inv not found warnings
   → produces coherent output in GOOD windows
```

### Reporter's Hypothesis

```
The combination (GPU pinned 100% + power-capped + ~0.02 tok/s decode + short
stop-token-248046 completions, no error) points at the GDN / mamba decode path
entering a DEGENERATE RUNTIME STATE, not memory/cache or hardware.
```

### Reproduction Note

```
Intermittent — not reliable on demand. Surfaces under sustained agentic/multi-turn
workload with long reasoning traces (thinking enabled), typically within ~30 min
of a fresh start. Fixed-prompt and short-conversation probes run clean in a healthy
window (so a one-shot repro proves nothing).
```

### Reporter's Questions for Maintainers

```
1. Is this a known GDN/Gated-DeltaNet (or extra_buffer mamba-scheduler) decode-path
   degeneracy on Ampere, and is there a fix on main / a dev tag not yet in a released version?
2. Would --mamba-scheduler-strategy no_buffer avoid it? (We use extra_buffer specifically
   to avoid the no_buffer+flashinfer SSM-state aliasing / accuracy regression — #20791 —
   so this is a trade-off, not a clean toggle; --disable-radix-cache is rejected with
   extra_buffer.)
3. Any recommended mamba_ssm_dtype / mamba_backend / max_mamba_cache_size setting for
   stability on A100?
```

---

## 3. Deep Analysis

### 3a. What Exactly Is "Intermittent Decode Degeneracy"?

```
★★★★★★★★★ TWO distinct but correlated manifestations in the SAME bad windows:

SYMPTOM 1 (Tiny-output loop):
  → GDN decode path enters a state where it generates short, degenerate completions
  → reasoning_tokens collapse from 260-690 to 15-90 → model "stops thinking early"
  → finish_reason matched:248046 → model hits a specific stop token prematurely
  → NOT a sampling bug (same prompt produces correct answer in good windows)
  → NOT a KV/cache bug (num_retractions=0, low pool usage)
  → The GDN recurrent STATE is corrupted → produces degenerate reasoning → early stop

SYMPTOM 2 (Decode-throughput collapse):
  → GPU pinned at 100% utilization + 304W power cap → GPU is DOING WORK
  → But decode throughput ~0.02 tok/s → work is NOT advancing our decode
  → cuda graph: True → the captured graph IS running (GPU busy confirms)
  → full token usage 0.06, mamba usage 0.03 → minimal pool pressure
  → ★★★★★★★★ GPU is computing something → but NOT the correct decode path
  → Hypothesis: degenerate GDN state causes the CUDA graph to compute a
    "dead loop" — recurrent state feeds back into itself without advancing

★★★★★★★★★ Both symptoms point to the SAME root cause:
  → GDN/SSM recurrent state enters a DEGENERATE attractor state
  → Once in this state, computation continues (GPU busy) but output is wrong
  → The state is NOT NaN (no error logged) — it's a valid but wrong state
  → This is analogous to a dynamical system falling into a fixed point / limit cycle
```

### 3b. Why Does It Worsen Over Uptime?

```
★★★★★★★★★ THREE hypotheses for "worsens over uptime":

HYPOTHESIS A: SSM State Pool Slot Corruption Accumulation
  → FlashInfer pool API (initial_state + initial_state_indices) reads/writes SSM
    state in-place (#20791 documented this)
  → Over many requests, state slots get reused → residual state from previous
    requests may not be fully cleared
  → Each reuse introduces small corruption → accumulates over hours
  → Eventually, corruption reaches threshold → degenerate attractor state
  → ★★★★★★★★ Supported by: #20791 showed no_buffer has aliasing bugs,
    and #28679 shows extra_buffer ALSO has bugs (just different manifestation)

HYPOTHESIS B: Radix Cache + SSM State Lifecycle Coupling Bug
  → extra_buffer strategy ties SSM state lifecycle to radix cache lifecycle
  → As radix cache fills up (multi-turn agentic workload), prefix sharing and
    eviction create complex SSM state slot management
  → When a radix cache node gets evicted or merged, its SSM state slot may
    not be properly cleaned or may be reassigned to a new request before
    the previous state is fully consumed
  → ★★★★★★★★ #28692 (NEW June 19!) fixes mamba radix partial page prefix
    matching — page alignment bug creates invalid empty radix nodes!
    → This could be DIRECTLY related to #28679's degeneracy
    → Invalid empty nodes + SSM state coupling = state corruption pathway

HYPOTHESIS C: GPU-Resident Cache Staleness (Same Pattern as #28676)
  → FlashInfer or SGLang caches GPU-resident tensors for GDN decode
  → Over time, these cached tensors accumulate stale data
  → Weight updates, radix cache operations, or memory reuse clobbers the
    cached tensors without invalidating them
  → Cache still hits (same shape/config) but contents are stale/wrong
  → ★★★★★★★★ #28676 MXFP8 cache clobber: EXACT same pattern
    → GPU-resident shuffle index cache clobbered by weight-region reuse
    → Cache hit with stale data → 64x accuracy blowup (0.06→3.83)
  → ★★★★★★★★ Could apply to GDN: cached SSM state indices, cached
    recurrent state fragments, or cached decode path parameters

★★★★★★★★★ MOST LIKELY: Hypothesis B (radix cache + SSM lifecycle coupling)
  → The #28692 mamba radix page alignment bug creates invalid radix nodes
  → Invalid nodes propagate corrupted SSM state slot assignments
  → Corruption accumulates over uptime as more invalid nodes are created
  → Restart clears ALL state → radix cache empty → fresh start → clean
  → "Within 30 min of fresh start" → radix cache starts filling → invalid
    nodes start accumulating → threshold reached → degeneracy
```

### 3c. Why Does Restart Clear It?

```
★★★★★★★★★ Restart clears ALL GPU-resident state:
  → SSM pool slots → reset to initial values
  → Radix cache → empty (no prefixes cached)
  → CUDA graph captures → re-captured fresh
  → GPU-resident cached tensors → re-initialized
  → Memory allocator → fresh (no reuse/fragmentation)

★★★★★★★★★ The "clear on restart" pattern confirms:
  → Bug is in STATE MANAGEMENT, not in compute kernels
  → The compute path itself is correct (good windows produce correct output)
  → Corruption is ACCUMULATED, not instantaneous
  → There is NO self-correction mechanism → once degenerate, stays degenerate
  → ★★★★★★★★ This is the WORST pattern for GRPO: once corrupted, training
    continues with corrupted output until manual restart
```

### 3d. Non-Deterministic on Byte-Identical Input

```
★★★★★★★★★ SAME prompt produces different results depending on when it's sent:
  → Healthy window → correct multi-hundred-token answer
  → Degenerate window → short 80-225 token loop with collapsed reasoning

★★★★★★★★★ This means the bug depends on ENGINE STATE, not input:
  → The SSM state pool, radix cache state, or GPU-resident cache state
    determines whether a request gets healthy or degenerate output
  → Byte-identical input → different recurrent state context → different output
  → ★★★★★★★★ For GRPO: same prompt could get good rollout in step 1
    but degenerate rollout in step 50 → REWARD INCONSISTENCY → training instability
```

---

## 4. Cross-Issue Connection Analysis

### 4a. Connection to SGLang #20791 (no_buffer SSM-state aliasing)

```
#20791 (CLOSED): FlashInfer GDN accuracy degradation with no_buffer scheduling

Key finding from #20791:
  → extra_buffer: GSM8K 0.990 (correct)
  → no_buffer: GSM8K 0.940 (degraded)
  → no_buffer + disable-radix-cache: GSM8K ~0.890 (severely degraded)
  → Triton FLA kernel: 0.990 regardless of scheduler (unaffected)

Root cause hypothesis from #20791:
  → FlashInfer pool API (initial_state + initial_state_indices) reads/writes
    in-place → state aliasing or ordering issue under no_buffer
  → Triton path does explicit gather/scatter → no aliasing

★★★★★★★★★ #28679's connection to #20791:
  → #20791 showed no_buffer has accuracy regression → user chose extra_buffer
  → #28679 shows extra_buffer ALSO has a bug (just different: intermittent vs
    systematic, degenerate vs accuracy-degraded)
  → ★★★★★★★★ BOTH mamba-scheduler strategies now have known bugs!
  → extra_buffer: intermittent degenerate attractor state (worsens over uptime)
  → no_buffer: systematic accuracy degradation (constant ~6% GSM8K drop)
  → Trade-off: no_buffer is predictably degraded; extra_buffer is intermittently
    catastrophic → extra_buffer is WORSE for long-running GRPO!
  → ★★★★★★★★ For GRPO training: intermittent catastrophic > systematic mild
    → better to have predictable 6% degradation than random total collapse
  → BUT: disable-radix-cache + no_buffer = ~11% degradation → ALSO unacceptable
  → ★★★★★★★★ There is NO safe mamba-scheduler configuration for GDN on SGLang!
```

### 4b. Connection to SGLang #28692 (mamba radix partial page prefix matching)

```
#28692 (NEW June 19!): fix mamba radix partial page prefix matching

Key finding:
  → MambaRadixCache (not unified) wasn't page aligning keys before prefix matching
  → With paged KV (trtllm_mha), a limited key like 63 logical tokens backed by
    64 raw tokens could match a 64 token child → round match len down to 0
  → Creates invalid empty radix node → later crashes during tombstone eviction
    with "parent does not have child key, ()"

★★★★★★★★★ #28692 is DIRECTLY relevant to #28679:
  → #28679 uses extra_buffer which ties SSM state to radix cache lifecycle
  → Invalid empty radix nodes → corrupted SSM state slot assignments
  → Over time, more invalid nodes accumulate → SSM state corruption worsens
  → ★★★★★★★★ #28692 fix might partially address #28679!
  → BUT: #28692 only fixes the page alignment bug → there may be OTHER
    radix + SSM coupling bugs that #28679 exposes
  → ★★★★★★★★ If #28692 is merged, test #28679 again — the degeneracy
    may partially improve but not fully resolve
```

### 4c. Connection to SGLang #28676 (MXFP8 cache clobber)

```
#28676: MXFP8 flashinfer_trtllm_routed MoE cache clobber on RL weight reload

Key finding:
  → GPU-resident cached index tensors (shape-dependent, memoized) get CLOBBERED
    when weight region GPU memory is reused for weight updates
  → Cache still hits (same shape) but contents now garbage
  → 64x accuracy blowup (0.06→3.83) on first RL weight update
  → Affects any FP8 MoE on flashinfer-trtllm path through weight update

★★★★★★★★★ #28676 shares the SAME pattern class as #28679:
  → GPU-resident state that persists beyond its validity window
  → No invalidation mechanism → stale/clobbered data silently used
  → Worsens over time (each weight update clobbers more)
  → Silent corruption (no error logged, just wrong output)

  BUT: #28676 is an RL weight update path issue → triggered by explicit
  weight reload. #28679 is a SSM state accumulation issue → triggered by
  sustained inference workload (no weight update needed).

  ★★★★★★★★ The PATTERN is the same: GPU-resident cached state that
    becomes stale without invalidation. The TRIGGER is different.
```

### 4d. Connection to SGLang #28612 (DSV4 C128 state mapping lifecycle)

```
#28612: DSV4 C128 state mapping lifecycle fix

Key finding:
  → C128 state slots were derived through SWA mapping (full_to_swa_index_mapping)
  → SWA mapping lifecycle can clear/reuse independently of the radix-cached full
    KV prefix that C128 actually depends on
  → When SWA mapping cleared but full KV still alive → C128 reads slot 0, old
    slot, or reused slot → accuracy degradation
  → Fix: derive C128 slots directly from full KV locations (full_loc / 128)

★★★★★★★★★ #28612 shares the SAME root cause pattern as #28679:
  → STATE tied to WRONG lifecycle → state becomes invalid when its
    lifecycle dependency gets cleared/reused independently
  → #28612: C128 state tied to SWA mapping lifecycle (wrong)
  → #28679: SSM state tied to radix cache lifecycle (possibly wrong)
  → ★★★★★★★★ Pattern: STATE LIFECYCLE MISMATCH → state persists past
    its validity window → silently corrupted → degenerate output
```

### 4e. Connection to vLLM #45819 (GDN batch invariance)

```
#45819 (OPEN): Add batch invariance support to GDN_ATTN backend

Key finding:
  → GDN models currently cannot use VLLM_BATCH_INVARIANT=1
  → Hard incompatibility: RuntimeError on startup
  → PR adds supports_batch_invariance() returning True to GDNAttentionBackend
  → GDN already uses stable sorting (torch.argsort with stable=True)

★★★★★★★★★ #45819 is DIFFERENT in scope from #28679:
  → #45819: deterministic inference across batch sizes (reproducibility)
  → #28679: intermittent degeneracy over uptime (state corruption)
  → BUT: both affect the GDN decode path → related architecture

  ★★★★★★★★ Cross-framework connection:
  → If SGLang's GDN decode has state corruption bugs (#28679), vLLM's
    GDN decode (#45819) may have SIMILAR bugs → need to check vLLM
    GDN decode for "worsening over uptime" pattern
  → vLLM doesn't have SSM pool + radix cache coupling → different
    architecture → may not have the SAME bug
  → ★★★★★★★★ But FlashInfer pool API is shared → aliasing risk exists
```

### 4f. Connection to SGLang #28569 (EAGLE3 CUDA graph crash)

```
#28569: EAGLE3 draft CUDA graph replay → illegal memory access on gpt-oss-120b

★★★★★★★★★ Different root cause (CUDA graph replay reads stale metadata)
  → #28569: captured CUDA graph replays with stale batch-dependent data
  → #28679: SSM state accumulates corruption over uptime

  ★★★★★★★★ BUT: both are "runtime state corruption" patterns in SGLang:
  → #28569: static capture vs dynamic runtime mismatch
  → #28679: state lifecycle vs state validity mismatch
  → Pattern family: STATE BECOMES INVALID BUT COMPUTATION CONTINUES
```

### 4g. Connection to SGLang #28618/#28620 (SM89 DSV4-Flash-FP8)

```
#28618/#28620: RFC + PR for SM89/L20 DSV4-Flash-FP8 support

★★★★★★★★★ Indirect connection through shared concerns:
  → Both involve non-Hopper GPU architectures (SM80 A100, SM89 L20/RTX 4090)
  → Both involve FlashInfer kernels on non-SM90 hardware
  → Both involve complex state management (DSV4: C128/MTP state, GDN: SSM state)

  ★★★★★★★★ For RTX 4090: BOTH DSV4 (#28618) and GDN (#28679) have
    architecture-specific risks on SM89 → RTX 4090 deployment requires
    extra caution for BOTH model families
```

### 4h. Connection to SGLang #28695 (ReplaySSM Ring Spec-Verify)

```
#28695 (NEW June 19): GDN ReplaySSM Ring Spec-Verify

★★★★★★★★★ Optimization PR for GDN speculative decode target-verify path:
  → Replaces full [V,K] recurrent state writes per draft token with
    circular (d,k,g) ring + frozen checkpoint
  → Ring-based rollback → pointer move instead of full state snapshot
  → Mathematically lossless (GSM8K parity)
  → Opt-in via --enable-gdn-replayssm-spec

  ★★★★★★★★ Connection to #28679:
  → ReplaySSM changes SSM state management in the spec-decode verify path
  → If #28679's root cause is in the NON-spec-decode path (the reporter
    uses no speculative decoding), ReplaySSM doesn't help
  → BUT: ReplaySSM's ring + checkpoint lifecycle is a DIFFERENT state
    management approach → could AVOID the radix-coupled lifecycle bug
  → ★★★★★★★★ ReplaySSM might be an alternative architecture that
    sidesteps #28679's lifecycle coupling issue (for spec-decode use)
```

---

## 5. Is This the 10th/11th DSV4 Failure? — Pattern Classification

```
★★★★★★★★★ NO. #28679 is NOT a DSV4 failure. It is a SEPARATE, GDN-specific bug.

Reasons:
1. Qwen3.6-27B-FP8 is DENSE (not MoE) — described as "dense, hybrid GDN"
2. No DSV4-specific features involved (no MoE routing, no DSA indexer,
   no MTP, no online compress, no MLA DCP)
3. Bug is in GDN/SSM recurrent state management, not DSV4 dynamic routing

★★★★★★★★★ HOWEVER, #28679 belongs to a BROADER pattern family:

Pattern Family: "GPU-resident state becomes stale/corrupted without
invalidation, silently produces wrong output, worsens over uptime,
clears on restart"

Members of this pattern family:
  | Issue | Framework | State Type | Trigger | Severity |
  |-------|-----------|-----------|---------|----------|
  | #28679 | SGLang | SSM pool + radix coupling | Sustained inference | Intermittent catastrophic |
  | #28676 | SGLang | MXFP8 shuffle index cache | RL weight update | Systematic 64x blowup |
  | #28612 | SGLang | C128 state via SWA mapping | Multi-turn prefix sharing | Correctness degradation |
  | #28692 | SGLang | MambaRadixCache page alignment | Paged KV + prefix match | Crash (tombstone eviction) |
  | #20791 | SGLang | FlashInfer SSM state aliasing | no_buffer scheduling | Systematic 6-11% accuracy drop |
  | #45972 | vLLM | CUDA graph captured metadata | DSV4 cudagraph replay | Garbage output |
  | #10684 | vLLM-Ascend | DSA Hadamard constant buffer | Sleep/wake state transfer | ALL-ZERO output |
  | #28569 | SGLang | EAGLE3 CUDA graph replay data | Batch shrink during decode | Illegal memory access |

★★★★★★★★★ This pattern family has 8 members across 3 frameworks!
★★★★★★★★★ Universal root cause: STATE LIFECYCLE MISMATCH
  → State persists beyond its validity window
  → No invalidation/cleanup mechanism
  → Silently used → wrong output or crash
```

---

## 6. RTX 4090 Implications for GRPO Training

### 6a. Direct Risk: Silent GRPO Corruption

```
★★★★★★★★★ WORST possible bug for long-running GRPO training:

Step 1 (healthy window):  prompt → correct 600-token reasoning → reward 0.85
Step 50 (degenerate window): SAME prompt → 80-token degenerate loop → reward 0.10
Step 100 (worse degenerate): ANY prompt → 15-token collapse → reward 0.02

→ Rewards are INCONSISTENT across steps for the SAME prompt
→ Advantage estimates become UNRELIABLE
→ Policy gradient updates on CORRUPTED signals
→ Training silently degrades without ANY error indication

★★★★★★★★★ verl HYBRID sleep/wake exacerbates this:
  → Rollout engine runs continuously for hours
  → SSM state accumulates corruption → degeneracy worsens
  → Sleep/wake cycle (LoRA adapter swap) adds MORE state churn
  → Sleep_level=1 tags=["kv_cache"] → BUT GDN/SSM state is NOT "kv_cache"
    → SSM state may NOT be properly preserved during sleep/wake!
  → ★★★★★★★★ CRITICAL: does verl's sleep/wake preserve GDN SSM state?
    → If not → EACH sleep/wake cycle introduces fresh SSM corruption
    → Worse than steady degradation → periodic corruption spikes
```

### 6b. Architecture Risk: SM89 vs SM80

```
★★★★★★★★★ A100 (SM80) bug → RTX 4090 (SM89) risk assessment:

SAME components:
  → FlashInfer pool API → same on SM89
  → SGLang mamba scheduler → same on SM89
  → SSM state management → same on SM89
  → Radix cache + SSM coupling → same on SM89

DIFFERENT on SM89:
  → CUDA capability (8,9) vs (8,0) → minor numerical differences
  → Memory size (24 GiB vs 80 GiB) → MORE pool pressure on RTX 4090
  → Power management → RTX 4090 power cap behavior different
  → No thread block clusters, no TMA → same as A100 (both non-Hopper)

★★★★★★★★★ RTX 4090 has HIGHER risk than A100:
  → 24 GiB vs 80 GiB → SSM pool fills faster → more slot reuse → more aliasing
  → RTX 4090 GRPO uses max_model_len ~4K-8K (vs 262K on A100)
    → Lower KV/SSM pool pressure BUT higher reuse frequency per token
  → GRPO runs for HOURS → uptime accumulation worse on constrained GPU

★★★★★★★★★ Qwen3.6-27B-FP8 viability on RTX 4090:
  → 27B FP8 model ≈ ~13.5 GiB weights → fits in 24 GiB (barely)
  → With SSM state + KV cache → likely OOM at max_model_len > 4K
  → ★★★★★★★★ Qwen3.6-27B-FP8 NOT viable on RTX 4090 for GRPO
    (too large, OOM risk, PLUS #28679 degeneracy bug)
  → ★★★★★★★★ But smaller GDN models (Qwen3-8B hybrid?) would be
    viable AND affected by the SAME #28679 bug pattern
```

### 6c. Mitigation Strategies for RTX 4090 GRPO

```
★★★★★★★★★ IMMEDIATE mitigations (before bug is fixed):

M1: Periodic engine restart during GRPO training
  → Restart SGLang/verl rollout engine every N steps
  → Clears accumulated SSM state corruption
  → Trade-off: restart latency (seconds) vs corruption risk
  → ★★★★★★★★ For RTX 4090: restart every 20-50 steps recommended
  → verl HYBRID mode already does sleep/wake → could integrate restart

M2: Output quality monitoring during GRPO
  → Track average completion_tokens per step
  → Track reasoning_tokens ratio per step
  → If completion_tokens drops below threshold → flag as corrupted
  → ★★★★★★★★ verl reward system could REJECT degenerate completions
  → Add reward = 0 for completions below token threshold
  → Prevents corrupted completions from poisoning policy gradient

M3: mamba-scheduler-strategy choice
  → extra_buffer: intermittent catastrophic (worse for GRPO)
  → no_buffer: systematic 6% degradation (predictable, manageable)
  → ★★★★★★★★ For GRPO: no_buffer may be SAFER than extra_buffer
    → Predictable 6% degradation < intermittent total collapse
    → BUT: no_buffer + disable-radix-cache = 11% degradation
    → Trade-off depends on tolerance vs catastrophic risk

M4: SSM state dtype and pool size tuning
  → Reporter asked about mamba_ssm_dtype / mamba_backend / max_mamba_cache_size
  → No answer yet → need maintainer guidance
  → ★★★★★★★★ Potential: lower SSM state dtype → less memory → less reuse
    → But: numerical precision loss → may worsen degradation

M5: CUDA graph avoidance
  → Reporter already uses disable_piecewise_cuda_graph=True
  → ★★★★★★★★ CUDA graph: True still shown in degenerate logs
    → piecewise disabled but OTHER cuda graphs still active?
    → Need: --disable-cuda-graph or enforce_eager equivalent for SGLang
    → enforce_eager=True in vLLM → need equivalent in SGLang

★★★★★★★★★ LONG-TERM fix (requires SGLang upstream):
  → Fix SSM state lifecycle coupling to radix cache
  → Fix mamba radix page alignment (#28692)
  → Add SSM state invalidation/verification mechanism
  → Add health check: if SSM state enters degenerate attractor → force reset
```

---

## 7. Timeline and Priority Assessment

```
★★★★★★★★★ Priority: P7 → significant for GRPO but needs upstream fix

Justification:
  → Bug affects GDN models in SGLang (Qwen3.6, potentially other hybrid GDN)
  → Silent corruption is dangerous but detectable with output monitoring
  → RTX 4090 risk is HIGH (same FlashInfer pool, same SSM state management)
  → Qwen3.6-27B-FP8 NOT viable on RTX 4090 regardless (OOM)
  → Smaller GDN models ARE viable AND affected → need monitoring

Potential P6 escalation:
  → If root cause is confirmed as SSM state lifecycle mismatch
    → Pattern applies across frameworks → cross-framework contribution
  → If RTX 4090 shows same degeneracy on smaller GDN models
    → Direct impact on RTX 4090 GRPO training → P6

Monitor actions:
  → Track #28679 for maintainer response
  → Track #28692 (mamba radix fix) — may partially address #28679
  → Track #28695 (ReplaySSM) — alternative state management approach
  → Track #20791 context — both scheduler strategies now have bugs
  → Check vLLM GDN decode for similar "worsening over uptime" pattern
```

---

## 8. Key Takeaways

```
★★★★★★★★★ 1. SILENT CORRUPTION IS THE WORST GRPO BUG PATTERN
  → No error, no NaN, no crash — just wrong output that looks normal
  → Worsens over uptime → longer training = more corruption
  → Clears on restart → no self-correction mechanism

★★★★★★★★★ 2. BOTH MAMBA-SCHEDULER STRATEGIES HAVE BUGS IN SGLang
  → extra_buffer (#28679): intermittent catastrophic degeneracy
  → no_buffer (#20791): systematic 6-11% accuracy degradation
  → NO safe configuration exists for GDN on SGLang currently

★★★★★★★★★ 3. NOT A DSV4 FAILURE — BUT SAME PATTERN FAMILY
  → "State lifecycle mismatch → stale state → silent wrong output"
  → 8 members across 3 frameworks (see Section 5 table)
  → Universal root cause: state persists past validity window

★★★★★★★★★ 4. RTX 4090 GRPO: ADD OUTPUT QUALITY MONITORING
  → Track completion_tokens, reasoning_tokens per step
  → Reject completions below threshold (reward = 0)
  → Periodic engine restart every 20-50 steps

★★★★★★★★★ 5. #28692 MAMBA RADIX FIX MAY PARTIALLY RESOLVE #28679
  → Page alignment bug creates invalid empty radix nodes
  → Invalid nodes + SSM coupling = corruption pathway
  → Merge #28692 → retest #28679 → check if degeneracy persists

★★★★★★★★★ 6. GRPO-SPECIFIC: does verl sleep/wake preserve GDN SSM state?
  → Sleep_level=1 tags=["kv_cache"] → KV cache preserved
  → BUT: GDN/SSM state is NOT "kv_cache" → may NOT be preserved!
  → Each sleep/wake cycle could introduce SSM state corruption
  → ★★★★★★★★ MUST verify: verl sleep/wake + GDN model = state corruption?
```

---

## 9. Related Issues Tracker

```
Directly related:
  → SGLang #28679 (this issue) — GDN intermittent decode degeneracy
  → SGLang #20791 (CLOSED) — no_buffer SSM state aliasing → BOTH strategies now buggy
  → SGLang #28692 (NEW June 19) — mamba radix page alignment fix → may partially address
  → SGLang #28695 (NEW June 19) — ReplaySSM Ring Spec-Verify → alternative SSM state approach
  → SGLang #23687 (OPEN) — FP8 weight_scale_inv dropped at load → reporter ruled this out

Same pattern family:
  → SGLang #28676 (OPEN) — MXFP8 cache clobber on RL weight reload
  → SGLang #28612 (OPEN) — DSV4 C128 state mapping lifecycle fix
  → SGLang #28569 (OPEN) — EAGLE3 CUDA graph replay crash
  → SGLang #28618/#28620 (OPEN) — SM89 DSV4-Flash-FP8 (RTX 4090 deployment pathway)
  → vLLM #45819 (OPEN) — GDN batch invariance (different scope, same GDN layer)
  → vLLM #45972 (MERGED) — DSV4 cudagraph revert (same "stale state" pattern)
  → vLLM-Ascend #10684 (OPEN) — DSA Hadamard sleep/wake ALL-ZERO (same "state lost" pattern)

RTX 4090 GRPO implications:
  → verl HYBRID sleep/wake — does it preserve GDN SSM state?
  → verl #6572 (OPEN) — full determinism
  → DeepSpeed #8068 — gradient clipping (GRPO MUST set 1.0)
  → DeepSpeed #8061 — overlap_comm NaN (MUST overlap_comm=False)
```

---

## 10. Update Log

```
2026-06-19: Initial deep reading created. 0 comments on issue, 0 reviews.
  No maintainer response yet. Issue created ~22 hours ago.
  Next update: check for maintainer triage/label assignment + comments.
```
