# vLLM #46085: aot_eager Piecewise Compilation Backend — Deep Reading

> 2026-06-19 | PR #46085 | Author: sachinkademane (first vLLM contribution)
> ★★★★★★★★ aot_eager = Dynamo + AOTAutograd WITHOUT Inductor → functionalization + decomposition but NO fusion → SM89 batch-invariant BY DESIGN!
> ★★★★★★★★ Potential RTX 4090 middle ground: aot_eager + piecewise CUDA graphs = deterministic + functionalization + graph acceleration
> ★★★★★★★★ P9 Fusion Guard complementary: aot_eager blocks ALL Inductor fusions → P9 blocks only reduction fusions → aot_eager more aggressive but broader guarantee
> ★★★★★★★★ DSV4 caveat: aot_eager avoids Inductor but NOT CUDA graph dynamic routing → DSV4 still needs piecewise + aot_eager or enforce_eager

---

## 1. Issue/PR Metadata

```
PR Number:      #46085
Title:          [Compile] Add aot_eager backend for piecewise compilation
Author:         sachinkademane (first vLLM contribution, external contributor)
State:          OPEN (mergeable_state: blocked, no CI run yet)
Created:        2026-06-18T18:37:51Z
Updated:        2026-06-18T18:51:23Z
Draft:          No (full PR, not draft)
Base:           vllm-project:main (SHA: 79ca54d)
Head:           sachinkademane:feature/compile-aot-eager-backend (SHA: 658dec4)
Additions:      71
Deletions:      8
Changed Files:  5
Commits:        1 (single commit PR)
Labels:         None (no labels yet)
Assignees:      None
Milestone:      None
AI Assistance:  Claude used in preparing the change (stated in commit message)
Requested Reviewers (11):
  tlrmchlsmth, mgoin, zou3519, ProExpertProg, BoyuanFeng,
  youkaichao, houseroad, yewentao256, WoosukKwon,
  robertgshaw2-redhat, vadiklyutiy
Review Comments: 0 (inline code review comments)
Issue Comments:  2 (bot welcome + author requesting CI)
Reviews:         0 (no formal reviews yet)
```

---

## 2. Full Description

### PR Body (verbatim)

```
## Summary

Adds an `aot_eager` backend option for `CompilationMode.VLLM_COMPILE` (mode 3)
piecewise compilation, mirroring PyTorch's `aot_eager` backend. The graph is
traced through AOTAutograd with a no-op forward/backward compiler — exercising
functionalization, decompositions, and the min-cut partitioner — without invoking
Inductor.

Primarily useful for debugging: it isolates Dynamo/AOTAutograd issues from
Inductor codegen issues. Previously, mode-3 piecewise compilation only accepted
`eager` and `inductor`.

## Changes

- New `AOTEagerAdaptor` in `compiler_interface.py`. Deep-copies the graph
  (AOTAutograd mutates in place), runs torch's `aot_eager`, and returns
  `(runner, None)` — no cache handle, since AOTAutograd doesn't support
  vLLM's out-of-band caching.
- The runner wraps each call in `record_function("Torch-Compiled Region: N")`
  so execution traces match stock `torch.compile(backend="aot_eager")` topology,
  letting profiler/HTA tooling attribute aten ops to per-subgraph regions.
  The subgraph index is passed via `compiler_config["vllm_subgraph_index"]`
  from `CompilerManager`.
- `make_compiler()`, the piecewise validation list, and `init_backend` accept
  `"aot_eager"`.
- `VLLM_USE_AOT_COMPILE` is disabled for `aot_eager` (like `eager`), since
  it doesn't support caching.

## Testing

Verified end-to-end with a Llama 3.1 8B run using mode-3 piecewise compilation:

  vllm bench throughput --model llama_3_1_8b_instruct \
      --dtype bfloat16 -cc.mode=3 -cc.backend=aot_eager \
      --gpu-memory-util 0.9 --no-enable-prefix-caching \
      --input-len 128 --output-len 128 --max-model-len 4096

Engine init succeeds, the run completes (≈2773 total tokens/s, 1000 prompts),
and an ExecutionTraceObserver capture shows 33 `Torch-Compiled Region: N`
parents — one per piecewise subgraph — matching stock torch.compile topology.
Invalid backends still rejected with the existing piecewise-compilation error.

## Notes

- AI assistance (Claude) was used in preparing this change.
```

### Commit Message (verbatim)

```
[Compile] Add aot_eager backend for piecewise compilation

Adds an `aot_eager` backend option for `CompilationMode.VLLM_COMPILE`
piecewise compilation, mirroring PyTorch's `aot_eager` backend. It traces
through AOTAutograd with a no-op forward/backward compiler, exercising
functionalization, decompositions, and the min-cut partitioner without
invoking Inductor. Primarily useful for debugging: it isolates
Dynamo/AOTAutograd issues from Inductor codegen issues.

The runner wraps each compiled subgraph in a `record_function("Torch-Compiled
Region: N")` so execution traces match stock torch.compile + aot_eager
topology. The subgraph index is passed via `compiler_config` from
`VllmBackend` rather than parsed out of the cache key.

AI assistance was used (Claude).
```

---

## 3. Comments/Reviews

### Issue Comments (2)

```
Comment 1 (2026-06-18T18:38:30Z): github-actions[bot]
  Standard vLLM PR welcome message + CI instructions.
  "PRs do not trigger a full CI run by default. Once the PR is approved and
  ready to go, your PR reviewer(s) can run CI to test the changes comprehensively
  before merging."

Comment 2 (2026-06-18T18:51:23Z): sachinkademane (author)
  "Hi! This is my first contribution. Could a maintainer add the ready label
  to trigger CI? Thanks. cc @youkaichao @zou3519, and @bnellnm"
```

### Review Comments (0)

```
No inline code review comments yet. PR was created just hours ago.
```

### Reviews (0)

```
No formal reviews yet. 11 reviewers requested including:
  youkaichao (vLLM compile lead), zou3519 (PyTorch core),
  WoosukKwon (vLLM founder), robertgshaw2-redhat (Red Hat),
  mgoin (vLLM core), BoyuanFeng (vLLM compiler)
```

---

## 4. Technical Analysis

### 4.1 What is aot_eager piecewise compilation?

```
★★★★★★★★★ torch.compile stack — 3 stages + 3 backends:

Stage 1: Dynamo
  → C-level eval_frame hook → bytecode → FX Graph
  → Graph breaks → multiple subgraphs → piecewise compilation
  → ALL backends go through Dynamo

Stage 2: AOTAutograd
  → Joint fwd+bwd tracing → functionalization → decompositions → min-cut partition
  → Functionalization: in-place ops → out-of-place (correct mutation handling)
  → Decompositions: composite ops → simpler ops (aten decomposition rules)
  → Min-cut partition: fwd/bwd separation → memory optimization
  → aot_eager and inductor go through AOTAutograd; eager SKIPS it

Stage 3: Inductor (code generation)
  → FX → IR → Scheduler → Triton/C++ codegen
  → Fusion scheduling (10 rounds → can_fuse checks)
  → Triton kernel generation + CachingAutotuner (XBLOCK selection)
  → autotune_at_compile_time → batch-dependent configs on SM89!
  → Only inductor backend goes through Inductor; eager and aot_eager SKIP it

★★★★★★★★★ Three vLLM compilation backends — what each does:

Backend    | Dynamo | AOTAutograd | Inductor | Caching | CUDA graphs compat
---------- | ------ | ----------- | -------- | ------- | -------------------
eager      | Yes    | No          | No       | No      | No (pure eager)
aot_eager  | Yes    | Yes         | No       | No      | Piecewise only
inductor   | Yes    | Yes         | Yes      | Yes     | Piecewise + Full

★★★★★★★★★ aot_eager = Dynamo + AOTAutograd WITHOUT Inductor:
  → Functionalization: in-place → out-of-place → correct mutation handling
  → Decompositions: composite ops → simpler ops → batch_invariant overrides work!
  → Min-cut partition: fwd/bwd separated → memory optimization
  → BUT: no Inductor fusion → no Triton codegen → no autotuning → no batch-dependent risk!
  → BUT: no caching support → VLLM_USE_AOT_COMPILE disabled → no AOT compilation caching

★★★★★★★★★ Why this matters for vLLM mode-3 piecewise compilation:

vLLM CompilationMode.VLLM_COMPILE (mode 3):
  → Graph breaks at specific ops (splitting_ops)
  → Each piece → compiled separately → piecewise subgraphs
  → Previously: only eager and inductor backends
  → Now: also aot_eager backend → middle ground!

Mode 3 piecewise with aot_eager:
  → Dynamo traces → identifies graph breaks → creates subgraphs
  → Each subgraph → AOTAutograd → functionalization + decomposition + partition
  → Each partitioned subgraph → NO-OP compiler → just runs eagerly
  → 33 piecewise subgraphs (verified with Llama 3.1 8B)
  → record_function("Torch-Compiled Region: N") → profiler attribution
```

### 4.2 How aot_eager differs from inductor/torch.compile backend

```
★★★★★★★★★ Key architectural differences:

| Aspect                | eager              | aot_eager          | inductor            |
|---------------------- |--------------------|--------------------|--------------------|
| Dynamo tracing       | Yes                | Yes                | Yes                |
| AOTAutograd          | SKIPPED            | Full               | Full               |
| Functionalization    | No                 | Yes                | Yes                |
| Decompositions       | No                 | Yes                | Yes                |
| Min-cut partition    | No                 | Yes                | Yes                |
| Inductor codegen     | No                 | SKIPPED            | Full               |
| Triton kernels       | No                 | No                 | Yes (generated)    |
| Fusion scheduling    | No                 | No                 | Yes (10 rounds)    |
| CachingAutotuner     | No                 | No                 | Yes (XBLOCK)       |
| Batch invariance     | YES (pure eager)   | YES (no fusion)    | NO on SM89         |
| Performance          | Slowest            | Medium             | Fastest            |
| Profiler attribution | No                 | Yes (region tags)  | Yes                |
| AOT compile caching  | No                 | No                 | Yes                |
| CUDA graph compat    | No                 | Piecewise only     | Piecewise + Full   |

★★★★★★★★★ aot_eager vs inductor — SM89 batch invariance:

Inductor on SM89:
  → Fusion: RMSNorm + mean → fused kernel → batch-dependent accumulation!
  → Autotuning: CachingAutotuner → XBLOCK selected per shape → different per batch
  → Result: VLLM_BATCH_INVARIANT=1 fails on SM89 (#39096)
  → Workaround: enforce_eager=True → disables compile AND graphs → slow!

aot_eager on SM89:
  → No Inductor fusion → RMSNorm, mean execute separately → vLLM overrides work!
  → No Triton codegen → no CachingAutotuner → no XBLOCK variability!
  → Result: batch-invariant BY DESIGN (no fusion → no batch-dependent kernel)
  → Potential: piecewise CUDA graphs + aot_eager = deterministic + graph acceleration!

★★★★★★★★★ aot_eager vs eager — what aot_eager adds beyond pure eager:

Pure eager (enforce_eager=True):
  → No Dynamo, no AOTAutograd, no functionalization, no decomposition
  → In-place ops execute as-is → mutation tracking not optimized
  → No graph partitioning → no fwd/bwd separation
  → No profiler region attribution → hard to debug performance
  → CUDA graphs disabled entirely → maximum kernel launch overhead

aot_eager:
  → Dynamo tracing → graph breaks → piecewise subgraphs
  → AOTAutograd → functionalization → in-place→out-of-place → correct mutation
  → Decompositions → composite ops → simpler → overrides work on decomposed ops
  → Min-cut partition → fwd/bwd separated → memory optimization
  → record_function region tags → profiler can attribute ops to subgraphs
  → Compatible with piecewise CUDA graphs → kernel launch overhead reduced!

★★★★★★★★★ Summary: aot_eager provides FUNCTIONALIZATION BENEFITS WITHOUT FUSION RISK:

What you GET with aot_eager:
  ✓ Functionalization (correct mutation handling)
  ✓ Decompositions (ops broken down → override-friendly)
  ✓ Min-cut partition (fwd/bwd separation → memory optimization)
  ✓ Profiler attribution (record_function region tags)
  ✓ Batch invariance on SM89 (no Inductor fusion → deterministic)

What you LOSE with aot_eager:
  ✗ Inductor fusion speedups (reduction+pointwise+GEMM fusion)
  ✗ Triton kernel generation (custom optimized kernels)
  ✗ CachingAutotuner (shape-specific kernel configs)
  ✗ AOT compilation caching (VLLM_USE_AOT_COMPILE disabled)
  ✗ Full CUDA graph capture (piecewise only, not FULL mode)
```

### 4.3 Relevance to SM89 batch invariance and P9 Fusion Guard

```
★★★★★★★★★ DIRECT relevance to SM89 batch invariance — this is the key insight!

Current SM89 situation (#39096):
  enforce_eager=True → disables compile AND graphs → CORRECT but SLOW
  enforce_eager=False → Inductor + graphs → FAST but batch-dependent → WRONG!

aot_eager provides a THIRD option:
  -cc.mode=3 -cc.backend=aot_eager → compile without Inductor → CORRECT!
  → Functionalization + decomposition → correctness benefits
  → No Inductor fusion → no batch-dependent → deterministic!
  → Piecewise CUDA graphs possible → kernel launch overhead reduced!

★★★★★★★★★ P9 Fusion Guard vs aot_eager — complementary approaches:

| Aspect            | P9 Fusion Guard          | aot_eager backend        |
|------------------ |--------------------------|--------------------------|
| Mechanism         | WhyNoFuse in choices.py  | Skip Inductor entirely   |
| Scope             | Reduction fusions only   | ALL Inductor fusions     |
| Precision         | Selective blocking       | Complete Inductor bypass |
| Implementation    | 5 lines in choices.py    | 50 lines AOTEagerAdaptor |
| Performance       | Preserves pointwise fusion| No fusion at all         |
| Production intent | Yes (production guard)   | Debugging (author intent)|
| vLLM integration  | PyTorch-level (upstream) | vLLM-level (this PR)     |
| Autotuning risk   | Still present            | Not present (no Inductor)|
| Caching           | Works with Inductor      | Not supported             |
| CUDA graphs       | Piecewise + Full         | Piecewise only            |

★★★★★★★★★ P9 + aot_eager are NOT competing — they serve different purposes:

P9 (production path):
  → Minimal change → 5 lines → upstreamable to PyTorch
  → Selective → blocks reduction fusions → preserves good fusions
  → Performance-oriented → still gets pointwise/GEMM fusion speedups
  → Works with Inductor → full compilation benefits minus bad fusions

aot_eager (debug/validation path):
  → Complete Inductor bypass → isolates Dynamo/AOTAutograd issues
  → Broader guarantee → ALL fusions blocked → cannot have batch-dependent issue
  → Can validate P9: compare Inductor+P9 vs aot_eager → same result = P9 correct!
  → Debugging tool → profiler attribution → region tags → systematic debugging

★★★★★★★★★ aot_eager as P9 VALIDATION tool:

On SM89 RTX 4090:
  Step 1: Run with aot_eager → guaranteed batch-invariant (no Inductor)
  Step 2: Run with Inductor + P9 → should be batch-invariant (P9 blocks bad fusions)
  Step 3: Compare outputs → bitwise identical = P9 is correct!
  Step 4: If different → P9 is missing some fusion → aot_eager identifies which!

This gives us a SYSTEMATIC validation methodology:
  aot_eager = ground truth (no fusion → definitely correct)
  Inductor + P9 = proposed solution (selective blocking → should be correct)
  Compare → validate P9's completeness!

★★★★★★★★★ aot_eager as SM89 batch invariance debugging tool:

Currently debugging SM89 batch invariance is hard:
  → Which specific fusion causes batch-dependent output?
  → Is it RMSNorm+mean fusion? matmul+activation fusion? RoPE+attention fusion?
  → With Inductor: all fusions present → can't isolate which one breaks!

With aot_eager:
  → No fusions → baseline batch-invariant output
  → Then switch to Inductor → batch-dependent output
  → Then add P9 → blocks reduction fusions → should restore invariance
  → If still batch-dependent → other fusion type causes it → need more blocking!

★★★★★★★★★ NEW RTX 4090 deployment possibility:

Current RTX 4090 GRPO deployment: enforce_eager=True
  → Disables torch.compile AND CUDA graphs → maximum performance penalty
  → Required for correctness → batch-dependent outputs = wrong rewards!

Potential with aot_eager:
  -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=PIECEWISE
  → Functionalization benefits → correct mutation handling in model forward
  → No Inductor fusion → batch-invariant → correct rewards!
  → Piecewise CUDA graphs → reduced kernel launch overhead!
  → Performance: likely BETWEEN enforce_eager=True and Inductor!

For verl GRPO on RTX 4090:
  → Rollout engine with aot_eager → deterministic inference → correct rewards!
  → Training engine with ZeRO-2 → no compile → already deterministic
  → Both sides batch-invariant → reward computation correct!
```

### 4.4 Connection to vLLM #45731 (PyTorch 2.13)

```
★★★★★★★★★ vLLM #45731: Update PyTorch to 2.13.0, Triton to 3.7.1

Key connections:
  → #45731 updates Triton → 3.7.1 → affects Inductor codegen
  → aot_eager SKIPS Inductor → Triton version has NO effect on aot_eager!
  → #45731 updates torch.compile → Dynamo + AOTAutograd improvements
  → aot_eager uses Dynamo + AOTAutograd → BENEFITS from torch.compile updates!
  → aot_eager uses torch._dynamo.backends.debugging.aot_eager → PyTorch 2.13 may improve this

★★★★★★★★★ If #45731 lands BEFORE #46085:
  → aot_eager implementation uses PyTorch 2.13's AOTAutograd → updated functionalization
  → PyTorch 2.13's decompositions may be more complete → better op breakdown
  → PyTorch 2.13's min-cut partition may be more optimal → better fwd/bwd separation

★★★★★★★★★ If #46085 lands BEFORE #45731:
  → aot_eager implementation uses current PyTorch → tested on current version
  → When #45731 upgrades PyTorch → aot_eager automatically gets improvements
  → No rework needed → aot_eager is a thin wrapper around PyTorch's aot_eager!

★★★★★★★★★ Triton 3.7.1 implications for aot_eager:
  → Triton 3.7.1: new autotuning, new SM120 support, improved codegen
  → aot_eager: no Triton codegen → Triton 3.7.1 irrelevant for aot_eager
  → BUT: if someone compares aot_eager vs Inductor → Triton version affects Inductor
  → Triton 3.7.1 may introduce NEW batch-dependent fusions → aot_eager still safe!
```

### 4.5 Connection to vLLM #45819 (GDN batch invariance)

```
★★★★★★★★★ vLLM #45819: Add batch invariance support to GDN_ATTN backend

Current GDN on SM89:
  → enforce_eager=True REQUIRED → Inductor fusion breaks GDN batch invariance
  → GDN (Gated-Delta-Net) is a recurrent/linear-attention architecture
  → State updates (delta-rule) are in-place → functionalization helps!
  → But Inductor fuses state updates → batch-dependent → WRONG!

★★★★★★★★★ aot_eager for GDN on SM89:

Potential improvement:
  → aot_eager → functionalization → in-place state updates → out-of-place → correct!
  → No Inductor fusion → state update ops execute separately → deterministic!
  → vLLM batch_invariant.py overrides work on decomposed ops → correct!

  Configuration:
    -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=PIECEWISE
    VLLM_BATCH_INVARIANT=1

  This could give GDN on SM89:
    ✓ Functionalization (correct state update handling)
    ✓ No fusion (deterministic GDN computation)
    ✓ Batch-invariant (VLLM_BATCH_INVARIANT=1 + no fusion risk)
    ✓ Piecewise CUDA graphs (reduced launch overhead)

★★★★★★★★★ But: GDN's recurrent nature → piecewise CUDA graphs may not help much:
  → Recurrent models: sequential processing → fewer parallel kernels
  → CUDA graph benefit: eliminates kernel launch overhead for parallel kernels
  → For sequential ops: less benefit → launch overhead is small vs compute time
  → Still: piecewise CUDA graphs provide SOME speedup for non-recurrent parts!
```

### 4.6 Connection to vLLM #45683 (MoE combine)

```
★★★★★★★★★ vLLM #45683: Deterministic MoE combine under VLLM_BATCH_INVARIANT

MoE combine issue:
  → Under DP + EP → cross-rank summation in MoE combine → not stable
  → reduce_scatterv needs deterministic execution → NCCL level issue
  → This is a COMMUNICATION operation → not Inductor fusion → aot_eager doesn't help!

★★★★★★★★★ aot_eager is NOT relevant to MoE combine determinism:
  → MoE combine = collective operation (NCCL reduce_scatterv)
  → Inductor doesn't fuse collective operations → already separate
  → The batch-dependency is in NCCL's summation order → not in Inductor
  → #45683 fixes this with deterministic reduce_scatter routing → aot_eager irrelevant

★★★★★★★★★ BUT: aot_eager helps with MoE EXPERT computation:
  → Expert forward: matmul + activation + output → Inductor can fuse
  → With aot_eager: expert ops execute separately → no fusion risk
  → On SM89: MoE expert computation becomes deterministic with aot_eager!
  → This is the compute path, not the communication path → aot_eager helps compute!
```

### 4.7 Connection to DSV4 systematic instability (enforce_eager alternative?)

```
★★★★★★★★★ DSV4 systematic instability — 9 issues in 4 days across 3 frameworks:

DSV4 root cause: MORE dynamic routing than any previous model
  → MoE (expert selection per token)
  → DSA (sparse attention indexer)
  → MTP (multi-token prediction)
  → Online Compress (KV cache compression)
  → MLA DCP (multi-head latent query replication)

★★★★★★★★★ Can aot_eager replace enforce_eager=True for DSV4?

Current DSV4 on SM89: enforce_eager=True → disables compile AND graphs → CORRECT but SLOW

aot_eager for DSV4:
  → Functionalization → correct mutation handling → good for DSA/Online Compress state updates
  → No Inductor fusion → no batch-dependent outputs → deterministic compute
  → BUT: CUDA graph dynamic routing STILL a problem!

★★★★★★★★★ Critical distinction: TWO separate DSV4 issues on SM89:

Issue 1: Inductor fusion → batch-dependent outputs (COMPUTE correctness)
  → enforce_eager=True: disables compile → no fusion → fixes this
  → aot_eager: disables Inductor → no fusion → ALSO fixes this!
  → P9 Fusion Guard: blocks reduction fusions → ALSO fixes this (selectively)

Issue 2: CUDA graph dynamic routing → stale routing decisions (GRAPH correctness)
  → MoE expert selection: different per step → graph replay uses stale selection
  → DSA indexer: different per step → graph replay uses stale index
  → MTP draft: different per step → graph replay uses stale draft
  → enforce_eager=True: disables graphs → no stale routing → fixes this
  → aot_eager: graphs still possible → stale routing NOT fixed!
  → cudagraph_mode=NONE: disables graphs → no stale routing → fixes this

★★★★★★★★★ DSV4 on SM89 with aot_eager — two configurations:

Config A: aot_eager + piecewise CUDA graphs (for non-DSV4 dynamic-routing models):
  → -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=PIECEWISE
  → Correct: no Inductor fusion → deterministic compute
  → CUDA graphs: piecewise → ops within each region captured → graph replay
  → Works for: Llama, Qwen (standard attention) → no dynamic routing → graph replay safe!
  → Faster than: enforce_eager=True (graph acceleration for non-dynamic parts)

Config B: aot_eager + cudagraph_mode=NONE (for DSV4 and other dynamic-routing models):
  → -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=NONE
  → Correct: no Inductor fusion → deterministic compute
  → No CUDA graphs → no stale dynamic routing → correct routing
  → Functionalization + decomposition → better than pure eager (enforce_eager=True)
  → Faster than: enforce_eager=True? Depends on functionalization benefit vs compile overhead

★★★★★★★★★ Is Config B actually faster than enforce_eager=True?

enforce_eager=True:
  → No compile → no functionalization → no decomposition → pure eager execution
  → No CUDA graphs → kernel launch overhead for every op
  → BUT: no compile overhead → no Dynamo tracing → no AOTAutograd → no warmup

aot_eager + cudagraph_mode=NONE:
  → Compile overhead: Dynamo tracing + AOTAutograd → warmup time
  → Functionalization: in-place→out-of-place → may create extra copies → slight overhead
  → Decomposition: composite→simpler → some ops decomposed → may be slower or faster
  → Min-cut partition: fwd/bwd separated → potential memory savings → enables larger batch
  → No CUDA graphs → same kernel launch overhead as enforce_eager

Net result: Probably SIMILAR throughput but with functionalization benefits:
  → aot_eager: functionalization + partition → correctness + memory optimization
  → enforce_eager=True: no functionalization → simpler but less correct mutation handling
  → For DSV4: aot_eager may be better for state updates (DSA, Online Compress)!

★★★★★★★★★ DSV4-specific aot_eager advantage:
  → DSA indexer: state mutation → functionalization makes this correct → important!
  → Online Compress: in-place KV update → functionalization → out-of-place → correct!
  → MoE: no state mutation → functionalization less important
  → MTP: no state mutation → functionalization less important
  → aot_eager's functionalization is MOST valuable for DSV4's state-mutating components!
```

---

## 5. RTX 4090 Impact

```
★★★★★★★★★ RTX 4090 (SM89) — aot_eager as middle ground:

Current RTX 4090 deployment options:
  1. enforce_eager=True → CORRECT but SLOW → current MUST for GRPO
  2. Inductor (enforce_eager=False) → FAST but WRONG → batch-dependent!
  3. Inductor + P9 → CORRECT + FAST (blocked bad fusions) → proposed solution

NEW option with aot_eager:
  4. aot_eager + piecewise CUDA graphs → CORRECT + MEDIUM → new possibility!

★★★★★★★★★ RTX 4090 performance ranking (inference):

  Inductor (enforce_eager=False)   > aot_eager + PIECEWISE graphs > aot_eager + NONE > enforce_eager=True
  [WRONG on SM89]                    [CORRECT, medium speed]        [CORRECT, similar]  [CORRECT, slowest]

★★★★★★★★★ RTX 4090 GRPO training implications:

For GRPO on RTX 4090 with verl:
  → Rollout engine: vLLM with aot_eager → deterministic inference → correct rewards!
  → Training engine: DeepSpeed ZeRO-2 + CPU_Adam → already deterministic
  → Both: batch-invariant → reward computation correct → GRPO converges!

Configuration for verl GRPO on RTX 4090 with aot_eager:
  vllm rollout:
    -cc.mode=3
    -cc.backend=aot_eager      ← new! instead of enforce_eager=True
    VLLM_BATCH_INVARIANT=1     ← still needed for override activation
    -cudagraph_mode=PIECEWISE  ← piecewise CUDA graphs (for standard models)
    OR cudagraph_mode=NONE     ← no graphs (for DSV4/dynamic routing models)

★★★★★★★★★ Key RTX 4090 question: is aot_eager + PIECEWISE actually faster?

Benchmark data from PR:
  → Llama 3.1 8B with aot_eager: ≈2773 total tokens/s (1000 prompts)
  → This is WITHOUT CUDA graphs (just piecewise compilation)
  → With piecewise CUDA graphs: likely 5-10% faster (kernel launch elimination)
  → enforce_eager=True: likely similar or slower (no functionalization optimization)

But: this benchmark is NOT on SM89 specifically!
  → The benchmark was run on unspecified GPU → likely SM90+ (where Inductor is fine)
  → On SM89: aot_eager vs enforce_eager=True comparison needed!
  → On SM89: aot_eager + PIECEWISE vs Inductor + P9 comparison needed!

★★★★★★★★★ RTX 4090 MUST test these configurations:
  1. enforce_eager=True (current baseline) → throughput + correctness
  2. -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=NONE → throughput + correctness
  3. -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=PIECEWISE → throughput + correctness
  4. -cc.mode=3 -cc.backend=inductor + P9 (when P9 lands) → throughput + correctness

★★★★★★★★★ RTX 4090 memory implications:

aot_eager min-cut partition → fwd/bwd separation → memory optimization:
  → Forward: computes and stores activations needed for backward
  → Min-cut algorithm: minimizes activation storage → saves memory!
  → On RTX 4090 (24 GiB): memory savings → enables larger batch → better throughput!
  → BUT: aot_eager has no Inductor fusion → no fused kernel memory savings
  → Net: probably similar memory to enforce_eager=True + min-cut savings!

★★★★★★★★★ RTX 4090 verl GRPO specific:

verl HYBRID sleep/wake + aot_eager:
  → Rollout: aot_eager → deterministic inference → correct logprobs → correct rewards
  → Sleep: LoRA adapter unload → memory freed → weight sync
  → Wake: LoRA adapter reload → resume inference
  → aot_eager functionalization: correct LoRA merge/unmerge handling → important!
  → No Inductor → no fusion of LoRA merge with attention → deterministic LoRA!

But: aot_eager doesn't support caching → no VLLM_USE_AOT_COMPILE:
  → vLLM's AOT compilation caching: saves compiled artifacts for reuse
  → aot_eager: no caching → re-compiles every time → warmup overhead!
  → For GRPO: warmup happens once → not a recurring cost → acceptable!
  → For long-running serving: warmup every restart → minor overhead
```

---

## 6. Connection Map

```
★★★★★★★★★ Cross-framework connection map for #46085:

vLLM #46085 (aot_eager piecewise compilation)
  │
  ├──→ PyTorch #187636 (autotune_at_compile_time flip)
  │     → aot_eager SKIPS Inductor → autotune flip IRRELEVANT for aot_eager
  │     → BUT: aot_eager provides ground truth for validating #187636 effect
  │     → Compare: Inductor+#187636 vs aot_eager → isolate autotune vs fusion impact
  │
  ├──→ PyTorch #184119 (SM89 fp8→bf16 prologue fusion guard)
  │     → aot_eager SKIPS Inductor → #184119 IRRELEVANT for aot_eager
  │     → BUT: #184119 validates P9 thesis → same SM89 class → aot_eager is alternative
  │     → Together: #184119 protects specific fp8 fusion → aot_eager protects ALL fusions
  │
  ├──→ vLLM #39096 (SM<90 batch invariance breaks with compile)
  │     → DIRECTLY relevant! aot_eager = compile WITHOUT Inductor → fixes root cause!
  │     → aot_eager + VLLM_BATCH_INVARIANT=1 = batch-invariant on SM89
  │     → aot_eager as alternative to enforce_eager=True for SM89
  │     → ★★★★★★★★ STRONGEST connection!
  │
  ├──→ vLLM #45819 (GDN batch invariance)
  │     → aot_eager functionalization helps GDN state updates
  │     → aot_eager avoids Inductor fusion → GDN batch-invariant on SM89
  │     → Potential: GDN + aot_eager = deterministic on SM89!
  │
  ├──→ vLLM #45683 (MoE combine determinism)
  │     → aot_eager NOT relevant for MoE combine (communication-level, not compute)
  │     → BUT: aot_eager helps MoE EXPERT computation on SM89 (no fusion)
  │
  ├──→ vLLM #45731 (PyTorch 2.13 update)
  │     → aot_eager uses PyTorch's aot_eager → benefits from PyTorch updates
  │     → Triton 3.7.1 irrelevant (aot_eager skips Inductor)
  │
  ├──→ vLLM #42251 (Auto-compile CustomOp fallbacks under enforce_eager)
  │     → With aot_eager: CustomOp still functionalized + decomposed → may not need auto-compile
  │     → aot_eager provides functionalization without Inductor → CustomOp fallbacks different
  │
  ├──→ vLLM #41219 (Auto-upgrade compilation mode when cudagraph requires VLLM_COMPILE)
  │     → aot_eager needs mode=3 → this auto-upgrade logic should include aot_eager
  │     → When cudagraph_mode requires VLLM_COMPILE → backend could default to aot_eager on SM89!
  │
  ├──→ DSV4 systematic instability (9 issues, 3 frameworks)
  │     → aot_eager avoids Inductor → fixes compute correctness
  │     → aot_eager does NOT fix CUDA graph dynamic routing → DSV4 still needs piecewise/NONE
  │     → DSV4 on SM89: aot_eager + cudagraph_mode=NONE = correct (compute + routing)
  │
  ├──→ P9 Inductor Fusion Guard (our contribution)
  │     → COMPLEMENTARY: P9 selective blocking vs aot_eager complete bypass
  │     → P9 = production path → aot_eager = debug/validation path
  │     → aot_eager can validate P9: ground truth comparison
  │     → ★★★★★★★★ KEY: aot_eager + P9 = two-pronged SM89 determinism approach
  │
  ├──→ vLLM batch_invariant.py (SM89 overrides)
  │     → aot_eager decomposes ops → vLLM overrides work on decomposed ops!
  │     → On SM89: aot_eager + VLLM_BATCH_INVARIANT=1 = overrides work + no fusion bypass!
  │     → ★★★★★★★★ aot_eager makes batch_invariant.py MORE effective (decomposition)!
  │
  ├──→ verl #6572 (full determinism)
  │     → VLLM_BATCH_INVARIANT=1 + aot_eager = complete determinism on SM89
  │     → verl GRPO on RTX 4090: rollout with aot_eager → correct rewards!
  │
  ├──→ PyTorch aot_eager backend (torch._dynamo.backends.debugging.aot_eager)
  │     → vLLM's AOTEagerAdaptor mirrors this → same mechanism
  │     → PyTorch #138264: aot_eager incorrectly raising error for batch_norm → known issue!
  │
  └──→ PyTorch #187435 (no_fuse_region, per-op blocking)
        → aot_eager = global Inductor bypass → #187435 = per-op blocking
        → Together: aot_eager for debugging → #187435 for fine-grained production
```

---

## 7. Key Takeaways

```
★★★★★★★★★ Top 10 takeaways from #46085:

1. aot_eager = Dynamo + AOTAutograd WITHOUT Inductor
   → Functionalization + decomposition + partition → no fusion
   → Middle ground between pure eager and full Inductor

2. aot_eager is batch-invariant BY DESIGN on SM89
   → No Inductor fusion → no batch-dependent kernels → deterministic!
   → Alternative to enforce_eager=True → with functionalization benefits

3. aot_eager + piecewise CUDA graphs = RTX 4090 middle ground
   → For standard models (Llama, Qwen): deterministic + graph acceleration
   → For DSV4: aot_eager + cudagraph_mode=NONE (no dynamic routing graphs)

4. aot_eager validates P9 Fusion Guard
   → aot_eager = ground truth (no fusion → definitely correct)
   → Inductor + P9 = proposed solution (selective blocking → should be correct)
   → Compare → validate P9 completeness!

5. aot_eager makes vLLM batch_invariant.py more effective
   → Decomposition breaks composite ops → overrides can intercept decomposed ops
   → On SM89: overrides work because no Inductor fusion bypasses them

6. aot_eager does NOT solve DSV4 CUDA graph dynamic routing
   → Inductor fusion is compute-level → aot_eager fixes this
   → CUDA graph dynamic routing is graph-level → aot_eager does NOT fix this
   → DSV4 still needs piecewise/NONE cudagraph_mode

7. aot_eager is primarily a DEBUGGING tool (author intent)
   → Isolate Dynamo/AOTAutograd issues from Inductor issues
   → Profiler attribution (record_function region tags)
   → BUT: RTX 4090 deployment potential exists (batch-invariant by design)

8. aot_eager has limitations: no caching, no fusion speedups
   → VLLM_USE_AOT_COMPILE disabled → no AOT compilation caching
   → No Inductor → no Triton kernels → no fusion → slower than Inductor
   → Performance: between enforce_eager=True and Inductor

9. New contributor (sachinkademane), 0 reviews, 0 comments
   → Fresh PR → needs CI → needs reviewer attention
   → Small change (71 additions, 8 deletions) → easy to review
   → 11 requested reviewers including youkaichao (compile lead)

10. P9 + aot_eager = two-pronged SM89 determinism strategy
    → P9: production path (5 lines, selective blocking, preserves good fusions)
    → aot_eager: debug/validation path (complete bypass, ground truth, profiler)
    → Together: systematic SM89 determinism → debug with aot_eager → deploy with P9
```

---

## 8. Monitoring Status

```
★★★★★★★★★ PR Status: OPEN, 0 reviews, 0 comments, needs CI

Monitor Items:
  1. CI run: author requested "ready" label → needs maintainer action
  2. Reviews: 11 reviewers requested → expect comments from youkaichao/zou3519
  3. Potential reviewer concerns:
     → "Primarily for debugging" → reviewer may question production value
     → No caching support → reviewer may want caching added
     → graph_returns_tuple handling → correctness edge case
     → Deep-copy for AOTAutograd → performance concern (memory overhead)
  4. Integration path: if merged → available in next vLLM release
  5. SM89 testing: benchmark was on unspecified GPU → need SM89 verification

★★★★★★★★★ Action Items for our project:

  1. WATCH this PR → when merged → test aot_eager on RTX 4090
     → Compare: enforce_eager=True vs aot_eager throughput on SM89
     → Compare: aot_eager + PIECEWISE vs aot_eager + NONE on SM89
     → Validate: aot_eager outputs vs enforce_eager=True outputs → bitwise identical?

  2. USE aot_eager as P9 validation tool (when available)
     → Run model with aot_eager → ground truth batch-invariant output
     → Run same model with Inductor + P9 → compare outputs
     → If identical → P9 is correct → confidence for upstream submission
     → If different → identify missing P9 fusions → improve P9!

  3. COMMENT on #46085 when appropriate (after CI + reviews)
     → Highlight SM89 batch invariance relevance → RTX 4090 angle
     → Suggest: aot_eager as SM89 deployment option → not just debugging
     → Note: aot_eager + cudagraph_mode=NONE for DSV4 on SM89
     → ★★★★★★★★ MUST wait for reviews first → DON'T comment prematurely!

  4. UPDATE rtx4090_oss_contribution_tracker.py
     → Add #46085 monitoring → aot_eager backend → SM89 middle ground
     → Track: when merged → when available in release → when tested on RTX 4090

  5. UPDATE grpo_troubleshooter_4090.py / rtx4090_grpo_config_reference.py
     → Add aot_eager backend option → mode=3 → backend=aot_eager
     → For standard models: aot_eager + PIECEWISE cudagraph
     → For DSV4 models: aot_eager + NONE cudagraph
     → Validate: batch-invariant + functionalization benefits

★★★★★★★★★ Dependencies:

  #46085 (aot_eager) depends on:
    → vLLM mode-3 piecewise compilation infrastructure (already exists)
    → PyTorch's aot_eager backend (torch._dynamo.backends.debugging.aot_eager)
    → No dependency on #45731 (PyTorch 2.13) → works with current PyTorch

  #46085 enables:
    → SM89 batch-invariant debugging (aot_eager vs Inductor comparison)
    → P9 validation methodology (ground truth vs selective blocking)
    → RTX 4090 aot_eager deployment (alternative to enforce_eager=True)
    → GDN on SM89 with aot_eager (functionalization + determinism)

★★★★★★★★★ Risk Assessment:

  Low risk PR:
    → 71 additions, 8 deletions → small change
    → Mirrors existing PyTorch backend → well-understood mechanism
    → Only adds new backend option → doesn't change existing behavior
    → No caching → simpler than Inductor → less surface for bugs

  Potential risks:
    → Deep-copy overhead: AOTAutograd mutates in place → deep-copy needed
    → No caching: warmup overhead on every restart → production concern
    → graph_returns_tuple edge case: piecewise subgraphs may not return tuple
    → Functionalization overhead: in-place→out-of-place → may create extra copies
```

---

## 9. Code Changes Deep Analysis

### 9.1 AOTEagerAdaptor (compiler_interface.py, +50 lines)

```
★★★★★★★★★ Key implementation details:

class AOTEagerAdaptor(CompilerInterface):
    name = "aot_eager"

    def compile(self, graph, example_inputs, compiler_config, compile_range, key=None):
        # Step 1: Count compilation
        compilation_counter.num_aot_eager_compiles += 1

        # Step 2: Import PyTorch's aot_eager backend
        from torch._dynamo.backends.debugging import aot_eager
        from torch._inductor.compile_fx import graph_returns_tuple
        from torch.profiler import record_function

        # Step 3: Set functorch config (same as other backends)
        set_functorch_config()

        # Step 4: DEEP-COPY the graph (AOTAutograd mutates in place!)
        # → Critical: shared graph must not be mutated
        # → Ref: https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        # Step 5: Handle tuple output (piecewise subgraphs may produce single tensor)
        unwrap_output = not graph_returns_tuple(graph)
        if unwrap_output:
            output_node = next(n for n in graph.graph.nodes if n.op == "output")
            output_node.args = ((output_node.args[0],),)
            graph.recompile()

        # Step 6: Run PyTorch's aot_eager (traces AOTAutograd, no-op compiler)
        compiled_graph = aot_eager(graph, example_inputs)

        # Step 7: Get subgraph index from compiler_config (passed by VllmBackend)
        subgraph_index = (compiler_config or {}).get("vllm_subgraph_index", 0)
        region_name = f"Torch-Compiled Region: {subgraph_index}"

        # Step 8: Wrap in record_function for profiler attribution
        def runner(*args):
            with record_function(region_name):
                out = compiled_graph(*args)
            return out[0] if unwrap_output else out

        # Step 9: Return runner, None (no cache handle)
        return runner, None

★★★★★★★★★ Design decisions worth noting:

1. Deep-copy: Why needed?
   → AOTAutograd's min-cut partitioner mutates the FX graph in place
   → vLLM's piecewise compilation shares the original graph across subgraphs
   → If aot_eager mutates without copying → other subgraphs corrupted!
   → PyTorch issue #138980 tracks this → known PyTorch bug/behavior

2. graph_returns_tuple handling: Why needed?
   → aot_module_simplified (used by aot_eager) requires tuple output
   → Some piecewise subgraphs produce single tensor → not tuple
   → Wrap single tensor in tuple → unwrap after execution
   → This is the SAME handling needed for Inductor → but Inductor handles it internally

3. Subgraph index via compiler_config: Why not cache key?
   → vllm_backend passes subgraph_index through compiler_config dict
   → Previously: Inductor used additional_inductor_config dict
   → Now: AOTEagerAdaptor reads from compiler_config → cleaner separation
   → This required modifying VllmBackend to pass vllm_subgraph_index in config

4. No cache handle: Why None?
   → AOTAutograd doesn't support vLLM's out-of-band caching mechanism
   → VLLM_USE_AOT_COMPILE disabled for aot_eager (like eager)
   → No persistent compiled artifact → recompile on every engine init
   → For GRPO: warmup once → acceptable → for serving: minor overhead
```

### 9.2 Supporting Changes (4 other files)

```
★★★★★★★★★ backends.py (+7/-1):
  → Import AOTEagerAdaptor alongside EagerAdaptor, InductorAdaptor
  → make_compiler() dispatch: elif backend == "aot_eager" → return AOTEagerAdaptor()
  → VllmBackend.compile: pass compiler_config dict (with vllm_subgraph_index)
    → Before: additional_inductor_config passed directly
    → After: compiler_config = dict(additional_inductor_config) + {"vllm_subgraph_index": graph_index}
    → This change affects ALL backends → not just aot_eager!
    → ★★★★★★★★ Important: this modifies how subgraph index is communicated to ALL compilers

★★★★★★★★★ counter.py (+2/-0):
  → num_aot_eager_compiles: int = 0 → new counter field
  → Mirrors num_inductor_compiles and num_eager_compiles

★★★★★★★★★ decorators.py (+6/-3):
  → VLLM_USE_AOT_COMPILE assertion: backend in ("eager", "aot_eager") → disabled
  → Before: only "eager" → now includes "aot_eager"
  → Warning message: "Detected %s backend, disabling AOT compile." → dynamic message
  → Before: "Detected eager backend, disabling AOT compile." → static

★★★★★★★★★ compilation.py (CompilationConfig, +6/-4):
  → Docstring: "available backends include eager, aot_eager, inductor, and custom backends"
  → Piecewise validation: backend not in ["", "eager", "aot_eager", "inductor"] → ValueError
  → init_backend: backend not in ["eager", "aot_eager", "inductor"] → custom backend message
  → All three locations updated consistently
```

---

## 10. vLLM Compilation Backend Taxonomy (Updated)

```
★★★★★★★★★ Complete vLLM compilation backend taxonomy with #46085:

CompilationMode:
  0: NONE → no compilation → pure eager execution
  1: DYNAMO_ONCE → compile once → no recompilation → backend sees whole graph
  2: COMPILATION_MODE_OVERRIDE → custom → backend sees whole graph
  3: VLLM_COMPILE → piecewise → backend sees per-subgraph → MOST flexible

Backend Options (mode 3 piecewise):
  eager:     Dynamo only → no AOTAutograd → no Inductor → pure eager subgraphs
  aot_eager: Dynamo + AOTAutograd → functionalization + decomposition → no Inductor
  inductor:  Dynamo + AOTAutograd + Inductor → full compilation → Triton codegen
  custom:    Via get_compile_backend → external backend → OOT (out-of-tree)

CUDA Graph Modes:
  NONE:             No CUDA graphs → all ops eager
  PIECEWISE:        Per-subgraph graph capture → flexible → compatible with aot_eager
  FULL:             Whole-model graph capture → rigid → only compatible with Inductor
  FULL_DECODE_ONLY: Full graph only for decode → hybrid approach
  FULL_AND_PIECEWISE: Decode=FULL + prefill=PIECEWISE → default mode

★★★★★★★★★ RTX 4090 recommended configurations:

Standard models (Llama, Qwen2.5, etc.):
  → -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=PIECEWISE VLLM_BATCH_INVARIANT=1
  → Correct (no Inductor fusion) + piecewise graph acceleration + batch-invariant overrides

Dynamic routing models (DSV4, MoE-heavy):
  → -cc.mode=3 -cc.backend=aot_eager -cudagraph_mode=NONE VLLM_BATCH_INVARIANT=1
  → Correct (no Inductor fusion + no stale routing) + functionalization benefits

When P9 lands (future optimal):
  → -cc.mode=3 -cc.backend=inductor + P9 Fusion Guard -cudagraph_mode=PIECEWISE
  → Correct (P9 blocks bad fusions) + full Inductor speedups + piecewise graphs

Current production (before P9 and aot_eager merge):
  → enforce_eager=True VLLM_BATCH_INVARIANT=1
  → Correct (no compile + no graphs) + slowest + simplest
```

---

## 11. Historical Context: PyTorch aot_eager

```
★★★★★★★★★ PyTorch's aot_eager backend history:

torch._dynamo.backends.debugging.aot_eager:
  → Part of PyTorch's debugging backend collection
  → Available since PyTorch 2.0 (2023)
  → Purpose: isolate Dynamo/AOTAutograd issues from Inductor issues
  → Same mechanism: traces through AOTAutograd, no-op compiler for fwd/bwd

Known PyTorch issues with aot_eager:
  → #138264: aot_eager backend incorrectly raising error for _native_batch_norm_legit
  → #138980: AOTAutograd mutates graph in place → deep-copy needed
  → These are the SAME issues vLLM's AOTEagerAdaptor addresses!

★★★★★★★★★ Other PyTorch debugging backends (not in vLLM yet):

torch._dynamo.backends.debugging collection:
  → aot_eager: AOTAutograd + no-op compiler (NOW in vLLM via #46085)
  → eager: no compilation (already in vLLM)
  → aot_nop: similar to aot_eager but different nop implementation
  → nop: no-op backend for Dynamo debugging only
  → interpret: Python interpreter backend

★★★★★★★★★ Could vLLM add more debugging backends in future?
  → aot_nop: slightly different from aot_eager → may have different edge cases
  → These backends help isolate SPECIFIC compile stage issues:
    → Dynamo-only issue: use eager backend
    → AOTAutograd-only issue: use aot_eager backend
    → Inductor-only issue: compare aot_eager vs inductor → isolate Inductor bug
  → ★★★★★★★★ aot_eager is the most useful debugging backend for SM89 batch invariance!
```
