# Cross-Framework Weight Synchronization Comparison — Sleep/Wake Architecture

> 2026-06-18 | How 7 frameworks handle weight sync between inference and training workers
> ★★★★★★★★ Universal pattern: sleep/wake + constant buffer = corruption risk!
> ★★★★★★★★ vLLM-Ascend #10684: DSA Hadamard ALL-ZERO after sleep/wake → CRITICAL!
> ★★★★★★★★ verl ZMQ IPC 512MB bucket → weight sync between rollout and training
> ★★★★★★★★ DeepSpeed ZeRO-2: CPU_Adam offload → implicit "sleep" (CPU state) + "wake" (GPU load)

---

## 1. Weight Synchronization Architectures

```
★★★★★★★★★ 7 frameworks, 5 weight sync architectures:

| Framework | Sync Mode | Sleep→Wake | Buffer Risk | Known Bug |
|-----------|-----------|------------|-------------|-----------|
| verl HYBRID | ZMQ IPC bucket | Rollout→Sleep→Reward→Update→Wake | 512MB bucket, LoRA prefix hardcode | #6699 detach leak |
| verl FSDP2 | PyTorch weight sync | Same as HYBRID | 8 hard-coded LoRA prefixes | #6468 CPU leak |
| vLLM-Ascend | NPUIPC (#10592) | Sleep→weight transfer→Wake | HCCS interconnect | #10684 Hadamard ALL-ZERO! |
| DeepSpeed ZeRO-2 | CPU_Adam offload | CPU offload→GPU load (implicit) | CPU optimizer state | #8061 overlap_comm NaN |
| rLLM Tinker | Single-step | No sleep/wake needed | None (single process) | #605 GRPO grouping |
| Megatron-LM | Megatron weight sync | Multi-process weight broadcast | TP group AllReduce | RouterReplay needed |
| SGLang standalone | No training sync | No sleep/wake | None (pure inference) | #28582 RCE (security) |
```

---

## 2. Sleep/Wake Architecture Deep Analysis

```
★★★★★★★★★ verl HYBRID mode (6-step sync):

Step 1: Rollout (vLLM/SGLang generates n=8 responses per prompt)
Step 2: Sleep → free GPU memory → only actor weights kept
  → vLLM/SGLang process releases GPU → KV cache freed → GPU empty
  → ★★★★★★★★ CRITICAL: "constant" buffers (Hadamard, router bias) must survive Sleep!
Step 3: Reward → external reward model or rule-based
Step 4: Advantage → GRPO group-relative normalization
Step 5: Update → actor parameters updated (FSDP2/ZeRO-2)
Step 6: Wake → restore GPU memory → load updated weights → next iteration
  → ★★★★★★★★ CRITICAL: "constant" buffers must be correctly loaded on Wake!

★★★★★★★★★ vLLM-Ascend NPUIPC mode (new):

Step 1: Rollout (Ascend NPU generates responses)
Step 2: Sleep → NPU memory freed → weights to CPU or another NPU
  → #10592 NPUIPC: HCCS interconnect → Ascend-native IPC → weight transfer
  → ★★★★★★★★ BUG #10684: DSA Hadamard becomes ALL-ZERO after Sleep/Wake!
  → → Hadamard matrix stored as buffer → NOT included in weight transfer protocol
  → → Sleep: Hadamard lost → Wake: Hadamard = zeros → ALL projection outputs = 0!
  → → Model produces garbage after Wake → training diverges!

★★★★★★★★★ DeepSpeed ZeRO-2 mode (implicit sleep/wake):

No explicit sleep/wake → but CPU_Adam creates implicit state transfer:
  → Forward/backward: GPU → parameters on GPU, optimizer on CPU
  → CPU_Adam step: CPU → optimizer states updated → gradients applied
  → After step: CPU optimizer state = "sleep" state → GPU params = "wake" state
  → ★★★★★★★★ #8061: overlap_comm + torch.compile → multi-stream GPU operations
  → → ZeRO-2 AllReduce + CPU_Adam overlap → data race → NaN!
  → → overlap_comm=False → sequential → safe on single GPU
```

---

## 3. The Universal Pattern: Constant Buffer Corruption

```
★★★★★★★★★ Pattern discovered from #10684 + RouterReplay (#4168) + DSV4 (5 failures):

RULE: Per-step dynamic data MUST NOT be cached across steps.
  → Applies to: CUDA graphs, NPU graphs, sleep/wake, weight sync, state transfer

EXTENDED RULE: Constant buffers that depend on initialization MUST be
  preserved during ALL state transitions (sleep, wake, offload, reload).

★★★★★★★★★ Known constant buffer corruption instances:

1. #10684 DSA Hadamard: ALL-ZERO after sleep/wake on Ascend
  → Hadamard = initialization-dependent constant → NOT learnable parameter
  → Sleep: weights transferred → Hadamard NOT transferred → lost
  → Wake: Hadamard not loaded → defaults to zero → garbage outputs
  → FIX: include buffers in weight transfer protocol → or regenerate on Wake

2. #4168 MoE RouterReplay: router bias stale after weight update
  → Router bias = learnable but initialized → changes during training
  → CUDA graph captures router state → after weight update → stale
  → FIX: replay router after weight update → RouterReplay pattern

3. DSV4 #45972: cudagraph revert → dynamic data cached across steps
  → Attention bias/position → varies per sequence length → cached in graph
  → After batch size change → stale → wrong results
  → FIX: break CUDA graph → enforce_eager=True

4. DSV4 #45979: flashinfer sparse cache → GSM8K 6.75% regression
  → Sparse attention cache → cached per layout → stale after layout change
  → FIX: clear cache between steps

5. DSV4 #28520: swa_loc cache → MTP accept-length collapse
  → Cached swa_loc from initial positions → all draft steps use same slot → overwrite
  → FIX: compute per-step → don't cache dynamic data

★★★★★★★★★ Universal fix pattern:
  → Option A: Include ALL buffers (learnable + initialization-dependent) in state transfer
  → Option B: Regenerate initialization-dependent buffers on Wake (cheaper, more robust)
  → Option C: Never cache dynamic data across steps (enforce_eager for serving)
  → Option B is RECOMMENDED → cheaper bandwidth → no transfer cost → deterministic regeneration
```

---

## 4. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 weight sync considerations:

verl HYBRID mode (recommended):
  → ZMQ IPC bucket: 512MB → works on single GPU → weight sync between rollout and training
  → Sleep/Wake: GPU memory freed → LoRA weights kept → wake restores
  → Hadamard risk: vLLM/SGLang on CUDA → same pattern as #10684!
  → → Must verify: does vLLM DSA Hadamard survive Sleep/Wake on CUDA?
  → → If not → same ALL-ZERO bug → need fix before RTX 4090 GRPO!

★★★★★★★★★ Critical verification needed (when GPU available):
  1. vLLM DSA: Hadamard matrix after Sleep/Wake → is it preserved?
  2. SGLang DSA: same check → does Hadamard survive state transfer?
  3. verl HYBRID: after actor update → does Hadamard stay valid?
  4. If Hadamard lost → add buffer preservation → P9-tier contribution!

★★★★★★★★★ DeepSpeed ZeRO-2 (safe by default):
  → No explicit Sleep/Wake → CPU_Adam implicit → no Hadamard risk
  → overlap_comm=False → sequential → no data race → safe
  → But: if torch.compile + overlap_comm → #8061 NaN → MUST overlap_comm=False!
```

---

## Key Findings Summary

★★★★★★★★★ #10684: Universal pattern → sleep/wake + constant buffer = corruption → across ALL platforms!
★★★★★★★★★ verl ZMQ IPC 512MB bucket → weight sync between rollout and training on CUDA
★★★★★★★★★ vLLM-Ascend NPUIPC → Ascend-native IPC → but Hadamard NOT in transfer protocol → bug!
★★★★★★★★★ DeepSpeed ZeRO-2 implicit sleep/wake → CPU_Adam → no explicit buffer corruption risk
★★★★★★★★★ rLLM Tinker: single-step → no sleep/wake → no buffer corruption → simplest!
★★★★★★★★★ RTX 4090: MUST verify DSA Hadamard preservation during verl HYBRID Sleep/Wake
★★★★★★★★★ Universal fix: regenerate initialization-dependent buffers on Wake (cheaper + deterministic)

---

## References

- vLLM-Ascend #10684: DSA Hadamard sleep/wake corruption
- vLLM-Ascend #10592: NPUIPC weight transfer engine
- vLLM-Ascend #10579: MoE NaN 1-line fix
- verl ZMQ IPC source: notebook/projects/verl-fsdp2-deep-source-reading.md
- verl HYBRID mode: notebook/projects/verl-rtx4090-grpo-training-flow.md
- DSV4 synthesis: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
- DeepSpeed ZeRO source: notebook/projects/deepspeed-zero-single-gpu-source-reading.md
- RouterReplay #4168: Megatron MoE router replay pattern
- BudgetRefiner: notebook/projects/vllm-v1-scheduler-budgetrefiner-source-reading.md
