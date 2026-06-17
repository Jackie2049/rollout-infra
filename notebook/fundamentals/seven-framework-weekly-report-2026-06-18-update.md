# 7-Framework Weekly Developments Update — June 16-18, 2026

> 2026-06-18 | Rapid developments across all 7 frameworks
> Key: SGLang LoRA csgmv CUDA graph fix (#28499 MERGED), vLLM DCP MLA query replication (#45964), DeepSpeed #8072/#8073 still OPEN, Megatron #5219 still OPEN

---

## SGLang (Most Active)

### MERGED June 17: #28499 — LoRA chunked SGMV (csgmv) CUDA graph segment replay fix
```
★★★★★★★★★ DIRECTLY relevant to #27097 multi-LoRA determinism bug!

Bug: In CUDA-graph mode, csgmv LoRA kernels baked grid size (batch_info.bs)
  and num_segments scalar into captured graph at capture time.
  On replay → real batch's segment count differs → wrong number of segments processed
  → LoRA delta dropped for some segments or stale tail metadata on live tokens.

Fix: Cherry-pick of #28371 onto sglang-miles branch.
  → ★★★★★★★★ This partially addresses Factor 2 of #27097 (CUDA graph metadata replay)
  → But NOT a complete fix → still has dynamic routing (Factor 1) + KV stale state (Factor 3)
```

### MERGED June 17: #28556 — sgl-kernel bumped to 0.4.4

### MERGED June 17: #28505 — MXFP8 dense scheme for Ascend NPU
```
Delegate process_weights_after_loading / apply to kernel (self.kernel)
  → hardware-agnostic weight creation → kernel-specific processing
```

### MERGED June 17: #28548 — Speculative weight-update fan-out centralized in scheduler
```
Refactor: centralize speculative weight-update fan-out in scheduler
  → Previously scattered → now single location → cleaner spec decode + LoRA interaction
```

### MERGED June 17: #28550 — spec-decode penalty H2D non-blocking + share decode cumulate path

### OPEN: #28566 — LoRA sentinel-pad token→lora mapping for DP-attention foreign tokens
```
★★★★★★★★★ CRITICAL for DP-attention + LoRA correctness:
  → Under DP-attention, gathered batch has local + foreign tokens
  → seg_indptr / weight_indices only cover local requests
  → _compute_moe_lora_info does wrong things at foreign positions
  → CUDA path: torch.empty() → uninited → random LoRA routing
  → Triton path: wrong lora_id at foreign positions → same bug as #27097 Factor 1!
  → ★★★★★★★★ Another source of multi-LoRA divergence under DP-attention
```

### OPEN: #28564 — LoRA virtual-experts kernel for gate_up fusion + MoE LoRA path
```
5 refinements to MoE LoRA stack from #25141:
  → merged_experts_fused_moe_lora_add accepts lora_b as single tensor or length-2 sequence
  → gate_up case: A has rank 2*r, each B has rank r
```

### OPEN: #28552 — Triton block-FP8 GEMM tuned configs for Qwen3.5 on Blackwell SM120
```
★★★★★★★★★ RTX 5090 (SM120) relevant!
  → Block-FP8 W8A8 on SM120 defaults to Triton backend (CUTLASS unavailable)
  → Tuned configs for Qwen3.5 → fills SM120 Triton FP8 gap
```

### OPEN: #28567 — get_parallel(): structured accessor for parallel topology state
```
Infrastructure improvement → cleaner parallel state access
```

---

## vLLM

### OPEN: #45683 — Deterministic MoE combine (updated June 16)
```
Still OPEN, no new comments since June 15
```

### OPEN: #45819 — GDN attention batch invariance (updated June 17)
```
Updated June 17 → 4 lines of code change → Qwen3.6 hybrid Mamba+GDN deterministic
```

### OPEN: #45964 — MLA decode query replication for DCP (DeepSeek)
```
★★★★★★★★★ NEW: DCP MLA decode optimization
  → Under Decode Context Parallelism → KV cache sharded across DCP group
  → Standard path: all-gather query every step → on decode critical path
  → New opt-in: replicate MLA query projection within DCP group at load time
  → Every rank already materializes full query → skip all-gather → faster decode
```

### DRAFT: #45966 — Pre-reserve packed A2A decode workspace (DCP long-term fix for #45487)
```
★★★★★★★★★ NEW: DCP workspace pre-reservation
  → A2A send/recv staging buffers from shared growable WorkspaceManager
  → Workspace sized to largest captured batch then locked
  → Pre-reserve packed A2A workspace → eliminates dynamic allocation → more stable
```

### MERGED June 17: #45917 — FlashInfer all-reduce fusion: pass TP group
```
Bugfix: TP group not passed to FlashInfer all-reduce fusion → hang on startup
```

### MERGED June 17: #45896 — MiniMax-M3 MXFP4 support

### MERGED June 17: #45863 — DSv4 flashinfer sparse index cache (2-4% TTFT improvement)

### MERGED June 17: #45868 — MRV2 model/config compatibility fixes

---

## DeepSpeed

### OPEN: #8058 — ZenFlow native CPU optimizer (updated June 15)
```
Still OPEN, 5 review comments
  → 954 additions, chunked copyback (32M elements/chunk = 128 MiB staging)
  → GPU spike 2944→256 MiB → POSIX semaphore → NUMA-local → covers Z1/Z2/Z3
```

### OPEN: #8072 — ZeRO-3 + PEFT LoRA regression (updated June 17)
```
★★★★★★★★★ Still OPEN, updated June 17!
  → TypeError in _allgather_params_coalesced → mixed bf16+fp32 → #8066 dtype cast
  → ZeRO-2 UNAFFECTED → only ZeRO-3+PEFT
```

### OPEN: #8073 — Fix: per-param dtype for output buffers (updated June 17)
```
★★★★★★★★★ Fix PR for #8072, updated June 17!
  → Use per-param dtype for output buffers in _allgather_params_coalesced
  → ★★★★★★★★ 2-line fix → same fix we documented in source reading
```

### MERGED: #8066 — Mixed-precision dtype preservation (merged in v0.19.2)

### MERGED: #8056 — Consistent fp32 grads flow (merged in v0.19.2)

### MERGED: #8047 — Muon muon_lr/adam_lr override test coverage

---

## Megatron-LM

### OPEN: #5219 — LayerWise Muon crash fix (updated June 17)
```
★★★★★★★★★ Still OPEN, Final Review label
  → +14/-7, None guard for dp_cp_params_list
  → Irony: expt_dp_params_list already had correct guard
```

### OPEN: #5395 — skip_grad_norm_clip for Muon (+15/-1)
```
Fix for #5394 ChainedOptimizer clipping stalls
```

### MERGED: #5389 — Restore fused GDN THD all-to-all on dev

### MERGED: #5372 — MimoModel.zero_grad_buffer delegates to active DDP submodules
```
★★★★★★★★★ NEW: MIMO wraps each submodule in own DDP
  → zero_grad_buffer delegates to active DDP-wrapped submodules
  → Skips raw/inactive submodules
```

---

## PyTorch

### OPEN: #187275 — Combo kernel dynamic reduction fix (updated June 17)
```
★★★★★★★★★ 5 comments, still OPEN
  → Batch invariance root cause fix → SM89 relevant
```

### OPEN: #187435 — no_fuse_region per-op (updated June 17)
```
★★★★★★★★★ 1 comment, still OPEN
  → Per-op fusion barrier → complementary to P9
```

### OPEN: #181248 — B200 Triton async fence nondeterminism (NOT SM89)

### MERGED June 16: #187347 — Revert cudagraph skip for kernel-free inductor graphs (release/2.13 cherry-pick)
```
★★★★★★★★★ Important revert in PyTorch 2.13!
  → Reverts #185760 which tried to skip CUDA graphs for inductor graphs with no kernel launches
  → The skip was causing issues → reverted on main AND release/2.13
  → ★★★★★★★★ Relevance: cudagraph handling affects batch invariance → SM89
```

---

## verl

### Tracked PRs:
```
#6699: MERGED June 12 → detach model_output → 4x memory reduction
#6717: MERGED June 15 → Tinker training worker primitives
#6688: MERGED June 12 → LoRA IPC buffer crash fix
#6738: MERGED June 16 → skip redundant clone SGLang weight sync
#6689: MERGED June 17 → offload process_vision_info to thread
#6790: MERGED June 17 → separate async trainer
#6572: OPEN → full determinism vLLM rollout
#6779: OPEN → Continuous Token Agentic Rollout (5024 additions)
#6731: OPEN → CPPO (Binary-TV variant, bypass_mode MANDATORY)
```

---

## rLLM

### OPEN: #605 — GRPO grouping bug (no update since June 2!)
```
★★★★★★★★★ STILL OPEN, NO activity for 16 days!
  → trajectory.uid vs task_ids → group size 1 → advantage = raw reward → GRPO BROKEN
  → ★★★★★★★★ CRITICAL blocker → needs community pressure
```

### MERGED: #663 — Fix: populate Step.output in trace_record_to_step

### MERGED: #658 — Cumulative token mode Tinker path (June 16)

### MERGED: #650 — Terminal-RL: Terminus-2 harness, AgentFlow redesign, warm-pool training
```
★★★★★★★★★ 71 files, +3673/-1413 → major feature merge!
  → Terminus-2 agent harness
  → pass@k via --attempts
  → warm-pool training → faster startup
```

---

## Key New Developments Summary

★★★★★★★★★ SGLang #28499 MERGED June 17: csgmv CUDA graph segment replay → partially fixes #27097 Factor 2
★★★★★★★★★ SGLang #28566 OPEN: LoRA sentinel-pad for DP-attention → another source of multi-LoRA divergence
★★★★★★★★★ vLLM #45964 OPEN: MLA decode query replication → DCP optimization for DeepSeek
★★★★★★★★★ DeepSpeed #8072/#8073: Still OPEN, updated June 17 → ZeRO-3+PEFT regression NOT fixed yet
★★★★★★★★★ PyTorch #187347: cudagraph skip reverted in 2.13 → batch invariance relevant
★★★★★★★★★ rLLM #605: STILL OPEN, 16 days without update → needs community engagement
★★★★★★★★★ rLLM #650: Terminal-RL major merge (3673 additions)
★★★★★★★★★ SGLang #28552: SM120 Triton block-FP8 → RTX 5090 (SM120) development continues
