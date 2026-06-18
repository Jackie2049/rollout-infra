# vLLM-Ascend Critical Developments — June 2026 Deep Scan

> 2026-06-18 | Deep scan of vLLM-Ascend/MindIE critical bugs and features
> ★★★★★★★★ #10684 DSA Hadamard becomes ALL-ZERO after sleep/wake → CRITICAL for verl RLHF on Ascend!
> ★★★★★★★★ #10579 MoE NaN: 1-line fix (remove torch.abs() before npu_moe_token_unpermute) → 0 human reviews!
> ★★★★★★★★ #10592 NPUIPC weight transfer → Ascend equivalent of verl weight sync → verl Ascend integration!

---

## 1. #10684 — DSA Hadamard Sleep/Wake Corruption (CRITICAL)

```
★★★★★★★★★ Bug description:
  → DSA (Data-Shared Architecture) uses Hadamard matrix for orthogonal projection
  → After sleep/wake_up cycle: Hadamard matrix becomes ALL-ZERO
  → → ALL projection outputs = 0 → model outputs garbage → NaN propagation!

★★★★★★★★★ Root cause (REFINED — in-place mutation!):
  → PRIMARY: Hadamard rotation in-place mutates the DSA shared attention buffer
  → → Before sleep: Hadamard transform applied IN-PLACE → original buffer destroyed
  → → On wake: system re-reads from the corrupted (already-mutated) shared buffer → garbage!
  → SECONDARY: sleep/wake_up cycle may also fail to transfer buffers (Hadamard not learnable)
  → → Hadamard matrix is a CONSTANT (not a learnable parameter) → stored as buffer
  → → Buffers NOT included in weight transfer protocol → also lost during sleep!
  → → → Double failure: in-place mutation + transfer exclusion = complete corruption!
  → → Same pattern as MoE RouterReplay: constant routing data lost during state transfer!

★★★★★★★★★ Impact on verl RLHF:
  → verl HYBRID mode uses sleep/wake cycle: Rollout→Sleep→Reward→Update→Wake
  → If verl runs on Ascend NPU → DSA Hadamard would be corrupted during Sleep phase!
  → → Reward computation after Sleep would use corrupted Hadamard → wrong rewards
  → → Actor update would be based on wrong rewards → training divergence!
  → ★★★★★★★★ This is a BLOCKER for verl RLHF on Ascend NPUs with DSA!

★★★★★★★★★ Connection to known patterns:
  → Same root cause class as MoE RouterReplay (#4168) → constant data lost during state transfer
  → Same root cause class as DSV4 CUDA graph instability → dynamic data cached across steps
  → Pattern: sleep/wake + constant buffer = corruption → universal across GPU/NPU architectures!
  → vLLM-Ascend #10193 DSV4 prefix cache → same sleep/wake boundary issue

★★★★★★★★★ Fix direction (REFINED — primary fix is copy, not transfer):
  → Option A: Don't mutate in-place → COPY buffer before Hadamard rotation → preserve original
  → → This is the PRIMARY fix → prevents corruption at source
  → → Cost: 1 extra buffer copy → minimal bandwidth → but ensures original preserved
  → Option B: Include buffers in weight transfer protocol → ensure Hadamard restored on wake_up
  → → This addresses SECONDARY cause → but doesn't fix in-place mutation
  → Option C: Regenerate Hadamard matrix on wake_up (it's deterministic, seed-based)
  → → Option C is MORE robust → no transfer bandwidth cost → same pattern as RouterReplay!
  → ★★★★★★★★ RECOMMENDED: Option A (copy before in-place mutation) + Option C (regenerate on wake)
  → → Option A prevents corruption at source → Option C provides safety net on wake
  → → Both together = COMPLETE protection against buffer corruption!
```

---

## 2. #10579 — MoE NaN Fix (1-line, 0 human reviews)

```
★★★★★★★★★ Bug: MoE token unpermute produces NaN on Ascend NPU
  → Root cause: torch.abs() applied to expanded_row_idx BEFORE npu_moe_token_unpermute
  → torch.abs() on row indices → some indices become DUPLICATED → unpermute corruption → NaN!

★★★★★★★★★ Fix: 1 line
  → Before: expanded_row_idx = torch.abs(expanded_row_idx)
  → After: REMOVE torch.abs() line → indices preserved → correct unpermute → no NaN

★★★★★★★★★ Why torch.abs() was wrong:
  → expanded_row_idx can have NEGATIVE values → these represent "padding" or "dropped" tokens
  → torch.abs() converts -1 → 1 → now TWO tokens map to row 1 → duplication!
  → npu_moe_token_unpermute sees duplicated indices → accumulation conflict → NaN
  → Correct approach: keep negative indices → unpermute ignores them → clean result

★★★★★★★★★ Impact: HIGH for MoE models on Ascend
  → Any MoE model using npu_moe_token_unpermute → potential NaN during inference
  → Qwen3-30B-A3B on Ascend → affected!
  → Same pattern as vLLM MoE combine (#45683) → cross-architecture MoE correctness issue

★★★★★★★★★ Review status: 0 human reviews → STALLED!
  → Same pattern as #8073 (DeepSpeed ZeRO-3+PEFT) → small fix, zero attention
  → Need community pressure to merge this critical 1-line fix
```

---

## 3. #10592 — NPUIPC Weight Transfer Engine (+787 lines)

```
★★★★★★★★★ What: NPUIPC = Ascend NPU equivalent of IPC-based weight transfer
  → +787 lines → significant infrastructure addition
  → Enables weight transfer between processes on Ascend NPU
  → Directly relevant to verl RLHF weight sync mechanism!

★★★★★★★★★ verl integration significance:
  → verl HYBRID mode: Rollout (vLLM/SGLang) → Sleep → Update (actor) → Wake → weight sync
  → Weight sync uses ZMQ IPC buckets (512MB) on CUDA GPUs
  → NPUIPC provides Ascend equivalent → enables verl on Ascend NPU!
  → ★★★★★★★★ NPUIPC + #10684 fix = prerequisite for verl RLHF on Ascend!

★★★★★★★★★ Technical comparison:
  → verl ZMQ IPC: bucket-based, 512MB max, CPU-mediated
  → NPUIPC: NPU-native, likely HCCS-based (Ascend interconnect), lower latency
  → Both enable cross-process weight sharing → same architectural pattern
  → NPUIPC may be MORE efficient → native NPU interconnect vs CPU-mediated ZMQ

★★★★★★★★★ Integration path:
  → Step 1: Fix #10684 (Hadamard sleep/wake) → ensure constant buffers survive transfer
  → Step 2: Merge #10592 NPUIPC → enable Ascend weight transfer
  → Step 3: verl Ascend backend → integrate NPUIPC as weight sync mechanism
  → Step 4: verl + vLLM-Ascend + NPUIPC = RLHF on Ascend NPU!
```

---

## 4. #10642 — DSA CP OTP (Orthogonal Token Projection)

```
★★★★★★★★★ What: DSA uses CP (Cross-Processor) OTP for distributed attention
  → OTP uses Hadamard matrix → orthogonal projection → reduces cross-processor communication
  → Related to #10684: same Hadamard matrix → same sleep/wake vulnerability!
  → If Hadamard corrupted → OTP produces garbage → distributed attention broken

★★★★★★★★★ Architecture significance:
  → DSA OTP = Ascend's approach to multi-NPU distributed inference
  → Similar to vLLM's TP (Tensor Parallelism) but with orthogonal projection
  → OTP reduces communication volume → enables larger models on fewer NPUs
  → RTX 4090 equivalent: not directly applicable (single GPU) → but pattern relevant
```

---

## 5. #10645 — DSV4 Chat Template Fix

```
★★★★★★★★★ What: Fix for #10628 → DSV4 (DeepSeek-V4) chat template incorrect on Ascend
  → DSV4 uses specific chat template format → Ascend port had wrong template
  → Fix: correct chat template → proper formatting for DSV4 conversations
  → ★★★★★★★★ Adds to DSV4 systematic instability evidence:
    → vLLM #45972 cudagraph revert
    → vLLM #45979 flashinfer sparse cache revert
    → SGLang #28591 MTP revert
    → SGLang #28569 EAGLE3 crash
    → vLLM-Ascend #10645 chat template bug → 6th DSV4 failure!
```

---

## 6. #10193 — DSV4 Prefix Cache (81% hit rate)

```
★★★★★★★★★ What: DSV4 prefix caching on Ascend NPU
  → block_size=32 → 81% hit rate in benchmark
  → Enables KV cache reuse for repeated prefixes → significant latency reduction
  → Same pattern as vLLM RadixAttention → but Ascend-native implementation

★★★★★★★★★ Relevance to verl RLHF:
  → verl HYBRID rollout generates n=8 responses per prompt → SAME prompt prefix!
  → Prefix cache = 81% reuse → 4.3x reduction in prefix computation per prompt
  → → Direct throughput improvement for verl rollout on Ascend!
  → ★★★★★★★★ Combined with NPUIPC (#10592) → Ascend RLHF throughput stack!

★★★★★★★★★ Connection to BudgetRefiner (P10):
  → BudgetRefiner manages token budget → prefix cache reduces prefix tokens
  → → BudgetRefiner + prefix cache = more tokens for decode → higher throughput
  → Same cross-architecture pattern → Ascend needs its own BudgetRefiner integration
```

---

## Key Findings Summary

★★★★★★★★★ #10684 DSA Hadamard sleep/wake → ALL-ZERO → BLOCKER for verl RLHF on Ascend!
★★★★★★★★★ #10579 MoE NaN: 1-line fix (remove torch.abs()), 0 reviews → STALLED!
★★★★★★★★★ #10592 NPUIPC weight transfer → Ascend equivalent of verl ZMQ IPC → verl Ascend integration pathway!
★★★★★★★★★ DSV4 6th failure (#10645 chat template) → DSV4 systematic instability extends to Ascend!
★★★★★★★★★ #10193 prefix cache 81% hit rate → verl rollout throughput improvement
★★★★★★★★★ Pattern: sleep/wake + constant buffer = corruption → universal across GPU/NPU!
★★★★★★★★★ verl Ascend integration path: fix #10684 → merge #10592 → verl Ascend backend

---

## References

- vLLM-Ascend #10684: DSA Hadamard sleep/wake corruption
- vLLM-Ascend #10579: MoE NaN 1-line fix
- vLLM-Ascend #10592: NPUIPC weight transfer engine
- vLLM-Ascend #10642: DSA CP OTP
- vLLM-Ascend #10645: DSV4 chat template fix
- vLLM-Ascend #10193: DSV4 prefix cache
- DSV4 synthesis: notebook/projects/dsv4-systematic-instability-pattern-synthesis.md
- verl FSDP2 source: notebook/projects/verl-fsdp2-deep-source-reading.md
- MindIE source: notebook/projects/mindie-vllm-ascend-deep-source-reading.md
