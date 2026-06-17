# SGLang Latest Developments June 2026 — Agent Research Findings

> 2026-06-18 | Agent research results | Key findings from SGLang latest PRs/issues/releases
> ★★★★★★★★ SGLang v0.5.13 released June 13 — Spec V2 default, Qwen3.5 Blackwell GDN
> ★★★★★★★★ #24459 MERGED May 6 — adds aten::rms_norm + mm.dtype → KERNEL-level strengthened!
> ★★★★★★★★ #27097 NEW bug — concurrent multi-LoRA diverge under deterministic inference

---

## 1. Version Releases

```
★★★★★★★★★ v0.5.13 (June 13, 2026) — latest release:
  → Spec V2 now DEFAULT speculative decoding path (V1 deprecated)
  → Qwen 3.5 Blackwell: FlashInfer GDN kernels + CuTeDSL GDN prefill
  → Piecewise (PCG) + Breakable (BCG) CUDA Graph extended to DSA/Kimi-K2.5/DSV4
  → DeepSeek V4: Context Parallel + MTP, sparse FlashMLA, FP4 indexer, SM120
  → experimental_sgl_trtllm MoE backend for FP8/NVFP4 LoRA (#27329 MERGED)
  → HiCache for hybrid models (SWA/Mamba) via UnifiedTree
  → Dependencies: transformers 5.8.1, flashinfer 0.6.12, sgl-kernel 0.4.3

★★★★★★★★★ v0.5.12 (May 16, 2026):
  → DeepSeek V4 Day-0, TokenSpeed MLA backend (Blackwell FP8 KV)
  → NVFP4 diffusion support, W4A4 MegaMoE kernels, HiSparse for DSv4
```

---

## 2. Deterministic Inference Developments

```
★★★★★★★★★ MERGED deterministic inference PRs (May-June 2026):

#27869 (MERGED June 14): Qwen3.5 deterministic batch-invariant logprobs
  → FLA gated layernorm Triton ROWS_PER_BLOCK → MAX_ROWS_PER_BLOCK in batch-invariant
  → MoE top-k combine sum-reduce → disables small-token torch.compile fast path
  → Qwen3.5-35B-A3B: batch 1/16/64 all bit-identical
  → Overhead: 33% at bs=1, 16% at bs=64

★★★★★★★★★ #24459 (MERGED May 6): Register aten::rms_norm + aten::mm.dtype in batch-invariant mode
  → rms_norm_batch_invariant → Triton constexpr kernel → now aten override!
  → mm.dtype specialization → separate constexpr paths for bf16/fp16
  → ★★★★★★★★ This STRENGTHENS SGLang's KERNEL-level position!
  → SGLang now has MORE overrides than originally counted (7+ → rms_norm + mm.dtype added)

★★★★★★★★★ #26815 (MERGED May 31): Deterministic token oracle + production write-input assertion
  → token_oracle sampling backend → per-request tokens known ahead of time
  → kv-canary validation → model consumed exactly expected input_ids/positions

★★★★★★★★★ Other MERGED:
  → #24629: Disable FlashInfer allreduce fusion under deterministic
  → #24395: Fix deterministic on models with SWAKVPool
  → #26412: Forward fixed_split_size in SWA/cross-attention paths
  → #21197: NPU adaptation for deterministic inference
  → #26814: rids/bootstrap-room int-hash plumbing for deterministic per-request

★★★★★★★★★ OPEN deterministic PRs:
  → #28063: FA3 chunked-prefill truncation alignment (2-line fix)
  → #26627: Kimi-K2.5 MoE deterministic (atomic_add + dual-stream bypass, -9.3% throughput)
  → #15041: TP-Invariant Inference (from Dec 2025)

★★★★★★★★★ NEW bug: #27097 (June 3) — concurrent multi-LoRA diverge under deterministic
  → Same (adapter, prompt) returns different text when concurrent
  → Sequential is stable → concurrency is the problem
  → Needs watching → may affect GRPO with LoRA
```

---

## 3. MoE / LoRA Developments

```
★★★★★★★★★ MoE LoRA now PRODUCTION-GRADE:

Merged:
  → #27329 (June 5): experimental_sgl_trtllm MoE backend for FP8/NVFP4 LoRA
  → #22122 (April 13): Virtual experts for LoRA MoE (--lora-use-virtual-experts)
  → #24160 (May 29): Share MoE LoRA Info
  → #28371/#28499 (June 17): Fix chunked SGMV CUDA graph segment replay
  → #24262: Optimize virtual experts (multi-block CUDA JIT histogram)
  → #24246: Remove GPU-CPU sync barrier in MoE LoRA path

★★★★★★★★★ Open MoE LoRA:
  → #28466: bf16 MoE-LoRA two-stream overlap (68-73% of no-LoRA ceiling on GB300)
  → #28395: Single-adapter LoRA with NEXTN/EAGLE speculative decoding
  → #25202: NVFP4 LoRA on FlashInfer-CUTLASS MoE backend

★★★★★★★★★ MoE LoRA + deterministic = UNIQUE SGLang capability:
  → experimental_sgl_trtllm backend + batch-invariant overrides
  → No other framework has this combination
```

---

## 4. RL / GRPO Developments

```
★★★★★★★★★ RL infrastructure rapidly maturing:

★★★★ #24781 (OPEN): RolloutKV prefix KV lifecycle for RL
  → Updated May 19: rollout_kv_pin_refcount, auto_unprotect, bulk unprotect
  → 22-42x logprob scoring speedup, 1.27x full RL step speedup
  → No attention kernel changes → safe for RTX 4090

★★★★ Other RL merged:
  → #23961: Init true on-policy with qwen_dense → contract-based across SGLang/Megatron/Miles
  → #26555: Avoid retokenization drift for pre-tokenized VLM RL requests
  → #26423: Fix crash when batch has mixed return_routed_experts True/False
  → #27268: load_lora_adapter_from_distributed API → RL weight update path
  → #27749: Update weights from distributed for speculative draft workers

★★★★ Other RL open:
  → #28015: Fix BF16 RL weight update with non-routed flashinfer_trtllm MoE
  → #26955: Avoid fp32 intermediate in RL on-policy logprob (inplace div + log_softmax reuse)
  → #27886: Weight version isolation in KV cache → RL hot-swap
  → #24408/24404: Qwen3 MoE true-on-policy parity/SP/PP

★★★★★★★★★ #9499 (OPEN): Return expert routing info for MoE routing replay
  → Still no unit tests or docs → needs more work
  → Changed dtype from int32 to uint8 for memory optimization
  → Enables GRPO RouterReplay data pipeline
```

---

## 5. Key Assessment for RTX 4090

```
★★★★★★★★★ RTX 4090 assessment:

1. Deterministic inference: SGLang remains gold standard
  → KERNEL-level now strengthened: rms_norm + mm.dtype aten overrides added (#24459)
  → Two open fixes: #28063 (FA3) and #26627 (Kimi-K2.5 MoE) → both minimal patches
  → NEW concern: #27097 multi-LoRA concurrency divergence → needs watching

2. MoE LoRA + deterministic = UNIQUE SGLang capability
  → No other framework has MoE LoRA with batch-invariant inference
  → experimental_sgl_trtllm backend → FP8/NVFP4/bf16 (pending #28466)
  → Direct RTX 4090 GRPO benefit → LoRA+MoE+deterministic all in one framework

3. RolloutKV (#24781): transformative for GRPO RL throughput
  → 22-42x logprob scoring → critical for RL step speed
  → Still open → needs CI and final review

4. Routing replay (#9499): enables GRPO RouterReplay
  → Still needs unit tests → data pipeline partially ready (#26423 merged)

5. NVFP4 / RTX 5090 (#28354): draft status with reload bug
  → Next-phase contribution window → CuTeDSL NVFP4 MoE
```

---

## References

- SGLang repo: https://github.com/sgl-project/sglang
- v0.5.13 release: https://github.com/sgl-project/sglang/releases/tag/v0.5.13
- #27869: https://github.com/sgl-project/sglang/pull/27869
- #24459: https://github.com/sgl-project/sglang/pull/24459
- #28063: https://github.com/sgl-project/sglang/pull/28063
- #26627: https://github.com/sgl-project/sglang/pull/26627
- #28466: https://github.com/sgl-project/sglang/pull/28466
- #24781: https://github.com/sgl-project/sglang/pull/24781
- #27097: https://github.com/sgl-project/sglang/issues/27097
- #9499: https://github.com/sgl-project/sglang/pull/9499
