# vLLM tl.constexpr Batch-Invariance Bug (#48650) — P9 Thesis Validation

**Date**: 2026-07-15 (Session 10)
**Issue**: vLLM #48650
**Author**: sbeurnier (systematic audit with Claude Code)
**Significance**: ★★★★★★★★ DIRECTLY validates our P9 Inductor SM<90 Fusion Guard thesis

---

## What It Is

Several in-tree vLLM Triton kernels declare **runtime-varying, batch-derived values as `tl.constexpr`**. Constexpr values are baked into the Triton JIT cache key, so each new value forces a full kernel recompile (hundreds of ms) at request time.

This is the **same bug class** as our P9 thesis:
- P9: Inductor fuses operations on SM<90 → kernels become batch-dependent → recompiles per batch size
- #48650: Triton kernels bake batch-derived values into constexpr → kernels become batch-dependent → recompiles per batch variation

Both are **batch-invariance violations** that cause silent performance degradation or TTFT stalls.

---

## Confirmed Instances (7)

| Kernel | Constexpr | Producer | Effect | Fix Difficulty |
|--------|-----------|----------|--------|---------------|
| `triton_unified_attention` | `MAX_MM_RANGES` | per-batch max image count | **HIGHEST impact**: recompiles entire unified attention kernel per distinct image count | Medium (masked loop bound or pow2 bucket) |
| `triton_merge_attn_states` | `prefill_tokens_with_context` | per-batch prefill token count | Recompiles per prefill batch size (CUDA C++ twin correctly uses runtime arg) | Easy (just demote to runtime arg) |
| `_pack_seq_kernel` | `N`, `_unpack_seq_kernel` `B` | per-value (token/req counts) | Pure cache-key pollution; params UNUSED in kernel body! | Easy (remove unused params) |
| `triton_decode_attention` | `NUM_KV_SPLITS` | already pow2-bucketed but ratchets ~10× | Residual recompiles across context growth | Medium (clamp to fixed small set) |
| `_count_expert_num_tokens` | `BLOCK_SIZE = next_power_of_2(min(numel,1024))` | per-batch numel | Trivial: just use constant 1024 | Easy (1-line fix) |
| `lora_shrink/lora_expand` | `SPLIT_K`, `BLOCK_K` | flip at batch<128 | LoRA batch-dependent config (same family as #48590!) | Medium |
| `_fill_logprob_token_ids_kernel` | `NUM_TOPK` | per-batch max requested logprobs | Recompiles per distinct topk value | Easy (demote to runtime arg) |

---

## P9 Thesis Validation

Our P9 thesis (PyTorch #184119 / Jackie2049/pytorch PR #1):
- Inductor creates SM<90 prologue fusions that bake batch-dependent shapes into compiled kernels
- Fix: `choices.py` guard blocks prologue fusion on SM<90 (`props.major < 9`)
- Result: kernels become batch-invariant, no recompiles

#48650 validates this from a different angle:
- Triton kernels bake batch-dependent values into `tl.constexpr` → same mechanism (JIT cache keyed on batch-derived values)
- Fix: demote constexpr to runtime arg (or use fixed constants)
- Result: kernels become batch-invariant, no recompiles

**The pattern is universal**: any JIT compilation system (Triton, Inductor, XLA) that keys its cache on batch-derived values will create batch-dependent kernels → performance degradation on serving path → TTFT stalls.

---

## Related Bug Chain

| Bug | System | Mechanism | Status |
|-----|--------|-----------|--------|
| P9 (#184119) | PyTorch Inductor | SM<90 prologue fusion → batch-dependent | Jackie2049/pytorch PR #1 |
| #48650 | vLLM Triton | tl.constexpr batch-derived → recompiles | OPEN (issue) |
| #46085 | vLLM | aot_eager = batch-invariant BY DESIGN | CLOSED (documentation) |
| #48613 | vLLM | GDN batch-invariance broken | OPEN |
| triton#10872 | triton_kernels | same bug class in OpenAI gpt-oss | FIXED |
| #187636 | PyTorch | autotune_at_compile_time=False reduces batch-dependent risk | MERGED |

---

## RTX 4090 Impact

- RTX 4090 = SM89 (Ada Lovelace) → Inductor creates batch-dependent fusions → our P9 guard blocks them
- vLLM on RTX 4090: Triton fallback path used (no FlashInfer) → all #48650 instances apply
- **Critical**: LoRA SPLIT_K/BLOCK_K flip at batch<128 = same family as #48590 LoRA NaN on Hopper
- **Mitigation**: aot_eager=True + no CUDA graph = batch-invariant by design (confirmed in #46085)

---

## Cross-Framework Lesson

The batch-invariance bug pattern appears in:
1. **Triton JIT**: tl.constexpr bake batch values → cache keyed per batch → recompile
2. **PyTorch Inductor**: prologue fusion bake batch shapes → compiled kernel keyed per shape → recompile
3. **CUDA Graph**: graph captured at batch_size N → replay only works at batch_size N → need multiple captures
4. **JAX/XLA**: shape polymorphism → XLA compiles per shape → recompile

**Universal fix**: Make all JIT compilation cache keys independent of batch-derived values. Either:
- Use runtime arguments instead of constexpr/compile-time constants
- Use fixed constants (pow2 bucket, clamp to max)
- Use aot_eager/piecewise mode that avoids fusion entirely

This is now the **3rd independent validation** of our P9 thesis (after #184119 and #46085).
