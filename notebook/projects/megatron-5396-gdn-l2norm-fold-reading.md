# Megatron #5396 — GDN L2-norm Fold Deep Reading

> 2026-06-18 | Source-level deep reading of GDN L2-norm fold PR
> ★★★★★★★★ Folds explicit l2norm(q/k) into gated_delta_rule kernel → 24 GiB backward activation savings at 128K context!
> ★★★★★★★★ Pure improvement with no downside → should adopt when available

---

## 1. What is GDN L2-norm Fold?

```
★★★★★★★★★ GDN (GatedDeltaNet) = linear attention layer in Qwen3.5-35B-A3B hybrid architecture:
  → ~3/4 of 32 layers use GDN (linear attention)
  → ~1/4 use dense attention
  → GDN normalizes query/key with L2-norm before gated_delta_rule kernel

★★★★★★★★★ Previously: explicit separate operation before kernel:
  → _prepare_qkv_for_gated_delta_rule():
    → if self.use_qk_l2norm:
    →   query_key = l2norm(query_key.contiguous())  # MATERIALIZE normalized activation
  → kernel invocation: use_qk_l2norm_in_kernel=False  # hard-coded

★★★★★★★★★ The fold (total +7/-4 lines):
  1. forward() line 478: use_qk_l2norm_in_kernel=False → use_qk_l2norm_in_kernel=self.use_qk_l2norm
  2. _prepare_qkv_for_gated_delta_rule(): remove explicit l2norm block → kernel handles it internally
```

---

## 2. How the Kernel Handles L2-norm Internally

```
★★★★★★★★★ FLA gated_delta_rule kernel (chunk.py) with use_qk_l2norm_in_kernel=True:

Forward:
  → if use_qk_l2norm_in_kernel:
  →   q, q_rstd = l2norm_fwd(q)  # Triton kernel → normalized q + reciprocal std
  →   k, k_rstd = l2norm_fwd(k)
  → Save for backward: q, q_rstd, k, k_rstd (tiny!) + v, g, beta, A, etc.

Backward:
  → if ctx.use_qk_l2norm_in_kernel:
  →   dq = l2norm_bwd(q, q_rstd, dq)  # Recomputes from original + rstd
  →   dk = l2norm_bwd(k, k_rstd, dk)

★★★★★★★★★ l2norm_fwd Triton kernel:
  → rstd = 1/sqrt(sum(x*x) + eps)  → per-head scalar vector [B, T, H]
  → Returns: normalized output + rstd (both needed for backward)

★★★★★★★★★ l2norm_bwd Triton kernel:
  → Recomputes normalization on-the-fly from original (un-normalized) q/k + saved rstd
  → Much cheaper than reading the full saved normalized tensor
```

---

## 3. Memory Impact — MASSIVE at Long Context

```
★★★★★★★★★ Backward activation savings:

At 128K context (B=1, Qwen3.5-35B-A3B — the PR test config):
  → OLD: query_key [1, 131072, 2*16, 128] in bf16 per GDN layer = ~1 GiB!
  → NEW: rstd [1, 131072, 32] in fp32 per GDN layer = ~16 KiB (negligible!)
  → ★★★★★★★★ ~24 GDN layers × ~1 GiB per layer = ~24 GiB total savings!
  → This is EXACTLY the 128K activation that must be kept for backward → game-changer!

At 4K context (typical RTX 4090):
  → OLD: query_key [1, 4096, 32, 128] in bf16 = ~16 MiB per layer
  → NEW: rstd [1, 4096, 32] in fp32 = ~0.5 KiB per layer
  → ~24 layers × 16 MiB = ~384 MiB total savings → meaningful for 24 GiB GPU!

★★★★★★★★★ Compute impact: negligible
  → Forward: same work whether outside or inside kernel
  → Backward: recompute from original+rstd → cheap Triton kernel
```

---

## 4. Numerical Correctness

```
★★★★★★★★★ Numerically lossless (verified in PR):
  → eps=1e-6 on ALL three paths (FLA default, FLA in-kernel, torch explicit) → matches
  → GQA: l2norm(repeat_interleave(x)) == repeat_interleave(l2norm(x)) per-head
    → l2norm over dim=-1, repeat_interleave only along dim=2 (head dim)
    → In-kernel normalizes post-repeat (Hv heads), old normalizes pre-repeat (Hk heads)
    → ★★★★★★★★ IDENTICAL per head → no numerical difference!
  → use_qk_l2norm=False → behavior unchanged → zero blast radius
```

---

## 5. RTX 4090 Relevance

```
★★★★★★★★★ Moderate-to-high RTX 4090 relevance:
  → 384 MiB savings at 4K context → meaningful for 24 GiB
  → Pure improvement with no downside → should adopt when available
  → If training Qwen3.5-35B-A3B (GDN hybrid) on RTX 4090:
    → Every backward activation saving helps under selective recomputation
    → GDN layers might not be recomputed → need to keep activations → fold reduces memory!
  → ★★★★★★★★ Risk: original code hard-coded use_qk_l2norm_in_kernel=False
    → If intentional for cu_seqlens/packed-sequence or CP correctness → could break those paths
    → Author asks maintainers to advise → 0 maintainer comments yet
```

---

## Key Findings Summary

★★★★★★★★★ GDN L2-norm fold: move explicit l2norm into FLA kernel → 24 GiB savings at 128K!
★★★★★★★★★ At RTX 4090 4K context: ~384 MiB savings → meaningful
★★★★★★★★★ Numerically lossless: eps=1e-6 matches, GQA per-head identical
★★★★★★★★★ Pure improvement +7/-4 lines → zero blast radius → should adopt
★★★★★★★★★ Risk: cu_seqlens/CP correctness unknown → maintainer input needed

---

## References

- Megatron #5396: https://github.com/NVIDIA/Megatron-LM/pull/5396
- FLA gated_delta_rule kernel: fla/ops/gated_delta_rule/chunk.py
- FLA l2norm Triton: fla/modules/l2norm.py
- GDN layer: megatron/core/ssm/gated_delta_net.py
