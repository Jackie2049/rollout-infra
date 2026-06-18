# Megatron #5384 — DSA/DSv4 Indexer Replay for RL Deep Reading

> 2026-06-18 | Feature request deep reading — RL training stability for DSA models
> ★★★★★★★★ DSA indexer top-k = discrete decision → train/rollout mismatch → same pattern as MoE R3!
> ★★★★★★★★ Directly relevant to verl GRPO/CPPO RL training → RTX 4090 high relevance

---

## 1. What is DSA (Dynamic Sparse Attention)?

```
★★★★★★★★★ DSA = DeepSeek V3.2's sparse attention mechanism (Megatron PR #2440, MERGED):

Architecture:
  → Lightweight DSAIndexer module predicts which KV positions are relevant per query
  → DSAIndexer has its own Q/K projections + RoPE + Hadamard rotation
  → Produces index_scores [B, Sq, Sk] → top-k selection → sparse attention
  → KL divergence auxiliary loss trains indexer (same pattern as MoE aux loss)
  → Indexer inputs DETACHED from main model → no gradient flow back
  → Loss attached via DSAIndexerLossAutoScaler (custom autograd, same as MoEAuxLossAutoScaler)

CLI flag: --experimental-attention-variant dsa
Requires: multi_latent_attention=True + context_parallel_size=1

★★★★★★★★★ Key operation:
  → index_scores = einsum('sbhd,tbd->sbht', q.float(), k.float())
  → ReLU + weight + top-k → DISCRETE, NON-SMOOTH operation!
  → Small numerical differences → different top-k selections → train/rollout mismatch
```

---

## 2. Indexer Replay vs MoE Router Replay

```
★★★★★★★★★ Analogy is direct but objects differ:

| Aspect | MoE Router Replay (R3) | DSA Indexer Replay (proposed) |
|--------|----------------------|-------------------------------|
| Discrete decision | Which expert per token (top-k expert indices) | Which KV positions per query (top-k position indices) |
| Decision maker | TopKRouter (gating logits → softmax → top-k) | DSAIndexer (index scores → ReLU + weights → top-k) |
| Record object | topk_ids [num_tokens, topk] per layer | indexer_topk [batch, seqlen_q, index_topk] per layer |
| Replay mechanism | Bypass top-k → use recorded indices | Bypass indexer top-k → use recorded indices |
| Gradient flow | Training logits still computed normally | Training scores still computed normally |
| Template | RouterReplay class (router_replay.py) | Would follow same pattern (~200-300 LOC) |

★★★★★★★★★ RouterReplay class (PR #2693, MERGED) provides the template:
  → RouterReplayAction enum: RECORD, REPLAY_FORWARD, REPLAY_BACKWARD
  → Per-layer RouterReplay instances registered globally
  → set_replay_data(all_layers_topk_indices) → distribute indices
  → get_recorded_data() → collect from all instances
  → Static buffer support for CUDA graph compatibility
  → Integration in TopKRouter.forward() via get_replay_topk()
```

---

## 3. Why This is Needed for RL Training Stability

```
★★★★★★★★★ Train/rollout mismatch problem (same root cause as MoE R3):

Rollout engine (SGLang/vLLM):
  → CUDA graph captured, fp16/bf16 inference kernels
  → Different batching, kernel implementations
  → DSA indexer computes: float matmul + ReLU + top-k

Training engine (Megatron):
  → Selective recomputation, different CUDA paths
  → Mixed precision with gradient accumulation
  → DSA indexer computes: possibly different float paths

★★★★★★★★★ Small numerical differences → different top-k selections:
  → Same tokens attend to different KV positions
  → Different probability distributions → logprobs mismatch
  → ★★★★★★★★ EXACTLY analogous to MoE R3 where ~10% routers disagree per forward
  → 3/3 baseline GRPO runs collapsed on Qwen3-30B-A3B due to router disagreement!

★★★★★★★★★ Community confirmation (rehan243):
  → "logprobs flipping because of minor CUDA kernel drift was real"
  → "especially nasty when using mixed precision and graph-captured inference"
  → Implemented workaround: stash meta['indexer_topk'] in rollout dict → replay during training
```

---

## 4. RTX 4090 GRPO Relevance

```
★★★★★★★★★ HIGH relevance for RTX 4090 GRPO:
  → DeepSeek-V4-Flash uses DSA → increasingly common in MoE landscape
  → Train/rollout mismatch = same problem as MoE R3 but for sparse attention
  → verl is primary RL framework → issue explicitly mentions verl integration
  → Indexer replay costs negligible memory (just integer indices) → no RTX 4090 overhead
  → ★★★★★★★★ Feature not yet implemented → feature request → needs implementation!

★★★★★★★★★ Integration path:
  → Megatron: implement DSAIndexerReplay (~200-300 LOC, follows RouterReplay pattern)
  → verl: carry recorded indexer_topk with rollout data
  → RTX 4090: when training DSA models → MUST use indexer replay for GRPO stability
```

---

## Key Findings Summary

★★★★★★★★★ DSA indexer top-k = discrete decision → train/rollout mismatch → same pattern as MoE R3
★★★★★★★★★ DSAIndexer produces index_scores → ReLU+weight+top-k → non-smooth → numerical sensitivity
★★★★★★★★★ RouterReplay class = template → ~200-300 LOC implementation needed
★★★★★★★★★ Community confirmed: logprobs flipping from CUDA kernel drift in mixed precision
★★★★★★★★★ RTX 4090 HIGH relevance: DSA models need indexer replay for GRPO stability
★★★★★★★★★ Feature not yet implemented → needs Megatron + verl integration

---

## References

- Megatron #5384: https://github.com/NVIDIA/Megatron-LM/issues/5384 (feature request)
- DSA implementation: megatron/core/transformer/experimental_attention_variant/dsa.py
- RouterReplay: megatron/core/transformer/moe/router_replay.py
- MoE R3 instability: notebook/projects/megatron-router-replay-source-reading.md
- verl GRPO: notebook/projects/verl-grpo-source-reading.md
