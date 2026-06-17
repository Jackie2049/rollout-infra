# Megatron #5391 — Compact LayerWise DDP Source Reading

> 2026-06-18 | PR #5391 (DRAFT) | +218/-58 additions | Author: Wohox (Pingtian Li)
> ★★★★★★★★ Main counterpart of #5388 (dev branch) → same change on top of main
> ★★★★★★★★ Removes dp_size * max(shard_load) padding → compact decoupled per-buffer DDP layout
> ★★★★★★★★ RTX 4090: dp=1 → padding removed entirely → memory efficiency gain for Muon+ZeRO-2

 > ★★★★★★★★ Experimental → --no-use-layer-wise-param-layout disables padded layout → selects compact path

 > ★★★★★★★★ Key design: use_distributed_optimizer becomes a PER-BUFFER property
> ★★★★★★★★ Muon buffers: compact no-padding DDP + locally disable DistributedOptimizer semantics
> ★★★★★★★★ Sibling buffers (embeddings, biases, layernorm): standard byte-level DistributedOptimizer layout

---

## 1. The Problem: dp_size Padding Waste

```
★★★★★★★★★ Current LayerWise DDP padded layout:

With dp=1 (RTX 4090 single GPU):
  → Buffer allocation: dp_size * max(shard_load) per buffer
  → For dp=1: padding = max(shard_load) → WASTEFUL
  → Example: 0.75B-param model with dp=1
    → Shard size = 0.75B / 1 = 0.75B
    → Buffer allocation = 1 * max(shard_load) = max(shard_load) = wasteful
  → ★★★★★★★★ ALL buffer allocations include dp_size multiplier → even dp=1 wastes memory

★★★★★★★★★ Why this matters for RTX 4090:
  → RTX 4090 training: dp=1 → every buffer has dp_size * max(shard_load) padding
  → Muon optimizer: layer-wise buffers → 2D matrices → padding = dp_size * max(shard_load)
 per matrix
  → For dp=1: padding = max(shard_load) → the biggest shard in the group dominates
  → ★★★★★★★★ Small Muon matrices (e.g., attention Q/K/V) → their shards are small
  → But allocated space = max(shard_load) → larger than needed → wasted GPU memory
  → This padding exists for ALL buffers → cumulative waste significant

★★★★★★★★★ The padding problem explained:
  → dp_size multiplier: ensures uniform allocation across dp ranks for all-reduce
  → BUT dp=1 → no inter-rank communication → padding is pure waste
  → max(shard_load) across ALL buffers → not just the current buffer
  → ★★★★★★★★ Each buffer padded to max(shard_load) across ALL groups → wasteful
```

---

## 2. Compact Layout Design

```
★★★★★★★★★ Compact decoupled layout (this PR):

Flag: --no-use-layer-wise-param-layout (default True → padded)
  → Setting False → compact per-buffer layout → no dp_size multiplier
  → ★★★★★★★★ Per-buffer use_distributed_optimizer:

Muon-managed buffers (2D matrices):
  → Compact no-padding DDP layout
  → Locally disable DistributedOptimizer semantics:
    - all-reduce gradients (not reduce-scatter)
    - legacy whole-param ping-pong ownership
    - allgather_params param sync
  → ★★★★★★★★ Muon buffers DON'T need shard-level partitioning → they manage their own layout

Sibling buffers (embeddings, biases, layernorm):
  → Standard byte-level DistributedOptimizer layout
  → reduce-scatter gradients
  → DistributedOptimizer param partitioning
  → shard-level param sync

★★★★★★★★★ Key insight: use_distributed_optimizer becomes PER-BUFFER property:
  → Previously: global flag → all buffers same layout
  → Now: per-buffer → Muon buffers vs sibling buffers can differ
  → partition_buckets splits force-single bucket group by effective per-bucket use_distributed_optimizer
  → ★★★★★★★★ Muon (all-reduce) and sibling (reduce-scatter) buckets NEVER share a group

  → When all buffers agree → collapses to single group → identical to prior behavior

★★★★★★★★★ compute_full_param_layout:
  → Padded: dp_size * max(shard_load) per buffer group
  → Compact: per-buffer shard load → no padding → no dp_size multiplier

★★★★★★★★★ For RTX 4090 dp=1:
  → Padded: 1 * max(shard_load) → wasteful → each buffer gets max shard space
  → Compact: actual shard_load → no waste → each buffer gets exactly what it needs
  → ★★★★★★★★ Memory savings: significant for Muon layer-wise buffers on single GPU
```

---

## 3. Compatibility & Constraints

```
★★★★★★★★★ Compatibility:

Default padded layout: UNCHANGED
  → use_layer_wise_param_layout=True (default) → padded → no change
  → ★★★★★★★★ Backward compatible → existing configs work → no migration needed

Compact layout: opt-in experimental
  → use_layer_wise_param_layout=False → compact → per-buffer layout
  → ★★★★★★★★ Requires num_distributed_optimizer_instances == 1
  → Non-DistOpt Muon buffers only all-reduce within single optimizer instance
  → ★★★★★★★★ Cannot mix DistOpt instances on compact path

Blockwise/MXFP8 compute:
  → fp8_param_gather=False (params persist in bf16) → SUPPORTED
  → ★★★★★★★★ FP8 compute with compact layout → works

FP8/FP4 parameter gather:
  → REJECTED at arg-validation for any layer-wise distributed optimizer
  → Requires DistributedOptimizer param buffers → layer-wise path does NOT provide
  → ★★★★★★★★ FP8 param sync incompatible with compact LayerWise DDP
```

---

## 4. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 single GPU (dp=1) specific:

Memory savings:
  → Padded layout: dp_size * max(shard_load) per buffer → wasteful on dp=1
  → Compact layout: per-buffer shard_load → no dp_size multiplier → efficient
  → ★★★★★★★★ Savings: up to max(shard_load) - actual_shard_load per buffer
  → For Muon matrices: savings = max(attention_shard) - actual_QKV_shard for small attention layers
  → Cumulative: significant across all layer-wise buffers

★★★★★★★★★ Makes Muon+ZeRO-2 more memory-efficient on RTX 4090:
  → Without compact: Muon buffers padded → wasted memory → less room for activations/LoRA
  → With compact: Muon buffers exact size → efficient → more room for training
  → ★★★★★★★★ Enables larger models or more LoRA parameters on RTX 4090

★★★★★★★★★ Status: DRAFT → experimental → not merged yet
  → ★★★★★★★★ When merged → RTX 4090 Muon training becomes more viable
  → Combined with #5219 (single-GPU crash fix) → Muon on RTX 4090 closer to production
```

---

## Key Findings Summary

★★★★★★★★★ #5391: compact LayerWise DDP → removes dp_size * max(shard_load) padding → per-buffer layout
★★★★★★★★★ dp=1 (RTX 4090): padding = max(shard_load) → wasteful → compact removes entirely
★★★★★★★★★ Per-buffer use_distributed_optimizer: Muon buffers compact (all-reduce), sibling buffers standard (reduce-scatter)
★★★★★★★★★ Backward compatible: default padded unchanged → opt-in compact experimental
★★★★★★★★★ RTX 4090: significant memory savings for Muon layer-wise buffers → more training headroom
★★★★★★★★★ Status: DRAFT → when merged → Muon+ZeRO-2 RTX 4090 more viable
★★★★★★★★★ Constraint: compact requires num_distributed_optimizer_instances == 1 → single GPU OK

---

## References

- Megatron #5391: https://github.com/NVIDIA/Megatron-LM/pull/5391
- Megatron #5388: https://github.com/NVIDIA/Megatron-LM/pull/5388 (dev branch version)
- Megatron #5394: https://github.com/NVIDIA/Megatron-LM/issues/5394 (Muon clipping bug)
- Megatron #5219: https://github.com/NVIDIA/Megatron-LM/pull/5219 (single-GPU crash fix)
- Muon optimizer source reading: notebook/projects/deepspeed-muon-optimizer-source-reading.md
