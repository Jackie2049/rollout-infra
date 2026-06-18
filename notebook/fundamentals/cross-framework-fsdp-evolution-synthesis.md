# Cross-Framework FSDP Evolution Synthesis — RTX 4090 Consultant Guide

> 2026-06-18 | Synthesis of FSDP developments across PyTorch, Megatron, verl
> ★★★★★★★★ FSDP implementations converging → summon/release lifecycle → partial offload → per-unit granularity
> ★★★★★★★★ RTX 4090: PyTorch FSDP2 + PartialOffloadPolicy = future best path
> ★★★★★★★★ Universal pattern: materialize what you need → compute → release → peak bounded by unit size

---

## 1. The Summon/Release Lifecycle Pattern

```
★★★★★★★★★ SAME pattern found independently across 4 implementations:

Pattern: summon → compute → release → peak bounded by unit size

Implementation 1: PyTorch FSDP2 (standard)
  → all-gather params before forward → compute → reshard after forward
  → Peak = largest FSDP unit (not whole model)
  → ★★★★★★★★ Standard approach → every major framework uses this

Implementation 2: Megatron MFSDPv2 (#5387)
  → unshard via DBuffer reallocate_storage → compute → reshard via release_storage(0)
  → release_storage(0) → resize untyped_storage to 0 → frees GPU allocation
  → ★★★★★★★★ Explicit lifecycle → more control → same summon/release pattern
  → BUT: autograd may save views sharing Storage → resize preserves aliases → safe

Implementation 3: verl per-unit LoRA summon (#6512)
  → summon_full_params per FSDP unit → extract LoRA params → send to rollout
  → Peak = largest FSDP unit (not whole model) → 10x reduction!
  → ★★★★★★★★ Same pattern applied to LoRA weight sync → not just forward compute

Implementation 4: DeepSpeed ZeRO-2 + CPU_Adam
  → all-gather params from CPU → compute → shard back → optimizer states on CPU
  → Host-device copy for every step → but optimizer states never on GPU
  → ★★★★★★★★ Full offload variant → same summon/release but from CPU → slower

★★★★★★★★★ Universal insight:
  → Peak memory = size of WHAT you materialize, not size of WHAT you store
  → Granularity = unit size (FSDP unit, LoRA per-expert, CPU offload per-param)
  → ★★★★★★★★ FINE granularity → LOW peak → FITS small GPU → RTX 4090 viable!
```

---

## 2. Three Levels of Memory Optimization

```
★★★★★★★★★ Level 1: Whole-model materialization (OLD, BROKEN on RTX 4090)
  → summon all params at once → peak = whole model → 60 GiB → OOM
  → Example: old verl LoRA root-level summon → FSDP root = entire model
  → ★★★★★★★★ DOES NOT WORK on RTX 4090 for models > 14B

★★★★★★★★★ Level 2: Per-unit materialization (CURRENT, WORKS on RTX 4090)
  → summon per FSDP unit → peak = largest unit → 6-8 GiB → fits!
  → Example: verl #6512 per-unit LoRA summon → FSDP1/FSDP2 compatible
  → PyTorch FSDP2 standard behavior → all-gather per unit
  → ★★★★★★★★ WORKS on RTX 4090 → but ALL optimizer states still on CPU (full offload)

★★★★★★★★★ Level 3: Fractional offload (FUTURE, BEST for RTX 4090)
  → PartialOffloadPolicy → offload only fraction → resident params stay on GPU
  → Peak = largest offloaded unit + resident params → balanced
  → ★★★★★★★★ Example: offload_ratio=0.3 → 30% params on CPU → 70% on GPU → faster forward
  → MoE expert params = largest → offloaded first → ideal for greedy selector
  → ★★★★★★★★ BEST for RTX 4090 → minimize host-device copy → maximize GPU residency
```

---

## 3. PyTorch FSDP2 vs Megatron MFSDPv2 — Architecture Comparison

```
★★★★★★★★★ Key architectural differences:

| Aspect | PyTorch FSDP2 | Megatron MFSDPv2 |
|--------|---------------|-------------------|
| Primitive | DTensor (standard) | DBuffer (custom) |
| Module | _FSDPParamGroup (static) | FsdpModule mixin (dynamic) |
| TP integration | separate (TP + FSDP) | native (TE metadata on nn.Parameter) |
| Storage | implicit (Python GC) | explicit (release/reallocate) |
| Meta params | native DTensor | #5369 (to_empty + reset_parameters) |
| Partial offload | #187620 (PartialOffloadPolicy) | NOT yet |
| State | Production, stable | Experimental, Final Review |

★★★★★★★★★ For RTX 4090 single GPU:
  → PyTorch FSDP2 + PartialOffloadPolicy → BEST single-GPU path
  → verl uses FSDP2 → integration path exists
  → Megatron MFSDPv2 → better for multi-GPU TP integration → future scale-out
  → ★★★★★★★★ Current recommendation: FSDP2 + ZeRO-2 CPU_Adam → future: FSDP2 + PartialOffloadPolicy
```

---

## 4. RTX 4090 Memory Budget Analysis with Partial Offload

```
★★★★★★★★★ Current vs Future RTX 4090 memory optimization:

Current (verl CPPO+bypass + FSDP2 + CPUOffloadPolicy):
  → ALL params/grads/optimizer on CPU → full host-device copy every step
  → Peak GPU: 16.2 GiB (with per-unit summon + bypass)
  → Forward: all-gather params from CPU → compute → shard back
  → ★★★★★★★★ Latency: host-device copy on EVERY forward → significant overhead

Future (verl CPPO+bypass + FSDP2 + PartialOffloadPolicy):
  → 30% params on CPU → 70% resident on GPU → partial copy only
  → Peak GPU: resident params + activations + offloaded params (during forward)
  → Forward: all-gather only offloaded params → resident params already on GPU
  → ★★★★★★★★ Latency: 30% host-device copy → 70% zero-copy → much faster!

★★★★★★★★★ Memory scenarios with PartialOffloadPolicy:

Qwen3-8B LoRA (fits easily):
  → offload_ratio=0.0 → all on GPU → 7.8 GiB margin → no need for offload

Qwen3.5-27B LoRA (tight):
  → offload_ratio=0.2 → 20% offloaded → ~3.2 GiB freed → comfortable margin
  → ★★★★★★★★ "Slightly over budget" → partial offload → fraction of copy cost

MoE models (very tight):
  → offload_ratio=0.35 → 35% offloaded → ~7 GiB freed → good margin
  → Expert params (largest) → offloaded first → ideal for greedy algorithm
  → ★★★★★★★★ MoE + partial offload = sweet spot for RTX 4090
```

---

## 5. Integration Path for verl

```
★★★★★★★★★ Current verl config (FSDP2 + CPUOffloadPolicy equivalent):

actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      param_offload: True       # = CPUOffloadPolicy
      optimizer_offload: True   # = optimizer states on CPU

★★★★★★★★★ Future verl config (FSDP2 + PartialOffloadPolicy):

actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      param_offload_policy: PartialOffloadPolicy  # NEW
      param_offload_ratio: 0.3                     # NEW → 30% offloaded
      optimizer_offload: True                       # optimizer states still fully CPU

★★★★★★★★★ Integration requirements:
  1. PyTorch #187620 must MERGE first → PartialOffloadPolicy available
  2. verl must add PartialOffloadPolicy config option → fsdp_config
  3. verl must update FSDPEngine to support new policy → pass to fully_shard()
  4. ★★★★★★★★ Timeline: #187620 DRAFT → months → but infrastructure exists to prepare

★★★★★★★★★ What can be done NOW:
  → Monitor #187620 for API direction resolution → watch maintainer feedback
  → Prepare verl config schema for offload_ratio → easy addition when #187620 merges
  → ★★★★★★★★ Continue with CPUOffloadPolicy (full offload) → it WORKS → stable → RTX 4090 #1
```

---

## Key Findings Summary

★★★★★★★★★ FSDP summon/release lifecycle = universal pattern → found in 4 implementations independently
★★★★★★★★★ Peak memory = materialized unit size, NOT stored model size → fine granularity = low peak
★★★★★★★★★ Three levels: whole-model (broken) → per-unit (works) → fractional offload (best)
★★★★★★★★★ PyTorch #187620 PartialOffloadPolicy → RTX 4090 game-changer → fractional CPU offload
★★★★★★★★★ Megatron #5387 MFSDPv2 → DBuffer primitives → explicit storage lifecycle → same pattern
★★★★★★★★★ verl #6512 per-unit LoRA summon → same summon/release pattern → 10x peak reduction
★★★★★★★★★ RTX 4090 path: FSDP2 + CPUOffloadPolicy (now) → FSDP2 + PartialOffloadPolicy (future)
★★★★★★★★★ MoE models benefit most from partial offload → expert params = largest → offloaded first
★★★★★★★★★ Integration: verl config needs offload_ratio → simple addition → but #187620 must merge first

---

## References

- PyTorch #187620: PartialOffloadPolicy (fractional CPU offload)
- Megatron #5387: MFSDPv2 fully_shard (DBuffer primitives)
- Megatron #5369: meta-parameter support (follow-up)
- verl #6512: per-unit LoRA summon (10x peak reduction)
- verl #6699: detach memory fix (4x reduction)
- verl #6731: CPPO bypass (18Ψ→3.8Ψ)
- PyTorch FSDP2 analysis: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- RTX 4090 runbook: notebook/projects/rtx4090-grpo-training-runbook.md
