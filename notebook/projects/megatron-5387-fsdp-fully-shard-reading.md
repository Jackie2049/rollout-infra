# Megatron #5387 — Experimental Megatron-FSDP fully_shard Reading

> 2026-06-18 | PR #5387 OPEN (Final Review) | +993/-3 | 8 files | MFSDPv2 label
> ★★★★★★★★ NVIDIA's own FSDP implementation using DBuffer primitives
> ★★★★★★★★ Per-module fully_shard → FsdpModule mixin + FsdpParameterGroup + Placements
> ★★★★★★★★ Forward/backward hooks for unshard/reshard/gradient reduction
> ★★★★★★★★ Author: wujingyue (Jingyue Wu) → recovered from #4976
> ★★★★★★★★ Follow-up: #5369 meta-parameter support (+2398/-0)

---

## 1. Architecture: Megatron's Own FSDP (MFSDPv2)

```
★★★★★★★★★ Key design choice: NOT using PyTorch FSDP2 → Megatron builds its own!

Why? Megatron has unique constraints:
  → Tensor-parallel sharding metadata lives on nn.Parameter (TE integration)
  → DBuffer primitives → lower-level control than FSDP2's DTensor-based API
  → Per-module granularity → each module = one FSDP unit → no nested wrapping
  → ★★★★★★★★ Megatron-FSDP = custom sharding on top of Megatron's existing TP infra

★★★★★★★★★ Architecture overview:

fully_shard(module, mesh, placements, mixed_precision_policy)
  → Attaches FsdpModule mixin to module → _attach_mixin() → dynamic class creation
  → Creates FsdpParameterGroup per dtype/requires_grad group
  → DBuffer manages sharded storage → local_buffer → all-gather/reduce-scatter
  → Forward hook: unshard → materialize full params → compute → reshard → release storage
  → Backward hook: gradient reduction → accumulate across backward calls

★★★★★★★★★ New primitives added:

fully_shard.py (61 lines):
  → fully_shard() entry point
  → _attach_mixin() → dynamic FsdpModule mixin → type(f"ExperimentalFsdp{cls}", (FsdpModule, cls), {})

module.py (161 lines):
  → FsdpModule mixin → forward/backward lifecycle management
  → _parameter_groups: tuple[FsdpParameterGroup, ...]
  → _ready_grad_parameters: set[nn.Parameter] → gradient completion tracking

parameter_group.py (~200 lines):
  → FsdpParameterGroup → groups params by dtype + requires_grad
  → DBuffer-based sharded/replicated storage management

Placements:
  → Placements → parameter/gradient/optimizer placement configuration
  → MeshAxis → specifies which mesh dimension for sharding
```

---

## 2. DBuffer Primitives Evolution

```
★★★★★★★★★ DBuffer additions in this PR:

Storage lifecycle:
  → reallocate_storage() → restore local buffer to logical size
  → release_storage() → release allocation WITHOUT replacing Storage object
  → ★★★★★★★★ Why: autograd may save views sharing this Storage → resizing preserves aliases
  → _resize_storage(numel) → resize untyped_storage by numel * element_size
  → ★★★★★★★★ Pattern: Storage resize(0) → release → Storage resize(N) → reallocate

Cast support:
  → cast(dtype) → return new DBuffer with same layout/placements in different dtype
  → ★★★★★★★★ Mixed precision: compute in bf16 → main weights in fp32 → cast between

★★★★★★★★★ Storage lifecycle pattern (RTX 4090 relevant):
  → Forward: unshard → allocate full params → compute → reshard → release_storage(0)
  → ★★★★★★★★ release_storage(0) → frees GPU allocation → saves peak memory!
  → After compute → don't need full params → release → peak = compute-only
  → This is the SAME pattern as per-unit LoRA summon (#6512) → summon → compute → release
  → ★★★★★★★★ Convergence: FSDP parameter lifecycle = LoRA weight lifecycle → same pattern!
```

---

## 3. MFSDPv2 vs PyTorch FSDP2 Comparison

```
★★★★★★★★★ Architecture comparison:

| Feature | Megatron MFSDPv2 | PyTorch FSDP2 |
|---------|-----------------|---------------|
| Sharding primitive | DBuffer (custom) | DTensor (standard) |
| Module attachment | FsdpModule mixin (dynamic) | _FSDPParamGroup (static) |
| TP integration | native (TE metadata on nn.Parameter) | separate (TP + FSDP layers) |
| Storage lifecycle | release/reallocate (explicit) | implicit (Python GC) |
| Meta params | #5369 (follow-up, +2398) | native DTensor support |
| Partial offload | NOT yet | #187620 (PartialOffloadPolicy) |
| Gradient reduction | completion-based hooks | post-backward hooks |
| Mixed precision | MixedPrecisionPolicy | MixedPrecisionPolicy |

★★★★★★★★★ Key difference:
  → Megatron: DBuffer = custom sharding primitive → lower-level → more control
  → PyTorch: DTensor = standard sharding → higher-level → more portable
  → ★★★★★★★★ Megatron needs TP-aware sharding → TE integration → custom DBuffer justified
  → PyTorch FSDP2 → more general → works across frameworks → verl uses it

★★★★★★★★★ RTX 4090 perspective:
  → Single GPU → dp=1 → FSDP sharding = identity → minimal difference
  → But: storage lifecycle (release/reallocate) → memory savings on ANY GPU
  → ★★★★★★★★ For RTX 4090: both approaches → same effect → dp=1 identity overhead
  → PyTorch FSDP2 + PartialOffloadPolicy → more flexible for single GPU
  → Megatron MFSDPv2 → better TP integration → multi-GPU scale-out
```

---

## 4. Follow-up: #5369 Meta-Parameter Support

```
★★★★★★★★★ #5369 (OPEN, +2398/-0, 9 files):

Purpose: restore experimental fully_shard handling for meta parameters
  → Materialize with to_empty() → reset_parameters() → then shard
  → ★★★★★★★★ Meta params = uninitialized → need device-aware initialization before sharding
  → DTensor sharding requires real data → can't shard meta params directly

Key changes:
  → Restore meta parameter materialization path
  → to_empty() → create uninitialized tensors on correct device
  → reset_parameters() / _reset_parameters() → initialize with model-specific logic
  → Then FSDPParameterGroup construction → shard initialized params

★★★★★★★★★ Why separated from #5387:
  → Meta parameter support is optional → some users don't need it
  → Can be kept, revised, or dropped independently
  → ★★★★★★★★ Modular → minimal FSDP (#5387) first → meta params (#5369) optional follow-up

★★★★★★★★★ RTX 4090 relevance:
  → Meta params → model initialization on device → memory-aware init
  → For large models → to_empty() → no actual allocation → then shard → then materialize
  → ★★★★★★★★ "Initialize on device → shard → compute" pattern → RTX 4090 memory-efficient
```

---

## 5. Relationship to Other Megatron Developments

```
★★★★★★★★★ Dependency chain for RTX 4090:

Current blockers:
  → #5219 (single-GPU Muon crash) → Final Review → close to merge
  → #5395 (skip_grad_norm_clip) → OPEN → needed for Muon
  → #5391 (compact LayerWise DDP) → OPEN → memory efficiency

New developments:
  → #5387 (MFSDPv2 fully_shard) → Final Review → experimental FSDP alternative
  → #5369 (meta params) → OPEN → follow-up to #5387
  → #5389 (GDN THD all-to-all) → MERGED June 17 → fused GDN THD restore on dev
  → #5372 (MimoModel zero_grad_buffer) → MERGED → DDP submodule delegation

★★★★★★★★★ RTX 4090 path with MFSDPv2 (future):
  → If MFSDPv2 matures → Megatron could use own FSDP → not PyTorch's
  → But: dp=1 → identity overhead → same as FSDP2 on single GPU
  → ★★★★★★★★ For RTX 4090: MFSDPv2 = future multi-GPU path → not immediate benefit
  → Current: PyTorch FSDP2 (verl) remains #1 RTX 4090 approach
```

---

## Key Findings Summary

★★★★★★★★★ #5387: Megatron's own FSDP implementation → DBuffer primitives → per-module fully_shard
★★★★★★★★★ FsdpModule mixin + FsdpParameterGroup + Placements → 993 additions
★★★★★★★★★ Storage lifecycle: release/reallocate → same pattern as per-unit LoRA summon (#6512)
★★★★★★★★★ MFSDPv2 vs FSDP2: custom DBuffer vs standard DTensor → different primitives → same goal
★★★★★★★★★ #5369 follow-up: meta parameter materialization (+2398/-0) → modular optional add-on
★★★★★★★★★ #5389 MERGED: GDN THD all-to-all restored on dev branch
★★★★★★★★★ #5372 MERGED: MimoModel zero_grad_buffer → DDP submodule delegation
★★★★★★★★★ RTX 4090: MFSDPv2 = future path → PyTorch FSDP2 + PartialOffloadPolicy remains #1
★★★★★★★★★ Convergence: FSDP storage lifecycle = LoRA weight lifecycle → same summon/release pattern

---

## References

- Megatron #5387: https://github.com/NVIDIA/Megatron-LM/pull/5387
- Megatron #5369: https://github.com/NVIDIA/Megatron-LM/pull/5369
- Megatron #4976: https://github.com/NVIDIA/Megatron-LM/pull/4976 (original, GitHub closed)
- Megatron #5389: https://github.com/NVIDIA/Megatron-LM/pull/5389 (GDN THD MERGED)
- Megatron #5372: https://github.com/NVIDIA/Megatron-LM/pull/5372 (MimoModel MERGED)
- PyTorch #187620: PartialOffloadPolicy (fractional offload)
- verl #6512: per-unit LoRA summon
- FSDP2 analysis: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
