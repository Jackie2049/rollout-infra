# PyTorch #187620 — FSDP2 PartialOffloadPolicy Reading

> 2026-06-18 | PR #187620 OPEN (DRAFT) | +153/-7 | 3 files: _fsdp_api.py, _fsdp_param.py, _fsdp_param_group.py
> ★★★★★★★★ Fractional CPU parameter offload → RTX 4090 GAME-CHANGER
> ★★★★★★★★ Offload only enough params to fit memory → fraction of copy cost
> ★★★★★★★★ Deterministic largest-first selector → no cross-rank communication
> ★★★★★★★★ DRAFT for API direction → not yet full CI tested

---

## 1. Problem: Full Offload vs No Offload = Binary Choice

```
★★★★★★★★★ Before #187620 (current FSDP2):

Two offload policies only:
  → OffloadPolicy → no offload → all params on GPU → OOM risk on 24GB
  → CPUOffloadPolicy → full offload → ALL params on CPU → host-device copy on EVERY param
  → ★★★★★★★★ Binary choice → no middle ground!

★★★★★★★★★ Why this matters for RTX 4090:
  → Qwen3-8B LoRA: ~16.2 GiB → fits with margin → OffloadPolicy works
  → Qwen3.5-27B LoRA: ~16.2 GiB with per-unit summon → barely fits
  → But if model is slightly over budget: MUST fully offload → pays latency tax
  → ★★★★★★★★ Gap: no way to offload just enough to fit → must pay ALL or NONE

★★★★★★★★★ #187620 fills this gap:
  → PartialOffloadPolicy(offload_ratio=0.3) → offload 30% of params → fit in budget
  → Only offloaded params pay host-device copy → 70% stay on GPU → faster!
  → ★★★★★★★★ "Slightly over budget" case → perfect for RTX 4090 24 GiB
```

---

## 2. PartialOffloadPolicy Design

```
★★★★★★★★★ API:

@dataclass(frozen=True)
class PartialOffloadPolicy(OffloadPolicy):
    offload_ratio: float = 1.0   # fraction of total sharded numel to offload
    pin_memory: bool = True       # pin offloaded memory (same as CPUOffloadPolicy)

★★★★★★★★★ Behavior contract:
  → offload_ratio=0.0 → identical to OffloadPolicy (no offload)
  → offload_ratio=1.0 → identical to CPUOffloadPolicy (full offload)
  → offload_ratio=0.3 → offload 30% of sharded parameter numel
  → ★★★★★★★★ Boundary equivalence → no behavioral change at extremes!

★★★★★★★★★ Usage:

from torch.distributed.fsdp import fully_shard, PartialOffloadPolicy

# Offload 40% of params to CPU → fit in 24 GiB budget
fully_shard(model, policy=PartialOffloadPolicy(offload_ratio=0.4))

# Or for RTX 4090 with Qwen3.5-27B LoRA:
# 16.2 GiB base → ~3 GiB margin → offload_ratio=0.2 might be enough
fully_shard(model, policy=PartialOffloadPolicy(offload_ratio=0.2))
```

---

## 3. Selection Algorithm: Deterministic Greedy Largest-First

```
★★★★★★★★★ _select_partial_offload(numels, offload_ratio) → list[bool]:

Algorithm:
  1. Calculate budget = floor(offload_ratio * total_numel)
  2. Sort params by (-numel, index) → largest-first, ties by original order
  3. Visit each param in sorted order:
     → If adding its numel keeps cumulative <= budget → offload it
     → If adding would exceed budget → skip it
     → BUT continue visiting → later smaller params may fit remaining budget
  4. Return boolean mask per parameter (True=offload, False=resident)

★★★★★★★★★ Key properties:
  → Deterministic: every rank computes IDENTICAL mask → no communication needed
  → Monotonic: higher ratio → superset of offloaded params → no param leaves set
  → Bounded: offloaded numel NEVER exceeds budget → safe
  → Largest-first: biggest params offloaded first → most memory freed per param

★★★★★★★★★ RTX 4090 example — CRITICAL dp=1 insight:

  → On dp=1 (single GPU): FSDP2 shard = identity → shard_size = total_size / 1 = total!
  → Resident params: sharded (full) storage ALWAYS on GPU → for 27B model = 37.8 GiB (ratio=0.3) → OOM!
  → ★★★★★★★★ PartialOffloadPolicy DOES NOT help RTX 4090 dp=1 for large models!
  → Resident shard = full param → can't fit 24 GiB → must offload ratio near 1.0 → ≈ full offload

★★★★★★★★★ The dp=1 limitation explained:

  → FSDP2 with dp=1: each rank has full params → sharding = identity → no memory savings from sharding
  → CPUOffloadPolicy: ALL shards on CPU → peak = largest all-gathered unit + activations → FITS
  → PartialOffloadPolicy: resident shards = full params on GPU → 70% of 27B = 37.8 GiB → OOM!
  → ★★★★★★★★ On single GPU: resident storage = full model portion → exceeds 24 GiB for any large model!

★★★★★★★★★ When PartialOffloadPolicy ACTUALLY helps:
  → Multi-GPU (dp>=2): shard_size = total/dp → resident shard = fraction → can fit
  → dp=2: 27B/2 = 13.5B → resident 70% = 9.45B → 18.9 GiB → FITS on each GPU!
  → dp=4: 27B/4 = 6.75B → resident 70% = 4.725B → 9.45 GiB → comfortable
  → ★★★★★★★★ PartialOffloadPolicy = multi-GPU optimization → NOT single GPU benefit

★★★★★★★★★ RTX 4090 single GPU conclusion:
  → CPUOffloadPolicy (full offload) → STILL the ONLY viable path on dp=1
  → PartialOffloadPolicy → reduces copy bandwidth → but resident storage too large for dp=1
  → ★★★★★★★★ The benefit is ONLY for multi-GPU scenarios → RTX 4090 stays with full offload
```

---

## 4. Implementation Details

```
★★★★★★★★★ 3 files modified:

1. _fsdp_api.py (+55 lines):
   → PartialOffloadPolicy dataclass
   → offload_ratio validation: must be float in [0.0, 1.0]
   → pin_memory attribute (default True)
   → __post_init__ validation (TypeError for non-float, ValueError for out of range)

2. _fsdp_param.py (+11/-4 lines):
   → FSDPParam.__init__ gains offload_to_cpu: bool | None parameter
   → None → derive from policy type (existing behavior preserved)
   → True/False → explicit per-parameter decision from PartialOffloadPolicy selector
   → pin_memory logic updated: offload_to_cpu AND policy.pin_memory

3. _fsdp_param_group.py (+87/-3 lines):
   → _select_partial_offload() pure function → 45 lines
   → Greedy largest-first selector → deterministic → no communication
   → FSDPParamGroup.__init__ computes offload_to_cpu_per_param list
   → If PartialOffloadPolicy → run selector → pass per-param bool to FSDPParam
   → If OffloadPolicy/CPUOffloadPolicy → pass None → existing behavior
   → _validate_cpu_offload_params updated → only check offloaded params on CPU

★★★★★★★★★ Additive API:
  → No change to OffloadPolicy or CPUOffloadPolicy
  → No change to collective hot path
  → No change to public signatures
  → ★★★★★★★★ Opt-in → backward compatible → zero risk for existing users
```

---

## 5. RTX 4090 Impact Assessment

```
★★★★★★★★★ RTX 4090 memory scenarios — CORRECTED with dp=1 insight:

★★★★★★★★★ CRITICAL: PartialOffloadPolicy DOES NOT help RTX 4090 dp=1!
  → FSDP2 dp=1: shard = identity → resident shard = full param → too large for 24 GiB
  → CPUOffloadPolicy (full offload) remains ONLY viable path for large models on single GPU

Scenario 1: Qwen3-8B LoRA (~16.2 GiB peak, dp=1)
  → Fits with full offload → 7.8 GiB margin → CPUOffloadPolicy works
  → PartialOffloadPolicy → resident shard = 8B * 2 = 16 GiB → FITS for ratio=0.0 only
  → ★★★★★★★★ No benefit from partial offload on dp=1 → full offload is sufficient

Scenario 2: Qwen3.5-27B LoRA (~16.2 GiB peak, dp=1, per-unit summon)
  → Full offload fits → CPUOffloadPolicy → 7.8 GiB margin
  → PartialOffloadPolicy → resident shard = 27B * 0.7 * 2 = 37.8 GiB → OOM!
  → ★★★★★★★★ MUST use full offload → can't have any resident params on dp=1

Scenario 3: MoE models (dp=1)
  → Qwen3-MoE (~19.8 GiB peak) → fits with full offload → 4.2 GiB margin
  → PartialOffloadPolicy → resident shard still too large → OOM!
  → ★★★★★★★★ Full offload is the only safe option on dp=1

★★★★★★★★★ PartialOffloadPolicy ACTUALLY helps (multi-GPU only):

Multi-GPU dp=2 example (2× RTX 4090):
  → Qwen3.5-27B with ratio=0.3: resident shard = 27B/2 * 0.7 * 2 = 18.9 GiB → FITS!
  → 30% offloaded → 70% resident → faster forward → less host-device copy
  → ★★★★★★★★ PartialOffloadPolicy = multi-GPU optimization → reduces copy bandwidth
  → On dp=2: each GPU holds resident shard (18.9 GiB) → FITS with 5 GiB margin

★★★★★★★★★ RTX 4090 single GPU conclusion:
  → CPUOffloadPolicy (full offload) → ONLY viable path → proven → stable → works NOW
  → PartialOffloadPolicy → multi-GPU only → NOT beneficial on dp=1
  → ★★★★★★★★ This CORRECTS earlier "RTX 4090 game-changer" claim → it's multi-GPU game-changer!

★★★★★★★★★ Integration path for verl:
  → verl supports FSDP2 backend → fsdp_config can specify offload policy
  → Current: param_offload=True → CPUOffloadPolicy equivalent
  → Future: param_offload_ratio=0.3 → PartialOffloadPolicy → fraction offload
  → ★★★★★★★★ verl config enhancement needed → but PyTorch foundation exists!
```

---

## 6. Relationship to Other Developments

```
★★★★★★★★★ Related PRs and context:

PyTorch RFC #114299: per-parameter FSDP direction → axis of this PR
  → PartialOffloadPolicy is on the parameter axis
  → Adjacent to activation offload (#174960)

PyTorch FSDP2 single GPU analysis (our reading):
  → dp=1 → FSDP2 all-gather = identity → minimal overhead
  → But param_offload still useful → host-device copy for offloaded params
  → ★★★★★★★★ PartialOffloadPolicy + dp=1 → offload only large params → minimal copy tax

verl #6512 (per-unit LoRA summon):
  → Per-unit summon → peak bounded by largest FSDP unit
  → PartialOffloadPolicy → per-unit offload control → complementary!
  → ★★★★★★★★ Combined: summon only what you need + offload only what you must

Megatron #5387 (FSDP fully_shard):
  → Megatron's own FSDP implementation → different approach
  → DBuffer primitives → release_storage/reallocate_storage → similar memory management
  → ★★★★★★★★ Two FSDP implementations evolving → Megatron + PyTorch → convergence?

DeepSpeed ZeRO-2 + CPU_Adam:
  → Current RTX 4090 approach → ALL optimizer states on CPU
  → PartialOffloadPolicy → FUTURE approach → fraction of params on CPU
  → ★★★★★★★★ When #187620 merges → RTX 4090 can choose: full offload (ZeRO-2) vs partial (FSDP2)
```

---

## Key Findings Summary

★★★★★★★★★ #187620: PartialOffloadPolicy → fractional CPU offload → multi-GPU benefit ONLY
★★★★★★★★★ CORRECTED: NOT RTX 4090 dp=1 game-changer → FSDP2 dp=1 shard=identity → resident too large
★★★★★★★★★ CPUOffloadPolicy (full offload) → STILL ONLY viable path for RTX 4090 single GPU
★★★★★★★★★ PartialOffloadPolicy helps dp>=2 → resident shard smaller → can fit → less copy bandwidth
★★★★★★★★★ Deterministic greedy largest-first selector → no cross-rank communication needed
★★★★★★★★★ 3 files, +153/-7 → additive API → backward compatible
★★★★★★★★★ RTX 4090 dp=1: MUST use full offload → partial offload resident storage exceeds 24 GiB
★★★★★★★★★ Multi-GPU future: dp=2 → PartialOffloadPolicy(ratio=0.3) → faster forward + less copy

---

## References

- PyTorch #187620: https://github.com/pytorch/pytorch/pull/187620
- PyTorch RFC #114299: per-parameter FSDP direction
- PyTorch #174960: activation offload
- verl #6512: per-unit LoRA summon
- Megatron #5387: FSDP fully_shard
- FSDP2 single GPU: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- RTX 4090 config: tools/rtx4090_grpo_config_reference.py
