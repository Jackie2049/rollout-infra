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

★★★★★★★★★ RTX 4090 example (Qwen3.5-27B LoRA, dp=1):
  → Total sharded param numel: ~5B (27B params / 1 rank = 27B)
  → Wait — dp=1 → FSDP2 sharding is identity → each rank has full params
  → Actually for dp=1: sharded numel = total numel / world_size = total / 1 = total
  → Budget = floor(0.2 * total) → offload largest params first
  → ★★★★★★★★ On single GPU: FSDP2 all-gather = identity → but offload still works!
  → Offloaded params stored on CPU → materialized on GPU for forward → then freed
  → Resident params always on GPU → no copy latency
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
★★★★★★★★★ RTX 4090 memory scenarios:

Scenario 1: Qwen3-8B LoRA (~16.2 GiB peak)
  → Fits with 7.8 GiB margin → PartialOffloadPolicy(0.0) → same as OffloadPolicy
  → No need for partial offload → but future larger models need it!

Scenario 2: Qwen3.5-27B LoRA (~16.2 GiB peak with per-unit summon)
  → Tight fit → 7.8 GiB margin → but what if optimizer states + activations overflow?
  → PartialOffloadPolicy(0.15) → offload 15% → ~2.4 GiB freed → more margin
  → ★★★★★★★★ "Slightly over budget" → partial offload → perfect fit

Scenario 3: MoE models (Qwen3.5-35B-A3B, ~20 GiB)
  → Very tight → 4 GiB margin → any overflow → OOM
  → PartialOffloadPolicy(0.3) → offload 30% → ~6 GiB freed → comfortable margin
  → ★★★★★★★★ MoE expert params = largest → offloaded first → ideal for greedy algo!

★★★★★★★★★ Comparison with current strategies:
  → Current: full CPU_Adam offload (ZeRO-2 style) → ALL params/grads/optimizer on CPU
  → PartialOffloadPolicy: only offload fraction → resident params stay GPU → faster forward
  → ★★★★★★★★ Potential benefit: 30-50% less host-device copy bandwidth per step
  → But: needs FSDP2 (verl supports FSDP2) → verl CPPO+bypass + PartialOffloadPolicy?

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

★★★★★★★★★ #187620: PartialOffloadPolicy → fractional CPU offload → RTX 4090 GAME-CHANGER
★★★★★★★★★ offload_ratio in [0.0, 1.0] → boundary equivalence with existing policies
★★★★★★★★★ Deterministic greedy largest-first selector → no cross-rank communication needed
★★★★★★★★★ 3 files, +153/-7 → additive API → backward compatible
★★★★★★★★★ RTX 4090: "slightly over budget" → offload just enough → fraction of copy cost
★★★★★★★★★ MoE models benefit most → expert params = largest → offloaded first
★★★★★★★★★ DRAFT → API direction review → not yet full CI → watch for progress
★★★★★★★★★ Integration: verl FSDP2 + PartialOffloadPolicy → future RTX 4090 config option
★★★★★★★★★ Complementary with #6512 per-unit summon → summon what you need + offload what you must

---

## References

- PyTorch #187620: https://github.com/pytorch/pytorch/pull/187620
- PyTorch RFC #114299: per-parameter FSDP direction
- PyTorch #174960: activation offload
- verl #6512: per-unit LoRA summon
- Megatron #5387: FSDP fully_shard
- FSDP2 single GPU: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- RTX 4090 config: tools/rtx4090_grpo_config_reference.py
