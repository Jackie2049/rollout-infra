# PyTorch #187620 -- FSDP2 PartialOffloadPolicy: Deep Technical Analysis

> 2026-06-18 | PR #187620 OPEN (DRAFT) | +153/-7 | 3 files | Author: joemunene-by (Joe Munene)
> Commit: 559722dfbc14c33d203f18cb561d9ec47653efeb (2026-06-17)
> Labels: release notes: distributed (fsdp), ciflow/inductor, ciflow/torchtitan
> Scoping issue: #187615 (same author, same day, 0 maintainer comments)
> Adjacent: #174960 (activation offload, niyunsheng, OPEN since 2026-02-13)
> Axis: #114299 (per-parameter FSDP RFC, awgu, OPEN since 2023-11-21)
> ★★★★★★★★ Fractional CPU parameter offload → multi-GPU optimization, NOT dp=1!
> ★★★★★★★★ CRITICAL RTX 4090: PartialOffloadPolicy DOES NOT work on dp=1!
> ★★★★★★★★ FSDP2 dp=1 → shard=identity → resident shard=full param → exceeds 24 GiB
> ★★★★★★★★ CPUOffloadPolicy (full offload) remains ONLY viable dp=1 path for RTX 4090

---

## 1. Problem Statement: The Binary Offload Gap

```
★★★★★★★★★ Current FSDP2 offload policies — TWO choices only:

  OffloadPolicy (base class, @dataclass):
    → No offload at all → all sharded params resident on GPU
    → All optimizer states on GPU → peak = params + grads + optimizer
    → For 8B bf16: ~16 GiB weights + ~16 GiB grads + ~32 GiB Adam = 64 GiB → WAY over 24!
    → Even with per-unit summon (#6512): peak ~16.2 GiB → FITS → no offload needed
    → BUT: larger models (27B+) → ~50+ GiB → OOM even with summon

  CPUOffloadPolicy (@dataclass, pin_memory=True by default):
    → ALL sharded params on CPU → ALL grads on CPU → ALL optimizer states on CPU
    → Peak GPU = all-gathered unit (forward) + activations → bounded by largest unit
    → For 8B with per-unit summon: peak ~16.2 GiB → 7.8 GiB margin → FITS
    → Host-device copy on EVERY forward pass → latency tax on ALL params
    → ★★★★★★★★ Works on dp=1 because ALL shards on CPU → GPU peak = forward-only

★★★★★★★★★ THE GAP that #187620 fills:

  "Slightly over budget" scenario:
    → Model peak with no offload = 24.5 GiB → OOM by 0.5 GiB
    → CPUOffloadPolicy → peak = ~16 GiB → FITS → but pays copy on ALL params
    → ★★★★★★★★ Only need 0.5 GiB offloaded → but current API forces ALL or NONE!
    → #187620: PartialOffloadPolicy(offload_ratio=0.02) → offload 2% → close the gap
    → 98% of params stay on GPU → zero copy → only 2% pay host-device transfer

★★★★★★★★★ PR author's exact motivation (from PR description):

  "Memory-constrained single-accelerator training needs to offload just enough
   parameter memory to fit the step. Full offload pays host-device copy latency
   on every parameter; no offload OOMs. A ratio lets the user spend exactly
   the copy bandwidth required to fit."

★★★★★★★★★ BUT: this motivation has a subtle flaw for dp=1:
  → "Slightly over budget" → implies resident params can stay on GPU
  → On dp=1: resident shard = FULL param → no memory savings from sharding
  → If model barely fits → resident params STILL occupy full model size → OOM!
  → Only CPUOffloadPolicy (all on CPU) → peak = forward-only → actually fits
  → ★★★★★★★★ PartialOffloadPolicy helps dp>=2 ONLY → see Section 5 for full analysis
```

---

## 2. RFC #187615 -- Scoping Issue and API Design Discussion

```
★★★★★★★★★ RFC #187615 (opened 2026-06-17, same author joemunene-by):

Title: "[FSDP2] Fractional CPU offload policy (PartialOffloadPolicy)"
State: OPEN, 0 maintainer comments → no direction yet

Key insight from RFC: the runtime ALREADY supports per-parameter offload!
  → Each FSDPParam carries offload_to_cpu boolean → set uniformly from policy
  → The runtime machinery is finer-grained than the policy surface
  → ★★★★★★★★ Policy = all-or-nothing layer → underneath = per-param capability
  → #187620 just lifts the per-param capability to the policy level!

★★★★★★★★★ RFC questions for maintainers (3 open design questions, 0 answers):

Q1: "Is parameter-numel-fraction the right knob, or would a memory-byte
     budget (accounting dtype) be preferred as the primary API?"
  → Current PR: offload_ratio operates on NUMEL (element count)
  → Alternative: byte budget → accounts for dtype → bf16 = 2 bytes vs fp32 = 4
  → Implication: numel-based is simpler → byte-based is more precise for mixed precision
  → ★★★★★★★★ For RTX 4090: numel-based is sufficient → all params bf16 → 2 bytes/element
  → For general case: mixed precision (fp32 master weights + bf16 compute) → byte-budget more precise

Q2: "Should the selector live in the policy, in _fsdp_init, or in the
     param group? Draft places the selection at the per-FSDPParam offload
     assignment site."
  → Current PR: selector in _fsdp_param_group.py → applied at FSDPParamGroup.__init__
  → ★★★★★★★★ Current placement is correct → group-level selection → per-param assignment
  → Alternative: in policy class → more self-contained but needs group context
  → Alternative: in _fsdp_init → closer to initialization flow but further from selector logic

Q3: "Preference on prefix-stop vs an exact-fill selector, given the
     monotonicity/superset property argues for prefix-stop?"
  → Prefix-stop: largest-first, stop when next param exceeds budget → monotonic
  → Exact-fill: try ALL combinations to fill budget exactly → complex, non-monotonic
  → ★★★★★★★★ PR chose prefix-stop → correct → monotonicity is critical for usability
  → Monotonicity: raising ratio only extends the prefix → offloaded set grows as superset
  → This means: ratio=0.3 offloads A,B,C → ratio=0.4 offloads A,B,C,D → NEVER removes A

★★★★★★★★★ Adjacent work (axis hierarchy):
  → RFC #114299: per-parameter FSDP direction → the AXIS this PR is on
    → Author: awgu (PyTorch FSDP team lead) → opened 2023-11-21 → OPEN 2.5+ years!
    → Per-parameter sharding → enables: flexible fp8, frozen+non-frozen mix, sharded state dicts
    → PartialOffloadPolicy is on the PARAMETER axis → #114299 is the umbrella direction
  → #174960: activation offload → adjacent on the ACTIVATION axis (not parameter)
    → Author: niyunsheng → opened 2026-02-13 → OPEN
    → Proposes activation offload WITHIN CPUOffloadPolicy → reuses prefetch infrastructure
    → Together: parameter offload + activation offload → complete memory control
```

---

## 3. PartialOffloadPolicy API Design and Code Diff

### 3.1 _fsdp_api.py -- New PartialOffloadPolicy Dataclass (+55 lines)

```
★★★★★★★★★ Full definition from PR diff:

@dataclass(frozen=True)
class PartialOffloadPolicy(OffloadPolicy):
    """
    This offload policy offloads only a fraction of a group's sharded
    parameters to CPU, leaving the remainder resident on device. Unlike
    CPUOffloadPolicy, which offloads all of a group's sharded parameters,
    this policy offloads only enough whole parameters to meet offload_ratio
    of the group's total sharded parameter numel.

    This targets the common case where a model is slightly over the
    device-memory budget and full offload pays an unnecessary host-device
    copy tax on every step. Offloading a fraction of the shards closes a
    small memory gap at a fraction of the copy cost.

    The selection is greedy and deterministic: parameters are offloaded
    largest-first (by sharded numel, ties broken by group order) until the
    cumulative offloaded numel would exceed offload_ratio of the group
    total. Because every rank holds the same shard shapes for a group,
    every rank makes the identical selection with no extra communication,
    preserving FSDP2's invariant that gather and reduce participants agree
    on layout.

    offload_ratio=0.0 is observably identical to OffloadPolicy;
    offload_ratio=1.0 is observably identical to CPUOffloadPolicy.
    """

    offload_ratio: float = 1.0
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.offload_ratio, (int, float)) or isinstance(
            self.offload_ratio, bool
        ):
            raise TypeError(
                "PartialOffloadPolicy.offload_ratio must be a float, got "
                f"{type(self.offload_ratio).__name__}"
            )
        if not (0.0 <= self.offload_ratio <= 1.0):
            raise ValueError(
                "PartialOffloadPolicy.offload_ratio must be in [0.0, 1.0], got "
                f"{self.offload_ratio}"
            )

★★★★★★★★★ Key design decisions in the API:
  → frozen=True → immutable → hashable → safe for configuration reuse
  → ★★★★★★★★ Contrast: OffloadPolicy and CPUOffloadPolicy are @dataclass (NOT frozen) → MUTABLE!
    → New policy is stricter → safer for distributed configs → prevents accidental mutation
  → Default offload_ratio=1.0 → equivalent to CPUOffloadPolicy at default → boundary-exact
  → pin_memory=True by default → matches CPUOffloadPolicy default behavior
  → Bool explicitly excluded → isinstance(offload_ratio, bool) → TypeError
    → Python: bool is subclass of int → isinstance(True, int) is True → must explicitly reject
    → Prevents accidental offload_ratio=True → would pass isinstance but not [0.0, 1.0] range check
  → ★★★★★★★★ Boundary equivalence built into the design:
    → offload_ratio=0.0 → NO offload → exactly OffloadPolicy behavior
    → offload_ratio=1.0 → ALL offload → exactly CPUOffloadPolicy behavior
```

### 3.2 _fsdp_param.py -- Per-Parameter Offload Flag (+12/-3 lines)

```
★★★★★★★★★ FSDPParam.__init__ gains offload_to_cpu parameter:

BEFORE (line 199-204):
  def __init__(
      self,
      param: nn.Parameter,
      module_info: ParamModuleInfo,
      mesh_info: DataParallelMeshInfo,
      post_forward_mesh_info: FSDPMeshInfo | None,
      device: torch.device,
      shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
      mp_policy: MixedPrecisionPolicy,
      offload_policy: OffloadPolicy,                    # ← original
  ):
      ...
      self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
      self.pin_memory = (
          self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
      )

AFTER (line 199-210):
  def __init__(
      self,
      ...
      offload_policy: OffloadPolicy,
      offload_to_cpu: bool | None = None,               # ← NEW parameter
  ):
      ...
      # offload_to_cpu may be set explicitly by the owning param group
      # (e.g. PartialOffloadPolicy selects a per-parameter subset to
      # offload). When None, fall back to the all-or-nothing decision
      # derived from the policy type, which preserves OffloadPolicy (no
      # offload) and CPUOffloadPolicy (full offload) behavior exactly.
      if offload_to_cpu is None:
          self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
      else:
          self.offload_to_cpu = offload_to_cpu
      self.pin_memory = self.offload_to_cpu and bool(
          getattr(offload_policy, "pin_memory", False)
      )

★★★★★★★★★ Critical change: pin_memory logic generalized!
  → BEFORE: cast(offload_policy, CPUOffloadPolicy).pin_memory → ONLY worked for CPUOffloadPolicy
  → AFTER: getattr(offload_policy, "pin_memory", False) → works for ALL policies
  → CPUOffloadPolicy.pin_memory → True/False → correctly read via getattr
  → PartialOffloadPolicy.pin_memory → True/False → correctly read via getattr
  → OffloadPolicy → no pin_memory attribute → getattr returns False → offload_to_cpu=False → False
  → ★★★★★★★★ This is the key change that enables per-param offload decisions!
  → ★★★★★★★★ Also fixes a subtle bug: cast(CPUOffloadPolicy, ...) would CRASH on OffloadPolicy
```

### 3.3 _fsdp_param_group.py -- Group-Level Selection (+89/-4 lines)

```
★★★★★★★★★ New pure function: _select_partial_offload (lines 153-206, ~55 lines):

def _select_partial_offload(numels: list[int], offload_ratio: float) -> list[bool]:
    """Choose which parameters in a group to offload to CPU.

    Given the ordered list of per-parameter sharded numels for a group and
    a target offload_ratio in [0.0, 1.0], return a boolean mask (one entry
    per parameter, in the original group order) marking the parameters to
    offload to host.

    Selection is greedy and deterministic:
    - Candidates are visited largest-numel-first, with ties broken by ascending
      original index so the order is total and stable.
    - A candidate is offloaded only if adding its numel keeps the cumulative
      offloaded numel at or below floor(offload_ratio * total_numel).
    - Visiting continues past a skipped candidate so a later, smaller parameter
      can still fit the remaining budget.

    Boundary behavior:
    - offload_ratio == 0.0 → budget 0 → no offload → identical to OffloadPolicy
    - offload_ratio == 1.0 → budget total → every parameter offloaded → identical to CPUOffloadPolicy

    The function is pure and depends only on numels and offload_ratio,
    so every rank computes the identical mask without communication.
    """
    if not (0.0 <= offload_ratio <= 1.0):
        raise ValueError(...)
    n = len(numels)
    mask = [False] * n
    if n == 0:
        return mask
    total = sum(numels)
    if total == 0 or offload_ratio == 0.0:
        return mask
    import math
    budget = math.floor(offload_ratio * total)
    order = sorted(range(n), key=lambda i: (-numels[i], i))
    used = 0
    for i in order:
        if used + numels[i] <= budget:
            mask[i] = True
            used += numels[i]
    return mask

★★★★★★★★★ FSDPParamGroup.__init__ computes per-param offload decisions (lines 230-243):

  offload_to_cpu_per_param: list[bool | None] = [None] * len(params)
  if isinstance(offload_policy, PartialOffloadPolicy):
      numels = [param.numel() for param in params]
      offload_to_cpu_per_param = list(
          _select_partial_offload(numels, offload_policy.offload_ratio)
      )
  # For OffloadPolicy/CPUOffloadPolicy → None → FSDPParam derives from type

  self.fsdp_params = [
      FSDPParam(
          param, module_info, mesh_info, post_forward_mesh_info,
          device, shard_placement_fn, mp_policy, offload_policy,
          offload_to_cpu,                          # ← NEW: per-param decision
      )
      for (param, module_info), offload_to_cpu in zip(
          zip(params, param_module_infos), offload_to_cpu_per_param
      )
  ]

★★★★★★★★★ Updated validation in _validate_cpu_offload_params (lines 1141-1156):

  BEFORE:
    if not isinstance(self.offload_policy, CPUOffloadPolicy):
        return
    fsdp_params_not_on_cpu = [
        fsdp_param
        for fsdp_param in self.fsdp_params
        if fsdp_param.sharded_param.device.type != "cpu"
    ]

  AFTER:
    if not isinstance(
        self.offload_policy, (CPUOffloadPolicy, PartialOffloadPolicy)
    ):
        return
    fsdp_params_not_on_cpu = [
        fsdp_param
        for fsdp_param in self.fsdp_params
        if fsdp_param.offload_to_cpu                   # ← NEW: only check offloaded params
        and fsdp_param.sharded_param.device.type != "cpu"
    ]

  → ★★★★★★★★ Resident params (offload_to_cpu=False) are unconstrained → can be on any device!
  → This is correct: if a param stays on GPU, its sharded_param.device is "cuda" → expected
```

---

## 4. Selection Algorithm: Deterministic Greedy Largest-First Prefix-Stop

```
★★★★★★★★★ _select_partial_offload(numels, offload_ratio) → complete algorithm:

Step 1: Validate and compute budget
  if not (0.0 <= offload_ratio <= 1.0): raise ValueError(...)
  n = len(numels)
  mask = [False] * n
  if n == 0: return mask
  total = sum(numels)
  if total == 0 or offload_ratio == 0.0: return mask
  budget = math.floor(offload_ratio * total)
  → ★★★★★★★★ floor() → never overshoot → integer budget → platform-independent
  → At ratio 1.0: floor(1.0 * total) = total → all params offloaded → exact boundary

Step 2: Sort by (-numel, index) → total order → deterministic
  order = sorted(range(n), key=lambda i: (-numels[i], i))
  → Largest params visited first → ties broken by ascending original index
  → ★★★★★★★★ Every rank has identical numels → identical sort → identical mask → no comm needed!
  → This preserves FSDP2's invariant: gather and reduce participants agree on layout

Step 3: Greedy prefix-stop with continuation
  used = 0
  for i in order:
      if used + numels[i] <= budget:
          mask[i] = True
          used += numels[i]
  → ★★★★★★★★ NOT strict prefix-stop → continues past skipped large params
  → Example: budget=100, params=[80, 30, 30]
  → 80 exceeds budget → SKIP → 30 fits → OFFLOAD (used=30) → 30 fits → OFFLOAD (used=60)
  → Result: 60 numel offloaded (not 80) → but stays within budget!
  → This "continuation past skipped candidates" → fills budget more precisely
  → ★★★★★★★★ This is DIFFERENT from strict prefix-stop (which would stop after first skip)

★★★★★★★★★ Five formal properties (from PR description):

1. Bounded: offloaded numel never exceeds offload_ratio * total
   → floor() guarantees integer budget → cumulative never overshoots
   → ★★★★★★★★ Safe → will never offload MORE than requested

2. Monotonic: higher ratio → superset of offloaded params
   → ratio=0.3 → {A, B, C}
   → ratio=0.4 → {A, B, C, D}
   → No parameter LEAVES the offloaded set as ratio increases
   → ★★★★★★★★ Critical for usability → tuning ratio is safe → no thrashing

3. Deterministic: stable tie-break by parameter index
   → (-numel, index) is a total order → unique sorting
   → Same input → same output → same mask on every rank
   → ★★★★★★★★ No communication → no race conditions → no distributed coordination

4. Boundary-exact: ratio=0.0 → no offload, ratio=1.0 → all offload
   → budget=0 → mask=[False]*n → identical to OffloadPolicy
   → budget=total → mask=[True]*n → identical to CPUOffloadPolicy
   → ★★★★★★★★ Seamless integration → existing policies are boundary cases

5. Largest-first: biggest params offloaded first → maximum memory freed
   → For MoE models: expert params are largest → offloaded first
   → For dense models: projection layers (qkv, out) → largest → offloaded first
   → ★★★★★★★★ Naturally aligns with RTX 4090 needs → free the most memory per decision

★★★★★★★★★ RTX 4090 worked example -- Qwen3-8B dense model:

  Typical param group (one decoder layer):
    → qkv_proj: ~12M elements (largest)
    → out_proj: ~4M elements
    → mlp.up_proj: ~12M elements (largest)
    → mlp.gate_proj: ~4M elements
    → mlp.down_proj: ~4M elements
    → layer_norm: ~0.008M elements (smallest)
    → Total per group: ~36M elements

  offload_ratio=0.5 → budget = floor(0.5 * 36M) = 18M:
    → Sort order: mlp.up_proj(12M, idx=3), qkv_proj(12M, idx=0), out_proj(4M, idx=1),
                   gate_proj(4M, idx=4), down_proj(4M, idx=5), norm(0.008M, idx=6)
    → mlp.up_proj(12M): used+12=12 <= 18 → OFFLOAD (used=12)
    → qkv_proj(12M): 12+12=24 > 18 → SKIP
    → out_proj(4M): 12+4=16 <= 18 → OFFLOAD (used=16)
    → gate_proj(4M): 16+4=20 > 18 → SKIP
    → down_proj(4M): same → SKIP
    → norm(0.008M): 16+0.008=16.008 <= 18 → OFFLOAD (used=16.008)
    → Result: offload mlp.up_proj + out_proj + norm = 16.008M
    → ★★★★★★★★ 44.5% offloaded → close to target 50% → largest params freed!

  ★★★★★★★★ BUT: on dp=1 → the "resident" params still occupy FULL size on GPU!
    → Resident: qkv_proj(12M full) + gate_proj(4M full) + down_proj(4M full) = 20M full on GPU
    → Each "resident shard" = FULL param because dp=1 → no actual sharding
    → See Section 5 for why this breaks on dp=1
```

---

## 5. CRITICAL: Why PartialOffloadPolicy Does NOT Work on RTX 4090 dp=1

```
★★★★★★★★★ THE FUNDAMENTAL ISSUE: FSDP2 shard mechanics on dp=1

FSDP2's fully_shard API shards parameters across the dp_shard dimension of
a DeviceMesh. The shard_size for each parameter is:

  shard_size = ceil(total_param_numel / dp_shard)

When dp_shard = 1 (single GPU):
  → shard_size = ceil(total / 1) = total → THE ENTIRE PARAMETER
  → Each rank holds the FULL parameter as its "shard"
  → This is the identity shard → no actual splitting occurs
  → Parameters are DTensors marked with Shard(0) placement
  → But Shard(0) on mesh_size=1 → mathematically = full tensor on single rank

★★★★★★★★★ What this means for each offload policy on dp=1:

OffloadPolicy (no offload):
  → Each param's "shard" = full param → resident on GPU → full param size on GPU
  → For 8B bf16 model: 8B * 2 bytes = 16 GiB → resident on GPU → with summon, fits
  → For 27B bf16 model: 27B * 2 bytes = 54 GiB → resident on GPU → OOM even with summon!

CPUOffloadPolicy (full offload):
  → Each param's "shard" = full param → but stored on CPU → zero GPU footprint
  → During forward: all-gather → materialize full params → compute → free back to CPU
  → Peak GPU = largest all-gathered unit + activations
  → With per-unit summon (#6512): peak bounded by largest FSDP unit
  → For 27B with per-unit summon: peak ~16.2 GiB → FITS with 7.8 GiB margin
  → ★★★★★★★★ WORKS on dp=1 because ALL shards on CPU → GPU peak = forward-only

PartialOffloadPolicy (fractional offload):
  → Some params: shard = full param → stored on CPU → offloaded → zero GPU footprint
  → Other params: shard = full param → resident on GPU → FULL param on GPU!
  → ★★★★★★★★ Resident params = FULL params on GPU → no sharding savings!
  → If offload_ratio=0.3: 70% of params resident → 70% of model on GPU
  → For 27B: 70% * 54 GiB = 37.8 GiB resident → EXCEEDS 24 GiB → OOM!
  → Even offload_ratio=0.5: 50% resident → 50% * 54 = 27 GiB → STILL OOM!

★★★★★★★★★ THE MATH: why ANY partial ratio fails on dp=1 for large models:

  For a model of P billion parameters in bf16:
    → Total model size = P * 2 GiB (bf16 weights alone)
    → Resident fraction = (1 - offload_ratio) * P * 2 GiB
    → Must be <= 24 GiB → (1 - ratio) * P * 2 <= 24
    → (1 - ratio) <= 24 / (P * 2) = 12 / P

    For P=8:  (1 - ratio) <= 12/8 = 1.5 → ratio >= -0.5 → ANY ratio works
              → 8B model fits with no offload → partial offload not needed
    For P=14: (1 - ratio) <= 12/14 = 0.86 → ratio >= 0.14 → only 14% need offload
              → BUT: resident = (1-0.14)*28 = 24.08 → barely fits? NO!
              → Peak = resident + forward gather of offloaded + activations → > 24!
    For P=27: (1 - ratio) <= 12/27 = 0.44 → ratio >= 0.56 → MUST offload >56%
              → resident 44% = 23.8 GiB → barely fits? NO!
              → Peak during forward = resident + offloaded gathered = P*2 = 54 GiB → OOM!

★★★★★★★★★ Forward pass mechanics on dp=1 with PartialOffloadPolicy:

  Step 1: All-gather for forward → materialize ALL params on GPU (including offloaded)
    → Offloaded params: CPU → GPU (prefetch) → temporary on GPU during forward
    → Resident params: already on GPU → zero copy cost
    → ALL params must be on GPU during forward → peak = full model + activations!

  Step 2: After forward → free offloaded params back to CPU → resident params stay
    → Peak during forward = ALL params on GPU simultaneously → same as OffloadPolicy!
    → ★★★★★★★★ Peak with ANY partial offload = same as NO offload during forward!
    → The "savings" only apply BETWEEN forward/backward → not during computation

  Step 3: All-gather for backward → same as forward → peak = full model + grads + activations
    → ★★★★★★★★ Peak during backward >= forward peak → OOM guaranteed for large models!

★★★★★★★★★ Why CPUOffloadPolicy works on dp=1:
  → ALL shards on CPU → ALL params freed from GPU after use
  → Peak = largest single unit during forward → bounded by FSDP unit size
  → With per-unit summon (#6512): peak = params of largest layer + activations
  → For 27B per-unit: ~1.7B * 2 = 3.4 GiB per layer + activations ~12 GiB → 16.2 GiB total
  → ★★★★★★★★ CPUOffloadPolicy peak = forward per-unit → FITS → resident footprint = 0!

★★★★★★★★★ CORRECTED RTX 4090 conclusion:
  → PartialOffloadPolicy DOES NOT help RTX 4090 dp=1 for any model >8B
  → The "slightly over budget" case doesn't exist on dp=1:
    → If model fits with no offload → use OffloadPolicy → no need for partial
    → If model doesn't fit with no offload → MUST fully offload → peak during forward = full model anyway
  → CPUOffloadPolicy remains ONLY viable path for dp=1 large models
  → ★★★★★★★★ PartialOffloadPolicy is a multi-GPU (dp>=2) optimization ONLY!
```

---

## 6. When PartialOffloadPolicy ACTUALLY Helps: Multi-GPU (dp>=2)

```
★★★★★★★★★ Multi-GPU sharding: shard_size = total / dp → actual memory savings!

dp=2 (2x RTX 4090):
  → shard_size = total / 2 → each GPU holds HALF of each parameter
  → Resident fraction: (1 - offload_ratio) params on GPU
  → Resident per GPU = (1 - ratio) * P * 2 / 2 GiB = (1 - ratio) * P GiB
  → For P=27, ratio=0.3: resident = 0.7 * 27 = 18.9 GiB → FITS on each GPU!
  → Offloaded: 0.3 * 27 * 2 / 2 = 8.1 GiB on CPU per GPU → reduced copy

dp=4 (4x RTX 4090):
  → Resident per GPU = (1 - ratio) * P * 2 / 4 GiB = (1 - ratio) * P / 2 GiB
  → For P=27, ratio=0.3: resident = 0.7 * 27/2 = 9.45 GiB → comfortable!
  → Only 30% of shards on CPU → 70% zero-copy → much faster forward

★★★★★★★★★ The key difference: dp>=2 → shards are ACTUAL fractions:
  → dp=2: shard = half of param → resident shard = half → real memory savings
  → dp=1: shard = full param → resident shard = full → NO memory savings
  → ★★★★★★★★ PartialOffloadPolicy requires dp>=2 → shards must be actual fractions!

★★★★★★★★★ Multi-GPU benefit quantification (2x RTX 4090, Qwen3.5-27B):

  Full offload (CPUOffloadPolicy, dp=2):
    → Peak per GPU = largest unit / 2 + activations = ~8 GiB + ~12 GiB = ~20 GiB
    → ALL shards on CPU → host-device copy for ALL params → max latency
    → Works → but pays copy tax on every shard every forward

  Partial offload (ratio=0.3, dp=2):
    → Resident per GPU = 18.9 GiB → 70% zero-copy → fast forward for 70%
    → Offloaded per GPU = 8.1 GiB on CPU → 30% pay host-device copy → less latency
    → Peak during forward = resident + all-gathered offloaded
    → But: all-gathered offloaded = temporarily on GPU → freed after forward
    → Peak = resident + largest offloaded unit + activations ≈ 18.9 + ~4 + ~3 = ~26 GiB
    → ★★★★★★★★ Tight fit → need careful unit sizing → but FASTER than full offload!

★★★★★★★★★ Optimal ratio for dp=2 RTX 4090:
  → Need: resident_per_gpu + forward_peak <= 24 GiB
  → resident_per_gpu = (1 - ratio) * P * 2 / dp GiB
  → forward_peak = resident + largest_unit_all_gathered + activations
  → For 27B, dp=2: resident = (1-ratio)*27 GiB
  → Need resident <= ~19 GiB (leaving 5 GiB for forward gather + activations)
  → (1-ratio)*27 <= 19 → ratio >= (27-19)/27 = 0.296 → ratio >= 0.3
  → ★★★★★★★★ ratio=0.3 is optimal for 27B on 2x RTX 4090 → frees ~30% → 70% resident

★★★★★★★★★ Why per-unit summon (#6512) synergizes with PartialOffloadPolicy on dp>=2:
  → Per-unit summon: only materialize needed params → peak bounded by unit
  → CPUOffloadPolicy: ALL params on CPU → summon from CPU → peak = unit/dp + activations
  → PartialOffloadPolicy: resident params on GPU + offloaded on CPU
  → ★★★★★★★★ Combined: only all-gather offloaded params per unit → resident params zero-copy
  → Forward peak per unit = resident_shard + all_gathered_offloaded + activations → bounded!
```

---

## 7. CPUOffloadPolicy: The ONLY Viable dp=1 Path for RTX 4090

```
★★★★★★★★★ CPUOffloadPolicy lifecycle on dp=1:

  Class definition (from _fsdp_api.py, CURRENT main branch):
  @dataclass
  class CPUOffloadPolicy(OffloadPolicy):
      pin_memory: bool = True    # ★★★★★★★★ DEFAULT IS TRUE (not False!)
  → Docstring: "Set this to False if you have insufficient CPU memory. (Default: True)"
  → ★★★★★★★★ NOTE: CPUOffloadPolicy is @dataclass (NOT frozen=True) → MUTABLE
  → Contrast: PartialOffloadPolicy is @dataclass(frozen=True) → IMMUTABLE → hashable → safer
  → ★★★★★★★★ PREVIOUS note had WRONG default (False) → CORRECTED to True

★★★★★★★★★ Lifecycle for each FSDPParam with CPUOffloadPolicy:

  1. Initialization:
    → sharded_param allocated on CPU → pinned if pin_memory=True
    → offload_to_cpu = True → sharded_param stays on CPU between steps

  2. Pre-forward (all-gather):
    → Prefetch offloaded shards from CPU to GPU (via pinned DMA if pin_memory)
    → All-gather constructs full param on GPU → compute proceeds

  3. Forward pass:
    → Full params on GPU → normal computation → activations computed

  4. Post-forward (free):
    → Free full (unsharded) params from GPU → shards return to CPU
    → ★★★★★★★★ Peak = largest all-gathered unit + activations → bounded!

  5. Pre-backward (all-gather again):
    → Prefetch shards from CPU to GPU → same as pre-forward
    → Compute gradients on GPU → grads stored on GPU temporarily

  6. Post-backward (reduce-scatter + free):
    → Reduce-scatter gradients → each rank gets its gradient shard
    → Gradient shards moved to CPU (if offloaded) → optimizer step on CPU
    → Free full params from GPU → back to baseline

★★★★★★★★★ RTX 4090 memory profile with CPUOffloadPolicy (Qwen3-8B LoRA, per-unit summon):

  Baseline (between steps):
    → GPU: activations from previous step + small buffers ≈ 2-4 GiB
    → CPU: ALL sharded params (8B * 2 = 16 GiB) + ALL optimizer states

  Peak (during forward, per-unit summon #6512):
    → GPU: one decoder layer full params (~1B * 2 = 2 GiB) + activations (~14 GiB)
    → Total peak ≈ 16 GiB → 8 GiB margin → FITS comfortably

  Peak (during backward, per-unit summon):
    → GPU: one layer full params + gradients + activations ≈ 16-18 GiB
    → Still FITS with margin → ★★★★★★★★ CPUOffloadPolicy + per-unit summon = RTX 4090 proven path!

★★★★★★★★★ pin_memory=True vs False — RTX 4090 impact:

  pin_memory=True (DEFAULT for CPUOffloadPolicy):
    → Uses CUDA pinned memory → DMA transfer → faster CPU→GPU copy
    → Pinned memory is non-pageable → locked in CPU RAM → faster but uses more RAM
    → Typically 30-50% faster prefetch → reduces step time
    → ★★★★★★★★ MUST use pin_memory=True for RTX 4090 → host-device copy is the bottleneck
    → ★★★★★★★★ NOTE: default is True → no need to explicitly set → already optimal!

  pin_memory=False:
    → Uses pageable memory → slower CPU→GPU transfer
    → More flexible CPU RAM usage → but slower training
    → Only use if CPU RAM is very constrained (<32 GiB system RAM)
    → Must explicitly set pin_memory=False to use this → most users should NOT
```

---

## 8. DeepSpeed ZeRO-2 CPU_Adam vs FSDP2 CPUOffloadPolicy -- The RTX 4090 Comparison

```
★★★★★★★★★ Both offload optimizer states to CPU → but different mechanisms:

DeepSpeed ZeRO-2 + CPU_Adam:
  → ZeRO-2 partitions optimizer states across ranks → dp=1 → partition = full model
  → CPU_Adam: specialized CPU-side Adam optimizer kernel → optimized for CPU compute
  → All optimizer state (momentum, variance) on CPU → 2x model size on CPU
  → Adam update computed ON CPU → reduces GPU-CPU round trips
  → ★★★★★★★★ CPU_Adam is a CUSTOM kernel → not standard PyTorch → faster on CPU
  → AVX512: 512-bit SIMD → 16 float32 per cycle → optimized → 5-7x faster than naive

FSDP2 CPUOffloadPolicy:
  → FSDP2 shards params across ranks → dp=1 → shard = full param (identity)
  → Standard PyTorch optimizer → step happens wherever params are
  → Params on CPU → optimizer step on CPU → standard torch.optim.Adam
  → No custom CPU optimizer → uses vanilla PyTorch CPU operations
  → ★★★★★★★★ Slower CPU optimizer → but no custom dependency → more portable

★★★★★★★★★ Performance comparison (single GPU, 8B model):

  DeepSpeed ZeRO-2 + CPU_Adam:
    → ~15-25% higher throughput vs FSDP2 CPUOffloadPolicy on single GPU
    → CPU_Adam kernel optimized → fused CPU operations → less overhead
    → Trade-off: requires DeepSpeed config JSON → custom optimizer → less portable

  FSDP2 CPUOffloadPolicy:
    → ~15-25% slower throughput → standard PyTorch CPU ops → not fused
    → Simpler config → Pythonic → CPUOffloadPolicy(pin_memory=True)
    → Works with ANY PyTorch optimizer → no custom dependency
    → Better integration: torch.compile, torchtitan, verl, torchft

★★★★★★★★★ Memory comparison (single GPU, dp=1):

  Both achieve similar memory savings:
    → GPU peak = forward activations + largest unit ≈ 16-18 GiB (8B LoRA)
    → CPU holds: params + optimizer states ≈ 48 GiB (8B model * 2 params + 2 * 8B optimizer)
    → ★★★★★★★★ Memory profile is comparable → difference is in THROUGHPUT not CAPACITY

★★★★★★★★★ RTX 4090 decision: ZeRO-2 vs FSDP2 for single GPU:

  Choose DeepSpeed ZeRO-2 + CPU_Adam IF:
    → Maximum throughput is priority → 15-25% faster step time
    → Using DeepSpeed ecosystem → compatible with AutoEP MoE (#7938)
    → Using DeepSpeed config system → already have DeepSpeed configs
    → Can tolerate custom optimizer dependency → CPU_Adam is non-standard

  Choose FSDP2 CPUOffloadPolicy IF:
    → Integration with verl CPPO+bypass → #1 RTX 4090 GRPO path
    → PyTorch ecosystem compatibility → torch.compile, torchtitan
    → Simpler configuration → no JSON config → Pythonic API
    → Per-unit summon (#6512) → 10x memory reduction → unique to FSDP2
    → ★★★★★★★★ verl CPPO+bypass + FSDP2 CPUOffloadPolicy = RTX 4090 #1 BEST PATH

★★★★★★★★★ Current RTX 4090 ranking:
  → verl CPPO+bypass + FSDP2 CPUOffloadPolicy → #1 → best GRPO path
  → DeepSpeed ZeRO-2 + CPU_Adam → #2.5 → faster per-step but no CPPO
  → verl GRPO+bypass + FSDP2 → #2 → without CPPO
  → Megatron core → #4 → no single GPU CPU offload yet
```

---

## 9. Complete RTX 4090 FSDP2 Decision Tree

```
★★★★★★★★★ RTX 4090 (24 GiB VRAM, SM89, single GPU, dp=1):

Decision 1: Does model fit with NO offload? (peak < 24 GiB)
  → Check: model_size * 2 (bf16) + activations + LoRA params < 24
  → Qwen3-1.7B: ~3.4 GiB + ~8 GiB activations = ~11 GiB → YES → OffloadPolicy
  → Qwen3-8B LoRA (with per-unit summon): ~16.2 GiB → YES → OffloadPolicy
  → BUT: without per-unit summon → ~64 GiB → NO → go to Decision 2
  → ★★★★★★★★ Per-unit summon (#6512) is CRITICAL → without it, even 8B OOMs!

  If YES → Use OffloadPolicy (no offload) → fastest forward → zero copy
  If NO → Go to Decision 2

Decision 2: Does model fit with FULL offload? (peak < 24 GiB)
  → Check: largest_unit + activations < 24
  → With per-unit summon: peak = largest_decoder_layer + activations
  → Qwen3-8B LoRA: ~2 GiB + ~14 GiB = ~16 GiB → YES → CPUOffloadPolicy
  → Qwen3.5-27B LoRA (per-unit): ~3 GiB + ~13 GiB = ~16 GiB → YES → CPUOffloadPolicy
  → Qwen3-MoE (per-unit): ~3 GiB + ~17 GiB = ~20 GiB → YES → CPUOffloadPolicy
  → ★★★★★★★★ ALL common RTX 4090 models FIT with CPUOffloadPolicy + per-unit summon!

  If YES → Use CPUOffloadPolicy(pin_memory=True) → proven → stable → works NOW
  If NO → Go to Decision 3

Decision 3: Model doesn't fit even with full offload (dp=1)?
  → This means: activations alone > 24 GiB → even with ALL params offloaded
  → Example: >30B dense model with large batch → activations exceed 24 GiB
  → Options:
    → a) Reduce batch size → fewer activations → may fit
    → b) Gradient checkpointing → reduce activations by ~70% → may fit
    → c) Multi-GPU → dp>=2 → shard activations → fits on each GPU
    → ★★★★★★★★ For RTX 4090: a+b usually sufficient → c for very large models

★★★★★★★★★ FUTURE: When #187620 merges (PartialOffloadPolicy available):

  Decision 4: Can PartialOffloadPolicy help on dp>=2?
  → Only for multi-GPU setups → dp>=2 → shards are actual fractions
  → dp=2, ratio=0.3: resident per GPU = (0.7 * P * 2) / dp GiB
  → For 27B: 0.7 * 54 / 2 = 18.9 GiB → FITS → faster forward (70% zero-copy)
  → ★★★★★★★★ PartialOffloadPolicy = multi-GPU copy bandwidth optimization
  → NOT applicable to dp=1 → see Section 5 for full explanation

★★★★★★★★★ Decision tree summary table:

  | Scenario             | dp | ratio | Policy               | RTX 4090?   |
  |----------------------|----|-------|----------------------|-------------|
  | <8B dense, summon   | 1  | 0.0   | OffloadPolicy        | YES (best)  |
  | 8-27B LoRA, summon  | 1  | 1.0   | CPUOffloadPolicy     | YES (#1)    |
  | 27B dense, no summon| 1  | 1.0   | CPUOffloadPolicy     | MAYBE(tight)|
  | >30B dense          | 1  | any   | NONE                 | NO (OOM)    |
  | 27B LoRA, dp=2      | 2  | 0.3   | PartialOffloadPolicy | YES (future)|
  | 27B LoRA, dp=2      | 2  | 1.0   | CPUOffloadPolicy     | YES (now)   |
  | 27B LoRA, dp=4      | 4  | 0.3   | PartialOffloadPolicy | YES (best)  |

★★★★★★★★★ MUST DO for RTX 4090:
  → 1. Use per-unit summon (#6512) → peak bounded by largest FSDP unit
  → 2. Use CPUOffloadPolicy(pin_memory=True) → ALL params on CPU → proven path
  → 3. Use verl CPPO+bypass → #1 RTX 4090 GRPO path → best throughput
  → 4. LoRA rank <= 32 → rank=64 breaks EOS in vLLM (#6782)
  → 5. overlap_comm=False → overlap_comm+compile = NaN (#8061)

★★★★★★★★★ MUST NOT for RTX 4090:
  → 1. PartialOffloadPolicy on dp=1 → resident shard = full param → OOM
  → 2. ZeRO-3 on single GPU → pure overhead → MUST use ZeRO-2
  → 3. Muon optimizer → 6 blockers → NOT viable → use AdamW/CPU_Adam
  → 4. DeepSpeed overlap_comm=True → NaN on single GPU (#8061)
  → 5. gradient_clipping default 0 → MUST set to 1.0 (#8068)
```

---

## 10. PR Status and Review Analysis

```
★★★★★★★★★ PR #187620 status (as of 2026-06-18):

  State: OPEN (DRAFT) → opened 2026-06-17 → author: joemunene-by (Joe Munene)
  Stats: +153/-7 lines, 3 files changed, 1 commit
  Commit: 559722dfbc14c33d203f18cb561d9ec47653efeb
  Labels: release notes: distributed (fsdp), ciflow/inductor, ciflow/torchtitan
  CI: NOT yet triggered → 5 workflows awaiting approval
    → Auto Request Review, BC Lint, docs-build, Lint, pull → all blocked
    → CIFlow labels (inductor, torchtitan) added but pending approval
  Reviews: ZERO review comments → no maintainer feedback yet
  Merge: NOT merged → NOT mergeable → DRAFT for API direction
  CLA: Signed (Linux Foundation EasyCLA verified)

★★★★★★★★★ Comment analysis:
  → Only bot comments so far:
    → pytorch-bot: Dr. CI status (5 workflows awaiting approval)
    → linux-foundation-easycla: CLA verification (signed)
    → pytorch-bot: ciflow/inductor and ciflow/torchtitan pending
  → NO human comments → NO maintainer reviews → NO community feedback
  → Scoping issue #187615 also has ZERO maintainer comments → no direction yet

★★★★★★★★★ What the PR author is waiting for:

  From PR description:
  "Opening as a draft to get maintainer direction on the API (per #187615)
   before investing in the full multi-rank test matrix."

  → The selector has standalone unit tests: 26/26 passing locally
  → BUT: test files NOT in the current commit diff → only 3 source files!
    → test_partial_offload_selector.py → mentioned but NOT committed yet
    → Distributed integration tests → mentioned but NOT committed yet
    → ★★★★★★★★ Tests are PLANNED → not yet pushed → waiting for API direction first
  → Distributed integration tests NOT yet run → needs multi-rank hardware
  → Full CI NOT triggered → awaiting maintainer approval
  → ★★★★★★★★ API direction must be resolved before investing in full test matrix!

★★★★★★★★★ Expected review timeline:
  → DRAFT opened June 17 → very new → likely weeks before review
  → FSDP2 maintainers (awgu, roxannefernandez) → need to weigh in on 3 RFC questions
  → Key question: numel-fraction vs byte-budget → may require design iteration
  → ★★★★★★★★ Timeline: likely 1-3 months before merge → API direction first → then full tests

★★★★★★★★★ Risks and potential issues:
  → 1. API knob debate: numel vs bytes → may change offload_ratio semantics
    → numel-based: simpler → but doesn't account for dtype differences
    → byte-based: more precise for mixed precision → but more complex API
    → ★★★★★★★★ For RTX 4090: numel is sufficient → but general case needs bytes
  → 2. Selector placement: policy vs param_group vs init → current choice seems right
    → Group-level: natural → same numels on every rank → identical selection → no comm
  → 3. Prefix-stop vs exact-fill debate → PR chose prefix-stop → monotonicity strong argument
    → Exact-fill could offload MORE within budget → but sacrifices monotonicity
    → ★★★★★★★★ Monotonicity is more valuable than exact filling → ratio tuning is safe
  → 4. Mixed precision: offload_ratio based on numel → doesn't account for dtype differences
    → bf16 shard = 2 bytes/element → fp32 optimizer copy = 4 bytes/element
    → For RTX 4090: all params bf16 → numel-based is fine → but general case?
  → 5. Gradient offload: PR offloads params → gradients follow params → correct?
    → CPUOffloadPolicy: params AND grads AND optimizer on CPU → all three
    → PartialOffloadPolicy: offloaded params → their grads on CPU → resident params → grads on GPU?
    → ★★★★★★★★ Mixed device gradients → reduce-scatter needs careful handling!
    → Each param's reduce-scatter destination depends on its offload status
    → Offloaded params: reduce-scatter result on CPU → optimizer on CPU
    → Resident params: reduce-scatter result on GPU → optimizer on GPU (or also CPU?)
    → ★★★★★★★★ This is a SUBTLE distributed correctness issue → needs thorough testing!
  → 6. Optimizer state split: if some params on CPU, some on GPU → optimizer splits
    → CPU params: CPU optimizer states → CPU step → Adam on CPU
    → GPU params: GPU optimizer states → GPU step → Adam on GPU
    → Mixed optimizer execution → correctness and performance implications
```

---

## 11. Related Issues and Cross-Framework Connections

```
★★★★★★★★★ Directly related PyTorch issues:

  #187615 (OPEN, 0 comments) → Scoping issue for this PR
    → Same author (joemunene-by) → same day (2026-06-17)
    → 3 open API questions → zero maintainer feedback
    → This is the prerequisite for PR direction

  #114299 (OPEN, awgu, 2023-11-21) → Per-Parameter-Sharding FSDP RFC
    → The AXIS this PR is on → per-parameter control in FSDP
    → Key direction: flexible fp8, frozen+non-frozen mix, sharded state dicts
    → ★★★★★★★★ PartialOffloadPolicy is a concrete instance of the #114299 direction
    → 2.5+ years open → FSDP2 is gradually implementing this direction

  #174960 (OPEN, niyunsheng, 2026-02-13) → Activation offload with async prefetch
    → Adjacent axis (ACTIVATION offload, not parameter offload)
    → Proposes activation offload WITHIN CPUOffloadPolicy → reuses prefetch infra
    → Together: parameter offload + activation offload → complete memory control
    → ★★★★★★★★ PartialOffloadPolicy + activation offload → full offload control surface

★★★★★★★★★ FSDP1-related offload issues (historical context):

  #130530 (OPEN) → "Fail to offload FSDP model weights and optimizer states
    without using CPUOffload(offload_params=True)"
    → User wants partial optimizer offload without param offload → similar desire
    → FSDP1 limitation: CPUOffload forces BOTH params AND optimizer to CPU
    → ★★★★★★★★ Same binary choice problem in FSDP1 → #187620 addresses it for FSDP2

  #91165 (OPEN) → "FSDP with CPU offload consumes 1.65X more GPU memory when
    training models with most of the params frozen"
    → LoRA training: most params frozen → CPU offload overhead on frozen params
    → 1.65x memory overhead → frozen params shouldn't need offload overhead
    → ★★★★★★★★ PartialOffloadPolicy could solve this → only offload trainable params
    → ratio based on trainable fraction → LoRA: ~0.2% trainable → offload those only

★★★★★★★★★ Cross-framework connections:

  DeepSpeed ZeRO-2 + CPU_Adam:
    → Already provides CPU optimizer offload → mature → production-tested
    → CPU_Adam custom kernel → 5-7x faster than vanilla CPU Adam
    → No fractional offload → all-or-nothing → same binary choice
    → ★★★★★★★★ If DeepSpeed adds fractional offload → would be equivalent but faster

  verl #6512 (MERGED June 18) → per-unit LoRA summon:
    → 10x memory reduction → 60→6-8 GiB peak for LoRA weight sync
    → Key pattern: summon → compute → release → same lifecycle as FSDP offload
    → CPUOffloadPolicy + per-unit summon = current #1 RTX 4090 GRPO path
    → ★★★★★★★★ Per-unit summon makes PartialOffloadPolicy viable on dp>=2!
    → Without per-unit summon: full model on GPU during forward → partial offload useless

  verl #6699 (MERGED June 18) → detach memory fix:
    → 4x reduction → FSDP fixed → 3 other backends UNFIXED
    → MUST use FSDP backend → same requirement as CPUOffloadPolicy

  verl #6794 (OPEN) → delta weight sync:
    → ~100x payload reduction → SGLang-only → LoRA deferred
    → 4 review issues → 2 CRITICAL → still evolving
    → ★★★★★★★★ delta sync + partial offload → further bandwidth optimization

  Megatron #5387 (APPROVED, CI triggered) → MFSDPv2 fully_shard:
    → Megatron's own FSDP → DBuffer primitives → per-module shard
    → Currently: no partial offload → same binary choice
    → ★★★★★★★★ Table comparison:

  | Feature              | PyTorch FSDP2        | DeepSpeed ZeRO-2    | Megatron MFSDPv2 |
  |----------------------|----------------------|---------------------|------------------|
  | CPU param offload    | CPUOffloadPolicy     | cpu_offload=True    | None (future)    |
  | Fractional offload   | #187620 (DRAFT)      | None                | None             |
  | CPU optimizer        | CPUOffloadPolicy     | CPU_Adam (custom)   | None             |
  | Per-unit summon      | #6512 (verl)         | None                | DBuffer lifecycle|
  | Activation offload   | #174960 (OPEN)       | None                | None             |
```

---

## 12. Integration Path for verl and RTX 4090 Ecosystem

```
★★★★★★★★★ verl integration requirements:

  Current verl FSDP2 config:
    → param_offload: bool → True → CPUOffloadPolicy → False → OffloadPolicy
    → Binary choice → same gap as PyTorch → no fractional option

  Future verl config (when #187620 merges):
    → param_offload_ratio: float → 0.0 = OffloadPolicy → 1.0 = CPUOffloadPolicy
    → 0.3 = PartialOffloadPolicy(offload_ratio=0.3) → fraction offload
    → ★★★★★★★★ Simple config addition → one new field → backward compatible

  verl engine code change:
    → In FSDPEngine or similar → construct offload policy from config
    → Current: if param_offload: CPUOffloadPolicy() else: OffloadPolicy()
    → Future: if param_offload_ratio == 0: OffloadPolicy()
             elif param_offload_ratio == 1: CPUOffloadPolicy(pin_memory=True)
             else: PartialOffloadPolicy(offload_ratio=param_offload_ratio, pin_memory=True)

★★★★★★★★★ RTX 4090 verl integration timeline:
  → Phase 1 (NOW): CPUOffloadPolicy → works → stable → USE THIS
  → Phase 2 (1-3 months): #187620 merges → PartialOffloadPolicy available
  → Phase 3 (3-6 months): verl adds offload_ratio config → partial offload on dp>=2
  → ★★★★★★★★ For dp=1 RTX 4090: Phase 1 is permanent → no Phase 2/3 benefit!

★★★★★★★★★ Synergy with verl per-unit summon (#6512):
  → Per-unit summon: only materialize needed LoRA params → peak bounded
  → CPUOffloadPolicy: ALL params on CPU → summon from CPU → peak = unit size + activations
  → PartialOffloadPolicy: resident params on GPU → offloaded on CPU → summon hybrid
  → ★★★★★★★★ Combined on dp>=2: summon + partial offload → maximum efficiency
  → BUT on dp=1: summon + full offload = current #1 path → partial adds NOTHING

★★★★★★★★★ LoRA + PartialOffloadPolicy interaction on dp>=2:
  → LoRA params: small → rank=32 → ~0.2% of model → typically NOT offloaded by selector
  → Largest-first selector → offloads big projection layers → keeps LoRA params resident
  → ★★★★★★★★ LoRA params on GPU → fast forward → no copy overhead for trainable params
  → Optimizer states for LoRA: tiny → even on GPU they're small → no issue
  → This is actually IDEAL for LoRA training → offload big frozen parts → keep trainable resident!
```

---

## Key Findings Summary

★★★★★★★★★ #187620: PartialOffloadPolicy → fractional CPU offload → +153/-7, 3 files, DRAFT, 0 reviews
★★★★★★★★★ CRITICAL: PartialOffloadPolicy DOES NOT work on dp=1 → shard=identity → resident=full param → OOM
★★★★★★★★★ Forward pass peak on dp=1: ALL params on GPU regardless of ratio → same as no offload
★★★★★★★★★ CPUOffloadPolicy (full offload) → ONLY viable path for RTX 4090 single GPU (dp=1)
★★★★★★★★★ CPUOffloadPolicy.pin_memory DEFAULT IS TRUE (not False!) → already optimal for RTX 4090
★★★★★★★★★ PartialOffloadPolicy helps dp>=2 ONLY → shards = real fractions → resident per GPU = fraction
★★★★★★★★★ Greedy largest-first prefix-stop selector → 5 properties: bounded/monotonic/deterministic/exact/largest-first
★★★★★★★★★ Selector continues past skipped large params → fills budget more precisely (NOT strict prefix-stop)
★★★★★★★★★ RFC #187615: 3 open questions → numel vs bytes, placement, prefix-stop vs exact-fill → 0 maintainer answers
★★★★★★★★★ PartialOffloadPolicy is @dataclass(frozen=True) → OffloadPolicy/CPUOffloadPolicy are @dataclass (mutable)
★★★★★★★★★ Test files NOT in current commit → only 3 source files → 26/26 local unit tests mentioned but not pushed
★★★★★★★★★ Gradient offload subtlety: mixed-device gradients → reduce-scatter needs careful handling per param
★★★★★★★★★ Optimizer state split: some params CPU optimizer, some GPU optimizer → mixed execution
★★★★★★★★★ #130530/#91165: FSDP1 offload issues → same binary choice problem → #187620 addresses for FSDP2
★★★★★★★★★ #91165 specifically: LoRA training → 1.65x memory overhead → PartialOffloadPolicy could solve!
★★★★★★★★★ DeepSpeed CPU_Adam 15-25% faster → but verl CPPO outweighs → FSDP2 #1 for GRPO
★★★★★★★★★ RTX 4090: CPUOffloadPolicy(pin_memory=True) + per-unit summon + CPPO+bypass = #1 BEST
★★★★★★★★★ Multi-GPU future: dp=2 → PartialOffloadPolicy(ratio=0.3) → 70% resident → faster + less copy
★★★★★★★★★ LoRA synergy: largest-first selector → offloads big frozen parts → keeps LoRA params resident → ideal!

---

## References

- PyTorch #187620: https://github.com/pytorch/pytorch/pull/187620
- PyTorch RFC #187615: https://github.com/pytorch/pytorch/issues/187615 (scoping issue, 0 maintainer comments)
- PyTorch #114299: https://github.com/pytorch/pytorch/issues/114299 (per-parameter FSDP RFC, awgu, OPEN since 2023-11-21)
- PyTorch #174960: https://github.com/pytorch/pytorch/issues/174960 (activation offload, niyunsheng, OPEN since 2026-02-13)
- PyTorch #130530: https://github.com/pytorch/pytorch/issues/130530 (FSDP offload params, OPEN)
- PyTorch #91165: https://github.com/pytorch/pytorch/issues/91165 (FSDP CPU offload 1.65x memory overhead with frozen params, OPEN)
- PyTorch FSDP2 API source: torch/distributed/fsdp/_fully_shard/_fsdp_api.py
- PyTorch FSDP2 param source: torch/distributed/fsdp/_fully_shard/_fsdp_param.py
- PyTorch FSDP2 param group source: torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py
- verl #6512: per-unit LoRA summon → 10x memory reduction → MERGED June 18
- verl #6699: detach memory fix → 4x reduction → MERGED June 18 → UNFIXED in 3 backends
- verl #6794: delta weight sync → ~100x payload reduction → OPEN → 2 CRITICAL review issues
- verl #6782: LoRA rank>64 breaks EOS in vLLM rollout
- DeepSpeed #8061: overlap_comm+compile=NaN → MUST overlap_comm=False on single GPU
- DeepSpeed #8068: gradient_clipping default 0→1.0 → MUST set 1.0 for GRPO
- DeepSpeed ZeRO-2 source: notebook/projects/deepspeed-zero-single-gpu-source-reading.md
- Megatron #5387: MFSDPv2 fully_shard → APPROVED → CI triggered
- RTX 4090 config: tools/rtx4090_grpo_config_reference.py
- RTX 4090 decision: tools/fsdp2_vs_zero2_decision_guide.py
- FSDP2 single GPU analysis: notebook/projects/pytorch-fsdp2-single-gpu-analysis.md
- FSDP2 vs ZeRO-2: notebook/projects/zero3-vs-fsdp2-system-comparison.md
