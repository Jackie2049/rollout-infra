# DeepSpeed #8064 AutoEP+AutoTP Parallel Folding — Source-Level Reading

> 2026-06-18 | PR #8064 (OPEN) | 2690 additions | Authors: DeepSpeed team
> ★★★★★★★★ TP for dense + EP for MoE on same GPU set → folding enables multi-GPU MoE+dense
> ★★★★★★★★ RTX 4090: TP=1+EP=1 = single GPU → folding is preparation for future multi-GPU
> ★★★★★★★★ Mode-aware gradient reduction → router/gate params handled differently per parallelism mode

---

## 1. Architecture: Parallel Folding

```
★★★★★★★★★ Core concept: two independent partitionings of same rank set:

Dense / attention / shared-expert params: stage_size = tp * dp
Routed-expert params: stage_size = ep * etp * edp

Invariant: tp * dp == ep * etp * edp == stage_size
  → dp and edp are DERIVED, never user-configured
  → Cannot break from config → always valid

★★★★★★★★★ Configuration (no new section):
  {
    "tensor_parallel": { "autotp_size": 2 },
    "expert_parallel": {
      "enabled": true,
      "autoep_size": 4,
      "expert_tensor_parallel_size": 1  // MUST be 1 currently
    }
  }

★★★★★★★★★ RTX 4090 configuration:
  → tp=1, ep=1, dp=1, edp=1, etp=1 → stage_size=1
  → ALL on single GPU → folding = identity → no change
  → Folding is preparation for future multi-GPU setups
  → Current: AutoEP+EP=1+ZeRO-2 = single GPU MoE → already viable!
```

---

## 2. Key Implementation Details

```
★★★★★★★★★ New files in #8064:

1. deepspeed/module_inject/auto_ep_folding.py (NEW, 261 lines)
   → Folded process-group derivation
   → Generalized expert/data-parallel group creation
   → mp_mode: TP-strided vs SP-consecutive ordering

2. deepspeed/moe/ep_tp_dispatch.py (NEW, 310 lines)
   → Route-full / partition-dispatch path for folded MoE
   → AutoTP skipping AutoEP subtrees → dense params skip EP path
   → Dispatch: dense params → TP reduce, expert params → EP dispatch

★★★★★★★★★ Mode-aware TP-replicated gradient reduction for router/gate params:
  → When parallelism mode partitions tokens → gradient SUMMED (scale 1.0)
  → When tokens are replicated → gradient AVERAGED (standard TP gradient)
  → Matches standard sequence-parallel / tensor-parallel gradient semantics
  → Router/gate gradient parity verified against non-folded baseline: ~1e-7 match

★★★★★★★★★ Per-parameter-family ZeRO checkpoint metadata:
  → routed-expert vs dense/router/shared placement → separate metadata
  → Folded ZeRO-1/2 optimizer-state handling → per-family state management
  → Checkpoint save/load works with multi-rank folding configurations

★★★★★★★★★ expert_tensor_parallel_size > 1 RESERVED for follow-up:
  → Currently MUST be 1 → fail-fast validation
  → Expert-internal TP = future work → MoE expert sharding within single expert
  → This would enable: expert_params split across GPUs → larger experts viable
```

---

## 3. Validation and Correctness

```
★★★★★★★★★ Correctness tests:

- Router/gate gradient parity: TP2 × EP4 (8-GPU) → folded gradient matches baseline to ~1e-7
- Full unit test suite (aws-torch-latest-full) passes on H100 GPUs
- New folding unit tests: config, group layout, dispatch, runtime, gradient parity, checkpoint save/load
- Multi-rank cases gated for GPU runners → CI validation on real hardware

★★★★★★★★★ RTX 4090 validation needed:
  → TP=1+EP=1 = folding identity → should pass trivially
  → But: should verify that folding doesn't break EP=1 path (the RTX 4090 critical path)
  → EP=1 branch (lines 563-574 of auto_ep_layer.py) → must remain intact
```

---

## 4. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 (single GPU) implications:

Current (before #8064):
  → AutoEP + EP=1 → ZeRO-2 → single GPU MoE → already viable
  → No folding needed → tp=1, ep=1 → both identity

After #8064 (if merged):
  → Folding adds code paths → but TP=1+EP=1 = identity → zero change to behavior
  → Risk: new code paths may have edge cases for EP=1 → needs verification
  → Benefit: preparation for future multi-GPU setups (e.g., 2x RTX 4090)

★★★★★★★★★ Future multi-GPU RTX 4090 configurations (hypothetical):
  → 2 GPUs: tp=2/ep=1 → TP for dense, EP=1 for MoE → dense split across GPUs
  → 2 GPUs: tp=1/ep=2 → EP for MoE, TP=1 for dense → experts split across GPUs
  → 4 GPUs: tp=2/ep=2 → both split → maximum utilization

★★★★★★★★★ Key: #8064 is preparation for future, NOT needed for current RTX 4090 GRPO:
  → Single GPU: EP=1 path unchanged → #8064 adds no value on single GPU
  → Multi-GPU: #8064 enables new configurations → future capability
  → Priority: LOW for RTX 4090 single GPU → MEDIUM for future multi-GPU
```

---

## 5. Relationship to Existing DeepSpeed Features

```
★★★★★★★★★ #8064 builds on existing features:

AutoEP #7938 (MERGED):
  → Config-only MoE → ZeRO-0/1/2 support → EP=1 branch
  → #8064 adds: folding → TP+EP coexistence → extends AutoEP to multi-GPU

AutoTP (existing):
  → autotp_size → automatic tensor parallelism for dense
  → #8064 adds: mode-aware gradient reduction → router/gate handled per mode

ZeRO-2 (existing):
  → Optimizer states on CPU → CPU_Adam → single GPU optimal
  → #8064 adds: per-parameter-family ZeRO checkpoint metadata → multi-family management

★★★★★★★★★ #8064 dependency chain:
  → #8060 (ZeRO-3 support) → planned as follow-up → not yet merged
  → #8064 → covers AutoEP+AutoTP folding → ZeRO-3 composition later
  → #8068 (gradient_clipping) → still OPEN → independent of #8064
```

---

## Key Findings Summary

★★★★★★★★★ DeepSpeed #8064: 2690 additions → TP+EP parallel folding → preparation for multi-GPU MoE
★★★★★★★★★ RTX 4090: TP=1+EP=1 = identity → #8064 adds zero value on single GPU
★★★★★★★★★ Mode-aware gradient reduction → router/gate params: summed when tokens partitioned, averaged when replicated
★★★★★★★★★ expert_tensor_parallel_size MUST be 1 currently → expert-internal TP reserved for follow-up
★★★★★★★★★ Validation: router/gate gradient parity ~1e-7, full unit tests pass on H100
★★★★★★★★★ Priority: LOW for RTX 4090 single GPU → MEDIUM for future multi-GPU

---

## References

- PR #8064: https://github.com/microsoft/DeepSpeed/pull/8064
- AutoEP #7938: notebook/projects/deepspeed-autoep-moe-source-reading.md
- DeepSpeed blockers: notebook/projects/deepspeed-v0.19.2-blockers-muon-cpu-offload-gap.md
