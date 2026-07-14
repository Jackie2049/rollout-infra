# DeepSpeed AutoEP + AutoTP Folding — Deep Reading

Based on PR #8064 by tohtana (tohtana/autoep-autotp-parallel-folding-design, +4066/-85)

## Core Innovation: Parallel Folding

AutoTP (tensor parallelism) for dense/attention path coexists with AutoEP (expert parallelism) for routed-expert path on the same ranks, WITHOUT forcing EP to be a subset of DP.

### Partitioning Formula

Two independent partitionings of the same rank set:
- **Dense/attention/shared-expert params**: `stage_size = tp * dp`
- **Routed-expert params**: `stage_size = ep * etp * edp`

Invariant: `tp * dp == ep * etp * edp == stage_size`

`dp` and `edp` are always derived, never user-configured. The only structural requirement: `stage_size % (ep * etp) == 0`.

### Cross-lane Expert Parallelism (Key Breakthrough)

EP no longer has to be a subset of DP:

| World | TP | EP | dp | edp | Layout |
|-------|----|----|-----|-----|--------|
| 4 | 2 | 4 | 2 | 1 | EP spans TP lanes and DP ranks |
| 4 | 4 | 4 | 1 | 1 | EP = whole TP group, 1 expert/rank |
| 8 | 4 | 4 | 2 | 2 | EP groups span TP lanes with replication |

EP groups are laid across a TP-lane-major rank ordering, so they may span TP lanes and dense-DP ranks.

## Architecture

### Topology (Pure Math — No Process Groups)

**`deepspeed/module_inject/auto_ep_folding.py`** (513 lines):
- `ParallelFoldingSpec` — immutable dataclass describing the topology
- `build_folding_spec()` — derives all sizes from public config
- `expected_folding_group_tables()` — computes TP/dense-DP/EP/EDP rank tables
- `local_folding_ranks()` — per-rank group membership

### Gradient Reduction Convention (Critical Design)

Per-family gradient reduction keyed to replication structure:

| Strategy | Applied to | Mechanism |
|----------|-----------|-----------|
| **AVERAGE** | Router/gate, dense/LayerNorm (replicated) | all_reduce then divide by tp_size |
| **EXPERT_TP_CANCEL** | Routed-expert weights | divide by tp_size only, NO all_reduce |
| **SKIP** | Genuinely TP-sharded params | owned by TP-specific path |
| **SUM** | Future SP-sharded params | reserved for sequence-parallel path |

**Why EXPERT_TP_CANCEL is needed**: The folded forward all-gathers expert outputs into a replicated full view via `restore_combined`, whose backward injects a `tp_size` factor on the expert-weight gradient. Experts are NOT TP-replicated, so they must NOT be TP all_reduced — the factor is simply divided out. Without this, folded expert gradients are over-scaled by `tp_size` — invisible to scale-invariant Adam but real for SGD/Lion/Muon and for gradient clipping (inflates expert contribution to global grad norm).

### Dispatch Mechanism

**`deepspeed/moe/ep_tp_dispatch.py`**:
- `partition_assignments()` — splits token-to-expert assignments across TP ranks
- `restore_combined()` — all-gathers expert outputs back into replicated full view
- `assert_tp_payload_consistent()` — optional validation hook
- `RoutedAssignmentPayload` — per-rank assignment tracking

### AutoEP Layer Integration

**`deepspeed/module_inject/auto_ep_layer.py`**:
- `set_deepspeed_parallelism(folding_group_handles=...)` — new code path for folded routing
- `forward()` — conditional branch: folded_tp uses `partition_assignments` + `restore_combined` instead of `combine_from_routed`
- `compute_split_plan_from_expert_indices()` — new EP plan computation from partitioned assignments

### Config Example

```json
{
  "tensor_parallel":  { "autotp_size": 4 },
  "expert_parallel":  { "enabled": true, "autoep_size": 4,
                        "expert_tensor_parallel_size": 1 }
}
```

## Validation

- 8×H100 confirmed: router/gate, LayerNorm, routed-expert gradient parity to non-folded baseline
- Cross-lane TP2×EP4 (4-rank, `ep>dp`) and TP4×EP4 (4-rank, `dp=1`) also validated
- SGD verified (not just Adam which is scale-invariant and masks errors)
- CPU/Gloo parity tests guard against 2.0x router/gate regression
- 9 new test files covering: config, groups, dispatch, runtime, grad parity, checkpoint, ZeRO-1 overlap, folding parity

## Current Limitations

| Limitation | Status | Impact |
|-----------|--------|--------|
| DeepCompile | **REJECTED** with folding | Must disable for AutoEP+AutoTP |
| Pipeline Parallel | `pp_size=1` only | Planned separately |
| Sequence Parallel | Mutually exclusive with AutoTP | SP folding follow-up |
| Expert-internal TP | `etp_size=1` only | Reserved follow-up |
| ZeRO-3 | Rejected for folding MVP | Separate lane planned (after #8060) |
| Offload | Not validated | Per-family replica groups needed |
| Universal checkpoint | Not supported | `ds_to_universal` raises NotImplementedError |

## RTX 4090 GRPO Relevance

- **AutoEP enables MoE training on multi-GPU** without manual parallelization code — critical for scaling from single 4090 to multi-node
- **Cross-lane EP** provides flexible allocation: can use smaller EP groups across limited GPUs
- **Gradient convention** must be respected: EXPERT_TP_CANCEL for routed experts prevents gradient inflation
- **No DeepCompile** with folding means must trade off compilation speed for memory efficiency
- **ZeRO-3 composition** is the final piece needed for memory-constrained RTX 4090 setups
- **etp=1 only** means expert-internal TP wait, but for small RTX 4090 clusters this is not a bottleneck

## References

- PR #8064: https://github.com/microsoft/DeepSpeed/pull/8064
- Bug #8102: AutoEP Python 3.9 compatibility (`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`)
- AutoEP presets: `deepspeed/module_inject/auto_ep_presets/base.py`
- Related: PR #8060 (ZeRO-3 foundation for folding)
- Related: Q3 Roadmap #8104

