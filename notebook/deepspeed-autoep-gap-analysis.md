# DeepSpeed AutoEP + AutoTP Folding — Gap Analysis

**Source**: Deep reading of #8064 (MERGED July 12), #7938, #8060, #8068
**Date**: 2026-07-14

## 9 Gaps Identified (after #8064 merge)

### Gap 1: ETP > 1 (Expert-Internal Tensor Parallelism) — HIGH
- Currently hard-rejected: `if spec.etp_size != 1: raise ValueError`
- Topology math already done: `stage_size = ep * etp * edp`
- Missing: expert weight sharding (w1/w2/w3 column/row split), expert-output all-gather, gradient strategy (SUM vs AVERAGE), ZeRO integration
- No PR or issue exists for ETP > 1

### Gap 2: PP + AutoEP Folding — MEDIUM-HIGH
- Currently rejected: `if spec.tp_size > 1 and spec.pp_size > 1: raise ValueError`
- Topology math pp-aware: `stage_size = world_size // pp_size`
- Missing: per-stage folding spec, process group creation per pipeline stage, pipeline schedule integration
- Q3 roadmap (#8104) lists "Pipeline parallelism with Ray" (separate initiative)

### Gap 3: ZeRO-3 + AutoEP Folding — HIGH
- Currently rejected: `if spec.tp_size > 1 and zero_stage == 3: raise ValueError`
- #8060 already merged ZeRO-3 for non-folded AutoEP (tp=1)
- Missing: ZeRO-3 partitioned gradient hooks with folded TP correction, per-family ZeRO-3 partition groups, checkpoint gathering
- PR #8064 explicitly says "This PR should be adjusted for ZeRO3 support after #8060 is merged"

### Gap 4: EXPERT_TP_CANCEL Improvements — MEDIUM
- Unbucketed per-param all_reduce (reviewer PKUWZP flagged)
- TP correction now runs before BF16 high-precision accumulation (correct)
- Needs: bucketed TP gradient reduction, gradient clipping + folding integration test, ZeRO-2 partitioned gradient parity test

### Gap 5: Route-full Dispatch Optimization — MEDIUM
- 4 separate all_gather_variable_rows in restore_combined
- Could combine token_indices + capacity_slots + weights into packed all-gather
- Assignment partitioning doesn't account for load imbalance

### Gap 6: SP + AutoEP Folding — HIGH (when pass contracts land)
- Currently rejected: `if spec.tp_size > 1 and sp_size > 1: raise ValueError`
- Marker infrastructure already exists: `AUTOEP_FOLDING_SP_SHARDED_LAYERNORM_PARAM`, `AUTOEP_FOLDING_GRAD_REDUCE_SUM`
- Depends on DeepCompile pass contracts (#8139)

### Gap 7: Universal Checkpoint for Folding — MEDIUM
- `autoep_universal.py` raises NotImplementedError
- Blocks checkpoint portability across parallelism configs

### Gap 8: Offload + Folding — MEDIUM-HIGH (RTX 4090 critical)
- Currently rejected: `if spec.tp_size > 1 and (zero_offload_optimizer or zero_offload_param): raise ValueError`
- CPU offload + folding needed for memory-constrained setups

### Gap 9: DeepCompile + Folding — MEDIUM
- Currently rejected: `if deepcompile_enabled and spec.tp_size > 1: raise ValueError`
- Depends on #8139 pass contracts framework

## Contribution Priority Matrix

| Priority | Gap | Impact | No existing PR |
|----------|-----|--------|----------------|
| HIGH | ZeRO-3 + Folding | RTX 4090 multi-GPU | YES |
| HIGH | ETP > 1 | Larger experts | YES |
| HIGH | SP + Folding | MoE training | YES (after #8139) |
| MED-HIGH | PP + Folding | Multi-stage models | YES |
| MED-HIGH | Offload + Folding | RTX 4090 memory | YES |
| MED | Bucketed TP grad reduction | Folding throughput | YES |
| MED | Route-full dispatch optimization | Communication overhead | YES |
| MED | Universal checkpoint | Portability | YES |
| MED | DeepCompile + Folding | Compilation | Depends on #8139 |
