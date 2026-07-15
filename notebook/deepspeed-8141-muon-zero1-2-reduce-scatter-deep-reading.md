# DeepSpeed #8141: Muon ZeRO-1/2 reduce_scatter Fix Deep Reading

**Date**: 2026-07-15 (Session 10 continuation)
**PR**: deepspeedai/DeepSpeed #8141
**Author**: (PR by DeepSpeed team, replaces fail-fast #8090)
**Lines**: +106/-58
**Status**: OPEN, 0 reviews, 0 comments, CI passing
**Significance**: ★★★★★★★★ Replaces #8090 fail-fast with targeted reduction path — unblocks Muon+ZeRO-1/2+reduce_scatter

---

## 1. Problem (Root Cause: #7807)

When Muon optimizer is used with ZeRO-1/2 + `reduce_scatter=True`, a Muon parameter that **crosses a ZeRO partition boundary** does NOT have a fully reduced gradient on any single rank.

### How reduce_scatter works:
- `reduce_scatter` copies each averaged slice **only to that slice's owner rank**
- Each rank sees: its own reduced slice + local unreduced slices from other ranks
- For normal parameters (Adam/AdamW), this is fine — each rank only needs its partition slice

### How Muon works differently:
- Muon runs **Newton-Schulz orthogonalization on the FULL matrix** before `get_flat_partition()` selects the local ZeRO slice
- With reduce_scatter, each owner rank has a **DIFFERENT matrix**: its reduced slice + unreduced slices
- Result: **partition-then-orthogonalize** = silently corrupted update (O(1) divergence, not fp16 rounding)

### Previous approach (#8090):
- Added `raise ValueError("Muon and reduce scatter cannot be used together")` — FAIL FAST
- This blocked legitimate use cases where Muon could work with reduce_scatter

---

## 2. Fix Strategy

### Core idea: "broadcast the relevant slices to all owners"

For a Muon parameter spanning **more than one ZeRO partition**:
1. Each reduced slice is **copied to every rank that owns part of that parameter**
2. All owner ranks now see the SAME full reduced gradient
3. Each rank runs `muon_update` on the full matrix → `get_flat_partition()` keeps only local slice

For parameters within **one partition**: unchanged (owner-only copy path, no extra collective)

### Key code change in `stage_1_and_2.py`:

```python
# OLD: unconditional fail-fast
if isinstance(self.optimizer, MuonWithAuxAdam) and self.reduce_scatter:
    raise ValueError("Muon and reduce scatter cannot be used together")

# NEW: only fail-fast when optimizer offload is also enabled
if isinstance(self.optimizer, MuonWithAuxAdam) and self.reduce_scatter and self.cpu_offload:
    raise ValueError("Muon with reduce scatter does not support optimizer offload ...")
```

### Copy target change in `allreduce_and_copy_with_multiple_ranks`:

```python
# OLD: single-rank copy target
if dist.get_rank(group=process_group) == bucket_rank:
    buf.copy_(synced)

# NEW: multi-rank copy target (frozenset for Muon cross-partition)
copy_to_local_rank = local_rank in bucket_rank if isinstance(bucket_rank, frozenset) else local_rank == bucket_rank
if copy_to_local_rank:
    buf.copy_(synced)
```

### Muon copy ranks computation in `average_tensor`:

```python
# NEW: compute which ranks need full gradient for cross-partition Muon params
muon_copy_ranks = (frozenset(partition_ids) if isinstance(self.optimizer, MuonWithAuxAdam)
                   and getattr(param, "use_muon", False) and len(partition_ids) > 1 else None)
```

- `frozenset(partition_ids)` = all ranks that own part of this Muon parameter
- `len(partition_ids) > 1` = only for cross-partition parameters (not single-partition)
- `None` for non-Muon or single-partition → unchanged behavior

---

## 3. Communication Overhead Analysis

### With `use_multi_rank_bucket_allreduce=True` (default):
- No extra collective needed — the bucket all-reduce already delivers full gradient to all ranks
- Only the copy target changes (from single rank to frozenset)
- **Zero communication overhead**

### With `use_multi_rank_bucket_allreduce=False`:
- Only ranges belonging to **split Muon parameters** switch from reduce_scatter to all-reduce
- Non-Muon ranges keep reduce_scatter (unchanged)
- **Minimal overhead**: only affected Muon matrix ranges pay all-reduce cost

### This is elegant because:
- Most parameters are NOT Muon (embeddings, layer norms, biases → Adam fallback)
- Most Muon parameters stay within one partition (small models)
- Only large Muon matrices that straddle partition boundaries need the extra broadcast
- The cost scales with the number of cross-partition Muon parameters, NOT total parameters

---

## 4. RTX 4090 Impact

### dp=1 (single GPU): NO IMPACT
- No partition boundaries at dp=1 → all parameters in single partition
- Muon with reduce_scatter works as-is (no cross-partition case)
- But we DON'T use Muon on RTX 4090 anyway (4 blockers: #5394, #8068, #7776, #5179)

### dp≥2 (multi-GPU): UNLOCKS Muon+ZeRO-1/2+reduce_scatter
- Previously: fail-fast → couldn't even start training
- Now: targeted reduction → works correctly
- Still NOT recommended for RTX 4090 (PCIe bottleneck, Muon clipping bugs)

### Optimizer offload limitation:
- Muon + reduce_scatter + cpu_offload = STILL BLOCKED
- Offload retains only partition slices → can't broadcast
- This is acceptable: cpu_offload with Muon is rare (Muon needs GPU for Newton-Schulz)

---

## 5. Test Coverage

### Numerical correctness tests (2-GPU):
1. ZeRO Stage 1 and 2
2. `gram` and `standard` Newton-Schulz methods on all-reduce path
3. reduce_scatter with gradient accumulation and overlap communication
4. `use_multi_rank_bucket_allreduce=false`
5. Extra-large parameters and non-contiguous gradients
6. Optimizer offload rejection (still fails fast)

### Helper tests:
- Existing integer copy target (ZenFlow compatibility)
- New multi-owner frozenset target
- record_stream on consumer buffers (overlap_comm safety)

---

## 6. Cross-Framework Connection

| Bug | Framework | Mechanism | Status |
|-----|-----------|-----------|--------|
| #7807 | DeepSpeed ZeRO-1/2 | Muon cross-partition reduce_scatter corruption | Fix: #8141 |
| #8090 | DeepSpeed | Fail-fast for Muon+reduce_scatter | Replaced by #8141 |
| #5394 | Megatron | Muon global clipping stalls optimizer | OPEN (#5395 fix) |
| #8068 | DeepSpeed | gradient_clipping default=0 (stalls Muon) | MERGED June 23 |
| #7776 | verl | Muon clipping gradient norm | OPEN |
| #5179 | Megatron | Muon PyPI stub (can't install) | OPEN |

### Pattern family: Muon Optimizer Integration Bugs
- **Common thread**: Muon's Newton-Schulz requires full matrix → conflicts with distributed partitioning
- **DeepSpeed**: reduce_scatter partition-then-orthogonalize → #8141 fixes
- **Megatron**: global clipping before Newton-Schulz → #5395 fixes
- **All**: Muon is a hybrid optimizer (2D weights only) → integration edge cases

---

## 7. Our Fork PR Connection

Our DeepSpeed fork PR #1 (#8061 overlap_comm multi-producer wait) is independent but complementary:
- #8141 fixes Muon+reduce_scatter data path
- Our PR #1 fixes overlap_comm stream race (affects ALL optimizers including Muon)
- Both needed for safe Muon+ZeRO-2 operation

### Recommendation:
- Track #8141 for merge status (0 reviews, likely 4-8 weeks)
- Our PR #1 should reference #8141 as complementary fix
- When both merge: Muon+ZeRO-2+reduce_scatter+overlap_comm becomes viable (but still NOT for RTX 4090 dp=1)

---

## Session Stats
- **Source diff**: +106/-58 across 4 files (stage_1_and_2.py core, tests, README)
- **PR status**: OPEN, 0 reviews, CI passing
- **Key insight**: frozenset-based multi-owner copy target replaces fail-fast — elegant zero-overhead design
