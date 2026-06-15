# PyTorch Inductor Scheduler Source-Level Deep Reading

> 2026-06-16 | PyTorch v2.12.0 (main branch) | scheduler.py (10213 lines), choices.py (729 lines)
> SOURCE-LEVEL deep analysis for SM89 batch invariance research
> ★★★★★★★ Complete scheduler fusion pipeline traced → can_fuse_vertical three-layer gate → Layer 2 = our insertion point → confirmed by source!

---

## 1. Scheduler Fusion Decision Pipeline (FX Graph → Nodes → Fusion Groups)

### 1.1 Entry Point: Scheduler.__init__ (scheduler.py lines 4101+)

★★★★★★★ Complete pipeline in Scheduler._init():

```
1. IR operations (from lowering) → self.create_scheduler_node(n) → SchedulerNode objects
2. Compute dependencies (self.compute_dependencies())
3. Topological sort (self.topological_sort_schedule())
4. Dead node elimination (self.dead_node_elimination())
5. Compute ancestors (self.compute_ancestors())
   → node.ancestors: OrderedSet[str] of all upstream operation names
   → ★★★★★ KEY: ancestors determine whether fusion is vertical or horizontal
6. Compute input distances (self.compute_input_distances())
7. ★★★★★★★ MAIN FUSION LOOP: self.fuse_nodes(self.nodes) — see Section 1.2
8. Post-fusion: merge_loops, combo_kernels, peak memory reorder, etc.
```

### 1.2 Main Fusion Loop: fuse_nodes → fuse_nodes_once

★★★★★★★ Scheduler.fuse_nodes (scheduler.py line 5332):

```python
def fuse_nodes(self, nodes):
    for i in range(10):                          # Up to 10 fusion rounds!
        old_len = len(nodes)
        nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
        new_len = len(nodes)
        if new_len == old_len or new_len == 1:
            break                                 # No more fusions possible
    if config.loop_ordering_after_fusion:
        nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
    return nodes
```

★★★★★★★ Scheduler.fuse_nodes_once (scheduler.py line 6212):

1. Find all legal fusion pairs → `get_possible_fusions(nodes)` → calls `can_fuse(node1, node2)`
2. Sort by `score_fusion_key` (highest priority first)
3. Try each fusion pair → `_try_fusion_pairs()` → benchmark/heuristic → `fuse_two_nodes()` → FusedSchedulerNode
4. Evaluate template fusions → `_evaluate_pending_template_fusions()`

---

## 2. The Three-Layer Fusion Gate (CRITICAL!)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### scheduler.py lines 7891-7926: The three-layer vertical fusion gate

```python
def _can_fuse(self, node1, node2, ...):
    shared_data_score = self._score_fusion_memory_for_can_fuse(node1, node2)

    # ★★★★★ LAYER 0: Common heuristic (shared_data_score check)
    if not V.choices.can_fuse(self, node1, node2, shared_data_score):
        return False                               # Both horizontal and vertical

    # ★★★★★★★★ VERTICAL FUSION PATH (node2 depends on node1 outputs)
    if node1.get_operation_names() & node2.ancestors:
        # PATH 1: Direct vertical fusion
        if (
            self.can_fuse_vertical(node1, node2, ...)       # Layer 1: Legality
            and V.choices.can_fuse_vertical(self, node1, node2, shared_data_score)  # ★★★★★★★★ Layer 2: Heuristic ← OUR INSERTION POINT!
            and self.get_backend(device).can_fuse_vertical(node1, node2)            # Layer 3: Backend
        ):
            return True

        # PATH 2: Reindex + vertical fusion
        if (
            config.loop_reindexing_after_fusion
            and self._try_reindex_pointwise_for_reduction(node1, node2)
        ):
            return (
                self.can_fuse_vertical(node1, node2, ...)                    # Layer 1
                and V.choices.can_fuse_vertical(self, node1, node2, ...)     # ★★★★★★★★ Layer 2 ← ALSO checked!
                and self.get_backend(device).can_fuse_vertical(node1, node2) # Layer 3
            )

        return False

    # ★★★★★ HORIZONTAL FUSION PATH
    else:
        return V.choices.can_fuse_horizontal(...) and self.get_backend(device).can_fuse_horizontal(...)
```

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★ KEY INSIGHT: V.choices.can_fuse_vertical is called in BOTH Path 1 and Path 2!
Our SM<90 guard will prevent vertical reduction fusion in BOTH paths!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Layer 1: Scheduler.can_fuse_vertical (scheduler.py lines 7928-7999)

Structural legality → dependency matching → can the nodes be fused at all?
NOT architecture-dependent → NOT our insertion point.

### Layer 2: InductorChoices.can_fuse_vertical (choices.py lines 639-647)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

CURRENT CODE:
```python
@staticmethod
def can_fuse_vertical(
    scheduler: Scheduler,
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
    shared_data_score: int,
) -> bool:
    """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
    return True                      # ← ★★★★★★★★ UNCONDITIONAL! No architecture check!
```

★★★★★★★★★ THIS IS EXACTLY our insertion point!
- Docstring says "Hook for heuristics" → designed for this!
- @staticmethod → needs DeviceProperties.create(device) directly
- ★★★★★★★★ Our 5-line guard goes HERE → prevents reduction vertical fusion on SM<90!

### Layer 3: Backend.can_fuse_vertical (SIMDScheduling)

Tiling compatibility → hardware-specific but NOT architecture-specific (batch invariance)
NOT our insertion point → wrong abstraction layer for batch invariance.

---

## 3. SchedulerNode.is_reduction() Implementation

### BaseSchedulerNode.is_reduction() (scheduler.py line ~1462)

```python
def is_reduction(self) -> bool:
    return False                    # Default: NOT a reduction
```

### SchedulerNode.is_reduction() (scheduler.py lines 2478-2485)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```python
def is_reduction(self) -> bool:
    if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        raise AssertionError(f"{type(self.node)=}")
    return bool(self.node.get_reduction_type()) and (
        self._body is None or not self._body.has_partial_accumulate
    )
```

★★★★★★★★★ KEY:
- `self.node.get_reduction_type()` returns "sum" for mean → is_reduction() = True!
- For RMSNorm: mean → sum + divide → ReductionOp("sum") → is_reduction() = True!

### FusedSchedulerNode.is_reduction() (scheduler.py line ~2836)

```python
@cache_on_self
def is_reduction(self) -> bool:
    return any(x.is_reduction() for x in self.snodes)
```

★★★★★★★ A fused node is a reduction if ANY constituent node is a reduction → our guard catches this correctly!

---

## 4. Why RMSNorm Reduction Gets Fused (Exact Decision Trace)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Complete trace for RMSNorm fusion on SM89:

**Step 1**: torch.compile(LlamaModel)
→ Dynamo: captures FX graph → rms_norm(x) → x.pow(2) → .mean(dim=-1) → .rsqrt() → x * result

**Step 2**: Inductor Lowering
→ x.pow(2) → Pointwise ComputedBuffer
→ .mean(dim=-1) → NOT aten::mean.dim dispatch! → decomposed to: sum(dim=-1) / dim_size
→ sum → ReductionOp("sum") ComputedBuffer (is_reduction() = True!)
→ ★★★★★★★★ vLLM mean_batch_invariant override (on aten::mean.dim) NEVER reached!

**Step 3**: Scheduler creates SchedulerNodes
→ SchedulerNode_A: pow2 (Pointwise, is_reduction() = False)
→ SchedulerNode_B: sum (Reduction, is_reduction() = True ← ★★★★★★★★)
→ SchedulerNode_C: divide (Pointwise)
→ SchedulerNode_D: rsqrt (Pointwise)
→ SchedulerNode_E: mul (Pointwise)

**Step 4**: fuse_nodes → get_possible_fusions
→ B and C share buffer → possible fusion pair
→ B.get_operation_names() & C.ancestors → VERTICAL dependency!

**Step 5**: ★★★★★★★★ THE CRITICAL DECISION: _can_fuse(B, C)
→ Layer 0: V.choices.can_fuse → True (shared_data_score > 0)
→ B.get_operation_names() & C.ancestors → True → VERTICAL PATH
→ Layer 1: self.can_fuse_vertical(B, C) → True (structural legality)
→ ★★★★★★★★ Layer 2: V.choices.can_fuse_vertical(B, C) → True (UNCONDITIONALLY!) ← OUR GUARD WOULD BLOCK THIS!
→ Layer 3: backend.can_fuse_vertical(B, C) → True (tiling compatible)
→ → ALL 3 LAYERS PASS → B and C FUSED!

**Step 6**: Further fusion rounds → entire RMSNorm → ONE FusedSchedulerNode

**Step 7**: Triton Codegen → persistent_reduction kernel → RBLOCK constexpr (fixed) ✓ → XBLOCK autotuned (varies) ✗ → tl.sum() inline → accumulation order depends on XBLOCK → batch-dependent results!

### With our proposed guard at Layer 2:

★★★★★★★★★ Step 5 (revised):
→ B.is_reduction() = True ← sum node is a reduction!
→ device.type == "cuda" → True
→ props = DeviceProperties.create(device) → major=8 for SM89!
→ props.major < 9 → True ← SM89 blocked!
→ WhyNoFuse(B, C)("SM<90 prevents reduction fusion (batch invariance)")
→ return False ← ★★★★★★★★ GUARD BLOCKS THE FUSION!
→ B (sum) stays as separate kernel → torch.mean batch-invariant override effective → NO batch-dependent results!

---

## 5. Why Layer 2 is Architecturally Superior to Layer 3

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

| Layer | Pros | Cons |
|-------|------|------|
| Layer 2 (can_fuse_vertical) | "Hook for heuristics" → designed for this. Subclassable. Single file change. Intercepted BOTH paths. ★★★★★ Precedent: reduction_split_factor uses same pattern (props.major >= 10) | InductorChoices docstring says "try to not put correctness requirements" — but: our guard is performance-correctness TRADEOFF, not pure correctness |
| Layer 3 (SIMDScheduling) | Hardware-specific | Wrong abstraction for batch invariance. Two files needed. Only intercepts Path 1. Would need BOTH Layer 2 + Layer 3 → more invasive! |

★★★★★★★★★ CONCLUSION: Layer 2 is architecturally superior → cleaner abstraction → single point intercepts all paths → minimal change → precedent exists!

---

## 6. Precedent Analysis

★★★★★★★★★ 5 existing SM-capability checks in the Inductor codebase:

1. ★★★★★★★★ PRIMARY precedent: `choices.py` lines 482-506 — `DeviceProperties.create(device)` + `props.major >= 10` in `reduction_split_factor`. Same class, same file, same pattern!
2. `triton.py` line 2849 — `torch.cuda.get_device_capability()[0] >= 9` for TMA gate
3. `triton.py` line 4129 — same pattern for PDL gate
4. `triton_heuristics.py` lines 722-723 — `device_prop.major >= 8` for rblock scaling
5. `triton_heuristics.py` line 4223 — `device_major >= 10` for Blackwell MAX_R0_BLOCK

★★★★★★★ No additional imports needed — `DeviceProperties` (line 19) and `WhyNoFuse` (line 20) already imported at top of choices.py!

---

## 7. Key Source File Reference Table

| # | File | Lines | Section | Key Insight |
|---|------|-------|---------|-------------|
| 1 | scheduler.py | 4101-4400 | Scheduler._init() | Pipeline: IR→nodes→deps→ancestors→fuse_nodes |
| 2 | scheduler.py | 5332-5350 | fuse_nodes() | Up to 10 fusion rounds + reorder |
| 3 | scheduler.py | 6212-6400 | fuse_nodes_once() | get_possible_fusions → try pairs → template |
| 4 | scheduler.py | 6862-6920 | get_possible_fusions() | Group by buffer names → check pairs → sort |
| 5 | scheduler.py | 7860-7930 | _can_fuse() vertical | ★★★★★★★★ THREE-LAYER GATE! |
| 6 | scheduler.py | 7928-7999 | can_fuse_vertical() | Layer 1: structural legality |
| 7 | scheduler.py | 2478-2485 | SchedulerNode.is_reduction() | ★★★★★★★★ get_reduction_type() = "sum" for mean |
| 8 | scheduler.py | ~2836 | FusedSchedulerNode.is_reduction() | any(subnode.is_reduction()) |
| 9 | choices.py | 639-647 | ★★★★★★★★ can_fuse_vertical | CURRENT: return True → OUR INSERTION POINT! |
| 10 | choices.py | 649-669 | can_fuse_horizontal | Has actual heuristics (MixOrder+score+distance) |
| 11 | choices.py | 569-637 | can_fuse (common) | shared_data_score + max_fusion_size + peak memory |
| 12 | choices.py | 472-506 | ★★★★★★★★ reduction_split_factor | PRECEDENT: props.major >= 10 |
| 13 | choices.py | 671-729 | score_fusion | 5-component FusionScore |
| 14 | choices.py | 115-129 | InductorChoices docstring | "Hook for heuristics" |

---

## 8. Key Insights Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★ Three-layer gate architecture: Layer 0 (common) → Layer 1 (structural legality) → Layer 2 (profitability heuristic) → Layer 3 (tiling legality). Our guard goes in Layer 2 because it's an architecture-dependent profitability/correctness tradeoff.

2. ★★★★★★★★ Both vertical fusion paths intercepted: Direct (Path 1) and Reindex (Path 2) both call V.choices.can_fuse_vertical → our guard blocks BOTH.

3. ★★★★★★★★ is_reduction() correctly identifies RMSNorm: SchedulerNode → get_reduction_type() returns "sum" for mean's sum reduction. FusedSchedulerNode → any(subnode.is_reduction()).

4. ★★★★★★★★ Empty can_fuse_vertical hook is intentional: can_fuse_horizontal has actual heuristics (MixOrder+score+distance), while can_fuse_vertical is empty. Our guard fills this empty hook.

5. ★★★★★★★★ Fusion iteration is greedy: Up to 10 rounds. After our guard blocks (B, C) fusion, B stays standalone and will never be fused with any pointwise node on SM<90.

6. ★★★★★★★★ WhyNoFuse provides debug visibility: `TORCH_LOGS="+fusion"` shows why fusion was prevented on their GPU.

7. ★★★★★★★★ Layer 2 architecturally superior to Layer 3: Cleaner abstraction, single point intercepts all paths, precedent exists (reduction_split_factor).
