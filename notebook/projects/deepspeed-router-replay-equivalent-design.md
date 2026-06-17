# DeepSpeed RouterReplay Equivalent Design — Closing AutoEP Coverage Gap

> 2026-06-18 | Design for routing determinism mechanism in DeepSpeed AutoEP MoE
> ★★★★★★★★ Coverage gap: DeepSpeed AutoEP viable for MoE GRPO on RTX 4090 BUT lacks RouterReplay → routing determinism for CUDA graph stability
> ★★★★★★★★ Key insight: freeze_moe_router = immediate zero-LOC solution for simple GRPO; full RouterReplay needed for CUDA graph future

---

## 1. Current DeepSpeed MoE Routing — Source-Level Analysis

```
★★★★★★★★★ DeepSpeed has TWO MoE implementations:

Legacy MoE (sharded_moe.py, 687 lines):
  → TopKGate (452-533) with top1gating/top2gating/topkgating
  → Routing: logits = F.linear(input, self.wg.weight) → top-k → no determinism
  → noisy_gate_policy (Jitter/RSample) ADDS randomness → anti-deterministic!
  → MOELayer.forward (589-686) → always computes routing from scratch

AutoEP MoE (ep_router.py, 188 lines + auto_ep_layer.py, 628 lines):
  → TokenChoiceTopKRouter (25-187) → ported from TorchTitan
  → Routing: scores = self.gate(x) → softmax/sigmoid → optional expert_bias → torch.topk → scores.gather
  → Returns RouterOutput(top_scores, selected_experts_indices, num_tokens_per_expert)
  → Supports expert_bias (dynamic) + e_score_correction_bias (DeepSeek-V3 noaux_tc)
  → ★★★★★★★★ NO replay, determinism control, or mode switching exists!

★★★★★★★★★ AutoEP layer forward path (EP=1, RTX 4090):
  → Router computes routing → TokenReorderer sorts by expert → permute_by_local_expert → GroupedExperts → unpermute → combine
  → EP=1: local computation only → no AllToAll → identity routing pattern
```

---

## 2. Does DeepSpeed Have Existing Routing Determinism?

```
★★★★★★★★★ NO. DeepSpeed has zero routing determinism mechanism:

  → No RouterReplay equivalent (grep: zero results for router.*replay|RouterReplay)
  → No mode switching (RECORD/REPLAY) in TokenChoiceTopKRouter
  → No static buffer mechanism for CUDA graph compatibility
  → ep_count.py has deterministic_safe flag for torch.bincount → NOT routing determinism
  → CUDA graph infrastructure (cuda_graph.py, 28 lines) → simple ABC → no MoE-specific support
  → No PR or issue for CUDA graph + MoE routing determinism

★★★★★★★★★ Confirmed: DeepSpeed AutoEP coverage gap for MoE GRPO routing determinism
```

---

## 3. RouterReplay Equivalent Design for DeepSpeed AutoEP

### Core Class: `DeepSpeedRouterReplay`

```python
class RouterReplayAction(Enum):
    RECORD = "record"           # Compute routing + store indices (forward pass)
    REPLAY_FORWARD = "replay_forward"  # Use stored indices, skip top-k (CUDA graph)
    REPLAY_BACKWARD = "replay_backward"  # Pop from backward list (act checkpointing)

class DeepSpeedRouterReplay:
    global_instances: List[DeepSpeedRouterReplay] = []  # Per AutoEPMoELayer

    # Per-instance state
    target_topk_idx: Optional[Tensor]      # Indices to replay
    recorded_topk_idx: Optional[Tensor]    # Indices from forward
    replay_backward_list: List[Tensor]     # Stack for backward recompute
    static_buffer: Optional[Tensor]        # [max_tokens, topk] for CUDA graph

    # Key method: intercept routing in TokenChoiceTopKRouter.forward()
    def get_replay_topk(self, scores, top_k, num_experts, default_compute_fn):
        if action == RECORD:
            probs, indices = default_compute_fn(scores)  # Normal top-k
            self.record_indices(indices)                  # Store
            return probs, indices
        elif action == REPLAY_FORWARD:
            top_indices = self.target_topk_idx            # Skip top-k compute
            probs = scores.gather(1, top_indices)        # Current probs
            return probs, top_indices
        elif action == REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return default_compute_fn(scores)            # Normal mode
```

### Integration Points

```
★★★★★★★★★ 2 integration points in DeepSpeed AutoEP:

1. TokenChoiceTopKRouter.forward() (ep_router.py, lines 134-187):
   Before: _, selected_experts_indices = torch.topk(scores_for_choice, k=self.top_k)
   After:  probs, selected_experts_indices = self._router_replay.get_replay_topk(
               scores_for_choice, self.top_k, self.num_experts,
               default_compute_fn=lambda s: torch.topk(s, k=self.top_k))

2. AutoEPMoELayer.__init__() (auto_ep_layer.py, line 387):
   Each layer creates DeepSpeedRouterReplay instance → auto-registers into global_instances

★★★★★★★★★ EP=1 simplification on RTX 4090:
  → No AllToAllV → no cross-rank coordination needed
  → No PP/SP gather/scatter (verl router_replay_utils.py has 150 lines for this — NOT needed!)
  → No micro-batch accumulation across ranks
  → ~150 LOC reduction compared to verl's RouterReplay adaptation
```

---

## 4. LOC Estimate and Complexity

| Component | LOC | Notes |
|-----------|-----|-------|
| RouterReplayAction enum | 6 | Same 3-mode as Megatron |
| DeepSpeedRouterReplay class | ~80 | Simpler: no PP/SP/VPP needed for EP=1 |
| Global static methods | ~50 | Subset needed (no cross-rank) |
| get_replay_topk | ~25 | Adapted to DeepSpeed routing API |
| Static buffer management | ~15 | Same pattern: [max_tokens, topk] |
| Integration in ep_router.py | ~15 | Wrap top-k in TokenChoiceTopKRouter.forward |
| Integration in auto_ep_layer.py | ~10 | Create RouterReplay instance per layer |
| Tests | ~100 | Unit tests for 3 modes + EP=1 |
| **Total** | **~300** | ~180 core, 120 tests |

★★★★★★★★★ Complexity: Tier 2 contribution (~300 LOC, clear integration point, high value for MoE GRPO stability)

---

## 5. Is RouterReplay Actually Needed for RTX 4090 MoE GRPO?

```
★★★★★★★★★ NUANCED answer: needed for CUDA graph future, NOT needed for simplest GRPO path today

Why RouterReplay IS needed (full production path):
  → GRPO requires deterministic rollout routing → training uses different routing → inconsistent hidden states
  → CUDA graph stability requires static routing per replay → dynamic routing breaks capture
  → Activation checkpointing backward recompute must match original routing

Why RouterReplay may NOT be needed (simplest RTX 4090 path):
  → EP=1 with frozen router → same input = same routing → deterministic automatically!
  → LoRA on attention only → router frozen → no routing changes during training
  → No CUDA graph currently → no requirement for static routing
  → rLLM Tinker explicitly rejects RouterReplay → disabled for simplest path

★★★★★★★★★ Per-scenario assessment:

| Scenario | RouterReplay Needed? | Reason |
|----------|---------------------|--------|
| Simple GRPO + frozen router + no CUDA graph | NO | Frozen router = deterministic automatically |
| GRPO + router training + no CUDA graph | YES (REPLAY_FORWARD) | Router weights change between rollout/training |
| GRPO + CUDA graph (any config) | YES (RECORD + REPLAY) | CUDA graph requires static routing |
| Act checkpointing + MoE | YES (REPLAY_BACKWARD) | Backward must match original routing |

★★★★★★★★★ RTX 4090 recommended path: freeze router FIRST → RouterReplay when CUDA graph support added
```

---

## 6. Simpler Alternatives — Prioritized

### Alternative A: Freeze Router (ZERO LOC, immediate)

```
★★★★★★★★★ SIMPLEST and IMMEDIATE solution:

Set self.router.gate.weight.requires_grad_(False)
→ No routing changes during training → deterministic by definition
→ verl's freeze_moe_router does exactly this
→ RTX 4090 optimal path already assumes "LoRA on attention only + freeze router"

★★★★★★★★★ LOC: 0 (just a config flag)
★★★★★★★★★ Priority: HIGHEST — do this first, always
★★★★★★★★★ Limitation: Cannot train router. But LoRA on attention + frozen router = correct for GRPO anyway
```

### Alternative B: Python-Level Routing Cache (~50 LOC)

```
★★★★ Quick but incomplete:

Cache routing decisions at Python level → reuse during training forward pass
→ No mode switching, no static buffers, no global registry
→ ~50 LOC, simple, sufficient for "rollout record, training replay" pattern

★★★★ Limitation: Not CUDA graph compatible, not act checkpointing compatible
★★★★ Priority: MEDIUM — quick fix but incomplete
```

### Alternative C: Act Checkpoint Routing Save (~30 LOC)

```
★★★★ Limited scope solution:

Save routing decisions as checkpoint state → restore during backward recompute
→ ~30 LOC, addresses REPLAY_BACKWARD only

★★★★ Limitation: Only addresses backward recompute, not forward determinism
★★★★ Priority: LOW — limited scope
```

### Alternative D: Full RouterReplay Adaptation (~300 LOC)

```
★★★★★★★★★ COMPLETE solution for all scenarios:

Adapt Megatron's 208-line RouterReplay to DeepSpeed's AutoEP API
→ 3 modes (RECORD/REPLAY_FORWARD/REPLAY_BACKWARD)
→ Static buffers for CUDA graph compatibility
→ Global per-layer control
→ ~300 LOC (core 180, tests 120)

★★★★★★★★★ LOC: ~300
★★★★★★★★★ Priority: MEDIUM — needed when CUDA graph support added to AutoEP (PR #6816)
★★★★★★★★★ This is the Tier 2 DeepSpeed contribution identified in our OSS strategy
★★★★★★★★★ EP=1 simplification: ~150 LOC saved compared to verl's PP/SP RouterReplay
```

---

## Recommendation Summary

| Alternative | LOC | Completeness | CUDA Graph? | RTX 4090 Priority |
|-------------|-----|-------------|-------------|--------------------|
| **A: Freeze Router** | 0 | Partial (frozen only) | No (not needed) | **HIGHEST** — do first |
| B: Python Cache | ~50 | Partial | No | Medium |
| C: Act Checkpoint | ~30 | Partial (backward only) | No | Low |
| **D: Full RouterReplay** | ~300 | Complete | Yes | Medium — for CUDA graph future |

★★★★★★★★★ Pragmatic RTX 4090 path:
  1. TODAY: Alternative A (freeze router) → zero LOC → immediate viability → matches optimal RTX 4090 MoE GRPO path
  2. FUTURE: Alternative D (full RouterReplay) → ~300 LOC → when CUDA graph added → complete production path
  3. Alternative D = Tier 2 DeepSpeed contribution → fills identified coverage gap

---

## Key Source Files

| File | Path | Key Content |
|------|------|-------------|
| ep_router.py | deepspeed/moe/ep_router.py (188 lines) | TokenChoiceTopKRouter → integration point |
| auto_ep_layer.py | deepspeed/module_inject/auto_ep_layer.py (628 lines) | AutoEPMoELayer → instance registration |
| ep_experts.py | deepspeed/moe/ep_experts.py (196 lines) | GroupedExperts |
| sharded_moe.py | deepspeed/moe/sharded_moe.py (687 lines) | Legacy TopKGate + MOELayer |
| cuda_graph.py | deepspeed/model_implementations/features/cuda_graph.py (28 lines) | Minimal CUDA graph ABC |
| router_replay.py | Megatron-LM/megatron/core/transformer/moe/router_replay.py (208 lines) | Design reference |

---

## References

- Megatron RouterReplay source: notebook/projects/megatron-router-replay-source-reading.md
- DeepSpeed AutoEP practical: notebook/projects/deepspeed-autoep-rtx4090-moe-practical.md
- MoE GRPO RTX 4090: notebook/projects/moe-grpo-training-rtx4090-reading.md
- verl RouterReplay utils: verl/utils/megatron/router_replay_utils.py (555 lines)
- verl VeOmni RouterReplay: verl/utils/veomni/router_replay.py (523 lines)
