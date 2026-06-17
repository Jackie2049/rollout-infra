# DeepSpeed AutoEP MoE Source Reading — RTX 4090 RouterReplay Gap Analysis

> 2026-06-18 | Source-level analysis of DeepSpeed AutoEP MoE implementation | TokenChoiceTopKRouter + AutoEPMoELayer + GroupedExperts
> ★★★★★★★★ Confirmed: EP=1 branch has no AllToAll, dynamic routing (no RouterReplay), freeze router = 0 LOC deterministic

---

## Architecture Overview

```
★★★★★★★★★ DeepSpeed AutoEP MoE architecture (3 key components):

  1. TokenChoiceTopKRouter (ep_router.py, 188 lines)
     → Dynamic token-choice routing
     → gate(x) → scores → topk → selected_experts + top_scores
     → NO RouterReplay → every forward computes routing afresh

  2. AutoEPMoELayer (auto_ep_layer.py, ~600 lines)
     → Drop-in replacement for HF MoE blocks
     → EP=1 branch (lines 563-574): no AllToAll, local computation only
     → EP>1 branch (lines 576-593): AllToAll dispatch/combine

  3. GroupedExperts (ep_experts.py, ~200 lines)
     → Sequential for-loop OR torch._grouped_mm
     → SwiGLU expert MLP: w1(w3*x) * w2 → gate * up * down
```

---

## 1. TokenChoiceTopKRouter — Dynamic Routing Analysis

```
★★★★★★★★★ TokenChoiceTopKRouter source code analysis (ep_router.py):

  __init__ (lines 49-74):
    → gate = nn.Linear(dim, num_experts, bias=gate_bias)
    → key params: num_experts, top_k, score_func, route_norm, route_scale
    → group-limited routing: num_expert_groups, num_limited_groups
    → e_score_correction_bias = None (DeepSeek-V3 noaux_tc)

  forward() (lines 134-187):
    → Step 1: gate projection → scores = self.gate(x)
    → Step 2: scoring → sigmoid or softmax (float32 for stability)
    → Step 3: expert_bias addition (if provided)
    → Step 4: e_score_correction_bias (if set)
    → Step 5: node-limited routing (if num_expert_groups configured)
    → Step 6: topk selection → selected_experts_indices
    → Step 7: score gathering → top_scores = scores.gather(dim=1, index=selected_experts_indices)
    → Step 8: optional normalization → top_scores / denominator
    → Step 9: route_scale application
    → Returns: (top_scores, selected_experts_indices, num_tokens_per_expert)

★★★★★★★★★ KEY FINDINGS — NO RouterReplay:

  → Every forward pass recomputes routing from scratch
  → torch.topk() is DYNAMIC — results vary per input
  → NO caching of routing decisions across iterations
  → This is the1.8-2.3x throughput gap for CUDA graph mode

★★★★★★★★★ freeze_moe_router = 0 LOC immediate solution:

  → self.gate.weight.requires_grad = False
  → Same input = same gate output = same routing
  → Deterministic by design — no variability across batches
  → Works in eager mode (no torch.compile needed)
  → DOES NOT help CUDA graph (routing still recomputed each iteration)
```

---

## 2. AutoEPMoELayer — EP=1 Branch Analysis

```
★★★★★★★★★ AutoEPMoELayer EP=1 branch (lines 563-574):

  if self.ep_size == 1:
      # No AllToAll needed - local computation only
      local_counts = count_tokens_per_expert(
          ro.selected_experts, self.num_local_experts, out_dtype=torch.int32)

      routed_input_permuted, perm_indices, aligned_counts, n_tokens = \
          permute_by_local_expert(routed_input, local_counts)
      expert_output = self.experts(routed_input_permuted, aligned_counts)
      expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)

★★★★★★★★★ EP=1 KEY INSIGHTS:

  → No AllToAll communication → single GPU → no cross-rank dispatch
  → count_tokens_per_expert → histogram for local expert assignment
  → permute_by_local_expert → sort tokens by expert for sequential/grouped computation
  → self.experts → GroupedExperts → for-loop or grouped_mm computation
  → unpermute_by_local_expert → restore original token order

★★★★★★★★★ EP>1 branch (lines 576-593):

  → compute_split_plan → AllToAll dispatch plan
  → _AllToAllV.apply → dispatch tokens to remote experts
  → permute + expert compute + unpermute → local expert computation
  → _AllToAllV.apply → combine results from remote experts
  → NOT viable on single GPU (requires ep_size GPUs)
```

---

## 3. GroupedExperts — SwiGLU Computation

```
★★★★★★★★★ GroupedExperts (ep_experts.py):

  Two computation paths:

  _run_experts_for_loop (lines 31-80):
    → Sequential for-loop over experts
    → Each expert: w3 * x → gate, w1 * x → up, SwiGLU = gate * up, w2 * SwiGLU → down
    → Works on all PyTorch versions → reference implementation

  _run_experts_grouped_mm (lines ~):
    → torch._grouped_mm → batched grouped matrix multiply
    → Single kernel for all experts → faster for large expert counts
    → Requires torch._grouped_mm support (PyTorch 2.5+)

★★★★★★★★★ SwiGLU formula:
    → gate = silu(w3 * x)  → sigmoid linear unit activation
    → up = w1 * x           → up projection
    → SwiGLU = gate * up    → gated linear unit output
    → down = w2 * SwiGLU    → down projection → final output
```

---

## 4. RTX 4090 Implications — RouterReplay Gap

```
★★★★★★★★★ RouterReplay gap analysis for RTX 4090:

  Current state (no RouterReplay):
    → Every forward: gate(x) → topk → routing recomputed
    → Dynamic routing = correct behavior but NO caching
    → freeze_moe_router = 0 LOC → immediate deterministic routing
    → BUT: CUDA graph requires static routing indices → NOT compatible

  RouterReplay needed for:
    → CUDA graph mode → static routing → replay cached decisions
    → ~1.8-2.3x throughput improvement (from Megatron RouterReplay benchmarks)
    → 3 modes: RECORD → REPLAY_FORWARD → REPLAY_BACKWARD

  RouterReplay-equivalent design for DeepSpeed (~200-300 LOC):
    → RECORD mode: cache routing decisions per layer during first forward
    → REPLAY_FORWARD: reuse cached routing during subsequent forwards
    → REPLAY_BACKWARD: reuse cached routing during backward pass
    → Implementation: hook into TokenChoiceTopKRouter.forward()
    → Cache: selected_experts_indices + top_scores per layer per iteration

★★★★★★★★★ RTX 4090 action items:

  Immediate (0 LOC):
    → freeze_moe_router = True → deterministic routing → eager mode OK

  Short-term (when CUDA graph needed, ~200-300 LOC):
    → Implement RouterReplay-equivalent in DeepSpeed AutoEP
    → Hook into AutoEPMoELayer.forward() → cache routing
    → Enable CUDA graph capture with static routing indices
    → Expected throughput: 1.8-2.3x improvement
```

---

## 5. Source File Reference

```
★★★★★★★★★ Key source files in _temp_deepspeed/:

  deepspeed/moe/ep_router.py (188 lines)
    → TokenChoiceTopKRouter class
    → forward() method → dynamic routing computation

  deepspeed/module_inject/auto_ep_layer.py (~600 lines)
    → AutoEPMoELayer class
    → EP=1 branch (lines 563-574) → no AllToAll
    → EP>1 branch (lines 576-593) → AllToAll dispatch/combine

  deepspeed/moe/ep_experts.py (~200 lines)
    → GroupedExperts class
    → _run_experts_for_loop → sequential computation
    → _run_experts_grouped_mm → batched grouped computation

  deepspeed/moe/ep_count.py
    → count_tokens_per_expert → histogram computation

  deepspeed/moe/ep_kernels.py
    → TokenReorderer → token reorder/unreorder operations
```

---

## Key Findings Summary

★★★★★★★★★ TokenChoiceTopKRouter = dynamic routing → no caching → no RouterReplay → every forward recomputes
★★★★★★★★★ EP=1 branch (lines 563-574) confirmed → no AllToAll → single GPU viable
★★★★★★★★★ freeze_moe_router = 0 LOC → same input = same routing → deterministic by design
★★★★★★★★★ RouterReplay gap: CUDA graph needs static routing → 1.8-2.3x throughput missed
★★★★★★★★★ RouterReplay-equivalent: ~200-300 LOC → hook into forward() → cache routing decisions
★★★★★★★★★ GroupedExperts: for-loop OR grouped_mm → SwiGLU computation path

---

## References

- RouterReplay design: notebook/projects/deepspeed-router-replay-equivalent-design.md
- AutoEP config: notebook/projects/deepspeed-autoep-moe-reading.md
- ZeRO safety: tools/deepspeed_zero_safety_checker.py
- MoE config generator: tools/deepspeed_config_generator.py
