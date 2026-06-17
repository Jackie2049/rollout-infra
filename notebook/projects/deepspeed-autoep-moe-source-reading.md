# DeepSpeed AutoEP MoE Source Reading — EP=1 RTX 4090 Branch + RouterReplay Gap

> 2026-06-18 | DeepSpeed AutoEP MoE implementation source analysis | RTX 4090 EP=1 + RouterReplay gap
> ★★★★★★★★ EP=1 confirmed: auto_ep_layer.py:563-574 explicit ep_size==1 branch → no AllToAll
> ★★★★★★★★ RouterReplay gap: TokenChoiceTopKRouter always computes top-k dynamically → no caching

---

## AutoEPMoELayer — EP=1 Branch (Lines 563-574)

```
★★★★★★★★★ Source: deepspeed/module_inject/auto_ep_layer.py

EP=1 branch (lines 563-574):
  if self.ep_size == 1:
      # No AllToAll needed - local computation only
      local_counts = count_tokens_per_expert(
          ro.selected_experts,
          self.num_local_experts,
          out_dtype=torch.int32,
      )

      routed_input_permuted, perm_indices, aligned_counts, n_tokens = permute_by_local_expert(
          routed_input, local_counts
      )
      expert_output = self.experts(routed_input_permuted, aligned_counts)
      expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)

★★★★★★★★★ EP=1 = NO AllToAll → RTX 4090 optimal:
  → All experts on single GPU → no communication needed
  → permute_by_local_expert → local reorder → same GPU
  → experts → GroupedExperts → for-loop or grouped_mm
  → unpermute_by_local_expert → local unsort → same GPU
  → → No NCCL → no latency → no NaN (#8061 avoided)

★★★★★★★★★ EP>1 branch (lines 575-593):
  → compute_split_plan → determines input/output splits
  → _AllToAllV.apply → sends tokens to expert-owning GPUs
  → permute_by_local_expert → reorder received tokens
  → experts → compute on received tokens
  → unpermute_by_local_expert → unsort output
  → _AllToAllV.apply → send output back to token-owning GPUs
  → → NOT applicable to single GPU RTX 4090
```

---

## TokenChoiceTopKRouter — Dynamic Routing (No RouterReplay)

```
★★★★★★★★★ Source: deepspeed/moe/ep_router.py

Router architecture:
  1. self.gate(x) → Linear(dim, num_experts) → compute logits per token
  2. score_func: "softmax" or "sigmoid" → convert logits to scores
  3. scores_for_choice = scores + expert_bias (optional load-balancing)
  4. torch.topk(scores_for_choice, k=self.top_k) → select top-k experts
  5. Route normalization: softmax → divide by sum, or just scale

★★★★★★★★★ KEY: Router is ALWAYS dynamic:
  → Every forward pass: gate(x) → compute logits → topk → select experts
  → NO caching of routing decisions → NO RouterReplay equivalent
  → Same input = same routing ONLY if gate weights unchanged (freeze router)
  → Different input = different routing → expected behavior

★★★★★★★★★ RouterReplay gap significance:
  → For GRPO training: NOT blocking → training works in eager mode
  → For CUDA graph: HIGH significance → CUDA graph requires static routing
  → freeze_moe_router = 0 LOC immediate solution → same input = same routing
  → Full RouterReplay (~300 LOC) → needed when CUDA graph support added to AutoEP
```

---

## GroupedExperts — Expert Computation

```
★★★★★★★★★ Source: deepspeed/moe/ep_experts.py

Two computation modes:

  1. _run_experts_for_loop (sequential, always available):
     → Iterate over each expert → SwiGLU MLP → concatenate
     → Works on all PyTorch versions → fallback path
     → Slower for many experts → but reliable

  2. torch._grouped_mm (fast path, if available):
     → Single grouped matmul call → batched expert computation
     → Requires PyTorch grouped_mm support → may not be available
     → Fail-fast RuntimeError if unavailable → falls back to for-loop

★★★★★★★★★ RTX 4090 implications:
  → For-loop path = reliable = always works
  → LoRA on experts: works in both paths → lora_target_modules: expert_mlp
  → MoE SwiGLU: gate + down + up weights per expert → our Triton fusion target
```

---

## Token Reordering — Permute/Unpermute Pattern

```
★★★★★★★★★ Source: deepspeed/moe/ep_repack.py

Two helper functions:

  permute_by_local_expert(routed_input, counts):
    → Reorder tokens so each expert's tokens are contiguous
    → Returns: permuted_input, perm_indices, aligned_counts, n_tokens
    → Essential for grouped computation → experts need contiguous input

  unpermute_by_local_expert(expert_output, perm_indices, n_tokens):
    → Undo the permutation → restore original token order
    → Returns: output in original token order
    → Combine with top_scores → weighted sum of expert outputs

★★★★★★★★★ EP=1 optimization:
  → permute/unpermute are local GPU operations → no inter-GPU communication
  → On RTX 4090: same GPU → fast → no bottleneck
  → LoRA-compatible: LoRA applied after permute → per-expert LoRA
```

---

## Combine Mechanism — Expert Output Merging

```
★★★★★★★★★ Source: deepspeed/module_inject/auto_ep_layer.py:594-601

combine_from_routed:
  → Merge expert outputs using routing scores as weights
  → top_scores → softmax normalization → multiply each expert output
  → Result: weighted sum of top-k expert outputs per token

★★★★★★★★★ RTX 4090 implications:
  → combine = local operation → no communication → fast
  → Deterministic IF routing is deterministic (freeze router)
  → → combine weights = top_scores → same router = same weights = deterministic
```

---

## Complete AutoEP Forward Flow on RTX 4090 (EP=1)

```
★★★★★★★★★ 10-step forward flow on single GPU:

  1. x → gate(x) → compute logits for each token
  2. logits → score_func (softmax/sigmoid) → routing scores
  3. scores → torch.topk → select top_k experts per token
  4. scores → softmax normalization → combine weights
  5. x[token_indices_sorted] → reorder by expert assignment
  6. permute_by_local_expert → contiguous per-expert groups
  7. experts(for_loop or grouped_mm) → compute expert outputs
  8. unpermute_by_local_expert → restore original order
  9. combine_from_routed → weighted sum using routing scores
  10. output → next layer

★★★★★★★★★ ALL 10 steps on single GPU:
  → No AllToAll → no NCCL → no communication → fast
  → freeze router → step 1-4 deterministic → entire pipeline deterministic
  → LoRA on attention + expert_mlp → trainable params = 0.2% → ~3.8GB optimizer
```

---

## Key Findings Summary

★★★★★★★★★ EP=1 explicit branch (auto_ep_layer.py:563-574) → no AllToAll → RTX 4090 optimal
★★★★★★★★★ TokenChoiceTopKRouter dynamic → no RouterReplay → freeze router = 0 LOC solution
★★★★★★★★★ GroupedExperts: for-loop (always) + grouped_mm (fast path, may not be available)
★★★★★★★★★ permute/unpermute pattern → local reorder → fast on single GPU
★★★★★★★★★ combine_from_routed → local weighted sum → deterministic if routing deterministic
★★★★★★★★★ 10-step forward flow ALL on single GPU → no inter-GPU communication needed

---

## References

- RouterReplay design: notebook/projects/deepspeed-router-replay-equivalent-design.md
- AutoEP analysis: notebook/fundamentals/deepspeed-autoep-moe-practical.md
- ZeRO safety: tools/deepspeed_zero_safety_checker.py
- Config generator: tools/deepspeed_config_generator.py
- GRPO config reference: tools/rtx4090_grpo_config_reference.py
