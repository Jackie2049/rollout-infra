# Megatron RouterReplay Source-Level Analysis

> 2026-06-17 | router_replay.py (207 lines) | MoE GRPO CUDA graph stability mechanism
> Key: 3 modes (RECORD/REPLAY_FORWARD/REPLAY_BACKWARD) → static buffers for CUDA graph → RTX 4090: EP=1 + RouterReplay + CUDA graph = deterministic MoE inference

---

## 1. RouterReplay Class Architecture

```
★★★★★★★★★ RouterReplay = 207 lines → global routing decision control for MoE:

Class hierarchy:
  RouterReplayAction (Enum) → 3 actions: RECORD, REPLAY_FORWARD, REPLAY_BACKWARD
  RouterReplay (class) → per-layer instance + global control methods

Key data structures:
  → global_router_replay_instances: List[RouterReplay] → one per MoE layer → order matters!
  → target_topk_idx: Optional[Tensor] → target indices for replay (set externally)
  → recorded_topk_idx: Optional[Tensor] → recorded indices from forward pass
  → router_replay_action: Optional[RouterReplayAction] → current mode for this layer
  → replay_backward_list: List[Tensor] → stack of indices for backward replay
  → static_buffer: Optional[Tensor] → CUDA graph compatible buffer [max_tokens, topk]
```

### 3-Mode Operation

| Mode | What | When | Output |
|------|------|------|--------|
| **RECORD** | Compute top-k + store indices | Forward pass first iteration | probs + indices (computed normally) |
| **REPLAY_FORWARD** | Use stored indices (skip top-k compute) | CUDA graph replay (forward) | probs gathered from stored indices |
| **REPLAY_BACKWARD** | Pop from backward_list (re-compute indices) | Backward pass re-computation | probs gathered from popped indices |
| **None** | Default top-k computation | Normal training | normal probs + indices |

★★★★★★★★★ **RECORD** mode: during initial forward pass, router computes top-k normally AND stores the indices. This is the "golden" routing decision that will be reused.

★★★★★★★★★ **REPLAY_FORWARD** mode: during CUDA graph replay, router skips top-k computation entirely — uses `target_topk_idx` directly → `scores.gather(1, top_indices)` → deterministic routing!

★★★★★★★★★ **REPLAY_BACKWARD** mode: during backward pass re-computation (activation checkpointing), pops indices from `replay_backward_list` → ensures gradient computation uses same routing → gradient consistency!

---

## 2. Global Control Methods

```python
# Static methods for controlling all routers at once:

RouterReplay.set_replay_data(all_layers_topk_indices)  # Distribute indices to all layers
RouterReplay.get_recorded_data()                        # Collect indices from all layers
RouterReplay.clear_global_indices()                     # Reset all stored indices
RouterReplay.set_global_router_replay_action(action)    # Set mode for ALL routers
RouterReplay.clear_global_router_replay_action()        # Clear mode for ALL routers
RouterReplay.clear_global_router_replay_instances()     # Prevent memory leaks
RouterReplay.set_global_static_buffers(static_buffer)   # CUDA graph buffer setup
RouterReplay.clear_global_static_buffers()              # Clear CUDA graph buffers
```

★★★★★★★★★ **set_global_static_buffers**: Takes a combined buffer of shape `[max_tokens, num_layers, topk]` → each layer gets a view `[max_tokens, topk]`. This is how CUDA graph compatibility is achieved — fixed-size buffers prevent dynamic allocation during graph replay!

---

## 3. CUDA Graph Compatibility — Static Buffers

```python
def record_indices(self, topk_indices: torch.Tensor):
    if self.static_buffer is not None:
        # CUDA graph path: copy into fixed-size buffer
        num_tokens = topk_indices.shape[0]
        self.static_buffer[:num_tokens].copy_(topk_indices)
        self.recorded_topk_idx = self.static_buffer[:num_tokens]
    else:
        # Non-CUDA graph path: just store reference
        self.recorded_topk_idx = topk_indices
```

★★★★★★★★★ **Why static buffers**: CUDA graphs require ALL tensors to be pre-allocated with fixed addresses. Dynamic tensor creation (new tensor each step) breaks CUDA graph. `static_buffer[:num_tokens].copy_(topk_indices)` writes into pre-allocated memory → CUDA graph replay sees same addresses → no breaks!

★★★★★ **Constraint**: `max_tokens` must be estimated upfront → if actual tokens exceed max_tokens → buffer overflow → need padding or max_tokens > worst case.

---

## 4. get_replay_topk — The Core Routing Decision Wrapper

```python
def get_replay_topk(self, scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
    if self.router_replay_action == RouterReplayAction.RECORD:
        # Step 1: compute top-k normally
        probs, top_indices = default_compute_topk(scores, topk, num_groups, group_topk)
        # Step 2: store indices for future replay
        self.record_indices(top_indices)
        return probs, top_indices

    elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
        # Skip top-k compute → use stored indices
        top_indices = self.target_topk_idx.to(scores.device)
        probs = scores.gather(1, top_indices)  # Gather probs for stored routing
        return probs, top_indices

    elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
        # Pop indices from backward stack
        top_indices = self.replay_backward_list.pop(0).to(scores.device)
        probs = scores.gather(1, top_indices)
        return probs, top_indices

    else:
        # Default: normal top-k computation (no replay active)
        return default_compute_topk(scores, topk, num_groups, group_topk)
```

★★★★★★★★★ **Key**: In REPLAY_FORWARD mode, the router doesn't compute top-k at all → uses stored indices → `scores.gather(1, top_indices)` only gathers the probability values for the stored expert indices. This means:
- Routing is DETERMINISTIC (same experts chosen every replay)
- Probs reflect current scores (not old scores) → correct gradient computation
- No top-k computation → faster during CUDA graph replay

---

## 5. RTX 4090 Impact

```
★★★★★★★★★ RouterReplay + RTX 4090:

Scenario A: MoE training (AutoEP EP=1) → CUDA graph + RouterReplay
  → EP=1 → all experts on same GPU → no AllToAll →
  → RouterReplay RECORD mode → initial routing →
  → RouterReplay REPLAY_FORWARD → CUDA graph replay → deterministic MoE forward!
  → ★★★★★ EP=1 + RouterReplay = deterministic + efficient → RTX 4090 viable!

Scenario B: MoE GRPO training → CUDA graph + RouterReplay
  → GRPO rollout → REPLAY_FORWARD → same routing per iteration →
  → GRPO training → RECORD → new routing each step →
  → ★★★★★★★★ GRPO needs BOTH: deterministic rollout (replay) + learning training (record)!
  → → RouterReplay = KEY mechanism for MoE GRPO stability with CUDA graphs!

Scenario C: MoE inference serving (vLLM/SGLang)
  → vLLM: no RouterReplay → dynamic routing each step →
  → SGLang: rl_on_policy_target → deterministic sampling → but no router replay per se
  → ★★★ inference doesn't need RouterReplay → each request is independent →
  → → but: MoE CUDA graph serving → could benefit from RouterReplay pattern → future work

★★★★★★★★★ RTX 4090 MoE GRPO pipeline:
  → DeepSpeed AutoEP EP=1 (AllToAll=identity) +
  → Megatron RouterReplay (deterministic CUDA graph routing) +
  → ZeRO-2 + CPU_Adam + LoRA +
  → ★★★★★★★★ BUT: Megatron crashes on single GPU (#5203) → need DeepSpeed instead!
  → → ★★★★★★★★ DeepSpeed AutoEP doesn't have RouterReplay → need to add or emulate!
  → → → ★★★ Coverage gap: DeepSpeed MoE GRPO needs RouterReplay-equivalent mechanism!
```

---

## 6. Comparison with Other Frameworks

| Framework | MoE Routing Replay | CUDA Graph Compatible | RTX 4090? |
|-----------|-------------------|---------------------|-----------|
| **Megatron** | RouterReplay (3-mode, static buffers) | Yes (static_buffer[:num_tokens].copy_) | ✗ (crashes #5203) |
| **DeepSpeed** | No RouterReplay | No explicit mechanism | ✓ (AutoEP EP=1) |
| **verl** | VeOmni Router Replay (R2/R3) | Partially (async) | ✓ (but no MoE opt) |
| **vLLM** | No routing replay | Dynamic routing per step | ✓ (Triton MoE) |
| **SGLang** | rl_on_policy_target (sampling deterministic) | Not routing replay | ✓ (Triton backend) |

★★★★★★★★★ **Gap**: DeepSpeed AutoEP (RTX 4090 MoE viable) doesn't have RouterReplay → MoE GRPO training with CUDA graphs on DeepSpeed needs routing determinism → potential contribution to DeepSpeed!

---

## Key Findings Summary

★★★★★★★★★ RouterReplay: 3 modes (RECORD/REPLAY_FORWARD/REPLAY_BACKWARD) → global per-layer control → 207 lines → clean design
★★★★★★★★★ RECORD: compute top-k + store indices → initial forward pass "golden" routing decision
★★★★★★★★★ REPLAY_FORWARD: skip top-k → use stored indices → CUDA graph compatible → deterministic!
★★★★★★★★★ REPLAY_BACKWARD: pop from backward_list → gradient consistency during checkpoint re-compute
★★★★★★★★★ Static buffers: [max_tokens, topk] per layer → combined [max_tokens, num_layers, topk] → CUDA graph pre-allocation
★★★★★★★★★ scores.gather(1, top_indices): current probs for stored routing → correct gradients → deterministic routing + correct probabilities
★★★★★★★★★ RTX 4090: EP=1 + RouterReplay + CUDA graph = deterministic MoE → BUT Megatron crashes (#5203) → need DeepSpeed equivalent!
★★★★★★★★★ DeepSpeed coverage gap: AutoEP (MoE viable) but no RouterReplay → MoE GRPO CUDA graph stability needs routing determinism → potential DeepSpeed contribution

---

## References

- router_replay.py source: Megatron-LM/megatron/core/transformer/moe/router_replay.py
- QB routing: notebook/projects/megatron-quantile-balancing-routing-source-reading.md
- Megatron GRPO SM89: notebook/projects/megatron-grpo-sm89-reading.md
- Megatron MoE architecture: notebook/projects/megatron-moe-architecture.md
- DeepSpeed AutoEP: notebook/projects/deepspeed-autoep-rtx4090-moe-practical.md
- verl VeOmni Router Replay: notebook/projects/verl-v08-veomni-flowgrpo-reading.md
