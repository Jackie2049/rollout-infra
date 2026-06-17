# Megatron RouterReplay vs DeepSpeed AutoEP MoE Routing — Comprehensive Source-Level Comparison

> 2026-06-18 | Source-verified comparison across Megatron RouterReplay, DeepSpeed TokenChoiceTopKRouter, verl VeOmniRouterReplay, and SGLang routing replay info
> ★★★★★★★★ Key finding: Megatron RouterReplay is the most complete CUDA-graph-ready design (207 lines, 3-mode), but #4256 R3 CLOSED — training integration did NOT land
> ★★★★★★★★ DeepSpeed freeze_moe_router = 0 LOC immediate solution for RTX 4090; RouterReplay ~300 LOC = Tier 2 contribution for CUDA graph future
> ★★★★★★★★ SGLang #9499 (routing replay info) and verl VeOmni R3 bridge the rollout-to-training gap that Megatron #4256 failed to merge

---

## 1. Architecture Comparison — Megatron 3-Mode vs DeepSpeed Always-Dynamic

### Megatron RouterReplay: 3-Mode State Machine (207 lines)

```
★★★★★★★★★ Megatron RouterReplay (router_replay.py, 208 lines):

RouterReplayAction Enum (lines 8-15):
  RECORD            = "record"           → compute top-k normally AND store indices
  REPLAY_FORWARD    = "replay_forward"   → skip top-k, use stored indices (CUDA graph)
  REPLAY_BACKWARD   = "replay_backward"  → pop from backward_list (activation checkpointing)

RouterReplay class (lines 18-208):
  global_router_replay_instances: List[RouterReplay]  → one per MoE layer, auto-registered
  target_topk_idx:     Optional[Tensor]  → indices to replay (set externally via set_replay_data)
  recorded_topk_idx:   Optional[Tensor]  → indices from forward pass (RECORD mode)
  router_replay_action: Optional[RouterReplayAction]  → current mode for this layer
  replay_backward_list: List[Tensor]     → FIFO stack for backward recompute
  static_buffer:       Optional[Tensor]  → [max_tokens, topk] → CUDA graph compatible

Integration in TopKRouter.__init__ (router.py, lines 217-219):
  self.router_replay = None
  if self.config.moe_enable_routing_replay:
      self.router_replay = RouterReplay()

★★★★★★★★★ The 3-mode state machine is the CORE architectural difference from DeepSpeed:

  1. RECORD mode (lines 160-165 of router_replay.py):
     → Compute top-k normally via default_compute_topk(scores, topk)
     → Store indices via self.record_indices(top_indices)
     → Returns BOTH probs and indices → full forward pass functionality
     → Purpose: GRPO rollout generation → initial routing → "golden" indices

  2. REPLAY_FORWARD mode (lines 166-172):
     → Skip top-k compute entirely
     → top_indices = self.target_topk_idx  → use pre-stored indices
     → probs = scores.gather(1, top_indices)  → gather current probs for stored routing
     → Purpose: CUDA graph replay → static routing → deterministic forward

  3. REPLAY_BACKWARD mode (lines 173-179):
     → Pop indices from FIFO stack: top_indices = self.replay_backward_list.pop(0)
     → probs = scores.gather(1, top_indices)  → gather current probs
     → Purpose: activation checkpointing recompute → must match original forward routing

  4. None (default, line 180-181):
     → Normal top-k computation → no replay active → fallback
```

### DeepSpeed TokenChoiceTopKRouter: Always-Dynamic (188 lines)

```
★★★★★★★★★ DeepSpeed TokenChoiceTopKRouter (ep_router.py, 188 lines):

__init__ (lines 49-74):
  self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
  → Parameters: num_experts, top_k, score_func, route_norm, route_scale
  → group-limited routing: num_expert_groups, num_limited_groups
  → e_score_correction_bias = None (DeepSeek-V3 noaux_tc)

forward() (lines 134-187):
  Step 1: scores = self.gate(x)                        → gate projection [T, num_experts]
  Step 2: sigmoid or softmax (float32)                 → scoring function
  Step 3: scores_for_choice = scores + expert_bias     → optional load-balancing bias
  Step 4: scores_for_choice += e_score_correction_bias → DeepSeek-V3 noaux_tc
  Step 5: _get_node_limited_routing_scores()           → group-limited routing
  Step 6: _, selected_experts_indices = torch.topk(    → ★★★★★★★★ DYNAMIC top-k
              scores_for_choice, k=self.top_k, dim=-1, sorted=False)
  Step 7: top_scores = scores.gather(dim=1,            → gather scores for selected experts
              index=selected_experts_indices)
  Step 8: optional normalization → route_scale
  Returns: (top_scores, selected_experts_indices, num_tokens_per_expert)

★★★★★★★★★ KEY DIFFERENCE from Megatron:
  → Line 173: torch.topk() called EVERY forward pass → NO caching, NO mode switching
  → NO RouterReplay equivalent exists anywhere in DeepSpeed
  → NO static buffer mechanism → NO CUDA graph compatible path
  → NO replay_backward_list → NO activation checkpointing backward consistency
  → Every forward recomputes routing from scratch → dynamic by design

★★★★★★★★★ Confirmed: DeepSpeed has ZERO routing determinism mechanism:
  → grep for "RouterReplay" or "routing_replay" → zero results across all DeepSpeed source
  → ep_count.py has deterministic_safe flag for torch.bincount → NOT routing determinism
  → cuda_graph.py is 28 lines → minimal ABC → no MoE-specific support
  → Legacy sharded_moe.py has noisy_gate_policy → ADDS randomness → ANTI-deterministic!
```

### Architecture Comparison Table

```
★★★★★★★★★ Source-verified architecture comparison:

| Feature                | Megatron TopKRouter         | DeepSpeed TokenChoiceTopKRouter |
|------------------------|----------------------------|--------------------------------|
| Source file            | router.py (738 lines)      | ep_router.py (188 lines)       |
| Routing computation    | topk_routing_with_score_function | torch.topk direct (line 173) |
| RouterReplay           | YES (207 lines, 3-mode)    | NO (zero)                      |
| Config flag            | moe_enable_routing_replay  | None                           |
| Gate implementation    | router_gating_linear (custom autograd) | nn.Linear (standard)    |
| Gate dtype control     | moe_router_dtype (fp32/fp64)| float32 cast in forward        |
| Score function         | softmax/sigmoid/sqrtsoftplus | softmax/sigmoid only          |
| Expert bias            | _apply_expert_bias (dynamic accumulation) | expert_bias + e_score_correction_bias |
| Group-limited routing  | group_limited_topk (moe_utils.py) | _get_node_limited_routing_scores |
| CUDA graph support     | static_buffer mechanism    | None                           |
| Act checkpoint backward| REPLAY_BACKWARD mode       | No mechanism                   |
| Input jitter           | moe_input_jitter_eps       | None                           |
| Z-loss                 | apply_z_loss               | None                           |
| Aux loss types         | aux_loss/seq_aux_loss/global_aux_loss | None (external)            |
| Pre-softmax routing    | moe_router_pre_softmax     | None                           |
| Inference router       | InferenceTopKRouter (@torch.compile) | None                        |
| Padding mask           | router.forward(padding_mask)| None                           |
| Token dropping         | apply_router_token_dropping| None (external)                |
```

---

## 2. CUDA Graph Compatibility — Static Buffers vs Dynamic Routing

### Megatron: Static Buffer Mechanism for CUDA Graph

```
★★★★★★★★★ Megatron CUDA graph compatibility — source-verified:

set_global_static_buffers (lines 78-92 of router_replay.py):
  Input: static_buffer tensor of shape [max_tokens, num_layers, topk]
  → assert static_buffer.shape[1] == num_layers
  → Each layer gets a view: static_buffer[:, layer_idx, :]
  → router_instance.set_static_buffer(static_buffer[:, layer_idx, :])
  → Shape per layer: [max_tokens, topk]

record_indices with static_buffer (lines 195-207):
  if self.static_buffer is not None:
      num_tokens = topk_indices.shape[0]
      self.static_buffer[:num_tokens].copy_(topk_indices)  ← ★★★★★★★★ KEY LINE
      self.recorded_topk_idx = self.static_buffer[:num_tokens]
  else:
      self.recorded_topk_idx = topk_indices

★★★★★★★★★ Why this works for CUDA graph:
  → CUDA graphs require ALL output tensors pre-allocated with fixed memory addresses
  → torch.topk() creates NEW tensors each invocation → different addresses → breaks CUDA graph
  → static_buffer[:num_tokens].copy_(topk_indices) writes into PRE-ALLOCATED memory
  → The slice self.static_buffer[:num_tokens] is a view of the same pre-allocated tensor
  → CUDA graph replay sees the same addresses → no breaks!

★★★★★★★★★ REPLAY_FORWARD also works because:
  → top_indices = self.target_topk_idx → stored in static_buffer → fixed address
  → scores.gather(1, top_indices) → uses stored indices → deterministic routing
  → No torch.topk() call → no dynamic tensor creation → CUDA graph compatible

★★★★★★★★★ Constraint: max_tokens must be estimated upfront
  → If actual tokens > max_tokens → buffer overflow → need padding
  → Recommended: max_tokens = worst_case_batch_size * max_seq_length
  → RTX 4090 with EP=1: max_tokens = batch_size * seq_len (no SP split)
```

### Megatron: MoE CUDA Graph Partial Capture Infrastructure

```
★★★★★★★★★ Megatron has BUILT-IN CUDA graph support for MoE (moe_utils.py, lines 1398-1626):

MoECudaGraphPartialCaptureSignal (lines 1398-1460):
  → Raised during CUDA graph capture to early-return from MoE layer forward
  → Two steps: "route" → capture router only; "preprocess" → capture router + preprocess
  → When CUDA graph replays: skips these steps → uses cached tensors

MoECudaGraphTensorStore (lines 1464-1514):
  → hidden_states, probs, routing_map, shared_expert_output
  → Filled during CUDA graph replay → skip router + preprocess computation
  → clear() between iterations

maybe_skip_or_early_return_by_cudagraph decorator (lines 1516-1626):
  → Wraps MoE layer forward methods: route, preprocess, shared_experts_compute
  → During replay: checks tensor_store → if not empty → skip computation → use cached values
  → During capture: raises MoECudaGraphPartialCaptureSignal → early return

★★★★★★★★★ This infrastructure works WITH RouterReplay:
  → RouterReplay RECORD mode → compute routing → store in static_buffer
  → CUDA graph capture → MoECudaGraphPartialCaptureSignal → capture router output
  → CUDA graph replay → skip router entirely → use cached probs/routing_map
  → RouterReplay REPLAY_FORWARD → provides deterministic indices for the cached routing_map
  → Together: RouterReplay + MoECudaGraphTensorStore = complete CUDA graph MoE solution!
```

### DeepSpeed: No CUDA Graph Mechanism for MoE

```
★★★★★★★★★ DeepSpeed has NO CUDA graph mechanism for MoE routing:

  cuda_graph.py (28 lines):
    → Minimal ABC: abstract capture() and replay() methods
    → No MoE-specific support
    → No static buffer mechanism
    → No partial capture signal infrastructure

  TokenChoiceTopKRouter.forward() (ep_router.py, line 173):
    → torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
    → ★★★★★★★★ This is a DYNAMIC operation → creates new tensors each call
    → CUDA graph CANNOT capture torch.topk with varying input → addresses change
    → No way to pre-allocate top-k output tensor → fundamentally incompatible

  ★★★★★★★★ The fundamental CUDA graph incompatibility:
    → torch.topk output shape depends on input shape → dynamic allocation
    → CUDA graph requires STATIC output addresses → pre-allocated buffers
    → Megatron solves this via RouterReplay REPLAY_FORWARD → skip torch.topk entirely
    → DeepSpeed has NO equivalent → torch.topk runs every time → CUDA graph breaks

  ★★★★★★★★ What DeepSpeed would need for CUDA graph MoE:
    → Step 1: RouterReplay-equivalent → skip torch.topk → use stored indices
    → Step 2: Static buffer for topk indices → [max_tokens, topk] pre-allocated
    → Step 3: MoE layer CUDA graph capture infrastructure (like Megatron's)
    → Step 4: Partial capture signal for MoE layer components
    → Estimated: ~300 LOC RouterReplay + ~100 LOC CUDA graph infra = ~400 LOC total
```

---

## 3. GRPO/RL Training Implications

### Why RouterReplay is Critical for GRPO

```
★★★★★★★★★ GRPO routing determinism requirements — source-verified:

GRPO training loop:
  1. Rollout: model generates responses → routing decisions computed per token
  2. Compute log probs: same model → forward pass → routing computed AGAIN
  3. Actor update: gradient computation → forward + backward → routing computed AGAIN

The PROBLEM without RouterReplay:
  → Each step computes DIFFERENT routing (torch.topk is input-dependent)
  → Rollout: token X → expert 3, 7
  → Compute log probs: token X → expert 3, 8  ← DIFFERENT routing!
  → Actor update: token X → expert 4, 7       ← DIFFERENT routing again!
  → → Hidden states differ → reward computation misaligned → training instability
  → → Gradient direction inconsistent → learning noise → slower convergence

★★★★★★★★★ RouterReplay solves this with 3 modes:

RECORD mode (rollout generation):
  → Router computes top-k normally AND stores indices
  → These are the "golden" routing decisions for this token set
  → Indices persisted → available for subsequent forward passes

REPLAY_FORWARD mode (compute log probs + actor update forward):
  → Router SKIPS top-k computation → uses stored indices
  → scores.gather(1, top_indices) → gathers current probability values
  → ★★★★★★★★ Same routing + current probs → correct gradient computation
  → Deterministic routing → hidden states identical → reward alignment correct

REPLAY_BACKWARD mode (actor update backward, activation checkpointing):
  → Pop indices from replay_backward_list → same indices as forward
  → ★★★★★★★★ Ensures backward recompute uses same routing as original forward
  → Without this: backward would recompute DIFFERENT routing → gradient inconsistency

★★★★★★★★★ scores.gather(1, top_indices) is the KEY design choice:
  → Megatron router_replay.py lines 170-172:
      top_indices = self.target_topk_idx.to(scores.device)
      probs = scores.gather(1, top_indices)
  → This gathers CURRENT probability values for STORED expert indices
  → → Routing is deterministic (same experts) BUT probabilities reflect current model state
  → → Gradient flows through current scores → correct learning signal
  → → NOT replaying old probabilities → those would be stale and wrong
```

### DeepSpeed GRPO Implications — Missing RouterReplay

```
★★★★★★★★★ DeepSpeed AutoEP MoE GRPO — WITHOUT RouterReplay:

  Scenario A: Frozen router + LoRA on attention only (RECOMMENDED RTX 4090 path):
    → self.gate.weight.requires_grad_(False) → deterministic by design
    → Same input → same gate output → same top-k → same routing
    → Rollout routing = compute_log_prob routing = actor update routing
    → ★★★★★★★★ freeze_router = 0 LOC → solves determinism for this scenario!
    → BUT: torch.topk still runs every forward → compute overhead → CUDA graph incompatible

  Scenario B: Router training (LoRA on router too):
    → Router weights change between rollout and training
    → Rollout routing ≠ training routing → hidden states differ → reward misaligned
    → ★★★★★★★★ freeze_router NOT viable → MUST have RouterReplay
    → RECORD during rollout → REPLAY_FORWARD during compute_log_prob + actor update
    → Without RouterReplay: GRPO with trainable router is UNSTABLE on DeepSpeed

  Scenario C: Activation checkpointing + MoE (common for memory optimization):
    → Backward recompute → forward runs again → routing recomputed
    → ★★★★★★★★ REPLAY_BACKWARD needed → same routing as original forward
    → Without it: backward routing ≠ forward routing → gradient noise
    → DeepSpeed has NO mechanism → gradient inconsistency under checkpointing

★★★★★★★★★ GRPO routing determinism gap per framework:

| Scenario                    | Megatron | DeepSpeed | verl VeOmni |
|-----------------------------|----------|-----------|-------------|
| Frozen router GRPO          | ✓ (R2 + freeze) | ✓ (freeze only) | ✓ (freeze_moe_router) |
| Trainable router GRPO       | ★★★★★★★ R2 (REPLAY_FORWARD) | ✗ (need RouterReplay) | ★★★★★★★ R2/R3 |
| CUDA graph MoE GRPO         | ★★★★★★★ (R2 + static_buffer) | ✗ (need full RouterReplay) | ★★★★★★★ (FSDP2 CG) |
| Act checkpointing MoE       | ★★★★★★★ (REPLAY_BACKWARD) | ✗ (need REPLAY_BACKWARD) | ★★★★★★★ (id-keyed) |
| R3 rollout-to-training      | ✗ (#4256 CLOSED) | ✗ | ★★★★★★★ (R3 mode) |
```

---

## 4. Implementation Comparison — Line-by-Line Key Differences

### Gate Computation

```
★★★★★★★★★ Gate computation comparison — source-verified:

Megatron Router.gating() (router.py, lines 86-108):
  → Custom RouterGatingLinearFunction autograd (moe_utils.py, lines 1237-1318)
  → router_gating_linear(inp, weight, bias, router_dtype)
  → router_dtype configurable: fp32/fp64 via moe_router_dtype config
  → Supports Transformer Engine te_general_gemm for TN layout
  → Weight is nn.Parameter → sequence_parallel attribute → TP-aware
  → CPU-to-GPU weight migration (lines 95-99)
  → ★★★★★★★★ Custom autograd → saves intermediate high-precision tensors → memory efficient

DeepSpeed TokenChoiceTopKRouter.forward() (ep_router.py, line 152):
  → scores = self.gate(x)
  → self.gate = nn.Linear(dim, num_experts, bias=gate_bias) (line 63)
  → Standard nn.Linear → no custom autograd
  → Float32 cast in forward for stability (lines 155-160)
  → ★★★★★★★★ Simpler but no memory optimization for router dtype
```

### Top-K Routing Dispatch

```
★★★★★★★★★ Top-k routing dispatch — THE CORE DIFFERENCE:

Megatron topk_routing_with_score_function (moe_utils.py, lines 672-839):
  → Lines 779-787: compute_topk() wrapper → delegates to RouterReplay if enabled

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if router_replay is None:
            return _compute_topk(scores, topk, num_groups, group_topk)
        else:
            return router_replay.get_replay_topk(
                scores, topk, num_groups, group_topk, _compute_topk)

  → ★★★★★★★★ SINGLE entry point for all routing → RouterReplay intercepts here
  → If router_replay is None → normal _compute_topk → torch.topk(scores, k=topk, dim=1)
  → If router_replay is set → mode-dependent dispatch → RECORD/REPLAY_FORWARD/REPLAY_BACKWARD

  → Line 643-654 in TopKRouter.routing():
    probs, routing_map = topk_routing_with_score_function(
        logits, self.topk, ..., router_replay=self.router_replay)

  → ★★★★★★★★ router_replay parameter threaded through → clean integration

DeepSpeed TokenChoiceTopKRouter.forward() (ep_router.py, line 173):
  → _, selected_experts_indices = torch.topk(
        scores_for_choice, k=self.top_k, dim=-1, sorted=False)

  → ★★★★★★★★ DIRECT torch.topk call → NO interception point
  → NO router_replay parameter → NO mode switching → always dynamic
  → sorted=False → faster but not deterministic across runs for same input
  → Megatron uses sorted=torch.is_grad_enabled() → sorted during training
```

### Score Function Processing

```
★★★★★★★★★ Score function processing — source-verified:

Megatron topk_routing_with_score_function (moe_utils.py, lines 793-818):
  → Line 793-799: softmax path
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probs, top_indices = compute_topk(scores, topk, ...)
    else:
        scores, top_indices = compute_topk(logits, topk, ...)  ← topk BEFORE softmax
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)

  → Line 800-811: sigmoid/sqrtsoftplus path
    scores = torch.sigmoid(logits.float())  OR  softplus(logits.float()).sqrt()
    if expert_bias is not None:
        scores_for_routing = scores + expert_bias.float()
        _, top_indices = compute_topk(scores_for_routing, ...)
        scores = torch.gather(scores, dim=1, index=top_indices)
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1

  → ★★★★★★★★ RouterReplay intercepts INSIDE compute_topk → applies BEFORE score math
  → REPLAY_FORWARD: indices come from stored target → probs = scores.gather(1, top_indices)
  → This means score math (softmax/sigmoid) still runs → correct probabilities

DeepSpeed TokenChoiceTopKRouter.forward() (ep_router.py, lines 155-183):
  → Line 155-160: score function
    if self.score_func == "sigmoid":
        scores = torch.sigmoid(scores.to(torch.float32))
    elif self.score_func == "softmax":
        scores = F.softmax(scores.to(torch.float32), dim=1)

  → Line 173: top-k AFTER score function
    _, selected_experts_indices = torch.topk(scores_for_choice, ...)

  → Line 176: gather scores
    top_scores = scores.gather(dim=1, index=selected_experts_indices)

  → ★★★★★★★★ Equivalent to Megatron's use_pre_softmax=True + softmax path
  → BUT: no pre_softmax option, no sqrtsoftplus, no RouterReplay interception
```

### Output Format

```
★★★★★★★★★ Output format comparison — source-verified:

Megatron TopKRouter.routing() (router.py, line 643-654):
  → Returns: (probs, routing_map)
  → probs: [num_tokens, num_experts] → SPARSE tensor → non-zero at top-k positions
  → routing_map: [num_tokens, num_experts] → boolean mask
  → Dense output mode for InferenceTopKRouter: [num_tokens, topk] probs + indices

  → When RouterReplay active:
    → RECORD: returns normal (probs, routing_map) → same as default
    → REPLAY_FORWARD: returns (probs, routing_map) → probs from scores.gather → routing_map from stored indices
    → ★★★★★★★★ Same output format regardless of mode → transparent to downstream

DeepSpeed TokenChoiceTopKRouter.forward() (ep_router.py, line 134-187):
  → Returns: (top_scores, selected_experts_indices, num_tokens_per_expert)
  → top_scores: [T, top_k] → DENSE tensor → only selected expert scores
  → selected_experts_indices: [T, top_k] → DENSE tensor → only selected expert indices
  → num_tokens_per_expert: [num_experts] → histogram

  → ★★★★★★★★ DENSE output format → simpler than Megatron's sparse format
  → BUT: no sparse routing_map → downstream must reconstruct if needed
  → For RouterReplay integration: selected_experts_indices is what needs caching
```

### Expert Bias Handling

```
★★★★★★★★★ Expert bias handling — source-verified:

Megatron TopKRouter._apply_expert_bias (router.py, lines 601-613):
  → self.local_tokens_per_expert += routing_map.sum(dim=0)  → accumulate per token
  → Dynamic bias → updated during training → load balancing
  → Applied in routing() via topk_routing_with_score_function expert_bias parameter
  → Requires torch.is_grad_enabled() → not applied during checkpointing recompute

DeepSpeed TokenChoiceTopKRouter.forward() (ep_router.py, lines 137-138, 162):
  → expert_bias parameter passed INTO forward() → NOT accumulated internally
  → scores_for_choice = scores + expert_bias (if provided) → additive bias
  → e_score_correction_bias (line 165-166) → DeepSeek-V3 noaux_tc routing
  → ★★★★★★★★ External bias → passed from training loop → not router's responsibility

  → For RouterReplay integration:
    → expert_bias changes across iterations → affects routing
    → RECORD mode: routing computed WITH expert_bias → stored indices reflect bias
    → REPLAY_FORWARD: stored indices used → expert_bias not needed for routing
    → BUT: probs.gather(1, top_indices) → current scores WITH current bias → correct probs
```

---

## 5. freeze_moe_router — 0 LOC Immediate Solution

```
★★★★★★★★★ freeze_moe_router = simplest and IMMEDIATE solution for RTX 4090:

What it does:
  → self.gate.weight.requires_grad_(False)
  → In DeepSpeed: self.router.gate.weight.requires_grad_(False)
  → Or: self.gate.weight.requires_grad = False  (property assignment)

Why it works:
  → Frozen router weight → same input → same gate(x) output → same scores
  → Same scores + same torch.topk → same selected_experts_indices
  → ★★★★★★★★ Deterministic routing by construction → NO code changes needed
  → Works in eager mode → no torch.compile required
  → Works with ZeRO-2 → weight frozen → no gradient → no optimizer state

Source verification:
  → DeepSpeed TokenChoiceTopKRouter.__init__ (ep_router.py, line 63):
    self.gate = nn.Linear(dim, num_experts, bias=gate_bias)
  → Setting requires_grad_(False) on this weight → routing frozen
  → verl's freeze_moe_router config does exactly this → proven in production

★★★★★★★★★ Limitations — source-verified:

  1. Cannot train router:
     → requires_grad=False → no gradient → router weight NEVER changes
     → GRPO with trainable router → freeze_router NOT viable
     → LoRA on attention only → router frozen → this IS the optimal path
     → LoRA on router → freeze_router contradicts LoRA purpose

  2. NOT CUDA graph compatible:
     → torch.topk still runs every forward (ep_router.py line 173)
     → Dynamic top-k output → new tensor addresses → breaks CUDA graph
     → freeze_router solves DETERMINISM but not DYNAMIC allocation
     → Need RouterReplay for CUDA graph → static_buffer → fixed addresses

  3. Activation checkpointing concern:
     → Frozen router → same input → same routing → seems safe
     → BUT: checkpointing recompute may have different padding → topk differs
     → With sorted=False (DeepSpeed line 173) → same scores may give different order
     → ★★★★★★★★ Megatron uses sorted=torch.is_grad_enabled() → safer
     → Recommendation: set sorted=True when freeze_router + checkpointing

★★★★★★★★★ RTX 4090 recommended config (freeze_router):

  DeepSpeed AutoEP + ZeRO-2 + freeze_router:
    → moe_type = "auto_ep"
    → ep_size = 1  → local computation only → no AllToAll
    → freeze_moe_router = True  → 0 LOC → deterministic routing
    → LoRA on attention layers only → router stays frozen
    → ZeRO-2 + CPU_Adam → optimal single-GPU memory
    → overlap_comm = False  → MUST (issue #8061, multi-stream NaN)
    → gradient_clipping = 1.0  → MUST (issue #8068, default was 0)
```

---

## 6. RouterReplay ~300 LOC — Porting Megatron Design to DeepSpeed

### Integration Points — Source-Verified

```
★★★★★★★★★ Two integration points in DeepSpeed AutoEP — source-verified:

Integration Point 1: TokenChoiceTopKRouter.forward() (ep_router.py, lines 134-187):

  Current code (line 173):
    _, selected_experts_indices = torch.topk(
        scores_for_choice, k=self.top_k, dim=-1, sorted=False)

  Proposed change (~15 LOC):
    def _compute_topk(self, scores, top_k):
        return torch.topk(scores, k=top_k, dim=-1, sorted=False)

    if self._router_replay is not None:
        probs, selected_experts_indices = self._router_replay.get_replay_topk(
            scores_for_choice, self.top_k, default_compute_fn=self._compute_topk)
    else:
        _, selected_experts_indices = self._compute_topk(
            scores_for_choice, self.top_k)

  → ★★★★★★★★ Then: top_scores = scores.gather(dim=1, index=selected_experts_indices)
  → This line works regardless → stored indices or computed indices

Integration Point 2: AutoEPMoELayer.__init__() (auto_ep_layer.py):
  → Each MoE layer creates a DeepSpeedRouterReplay instance
  → Auto-registers into global_instances list
  → ~10 LOC: self._router_replay = DeepSpeedRouterReplay() if config.moe_enable_routing_replay
```

### DeepSpeedRouterReplay Class Design

```
★★★★★★★★★ DeepSpeedRouterReplay class — adapted from Megatron RouterReplay:

class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"

class DeepSpeedRouterReplay:
    global_instances: List[DeepSpeedRouterReplay] = []

    def __init__(self):
        self.target_topk_idx: Optional[Tensor] = None
        self.recorded_topk_idx: Optional[Tensor] = None
        self.router_replay_action: Optional[RouterReplayAction] = None
        self.replay_backward_list: List[Tensor] = []
        self.static_buffer: Optional[Tensor] = None
        DeepSpeedRouterReplay.global_instances.append(self)

    def get_replay_topk(self, scores, top_k, default_compute_fn):
        if self.router_replay_action == RouterReplayAction.RECORD:
            _, indices = default_compute_fn(scores, top_k)
            self.record_indices(indices)
            top_scores = scores.gather(1, indices)
            return top_scores, indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx.to(scores.device)
            top_scores = scores.gather(1, top_indices)
            return top_scores, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0).to(scores.device)
            top_scores = scores.gather(1, top_indices)
            return top_scores, top_indices
        else:
            _, indices = default_compute_fn(scores, top_k)
            top_scores = scores.gather(1, indices)
            return top_scores, indices

    def record_indices(self, topk_indices):
        if self.static_buffer is not None:
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices

★★★★★★★★★ EP=1 simplification — what DeepSpeed DOESN'T need:

  → verl router_replay_utils.py has 555 lines → PP/SP/VPP complexity
  → DeepSpeed EP=1 on single GPU → SIMPLER by ~150 LOC:

  NOT needed for DeepSpeed EP=1:
    → No pipeline parallel (PP) → no pp_gather → no micro-batch across PP stages
    → No sequence parallel (SP) → no gather_from/spread_to_sequence_parallel_region
    → No virtual pipeline parallel (VPP) → no schedule_table → no reorder_and_merge_vpp_layers
    → No cross-rank coordination → no all_gather across ranks
    → No packed sequence handling → no preprocess/postprocess_packed_seqs
    → No attention_mask-based routing → no THD engine integration

  Still needed:
    → Global instance registry → set/get/clear methods
    → Mode switching → set/clear action methods
    → Static buffer management → set/clear buffer methods
    → get_replay_topk → core routing dispatch
    → record_indices → CUDA graph compatible storage
```

### LOC Estimate

```
★★★★★★★★★ LOC estimate for DeepSpeedRouterReplay — detailed:

| Component                        | LOC  | Notes                                    |
|----------------------------------|------|------------------------------------------|
| RouterReplayAction enum          | 6    | Same 3-mode as Megatron                  |
| DeepSpeedRouterReplay.__init__   | 15   | Instance state + auto-register           |
| get_replay_topk method           | 25   | Mode dispatch + scores.gather            |
| record_indices method            | 10   | Static buffer copy or reference          |
| set_target_indices               | 5    | Set target + append to backward list     |
| Global static methods            | 40   | set/get/clear actions, set/get/clear data|
| Static buffer methods            | 15   | set/clear buffers, set_global_static_buffers|
| Config flag addition             | 5    | moe_enable_routing_replay in AutoEP config|
| Integration in ep_router.py      | 15   | Wrap torch.topk in get_replay_topk       |
| Integration in auto_ep_layer.py  | 10   | Create RouterReplay per layer            |
| Tests                            | 70   | 3 modes + EP=1 + static buffer           |
| Total                            | ~206 | core 136, integration 30, tests 70       |

★★★★★★★★★ Revised estimate: ~206 LOC → closer to Megatron's 207 than initial 300 estimate
  → EP=1 simplification is more significant than initially estimated
  → Megatron RouterReplay itself is 207 lines → DeepSpeed adaptation similar scope
  → The 300 LOC estimate included verl-style PP/SP → NOT needed for EP=1
```

---

## 7. RTX 4090 Implications — EP=1, LoRA, RouterReplay

### EP=1 Scenario — Source-Verified

```
★★★★★★★★★ EP=1 on RTX 4090 — source-verified from AutoEPMoELayer:

AutoEPMoELayer EP=1 branch (auto_ep_layer.py, lines 563-574):
  if self.ep_size == 1:
      # No AllToAll needed - local computation only
      local_counts = count_tokens_per_expert(
          ro.selected_experts, self.num_local_experts, out_dtype=torch.int32)
      routed_input_permuted, perm_indices, aligned_counts, n_tokens = \
          permute_by_local_expert(routed_input, local_counts)
      expert_output = self.experts(routed_input_permuted, aligned_counts)
      expert_output = unpermute_by_local_expert(expert_output, perm_indices, n_tokens)

★★★★★★★★★ EP=1 implications:
  → No AllToAll → no cross-rank communication → single GPU only
  → count_tokens_per_expert → histogram of local expert assignments
  → permute_by_local_expert → sort tokens by assigned expert
  → GroupedExperts → for-loop or grouped_mm computation
  → unpermute → restore original token order + combine with routing probs

★★★★★★★★★ RTX 4090 RouterReplay implications for EP=1:

  Static buffer sizing:
    → max_tokens = batch_size * max_seq_length (no SP split)
    → For GRPO: batch_size = num_prompts, max_seq_length = max_response_length
    → Example: 8 prompts * 512 tokens = max_tokens = 4096
    → Buffer shape: [4096, num_layers, top_k] → e.g., [4096, 28, 8] for Qwen3-MoE
    → Memory: 4096 * 28 * 8 * 8 bytes (int64) = ~7.3 MB → negligible

  RouterReplay benefit quantification:
    → torch.topk overhead: ~0.1-0.3ms per layer per forward (depends on num_experts)
    → 28 MoE layers * 2 forward passes (compute_log_prob + actor update) = 56 topk calls
    → REPLAY_FORWARD skips ALL topk → saves ~5.6-16.8ms per GRPO iteration
    → ★★★★★★★★ BUT: freeze_router gives determinism WITHOUT topk skip → no time savings
    → Time savings only matter when CUDA graph enabled → full RouterReplay needed

  MoE training with LoRA:
    → LoRA on attention only → router frozen → freeze_router = 0 LOC solution
    → LoRA on MoE experts → router still frozen → freeze_router still works
    → LoRA on router → router trainable → freeze_router NOT viable → need RouterReplay
    → ★★★★★★★★ Recommended: LoRA on attention + frozen router → simplest RTX 4090 path
```

### RTX 4090 Scenario Assessment

```
★★★★★★★★★ RTX 4090 RouterReplay priority assessment — per scenario:

SCENARIO 1: Simple GRPO + frozen router + no CUDA graph (MOST LIKELY):
  → freeze_moe_router = True → 0 LOC → deterministic routing
  → Same input = same gate output = same routing → correct by design
  → LoRA on attention only → router frozen → matches optimal path
  → ★★★★★★★★ RouterReplay NOT needed → freeze_router solves everything
  → Recommended: DeepSpeed AutoEP EP=1 + ZeRO-2 + CPU_Adam + freeze_router

SCENARIO 2: GRPO + router training + no CUDA graph:
  → Router weights change between rollout/training → freeze_router NOT viable
  → REPLAY_FORWARD needed → record routing during rollout → replay during training
  → Python-level cache (~50 LOC) sufficient → no CUDA graph requirement
  → ★★★★★★★★ RouterReplay needed → but simple version suffices

SCENARIO 3: GRPO + CUDA graph (any config):
  → CUDA graph requires static routing → dynamic top-k breaks capture
  → Full RouterReplay with static buffers needed → ~206 LOC
  → RECORD → capture initial routing → REPLAY_FORWARD → reuse for CG replay
  → ★★★★★★★★ Full RouterReplay needed → Alternative D

SCENARIO 4: Activation checkpointing + MoE:
  → Backward recompute must match original routing → REPLAY_BACKWARD
  → Without it → gradient inconsistency → training instability
  → ★★★★★★★★ REPLAY_BACKWARD needed → even without CUDA graph
  → DeepSpeed ZeRO-2 commonly uses activation checkpointing → this is relevant!
```

---

## 8. Cross-Framework Synthesis — SGLang, verl, and Megatron

### SGLang #9499 — Routing Replay Info for RL

```
★★★★★★★★★ SGLang #9499 OPEN → MoE routing replay info → RL training bridge:

What #9499 proposes:
  → During inference/generation, capture MoE routing decisions as metadata
  → Expose routing info (which experts each token was routed to) as part of rollout output
  → This enables RL training to use the SAME routing decisions → RouterReplay-like behavior
  → ★★★★★★★★ SGLang generates routing info → training framework consumes it

How it relates to RouterReplay:
  → SGLang #9499 = the R3 INFERENCE side → captures routing during rollout generation
  → Megatron RouterReplay R3 = the R3 TRAINING side → replays captured routing
  → Together: #9499 + RouterReplay = complete R3 pipeline (rollout → training)
  → ★★★★★★★★ #4256 tried to merge both sides into Megatron → CLOSED → too complex
  → SGLang #9499 is MORE modular → inference captures → training consumes → cleaner

SGLang MoE + deterministic inference context:
  → SGLang has MoE LoRA support with Triton SGMV kernel
  → SGLang deterministic inference: 7 aten overrides → KERNEL-level batch-invariant
  → murmur_hash32 Gumbel-max float64 sampling → gold standard for GRPO
  → rl_on_policy_target auto-enable → deterministic sampling for RL
  → ★★★★★★★★ SGLang = BEST inference engine for MoE GRPO → deterministic + LoRA

★★★★★★★★★ SGLang #9499 + DeepSpeed RouterReplay design:
  → SGLang generates rollout → captures routing indices per token → #9499
  → Routing indices transferred to DeepSpeed training → set_replay_data()
  → DeepSpeed RouterReplay REPLAY_FORWARD → uses stored indices
  → ★★★★★★★★ This is the R3 pipeline: SGLang rollout → DeepSpeed training
  → No VeOmni dependency → SGLang is independent → cleaner than verl VeOmni

★★★★★★★★★ SGLang #9499 vs verl VeOmni R3 comparison:

| Feature              | SGLang #9499               | verl VeOmni R3              |
|----------------------|----------------------------|-----------------------------|
| Scope                | Inference-only → expose routing info | Full pipeline → capture + replay |
| Routing capture      | Inference engine side      | Rollout backend side        |
| Training consumption | Training framework consumes | VeOmni RouterReplay consumes|
| Response-only mask   | Not specified              | _replay_mask (prompt tokens native)|
| External dependency  | None (SGLang standalone)   | veomni pip package          |
| Integration          | Modular → any training FW  | Tightly coupled to VeOmni   |
| Status               | OPEN → not merged          | MERGED in verl v0.8.0       |
| RTX 4090 compatible  | YES (Triton backend SM89)  | YES (FSDP2)                 |
```

### verl MoE RouterReplay — Three Implementations

```
★★★★★★★★★ verl has 3 RouterReplay implementations — source-verified:

1. router_replay_patch.py (~180 lines):
   → Adapts Megatron RouterReplay for verl's Megatron backend
   → Same 3-mode enum: RECORD, REPLAY_FORWARD, REPLAY_BACKWARD
   → Patches TopKRouter.forward → wraps routing computation
   → References: THUDM/slime routing_replay.py → academic source
   → Lightweight → no PP/SP complexity → just core replay logic
   → ★★★★★★★★ Still depends on Megatron → inherits #5203 crash risk

2. router_replay_utils.py (555 lines) — source-verified:
   → Full RL training integration → PP/SP/VPP support
   → merge_router_topk_indices() (lines 219-273):
     → Collects recorded indices from all router instances
     → Stack per-layer → permute(1,0,2) → gather_from_sequence_parallel_region
     → Pack/unpack with attention_mask + input_ids → sequence alignment
     → Append to mini_layer_topk_idx_list → per micro-batch
   → set_router_replay_data() (lines 276-338):
     → Scatter routing data back to sequence-parallel ranks
     → Pack/unpack reverse → slice_input_tensor → scatter_to_sequence_parallel_region
     → Per-layer distribution → router.set_target_indices()
   → reorder_and_merge_vpp_layers() (lines 341-389):
     → VPP support → schedule_table → reorder micro-batches by model chunk
     → Reshape from [bs*vpp_size, max_token_len, layer_num_per_vpp, topk]
     → → to [bs, max_token_len, layer_num, topk]
   → pp_gather() (lines 422-483):
     → All-gather across PP ranks → concatenate layer dimensions
     → VPP offset computation → per-stage MoE layer counting
   → RouterReplayHelper (lines 486-555):
     → get_micro_batch_router_list() → locate local RouterReplay instances
     → is_r2_record_action() / is_replay_forward_action() / is_replay_backward_action()
   → ★★★★★★★★ 555 lines = 3x Megatron RouterReplay → PP/SP overhead significant

3. VeOmniRouterReplay (523 lines, veomni/router_replay.py) — source-verified:
   → Self-contained for VeOmni engine (FSDP2 + Ulysses SP, NO PP)
   → 3 actions: DISABLED, RECORD, REPLAY (2 replay modes, not 3)
   → id-keyed positional indexing → _id_to_pos dict → activation checkpointing safe
   → on_router_forward() (lines 188-315):
     → RECORD: slot.append(top_indices.detach().clone()) → recompute detection via slot density
     → REPLAY: substituted = target → torch.where(mask.unsqueeze(-1), target, top_indices)
     → Duplicate detection: sorted_sub.sort(dim=-1) → has_duplicate check
     → ★★★★★★★★ Falls back to native routing for prompt tokens AND duplicates
   → R3 mode support:
     → Response-only masking via _replay_mask → prompt tokens native routing
     → ★★★★★★★★ This is what #4256 tried to add → verl VeOmni HAS IT!
   → collect_recorded() (lines 417-459):
     → Stack per-layer → torch.stack(per_layer, dim=1) → [local_nnz, L, topk]
     → All-gather along Ulysses SP → trim padding suffix
   → slice_microbatch_replay_targets() (lines 462-486):
     → slice_input_tensor → same pad+split rule as input_ids
     → Unbind dim=1 → per-layer list → feed set_microbatch_targets
```

### Cross-Framework Routing Replay Status

```
★★★★★★★★★ Cross-framework routing replay status — comprehensive:

| Framework     | Implementation      | LOC  | Modes        | CUDA Graph | R3?  | RTX 4090?   |
|---------------|--------------------|------|-------------|-----------|------|------------|
| Megatron      | router_replay.py   | 207  | 3 (R/RF/RB) | static_buffer | ✗ (#4256 CLOSED) | ✗ crash #5203 |
| DeepSpeed     | NONE               | 0    | 0           | ✗         | ✗    | ✓ AutoEP EP=1 |
| DeepSpeed     | Proposed           | ~206 | 3 (R/RF/RB) | ✓ (future)| ✗ (R3 from SGLang)| ✓ (EP=1 + freeze) |
| verl          | router_replay_patch| ~180 | 3 (R/RF/RB) | ✓ (Megatron CG) | ✗  | ✗ (needs Megatron) |
| verl          | router_replay_utils| 555  | 3 (R/RF/RB) | ✓ (Megatron CG) | ✗  | ✗ (needs Megatron) |
| verl VeOmni   | veomni/router_replay| 523 | 3 (DIS/R/RP)| ✓ (FSDP2 CG) | ★★★★★★★ R3 | ✓ (FSDP2) |
| SGLang        | #9499 proposal     | N/A  | N/A (info)  | N/A       | ★★★★★★★ capture side| ✓ (Triton SM89) |

★★★★★★★★★ The R3 pipeline gap — what's missing:

  Complete R3 requires TWO sides:
    1. INFERENCE side: capture routing during generation → SGLang #9499 OR VeOmni rollout
    2. TRAINING side: replay captured routing during training → RouterReplay REPLAY_FORWARD

  Current status:
    → SGLang #9499: inference-side capture → OPEN → not merged → but modular design
    → verl VeOmni R3: both sides → MERGED → but requires veomni external dep
    → Megatron R3: both sides → CLOSED → did not land → gap REMAINS
    → DeepSpeed: NEITHER side → zero routing replay → gap REMAINS

  ★★★★★★★★ Recommended R3 path for RTX 4090:
    → SGLang #9499 (when merged) → inference-side routing capture → rollout generation
    → DeepSpeed RouterReplay (when implemented) → training-side routing replay
    → Together: complete R3 pipeline → SGLang generates → DeepSpeed replays
    → No VeOmni dependency → both SGLang and DeepSpeed are independent frameworks
    → Modular: SGLang can also feed Megatron or verl → universal inference provider
```

### Synthesis: What Each Framework Needs for Complete MoE GRPO

```
★★★★★★★★★ Complete MoE GRPO routing determinism requirements:

Requirement 1: Deterministic routing across rollout + training:
  → GRPO rollout routing must match training routing
  → Without: rollout uses different experts → hidden states differ → reward misaligned

Requirement 2: CUDA graph compatible routing:
  → CUDA graph requires static routing → dynamic top-k breaks capture
  → Static buffers → pre-allocated [max_tokens, topk] → fixed addresses

Requirement 3: Activation checkpointing backward recompute:
  → Backward recompute must use same routing as original forward
  → REPLAY_BACKWARD → pop from list → gradient consistency

Requirement 4: Response-only masking (R3):
  → Rollout records response-token routing → prompt tokens NOT recorded
  → Prompt tokens use native routing → response tokens use recorded routing

★★★★★★★★★ Per-framework status — updated with SGLang #9499:

| Requirement              | Megatron    | DeepSpeed   | verl VeOmni  | SGLang #9499  |
|--------------------------|-------------|-------------|--------------|---------------|
| Deterministic routing    | ★★★★★★★★ R2 | ✗ (freeze)  | ★★★★★★★ R2/R3| N/A (inference)|
| CUDA graph compatible    | ★★★★★★★★ static_buffer | ✗  | ★★★★★★★ (FSDP2 CG)| ✓ (Triton)|
| Act checkpoint backward  | ★★★★★★★★ REPLAY_BACKWARD| ✗ | ★★★★★★★★ (id-keyed)| N/A |
| Response-only masking    | ✗ (#4256 CLOSED)| ✗    | ★★★★★★★★ _replay_mask| capture side|
| R3 (rollout-side)        | ✗ (#4256 CLOSED)| ✗    | ★★★★★★★★ YES   | ★★★★★★★ capture side|
| Single-GPU viable        | ✗ (#5203)   | ★★★★★★★★ AutoEP EP=1| ✓ (FSDP2)| ✓ (Triton SM89)|
| External dependency      | None        | None        | veomni pip   | None          |

★★★★★★★★★ Recommended cross-framework R3 pipeline:

  SGLang (inference) → routing indices → DeepSpeed (training) → RouterReplay

  → SGLang #9499 provides routing capture during rollout generation
  → DeepSpeed RouterReplay (~206 LOC) provides routing replay during training
  → Together: complete R3 pipeline → no VeOmni dependency → modular and portable
  → This is the RTX 4090 MoE GRPO production path:
    SGLang rollout (deterministic + LoRA) → DeepSpeed AutoEP training (EP=1 + ZeRO-2)
```

---

## RTX 4090 Action Items

```
★★★★★★★★★ RTX 4090 RouterReplay action items — prioritized:

IMMEDIATE (0 LOC, CPU-only, no GPU needed):
  → [DONE] Document freeze_moe_router = 0 LOC solution → this note
  → [DONE] Document DeepSpeed RouterReplay equivalent design → existing note
  → [TODO] Comment on SGLang #9499 → suggest DeepSpeed RouterReplay consumption of routing info
  → [TODO] Comment on Megatron #5386 → suggest RTX 4090 single-GPU R3 path
  → [TODO] Comment on verl #6572 → mention RouterReplay gap in DeepSpeed

WHEN GPU ONLINE:
  → [TODO] Validate freeze_moe_router on RTX 4090 → DeepSpeed AutoEP EP=1 + Qwen3-MoE
  → [TODO] Profile MoE routing overhead → quantify top-k computation cost
  → [TODO] Test activation checkpointing + MoE routing consistency → backward recompute

TIER 2 CONTRIBUTION (after P10/P9/P6):
  → [TODO] Implement DeepSpeedRouterReplay (~206 LOC) → AutoEP TokenChoiceTopKRouter integration
  → [TODO] Write unit tests for 3 modes + EP=1
  → [TODO] Submit PR to DeepSpeed → Tier 2 contribution → fills AutoEP coverage gap

MONITOR:
  → SGLang #9499 → OPEN → routing replay info → when merged → enables R3 inference side
  → Megatron #4256 → CLOSED → no R3 → but could be re-opened with community demand
  → Megatron #5386 → OPEN → DSA/DSv4 Indexer Replay → extends replay concept
  → Megatron #5219 → single-GPU Muon → Final Review → if merged → RouterReplay on single GPU possible
  → DeepSpeed #8061 → overlap_comm NaN → MUST overlap_comm=False → affects CUDA graph future
  → DeepSpeed v0.19.2 → AutoEP merged → RouterReplay gap REMAINS → our contribution opportunity
  → verl VeOmni → R3 RouterReplay merged → reference for our DeepSpeed design
  → verl v0.8.0 → MoE RouterReplay via VeOmni → production-ready → BUT external veomni dep
```

---

## Key Findings Summary

★★★★★★★★★ Megatron RouterReplay #4168: 207 lines, 3-mode (RECORD/REPLAY_FORWARD/REPLAY_BACKWARD), static buffers for CUDA graph, global per-layer control, scores.gather(1, top_indices) for correct gradient computation -- CLEANEST design in OSS

★★★★★★★★★ #4256 RouterReplay R3 CLOSED: RL training integration did NOT land -- R3 (rollout-side routing capture + response-only masking) missing from Megatron -- same gap as DeepSpeed

★★★★★★★★★ DeepSpeed TokenChoiceTopKRouter: 188 lines, always-dynamic torch.topk (line 173), NO RouterReplay, NO determinism control, NO CUDA graph mechanism, NO static buffer -- confirmed coverage gap

★★★★★★★★★ freeze_moe_router = 0 LOC: immediate solution for RTX 4090 MoE GRPO -- self.gate.weight.requires_grad_(False) -- same input = same routing -- deterministic by design -- works for simple GRPO path -- NOT CUDA graph compatible -- NOT for trainable router

★★★★★★★★★ DeepSpeed RouterReplay ~206 LOC (revised estimate): Tier 2 contribution -- EP=1 simplification eliminates PP/SP/VPP complexity (~150 LOC savings vs verl) -- fills AutoEP coverage gap -- needed for CUDA graph future

★★★★★★★★★ CUDA graph incompatibility: DeepSpeed torch.topk creates new tensors each call (dynamic addresses) -- Megatron static_buffer[:num_tokens].copy_() writes into pre-allocated memory -- fundamental architectural difference

★★★★★★★★★ SGLang #9499: routing replay info OPEN -- inference-side routing capture -- modular design -- enables R3 pipeline with ANY training framework (DeepSpeed, Megatron, verl) -- no VeOmni dependency

★★★★★★★★★ verl VeOmni RouterReplay: 523 lines, R2/R3 modes, id-keyed positional indexing, _replay_mask for response-only masking -- MOST COMPLETE MoE GRPO routing determinism -- BUT requires external veomni pip package

★★★★★★★★★ Recommended R3 pipeline for RTX 4090: SGLang #9499 (inference capture) + DeepSpeed RouterReplay (training replay) = complete rollout-to-training bridge without VeOmni dependency

★★★★★★★★★ RTX 4090 pragmatic path: freeze_router NOW (0 LOC) -> Python cache when router training needed (~50 LOC) -> Full RouterReplay when CUDA graph (~206 LOC)

---

## Source Files Referenced

| File | Path | Lines | Key Content |
|------|------|-------|-------------|
| router_replay.py | _temp_megatron/megatron/core/transformer/moe/router_replay.py | 208 | RouterReplay + RouterReplayAction + static_buffer |
| router.py | _temp_megatron/megatron/core/transformer/moe/router.py | 738 | TopKRouter + Router base + moe_enable_routing_replay |
| moe_utils.py | _temp_megatron/megatron/core/transformer/moe/moe_utils.py | 1626 | topk_routing_with_score_function + CUDA graph infra |
| router_replay.md | _temp_megatron/docs/api-guide/router_replay.md | 187 | Official RouterReplay design document |
| ep_router.py | _temp_deepspeed/deepspeed/moe/ep_router.py | 188 | TokenChoiceTopKRouter + torch.topk direct |
| router_replay_utils.py | _temp_verl/verl/utils/megatron/router_replay_utils.py | 555 | PP/SP/VPP RouterReplay integration |
| router_replay.py (VeOmni) | _temp_verl/verl/utils/veomni/router_replay.py | 523 | VeOmniRouterReplay R2/R3 + id-keyed indexing |

## References

- Megatron RouterReplay source reading: notebook/projects/megatron-router-replay-source-reading.md
- Megatron RouterReplay design doc: _temp_megatron/docs/api-guide/router_replay.md
- DeepSpeed RouterReplay equivalent design: notebook/projects/deepspeed-router-replay-equivalent-design.md
- DeepSpeed AutoEP MoE source: notebook/projects/deepspeed-autoep-moe-source-reading.md
- DeepSpeed AutoEP RTX 4090 practical: notebook/projects/deepspeed-autoep-rtx4090-moe-practical.md
- SGLang #9499 routing replay info: https://github.com/sgl-project/sglang/issues/9499
- SGLang deterministic inference: notebook/projects/sglang-deterministic-inference-source-reading.md
- verl VeOmni RouterReplay: notebook/projects/verl-v08-veomni-flowgrpo-reading.md
- verl #6572 full determinism: https://github.com/verl-project/verl/pull/6572
- Megatron #4256 RouterReplay R3: https://github.com/NVIDIA/Megatron-LM/pull/4256
- Megatron #5203 single-GPU crash: https://github.com/NVIDIA/Megatron-LM/issues/5203
- DeepSpeed #8061 overlap_comm NaN: https://github.com/deepspeedai/DeepSpeed/issues/8061
- DeepSpeed #8068 gradient_clipping: https://github.com/deepspeedai/DeepSpeed/pull/8068
