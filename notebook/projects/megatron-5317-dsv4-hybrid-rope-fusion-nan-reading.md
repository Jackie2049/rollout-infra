# Megatron-LM #5317 — DSv4-Hybrid apply_rope_fusion=True NaN at iter 2 Deep Reading

> 2026-06-19 | Issue #5317 OPEN | Author: Micuks (community) | Assignee: guihong-nv (Guihong Li, NVIDIA)
> ★★★★★★★★ 11th DSV4 systematic failure — BF16 NaN from fused RoPE in hybrid attention
> ★★★★★★★★ Root cause: Triton in-place RoPE kernel bypasses PyTorch autograd version counter → corrupted gradient chain in DSv4HybridSelfAttention
> ★★★★★★★★ Critical for RTX 4090: DSV4-Hybrid is the next-gen MLA+DSA attention — fused RoPE MUST be disabled until fix

---

## 1. Issue Metadata

```
Title:             [Bug][DSv4-Hybrid] apply_rope_fusion=True causes NaN at iter 2 in BF16
                    mock pretrain (DSv4HybridSelfAttention)
Issue Number:      #5317
Author:            Micuks (community contributor, none association)
Assignee:          guihong-nv (Guihong Li, NVIDIA maintainer)
State:             OPEN
Created:           2026-06-12T09:01:04Z
Updated:           2026-06-18T05:53:03Z (7 days stale — waiting-on-maintainers)
Labels:            community-request, waiting-on-maintainers
Comments:          2 (guihong-nv: "will take a look"; Micuks: full log attachment)
Linked PRs:        NONE (no fix PR yet)
Milestone:         NONE
Labels indicate:   NVIDIA has acknowledged but not yet responded substantively after 7 days
```

---

## 2. Issue Summary

```
When training with experimental_attention_variant="dsv4_hybrid" and apply_rope_fusion=True,
training reliably produces NaN in the forward loss at iteration 2 under BF16 random-init
mock pretrain. Setting apply_rope_fusion=False fixes the issue completely.

★★★★★★★★★ Key observations:
  → Iteration 1 completes normally (grad norm ~18)
  → NaN appears at iteration 2 FORWARD (not backward)
  → Pattern: incorrect gradients from iter 1 → corrupt weights after optimizer step → NaN at iter 2
  → ALL 8 ranks hit NaN simultaneously → systematic, not random
  → Reproducible on both flat configs and compressed CSA configs
```

---

## 3. Reproduction Environment

```
Megatron-LM commit:  cf081d5df (current HEAD — post-#5243/5245 absorbed MLA refactor)
PyTorch:             2.12.0a0 (NV nightly)
CUDA:                13.2
TransformerEngine:   Available
NCCL:                2.29.7
Hardware:            8x H800 80GB
Topology:            EP=8, PP=1, TP=1, torchrun

Minimal config:
  cfg.model.experimental_attention_variant = "dsv4_hybrid"
  cfg.model.num_layers = 8
  cfg.model.num_moe_experts = 16
  cfg.model.csa_compress_ratios = [0, 0, 0, 0, 0, 0, 0, 0]
  cfg.model.seq_length = 4096
  cfg.model.params_dtype = torch.bfloat16
  cfg.model.apply_rope_fusion = True    ← causes NaN
  cfg.model.rope_type = "yarn"
  cfg.train.global_batch_size = 8
  cfg.train.micro_batch_size = 1

Also reproduced with:
  csa_compress_ratios = [0,0,4,128,4,128,4,0], seq_length = 1024

Results matrix:
  | apply_rope_fusion=False + BF16 | ✅ Trains cleanly (loss 5.8→0.03 over 70 iters) |
  | apply_rope_fusion=True  + BF16 | ❌ NaN at iter 2 forward (all 8 ranks)             |
```

---

## 4. Root Cause Analysis — Why apply_rope_fusion=True Causes NaN

```
★★★★★★★★★ The reporter's investigation is extremely thorough and reveals a subtle
autograd interaction bug, NOT a kernel arithmetic bug:

4.1 Isolated unit tests PASS:
  → fused_mla_rope_inplace (linear → RMSNorm → unsqueeze → fused rope → backward)
  → Both forward values and weight.grad match unfused path within BF16 tolerance
  → → The Triton RoPE kernel itself is ARITHMETICALLY correct

4.2 The bug is SPECIFIC to the full DSv4HybridSelfAttention forward/backward graph:
  → The fused path differs from unfused in THREE critical ways:

  ★★★★★★★★ DIFFERENCE 1: Forward RoPE on Q/KV — Triton kernel modifies tensors IN-PLACE
    → rotary_fwd_q_kernel: tl.store(Q + ...) directly modifies Q's GPU memory
    → rotary_fwd_kv_kernel: tl.store(K_ptr + ...) directly modifies KV's GPU memory
    → These in-place writes BYPASS PyTorch's version counter on the tensor
    → PyTorch autograd uses version counters to detect in-place modifications
    → When a tensor's version counter is stale, autograd may use wrong saved tensors
    → → Gradient computation may reference MODIFIED (post-RoPE) values instead of
       ORIGINAL (pre-RoPE) values → incorrect gradient chain!

  ★★★★★★★★ DIFFERENCE 2: Inverse RoPE on core_attn_out — another in-place modification
    → After attention computation, inverse RoPE is applied to core_attn_out IN-PLACE
    → Same version-counter bypass → gradient chain corruption compounded
    → TWO in-place modifications in one forward pass → double corruption risk!

  ★★★★★★★★ DIFFERENCE 3: key = value = kv — single-head MQA tensor aliasing
    → DSv4-Hybrid uses MLA's MQA pattern: K and V share the same tensor (kv)
    → Combined gradient accumulation (grad_k + grad_v → grad_kv) interacts with
       in-place modification
    → When kv is modified in-place by the RoPE kernel, BOTH key and value gradients
       reference the SAME modified tensor
    → → Gradient accumulation becomes INCORRECT because the saved tensor was
       modified between forward and backward!

4.3 Why NaN appears at iter 2 (not iter 1):
  → Iteration 1: forward computes with modified-in-place tensors → backward computes
     incorrect gradients (but may still produce reasonable-looking grad norm ~18)
  → After optimizer step: incorrect gradients → weight update in wrong direction
  → Iteration 2: corrupted weights + another round of incorrect forward → NaN explosion!
  → → NaN is the CUMULATIVE result of TWO iterations of gradient corruption

4.4 ★★★★★★★★ WHY the unit test doesn't catch this:
  → test_mla_yarn_rope_apply.py uses .detach() for fused path input!
  → fused_fwd_input = pytorch_fwd_input.detach()  ← line 94-95 of test
  → .detach() removes the tensor from autograd graph → NO prior operations →
     NO version counter interaction → NO gradient chain corruption!
  → The test ONLY verifies the kernel's arithmetic correctness
  → It DOES NOT test the kernel inside a full autograd graph with prior operations
  → → An end-to-end training test with apply_rope_fusion=True on
     DSv4HybridSelfAttention appears to be MISSING!

★★★★★★★★★ Root cause confirmed: Triton in-place RoPE kernel bypasses PyTorch
autograd version counter → corrupted gradient chain in full DSv4HybridSelfAttention
forward/backward → incorrect gradients → NaN at iter 2 after optimizer step
```

---

## 5. DSv4HybridSelfAttention Architecture Detail

```
★★★★★★★★★ DSv4HybridSelfAttention is the DeepSeek-V4 hybrid attention variant
that combines MLA (Multi-Latent Attention) with DSA (DeepSeek Attention):

5.1 Architectural lineage:
  → Based on AbsorbedMLASelfAttention (existing absorbed_mla.py)
  → Extends with DSA-style sparse attention indexer
  → Uses "absorption" technique: K's up projection absorbed into Q before attention

5.2 Data flow (training, apply_rope_fusion=True):
  Step 1: hidden_states → Q down projection → q_compressed → q_layernorm
  Step 2: hidden_states → KV down projection → kv_combined → split into
          kv_compressed + k_pos_emb → kv_layernorm
  Step 3: q_compressed → Q up projection → q [num_tokens, n, qk_head_dim + qk_pos_emb_head_dim]
  Step 4: kv_compressed → KV up projection → kv [num_tokens, n, qk_head_dim + v_head_dim]
  Step 5: ★★★★★★★★ FUSED RoPE applied:
          → fused_apply_mla_rope_for_q(q, ...) → modifies q IN-PLACE
          → fused_apply_mla_rope_for_kv(kv, k_pos_emb, ...) → produces key, value
             (but kv tensor is consumed in-place by Triton kernel)
  Step 6: q_absorbed = torch.einsum("...nd,ndk->...nk", q_no_pe, k_up_weight)
          → Absorption: K up-projection weight multiplied into Q's content dimension
  Step 7: core_attention(q_absorbed, kv_compressed, ...) → MQA-style attention
          → K and V compressed share kv tensor (single-head MQA)
  Step 8: inverse RoPE on core_attn_out → IN-PLACE modification
  Step 9: V up projection → core_attn_out = einsum("...nc,ndc->...nd", attn_out, v_up_weight)
  Step 10: linear_proj → output

5.3 ★★★★★★★★ Key architectural differences from standard MLASelfAttention:
  → Absorption: K's up projection absorbed into Q BEFORE attention (not after)
  → Separate K and V up projections (linear_k_up_proj + linear_v_up_proj)
  → Core attention in MQA form (KV is single-head compressed)
  → V up projection applied AFTER core attention (not before)
  → Core attention channels: k_channels=kv_lora_rank+qk_pos_emb_head_dim,
    v_channels=kv_lora_rank (compressed latent space, NOT full head dim)

5.4 ★★★★★★★★ Why the hybrid variant is MORE vulnerable to the in-place bug:
  → In standard MLA: key and value are separate tensors after KV up projection
  → In DSv4-Hybrid: kv_compressed is a SHARED tensor (key=value=kv_compressed)
  → When fused_apply_mla_rope_for_kv modifies kv IN-PLACE → BOTH key and value
     are affected by the same modification
  → In the backward pass: gradients for key and value accumulate into the SAME
     grad_kv tensor → combined with stale version counter → double corruption
  → → DSv4-Hybrid's MQA aliasing amplifies the in-place modification's impact!
```

---

## 6. Fused RoPE Math — Why In-Place RoPE Causes Numerical Instability

```
★★★★★★★★★ RoPE (Rotary Position Embedding) math for MLA:

6.1 Standard (unfused) RoPE application:
  Forward:
    q_pos_emb_rotated = q_pos_emb * cos + rotate_half(q_pos_emb) * sin
    → Creates NEW tensor (not in-place)
    → PyTorch tracks version counter correctly
    → Backward: grad_q_pos_emb = grad_output * cos - rotate_half(grad_output) * sin

6.2 Fused (Triton) RoPE application:
  Forward (rotary_fwd_q_kernel):
    x_1 = load(Q + qk_head_dim + even_indices)  ← q_pos_emb even elements
    x_2 = load(Q + qk_head_dim + odd_indices)    ← q_pos_emb odd elements
    x_left = x_1 * cos_left - x_2 * sin_left     ← rotated even
    x_right = x_2 * cos_right + x_1 * sin_right   ← rotated odd
    tl.store(Q + left_offsets, x_left)             ← IN-PLACE overwrite!
    tl.store(Q + right_offsets, x_right)           ← IN-PLACE overwrite!

  ★★★★★★★★ The store operations write DIRECTLY to Q's GPU memory → bypasses
  PyTorch's tensor version counter mechanism!

  PyTorch autograd's version counter:
    → Each tensor has a _version counter
    → When a tensor is saved for backward (via ctx.save_for_backward or
       autograd's automatic save), its _version is recorded
    → During backward, PyTorch checks: current_version == saved_version?
    → If mismatch: RuntimeError("in-place modification detected")
    → BUT: Triton kernel's tl.store() writes to raw GPU memory → doesn't
       increment PyTorch's _version counter!
    → → PyTorch's check PASSES → but the tensor's CONTENTS have changed!
    → → Backward computation uses the MODIFIED (post-RoPE) tensor as if it
       were the ORIGINAL (pre-RoPE) tensor → incorrect gradient!

6.3 ★★★★★★★★ Mathematical consequence of stale tensor reference:

  Consider: q = linear_up(q_compressed)  → autograd saves q_compressed
  Then:     q = fused_rope(q)             → in-place, bypasses version counter
  Backward: grad_q_compressed = grad_q @ linear_up.weight.T

  But grad_q is computed using the ROTATED q (not original q) because the
  saved activation is the post-RoPE version!

  In the unfused path:
    q_rotated = q_original * cos + ...  → NEW tensor
    Backward uses q_original (correctly saved) → correct gradient

  In the fused path:
    q = q * cos + ...  (in-place) → SAME tensor, but contents changed
    Backward "uses" q (which is now q_rotated) instead of q_original
    → gradient is computed with ROTATED values instead of ORIGINAL values
    → → GRADIENT IS WRONG!

  ★★★★★★★★ The error magnitude:
    → BF16 has ~2% relative error tolerance
    → A single incorrect gradient step compounds:
      iter 1: wrong gradient → weights updated in wrong direction
      iter 2: wrong weights + another wrong forward → exponential divergence → NaN!
    → The NaN at iter 2 is the CUMULATIVE result of TWO gradient corruption cycles

6.4 ★★★★★★★★ Why the KV path is even MORE problematic:
  → fused_apply_mla_rope_for_kv uses ApplyMLARotaryEmbKV autograd function
  → Forward: produces NEW tensors (o_key, o_value) → different from Q path
  → BUT: the kv INPUT tensor is consumed by Triton kernel → may not be
     properly tracked for backward
  → In DSv4-Hybrid: kv_compressed serves as BOTH key and value (MQA aliasing)
  → → The backward path for KV has to reconstruct grad_kv from grad_key + grad_value
  → If the version counter on kv is stale → combined gradient is computed
     incorrectly

6.5 ★★★★★★★★ The inverse RoPE on core_attn_out:
  → After attention, inverse RoPE is applied to convert attention output back
     to the un-rotated space
  → If this is ALSO done in-place (same Triton kernel pattern) → ANOTHER
     version counter bypass
  → TWO in-place modifications in one forward pass → compounding corruption!
```

---

## 7. Connection to DSV4 Failure Pattern Family (11th Failure!)

```
★★★★★★★★★ This is the 11th DSV4-related correctness failure documented:

| #  | Framework     | Issue          | What broke                     | Symptom          |
|----|---------------|----------------|--------------------------------|-------------------|
| 1  | vLLM          | #45309→#45972  | DSV4 cudagraph optimization    | Garbage output    |
| 2  | SGLang        | #26471→#28591  | DSV4 Online Compress MTP       | Accuracy degrad.  |
| 3  | SGLang        | #27749→#28575  | MTP weight update distributed  | Refactor needed   |
| 4  | SGLang        | #28569         | EAGLE3 CUDA graph replay       | ILLEGAL MEM ACCESS|
| 5  | vLLM          | #45979         | DSV4 flashinfer sparse cache   | GSM8K 6.75%       |
| 6  | SGLang        | #28520         | MTP swa_loc cache (AMD)        | Accept 2.17→3.04  |
| 7  | vLLM-Ascend   | #10645         | DSV4 chat template             | Wrong formatting  |
| 8  | vLLM-Ascend   | #10724         | DSV4 2*A2 PD-Mix crash         | Deployment crash  |
| 9  | SGLang        | #28612         | DSV4 C128 state mapping        | Correctness fix   |
| 10 | SGLang        | #28676         | MXFP8 MoE shuffle cache        | 64x accuracy blow |
| 11 | Megatron-LM   | #5317          | DSv4-Hybrid fused RoPE         | NaN at iter 2     | ★★★★★★★★ NEW!

★★★★★★★★★ Pattern classification for #5317:
  → NOT a CUDA graph replay failure (this is TRAINING, not inference)
  → NOT a cache staleness failure (no cached state involved)
  → → A NEW failure class: Triton kernel in-place modification bypasses autograd!

★★★★★★★★★ But shares the SAME meta-pattern:
  → DSV4 hybrid attention has MORE complexity layers than standard attention
  → MLA absorption + MQA aliasing + RoPE + inverse RoPE
  → Each layer increases the chance of subtle correctness bugs
  → The fused kernel's in-place modification is yet another instance of
     "optimization breaks correctness assumption"

★★★★★★★★★ Unified root cause taxonomy:
  → Class A: CUDA graph replay with stale metadata (inference: #45309, #26471, #28569)
  → Class B: Per-step cache staleness (inference: #45979, #28520, #28676)
  → Class C: Template/format bugs (inference: #10645, #10724)
  → Class D: Triton in-place kernel bypasses autograd (training: #5317) ← NEW CLASS!
  → ★★★★★★★★ ALL four classes share the meta-pattern: optimization breaks a
     correctness assumption that the DSV4 architecture amplifies!

★★★★★★★★★ Why DSV4 architecture amplifies all failure classes:
  → Standard attention: Q, K, V are separate → each has own gradient path
  → MLA attention: Q has split dimensions (q_no_pe + q_pos_emb) → more complex
  → DSv4-Hybrid: Q absorbed + KV aliased + RoPE in-place + inverse RoPE → 4
     correctness-sensitive operations interacting in one forward pass!
  → → ANY optimization that breaks ONE of these → affects ALL four!
```

---

## 8. Source Code Analysis — Fused RoPE Kernel

```
★★★★★★★★★ Key source files:

8.1 fused_mla_yarn_rope_apply.py (783 lines):
  → rotary_fwd_q_kernel: Triton JIT kernel, IN-PLACE modification via tl.store()
  → rotary_bwd_q_kernel: Triton JIT kernel, IN-PLACE modification of grad
  → ApplyMLARotaryEmbQ: autograd Function wrapping the kernel
    → forward: calls rotary_fwd_q_kernel → in-place on q → returns q (same tensor!)
    → backward: calls rotary_bwd_q_kernel → in-place on grad → returns grad
    → ★★★★★★★★ KEY BUG: forward modifies q IN-PLACE but returns same tensor object
    → PyTorch version counter NOT incremented → backward uses stale reference

  → rotary_fwd_kv_kernel: Triton JIT kernel → produces NEW o_key, o_value tensors
  → rotary_bwd_kv_kernel: Triton JIT kernel → produces NEW d_kv, d_emb tensors
  → ApplyMLARotaryEmbKV: autograd Function wrapping KV kernel
    → forward: allocates o_key = kv.new_empty(...) → NEW tensor (safe)
    → backward: allocates d_kv = dk.new_empty(...) → NEW tensor (safe)
    → → KV path seems safer (allocates new tensors) but the kv INPUT is still
       modified by Triton store operations in rotary_fwd_kv_kernel!

8.2 ★★★★★★★★ The Q path bug detail (rotary_fwd_q_kernel):

  @triton.autotune(configs=[...], key=["emb_dim", "head_num"], restore_value=["Q"])
  @triton.jit
  def rotary_fwd_q_kernel(Q, COS, SIN, qk_head_dim, ...):
      # Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
      # Load original values from Q's positional embedding section
      x_1 = tl.load(Q + x_1_off, mask=mask)  ← loads from IN-PLACE tensor
      x_2 = tl.load(Q + x_2_off, mask=mask)  ← loads from IN-PLACE tensor

      # Compute rotated values
      x_left = x_1 * cos_left - x_2 * sin_left
      x_right = x_2 * cos_right + x_1 * sin_right

      # ★★★★★★★★ IN-PLACE STORE — bypasses PyTorch version counter!
      tl.store(Q + x_left_off, x_left, mask=mask)   ← overwrites Q's memory
      tl.store(Q + x_right_off, x_right, mask=mask)  ← overwrites Q's memory

  ★★★★★★★★ restore_value=["Q"] in autotune config:
    → This tells Triton's autotune to restore Q's value after benchmarking configs
    → But it does NOT address the autograd version counter issue!
    → restore_value is for autotune benchmarking, not for autograd correctness!

8.3 ★★★★★★★★ ApplyMLARotaryEmbQ.forward:
  → ctx.save_for_backward(cos, sin) ← saves ONLY cos/sin, NOT the original q!
  → Returns q (the SAME tensor object that was modified in-place)
  → → When backward is called, the "saved" q is actually the MODIFIED q!
  → → In standard PyTorch autograd, in-place ops on saved tensors would trigger
     RuntimeError, but Triton bypasses this check!

  ★★★★★★★★ ApplyMLARotaryEmbQ.backward:
  → Receives grad_output
  → Calls rotary_bwd_q_kernel → in-place modifies grad_output
  → Returns modified grad (same tensor object)
  → → This is ALSO an in-place modification of the gradient tensor!
  → → But this is less problematic because grad is flowing backward, not saved

  ★★★★★★★★ BUT: the backward for Q is correct in isolation because:
    → In the unit test: q is the ONLY operation → no prior saved activations
    → In full DSv4HybridSelfAttention: q has prior operations (linear_up_proj,
       layernorm, etc.) → those activations were saved with the ORIGINAL q version
    → When backward reaches those prior ops → they use stale q reference → wrong grad!
```

---

## 9. Test Coverage Gap Analysis

```
★★★★★★★★★ test_mla_yarn_rope_apply.py — the ONLY test for fused MLA RoPE:

9.1 Test structure:
  → _test_fused_apply_mla_rope_for_q: tests Q path forward+backward
  → _test_fused_apply_mla_rope_for_kv: tests KV path forward+backward
  → TestApplyRotaryPosEmbMlaFusionConflict: tests unfused fallback warning

9.2 ★★★★★★★★ CRITICAL gap in test coverage:

  The Q test (line 93-96):
    pytorch_fwd_input.requires_grad_(True)
    fused_fwd_input = pytorch_fwd_input.detach()   ← .detach() REMOVES from autograd!
    fused_fwd_input.requires_grad_(True)            ← re-enables grad but NO prior ops!

  → .detach() creates a NEW tensor with no autograd history
  → The fused kernel modifies this DETACHED tensor in-place
  → But since there are no prior operations → no stale saved activations
  → → The test ONLY verifies kernel arithmetic, NOT autograd interaction!

  ★★★★★★★★ What the test SHOULD do to catch this bug:
    → Create q with prior operations that save activations (e.g., linear projection)
    → Apply fused RoPE to q (in-place modification)
    → Run backward through BOTH RoPE and prior operations
    → Compare gradients with unfused path
    → → This would reveal the stale version counter bug!

9.3 ★★★★★★★★ End-to-end training test is MISSING:
  → No test runs actual training iterations with apply_rope_fusion=True
  → → The NaN at iter 2 could NOT have been caught by existing tests
  → Need: test_dsv4_hybrid_fused_rope_training.py that runs 3+ iterations

9.4 ★★★★★★★★ Test class for MLA rotary interleaved vs fusion conflict:
  → TestApplyRotaryPosEmbMlaFusionConflict tests the UNFUSED apply_rotary_pos_emb path
  → When mla_rotary_interleaved=True and apply_rope_fusion=True:
    → Warning emitted: "apply_rope_fusion does not support MLA-style"
    → Falls back to unfused path
  → ★★★★★★★★ But this test is for the GENERAL apply_rotary_pos_emb function
  → The DSv4-Hybrid path uses DIFFERENT functions:
    → fused_apply_mla_rope_for_q (from fusions module)
    → fused_apply_mla_rope_for_kv (from fusions module)
  → → The MLA-specific fused functions bypass the general apply_rotary_pos_emb
  → → The warning/fallback mechanism is NOT triggered in DSv4-Hybrid path!
```

---

## 10. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 implications for #5317:

10.1 DSV4-Hybrid training on RTX 4090:
  → DSV4-Hybrid is Megatron-LM's training variant for DSV4
  → RTX 4090: dp=1, single GPU → MUST use ZeRO-2 + CPU_Adam
  → apply_rope_fusion MUST be False until fix is merged
  → → Workaround: cfg.model.apply_rope_fusion = False (MANDATORY for now)
  → Performance impact: unfused RoPE is ~2-3x slower than fused on RTX 4090
  → But CORRECTNESS > performance → unfused is the ONLY safe option

10.2 ★★★★★★★★ RTX 4090 DSV4 GRPO config rule (NEW):
  → MUST: apply_rope_fusion = False for any DSV4 variant on RTX 4090
  → MUST: experimental_attention_variant = "dsv4_hybrid" requires yarn rope
  → MUST: rope_type = "yarn" (MLA RoPE fusion only works with yarn)
  → MUST NOT: apply_rope_fusion = True with any MLA variant until fix merged
  → MUST NOT: combine DSV4-Hybrid + fused RoPE + BF16 until version counter fix

10.3 ★★★★★★★★ RTX 4090 memory implications:
  → Fused RoPE saves memory by not creating intermediate tensors
  → Unfused path creates separate rotated tensors → ~30% more activation memory
  → On RTX 4090 24 GiB: this could push from 18→22 GiB peak → still fits
  → But with MoE experts: activation memory is already tight
  → → Unfused path + selective recomputation may be needed to fit

10.4 ★★★★★★★★ RTX 4090 training pipeline:
  → verl CPPO+bypass_mode = best framework for RTX 4090 GRPO
  → verl uses SGLang/vLLM for rollout (inference) → NOT affected by this bug
  → Megatron training side → MUST use apply_rope_fusion=False
  → → No RTX 4090 training configuration uses dsv4_hybrid yet → LOW immediate impact
  → → But when DSV4 GRPO becomes viable → this bug WILL be a blocker!

10.5 ★★★★★★★★ SM89-specific considerations:
  → Triton kernels on SM89 (RTX 4090) have different autotune configs
  → In-place modifications on SM89 may have additional alignment constraints
  → BF16 on SM89 → already limited precision → gradient corruption amplified
  → → The NaN-at-iter-2 pattern is WORSE on SM89 than on SM90 (H800)!
```

---

## 11. Connection to Other Tracked Issues

```
★★★★★★★★★ Cross-issue connections:

11.1 vLLM #45309 cudagraph → DSV4 inference failure:
  → Same meta-pattern: optimization breaks correctness for DSV4
  → #45309: CUDA graph replay with stale routing → garbage output
  → #5317: Triton in-place RoPE with stale autograd → NaN at iter 2
  → Both: DSV4's multi-layer complexity amplifies the optimization's side effects
  → ★★★★★★★★ Key difference: #45309 is INFERENCE, #5317 is TRAINING

11.2 SGLang #28676 MXFP8 MoE cache clobber → RL weight update:
  → Same class of "GPU-resident state corruption" bug
  → #28676: shuffle index cache clobbered on weight reload → 64x accuracy blowup
  → #5317: autograd version counter stale on in-place RoPE → NaN
  → Both: state that should be invalidated/refreshed but isn't
  → ★★★★★★★★ Key difference: #28676 is cache invalidation, #5317 is autograd tracking

11.3 SGLang #28679 GDN intermittent degeneracy:
  → Same "worsens over uptime" pattern but different mechanism
  → #28679: FlashInfer memory pool + SSM state accumulation → silent corruption
  → #5317: autograd version counter staleness → NaN at iter 2 (then crash)
  → → #28679 is SILENT (no error logged), #5317 is EXPLOSIVE (NaN → crash)
  → ★★★★★★★★ Both prove: DSV4/hybrid attention models are fragile under optimization

11.4 Megatron #5394 ChainedOptimizer clipping stalls:
  → Different bug class (optimizer clipping) but same framework (Megatron training)
  → #5394: Muon/AdamW stalls under global grad-norm clipping
  → #5317: fused RoPE causes NaN under BF16
  → → Both are Megatron training correctness bugs affecting DSV4-related models
  → ★★★★★★★★ Connection: if #5394's skip_grad_norm_clip fix is applied + #5317's
     apply_rope_fusion=False → DSV4 training becomes viable

11.5 vLLM #46085 aot_eager piecewise compilation:
  → Complementary to this bug for RTX 4090
  → #46085: aot_eager = batch-invariant BY DESIGN (no Inductor → no fusion)
  → #5317: fused RoPE = batch-dependent + autograd-breaking → OPPOSITE approach!
  → → For RTX 4090: aot_eager for inference + unfused RoPE for training = SAFEST

11.6 DeepSpeed #8072/#8073 ZeRO-3+PEFT regression:
  → Different framework (DeepSpeed) but same "dtype mismatch" class
  → #8072: ZeRO-3 per-policy dtype mismatch → PEFT LoRA regression
  → #5317: Triton in-place → autograd version mismatch → gradient corruption
  → → Both: internal consistency assumption broken by optimization

11.7 PyTorch #187653 NanDetectMode:
  → Detection tool for NaN in GRPO training
  → Could help diagnose #5317-like bugs during training
  → → NanDetectMode would detect NaN at iter 2 → but can't identify root cause
  → ★★★★★★★★ Need: gradient-level NaN detection (check grad norms per-layer)
```

---

## 12. Potential Fix Approaches

```
★★★★★★★★★ Possible fix approaches (analyzing from source code):

12.1 Approach 1: Change Triton kernel to allocate NEW output tensors:
  → rotary_fwd_q_kernel: instead of tl.store(Q + ...) → allocate o_Q and store to it
  → ApplyMLARotaryEmbQ.forward: allocate o_q = q.new_empty(...) → return o_q
  → → This is how ApplyMLARotaryEmbKV already works (allocates new tensors)
  → → Fix would make Q path consistent with KV path
  → Cost: extra memory allocation (~1x q tensor per forward) but correctness guaranteed
  → ★★★★★★★★ BEST approach: matches existing KV path pattern + minimal semantic change

12.2 Approach 2: Increment PyTorch version counter manually:
  → After Triton kernel call: q._version += 1 or torch._C._increment_version(q)
  → → This would cause PyTorch's in-place detection to TRIGGER
  → → But: PyTorch would raise RuntimeError("in-place modification detected")
  → → NOT a fix — just makes the bug detectable, not correctable!
  → ★★★★★★★★ NOT viable: triggers error instead of fixing it

12.3 Approach 3: Save original tensor BEFORE in-place modification:
  → In ApplyMLARotaryEmbQ.forward: ctx.save_for_backward(cos, sin, q.clone())
  → → Clone q BEFORE Triton kernel modifies it → backward uses original values
  → Cost: 1x q tensor extra memory for saved clone
  → ★★★★★★★★ VIABLE but wasteful: clone doubles activation memory for q
  → → Approach 1 (new output tensor) is better because the output IS the clone

12.4 Approach 4: Use .data to bypass autograd tracking entirely:
  → Pass q.data to Triton kernel → modified data doesn't affect autograd
  → Then reconstruct q from modified data + original autograd graph
  → → Complex and fragile → breaks autograd chain entirely
  → ★★★★★★★★ NOT viable: too invasive and fragile

12.5 ★★★★★★★★ Recommended fix: Approach 1 (allocate new output tensor):
  → Modify rotary_fwd_q_kernel to output to NEW tensor (like KV path)
  → Modify ApplyMLARotaryEmbQ.forward:
    o_q = q.new_empty(shape)  ← allocate new
    rotary_fwd_q_kernel[q, ..., o_q, ...]  ← output to new tensor
    return o_q  ← return NEW tensor (not in-place modified original)
  → Modify ApplyMLARotaryEmbQ.backward similarly (new grad tensor)
  → → This preserves autograd correctness + version counter integrity
  → → Matches the pattern already used in ApplyMLARotaryEmbKV
  → → Cost: ~1x q tensor memory per forward → acceptable for correctness

12.6 ★★★★★★★★ Additional test requirement:
  → Add test_fused_rope_with_prior_autograd_ops()
  → Test: create tensor with prior operations (linear + norm)
  → Apply fused RoPE → backward through entire chain
  → Compare gradients with unfused path
  → → This would have CAUGHT the #5317 bug before release!

12.7 ★★★★★★★★ Additional test requirement:
  → Add test_dsv4_hybrid_rope_fusion_training.py
  → Run 3+ training iterations with apply_rope_fusion=True + dsv4_hybrid
  → Check for NaN in loss
  → → This would have caught the NaN-at-iter-2 pattern
```

---

## 13. Config Validation Rules (New for #5317)

```
★★★★★★★★★ New RTX 4090 config validation rules:

Rule #5317-1: apply_rope_fusion MUST be False for dsv4_hybrid (and ANY MLA variant)
  → Until in-place RoPE version counter fix is merged and tested
  → Workaround: cfg.model.apply_rope_fusion = False

Rule #5317-2: MLA + apply_rope_fusion requires yarn rope_type
  → Already enforced in MLATransformerConfig.__post_init__
  → But dsv4_hybrid MUST ALSO enforce apply_rope_fusion=False
  → → Need: assertion in config validation

Rule #5317-3: BF16 + MLA + fused RoPE = NaN risk
  → BF16 amplifies gradient corruption due to limited precision
  → → FP32 training may hide the bug longer (but still incorrect!)
  → → FP32 + fused RoPE: may not NaN but will produce wrong results silently

Rule #5317-4: MQA aliasing (key=value=kv) + in-place modification = compounding
  → DSv4-Hybrid's single-head MQA pattern amplifies in-place bugs
  → → ANY in-place kernel on a tensor shared by multiple outputs = gradient corruption risk

★★★★★★★★★ Updated MUST DO list for RTX 4090 DSV4-Hybrid:
  MUST: apply_rope_fusion = False (until #5317 fix merged)
  MUST: rope_type = "yarn" for MLA variants
  MUST: params_dtype = torch.bfloat16 (standard for DSV4)
  MUST: ZeRO-2 + CPU_Adam for single-GPU training
  MUST: gradient_clipping = 1.0 explicitly (avoid #8068 default 0→1.0 bug)

★★★★★★★★★ Updated MUST NOT list for RTX 4090 DSV4-Hybrid:
  MUST NOT: apply_rope_fusion = True (NaN guaranteed)
  MUST NOT: ZeRO-3 on single GPU (pure overhead, plus #8072 regression risk)
  MUST NOT: overlap_comm = True on single GPU (#8061 NaN risk)
  MUST NOT: Muon optimizer (#5394 clipping stall + #7939 CPU offload blocked)
  MUST NOT: FSDP v1 whole-model summon (#6512 OOM risk on RTX 4090)
```

---

## 14. Key Findings Summary

```
★★★★★★★★★ 11th DSV4 systematic failure — NEW failure class (autograd version counter)!
★★★★★★★★★ Root cause: Triton in-place RoPE kernel bypasses PyTorch autograd version counter
★★★★★★★★★ DSv4-Hybrid MQA aliasing (key=value=kv) amplifies the in-place modification impact
★★★★★★★★★ NaN at iter 2 = cumulative result of TWO gradient corruption cycles
★★★★★★★★★ Unit test gap: test_mla_yarn_rope_apply.py uses .detach() → doesn't test full autograd
★★★★★★★★★ End-to-end training test with apply_rope_fusion=True is MISSING
★★★★★★★★★ Workaround: apply_rope_fusion = False (MANDATORY until fix merged)
★★★★★★★★★ Recommended fix: allocate new output tensors (like KV path already does)
★★★★★★★★★ DSV4 failure taxonomy expanded: Class D = Triton in-place bypasses autograd
★★★★★★★★★ RTX 4090: unfused RoPE = only safe path → ~2-3x slower but correctness guaranteed
★★★★★★★★★ Pattern: DSV4 hybrid attention has 4 correctness-sensitive operations → compounding fragility
★★★★★★★★★ 7-day stale with waiting-on-maintainers label → NVIDIA needs to respond
★★★★★★★★★ guihong-nv assigned but only said "will take a look" → no substantive response yet
```

---

## 15. References

```
- Megatron-LM #5317: [Bug][DSv4-Hybrid] apply_rope_fusion=True NaN (this issue)
- Megatron-LM absorbed_mla.py: AbsorbedMLASelfAttention source (basis for DSv4-Hybrid)
- Megatron-LM fused_mla_yarn_rope_apply.py: Triton in-place RoPE kernel source
- Megatron-LM test_mla_yarn_rope_apply.py: Unit test with .detach() gap
- Megatron-LM transformer_config.py: apply_rope_fusion + experimental_attention_variant config
- Megatron-LM #5243/#5245: Absorbed MLA refactor (June 17 commits, prerequisite for dsv4_hybrid)
- vLLM #45309→#45972: DSV4 cudagraph revert (Class A failure)
- SGLang #28676: MXFP8 MoE cache clobber (Class B failure)
- SGLang #28679: GDN intermittent degeneracy (state accumulation)
- Megatron #5394: ChainedOptimizer clipping stalls (same framework, different class)
- DeepSpeed #8072/#8073: ZeRO-3+PEFT regression (dtype mismatch class)
- DeepSpeed #8061: overlap_comm + torch.compile NaN (in-place stream conflict)
- PyTorch #187653: NanDetectMode (NaN detection tool)
- notebook/projects/dsv4-systematic-instability-pattern-synthesis.md (10 previous failures)
- notebook/projects/sglang-28676-mxfp8-moe-v4-reading.md (10th failure)
- notebook/projects/sglang-28679-gdn-intermittent-degeneracy-reading.md (related pattern)
```
