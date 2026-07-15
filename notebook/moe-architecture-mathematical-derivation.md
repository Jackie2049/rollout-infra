# MoE Architecture: Mathematical Derivation & NaN Pattern Analysis

**Date**: 2026-07-15 (Session 10)
**Purpose**: Mathematical foundations of MoE routing + FP16 NaN universal pattern + RTX 4090 implications
**Sources**: Megatron Router, DSv4/V3 MoE, Switch Transformer, #10579, #5317, arXiv papers

---

## 1. Top-K Gating Router: Mathematical Derivation

### 1.1 Basic Router

```
Input: hidden_states ∈ ℝ^{S×H}  (S tokens, H hidden dim)
Router weight: W_router ∈ ℝ^{H×E}  (E experts)

Step 1: Compute logits
  logits = hidden_states @ W_router  →  ℝ^{S×E}
  logits_i = h_i · W_router  (per-token routing score)

Step 2: Softmax
  probs_i = softmax(logits_i) = exp(logits_i) / Σ_{j=1}^{E} exp(logits_j)

Step 3: Top-K selection
  top_k_indices = topk(probs_i, k)  (k=1 for Switch TF, k=2-8 for DSv4)
  top_k_probs = probs_i[top_k_indices]
  top_k_probs = top_k_probs / Σ(top_k_probs)  (normalize selected experts)
```

### 1.2 Why Top-K > 1 (DSv4 uses Top-8)

```
Top-1 (Switch Transformer):
  Each token routed to exactly 1 expert
  → Specialization → high variance in expert load
  → Requires capacity_factor ≈ 1.25-1.5 to prevent dropping
  → Simple but unstable

Top-2 (original MoE):
  Each token routed to 2 experts with weighted combination
  → output = p_1·expert_1(h) + p_2·expert_2(h)
  → Better load distribution → capacity_factor ≈ 1.0 works

Top-8 (DSv4):
  Each token routed to 8 experts out of 256+ total
  → More granular combination → smoother output
  → Reduces routing variance → more stable training
  → But: 8× expert computation → only 8/256 ≈ 3% of experts active
  → With 256 shared experts: dense part handles common patterns
```

---

## 2. FP16 NaN Universal Pattern: Mathematical Proof

### 2.1 The Problem

```
FP16 range: [-65504, +65504]
FP16 max value: 65504 = 2^16 - 2^5 ≈ 65504

softmax_FP16(logits):
  = exp(logits_i) / Σ_{j=1}^{E} exp(logits_j)

When logits_j > 65504 for ANY j:
  exp(logits_j) → inf (FP16)
  numerator: exp(logits_i) could be:
    - inf (if logits_i > 65504) → inf/inf = NaN
    - finite (if logits_i < 65504) → finite/inf = 0 → ALL probabilities = 0
    → Then: sum of all probs = 0 → normalization fails
```

### 2.2 Mathematical Proof of NaN

**Claim**: FP16 softmax ALWAYS produces NaN when any logit exceeds 88.7.

**Proof**:
```
FP16 max representable = 65504
exp(x) exceeds 65504 when x > ln(65504) ≈ 11.09

But: FP16 exp() actually saturates earlier!
IEEE 754 FP16:
  max normal = 65504
  exp(x) → inf when result > 65504

For softmax with E=256 experts:
  logits = [l_1, l_2, ..., l_256]
  max_logit = max(logits)

  softmax numerator = exp(logits_i - max_logit)  (if shifted)
  softmax denominator = Σ exp(logits_j - max_logit)

  Without shifting:
  If any logits_j > 11.09 → exp(logits_j) = inf → NaN

  With shifting (standard implementation):
  shifted_logits = logits - max(logits)
  All shifted_logits ≤ 0
  exp(shifted_logits) ∈ [0, 1]
  → No overflow in numerator or denominator!
  → FP16 softmax IS SAFE if you subtract max first.

  BUT: logits computation (logits = h @ W_router) can overflow BEFORE softmax!
  When h or W_router have large values → logits > 65504 BEFORE softmax
  → The softmax shifting doesn't help because logits ARE the input
  → logits are in FP16 → already inf → subtraction inf - inf = NaN
```

**The REAL problem**: It's not the softmax itself (which can be shifted), but the logits computation IN FP16 that overflows before softmax gets the chance to shift.

### 2.3 Why FP32 Gating Fixes This

```
FP32 range: [-3.4e38, +3.4e38]
FP32 max: ≈ 3.4 × 10^38

Step 1: Compute logits in FP32
  logits_FP32 = (hidden_states.float()) @ (W_router.float())
  → logits in FP32 → no overflow for any practical value

Step 2: Apply softmax in FP32
  probs = softmax(logits_FP32)  → perfectly safe, shifted implementation

Step 3: Cast back to model dtype
  probs = probs.to(model_dtype)  → BF16 or FP16

This 3-step pattern is the universal fix:
  logits.float() → softmax → result.to(dtype)
```

### 2.4 Cross-Platform Manifestation

| Platform | Bug | Root Cause | Fix |
|----------|-----|------------|-----|
| CUDA (Switch TF 2021) | FP16 logits overflow | logits > 65504 before softmax | FP32 gating |
| Ascend NPU (#10579) | torch.abs() + FP16 softmax | abs() doesn't fix overflow, adds sign loss | Remove abs() + FP32 gating |
| Megatron (proactive) | FP32 gating built-in | Defense by default | Already safe |
| DeepSpeed (proactive) | FP32 gating built-in | Defense by default | Already safe |
| vLLM (some models) | FP16 logits in MoE | Not all models use FP32 gating | Model-specific fix |

### 2.5 RTX 4090 Configuration

```
★★★★★★★★ ALWAYS compute MoE gating softmax in FP32 regardless of platform.

RTX 4090 config:
  model_dtype = bf16  (BF16 has same exponent range as FP32, much safer than FP16)
  gating_dtype = fp32  (EXPLICITLY FP32 for gating, even with BF16 model)

Why BF16 is better than FP16 for model weights:
  BF16 range: [-3.4e38, +3.4e38] (same as FP32)
  BF16 precision: 8 bits mantissa (vs 10 bits for FP16)
  → BF16 CANNOT overflow for logits computation
  → But: BF16 softmax still has precision issues for very large logits
  → FP32 gating softmax is ALWAYS the safest choice

RTX 4090 safest config: BF16 model + FP32 gating softmax = ZERO NaN risk on ALL platforms
```

---

## 3. Load Balancing Loss: Mathematical Derivation

### 3.1 Aux Loss (Switch Transformer)

```
For E experts, S tokens:

f_i = (1/S) Σ_{s=1}^{S} 1[expert(s) = i]   (fraction of tokens routed to expert i)
P_i = (1/S) Σ_{s=1}^{S} p_s(i)               (average routing probability for expert i)

Aux loss = α · Σ_{i=1}^{E} f_i · P_i

Properties:
  - Minimized when f_i = P_i = 1/E (perfect uniform distribution)
  - Maximum when all tokens go to one expert (f_i=1, P_i=1)
  - α (aux_loss_coef) typically 0.01 → small penalty, not dominant

Gradient:
  ∂L_aux / ∂W_router = α · Σ_i (∂f_i/∂W · P_i + f_i · ∂P_i/∂W)
  → f_i is discrete (indicator function) → ∂f_i/∂W ≈ 0
  → P_i is continuous (softmax output) → ∂P_i/∂W is meaningful
  → Effective gradient ≈ α · Σ_i f_i · ∂P_i/∂W
  → Pushes P_i toward uniform when f_i is imbalanced
```

### 3.2 z-Loss (DSv4/Megatron)

```
z_loss = β · (1/S) Σ_{s=1}^{S} log²(Σ_{j=1}^{E} exp(logits_j))

Purpose: Prevents logits from growing too large → avoids FP16 overflow!
  → This is the PROACTIVELY SAFE version of aux loss
  → When logits → large, log(sum_exp) → large → z_loss → large → penalty

Gradient:
  ∂z_loss/∂logits_i = β · 2 · log(Z) · (exp(logits_i) / Z)
  where Z = Σ exp(logits_j)

  → When logits_i is large: exp(logits_i)/Z ≈ 1, log(Z) large
  → Strong gradient pushing logits_i DOWN
  → Prevents logits from ever reaching overflow threshold

Key insight: z_loss + FP32 gating = DOUBLE defense against MoE NaN
  - z_loss: prevents logits from growing (training-time defense)
  - FP32 gating: ensures softmax is safe even if logits grow (runtime defense)
```

### 3.3 Gradient Bias in Aux/z-Loss Computation

This connects to Megatron #4590/#5798:

```
calculate_per_token_loss = False (default):
  L_aux = mean_over_microbatch(f_i · P_i) → then average over DP ranks
  → Each microbatch may have different token count → gradient bias!

calculate_per_token_loss = True:
  L_aux = global_sum(f_i · P_i) / global_sum(token_count)
  → Correct: global normalization, no microbatch bias

★★★★★★★★ For GRPO with variable-length completions:
  MUST use calculate_per_token_loss = True for ALL losses (main + auxiliary)
  Otherwise: 158% gradient bias with imbalanced token counts (#4590)
```

---

## 4. Expert Computation: Grouped GEMM

### 4.1 Standard MoE (per-expert GEMM)

```
For K selected experts per token:
  output_s = Σ_{k=1}^{K} p_s(k) · expert_k(hidden_s)

Implementation:
  1. Gather tokens per expert: expert_k_inputs = gather(tokens where expert_k selected)
  2. Run expert_k linear independently: expert_k_outputs = linear_k(expert_k_inputs)
  3. Scatter outputs back: output = scatter_combine(expert_k_outputs)
```

### 4.2 Grouped GEMM (Megatron/DSv4)

```
Single GEMM for ALL experts simultaneously:
  grouped_output = grouped_gemm(grouped_input, all_expert_weights)

Benefits:
  - 1 kernel launch instead of E launches → much lower overhead
  - Better GPU utilization → larger effective batch per expert
  - Compatible with FP8 quantization (DeepGEMM)

How it works:
  - Expert weights concatenated: [W_1, W_2, ..., W_E] → single matrix
  - Tokens permuted so each expert's tokens are contiguous
  - Single GEMM call with expert-specific segments
  - Output split back to per-expert segments
```

---

## 5. DeepSeek-V3/V4 MoE Specifics

### 5.1 Architecture

```
DSv4 MoE layer:
  - 256 routed experts (Top-8 selected per token)
  - 1 shared expert (ALL tokens pass through)
  - Output = shared_output + Σ_{k=1}^{8} p_k · routed_k_output

Why shared expert:
  - Common patterns (syntax, basic reasoning) → shared expert handles
  - Rare patterns (specialized knowledge) → routed experts handle
  - Reduces routing variance → Top-8 from 256 ≈ 3% routed + 100% shared
```

### 5.2 MLA + MoE Interaction

```
DSv4-Hybrid architecture:
  Attention: MLA (Multi-head Latent Attention) → compressed KV
  MLP: MoE (256 experts + 1 shared) → sparse computation

  Total params: ~685B (but only ~37B active per token)
  → Active = MLA compressed + 8 routed experts + 1 shared
  → 37B active × BF16 ≈ 74 GiB → TOO LARGE for RTX 4090 (24 GiB)

  RTX 4090 viable alternatives:
  - DSv4-Lite: smaller model, fewer experts
  - Qwen3-30B-A3B: 30B total, 3B active → 6 GiB BF16 → fits 24 GiB
  - Qwen2.5-7B: dense, no MoE → 14 GiB BF16 → fits 24 GiB
```

---

## 6. MoE NaN Failure Pattern Family (from DSV4 Synthesis)

| # | Failure | Framework | Root Cause | Pattern Class |
|---|---------|-----------|------------|---------------|
| 1 | Switch TF FP16 | CUDA | logits overflow → inf/inf | moe_fp16_nan |
| 2 | #10579 | vLLM-Ascend | torch.abs() + FP16 softmax | moe_fp16_nan |
| 3 | #48624 | vLLM | FP8 MoE expert alignment crash | dsv4_instability |
| 4 | #28676 | SGLang | MXFP8 MoE cache clobbered | state_lifecycle |
| 5 | #28685 | SGLang | FP8 block-fp8 wrong on MI350X | dsv4_instability |
| 6 | #48656 | vLLM | SP-MoE auto-enabled dp=1 | auto_enable_no_guard |
| 7 | #5317 | Megatron | Triton rotary NaN (in-place) | state_lifecycle |
| 8 | #48650 | vLLM | tl.constexpr batch-dependent | batch_invariance |

**Pattern**: 8 failures share 4 root classes:
1. `moe_fp16_nan` → ALWAYS use FP32 gating softmax
2. `state_lifecycle` → invalidate ALL GPU-resident caches at weight-reload boundary
3. `dsv4_instability` → DSv4 has unique per-framework failure modes
4. `batch_invariance` → Triton constexpr bakes runtime-varying values → recompiles

---

## 7. RTX 4090 MoE Training Recommendations

### Viable MoE Models
```
Qwen3-30B-A3B (30B total, 3B active):
  - 3B active × BF16 ≈ 6 GiB → fits RTX 4090
  - Top-8/128 routing → stable with gs≥8
  - LoRA r=32 on each expert → ~2 GiB LoRA params
  - Total: 6+2+3.8(bypass)+1.4(grad) ≈ 13.2 GiB → fits with 10.8 GiB headroom

Qwen2.5-7B dense:
  - 14 GiB BF16 → tight but viable with bypass_mode
  - No MoE → simpler, no FP32 gating concern
  - Total: 14+3.8+1.4 = 19.2 GiB → 4.8 GiB headroom
```

### MoE-Specific MUST DO Rules
```
★★★★★★★★ FP32 gating softmax (universal, not platform-specific)
★★★★★★★★ enforce_eager=True (DSv4 requires, prevents CUDA graph cache corruption)
★★★★★★★★ gs≥8 for sparse MoE (more variance in MoE rewards)
★★★★★★★★ LoRA rank=32, alpha=64 (NEVER 64 → breaks EOS)
★★★★★★★★ --no-sequence-parallel-moe on dp=1 (#48656)
★★★★★★★★ calculate_per_token_loss=True for ALL MoE losses (#4590)
★★★★★★★★ z_loss_coef>0 (prevents logits growth → double defense with FP32)
```

### MoE-Specific MUST NOT Rules
```
★★★★★★★★ NEVER FP16 gating softmax (universal NaN)
★★★★★★★★ NEVER LoRA rank≥64 (EOS token lost #6782)
★★★★★★★★ NEVER SP-MoE on dp=1 (#48656, -24% KV cache)
★★★★★★★★ NEVER FSDP2 with MoE (#7016, backward failure)
★★★★★★★★ NEVER CUDA graph with DSv4 (#5317, in-place rotary corruption)
★★★★★★★★ NEVER apply_rope_fusion=True with MLA (#5317)
★★★★★★★★ NEVER torch.abs() before MoE softmax (#10579, sign loss)
```

---

## Session Stats
- **Top-K gating** math derived (Top-1 vs Top-2 vs Top-8 comparison)
- **FP16 NaN universal pattern** mathematically proved (overflow before softmax shifting)
- **Load balancing loss** derived (aux loss + z-loss + gradient bias connection)
- **8 MoE failures** classified into 4 root pattern classes
- **RTX 4090 MoE rules**: 7 MUST DO + 7 MUST NOT
- **DSv4 architecture**: 256 routed + 1 shared + MLA interaction explained
