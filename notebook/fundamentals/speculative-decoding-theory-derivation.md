# Speculative Decoding Theory — Mathematical Derivation for LLM Inference

> 2026-06-19 | Algorithm Theory | Draft-then-Verify, Acceptance Rate Math, 6 SGLang Methods, DSV4 MTP
> ★★★★★★★★ RTX 4090 Context: decode is 99.4% memory-bound → speculative decoding utilizes "free" GPU compute → BUT draft quality (KL divergence) determines everything — untrained draft = NEGATIVE speedup (0.13-0.76x)!
> ★★★★★★★★ SGLang #28569: EAGLE3 CUDA graph crash (illegal memory access when batch shrinks) → spec decode on hybrid/sliding-window models is unstable → RTX 4090 MUST use topk=1 chain topology for CUDA graph safety
> ★★★★★★★★ DeepSeek-V4 MTP: Online C128 compress + MTP → state mapping lifecycle bug → #28591 revert OPEN → #28612 fix OPEN → DSV4 speculative decoding NOT production-ready yet!

---

## Table of Contents

1. [The Memory-Bound Problem](#1-the-memory-bound-problem)
2. [Draft-then-Verify Mathematical Derivation](#2-draft-then-verify-mathematical-derivation)
3. [Acceptance Rate Mathematics](#3-acceptance-rate-mathematics)
4. [Approach Taxonomy](#4-approach-taxonomy)
5. [SGLang's 6 Speculative Decoding Methods](#5-sglangs-6-speculative-decoding-methods)
6. [vLLM Speculative Decoding Support](#6-vllm-speculative-decoding-support)
7. [DeepSeek V4 MTP as Speculative Decoding](#7-deepseek-v4-mtp-as-speculative-decoding)
8. [RTX 4090 Implications for GRPO Training](#8-rtx-4090-implications-for-grpo-training)
9. [Critical Issues Tracker](#9-critical-issues-tracker)
10. [References](#10-references)

---

## 1. The Memory-Bound Problem

### 1.1 Autoregressive Decode Bottleneck

Each decode step produces exactly 1 token:

```
Input: "The capital of France is"
Step 1: forward_pass() → "Paris"     ← 1 token per forward pass
Step 2: forward_pass() → " and"
Step 3: forward_pass() → " it"
...

Latency scales linearly with output length:
  7B model, B=1: ~10ms/token
  70B model, B=1: ~50ms/token
```

### 1.2 Why Decode is Memory-Bound

The key mathematical insight: for a single decode step, the transformer's arithmetic intensity is catastrophically low.

**Single-step decode FLOPs vs HBM IO**:

$$\text{FLOPs}_{\text{decode}} = 2 \times d \times V \quad \text{(one matmul: hidden} \rightarrow \text{vocab)}$$

$$\text{HBM IO}_{\text{decode}} = 2 \times d \times V \quad \text{(weight read)}$$

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{IO}} = 1.0 \quad \text{← always memory-bound!}$$

On A100 (312 TFLOPS FP16, 2 TB/s HBM): roofline crossover at AI = 156. Decode AI = 1.0 → **99.4% of compute capacity is wasted**.

**The key observation**: GPU has massive compute headroom during decode, but is throttled by memory bandwidth. Speculative decoding exploits this compute headroom by "spending" it on verifying multiple tokens at once.

### 1.3 Why Verify is "Almost Free"

For K draft tokens, verification is a single batched matmul:

$$H_{\text{batch}} = [h_1, h_2, \ldots, h_K] \in \mathbb{R}^{K \times d}$$

$$\text{probs} = H_{\text{batch}} \times W \in \mathbb{R}^{K \times V}$$

$$\text{HBM IO}_{\text{verify}} = 2dV + 2Kd \quad \text{(weights read once! + hidden vectors)}$$

Since $K \ll V$ (typically $K = 3-5$, $V = 32K-128K$):

$$\text{HBM IO}_{\text{verify}} \approx \text{HBM IO}_{\text{single decode}} = 2dV$$

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**The foundational insight**: Verifying K tokens costs approximately the same HBM IO as verifying 1 token. The weight matrix is read once regardless of K. This is why speculative decoding works — it turns wasted compute capacity into useful work.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 2. Draft-then-Verify Mathematical Derivation

### 2.1 The Algorithm

Given:
- Target model $p(x)$ — the large, accurate model
- Draft model $q(x)$ — the small, fast model
- Draft length $K$ — number of candidate tokens

**Step 1: Draft Generation**

The draft model generates K tokens autoregressively:

$$x_1 \sim q(x | \text{context}), \quad x_2 \sim q(x | \text{context}, x_1), \quad \ldots, \quad x_K \sim q(x | \text{context}, x_1, \ldots, x_{K-1})$$

**Step 2: Verification**

The target model processes all K+1 positions (K draft tokens + 1 bonus position) in a single forward pass:

$$p(x_i | \text{context}, x_1, \ldots, x_{i-1}) \quad \text{for } i = 1, 2, \ldots, K$$

This is the "almost free" step — the target model reads its weights once and computes logits for all positions simultaneously.

**Step 3: Accept/Reject via Rejection Sampling**

For each draft token $x_i$ (in order):

$$\text{Accept } x_i \quad \text{if} \quad r < \min\left(1, \frac{p(x_i | x_{<i})}{q(x_i | x_{<i})}\right) \quad \text{where } r \sim \text{Uniform}(0, 1)$$

If accepted: continue to next position.

If rejected:
1. Output a **corrected token** sampled from the residual distribution:

$$x_{\text{corrected}} \sim \text{Normalize}\left(\max(0, p(x | x_{<i}) - q(x | x_{<i}))\right)$$

2. **Discard all subsequent draft tokens** ($x_{i+1}, \ldots, x_K$) — their KV cache entries are freed.

**Step 4: Bonus Token**

If all K draft tokens are accepted, sample one bonus token from the target distribution:

$$x_{\text{bonus}} \sim p(x | \text{context}, x_1, \ldots, x_K)$$

This gives $K + 1$ tokens from a single iteration — the maximum possible output.

### 2.2 Why Rejection Sampling Preserves Distribution

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**The distribution-preserving guarantee**: The output distribution of speculative decoding is **exactly** the target distribution $p(x)$, with zero approximation error. This is proven via the rejection sampling identity.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

**Proof sketch** (Leviathan et al., ICML 2023):

For a single position, the probability of outputting token $t$ is:

$$P(\text{output} = t) = P(\text{accept } t) + P(\text{reject some } x, \text{resample gives } t)$$

The acceptance probability for token $t$ (when draft proposes $t$):

$$P(\text{accept } t | \text{draft proposed } t) = \min\left(1, \frac{p(t)}{q(t)}\right)$$

The probability that draft proposes $t$ AND it is accepted:

$$P(\text{draft proposes } t) \times P(\text{accept } t) = q(t) \times \min\left(1, \frac{p(t)}{q(t)}\right) = \min(q(t), p(t))$$

The rejection-resample probability:

$$P(\text{reject draft } x, \text{resample gives } t) = P(\text{reject}) \times P(\text{resample gives } t)$$

$$= \left(1 - \sum_x \min(q(x), p(x))\right) \times \frac{\max(0, p(t) - q(t))}{\sum_x \max(0, p(x) - q(x))}$$

$$= \left(\sum_x \max(0, p(x) - q(x))\right) \times \frac{\max(0, p(t) - q(t))}{\sum_x \max(0, p(x) - q(x))}$$

$$= \max(0, p(t) - q(t))$$

**Combining**:

$$P(\text{output} = t) = \min(q(t), p(t)) + \max(0, p(t) - q(t)) = p(t) \quad \text{← exact!}$$

This proves the output distribution equals $p(x)$ regardless of draft quality. Even a terrible draft model produces the same distribution — just slower (more rejections).

### 2.3 Greedy Decoding Simplification

When temperature = 0 (greedy decoding), rejection sampling simplifies to argmax comparison:

$$\text{Accept } x_i \quad \text{iff} \quad \text{argmax}_x \, p(x | x_{<i}) = x_i$$

This is equivalent to: the target model's top-1 prediction matches the draft token. Greedy verification is deterministic and typically has higher acceptance rates than stochastic sampling.

---

## 3. Acceptance Rate Mathematics

### 3.1 Per-Token Acceptance Rate

The acceptance rate at position $i$ depends on the distributional overlap between $p$ and $q$:

$$\alpha_i = P(\text{accept at position } i) = \sum_x \min(p(x | x_{<i}), q(x | x_{<i}))$$

This is the **total variation overlap**: $\alpha = 1 - \text{TV}(p, q)$ where $\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$.

**Key factors affecting $\alpha$**:

| Factor | Effect on $\alpha$ | Explanation |
|--------|-------------------|-------------|
| Draft quality (KL divergence) | KL < 1 → $\alpha$ > 0.8; KL > 5 → $\alpha$ < 0.5 | Distributional closeness determines everything |
| Temperature | Higher T → higher $\alpha$ | Flattened distributions overlap more |
| Position decay | $\alpha_i$ decreases with $i$ | Draft error accumulates; deeper positions have lower $\alpha$ |
| Draft-target architecture match | Same family → higher $\alpha$ | LLaMA draft + LLaMA target > GPT-2 draft + LLaMA target |

### 3.2 Expected Tokens Per Iteration

Assuming per-position acceptance rate $\alpha$ (simplified, assuming independence):

$$E[\text{accepted tokens}] = \sum_{i=1}^{K} \alpha^i = \frac{\alpha(1 - \alpha^K)}{1 - \alpha}$$

Including the bonus token (always produced if all K accepted):

$$E[\text{total output tokens per iteration}] = 1 + \sum_{i=1}^{K} \alpha^i = 1 + \frac{\alpha(1 - \alpha^K)}{1 - \alpha}$$

**Numerical examples**:

| $\alpha$ | K=3 | K=5 | K=8 | K=16 | Limit ($K \to \infty$) |
|----------|------|------|------|------|------------------------|
| 0.50 | 1.88 | 1.97 | 1.99 | 2.00 | $1/(1-0.5) = 2.0$ |
| 0.70 | 2.18 | 2.41 | 2.57 | 2.66 | $1/(1-0.3) = 3.33$ |
| 0.80 | 2.44 | 2.69 | 2.88 | 3.00 | $1/(1-0.2) = 5.0$ |
| 0.90 | 2.71 | 3.10 | 3.46 | 3.89 | $1/(1-0.1) = 10.0$ |
| 0.95 | 2.86 | 3.36 | 3.85 | 4.57 | $1/(1-0.05) = 20.0$ |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**Key insight**: The limit $K \to \infty$ gives $E[\text{output}] = 1/(1-\alpha)$ tokens per iteration. For $\alpha = 0.8$, the theoretical ceiling is 5x speedup. For $\alpha = 0.9$, it is 10x. But practical K is limited by draft overhead and position decay, so real speedup is 2-4x.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 3.3 Speedup Formula Including Draft Overhead

The total wall-clock time per iteration:

$$T_{\text{spec}} = K \times t_{\text{draft}} + t_{\text{verify}} + t_{\text{sample}}$$

The equivalent standard AR time for the same output:

$$T_{\text{AR}} = E[\text{output}] \times t_{\text{target}}$$

**Speedup**:

$$S = \frac{E[\text{output}] \times t_{\text{target}}}{K \times t_{\text{draft}} + t_{\text{verify}}}$$

Define the **latency ratio** $\beta = t_{\text{draft}} / t_{\text{target}}$. Since $t_{\text{verify}} \approx t_{\text{target}}$ (memory-bound, single weight read):

$$S = \frac{1 + \alpha(1 - \alpha^K)/(1 - \alpha)}{1 + K\beta}$$

**Optimal K** (approximate):

$$K^* \approx \sqrt{\frac{1-\alpha}{\alpha\beta}} - \frac{1}{\beta}$$

**Numerical speedup examples**:

| $\alpha$ | $\beta$ | Optimal K | Speedup |
|----------|---------|-----------|---------|
| 0.80 | 0.1 | 4 | 2.4x |
| 0.80 | 0.2 | 3 | 1.9x |
| 0.90 | 0.1 | 9 | 5.0x |
| 0.90 | 0.2 | 4 | 3.4x |
| 0.85 | 0.05 | 7 | 4.7x |
| 0.70 | 0.1 | 3 | 1.8x |

### 3.4 RTX 4090 Acceptance Rate Reality

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**RTX 4090实测数据** (from benchmark):

| Draft | Params | Latency Ratio $\beta$ | KL Divergence | Acceptance $\alpha$ | K=1 Speedup |
|-------|--------|----------------------|---------------|---------------------|-------------|
| tiny_0.5M | 4.2M | 0.54 | 21.45 | 18% | **0.76x (NEGATIVE!)** |
| small_2M | 9.0M | 0.89 | 21.46 | 12% | **0.59x (NEGATIVE!)** |
| medium_7M | 19.6M | 0.91 | ~0 | 100% | 1.05x |

**Untrained draft models produce NEGATIVE speedup**: KL > 5 → $\alpha$ < 0.22 → draft overhead exceeds acceptance benefit. The bottleneck is NOT draft model size, but draft model QUALITY (KL divergence).

**The viability threshold**: $\beta \geq 0.5$ → speculative decoding has NO benefit. RTX 4090 requires $\beta \leq 0.2$ → draft must be at least 5x faster than target. For 7B target: draft $\leq$ 1.4B → $\beta \approx 0.2$.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 4. Approach Taxonomy

### 4.1 Classification Framework

All speculative decoding approaches can be classified by:

1. **Draft source**: Where do candidate tokens come from?
2. **Draft topology**: Chain (sequential) or Tree (branching)?
3. **Feature level**: Token-level or Feature-level draft?
4. **Overhead**: Extra compute, memory, and parameters

| Approach | Draft Source | Topology | Feature Level | Extra Params | Acceptance |
|----------|-------------|----------|---------------|-------------|------------|
| **Draft Model** | Separate small model | Chain | Token-level | Full model | 0.80-0.95 |
| **Self-Speculative** | Same model, skip layers | Chain | Feature-level | 0 | 0.80-0.90 |
| **EAGLE/EAGLE3** | Lightweight draft head | Tree/Chain | Feature-level | ~0.5GB | 0.85-0.92 |
| **Medusa** | MLP prediction heads | Tree | Token-level | ~1% | 0.50-0.65 |
| **MTP (DeepSeek)** | Sequential ngram heads | Chain | Feature-level | ~2%/depth | 0.70-0.85 |
| **N-gram** | Context pattern matching | Chain | Lookup | 0 | 0.30-0.50 |
| **DFlash** | Non-causal draft block | Block | Feature-level | Full model | 0.80-0.90 |
| **Lookahead** | Jacobian iteration | Chain | Feature-level | 0 | 0.60-0.80 |
| **Suffix Decoding** | Suffix tree matching | Chain | Lookup | 0 | 0.40-0.60 |

### 4.2 Draft Model Approach

The original speculative decoding approach (Leviathan et al., 2023):

```
Architecture: Independent small model (e.g., LLaMA-7B as draft for LLaMA-70B)
  → Full forward pass for each draft token → autoregressive K steps
  → Requires: same vocab_size, same tokenizer, compatible architecture

Pros:
  → Simple to implement → just load two models
  → High acceptance rate when draft is from same model family
  → No modification to target model needed

Cons:
  → Full model overhead → latency ratio β = draft_size/target_size
  → Memory: need to store entire draft model weights
  → Vocab mismatch between architectures → zero acceptance
  → Optimal draft size ≈ 1/10 of target (β ≈ 0.1)

RTX 4090 viability:
  → 7B target + 0.7B draft → β ≈ 0.1 → feasible but ~1.4GB extra memory
  → → Better alternatives: EAGLE (lighter) or N-gram (zero overhead)
```

### 4.3 Self-Speculative Decoding

Use the SAME model but with skipped layers as draft:

```
Architecture: Target model layers 1..N → draft uses layers 1..N/2
  → Early exit: run only first half of transformer layers → fast draft
  → Verify: run all layers → full verification

  Mathematically:
    q(x) = softmax(lm_head(layer_{N/2}(hidden)))
    p(x) = softmax(lm_head(layer_N(hidden)))

  Acceptance rate ≈ 0.80-0.90 (same model, same representation space)

Pros:
  → No extra model → zero memory overhead
  → Same architecture → distribution naturally close

Cons:
  → Early exit quality varies → some tasks have low α
  → KV cache complexity → draft layers need separate cache management
  → Limited to specific model architectures that support early exit
```

### 4.4 EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**EAGLE is the current best speculative decoding approach for production**: feature-level draft (not token-level) → uses target model's hidden states → avoids the "token bottleneck" → acceptance rate 0.85-0.92 → vLLM and SGLang recommended default.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
Architecture: Lightweight draft model operating on FEATURE space
  → Input: target model's last hidden state h_t + token embedding e(x_t)
  → → Concatenation: [h_t, e(x_t)] → Linear projection → draft hidden state
  → → → Draft model: shallow transformer (1-2 layers) → autoregressive K steps
  → → → → Output: draft logits → softmax → topk sampling → draft tokens

  Key insight: Feature-level draft avoids the "token bottleneck"
  → Standard draft: h_t → token x → embedding(x) → new hidden → draft
  → → Token bottleneck: quantization error + discrete sampling → information loss
  → EAGLE: h_t → direct feature processing → continuous information flow
  → → → No token bottleneck → higher draft quality → higher acceptance

  EAGLE v1:
    → fc = Linear(hidden_size * 2, hidden_size)  ← concat embed + target hidden
    → 1 layer Transformer block (no input_layernorm → identity instead)
    → Shares target's lm_head for logit computation
    → KV cache sharing: draft attention layers use target's KV cache

  EAGLE3:
    → fc = Linear(hidden_size * num_aux, hidden_size)  ← multi-layer hidden states
    → → num_aux = 3 (default): concatenates hidden states from 3 intermediate layers
    → → Optional fc_norm: per-aux RMSNorm normalization before concatenation
    → → First decoder layer: QKV accepts 2 * hidden_size (embed + projected features)
    → Hot token optimization: draft model reduces vocab to ~10K "hot" tokens

  Tree vs Chain:
    → topk > 1: tree topology → multiple candidate paths → higher coverage
    → → But: larger verify batch, complex KV management, variable CUDA graph shapes
    → topk = 1: chain topology → deterministic shapes → enables adaptive speculation
    → → SGLang adaptive controller: pre-build CUDA graphs for all step counts → atomic swap
    → → → RTX 4090 MUST use topk=1 for CUDA graph safety (#28569 crash proves topk>1 risky)
```

### 4.5 Medusa (Multiple Decoding Heads)

```
Architecture: Add N independent MLP prediction heads to target model
  → Head 1: h_t → MLP → P(x_{t+1} | x_{≤t})  ← standard next-token
  → Head 2: h_t → MLP → P(x_{t+2} | x_{≤t})  ← independent prediction
  → Head 3: h_t → MLP → P(x_{t+3} | x_{≤t})  ← independent prediction
  → ...
  → Head N: h_t → MLP → P(x_{t+N} | x_{≤t})

  All heads run in a SINGLE forward pass → no autoregressive draft loop!
  → Each head is a simple MLP (no attention) → extremely lightweight
  → Tree-based verification: consider all head combinations → tree topology

  Acceptance rate per head: ~0.50-0.65
  → Why low? Independent heads lack causal conditioning
  → → Head 2 doesn't know what Head 1 predicted → "blind guess" for position 2
  → → → TV distance is larger → lower acceptance

  Mathematically:
    P(x_{t+2} | x_{≤t}) ← Head 2 only conditions on x_{≤t}
    vs
    P(x_{t+2} | x_{≤t}, x_{t+1}) ← Sequential MTP conditions on x_{t+1}

    → Sequential conditioning gives more information → lower TV → higher α

  Pros:
    → Zero draft latency (single forward pass, no autoregressive loop)
    → Lightweight: only MLP heads, no attention layers
    → Easy to add to existing models (fine-tune heads only)

  Cons:
    → Low per-head acceptance (0.50-0.65) → needs tree verification to compensate
    → Tree verification → complex, variable batch sizes
    → Requires fine-tuning Medusa heads on target model's training data

  RTX 4090 viability:
    → Good for quick experiments → add Medusa heads → train 1-2 epochs → deploy
    → → But EAGLE has higher α → better long-term choice
```

### 4.6 Lookahead Decoding (Jacobian Iteration)

```
Architecture: Use Jacobi iteration for fixed-point draft
  → Key idea: decode is a fixed-point iteration x_{t+1} = f(x_t)
  → → Can solve via Jacobi iteration: guess x_{t+1}, verify, refine
  → → → Multiple tokens in parallel → "lookahead" draft

  Algorithm:
    1. Initialize: guess K tokens ahead (e.g., from previous decode positions)
    2. Verify: run target model on all K guessed positions
    3. Refine: update guesses based on verification results
    4. Iterate until convergence or rejection

  Acceptance rate: ~0.60-0.80
  → Depends on convergence speed of Jacobi iteration
  → → Deterministic tasks (code, structured output) → faster convergence → higher α

  Pros:
    → No extra model → uses target model only
    → No training overhead → works out of the box
    → Parallelizable → multiple positions simultaneously

  Cons:
    → Variable convergence → some sequences don't converge quickly
    → Complex implementation → requires careful Jacobi iteration management
    → Limited practical adoption → EAGLE/Medusa more popular

  RTX 4090 viability:
    → Interesting theoretical approach → but less mature than EAGLE
    → → Recommend EAGLE or N-gram for production
```

### 4.7 N-gram / Suffix Decoding (Zero-Overhead)

```
Architecture: Pattern matching from context → no model needed!
  → N-gram: search for longest matching n-gram in generated text
  → → Use subsequent tokens as draft → zero compute overhead
  → Suffix Decoding: suffix tree-based matching (Arctic Inference)
  → → More sophisticated matching → higher acceptance than n-gram

  N-gram algorithm:
    Given generated sequence: [A, B, C, D, E, B, C, D, F, G]
    Current suffix: [B, C, D] (n=3)
    Match found: [B, C, D, E] in earlier positions
    Draft tokens: [E] ← token following the matched n-gram

  Acceptance rate: ~0.30-0.50 (n-gram), ~0.40-0.60 (suffix)
  → Limited by context information → can't predict novel content
  → → Good for: repetitive text, code, structured output, few-shot prompts

  Pros:
    → ZERO compute overhead → no draft model, no forward pass
    → ZERO memory overhead → no extra model weights
    → Works with ANY target model → no architecture constraints
    → Simple implementation → CPU-based matching (Numba JIT)

  Cons:
    → Low acceptance rate → only useful for repetitive content
    → No prediction for novel content → first occurrence has zero draft
    → Requires sufficient context → short prompts → no matches

  RTX 4090 viability:
    → ★★★★★★★★★★★ RECOMMENDED for RTX 4090 GRPO rollout phase!
    → → Zero overhead → no impact on training memory budget
    → → → In GRPO: rollout generates many similar completions → n-gram works well!
    → → → → verl SGLang backend: n-gram speculative → 2-3x rollout speedup
```

---

## 5. SGLang's 6 Speculative Decoding Methods

### 5.1 Algorithm Registry

SGLang defines 6 built-in speculative algorithms in `SpeculativeAlgorithm` (spec_info.py):

| # | Algorithm | Enum | Draft Source | Overhead | Overlap (v2) | Status |
|---|-----------|------|-------------|----------|--------------|--------|
| 1 | **EAGLE** | `EAGLE` | Feature-level draft head | ~0.5GB | Yes (v2) | Production |
| 2 | **EAGLE3** | `EAGLE3` | Multi-layer aux hidden states | ~0.8GB | Yes (v2) | Production |
| 3 | **N-gram** | `NGRAM` | C++ NgramCorpus BFS lookup | 0 | No (v1 only) | Production |
| 4 | **DFlash** | `DFLASH` | Non-causal block drafting | Full model | No (v1 only) | Experimental |
| 5 | **Standalone** | `STANDALONE` | Independent vanilla LLM | Full model | Yes (v2) | Experimental |
| 6 | **Frozen KV MTP** | `FROZEN_KV_MTP` | MTP heads with frozen KV | ~2%/depth | — | Model-specific |

Plus: Plugin system via `SpeculativeAlgorithm.register("NAME", supports_overlap=True)` decorator.

### 5.2 Architecture: Spec v1 vs v2

SGLang has two speculative decoding architectures:

**Spec v1** (`EAGLEWorker` extends `TpModelWorker`):
- Worker IS the draft model → runs both draft and target
- 3-phase pipeline: draft → verify → draft_extend
- Draft and target share `req_to_token_pool` and `token_to_kv_pool_allocator`

**Spec v2** (`EAGLEWorkerV2` composes draft + target):
- Separation of concerns: `EagleDraftWorker` (BaseDraftWorker) + `TpModelWorker` (target)
- Overlap scheduling: `plan_stream` (separate CUDA stream) pipelines verify preparation with draft output
- topk=1 fast path: pre-allocates constant parent/score_index buffers for chain topology

### 5.3 EAGLE Implementation Details

```
SGLang EAGLE worker key components:
  → Draft: multi-step autoregressive loop
    for i in range(speculative_num_steps):
      → select_top_k_tokens → input_ids, hidden_states, scores
      → → Run draft model forward → softmax → topk sampling → next draft tokens
      → → → Increment positions by 1

  → Tree construction (organize_draft_results):
    → Concatenate per-step scores → [bs, topk * num_steps]
    → → Select top (num_draft_tokens - 1) by score
    → → → Sort indices for tree mask construction

  → Verify: target model single forward pass (TARGET_VERIFY mode)
    → Process ALL draft tokens in batch prefill
    → → Tree mask: each draft position only attends to its causal ancestors

  → Sampling: two paths
    → Greedy: verify_tree_greedy_func (sgl_kernel) → argmax comparison
    → Stochastic: tree_speculative_sampling_target_only → rejection sampling
    → → → TP sync: broadcast predict/accept_index from rank 0

  → Post-verify KV cache management:
    → page_size=1: simple free via boolean mask
    → topk=1 + page_size>1: align evict mask to page boundaries
    → topk>1 + page_size>1: shift accepted to front, copy KV, free tail

  → CUDA graphs:
    → EAGLEDraftCudaGraphRunner: entire multi-step draft as single graph
    → → Captured for multiple batch sizes (power-of-2 bucketing)
    → EAGLEDraftExtendCudaGraphRunner: post-verify extend
    → → Fixed-length extend (num_draft_tokens per request)
```

### 5.4 N-gram Worker (Zero-Overhead)

```
SGLang N-gram uses C++ NgramCorpus for token matching:
  → Corpus: trie-based data structure built from request token streams
  → → Online insertion: from ongoing requests (batch_put after verify)
  → → External corpus: load pre-tokenized text for domain-specific matching
  → → Lookup: batch_get → BFS with configurable breadth (min_bfs_breadth to max_bfs_breadth)
  → → Match types: exact or fuzzy (configurable via speculative_ngram_match_type)

  Key design choices:
  → Pre-allocates tensors for all batch sizes → avoid dynamic allocation
  → Uses reconstruct_indices_from_tree_mask (sgl_kernel) for tree traversal
  → Always produces draft_token_num candidates per request (padded if needed)

  Limitations:
  → Overlap scheduling NOT supported (spec_v1 only)
  → No CUDA graph for draft generation (CPU lookup)
  → → But: zero GPU compute → latency ratio β = 0 → ideal for RTX 4090!
```

### 5.5 DFlash Worker (Parallel Drafting)

```
SGLang DFlash uses a separate draft model with non-causal attention:
  → Architecture: separate TpModelWorker → own KV cache pool and attention backend
  → → Block drafting: generates fixed-size block in ONE forward pass
  → → → Non-causal attention: context from target model → query tokens drafted in parallel

  Key innovations:
  → Fused KV materialization: Triton kernels project target hidden states
    directly into draft KV cache for ALL layers in one pass
  → → Avoids sequential layer-by-layer forward → much faster
  → Vocabulary-parallel greedy sampling:
    → Each TP rank computes local max over its vocab shard
    → → All-gather maxima → global argmax via gather
  → Sliding window draft cache: compact KV cache for draft model
    → Only recent tokens visible → saves memory
    → → Rebuilds compact view from committed target state after each verify
```

### 5.6 Frozen KV MTP

```
SGLang Frozen KV MTP: Multi-Token Prediction with frozen (pre-computed) KV cache
  → Used for models with built-in MTP heads (DeepSeek-V3/V4, Step3.5, Gemma4)
  → → MTP heads share target model's embedding + lm_head
  → → → KV cache for MTP heads is computed once and reused → "frozen"

  Architecture:
  → MTP depth 1: h_t + embedding(x_{t+1}) → 1 layer Transformer → P(x_{t+2})
  → MTP depth 2: h_t + embedding(x_{t+1}) + embedding(x_{t+2}_pred) → P(x_{t+3})
  → → Ngram conditioning → sequential → higher acceptance than Medusa

  constant_draft_positions=True (Gemma4):
  → All draft steps use same positions → shares KV cache with target model
  → → Different from EAGLE where positions increment each step
```

### 5.7 Adaptive Speculation

```
SGLang Adaptive Controller (adaptive_runtime_state.py):
  → Manages runtime state switching based on acceptance rate
  → → Pre-builds CUDA graphs for all candidate step counts at startup
  → → → Atomic pointer swap for instant switching

  AdaptiveSpeculativeParams (adaptive_spec_params.py):
  → EMA tracking: ema_accept_len = (1-α) * ema + α * batch_avg (α=0.2)
  → Warmup: ignores first 10 batches
  → Update interval: every 5 batches
  → Hysteresis: down_hysteresis=-0.25 (strong evidence needed to reduce)
  → Candidate steps: discrete set [1, 3, 7] (always includes initial)

  Constraints:
  → Only EAGLE or EAGLE3 algorithms
  → topk=1 ONLY (chain topology → deterministic CUDA graph shapes)
  → NOT with DP attention, multi-layer eagle, TBO, or PDMux

  RTX 4090 recommendation:
  → ★★★★★★★★ MUST use adaptive + topk=1 for safety
  → → topk>1 → variable CUDA graph shapes → #28569 crash risk!
  → → → Adaptive adjusts K based on real acceptance → optimal K for each workload
```

### 5.8 SGLang Launch Configuration

```bash
# EAGLE (recommended for RTX 4090)
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path yuhuili/EAGLE-LLaMA3-8B \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4

# EAGLE3
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path yuhuili/EAGLE3-LLaMA3.1-8B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4

# N-gram (zero overhead, RTX 4090 safest option)
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 5 \
  --speculative-num-draft-tokens 5

# DFlash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path Qwen/Qwen3-DFlash-0.5B \
  --speculative-num-steps 5

# Frozen KV MTP (DeepSeek-V4)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4 \
  --speculative-algorithm FROZEN_KV_MTP \
  --speculative-num-steps 2

# Standalone (vanilla draft model)
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-70B \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --speculative-num-steps 5
```

---

## 6. vLLM Speculative Decoding Support

### 6.1 Proposer Architecture

vLLM V1 implements speculative decoding via a **Proposer-Scorer** dual-layer system:

```
SpecDecodeProposer (draft generation):
  → SpecDecodeBaseProposer ← base class for all LLM-based proposers
    → EagleProposer (EAGLE/EAGLE3)
    → DraftModelProposer (independent small model)
    → DFlashProposer (parallel drafting)
    → Gemma4Proposer (Gemma4 MTP, constant positions)
    → Step3p5MTPProposer (Step3.5 MTP)
  → MedusaProposer (independent MLP heads, NOT from base class)
  → NgramProposer (CPU, Numba JIT)
  → NgramProposerGPU (GPU, torch.compile)
  → SuffixDecodingProposer (suffix tree matching)
  → CustomClassProposer (user-defined)

RejectionSampler (verification):
  → Standard: vllm/v1/sample/rejection_sampler.py (for Ngram/Medusa/Suffix)
  → → GPU-native: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (for EAGLE/DraftModel)
  → → → Triton kernels: 3 phases (block_stats → rejection → resample)
```

### 6.2 Key Implementation Features

```
1. Two rejection samplers:
   → Standard (CPU-based proposers): handles Ngram, Medusa, Suffix
   → → GPU-native (EAGLE/DraftModel): Triton kernels for on-GPU verification
   → → → 3-phase pipeline: block stats → rejection sampling → Gumbel resample

2. SpecDecodeMetadata data structure:
   → draft_token_ids: [num_tokens] all draft token IDs
   → → num_draft_tokens: [batch_size] per-request draft count
   → → cu_num_draft_tokens: [batch_size] cumulative draft count
   → → target_logits_indices: [num_tokens] positions to verify
   → → bonus_logits_indices: [batch_size] bonus token positions

3. Async speculative decoding:
   → use_async_spec_decode: target and draft models overlap execution
   → → EAGLE/DraftModel: can use GPU sampled tokens → no CPU sync needed
   → → → Ngram: needs CPU tokens → must wait for bookkeeping → less overlap

4. Synthetic acceptance rate mode:
   → Predefined acceptance rates [r0, r1, r2, ...] → conditional_rate[i] = rate[i]/rate[i-1]
   → → For benchmarking/debugging → not production use
```

### 6.3 vLLM Configuration

```bash
# EAGLE
--speculative-model "yuhuili/EAGLE-LLaMA3-8B" \
--num-speculative-tokens 5

# EAGLE3
--speculative-model "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B" \
--speculative-method eagle3 \
--num-speculative-tokens 5

# Medusa
--speculative-model "vllm/Medusa-FastChat-Llama3-8B" \
--speculative-method medusa \
--num-speculative-tokens 5

# Draft Model
--speculative-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
--speculative-method draft_model \
--num-speculative-tokens 5

# N-gram (CPU)
--speculative-method ngram \
--num-speculative-tokens 5 \
--prompt-lookup-min 3 \
--prompt-lookup-max 5

# N-gram (GPU)
--speculative-method ngram_gpu \
--num-speculative-tokens 5 \
--prompt-lookup-min 3 \
--prompt-lookup-max 5

# DFlash
--speculative-model "Qwen/Qwen3-DFlash-0.5B" \
--speculative-method dflash \
--num-speculative-tokens 5

# Suffix Decoding
--speculative-method suffix \
--num-speculative-tokens 5

# Step3.5 MTP
--speculative-model "step/Step3.5" \
--speculative-method step3p5_mtp \
--num-speculative-tokens 5
```

### 6.4 vLLM vs SGLang Spec Decode Comparison

| Aspect | SGLang | vLLM V1 |
|--------|--------|---------|
| **Algorithm count** | 6 built-in + plugins | 10+ proposers |
| **Worker model** | Composed (v2) or Inherited (v1) | Separate model runners |
| **Tree verify** | Custom tree mask + batched prefill | Similar tree attention |
| **N-gram** | C++ NgramCorpus BFS + external corpus | Python-based (CPU) / torch.compile (GPU) |
| **DFlash** | Full implementation with fused KV Triton | Not available |
| **Adaptive** | Pre-built runtime states, atomic swap | N/A |
| **Overlap scheduling** | plan_stream (v2) | use_async_spec_decode |
| **topk=1 fast path** | Pre-allocated constant buffers | Not available |
| **CUDA graph crash** | #28569 (EAGLE3 on hybrid models) | Similar risk on variable-shape models |
| **MTP support** | FROZEN_KV_MTP (DeepSeek-V4) | Step3p5MTP, Gemma4MTP |

---

## 7. DeepSeek V4 MTP as Speculative Decoding

### 7.1 MTP Architecture (DeepSeek-V3/V4 Style)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DeepSeek-V4 uses MTP (Multi-Token Prediction) as its built-in speculative decoding mechanism**: MTP heads are trained jointly with the main model → dual benefit (training signal density + inference speedup) → but requires model redesign, not applicable to existing models.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
DeepSeek-V3/V4 MTP Architecture (Sequential/Ngram Dependency):

  Main model trunk → hidden state h_t
    → Main Head: P(x_{t+1} | x_{≤t}) ← standard NTP output

    → Depth 1 MTP Head:
      input = RMSNorm(h_t) + embedding(x_{t+1})
      → 1 layer Transformer block
      → → output projection (shared lm_head)
      → → → P(x_{t+2} | x_{≤t}, x_{t+1}) ← ngram conditioned!

    → Depth 2 MTP Head:
      input = RMSNorm(h_t) + embedding(x_{t+1}) + embedding(x_{t+2}_predicted)
      → 1 layer Transformer block
      → → output projection (shared lm_head)
      → → → P(x_{t+3} | x_{≤t}, x_{t+1}, x_{t+2}) ← sequential chain

  Key design choices:
  → Shared embedding: all depths use same embedding layer → saves parameters
  → → Shared lm_head: all depths use same vocab projection → saves parameters
  → → → Each depth adds only ~2% extra parameters (1 Transformer layer)

  → Embedding residual connection:
    → Depth d input = trunk_hidden + embedding(previous_depth_prediction)
    → → This is the "ngram dependency" → causal conditioning → higher acceptance

  Loss formulation:
    L = L_NTP(main) + λ₁ · L_{depth1} + λ₂ · L_{depth2}
    λ₁, λ₂ < 1 (typically 0.3-0.5 → auxiliary loss weight)

  DeepSeek-V3 MTP training results:
    | Config | MATH | Code (HumanEval) |
    |--------|------|-------------------|
    | No MTP | baseline | baseline |
    | D=1 MTP | +2.1% | +4.5% |
    | D=2 MTP | +4.8% | +8.2% |
    | D=3 MTP | +5.0% | +8.5% (diminishing) |

    → D=2 is optimal: significant gain + reasonable overhead (~12% training time)
```

### 7.2 MTP vs Other Speculative Methods

| Method | Draft Source | Causal Conditioning | Acceptance | Extra Params | Training Needed | Deploy Difficulty |
|--------|-------------|--------------------|------------|-------------|----------------|-------------------|
| **MTP (DS-V4)** | Shared trunk + ngram heads | YES (sequential) | 0.70-0.85 | ~4% (D=2) | Joint training | High (need new model) |
| **EAGLE** | Feature-level draft head | YES (autoregressive) | 0.85-0.92 | ~0.5GB | Draft-only training | Low (plugin) |
| **Medusa** | MLP heads | NO (independent) | 0.50-0.65 | ~1% | Fine-tune heads | Low (plugin) |
| **N-gram** | Context matching | — | 0.30-0.50 | 0 | None | Lowest |

**Why MTP acceptance is higher than Medusa but lower than EAGLE**:

$$\alpha_{\text{MTP}} > \alpha_{\text{Medusa}}$$ because MTP conditions on previous predictions (ngram dependency) → $\text{TV}(P, Q_{\text{MTP}}) < \text{TV}(P, Q_{\text{Medusa}})$.

$$\alpha_{\text{EAGLE}} > \alpha_{\text{MTP}}$$ because EAGLE operates in feature space (continuous hidden states) while MTP still passes through token embedding bottleneck → feature-level draft preserves more information.

### 7.3 DeepSeek-V4 MTP Inference Flow

```
MTP speculative decoding inference flow:

  Step 1: Main model decode → x_{t+1} (1 forward pass)
  Step 2: MTP Depth 1 → draft x_{t+2} (uses hidden state → ~zero cost!)
  Step 3: MTP Depth 2 → draft x_{t+3} (similarly ~zero cost)
  Step 4: Main model verify → 1 forward → check x_{t+2}, x_{t+3}
  Step 5: Accept/reject → rejection sampling → output tokens

  Speedup:
    → α ≈ 0.7, D = 2 → E[output] = 1 + 0.7 + 0.49 = 2.19 → ~1.8x practical
    → α ≈ 0.85, D = 2 → E[output] = 1 + 0.85 + 0.72 = 2.57 → ~2.4x practical

  DeepSeek-V3 reported: **1.8x speedup** on production inference
  → α ≈ 0.7 in practice (lower than theory due to position decay + distribution shift)
```

### 7.4 DSV4 MTP Current Issues

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DSV4 MTP is NOT production-ready as of 2026-06-19**: Three open issues reveal systematic instability.

**#26471 (MERGED → reverted)**: DeepSeek-V4 Online Compress support MTP
→ Online C128 compress + MTP → performance only 2% lower than NO_ONLINE+MTP
→ → BUT introduced state mapping lifecycle bug → needed revert

**#28591 (OPEN)**: Revert #26471 for testing
→ Reverts the online compress MTP support → testing stability
→ → Fix PR #28612 proposed: "Fix DSV4 C128 state mapping lifecycle"
→ → → Root cause: C128 compression state slots get corrupted during MTP verification
→ → → → Same pattern as MoE RouterReplay → constant buffer lost during state transfer

**#28569 (OPEN)**: EAGLE3 CUDA graph crash on gpt-oss-120b
→ EAGLE3 draft CUDA graph replay → illegal memory access
→ → Crash happens when running batch shrinks (32 → ~12 requests finish)
→ → → Running eager (--disable-cuda-graph) does NOT crash
→ → → → OOB access inside captured draft graph → attention KV indexing for padded bucket shape
→ → → → → Possibly same family as #28011 (Spec V2 verify + overlap stuck/OOB)

**Pattern**: All three issues involve **state corruption during speculative verification**:
→ C128 compress: state slots corrupted → MTP verification fails
→ EAGLE3 CUDA graph: attention indices corrupted → OOB access
→ → Common root cause: **dynamic data MUST NOT be cached in speculative verification** (same pattern as DSV4 systematic instability #10724)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 8. RTX 4090 Implications for GRPO Training

### 8.1 Can Speculative Decoding Help GRPO?

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**YES — speculative decoding can help GRPO training, specifically in the rollout phase**: verl's HYBRID architecture separates rollout (inference) from training → rollout engine runs decode → speculative decoding accelerates decode → faster rollout → faster GRPO training loop.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
GRPO training loop:
  1. Rollout: generate completions for prompts (INFERENCE — decode phase!)
     → → This is where speculative decoding helps!
  2. Reward: compute rewards for each completion
  3. Training: update policy via GRPO gradient step (TRAINING — not helped by spec decode)

  Rollout bottleneck:
    → B=1: 174 tok/s → ~99.4% memory-bound → GPU idle during rollout!
    → → Speculative decoding utilizes idle compute → faster rollout → faster training loop

  verl HYBRID architecture (sleep/wake):
    → Rollout engine: SGLang/vLLM → decode → speculative decoding applicable!
    → → Training engine: FSDP2 → gradient computation → NOT helped by spec decode
    → → → sleep_level=1: LoRA adapter tags=["kv_cache"] → 80x payload reduction
    → → → → Weight sync between engines → NOT related to spec decode
```

### 8.2 RTX 4090 Speculative Decoding Decision Matrix

| Scenario | Recommended Method | Expected Speedup | Extra Memory | Risk Level |
|----------|-------------------|-----------------|-------------|------------|
| **GRPO rollout (verl SGLang)** | N-gram | 2-3x | 0 | Lowest |
| **GRPO rollout (verl vLLM)** | N-gram GPU | 2-3x | ~0 | Low |
| **GRPO rollout (7B target)** | EAGLE topk=1 | 3-4x | ~0.5GB | Medium |
| **GRPO rollout (13B target)** | EAGLE topk=1 | 3-5x | ~0.5GB | Medium |
| **GRPO rollout (MoE target)** | N-gram only | 1.5-2x | 0 | Low |
| **Single GPU inference** | EAGLE topk=1 adaptive | 2-4x | ~0.5GB | Medium |
| **Multi-GPU inference** | EAGLE3 | 3-5x | ~0.8GB | Higher |

### 8.3 RTX 4090 MUST DO / MUST NOT

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

**RTX 4090 MUST DO**:
1. MUST use topk=1 chain topology for CUDA graph safety (topk>1 crashes per #28569)
2. MUST use adaptive speculation (adjusts K based on real acceptance rate)
3. MUST use N-gram for GRPO rollout (zero overhead, no extra memory)
4. MUST verify draft quality before deploying (KL > 5 → NEGATIVE speedup)
5. MUST set `--disable-cuda-graph` if EAGLE3 on hybrid/sliding-window models
6. MUST use `enforce_eager=True` for DSV4 models (8 systematic failures)
7. MUST measure acceptance rate in production (α < 0.5 → disable spec decode)
8. MUST use EAGLE draft model trained on same task (distribution matching)

**RTX 4090 MUST NOT**:
1. MUST NOT use untrained draft models (KL > 5 → 0.13-0.76x → NEGATIVE speedup!)
2. MUST NOT use EAGLE3 CUDA graphs on hybrid/sliding-window models (#28569 crash)
3. MUST NOT use speculative decoding for training phase (only rollout/decode phase)
4. MUST NOT use topk>1 on RTX 4090 (CUDA graph variable shapes → crash risk)
5. MUST NOT use speculative decoding when batch size > 32 (GPU already utilized)
6. MUST NOT use DSV4 MTP until #28612 fix is merged (#28591 revert still OPEN)
7. MUST NOT use speculative decoding with high temperature sampling (α drops)
8. MUST NOT use draft model larger than 1.4B for 7B target (β > 0.2 → no benefit)

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### 8.4 GRPO Rollout Speedup Estimation

```
RTX 4090 GRPO rollout parameters (verl HYBRID):
  → Target model: Qwen3-8B (BF16, ~16GB)
  → → Rollout: SGLang backend, B=1, decode ~174 tok/s baseline
  → → → N-gram speculative: K=5, α≈0.4 → speedup ≈ 1 + 0.4 + 0.16 + 0.06 + 0.03 = 1.65x
  → → → → For repetitive completions (GRPO with same prompt prefix): α can reach 0.6-0.8
  → → → → → α=0.7, K=3 → 1 + 0.7 + 0.49 = 2.19x → practical rollout speedup

  EAGLE on RTX 4090 (7B target):
  → Draft: 0.5GB EAGLE model → latency_ratio ≈ 0.05
  → → α=0.85, K=5 → speedup = (1+3.36)/(1+0.25) = 3.5x → theoretical
  → → → Practical: ~2.5-3x (accounting for overhead + position decay)

  Memory budget:
  → RTX 4090 24GB total
  → → 7B BF16 model: ~16GB
  → → → KV cache: ~4GB (2048 tokens, B=1)
  → → → → EAGLE draft: ~0.5GB → total ~20.5GB → FITS!
  → → → → → N-gram: 0 → total ~20GB → fits easily!
  → → → → → → EAGLE3 draft: ~0.8GB → total ~20.8GB → fits but tight
```

### 8.5 verl Integration Path

```
verl HYBRID + speculative decoding integration:

  Current verl rollout engines:
  → SGLang backend: supports EAGLE, NGRAM, DFLASH, FROZEN_KV_MTP
  → → vLLM backend: supports EAGLE, ngram, medusa, draft_model
  → → → Both engines can enable speculative decoding in the rollout config

  verl rollout config example (SGLang + N-gram):
    rollout_name = "sglang"
    engine_kwargs = {
      "speculative_algorithm": "NGRAM",
      "speculative_num_steps": 5,
      "speculative_num_draft_tokens": 5,
    }

  verl rollout config example (vLLM + EAGLE):
    rollout_name = "vllm"
    engine_kwargs = {
      "speculative_model": "yuhuili/EAGLE-LLaMA3-8B",
      "num_speculative_tokens": 5,
    }

  ★★★★★★★★★★ verl SGLang N-gram is the RECOMMENDED RTX 4090 GRPO rollout config:
  → Zero extra memory → no impact on training memory budget
  → → Works with LoRA sleep/wake → no interference with weight sync
  → → → α can reach 0.6-0.8 in GRPO (repetitive prefix + similar completions)
  → → → → 2-3x rollout speedup → ~30-50% total GRPO training speedup
```

---

## 9. Critical Issues Tracker

### 9.1 Active Speculative Decoding Issues

| Issue | Framework | Severity | Description | RTX 4090 Impact |
|-------|-----------|----------|-------------|-----------------|
| **#28569** | SGLang | CRITICAL | EAGLE3 CUDA graph crash (illegal memory access on hybrid/sliding-window models when batch shrinks) | MUST use `--disable-cuda-graph` or topk=1 |
| **#28591** | SGLang | HIGH | DSV4 MTP revert (online C128 compress + MTP state mapping bug) | MUST NOT use DSV4 MTP until #28612 merged |
| **#28612** | SGLang | HIGH (fix) | DSV4 C128 state mapping lifecycle fix → needs testing on pinchbench | Pending validation |
| **#28575** | SGLang | MEDIUM | MTP weight update reimplementation (revert #27749 + re-impl on v2 worker layout) | Selector-driven fan-out for draft workers |
| **#28407** | SGLang | MEDIUM | NEXTN (MTP) + structured output (xgrammar) crashes scheduler on Mamba-hybrid models | Hybrid model + spec decode = risky |
| **#28484** | SGLang | MEDIUM | Spec v2 + Mamba extra_buffer crash (NoneType in set_mamba_track_indices) | Mamba hybrid models spec decode unstable |
| **#24233** | SGLang | MEDIUM | NSA FA3 crash with DP Attention on padded speculative batches | DP attention + spec decode = crash risk |
| **#28011** | SGLang | MEDIUM | Spec V2 verify + overlap stuck/OOB (same family as #28569) | Overlap scheduling risk |
| **#23705** | SGLang | LOW | Adaptive Speculative Decoding Roadmap | Future improvement |
| **#27462** | SGLang | LOW | Parallel Speculative Decoding Roadmap | Future improvement |

### 9.2 DSV4 Systematic Instability Pattern

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
**DSV4 + speculative decoding = 8+ systematic failures across 3 frameworks**: The root cause pattern is consistent — **dynamic data (C128 compress state, attention indices, sparse cache) MUST NOT be cached or pre-computed during speculative verification**. Each verification pass may reject tokens → state must be recomputed from scratch → caching stale state causes corruption → crash or garbage output.

**Mandatory mitigation**: `enforce_eager=True` for all DSV4 speculative decoding. Do NOT use CUDA graphs with DSV4 + MTP until #28612 is validated.
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

---

## 10. References

### 10.1 Core Papers

1. **Leviathan et al.**, "Fast Inference from Transformers via Speculative Decoding", ICML 2023 (arXiv:2302.01318)
   → Original rejection sampling proof → distribution-preserving guarantee

2. **Chen et al.**, "Accelerating Large Language Model Decoding with Speculative Sampling", 2023 (arXiv:2305.09781)
   → Independent derivation → same algorithm → parallel discovery

3. **Cai et al.**, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024 (arXiv:2401.10774)
   → Independent MLP heads → tree verification → zero draft latency

4. **Li et al.**, "EAGLE: Speculative Sampling Requires Rethinking Feature-Level Draft Model Architecture", 2024 (arXiv:2406.16858)
   → Feature-level draft → avoids token bottleneck → highest acceptance rate

5. **Li et al.**, "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Tree Structure", 2024
   → Dynamic tree topology → adaptive draft tree → EAGLE improvement

6. **Gloeckle et al.**, "Better & Faster Large Language Models via Multi-Token Prediction", Meta, 2024 (arXiv:2404.19737)
   → Independent MTP heads → planning ahead → dual benefit

7. **DeepSeek-AI**, "DeepSeek-V3 Technical Report", 2024 (arXiv:2412.19437)
   → Sequential MTP (ngram dependency) → D=2 → 1.8x inference speedup

8. **Lookahead Decoding**, "Break the Sequential Dependency of LLM Inference: Lookahead Decoding", 2023
   → Jacobi iteration → parallel fixed-point → no draft model needed

### 10.2 Framework Documentation

- SGLang speculative decoding: `python/sglang/srt/speculative/` (6 algorithms, v1/v2 architectures)
- vLLM speculative decoding: `vllm/v1/spec_decode/` (10+ proposers, two rejection samplers)
- SGLang #28569: EAGLE3 CUDA graph crash → https://github.com/sgl-project/sglang/issues/28569
- SGLang #28575: MTP weight update reimplementation → https://github.com/sgl-project/sglang/pull/28575
- SGLang #28591: DSV4 MTP revert → https://github.com/sgl-project/sglang/pull/28591
- SGLang #28612: DSV4 C128 state mapping fix → https://github.com/sgl-project/sglang/pull/28612
- SGLang #26471: DSV4 Online Compress MTP (MERGED, then reverted) → https://github.com/sgl-project/sglang/pull/26471

### 10.3 Related Reading Notes

- `/notebook/fundamentals/speculative-decoding.md` — Chinese version basic theory
- `/notebook/fundamentals/speculative-decoding-algorithm-deep-dive.md` — Algorithm theory + vLLM
- `/notebook/fundamentals/speculative-decoding-benchmark-rtx4090.md` — RTX 4090 benchmark data
- `/notebook/fundamentals/multi-token-prediction-deep-dive.md` — MTP training + inference dual benefit
- `/notebook/projects/vllm-spec-decode-reading.md` — vLLM V1 spec decode source reading
- `/notebook/projects/sglang-spec-decode-reading.md` — SGLang spec decode source reading
- `/notebook/projects/sglang-architecture.md` — SGLang architecture overview
- `/notebook/fundamentals/deepseek-v3-architecture-deep-dive.md` — DSV3 MLA+MoE+MTP
- `/notebook/projects/dsv4-systematic-instability-pattern-synthesis.md` — DSV4 instability pattern

### 10.4 Project Tools

- `tools/speculative_decoding_sim.py` — Speculative decoding simulator (5 experiments)
- `tools/speculative_decoding_benchmark_4090.py` — RTX 4090 benchmark (3 draft x 6 K values)
- `tools/mtp_training_simulator.py` — MTP training + speculative decoding simulator
- `tools/mtp_train_experiment.py` — MTP training experiment runner
- `tools/deepseek_v3_moe_sim.py` — DeepSeek-V3 MoE simulator

---

## Appendix A: Mathematical Derivation Summary

### A.1 Complete Speedup Formula

$$S(\alpha, K, \beta) = \frac{1 + \frac{\alpha(1 - \alpha^K)}{1 - \alpha}}{1 + K\beta}$$

Where:
- $\alpha$ = per-position acceptance rate = $1 - \text{TV}(p, q)$
- $K$ = number of draft tokens
- $\beta$ = $t_{\text{draft}} / t_{\text{target}}$ = latency ratio

### A.2 Optimal K (Approximate)

$$K^* \approx \sqrt{\frac{1 - \alpha}{\alpha \beta}} - \frac{1}{\beta}$$

### A.3 Speedup Ceiling (K → ∞)

$$S_{\max} = \frac{1}{(1 - \alpha)(1 + K^*\beta)} \approx \frac{1}{1 - \alpha} \quad \text{(when } \beta \ll 1\text{)}$$

### A.4 Viability Threshold

$$S > 1 \iff \alpha > \frac{K\beta}{1 + K\beta}$$

For K=5, $\beta=0.2$: $\alpha > 0.5$ → need acceptance > 50% for positive speedup.
For K=3, $\beta=0.1$: $\alpha > 0.25$ → easier threshold.

### A.5 RTX 4090 Viability Thresholds

| Target Size | Draft Size | $\beta$ | $\alpha$ needed (K=5) | Feasible? |
|-------------|------------|---------|----------------------|-----------|
| 7B | 0.7B | 0.10 | >0.33 | YES (n-gram/EAGLE) |
| 7B | 1.4B | 0.20 | >0.50 | Marginal (need good draft) |
| 7B | 7B (same) | 1.00 | >0.83 | NO (β too high) |
| 13B | 1.3B | 0.10 | >0.33 | YES |
| 13B | 2.6B | 0.20 | >0.50 | Marginal |
| MoE 30B | — | — | — | n-gram only (draft can't compress experts) |

---

## Appendix B: Rejection Sampling Implementation Comparison

### B.1 vLLM GPU-native Rejection Sampler (Triton)

```
3-phase Triton kernel pipeline:

Phase 1: _compute_block_stats_kernel
  → Per logit position, per vocab block:
  → → Compute local max + softmax denominator
  → → → Compute local argmax (greedy)
  → → → → Process target AND draft logits simultaneously

Phase 2: _rejection_kernel
  → Per request:
  → → accepted = True
  → → for i in range(num_tokens - 1):
  → → → if accepted:
  → → → → if temperature == 0:
  → → → → → target_argmax == draft_sampled? → accept/reject
  → → → → else:
  → → → → → target_log_prob > log(u) + draft_log_prob? → accept/reject

Phase 3: _resample_kernel
  → Rejected position or bonus:
  → → if bonus: residual = target_logits (all available)
  → → elif HAS_DRAFT_LOGITS: residual = log(max(p(x) - q(x), 0))
  → → else: residual = target_logits (exclude draft token)
  → → → Gumbel block argmax sampling
```

### B.2 SGLang Verification

```
Two verification paths:

Greedy (verify_tree_greedy_func, sgl_kernel):
  → Compare target argmax with draft candidates along tree paths
  → → Accept matching tokens, stop at first mismatch
  → → → Append target's prediction as bonus token

Stochastic (tree_speculative_sampling_target_only, sgl_kernel):
  → Compute target probabilities: softmax(logits/temperature) → top_k_renorm → top_p_renorm
  → → Rejection sampling: accept if uniform_random < target_prob[token]
  → → → If rejected: sample from max(0, target_prob - draft_prob) (rescaled)
  → → → → Uses threshold_single and threshold_acc acceptance thresholds
  → → → → → TP sync: broadcast predict/accept_index from rank 0
```
