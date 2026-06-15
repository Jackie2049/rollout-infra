# RTX 4090 Architecture Decision Records (ADR)

> 2026-06-15 | 正式记录RTX 4090 AI infra架构决策 → 基于57+深度阅读
> ADR格式: Context→Decision→Rationale→Consequences→Alternatives Considered

---

## ADR-001: GRPO over PPO for RL Training

**Status**: Accepted

**Context**: RTX 4090 has 24GB VRAM. PPO requires separate critic model (~14GB) + value loss + GAE computation → total ~270GB → impossible on 24GB.

**Decision**: Use GRPO (Group Relative Policy Optimization) for all RL training on RTX 4090.

**Rationale**:
- GRPO eliminates critic → saves 50% compute + 50% memory
- Group-relative advantage = natural normalization → mathematically sound
- need_critic()=False in verl → automatic detection
- GRPO_VECTORIZED → 10-100x faster advantage computation than loop
- Dr.GRPO (norm_adv_by_std=False) → prevents gradient vanishing with small groups
- Production proven: DeepSeek-R1/V4 trained with GRPO

**Consequences**:
- (+) Only 17GB training memory → 7GB headroom → feasible
- (+) Single GPU feasible → no distributed needed
- (+) GRPO_VECTORIZED → efficient computation
- (-) Group-relative advantage → less precise than GAE for small rollout_n
- (-) Requires rollout_n≥4 → singleton n=1 → advantage=raw_reward → unstable

**Alternatives Considered**:
- PPO: ✗ (270GB → impossible)
- DPO: different paradigm (offline) → not on-policy → not applicable
- REINFORCE: no baseline → high variance → GRPO better

---

## ADR-002: LoRA-32 for Parameter-Efficient Training

**Status**: Accepted

**Context**: Full parameter training 7B → 14GB weights + 14GB gradients + 14GB optimizer = 42GB → impossible. LoRA reduces trainable params to 0.8%.

**Decision**: Use LoRA rank=32 targeting all linear layers (attn/mlp/unembed).

**Rationale**:
- LoRA-32 → 0.8% params → ~2.6GB weights → +0.5GB optimizer → total ~17GB feasible
- rLLM auto-init → create_lora_training_client_async → zero configuration
- Merge after training → equivalent to full-parameter training → no quality loss
- INT4 quantization after merge → LoRA information preserved in quantized weights
- LoRA enables in-process weight sync → GPU-only merge → ~0ms

**Consequences**:
- (+) Memory feasible: 17GB / 24GB → 7GB headroom
- (+) Merge = equivalent to full training → no loss
- (+) Auto-init in rLLM → zero config → simplest path
- (+) LoRA weights → GPU-only merge → zero-copy weight sync
- (-) 37% slower during training (LoRA overhead)
- (-) Merge required before deployment → extra step

**Alternatives Considered**:
- LoRA-16: ✓ less memory but might be insufficient for quality
- LoRA-64: ✓ more expressive but 18GB → closer to limit
- Full params: ✗ (42GB → impossible)
- QLoRA (INT4 train): ✗ (BF16 is only correct training precision for SM89)

---

## ADR-003: Rule-Based Reward over Reward Model

**Status**: Accepted

**Context**: Reward Model requires ~14GB GPU → 24GB minus 14GB = 9GB → not enough for actor. Colocated RM (sleep/wake) needs memory swap → 300ms overhead per step.

**Decision**: Use rule-based reward functions (math/code/format) running on CPU.

**Rationale**:
- Rule-based → pure CPU → zero GPU memory overhead
- Math reward → correct/incorrect → deterministic → binary score
- Code reward → execution test → pass/fail → objective
- Outcome reward → only last valid token nonzero → GRPO naturally aligned
- verl custom_reward_function → dynamic import → flexible
- GDPO → rule-based returns dict → per-dimension normalization → prevents dominant signal

**Consequences**:
- (+) Zero GPU overhead → full 24GB for actor+LoRA
- (+) Deterministic → reproducible → no RM variance
- (+) Outcome reward → GRPO-aligned → group-relative advantage correct
- (+) GDPO dict → per-dimension → more nuanced scoring
- (-) Limited to math/code/format domains → subjective domains impossible
- (-) Binary/integer scores → less granular than RM continuous scores
- (-) Manual implementation needed → each domain requires custom function

**Alternatives Considered**:
- Colocated RM (sleep/wake): ✗ (memory too tight → 300ms overhead)
- Separate GPU RM: ✗ (needs additional GPU → cost)
- DisRM (HTTP API): ✗ (needs RM server → GPU resource)

---

## ADR-004: rLLM TinkerBackend over verl HYBRID

**Status**: Accepted

**Context**: Both rLLM Tinker and verl HYBRID can run GRPO+LoRA on single GPU. Need to choose optimal framework.

**Decision**: Use rLLM TinkerBackend as primary RL training framework for RTX 4090.

**Rationale**:
- In-process → zero IPC → zero-copy → ~0ms weight sync → fastest
- LoRA auto-init → zero configuration → rank=32 automatic
- bypass_mode=true → π_old=rollout logprobs → saves one forward pass → ~40% faster
- ServiceClient/SamplingClient → GPU-only merge → new client creation → zero-copy
- 5 loss types → PPO/IS/cispo/dro/CE → flexible
- GRPO→PPO automatic mapping → simplest path
- Berkeley community → 5.6K stars → active development
- Terminal-RL → pass@k ↔ GRPO rollout_n alignment → eval+train consistent

**Consequences**:
- (+) In-process → simplest architecture → no Ray/IPC
- (+) bypass_mode → ~40% compute savings → critical for 24GB
- (+) Auto-init → zero LoRA configuration
- (+) Zero-copy weight sync → ~0ms → fastest
- (-) Small community → less mature than verl
- (-) Fewer advantage estimators → verl has 15
- (-) No GPU cluster support → single GPU only

**Alternatives Considered**:
- verl HYBRID: ✓ (more mature, 5K stars, Ray ecosystem, 15 advantage estimators) → but Ray overhead+IPC+more complex → secondary option for multi-GPU
- DeepSpeed ZeRO-2+LoRA: ✗ (no GRPO built-in → need custom implementation)
- Megatron: ✗ (overkill, no LoRA, complex config → 970-line reading confirms)
- PyTorch compile: ✗ (need custom GRPO implementation → overlay only)

---

## ADR-005: INT4+INT8KV for Inference Deployment

**Status**: Accepted

**Context**: BF16 inference → 14GB weights + 10GB KV = 24GB → exactly fills VRAM → zero headroom → multi-turn/long-context impossible.

**Decision**: Use INT4 GPTQ quantization + INT8 KV cache for all production inference on RTX 4090.

**Rationale**:
- INT4 weights → 3.5GB vs 14GB → 4x memory savings
- INT8 KV → 5GB vs 10GB → 2x KV savings → multi-turn feasible
- Total ~11GB → 13GB headroom → room for CUDA graph + EAGLE + batch
- GPTQ Marlin kernel → SM 8.9 supported → high performance
- TritonW4A16 fallback → PR#43731 → non-Marlin shapes now work (DeepSeek-V2-Lite)
- vLLM INT4+INT8KV+CUDA graph+FlashInfer → 4,791 tok/s baseline
- EAGLE + INT4 → 9,088 tok/s → 1.9x speculative acceleration

**Consequences**:
- (+) 13GB headroom → multi-turn/long-context/batch all feasible
- (+) 4,791 tok/s baseline → 2x faster than BF16 decode
- (+) EAGLE → 9,088 tok/s → near-real-time
- (+) Triton fallback → more MoE models loadable (DS-V2-Lite, Qwen2-MoE)
- (-) INT4 ~2-5% quality loss vs BF16 → acceptable for inference
- (-) Triton fallback layers ~2-5x slower than Marlin → ~5-15% overall hit
- (-) MRv2 doesn't support quantized models → stuck with V1 runner for now

**Alternatives Considered**:
- BF16 inference: ✗ (24GB exact fill → zero headroom → multi-turn impossible)
- AWQ INT4: ✓ (similar compression but slightly less throughput)
- FP8 inference: ✗ (SM89 no dedicated kernel acceleration → marginal benefit)
- INT8 weights: ✓ (2x compression → less savings than INT4 → 7GB vs 3.5GB)

---

## ADR-006: Linear CUDA Graph Sizing for GRPO

**Status**: Accepted

**Context**: Megatron PR#3509 introduced exponential cudagraph sizing → +13.6% peak memory (69.2→60.9GB). PR#5280 fixed with linear sizing.

**Decision**: Always use linear CUDA graph sizing for GRPO training scenarios on RTX 4090.

**Rationale**:
- Exponential → few large graphs → saves ~15GB memory for pure inference
- But GRPO → variable batch sizes → rollout shapes unpredictable → exponential regression
- Linear → dense small-size graphs → stable → predictable → GRPO-friendly
- +13.6% peak memory → 24GB → 2.7GB more → from 7GB to 4.3GB headroom → dangerous
- vLLM and Megatron both support linear sizing → --cuda-graph-sizing-distribution linear
- rLLM Tinker → inference and training same process → graph sizing must be linear

**Consequences**:
- (+) Stable memory → predictable → safe on 24GB
- (+) Dense small-size coverage → rollout shapes well-served
- (-) More graphs → more graph pool memory → ~2GB fixed
- (-) Pure inference could save ~15GB with exponential → but not GRPO scenario

**Alternatives Considered**:
- Exponential sizing: ✗ (+13.6% peak → regression → dangerous on 24GB)
- Fixed sizes [1,2,4,8,16,32]: ✓ (manual linear → acceptable alternative)
- Piecewise: ✓ (split between prefill/decode → more complex → vLLM FULL_AND_PIECEWISE)

---

## ADR-007: BF16 Training + INT4 Inference = Dual Precision

**Status**: Accepted

**Context**: SM 8.9 (Ada) does not support FP8 training, FP8 E5M2, FP8 AllGather. INT4 cannot be used for training gradients.

**Decision**: Use BF16 for all training and INT4 for all inference → dual precision by phase.

**Rationale**:
- BF16 → only correct training precision on SM 8.9 → no FP8 alternative
- INT4 → only feasible inference precision → BF16 fills 24GB exactly → impossible
- LoRA merge → BF16 weights → then INT4 quantize → LoRA information preserved
- Different phases → different precision requirements → each optimal at its stage
- Training: compute-bound → BF16 sufficient → gradient accumulation dtype=fp32 recommended
- Inference: memory-bound → INT4 reduces weight reads 4x → decode 3.4x faster

**Consequences**:
- (+) Each phase uses optimal precision → no compromises
- (+) Training → BF16 → no quality loss → correct gradients
- (+) Inference → INT4 → memory feasible → throughput maximized
- (-) Quantization step needed between training and inference
- (-) ~2-5% quality loss from INT4 → acceptable for inference
- (-) Pipeline more complex → 6 steps instead of 2

**Alternatives Considered**:
- BF16 training+BF16 inference: ✗ (24GB exact → impossible)
- INT4 training: ✗ (gradient computation incorrect → quality loss → not supported)
- FP8 training: ✗ (SM89不支持)
- FP8 inference: ✗ (SM89 no dedicated acceleration → marginal benefit vs INT4)

---

## ADR-008: Single GPU over Multi-GPU PCIe

**Status**: Accepted

**Context**: RTX 4090 PCIe scaling disaster: 7B 8GPU = 0.46x throughput → communication overhead dominates computation.

**Decision**: Always run on single GPU. All distributed frameworks (ZeRO-3, FSDP2, Megatron TP/PP, DeepCompile) are not applicable.

**Rationale**:
- PCIe bandwidth → 32GB/s vs NVLink 900GB/s → 28x slower
- 7B 8GPU → 0.46x → more GPUs = slower → anti-scaling
- AllReduce on PCIe → dominated by communication → compute negligible
- ZeRO-3 → 3Ψ AllGather → PCIe → each step dominated by communication
- FSDP2 → 2Ψ AllGather → same problem → PCIe bottleneck
- Megatron TP → AllReduce per layer → PCIe → each layer slow
- DeepCompile → compiler-level ZeRO → still needs multiple GPUs → PCIe disaster
- LoRA → no distributed → 0 communication → in-process → fastest

**Consequences**:
- (+) No communication overhead → pure compute
- (+) LoRA → 17GB → feasible → simplest architecture
- (+) In-process → zero IPC → zero-copy → fastest
- (-) Limited to 7B scale → 14GB base model → larger needs more GPU
- (-) No distributed fault tolerance → single point of failure
- (-) Batch size limited → gradient accumulation needed

**Alternatives Considered**:
- 2-GPU PCIe: ✗ (0.46x scaling → slower than single)
- 4-GPU PCIe: ✗✗✗ (even worse)
- NVLink RTX 4090: not available (consumer GPU → no NVLink)
- Cloud GPU (H100): ✓ (NVLink → different hardware → different decisions)

---

## Summary: RTX 4090 Architecture Stack

```
★ ★ ★ ★ ★ RTX 4090 Optimal Stack (based on 8 ADRs):

Training:    rLLM TinkerBackend + GRPO + LoRA-32 + bypass_mode + rule-based reward
             → 17GB / 24GB → 7GB headroom → ✓✓✓

Merge:       LoRA merge into base → BF16 → equivalent to full training
             → <1ms GPU-only merge → zero-copy

Quantize:    GPTQ INT4 + INT8 KV config → 3.5GB weights → 4x compression
             → ~1min on CPU

Deploy:      vLLM INT4 + INT8KV + GQA-8 + prefix caching + CUDA graph (linear sizing)
             → 4,791 tok/s → EAGLE → 9,088 tok/s

Evaluate:    rllm eval --attempts 8 → pass@k ↔ GRPO rollout_n
             → CPU warm-pool → no GPU

★★★★★ Key Principle: BF16 train + INT4 inference → dual precision → each optimal
★★★★★ No distributed → single GPU → LoRA → no ZeRO/FSDP/DeepCompile needed
★★★★★ rLLM Tinker = simplest fastest → zero IPC → zero-copy → bypass_mode
```
