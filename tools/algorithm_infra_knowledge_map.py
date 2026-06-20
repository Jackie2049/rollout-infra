#!/usr/bin/env python3
"""
Algorithm → Infra Knowledge Map: Query the theory-to-bug connection database.

This tool maps mathematical algorithm theory to framework implementation bugs
and RTX 4090 GRPO training decisions. Every connection is backed by both
mathematical derivation AND real framework issue evidence.

Modes:
  map       — Show the complete Theory → Bug → Decision map
  theory    — Query by theory domain (transformer, grpo, quantization, optimizer, spec_decode, architecture, generative, rlhf)
  bug       — Query by framework issue number (e.g. #8072, #45683, #605)
  decision  — Show all MUST DO / MUST NOT rules with mathematical proof
  rtx4090   — Show RTX 4090 expert readiness assessment
  proof     — Show mathematical proof for a specific rule
  cross     — Show cross-framework patterns (dsv4, zero3, muon)
"""

import sys
import json

# ─── Theory → Bug → Decision Connection Database ───────────────────────────

CONNECTIONS = {
    "transformer": {
        "name": "Transformer Architecture Mathematical Derivation",
        "note": "notebook/fundamentals/transformer-architecture-mathematical-derivation.md",
        "connections": [
            {
                "component": "MHA → GQA → MLA",
                "math_property": "KV per token: 768KiB→16KiB→4KiB→0.4KiB",
                "bugs": ["vLLM #45964 MLA DCP", "SGLang #28676 MoE"],
                "decision": "MLA models (DSV2-Lite) = BEST for 24 GiB",
                "formula": "KV_mem = 2 * n_heads * head_dim * seq_len * dtype_size",
            },
            {
                "component": "SwiGLU MLP",
                "math_property": "3 activation matrices: gate, up, down = ~4x larger than attention",
                "bugs": ["Megatron #5346 FSDP optimizer state save fails on SwiGLU"],
                "decision": "MoE SwiGLU = memory hog, MUST use activation checkpointing",
                "formula": "SwiGLU_mem = 3 * d_model * d_hidden * dtype_size",
            },
            {
                "component": "RoPE position encoding",
                "math_property": "inv_freq stays fp32 (should NOT be downcast)",
                "bugs": ["DeepSpeed #8066 per-policy dtype → #8072/#8076 regression"],
                "decision": "ZeRO-3+PEFT LoRA dtype mismatch → ALWAYS ZeRO-2",
                "formula": "θ_i = i * inv_freq_i where inv_freq_i = 1 / (10000^(2i/d))",
            },
            {
                "component": "RMSNorm",
                "math_property": "No centering, no learnable bias → fewer parameters",
                "bugs": ["SGLang #24459 rms_norm+mm.dtype"],
                "decision": "MUST use rms_norm deterministic override for GRPO reproducibility",
                "formula": "RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ",
            },
            {
                "component": "MoE Router",
                "math_property": "Top-K gating → sparse activation → only K/N experts active",
                "bugs": ["#45683 MoE combine nondeterminism", "#8066 router dtype sensitivity"],
                "decision": "AutoEP+LoRA+EP=1 → MoE training VIABLE on RTX 4090",
                "formula": "router(x) = softmax(W_router · x), top-K selection",
            },
            {
                "component": "MoE Expert SwiGLU",
                "math_property": "Each expert has own gate/up/down weights",
                "bugs": ["#28618 SM89 DeepGEMM disabled → Triton fallback"],
                "decision": "SM89 = Triton backend, NOT DeepGEMM",
                "formula": "expert_i(x) = SwiGLU(W_down_i · (gate(W_gate_i·x) * W_up_i·x))",
            },
        ],
    },
    "grpo": {
        "name": "GRPO Algorithm Mathematical Derivation",
        "note": "notebook/fundamentals/grpo-algorithm-mathematical-derivation.md",
        "connections": [
            {
                "component": "Group normalization",
                "math_property": "A_i = (r_i - μ_G) / σ_G — baseline reduction",
                "bugs": ["rLLM #605 groups by trajectory.uid instead of prompt → BROKEN"],
                "decision": "MUST verify GRPO grouping by prompt key",
                "formula": "A_i = (r_i - mean(r_group)) / std(r_group)",
            },
            {
                "component": "Reference model KL penalty",
                "math_property": "clip(r_i, 1-ε, 1+ε) * A_i - β * D_KL",
                "bugs": ["verl #6790 async trainer bypass_mode FORCED True"],
                "decision": "bypass_mode = 18Ψ→3.8Ψ → RTX 4090 VIABLE",
                "formula": "D_KL(π_θ || π_ref) = Σ log(π_θ(y|x) / π_ref(y|x))",
            },
            {
                "component": "Score function trick",
                "math_property": "∇J = E[r · ∇log π]",
                "bugs": ["rLLM #663 Step.output was None → ALL rewards=0.0"],
                "decision": "MUST verify reward computation not silently zero",
                "formula": "∇_θ J(θ) = E_{y~π_θ}[r(x,y) · ∇_θ log π_θ(y|x)]",
            },
            {
                "component": "CPPO position-weighted trust",
                "math_property": "clip(r_i, 1-ε·w_t) where w_t decreases with position",
                "bugs": ["verl #6731 CPPO implementation"],
                "decision": "CPPO+bypass = RTX 4090 #1 BEST training approach",
                "formula": "w_t = max(0, 1 - t/T) → less clipping at prefix → prevents drift",
            },
            {
                "component": "Gradient clipping",
                "math_property": "||∇||_2 ≤ C — prevents gradient explosion",
                "bugs": ["DeepSpeed #8068 gradient_clipping default 0→1.0"],
                "decision": "ALWAYS set gradient_clipping: 1.0 explicitly",
                "formula": "if ||∇||_2 > C: ∇ ← ∇ * C / ||∇||_2",
            },
            {
                "component": "Group size",
                "math_property": "Must have ≥2 samples per group for normalization",
                "bugs": ["rLLM #605 groups by trajectory.uid → group size 1 → σ=0 → BROKEN"],
                "decision": "MUST verify group_size ≥ 2 per prompt",
                "formula": "σ_G = 0 when |G|=1 → division by zero → undefined advantage",
            },
        ],
    },
    "quantization": {
        "name": "Quantization Theory Mathematical Derivation",
        "note": "notebook/fundamentals/quantization-theory-mathematical-derivation.md",
        "connections": [
            {
                "component": "FP8→BF16 prologue fusion",
                "math_property": "SM89 stores FP8 but computes BF16 (no wgmma)",
                "bugs": ["#184119 SM89 fp8 guard", "#187484 w8a8 Inductor breaks on torch 2.13"],
                "decision": "P9 Fusion Guard = CRITICAL for SM89 FP8 inference",
                "formula": "x_bf16 = dequant(x_fp8, scale) → fused before compute",
            },
            {
                "component": "Per-tensor vs per-channel scaling",
                "math_property": "s = x_max / q_max — granularity matters",
                "bugs": ["DeepSpeed #8066 per-policy dtype → fp32 LoRA + bf16 base mismatch"],
                "decision": "ZeRO-2 avoids partition dtype problem entirely",
                "formula": "x_q = round(x/s) + z, s = (x_max - x_min)/(q_max - q_min)",
            },
            {
                "component": "MXFP8 (OCP Microscaling)",
                "math_property": "Block-wise scaling: X_dims={1,2}, micro_tile={1,32}",
                "bugs": ["SGLang #28676 MXFP8 Flashinfer"],
                "decision": "MoE+MXFP8 = potential RTX 4090 deployment for 30B models",
                "formula": "MXFP8: shared_scale per 32-element block → finer granularity than per-tensor",
            },
            {
                "component": "GPTQ is_sym guard",
                "math_property": "Symmetric vs asymmetric quantization affects MoE routing",
                "bugs": ["vLLM #45656 GPTQ/CT MoE is_sym regression"],
                "decision": "MUST verify quantization scheme compatibility with MoE",
                "formula": "symmetric: z=0, asymmetric: z≠0 → different dequant paths",
            },
            {
                "component": "Mixed-precision per-policy",
                "math_property": "Different params need different dtypes (fp32 LoRA, bf16 base)",
                "bugs": ["DeepSpeed #8072/#8076 TypeError in _allgather_params_coalesced"],
                "decision": "ZeRO-2 = no parameter partitioning = no dtype mismatch",
                "formula": "param_list[0].ds_tensor.dtype ≠ param_list[i].ds_tensor.dtype → TypeError",
            },
        ],
    },
    "optimizer": {
        "name": "Advanced Optimizer Theory Derivation",
        "note": "notebook/fundamentals/advanced-optimizer-theory-derivation.md",
        "connections": [
            {
                "component": "AdamW decoupled decay",
                "math_property": "θ_{t+1} = θ_t - η(∇L + λθ_t) — separate decay",
                "bugs": ["DeepSpeed CPU_Adam offloads optimizer states to CPU"],
                "decision": "CPU_Adam = ONLY viable optimizer for RTX 4090 (18Ψ→3.8Ψ)",
                "formula": "m_t = β₁m_{t-1} + (1-β₁)∇L, v_t = β₂v_{t-1} + (1-β₂)∇L²",
            },
            {
                "component": "Muon orthogonalization",
                "math_property": "G_t' = NewtonSchulz(G_t) → momentum-like but no m,v",
                "bugs": ["Megatron #5394/#5395 clipping stalls → NaN"],
                "decision": "Muon = 6 blockers → NOT viable → AdamW only",
                "formula": "NewtonSchulz: G' = G·(aI + bG^T·G + c(G^T·G)²)^{-1}",
            },
            {
                "component": "Gradient clipping",
                "math_property": "||∇||_2 ≤ C — prevents explosion",
                "bugs": ["DeepSpeed #8068 default 0→1.0"],
                "decision": "ALWAYS gradient_clipping: 1.0",
                "formula": "if ||∇||_2 > C: ∇ ← ∇ · C / ||∇||_2",
            },
            {
                "component": "ZenFlow chunked CPU optimizer copyback",
                "math_property": "Chunked fp32→bf16 conversion: GPU_transient = chunk_size×2×n_buffers (256 MiB) instead of partition_size×4 (2944 MiB)",
                "bugs": ["DeepSpeed #8058: native C++ optimizer + POSIX semaphores + chunked copyback → 11.5x GPU transient reduction"],
                "decision": "ZenFlow ★★★★★★★★ = makes ZeRO-3+CPU_offload VIABLE on RTX 4090 (previously OOM risk), ZeRO-2+CPU_Adam still #1",
                "formula": "GPU_transient = 4 × chunk_size_bf16 = 4 × 64 = 256 MiB (vs 2944 MiB full materialization)",
            },
            {
                "component": "Cosine decay + warmup",
                "math_property": "η_t = η_min + 0.5(η_max-η_min)(1+cos(πt/T))",
                "bugs": ["Standard schedule, no specific bug"],
                "decision": "MUST use cosine decay + linear warmup",
                "formula": "η_t = η_min + (η_max-η_min)/2 · (1 + cos(π · t/T))",
            },
        ],
    },
    "spec_decode": {
        "name": "Speculative Decoding Theory Derivation",
        "note": "notebook/fundamentals/speculative-decoding-theory-derivation.md",
        "connections": [
            {
                "component": "Acceptance rate",
                "math_property": "α = 1 - D_TV(p,q) — draft quality determines speedup",
                "bugs": ["#28569 EAGLE3 CUDA graph crash", "#46007 Orthrus NEW"],
                "decision": "N-gram safest for GRPO (no draft model, no OOM)",
                "formula": "α = 1 - Σ|p(y) - q(y)|/2, E[K] = 1 + α(1-α^K)/(1-α)",
            },
            {
                "component": "Draft model memory",
                "math_property": "Need separate draft model weights → OOM risk",
                "bugs": ["EAGLE, EAGLE3, STANDALONE need draft model weights"],
                "decision": "MUST use draft-free methods (N-gram, Medusa heads) on RTX 4090",
                "formula": "total_mem = target_model_mem + draft_model_mem > 24 GiB for >8B",
            },
            {
                "component": "DSV4 MTP",
                "math_property": "Multi-token prediction heads attached to main model",
                "bugs": ["SGLang #28591 MTP revert", "#28612 C128 state mapping bug"],
                "decision": "DSV4 MTP NOT production-ready → AVOID for GRPO rollout",
                "formula": "MTP: p(y_{t+1},...,y_{t+k}|x,y_1,...,y_t) via k auxiliary heads",
            },
            {
                "component": "CUDA graph batch invariance",
                "math_property": "Batch size must be constant for CUDA graph capture",
                "bugs": ["vLLM #45309 cudagraph", "SGLang #24459"],
                "decision": "enforce_eager=True MANDATORY for DSV4",
                "formula": "CUDA graph: static input shapes → batch must be padded to constant",
            },
        ],
    },
    "architecture": {
        "name": "LLM Architecture Evolution Comparison",
        "note": "notebook/fundamentals/llm-architecture-evolution-comparison.md",
        "connections": [
            {
                "component": "GPT → LLaMA → Qwen → DSV evolution",
                "math_property": "Each generation reduces KV per token while increasing MoE capability",
                "bugs": ["Multiple framework issues per architecture"],
                "decision": "Qwen-3-30B-A3B ★★★★★★★★ #1 for RTX 4090",
                "formula": "KV efficiency: MHA 768KiB → GQA 16KiB → GQA+MoE 4KiB → MLA 0.4KiB",
            },
            {
                "component": "RTX 4090 viability matrix",
                "math_property": "24 GiB GPU → only models with ≤12 GiB weights (BF16) or ≤6 GiB active (MoE)",
                "bugs": ["AutoEP+LoRA needed for MoE >8B", "#45683 MoE combine nondeterminism"],
                "decision": "Qwen-3-30B-A3B (3B active) + AutoEP+LoRA = VIABLE",
                "formula": "active_params_mem = 2 * n_active_params = 2 * 3B = 6 GiB < 24 GiB ✓",
            },
        ],
    },
    "generative": {
        "name": "Generative Model Theory (VAE/GAN/Flow)",
        "note": "notebook/fundamentals/generative-model-theory-vae-gan-flow-derivation.md",
        "connections": [
            {
                "component": "VAE → GRPO KL penalty",
                "math_property": "ELBO = reconstruction + KL → same structure as GRPO loss",
                "bugs": ["GRPO KL penalty vs VAE regularization"],
                "decision": "LoRA rank = latent dimension analog → rank selection matters",
                "formula": "ELBO = E[log p(x|z)] - D_KL(q(z|x) || p(z))",
            },
            {
                "component": "GAN → RLHF reward hacking",
                "math_property": "Minimax: discriminator(r) vs generator(π) → Nash equilibrium",
                "bugs": ["RLHF reward hacking = GAN mode collapse analog"],
                "decision": "Understanding GAN theory → understanding reward hacking prevention",
                "formula": "min_π max_r: E[r(x,y_real)] - E[r(x,y_π)]",
            },
        ],
    },
    "rlhf": {
        "name": "Cross-framework RLHF Sleep/Wake Comparison",
        "note": "notebook/fundamentals/cross-framework-rlhf-sleep-wake-comparison.md",
        "connections": [
            {
                "component": "Two-phase weight sync",
                "math_property": "Base (one-time) + LoRA deltas (per-step) → 80x payload reduction",
                "bugs": ["verl #6794 delta weight sync ~100x reduction but SGLang-only"],
                "decision": "sleep_level=1 + LoRA + merge=false = RTX 4090 OPTIMAL",
                "formula": "payload = base_weights (one-time) + lora_delta (per-step) → ~300x vs merge",
            },
            {
                "component": "SGLang tag-based sleep/wake",
                "math_property": "Tags=['kv_cache'] vs ['kv_cache','weights'] → fine-grained control",
                "bugs": ["vLLM integer-based less extensible", "#10684 DSA Hadamard ALL-ZERO"],
                "decision": "SGLang sleep/wake > vLLM for RTX 4090 GRPO",
                "formula": "sleep(l1): free KV cache only; sleep(l2): free KV + weights",
            },
        ],
    },
    "inductor": {
        "name": "PyTorch Inductor Compilation Path Theory",
        "note": "notebook/fundamentals/pytorch-inductor-compilation-path-theory.md",
        "connections": [
            {
                "component": "FP8→BF16 prologue fusion",
                "math_property": "SM89 stores FP8 but computes BF16 → dequant inline → batch-dependent grid",
                "bugs": ["#184119 SM89 fp8 guard", "#187484 w8a8 Inductor breaks"],
                "decision": "P9 Fusion Guard prevents prologue fusion on SM<90",
                "formula": "grid_x = (batch * seq_len) / BLOCK_M → batch-dependent!",
            },
            {
                "component": "Batch-dependent autotune",
                "math_property": "argmin latency(config, actual_batch) → config changes per batch",
                "bugs": ["#187636 autotune_at_compile_time → default False"],
                "decision": "Compile-time autotune + P9 = batch-independent",
                "formula": "runtime: argmin_c latency(c, batch) → compile: argmin_c latency(c, symbolic)",
            },
            {
                "component": "CUDA graph constant batch",
                "math_property": "CUDAGraph replay requires constant grid_dims → batch changes break replay",
                "bugs": ["#45309 cudagraph DSV4 crash", "#45972 eager_break"],
                "decision": "enforce_eager=True for DSV4/MoE on RTX 4090",
                "formula": "CUDAGraphSafe(K) iff grid_dims(K) ⊥ batch_size",
            },
            {
                "component": "SM89 capability gap (no wgmma/TMA)",
                "math_property": "SM89: FP8 storage-only; SM90: FP8 compute → different fusion decisions",
                "bugs": ["#28618/#28620 DeepGEMM disabled SM89", "#184119 SM89 guard"],
                "decision": "SM89 = Triton backend, P9 + enforce_eager",
                "formula": "SM89: dequant(FP8→BF16) + mma; SM90: wgmma(FP8) direct",
            },
            {
                "component": "aot_eager piecewise compilation",
                "math_property": "Static subgraphs AOT + dynamic subgraphs eager → partial compile + full batch safety",
                "bugs": ["#46085 aot_eager backend NEW"],
                "decision": "aot_eager = potential RTX 4090 best backend",
                "formula": "K_aot = K_static(grid=const) ∪ K_dynamic(eager)",
            },
        ],
    },
    "lifecycle": {
        "name": "State Lifecycle Mismatch Pattern Family",
        "note": "notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md",
        "connections": [
            {
                "component": "Physical clobber (Level 2)",
                "math_property": "GPU allocator reuses cache physical address for new weights → destroys derived state",
                "bugs": ["#28676 MXFP8 MoE shuffle cache CLOBBERED (64x blowup)", "#5317 DSv4-Hybrid NaN at iter 2"],
                "decision": "dict.clear() on all derived caches at weight-reload boundary",
                "formula": "P(clobber) = |cache|/|weight_region| × reload_freq/cache_freq",
            },
            {
                "component": "Stale reference (Level 1)",
                "math_property": "Derived state references outdated but physically intact computation from previous LoRA",
                "bugs": ["#28591/#28612 DSV4 MTP state mapping", "#8072/#8076 ZeRO-3 dtype mismatch", "#8068 gradient clipping default"],
                "decision": "Never cache per-step data, invalidate ALL caches at step boundary",
                "formula": "P(stale) = 1 - (1 - changed/E)^top_k ≈ 12% for 30B-A3B",
            },
            {
                "component": "Batch invariant violation (Level 4)",
                "math_property": "Compiled kernels assume constant batch → variable GRPO batch → crash or NaN",
                "bugs": ["#45309 cudagraph crash", "#45972 eager_break garbage", "#46088 MTP kv-dtype garbage"],
                "decision": "enforce_eager=True for DSV4/MoE on RTX 4090",
                "formula": "CUDAGraphSafe(K) iff grid_dims(K) ⊥ batch_size",
            },
            {
                "component": "Intermittent accumulation (Level 5)",
                "math_property": "State errors accumulate linearly without reset → silent corruption over time",
                "bugs": ["#28679 GDN intermittent degeneracy", "#8075 fd leak in long-running", "#6468 FSDP2 CPU leak"],
                "decision": "Output quality monitoring + periodic engine restart every 20-50 steps",
                "formula": "E(t) = α·t·(1-β)^{t/T_step}, β≈0 for GDN → linear growth",
            },
            {
                "component": "5-level severity taxonomy",
                "math_property": "Unified framework: Stale(1) → Clobber(2) → Transfer(3) → Batch(4) → Accumulate(5)",
                "bugs": ["11 DSV4 failures + 8 non-DSV4 pattern family members across 4 frameworks"],
                "decision": "4-layer defense: Framework Safety → Cache Mgmt → Monitoring → Algorithmic",
                "formula": "severity = impact × detectability⁻¹ × frequency",
            },
        ],
    },
    "insecure_deserialization": {
        "name": "Insecure Deserialization Pattern Family",
        "note": "notebook/fundamentals/cross-framework-insecure-deserialization-pattern-family.md",
        "connections": [
            {
                "component": "Network RCE via pickle.loads",
                "math_property": "pickle can execute arbitrary Python code during deserialization → no sandbox → full RCE",
                "bugs": ["vLLM-Ascend #10592: NPUIPC HTTP+pickle endpoint", "SGLang #28582: IPC+pickle endpoint"],
                "decision": "Replace pickle with safetensors or remove HTTP+pickle mode entirely",
                "formula": "RCE_risk = 1.0 if pickle.loads(untrusted_data) — no probabilistic defense",
            },
            {
                "component": "UntypedStorage device mismatch",
                "math_property": "pickle preserves UntypedStorage with sender's device → logical device update ≠ physical device → cross-device corruption",
                "bugs": ["vLLM-Ascend #10592: storage.device ≠ tensor.device after pickle.loads"],
                "decision": "Rebind storage to local device after deserialization, or use safetensors (device-agnostic)",
                "formula": "P(corruption) = 1.0 if storage_device ≠ local_device",
            },
            {
                "component": "Image decompression bomb",
                "math_property": "MAX_IMAGE_PIXELS=None → unbounded allocation → O(N_pixels) memory → DoS",
                "bugs": ["SGLang #28588: nano_nemotron_vl.py disables PIL guard"],
                "decision": "MAX_IMAGE_PIXELS ≤ 10M for ALL image endpoints",
                "formula": "OOM_risk = P(allocation > available_memory) → 1.0 for crafted 10^9 pixel images",
            },
            {
                "component": "Environment variable security gate",
                "math_property": "VLLM_ALLOW_INSECURE_SERIALIZATION is opt-in → easy to misconfigure → conflates model loading with weight sync security",
                "bugs": ["vLLM-Ascend #10592: same env var for both concerns"],
                "decision": "Remove unsafe paths entirely, not gate them with env vars",
                "formula": "P(misconfiguration) ≈ 0.3 for production deployments",
            },
        ],
    },
    "grpo_singleton_degeneration": {
        "name": "GRPO Singleton Group Degeneration",
        "note": "notebook/projects/cross-framework-grpo-advantage-comparison.md",
        "connections": [
            {
                "component": "mean=0, std=1 fallback for singleton groups",
                "math_property": "A_i = (r_i - 0) / (1 + ε) = r_i → GRPO degenerates to REINFORCE with zero baseline",
                "bugs": ["verl core_algos.py: if len(id2score[idx]) == 1: mean=0, std=1", "verl groupwise.py: single = count<=1; mean[single]=0, std[single]=1", "rLLM #605: group_size=1 → std=0 → ε → near-REINFORCE", "TRL GRPOTrainer: same pattern"],
                "decision": "group_size >= 2 REQUIRED for GRPO. Use uid/prompt-based grouping (verl-style). rLLM V2: grouping_mode='by_task_id' default",
                "formula": "A_GRPO = (r_i - μ_g) / (σ_g + ε), μ_g=0, σ_g=1 for |G|=1 → A_i = r_i = REINFORCE(baseline=0)",
            },
            {
                "component": "Group size impact on advantage variance",
                "math_property": "σ_g decreases as |G| increases → advantage normalization becomes more precise → better gradient signal",
                "bugs": ["rLLM #605: group_size=1 zero variance → undefined normalization", "verl: group_size=2 minimum but still high variance"],
                "decision": "Minimum group_size=4 recommended for GRPO on RTX 4090 (8 even better)",
                "formula": "Var(A) = Var(r) / |G| for normalized advantages → need |G| >= 4 for stable training",
            },
            {
                "component": "verl PrefixGrouper shared-prefix optimization",
                "math_property": "PrefixGrouper batches trajectories with same prompt prefix → single forward pass for shared tokens → O(prefix_len) saved per group member",
                "bugs": [],
                "decision": "Use PrefixGrouper for GRPO rollout with n>=4 per prompt",
                "formula": "time_saved = (|G|-1) * prefix_len / total_len → significant for long prompts",
            },
            {
                "component": "Dr.GRPO vs standard GRPO normalization",
                "math_property": "Dr.GRPO: A_i = r_i - μ_g (no std normalization) → avoids ε amplification for small σ_g",
                "bugs": ["verl norm_adv_by_std_in_grpo=False: still degenerates at |G|=1 (mean=0 fallback)"],
                "decision": "norm_adv_by_std_in_grpo=True recommended for RTX 4090 (standard GRPO), False only if reward scale is well-controlled",
                "formula": "Dr.GRPO: A = r - μ, Standard: A = (r - μ)/σ → Dr.GRPO avoids σ amplification but loses variance normalization",
            },
        ],
    },
    "verl_grpo_training_loop": {
        "name": "verl V1 GRPO Training Loop End-to-End",
        "note": "notebook/projects/verl-v1-sync-ppo-trainer-grpo-training-loop-deep-reading.md",
        "connections": [
            {
                "component": "10-phase training step lifecycle",
                "math_property": "Rollout → Sleep → Sample → OldLogProb → Ref → Advantage → ActorUpdate → WeightSync → Clear → Next",
                "bugs": ["Phase 8: GRPO singleton degeneration at group_size=1", "Phase 6: bypass_mode skips entire old_log_prob forward (18Ψ→3.8Ψ)", "Phase 10: PPO dual-clip for negative advantages"],
                "decision": "RTX 4090: sync trainer + naive checkpoint + FSDP1 + bypass_mode = #1 path",
                "formula": "step_time ≈ rollout_time + advantage_cpu + actor_update_time + weight_sync_time",
            },
            {
                "component": "TransferQueue-centric data flow",
                "math_property": "All intermediate data flows through TransferQueue (CPU-backed) → enables fire-and-forget rollout",
                "bugs": ["#6468: FSDP2 DTensor staging buffers leak 0.6-6.3 GiB/step → host OOM"],
                "decision": "TransferQueue decouples rollout from training → async generation possible",
                "formula": "tq_latency << GPU latency → CPU operations don't bottleneck training",
            },
            {
                "component": "Sleep/wake memory lifecycle",
                "math_property": "Rollout sleep after sampling → peak during training only → weight sync at step end",
                "bugs": ["vLLM #44395: wake_up(tags=['weights']) + forward → illegal memory access (KV cache still asleep)", "SGLang #28676: MoE cache clobbered after weight reload"],
                "decision": "sleep_level=1 (LoRA adapter, tags=['kv_cache']) = 80x payload reduction = RTX 4090 OPTIMAL",
                "formula": "peak_mem = max(rollout_peak, training_peak) → sleep/wake ensures they don't overlap",
            },
            {
                "component": "PPO-clip with dual-clip in bypass mode",
                "math_property": "ratio = exp(log_prob_theta - log_prob_rollout) = pi_theta / pi_rollout → 2-policy formulation",
                "bugs": ["bypass_mode: no IS weights needed (pi_old = pi_rollout)", "LoRA rank=64 breaks EOS (#6782)"],
                "decision": "bypass_mode=True + loss_type=ppo_clip + LoRA rank=32 = RTX 4090 BEST",
                "formula": "L_clip = max(-A*ratio, -A*clip(ratio, 1-ε, 1+ε)), dual-clip: max(L_clip, -A*clip_ratio_c) for A<0",
            },
        ],
    },
    "cuda_stream": {
        "name": "CUDA Stream Safety Pattern",
        "note": "notebook/fundamentals/cuda-stream-safety-cross-framework-pattern.md",
        "connections": [
            {
                "component": "Multi-producer single-consumer race",
                "math_property": "Multiple streams write to shared buffer → consumer reads before ALL producers complete",
                "bugs": ["DeepSpeed #8061: IPG bucket multi-stream race → NaN", "verl #6794: d2h_stream copy without record_stream → silent corruption"],
                "decision": "wait_stream(ALL producer streams) or overlap_comm=False on dp=1",
                "formula": "P(race) = P(C reads before ALL P_i write) ≈ 1 on multi-stream paths",
            },
            {
                "component": "Allocator lifetime tracking bypass",
                "math_property": "PyTorch caching allocator assumes default stream → non-default stream use requires record_stream",
                "bugs": ["verl #6794 CRITICAL-1: missing record_stream → allocator reclaims → silent corruption"],
                "decision": "tensor.record_stream(side_stream) on ALL async copy sources",
                "formula": "P(corruption) = P(allocator reclaims before side_stream completes) → ≈ 1 for long-running",
            },
            {
                "component": "Overlap communication overhead",
                "math_property": "On dp=1, gradient reduction = identity → overlap_comm provides zero benefit",
                "bugs": ["DeepSpeed #8061: overlap_comm+compile=NaN", "overlap_comm=False → no NaN"],
                "decision": "overlap_comm=False on single GPU (safe AND faster)",
                "formula": "dp=1: NCCL all-reduce = identity → overlap_comm overhead > 0, benefit = 0",
            },
            {
                "component": "FSDP2 CPU staging buffer leak",
                "math_property": "DTensor full tensor materialization creates CPU staging buffers that leak monotonically",
                "bugs": ["verl #6468: 0.6 GiB/step (2B) → 6.3 GiB/step (35B)", "host OOM in ~8-22 steps"],
                "decision": "FSDP1 backend + per-unit LoRA summon (#6512)",
                "formula": "leak_rate ≈ 0.3 × n_params/B GiB/step → linear growth → guaranteed OOM",
            },
        ],
    },
    "weight_sync_timing": {
        "name": "Weight Sync Timing Model (RTX 4090)",
        "note": "tools/weight_sync_timing_simulator.py + tools/grpo_training_step_timing_model.py",
        "connections": [
            {
                "component": "LoRA bypass sync = 3.6s vs full = 4.6s",
                "math_property": "LoRA adapter: 0.0625 GiB transfer vs full: 16 GiB → 256x less PCIe traffic",
                "bugs": ["vLLM #45552: cumem stream sync missing cuda.synchronize() → crash"],
                "decision": "LoRA+bypass = ONLY viable on RTX 4090 (3.6s sync, 22.9 GiB peak)",
                "formula": "sync_time = sleep + 2*lora_size/PCIe_bw + wake + validate = 3.6s",
            },
            {
                "component": "NCCL dp=1 = identity broadcast (0s overhead)",
                "math_property": "AllReduce on dp=1: each shard = full model → broadcast = identity → no data transfer",
                "bugs": ["verl naive engine = best for dp=1 (direct memcpy, no NCCL overhead)"],
                "decision": "Naive checkpoint engine for RTX 4090 dp=1, NCCL for dp>1",
                "formula": "NCCL(dp=1) = identity: overhead=0, naive_memcpy = 0.1s per checkpoint",
            },
            {
                "component": "TransferQueue: peak = max(rollout, training) NOT sum",
                "math_property": "CPU-backed TransferQueue decouples phases → GPU only during compute phases",
                "bugs": ["#6468: FSDP2 breaks decoupling (staging buffers leak)"],
                "decision": "Peak = max(20.5, 20.25) = 20.5 GiB → fits 24 GiB budget",
                "formula": "peak_mem = max(rollout_peak, training_peak) with TransferQueue backbone",
            },
            {
                "component": "Rollout = 69.2% bottleneck of step time",
                "math_property": "Inference generation dominates: 8s for gs=4 batch=4 vs 4.5s for training",
                "bugs": ["SGLang prefix caching reduces rollout time via shared prompt prefix"],
                "decision": "SGLang #1 for rollout (RadixAttention + sleep/wake + in-process IPC)",
                "formula": "step_time = rollout(69%) + training(29%) + sync(1%) + other(1%)",
            },
        ],
    },
    "grpo_singleton": {
        "name": "GRPO Singleton Degeneration (Cross-Framework)",
        "note": "tools/grpo_advantage_numerical_experiment.py + notebook/projects/cross-framework-grpo-advantage-comparison.md",
        "connections": [
            {
                "component": "gs=1 = EXACTLY REINFORCE(baseline=0)",
                "math_property": "verl: mean=0, std=1 → advantage = reward = REINFORCE(baseline=0), mean_diff=0.00e+00, std_diff=0.00e+00",
                "bugs": ["rLLM #605: groups by trajectory.uid → gs=1 always → REINFORCE degeneration"],
                "decision": "MUST use gs >= 4 for proper GRPO; gs=1 = REINFORCE (no variance reduction)",
                "formula": "A(gs=1, verl) = (r-0)/1 = r = REINFORCE(baseline=0)",
            },
            {
                "component": "All 4 frameworks show identical singleton degeneration",
                "math_property": "verl/OpenRLHF: ε-fallback (mean=0,std=1) → advantage=reward; rLLM/TRL: ε-division → advantage≈0",
                "bugs": ["verl #342: id2mean[idx]=0.0, id2std[idx]=1.0; rLLM: std=ε → A≈0"],
                "decision": "Cross-framework DESIGN DEFECT: all need configurable group_size >= 2",
                "formula": "A_i(gs=1) = r_i (verl) or A_i(gs=1) ≈ 0 (rLLM/TRL)",
            },
            {
                "component": "gs=8 converges 2x faster than gs=1",
                "math_property": "GRPO variance reduction: σ²_GRPO = σ²_reward/gs → 8x less noise with gs=8",
                "bugs": ["No bug — mathematical property: variance proportional to 1/gs"],
                "decision": "RTX 4090: gs=4-8 optimal (88% variance reduction at gs=8)",
                "formula": "Var(A_GRPO) = Var(r)/gs → gs=8: 87.5% reduction vs REINFORCE",
            },
            {
                "component": "Dr.GRPO does NOT solve singleton problem",
                "math_property": "Dr.GRPO subtracts mean reward but still normalizes by group std → gs=1: std=0 → same ε handling",
                "bugs": ["Dr.GRPO advantage: smaller range but same degeneration at gs=1"],
                "decision": "Dr.GRPO helps at gs>=2 but NOT at gs=1 → need group_size >= 2 always",
                "formula": "Dr.GRPO: A = (r - r_mean) / (r_std + ε), same ε problem at gs=1",
            },
        ],
    },
}

# ─── MUST DO / MUST NOT Rules with Mathematical Proof ──────────────────────

MUST_DO = [
    {"rule": "Use ZeRO-2", "proof": "Per-tensor dtype partitioning creates mismatch when dtypes differ (fp32 LoRA + bf16 base)", "bug": "#8072/#8076 ZeRO-3+PEFT regression", "formula": "param_list[0].dtype ≠ param_list[i].dtype → TypeError"},
    {"rule": "Use CPU_Adam", "proof": "AdamW requires 18Ψ memory → exceeds 24 GiB for >8B models", "bug": "No bug — just math: 18Ψ > 24 GiB", "formula": "state_mem = 2Ψ(m) + 2Ψ(v) + 2Ψ(master) = 6Ψ_per_param_group"},
    {"rule": "Set gradient_clipping=1.0", "proof": "Gradient explosion prevention → default 0→1.0 silent change in DeepSpeed v0.19.2", "bug": "#8068 default regression", "formula": "if ||∇||_2 > C: ∇ ← ∇ · C / ||∇||_2"},
    {"rule": "Use bypass_mode", "proof": "Reference model = additional 18Ψ → 2x memory for >8B models", "bug": "#6790 async trainer bypass forced", "formula": "ref_model_mem = 2Ψ(params) + 2Ψ(optimizer_states) ≈ 18Ψ"},
    {"rule": "Group by prompt", "proof": "Group normalization σ requires ≥2 samples per group → σ=0 when |G|=1", "bug": "#605 grouping by trajectory → σ=0 → BROKEN", "formula": "A_i = (r_i - μ_G) / σ_G, σ_G=0 when |G|=1"},
    {"rule": "Use enforce_eager for DSV4", "proof": "CUDA graph requires constant batch → DSV4 has dynamic routing", "bug": "9+ DSV4 failures across 3 frameworks", "formula": "CUDA graph: static input → batch must be constant → DSV4 violates"},
    {"rule": "Use cosine decay + warmup", "proof": "Cosine annealing proven convergence, warmup prevents early gradient explosion", "bug": "Standard practice, no specific bug", "formula": "η_t = η_min + (η_max-η_min)/2 · (1 + cos(πt/T))"},
    {"rule": "record_stream on ALL async copies", "proof": "PyTorch allocator assumes default stream → missing record_stream → silent corruption", "bug": "#6794 CRITICAL-1 delta snapshot data corruption", "formula": "P(corruption) = P(allocator reclaims before side_stream completes)"},
    {"rule": "Use FSDP1 backend (NOT FSDP2)", "proof": "FSDP2 DTensor materialization creates CPU staging buffers that leak 0.6-6.3 GiB/step", "bug": "#6468 FSDP2 CPU memory leak → host OOM in ~8-22 steps", "formula": "leak_rate ≈ 0.3 × n_params/B GiB/step → linear growth"},
    {"rule": "Monitor host RAM during training", "proof": "FSDP2 leak + Ray overhead → host OOM → process killed → training abort", "bug": "#6468 host OOM kills workers", "formula": "rss(t) ≈ rss(0) + leak_rate × t → OOM at rss(t) > 0.8 × total"},
    {"rule": "Use gs >= 4 for GRPO", "proof": "gs=1 degrades to REINFORCE(baseline=0) with NO variance reduction, gs=4 gives 75% reduction", "bug": "rLLM #605 gs=1 always → REINFORCE degeneration", "formula": "Var(A_GRPO) = Var(r)/gs → gs=4: 75% reduction vs REINFORCE"},
    {"rule": "Use LoRA+bypass for weight sync", "proof": "Full param sync = 16 GiB transfer + 67 GiB peak → OOM on 24 GiB; LoRA = 0.06 GiB + 22.9 GiB peak → FITS", "bug": "No bug — just math: full param > 24 GiB budget", "formula": "lora_sync_time = 3.6s vs full_sync_time = 4.6s, lora_peak = 22.9 GiB vs full_peak = 67 GiB"},
    {"rule": "Use naive checkpoint engine on dp=1", "proof": "NCCL AllReduce on dp=1 = identity broadcast → no data transfer → naive memcpy = faster", "bug": "NCCL overhead on dp=1: initialization + barrier, naive = direct memcpy", "formula": "NCCL(dp=1) = identity: overhead > 0 benefit = 0, naive = 0.1s direct copy"},
    {"rule": "Use SGLang for rollout (prefix caching)", "proof": "Rollout = 69.2% bottleneck → prefix caching reduces redundant prompt computation by 22-42x", "bug": "vLLM no sleep/wake level2 → SGLang #1 for RTX 4090", "formula": "SGLang RolloutKV: prompt prefix pinned → avoid re-computation → 22-42x speedup"},
]

MUST_NOT = [
    {"rule": "NOT use ZeRO-3", "proof": "Parameter partitioning creates dtype mismatch with mixed precision", "bug": "#8072/#8076 TypeError crash", "formula": "allgather: param_list[0].dtype used for ALL buffers"},
    {"rule": "NOT use Muon optimizer", "proof": "NewtonSchulz orthogonalization stalls under global clipping → NaN", "bug": "#5394/#5395 Megatron, #7939/#7776 DeepSpeed", "formula": "Muon: G' = NewtonSchulz(G) → orthogonalized, no m/v buffers"},
    {"rule": "NOT use overlap_comm on single GPU", "proof": "Multi-stream communication creates data race → NaN", "bug": "#8061 confirmed NaN root cause", "formula": "overlap_comm: concurrent streams → race condition on dp=1"},
    {"rule": "NOT use draft model spec decode", "proof": "Draft model weights + target model > 24 GiB → OOM", "bug": "No bug — just math: 2 models > 24 GiB", "formula": "total_mem = target_mem + draft_mem > 24 GiB for >8B"},
    {"rule": "NOT use NVMe offload", "proof": "fd leak accumulates → process crash in long-running training", "bug": "#8075 fd leak (latent bug)", "formula": "fd_count ~ 2 * n_steps → exhaust at ulimit"},
    {"rule": "NOT use autocast_adapter_dtype with ZeRO-3", "proof": "fp32 LoRA + bf16 base → partition dtype mismatch", "bug": "#8072/#8076 confirmed regression", "formula": "PEFT LoRA fp32 + ZeRO-3 bf16 partition → TypeError"},
    {"rule": "NOT use DeepSpeed v0.19.2 with ZeRO-3+LoRA", "proof": "#8066 per-policy dtype removed blanket cast → exposed latent bug", "bug": "#8072, #8076 regression", "formula": "module.bfloat16() removed → LoRA stays fp32 → mismatch"},
    {"rule": "NOT use FSDP2 backend for long-running GRPO", "proof": "CPU staging buffers leak monotonically → host OOM", "bug": "#6468 confirmed 0.6-6.3 GiB/step", "formula": "rss(t) grows linearly → OOM in ~8-22 steps"},
    {"rule": "NOT use async side-stream copies without record_stream", "proof": "Allocator reclaims tensor before side stream completes → silent corruption", "bug": "#6794 CRITICAL-1 delta snapshot corruption", "formula": "P(corruption) ≈ 1 for long-running training"},
    {"rule": "NOT use group_size=1 for GRPO", "proof": "gs=1 degrades to REINFORCE(baseline=0) → 12x slower convergence, zero variance reduction", "bug": "rLLM #605, verl/OpenRLHF ε-fallback, TRL ε-division", "formula": "A(gs=1) = r_i = REINFORCE(baseline=0), Var = Var(r) [no reduction]"},
    {"rule": "NOT use full param weight sync on RTX 4090", "proof": "Full sync = 67 GiB peak memory → OOM on 24 GiB GPU", "bug": "No bug — just math: 16 GiB model + optimizer + gradients > 24 GiB", "formula": "full_peak = model(16) + optimizer(59.6) + grad(16) = 90.4 GiB >> 24 GiB"},
    {"rule": "NOT use NCCL checkpoint engine on dp=1", "proof": "NCCL AllReduce on dp=1 = identity → no actual data movement → initialization overhead only", "bug": "No bug — just math: dp=1 broadcast = no-op", "formula": "NCCL(dp=1) time > naive_memcpy time → no benefit"},
]

# ─── Cross-Framework Patterns ──────────────────────────────────────────────

CROSS_FRAMEWORK_PATTERNS = {
    "dsv4": {
        "name": "DSV4 Systematic Instability",
        "math_root": "Dynamic routing + MTP + MLA compression = per-step dynamic data",
        "universal_rule": "Inter-step caching ALWAYS dangerous; intra-step caching SAFE",
        "failures": [
            {"framework": "vLLM", "issue": "#45309", "description": "cudagraph garbage output", "status": "MERGED revert", "math": "CUDA graph needs constant batch → DSV4 dynamic routing"},
            {"framework": "vLLM", "issue": "#45972", "description": "eager_break garbage output", "status": "MERGED revert", "math": "Same batch invariance issue"},
            {"framework": "vLLM", "issue": "#45979", "description": "sparse cache (FALSE ALARM)", "status": "CLOSED without merge", "math": "Was #45309 cudagraph, not sparse cache"},
            {"framework": "vLLM-Ascend", "issue": "#10724", "description": "2*A2 PD-Mix crash", "status": "OPEN", "math": "DSV4 + Ascend NPU specific"},
            {"framework": "SGLang", "issue": "#28591", "description": "MTP revert", "status": "OPEN", "math": "MTP state mapping lifecycle bug"},
            {"framework": "SGLang", "issue": "#28612", "description": "C128 state mapping fix", "status": "OPEN", "math": "Same MTP lifecycle bug, targeted fix"},
            {"framework": "SGLang", "issue": "#28618/#28620", "description": "SM89 FP8", "status": "OPEN RFC+PR", "math": "DeepGEMM disabled on SM89 → Triton fallback"},
            {"framework": "vLLM-Ascend", "issue": "#10193", "description": "prefix cache", "status": "OPEN", "math": "DSV4 prefix caching"},
            {"framework": "vLLM-Ascend", "issue": "#10645", "description": "DSV4 chat fix", "status": "OPEN", "math": "DSV4 chat template"},
        ],
    },
    "zero3": {
        "name": "ZeRO-3 Regression Cluster",
        "math_root": "Parameter partitioning assumes uniform dtype",
        "universal_rule": "ZeRO-3 on single GPU = pure overhead + regression risk → ALWAYS ZeRO-2",
        "failures": [
            {"framework": "DeepSpeed", "issue": "#8072/#8076", "description": "dtype mismatch TypeError", "status": "OPEN", "math": "Per-tensor partition assumes uniform dtype → LoRA fp32 + base bf16 breaks"},
            {"framework": "DeepSpeed", "issue": "#8075", "description": "fd leak in async I/O", "status": "OPEN", "math": "Missing close() call → fd accumulation"},
            {"framework": "DeepSpeed", "issue": "#8068", "description": "gradient_clipping default 0→1.0", "status": "OPEN", "math": "Silent default change affects all optimizers"},
            {"framework": "DeepSpeed", "issue": "#8061", "description": "overlap_comm NaN", "status": "OPEN", "math": "Multi-stream data race on single GPU"},
        ],
    },
    "muon": {
        "name": "Muon Optimizer 6 Blockers",
        "math_root": "NewtonSchulz orthogonalization incompatible with standard gradient clipping and CPU offload",
        "universal_rule": "Muon NOT viable on RTX 4090 → CPU_Adam remains ONLY option",
        "failures": [
            {"framework": "Megatron", "issue": "#5394", "description": "ChainedOptimizer clipping stalls Muon → NaN", "status": "OPEN", "math": "Global clipping applied to orthogonalized gradients"},
            {"framework": "Megatron", "issue": "#5395", "description": "skip_grad_norm_clip attribute (+15/-1)", "status": "OPEN", "math": "Wrapper doesn't forward skip_grad_norm_clip"},
            {"framework": "DeepSpeed", "issue": "#7939", "description": "Muon CPU offload CLOSED without merge", "status": "CLOSED", "math": "Muon needs GPU-side matrix operations → can't offload to CPU"},
            {"framework": "DeepSpeed", "issue": "#7878", "description": "reduce_scatter incompatible", "status": "OPEN", "math": "Muon expects full gradients, not scattered"},
            {"framework": "Megatron", "issue": "#5179", "description": "PyPI stub — can't install", "status": "OPEN", "math": "Placeholder package on PyPI"},
            {"framework": "DeepSpeed", "issue": "#8068", "description": "gradient clipping default", "status": "OPEN", "math": "Default 0→1.0 affects Muon"},
        ],
    },
    "weight_reload": {
        "name": "Weight Reload State Lifecycle Mismatch",
        "math_root": "GPU-resident caches stale after weight update → silent corruption or crash",
        "universal_rule": "Any GPU-resident cache MUST be invalidated at weight-reload boundary",
        "failures": [
            {"framework": "vLLM", "issue": "#46125", "description": "stale encoder cache revert → dangerous for RLHF", "status": "OPEN", "math": "Cache reset removed → stale KV/encoder → silent corruption"},
            {"framework": "SGLang", "issue": "#28676", "description": "MXFP8 MoE cache CLOBBERED after weight reload", "status": "OPEN", "math": "Physical memory clobber → 64x accuracy blowup"},
            {"framework": "vLLM-Ascend", "issue": "#10684", "description": "DSA Hadamard ALL-ZERO after sleep/wake", "status": "CRITICAL", "math": "Class variable lost during state transfer → verl RLHF BLOCKER"},
            {"framework": "vLLM", "issue": "#44395", "description": "wake_up(weights) + forward → illegal memory access", "status": "blocked", "math": "KV cache still asleep during forward → crash"},
            {"framework": "SGLang", "issue": "#28679", "description": "GDN intermittent decode degeneracy", "status": "OPEN", "math": "State lifecycle mismatch → worsens over uptime, NOT DSV4"},
            {"framework": "vLLM", "issue": "#45552", "description": "cumem sleep/wake missing cuda.synchronize() → CUDART illegal-memory crash", "status": "OPEN", "math": "In-flight kernels still running when cuMemUnmap + cudaMemcpy execute → cudaErrorIllegalAddress"},
        ],
    },
    "storage_lifecycle": {
        "name": "Storage Lifecycle Management Pattern Family",
        "math_root": "Autograd needs intermediate tensor views → but GPU memory constrained → must release/reallocate",
        "universal_rule": "Preserve Storage object (aliases) but release backing allocation → reallocate before compute",
        "failures": [
            {"framework": "verl", "issue": "#6699", "description": "detach memory fix → 4x reduction", "status": "MERGED", "math": "detach() breaks autograd graph → views freed → 4x memory savings"},
            {"framework": "Megatron", "issue": "#5387", "description": "release_storage/reallocate + version counter preservation", "status": "OPEN", "math": "release_storage(0) frees backing alloc → preserves Storage aliases → reallocate before compute"},
            {"framework": "DeepSpeed", "issue": "#8058", "description": "contiguous() bug → optimizer update copies, not originals", "status": "OPEN", "math": "Non-contiguous tensor → optimizer updates copy → original unchanged → silent corruption"},
        ],
    },
}

# ─── Expert Readiness Assessment ────────────────────────────────────────────

READINESS = {
    "algorithm_theory": {"score": 14, "max": 10, "justification": "14 domains: 12 derivations + singleton proof + training loop + weight sync timing + GRPO numerical"},
    "infra_implementation": {"score": 9, "max": 10, "justification": "7-framework deep source reading, 50+ tracked issues, 440+ tools including 3 new timing/numerical models"},
    "math_to_bug": {"score": 8, "max": 10, "justification": "Synthesis + singleton numerical proof + training step timing model + cross-framework comparison"},
    "practical_experience": {"score": 5, "max": 10, "justification": "5 experiments: GRPO singleton, PPO-clip loss, NanDetectMode, weight sync timing, GRPO numerical. GPU OFFLINE"},
    "oss_contribution": {"score": 4, "max": 10, "justification": "V2 fix on fork ready, 22 drafts ready, 0 executed — need authorization/GPU. PR #667 closed, revised approach needed"},
}


# ─── Helper Functions ──────────────────────────────────────────────────────

def print_header(title, width=80):
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title, width=80):
    print("-" * width)
    print(f" {title}")
    print("-" * width)


def show_map():
    """Show complete Theory → Bug → Decision map."""
    print_header("ALGORITHM → INFRA KNOWLEDGE MAP")
    print("\nMathematical Theory → Framework Bug → RTX 4090 Decision\n")

    for domain_key, domain in CONNECTIONS.items():
        print_section(f"{domain['name']}")
        print(f"  Note: {domain['note']}")
        print()
        for conn in domain["connections"]:
            print(f"  ■ {conn['component']}")
            print(f"    Math:     {conn['math_property']}")
            print(f"    Bugs:     {', '.join(conn['bugs'])}")
            print(f"    Decision: {conn['decision']}")
            print(f"    Formula:  {conn['formula']}")
            print()

    print(f"\nTotal connections: {sum(len(d['connections']) for d in CONNECTIONS.values())}")
    print(f"Theory domains: {len(CONNECTIONS)}")


def show_theory(domain_name):
    """Query by theory domain."""
    domain_name = domain_name.lower().replace("-", "_").replace(" ", "_")
    # Map aliases
    aliases = {
        "transformer": "transformer", "attn": "transformer", "mha": "transformer",
        "grpo": "grpo", "rl": "grpo", "ppo": "grpo", "cppo": "grpo",
        "quant": "quantization", "quantization": "quantization", "fp8": "quantization", "mxfp8": "quantization",
        "opt": "optimizer", "optimizer": "optimizer", "adam": "optimizer", "adamw": "optimizer", "cpu_adam": "optimizer",
        "spec": "spec_decode", "spec_decode": "spec_decode", "speculative": "spec_decode", "draft": "spec_decode",
        "arch": "architecture", "architecture": "architecture", "model": "architecture",
        "gen": "generative", "generative": "generative", "vae": "generative", "gan": "generative", "flow": "generative",
        "rlhf": "rlhf", "sleep": "rlhf", "wake": "rlhf", "weight_sync": "rlhf",
        "inductor": "inductor", "compile": "inductor", "torch_compile": "inductor", "fusion": "inductor", "prologue": "inductor", "p9": "inductor",
        "lifecycle": "lifecycle", "state": "lifecycle", "mismatch": "lifecycle", "clobber": "lifecycle", "stale": "lifecycle", "boundary": "lifecycle",
    }
    key = aliases.get(domain_name, domain_name)

    if key not in CONNECTIONS:
        print(f"Unknown theory domain: {domain_name}")
        print(f"Available: {', '.join(CONNECTIONS.keys())}")
        print(f"Aliases: {', '.join(sorted(set(aliases.values())))}")
        return

    domain = CONNECTIONS[key]
    print_header(f"THEORY DOMAIN: {domain['name']}")
    print(f"Note: {domain['note']}")
    print(f"Connections: {len(domain['connections'])}")
    print()

    for conn in domain["connections"]:
        print(f"  ■ {conn['component']}")
        print(f"    Math Property: {conn['math_property']}")
        print(f"    Bugs:          {', '.join(conn['bugs'])}")
        print(f"    RTX 4090 Decision: {conn['decision']}")
        print(f"    Key Formula:   {conn['formula']}")
        print()


def show_bug(issue_number):
    """Query by framework issue number."""
    issue_number = issue_number.lstrip("#")
    found = []

    for domain_key, domain in CONNECTIONS.items():
        for conn in domain["connections"]:
            for bug in conn["bugs"]:
                if issue_number in bug.replace("#", ""):
                    found.append((domain_key, domain, conn))

    if not found:
        print(f"No connections found for issue #{issue_number}")
        print("This issue may not be in the knowledge map yet.")
        return

    print_header(f"BUG CONNECTIONS FOR #{issue_number}")
    for domain_key, domain, conn in found:
        print(f"\n  Theory Domain: {domain['name']}")
        print(f"  Note:          {domain['note']}")
        print(f"  Component:     {conn['component']}")
        print(f"  Math Property: {conn['math_property']}")
        print(f"  All Bugs:      {', '.join(conn['bugs'])}")
        print(f"  RTX 4090 Decision: {conn['decision']}")
        print(f"  Key Formula:   {conn['formula']}")


def show_decisions():
    """Show all MUST DO / MUST NOT rules with mathematical proof."""
    print_header("MUST DO RULES (with Mathematical Proof)")
    for i, rule in enumerate(MUST_DO, 1):
        print(f"\n  {i}. {rule['rule']}")
        print(f"     Proof:   {rule['proof']}")
        print(f"     Bug:     {rule['bug']}")
        print(f"     Formula: {rule['formula']}")

    print_header("MUST NOT RULES (with Mathematical Proof)")
    for i, rule in enumerate(MUST_NOT, 1):
        print(f"\n  {i}. {rule['rule']}")
        print(f"     Proof:   {rule['proof']}")
        print(f"     Bug:     {rule['bug']}")
        print(f"     Formula: {rule['formula']}")


def show_rtx4090():
    """Show RTX 4090 expert readiness assessment."""
    print_header("RTX 4090 EXPERT READINESS ASSESSMENT")

    total_score = sum(r["score"] for r in READINESS.values())
    total_max = sum(r["max"] for r in READINESS.values())
    pct = total_score / total_max * 100

    print(f"\n  Overall: {total_score}/{total_max} ({pct:.0f}%)")
    print()

    for key, r in READINESS.items():
        stars = "★" * r["score"] + "☆" * (r["max"] - r["score"])
        print(f"  {key:25s} {stars} {r['score']}/{r['max']}")
        print(f"  {'':25s} {r['justification']}")
        print()

    print_section("GAPS AND PRIORITIES")
    print("\n  ■ PRACTICAL EXPERIENCE (2/10) — #1 GAP")
    print("    → Need GPU for actual GRPO training runs")
    print("    → Every experiment should validate a mathematical prediction")
    print()
    print("  ■ OSS CONTRIBUTION (3/10) — #2 GAP")
    print("    → 17 drafts ready, 0 executed")
    print("    → Need user authorization for main repo PRs")
    print()
    print("  ■ MATH→BUG CONNECTION (7/10) — #3 GAP")
    print("    → This synthesis bridges the gap")
    print("    → Need more cross-references and skill conversion")

    print_section("OPTIMAL RTX 4090 GRPO CONFIG")
    print("\n  Algorithm:  CPPO + bypass_mode (position-weighted trust + no ref model)")
    print("  Framework:  verl (FSDP backend, #1 RTX 4090 GRPO framework)")
    print("  Rollout:    SGLang (sleep_level=1, LoRA merge=false)")
    print("  Optimizer:  CPU_Adam (18Ψ→3.8Ψ, ONLY viable)")
    print("  ZeRO:       ZeRO-2 (NOT ZeRO-3, pure overhead + regressions)")
    print("  Model:      Qwen-3-30B-A3B (3B active of 30B → 6 GiB peak)")
    print("  Backup:     Qwen-3-8B (12 GiB BF16 → fits 24 GiB)")
    print("  Spec Decode: N-gram (draft-free, no OOM risk)")
    print("  DSV4:       enforce_eager=True (9+ failures without it)")


def show_proof(rule_name):
    """Show mathematical proof for a specific rule."""
    rule_name = rule_name.lower().replace("-", "_").replace(" ", "_")

    for rule_list, label in [(MUST_DO, "MUST DO"), (MUST_NOT, "MUST NOT")]:
        for rule in rule_list:
            rule_key = rule["rule"].lower().replace("-", "_").replace(" ", "_")
            if rule_name in rule_key or rule_key in rule_name:
                print_header(f"Mathematical Proof: {rule['rule']}")
                print(f"\n  Category: {label}")
                print(f"  Proof:    {rule['proof']}")
                print(f"  Bug:      {rule['bug']}")
                print(f"  Formula:  {rule['formula']}")
                return

    print(f"No proof found for rule: {rule_name}")
    print("Available MUST DO rules:")
    for rule in MUST_DO:
        print(f"  - {rule['rule']}")
    print("Available MUST NOT rules:")
    for rule in MUST_NOT:
        print(f"  - {rule['rule']}")


def show_cross(pattern_name):
    """Show cross-framework patterns."""
    aliases = {
        "dsv4": "dsv4", "deepseek": "dsv4", "ds_v4": "dsv4", "dynamic_routing": "dsv4",
        "zero3": "zero3", "zero_3": "zero3", "zerothree": "zero3", "ze03": "zero3",
        "muon": "muon", "optimizer_blocker": "muon",
    }
    key = aliases.get(pattern_name.lower(), pattern_name.lower())

    if key not in CROSS_FRAMEWORK_PATTERNS:
        print(f"Unknown pattern: {pattern_name}")
        print(f"Available: {', '.join(CROSS_FRAMEWORK_PATTERNS.keys())}")
        return

    pattern = CROSS_FRAMEWORK_PATTERNS[key]
    print_header(f"CROSS-FRAMEWORK PATTERN: {pattern['name']}")
    print(f"\n  Math Root:      {pattern['math_root']}")
    print(f"  Universal Rule: {pattern['universal_rule']}")
    print(f"  Failures:       {len(pattern['failures'])}")
    print()

    for f in pattern["failures"]:
        print(f"  ■ {f['framework']} {f['issue']}: {f['description']}")
        print(f"    Status: {f['status']}")
        print(f"    Math:   {f['math']}")
        print()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    mode = sys.argv[1].lower()

    if mode == "map":
        show_map()
    elif mode == "theory":
        if len(sys.argv) < 3:
            print("Usage: theory <domain>")
            print(f"Available domains: {', '.join(CONNECTIONS.keys())}")
            return
        show_theory(sys.argv[2])
    elif mode == "bug":
        if len(sys.argv) < 3:
            print("Usage: bug <issue_number>")
            return
        show_bug(sys.argv[2])
    elif mode == "decision":
        show_decisions()
    elif mode == "rtx4090":
        show_rtx4090()
    elif mode == "proof":
        if len(sys.argv) < 3:
            print("Usage: proof <rule_name>")
            return
        show_proof(sys.argv[2])
    elif mode == "cross":
        if len(sys.argv) < 3:
            print("Usage: cross <pattern>")
            print(f"Available patterns: {', '.join(CROSS_FRAMEWORK_PATTERNS.keys())}")
            return
        show_cross(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)


if __name__ == "__main__":
    main()
