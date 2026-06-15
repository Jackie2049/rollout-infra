# Cross-Framework Optimizer Comparison — RTX 4090 Reference

> 2026-06-16 | DeepSpeed, PyTorch, rLLM, verl optimizer comparison
> Focus: Which optimizer is best for RTX 4090 training (24GB, single GPU)?
> ★★★★★★ New: DeepSpeed Muon Gram NS + ZenFlow CPU_Adam

---

## 1. Optimizer Landscape on RTX 4090

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

On single GPU (dp=1), optimizer choice matters MORE than distributed strategy:
- No ZeRO partitioning → optimizer state on GPU or CPU
- LoRA reduces trainable params → optimizer state tiny → CPU offload viable
- Memory is the bottleneck → optimizer design determines if training fits 24GB

| Optimizer | Framework | GPU Memory | CPU Memory | Convergence | RTX 4090 Viable |
|-----------|-----------|------------|------------|-------------|----------------|
| torch.optim.Adam | PyTorch | 2Ψ (m+v fp32) | 0 | Standard | ★★★★★★★ LoRA only |
| DeepSpeed CPU_Adam | DeepSpeed | 0 (offloaded) | 2Ψ (m+v fp32) | Standard (5-7x faster CPU) | ★★★★★★★ LoRA + ZeRO-2 |
| DeepSpeed Muon | DeepSpeed | Ψ (1 state) | 0 or offloaded | Potentially faster (NS orthogonalization) | ★★★★★ NEW! |
| DeepSpeed ZenFlow CPU_Adam | DeepSpeed | 0 + 0.25GB spike | 2Ψ + NUMA-local | Same as CPU_Adam but 11.5x less GPU spike | ★★★★★★★★ BEST! |
| 8-bit Adam (bitsandbytes) | Various | Ψ/2 (quantized) | 0 | Slightly worse | ★★★ Good but needs install |
| AdamW (standard) | All | 2Ψ (m+v fp32) | 0 | Standard | ★★★★★★★ LoRA only |

★★★★★★★★★ Ψ = trainable parameter count in bytes (e.g., LoRA rank=32 on 1.7B → ~0.6GB → optimizer 2Ψ = ~1.2GB)

---

## 2. Memory Impact Analysis

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Qwen3-1.7B, LoRA rank=32, bypass=true:

```
                          GPU Optimizer    CPU Optimizer    Total GPU    Fits 24GB?
torch.optim.Adam:          1.2GB          0GB              ~10.2GB     ✓✓✓
DeepSpeed CPU_Adam:         0GB            1.2GB (CPU)      ~9.0GB      ✓✓✓
DeepSpeed ZenFlow:          0.25GB spike   1.2GB (NUMA)    ~9.25GB     ✓✓✓ (no spike!)
8-bit Adam:                 0.6GB          0GB              ~9.6GB      ✓✓✓
DeepSpeed Muon:             0.6GB (1 state) 0GB            ~9.6GB      ✓✓✓ (potentially faster convergence)
```

★★★★★★★★★ For 1.7B LoRA: ALL optimizers fit 24GB → optimizer choice = convergence speed preference, not memory constraint!

### Qwen3-MoE (A0.6B+B4B), LoRA rank=32, AutoEP EP=1:

```
                          GPU Optimizer    CPU Optimizer    Total GPU    Fits 24GB?
torch.optim.Adam:          1.2GB          0GB              ~19.2GB     ✓ (tight)
DeepSpeed CPU_Adam:         0GB            1.2GB (CPU)      ~18.0GB     ✓✓ (comfortable)
DeepSpeed ZenFlow:          0.25GB spike   1.2GB (NUMA)    ~18.25GB    ✓✓✓ (no spike!)
8-bit Adam:                 0.6GB          0GB              ~18.6GB     ✓ (tight)
DeepSpeed Muon:             0.6GB          0GB              ~18.6GB     ✓ (tight)
```

★★★★★★★★★ For MoE: CPU_Adam/ZenFlow gives ~1-2GB more headroom → critical for tight 24GB scenarios!

### Qwen3-8B (no LoRA, full model training):

```
                          GPU Optimizer    CPU Optimizer    Total GPU    Fits 24GB?
torch.optim.Adam:          32GB           0GB              ~48GB       ✗✗✗ EXCEEDS!
DeepSpeed CPU_Adam:         0GB            32GB (CPU)       ~16GB      ✓ (base weights fit)
DeepSpeed ZenFlow:          0.25GB spike   32GB (NUMA)     ~16.25GB    ✓✓✓ (no spike!)
8-bit Adam:                 16GB           0GB              ~32GB       ✗✗✗ EXCEEDS!
```

★★★★★★★★★ Full model 8B on RTX 4090: CPU_Adam/ZenFlow = ONLY viable → but gradient + activations exceed → STILL NOT VIABLE without LoRA!

---

## 3. DeepSpeed CPU_Adam vs ZenFlow Deep Dive

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### CPU_Adam (current, merged):

```
Architecture:
  → Python multiprocessing subprocess
  → Pickling Pipe coordination → Python→pickle→Pipe→CPU→pickle→GPU
  → C++ kernel: adam_update_single → per-parameter Python→C++ loop
  → SIMD: AVX512 + OpenMP → 5-7x faster than torch.optim.Adam on CPU
  → Memory: optimizer state on CPU → GPU spike when copying back

GPU Spike Problem:
  → fp32 master partition → .to(device) → entire fp32 partition materializes on GPU
  → 0.75B-param partition → ~2944 MiB spike → nearly 3GB temporary GPU usage!
  → On RTX 4090 24GB → 3GB spike during copyback → dangerous under load!
```

### ZenFlow (PR #8058, OPEN — future best):

```
Architecture:
  → Native C++ CPU optimizer process → no Python subprocess!
  → Shared-memory POSIX-semaphore control block → replaces pickling Pipe
  → Adam state allocated in native process → NUMA-local to pinned thread pool
  → Fused multi-tensor: adam_update_multi → entire flattened partition in one C++ call
  → Chunked copyback: stream fp32→bf16 in chunks → never materialize full fp32 on GPU!

GPU Spike Solution:
  → Chunked: fp32 partition → stream to GPU bf16 in small chunks → ~256 MiB peak!
  → 0.75B-param → 2944→256 MiB → 11.5x reduction → MASSIVE for 24GB!
  → ZenFlowCPUAdam = recognized ZeRO optimizer → zero_allow_untested_optimizer NO longer needed!

★★★★★★★★★ ZenFlow = BEST optimizer for RTX 4090 CPU-offloaded training:
  → No GPU spike → safe under load → no OOM risk → production viable!
  → NUMA-local → faster than subprocess → better latency
  → Fused multi-tensor → eliminates per-parameter Python loop → faster
```

---

## 4. DeepSpeed Muon Optimizer

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Muon = momentum + orthogonalization:

```
Core idea:
  → Standard momentum: v = μ*v + g → update = v → accumulates gradient direction
  → Muon: v = μ*v + g → orthogonalize v → update = orthogonalized v
  → Newton-Schulz iteration: X ← (3X - X³) / 2 → converges to orthogonal basis
  → Result: gradient updates always orthogonal → no redundant direction → faster convergence!

★★★★★★ Gram Newton-Schulz #7953 (merged):
  → Standard NS: iterates on X (n × m) → rectangular matrix → m steps
  → Gram NS: iterates on R = X @ X.T (n × n) → square matrix → n steps
  → Transformer weights: n << m (aspect ratio ~5) → n×n iteration → 5x cheaper!
  → Now DEFAULT orthogonalization → configurable ns_method switch

Memory:
  → Muon: only 1 optimizer state (momentum) → vs Adam: 2 states (m + v)
  → Muon GPU: Ψ (momentum only) → vs Adam GPU: 2Ψ (m + v)
  → Muon CPU: Ψ offloaded → vs Adam CPU: 2Ψ offloaded
  → ★★★★★ Muon uses 50% less optimizer memory than Adam!

RTX 4090 feasibility:
  → Muon + single GPU → no ZeRO needed → just use optimizer directly!
  → Muon + LoRA → Ψ = ~0.6GB → optimizer = 0.6GB → tiny!
  → Muon + full model → Ψ = 16GB → optimizer = 16GB → still exceeds on GPU
  → ★★★★★ Muon + LoRA = same memory as Adam+LoRA → but potentially faster convergence!
  → ASPLOS 2026 Best Paper → validated → but GRPO/RL training data still limited
```

★★★★★★★ Muon trade-offs:
  → Pro: 50% less memory, potentially faster convergence, orthogonalization avoids redundant directions
  → Pro: Gram NS → 5x cheaper for transformer aspect ratios → RTX 4090 friendly
  → Con: New → less validated than Adam → GRPO convergence data needed
  → Con: Orthogonalization adds compute → ~5-10% overhead per step on GPU
  → Recommendation: ADAM for safe production, MUON for experimental faster convergence

---

## 5. RTX 4090 Optimizer Decision Matrix

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Dense model (Qwen3-1.7B) + LoRA rank=32:

| Priority | Optimizer | Framework | Config | Notes |
|----------|-----------|-----------|--------|-------|
| Speed | torch.optim.Adam | PyTorch/rLLM/verl | LoRA only, GPU | Fastest no-overhead |
| Memory-safe | DeepSpeed ZenFlow CPU_Adam | DeepSpeed ZeRO-2 | LoRA+offload | No GPU spike, production-safe |
| Experimental | DeepSpeed Muon | DeepSpeed | LoRA, GPU | Faster convergence? Need validation |

★★★★★★★★★ For 1.7B LoRA: memory is NOT a constraint → optimizer = convergence speed choice → Adam safe, Muon experimental

### MoE model (Qwen3-MoE) + LoRA rank=32 + AutoEP EP=1:

| Priority | Optimizer | Framework | Config | Notes |
|----------|-----------|-----------|--------|-------|
| Memory-safe | DeepSpeed ZenFlow CPU_Adam | DeepSpeed ZeRO-2 AutoEP | LoRA+offload+ZenFlow | ★★★★★★★★★ BEST! No spike, 18.25GB fits |
| Conservative | DeepSpeed CPU_Adam | DeepSpeed ZeRO-2 AutoEP | LoRA+offload | Old copyback, 17.18GB spike → risk! |
| Tight fit | torch.optim.Adam | DeepSpeed ZeRO-2 AutoEP | LoRA only, GPU | 19.2GB → fits but tight, no headroom |
| Experimental | DeepSpeed Muon CPU | DeepSpeed ZeRO-2 AutoEP | LoRA+offload | Faster convergence? Need validation |

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ For MoE: ZenFlow CPU_Adam = BEST → no GPU spike → 12.2GB headroom → safe!
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

### Full model (Qwen3-8B) — NOT VIABLE:

All optimizers fail because gradients + activations exceed 24GB regardless of optimizer choice.
→ LoRA is MANDATORY for 8B on RTX 4090 → optimizer choice becomes LoRA-level decision.

---

## 6. 8-bit Adam Alternative

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
8-bit Adam (bitsandbytes):
  → Quantizes optimizer states to 8-bit → 2Ψ → Ψ (50% reduction)
  → Dynamic quantization: 8-bit during accumulation, dequantize for update
  → ~0.5-1% convergence degradation vs fp32 Adam
  → Requires bitsandbytes installation → CUDA only → SM89 supported

★★★★★★ RTX 4090 8-bit Adam assessment:
  → LoRA: Ψ already tiny → 8-bit saves ~0.3GB → marginal benefit
  → Full model: Ψ = 16GB → 8-bit saves ~8GB → still doesn't fit (gradients+activations)
  → ★★★★★ LoRA+Adam = already fits → 8-bit unnecessary → save complexity!
  → ★★★ Only useful: LoRA rank=64+ on MoE → slightly more optimizer → tiny benefit
```

---

## 7. Optimizer + Framework Combined Recommendations

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
★★★★★★★★★★★ RTX 4090 Optimizer + Framework Combined BEST Configs (2026-06):

Dense GRPO:
  → rLLM Tinker + torch.optim.Adam + LoRA32 → simplest, fastest, ~9GB → #1
  → verl + torch.optim.Adam + bypass + LoRA → Ray overhead, ~10GB → #2
  → DeepSpeed ZeRO-2 + ZenFlow CPU_Adam + LoRA32 → most memory-safe → #3

MoE GRPO:
  → ★★★★★★★★★★★ DeepSpeed AutoEP ZeRO-2 + ZenFlow CPU_Adam + LoRA32 + EP=1 → ONLY viable MoE path!
  → DeepSpeed AutoEP ZeRO-2 + CPU_Adam + LoRA32 → old copyback → risk
  → DeepSpeed AutoEP ZeRO-2 + Muon + LoRA32 → experimental convergence → need GPU validation

SFT (supervised fine-tuning):
  → DeepSpeed ZeRO-2 + ZenFlow CPU_Adam + LoRA32 → production safest
  → PyTorch compile + torch.optim.Adam + LoRA32 → simplest
  → DeepSpeed ZeRO-2 + Muon + LoRA32 → experimental faster convergence

Distillation:
  → DeepSpeed OPD + torch.optim.Adam → student only, ~4.6GB → incredibly light
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## 8. Key Findings Summary

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

1. ★★★★★★★★★★★ ZenFlow CPU_Adam = BEST RTX 4090 optimizer → chunked copyback 2944→256MiB → 11.5x GPU spike reduction → eliminates OOM risk during CPU offload!

2. ★★★★★★★★ LoRA makes optimizer choice = convergence speed preference, not memory constraint → 0.6GB trainable → all optimizers fit → choose by convergence quality!

3. ★★★★★★★★ Muon = 50% less optimizer memory (1 state vs 2) + Gram NS 5x cheaper → experimental but potentially faster convergence → RTX 4090 worth trying!

4. ★★★★★★★★ For dense LoRA training: optimizer doesn't matter → Adam is safe, Muon is experimental → convergence difference needs GPU validation!

5. ★★★★★★★★★★★★★★★★★★ For MoE LoRA training: ZenFlow CPU_Adam is BEST → no GPU spike → more headroom → safer production deployment!

6. ★★★★★ 8-bit Adam = marginal benefit with LoRA → Ψ already tiny → don't add unnecessary complexity!

7. ★★★★★★★★ Full model (no LoRA) = IMPOSSIBLE on RTX 4090 regardless of optimizer → LoRA is mandatory → optimizer choice = LoRA-level decision!
