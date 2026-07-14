# ZeRO Memory Optimization for RTX 4090 GRPO — Complete Guide

## Motivation

RTX 4090 has 24 GiB VRAM. Training models >7B parameters requires aggressive memory optimization. ZeRO (Zero Redundancy Optimizer) is the primary mechanism, but its memory behavior differs dramatically across stages, especially after ZenFlow chunked copyback.

## ZeRO Stage Memory Breakdown

### Baseline (no ZeRO)
Each model parameter exists in 5 states during training:
```
param (fp16) + grad (fp16) + optimizer_state (fp32) + master_param (fp32) + momentum (fp32) + variance (fp32)
= 2 + 2 + 4 + 4 + 4 + 4 = 20 bytes per parameter
```

For a 7B model: `7e9 × 20 = 140 GiB` → impossible on single GPU.

With LoRA (r=32, target q_proj, v_proj): only ~0.5% of params trainable:
```
LoRA params: 7e9 × 0.005 ≈ 35M trainable params
Full model: 7B params at fp16 = 14 GiB (frozen, no grad/optimizer state)
LoRA: 35M × 20 = 0.7 GiB
Total: ~14.7 GiB ✓ fits on 24 GiB
```

### ZeRO Stage 1: Optimizer State Sharding
```
ZeRO-1: optimizer_state (fp32) + momentum + variance sharded across DP ranks
Per rank: 4 + 4 + 4 = 12 bytes per param → / DP_rank
```
On dp=1 (RTX 4090 single GPU): `shard_factor = 1 → no reduction`.
**ZeRO-1 on single GPU = pure overhead.** Skip it.

### ZeRO Stage 2: Gradient Sharding
```
ZeRO-2: gradients sharded in addition to optimizer states
Per rank: grad (fp16) sharded → / DP_rank
```
On dp=1: again no sharding benefit. But ZeRO-2 enables **gradient checkpointing compatibility** that reduces activation memory.

### ZeRO Stage 3: Parameter Sharding
```
ZeRO-3: parameters sharded → all-gather on forward/backward
Per rank: param (fp16) sharded → / DP_rank
```
On dp=1: shard_factor = 1 → **ALL parameters still fully materialized**.
ZeRO-3 on single GPU = pure communication overhead. **MUST NOT use.**

## The RTX 4090 Dilemma

On single GPU (dp=1), standard ZeRO provides NO memory benefit:
```
ZeRO-1 on dp=1: same memory as no ZeRO
ZeRO-2 on dp=1: same memory as no ZeRO
ZeRO-3 on dp=1: same memory + communication overhead
```

**So where do the savings come from?**
1. CPU_Adam optimizer offload (moves optimizer states to CPU)
2. ZenFlow chunked copyback (reduces GPU memory spikes during optimizer step)
3. LoRA (reduces trainable parameter count)

## CPU_Adam: The Primary Memory Mechanism

CPU_Adam offloads optimizer states (fp32 master weights, momentum, variance) to CPU RAM:

```
Without CPU_Adam on GPU:
  param (fp16) + grad (fp16) + master_fp32 + momentum + variance
  = 2 + 2 + 4 + 4 + 4 = 16 bytes per trainable param on GPU

With CPU_Adam:
  param (fp16) + grad (fp16) on GPU  (4 bytes)
  master_fp32 + momentum + variance on CPU  (12 bytes)

GPU memory: 4 bytes per trainable param
```

For 7B model + LoRA (35M trainable):
```
GPU: 7B × 2 (frozen fp16) + 35M × 4 (LoRA fp16+grad) = 14.0 + 0.14 = 14.14 GiB ✓
CPU: 35M × 12 = 0.42 GiB (trivial for 64GB system RAM)
Still need activation memory (~4-6 GiB for 512-seqlen)
Total GPU: ~18-20 GiB ✓ fits on 24 GiB!
```

**The challenge**: During the optimizer step, gradients are copied to CPU. The pre-ZenFlow implementation copies ALL gradients at once, creating a GPU memory spike:

```
Pre-ZenFlow optimizer step GPU spike:
  gradients gathering on GPU before CPU transfer:
  35M × 2 (fp16 grad) = 0.07 GiB  -- trivial for LoRA
  BUT for full fine-tune (7B trainable):
  7B × 2 (fp16 grad) = 14 GiB spike + 14 GiB model + 4-6 GiB activations = 32-34 GiB → OOM!
```

## ZenFlow Chunked Copyback — The Fix

**PR #8058** (MERGED July 7, 2026): ZenFlow replaces bulk gradient-to-CPU transfer with chunked copyback.

### Mechanism

```
Before ZenFlow:
  optimizer.step():
    gather ALL gradients on GPU  ← 14 GiB peak!
    copy ALL to CPU
    CPU_Adam.update()  ← on CPU
    free all on GPU

After ZenFlow (chunked):
  optimizer.step():
    for each chunk of params:
      gather chunk gradients on GPU  ← 256 MiB peak!
      copy chunk to CPU
      free chunk from GPU
      CPU_Adam.update(chunk)  ← on CPU, pipelineable
    [chunk N → next step forward can overlap]
```

### Memory Impact

| Configuration | Pre-ZenFlow GPU Spike | Post-ZenFlow | Reduction |
|--------------|----------------------|--------------|-----------|
| ZeRO-2 + Full 7B | 14.0 GiB | 256 MiB | 56× |
| ZeRO-2 + LoRA 7B | 0.07 GiB | 0.07 GiB (unchanged) | 1× |
| ZeRO-3 + Full 7B | 14.0 GiB + param gather | 14.0 GiB + param gather | **NO CHANGE** |

### Critical Limitation: Stage 3 NOT Improved

**ZenFlow chunked copyback ONLY applies to ZeRO Stage 1 and 2.** Stage 3 still:
1. All-gathers ALL parameters from all DP ranks
2. Materializes full fp32 gradients
3. Performs optimizer update on fully materialized params
4. Discards and re-shards

For ZeRO-3 on dp=1:
```
Forward: gather_all_params → compute → shard
Backward: gather_all_params → compute_grad → shard
Optimizer: gather_all_grads → fp32 materialize → update → shard
Peak: 14 GiB (model fp16) + 14 GiB (grad fp16) + 28 GiB (fp32 opt) = 56 GiB → OOM on 24 GiB!
```

**ZeRO-3 on single GPU is NEVER viable**, with or without ZenFlow.

## overlap_comm NaN Bug (#8061)

**Status**: OPEN, maintainers engaged (hwchen2017, cx2009). Production-confirmed.

### Root Cause

When `overlap_comm=True`, ZeRO-2 overlaps gradient all-reduce with backward computation using multiple CUDA streams. The gradient bucket is ready as soon as all producers on its stream finish:

```python
# Simplified: each gradient bucket uses a stream
for bucket in gradient_buckets:
    bucket.stream = get_stream(bucket_id)
    with torch.cuda.stream(bucket.stream):
        # ... backward produces grad on bucket.stream
        bucket.all_reduce()  # NCCL on bucket.stream
```

**The bug**: `average_tensor()` (which divides all-reduced gradients by DP_rank) only waits for the **current stream**, not the producer stream. When the optimizer reads the gradient bucket before the producer stream finishes:

```python
# In optimizer.step():
grad = bucket.grad  # ← may not be ready!
# average_tensor() reads bucket.grad on current stream
# bucket.stream may still be writing!
```

### Evidence Matrix

| Config | NaN | Without fix |
|--------|-----|------------|
| `torch.compile=True` + `overlap_comm=True` | YES | NaN from step 1 |
| `torch.compile=False` + `overlap_comm=True` | NO | Works fine |
| `overlap_comm=False` + `torch.compile=True` | NO | Works fine |
| device synchronize before reduction | NO | Confirms stream race |
| stream wait from IPG bucket copy streams | NO | Confirms missing synchronization |

### RTX 4090 Implication

Single GPU (dp=1): `overlap_comm` has **no effect** (no all-reduce needed). The bug can't trigger. But the Eager vs torch.compile mode affects NaN behavior in other ways:

```python
# RTX 4090 safe config:
overlap_comm = False  # No-op on dp=1, but harmless to disable
```

## gradient_clipping (#8068)

**Status**: MERGED June 23 (default 0→1.0).

The DeepSpeed ZeRO gradient clipping default was 0 (no clipping). PR #8068 changed it to 1.0. This is critical for GRPO because:

- GRPO advantage can have extreme values (especially with small group sizes)
- Unclipped gradients → NaN within 1-2 steps
- Clip at 1.0 protects against outlier gradients without reducing signal

```python
# ALWAYS set explicitly:
trainer = DeepSpeedZeroOptimizer(
    clip_grad=1.0,  # NOT the default anymore, but be explicit
    ...
)
```

## Complete RTX 4090 Config Generator

### Memory Budget (24 GiB)

```
Available:          24,576 MiB
OS/reserved:        ~1,024 MiB (CUDA contexts, etc.)
Training budget:    23,552 MiB

Breakdown for 7B + LoRA:
  Model (fp16):         14,336 MiB  (7B × 2 bytes)
  LoRA weights (fp16):     70 MiB  (35M × 2)
  LoRA grads (fp16):       70 MiB  (35M × 2)
  Activations (512 ctx): 4,096 MiB  (depends on batch_size)
  Residual:              4,980 MiB  (buffer, peak allocs)

Total:                ~18,572 MiB  ✓ fits
```

### Model Size Calculator

```python
def estimate_peak_memory(
    model_params_b: float,     # e.g. 7 for 7B
    lora_r: int = 32,         # LoRA rank (0 = full fine-tune)
    lora_target_modules: float = 0.05,  # fraction of params targeted
    seq_len: int = 512,
    batch_size: int = 4,
    use_cpu_adam: bool = True,
    use_zenflow: bool = True,
    zero_stage: int = 2,
):
    """Estimate peak GPU memory for DeepSpeed ZeRO training on RTX 4090."""
    n = model_params_b * 1e9

    # Frozen model (fp16)
    model_fp16 = n * 2  # bytes

    # Trainable params (LoRA)
    trainable = n * lora_r * lora_target_modules if lora_r > 0 else n
    lora_fp16 = trainable * 2
    lora_grad = trainable * 2

    # Optimizer states
    if use_cpu_adam:
        # States on CPU
        opt_gpu = 0
        opt_cpu = trainable * 12  # fp32 master + momentum + variance
    else:
        # States on GPU (not viable for models >1B)
        opt_gpu = trainable * 12

    # Activations
    act_mem = estimate_activation_memory(n, seq_len, batch_size)

    # ZenFlow peak (optimizer step transient)
    if use_zenflow and zero_stage in (1, 2) and not use_cpu_adam:
        zenflow_peak = 256 * 1024 * 1024  # 256 MiB chunk
    elif not use_cpu_adam:
        zenflow_peak = lora_grad  # full grad on GPU
    else:
        zenflow_peak = 0  # CPU_Adam: grads transferred immediately

    total = model_fp16 + lora_fp16 + lora_grad + opt_gpu + act_mem + zenflow_peak
    return {
        "total_gib": total / (1024**3),
        "fits_24gb": total < 24 * 1024**3,
        "breakdown": {
            "model_fp16": model_fp16 / (1024**3),
            "lora": (lora_fp16 + lora_grad) / (1024**3),
            "optimizer_gpu": opt_gpu / (1024**3),
            "activations": act_mem / (1024**3),
            "zenflow_peak": zenflow_peak / (1024**3) if zenflow_peak else 0,
        },
    }


def estimate_activation_memory(n, seq_len, batch_size):
    """Rough activation memory estimate for transformer."""
    # Each transformer layer: 2 × seq_len × hidden × batch_size × 2(fp16)
    # Assume 30 layers, hidden ≈ 4096 for 7B
    n_layers = 30
    hidden = int((n / n_layers / 4) ** 0.5) * 4  # rough hidden dim estimate
    bytes_per_activation = 2  # fp16
    # activations = batch × seq × hidden × layers × bytes × checkpointing_factor
    act_checkpoint_factor = 1 / n_layers if n_layers > 0 else 1  # with grad ckpt
    return (
        batch_size
        * seq_len
        * hidden
        * n_layers
        * bytes_per_activation
        * n_layers  # total across layers
        * act_checkpoint_factor  # only need input activations with grad ckpt
    )
```

### RTX 4090 Config Recommendations

| Model Size | LoRA | ZeRO | ZenFlow | CPU_Adam | Batch | Fits? |
|-----------|------|------|---------|----------|-------|-------|
| 1-3B | r=32 | 2 | Yes | Yes | 8 | ✓ |
| 7B | r=32 | 2 | Yes | Yes | 4 | ✓ |
| 7B | Full | 2 | Yes | Yes | 1 | ✗ (OOM) |
| 13B | r=32 | 2 | Yes | Yes | 1 | ✓ (tight) |
| 13B | Full | - | - | - | - | ✗ (impossible) |
| 30B | r=16 | 2 | Yes | Yes | 1 | ✓ (Qwen3-30B-A3B MoE) |
| 70B | any | - | - | - | - | ✗ (impossible) |

### DeepSpeed Config Template

```json
{
  "train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "cpu_offload": true,
    "cpu_offload_params": false,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000,
    "round_robin_gradients": true,
    "zero_hpz_partition_size": 1
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "bf16": {
    "enabled": true
  },
  "communication_data_type": "bf16",
  "wall_clock_breakdown": false
}
```

Key settings for RTX 4090:
- `stage: 2` — ZeRO-3 is pure overhead on dp=1
- `overlap_comm: false` — avoids #8061 NaN, no benefit on dp=1
- `cpu_offload: true` — moves optimizer to CPU (CPU_Adam)
- `contiguous_gradients: true` — enables ZenFlow compatibility
- `gradient_clipping: 1.0` — #8068 fix, protects against GRPO outliers

## ZenFlow + Muon: Why Muon is Blocked

**The Muon optimizer has 6 blockers on DeepSpeed**, including:

1. **#7939** CPU offload BLOCKED (CLOSED without merge)
2. **#5179** PyPI stub (placeholder, can't even install)
3. **#5394** ChainedOptimizer clipping interaction
4. **#5395** skip_grad_norm_clip fix (still OPEN)
5. **Muon grads inflate shared grad_norm** — optimizer-agnostic
6. **No AdamW equivalent fallback path**

Since ZenFlow only helps with optimizer state transfer (CPU_Adam path), and Muon has no CPU offload path, **Muon + ZenFlow synergy is zero**. CPU_Adam remains the ONLY viable optimizer for RTX 4090 GRPO.

## Timeline: ZenFlow Readiness for RTX 4090

```
Pre-ZenFlow (before July 2026):
  Full fine-tune on RTX 4090: IMPOSSIBLE (OOM at optimizer step)
  LoRA fine-tune: WORKS but 14 GiB spike wastes GPU cycles

ZenFlow merged (July 7, 2026):
  Full fine-tune ZeRO-2: WORKS (256 MiB spike only)
  Full fine-tune ZeRO-3: STILL doesn't work (no chunked copyback)
  LoRA fine-tune ZeRO-2: IMPROVED (less peak memory)

Future (requires multi-GPU):
  ZeRO-3 + ZenFlow chunked copyback: NOT implemented, may never be
  (chunking fp32 materialization requires fundamental redesign)
```

## Summary: 10 MUST DO / 5 MUST NOT Rules

### MUST DO
1. MUST use ZeRO-2 (NOT ZeRO-3) on single GPU RTX 4090
2. MUST use CPU_Adam for optimizer offload to CPU RAM
3. MUST set `overlap_comm: false` on dp=1 (avoids #8061)
4. MUST set `gradient_clipping: 1.0` (GRPO safety, #8068)
5. MUST use LoRA for models >3B (keeps trainable params << total)
6. MUST enable gradient checkpointing (reduces act memory 5-10×)
7. MUST use `bf16` mixed precision (fp32 impossible on 24 GiB)
8. MUST set `contiguous_gradients: true` (ZenFlow prerequisite)
9. MUST pin GPU memory for CPU_Adam transfer (faster CPU↔GPU)
10. MUST monitor peak memory with ZenFlow (256 MiB expected)

### MUST NOT
1. MUST NOT use ZeRO-3 on dp=1 (pure overhead, no benefit)
2. MUST NOT use full fine-tune for models >7B (even with ZenFlow)
3. MUST NOT use Muon on RTX 4090 (no CPU offload path)
4. MUST NOT use `overlap_comm: true` with `torch.compile` (#8061 NaN)
5. MUST NOT expect ZenFlow to help ZeRO-3 (Stage 1/2 only)

## References

- DeepSpeed #8058: ZenFlow chunked copyback (MERGED July 7)
- DeepSpeed #8061: overlap_comm NaN (OPEN, maintainers engaged)
- DeepSpeed #8068: gradient_clipping default 0→1.0 (MERGED June 23)
- DeepSpeed #8075: fd leak (MERGED June 23)
- DeepSpeed #8104: Q3 Roadmap (AutoEP, DeepCompile, OPD Trainer)
- ZenFlow limitation: 5 delock comments, missing unit tests, contiguous() bug
