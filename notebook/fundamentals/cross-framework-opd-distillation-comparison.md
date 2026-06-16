# Cross-Framework OPD (On-Policy Distillation) Comparison — RTX 4090 Focus

> 2026-06-16 | DeepSpeed #8027 vs verl #6072/#5041 vs rLLM #356 | Consumer GPU viability
> Key: which framework makes distillation viable on a single 24GB GPU?

---

## 1. OPD Concept: Why It Matters for Consumer GPUs

On-Policy Distillation (OPD) trains a small **student** model to imitate a large frozen **teacher** on the student's **own** generated rollouts. This differs from:

- **Traditional KD**: teacher generates data offline, student trains on it (distribution mismatch)
- **GRPO/RL**: reward-guided optimization (needs reward model, can exceed teacher)
- **SFT distillation**: teacher outputs as training data (DeepSeek-R1 style, simpler but static)

OPD's advantage: the student learns from states it actually visits, so the learning signal adapts as the student improves. The teacher only needs to score (forward pass) -- no backward, no GPU residency required.

**Consumer GPU breakthrough**: if the teacher lives on CPU RAM and only visits GPU briefly during forward, the GPU only needs to hold the student. A 0.5B student (~1GB) + teacher chunk (~1GB transient) = ~3-7GB total GPU. This is the first training paradigm where a 7B teacher can supervise a 0.5B student on a single RTX 4090.

### Associated Research Papers

1. **TIP** (arxiv 2604.14084) -- Token Importance in OPD: which positions carry the most learning signal
2. **PACED** (arxiv 2603.11178) -- Forward-then-reverse KL two-stage schedule, student competence weighting
3. **Beyond GRPO** (arxiv 2605.12483) -- RL-trained teacher explores, then distills with dense token-level supervision

---

## 2. Architecture Comparison

### 2.1 DeepSpeed OPSD (#8027, DRAFT OPEN)

```
Phase 0: Student Rollout
  prompts -> RolloutEngine.generate() -> RolloutBatch

Phase 1: Teacher Forward -> CPU Logit Cache
  input_ids -> TeacherWrapper.forward_to_cache()
  -> TeacherLogitCache (CPU-pinned, bf16, detach)
  -> ZeRO-3 gathers teacher params to GPU -> forward -> logits -> CPU cache -> params released

Phase 2: Student Forward + Streamed Divergence + Backward
  student_logits [B,T,V] on GPU
  -> teacher_chunk_fetcher(start,end) fetches [B,chunk,V] from CPU one chunk at a time
  -> streamed_distillation_loss (forward-KL / reverse-KL / JSD)
  -> backward + optimizer step
  -> teacher_cache.free() drops CPU buffer
```

Key innovation: **TeacherLogitCache** stores full `[B,T,V]` on CPU in bf16 with pinned memory. During student backward, only one `[B,chunk,V]` chunk transfers to GPU at a time. The full teacher tensor never co-resides with student logits on GPU.

Teacher placement: ZeRO-3 + `offload_param: {device: cpu}` -- teacher weights live on CPU RAM, gathered to GPU per-forward, released after. No permanent GPU occupancy.

Loss types: forward-KL (mode-covering), reverse-KL (mode-seeking), JSD (balanced). All with temperature scaling (Hinton T^2 convention).

Rollout backends: HybridEngine (in-process, no weight sync needed) or vLLM (disjoint GPUs, has deadlock issue on single-GPU launch).

Status: DRAFT, not merged. 3246 LOC, 87 CPU-only tests. Validated on 2xH200.

### 2.2 verl OPD (#5041 merged, #6072 VeOmni merged)

```
Student Model (actor worker, FSDP2/Megatron/VeOmni)
    -> rollout via vLLM/SGLang server
    -> student trajectories

Teacher LLM Server (AsyncTeacher, independent vLLM server)
    -> computes teacher_log_probs on separate GPU(s)
    -> sends log_probs to student worker

-> distillation_loss computation
-> student update (FSDP2 sharded backward)
```

Key innovation: **AsyncTeacher** runs as an independent vLLM/SGLang server. Teacher never shares training GPU. Teacher log_probs arrive asynchronously -- training GPU only receives compressed log_probs, not full `[B,T,V]`.

Loss types: Forward top-K KL (sparse, only top-k logits from teacher), k1/k3 reverse KL estimator. Both use **fused top-K distillation kernel** (#6511) -- only computes divergence over top-k teacher logits, skipping the vast zero-probability vocab tail.

Multi-teacher: supported (#6051 merged) -- multiple teachers provide different supervision dimensions.

Training engine: FSDP2 (default) or Megatron or VeOmni (Ascend NPU). Student sharded across GPUs.

Status: MERGED, production-tested. Part of verl v0.8.0.

### 2.3 rLLM OPD (#356, early implementation)

```
rllm/trainer/distill/  <- On-policy distillation module
    -> Uses TinkerBackend (in-process, single GPU)
    -> Teacher model forward on CPU or same GPU
    -> Distillation loss applied to student
```

rLLM's distill module exists in the source tree (`rllm/trainer/distill/`) but is an early implementation. It leverages the TinkerBackend's in-process architecture -- student and teacher share the same process, no Ray dependency.

Key characteristics (based on architecture reading):
- In-process: no Ray, no separate servers
- BackendProtocol: same 6-method ABC as verl/tinker/fireworks backends
- Loss functions: inherits from rLLM's transform pipeline (GRPO/REINFORCE/RLOO) -- distillation is a new transform type
- Gateway-integrated: Model Gateway captures token_ids + logprobs from both student and teacher

Status: Early, #356 PR. Not as thoroughly documented as DeepSpeed or verl OPD. The module exists but lacks the CPU-offload optimization and chunked streaming that DeepSpeed has.

---

## 3. Memory Budget on RTX 4090 (24GB)

### 3.1 DeepSpeed OPSD (Best for single GPU)

| Component | GPU Size | CPU Size |
|-----------|----------|----------|
| Student model (0.5B, bf16) | ~1.23 GB | -- |
| Student gradients (bf16, transient) | ~1.23 GB | -- |
| Optimizer (ZeRO-0: Adam fp32) | ~9.84 GB | -- |
| Optimizer (ZeRO-2: CPU_Adam) | -- | ~9.84 GB |
| Student logits [B,T,V] | ~0.3-2.5 GB | -- |
| One teacher chunk [B,chunk,V] | ~0.15-1.2 GB | -- |
| Teacher model (7B, bf16, CPU-offloaded) | transient | ~14 GB |
| Teacher logits cache [B,T,V] | -- | ~2.5 GB |
| Activations | ~0.5-2 GB | -- |

**ZeRO-2 + CPU_Adam config** (recommended):
- GPU total: ~3-7 GB -- very comfortable on 24GB
- CPU total: ~26 GB (optimizer + teacher model + logits cache) -- needs 32+ GB system RAM

**ZeRO-0 config** (simpler, for small batches):
- GPU total: ~14-17 GB -- fits but tight
- CPU total: ~3-6 GB (small teacher logits only)

### 3.2 verl OPD (Requires multi-GPU or separate teacher server)

| Component | Training GPU | Teacher GPU(s) |
|-----------|-------------|----------------|
| Student model (0.5B, FSDP2 sharded) | shard only | -- |
| FSDP2 optimizer shards | shard only | -- |
| Student logits [B,T,V] | ~0.3-2.5 GB | -- |
| Teacher log_probs (compressed, top-k) | ~small | -- |
| Teacher model (vLLM server) | -- | ~14 GB full model |

**Single GPU**: FSDP2 degenerates on dp=1 (no sharding benefit). Student model + optimizer + logits must all fit on one GPU. Teacher needs a **separate GPU** for the vLLM server -- RTX 4090 single-GPU setup cannot host both.

**Multi-GPU**: FSDP2 shards student across N GPUs. Teacher vLLM server on additional GPU(s). This is the intended deployment pattern.

### 3.3 rLLM OPD (In-process, limited optimization)

| Component | GPU Size | CPU Size |
|-----------|----------|----------|
| Student model (0.5B, bf16) | ~1.23 GB | -- |
| Teacher model (7B, bf16) | ~14 GB if on GPU | -- |
| Teacher model (CPU) | -- | ~14 GB |
| Student logits + teacher logits | ~3-5 GB both on GPU | -- |
| Optimizer (Adam, in-process) | ~9.84 GB | -- |

**Problem**: rLLM's current distill module does NOT have DeepSpeed's CPU-offload optimization. If teacher is on GPU, total = ~28 GB -- exceeds 24GB. If teacher is on CPU, forward pass requires gathering 7B params to GPU -- peak GPU = student + teacher params + logits = way over 24GB during teacher forward.

**Potential fix**: rLLM could implement a TeacherLogitCache-style CPU caching mechanism (since TinkerBackend is in-process, it has more control than verl's distributed setup). But this is NOT currently implemented.

---

## 4. Training Loop Comparison

| Dimension | DeepSpeed OPSD | verl OPD | rLLM OPD |
|-----------|---------------|----------|----------|
| Phases per step | 3 (rollout, teacher CPU, student backward) | 2 (rollout, distill loss) | 2 (rollout+teacher, student backward) |
| Teacher forward | Every step, CPU-offloaded, ZeRO-3 gather/release | Async vLLM server, separate GPU | In-process, on CPU or GPU |
| Logit handling | CPU-pinned TeacherLogitCache, chunked GPU fetch | Async log_probs from server, top-K sparse | Likely full GPU tensor (no streaming) |
| Student backward | Streamed loss (chunk-by-chunk teacher fetch) | FSDP2 sharded backward | In-process, full backward |
| Weight sync | No-op for hybrid engine; pickle for vLLM | FSDP2 gather + bucketed transfer | In-process, zero-copy (Tinker) |
| KV cache sharing | None (rollout engine separate) | vLLM/SGLang KV cache for rollout | Model Gateway capture |
| Multi-teacher | Not supported | Supported (#6051) | Not supported |

### Loop Timing Estimate (RTX 4090, Qwen2.5-0.5B student)

| Phase | DeepSpeed (ZeRO-2+CPU_Adam) | verl (single GPU infeasible) | rLLM (teacher on CPU, no streaming) |
|-------|-----------------------------|------------------------------|--------------------------------------|
| Rollout | ~1-3s (0.5B hybrid engine) | ~1-3s (vLLM server) | ~1-2s (Tinker in-process) |
| Teacher forward | ~10-30s (7B CPU gather+forward+release) | ~2-5s (async, separate GPU) | ~10-30s (7B CPU gather+forward) |
| Student forward+backward | ~2-5s (streamed chunks) | ~1-3s (FSDP2, top-K sparse) | ~2-3s (full tensor on GPU) |
| Total per step | ~13-38s | N/A (multi-GPU only) | ~13-35s |

---

## 5. Loss Function Comparison

| Loss Type | DeepSpeed OPSD | verl OPD | Practical Implication |
|-----------|---------------|----------|----------------------|
| Forward KL (mode-covering) | Full vocab KL | Top-K sparse KL | verl's top-K skips zero-prob tail -- faster, similar quality |
| Reverse KL (mode-seeking) | Full vocab KL | k1/k3 estimator | DeepSpeed exact; verl approximated -- but estimator works well |
| JSD (balanced) | Full vocab JSD | Not available | DeepSpeed-only option |
| Temperature scaling | T^2 Hinton convention | T^2 convention | Same |
| Chunk/streaming | Sequence-axis chunks (512 tokens) | Not needed (top-K sparse) | DeepSpeed needs chunks for memory; verl avoids with sparsity |
| Numerical stability | fp32 promotion inside divergence | fp32 in fused kernel | Both promote for stability |

**RTX 4090 implication**: DeepSpeed's streaming approach is memory-efficient but slower (loop over chunks). verl's top-K fused kernel is faster but requires teacher logits accessible on GPU (not CPU-streamed). On single GPU, DeepSpeed's streaming wins because it keeps GPU footprint small.

---

## 6. Quantization Options for Student Model

| Quantization | Student Size | GPU Memory | Quality Impact | Viability |
|-------------|-------------|------------|---------------|-----------|
| BF16 (baseline) | 0.5B = ~1GB | ~3-7 GB total | 100% | Recommended starting point |
| INT8 weights (student only) | ~0.5GB weights | ~2-4 GB total | ~99.9% | Good for tighter budgets |
| INT4 AWQ (student only) | ~0.25GB weights | ~1.5-3 GB total | ~95-97% | Aggressive, quality loss |
| LoRA (rank=32) | ~1GB base + ~0.6GB adapter | ~2.6 GB total | Depends on task | Best memory efficiency |

**Important**: Quantizing the student for OPD is different from quantizing for GRPO. In OPD, the student generates rollouts and then does forward+backward against teacher logits. INT8/INT4 quantization affects both generation quality AND divergence computation. LoRA is the safest memory reduction because base weights stay frozen (no optimizer) and adapter params are small.

**DeepSpeed OPD + LoRA** (potential, NOT currently wired):
- Student base weights frozen (ds_optim_param=True)
- Only LoRA params need optimizer: ~0.6GB * 12 bytes = ~7.2GB (CPU_Adam)
- GPU: ~1GB base + ~0.6GB LoRA + ~0.3GB logits + ~0.5GB activations = ~2.6 GB
- This would be incredibly light but requires code changes

**verl OPD + LoRA**: verl already has LoRA support (merge=True for inference). OPD with LoRA would be:
- Student base weights frozen, LoRA adapter trained
- Optimizer for LoRA only: ~7.2GB on GPU (FSDP2 sharded across GPUs)
- But on single GPU, FSDP2 degenerates -- same problem as without LoRA

---

## 7. RTX 4090 OPD Ranking

```
#1: DeepSpeed OPSD + ZeRO-2 + CPU_Adam + CPU-offloaded teacher
    Student-only GPU (~3-7 GB)
    Teacher on CPU (~26 GB total system RAM)
    Qwen2.5-0.5B student + Qwen2.5-Math-7B teacher
    Streaming loss = controlled GPU memory
    UNIQUE: ZeRO-3 CPU offload for teacher is DeepSpeed-only capability
    Status: DRAFT, not merged -- but architecture is sound

#2: rLLM Tinker + distill module (potential, early implementation)
    In-process, zero-copy weight sync
    But: NO CPU-offload optimization yet
    NO streaming teacher logits yet
    Would need TeacherLogitCache-style implementation
    Mid-term: could become #1 if CPU-offload is added (in-process advantage)

#3: verl VeOmni OPD + FSDP2 (NOT viable on single GPU)
    Requires separate GPU for teacher vLLM server
    FSDP2 useless on single GPU (dp=1)
    Top-K fused kernel excellent for multi-GPU
    Multi-teacher support valuable for data-center setups
    Single RTX 4090: cannot host both student training AND teacher server
```

---

## 8. Practical Model Choices

### Best Configuration for RTX 4090 (DeepSpeed OPSD)

```json
{
    "student": "Qwen/Qwen2.5-0.5B-Instruct",
    "teacher": "Qwen/Qwen2.5-Math-7B-Instruct",
    "student_dtype": "bfloat16",
    "teacher_dtype": "bfloat16",
    "offload_to_cpu": true,
    "rollout_engine": "hybrid_engine",
    "loss_type": "reverse_kl",
    "temperature": 1.0,
    "chunk_size": 512,
    "train_batch_size": 8,
    "micro_batch_size_per_gpu": 2,
    "learning_rate": 1e-6,
    "deepspeed_config": "ZeRO-2 + CPU_Adam"
}
```

GPU: ~5-7 GB. CPU: ~26.5 GB. Needs 32+ GB system RAM.

### Scaling Ladder

| Student | Teacher | GPU (ZeRO-2+CPU_Adam) | CPU RAM | Viable? |
|---------|---------|-----------------------|---------|---------|
| Qwen2.5-0.5B | Qwen2.5-1.5B | ~3 GB | ~16 GB | Yes (easy) |
| Qwen2.5-0.5B | Qwen2.5-7B | ~3 GB | ~26 GB | Yes (32+ GB RAM) |
| Qwen2.5-0.5B | Qwen2.5-Math-7B | ~3 GB | ~26 GB | Yes (32+ GB RAM) |
| Qwen2.5-1.5B | Qwen2.5-7B | ~6 GB | ~26 GB | Yes (comfortable) |
| Qwen2.5-1.5B | Qwen2.5-72B | ~6 GB | ~144 GB | No (RAM too large) |
| Qwen2.5-3B | Qwen2.5-7B | ~15 GB | ~26 GB | Tight (GPU) |

### What if you only have 16GB system RAM?

Use a 1.5B teacher instead of 7B:
- CPU: ~3GB teacher + ~6GB optimizer + ~0.5GB logits = ~9.5 GB total
- GPU: ~3 GB (student-only)
- Fits comfortably even on 16GB systems

Or use Qwen2.5-Math-1.5B-Instruct as teacher -- still stronger than 0.5B student, but requires less CPU.

---

## 9. OPD vs GRPO on RTX 4090

| Dimension | OPD (Distillation) | GRPO (RL) |
|-----------|---------------------|-----------|
| Goal | Imitate teacher distribution | Maximize reward signal |
| Training signal | Per-token divergence to teacher | Group-relative advantage + reward |
| Teacher/reward model | Frozen teacher (CPU-offloaded) | Reward model (extra GPU/CPU cost) |
| GPU memory | ~3-7 GB (student + one chunk) | ~18 GB (bypass_mode) or ~35.8 GB (full) |
| Quality ceiling | Teacher's quality | Beyond teacher (if reward allows) |
| Data source | Student's own rollouts | Student's own rollouts |
| Best for | Transferring teacher capability | Exploring beyond teacher |
| CPU RAM needed | ~26 GB (teacher + optimizer) | ~0 GB (bypass_mode) |
| Framework | DeepSpeed OPSD (best single GPU) | rLLM Tinker / verl (best) |

**When to choose OPD**:
- You have 32+ GB system RAM but limited GPU VRAM
- You want to compress a known strong model (e.g., math specialist)
- You don't have a reward function
- Quality ceiling = teacher quality is acceptable

**When to choose GRPO**:
- You have 24GB GPU but limited system RAM
- You have a good reward function
- You want the student to potentially exceed the teacher
- You want to explore new reasoning behaviors

**Combined approach** (OPD first, then GRPO):
- Phase 1: OPD distill from 7B teacher -> 0.5B student learns teacher's distribution
- Phase 2: GRPO with reward function -> student explores beyond teacher
- This is the "Beyond GRPO" paper (arxiv 2605.12483) pipeline
- RTX 4090: OPD first (low GPU, high CPU), then switch to GRPO (high GPU, low CPU)

---

## 10. Key Takeaways

1. **DeepSpeed OPSD is the only framework that makes 7B-teacher OPD viable on a single RTX 4090**. ZeRO-3 CPU offload for the teacher + TeacherLogitCache + streaming loss = student-only GPU footprint. No other framework has this combination.

2. **verl OPD is the best for multi-GPU setups** but cannot run on a single RTX 4090. The AsyncTeacher server requires a separate GPU, and FSDP2 degenerates on dp=1.

3. **rLLM OPD has in-process advantages** (zero-copy, no Ray) but lacks the CPU-offload optimization. Mid-term, it could surpass DeepSpeed for single-GPU OPD if TeacherLogitCache-style caching is added.

4. **The binding constraint is CPU RAM, not GPU VRAM**. 7B teacher + optimizer + logits cache = ~26 GB CPU. Most RTX 4090 workstations have 32-64 GB, so this is feasible.

5. **OPD is a complement to GRPO, not a replacement**. OPD transfers capability; GRPO explores beyond. The best pipeline may be OPD first (learn teacher), then GRPO (improve further).

6. **Practical model choice: Qwen2.5-0.5B student + Qwen2.5-Math-7B teacher**. 0.5B student is light enough (~3-7 GB GPU). Math-7B teacher is strong enough for math/coding distillation. 1.5B teacher is an alternative for 16GB-RAM systems.

7. **LoRA + OPD would be incredibly efficient** (~2.6 GB GPU) but is NOT currently wired in any framework. This is a clear gap and opportunity.

---

## References

- DeepSpeed OPSD: PR #8027 (DRAFT OPEN) -- `notebook/projects/deepspeed-opd-trainer-source-reading.md`
- verl OPD: PR #5041 (merged), PR #6072 VeOmni (merged), PR #6051 multi-teacher (merged), PR #6511 fused top-K kernel (merged)
- rLLM OPD: PR #356 (On Policy Distillation) -- `notebook/projects/rllm-architecture-reading.md`, `rllm/trainer/distill/`
- OPSD papers: TIP (arxiv 2604.14084), PACED (arxiv 2603.11178), Beyond GRPO (arxiv 2605.12483)
- Distillation fundamentals: `notebook/fundamentals/distillation-deep-dive.md`, `notebook/fundamentals/knowledge-distillation-theory.md`
- verl VeOmni reading: `notebook/projects/verl-v08-veomni-flowgrpo-reading.md`
- verl v0.8 features: `notebook/projects/verl-v0.8-features-reading.md`
