# DeepSpeed OPD Trainer — Source-Level Analysis

> 2026-06-16 | PR #8027 (DRAFT OPEN) | Author: PKUWZP (Zhip Wang, Peking University)
> Repository: deepspeedai/DeepSpeed | Branch: zhipwang_opd_pr
> Status: DRAFT, not merged | 3246 additions, 32 new files, 87 CPU-only tests
> Validated: Qwen2.5-0.5B-Instruct (student) + Qwen2.5-1.5B-Instruct (teacher) on 2xH200
> Official title: "[Draft] Add On-Policy Distillation (OPSD) Trainer in DeepSpeed"

---

## 1. Overview: What OPSD Is and Why It Matters

★★★★★★★★★ OPSD (On-Policy Distillation) trains a small **student** model to imitate a large frozen **teacher** on the student's **own** generated rollouts. This is fundamentally different from traditional distillation (off-policy, teacher-generated data) and from GRPO/RL (reward-guided). The student generates responses, the teacher scores them with its distribution, and the student is updated by per-token divergence against the teacher.

★★★★★★★★ Three-phase architecture: student rollout → teacher forward + CPU logit cache → student forward + streamed divergence + backward. The full `[B, T, V]` teacher tensor **never co-resides** with student logits on the training device. This is the key memory innovation.

★★★★★★★★★ For RTX 4090 (24GB VRAM): Qwen2.5-0.5B-Instruct student occupies ~1GB in bf16. Teacher logits on CPU. Student-only GPU. This makes distillation viable on consumer GPUs for the first time in DeepSpeed's ecosystem.

### OPSD Reference Papers (from HJSang/OPSD_OnPolicyDistillation)

The DeepSpeed OPSD trainer is a port of the original OPSD reference implementation (HJSang/OPSD_OnPolicyDistillation), which was built on verl. Three associated papers:

1. **TIP** (arxiv 2604.14084) — Token Importance in On-Policy Distillation: which token positions carry the most useful learning signal based on student entropy and teacher-student divergence
2. **PACED** (arxiv 2603.11178) — Distillation and On-Policy Self-Distillation at the Frontier of Student Competence: weighting problems by student empirical pass rate, forward-then-reverse KL two-stage schedule
3. **Beyond GRPO** (arxiv 2605.12483) — Use RL on a strong teacher model to explore high-reward reasoning behaviors, then distill into a smaller student with dense token-level supervision (FKL-OPD two-stage pipeline)

---

## 2. Architecture: Three-Phase Training Loop (Source-Level Trace)

★★★★★★★★★ The core loop lives in `opsd/trainer.py` (197 lines). Each `_train_step` is exactly three sequential phases:

```
Phase 0: Student Rollout
    prompts → RolloutEngine.generate() → RolloutBatch (input_ids + attention_mask + response_start_idx)

Phase 1: Teacher Forward → CPU Logit Cache
    input_ids + attention_mask → TeacherWrapper.forward_to_cache() → TeacherLogitCache (CPU-resident)

Phase 2: Student Forward + Streamed Divergence + Backward
    input_ids + attention_mask → student_engine() → student_logits [B, T, V]
    → shift for next-token prediction: student_logits[:, :-1, :] vs mask[:, 1:]
    → streamed_distillation_loss(student_logits_shifted, teacher_chunk_fetcher, mask_shifted)
    → student_engine.backward(loss)
    → student_engine.step()
    → teacher_cache.free()  # release CPU buffer before next step
```

### Source trace of `_train_step` (trainer.py lines 120-204):

```python
def _train_step(self, batch) -> dict:
    # --- Move prompts to device ---
    prompt_ids = batch["prompt_ids"].to(self.device, non_blocking=True)
    prompt_attn = batch["prompt_attention_mask"].to(self.device, non_blocking=True)

    # --- Optional weight sync for vLLM backend ---
    if self.step % self.cfg.rollout.weight_sync_interval == 0:
        self.rollout.sync_weights_from_student(self.step)

    # --- Phase 0: Rollout ---
    sampling = SamplingConfig(
        max_new_tokens=self.cfg.rollout.max_response_length,
        temperature=self.cfg.rollout.temperature,
        ...
    )
    roll = self.rollout.generate(RolloutRequest(prompt_ids, prompt_attn), sampling)
    input_ids = roll.input_ids.to(self.device, non_blocking=True)
    attention_mask = roll.attention_mask.to(self.device, non_blocking=True)
    response_start_idx = roll.response_start_idx.to(self.device, non_blocking=True)
    response_mask = build_response_mask(response_start_idx, attention_mask)

    # --- Phase 1: Teacher forward → CPU-cached logits ---
    teacher_cache = self.teacher.forward_to_cache(input_ids, attention_mask)

    # --- Phase 2: Student forward + streamed KL + backward ---
    self.student_engine.train()
    outputs = self.student_engine(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    student_logits_shifted = student_logits[:, :-1, :]
    mask_shifted = response_mask[:, 1:].contiguous()

    # Chunk fetcher bridges CPU cache → GPU one chunk at a time
    def _fetch(start: int, end: int) -> torch.Tensor:
        return teacher_cache.chunk_to_device(start, end,
                                              device=student_logits_shifted.device,
                                              dtype=student_logits_shifted.dtype)

    loss = streamed_distillation_loss(
        student_logits=student_logits_shifted,
        teacher_chunk_fetcher=_fetch,
        response_mask=mask_shifted,
        loss_type=self.cfg.distillation.loss_type,
        temperature=self.cfg.distillation.temperature,
        chunk_size=self.cfg.distillation.chunk_size,
    )

    self.student_engine.backward(loss)
    self.student_engine.step()
    teacher_cache.free()  # Drop CPU buffer — critical for memory management
```

★★★★★★ Key design choices in the trainer:
- `_fetch` closure bridges the CPU TeacherLogitCache to GPU one chunk at a time — the full teacher `[B, T, V]` tensor never re-materializes on GPU
- Shift for next-token prediction: logits at position t predict token at t+1, so `[:, :-1, :]` aligns with mask `[:, 1:]`
- `teacher_cache.free()` explicitly drops the CPU buffer after the step — prevents accumulation across steps
- Timing metrics per phase: `rollout_s`, `teacher_s`, `student_s` — enables phase-level profiling
- Loss reduction across ranks: `dist.all_reduce(loss_for_log) / dist.get_world_size()`

---

## 3. TeacherLogitCache: Host-Resident Chunk Fetch (Source-Level)

★★★★★★★★ `TeacherLogitCache` (teacher.py lines 1869-1933) is the critical memory-saving component. It stores teacher logits on CPU (host) memory in low precision (bf16 by default), then fetches them to GPU chunk-by-chunk during the student backward phase.

### Architecture:

```
TeacherForward → logits [B, T, V] on GPU
                    ↓ detach() + downcast to bf16 + .cpu() (pinned memory when possible)
                TeacherLogitCache.cpu_logits [B, T, V] on CPU host
                    ↓ chunk_to_device(start, end, device, dtype)
                [B, chunk_size, V] on GPU (one chunk at a time)
```

★★★★★★★★ Memory savings math: For a 7B teacher with vocab V=152K, bf16 logits for batch B=8, sequence T=1024:
- Full GPU tensor: 8 * 1024 * 152000 * 2 bytes = ~2.5 GB
- But teacher logits co-residing with student logits would double GPU pressure
- CPU cache: 2.5 GB on host RAM (not GPU VRAM!)
- Per-chunk GPU transfer: 8 * 512 * 152000 * 2 bytes = ~1.2 GB per chunk (with chunk_size=512)
- The chunk is consumed, divergence computed, and chunk freed before next chunk arrives

### Source trace of `TeacherLogitCache`:

```python
@dataclass
class TeacherLogitCache:
    cpu_logits: torch.Tensor  # [B, T, V] on CPU

    @classmethod
    def from_gpu_logits(cls, logits: torch.Tensor,
                        store_dtype: torch.dtype = torch.bfloat16) -> "TeacherLogitCache":
        downcast = logits.detach().to(dtype=store_dtype)
        try:
            host = torch.empty(downcast.shape, dtype=store_dtype, pin_memory=True)
            host.copy_(downcast, non_blocking=True)  # Async CPU copy with pinned memory
        except RuntimeError:
            host = downcast.cpu()  # Fallback for CPU-only test environments
        return cls(cpu_logits=host)

    def chunk_to_device(self, start: int, end: int,
                        device: torch.device,
                        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        chunk = self.cpu_logits[:, start:end]
        out = chunk.to(device=device, dtype=dtype if dtype is not None else chunk.dtype,
                        non_blocking=True)
        return out

    def free(self) -> None:
        self.cpu_logits = torch.empty(0)  # Drop buffer for GC
```

★★★★★★★ Design insights:
- **Pinned memory** (`pin_memory=True` + `non_blocking=True`): enables async CPU-to-GPU transfer overlapping with computation. Falls back gracefully on CPU-only test environments.
- **bf16 storage**: halves CPU memory vs fp32. The `streamed_distillation_loss` promotes back to fp32 inside the divergence for numerical stability.
- **`detach()` on teacher logits**: gradient never flows to teacher — frozen teacher, no backward needed.
- **`free()` method**: explicit buffer release — prevents the CPU buffer from being held across steps, which could cause host memory accumulation.

---

## 4. TeacherWrapper: ZeRO-3 CPU Offload for Teacher (Source-Level)

★★★★★★★ `TeacherWrapper` (teacher.py lines 1951-2027) has two modes depending on `cfg.offload_to_cpu`:

**Mode 1: `offload_to_cpu=True`** (the RTX 4090-relevant path):
```python
# Wrap the teacher with DeepSpeed ZeRO-3 + CPU offload
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},  # Teacher weights on CPU!
    },
}
engine, *_ = deepspeed.initialize(model=model, config=ds_config)
```

★★★★★★★★ This is the key RTX 4090 innovation: the teacher model's **parameters** live on CPU RAM (via ZeRO-3 offload), and are gathered to GPU only during the teacher forward pass (Phase 1), then immediately released. The teacher never permanently occupies GPU VRAM.

**Mode 2: `offload_to_cpu=False`** (multi-GPU setups where teacher fits alongside student):
```python
model.to(device)  # Full teacher on GPU
self._callable = model
```

★★★★★ Important note from the source: The teacher loads the full checkpoint on each rank before DeepSpeed partitions. They intentionally do NOT wrap `from_pretrained` in `deepspeed.zero.Init()` because HF's loader partitions `low_cpu_mem_usage` params to zero-width shards before the checkpoint can fill them, causing "size mismatch" errors.

### Forward path:

```python
@torch.no_grad()
def forward_to_cache(self, input_ids, attention_mask,
                     store_dtype=torch.bfloat16) -> TeacherLogitCache:
    outputs = self._callable(input_ids=input_ids, attention_mask=attention_mask)
    return TeacherLogitCache.from_gpu_logits(outputs.logits, store_dtype=store_dtype)
```

The ZeRO-3 engine automatically:
1. Gathers parameters from CPU to GPU (per-forward)
2. Runs the forward pass
3. Releases parameters back to CPU after forward
4. Returns logits → detached + downcasted → CPU-pinned TeacherLogitCache

★★★★★★★★ Memory lifecycle per step on RTX 4090:
```
Phase 0 (Rollout):     Student weights on GPU (~1GB for 0.5B bf16) + KV cache for generation
Phase 1 (Teacher):     ZeRO-3 gathers teacher params to GPU temporarily → forward → logits → CPU cache → params released back to CPU
                        Peak GPU: student + teacher params + teacher logits (briefly, before offload)
                        After Phase 1: student weights + CPU cache of teacher logits
Phase 2 (Student):     Student forward → student logits [B, T, V] on GPU
                        + chunked teacher fetch [B, chunk, V] on GPU (one at a time)
                        Peak GPU: student weights + student logits + one teacher chunk
                        After backward: gradients, optimizer state, next step ready
```

---

## 5. Streamed/Chunked Loss Computation (Source-Level)

★★★★★★★★ `losses.py` (192 lines) implements three divergence functions with sequence-axis chunking. The key function is `streamed_distillation_loss` which takes a `teacher_chunk_fetcher` callable instead of the full teacher tensor.

### Three divergence types:

```python
def _forward_kl(student_logits, teacher_logits, temperature):
    # D_KL(teacher || student) — mode-covering, student learns full teacher distribution
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    t_probs = t_log_probs.exp()
    kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    return kl * (temperature**2)

def _reverse_kl(student_logits, teacher_logits, temperature):
    # D_KL(student || teacher) — mode-seeking, student focuses on teacher's peaks
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_probs = s_log_probs.exp()
    kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    return kl * (temperature**2)

def _jsd(student_logits, teacher_logits, temperature):
    # JSD = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), M = (P+Q)/2
    # Symmetric, balances mode-covering and mode-seeking
    s_probs = s_log_probs.exp()
    t_probs = t_log_probs.exp()
    m_probs = 0.5 * (s_probs + t_probs)
    m_log_probs = m_probs.clamp_min(1e-12).log()  # Guard against log(0)
    kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
    kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
    return 0.5 * (kl_s + kl_t) * (temperature**2)
```

★★★★★★★★★ `streamed_distillation_loss` — the memory-efficient version:

```python
def streamed_distillation_loss(
    student_logits,              # [B, T, V] on GPU
    teacher_chunk_fetcher,       # fn(start, end) -> [B, end-start, V] on GPU
    response_mask,               # [B, T]
    loss_type="reverse_kl",
    temperature=1.0,
    chunk_size=512,
):
    mask_f = response_mask.to(torch.float32)
    total_tokens = mask_f.sum().clamp_min(1.0)
    total_loss = student_logits.new_zeros((), dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_mask = mask_f[:, start:end]
        if chunk_mask.sum().item() == 0:   # Skip empty chunks (padding regions)
            continue
        teacher_chunk = teacher_chunk_fetcher(start, end)  # CPU → GPU transfer
        per_tok = fn(
            student_logits[:, start:end].float(),  # Promote to fp32 for stability
            teacher_chunk.float(),                  # Also fp32
            temperature,
        )
        total_loss = total_loss + (per_tok * chunk_mask).sum()

    return total_loss / total_tokens
```

★★★★★★★★★ Key memory insight: At any point during the loop, GPU holds only:
- `student_logits[:, start:end]` — one chunk slice of student logits (a **view**, not a copy)
- `teacher_chunk` — one `[B, chunk_size, V]` tensor freshly transferred from CPU
- `per_tok` — one `[B, chunk_size]` divergence result

The full `[B, T, V]` teacher tensor stays on CPU. Each chunk is fetched, divergence computed, accumulated into `total_loss`, and the chunk can be freed (Python GC handles this since `teacher_chunk` is not referenced after the loop iteration).

★★★★★ Also available: `chunked_distillation_loss` — the simpler variant where both tensors are already on GPU. This is useful when the teacher is also on GPU (offload_to_cpu=False) or for testing. The streamed version is the production path for RTX 4090.

### Temperature scaling follows standard KD convention:
- Divide logits by T before softmax → softer distributions
- Multiply result by T^2 → gradient magnitudes comparable across temperatures
- This is the Hinton et al. (2015) KD convention, preserved faithfully

---

## 6. RolloutEngine ABC and Two Backends

★★★★★ `opsd/rollout/base.py` (117 lines) defines the abstract interface:

```python
class RolloutEngine(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        """Run student generate, return prompt+response."""

    @abstractmethod
    def sync_weights_from_student(self, step: int) -> None:
        """Push student weights into rollout backend. No-op for hybrid engine."""

    def shutdown(self) -> None:
        """Release backend resources."""
```

### Data classes:

```python
@dataclass
class RolloutRequest:
    prompt_ids: torch.Tensor           # [B, T_p] left-padded
    prompt_attention_mask: torch.Tensor  # [B, T_p]

@dataclass
class RolloutBatch:
    input_ids: torch.Tensor            # [B', T_p + T_r] right-padded
    attention_mask: torch.Tensor       # [B', T_p + T_r]
    response_start_idx: torch.Tensor   # [B'] int — column where response begins
```

★★★★★★★ **Hybrid Engine Rollout** (hybrid_engine.py, 119 lines):
- Uses DeepSpeed's built-in hybrid engine for generation
- Student weights shared in-process (no copy, no sync needed)
- Falls back to `GatheredParameters` + HF `model.generate()` for architectures without a DeepSpeed inference policy (Qwen2/Qwen3 fall in this category)
- The fallback is ~3-5x slower than the accelerated path but correct
- `sync_weights_from_student()` is a no-op — weights are live

★★★★★ **vLLM Rollout** (vllm.py, 314 lines):
- vLLM on disjoint GPU group
- Weight sync via `GatheredParameters` + `LLM.collective_rpc("load_weights")`
- Pickle-based CPU tensor transfer (slow, ~seconds per 7B model)
- **KNOWN BLOCKING ISSUE**: vLLM's worker init calls `new_group()` on the global process group, but under `deepspeed --num_gpus N`, only rank 0 calls into vLLM → deadlock. Fix requires separate-process architecture (TRL/OpenRLHF pattern).
- Currently verified at unit-test level only, NOT end-to-end

---

## 7. WeightBridge: Per-Architecture TP Slicing for vLLM Sync

★★★★★ `opsd/weight_bridge/` provides architecture-specific slicing of HuggingFace weights for vLLM tensor parallelism. Currently supports Qwen2/Qwen2.5 and Qwen3 dense models.

```python
class ParallelKind(str, Enum):
    COLUMN = "column"   # Split output dim (dim 0) — Q/K/V, gate/up MLP
    ROW = "row"         # Split input dim (dim 1) — attention output, MLP down
    VOCAB = "vocab"     # Split vocab dim — embed_tokens, lm_head
    REPLICATED = "replicated"  # Same on every rank — layer norms, RMSNorm
```

★★★★★ Qwen2 mapping (qwen2.py):
- `self_attn.{q,k,v}_proj.weight` → COLUMN
- `self_attn.o_proj.weight` → ROW
- `mlp.{gate,up}_proj.weight` → COLUMN
- `mlp.down_proj.weight` → ROW
- `embed_tokens.weight`, `lm_head.weight` → VOCAB
- Layer norms → REPLICATED

★★★★★ Qwen3 addition (qwen3.py): inherits Qwen2 + adds `q_norm.weight` and `k_norm.weight` → REPLICATED (1-D per-head-dim scalars, same on every TP rank)

★★★ The bridge does NOT pre-fuse QKV or gate-up — vLLM's loader handles fusion internally from the standard HF layout.

---

## 8. Configuration System (OPSDConfig)

★★★★★ Plain dataclasses (no Hydra/pydantic), loaded from JSON:

```python
@dataclass
class OPSDConfig:
    student: StudentConfig       # model_name_or_path, dtype, arch
    teacher: TeacherConfig       # model_name_or_path, dtype, offload_to_cpu
    rollout: RolloutConfig       # engine, max_prompt/response_length, temperature, gpus (vLLM)
    distillation: DistillationConfig  # loss_type, temperature, chunk_size
    training: TrainingConfig    # batch_size, lr, epochs, save_steps
    data: DataConfig             # path, prompt_field, chat_template
    deepspeed_config: str        # Path to separate DeepSpeed JSON config
```

★★★★★ Production config example (opsd_hybrid_engine.json):
- Student: Qwen2.5-0.5B-Instruct, bf16
- Teacher: Qwen2.5-Math-7B-Instruct, bf16, offload_to_cpu=True
- Rollout: hybrid_engine, max_prompt=1024, max_response=1024, temperature=1.0
- Distillation: reverse_kl, temperature=1.0, chunk_size=512
- Training: batch_size=8, lr=1e-6

★★★★★ Smoke test config (smoke_hybrid.json):
- Student: Qwen2.5-0.5B-Instruct
- Teacher: Qwen2.5-1.5B-Instruct, offload_to_cpu=False (small enough for GPU)
- max_steps=5, batch_size=2, chunk_size=128

---

## 9. ZeRO-0 Viability on RTX 4090 (Student-Only GPU)

★★★★★★★★★★★ This is the most important RTX 4090 insight from the entire PR. Let's work through the memory math:

### Qwen2.5-0.5B-Instruct student model:
- Parameters: ~0.5B = 614M
- bf16 model weights: 614M * 2 bytes = ~1.23 GB
- Optimizer state (AdamW): 614M * (2 + 2) * 4 bytes (fp32 master + momentum + variance) = ~9.84 GB
  - But: ZeRO-0 on single GPU → all optimizer state on GPU
  - Alternative: ZeRO-2 + CPU_Adam → optimizer offloaded to CPU → GPU only needs gradients + model

★★★★★★★★★ ZeRO-0 (no sharding) on single GPU for student-only training:

| Component | Size (bf16) | Notes |
|-----------|------------|-------|
| Student model weights | ~1.23 GB | bf16, on GPU |
| Student gradients | ~1.23 GB | bf16, transient during backward |
| AdamW optimizer (fp32) | ~9.84 GB | fp32 master weights + m + v on GPU |
| Student logits [B,T,V] | ~0.3-2.5 GB | Depends on batch/seq/vocab |
| One teacher chunk [B,chunk,V] | ~0.15-1.2 GB | bf16, transient |
| Activation memory | ~0.5-2 GB | Depends on batch/seq |
| **Total (ZeRO-0)** | **~14-17 GB** | Tight for 24GB! |

★★★★★★★★ ZeRO-2 + CPU_Adam (better for RTX 4090):

| Component | Size | Location |
|-----------|------|----------|
| Student model weights | ~1.23 GB | GPU |
| Student gradients (bf16) | ~1.23 GB | GPU, transient |
| Optimizer state | ~9.84 GB | **CPU** (CPU_Adam) |
| Student logits | ~0.3-2.5 GB | GPU, transient |
| One teacher chunk | ~0.15-1.2 GB | GPU, transient |
| Activation memory | ~0.5-2 GB | GPU |
| **Total GPU** | **~3-7 GB** | Easily fits in 24GB! |
| **Total CPU** | **~9.84 GB** + teacher model CPU | ~10-17 GB CPU RAM |

★★★★★★★★★★★ Teacher model CPU memory estimate (Qwen2.5-Math-7B, offload_to_cpu=True):
- Teacher model params: 7B * 2 bytes (bf16) = ~14 GB on CPU
- ZeRO-3 partitions across ranks, but on single GPU (dp=1) → full model on CPU
- Teacher logits cache: 8 * 1024 * 152000 * 2 bytes = ~2.5 GB on CPU
- **Total CPU for teacher**: ~16.5 GB

★★★★★★★ RTX 4090 total system memory requirements:
- GPU VRAM: ~3-7 GB (student-only with ZeRO-2 + CPU_Adam) — very comfortable
- CPU RAM: ~26-34 GB (student optimizer + teacher model + teacher logits) — feasible on systems with 32-64GB RAM
- The CPU RAM is the binding constraint, not GPU VRAM!

★★★★★★★ What if using ZeRO-0 (simplest config)?
- GPU: ~14-17 GB — still fits in 24GB but tight, especially with large batch/sequence
- ZeRO-0 is viable for very small models (0.5B student) with small batch sizes
- ZeRO-2 + CPU_Adam is safer and leaves more GPU room for activations + logits

★★★★★★★★★ Why ZeRO-3 is unnecessary (and harmful) for single GPU OPD:
- ZeRO-3 partitions optimizer state across DP ranks → with dp=1, no partitioning benefit
- ZeRO-3 adds communication overhead (all-gather, reduce-scatter) that is pure latency on single GPU
- The teacher already uses ZeRO-3 + CPU offload, but that's for the frozen teacher, not the student
- For student: ZeRO-0 or ZeRO-2 is optimal; ZeRO-3 is overhead-only on dp=1

---

## 10. Comparison with Traditional Distillation Approaches

### Traditional (Off-Policy) Distillation:

```
┌───────────────────┐
│ Pre-computed       │ ← Teacher generates responses offline → stored as dataset
│ teacher data       │
└───────────────────┘
        ↓
┌───────────────────┐
│ Student forward    │ ← Student learns on teacher's distribution
│ on teacher data    │ ← KL divergence between student and teacher logits
└───────────────────┘
```

★★★★ Problems with traditional distillation for LLM:
1. **Distribution mismatch**: Teacher's state distribution differs from student's — student learns from states it would never visit
2. **Static dataset**: Cannot adapt as student improves — same teacher data forever
3. **GPU memory**: Must hold teacher model on GPU during training OR pre-compute all logits (disk storage)
4. **No exploration**: Student never generates its own responses → cannot learn from its own mistakes

### OPSD (On-Policy Distillation):

```
┌────────────┐   prompts   ┌──────────────────┐   prompt+response   ┌────────────┐
│ Dataloader │ ──────────▶ │ Student rollout  │ ──────────────────▶ │  Teacher   │
└────────────┘             │ (hybrid / vLLM)  │                     │  forward   │
                           └──────────────────┘                     └─────┬──────┘
                                                                          │ logits → CPU cache
                                                                          ▼
                                                              ┌─────────────────────┐
                                                              │ Student forward +   │
                                                              │ streamed KL / JSD + │
                                                              │ backward / step     │
                                                              └─────────────────────┘
```

★★★★★★★★★ OPSD advantages:
1. **On-policy data**: Student generates its own responses → learns from its actual distribution
2. **Dynamic adaptation**: As student improves, its rollouts change → teacher scoring adapts
3. **Memory efficient**: Teacher logits on CPU, chunked fetch → student-only GPU footprint
4. **Three divergence choices**: forward-KL (mode-covering), reverse-KL (mode-seeking), JSD (balanced)
5. **No ground truth needed**: Teacher provides the supervision — no labeled data required
6. **Iterative improvement**: Each step the student generates new data → continuous learning signal

★★★★★★ OPSD disadvantages:
1. **Slower**: Three phases per step vs one phase in traditional distillation
2. **Teacher forward every step**: Must run teacher forward on every batch (but CPU-offloaded, so GPU cost is transient)
3. **CPU RAM requirement**: Teacher model + logits on CPU → needs sufficient system RAM
4. **Still DRAFT**: Not production-ready, known bugs (vLLM deadlock)

### Comparison with verl OPD (VeOmni engine):

★★★★★★★★ Key differences between DeepSpeed OPSD and verl VeOmni OPD:

| Feature | DeepSpeed OPSD (#8027) | verl VeOmni OPD (#6072) |
|---------|------------------------|-------------------------|
| Backend | DeepSpeed ZeRO engine | FSDP2-based |
| Teacher offload | ZeRO-3 CPU offload | Async teacher server (vLLM) |
| Logit storage | CPU-pinned TeacherLogitCache | Teacher log_probs computed by vLLM server |
| Loss chunking | Sequence-axis chunked fetch | Top-K sparse (fused kernel #6511) |
| Rollout backend | Hybrid engine / vLLM (disjoint) | vLLM / SGLang (separate server) |
| Weight sync | GatheredParameters + collective_rpc | FSDP2 gather + bucketed transfer |
| Multi-teacher | Not supported | Yes (#6051 merged) |
| RTX 4090 viability | ★★★★★★★★ Student-only, ZeRO-0/2 | ★★★ FSDP2 multi-GPU required |
| Loss types | forward-KL, reverse-KL, JSD | Forward top-K KL, k1/k3 reverse KL estimator |
| Architecture support | Qwen2, Qwen2.5, Qwen3 dense | Generic (FSDP2 handles any HF model) |
| Code location | examples/opsd/ (standalone) | verl/training/ (integrated) |

★★★★★★★★★ DeepSpeed OPSD is superior for RTX 4090 because:
1. Teacher can be CPU-offloaded (ZeRO-3 + offload_param) → no teacher GPU memory
2. ZeRO-0/2 viable for student → minimal GPU overhead
3. No dependency on FSDP2 (which requires multi-GPU for benefit)
4. Chunked CPU fetch → controllable GPU memory usage

★★★★ verl VeOmni OPD is superior for multi-GPU because:
1. Async teacher server (vLLM/SGLang) → teacher doesn't block training
2. Fused top-K distillation kernel → sparse KL computation, much faster
3. Multi-teacher support → richer supervision
4. More mature (merged, production-tested)

---

## 11. Comparison with GRPO/RL Training on RTX 4090

★★★★★★★★ OPSD vs GRPO for RTX 4090 — fundamentally different paradigms:

| Dimension | OPSD (Distillation) | GRPO (RL) |
|-----------|---------------------|-----------|
| Goal | Imitate teacher distribution | Maximize reward signal |
| Training signal | Per-token divergence to teacher | Group-relative advantage + reward |
| Teacher/reward model | Frozen teacher (CPU-offloaded) | Reward model (extra GPU/CPU cost) |
| Memory footprint | Student + one teacher chunk | Student + ref model + reward model |
| Quality ceiling | Teacher's quality | Beyond teacher (if reward allows) |
| Data source | Student's own rollouts | Student's own rollouts |
| Best for | Transferring teacher capability | Exploring beyond teacher |

★★★★★★★ RTX 4090 memory comparison:
- OPSD: ~3-7 GB GPU (student + one chunk) + ~26 GB CPU (teacher + optimizer)
- GRPO (bypass_mode): ~18 GB GPU (student + rollout) — no ref/reward model
- GRPO (full): ~35.8 GB GPU — impossible on RTX 4090 without bypass

★★★★★★★★ OPSD has a clear memory advantage: the teacher is entirely CPU-offloaded, while GRPO needs reward/ref models on GPU (or bypass_mode to skip them). OPSD is viable at smaller model sizes (0.5B-1.5B student) where the student easily fits in GPU memory.

★★★★★ But OPSD cannot surpass the teacher's quality — it's bounded by teacher capability. GRPO with a good reward function can potentially exceed the teacher. For many practical use cases (math, coding, reasoning), OPSD is sufficient because the teacher is already strong (e.g., Qwen2.5-Math-7B).

---

## 12. RTX 4090 Practical Implications

★★★★★★★★★★★ RTX 4090 OPD Distillation Ranking (2026-06):

```
#1: ★★★★★★★★★★★★★★★ DeepSpeed OPSD + ZeRO-0/2 + CPU-offloaded teacher
    → Student-only GPU (~3-7 GB) → teacher on CPU (~16-26 GB)
    → Qwen2.5-0.5B student → viable!
    → Qwen2.5-1.5B student → viable (more GPU, but still fits)
    → Best memory efficiency for distillation on consumer GPUs

#2: ★★★★★★ verl VeOmni OPD + FSDP2
    → Requires multi-GPU for FSDP2 benefit
    → Single GPU: FSDP2 degenerates → no sharding benefit
    → Teacher needs separate server → extra GPU or CPU inference
    → More mature code but RTX 4090 limited

#3: ★★★★ rLLM Tinker + distillation (potential, not implemented)
    → In-process → efficient
    → But no distillation support yet
    → Future possibility
```

★★★★★★★★★ Recommended RTX 4090 OPSD configurations:

**Config A: Minimal (ZeRO-0, simplest)**
```json
{
    "student": { "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "bfloat16", "arch": "qwen2" },
    "teacher": { "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "dtype": "bfloat16", "offload_to_cpu": true },
    "rollout": { "engine": "hybrid_engine", "max_prompt_length": 512, "max_response_length": 512, "temperature": 1.0 },
    "distillation": { "loss_type": "reverse_kl", "temperature": 1.0, "chunk_size": 256 },
    "training": { "train_batch_size": 4, "micro_batch_size_per_gpu": 1, "learning_rate": 1e-6 }
}
```
- GPU: ~14 GB (ZeRO-0 Adam on GPU)
- CPU: ~3 GB (1.5B teacher) + ~6 GB optimizer + ~0.5 GB logits cache

**Config B: Optimal (ZeRO-2 + CPU_Adam)**
```json
{
    "student": { "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "bfloat16", "arch": "qwen2" },
    "teacher": { "model_name_or_path": "Qwen/Qwen2.5-Math-7B-Instruct", "dtype": "bfloat16", "offload_to_cpu": true },
    "rollout": { "engine": "hybrid_engine", "max_prompt_length": 1024, "max_response_length": 1024, "temperature": 1.0 },
    "distillation": { "loss_type": "reverse_kl", "temperature": 1.0, "chunk_size": 512 },
    "training": { "train_batch_size": 8, "micro_batch_size_per_gpu": 2, "learning_rate": 1e-6 }
}
```
- GPU: ~5-7 GB (student model + logits + activations + one teacher chunk)
- CPU: ~14 GB (7B teacher) + ~10 GB optimizer + ~2.5 GB logits cache = ~26.5 GB total CPU
- **Requires 32+ GB system RAM** — common on RTX 4090 workstations

★★★★★★★★★ Scaling ladder for RTX 4090 OPD:

| Student Model | Teacher Model | GPU (ZeRO-2+CPU_Adam) | CPU | Viable? |
|---------------|---------------|------------------------|-----|---------|
| Qwen2.5-0.5B | Qwen2.5-1.5B | ~3 GB | ~16 GB | Yes (easy) |
| Qwen2.5-0.5B | Qwen2.5-7B | ~3 GB | ~26 GB | Yes (32+ GB RAM) |
| Qwen2.5-0.5B | Qwen2.5-Math-7B | ~3 GB | ~26 GB | Yes (32+ GB RAM) |
| Qwen2.5-1.5B | Qwen2.5-7B | ~6 GB | ~26 GB | Yes (comfortable) |
| Qwen2.5-1.5B | Qwen2.5-72B | ~6 GB | ~144 GB | No (RAM too large) |
| Qwen2.5-3B | Qwen2.5-7B | ~15 GB | ~26 GB | Tight (GPU) |
| Qwen2.5-3B | Qwen2.5-Math-7B | ~15 GB | ~26 GB | Tight (GPU) |

★★★★★★★ OPD with LoRA (potential future enhancement):
- LoRA on student (rank=32, ~0.6 GB trainable) → GPU even smaller
- Student base weights frozen → no optimizer for base params
- Only LoRA params need Adam → ~0.6 GB * 12 bytes = ~7.2 GB optimizer → CPU
- GPU: ~1.23 GB base + ~0.6 GB LoRA + ~0.3 GB logits + ~0.5 GB activations = ~2.6 GB
- This would be incredibly light! But LoRA is NOT yet wired into the OPSD trainer.

---

## 13. Known Limitations and Follow-ups

★★★★★★★★ Known issues documented in PR description and README:

1. **vLLM rollout deadlock**: vLLM's worker init calls `new_group()` on global PG → only rank 0 participates → hangs. Fix requires separate-process architecture (TRL/OpenRLHF pattern). Currently vLLM rollout is unit-test verified only, not end-to-end.

2. **Qwen-family hybrid engine fallback**: DeepSpeed's inference policy list only covers GPT2/GPT-NeoX/OPT/BLOOM/LLAMA/LLAMA2/InternLM. Qwen2/3 use the `GatheredParameters` fallback → ~3-5x slower generation than accelerated path.

3. **vLLM weight sync goes through pickle**: `LLM.collective_rpc("load_weights", args=((name, tensor_on_cpu),))` → several seconds per sync on a 7B model. A faster v2 would broadcast tensors via NCCL (reference: verl's `bucketed_weight_transfer.py`).

4. **Reward-weighted distillation NOT ported**: OPSD's `opd.reward_beta` knob (from the original verl implementation) is not in this PR. Easy to add: scale `per_tok` by reward weight.

5. **GRPO/RL recipes out of scope**: The RolloutEngine/WeightBridge abstractions are reusable, but GRPO would add its own advantage/KL logic.

6. **Qwen3-MoE not covered**: Need a separate `weight_bridge/qwen3_moe.py` for MoE models.

7. **DistributedSampler missing**: Codex review flagged that the dataloader is created without `DistributedSampler` → every rank iterates the full dataset instead of a shard.

8. **LR/warmup not propagated**: Only batch-size fields are copied from OPSDConfig.training into the DeepSpeed config → optimizer/scheduler knobs ignored.

★★★★★ Most critical for RTX 4090: The hybrid engine path is the only validated live path. The vLLM path has the deadlock issue. For RTX 4090 with single GPU, hybrid engine is the natural choice anyway (vLLM needs disjoint GPUs).

---

## 14. How This Opens a NEW Market for DeepSpeed on Consumer GPUs

★★★★★★★★★★★ DeepSpeed has traditionally been a multi-GPU, data-center framework. On single GPU (dp=1), ZeRO degenerates to no benefit. OPSD changes this fundamentally:

**Before OPD**: DeepSpeed on RTX 4090 was useful only for ZeRO-2 + CPU_Adam (optimizer offload), which is a modest benefit. Full model training (8B+) is impossible. MoE training was impossible until AutoEP (#7938).

**After OPD**: DeepSpeed on RTX 4090 gains a completely new use case — **distillation**. A 0.5B student model learning from a 7B (or larger) teacher. The teacher is CPU-offloaded, the student is GPU-only. This is:
- A training paradigm that ONLY works on consumer GPUs with sufficient CPU RAM
- A market segment that no other framework serves well on RTX 4090:
  - verl VeOmni OPD: requires FSDP2 multi-GPU
  - TRL: standard distillation, no CPU-offload optimization
  - Megatron: no distillation trainer at all
  - rLLM: no distillation support yet

★★★★★★★★★★★ This positions DeepSpeed as the #1 framework for consumer GPU distillation:
1. ZeRO-3 CPU offload for teacher → unique capability
2. TeacherLogitCache → memory-efficient CPU-GPU bridge
3. Streamed loss computation → controllable GPU memory usage
4. Hybrid engine for in-process generation → no extra GPU needed

★★★★★★★★★ Distillation is a growing market:
- Model compression for deployment (7B → 0.5B)
- Task-specific specialization (general → math/coding)
- Self-distillation (same model, different checkpoint)
- Multi-teacher distillation (combining capabilities)
- All of these are viable on RTX 4090 with OPSD + CPU offload

★★★★★★★★★★★★★★★★ Strategic insight: OPD + AutoEP + LoRA = three consumer-GPU viable paths for DeepSpeed:
- AutoEP (#7938): MoE training (Qwen3-MoE 4B, EP=1 singleton)
- OPD (#8027): Distillation (Qwen2.5-0.5B from 7B teacher)
- LoRA + CPU_Adam: Dense training (LoRAOptimizedLinear)

Each serves a different use case, all viable on RTX 4090. This triples DeepSpeed's consumer GPU relevance.

---

## 15. Source File References

All files live under `examples/opsd/` in the DeepSpeed repo (PR branch `zhipwang_opd_pr`):

| File | Lines | Role |
|------|-------|------|
| `opsd/trainer.py` | 197 | Three-phase training loop, OPSDTrainer class |
| `opsd/teacher.py` | 191 | TeacherWrapper + TeacherLogitCache |
| `opsd/losses.py` | 192 | Forward-KL, reverse-KL, JSD + chunked/streamed loss |
| `opsd/config.py` | 149 | OPSDConfig + all sub-configs dataclasses |
| `opsd/data.py` | 108 | PromptDataset + LeftPaddedPromptCollator |
| `opsd/utils.py` | 52 | build_response_mask + shift_for_next_token_prediction |
| `opsd/rollout/base.py` | 117 | RolloutEngine ABC + RolloutRequest/RolloutBatch/SamplingConfig |
| `opsd/rollout/hybrid_engine.py` | 119 | HybridEngineRollout (DeepSpeed hybrid engine + fallback) |
| `opsd/rollout/vllm.py` | 314 | VLLMRollout (disjoint GPUs) + stitch_rollout helper |
| `opsd/weight_bridge/base.py` | 109 | WeightBridge ABC + ParallelKind + _even_slice |
| `opsd/weight_bridge/qwen2.py` | 84 | Qwen2/Qwen2.5 TP mapping |
| `opsd/weight_bridge/qwen3.py` | 37 | Qwen3 dense (adds q_norm/k_norm) |
| `opsd/__init__.py` | 17 | Module init + version 0.1.0 |
| `opsd/rollout/__init__.py` | 39 | build_rollout factory |
| `opsd/weight_bridge/__init__.py` | 32 | get_bridge factory |
| `main.py` | 135 | Entry point, deepspeed.initialize |
| `configs/ds_zero3.json` | 43 | ZeRO-3 config for hybrid engine |
| `configs/opsd_hybrid_engine.json` | 49 | Production hybrid-engine OPSD config |
| `configs/opsd_vllm_disjoint.json` | 54 | Production vLLM disjoint OPSD config |
| `configs/smoke_hybrid.json` | 49 | Smoke test (0.5B/1.5B, 5 steps) |
| `configs/smoke_vllm.json` | 55 | Smoke test vLLM variant |
| `configs/smoke_ds_zero3.json` | 35 | ZeRO-3 config tuned for smoke runs |
| `README.md` | 232 | Architecture docs, vLLM status, known limitations |
| `tests/test_losses.py` | 166 | CPU-only numerics tests (divergence math) |
| `tests/test_rollout_interface.py` | 156 | RolloutEngine interface conformance |
| `tests/test_teacher_caching.py` | 101 | TeacherLogitCache unit tests |
| `tests/test_vllm_stitch.py` | 97 | stitch_rollout CPU unit test |
| `tests/test_weight_bridge.py` | 259 | TP slicing round-trip tests (Qwen2/Qwen3) |
| `scripts/train_opsd_hybrid.sh` | 14 | Hybrid engine launch script |
| `scripts/train_opsd_vllm.sh` | 19 | vLLM launch script |
| `data/prompts.jsonl` | 20 | 20 math prompt examples |
| `requirements.txt` | 5 | transformers + datasets + numpy |

**Total: 32 files, 3246 LOC, 87 CPU-only tests**

---

## 16. Key Insight Summary

★★★★★★★★★★★ #1: **Three-phase loop with CPU-offloaded teacher logits** is the architectural innovation. Student generates → teacher scores → student learns. The teacher never permanently occupies GPU. TeacherLogitCache on CPU + chunk_to_device fetch = controlled GPU memory.

★★★★★★★★★★★ #2: **ZeRO-3 CPU offload for teacher** enables the 7B teacher to live entirely in system RAM, with per-forward gather/release. This is DeepSpeed's unique capability — no other RL/distillation framework offers this.

★★★★★★★★★★★ #3: **Streamed loss computation** via `teacher_chunk_fetcher` callable ensures the full `[B, T, V]` teacher tensor never co-resides with student logits on GPU. Peak GPU for teacher data is only `[B, chunk_size, V]`.

★★★★★★★★★★★ #4: **RTX 4090 viability**: Qwen2.5-0.5B student + 7B teacher + ZeRO-2 + CPU_Adam = ~3-7 GB GPU + ~26 GB CPU. Fits comfortably on RTX 4090 workstations with 32+ GB RAM. This is the first DeepSpeed-native distillation path viable on consumer GPUs.

★★★★★★★★★★ #5: **NEW market for DeepSpeed**: Distillation on consumer GPUs — a use case no other framework serves well. Combined with AutoEP (MoE training) and LoRA+CPU_Adam (dense training), DeepSpeed now has three consumer-GPU viable paths, tripling its relevance.

★★★★★★★ #6: **OPSD vs GRPO**: Different paradigms. OPSD bounded by teacher quality, but memory-efficient. GRPO can exceed teacher but needs reward/ref models. OPSD is the practical choice for capability transfer on constrained hardware.

★★★★ #7: **Known gaps**: vLLM rollout deadlock, no LoRA wiring, no reward-weighted distillation, no DistributedSampler, LR/warmup not propagated into DS config. All documented, all have clear follow-up paths.

★★★★★★★★ #8: **The OPSD reference research** (TIP, PACED, Beyond GRPO papers) provides deeper insights: token importance weighting, forward-then-reverse KL two-stage schedule, RL-trained teacher + distillation pipeline. These could enhance the DeepSpeed implementation in follow-up PRs.
