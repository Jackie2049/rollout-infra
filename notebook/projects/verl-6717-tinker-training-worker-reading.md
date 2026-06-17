# verl #6717 — Tinker Training Worker Primitives Source Reading

> 2026-06-18 | PR #6717 (MERGED June 15) | 334 additions, 10 deletions | Author: Luosuu (鐘天楽)
> ★★★★★★★★ Enables rLLM Tinker-style split training IN verl → no framework switch needed
> ★★★★★★★★ TinkerTrainingWorker: optimizer_zero_grad/forward_backward/optimizer_step split primitives
> ★★★★★★★★ Accumulated gradients across multiple forward_backward calls → micro-batch control

---

## 1. Architecture Overview

```
★★★★★★★★★ Key design decisions:

Atomic API preserved:
  → TrainingWorker.train_batch() remains the atomic mini-batch training API
  → Existing callers unchanged → train_batch(data) semantics preserved
  → ★★★★★★★★ No breaking change → backward compatible

New split primitives (opt-in subclass):
  → TinkerTrainingWorker: subclass of TrainingWorker
  → Exposes optimizer_zero_grad(), forward_backward(), optimizer_step()
  → ★★★★★★★★ Caller drives training step in smaller phases

★★★★★★★★★ Split primitive usage pattern:

Default (unchanged):
  worker.train_batch(data)

Tinker split (opt-in):
  worker.optimizer_zero_grad()
  worker.forward_backward(data_1)
  worker.forward_backward(data_2)
  worker.optimizer_step()

★★★★★★★★★ Key difference from train_batch():
  → train_batch(): zero_grad → forward_backward → optimizer_step in ONE call
  → Tinker split: caller controls phase boundaries → can accumulate gradients
  → Multiple forward_backward calls between zero_grad and step → accumulated grads
  → ★★★★★★★★ Enables GRPO micro-batch control → accumulate across multiple rollouts
```

---

## 2. Class Architecture

```
★★★★★★★★★ New classes in engine_workers_tinker.py:

TinkerTrainingWorker (subclass of TrainingWorker):
  → optimizer_zero_grad(): clear gradients → prepare for forward_backward
  → forward_backward(data): run forward + backward → accumulate gradients
  → optimizer_step(): apply accumulated gradients → update parameters
  → ★★★★★★★★ Exception safety: clears gradients when train-mode context exits via exception

TinkerActorRolloutRefWorker (subclass of ActorRolloutRefWorker):
  → Routes same split primitives to actor worker
  → Uses actor/ref worker class attributes for class selection
  → ★★★★★★★★ Engine_workers.py stays on regular worker path
  → Tinker-specific APIs live in separate module → clean separation

★★★★★★★★★ Class attribute routing:
  → ActorRolloutRefWorker chooses actor/ref worker classes through class attributes
  → Default path: TrainingWorker + ActorWorker → engine_workers.py
  → Tinker path: TinkerTrainingWorker → engine_workers_tinker.py
  → ★★★★★★★★ No code duplication → subclass inherits → only overrides split methods
```

---

## 3. Gradient Accumulation Behavior

```
★★★★★★★★★ Accumulated gradients across split primitives:

Normal TrainingWorker.train_batch():
  → Each call: zero_grad → forward_backward → step → complete
  → No accumulation → single gradient application per call

Tinker split:
  → optimizer_zero_grad() → clears all gradients
  → forward_backward(data_1) → accumulates gradients for data_1
  → forward_backward(data_2) → accumulates gradients for data_2 (ADDS to existing)
  → optimizer_step() → applies accumulated gradients from BOTH forward_backward calls
  → ★★★★★★★★ Equivalent to gradient_accumulation_steps=2 → but caller-controlled

★★★★★★★★★ Exception safety:
  → If train-mode context exits through exception → gradients CLEARED
  → Prevents stale gradient state → no leaked gradient from failed forward_backward
  → ★★★★★★★★ Safety mechanism: ensures clean state after any error

★★★★★★★★★ GRPO benefit:
  → GRPO needs multiple forward_backward calls for rollout→reward→update pipeline
  → Without split: train_batch() handles entire pipeline → no micro-batch control
  → With split: caller can accumulate gradients from multiple micro-batches
  → ★★★★★★★★ Enables verl to do rLLM Tinker-style split training → no framework switch!
```

---

## 4. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 LoRA GRPO benefit:

Without Tinker primitives:
  → verl uses train_batch() → atomic → no micro-batch control
  → For accumulated gradients → need gradient_accumulation_steps config
  → For fine-grained control → need rLLM Tinker → framework switch

With Tinker primitives:
  → verl can split training → same pattern as rLLM Tinker
  → No framework switch → verl stays in-process → simpler pipeline
  → ★★★★★★★★ verl + Tinker primitives = best RTX 4090 training story
  → Combined with bypass_mode → 18Ψ→3.8Ψ → CPU_Adam → 16.2 GiB → fits!

★★★★★★★★★ Combined config (RTX 4090 #1.5):
  → verl HYBRID + bypass_mode + TinkerTrainingWorker
  → optimizer_zero_grad → forward_backward(micro_batch_1) → forward_backward(micro_batch_2) → step
  → No ref model (bypass) → no framework switch (Tinker in verl) → minimal memory
  → ★★★★★★★★ Best practical RTX 4090 training path available today
```

---

## Key Findings Summary

★★★★★★★★★ #6717: TinkerTrainingWorker split primitives → optimizer_zero_grad/forward_backward/optimizer_step
★★★★★★★★★ Backward compatible → train_batch() preserved → opt-in subclass → no breaking change
★★★★★★★★★ Gradient accumulation across multiple forward_backward calls → caller-controlled
★★★★★★★★★ Exception safety → cleared gradients on error → no stale state
★★★★★★★★★ Enables rLLM Tinker-style split training IN verl → no framework switch needed
★★★★★★★★★ RTX 4090: best practical training path → verl+bypass+Tinker → 16.2 GiB → fits 24 GiB

---

## References

- verl #6717: https://github.com/verl-project/verl/pull/6717
- verl #6699: https://github.com/verl-project/verl/pull/6699 (detach memory fix)
- rLLM Tinker: notebook/projects/rllm-async-trainer-sync-coordinator-source-reading.md
- RTX 4090 GRPO config reference: tools/rtx4090_grpo_config_reference.py
