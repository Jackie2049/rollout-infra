# verl #6765 — Per-Step Optimizer Overrides (OptimStepParams) Reading

> 2026-06-18 | PR #6765 OPEN | ~110 additions | TinkerTrainingWorker extension
> ★★★★★★★★ Enables Muon-style LR scheduling + per-group optimizer param control
> ★★★★★★★★ Extends #6717 Tinker primitives → fine-grained optimizer stepping
> ★★★★★★★★ Scoped to Tinker worker → regular TrainingWorker API unchanged

---

## 1. OptimStepParams: Lightweight Runtime Payload

```
★★★★★★★★★ OptimStepParams = dict of optimizer param-group values per step:

Supported keys:
  → lr: learning rate for this step
  → betas: Adam beta1, beta2 (or Muon beta)
  → eps: epsilon for numerical stability
  → weight_decay: weight decay for this step

★★★★★★★★★ Usage pattern (Tinker worker):

# Default path — unchanged:
worker.train_batch(data)

# Tinker path — per-step control:
worker.optimizer_zero_grad()
worker.forward_backward(data_1)
worker.forward_backward(data_2)
worker.optimizer_step(optim_step_params={"lr": 5e-4})

★★★★★★★★★ Why this matters for RTX 4090:
  → Muon needs different LR at different training phases
  → LR warmup: lr=0 → linear warmup → then full lr
  → Muon + AdamW hybrid: different lr for Muon groups vs AdamW groups
  → ★★★★★★★★ OptimStepParams enables this WITHOUT modifying the engine API
  → Regular train_batch() path → unchanged → backward compatible

★★★★★★★★★ Implementation:
  → Apply params directly to optimizer param groups immediately before step
  → Runtime key/type checks against actual optimizer param groups
  → Flatten VeOmni MultiOptimizer child param groups for global params
  → Fail fast for optimizer-specific or group-specific keys
  → ★★★★★★★★ LR scheduler ownership stays with caller → not in engine API
```

---

## 2. Relationship to Tinker Primitives (#6717)

```
★★★★★★★★★ #6765 extends #6717's split training primitives:

#6717 (MERGED June 15) — base primitives:
  → optimizer_zero_grad()  → clears gradients
  → forward_backward(data) → runs forward + backward
  → optimizer_step()       → applies gradients to parameters

#6765 (OPEN) — per-step control:
  → optimizer_step(optim_step_params={...}) → override per-group params for THIS step
  → ★★★★★★★★ Optional → default optimizer_step() → unchanged → uses scheduler lr

★★★★★★★★★ Combined Tinker training flow with OptimStepParams:

# Example: Muon-style warmup schedule
for epoch in range(num_epochs):
    worker.optimizer_zero_grad()
    for micro_batch in data_loader:
        worker.forward_backward(micro_batch)
    # Muon warmup: linear LR increase
    warmup_lr = base_lr * min(1.0, epoch / warmup_steps)
    worker.optimizer_step(optim_step_params={"lr": warmup_lr})

★★★★★★★★★ RTX 4090 use case: Muon + AdamW hybrid scheduling
  → Muon groups: lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
  → AdamW groups: lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
  → ★★★★★★★★ Per-step override → different params per group → per epoch
```

---

## 3. TinkerActorRolloutRefWorker Fix

```
★★★★★★★★★ Bug fix in forward_backward():

  → TinkerActorRolloutRefWorker.forward_backward()
  → Previously: didn't dispatch on actor mesh correctly
  → Fix: dispatch on actor mesh at composite worker layer
  → ★★★★★★★★ Important for multi-GPU → but single GPU (RTX 4090) unaffected
```

---

## Key Findings Summary

★★★★★★★★★ #6765: OptimStepParams → per-step optimizer param override → enables Muon LR scheduling
★★★★★★★★★ Scoped to Tinker worker → regular TrainingWorker unchanged → backward compatible
★★★★★★★★★ lr, betas, eps, weight_decay per step → runtime key/type checks
★★★★★★★★★ Extends #6717 Tinker primitives → fine-grained optimizer stepping
★★★★★★★★★ RTX 4090: enables Muon warmup + per-group scheduling when Muon becomes viable
★★★★★★★★★ Muon still BLOCKED → but OptimStepParams READY → infrastructure exists for future use

---

## References

- verl #6765: https://github.com/verl-project/verl/pull/6765
- verl #6717: https://github.com/verl-project/verl/pull/6717 (Tinker primitives)
- Muon status: notebook/fundamentals/cross-framework-muon-optimizer-status-synthesis.md
