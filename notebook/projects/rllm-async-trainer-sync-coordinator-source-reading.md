# rLLM Async Trainer SyncCoordinator Architecture Source Reading

> 2026-06-18 | rLLM Tinker async architecture | SyncCoordinator 172 lines + TrajectoryGroupBuffer 421 lines
> ★★★★★★★★ Tinker client-server SDK pattern (NOT in-process) → different from verl HYBRID

---

## Core Architecture: SyncCoordinator

```
★★★★★★★★★ SyncCoordinator — 172 lines pure asyncio, zero external deps:

  Design: pure Python asyncio → no multiprocessing → no external deps
  → Single process event loop → lightweight → RTX 4090 friendly
  → 172 lines total → minimal → understandable → debuggable

★★★★★★★★★ SyncCoordinator responsibilities:
  1. Weight sync coordination → signal training server when rollout weights ready
  2. Trajectory collection → aggregate trajectories from serving backend
  3. Advantage computation → trigger advantage calculation
  4. Update trigger → signal training step start

★★★★★★★★★ Communication pattern:
  → Client-server SDK → NOT HTTP/IPC → NOT in-process
  → SDK calls → direct Python API → Tinker ↔ serving backend
  → Different from verl HYBRID (in-process generator → zero IPC)
  → Different from DeepSpeed (single-process ZeRO → no external serving)
```

---

## TrajectoryGroupBuffer — 421 Lines

```
★★★★★★★★★ TrajectoryGroupBuffer — trajectory management with NVMe offload:

  Core: buffer trajectories for GRPO group sampling
  → GRPO n=8 → TrajectoryGroup stores all 8 responses together
  → Group-level advantage computation → n responses = 1 group

  NVMe offload:
  → Trajectories too large for GPU memory → offload to NVMe
  → Load back during training → GPU memory freed during rollout
  → Similar to DeepSpeed ZeRO-3 offload philosophy → but for trajectories, not weights

  ★★★★★★★★ Advantage pre-computation:
    → Compute advantage BEFORE training step → store with trajectory
    → Avoids recomputing → saves GPU time → efficient
    → GRPO: group advantage = mean + std normalization → computed once per group
```

---

## PR #576 MergedSegment + TokenOps Protocol

```
★★★★★★★★★ rLLM PR #576 — MergedSegment + TokenOps Protocol:
  Status: OPEN (NOT merged)
  Scope: trajectory merge backend-agnostic → step merge across different serving backends

  MergedSegment:
    → Merge trajectories from different backends → vLLM/SGLang/any
    → Backend-agnostic → same interface regardless of serving engine
    → Token-level merge → handles padding + truncation

  TokenOps Protocol:
    → Standardized token-level operations → encode/decode/pad/truncate
    → Protocol definition → abstract interface → backend implements
    → Enables swapping vLLM → SGLang without changing training code

★★★★★★★★★ Implication for RTX 4090:
  → Tinker can swap vLLM → SGLang → inherit SGLang deterministic inference
  → MergedSegment handles trajectory merge → training code unchanged
  → ★★★★★★★★ Backend-agnostic = swap serving engine WITHOUT changing training pipeline!
```

---

## rLLM vs verl Architecture Comparison

```
★★★★★★★★★ rLLM Tinker vs verl HYBRID — key architectural differences:

| Feature | rLLM Tinker | verl HYBRID |
|---------|-------------|-------------|
| Communication | Client-server SDK | In-process generator |
| Weight sync | Checkpoint-based | Zero-copy Python generator |
| Serving backend | External vLLM/SGLang | In-process vLLM engine |
| Memory | NVMe trajectory offload | Sleep/wake memory management |
| Staleness | None (single-step) | None (on-policy) |
| Complexity | Lower (client-server) | Higher (in-process) |
| Determinism | Inherits backend | Inherits vLLM |

★★★★★★★★★ Tinker advantage:
  → Simpler architecture → client-server → less coupling
  → Can use SGLang → best deterministic inference → KERNEL-level
  → Single-step training → no IS correction → simplest story
  → Backend-agnostic → swap serving engine freely

★★★★★★★★★ HYBRID advantage:
  → Zero IPC → in-process → fastest weight sync
  → Sleep/wake → fine-grained memory management
  → On-policy → same GPU → simplest determinism story
  → BUT: tied to vLLM → inherits vLLM determinism issues
```

---

## RTX 4090 Implications

```
★★★★★★★★★ rLLM Tinker RTX 4090 strategy:

  Recommended config: Tinker + SGLang
    → SGLang deterministic inference → KERNEL-level → gold standard
    → Tinker client-server → simpler → less coupling → easier debugging
    → Single-step training → no multi-iteration staleness
    → bypass_mode compatible → 18Ψ → 3.8Ψ same as verl

  ★★★★★★★★ Tinker checkpoint gap:
    → Tinker checkpoint NOT standard PEFT → custom format
    → Cannot directly load into vLLM/SGLang → need export
    → verl #6713: LoRA adapter export → addresses similar gap
    → Tinker needs: checkpoint → HuggingFace PEFT → serving engine

  Memory budget:
    → NVMe offload for trajectories → GPU freed during rollout
    → Only weights + activations on GPU during training
    → ~19GB peak → fits 24GB → similar to verl HYBRID budget
```

---

## Key Findings Summary

★★★★★★★★★ SyncCoordinator = 172 lines pure asyncio → lightweight → zero external deps
★★★★★★★★★ Client-server SDK pattern → NOT in-process → different from verl HYBRID
★★★★★★★★★ TrajectoryGroupBuffer 421 lines → NVMe offload + advantage pre-computation
★★★★★★★★★ PR #576 OPEN → MergedSegment + TokenOps → backend-agnostic trajectory merge
★★★★★★★★★ Tinker + SGLang = simplest deterministic path for RTX 4090 GRPO
★★★★★★★★★ Tinker checkpoint NOT standard PEFT → export gap needs resolution

---

## References

- verl GRPO flow: notebook/fundamentals/verl-rtx4090-grpo-training-flow.md
- Deterministic decision guide: notebook/fundamentals/rtx4090-deterministic-inference-decision-guide.md
- RTX 4090 decision flowchart: notebook/fundamentals/rtx4090-grpo-training-decision-flowchart.md
