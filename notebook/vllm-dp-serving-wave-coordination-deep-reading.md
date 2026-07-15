# vLLM Data Parallel Serving Architecture: DPEngineCoreProc + Wave Coordination

**Date**: 2026-07-15 (Session 10)
**Source**: vllm/v1/engine/core.py (DPEngineCoreProc, lines 1812-2050) + coordinator.py
**Significance**: ★★★★★★★ MoE-only DP serving — Wave coordination for GRPO multi-GPU rollout

---

## 1. Architecture

DPEngineCoreProc extends EngineCoreProc for data-parallel serving. **Assertion**: `is_moe` — only MoE models use DP serving in vLLM.

### Key Components:
```
DPCoordinator (ZMQ)
  ├── DPCoordinatorProc — coordinator process
  ├── DPEngineCoreProc × dp_size — engine processes
  └── Wave coordination via ZMQ messages

DPEngineCoreProc extends EngineCoreProc:
  ├── step_counter — global step count (synchronized every 32 steps)
  ├── current_wave — current wave number
  ├── engines_running — whether engines are actively stepping
  ├── pending_pause — two-phase pause protocol flag
  ├── ignore_start_dp_wave — prevents stale wave messages
  ├── dp_group — stateless torch.distributed process group
  └── prefill_schedule_interval — throttles prefill admission
```

---

## 2. Wave Coordination

### Wave = group of requests processed across all DP ranks simultaneously

**Lifecycle**:
1. New request arrives → assigned to current_wave
2. Coordinator broadcasts START_DP_WAVE to all engines
3. All engines step through the wave together
4. When ALL engines finish (AllReduce confirms) → wave_complete notification
5. Engines pause → wait for next START_DP_WAVE
6. New wave begins: current_wave += 1, step_counter = 0

### Wave transitions:
```
Idle (engines_running=False)
  → START_DP_WAVE arrives
  → engines_running = True
  → step loop begins

Stepping (engines_running=True)
  → _process_engine_step() each iteration
  → _has_global_unfinished_reqs() every 32 steps
  → AllReduce: all ranks confirm no unfinished requests
  → engines_running = False
  → wave_complete notification (dp_rank 0 → coordinator)
  → current_wave += 1
```

---

## 3. AllReduce Every 32 Steps

```python
def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
    self.step_counter += 1
    if self.step_counter % 32 != 0:
        return True  # Skip sync, assume running

    has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
        self.dp_group,
        has_unfinished=local_unfinished,
        pending_pause=self.pending_pause,
    )

    if pause_consensus:
        self.ignore_start_dp_wave = True
        self.pending_pause = False

    return has_unfinished
```

**Why every 32 steps**: AllReduce is expensive (NCCL latency ~1-5ms). Checking every step would waste ~0.3-5% throughput. Every 32 steps = ~1% overhead, acceptable for MoE serving.

**pause_consensus**: When all DP ranks have `pending_pause=True`, they collectively agree to stop. This prevents partial pause where some ranks stop while others continue → deadlock.

---

## 4. Two-Phase Pause Protocol

### Phase 1 (local):
```python
def _pause_complete(self) -> bool:
    self.pending_pause = True
    self.engines_running = True  # Kick-start stepping loop
    return False  # Don't pause yet
```

- Set `pending_pause=True` but keep `engines_running=True`
- Engine continues stepping (dummy batches if no real work)
- Reaches the AllReduce checkpoint in `_has_global_unfinished_reqs`

### Phase 2 (collective):
```python
# Inside _has_global_unfinished_reqs:
if pause_consensus:  # All ranks agree to pause
    self.ignore_start_dp_wave = True
    self.pending_pause = False
```

- All-reduce confirms ALL ranks want to pause
- Set `ignore_start_dp_wave=True` → stale START_DP_WAVE messages can't re-wake
- Engine safely stops stepping

**Why two phases**: Without consensus, a stale START_DP_WAVE message (from a request arriving during pause transition) could re-wake one engine while others are paused → deadlock.

---

## 5. Prefill Throttling

```python
def _should_throttle_prefills(self) -> bool:
    return (
        self.prefill_schedule_interval > 1
        and self.step_counter % self.prefill_schedule_interval != 0
    )
```

- Prefills only admitted when `step_counter % interval == 0`
- Ensures all DP ranks start prefills at the same step → balanced load
- On fresh wave: step_counter=0 → prefills admitted immediately

**GRPO implication**: Prefill throttling adds latency for GRPO rollout generation. For dp=1 (RTX 4090), DP serving is not used → no throttling.

---

## 6. Dummy Batches

When no ready requests exist but engines are still running:
```python
if not executed and not local_unfinished_reqs:
    if self.engines_running:
        self.execute_dummy_batch()  # Keep step_counter advancing
```

**Purpose**: Maintain step_counter synchronization across DP ranks. If one rank has no work, it must still advance the counter so AllReduce checkpoints align.

**Cost**: ~0.1ms per dummy batch (CUDA graph replay with no real work). Negligible.

---

## 7. GRPO Relevance

### dp=1 (RTX 4090): NOT USED
- Single GPU → no DP serving → no wave coordination
- EngineCoreProc (not DPEngineCoreProc) used
- No AllReduce, no prefill throttling, no dummy batches

### dp≥2 (multi-GPU MoE): CRITICAL for GRPO rollout
- MoE models (Qwen3-MoE, DSv3-Lite) require DP for rollout
- Wave coordination ensures all ranks generate together → consistent batch
- Prefill throttling may add latency → configure prefill_schedule_interval=1 for GRPO
- Every 32 steps AllReduce = ~1% overhead for MoE serving

### verl integration:
- verl HYBRID uses vLLM/sGLang for rollout → DP serving applicable
- Weight sync during wave pause = clean opportunity (engines idle)
- Multi-LoRA + DP serving = each DP rank has its own LoRA adapter

---

## 8. Comparison with SGLang DP

| Aspect | vLLM DP Serving | SGLang DP Serving |
|--------|-----------------|-------------------|
| Scope | MoE models only (assert is_moe) | All models |
| Coordination | Wave + AllReduce every 32 steps | Not explicitly documented |
| Pause protocol | Two-phase + pause_consensus | Not documented |
| Prefill throttling | Step-counter aligned | Not documented |
| Dummy batches | Yes (keep step_counter sync) | Not documented |
| Process group | Stateless init (no global state) | Uses existing NCCL |

---

## Session Stats
- **Source code**: vllm/v1/engine/core.py (DPEngineCoreProc, 1812-2050) + coordinator.py (DPCoordinator, 23-460)
- **Key mechanism**: Wave coordination + AllReduce every 32 steps + two-phase pause
- **MoE-only assertion**: DPEngineCoreProc requires is_moe=True
- **RTX 4090**: Not used (dp=1 → EngineCoreProc only)
