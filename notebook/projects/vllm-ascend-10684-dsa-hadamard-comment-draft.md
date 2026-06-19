# vLLM-Ascend #10684: DSA Hadamard Sleep/Wake — Comment Draft

> 2026-06-19 | Comment draft for posting on vllm-ascend #10684
> ★★★★★★★★ CRITICAL: DSA Hadamard class variable lost during sleep/wake → ALL-ZERO inference
> ★★★★★★★★ Pattern family: State Lifecycle Mismatch — identical to SGLang #28676, #28679, vLLM #44395
> ★★★★★★★★ BLOCKER for verl RLHF on Ascend NPU with DSA models

---

## Comment Body Draft

```markdown
## Cross-Framework Pattern Analysis: DSA Hadamard Class Variable Loss → ALL-ZERO after sleep/wake

This is a textbook **State Lifecycle Mismatch** bug with confirmed root cause.

### Root Cause Confirmation

The hadamard transform matrix is stored as a **class variable** on `AscendDSACPMetadataBuilder.hadamard`, NOT as an instance variable or model buffer. This means:

1. `CaMemAllocator.sleep()` offloads NPU memory tagged as "weights" to CPU
2. `worker.sleep()` saves `model.named_buffers()` (lines 221-223)
3. Hadamard is a **class attribute** → invisible to `named_buffers()` → NOT saved
4. After `wake_up()`, the NPU memory backing the hadamard tensor is invalidated/zeroed
5. ALL downstream DSA attention produces zero output → ALL-ZERO inference

### Why Class Variables are Invisible to Save Mechanisms

Python class variables (defined at class level, not in `__init__`) do NOT appear in:
- `model.named_buffers()` — only registered buffers appear
- `model.named_parameters()` — only registered parameters appear
- `model.state_dict()` — only buffers and parameters appear

This is a fundamental Python attribute scoping issue: class-level attributes are shared across all instances and exist on the class object itself, not on individual model instances.

### Cross-Framework Pattern: State Lifecycle Mismatch

This bug belongs to the **State Lifecycle Mismatch** pattern family — identical structural pattern to 4 other confirmed bugs across 3 frameworks:

| Framework | Issue | Pattern | State Boundary | Fix |
|-----------|-------|---------|---------------|-----|
| vLLM-Ascend | **#10684** | Class variable invisible to named_buffers() → lost at sleep/wake | Sleep/wake | Register as model buffer |
| SGLang | #28676 | MXFP8 shuffle cache clobbered on weight reload | Weight reload | dict.clear() at boundary |
| SGLang | #28679 | GDN state accumulates stale data over uptime | Time accumulation | Periodic flush mechanism |
| vLLM | #44395 | KV cache still asleep after partial wake_up | Partial wake | Staged wake ALL tags |
| vLLM-Ascend | #10579 | torch.abs() destroys negative-index contract | Operator semantics | Remove abs() |

All share the same root structure: **GPU-resident state that should be preserved across a lifecycle boundary is either not saved, not restored, or corrupted by an incorrect transformation.**

### Fix Directions

**Option 1 (Best — Automatic lifecycle):** Register hadamard as a model buffer:
```python
# In model definition (e.g., deepseek_v4.py):
self.register_buffer('hadamard', hadamard_tensor, persistent=True)
```
Now appears in `named_buffers()` → automatically saved/restored by sleep/wake. Zero additional code in sleep/wake path.

**Option 2 (Quick workaround):** Re-compute hadamard after `wake_up()`:
```python
# In AscendDSACPMetadataBuilder.wake_up() or worker:
def wake_up(self, ...):
    # ... existing wake_up logic ...
    self.hadamard = create_hadamard_matrix(hadamard_size).to(device)
```
Minimal change, doesn't modify model architecture. Need to know dimensions at wake time.

**Option 3 (Complementary):** Add validation check after wake_up:
```python
# After wake_up, verify hadamard is non-zero:
if self.hadamard is not None and self.hadamard.abs().sum() == 0:
    logger.error("DSA Hadamard is ALL-ZERO after wake_up! Re-computing...")
    self.hadamard = create_hadamard_matrix(hadamard_size).to(device)
```
Provides both detection AND self-healing.

### verl RLHF Impact

★★★★ This bug is a **BLOCKER** for verl RLHF/GRPO training on Ascend NPU. verl's HYBRID sleep/wake architecture cycles: Rollout → Sleep → Train → Wake → Rollout, every training step. If wake doesn't restore hadamard, ALL rollout forward passes produce zero output → ALL rewards = 0 → GRPO/PPO completely broken from step 1.

### RTX 4090 / CUDA Pattern Transfer

This pattern is NOT Ascend-specific — it applies to ANY platform with sleep/wake weight transfer:
- **CUDA (vLLM #44395):** Partial wake leaves KV cache asleep → illegal memory access
- **CUDA (SGLang #28676):** Weight reload clobbers MoE shuffle cache → 64x accuracy blowup
- **Lesson:** ANY GPU-resident constant buffer MUST be stored as a model buffer (not class variable) for automatic save/restore at state lifecycle boundaries

Thanks for raising this critical issue!
```

---

## Posting Strategy

1. ★★★★★★★★ MUST get user authorization before posting on vllm-project/vllm-ascend #10684
2. Post this comment → provides cross-framework State Lifecycle Mismatch pattern analysis
3. Propose 3 fix directions with code examples
4. Highlight verl RLHF blocker impact
5. Track engagement → if maintainers respond, collaborate on review

## Priority: P7 C17 (HIGH) — DSA Hadamard sleep/wake fix for Ascend NPU

★★★★★★★★★ This is a UNIQUE contribution:
  → Root cause confirmation: class variable vs model buffer
  → Cross-framework State Lifecycle Mismatch pattern family (5 instances)
  → 3 fix directions with code examples
  → verl RLHF blocker impact analysis
  → RTX 4090 pattern transfer (CUDA equivalents)

---

## References

- Issue: https://github.com/vllm-project/vllm-ascend/issues/10684
- Deep reading: notebook/projects/vllm-ascend-10684-dsa-hadamard-sleep-wake-reading.md
- SGLang #28676: MoE cache clobber (same pattern family)
- vLLM #44395: KV cache partial wake (same pattern family)
- Pattern family: notebook/fundamentals/state-lifecycle-mismatch-pattern-family-derivation.md
