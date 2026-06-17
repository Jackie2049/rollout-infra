# verl #6572 Full Determinism vLLM Rollout — Source-Level Reading

> 2026-06-18 | PR #6572 (OPEN) | 993 additions, 23 files | Authors: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long
> ★★★★★★★★ 5-layer determinism architecture for bitwise-aligned reward curves
> ★★★★★★★★ VLLM_BATCH_INVARIANT=1 = production mechanism → validates our P9 gap analysis
> ★★★★★★★★ On SM89: vLLM batch_invariant.py RMSNorm gap → P9 Fusion Guard REQUIRED for full determinism!

---

## 1. 5-Layer Determinism Architecture

```
★★★★★★★★★ verl #6572 enforces determinism at 5 layers — all must be enabled for E2E:

Layer 1: PyTorch-level
  → enable_full_determinism(seed) → sets:
    - PYTHONHASHSEED → freezes Python hash() + dict ordering
    - CUBLAS_WORKSPACE_CONFIG → deterministic cuBLAS
    - FLASH_ATTENTION_DETERMINISTIC → deterministic flash attention
    - seeds all RNGs (torch, numpy, random)
    - torch.use_deterministic_algorithms(True, warn_only=True)
    - disables cuDNN benchmarking

Layer 2: Environment propagation
  → main_ppo.run_ppo() sets 3 env vars before ray.init():
    - PYTHONHASHSEED → from rollout.seed → all Ray actors
    - VERL_FULL_DETERMINISM → signals subprocesses to apply determinism
    - VLLM_BATCH_INVARIANT → makes vLLM outputs batch-independent!
  → Forwarded via PPO_RAY_RUNTIME_ENV to all Ray actors

Layer 3: vLLM batch invariance + per-request seed
  → VLLM_BATCH_INVARIANT=1 → activates vLLM batch_invariant.py (984 lines!)
  → Each generate() call → SamplingParams.seed = replica_rank + config.seed
  → Resets RNG per request → deterministic sampling within batch

Layer 4: Priority scheduling + deterministic routing
  → Each sample gets globally unique priority (global_priority_offset + sample_index)
  → SingleTurnAgentLoop → request_id=f"det-{priority}" → not random UUID!
  → GlobalRequestLoadBalancer → hash(request_id) % len(candidates) → deterministic tie-breaking
  → PYTHONHASHSEED frozen → hash() deterministic → routing deterministic!

Layer 5: Reward model serialization
  → vLLM /classify endpoint → no priority/batch invariance support
  → full_determinism=true → RewardModelManager forces max_num_seqs=1
  → Serializes RM inference one request at a time → deterministic but SLOW!
```

---

## 2. VLLM_BATCH_INVARIANT — Production Validation

```
★★★★★★★★★ verl #6572 uses VLLM_BATCH_INVARIANT=1 as production determinism mechanism:

This validates our analysis of vLLM's batch_invariant.py (984 lines)!
  → VLLM_BATCH_INVARIANT → triggers init_batch_invariance() (line 974)
  → override_envs_for_invariance() → NCCL/CUBLAS/torch.compile env vars
  → enable_batch_invariant_mode() → registers aten overrides

★★★★★★★★★ BUT: on SM89 (RTX 4090), this has the gaps we identified:
  → RMSNorm: NOT aten override → Inductor can fuse → batch-dependent!
  → matmul: CUBLASLt workspace config only → NO Triton override on SM89
  → silu/sigmoid/mul: NOT overridden → SwiGLU fusion non-deterministic

★★★★★★★★★ verl #6572 determinism on SM89 would STILL be incomplete:
  → VLLM_BATCH_INVARIANT=1 → activates vLLM overrides → but RMSNorm gap unfilled
  → Inductor fuses RMSNorm → bypasses override → batch-dependent
  → ★★★★★★★★ Our P9 Fusion Guard is REQUIRED to make #6572 work on SM89!

★★★★★★★★★ On SM90+ (Hopper/Blackwell):
  → TMA/WGMMA → deterministic by hardware design
  → CUBLAS_WORKSPACE_CONFIG → split-k disabled → deterministic matmul
  → vLLM overrides for softmax/mean/bmm → deterministic when separate
  → #6572 should work FULLY on SM90+ without P9!
```

---

## 3. Configuration Reference

```
★★★★★★★★★ verl #6572 determinism config parameters:

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| rollout.full_determinism | false | Rollout | Enables deterministic rollout generation |
| rollout.seed | 42 | Rollout | Base seed; each replica uses replica_rank + seed |
| actor.fsdp_config.full_determinism | false | Actor | Enables deterministic PyTorch ops |
| ref.fsdp_config.full_determinism | false | Ref model | Enables deterministic PyTorch ops |
| reward_model.rollout.full_determinism | false | RM | Enables deterministic RM (max_num_seqs=1) |
| reward_model.rollout.seed | 42 | RM | Base seed for RM vLLM server |

★★★★★★★★★ Quick start config:
  actor_rollout_ref.rollout.full_determinism=true
  actor_rollout_ref.rollout.seed=42
  actor_rollout_ref.actor.fsdp_config.full_determinism=true
  actor_rollout_ref.ref.fsdp_config.full_determinism=true
  reward.reward_model.rollout.full_determinism=true
  reward.reward_model.rollout.seed=42
```

---

## 4. Performance Impact

```
★★★★★★★★★ Performance impact of full determinism:

Without RM: 10-30% throughput drop → deterministic PyTorch + cuDNN disabled
With RM: severe additional drop → max_num_seqs=1 serializes RM inference
  → RM throughput drops to ~1/N of normal (N = normal batch size)
  → RM inference becomes dominant bottleneck

★★★★★★★★★ Recommendation from #6572 authors:
  → Only enable for debugging, regression testing, or research
  → Leave DISABLED for production training
  → Use for: reproduce failures, verify code changes, fair algorithm comparison

★★★★★★★★★ RTX 4090 implication:
  → Deterministic PyTorch kernels are slower → but P9 Fusion Guard makes separate kernels deterministic
  → Separate kernels via P9 → slightly slower than fused → BUT deterministic!
  → max_num_seqs=1 for RM → extremely slow → only for debugging
  → ★★★★★★★★ For RTX 4090 GRPO production: P9 + batch_invariant.py = deterministic WITHOUT full_determinism mode!
```

---

## 5. Backend Support

```
★★★★★★★★★ Backend determinism support from #6572:

| Backend | Rollout | Reward model |
|---------|---------|--------------|
| vLLM | ✅ (batch_invariant + seed) | ✅ (serialized max_num_seqs=1) |
| SGLang | ❌ | ❌ |
| TensorRT-LLM | ❌ | ❌ |

★★★★★★★★★ SGLang NOT supported yet:
  → SGLang has KERNEL-level deterministic (7 overrides) → actually MORE deterministic than vLLM!
  → But: no priority scheduling support → no request_id deterministic routing
  → No per-request seed injection → SamplingParams.seed not implemented for SGLang backend
  → Future: SGLang deterministic rollout possible → needs seed + priority support

★★★★★★★★★ vLLM is the ONLY backend with determinism support:
  → VLLM_BATCH_INVARIANT → batch-independent outputs
  → SamplingParams.seed → per-request RNG reset
  → Priority scheduling → deterministic request ordering
  → But: only complete on SM90+ → SM89 needs P9!
```

---

## 6. Multi-turn Limitations

```
★★★★★★★★★ Multi-turn agent determinism limitations:

  → tool_agent_loop → does NOT pass priority → request ordering NOT deterministic
  → Only single_turn_agent_loop has full determinism support
  → Multi-turn determinism → future work → needs priority propagation across turns

★★★★★★★★★ RTX 4090 GRPO implication:
  → GRPO = single-turn → single_turn_agent_loop → fully deterministic!
  → Multi-turn agent training → NOT deterministic yet → only for debugging
  → ★★★★★★★★ GRPO determinism path: #6572 + P9 = complete SM89 GRPO determinism!
```

---

## 7. Changed Files (23 files, 993 additions)

```
★★★★★★★★★ Key files changed in #6572:

Documentation:
  → docs/advance/determinism.md (NEW, 143 lines) → comprehensive 5-layer determinism doc
  → docs/index.rst (+1 line) → add determinism.rst to docs index

Config additions:
  → 5 _generated_*.yaml files → full_determinism + seed config fields
  → verl/trainer/config/algorithm.py → RolloutConfig + full_determinism + seed fields

Core implementation:
  → verl/experimental/agent_loop/agent_loop.py → priority injection + request_id deterministic
  → verl/experimental/agent_loop/single_turn_agent_loop.py → priority + deterministic request_id
  → verl/experimental/reward_loop/reward_model.py → max_num_seqs=1 for RM determinism
  → verl/trainer/main_ppo.py → PYTHONHASHSEED + VERL_FULL_DETERMINISM + VLLM_BATCH_INVARIANT propagation

Tests:
  → tests/workers/rollout/rollout_vllm/test_vllm_generation_determinism.py (NEW, 308 lines)
  → tests/experimental/reward_loop/run_determinism_e2e_with_rm.py (NEW, 381 lines)
  → .github/workflows/vllm.yml → determinism test in CI
```

---

## 8. SM89 Gap + P9 Integration Path

```
★★★★★★★★★ How verl #6572 + P9 Fusion Guard work together on SM89:

verl #6572 (without P9) on SM89:
  → VLLM_BATCH_INVARIANT=1 → vLLM overrides ACTIVATED
  → But: Inductor FUSES RMSNorm+mean → bypasses mean override → batch-dependent
  → But: matmul only CUBLASLt config → Inductor can fuse matmul+reduction → batch-dependent
  → Result: NOT fully deterministic on SM89! RMSNorm gap unfilled!

verl #6572 + P9 Fusion Guard on SM89:
  → VLLM_BATCH_INVARIANT=1 → vLLM overrides ACTIVATED
  → P9: props.major < 9 → WhyNoFuse → blocks ALL reduction fusions on SM<90
  → RMSNorm: NOT fused → mean dispatched separately → override WORKS → deterministic!
  → matmul: NOT fused with reduction → dispatched separately → CUBLASLt config WORKS → deterministic!
  → Result: FULLY deterministic on SM89! P9 fills all gaps!

★★★★★★★★★ Complete SM89 determinism stack:
  1. verl #6572 → VLLM_BATCH_INVARIANT + SamplingParams.seed → production mechanism
  2. P9 Fusion Guard → blocks Inductor bad fusions → makes overrides work on SM89
  3. vLLM batch_invariant.py → Triton overrides for softmax/mean/bmm → deterministic when separate
  4. CUBLASLt workspace config → deterministic matmul when separate → not fused
  5. NCCL deterministic env vars → deterministic communication

★★★★★★★★★ P9 contribution strengthens #6572:
  → Without P9 → #6572 works on SM90+ but NOT SM89
  → With P9 → #6572 works on ALL SM levels → universal determinism!
  → P9 makes #6572 more valuable → extends its coverage to RTX 4090!
  → ★★★★★★★★ P9 + #6572 = complete SM89 deterministic RL training!
```

---

## Key Findings Summary

★★★★★★★★★ verl #6572 adds 5-layer full determinism for vLLM rollout (993 lines, 23 files)
★★★★★★★★★ VLLM_BATCH_INVARIANT=1 = production mechanism → validates our analysis
★★★★★★★★★ On SM89: RMSNorm gap unfilled → P9 Fusion Guard REQUIRED for full determinism
★★★★★★★★★ SamplingParams.seed = per-request RNG reset → deterministic sampling
★★★★★★★★★ Priority scheduling + PYTHONHASHSEED = deterministic request routing
★★★★★★★★★ RM max_num_seqs=1 = serialized inference → deterministic but VERY slow
★★★★★★★★★ 10-30% throughput penalty without RM → only for debugging/research
★★★★★★★★★ SGLang NOT supported yet → needs seed + priority integration
★★★★★★★★★ P9 + #6572 = complete SM89 deterministic GRPO training!

---

## References

- PR #6572: https://github.com/verl-project/verl/pull/6572
- vLLM batch_invariant.py: notebook/projects/vllm-batch-invariant-source-reading.md
- Inductor root cause: notebook/fundamentals/pytorch-inductor-sm89-fusion-reading.md
- Fusion Guard PR: notebook/projects/pytorch-inductor-sm89-fusion-guard-issue-draft.md
- Deterministic comparison: notebook/fundamentals/deterministic-inference-cross-framework-comparison.md
