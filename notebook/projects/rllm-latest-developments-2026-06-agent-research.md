# rLLM Latest Developments June 2026 — Agent Research Findings

> 2026-06-18 | Agent research results | Key findings from rLLM latest PRs/issues/releases
> ★★★★★★★★ CRITICAL: rLLM repo is rllm-org/rllm (NOT agent-RL/rLLM as previously thought)
> ★★★★★★★★ CRITICAL: #6717 DOES NOT EXIST — was an error in memory, now removed
> ★★★★★★★★ CRITICAL: #605 OPEN — GRPO grouping bug (trajectory.uid vs task_ids → group size 1)

---

## 1. Repository Details

```
★★★★★★★★★ rLLM repo: https://github.com/rllm-org/rllm
  → Stars: 5,628 | Language: Python | Last push: June 17
  → Latest release: v0.3.0-pre (April 30, 2026)
  → No newer stable release since v0.2.1.post1 (Dec 2025)
  → ★★★★★★★★ CORRECTION: repo is rllm-org/rllm, NOT agent-RL/rLLM!
```

---

## 2. Key PRs Status

```
★★★★★★★★★ #658 (MERGED June 16): Cumulative token mode for Tinker
  → Extends cumulative_token_mode to in-process local_handler (Tinker) path
  → Previously only worked for HTTP-proxy (vLLM/verl) backend
  → Now Tinker detects pre-tokenized prompt (list[int]) → samples directly
  → Byte-for-byte prefix-extension guarantee: turn N prompt = prior turns prompt+completion
  → +428/-31, 4 files changed, authored by jeffreysijuntan
  → ★★★★★★★★ Key: drift-free multi-turn for Tinker → RTX 4090 GRPO benefit!

★★★★★★★★★ #630 (MERGED June 9): Deterministic optimizer steps
  → Decouples mini_batch_size from global_batch_size
  → Number of optimizer steps now deterministic: r = train_batch_size // ppo_mini_batch_size
  → Loss denominator fixed at total_rollouts // r
  → Padding reduced: example dp_size=2, ppo_mbs=32, rollout_n=4, N=298 → old 86 wasted, new 2 wasted

★★★★★★★★★ #576 (OPEN, stalled 6+ weeks): MergedSegment
  → Last activity: May 12 → no updates since → 6+ weeks stale
  → Backend-agnostic step merge → Tinker and verl thin adapters
  → Net diff: -262/+136 lines → 33 tests → waiting on e2e tests

★★★★★★★★★ #6717 DOES NOT EXIST in rllm-org/rllm!
  → Previous memory listed this as "Tinker training worker primitives MERGED"
  → This was an ERROR → no such PR number exists in the repo
  → Removed from memory → closest is #658 (cumulative token) or #607 (unified trainer)
```

---

## 3. CRITICAL Bug: #605 GRPO Grouping

```
★★★★★★★★★ #605 (OPEN, June 2): GRPO groups by trajectory.uid instead of task_ids

Problem:
  → GRPO grouping currently uses trajectory.uid (per-trajectory UUID)
  → NOT task_ids (prompt-level identifier)
  → Result: 4 rollouts of same prompt → NOT grouped → advantage = raw reward (group size 1)
  → Expected: group mean 0.5, advantages ±0.87 with norm_adv
  → Actual: advantages = raw reward values → no normalization!

★★★★★★★★★ This is a CORRECTNESS bug for GRPO on rLLM:
  → Without proper grouping → GRPO advantage = individual reward → no relative comparison
  → GRPO relies on group-relative advantage → n=8 same prompt → same group → normalize
  → If grouped by uid → each trajectory unique → group size 1 → no normalization
  → Suggests using task_ids for enable=False/broadcast, {task_id}_step{k} for per_step

★★★★★★★★★ Affects agent_workflow_trainer.py and transform.py
  → Significant correctness bug → needs fixing before GRPO can work properly
```

---

## 4. Recent Merged PRs (June 1-17)

```
★★★★★★★★★ 20+ PRs merged in last 2-3 weeks:

Key Tinker/GRPO related:
  → #661: Sandbox concurrency 4→64 → directly impacts Tinker throughput
  → #658: Cumulative token Tinker → drift-free multi-turn
  → #630: Deterministic optimizer steps → decoupled batch sizes
  → #607: Graduated unified trainer from experimental → canonical
  → #627: Backend-agnostic StatefulTaskDataLoader → checkpoint resume

Key infrastructure:
  → #650: Terminal-RL mega-merge (71 files) → Terminus-2, AgentFlow redesign, warm-pool
  → #649: pass@k via --attempts → unbiased pass@k estimator
  → #647: Terminus-2 agent harness
  → #638: AgentFlow phase 3 → engine owns sandbox
  → #636: AgentFlow phases 0-1 → always-hooks engine
  → #634: RLLM_* env vars (12 operational knobs)
  → #632: Snapshot + warm-pool acceleration
```

---

## 5. Open Issues

```
★★★★★★★★★ Key open issues:

#605 (June 2): GRPO grouping bug → trajectory.uid vs task_ids → CRITICAL
#655 (June 15): Dockerfile RUN-replay corrupts Harbor task setup
#434 (recently updated June 17): Qwen3.5 support request
```

---

## 6. Checkpoint Export Status

```
★★★★★★★★★ NO progress on checkpoint export:
  → No PRs or issues related to exporting Tinker/LoRA checkpoints to PEFT format
  → #627 improved Tinker checkpoint resume (mirrors verl layout) → but for resume, not export
  → The "export gap" remains unfilled → Tinker checkpoint NOT standard PEFT
```

---

## 7. verl Integration

```
★★★★★★★★★ verl integration deepening:
  → #607: verl code graduated to canonical (rllm/trainer/verl/)
  → #627: Config sync → rllm CLI > native CLI > rllm yaml > native yaml
  → #658: verl cumulative token mode now works on Tinker (was verl-only)
  → #543: verl R2 and R3 MoE router replay support (MERGED earlier)
```

---

## Key Findings Summary

★★★★★★★★★ rLLM repo: rllm-org/rllm (NOT agent-RL/rLLM) — corrected!
★★★★★★★★★ #6717 DOES NOT EXIST — removed from memory
★★★★★★★★★ #605 CRITICAL: GRPO grouping bug → trajectory.uid vs task_ids → group size 1 → no advantage normalization
★★★★★★★★★ #658 MERGED: cumulative token Tinker → drift-free multi-turn → RTX 4090 benefit
★★★★★★★★★ #630 MERGED: deterministic optimizer steps → decoupled batch sizes
★★★★★★★★★ #576 stalled 6+ weeks → MergedSegment pending e2e tests
★★★★★★★★★ Checkpoint export gap: STILL unfilled → no progress

---

## References

- rLLM repo: https://github.com/rllm-org/rllm
- #658: https://github.com/rllm-org/rllm/pull/658
- #630: https://github.com/rllm-org/rllm/pull/630
- #605: https://github.com/rllm-org/rllm/issues/605
- #576: https://github.com/rllm-org/rllm/pull/576
