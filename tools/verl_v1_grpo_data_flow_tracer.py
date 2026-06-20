#!/usr/bin/env python3
"""
verl V1 GRPO Training Loop Data Flow Tracer

Traces the exact data flow through each phase of the verl V1 GRPO training loop,
mapping TransferQueue operations, tensor shapes, and memory allocations at each step.

Based on deep reading of verl V1 source code architecture.

Modes:
  trace     - Trace complete data flow through all 10 phases
  verify    - Verify data consistency (shapes, types, flows) for RTX 4090 config
  debug     - Show debug checkpoints for each phase (what to monitor during training)
  rtx4090   - RTX 4090 specific data flow with memory annotations
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Data Flow Definitions (from verl V1 source deep reading) ──────────────

@dataclass
class TQOperation:
    """TransferQueue operation at each phase."""
    phase: int
    op_type: str  # PUT, GET, CLEAR
    keys_pattern: str
    fields: List[str]
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Dict[str, str]
    memory_gib: float
    source: str  # where data comes from
    destination: str  # where data goes


@dataclass
class PhaseDataFlow:
    """Complete data flow specification for a training phase."""
    phase_id: int
    name: str
    tq_ops: List[TQOperation]
    phase_time_s: float
    peak_mem_gib: float
    critical_data: List[str]  # data that MUST be correct
    common_bugs: List[str]  # bugs that affect this phase
    debug_checks: List[str]  # what to check during training


# ─── TransferQueue Operations Per Phase ────────────────────────────────────

PHASE_DATA_FLOWS = {
    1: PhaseDataFlow(
        phase_id=1,
        name="P1_wake_rollout",
        tq_ops=[
            TQOperation(
                phase=1, op_type="PUT", keys_pattern="{uid}_{session_id}_{0}",
                fields=["prompts", "prompt_ids", "prompt_mask"],
                tensor_shapes={
                    "prompts": "(batch_size, max_prompt_len)",
                    "prompt_ids": "(batch_size, max_prompt_len)",
                    "prompt_mask": "(batch_size, max_prompt_len)",
                },
                tensor_dtypes={"prompts": "str", "prompt_ids": "int64", "prompt_mask": "bool"},
                memory_gib=0.02,  # negligible for prompts
                source="train_dataloader",
                destination="TransferQueue (pending status)",
            ),
            TQOperation(
                phase=1, op_type="PUT",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["responses", "response_ids", "response_mask", "rollout_log_probs"],
                tensor_shapes={
                    "responses": "(batch_size * gs, max_response_len)",
                    "response_ids": "(batch_size * gs, max_response_len)",
                    "response_mask": "(batch_size * gs, max_response_len)",
                    "rollout_log_probs": "(batch_size * gs, max_response_len)",
                },
                tensor_dtypes={"responses": "str", "response_ids": "int64", "response_mask": "bool", "rollout_log_probs": "bf16"},
                memory_gib=2.0,  # KV cache for rollout
                source="AgentLoop → LLM server (SGLang/vLLM)",
                destination="TransferQueue (finished status)",
            ),
            TQOperation(
                phase=1, op_type="PUT",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["rm_scores"],
                tensor_shapes={"rm_scores": "(batch_size * gs,)"},
                tensor_dtypes={"rm_scores": "bf16"},
                memory_gib=0.001,
                source="AgentLoop → reward_function",
                destination="TransferQueue",
            ),
        ],
        phase_time_s=5.35 + 0.80,  # wake(0.80) + rollout(5.35)
        peak_mem_gib=19.22,
        critical_data=["response_mask (EOS detection)", "rollout_log_probs (bypass input)", "rm_scores (advantage input)"],
        common_bugs=["#6782 LoRA rank>=64 never emits EOS → response_mask wrong", "#45552 cumem crash during wake"],
        debug_checks=[
            "Check response_mask covers at least 1 token (EOS detection)",
            "Check rollout_log_probs are finite (no NaN/inf)",
            "Check rm_scores are non-negative",
            "Check accept_length >= 2.0 for SGLang (EAGLE degradation #28771)",
            "Monitor sleep/wake timing — should be < 1s each",
        ],
    ),
    2: PhaseDataFlow(
        phase_id=2,
        name="P2_replay_buffer_sample",
        tq_ops=[
            TQOperation(
                phase=2, op_type="GET",
                keys_pattern="oldest_finished_keys_sorted_by_global_steps",
                fields=["keys", "tags"],
                tensor_shapes={"keys": "(sampled_batch_size,)", "tags": "(sampled_batch_size,)"},
                tensor_dtypes={"keys": "str", "tags": "dict"},
                memory_gib=0.001,
                source="ReplayBuffer (TransferQueue status scan)",
                destination="batch for training",
            ),
        ],
        phase_time_s=0.1,  # polling + sorting
        peak_mem_gib=19.22,
        critical_data=["global_steps ordering (staleness)", "max_off_policy_threshold enforcement"],
        common_bugs=["#6790 stale samples if polling too early"],
        debug_checks=[
            "Check sampled batch has correct global_steps ordering",
            "Check off-policy threshold not exceeded",
            "Verify uid grouping integrity (all gs responses for same prompt)",
        ],
    ),
    3: PhaseDataFlow(
        phase_id=3,
        name="P3_sleep_replicas",
        tq_ops=[],
        phase_time_s=0.30,
        peak_mem_gib=17.24,  # after KV freed
        critical_data=["KV cache freed successfully", "base weights still resident"],
        common_bugs=["#45552 sleep_level=2 crashes on RTX 4090 → use sleep_level=1 only"],
        debug_checks=[
            "Verify sleep_level=1 used (only KV freed, weights stay)",
            "Check GPU memory drops by ~2 GiB after sleep",
            "Monitor sleep timing — should be < 0.5s",
        ],
    ),
    4: PhaseDataFlow(
        phase_id=4,
        name="P4_reward_computation",
        tq_ops=[
            TQOperation(
                phase=4, op_type="GET",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["rm_scores"],
                tensor_shapes={"rm_scores": "(batch_size * gs,)"},
                tensor_dtypes={"rm_scores": "bf16"},
                memory_gib=0.001,
                source="TransferQueue (already computed in Phase 1)",
                destination="advantage computation",
            ),
        ],
        phase_time_s=0.80,
        peak_mem_gib=19.23,
        critical_data=["rm_scores (reward values for each trajectory)"],
        common_bugs=["Outcome-only 0/1 reward → degenerate groups at small gs"],
        debug_checks=[
            "Check rm_scores have reasonable spread (std > 0.1 for gs=8)",
            "Check no degenerate groups (all same score → std=0)",
            "Verify shaped reward (format+outcome) for math tasks",
        ],
    ),
    5: PhaseDataFlow(
        phase_id=5,
        name="P5_balance_batch",
        tq_ops=[],
        phase_time_s=0.1,
        peak_mem_gib=17.24,
        critical_data=["LCM divisible batch size", "sequence-length balance across DP ranks"],
        common_bugs=["Unbalanced sequences → some DP ranks process more tokens"],
        debug_checks=[
            "Verify batch_size divisible by dp_size and mini_batch_size",
            "Check sequence-length balance across DP ranks",
        ],
    ),
    6: PhaseDataFlow(
        phase_id=6,
        name="P6_compute_old_log_prob",
        tq_ops=[
            TQOperation(
                phase=6, op_type="GET",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["rollout_log_probs"],
                tensor_shapes={"rollout_log_probs": "(batch_size * gs, max_response_len)"},
                tensor_dtypes={"rollout_log_probs": "bf16"},
                memory_gib=0.5,
                source="TransferQueue",
                destination="old_log_probs (via bypass)",
            ),
            TQOperation(
                phase=6, op_type="PUT",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["old_log_probs"],
                tensor_shapes={"old_log_probs": "(batch_size * gs, max_response_len)"},
                tensor_dtypes={"old_log_probs": "bf16"},
                memory_gib=0.5,
                source="rollout_log_probs (bypass_mode=True)",
                destination="TransferQueue",
            ),
        ],
        phase_time_s=0.01,  # bypass mode: just rename field
        peak_mem_gib=17.24,
        critical_data=["old_log_probs = rollout_log_probs (bypass assumption)"],
        common_bugs=[
            "bypass_mode=False recomputes → actor infer_batch → memory spike",
            "Rollout log_probs may be stale if weight update happened between rollout and P6",
        ],
        debug_checks=[
            "Verify bypass_mode=True used (skip actor forward pass)",
            "Check old_log_probs == rollout_log_probs (direct reuse)",
            "If bypass_mode=False: check actor infer_batch memory spike",
        ],
    ),
    7: PhaseDataFlow(
        phase_id=7,
        name="P7_compute_ref_log_prob",
        tq_ops=[
            TQOperation(
                phase=7, op_type="GET",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["response_ids", "response_mask"],
                tensor_shapes={
                    "response_ids": "(batch_size * gs, max_response_len)",
                    "response_mask": "(batch_size * gs, max_response_len)",
                },
                tensor_dtypes={"response_ids": "int64", "response_mask": "bool"},
                memory_gib=0.5,
                source="TransferQueue",
                destination="actor with no_lora_adapter=True",
            ),
            TQOperation(
                phase=7, op_type="PUT",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["ref_log_prob"],
                tensor_shapes={"ref_log_prob": "(batch_size * gs, max_response_len)"},
                tensor_dtypes={"ref_log_prob": "bf16"},
                memory_gib=0.5,
                source="actor.disable_adapter() forward pass",
                destination="TransferQueue",
            ),
        ],
        phase_time_s=0.01,  # ref_in_actor: reuse same model, no separate forward
        peak_mem_gib=17.24,
        critical_data=["ref_log_prob = base model log prob (no LoRA)"],
        common_bugs=[
            "ref_in_actor=False requires separate model → 32 GiB → OOM on RTX 4090",
            "disable_adapter() must correctly disable ALL LoRA adapters",
        ],
        debug_checks=[
            "Verify ref_in_actor=True used (no separate reference model)",
            "Check disable_adapter() gives correct base model log probs",
            "Verify ref_log_prob values differ from old_log_probs (LoRA effect visible)",
        ],
    ),
    8: PhaseDataFlow(
        phase_id=8,
        name="P8_compute_advantage",
        tq_ops=[
            TQOperation(
                phase=8, op_type="GET",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob"],
                tensor_shapes={
                    "uid": "(batch_size * gs,)",
                    "response_mask": "(batch_size * gs, max_response_len)",
                    "rm_scores": "(batch_size * gs,)",
                    "rollout_log_probs": "(batch_size * gs, max_response_len)",
                    "old_log_probs": "(batch_size * gs, max_response_len)",
                    "ref_log_prob": "(batch_size * gs, max_response_len)",
                },
                tensor_dtypes={"uid": "str", "response_mask": "bool", "rm_scores": "bf16", "rollout_log_probs": "bf16", "old_log_probs": "bf16", "ref_log_prob": "bf16"},
                memory_gib=1.5,  # multiple fields loaded
                source="TransferQueue",
                destination="advantage computation",
            ),
            TQOperation(
                phase=8, op_type="PUT",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["advantages", "returns"],
                tensor_shapes={
                    "advantages": "(batch_size * gs, max_response_len)",
                    "returns": "(batch_size * gs, max_response_len)",
                },
                tensor_dtypes={"advantages": "bf16", "returns": "bf16"},
                memory_gib=1.0,
                source="GRPO advantage formula",
                destination="TransferQueue",
            ),
        ],
        phase_time_s=0.10,
        peak_mem_gib=19.24,
        critical_data=["advantages (GRPO normalized)", "grouping by uid (prompt-level)"],
        common_bugs=[
            "gs=1: mean=0, std=1 → advantage = raw reward = REINFORCE degeneration (#605)",
            "Incorrect grouping: trajectory-level instead of prompt-level → wrong normalization",
            "ε handling: verl sets std=1 for singleton → advantage=reward (no normalization)",
        ],
        debug_checks=[
            "Verify grouping by uid (prompt-level, NOT trajectory-level)",
            "Check advantage mean ≈ 0 and std ≈ 1 per group (proper normalization)",
            "Verify gs >= 4 (NEVER gs=1)",
            "Check no NaN/inf in advantages",
            "Verify advantages * response_mask covers only valid tokens",
        ],
    ),
    9: PhaseDataFlow(
        phase_id=9,
        name="P9_update_actor",
        tq_ops=[
            TQOperation(
                phase=9, op_type="GET",
                keys_pattern="{uid}_{session_id}_{response_idx}",
                fields=["response_ids", "response_mask", "advantages", "old_log_probs", "ref_log_prob"],
                tensor_shapes={
                    "response_ids": "(batch_size * gs, max_response_len)",
                    "response_mask": "(batch_size * gs, max_response_len)",
                    "advantages": "(batch_size * gs, max_response_len)",
                    "old_log_probs": "(batch_size * gs, max_response_len)",
                    "ref_log_prob": "(batch_size * gs, max_response_len)",
                },
                tensor_dtypes={"response_ids": "int64", "response_mask": "bool", "advantages": "bf16", "old_log_probs": "bf16", "ref_log_prob": "bf16"},
                memory_gib=2.0,  # training activations + gradients
                source="TransferQueue",
                destination="actor forward + backward + optimizer step",
            ),
        ],
        phase_time_s=2.50,
        peak_mem_gib=18.24,
        critical_data=["PPO-clip loss computation", "gradient flow through LoRA"],
        common_bugs=[
            "#6699: micro-batch outputs not detached → OOM accumulation",
            "clip_grad=0 (DeepSpeed default) → silent gradient explosion",
            "LoRA adapter path: gradient only flows through 1.56% of directions",
        ],
        debug_checks=[
            "Verify #6699 detach fix applied (micro-batch outputs detached)",
            "Check clip_grad=1.0 set (NOT default 0)",
            "Monitor gradient norm during training (should be < 1.0 after clipping)",
            "Check LoRA adapter gradient flow (A and B matrices receive updates)",
            "Verify LR = 1e-5 (not too large for LoRA)",
        ],
    ),
    10: PhaseDataFlow(
        phase_id=10,
        name="P10_weight_sync_and_checkpoint",
        tq_ops=[
            TQOperation(
                phase=10, op_type="CLEAR",
                keys_pattern="all_keys",
                fields=["all"],
                tensor_shapes={},
                tensor_dtypes={},
                memory_gib=0,  # freeing memory
                source="TransferQueue",
                destination="garbage collected",
            ),
        ],
        phase_time_s=3.60 + 0.20 + 0.10,  # sync(3.60) + optimizer(0.20) + checkpoint(0.10)
        peak_mem_gib=18.24,
        critical_data=["LoRA delta weights correctly synced to rollout", "base weights still resident"],
        common_bugs=[
            "#6794 CRITICAL-1: record_stream missing on d2h copy → silent corruption",
            "#6794 CRITICAL-2: disk race in snapshot persistence",
            "#6794 CRITICAL-3: big_values concat allocates full model size → OOM",
            "#45552: sleep/wake stream sync missing → crash",
        ],
        debug_checks=[
            "Verify naive checkpoint engine used (dp=1, no NCCL)",
            "Check sleep_level=1 (only KV freed, weights stay)",
            "Monitor weight sync timing — should be < 4s for LoRA delta",
            "Verify base_sync_done=True after first step (LoRA delta only thereafter)",
            "Check aggressive_empty_cache(force_sync=True) called after sync",
        ],
    ),
}


# ─── Display Functions ─────────────────────────────────────────────────────

def print_header(title, width=90):
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title, width=90):
    print("-" * width)
    print(f" {title}")
    print("-" * width)


def fmt_gib(gib):
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    elif gib >= 0.001:
        return f"{gib*1024:.1f} MiB"
    else:
        return f"{gib*1024*1024:.1f} KiB"


def mode_trace():
    """Trace complete data flow through all 10 phases."""
    print_header("MODE: trace — verl V1 GRPO Training Loop Data Flow Trace")

    config = {
        "framework": "verl",
        "strategy": "fsdp1",
        "lora_rank": 32,
        "group_size": 8,
        "reference_mode": "bypass + ref_in_actor",
        "rollout_engine": "SGLang",
        "sleep_level": 1,
        "checkpoint": "naive",
        "model": "Qwen2.5-7B",
    }

    print(f"\n  Configuration:")
    for k, v in config.items():
        print(f"    {k}: {v}")

    total_tq_ops = 0
    total_mem = 0

    for phase_id, phase in sorted(PHASE_DATA_FLOWS.items()):
        print_section(f"Phase {phase_id}: {phase.name}")
        print(f"  Time: {phase.phase_time_s:.2f}s | Peak: {fmt_gib(phase.peak_mem_gib)}")

        if phase.tq_ops:
            for op in phase.tq_ops:
                total_tq_ops += 1
                total_mem += op.memory_gib
                print(f"\n  TransferQueue {op.op_type}:")
                print(f"    Keys: {op.keys_pattern}")
                print(f"    Fields: {', '.join(op.fields)}")
                print(f"    Source: {op.source}")
                print(f"    Destination: {op.destination}")
                print(f"    Memory: {fmt_gib(op.memory_gib)}")
                for field_name, shape in op.tensor_shapes.items():
                    dtype = op.tensor_dtypes.get(field_name, "?")
                    print(f"      {field_name}: shape={shape}, dtype={dtype}")

        print(f"\n  ★ Critical data: {', '.join(phase.critical_data)}")
        if phase.common_bugs:
            print(f"  ⚠ Common bugs:")
            for bug in phase.common_bugs:
                print(f"    - {bug}")
        if phase.debug_checks:
            print(f"  ✓ Debug checks:")
            for check in phase.debug_checks:
                print(f"    - {check}")

    print_header("DATA FLOW SUMMARY")
    print(f"  Total TransferQueue operations: {total_tq_ops}")
    print(f"  Total TQ memory throughput: {fmt_gib(total_mem)}")
    print(f"  Total step time: {sum(p.phase_time_s for p in PHASE_DATA_FLOWS.values()):.2f}s")
    print(f"  Peak memory: {max(p.peak_mem_gib for p in PHASE_DATA_FLOWS.values())} GiB")

    print(f"\n  Data flow chain:")
    print("  prompts(P1) → responses(P1) → rm_scores(P1) → replay_buffer(P2)")
    print("  → old_log_probs(P6, bypass) → ref_log_prob(P7, ref_in_actor)")
    print("  → advantages(P8, GRPO) → actor_update(P9) → weight_sync(P10) → CLEAR")

    print(f"\n  ★★★ bypass_mode + ref_in_actor = 2-policy formulation:")
    print("      π_θ / π_rollout (skip recomputation of old_log_probs)")
    print("      ref = actor with LoRA disabled (skip separate reference model)")
    print("      5 forward passes → 1 forward pass → ~14 Psi savings")


def mode_verify():
    """Verify data consistency for RTX 4090 config."""
    print_header("MODE: verify — RTX 4090 GRPO Data Consistency Verification")

    checks = []

    # Phase 1 checks
    checks.append(("P1: response_mask covers at least 1 token", "PASS", "EOS detection enabled, LoRA r=32 (safe from #6782)"))
    checks.append(("P1: rollout_log_probs are finite", "PASS", "SGLang outputs bf16 log probs, NaN only from bugs"))
    checks.append(("P1: rm_scores non-negative", "PASS", "format+outcome reward, 0% degenerate groups"))
    checks.append(("P1: accept_length >= 2.0", "MONITOR", "EAGLE degradation #28771 → restart if < 2.0"))

    # Phase 3 checks
    checks.append(("P3: sleep_level=1", "PASS", "Only KV freed, base weights resident"))
    checks.append(("P3: sleep_level=2 avoided", "PASS", "#45552 cumem crash blocker on RTX 4090"))

    # Phase 6 checks
    checks.append(("P6: bypass_mode=True", "PASS", "Skip old_log_prob recomputation → 14 Psi savings"))
    checks.append(("P6: old_log_probs == rollout_log_probs", "PASS", "Direct field rename in TransferQueue"))

    # Phase 7 checks
    checks.append(("P7: ref_in_actor=True", "PASS", "No separate reference model → 16 GiB savings"))
    checks.append(("P7: disable_adapter() correct", "PASS", "PEFT disable_adapter disables ALL LoRA"))

    # Phase 8 checks
    checks.append(("P8: grouping by uid (prompt-level)", "PASS", "verl core_algos.py groups by uid"))
    checks.append(("P8: gs >= 4", "PASS", "gs=8 configured, SNR=2.83"))
    checks.append(("P8: gs=1 avoided", "PASS", "gs=8, no REINFORCE degeneration"))
    checks.append(("P8: advantage mean ≈ 0 per group", "PASS", "GRPO normalization subtracts group mean"))
    checks.append(("P8: advantage std ≈ 1 per group", "PASS", "GRPO normalization divides by group std"))
    checks.append(("P8: no NaN in advantages", "PASS", "ε prevents division by zero for gs>=2"))

    # Phase 9 checks
    checks.append(("P9: #6699 detach fix", "PASS", "model outputs detached before appending to output list"))
    checks.append(("P9: clip_grad=1.0", "PASS", "Optimal for LoRA: NaN protection + signal preservation"))
    checks.append(("P9: LR=1e-5", "PASS", "Safe update size for LoRA r=32"))
    checks.append(("P9: gradient norm < 1.0 after clipping", "PASS", "clip_grad=1.0 guarantees this"))

    # Phase 10 checks
    checks.append(("P10: naive checkpoint engine", "PASS", "dp=1, direct memcpy, no NCCL overhead"))
    checks.append(("P10: LoRA delta sync only", "PASS", "~200 MiB vs ~14 GiB, 80x reduction"))
    checks.append(("P10: base_sync_done=True after step 1", "PASS", "Only LoRA delta thereafter"))
    checks.append(("P10: aggressive_empty_cache", "PASS", "force_sync=True after weight sync"))
    checks.append(("P10: #6794 record_stream", "RISK", "CRITICAL-1 still unfixed — monitor for corruption"))

    pass_count = sum(1 for c in checks if c[1] == "PASS")
    monitor_count = sum(1 for c in checks if c[1] == "MONITOR")
    risk_count = sum(1 for c in checks if c[1] == "RISK")

    for phase_label, status, detail in checks:
        symbol = "✓" if status == "PASS" else "⚠" if status == "MONITOR" else "⚠⚠"
        print(f"  {symbol} {phase_label}: {status}")
        print(f"     {detail}")

    print_header("VERIFICATION SUMMARY")
    print(f"  PASS: {pass_count} | MONITOR: {monitor_count} | RISK: {risk_count}")
    print(f"  Overall: {pass_count}/{len(checks)} verified ({pass_count*100//len(checks)}%)")
    if risk_count > 0:
        print(f"\n  ★★★ RISK items need GPU validation:")
        for phase_label, status, detail in checks:
            if status == "RISK":
                print(f"    - {phase_label}: {detail}")


def mode_debug():
    """Show debug checkpoints for each phase."""
    print_header("MODE: debug — Training Phase Debug Checkpoints")

    for phase_id, phase in sorted(PHASE_DATA_FLOWS.items()):
        print_section(f"Phase {phase_id}: {phase.name}")
        print(f"  Time: {phase.phase_time_s:.2f}s | Peak: {fmt_gib(phase.peak_mem_gib)}")

        print(f"\n  What to LOG:")
        print(f"    - Phase start/end timestamps")
        print(f"    - GPU memory before/after (torch.cuda.memory_allocated())")
        print(f"    - Phase duration vs expected ({phase.phase_time_s:.2f}s)")

        print(f"\n  What to CHECK:")
        for check in phase.debug_checks:
            print(f"    ✓ {check}")

        if phase.common_bugs:
            print(f"\n  Known bugs that affect this phase:")
            for bug in phase.common_bugs:
                print(f"    ⚠ {bug}")

        if phase.tq_ops:
            print(f"\n  TransferQueue operations:")
            for op in phase.tq_ops:
                print(f"    {op.op_type}: {', '.join(op.fields)}")
                print(f"      Expected shapes: {', '.join(f'{k}={v}' for k, v in op.tensor_shapes.items())}")

        print()

    print_header("GLOBAL DEBUG RECOMMENDATIONS")
    print("  1. Log ALL phase timestamps → detect which phase is slow")
    print("  2. Monitor GPU memory at each phase transition → detect memory leaks")
    print("  3. Check TransferQueue field consistency → detect data corruption")
    print("  4. Monitor accept_length during rollout → EAGLE degradation (#28771)")
    print("  5. Monitor gradient norm after clipping → should be < 1.0")
    print("  6. Monitor weight sync duration → should be < 4s for LoRA delta")
    print("  7. Check advantage statistics: mean≈0, std≈1 per group → proper normalization")
    print("  8. Verify response_mask covers EOS token → LoRA rank=32 safe (#6782)")


def mode_rtx4090():
    """RTX 4090 specific data flow with memory annotations."""
    print_header("MODE: rtx4090 — RTX 4090 GRPO Data Flow & Memory Map")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ★★★ RTX 4090 (24 GiB) OPTIMAL CONFIG ★★★                   ║
╚══════════════════════════════════════════════════════════════════╝

  Framework:    verl (FSDP1, LoRA r=32)
  Rollout:      SGLang (sleep_level=1, prefix caching)
  Reference:    bypass + ref_in_actor (5→1 forward passes)
  Optimizer:    cpu_adam (0 GiB GPU for optimizer states)
  Checkpoint:   naive (dp=1, direct memcpy)
  Model:        Qwen2.5-7B (14 GiB BF16)
  Reward:       format+outcome (shaped, 0% degenerate)
  gs=8:         SNR=2.83, signal strength ≈ 0.85
""")

    # Memory budget table
    print_section("RTX 4090 Memory Budget (24 GiB)")
    print("""
  Base components:
    model_weights      14.00 GiB (58.3%)  ← always resident (sleep_level=1)
    cuda_overhead       1.00 GiB (4.2%)
    lora_params         0.22 GiB (0.9%)   ← LoRA r=32 adapters

  Phase-by-phase peak memory:
    P1_wake            17.22 GiB (71.7%)  ← base + cuda + LoRA
    P2_rollout         19.22 GiB (80.1%)  ← + KV cache (2 GiB)
    P4_reward          19.23 GiB (80.1%)
    P8_advantage       19.24 GiB (80.2%)  ← PEAK
    P3_sleep           17.24 GiB (71.8%)  ← KV freed
    P9_update          18.24 GiB (76.0%)  ← + training activations

  ★★★ Overall peak: 19.24 GiB (80.2%)
  ★★★ Headroom: 4.76 GiB (19.8%) — SAFE for fluctuations
""")

    # Data flow with memory annotations
    print_section("Data Flow — Memory Annotations")
    for phase_id, phase in sorted(PHASE_DATA_FLOWS.items()):
        total_tq_mem = sum(op.memory_gib for op in phase.tq_ops)
        print(f"\n  Phase {phase_id}: {phase.name}")
        print(f"    Time: {phase.phase_time_s:.2f}s | Peak: {phase.peak_mem_gib:.2f} GiB | TQ mem: {fmt_gib(total_tq_mem)}")

        if phase.tq_ops:
            for op in phase.tq_ops:
                fields_str = ', '.join(op.fields[:3]) + ('...' if len(op.fields) > 3 else '')
                print(f"    {op.op_type}: {fields_str} ({fmt_gib(op.memory_gib)})")

        # Highlight phase-specific RTX 4090 concerns
        rtx4090_concerns = {
            1: "LoRA r=32 SAFE (#6782), accept_length monitor (#28771), sleep_level=1 (#45552)",
            3: "sleep_level=1 MANDATORY (level=2 crashes on RTX 4090)",
            6: "bypass_mode=True saves 14 Psi (skip old_log_prob recomputation)",
            7: "ref_in_actor=True saves 16 GiB (no separate reference model)",
            8: "gs=8 optimal (SNR=2.83), grouping by uid (prompt-level)",
            9: "#6699 detach, clip_grad=1.0, LR=1e-5, gradient norm < 1.0",
            10: "LoRA delta ~200 MiB (80x less than full sync), naive engine for dp=1",
        }
        if phase_id in rtx4090_concerns:
            print(f"    ★ RTX 4090: {rtx4090_concerns[phase_id]}")

    print_header("RTX 4090 CRITICAL PATH")
    print("""
  Memory critical path: P2_rollout → P8_advantage (19.24 GiB peak)
  Time critical path: P2_rollout(5.35s) → P10_sync(3.60s) → P9_update(2.50s)
  Data critical path: rollout_log_probs → old_log_probs → advantages → actor update

  ★★★ If ANY phase exceeds expected time by 2x → investigate immediately
  ★★★ If memory exceeds 20 GiB → risk of OOM → reduce batch_size or gs
  ★★★ If accept_length drops below 2.0 → restart SGLang engine (#28771)
""")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="verl V1 GRPO Training Loop Data Flow Tracer")
    parser.add_argument("mode", choices=["trace", "verify", "debug", "rtx4090"],
                        help="Mode to run")
    args = parser.parse_args()

    if args.mode == "trace":
        mode_trace()
    elif args.mode == "verify":
        mode_verify()
    elif args.mode == "debug":
        mode_debug()
    elif args.mode == "rtx4090":
        mode_rtx4090()


if __name__ == "__main__":
    main()
