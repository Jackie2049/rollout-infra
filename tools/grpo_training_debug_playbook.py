#!/usr/bin/env python3
"""
verl V1 GRPO Training Debug Playbook

Comprehensive debug guide combining knowledge from:
- DeepSpeed ZeRO internals (gradient flow, stream safety, #8061/#8080)
- verl V1 architecture (10-phase pipeline, TransferQueue, sleep/wake)
- CUDA stream safety pattern family (8 members, 7 pattern classes)
- Training loop data flow analysis (TQ operations, tensor shapes)

Modes:
  playbook  — Complete debug guide with all symptoms, causes, and fixes
  symptoms  — Symptom → root cause → diagnostic commands → fix mapping
  fixes     — Fix recipes for each tracked issue (ready-to-apply)
  rtx4090   — RTX 4090 specific debug procedures and monitoring
"""

import argparse
import sys


# ─── Symptom Database ───────────────────────────────────────────────────────

SYMPTOMS = [
    {
        "symptom": "NaN in training loss",
        "severity": "CRITICAL",
        "possible_causes": [
            ("#8061 overlap_comm stream race", "HIGH", "gradient data corrupted by multi-stream race → NaN propagates through optimizer"),
            ("#8068 gradient clipping=0", "HIGH", "DeepSpeed v0.19.2 default change → no clipping → gradient explosion → NaN"),
            ("#5394 Muon clipping stalls", "MEDIUM", "global grad-norm clipping orthogonalizes Muon → NaN after enough steps"),
            ("FSDP2 CPU memory leak (#6468)", "MEDIUM", "host OOM kills worker → NaN in surviving workers' data"),
            ("DSV4 CUDA graph (#45309)", "HIGH", "dynamic routing violates CUDA graph static batch assumption → garbage → NaN"),
        ],
        "diagnostic_commands": [
            "torch.cuda.synchronize() before/after gradient reduction",
            "torch.distributed.all_reduce(gradient_norm) across workers",
            "torch.autograd.detect_anomaly() to find NaN source",
            "Check DeepSpeed config: gradient_clipping value",
            "Check overlap_comm setting (dp=1: MUST be False)",
        ],
        "fixes": [
            "overlap_comm=False on dp=1 (avoid stream race entirely)",
            "gradient_clipping=1.0 (prevent explosion)",
            "enforce_eager=True for DSV4 models (avoid CUDA graph)",
            "Use FSDP1 instead of FSDP2 (avoid CPU memory leak)",
            "NOT use Muon optimizer (stalls under global clipping)",
        ],
    },
    {
        "symptom": "OOM (Out of Memory) crash",
        "severity": "CRITICAL",
        "possible_causes": [
            ("Full param weight sync", "HIGH", "67 GiB peak > 24 GiB on RTX 4090"),
            ("Separate reference model", "HIGH", "32 GiB (actor+ref) > 24 GiB on RTX 4090"),
            ("PPO-clip", "HIGH", "65 GiB peak > 24 GiB on RTX 4090"),
            ("ZeRO-3 + LoRA dtype mismatch (#8072)", "MEDIUM", "TypeError triggers before OOM but blocks training"),
            ("#6794-CRITICAL-3 big_values", "MEDIUM", "delta sync concat allocates full model size → OOM"),
            ("No gradient checkpointing", "LOW", "activations accumulate per micro-batch → OOM (#6699)"),
        ],
        "diagnostic_commands": [
            "torch.cuda.memory_allocated() at each phase transition",
            "torch.cuda.max_memory_allocated() after step",
            "Compare peak to GPU VRAM budget",
            "Check LoRA rank (r=32 safe, r>=64 broken + more memory)",
            "Check reference_mode (separate = OOM on RTX 4090)",
            "Check optimizer (cpu_adam = 0 GPU, gpu_adam = massive)",
        ],
        "fixes": [
            "LoRA r=32 + bypass + ref_in_actor (peak 19.24 GiB, fits 24 GiB)",
            "cpu_adam optimizer (0 GiB GPU for optimizer states)",
            "sleep_level=1 (only KV freed, base weights resident)",
            "naive checkpoint engine (no NCCL process group overhead)",
            "gradient checkpointing + #6699 detach fix",
            "NEVER use PPO-clip on RTX 4090 (65 GiB > 24 GiB)",
        ],
    },
    {
        "symptom": "Training doesn't converge (reward stays flat)",
        "severity": "HIGH",
        "possible_causes": [
            ("gs=1 REINFORCE degeneration (#605)", "CRITICAL", "advantage = raw reward, no variance reduction → 12x slower convergence"),
            ("Outcome-only 0/1 reward + small gs", "HIGH", ">30% degenerate groups → zero gradient signal in those groups"),
            ("gradient_clipping=0 (DeepSpeed default)", "MEDIUM", "no clipping → large updates overshoot → reward oscillates"),
            ("LoRA rank too small (r=4)", "LOW", "insufficient expressiveness for alignment task"),
        ],
        "diagnostic_commands": [
            "Check group_size config (MUST be >= 4)",
            "Compute advantage statistics: mean≈0, std≈1 per group",
            "Check degenerate group fraction (should be < 5%)",
            "Check gradient norm after clipping (should be < 1.0)",
            "Check LoRA rank (r=32 recommended, r=4 too small)",
        ],
        "fixes": [
            "gs=8 (SNR=2.83, optimal balance)",
            "Shaped reward (format+outcome, 0% degenerate groups)",
            "clip_grad=1.0 (NaN protection + signal preservation)",
            "LoRA r=32 (1.56% params, effective 15.6% with 10x LR)",
            "LR=1e-5 (safe update size for LoRA)",
        ],
    },
    {
        "symptom": "vLLM/SGLang rollout crashes during training",
        "severity": "CRITICAL",
        "possible_causes": [
            ("#45552 cumem stream sync crash", "CRITICAL", "sleep_level=2 → CUDART illegal-memory crash within first few steps on RTX 4090"),
            ("#6782 LoRA EOS bug", "CRITICAL", "rank>=64 → never emits EOS → all responses truncated → no valid completions"),
            ("#8061 overlap_comm crash", "HIGH", "gradient stream race → NaN → worker death → rollout orphaned"),
            ("#28771 EAGLE degradation", "HIGH", "accept_length drops → throughput loss → eventually may timeout"),
        ],
        "diagnostic_commands": [
            "Check sleep_level (1 safe, 2 crashes on RTX 4090)",
            "Check LoRA rank (32 safe, >=64 broken)",
            "Monitor accept_length metric (should be > 2.0)",
            "Check response_mask covers EOS token",
            "Check rollout server health (HTTP /health endpoint)",
        ],
        "fixes": [
            "sleep_level=1 only on RTX 4090 (#45552 avoidance)",
            "LoRA r=32 only (#6782 avoidance)",
            "Monitor accept_length, restart if < 2.0 (#28771 mitigation)",
            "SGLang preferred over vLLM (prefix caching + tag-based sleep/wake)",
            "enforce_eager=True (avoid CUDA graph issues)",
        ],
    },
    {
        "symptom": "Silent weight corruption (training produces wrong results)",
        "severity": "CRITICAL (hard to detect)",
        "possible_causes": [
            ("#6794-CRITICAL-1 record_stream", "CRITICAL", "d2h_stream.copy_() without record_stream → allocator reclaims → silent corruption"),
            ("#8061 overlap_comm stream race", "HIGH", "gradient data race → not always NaN, sometimes just wrong values"),
            ("#28676 MoE cache clobber", "MEDIUM", "stale expert weights after reload → wrong inference → wrong reward → wrong training"),
        ],
        "diagnostic_commands": [
            "Compare LoRA weights before/after sync (should match)",
            "Log weight hash/checksum at each sync",
            "Compare inference outputs before/after weight update",
            "Check rollout_log_probs consistency across steps",
        ],
        "fixes": [
            "Add record_stream(current_stream) after every d2h_stream.copy_() (#6794 fix)",
            "overlap_comm=False on dp=1 (#8061 avoidance)",
            "Cache invalidation after weight update (#28676 fix)",
            "Monitor weight sync timing — unexpected slowdown = potential corruption",
            "If corruption detected: restart from last checkpoint",
        ],
    },
    {
        "symptom": "Training throughput degrades over time",
        "severity": "HIGH",
        "possible_causes": [
            ("#28771 EAGLE accept_length degradation", "CRITICAL", "accept_length 3.4→1.9 over 2 hours = 44% throughput loss"),
            ("#6468 FSDP2 CPU memory leak", "HIGH", "host RAM grows → system swapping → throughput degradation"),
            ("KV cache fragmentation", "MEDIUM", "SGLang/vLLM KV cache fragmentation → reduced effective capacity"),
        ],
        "diagnostic_commands": [
            "Monitor accept_length metric (declining = #28771)",
            "Monitor host RAM usage (growing = FSDP2 leak)",
            "Monitor GPU memory pattern (growing = fragmentation or leak)",
            "Check steps/hr timing (declining = throughput loss)",
        ],
        "fixes": [
            "Monitor accept_length, restart SGLang if < 2.0 (#28771)",
            "Use FSDP1 instead of FSDP2 (#6468 avoidance)",
            "Periodic KV cache cleanup (aggressive_empty_cache)",
            "Log throughput at regular intervals to detect degradation",
        ],
    },
]


# ─── Fix Recipes ────────────────────────────────────────────────────────────

FIX_RECIPES = [
    {
        "issue": "#45552 — vLLM cumem stream sync crash",
        "severity": "★★★ CRITICAL (RTX 4090 BLOCKER)",
        "fix_type": "Config avoidance",
        "fix": "Set sleep_level=1 instead of sleep_level=2",
        "why_it_works": "sleep_level=1 only frees KV cache (LoRA offload), NOT weights. CuMemAllocator (which has the bug) is only used for weight offload in sleep_level=2. sleep_level=1 uses LoRA adapter path which doesn't trigger CuMemAllocator",
        "code_changes": "Config: rollout.sleep_level=1 (or use LoRA merge=False which auto-sets sleep_level=1)",
        "verification": "Check that sleep/wake completes without crash in first 10 steps. Monitor GPU memory: should drop ~2 GiB after sleep, return after wake",
    },
    {
        "issue": "#8061 — DeepSpeed overlap_comm stream race → NaN",
        "severity": "★★★ CRITICAL",
        "fix_type": "Config avoidance",
        "fix": "Set overlap_comm=False on dp=1",
        "why_it_works": "On dp=1, reduce-scatter is identity (no data movement). overlap_comm uses a separate reduction_stream that creates stream race risk. With overlap_comm=False, all gradient operations stay on default stream → no race",
        "code_changes": "Config: zero_optimization.overlap_comm=False",
        "verification": "Check that training loss decreases normally (no NaN). On dp>1: wait for #8080 merge then overlap_comm can be safely enabled",
    },
    {
        "issue": "#6782 — verl LoRA rank>=64 never emits EOS",
        "severity": "★★★ CRITICAL",
        "fix_type": "Config avoidance",
        "fix": "Use LoRA rank=32, alpha=64",
        "why_it_works": "rank=32/alpha=64 = scaling factor 2.0, same as rank=64/alpha=128. But rank=32 produces smaller LoRA delta that doesn't distort the logit distribution enough to suppress EOS. Empirically verified working",
        "code_changes": "Config: actor_rollout_ref.model.lora.rank=32, actor_rollout_ref.model.lora.lora_alpha=64",
        "verification": "Check that responses end with EOS token (not truncated at max_response_length). Check response_mask covers EOS position",
    },
    {
        "issue": "#6794-CRITICAL-1 — verl delta sync missing record_stream",
        "severity": "★★★ CRITICAL (silent corruption)",
        "fix_type": "Code fix needed",
        "fix": "Add record_stream(current_stream) after every d2h_stream.copy_() call",
        "why_it_works": "record_stream tells PyTorch allocator that the tensor is also being used on current_stream, preventing it from reclaiming the memory before the side stream completes",
        "code_changes": "In delta snapshot code: after d2h_stream.copy_(dst, src), add src.record_stream(torch.cuda.current_stream())",
        "verification": "Compare LoRA delta weights before/after sync (should be identical). Log weight checksums at each step. Run 100+ steps without corruption",
    },
    {
        "issue": "#8072/#8076 — ZeRO-3 + LoRA dtype mismatch",
        "severity": "★★ HIGH",
        "fix_type": "Config avoidance",
        "fix": "Use ZeRO-2 or FSDP1 instead of ZeRO-3",
        "why_it_works": "ZeRO-2 doesn't partition parameters (no dtype mismatch). FSDP1 handles sharding correctly with mixed precision",
        "code_changes": "Config: zero_optimization.stage=2 (or strategy=fsdp1 with verl)",
        "verification": "Check that training starts without TypeError. Check dtype consistency across parameters",
    },
    {
        "issue": "#28771 — SGLang EAGLE accept_length degradation",
        "severity": "★★★ CRITICAL (44% throughput loss)",
        "fix_type": "Monitoring + restart mitigation",
        "fix": "Monitor accept_length, restart SGLang engine if accept_length < 2.0",
        "why_it_works": "Restarting clears HiCache state that may have accumulated stale/partial KV transfers. Fresh start restores accept_length to ~3.4",
        "code_changes": "Add monitoring: if sglang_metrics.accept_length < 2.0: restart_engine(). Also consider disabling EAGLE spec decode entirely for stability",
        "verification": "Track accept_length over time. Should stay > 2.5 with periodic restarts. Without restart: expect gradual decline to 1.9",
    },
    {
        "issue": "gs=1 REINFORCE degeneration",
        "severity": "★★★ CRITICAL (training doesn't converge)",
        "fix_type": "Config fix",
        "fix": "Set group_size >= 4 (recommended: 8)",
        "why_it_works": "gs>=4 provides variance reduction (75% at gs=4, 87.5% at gs=8). gs=1 gives NO variance reduction → advantage = raw reward = REINFORCE(baseline=0)",
        "code_changes": "Config: rollout.n=8 (verl group_size), or group_size=8",
        "verification": "Check advantage statistics: mean≈0, std≈1 per group. Check degenerate group fraction < 5%. Check reward improving over steps",
    },
    {
        "issue": "#8068 — DeepSpeed gradient clipping=0 default",
        "severity": "★★ HIGH",
        "fix_type": "Config fix",
        "fix": "Set gradient_clipping=1.0",
        "why_it_works": "clip_grad=1.0 prevents gradient explosion while preserving signal. LoRA gradient flow: 17x stronger than PPO → 1.0 threshold is optimal",
        "code_changes": "Config: gradient_clipping=1.0",
        "verification": "Monitor gradient norm after clipping (should be < 1.0). Check training loss decreases smoothly without NaN",
    },
]


# ─── RTX 4090 Debug Procedures ─────────────────────────────────────────────

RTX4090_PROCEDURES = {
    "pre_flight": {
        "name": "Pre-flight Checklist",
        "steps": [
            "1. ulimit -n 65535 (increase file descriptor limit)",
            "2. nvidia-smi (verify GPU visible, check VRAM)",
            "3. python -c 'import torch; print(torch.cuda.get_device_name())' (verify CUDA)",
            "4. conda env check: python, torch, verl, sglang versions",
            "5. Config validation: run grpo_config_validator.py rtx4090",
            "6. Memory planner: run grpo_memory_planner.py plan",
            "7. Data flow: run verl_v1_grpo_data_flow_tracer.py verify",
        ],
    },
    "nan_debug": {
        "name": "NaN Debug Procedure",
        "steps": [
            "1. Add torch.autograd.detect_anomaly() to detect NaN source",
            "2. Check overlap_comm=False (dp=1 MUST)",
            "3. Check gradient_clipping=1.0 (NOT default 0)",
            "4. Check enforce_eager=True (DSV4 models)",
            "5. Check FSDP1 (NOT FSDP2)",
            "6. Check LoRA r=32 (NOT r>=64)",
            "7. If NaN persists: add torch.cuda.synchronize() at phase boundaries",
            "8. If NaN in gradients: check #8061 pattern (stream race)",
            "9. If NaN in weights: check #6794 pattern (record_stream corruption)",
        ],
    },
    "oom_debug": {
        "name": "OOM Debug Procedure",
        "steps": [
            "1. torch.cuda.max_memory_allocated() → compare to 24 GiB budget",
            "2. Check peak phase: rollout (19.24 GiB) or training (18.24 GiB)",
            "3. If rollout OOM: reduce batch_size or group_size",
            "4. If training OOM: check reference_mode (MUST bypass)",
            "5. Check optimizer: MUST cpu_adam (0 GiB GPU)",
            "6. Check sleep_level: MUST 1 (level=2 crashes anyway)",
            "7. Check LoRA rank: r=32 (NOT full param)",
            "8. Run grpo_memory_planner.py plan for alternative configs",
        ],
    },
    "convergence_debug": {
        "name": "Convergence Debug Procedure",
        "steps": [
            "1. Check group_size: MUST >= 4 (gs=1 = REINFORCE degeneration)",
            "2. Check advantage stats: mean≈0, std≈1 per group",
            "3. Check reward function: shaped (format+outcome) vs outcome-only",
            "4. Check degenerate groups: should be < 5%",
            "5. Check gradient_clipping=1.0 (NOT 0 or too large)",
            "6. Check LR=1e-5 (NOT too large for LoRA)",
            "7. Check LoRA rank=32 (NOT too small)",
            "8. If reward stuck: check rollout quality (accept_length > 2.0)",
            "9. If reward oscillates: check clip_grad, LR stability",
        ],
    },
    "monitoring": {
        "name": "Continuous Monitoring",
        "steps": [
            "1. Log phase timestamps at each step (detect slow phases)",
            "2. torch.cuda.memory_allocated() at phase transitions",
            "3. SGLang accept_length metric (restart if < 2.0)",
            "4. Gradient norm after clipping (should be < 1.0)",
            "5. Weight sync timing (should be < 4s for LoRA delta)",
            "6. Host RAM usage (detect FSDP2 leak if applicable)",
            "7. Reward trajectory (should improve over 100+ steps)",
            "8. LoRA delta weight checksum (detect #6794 corruption)",
        ],
    },
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


def mode_playbook():
    """Show complete debug guide."""
    print_header("MODE: playbook — verl V1 GRPO Training Debug Playbook")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  verl V1 GRPO Training Debug Playbook                         ║
║  Based on: ZeRO internals + verl V1 architecture +             ║
║  CUDA stream safety (8 members) + data flow analysis            ║
╚══════════════════════════════════════════════════════════════════╝

  Symptoms tracked: {len(SYMPTOMS)}
  Fix recipes: {len(FIX_RECIPES)}
  RTX 4090 procedures: {len(RTX4090_PROCEDURES)}
""")

    for i, s in enumerate(SYMPTOMS, 1):
        print_section(f"Symptom {i}: {s['symptom']} [{s['severity']}]")

        print(f"\n  Possible causes:")
        for cause, prob, desc in s['possible_causes']:
            symbol = "★★★" if prob == "CRITICAL" else "★★" if prob == "HIGH" else "★"
            print(f"    {symbol} {cause} ({prob}): {desc}")

        print(f"\n  Diagnostic commands:")
        for cmd in s['diagnostic_commands']:
            print(f"    → {cmd}")

        print(f"\n  Fixes:")
        for fix in s['fixes']:
            print(f"    ✓ {fix}")

    print_section("Fix Recipes (ready-to-apply)")
    for i, recipe in enumerate(FIX_RECIPES, 1):
        print(f"\n  Recipe #{i}: {recipe['issue']}")
        print(f"    Severity: {recipe['severity']}")
        print(f"    Fix type: {recipe['fix_type']}")
        print(f"    Fix: {recipe['fix']}")
        print(f"    Why: {recipe['why_it_works']}")
        print(f"    Code: {recipe['code_changes']}")
        print(f"    Verify: {recipe['verification']}")


def mode_symptoms():
    """Symptom → root cause mapping."""
    print_header("MODE: symptoms — Symptom → Root Cause Mapping")

    print(f"\n  {'Symptom':<30} {'Most Likely Cause':<35} {'Quick Fix':<30}")
    print("-" * 95)

    quick_fixes = {
        "NaN in training loss": ("#8061 overlap_comm race", "overlap_comm=False"),
        "OOM crash": ("Full param/ref model", "LoRA+bypass+ref_in_actor"),
        "Training doesn't converge": ("gs=1 REINFORCE", "gs=8"),
        "Rollout crashes": ("#45552 cumem crash", "sleep_level=1"),
        "Silent weight corruption": ("#6794 record_stream", "monitor checksums"),
        "Throughput degrades": ("#28771 EAGLE degrade", "restart if < 2.0"),
    }

    for symptom, (cause, fix) in quick_fixes.items():
        print(f"  {symptom:<30} {cause:<35} {fix:<30}")

    print_section("Detailed Symptom → Root Cause → Fix Chain")

    for symptom_data in SYMPTOMS:
        print(f"\n  ★★★ {symptom_data['symptom']} [{symptom_data['severity']}]")

        # Most likely cause first
        causes_sorted = sorted(symptom_data['possible_causes'], key=lambda x: -1 if x[1] == "CRITICAL" else 0 if x[1] == "HIGH" else -2)
        for cause, prob, desc in causes_sorted[:3]:
            print(f"    {prob}: {cause}")
            print(f"      → {desc}")

        # Primary fix
        print(f"    Primary fix: {symptom_data['fixes'][0]}")


def mode_fixes():
    """Fix recipes for each tracked issue."""
    print_header("MODE: fixes — Fix Recipes (Ready-to-Apply)")

    config_avoidances = [r for r in FIX_RECIPES if r['fix_type'] == "Config avoidance"]
    config_fixes = [r for r in FIX_RECIPES if r['fix_type'] == "Config fix"]
    code_fixes = [r for r in FIX_RECIPES if r['fix_type'] == "Code fix needed"]
    monitoring = [r for r in FIX_RECIPES if r['fix_type'] == "Monitoring + restart mitigation"]

    print_section(f"Config Avoidances ({len(config_avoidances)} — safest, no code changes)")
    for recipe in config_avoidances:
        print(f"\n  ★★★ {recipe['issue']}")
        print(f"    Fix: {recipe['fix']}")
        print(f"    Config: {recipe['code_changes']}")
        print(f"    Verify: {recipe['verification']}")

    print_section(f"Config Fixes ({len(config_fixes)} — change config values)")
    for recipe in config_fixes:
        print(f"\n  ★★ {recipe['issue']}")
        print(f"    Fix: {recipe['fix']}")
        print(f"    Config: {recipe['code_changes']}")
        print(f"    Verify: {recipe['verification']}")

    print_section(f"Code Fixes ({len(code_fixes)} — need code changes)")
    for recipe in code_fixes:
        print(f"\n  ★★★ {recipe['issue']}")
        print(f"    Fix: {recipe['fix']}")
        print(f"    Why: {recipe['why_it_works']}")
        print(f"    Code: {recipe['code_changes']}")
        print(f"    Verify: {recipe['verification']}")

    print_section(f"Monitoring Mitigations ({len(monitoring)} — detect and respond)")
    for recipe in monitoring:
        print(f"\n  ★★★ {recipe['issue']}")
        print(f"    Fix: {recipe['fix']}")
        print(f"    Code: {recipe['code_changes']}")
        print(f"    Verify: {recipe['verification']}")


def mode_rtx4090():
    """RTX 4090 specific debug procedures."""
    print_header("MODE: rtx4090 — RTX 4090 GRPO Debug Procedures")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ★★★ RTX 4090 (24 GiB) GRPO Debug Procedures ★★★             ║
║  Optimal config avoids 5 of 8 CUDA stream safety bugs          ║
║  2 bugs still at risk: #6794 (silent), #28771 (gradual)        ║
╚══════════════════════════════════════════════════════════════════╝
""")

    for proc_key, proc in RTX4090_PROCEDURES.items():
        print_section(proc['name'])
        for step in proc['steps']:
            print(f"  {step}")

    print_section("RTX 4090 Bug Avoidance Summary")
    print("""
  ★★★ AVOIDED by optimal config:
    #45552 (cumem crash)     → sleep_level=1 (LoRA offload, NOT CuMemAllocator)
    #8061 (stream race)      → overlap_comm=False (no reduction_stream)
    #8072 (ZeRO-3 dtype)     → FSDP1 (no parameter partitioning)
    #6782 (LoRA EOS)         → LoRA r=32 (NOT r>=64)
    #5394 (Muon clipping)    → cpu_adam (NOT Muon)

  ★★★ STILL AT RISK (monitoring required):
    #6794 (record_stream)    → ★★★ SILENT CORRUPTION — monitor weight checksums
    #28771 (EAGLE degrade)   → ★★★ 44% THROUGHPUT LOSS — monitor accept_length < 2.0
    #28676 (MoE cache)       → HIGH for MoE models — cache invalidation

  ★★★ Monitoring dashboard:
    1. LoRA delta checksum: hash before/after sync → detect #6794
    2. accept_length: track over time → restart if < 2.0 (#28771)
    3. GPU memory: torch.cuda.memory_allocated() at phases → detect leaks
    4. Gradient norm: after clipping → should be < 1.0
    5. Reward trajectory: should improve monotonically
    6. Phase timing: log all phases → detect slow phases

  ★★★ If any monitoring indicator triggers:
    1. Log the anomaly with timestamps
    2. Check which bug pattern matches the symptom
    3. Apply the corresponding fix recipe
    4. If no fix available: restart training from last checkpoint
    5. Continue monitoring after restart
""")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="verl V1 GRPO Training Debug Playbook")
    parser.add_argument("mode", choices=["playbook", "symptoms", "fixes", "rtx4090"],
                        help="Mode to run")
    args = parser.parse_args()

    if args.mode == "playbook":
        mode_playbook()
    elif args.mode == "symptoms":
        mode_symptoms()
    elif args.mode == "fixes":
        mode_fixes()
    elif args.mode == "rtx4090":
        mode_rtx4090()


if __name__ == "__main__":
    main()
