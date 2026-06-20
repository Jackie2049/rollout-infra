#!/usr/bin/env python3
"""
CUDA Stream Safety Pattern Family Synthesis

Maps the cross-framework CUDA stream safety pattern family — the systemic bug
where multi-stream GPU operations miss synchronization with all producer streams.

Based on deep readings: DeepSpeed ZeRO (#8061/#8080), verl V1 (#6794),
vLLM (#45552), SGLang (#28676/#28771), DeepSpeed ZeRO-3 (#8072).

Modes:
  map       - Show complete pattern family map with all members
  analyze   - Deep analysis of pattern root cause and fix strategies
  compare   - Compare fix approaches across frameworks
  rtx4090   - RTX 4090 specific implications and mitigations
"""

import argparse
import sys


# ─── Pattern Family Members ────────────────────────────────────────────────

PATTERN_MEMBERS = [
    {
        "id": 1,
        "framework": "DeepSpeed",
        "issue": "#8061",
        "title": "IPG bucket average_tensor stream race → NaN",
        "severity": "CRITICAL",
        "status": "FIX PR #8080 pending review",
        "root_cause": "average_tensor() only synchronizes with current_stream, misses other producer streams that wrote gradient data via torch.compile multi-stream kernels",
        "fix": "IPGBucket.copy_streams: set tracks ALL producer streams; average_tensor() iterates and wait_stream() on each",
        "fix_lines": "+19/-1 code, +72/-0 tests",
        "fix_contributor": "arunshar (external)",
        "merge_probability": "8/10",
        "merge_timeline": "2-6 weeks",
        "rtx4090_impact": "overlap_comm pointless on dp=1 (no reduce-scatter), but safe for dp>1 after merge. overlap_comm=False already safe",
        "pattern_class": "gradient_accumulation → multi_stream_producer → read_before_sync",
        "source_file": "deepspeed/runtime/zero/partitioned_param_coordinator.py",
        "source_lines": "average_tensor() method",
        "reading_note": "notebook/projects/deepspeed-8061-overlap-comm-multi-stream-race-reading.md",
    },
    {
        "id": 2,
        "framework": "DeepSpeed",
        "issue": "#8080",
        "title": "Fix for #8061 — copy_streams set for multi-stream safety",
        "severity": "HIGH (fix PR)",
        "status": "OPEN (awaiting review)",
        "root_cause": "Fix approach: adds copy_streams set to IPGBucket to track all producer streams that wrote into the bucket buffer",
        "fix": "N/A (this IS the fix)",
        "fix_lines": "+19/-1, +72/-0 tests",
        "fix_contributor": "arunshar",
        "merge_probability": "8/10",
        "merge_timeline": "2-6 weeks",
        "rtx4090_impact": "Makes overlap_comm safe for dp>1. overlap_comm still pointless on dp=1",
        "pattern_class": "fix_pattern: track_all_producers → wait_stream_each → safe_read",
        "source_file": "deepspeed/runtime/zero/partitioned_param_coordinator.py",
        "source_lines": "IPGBucket.copy_streams addition + average_tensor() loop",
        "reading_note": "notebook/projects/deepspeed-8080-cuda-stream-race-fix-reading.md",
    },
    {
        "id": 3,
        "framework": "verl",
        "issue": "#6794-CRITICAL-1",
        "title": "Delta weight snapshot missing record_stream → silent data corruption",
        "severity": "CRITICAL",
        "status": "UNFIXED (open in PR review)",
        "root_cause": "d2h_stream.copy_() for delta weight snapshot doesn't call record_stream(current_stream) → PyTorch allocator may reclaim tensor before d2h_stream completes",
        "fix": "Add record_stream(current_stream) after every d2h_stream.copy_() call",
        "fix_lines": "+2 lines",
        "fix_contributor": "identified by reviewer, not yet implemented",
        "merge_probability": "7/10 (when implemented)",
        "merge_timeline": "unknown (still open)",
        "rtx4090_impact": "CRITICAL — silent corruption in weight sync → training produces wrong results without any error. Hard to detect. LoRA delta path affected",
        "pattern_class": "async_copy_on_side_stream → missing_record_stream → allocator_reclaim_before_completion",
        "source_file": "verl/checkpoint_engine/ (delta snapshot logic)",
        "source_lines": "d2h_stream.copy_() calls",
        "reading_note": "notebook/projects/verl-6794-delta-weight-sync-deep-reading.md",
    },
    {
        "id": 4,
        "framework": "vLLM",
        "issue": "#45552",
        "title": "CuMemAllocator sleep/wake missing torch.cuda.synchronize() → CUDART illegal-memory crash",
        "severity": "CRITICAL (RTX 4090 BLOCKER)",
        "status": "FIX available (2-line synchronize addition) but not merged",
        "root_cause": "CuMemAllocator.release() and CuMemAllocator.resume() don't call torch.cuda.synchronize() → operations on weight tensors may still be in-flight on other streams when memory is freed/resumed",
        "fix": "Add torch.cuda.synchronize() after CuMemAllocator.release() and before CuMemAllocator.resume()",
        "fix_lines": "+2 lines",
        "fix_contributor": "identified in issue comments",
        "merge_probability": "9/10",
        "merge_timeline": "1-3 weeks",
        "rtx4090_impact": "★★★ BLOCKER: sleep_level=2 crashes within first few training steps on RTX 4090. sleep_level=1 AVOIDS bug entirely (LoRA offload doesn't use CuMemAllocator)",
        "pattern_class": "weight_free/resume → missing_stream_sync → in_flight_ops_on_freed_memory",
        "source_file": "vllm/worker/worker.py (sleep/wake methods)",
        "source_lines": "CuMemAllocator.release() and resume()",
        "reading_note": "notebook/projects/vllm-45552-cumem-sleep-wake-stream-sync-reading.md",
    },
    {
        "id": 5,
        "framework": "DeepSpeed",
        "issue": "#8072/#8076",
        "title": "ZeRO-3 + PEFT LoRA weight sync dtype mismatch race",
        "severity": "HIGH",
        "status": "STALLED (0 maintainer comments for days)",
        "root_cause": "ZeRO-3 partition dtype mismatch between fp32 LoRA and bf16 base → per-tensor partition creates type conflict when params have different dtypes",
        "fix": "Use ZeRO-2 instead (no parameter partitioning, no dtype mismatch)",
        "fix_lines": "Config change only (strategy='zero2' instead of 'zero3')",
        "fix_contributor": "community workaround",
        "merge_probability": "3/10 (stalled)",
        "merge_timeline": "unknown",
        "rtx4090_impact": "ZeRO-3 unusable with LoRA on RTX 4090 → ZeRO-2 + FSDP1 required",
        "pattern_class": "dtype_partition_race → mismatch_in_partition_buffer → TypeError",
        "source_file": "deepspeed/runtime/zero/stage3.py",
        "source_lines": "partition_param_list() dtype handling",
        "reading_note": "notebook/projects/deepspeed-8072-8073-zero3-peft-regression-source-code-deep-reading.md",
    },
    {
        "id": 6,
        "framework": "SGLang",
        "issue": "#28676",
        "title": "MoE cache clobber during weight reload",
        "severity": "HIGH",
        "status": "UNFIXED",
        "root_cause": "MoE expert weights in cache may be overwritten during weight reload without proper synchronization → stale cache reads after weight update",
        "fix": "Add cache invalidation + torch.cuda.synchronize() after weight reload",
        "fix_lines": "Estimated ~5-10 lines",
        "fix_contributor": "Not yet implemented",
        "merge_probability": "5/10",
        "merge_timeline": "unknown",
        "rtx4090_impact": "HIGH for MoE models with SGLang rollout → cache reads stale after weight update → wrong inference results",
        "pattern_class": "weight_reload → cache_invalidations_missing → stale_reads_on_other_stream",
        "source_file": "sglang/srt/lora/lora_manager.py (MoE cache handling)",
        "source_lines": "update_weights() method",
        "reading_note": "notebook/projects/sglang-28676-moe-cache-clobber-comment-draft.md",
    },
    {
        "id": 7,
        "framework": "SGLang",
        "issue": "#28771",
        "title": "EAGLE accept_length degradation (HiCache async race with draft forward)",
        "severity": "CRITICAL",
        "status": "OPEN (0 comments, 0 triage)",
        "root_cause": "HiCache swaps KV pages between GPU/Host asynchronously — draft model may read partially-loaded pages during forward pass → subtle numerical imprecision → gradually reducing prediction accuracy",
        "fix": "Add layer_transfer_counter synchronization for draft model KV reads during HiCache swaps",
        "fix_lines": "Estimated ~10-20 lines",
        "fix_contributor": "Not yet implemented",
        "merge_probability": "5/10",
        "merge_timeline": "unknown",
        "rtx4090_impact": "★★★ CRITICAL: accept_length 3.4→1.9 over 2 hours = 44% throughput loss. GRPO rollout = 69.2% bottleneck → spec decode throughput directly impacts training",
        "pattern_class": "async_cache_swap → draft_model_read_before_sync → gradual_degradation_not_crash",
        "source_file": "sglang/srt/layers/radix_attention.py (HiCache)",
        "source_lines": "eagle_worker.py line 596: maybe_evict_swa()",
        "reading_note": "notebook/projects/sglang-28771-eagle-accept-length-degradation-reading.md",
    },
    {
        "id": 8,
        "framework": "DeepSpeed",
        "issue": "#8075",
        "title": "NVMe offload fd leak accumulates → process crash",
        "severity": "HIGH (latent)",
        "status": "STALLED",
        "root_cause": "NVMe offload creates file descriptors per swap operation → never properly closed → fd count grows → exhausts ulimit → process hangs/crashes",
        "fix": "Add proper fd close() after swap operation completes",
        "fix_lines": "+3 lines",
        "fix_contributor": "identified in issue, not yet implemented",
        "merge_probability": "4/10 (stalled)",
        "merge_timeline": "unknown",
        "rtx4090_impact": "HIGH for long-running training with NVMe offload — but MUST NOT use NVMe offload on RTX 4090 anyway (fd leak + unnecessary)",
        "pattern_class": "resource_leak → fd_not_closed → ulimit_exhaustion → process_crash",
        "source_file": "deepspeed/runtime/zero/offload_config.py",
        "source_lines": "swap operation fd handling",
        "reading_note": "notebook/projects/deepspeed-8075-fd-leak-reading.md",
    },
]


# ─── Pattern Analysis ──────────────────────────────────────────────────────

PATTERN_CLASSES = {
    "gradient_accumulation → multi_stream_producer → read_before_sync": {
        "description": "Gradient accumulation on multiple CUDA streams → producer streams write data into shared buffer → consumer reads before all producers complete → data race → NaN/corruption",
        "mathematical_model": "P(corruption) = P(allocator_reclaims or consumer_reads before producer_stream_completes) → near 1.0 for long-running training",
        "universal_fix": "Track ALL producer streams → wait_stream() on each before reading buffer",
        "frameworks_affected": ["DeepSpeed (#8061)"],
    },
    "async_copy_on_side_stream → missing_record_stream → allocator_reclaim_before_completion": {
        "description": "Data copied on side stream (d2h, h2d) without record_stream → PyTorch allocator assumes default stream → may reclaim tensor before side stream completes → silent corruption",
        "mathematical_model": "P(corruption) ≈ 1 for long-running training (eventually allocator reclaims during side stream operation)",
        "universal_fix": "Call record_stream(current_stream) after every async copy on side stream",
        "frameworks_affected": ["verl (#6794-CRITICAL-1)"],
    },
    "weight_free/resume → missing_stream_sync → in_flight_ops_on_freed_memory": {
        "description": "GPU memory freed/resumed without torch.cuda.synchronize() → operations on weights/KV still in-flight on other streams → crash (CUDART illegal-memory)",
        "mathematical_model": "P(crash) = P(any_in_flight_op_accesses_freed_memory) → near 1.0 within first few steps",
        "universal_fix": "torch.cuda.synchronize() after free, before resume",
        "frameworks_affected": ["vLLM (#45552)"],
    },
    "dtype_partition_race → mismatch_in_partition_buffer → TypeError": {
        "description": "ZeRO-3 partitions parameters by first param's dtype → but mixed dtype (fp32 LoRA + bf16 base) creates mismatch → TypeError crash",
        "mathematical_model": "P(crash) = 1.0 when LoRA rank > 0 + ZeRO-3 (always mismatch)",
        "universal_fix": "Use ZeRO-2 (no parameter partitioning) or FSDP1",
        "frameworks_affected": ["DeepSpeed (#8072/#8076)"],
    },
    "weight_reload → cache_invalidations_missing → stale_reads_on_other_stream": {
        "description": "Model weights updated but caches (MoE expert cache, KV cache) not invalidated → stale reads from other streams → wrong inference results",
        "mathematical_model": "P(stale_read) = P(cache_access_after_weight_update_without_invalidation) → 1.0 for first few inference calls after update",
        "universal_fix": "Cache invalidation + synchronize() after weight update, before next inference",
        "frameworks_affected": ["SGLang (#28676)"],
    },
    "async_cache_swap → draft_model_read_before_sync → gradual_degradation_not_crash": {
        "description": "HiCache asynchronously swaps KV pages between GPU/Host → draft model reads partially-loaded pages → subtle numerical imprecision → gradual throughput degradation (NOT crash)",
        "mathematical_model": "P(degradation) accumulates: accept_length(t) ≈ accept_length(0) - k*t where k depends on cache pressure. Correlation: ρ(accept_length, token_usage) ≈ -0.99",
        "universal_fix": "Synchronize layer transfer counter before draft model forward pass",
        "frameworks_affected": ["SGLang (#28771)"],
    },
    "resource_leak → fd_not_closed → ulimit_exhaustion → process_crash": {
        "description": "File descriptors opened per swap/IO operation → never closed → fd count grows monotonically → ulimit reached → process hangs/crashes",
        "mathematical_model": "fd_count(t) = fd_count(0) + leak_rate * t → crash when fd_count(t) > ulimit (default 1024)",
        "universal_fix": "Close file descriptors after operation completes, set ulimit 65535",
        "frameworks_affected": ["DeepSpeed (#8075)"],
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


def mode_map():
    """Show complete pattern family map."""
    print_header("MODE: map — CUDA Stream Safety Pattern Family Map")

    print(f"\n  Pattern Family: CUDA Stream Safety / State Lifecycle Mismatch")
    print(f"  Members: {len(PATTERN_MEMBERS)} across {len(set(m['framework'] for m in PATTERN_MEMBERS))} frameworks")
    print(f"  Pattern Classes: {len(PATTERN_CLASSES)}")

    print_section("Members by Framework")
    for framework in sorted(set(m['framework'] for m in PATTERN_MEMBERS)):
        members = [m for m in PATTERN_MEMBERS if m['framework'] == framework]
        print(f"\n  {framework} ({len(members)} members):")
        for m in members:
            severity_symbol = "★★★" if m['severity'] == "CRITICAL" else "★★" if m['severity'] == "HIGH" else "★"
            print(f"    {severity_symbol} {m['issue']}: {m['title']}")
            print(f"      Severity: {m['severity']} | Status: {m['status']}")
            print(f"      Pattern: {m['pattern_class']}")
            print(f"      RTX 4090: {m['rtx4090_impact']}")

    print_section("Pattern Classes")
    for cls_name, cls_data in PATTERN_CLASSES.items():
        print(f"\n  Class: {cls_name}")
        print(f"    Description: {cls_data['description']}")
        print(f"    Math: {cls_data['mathematical_model']}")
        print(f"    Fix: {cls_data['universal_fix']}")
        print(f"    Affected: {', '.join(cls_data['frameworks_affected'])}")


def mode_analyze():
    """Deep analysis of pattern root cause and fix strategies."""
    print_header("MODE: analyze — CUDA Stream Safety Root Cause Analysis")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ROOT CAUSE: PyTorch CUDA Stream Memory Model                  ║
╚══════════════════════════════════════════════════════════════════╝

  PyTorch's CUDA caching allocator assumes DEFAULT STREAM as the
  primary execution stream. When tensors are created, modified, or
  freed on the default stream, the allocator can safely track their
  lifecycle.

  But when operations happen on OTHER streams (side streams, reduction
  streams, d2h/h2d streams, HiCache transfer streams), the allocator
  loses track. It may:
  1. Reclaim a tensor's memory while a side stream is still using it
  2. Assume an operation completed when it's still in-flight
  3. Free GPU memory while other streams still have pending ops

  The fix pattern has 3 levels:
""")

    fix_levels = [
        ("Level 1: record_stream (verl #6794)",
         "After every async copy on a side stream, call tensor.record_stream(current_stream)\n  This tells the allocator: 'I'm using this tensor on current_stream too, don't reclaim\n  until current_stream reaches this point'",
         "2 lines per copy operation"),
        ("Level 2: synchronize (vLLM #45552)",
         "After freeing GPU memory (sleep) or before resuming (wake), call torch.cuda.synchronize()\n  This ensures ALL streams have completed before memory is freed/resumed",
         "2 lines per free/resume operation"),
        ("Level 3: copy_streams set (DeepSpeed #8080)",
         "Track ALL producer streams that wrote into a shared buffer. Before reading the buffer,\n  iterate over ALL producer streams and call wait_stream() on each",
         "19 lines code + 72 lines tests"),
    ]

    for name, desc, effort in fix_levels:
        print_section(name)
        print(f"  {desc}")
        print(f"  Effort: {effort}")

    print_section("Why This Pattern is Systemic")
    print("""
  1. PyTorch's allocator is stream-agnostic by default → only tracks default stream
  2. Multi-stream GPU operations are increasingly common:
     - torch.compile generates multi-stream kernels
     - NCCL reduction runs on reduction_stream
     - d2h/h2d copies run on side streams
     - HiCache transfers run on async transfer streams
  3. All RL training frameworks use multi-stream operations:
     - DeepSpeed: overlap_comm reduction_stream
     - verl: d2h_stream for weight sync
     - vLLM: CuMemAllocator for sleep/wake
     - SGLang: HiCache async transfers
  4. The bug is SILENT in most cases (no crash, no NaN, just wrong data)
     - Only vLLM #45552 crashes (CUDART illegal-memory is loud)
     - verl #6794 is silent corruption (harder to detect)
     - SGLang #28771 is gradual degradation (hardest to detect)

  ★★★ Conclusion: CUDA stream safety is a systemic cross-framework problem
      that requires explicit synchronization at EVERY multi-stream boundary.
""")


def mode_compare():
    """Compare fix approaches across frameworks."""
    print_header("MODE: compare — Fix Approach Comparison")

    print(f"\n  {'Issue':<20} {'Framework':<12} {'Fix Type':<20} {'Effort':<15} {'Merge Prob':<10} {'RTX4090 Impact':<25}")
    print("-" * 105)

    fixes = [
        ("#8080 (fix)", "DeepSpeed", "copy_streams set", "+19/+72 tests", "8/10", "Safe for dp>1"),
        ("#6794-C1", "verl", "record_stream", "+2 lines", "7/10", "★★★ BLOCKER: silent corruption"),
        ("#45552", "vLLM", "synchronize()", "+2 lines", "9/10", "★★★ BLOCKER: crashes RTX 4090"),
        ("#8072/#8076", "DeepSpeed", "ZeRO-2 config", "Config change", "N/A", "ZeRO-3 unusable"),
        ("#28676", "SGLang", "cache_invalidation", "~5-10 lines", "5/10", "HIGH for MoE"),
        ("#28771", "SGLang", "sync_layer_transfer", "~10-20 lines", "5/10", "★★★ 44% throughput loss"),
        ("#8075", "DeepSpeed", "fd_close + ulimit", "+3 lines", "4/10", "MUST NOT NVMe offload"),
    ]

    for issue, fw, fix_type, effort, prob, impact in fixes:
        print(f"  {issue:<20} {fw:<12} {fix_type:<20} {effort:<15} {prob:<10} {impact:<25}")

    print_section("Fix Strategy Comparison")
    print("""
  | Fix Type          | Applicable When            | Pros                    | Cons                    |
  |-------------------|---------------------------|-------------------------|-------------------------|
  | record_stream     | Side stream async copy     | Minimal, 2 lines       | Must add to EVERY copy  |
  | synchronize()     | Memory free/resume        | Total safety, 2 lines  | Blocks ALL streams      |
  | copy_streams set  | Shared buffer multi-write | Precise tracking       | 19+ lines, more complex |
  | config change     | Architecture-level        | Zero code change       | May lose features       |
  | cache_invalid.    | Weight update + cache     | Clean model            | Must track all caches   |

  ★★★ For RTX 4090 GRPO:
  1. sleep_level=1 AVOIDS #45552 entirely (LoRA offload, not CuMemAllocator)
  2. overlap_comm=False AVOIDS #8061 entirely (no reduction_stream)
  3. naive checkpoint AVOIDS NCCL overhead (no side stream for sync)
  4. LoRA r=32 AVOIDS #6782 entirely (EOS works at rank 32)
  5. ZeRO-2 AVOIDS #8072/#8076 entirely (no param partitioning)
  6. gs=8 AVOIDS singleton degeneration (strong normalization)

  ★★★ BUT: #6794 record_stream still unfixed → monitor for silent corruption
  ★★★ AND: #28771 HiCache still unfixed → monitor accept_length < 2.0
""")


def mode_rtx4090():
    """RTX 4090 specific implications and mitigations."""
    print_header("MODE: rtx4090 — RTX 4090 CUDA Stream Safety Impact & Mitigations")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ★★★ RTX 4090 (24 GiB, PCIe only) CUDA Stream Safety ★★★     ║
╚══════════════════════════════════════════════════════════════════╝
""")

    mitigations = [
        ("★★★ AVOIDED by config", [
            ("#45552 cumem crash", "sleep_level=1 → LoRA offload doesn't use CuMemAllocator → crash avoided"),
            ("#8061 stream race", "overlap_comm=False → no reduction_stream → no race condition"),
            ("#8072 ZeRO-3 dtype", "ZeRO-2/FSDP1 → no parameter partitioning → no dtype mismatch"),
            ("#6782 LoRA EOS bug", "LoRA r=32 → EOS works normally, r>=64 = broken"),
            ("#5394 Muon clipping", "Not using Muon → AdamW/cpu_adam only"),
        ]),
        ("★★★ STILL AT RISK", [
            ("#6794 record_stream", "★★★ CRITICAL: silent corruption in LoRA delta weight sync → UNFIXED → monitor"),
            ("#28771 EAGLE degrade", "★★★ CRITICAL: 44% throughput loss over time → monitor accept_length < 2.0 → restart"),
            ("#28676 MoE cache", "HIGH for MoE models with SGLang → cache invalidation missing"),
        ]),
        ("★★★ NOT RELEVANT", [
            ("#8075 NVMe offload", "MUST NOT use NVMe offload → fd leak + unnecessary on RTX 4090"),
            ("#8080 overlap_comm fix", "overlap_comm pointless on dp=1 → fix irrelevant"),
        ]),
    ]

    for category, items in mitigations:
        print_section(category)
        for issue, mitigation in items:
            print(f"  {issue:<25} → {mitigation}")

    print_section("RTX 4090 Monitoring Checklist")
    print("""
  During GRPO training on RTX 4090, monitor these CUDA stream safety indicators:

  1. ★★★ Weight sync corruption (#6794):
     - Check LoRA delta weights are finite after sync (no NaN/inf)
     - Compare weight values before/after sync (should match)
     - Log weight sync timing — unexpected slowdown = potential corruption
     - If corruption detected: restart training from last checkpoint

  2. ★★★ EAGLE accept_length degradation (#28771):
     - Monitor SGLang accept_length metric every 100 steps
     - Threshold: restart SGLang engine if accept_length < 2.0
     - Expected: accept_length ≈ 3.0-3.5 initially, gradual decline
     - Alternative: disable EAGLE speculative decoding entirely

  3. ★★★ sleep/wake timing (#45552):
     - Log sleep/wake duration at each step
     - Expected: sleep < 0.5s, wake < 1s (sleep_level=1)
     - If timing spikes > 2x → potential stream sync issue
     - sleep_level=1 AVOIDS the bug, but monitor anyway

  4. ★★★ GPU memory allocation pattern:
     - torch.cuda.memory_allocated() at each phase transition
     - Expected: ~17 GiB during training, ~19 GiB during rollout
     - Unexpected growth > 0.5 GiB/step → memory leak investigation
""")

    print_header("RTX 4090 SAFE CONFIG SUMMARY")
    print("""
  The following config AVOIDS all 5 CUDA stream safety bugs that can crash RTX 4090:

  framework           = verl
  strategy            = fsdp1 (NOT ZeRO-3 → avoids #8072)
  lora_rank           = 32 (NOT >= 64 → avoids #6782)
  reference_mode      = bypass + ref_in_actor (5→1 forward passes)
  optimizer           = cpu_adam (0 GiB GPU optimizer states)
  rollout_engine      = sglang (prefix caching, sleep_level=1)
  sleep_level         = 1 (NOT 2 → avoids #45552 crash)
  checkpoint_engine   = naive (dp=1 direct memcpy → no side streams)
  overlap_comm        = False (no reduction_stream → avoids #8061)
  group_size          = 8 (NOT 1 → avoids REINFORCE degeneration)

  ★★★ Only remaining risks: #6794 (silent) and #28771 (gradual)
  ★★★ Both mitigated by monitoring → detect and restart
""")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUDA Stream Safety Pattern Family Synthesis")
    parser.add_argument("mode", choices=["map", "analyze", "compare", "rtx4090"],
                        help="Mode to run")
    args = parser.parse_args()

    if args.mode == "map":
        mode_map()
    elif args.mode == "analyze":
        mode_analyze()
    elif args.mode == "compare":
        mode_compare()
    elif args.mode == "rtx4090":
        mode_rtx4090()


if __name__ == "__main__":
    main()
