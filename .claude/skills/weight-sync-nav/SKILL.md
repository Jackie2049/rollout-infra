# Weight Sync Navigation Skill

Navigate RLHF training weight synchronization issues across 7 frameworks (verl, DeepSpeed, rLLM, Megatron, vLLM, SGLang, vLLM-Ascend).

## Trigger
- User asks about weight sync, sleep/wake, buffer preservation, LoRA transfer, or state transitions in RL training
- User mentions verl HYBRID, DeepSpeed ZeRO sleep/wake, or NPU/CUDA weight sync issues
- User wants to check if DSA Hadamard matrices survive sleep/wake transitions
- User mentions vLLM-Ascend #10684, buffer corruption, or constant buffer preservation

## Knowledge Base

### Key Tools
- `tools/rlhf_weight_sync_safety_checker.py` — 3 modes (check/compare/rtx4090), 7 framework profiles
- `tools/dsa_hadamard_sleep_wake_validator.py` — 3 modes (check/rtx4090/analyze), Hadamard preservation + pattern taxonomy
- `tools/seven_framework_critical_dashboard.py` — 23 issues, 12 CRITICAL, includes vLLM-Ascend
- `tools/rtx4090_oss_contribution_tracker.py` — 17 contributions tracked

### Key Notes
- `notebook/fundamentals/cross-framework-weight-sync-sleep-wake-comparison.md` — universal pattern
- `notebook/projects/vllm-ascend-critical-developments-2026-06-18-reading.md` — #10684 + #10579 + #10592
- `notebook/projects/dsv4-systematic-instability-pattern-synthesis.md` — 7 DSV4 failures + sleep/wake pattern
- `notebook/projects/verl-worker-lifecycle-ray-weight-sync-reading.md` — verl 6-step sync lifecycle

### Critical Findings
- ★★★★★★★★ #10684: DSA Hadamard ALL-ZERO after sleep/wake → PRIMARY: in-place mutation, SECONDARY: transfer exclusion
- ★★★★★★★★ Universal rule: sleep/wake + constant buffer NOT in transfer = corruption risk
- ★★★★★★★★ Per-step dynamic data MUST NOT be cached across steps (not just CUDA graphs!)
- ★★★★★★★★ verl HYBRID: sleep_level=1 (LoRA) → only KV released → buffers should survive → BUT unverified on CUDA!
- ★★★★★★★★ DeepSpeed ZeRO-2: overlap_comm + torch.compile = NaN (#8061) → MUST overlap_comm=False

### RTX 4090 Recommendations
- verl HYBRID: FSDP2 + CPPO + bypass_mode + sync TransferQueue trainer
- LoRA: rank=32/alpha=64 MANDATORY (rank=64 breaks EOS #6782)
- DSV4 models: enforce_eager=True MANDATORY
- DeepSpeed ZeRO-2: overlap_comm=False + gradient_clipping=1.0
- Verify DSA Hadamard preservation on CUDA GPU (when available) → P8 contribution if bug found!

## Actions
1. For framework-specific questions: run `python3 tools/rlhf_weight_sync_safety_checker.py check --framework <name>`
2. For cross-framework comparison: run `python3 tools/rlhf_weight_sync_safety_checker.py compare`
3. For RTX 4090 config validation: run `python3 tools/rlhf_weight_sync_safety_checker.py rtx4090 --config <config>`
4. For Hadamard preservation check: run `python3 tools/dsa_hadamard_sleep_wake_validator.py check --engine <name>`
5. For DSA pattern analysis: run `python3 tools/dsa_hadamard_sleep_wake_validator.py analyze`
6. For critical issues: run `python3 tools/seven_framework_critical_dashboard.py --brief`
7. Read relevant notebook notes for deep analysis
8. Check GPU availability before any validation experiments
