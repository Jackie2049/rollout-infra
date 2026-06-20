# Claude Code Skills Conversion Scaffold

> Created: 2026-06-20 | Priority: HIGH — user requirement for future skill conversion
> Purpose: Map the rollout-infra knowledge base, tools, and domains to Claude Code skill definitions

## 1. Current Project Structure → Skill Mapping

### Knowledge Base (25 domains → 7 skill families)

| Skill Family | Source Domains | Tools | Skill Type |
|-------------|---------------|-------|-----------|
| **rtx4090-training** | transformer, grpo, ppo_vs_grpo, lora, zero_gradient_flow, fsdp_decision, storage_lifecycle | grpo_memory_planner, grpo_config_generator_4090, rtx4090_grpo_config_validator, rtx4090_grpo_quick_reference | Training pipeline skill |
| **grpo-debug** | grpo, verl_training_loop, verl_v1_bugs, zero_gradient_flow, muon_clipping | grpo_training_debug_playbook, grpo_nan_debugging_guide, megatron_muon_clipping_avoidance_tool | Debug/diagnostic skill |
| **cross-framework-avoidance** | cross_framework_bug_avoidance, muon_clipping, discrete_decision_mismatch, lora_distortion, moe_nan | cross_framework_grpo_bug_avoidance_matrix, moe_nan_fp32_softmax_avoidance_tool | Validation skill |
| **vllm-v1-nav** | vllm_v1_bugs, vllm_v1_architecture (existing skill) | (existing skill) | Navigation skill |
| **megatron-nav** | megatron_muon_clipping, megatron_core_training, moe_routing (existing skill) | (existing skill) | Navigation skill |
| **distributed-nav** | zero_gradient_flow, fsdp_decision, cuda_stream_safety | fsdp1_vs_fsdp2_decision_guide, deepspeed_zero_safety_checker | Navigation skill |
| **inference-perf** | transformer, spec_decode, quantization, kv_cache (existing skill) | (existing skill) | Estimation skill |

### Tool Inventory (11 session tools → skill commands)

| Tool | Modes | Skill Command Pattern | Description |
|------|-------|---------------------|-------------|
| `verl_v1_grpo_training_loop_simulator.py` | simulate/compare/rtx4090/lifecycle | `/rtx4090-training simulate` | Simulate GRPO training loop |
| `verl_v1_grpo_data_flow_tracer.py` | trace/verify/debug/rtx4090 | `/rtx4090-training trace` | Trace TransferQueue data flow |
| `grpo_memory_planner.py` | plan/compare/verify/rtx4090 | `/rtx4090-training plan` | Plan memory budget |
| `cuda_stream_safety_pattern_synthesis.py` | map/analyze/compare/rtx4090 | `/grpo-debug stream-safety` | Analyze CUDA stream patterns |
| `grpo_training_debug_playbook.py` | playbook/symptoms/fixes/rtx4090 | `/grpo-debug playbook` | Debug playbook |
| `megatron_muon_clipping_avoidance_tool.py` | check/diagnose/avoid/rtx4090 | `/grpo-debug muon-check` | Check Muon clipping config |
| `cross_framework_grpo_bug_avoidance_matrix.py` | matrix/validate/rtx4090/cross-framework | `/cross-framework-avoidance validate` | Validate config against 34 rules |
| `moe_nan_fp32_softmax_avoidance_tool.py` | check/diagnose/avoid/rtx4090 | `/cross-framework-avoidance moe-check` | Check MoE NaN config |
| `fsdp1_vs_fsdp2_decision_guide.py` | guide/compare/validate/rtx4090 | `/distributed-nav fsdp-guide` | FSDP decision guide |
| `algorithm_infra_knowledge_map.py` | map/theory/bug/decision/rtx4090/proof/cross | `/rtx4090-training knowledge-map` | Query theory→bug→decision map |
| `grpo_config_generator_4090.py` | generate | `/rtx4090-training config` | Generate GRPO config |

## 2. Skill Definition Template

Each skill should follow this format for Claude Code conversion:

```yaml
skill_name: rtx4090-training
description: "RTX 4090 GRPO training pipeline configuration, simulation, and debugging"
trigger: "when user asks about RTX 4090 training, GRPO config, memory planning, or debugging"
commands:
  - /rtx4090-training simulate   → python3 tools/verl_v1_grpo_training_loop_simulator.py rtx4090
  - /rtx4090-training trace      → python3 tools/verl_v1_grpo_data_flow_tracer.py rtx4090
  - /rtx4090-training plan       → python3 tools/grpo_memory_planner.py rtx4090
  - /rtx4090-training config     → python3 tools/grpo_config_generator_4090.py generate
  - /rtx4090-training knowledge-map → python3 tools/algorithm_infra_knowledge_map.py rtx4090
  - /rtx4090-training debug      → python3 tools/grpo_training_debug_playbook.py rtx4090
  - /rtx4090-training validate   → python3 tools/cross_framework_grpo_bug_avoidance_matrix.py validate
knowledge_sources:
  - notebook/auto-memory.md (always loaded)
  - notebook/projects/rtx4090-grpo-training-runbook.md
  - notebook/projects/7-framework-status-update-2026-06-20-session3.md
  - notebook/projects/oss-contribution-action-plan-2026-06.md
rules:
  MUST_DO: 19 rules from algorithm_infra_knowledge_map.py
  MUST_NOT: 17 rules from algorithm_infra_knowledge_map.py
  priority_matrix: RTX 4090 GRPO Impact Priority Matrix from megatron_muon_clipping_avoidance_tool.py rtx4090
```

## 3. Skill Conversion Priority

| Priority | Skill | Reason | Current Coverage |
|----------|-------|--------|-----------------|
| ★★★★★★★★ P0 | rtx4090-training | Most-used skill, 6 tools, comprehensive | 11 tools ready |
| ★★★★★★★★ P0 | grpo-debug | Debug/diagnostic, most valuable for training | 5 tools ready |
| ★★★★★★★★ P0 | cross-framework-avoidance | 36-rule validation, unique | 2 tools ready |
| ★★★★★★★★ P1 | distributed-nav | FSDP decision + ZeRO safety | 2 tools ready |
| ★★★ P2 | vllm-v1-nav | Already exists as skill | existing |
| ★★★ P2 | megatron-nav | Already exists as skill | existing |
| ★★★ P2 | inference-perf | Already exists as skill | existing |

## 4. Conversion Checklist

### Before conversion:
- [x] All tools are runnable Python scripts with 4+ modes
- [x] Each tool has argparser with clear help text
- [x] Knowledge map is queryable (algorithm_infra_knowledge_map.py)
- [x] Auto-memory.md provides persistent context
- [x] MUST DO / MUST NOT rules are encoded in tools
- [x] RTX 4090 specific reports in each tool
- [ ] Skill YAML definitions need to be created
- [ ] Command routing needs to be implemented
- [ ] Knowledge source loading needs automation

### Conversion steps:
1. Create `.claude/skills/` directory with YAML skill definitions
2. Map each tool's modes to skill command patterns
3. Define trigger conditions for each skill
4. Create knowledge source loading rules
5. Implement command routing (bash → python3 tools/...)
6. Test skill invocations end-to-end
7. Document skill API reference

## 5. Knowledge Source Loading Strategy

```python
# Skill knowledge loading pattern:
# 1. Always load: auto-memory.md (project context, readiness, rules)
# 2. On-demand load: domain-specific reading notes
# 3. Query: algorithm_infra_knowledge_map.py (theory→bug→decision)
# 4. Validate: cross_framework_grpo_bug_avoidance_matrix.py (config checker)

ALWAYS_LOAD = [
    "notebook/auto-memory.md",
    "notebook/projects/oss-contribution-action-plan-2026-06.md",
]

ON_DEMAND_LOAD = {
    "rtx4090": [
        "notebook/projects/rtx4090-grpo-training-runbook.md",
        "notebook/projects/7-framework-status-update-2026-06-20-session3.md",
    ],
    "megatron": [
        "notebook/projects/megatron-lm-june-2026-critical-developments-reading.md",
        "notebook/projects/megatron-5394-chained-optimizer-muon-clipping-reading.md",
    ],
    "vllm": [
        "notebook/projects/vllm-v1-architecture-critical-bugs-deep-reading.md",
        "notebook/projects/vllm-v1-sleep-wake-cumemallocator-deep-reading.md",
    ],
    "verl": [
        "notebook/projects/verl-v1-grpo-training-loop-architecture-deep-reading.md",
        "notebook/projects/verl-v1-critical-bugs-issues-2026-06-20.md",
    ],
    "rllm": [
        "notebook/projects/rllm-latest-developments-2026-06-session3-reading.md",
    ],
    "mindie": [
        "notebook/projects/mindie-vllm-ascend-latest-developments-2026-06-session3-reading.md",
    ],
    "deepspeed": [
        "notebook/projects/deepspeed-zero1-2-gradient-flow-stream-safety-deep-reading.md",
    ],
    "pytorch": [
        "notebook/projects/pytorch-fsdp2-2026-deep-reading.md",
    ],
}
```

## 6. Tool API Reference (for skill conversion)

### Common mode pattern:
All session tools follow the same 4-mode pattern:
- `rtx4090` mode: comprehensive RTX 4090 report
- `check/validate/plan/guide` mode: validate config or generate recommendations
- `diagnose/trace/debug` mode: symptom→bug mapping
- `compare/matrix/analyze` mode: comparison or analysis

### Tool invocation format:
```bash
python3 tools/{tool_name}.py {mode} [--config JSON] [--symptoms STR] [--optimizer TYPE]
```

### Output format:
All tools produce structured text output with:
- ★★★★★★★★★ markers for critical findings
- PASS/FAIL/WARN status for validations
- Bug IDs and pattern classes for cross-reference
- Mathematical proofs for rules
- Memory budget calculations for RTX 4090

## 7. Next Steps

1. Create `.claude/skills/rtx4090-training.yaml` skill definition
2. Create `.claude/skills/grpo-debug.yaml` skill definition
3. Create `.claude/skills/cross-framework-avoidance.yaml` skill definition
4. Create `.claude/skills/distributed-nav.yaml` skill definition
5. Test skill invocations with actual tool runs
6. Refine trigger conditions based on usage patterns
7. Add skill documentation to README

**Timeline**: Skills conversion can begin when Claude Code skill framework is finalized. Current tools and knowledge base are ready for conversion — just need YAML skill definitions and command routing.
