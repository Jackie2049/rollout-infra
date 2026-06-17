# Continuous Token Cross-Framework Comparison — verl #6779 vs rLLM #658

> 2026-06-18 | Cross-framework comparison | Two approaches to drift-free multi-turn tokenization for RL
> ★★★★★★★★ verl #6779: ContinuousTokenBuilder (553 lines core + 209 wiring + 7 model families)
> ★★★★★★★★ rLLM #658: cumulative_token_mode for Tinker (byte-for-byte prefix extension)
> ★★★★★★★★ Both solve same problem: multi-turn RL needs drift-free token continuity across turns

---

## 1. The Problem: Multi-Turn Token Drift

```
★★★★★★★★★ Multi-turn RL (GRPO/agent training) needs token continuity:

Turn 1: prompt tokens → assistant generates → turn 1 complete
Turn 2: prompt + turn1_assistant + tool_response → assistant generates → turn 2 complete
Turn 3: prompt + turn1 + tool1 + turn2 + tool2 → assistant generates

★★★★★★★★★ Without continuous token handling:
  → Each turn re-tokenizes entire conversation → different token IDs
  → Token drift: same text → different tokens → different model input
  → Non-deterministic: re-tokenization depends on tokenizer implementation
  → GRPO advantage computation broken: different tokens = different positions = different values

★★★★★★★★★ With continuous token handling:
  → Turn N prompt = turn (N-1) prompt + completion + new tool response tokens
  → Prefix tokens preserved byte-for-byte → no drift → deterministic
  → Only append NEW tokens → prefix unchanged → model sees same context
  → GRPO advantage: same prefix = same positions = consistent values
```

---

## 2. verl #6779: ContinuousTokenBuilder

```
★★★★★★★★★ Architecture (5024 additions, 22 files):

Core module: verl/utils/continuous_token.py (553 lines)
  → ContinuousTokenBuilder class
    → build_initial_tokens() → initial prompt tokenization
    → merge_tokens() → append tool response + boundary tokens
  → MergeResult dataclass → describes token edits at runtime-prefix boundary
    → token_ids, appended_token_count, kind (assistant/non_assistant)
    → inserted_token_ids → CT-created boundary tokens → not model-generated
    → removed_prefix_token_count → prefix trimming for re-tokenization edge cases
  → ct_align_response_metadata() → align response_mask + logprobs after merge
    → Mirrors token edits for response-side metadata
    → Inserted tokens: mask=0, logprobs=0.0 → not model-generated → no loss
    → Assistant tokens: mask=1, logprobs from assistant_logprobs

★★★★★★★★★ Factory module: verl/utils/continuous_token_wiring.py (209 lines)
  → ContinuousTokenModelFamily enum → 14 model families
  → Builder registry → 5 builder classes:
    → ContinuousTokenBuilder (DEFAULT) → generic
    → QwenContinuousTokenBuilder → Qwen2.5/3/3.5 specific boundary handling
    → MiniMaxContinuousTokenBuilder → MiniMax M2/M25/M27 boundary
    → GLMContinuousTokenBuilder → GLM-47/GLM-5 boundary
    → Gemma4ContinuousTokenBuilder → Gemma 4 boundary
    → GptOssContinuousTokenBuilder → GPT-OSS boundary
  → create_continuous_token_builder() → factory function
  → resolve_continuous_token_model_family() → auto-detect from HF config

★★★★★★★★★ Key design choices:
  → Disabled by default → existing rollout behavior unchanged
  → Each model family has different chat template boundary handling
  → Boundary tokens: inserted at merge junction → NOT model-generated → no loss
  → MergeResult tracks inserted vs appended → response_mask alignment
  → 723-line CPU test suite → verl/utils/test_continuous_token_on_cpu.py
```

---

## 3. rLLM #658: Cumulative Token Mode for Tinker

```
★★★★★★★★★ Architecture (+428/-31, 4 files):

Core change: Tinker local_handler cumulative token support
  → Tinker detects pre-tokenized prompt (list[int]) → samples directly
  → get_token_output_from_token_input() → samples from token IDs
  → Previously: each turn re-rendered and re-tokenized full conversation → drift
  → Now: turn N prompt tokens = prior turns prompt+completion tokens → preserved

★★★★★★★★★ Key design choices:
  → Extends existing cumulative_token_mode flag to Tinker (was HTTP-proxy only)
  → Proxy routes cumulative requests to local_handler when present
  → No HTTP worker needed → in-process → faster than HTTP proxy
  → Guarantees byte-for-byte prefix extension → turn-2 prompt IDs = turn-1 IDs

★★★★★★★★★ Simplified vs verl #6779:
  → rLLM: simpler approach → just preserve token IDs → no builder layer
  → verl: more sophisticated → builder layer + model-family registry + boundary handling
  → rLLM: works at Tinker level → in-process → no server-side changes
  → verl: works at agent_loop level → more general → handles chat template boundaries

★★★★★★★★★ NOT yet validated against live Tinker run (from PR comments)
```

---

## 4. Comparison Matrix

```
★★★★★★★★★ verl #6779 vs rLLM #658:

| Aspect | verl #6779 | rLLM #658 |
|--------|-----------|-----------|
| Size | 5024 additions, 22 files | 428 additions, 4 files |
| Core module | ContinuousTokenBuilder (553 lines) | cumulative_token_mode (simple extension) |
| Model families | 14 (7 builder classes) | 1 (generic) |
| Boundary handling | Per-model chat template differences | Generic (prefix preservation) |
| Token alignment | MergeResult + ct_align_response_metadata | Direct token ID reuse |
| Disabled by default | Yes | Yes (cumulative_token_mode flag) |
| Complexity | High (builder pattern + registry) | Low (just preserve IDs) |
| Precision | Per-model boundary token handling | Byte-for-byte prefix extension |
| Test coverage | 723-line CPU test + 1309-line comparison | Limited (needs live validation) |
| Status | OPEN (5024 additions) | MERGED (June 16) |
| RTX 4090 relevance | ★★★★★ GRPO multi-turn agent | ★★★★★ Tinker GRPO multi-turn |

★★★★★★★★★ Key insight: both solve same problem but at different sophistication levels:
  → rLLM #658: simple → just preserve token IDs → works for most models → but may not handle edge cases
  → verl #6779: sophisticated → per-model boundary handling → handles chat template differences
  → verl approach is more robust → but also more complex → 12x more LOC

★★★★★★★★★ For RTX 4090 GRPO:
  → Single-turn GRPO → no multi-turn → no continuous token needed!
  → Multi-turn agent GRPO → continuous token essential → both frameworks provide it
  → rLLM Tinker #658 → simpler → faster to deploy → but may need model-specific tweaks
  → verl #6779 → more robust → handles all model families → but needs merge first
```

---

## Key Findings Summary

★★★★★★★★★ Both verl #6779 and rLLM #658 solve multi-turn token drift for RL
★★★★★★★★★ verl: sophisticated builder pattern + 7 model families + boundary token handling (5024 additions)
★★★★★★★★★ rLLM: simple token ID preservation + cumulative_token_mode (428 additions, MERGED)
★★★★★★★★★ Single-turn GRPO → no continuous token needed → but multi-turn agent RL requires it
★★★★★★★★★ verl approach more robust but 12x more LOC → rLLM approach simpler but may miss edge cases

---

## References

- verl #6779: https://github.com/verl-project/verl/pull/6779
- rLLM #658: https://github.com/rllm-org/rllm/pull/658
- rLLM agent research: notebook/projects/rllm-latest-developments-2026-06-agent-research.md
