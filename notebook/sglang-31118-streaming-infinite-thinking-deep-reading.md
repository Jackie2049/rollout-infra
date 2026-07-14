# SGLang #31118/#31119: Qwen3 Streaming Infinite Thinking

## Overview
- **Issue**: sgl-project/sglang #31118 by HH1162, July 14, 2026
- **Fix PR**: #31119 (Draft) — adds stateful streaming parser
- **Status**: OPEN, no labels/assignees
- **Severity**: ★★★★★★★★ — causes infinite reasoning, request timeouts, resource exhaustion

## The Bug: Cross-Chunk Tag Truncation

### What Happens
Qwen3 models with reasoning (thinking tags) deployed via SGLang streaming occasionally enter an infinite thinking state — continuously generating reasoning content without transitioning to answer.

### Root Cause
SGLang's streaming chunking mechanism splits output into chunks. When the thinking end tag (`</think>`) gets physically **split across two consecutive chunks**, the parser only sees each chunk individually. With naive string matching on individual chunks, the parser never detects the complete tag:

```
Chunk 1: "... some reasoning content </th"
Chunk 2: "ink>\n\nThe final answer is..."
```

Since the parser never sees `</think>`, it never emits the end-of-thinking signal → model continues reasoning indefinitely.

### Affected Scope
- All models using `qwen3` reasoning parser (Qwen3, QwQ)
- All streaming requests (`stream: true`)
- All environments where streaming chunking splits tag boundaries

## Fix (PR #31119)

Replaces fragile regex matching with a robust streaming parser:

1. **Cross-Chunk Prefix Buffering**: `_calculate_safe_length` function detects partial tag prefixes at chunk boundaries and buffers them until next chunk completes the match.

2. **While-Loop State Machine**: Continuously consumes concatenated text using `find()` + physical slicing (`before_tag`/`after_tag`) to discard matched tags.

3. **Input Circuit Breaker**: `MAX_INPUT_SIZE = 1MB` prevents OOM from malicious/garbled input.

Key files:
- `sglang/srt/managers/qwen3_parser.py` (new)
- `sglang/srt/entrypoints/http_server.py` (modified)
- `sglang/srt/entrypoints/openai/serving_chat.py` (modified)
- `sglang/srt/entrypoints/openai/serving_completions.py` (modified)

## Cross-Framework Connection to TRL #6361

**This is the exact same pattern as TRL #6361** (our fork fix!):

| Framework | Issue | Pattern | Root Cause | Fix |
|-----------|-------|---------|------------|-----|
| SGLang | #31118 | Streaming chunks split `</think>` → infinite thinking | Naive per-chunk regex matching | Stateful streaming parser with prefix buffering |
| TRL (HF) | #6361 | Chat template produces duplicate `<think>` | `if '<think>' in content and '</think>' in content` fails when tag split across template boundaries | Added `elif '</think>' in content` branch |

**Key insight**: Both frameworks had the same class of bug — thinking tag detection logic that fails when tags are split across boundaries (streaming chunks in SGLang, template blocks in TRL). The fix approach differs (stateful parser vs template branch) but addresses the same root pattern.

## GRPO Relevance
- Qwen3/QwQ models are commonly used for RL training
- Streaming infinite thinking causes GRPO rollout generation to hang
- Request timeouts waste GPU resources in training pipelines
- Fix ensures reliable rollout generation for reasoning models

## Monitoring
- #31119 is Draft, needs CI and review
- PR #31119 linked, authored same person
- SGLang maintainers need to review
- After merge: enables reliable Qwen3 streaming for GRPO rollouts
