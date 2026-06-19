# verl PR #6779 — Continuous Token for Agentic Rollout (Multi-Turn Tokenization Builder)

> 2026-06-19 | verl-project/verl | PR #6779 | gxlvera | OPEN | +3413/-36 | 20 files
> RFC: #6719 (Continuous Token Support for Multi-Turn AgentLoop Rollout)
> Previous stacked PRs: #6720 (builder core), #6721 (agent loop integration) — now unified into #6779
> ★★★★★ Continuous Token → append-only token stream → multi-turn agent RL → preserves real inference token ids
> ★★★★★ Disabled by default → multi_turn.continuous_token.enable=True → existing rollout unchanged
> ★★★★★★★★ Model-family adapters → Qwen/MiniMax/GLM/Gemma4/GptOss → boundary-specific handling
> ★★★★★★★★ Synthetic context → tool responses need preceding assistant tool_call → correct incremental extraction
> ★★★ RTX 4090: short-term not needed (GRPO single-turn) → medium-term essential (agent RL multi-turn)
> ★★★★★★★★ Same pattern family as SGLang #28676/#28679, vLLM #46118, Megatron #5317 — State Lifecycle Mismatch

---

## 1. PR Overview and Motivation

### 1.1 What is Continuous Token?

Continuous Token (CT) is a **tokenization builder layer** that keeps the runtime token stream across multi-turn agentic rollout instead of reconstructing previous assistant turns from text. It is the rigorous implementation of the **token-in-token-out (TITO)** principle for multi-turn scenarios.

**Core idea**: In multi-turn rollout, the model generates tokens across multiple turns (assistant generation → tool call → tool response → next assistant generation). The training side must use the token ids that were *actually produced and observed on the inference side*, not ids reconstructed by re-encoding text through the chat template.

**Why this matters**: Many open-source model chat templates are not simply append-only. When incremental token ids are extracted carelessly and boundaries are not merged rigorously, the resulting multi-turn prompt can be wrong — silently wrong, with no error signal, which is the worst possible bug pattern for RL training.

### 1.2 RFC #6719: Four Pitfalls of Multi-Turn TITO

The RFC document (#6719) identifies four common pitfalls in generalizable multi-turn TITO design:

1. **Full messages must not be retokenized**: Preserve the real token ids accumulated from previous turns. Re-encoding causes BPE boundary changes and position-dependent rendering differences.

2. **Incremental non-assistant token ids must be extracted from an appropriate synthetic context**: Tool responses depend on preceding assistant tool calls. Without a synthetic assistant context, templates like MiniMax raise TemplateError, Nemotron omits `<|im_start|>user`, and Llama-3.1-8B-Instruct raises "Cannot put tools in the first user message when there's no first user message!"

3. **When incremental ids are merged back into old token ids, some models require explicit boundary handling**:
   - Qwen3: generation stops at `<|im_end|>`, but template needs `<|im_end|>\n` — missing newline
   - GLM4.7: `observation` and `user` tokens serve as both stop tokens and next-message BOS — duplicate boundary
   - MiniMax: generation stops at `[e~[`, template needs `[e~[\n` — missing newline

4. **A comparator is needed to check structural errors after construction**: Differences between assistant content and the canonical render of the full chat template can be tolerated, but deviations in special-token boundaries cannot.

### 1.3 Counterexamples from RFC

**DeepSeek-R1-Distill-Qwen-1.5B**: The generation prompt is `<｜Assistant｜>` + `thinking\n` (3 tokens). If padding replaces the assistant content, the template removes the `thinking\n` prompt tokens, causing position isomorphism failure. This is not BPE — it's completed-message templating eating generation-prompt tokens.

**GLM4.7**: Default generation prompt is `user\nthinking\n`. But historical assistant without `reasoning_content` renders as `user\nanswer\n{content}`, changing the marker before padding from real runtime value `thinking` to `answer`. This token appears before the padding span and cannot be recovered.

**Llama-3.1-8B-Instruct**: Constructing only `[_DUMMY_SYSTEM] + [tool_response]` for incremental extraction raises TemplateError because the template tries to place tool schema into the first user message, which doesn't exist in the synthetic context.

**Qwen3**: Without a synthetic assistant preceding tool messages, Qwen3 inserts or removes `thinking...answer` based on whether the assistant is after the last real user and whether it's the last message. The dummy assistant must not have a dummy user preceding it, because that would trigger thinking-block insertion/removal across the suffix diff boundary.

### 1.4 PR Stats

| Metric | Value |
|--------|-------|
| Title | [rollout] feat: add Continuous Token for Agentic Rollout |
| Author | gxlvera |
| State | OPEN |
| Branch | gxl-ct-dev → main |
| Commits | 37eaacf546d679b15c82abd67abd809d117d3e74 (latest) |
| Additions | 3413 |
| Deletions | 36 |
| Files changed | 20 |
| Created | 2026-06-16 |
| Last updated | 2026-06-18 |
| Mergeable | true |

### 1.5 CT vs Legacy Comparison Results

From the PR description, comparing CT-enabled vs legacy agent loop across model families:

```
Family        Models  Runs  Pass  Mismatch  Error  Notes
Qwen          9       27    9     18        0      Single-turn passes; tool cases mismatch because legacy misses newline after EOS. CT is correct.
GLM           4       12    4     8         0      Single-turn CT-vs-legacy passes; tool cases mismatch on observation boundary. CT is correct.
Kimi          3       9     9     0         0      All default trajectories pass with tool_parser=kimi.
Seed-OSS      1       3     3     0         0      All default trajectories pass with tool_parser=seed.
MiniMax       3       9     3     0         6      Single-turn passes; tool cases are legacy-only TemplateError.
MiMo          2       6     6     0         0      All default trajectories pass with Hermes parser.
Nemotron      3       9     3     6         0      Single-turn passes; tool cases mismatch because legacy misses tool-response user header. CT is correct.
```

Key finding: **Legacy path is silently wrong for 4 model families (Qwen, GLM, MiniMax, Nemotron)**. MiniMax outright crashes with TemplateError. CT is correct in all cases.

---

## 2. Architecture Analysis

### 2.1 Continuous Token Builder Core (`verl/utils/continuous_token.py`, 553 lines)

**ContinuousTokenBuilder** — the base class that implements the core TITO invariant:

```python
class ContinuousTokenBuilder:
    """Continuous Token builder for runtime prefix reuse."""
    allowed_append_roles: frozenset[str] = {"tool", "user", "system"}

    def build_initial_tokens(self, messages, *, tools=None) -> list[int]:
        """Build initial prompt token ids from messages with generation prompt."""
        return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

    def tokenize_incremental_messages(self, previous_messages, updated_messages, *, tools=None) -> list[int]:
        """Extract incremental token ids for appended messages via suffix diff."""
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages):]
        # Process each append group (tool, user, system)
        # Each group extracted via synthetic context + suffix diff
        # Finally append generation prompt suffix
        ...

    def merge_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None) -> MergeResult:
        """Merge runtime prefix tokens and appended tokens with boundary handling."""
        appended_ids = self.tokenize_incremental_messages(...)
        return self._merge_token_ids(runtime_token_ids, appended_ids)

    def append_assistant_tokens(self, runtime_token_ids, assistant_token_ids) -> MergeResult:
        """Append model-generated assistant tokens."""
        # Simple concatenation, kind="assistant"
        ...
```

**Key design decisions**:

1. **Append-only invariant**: `_assert_append_only()` validates that `updated_messages` starts with `previous_messages` and only appends roles in `allowed_append_roles`. This is the core TITO invariant at the message level.

2. **Suffix diff extraction**: `render_delta_token_id()` renders the prefix and full prompts separately, then returns the token-level suffix. This avoids retokenizing the entire conversation. Validation: `full_token_ids[:len(prefix_token_ids)] != prefix_token_ids` raises ValueError if the suffix diff fails.

3. **Synthetic context for tool messages**: `_tokenize_tool_group()` constructs `[_SYNTHETIC_SYSTEM, _SYNTHETIC_USER, synthetic_assistant_with_tool_calls]` as the prefix context, then renders `prefix + tool_messages` and extracts the suffix. The synthetic assistant carries tool call names, IDs, and `reasoning_content` marker.

4. **Synthetic context for user/system messages**: `_tokenize_single_non_tool()` uses `[_SYNTHETIC_SYSTEM, _SYNTHETIC_USER]` as prefix, renders `prefix + [message]`, extracts suffix.

5. **Append group iteration**: `_iter_append_groups()` groups consecutive tool messages together (for multi-tool parallel calls) and treats individual user/system messages separately.

**Why synthetic context matters**: Tool responses depend on the preceding assistant tool call. Without it, many templates crash (MiniMax TemplateError) or produce incorrect boundaries (Nemotron missing `<|im_start|>user`). The synthetic context restores the real preceding state of the rollout inside the extraction step.

**Why not dummy user**: Qwen3 decides whether to insert/clean `thinking...answer` based on whether the assistant is after the last real user and whether it's the last message. A dummy user would trigger thinking-block insertion/removal across the suffix diff boundary. The design explicitly avoids adding a dummy user before the synthetic assistant.

### 2.2 MergeResult Data Structure

```python
@dataclass(frozen=True)
class MergeResult:
    """Token merge result with the token-level delta needed by callers."""
    token_ids: list[int]              # Final merged token sequence
    appended_token_count: int         # Number of newly appended tokens
    kind: MergeKind = "non_assistant" # "assistant" or "non_assistant"
    inserted_token_ids: list[int] = field(default_factory=list)  # CT-created boundary tokens
    removed_prefix_token_count: int = 0  # Number of prefix tokens removed at boundary
```

**Why MergeResult matters for training**:

- `inserted_token_ids`: These are CT-created boundary tokens (e.g., the newline inserted by Qwen after `<|im_end|>`). They are NOT model-generated tokens. They must carry `response_mask[i]=0` (no loss) and `response_logprobs[i]=0.0` (no logprob).

- `removed_prefix_token_count`: Tokens removed from the prefix at the boundary (e.g., GLM removes the `observation` stop token before appending the next message). These tokens existed in the previous `response_mask` and `response_logprobs` and must be trimmed from the tail.

- `kind`: Determines how `appended_token_count` tokens should be masked. `kind="assistant"` → `response_mask[i]=1` (loss-bearing) with model logprobs. `kind="non_assistant"` → `response_mask[i]=0` (no loss) with 0.0 logprobs.

### 2.3 Response Metadata Alignment (`ct_align_response_metadata()`)

```python
def ct_align_response_metadata(
    merge_result: MergeResult,
    response_mask: list[int],
    response_logprobs: list[float] | None = None,
    *,
    assistant_logprobs: list[float] | None = None,
) -> tuple[list[int], list[float] | None]:
```

**Operation sequence**:
1. **Trim removed prefix tokens**: `aligned_mask = response_mask[:-removed_prefix_token_count]`
2. **Insert boundary tokens**: `aligned_mask += [0] * len(inserted_token_ids)` (CT-created, no loss)
3. **Append new tokens by kind**:
   - `kind="assistant"`: `aligned_mask += [1] * appended_token_count` + `aligned_logprobs += assistant_logprobs`
   - `kind="non_assistant"`: `aligned_mask += [0] * appended_token_count` + `aligned_logprobs += [0.0] * appended_token_count`

This ensures that the training side receives a correctly aligned `response_mask` and `response_logprobs` that mirrors the token edits at the merge boundary.

### 2.4 Tool Name Resolution (`_assistant_tool_call_names()`)

When constructing the synthetic assistant for tool responses, the builder must know the tool call names. It resolves them from context messages:

1. **tool_names_by_id**: Maps `tool_call.id → name` from the most recent assistant message with tool_calls
2. **positional_tool_names**: Ordered list of tool call names from the most recent assistant message

Then `_resolve_tool_name()` tries:
1. Match by `tool_call_id` → exact mapping
2. Match by positional index → `positional_tool_names[index]`
3. Match by `message["name"]` → explicit name field
4. Fallback → `"continuous_token_tool"` (placeholder)

This handles the case where tool messages may have `tool_call_id` (modern format) or just positional correspondence (older format).

### 2.5 Model-Family Adapters (Boundary Handling)

Each model family overrides `_merge_token_ids()` (and sometimes other methods) for boundary-specific behavior.

#### QwenContinuousTokenBuilder

**Problem**: Qwen2.5/Qwen3/Qwen3.5 chat templates render `<|im_end|>\n` after each turn. But generation stops at `<|im_end|>` without the newline. When the runtime prefix ends at `<|im_end|>`, the newline is missing before appending non-assistant tokens.

**Solution**: Insert newline token after `<|im_end|>`:

```python
class QwenContinuousTokenBuilder(ContinuousTokenBuilder):
    def _merge_token_ids(self, runtime_token_ids, appended_token_ids) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids = []
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)  # Insert missing newline
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )
```

**Validation**: Constructor asserts `newline_ids = tokenizer.encode("\n")` must be exactly 1 token. If not, raises ValueError — this catches tokenizers where `\n` encodes to multiple tokens (rare but possible for some BPE schemes).

**RTX 4090 relevance**: Qwen is the most common GRPO model family on RTX 4090. Without this fix, multi-turn GRPO on Qwen would silently produce wrong token boundaries — the missing newline would cause the next turn's template to render incorrectly, and the response_mask would be misaligned.

#### MiniMaxContinuousTokenBuilder

**Problem**: MiniMax templates render `[e~[\n` after each turn. Generation stops at `[e~[` without the newline.

**Solution**: Same pattern as Qwen — insert newline after `[e~[`:

```python
class MiniMaxContinuousTokenBuilder(ContinuousTokenBuilder):
    def _merge_token_ids(self, runtime_token_ids, appended_token_ids) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids = []
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(...)
```

**Additional problem**: Without a synthetic assistant preceding tool messages, MiniMax's chat template raises TemplateError. CT's synthetic context solves this.

#### GLMContinuousTokenBuilder

**Problem**: GLM's `observation` and `user` tokens serve as both assistant stop tokens and next-message start tokens. If the runtime prefix ends with either, the boundary is duplicated when the next message starts.

**Solution**: Remove the boundary token from the prefix before appending:

```python
class GLMContinuousTokenBuilder(ContinuousTokenBuilder):
    def _merge_token_ids(self, runtime_token_ids, appended_token_ids) -> MergeResult:
        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]  # Remove duplicate boundary
            removed_prefix_token_count = 1
        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            removed_prefix_token_count=removed_prefix_token_count,
        )
```

**This is different from Qwen/MiniMax**: Instead of inserting a token, GLM removes one. The `removed_prefix_token_count` field in MergeResult captures this, and `ct_align_response_metadata()` trims the corresponding entries from `response_mask` and `response_logprobs`.

**Why this is tricky**: The removed token existed in the previous response_mask (it was the last token of the previous turn's response). Removing it means the training side computes loss on one fewer token. This is correct — the `observation`/`user` token at the boundary shouldn carry loss for the previous assistant turn, and it will be covered by the next turn's template rendering.

#### Gemma4ContinuousTokenBuilder

**Problem**: Gemma4 requires `<|tool_response>` token before tool response content. The runtime prefix may not end with this token.

**Solution**: Override `merge_tokens()` (not `_merge_token_ids()`) to insert `<|tool_response>` if the prefix doesn't already end with it:

```python
class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    def merge_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None) -> MergeResult:
        appended_token_ids = self.tokenize_incremental_messages(...)
        appended_messages = updated_messages[len(previous_messages):]
        prefix = list(runtime_token_ids)
        inserted_token_ids = []
        if appended_messages and prefix[-1:] != [self._tool_response_id]:
            prefix.append(self._tool_response_id)
            inserted_token_ids.append(self._tool_response_id)
        return MergeResult(...)
```

**Note**: This overrides `merge_tokens()` rather than `_merge_token_ids()` because the insertion depends on whether there are appended messages (not just on the boundary token). If the prefix already ends with `<|tool_response>`, no insertion is needed.

#### GptOssContinuousTokenBuilder

**Problem**: GPT-OSS (Harmony format) uses a completely different tool response format that doesn't go through the chat template. The format is:

```
<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{json_content}<|end|>
```

**Solution**: Override `_tokenize_tool_group()` to bypass suffix diff entirely:

```python
class GptOssContinuousTokenBuilder(ContinuousTokenBuilder):
    def _tokenize_tool_group(self, tool_messages, *, context_messages=None, tools=None) -> list[int]:
        # Build response text directly with Harmony format
        response_text = "".join(
            self._format_tool_response(tool_message, tool_name)
            for tool_message, tool_name in ...
        )
        return self.tokenizer.encode(response_text, add_special_tokens=False)

    @staticmethod
    def _format_tool_response(tool_message, tool_name) -> str:
        content = json.dumps(_stringify_tool_content(tool_message.get("content", "")), ensure_ascii=False)
        return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>"
```

**Why bypass suffix diff**: The Harmony format is not rendered by `apply_chat_template()`. It's a special token-based format that must be manually constructed. Suffix diff doesn't apply here because there's no chat template to render.

### 2.6 Builder Registry and Wiring (`verl/utils/continuous_token_wiring.py`, 209 lines)

**ContinuousTokenModelFamily enum** — 13 values:
```python
class ContinuousTokenModelFamily(StrEnum):
    AUTO = "auto"         # Infer from model path/tokenizer
    DEFAULT = "default"   # Plain ContinuousTokenBuilder
    QWEN = "qwen"         # QwenContinuousTokenBuilder
    QWEN25 = "qwen25"     # QwenContinuousTokenBuilder (same class)
    QWEN3 = "qwen3"       # QwenContinuousTokenBuilder (same class)
    QWEN35 = "qwen35"     # QwenContinuousTokenBuilder (same class)
    MINIMAX = "minimax"   # MiniMaxContinuousTokenBuilder
    MINIMAX_M2 = "minimaxm2"   # MiniMaxContinuousTokenBuilder
    MINIMAX_M25 = "minimaxm25" # MiniMaxContinuousTokenBuilder
    MINIMAX_M27 = "minimaxm27" # MiniMaxContinuousTokenBuilder
    GLM47 = "glm47"       # GLMContinuousTokenBuilder
    GLM5 = "glm5"         # GLMContinuousTokenBuilder
    GEMMA4 = "gemma4"     # Gemma4ContinuousTokenBuilder
    GPTOSS = "gptoss"     # GptOssContinuousTokenBuilder
```

**Registry mapping**: Each family maps to its builder class. Qwen2.5/Qwen3/Qwen3.5 all use `QwenContinuousTokenBuilder` (same boundary behavior). MiniMax M2/M2.5/M2.7 all use `MiniMaxContinuousTokenBuilder`. GLM-4.7/GLM-5 both use `GLMContinuousTokenBuilder`.

**Auto-detection priority** (`infer_continuous_token_model_family()`):
```
glm-5 / glm_5 / glm5  → GLM5
glm-4.7 / glm_4.7 / glm47 → GLM47
gemma-4 / gemma_4 / gemma4 → GEMMA4
gpt-oss / gpt_oss / gptoss → GPTOSS
minimaxm27 → MINIMAX_M27
minimaxm25 → MINIMAX_M25
minimaxm2 → MINIMAX_M2
minimax → MINIMAX
qwen3.5 / qwen3_5 / qwen35 → QWEN35
qwen2.5 / qwen2_5 / qwen25 → QWEN25
qwen3 → QWEN3
(unmatched) → DEFAULT (conservative fallback)
```

**Detection mechanism**: Concatenates `model_path`, `tokenizer_name_or_path`, and `tokenizer.name_or_path` into a haystack string, strips non-alphanumeric, then matches against known markers. The ordering is important — `minimaxm2` must be checked before `minimax` to avoid false matches (same for `qwen2.5` before `qwen`).

**Conservative fallback**: Unknown models fall back to `DEFAULT` (plain concatenation). This is correct — CT doesn't require model-specific handling if the tokenizer is prefix-stable and has no boundary issues. The `chat_template_checker.py` tool can validate this.

**Factory function**:
```python
def create_continuous_token_builder(tokenizer, *, model_family, model_path=None,
    tokenizer_name_or_path=None, chat_template_kwargs=None, **builder_kwargs):
    resolved_family = resolve_continuous_token_model_family(model_family, ...)
    builder_cls = get_continuous_token_builder_class(resolved_family)
    return builder_cls(tokenizer, chat_template_kwargs=chat_template_kwargs, **builder_kwargs)
```

---

## 3. Source Code Walkthrough

### 3.1 AgentLoopBase Integration (`verl/experimental/agent_loop/agent_loop.py`, +95/-8)

**Initialization change**: When `continuous_token_config.enable=True` and `processor is None` (text-only), the builder is created and CT is enabled. When processor is present (multimodal), CT falls back to legacy path with a warning.

```python
# In AgentLoopBase.__init__():
self.continuous_token_builder = None
self.enable_continuous_token = False
continuous_token_config = self.rollout_config.multi_turn.continuous_token
if continuous_token_config.enable and self.processor is None:
    model_config = self.config.actor_rollout_ref.model
    self.continuous_token_builder = create_continuous_token_builder(
        self.tokenizer,
        model_family=continuous_token_config.model_family,
        model_path=model_config.path,
        tokenizer_name_or_path=model_config.tokenizer_path,
        chat_template_kwargs=self.apply_chat_template_kwargs,
    )
    self.enable_continuous_token = True
    self.system_prompt = None  # CT doesn't use legacy removable system prompt
else:
    if continuous_token_config.enable and self.processor is not None:
        logger.warning("Continuous Token enabled but processor set; falling back to legacy multimodal path.")
    processing_class = self.processor if self.processor is not None else self.tokenizer
    self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
```

**Three CT hook methods** added to `AgentLoopBase`:

1. **ct_build_initial_tokens()**: Builds initial prompt via CT builder's `build_initial_tokens()`. Returns `_cap_text_prompt_length(prompt_ids)`.

2. **ct_merge_assistant_token()**: Appends model-generated assistant tokens via `append_assistant_tokens()` + `align_response_metadata()`.

3. **ct_merge_non_assistant_msg()**: Merges appended tool/user/system messages via `merge_tokens()` + `align_response_metadata()`.

All three methods run in executor threads (`loop.run_in_executor()`) to avoid blocking the async event loop.

**Why system_prompt=None**: CT doesn't use the legacy "removable system prompt" mechanism. The legacy path extracts the system prompt length and strips it from incremental message tokenization. CT's suffix diff approach doesn't need this — it renders prefix and suffix separately and computes the delta. Setting `system_prompt=None` prevents the legacy system prompt from being injected.

### 3.2 SingleTurnAgentLoop Integration (`verl/experimental/agent_loop/single_turn_agent_loop.py`, +28/-10)

```python
use_continuous_token = self.enable_continuous_token and not multi_modal_data
if use_continuous_token:
    prompt_ids = await self.ct_build_initial_tokens(messages)
else:
    prompt_ids = await self.apply_chat_template(messages, ...)

# After generation:
if use_continuous_token:
    merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
        prompt_ids, output.token_ids, [], [] if output.log_probs else None,
        assistant_logprobs=output.log_probs if output.log_probs else None,
    )
    response_ids = merge_result.token_ids[-len(response_mask):]
    prompt_ids = merge_result.token_ids[:len(merge_result.token_ids) - len(response_mask)]
else:
    response_ids = output.token_ids
    response_mask = [1] * len(output.token_ids)
    response_logprobs = output.log_probs
```

**For single-turn**: The CT path produces the same result as legacy — `response_mask = [1] * len(token_ids)` with logprobs from the model. The only difference is that `prompt_ids + response_ids = merge_result.token_ids` (a single MergeResult), which is structurally the same as `prompt_ids += response_ids`.

**Guard**: `use_continuous_token = self.enable_continuous_token and not multi_modal_data`. This ensures CT is not used for multimodal samples (processor not yet supported).

### 3.3 ToolAgentLoop Integration (`verl/experimental/agent_loop/tool_agent_loop.py`, +83/-18)

**State machine with CT**:

```
PENDING → ct_build_initial_tokens() → GENERATING
GENERATING → server.generate() → ct_merge_assistant_token() → extract_tool_calls → _build_assistant_message()
  → if tool_calls: PROCESSING_TOOLS
  → else: TERMINATED
PROCESSING_TOOLS → tool execution → ct_merge_non_assistant_msg() → GENERATING
  → if response_length exceeded: TERMINATED
```

**_handle_pending_state**: Uses `ct_build_initial_tokens()` when CT is enabled.

**_handle_generating_state**: Uses `ct_merge_assistant_token()` to append model-generated tokens and align metadata. Then extracts tool calls and builds assistant message for CT context.

**_handle_processing_tools_state**: Uses `ct_merge_non_assistant_msg()` to merge tool/user responses. Falls back to legacy per-model formatting (gpt-oss, gemma4) when CT is disabled.

**_build_assistant_message()**: New method that reconstructs the assistant message dict from model output for CT context. Creates `{"role": "assistant", "content": content, "tool_calls": [...]}` with proper `OpenAIFunctionCallSchema` formatting. This message is appended to `agent_data.messages` for use in the next turn's CT context.

**Tool call ID preservation**: The diff adds `tool_call_id` propagation from `tool_call.tool_call_id` to the tool response message. This is required for CT's `_synthetic_assistant_for_tools()` to correctly match tool responses to their preceding assistant tool calls.

### 3.4 Tool Parser Architecture (`verl/experimental/agent_loop/tool_parser.py`, +277 lines)

**8 tool parsers** added/extended in this PR:

| Parser | Format | stop_token_ids | Notes |
|--------|--------|----------------|-------|
| `hermes` | `<|tool_calls_begin|>JSON<|tool_calls_end|>` | None | Existing, simplest |
| `gpt-oss` | Harmony `<|start|>functions.name<|channel|>...` | None | Strips COT, keeps special tokens |
| `qwen3_coder` | XML `<function=name><parameter=...>` | None | Type-aware parameter conversion |
| `glm` | `observation name<|param|>value</param|>` | `observation` id | Strips thinking markers |
| `seed` | `<seed:tool_call><function=name>...` | None | ByteDance Seed XML format |
| `minimax` | `<minimax:tool_call><invoke name=...>` | None | MiniMax XML invoke format |
| `kimi` | Special tokens `<|tool_call_begin|>...` | `<|im_end|>` id | Name inference from schema |
| `gemma4` | `<|tool_call>call:name{key:value}` | `<tool_call|>` id | Gemma4 pipe-delimited args |

**Registry pattern**: `ToolParser._registry` dict with `@ToolParser.register("name")` decorator. Accessed via `ToolParser.get_tool_parser(name, tokenizer)`.

**stop_token_ids property**: Some parsers need explicit stop tokens because the model doesn't emit EOS after tool calls. Gemma4 continues generating after `<tool_call|>` without EOS, so that token must be a stop token. GLM stops at `observation`. Kimi stops at `<|im_end|>`. These are injected into `sampling_params["stop_token_ids"]` during generation.

**KimiToolParser name inference**: When the model emits a tool name that doesn't exactly match a known tool, `_infer_tool_name()` tries:
1. Exact match against tool names
2. Substring match (normalized)
3. Schema match by required parameter keys
4. Fallback to raw name

This is a sophisticated feature for handling model hallucinations in tool names.

**Qwen3XMLToolParser parameter type conversion**: `_parse_xml_function_call()` converts parameter values to the correct types based on the tool schema:
- int/uint/long → `int()`
- float/number → `float()`
- bool → `True`/`False`
- string → string
- object/dict → `json.loads()` then `ast.literal_eval()` then fallback string

Uses `ast.literal_eval()` instead of `eval()` for safety (untrusted model output).

### 3.5 Chat Template Checker (`scripts/chat_template_checker.py`, 671 lines)

A standalone validation tool that runs mock trajectories through two layers:

1. **Raw template prefix diagnostics**: Checks whether applying the raw chat template to a prefix produces token IDs that remain a prefix after later messages are rendered. Failures are warnings (CT doesn't strictly require global append-only).

2. **Production-shaped CT builder checks**: Incrementally rebuilds the runtime token stream turn by turn with CT logic, compares final assembled IDs with direct chat template rendering of the complete message list.

**Usage**:
```bash
python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B
python scripts/chat_template_checker.py --model zai-org/GLM-4.7-Flash --allow-download
python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B --template /path/to/chat_template.jinja
```

This is a critical tool for validating new model integrations before using CT in production.

### 3.6 Mock Trajectories (`verl/utils/test_utils/mock_trajectories.py`, 529 lines)

Structured test data for CT-vs-legacy comparison:

```python
@dataclass(frozen=True)
class TrajectoryStep:
    assistant: dict[str, Any]
    appended_messages: tuple[dict[str, Any], ...] = ()

@dataclass(frozen=True)
class SingleTurnTrajectory:
    name, description, raw_prompt, assistant_response, expected_num_turns

@dataclass(frozen=True)
class ToolAgentTrajectory:
    name, description, raw_prompt, steps, tools, max_parallel_calls
```

**TRAJECTORIES dict**: Contains model-specific trajectories for Qwen, GLM, Kimi, Seed, MiniMax, MiMo, Nemotron families — 9+ models with single-turn and tool-agent scenarios.

**Reviewer suggestion (wuxibin89)**: Move test utility from `verl/utils/test_utils/` to `tests/` folder. Author hasn't addressed this yet.

---

## 4. RTX 4090 GRPO Relevance Analysis

### 4.1 Short-Term: Not Needed for Standard GRPO

Standard GRPO on RTX 4090 is single-turn: prompt -> response -> reward -> update. The CT mechanism is designed for multi-turn agentic rollout where the model calls tools, receives responses, and continues generating. For single-turn GRPO:

- `SingleTurnAgentLoop` with `enable_continuous_token=False` works correctly
- Even with CT enabled, single-turn behavior is functionally identical (mask=[1]*len, same logprobs)
- No performance or memory impact on single-turn

**Recommendation**: Leave CT disabled for standard GRPO on RTX 4090. The overhead of CT builder initialization and merge calls is unnecessary for single-turn.

### 4.2 Medium-Term: Valuable for Agent RL and Multi-Turn GRPO

As verl's agent loop capabilities mature (ToolAgentLoop, custom agent loops), multi-turn GRPO becomes feasible:

- **Tool-using agents**: Model calls tools (search, code execution), receives results, continues reasoning
- **Multi-turn dialogue**: Extended conversations with reward at each turn
- **Agentic RL**: The model learns tool selection and usage through RL

CT is essential for correct multi-turn GRPO because:
1. Legacy path silently corrupts token boundaries (Qwen newline, GLM observation, MiniMax TemplateError)
2. Incorrect response_mask alignment means wrong loss computation
3. BPE re-encoding changes token sequences — training sees different tokens than inference produced

### 4.3 Specific RTX 4090 Model Considerations

| Model | CT Adapter | RTX 4090 Viability | Boundary Bug Impact |
|-------|-----------|-------------------|---------------------|
| Qwen2.5-7B-Instruct | QwenCTBuilder | VIABLE (7B fits in 24GB with ZeRO-2+CPU_Adam) | `<|im_end|>\n` missing newline — silent corruption in multi-turn |
| Qwen3-8B | QwenCTBuilder | VIABLE (MoE A3B variant) | Same `<|im_end|>\n` issue |
| Qwen3-0.6B/1.7B | QwenCTBuilder | IDEAL (small, fast, testing) | Same issue, easier to debug |
| GLM-4.7-Flash | GLMCTBuilder | VIABLE (small enough) | `observation` boundary duplicate — wrong loss mask |
| MiniMax-M2 | MiniMaxCTBuilder | NEEDS TESTING | TemplateError without CT — crash on legacy path |
| Gemma4-27b | Gemma4CTBuilder | TOO LARGE (27B > 24GB) | Not viable on single RTX 4090 |

**Qwen family is most relevant**: Qwen2.5/Qwen3 are the top GRPO candidates on RTX 4090. The `<|im_end|>\n` boundary bug would silently corrupt multi-turn training. CT fixes this correctly.

### 4.4 Memory Impact

CT has negligible direct memory impact on RTX 4090:
- Token sequences are the same length (CT fixes boundary tokens, not adds excess)
- `ContinuousTokenBuilder` is a lightweight object holding tokenizer reference + a few token IDs
- No additional GPU memory (all operations are CPU-side token manipulation)
- `response_mask` and `response_logprobs` are Python lists (CPU memory, not GPU tensors)

**Indirect impact**: If CT correctly handles boundaries, the training side sees exactly the tokens the model produced — no wasted loss computation on structural tokens — slightly more efficient training (marginal for single-turn).

### 4.5 Compute Impact

CT adds CPU-side compute for:
- `build_initial_tokens()`: one `apply_chat_template()` call (same as legacy)
- `tokenize_incremental_messages()`: multiple `apply_chat_template()` calls for suffix diff (2 calls per group + 2 for generation prompt)
- `merge_tokens()`: suffix diff + boundary adjustment
- `ct_align_response_metadata()`: list manipulation for mask/logprob alignment

**On RTX 4090**: These are all CPU operations running in executor threads (not blocking GPU). The overhead is:
- Single-turn: ~2 extra `apply_chat_template()` calls — negligible
- Multi-turn: ~4-6 extra calls per turn — still CPU-bound, not GPU-bound

**Not a bottleneck**: GPU generation time dominates. CT's CPU overhead is unlikely to be measurable.

---

## 5. Integration with Existing verl Infrastructure

### 5.1 V1 Trainer Architecture (register_trainer pattern)

The V1 unified trainer uses `register_trainer()` with three types:
- `sync` — synchronous rollout+training (RTX 4090 best path)
- `colocate_async` — rollout and training on same GPUs, async
- `separate_async` — rollout and training on different GPUs (NOT viable dp=1 on RTX 4090)

CT integration is at the `AgentLoopBase` level, which is the rollout-side abstraction. The trainer doesn't need to know about CT — it receives `AgentLoopOutput` with correctly aligned `response_mask` and `response_logprobs`.

**Compatibility matrix**:

| Trainer Type | CT Compatible | Notes |
|-------------|--------------|-------|
| sync | YES | CT runs during rollout phase, no interaction with training |
| colocate_async | YES | CT runs in agent loop worker, sleep/wake is weight sync only |
| separate_async | YES | CT is purely rollout-side, no cross-process dependency |

**RTX 4090 best path**: `trainer_sync` with `bypass_mode=True`. CT doesn't change this recommendation — it's a rollout-side feature.

### 5.2 Sleep/Wake Architecture (verl HYBRID mode)

The HYBRID sleep/wake architecture has two levels:
- `sleep_level=1`: LoRA adapter offload (tags=["kv_cache"]) — 80x payload reduction
- `sleep_level=2`: full weight offload (tags=["kv_cache","weights"]) — full re-transfer

**CT + sleep/wake interaction**: Different lifecycle boundaries:

- **CT boundary**: within a single rollout step, between turns of the same trajectory
- **Sleep/wake boundary**: between rollout steps, when weights are updated

**No direct conflict**: CT's runtime token stream is ephemeral per rollout step. Sleep/wake transfers weights, not token streams. The token stream is built fresh each step and discarded after.

**Potential concern**: If sleep/wake causes KV cache invalidation (like SGLang #28676 MoE cache clobber), the rollout engine's internal state may be corrupted. But CT's token stream is maintained by the agent loop (Python CPU memory), not the inference engine. Even if the engine's KV cache is corrupted after weight reload, the agent loop's assembled token stream is unaffected.

**Pattern family connection**: CT's append-only invariant is the same pattern family as:
- SGLang #28679 (GDN intermittent degeneracy)
- SGLang #28676 (MoE cache clobber)
- vLLM-Ascend #10684 (DSA Hadamard all-zero)
- vLLM #46118 (MTP+grammar FSM conflict)

All are "state lifecycle mismatch" patterns where stale state is used after a lifecycle boundary. CT explicitly addresses this at the tokenization layer.

### 5.3 Weight Sync Mechanism

The weight sync mechanism (ZMQ IPC for SGLang, NCCL for multi-GPU) transfers updated weights from trainer to rollout engine. CT doesn't interact with weight sync:

- Weight sync operates on model parameters (GPU tensors)
- CT operates on token sequences (CPU Python lists)
- Different data planes, different lifecycle boundaries

**Delta weight sync (#6794)**: Reduces payload by ~100x by sending only LoRA deltas. CT doesn't affect this — token sequences aren't part of weight sync payloads.

### 5.4 DataProto and Training Data Flow

CT's output is an `AgentLoopOutput` with:
- `prompt_ids`: merged token sequence (prompt + all turns)
- `response_ids`: response portion (masked by response_mask)
- `response_mask`: aligned mask (1 for assistant, 0 for non-assistant/boundary)
- `response_logprobs`: aligned logprobs (model logprobs for assistant, 0.0 for non-assistant)

This flows into `DataProto` -> trainer -> loss computation. The trainer uses `response_mask` to select loss-bearing tokens.

**Potential issue**: If `response_logprobs` contains `0.0` entries for non-assistant tokens (rather than `None`), downstream code that computes log-prob ratios may include these zeros. Need to verify that the trainer correctly handles `logprobs=0.0` as "no logprob" rather than "logprob of 0" (= log(1) = probability 1.0 — obviously wrong).

**Recommendation**: Verify that `ct_align_response_metadata()`'s `[0.0] * appended_token_count` for non-assistant tokens is handled correctly by the trainer's log-prob computation. The trainer should skip log-prob computation for `response_mask[i] == 0`, which naturally excludes these entries.

### 5.5 FSDP Weight Sync and LoRA Lifecycle

CT doesn't interact with FSDP weight summon/release or LoRA lifecycle. These are trainer-side mechanisms on model parameters. CT is rollout-side only.

**Per-unit LoRA summon (#6512)**: Dynamic FSDP unit discovery replaces 8 hard-coded prefixes, reducing peak memory from 60 to 6-8 GiB. CT doesn't affect this — it's about token sequences, not model parameters.

---

## 6. Potential Concerns and Recommendations

### 6.1 CRITICAL: ValueError on Invalid Tool Call Arguments

**Location**: `tool_agent_loop.py` `_build_assistant_message()`

```python
if has_decode_error:
    raise ValueError(
        f"Invalid tool call arguments for '{tool_call_name}': expected a JSON object string, "
        f"got {tool_call_arguments!r}"
    )
```

**Problem**: In RL training, models frequently generate malformed JSON during exploration. This `raise ValueError` will crash the entire rollout loop, terminating the training step. This is catastrophic for training stability.

**Severity**: HIGH for RTX 4090 GRPO. During early training epochs, the model will produce invalid tool calls regularly. Each crash wastes a full rollout step.

**Gemini Code Assist flag**: The bot flagged this as high priority and recommended: "In RL training, models frequently generate malformed JSON during exploration. It is highly recommended to handle this gracefully (e.g., by logging a warning and proceeding with empty/partial arguments) so that the downstream tool execution can fail gracefully and return the error message back to the model, allowing it to learn from the mistake without crashing the run."

**Recommendation**: Replace with graceful handling:
```python
if has_decode_error:
    logger.warning("Invalid tool call arguments for '%s': ...", tool_call_name, tool_call_arguments)
    # Proceed with empty/partial arguments -> tool execution fails -> model learns from mistake
```

This follows the RL principle: let the model make mistakes and learn from the reward signal, not crash the training loop.

### 6.2 MEDIUM: Multimodal Not Yet Supported

**Status**: WIP per author's response to reviewer (wuxibin89).

**Impact on RTX 4090**: For GRPO with VLMs (Qwen2.5-VL, etc.), CT cannot be used yet. The fallback to legacy `apply_chat_template()` is automatic when `processor is not None` or `multi_modal_data` is present.

**Concern**: The `SingleTurnAgentLoop` guard `use_continuous_token = self.enable_continuous_token and not multi_modal_data` means that if a sample has mixed text+image data, CT is silently disabled for that sample. In a batch with mixed modalities, some samples would use CT and others legacy — potentially causing inconsistent tokenization within the same batch.

**Recommendation**: Leave CT disabled when using VLMs. When multimodal CT support lands, validate thoroughly.

### 6.3 MEDIUM: Reviewer Suggestion to Enable by Default

**wuxibin89** (collaborator) suggests: "Should we enable it by default and drop the old implementation? For single_turn_agent_loop and tool_agent_loop, we can use Continuous Token by default."

**gxlvera** (author) responds: Agrees, considering removing `enable` flag for built-in loops. For custom agent loops, it should remain optional.

**RTX 4090 impact**: If CT becomes default for `SingleTurnAgentLoop` and `ToolAgentLoop`, RTX 4090 GRPO configs would automatically use CT. This is fine for single-turn (functionally identical) and beneficial for multi-turn (fixes boundary bugs). But:
- The `ContinuousTokenConfig` validation in `__post_init__` would need updating
- The YAML config would need `enable: True` by default
- The multimodal fallback logic must be robust

**Recommendation**: Support this direction. Enabling CT by default for built-in loops is the right long-term choice — it fixes known bugs and is backward-compatible for single-turn. Keep `enable` flag for custom agent loops.

### 6.4 LOW: Test Utility Location

**wuxibin89** suggests moving `verl/utils/test_utils/mock_trajectories.py` to the `tests/` folder.

**Current location**: `verl/utils/test_utils/` — inside the main package, which means it ships with verl. Test utilities shouldn't be in the main package.

**Recommendation**: Move to `tests/utils/mock_trajectories.py` or `tests/experimental/agent_loop/continuous_token/mock_trajectories.py`.

### 6.5 LOW: Suffix Diff Validation May Fail on Edge Cases

**Location**: `ContinuousTokenBuilder.render_delta_token_id()`

```python
if full_token_ids[:len(prefix_token_ids)] != prefix_token_ids:
    raise ValueError(f"Continuous Token token-id suffix diff failed for roles: {roles}")
```

**Concern**: This strict prefix check may fail on tokenizers where BPE merging or position-dependent rendering causes minor prefix instability even in synthetic context. The `_NonPrefixStableTokenizer` test case demonstrates this scenario.

**Current handling**: Error is raised immediately. Correct — if suffix diff fails, incremental tokens cannot be safely extracted, and retokenizing the full conversation violates TITO invariant.

**Recommendation**: For edge-case tokenizers, users should specify `model_family=default` and verify prefix-stability. If not prefix-stable, they need a custom builder subclass. The `chat_template_checker.py` tool validates this.

### 6.6 LOW: GPT-OSS Bypasses Suffix Diff for Tool Groups

**Location**: `GptOssContinuousTokenBuilder._tokenize_tool_group()`

This builder bypasses `render_delta_token_id()` entirely for tool groups, directly encoding the response text. Correct for GPT-OSS (Harmony format doesn't go through chat template), but suffix diff validation doesn't cover this path.

**Recommendation**: No action needed. GPT-OSS format is well-defined and doesn't need suffix diff validation.

### 6.7 RTX 4090-Specific Recommendations

1. **For standard GRPO (single-turn)**: Leave CT disabled. No benefit, no risk.
2. **For multi-turn agent RL**: Enable CT with `model_family=auto`. Essential for correct token boundaries, especially on Qwen models.
3. **For VLM GRPO**: Leave CT disabled until multimodal support lands.
4. **Configuration for Qwen3-8B multi-turn**:
   ```yaml
   actor_rollout_ref:
     rollout:
       multi_turn:
         enable: True
         continuous_token:
           enable: True
           model_family: auto
         format: hermes
   ```
5. **Monitor the ValueError crash risk**: If using ToolAgentLoop, patch `_build_assistant_message()` to handle invalid JSON gracefully before running production training.
6. **Verify response_logprobs alignment**: Ensure trainer's log-prob computation correctly skips `0.0` entries where `response_mask[i] == 0`.

### 6.8 Pattern Family Connections

CT addresses the **State Lifecycle Mismatch** pattern family at the tokenization level:

| Pattern | Framework | Issue | CT Relevance |
|---------|-----------|-------|-------------|
| Token boundary mismatch | verl legacy | Qwen/MiniMax newline missing | DIRECTLY FIXED by CT |
| Token boundary duplicate | verl legacy | GLM observation duplication | DIRECTLY FIXED by CT |
| Template error on tool response | verl legacy | MiniMax/Nemotron/Llama | DIRECTLY FIXED by CT synthetic context |
| MoE cache clobber | SGLang #28676 | MXFP8 shuffle cache on weight reload | SAME pattern family, different layer |
| GDN intermittent degeneracy | SGLang #28679 | decode degeneracy over uptime | SAME pattern family, different layer |
| DSA Hadamard all-zero | vLLM-Ascend #10684 | constant buffer lost in sleep/wake | SAME pattern family, different layer |
| MTP+grammar FSM conflict | vLLM #46118 | FSM state not reset between calls | SAME pattern family, different layer |
| Triton rotary NaN | Megatron #5317 | in-place kernel bypasses autograd | SAME pattern family, different layer |

CT is the tokenization-layer solution. The other instances are at the inference engine layer (KV cache, routing, attention). Both layers need state lifecycle awareness at transition boundaries.

---

## 7. Test Coverage Analysis

### 7.1 CPU Unit Tests (`tests/utils/test_continuous_token_on_cpu.py`, 723 lines)

Extensive coverage with dummy tokenizers:
- `ContinuousTokenBuilder`: build_initial_tokens, tokenize_incremental_messages, merge_tokens, append_assistant_tokens
- `QwenContinuousTokenBuilder`: newline insertion after `<|im_end|`
- `GLMContinuousTokenBuilder`: boundary removal for `observation`/`user`
- `MiniMaxContinuousTokenBuilder`: newline insertion after `[e~[`
- `Gemma4ContinuousTokenBuilder`: `<|tool_response>` insertion
- `GptOssContinuousTokenBuilder`: custom tool format
- `ct_align_response_metadata()`: mask/logprob alignment for all MergeResult variants
- `ContinuousTokenModelFamily`: auto inference, registry, factory
- Edge cases: missing special tokens, list-returning convert_tokens_to_ids, non-prefix-stable tokenizer

### 7.2 Tool Call ID Tests (`tests/experimental/agent_loop/test_tool_call_id_on_cpu.py`, 175 lines)

Covers `tool_call_id` preservation from tool parser output to assistant `tool_calls[].id` to tool response messages.

### 7.3 Agent Loop Smoke Tests

Added to both vLLM and SGLang CI workflows:
```yaml
ENABLE_CONTINUOUS_TOKEN=1 ROLLOUT_NAME=vllm pytest tests/experimental/agent_loop/test_basic_agent_loop.py::test_single_turn
ENABLE_CONTINUOUS_TOKEN=1 ROLLOUT_NAME=vllm pytest tests/experimental/agent_loop/test_basic_agent_loop.py::test_tool_agent
ENABLE_CONTINUOUS_TOKEN=1 ROLLOUT_NAME=sglang pytest ... (same tests)
```

**No training E2E test**: PR body states "CT is primarily a rollout/agent-loop tokenization and metadata-alignment change, so CI focuses on CPU unit tests + vLLM/SGLang agent-loop E2E smoke tests."

**Gap**: No multi-turn GRPO training E2E test. Acceptable for a rollout-side PR, but future work should validate CT's response_mask alignment in actual training.

### 7.4 Chat Template Checker (`scripts/chat_template_checker.py`, 671 lines)

Standalone validation tool for CT behavior with any model tokenizer. Not part of CI but available for manual validation.

---

## 8. Summary and Assessment

### 8.1 Design Quality: EXCELLENT

Well-architected with clear separation of concerns:
- Core invariant (append-only, TITO) at the builder level
- Model-specific boundary handling via inheritance (not config flags)
- Synthetic context construction handles tool-response dependency
- MergeResult + ct_align_response_metadata() ensures correct mask/logprob alignment
- Suffix diff validation catches template instability
- Auto-detection with conservative fallback (DEFAULT for unknown models)

### 8.2 Code Quality: GOOD

- Clean, well-documented code with extensive type hints
- Frozen dataclasses for immutable results
- Comprehensive test coverage (723-line CPU unit test file)
- Edge case handling (missing tokens, list-returning APIs, non-prefix-stable tokenizers)
- Reviewer feedback being addressed

### 8.3 Bugs/Concerns: 2 items

1. **CRITICAL**: `ValueError` crash on invalid tool call arguments in RL exploration — must be graceful handling
2. **MEDIUM**: Multimodal not yet supported — fallback is automatic but inconsistent within mixed batches

### 8.4 RTX 4090 Assessment: 3 stars (medium-term value)

- Short-term: not needed for standard GRPO, leave disabled
- Medium-term: essential for multi-turn agent RL, especially Qwen family
- No direct memory/compute impact
- Potential training stability risk from ValueError crash (patch before production use)

### 8.5 Integration Assessment: 5 stars (clean integration)

- No interaction with V1 trainer (rollout-side only)
- No interaction with weight sync (different data plane)
- No interaction with sleep/wake (different lifecycle boundary)
- Same pattern family as known bugs (state lifecycle mismatch) — CT is the tokenization-layer fix

### 8.6 Recommendation: APPROVE with conditions

1. Fix ValueError crash — graceful handling for invalid tool call arguments
2. Move mock_trajectories to tests/ folder
3. Consider enabling CT by default for built-in agent loops (per reviewer suggestion)
4. Add training E2E validation in future PR (response_mask alignment in actual GRPO loss)

---

## 9. Key Source Files Reference

| File | Lines | Key Content |
|------|-------|-------------|
| `verl/utils/continuous_token.py` | 553 | ContinuousTokenBuilder, MergeResult, ct_align_response_metadata, Qwen/MiniMax/GLM/Gemma4/GptOss adapters |
| `verl/utils/continuous_token_wiring.py` | 209 | ContinuousTokenModelFamily enum, registry, auto-detection, factory |
| `verl/experimental/agent_loop/agent_loop.py` | +95/-8 | AgentLoopBase CT hooks (ct_build_initial_tokens, ct_merge_assistant_token, ct_merge_non_assistant_msg) |
| `verl/experimental/agent_loop/tool_agent_loop.py` | +83/-18 | ToolAgentLoop CT integration, _build_assistant_message |
| `verl/experimental/agent_loop/single_turn_agent_loop.py` | +28/-10 | SingleTurnAgentLoop CT integration |
| `verl/experimental/agent_loop/tool_parser.py` | +277 | 8 tool parsers (hermes, gpt-oss, qwen3_coder, glm, seed, minimax, kimi, gemma4) |
| `verl/workers/config/rollout.py` | +12 | ContinuousTokenConfig dataclass |
| `verl/trainer/config/rollout/rollout.yaml` | +14 | YAML config for continuous_token.enable/model_family |
| `tests/utils/test_continuous_token_on_cpu.py` | 723 | CPU unit tests |
| `tests/experimental/agent_loop/test_tool_call_id_on_cpu.py` | 175 | Tool call ID preservation tests |
| `scripts/chat_template_checker.py` | 671 | Standalone validation tool |
| `verl/utils/test_utils/mock_trajectories.py` | 529 | Structured test trajectories |

---

## 10. Related PRs and Issues

| Reference | Relationship | Notes |
|-----------|-------------|-------|
| RFC #6719 | Design document | Extensive motivation, 4 pitfalls, counterexamples |
| PR #6720 | Previous stacked PR (builder core) | Now merged into #6779 |
| PR #6721 | Previous stacked PR (agent loop integration) | Now merged into #6779 |
| SGLang #28676 | Same pattern family | MoE cache clobber on weight reload |
| SGLang #28679 | Same pattern family | GDN intermittent degeneracy |
| vLLM-Ascend #10684 | Same pattern family | DSA Hadamard all-zero after sleep/wake |
| vLLM #46118 | Same pattern family | MTP+grammar FSM state lifecycle mismatch |
| Megatron #5317 | Same pattern family | Triton rotary in-place bypasses autograd |
| verl #6794 | Weight sync | Delta weight sync — different data plane, no interaction |
| verl #6512 | Per-unit LoRA summon | Different layer (FSDP params), no interaction |
| verl V1 trainer | Architecture | register_trainer pattern — CT is rollout-side, compatible |
| DeepSpeed #8061 | NaN bug | overlap_comm+compile — different bug class |
