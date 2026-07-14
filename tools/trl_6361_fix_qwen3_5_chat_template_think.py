"""
TRL #6361: Qwen3.5 training chat template duplicate <think> tags

## Root Cause

Qwen3.5's generation prompt ALWAYS emits `<think>\n` (unlike Qwen3 which only
emits it for no-think mode). When the model generates a response, the raw
output starts with reasoning text, NOT `<think>`:

```
Generation prompt:  <|im_start|>assistant\n<think>\n
Model generates:    Let me calculate 2+2...\nThe answer is 4\n</think>\n\n4
Stored response:    Let me calculate 2+2...\nThe answer is 4\n</think>\n\n4
```

The training template (`qwen3_5_think_training.jinja`) wraps ALL assistant
messages with `<think>\n...\n</think>\n\n`. The thinking detection checks:

```jinja
{%- if '<think>' in content and '</think>' in content %}
```

When processing the stored response:
- `'<think>' in content` → FALSE (no `<think>` tag — it was in the prompt)
- `'</think>' in content` → TRUE
- Condition fails → `reasoning_content = ''`

Rendering produces an **empty think block**:
```
<think>\n\n</think>\n\nLet me calculate...\nThe answer is 4\n</think>\n\n4
```

The issue reporter sees: `<think> </think> [reasoning] </think> [output]`

## Fix

Add an `{%- elif '</think>' in content %}` branch to handle the case where
`<think>` was from the generation prompt and `</think>` in the content closes it:

```diff
--- a/trl/chat_templates/qwen3_5_think_training.jinja
+++ b/trl/chat_templates/qwen3_5_think_training.jinja
             {%- if '<think>' in content and '</think>' in content %}
                 {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                 {%- set content = content.split('</think>')[-1].lstrip('\n') %}
+            {%- elif '</think>' in content %}
+                {#- <think> was from generation prompt; </think> in content closes it #}
+                {%- set reasoning_content = content.split('</think>')[0].lstrip('\n') %}
+                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
             {%- endif %}
```

This fix should also apply to the Qwen3.5 NO-think training template and the
base Qwen3.5 think template for consistency. The same pattern may affect
Qwen3.6 templates.

## Same Fix for qwen3_5_nothink_training.jinja

The no-think flavor has the same issue (same template structure).

```diff
--- a/trl/chat_templates/qwen3_5_nothink_training.jinja
+++ b/trl/chat_templates/qwen3_5_nothink_training.jinja
             {%- if '<think>' in content and '</think>' in content %}
                 {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                 {%- set content = content.split('</think>')[-1].lstrip('\n') %}
+            {%- elif '</think>' in content %}
+                {%- set reasoning_content = content.split('</think>')[0].lstrip('\n') %}
+                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
             {%- endif %}
```

## Same Fix for qwen3_5_think.jinja (base template)

The base template uses `{%- if '</think>' in content %}` (single tag check)
which is MORE permissive but ALSO wraps content with think tags. However,
the base template only adds think tags for non-last-query messages:

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

The base template's condition is correct for its use case because the
`else` branch skips think wrapping entirely. The training template ALWAYS
wraps, which causes the bug.

## Testing

1. Generate a response with Qwen3.5 model from a `<think>\n` prompt
2. Store response_text as assistant content (without the prompt prefix)
3. Call apply_chat_template with the training template on
   [user, assistant, user] conversation with add_generation_prompt=True
4. Before fix: `<think>\n\n</think>\n\n[reasoning]\n</think>\n\n[output]`
5. After fix: `<think>\n[reasoning]\n</think>\n\n[output]`

## Issue Reference

https://github.com/huggingface/trl/issues/6361
Opened 2026-07-11, 0 comments (untriaged)
"""
