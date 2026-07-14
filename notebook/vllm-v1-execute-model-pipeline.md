# vLLM V1 GPUModelRunner: execute_model Pipeline

## Overview
The `GPUModelRunner.execute_model` (lines 4096–4464, ~336 KB file) is the core generation loop used by GRPO rollout engines. Understanding this pipeline is critical for optimizing GRPO generation performance.

## 12-Step Pipeline

| Step | Method | Purpose | GRPO Relevance |
|------|--------|---------|----------------|
| 1 | `_determine_batch_execution_and_padding` | Decide CUDA graph vs eager + compute padding | CUDA Graph = 0.009ms vs eager = 0.068ms (per SGLang #31190) |
| 2 | `dispatch_cudagraph` | Route to cached CUDA graph if applicable | Deterministic latency critical for rollout timing |
| 3 | `_get_slot_mappings` | Map token positions to KV cache slots | KV cache management for long rollouts |
| 4 | `_prepare_inputs` | Build input tensors (token IDs, positions, block tables) | Multi-step generation setup |
| 5 | `_build_attention_metadata` | Construct per-layer attention metadata | GDN/MLA attention metadata (relevant to #48613) |
| 6 | `_execute_mm_encoder` | Run multimodal encoder for vision inputs | Qwen-VL GRPO support |
| 7 | `_gather_mm_embeddings` | Collect multimodal embeddings | Vision-language GRPO |
| 8 | `_preprocess` | Final preprocessing before model forward | Sanity checks on padded inputs |
| 9 | `_model_forward` | Execute the actual Transformer forward pass | Core compute — FP8, GDN, MLA all matter here |
| 10 | `_sample` | Sample next tokens from logits | Temperature, top_p for generation diversity |
| 11 | `_update_states` | Update scheduler state | Streaming request management |
| 12 | `_update_states_after_model_execute` | Post-execution bookkeeping | KV cache, scheduling metadata |

## Mixin Architecture
`GPUModelRunner` composes 3 mixins via multiple inheritance:
- **ec_connector_mixin**: Encoder-decoder connector output
- **kv_connector_mixin**: KV cache connector (disaggregated prefill/decode)
- **lora_mixin**: LoRA adapter support

`ExecuteModelState` dataclass carries fields for mixin communication (ec_connector_output, spec_decode_metadata, hidden_states, slot_mappings).

## CUDA Graph Pipeline
1. **Profile**: `profile_cudagraph_memory` — determines max batch size per input shape
2. **Capture**: `capture_model` → `_warmup_and_capture` → `_capture_cudagraphs`
3. **Replay**: `dispatch_cudagraph` — checks cached graph for (batch_size, seq_len, padding)

Each unique (batch_size, seq_len) tuple gets its own captured graph. Fixed shapes only.

## Spec Decode Batch Expansion
Batch expands from `N` sequences → `N × (draft_len + 1)` slots. Each slot gets position ID, block mapping, attention metadata. After forward pass, `correct_spec_decode_token_counts` reconciles accepted/rejected tokens.

## GRPO Relevance
- **Rollout generation**: Each GRPO step calls this pipeline for `num_generations` per prompt
- **CUDA Graph**: Critical for reducing per-token latency (especially with FP8 KV cache on SM89)
- **LoRA mixin**: Enables GRPO+LoRA without separate model instances
- **Batch expansion pattern**: Same concept as GRPO's multiple generations verification

## Key Files
- `vllm/v1/worker/gpu_model_runner.py` (~7550 lines, ~336 KB) — main file
- `vllm/v1/worker/ec_connector_model_runner_mixin.py` — encoder-decoder mixin
- `vllm/v1/worker/kv_connector_model_runner_mixin.py` — KV connector mixin
- `vllm/v1/worker/lora_model_runner_mixin.py` — LoRA mixin
