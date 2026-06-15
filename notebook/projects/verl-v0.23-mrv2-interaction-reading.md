# verl x vLLM MRv2 Interaction: RTX 4090 GRPO Training Impact

> 2026-06-16 | Source: verl v0.8.0 (2026-06-01), vLLM v0.23.0 (2026-06-12)
> Core: verl has ZERO explicit MRv2 handling; vLLM's oracle auto-activates MRv2 for eligible models
> Risk: BF16 Qwen3 models auto-switch to MRv2 when upgrading to vLLM v0.23.0 -- untested with verl

## 1. The Oracle: How vLLM Decides MRv1 vs MRv2

```
vllm/config/vllm.py: use_v2_model_runner property (3-layer decision):

Layer 1: VLLM_USE_V2_MODEL_RUNNER env var
  → If set (True/False): OVERRIDES everything, no further checks
  → If None (unset): falls through to oracle

Layer 2: _is_default_v2_model_runner_model()
  → model_config.runner_type != "generate" → False (pooling models excluded)
  → architecture NOT in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES → False
  → model_config.is_moe → False (MoE excluded)
  → model_config.is_quantized → False (INT4/GPTQ/FP8 excluded)
  → All pass → True → proceed to Layer 3

Layer 3: _get_v2_model_runner_unsupported_features()
  → Checks Triton availability
  → Checks unsupported features list (processed_logits, raw_logits, SP>1, etc.)
  → If ANY unsupported → log warning → return False (MRv1 fallback)
  → If ALL supported → return True → MRv2 activates!

DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (v0.23.0):
  {LlamaForCausalLM, MistralForCausalLM, Qwen3ForCausalLM}
  → ONLY these 3 dense BF16 architectures auto-get MRv2!
  → All others (DeepseekV2, Qwen2Moe, Gemma, etc.) → MRv1 always
```

## 2. verl's Complete MRv2 Blindness

```
★★★ KEY FINDING: verl has ZERO MRv2 awareness!

Evidence:
  1. Grep VLLM_USE_V2_MODEL_RUNNER across entire verl tree → 0 matches
  2. Grep MRv2/model_runner_v2/ModelRunnerV2 → 0 matches in verl code
  3. gh search issues/PRs on verl-project/verl for MRv2 → 0 results
  4. gh search issues/PRs for VLLM_USE_V2_MODEL_RUNNER → 0 results
  5. verl v0.8.0 release notes → NO MRv2-related changes

What this means:
  → verl does NOT set VLLM_USE_V2_MODEL_RUNNER=0 to force MRv1
  → verl does NOT set VLLM_USE_V2_MODEL_RUNNER=1 to force MRv2
  → verl relies entirely on vLLM's oracle to pick the runner
  → When upgrading to vLLM v0.23.0, eligible models SILENTLY switch to MRv2
  → This is an UNTESTED configuration for verl's RL training pipeline!
```

## 3. verl Launches vLLM: Where MRv2 Decision Happens

```
verl's vLLM async server launch path:

vLLMHttpServer.launch_server()
  → builds CLI args from RolloutConfig
  → sets compilation_config: cudagraph_mode = "FULL_AND_PIECEWISE" (default)
  → sets enable_sleep_mode, enable_chunked_prefill, enable_prefix_caching
  → calls AsyncEngineArgs.from_cli_args(args)
  → calls engine_args.create_engine_config(usage_context=OPENAI_API_SERVER)
  → ★ vllm_config = engine_args.create_engine_config() runs the oracle!
  → vllm_config.use_v2_model_runner is computed here (3-layer decision)
  → AsyncLLM.from_vllm_config(vllm_config=vllm_config) → creates engine
  → GPUWorker.__init__(vllm_config) → self.use_v2_model_runner = vllm_config.use_v2_model_runner
  → GPUWorker.init_worker() → if self.use_v2_model_runner: import GPUModelRunnerV2

★★ The MRv2 oracle runs INSIDE vLLM's config creation, not in verl!
★★ verl has no visibility into which runner vLLM picked!
★★ No logging, no assertion, no config option to control this!
```

## 4. RTX 4090 Model-by-Model MRv2 Impact Matrix

```
| Model                     | Architecture         | Quantized | MoE | MRv2?  | Runner | verl Impact |
|---------------------------|----------------------|-----------|-----|--------|---------|-------------|
| Qwen3-1.7B BF16           | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | AUTO-SWITCH! |
| Qwen3-4B BF16             | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | AUTO-SWITCH! |
| Qwen3-8B BF16             | Qwen3ForCausalLM     | No        | No  | YES★★ | MRv2    | AUTO-SWITCH! |
| Llama-3.1-8B BF16         | LlamaForCausalLM     | No        | No  | YES★★ | MRv2    | AUTO-SWITCH! |
| Mistral-7B BF16           | MistralForCausalLM   | No        | No  | YES★★ | MRv2    | AUTO-SWITCH! |
| Qwen3-8B INT4/GPTQ        | Qwen3ForCausalLM     | Yes★★     | No  | NO★★★ | MRv1    | SAME AS BEFORE |
| Qwen2.5-7B BF16           | Qwen2ForCausalLM     | No        | No  | NO★★★ | MRv1    | SAME AS BEFORE |
| Qwen3-30B-A3B MoE BF16    | Qwen3MoeForCausalLM  | No        | Yes | NO★★★ | MRv1    | SAME AS BEFORE |
| DeepSeek-V2 BF16          | DeepseekV2ForCausalLM| No        | Yes | NO★★★ | MRv1    | SAME AS BEFORE |

★★★ Critical: Only Qwen3, Llama, Mistral BF16 dense models auto-switch to MRv2
★★★ INT4/GPTQ models ALWAYS stay on MRv1 (is_quantized=True blocks MRv2)
★★★ MoE models ALWAYS stay on MRv1 (is_moe=True blocks MRv2)
★★★ Qwen2/Qwen2.5 NOT in DEFAULT_V2 → always MRv1
```

## 5. Sleep/Wake Cycle: MRv2 vs MRv1 Worker-Level Compatibility

```
★★★ Sleep/wake is at GPUWorker level, NOT model runner level!

GPUWorker.sleep(level):
  → Both MRv1 and MRv2 use the SAME sleep() method
  → level=1: allocator.sleep(offload_tags=("weights",)) → offload weights only
  → level=2: allocator.sleep(offload_tags=()) → offload weights + KV cache + all buffers
  → ★ Saves model buffers to CPU before level=2 sleep
  → ★ Same for both runners → COMPATIBLE!

GPUWorker.wake_up(tags):
  → Both MRv1 and MRv2 use the SAME wake_up() method
  → allocator.wake_up(tags) → restore GPU memory by tag
  → Restores model buffers from CPU after level=2 sleep
  → ★ Then calls model_runner.post_kv_cache_wake_up() → runner-specific hook

post_kv_cache_wake_up() DIFFERS:
  MRv1 (gpu_model_runner.py):
    → self.init_fp8_kv_scales() → re-zero KV cache + reset FP8 scales to 1.0
    → ★ Important for FP8 KV on RTX 4090 (though FP8 KV is problematic on SM89)

  MRv2 (gpu/model_runner.py):
    → self.block_tables.init_block_table_layout_tensors() → re-init block table tensors
    → ★ Different hook but same purpose: restore state after wake

★★★ Both hooks are called from the same GPUWorker.wake_up() → COMPATIBLE with verl flow!

verl's HYBRID sleep/wake flow:
  1. engine.sleep(level=1 or level=2) → release GPU memory for FSDP training
  2. FSDP training runs → actor updates weights
  3. engine.wake_up(tags=["weights"]) → restore weights to GPU
  4. update_weights_from_ipc() → sync new weights via ZMQ bucketed transfer
  5. engine.wake_up(tags=["kv_cache"]) → restore KV cache memory
  6. Generation → vLLM produces rollout tokens
  7. Back to step 1

★★ This flow works with both MRv1 and MRv2 because:
  → sleep/wake is worker-level, not runner-level
  → post_kv_cache_wake_up() is a hook that both runners implement
  → No runner-specific sleep/wake logic in verl's code
```

## 6. Weight Sync: verl's Bucketed IPC Transfer with MRv2

```
verl's weight update path (naive/async):

vLLMColocateWorkerExtension.update_weights_from_ipc():
  → BucketedWeightReceiver receives weights via ZMQ IPC
  → Calls _update_weights() per bucket
  → _update_weights() → model.load_weights(weights) for standard weights
  → Or load_quanted_weights() for FP8 models
  → model = self.model_runner.model → ★ same nn.Module regardless of runner!

★★★ Weight sync is COMPATIBLE with MRv2 because:
  1. model.load_weights() operates on the nn.Module, not the runner
  2. Both MRv1 and MRv2 wrap the same underlying PyTorch model
  3. The ZMQ IPC bucketed transfer is at the worker extension level
  4. No runner-specific weight loading logic in verl

★★★ Potential concern: CUDA graph invalidation
  → MRv2 uses Breakable CUDA Graph (BCG) → captured at warmup time
  → After weight update, BCG segments contain OLD weight references
  → vLLM handles this: graphs are invalidated and re-captured on next warmup
  → verl's flow: sleep → wake → update_weights → generation
  → During generation, vLLM re-captures CUDA graphs if weights changed
  → ★ This should work BUT is untested with verl's specific flow!

★★★ Additional concern: compilation_config interaction
  → verl sets cudagraph_mode = "FULL_AND_PIECEWISE" (default)
  → MRv2's BCG replaces FULL_AND_PIECEWISE compilation
  → The compilation_config is processed by vLLM's config creation
  → When MRv2 is active, BCG takes precedence over torch.compile
  → ★ This should be fine but the interaction is not explicitly tested
```

## 7. logprobs_mode: Critical Compatibility Check

```
★★★★ MOST IMPORTANT compatibility concern!

verl's RolloutConfig default:
  logprobs_mode: Optional[str] = "processed_logprobs"

vLLM v0.23.0 MRv2 unsupported features:
  _get_v2_model_runner_unsupported_features():
    → "raw_logits" → UNSUPPORTED → forces MRv1
    → "processed_logits" → UNSUPPORTED → forces MRv1
    → ★ "processed_logprobs" → NOT in unsupported list → MRv2 OK!

★★★ verl's default "processed_logprobs" IS COMPATIBLE with MRv2!
  → This is the standard/default logprobs mode in vLLM
  → MRv2 supports it
  → No oracle fallback triggered

★★★ BUT: if user sets logprobs_mode to "processed_logits" or "raw_logits":
  → MRv2 oracle detects unsupported feature → falls back to MRv1
  → Silent fallback with warning log
  → verl doesn't validate or warn about this!
  → User might not realize they're on MRv1 despite MRv2-eligible model

RTX 4090 GRPO training: verl needs processed_logprobs for:
  → log_prob computation in actor's compute_log_prob()
  → PPO/GRPO advantage calculation requires per-token logprobs
  → "processed_logprobs" = standard mode → MRv2 compatible → OK!
```

## 8. MRv2 Unsupported Features: verl Relevance

```
Features that would block MRv2 activation (if enabled):

| Feature                     | Unsupported by MRv2 | verl Default | Risk |
|-----------------------------|---------------------|--------------|------|
| processed_logprobs          | NO (supported)      | YES (default)| OK★★ |
| processed_logits            | YES                 | NO           | Safe |
| raw_logits                  | YES                 | NO           | Safe |
| SP with TP>1                | YES                 | NO (TP=2)    | ★★★★|
| KV sharing fast prefill     | YES                 | NO           | Safe |
| EC transfer (PD disag)      | YES                 | NO           | Safe |
| Ngram spec decode           | YES                 | NO           | Safe |
| Dynamic spec decode         | YES                 | NO           | Safe |
| Elastic EP                  | YES                 | NO           | Safe |
| LoRA                        | NO (supported★★)   | Optional     | OK★★ |
| Custom logits processors    | YES                 | NO           | Safe |
| Sequence parallelism TP>1   | YES                 | N/A (TP=2)   | ★★★★|

★★★ SP with TP>1: verl defaults to tensor_model_parallel_size=2
  → If SP is enabled alongside TP=2 → MRv2 blocks → falls back to MRv1
  → But SP is NOT default in verl → Safe for default config
  → ★★★ If user enables SP + TP>1 → silent MRv1 fallback!

★★★ LoRA: MRv2 inherits LoRAModelRunnerMixin → SUPPORTED
  → verl's LoRA integration works with MRv2
  → LoRA weight sync via add_lora/remove_lora → runner-independent
```

## 9. Breakable CUDA Graph (BCG) + verl Interaction

```
★★★★★ BCG is MRv2's most impactful change for verl!

BCG (v0.23.0):
  → Replaces torch.compile + Inductor for CUDA graph capture
  → Single capture context → encounters @eager_break_during_capture → break
  → Produces segments: graph.replay + eager_fn → replay sequentially
  → ★ No torch.compile dependency → works on ANY SM architecture (SM89 OK!)

BCG vs verl's compilation_config:
  → verl sets: cudagraph_mode = "FULL_AND_PIECEWISE"
  → MRv2's BCG replaces the PIECEWISE torch.compile path
  → ★ When MRv2 is active, BCG takes over CUDA graph management
  → FULL mode still captured for decode batch sizes (same as MRv1)

BCG + verl weight update:
  → After update_weights_from_ipc(), model weights change
  → BCG segments reference stale weight tensors
  → vLLM invalidates graphs → re-captures on next iteration
  → ★ The weight update → graph invalidation → recapture cycle SHOULD work
  → ★★★ But this cycle is NOT tested in verl's RL training context!

BCG + verl sleep/wake:
  → sleep(level=2) offloads weights → wake restores → weights may be at different addresses
  → CUDA graphs reference specific tensor addresses → need recapture after wake
  → ★ MRv2 handles this: post_kv_cache_wake_up re-initializes state
  → ★★★ But the FULL recapture after wake is NOT verified with verl!

★★★ Risk assessment:
  → BCG + weight update: SHOULD work (vLLM designed for this)
  → BCG + sleep/wake: SHOULD work (standard GPUWorker mechanism)
  → BCG + verl's specific HYBRID training flow: UNTESTED
  → ★★★ Recommend: test BF16 Qwen3-1.7B GRPO with MRv2 before production
```

## 10. verl v0.8.0: MRv2 Status

```
verl v0.8.0 (2026-06-01):
  → Stable vLLM version: v0.20.2 (NOT v0.23.0)
  → ★★★ MRv2 was introduced in v0.23.0 → verl v0.8.0 predates MRv2!
  → No MRv2-specific changes in v0.8.0 release notes
  → No MRv2-related issues/PRs in verl-project/verl

Key v0.8.0 vLLM changes relevant to future MRv2:
  → #6456: Use engine.sleep() instead of collective_rpc("sleep")
    → ★★★ This fix ensures sleep propagates to ALL DP workers
    → ★★★ Critical for MRv2 because MRv2's DP handling differs from MRv1
    → ★★★ Without this fix, sleep wouldn't reach MRv2 workers properly
  → #6091: Split large weights into chunks in NCCL/NIXL checkpoint engine
    → Weight transfer optimization → runner-independent
  → #6373: MooncakeStoreConnector hard-reset on weight update
    → KV store invalidation → runner-independent

★★★ When upgrading verl from v0.20.2 to v0.23.0:
  → #6456's engine.sleep() fix is CRITICAL for MRv2 + DP>1
  → But v0.23.0 already includes this fix (it was in v0.20.x)
  → ★★★ The upgrade path should be smooth for sleep/wake
  → ★★★ But MRv2 itself is new → needs explicit testing!
```

## 11. verl → vLLM MRv2 Interaction Flow (E2E)

```
Step 1: verl config → vLLM CLI args
  RolloutConfig → build_cli_args_from_config() → CLI args dict
  Key args passed: dtype, load_format, enable_sleep_mode, enforce_eager,
    compilation_config (cudagraph_mode=FULL_AND_PIECEWISE),
    quantization, logprobs_mode, tensor_parallel_size, max_model_len

Step 2: vLLM config creation → oracle runs
  AsyncEngineArgs.from_cli_args() → create_engine_config()
  → model_config created (architecture, is_quantized, is_moe determined)
  → use_v2_model_runner property computed (3-layer decision)
  → ★★★ If Qwen3 BF16 dense + no unsupported features → MRv2 = True
  → ★★★ If INT4/GPTQ → MRv2 = False (is_quantized blocks)

Step 3: Engine creation → worker gets runner type
  AsyncLLM.from_vllm_config(vllm_config) → creates EngineCore
  → GPUWorker.init_worker() → self.use_v2_model_runner = vllm_config.use_v2_model_runner
  → ★★★ If True: import GPUModelRunnerV2 → MRv2 path
  → ★★★ If False: import GPUModelRunner → MRv1 path (existing, tested)

Step 4: Model loading → same for both runners
  Both runners call model.load_weights() → same nn.Module
  verl's vLLMColocateWorkerExtension handles weight loading → runner-independent

Step 5: Warmup → DIFFERENT for MRv2!
  MRv1: captures FULL CUDA graphs per batch size + torch.compile PIECEWISE
  MRv2: captures FULL CUDA graphs per batch size + BCG (no torch.compile)
  → ★★★ BCG warmup may take different time → potential latency difference

Step 6: Generation → DIFFERENT execution!
  MRv1: execute_model() → forward + sample in one step → monolithic
  MRv2: execute_model() → forward only → sample_tokens() separately
  → ★★★ MRv2's 2-phase execution enables PP overlap (not relevant for TP=1 RTX 4090)
  → ★★★ FlashInfer sampler activates for temperature>0 (relevant for GRPO!)

Step 7: Sleep/wake → SAME worker mechanism
  Both runners share GPUWorker.sleep()/wake_up()
  Different post_kv_cache_wake_up() hooks → but both called from same place

Step 8: Weight sync → SAME mechanism
  vLLMColocateWorkerExtension.update_weights_from_ipc()
  → model.load_weights() → runner-independent
  → ★★★ After weight update, CUDA graphs invalidated → re-captured next iteration
```

## 12. RTX 4090 GRPO Training: Specific Recommendations

```
★★★★★ Configuration Matrix:

Scenario A: Qwen3-1.7B BF16 + GRPO (most common small model)
  → Architecture: Qwen3ForCausalLM → in DEFAULT_V2 → MRv2 AUTO-ACTIVATES!
  → ★★★★ UPGRADING TO v0.23.0 SILENTLY SWITCHES THIS TO MRv2!
  → Risk: MEDIUM — MRv2 untested with verl's RL training flow
  → Benefit: FlashInfer sampler for temperature>0 decode → faster sampling
  → Benefit: BCG eliminates torch.compile dependency → simpler warmup
  → ★★★ Recommendation: TEST EXPLICITLY before production deployment
  → ★★★ Fallback: Set VLLM_USE_V2_MODEL_RUNNER=0 in env_vars to force MRv1

Scenario B: Qwen3-8B INT4/GPTQ + GRPO (RTX 4090 most-used config)
  → is_quantized=True → MRv1 FORCED → NO CHANGE from upgrade!
  → ★★★ SAFE — same runner as before, no interaction concerns
  → ★★★★★ RTX 4090 INT4 optimal config unchanged: V1 INT4 + INT8KV + EAGLE
  → ★★★ MRv2 INT4 support is FUTURE — track vLLM progress!

Scenario C: Llama-3.1-8B BF16 + GRPO
  → Architecture: LlamaForCausalLM → in DEFAULT_V2 → MRv2 AUTO-ACTIVATES!
  → Same concerns as Scenario A — needs explicit testing

Scenario D: Qwen2.5-7B BF16 + GRPO
  → Architecture: Qwen2ForCausalLM → NOT in DEFAULT_V2 → MRv1 ALWAYS
  → ★★★ SAFE — same runner as before, no interaction concerns

★★★★★ Action Items for RTX 4090 GRPO:

1. IMMEDIATE (before v0.23.0 upgrade):
   → Pin VLLM_USE_V2_MODEL_RUNNER=0 in verl's env_vars for safety
   → This forces MRv1 for ALL models until MRv2 is explicitly validated
   → Location: vLLMReplica.launch_servers() → env_vars dict → add key

2. TESTING (after v0.23.0 upgrade):
   → Run Qwen3-1.7B BF16 GRPO with MRv2 enabled (no env override)
   → Verify: generation produces valid tokens
   → Verify: log_prob computation correct
   → Verify: sleep/wake cycle stable across 100+ training steps
   → Verify: weight sync produces correct model after IPC transfer
   → Compare: MRv2 vs MRv1 throughput on same GRPO workload

3. MONITORING:
   → Watch for vLLM MRv2 + RL training issues on GitHub
   → Track when MRv2 supports INT4 quantization (future vLLM PR)
   → Track when verl adds explicit MRv2 configuration options

4. LONG-TERM:
   → When MRv2 INT4 support arrives → RTX 4090 INT4 + BCG = best combo
   → FlashInfer sampler for GRPO temperature>0 → significant decode speedup
   → BCG eliminates torch.compile warmup on SM89 → faster startup
```

## 13. Key Source Files

```
| Component              | Path (verl)                                               | Path (vLLM)                                      |
|------------------------|-----------------------------------------------------------|--------------------------------------------------|
| MRv2 oracle            | N/A (no verl code)                                        | vllm/config/vllm.py:519-558                      |
| VLLM_USE_V2 env var    | N/A (not referenced)                                      | vllm/envs.py:267                                 |
| DEFAULT_V2 architectures| N/A                                                      | vllm/config/vllm.py:69-75                        |
| Unsupported features   | N/A                                                       | vllm/config/vllm.py:1982-2060                    |
| GPU worker runner select| N/A                                                      | vllm/v1/worker/gpu_worker.py:161,302,327         |
| MRv2 model runner      | N/A                                                       | vllm/v1/worker/gpu/model_runner.py               |
| MRv1 model runner      | N/A                                                       | vllm/v1/worker/gpu_model_runner.py               |
| Sleep/wake (worker)    | N/A                                                       | vllm/v1/worker/gpu_worker.py:165-200             |
| verl async server      | verl/workers/rollout/vllm_rollout/vllm_async_server.py   | N/A                                              |
| verl rollout config    | verl/workers/config/rollout.py                            | N/A                                              |
| verl engine workers    | verl/workers/engine_workers.py                            | N/A                                              |
| verl weight sync       | verl/workers/rollout/vllm_rollout/utils.py                | N/A                                              |
| verl VLLM_SLEEP_LEVEL  | verl/third_party/vllm/__init__.py                         | N/A                                              |
| verl MRv2 search       | N/A (0 results)                                           | N/A                                              |

★★★ Local repo paths:
  verl: /Users/jackiemac/workspace/rollout-infra/verl/
  vLLM: /Users/jackiemac/workspace/rollout-infra/vllm-latest/
```

## 14. Summary Risk Assessment

```
★★★★★ Overall assessment: MEDIUM RISK for BF16 Qwen3/Llama/Mistral, ZERO RISK for INT4/MoE

SAFE scenarios (no action needed):
  → INT4/GPTQ quantized models → MRv1 forced → same as before
  → MoE models → MRv1 forced → same as before
  → Qwen2/Qwen2.5 models → not in DEFAULT_V2 → MRv1 always
  → NPU/Ascend devices → MRv1 forced → same as before

RISK scenarios (need testing):
  → Qwen3 BF16 dense → MRv2 auto-activates → UNTESTED with verl
  → Llama BF16 dense → MRv2 auto-activates → UNTESTED with verl
  → Mistral BF16 dense → MRv2 auto-activates → UNTESTED with verl

Specific concerns for risky scenarios:
  1. BCG + verl weight update cycle → should work but untested
  2. MRv2 2-phase forward/sample → different timing → may affect verl's generation timeout
  3. FlashInfer sampler → temperature>0 GRPO → BENEFICIAL but new path
  4. compilation_config interaction → BCG replaces PIECEWISE → different warmup
  5. DP>1 with MRv2 → #6456 fix needed → already in v0.23.0

★★★★★ Recommendation: Set VLLM_USE_V2_MODEL_RUNNER=0 until explicit testing confirms MRv2 compatibility with verl's RL training flow.
