# Megatron-Bridge (NeMo2) LoRA Source-Level Analysis

> 2026-06-18 | megatron.bridge.peft | P5 coverage gap filled
> Key: ONLY existing LoRA path for Megatron-based training → standalone pip package → verl integration via create_peft hook
> ★★★★★★★★ Core Megatron-LM has NO LoRA at all → ALL LoRA goes through Megatron-Bridge

---

## 1. Repository Architecture

```
★★★★★★★★★ Two distinct LoRA code paths in verl ecosystem:

Standalone package (pip-installable): megatron-bridge (NVIDIA/Megatron-Bridge)
  → Module: megatron.bridge.peft
  → Key import: from megatron.bridge.peft.utils import create_peft
  → verl requires Megatron-Bridge > 0.2.0
  → Recommended commit: 0a21da47ecfc609e546e790b901f1657808717b5

Legacy NeMo monolith (embedded): nemo/collections/nlp/modules/common/peft/
  → Contains older NeMo-style PEFT: LoRALinear, LoRAColumnParallelLinear, LoRARowParallelLinear
  → Predates standalone package → tightly coupled to NeMo training loop

★★★★★★★★★ verl uses TWO bridge paths:
  → Vanilla mbridge (mbridge.py): from mbridge import AutoBridge → older package
  → Non-vanilla Megatron-Bridge (bridge.py): from megatron.bridge import AutoBridge → newer standalone

★★★★★★★★★ CONFIRMED: Core Megatron-LM has NO peft/lora directories → LoRA entirely delegated to Bridge
```

---

## 2. LoRA Types: lora, canonical_lora, vlm_lora, dora

### `lora` (fused LoRA) — Default

```
★★★★★ Fused LoRA: applies to Megatron's fused parallel linear layers
  → Target modules: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
  → LoRA applied to combined QKV gate-up projection layers directly
  → Best for standard LLM fine-tuning with TP support
  → LoRALinear wraps ColumnParallelLinear and RowParallelLinear
```

### `canonical_lora`

```
★★★★ canonical_lora: unfuses QKV into separate projections
  → Target modules: ["linear_q", "linear_k", "linear_v", "linear_proj",
                      "linear_fc1_up", "linear_fc1_gate", "linear_fc2"]
  → Follows standard Hu et al. (2021) LoRA convention → each sub-projection gets own A/B
  → Advantage: finer-grained control over which sub-projections get adapters
  → Disadvantage: slightly more parameters → each sub-projection has own A/B matrices
  → ★★★★★★★★ canonical_lora = standard PEFT layout → vLLM directly compatible
```

### `vlm_lora` (Vision-Language Model)

```
★★★★ vlm_lora: extends LoRA to vision-language multimodal models
  → freeze_vision_model: True → freeze vision encoder
  → freeze_vision_projection: True → freeze vision-to-language projection
  → freeze_language_model: True → freeze language model base weights
  → Handles cross-modal attention and projection layers
```

### `dora` (Weight-Decomposed Low-Rank Adaptation)

```
★★★★★ dora: Liu et al. (2024) → magnitude + direction decomposition
  → W = m * (W_d + BA/r) → m = magnitude (learnable/fixed), W_d = normalized direction
  → LoRA on directional component only → better performance than standard LoRA at same rank
  → ★★★★★ TP: magnitude vector replicated, directional LoRA follows TP partitioning
```

---

## 3. create_peft() — Central Factory Entry Point

```python
# verl/workers/config/megatron_peft.py
from megatron.bridge.peft.utils import create_peft

def get_peft_cls(model_config, bridge, provider, dtype=None):
    lora_cfg = model_config.lora
    if lora_cfg.get("rank", 0) <= 0:
        return None
    assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"
    peft_cls = create_peft(lora_cfg, dtype=dtype)
    return peft_cls
```

```
★★★★★★★★★ create_peft() flow:
  1. Reads lora_cfg["type"] → selects PEFT variant (lora/canonical_lora/vlm_lora/dora)
  2. Configures rank, alpha, dropout, target_modules, exclude_modules
  3. Creates PEFT adapter class → when applied as hook:
    → Walk named modules → match target modules (wildcard support)
    → Replace matched linear layers with LoRA-wrapped equivalents
    → Freeze base model parameters (requires_grad=False)
    → Initialize: lora_A with kaiming (default), lora_B with zero (default)

★★★★★★★★★ peft_cls used in 3 ways:
  1. Pre-wrap hook: create_peft_hook(peft_cls, training=True) → registered BEFORE DDP wrapping
  2. Adapter checkpoint filtering: apply_peft_adapter_filter_to_state_dict()
  3. Adapter weight export: bridge.save_hf_adapter(model, path, peft_cls)

★★★★★★★★★ CRITICAL: assertion "bridge is not None and provider is not None"
  → LoRA ONLY works via Megatron-Bridge provider → NOT via vanilla mbridge or standalone Megatron
```

---

## 4. PEFT Pre-Wrap Hook Ordering (Critical for Correctness)

```
★★★★★★★★★ PEFT must be applied BEFORE DDP wrapping:
  1. PEFT freezes base model parameters (requires_grad=False)
  2. DDP must know which parameters are trainable when building gradient buckets
  3. Distributed optimizer must only track trainable (adapter) parameters

★★★★★★★★★ verl integration flow (from megatron_utils.py):
  → Register create_peft_hook(peft_cls, training=True) as pre-wrap hook on provider
  → If adapter_path set → also register load_peft_adapter_checkpoint hook
  → Register peft_info_hook to print adapter parameter count
  → Call provider.provide_distributed_model() with all hooks registered
```

---

## 5. Weight Export — Three Paths

```python
# verl/workers/engine/megatron/transformer_impl.py
def get_per_tensor_param(self, base_sync_done=False, **kwargs):
    adapter_only = base_sync_done and non_merge_lora_sync

    if self.vanilla_bridge:
        per_tensor_param = self.bridge.export_weights(self.module)          # Path 1
    elif adapter_only:
        per_tensor_param = self.bridge.export_adapter_weights(self.module)  # Path 2
    else:
        per_tensor_param = self.bridge.export_hf_weights(self.module, ...)  # Path 3
```

```
★★★★★★★★★ Three export paths:

Path 1: Vanilla bridge → bridge.export_weights(module)
  → ALL weights (base + adapter merged) → for non-Megatron-Bridge setups

Path 2: Adapter-only → bridge.export_adapter_weights(module)
  → Only LoRA adapter delta (lora_A, lora_B, scaling) → for incremental RL weight sync
  → ★★★★★★★★ After base sync once → subsequent syncs only transfer lightweight adapter deltas
  → ★★★★★★★★ This is the path PR #6713 enhances with EP gather + 3D MoE pack

Path 3: HF export → bridge.export_hf_weights(module, merge_adapter_weights=False/True)
  → HuggingFace format → with optional adapter merging
  → add_base_layer_suffix() handles .base_layer naming for vLLM stacked TP params

★★★★★★★★★ Bridge method mapping:
| Method              | Vanilla mbridge       | Non-vanilla Megatron-Bridge        |
|---------------------|-----------------------|------------------------------------|
| Full model export   | save_weights()        | save_hf_weights()                  |
| Adapter-only export | NOT available         | save_hf_adapter()                  |
| Inference export    | export_weights()      | export_hf_weights() or export_adapter_weights() |
```

---

## 6. Checkpoint Layout (v2 Schema)

```
checkpoint_path/
  model/
    huggingface/          # mbridge-saved HF weights + config + tokenizer
      adapter/            # PEFT adapter in HF format (when peft_cls set)
    dist_ckpt/            # Megatron sharded model shards (PEFT adapter shards)
  optimizer/dist_ckpt/    # optimizer state
  extra/dist_ckpt/        # rng_state

★★★★★★★★★ Adapter checkpoint filtering:
  → apply_peft_adapter_filter_to_state_dict(state_dict, peft_cls)
  → Filters dist_ckpt to adapter-only shards → for resume training
  → Manifest tracks peft_adapters as separate content type alongside base model
```

---

## 7. TP and EP Handling for LoRA

```
★★★★★★★★★ TP handling:
  → Fused lora: LoRA on already-TP-partitioned fused layers (linear_qkv column-parallel)
  → LoRA A: partitioned along input dimension (replicated across TP ranks)
  → LoRA B: partitioned along output dimension (split across TP ranks like base)
  → canonical_lora: individual sub-projections (q,k,v) already partitioned → each gets TP-aware LoRA
  → ★★★★★★★★ RTX 4090 TP=1: LoRA layers wrap standard nn.Linear → zero partitioning overhead

★★★★★★★★★ EP handling for MoE LoRA:
  → Qwen3-30B-A3B example: actor_ep=4, actor_etp=1
  → LoRA on expert linear layers (linear_fc1, linear_fc2) as well as attention
  → ★★★★★★★★ MoE routers EXCLUDED from LoRA by default → must add "router" to target_modules
  → EP-sharded LoRA parameters handled via expert_parallel_buffers
  → verl weight sync: all_gather across EP groups → further handle ETP

★★★★★★★★★ Weight sync between training (Megatron) and inference (vLLM):
  → merge=True: LoRA merged into base weights → simpler but potential precision loss
  → merge=False: base sync once → adapter-only deltas → efficient for RL loops
```

---

## 8. Open PRs

```
★★★★ PR #4218 — Multi-LoRA / Multi-Adapter:
  → MultiLoraManager class → manage multiple adapters across TP/PP ranks
  → Batched requests with different LoRA adapters per sequence
  → Dynamic hot-loading/unloading → no restart needed
  → ★★★ Primarily serving feature → not directly relevant for single-GPU GRPO training

★★★★ PR #4198 — LoRALinear Properties Enhancement:
  → Refactoring LoRALinear → better configuration properties
  → Improved interface for setting/querying LoRA hyperparameters
  → Foundation for #4218 → better for diagnostic tools

★★★★ PR #4012 — MLPerf Integration:
  → Benchmark configurations → low relevance for RTX 4090
```

---

## 9. Comparison: Megatron-Bridge vs Megatron Lite vs Core Megatron

| Feature | Megatron-Bridge LoRA | Megatron Lite #4885 | Core Megatron-LM |
|---------|---------------------|---------------------|------------------|
| **LoRA support** | Full: lora, canonical_lora, vlm_lora, dora | Experimental (lite runtime) | NONE |
| **Source** | megatron.bridge.peft (pip package) | megatron.lite (experimental) | No PEFT module |
| **TP/PP/EP** | Full distributed LoRA | Lite runtime (simpler setups) | N/A |
| **Weight export** | export_weights, export_adapter_weights, export_hf_weights, save_hf_adapter | HF safetensors helpers | N/A |
| **verl integration** | Direct: create_peft hook + provider | None yet | None |
| **MoE LoRA** | Supported (Qwen3-30B-A3B) | Qwen3/Qwen3.5 model impls | N/A |
| **Single GPU** | Works TP=1, PP=1 | Designed for lighter setups | N/A |

---

## 10. RTX 4090 Implications

```
★★★★★★★★★ Megatron-Bridge LoRA CAN work on single GPU (TP=1, PP=1) but with caveats:

What works:
  → TP=1, PP=1: LoRALinear wraps standard nn.Linear → zero partitioning overhead
  → Fused lora type: target_modules work at TP=1 → standard linear layers
  → canonical_lora: individual sub-projections → standard linear layers
  → dora: works TP=1 but adds magnitude decomposition overhead

What does NOT work well:
  → EP requires multiple GPUs → MoE LoRA EP=1 only → all experts in 24GB VRAM
  → Bridge abstraction overhead: provider hooks + pre-wrap callbacks → ~3-5% overhead vs DeepSpeed
  → Memory: loads full HF model via load_hf_weights + PEFT on top → 7B fits, 13B+ needs QLoRA
  → DistributedOptimizer on single GPU = pure overhead (same as #8061 context)

★★★★★★★★★ RTX 4090 Megatron-Bridge LoRA ranking: #3-#4
  → rLLM Tinker #1 > verl CPPO+bypass #2 > DeepSpeed ZeRO-2 #2.5 > Megatron-Bridge #3-4
  → Bridge overhead designed for multi-GPU → not optimized for single GPU
  → ★★★★★★★★ BUT: Megatron-Bridge enables MoE LoRA → DeepSpeed cannot do MoE easily → unique value!

★★★★★★★★★ Recommended RTX 4090 Megatron-Bridge LoRA config:
lora:
  type: lora              # or canonical_lora for vLLM compatibility
  rank: 32
  alpha: 32
  merge: False            # adapter-only sync for RL efficiency
  target_modules: [linear_qkv, linear_proj, linear_fc1, linear_fc2]
  dropout: 0.0
  lora_A_init_method: kaiming
  lora_B_init_method: zero
actor:
  megatron:
    use_mbridge: True
    vanilla_mbridge: False
    tensor_model_parallel_size: 1    # TP=1 single GPU
    pipeline_model_parallel_size: 1  # PP=1 single GPU
    param_offload: True              # critical for RTX 4090 memory
    optimizer_offload: True
    use_distributed_optimizer: False  # pure overhead on single GPU
```

---

## Key Findings Summary

★★★★★★★★★ Core Megatron-LM has NO LoRA → ALL LoRA goes through Megatron-Bridge → standalone pip package
★★★★★★★★★ 4 LoRA types: lora (fused), canonical_lora (unfused, PEFT-compatible), vlm_lora (VLM), dora (magnitude+direction)
★★★★★★★★★ create_peft() = central factory → config dict → PEFT adapter class → pre-wrap hook before DDP
★★★★★★★★★ PEFT pre-wrap hook ordering: LoRA before DDP → freeze base params → DDP aware of trainable params
★★★★★★★★★ 3 export paths: export_weights (full), export_adapter_weights (adapter-only, PR #6713 enhanced), export_hf_weights (HF format)
★★★★★★★★★ Adapter-only export = incremental RL weight sync → base sync once → subsequent syncs only transfer lightweight LoRA deltas
★★★★★★★★★ TP=1 on RTX 4090: LoRALinear wraps nn.Linear → zero overhead → BUT bridge abstraction adds ~3-5%
★★★★★★★★★ MoE LoRA: routers excluded by default → must add "router" to target_modules → EP=1 for single GPU
★★★★★★★★★ RTX 4090 ranking: Megatron-Bridge LoRA #3-4 → bridge overhead not optimized for single GPU
★★★★★★★★★ Megatron-Bridge unique value: MoE LoRA → DeepSpeed cannot do MoE easily on single GPU

---

## References

- Megatron-Bridge repo: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- verl megatron_peft.py: verl/workers/config/megatron_peft.py
- verl ppo_lora.rst: verl/docs/advance/ppo_lora.rst
- verl PR #6713: notebook/projects/verl-pr6713-megatron-lora-export-reading.md
- Megatron Lite #4885: notebook/projects/megatron-lite-reading.md
- DeepSpeed AutoEP: notebook/projects/deepspeed-autoep-rtx4090-moe-practical.md
- verl #6713 tracker: notebook/projects/7-framework-developments-tracker-2026-06-16.md
