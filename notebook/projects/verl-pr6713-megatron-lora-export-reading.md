# verl PR #6713 — Megatron LoRA Adapter Export for Rollout

> 2026-06-17 | verl-project/verl PR #6713 (DRAFT) | Adapter-only LoRA weight export for vLLM rollout
> ★★★★★★★★ CRITICAL: This is a **verl** PR, NOT a Megatron-LM PR. Megatron-LM PR numbering is ~#5393, NOT #6713.
> ★★★★★★★★ Part of 3-repo coordinated effort: verl #6713 + verl-omni #169 + vllm-omni #4388

---

## 1. Purpose and Scope

```
★★★★★★★★★ PR #6713 exports adapter-only Megatron LoRA tensors from verl's Megatron training backend → vLLM-compatible format for rollout weight sync

Specifically targets: Qwen3-Omni 3D MoE architecture
  → EP gather: local expert LoRA → global expert LoRA across EP ranks
  → 3D MoE pack: per-expert LoRA → vLLM 3D MoE layout format

Two new functions in megatron_peft_utils.py:
  1. gather_ep_lora_adapter_weights_for_vllm()  — EP rank gather + local→global ID remap
  2. pack_3d_moe_lora_adapter_weights_for_vllm() — Qwen3-Omni 3D MoE LoRA packing

★★★★★★★★★ Environment toggles (default ON):
  VERL_MEGATRON_LORA_GATHER_EP_ADAPTER_WEIGHTS=1
  VERL_MEGATRON_LORA_GATHER_EP_ADAPTER_DEBUG=0
  VERL_MEGATRON_LORA_PACK_3D_MOE_ADAPTER_WEIGHTS=1
  VERL_MEGATRON_LORA_RECOMPUTE_INPUT_GRAD=1
```

---

## 2. Key Code Changes (4 files modified + 1 test)

### A. `verl/utils/megatron_peft_utils.py` — Main new functionality

```
★★★★★★★★★ gather_ep_lora_adapter_weights_for_vllm():
  → When ep_size > 1: _all_gather_tensor_for_ep() across EP ranks
  → Rewrite local→global expert IDs: global_id = ep_rank * local_expert_count + local_id
  → Example: EP rank 0 has experts 0,1; EP rank 1 has experts 0,1
    → Global mapping: rank 0 = [0,1], rank 1 = [2,3]
  → ★★★★★ RTX 4090 EP=1: passthrough — zero overhead, no all-gather needed!

★★★★★★★★★ pack_3d_moe_lora_adapter_weights_for_vllm():
  → For Qwen3-Omni model type only
  → Validates: gate_proj and up_proj share same lora_A (architecture requirement)
  → Packs into 4 tensors per expert group:
    base_layer.lora_A.weight: (num_experts * rank, hidden) — concat gate+up A
    base_layer.lora_B.weight: (2 * intermediate, num_experts * rank) — concat gate+up B
    lora_A.weight: (num_experts * rank, intermediate) — concat down A
    lora_B.weight: (hidden, num_experts * rank) — concat down B
  → ★★★★★ Only for Qwen3-Omni; standard MoE (Qwen3-30B-A3B) uses PEFT path directly
```

### B. `verl/workers/engine/megatron/transformer_impl.py`

```
get_per_tensor_param() with adapter_only=True now chains:
  per_tensor_param = self.bridge.export_adapter_weights(self.module)
  per_tensor_param = gather_ep_lora_adapter_weights_for_vllm(per_tensor_param)
  per_tensor_param = pack_3d_moe_lora_adapter_weights_for_vllm(
      per_tensor_param, model_type=self.model_config.hf_config.model_type
  )
★★★★★★★★★ Integrates new functions into actual weight sync pipeline
```

### C. `verl/models/mcore/bridge.py`

```
Overrides LinearForLastLayer from Megatron-Bridge:
  → Add sequence_parallel gather with gradient preservation
  → Return (logits, None) tuple instead of bare logits
  → ★★★★★ Fixes LM-head gather during LoRA-only training with sequence parallelism
```

### D. `verl/models/mcore/model_forward_fused.py`

```
★★★★ CRITICAL REVIEW ISSUE flagged by Gemini Code Assist:
  → decoder_input.requires_grad_(True) on non-leaf tensor → runtime error!
  → Recommendation: detach first before requires_grad_(True)
  → Environment toggle: VERL_MEGATRON_LORA_RECOMPUTE_INPUT_GRAD (default "1")
```

### E. New test file: `tests/utils/test_megatron_lora_export_on_cpu.py`

```
5 test functions:
  test_build_peft_config_for_vllm_expands_megatron_target_modules
  test_gather_ep_lora_adapter_weights_passthrough_without_ep
  test_gather_ep_lora_adapter_weights_renames_local_to_global_experts
  test_pack_3d_moe_lora_adapter_weights_for_vllm_packs_complete_expert_groups
  test_pack_3d_moe_lora_adapter_weights_for_vllm_rejects_split_gate_up_lora_a
★★★★★★★★★ EP=1 passthrough test explicitly validates RTX 4090 scenario!
```

---

## 3. 3-Repo Coordinated Ecosystem

| Repo | PR | Status | Purpose |
|------|-----|--------|---------|
| **verl-project/verl** | #6713 | DRAFT | Export Megatron LoRA adapter weights → vLLM format |
| **verl-project/verl-omni** | #169 | DRAFT | Qwen3-Omni Megatron LoRA rollout: OmniTensorLoRARequest, ZMQ rank-indexed handles, AR rollout mode |
| **vllm-project/vllm-omni** | #4388 | DRAFT | Declare Qwen3-Omni thinker LoRA metadata (embedding, 3D MoE, non-gated behavior) |

★★★★★★★★★ All 3 DRAFT, same author (hbhflw2000/Wei), coordinated effort — NOT yet production-ready

---

## 4. MoE LoRA Export — Detailed Mechanism

```
★★★★★★★★★ Two mechanisms for MoE LoRA handling:

Mechanism 1: EP Gather (gather_ep_lora_adapter_weights_for_vllm)
  → Training with EP > 1: each EP rank has subset of experts
  → LoRA weights for local experts → gather across ranks → remap local→global IDs
  → Formula: global_id = ep_rank * local_expert_count + local_id
  → ★★★★★ RTX 4090 EP=1: passthrough, zero overhead

Mechanism 2: 3D MoE Pack (pack_3d_moe_lora_adapter_weights_for_vllm)
  → vLLM expects Qwen3-Omni MoE LoRA in specific 3D layout (base_layer + packed experts)
  → Transforms per-expert (gate_proj, up_proj, down_proj) LoRA_A/B → packed format
  → Validation: gate_proj and up_proj MUST share same lora_A matrix
  → Only for _QWEN3_OMNI_3D_MOE_MODEL_TYPES set
  → ★★★ Standard MoE models use PEFT path directly, no 3D packing needed

★★★★★★★★★ RTX 4090 MoE LoRA export path:
  → Qwen3-30B-A3B (standard MoE): EP=1 passthrough → PEFT standard → vLLM serve
  → Qwen3-Omni (3D MoE): EP=1 passthrough → 3D pack → vLLM-omni serve
  → Both scenarios: EP=1 = zero gather overhead on RTX 4090
```

---

## 5. Megatron-Bridge LoRA — Existing Infrastructure

```
★★★★ Megatron-Bridge (NeMo2) = ONLY existing LoRA path for Megatron-based training:

Key functions used by verl:
  create_peft() → Creates PEFT/LoRA adapter objects from config
  export_adapter_weights() → Exports adapter-only weights (LoRA matrices) from sharded model
  export_weights() / export_hf_weights() → Full weight export (base + LoRA merged) in HF format
  save_hf_weights() → Saves full model weights to HF format

★★★★★★★★★ Megatron-Bridge LoRA uses Megatron-style sharded format, NOT standard PEFT/HF safetensors
  → LoRA weights live within Megatron's distributed checkpoint structure (TP-sharded, EP-sharded)
  → verl acts as format bridge: Megatron-Bridge export_adapter_weights() → megatron_peft_utils.py → vLLM format

★★★★ Megatron-Bridge LoRA PRs (current):
  #4218: Support multi-lora for Megatron models (OPEN)
  #4198: Add weight and bias property for LoRALinear (OPEN)
  #4012: MLPerf v6.0 parity recipes (OPEN)
```

---

## 6. RTX 4090 Implications

```
★★★★★★★★★ RTX 4090 + PR #6713:

Direct training: NOT viable
  → verl Megatron backend requires Megatron-Bridge + multi-GPU setup
  → Core Megatron crashes on single GPU (#5203)
  → RTX 4090 = single GPU → cannot use verl Megatron backend for training

Indirect serving: YES viable
  → Train on multi-GPU (verl Megatron backend) → export adapter-only LoRA via #6713
  → Deploy adapter on RTX 4090 vLLM serving → LoRA hot-swap
  → ★★★★★★★★ This bridges: multi-GPU training → single-GPU serving

★★★★★★★★★ BEST RTX 4090 training paths remain:
  → DeepSpeed ZeRO-2 + LoRA → PEFT export → vLLM serve (standard path)
  → rLLM Tinker → PEFT save → vLLM serve (simplest, 3 commands)
  → ★★★★★★★★ PR #6713 is for verl Megatron backend → multi-GPU → NOT RTX 4090 training
```

---

## 7. Format Comparison — Updated

| Framework | LoRA Export | MoE LoRA | RTX 4090 Training? |
|-----------|------------|----------|---------------------|
| **DeepSpeed** | PEFT/HF safetensors (merge_and_unload) | AutoEP LoRA → PEFT but no RouterReplay | YES (ZeRO-2) |
| **verl FSDP** | PEFT save_pretrained → HF safetensors | Not MoE-specific | YES (bypass_mode) |
| **verl Megatron (pre-#6713)** | Megatron-Bridge export → manual conversion | No EP gather or 3D pack | NO (multi-GPU) |
| **verl Megatron (#6713)** | EP gather + 3D pack → vLLM format | YES (EP gather + 3D pack) | NO (multi-GPU) |
| **rLLM Tinker** | PEFT save_pretrained → HF safetensors | Dense models only | YES (#1) |
| **Megatron Lite** | PEFT/Mint import/export | MoE LoRA (lora_adapter.py) | Experimental |
| **vLLM/SGLang** | N/A (serving endpoint) | N/A | N/A |

---

## Key Findings Summary

★★★★★★★★★ CRITICAL CORRECTION: PR #6713 is a **verl** PR, NOT Megatron-LM — project notes must be updated
★★★★★★★★★ PR #6713: verl Megatron backend → adapter-only LoRA export → vLLM format (EP gather + 3D MoE pack)
★★★★★★★★★ 3-repo coordinated: verl #6713 + verl-omni #169 + vllm-omni #4388 — all DRAFT, same author
★★★★★★★★★ EP=1 on RTX 4090 = passthrough (zero gather overhead) — but can't train on Megatron backend (single GPU crash #5203)
★★★★★★★★★ Critical review issue: requires_grad_(True) on non-leaf tensor → runtime error → needs fix before merge
★★★★★★★★★ RTX 4090 LoRA training: DeepSpeed/rLLM still best; PR #6713 useful for multi-GPU train → RTX 4090 serve bridge
★★★★★★★★★ Megatron-Bridge = ONLY Megatron LoRA path; uses sharded format, not standard PEFT; verl bridges format
★★★★★★★★★ Qwen3-Omni 3D MoE LoRA: specific packing format needed for vLLM; standard MoE uses PEFT directly

---

## References

- verl PR #6713: https://github.com/verl-project/verl/pull/6713 (DRAFT)
- verl-omni PR #169: https://github.com/verl-project/verl-omni/pull/169 (DRAFT)
- vllm-omni PR #4388: https://github.com/vllm-project/vllm-omni/pull/4388 (DRAFT)
- Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- Megatron-LM crash #5203: notebook/projects/megatron-grpo-sm89-reading.md
- LoRA export comparison: notebook/fundamentals/cross-framework-lora-adapter-export-comparison.md
- Megatron Lite: notebook/projects/megatron-lite-reading.md
