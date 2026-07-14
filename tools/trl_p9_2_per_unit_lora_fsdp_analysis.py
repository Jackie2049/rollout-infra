"""
TRL P9-2: Per-unit LoRA summon for FSDP — Design Analysis

## Reference Implementation
verl PR #6512 (commit 8a69493):
  verl/utils/fsdp_utils.py → layered_summon_lora_params()

## Core Mechanism

Instead of summoning the entire root FSDP module at once (peak = full model),
iterate each FSDP unit individually via named_modules():

  1. For each FSDP submodule (layer/MLP/expert):
     a. Skip root, non-FSDP, and nested FSDP children (to avoid double-gather)
     b. Skip units with no LoRA parameters
     c. FSDP1: summon_full_params(submodule, writeback=False)
        FSDP2: DTensor → param.full_tensor() on demand (no context manager)
     d. Collect LoRA adapter params from the gathered unit
     e. Copy gathered tensors to CPU, free GPU memory
  2. Return OrderedDict of all LoRA params (on CPU)

## Memory Impact
  - Peak GPU: largest single FSDP unit (one layer or MLP expert)
  - NOT full model (10x reduction: 60 GiB → 6 GiB)
  - Additional savings via _verl_strip_modules (delete unused sub-modules)

## TRL Port Challenges
  - TRL uses HF Trainer + Accelerate FSDPPlugin, NOT verl's custom FSDP
  - Accelerate's FSDPPlugin wraps the model after _init() but before training
  - Need to hook into the FSDP wrapping to know unit boundaries
  - The layered_summon would be needed where TRL currently calls
    model.forward() for ref logprobs with PEFT adapter disabled
  - Existing TRL code: is_peft_model() checks already present

## Key Design Decision
  - Option A: New fsdp_unit.py module + monkey-patch accelerate FSDP
  - Option B: Custom FSDP wrapping that extends accelerate.FSDPPlugin
  - Option C: Separate FSDP utility (like verl) that wraps after accelerate

  Option C is most compatible with TRL's architecture and minimize
  changes to the existing HF Trainer integration.

## Dependencies
  - accelerate.FSDPPlugin internals (isinstance checks, wrap policy)
  - PEFT model state dict (get_peft_model_state_dict)
  - FSDP1 vs FSDP2 detection (fsdp_version utility)

## Status
  ANALYSIS COMPLETE — Implementation pending user authorization.
  Will need actual TRL source to verify accelerate FSDP integration.
"""
