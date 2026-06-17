# verl #6512 — Per-Unit LoRA Summon for FSDP1/FSDP2 Reading

> 2026-06-18 | PR #6512 OPEN | +193/-38 | 3 files: fsdp_utils.py, transformer_impl.py, monkey_patch.py
> ★★★★★★★★ Peak memory bounded by LARGEST FSDP unit, not WHOLE model!
> ★★★★★★★★ CRITICAL for RTX 4090: 30B+ MoE LoRA now viable → per-expert summon
> ★★★★★★★★ FSDP1/FSDP2 compatibility + skip degenerate units + LoRA dtype cast + strip_modules

---

## 1. Problem: Root-Level LoRA Summon = Full Model Unsharded

```
★★★★★★★★★ Before #6512 (current verl):

layered_summon_lora_params summons the ROOT FSDP module:
  → FSDP root module = entire model wrapped in single FSDP unit
  → summon_full_params(root) → ALL parameters unsharded at once
  → Peak memory = FULL MODEL UNSHARDED!
  → For Qwen3-30B: ~60 GiB just for weight summon → OOM on RTX 4090!

★★★★★★★★★ Why this matters for RTX 4090:
  → LoRA weight sync = summon LoRA params → send to rollout
  → Before #6512: summon ALL params → 60 GiB peak → can't fit 24 GiB!
  → After #6512: summon per-FSDP-unit → peak = largest unit → ~6-8 GiB
  → ★★★★★★★★ 10x memory reduction for weight sync → RTX 4090 NOW VIABLE for 30B+ models!
```

---

## 2. The Fix: Per-Unit LoRA Summon (+193/-38)

```
★★★★★★★★★ Key changes in fsdp_utils.py — layered_summon_lora_params rewrite:

Before:
  → summon_full_params(root_module) → all params unsharded → peak = whole model

After:
  → Iterate every FSDP unit via named_modules()
  → Summon each unit individually via summon_full_params(submodule)
  → ★★★★★★★★ Peak memory = LARGEST SINGLE FSDP UNIT, not whole model!
  → For MoE models → wrap granularity at expert/MLP level → peak = single expert

★★★★★★★★★ FSDP1 / FSDP2 compatibility:

  → Gate summon_full_params on fsdp_version(submodule) == 1
  → FSDP2 modules: parameters are DTensor → param.full_tensor() all-gathers on demand
  → No context manager needed for FSDP2 → simpler summon
  → ★★★★★★★★ Works with BOTH FSDP versions → verl supports both

★★★★★★★★★ Skip degenerate FSDP units:

  → Frozen sub-encoders excluded from LoRA → FSDP-wrapped but never forwarded
  → lazy state (_all_handles) uninitialized → state_dict() crashes
  → Skip any unit whose named_parameters() has NO lora_* entries
  → ★★★★★★★★ Prevents crash from non-LoRA sub-encoders in multi-stage models

★★★★★★★★★ named_parameters() instead of state_dict():

  → FSDP1 only initializes _all_handles on root module
  → state_dict() on non-root → _pre_state_dict_hook → dereferences _all_handles → CRASH
  → dict(submodule.named_parameters()) → same unsharded tensors WITHOUT invoking hook
  → ★★★★★★★★ Avoids FSDP1 crash pattern that bit many LoRA users

★★★★★★★★★ LoRA wrap policy guard:

  → Skip lambda wrap policy when min_num_params > 0
  → Combining size policy + lambda policy → nested FSDP units
  → Allgather order can diverge across ranks under variable-length inputs (use_remove_padding)
  → ★★★★★★★★ NCCL deadlocks during backward → CRITICAL for padded models
```

---

## 3. Additional Features

```
★★★★★★★★★ transformer_impl.py — _verl_strip_modules support:

  → After loading weights → delete sub-modules listed in model._verl_strip_modules
  → Multi-stage models declare which stages NOT needed for training
  → e.g. talker, code2wav → never allocated on GPU → memory savings
  → ★★★★★★★★ For Qwen3-Omni: strip talker + code2wav → ~8-10 GiB savings

★★★★★★★★★ LoRA dtype cast:

  → Cast lora_A / lora_B params to base model dtype (bf16) before FSDP wrap
  → LoRA adapters default to fp32 → dtype mismatch during forward → CRASH
  → ★★★★★★★★ Simple fix: .to(bf16) → prevents fp32/bf16 mixed dtype errors
  → Same pattern as DeepSpeed #8072/#8073 root cause (but LoRA-level, not ZeRO-level)

★★★★★★★★★ monkey_patch.py — config.get_text_config():

  → Replace direct model.config.text_config access
  → Multi-stage models nest text config inside thinker_config.text_config
  → get_text_config() resolves correct nesting level regardless of depth
```

---

## 4. RTX 4090 Impact

```
★★★★★★★★★ RTX 4090 LoRA GRPO with #6512:

Memory savings breakdown:
  → Per-unit summon: 60→6-8 GiB peak for weight sync (10x reduction!)
  → strip_modules: ~8-10 GiB from removing unused stages
  → LoRA bf16 cast: ~0.5 GiB from fp32→bf16 LoRA params
  → ★★★★★★★★ Combined: ~18-20 GiB savings → RTX 4090 24 GiB FITS!

★★★★★★★★★ Config recommendation with #6512:
  → MUST: layered_summon=True (per-unit summon)
  → MUST: lora.merge=False (unmerged for HYBRID mode)
  → MUST: lora_rank=32, lora_alpha=64 (rank>64 breaks EOS #6782!)
  → MUST: bypass_mode=True (18Ψ→3.8Ψ)
  → MUST: enforce_eager=True (avoid CUDA graph issues)
  → SHOULD: model._verl_strip_modules = ['talker', 'code2wav'] for multi-stage models

★★★★★★★★★ Ranking impact:
  → Before #6512: verl GRPO+bypass #2 → limited to ~14B models on RTX 4090
  → After #6512: verl GRPO+bypass #1 → 30B+ MoE models viable!
  → ★★★★★★★★ verl CPPO+bypass still #1 → but #6512 extends range dramatically
```

---

## Key Findings Summary

★★★★★★★★★ #6512: Per-unit LoRA summon → peak memory bounded by largest FSDP unit (10x reduction!)
★★★★★★★★★ FSDP1/FSDP2 compatibility → works with both versions
★★★★★★★★★ Skip degenerate FSDP units → prevents crash from non-LoRA sub-encoders
★★★★★★★★★ named_parameters() instead of state_dict() → avoids FSDP1 crash
★★★★★★★★★ LoRA dtype cast to bf16 → prevents fp32/bf16 mixed dtype errors
★★★★★★★★★ strip_modules → delete unused sub-modules → 8-10 GiB savings
★★★★★★★★★ RTX 4090: 30B+ MoE LoRA GRPO NOW VIABLE with #6512 + bypass_mode + rank=32
★★★★★★★★★ 193 additions, 38 deletions, 3 files

---

## References

- verl #6512: https://github.com/verl-project/verl/pull/6512
- verl #6782: https://github.com/verl-project/verl/issues/6782 (LoRA rank>32 breaks EOS)
- verl #6688: https://github.com/verl-project/verl/pull/6688 (LoRA IPC buffer crash fix)
- RTX 4090 config: tools/rtx4090_grpo_config_reference.py
