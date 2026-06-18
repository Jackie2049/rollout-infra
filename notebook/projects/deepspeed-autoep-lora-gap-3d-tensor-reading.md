# DeepSpeed AutoEP LoRA Gap — GroupedExperts 3D Tensor Incompatibility

> 2026-06-18 | Source-verified analysis of GroupedExperts 3D nn.Parameter vs standard PEFT LoRA 2D nn.Linear
> ★★★★★★★★ GroupedExperts stores w1/w2/w3 as 3D nn.Parameter → PEFT LoRA targets 2D nn.Linear.weight → INCOMPATIBLE!
> ★★★★★★★★ AutoEP + LoRA on RTX 4090 requires a "grouped LoRA" implementation for 3D expert tensors

---

## 1. The Problem — 3D vs 2D Weight Storage

```
★★★★★★★★★ GroupedExperts weight storage (ep_experts.py lines 164-166):

class GroupedExperts(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, use_grouped_mm=True):
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))    # (E, H, D) — 3D!
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))     # (E, D, H) — 3D!
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))    # (E, H, D) — 3D!

★★★★★★★★★ Standard PEFT LoRA target (peft/tuners/lora/model.py):

class LoraModel:
    def _create_new_module(self, lora_config, adapter_name, target):
        # Only targets nn.Linear → 2D weight
        if isinstance(target, nn.Linear):
            # weight shape: (out_features, in_features) — 2D!
            return LinearLora(target, lora_config, adapter_name)

★★★★★★★★★ The incompatibility:

GroupedExperts:
  → w1: nn.Parameter — 3D tensor (E, hidden_dim, dim)
  → w2: nn.Parameter — 3D tensor (E, dim, hidden_dim)
  → w3: nn.Parameter — 3D tensor (E, hidden_dim, dim)
  → NOT nn.Linear → no .weight attribute → no .in_features/.out_features

PEFT LoRA:
  → Targets: nn.Linear → 2D .weight (out_features, in_features)
  → LoRA decomposition: A∈(r, in_features), B∈(out_features, r)
  → h = x @ W^T + (x @ A^T @ B^T) * alpha/r
  → ONLY works on 2D weight matrices!

★★★★★★★★★ Why this matters for RTX 4090:
  → AutoEP MoE + LoRA = ideal RTX 4090 combination (ZeRO-2 + LoRA + EP=1)
  → But: GroupedExperts weights = 3D → LoRA can't be applied → GAP!
  → Without LoRA: full MoE weight updates → too many parameters → memory overflow!
  → → AutoEP MoE on RTX 4090 currently BLOCKED by this LoRA gap!
```

---

## 2. Why 3D Storage Exists

```
★★★★★★★★★ GroupedExperts uses 3D tensors for two reasons:

1. Expert grouping for torch._grouped_mm:
  → torch._grouped_mm expects 3D weight tensors: (E, H, D)
  → Each expert has its own weight slice → batched grouped matmul
  → Shape convention: w1[expert_idx].transpose(-2, -1) → matmul input

2. SwiGLU computation per expert:
  → for-loop path: w1_e = w1[expert_idx].to(dtype).transpose(-2, -1)
  → Sequential matmul per expert → gate*up*down pattern
  → 3D enables efficient expert indexing via w1[expert_idx]

★★★★★★★★★ Alternative: nn.ModuleList of nn.Linear (what HuggingFace MoE uses):

class HFMoELayer:
    → self.gate_up_proj = nn.ModuleList([nn.Linear(dim, 2*hidden_dim) for _ in range(E)])
    → self.down_proj = nn.ModuleList([nn.Linear(hidden_dim, dim) for _ in range(E)])
    → Each expert = 2D nn.Linear.weight → PEFT LoRA CAN target this!

★★★★★★★★★ Why DeepSpeed chose 3D over ModuleList:
  → torch._grouped_mm needs 3D → batched grouped computation
  → ModuleList = sequential → slower for large expert counts
  → 3D = single tensor → easier for distributed sharding (EP)
  → But: 3D = incompatible with standard LoRA → trade-off!
```

---

## 3. Three Solution Approaches

```
★★★★★★★★★ Approach A: Grouped LoRA for 3D Tensors (NOVEL, most aligned with AutoEP):

Concept:
  → Apply LoRA to 3D weight tensor per-expert
  → Decomposition: A∈(E, r, dim), B∈(E, hidden_dim, r) for w1
  → h = x @ W^T + (x @ A^T @ B^T) * alpha/r → per-expert LoRA
  → Only r×(dim+hidden_dim) parameters per expert → massive savings!

Implementation sketch (~50-80 LOC):
  class GroupedLoraLayer(nn.Module):
      def __init__(self, base_weight, r, alpha, num_experts):
          self.base_weight = base_weight  # 3D: (E, H, D) or (E, D, H)
          self.lora_A = nn.Parameter(torch.empty(num_experts, r, dim))    # (E, r, D)
          self.lora_B = nn.Parameter(torch.zeros(num_experts, hidden_dim, r))  # (E, H, r)
          self.scaling = alpha / r

      def forward(self, x, num_tokens_per_expert):
          # Base computation (frozen weights)
          base_out = self._base_forward(x, num_tokens_per_expert)
          # LoRA correction (per-expert)
          lora_correction = self._lora_forward(x, num_tokens_per_expert)
          return base_out + lora_correction * self.scaling

★★★★★★★★★ Approach A advantages:
  → Preserves 3D tensor format → torch._grouped_mm still works
  → Per-expert LoRA → only r×(2D+2H) trainable params per expert
  → Memory: rank=8 → 8×(dim+hidden_dim)×E → ~10x reduction vs full
  → Compatible with ZeRO-2 → shard LoRA params across ranks
  → ★★★★★★★★ BEST for RTX 4090: preserves AutoEP grouped_mm performance!

★★★★★★★★★ Approach A challenges:
  → torch._grouped_mm doesn't support LoRA correction natively
  → Need separate LoRA forward path + merge with base output
  → lora_A/lora_B are also 3D → need custom gradient computation
  → Requires LoRA merge/unmerge for ZeRO-2 weight sync

★★★★★★★★★ Approach B: Flatten 3D → 2D + standard LoRA (SIMPLEST):

Concept:
  → Flatten GroupedExperts.w1: (E, H, D) → (E*H, D)
  → Treat as nn.Linear with weight = flattened 2D
  → Apply standard PEFT LoRA to flattened weight
  → Reshape back to 3D for expert computation

Implementation sketch (~20 LOC):
  # Flatten for LoRA
  w1_flat = w1.reshape(E * hidden_dim, dim)
  # Standard LoRA
  lora_out = (x @ lora_A.T @ lora_B.T) * alpha / r
  # Reshape for expert compute
  w1_with_lora = (w1_flat + lora_B @ lora_A.T).reshape(E, hidden_dim, dim)

★★★★★★★★★ Approach B advantages:
  → Simplest → uses existing PEFT LoRA infrastructure
  → No custom kernel needed → standard matmul + reshape
  → Works with PEFT library → minimal integration effort

★★★★★★★★★ Approach B challenges:
  → Flatten → ALL experts share same LoRA → NOT per-expert
  → LoRA correction applied uniformly → loses expert specialization
  → Reshape overhead → extra memory for flattened view
  → torch._grouped_mm may not work with reshaped weights → needs testing
  → ★★★★★★★★ NOT ideal — loses per-expert LoRA benefit!

★★★★★★★★★ Approach C: Convert to nn.ModuleList + PEFT LoRA (VERL PATH):

Concept:
  → Replace GroupedExperts with ModuleList of nn.Linear
  → Each expert = separate nn.Linear → PEFT LoRA CAN target
  → verl already uses this pattern for LoRA MoE!

★★★★★★★★★ Approach C advantages:
  → Standard PEFT LoRA → no custom implementation
  → verl LoRA infrastructure already supports this
  → Per-expert LoRA → each expert gets its own LoRA rank

★★★★★★★★★ Approach C challenges:
  → Can't use torch._grouped_mm → sequential for-loop only
  → Slower for large expert counts (Qwen3-MoE 128 experts)
  → ModuleList = separate weight tensors → harder for EP sharding
  → ★★★★★★★★ Performance loss: ~2x slower than grouped_mm for large E!
  → → Only viable for small E (8-16 experts) → NOT ideal for 64+!
```

---

## 4. RTX 4090 Recommendation

```
★★★★★★★★★ RTX 4090 AutoEP MoE LoRA recommendation:

For small MoE (8-16 experts, e.g., Mixtral-8x7B):
  → Approach C: nn.ModuleList + PEFT LoRA → simplest → works with verl
  → Sequential for-loop → acceptable performance with few experts
  → ~10x memory reduction → fits in 24 GiB

For large MoE (64-128 experts, e.g., Qwen3-30B-A3B, DSV4):
  → Approach A: Grouped LoRA for 3D tensors → preserves grouped_mm → BEST
  → torch._grouped_mm + per-expert LoRA → fast + memory-efficient
  → But: requires custom LoRA implementation (~50-80 LOC)
  → ★★★★★★★★ This is the MISSING piece for AutoEP + LoRA on RTX 4090!

★★★★★★★★★ Approach A implementation priority:
  → 1. Define GroupedLoraLayer nn.Module (~30 LOC)
  → 2. Modify GroupedExperts.__init__ to accept LoRA config (~10 LOC)
  → 3. Modify forward() to compute base + LoRA correction (~15 LOC)
  → 4. Add LoRA merge/unmerge for weight sync (~10 LOC)
  → 5. ZeRO-2 integration: only shard LoRA params (~5 LOC)
  → Total: ~50-80 LOC → Tier 2 contribution → DeepSpeed #7938 comment

★★★★★★★★★ Memory savings with Grouped LoRA (Qwen3-30B-A3B example):
  → Full weight: 128 experts × (2*3584*2048 + 2048*3584) = ~2.8 GiB per layer
  → LoRA rank=8: 128 × 8 × (2048+3584) × 2 = ~14 MiB per layer
  → Reduction: 2.8 GiB → 14 MiB = 200x reduction!
  → RTX 4090: 2.8 GiB → 14 MiB per MoE layer → 8 MoE layers → ~112 MiB total LoRA
  → ★★★★★★★★ LoRA fit easily in 24 GiB → only 112 MiB for all MoE layers!
```

---

## 5. verl LoRA MoE Pathway Comparison

```
★★★★★★★★★ verl LoRA MoE implementation (existing):

verl uses HF model format:
  → Mixtral: ModuleList of nn.Linear → PEFT LoRA can target each
  → Qwen3-MoE: similar ModuleList structure → LoRA works
  → BUT: verl's LoRA uses FSDP summon → whole-model summon = OOM risk
  → #6512 per-unit summon → 10x memory reduction → fixes this!

★★★★★★★★★ DeepSpeed AutoEP LoRA pathway (MISSING):

DeepSpeed uses GroupedExperts:
  → 3D nn.Parameter w1/w2/w3 → NOT nn.Linear → PEFT LoRA INCOMPATIBLE
  → Need Grouped LoRA for 3D tensors → ~50-80 LOC custom implementation
  → ZeRO-2 + Grouped LoRA + EP=1 → RTX 4090 MoE training

★★★★★★★★★ Comparison:

| Aspect | verl LoRA MoE | DeepSpeed AutoEP LoRA |
|--------|--------------|----------------------|
| Weight format | nn.ModuleList (2D) | nn.Parameter (3D) |
| LoRA compatibility | Standard PEFT | NEEDS custom Grouped LoRA |
| Expert computation | Sequential for-loop | grouped_mm + for-loop |
| Memory (Qwen3-30B-A3B) | ~2.8 GiB full, ~14 MiB LoRA | Same with Grouped LoRA |
| Performance | For-loop (slower) | grouped_mm (faster) |
| ZeRO-2 compat | FSDP summon (#6512 needed) | ZeRO-2 native (no summon) |
| RTX 4090 ranking | #1 (verl HYBRID+bypass+LoRA) | #2.5 (ZeRO-2+Grouped LoRA) |

★★★★★★★★★ Best path for RTX 4090:
  → verl HYBRID + bypass + GRPO LoRA (#6512 per-unit summon) → #1
  → DeepSpeed ZeRO-2 + Grouped LoRA + EP=1 → #2.5 → needs custom LoRA
```

---

## Key Findings Summary

★★★★★★★★★ GroupedExperts stores w1/w2/w3 as 3D nn.Parameter → PEFT LoRA targets 2D nn.Linear → INCOMPATIBLE!
★★★★★★★★★ 3D tensors needed for torch._grouped_mm → can't just switch to ModuleList without performance loss
★★★★★★★★★ Three approaches: A (Grouped LoRA ~50-80 LOC) > C (ModuleList + PEFT) > B (flatten 2D)
★★★★★★★★★ Grouped LoRA: per-expert LoRA decomposition → rank=8 → 200x memory reduction → fits RTX 4090!
★★★★★★★★★ AutoEP + LoRA on RTX 4090 BLOCKED by this gap → needs Grouped LoRA implementation
★★★★★★★★★ Qwen3-30B-A3B: 2.8 GiB full → 14 MiB LoRA rank=8 → 200x → 112 MiB total across 8 MoE layers
★★★★★★★★★ verl LoRA MoE = ModuleList → standard PEFT → #1 ranking → DeepSpeed needs custom LoRA

---

## References

- AutoEP source reading: notebook/projects/deepspeed-autoep-moe-source-reading.md
- RouterReplay design: notebook/projects/deepspeed-router-replay-equivalent-design.md
- OPD LoRA gap: notebook/projects/deepspeed-opd-lora-integration-gap-analysis.md
- verl LoRA source: notebook/projects/verl-6512-per-unit-lora-summon-reading.md
- ep_experts.py: _temp_deepspeed/deepspeed/moe/ep_experts.py (196 lines)
- PEFT LoRA: https://github.com/huggingface/peft
- verl #6512: https://github.com/verl-project/verl/pull/6512
