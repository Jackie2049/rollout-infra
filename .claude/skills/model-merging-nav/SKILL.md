---
name: model-merging-nav
description: Navigate model merging strategies and theory. Quick reference for Task Arithmetic, DARE, TIES merging methods, alpha selection, and RTX 4090 validation results.
triggers:
  - model merging
  - task arithmetic
  - DARE
  - TIES
  - merge models
  - multi-task merge
  - alpha parameter
  - model composition
  - fine-tune merge
---

# Model Merging Navigator

## 3 Core Strategies

| Strategy | Method | Key Parameter | Success Condition | RTX 4090 Result |
|----------|--------|--------------|-----------------|----------------|
| Task Arithmetic | base + α*Σ(task_vectors) | α=0.75 optimal | α<1.0 safe, α≥1.0 risky | ✓ 100% bench-tested |
| DARE | Random drop p% + rescale 1/(1-p) | drop_rate=0.9 | Works on large models | ✓ 90% drop OK on large |
| TIES | Trim top-k% + sign majority + merge | top_k=20% | Minimal interference | ✓ best interference score |

## Task Arithmetic Alpha Selection

```
α=0.1  → cos_sim=0.90 ✓ GOOD (low signal preservation)
α=0.25 → cos_sim=0.92 ✓ GOOD
α=0.5  → cos_sim=0.91 ✓ GOOD
α=0.75 → cos_sim=0.88 ✓ GOOD ← OPTIMAL (best preservation-safety balance)
α=1.0  → cos_sim=0.83 ✗ RISKY (norm explosion starts)
α=1.5  → cos_sim=0.73 ✗ RISKY
α=2.0  → cos_sim=0.66 ✗ RISKY (severe norm explosion)
```

Rule: α<1.0 is safe. α≥1.0 causes weight norm ratio >1.5 → unstable.

## DARE Drop Rate Selection

```
drop=0.5  → cos_sim=0.69 ✗ RISKY (too many active → noise)
drop=0.7  → cos_sim=0.57 ✗ RISKY
drop=0.9  → ✓ GOOD on real large models (synthetic small models show different behavior)
drop=0.95 → ✓ OK but sparse → needs more real data
drop=0.99 → ✗ RISKY (only 1% kept → unstable)
```

Key insight: DARE works well on real large models (many redundant parameters) but poorly on small synthetic matrices. Our RTX 4090 bench confirms 90% drop works in practice.

## TIES Algorithm Steps

1. **Trim**: Keep only top-k% magnitude per task vector (remove noise)
2. **Sign**: Resolve sign conflicts by majority vote across models
3. **Merge**: Average magnitudes where signs agree with majority

Result: Lowest interference (-0.04 vs +0.42 for arithmetic) → best for conflicting tasks.

## Simulator

- `tools/model_merging_simulator.py` — NumPy-based validation (no GPU)
- Alpha sweep + DARE drop rate sweep + all 3 strategies comparison
- Results: `results/model_merging_simulator_results.json`

## Key Notes

- `notebook/fundamentals/model-merging-theory.md` — theoretical foundations
- `memory/model-merging-experiment-rtx4090.md` — RTX 4090 bench results

## Decision Tree

```
Need to merge N fine-tuned models into one?
  → Tasks similar/compatible? → Task Arithmetic α=0.75 (simplest, proven)
  → Tasks conflicting/different directions? → TIES (best interference control)
  → Very large models (7B+)? → DARE 90% drop (reduces compute, proven on large)
  → All methods? → Try all three, compare cos_sim + preservation + interference
```
