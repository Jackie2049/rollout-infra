---
name: interpretability-nav
description: Navigate mechanistic interpretability concepts, tools, and SAE architectures. Quick reference for SAE variants, feature steering methods, circuit discovery, and trust calibration.
triggers:
  - interpretability
  - SAE
  - sparse autoencoder
  - mechanistic interpretability
  - feature steering
  - refusal circuit
  - circuit discovery
  - ACDC
  - attribution patching
  - causal scrubbing
  - ActAdd
  - activation steering
  - RepE
  - trust calibration
  - sycophancy
  - TopK SAE
  - Gated SAE
---

# Mechanistic Interpretability Navigator

## 3-Layer Tool Chain

```
Layer 1: SAE → decompose activations → features → interpretable units
Layer 2: Circuit Discovery → connect features → circuits → behavior mechanisms
Layer 3: Feature Steering → manipulate features → control behavior → applications
```

## SAE 3 Generations

| Gen | Architecture | L0 Control | Dead Features | Absorption | Shrinkage | 2025 Rec |
|-----|-------------|-----------|--------------|-----------|-----------|---------|
| Standard | h=ReLU(W_enc*x+b) | Indirect(λ) | Yes | Yes | Yes | Baseline |
| Gated | gate=ReLU(W_gate*x), mag=ReLU(W_mag*x), h=gate*mag | Better | Reduced | Reduced | No | Frontier |
| TopK | h=TopK(W_enc*x+b, K) | Exact=K | None | None | No | ✓ Best |

### Key Math

- Standard: L = ||x-x̂||² + λ||h||₁
- Gated: L = ||x-x̂||² + λ||gate||₁ (magnitude NOT penalized!)
- TopK: L0=K per token (exact control, no L1 needed)

### RTX 4090 Implementation

- 7B INT4 residual stream dim=4096
- SAE features=16x→65,536 per layer
- TopK L0=50-100 per token
- TransformerLens → hook-based activation extraction
- Training ~2-4h estimated on RTX 4090
- Simulator: `tools/sae_simulator.py`

## Feature Steering 5 Methods

| Method | Approach | Stability | Complexity | Production |
|--------|---------|-----------|-----------|-----------|
| ActAdd | s=a⁺-a⁻ contrastive difference | Low(brittle) | Lowest | Research |
| CAA | Average N contrastive pairs | Medium | Low | Research |
| RepE | PCA directions + multi-layer | Good | Medium | Possible |
| DiffSteer | Functional/task-specific | Good | Medium | Frontier |
| LoRA-Steer | Learned low-rank adapter | ✓Best | High | ✓ Production |

## Circuit Discovery 3-Layer Pipeline

```
ACDC (auto prune edges) → Attribution Patching (gradient 100x faster) → Causal Scrubbing (strict verification)
Fast screening → Approximate refinement → Gold-standard confirmation
```

## Refusal Circuit

```
Harmful request → detection_features → intermediate_features → suppress_output → "I cannot help"
```

Dual-use: transparency vs security tension. Publishing refusal circuit details helps adversaries bypass safety.

## Trust Calibration

- Overtrust: user trusts AI when it shouldn't → error propagation
- Undertrust: user doesn't trust AI when it should → efficiency loss
- Calibrated: appropriate trust matching AI capability → optimal
- Conversational framing increases 30-40% perceived competence → overtrust accelerator
- Simulator: `tools/trust_calibration_simulator.py`

## Key Notes

- `notebook/fundamentals/mechanistic-interpretability-deep-dive.md` — comprehensive theory
- `notebook/fundamentals/ai-safety-guardrails-production-deep-dive.md` — Defense-in-Depth connection
- `notebook/fundamentals/ai-economic-impact-human-ai-interaction-deep-dive.md` — trust/sycophancy/anthropomorphism

## Key Tools

- TransformerLens (neel nanda) — activation extraction + hooks
- steering-vectors — ActAdd for HuggingFace models
- ACDC (TransformerLensOrg) — automated circuit discovery
- Neuron Viewer (Anthropic) — SAE feature visualization
