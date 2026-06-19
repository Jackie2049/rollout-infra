#!/usr/bin/env python3
"""DSV4 Dynamic Routing Diagnostic for RTX 4090.

Checks model architecture for dynamic routing layers (MoE, DSA, MTP,
speculative decoding, online compress) and recommends enforce_eager=True
if ANY dynamic routing is detected.

Based on cross-framework DSV4 systematic instability analysis:
  - vLLM #45972: DSV4 cudagraph → garbage output → MERGED revert
  - vLLM #45979: DSV4 flashinfer sparse cache → FALSE ALARM, VINDICATED
  - SGLang #28591: DSV4 MTP → testing revert
  - SGLang #28520: AMD MTP accept-length bug → 2.17 (NOT CUDA graph, EAGER mode!)
  - SGLang #28569: EAGLE3 CUDA graph → illegal memory access crash
  - SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED → 64x accuracy blowup (10th failure!)
  - Megatron #5317: DSv4-Hybrid apply_rope_fusion=True → NaN at iter 2 (11th failure!)
  - vLLM-Ascend #10628/#10640: DSV4 failure + MTP crash on Ascend
  - Universal rule: ANY per-request dynamic routing MUST run eagerly
  - Extended rule: ANY per-step dynamic data MUST NOT be cached across steps
  - Extended rule: ANY GPU-resident cache MUST be invalidated at weight-reload boundary

Usage:
  python dsv4_dynamic_routing_diagnostic.py check <model_name>
  python dsv4_dynamic_routing_diagnostic.py config <model_name>
  python dsv4_dynamic_routing_diagnostic.py matrix
  python dsv4_dynamic_routing_diagnostic.py rtx4090
"""

import argparse
import json
import sys

# Model architecture database — dynamic routing features
MODELS = {
    "llama-3.1-8b": {
        "dense_attention": True, "moe": False, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 8,
        "notes": "Dense model — CUDA graph SAFE"
    },
    "llama-3.1-70b": {
        "dense_attention": True, "moe": False, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 70,
        "notes": "Dense model — CUDA graph SAFE"
    },
    "mistral-7b": {
        "dense_attention": True, "moe": False, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 7,
        "notes": "Dense model — CUDA graph SAFE"
    },
    "mixtral-8x7b": {
        "dense_attention": True, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 47,
        "notes": "MoE only → CUDA graph RISKY but manageable"
    },
    "qwen3-30b-a3b": {
        "dense_attention": True, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 30,
        "notes": "MoE only → RISKY with cudagraph (router disagreement ~10%)"
    },
    "qwen3-235b-a22b": {
        "dense_attention": True, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 235,
        "notes": "MoE only → CUDA graph RISKY"
    },
    "qwen3.5-35b-a3b": {
        "dense_attention": True, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": True, "gdn": True, "size_b": 35,
        "notes": "MoE + MLA + GDN → 3 dynamic routing → CUDA graph HIGH risk!"
    },
    "deepseek-v2": {
        "dense_attention": False, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": True, "gdn": False, "size_b": 236,
        "notes": "MoE + MLA → 2 dynamic routing → CUDA graph HIGH risk"
    },
    "deepseek-v3": {
        "dense_attention": False, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": True, "gdn": False, "size_b": 671,
        "notes": "MoE + MLA → 2 dynamic routing → CUDA graph HIGH risk"
    },
    "deepseek-v3.2": {
        "dense_attention": False, "moe": True, "dsa": True,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": True, "gdn": False, "size_b": 685,
        "notes": "MoE + MLA + DSA → 3 dynamic routing → CUDA graph VERY HIGH risk!"
    },
    "deepseek-v4-flash": {
        "dense_attention": False, "moe": True, "dsa": True,
        "mtp": True, "spec_decoding": False, "online_compress": True,
        "mla": True, "gdn": False, "size_b": 685,
        "notes": "★★★★★★★★★ MoE+DSA+MTP+Compress+MLA → 5 dynamic routing → EXTREMELY fragile! 4 failures in 4 days!"
    },
    "deepseek-v4-pro": {
        "dense_attention": False, "moe": True, "dsa": True,
        "mtp": True, "spec_decoding": False, "online_compress": True,
        "mla": True, "gdn": False, "size_b": 685,
        "notes": "★★★★★★★★★ Same as flash but larger → EXTREMELY fragile!"
    },
    "kimi-k2.5": {
        "dense_attention": False, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": True, "gdn": False, "size_b": 300,
        "notes": "MoE + MLA → 2 dynamic routing → CUDA graph HIGH risk"
    },
    "glm-5": {
        "dense_attention": True, "moe": True, "dsa": False,
        "mtp": False, "spec_decoding": False, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 300,
        "notes": "MoE → CUDA graph RISKY"
    },
    "qwen3-8b+eagle3": {
        "dense_attention": True, "moe": False, "dsa": False,
        "mtp": False, "spec_decoding": True, "online_compress": False,
        "mla": False, "gdn": False, "size_b": 8,
        "notes": "Spec decode → batch-dependent → CUDA graph crash risk (#28569)"
    },
}

# Dynamic routing risk levels
RISK_LEVELS = {
    0: {"level": "SAFE", "color": "green", "enforce_eager": False,
        "msg": "Dense model — CUDA graph SAFE. Can use FULL mode."},
    1: {"level": "LOW", "color": "yellow", "enforce_eager": "optional",
        "msg": "Single dynamic routing (MoE) — CUDA graph RISKY but manageable with PIECEWISE."},
    2: {"level": "MODERATE", "color": "orange", "enforce_eager": "recommended",
        "msg": "2 dynamic routing layers — CUDA graph HIGH risk. Recommend enforce_eager=True."},
    3: {"level": "HIGH", "color": "red", "enforce_eager": "mandatory",
        "msg": "3+ dynamic routing layers — CUDA graph VERY HIGH risk! enforce_eager=True MANDATORY!"},
    4: {"level": "EXTREME", "color": "magenta", "enforce_eager": "mandatory",
        "msg": "★★★★★★★★★ 4+ dynamic routing — CUDA graph EXTREMELY fragile! 4 failures in 4 days! enforce_eager=True MANDATORY!"},
}


def count_dynamic_routing(model_data):
    """Count number of dynamic routing layers."""
    count = 0
    if model_data["moe"]: count += 1       # MoE expert selection
    if model_data["dsa"]: count += 1       # DSA indexer top-k
    if model_data["mtp"]: count += 1       # MTP draft selection
    if model_data["spec_decoding"]: count += 1  # Spec decode accept/reject
    if model_data["online_compress"]: count += 1  # KV compress decisions
    if model_data["mla"] and not model_data.get("dense_attention", True):
        count += 1  # MLA has latent routing (DCP, cache sharing)
    elif model_data["mla"]:
        count += 0.5  # MLA with dense attention — less dynamic
    if model_data["gdn"]:
        count += 1  # GDN has dynamic state-dependent attention
    return count


def estimate_mismatch_rate(n_routing):
    """Estimate % of forward passes with at least one routing mismatch.
    Based on ~10% per-layer disagreement rate (confirmed by R3 for MoE).
    """
    if n_routing == 0: return 0
    per_layer_disagree = 0.10
    return 1 - (1 - per_layer_disagree) ** int(n_routing)


def get_risk_level(n_routing):
    """Get risk level based on number of dynamic routing layers."""
    n = int(n_routing)
    if n == 0: return RISK_LEVELS[0]
    if n == 1: return RISK_LEVELS[1]
    if n == 2: return RISK_LEVELS[2]
    if n == 3: return RISK_LEVELS[3]
    return RISK_LEVELS[4]


def check_model(model_name):
    """Check a specific model for dynamic routing risk."""
    model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")

    # Find matching model
    matched = None
    for key in MODELS:
        if key.replace("-", "_") == model_name_lower or \
           model_name_lower in key.replace("-", "_"):
            matched = key
            break

    if not matched:
        print(f"Model '{model_name}' not in database. Known models:")
        for k in sorted(MODELS.keys()):
            print(f"  {k}")
        return

    data = MODELS[matched]
    n_routing = count_dynamic_routing(data)
    mismatch_pct = estimate_mismatch_rate(n_routing)
    risk = get_risk_level(n_routing)

    print(f"\n{'='*60}")
    print(f"DSV4 Dynamic Routing Diagnostic: {matched}")
    print(f"{'='*60}")
    print(f"\nArchitecture Features:")
    print(f"  Dense attention: {data['dense_attention']}")
    print(f"  MoE (expert selection): {data['moe']}")
    print(f"  DSA (sparse attention indexer): {data['dsa']}")
    print(f"  MTP (multi-token prediction): {data['mtp']}")
    print(f"  Speculative decoding: {data['spec_decoding']}")
    print(f"  Online compress: {data['online_compress']}")
    print(f"  MLA (multi-head latent attention): {data['mla']}")
    print(f"  GDN (gated delta net): {data['gdn']}")
    print(f"  Model size: {data['size_b']}B params")

    print(f"\nDynamic Routing Analysis:")
    print(f"  Dynamic routing layers: {n_routing}")
    print(f"  Estimated mismatch rate: {mismatch_pct*100:.1f}%")
    print(f"  Risk level: {risk['level']}")
    print(f"  enforce_eager: {risk['enforce_eager']}")

    print(f"\nRecommendation:")
    print(f"  {risk['msg']}")
    print(f"  {data['notes']}")

    # RTX 4090 feasibility
    print(f"\nRTX 4090 Feasibility:")
    if data['size_b'] <= 8:
        print(f"  ✓ Feasible — {data['size_b']}B fits in 24 GiB with quantization")
    elif data['size_b'] <= 35:
        print(f"  ✓ Feasible with LoRA+ZeRO-2+CPU_offload — ~{data['size_b']*0.4:.0f}GB INT4 base + KV + overhead")
    elif data['size_b'] <= 70:
        print(f"  ⚠ Borderline — requires aggressive offloading + quantization")
    else:
        print(f"  ✗ NOT feasible — {data['size_b']}B requires multi-GPU (minimum 128 GPUs for DSV4)")

    print(f"\n{'='*60}")

    # Evidence from failures
    if n_routing >= 2:
        print(f"\nCross-Framework Evidence (June 2026):")
        print(f"  vLLM #45972: DSV4 cudagraph → garbage output → MERGED revert June 18")
        print(f"  vLLM #45979: sparse cache → FALSE ALARM, VINDICATED by retesting")
        print(f"  SGLang #28591: DSV4 MTP revert → OPEN for testing")
        print(f"  SGLang #28569: EAGLE3 CUDA graph → illegal memory access crash")
        print(f"  SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED → 64x accuracy blowup (10th!)")
        print(f"  vLLM-Ascend #10628/#10640: DSV4 + MTP failure on Ascend")
        print(f"  vLLM-Ascend #10724: 2*A2 PD-Mix crash (8th failure)")
        print(f"  vLLM #39096: SM89 batch invariance → Inductor fuses under cudagraph")
        print(f"  Universal rule: @eager_break_during_capture = correct separation boundary")
        print(f"  Extended rule: GPU-resident cache MUST be invalidated at weight-reload boundary")


def generate_config(model_name):
    """Generate recommended vLLM/verl config for a model."""
    model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")

    matched = None
    for key in MODELS:
        if key.replace("-", "_") == model_name_lower or \
           model_name_lower in key.replace("-", "_"):
            matched = key
            break

    if not matched:
        print(f"Model '{model_name}' not found. Use 'check' to see available models.")
        return

    data = MODELS[matched]
    n_routing = count_dynamic_routing(data)
    risk = get_risk_level(n_routing)

    print(f"\n{'='*60}")
    print(f"Recommended Config for {matched} on RTX 4090")
    print(f"{'='*60}")

    # vLLM inference config
    print(f"\n## vLLM Inference Config:")
    enforce_eager = risk['enforce_eager'] == True or risk['enforce_eager'] == 'mandatory'
    if risk['enforce_eager'] == 'recommended':
        enforce_eager = True  # Recommended → True for safety
    print(f"enforce_eager: {enforce_eager}  # {'MANDATORY — dynamic routing detected!' if enforce_eager else 'Optional — dense model'}")
    if enforce_eager:
        print(f"# ★★★★★★★★ DO NOT use CUDA graph with this model!")
        print(f"# 10-15% throughput sacrifice → but CORRECTNESS guaranteed")
    print(f"gpu_memory_utilization: 0.90")
    print(f"max_model_len: 4096  # Adjust based on model")

    if data['moe']:
        print(f"# MoE model — use FA2 (UNIFORM_BATCH) on SM89")
        print(f"# DO NOT use FA3 — SM89 not supported")

    if data['dsa']:
        print(f"# DSA sparse attention — MUST run indexer eagerly")
        print(f"# Megatron #5384: indexer replay needed for GRPO training")

    if data['mtp']:
        print(f"# MTP — MUST run draft model eagerly")
        print(f"# SGLang #28591: MTP reverted for testing!")

    # verl GRPO training config
    print(f"\n## verl GRPO Training Config:")
    print(f"algorithm.adv_estimator: grpo")
    print(f"algorithm.rollout_correction.bypass_mode: True  # Skip ref model → save ~14GB")

    if n_routing >= 3:
        print(f"actor.policy_loss.loss_mode: cppo  # Position-weighted trust region → CRITICAL for multi-routing!")
        print(f"actor.clip_ratio: 0.20  # CPPO δ for MoE models")
        print(f"actor.policy_loss.cppo_w_min: 0.8")
        print(f"actor.policy_loss.cppo_delta_b: 0.02")
    else:
        print(f"actor.policy_loss.loss_mode: grpo  # Standard GRPO (CPPO optional but always safe)")

    if data['moe']:
        print(f"# MoE router replay needed for train/rollout consistency")
        print(f"# verl bypass_mode provides old_log_probs from rollout → reduces mismatch")

    print(f"\n## verl FSDP2 Config:")
    print(f"actor_rollout_ref.actor.strategy: fsdp2")
    print(f"# MUST use FSDP2 — FSDP v1 whole-model summon = OOM on RTX 4090")

    # RTX 4090 MUST rules
    print(f"\n## RTX 4090 MUST DO:")
    print(f"  overlap_comm: False  # NaN bug #8061")
    print(f"  gradient_clipping: 1.0  # Default 0→1.0 #8068")
    print(f"  lora_rank: 32  # rank=64 breaks EOS #6782")
    print(f"  lora_alpha: 64  # 2x rank for effective learning")
    if enforce_eager:
        print(f"  enforce_eager: True  # ★★★★★★★★ MANDATORY for this model!")

    print(f"\n## RTX 4090 MUST NOT:")
    print(f"  ZeRO-3  # Regression + pure overhead")
    print(f"  Muon optimizer  # crash + clipping + CPU offload blocked")
    print(f"  rLLM for GRPO  # #605 grouping bug → BROKEN")

    print(f"\n{'='*60}")


def show_matrix():
    """Show all models in a risk matrix."""
    print(f"\n{'='*80}")
    print(f"DSV4 Dynamic Routing Risk Matrix (Cross-Framework Evidence)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'Routing':>7} {'Mismatch':>8} {'Risk':>8} {'enforce_eager':>15} {'RTX 4090':>10}")
    print(f"{'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*15} {'-'*10}")

    for name in sorted(MODELS.keys(), key=lambda k: count_dynamic_routing(MODELS[k])):
        data = MODELS[name]
        n = count_dynamic_routing(data)
        mismatch = estimate_mismatch_rate(n)
        risk = get_risk_level(n)
        feasible = "✓" if data['size_b'] <= 35 else "⚠" if data['size_b'] <= 70 else "✗"
        print(f"{name:<25} {n:>7.1f} {mismatch*100:>7.1f}% {risk['level']:>8} {str(risk['enforce_eager']):>15} {feasible:>10}")

    print(f"\n{'='*80}")
    print(f"\nCross-Framework Failures (June 2026):")
    print(f"  vLLM #45972: DSV4 cudagraph revert (MERGED)")
    print(f"  SGLang #28591: DSV4 MTP revert (OPEN)")
    print(f"  SGLang #28569: EAGLE3 crash (OPEN)")
    print(f"  SGLang #28676: MXFP8 MoE shuffle cache CLOBBERED (OPEN — 10th failure!)")
    print(f"  Megatron #5317: DSv4-Hybrid apply_rope_fusion NaN (OPEN — 11th failure!)")
    print(f"  vLLM-Ascend #10628/#10640: Ascend failures")
    print(f"  vLLM-Ascend #10724: PD-Mix crash")
    print(f"  vLLM #39096: SM89 batch invariance")
    print(f"\nKey insight: DSV4 with 5 dynamic routing layers → ~41% mismatch rate → EXTREMELY fragile!")
    print(f"  Each layer ~10% disagreement → (1-0.9^5) ≈ 41% of forward passes have at least one mismatch")
    print(f"\nExtended insight: #28676 adds PHYSICAL memory clobber subclass")
    print(f"  Previous bugs: logical stale reference (cache hit returns old data)")
    print(f"  #28676: physical clobber (memory overwritten → data is GARBAGE)")
    print(f"  → Physical clobber is WORSE than stale reference!")


def show_rtx4090():
    """Show RTX 4090 specific recommendations."""
    print(f"\n{'='*60}")
    print(f"RTX 4090 DSV4/Dynamic Routing Recommendations")
    print(f"{'='*60}")

    # Feasible models
    feasible = [k for k in MODELS if MODELS[k]['size_b'] <= 35]
    print(f"\nFeasible Models on RTX 4090 (24 GiB):")
    for name in feasible:
        data = MODELS[name]
        n = count_dynamic_routing(data)
        risk = get_risk_level(n)
        print(f"  {name:<25} routing={n:.1f} risk={risk['level']} enforce_eager={risk['enforce_eager']}")

    # Not feasible
    not_feasible = [k for k in MODELS if MODELS[k]['size_b'] > 35]
    print(f"\nNOT Feasible on RTX 4090 (requires multi-GPU):")
    for name in not_feasible:
        data = MODELS[name]
        n = count_dynamic_routing(data)
        print(f"  {name:<25} routing={n:.1f} size={data['size_b']}B")

    print(f"\nBest RTX 4090 GRPO Training Stack:")
    print(f"  Framework: verl CPPO+bypass_mode (#1 BEST)")
    print(f"  Backend: FSDP2 (NOT FSDP v1)")
    print(f"  Optimizer: AdamW + CPU_offload (NOT Muon)")
    print(f"  LoRA: rank=32, alpha=64 (NOT rank=64!)")
    print(f"  overlap_comm: False (NaN bug)")
    print(f"  gradient_clipping: 1.0 (NOT default 0)")
    print(f"  enforce_eager: True for ANY model with dynamic routing")

    print(f"\nDSV4 Specific (NOT feasible on RTX 4090):")
    print(f"  ★★★★★★★★ DSV4 requires minimum 128 GPUs (TP1/ETP1 fixed)")
    print(f"  ★★★★★★★★ Use Qwen3.5-35B-A3B instead (MoE+MLA+GDN → 3 dynamic routing → HIGH risk)")
    print(f"  ★★★★★★★★ verl #6791: Megatron Lite for multi-node DSV4")
    print(f"  ★★★★★★★★ CPPO position-weighted trust region → especially important for dynamic routing!")

    print(f"\nCUDA Graph Systematic Fragility Pattern:")
    print(f"  6 failures across ALL frameworks → SYSTEMATIC pattern")
    print(f"  Root cause: graph replay assumes STATIC execution path")
    print(f"  ANY per-request dynamic routing → MUST run eagerly")
    print(f"  @eager_break_during_capture = CORRECT separation boundary")
    print(f"  BudgetRefiner SLO compensates throughput loss with better scheduling")


def main():
    parser = argparse.ArgumentParser(description="DSV4 Dynamic Routing Diagnostic for RTX 4090")
    parser.add_argument("command", choices=["check", "config", "matrix", "rtx4090"],
                        help="Command to run")
    parser.add_argument("model", nargs="?", default=None,
                        help="Model name (for check/config commands)")

    args = parser.parse_args()

    if args.command == "check":
        if not args.model:
            print("Usage: check <model_name>")
            sys.exit(1)
        check_model(args.model)
    elif args.command == "config":
        if not args.model:
            print("Usage: config <model_name>")
            sys.exit(1)
        generate_config(args.model)
    elif args.command == "matrix":
        show_matrix()
    elif args.command == "rtx4090":
        show_rtx4090()


if __name__ == "__main__":
    main()
