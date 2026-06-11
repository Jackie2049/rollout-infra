#!/usr/bin/env python3
"""Multi-Token Prediction (MTP) Training + Speculative Decoding Simulator

Simulates MTP training benefits, speculative decoding with MTP heads,
and compares with Eagle/Medusa/n-gram approaches.

Based on:
- Meta MTP (arXiv 2404.19737) — Independent heads
- DeepSeek-V3 (arXiv 2412.19437) — Sequential ngram heads
- Our RTX 4090 speculative decoding benchmarks

Usage:
  python tools/mtp_training_simulator.py --mode training --model 125m --depth 2
  python tools/mtp_training_simulator.py --mode speculative --model 7b --draft mtp --depth 2
  python tools/mtp_training_simulator.py --mode compare --model 7b
  python tools/mtp_training_simulator.py --mode all
"""

import argparse
import json
import math
import math

# ==========================================
# Model configurations
# ==========================================
MODEL_SIZES = {
    '125m': {'params': 125e6, 'hidden': 768, 'layers': 12, 'heads': 12, 'vocab': 50257, 'd_head': 64},
    '0.5b': {'params': 500e6, 'hidden': 1024, 'layers': 24, 'heads': 16, 'vocab': 32000, 'd_head': 64},
    '1.4b': {'params': 1.4e9, 'hidden': 2048, 'layers': 24, 'heads': 16, 'vocab': 32000, 'd_head': 128},
    '7b':   {'params': 7e9,  'hidden': 4096, 'layers': 32, 'heads': 32, 'vocab': 32000, 'd_head': 128},
    '13b':  {'params': 13e9, 'hidden': 5120, 'layers': 40, 'heads': 40, 'vocab': 32000, 'd_head': 128},
}

# ==========================================
# RTX 4090 benchmark data
# ==========================================
RTX4090_DATA = {
    'hbm_bandwidth_gbs': 890.8,
    'gemm_peak_tflops': 169.6,
    'gpu_memory_gb': 24,
    'decode_throughput_b1': 174,  # tok/s for 7B OPT B=1
    'decode_throughput_b32': 2468,  # tok/s for 7B B=32 (from inference calc)
    'spec_ngram_acceptance': 0.40,
    'spec_ngram_speedup': 2.14,
    'spec_eagle_acceptance': 0.85,
    'spec_eagle_speedup': 4.2,
    'spec_medusa_acceptance': 0.55,
    'spec_medusa_speedup': 2.5,
    'spec_untrained_draft_acceptance': 0.18,  # from our benchmark
}

# ==========================================
# MTP Training Simulator
# ==========================================

def simulate_mtp_training(model, depth, steps=500):
    """Simulate MTP training benefits vs NTP baseline"""
    model_info = MODEL_SIZES.get(model, MODEL_SIZES['7b'])
    n_params = model_info['params']
    hidden = model_info['hidden']
    layers = model_info['layers']
    vocab = model_info['vocab']

    # 1. MTP overhead estimation
    # Each depth adds ~1 Transformer layer + projection
    # Extra params per depth ≈ 4 * hidden^2 (standard transformer layer)
    extra_params_per_depth = 4 * hidden ** 2 + hidden * vocab  # Transformer block + output projection
    total_extra_params = depth * extra_params_per_depth
    extra_params_pct = total_extra_params / n_params * 100

    # Extra compute per step ≈ depth * (forward + backward of 1 layer)
    # Standard Transformer layer forward ≈ 6 * hidden^2 * seq_len
    # Relative to full model: 1 layer / total_layers ≈ 1/L
    extra_compute_pct = depth * (1.0 / layers) * 100 * 2  # forward + backward, approximate

    # 2. Training time overhead
    training_time_overhead_pct = extra_compute_pct  # approximate linear scaling

    # 3. Signal density improvement
    # NTP: 1 signal per position
    # MTP: (1 + depth) signals per position
    signal_density_ratio = (1 + depth) / 1.0

    # 4. Performance improvement (based on Meta + DeepSeek-V3 data)
    # Math/Coding tasks benefit most
    if depth == 0:
        perf_improvement_math = 0.0
        perf_improvement_code = 0.0
    elif depth == 1:
        perf_improvement_math = 2.1
        perf_improvement_code = 4.5
    elif depth == 2:
        perf_improvement_math = 4.8
        perf_improvement_code = 8.2
    elif depth == 3:
        perf_improvement_math = 5.0
        perf_improvement_code = 8.5
    else:
        perf_improvement_math = 5.0 + (depth - 3) * 0.3
        perf_improvement_code = 8.5 + (depth - 3) * 0.4

    # NLP improvement is minimal
    perf_improvement_nlp = max(0, depth * 0.5)

    # 5. Convergence speedup (richer gradient signal)
    # MTP converges faster due to D times more gradient signals
    convergence_speedup = min(signal_density_ratio, 1.0 + depth * 0.3)

    # 6. Training cost analysis
    # Extra GPU-hours for MTP training
    # Based on: 12% overhead for D=2 on 671B model
    base_gpu_hours_per_1t_tokens = 188  # DeepSeek-V3: 2.788M GPU-hrs for 14.8T tokens ≈ 188 per 1T
    mtp_gpu_hours_per_1t = base_gpu_hours_per_1t_tokens * (1 + training_time_overhead_pct / 100)

    result = {
        'config': {
            'model': model,
            'mtp_depth': depth,
            'n_params': n_params,
            'steps': steps,
        },
        'mtp_overhead': {
            'extra_params_per_depth': extra_params_per_depth,
            'total_extra_params': total_extra_params,
            'extra_params_pct': round(extra_params_pct, 2),
            'extra_compute_pct': round(extra_compute_pct, 2),
            'training_time_overhead_pct': round(training_time_overhead_pct, 2),
        },
        'training_benefit': {
            'signal_density_ratio': round(signal_density_ratio, 2),
            'convergence_speedup': round(convergence_speedup, 2),
            'perf_improvement_math_pct': perf_improvement_math,
            'perf_improvement_code_pct': perf_improvement_code,
            'perf_improvement_nlp_pct': perf_improvement_nlp,
        },
        'cost_analysis': {
            'base_gpu_hours_per_1t_tokens': base_gpu_hours_per_1t_tokens,
            'mtp_gpu_hours_per_1t_tokens': round(mtp_gpu_hours_per_1t, 1),
            'mtp_cost_increase_pct': round(training_time_overhead_pct, 2),
        },
    }

    return result


# ==========================================
# MTP Speculative Decoding Simulator
# ==========================================

def simulate_speculative_decoding(model, draft_type, depth, batch_size=1):
    """Simulate speculative decoding with various draft approaches"""
    model_info = MODEL_SIZES.get(model, MODEL_SIZES['7b'])
    n_params = model_info['params']

    # Base decode throughput (memory-bound on RTX 4090)
    if model == '7b':
        base_throughput = RTX4090_DATA['decode_throughput_b1'] if batch_size == 1 else RTX4090_DATA['decode_throughput_b32']
    elif model == '125m':
        base_throughput = 174 * 8  # 125M much faster than 7B
    else:
        # Scale estimate: throughput ≈ 174 * sqrt(7e9/n_params)
        base_throughput = 174 * math.sqrt(7e9 / n_params)

    # Draft model configurations
    draft_configs = {
        'mtp_sequential': {
            'acceptance_rate': 0.75,  # DeepSeek-V3 style, sequential ngram
            'draft_time_ms': 0.02,  # 1 layer forward, very cheap
            'verify_time_ms': 5.73,  # same as single decode step (memory-bound)
            'extra_memory_pct': 4,  # MTP heads parameters
            'description': 'DeepSeek-V3 sequential MTP heads (ngram dependency)',
        },
        'mtp_independent': {
            'acceptance_rate': 0.55,  # Meta style, independent heads
            'draft_time_ms': 0.01,  # linear projection, even cheaper
            'verify_time_ms': 5.73,
            'extra_memory_pct': 2,  # simpler heads
            'description': 'Meta independent MTP heads (no causal chain)',
        },
        'eagle': {
            'acceptance_rate': RTX4090_DATA['spec_eagle_acceptance'],
            'draft_time_ms': 0.5,  # 1 layer TF + feature-level, moderate
            'verify_time_ms': 5.73,
            'extra_memory_pct': 3,  # ~0.5GB for 7B
            'description': 'Eagle feature-level draft (best for existing models)',
        },
        'medusa': {
            'acceptance_rate': RTX4090_DATA['spec_medusa_acceptance'],
            'draft_time_ms': 0.01,  # MLP heads, very cheap
            'verify_time_ms': 5.73,
            'extra_memory_pct': 1,
            'description': 'Medusa MLP heads (lightweight, independent)',
        },
        'ngram': {
            'acceptance_rate': RTX4090_DATA['spec_ngram_acceptance'],
            'draft_time_ms': 0.001,  # lookup, essentially zero
            'verify_time_ms': 5.73,
            'extra_memory_pct': 0,
            'description': 'N-gram context matching (zero cost)',
        },
        'untrained': {
            'acceptance_rate': RTX4090_DATA['spec_untrained_draft_acceptance'],
            'draft_time_ms': 2.0,  # small model forward
            'verify_time_ms': 5.73,
            'extra_memory_pct': 10,  # separate small model
            'description': 'Untrained draft (DISASTER — negative speedup)',
        },
    }

    draft_config = draft_configs.get(draft_type, draft_configs['ngram'])
    alpha = draft_config['acceptance_rate']

    # Speculative decoding speedup calculation
    # Average tokens per step = 1 + alpha + alpha^2 + ... + alpha^D
    tokens_per_step = sum(alpha**i for i in range(depth + 1))  # including main token (i=0)

    # Time per speculative step
    # Draft: depth * draft_time_ms
    # Verify: verify_time_ms (one forward for all positions)
    draft_time = depth * draft_config['draft_time_ms']
    total_time_ms = draft_time + draft_config['verify_time_ms']

    # Time per non-speculative step (just decode)
    base_time_ms = draft_config['verify_time_ms']

    # Effective throughput
    spec_throughput = tokens_per_step / (total_time_ms / 1000)  # tokens/second
    base_throughput_calc = 1.0 / (base_time_ms / 1000)  # tokens/second

    speedup = spec_throughput / base_throughput_calc

    # Special case: n-gram and zero-cost drafts can still speedup even with low α
    # because draft cost is negligible → speedup ≈ tokens_per_step / (1 + negligible)
    # But for untrained draft with significant compute, α<0.5 is truly negative
    if draft_config['draft_time_ms'] > 1.0 and alpha < 0.5:
        # Significant draft cost + low acceptance → negative speedup
        # Most drafts rejected → only 1 token guaranteed + wasted draft time
        rejection_overhead = depth * draft_config['draft_time_ms']
        effective_time_ms = base_time_ms + rejection_overhead
        speedup = base_time_ms / effective_time_ms  # < 1 = negative!

    # Memory overhead
    model_mem_gb = n_params * 2 / 1e9  # BF16
    extra_mem_gb = model_mem_gb * draft_config['extra_memory_pct'] / 100

    result = {
        'config': {
            'model': model,
            'draft_type': draft_type,
            'depth': depth,
            'batch_size': batch_size,
            'draft_description': draft_config['description'],
        },
        'speculative_decoding': {
            'acceptance_rate': alpha,
            'tokens_per_step': round(tokens_per_step, 3),
            'draft_time_ms': round(draft_time, 3),
            'verify_time_ms': draft_config['verify_time_ms'],
            'total_time_ms': round(total_time_ms, 3),
            'base_time_ms': base_time_ms,
            'speedup': round(speedup, 3),
            'extra_memory_gb': round(extra_mem_gb, 3),
            'extra_memory_pct': draft_config['extra_memory_pct'],
        },
        'recommendation': alpha >= 0.7,
    }

    # Add recommendation text
    if speedup < 1.0:
        result['recommendation_text'] = "NEGATIVE speedup! Do NOT use speculative decoding with this draft."
    elif alpha < 0.5 and draft_config['draft_time_ms'] > 0.5:
        result['recommendation_text'] = "Low acceptance + high draft cost. Likely negative or marginal speedup."
    elif alpha < 0.5 and draft_config['draft_time_ms'] < 0.01:
        result['recommendation_text'] = "Low acceptance but zero draft cost → still positive speedup. Simplest option (n-gram)."
    elif alpha < 0.7:
        result['recommendation_text'] = "Marginal benefit. Consider better draft model (Eagle or MTP sequential)."
    elif alpha >= 0.85:
        result['recommendation_text'] = "Recommended! High acceptance rate → significant speedup."
    else:
        result['recommendation_text'] = "Acceptable. MTP-style sequential draft recommended for new models."

    return result


# ==========================================
# Comparison Simulator
# ==========================================

def simulate_compare(model):
    """Compare all speculative decoding approaches"""
    results = {}

    # 1. Training comparison (MTP depths 0-4)
    training_results = {}
    for d in range(5):
        key = f"ntp_mtp_d{d}"
        training_results[key] = simulate_mtp_training(model, d)
    results['training_comparison'] = training_results

    # 2. Speculative decoding comparison
    draft_types = ['mtp_sequential', 'mtp_independent', 'eagle', 'medusa', 'ngram', 'untrained']
    spec_results = {}
    for dt in draft_types:
        for depth in [2, 3, 5]:
            key = f"{dt}_d{depth}"
            spec_results[key] = simulate_speculative_decoding(model, dt, depth)
    results['spec_comparison'] = spec_results

    # 3. ROI analysis: MTP training cost vs inference benefit
    mtp_training = simulate_mtp_training(model, 2)
    mtp_spec = simulate_speculative_decoding(model, 'mtp_sequential', 2)

    training_overhead_pct = mtp_training['mtp_overhead']['training_time_overhead_pct']
    inference_speedup = mtp_spec['speculative_decoding']['speedup']

    # Net ROI: if training costs X% more, but inference is 1.8x faster
    # Serving 100M requests: how many GPU-hours saved?
    # Base: 100M requests * avg_500_tokens / 174 tok/s / 3600 ≈ 160K GPU-hours
    base_serving_gpu_hours = 100e6 * 500 / 174 / 3600
    mtp_serving_gpu_hours = base_serving_gpu_hours / inference_speedup
    serving_savings = base_serving_gpu_hours - mtp_serving_gpu_hours

    # Training extra cost (for 1T tokens)
    base_training_gpu_hours = mtp_training['cost_analysis']['base_gpu_hours_per_1t_tokens']
    mtp_training_gpu_hours = mtp_training['cost_analysis']['mtp_gpu_hours_per_1t_tokens']
    training_extra = mtp_training_gpu_hours - base_training_gpu_hours

    results['roi_analysis'] = {
        'training_overhead_pct': training_overhead_pct,
        'inference_speedup': round(inference_speedup, 3),
        'base_serving_gpu_hours_100m_requests': round(base_serving_gpu_hours, 1),
        'mtp_serving_gpu_hours_100m_requests': round(mtp_serving_gpu_hours, 1),
        'serving_savings_gpu_hours': round(serving_savings, 1),
        'training_extra_gpu_hours_per_1t': round(training_extra, 1),
        'roi_ratio': round(serving_savings / max(training_extra, 1), 1),
        'roi_conclusion': f"MTP training costs {training_extra:.0f} extra GPU-hrs per 1T tokens, "
                         f"but saves {serving_savings:.0f} GPU-hrs serving 100M requests. "
                         f"ROI = {serving_savings / max(training_extra, 1):.0f}x. MTP is worth it if serving volume is high!",
    }

    # 4. RTX 4090 recommendation matrix
    results['rtx4090_recommendation'] = {
        'existing_7b_no_mtp': {
            'best': 'Eagle d=5',
            'speedup': 4.2,
            'cost': '0.5GB extra',
            'reason': 'Plugin-style, highest acceptance rate, best production choice',
        },
        'existing_7b_simplest': {
            'best': 'n-gram d=3',
            'speedup': 2.14,
            'cost': 'zero',
            'reason': 'No draft model needed, simplest setup',
        },
        'new_training_7b': {
            'best': 'MTP D=2 + n-gram',
            'speedup': 2.14,  # baseline n-gram, MTP adds more
            'cost': '+4% params during training',
            'reason': 'Dual benefit: training quality +4-8% + inference 1.8x via MTP heads',
        },
        'small_model_125m': {
            'best': 'No speculative decoding',
            'speedup': 1.0,
            'cost': 'none',
            'reason': 'Small model decode already fast, spec decode overhead exceeds benefit',
        },
    }

    return results


# ==========================================
# MTP Architecture Simulator
# ==========================================

def simulate_mtp_architecture(model, depth, mode='sequential'):
    """Simulate MTP architecture details"""
    model_info = MODEL_SIZES.get(model, MODEL_SIZES['7b'])
    hidden = model_info['hidden']
    layers = model_info['layers']
    vocab = model_info['vocab']

    if mode == 'sequential':
        # DeepSeek-V3 style: each depth has 1 Transformer block
        # Input: trunk hidden + embedding residual from previous prediction
        per_depth_params = 4 * hidden**2 + hidden * vocab  # 1 TF block + output projection
        total_extra_params = depth * per_depth_params
        description = "Sequential ngram-dependent heads (DeepSeek-V3 style)"
        acceptance_rate = 0.75  # higher due to causal chain
    elif mode == 'independent':
        # Meta style: independent heads, no causal chain
        per_depth_params = hidden * vocab  # just linear projection
        total_extra_params = depth * per_depth_params
        description = "Independent heads (Meta MTP style)"
        acceptance_rate = 0.55  # lower, no causal chain
    else:
        raise ValueError(f"Unknown mode: {mode}")

    extra_params_pct = total_extra_params / model_info['params'] * 100

    # Loss formulation
    loss_coeffs = {}
    for d in range(1, depth + 1):
        # Weight decay: lambda_1 = 1, lambda_d < 1 for d > 1
        if d == 1:
            loss_coeffs[f'lambda_{d}'] = 1.0
        else:
            loss_coeffs[f'lambda_{d}'] = 0.5 ** (d - 1)  # geometric decay

    result = {
        'architecture': {
            'mode': mode,
            'description': description,
            'depth': depth,
            'per_depth_params': per_depth_params,
            'total_extra_params': total_extra_params,
            'extra_params_pct': round(extra_params_pct, 2),
            'acceptance_rate': acceptance_rate,
        },
        'loss_formulation': {
            'type': 'composite_cross_entropy',
            'formula': f'L_total = L_NTP + Σ λ_d * L_d for d=1..{depth}',
            'coefficients': loss_coeffs,
        },
        'training_config': {
            'shared_embedding': True,
            'shared_output_head': True,
            'depth_transformer_layers': 1 if mode == 'sequential' else 0,
            'embedding_residual': mode == 'sequential',
        },
    }

    return result


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='MTP Training + Speculative Decoding Simulator')
    parser.add_argument('--mode', default='compare', choices=['training', 'speculative', 'compare', 'architecture', 'all'])
    parser.add_argument('--model', default='7b', choices=list(MODEL_SIZES.keys()))
    parser.add_argument('--depth', type=int, default=2, help='MTP depth (number of future tokens)')
    parser.add_argument('--draft', default='mtp_sequential',
                       choices=['mtp_sequential', 'mtp_independent', 'eagle', 'medusa', 'ngram', 'untrained'])
    parser.add_argument('--mtp_mode', default='sequential', choices=['sequential', 'independent'])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', default='results/mtp_simulator.json')
    args = parser.parse_args()

    all_results = {}

    if args.mode == 'training':
        for d in range(5):
            key = f"depth_{d}"
            all_results[key] = simulate_mtp_training(args.model, d)
        # Also show specific depth
        all_results['specific'] = simulate_mtp_training(args.model, args.depth)

    elif args.mode == 'speculative':
        all_results[args.draft] = simulate_speculative_decoding(args.model, args.draft, args.depth, args.batch)

    elif args.mode == 'architecture':
        all_results['sequential'] = simulate_mtp_architecture(args.model, args.depth, 'sequential')
        all_results['independent'] = simulate_mtp_architecture(args.model, args.depth, 'independent')

    elif args.mode == 'compare':
        all_results = simulate_compare(args.model)

    elif args.mode == 'all':
        # Run all simulations
        all_results['training_depth_sweep'] = {}
        for d in range(5):
            all_results['training_depth_sweep'][f'd{d}'] = simulate_mtp_training(args.model, d)

        all_results['spec_comparison'] = {}
        draft_types = ['mtp_sequential', 'mtp_independent', 'eagle', 'medusa', 'ngram', 'untrained']
        for dt in draft_types:
            for depth in [2, 3, 5]:
                all_results['spec_comparison'][f'{dt}_d{depth}'] = simulate_speculative_decoding(args.model, dt, depth)

        all_results['architecture_sequential'] = simulate_mtp_architecture(args.model, 2, 'sequential')
        all_results['architecture_independent'] = simulate_mtp_architecture(args.model, 2, 'independent')

        all_results['roi_analysis'] = simulate_compare(args.model)['roi_analysis']
        all_results['rtx4090_recommendation'] = simulate_compare(args.model)['rtx4090_recommendation']

    # Print summary
    print("=" * 70)
    print("MTP Training + Speculative Decoding Simulator")
    print("=" * 70)

    if args.mode == 'training' or args.mode == 'all':
        print("\n--- MTP Training Benefits ---")
        for d in range(5):
            r = all_results.get('training_depth_sweep', all_results).get(f'd{d}',
                  all_results.get(f'depth_{d}', {}))
            if isinstance(r, dict) and 'training_benefit' in r:
                print(f"  D={d}: overhead={r['mtp_overhead']['training_time_overhead_pct']:.1f}%, "
                      f"math+{r['training_benefit']['perf_improvement_math_pct']}%, "
                      f"code+{r['training_benefit']['perf_improvement_code_pct']}%, "
                      f"convergence={r['training_benefit']['convergence_speedup']}x")

    if args.mode == 'speculative' or args.mode == 'all' or args.mode == 'compare':
        print("\n--- Speculative Decoding Comparison ---")
        spec_data = all_results.get('spec_comparison', {})
        if args.mode == 'speculative':
            spec_data = {args.draft: all_results[args.draft]}
        for key, r in spec_data.items():
            if isinstance(r, dict) and 'speculative_decoding' in r:
                alpha = r['speculative_decoding']['acceptance_rate']
                speedup = r['speculative_decoding']['speedup']
                rec = r.get('recommendation_text', '')
                print(f"  {key}: α={alpha:.2f}, speedup={speedup:.2f}x, {rec}")

    if args.mode == 'compare' or args.mode == 'all':
        roi = all_results.get('roi_analysis', {})
        if roi:
            print(f"\n--- ROI Analysis ---")
            print(f"  Training overhead: {roi.get('training_overhead_pct', 0):.1f}%")
            print(f"  Inference speedup: {roi.get('inference_speedup', 0):.2f}x")
            print(f"  {roi.get('roi_conclusion', '')}")

        rec = all_results.get('rtx4090_recommendation', {})
        if rec:
            print(f"\n--- RTX 4090 Recommendation ---")
            for scenario, info in rec.items():
                print(f"  {scenario}: {info['best']} ({info['speedup']}x) — {info['reason']}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()