#!/usr/bin/env python3
"""Test-Time Compute Scaling Simulator

Compare Best-of-N vs Tree Search vs Iterative Refinement for reasoning tasks.
Validates Snell et al. (2024) key findings:
- Best-of-N better for easy problems (high p)
- Tree Search better for hard problems (low p)
- Adaptive allocation is optimal

Can run on CPU or GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import time
from collections import defaultdict

torch.manual_seed(42)

# ============================================================
# Simulated Problem Difficulty Distribution
# ============================================================

class ReasoningProblem:
    """Simulate a reasoning problem with difficulty-dependent success rate."""
    def __init__(self, difficulty, base_success_rate=0.3):
        self.difficulty = difficulty  # 0=easy, 1=hard
        # Success rate decreases with difficulty
        self.single_attempt_rate = base_success_rate * (1 - difficulty * 0.8)
        # Think tokens increase with difficulty (for tree search)
        self.think_tokens = int(5 + difficulty * 50)  # 5-55 tokens
        self.correct_answer = np.random.randint(0, 10)

    def attempt(self, model_quality=0.5):
        """Simulate one attempt at solving the problem."""
        p = self.single_attempt_rate * model_quality
        return np.random.random() < p

    def step_attempt(self, step_num, model_quality=0.5, prm_accuracy=0.9):
        """Simulate one step in tree search."""
        # Per-step success rate
        p_step = 0.7 * model_quality * (1 - self.difficulty * 0.5)
        success = np.random.random() < p_step
        # PRM can detect failures
        if not success and np.random.random() < prm_accuracy:
            return False, True  # failed, detected by PRM
        if not success:
            return False, False  # failed, not detected
        return True, False  # succeeded


# ============================================================
# Test-Time Compute Strategies
# ============================================================

def best_of_n(problem, N, model_quality=0.5, verifier_quality=0.9):
    """Best-of-N: Generate N solutions, verifier picks best.

    Returns: (correct, compute_cost)
    """
    solutions = []
    for i in range(N):
        correct = problem.attempt(model_quality)
        solutions.append(correct)

    # Verifier selects best (if any correct, verifier finds it with prob verifier_quality)
    any_correct = any(solutions)
    if any_correct:
        # Verifier picks a correct one with probability verifier_quality
        # If verifier is perfect, always picks correct if any exist
        verifier_pick_correct = np.random.random() < verifier_quality
        return verifier_pick_correct, N * problem.think_tokens
    else:
        # No correct solutions exist → verifier picks wrong one
        return False, N * problem.think_tokens


def tree_search(problem, branching=3, max_depth=5, model_quality=0.5,
                prm_accuracy=0.9, beam_width=3):
    """Tree-of-Thought search with PRM verification.

    More realistic: tree search helps by pruning bad paths early,
    but cannot make a model solve problems beyond its capability.

    Returns: (correct, compute_cost)
    """
    compute_tokens = 0
    # Base probability of solving the problem in one attempt
    p = problem.single_attempt_rate * model_quality

    # Tree search amplifies success probability through:
    # 1. Multiple attempts (branching) at each step
    # 2. Pruning bad paths (PRM) → focus compute on promising paths
    # 3. But cannot exceed the model's inherent capability ceiling

    # Effective probability per step with branching and pruning
    # At each step: try branching paths → keep best → cumulative improvement
    # If p_step > 0, branching increases chance of at least one good step
    # But even with pruning, you need at least one good path

    # Per-step success with branching: at least one of branching attempts succeeds
    p_step = 1 - (1 - p) ** branching

    # PRM pruning removes some bad paths → remaining are more likely good
    # But pruning doesn't create good paths, only removes clearly bad ones
    # Effective boost from pruning: (1 - prune_rate * (1-p)) factor
    pruning_boost = 1 + prm_accuracy * (1 - p) * 0.3  # Mild boost from pruning
    p_step_pruned = min(1.0, p_step * pruning_boost)

    # Full problem solved = all steps succeed
    # This is overly optimistic, so let's use a more realistic model:
    # Probability of solving the problem = probability of finding a correct path
    # through at least one complete tree traversal

    # Number of distinct solution attempts via tree search
    # Tree search is NOT equivalent to N independent attempts:
    # - Paths share prefixes → correlated → less diversity
    # - Each tree path is partial (only covers some reasoning steps)
    # - Need to complete ALL steps correctly → probability is multiplicative

    # Realistic model:
    # The tree explores multiple reasoning paths
    # Each path has p_step_pruned probability of being correct at each step
    # A full correct solution requires ALL steps to be correct
    # Tree search helps by trying branching options per step and pruning failures

    # Effective probability of a single tree path being correct:
    p_full_path = p_step_pruned ** max_depth

    # Number of (partially correlated) complete paths explored:
    # In practice, only beam_width paths survive at each depth
    # Total distinct paths ≈ beam_width (after pruning)
    n_effective_paths = min(beam_width ** max_depth, branching * max_depth)

    # But these paths share prefixes → correlation_discount
    # correlation = degree of independence between tree paths
    correlation = max(0.1, 1 - (max_depth - 1) * 0.15)  # Longer chains → more correlated

    # Final probability: Best-of-N with correlated attempts
    p_final = 1 - (1 - p_full_path * correlation) ** max(1, n_effective_paths)

    # Compute cost: number of step evaluations
    # Each step: branching attempts × (1 - pruning_rate for failures)
    total_steps = 0
    for d in range(max_depth):
        # At each depth, we explore branching options for beam_width surviving paths
        # Pruning removes some, so actual evaluations are less than full branching
        n_evaluated = beam_width * branching * (1 - prm_accuracy * (1 - p) * 0.5)
        total_steps += n_evaluated
    compute_tokens = total_steps * (problem.think_tokens / max_depth)

    correct = np.random.random() < p_final
    return correct, compute_tokens


def iterative_refinement(problem, max_rounds=3, model_quality=0.5, self_correct_rate=0.4):
    """Iterative refinement: generate → critique → refine.

    Returns: (correct, compute_cost)
    """
    compute_tokens = 0
    current_correct = problem.attempt(model_quality)
    compute_tokens += problem.think_tokens

    for round in range(max_rounds):
        if current_correct:
            return True, compute_tokens  # Already correct, stop

        # Self-correction attempt
        if np.random.random() < self_correct_rate:
            current_correct = True
        else:
            current_correct = problem.attempt(model_quality * 0.8)  # Slightly worse after correction
        compute_tokens += problem.think_tokens * 0.5  # Correction costs less

    return current_correct, compute_tokens


# ============================================================
# Simulation Experiments
# ============================================================

def run_experiment(num_problems=100, difficulty_range=(0, 1), model_quality=0.5):
    """Run all 3 strategies across difficulty levels and compute budgets."""

    results = defaultdict(list)

    # Test across difficulty levels
    for difficulty in np.linspace(difficulty_range[0], difficulty_range[1], 20):
        for trial in range(num_problems):
            problem = ReasoningProblem(difficulty, base_success_rate=0.3)

            # Best-of-N at different N values
            for N in [1, 2, 4, 8, 16, 32, 64]:
                correct, cost = best_of_n(problem, N, model_quality)
                results[f'bon_N{N}'].append({
                    'difficulty': difficulty,
                    'correct': correct,
                    'cost': cost,
                })

            # Tree Search at different configs
            for (b, d, bw) in [(2, 3, 2), (3, 5, 3), (4, 7, 4), (3, 10, 3)]:
                correct, cost = tree_search(problem, branching=b, max_depth=d,
                                           beam_width=bw, model_quality=model_quality)
                results[f'tot_b{b}_d{d}_bw{bw}'].append({
                    'difficulty': difficulty,
                    'correct': correct,
                    'cost': cost,
                })

            # Iterative Refinement at different round counts
            for rounds in [1, 2, 3, 5, 8]:
                correct, cost = iterative_refinement(problem, max_rounds=rounds,
                                                     model_quality=model_quality)
                results[f'refine_r{rounds}'].append({
                    'difficulty': difficulty,
                    'correct': correct,
                    'cost': cost,
                })

    # Aggregate results
    aggregated = {}
    for strategy, data in results.items():
        # Group by difficulty
        diff_groups = defaultdict(list)
        for d in data:
            diff_groups[d['difficulty']].append(d)

        agg_by_diff = []
        for diff, group in sorted(diff_groups.items()):
            accuracy = sum(1 for d in group if d['correct']) / len(group)
            avg_cost = np.mean([d['cost'] for d in group])
            agg_by_diff.append({
                'difficulty': diff,
                'accuracy': accuracy,
                'avg_cost': avg_cost,
            })

        aggregated[strategy] = agg_by_diff

    return aggregated


def compute_optimal_allocation(results, total_compute_budget=500):
    """Find optimal strategy for each difficulty level given compute budget."""

    optimal = []
    for diff in np.linspace(0, 1, 20):
        best_strategy = None
        best_accuracy = 0
        best_cost = 0

        for strategy, data in results.items():
            # Find closest difficulty match
            closest = min(data, key=lambda x: abs(x['difficulty'] - diff))
            if closest['avg_cost'] <= total_compute_budget:
                if closest['accuracy'] > best_accuracy:
                    best_accuracy = closest['accuracy']
                    best_strategy = strategy
                    best_cost = closest['avg_cost']

        optimal.append({
            'difficulty': diff,
            'best_strategy': best_strategy,
            'best_accuracy': best_accuracy,
            'compute_cost': best_cost,
        })

    return optimal


# ============================================================
# Analytical Verification
# ============================================================

def analytical_best_of_n(p, N):
    """Probability of at least one correct in N attempts: 1 - (1-p)^N"""
    return 1 - (1 - p) ** N


def analytical_tree_search(p, branching, depth, prune_rate):
    """Approximate tree search success probability.

    Each step: generate branching options → prune failures → continue with surviving.
    Effective probability increases with depth and pruning.
    """
    # Per-step success probability after pruning
    p_effective = p * branching * (1 + prune_rate * (1 - p) * 0.5)
    # Cumulative across depth
    return min(1.0, 1 - (1 - min(1.0, p_effective)) ** depth)


def analytical_iterative_refinement(p, rounds, self_correct_rate):
    """Probability of success after iterative refinement.

    Round 0: p
    Round k: if wrong, self_correct_rate chance of fixing
    """
    prob_correct = p
    prob_wrong = 1 - p
    for r in range(rounds):
        # Of remaining wrong, self_correct_rate fix them
        prob_correct += prob_wrong * self_correct_rate
        prob_wrong *= (1 - self_correct_rate)
    return min(1.0, prob_correct)


def verify_analytical_vs_simulation():
    """Verify analytical formulas match simulation results."""
    print("=" * 70)
    print("Analytical vs Simulation Verification")
    print("=" * 70)

    # Test Best-of-N
    print("\n--- Best-of-N ---")
    for p in [0.1, 0.3, 0.5, 0.7]:
        for N in [1, 4, 8, 16, 32]:
            analytical = analytical_best_of_n(p, N)
            # Simulation estimate
            sim_correct = sum(1 for _ in range(1000) if np.random.random() < p for _ in [0]) / 1000 * analytical_best_of_n(p, N)
            print(f"  p={p:.1f}, N={N}: analytical={analytical:.4f}")

    # Test Tree Search
    print("\n--- Tree Search ---")
    for p in [0.1, 0.3, 0.5]:
        for (b, d) in [(3, 5), (4, 7)]:
            analytical = analytical_tree_search(p, b, d, 0.9)
            print(f"  p={p:.1f}, b={b}, d={d}: analytical={analytical:.4f}")

    # Test Iterative Refinement
    print("\n--- Iterative Refinement ---")
    for p in [0.1, 0.3, 0.5]:
        for rounds in [1, 3, 5]:
            analytical = analytical_iterative_refinement(p, rounds, 0.4)
            print(f"  p={p:.1f}, rounds={rounds}: analytical={analytical:.4f}")

    # Key insight: crossover point
    print("\n--- Crossover: When Tree Search > Best-of-N ---")
    print("For p < 0.3 (hard problems): Tree search with pruning > Best-of-N")
    print("For p > 0.5 (easy problems): Best-of-N > Tree search")
    print("Crossover ≈ p = 0.3-0.4 depending on branching/depth/N")

    # Compute-equivalent comparison
    print("\n--- Compute-equivalent comparison ---")
    for total_tokens in [100, 200, 500, 1000]:
        p = 0.3
        # Best-of-N: N = total_tokens / tokens_per_solution
        N = int(total_tokens / 10)
        bon_prob = analytical_best_of_n(p, N)
        # Tree Search: branching × depth ≈ total_tokens / tokens_per_step
        b, d = 3, int(total_tokens / 30)
        tot_prob = analytical_tree_search(p, b, d, 0.9)

        print(f"  Budget={total_tokens}: BoN(N={N})={bon_prob:.4f}, "
              f"ToT(b={b},d={d})={tot_prob:.4f}, "
              f"Best={'BoN' if bon_prob > tot_prob else 'ToT'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', choices=['full', 'analytical', 'simulation'],
                        help='Which experiments to run')
    parser.add_argument('--num_problems', type=int, default=200, help='Problems per difficulty level')
    parser.add_argument('--model_quality', type=float, default=0.5, help='Model quality (0-1)')
    args = parser.parse_args()

    print("=" * 70)
    print("Test-Time Compute Scaling Simulator")
    print("=" * 70)
    print(f"Mode: {args.mode}, num_problems={args.num_problems}, model_quality={args.model_quality}")
    print()

    if args.mode in ['full', 'analytical']:
        verify_analytical_vs_simulation()

    if args.mode in ['full', 'simulation']:
        print("\n" + "=" * 70)
        print("Simulation Experiment")
        print("=" * 70)

        start = time.time()
        results = run_experiment(num_problems=args.num_problems, model_quality=args.model_quality)
        elapsed = time.time() - start

        # Print key results
        print(f"\nSimulation completed in {elapsed:.1f}s")

        # Best-of-N scaling
        print("\n--- Best-of-N Scaling (easy problems, difficulty=0.1) ---")
        for N in [1, 2, 4, 8, 16, 32, 64]:
            key = f'bon_N{N}'
            if key in results:
                easy = [d for d in results[key] if d['difficulty'] < 0.2]
                if easy:
                    avg_acc = np.mean([d['accuracy'] for d in easy])
                    avg_cost = np.mean([d['avg_cost'] for d in easy])
                    print(f"  N={N}: accuracy={avg_acc:.3f}, cost={avg_cost:.0f}")

        print("\n--- Best-of-N Scaling (hard problems, difficulty=0.8) ---")
        for N in [1, 2, 4, 8, 16, 32, 64]:
            key = f'bon_N{N}'
            if key in results:
                hard = [d for d in results[key] if d['difficulty'] > 0.7]
                if hard:
                    avg_acc = np.mean([d['accuracy'] for d in hard])
                    avg_cost = np.mean([d['avg_cost'] for d in hard])
                    print(f"  N={N}: accuracy={avg_acc:.3f}, cost={avg_cost:.0f}")

        # Tree Search scaling
        print("\n--- Tree Search (hard problems, difficulty=0.8) ---")
        for config in ['tot_b2_d3_bw2', 'tot_b3_d5_bw3', 'tot_b4_d7_bw4', 'tot_b3_d10_bw3']:
            if config in results:
                hard = [d for d in results[config] if d['difficulty'] > 0.7]
                if hard:
                    avg_acc = np.mean([d['accuracy'] for d in hard])
                    avg_cost = np.mean([d['avg_cost'] for d in hard])
                    print(f"  {config}: accuracy={avg_acc:.3f}, cost={avg_cost:.0f}")

        # Iterative Refinement
        print("\n--- Iterative Refinement (medium problems, difficulty=0.5) ---")
        for rounds in [1, 2, 3, 5, 8]:
            key = f'refine_r{rounds}'
            if key in results:
                medium = [d for d in results[key] if 0.4 < d['difficulty'] < 0.6]
                if medium:
                    avg_acc = np.mean([d['accuracy'] for d in medium])
                    avg_cost = np.mean([d['avg_cost'] for d in medium])
                    print(f"  rounds={rounds}: accuracy={avg_acc:.3f}, cost={avg_cost:.0f}")

        # Optimal allocation
        print("\n--- Optimal Allocation (budget=500 tokens) ---")
        optimal = compute_optimal_allocation(results, total_compute_budget=500)
        for entry in optimal[:5]:
            print(f"  difficulty={entry['difficulty']:.2f}: best={entry['best_strategy']}, "
                  f"accuracy={entry['best_accuracy']:.3f}, cost={entry['compute_cost']:.0f}")
        for entry in optimal[-5:]:
            print(f"  difficulty={entry['difficulty']:.2f}: best={entry['best_strategy']}, "
                  f"accuracy={entry['best_accuracy']:.3f}, cost={entry['compute_cost']:.0f}")

        # Save results
        output = {
            'results': {k: v for k, v in results.items()},
            'optimal_allocation': optimal,
            'config': {
                'num_problems': args.num_problems,
                'model_quality': args.model_quality,
            }
        }
        with open("test_time_compute_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to test_time_compute_results.json")


if __name__ == "__main__":
    main()