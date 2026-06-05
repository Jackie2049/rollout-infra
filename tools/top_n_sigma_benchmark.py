#!/usr/bin/env python3
"""Top-nσ Sampling Benchmark — Verify PR #7 Value
==================================================
Compares Top-nσ against top-p, top-k, min-p, and greedy sampling.

Paper: "Top-nσ: Not All Logits Are You Need" (Tang et al., ACL 2025)
arXiv: 2411.07641

Key claim: Logits separate into Gaussian noise region + informative region.
Top-nσ: threshold = max(logits) - n * std(logits), set below to -inf.

Experiments:
1. Logit distribution analysis — verify Gaussian noise + informative split
2. Token filtering comparison — how many tokens survive each method
3. Temperature stability — does each method maintain stable sampling space?
4. Perplexity impact — does filtering hurt language modeling quality?
5. Reasoning task simulation — does Top-nσ really beat greedy?
"""

import torch
import torch.nn.functional as F
import math
import json
import time
import os

# ============================================================
# Sampling Methods
# ============================================================

def top_n_sigma_filter(logits, n=2.0):
    """Top-nσ: filter logits using statistical threshold.

    threshold = max(logits) - n * std(logits)
    Set logits below threshold to -inf.
    """
    threshold = logits.max(dim=-1, keepdim=True).values - n * logits.std(dim=-1, keepdim=True)
    return logits.masked_fill(logits < threshold, float('-inf'))


def top_k_filter(logits, k=50):
    """Top-K: keep only top-K logits."""
    val, idx = logits.topk(k, dim=-1)
    threshold = val[..., -1:].expand_as(logits)
    return logits.masked_fill(logits < threshold, float('-inf'))


def top_p_filter(logits, p=0.9):
    """Top-p (nucleus): keep smallest set with cumulative prob >= p."""
    sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # Remove tokens with cumulative prob above threshold
    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_logits[sorted_mask] = float('-inf')

    # Scatter back
    mask = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    return logits.masked_fill(mask == float('-inf'), float('-inf'))


def min_p_filter(logits, p=0.05):
    """Min-p: keep tokens with prob >= p * max_prob."""
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < p * max_prob, float('-inf'))


def sample_from_logits(logits, temperature=1.0, method='greedy', **kwargs):
    """Unified sampling interface."""
    logits = logits / temperature

    if method == 'greedy':
        return logits.argmax(dim=-1)

    if method == 'top_n_sigma':
        logits = top_n_sigma_filter(logits, n=kwargs.get('n', 2.0))
    elif method == 'top_k':
        logits = top_k_filter(logits, k=kwargs.get('k', 50))
    elif method == 'top_p':
        logits = top_p_filter(logits, p=kwargs.get('p', 0.9))
    elif method == 'min_p':
        logits = min_p_filter(logits, p=kwargs.get('min_p', 0.05))

    probs = F.softmax(logits, dim=-1)
    # Handle all -inf case
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return torch.multinomial(probs, 1).squeeze(-1)


# ============================================================
# Mini LLM for testing
# ============================================================

class MiniLLM(torch.nn.Module):
    """Small GPT-like model for testing sampling methods."""
    def __init__(self, vocab_size=1000, d_model=256, n_heads=4, n_layers=4, max_seq=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embedding = torch.nn.Embedding(max_seq, d_model)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.head(h)


# ============================================================
# Experiments
# ============================================================

def experiment1_logit_distribution(device='cuda'):
    """Exp1: Analyze logit distribution — verify Gaussian noise + informative split."""
    print("\n" + "="*70)
    print("Experiment 1: Logit Distribution Analysis")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 32000  # LLM-like vocab
    results = {}

    # Simulate logits from different distribution shapes
    scenarios = {
        'sharp': ('One dominant token', lambda: torch.randn(1, vocab_size, device=device) * 0.5),
        'medium': ('A few strong candidates', lambda: torch.randn(1, vocab_size, device=device) * 2.0),
        'flat': ('Many candidates', lambda: torch.randn(1, vocab_size, device=device) * 5.0),
        'bimodal': ('Two peaks', lambda: _bimodal_logits(1, vocab_size, device)),
    }

    for name, (desc, gen_fn) in scenarios.items():
        logits = gen_fn()
        sorted_logits, _ = logits.sort(descending=True, dim=-1)

        # Top-nσ threshold
        max_logit = logits.max().item()
        std_logit = logits.std().item()
        threshold_2sigma = max_logit - 2.0 * std_logit

        # Count surviving tokens
        n_survive = (logits > threshold_2sigma).sum().item()

        # Fit Gaussian to tail (logits below top-10)
        tail = sorted_logits[0, 10:].cpu()
        tail_mean = tail.mean().item()
        tail_std = tail.std().item()

        # Shapiro-Wilk-like test: check if tail looks Gaussian
        # Use simple kurtosis test
        tail_kurtosis = ((tail - tail.mean())**4).mean() / (tail.std()**4 + 1e-10).item()

        print(f"\n  [{name}] {desc}")
        print(f"    max={max_logit:.3f}, std={std_logit:.3f}, threshold={threshold_2sigma:.3f}")
        print(f"    Surviving tokens: {n_survive}/{vocab_size} ({100*n_survive/vocab_size:.2f}%)")
        print(f"    Tail (after top-10): mean={tail_mean:.3f}, std={tail_std:.3f}, kurtosis={tail_kurtosis:.2f}")
        print(f"    Gaussian kurtosis = 3.0, measured = {tail_kurtosis:.2f}")

        results[name] = {
            'max': max_logit, 'std': std_logit, 'threshold': threshold_2sigma,
            'n_survive': n_survive, 'pct_survive': 100*n_survive/vocab_size,
            'tail_kurtosis': tail_kurtosis,
            'is_gaussian_tail': abs(tail_kurtosis - 3.0) < 1.0,
        }

    print(f"\n  Paper claim verified: tail kurtosis ≈ 3.0 (Gaussian)?")
    for name, r in results.items():
        status = "YES" if r['is_gaussian_tail'] else "NO"
        print(f"    {name}: kurtosis={r['tail_kurtosis']:.2f} → {status}")

    return results


def _bimodal_logits(B, V, device):
    """Generate bimodal logit distribution."""
    logits = torch.randn(B, V, device=device) * 1.0
    logits[0, :5] += 8.0  # First peak
    logits[0, 100:105] += 5.0  # Second peak
    return logits


def experiment2_token_filtering(device='cuda'):
    """Exp2: Compare how many tokens survive each filtering method."""
    print("\n" + "="*70)
    print("Experiment 2: Token Filtering Comparison")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 32000
    results = []

    for sharpness in [0.5, 1.0, 2.0, 5.0]:
        logits = torch.randn(1, vocab_size, device=device) * sharpness

        # Temperature scaling
        for temp in [0.5, 1.0, 2.0, 5.0]:
            scaled = logits / temp

            methods = {
                'top_n_sigma_n1.0': top_n_sigma_filter(scaled.clone(), n=1.0),
                'top_n_sigma_n2.0': top_n_sigma_filter(scaled.clone(), n=2.0),
                'top_n_sigma_n3.0': top_n_sigma_filter(scaled.clone(), n=3.0),
                'top_k_50': top_k_filter(scaled.clone(), k=50),
                'top_p_0.9': top_p_filter(scaled.clone(), p=0.9),
                'min_p_0.05': min_p_filter(scaled.clone(), p=0.05),
            }

            row = {'sharpness': sharpness, 'temperature': temp}
            for method_name, filtered in methods.items():
                n_survive = (filtered > float('-inf')).sum().item()
                row[f'{method_name}_survive'] = n_survive
                row[f'{method_name}_pct'] = 100 * n_survive / vocab_size

            results.append(row)

            if sharpness == 2.0 and temp == 1.0:
                print(f"\n  [sharpness={sharpness}, temp={temp}] (typical LLM scenario)")
                for m in ['top_n_sigma_n2.0', 'top_k_50', 'top_p_0.9', 'min_p_0.05']:
                    print(f"    {m}: {row[f'{m}_survive']} tokens ({row[f'{m}_pct']:.2f}%)")

    # Print temperature stability table
    print(f"\n  Temperature Stability (sharpness=2.0):")
    print(f"  {'Temp':>5} | {'nσ(2.0)':>8} | {'top-k':>8} | {'top-p':>8} | {'min-p':>8}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in results:
        if r['sharpness'] == 2.0:
            print(f"  {r['temperature']:5.1f} | {r['top_n_sigma_n2.0_pct']:7.2f}% | "
                  f"{r['top_k_50_pct']:7.2f}% | {r['top_p_0.9_pct']:7.2f}% | "
                  f"{r['min_p_0.05_pct']:7.2f}%")

    return results


def experiment3_temperature_stability(device='cuda'):
    """Exp3: Does Top-nσ maintain stable sampling space across temperatures?"""
    print("\n" + "="*70)
    print("Experiment 3: Temperature Stability Analysis")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 32000
    results = []

    # Generate "realistic" logits — a few dominant tokens + noise
    for trial in range(5):
        logits = torch.randn(1, vocab_size, device=device) * 2.0
        # Make a few tokens dominant
        top_indices = torch.randint(0, vocab_size, (10,))
        for idx in top_indices:
            logits[0, idx] += 5.0

        for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            scaled = logits / temp

            # Top-nσ
            filt_ns = top_n_sigma_filter(scaled.clone(), n=2.0)
            ns_survive = (filt_ns > float('-inf')).sum().item()
            ns_probs = F.softmax(filt_ns, dim=-1)
            ns_entropy = -(ns_probs * (ns_probs + 1e-10).log()).sum(dim=-1).item()

            # Top-p
            filt_tp = top_p_filter(scaled.clone(), p=0.9)
            tp_survive = (filt_tp > float('-inf')).sum().item()
            tp_probs = F.softmax(filt_tp, dim=-1)
            tp_entropy = -(tp_probs * (tp_probs + 1e-10).log()).sum(dim=-1).item()

            # Min-p
            filt_mp = min_p_filter(scaled.clone(), p=0.05)
            mp_survive = (filt_mp > float('-inf')).sum().item()
            mp_probs = F.softmax(filt_mp, dim=-1)
            mp_entropy = -(mp_probs * (mp_probs + 1e-10).log()).sum(dim=-1).item()

            results.append({
                'trial': trial, 'temperature': temp,
                'ns_survive': ns_survive, 'ns_entropy': ns_entropy,
                'tp_survive': tp_survive, 'tp_entropy': tp_entropy,
                'mp_survive': mp_survive, 'mp_entropy': mp_entropy,
            })

    # Print summary: std of surviving token count across temperatures
    print(f"\n  Surviving token count variation (across T=0.1..10.0):")
    for method in ['ns', 'tp', 'mp']:
        counts = {}
        for r in results:
            t = r['temperature']
            if t not in counts:
                counts[t] = []
            counts[t].append(r[f'{method}_survive'])

        avg_counts = {t: sum(v)/len(v) for t, v in counts.items()}
        all_avgs = list(avg_counts.values())
        variation = max(all_avgs) / max(min(all_avgs), 1)

        name = {'ns': 'Top-nσ', 'tp': 'Top-p', 'mp': 'Min-p'}[method]
        print(f"    {name}: min={min(all_avgs):.0f}, max={max(all_avgs):.0f}, "
              f"ratio={variation:.2f}x")

    # Detailed table
    print(f"\n  Average surviving tokens across temperatures:")
    print(f"  {'Temp':>5} | {'Top-nσ':>8} | {'Top-p':>8} | {'Min-p':>8}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ns_avg = sum(r['ns_survive'] for r in results if r['temperature'] == temp) / 5
        tp_avg = sum(r['tp_survive'] for r in results if r['temperature'] == temp) / 5
        mp_avg = sum(r['mp_survive'] for r in results if r['temperature'] == temp) / 5
        print(f"  {temp:5.1f} | {ns_avg:8.0f} | {tp_avg:8.0f} | {mp_avg:8.0f}")

    return results


def experiment4_perplexity_impact(device='cuda'):
    """Exp4: Does Top-nσ filtering hurt perplexity?"""
    print("\n" + "="*70)
    print("Experiment 4: Perplexity Impact on Mini LLM")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 500
    seq_len = 64
    batch_size = 8

    # Create and train a small model
    model = MiniLLM(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq=256).to(device)
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    x, y = data[:, :-1], data[:, 1:]

    # Quick training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(200):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

    # Get logits
    with torch.no_grad():
        logits = model(x)  # (B, T, V)

    results = {}

    # Baseline: full softmax perplexity
    full_probs = F.softmax(logits.reshape(-1, vocab_size), dim=-1)
    full_nll = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    full_ppl = math.exp(full_nll.item())
    results['no_filter'] = {'nll': full_nll.item(), 'ppl': full_ppl}
    print(f"\n  Baseline (no filter): NLL={full_nll.item():.4f}, PPL={full_ppl:.2f}")

    # Test each method at different settings
    methods = {
        'top_n_sigma': [(1.0, 'n=1.0'), (2.0, 'n=2.0'), (3.0, 'n=3.0'), (5.0, 'n=5.0')],
        'top_k': [(10, 'k=10'), (50, 'k=50'), (100, 'k=100'), (200, 'k=200')],
        'top_p': [(0.5, 'p=0.5'), (0.9, 'p=0.9'), (0.95, 'p=0.95'), (0.99, 'p=0.99')],
        'min_p': [(0.01, 'mp=0.01'), (0.05, 'mp=0.05'), (0.1, 'mp=0.1'), (0.2, 'mp=0.2')],
    }

    for method_name, configs in methods.items():
        for val, label in configs:
            if method_name == 'top_n_sigma':
                filtered = top_n_sigma_filter(logits.clone(), n=val)
            elif method_name == 'top_k':
                filtered = top_k_filter(logits.clone(), k=val)
            elif method_name == 'top_p':
                filtered = top_p_filter(logits.clone(), p=val)
            elif method_name == 'min_p':
                filtered = min_p_filter(logits.clone(), p=val)

            # Compute NLL with filtered logits
            filt_nll = F.cross_entropy(filtered.reshape(-1, vocab_size), y.reshape(-1))
            # Handle case where true token was filtered out
            true_token_logits = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            filt_token_logits = filtered.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            n_filtered_out = (filt_token_logits == float('-inf')).sum().item()

            results[f'{method_name}_{label}'] = {
                'nll': filt_nll.item(),
                'n_filtered_out': n_filtered_out,
                'total_tokens': batch_size * seq_len,
                'filter_rate': 100 * n_filtered_out / (batch_size * seq_len),
            }
            print(f"  {method_name:12s} {label:8s}: NLL={filt_nll.item():.4f}, "
                  f"filtered_out={n_filtered_out}/{batch_size*seq_len} "
                  f"({100*n_filtered_out/(batch_size*seq_len):.1f}%)")

    # Key analysis: at similar filtering rates, which method has lower NLL?
    print(f"\n  Comparison at similar filtering rates:")
    print(f"  Top-nσ n=2.0: filter_rate={results['top_n_sigma_n=2.0']['filter_rate']:.1f}%")
    print(f"  Top-k k=50:   filter_rate={results['top_k_k=50']['filter_rate']:.1f}%")
    print(f"  Top-p p=0.9:  filter_rate={results['top_p_p=0.9']['filter_rate']:.1f}%")
    print(f"  Min-p mp=0.05: filter_rate={results['min_p_mp=0.05']['filter_rate']:.1f}%")

    return results


def experiment5_reasoning_simulation(device='cuda'):
    """Exp5: Simulate reasoning task — does Top-nσ beat greedy?"""
    print("\n" + "="*70)
    print("Experiment 5: Reasoning Task Simulation")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 100  # Small vocab for tractability
    n_questions = 100
    seq_len = 16

    # Create a "reasoning" model: each question has a correct answer
    # Model learns pattern: input tokens → output = specific token
    model = MiniLLM(vocab_size=vocab_size, d_model=64, n_heads=2, n_layers=2, max_seq=32).to(device)

    # Generate data with clear patterns
    questions = torch.randint(0, vocab_size//2, (n_questions, seq_len), device=device)
    # "Correct answer" is deterministic function of input
    correct_answers = (questions.sum(dim=1)) % vocab_size

    # Train
    x = questions
    y = correct_answers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(500):
        logits = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

    # Test different sampling methods
    with torch.no_grad():
        logits = model(x)[:, -1, :]  # Last position logits (B, V)

    results = {}
    methods = {
        'greedy': {},
        'top_n_sigma_n1': {'method': 'top_n_sigma', 'n': 1.0},
        'top_n_sigma_n2': {'method': 'top_n_sigma', 'n': 2.0},
        'top_n_sigma_n3': {'method': 'top_n_sigma', 'n': 3.0},
        'top_k_10': {'method': 'top_k', 'k': 10},
        'top_k_50': {'method': 'top_k', 'k': 50},
        'top_p_09': {'method': 'top_p', 'p': 0.9},
        'min_p_005': {'method': 'min_p', 'min_p': 0.05},
    }

    n_trials = 50  # Multiple trials for stochastic methods
    for name, kwargs in methods.items():
        if name == 'greedy':
            preds = logits.argmax(dim=-1)
            accuracy = (preds == y).float().mean().item()
            results[name] = {'accuracy': accuracy, 'std': 0.0}
            print(f"  {name:18s}: accuracy={accuracy:.4f}")
        else:
            method = kwargs.pop('method')
            accs = []
            for _ in range(n_trials):
                preds = sample_from_logits(logits.clone(), temperature=0.6, method=method, **kwargs)
                accs.append((preds == y).float().mean().item())
            avg_acc = sum(accs) / len(accs)
            std_acc = (sum((a - avg_acc)**2 for a in accs) / len(accs))**0.5
            results[name] = {'accuracy': avg_acc, 'std': std_acc}
            print(f"  {name:18s}: accuracy={avg_acc:.4f} ± {std_acc:.4f} (T=0.6, {n_trials} trials)")

    # Test at different temperatures
    print(f"\n  Temperature sweep (Top-nσ n=2.0 vs Top-p p=0.9 vs Min-p 0.05):")
    print(f"  {'Temp':>5} | {'Greedy':>7} | {'nσ(2)':>7} | {'top-p':>7} | {'min-p':>7}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    greedy_acc = results['greedy']['accuracy']
    for temp in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        accs = {}
        for m_name, m_kwargs in [('nσ(2)', {'method': 'top_n_sigma', 'n': 2.0}),
                                  ('top-p', {'method': 'top_p', 'p': 0.9}),
                                  ('min-p', {'method': 'min_p', 'min_p': 0.05})]:
            method = m_kwargs.pop('method')
            trial_accs = []
            for _ in range(20):
                preds = sample_from_logits(logits.clone(), temperature=temp, method=method, **m_kwargs)
                trial_accs.append((preds == y).float().mean().item())
            accs[m_name] = sum(trial_accs) / len(trial_accs)
            # restore kwargs
            if method == 'top_n_sigma': m_kwargs['method'] = method
            elif method == 'top_p': m_kwargs['method'] = method
            elif method == 'min_p': m_kwargs['method'] = method

        print(f"  {temp:5.1f} | {greedy_acc:7.4f} | {accs['nσ(2)']:7.4f} | "
              f"{accs['top-p']:7.4f} | {accs['min-p']:7.4f}")

    return results


def experiment6_n_sweep(device='cuda'):
    """Exp6: Sweep n parameter — what's the optimal n?"""
    print("\n" + "="*70)
    print("Experiment 6: n Parameter Sweep")
    print("="*70)

    torch.manual_seed(42)
    vocab_size = 32000
    results = []

    for trial in range(3):
        logits = torch.randn(1, vocab_size, device=device) * 2.0
        # Add a few dominant tokens
        logits[0, :5] += 5.0

        sorted_logits, _ = logits.sort(descending=True, dim=-1)

        for n in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            filtered = top_n_sigma_filter(logits.clone(), n=n)
            n_survive = (filtered > float('-inf')).sum().item()

            # Entropy of surviving distribution
            probs = F.softmax(filtered, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()

            results.append({
                'trial': trial, 'n': n,
                'n_survive': n_survive,
                'pct_survive': 100 * n_survive / vocab_size,
                'entropy': entropy,
            })

    # Print summary
    print(f"\n  {'n':>5} | {'Avg survive':>12} | {'Avg %':>8} | {'Avg entropy':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}")
    for n in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        subset = [r for r in results if r['n'] == n]
        avg_survive = sum(r['n_survive'] for r in subset) / len(subset)
        avg_pct = sum(r['pct_survive'] for r in subset) / len(subset)
        avg_entropy = sum(r['entropy'] for r in subset) / len(subset)
        print(f"  {n:5.1f} | {avg_survive:12.0f} | {avg_pct:7.2f}% | {avg_entropy:12.4f}")

    return results


# ============================================================
# Main
# ============================================================

def run_all_experiments(device='cuda'):
    print("="*70)
    print("Top-nσ Sampling Benchmark")
    print(f"Paper: 'Top-nσ: Not All Logits Are You Need' (Tang et al., ACL 2025)")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)

    all_results = {}

    all_results['exp1_distribution'] = experiment1_logit_distribution(device)
    all_results['exp2_filtering'] = experiment2_token_filtering(device)
    all_results['exp3_stability'] = experiment3_temperature_stability(device)
    all_results['exp4_perplexity'] = experiment4_perplexity_impact(device)
    all_results['exp5_reasoning'] = experiment5_reasoning_simulation(device)
    all_results['exp6_n_sweep'] = experiment6_n_sweep(device)

    # ============================================================
    # Final Verdict
    # ============================================================
    print("\n" + "="*70)
    print("FINAL VERDICT: PR #7 Value Assessment")
    print("="*70)

    print("""
    Paper Claims vs Our Findings:

    1. "Logits separate into Gaussian noise + informative region"
       → Exp1: [SEE RESULTS] — tail kurtosis analysis

    2. "Top-nσ maintains stable sampling space across temperatures"
       → Exp3: [SEE RESULTS] — surviving token variation

    3. "Top-nσ outperforms greedy on reasoning tasks"
       → Exp5: [SEE RESULTS] — accuracy comparison

    4. "Top-nσ is simpler than top-p/min-p"
       → Implementation: YES, 3 lines of code
       → No sorting needed (vs top-p), no cumulative sum

    PR #7 Value Assessment:
    - Code quality: [SEE RESULTS]
    - Novelty: Moderate (ACL 2025 paper, but simple implementation)
    - vLLM integration: LogitsProcessor is the right abstraction
    - Practical impact: [SEE RESULTS]
    """)

    # Save results
    # Convert tensors to floats for JSON
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open('top_n_sigma_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print("Results saved to top_n_sigma_results.json")

    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_all_experiments(device=device)
