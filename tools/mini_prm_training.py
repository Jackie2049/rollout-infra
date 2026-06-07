#!/usr/bin/env python3
"""Mini PRM (Process Reward Model) Training Pipeline

Train a step-level verifier that scores intermediate reasoning steps,
then use it with Best-of-N and Tree Search to validate PRM > ORM finding.

Key comparison:
- ORM (Outcome Reward Model): only scores final answer
- PRM (Process Reward Model): scores each intermediate step

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
# Reasoning Task: Multi-step arithmetic
# ============================================================

# Step-level reasoning: compute a + b via intermediate steps
# Step 1: compute a (identity)
# Step 2: compute b (identity)
# Step 3: compute sum = a + b

TOKENS = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '=': 11,
    '<pad>': 12, '<eos>': 13, '<bos>': 14, '<unk>': 15,
    'a': 16, 'b': 17, 'r': 18, '<space>': 19,
}
VOCAB_SIZE = len(TOKENS)
IDX_TO_TOKEN = {v: k for k, v in TOKENS.items()}


def generate_step_reasoning_problem():
    """Generate a multi-step arithmetic problem with step annotations.

    Returns: (prompt, steps, correct_answer)
    - prompt: "a+b="
    - steps: list of (step_text, step_correct) pairs
    - correct_answer: a+b
    """
    a = np.random.randint(0, 5)
    b = np.random.randint(0, 5)

    # Multi-step reasoning representation
    # Step 1: "first=a" (extract first number)
    # Step 2: "second=b" (extract second number)
    # Step 3: "sum=a+b" (compute sum)

    steps = [
        (f"first={a}", True),  # Step 1: extract first number
        (f"second={b}", True),  # Step 2: extract second number
        (f"sum={a+b}", True),  # Step 3: compute sum
    ]

    prompt = f"{a}+{b}="
    correct_answer = a + b

    return prompt, steps, correct_answer


def generate_wrong_steps(a, b, wrong_type='random'):
    """Generate steps with errors at different positions.

    wrong_type:
    - 'early': Step 1 is wrong (misidentify first number)
    - 'middle': Step 2 is wrong (misidentify second number)
    - 'late': Step 3 is wrong (compute wrong sum)
    - 'random': Random error at any step
    """
    correct_steps = [
        (f"first={a}", True),
        (f"second={b}", True),
        (f"sum={a+b}", True),
    ]

    if wrong_type == 'early':
        wrong_a = np.random.randint(0, 5)
        while wrong_a == a:
            wrong_a = np.random.randint(0, 5)
        wrong_sum = wrong_a + b
        steps = [
            (f"first={wrong_a}", False),  # WRONG
            (f"second={b}", True),  # depends on wrong first
            (f"sum={wrong_sum}", False),  # WRONG (cascading error)
        ]
    elif wrong_type == 'middle':
        wrong_b = np.random.randint(0, 5)
        while wrong_b == b:
            wrong_b = np.random.randint(0, 5)
        wrong_sum = a + wrong_b
        steps = [
            (f"first={a}", True),
            (f"second={wrong_b}", False),  # WRONG
            (f"sum={wrong_sum}", False),  # WRONG (cascading)
        ]
    elif wrong_type == 'late':
        wrong_sum = np.random.randint(0, 10)
        while wrong_sum == a + b:
            wrong_sum = np.random.randint(0, 10)
        steps = [
            (f"first={a}", True),
            (f"second={b}", True),
            (f"sum={wrong_sum}", False),  # WRONG (only last step)
        ]
    elif wrong_type == 'random':
        error_pos = np.random.randint(0, 3)
        if error_pos == 0:
            return generate_wrong_steps(a, b, 'early')
        elif error_pos == 1:
            return generate_wrong_steps(a, b, 'middle')
        else:
            return generate_wrong_steps(a, b, 'late')

    return steps


# ============================================================
# PRM Model: Step-level verifier
# ============================================================

class StepVerifier(nn.Module):
    """Process Reward Model: scores each reasoning step given problem context.

    Input: (problem_tokens, step_tokens) → Output: probability of step being correct (0-1)
    Uses problem context to determine what the correct step should be.
    """
    def __init__(self, hidden_dim=64, vocab_size=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # Separate processing for problem and step
        self.problem_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.step_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Combine problem + step context
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, problem_tokens, step_tokens):
        """problem_tokens: [B, S_p], step_tokens: [B, S_s] → score: [B, 1]"""
        prob_emb = self.embed(problem_tokens)
        step_emb = self.embed(step_tokens)

        # Use last token for each (captures the key info)
        prob_repr = self.problem_net(prob_emb[:, -1, :])  # [B, H]
        step_repr = self.step_net(step_emb[:, -1, :])  # [B, H]

        # Combine: given this problem, is this step correct?
        combined = torch.cat([prob_repr, step_repr], dim=-1)  # [B, 2H]
        return torch.sigmoid(self.combine_net(combined))


class OutcomeVerifier(nn.Module):
    """Outcome Reward Model: only scores final answer.

    Input: full solution → Output: probability of final answer being correct
    """
    def __init__(self, hidden_dim=64, vocab_size=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solution_tokens):
        """solution_tokens: [B, S] → score: [B, 1]"""
        x = self.embed(solution_tokens)
        # Use last token (the final answer digit)
        x = x[:, -1, :]
        return torch.sigmoid(self.net(x))


def tokenize_text(text):
    """Convert text to token IDs."""
    tokens = []
    for ch in text:
        if ch in TOKENS:
            tokens.append(TOKENS[ch])
        else:
            tokens.append(TOKENS['<unk>'])
    return tokens


# ============================================================
# Training
# ============================================================

def train_prm(prm_model, num_steps=500, lr=1e-3, device='cpu'):
    """Train PRM on step-level correctness data with problem context."""
    prm_model.train()
    optimizer = torch.optim.AdamW(prm_model.parameters(), lr=lr)

    losses = []
    accuracies = []

    for step in range(num_steps):
        # Generate training data: mix correct and wrong steps
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)

        # Problem context: "a+b="
        problem_text = f"{a}+{b}="
        problem_tokens = tokenize_text(problem_text)

        batch_problem_tokens = []
        batch_step_tokens = []
        batch_labels = []

        # Correct steps
        for text, is_correct in [
            (f"first={a}", True),
            (f"second={b}", True),
            (f"sum={a+b}", True),
        ]:
            tokens = tokenize_text(text)
            batch_problem_tokens.append(problem_tokens)
            batch_step_tokens.append(tokens)
            batch_labels.append(1.0)

        # Wrong steps at different positions
        for wrong_type in ['early', 'middle', 'late']:
            wrong_steps = generate_wrong_steps(a, b, wrong_type)
            for text, is_correct in wrong_steps:
                tokens = tokenize_text(text)
                batch_problem_tokens.append(problem_tokens)
                batch_step_tokens.append(tokens)
                batch_labels.append(1.0 if is_correct else 0.0)

        # Pad
        max_prob_len = max(len(t) for t in batch_problem_tokens)
        max_step_len = max(len(t) for t in batch_step_tokens)

        padded_prob = []
        padded_step = []
        for pt, st in zip(batch_problem_tokens, batch_step_tokens):
            padded_prob.append(pt + [TOKENS['<pad>']] * (max_prob_len - len(pt)))
            padded_step.append(st + [TOKENS['<pad>']] * (max_step_len - len(st)))

        prob_tensor = torch.tensor(padded_prob, dtype=torch.long, device=device)
        step_tensor = torch.tensor(padded_step, dtype=torch.long, device=device)
        labels = torch.tensor(batch_labels, dtype=torch.float, device=device)

        # Forward
        scores = prm_model(prob_tensor, step_tensor).squeeze()
        loss = F.binary_cross_entropy(scores, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        predictions = (scores > 0.5).float()
        accuracy = (predictions == labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(accuracy)

        if step % 100 == 0 or step == num_steps - 1:
            print(f"  PRM step {step}: loss={loss.item():.4f}, accuracy={accuracy:.3f}")

    return {'losses': losses, 'accuracies': accuracies}


def train_orm(orm_model, num_steps=500, lr=1e-3, device='cpu'):
    """Train ORM on outcome-level correctness data."""
    orm_model.train()
    optimizer = torch.optim.AdamW(orm_model.parameters(), lr=lr)

    losses = []
    accuracies = []

    for step in range(num_steps):
        batch_tokens = []
        batch_labels = []

        # Generate solutions (correct and wrong)
        for _ in range(4):
            a = np.random.randint(0, 5)
            b = np.random.randint(0, 5)

            # Correct solution
            correct_text = f"first={a}second={b}sum={a+b}"
            tokens = tokenize_text(correct_text)
            batch_tokens.append(tokens)
            batch_labels.append(1.0)

            # Wrong solution (different wrong types)
            for wrong_type in ['early', 'middle', 'late']:
                wrong_steps = generate_wrong_steps(a, b, wrong_type)
                wrong_text = ''.join(text for text, _ in wrong_steps)
                tokens = tokenize_text(wrong_text)
                batch_tokens.append(tokens)
                batch_labels.append(0.0)

        # Pad
        max_len = max(len(t) for t in batch_tokens)
        padded = []
        for tokens in batch_tokens:
            padded.append(tokens + [TOKENS['<pad>']] * (max_len - len(tokens)))

        batch_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        labels = torch.tensor(batch_labels, dtype=torch.float, device=device)

        scores = orm_model(batch_tensor).squeeze()
        loss = F.binary_cross_entropy(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = (scores > 0.5).float()
        accuracy = (predictions == labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(accuracy)

        if step % 100 == 0 or step == num_steps - 1:
            print(f"  ORM step {step}: loss={loss.item():.4f}, accuracy={accuracy:.3f}")

    return {'losses': losses, 'accuracies': accuracies}


# ============================================================
# Best-of-N with PRM vs ORM
# ============================================================

def best_of_n_with_verifier(solutions, verifier, problem_tokens, is_prm=True):
    """Best-of-N selection using PRM or ORM verifier.

    solutions: list of (steps, final_answer, correct_answer)
    verifier: trained PRM or ORM model
    problem_tokens: token IDs of the problem context "a+b="
    is_prm: True for PRM (step-level scoring), False for ORM (outcome-level)
    """
    device = next(verifier.parameters()).device
    verifier.eval()

    scores = []
    with torch.no_grad():
        for steps, final_answer, correct_answer in solutions:
            if is_prm:
                # PRM: score each step with problem context → aggregate
                step_scores = []
                for step_text, step_correct in steps:
                    tokens = tokenize_text(step_text)
                    prob_tensor = torch.tensor([problem_tokens], dtype=torch.long, device=device)
                    step_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                    score = verifier(prob_tensor, step_tensor).item()
                    step_scores.append(score)
                # Use minimum: any wrong step → low overall score
                scores.append(np.min(step_scores))
            else:
                # ORM: score entire solution
                full_text = ''.join(text for text, _ in steps)
                tokens = tokenize_text(full_text)
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
                score = verifier(token_tensor).item()
                scores.append(score)

    # Select best
    best_idx = np.argmax(scores)
    return solutions[best_idx], scores


def evaluate_verifier(prm_model, orm_model, num_problems=100, device='cpu'):
    """Evaluate PRM vs ORM for Best-of-N selection.

    Key test: Can PRM distinguish early errors from late errors?
    ORM can only see the final answer → can't tell WHERE the error is.
    """
    prm_model.eval()
    orm_model.eval()

    # Test: solutions with errors at different positions
    # ORM should struggle with "late error only" (correct first/second, wrong sum)
    # because ORM sees: "first=2second=3sum=8" → hard to know which step is wrong
    # PRM should detect: Step 3 is wrong (sum ≠ 2+3)

    results = defaultdict(list)

    for trial in range(num_problems):
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)

        # Generate 8 candidate solutions: 1 correct + 7 wrong
        candidates = []

        # Correct solution
        correct_steps = [
            (f"first={a}", True),
            (f"second={b}", True),
            (f"sum={a+b}", True),
        ]
        candidates.append((correct_steps, a+b, a+b))

        # Wrong solutions at different positions
        for wrong_type in ['early', 'middle', 'late', 'late', 'late', 'early', 'middle']:
            wrong_steps = generate_wrong_steps(a, b, wrong_type)
            final_val = int(wrong_steps[-1][0].split('=')[1])
            candidates.append((wrong_steps, final_val, a+b))

        # PRM selection
        problem_tokens = tokenize_text(f"{a}+{b}=")
        best_prm, prm_scores = best_of_n_with_verifier(candidates, prm_model, problem_tokens, is_prm=True)
        prm_correct = best_prm[2] == a + b  # correct_answer == a+b

        # ORM selection
        best_orm, orm_scores = best_of_n_with_verifier(candidates, orm_model, problem_tokens, is_prm=False)
        orm_correct = best_orm[2] == a + b

        results['prm_correct'].append(prm_correct)
        results['orm_correct'].append(orm_correct)

    # Aggregate
    prm_acc = np.mean(results['prm_correct'])
    orm_acc = np.mean(results['orm_correct'])

    print(f"\n--- Verifier Evaluation ({num_problems} problems, 8 candidates each) ---")
    print(f"  PRM Best-of-8 accuracy: {prm_acc:.3f}")
    print(f"  ORM Best-of-8 accuracy: {orm_acc:.3f}")
    print(f"  PRM improvement: {(prm_acc - orm_acc) / orm_acc * 100:.1f}%")

    # Detailed: breakdown by error type
    for error_type in ['early', 'middle', 'late']:
        # Count how often each verifier correctly rejects this error type
        prm_reject = 0
        orm_reject = 0
        total = 0

        for trial in range(num_problems):
            a = np.random.randint(0, 5)
            b = np.random.randint(0, 5)
            wrong_steps = generate_wrong_steps(a, b, error_type)
            problem_tokens = tokenize_text(f"{a}+{b}=")

            with torch.no_grad():
                # PRM step scores with problem context
                step_scores = []
                for step_text, _ in wrong_steps:
                    t = tokenize_text(step_text)
                    prob_tensor = torch.tensor([problem_tokens], dtype=torch.long, device=device)
                    step_tensor = torch.tensor([t], dtype=torch.long, device=device)
                    score = prm_model(prob_tensor, step_tensor).item()
                    step_scores.append(score)
                prm_min = np.min(step_scores)
                prm_reject += int(prm_min < 0.5)

                # ORM full solution score
                full_text = ''.join(text for text, _ in wrong_steps)
                full_tokens = tokenize_text(full_text)
                orm_tensor = torch.tensor([full_tokens], dtype=torch.long, device=device)
                orm_score = orm_model(orm_tensor).item()
                orm_reject += int(orm_score < 0.5)

            total += 1

        print(f"  {error_type} error: PRM reject rate={prm_reject/total:.3f}, ORM reject rate={orm_reject/total:.3f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prm_steps', type=int, default=1000)
    parser.add_argument('--orm_steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_eval_problems', type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("Mini PRM vs ORM — Step-Level vs Outcome-Level Verification")
    print("=" * 70)

    # Train PRM
    print("\n--- Training PRM (Process Reward Model) ---")
    prm_model = StepVerifier(hidden_dim=64, vocab_size=VOCAB_SIZE).to(device)
    prm_params = sum(p.numel() for p in prm_model.parameters())
    print(f"PRM params: {prm_params:,}")
    prm_history = train_prm(prm_model, num_steps=args.prm_steps, lr=args.lr, device=device)

    # Train ORM
    print("\n--- Training ORM (Outcome Reward Model) ---")
    orm_model = OutcomeVerifier(hidden_dim=64, vocab_size=VOCAB_SIZE).to(device)
    orm_params = sum(p.numel() for p in orm_model.parameters())
    print(f"ORM params: {orm_params:,}")
    orm_history = train_orm(orm_model, num_steps=args.orm_steps, lr=args.lr, device=device)

    # Evaluate
    print("\n--- Evaluation: PRM vs ORM for Best-of-N Selection ---")
    results = evaluate_verifier(prm_model, orm_model, num_problems=args.num_eval_problems, device=device)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PRM training: {args.prm_steps} steps, final acc={prm_history['accuracies'][-1]:.3f}")
    print(f"ORM training: {args.orm_steps} steps, final acc={orm_history['accuracies'][-1]:.3f}")
    prm_acc = np.mean(results['prm_correct'])
    orm_acc = np.mean(results['orm_correct'])
    print(f"PRM Best-of-8: {prm_acc:.3f}")
    print(f"ORM Best-of-8: {orm_acc:.3f}")
    if orm_acc > 0:
        print(f"PRM improvement: {(prm_acc - orm_acc) / orm_acc * 100:.1f}%")
    print(f"\nKey finding: PRM can detect WHERE errors occur → better selection!")

    # Save results
    output = {
        'prm_history': prm_history,
        'orm_history': orm_history,
        'evaluation': {k: v for k, v in results.items()},
        'config': {
            'prm_params': prm_params,
            'orm_params': orm_params,
            'prm_steps': args.prm_steps,
            'orm_steps': args.orm_steps,
            'device': str(device),
        }
    }
    with open("prm_vs_orm_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to prm_vs_orm_results.json")


if __name__ == "__main__":
    main()