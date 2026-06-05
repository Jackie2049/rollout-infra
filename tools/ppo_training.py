#!/usr/bin/env python3
"""PPO Training from Scratch — Simplified RLHF Simulation
==========================================================
Demonstrates PPO (Proximal Policy Optimization) in an RLHF-like setting:
1. Policy (LLM) generates responses
2. Reward model scores them
3. PPO updates policy with clipping

Reference: Schulman et al., 2017 + InstructGPT (Ouyang et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


class PolicyModel(nn.Module):
    """Simple policy network (simulates LLM)."""
    def __init__(self, state_dim=16, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def get_action(self, x):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


class RewardModel(nn.Module):
    """Reward model (simulates trained RM)."""
    def __init__(self, state_dim=16, action_dim=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action_onehot):
        return self.net(torch.cat([state, action_onehot], dim=-1))


def create_env_data(n_states=100, state_dim=16, action_dim=4, device='cuda'):
    """Create a simple reward environment."""
    states = torch.randn(n_states, state_dim, device=device)

    # Ground truth reward: certain actions are better for certain states
    # R(s, a) = cos(s · w_a) where w_a is a weight vector per action
    torch.manual_seed(123)
    W = torch.randn(action_dim, state_dim, device=device) * 0.3
    # Add bonus for "safe" actions
    W[0] += 0.5  # action 0 has bonus

    rewards = torch.zeros(n_states, action_dim, device=device)
    for a in range(action_dim):
        rewards[:, a] = (states @ W[a]).cos().squeeze()

    return states, rewards, W


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    values = values.tolist()
    rewards = rewards.tolist()

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


def run_experiments(device='cuda'):
    print("=" * 70)
    print("PPO Training from Scratch — RLHF Simulation")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}
    state_dim, action_dim, hidden = 16, 4, 64
    n_states = 200

    states, gt_rewards, W = create_env_data(n_states, state_dim, action_dim, device)

    # ----------------------------------------------------------
    # Experiment 1: PPO Training Dynamics
    # ----------------------------------------------------------
    print("\n--- Experiment 1: PPO Training Dynamics ---")

    for clip_eps in [0.1, 0.2, 0.3]:
        torch.manual_seed(42)
        policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        n_epochs = 200
        rollout_size = 64
        ppo_epochs = 4
        minibatch_size = 32

        epoch_rewards = []
        epoch_losses = []
        epoch_kl = []

        for epoch in range(n_epochs):
            # Rollout
            idx = torch.randint(0, n_states, (rollout_size,))
            batch_states = states[idx]

            actions, log_probs, values = policy.get_action(batch_states)

            # Get rewards (simulate reward model)
            action_onehot = F.one_hot(actions, action_dim).float()
            with torch.no_grad():
                rewards = (batch_states @ W[0]).cos() * 0.5
                for a in range(action_dim):
                    mask = (actions == a).float()
                    rewards += mask * (batch_states @ W[a]).cos()

            epoch_rewards.append(rewards.mean().item())

            # Compute advantages
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            old_log_probs = log_probs.detach()

            for _ in range(ppo_epochs):
                new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, rewards)

                entropy_loss = -0.01 * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

            # Track KL divergence
            with torch.no_grad():
                new_log_probs_2, _, _ = policy.evaluate(batch_states, actions)
                kl = (torch.exp(old_log_probs) * (old_log_probs - new_log_probs_2)).mean()
                epoch_kl.append(kl.item())

            epoch_losses.append(loss.item())

        print(f"  ε={clip_eps}: reward {epoch_rewards[0]:.3f}→{epoch_rewards[-1]:.3f}, "
              f"loss {epoch_losses[-1]:.4f}, KL={epoch_kl[-1]:.4f}")

        results[f'ppo_eps{clip_eps}'] = {
            'final_reward': epoch_rewards[-1],
            'rewards': epoch_rewards[-10:],
            'final_kl': epoch_kl[-1],
        }

    # ----------------------------------------------------------
    # Experiment 2: PPO vs Vanilla PG
    # ----------------------------------------------------------
    print("\n--- Experiment 2: PPO vs Vanilla Policy Gradient ---")

    for algo in ['PPO', 'VanillaPG', 'PPO_no_clip']:
        torch.manual_seed(42)
        policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        rewards_history = []

        for epoch in range(200):
            idx = torch.randint(0, n_states, (64,))
            batch_states = states[idx]
            actions, log_probs, values = policy.get_action(batch_states)

            action_onehot = F.one_hot(actions, action_dim).float()
            with torch.no_grad():
                rewards = torch.zeros(64, device=device)
                for a in range(action_dim):
                    mask = (actions == a).float()
                    rewards += mask * (batch_states @ W[a]).cos()

            rewards_history.append(rewards.mean().item())
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if algo == 'PPO':
                old_log_probs = log_probs.detach()
                for _ in range(4):
                    new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                    loss = -torch.min(surr1, surr2).mean()
                    loss += 0.5 * F.mse_loss(new_values, rewards)
                    loss -= 0.01 * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            elif algo == 'PPO_no_clip':
                old_log_probs = log_probs.detach()
                for _ in range(4):
                    new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    loss = -(ratio * advantages).mean()
                    loss += 0.5 * F.mse_loss(new_values, rewards)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            else:  # VanillaPG
                loss = -(log_probs * advantages).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print(f"  {algo:15s}: reward {rewards_history[0]:.3f}→{rewards_history[-1]:.3f}, "
              f"max={max(rewards_history):.3f}")

        results[f'algo_{algo}'] = {
            'final_reward': rewards_history[-1],
            'max_reward': max(rewards_history),
            'history': rewards_history[-10:],
        }

    # ----------------------------------------------------------
    # Experiment 3: KL Penalty Effect (RLHF-specific)
    # ----------------------------------------------------------
    print("\n--- Experiment 3: KL Penalty (RLHF Simulation) ---")

    for beta_kl in [0.0, 0.01, 0.1, 0.5]:
        torch.manual_seed(42)
        policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        ref_policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        ref_policy.load_state_dict(policy.state_dict())  # reference = initial policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        rewards_history = []
        kl_history = []

        for epoch in range(200):
            idx = torch.randint(0, n_states, (64,))
            batch_states = states[idx]
            actions, log_probs, values = policy.get_action(batch_states)

            with torch.no_grad():
                rewards = torch.zeros(64, device=device)
                for a in range(action_dim):
                    mask = (actions == a).float()
                    rewards += mask * (batch_states @ W[a]).cos()

                # KL penalty per token (simulated)
                ref_log_probs, _, _ = ref_policy.evaluate(batch_states, actions)
                kl_per_token = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(-1)
                kl_penalty = beta_kl * kl_per_token
                effective_reward = rewards - kl_penalty

            rewards_history.append(rewards.mean().item())
            kl_history.append(kl_per_token.mean().item())

            advantages = effective_reward - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_log_probs = log_probs.detach()
            for _ in range(4):
                new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2).mean()
                loss += 0.5 * F.mse_loss(new_values, effective_reward)
                loss -= 0.01 * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(f"  β_KL={beta_kl:4.2f}: reward {rewards_history[0]:.3f}→{rewards_history[-1]:.3f}, "
              f"KL={kl_history[-1]:.4f}")

        results[f'kl_beta{beta_kl}'] = {
            'final_reward': rewards_history[-1],
            'final_kl': kl_history[-1],
        }

    # ----------------------------------------------------------
    # Experiment 4: PPO Epochs Effect
    # ----------------------------------------------------------
    print("\n--- Experiment 4: PPO Update Epochs ---")

    for n_ppo_epochs in [1, 2, 4, 8, 16]:
        torch.manual_seed(42)
        policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        rewards_history = []

        for epoch in range(200):
            idx = torch.randint(0, n_states, (64,))
            batch_states = states[idx]
            actions, log_probs, values = policy.get_action(batch_states)

            with torch.no_grad():
                rewards = torch.zeros(64, device=device)
                for a in range(action_dim):
                    mask = (actions == a).float()
                    rewards += mask * (batch_states @ W[a]).cos()

            rewards_history.append(rewards.mean().item())
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_log_probs = log_probs.detach()
            for _ in range(n_ppo_epochs):
                new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2).mean()
                loss += 0.5 * F.mse_loss(new_values, rewards)
                loss -= 0.01 * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(f"  epochs={n_ppo_epochs:2d}: reward {rewards_history[0]:.3f}→{rewards_history[-1]:.3f}")

        results[f'epochs_{n_ppo_epochs}'] = {
            'final_reward': rewards_history[-1],
        }

    # ----------------------------------------------------------
    # Experiment 5: Advantage Normalization
    # ----------------------------------------------------------
    print("\n--- Experiment 5: Advantage Normalization ---")

    for norm_name, do_norm in [('normalized', True), ('unnormalized', False)]:
        torch.manual_seed(42)
        policy = PolicyModel(state_dim, action_dim, hidden).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        rewards_history = []

        for epoch in range(200):
            idx = torch.randint(0, n_states, (64,))
            batch_states = states[idx]
            actions, log_probs, values = policy.get_action(batch_states)

            with torch.no_grad():
                rewards = torch.zeros(64, device=device)
                for a in range(action_dim):
                    mask = (actions == a).float()
                    rewards += mask * (batch_states @ W[a]).cos()

            rewards_history.append(rewards.mean().item())
            advantages = rewards - values.detach()
            if do_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_log_probs = log_probs.detach()
            for _ in range(4):
                new_log_probs, new_values, entropy = policy.evaluate(batch_states, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2).mean()
                loss += 0.5 * F.mse_loss(new_values, rewards)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(f"  {norm_name:15s}: reward {rewards_history[0]:.3f}→{rewards_history[-1]:.3f}")

        results[f'norm_{norm_name}'] = {'final_reward': rewards_history[-1]}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PPO Training Key Findings")
    print("=" * 70)
    print("""
1. PPO vs Vanilla PG: PPO is more stable due to clipping
2. Clip ratio ε=0.2: Standard choice, balances exploration/exploitation
3. KL penalty (RLHF): Prevents reward hacking, β=0.01-0.1 typical
4. Multiple PPO epochs: 4 is standard, more epochs risk overfitting
5. Advantage normalization: Critical for training stability
    """)

    with open('ppo_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to ppo_training_results.json")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_experiments(device=device)
