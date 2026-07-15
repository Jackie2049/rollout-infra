"""
Test UP-GRPO policy loss on H20-3e GPU.

This script validates the UP-GRPO implementation by:
1. Comparing UP-GRPO loss vs vanilla PPO-clip loss with same inputs
2. Verifying: UP-GRPO produces LOWER loss for positive advantages (unbounded)
3. Verifying: UP-GRPO produces SAME loss for negative advantages (dual-clipped)
4. Testing with GRPO-style group-normalized advantages

Based on arXiv:2607.06987 and Jackie2049/verl PR #9.
"""

import torch
import math

def compute_up_grpo_loss(
    old_log_prob, log_prob, advantages, response_mask,
    clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None,
    clip_ratio_c=3.0, loss_agg_mode="token-mean",
):
    """UP-GRPO loss: unbounded positive, dual-clipped negative."""
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # Positive advantages: NO upper clip (unbounded positive)
    pg_losses_positive = -advantages * ratio

    # Negative advantages: SAME as vanilla dual-clip
    # max(unclipped, clipped) → conservative (takes worse option)
    # then min with clip_ratio_c → caps the penalty
    pg_losses_clipped = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses_dual_clipped = -advantages * clip_ratio_c

    clip_pg_losses1 = torch.maximum(pg_losses_positive, pg_losses_clipped)
    clip_pg_losses_negative = torch.min(pg_losses_dual_clipped, clip_pg_losses1)

    # Combine: positive → unbounded, negative → dual-clipped
    pg_losses = torch.where(advantages >= 0, pg_losses_positive, clip_pg_losses_negative)

    # Aggregate: token-mean
    loss = (pg_losses * response_mask).sum() / response_mask.sum().clamp(min=1)

    return loss, ratio

def compute_vanilla_loss(
    old_log_prob, log_prob, advantages, response_mask,
    clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None,
    clip_ratio_c=3.0, loss_agg_mode="token-mean",
):
    """Vanilla PPO-clip loss: dual-clip both directions."""
    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    loss = (pg_losses * response_mask).sum() / response_mask.sum().clamp(min=1)

    return loss, ratio

def test_up_grpo_vs_vanilla():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}, sm_{compute_cap[0]}{compute_cap[1]}")

    # ============================================================
    # Test 1: Positive advantages — UP-GRPO should have LOWER loss
    # ============================================================
    print("\n" + "="*60)
    print("Test 1: Positive advantages (A > 0)")
    print("="*60)

    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16
    old_log_prob = torch.randn(batch_size, seq_len, device=device) - 1.0
    log_prob = torch.randn(batch_size, seq_len, device=device) - 1.0
    advantages = torch.abs(torch.randn(batch_size, seq_len, device=device)) + 0.5  # All positive
    response_mask = torch.ones(batch_size, seq_len, device=device)

    up_loss, up_ratio = compute_up_grpo_loss(old_log_prob, log_prob, advantages, response_mask)
    van_loss, van_ratio = compute_vanilla_loss(old_log_prob, log_prob, advantages, response_mask)

    print(f"  UP-GRPO loss:    {up_loss.item():.6f}")
    print(f"  Vanilla loss:    {van_loss.item():.6f}")
    print(f"  Loss difference: {(van_loss.item() - up_loss.item()):.6f}")

    # UP-GRPO should produce lower (more negative) loss for positive A
    # because it doesn't clip the ratio upward
    test1_pass = up_loss.item() <= van_loss.item()
    print(f"  Test 1 PASS: {test1_pass} (UP-GRPO loss <= Vanilla loss for A>0)")

    # ============================================================
    # Test 2: Negative advantages — UP-GRPO should have SAME loss
    # ============================================================
    print("\n" + "="*60)
    print("Test 2: Negative advantages (A < 0)")
    print("="*60)

    torch.manual_seed(42)
    advantages_neg = -(torch.abs(torch.randn(batch_size, seq_len, device=device)) + 0.5)  # All negative

    up_loss_neg, _ = compute_up_grpo_loss(old_log_prob, log_prob, advantages_neg, response_mask)
    van_loss_neg, _ = compute_vanilla_loss(old_log_prob, log_prob, advantages_neg, response_mask)

    print(f"  UP-GRPO loss:    {up_loss_neg.item():.6f}")
    print(f"  Vanilla loss:    {van_loss_neg.item():.6f}")
    print(f"  Loss difference: {abs(van_loss_neg.item() - up_loss_neg.item()):.6f}")

    # For negative advantages, UP-GRPO uses dual-clip which should match vanilla
    # But note: UP-GRPO treats A=0 as positive (unbounded), vanilla treats A=0 as negative (clipped)
    # So we need strict negative advantages to get exact match
    loss_diff = abs(up_loss_neg.item() - van_loss_neg.item())
    test2_pass = loss_diff < 0.01  # Small tolerance due to numerical precision
    print(f"  Test 2 PASS: {test2_pass} (UP-GRPO ≈ Vanilla for A<0, diff={loss_diff:.6f})")

    # ============================================================
    # Test 3: Mixed advantages (GRPO-style)
    # ============================================================
    print("\n" + "="*60)
    print("Test 3: Mixed advantages (GRPO-style group-normalized)")
    print("="*60)

    torch.manual_seed(42)
    # Simulate GRPO advantages: group-normalized scores
    group_size = 8
    num_groups = 4
    total_samples = group_size * num_groups

    # Generate scores per group, then normalize
    scores = torch.randn(total_samples, device=device)
    advantages_grpo = torch.zeros_like(scores)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_scores = scores[start:end]
        group_mean = group_scores.mean()
        group_std = group_scores.std().clamp(min=1e-8)
        advantages_grpo[start:end] = (group_scores - group_mean) / group_std

    # Broadcast to sequence-level
    advantages_seq = advantages_grpo.unsqueeze(-1).expand(total_samples, seq_len)
    old_lp = torch.randn(total_samples, seq_len, device=device) - 1.0
    new_lp = torch.randn(total_samples, seq_len, device=device) - 1.0
    mask = torch.ones(total_samples, seq_len, device=device)

    up_loss_mix, _ = compute_up_grpo_loss(old_lp, new_lp, advantages_seq, mask)
    van_loss_mix, _ = compute_vanilla_loss(old_lp, new_lp, advantages_seq, mask)

    # Count positive vs negative tokens
    pos_frac = (advantages_seq > 0).float().mean().item()
    neg_frac = (advantages_seq < 0).float().mean().item()

    print(f"  Positive advantage fraction: {pos_frac:.2%}")
    print(f"  Negative advantage fraction: {neg_frac:.2%}")
    print(f"  UP-GRPO loss:    {up_loss_mix.item():.6f}")
    print(f"  Vanilla loss:    {van_loss_mix.item():.6f}")
    print(f"  Loss difference: {(van_loss_mix.item() - up_loss_mix.item()):.6f}")

    # UP-GRPO should produce lower loss when there are positive advantages
    test3_pass = up_loss_mix.item() <= van_loss_mix.item()
    print(f"  Test 3 PASS: {test3_pass} (UP-GRPO lower with mixed advantages)")

    # ============================================================
    # Test 4: Large ratio — verify unbounded positive
    # ============================================================
    print("\n" + "="*60)
    print("Test 4: Large ratio (ratio=10.0) — unbounded positive")
    print("="*60)

    # When ratio is very large (policy strongly improved), vanilla clips it
    # to 1+clip_ratio=1.2, but UP-GRPO does NOT clip for positive A
    old_lp4 = torch.zeros(1, 1, device=device)  # log_prob_old = 0
    new_lp4 = torch.tensor([[math.log(10.0)]], device=device)  # ratio = 10.0
    adv4 = torch.tensor([[1.0]], device=device)  # positive advantage
    mask4 = torch.ones(1, 1, device=device)

    up4, r4 = compute_up_grpo_loss(old_lp4, new_lp4, adv4, mask4)
    van4, _ = compute_vanilla_loss(old_lp4, new_lp4, adv4, mask4)

    print(f"  Ratio: 10.0 (policy 10x improved)")
    print(f"  UP-GRPO loss:    {up4.item():.6f} (= -1.0 * 10.0 = -10.0)")
    print(f"  Vanilla loss:    {van4.item():.6f} (= max(-1.0*10.0, -1.0*1.2) = -1.2)")

    # UP-GRPO: -A * ratio = -1.0 * 10.0 = -10.0 (unbounded)
    # Vanilla: max(-A*ratio, -A*clip(ratio)) = max(-10.0, -1.2) = -1.2 (clipped)
    test4_pass = up4.item() < van4.item() - 5.0  # UP-GRPO much lower
    print(f"  Test 4 PASS: {test4_pass} (UP-GRPO much lower for large ratio + positive A)")

    # ============================================================
    # Test 5: Gradient flow — UP-GRPO produces valid gradients
    # ============================================================
    print("\n" + "="*60)
    print("Test 5: Gradient flow validation")
    print("="*60)

    torch.manual_seed(42)
    # Create a leaf tensor for gradient flow test
    params = torch.randn(4, 16, device=device)
    new_lp5 = params + torch.randn(4, 16, device=device) * 0.1 - 1.0
    new_lp5.requires_grad_(True)  # Make it a proper leaf tensor
    old_lp5 = torch.randn(4, 16, device=device) - 1.0
    adv5 = torch.randn(4, 16, device=device)
    mask5 = torch.ones(4, 16, device=device)

    up5, _ = compute_up_grpo_loss(old_lp5, new_lp5, adv5, mask5)
    up5.backward()

    has_grad = new_lp5.grad is not None and new_lp5.grad.numel() > 0
    grad_nonzero = (new_lp5.grad != 0).any().item() if has_grad else False
    grad_finite = torch.isfinite(new_lp5.grad).all().item() if has_grad else False
    grad_norm = new_lp5.grad.norm().item() if has_grad else 0.0

    print(f"  Gradient exists:  {has_grad}")
    print(f"  Gradient nonzero: {grad_nonzero}")
    print(f"  Gradient finite:  {grad_finite}")
    print(f"  Grad norm:        {grad_norm:.6f}")

    test5_pass = has_grad and grad_nonzero and grad_finite
    print(f"  Test 5 PASS: {test5_pass} (valid gradients flow through UP-GRPO)")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GPU: {gpu_name} (sm_{compute_cap[0]}{compute_cap[1]})")
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
    results = {
        "Test 1 (A>0: UP<=Vanilla)": test1_pass,
        "Test 2 (A<0: UP≈Vanilla)": test2_pass,
        "Test 3 (Mixed GRPO adv)": test3_pass,
        "Test 4 (Large ratio unbounded)": test4_pass,
        "Test 5 (Gradient flow)": test5_pass,
    }
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  Total: {passed}/{total} PASSED")

    if all_pass:
        print("\n★★★★★★★★ ALL TESTS PASSED! UP-GRPO implementation verified on H20-3e.")
    else:
        print("\n  Some tests FAILED — investigate before creating PR.")

    return all_pass

if __name__ == "__main__":
    test_up_grpo_vs_vanilla()
