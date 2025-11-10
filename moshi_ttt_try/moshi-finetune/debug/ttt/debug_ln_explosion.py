"""
Debug script to identify the exact cause of the ln_reconstruction_target explosion.

This creates a minimal test case that reproduces the issue.
"""

import torch
import torch.nn as nn

def ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias):
    """Original implementation from ttt_layer.py:275-304"""
    XV = XV - XK
    eps = 1e-8
    # Compute mean and std over the head dimension (last dimension)
    mean = XV.mean(dim=-1, keepdim=True)
    std = XV.std(dim=-1, keepdim=True)

    print(f"  After XV - XK:")
    print(f"    XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"    mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"    std range: [{std.min():.6f}, {std.max():.6f}]")
    print(f"    std min value: {std.min():.12f}")

    # Normalize
    XV = (XV - mean) / (std + eps)

    print(f"  After normalization (before affine):")
    print(f"    XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"    XV mean: {XV.mean():.6f}, std: {XV.std():.6f}")

    # Apply per-head weight and bias.
    print(f"  ttt_norm_weight range: [{ttt_norm_weight.min():.6f}, {ttt_norm_weight.max():.6f}]")
    print(f"  ttt_norm_bias range: [{ttt_norm_bias.min():.6f}, {ttt_norm_bias.max():.6f}]")

    XV = ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV + ttt_norm_bias.unsqueeze(0).unsqueeze(0)

    print(f"  After affine transform (before + XK):")
    print(f"    XV range: [{XV.min():.6f}, {XV.max():.6f}]")

    result = XV + XK

    print(f"  Final result (after + XK):")
    print(f"    result range: [{result.min():.6f}, {result.max():.6f}]")

    return result


def test_scenario_1_normal():
    """Test with normal inputs (should work fine)"""
    print("\n" + "="*70)
    print("SCENARIO 1: Normal inputs (baseline)")
    print("="*70)

    B, L, H, HD = 1, 192, 32, 128

    # Normal distributions
    XV = torch.randn(B, L, H, HD) * 2.0
    XK = torch.randn(B, L, H, HD) * 2.0

    # Properly initialized norm params
    ttt_norm_weight = torch.ones(H, HD)
    ttt_norm_bias = torch.zeros(H, HD)

    print(f"Input XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"Input XK range: [{XK.min():.6f}, {XK.max():.6f}]")

    result = ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias)


def test_scenario_2_low_variance():
    """Test with very similar XV and XK (low variance in XV-XK)"""
    print("\n" + "="*70)
    print("SCENARIO 2: Low variance in (XV - XK)")
    print("="*70)

    B, L, H, HD = 1, 192, 32, 128

    # XV and XK are very similar → low variance in difference
    base = torch.randn(B, L, H, HD) * 2.0
    XV = base + torch.randn(B, L, H, HD) * 0.001  # Very small noise
    XK = base

    # Properly initialized norm params
    ttt_norm_weight = torch.ones(H, HD)
    ttt_norm_bias = torch.zeros(H, HD)

    print(f"Input XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"Input XK range: [{XK.min():.6f}, {XK.max():.6f}]")
    print(f"(XV - XK) std: {(XV - XK).std():.12f}")

    result = ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias)


def test_scenario_3_bad_norm_params():
    """Test with improperly initialized norm parameters"""
    print("\n" + "="*70)
    print("SCENARIO 3: Bad norm parameters")
    print("="*70)

    B, L, H, HD = 1, 192, 32, 128

    # Normal distributions
    XV = torch.randn(B, L, H, HD) * 2.0
    XK = torch.randn(B, L, H, HD) * 2.0

    # BADLY initialized norm params (random instead of ones/zeros)
    ttt_norm_weight = torch.randn(H, HD) * 10.0 + 5.0  # Mean ~5, large values
    ttt_norm_bias = torch.randn(H, HD) * 100.0  # Large bias

    print(f"Input XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"Input XK range: [{XK.min():.6f}, {XK.max():.6f}]")

    result = ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias)


def test_scenario_4_uninitialized():
    """Test with uninitialized (very large random) norm parameters"""
    print("\n" + "="*70)
    print("SCENARIO 4: Uninitialized norm parameters (simulating meta device)")
    print("="*70)

    B, L, H, HD = 1, 192, 32, 128

    # Normal distributions
    XV = torch.randn(B, L, H, HD) * 2.0
    XK = torch.randn(B, L, H, HD) * 2.0

    # Uninitialized (could be huge random values from meta device)
    ttt_norm_weight = torch.empty(H, HD)  # Uninitialized!
    ttt_norm_bias = torch.empty(H, HD)    # Uninitialized!

    # Fill with typical "uninitialized memory" patterns
    ttt_norm_weight.fill_(1e6)  # Simulating garbage values
    ttt_norm_bias.fill_(1e6)

    print(f"Input XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"Input XK range: [{XK.min():.6f}, {XK.max():.6f}]")

    result = ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias)


def test_scenario_5_zero_std():
    """Test with constant XV-XK (std = 0)"""
    print("\n" + "="*70)
    print("SCENARIO 5: Zero std in (XV - XK)")
    print("="*70)

    B, L, H, HD = 1, 192, 32, 128

    # XV and XK are identical → std = 0
    base = torch.randn(B, L, H, HD) * 2.0
    XV = base.clone()
    XK = base.clone()

    # Properly initialized norm params
    ttt_norm_weight = torch.ones(H, HD)
    ttt_norm_bias = torch.zeros(H, HD)

    print(f"Input XV range: [{XV.min():.6f}, {XV.max():.6f}]")
    print(f"Input XK range: [{XK.min():.6f}, {XK.max():.6f}]")
    print(f"(XV - XK) std: {(XV - XK).std():.12f}")

    result = ln_reconstruction_target_original(XV, XK, ttt_norm_weight, ttt_norm_bias)


if __name__ == "__main__":
    print("🔍 Debugging ln_reconstruction_target explosion")
    print("Testing different scenarios to identify the root cause...")

    # Run all test scenarios
    test_scenario_1_normal()
    test_scenario_2_low_variance()
    test_scenario_3_bad_norm_params()
    test_scenario_4_uninitialized()
    test_scenario_5_zero_std()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Compare the scenarios above to identify which matches your log output.")
    print("The scenario that produces values in the millions is the culprit.")
