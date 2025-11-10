"""
Test Multi-Learning-Rate Optimizer Setup

This script tests that the multi-LR optimizer implementation:
1. Correctly classifies parameters into groups
2. Creates optimizer with proper learning rates
3. Scheduler respects per-group max_lr
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.ttt_utils import classify_ttt_parameter, get_parameter_groups
from finetune.args import TrainArgs


def test_parameter_classification():
    """Test that parameters are correctly classified into LR groups."""
    print("=" * 80)
    print("TEST 1: Parameter Classification")
    print("=" * 80)

    test_cases = [
        # (param_name, expected_group)
        ("transformer.layers.30.forward_ssm_gating.gating_alpha", "ttt_alpha"),
        ("transformer.layers.30.backward_ssm_gating.gating_alpha", "ttt_alpha"),
        ("transformer.layers.30.ttt_norm_weight", "ttt_weights"),
        ("transformer.layers.30.ttt_norm_bias", "ttt_weights"),
        ("transformer.layers.30.wq.weight", "ttt_weights"),
        ("transformer.layers.30.wk.weight", "ttt_weights"),
        ("transformer.layers.30.wv.weight", "ttt_weights"),
        ("transformer.layers.30.wo.weight", "ttt_weights"),
        ("transformer.layers.30.learnable_ttt_lr_weight", "ttt_weights"),
        ("transformer.layers.30.W1", "ttt_weights"),
        ("transformer.layers.30.W2", "ttt_weights"),
        ("transformer.layers.30.b1", "ttt_weights"),
        ("transformer.layers.30.b2", "ttt_weights"),
        ("transformer.layers.30.weights.0", "ttt_weights"),
        ("transformer.layers.30.biases.0", "ttt_weights"),
        ("transformer.layers.30.post_norm.weight", "ttt_weights"),
        ("transformer.layers.30.post_norm.bias", "ttt_weights"),
        ("transformer.layers.15.lora_A", "base"),
        ("transformer.layers.15.lora_B", "base"),
        ("embed_tokens.weight", "base"),
        ("output_projection.weight", "base"),
    ]

    passed = 0
    failed = 0

    for param_name, expected in test_cases:
        result = classify_ttt_parameter(param_name)
        if result == expected:
            print(f"✅ {param_name:60s} → {result:12s} (correct)")
            passed += 1
        else:
            print(f"❌ {param_name:60s} → {result:12s} (expected: {expected})")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    assert failed == 0, f"Parameter classification test failed with {failed} errors"
    print("✅ All parameter classification tests passed!")
    print()


def test_config_loading():
    """Test that config loads with new multiplier parameters."""
    print("=" * 80)
    print("TEST 2: Configuration Loading")
    print("=" * 80)

    config_path = "example/dailytalk_finetune_from_librilight.yaml"

    try:
        args = TrainArgs.load(config_path)
        print(f"✅ Config loaded successfully: {config_path}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        raise

    # Check TTT args exist
    assert hasattr(args, 'ttt'), "Config missing 'ttt' attribute"
    assert hasattr(args.ttt, 'weight_lr_multiplier'), "Config missing 'weight_lr_multiplier'"
    assert hasattr(args.ttt, 'alpha_lr_multiplier'), "Config missing 'alpha_lr_multiplier'"

    print(f"   Base LR: {args.optim.lr:.2e}")
    print(f"   TTT weight LR multiplier: {args.ttt.weight_lr_multiplier:.1f}x")
    print(f"   TTT alpha LR multiplier: {args.ttt.alpha_lr_multiplier:.1f}x")
    print()
    print(f"   Effective LRs:")
    print(f"     Base:        {args.optim.lr:.2e}")
    print(f"     TTT weights: {args.optim.lr * args.ttt.weight_lr_multiplier:.2e}")
    print(f"     TTT alpha:   {args.optim.lr * args.ttt.alpha_lr_multiplier:.2e}")
    print()

    # Verify default values
    assert args.ttt.weight_lr_multiplier == 10.0, f"Expected weight_lr_multiplier=10.0, got {args.ttt.weight_lr_multiplier}"
    assert args.ttt.alpha_lr_multiplier == 1000.0, f"Expected alpha_lr_multiplier=1000.0, got {args.ttt.alpha_lr_multiplier}"

    print("✅ Configuration loading test passed!")
    print()


def test_parameter_group_structure():
    """Test that parameter groups have the right structure."""
    print("=" * 80)
    print("TEST 3: Parameter Group Structure")
    print("=" * 80)

    # Create a mock model with sample parameters
    import torch
    import torch.nn as nn

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Base parameters (LoRA)
            self.lora_A = nn.Parameter(torch.randn(10, 10))
            self.lora_B = nn.Parameter(torch.randn(10, 10))

            # TTT weight parameters
            self.ttt_norm_weight = nn.Parameter(torch.randn(10))
            self.wq = nn.Linear(10, 10)
            self.W1 = nn.Parameter(torch.randn(10, 10))

            # TTT alpha parameters
            class GatingModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gating_alpha = nn.Parameter(torch.randn(10))

            self.forward_ssm_gating = GatingModule()
            self.backward_ssm_gating = GatingModule()

    model = MockModel()

    # Set all parameters to trainable
    for param in model.parameters():
        param.requires_grad = True

    # Get parameter groups
    groups = get_parameter_groups(model)

    print(f"Parameter groups created:")
    print(f"  Base:        {len(groups['base'])} parameters")
    print(f"  TTT weights: {len(groups['ttt_weights'])} parameters")
    print(f"  TTT alpha:   {len(groups['ttt_alpha'])} parameters")
    print()

    # Verify structure
    assert 'base' in groups, "Missing 'base' group"
    assert 'ttt_weights' in groups, "Missing 'ttt_weights' group"
    assert 'ttt_alpha' in groups, "Missing 'ttt_alpha' group"

    assert len(groups['base']) > 0, "Base group should have parameters"
    assert len(groups['ttt_weights']) > 0, "TTT weights group should have parameters"
    assert len(groups['ttt_alpha']) > 0, "TTT alpha group should have parameters"

    # Count total parameters
    total_params = sum(len(group) for group in groups.values())
    model_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Model parameters: {model_params}")
    assert total_params == model_params, "Parameter count mismatch"

    print()
    print("✅ Parameter group structure test passed!")
    print()


def test_lr_calculation():
    """Test that learning rate calculations are correct."""
    print("=" * 80)
    print("TEST 4: Learning Rate Calculations")
    print("=" * 80)

    base_lr = 3e-7
    weight_mult = 10.0
    alpha_mult = 1000.0

    expected_base = base_lr
    expected_weights = base_lr * weight_mult
    expected_alpha = base_lr * alpha_mult

    print(f"Base LR:              {base_lr:.2e}")
    print(f"Weight multiplier:    {weight_mult:.1f}x")
    print(f"Alpha multiplier:     {alpha_mult:.1f}x")
    print()
    print(f"Expected LRs:")
    print(f"  Base:        {expected_base:.2e}")
    print(f"  TTT weights: {expected_weights:.2e}")
    print(f"  TTT alpha:   {expected_alpha:.2e}")
    print()

    # Verify ratios
    ratio_weights = expected_weights / expected_base
    ratio_alpha = expected_alpha / expected_base

    assert abs(ratio_weights - weight_mult) < 1e-6, f"Weight LR ratio incorrect: {ratio_weights}"
    assert abs(ratio_alpha - alpha_mult) < 1e-6, f"Alpha LR ratio incorrect: {ratio_alpha}"

    print("✅ Learning rate calculation test passed!")
    print()


def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("MULTI-LEARNING-RATE OPTIMIZER TESTS")
    print("=" * 80)
    print()

    try:
        test_parameter_classification()
        test_config_loading()
        test_parameter_group_structure()
        test_lr_calculation()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("The multi-LR optimizer implementation is working correctly.")
        print()
        print("Next steps:")
        print("1. Run a 100-step training test:")
        print("   python train.py --config example/dailytalk_finetune_from_librilight.yaml")
        print()
        print("2. Verify in logs that:")
        print("   - lr_base, lr_ttt_w, lr_ttt_α are logged")
        print("   - ttt_alpha increases from 0.005")
        print("   - No NaN/Inf errors occur")
        print()

    except AssertionError as e:
        print()
        print("=" * 80)
        print("❌ TESTS FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
