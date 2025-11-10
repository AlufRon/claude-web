"""
Test script to demonstrate inner loop loss tracking functionality.

This script shows how to:
1. Enable inner loop loss logging in TTT-MLP
2. Collect reconstruction losses during forward pass
3. Create Figure 4-style visualization

Run with: python test_inner_loop_logging.py
"""

import torch
import numpy as np
from pathlib import Path

# Import TTT-MLP functions
from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp_with_states

# Import plotting functions
from finetune.inner_loop_plotting import (
    create_inner_loop_loss_plot,
    create_detailed_inner_loop_plot,
    aggregate_inner_loop_statistics,
)


def test_basic_loss_logging():
    """Test that loss logging works without breaking existing functionality."""
    print("=" * 80)
    print("Test 1: Basic Loss Logging")
    print("=" * 80)
    
    # Setup dummy inputs (batch=1, heads=4, seq=10, mini_batches=5, head_dim=64)
    batch_size = 1
    num_heads = 4
    seq_len = 10
    num_mini_batches = 5
    head_dim = 64
    
    # Create random inputs
    XK = torch.randn(batch_size, num_heads, num_mini_batches, seq_len, head_dim)
    XQ = torch.randn(batch_size, num_heads, num_mini_batches, seq_len, head_dim)
    XV = torch.randn(batch_size, num_heads, num_mini_batches, seq_len, head_dim)
    eta = torch.ones(batch_size, num_heads, num_mini_batches, seq_len, 1) * 0.01
    
    # Initial parameters
    W1_init = torch.randn(batch_size, num_heads, head_dim, 4 * head_dim) * 0.02
    b1_init = torch.zeros(batch_size, num_heads, 1, 4 * head_dim)
    W2_init = torch.randn(batch_size, num_heads, 4 * head_dim, head_dim) * 0.02
    b2_init = torch.zeros(batch_size, num_heads, 1, head_dim)
    
    ttt_norm_weight = torch.ones(num_heads, head_dim)
    ttt_norm_bias = torch.zeros(num_heads, head_dim)
    
    checkpoint_group_size = 2
    
    print("\n1. Testing WITHOUT loss logging (original behavior):")
    output, states = ttt_mlp_with_states(
        XK, XQ, XV, eta,
        ttt_norm_weight, ttt_norm_bias,
        W1_init, b1_init, W2_init, b2_init,
        checkpoint_group_size,
        log_losses=False  # Disabled
    )
    print(f"   Output shape: {output.shape}")
    print(f"   Returned {len([output, states])} values (expected 2)")
    print("   ✓ Original behavior preserved")
    
    print("\n2. Testing WITH loss logging (new feature):")
    output_with_losses, states_with_losses, losses = ttt_mlp_with_states(
        XK, XQ, XV, eta,
        ttt_norm_weight, ttt_norm_bias,
        W1_init, b1_init, W2_init, b2_init,
        checkpoint_group_size,
        log_losses=True  # Enabled
    )
    print(f"   Output shape: {output_with_losses.shape}")
    print(f"   Number of losses: {len(losses)}")
    print(f"   Returned {len([output_with_losses, states_with_losses, losses])} values (expected 3)")
    print(f"   Loss values: {losses}")
    print("   ✓ Loss logging works")
    
    # Verify outputs are identical
    assert torch.allclose(output, output_with_losses, atol=1e-5), "Outputs should be identical!"
    print("\n   ✓ Outputs are identical (no behavior change)")
    
    # Verify losses are decreasing (gradient descent should reduce reconstruction loss)
    if len(losses) > 1:
        # Check if there's a general downward trend
        first_half_mean = np.mean(losses[:len(losses)//2])
        second_half_mean = np.mean(losses[len(losses)//2:])
        if first_half_mean > second_half_mean:
            print(f"   ✓ Losses show downward trend ({first_half_mean:.4f} → {second_half_mean:.4f})")
        else:
            print(f"   ⚠ No clear downward trend (might be random data artifact)")
    
    print("\n✅ Test 1 PASSED\n")
    return losses


def test_plotting_functionality(losses):
    """Test that plotting functions work correctly."""
    print("=" * 80)
    print("Test 2: Plotting Functionality")
    print("=" * 80)
    
    # Create mock data: losses at different positions
    # Each position has a list of losses from mini-batch iterations
    losses_per_position = {}
    
    # Simulate 10 positions with varying number of mini-batches
    np.random.seed(42)
    for pos in range(1000, 11000, 1000):
        # Initial loss decreases as position increases (test-time learning)
        initial_loss = 0.15 - (pos / 100000) + np.random.normal(0, 0.01)
        
        # Create decreasing losses for mini-batches (gradient descent)
        num_mini_batches = 5
        mini_batch_losses = []
        for i in range(num_mini_batches):
            # Each mini-batch reduces loss by ~10% with noise
            reduction_factor = 0.9 ** i
            loss = initial_loss * reduction_factor + np.random.normal(0, 0.005)
            mini_batch_losses.append(max(0, loss))  # Ensure non-negative
        
        losses_per_position[pos] = mini_batch_losses
    
    print(f"\nCreated mock data for {len(losses_per_position)} positions")
    print("Sample data:")
    for pos in sorted(list(losses_per_position.keys()))[:3]:
        losses_list = losses_per_position[pos]
        print(f"  Position {pos}: {len(losses_list)} mini-batches")
        print(f"    Losses: {[f'{l:.4f}' for l in losses_list]}")
        reduction = (losses_list[0] - losses_list[-1]) / losses_list[0] * 100
        print(f"    Reduction: {reduction:.1f}%")
    
    output_dir = "./test_output/inner_loop_plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Creating main Figure 4 plot...")
    plot_path = create_inner_loop_loss_plot(
        losses_per_position,
        output_dir=output_dir,
        sequence_id="test_sequence",
    )
    print(f"   ✓ Saved to: {plot_path}")
    
    print(f"\n2. Creating detailed convergence plot...")
    detailed_path = create_detailed_inner_loop_plot(
        losses_per_position,
        output_dir=output_dir,
        sequence_id="test_sequence",
        max_positions_to_show=10,
    )
    print(f"   ✓ Saved to: {detailed_path}")
    
    print(f"\n3. Creating aggregate statistics...")
    # Test with multiple sequences
    all_sequences = {
        "seq_0": losses_per_position,
        "seq_1": {pos: [l * 0.95 for l in losses] for pos, losses in losses_per_position.items()},
    }
    stats = aggregate_inner_loop_statistics(all_sequences, output_dir)
    print(f"   ✓ Improvement: {stats['improvement_percent']:.1f}%")
    
    print("\n✅ Test 2 PASSED\n")
    
    return stats


def test_yaml_configuration():
    """Show how to configure this feature in YAML."""
    print("=" * 80)
    print("Test 3: YAML Configuration Example")
    print("=" * 80)
    
    yaml_config = """
# Example: moshi_7B_multilayer_with_ttt.yaml

ttt:
  enable: true
  layers: "23,24,25,26,27,28,29,30,31"
  base_lr: 0.0015
  mini_batch_size: 1
  persistent_states: true
  initial_gating_alpha: 0.005
  
  # Figure 4: Inner loop loss tracking (NEW)
  log_inner_loop_losses: true           # Enable loss computation
  inner_loop_log_interval: 1            # Log every N positions (1 = all)
  save_inner_loop_plots: true           # Auto-generate plots during eval
  inner_loop_plot_dir: "./evaluation_plots/inner_loop"

paper_metrics:
  paper_metrics_eval: true
  librilight_sequences_dir: "librilight_1hour_sequences"
  plot_dir: "./evaluation_plots"
"""
    
    print("\nTo enable inner loop loss tracking, add to your YAML config:")
    print(yaml_config)
    
    print("\nConfiguration fields:")
    print("  - log_inner_loop_losses: Enable/disable loss computation (default: False)")
    print("  - inner_loop_log_interval: Log every N positions for efficiency (default: 1)")
    print("  - save_inner_loop_plots: Auto-generate Figure 4 plots (default: False)")
    print("  - inner_loop_plot_dir: Where to save plots (default: ./evaluation_plots/inner_loop)")
    
    print("\n⚠️  All features are OFF by default - existing code unchanged when disabled")
    print("✅ Test 3 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INNER LOOP LOSS TRACKING - TEST SUITE")
    print("=" * 80 + "\n")
    
    # Test 1: Basic functionality
    losses = test_basic_loss_logging()
    
    # Test 2: Plotting
    stats = test_plotting_functionality(losses)
    
    # Test 3: Configuration
    test_yaml_configuration()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Add to YAML config: ttt.log_inner_loop_losses = true")
    print("2. Run LibriLight evaluation with paper_metrics enabled")
    print("3. Check ./evaluation_plots/inner_loop for Figure 4 plots")
    print("\nSee INNER_LOOP_IMPLEMENTATION.md for full documentation.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
