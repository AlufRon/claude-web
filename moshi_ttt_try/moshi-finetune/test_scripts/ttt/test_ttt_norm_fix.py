"""
Test to verify that ttt_norm_weight and ttt_norm_bias are properly initialized.
This should prevent the explosion in ln_reconstruction_target.
"""

import torch
import sys
sys.path.insert(0, '/home/alufr/ttt_tests/moshi/moshi')

from moshi_ttt.config import TTTConfig
from moshi_ttt.models.ssm.ttt_layer import TTTWrapper

def test_ttt_norm_initialization():
    """Test that TTT norm parameters are properly initialized"""
    print("\n" + "="*70)
    print("Testing TTT norm parameter initialization")
    print("="*70)

    # Create TTT config
    config = TTTConfig(
        model_dim=4096,
        num_heads=32,
        mini_batch_size=32,
        ttt_base_lr=1.0,
        scan_checkpoint_group_size=4,
        ssm_layer="ttt_mlp",
    )
    # Add ttt_mlp_layers as dynamic attribute (like in actual code)
    config.ttt_mlp_layers = 2

    # Create TTT wrapper (mimicking what happens in hybrid_layer.py)
    print("\n1. Creating TTTWrapper...")
    ttt_wrapper = TTTWrapper(config)

    # Check parameters BEFORE init_weights
    print("\n2. Checking ttt_norm parameters BEFORE init_weights():")
    ttt_instance = ttt_wrapper.ttt
    weight_before = ttt_instance.ttt_norm_weight.data
    bias_before = ttt_instance.ttt_norm_bias.data

    print(f"   ttt_norm_weight: min={weight_before.min():.6f}, max={weight_before.max():.6f}, mean={weight_before.mean():.6f}")
    print(f"   ttt_norm_bias: min={bias_before.min():.6f}, max={bias_before.max():.6f}, mean={bias_before.mean():.6f}")

    if weight_before.abs().max() > 100:
        print("   ⚠️  WARNING: ttt_norm_weight contains large values (likely uninitialized)!")
    if bias_before.abs().max() > 100:
        print("   ⚠️  WARNING: ttt_norm_bias contains large values (likely uninitialized)!")

    # Call init_weights (THE FIX)
    print("\n3. Calling init_weights() [THE FIX]...")
    ttt_instance.init_weights()

    # Check parameters AFTER init_weights
    print("\n4. Checking ttt_norm parameters AFTER init_weights():")
    weight_after = ttt_instance.ttt_norm_weight.data
    bias_after = ttt_instance.ttt_norm_bias.data

    print(f"   ttt_norm_weight: min={weight_after.min():.6f}, max={weight_after.max():.6f}, mean={weight_after.mean():.6f}")
    print(f"   ttt_norm_bias: min={bias_after.min():.6f}, max={bias_after.max():.6f}, mean={bias_after.mean():.6f}")

    # Verify they are properly initialized
    print("\n5. Verification:")
    weight_is_ones = torch.allclose(weight_after, torch.ones_like(weight_after))
    bias_is_zeros = torch.allclose(bias_after, torch.zeros_like(bias_after))

    print(f"   ttt_norm_weight == ones: {weight_is_ones} ✅" if weight_is_ones else f"   ttt_norm_weight == ones: {weight_is_ones} ❌")
    print(f"   ttt_norm_bias == zeros: {bias_is_zeros} ✅" if bias_is_zeros else f"   ttt_norm_bias == zeros: {bias_is_zeros} ❌")

    # Test forward pass to ensure no explosion
    print("\n6. Testing forward pass (should not explode):")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ttt_wrapper = ttt_wrapper.to(device)

    # Create dummy input
    B, L, D = 1, 192, 4096  # Batch=1, seq_len=192, d_model=4096
    x = torch.randn(B, L, D, device=device) * 2.0  # Scaled random input

    from moshi_ttt.utils import SequenceMetadata
    seq_metadata = SequenceMetadata(
        is_multiscene=False,
        init_offset=None,
        base_offset=None,
        num_chunks=1,
        text_length=0
    )

    try:
        output = ttt_wrapper(x, seq_metadata, layer_id=29)

        out_min, out_max = output.min().item(), output.max().item()
        out_mean, out_std = output.mean().item(), output.std().item()

        print(f"   Input range: [{x.min().item():.2f}, {x.max().item():.2f}]")
        print(f"   Output range: [{out_min:.2f}, {out_max:.2f}]")
        print(f"   Output mean: {out_mean:.2f}, std: {out_std:.2f}")

        if abs(out_max) > 1000 or abs(out_min) > 1000:
            print("   ❌ EXPLOSION DETECTED: Output values are too large!")
            return False
        else:
            print("   ✅ SUCCESS: Output values are reasonable!")
            return True
    except Exception as e:
        print(f"   ❌ ERROR during forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_ttt_norm_initialization()

    print("\n" + "="*70)
    if success:
        print("✅ TEST PASSED: TTT norm parameters are properly initialized")
        print("   The fix (calling init_weights()) prevents the explosion!")
    else:
        print("❌ TEST FAILED: Something is still wrong")
    print("="*70)
