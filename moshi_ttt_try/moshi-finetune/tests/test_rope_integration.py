"""
Test RoPE Integration

Tests the full TTT layer with RoPE enabled to ensure end-to-end functionality.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moshi_ttt.config import TTTConfig
from moshi_ttt.ttt_layer import TTTMLP
from moshi_ttt.format_utils import SequenceMetadata, create_sequence_metadata


def test_ttt_with_rope_enabled():
    """Test TTT forward pass with RoPE enabled"""
    config = TTTConfig(
        model_dim=512,
        num_heads=8,
        mini_batch_size=16,
        use_rope=True,  # Enable RoPE
        rope_theta=10000.0,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    # Verify RoPE was initialized
    assert ttt.rotary_emb is not None, "rotary_emb should be initialized when RoPE enabled"

    B, L = 2, 128
    hidden_states = torch.randn(B, L, config.model_dim)

    # Create metadata with position_ids
    seq_metadata = create_sequence_metadata(hidden_states, config)
    assert seq_metadata.position_ids is not None, "position_ids should be generated when RoPE enabled"

    # Forward pass
    with torch.no_grad():
        output = ttt(hidden_states, seq_metadata)

    assert output.shape == (B, L, config.model_dim)
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"

    print("✅ TTT with RoPE enabled test passed")


def test_long_sequence_with_position_modulo():
    """
    CRITICAL TEST: Verify that position modulo prevents issues on long sequences.

    This is the main advantage of RoPE in TTT - it should handle arbitrarily
    long sequences without degradation.
    """
    config = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    # Test with very long sequence (positions up to 5000)
    B = 1
    L = 5000  # Much longer than mini_batch_size!
    hidden_states = torch.randn(B, L, config.model_dim)

    # Generate position_ids that go beyond mini_batch_size
    position_ids = torch.arange(L, device=hidden_states.device, dtype=torch.long).unsqueeze(0)

    seq_metadata = SequenceMetadata(
        seq_length=L,
        mini_batch_size=config.mini_batch_size,
        position_ids=position_ids,
    )

    with torch.no_grad():
        output = ttt(hidden_states, seq_metadata)

    # Check that output is reasonable (no NaN, not collapsed)
    assert output.shape == (B, L, config.model_dim)
    assert not torch.isnan(output).any(), "Output should not contain NaN even on long sequences"
    assert output.std() > 0.01, "Output should not collapse to zero"

    print(f"✅ Long sequence test passed: L={L}, positions 0-{L-1}, output healthy")


def test_continuous_positions_across_chunks():
    """Test that continuous positioning works across chunks of the same file"""
    config = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    B = 1
    chunk_size = 128

    # Simulate 3 consecutive chunks of the same file
    for i in range(3):
        hidden_states = torch.randn(B, chunk_size, config.model_dim)

        # Create metadata with chunk_index for continuous positioning
        seq_metadata = create_sequence_metadata(
            hidden_states,
            config,
            file_id="test_file.wav",
            chunk_index=i
        )

        # Verify position_ids are continuous
        assert seq_metadata.position_ids is not None
        expected_start = i * chunk_size
        expected_end = (i + 1) * chunk_size
        actual_positions = seq_metadata.position_ids[0]
        expected_positions = torch.arange(expected_start, expected_end)

        assert torch.equal(actual_positions, expected_positions), \
            f"Chunk {i}: positions should be [{expected_start}, {expected_end})"

        # Forward pass should work
        with torch.no_grad():
            output = ttt(hidden_states, seq_metadata)

        assert output.shape == (B, chunk_size, config.model_dim)

        print(f"✅ Chunk {i}: positions {expected_start}-{expected_end-1} processed correctly")

    print("✅ Continuous positions across chunks test passed")


def test_rope_with_different_mini_batch_sizes():
    """Test that RoPE works correctly with different mini_batch_size values"""
    for mini_batch_size in [8, 16, 32, 64]:
        config = TTTConfig(
            model_dim=256,
            num_heads=4,
            mini_batch_size=mini_batch_size,
            use_rope=True,
        )

        ttt = TTTMLP(config, layer_id=0)
        ttt.eval()

        # Verify RoPE max_position_embeddings matches mini_batch_size
        assert ttt.rotary_emb.max_position_embeddings == mini_batch_size, \
            f"RoPE max_pos should be {mini_batch_size}, got {ttt.rotary_emb.max_position_embeddings}"

        B, L = 2, mini_batch_size * 4  # 4 mini-batches
        hidden_states = torch.randn(B, L, config.model_dim)
        seq_metadata = create_sequence_metadata(hidden_states, config)

        with torch.no_grad():
            output = ttt(hidden_states, seq_metadata)

        assert output.shape == (B, L, config.model_dim)

        print(f"✅ RoPE with mini_batch_size={mini_batch_size} test passed")


def test_rope_gradient_flow_in_ttt():
    """Test that gradients flow correctly through TTT with RoPE"""
    config = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.train()

    B, L = 2, 64
    hidden_states = torch.randn(B, L, config.model_dim, requires_grad=True)
    seq_metadata = create_sequence_metadata(hidden_states, config)

    # Forward pass
    output = ttt(hidden_states, seq_metadata)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert hidden_states.grad is not None, "Gradients should flow to input"
    assert not torch.isnan(hidden_states.grad).any(), "Gradients should not be NaN"
    assert not torch.isinf(hidden_states.grad).any(), "Gradients should not be Inf"

    # Check that TTT parameters got gradients
    for name, param in ttt.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print("✅ RoPE gradient flow in TTT test passed")


def test_checkpoint_loading_with_rope():
    """Test that checkpoints with RoPE weights save and load correctly"""
    config = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )

    # Create model with RoPE
    ttt = TTTMLP(config, layer_id=0)
    state_dict = ttt.state_dict()

    # Verify RoPE keys exist in state dict
    rope_keys = [k for k in state_dict.keys() if 'rotary_emb' in k]
    assert len(rope_keys) > 0, f"State dict should contain RoPE keys"

    # Expected RoPE keys (buffers, not parameters)
    expected_rope_keys = ['rotary_emb.inv_freq', 'rotary_emb.cos_cached', 'rotary_emb.sin_cached']
    for key in expected_rope_keys:
        assert key in state_dict, f"Expected RoPE key '{key}' not found in state dict"

    # Create new model and load state dict
    ttt_new = TTTMLP(config, layer_id=0)
    ttt_new.load_state_dict(state_dict, strict=True)

    # Verify loaded RoPE matches original
    assert torch.allclose(ttt.rotary_emb.inv_freq, ttt_new.rotary_emb.inv_freq)
    assert torch.allclose(ttt.rotary_emb.cos_cached, ttt_new.rotary_emb.cos_cached)
    assert torch.allclose(ttt.rotary_emb.sin_cached, ttt_new.rotary_emb.sin_cached)

    print("✅ Checkpoint loading with RoPE test passed")


def test_checkpoint_compatibility_mixed():
    """
    Test checkpoint compatibility scenarios:
    1. Old checkpoint (no RoPE) + new model with RoPE disabled -> works
    2. Old checkpoint (no RoPE) + new model with RoPE enabled -> works (RoPE init randomly)
    3. New checkpoint (with RoPE) + new model with RoPE enabled -> works
    """
    # Scenario 1: Old checkpoint (no RoPE) + model with RoPE disabled
    config_no_rope = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=False,
    )
    ttt_old = TTTMLP(config_no_rope, layer_id=0)
    old_state_dict = ttt_old.state_dict()

    ttt_new_no_rope = TTTMLP(config_no_rope, layer_id=0)
    ttt_new_no_rope.load_state_dict(old_state_dict, strict=True)  # Should work perfectly
    print("✅ Scenario 1: Old checkpoint + RoPE disabled -> works")

    # Scenario 2: Old checkpoint (no RoPE) + model with RoPE enabled
    config_with_rope = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )
    ttt_new_with_rope = TTTMLP(config_with_rope, layer_id=0)
    # Load with strict=False to allow missing RoPE keys
    missing_keys, unexpected_keys = ttt_new_with_rope.load_state_dict(old_state_dict, strict=False)

    # RoPE keys should be in missing_keys
    assert any('rotary_emb' in k for k in missing_keys), "RoPE keys should be missing"
    print(f"✅ Scenario 2: Old checkpoint + RoPE enabled -> works (RoPE initialized randomly)")
    print(f"   Missing keys: {[k for k in missing_keys if 'rotary_emb' in k]}")

    # Scenario 3: New checkpoint (with RoPE) + model with RoPE enabled
    ttt_with_rope_source = TTTMLP(config_with_rope, layer_id=0)
    new_state_dict = ttt_with_rope_source.state_dict()

    ttt_with_rope_target = TTTMLP(config_with_rope, layer_id=0)
    ttt_with_rope_target.load_state_dict(new_state_dict, strict=True)  # Should work perfectly
    print("✅ Scenario 3: New checkpoint with RoPE + RoPE enabled -> works")


def test_rope_with_padding():
    """Test that RoPE works correctly with padded sequences"""
    config = TTTConfig(
        model_dim=256,
        num_heads=4,
        mini_batch_size=16,
        use_rope=True,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    # Sequence length not divisible by mini_batch_size
    B = 2
    L = 63  # Will be padded to 64 (4 * 16)
    hidden_states = torch.randn(B, L, config.model_dim)

    seq_metadata = create_sequence_metadata(hidden_states, config)

    with torch.no_grad():
        output = ttt(hidden_states, seq_metadata)

    # Output should match original length (not padded length)
    assert output.shape == (B, L, config.model_dim), f"Expected shape ({B}, {L}, {config.model_dim}), got {output.shape}"

    print("✅ RoPE with padding test passed")


if __name__ == "__main__":
    print("Running RoPE Integration Tests...\n")

    test_ttt_with_rope_enabled()
    test_long_sequence_with_position_modulo()
    test_continuous_positions_across_chunks()
    test_rope_with_different_mini_batch_sizes()
    test_rope_gradient_flow_in_ttt()
    test_checkpoint_loading_with_rope()
    test_checkpoint_compatibility_mixed()
    test_rope_with_padding()

    print("\n" + "="*60)
    print("✅ All RoPE integration tests passed!")
    print("="*60)
