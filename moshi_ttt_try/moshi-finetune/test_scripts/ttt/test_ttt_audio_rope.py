#!/usr/bin/env python3
"""
Test TTT layer with 1D Audio RoPE integration.
Verifies that the TTT layer can initialize and process audio sequences correctly.
"""

import torch
import sys
import os

# Add the moshi_ttt directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from moshi_ttt.models.ssm.ttt_layer import TTTWrapper
from moshi_ttt.config import TTTConfig
from moshi_ttt.utils import SequenceMetadata

def test_ttt_layer_initialization():
    """Test TTT layer can be initialized with reasonable config."""
    print("Testing TTT layer initialization...")
    
    # Create a config with realistic dimensions (compatible with Triton kernels)
    config = TTTConfig(
        model_dim=512,  # Larger dimension to meet Triton requirements
        num_heads=8,    # 512/8 = 64 head_dim, meets minimum requirements
        rope_theta=10000.0,
        mini_batch_size=8,
        ttt_base_lr=0.1,
        ssm_layer="ttt_linear",  # Use linear version for simplicity
        scan_checkpoint_group_size=0,
    )
    
    try:
        ttt_layer = TTTWrapper(config)
        print("✅ TTT layer initialized successfully")
        return ttt_layer, config
    except Exception as e:
        print(f"❌ TTT layer initialization failed: {e}")
        return None, None

def test_audio_rope_computation():
    """Test that 1D Audio RoPE frequencies are computed correctly."""
    print("\nTesting 1D Audio RoPE computation...")
    
    ttt_layer, config = test_ttt_layer_initialization()
    if ttt_layer is None:
        return False
    
    # Test different sequence lengths
    test_seq_lens = [32, 64, 128, 256]
    
    for seq_len in test_seq_lens:
        try:
            freqs_cis = ttt_layer._precompute_audio_rope_1d(seq_len)
            
            # Check shape: should be [seq_len, head_dim//2]
            expected_shape = (seq_len, config.model_dim // config.num_heads // 2)
            assert freqs_cis.shape == expected_shape, f"Shape mismatch: got {freqs_cis.shape}, expected {expected_shape}"
            
            # Check that frequencies are complex numbers
            assert freqs_cis.dtype == torch.complex64, f"Expected complex64, got {freqs_cis.dtype}"
            
            # Check that all values are finite
            assert torch.all(torch.isfinite(freqs_cis)), "RoPE frequencies contain non-finite values"
            
            print(f"✅ RoPE computation for seq_len={seq_len}: shape={freqs_cis.shape}")
            
        except Exception as e:
            print(f"❌ RoPE computation failed for seq_len={seq_len}: {e}")
            return False
    
    return True

def test_forward_pass():
    """Test TTT layer forward pass with audio sequence."""
    print("\nTesting TTT layer forward pass...")
    
    ttt_layer, config = test_ttt_layer_initialization()
    if ttt_layer is None:
        return False
    
    # Create synthetic audio sequence
    batch_size = 2
    seq_len = 64  # Must be multiple of mini_batch_size (64 = 8 * 8)
    
    # Input: [batch_size, seq_len, model_dim]
    hidden_states = torch.randn(batch_size, seq_len, config.model_dim)
    
    # Create minimal sequence metadata for audio (no video-specific fields)
    seq_metadata = SequenceMetadata(
        init_offset=None,
        base_offset=None,
        text_length=0,
        num_chunks=1,
        seq_text_length=0,  # No text for pure audio
        is_multiscene=False,  # No video scenes
    )
    
    try:
        # Test forward pass
        result = ttt_layer.forward(hidden_states, seq_metadata)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output type: {type(result)}")
        
        # Check if output is reasonable
        if hasattr(result, 'shape'):
            print(f"   Output shape: {result.shape}")
        elif isinstance(result, dict):
            print(f"   Output keys: {list(result.keys())}")
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"   {key} shape: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing TTT Layer with 1D Audio RoPE")
    print("=" * 50)
    
    # Test 1: Initialization
    init_success = test_ttt_layer_initialization()[0] is not None
    
    # Test 2: RoPE computation
    rope_success = test_audio_rope_computation() if init_success else False
    
    # Test 3: Forward pass
    forward_success = test_forward_pass() if init_success else False
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"   Initialization: {'✅ PASS' if init_success else '❌ FAIL'}")
    print(f"   RoPE Computation: {'✅ PASS' if rope_success else '❌ FAIL'}")
    print(f"   Forward Pass: {'✅ PASS' if forward_success else '❌ FAIL'}")
    
    all_success = init_success and rope_success and forward_success
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)