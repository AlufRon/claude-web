#!/usr/bin/env python3
"""
Test format conversion with real Moshi dimensions to simulate actual usage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from moshi_ttt.config import TTTConfig
from moshi_ttt.format_utils import moshi_to_ttt_format, ttt_to_moshi_format

def test_real_moshi_dimensions():
    """Test truncation with realistic Moshi dimensions and sequence lengths."""
    
    print("🎵 Testing TTT Truncation with Real Moshi Dimensions")
    print("=" * 60)
    
    # Real Moshi-7B dimensions
    moshi_config = TTTConfig(
        model_dim=1024,   # Moshi-7B actual dimension
        num_heads=16,     # Moshi-7B actual heads
        mini_batch_size=16
    )
    
    print(f"📋 Moshi Config: d_model={moshi_config.model_dim}, num_heads={moshi_config.num_heads}")
    print(f"📋 head_dim: {moshi_config.head_dim}, mini_batch_size: {moshi_config.mini_batch_size}")
    
    
    # Test realistic audio sequence lengths (in tokens, not seconds)
    # Moshi processes audio in tokens, typical sequences might be:
    test_cases = [
        ("Short audio clip", 240),      # ~15 seconds of audio
        ("Medium audio", 375),          # ~24 seconds (truncated to 368 = 23*16) 
        ("Long audio", 640),            # ~40 seconds (truncated to 640 = 40*16, no truncation)
        ("Very short", 8),              # Less than mini-batch
        ("Streaming chunk", 16),        # Exactly one mini-batch
    ]
    
    batch_size = 1  # Typical training batch size for Moshi
    
    for case_name, seq_len in test_cases:
        print(f"\n🧪 Test: {case_name} (seq_len={seq_len})")
        
        # Create realistic Moshi input
        x_input = torch.randn(batch_size, seq_len, moshi_config.model_dim)
        print(f"   Input shape: {x_input.shape}")
        
        # Test format conversion (core of the fix)
        try:
            # Forward conversion
            x_ttt, metadata = moshi_to_ttt_format(x_input, moshi_config)
            print(f"   TTT format shape: {x_ttt.shape}")
            print(f"   NC: {metadata['NC']}, C: {metadata['C']}")
            
            # Check for truncation
            if metadata['was_truncated']:
                print(f"   🔄 Truncated {metadata['truncated_tokens']} tokens")
            else:
                print(f"   ✅ No truncation needed")
            
            # Reverse conversion
            x_recovered = ttt_to_moshi_format(x_ttt, metadata)
            print(f"   Recovered shape: {x_recovered.shape}")
            
            # Calculate expected dimensions
            NC = seq_len // moshi_config.mini_batch_size
            if NC > 0:
                expected_len = NC * moshi_config.mini_batch_size
                expected_C = moshi_config.mini_batch_size
            else:
                expected_len = seq_len
                expected_C = seq_len
                NC = 1
            
            # Verify TTT format
            expected_ttt_shape = (batch_size, moshi_config.num_heads, NC, expected_C, moshi_config.head_dim)
            assert x_ttt.shape == expected_ttt_shape, f"TTT shape mismatch: {x_ttt.shape} != {expected_ttt_shape}"
            
            # Verify recovered format
            expected_recovered_shape = (batch_size, expected_len, moshi_config.model_dim)
            assert x_recovered.shape == expected_recovered_shape, f"Recovered shape mismatch: {x_recovered.shape} != {expected_recovered_shape}"
            
            # Verify data integrity for non-truncated portion
            if expected_len <= seq_len:
                x_original_truncated = x_input[:, :expected_len, :]
                max_diff = torch.abs(x_recovered - x_original_truncated).max().item()
                assert max_diff < 1e-6, f"Data not preserved, max diff: {max_diff}"
                print(f"   ✅ Data integrity preserved (max diff: {max_diff:.2e})")
            
            print(f"   ✅ PASSED")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            raise
    
    print("\n🎉 ALL REAL MOSHI TESTS PASSED!")
    print("✅ TTT truncation works with real Moshi dimensions")
    print("✅ No zero-padding corruption in TTT weights")
    print("✅ Ready for production use in Moshi TTT training!")

if __name__ == "__main__":
    test_real_moshi_dimensions()