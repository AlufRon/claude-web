#!/usr/bin/env python3
"""
Test script to verify TTT truncation fix works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from moshi_ttt.config import TTTConfig
from moshi_ttt.format_utils import moshi_to_ttt_format, ttt_to_moshi_format, test_format_conversions

def test_truncation_scenarios():
    """Test various truncation scenarios."""
    
    # Create TTT config
    ttt_config = TTTConfig(
        model_dim=512,
        num_heads=8,
        mini_batch_size=16
    )
    
    print("🧪 Testing TTT Truncation Fix")
    print("=" * 50)
    
    # Test Case 1: Exact divisible sequence (no truncation needed)
    print("\n📋 Test Case 1: Exact divisible sequence (seq_len=32, mini_batch=16)")
    seq_len = 32  # 32 = 2 * 16, exactly divisible
    x1 = torch.randn(2, seq_len, 512)
    x1_ttt, meta1 = moshi_to_ttt_format(x1, ttt_config)
    x1_recovered = ttt_to_moshi_format(x1_ttt, meta1)
    print(f"   Original shape: {x1.shape}")
    print(f"   TTT shape: {x1_ttt.shape}")
    print(f"   Recovered shape: {x1_recovered.shape}")
    print(f"   Was truncated: {meta1['was_truncated']}")
    print(f"   Truncated tokens: {meta1['truncated_tokens']}")
    assert not meta1['was_truncated'], "Should not be truncated"
    assert x1_recovered.shape == x1.shape, "Should recover original shape"
    print("   ✅ PASSED")
    
    # Test Case 2: Truncation needed (seq_len > mini_batch_size but not divisible)
    print("\n📋 Test Case 2: Truncation needed (seq_len=25, mini_batch=16)")
    seq_len = 25  # 25 // 16 = 1, truncated_len = 16, 9 tokens truncated
    x2 = torch.randn(2, seq_len, 512)
    x2_ttt, meta2 = moshi_to_ttt_format(x2, ttt_config)
    x2_recovered = ttt_to_moshi_format(x2_ttt, meta2)
    print(f"   Original shape: {x2.shape}")
    print(f"   TTT shape: {x2_ttt.shape}")
    print(f"   Recovered shape: {x2_recovered.shape}")
    print(f"   Was truncated: {meta2['was_truncated']}")
    print(f"   Truncated tokens: {meta2['truncated_tokens']}")
    assert meta2['was_truncated'], "Should be truncated"
    assert meta2['truncated_tokens'] == 9, f"Should truncate 9 tokens, got {meta2['truncated_tokens']}"
    assert x2_recovered.shape == (2, 16, 512), f"Should recover truncated shape, got {x2_recovered.shape}"
    # Verify non-truncated data is preserved
    x2_original_truncated = x2[:, :16, :]
    max_diff = torch.abs(x2_recovered - x2_original_truncated).max().item()
    assert max_diff < 1e-6, f"Non-truncated data should be preserved, max diff: {max_diff}"
    print("   ✅ PASSED")
    
    # Test Case 3: Very short sequence (shorter than mini_batch_size)
    print("\n📋 Test Case 3: Short sequence (seq_len=8, mini_batch=16)")
    seq_len = 8  # 8 < 16, NC=1, C=8
    x3 = torch.randn(2, seq_len, 512)
    x3_ttt, meta3 = moshi_to_ttt_format(x3, ttt_config)
    x3_recovered = ttt_to_moshi_format(x3_ttt, meta3)
    print(f"   Original shape: {x3.shape}")
    print(f"   TTT shape: {x3_ttt.shape}")
    print(f"   Recovered shape: {x3_recovered.shape}")
    print(f"   Was truncated: {meta3['was_truncated']}")
    print(f"   Truncated tokens: {meta3['truncated_tokens']}")
    print(f"   NC: {meta3['NC']}, C: {meta3['C']}")
    assert not meta3['was_truncated'], "Should not be truncated"
    assert meta3['NC'] == 1, f"Should have NC=1, got {meta3['NC']}"
    assert meta3['C'] == seq_len, f"Should have C={seq_len}, got {meta3['C']}"
    assert x3_recovered.shape == x3.shape, "Should recover original shape"
    print("   ✅ PASSED")
    
    # Test Case 4: Built-in test function
    print("\n📋 Test Case 4: Running built-in format conversion tests")
    try:
        # Test with various sequence lengths
        test_lengths = [63, 32, 17, 8, 1]
        for seq_len in test_lengths:
            print(f"   Testing seq_len={seq_len}...")
            result = test_format_conversions(ttt_config, batch_size=2, seq_len=seq_len)
            assert result, f"Built-in test failed for seq_len={seq_len}"
        print("   ✅ All built-in tests PASSED")
    except Exception as e:
        print(f"   ❌ Built-in test FAILED: {e}")
        raise
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ TTT truncation fix is working correctly")
    
    # Print summary
    print("\n📊 SUMMARY:")
    print("- Exact divisible sequences: No truncation, perfect recovery")
    print("- Non-divisible sequences: Truncation applied, non-truncated data preserved")  
    print("- Short sequences: Handled with dynamic mini-batch size")
    print("- No more zero-padding corruption in TTT weights!")

if __name__ == "__main__":
    test_truncation_scenarios()