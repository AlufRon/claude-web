#!/usr/bin/env python3
"""
Step 3.2.2 & 3.2.3 Tests: Format conversion and metadata functionality
Test format conversions and sequence metadata creation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add original Moshi to path
sys.path.append('/home/alufr/ttt_tests/moshi/moshi')

from moshi_ttt.config import TTTConfig
from moshi_ttt.format_utils import (
    moshi_to_ttt_format, 
    ttt_to_moshi_format, 
    test_format_conversions
)
from moshi_ttt.moshi_metadata import MoshiSequenceMetadata
import torch


def test_format_conversion():
    """Test Step 3.2.2: Format conversion utilities"""
    print("🧪 Testing Step 3.2.2: Format conversion utilities...")
    
    ttt_config = TTTConfig()
    
    try:
        # Test various sequence lengths
        test_cases = [16, 32, 63, 128]
        
        for seq_len in test_cases:
            print(f"  Testing format conversion for seq_len={seq_len}")
            
            # Use the built-in test function
            success = test_format_conversions(ttt_config, batch_size=2, seq_len=seq_len)
            assert success, f"Format conversion test failed for seq_len={seq_len}"
            
        print("✅ Step 3.2.2 SUCCESS: Format conversion utilities work!")
        return True
        
    except Exception as e:
        print(f"❌ Step 3.2.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moshi_metadata():
    """Test Step 3.2.3: Moshi sequence metadata"""
    print("🧪 Testing Step 3.2.3: Moshi sequence metadata...")
    
    ttt_config = TTTConfig()
    batch_size = 2
    d_model = ttt_config.model_dim
    
    try:
        # Test different sequence lengths
        test_cases = [
            16,  # Exact multiple of mini_batch_size
            32,  # Another multiple
            63,  # Non-multiple (needs padding)
            128  # Larger sequence
        ]
        
        for seq_len in test_cases:
            print(f"  Testing metadata for seq_len={seq_len}")
            
            # Create metadata
            metadata = MoshiSequenceMetadata(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                ttt_config=ttt_config
            )
            
            # Test format info
            format_info = metadata.get_format_info()
            assert format_info['seq_len'] == seq_len
            assert format_info['d_model'] == d_model
            
            # Test tensor validation
            moshi_tensor = torch.randn(batch_size, seq_len, d_model)
            assert metadata.validate_moshi_tensor(moshi_tensor)
            
            # Test TTT tensor validation
            ttt_tensor = torch.randn(*metadata.ttt_shape)
            assert metadata.validate_ttt_tensor(ttt_tensor)
            
            # Test padding mask creation
            padding_mask = metadata.create_padding_mask()
            if metadata.needs_padding:
                assert padding_mask is not None
                assert padding_mask.shape == (batch_size, metadata.padded_len)
                assert padding_mask.dtype == torch.bool
                
                # Test TTT padding mask
                ttt_padding_mask = metadata.create_ttt_padding_mask()
                assert ttt_padding_mask is not None
                expected_shape = (batch_size, metadata.H, metadata.NC, metadata.C)
                assert ttt_padding_mask.shape == expected_shape
            else:
                assert padding_mask is None
                
            # Test streaming info
            streaming_info = metadata.get_streaming_info()
            assert streaming_info['chunk_size'] == metadata.C
            assert streaming_info['num_chunks'] == metadata.NC
            assert streaming_info['supports_streaming'] == True
            
            print(f"    ✅ {metadata}")
            
        print("✅ Step 3.2.3 SUCCESS: Moshi sequence metadata works!")
        return True
        
    except Exception as e:
        print(f"❌ Step 3.2.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_workflow():
    """Test that format conversion and metadata work together"""
    print("🧪 Testing integrated workflow (format conversion + metadata)...")
    
    try:
        ttt_config = TTTConfig()
        batch_size = 2
        seq_len = 63  # Non-multiple to test padding
        d_model = ttt_config.model_dim
        
        # Create metadata
        metadata = MoshiSequenceMetadata(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            ttt_config=ttt_config
        )
        
        # Create test input
        x_moshi = torch.randn(batch_size, seq_len, d_model)
        
        # Convert to TTT format using format_utils
        x_ttt, conversion_metadata = moshi_to_ttt_format(x_moshi, ttt_config)
        
        # Validate TTT tensor using sequence metadata
        assert metadata.validate_ttt_tensor(x_ttt)
        
        # Check that conversion metadata matches sequence metadata
        assert conversion_metadata['NC'] == metadata.NC
        assert conversion_metadata['C'] == metadata.C
        assert conversion_metadata['H'] == metadata.H
        assert conversion_metadata['HD'] == metadata.HD
        
        # Convert back to Moshi format
        x_recovered = ttt_to_moshi_format(x_ttt, conversion_metadata)
        
        # Validate recovered tensor
        assert metadata.validate_moshi_tensor(x_recovered)
        
        # Check data preservation
        max_diff = torch.abs(x_recovered - x_moshi).max().item()
        assert max_diff < 1e-6, f"Data not preserved: max diff = {max_diff}"
        
        print("✅ Integrated workflow SUCCESS: Format conversion and metadata work together!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated workflow FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Running Step 3.2.2 & 3.2.3 Tests")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_format_conversion()
    print()
    test2_passed = test_moshi_metadata()
    print()
    test3_passed = test_integrated_workflow()
    print()
    
    # Summary
    if test1_passed and test2_passed and test3_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Step 3.2.2 & 3.2.3 COMPLETE: Format conversion and metadata utilities ready!")
        print("Next: Step 3.3 - Implement TTT Processing")
    else:
        print("❌ Some tests failed. Please fix before proceeding.")
        if not test1_passed:
            print("  - Step 3.2.2: Format conversion issues")
        if not test2_passed:
            print("  - Step 3.2.3: Metadata issues")
        if not test3_passed:
            print("  - Integration issues")
